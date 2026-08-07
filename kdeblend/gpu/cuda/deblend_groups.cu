// The kdeblend group-replace fitter, whole-fit-per-block: one
// group per block, the complete sweep loop on device.  This
// driver stages each member's weight/phasors, runs the
// block-cooperative mode-sum reductions, and calls the thread-0
// phase functions; the numerics live in the headers.
#include "common.cuh"
#include "model.cuh"
#include "state.cuh"
#include "reduce.cuh"
#include "update.cuh"
#include "sweep.cuh"
#include "measure.cuh"

extern "C" __global__ void deblend_groups(
    // mode data (CSR by group)
    const KIMT* __restrict__ kim,
    const int* __restrict__ iy,
    const int* __restrict__ ix,
    const MODET* __restrict__ kv,
    const MODET* __restrict__ ku,
    const MODET* __restrict__ ef2,
    const long* __restrict__ moff,   // ngroup+1
    // fixed-model comps (CSR by group): F, So(3), pv, pu
    const double* __restrict__ fcomp,  // nfc x 6
    const long* __restrict__ foff,     // ngroup+1
    // per-group scalars
    const int* __restrict__ nobja,
    const int* __restrict__ dima,
    const double* __restrict__ df2a,
    const double* __restrict__ drowa,
    const double* __restrict__ dcola,
    const double* __restrict__ jaca,   // ngroup x 4 (a00,a01,a10,a11)
    const double* __restrict__ wta,    // weight
    const double* __restrict__ detia,  // detAtinv
    const double* __restrict__ tsma,   // Tsmooth
    const double* __restrict__ tola,   // tol
    const double* __restrict__ ftola,
    const double* __restrict__ ctola,
    const int* __restrict__ maxitera,
    const int* __restrict__ recentera,
    const double* __restrict__ censig0a,
    // state block pointers (flat arrays indexed by group offsets)
    int* otype_g, double* F_g, double* fscale_g,
    double* cov_g, double* sw_g, double* pos_g, double* dpos_g,
    double* cpull_g, double* censig_g,
    int* nfail_g, int* nrestart_g, int* dbflags_g, int* fixcen_g,
    const long* __restrict__ ooff,   // ngroup+1 object offsets
    double* hist_g, double* scales_g, double* saved_g,
    double* xtmp_g,
    const long* __restrict__ soff,   // ngroup+1 scratch offsets
    // final-state measurement outputs per object
    double* objsums_g,               // nobj_tot x 6
    double* objcov_g,                // nobj_tot x 13
    // outputs per group
    int* numiter_g, int* converged_g, int* nskip_g, int* err_g)
{
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    Params P;
    P.nobj = nobja[gid];
    P.dim = dima[gid];
    P.df2 = df2a[gid];
    P.drow = drowa[gid];
    P.dcol = dcola[gid];
    P.a00 = jaca[gid * 4]; P.a01 = jaca[gid * 4 + 1];
    P.a10 = jaca[gid * 4 + 2]; P.a11 = jaca[gid * 4 + 3];
    P.weight = wta[gid];
    P.detatinv = detia[gid];
    P.fac = P.weight * P.detatinv;
    P.Tsmooth = tsma[gid];
    P.smoothcov = P.Tsmooth / 2.0;
    P.tol = tola[gid];
    P.flux_tol = ftola[gid];
    P.cen_tol = ctola[gid];
    P.maxiter = maxitera[gid];
    P.recenter = recentera[gid];
    P.cen_sigma0 = censig0a[gid];

    P.m0 = moff[gid];
    P.nm = moff[gid + 1] - P.m0;
    P.f0 = foff[gid];
    P.nfc = (int)(foff[gid + 1] - P.f0);
    P.o0 = ooff[gid];
    const long s0 = soff[gid];
    P.smax = (int)(soff[gid + 1] - s0);

    GState g;
    g.otype = otype_g + P.o0;
    g.F = F_g + P.o0;
    g.fscale = fscale_g + P.o0;
    g.cov = cov_g + P.o0 * 3;
    g.sw = sw_g + P.o0 * 3;
    g.pos = pos_g + P.o0 * 2;
    g.dpos = dpos_g + P.o0 * 2;
    g.cen_pull = cpull_g + P.o0 * 2;
    g.censig = censig_g + P.o0;
    g.nfail = nfail_g + P.o0;
    g.nrestart = nrestart_g + P.o0;
    g.dbflags = dbflags_g + P.o0;
    g.fixcen = fixcen_g + P.o0;
    g.hist = hist_g + 3 * s0;
    g.scales = scales_g + s0;
    g.saved = saved_g + s0;
    g.xtmp = xtmp_g + s0;

    __shared__ Shm sh;

    // thread-0 sweep-level state
    SweepState ss;
    ss.hprev[0] = -1.0; ss.hprev[1] = -1.0; ss.hprev[2] = -1.0;
    ss.hlen[0] = 0; ss.hlen[1] = 0; ss.hlen[2] = 0;
    ss.have_prev = 0;
    ss.nskip = 0;
    ss.nhist = 0;
    ss.have_scales = 0;
    ss.converged = 0;

    if (tid == 0) {
        for (int i = 0; i < P.nobj; i++) {
            ss.win_max[i] = 0.0;
            ss.win_nfail[i] = 0;
        }
        err_g[gid] = 0;
        sh.ctrl = 0;
    }
    __syncthreads();

    int it = 0;
    for (it = 0; it < P.maxiter; it++) {
        double ch_class[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < P.nobj; i++) {
            if (tid == 0)
                load_member_weight(P, g, i, sh);
            __syncthreads();
            const double w00 = sh.Swsh[0], w01 = sh.Swsh[1],
                         w11 = sh.Swsh[2];

            phasor_tables(P, sh, tid);
            __syncthreads();

            mode_sum_partials(P, sh, tid, nt,
                              kim, iy, ix, kv, ku,
                              w00, w01, w11);
            __syncthreads();
            block_reduce6(sh, tid, nt);

            if (tid == 0)
                member_update(P, g, i, sh, fcomp,
                              w00, w01, w11,
                              ch_class, ss, err_g + gid);
            __syncthreads();

            // optional censig reduction at the NEW weight
            if (sh.ctrl & 2) {
                if (tid == 0) {
                    sh.Swsh[0] = g.sw[i * 3];
                    sh.Swsh[1] = g.sw[i * 3 + 1];
                    sh.Swsh[2] = g.sw[i * 3 + 2];
                }
                __syncthreads();
                censig_partials(P, sh, tid, nt, kv, ku, ef2);
                __syncthreads();
                block_reduce2(sh, tid, nt);
                if (tid == 0)
                    censig_finish(P, g, i, sh, ch_class, ss);
                __syncthreads();
            }
        }

        // thread 0: sweep-level machinery
        if (tid == 0) {
            if (sweep_converged(P, ch_class, ss)) {
                ss.converged = 1;
                sh.ctrl |= 1;
            }
            if (!(sh.ctrl & 1) && ((it + 1) % NC_WINDOW == 0))
                noncontraction_check(P, g, ss);
            if (!(sh.ctrl & 1))
                steffensen_step(P, g, ss);
        }
        __syncthreads();
        if (sh.ctrl & 1) break;
        __syncthreads();
    }

    // final-state measurement: the neighbor-corrected sums and
    // the error cross sums the CPU result path needs, saving its
    // two per-object mode passes
    for (int i = 0; i < P.nobj; i++) {
        if (tid == 0)
            load_member_weight(P, g, i, sh);
        __syncthreads();
        const double w00 = sh.Swsh[0], w01 = sh.Swsh[1],
                     w11 = sh.Swsh[2];

        phasor_tables(P, sh, tid);
        __syncthreads();

        // moment-sum reduction at the final state
        mode_sum_partials(P, sh, tid, nt,
                          kim, iy, ix, kv, ku,
                          w00, w01, w11);
        __syncthreads();
        block_reduce6(sh, tid, nt);

        if (tid == 0)
            final_object_sums(P, g, i, sh, fcomp,
                              w00, w01, w11, objsums_g);
        __syncthreads();

        // error cross-sum reductions, chunks of <= 6 slots
        for (int c = 0; c < 3; c++) {
            const int cs = c * 6;
            const int ns = (13 - cs < 6) ? 13 - cs : 6;
            cross_sum_partials(P, sh, tid, nt, kv, ku, ef2,
                               w00, w01, w11, cs, ns);
            __syncthreads();
            block_reduce6(sh, tid, nt);
            if (tid == 0) {
                double* oc = objcov_g + (P.o0 + i) * 13;
                for (int s = 0; s < ns; s++)
                    oc[cs + s] = sh.red[s][0];
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        numiter_g[gid] = (it < P.maxiter) ? it + 1 : P.maxiter;
        converged_g[gid] = ss.converged;
        nskip_g[gid] = ss.nskip;
    }
}
