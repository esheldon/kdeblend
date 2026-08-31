// Per-object measured flux sums at the guess weights: the
// device analog of ngmix admom_ksums, used by the device-prep
// path to feed the flux initialization (_init_fluxes) without
// bringing the mode arrays to the host.  One block per
// (group, object) pair, reusing the shared phasor and mode-sum
// machinery; geometry (iy/ix/kv/ku) is a separate CSR shared by
// all groups with the same (dim, jacobian, Tsmooth).
#include "common.cuh"
#include "reduce.cuh"

extern "C" __global__ void init_sums(
    // mode data (CSR by group)
    const KIMT* __restrict__ kim,
    const long* __restrict__ moff,        // ngroup+1
    // geometry (CSR by geometry class)
    const int* __restrict__ iy_geo,
    const int* __restrict__ ix_geo,
    const MODET* __restrict__ kv_geo,
    const MODET* __restrict__ ku_geo,
    const long* __restrict__ goff,        // ngeom+1
    const int* __restrict__ geom_of_group,
    // per-group scalars
    const int* __restrict__ dima,
    const double* __restrict__ df2a,
    const double* __restrict__ drowa,
    const double* __restrict__ dcola,
    const double* __restrict__ jaca,      // ngroup x 4
    // per-pair inputs
    const int* __restrict__ pair_group,
    const double* __restrict__ dva,       // v - vcen
    const double* __restrict__ dua,       // u - ucen
    const double* __restrict__ swa,       // npair x 3
    double* __restrict__ esums_out)       // npair x 6
{
    const int p = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int gid = pair_group[p];
    const int geom = geom_of_group[gid];

    Params P;
    P.dim = dima[gid];
    P.nm = moff[gid + 1] - moff[gid];
    const long m0k = moff[gid];
    const long m0g = goff[geom];
    const double df2 = df2a[gid];
    const double a00 = jaca[gid * 4], a01 = jaca[gid * 4 + 1];
    const double a10 = jaca[gid * 4 + 2], a11 = jaca[gid * 4 + 3];
    const double w00 = swa[p * 3], w01 = swa[p * 3 + 1],
                 w11 = swa[p * 3 + 2];

    __shared__ Shm sh;
    if (tid == 0) {
        sh.absh[0] = drowa[gid] + a00 * dva[p] + a10 * dua[p];
        sh.absh[1] = dcola[gid] + a01 * dva[p] + a11 * dua[p];
    }
    __syncthreads();
    phasor_tables(P, sh, tid);
    __syncthreads();
    mode_sum_partials(P, sh, tid, nt,
                      kim, iy_geo, ix_geo, kv_geo, ku_geo,
                      w00, w01, w11, m0k, m0g);
    __syncthreads();
    block_reduce6(sh, tid, nt);

    if (tid == 0) {
        const double r0 = sh.red[0][0], rv = sh.red[1][0],
                     ru = sh.red[2][0];
        const double rvv = sh.red[3][0], rvu = sh.red[4][0],
                     ruu = sh.red[5][0];
        const double vv = w00 * r0 - rvv;
        const double vu = w01 * r0 - rvu;
        const double uu = w11 * r0 - ruu;
        double* out = esums_out + (long)p * 6;
        out[0] = rv * df2;
        out[1] = ru * df2;
        out[2] = (uu - vv) * df2;
        out[3] = 2.0 * vu * df2;
        out[4] = (uu + vv) * df2;
        out[5] = r0 * df2;
    }
}
