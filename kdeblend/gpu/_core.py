"""
The kdeblend group-replace fitter as one CUDA kernel: whole-fit-per-block.

Boundary: construction on CPU (kd.build_deblender); the kernel runs
the complete sweep loop (member steps, gauss/star/exp updates,
damped mixture steps, skip/containment/restart/demote, matched and
predicted-ratio fluxes, ratcheted change classes, projected-residual
convergence, noncontraction window, Steffensen extrapolation with
validity rollback, regularized recentering with lazy pull noise) and
returns the final state for state-level comparison with the class.

Scope v1: gauss + star + exp, single band/epoch, e_sigma0=0, no
bdf/dev.  fp64 throughout; the kernel mirrors diffrig.py's my_*
functions (the bitwise-validated executable spec) line by line.
Sum reduction order differs from the serial CPU kernels, so parity
is expected at ~1e-13 state level with occasional threshold flips,
not bitwise.

fp32-mixed variant (get_kernel(fp32=True) + pack_groups(fp32=True)):
mode data stored fp32 (complex64 kim, float32 kv/ku/ef2), per-mode
math in fp32 with Kahan-compensated per-thread bins, tree reduction
and ALL state/decision logic unchanged in fp64.  Same source,
compiled with -DMODE_FP32; the fp64 path is byte-identical to the
validated kernel.
"""
import numpy as np
import cupy as cp

from ngmix.prepsfadmom.models import get_profile_comps

TGAUSS = 0
TSTAR = 1
TEXP = 2
TYPECODE = {'gauss': TGAUSS, 'star': TSTAR, 'exp': TEXP}
TYPENAME = {v: k for k, v in TYPECODE.items()}

_EXP_COMPS = get_profile_comps('exp')
NEXPC = len(_EXP_COMPS)

_EXPVALS = ', '.join(f'{v:.17e}' for v in np.exp(np.arange(-15, 1)))
_EXPVALSF = ', '.join(
    f'{v:.9e}f' for v in np.exp(np.arange(-15, 1)).astype(np.float32)
)
_EXPFRAC = ', '.join(f'{f:.17e}' for f, cT in _EXP_COMPS)
_EXPCT = ', '.join(f'{cT:.17e}' for f, cT in _EXP_COMPS)

GMAX = 64
SPO = 9         # packed slots per object (F,1 + cen,2 + cov,3 + Sw,3)
NT = 256
DIM_MAX = 512

KSRC = r'''
#define M_PI 3.14159265358979323846
#define NT_H NTHREADS_H
#define NFAIL_LIMIT 10
#define RHO_CAP 0.999
#define NC_WINDOW 50
#define NC_FAC 0.7
#define RECENTER_CLIP_FAC 0.5
#define RESTARTED 2
#define DEBLENDED_AS_PSF 1
#define TGAUSS 0
#define TSTAR 1
#define TEXP 2

#ifdef MODE_FP32
#define KIMT float2
#define MODET float
#else
#define KIMT double2
#define MODET double
#endif

__constant__ double EXPLOOK[16] = {EXPVALS_HERE};
__constant__ float EXPLOOKF[16] = {EXPVALSF_HERE};

// the 13 unique nonzero entries of the 6x6 error cross-sum matrix
// (odd/center block 3 + even block 10; odd x even vanish by parity)
__constant__ int CROSSA[13] = {0,0,1,2,2,2,2,3,3,3,4,4,5};
__constant__ int CROSSB[13] = {0,1,1,2,3,4,5,3,4,5,4,5,5};
__constant__ double EXPFRAC[NEXPC_H] = {EXPFRAC_HERE};
__constant__ double EXPCT[NEXPC_H] = {EXPCT_HERE};

__device__ inline double fexp(double x)
{
    int ival = (int)(x - 0.5);
    double f = x - ival;
    double e = EXPLOOK[ival + 15];
    return e * (1.0000011318561302 + f*(0.999993601071577
        + f*(0.49992478810274166 + f*(0.16674612720799442
        + f*(0.042330947141114836 + f*0.008197933236258961)))));
}

// same table/poly rounded to fp32; matches the CPU fexp curve to
// fp32 rounding rather than introducing an exp-vs-fexp systematic
__device__ inline float fexpf_(float x)
{
    int ival = (int)(x - 0.5f);
    float f = x - ival;
    float e = EXPLOOKF[ival + 15];
    return e * (1.0000011318561302f + f*(0.999993601071577f
        + f*(0.49992478810274166f + f*(0.16674612720799442f
        + f*(0.042330947141114836f + f*0.008197933236258961f)))));
}

// Kahan-compensated fp32 add (adds/subs only: no FMA contraction,
// and NVRTC does not reassociate, so the compensation survives)
#define KAHAN(s, c, xin) do { \
    const float y_ = (xin) - (c); \
    const float t_ = (s) + y_; \
    (c) = (t_ - (s)) - y_; \
    (s) = t_; \
    } while (0)

// per-group state layout in global memory (see pack_groups)
struct GState {
    // per object
    int* otype;
    double* F;        // nobj
    double* fscale;   // nobj
    double* cov;      // nobj x 3 (cov_sm for gauss/star, cov for exp)
    double* sw;       // nobj x 3
    double* pos;      // nobj x 2
    double* dpos;     // nobj x 2 (detection positions)
    double* cen_pull; // nobj x 2
    double* censig;   // nobj (-1 unset)
    int* nfail;
    int* nrestart;
    int* dbflags;
    int* fixcen;
    // scratch for extrapolation (each length smax = SPO*nobj)
    double* hist;     // 3 x smax
    double* scales;   // smax (scales[0] < 0 => unset)
    double* saved;    // smax (rollback buffer, raw values)
    double* xtmp;     // smax
};

// closed-form weighted sums of one gaussian component (detAtinv=1)
__device__ void comp_sums(
    double F, double So00, double So01, double So11,
    double dv, double du,
    double sw00, double sw01, double sw11, double* out)
{
    const double C00 = sw00 + So00;
    const double C01 = sw01 + So01;
    const double C11 = sw11 + So11;
    const double det = C00 * C11 - C01 * C01;
    const double idet = 1.0 / det;
    const double Ci00 = C11 * idet;
    const double Ci01 = -C01 * idet;
    const double Ci11 = C00 * idet;
    const double Cd0 = Ci00 * dv + Ci01 * du;
    const double Cd1 = Ci01 * dv + Ci11 * du;
    const double sflux = F * exp(-0.5 * (dv * Cd0 + du * Cd1))
        / (2.0 * M_PI * sqrt(det));
    const double mu0 = sw00 * Cd0 + sw01 * Cd1;
    const double mu1 = sw01 * Cd0 + sw11 * Cd1;
    const double A00 = sw00 * Ci00 + sw01 * Ci01;
    const double A01 = sw00 * Ci01 + sw01 * Ci11;
    const double A10 = sw01 * Ci00 + sw11 * Ci01;
    const double A11 = sw01 * Ci01 + sw11 * Ci11;
    const double Sp00 = A00 * So00 + A01 * So01;
    const double Sp01 = A00 * So01 + A01 * So11;
    const double Sp11 = A10 * So01 + A11 * So11;
    const double vv = Sp00 + mu0 * mu0;
    const double vu = Sp01 + mu0 * mu1;
    const double uu = Sp11 + mu1 * mu1;
    out[0] += sflux * mu0;
    out[1] += sflux * mu1;
    out[2] += sflux * (uu - vv);
    out[3] += sflux * 2.0 * vu;
    out[4] += sflux * (uu + vv);
    out[5] += sflux;
}

// model comps of object j under weight sw at offset, accumulated
__device__ void model_sums(
    const GState& g, int j, double dv, double du, double Tsmooth,
    double sw00, double sw01, double sw11, double* out)
{
    if (g.otype[j] == TEXP) {
        const double smooth = Tsmooth / 2.0;
        for (int c = 0; c < NEXPC_H; c++) {
            const double cT = EXPCT[c];
            comp_sums(
                g.F[j] * EXPFRAC[c],
                cT * g.cov[j * 3 + 0] + smooth,
                cT * g.cov[j * 3 + 1],
                cT * g.cov[j * 3 + 2] + smooth,
                dv, du, sw00, sw01, sw11, out);
        }
    } else {
        // gauss and star: cov holds cov_sm (already smoothed plane)
        comp_sums(
            g.F[j], g.cov[j * 3 + 0], g.cov[j * 3 + 1],
            g.cov[j * 3 + 2], dv, du, sw00, sw01, sw11, out);
    }
}

__device__ int mixture_valid(
    double Sf00, double Sf01, double Sf11,
    double sw00, double sw01, double sw11, double Tsmooth)
{
    const double smooth = Tsmooth / 2.0;
    for (int c = 0; c < NEXPC_H; c++) {
        const double cT = EXPCT[c];
        const double C00 = sw00 + cT * Sf00 + smooth;
        const double C01 = sw01 + cT * Sf01;
        const double C11 = sw11 + cT * Sf11 + smooth;
        const double det = C00 * C11 - C01 * C01;
        if (C00 <= 0 || C11 <= 0 || det <= 1.0e-6 * C00 * C11)
            return 0;
    }
    return 1;
}

// deweight: (M^-1 - Sw^-1)^-1; returns 0 flags ok
__device__ int deweight(
    double Mvv, double Mvu, double Muu,
    double s00, double s01, double s11,
    double* n00, double* n01, double* n11)
{
    const double detm = Mvv * Muu - Mvu * Mvu;
    if (detm <= 1e-200) return 1;
    const double detw = s00 * s11 - s01 * s01;
    if (detw <= 1e-200) return 1;
    const double idm = 1.0 / detm;
    const double idw = 1.0 / detw;
    const double Nvv = Muu * idm - s11 * idw;
    const double Nuu = Mvv * idm - s00 * idw;
    const double Nvu = -Mvu * idm + s01 * idw;
    const double detn = Nvv * Nuu - Nvu * Nvu;
    if (detn <= 1e-200 || Nvv <= 0 || Nuu <= 0) return 1;
    const double idn = 1.0 / detn;
    *n00 = Nuu * idn;
    *n01 = -Nvu * idn;
    *n11 = Nvv * idn;
    return 0;
}

// pack the group state (normalized); returns length
__device__ int pack_state(
    const GState& g, int nobj, int recenter, double* x,
    double* scales, int set_scales)
{
    int k = 0;
    for (int i = 0; i < nobj; i++) {
        x[k++] = g.F[i];
        if (recenter) {
            x[k++] = g.pos[i * 2] - g.dpos[i * 2] + 1.0;
            x[k++] = g.pos[i * 2 + 1] - g.dpos[i * 2 + 1] + 1.0;
        }
        if (g.otype[i] != TSTAR) {
            x[k++] = g.cov[i * 3];
            x[k++] = g.cov[i * 3 + 1];
            x[k++] = g.cov[i * 3 + 2];
            x[k++] = g.sw[i * 3];
            x[k++] = g.sw[i * 3 + 1];
            x[k++] = g.sw[i * 3 + 2];
        }
    }
    if (set_scales) {
        for (int m = 0; m < k; m++) {
            double a = fabs(x[m]);
            scales[m] = (a > 1.0e-10) ? a : 1.0e-10;
        }
    }
    for (int m = 0; m < k; m++) x[m] /= scales[m];
    return k;
}

__device__ void unpack_state(
    const GState& g, int nobj, int recenter, const double* x,
    const double* scales)
{
    int k = 0;
    for (int i = 0; i < nobj; i++) {
        g.F[i] = x[k] * scales[k]; k++;
        if (recenter) {
            g.pos[i * 2] = g.dpos[i * 2]
                + x[k] * scales[k] - 1.0; k++;
            g.pos[i * 2 + 1] = g.dpos[i * 2 + 1]
                + x[k] * scales[k] - 1.0; k++;
        }
        if (g.otype[i] != TSTAR) {
            g.cov[i * 3] = x[k] * scales[k]; k++;
            g.cov[i * 3 + 1] = x[k] * scales[k]; k++;
            g.cov[i * 3 + 2] = x[k] * scales[k]; k++;
            g.sw[i * 3] = x[k] * scales[k]; k++;
            g.sw[i * 3 + 1] = x[k] * scales[k]; k++;
            g.sw[i * 3 + 2] = x[k] * scales[k]; k++;
        }
    }
}

__device__ int state_valid(
    const GState& g, int nobj, int recenter, double Tsmooth)
{
    for (int i = 0; i < nobj; i++) {
        const double s00 = g.sw[i * 3], s01 = g.sw[i * 3 + 1],
                     s11 = g.sw[i * 3 + 2];
        if (s00 <= 0 || s11 <= 0 || s00 * s11 - s01 * s01 <= 0)
            return 0;
        if (g.otype[i] == TEXP) {
            if (!mixture_valid(
                    g.cov[i * 3], g.cov[i * 3 + 1],
                    g.cov[i * 3 + 2], 0.0, 0.0, 0.0, Tsmooth))
                return 0;
        } else if (g.otype[i] == TGAUSS) {
            if (g.cov[i * 3] * g.cov[i * 3 + 2]
                - g.cov[i * 3 + 1] * g.cov[i * 3 + 1] <= 0)
                return 0;
        }
        if (recenter) {
            const double d0 = g.pos[i * 2] - g.dpos[i * 2];
            const double d1 = g.pos[i * 2 + 1] - g.dpos[i * 2 + 1];
            if (d0 * d0 + d1 * d1
                > RECENTER_CLIP_FAC * RECENTER_CLIP_FAC * Tsmooth)
                return 0;
        }
    }
    return 1;
}

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

    const int nobj = nobja[gid];
    const int dim = dima[gid];
    const double df2 = df2a[gid];
    const double drow = drowa[gid];
    const double dcol = dcola[gid];
    const double a00 = jaca[gid * 4], a01 = jaca[gid * 4 + 1];
    const double a10 = jaca[gid * 4 + 2], a11 = jaca[gid * 4 + 3];
    const double weight = wta[gid];
    const double detatinv = detia[gid];
    const double fac = weight * detatinv;
    const double Tsmooth = tsma[gid];
    const double smoothcov = Tsmooth / 2.0;
    const double tol = tola[gid];
    const double flux_tol = ftola[gid];
    const double cen_tol = ctola[gid];
    const int maxiter = maxitera[gid];
    const int recenter = recentera[gid];
    const double cen_sigma0 = censig0a[gid];

    const long m0 = moff[gid];
    const long nm = moff[gid + 1] - m0;
    const long f0 = foff[gid];
    const int nfc = (int)(foff[gid + 1] - f0);
    const long o0 = ooff[gid];
    const long s0 = soff[gid];
    const int smax = (int)(soff[gid + 1] - s0);

    GState g;
    g.otype = otype_g + o0;
    g.F = F_g + o0;
    g.fscale = fscale_g + o0;
    g.cov = cov_g + o0 * 3;
    g.sw = sw_g + o0 * 3;
    g.pos = pos_g + o0 * 2;
    g.dpos = dpos_g + o0 * 2;
    g.cen_pull = cpull_g + o0 * 2;
    g.censig = censig_g + o0;
    g.nfail = nfail_g + o0;
    g.nrestart = nrestart_g + o0;
    g.dbflags = dbflags_g + o0;
    g.fixcen = fixcen_g + o0;
    g.hist = hist_g + 3 * s0;
    g.scales = scales_g + s0;
    g.saved = saved_g + s0;
    g.xtmp = xtmp_g + s0;

    __shared__ MODET pyre[DIM_MAX_H], pyim[DIM_MAX_H];
    __shared__ MODET pxre[DIM_MAX_H], pxim[DIM_MAX_H];
    __shared__ double red[6][NT_H];
    __shared__ double Swsh[3];
    __shared__ double absh[2];   // alpha, beta
    __shared__ int ctrl;         // bit0: stop; bit1: censig pass

    // thread-0 sweep-level state
    double hprev[3] = {-1.0, -1.0, -1.0};  // flux, struct, cen
    int hlen[3] = {0, 0, 0};
    double win_max_l[GMAX_H];
    double prev_win[GMAX_H];
    long win_nfail_l[GMAX_H];
    long prev_nfail[GMAX_H];
    int have_prev = 0;
    int nskip = 0;
    int nhist = 0;      // extrapolation history length (0..3)
    int have_scales = 0;
    int converged = 0;
    int it = 0;

    if (tid == 0) {
        for (int i = 0; i < nobj; i++) {
            win_max_l[i] = 0.0;
            win_nfail_l[i] = 0;
        }
        err_g[gid] = 0;
        ctrl = 0;
    }
    __syncthreads();

    for (it = 0; it < maxiter; it++) {
        double ch_class[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < nobj; i++) {
            // ---- thread 0: set weight/phases for the member ----
            if (tid == 0) {
                Swsh[0] = g.sw[i * 3];
                Swsh[1] = g.sw[i * 3 + 1];
                Swsh[2] = g.sw[i * 3 + 2];
                const double vi = g.pos[i * 2];
                const double ui = g.pos[i * 2 + 1];
                absh[0] = drow + a00 * vi + a10 * ui;
                absh[1] = dcol + a01 * vi + a11 * ui;
            }
            __syncthreads();
            const double w00 = Swsh[0], w01 = Swsh[1],
                         w11 = Swsh[2];

            // ---- phasor tables (complex recurrence) ----
            {
                const int half = (dim + 1) / 2;
                if (tid < 2) {
                    const double facp = 2.0 * M_PI / dim
                        * absh[tid];
                    MODET* pre = (tid == 0) ? pyre : pxre;
                    MODET* pim = (tid == 0) ? pyim : pxim;
                    double s1, c1;
                    sincos(facp, &s1, &c1);
                    double pr = 1.0, pi = 0.0;
                    pre[0] = 1.0; pim[0] = 0.0;
                    for (int f = 1; f < half; f++) {
                        const double nr = pr * c1 - pi * s1;
                        pi = pr * s1 + pi * c1; pr = nr;
                        pre[f] = pr; pim[f] = pi;
                    }
                    pr = 1.0; pi = 0.0;
                    for (int f = -1; f >= half - dim; f--) {
                        const double nr = pr * c1 + pi * s1;
                        pi = pi * c1 - pr * s1; pr = nr;
                        pre[f + dim] = pr; pim[f + dim] = pi;
                    }
                }
            }
            __syncthreads();

            // ---- mode-sum reduction ----
#ifdef MODE_FP32
            {
                const float w00f = (float)w00,
                            w01f = (float)w01,
                            w11f = (float)w11;
                float s0_ = 0, sv = 0, su = 0,
                      svv = 0, svu = 0, suu = 0;
                float k0_ = 0, kv_ = 0, ku_ = 0,
                      kvv = 0, kvu = 0, kuu = 0;
                for (long m = tid; m < nm; m += nt) {
                    const float kvi = kv[m0 + m];
                    const float kui = ku[m0 + m];
                    const float Sv = w00f * kvi + w01f * kui;
                    const float Su = w01f * kvi + w11f * kui;
                    const float chi2 = kvi * Sv + kui * Su;
                    if (chi2 > 25.0f || chi2 < 0.0f) continue;
                    const float wk = fexpf_(-0.5f * chi2);
                    const int y = iy[m0 + m];
                    const int x = ix[m0 + m];
                    const float pr = pyre[y] * pxre[x]
                        - pyim[y] * pxim[x];
                    const float pi = pyre[y] * pxim[x]
                        + pyim[y] * pxre[x];
                    const float2 val = kim[m0 + m];
                    const float re = val.x * pr - val.y * pi;
                    const float im = val.y * pr + val.x * pi;
                    const float wre = wk * re;
                    const float wim = wk * im;
                    KAHAN(s0_, k0_, wre);
                    KAHAN(sv, kv_, -(Sv * wim));
                    KAHAN(su, ku_, -(Su * wim));
                    KAHAN(svv, kvv, Sv * Sv * wre);
                    KAHAN(svu, kvu, Sv * Su * wre);
                    KAHAN(suu, kuu, Su * Su * wre);
                }
                red[0][tid] = (double)s0_ + (double)k0_;
                red[1][tid] = (double)sv + (double)kv_;
                red[2][tid] = (double)su + (double)ku_;
                red[3][tid] = (double)svv + (double)kvv;
                red[4][tid] = (double)svu + (double)kvu;
                red[5][tid] = (double)suu + (double)kuu;
            }
#else
            {
                double s0_ = 0, sv = 0, su = 0,
                       svv = 0, svu = 0, suu = 0;
                for (long m = tid; m < nm; m += nt) {
                    const double kvi = kv[m0 + m];
                    const double kui = ku[m0 + m];
                    const double Sv = w00 * kvi + w01 * kui;
                    const double Su = w01 * kvi + w11 * kui;
                    const double chi2 = kvi * Sv + kui * Su;
                    if (chi2 > 25.0 || chi2 < 0.0) continue;
                    const double wk = fexp(-0.5 * chi2);
                    const int y = iy[m0 + m];
                    const int x = ix[m0 + m];
                    const double pr = pyre[y] * pxre[x]
                        - pyim[y] * pxim[x];
                    const double pi = pyre[y] * pxim[x]
                        + pyim[y] * pxre[x];
                    const double2 val = kim[m0 + m];
                    const double re = val.x * pr - val.y * pi;
                    const double im = val.y * pr + val.x * pi;
                    const double wre = wk * re;
                    const double wim = wk * im;
                    s0_ += wre;
                    sv -= Sv * wim;
                    su -= Su * wim;
                    svv += Sv * Sv * wre;
                    svu += Sv * Su * wre;
                    suu += Su * Su * wre;
                }
                red[0][tid] = s0_; red[1][tid] = sv;
                red[2][tid] = su; red[3][tid] = svv;
                red[4][tid] = svu; red[5][tid] = suu;
            }
#endif
            __syncthreads();
            for (int st = nt / 2; st > 0; st >>= 1) {
                if (tid < st)
                    for (int k = 0; k < 6; k++)
                        red[k][tid] += red[k][tid + st];
                __syncthreads();
            }

            // ---- thread 0: the member update ----
            if (tid == 0) {
                const double vi = g.pos[i * 2];
                const double ui = g.pos[i * 2 + 1];
                // neighbor + fixed closed-form sums, detAtinv=1
                double base[6] = {0, 0, 0, 0, 0, 0};
                for (int j = 0; j < nobj; j++) {
                    if (j == i) continue;
                    model_sums(g, j,
                               g.pos[j * 2] - vi,
                               g.pos[j * 2 + 1] - ui,
                               Tsmooth, w00, w01, w11, base);
                }
                for (int c = 0; c < nfc; c++) {
                    const double* fc = fcomp + (f0 + c) * 6;
                    comp_sums(fc[0], fc[1], fc[2], fc[3],
                              fc[4] - vi, fc[5] - ui,
                              w00, w01, w11, base);
                }
                // predicted sums for exp
                double psums[6] = {0, 0, 0, 0, 0, 0};
                const int mytype = g.otype[i];
                if (mytype == TEXP) {
                    model_sums(g, i, 0.0, 0.0, Tsmooth,
                               w00, w01, w11, psums);
                }
                const double r0 = red[0][0], rv = red[1][0],
                             ru = red[2][0];
                const double rvv = red[3][0], rvu = red[4][0],
                             ruu = red[5][0];
                const double vv = w00 * r0 - rvv;
                const double vu = w01 * r0 - rvu;
                const double uu = w11 * r0 - ruu;
                double esums[6];
                esums[0] = rv * df2;
                esums[1] = ru * df2;
                esums[2] = (uu - vv) * df2;
                esums[3] = 2.0 * vu * df2;
                esums[4] = (uu + vv) * df2;
                esums[5] = r0 * df2;
                double sums[6];
                for (int k = 0; k < 6; k++)
                    sums[k] = fac * (esums[k] - base[k] / detatinv);
                const double fs = sums[5];
                const double ws = weight;
                double pred[6] = {0, 0, 0, 0, 0, 0};
                double fs_pred = 0.0;
                if (mytype == TEXP) {
                    for (int k = 0; k < 6; k++)
                        pred[k] = fac * (psums[k] / detatinv);
                    fs_pred = pred[5];
                }
                if (sums[5] > 0) {
                    g.cen_pull[i * 2] = sums[0] / sums[5];
                    g.cen_pull[i * 2 + 1] = sums[1] / sums[5];
                }

                double change = 0.0;
                // deweight of the measured moments
                double n00 = 0, n01 = 0, n11 = 0;
                int have_new = 0;
                if (sums[5] > 0 && sums[4] > 0) {
                    const double finv = 1.0 / sums[5];
                    const double M1 = sums[2] * finv;
                    const double M2 = sums[3] * finv;
                    const double Tm = sums[4] * finv;
                    if (!deweight(0.5 * (Tm - M1), 0.5 * M2,
                                  0.5 * (Tm + M1),
                                  g.sw[i * 3], g.sw[i * 3 + 1],
                                  g.sw[i * 3 + 2],
                                  &n00, &n01, &n11))
                        have_new = 1;
                }

                if (mytype == TSTAR) {
                    // flux-only: matched flux at cov_sm
                    double dd = (g.sw[i * 3] + g.cov[i * 3])
                        * (g.sw[i * 3 + 2] + g.cov[i * 3 + 2])
                        - (g.sw[i * 3 + 1] + g.cov[i * 3 + 1])
                        * (g.sw[i * 3 + 1] + g.cov[i * 3 + 1]);
                    const double newF = fs / ws * 2.0 * M_PI
                        * sqrt(dd);
                    if (fabs(newF) > g.fscale[i])
                        g.fscale[i] = fabs(newF);
                    const double chf = fabs(newF - g.F[i])
                        / g.fscale[i];
                    g.F[i] = newF;
                    if (chf > ch_class[0]) ch_class[0] = chf;
                    change = chf;
                } else if (!have_new) {
                    // skip-structure path
                    nskip += 1;
                    if (nskip > 100 * nobj) {
                        err_g[gid] = 1;
                        ctrl = 1;
                    }
                    if (mytype == TGAUSS) {
                        double dd = (g.sw[i * 3] + g.cov[i * 3])
                            * (g.sw[i * 3 + 2]
                               + g.cov[i * 3 + 2])
                            - (g.sw[i * 3 + 1]
                               + g.cov[i * 3 + 1])
                            * (g.sw[i * 3 + 1]
                               + g.cov[i * 3 + 1]);
                        const double newF = fs / ws * 2.0 * M_PI
                            * sqrt(dd);
                        if (fabs(newF) > g.fscale[i])
                            g.fscale[i] = fabs(newF);
                        const double chf = fabs(newF - g.F[i])
                            / g.fscale[i];
                        g.F[i] = newF;
                        if (chf > ch_class[0]) ch_class[0] = chf;
                    } else if (fs_pred != 0.0) {
                        const double newF = g.F[i] * fs / fs_pred;
                        if (fabs(newF) > g.fscale[i])
                            g.fscale[i] = fabs(newF);
                        const double chf = fabs(newF - g.F[i])
                            / g.fscale[i];
                        g.F[i] = newF;
                        if (chf > ch_class[0]) ch_class[0] = chf;
                    }
                    // contain failure (non-forced)
                    g.nfail[i] += 1;
                    win_nfail_l[i] += 1;
                    if (g.nfail[i] >= NFAIL_LIMIT) {
                        g.nfail[i] = 0;
                        g.sw[i * 3] = smoothcov;
                        g.sw[i * 3 + 1] = 0.0;
                        g.sw[i * 3 + 2] = smoothcov;
                        if (recenter) {
                            g.pos[i * 2] = g.dpos[i * 2];
                            g.pos[i * 2 + 1] = g.dpos[i * 2 + 1];
                        }
                        if (g.nrestart[i] == 0) {
                            g.nrestart[i] = 1;
                            g.dbflags[i] |= RESTARTED;
                            if (g.otype[i] == TEXP) {
                                g.cov[i * 3] = 0.0;
                                g.cov[i * 3 + 1] = 0.0;
                                g.cov[i * 3 + 2] = 0.0;
                            } else {
                                g.cov[i * 3] = smoothcov;
                                g.cov[i * 3 + 1] = 0.0;
                                g.cov[i * 3 + 2] = smoothcov;
                            }
                            nhist = 0;
                            hprev[0] = -1; hprev[1] = -1;
                            hprev[2] = -1;
                            hlen[0] = 0; hlen[1] = 0; hlen[2] = 0;
                        } else {
                            g.dbflags[i] |= DEBLENDED_AS_PSF;
                            g.otype[i] = TSTAR;
                            g.cov[i * 3] = smoothcov;
                            g.cov[i * 3 + 1] = 0.0;
                            g.cov[i * 3 + 2] = smoothcov;
                            nhist = 0;
                            have_scales = 0;
                            hprev[0] = -1; hprev[1] = -1;
                            hprev[2] = -1;
                            hlen[0] = 0; hlen[1] = 0; hlen[2] = 0;
                        }
                    }
                    if (1.0 > ch_class[1]) ch_class[1] = 1.0;
                    change = 1.0;
                } else if (mytype == TGAUSS) {
                    // accept gauss update
                    double dd = (g.sw[i * 3] + n00)
                        * (g.sw[i * 3 + 2] + n11)
                        - (g.sw[i * 3 + 1] + n01)
                        * (g.sw[i * 3 + 1] + n01);
                    const double newF = fs / ws * 2.0 * M_PI
                        * sqrt(dd);
                    const double Twt = g.sw[i * 3]
                        + g.sw[i * 3 + 2];
                    double sm = fabs(n00 - g.sw[i * 3]);
                    const double s2 = fabs(n01 - g.sw[i * 3 + 1]);
                    const double s3 = fabs(n11 - g.sw[i * 3 + 2]);
                    if (s2 > sm) sm = s2;
                    if (s3 > sm) sm = s3;
                    const double chs = sm / Twt;
                    if (chs > ch_class[1]) ch_class[1] = chs;
                    if (fabs(newF) > g.fscale[i])
                        g.fscale[i] = fabs(newF);
                    const double chf = fabs(newF - g.F[i])
                        / g.fscale[i];
                    if (chf > ch_class[0]) ch_class[0] = chf;
                    change = (chs > chf) ? chs : chf;
                    g.cov[i * 3] = n00;
                    g.cov[i * 3 + 1] = n01;
                    g.cov[i * 3 + 2] = n11;
                    g.F[i] = newF;
                    g.nfail[i] = 0;
                    g.sw[i * 3] = n00;
                    g.sw[i * 3 + 1] = n01;
                    g.sw[i * 3 + 2] = n11;
                } else {
                    // exp mixture update
                    double sh00, sh01, sh11;
                    // mixture shift: deweight of predicted moments
                    double p00 = 0, p01 = 0, p11 = 0;
                    int pok = 0;
                    if (pred[5] != 0.0) {
                        const double pfinv = 1.0 / pred[5];
                        const double pM1 = pred[2] * pfinv;
                        const double pM2 = pred[3] * pfinv;
                        const double pT = pred[4] * pfinv;
                        if (!deweight(0.5 * (pT - pM1),
                                      0.5 * pM2,
                                      0.5 * (pT + pM1),
                                      g.sw[i * 3],
                                      g.sw[i * 3 + 1],
                                      g.sw[i * 3 + 2],
                                      &p00, &p01, &p11))
                            pok = 1;
                    }
                    if (pok) {
                        sh00 = n00 - p00;
                        sh01 = n01 - p01;
                        sh11 = n11 - p11;
                    } else {
                        const double Tp = pred[4]
                            * (1.0 / pred[5]);
                        const double Tf = g.cov[i * 3]
                            + g.cov[i * 3 + 2];
                        const double facr = sums[4] / sums[5] / Tp;
                        const double de1 = sums[2] / sums[4]
                            - pred[2] / pred[4];
                        const double de2 = sums[3] / sums[4]
                            - pred[3] / pred[4];
                        sh00 = (facr - 1.0) * g.cov[i * 3]
                            + 0.5 * facr * Tf * (-de1);
                        sh01 = (facr - 1.0) * g.cov[i * 3 + 1]
                            + 0.5 * facr * Tf * de2;
                        sh11 = (facr - 1.0) * g.cov[i * 3 + 2]
                            + 0.5 * facr * Tf * de1;
                    }
                    // damped step
                    int accepted = 0;
                    int idamp = 0;
                    double pr00 = 0, pr01 = 0, pr11 = 0;
                    for (idamp = 0; idamp < 10; idamp++) {
                        pr00 = g.cov[i * 3] + sh00;
                        pr01 = g.cov[i * 3 + 1] + sh01;
                        pr11 = g.cov[i * 3 + 2] + sh11;
                        if (mixture_valid(pr00, pr01, pr11,
                                          0.0, 0.0, 0.0,
                                          Tsmooth)) {
                            accepted = 1;
                            break;
                        }
                        sh00 *= 0.5; sh01 *= 0.5; sh11 *= 0.5;
                    }
                    // flux first (always)
                    const double newF = g.F[i] * fs / fs_pred;
                    if (fabs(newF) > g.fscale[i])
                        g.fscale[i] = fabs(newF);
                    const double chf = fabs(newF - g.F[i])
                        / g.fscale[i];
                    g.F[i] = newF;
                    if (chf > ch_class[0]) ch_class[0] = chf;
                    change = chf;
                    int contained = 0;
                    if (!accepted) {
                        nskip += 1;
                        if (nskip > 100 * nobj) {
                            err_g[gid] = 1;
                            ctrl = 1;
                        }
                        if (1.0 > ch_class[1]) ch_class[1] = 1.0;
                        if (1.0 > change) change = 1.0;
                        g.nfail[i] += 1;
                        win_nfail_l[i] += 1;
                        if (g.nfail[i] >= NFAIL_LIMIT) {
                            contained = 1;
                            g.nfail[i] = 0;
                            g.sw[i * 3] = smoothcov;
                            g.sw[i * 3 + 1] = 0.0;
                            g.sw[i * 3 + 2] = smoothcov;
                            if (recenter) {
                                g.pos[i * 2] = g.dpos[i * 2];
                                g.pos[i * 2 + 1] =
                                    g.dpos[i * 2 + 1];
                            }
                            if (g.nrestart[i] == 0) {
                                g.nrestart[i] = 1;
                                g.dbflags[i] |= RESTARTED;
                                g.cov[i * 3] = 0.0;
                                g.cov[i * 3 + 1] = 0.0;
                                g.cov[i * 3 + 2] = 0.0;
                                nhist = 0;
                                hprev[0] = -1; hprev[1] = -1;
                                hprev[2] = -1;
                                hlen[0] = 0; hlen[1] = 0;
                                hlen[2] = 0;
                            } else {
                                g.dbflags[i] |= DEBLENDED_AS_PSF;
                                g.otype[i] = TSTAR;
                                g.cov[i * 3] = smoothcov;
                                g.cov[i * 3 + 1] = 0.0;
                                g.cov[i * 3 + 2] = smoothcov;
                                nhist = 0;
                                have_scales = 0;
                                hprev[0] = -1; hprev[1] = -1;
                                hprev[2] = -1;
                                hlen[0] = 0; hlen[1] = 0;
                                hlen[2] = 0;
                            }
                        }
                    } else if (idamp > 0) {
                        if (1.0 > ch_class[1]) ch_class[1] = 1.0;
                        if (1.0 > change) change = 1.0;
                        g.cov[i * 3] = pr00;
                        g.cov[i * 3 + 1] = pr01;
                        g.cov[i * 3 + 2] = pr11;
                        g.nfail[i] = 0;
                        win_nfail_l[i] += 1;
                    } else {
                        const double Twt = g.sw[i * 3]
                            + g.sw[i * 3 + 2];
                        double sm = fabs(sh00);
                        if (fabs(sh01) > sm) sm = fabs(sh01);
                        if (fabs(sh11) > sm) sm = fabs(sh11);
                        const double chs = sm / Twt;
                        if (chs > ch_class[1]) ch_class[1] = chs;
                        if (chs > change) change = chs;
                        g.cov[i * 3] = pr00;
                        g.cov[i * 3 + 1] = pr01;
                        g.cov[i * 3 + 2] = pr11;
                        g.nfail[i] = 0;
                    }
                    // Sw moves to measured deweight unless the
                    // containment intervened
                    if (!contained) {
                        g.sw[i * 3] = n00;
                        g.sw[i * 3 + 1] = n01;
                        g.sw[i * 3 + 2] = n11;
                    }
                }

                // recenter (lazy pull noise -> request pass)
                if (recenter && sums[5] > 0 && !g.fixcen[i]) {
                    if (g.censig[i] < 0) {
                        // stash sums5; run the censig reduction
                        red[0][0] = sums[5];
                        ctrl |= 2;
                    } else {
                        const double s0c = cen_sigma0;
                        const double sig = g.censig[i];
                        const double denom = s0c * s0c + sig * sig;
                        if (isfinite(sig) && denom != 0.0) {
                            const double kk = s0c * s0c / denom;
                            const double v = g.pos[i * 2];
                            const double u = g.pos[i * 2 + 1];
                            const double v0 = g.dpos[i * 2];
                            const double u0 = g.dpos[i * 2 + 1];
                            double newv = v
                                + kk * g.cen_pull[i * 2]
                                + (1.0 - kk) * (v0 - v);
                            double newu = u
                                + kk * g.cen_pull[i * 2 + 1]
                                + (1.0 - kk) * (u0 - u);
                            const double clip = RECENTER_CLIP_FAC
                                * sqrt(Tsmooth);
                            double d0 = newv - v0;
                            double d1 = newu - u0;
                            const double nn = sqrt(
                                d0 * d0 + d1 * d1);
                            if (nn > clip) {
                                d0 *= clip / nn;
                                d1 *= clip / nn;
                                newv = v0 + d0;
                                newu = u0 + d1;
                            }
                            double dmax = fabs(newv - v);
                            if (fabs(newu - u) > dmax)
                                dmax = fabs(newu - u);
                            g.pos[i * 2] = newv;
                            g.pos[i * 2 + 1] = newu;
                            const double Twt = g.sw[i * 3]
                                + g.sw[i * 3 + 2];
                            const double chc = dmax / sqrt(Twt);
                            if (chc > ch_class[2])
                                ch_class[2] = chc;
                            if (chc > change) change = chc;
                        }
                    }
                }
                if (change > win_max_l[i]) win_max_l[i] = change;
            }
            __syncthreads();

            // ---- optional censig reduction at NEW weight ----
            if (ctrl & 2) {
                if (tid == 0) {
                    Swsh[0] = g.sw[i * 3];
                    Swsh[1] = g.sw[i * 3 + 1];
                    Swsh[2] = g.sw[i * 3 + 2];
                }
                __syncthreads();
#ifdef MODE_FP32
                const float u00 = (float)Swsh[0],
                            u01 = (float)Swsh[1],
                            u11 = (float)Swsh[2];
                float c0 = 0, c1 = 0, cc0 = 0, cc1 = 0;
                for (long m = tid; m < nm; m += nt) {
                    const float kvi = kv[m0 + m];
                    const float kui = ku[m0 + m];
                    const float Sv = u00 * kvi + u01 * kui;
                    const float Su = u01 * kvi + u11 * kui;
                    const float chi2 = kvi * Sv + kui * Su;
                    if (chi2 > 25.0f || chi2 < 0.0f) continue;
                    const float wk = fexpf_(-0.5f * chi2);
                    const float e2 = ef2[m0 + m];
                    KAHAN(c0, cc0, Sv * wk * Sv * wk * e2);
                    KAHAN(c1, cc1, Su * wk * Su * wk * e2);
                }
                red[1][tid] = (double)c0 + (double)cc0;
                red[2][tid] = (double)c1 + (double)cc1;
#else
                const double u00 = Swsh[0], u01 = Swsh[1],
                             u11 = Swsh[2];
                double c0 = 0, c1 = 0;
                for (long m = tid; m < nm; m += nt) {
                    const double kvi = kv[m0 + m];
                    const double kui = ku[m0 + m];
                    const double Sv = u00 * kvi + u01 * kui;
                    const double Su = u01 * kvi + u11 * kui;
                    const double chi2 = kvi * Sv + kui * Su;
                    if (chi2 > 25.0 || chi2 < 0.0) continue;
                    const double wk = fexp(-0.5 * chi2);
                    const double e2 = ef2[m0 + m];
                    c0 += Sv * wk * Sv * wk * e2;
                    c1 += Su * wk * Su * wk * e2;
                }
                red[1][tid] = c0;
                red[2][tid] = c1;
#endif
                __syncthreads();
                for (int st = nt / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        red[1][tid] += red[1][tid + st];
                        red[2][tid] += red[2][tid + st];
                    }
                    __syncthreads();
                }
                if (tid == 0) {
                    const double sums5 = red[0][0];
                    const double nfac = df2 * df2;
                    const double covj = fac * fac * nfac
                        * (red[1][0] + red[2][0]);
                    const double var = covj / (sums5 * sums5);
                    g.censig[i] = (var > 0) ? sqrt(var) : 0.0;
                    // now do the deferred center update
                    const double s0c = cen_sigma0;
                    const double sig = g.censig[i];
                    const double denom = s0c * s0c + sig * sig;
                    if (isfinite(sig) && denom != 0.0) {
                        const double kk = s0c * s0c / denom;
                        const double v = g.pos[i * 2];
                        const double u = g.pos[i * 2 + 1];
                        const double v0 = g.dpos[i * 2];
                        const double u0 = g.dpos[i * 2 + 1];
                        double newv = v + kk * g.cen_pull[i * 2]
                            + (1.0 - kk) * (v0 - v);
                        double newu = u
                            + kk * g.cen_pull[i * 2 + 1]
                            + (1.0 - kk) * (u0 - u);
                        const double clip = RECENTER_CLIP_FAC
                            * sqrt(Tsmooth);
                        double d0 = newv - v0;
                        double d1 = newu - u0;
                        const double nn = sqrt(d0 * d0 + d1 * d1);
                        if (nn > clip) {
                            d0 *= clip / nn;
                            d1 *= clip / nn;
                            newv = v0 + d0;
                            newu = u0 + d1;
                        }
                        double dmax = fabs(newv - v);
                        if (fabs(newu - u) > dmax)
                            dmax = fabs(newu - u);
                        g.pos[i * 2] = newv;
                        g.pos[i * 2 + 1] = newu;
                        const double Twt = g.sw[i * 3]
                            + g.sw[i * 3 + 2];
                        const double chc = dmax / sqrt(Twt);
                        if (chc > ch_class[2]) ch_class[2] = chc;
                        if (chc > win_max_l[i])
                            win_max_l[i] = chc;
                    }
                    ctrl &= ~2;
                }
                __syncthreads();
            }
        }

        // ---- thread 0: sweep-level machinery ----
        if (tid == 0) {
            // projected-residual convergence (flux, struct, cen)
            const double tols[3] = {flux_tol, tol, cen_tol};
            int conv = 1;
            for (int c = 0; c < 3; c++) {
                const double d = ch_class[c];
                if (d > 0.0) {
                    double rho = RHO_CAP;
                    if (hlen[c] > 0 && hprev[c] > 0.0
                        && d < hprev[c]) {
                        rho = d / hprev[c];
                        if (rho > RHO_CAP) rho = RHO_CAP;
                    }
                    if (d * rho / (1.0 - rho) >= tols[c])
                        conv = 0;
                }
                hprev[c] = d;
                hlen[c] = 1;
            }
            if (conv) {
                converged = 1;
                ctrl |= 1;
            }
            // noncontraction window: check against the OLD
            // baseline, then ALWAYS commit the new one;
            // have_prev drops only on intervention
            if (!(ctrl & 1) && ((it + 1) % NC_WINDOW == 0)) {
                int intervened = 0;
                if (have_prev) {
                    int worst = -1;
                    double wmax = -1.0;
                    for (int i2 = 0; i2 < nobj; i2++) {
                        if (g.otype[i2] == TSTAR) continue;
                        if (win_max_l[i2] > tol
                            && win_max_l[i2]
                               > NC_FAC * prev_win[i2]
                            && (win_nfail_l[i2]
                                + prev_nfail[i2]) > 0) {
                            if (win_max_l[i2] > wmax) {
                                wmax = win_max_l[i2];
                                worst = i2;
                            }
                        }
                    }
                    if (worst >= 0) {
                        intervened = 1;
                        // forced containment
                        const int i2 = worst;
                        g.nfail[i2] = 0;
                        g.sw[i2 * 3] = smoothcov;
                        g.sw[i2 * 3 + 1] = 0.0;
                        g.sw[i2 * 3 + 2] = smoothcov;
                        if (recenter) {
                            g.pos[i2 * 2] = g.dpos[i2 * 2];
                            g.pos[i2 * 2 + 1] =
                                g.dpos[i2 * 2 + 1];
                        }
                        if (g.nrestart[i2] == 0) {
                            g.nrestart[i2] = 1;
                            g.dbflags[i2] |= RESTARTED;
                            if (g.otype[i2] == TEXP) {
                                g.cov[i2 * 3] = 0.0;
                                g.cov[i2 * 3 + 1] = 0.0;
                                g.cov[i2 * 3 + 2] = 0.0;
                            } else {
                                g.cov[i2 * 3] = smoothcov;
                                g.cov[i2 * 3 + 1] = 0.0;
                                g.cov[i2 * 3 + 2] = smoothcov;
                            }
                        } else {
                            g.dbflags[i2] |= DEBLENDED_AS_PSF;
                            g.otype[i2] = TSTAR;
                            g.cov[i2 * 3] = smoothcov;
                            g.cov[i2 * 3 + 1] = 0.0;
                            g.cov[i2 * 3 + 2] = smoothcov;
                            have_scales = 0;
                        }
                        nhist = 0;
                        hprev[0] = -1; hprev[1] = -1;
                        hprev[2] = -1;
                        hlen[0] = 0; hlen[1] = 0; hlen[2] = 0;
                    }
                }
                for (int i2 = 0; i2 < nobj; i2++) {
                    prev_win[i2] = win_max_l[i2];
                    prev_nfail[i2] = win_nfail_l[i2];
                    win_max_l[i2] = 0.0;
                    win_nfail_l[i2] = 0;
                }
                have_prev = intervened ? 0 : 1;
            }
            // Steffensen extrapolation
            if (!(ctrl & 1)) {
                // shift history and append current packed state
                double* h0 = g.hist;
                double* h1 = g.hist + smax;
                double* h2 = g.hist + 2 * smax;
                int slen = 0;
                if (nhist == 3) {
                    for (int m = 0; m < smax; m++) h0[m] = h1[m];
                    for (int m = 0; m < smax; m++) h1[m] = h2[m];
                    nhist = 2;
                }
                double* dst = (nhist == 0) ? h0
                    : (nhist == 1) ? h1 : h2;
                slen = pack_state(g, nobj, recenter, dst,
                                  g.scales, have_scales == 0);
                have_scales = 1;
                nhist += 1;
                if (nhist == 3) {
                    double d1d1 = 0.0, d2d1 = 0.0;
                    for (int m = 0; m < slen; m++) {
                        const double e1 = h1[m] - h0[m];
                        const double e2 = h2[m] - h1[m];
                        d1d1 += e1 * e1;
                        d2d1 += e2 * e1;
                    }
                    const double rho = (d1d1 > 0.0)
                        ? d2d1 / d1d1 : 0.0;
                    if (rho > 0.2 && rho < 0.998) {
                        // save raw state for rollback
                        int k = 0;
                        for (int i2 = 0; i2 < nobj; i2++) {
                            g.saved[k++] = g.F[i2];
                            if (recenter) {
                                g.saved[k++] = g.pos[i2 * 2];
                                g.saved[k++] = g.pos[i2 * 2 + 1];
                            }
                            if (g.otype[i2] != TSTAR) {
                                g.saved[k++] = g.cov[i2 * 3];
                                g.saved[k++] = g.cov[i2 * 3 + 1];
                                g.saved[k++] = g.cov[i2 * 3 + 2];
                                g.saved[k++] = g.sw[i2 * 3];
                                g.saved[k++] = g.sw[i2 * 3 + 1];
                                g.saved[k++] = g.sw[i2 * 3 + 2];
                            }
                        }
                        int accepted = 0;
                        const double fracs[3] = {1.0, 0.5, 0.25};
                        for (int fi = 0; fi < 3; fi++) {
                            const double boost = fracs[fi] * rho
                                / (1.0 - rho);
                            for (int m = 0; m < slen; m++)
                                g.xtmp[m] = h2[m]
                                    + (h2[m] - h1[m]) * boost;
                            unpack_state(g, nobj, recenter,
                                         g.xtmp, g.scales);
                            if (state_valid(g, nobj, recenter,
                                            Tsmooth)) {
                                accepted = 1;
                                break;
                            }
                            // rollback
                            k = 0;
                            for (int i2 = 0; i2 < nobj; i2++) {
                                g.F[i2] = g.saved[k++];
                                if (recenter) {
                                    g.pos[i2 * 2] = g.saved[k++];
                                    g.pos[i2 * 2 + 1] =
                                        g.saved[k++];
                                }
                                if (g.otype[i2] != TSTAR) {
                                    g.cov[i2 * 3] = g.saved[k++];
                                    g.cov[i2 * 3 + 1] =
                                        g.saved[k++];
                                    g.cov[i2 * 3 + 2] =
                                        g.saved[k++];
                                    g.sw[i2 * 3] = g.saved[k++];
                                    g.sw[i2 * 3 + 1] =
                                        g.saved[k++];
                                    g.sw[i2 * 3 + 2] =
                                        g.saved[k++];
                                }
                            }
                        }
                        if (accepted) {
                            nhist = 0;
                            hprev[0] = -1; hprev[1] = -1;
                            hprev[2] = -1;
                            hlen[0] = 0; hlen[1] = 0;
                            hlen[2] = 0;
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (ctrl & 1) break;
        __syncthreads();
    }

    // ---- final-state measurement: the neighbor-corrected sums
    // and the error cross sums the CPU result path needs, saving
    // its two per-object mode passes ----
    for (int i = 0; i < nobj; i++) {
        if (tid == 0) {
            Swsh[0] = g.sw[i * 3];
            Swsh[1] = g.sw[i * 3 + 1];
            Swsh[2] = g.sw[i * 3 + 2];
            const double vi = g.pos[i * 2];
            const double ui = g.pos[i * 2 + 1];
            absh[0] = drow + a00 * vi + a10 * ui;
            absh[1] = dcol + a01 * vi + a11 * ui;
        }
        __syncthreads();
        const double w00 = Swsh[0], w01 = Swsh[1], w11 = Swsh[2];
        {
            const int half = (dim + 1) / 2;
            if (tid < 2) {
                const double facp = 2.0 * M_PI / dim * absh[tid];
                MODET* pre = (tid == 0) ? pyre : pxre;
                MODET* pim = (tid == 0) ? pyim : pxim;
                double s1, c1;
                sincos(facp, &s1, &c1);
                double pr = 1.0, pi = 0.0;
                pre[0] = 1.0; pim[0] = 0.0;
                for (int f = 1; f < half; f++) {
                    const double nr = pr * c1 - pi * s1;
                    pi = pr * s1 + pi * c1; pr = nr;
                    pre[f] = pr; pim[f] = pi;
                }
                pr = 1.0; pi = 0.0;
                for (int f = -1; f >= half - dim; f--) {
                    const double nr = pr * c1 + pi * s1;
                    pi = pi * c1 - pr * s1; pr = nr;
                    pre[f + dim] = pr; pim[f + dim] = pi;
                }
            }
        }
        __syncthreads();

        // moment-sum reduction at the final state
#ifdef MODE_FP32
        {
            const float w00f = (float)w00,
                        w01f = (float)w01,
                        w11f = (float)w11;
            float s0_ = 0, sv = 0, su = 0,
                  svv = 0, svu = 0, suu = 0;
            float k0_ = 0, kv_ = 0, ku_ = 0,
                  kvv = 0, kvu = 0, kuu = 0;
            for (long m = tid; m < nm; m += nt) {
                const float kvi = kv[m0 + m];
                const float kui = ku[m0 + m];
                const float Sv = w00f * kvi + w01f * kui;
                const float Su = w01f * kvi + w11f * kui;
                const float chi2 = kvi * Sv + kui * Su;
                if (chi2 > 25.0f || chi2 < 0.0f) continue;
                const float wk = fexpf_(-0.5f * chi2);
                const int y = iy[m0 + m];
                const int x = ix[m0 + m];
                const float pr = pyre[y] * pxre[x]
                    - pyim[y] * pxim[x];
                const float pi = pyre[y] * pxim[x]
                    + pyim[y] * pxre[x];
                const float2 val = kim[m0 + m];
                const float re = val.x * pr - val.y * pi;
                const float im = val.y * pr + val.x * pi;
                const float wre = wk * re;
                const float wim = wk * im;
                KAHAN(s0_, k0_, wre);
                KAHAN(sv, kv_, -(Sv * wim));
                KAHAN(su, ku_, -(Su * wim));
                KAHAN(svv, kvv, Sv * Sv * wre);
                KAHAN(svu, kvu, Sv * Su * wre);
                KAHAN(suu, kuu, Su * Su * wre);
            }
            red[0][tid] = (double)s0_ + (double)k0_;
            red[1][tid] = (double)sv + (double)kv_;
            red[2][tid] = (double)su + (double)ku_;
            red[3][tid] = (double)svv + (double)kvv;
            red[4][tid] = (double)svu + (double)kvu;
            red[5][tid] = (double)suu + (double)kuu;
        }
#else
        {
            double s0_ = 0, sv = 0, su = 0,
                   svv = 0, svu = 0, suu = 0;
            for (long m = tid; m < nm; m += nt) {
                const double kvi = kv[m0 + m];
                const double kui = ku[m0 + m];
                const double Sv = w00 * kvi + w01 * kui;
                const double Su = w01 * kvi + w11 * kui;
                const double chi2 = kvi * Sv + kui * Su;
                if (chi2 > 25.0 || chi2 < 0.0) continue;
                const double wk = fexp(-0.5 * chi2);
                const int y = iy[m0 + m];
                const int x = ix[m0 + m];
                const double pr = pyre[y] * pxre[x]
                    - pyim[y] * pxim[x];
                const double pi = pyre[y] * pxim[x]
                    + pyim[y] * pxre[x];
                const double2 val = kim[m0 + m];
                const double re = val.x * pr - val.y * pi;
                const double im = val.y * pr + val.x * pi;
                const double wre = wk * re;
                const double wim = wk * im;
                s0_ += wre;
                sv -= Sv * wim;
                su -= Su * wim;
                svv += Sv * Sv * wre;
                svu += Sv * Su * wre;
                suu += Su * Su * wre;
            }
            red[0][tid] = s0_; red[1][tid] = sv;
            red[2][tid] = su; red[3][tid] = svv;
            red[4][tid] = svu; red[5][tid] = suu;
        }
#endif
        __syncthreads();
        for (int st = nt / 2; st > 0; st >>= 1) {
            if (tid < st)
                for (int k = 0; k < 6; k++)
                    red[k][tid] += red[k][tid + st];
            __syncthreads();
        }
        if (tid == 0) {
            const double vi = g.pos[i * 2];
            const double ui = g.pos[i * 2 + 1];
            double base[6] = {0, 0, 0, 0, 0, 0};
            for (int j = 0; j < nobj; j++) {
                if (j == i) continue;
                model_sums(g, j,
                           g.pos[j * 2] - vi,
                           g.pos[j * 2 + 1] - ui,
                           Tsmooth, w00, w01, w11, base);
            }
            for (int c = 0; c < nfc; c++) {
                const double* fc = fcomp + (f0 + c) * 6;
                comp_sums(fc[0], fc[1], fc[2], fc[3],
                          fc[4] - vi, fc[5] - ui,
                          w00, w01, w11, base);
            }
            const double r0 = red[0][0], rv = red[1][0],
                         ru = red[2][0];
            const double rvv = red[3][0], rvu = red[4][0],
                         ruu = red[5][0];
            const double vv = w00 * r0 - rvv;
            const double vu = w01 * r0 - rvu;
            const double uu = w11 * r0 - ruu;
            double esums[6];
            esums[0] = rv * df2;
            esums[1] = ru * df2;
            esums[2] = (uu - vv) * df2;
            esums[3] = 2.0 * vu * df2;
            esums[4] = (uu + vv) * df2;
            esums[5] = r0 * df2;
            double* os = objsums_g + (o0 + i) * 6;
            for (int k = 0; k < 6; k++)
                os[k] = fac * (esums[k] - base[k] / detatinv);
        }
        __syncthreads();

        // error cross-sum reductions, chunks of <= 6 slots
        for (int c = 0; c < 3; c++) {
            const int cs = c * 6;
            const int ns = (13 - cs < 6) ? 13 - cs : 6;
#ifdef MODE_FP32
            {
                const float w00f = (float)w00,
                            w01f = (float)w01,
                            w11f = (float)w11;
                float acc[6], comp[6];
                for (int s = 0; s < 6; s++) {
                    acc[s] = 0.0f; comp[s] = 0.0f;
                }
                for (long m = tid; m < nm; m += nt) {
                    const float kvi = kv[m0 + m];
                    const float kui = ku[m0 + m];
                    const float Sv = w00f * kvi + w01f * kui;
                    const float Su = w01f * kvi + w11f * kui;
                    const float chi2 = kvi * Sv + kui * Su;
                    if (chi2 > 25.0f || chi2 < 0.0f) continue;
                    const float wk = fexpf_(-0.5f * chi2);
                    float kern[6];
                    kern[0] = Sv * wk;
                    kern[1] = Su * wk;
                    const float vvk = (w00f - Sv * Sv) * wk;
                    const float vuk = (w01f - Sv * Su) * wk;
                    const float uuk = (w11f - Su * Su) * wk;
                    kern[2] = uuk - vvk;
                    kern[3] = 2.0f * vuk;
                    kern[4] = uuk + vvk;
                    kern[5] = wk;
                    const float e2 = ef2[m0 + m];
                    for (int s = 0; s < ns; s++) {
                        KAHAN(acc[s], comp[s],
                              kern[CROSSA[cs + s]]
                              * kern[CROSSB[cs + s]] * e2);
                    }
                }
                for (int s = 0; s < 6; s++)
                    red[s][tid] = (s < ns)
                        ? (double)acc[s] + (double)comp[s] : 0.0;
            }
#else
            {
                double acc[6];
                for (int s = 0; s < 6; s++) acc[s] = 0.0;
                for (long m = tid; m < nm; m += nt) {
                    const double kvi = kv[m0 + m];
                    const double kui = ku[m0 + m];
                    const double Sv = w00 * kvi + w01 * kui;
                    const double Su = w01 * kvi + w11 * kui;
                    const double chi2 = kvi * Sv + kui * Su;
                    if (chi2 > 25.0 || chi2 < 0.0) continue;
                    const double wk = fexp(-0.5 * chi2);
                    double kern[6];
                    kern[0] = Sv * wk;
                    kern[1] = Su * wk;
                    const double vvk = (w00 - Sv * Sv) * wk;
                    const double vuk = (w01 - Sv * Su) * wk;
                    const double uuk = (w11 - Su * Su) * wk;
                    kern[2] = uuk - vvk;
                    kern[3] = 2.0 * vuk;
                    kern[4] = uuk + vvk;
                    kern[5] = wk;
                    const double e2 = ef2[m0 + m];
                    for (int s = 0; s < ns; s++)
                        acc[s] += kern[CROSSA[cs + s]]
                            * kern[CROSSB[cs + s]] * e2;
                }
                for (int s = 0; s < 6; s++)
                    red[s][tid] = (s < ns) ? acc[s] : 0.0;
            }
#endif
            __syncthreads();
            for (int st = nt / 2; st > 0; st >>= 1) {
                if (tid < st)
                    for (int k = 0; k < 6; k++)
                        red[k][tid] += red[k][tid + st];
                __syncthreads();
            }
            if (tid == 0) {
                double* oc = objcov_g + (o0 + i) * 13;
                for (int s = 0; s < ns; s++)
                    oc[cs + s] = red[s][0];
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        numiter_g[gid] = (it < maxiter) ? it + 1 : maxiter;
        converged_g[gid] = converged;
        nskip_g[gid] = nskip;
    }
}
'''


_KERNELS = {}


def get_kernel(fp32=False, nt=NT):
    nt = int(nt)
    assert nt >= 2 and (nt & (nt - 1)) == 0, 'nt must be power of 2'
    key = (bool(fp32), nt)
    if key not in _KERNELS:
        src = (KSRC
               .replace('EXPVALS_HERE', _EXPVALS)
               .replace('EXPVALSF_HERE', _EXPVALSF)
               .replace('EXPFRAC_HERE', _EXPFRAC)
               .replace('EXPCT_HERE', _EXPCT)
               .replace('NEXPC_H', str(NEXPC))
               .replace('NTHREADS_H', str(nt))
               .replace('DIM_MAX_H', str(DIM_MAX))
               .replace('GMAX_H', str(GMAX)))
        opts = ('-DMODE_FP32',) if key[0] else ()
        _KERNELS[key] = cp.RawKernel(
            src, 'deblend_groups', options=opts,
        )
    return _KERNELS[key]


def pack_host(debs):
    """pack constructed _Deblender instances into host numpy arrays
    (npz-serializable; convert with to_gpu, or use pack_groups)"""
    ng = len(debs)
    per = {k: [] for k in
           ['nobj', 'dim', 'df2', 'drow', 'dcol', 'jac', 'wt',
            'deti', 'tsm', 'tol', 'ftol', 'ctol', 'maxiter',
            'recenter', 'censig0']}
    modes = {k: [] for k in ['kim', 'iy', 'ix', 'kv', 'ku', 'ef2']}
    fixed = []
    state = {k: [] for k in
             ['otype', 'F', 'fscale', 'cov', 'sw', 'pos', 'dpos',
              'cpull', 'censig', 'nfail', 'nrestart', 'dbflags',
              'fixcen']}
    moff = [0]
    foff = [0]
    ooff = [0]
    soff = [0]

    for deb in debs:
        assert deb.nband == 1
        assert deb.e_sigma0 == 0.0
        eps = deb.epochs_per_obj[0]
        assert len(eps) == 1
        ep = eps[0]
        assert ep['dim'] <= DIM_MAX
        n = deb.nobj
        assert n <= GMAX

        per['nobj'].append(n)
        per['dim'].append(ep['dim'])
        per['df2'].append(ep['df2'])
        per['drow'].append(ep['drow'])
        per['dcol'].append(ep['dcol'])
        A = ep['Atinv']
        per['jac'].append([A[0, 0], A[0, 1], A[1, 0], A[1, 1]])
        per['wt'].append(ep['weight'])
        per['deti'].append(ep['detAtinv'])
        per['tsm'].append(deb.Tsmooth)
        per['tol'].append(deb.tol)
        per['ftol'].append(deb.flux_tol)
        per['ctol'].append(deb.cen_tol)
        per['maxiter'].append(deb.maxiter)
        per['recenter'].append(1 if deb.recenter else 0)
        per['censig0'].append(deb.cen_sigma0)

        modes['kim'].append(ep['kim'])
        modes['iy'].append(ep['iy'].astype(np.int32))
        modes['ix'].append(ep['ix'].astype(np.int32))
        modes['kv'].append(ep['kv'])
        modes['ku'].append(ep['ku'])
        modes['ef2'].append(ep['err_fac2'])
        moff.append(moff[-1] + ep['kim'].size)

        # fixed models: pre-expanded comps under any weight
        from ngmix.prepsfadmom.models import model_comps
        nf = 0
        for p, fm in zip(deb.fpositions, deb.fmodels):
            fracs, So00, So01, So11 = model_comps(fm, deb.Tsmooth)
            for c in range(fracs.size):
                fixed.append([
                    fm['F'][0] * fracs[c], So00[c], So01[c],
                    So11[c], p[0], p[1],
                ])
                nf += 1
        foff.append(foff[-1] + nf)

        for i in range(n):
            m = deb.models[i]
            state['otype'].append(TYPECODE[m['type']])
            state['F'].append(m['F'][0])
            state['fscale'].append(deb._fscales[i][0])
            c = m['cov_sm'] if m['type'] in ('gauss', 'star') \
                else m['cov']
            state['cov'].append([c[0, 0], c[0, 1], c[1, 1]])
            s = deb.Sw[i]
            state['sw'].append([s[0, 0], s[0, 1], s[1, 1]])
            state['pos'].append(list(deb.positions[i]))
            state['dpos'].append(list(deb.det_positions[i]))
            state['cpull'].append(list(deb.cen_pull[i]))
            state['censig'].append(deb._cen_sigma_sweep[i])
            state['nfail'].append(deb.nfail[i])
            state['nrestart'].append(deb.nrestart[i])
            state['dbflags'].append(deb.dbflags[i])
            state['fixcen'].append(1 if deb.fixcen[i] else 0)
        ooff.append(ooff[-1] + n)
        soff.append(soff[-1] + SPO * n)

    def hn(x, dt):
        return np.asarray(x, dtype=dt)

    return dict(
        ng=np.int64(ng),
        kim=hn(np.concatenate(modes['kim']), np.complex128),
        iy=hn(np.concatenate(modes['iy']), np.int32),
        ix=hn(np.concatenate(modes['ix']), np.int32),
        kv=hn(np.concatenate(modes['kv']), np.float64),
        ku=hn(np.concatenate(modes['ku']), np.float64),
        ef2=hn(np.concatenate(modes['ef2']), np.float64),
        moff=hn(moff, np.int64),
        fcomp=hn(np.array(fixed, dtype=np.float64).reshape(-1, 6)
                 if fixed else np.zeros((0, 6)), np.float64),
        foff=hn(foff, np.int64),
        nobj=hn(per['nobj'], np.int32),
        dim=hn(per['dim'], np.int32),
        df2=hn(per['df2'], np.float64),
        drow=hn(per['drow'], np.float64),
        dcol=hn(per['dcol'], np.float64),
        jac=hn(np.array(per['jac']).ravel(), np.float64),
        wt=hn(per['wt'], np.float64),
        deti=hn(per['deti'], np.float64),
        tsm=hn(per['tsm'], np.float64),
        tol=hn(per['tol'], np.float64),
        ftol=hn(per['ftol'], np.float64),
        ctol=hn(per['ctol'], np.float64),
        maxiter=hn(per['maxiter'], np.int32),
        recenter=hn(per['recenter'], np.int32),
        censig0=hn(per['censig0'], np.float64),
        otype=hn(state['otype'], np.int32),
        F=hn(state['F'], np.float64),
        fscale=hn(state['fscale'], np.float64),
        cov=hn(np.array(state['cov']).ravel(), np.float64),
        sw=hn(np.array(state['sw']).ravel(), np.float64),
        pos=hn(np.array(state['pos']).ravel(), np.float64),
        dpos=hn(np.array(state['dpos']).ravel(), np.float64),
        cpull=hn(np.array(state['cpull']).ravel(), np.float64),
        censig=hn(state['censig'], np.float64),
        nfail=hn(state['nfail'], np.int32),
        nrestart=hn(state['nrestart'], np.int32),
        dbflags=hn(state['dbflags'], np.int32),
        fixcen=hn(state['fixcen'], np.int32),
        ooff=hn(ooff, np.int64),
        soff=hn(soff, np.int64),
    )


MODE_FIELDS = ('kim', 'kv', 'ku', 'ef2', 'iy', 'ix')


def _mode_dtypes(fp32):
    kdt = np.complex64 if fp32 else np.complex128
    mdt = np.float32 if fp32 else np.float64
    return {'kim': kdt, 'kv': mdt, 'ku': mdt, 'ef2': mdt,
            'iy': np.int32, 'ix': np.int32}


def _alloc_scratch(packed, ns, ng, nobj_tot):
    packed['hist'] = cp.zeros(3 * ns, dtype=cp.float64)
    packed['scales'] = cp.zeros(ns, dtype=cp.float64)
    packed['saved'] = cp.zeros(ns, dtype=cp.float64)
    packed['xtmp'] = cp.zeros(ns, dtype=cp.float64)
    packed['objsums'] = cp.zeros(nobj_tot * 6, dtype=cp.float64)
    packed['objcov'] = cp.zeros(nobj_tot * 13, dtype=cp.float64)
    packed['numiter'] = cp.zeros(ng, dtype=cp.int32)
    packed['converged'] = cp.zeros(ng, dtype=cp.int32)
    packed['nskip'] = cp.zeros(ng, dtype=cp.int32)
    packed['err'] = cp.zeros(ng, dtype=cp.int32)


def to_gpu(h, fp32=False):
    """host pack -> device arrays + scratch/output allocations"""
    dts = _mode_dtypes(fp32)
    ng = int(h['ng'])
    packed = {'ng': ng, 'fp32': fp32}
    for k, v in h.items():
        if k == 'ng':
            continue
        if k in dts:
            packed[k] = cp.asarray(np.asarray(v, dtype=dts[k]))
        else:
            packed[k] = cp.asarray(v)
    _alloc_scratch(packed, int(h['soff'][-1]), ng,
                   int(h['ooff'][-1]))
    return packed


def to_gpu_multi(small_h, mode_list, fp32=False):
    """assemble a batch from a host-merged SMALL-array pack plus a
    list of per-submission mode-array dicts: the mode fields (the
    bulk of the bytes) are copied per submission straight into
    device slabs, no host merge"""
    dts = _mode_dtypes(fp32)
    ng = int(small_h['ng'])
    packed = {'ng': ng, 'fp32': fp32}
    nms = [int(m['kim'].size) for m in mode_list]
    nm_tot = sum(nms)
    for k, dt in dts.items():
        packed[k] = cp.empty(nm_tot, dtype=dt)
    off = 0
    for m, n in zip(mode_list, nms):
        for k, dt in dts.items():
            src = np.ascontiguousarray(np.asarray(m[k], dtype=dt))
            packed[k][off:off + n].set(src)
        off += n
    for k, v in small_h.items():
        if k == 'ng':
            continue
        packed[k] = cp.asarray(v)
    _alloc_scratch(packed, int(small_h['soff'][-1]), ng,
                   int(small_h['ooff'][-1]))
    return packed


def pack_groups(debs, fp32=False):
    """pack constructed _Deblender instances for the kernel"""
    return to_gpu(pack_host(debs), fp32=fp32)


def launch_gpu(packed, nt=NT):
    """launch the kernel and synchronize; no output fetch"""
    kern = get_kernel(fp32=packed.get('fp32', False), nt=nt)
    p = packed
    args = (
        p['kim'], p['iy'], p['ix'], p['kv'], p['ku'], p['ef2'],
        p['moff'], p['fcomp'], p['foff'],
        p['nobj'], p['dim'], p['df2'], p['drow'], p['dcol'],
        p['jac'], p['wt'], p['deti'], p['tsm'],
        p['tol'], p['ftol'], p['ctol'], p['maxiter'],
        p['recenter'], p['censig0'],
        p['otype'], p['F'], p['fscale'], p['cov'], p['sw'],
        p['pos'], p['dpos'], p['cpull'], p['censig'],
        p['nfail'], p['nrestart'], p['dbflags'], p['fixcen'],
        p['ooff'], p['hist'], p['scales'], p['saved'], p['xtmp'],
        p['soff'], p['objsums'], p['objcov'],
        p['numiter'], p['converged'], p['nskip'], p['err'],
    )
    kern((p['ng'],), (int(nt),), args)
    cp.cuda.Device().synchronize()


def fetch_out(packed):
    out = {}
    for k in ['otype', 'F', 'cov', 'sw', 'pos', 'cpull', 'dbflags',
              'nrestart', 'numiter', 'converged', 'nskip', 'err',
              'objsums', 'objcov']:
        out[k] = cp.asnumpy(packed[k])
    out['ooff'] = cp.asnumpy(packed['ooff'])
    return out


def run_gpu(packed, nt=NT):
    launch_gpu(packed, nt=nt)
    return fetch_out(packed)
