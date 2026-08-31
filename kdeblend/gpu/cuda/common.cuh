// Shared definitions for the deblend_groups kernel:
// compile-time configuration, constant tables, fexp, and the
// state structs.
//
// The Python loader (kdeblend/gpu/_core.py) prepends a generated
// prologue defining NTHREADS_H, NEXPC_H, DIM_MAX_H, GMAX_H and
// the EXP*_H table values, then concatenates these files in
// fixed order, stripping the #include/#pragma lines (those exist
// for editor tooling only).
#pragma once

// editor/tooling fallback ONLY: real compiles always get these
// from the generated prologue, which defines them first
#ifndef NTHREADS_H
#define NTHREADS_H 256
#define NEXPC_H 6
#define DIM_MAX_H 512
#define GMAX_H 64
#define EXPVALS_H 0.0
#define EXPVALSF_H 0.0f
#define EXPFRAC_H 0.0
#define EXPCT_H 0.0
#endif

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

__constant__ double EXPLOOK[16] = {EXPVALS_H};
__constant__ float EXPLOOKF[16] = {EXPVALSF_H};

// the 13 unique nonzero entries of the 6x6 error cross-sum matrix
// (odd/center block 3 + even block 10; odd x even vanish by parity)
__constant__ int CROSSA[13] = {0,0,1,2,2,2,2,3,3,3,4,4,5};
__constant__ int CROSSB[13] = {0,1,1,2,3,4,5,3,4,5,4,5,5};
__constant__ double EXPFRAC[NEXPC_H] = {EXPFRAC_H};
__constant__ double EXPCT[NEXPC_H] = {EXPCT_H};

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

// per-group scalars, loaded once at kernel entry by every thread
struct Params {
    int nobj, dim, maxiter, recenter, nfc, smax;
    long m0, nm, f0, o0;
    double df2, drow, dcol;
    double a00, a01, a10, a11;
    double weight, detatinv, fac;
    double Tsmooth, smoothcov;
    double tol, flux_tol, cen_tol, cen_sigma0;
};

// thread-0 sweep-level fit state carried across sweeps
struct SweepState {
    double hprev[3];         // flux, struct, cen
    int hlen[3];
    double win_max[GMAX_H];
    double prev_win[GMAX_H];
    long win_nfail[GMAX_H];
    long prev_nfail[GMAX_H];
    int have_prev;
    int nskip;
    int nhist;               // extrapolation history length (0..3)
    int have_scales;
    int converged;
};

// the block's shared scratch
struct Shm {
    MODET pyre[DIM_MAX_H], pyim[DIM_MAX_H];
    MODET pxre[DIM_MAX_H], pxim[DIM_MAX_H];
    double red[6][NT_H];
    double Swsh[3];
    double absh[2];          // alpha, beta
    int ctrl;                // bit0: stop; bit1: censig pass
};
