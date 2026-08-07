// Block-cooperative pieces: phasor tables, the mode-sum partial
// accumulations (fp64, and Kahan-compensated fp32), and the
// shared-memory tree reductions.
#pragma once
#include "common.cuh"

// thread 0: stage member i's weight and phase center into shared
__device__ void load_member_weight(
    const Params& P, const GState& g, int i, Shm& sh)
{
    sh.Swsh[0] = g.sw[i * 3];
    sh.Swsh[1] = g.sw[i * 3 + 1];
    sh.Swsh[2] = g.sw[i * 3 + 2];
    const double vi = g.pos[i * 2];
    const double ui = g.pos[i * 2 + 1];
    sh.absh[0] = P.drow + P.a00 * vi + P.a10 * ui;
    sh.absh[1] = P.dcol + P.a01 * vi + P.a11 * ui;
}

// row/col phasor tables (complex recurrence) for the center in
// sh.absh; thread 0 fills y, thread 1 fills x; caller syncs after
__device__ void phasor_tables(const Params& P, Shm& sh, int tid)
{
    const int half = (P.dim + 1) / 2;
    if (tid < 2) {
        const double facp = 2.0 * M_PI / P.dim * sh.absh[tid];
        MODET* pre = (tid == 0) ? sh.pyre : sh.pxre;
        MODET* pim = (tid == 0) ? sh.pyim : sh.pxim;
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
        for (int f = -1; f >= half - P.dim; f--) {
            const double nr = pr * c1 + pi * s1;
            pi = pi * c1 - pr * s1; pr = nr;
            pre[f + P.dim] = pr; pim[f + P.dim] = pi;
        }
    }
}

// weighted mode-moment partial sums into sh.red[0..5][tid];
// caller syncs then tree-reduces with block_reduce6
__device__ void mode_sum_partials(
    const Params& P, Shm& sh, int tid, int nt,
    const KIMT* __restrict__ kim,
    const int* __restrict__ iy, const int* __restrict__ ix,
    const MODET* __restrict__ kv, const MODET* __restrict__ ku,
    double w00, double w01, double w11)
{
#ifdef MODE_FP32
    {
        const float w00f = (float)w00,
                    w01f = (float)w01,
                    w11f = (float)w11;
        float s0_ = 0, sv = 0, su = 0,
              svv = 0, svu = 0, suu = 0;
        float k0_ = 0, kv_ = 0, ku_ = 0,
              kvv = 0, kvu = 0, kuu = 0;
        for (long m = tid; m < P.nm; m += nt) {
            const float kvi = kv[P.m0 + m];
            const float kui = ku[P.m0 + m];
            const float Sv = w00f * kvi + w01f * kui;
            const float Su = w01f * kvi + w11f * kui;
            const float chi2 = kvi * Sv + kui * Su;
            if (chi2 > 25.0f || chi2 < 0.0f) continue;
            const float wk = fexpf_(-0.5f * chi2);
            const int y = iy[P.m0 + m];
            const int x = ix[P.m0 + m];
            const float pr = sh.pyre[y] * sh.pxre[x]
                - sh.pyim[y] * sh.pxim[x];
            const float pi = sh.pyre[y] * sh.pxim[x]
                + sh.pyim[y] * sh.pxre[x];
            const float2 val = kim[P.m0 + m];
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
        sh.red[0][tid] = (double)s0_ + (double)k0_;
        sh.red[1][tid] = (double)sv + (double)kv_;
        sh.red[2][tid] = (double)su + (double)ku_;
        sh.red[3][tid] = (double)svv + (double)kvv;
        sh.red[4][tid] = (double)svu + (double)kvu;
        sh.red[5][tid] = (double)suu + (double)kuu;
    }
#else
    {
        double s0_ = 0, sv = 0, su = 0,
               svv = 0, svu = 0, suu = 0;
        for (long m = tid; m < P.nm; m += nt) {
            const double kvi = kv[P.m0 + m];
            const double kui = ku[P.m0 + m];
            const double Sv = w00 * kvi + w01 * kui;
            const double Su = w01 * kvi + w11 * kui;
            const double chi2 = kvi * Sv + kui * Su;
            if (chi2 > 25.0 || chi2 < 0.0) continue;
            const double wk = fexp(-0.5 * chi2);
            const int y = iy[P.m0 + m];
            const int x = ix[P.m0 + m];
            const double pr = sh.pyre[y] * sh.pxre[x]
                - sh.pyim[y] * sh.pxim[x];
            const double pi = sh.pyre[y] * sh.pxim[x]
                + sh.pyim[y] * sh.pxre[x];
            const double2 val = kim[P.m0 + m];
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
        sh.red[0][tid] = s0_; sh.red[1][tid] = sv;
        sh.red[2][tid] = su; sh.red[3][tid] = svv;
        sh.red[4][tid] = svu; sh.red[5][tid] = suu;
    }
#endif
}

// pull-noise partial sums at the (new) weight staged in sh.Swsh,
// into lanes 1 and 2; caller syncs then uses block_reduce2
__device__ void censig_partials(
    const Params& P, Shm& sh, int tid, int nt,
    const MODET* __restrict__ kv, const MODET* __restrict__ ku,
    const MODET* __restrict__ ef2)
{
#ifdef MODE_FP32
    const float u00 = (float)sh.Swsh[0],
                u01 = (float)sh.Swsh[1],
                u11 = (float)sh.Swsh[2];
    float c0 = 0, c1 = 0, cc0 = 0, cc1 = 0;
    for (long m = tid; m < P.nm; m += nt) {
        const float kvi = kv[P.m0 + m];
        const float kui = ku[P.m0 + m];
        const float Sv = u00 * kvi + u01 * kui;
        const float Su = u01 * kvi + u11 * kui;
        const float chi2 = kvi * Sv + kui * Su;
        if (chi2 > 25.0f || chi2 < 0.0f) continue;
        const float wk = fexpf_(-0.5f * chi2);
        const float e2 = ef2[P.m0 + m];
        KAHAN(c0, cc0, Sv * wk * Sv * wk * e2);
        KAHAN(c1, cc1, Su * wk * Su * wk * e2);
    }
    sh.red[1][tid] = (double)c0 + (double)cc0;
    sh.red[2][tid] = (double)c1 + (double)cc1;
#else
    const double u00 = sh.Swsh[0], u01 = sh.Swsh[1],
                 u11 = sh.Swsh[2];
    double c0 = 0, c1 = 0;
    for (long m = tid; m < P.nm; m += nt) {
        const double kvi = kv[P.m0 + m];
        const double kui = ku[P.m0 + m];
        const double Sv = u00 * kvi + u01 * kui;
        const double Su = u01 * kvi + u11 * kui;
        const double chi2 = kvi * Sv + kui * Su;
        if (chi2 > 25.0 || chi2 < 0.0) continue;
        const double wk = fexp(-0.5 * chi2);
        const double e2 = ef2[P.m0 + m];
        c0 += Sv * wk * Sv * wk * e2;
        c1 += Su * wk * Su * wk * e2;
    }
    sh.red[1][tid] = c0;
    sh.red[2][tid] = c1;
#endif
}

// error cross-sum partials for slots [cs, cs+ns) of the 13
// unique entries; caller syncs then tree-reduces
__device__ void cross_sum_partials(
    const Params& P, Shm& sh, int tid, int nt,
    const MODET* __restrict__ kv, const MODET* __restrict__ ku,
    const MODET* __restrict__ ef2,
    double w00, double w01, double w11, int cs, int ns)
{
#ifdef MODE_FP32
    {
        const float w00f = (float)w00,
                    w01f = (float)w01,
                    w11f = (float)w11;
        float acc[6], comp[6];
        for (int s = 0; s < 6; s++) {
            acc[s] = 0.0f; comp[s] = 0.0f;
        }
        for (long m = tid; m < P.nm; m += nt) {
            const float kvi = kv[P.m0 + m];
            const float kui = ku[P.m0 + m];
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
            const float e2 = ef2[P.m0 + m];
            for (int s = 0; s < ns; s++) {
                KAHAN(acc[s], comp[s],
                      kern[CROSSA[cs + s]]
                      * kern[CROSSB[cs + s]] * e2);
            }
        }
        for (int s = 0; s < 6; s++)
            sh.red[s][tid] = (s < ns)
                ? (double)acc[s] + (double)comp[s] : 0.0;
    }
#else
    {
        double acc[6];
        for (int s = 0; s < 6; s++) acc[s] = 0.0;
        for (long m = tid; m < P.nm; m += nt) {
            const double kvi = kv[P.m0 + m];
            const double kui = ku[P.m0 + m];
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
            const double e2 = ef2[P.m0 + m];
            for (int s = 0; s < ns; s++)
                acc[s] += kern[CROSSA[cs + s]]
                    * kern[CROSSB[cs + s]] * e2;
        }
        for (int s = 0; s < 6; s++)
            sh.red[s][tid] = (s < ns) ? acc[s] : 0.0;
    }
#endif
}

// tree-reduce sh.red lanes 0..5 across the block; the totals
// land in sh.red[k][0].  All threads must call this together.
__device__ void block_reduce6(Shm& sh, int tid, int nt)
{
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st)
            for (int k = 0; k < 6; k++)
                sh.red[k][tid] += sh.red[k][tid + st];
        __syncthreads();
    }
}

// tree-reduce lanes 1 and 2 only (the censig pass: lane 0 holds
// the stashed sums5 and must survive)
__device__ void block_reduce2(Shm& sh, int tid, int nt)
{
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) {
            sh.red[1][tid] += sh.red[1][tid + st];
            sh.red[2][tid] += sh.red[2][tid + st];
        }
        __syncthreads();
    }
}
