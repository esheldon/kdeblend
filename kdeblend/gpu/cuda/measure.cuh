// Final-state measurement: the neighbor-corrected sums the CPU
// result path needs (its error cross-sum partner lives in
// reduce.cuh as cross_sum_partials).
#pragma once
#include "common.cuh"
#include "model.cuh"

// thread 0: neighbor-corrected final sums for object i, after
// the mode-sum reduction has landed in sh.red[.][0]
__device__ void final_object_sums(
    const Params& P, const GState& g, int i, Shm& sh,
    const double* __restrict__ fcomp,
    double w00, double w01, double w11,
    double* __restrict__ objsums_g)
{
    double base[6] = {0, 0, 0, 0, 0, 0};
    neighbor_fixed_sums(P, g, i, fcomp, w00, w01, w11, base);
    const double r0 = sh.red[0][0], rv = sh.red[1][0],
                 ru = sh.red[2][0];
    const double rvv = sh.red[3][0], rvu = sh.red[4][0],
                 ruu = sh.red[5][0];
    const double vv = w00 * r0 - rvv;
    const double vu = w01 * r0 - rvu;
    const double uu = w11 * r0 - ruu;
    double esums[6];
    esums[0] = rv * P.df2;
    esums[1] = ru * P.df2;
    esums[2] = (uu - vv) * P.df2;
    esums[3] = 2.0 * vu * P.df2;
    esums[4] = (uu + vv) * P.df2;
    esums[5] = r0 * P.df2;
    double* os = objsums_g + (P.o0 + i) * 6;
    for (int k = 0; k < 6; k++)
        os[k] = P.fac * (esums[k] - base[k] / P.detatinv);
}
