// Closed-form gaussian-mixture model sums and the deweight step.
#pragma once
#include "common.cuh"

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

// neighbor + fixed-model closed-form sums at the center of
// member i, accumulated into out (detAtinv=1)
__device__ void neighbor_fixed_sums(
    const Params& P, const GState& g, int i,
    const double* __restrict__ fcomp,
    double w00, double w01, double w11, double* out)
{
    const double vi = g.pos[i * 2];
    const double ui = g.pos[i * 2 + 1];
    for (int j = 0; j < P.nobj; j++) {
        if (j == i) continue;
        model_sums(g, j,
                   g.pos[j * 2] - vi,
                   g.pos[j * 2 + 1] - ui,
                   P.Tsmooth, w00, w01, w11, out);
    }
    for (int c = 0; c < P.nfc; c++) {
        const double* fc = fcomp + (P.f0 + c) * 6;
        comp_sums(fc[0], fc[1], fc[2], fc[3],
                  fc[4] - vi, fc[5] - ui,
                  w00, w01, w11, out);
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
