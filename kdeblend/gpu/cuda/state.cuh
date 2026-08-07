// Global-state pack/unpack for the Steffensen extrapolation,
// validity checks, and the raw save/rollback buffers.
#pragma once
#include "common.cuh"
#include "model.cuh"

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

// save the raw (unnormalized) state to g.saved for rollback
__device__ void save_raw_state(
    const GState& g, int nobj, int recenter)
{
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
}

// restore the raw state from g.saved (mirror of save_raw_state)
__device__ void restore_raw_state(
    const GState& g, int nobj, int recenter)
{
    int k = 0;
    for (int i2 = 0; i2 < nobj; i2++) {
        g.F[i2] = g.saved[k++];
        if (recenter) {
            g.pos[i2 * 2] = g.saved[k++];
            g.pos[i2 * 2 + 1] = g.saved[k++];
        }
        if (g.otype[i2] != TSTAR) {
            g.cov[i2 * 3] = g.saved[k++];
            g.cov[i2 * 3 + 1] = g.saved[k++];
            g.cov[i2 * 3 + 2] = g.saved[k++];
            g.sw[i2 * 3] = g.saved[k++];
            g.sw[i2 * 3 + 1] = g.saved[k++];
            g.sw[i2 * 3 + 2] = g.saved[k++];
        }
    }
}
