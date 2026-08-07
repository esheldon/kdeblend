// Thread-0 sweep-level machinery: projected-residual
// convergence, the noncontraction window, and the Steffensen
// extrapolation with validity rollback.
#pragma once
#include "common.cuh"
#include "state.cuh"
#include "update.cuh"

// projected-residual convergence over the three change classes;
// commits this sweep's changes to the history and returns 1 when
// all classes project below tolerance
__device__ int sweep_converged(
    const Params& P, const double* ch_class, SweepState& ss)
{
    const double tols[3] = {P.flux_tol, P.tol, P.cen_tol};
    int conv = 1;
    for (int c = 0; c < 3; c++) {
        const double d = ch_class[c];
        if (d > 0.0) {
            double rho = RHO_CAP;
            if (ss.hlen[c] > 0 && ss.hprev[c] > 0.0
                && d < ss.hprev[c]) {
                rho = d / ss.hprev[c];
                if (rho > RHO_CAP) rho = RHO_CAP;
            }
            if (d * rho / (1.0 - rho) >= tols[c])
                conv = 0;
        }
        ss.hprev[c] = d;
        ss.hlen[c] = 1;
    }
    return conv;
}

// noncontraction window: check against the OLD baseline, then
// ALWAYS commit the new one; have_prev drops only on intervention
__device__ void noncontraction_check(
    const Params& P, const GState& g, SweepState& ss)
{
    int intervened = 0;
    if (ss.have_prev) {
        int worst = -1;
        double wmax = -1.0;
        for (int i2 = 0; i2 < P.nobj; i2++) {
            if (g.otype[i2] == TSTAR) continue;
            if (ss.win_max[i2] > P.tol
                && ss.win_max[i2]
                   > NC_FAC * ss.prev_win[i2]
                && (ss.win_nfail[i2]
                    + ss.prev_nfail[i2]) > 0) {
                if (ss.win_max[i2] > wmax) {
                    wmax = ss.win_max[i2];
                    worst = i2;
                }
            }
        }
        if (worst >= 0) {
            intervened = 1;
            // forced containment
            contain_object(P, g, worst, ss);
        }
    }
    for (int i2 = 0; i2 < P.nobj; i2++) {
        ss.prev_win[i2] = ss.win_max[i2];
        ss.prev_nfail[i2] = ss.win_nfail[i2];
        ss.win_max[i2] = 0.0;
        ss.win_nfail[i2] = 0;
    }
    ss.have_prev = intervened ? 0 : 1;
}

// Steffensen extrapolation: shift the packed-state history,
// append the current state, and when three points are held try
// the boosted step at decreasing fractions, rolling back any
// invalid result
__device__ void steffensen_step(
    const Params& P, const GState& g, SweepState& ss)
{
    double* h0 = g.hist;
    double* h1 = g.hist + P.smax;
    double* h2 = g.hist + 2 * P.smax;
    int slen = 0;
    if (ss.nhist == 3) {
        for (int m = 0; m < P.smax; m++) h0[m] = h1[m];
        for (int m = 0; m < P.smax; m++) h1[m] = h2[m];
        ss.nhist = 2;
    }
    double* dst = (ss.nhist == 0) ? h0
        : (ss.nhist == 1) ? h1 : h2;
    slen = pack_state(g, P.nobj, P.recenter, dst,
                      g.scales, ss.have_scales == 0);
    ss.have_scales = 1;
    ss.nhist += 1;
    if (ss.nhist == 3) {
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
            save_raw_state(g, P.nobj, P.recenter);
            int accepted = 0;
            const double fracs[3] = {1.0, 0.5, 0.25};
            for (int fi = 0; fi < 3; fi++) {
                const double boost = fracs[fi] * rho
                    / (1.0 - rho);
                for (int m = 0; m < slen; m++)
                    g.xtmp[m] = h2[m]
                        + (h2[m] - h1[m]) * boost;
                unpack_state(g, P.nobj, P.recenter,
                             g.xtmp, g.scales);
                if (state_valid(g, P.nobj, P.recenter,
                                P.Tsmooth)) {
                    accepted = 1;
                    break;
                }
                // rollback
                restore_raw_state(g, P.nobj, P.recenter);
            }
            if (accepted) {
                ss.nhist = 0;
                ss.hprev[0] = -1; ss.hprev[1] = -1;
                ss.hprev[2] = -1;
                ss.hlen[0] = 0; ss.hlen[1] = 0;
                ss.hlen[2] = 0;
            }
        }
    }
}
