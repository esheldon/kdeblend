// The thread-0 member update: closed-form neighbor subtraction,
// deweight, the per-type state updates with failure containment,
// and the regularized recentering.
#pragma once
#include "common.cuh"
#include "model.cuh"

// nfail-limit containment: reset the weight (and center), first
// time restart, second time demote to star; clears the
// extrapolation history
__device__ void contain_object(
    const Params& P, const GState& g, int i, SweepState& ss)
{
    g.nfail[i] = 0;
    g.sw[i * 3] = P.smoothcov;
    g.sw[i * 3 + 1] = 0.0;
    g.sw[i * 3 + 2] = P.smoothcov;
    if (P.recenter) {
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
            g.cov[i * 3] = P.smoothcov;
            g.cov[i * 3 + 1] = 0.0;
            g.cov[i * 3 + 2] = P.smoothcov;
        }
    } else {
        g.dbflags[i] |= DEBLENDED_AS_PSF;
        g.otype[i] = TSTAR;
        g.cov[i * 3] = P.smoothcov;
        g.cov[i * 3 + 1] = 0.0;
        g.cov[i * 3 + 2] = P.smoothcov;
        ss.have_scales = 0;
    }
    ss.nhist = 0;
    ss.hprev[0] = -1; ss.hprev[1] = -1; ss.hprev[2] = -1;
    ss.hlen[0] = 0; ss.hlen[1] = 0; ss.hlen[2] = 0;
}

// regularized recenter of object i toward the pull, clipped to
// the detection position; returns the change measure, or -1.0
// when censig is not usable (comparisons against it then no-op,
// matching the inline original's nan semantics)
__device__ double center_update(
    const Params& P, const GState& g, int i)
{
    const double s0c = P.cen_sigma0;
    const double sig = g.censig[i];
    const double denom = s0c * s0c + sig * sig;
    if (!(isfinite(sig) && denom != 0.0)) return -1.0;
    const double kk = s0c * s0c / denom;
    const double v = g.pos[i * 2];
    const double u = g.pos[i * 2 + 1];
    const double v0 = g.dpos[i * 2];
    const double u0 = g.dpos[i * 2 + 1];
    double newv = v + kk * g.cen_pull[i * 2]
        + (1.0 - kk) * (v0 - v);
    double newu = u + kk * g.cen_pull[i * 2 + 1]
        + (1.0 - kk) * (u0 - u);
    const double clip = RECENTER_CLIP_FAC * sqrt(P.Tsmooth);
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
    if (fabs(newu - u) > dmax) dmax = fabs(newu - u);
    g.pos[i * 2] = newv;
    g.pos[i * 2 + 1] = newu;
    const double Twt = g.sw[i * 3] + g.sw[i * 3 + 2];
    return dmax / sqrt(Twt);
}

// the member update for object i (thread 0 only), after the
// mode-sum reduction has landed in sh.red[.][0]
__device__ void member_update(
    const Params& P, const GState& g, int i, Shm& sh,
    const double* __restrict__ fcomp,
    double w00, double w01, double w11,
    double* ch_class, SweepState& ss, int* err)
{
    // neighbor + fixed closed-form sums, detAtinv=1
    double base[6] = {0, 0, 0, 0, 0, 0};
    neighbor_fixed_sums(P, g, i, fcomp, w00, w01, w11, base);
    // predicted sums for exp
    double psums[6] = {0, 0, 0, 0, 0, 0};
    const int mytype = g.otype[i];
    if (mytype == TEXP) {
        model_sums(g, i, 0.0, 0.0, P.Tsmooth,
                   w00, w01, w11, psums);
    }
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
    double sums[6];
    for (int k = 0; k < 6; k++)
        sums[k] = P.fac * (esums[k] - base[k] / P.detatinv);
    const double fs = sums[5];
    const double ws = P.weight;
    double pred[6] = {0, 0, 0, 0, 0, 0};
    double fs_pred = 0.0;
    if (mytype == TEXP) {
        for (int k = 0; k < 6; k++)
            pred[k] = P.fac * (psums[k] / P.detatinv);
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
        ss.nskip += 1;
        if (ss.nskip > 100 * P.nobj) {
            *err = 1;
            sh.ctrl = 1;
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
        ss.win_nfail[i] += 1;
        if (g.nfail[i] >= NFAIL_LIMIT)
            contain_object(P, g, i, ss);
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
                              P.Tsmooth)) {
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
            ss.nskip += 1;
            if (ss.nskip > 100 * P.nobj) {
                *err = 1;
                sh.ctrl = 1;
            }
            if (1.0 > ch_class[1]) ch_class[1] = 1.0;
            if (1.0 > change) change = 1.0;
            g.nfail[i] += 1;
            ss.win_nfail[i] += 1;
            if (g.nfail[i] >= NFAIL_LIMIT) {
                contained = 1;
                contain_object(P, g, i, ss);
            }
        } else if (idamp > 0) {
            if (1.0 > ch_class[1]) ch_class[1] = 1.0;
            if (1.0 > change) change = 1.0;
            g.cov[i * 3] = pr00;
            g.cov[i * 3 + 1] = pr01;
            g.cov[i * 3 + 2] = pr11;
            g.nfail[i] = 0;
            ss.win_nfail[i] += 1;
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
        // wt_cov moves to measured deweight unless the
        // containment intervened
        if (!contained) {
            g.sw[i * 3] = n00;
            g.sw[i * 3 + 1] = n01;
            g.sw[i * 3 + 2] = n11;
        }
    }

    // recenter (lazy pull noise -> request pass)
    if (P.recenter && sums[5] > 0 && !g.fixcen[i]) {
        if (g.censig[i] < 0) {
            // stash sums5; run the censig reduction
            sh.red[0][0] = sums[5];
            sh.ctrl |= 2;
        } else {
            const double chc = center_update(P, g, i);
            if (chc > ch_class[2]) ch_class[2] = chc;
            if (chc > change) change = chc;
        }
    }
    if (change > ss.win_max[i]) ss.win_max[i] = change;
}

// thread-0 tail of the censig pass: pull noise from the reduced
// lanes, then the deferred center update
__device__ void censig_finish(
    const Params& P, const GState& g, int i, Shm& sh,
    double* ch_class, SweepState& ss)
{
    const double sums5 = sh.red[0][0];
    const double nfac = P.df2 * P.df2;
    const double covj = P.fac * P.fac * nfac
        * (sh.red[1][0] + sh.red[2][0]);
    const double var = covj / (sums5 * sums5);
    g.censig[i] = (var > 0) ? sqrt(var) : 0.0;
    // now do the deferred center update
    const double chc = center_update(P, g, i);
    if (chc > ch_class[2]) ch_class[2] = chc;
    if (chc > ss.win_max[i]) ss.win_max[i] = chc;
    sh.ctrl &= ~2;
}
