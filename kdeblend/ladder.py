"""
The free-amplitude concentric gaussian ladder model type.

A ladder object is a concentric gaussian mixture whose component
covariances ("rungs") are fixed multiples of the object's
adaptive frame -- the eigenvalue-floored pre-smoothing part of
its adaptive weight -- and whose per-band amplitudes are free,
fit by a scene-wide regularized linear solve.  The design and
its experimental evidence are recorded in the TODO ledger
(DONE 2026-08-31) and experiments/.

Structural properties:

- The weight/shape iteration is the data-driven gauss deweight
  path: the object's own amplitudes never enter its weight
  update, so the bdf-style size/profile feedback cannot occur.
- The amplitudes are solved jointly for all ladder objects of a
  group every LADDER_SOLVE_EVERY sweeps: the per-object
  Gauss-Seidel amp iteration contracts at rho -> 1 for close
  pairs (SPD, so slow rather than divergent) while the joint
  system's conditioning under the standard prior is nearly
  independent of separation.  Between solves the amp vectors
  are rescaled by the per-sweep flux updates (stale shape,
  fresh flux, like the bdf split).
- Priors, in fraction units (amps / per-band flux scale): a
  gaussian prior of width LADDER_TAU0 toward the exp profile
  expressed on the current rungs (with noise-weighted rows this
  reverts faint objects to exp smoothly), and a cross-band
  prior of width LADDER_TAUX tying the per-band fraction
  vectors (color-gradient freedom that reverts to shared
  structure at low s/n).
- sum(amps) is the wing-dominated model total flux and is never
  a reported quantity; the catalog totals come from derived
  functionals (see the TODO ledger).
"""
import numpy as np

from ngmix.prepsfadmom.models import get_profile_comps
from ngmix.prepsfadmom.models_nb import gauss_comps_ksums
from ngmix.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import (
    admom_ksums, admom_finalize,
)

# rung sizes as multiples of the pre-smoothing frame covariance,
# geometric ratio 2.  Sub-smoothing rungs are omitted: after the
# common smoothing they are nearly degenerate with the smallest
# retained rung
LADDER_RUNGS = 0.2 * 2.0 ** np.arange(8)      # 0.2 ... 25.6

# measurement apertures as multiples of the adaptive weight
LADDER_AP_FACS = np.geomspace(0.25, 32.0, 12)

# prior widths in fraction units (see module docstring)
LADDER_TAU0 = 0.5
LADDER_TAUX = 0.2

# scene-wide amp solve cadence: first solve at sweep
# LADDER_WARMUP (early sweeps subtract the exp-profile init),
# then every LADDER_SOLVE_EVERY sweeps.  The carried last-change
# keeps convergence honest between solves.  2 matches the bdf
# split cadence: 4 defeats the projected-residual rho estimate
# (long carried plateaus) and destabilizes the extrapolation,
# while 1 lets the amps chase the weights into a limit cycle --
# the one-sweep lag is load-bearing (measured on the
# ladder+gauss pair test)
LADDER_SOLVE_EVERY = 2
LADDER_WARMUP = 3

# frame eigenvalue floor, as a fraction of Tsmooth/2: a noisy
# weight can drop below the smoothing; the rungs stay valid
LADDER_BASE_FLOOR = 0.1

# consistency row: the T sum under each ladder object's own
# adaptive weight joins the solve as a soft noise-weighted row,
# so the model reproduces the weighted size the deweight step
# consumed (deweight(model sums) matches Sw in T).  The M1/M2
# rows are deliberately NOT included: every rung shares the
# frame's ellipticity, which is the gaussian-deweighted (hence
# biased: 0.079 for a true 0.10 on a sersic n=3) ellipticity,
# so no amplitude vector can satisfy them; forcing them only
# drags the least-squares compromise and corrupts the flux and
# T rows (measured: T residual 4e-5 -> 2e-4, flux 7e-6 ->
# 1e-3).  Ellipticity consistency needs a frame-ellipticity
# degree of freedom, a nonlinear follow-on
LADDER_MOMENT_ROWS = True
_MOM_IDX = (4,)

# chi2 cap for the numpy noise-variance pass, matching the
# fastexp cutoff regime (the tail contributes nothing)
_CHI2_CAP = 200.0


def _frame_base(Sw, Tsmooth):
    """the eigenvalue-floored pre-smoothing base covariance"""
    sm = Tsmooth / 2
    Sb = np.asarray(Sw) - np.diag([sm, sm])
    evals, evecs = np.linalg.eigh(Sb)
    evals = np.maximum(evals, LADDER_BASE_FLOOR * sm)
    return (evecs * evals) @ evecs.T


def ladder_rung_covs(Sw, Tsmooth):
    """the smoothed rung covariances (S00, S01, S11 arrays) for
    the frame implied by the weight Sw"""
    sm = Tsmooth / 2
    Sb = _frame_base(Sw, Tsmooth)
    return (
        LADDER_RUNGS * Sb[0, 0] + sm,
        LADDER_RUNGS * Sb[0, 1],
        LADDER_RUNGS * Sb[1, 1] + sm,
    )


def _comps_sums(F, S00, S01, S11, dv, du, W):
    """closed-form moment sums [v, u, M1, M2, T, flux] of
    gaussian components offset (dv, du) from the center of
    weight W, at detAtinv=1"""
    F = np.atleast_1d(np.asarray(F, dtype='f8'))
    n = F.size
    sums = np.zeros(6)
    gauss_comps_ksums(
        F,
        np.atleast_1d(np.asarray(S00, dtype='f8')),
        np.atleast_1d(np.asarray(S01, dtype='f8')),
        np.atleast_1d(np.asarray(S11, dtype='f8')),
        np.full(n, float(dv)), np.full(n, float(du)),
        W[0, 0], W[0, 1], W[1, 1], 1.0, sums,
    )
    return sums


def _comps_flux_sum(F, S00, S01, S11, dv, du, W):
    """the flux sum of _comps_sums"""
    return _comps_sums(F, S00, S01, S11, dv, du, W)[5]


def band_comps(model, Tsmooth):
    """
    per-band component fluxes and shared smoothed covariances of
    any model type.

    Returns
    -------
    Fb, So00, So01, So11
        Fb has shape (nband, ncomp): the per-band component
        fluxes.  For the standard types this is the outer
        product of the per-band flux with the fixed fractions;
        for a ladder it is the amplitude matrix itself.
    """
    if model['type'] == 'ladder':
        S00, S01, S11 = model['rungs']
        return model['amps'], S00, S01, S11
    from ngmix.prepsfadmom.models import model_comps
    fracs, S00, S01, S11 = model_comps(model, Tsmooth)
    return np.outer(model['F'], fracs), S00, S01, S11


def ladder_exp_fracs(rungs, Sw, Tsmooth):
    """
    the unit-total-flux exp profile expressed on the rungs: the
    prior center of the amp solve and the amp initialization.
    Closed form: match the exp model's aperture flux sums under
    the standard aperture set with a tiny ridge; the exp family
    covariance is the frame base of Sw
    """
    sm = Tsmooth / 2
    Sb = _frame_base(Sw, Tsmooth)
    comps = get_profile_comps('exp')
    efr = np.array([c[0] for c in comps])
    ecT = np.array([c[1] for c in comps])
    eS00 = ecT * Sb[0, 0] + sm
    eS01 = ecT * Sb[0, 1]
    eS11 = ecT * Sb[1, 1] + sm
    S00, S01, S11 = rungs

    nap = LADDER_AP_FACS.size
    K = LADDER_RUNGS.size
    M = np.zeros((nap, K))
    d0 = np.zeros(nap)
    Sw = np.asarray(Sw)
    for j, af in enumerate(LADDER_AP_FACS):
        W = af * Sw
        d0[j] = _comps_flux_sum(efr, eS00, eS01, eS11, 0.0, 0.0, W)
        for k in range(K):
            M[j, k] = _comps_flux_sum(
                1.0, S00[k], S01[k], S11[k], 0.0, 0.0, W,
            )
    MtM = M.T @ M
    lam = 1.0e-8 * np.trace(MtM) / K
    return np.linalg.solve(MtM + lam * np.eye(K), M.T @ d0)


def solve_group_amps(deb):
    """
    the scene-wide joint amplitude solve for all ladder objects
    of the group, in place.

    Rows are, per ladder object, the per-aperture, per-band
    measured flux sums and (with LADDER_MOMENT_ROWS) the T sum
    under the object's own adaptive weight, with
    non-ladder members and fixed externals subtracted in closed
    form, noise-weighted by the analytic aperture-sum sigmas;
    columns are all ladder objects' per-band fraction vectors
    with closed-form cross-object blocks; the priors are
    described in the module docstring.  Updates the models'
    amps and deb.ladder_last_da (the change of the model's row
    predictions relative to the flux scale, see below) and
    returns the maximum such change, or None when there are no
    ladder objects or a non-finite input made the solve unsafe
    (the amps are then left unchanged)
    """
    idx = [
        i for i, m in enumerate(deb.models) if m['type'] == 'ladder'
    ]
    if not idx:
        return None
    K = LADDER_RUNGS.size
    nap = LADDER_AP_FACS.size
    nmom = len(_MOM_IDX) if LADDER_MOMENT_ROWS else 0
    nrows = nap + nmom
    nband = deb.nband
    nlad = len(idx)
    Z = nlad * K

    others = []
    for j, m in enumerate(deb.models):
        if m['type'] != 'ladder':
            others.append(
                (deb.positions[j], band_comps(m, deb.Tsmooth)),
            )
    for p, fm in zip(deb.fpositions, deb.fmodels):
        others.append((p, band_comps(fm, deb.Tsmooth)))

    aps = []
    Sws = []
    Tws = np.zeros(nlad)
    Fhat = np.zeros((nlad, nband))
    for io, i in enumerate(idx):
        Sw = np.asarray(deb.Sw[i])
        Sws.append(Sw)
        Tws[io] = Sw[0, 0] + Sw[1, 1]
        aps.append([af * Sw for af in LADDER_AP_FACS])
        F = deb.models[i]['F']
        Fhat[io] = np.where(
            np.isfinite(F) & (np.abs(F) > 1.0e-12),
            np.abs(F), 1.0e-12,
        )

    # measured rows and their noise variances.  The variances
    # only set the relative row weights, so they are cached per
    # object and refreshed when the weight has moved by more
    # than 10 percent: the mode passes dominate the solve cost
    # otherwise
    cache = getattr(deb, '_ladder_sig_cache', None)
    if cache is None:
        cache = deb._ladder_sig_cache = {}
    esums = np.zeros(6)
    cov_raw = np.zeros((6, 6))
    d = np.zeros((nlad, nrows, nband))
    var = np.zeros((nlad, nrows, nband))
    wsum = np.zeros((nlad, nband))
    for io, i in enumerate(idx):
        vi, ui = deb.positions[i]
        Sw = Sws[io]
        ent = cache.get(i)
        need_var = (
            ent is None or ent[1].shape != (nrows, nband)
            or abs(ent[0] / Tws[io] - 1) > 0.1
        )
        for ep in deb.epochs_per_obj[i]:
            band = ep['band']
            fac = ep['weight'] * ep['detAtinv']
            wsum[io, band] += ep['weight']
            alpha, beta = get_phase_angles(
                ep, vi - ep['vcen'], ui - ep['ucen'],
            )
            for j in range(nap):
                W = aps[io][j]
                admom_ksums(
                    ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                    alpha, beta, ep['kv'], ep['ku'],
                    W[0, 0], W[0, 1], W[1, 1], ep['df2'],
                    esums,
                )
                d[io, j, band] += fac * esums[5]
                if need_var:
                    chi2 = (
                        ep['kv'] * (W[0, 0] * ep['kv']
                                    + W[0, 1] * ep['ku'])
                        + ep['ku'] * (W[0, 1] * ep['kv']
                                      + W[1, 1] * ep['ku'])
                    )
                    wk2 = np.exp(-np.minimum(chi2, _CHI2_CAP))
                    var[io, j, band] += (
                        (fac * ep['df2']) ** 2
                        * float(np.sum(wk2 * ep['err_fac2']))
                    )
            if nmom:
                if need_var:
                    admom_finalize(
                        ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                        alpha, beta, ep['kv'], ep['ku'],
                        Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                        ep['err_fac2'], esums, cov_raw,
                    )
                    for r, a in enumerate(_MOM_IDX):
                        var[io, nap + r, band] += (
                            (fac * ep['df2']) ** 2 * cov_raw[a, a]
                        )
                else:
                    admom_ksums(
                        ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                        alpha, beta, ep['kv'], ep['ku'],
                        Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                        esums,
                    )
                for r, a in enumerate(_MOM_IDX):
                    d[io, nap + r, band] += fac * esums[a]
        if need_var:
            cache[i] = (Tws[io], var[io].copy())
        else:
            var[io] = ent[1]
        # subtract the non-ladder members and fixed externals
        for p, (Fb, oS00, oS01, oS11) in others:
            dv = p[0] - vi
            du = p[1] - ui
            for band in range(nband):
                for j in range(nap):
                    d[io, j, band] -= wsum[io, band] * (
                        _comps_flux_sum(
                            Fb[band], oS00, oS01, oS11,
                            dv, du, aps[io][j],
                        )
                    )
                if nmom:
                    s = _comps_sums(
                        Fb[band], oS00, oS01, oS11, dv, du, Sw,
                    )
                    for r, a in enumerate(_MOM_IDX):
                        d[io, nap + r, band] -= wsum[io, band] * s[a]

    # band-independent unit template matrix
    Mt = np.zeros((nlad * nrows, Z))
    for io, i in enumerate(idx):
        vi, ui = deb.positions[i]
        r0 = io * nrows
        for jo, ip in enumerate(idx):
            dv = deb.positions[ip][0] - vi
            du = deb.positions[ip][1] - ui
            S00, S01, S11 = deb.models[ip]['rungs']
            for k in range(K):
                col = jo * K + k
                for j in range(nap):
                    Mt[r0 + j, col] = _comps_flux_sum(
                        1.0, S00[k], S01[k], S11[k], dv, du,
                        aps[io][j],
                    )
                if nmom:
                    s = _comps_sums(
                        1.0, S00[k], S01[k], S11[k], dv, du, Sws[io],
                    )
                    for r, a in enumerate(_MOM_IDX):
                        Mt[r0 + nap + r, col] = s[a]

    if not (
        np.all(np.isfinite(d)) and np.all(np.isfinite(Mt))
        and np.all(np.isfinite(var))
    ):
        return None

    # prior centers: exp profile on each object's current rungs
    a0 = np.zeros(Z)
    for io, i in enumerate(idx):
        a0[io * K:(io + 1) * K] = ladder_exp_fracs(
            deb.models[i]['rungs'], deb.Sw[i], deb.Tsmooth,
        )

    lam0 = 1.0 / LADDER_TAU0 ** 2
    lamx = 1.0 / LADDER_TAUX ** 2
    N = nband * Z
    A = np.zeros((N, N))
    rhs = np.zeros(N)
    eye = np.eye(Z)
    for b in range(nband):
        sig = np.sqrt(var[:, :, b]).reshape(nlad * nrows)
        sig = np.where(sig > 0, sig, 1.0)
        rows = np.repeat(wsum[:, b], nrows)  # model-side epoch scale
        Mb = Mt * rows[:, None]
        cs = np.repeat(Fhat[:, b], K)
        Mw = (Mb / sig[:, None]) * cs[None, :]
        db = d[:, :, b].reshape(nlad * nrows) / sig
        sl = slice(b * Z, (b + 1) * Z)
        A[sl, sl] = (
            Mw.T @ Mw + (lam0 + lamx * (nband - 1)) * eye
        )
        rhs[b * Z:(b + 1) * Z] = Mw.T @ db + lam0 * a0
        for b2 in range(nband):
            if b2 != b:
                A[sl, b2 * Z:(b2 + 1) * Z] = -lamx * eye

    try:
        X = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(X)):
        return None

    # the change metric lives in observable space: the change of
    # the model's row predictions, relative to the flux scale
    # (times the weight T for the moment rows).  The raw amp
    # vectors carry prior-dominated degenerate directions that
    # amplify ~1e-8 frame noise into ~1e-4 amp swings while the
    # model as subtracted is unchanged; a raw-amp metric then
    # never converges (measured on the two-ladder pair test)
    old_full = np.concatenate(
        [deb.models[i]['amps'] for i in idx], axis=1,
    )
    new_full = np.empty((nband, Z))
    for b in range(nband):
        new_full[b] = X[b * Z:(b + 1) * Z] * np.repeat(
            Fhat[:, b], K,
        )
    das = np.zeros(nlad)
    for b in range(nband):
        rows_b = np.repeat(wsum[:, b], nrows)
        rowchange = (Mt * rows_b[:, None]) @ (
            new_full[b] - old_full[b]
        )
        for io in range(nlad):
            den = wsum[io, b] * max(Fhat[io, b], 1.0e-30)
            if den <= 0:
                continue
            r0 = io * nrows
            da = np.abs(rowchange[r0:r0 + nap]).max() / den
            if nmom:
                da = max(da, np.abs(
                    rowchange[r0 + nap:r0 + nrows]
                ).max() / (den * Tws[io]))
            das[io] = max(das[io], da)
    for io, i in enumerate(idx):
        deb.ladder_last_da[i] = das[io]
        deb.models[i]['amps'] = new_full[
            :, io * K:(io + 1) * K
        ].copy()
    return das.max() if nlad else 0.0
