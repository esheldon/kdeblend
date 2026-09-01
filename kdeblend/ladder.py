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
  independent of separation.  The amps change only at the
  solve (a per-sweep rescale by the flux update was tried and
  drives overlapping ladders into a flux-trading limit cycle;
  it also made the amps depend on the flux history, a hidden
  state the fixed-point errors could not see).
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
from numba import njit

from ngmix.prepsfadmom.models import get_profile_comps
from ngmix.prepsfadmom.models_nb import gauss_comps_ksums, DET_REL_TOL
from ngmix.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import (
    admom_ksums, admom_finalize, _fill_phasors,
)
from ngmix.fastexp_nb import FASTEXP_MAX_CHI2  # noqa: F401

# rung sizes as multiples of the pre-smoothing frame covariance,
# geometric ratio 2.  Sub-smoothing rungs are omitted: after the
# common smoothing they are nearly degenerate with the smallest
# retained rung
LADDER_RUNGS = 0.2 * 2.0 ** np.arange(8)      # 0.2 ... 25.6

# measurement apertures as multiples of the adaptive weight:
# dyadic, so the fused kernel (ladder_apsums) gets every
# aperture's weight per mode from one exact exponential at the
# a=1 aperture, two square roots down and five squarings up (the
# squarings amplify relative error 2^n-fold, so the root must be
# the exact exponential, not the table one).  J=8 with K=8 rungs
# and the T row: measured no contamination loss against the
# log-spaced J=12 set.  Measured dead end: one table exponential
# per aperture instead of the chain is ~10 percent slower (the
# per-aperture branch costs more than the straight-line chain)
LADDER_AP_FACS = 0.25 * 2.0 ** np.arange(8)
_NAP = LADDER_AP_FACS.size

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

# solve gating: at a cadence point the solve is skipped when no
# component of the packed state (fluxes, centers, covariances
# and weights of every member -- everything the solve depends
# on; the amps are not packed) has moved by more than this, in
# the normalized packed units, since the last solve.  The
# grinding groups re-solve for hundreds of sweeps while the
# weights barely move; a final solve at the converged state
# keeps the reported amps exact
LADDER_GATE_TOL = 1.0e-4
# Measured dead end, do not revisit: an adaptive cadence that
# doubled the solve interval (up to 16 sweeps) after solves whose
# model change was below 1e-3 cut the solves by 22 percent on a
# wldb field but lagged the amps behind the weights: sweeps
# +25 percent, converged 0.991 -> 0.928, the >= 300-sweep
# groups 6 -> 28 (noisy pairs: 32 -> 75 and 24 -> 167 sweeps
# for the same answer).  The two-sweep lag is the most the
# coupled iteration tolerates, as the every-4 test also showed

# frame eigenvalue floor, as a fraction of Tsmooth/2: a noisy
# weight can drop below the smoothing; the rungs stay valid
LADDER_BASE_FLOOR = 0.1

# derived flux functionals (see ladder_derived): the total flux
# is sum(amps) of a second solve of the same rows with this
# tighter uniform prior toward the exp profile (free core,
# prior-completed wings: measured n=2 +0.0 +- 6.6 percent at
# s2n=30 vs exp -17.5, n=4 -7.7 +- 1.9 at s2n=100 vs exp -32).
# It is identically the adaptive-aperture flux times that
# model's aperture completion (the consistency row makes the
# model reproduce the measured aperture sum), i.e. an
# aperture-corrected adaptive flux whose completion comes from
# the fitted shape; the completion from the prior shape alone
# is biased low (-12 percent on an exp truth, -35 on sersic
# n=3), and sum(amps) of the subtraction solve is never used,
# and the fixed-aperture flux is the model flux sum under a
# round gaussian weight of this fwhm in the smoothed plane,
# star-normalized (exact from the ladder mixture; the fixed
# aperture is uniform across objects and psf-model calibratable)
LADDER_TAU_TOTAL = 0.3
# the largest aperture factor (of LADDER_AP_FACS) whose flux row
# informs the total-flux solve; None keeps all.  The outer
# apertures integrate ~100 arcsec^2 of whatever is not modelled
# (undetected galaxies, sky residuals): a uniform background of 2
# percent of the flux inside r=2 arcsec raises sum(amps) by 14-37
# percent through the a=32 row, measured noiselessly, and the
# wldb fields show +3-4 percent on isolated bright objects.  With
# the cap the prior (exp projection) completes beyond it: on 10
# wldb fields the cap at 4 (weight sigma twice the object's)
# takes the isolated excess to 0-2 percent (tracking the exp
# model within 2 at every size and recovering the largest
# galaxies where exp reads 0.96), cuts the low-s2n scatter 40
# percent and halves the blended excess; the unit check at s2n
# ~100 costs 1-3 percent on wide pure-exp profiles and 5 on n=4
# wings.  Noiseless rows never engage the prior, so with a cap
# the outer rungs become an ill-posed extrapolation there: the
# cap only means something with real noise
LADDER_TOTAL_MAX_AP = 4.0
LADDER_FIXED_FWHM = 2.0

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


@njit
def gauss_pairs_sums(So00, So01, So11, dv, du, sw00, sw01, sw11, out):
    """
    per-pair moment sums [v, u, M1, M2, T, flux] of unit-flux
    gaussian components So offset (dv, du) from the centers of
    weights sw, at detAtinv=1: the gauss_comps_ksums algebra
    without the accumulation, one weight per entry.  All inputs
    are flat arrays of equal length; out is (n, 6).  A non
    positive definite total covariance gives nan for that entry
    """
    n = So00.size
    nrm = 2 * np.pi
    for k in range(n):
        C00 = sw00[k] + So00[k]
        C01 = sw01[k] + So01[k]
        C11 = sw11[k] + So11[k]
        det = C00 * C11 - C01 * C01
        if not (C00 > 0.0 and det > DET_REL_TOL * C00 * C11):
            for j in range(6):
                out[k, j] = np.nan
            continue
        idet = 1.0 / det
        Ci00 = C11 * idet
        Ci01 = -C01 * idet
        Ci11 = C00 * idet
        Cd0 = Ci00 * dv[k] + Ci01 * du[k]
        Cd1 = Ci01 * dv[k] + Ci11 * du[k]
        sflux = np.exp(
            -0.5 * (dv[k] * Cd0 + du[k] * Cd1)
        ) / (nrm * np.sqrt(det))
        mu0 = sw00[k] * Cd0 + sw01[k] * Cd1
        mu1 = sw01[k] * Cd0 + sw11[k] * Cd1
        A00 = sw00[k] * Ci00 + sw01[k] * Ci01
        A01 = sw00[k] * Ci01 + sw01[k] * Ci11
        A10 = sw01[k] * Ci00 + sw11[k] * Ci01
        A11 = sw01[k] * Ci01 + sw11[k] * Ci11
        Sp00 = A00 * So00[k] + A01 * So01[k]
        Sp01 = A00 * So01[k] + A01 * So11[k]
        Sp11 = A10 * So01[k] + A11 * So11[k]
        vv = Sp00 + mu0 * mu0
        vu = Sp01 + mu0 * mu1
        uu = Sp11 + mu1 * mu1
        out[k, 0] = sflux * mu0
        out[k, 1] = sflux * mu1
        out[k, 2] = sflux * (uu - vv)
        out[k, 3] = sflux * 2 * vu
        out[k, 4] = sflux * (uu + vv)
        out[k, 5] = sflux


@njit
def gauss_grid_sums(S00, S01, S11, W00, W01, W11, DV, DU, out):
    """
    the moment sums [v, u, M1, M2, T, flux] of every unit-flux
    component c (covariances S00, S01, S11, arrays of nc) under
    every weight r (W00, W01, W11, arrays of nr) at the per-pair
    offsets DV[r, c], DU[r, c], at detAtinv=1: out (nr, nc, 6).
    The gauss_pairs_sums algebra on a weights-by-components grid,
    without any broadcasting on the caller's side
    """
    nr = W00.size
    nc = S00.size
    nrm = 2 * np.pi
    for r in range(nr):
        sw00 = W00[r]
        sw01 = W01[r]
        sw11 = W11[r]
        for c in range(nc):
            C00 = sw00 + S00[c]
            C01 = sw01 + S01[c]
            C11 = sw11 + S11[c]
            det = C00 * C11 - C01 * C01
            if not (C00 > 0.0 and det > DET_REL_TOL * C00 * C11):
                for j in range(6):
                    out[r, c, j] = np.nan
                continue
            dv = DV[r, c]
            du = DU[r, c]
            idet = 1.0 / det
            Ci00 = C11 * idet
            Ci01 = -C01 * idet
            Ci11 = C00 * idet
            Cd0 = Ci00 * dv + Ci01 * du
            Cd1 = Ci01 * dv + Ci11 * du
            sflux = np.exp(
                -0.5 * (dv * Cd0 + du * Cd1)
            ) / (nrm * np.sqrt(det))
            mu0 = sw00 * Cd0 + sw01 * Cd1
            mu1 = sw01 * Cd0 + sw11 * Cd1
            A00 = sw00 * Ci00 + sw01 * Ci01
            A01 = sw00 * Ci01 + sw01 * Ci11
            A10 = sw01 * Ci00 + sw11 * Ci01
            A11 = sw01 * Ci01 + sw11 * Ci11
            Sp00 = A00 * S00[c] + A01 * S01[c]
            Sp01 = A00 * S01[c] + A01 * S11[c]
            Sp11 = A10 * S01[c] + A11 * S11[c]
            vv = Sp00 + mu0 * mu0
            vu = Sp01 + mu0 * mu1
            uu = Sp11 + mu1 * mu1
            out[r, c, 0] = sflux * mu0
            out[r, c, 1] = sflux * mu1
            out[r, c, 2] = sflux * (uu - vv)
            out[r, c, 3] = sflux * 2 * vu
            out[r, c, 4] = sflux * (uu + vv)
            out[r, c, 5] = sflux


def grid_sums(S00, S01, S11, W00, W01, W11, DV, DU):
    """gauss_grid_sums with the output allocated: (nr, nc, 6)"""
    out = np.empty((W00.size, S00.size, 6))
    gauss_grid_sums(
        np.ascontiguousarray(S00, dtype='f8'),
        np.ascontiguousarray(S01, dtype='f8'),
        np.ascontiguousarray(S11, dtype='f8'),
        np.ascontiguousarray(W00, dtype='f8'),
        np.ascontiguousarray(W01, dtype='f8'),
        np.ascontiguousarray(W11, dtype='f8'),
        np.ascontiguousarray(DV, dtype='f8'),
        np.ascontiguousarray(DU, dtype='f8'), out,
    )
    return out


def pairs_sums(S00, S01, S11, dv, du, W00, W01, W11):
    """batched closed-form sums: broadcasts the inputs to a
    common shape and returns the (..., 6) sums"""
    arrs = np.broadcast_arrays(
        np.asarray(S00, dtype='f8'), np.asarray(S01, dtype='f8'),
        np.asarray(S11, dtype='f8'), np.asarray(dv, dtype='f8'),
        np.asarray(du, dtype='f8'), np.asarray(W00, dtype='f8'),
        np.asarray(W01, dtype='f8'), np.asarray(W11, dtype='f8'),
    )
    shape = arrs[0].shape
    flat = [np.ascontiguousarray(a).ravel() for a in arrs]
    out = np.empty((flat[0].size, 6))
    gauss_pairs_sums(*flat, out)
    return out.reshape(shape + (6,))


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

    K = LADDER_RUNGS.size
    Sw = np.asarray(Sw)
    nap = LADDER_AP_FACS.size
    # the rungs and the exp components in one grid call
    zero = np.zeros((nap, K + efr.size))
    ME = grid_sums(
        np.concatenate([S00, eS00]), np.concatenate([S01, eS01]),
        np.concatenate([S11, eS11]),
        LADDER_AP_FACS * Sw[0, 0], LADDER_AP_FACS * Sw[0, 1],
        LADDER_AP_FACS * Sw[1, 1], zero, zero,
    )[..., 5]
    M = ME[:, :K]
    d0 = ME[:, K:] @ efr
    MtM = M.T @ M
    lam = 1.0e-8 * np.trace(MtM) / K
    return np.linalg.solve(MtM + lam * np.eye(K), M.T @ d0)


def ladder_context(deb):
    """the ladder objects and their per-object aperture and
    frame quantities at the current state: (idx, aps, Sws, Tws,
    Fhat) with aps[io] the aperture weights, Sws[io] the
    adaptive weight, Tws[io] its trace and Fhat[io] the
    per-band flux scale of the fraction units"""
    idx = [
        i for i, m in enumerate(deb.models) if m['type'] == 'ladder'
    ]
    aps = []
    Sws = []
    Tws = np.zeros(len(idx))
    Fhat = np.zeros((len(idx), deb.nband))
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
    return idx, aps, Sws, Tws, Fhat


def ladder_others(deb):
    """the non-ladder members and fixed externals as
    (position, band_comps) pairs, subtracted from the data side"""
    others = []
    for j, m in enumerate(deb.models):
        if m['type'] != 'ladder':
            others.append(
                (deb.positions[j], band_comps(m, deb.Tsmooth)),
            )
    for p, fm in zip(deb.fpositions, deb.fmodels):
        others.append((p, band_comps(fm, deb.Tsmooth)))
    return others


def _nrows():
    nmom = len(_MOM_IDX) if LADDER_MOMENT_ROWS else 0
    return LADDER_AP_FACS.size, nmom, LADDER_AP_FACS.size + nmom


@njit
def ladder_apsums(kim, iy, ix, dim, alpha, beta, kv, ku,
                  w00, w01, w11, df2, err_fac2, need_var,
                  flux, sums1, var):
    """
    the fused ladder data pass: for the adaptive weight
    (w00, w01, w11) at the centering phases (alpha, beta), one
    loop over the modes accumulates the flux sums under the
    eight dyadic apertures a = 0.25 ... 32 times the weight, the
    six moment sums under the a=1 weight itself (the T row), and
    with need_var the raw kernel cross sums for their noise
    variances (multiply by (fac df2)^2, as for admom_finalize).
    Per mode the a=1 weight is one exact exponential; the others
    follow by two square roots and five squarings

    Parameters
    ----------
    kim, iy, ix, dim, alpha, beta, kv, ku, df2: as admom_ksums
    w00, w01, w11: float
        the adaptive weight covariance
    err_fac2: array
        the per-mode noise power factors (see admom_finalize)
    need_var: bool
        accumulate the variances
    flux: array of size 8
        output aperture flux sums, overwritten
    sums1: array of size 6
        output [v, u, M1, M2, T, flux] sums under the weight
    var: array of size 9
        output raw variance sums: the 8 aperture flux sums and
        the T sum (index 8); zeros when need_var is False
    """
    pyre = np.empty(dim)
    pyim = np.empty(dim)
    pxre = np.empty(dim)
    pxim = np.empty(dim)
    _fill_phasors(dim, alpha, pyre, pyim)
    _fill_phasors(dim, beta, pxre, pxim)

    nap = 8
    for j in range(8):
        flux[j] = 0.0
    for j in range(9):
        var[j] = 0.0
    s0 = 0.0
    sv = 0.0
    su = 0.0
    svv = 0.0
    svu = 0.0
    suu = 0.0

    for i in range(kim.size):
        kvi = kv[i]
        kui = ku[i]
        Sv = w00 * kvi + w01 * kui
        Su = w01 * kvi + w11 * kui
        chi2 = kvi * Sv + kui * Su
        if chi2 < 0:
            continue
        w1 = np.exp(-0.5 * chi2)

        y = iy[i]
        x = ix[i]
        pr = pyre[y] * pxre[x] - pyim[y] * pxim[x]
        pi = pyre[y] * pxim[x] + pyim[y] * pxre[x]
        val = kim[i]
        re = val.real * pr - val.imag * pi
        im = val.imag * pr + val.real * pi

        wre = w1 * re
        wim = w1 * im
        s0 += wre
        sv -= Sv * wim
        su -= Su * wim
        svv += Sv * Sv * wre
        svu += Sv * Su * wre
        suu += Su * Su * wre

        wh = np.sqrt(w1)
        wq = np.sqrt(wh)
        flux[0] += wq * re
        flux[1] += wh * re
        flux[2] += wre
        w2 = w1 * w1
        flux[3] += w2 * re
        w4 = w2 * w2
        flux[4] += w4 * re
        w8 = w4 * w4
        flux[5] += w8 * re
        w16 = w8 * w8
        flux[6] += w16 * re
        w32 = w16 * w16
        flux[7] += w32 * re

        if need_var:
            ef2 = err_fac2[i]
            var[0] += wh * ef2
            var[1] += w1 * ef2
            var[2] += w2 * ef2
            var[3] += w4 * ef2
            var[4] += w8 * ef2
            var[5] += w16 * ef2
            var[6] += w32 * ef2
            var[7] += w32 * w32 * ef2
            kern4 = ((w11 - Su * Su) + (w00 - Sv * Sv)) * w1
            var[8] += kern4 * kern4 * ef2

    vv = w00 * s0 - svv
    vu = w01 * s0 - svu
    uu = w11 * s0 - suu
    sums1[0] = sv * df2
    sums1[1] = su * df2
    sums1[2] = (uu - vv) * df2
    sums1[3] = 2 * vu * df2
    sums1[4] = (uu + vv) * df2
    sums1[5] = s0 * df2
    for j in range(nap):
        flux[j] *= df2


def ladder_measure_rows(deb, idx, aps, Sws, Tws, use_cache=True):
    """
    the measured rows: per ladder object, per aperture flux sum
    and (with LADDER_MOMENT_ROWS) the T row under the object's
    own weight, per band, with the epoch factors applied, plus
    their noise variances and the per-band epoch weight sums,
    all from one fused pass per epoch (ladder_apsums).  The
    variances only set the relative row weights, so with
    use_cache they are cached per object on the deblender and
    refreshed when the weight has moved by more than 10 percent.
    Also returns raw[io], the (nap, nep) per-epoch aperture flux
    sums before the epoch factor, for the linearization used by
    the full errors

    Returns
    -------
    d, var: (nlad, nrows, nband); wsum: (nlad, nband);
    raw: list of (nap, nep) arrays
    """
    nap, nmom, nrows = _nrows()
    if nmom and tuple(_MOM_IDX) != (4,):
        raise NotImplementedError(
            'the fused ladder pass provides the T row only'
        )
    nband = deb.nband
    nlad = len(idx)
    cache = None
    if use_cache:
        cache = getattr(deb, '_ladder_sig_cache', None)
        if cache is None:
            cache = deb._ladder_sig_cache = {}
    flux = np.zeros(nap)
    sums1 = np.zeros(6)
    vraw = np.zeros(nap + 1)
    d = np.zeros((nlad, nrows, nband))
    var = np.zeros((nlad, nrows, nband))
    wsum = np.zeros((nlad, nband))
    raw = []
    for io, i in enumerate(idx):
        vi, ui = deb.positions[i]
        Sw = Sws[io]
        epochs = deb.epochs_per_obj[i]
        rawi = np.zeros((nap, len(epochs)))
        ent = cache.get(i) if cache is not None else None
        need_var = (
            ent is None or ent[1].shape != (nrows, nband)
            or abs(ent[0] / Tws[io] - 1) > 0.1
        )
        for iep, ep in enumerate(epochs):
            band = ep['band']
            fac = ep['weight'] * ep['detAtinv']
            wsum[io, band] += ep['weight']
            alpha, beta = get_phase_angles(
                ep, vi - ep['vcen'], ui - ep['ucen'],
            )
            ladder_apsums(
                ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                alpha, beta, ep['kv'], ep['ku'],
                Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                ep['err_fac2'], need_var, flux, sums1, vraw,
            )
            rawi[:, iep] = flux
            d[io, :nap, band] += fac * flux
            if nmom:
                d[io, nap, band] += fac * sums1[4]
            if need_var:
                nf = (fac * ep['df2']) ** 2
                var[io, :nap, band] += nf * vraw[:nap]
                if nmom:
                    var[io, nap, band] += nf * vraw[nap]
        if need_var:
            if cache is not None:
                cache[i] = (Tws[io], var[io].copy())
        else:
            var[io] = ent[1]
        raw.append(rawi)
    return d, var, wsum, raw


def _row_weights(deb, idx, aps, Sws):
    """the row weights of the ladder system in row order: per
    object its nap aperture covariances then (with the T row)
    its own weight; returns W00, W01, W11 (nlad nrows,), the row
    positions (nlad nrows, 2) and the row kinds (nlad nrows,)
    with -1 for a flux row and the moment index for a T row"""
    nap, nmom, nrows = _nrows()
    nlad = len(idx)
    W = np.empty((nlad * nrows, 3))
    rpos = np.empty((nlad * nrows, 2))
    kind = np.full(nlad * nrows, -1, dtype=np.int64)
    for io, i in enumerate(idx):
        r0 = io * nrows
        for j, w in enumerate(aps[io]):
            W[r0 + j] = (w[0, 0], w[0, 1], w[1, 1])
        for r, a in enumerate(_MOM_IDX[:nmom]):
            Sw = Sws[io]
            W[r0 + nap + r] = (Sw[0, 0], Sw[0, 1], Sw[1, 1])
            kind[r0 + nap + r] = a
        rpos[r0:r0 + nrows] = deb.positions[i]
    return W[:, 0], W[:, 1], W[:, 2], rpos, kind


def _rung_comps(deb, idx):
    """all ladder objects' rungs as one component set in column
    order (jo-major): S00, S01, S11 (nlad K,) and the component
    positions (nlad K, 2)"""
    K = LADDER_RUNGS.size
    R = np.array([deb.models[i]['rungs'] for i in idx], dtype='f8')
    cpos = np.repeat(
        np.array([deb.positions[i] for i in idx], dtype='f8'), K, axis=0,
    )
    return R[:, 0, :].ravel(), R[:, 1, :].ravel(), R[:, 2, :].ravel(), cpos


def ladder_subtract_others(deb, idx, aps, Sws, wsum, d, others=None):
    """subtract the non-ladder members and fixed externals from
    the data rows, in place"""
    nap, nmom, nrows = _nrows()
    if others is None:
        others = ladder_others(deb)
    if not others or not idx:
        return
    nlad = len(idx)
    W00, W01, W11, rpos, kind = _row_weights(deb, idx, aps, Sws)
    nrt = W00.size
    rowidx = np.arange(nrt) % nrows
    for p, (Fb, oS00, oS01, oS11) in others:
        DV = np.repeat(p[0] - rpos[:, 0:1], oS00.size, axis=1)
        DU = np.repeat(p[1] - rpos[:, 1:2], oS00.size, axis=1)
        out = grid_sums(oS00, oS01, oS11, W00, W01, W11, DV, DU)
        # each row picks its own sum index: flux or its moment
        vals = np.where(
            kind[:, None] < 0, out[:, :, 5],
            out[np.arange(nrt), :, np.maximum(kind, 0)][:, :],
        )                                          # (nrt, ncomp)
        contrib = vals @ Fb.T                      # (nrt, nband)
        d -= (np.repeat(wsum, nrows, axis=0) * contrib).reshape(
            nlad, nrows, -1,
        )


def ladder_template(deb, idx, aps, Sws):
    """the band-independent unit template matrix
    (nlad nrows, nlad K): the row predictions of every rung of
    every ladder object at unit amplitude, in one grid-kernel
    evaluation"""
    nap, nmom, nrows = _nrows()
    K = LADDER_RUNGS.size
    nlad = len(idx)
    if nlad == 0:
        return np.zeros((0, 0))
    W00, W01, W11, rpos, kind = _row_weights(deb, idx, aps, Sws)
    S00, S01, S11, cpos = _rung_comps(deb, idx)
    DV = cpos[None, :, 0] - rpos[:, None, 0]
    DU = cpos[None, :, 1] - rpos[:, None, 1]
    out = grid_sums(S00, S01, S11, W00, W01, W11, DV, DU)
    nrt = W00.size
    Mt = np.where(
        kind[:, None] < 0, out[:, :, 5],
        out[np.arange(nrt), :, np.maximum(kind, 0)],
    )
    return np.ascontiguousarray(Mt)


def ladder_neighbor_unit_sums(deb, idx):
    """the (nobj, nlad, K, 6) unit rung sums of every ladder
    object's rungs under every object's weight at the relative
    offsets -- the neighbor-sum template d(NS_k)/d(amps_j),
    zero on the object's own block"""
    K = LADDER_RUNGS.size
    nobj = deb.nobj
    nlad = len(idx)
    if nlad == 0:
        return np.zeros((nobj, nlad, K, 6))
    pos_all = np.array(deb.positions, dtype='f8')
    Sw_all = np.array(
        [[sw[0, 0], sw[0, 1], sw[1, 1]] for sw in deb.Sw], dtype='f8',
    )
    S00, S01, S11, cpos = _rung_comps(deb, idx)
    DV = cpos[None, :, 0] - pos_all[:, None, 0]
    DU = cpos[None, :, 1] - pos_all[:, None, 1]
    U = grid_sums(
        S00, S01, S11, Sw_all[:, 0], Sw_all[:, 1], Sw_all[:, 2], DV, DU,
    ).reshape(nobj, nlad, K, 6)
    for jo, i in enumerate(idx):
        U[i, jo] = 0.0
    return U


def ladder_prior(deb, idx, Sws, tol=0.0):
    """the stacked prior center: the exp profile on each ladder
    object's current rungs.  Cached per object on the deblender
    against the weight it was computed for: reused when the
    weight is unchanged (tol=0, the error evaluations, where
    only a perturbed weight column changes an object's frame
    and the derivative must stay exact), or within tol
    relative (the fit, where a slightly stale prior center is
    harmless)"""
    K = LADDER_RUNGS.size
    cache = getattr(deb, '_ladder_prior_cache', None)
    if cache is None:
        cache = deb._ladder_prior_cache = {}
    a0 = np.zeros(len(idx) * K)
    for io, i in enumerate(idx):
        Sw = np.asarray(Sws[io])
        ent = cache.get(i)
        hit = False
        if ent is not None:
            dS = np.abs(Sw - ent[0]).max()
            if tol <= 0:
                hit = dS == 0.0
            else:
                hit = dS <= tol * (Sw[0, 0] + Sw[1, 1])
        if hit:
            a0i = ent[1]
        else:
            a0i = ladder_exp_fracs(
                deb.models[i]['rungs'], Sw, deb.Tsmooth,
            )
            cache[i] = (Sw.copy(), a0i)
        a0[io * K:(io + 1) * K] = a0i
    return a0


def ladder_assemble(deb, idx, var, wsum, Sws, Fhat, Mt, a0=None,
                    prior_tol=0.0):
    """the prior-width-independent solve pieces at the current
    state: the (N, N) system matrix WITHOUT the lam0 diagonal
    (the data blocks and the cross-band coupling; a solve at
    prior width tau0 adds eye / tau0^2), and per band the
    noise-weighted, fraction-scaled template Mw (nlad nrows, Z),
    the row sigmas, the column scales and the prior center a0
    (computed here unless given)"""
    nap, nmom, nrows = _nrows()
    K = LADDER_RUNGS.size
    nband = deb.nband
    nlad = len(idx)
    Z = nlad * K
    if a0 is None:
        a0 = ladder_prior(deb, idx, Sws, tol=prior_tol)
    lamx = 1.0 / LADDER_TAUX ** 2
    N = nband * Z
    A = np.zeros((N, N))
    eye = np.eye(Z)
    Mws = []
    sigs = []
    css = []
    for b in range(nband):
        sig = np.sqrt(var[:, :, b]).reshape(nlad * nrows)
        sig = np.where(sig > 0, sig, 1.0)
        rows = np.repeat(wsum[:, b], nrows)
        cs = np.repeat(Fhat[:, b], K)
        Mw = ((Mt * rows[:, None]) / sig[:, None]) * cs[None, :]
        sl = slice(b * Z, (b + 1) * Z)
        A[sl, sl] = Mw.T @ Mw + lamx * (nband - 1) * eye
        for b2 in range(nband):
            if b2 != b:
                A[sl, b2 * Z:(b2 + 1) * Z] = -lamx * eye
        Mws.append(Mw)
        sigs.append(sig)
        css.append(cs)
    return A, Mws, sigs, css, a0


def ladder_solve_pieces(deb, idx, d, pieces, tau0=None):
    """the solve from assembled pieces at prior width tau0;
    returns the (nband, Z) amps in flux units or None"""
    A, Mws, sigs, css, a0 = pieces
    nap, nmom, nrows = _nrows()
    K = LADDER_RUNGS.size
    nband = deb.nband
    nlad = len(idx)
    Z = nlad * K
    if tau0 is None:
        tau0 = LADDER_TAU0
    lam0 = 1.0 / tau0 ** 2
    rhs = np.zeros(nband * Z)
    for b in range(nband):
        db = d[:, :, b].reshape(nlad * nrows) / sigs[b]
        rhs[b * Z:(b + 1) * Z] = Mws[b].T @ db + lam0 * a0
    try:
        X = np.linalg.solve(A + lam0 * np.eye(nband * Z), rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(X)):
        return None
    new_full = np.empty((nband, Z))
    for b in range(nband):
        new_full[b] = X[b * Z:(b + 1) * Z] * css[b]
    return new_full


def ladder_solve_rows(deb, idx, d, var, wsum, Sws, Fhat, Mt,
                      tau0=None, a0=None, prior_tol=0.0):
    """the regularized joint solve from the assembled rows;
    returns the (nband, nlad K) amplitude matrix in flux units,
    or None when the solve is unsafe"""
    if not (
        np.all(np.isfinite(d)) and np.all(np.isfinite(Mt))
        and np.all(np.isfinite(var))
    ):
        return None
    pieces = ladder_assemble(
        deb, idx, var, wsum, Sws, Fhat, Mt, a0=a0,
        prior_tol=prior_tol,
    )
    return ladder_solve_pieces(deb, idx, d, pieces, tau0=tau0)


def ladder_change(deb, idx, Mt, wsum, Fhat, Tws, old_full, new_full):
    """the per-object change of the model's row predictions
    relative to the flux scale (times the weight T for the
    moment rows).  The raw amp vectors carry prior-dominated
    degenerate directions that amplify ~1e-8 frame noise into
    ~1e-4 amp swings while the model as subtracted is
    unchanged; a raw-amp metric then never converges (measured
    on the two-ladder pair test)"""
    nap, nmom, nrows = _nrows()
    nlad = len(idx)
    das = np.zeros(nlad)
    for b in range(deb.nband):
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
    return das


def ladder_write_amps(deb, idx, new_full):
    K = LADDER_RUNGS.size
    for io, i in enumerate(idx):
        deb.models[i]['amps'] = new_full[:, io * K:(io + 1) * K].copy()


def solve_group_amps(deb):
    """
    the scene-wide joint amplitude solve for all ladder objects
    of the group, in place.

    Rows are, per ladder object, the per-aperture, per-band
    measured flux sums and (with LADDER_MOMENT_ROWS) the T sum
    under the object's own adaptive weight, with non-ladder
    members and fixed externals subtracted in closed form,
    noise-weighted by the analytic aperture-sum sigmas; columns
    are all ladder objects' per-band fraction vectors with
    closed-form cross-object blocks; the priors are described in
    the module docstring.  Updates the models' amps and
    deb.ladder_last_da (see ladder_change) and returns the
    maximum change, or None when there are no ladder objects or
    a non-finite input made the solve unsafe (the amps are then
    left unchanged)
    """
    idx, aps, Sws, Tws, Fhat = ladder_context(deb)
    if not idx:
        return None
    d, var, wsum, _ = ladder_measure_rows(deb, idx, aps, Sws, Tws)
    ladder_subtract_others(deb, idx, aps, Sws, wsum, d)
    Mt = ladder_template(deb, idx, aps, Sws)
    new_full = ladder_solve_rows(
        deb, idx, d, var, wsum, Sws, Fhat, Mt, prior_tol=1.0e-2,
    )
    if new_full is None:
        return None
    old_full = np.concatenate(
        [deb.models[i]['amps'] for i in idx], axis=1,
    )
    das = ladder_change(
        deb, idx, Mt, wsum, Fhat, Tws, old_full, new_full,
    )
    for io, i in enumerate(idx):
        deb.ladder_last_da[i] = das[io]
    ladder_write_amps(deb, idx, new_full)
    return das.max()


def ladder_fixed_weight(Tsmooth):
    """the fixed round aperture weight in the smoothed plane and
    the unit point-source (smoothing gaussian) flux sum under
    it, the star normalization"""
    from ngmix.moments import fwhm_to_T
    T2 = fwhm_to_T(LADDER_FIXED_FWHM)
    W2 = np.diag([T2 / 2, T2 / 2])
    sm = Tsmooth / 2
    s_star = pairs_sums(sm, 0.0, sm, 0.0, 0.0, W2[0, 0], W2[0, 1],
                        W2[1, 1])[5]
    return W2, float(s_star)


def ladder_fixed_fluxes(deb, idx, amps_full, W2, s_star):
    """the star-normalized fixed-aperture fluxes of all ladder
    objects, (nlad, nband), from the (nband, nlad K) amps in one
    batched evaluation"""
    K = LADDER_RUNGS.size
    nlad = len(idx)
    u = ladder_fixed_units(deb, idx, W2, s_star)  # (nlad, K)
    out = np.empty((nlad, deb.nband))
    for io in range(nlad):
        out[io] = amps_full[:, io * K:(io + 1) * K] @ u[io]
    return out


def ladder_fixed_units(deb, idx, W2, s_star):
    """the star-normalized unit rung flux sums under the fixed
    aperture for every ladder object, (nlad, K)"""
    K = LADDER_RUNGS.size
    S00, S01, S11, _ = _rung_comps(deb, idx)
    zero = np.zeros((1, S00.size))
    u = grid_sums(
        S00, S01, S11, np.array([W2[0, 0]]), np.array([W2[0, 1]]),
        np.array([W2[1, 1]]), zero, zero,
    )[0, :, 5]
    return u.reshape(len(idx), K) / s_star


def ladder_fixed_flux(amps, rungs, W2, s_star):
    """the star-normalized fixed-aperture flux per band of a
    ladder model: its flux sum under W2 at its own center
    divided by the point-source sum"""
    S00, S01, S11 = rungs
    u = pairs_sums(S00, S01, S11, 0.0, 0.0, W2[0, 0], W2[0, 1],
                   W2[1, 1])[..., 5]
    return (amps @ u) / s_star


def ladder_total_var(d, var):
    """
    the row variances of the total-flux solve: the aperture rows
    beyond LADDER_TOTAL_MAX_AP deweighted relative to noise and
    signal so the prior completes them (the row layout is shared
    with the subtraction solve, so the rows stay in place); var
    itself when there is no cap.  d, var: (nlad, nrows, nband)
    """
    if LADDER_TOTAL_MAX_AP is None:
        return var
    var = var.copy()
    drop = np.flatnonzero(LADDER_AP_FACS > LADDER_TOTAL_MAX_AP)
    var[:, drop, :] = 1.0e12 * (var[:, drop, :] + d[:, drop, :] ** 2)
    return var


def ladder_derived(deb):
    """
    the derived flux functionals of every ladder object at the
    current (converged) state, as {i: {'total_flux', 'fixed_flux'}}
    with (nband,) arrays: the total flux is sum(amps) of a second
    joint solve of the freshly measured rows with the
    LADDER_TAU_TOTAL prior (free core, prior-completed wings;
    sum(amps) of the subtraction solve itself is wing-dominated
    and never reported), the fixed flux is ladder_fixed_flux of
    the subtraction amps.  None values when the total solve is
    unsafe
    """
    idx, aps, Sws, Tws, Fhat = ladder_context(deb)
    out = {}
    if not idx:
        return out
    K = LADDER_RUNGS.size
    d, var, wsum, _ = ladder_measure_rows(
        deb, idx, aps, Sws, Tws, use_cache=False,
    )
    ladder_subtract_others(deb, idx, aps, Sws, wsum, d)
    Mt = ladder_template(deb, idx, aps, Sws)
    tot = ladder_solve_rows(
        deb, idx, d, ladder_total_var(d, var), wsum, Sws, Fhat, Mt,
        tau0=LADDER_TAU_TOTAL,
    )
    W2, s_star = ladder_fixed_weight(deb.Tsmooth)
    for io, i in enumerate(idx):
        m = deb.models[i]
        ent = {
            'fixed_flux': ladder_fixed_flux(
                m['amps'], m['rungs'], W2, s_star,
            ),
            'total_flux': (
                tot[:, io * K:(io + 1) * K].sum(axis=1)
                if tot is not None else np.full(deb.nband, np.nan)
            ),
        }
        out[i] = ent
    return out


def color_gradient(fixed_flux, flux):
    """the per adjacent-band-pair color gradient in magnitudes:
    the fixed-aperture color minus the adaptive-aperture color,
    nan where a flux is not positive"""
    nband = flux.size
    out = np.full(max(nband - 1, 0), np.nan)
    for c in range(nband - 1):
        f2a, f2b = fixed_flux[c], fixed_flux[c + 1]
        fa, fb = flux[c], flux[c + 1]
        if f2a > 0 and f2b > 0 and fa > 0 and fb > 0:
            out[c] = -2.5 * (
                np.log10(f2a / f2b) - np.log10(fa / fb)
            )
    return out
