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
  reverts faint objects to exp smoothly; the width is the same
  on every rung, or scales with each rung's expected amplitude,
  see LADDER_PRIOR_MODE and ladder_prior_lambda), and a
  cross-band prior of width LADDER_TAUX tying the per-band
  fraction vectors (color-gradient freedom that reverts to
  shared structure at low s/n).
- sum(amps) is the wing-dominated model total flux and is never
  a reported quantity; the catalog totals come from derived
  functionals (see ladder_derived).

Names used throughout this module and the ladder parts of
full_errors, per group:

- idx: the deblender indices of the ladder objects; io indexes
  them (nlad of them), i the deblender's objects
- K: the number of rungs; Z = nlad K amp columns per band
- Sws[io], Tws[io]: the object's adaptive weight and its trace;
  aps[io]: its aperture weights, LADDER_AP_FACS times Sws[io]
- Fhat: (nlad, nband) per-band flux scales of the fraction units
- nap, nrows: apertures per object and rows per object (the
  apertures plus, with LADDER_MOMENT_ROWS, the T row)
- d, var: (nlad, nrows, nband) measured rows and their noise
  variances; wsum: (nlad, nband) epoch weight sums
- Mt: (nlad nrows, Z) unit template, the row predictions of
  every rung of every ladder object at unit amplitude
- amps: (nband, K) per object; new_full / amps_full: (nband, Z)
  for the whole group in flux units
"""
import numpy as np
from numba import njit

from ngmix.prepsfadmom.models import get_profile_comps, model_comps
from ngmix.prepsfadmom.models_nb import DET_REL_TOL
from ngmix.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import _fill_phasors

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

# prior widths in fraction units (see module docstring)
LADDER_TAU0 = 0.5
LADDER_TAUX = 0.2

# the shape of the profile prior's width across the rungs.
# 'multiplicative' (the default): rung k may deviate from the exp
# projection by tau sqrt(a0_k^2 + floor^2) in fraction units, a
# fraction of its own expected amplitude, with the floor keeping
# rungs whose projection is ~0 from being pinned; the data still
# move any rung whose rows are significant, and a rung without
# signal falls back to the profile.  'uniform': every rung may
# deviate by tau, i.e. by tau times the object's flux -- the outer
# rungs, whose exp-projection amplitude is a percent of the flux,
# are then nearly free in relative terms, and a bright object's
# wings have enough absolute freedom to absorb undetected light and
# detected companions (measured 2026-09-02: isolated totals +3-5
# percent from undetected neighbors; a 637-flux companion at 9.5 px
# from a 3x brighter ladder ends up in the neighbor's outer rungs).
# Floor 0.2 measured on 960 paired wldb fields: clean-object totals
# 1.033/1.016/1.010/1.001 -> 1.007/0.996/1.004/1.008 by s2n bin, the
# over-subtraction near detected neighbors halved, bulge-dominated
# objects (bt >= 0.6, ~4 percent) 3-5 points less complete (0.85/
# 0.86/0.91 vs 0.88/0.91/0.94); floor 0.1 gave the same gains with
# twice the bulge loss, 0.05 pushed a bright dev neighbor's real
# wing light into faint targets
LADDER_PRIOR_MODE = 'multiplicative'
LADDER_PRIOR_FLOOR = 0.2

# scene-wide amp solve cadence: first solve at sweep
# LADDER_WARMUP (early sweeps subtract the exp-profile init),
# then every LADDER_SOLVE_EVERY sweeps.  The carried last-change
# keeps convergence honest between solves.  2 matches the bdf
# split cadence: 4 defeats the projected-residual rho estimate
# (long carried plateaus) and destabilizes the extrapolation,
# while 1 lets the amps chase the weights into a limit cycle --
# the one-sweep lag is load-bearing (measured on the
# ladder+gauss pair test).  Measured dead end, do not revisit:
# an adaptive cadence that doubled the solve interval (up to 16
# sweeps) after solves whose model change was below 1e-3 cut the
# solves by 22 percent on a wldb field but lagged the amps
# behind the weights: sweeps +25 percent, converged 0.991 ->
# 0.928, the >= 300-sweep groups 6 -> 28
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
# n=3), and sum(amps) of the subtraction solve is never used.
# The fixed-aperture flux is the model flux sum under a round
# gaussian weight of LADDER_FIXED_FWHM in the smoothed plane,
# star-normalized (exact from the ladder mixture; the fixed
# aperture is uniform across objects and psf-model calibratable)
LADDER_TAU_TOTAL = 0.3
LADDER_FIXED_FWHM = 2.0
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

# indices into the moment-sum vector [v, u, M1, M2, T, flux]
T_ROW_INDEX = 4
_FLUX_INDEX = 5


def row_layout():
    """
    The row layout per ladder object.

    Returns (nap, nrows): the apertures per object and the rows per
    object, the apertures plus the T row when it is on.
    """
    nap = LADDER_AP_FACS.size
    return nap, nap + len(t_row_indices())


def t_row_indices():
    """
    The moment-sum indices of the consistency rows.

    In row order after the apertures: (T_ROW_INDEX,) with
    LADDER_MOMENT_ROWS, else empty.
    """
    return (T_ROW_INDEX,) if LADDER_MOMENT_ROWS else ()


# ---------------------------------------------------------------
# frame and rungs

def _frame_base(Sw, Tsmooth):
    """the eigenvalue-floored pre-smoothing base covariance"""
    sm = Tsmooth / 2
    Sb = np.asarray(Sw) - np.diag([sm, sm])
    evals, evecs = np.linalg.eigh(Sb)
    evals = np.maximum(evals, LADDER_BASE_FLOOR * sm)
    return (evecs * evals) @ evecs.T


def ladder_rung_covs(Sw, Tsmooth):
    """
    The smoothed rung covariances of the frame implied by a weight.

    Returns the arrays (S00, S01, S11) over the rungs.
    """
    sm = Tsmooth / 2
    Sb = _frame_base(Sw, Tsmooth)
    return (
        LADDER_RUNGS * Sb[0, 0] + sm,
        LADDER_RUNGS * Sb[0, 1],
        LADDER_RUNGS * Sb[1, 1] + sm,
    )


# ---------------------------------------------------------------
# closed-form moment sums of gaussian components under gaussian
# weights

@njit(cache=True)
def gauss_grid_sums(S00, S01, S11, W00, W01, W11, DV, DU, out):
    """
    Moment sums of every unit component under every weight, on a grid.

    The moment sums [v, u, M1, M2, T, flux] of every unit-flux
    component c (covariances S00, S01, S11, arrays of nc) under every
    weight r (W00, W01, W11, arrays of nr) at the per-pair offsets
    DV[r, c], DU[r, c], at detAtinv=1: out (nr, nc, 6).  The
    gauss_comps_ksums algebra on a weights-by-components grid, without
    any broadcasting on the caller's side; a non positive definite
    total covariance gives nan for that pair.
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


def unit_flux_sums(S00, S01, S11, W):
    """
    Flux sums of unit components under one weight at zero offset.

    S00, S01, S11 are arrays of n components; returns (n,).
    """
    S00 = np.atleast_1d(np.asarray(S00, dtype='f8'))
    zero = np.zeros((1, S00.size))
    return grid_sums(
        S00, S01, S11,
        np.array([W[0, 0]]), np.array([W[0, 1]]), np.array([W[1, 1]]),
        zero, zero,
    )[0, :, _FLUX_INDEX]


def band_comps(model, Tsmooth):
    """
    Per-band component fluxes and smoothed covariances of any model.

    Returns
    -------
    Fb, So00, So01, So11
        Fb has shape (nband, ncomp): the per-band component fluxes.
        For the standard types this is the outer product of the
        per-band flux with the fixed fractions; for a ladder it is
        the amplitude matrix itself.
    """
    if model['type'] == 'ladder':
        S00, S01, S11 = model['rungs']
        return model['amps'], S00, S01, S11
    fracs, S00, S01, S11 = model_comps(model, Tsmooth)
    return np.outer(model['F'], fracs), S00, S01, S11


def ladder_exp_fracs(rungs, Sw, Tsmooth):
    """
    The unit-total-flux exp profile expressed on the rungs.

    The prior center of the amp solve and the amp initialization.
    Closed form: match the exp model's aperture flux sums under the
    standard aperture set with a tiny ridge; the exp family
    covariance is the frame base of Sw.
    """
    sm = Tsmooth / 2
    Sb = _frame_base(Sw, Tsmooth)
    exp_comps = get_profile_comps('exp')
    exp_fracs = np.array([c[0] for c in exp_comps])
    exp_size_facs = np.array([c[1] for c in exp_comps])
    exp_S00 = exp_size_facs * Sb[0, 0] + sm
    exp_S01 = exp_size_facs * Sb[0, 1]
    exp_S11 = exp_size_facs * Sb[1, 1] + sm
    S00, S01, S11 = rungs

    K = LADDER_RUNGS.size
    Sw = np.asarray(Sw)
    nap = LADDER_AP_FACS.size
    # the rungs and the exp components in one grid call
    zero = np.zeros((nap, K + exp_fracs.size))
    flux_sums = grid_sums(
        np.concatenate([S00, exp_S00]), np.concatenate([S01, exp_S01]),
        np.concatenate([S11, exp_S11]),
        LADDER_AP_FACS * Sw[0, 0], LADDER_AP_FACS * Sw[0, 1],
        LADDER_AP_FACS * Sw[1, 1], zero, zero,
    )[..., _FLUX_INDEX]
    rung_sums = flux_sums[:, :K]                    # (nap, K)
    exp_sums = flux_sums[:, K:] @ exp_fracs         # (nap,)
    MtM = rung_sums.T @ rung_sums
    ridge = 1.0e-8 * np.trace(MtM) / K
    return np.linalg.solve(
        MtM + ridge * np.eye(K), rung_sums.T @ exp_sums,
    )


# ---------------------------------------------------------------
# the group's ladder context and row bookkeeping

def ladder_context(deb):
    """
    The ladder objects of a group and their per-object quantities.

    Returns (idx, aps, Sws, Tws, Fhat) at the current state, see the
    module docstring.
    """
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


def _others_comps(deb):
    """
    The non-ladder members and fixed externals as components.

    A list of (position, band_comps) pairs, subtracted from the data
    side of the rows.
    """
    others = []
    for j, m in enumerate(deb.models):
        if m['type'] != 'ladder':
            others.append(
                (deb.positions[j], band_comps(m, deb.Tsmooth)),
            )
    for p, fm in zip(deb.fpositions, deb.fmodels):
        others.append((p, band_comps(fm, deb.Tsmooth)))
    return others


def _row_weights(deb, idx, aps, Sws):
    """
    The row weights of the ladder system in row order.

    Per object its nap aperture covariances then, with the T row, its
    own weight.  Returns W00, W01, W11 (nlad nrows,), the row
    positions (nlad nrows, 2) and is_t_row (nlad nrows,).
    """
    nap, nrows = row_layout()
    nlad = len(idx)
    W = np.empty((nlad * nrows, 3))
    rpos = np.empty((nlad * nrows, 2))
    is_t_row = np.zeros(nlad * nrows, dtype=bool)
    for io, i in enumerate(idx):
        r0 = io * nrows
        for j, w in enumerate(aps[io]):
            W[r0 + j] = (w[0, 0], w[0, 1], w[1, 1])
        for r in range(nap, nrows):
            Sw = Sws[io]
            W[r0 + r] = (Sw[0, 0], Sw[0, 1], Sw[1, 1])
            is_t_row[r0 + r] = True
        rpos[r0:r0 + nrows] = deb.positions[i]
    return W[:, 0], W[:, 1], W[:, 2], rpos, is_t_row


def _row_values(sums, is_t_row):
    """
    Each row's own entry of a (nrows, ncomp, 6) grid of moment sums.

    The flux sum for an aperture row, the T sum for a T row; returns
    (nrows, ncomp).
    """
    return np.where(
        is_t_row[:, None], sums[:, :, T_ROW_INDEX],
        sums[:, :, _FLUX_INDEX],
    )


def _rung_comps(deb, idx):
    """
    All ladder objects' rungs as one component set.

    In column order (jo-major): S00, S01, S11 (nlad K,) and the
    component positions (nlad K, 2).
    """
    K = LADDER_RUNGS.size
    R = np.array([deb.models[i]['rungs'] for i in idx], dtype='f8')
    cpos = np.repeat(
        np.array([deb.positions[i] for i in idx], dtype='f8'), K, axis=0,
    )
    return R[:, 0, :].ravel(), R[:, 1, :].ravel(), R[:, 2, :].ravel(), cpos


# ---------------------------------------------------------------
# the measured rows

@njit(cache=True)
def ladder_apsums(kim, iy, ix, dim, alpha, beta, kv, ku,
                  w00, w01, w11, df2, err_fac2, need_var,
                  flux, sums1, var):
    """
    The fused ladder data pass over the modes of one epoch.

    For the adaptive weight (w00, w01, w11) at the centering phases
    (alpha, beta), one loop over the modes accumulates the flux sums
    under the eight dyadic apertures a = 0.25 ... 32 times the weight,
    the six moment sums under the a=1 weight itself (the T row), and
    with need_var the raw kernel cross sums for their noise variances
    (multiply by (fac df2)^2, as for admom_finalize).  Per mode the
    a=1 weight is one exact exponential; the others follow by two
    square roots and five squarings.

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
        output raw variance sums: the 8 aperture flux sums and the T
        sum (index 8); zeros when need_var is False
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


def _measure_object_rows(deb, i, Sw, need_var):
    """
    The rows of one ladder object from the fused pass over its epochs.

    Returns the per-band rows (nrows, nband) with the epoch factors
    applied, their noise variances (zeros unless need_var), the
    per-band epoch weight sums (nband,) and the raw per-epoch aperture
    flux sums (nap, nep) before the epoch factor, the linearization
    point of the full errors.
    """
    nap, nrows = row_layout()
    nband = deb.nband
    vi, ui = deb.positions[i]
    epochs = deb.epochs_per_obj[i]
    flux = np.zeros(nap)
    sums1 = np.zeros(6)
    var_raw = np.zeros(nap + 1)
    rows = np.zeros((nrows, nband))
    var = np.zeros((nrows, nband))
    wsum = np.zeros(nband)
    raw_flux = np.zeros((nap, len(epochs)))
    for iep, ep in enumerate(epochs):
        band = ep['band']
        fac = ep['weight'] * ep['detAtinv']
        wsum[band] += ep['weight']
        alpha, beta = get_phase_angles(
            ep, vi - ep['vcen'], ui - ep['ucen'],
        )
        ladder_apsums(
            ep['kim'], ep['iy'], ep['ix'], ep['dim'],
            alpha, beta, ep['kv'], ep['ku'],
            Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
            ep['err_fac2'], need_var, flux, sums1, var_raw,
        )
        raw_flux[:, iep] = flux
        rows[:nap, band] += fac * flux
        if nrows > nap:
            rows[nap, band] += fac * sums1[T_ROW_INDEX]
        if need_var:
            noise_fac = (fac * ep['df2']) ** 2
            var[:nap, band] += noise_fac * var_raw[:nap]
            if nrows > nap:
                var[nap, band] += noise_fac * var_raw[nap]
    return rows, var, wsum, raw_flux


def ladder_measure_rows(deb, idx, Sws, Tws, use_cache=True):
    """
    The measured rows of every ladder object, stacked.

    Returns d, var (nlad, nrows, nband), wsum (nlad, nband) and raw,
    the list of (nap, nep) raw aperture sums (see
    _measure_object_rows).

    With use_cache two per-object caches on the deblender apply: the
    variances only set the relative row weights, so they are reused
    until the weight has moved by more than 10 percent; and a member
    whose position and weight are bit-identical to its last
    measurement reuses its rows outright (they are data-only; the
    neighbor subtraction is applied afterwards), exact by
    construction.  Measured motion between solves is bimodal --
    frozen below 1e-12 or above 1e-3 in a grinding group -- so no
    tolerance would add hits without adding discontinuities against
    the deblender tolerance; the hit rate is 10 percent on small
    groups, 17-29 on large ones.
    """
    nap, nrows = row_layout()
    nband = deb.nband
    nlad = len(idx)
    sig_cache = None
    row_cache = None
    if use_cache:
        sig_cache = getattr(deb, '_ladder_sig_cache', None)
        if sig_cache is None:
            sig_cache = deb._ladder_sig_cache = {}
        row_cache = getattr(deb, '_ladder_row_cache', None)
        if row_cache is None:
            row_cache = deb._ladder_row_cache = {}
    d = np.zeros((nlad, nrows, nband))
    var = np.zeros((nlad, nrows, nband))
    wsum = np.zeros((nlad, nband))
    raw = []
    for io, i in enumerate(idx):
        Sw = Sws[io]
        sig_entry = sig_cache.get(i) if sig_cache is not None else None
        need_var = (
            sig_entry is None or sig_entry[1].shape != (nrows, nband)
            or abs(sig_entry[0] / Tws[io] - 1) > 0.1
        )
        vi, ui = deb.positions[i]
        state_key = (vi, ui, Sw[0, 0], Sw[0, 1], Sw[1, 1])
        row_entry = row_cache.get(i) if row_cache is not None else None
        if (
            row_entry is not None and row_entry[0] == state_key
            and not need_var
        ):
            d[io] = row_entry[1]
            wsum[io] = row_entry[2]
            var[io] = sig_entry[1]
            raw.append(row_entry[3])
            continue
        d[io], var_io, wsum[io], raw_flux = _measure_object_rows(
            deb, i, Sw, need_var,
        )
        if need_var:
            var[io] = var_io
            if sig_cache is not None:
                sig_cache[i] = (Tws[io], var_io.copy())
        else:
            var[io] = sig_entry[1]
        if row_cache is not None:
            row_cache[i] = (
                state_key, d[io].copy(), wsum[io].copy(), raw_flux,
            )
        raw.append(raw_flux)
    return d, var, wsum, raw


def ladder_subtract_others(deb, idx, aps, Sws, wsum, d):
    """
    Subtract the non-ladder members and fixed externals from the rows.

    In place on d.
    """
    nap, nrows = row_layout()
    others = _others_comps(deb)
    if not others or not idx:
        return
    nlad = len(idx)
    W00, W01, W11, rpos, is_t_row = _row_weights(deb, idx, aps, Sws)
    epoch_weights = np.repeat(wsum, nrows, axis=0)     # (nlad nrows, nband)
    for pos, (Fb, oS00, oS01, oS11) in others:
        DV = np.repeat(pos[0] - rpos[:, 0:1], oS00.size, axis=1)
        DU = np.repeat(pos[1] - rpos[:, 1:2], oS00.size, axis=1)
        vals = _row_values(
            grid_sums(oS00, oS01, oS11, W00, W01, W11, DV, DU), is_t_row,
        )                                          # (nlad nrows, ncomp)
        contrib = vals @ Fb.T                      # (nlad nrows, nband)
        d -= (epoch_weights * contrib).reshape(nlad, nrows, -1)


def ladder_template(deb, idx, aps, Sws):
    """
    The band-independent unit template matrix Mt.

    Shape (nlad nrows, nlad K): the row predictions of every rung of
    every ladder object at unit amplitude, in one grid-kernel
    evaluation.
    """
    if not idx:
        return np.zeros((0, 0))
    W00, W01, W11, rpos, is_t_row = _row_weights(deb, idx, aps, Sws)
    S00, S01, S11, cpos = _rung_comps(deb, idx)
    DV = cpos[None, :, 0] - rpos[:, None, 0]
    DU = cpos[None, :, 1] - rpos[:, None, 1]
    Mt = _row_values(
        grid_sums(S00, S01, S11, W00, W01, W11, DV, DU), is_t_row,
    )
    return np.ascontiguousarray(Mt)


def ladder_neighbor_unit_sums(deb, idx):
    """
    The neighbor-sum template d(NS_k)/d(amps_j).

    The (nobj, nlad, K, 6) unit rung sums of every ladder object's
    rungs under every object's weight at the relative offsets, zero
    on the object's own block.
    """
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


def ladder_rows(deb, use_cache):
    """
    Everything the joint solve needs at the current state.

    A dict with the context (idx, aps, Sws, Tws, Fhat), the measured
    rows d with the non-ladder members subtracted, var, wsum, raw (see
    ladder_measure_rows) and the unit template Mt.  The fit uses the
    caches; the derived functionals and the error setup measure
    afresh.
    """
    idx, aps, Sws, Tws, Fhat = ladder_context(deb)
    d, var, wsum, raw = ladder_measure_rows(
        deb, idx, Sws, Tws, use_cache=use_cache,
    )
    ladder_subtract_others(deb, idx, aps, Sws, wsum, d)
    return {
        'idx': idx, 'aps': aps, 'Sws': Sws, 'Tws': Tws, 'Fhat': Fhat,
        'd': d, 'var': var, 'wsum': wsum, 'raw': raw,
        'Mt': ladder_template(deb, idx, aps, Sws),
    }


# ---------------------------------------------------------------
# the joint amplitude solve

def ladder_prior(deb, idx, Sws, tol=0.0):
    """
    The stacked prior center: the exp profile on each object's rungs.

    Cached per object on the deblender against the weight it was
    computed for: reused when the weight is unchanged (tol=0, the
    error evaluations, where only a perturbed weight column changes
    an object's frame and the derivative must stay exact), or within
    tol relative (the fit, where a slightly stale prior center is
    harmless).
    """
    K = LADDER_RUNGS.size
    cache = getattr(deb, '_ladder_prior_cache', None)
    if cache is None:
        cache = deb._ladder_prior_cache = {}
    a0 = np.zeros(len(idx) * K)
    for io, i in enumerate(idx):
        Sw = np.asarray(Sws[io])
        entry = cache.get(i)
        hit = False
        if entry is not None:
            dSw = np.abs(Sw - entry[0]).max()
            if tol <= 0:
                hit = dSw == 0.0
            else:
                hit = dSw <= tol * (Sw[0, 0] + Sw[1, 1])
        if hit:
            a0_io = entry[1]
        else:
            a0_io = ladder_exp_fracs(
                deb.models[i]['rungs'], Sw, deb.Tsmooth,
            )
            cache[i] = (Sw.copy(), a0_io)
        a0[io * K:(io + 1) * K] = a0_io
    return a0


def ladder_prior_lambda(a0, tau0):
    """
    The per-column prior precisions and their derivative in the center.

    In fraction units for one band's Z columns, from the prior center
    a0 (see ladder_prior) at width tau0.  Returns (lam, dlam_da0):
    'uniform' mode has lam = 1 / tau0^2 everywhere and a zero
    derivative; 'multiplicative' mode has lam_k = 1 / (tau0^2 (a0_k^2
    + floor^2)), so the width scales with the rung's own expected
    amplitude, and dlam_k / da0_k = -2 a0_k lam_k / (a0_k^2 +
    floor^2).  See LADDER_PRIOR_MODE.
    """
    a0 = np.asarray(a0, dtype='f8')
    if LADDER_PRIOR_MODE == 'uniform':
        lam = np.full(a0.size, 1.0 / tau0 ** 2)
        return lam, np.zeros(a0.size)
    if LADDER_PRIOR_MODE != 'multiplicative':
        raise ValueError(f'bad LADDER_PRIOR_MODE {LADDER_PRIOR_MODE!r}')
    s2 = a0 ** 2 + LADDER_PRIOR_FLOOR ** 2
    lam = 1.0 / (tau0 ** 2 * s2)
    return lam, -2.0 * a0 * lam / s2


def ladder_assemble(deb, idx, var, wsum, Sws, Fhat, Mt, prior_tol=0.0):
    """
    The prior-width-independent solve pieces at the current state.

    Returns the (N, N) system matrix WITHOUT the lam0 diagonal (the
    data blocks and the cross-band coupling; a solve at prior width
    tau0 adds eye / tau0^2), and per band the noise-weighted,
    fraction-scaled template Mw (nlad nrows, Z), the row sigmas, the
    column scales, and the prior center a0 (see ladder_prior).
    """
    nap, nrows = row_layout()
    K = LADDER_RUNGS.size
    nband = deb.nband
    nlad = len(idx)
    Z = nlad * K
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
        epoch_weights = np.repeat(wsum[:, b], nrows)
        cs = np.repeat(Fhat[:, b], K)
        Mw = ((Mt * epoch_weights[:, None]) / sig[:, None]) * cs[None, :]
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
    """
    The solve from assembled pieces at prior width tau0.

    Returns the (nband, Z) amps in flux units, or None when the
    system is singular or the solution is not finite.
    """
    A, Mws, sigs, css, a0 = pieces
    nap, nrows = row_layout()
    K = LADDER_RUNGS.size
    nband = deb.nband
    nlad = len(idx)
    Z = nlad * K
    if tau0 is None:
        tau0 = LADDER_TAU0
    lam, _ = ladder_prior_lambda(a0, tau0)
    rhs = np.zeros(nband * Z)
    for b in range(nband):
        db = d[:, :, b].reshape(nlad * nrows) / sigs[b]
        rhs[b * Z:(b + 1) * Z] = Mws[b].T @ db + lam * a0
    try:
        X = np.linalg.solve(A + np.diag(np.tile(lam, nband)), rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(X)):
        return None
    new_full = np.empty((nband, Z))
    for b in range(nband):
        new_full[b] = X[b * Z:(b + 1) * Z] * css[b]
    return new_full


def ladder_solve_rows(deb, idx, d, var, wsum, Sws, Fhat, Mt,
                      tau0=None, prior_tol=0.0):
    """
    The regularized joint solve from the assembled rows.

    Returns the (nband, nlad K) amplitude matrix in flux units, or
    None when the solve is unsafe.
    """
    if not (
        np.all(np.isfinite(d)) and np.all(np.isfinite(Mt))
        and np.all(np.isfinite(var))
    ):
        return None
    pieces = ladder_assemble(
        deb, idx, var, wsum, Sws, Fhat, Mt, prior_tol=prior_tol,
    )
    return ladder_solve_pieces(deb, idx, d, pieces, tau0=tau0)


def ladder_change(deb, idx, Mt, wsum, Fhat, Tws, old_full, new_full):
    """
    The per-object change of the model's row predictions.

    Relative to the flux scale (times the weight T for the T row).
    The raw amp vectors carry prior-dominated degenerate directions
    that amplify ~1e-8 frame noise into ~1e-4 amp swings while the
    model as subtracted is unchanged; a raw-amp metric then never
    converges (measured on the two-ladder pair test).
    """
    nap, nrows = row_layout()
    nlad = len(idx)
    changes = np.zeros(nlad)
    for b in range(deb.nband):
        epoch_weights = np.repeat(wsum[:, b], nrows)
        row_change = (Mt * epoch_weights[:, None]) @ (
            new_full[b] - old_full[b]
        )
        for io in range(nlad):
            scale = wsum[io, b] * max(Fhat[io, b], 1.0e-30)
            if scale <= 0:
                continue
            r0 = io * nrows
            change = np.abs(row_change[r0:r0 + nap]).max() / scale
            if nrows > nap:
                change = max(change, np.abs(
                    row_change[r0 + nap:r0 + nrows]
                ).max() / (scale * Tws[io]))
            changes[io] = max(changes[io], change)
    return changes


def ladder_write_amps(deb, idx, new_full):
    """
    Store the (nband, Z) group amps on the models.
    """
    K = LADDER_RUNGS.size
    for io, i in enumerate(idx):
        deb.models[i]['amps'] = new_full[:, io * K:(io + 1) * K].copy()


def solve_group_amps(deb):
    """
    The scene-wide joint amplitude solve for the group's ladder objects.

    In place.  Rows are, per ladder object, the per-aperture, per-band
    measured flux sums and (with LADDER_MOMENT_ROWS) the T sum under
    the object's own adaptive weight, with non-ladder members and
    fixed externals subtracted in closed form, noise-weighted by the
    analytic aperture-sum sigmas; columns are all ladder objects'
    per-band fraction vectors with closed-form cross-object blocks;
    the priors are described in the module docstring.  Updates the
    models' amps and deb.ladder_last_da (see ladder_change) and
    returns the maximum change, or None when there are no ladder
    objects or a non-finite input made the solve unsafe (the amps are
    then left unchanged).
    """
    R = ladder_rows(deb, use_cache=True)
    idx = R['idx']
    if not idx:
        return None
    new_full = ladder_solve_rows(
        deb, idx, R['d'], R['var'], R['wsum'], R['Sws'], R['Fhat'],
        R['Mt'], prior_tol=1.0e-2,
    )
    if new_full is None:
        return None
    old_full = np.concatenate(
        [deb.models[i]['amps'] for i in idx], axis=1,
    )
    changes = ladder_change(
        deb, idx, R['Mt'], R['wsum'], R['Fhat'], R['Tws'],
        old_full, new_full,
    )
    for io, i in enumerate(idx):
        deb.ladder_last_da[i] = changes[io]
    ladder_write_amps(deb, idx, new_full)
    return changes.max()


# ---------------------------------------------------------------
# derived flux functionals

def ladder_fixed_weight(Tsmooth):
    """
    The fixed round aperture weight and its star normalization.

    Returns the weight W2 in the smoothed plane and the unit
    point-source (smoothing gaussian) flux sum under it.
    """
    from ngmix.moments import fwhm_to_T
    T2 = fwhm_to_T(LADDER_FIXED_FWHM)
    W2 = np.diag([T2 / 2, T2 / 2])
    sm = Tsmooth / 2
    s_star = unit_flux_sums([sm], [0.0], [sm], W2)[0]
    return W2, float(s_star)


def ladder_fixed_units(deb, idx, W2, s_star):
    """
    The star-normalized unit rung flux sums under the fixed aperture.

    For every ladder object, (nlad, K).
    """
    K = LADDER_RUNGS.size
    S00, S01, S11, _ = _rung_comps(deb, idx)
    return unit_flux_sums(S00, S01, S11, W2).reshape(len(idx), K) / s_star


def ladder_fixed_fluxes(deb, idx, amps_full, W2, s_star):
    """
    The star-normalized fixed-aperture fluxes of all ladder objects.

    Returns (nlad, nband) from the (nband, nlad K) amps.
    """
    K = LADDER_RUNGS.size
    nlad = len(idx)
    units = ladder_fixed_units(deb, idx, W2, s_star)     # (nlad, K)
    out = np.empty((nlad, deb.nband))
    for io in range(nlad):
        out[io] = amps_full[:, io * K:(io + 1) * K] @ units[io]
    return out


def ladder_fixed_flux(amps, rungs, W2, s_star):
    """
    The star-normalized fixed-aperture flux per band of one ladder model.

    Its flux sum under W2 at its own center divided by the
    point-source sum.
    """
    S00, S01, S11 = rungs
    return (amps @ unit_flux_sums(S00, S01, S11, W2)) / s_star


def ladder_total_var(d, var):
    """
    The row variances of the total-flux solve.

    The aperture rows beyond LADDER_TOTAL_MAX_AP are deweighted
    relative to noise and signal so the prior completes them (the row
    layout is shared with the subtraction solve, so the rows stay in
    place); var itself when there is no cap.  d, var: (nlad, nrows,
    nband).
    """
    if LADDER_TOTAL_MAX_AP is None:
        return var
    var = var.copy()
    drop = np.flatnonzero(LADDER_AP_FACS > LADDER_TOTAL_MAX_AP)
    var[:, drop, :] = 1.0e12 * (var[:, drop, :] + d[:, drop, :] ** 2)
    return var


def ladder_derived(deb):
    """
    The derived flux functionals of every ladder object.

    At the current (converged) state, as {i: {'total_flux',
    'fixed_flux'}} with (nband,) arrays: the total flux is sum(amps)
    of a second joint solve of the freshly measured rows with the
    LADDER_TAU_TOTAL prior (free core, prior-completed wings;
    sum(amps) of the subtraction solve itself is wing-dominated and
    never reported), the fixed flux is ladder_fixed_flux of the
    subtraction amps.  Nan totals when the total solve is unsafe.
    """
    R = ladder_rows(deb, use_cache=False)
    idx = R['idx']
    if not idx:
        return {}
    K = LADDER_RUNGS.size
    totals = ladder_solve_rows(
        deb, idx, R['d'], ladder_total_var(R['d'], R['var']), R['wsum'],
        R['Sws'], R['Fhat'], R['Mt'], tau0=LADDER_TAU_TOTAL,
    )
    W2, s_star = ladder_fixed_weight(deb.Tsmooth)
    out = {}
    for io, i in enumerate(idx):
        m = deb.models[i]
        out[i] = {
            'fixed_flux': ladder_fixed_flux(
                m['amps'], m['rungs'], W2, s_star,
            ),
            'total_flux': (
                totals[:, io * K:(io + 1) * K].sum(axis=1)
                if totals is not None else np.full(deb.nband, np.nan)
            ),
        }
    return out


def color_gradient(fixed_flux, flux):
    """
    The per adjacent-band-pair color gradient in magnitudes.

    The fixed-aperture color minus the adaptive-aperture color, nan
    where a flux is not positive.
    """
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
