"""
Full (fixed-point) errors for the deblender.

The per-object error path evaluates the sandwich with the data
replaced by the converged model and treats subtracted neighbor
models as deterministic.  Both approximations bite: the
deterministic-neighbor assumption under-predicts blend-member
flux errors by 10-30 percent at 2 arcsec (2x for tight
recentered pairs) and T errors by 35 percent, and the
model-consistency substitution under-predicts T errors by ~12
percent for every object when the model is mismatched (real
morphologies fit with exp).  This module instead evaluates the
covariance of the coupled estimating equations at the converged
fixed point of the actual sweep map,

    Cov(x*) = (I - J)^-1 dphi/dS Cov(S) dphi/dS^T (I - J)^-T

using the ngmix.prepsfadmom.full_errors building blocks: the
data moment sums are linear in the prepped k-space image with
closed-form kernels, their covariance is exact via real-space
influence kernels (including the cross-member blocks on shared
pixels), and the sums are linearized about the solution with the
analytic kernel derivatives so the Jacobian evaluations never
touch the data modes.  Applied to every converged object,
singles included.  With recentering, the anchor (detection)
positions are a noisy input whose linear response is priced when
anchor_sigma is set.

Validated against finite-difference sandwiches, empirical refit
ensembles and field-scale FD probes (2026-07-29): fluxes,
colors (via flux_cov) and T/e1/e2 all calibrated at 0.95-1.06
across isolation bins where the per-object path misses by up to
2x, and calibrated under model mismatch (dev truth fit with
exp: per-object T_err low by 14 percent, full errors 0.95).

Objects of type gauss/exp/dev/star/ladder are treated (bdf falls back
to the per-object errors).  Star members take the per-object
micro-FD fallback for their update derivatives (the analytic
algebra covers the weight-adaptive types) and get the flux
entries only: flux_err, flux_cov and s2n gain the cross-member
response through shared pixels, the dominant blending term in
crowded stellar fields, while the structure entries are not
defined for a delta function.  Apodized preps are
handled exactly: the mask enters the influence kernels as a
pixel-space factor (see ngmix influence_kernels).
"""
import numpy as np

from ngmix.prepsfadmom.full_errors import (
    moment_kernels, dsums_dtheta, influence_kernels, sums_cov,
    sym3_mat as _sym3_mat,
    dw_derivs as _dw_derivs,
    _ingredients, WK_CHI2_MAX,
)
from ngmix.prepsfadmom.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums
from ngmix.prepsfadmom.models_nb import gauss_comps_ksums

import functools

from .ladder import (
    ladder_context, ladder_rows, ladder_subtract_others,
    ladder_template, ladder_write_amps, ladder_assemble,
    ladder_solve_pieces, ladder_neighbor_unit_sums, ladder_total_var,
    ladder_fixed_weight, ladder_fixed_fluxes, ladder_fixed_units,
    ladder_exp_fracs, ladder_rung_covs, unit_flux_sums,
    t_row_indices, LADDER_TAU0, LADDER_AP_FACS, LADDER_RUNGS,
    LADDER_TAU_TOTAL, ladder_prior_lambda,
)

SUPPORTED_TYPES = ('gauss', 'exp', 'dev', 'star', 'ladder')


class LadderResolveError(RuntimeError):
    """the ladder amp re-solve failed inside an error evaluation"""


# central difference steps: the packed state is normalized to
# O(1); the sum steps are scaled to the noise
FD_H = 1.0e-4
DS_FAC = 0.1

# deblender attributes an update evaluation can mutate; saved and
# restored around every Jacobian evaluation.  The enumeration is
# guarded by the restore-fidelity test
MUTABLE_ATTRS = (
    'models', 'wt_cov', 'positions', 'cen_pull', '_sweep_changes',
    '_fscales', '_win_max', '_win_nfail', '_prev_win_max',
    '_prev_win_nfail', '_change_hist', 'hist', '_boost_pre',
    '_boost_unprod', 'nskip', 'nfail',
    'nrestart', 'isweep', 'dbflags', 'bdf_info', 'bdf_last_dfd',
    '_cen_sigma_sweep', '_bdf_noise_cache', 'ladder_last_da',
    '_ladder_sig_cache', '_ladder_last_x', '_ladder_next',
    '_ladder_prior_cache', '_ladder_row_cache',
)


def apply_full_errors(deb, mbobs, res, anchor_sigma=0.0):
    """
    Replace a converged deblend's per-object errors with the full values.

    The full (fixed-point) flux and structure errors replace the
    per-object ones, and the cross-band flux covariance is added.

    Parameters
    ----------
    deb: _Deblender
        The converged deblender instance
    mbobs: ngmix.MultiBandObsList
        The observations the deblender was built from, for the
        weight maps
    res: dict
        The deblend result, modified in place: each object gains
        flux_cov (nband, nband) and has flux_err, s2n, T_err,
        e1_err and e2_err replaced; s2n is the covariance-aware
        joint value sqrt(F^T C^-1 F) over the usable bands
        (quadrature fallback for a non positive definite
        block).  The gauss-estimator entries
        are replaced too: gauss_T_err, gauss_e1_err and
        gauss_e2_err from the weight (wt_cov) rows of the state
        covariance, and gauss_flux, gauss_flux_err, gauss_s2n
        plus the new gauss_flux_cov from the flux response.
        Star members get the flux entries only (flux_err,
        flux_cov, s2n): no structure or gauss entries exist for
        a delta function, but the fluxes gain the cross-member
        response through shared pixels that the per-object path
        treats as deterministic.
        Singles are treated too: the m=1 machinery solves the
        same estimating equations but differentiates the actual
        update map at the actual data, with no model-consistency
        substitution, so unlike the per-object model_sandwich
        errors it stays calibrated under model mismatch (dev
        truth fit with exp: per-object T_err low by 14 percent,
        full errors 0.95).  For the gauss estimator every
        non-gaussian profile is mismatched, so the delta-method
        errors are low on all real galaxies (PAdmomFitter MC:
        T_err 13 percent low on exp truth, 32 on dev; full
        errors 1.00-1.02)
    anchor_sigma: float or array, optional
        With recentering, the noise of the anchor (detection)
        positions: a scalar sigma in arcsec (isotropic, shared),
        an (nobj,) array of per-object sigmas, or an
        (nobj, 2, 2) array of per-object position covariances in
        arcsec^2 with (v, u) ordering (e.g. from the sep centroid
        error moments).  The linear anchor response is added to
        the covariance.  0 (default) leaves the errors
        conditional on the anchors

    Returns
    -------
    True when the full errors were applied, False when the
    deblend is not eligible (unconverged, unsupported member
    types) and the per-object errors were left in place
    """
    if not res.get('converged', False):
        return False
    for m in deb.models:
        if m['type'] not in SUPPORTED_TYPES:
            return False

    try:
        cov, slices, extras = full_covariance(
            deb, mbobs, anchor_sigma=anchor_sigma,
        )
    except LadderResolveError:
        return False

    from .deblender import _joint_s2n, _adaptive

    # the packed family-covariance components map to the
    # (M1, M2, T) basis of the reported structure errors as
    # M1 = c11 - c00, M2 = 2 c01, T = c00 + c11
    L = np.array([
        [-1.0, 0.0, 1.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 1.0],
    ])

    nband = deb.nband
    ncen = 2 if deb.recenter else 0
    for i, robj in enumerate(res['objects']):
        i0 = slices[i]
        idx = [i0 + b for b in range(nband)]
        fcov = cov[np.ix_(idx, idx)]
        var = np.diag(fcov)
        flux_err = np.where(var > 0, np.sqrt(var), np.nan)
        robj['flux_cov'] = fcov
        robj['flux_err'] = flux_err
        wgood = var > 0
        if np.any(wgood):
            # covariance-aware total s/n; the quadrature sum is
            # the fallback for a non positive definite block
            s2n = _joint_s2n(
                robj['flux'][wgood], fcov[np.ix_(wgood, wgood)],
            )
            if s2n is None:
                s2n = np.sqrt(np.sum(
                    robj['flux'][wgood] ** 2 / var[wgood],
                ))
            robj['s2n'] = s2n

        lad = extras.get('ladder', {}).get(i)
        if lad is not None:
            for which in ('fixed', 'total'):
                c = lad[which + '_flux_cov']
                v = np.diag(c)
                robj[which + '_flux_cov'] = c
                robj[which + '_flux_err'] = np.where(
                    v > 0, np.sqrt(v), np.nan,
                )
            gv = lad['gradient_var']
            robj['gradient_err'] = np.where(gv > 0, np.sqrt(gv), np.nan)

        if deb.models[i]['type'] == 'star':
            # a delta function has no structure or weight state:
            # the packed block ends at the (optional) center
            # columns, and the structure offsets below would read
            # the next object's block.  The per-object structure
            # entries (T = 0, flagged shapes) stand
            continue

        if _adaptive(deb.models[i]):
            _apply_structure_errors(robj, cov, i0 + nband + ncen, L)

        gF = extras['gauss_flux'][i]
        gfc = extras['gauss_flux_cov'][i]
        if gfc is not None:
            gvar = np.diag(gfc)
            robj['gauss_flux'] = gF
            robj['gauss_flux_cov'] = gfc
            robj['gauss_flux_err'] = np.where(
                gvar > 0, np.sqrt(gvar), np.nan,
            )
            wg = gvar > 0
            if np.any(wg):
                gs2n = _joint_s2n(
                    gF[wg], gfc[np.ix_(wg, wg)],
                )
                if gs2n is None:
                    gs2n = np.sqrt(np.sum(
                        gF[wg] ** 2 / gvar[wg],
                    ))
                robj['gauss_s2n'] = gs2n
    return True


def _apply_structure_errors(robj, cov, ic, L):
    """
    Replace one adaptive object's structure and gauss-shape errors.

    From its family-covariance block at packed offset ic and the
    weight block that follows it.  Replaced only when the full values
    are usable, so a degenerate block cannot degrade a usable row.
    """
    from ngmix.flags import NONPOS_SHAPE_VAR

    from .deblender import _shape_errors

    cblock = cov[np.ix_(
        [ic, ic + 1, ic + 2], [ic, ic + 1, ic + 2],
    )]
    fam_err_cov = L @ cblock @ L.T
    if np.all(np.isfinite(fam_err_cov)) and fam_err_cov[2, 2] > 0:
        robj['T_err'] = np.sqrt(fam_err_cov[2, 2])
        if np.isfinite(robj['e1']) and robj['T'] > 0:
            e1e, e2e, eflags = _shape_errors(
                robj['e1'], robj['e2'], robj['T'], fam_err_cov,
            )
            if eflags == 0:
                robj['e1_err'] = e1e
                robj['e2_err'] = e2e
                # e_flags describes the reported errors: these
                # replace the per-object sandwich values, so a
                # shape-variance failure there no longer applies.
                # NONPOS_SIZE cannot be set on this branch (it
                # requires the shape itself usable), so this
                # restores e_flags == 0 iff shape and errors usable
                robj['e_flags'] &= ~NONPOS_SHAPE_VAR

    # gauss-estimator structure errors from the weight rows: the
    # gauss family is the weight minus the constant smoothing, so
    # its covariance is the wt_cov block
    isw = ic + 3
    gblock = cov[np.ix_(
        [isw, isw + 1, isw + 2], [isw, isw + 1, isw + 2],
    )]
    gfam_err_cov = L @ gblock @ L.T
    if np.all(np.isfinite(gfam_err_cov)) and gfam_err_cov[2, 2] > 0:
        robj['gauss_T_err'] = np.sqrt(gfam_err_cov[2, 2])
        if np.isfinite(robj['gauss_e1']) and robj['gauss_T'] > 0:
            e1e, e2e, eflags = _shape_errors(
                robj['gauss_e1'], robj['gauss_e2'],
                robj['gauss_T'], gfam_err_cov,
            )
            if eflags == 0:
                robj['gauss_e1_err'] = e1e
                robj['gauss_e2_err'] = e2e
                robj['gauss_e_flags'] &= ~NONPOS_SHAPE_VAR


def full_covariance(deb, mbobs, anchor_sigma=0.0,
                    use_chain=None):
    """
    The full covariance of the packed deblend state at the fixed point.

    In physical units, with the per-object state offsets, and the
    derived gauss-estimator flux values and covariances as a dict of
    per-object lists {'gauss_flux', 'gauss_flux_cov'} (None entries
    for stars).  See the module docstring.

    use_chain=True (the default) assembles the Jacobian and
    data response by the chain rule: analytic data-sum
    derivatives, closed-form model-sum derivatives, and the
    hand-differentiated update algebra (_phi_healthy), with
    per-object micro-FD fallback on guarded branches.  With the
    analytic algebra it is faster than the direct finite
    differences at every group size (measured 21 vs 25 ms for a
    single up to 126 vs 219 ms for five members, 2026-07-29).
    use_chain=False uses the full finite-difference
    evaluations, retained as the reference for the equivalence
    tests

    The dense products run under single_core_blas: the stack is
    single core by design (parallelism is process level), and
    the covariance/Jacobian products are large enough for a
    threaded BLAS to fan out to every core, which measures
    slower even in a single process
    """
    from ngmix.util import single_core_blas

    with single_core_blas():
        return _full_covariance(
            deb=deb, mbobs=mbobs, anchor_sigma=anchor_sigma,
            use_chain=use_chain,
        )


def _full_covariance(deb, mbobs, anchor_sigma, use_chain):
    if use_chain is None:
        use_chain = True
    nobj = deb.nobj

    obs_flat = [
        tobs for obslist in mbobs for tobs in obslist
    ]
    epochs = deb.epochs_per_obj[0]

    caches = [
        [_data_esums(deb, i, ep) for ep in epochs]
        for i in range(nobj)
    ]
    Ds = [
        [
            dsums_dtheta(
                ep, deb.wt_cov[i],
                deb.positions[i][0] - ep['vcen'],
                deb.positions[i][1] - ep['ucen'],
            )
            for ep in epochs
        ]
        for i in range(nobj)
    ]
    theta0s = [_theta_of(deb, i) for i in range(nobj)]

    # ladder groups: the amps are the closed-form response to
    # the state and the aperture/T-row data modes (see
    # _ladder_setup); make the stored amps the exact-weight
    # solve at the solution so the snapshot is self-consistent
    L = None
    W2 = s_star = None
    if any(m['type'] == 'ladder' for m in deb.models):
        L = _ladder_setup(deb, epochs)
        _ladder_resolve(deb, L, caches, Ds, theta0s, {}, {})
        W2, s_star = ladder_fixed_weight(deb.Tsmooth)

    snap = _save_state(deb)
    x0 = deb._pack_state()
    npars = x0.size
    slices, pers = _object_layout(deb)

    covS = _cov_sums(deb, obs_flat, epochs, L)

    cols = _column_map(deb)
    dNSm = None
    lstate = None
    if use_chain:
        J, dFdS, dFda, dNS, dNSm, lstate = _chain_pieces(
            deb, snap, x0, caches, Ds, theta0s, slices, pers,
            covS, epochs, L, W2, s_star,
        )
    else:
        resolve0 = None
        if L is not None:
            resolve0 = functools.partial(
                _ladder_resolve, deb, L, caches, Ds, theta0s,
                {}, {},
            )
        patched0 = _make_patched(deb, caches, Ds, theta0s, {})
        J = _fd_jacobian(
            deb, snap, x0, patched0, slices, pers, resolve0,
        )
        dFdS = _fd_dFdS(
            deb, snap, x0, caches, Ds, theta0s, slices, pers,
            covS, epochs, L,
        )
        dFda = None
        if L is None:
            dNS, _ = _model_sum_derivs(deb, cols)
        else:
            dNS, dNSm, lstate = _ladder_derivs(
                deb, snap, x0, L, caches, Ds, theta0s, cols,
                covS, epochs, W2, s_star,
            )

    if deb.recenter and np.any(deb.fixcen):
        # a fixed center is pinned to its injected anchor: its
        # packed offset is constant, so its update rows are zero.
        # The FD paths instead see the unpack/repack identity
        # (the update skips the center, leaving the perturbed
        # state in place), which would make I - J singular; the
        # analytic algebra (_phi_healthy) already uses the
        # pinned convention.  Applies to every fixcen object on
        # the fallback paths -- all stars, and weight-adaptive
        # members on guarded branches
        for i in range(nobj):
            if deb.fixcen[i]:
                icen = slices[i] + deb.nband
                J[icen:icen + 2, :] = 0.0

    M = np.eye(npars) - J
    Tx = np.linalg.solve(M, dFdS)
    cov_norm = Tx @ covS @ Tx.T

    anchor_cov = _anchor_cov(anchor_sigma, nobj)
    Ra = None
    if deb.recenter and anchor_cov is not None:
        if dFda is None:
            patched0 = _make_patched(
                deb, caches, Ds, theta0s, {},
            )
            dFda = _dF_danchor(
                deb, snap, x0, patched0, slices, pers,
            )
        Ra = np.linalg.solve(M, dFda)
        cov_norm = cov_norm + Ra @ anchor_cov @ Ra.T

    extras = _gauss_flux_covs(
        deb, caches, Ds, dNS, cols, slices, epochs,
        Tx, covS, Ra, anchor_cov, dNSm,
    )
    if L is not None:
        extras['ladder'] = _ladder_functional_covs(
            deb, snap, x0, L, caches, Ds, theta0s, cols, covS,
            epochs, Tx, Ra, anchor_cov, slices, lstate, W2, s_star,
        )

    D = np.diag(deb.scales)
    _restore_state(deb, snap)
    return D @ cov_norm @ D, slices, extras


def _flux_kernel_and_dtheta(ep, W, v0, u0):
    """
    The flux-sum kernel row of one weight and its state derivative.

    Returns the kernel row (nmodes, complex; the sum is Re(G @ kim))
    of weight W at center offset (v0, u0), and the analytic (5,)
    derivative of the flux sum with respect to (W00, W01, W11, v, u):
    the flux rows of moment_kernels and dsums_dtheta from one
    evaluation of the shared ingredients.  The aperture members of a
    ladder group need only these rows, and the full routines cost 6x
    more.  The referee of _flux_kernels_and_dtheta_dyadic.
    """
    Sv, Su, base, yf, xf, dim, _ = _ingredients(ep, W, v0, u0)
    kv, ku, kim = ep['kv'], ep['ku'], ep['kim']
    D = np.zeros(5)
    for w, dchi in enumerate((kv * kv, 2 * kv * ku, ku * ku)):
        D[w] = ((-0.5 * dchi * base) @ kim).real
    a0, b0 = get_phase_angles(ep, v0, u0)
    av, bv = get_phase_angles(ep, v0 + 1.0, u0)
    au, bu = get_phase_angles(ep, v0, u0 + 1.0)
    dSa = ((2j * np.pi / dim) * yf * base @ kim).real
    dSb = ((2j * np.pi / dim) * xf * base @ kim).real
    D[3] = dSa * (av - a0) + dSb * (bv - b0)
    D[4] = dSa * (au - a0) + dSb * (bu - b0)
    return base, D


def _flux_kernels_and_dtheta_dyadic(ep, wt_cov, v0, u0):
    """
    The flux kernel rows and derivatives of the eight dyadic apertures.

    _flux_kernel_and_dtheta for the apertures a wt_cov, a = LADDER_AP_FACS,
    from one evaluation of the shared ingredients: the phase and the
    quadratic form are common, the a=1 weight is one exact
    exponential and the others its square roots and squares, each
    with the same cutoff the per-aperture route applies.  Returns G
    (8, nmodes) and D (8, 5), the flux-sum derivatives with respect
    to each aperture's own covariance components and the center.
    """
    alpha, beta = get_phase_angles(ep, v0, u0)
    dim = ep['dim']
    iy = ep['iy'].astype(np.int64)
    ix = ep['ix'].astype(np.int64)
    kv, ku, kim = ep['kv'], ep['ku'], ep['kim']
    Sv = wt_cov[0, 0] * kv + wt_cov[0, 1] * ku
    Su = wt_cov[0, 1] * kv + wt_cov[1, 1] * ku
    chi2 = kv * Sv + ku * Su
    yf = np.where(iy < (dim + 1) // 2, iy, iy - dim)
    xf = np.where(ix < (dim + 1) // 2, ix, ix - dim)
    pd = np.exp(2j * np.pi / dim * (alpha * yf + beta * xf)) * ep['df2']
    w1 = np.exp(-0.5 * chi2)
    wh = np.sqrt(w1)
    ws = [np.sqrt(wh), wh, w1]
    w = w1
    for _ in range(5):
        w = w * w
        ws.append(w)
    dchi = (kv * kv, 2 * kv * ku, ku * ku)
    a0, b0 = alpha, beta
    av, bv = get_phase_angles(ep, v0 + 1.0, u0)
    au, bu = get_phase_angles(ep, v0, u0 + 1.0)
    nap = LADDER_AP_FACS.size
    G = np.empty((nap, kim.size), dtype=complex)
    D = np.zeros((nap, 5))
    for j, af in enumerate(LADDER_AP_FACS):
        wa = np.where(af * chi2 < WK_CHI2_MAX, ws[j], 0.0)
        base = wa * pd
        G[j] = base
        for wq in range(3):
            D[j, wq] = ((-0.5 * dchi[wq] * base) @ kim).real
        dSa = ((2j * np.pi / dim) * yf * base @ kim).real
        dSb = ((2j * np.pi / dim) * xf * base @ kim).real
        D[j, 3] = dSa * (av - a0) + dSb * (bv - b0)
        D[j, 4] = dSa * (au - a0) + dSb * (bu - b0)
    return G, D


def _ladder_setup(deb, epochs):
    """
    The ladder error context at the solution.

    The amps are not state: they are the closed-form response of the
    joint solve to the state (apertures, rungs, flux scales, priors,
    the non-ladder members' models) and to two kinds of data modes,
    the aperture flux sums and the T-row moment sum.  The context
    holds the objects, the exact aperture-row variances and epoch
    weights, the raw per-epoch aperture flux sums and their analytic
    derivatives with respect to the object's weight and center (the
    flux row of dsums_dtheta at the aperture weight; the aperture is
    a fixed multiple of the weight, so the chain factor is that
    multiple).  The kernel rows themselves are not kept: (8, nmodes)
    complex per object-epoch is GBs on a large group, so _cov_sums
    rebuilds them where they are consumed.
    """
    R = ladder_rows(deb, use_cache=False)
    idx, aps, wt_covs = R['idx'], R['aps'], R['wt_covs']
    d, var, wsum, raw = R['d'], R['var'], R['wsum'], R['raw']
    nap = LADDER_AP_FACS.size
    Dap = []
    for io, i in enumerate(idx):
        v0, u0 = deb.positions[i]
        Dio = [[] for _ in range(nap)]
        for ep in epochs:
            # the kernel rows themselves are not kept: (8, nmodes)
            # complex per object-epoch is GBs on a large group;
            # _cov_sums rebuilds them where they are consumed
            _, D = _flux_kernels_and_dtheta_dyadic(
                ep, wt_covs[io], v0 - ep['vcen'], u0 - ep['ucen'],
            )
            for j in range(nap):
                Dio[j].append(D[j])
        Dap.append(Dio)
    return {
        'idx': idx, 'nap': nap, 'var': var, 'wsum': wsum,
        'var_total': ladder_total_var(d, var),
        'raw': raw, 'Dap': Dap, 'aps0': aps, 'wt_covs0': wt_covs,
    }


def _ladder_rows_at(deb, L, caches, Ds, theta0s, dap, dT):
    """
    The linearized rows, template and prior at the current state.

    At the CURRENT (unpacked) deblender state: the aperture sums move
    with the object's weight and center through the analytic kernel
    derivatives, the T row through the moment sum derivatives, and
    the deltas dap[(io, j, iep)] and dT[(i, iep, a)] inject data-mode
    perturbations.  Everything a solve at any prior width needs; the
    row weights are frozen at the solution.
    """
    idx = L['idx']
    nap = L['nap']
    _, aps, wt_covs, Tws, Fhat = ladder_context(deb)
    nband = deb.nband
    nlad = len(idx)
    nmom = len(t_row_indices())
    d = np.zeros((nlad, nap + nmom, nband))
    for io, i in enumerate(idx):
        dth = _theta_of(deb, i) - theta0s[i]
        for iep, ep in enumerate(deb.epochs_per_obj[i]):
            band = ep['band']
            fac = ep['weight'] * ep['detAtinv']
            for j in range(nap):
                Dj = L['Dap'][io][j][iep]
                s = (
                    L['raw'][io][j, iep]
                    + LADDER_AP_FACS[j] * (Dj[0:3] @ dth[0:3])
                    + Dj[3:5] @ dth[3:5]
                    + dap.get((io, j, iep), 0.0)
                )
                d[io, j, band] += fac * s
            for r, a in enumerate(t_row_indices()):
                s = (
                    caches[i][iep][a] + Ds[i][iep][a] @ dth
                    + dT.get((i, iep, a), 0.0)
                )
                d[io, nap + r, band] += fac * s
    ladder_subtract_others(deb, idx, aps, wt_covs, L['wsum'], d)
    Mt = ladder_template(deb, idx, aps, wt_covs)
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(Mt))):
        raise LadderResolveError('non-finite ladder rows')
    pieces = ladder_assemble(
        deb, idx, L['var'], L['wsum'], wt_covs, Fhat, Mt,
    )
    if L['var_total'] is L['var']:
        pieces_total = pieces
    else:
        # the total-flux solve sees the capped rows (frozen
        # weights, like the others)
        pieces_total = ladder_assemble(
            deb, idx, L['var_total'], L['wsum'], wt_covs, Fhat, Mt,
        )
    return {'wt_covs': wt_covs, 'Fhat': Fhat, 'd': d, 'Mt': Mt,
            'pieces': pieces, 'pieces_total': pieces_total}


def _pieces_for(ctx, tau0):
    """
    The assembled solve pieces for a prior width.

    The total-flux solve (tau0 == LADDER_TAU_TOTAL) has its own, from
    the capped rows.
    """
    if tau0 is not None and tau0 == LADDER_TAU_TOTAL:
        return ctx['pieces_total']
    return ctx['pieces']


def _ladder_solve_at(deb, L, ctx, tau0=None):
    """
    The amps from the rows of _ladder_rows_at at a prior width.

    The tau-independent assembly is shared; a failed solve raises
    LadderResolveError.
    """
    new_full = ladder_solve_pieces(
        deb, L['idx'], ctx['d'], _pieces_for(ctx, tau0), tau0=tau0,
    )
    if new_full is None:
        raise LadderResolveError('ladder re-solve failed')
    return new_full


def _ladder_resolve(deb, L, caches, Ds, theta0s, dap, dT, tau0=None):
    """
    Re-solve the ladder amps at the current deblender state.

    At the CURRENT (unpacked) state (see _ladder_rows_at); stores the
    amps in the models and returns them as (nband, Z).  tau0
    overrides the prior width (the derived total flux).
    """
    ctx = _ladder_rows_at(deb, L, caches, Ds, theta0s, dap, dT)
    new_full = _ladder_solve_at(deb, L, ctx, tau0)
    ladder_write_amps(deb, L['idx'], new_full)
    return new_full


def _ladder_state_derivs(deb, snap, x0, L, caches, Ds, theta0s,
                         cols, W2, s_star):
    """
    The FD referee of the ladder's state response.

    One central-FD loop over the packed state columns for everything
    the ladder needs per state: the neighbor sums with the re-solved
    subtraction amps (dNS[(i, col)], physical units per unit physical
    change) and the derived flux functionals -- the fixed-aperture
    flux of the subtraction amps and the total flux of the
    LADDER_TAU_TOTAL solve -- as (nlad, nband, npars) responses in
    the normalized columns, plus their values at the solution.  One
    row/template/prior build per evaluation serves both solves.
    """
    nobj = deb.nobj
    idx = L['idx']
    nlad = len(idx)
    K = LADDER_RUNGS.size
    nband = deb.nband
    npars = x0.size
    scales = deb.scales

    def evaluate(x):
        _restore_state(deb, snap)
        deb._unpack_state(x)
        ctx = _ladder_rows_at(deb, L, caches, Ds, theta0s, {}, {})
        asub = _ladder_solve_at(deb, L, ctx, None)
        ladder_write_amps(deb, idx, asub)
        ns = [deb._get_neighbor_sums(i) for i in range(nobj)]
        atot = _ladder_solve_at(deb, L, ctx, LADDER_TAU_TOTAL)
        fixed = ladder_fixed_fluxes(deb, idx, asub, W2, s_star)
        total = np.array([
            atot[:, io * K:(io + 1) * K].sum(axis=1)
            for io in range(nlad)
        ])
        _restore_state(deb, snap)
        return ns, fixed, total

    dNS = {}
    G_fixed = np.zeros((nlad, nband, npars))
    G_total = np.zeros((nlad, nband, npars))
    for ic in range(npars):
        xp = x0.copy()
        xm = x0.copy()
        xp[ic] += FD_H
        xm[ic] -= FD_H
        nsp, fpf, fpt = evaluate(xp)
        nsm, fmf, fmt = evaluate(xm)
        for i in range(nobj):
            d = (nsp[i] - nsm[i]) / (2 * FD_H * scales[ic])
            if np.any(d != 0):
                dNS[(i, ic)] = d
        G_fixed[:, :, ic] = (fpf - fmf) / (2 * FD_H)
        G_total[:, :, ic] = (fpt - fmt) / (2 * FD_H)
    _, f0f, f0t = evaluate(x0)
    return dNS, {
        'G_fixed': G_fixed, 'G_total': G_total,
        'f0_fixed': f0f, 'f0_total': f0t,
    }


def _ladder_state_response(deb, snap, x0, L, caches, Ds, theta0s,
                           cols, W2, s_star, dNS_direct):
    """
    The analytic state response of the ladder.

    For every packed state column, the amps' response through the
    solve chain dX = A^-1 (d rhs - dA X) at both prior widths, with
    the template, prior and subtracted-row derivatives from central
    micro-FD on the batched closed-form kernels and the data-row
    derivatives from the analytic kernel derivatives; then the
    neighbor sums (the fixed-amps direct part dNS_direct plus the
    amps channel through the unit rung sums) and the derived
    functionals.  Returns dNS (physical units per unit physical
    change) and the lstate dict of _ladder_state_derivs, of which
    this is the fast replacement (that FD loop stays as the
    referee).
    """
    _restore_state(deb, snap)
    idx = L['idx']
    nap = L['nap']
    nobj = deb.nobj
    nlad = len(idx)
    K = LADDER_RUNGS.size
    nband = deb.nband
    npars = x0.size
    scales = deb.scales
    nmom = len(t_row_indices())
    nrows = nap + nmom
    Z = nlad * K
    N = nband * Z
    wsum = L['wsum']
    lad_of = {i: io for io, i in enumerate(idx)}
    taus = (LADDER_TAU0, LADDER_TAU_TOTAL)

    # the solution: rows, template, pieces, amps at both widths
    ctx = _ladder_rows_at(deb, L, caches, Ds, theta0s, {}, {})
    d0 = ctx['d']
    Mt = ctx['Mt']
    _, _, _, css, a0 = ctx['pieces']
    pieces_of = {tau: _pieces_for(ctx, tau) for tau in taus}
    wt_covs = ctx['wt_covs']
    aps = [[af * wt_covs[io] for af in LADDER_AP_FACS] for io in range(nlad)]
    Ainv = {}
    X = {}
    amps = {}
    lams = {}
    dlams = {}
    for tau in taus:
        A0 = pieces_of[tau][0]
        a0_tau = pieces_of[tau][4]
        lams[tau], dlams[tau] = ladder_prior_lambda(a0_tau, tau)
        Ainv[tau] = np.linalg.inv(A0 + np.diag(np.tile(lams[tau], nband)))
        full = ladder_solve_pieces(deb, idx, d0, pieces_of[tau], tau0=tau)
        if full is None:
            raise LadderResolveError('ladder solve failed')
        amps[tau] = full
        X[tau] = np.concatenate([full[b] / css[b] for b in range(nband)])
    U = ladder_neighbor_unit_sums(deb, idx)      # (nobj, nlad, K, 6)
    u_fix = ladder_fixed_units(deb, idx, W2, s_star)   # (nlad, K)
    rows_b = [np.repeat(wsum[:, b], nrows) for b in range(nband)]
    f0_fixed = ladder_fixed_fluxes(deb, idx, amps[LADDER_TAU0], W2, s_star)
    f0_total = np.array([
        amps[LADDER_TAU_TOTAL][:, io * K:(io + 1) * K].sum(axis=1)
        for io in range(nlad)
    ])

    def others_rows():
        """
        The subtracted rows at the current in-place state.

        Minus the others' contribution only, on zero data rows.
        """
        dd = np.zeros((nlad, nrows, nband))
        ladder_subtract_others(deb, idx, aps, wt_covs, wsum, dd)
        return dd

    def chain(dMt, dd, da0, dcss):
        """
        The amps' response at both prior widths.

        For the given derivatives of the template, rows, prior and column
        scales.
        """
        out = {}
        for tau in taus:
            lam = lams[tau]
            a0_tau = pieces_of[tau][4]
            # the prior precisions move with the center in the
            # multiplicative mode (zero in the uniform mode)
            dlam = dlams[tau] * da0
            drhs = np.zeros(N)
            dAX = np.zeros(N)
            for b in range(nband):
                sl = slice(b * Z, (b + 1) * Z)
                Mw = pieces_of[tau][1][b]
                sig = pieces_of[tau][2][b]
                dMw = (
                    ((dMt * rows_b[b][:, None]) / sig[:, None])
                    * css[b][None, :]
                    + ((Mt * rows_b[b][:, None]) / sig[:, None])
                    * dcss[b][None, :]
                )
                db = d0[:, :, b].reshape(-1) / sig
                ddb = dd[:, :, b].reshape(-1) / sig
                xb = X[tau][sl]
                drhs[sl] = (
                    dMw.T @ db + Mw.T @ ddb + lam * da0 + dlam * a0_tau
                )
                dAX[sl] = (
                    dMw.T @ (Mw @ xb) + Mw.T @ (dMw @ xb) + dlam * xb
                )
            dX = Ainv[tau] @ (drhs - dAX)
            out[tau] = np.stack([
                dX[b * Z:(b + 1) * Z] * css[b]
                + X[tau][b * Z:(b + 1) * Z] * dcss[b]
                for b in range(nband)
            ])
        return out

    Tw0 = deb.wt_cov[0][0, 0] + deb.wt_cov[0][1, 1]
    h_cov = 1.0e-6 * max(Tw0, 0.1)
    h_pos = 1.0e-6
    zero_css = [np.zeros(Z) for _ in range(nband)]
    dNS = {}
    G_fixed = np.zeros((nlad, nband, npars))
    G_total = np.zeros((nlad, nband, npars))

    for ic, (k, kind, sub) in enumerate(cols):
        m = deb.models[k]
        dMt = None
        dd = np.zeros((nlad, nrows, nband))
        da0 = np.zeros(Z)
        dcss = zero_css
        du_fix = None
        if k in lad_of:
            io = lad_of[k]
            blk = slice(io * K, (io + 1) * K)
            if kind == 'F':
                F = m['F'][sub]
                if abs(F) > 1.0e-12:
                    dcss = [np.zeros(Z) for _ in range(nband)]
                    dcss[sub][blk] = np.sign(F)
                else:
                    continue
            elif kind == 'cov':
                # cov_sm enters nothing of the ladder
                continue
            elif kind == 'sw':
                r, c = _SYM[sub]
                wt_cov0 = deb.wt_cov[k]
                rungs0 = m['rungs']
                Mts = []
                a0s = []
                dds = []
                us = []
                for sgn in (1.0, -1.0):
                    wt_cov_p = wt_cov0.copy()
                    wt_cov_p[r, c] += sgn * h_cov
                    wt_cov_p[c, r] = wt_cov_p[r, c]
                    deb.wt_cov[k] = wt_cov_p
                    m['rungs'] = ladder_rung_covs(wt_cov_p, deb.Tsmooth)
                    wt_covs[io] = wt_cov_p
                    aps[io] = [af * wt_cov_p for af in LADDER_AP_FACS]
                    Mts.append(ladder_template(deb, idx, aps, wt_covs))
                    a0s.append(ladder_exp_fracs(
                        m['rungs'], wt_cov_p, deb.Tsmooth,
                    ))
                    dds.append(others_rows())
                    S00, S01, S11 = m['rungs']
                    us.append(unit_flux_sums(S00, S01, S11, W2) / s_star)
                deb.wt_cov[k] = wt_cov0
                m['rungs'] = rungs0
                wt_covs[io] = wt_cov0
                aps[io] = [af * wt_cov0 for af in LADDER_AP_FACS]
                dMt = (Mts[0] - Mts[1]) / (2 * h_cov)
                da0[blk] = (a0s[0] - a0s[1]) / (2 * h_cov)
                dd += (dds[0] - dds[1]) / (2 * h_cov)
                du_fix = (us[0] - us[1]) / (2 * h_cov)
                # the data rows through the kernel derivatives
                for iep, ep in enumerate(deb.epochs_per_obj[k]):
                    b = ep['band']
                    fac = ep['weight'] * ep['detAtinv']
                    for j in range(nap):
                        dd[io, j, b] += (
                            fac * LADDER_AP_FACS[j]
                            * L['Dap'][io][j][iep][sub]
                        )
                    for rr, a in enumerate(t_row_indices()):
                        dd[io, nap + rr, b] += (
                            fac * Ds[k][iep][a, sub]
                        )
            elif kind == 'cen':
                pos0 = deb.positions[k]
                Mts = []
                dds = []
                for sgn in (1.0, -1.0):
                    pp = list(pos0)
                    pp[sub] += sgn * h_pos
                    deb.positions[k] = tuple(pp)
                    Mts.append(ladder_template(deb, idx, aps, wt_covs))
                    dds.append(others_rows())
                deb.positions[k] = pos0
                dMt = (Mts[0] - Mts[1]) / (2 * h_pos)
                dd += (dds[0] - dds[1]) / (2 * h_pos)
                for iep, ep in enumerate(deb.epochs_per_obj[k]):
                    b = ep['band']
                    fac = ep['weight'] * ep['detAtinv']
                    for j in range(nap):
                        dd[io, j, b] += (
                            fac * L['Dap'][io][j][iep][3 + sub]
                        )
                    for rr, a in enumerate(t_row_indices()):
                        dd[io, nap + rr, b] += (
                            fac * Ds[k][iep][a, 3 + sub]
                        )
            else:
                continue
        else:
            # a non-ladder member: only the subtracted rows move
            if kind == 'sw' or kind == 'fracdev':
                # its weight enters nothing of the ladder
                dds = None
            elif kind == 'F':
                F0 = m['F'].copy()
                dds = []
                for sgn in (1.0, -1.0):
                    m['F'] = F0.copy()
                    m['F'][sub] += sgn * 1.0
                    dds.append(others_rows())
                m['F'] = F0
                dd += (dds[0] - dds[1]) / 2.0
            elif kind == 'cov':
                key = _covkey(m)
                r, c = _SYM[sub]
                cov0 = m[key].copy()
                dds = []
                for sgn in (1.0, -1.0):
                    covp = cov0.copy()
                    covp[r, c] += sgn * h_cov
                    covp[c, r] = covp[r, c]
                    m[key] = covp
                    dds.append(others_rows())
                m[key] = cov0
                dd += (dds[0] - dds[1]) / (2 * h_cov)
            elif kind == 'cen':
                pos0 = deb.positions[k]
                dds = []
                for sgn in (1.0, -1.0):
                    pp = list(pos0)
                    pp[sub] += sgn * h_pos
                    deb.positions[k] = tuple(pp)
                    dds.append(others_rows())
                deb.positions[k] = pos0
                dd += (dds[0] - dds[1]) / (2 * h_pos)
            if dds is None:
                continue

        if dMt is None:
            dMt = np.zeros_like(Mt)
        damps = chain(dMt, dd, da0, dcss)
        dsub = damps[LADDER_TAU0].reshape(nband, nlad, K)
        for i in range(nobj):
            dn = np.einsum('jcs,bjc->bs', U[i], dsub)
            base = dNS_direct.get((i, ic))
            if base is not None:
                dn = dn + base
            if np.any(dn != 0):
                dNS[(i, ic)] = dn
        for jo in range(nlad):
            gf = dsub[:, jo, :] @ u_fix[jo]
            if du_fix is not None and jo == lad_of.get(k, -1):
                gf = gf + amps[LADDER_TAU0][:, jo * K:(jo + 1) * K] @ du_fix
            G_fixed[jo, :, ic] = gf * scales[ic]
            G_total[jo, :, ic] = (
                damps[LADDER_TAU_TOTAL].reshape(nband, nlad, K)[:, jo, :]
                .sum(axis=1) * scales[ic]
            )
    # the direct part alone for the columns skipped above
    for (i, ic), base in dNS_direct.items():
        if (i, ic) not in dNS and np.any(base != 0):
            dNS[(i, ic)] = base
    _restore_state(deb, snap)
    return dNS, {
        'G_fixed': G_fixed, 'G_total': G_total,
        'f0_fixed': f0_fixed, 'f0_total': f0_total,
    }


def _ladder_derivs(deb, snap, x0, L, caches, Ds, theta0s, cols,
                   covS, epochs, W2, s_star, dNS_direct=None):
    """
    The neighbor-sum derivatives of a ladder group.

    dNS[(i, col)] per packed state column (physical units per unit
    physical change, the complete dependence including the non-ladder
    members' direct terms, the rung frames and the amps), and
    dNSm[(i, col)] per data mode (the aperture sums and the T-row
    moment sums of the ladder objects) analytically: the rows enter
    the solve's right-hand side linearly, so the amp response to a
    row is a column of A^-1 Mw^T / sigma times the epoch factor, and
    the neighbor sums respond to the amps through the unit rung sums
    (ladder_neighbor_unit_sums).  Also returns the derived-functional
    state responses.  With dNS_direct (the fixed-amps direct part
    from _model_sum_derivs) the state channel is the analytic
    _ladder_state_response; without it the FD referee
    _ladder_state_derivs.
    """
    nobj = deb.nobj
    nep = len(epochs)
    idx = L['idx']
    nap = L['nap']
    nS0 = 6 * nobj * nep

    if dNS_direct is None:
        dNS, lstate = _ladder_state_derivs(
            deb, snap, x0, L, caches, Ds, theta0s, cols, W2, s_star,
        )
    else:
        dNS, lstate = _ladder_state_response(
            deb, snap, x0, L, caches, Ds, theta0s, cols, W2, s_star,
            dNS_direct,
        )

    # data modes, analytic through the solve at the solution
    _restore_state(deb, snap)
    nmom = len(t_row_indices())
    K = LADDER_RUNGS.size
    nband = deb.nband
    nlad = len(idx)
    Z = nlad * K
    nrows = nap + nmom
    _, aps, wt_covs, Tws, Fhat = ladder_context(deb)
    Mt = ladder_template(deb, idx, aps, wt_covs)
    A, Mws, sigs, css, a0 = ladder_assemble(
        deb, idx, L['var'], L['wsum'], wt_covs, Fhat, Mt,
    )
    lam, _ = ladder_prior_lambda(a0, LADDER_TAU0)
    Ainv = np.linalg.inv(A + np.diag(np.tile(lam, nband)))
    R = [
        Ainv[:, b * Z:(b + 1) * Z] @ (Mws[b].T / sigs[b][None, :])
        for b in range(nband)
    ]
    U = ladder_neighbor_unit_sums(deb, idx)   # (nobj, nlad, K, 6)

    def mode_resp(b, row, fac):
        dX = R[b][:, row] * fac
        damps = np.stack([
            dX[bp * Z:(bp + 1) * Z] * css[bp] for bp in range(nband)
        ]).reshape(nband, nlad, K)
        out = {}
        for k in range(nobj):
            d = np.einsum('jcs,bjc->bs', U[k], damps)
            if np.any(d != 0):
                out[k] = d
        return out

    dNSm = {}
    for io, i in enumerate(idx):
        for iep, ep in enumerate(epochs):
            b = ep['band']
            fac = ep['weight'] * ep['detAtinv']
            for j in range(nap):
                col = nS0 + (io * nap + j) * nep + iep
                for k, d in mode_resp(b, io * nrows + j, fac).items():
                    dNSm[(k, col)] = d
            for r, a in enumerate(t_row_indices()):
                col = (i * nep + iep) * 6 + a
                for k, d in mode_resp(
                        b, io * nrows + nap + r, fac).items():
                    dNSm[(k, col)] = d
    return dNS, dNSm, lstate


def _ladder_functional_covs(deb, snap, x0, L, caches, Ds, theta0s,
                            cols, covS, epochs, Tx, Ra, anchor_cov,
                            slices, lstate, W2, s_star):
    """
    The covariances of the ladder objects' derived flux functionals.

    The fixed-aperture flux (a linear functional of the subtraction
    amps) and the total flux (sum of the amps of the LADDER_TAU_TOTAL
    solve of the same rows), each with the direct data channel
    analytic through the solve matrix and the state channel from the
    state response, chained through Tx; and the variance of the color
    gradient (fixed minus adaptive color per adjacent band pair) from
    the fixed-flux response and the flux rows of Tx.  Returns
    {i: {'fixed_flux_cov', 'total_flux_cov', 'gradient_var'}}.
    """
    idx = L['idx']
    nap = L['nap']
    nobj = deb.nobj
    nep = len(epochs)
    nband = deb.nband
    nlad = len(idx)
    K = LADDER_RUNGS.size
    Z = nlad * K
    nmom = len(t_row_indices())
    nrows = nap + nmom
    nS0 = 6 * nobj * nep
    nS = covS.shape[0]
    npars = x0.size
    scales = deb.scales

    # direct data channel at the solution
    _restore_state(deb, snap)
    _, aps, wt_covs, Tws, Fhat = ladder_context(deb)
    Mt = ladder_template(deb, idx, aps, wt_covs)
    us = list(ladder_fixed_units(deb, idx, W2, s_star))
    Rd = {
        'fixed': np.zeros((nlad, nband, nS)),
        'total': np.zeros((nlad, nband, nS)),
    }
    for which, tau, var_w in (('fixed', LADDER_TAU0, L['var']),
                              ('total', LADDER_TAU_TOTAL,
                               L['var_total'])):
        A0, Mws, sigs, css, a0_w = ladder_assemble(
            deb, idx, var_w, L['wsum'], wt_covs, Fhat, Mt,
        )
        lam_w, _ = ladder_prior_lambda(a0_w, tau)
        Ainv = np.linalg.inv(A0 + np.diag(np.tile(lam_w, nband)))
        R = [
            Ainv[:, b * Z:(b + 1) * Z] @ (Mws[b].T / sigs[b][None, :])
            for b in range(nband)
        ]
        for io, i in enumerate(idx):
            for iep, ep in enumerate(epochs):
                b = ep['band']
                fac = ep['weight'] * ep['detAtinv']
                modes = [
                    (nS0 + (io * nap + j) * nep + iep, io * nrows + j)
                    for j in range(nap)
                ] + [
                    ((i * nep + iep) * 6 + a, io * nrows + nap + r)
                    for r, a in enumerate(t_row_indices())
                ]
                for col, row in modes:
                    dX = R[b][:, row] * fac
                    for jo in range(nlad):
                        slj = slice(jo * K, (jo + 1) * K)
                        for bp in range(nband):
                            da = dX[bp * Z + jo * K:bp * Z + (jo + 1) * K] \
                                * css[bp][slj]
                            if which == 'fixed':
                                Rd[which][jo, bp, col] += da @ us[jo]
                            else:
                                Rd[which][jo, bp, col] += da.sum()

    # state channel from the shared FD loop
    G = {'fixed': lstate['G_fixed'], 'total': lstate['G_total']}
    _restore_state(deb, snap)

    cen_cols = {
        (k, sub): ic for ic, (k, kind, sub) in enumerate(cols)
        if kind == 'cen'
    }

    def anchor_term(g):
        ga = g @ Ra
        for (k, sub), ic in cen_cols.items():
            ga[:, 2 * k + sub] += g[:, ic] / scales[ic]
        return ga @ anchor_cov @ ga.T

    out = {}
    lnk = -2.5 / np.log(10.0)
    for io, i in enumerate(idx):
        ent = {}
        Rg = {}
        for which in ('fixed', 'total'):
            Rg[which] = Rd[which][io] + G[which][io] @ Tx
            cov = Rg[which] @ covS @ Rg[which].T
            if Ra is not None:
                cov = cov + anchor_term(G[which][io])
            ent[which + '_flux_cov'] = cov
        # the color gradient: fixed color minus adaptive color
        f2 = lstate['f0_fixed'][io]
        F = deb.models[i]['F']
        i0 = slices[i]
        RF = Tx[i0:i0 + nband] * scales[i0:i0 + nband, None]
        GF = np.zeros((nband, npars))
        for b in range(nband):
            GF[b, i0 + b] = scales[i0 + b]
        gvar = np.full(max(nband - 1, 0), np.nan)
        for c in range(nband - 1):
            if not (f2[c] > 0 and f2[c + 1] > 0
                    and F[c] > 0 and F[c + 1] > 0):
                continue
            rg = lnk * (
                Rg['fixed'][c] / f2[c] - Rg['fixed'][c + 1] / f2[c + 1]
                - RF[c] / F[c] + RF[c + 1] / F[c + 1]
            )
            v = rg @ covS @ rg
            if Ra is not None:
                gg = lnk * (
                    G['fixed'][io][c] / f2[c]
                    - G['fixed'][io][c + 1] / f2[c + 1]
                    - GF[c] / F[c] + GF[c + 1] / F[c + 1]
                )
                v = v + anchor_term(gg[None, :])[0, 0]
            gvar[c] = v
        ent['gradient_var'] = gvar
        out[i] = ent
    return out


def _fd_jacobian(deb, snap, x0, patched0, slices, pers,
                 resolve0=None):
    """
    The Jacobi-form sweep Jacobian by central FD over the packed state.

    The reference implementation for the chain.  For ladder groups
    resolve0 re-solves the amps at each perturbed state.
    """
    npars = x0.size
    J = np.zeros((npars, npars))
    for j in range(npars):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += FD_H
        xm[j] -= FD_H
        for i in range(deb.nobj):
            sl = slice(slices[i], slices[i] + pers[i])
            bp = _jacobi_block(deb, snap, xp, i, patched0, resolve0)
            bm = _jacobi_block(deb, snap, xm, i, patched0, resolve0)
            J[sl, j] = (bp[sl] - bm[sl]) / (2 * FD_H)
    return J


def _fd_dFdS(deb, snap, x0, caches, Ds, theta0s, slices, pers,
             covS, epochs, L=None):
    """
    The data response by central FD.

    The reference implementation for the chain.  For ladder groups
    the T-row modes of the ladder objects and the aperture modes
    reach every member through the re-solved amps.
    """
    nobj = deb.nobj
    npars = x0.size
    nep = len(epochs)
    nS = covS.shape[0]
    dsteps = DS_FAC * np.sqrt(np.diag(covS))
    dFdS = np.zeros((npars, nS))
    lidx = L['idx'] if L is not None else []

    def resolver(dap, dT):
        return functools.partial(
            _ladder_resolve, deb, L, caches, Ds, theta0s, dap, dT,
        )

    for i in range(nobj):
        for iep in range(nep):
            for a in range(6):
                col = (i * nep + iep) * 6 + a
                h = dsteps[col]
                if h == 0:
                    # a zero-variance sum (every live pixel has a
                    # zero influence kernel, e.g. an epoch whose
                    # positive-weight pixels are all apodized to
                    # zero): its covS row and column are exactly
                    # zero, so the response column is irrelevant;
                    # leave it zero rather than form 0/0
                    continue
                d = np.zeros(6)
                d[a] = h
                pp = _make_patched(
                    deb, caches, Ds, theta0s, {(i, iep): d},
                )
                pm = _make_patched(
                    deb, caches, Ds, theta0s, {(i, iep): -d},
                )
                amp_mode = (
                    L is not None and i in lidx
                    and a in t_row_indices()
                )
                if amp_mode:
                    rp = resolver({}, {(i, iep, a): h})
                    rm = resolver({}, {(i, iep, a): -h})
                    ks = range(nobj)
                else:
                    rp = rm = None
                    ks = [i]
                for k in ks:
                    slk = slice(slices[k], slices[k] + pers[k])
                    bp = _jacobi_block(deb, snap, x0, k, pp, rp)
                    bm = _jacobi_block(deb, snap, x0, k, pm, rm)
                    dFdS[slk, col] = (bp[slk] - bm[slk]) / (2 * h)
    if L is not None:
        nS0 = 6 * nobj * nep
        nap = L['nap']
        patched0 = _make_patched(deb, caches, Ds, theta0s, {})
        for io in range(len(lidx)):
            for j in range(nap):
                for iep in range(nep):
                    col = nS0 + (io * nap + j) * nep + iep
                    h = dsteps[col]
                    if h == 0:
                        continue
                    rp = resolver({(io, j, iep): h}, {})
                    rm = resolver({(io, j, iep): -h}, {})
                    for k in range(nobj):
                        slk = slice(slices[k], slices[k] + pers[k])
                        bp = _jacobi_block(
                            deb, snap, x0, k, patched0, rp,
                        )
                        bm = _jacobi_block(
                            deb, snap, x0, k, patched0, rm,
                        )
                        dFdS[slk, col] = (
                            bp[slk] - bm[slk]
                        ) / (2 * h)
    return dFdS


def _fastcopy(obj):
    """
    A cheap recursive copy for the small deblender state.

    Arrays, dicts, lists, tuples and scalars; much faster than
    copy.deepcopy for this shape of data.
    """
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _fastcopy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fastcopy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_fastcopy(v) for v in obj)
    return obj


def _save_state(deb):
    return {
        k: _fastcopy(deb.__dict__[k])
        for k in MUTABLE_ATTRS if k in deb.__dict__
    }


def _restore_state(deb, snap):
    for k, v in snap.items():
        deb.__dict__[k] = _fastcopy(v)


def _anchor_cov(anchor_sigma, nobj):
    """
    The block-diagonal anchor covariance from the anchor_sigma input.

    (2 nobj, 2 nobj), from a scalar sigma in arcsec (isotropic,
    shared), an (nobj,) array of per-object sigmas, or an
    (nobj, 2, 2) array of per-object position covariances in
    arcsec^2 with (v, u) ordering.  None when every entry is zero
    (the errors then condition on the anchors).
    """
    a = np.asarray(anchor_sigma, dtype=float)
    if not np.any(a != 0):
        return None
    out = np.zeros((2 * nobj, 2 * nobj))
    for i in range(nobj):
        sl = slice(2 * i, 2 * i + 2)
        if a.ndim == 0:
            out[sl, sl] = a ** 2 * np.eye(2)
        elif a.ndim == 1:
            if a.size != nobj:
                raise ValueError(
                    'per-object anchor_sigma must have one '
                    f'entry per object, got {a.size} for '
                    f'{nobj}'
                )
            out[sl, sl] = a[i] ** 2 * np.eye(2)
        elif a.ndim == 3:
            if a.shape != (nobj, 2, 2):
                raise ValueError(
                    'anchor covariances must have shape '
                    f'(nobj, 2, 2), got {a.shape}'
                )
            out[sl, sl] = a[i]
        else:
            raise ValueError(
                'anchor_sigma must be a scalar, (nobj,) sigmas '
                f'or (nobj, 2, 2) covariances, got shape '
                f'{a.shape}'
            )
    return out


def _object_layout(deb):
    """
    Per-object offsets and sizes in the packed state.

    Replays the _pack_state layout.
    """
    from .deblender import _adaptive

    slices = []
    pers = []
    k = 0
    for m in deb.models:
        slices.append(k)
        per = deb.nband
        if deb.recenter:
            per += 2
        if _adaptive(m):
            if m['type'] in ('gauss', 'ladder'):
                per += 3
            elif m['type'] in ('exp', 'dev'):
                per += 3
            elif m['type'] == 'bdf':
                per += 4
            per += 3
        pers.append(per)
        k += per
    return slices, pers


def _theta_of(deb, i):
    sw = deb.wt_cov[i]
    v, u = deb.positions[i]
    return np.array([sw[0, 0], sw[0, 1], sw[1, 1], v, u])


def _data_esums(deb, i, ep):
    """
    The raw data sums for object i on one epoch at the converged state.
    """
    vi, ui = deb.positions[i]
    alpha, beta = get_phase_angles(
        ep, vi - ep['vcen'], ui - ep['ucen'],
    )
    sw = deb.wt_cov[i]
    sums = np.zeros(6)
    admom_ksums(
        ep['kim'], ep['iy'], ep['ix'], ep['dim'], alpha, beta,
        ep['kv'], ep['ku'], sw[0, 0], sw[0, 1], sw[1, 1],
        ep['df2'], sums,
    )
    return sums


# rows per influence-kernel call: the kernels are built on the
# padded fft grid, (rows, dim, dim) plus the complex half plane,
# so a large group with the ladder's aperture members (6 + 8 per
# object per epoch) would take tens of GB in one call (measured
# 33 GB on a wldb field); chunking bounds it at ~0.5 GB with the
# same result
_KERNEL_CHUNK = 16

# the fit's epochs are padded 4x (ngmix prep_epoch pad_factor),
# and the influence kernels inherit that grid although they are
# compact: a box of 4 sigma + 2 psf fwhm around the object holds
# > 99.99 percent of every row's energy (39-member wldb group).
# Keeping every s-th mode builds the kernel on a (dim/s)^2 grid,
# periodized with period dim/s, which is the exact kernel on the
# image wherever no wrapped copy reaches it: dim/s >= image size
# + the kernel extent.  Measured on that group at s=2: kernel
# relative error 5e-7 median, Cov(S) entries within 3e-6 of the
# full grid, irfft2 5x faster; rows whose extent does not allow
# it (the ladder's widest apertures on large objects) stay on
# the full grid, so the construction is exact by construction.
# Extent per row: KERNEL_EXTENT_SIGMA sigma + KERNEL_EXTENT_FWHM
# smoothing fwhm, conservative (5 sigma alone holds 99.998)
KERNEL_EXTENT_SIGMA = 5.0
KERNEL_EXTENT_FWHM = 3.0
_SUBSAMPLE_FACTORS = (8, 6, 5, 4, 3, 2)


def _kernels_subsampled(ep, G, shape, s):
    """
    The influence kernels on the (dim/s)^2 grid from every s-th mode.

    Equals the full-grid kernel periodized with period dim/s,
    restricted to the image, with the apodization mask applied as in
    ngmix.
    """
    dim = ep['dim']
    iy, ix = ep['iy'], ep['ix']
    D = dim // s
    keep = (iy % s == 0) & (ix % s == 0)
    A = G[:, keep] * ep['ktransfer'][keep]
    jy, jx = iy[keep] // s, ix[keep] // s
    half = D // 2 + 1
    C = np.zeros((G.shape[0], D, half), dtype=complex)
    C[:, jy, jx] = 0.5 * A
    sc = (jx == 0) | (jx == D // 2)
    if np.any(sc):
        np.add.at(
            C, (slice(None), (D - jy[sc]) % D, jx[sc]),
            0.5 * np.conj(A[:, sc]),
        )
    h = np.fft.irfft2(np.conj(C), s=(D, D), axes=(-2, -1))
    ny, nx = shape
    h = h[:, :ny, :nx] * (D ** 2 * s ** 2)
    ap_rad = float(ep.get('ap_rad', 0.0))
    if ap_rad > 0:
        from ngmix.prepsfmom import _build_square_apodization_mask
        mask = np.ones(shape)
        _build_square_apodization_mask(ap_rad, mask)
        h = h * mask
    return h


def _subsample_factor(dim, shape, extent):
    """
    The largest subsampling factor whose grid clears image plus extent.

    extent in pixels; 1 when none does.
    """
    need = max(shape) + extent
    for s in _SUBSAMPLE_FACTORS:
        if dim % s == 0 and dim // s >= need:
            return s
    return 1


def _influence_kernels_chunked(ep, G, shape, extents=None):
    """
    The real-space influence kernels of the rows of G, in chunks.

    The full-grid arrays of a large group are GBs; with extents
    (pixels per row) each row uses the coarsest exact grid.
    """
    nrows = G.shape[0]
    if extents is None:
        fac = np.ones(nrows, dtype=np.int64)
    else:
        fac = np.array([
            _subsample_factor(ep['dim'], shape, e) for e in extents
        ])
    out = np.empty((nrows,) + tuple(shape))
    for s in np.unique(fac):
        rows = np.flatnonzero(fac == s)
        for i0 in range(0, rows.size, _KERNEL_CHUNK):
            r = rows[i0:i0 + _KERNEL_CHUNK]
            if s == 1:
                out[r] = influence_kernels(ep, G[r], shape)
            else:
                out[r] = _kernels_subsampled(ep, G[r], shape, s)
    return out


def _cov_sums(deb, obs_flat, epochs, L=None, subsample=True):
    """
    The covariance of the stacked data sums.

    The 6 moment sums per object per epoch, followed (for ladder
    groups) by the aperture flux sums per ladder object, aperture and
    epoch.  Epoch-block-diagonal (independent noise per epoch), full
    cross-member within an epoch via the ngmix influence kernels; the
    aperture sums are flux sums under further weights, so they are
    extra kernel rows of the same construction.  subsample=False
    builds every kernel on the full padded grid (the referee of the
    coarser exact grids).
    """
    nobj = deb.nobj
    nep = len(epochs)
    nS0 = 6 * nobj * nep
    nap = L['nap'] if L is not None else 0
    lidx = L['idx'] if L is not None else []
    nlad = len(lidx)
    nS = nS0 + nlad * nap * nep
    covS = np.zeros((nS, nS))

    def apcol(io, j, iep):
        return nS0 + (io * nap + j) * nep + iep

    for iep, ep in enumerate(epochs):
        obs = obs_flat[iep]
        # the rows stream through the kernel builder in chunks:
        # the full (nrows, nmodes) complex stack of a large group
        # is GBs, the real-space kernels (nrows, ny, nx) are what
        # the covariance needs
        parts = []
        pend = []
        pext = []
        npend = 0
        scale = obs.jacobian.get_scale()
        pad = KERNEL_EXTENT_FWHM * 2.3548 * np.sqrt(deb.Tsmooth / 2) / scale

        def extent(W):
            sig = np.sqrt(0.5 * (W[0, 0] + W[1, 1])) / scale
            return KERNEL_EXTENT_SIGMA * sig + pad

        def flush():
            parts.append(
                _influence_kernels_chunked(
                    ep, np.vstack(pend), obs.image.shape,
                    extents=np.concatenate(pext) if subsample else None,
                )
            )
            pend.clear()
            pext.clear()

        for i in range(nobj):
            pend.append(moment_kernels(
                ep, deb.wt_cov[i],
                deb.positions[i][0] - ep['vcen'],
                deb.positions[i][1] - ep['ucen'],
            ))
            pext.append(np.full(6, extent(deb.wt_cov[i])))
            npend += 6
            if npend >= _KERNEL_CHUNK:
                flush()
                npend = 0
        for io, i in enumerate(lidx):
            v0, u0 = deb.positions[i]
            G, _ = _flux_kernels_and_dtheta_dyadic(
                ep, L['wt_covs0'][io], v0 - ep['vcen'], u0 - ep['ucen'],
            )
            pend.append(G)
            pext.append(np.array([
                extent(af * L['wt_covs0'][io]) for af in LADDER_AP_FACS
            ]))
            npend += nap
            if npend >= _KERNEL_CHUNK:
                flush()
                npend = 0
        if pend:
            flush()
        hs = np.concatenate(parts, axis=0)
        parts.clear()
        cb = sums_cov(hs, obs.weight)
        del hs
        rows = []
        for i in range(nobj):
            rows += [
                ((i * nep + iep) * 6 + a, 6 * i + a) for a in range(6)
            ]
        for io in range(nlad):
            for j in range(nap):
                rows.append((apcol(io, j, iep), 6 * nobj + io * nap + j))
        for ra, ca in rows:
            for rb, cbb in rows:
                covS[ra, rb] = cb[ca, cbb]
    return covS


def _make_patched(deb, caches, Ds, theta0s, deltas,
                  nsums_cache=None, psums_cache=None,
                  nsums_delta=None, psums_delta=None,
                  freeze_theta=False):
    """
    A _get_object_sums replacement built from linearized data sums.

    The linearized data sums plus the cheap closed-form
    neighbor/predicted sums, so Jacobian evaluations never touch
    the data modes.  With nsums_cache/psums_cache the model sums
    are not recomputed either (pure update algebra); the delta
    arguments inject perturbations for derivative evaluations, and
    freeze_theta holds the data sums at the cached values (the
    theta channel is then added analytically by the chain).
    """

    def patched(i):
        m = deb.models[i]
        is_mix = m['type'] in ('exp', 'dev', 'bdf')
        if nsums_cache is not None:
            base_nsums = nsums_cache[i]
        else:
            base_nsums = deb._get_neighbor_sums(i)
        if nsums_delta is not None and i in nsums_delta:
            base_nsums = base_nsums + nsums_delta[i]
        if is_mix:
            if psums_cache is not None:
                base_psums = psums_cache[i]
            else:
                base_psums = deb._get_predicted_sums(i)
            if psums_delta is not None and i in psums_delta:
                base_psums = base_psums + psums_delta[i]

        if freeze_theta:
            dth = np.zeros(5)
        else:
            dth = _theta_of(deb, i) - theta0s[i]
        nband = deb.nband
        sums = np.zeros(6)
        fs = np.zeros(nband)
        ws = np.zeros(nband)
        pred = np.zeros(6)
        fs_pred = np.zeros(nband)
        for iep, ep in enumerate(deb.epochs_per_obj[i]):
            esums = caches[i][iep] + Ds[i][iep] @ dth
            d = deltas.get((i, iep))
            if d is not None:
                esums = esums + d
            csums = (
                esums - base_nsums[ep['band']] / ep['detAtinv']
            )
            fac = ep['weight'] * ep['detAtinv']
            sums += fac * csums
            fs[ep['band']] += fac * csums[5]
            ws[ep['band']] += ep['weight']
            if is_mix:
                psums = base_psums[ep['band']] / ep['detAtinv']
                pred += fac * psums
                fs_pred[ep['band']] += fac * psums[5]
        return sums, fs, ws, pred, fs_pred

    return patched


def _jacobi_block(deb, snap, x, i, patched, resolve=None):
    """
    Object i's update evaluated from state x with the patched sums.

    The Jacobi map: every object from the same state.  resolve, when
    given, re-solves the ladder amps at the state (and injected data
    deltas) before the update.
    """
    _restore_state(deb, snap)
    deb._unpack_state(x)
    if resolve is not None:
        resolve()
    deb._get_object_sums = patched
    deb._update_object(i)
    out = deb._pack_state()
    del deb.__dict__['_get_object_sums']
    _restore_state(deb, snap)
    return out


def _jacobi_block_anchor(deb, snap, x, i, patched, danchor):
    """
    Object i's Jacobi update with the anchor positions perturbed.

    danchor is (nobj, 2) in arcsec.
    """
    _restore_state(deb, snap)
    det0 = deb.det_positions
    deb.det_positions = [
        (v + danchor[j, 0], u + danchor[j, 1])
        for j, (v, u) in enumerate(det0)
    ]
    deb._unpack_state(x)
    deb._get_object_sums = patched
    deb._update_object(i)
    out = deb._pack_state()
    del deb.__dict__['_get_object_sums']
    deb.det_positions = det0
    _restore_state(deb, snap)
    return out


def _dF_danchor(deb, snap, x0, patched, slices, pers):
    """
    The Jacobi-map response to the anchor positions, per arcsec.

    By central FD; the anchor enters through the recentering
    regularization and the packed center convention.
    """
    nobj = deb.nobj
    npars = x0.size
    ha = 1.0e-4
    out = np.zeros((npars, 2 * nobj))
    for ja in range(2 * nobj):
        dp = np.zeros((nobj, 2))
        dp[ja // 2, ja % 2] = ha
        for i in range(nobj):
            sl = slice(slices[i], slices[i] + pers[i])
            bp = _jacobi_block_anchor(
                deb, snap, x0, i, patched, dp,
            )
            bm = _jacobi_block_anchor(
                deb, snap, x0, i, patched, -dp,
            )
            out[sl, ja] = (bp[sl] - bm[sl]) / (2 * ha)
    return out


def _column_map(deb):
    """
    The (object, kind, sub) description of every packed column.

    kind is one of 'F' (sub = band), 'cen' (sub 0 = v, 1 = u), 'cov'
    (sub = 00, 01, 11 component), 'fracdev', 'sw'.
    """
    from .deblender import _adaptive

    cols = []
    for k, m in enumerate(deb.models):
        cols += [(k, 'F', b) for b in range(deb.nband)]
        if deb.recenter:
            cols += [(k, 'cen', 0), (k, 'cen', 1)]
        if not _adaptive(m):
            continue
        cols += [(k, 'cov', c) for c in range(3)]
        if m['type'] == 'bdf':
            cols += [(k, 'fracdev', 0)]
        cols += [(k, 'sw', c) for c in range(3)]
    return cols


def _covkey(m):
    return 'cov_sm' if m['type'] in ('gauss', 'ladder') else 'cov'


_SYM = [(0, 0), (0, 1), (1, 1)]


def _pair_sums(deb, i, j, wt_cov=None):
    """
    The sums of model j alone under object i's weight.

    Or under the given weight, at detAtinv=1: one term of
    _get_neighbor_sums.
    """
    from .ladder import band_comps
    vi, ui = deb.positions[i]
    if wt_cov is None:
        wt_cov = deb.wt_cov[i]
    Fb, cov_sm00, cov_sm01, cov_sm11 = band_comps(deb.models[j], deb.Tsmooth)
    pj = deb.positions[j]
    dv = np.full(cov_sm00.size, pj[0] - vi)
    du = np.full(cov_sm00.size, pj[1] - ui)
    out = np.zeros((deb.nband, 6))
    for band in range(deb.nband):
        gauss_comps_ksums(
            np.ascontiguousarray(Fb[band]), cov_sm00, cov_sm01, cov_sm11, dv,
            du, wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], 1.0, out[band],
        )
    return out


def _model_sum_derivs(deb, cols):
    """
    Closed-form model-sum derivatives by micro central FD.

    dNS[(i, col)] and dPS[(i, col)] as (nband, 6) arrays per affected
    object, in physical units per unit physical change of the column
    quantity.  The FD is taken on the perturbed object's own pair
    term wherever it is a neighbor (the sums are additive, so the
    difference of the full sets is the difference of the pair,
    exactly): O(nobj^2) small kernel calls per group instead of
    O(nobj^3) model expansions (on a 39-member field the full form
    called band_comps 2.2 million times, 15-18 s for either model).
    The full set is only differenced under the object's own weight
    or center.  _model_sum_derivs_full is the full-set referee.
    """
    nobj = deb.nobj
    nband = deb.nband
    Tw = deb.wt_cov[0][0, 0] + deb.wt_cov[0][1, 1]
    h_cov = 1.0e-6 * max(Tw, 0.1)
    h_pos = 1.0e-6

    dNS = {}
    dPS = {}

    def ns(i, wt_cov=None):
        return deb._get_neighbor_sums(i, wt_cov=wt_cov)

    def pair(i, k):
        return _pair_sums(deb, i, k)

    for ic, (k, kind, sub) in enumerate(cols):
        m = deb.models[k]
        if kind == 'sw':
            r, c = _SYM[sub]
            swp = deb.wt_cov[k].copy()
            swm = deb.wt_cov[k].copy()
            swp[r, c] += h_cov
            swp[c, r] = swp[r, c]
            swm[r, c] -= h_cov
            swm[c, r] = swm[r, c]
            if m['type'] == 'ladder':
                rungs0 = m['rungs']
                m['rungs'] = ladder_rung_covs(swp, deb.Tsmooth)
                nsp = [ns(k, wt_cov=swp) if i == k else pair(i, k)
                       for i in range(nobj)]
                m['rungs'] = ladder_rung_covs(swm, deb.Tsmooth)
                nsm = [ns(k, wt_cov=swm) if i == k else pair(i, k)
                       for i in range(nobj)]
                m['rungs'] = rungs0
                for i in range(nobj):
                    d = (nsp[i] - nsm[i]) / (2 * h_cov)
                    if np.any(d != 0):
                        dNS[(i, ic)] = d
            else:
                dNS[(k, ic)] = (
                    ns(k, wt_cov=swp) - ns(k, wt_cov=swm)
                ) / (2 * h_cov)
            if m['type'] in ('exp', 'dev', 'bdf'):
                sw0 = deb.wt_cov[k]
                deb.wt_cov[k] = swp
                psp = deb._get_predicted_sums(k)
                deb.wt_cov[k] = swm
                psm = deb._get_predicted_sums(k)
                deb.wt_cov[k] = sw0
                dPS[(k, ic)] = (psp - psm) / (2 * h_cov)
        elif kind == 'cen':
            pos0 = deb.positions[k]
            for i in range(nobj):
                pp = list(pos0)
                pp[sub] += h_pos
                deb.positions[k] = tuple(pp)
                nsp = ns(k) if i == k else pair(i, k)
                pp[sub] -= 2 * h_pos
                deb.positions[k] = tuple(pp)
                nsm = ns(k) if i == k else pair(i, k)
                deb.positions[k] = pos0
                d = (nsp - nsm) / (2 * h_pos)
                if np.any(d != 0):
                    dNS[(i, ic)] = d
        elif kind == 'F':
            F0 = m['F'].copy()
            for i in range(nobj):
                if i == k:
                    continue
                if m['type'] == 'ladder':
                    # the ladder's sums are its amps, not F
                    dNS[(i, ic)] = np.zeros((nband, 6))
                    continue
                m['F'] = F0.copy()
                m['F'][sub] += 1.0
                nsp = pair(i, k)
                m['F'] = F0.copy()
                m['F'][sub] -= 1.0
                nsm = pair(i, k)
                m['F'] = F0
                dNS[(i, ic)] = (nsp - nsm) / 2.0
            if m['type'] in ('exp', 'dev', 'bdf'):
                m['F'] = F0.copy()
                m['F'][sub] += 1.0
                psp = deb._get_predicted_sums(k)
                m['F'] = F0.copy()
                m['F'][sub] -= 1.0
                psm = deb._get_predicted_sums(k)
                m['F'] = F0
                dPS[(k, ic)] = (psp - psm) / 2.0
        elif kind == 'cov':
            key = _covkey(m)
            r, c = _SYM[sub]
            cov0 = m[key].copy()
            covp = cov0.copy()
            covp[r, c] += h_cov
            covp[c, r] = covp[r, c]
            covm = cov0.copy()
            covm[r, c] -= h_cov
            covm[c, r] = covm[r, c]
            for i in range(nobj):
                if i == k:
                    continue
                m[key] = covp
                nsp = pair(i, k)
                m[key] = covm
                nsm = pair(i, k)
                m[key] = cov0
                dNS[(i, ic)] = (nsp - nsm) / (2 * h_cov)
            if m['type'] in ('exp', 'dev', 'bdf'):
                m[key] = covp
                psp = deb._get_predicted_sums(k)
                m[key] = covm
                psm = deb._get_predicted_sums(k)
                m[key] = cov0
                dPS[(k, ic)] = (psp - psm) / (2 * h_cov)
    return dNS, dPS


def _model_sum_derivs_full(deb, cols):
    """
    The full-set referee of _model_sum_derivs.

    Closed-form model-sum derivatives by micro central FD on the
    (cheap, analytic) neighbor and predicted sum functions of the
    whole neighbor set: dNS[(i, col)] and dPS[(i, col)] as (nband, 6)
    arrays per affected object, in physical units per unit physical
    change of the column quantity.
    """
    nobj = deb.nobj
    Tw = deb.wt_cov[0][0, 0] + deb.wt_cov[0][1, 1]
    h_cov = 1.0e-6 * max(Tw, 0.1)
    h_pos = 1.0e-6

    dNS = {}
    dPS = {}

    def ns(i, wt_cov=None):
        return deb._get_neighbor_sums(i, wt_cov=wt_cov)

    for ic, (k, kind, sub) in enumerate(cols):
        m = deb.models[k]
        if kind == 'sw':
            r, c = _SYM[sub]
            swp = deb.wt_cov[k].copy()
            swm = deb.wt_cov[k].copy()
            swp[r, c] += h_cov
            swp[c, r] = swp[r, c]
            swm[r, c] -= h_cov
            swm[c, r] = swm[r, c]
            if m['type'] == 'ladder':
                # the weight sets the rungs too: at fixed amps
                # every neighbor of k sees them (the amps'
                # own response is the analytic chain)
                rungs0 = m['rungs']
                m['rungs'] = ladder_rung_covs(swp, deb.Tsmooth)
                nsp = [ns(i, wt_cov=swp if i == k else None)
                       for i in range(nobj)]
                m['rungs'] = ladder_rung_covs(swm, deb.Tsmooth)
                nsm = [ns(i, wt_cov=swm if i == k else None)
                       for i in range(nobj)]
                m['rungs'] = rungs0
                for i in range(nobj):
                    d = (nsp[i] - nsm[i]) / (2 * h_cov)
                    if np.any(d != 0):
                        dNS[(i, ic)] = d
            else:
                # only object k's own sums use its weight
                dNS[(k, ic)] = (
                    ns(k, wt_cov=swp) - ns(k, wt_cov=swm)
                ) / (2 * h_cov)
            if m['type'] in ('exp', 'dev', 'bdf'):
                sw0 = deb.wt_cov[k]
                deb.wt_cov[k] = swp
                psp = deb._get_predicted_sums(k)
                deb.wt_cov[k] = swm
                psm = deb._get_predicted_sums(k)
                deb.wt_cov[k] = sw0
                dPS[(k, ic)] = (psp - psm) / (2 * h_cov)
        elif kind == 'cen':
            # object k's own weighted sums move with its center,
            # and it moves as a neighbor of every other object
            pos0 = deb.positions[k]
            for i in range(nobj):
                pp = list(pos0)
                pp[sub] += h_pos
                deb.positions[k] = tuple(pp)
                nsp = ns(i)
                pp[sub] -= 2 * h_pos
                deb.positions[k] = tuple(pp)
                nsm = ns(i)
                deb.positions[k] = pos0
                d = (nsp - nsm) / (2 * h_pos)
                if np.any(d != 0):
                    dNS[(i, ic)] = d
            # predicted sums are evaluated at the object's own
            # center, so they do not move with it
        elif kind == 'F':
            F0 = m['F'].copy()
            for i in range(nobj):
                if i == k:
                    continue
                m['F'] = F0.copy()
                m['F'][sub] += 1.0
                nsp = ns(i)
                m['F'] = F0.copy()
                m['F'][sub] -= 1.0
                nsm = ns(i)
                m['F'] = F0
                dNS[(i, ic)] = (nsp - nsm) / 2.0
            if m['type'] in ('exp', 'dev', 'bdf'):
                m['F'] = F0.copy()
                m['F'][sub] += 1.0
                psp = deb._get_predicted_sums(k)
                m['F'] = F0.copy()
                m['F'][sub] -= 1.0
                psm = deb._get_predicted_sums(k)
                m['F'] = F0
                dPS[(k, ic)] = (psp - psm) / 2.0
        elif kind == 'cov':
            key = _covkey(m)
            r, c = _SYM[sub]
            cov0 = m[key].copy()
            covp = cov0.copy()
            covp[r, c] += h_cov
            covp[c, r] = covp[r, c]
            covm = cov0.copy()
            covm[r, c] -= h_cov
            covm[c, r] = covm[r, c]
            for i in range(nobj):
                if i == k:
                    continue
                m[key] = covp
                nsp = ns(i)
                m[key] = covm
                nsm = ns(i)
                m[key] = cov0
                dNS[(i, ic)] = (nsp - nsm) / (2 * h_cov)
            if m['type'] in ('exp', 'dev', 'bdf'):
                m[key] = covp
                psp = deb._get_predicted_sums(k)
                m[key] = covm
                psm = deb._get_predicted_sums(k)
                m[key] = cov0
                dPS[(k, ic)] = (psp - psm) / (2 * h_cov)
        # 'fracdev' unused: bdf is excluded by the type guard
    return dNS, dPS


def _gauss_flux_covs(deb, caches, Ds, dNS, cols, slices,
                     epochs, Tx, covS, Ra, anchor_cov,
                     dNS_modes=None):
    """
    The gauss-estimator fluxes and their cross-band covariance per object.

    F_b = 4 pi sqrt(det wt_cov) fs_b / ws_b and its (nband, nband)
    covariance, as {'gauss_flux', 'gauss_flux_cov'} lists with None
    entries for stars.  fs_b is linear in the data at the converged
    state, so the response is the direct data channel plus the chain
    through the state: the own kernel (weight and center) via the
    analytic data-sum derivatives, the neighbor subtraction via the
    model-sum derivatives, and the explicit sqrt(det wt_cov)
    normalization; the anchor response is added when present.  The
    per-epoch weights ws_b are fixed at prep, so they carry no state
    dependence.
    """
    nobj = deb.nobj
    nband = deb.nband
    nep = len(epochs)
    npars = Tx.shape[0]
    scales = deb.scales

    ws = np.zeros(nband)
    facs = np.zeros(nep)
    for iep, ep in enumerate(epochs):
        ws[ep['band']] += ep['weight']
        facs[iep] = ep['weight'] * ep['detAtinv']

    fluxes = []
    covs = []
    for i in range(nobj):
        wt_cov = deb.wt_cov[i]
        detS = wt_cov[0, 0] * wt_cov[1, 1] - wt_cov[0, 1] ** 2
        if deb.models[i]['type'] == 'star' or detS <= 0:
            fluxes.append(None)
            covs.append(None)
            continue
        cnorm = 4.0 * np.pi * np.sqrt(detS)

        # fs_b = sum_ep fac esums_5 - ws_b NS_b5 since the
        # fac / detAtinv on the neighbor term is the epoch weight
        nsums = deb._get_neighbor_sums(i)
        fs = -ws * nsums[:, 5]
        Rd = np.zeros((nband, Tx.shape[1]))
        for iep, ep in enumerate(epochs):
            b = ep['band']
            fs[b] += facs[iep] * caches[i][iep][5]
            Rd[b, (i * nep + iep) * 6 + 5] += (
                cnorm * facs[iep] / ws[b]
            )
        F = cnorm * fs / ws
        if dNS_modes is not None:
            # the aperture and T-row data modes move the
            # neighbor subtraction through the re-solved amps
            for (k, col), dm in dNS_modes.items():
                if k == i:
                    Rd[:, col] -= cnorm * dm[:, 5]

        # d ln cnorm / d(sw3)
        dlnc = np.array([
            wt_cov[1, 1], -2.0 * wt_cov[0, 1], wt_cov[0, 0],
        ]) / (2.0 * detS)

        gx = np.zeros((nband, npars))
        for ic, (k, kind, sub) in enumerate(cols):
            row = np.zeros(nband)
            if k == i and kind in ('sw', 'cen'):
                tcol = sub if kind == 'sw' else 3 + sub
                for iep, ep in enumerate(epochs):
                    b = ep['band']
                    row[b] += (
                        cnorm * facs[iep] *
                        Ds[i][iep][5, tcol] / ws[b]
                    )
                if kind == 'sw':
                    row += F * dlnc[sub]
            d = dNS.get((i, ic))
            if d is not None:
                row -= cnorm * d[:, 5]
            gx[:, ic] = row * scales[ic]

        Rg = Rd + gx @ Tx
        fcov = Rg @ covS @ Rg.T
        if Ra is not None:
            # the anchor moves the actual position 1:1 at fixed
            # packed offset, so the direct channel is the
            # physical center row
            ga = gx @ Ra
            for ic, (k, kind, sub) in enumerate(cols):
                if kind == 'cen':
                    ga[:, 2 * k + sub] += gx[:, ic] / scales[ic]
            fcov = fcov + ga @ anchor_cov @ ga.T
        fluxes.append(F)
        covs.append(fcov)
    return {'gauss_flux': fluxes, 'gauss_flux_cov': covs}


def _chain_pieces(deb, snap, x0, caches, Ds, theta0s, slices,
                  pers, covS, epochs, L=None, W2=None, s_star=None):
    """
    The Jacobi Jacobian, data response and anchor response by the chain rule.

    Micro finite differences of the pure update algebra (all sums
    cached) chained with the analytic data-sum derivatives and the
    closed-form model-sum derivatives.  Structural zeros (a member's
    update does not depend on other members' weights) are never
    evaluated.
    """
    nobj = deb.nobj
    npars = x0.size
    nep = len(epochs)
    cols = _column_map(deb)
    scales = deb.scales

    nsums_cache = [
        deb._get_neighbor_sums(i) for i in range(nobj)
    ]
    psums_cache = [
        deb._get_predicted_sums(i)
        if deb.models[i]['type'] in ('exp', 'dev', 'bdf')
        else None
        for i in range(nobj)
    ]

    # A/B/P/C per object: the hand-differentiated update
    # algebra where the evaluation at the solution is on the
    # healthy branch, micro finite differences otherwise
    dsteps = DS_FAC * np.sqrt(np.diag(covS))
    A = []
    B = []
    P = []
    C = []
    n_fallback = 0
    for i in range(nobj):
        inputs = _base_inputs(
            deb, i, caches, nsums_cache, psums_cache, epochs,
        )
        phi = _phi_healthy(deb, i, *inputs)
        if phi is not None:
            Ai, Bi, Pi, Ci = _analytic_ABPC(
                deb, i, phi, epochs, slices, pers,
            )
        else:
            n_fallback += 1
            Ai, Bi, Pi, Ci = None, None, None, None
        A.append(Ai)
        B.append(Bi)
        P.append(Pi)
        C.append(Ci)

    dFdS = np.zeros((npars, covS.shape[0]))
    for i in range(nobj):
        if A[i] is not None:
            sl = slice(slices[i], slices[i] + pers[i])
            dFdS[sl, i * nep * 6:(i + 1) * nep * 6] = A[i]
            continue
        sl = slice(slices[i], slices[i] + pers[i])
        Ai = np.zeros((pers[i], nep * 6))
        for iep in range(nep):
            for a in range(6):
                col = (i * nep + iep) * 6 + a
                h = dsteps[col]
                if h == 0:
                    # zero-variance sum: covS row and column are
                    # exactly zero, so the column is irrelevant;
                    # see the matching guard in _fd_dFdS
                    continue
                d = np.zeros(6)
                d[a] = h
                patched_p = _make_patched(
                    deb, caches, Ds, theta0s, {(i, iep): d},
                    nsums_cache=nsums_cache,
                    psums_cache=psums_cache, freeze_theta=True,
                )
                patched_m = _make_patched(
                    deb, caches, Ds, theta0s, {(i, iep): -d},
                    nsums_cache=nsums_cache,
                    psums_cache=psums_cache, freeze_theta=True,
                )
                bp = _jacobi_block(deb, snap, x0, i, patched_p)
                bm = _jacobi_block(deb, snap, x0, i, patched_m)
                Ai[:, iep * 6 + a] = (
                    bp[sl] - bm[sl]
                ) / (2 * h)
        A[i] = Ai
        dFdS[sl, i * nep * 6:(i + 1) * nep * 6] = Ai

    # B_i / P_i fallbacks
    nband = deb.nband
    for i in range(nobj):
        if B[i] is not None:
            continue
        sl = slice(slices[i], slices[i] + pers[i])
        hb = 1.0e-3 * max(
            np.max(np.abs(nsums_cache[i])), 1.0,
        )
        Bi = np.zeros((pers[i], nband * 6))
        for band in range(nband):
            for a in range(6):
                d = np.zeros((nband, 6))
                d[band, a] = hb
                patched_p = _make_patched(
                    deb, caches, Ds, theta0s, {},
                    nsums_cache=nsums_cache,
                    psums_cache=psums_cache, freeze_theta=True,
                    nsums_delta={i: d},
                )
                patched_m = _make_patched(
                    deb, caches, Ds, theta0s, {},
                    nsums_cache=nsums_cache,
                    psums_cache=psums_cache, freeze_theta=True,
                    nsums_delta={i: -d},
                )
                bp = _jacobi_block(deb, snap, x0, i, patched_p)
                bm = _jacobi_block(deb, snap, x0, i, patched_m)
                Bi[:, band * 6 + a] = (
                    bp[sl] - bm[sl]
                ) / (2 * hb)
        B[i] = Bi
        if psums_cache[i] is not None:
            hp = 1.0e-3 * max(
                np.max(np.abs(psums_cache[i])), 1.0,
            )
            Pi = np.zeros((pers[i], nband * 6))
            for band in range(nband):
                for a in range(6):
                    d = np.zeros((nband, 6))
                    d[band, a] = hp
                    patched_p = _make_patched(
                        deb, caches, Ds, theta0s, {},
                        nsums_cache=nsums_cache,
                        psums_cache=psums_cache,
                        freeze_theta=True,
                        psums_delta={i: d},
                    )
                    patched_m = _make_patched(
                        deb, caches, Ds, theta0s, {},
                        nsums_cache=nsums_cache,
                        psums_cache=psums_cache,
                        freeze_theta=True,
                        psums_delta={i: -d},
                    )
                    bp = _jacobi_block(
                        deb, snap, x0, i, patched_p,
                    )
                    bm = _jacobi_block(
                        deb, snap, x0, i, patched_m,
                    )
                    Pi[:, band * 6 + a] = (
                        bp[sl] - bm[sl]
                    ) / (2 * hp)
            P[i] = Pi
        else:
            P[i] = None

    # C_i fallbacks: direct dependence of the update algebra on
    # the object's own packed state, all sums frozen
    patched_frozen = _make_patched(
        deb, caches, Ds, theta0s, {},
        nsums_cache=nsums_cache, psums_cache=psums_cache,
        freeze_theta=True,
    )
    for i in range(nobj):
        if C[i] is not None:
            continue
        sl = slice(slices[i], slices[i] + pers[i])
        Ci = np.zeros((pers[i], pers[i]))
        for jl in range(pers[i]):
            j = slices[i] + jl
            xp = x0.copy()
            xm = x0.copy()
            xp[j] += FD_H
            xm[j] -= FD_H
            bp = _jacobi_block(deb, snap, xp, i, patched_frozen)
            bm = _jacobi_block(deb, snap, xm, i, patched_frozen)
            Ci[:, jl] = (bp[sl] - bm[sl]) / (2 * FD_H)
        C[i] = Ci

    dNSm = None
    lstate = None
    if L is None:
        dNS, dPS = _model_sum_derivs(deb, cols)
    else:
        # the ladder re-solve gives the complete neighbor-sum
        # state dependence (non-ladder members' direct terms,
        # the rung frames and the amps); the predicted-sum
        # derivatives of any mixture members still come from
        # the closed-form micro-FD
        dNS_direct, dPS = _model_sum_derivs(deb, cols)
        dNS, dNSm, lstate = _ladder_derivs(
            deb, snap, x0, L, caches, Ds, theta0s, cols, covS,
            epochs, W2, s_star, dNS_direct=dNS_direct,
        )
        # the aperture and T-row data modes reach every update
        # through the re-solved amps in the neighbor sums
        for (k, col), d in dNSm.items():
            slk = slice(slices[k], slices[k] + pers[k])
            dFdS[slk, col] += B[k] @ d.ravel()

    # assemble J in the normalized packed units
    J = np.zeros((npars, npars))
    for ic, (k, kind, sub) in enumerate(cols):
        sc = scales[ic]
        # own-block direct dependence
        slk = slice(slices[k], slices[k] + pers[k])
        J[slk, ic] += C[k][:, ic - slices[k]]
        # data-sum channel: only the weight and center of k
        if kind in ('sw', 'cen'):
            tcol = sub if kind == 'sw' else 3 + sub
            desums = np.concatenate([
                Ds[k][iep][:, tcol] for iep in range(nep)
            ])
            J[slk, ic] += A[k] @ (desums * sc)
        # model-sum channels
        for i in range(nobj):
            d = dNS.get((i, ic))
            if d is not None:
                sli = slice(slices[i], slices[i] + pers[i])
                J[sli, ic] += B[i] @ (d.ravel() * sc)
        d = dPS.get((k, ic))
        if d is not None and P[k] is not None:
            J[slk, ic] += P[k] @ (d.ravel() * sc)

    # anchor response: the anchor moves object k's position 1:1
    # (the packed center offset and the regularization pull are
    # both anchor-relative), so only the sums channels respond
    dFda = None
    if deb.recenter:
        dFda = np.zeros((npars, 2 * nobj))
        cen_cols = {
            (k, sub): ic
            for ic, (k, kind, sub) in enumerate(cols)
            if kind == 'cen'
        }
        for k in range(nobj):
            for sub in range(2):
                ja = 2 * k + sub
                slk = slice(slices[k], slices[k] + pers[k])
                tcol = 3 + sub
                desums = np.concatenate([
                    Ds[k][iep][:, tcol] for iep in range(nep)
                ])
                dFda[slk, ja] += A[k] @ desums
                ic = cen_cols[(k, sub)]
                for i in range(nobj):
                    d = dNS.get((i, ic))
                    if d is not None:
                        sli = slice(
                            slices[i], slices[i] + pers[i],
                        )
                        dFda[sli, ja] += B[i] @ d.ravel()
    return J, dFdS, dFda, dNS, dNSm, lstate


# ---------------------------------------------------------------
# hand-differentiated update algebra: the healthy-branch
# derivatives of one object update with respect to its inputs
# (sums, fs, pred, fs_pred) and its own state, replacing the
# A/B/P/C micro finite differences.  Objects whose evaluation at
# the solution takes a guarded branch (failed deweight, damped or
# rejected step, active recenter clip) fall back to the micro-FD
# construction; the micro-FD path is also the referee in the
# tests
# ---------------------------------------------------------------


def _dmm_dsums(sums):
    """
    The moment matrix in sym3 and its derivative with respect to the raw sums.

    The derivative is (3, 6).
    """
    finv = 1.0 / sums[5]
    m = 0.5 * finv * np.array([
        sums[4] - sums[2], sums[3], sums[4] + sums[2],
    ])
    D = np.zeros((3, 6))
    D[0, 2], D[0, 4] = -0.5 * finv, 0.5 * finv
    D[1, 3] = 0.5 * finv
    D[2, 2], D[2, 4] = 0.5 * finv, 0.5 * finv
    D[:, 5] = -m * finv
    return m, D


def _ddetsqrt(A):
    """d sqrt(det A) / d(a00, a01, a11)"""
    det = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    sq = np.sqrt(det)
    return np.array([
        A[1, 1], -2 * A[0, 1], A[0, 0],
    ]) / (2 * sq), sq


def _phi_healthy(deb, i, sums, fs, ws, pred, fs_pred):
    """
    The analytic update-algebra derivatives for object i.

    At the given converged inputs, or None when the evaluation takes
    a guarded branch.  Output rows follow the packed block layout
    [F, (cen), cov, wt_cov] in PHYSICAL units; the caller applies the
    packing scales.
    """
    from .deblender import (
        RECENTER_CLIP_FAC, ZERO_WEIGHT, mixture_model_valid,
    )

    m = deb.models[i]
    mtype = m['type']
    if mtype not in ('gauss', 'exp', 'dev', 'ladder'):
        return None
    if not (sums[5] > 0 and sums[4] > 0):
        return None

    nband = deb.nband
    rc = deb.recenter
    per = nband + (2 if rc else 0) + 6
    nrow = per

    wt_cov_old = deb.wt_cov[i]
    Mm, dMm = _dmm_dsums(sums)
    new_wt_cov, DWm_M, DWm_S, ok = _dw_derivs(
        _sym3_mat(Mm), wt_cov_old,
    )
    if not ok:
        return None

    # rows: F [0:nband], cen [nband:nband+2] if rc, cov, wt_cov
    icen = nband
    icov = nband + (2 if rc else 0)
    isw = icov + 3

    dsums = np.zeros((nrow, 6))
    dfs = np.zeros((nrow, nband))
    dpred = np.zeros((nrow, 6))
    dfsp = np.zeros((nrow, nband))
    dFold = np.zeros((nrow, nband))
    dcovold = np.zeros((nrow, 3))
    dswold = np.zeros((nrow, 3))
    dpos = np.zeros((nrow, 2))

    # the weight rows: new_wt_cov = DW(Mm, wt_cov_old) for every type
    dsums[isw:isw + 3] = DWm_M @ dMm
    dswold[isw:isw + 3] = DWm_S

    if mtype in ('gauss', 'ladder'):
        # cov_sm = wt_cov = new_wt_cov (the ladder's weight/flux update is
        # the gauss path; its amps are not state, see
        # _ladder_setup)
        dsums[icov:icov + 3] = dsums[isw:isw + 3]
        dswold[icov:icov + 3] = dswold[isw:isw + 3]
        # matched flux: F_b = fs_b / ws_b * 2 pi sqrt(det(wt_cov_old
        # + new_wt_cov))
        dd, sq = _ddetsqrt(wt_cov_old + new_wt_cov)
        fac = 2.0 * np.pi * sq / ws
        for b in range(nband):
            dfs[b, b] = fac[b]
            pref = fs[b] / ws[b] * 2.0 * np.pi
            # through new_wt_cov (sums, wt_cov_old) and directly wt_cov_old
            dsums[b] = pref * (
                dd @ dsums[isw:isw + 3]
            )
            dswold[b] = pref * (
                dd @ (dswold[isw:isw + 3] + np.eye(3))
            )
    else:
        # exp/dev mixture: shift = new_wt_cov - DW(Mp) (main branch)
        # or the ratio fallback; prop = cov_old + shift
        Mp, dMp = _dmm_dsums(pred)
        # Sp: the deweight image of the predicted moments
        Sp, DWp_M, DWp_S, pok = _dw_derivs(
            _sym3_mat(Mp), wt_cov_old,
        )
        if pok:
            dcov_dsums = DWm_M @ dMm
            dcov_dpred = -(DWp_M @ dMp)
            dcov_dswold = DWm_S - DWp_S
            shift3 = np.array([
                new_wt_cov[0, 0] - Sp[0, 0], new_wt_cov[0, 1] - Sp[0, 1],
                new_wt_cov[1, 1] - Sp[1, 1],
            ])
        else:
            # gain-1 ratio fallback
            fam_cov = m['cov']
            Tp = pred[4] / pred[5]
            Tf = fam_cov[0, 0] + fam_cov[1, 1]
            fac = sums[4] / sums[5] / Tp
            de1 = sums[2] / sums[4] - pred[2] / pred[4]
            de2 = sums[3] / sums[4] - pred[3] / pred[4]
            base = 0.5 * fac * Tf
            shift3 = np.array([
                (fac - 1) * fam_cov[0, 0] - base * de1,
                (fac - 1) * fam_cov[0, 1] + base * de2,
                (fac - 1) * fam_cov[1, 1] + base * de1,
            ])
            sgn = np.array([-1.0, 1.0, 1.0])
            # d fac
            dfac_ds = np.zeros(6)
            dfac_ds[4] = 1.0 / sums[5] / Tp
            dfac_ds[5] = -fac / sums[5]
            dfac_dp = np.zeros(6)
            dfac_dp[4] = -fac / pred[4]
            dfac_dp[5] = fac / pred[5]
            # d de1, de2
            dde1_ds = np.zeros(6)
            dde1_ds[2] = 1.0 / sums[4]
            dde1_ds[4] = -sums[2] / sums[4] ** 2
            dde2_ds = np.zeros(6)
            dde2_ds[3] = 1.0 / sums[4]
            dde2_ds[4] = -sums[3] / sums[4] ** 2
            dde1_dp = np.zeros(6)
            dde1_dp[2] = -1.0 / pred[4]
            dde1_dp[4] = pred[2] / pred[4] ** 2
            dde2_dp = np.zeros(6)
            dde2_dp[3] = -1.0 / pred[4]
            dde2_dp[4] = pred[3] / pred[4] ** 2
            Sf3 = np.array([
                fam_cov[0, 0], fam_cov[0, 1], fam_cov[1, 1],
            ])
            dde_ds = [dde1_ds, dde2_ds, dde1_ds]
            dde_dp = [dde1_dp, dde2_dp, dde1_dp]
            dcov_dsums = np.zeros((3, 6))
            dcov_dpred = np.zeros((3, 6))
            for r in range(3):
                dcov_dsums[r] = (
                    Sf3[r] * dfac_ds
                    + sgn[r] * 0.5 * Tf * (
                        de1 if r != 1 else de2
                    ) * dfac_ds
                    + sgn[r] * base * dde_ds[r]
                )
                dcov_dpred[r] = (
                    Sf3[r] * dfac_dp
                    + sgn[r] * 0.5 * Tf * (
                        de1 if r != 1 else de2
                    ) * dfac_dp
                    + sgn[r] * base * dde_dp[r]
                )
            dcov_dswold = np.zeros((3, 3))
            # cov_old direct: (fac-1) I + the Tf trace channel
            dcov_dcovold_extra = np.zeros((3, 3))
            for r in range(3):
                dcov_dcovold_extra[r, r] += fac - 1.0
                tr_d = np.array([1.0, 0.0, 1.0])
                dcov_dcovold_extra[r] += (
                    sgn[r] * 0.5 * fac
                    * (de1 if r != 1 else de2) * tr_d
                )

        # damping/validity at the converged point
        prop = m['cov'] + _sym3_mat(shift3)
        if not mixture_model_valid(
            m, prop, ZERO_WEIGHT, deb.Tsmooth,
        ):
            return None

        dcov = slice(icov, icov + 3)
        dsums[dcov] = dcov_dsums
        dpred[dcov] = dcov_dpred
        dswold[dcov] = dcov_dswold
        dcovold[dcov] = np.eye(3)
        if not pok:
            dcovold[dcov] += dcov_dcovold_extra

        # flux ratio update
        for b in range(nband):
            if fs_pred[b] == 0:
                return None
            dfs[b, b] = m['F'][b] / fs_pred[b]
            dfsp[b, b] = -m['F'][b] * fs[b] / fs_pred[b] ** 2
            dFold[b, b] = fs[b] / fs_pred[b]

    if rc and not deb.fixcen[i]:
        s0 = deb.cen_sigma0
        sig = deb._cen_sigma_sweep[i]
        if not np.isfinite(sig) or (s0 ** 2 + sig ** 2) == 0:
            return None
        k = s0 ** 2 / (s0 ** 2 + sig ** 2)
        pull = sums[0:2] / sums[5]
        v, u = deb.positions[i]
        v0, u0 = deb.det_positions[i]
        newv = v + k * pull[0] + (1.0 - k) * (v0 - v)
        newu = u + k * pull[1] + (1.0 - k) * (u0 - u)
        clip = RECENTER_CLIP_FAC * np.sqrt(deb.Tsmooth)
        if np.hypot(newv - v0, newu - u0) > clip:
            return None
        for c in range(2):
            dsums[icen + c, c] = k / sums[5]
            dsums[icen + c, 5] = -k * pull[c] / sums[5]
            dpos[icen + c, c] = k
    elif rc:
        # fixed center: the packed offset does not move
        for c in range(2):
            dpos[icen + c, c] = 0.0

    return {
        'dsums': dsums, 'dfs': dfs, 'dpred': dpred,
        'dfsp': dfsp, 'dFold': dFold, 'dcovold': dcovold,
        'dswold': dswold, 'dpos': dpos, 'per': per,
        'icen': icen, 'icov': icov, 'isw': isw,
    }


def _base_inputs(deb, i, caches, nsums_cache, psums_cache,
                 epochs):
    """
    The converged (sums, fs, ws, pred, fs_pred) for object i.

    From the cached raw data and model sums, replicating the
    _get_object_sums accumulation.
    """
    nband = deb.nband
    is_mix = deb.models[i]['type'] in ('exp', 'dev', 'bdf')
    sums = np.zeros(6)
    fs = np.zeros(nband)
    ws = np.zeros(nband)
    pred = np.zeros(6)
    fs_pred = np.zeros(nband)
    for iep, ep in enumerate(epochs):
        esums = caches[i][iep]
        csums = (
            esums - nsums_cache[i][ep['band']] / ep['detAtinv']
        )
        fac = ep['weight'] * ep['detAtinv']
        sums += fac * csums
        fs[ep['band']] += fac * csums[5]
        ws[ep['band']] += ep['weight']
        if is_mix:
            psums = psums_cache[i][ep['band']] / ep['detAtinv']
            pred += fac * psums
            fs_pred[ep['band']] += fac * psums[5]
    return sums, fs, ws, pred, fs_pred


def _analytic_ABPC(deb, i, phi, epochs, slices, pers):
    """
    Map the physical update-algebra derivatives to the packed matrices.

    A (data sums), B (neighbor sums), P (predicted sums) and C (own
    state), in the packed-normalized units.
    """
    nband = deb.nband
    nep = len(epochs)
    per = pers[i]
    i0 = slices[i]
    rs = deb.scales[i0:i0 + per]

    dsums = phi['dsums']
    dfs = phi['dfs']
    dpred = phi['dpred']
    dfsp = phi['dfsp']

    Ai = np.zeros((per, nep * 6))
    wband = np.zeros(nband)
    for iep, ep in enumerate(epochs):
        fac = ep['weight'] * ep['detAtinv']
        band = ep['band']
        wband[band] += ep['weight']
        for a in range(6):
            col = dsums[:, a] * fac
            if a == 5:
                col = col + dfs[:, band] * fac
            Ai[:, iep * 6 + a] = col / rs

    Bi = np.zeros((per, nband * 6))
    Pi = (
        np.zeros((per, nband * 6))
        if deb.models[i]['type'] in ('exp', 'dev') else None
    )
    for band in range(nband):
        for a in range(6):
            col = -wband[band] * dsums[:, a]
            if a == 5:
                col = col - wband[band] * dfs[:, band]
            Bi[:, band * 6 + a] = col / rs
            if Pi is not None:
                colp = wband[band] * dpred[:, a]
                if a == 5:
                    colp = colp + wband[band] * dfsp[:, band]
                Pi[:, band * 6 + a] = colp / rs

    # C: own-block packed columns [F, (cen), cov, wt_cov]
    Ci = np.zeros((per, per))
    icen = phi['icen']
    icov = phi['icov']
    isw = phi['isw']
    for b in range(nband):
        Ci[:, b] = phi['dFold'][:, b] * deb.scales[i0 + b] / rs
    if deb.recenter:
        for c in range(2):
            Ci[:, icen + c] = (
                phi['dpos'][:, c]
                * deb.scales[i0 + icen + c] / rs
            )
    for c in range(3):
        Ci[:, icov + c] = (
            phi['dcovold'][:, c]
            * deb.scales[i0 + icov + c] / rs
        )
        Ci[:, isw + c] = (
            phi['dswold'][:, c]
            * deb.scales[i0 + isw + c] / rs
        )
    return Ai, Bi, Pi, Ci
