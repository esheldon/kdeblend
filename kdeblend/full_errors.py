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

Objects of type gauss/exp/dev/star are treated (bdf falls back
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
)
from ngmix.prepsfadmom.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums

SUPPORTED_TYPES = ('gauss', 'exp', 'dev', 'star')

# central difference steps: the packed state is normalized to
# O(1); the sum steps are scaled to the noise
FD_H = 1.0e-4
DS_FAC = 0.1

# deblender attributes an update evaluation can mutate; saved and
# restored around every Jacobian evaluation.  The enumeration is
# guarded by the restore-fidelity test
MUTABLE_ATTRS = (
    'models', 'Sw', 'positions', 'cen_pull', '_sweep_changes',
    '_fscales', '_win_max', '_win_nfail', '_prev_win_max',
    '_prev_win_nfail', '_change_hist', 'hist', 'nskip', 'nfail',
    'nrestart', 'isweep', 'dbflags', 'bdf_info', 'bdf_last_dfd',
    '_cen_sigma_sweep', '_bdf_noise_cache',
)


def apply_full_errors(deb, mbobs, res, anchor_sigma=0.0):
    """
    replace the per-object flux and structure errors of a
    converged deblend with the full (fixed-point) values, and add
    the cross-band flux covariance.

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
        gauss_e2_err from the weight (Sw) rows of the state
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

    cov, slices, extras = full_covariance(
        deb, mbobs, anchor_sigma=anchor_sigma,
    )

    from .deblender import _shape_errors, _joint_s2n

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

        if deb.models[i]['type'] == 'star':
            # a delta function has no structure or weight state:
            # the packed block ends at the (optional) center
            # columns, and the structure offsets below would read
            # the next object's block.  The per-object structure
            # entries (T = 0, flagged shapes) stand
            continue

        # structure errors from the family-covariance block;
        # replaced only when the full values are usable, so a
        # degenerate block cannot degrade a usable row
        ic = i0 + nband + ncen
        cblock = cov[np.ix_(
            [ic, ic + 1, ic + 2], [ic, ic + 1, ic + 2],
        )]
        fam_cov = L @ cblock @ L.T
        if np.all(np.isfinite(fam_cov)) and fam_cov[2, 2] > 0:
            robj['T_err'] = np.sqrt(fam_cov[2, 2])
            if np.isfinite(robj['e1']) and robj['T'] > 0:
                e1e, e2e, eflags = _shape_errors(
                    robj['e1'], robj['e2'], robj['T'], fam_cov,
                )
                if eflags == 0:
                    robj['e1_err'] = e1e
                    robj['e2_err'] = e2e

        # gauss-estimator structure errors from the weight rows:
        # the gauss family is the weight minus the constant
        # smoothing, so its covariance is the Sw block
        isw = ic + 3
        gblock = cov[np.ix_(
            [isw, isw + 1, isw + 2], [isw, isw + 1, isw + 2],
        )]
        gfam_cov = L @ gblock @ L.T
        if np.all(np.isfinite(gfam_cov)) and gfam_cov[2, 2] > 0:
            robj['gauss_T_err'] = np.sqrt(gfam_cov[2, 2])
            if np.isfinite(robj['gauss_e1']) \
                    and robj['gauss_T'] > 0:
                e1e, e2e, eflags = _shape_errors(
                    robj['gauss_e1'], robj['gauss_e2'],
                    robj['gauss_T'], gfam_cov,
                )
                if eflags == 0:
                    robj['gauss_e1_err'] = e1e
                    robj['gauss_e2_err'] = e2e

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


def full_covariance(deb, mbobs, anchor_sigma=0.0,
                    use_chain=None):
    """
    the full covariance of the packed deblend state at the
    converged fixed point, in physical units, the per-object
    state offsets, and the derived gauss-estimator flux values
    and covariances as a dict of per-object lists
    {'gauss_flux', 'gauss_flux_cov'} (None entries for stars).
    See the module docstring.

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
    snap = _save_state(deb)
    x0 = deb._pack_state()
    npars = x0.size
    nobj = deb.nobj
    slices, pers = _object_layout(deb)

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
                ep, deb.Sw[i],
                deb.positions[i][0] - ep['vcen'],
                deb.positions[i][1] - ep['ucen'],
            )
            for ep in epochs
        ]
        for i in range(nobj)
    ]
    theta0s = [_theta_of(deb, i) for i in range(nobj)]

    covS = _cov_sums(deb, obs_flat, epochs)

    cols = _column_map(deb)
    if use_chain:
        J, dFdS, dFda, dNS = _chain_pieces(
            deb, snap, x0, caches, Ds, theta0s, slices, pers,
            covS, epochs,
        )
    else:
        patched0 = _make_patched(deb, caches, Ds, theta0s, {})
        J = _fd_jacobian(
            deb, snap, x0, patched0, slices, pers,
        )
        dFdS = _fd_dFdS(
            deb, snap, x0, caches, Ds, theta0s, slices, pers,
            covS, epochs,
        )
        dFda = None
        dNS, _ = _model_sum_derivs(deb, cols)

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
        Tx, covS, Ra, anchor_cov,
    )

    D = np.diag(deb.scales)
    _restore_state(deb, snap)
    return D @ cov_norm @ D, slices, extras


def _fd_jacobian(deb, snap, x0, patched0, slices, pers):
    """the Jacobi-form sweep Jacobian by central FD over the
    packed state; the reference implementation for the chain"""
    npars = x0.size
    J = np.zeros((npars, npars))
    for j in range(npars):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += FD_H
        xm[j] -= FD_H
        for i in range(deb.nobj):
            sl = slice(slices[i], slices[i] + pers[i])
            bp = _jacobi_block(deb, snap, xp, i, patched0)
            bm = _jacobi_block(deb, snap, xm, i, patched0)
            J[sl, j] = (bp[sl] - bm[sl]) / (2 * FD_H)
    return J


def _fd_dFdS(deb, snap, x0, caches, Ds, theta0s, slices, pers,
             covS, epochs):
    """the data response by central FD; the reference
    implementation for the chain"""
    nobj = deb.nobj
    npars = x0.size
    nep = len(epochs)
    nS = 6 * nobj * nep
    dsteps = DS_FAC * np.sqrt(np.diag(covS))
    dFdS = np.zeros((npars, nS))
    for i in range(nobj):
        sl = slice(slices[i], slices[i] + pers[i])
        for iep in range(nep):
            for a in range(6):
                col = (i * nep + iep) * 6 + a
                h = dsteps[col]
                d = np.zeros(6)
                d[a] = h
                pp = _make_patched(
                    deb, caches, Ds, theta0s, {(i, iep): d},
                )
                pm = _make_patched(
                    deb, caches, Ds, theta0s, {(i, iep): -d},
                )
                bp = _jacobi_block(deb, snap, x0, i, pp)
                bm = _jacobi_block(deb, snap, x0, i, pm)
                dFdS[sl, col] = (bp[sl] - bm[sl]) / (2 * h)
    return dFdS


def _fastcopy(obj):
    """a cheap recursive copy for the small deblender state
    (arrays, dicts, lists, tuples, scalars); much faster than
    copy.deepcopy for this shape of data"""
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
    the (2 nobj, 2 nobj) block-diagonal anchor covariance from
    the anchor_sigma input: a scalar sigma in arcsec (isotropic,
    shared), an (nobj,) array of per-object sigmas, or an
    (nobj, 2, 2) array of per-object position covariances in
    arcsec^2 with (v, u) ordering.  None when every entry is
    zero (the errors then condition on the anchors)
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
    """per-object offsets and sizes in the packed state,
    replaying the _pack_state layout"""
    slices = []
    pers = []
    k = 0
    for m in deb.models:
        slices.append(k)
        per = deb.nband
        if deb.recenter:
            per += 2
        if m['type'] == 'gauss':
            per += 3
        elif m['type'] in ('exp', 'dev'):
            per += 3
        elif m['type'] == 'bdf':
            per += 4
        if m['type'] != 'star':
            per += 3
        pers.append(per)
        k += per
    return slices, pers


def _theta_of(deb, i):
    sw = deb.Sw[i]
    v, u = deb.positions[i]
    return np.array([sw[0, 0], sw[0, 1], sw[1, 1], v, u])


def _data_esums(deb, i, ep):
    """the raw data sums for object i on one epoch at the
    converged state"""
    vi, ui = deb.positions[i]
    alpha, beta = get_phase_angles(
        ep, vi - ep['vcen'], ui - ep['ucen'],
    )
    sw = deb.Sw[i]
    sums = np.zeros(6)
    admom_ksums(
        ep['kim'], ep['iy'], ep['ix'], ep['dim'], alpha, beta,
        ep['kv'], ep['ku'], sw[0, 0], sw[0, 1], sw[1, 1],
        ep['df2'], sums,
    )
    return sums


def _cov_sums(deb, obs_flat, epochs):
    """the (6 nobj nep, 6 nobj nep) covariance of the stacked
    data sums: epoch-block-diagonal (independent noise per
    epoch), full cross-member within an epoch via the ngmix
    influence kernels"""
    nobj = deb.nobj
    nep = len(epochs)
    nS = 6 * nobj * nep
    covS = np.zeros((nS, nS))
    for iep, ep in enumerate(epochs):
        obs = obs_flat[iep]
        G = np.vstack([
            moment_kernels(
                ep, deb.Sw[i],
                deb.positions[i][0] - ep['vcen'],
                deb.positions[i][1] - ep['ucen'],
            )
            for i in range(nobj)
        ])
        hs = influence_kernels(ep, G, obs.image.shape)
        cb = sums_cov(hs, obs.weight)
        for i in range(nobj):
            for j in range(nobj):
                covS[
                    (i * nep + iep) * 6:(i * nep + iep + 1) * 6,
                    (j * nep + iep) * 6:(j * nep + iep + 1) * 6,
                ] = cb[6 * i:6 * i + 6, 6 * j:6 * j + 6]
    return covS


def _make_patched(deb, caches, Ds, theta0s, deltas,
                  nsums_cache=None, psums_cache=None,
                  nsums_delta=None, psums_delta=None,
                  freeze_theta=False):
    """a _get_object_sums replacement: the linearized data sums
    plus the cheap closed-form neighbor/predicted sums, so
    Jacobian evaluations never touch the data modes.  With
    nsums_cache/psums_cache the model sums are not recomputed
    either (pure update algebra); the delta arguments inject
    perturbations for derivative evaluations, and freeze_theta
    holds the data sums at the cached values (the theta channel
    is then added analytically by the chain)"""

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


def _jacobi_block(deb, snap, x, i, patched):
    """object i's update evaluated from state x with the patched
    sums (the Jacobi map: every object from the same state)"""
    _restore_state(deb, snap)
    deb._unpack_state(x)
    deb._get_object_sums = patched
    deb._update_object(i)
    out = deb._pack_state()
    del deb.__dict__['_get_object_sums']
    _restore_state(deb, snap)
    return out


def _jacobi_block_anchor(deb, snap, x, i, patched, danchor):
    """object i's Jacobi update with the anchor positions
    perturbed by danchor (nobj, 2) arcsec"""
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
    """the Jacobi-map response to the anchor (detection)
    positions, per arcsec, by central FD; the anchor enters
    through the recentering regularization and the packed
    center convention"""
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
    """per packed column: (object, kind, sub) with kind one of
    'F' (sub = band), 'cen' (sub 0 = v, 1 = u), 'cov'
    (sub = 00, 01, 11 component), 'fracdev', 'sw'"""
    cols = []
    for k, m in enumerate(deb.models):
        cols += [(k, 'F', b) for b in range(deb.nband)]
        if deb.recenter:
            cols += [(k, 'cen', 0), (k, 'cen', 1)]
        if m['type'] in ('gauss', 'exp', 'dev', 'bdf'):
            cols += [(k, 'cov', c) for c in range(3)]
        if m['type'] == 'bdf':
            cols += [(k, 'fracdev', 0)]
        if m['type'] != 'star':
            cols += [(k, 'sw', c) for c in range(3)]
    return cols


def _covkey(m):
    return 'cov_sm' if m['type'] == 'gauss' else 'cov'


_SYM = [(0, 0), (0, 1), (1, 1)]


def _model_sum_derivs(deb, cols):
    """closed-form model-sum derivatives by micro central FD on
    the (cheap, analytic) neighbor and predicted sum functions:
    dNS[(i, col)] and dPS[(i, col)] as (nband, 6) arrays per
    affected object, in physical units per unit physical change
    of the column quantity"""
    nobj = deb.nobj
    Tw = deb.Sw[0][0, 0] + deb.Sw[0][1, 1]
    h_cov = 1.0e-6 * max(Tw, 0.1)
    h_pos = 1.0e-6

    dNS = {}
    dPS = {}

    def ns(i, Sw=None):
        return deb._get_neighbor_sums(i, Sw=Sw)

    for ic, (k, kind, sub) in enumerate(cols):
        m = deb.models[k]
        if kind == 'sw':
            # only object k's own sums use its weight
            r, c = _SYM[sub]
            swp = deb.Sw[k].copy()
            swm = deb.Sw[k].copy()
            swp[r, c] += h_cov
            swp[c, r] = swp[r, c]
            swm[r, c] -= h_cov
            swm[c, r] = swm[r, c]
            dNS[(k, ic)] = (
                ns(k, Sw=swp) - ns(k, Sw=swm)
            ) / (2 * h_cov)
            if m['type'] in ('exp', 'dev', 'bdf'):
                sw0 = deb.Sw[k]
                deb.Sw[k] = swp
                psp = deb._get_predicted_sums(k)
                deb.Sw[k] = swm
                psm = deb._get_predicted_sums(k)
                deb.Sw[k] = sw0
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
                     epochs, Tx, covS, Ra, anchor_cov):
    """the gauss-estimator fluxes F_b = 4 pi sqrt(det Sw)
    fs_b / ws_b and their (nband, nband) covariance per object,
    as {'gauss_flux', 'gauss_flux_cov'} lists with None entries
    for stars.  fs_b is linear in the data at the converged
    state, so the response is the direct data channel plus the
    chain through the state: the own kernel (weight and center)
    via the analytic data-sum derivatives, the neighbor
    subtraction via the model-sum derivatives, and the explicit
    sqrt(det Sw) normalization; the anchor response is added
    when present.  The per-epoch weights ws_b are fixed at prep,
    so they carry no state dependence"""
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
        Sw = deb.Sw[i]
        detS = Sw[0, 0] * Sw[1, 1] - Sw[0, 1] ** 2
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

        # d ln cnorm / d(sw3)
        dlnc = np.array([
            Sw[1, 1], -2.0 * Sw[0, 1], Sw[0, 0],
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
                  pers, covS, epochs):
    """the Jacobi Jacobian, data response and anchor response
    assembled by the chain rule: micro finite differences of the
    pure update algebra (all sums cached) chained with the
    analytic data-sum derivatives and the closed-form model-sum
    derivatives.  Structural zeros (a member's update does not
    depend on other members' weights) are never evaluated"""
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

    dFdS = np.zeros((npars, 6 * nobj * nep))
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

    dNS, dPS = _model_sum_derivs(deb, cols)

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
    return J, dFdS, dFda, dNS


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
    """the moment matrix in sym3 and its (3, 6) derivative with
    respect to the raw sums"""
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
    """the analytic update-algebra derivatives for object i at
    the given converged inputs, or None when the evaluation
    takes a guarded branch.  Output rows follow the packed block
    layout [F, (cen), cov, Sw] in PHYSICAL units; the caller
    applies the packing scales"""
    from .deblender import (
        RECENTER_CLIP_FAC, ZERO_WEIGHT, mixture_model_valid,
    )

    m = deb.models[i]
    mtype = m['type']
    if mtype not in ('gauss', 'exp', 'dev'):
        return None
    if not (sums[5] > 0 and sums[4] > 0):
        return None

    nband = deb.nband
    rc = deb.recenter
    per = nband + (2 if rc else 0) + 6
    nrow = per

    Swold = deb.Sw[i]
    Mm, dMm = _dmm_dsums(sums)
    newSw, DWm_M, DWm_S, ok = _dw_derivs(
        _sym3_mat(Mm), Swold,
    )
    if not ok:
        return None

    # rows: F [0:nband], cen [nband:nband+2] if rc, cov, Sw
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

    # the weight rows: newSw = DW(Mm, Swold) for every type
    dsums[isw:isw + 3] = DWm_M @ dMm
    dswold[isw:isw + 3] = DWm_S

    if mtype == 'gauss':
        # cov_sm = Sw = newSw
        dsums[icov:icov + 3] = dsums[isw:isw + 3]
        dswold[icov:icov + 3] = dswold[isw:isw + 3]
        # matched flux: F_b = fs_b / ws_b * 2 pi sqrt(det(Swold
        # + newSw))
        dd, sq = _ddetsqrt(Swold + newSw)
        fac = 2.0 * np.pi * sq / ws
        for b in range(nband):
            dfs[b, b] = fac[b]
            pref = fs[b] / ws[b] * 2.0 * np.pi
            # through newSw (sums, Swold) and directly Swold
            dsums[b] = pref * (
                dd @ dsums[isw:isw + 3]
            )
            dswold[b] = pref * (
                dd @ (dswold[isw:isw + 3] + np.eye(3))
            )
    else:
        # exp/dev mixture: shift = newSw - DW(Mp) (main branch)
        # or the ratio fallback; prop = cov_old + shift
        Mp, dMp = _dmm_dsums(pred)
        Sp, DWp_M, DWp_S, pok = _dw_derivs(
            _sym3_mat(Mp), Swold,
        )
        if pok:
            dcov_dsums = DWm_M @ dMm
            dcov_dpred = -(DWp_M @ dMp)
            dcov_dswold = DWm_S - DWp_S
            shift3 = np.array([
                newSw[0, 0] - Sp[0, 0], newSw[0, 1] - Sp[0, 1],
                newSw[1, 1] - Sp[1, 1],
            ])
        else:
            # gain-1 ratio fallback
            Sfam = m['cov']
            Tp = pred[4] / pred[5]
            Tf = Sfam[0, 0] + Sfam[1, 1]
            fac = sums[4] / sums[5] / Tp
            de1 = sums[2] / sums[4] - pred[2] / pred[4]
            de2 = sums[3] / sums[4] - pred[3] / pred[4]
            base = 0.5 * fac * Tf
            shift3 = np.array([
                (fac - 1) * Sfam[0, 0] - base * de1,
                (fac - 1) * Sfam[0, 1] + base * de2,
                (fac - 1) * Sfam[1, 1] + base * de1,
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
                Sfam[0, 0], Sfam[0, 1], Sfam[1, 1],
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
    """the converged (sums, fs, ws, pred, fs_pred) for object i
    from the cached raw data and model sums, replicating the
    _get_object_sums accumulation"""
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
    """map the physical update-algebra derivatives to the
    packed-normalized A (data sums), B (neighbor sums), P
    (predicted sums) and C (own state) matrices"""
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

    # C: own-block packed columns [F, (cen), cov, Sw]
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
