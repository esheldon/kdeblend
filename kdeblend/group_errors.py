"""
Group-coupled (adjoint) sandwich errors for the deblender.

The per-object sandwiches treat the subtracted neighbor models as
deterministic, but the neighbors are fit from the same noisy
pixels; for blend members the reported flux errors are low by
10-30 percent at 2 arcsec separations and up to 2x for tight
recentered pairs, and the member-member and cross-band flux
covariances are not available at all.  This module completes the
sandwich to the group-coupled estimating equations at the
converged fixed point,

    Cov(x*) = (I - J)^-1 dphi/dS Cov(S) dphi/dS^T (I - J)^-T

where x is the packed group state, J the Jacobian of the (Jacobi
form) sweep map, S the raw per-object per-epoch data moment sums,
and Cov(S) is exact under the k-space prep: the sums are linear
in the prepped image with closed-form kernels, so their
covariance follows from real-space influence kernels contracted
with the per-pixel variance, including the cross-member blocks
(the neighbor-noise coupling) which live on the shared pixels.

The Jacobian never touches the data modes: the data sums are
evaluated once at the solution and extended to first order in
(Sw, v, u) with the analytic kernel derivatives; every Jacobian
evaluation then costs only the closed-form neighbor/predicted
model sums and the update algebra.  With recentering, the anchor
positions are a noisy input (the detection centroids); their
linear response is priced when anchor_sigma is set.

Validated against finite-difference sandwiches and empirical
refit ensembles (2026-07-29): flux calibration 0.95-1.06 for
singles and pairs at 2 and 1 arcsec, fixed or adaptive centers,
with honest colors where the independence combination
over-predicts, and the anchor term reproducing the jittered
ensembles to a few percent.

Only groups whose members are all gauss/exp/dev are treated
(stars and bdf fall back to the per-object errors), and only the
ap_rad=0 prep (no apodization; the production setting) supports
the impulse-measured transfer.
"""
import copy

import numpy as np

from ngmix.prepsfadmom.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums

SUPPORTED_TYPES = ('gauss', 'exp', 'dev')

# state snapshot: everything but the shared prepped epochs and
# the scratch sum buffer
SNAP_SKIP = ('epochs_per_obj', 'esums')

# central difference steps: the packed state is normalized to
# O(1); the sum steps are scaled to the noise
FD_H = 1.0e-4
DS_FAC = 0.1

# the kernel weight support, matching FASTEXP_MAX_CHI2
WK_CHI2_MAX = 25.0


def apply_group_errors(deb, mbobs, res, anchor_sigma=0.0):
    """
    replace the per-object flux errors of a converged multi-object
    deblend with the group-coupled sandwich values, and add the
    cross-band flux covariance.

    Parameters
    ----------
    deb: _Deblender
        The converged deblender instance
    mbobs: ngmix.MultiBandObsList
        The observations the deblender was built from, for the
        weight maps and the impulse transfer
    res: dict
        The deblend result, modified in place: each object gains
        flux_cov (nband, nband) and has flux_err and s2n replaced
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
    True when the group errors were applied, False when the group
    is not eligible (unconverged, single, unsupported member
    types) and the per-object errors were left in place
    """
    if not res.get('converged', False):
        return False
    if deb.nobj < 2:
        return False
    for m in deb.models:
        if m['type'] not in SUPPORTED_TYPES:
            return False

    cov, slices = group_covariance(
        deb, mbobs, anchor_sigma=anchor_sigma,
    )

    nband = deb.nband
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
            robj['s2n'] = np.sqrt(np.sum(
                robj['flux'][wgood] ** 2 / var[wgood],
            ))
    return True


def group_covariance(deb, mbobs, anchor_sigma=0.0):
    """
    the full covariance of the packed group state at the converged
    fixed point, in physical units, and the per-object state
    offsets.  See the module docstring
    """
    snap = _snapshot(deb)
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
        [_dS_dtheta(deb, i, ep) for ep in epochs]
        for i in range(nobj)
    ]
    theta0s = [_theta_of(deb, i) for i in range(nobj)]

    covS = _cov_sums(deb, obs_flat, epochs)

    patched0 = _make_patched(deb, caches, Ds, theta0s, {})

    # the Jacobi-form sweep Jacobian: block rows through the
    # cheap evaluator (no data-mode work per evaluation)
    J = np.zeros((npars, npars))
    for j in range(npars):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += FD_H
        xm[j] -= FD_H
        for i in range(nobj):
            sl = slice(slices[i], slices[i] + pers[i])
            bp = _jacobi_block(deb, snap, xp, i, patched0)
            bm = _jacobi_block(deb, snap, xm, i, patched0)
            J[sl, j] = (bp[sl] - bm[sl]) / (2 * FD_H)

    # the data response, block diagonal in Jacobi form
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

    covU = dFdS @ covS @ dFdS.T
    M = np.eye(npars) - J
    cov_norm = np.linalg.solve(
        M, np.linalg.solve(M, covU.T).T,
    )

    anchor_cov = _anchor_cov(anchor_sigma, nobj)
    if deb.recenter and anchor_cov is not None:
        dFda = _dF_danchor(
            deb, snap, x0, patched0, slices, pers,
        )
        R = np.linalg.solve(M, dFda)
        cov_norm = cov_norm + R @ anchor_cov @ R.T

    D = np.diag(deb.scales)
    _restore(deb, snap)
    return D @ cov_norm @ D, slices


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


def _snapshot(deb):
    return {
        k: copy.deepcopy(v) for k, v in deb.__dict__.items()
        if k not in SNAP_SKIP
    }


def _restore(deb, snap):
    for k, v in snap.items():
        deb.__dict__[k] = copy.deepcopy(v)


def _theta_of(deb, i):
    sw = deb.Sw[i]
    v, u = deb.positions[i]
    return np.array([sw[0, 0], sw[0, 1], sw[1, 1], v, u])


def _kernel_ingredients(deb, i, ep):
    vi, ui = deb.positions[i]
    alpha, beta = get_phase_angles(
        ep, vi - ep['vcen'], ui - ep['ucen'],
    )
    dim = ep['dim']
    iy = ep['iy'].astype(np.int64)
    ix = ep['ix'].astype(np.int64)
    kv, ku = ep['kv'], ep['ku']
    sw = deb.Sw[i]
    wvv, wvu, wuu = sw[0, 0], sw[0, 1], sw[1, 1]

    Sv = wvv * kv + wvu * ku
    Su = wvu * kv + wuu * ku
    chi2 = kv * Sv + ku * Su
    wk = np.where(
        chi2 < WK_CHI2_MAX, np.exp(-0.5 * chi2), 0.0,
    )
    yf = np.where(iy < (dim + 1) // 2, iy, iy - dim)
    xf = np.where(ix < (dim + 1) // 2, ix, ix - dim)
    phase = np.exp(
        2j * np.pi / dim * (alpha * yf + beta * xf),
    )
    base = wk * phase * ep['df2']
    return Sv, Su, base, yf, xf, dim, (wvv, wvu, wuu)


def _build_kernels(deb, i, ep):
    """the (6, nmodes) complex kernels with S = Re(G @ kim): the
    admom_ksums accumulators in closed form"""
    Sv, Su, base, _, _, _, (wvv, wvu, wuu) = (
        _kernel_ingredients(deb, i, ep)
    )
    G = np.empty((6, Sv.size), dtype=complex)
    G[0] = 1j * Sv * base
    G[1] = 1j * Su * base
    G[2] = ((wuu - wvv) - (Su ** 2 - Sv ** 2)) * base
    G[3] = 2.0 * (wvu - Sv * Su) * base
    G[4] = ((wuu + wvv) - (Su ** 2 + Sv ** 2)) * base
    G[5] = base
    return G


def _dS_dtheta(deb, i, ep):
    """analytic (6, 5) derivative of the data sums with respect
    to theta = (Sw00, Sw01, Sw11, v, u), contracted with the
    data kim.  The kernel derivatives are exact; finite
    differences against admom_ksums itself only agree to ~1e-3
    because of its table exponential"""
    Sv, Su, base, yf, xf, dim, (wvv, wvu, wuu) = (
        _kernel_ingredients(deb, i, ep)
    )
    kv, ku = ep['kv'], ep['ku']
    kim = ep['kim']

    c = [
        1j * Sv, 1j * Su,
        (wuu - wvv) - (Su ** 2 - Sv ** 2),
        2.0 * (wvu - Sv * Su),
        (wuu + wvv) - (Su ** 2 + Sv ** 2),
        np.ones_like(Sv),
    ]
    zero = np.zeros_like(Sv)
    dc = {
        0: [1j * kv, 1j * ku, zero],
        1: [zero, 1j * kv, 1j * ku],
        2: [
            -1.0 + 2 * Sv * kv,
            -2 * Su * kv + 2 * Sv * ku,
            1.0 - 2 * Su * ku,
        ],
        3: [
            -2 * kv * Su,
            2 * (1.0 - ku * Su - kv * Sv),
            -2 * Sv * ku,
        ],
        4: [
            1.0 - 2 * Sv * kv,
            -2 * (Sv * ku + Su * kv),
            1.0 - 2 * Su * ku,
        ],
        5: [zero, zero, zero],
    }
    dchi = [kv * kv, 2 * kv * ku, ku * ku]

    # d(alpha, beta)/d(v, u) is linear; measure it exactly
    vi, ui = deb.positions[i]
    a0, b0 = get_phase_angles(
        ep, vi - ep['vcen'], ui - ep['ucen'],
    )
    av, bv = get_phase_angles(
        ep, vi + 1.0 - ep['vcen'], ui - ep['ucen'],
    )
    au, bu = get_phase_angles(
        ep, vi - ep['vcen'], ui + 1.0 - ep['ucen'],
    )
    pj = np.array([
        [av - a0, au - a0],
        [bv - b0, bu - b0],
    ])

    D = np.zeros((6, 5))
    for a in range(6):
        for w in range(3):
            dG = (dc[a][w] - 0.5 * c[a] * dchi[w]) * base
            D[a, w] = (dG @ kim).real
        Ga = c[a] * base
        dSa = ((2j * np.pi / dim) * yf * Ga @ kim).real
        dSb = ((2j * np.pi / dim) * xf * Ga @ kim).real
        D[a, 3] = dSa * pj[0, 0] + dSb * pj[1, 0]
        D[a, 4] = dSa * pj[0, 1] + dSb * pj[1, 1]
    return D


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


def _measure_transfer(deb, obs, band):
    """the image->kim map measured convention-free: kim of a
    unit impulse at image pixel (0, 0) through the identical
    prep.  Valid for the ap_rad=0 prep, where the map is
    diagonal in k up to the shift phase, with the rfft folding
    and padding placement absorbed"""
    import ngmix
    from .deblender import _prep_epochs

    im = np.zeros(obs.image.shape)
    im[0, 0] = 1.0
    iobs = ngmix.Observation(
        im,
        weight=obs.weight.copy(),
        jacobian=obs.jacobian.copy(),
        psf=obs.psf.copy(),
    )
    ieps = _prep_epochs(
        ngmix.observation.get_mb_obs(iobs),
        fwhm_smooth=deb.fwhm_smooth, ap_rad=0.0,
        use_noise_image=False, vcen=0.0, ucen=0.0,
    )
    return ieps[0]['kim'].copy()


def _cov_sums(deb, obs_flat, epochs):
    """the (6 nobj nep, 6 nobj nep) covariance of the stacked
    data sums: epoch-block-diagonal (independent noise per
    epoch), full cross-member within an epoch via the real-space
    influence kernels contracted with the per-pixel variance"""
    nobj = deb.nobj
    nep = len(epochs)
    nS = 6 * nobj * nep
    covS = np.zeros((nS, nS))
    for iep, ep in enumerate(epochs):
        obs = obs_flat[iep]
        That = _measure_transfer(deb, obs, ep['band'])
        dim = ep['dim']
        iy, ix = ep['iy'], ep['ix']
        ny, nx = obs.image.shape
        var = np.where(
            obs.weight > 0,
            1.0 / np.clip(obs.weight, 1.0e-300, None),
            0.0,
        )
        A = np.zeros((6 * nobj, dim, dim), dtype=complex)
        for i in range(nobj):
            G = _build_kernels(deb, i, ep)
            A[6 * i:6 * i + 6][:, iy, ix] = G * That
        hs = np.fft.fft2(A, axes=(-2, -1)).real[:, :ny, :nx]
        hv = hs * var
        cb = np.tensordot(hv, hs, axes=([1, 2], [1, 2]))
        for i in range(nobj):
            for j in range(nobj):
                covS[
                    (i * nep + iep) * 6:(i * nep + iep + 1) * 6,
                    (j * nep + iep) * 6:(j * nep + iep + 1) * 6,
                ] = cb[6 * i:6 * i + 6, 6 * j:6 * j + 6]
    return covS


def _make_patched(deb, caches, Ds, theta0s, deltas):
    """a _get_object_sums replacement: the linearized data sums
    plus the cheap closed-form neighbor/predicted sums, so
    Jacobian evaluations never touch the data modes"""

    def patched(i):
        m = deb.models[i]
        is_mix = m['type'] in ('exp', 'dev', 'bdf')
        base_nsums = deb._get_neighbor_sums(i)
        if is_mix:
            base_psums = deb._get_predicted_sums(i)

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
    _restore(deb, snap)
    deb._unpack_state(x)
    deb._get_object_sums = patched
    deb._update_object(i)
    out = deb._pack_state()
    del deb.__dict__['_get_object_sums']
    _restore(deb, snap)
    return out


def _jacobi_block_anchor(deb, snap, x, i, patched, danchor):
    """object i's Jacobi update with the anchor positions
    perturbed by danchor (nobj, 2) arcsec"""
    _restore(deb, snap)
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
    _restore(deb, snap)
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
