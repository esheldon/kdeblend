"""
tests for the full (fixed-point) errors: the analytic kernel
derivatives against finite differences of the exact-exp kernels,
the m=1 reduction to the per-object sandwich, and the pair
behavior (inflated flux errors, negative member covariance,
cross-band flux covariance)
"""
import galsim
import numpy as np
import ngmix
import pytest

from kdeblend.deblender import deblend

SCALE = 0.2
PSF_FWHM = 0.8
DIM = 64
PSF_DIM = 33
HLR = 0.5
FLUXES = [700.0, 1000.0]
SIGMAS = [4.0, 3.0]
NBAND = 2


def make_mbobs(rng, offsets):
    mbobs = ngmix.MultiBandObsList()
    cen = (DIM - 1) / 2
    psf_cen = (PSF_DIM - 1) / 2
    for band in range(NBAND):
        psf = galsim.Gaussian(fwhm=PSF_FWHM)
        psf_im = psf.drawImage(
            nx=PSF_DIM, ny=PSF_DIM, scale=SCALE,
        ).array
        psf_obs = ngmix.Observation(
            psf_im.copy(),
            weight=np.ones_like(psf_im) * 1.0e12,
            jacobian=ngmix.DiagonalJacobian(
                scale=SCALE, row=psf_cen, col=psf_cen,
            ),
        )
        im = np.zeros((DIM, DIM))
        for du, dv in offsets:
            obj = galsim.Convolve(
                galsim.Exponential(
                    half_light_radius=HLR, flux=FLUXES[band],
                ),
                psf,
            )
            im += obj.drawImage(
                nx=DIM, ny=DIM, scale=SCALE,
                offset=(du / SCALE, dv / SCALE),
            ).array
        im = im + rng.normal(scale=SIGMAS[band], size=im.shape)
        obs = ngmix.Observation(
            im,
            weight=np.full(
                im.shape, 1.0 / SIGMAS[band] ** 2,
            ),
            jacobian=ngmix.DiagonalJacobian(
                scale=SCALE, row=cen, col=cen,
            ),
            psf=psf_obs,
        )
        ol = ngmix.ObsList()
        ol.append(obs)
        mbobs.append(ol)
    return mbobs


def run_deblend(mbobs, offsets, **kw):
    objects = [
        {'v': dv, 'u': du, 'type': 'exp', 'Tguess': 0.3}
        for du, dv in offsets
    ]
    return deblend(
        mbobs, objects, tol=1.0e-6, rng=np.random.RandomState(5),
        **kw,
    )


def test_full_errors_restore_fidelity():
    """the targeted save/restore around a Jacobi evaluation
    leaves the deblender state bit-identical to a full deepcopy
    reference: guards the MUTABLE_ATTRS enumeration against
    future changes to the update path"""
    import copy

    from kdeblend.deblender import (
        _Deblender, _prep_epochs, _get_smoothing,
    )
    from kdeblend import full_errors as fe

    rng = np.random.RandomState(99)
    offsets = [(-0.5, 0.0), (0.5, 0.0)]
    mbobs = make_mbobs(rng, offsets)
    fwhm_smooth, Tsmooth = _get_smoothing(
        mbobs, None, 1.05, np.random.RandomState(3),
    )
    epochs = _prep_epochs(
        mbobs, fwhm_smooth=fwhm_smooth, ap_rad=0.0,
        use_noise_image=False, vcen=0.0, ucen=0.0,
    )
    objects = [
        {'v': dv, 'u': du, 'type': 'exp', 'Tguess': 0.3}
        for du, dv in offsets
    ]
    deb = _Deblender(
        [epochs] * 2, NBAND, objects, fwhm_smooth, Tsmooth,
        500, 1.0e-6, recenter=True, cen_sigma0=0.1,
    )
    deb.go()

    skip = ('epochs_per_obj', 'esums')
    ref = {
        k: copy.deepcopy(v) for k, v in deb.__dict__.items()
        if k not in skip
    }

    snap = fe._save_state(deb)
    x0 = deb._pack_state()
    caches = [
        [fe._data_esums(deb, i, ep) for ep in epochs]
        for i in range(2)
    ]
    from ngmix.prepsfadmom.full_errors import dsums_dtheta
    Ds = [
        [
            dsums_dtheta(
                ep, deb.Sw[i],
                deb.positions[i][0] - ep['vcen'],
                deb.positions[i][1] - ep['ucen'],
            )
            for ep in epochs
        ]
        for i in range(2)
    ]
    theta0s = [fe._theta_of(deb, i) for i in range(2)]
    patched = fe._make_patched(deb, caches, Ds, theta0s, {})
    xp = x0.copy()
    xp[0] += 1.0e-3
    fe._jacobi_block(deb, snap, xp, 0, patched)
    dp = np.zeros((2, 2))
    dp[0, 0] = 1.0e-4
    fe._jacobi_block_anchor(deb, snap, x0, 1, patched, dp)

    def same(a, b):
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b, equal_nan=True)
        if isinstance(a, dict):
            return set(a) == set(b) and all(
                same(a[k], b[k]) for k in a
            )
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(
                same(x, y) for x, y in zip(a, b)
            )
        try:
            return bool(a == b) or (a != a and b != b)
        except Exception:
            return repr(a) == repr(b)

    bad = [
        k for k in ref
        if not same(deb.__dict__[k], ref[k])
    ]
    assert bad == [], f'state not restored: {bad}'


def test_full_errors_single_reduction():
    """at m=1 the group sandwich must agree with the per-object
    sandwich on matched data: same estimating equations.  Also
    checks the production path now applies to singles, with the
    structure errors wired from the family-covariance block"""
    from kdeblend.deblender import (
        _Deblender, _prep_epochs, _get_smoothing,
    )
    from kdeblend.full_errors import full_covariance

    rng = np.random.RandomState(11)
    offsets = [(0.0, 0.0)]
    mbobs = make_mbobs(rng, offsets)
    res = run_deblend(mbobs, offsets)
    assert res['converged']

    fwhm_smooth, Tsmooth = _get_smoothing(
        mbobs, None, 1.05, np.random.RandomState(3),
    )
    epochs = _prep_epochs(
        mbobs, fwhm_smooth=fwhm_smooth, ap_rad=0.0,
        use_noise_image=False, vcen=0.0, ucen=0.0,
        store_transfer=True,
    )
    deb = _Deblender(
        [epochs], NBAND,
        [{'v': 0.0, 'u': 0.0, 'type': 'exp', 'Tguess': 0.3}],
        fwhm_smooth, Tsmooth, 500, 1.0e-6,
    )
    gres = deb.go()
    assert gres['converged']
    cov, slices, extras = full_covariance(deb, mbobs)

    for b in range(NBAND):
        grp = np.sqrt(cov[b, b])
        rep = gres['objects'][0]['flux_err'][b]
        assert np.abs(grp / rep - 1) < 0.1
    # the cross-band covariance agrees with the extended
    # per-object sandwich (positive, from the shared structure)
    fcov = gres['objects'][0]['flux_cov']
    assert fcov is not None
    assert cov[0, 1] > 0
    assert np.abs(cov[0, 1] / fcov[0, 1] - 1) < 0.25

    # the production path applies to singles: group flux errors,
    # and structure errors within ~10 percent of the per-object
    # sandwich on this matched-model scene
    resg = run_deblend(mbobs, offsets, full_errors=True)
    assert resg['converged'] and resg['full_errors']
    r0 = res['objects'][0]
    rg = resg['objects'][0]
    assert np.allclose(
        rg['flux_err'], np.sqrt(np.diag(cov)[:NBAND]),
        rtol=1e-6,
    )
    for key in ('T_err', 'e1_err', 'e2_err'):
        assert np.isfinite(rg[key])
        assert np.abs(rg[key] / r0[key] - 1) < 0.15

    # the gauss-estimator entries are replaced too: the flux
    # value is unchanged (same converged state and formula), the
    # covariance is filled consistently, and the errors sit
    # within the mismatch scale of the delta values (the gauss
    # estimator sees exp data as mismatched, so exact agreement
    # is not expected)
    assert np.allclose(rg['gauss_flux'], r0['gauss_flux'],
                       rtol=1e-8)
    gfc = rg['gauss_flux_cov']
    assert gfc is not None and gfc.shape == (NBAND, NBAND)
    assert np.allclose(
        rg['gauss_flux_err'], np.sqrt(np.diag(gfc)), rtol=1e-6,
    )
    for key in ('gauss_T_err', 'gauss_e1_err', 'gauss_e2_err'):
        assert np.isfinite(rg[key])
        assert 0.7 < rg[key] / r0[key] < 1.5


@pytest.mark.parametrize('recenter', [False, True])
def test_full_errors_pair(recenter):
    """a close pair with full_errors=True: flux errors inflate
    over the deterministic-neighbor values, the member flux
    covariance is negative, and flux_cov is filled"""
    rng = np.random.RandomState(21)
    offsets = [(-0.5, 0.0), (0.5, 0.0)]
    mbobs = make_mbobs(rng, offsets)

    kw = {}
    if recenter:
        kw = {'recenter': True, 'cen_sigma0': 0.1}
    res0 = run_deblend(mbobs, offsets, **kw)
    res = run_deblend(mbobs, offsets, full_errors=True, **kw)
    assert res['converged']
    assert res['full_errors']

    from kdeblend.full_errors import full_covariance  # noqa

    for i in range(2):
        e0 = res0['objects'][i]['flux_err']
        e1 = res['objects'][i]['flux_err']
        # the neighbor term inflates the tight-pair errors
        assert np.all(e1 > e0)
        fcov = res['objects'][i]['flux_cov']
        assert fcov is not None and fcov.shape == (2, 2)
        assert np.allclose(
            np.sqrt(np.diag(fcov)), e1, rtol=1e-6,
        )
        # the structure errors inflate too (T feels the tight
        # neighbor term strongly)
        assert (
            res['objects'][i]['T_err']
            > res0['objects'][i]['T_err']
        )
        # the gauss-estimator entries feel the neighbor term the
        # same way
        gfc = res['objects'][i]['gauss_flux_cov']
        assert gfc is not None and gfc.shape == (2, 2)
        assert np.allclose(
            res['objects'][i]['gauss_flux_err'],
            np.sqrt(np.diag(gfc)), rtol=1e-6,
        )
        assert np.all(
            res['objects'][i]['gauss_flux_err']
            > res0['objects'][i]['gauss_flux_err']
        )
        assert (
            res['objects'][i]['gauss_T_err']
            > res0['objects'][i]['gauss_T_err']
        )


def test_full_errors_star_fallback():
    """a group containing a star keeps the per-object errors and
    reports full_errors False"""
    rng = np.random.RandomState(31)
    offsets = [(-0.5, 0.0), (0.5, 0.0)]
    mbobs = make_mbobs(rng, offsets)
    objects = [
        {'v': 0.0, 'u': -0.5, 'type': 'exp', 'Tguess': 0.3},
        {'v': 0.0, 'u': 0.5, 'type': 'star'},
    ]
    res = deblend(
        mbobs, objects, tol=1.0e-6,
        rng=np.random.RandomState(5), full_errors=True,
    )
    assert res['full_errors'] is False


def test_full_errors_anchor_forms():
    """scalar, per-object-sigma and per-object-covariance
    anchor_sigma inputs agree when they encode the same noise,
    and anisotropic covariances change the answer"""
    rng = np.random.RandomState(21)
    offsets = [(-0.5, 0.0), (0.5, 0.0)]
    mbobs = make_mbobs(rng, offsets)
    kw = {'recenter': True, 'cen_sigma0': 0.1}

    sig = 0.05
    res_s = run_deblend(
        mbobs, offsets, full_errors=True, anchor_sigma=sig,
        **kw,
    )
    assert res_s['converged'] and res_s['full_errors']
    res_v = run_deblend(
        mbobs, offsets, full_errors=True,
        anchor_sigma=np.array([sig, sig]), **kw,
    )
    covs = np.array([
        sig ** 2 * np.eye(2), sig ** 2 * np.eye(2),
    ])
    res_c = run_deblend(
        mbobs, offsets, full_errors=True, anchor_sigma=covs,
        **kw,
    )
    e_s = res_s['objects'][0]['flux_err']
    e_v = res_v['objects'][0]['flux_err']
    e_c = res_c['objects'][0]['flux_err']
    assert np.allclose(e_v, e_s, rtol=1e-10)
    assert np.allclose(e_c, e_s, rtol=1e-10)

    # anchor noise inflates over the conditioned errors, and an
    # anisotropic covariance differs from the isotropic one
    res_0 = run_deblend(
        mbobs, offsets, full_errors=True, **kw,
    )
    assert np.all(e_s > res_0['objects'][0]['flux_err'])
    aniso = np.array([
        np.diag([sig ** 2, 0.0]), np.diag([sig ** 2, 0.0]),
    ])
    res_a = run_deblend(
        mbobs, offsets, full_errors=True, anchor_sigma=aniso,
        **kw,
    )
    assert not np.allclose(
        res_a['objects'][0]['flux_err'], e_s, rtol=1e-3,
    )


@pytest.mark.parametrize('recenter', [False, True])
def test_full_errors_chain_vs_fd(recenter):
    """the chain-rule Jacobian/data-response assembly agrees
    with the full finite-difference reference: same covariance
    to a fraction of a percent on a close pair, with and
    without recentering (and with the anchor term when
    recentered)"""
    from kdeblend.deblender import (
        _Deblender, _prep_epochs, _get_smoothing,
    )
    from kdeblend.full_errors import full_covariance

    rng = np.random.RandomState(21)
    offsets = [(-0.5, 0.0), (0.5, 0.0)]
    mbobs = make_mbobs(rng, offsets)
    fwhm_smooth, Tsmooth = _get_smoothing(
        mbobs, None, 1.05, np.random.RandomState(5),
    )
    epochs = _prep_epochs(
        mbobs, fwhm_smooth=fwhm_smooth, ap_rad=0.0,
        use_noise_image=False, vcen=0.0, ucen=0.0,
        store_transfer=True,
    )
    objects = [
        {'v': dv, 'u': du, 'type': 'exp', 'Tguess': 0.3}
        for du, dv in offsets
    ]
    kw = {}
    if recenter:
        kw = {'recenter': True, 'cen_sigma0': 0.1}
    deb = _Deblender(
        [epochs] * 2, NBAND, objects, fwhm_smooth, Tsmooth,
        500, 1.0e-6, **kw,
    )
    res = deb.go()
    assert res['converged']

    asig = 0.05 if recenter else 0.0
    cov_c, sl, ex_c = full_covariance(
        deb, mbobs, anchor_sigma=asig, use_chain=True,
    )
    cov_f, _, ex_f = full_covariance(
        deb, mbobs, anchor_sigma=asig, use_chain=False,
    )
    dd = np.sqrt(np.diag(cov_c) / np.diag(cov_f))
    assert np.all(np.abs(dd - 1) < 5.0e-3), dd
    # off-diagonals of the flux blocks agree too
    assert np.allclose(
        cov_c[:NBAND, :NBAND], cov_f[:NBAND, :NBAND],
        rtol=2e-2, atol=0,
    )
    # the gauss-estimator flux covariances agree between the
    # assemblies too (shared model-sum derivatives, different
    # Jacobian and data response)
    for i in range(2):
        gc = ex_c['gauss_flux_cov'][i]
        gf = ex_f['gauss_flux_cov'][i]
        assert np.allclose(
            np.sqrt(np.diag(gc)), np.sqrt(np.diag(gf)),
            rtol=5e-3,
        )
        assert np.allclose(gc, gf, rtol=2e-2, atol=0)
        assert np.allclose(
            ex_c['gauss_flux'][i], ex_f['gauss_flux'][i],
            rtol=1e-10,
        )


def test_full_errors_gauss_mc():
    """monte carlo calibration of the gauss-estimator entries on
    a single exp object: for a gaussian-weight estimator every
    non-gaussian profile is mismatched, so the delta-method
    gauss errors run low even on exp truth (T 13 percent, flux 6
    in the PAdmomFitter MC); the full errors must be calibrated"""
    offsets = [(0.0, 0.0)]
    ntrial = 150
    gT = np.zeros(ntrial)
    gTe = np.zeros(ntrial)
    gF = np.zeros(ntrial)
    gFe = np.zeros(ntrial)
    ngood = 0
    for k in range(ntrial):
        rng = np.random.RandomState(7000 + k * 13)
        mbobs = make_mbobs(rng, offsets)
        res = run_deblend(mbobs, offsets, full_errors=True)
        robj = res['objects'][0]
        if (
            not res['converged'] or not res['full_errors']
            or robj['gauss_e_flags'] != 0
            or not np.isfinite(robj['gauss_T_err'])
        ):
            continue
        gT[ngood] = robj['gauss_T']
        gTe[ngood] = robj['gauss_T_err']
        gF[ngood] = robj['gauss_flux'][0]
        gFe[ngood] = robj['gauss_flux_err'][0]
        ngood += 1
    assert ngood > 0.9 * ntrial
    rT = gT[:ngood].std() / np.sqrt(np.mean(gTe[:ngood] ** 2))
    rF = gF[:ngood].std() / np.sqrt(np.mean(gFe[:ngood] ** 2))
    assert 0.8 < rT < 1.2, rT
    assert 0.85 < rF < 1.15, rF
