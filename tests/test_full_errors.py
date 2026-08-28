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
        mbobs, objects, tol=1.0e-6, maxiter=2000,
        rng=np.random.RandomState(5),
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
    offsets = [(-0.625, 0.0), (0.625, 0.0)]
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
        2000, 1.0e-6, recenter=True, cen_sigma0=0.1,
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
        fwhm_smooth, Tsmooth, 2000, 1.0e-6,
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
    offsets = [(-0.625, 0.0), (0.625, 0.0)]
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


def test_full_errors_star_member():
    """a group containing a star applies the full errors: the
    star row gets the flux entries (its structure entries stand)
    and the galaxy row gets the full structure errors.  bdf
    members still fall back"""
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
    assert res['converged']
    assert res['full_errors'] is True
    rgal, rstar = res['objects']
    assert np.all(np.isfinite(rgal['flux_err']))
    assert np.isfinite(rgal['T_err'])
    assert np.all(np.isfinite(rstar['flux_err']))
    assert rstar['flux_cov'] is not None
    assert np.isfinite(rstar['s2n'])
    assert rstar['T'] == 0.0
    assert not np.isfinite(rstar['T_err'])


def test_full_errors_anchor_forms():
    """scalar, per-object-sigma and per-object-covariance
    anchor_sigma inputs agree when they encode the same noise,
    and anisotropic covariances change the answer"""
    rng = np.random.RandomState(21)
    offsets = [(-0.625, 0.0), (0.625, 0.0)]
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
    offsets = [(-0.625, 0.0), (0.625, 0.0)]
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
        2000, 1.0e-6, **kw,
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


def test_full_errors_apodized_pair_mc():
    """monte carlo calibration of the full errors with the
    production apodization (ap_rad=1.5): the influence kernels
    carry the mask in pixel space, so a close pair's flux and T
    pulls must stay calibrated.  Guards the masked-transfer path
    end to end through the joint chain"""
    offsets = [(-0.625, 0.0), (0.625, 0.0)]
    ntrial = 150
    F = np.zeros((ntrial, 2))
    Fe = np.zeros((ntrial, 2))
    T = np.zeros((ntrial, 2))
    Te = np.zeros((ntrial, 2))
    ngood = 0
    for k in range(ntrial):
        rng = np.random.RandomState(9000 + k * 17)
        mbobs = make_mbobs(rng, offsets)
        res = run_deblend(
            mbobs, offsets, full_errors=True, ap_rad=1.5,
        )
        if not res['converged'] or not res['full_errors']:
            continue
        ok = True
        for i in range(2):
            robj = res['objects'][i]
            if (
                robj['deblend_flags'] != 0
                or not np.isfinite(robj['T_err'])
                or not np.all(np.isfinite(robj['flux_err']))
            ):
                ok = False
        if not ok:
            continue
        for i in range(2):
            robj = res['objects'][i]
            F[ngood, i] = robj['flux'][0]
            Fe[ngood, i] = robj['flux_err'][0]
            T[ngood, i] = robj['T']
            Te[ngood, i] = robj['T_err']
        ngood += 1
    assert ngood > 0.9 * ntrial
    F, Fe = F[:ngood], Fe[:ngood]
    T, Te = T[:ngood], Te[:ngood]
    for i in range(2):
        rF = F[:, i].std() / np.sqrt(np.mean(Fe[:, i] ** 2))
        rT = T[:, i].std() / np.sqrt(np.mean(Te[:, i] ** 2))
        assert 0.85 < rF < 1.15, (i, rF)
        assert 0.8 < rT < 1.2, (i, rT)


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


def test_covariance_aware_s2n():
    """
    the total flux s/n is the joint (Wald) value wherever the
    cross-band flux covariance is available: on both the
    per-object and full-errors paths s2n satisfies
    s2n = sqrt(F^T C^-1 F) with the reported flux_cov, and the
    positive cross-band correlation from the shared family
    response puts it below the independent-band quadrature sum.
    On the full-errors path the gauss entries satisfy the same
    identity with gauss_flux_cov
    """
    rng = np.random.RandomState(31)
    offsets = [(-0.625, 0.0), (0.625, 0.0)]
    mbobs = make_mbobs(rng, offsets)

    for kw in ({}, {'full_errors': True}):
        res = run_deblend(mbobs, offsets, **kw)
        assert res['converged']
        for robj in res['objects']:
            C = robj['flux_cov']
            assert C is not None
            assert C[0, 1] > 0
            F = robj['flux']
            expected = np.sqrt(F @ np.linalg.solve(C, F))
            assert np.allclose(robj['s2n'], expected)
            quad = np.sqrt(np.sum((F / robj['flux_err']) ** 2))
            assert robj['s2n'] < quad

            gC = robj.get('gauss_flux_cov')
            if gC is not None:
                gF = robj['gauss_flux']
                gexp = np.sqrt(gF @ np.linalg.solve(gC, gF))
                assert np.allclose(robj['gauss_s2n'], gexp)


def make_star_mbobs(rng, offsets, fluxes=(400.0, 600.0)):
    """point-source scene: the psf profile at each offset, per
    band flux from the fluxes entry"""
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
            obj = psf.withFlux(fluxes[band])
            im += obj.drawImage(
                nx=DIM, ny=DIM, scale=SCALE,
                offset=(du / SCALE, dv / SCALE),
            ).array
        im = im + rng.normal(scale=SIGMAS[band], size=im.shape)
        obs = ngmix.Observation(
            im,
            weight=np.full(im.shape, 1.0 / SIGMAS[band] ** 2),
            jacobian=ngmix.DiagonalJacobian(
                scale=SCALE, row=cen, col=cen,
            ),
            psf=psf_obs,
        )
        ol = ngmix.ObsList()
        ol.append(obs)
        mbobs.append(ol)
    return mbobs


def star_objects(offsets, fixcen=None):
    return [
        {
            'v': dv, 'u': du, 'type': 'star',
            'fixcen': bool(fixcen[k]) if fixcen is not None
            else False,
        }
        for k, (du, dv) in enumerate(offsets)
    ]


def test_full_errors_star_single():
    """m=1 star anchor: with a frozen weight and no neighbors the
    per-object flux errors are exact, so the full errors must
    reproduce them; the cross-band covariance is diagonal (the
    bands share no state)"""
    rng = np.random.RandomState(41)
    offsets = [(0.0, 0.0)]
    mbobs = make_star_mbobs(rng, offsets)

    res = deblend(
        mbobs, star_objects(offsets), tol=1.0e-6, maxiter=2000,
        rng=np.random.RandomState(5),
    )
    resg = deblend(
        mbobs, star_objects(offsets), tol=1.0e-6, maxiter=2000,
        rng=np.random.RandomState(5), full_errors=True,
    )
    assert res['converged']
    assert resg['converged'] and resg['full_errors']

    r0 = res['objects'][0]
    rg = resg['objects'][0]
    assert np.all(np.isfinite(rg['flux_err']))
    assert np.allclose(rg['flux_err'], r0['flux_err'], rtol=2e-2)
    assert np.isfinite(rg['s2n'])

    C = rg['flux_cov']
    assert C is not None and C.shape == (NBAND, NBAND)
    assert np.allclose(np.diag(C), rg['flux_err'] ** 2, rtol=1e-6)
    assert abs(C[0, 1]) < 0.05 * np.sqrt(C[0, 0] * C[1, 1])

    # the structure entries stand: a delta function has none
    assert rg['T'] == 0.0
    assert not np.isfinite(rg['T_err'])


def test_full_errors_star_pair_mc():
    """a blended star pair: the per-object flux errors treat the
    neighbor subtraction as deterministic and underpredict; the
    full errors price the shared-pixel coupling and match the
    observed scatter, with the expected negative member-member
    covariance"""
    from kdeblend.deblender import (
        _Deblender, _prep_epochs, _get_smoothing,
    )
    from kdeblend.full_errors import full_covariance

    offsets = [(-0.25, 0.0), (0.25, 0.0)]
    ntrial = 150

    fluxes = {0: [[], []], 1: [[], []]}
    rep_full = None
    rep_po = None
    rng = np.random.RandomState(3000)
    nconv = 0
    for trial in range(ntrial):
        mbobs = make_star_mbobs(rng, offsets)
        resg = deblend(
            mbobs, star_objects(offsets), tol=1.0e-6,
            maxiter=2000, rng=np.random.RandomState(5),
            full_errors=True,
        )
        if not (resg['converged'] and resg['full_errors']):
            continue
        nconv += 1
        for i in range(2):
            for b in range(NBAND):
                fluxes[i][b].append(
                    resg['objects'][i]['flux'][b]
                )
        if rep_full is None:
            rep_full = [
                resg['objects'][i]['flux_err'].copy()
                for i in range(2)
            ]
            res0 = deblend(
                mbobs, star_objects(offsets), tol=1.0e-6,
                maxiter=2000, rng=np.random.RandomState(5),
            )
            rep_po = [
                res0['objects'][i]['flux_err'].copy()
                for i in range(2)
            ]
    assert nconv > 0.9 * ntrial

    for i in range(2):
        for b in range(NBAND):
            emp = np.std(fluxes[i][b])
            assert np.abs(rep_full[i][b] / emp - 1) < 0.2, (
                i, b, rep_full[i][b] / emp,
            )
            # the per-object errors underpredict for this tight
            # pair
            assert rep_po[i][b] < 0.95 * rep_full[i][b]

    # cross-member covariance: negative (flux splitting), and
    # matching the observed one
    rng2 = np.random.RandomState(77)
    mbobs = make_star_mbobs(rng2, offsets)
    fwhm_smooth, Tsmooth = _get_smoothing(
        mbobs, None, 1.05, np.random.RandomState(3),
    )
    epochs = _prep_epochs(
        mbobs, fwhm_smooth=fwhm_smooth, ap_rad=0.0,
        use_noise_image=False, vcen=0.0, ucen=0.0,
        store_transfer=True,
    )
    deb = _Deblender(
        [epochs] * 2, NBAND, star_objects(offsets),
        fwhm_smooth, Tsmooth, 2000, 1.0e-6,
    )
    gres = deb.go()
    assert gres['converged']
    cov, slices, _ = full_covariance(deb, mbobs)
    b = 0
    ia = slices[0] + b
    ib = slices[1] + b
    rep_corr = cov[ia, ib] / np.sqrt(cov[ia, ia] * cov[ib, ib])
    emp_corr = np.corrcoef(fluxes[0][b], fluxes[1][b])[0, 1]
    assert rep_corr < -0.1
    assert emp_corr < -0.1
    assert np.abs(rep_corr - emp_corr) < 0.2


def test_full_errors_star_galaxy_mc():
    """a star blended with a galaxy: both members' full flux
    errors match the observed scatter, and the star gains a
    cross-band flux covariance through the galaxy's shared
    structure response"""
    star_off = (-0.5, 0.0)
    gal_off = (0.5, 0.0)
    ntrial = 150

    def make_scene(rng):
        # star scene plus a galaxy: reuse the exp machinery from
        # make_mbobs by adding the images
        mbobs = make_star_mbobs(rng, [star_off])
        for band in range(NBAND):
            obs = mbobs[band][0]
            psf = galsim.Gaussian(fwhm=PSF_FWHM)
            gal = galsim.Convolve(
                galsim.Exponential(
                    half_light_radius=HLR, flux=FLUXES[band],
                ),
                psf,
            )
            gim = gal.drawImage(
                nx=DIM, ny=DIM, scale=SCALE,
                offset=(gal_off[0] / SCALE, gal_off[1] / SCALE),
            ).array
            with obs.writeable():
                obs.image = obs.image + gim
        return mbobs

    objects = [
        {'v': star_off[1], 'u': star_off[0], 'type': 'star'},
        {'v': gal_off[1], 'u': gal_off[0], 'type': 'exp',
         'Tguess': 0.3},
    ]

    fluxes = {0: [[], []], 1: [[], []]}
    rep_full = None
    star_fcov = None
    rng = np.random.RandomState(4000)
    nconv = 0
    for trial in range(ntrial):
        mbobs = make_scene(rng)
        resg = deblend(
            mbobs, objects, tol=1.0e-6, maxiter=2000,
            rng=np.random.RandomState(5), full_errors=True,
        )
        if not (resg['converged'] and resg['full_errors']):
            continue
        nconv += 1
        for i in range(2):
            for b in range(NBAND):
                fluxes[i][b].append(
                    resg['objects'][i]['flux'][b]
                )
        if rep_full is None:
            rep_full = [
                resg['objects'][i]['flux_err'].copy()
                for i in range(2)
            ]
            star_fcov = resg['objects'][0]['flux_cov'].copy()
    assert nconv > 0.85 * ntrial

    for i in range(2):
        for b in range(NBAND):
            emp = np.std(fluxes[i][b])
            assert np.abs(rep_full[i][b] / emp - 1) < 0.25, (
                i, b, rep_full[i][b] / emp,
            )

    # the star's cross-band covariance is filled and physical
    assert star_fcov is not None
    rho = star_fcov[0, 1] / np.sqrt(
        star_fcov[0, 0] * star_fcov[1, 1],
    )
    assert -0.9 < rho < 0.9
    emp_rho = np.corrcoef(fluxes[0][0], fluxes[0][1])[0, 1]
    assert np.abs(rho - emp_rho) < 0.25


def test_full_errors_star_chain_fd_fixcen():
    """chain vs FD equivalence on a star-bearing family with a
    fixcen member under recentering: the fixcen center rows are
    pinned (no singular I - J), and the two constructions agree"""
    from kdeblend.deblender import (
        _Deblender, _prep_epochs, _get_smoothing,
    )
    from kdeblend.full_errors import full_covariance

    offsets = [(-0.4, 0.0), (0.4, 0.0)]
    rng = np.random.RandomState(88)
    mbobs = make_star_mbobs(rng, offsets)

    fwhm_smooth, Tsmooth = _get_smoothing(
        mbobs, None, 1.05, np.random.RandomState(3),
    )
    epochs = _prep_epochs(
        mbobs, fwhm_smooth=fwhm_smooth, ap_rad=0.0,
        use_noise_image=False, vcen=0.0, ucen=0.0,
        store_transfer=True,
    )
    deb = _Deblender(
        [epochs] * 2, NBAND,
        star_objects(offsets, fixcen=[False, True]),
        fwhm_smooth, Tsmooth, 2000, 1.0e-6,
        recenter=True, cen_sigma0=0.1,
    )
    gres = deb.go()
    assert gres['converged']

    cov_c, slices, _ = full_covariance(deb, mbobs, use_chain=True)
    cov_f, _, _ = full_covariance(deb, mbobs, use_chain=False)

    assert np.all(np.isfinite(cov_c))
    assert np.all(np.isfinite(cov_f))
    da = np.sqrt(np.diag(cov_c))
    db = np.sqrt(np.diag(cov_f))
    wpos = (da > 0) & (db > 0)
    assert np.allclose(da[wpos], db[wpos], rtol=2e-2)

    # the fixcen member's center variance is pinned to zero;
    # the free member's is positive
    icen1 = slices[1] + NBAND
    assert np.allclose(cov_c[icen1:icen1 + 2, icen1:icen1 + 2],
                       0.0, atol=1e-12)
    icen0 = slices[0] + NBAND
    assert cov_c[icen0, icen0] > 0
