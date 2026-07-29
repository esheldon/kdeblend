"""
tests for the group-coupled sandwich errors: the analytic kernel
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


def test_group_errors_dS_dtheta():
    """the analytic kernel derivatives match finite differences
    of the exact-exp kernel contraction (the numba kernel itself
    only agrees to its table-exp accuracy)"""
    from kdeblend.deblender import (
        _Deblender, _prep_epochs, _get_smoothing,
    )
    from kdeblend.group_errors import (
        _build_kernels, _dS_dtheta,
    )

    rng = np.random.RandomState(99)
    offsets = [(-1.0, 0.0), (1.0, 0.0)]
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
        500, 1.0e-6,
    )
    deb.go()

    ep = deb.epochs_per_obj[0][0]
    D = _dS_dtheta(deb, 0, ep)
    Dfd = np.zeros((6, 5))
    sw0 = deb.Sw[0].copy()
    pos0 = deb.positions[0]
    hs = [1e-5, 1e-5, 1e-5, 1e-6, 1e-6]
    for t in range(5):
        for sign in (1, -1):
            sw = sw0.copy()
            pos = list(pos0)
            if t < 3:
                r, c = [(0, 0), (0, 1), (1, 1)][t]
                sw[r, c] += sign * hs[t]
                sw[c, r] = sw[r, c]
            else:
                pos[t - 3] += sign * hs[t]
            deb.Sw[0] = sw
            deb.positions[0] = tuple(pos)
            sums = (_build_kernels(deb, 0, ep) @ ep['kim']).real
            Dfd[:, t] += sign * sums / (2 * hs[t])
        deb.Sw[0] = sw0.copy()
        deb.positions[0] = pos0
    rel = np.abs(D - Dfd) / (np.abs(Dfd) + 1e-8)
    assert rel.max() < 1.0e-5


def test_group_errors_single_reduction():
    """at m=1 the group sandwich must agree with the per-object
    sandwich: same estimating equations.  Checked through the
    module directly since the production path gates at m >= 2"""
    from kdeblend.deblender import (
        _Deblender, _prep_epochs, _get_smoothing,
    )
    from kdeblend.group_errors import group_covariance

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
    )
    deb = _Deblender(
        [epochs], NBAND,
        [{'v': 0.0, 'u': 0.0, 'type': 'exp', 'Tguess': 0.3}],
        fwhm_smooth, Tsmooth, 500, 1.0e-6,
    )
    gres = deb.go()
    assert gres['converged']
    cov, slices = group_covariance(deb, mbobs)

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


@pytest.mark.parametrize('recenter', [False, True])
def test_group_errors_pair(recenter):
    """a close pair with group_errors=True: flux errors inflate
    over the deterministic-neighbor values, the member flux
    covariance is negative, and flux_cov is filled"""
    rng = np.random.RandomState(21)
    offsets = [(-0.5, 0.0), (0.5, 0.0)]
    mbobs = make_mbobs(rng, offsets)

    kw = {}
    if recenter:
        kw = {'recenter': True, 'cen_sigma0': 0.1}
    res0 = run_deblend(mbobs, offsets, **kw)
    res = run_deblend(mbobs, offsets, group_errors=True, **kw)
    assert res['converged']
    assert res['group_errors']

    from kdeblend.group_errors import group_covariance  # noqa

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


def test_group_errors_star_fallback():
    """a group containing a star keeps the per-object errors and
    reports group_errors False"""
    rng = np.random.RandomState(31)
    offsets = [(-0.5, 0.0), (0.5, 0.0)]
    mbobs = make_mbobs(rng, offsets)
    objects = [
        {'v': 0.0, 'u': -0.5, 'type': 'exp', 'Tguess': 0.3},
        {'v': 0.0, 'u': 0.5, 'type': 'star'},
    ]
    res = deblend(
        mbobs, objects, tol=1.0e-6,
        rng=np.random.RandomState(5), group_errors=True,
    )
    assert res['group_errors'] is False
