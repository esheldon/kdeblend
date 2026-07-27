"""
tests for the composite exp+dev ('bdf') object type in the
deblender

The bdf components are rendered from the same ngmix gaussian
expansions the model fits, so the model is exact and high-s2n
fits must recover the truth
"""
import numpy as np
import galsim
import pytest

from ngmix.prepsfadmom.models import (
    cov_from_e, get_profile_comps,
)
from kdeblend import deblend, deblend_stamps

from _sims import GSPARAMS, make_blend_obs

FWHM_SMOOTH = 1.2
TOL = 1.0e-8


def make_bdf_profile(comp):
    """
    the exact composite: the ngmix exp expansion plus the dev
    expansion with sizes scaled by TdByTe, shared shape and
    center, split by fracdev
    """
    fd = comp['fracdev']
    r = comp['TdByTe']
    parts = []
    for name, fam_flux, fam_T in (
        ('exp', (1 - fd) * comp['flux'], comp['T']),
        ('dev', fd * comp['flux'], r * comp['T']),
    ):
        for frac, cT in get_profile_comps(name):
            cov = cov_from_e(comp['e1'], comp['e2'], cT * fam_T)
            parts.append(
                galsim.Gaussian(
                    sigma=np.linalg.det(cov) ** 0.25,
                    gsparams=GSPARAMS,
                ).shear(e1=comp['e1'], e2=comp['e2'])
                * (fam_flux * frac)
            )
    return galsim.Add(parts).shift(dx=comp['u'], dy=comp['v'])


def make_bdf_blend_obs(comps, psf_fwhm, noise=1.0e-9, dim=64):
    """
    render the composite components through make_blend_obs by
    passing prebuilt profiles via the 'gauss' escape: we build the
    image ourselves instead
    """
    import ngmix

    psf = galsim.Gaussian(fwhm=psf_fwhm, gsparams=GSPARAMS)
    prof = galsim.Add([make_bdf_profile(c) for c in comps])
    scale = 0.25
    im = galsim.Convolve(prof, psf).drawImage(
        nx=dim, ny=dim, scale=scale, method='no_pixel',
    ).array
    psf_im = psf.drawImage(
        nx=dim, ny=dim, scale=scale, method='no_pixel',
    ).array

    cen = (dim - 1) / 2
    jacobian = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)
    psf_obs = ngmix.Observation(
        psf_im, weight=np.ones((dim, dim)) * 1.0e12,
        jacobian=jacobian,
    )
    return ngmix.Observation(
        im, weight=np.ones((dim, dim)) / noise ** 2,
        jacobian=jacobian, psf=psf_obs,
    )


def test_bdf_pair():
    """
    a blended pair of composites: totals, splits, component
    fluxes and structures are recovered in shared-image mode
    """
    comps = [
        dict(v=0.0, u=-1.2, e1=0.1, e2=-0.05, T=0.5,
             flux=200.0, fracdev=0.2, TdByTe=1.0),
        dict(v=0.4, u=1.4, e1=-0.05, e2=0.08, T=0.35,
             flux=120.0, fracdev=0.6, TdByTe=1.0),
    ]
    obs = make_bdf_blend_obs(comps, 0.9)

    objects = [
        dict(v=c['v'], u=c['u'], type='bdf', Tguess=0.4,
             TdByTe=1.0)
        for c in comps
    ]
    res = deblend(
        obs, objects, fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )

    for c, r in zip(comps, res['objects']):
        assert r['type'] == 'bdf'
        assert r['deblend_flags'] == 0
        assert abs(r['flux'][0] / c['flux'] - 1) < 2.0e-3
        assert abs(r['T'] / c['T'] - 1) < 5.0e-3
        assert abs(r['e1'] - c['e1']) < 2.0e-3
        assert abs(r['e2'] - c['e2']) < 2.0e-3
        assert abs(r['fracdev'] - c['fracdev']) < 0.02
        assert abs(
            r['flux_exp'][0] / ((1 - c['fracdev']) * c['flux']) - 1
        ) < 0.05
        assert abs(
            r['flux_dev'][0] / (c['fracdev'] * c['flux']) - 1
        ) < 0.05
        assert r['TdByTe'] == 1.0
        assert np.isfinite(r['fracdev_gls'])
        assert np.isfinite(r['fracdev_err'])
        # gauss entries come for free as for every type
        assert np.isfinite(r['gauss_e1'])


def test_bdf_stamps_single():
    """
    stamps mode: a single composite recovers the truth
    """
    comp = dict(v=0.0, u=0.0, e1=0.08, e2=-0.04, T=0.5,
                flux=150.0, fracdev=0.4, TdByTe=1.5)
    obs = make_bdf_blend_obs([comp], 0.9)

    res = deblend_stamps(
        [obs],
        [dict(v=0.0, u=0.0, type='bdf', Tguess=0.4, TdByTe=1.5)],
        fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )
    r = res['objects'][0]
    assert abs(r['flux'][0] / comp['flux'] - 1) < 2.0e-3
    assert abs(r['T'] / comp['T'] - 1) < 5.0e-3
    assert abs(r['fracdev'] - comp['fracdev']) < 0.02


def test_bdf_shrinkage_freeze():
    """
    with sigma0=0 the model split is frozen at fracdev0; on pure
    exp-expansion data frozen at zero the fit matches the exp
    object type
    """
    gal = dict(kind='gauss', e1=0.08, e2=-0.04, T=0.5, flux=150.0,
               v=0.0, u=0.0)
    # a pure exp rendered exactly: fracdev=0 composite
    comp = dict(v=0.0, u=0.0, e1=0.08, e2=-0.04, T=0.5,
                flux=150.0, fracdev=0.0, TdByTe=1.0)
    obs = make_bdf_blend_obs([comp], 0.9)
    del gal

    res_bdf = deblend(
        obs,
        [dict(v=0.0, u=0.0, type='bdf', Tguess=0.4, TdByTe=1.0,
              fracdev0=0.0, fracdev_sigma0=0.0)],
        fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )
    res_exp = deblend(
        obs, [dict(v=0.0, u=0.0, type='exp', Tguess=0.4)],
        fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )
    rb = res_bdf['objects'][0]
    re = res_exp['objects'][0]
    assert rb['fracdev'] == 0.0
    assert abs(rb['flux'][0] / re['flux'][0] - 1) < 1.0e-6
    assert abs(rb['T'] / re['T'] - 1) < 1.0e-6
    assert abs(rb['e1'] - re['e1']) < 1.0e-8


def test_bdf_error_calibration():
    """
    the joint-sandwich errors (coupled structure and split, with
    the split-noise cross covariance) match the robust MC scatter
    for a noisy isolated composite.  The MAD width is used since
    the free split has a small nonlinear tail
    """
    import ngmix

    comp = dict(v=0.0, u=0.0, e1=0.08, e2=-0.04, T=0.5,
                flux=150.0, fracdev=0.3, TdByTe=1.0)
    obs0 = make_bdf_blend_obs([comp], 0.9)
    im0 = obs0.image.copy()
    noise = 0.25

    rng = np.random.RandomState(19)
    vals = []
    perr = []
    for i in range(50):
        im = im0 + rng.normal(scale=noise, size=im0.shape)
        obs = ngmix.Observation(
            im, weight=np.ones(im0.shape) / noise ** 2,
            jacobian=obs0.jacobian, psf=obs0.psf,
        )
        res = deblend(
            obs,
            [dict(v=0.0, u=0.0, type='bdf', Tguess=0.4,
                  TdByTe=1.0)],
            fwhm_smooth=FWHM_SMOOTH, tol=1.0e-6,
        )
        r = res['objects'][0]
        if r['deblend_flags'] != 0 or r['e_flags'] != 0:
            continue
        vals.append((r['e1'], r['flux'][0], r['fracdev']))
        perr.append((r['e1_err'], r['flux_err'][0],
                     r['fracdev_err']))

    assert len(vals) >= 45
    vals = np.array(vals)
    perr = np.array(perr)
    assert np.all(np.isfinite(perr))

    def mad_sigma(x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))

    for col in range(3):
        ratio = mad_sigma(vals[:, col]) / np.median(perr[:, col])
        assert 0.6 < ratio < 1.5


def test_bdf_validation():
    """
    missing TdByTe and mismatched shrinkage pairs raise
    """
    comp = dict(v=0.0, u=0.0, e1=0.0, e2=0.0, T=0.5,
                flux=100.0, fracdev=0.3, TdByTe=1.0)
    obs = make_bdf_blend_obs([comp], 0.9)

    with pytest.raises(ValueError, match='TdByTe'):
        deblend(
            obs, [dict(v=0.0, u=0.0, type='bdf', Tguess=0.4)],
            fwhm_smooth=FWHM_SMOOTH,
        )

    with pytest.raises(ValueError, match='shrinkage'):
        deblend(
            obs,
            [dict(v=0.0, u=0.0, type='bdf', Tguess=0.4,
                  TdByTe=1.0, fracdev0=0.0)],
            fwhm_smooth=FWHM_SMOOTH,
        )


def test_bdf_fixed_model():
    """
    a bdf entry works as a fixed external model: a neighbor
    subtracted in closed form
    """
    target = dict(kind='gauss', e1=0.05, e2=0.0, T=0.4,
                  flux=100.0, v=0.0, u=-1.0)
    nbr = dict(v=0.0, u=1.8, e1=0.0, e2=0.0, T=0.6,
               flux=300.0, fracdev=0.5, TdByTe=1.0)

    import ngmix
    psf = galsim.Gaussian(fwhm=0.9, gsparams=GSPARAMS)
    from _sims import make_profile
    prof = galsim.Add([
        make_profile(target), make_bdf_profile(nbr),
    ])
    dim = 64
    scale = 0.25
    im = galsim.Convolve(prof, psf).drawImage(
        nx=dim, ny=dim, scale=scale, method='no_pixel',
    ).array
    psf_im = psf.drawImage(
        nx=dim, ny=dim, scale=scale, method='no_pixel',
    ).array
    cen = (dim - 1) / 2
    jacobian = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)
    obs = ngmix.Observation(
        im, weight=np.ones((dim, dim)) * 1.0e18,
        jacobian=jacobian,
        psf=ngmix.Observation(
            psf_im, weight=np.ones((dim, dim)) * 1.0e12,
            jacobian=jacobian,
        ),
    )

    res = deblend(
        obs,
        [dict(v=target['v'], u=target['u'], type='gauss',
              Tguess=0.4)],
        fwhm_smooth=FWHM_SMOOTH, tol=TOL,
        fixed_models=[dict(
            v=nbr['v'], u=nbr['u'], type='bdf',
            flux=np.array([nbr['flux']]),
            e1=nbr['e1'], e2=nbr['e2'], T=nbr['T'],
            fracdev=nbr['fracdev'], TdByTe=nbr['TdByTe'],
        )],
    )
    r = res['objects'][0]
    assert abs(r['flux'][0] / target['flux'] - 1) < 5.0e-3
    assert abs(r['e1'] - target['e1']) < 5.0e-3


def test_make_blend_obs_import():
    """
    keep the shared sim helpers importable from this module's
    directory (guards against test layout changes)
    """
    assert callable(make_blend_obs)
