import numpy as np
import galsim
import ngmix

from ngmix.moments import cov_from_e
from ngmix.prepsfadmom.models import det2, get_profile_comps
from kdeblend.render import render_model

from _sims import GSPARAMS, SCALE, DIM, make_blend_obs

PSF_FWHM = 0.9


def _gauss(e1, e2, T):
    cov = cov_from_e(e1, e2, T)
    return galsim.Gaussian(
        sigma=det2(cov) ** 0.25, gsparams=GSPARAMS,
    ).shear(e1=e1, e2=e2)


def test_render_gauss_star():
    """
    rendering truth-parameter fitted-object dicts reproduces the
    noiseless scene for the exact model types
    """
    gal = dict(kind='gauss', e1=-0.05, e2=0.08, T=0.35, flux=5.0,
               v=0.2, u=1.5)
    star = dict(kind='star', flux=20.0, v=-0.4, u=-1.0)
    obs = make_blend_obs([gal, star], PSF_FWHM)

    objects = [
        dict(type='gauss', flux=np.array([gal['flux']]),
             cen=np.array([gal['v'], gal['u']]),
             T=gal['T'], e1=gal['e1'], e2=gal['e2']),
        dict(type='star', flux=np.array([star['flux']]),
             cen=np.array([star['v'], star['u']]),
             T=0.0, e1=np.nan, e2=np.nan),
    ]
    model = render_model(objects, obs, 0)

    assert model.shape == obs.image.shape
    maxdiff = np.abs(model - obs.image).max()
    # the star is the worst case for the interpolated psf, at the
    # few-e-3 accuracy of the default quintic interpolant
    assert maxdiff < 5.0e-3 * obs.image.max()


def test_render_exp_mixture():
    """
    the rendered exp model matches a direct galsim rendering of the
    same 6-gaussian expansion
    """
    e1, e2, T = 0.1, -0.05, 0.8
    flux, v, u = 40.0, -0.3, 0.9

    # any scene; only the psf, jacobian and shape are used
    obs = make_blend_obs([dict(kind='star', flux=1.0, v=0, u=0)],
                         PSF_FWHM)

    objects = [
        dict(type='exp', flux=np.array([flux]), cen=np.array([v, u]),
             T=T, e1=e1, e2=e2),
    ]
    model = render_model(objects, obs, 0)

    mix = galsim.Add([
        _gauss(e1, e2, cT * T) * (flux * frac)
        for frac, cT in get_profile_comps('exp')
    ]).shift(dx=u, dy=v)
    psf = galsim.Gaussian(fwhm=PSF_FWHM, gsparams=GSPARAMS)
    expected = galsim.Convolve(mix, psf, gsparams=GSPARAMS).drawImage(
        nx=DIM, ny=DIM, scale=SCALE,
    ).array

    maxdiff = np.abs(model - expected).max()
    assert maxdiff < 1.0e-3 * expected.max()


def test_render_offcenter_jacobian():
    """
    the model lands at jacobian center plus the object offset for a
    jacobian center away from the stamp center
    """
    row0, col0 = 20.3, 35.7
    v, u = 0.5, -0.75

    cen = (DIM - 1) / 2
    psf = galsim.Gaussian(fwhm=PSF_FWHM, gsparams=GSPARAMS)
    psf_im = psf.drawImage(nx=DIM, ny=DIM, scale=SCALE).array
    psf_jac = ngmix.DiagonalJacobian(scale=SCALE, row=cen, col=cen)

    obs = ngmix.Observation(
        np.zeros((DIM, DIM)),
        jacobian=ngmix.DiagonalJacobian(scale=SCALE, row=row0,
                                        col=col0),
        psf=ngmix.Observation(psf_im, jacobian=psf_jac),
    )

    objects = [
        dict(type='gauss', flux=np.array([10.0]), cen=np.array([v, u]),
             T=0.4, e1=0.0, e2=0.0),
    ]
    model = render_model(objects, obs, 0)

    rows, cols = np.mgrid[0:DIM, 0:DIM]
    rowcen = (rows * model).sum() / model.sum()
    colcen = (cols * model).sum() / model.sum()

    assert np.abs(rowcen - (row0 + v / SCALE)) < 0.01
    assert np.abs(colcen - (col0 + u / SCALE)) < 0.01


def test_render_offcenter_psf():
    """
    a psf stamp with the psf off the stamp center, recorded in the
    psf jacobian center, renders the same model as a centered psf
    """
    gal = dict(kind='gauss', e1=0.1, e2=-0.05, T=0.5, flux=10.0,
               v=0.3, u=-0.5)
    obs = make_blend_obs([gal], PSF_FWHM)

    doff = (0.37, -0.29)  # dx, dy pixels
    cen = (DIM - 1) / 2
    psf = galsim.Gaussian(fwhm=PSF_FWHM, gsparams=GSPARAMS)
    psf_im = psf.drawImage(
        nx=DIM, ny=DIM, scale=SCALE, offset=doff,
    ).array
    psf_jac = ngmix.DiagonalJacobian(
        scale=SCALE, row=cen + doff[1], col=cen + doff[0],
    )
    obs_off = ngmix.Observation(
        obs.image.copy(),
        jacobian=obs.jacobian,
        psf=ngmix.Observation(psf_im, jacobian=psf_jac),
    )

    objects = [
        dict(type='gauss', flux=np.array([gal['flux']]),
             cen=np.array([gal['v'], gal['u']]),
             T=gal['T'], e1=gal['e1'], e2=gal['e2']),
    ]
    model = render_model(objects, obs, 0)
    model_off = render_model(objects, obs_off, 0)

    maxdiff = np.abs(model - model_off).max()
    assert maxdiff < 1.0e-3 * model.max()


def test_render_bad_pars():
    """
    pathological fitted parameters, possible for noisy fits, still
    render finite images
    """
    obs = make_blend_obs([dict(kind='star', flux=1.0, v=0, u=0)],
                         PSF_FWHM)

    objects = [
        dict(type='gauss', flux=np.array([5.0]),
             cen=np.array([0.0, 0.0]), T=-1.0, e1=0.8, e2=0.8),
        dict(type='exp', flux=np.array([5.0]),
             cen=np.array([1.0, -1.0]), T=0.01, e1=-0.95, e2=0.3),
    ]
    model = render_model(objects, obs, 0)
    assert np.all(np.isfinite(model))
