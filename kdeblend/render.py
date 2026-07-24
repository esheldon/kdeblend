"""
render fitted object models in real space

Models are built as pre-psf galsim profiles from the fitted
parameters and convolved with the psf of an observation, so the
result is directly comparable to the observed image.

This module needs galsim, which is not a dependency of the package;
it is not imported by the package __init__.  Use

    from kdeblend import render
"""
import numpy as np
import galsim

from ngmix.prepsfadmom.models import (
    bdf_comps, cov_from_e, det2, get_profile_comps,
)


def render_model(objects, obs, band):
    """
    Render the summed fitted models for one band in real space,
    convolved with the psf of the given observation.

    Parameters
    ----------
    objects: list of dicts
        The fitted objects, the 'objects' entry of the deblend
        result.
    obs: Observation
        A single observation; the model is drawn with its psf,
        jacobian and image shape.
    band: int
        The band index into the object fluxes.

    Returns
    -------
    image array with the same shape as obs.image
    """
    parts = [_object_profile(obj, band) for obj in objects]
    psf = _get_psf_interp(obs.psf)
    scene = galsim.Convolve(galsim.Add(parts), psf)

    nrow, ncol = obs.image.shape
    jrow, jcol = obs.jacobian.get_cen()
    offset = (jcol - (ncol - 1) / 2, jrow - (nrow - 1) / 2)

    # the interpolated psf includes the pixel, so no_pixel
    return scene.drawImage(
        nx=ncol, ny=nrow, wcs=obs.jacobian.get_galsim_wcs(),
        offset=offset, method='no_pixel',
    ).array


def _object_profile(obj, band):
    """pre-psf galsim profile of a fitted object in one band; the exp
    model is the 6-gaussian expansion used in the fit"""
    flux = obj['flux'][band]
    if obj['type'] == 'star':
        p = galsim.DeltaFunction() * flux
    elif obj['type'] == 'gauss':
        p = _gauss_profile(obj['e1'], obj['e2'], obj['T']) * flux
    elif obj['type'] == 'bdf':
        # the composite table from the fitted split; a flagged nan
        # split renders as pure exp
        fracdev = obj['fracdev']
        if not np.isfinite(fracdev):
            fracdev = 0.0
        p = galsim.Add([
            _gauss_profile(obj['e1'], obj['e2'], cT * obj['T'])
            * (flux * frac)
            for frac, cT in bdf_comps(fracdev, obj['TdByTe'])
        ])
    else:
        p = galsim.Add([
            _gauss_profile(obj['e1'], obj['e2'], cT * obj['T'])
            * (flux * frac)
            for frac, cT in get_profile_comps(obj['type'])
        ])

    return p.shift(dx=obj['cen'][1], dy=obj['cen'][0])


def _gauss_profile(e1, e2, T):
    """galsim gaussian with covariance cov_from_e(e1, e2, T); the
    size and |e| are limited, and flagged nan shapes render round,
    so noisy and partially flagged fits still render"""
    if not np.isfinite(T):
        T = 0.0
    T = max(T, 1.0e-6)
    if not (np.isfinite(e1) and np.isfinite(e2)):
        e1 = 0.0
        e2 = 0.0
    etot = np.hypot(e1, e2)
    if etot > 0.95:
        e1 = e1 * 0.95 / etot
        e2 = e2 * 0.95 / etot
    cov = cov_from_e(e1, e2, T)
    return galsim.Gaussian(sigma=det2(cov) ** 0.25).shear(e1=e1, e2=e2)


def _get_psf_interp(psf_obs):
    """interpolated image of the psf, centered at its jacobian
    center; includes the pixel, so draw models with method='no_pixel'
    """
    nrow, ncol = psf_obs.image.shape
    jrow, jcol = psf_obs.jacobian.get_cen()
    offset = (jcol - (ncol - 1) / 2, jrow - (nrow - 1) / 2)
    return galsim.InterpolatedImage(
        galsim.Image(
            psf_obs.image.copy(),
            wcs=psf_obs.jacobian.get_galsim_wcs(),
        ),
        offset=offset,
    )
