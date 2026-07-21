"""
simulation helpers for the tests
"""
import numpy as np
import galsim
import ngmix

from ngmix.prepsfadmom.models import cov_from_e

GSPARAMS = galsim.GSParams(
    folding_threshold=1.0e-8,
    maxk_threshold=1.0e-8,
    kvalue_accuracy=1.0e-8,
    xvalue_accuracy=1.0e-8,
)

SCALE = 0.25
DIM = 64


def make_profile(comp):
    """
    galsim pre-psf profile from a component dict with entries
    kind ('gauss', 'star', 'exp', 'dev'), flux, v, u and
    shape/size pars (e1, e2, T for gauss; e1, e2, hlr for exp; e1,
    e2, T for dev, rendered as the exact ngmix 10-gaussian
    expansion)
    """
    if comp['kind'] == 'star':
        p = galsim.DeltaFunction(gsparams=GSPARAMS) * comp['flux']
    elif comp['kind'] == 'gauss':
        cov = cov_from_e(comp['e1'], comp['e2'], comp['T'])
        p = galsim.Gaussian(
            sigma=np.linalg.det(cov) ** 0.25, gsparams=GSPARAMS,
        ).shear(e1=comp['e1'], e2=comp['e2']) * comp['flux']
    elif comp['kind'] == 'exp':
        p = galsim.Exponential(
            half_light_radius=comp['hlr'], gsparams=GSPARAMS,
        ).shear(e1=comp['e1'], e2=comp['e2']) * comp['flux']
    elif comp['kind'] == 'dev':
        from ngmix.prepsfadmom.models import get_profile_comps
        parts = []
        for frac, cT in get_profile_comps('dev'):
            cov = cov_from_e(comp['e1'], comp['e2'], cT * comp['T'])
            parts.append(
                galsim.Gaussian(
                    sigma=np.linalg.det(cov) ** 0.25,
                    gsparams=GSPARAMS,
                ).shear(e1=comp['e1'], e2=comp['e2'])
                * (comp['flux'] * frac)
            )
        p = galsim.Add(parts)
    else:
        raise ValueError(f"bad kind {comp['kind']}")

    return p.shift(dx=comp['u'], dy=comp['v'])


def make_blend_obs(comps, psf_fwhm, noise=1.0e-9, rng=None, dim=DIM):
    """
    render a blend of components convolved with a gaussian psf and
    return an ngmix Observation
    """
    cen = (dim - 1) / 2
    parts = [make_profile(c) for c in comps]
    psf = galsim.Gaussian(fwhm=psf_fwhm, gsparams=GSPARAMS)
    scene = galsim.Convolve(galsim.Add(parts), psf, gsparams=GSPARAMS)
    im = scene.drawImage(nx=dim, ny=dim, scale=SCALE).array
    if rng is not None:
        im = im + rng.normal(scale=noise, size=im.shape)
    psf_im = psf.drawImage(nx=dim, ny=dim, scale=SCALE).array

    jac = ngmix.DiagonalJacobian(scale=SCALE, row=cen, col=cen)
    return ngmix.Observation(
        im, jacobian=jac, weight=np.ones_like(im) / noise**2,
        psf=ngmix.Observation(psf_im, jacobian=jac),
    )


def make_blend_mbobs(comps_per_band, psf_fwhms, noise=1.0e-9, rng=None,
                     dim=DIM):
    """
    a MultiBandObsList for a blend; comps_per_band is a list over
    bands of component lists
    """
    mbobs = ngmix.MultiBandObsList()
    for comps, pf in zip(comps_per_band, psf_fwhms):
        obslist = ngmix.ObsList()
        obslist.append(
            make_blend_obs(comps, pf, noise=noise, rng=rng, dim=dim),
        )
        mbobs.append(obslist)
    return mbobs
