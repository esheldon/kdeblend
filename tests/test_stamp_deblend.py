"""
The deblender can process a field as one set of images, with a
single FFT of the full scene (deblend), or with a postage stamp per
object (deblend_stamps): each object is measured from a small stamp
cut around it, with its jacobian centered on the object, and the
group of objects is deblended jointly from the collection of stamps.
Neighbor light in a stamp is subtracted in closed form using the
neighbor models fit from their own stamps.

The objects deblended together must include every object whose light
significantly contaminates any member's measurement: a bright
unmodeled neighbor of a member corrupts the member's fit, which then
biases other objects through the member's wrong model.  Fields are
processed by friends-of-friends groups, which provide exactly that
closure.

This test builds a wide field of tightly blended three-object
clusters and checks that the stamp-based results agree with the
full-image results and the truth.  Run directly for a timing and
accuracy report:

    python tests/test_stamp_deblend.py
"""
import time

import numpy as np
import galsim
import ngmix

from kdeblend import deblend, deblend_stamps

from _sims import (
    GSPARAMS, SCALE, make_profile, make_blend_obs, make_blend_mbobs,
)

PSF_FWHMS = [1.1, 0.95, 0.8]
FWHM_SMOOTH = 1.2
TOL = 1.0e-6

DIM = 224
STAMP_DIM = 48
GROUP_LINK = 5.0

CLUSTER_GRID = [-16.0, 0.0, 16.0]
KINDS = ['exp', 'gauss', 'star']


def make_scene(rng):
    """
    a wide field of three-object clusters: each cluster has a
    brighter central object with two satellites at 1.5-2.5 arcsec,
    mixed types, and a different color for every object
    """
    scene = []
    icl = 0
    for vc in CLUSTER_GRID:
        for uc in CLUSTER_GRID:
            v0 = vc + rng.uniform(-1.5, 1.5)
            u0 = uc + rng.uniform(-1.5, 1.5)
            pos = [(v0, u0)]
            for _ in range(2):
                r = rng.uniform(1.5, 2.5)
                theta = rng.uniform(0, 2 * np.pi)
                pos.append(
                    (v0 + r * np.sin(theta), u0 + r * np.cos(theta)),
                )

            for k, (v, u) in enumerate(pos):
                kind = KINDS[(icl + k) % len(KINDS)]
                comp = dict(
                    kind=kind,
                    v=v,
                    u=u,
                    base=(rng.uniform(30.0, 100.0) if k == 0
                          else rng.uniform(5.0, 60.0)),
                    colors=rng.uniform(0.6, 1.6, size=3),
                )
                if kind != 'star':
                    comp['e1'] = rng.uniform(-0.2, 0.2)
                    comp['e2'] = rng.uniform(-0.2, 0.2)
                if kind == 'exp':
                    comp['hlr'] = rng.uniform(0.3, 0.55)
                elif kind == 'gauss':
                    comp['T'] = rng.uniform(0.3, 0.5)
                scene.append(comp)
            icl += 1
    return scene


def find_groups(objects, link):
    """
    friends-of-friends groups with the given link length
    """
    unused = set(range(len(objects)))
    groups = []
    while unused:
        stack = [unused.pop()]
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            close = [
                j for j in unused
                if np.hypot(
                    objects[j]['v'] - objects[i]['v'],
                    objects[j]['u'] - objects[i]['u'],
                ) < link
            ]
            for j in close:
                unused.remove(j)
            stack.extend(close)
        groups.append(sorted(group))
    return groups


def cut_stamp_obs(obs, v, u, nrow, ncol):
    """
    cut a postage stamp centered on an object, with the stamp
    jacobian at the object position and the psf image cut to the
    stamp size
    """
    dim_r, dim_c = obs.image.shape
    row = (dim_r - 1) / 2 + v / SCALE
    col = (dim_c - 1) / 2 + u / SCALE
    row_start = int(np.clip(
        round(row - (nrow - 1) / 2), 0, dim_r - nrow,
    ))
    col_start = int(np.clip(
        round(col - (ncol - 1) / 2), 0, dim_c - ncol,
    ))
    sl = np.s_[row_start:row_start + nrow, col_start:col_start + ncol]

    pim = obs.psf.image
    pstart_r = (pim.shape[0] - nrow) // 2
    pstart_c = (pim.shape[1] - ncol) // 2
    psl = np.s_[pstart_r:pstart_r + nrow, pstart_c:pstart_c + ncol]
    prow0, pcol0 = obs.psf.jacobian.get_cen()

    return ngmix.Observation(
        obs.image[sl].copy(),
        weight=obs.weight[sl].copy(),
        jacobian=ngmix.DiagonalJacobian(
            scale=SCALE, row=row - row_start, col=col - col_start,
        ),
        psf=ngmix.Observation(
            pim[psl].copy(),
            jacobian=ngmix.DiagonalJacobian(
                scale=SCALE,
                row=prow0 - pstart_r, col=pcol0 - pstart_c,
            ),
        ),
    )


def get_stamps_mbobs(mbobs, obj, sdim):
    """
    the per-object MultiBandObsList of stamps for one object
    """
    m = ngmix.MultiBandObsList()
    for obslist in mbobs:
        sobslist = ngmix.ObsList()
        sobslist.append(
            cut_stamp_obs(obslist[0], obj['v'], obj['u'], sdim, sdim),
        )
        m.append(sobslist)
    return m


def deblend_stamps_field(mbobs, objects):
    """
    deblend the field with a postage stamp per object, jointly per
    friends-of-friends group.  Returns the per-object results in
    input order.
    """
    groups = find_groups(objects, GROUP_LINK)
    results = [None] * len(objects)
    for group in groups:
        mbobs_list = [
            get_stamps_mbobs(mbobs, objects[j], STAMP_DIM)
            for j in group
        ]
        res = deblend_stamps(
            mbobs_list, [objects[j] for j in group],
            fwhm_smooth=FWHM_SMOOTH, tol=TOL,
        )
        for k, j in enumerate(group):
            results[j] = res['objects'][k]
    return results


def test_stamps_match_shared():
    """
    given the same pixels, per-object stamps with jacobians
    recentered on the objects give the same result as the
    shared-image mode
    """
    gal = dict(kind='gauss', e1=0.10, e2=-0.05, T=0.40, flux=6.0,
               v=0.3, u=-1.0)
    nbr = dict(kind='exp', e1=0.05, e2=0.02, hlr=0.5, flux=45.0,
               v=-0.2, u=1.0)
    obs = make_blend_obs([gal, nbr], 0.9)
    dim = obs.image.shape[0]
    cen = (dim - 1) / 2

    objects = [
        dict(v=gal['v'], u=gal['u'], Tguess=0.4),
        dict(v=nbr['v'], u=nbr['u'], type='exp', Tguess=0.4),
    ]

    res = deblend(obs, objects, fwhm_smooth=FWHM_SMOOTH)

    mbobs_list = []
    for o in objects:
        mbobs_list.append(ngmix.Observation(
            obs.image.copy(),
            weight=obs.weight.copy(),
            jacobian=ngmix.DiagonalJacobian(
                scale=SCALE,
                row=cen + o['v'] / SCALE,
                col=cen + o['u'] / SCALE,
            ),
            psf=obs.psf,
        ))
    sres = deblend_stamps(mbobs_list, objects, fwhm_smooth=FWHM_SMOOTH)

    for r, sr in zip(res['objects'], sres['objects']):
        assert np.allclose(sr['flux'], r['flux'], rtol=1.0e-6)
        assert np.allclose(sr['T'], r['T'], rtol=1.0e-6, atol=1.0e-9)
        if np.isfinite(r['e1']):
            assert np.abs(sr['e1'] - r['e1']) < 1.0e-6
            assert np.abs(sr['e2'] - r['e2']) < 1.0e-6


def _make_rect_obs(comps, nrow, ncol, psf_fwhm=0.9):
    """noiseless blend observation on a rectangular grid"""
    psf = galsim.Gaussian(fwhm=psf_fwhm, gsparams=GSPARAMS)
    scene = galsim.Convolve(
        galsim.Add([make_profile(c) for c in comps]),
        psf, gsparams=GSPARAMS,
    )
    im = scene.drawImage(nx=ncol, ny=nrow, scale=SCALE).array
    psf_im = psf.drawImage(nx=ncol, ny=nrow, scale=SCALE).array
    jac = ngmix.DiagonalJacobian(
        scale=SCALE, row=(nrow - 1) / 2, col=(ncol - 1) / 2,
    )
    return ngmix.Observation(
        im,
        weight=np.ones_like(im) * 1.0e18,
        jacobian=jac,
        psf=ngmix.Observation(psf_im, jacobian=jac),
    )


def test_rect_images():
    """
    rectangular full images and rectangular per-object stamps give
    the same results as square images
    """
    gal = dict(kind='gauss', e1=0.15, e2=-0.08, T=0.5, flux=5.0,
               v=0.05, u=-1.0)
    nbr = dict(kind='exp', e1=-0.05, e2=0.10, hlr=0.4, flux=20.0,
               v=-0.05, u=1.0)
    comps = [gal, nbr]
    objects = [
        dict(v=gal['v'], u=gal['u'], Tguess=0.4),
        dict(v=nbr['v'], u=nbr['u'], type='exp', Tguess=0.4),
    ]

    res_sq = deblend(
        _make_rect_obs(comps, 64, 64), objects,
        fwhm_smooth=FWHM_SMOOTH,
    )
    obs_rect = _make_rect_obs(comps, 64, 88)
    res_rect = deblend(obs_rect, objects, fwhm_smooth=FWHM_SMOOTH)

    for rs, rr in zip(res_sq['objects'], res_rect['objects']):
        assert np.allclose(rr['flux'], rs['flux'], rtol=1.0e-4)
        assert np.abs(rr['T'] / rs['T'] - 1) < 1.0e-3

    # rectangular per-object stamps cut from the rectangular image
    mbobs_list = []
    for o in objects:
        m = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(cut_stamp_obs(obs_rect, o['v'], o['u'], 40, 56))
        m.append(obslist)
        mbobs_list.append(m)
    res_st = deblend_stamps(
        mbobs_list, objects, fwhm_smooth=FWHM_SMOOTH,
    )
    for rr, rst in zip(res_rect['objects'], res_st['objects']):
        assert np.allclose(rst['flux'], rr['flux'], rtol=1.0e-3)
        assert np.abs(rst['T'] / rr['T'] - 1) < 1.0e-3


def test_deblend_stamps_errors():
    """
    mismatched inputs are caught
    """
    obs = make_blend_obs(
        [dict(kind='star', flux=1.0, v=0, u=0)], 0.9,
    )
    objects = [dict(v=0.0, u=0.0), dict(v=1.0, u=1.0)]
    try:
        deblend_stamps([obs], objects, fwhm_smooth=FWHM_SMOOTH)
        raise AssertionError('should have raised')
    except ValueError:
        pass


def test_stamp_vs_full():
    """
    the stamp-based approach matches the full-image approach and the
    truth on a wide field of blended clusters
    """
    rng = np.random.RandomState(31)
    scene = make_scene(rng)
    nband = len(PSF_FWHMS)

    comps_per_band = [
        [dict(c, flux=c['base'] * c['colors'][band]) for c in scene]
        for band in range(nband)
    ]
    mbobs = make_blend_mbobs(comps_per_band, PSF_FWHMS, dim=DIM)

    objects = [
        dict(v=c['v'], u=c['u'], type=c['kind'], Tguess=0.4)
        for c in scene
    ]
    groups = find_groups(objects, GROUP_LINK)
    assert all(len(g) == 3 for g in groups)

    # warm the numba jit before timing
    deblend_stamps_field(mbobs, objects[:3])

    t0 = time.perf_counter()
    res_full = deblend(
        mbobs, objects, fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    stamp_objects = deblend_stamps_field(mbobs, objects)
    t_stamps = time.perf_counter() - t0

    ftrue = np.array([
        [c['base'] * c['colors'][band] for band in range(nband)]
        for c in scene
    ])
    ffull = np.array([r['flux'] for r in res_full['objects']])
    fstamp = np.array([r['flux'] for r in stamp_objects])

    err_full = np.abs(ffull / ftrue - 1)
    err_stamp = np.abs(fstamp / ftrue - 1)
    agree = np.abs(fstamp / ffull - 1)

    print()
    print(f'{len(scene)} objects in {len(groups)} groups, '
          f'{DIM}^2 image, {STAMP_DIM}^2 stamps')
    print(f'time full:   {t_full:.2f} sec  '
          f'({res_full["numiter"]} sweeps)')
    print(f'time stamps: {t_stamps:.2f} sec')
    print(f'flux err vs truth, full:   max {err_full.max():.2e}  '
          f'median {np.median(err_full):.2e}')
    print(f'flux err vs truth, stamps: max {err_stamp.max():.2e}  '
          f'median {np.median(err_stamp):.2e}')
    print(f'stamp vs full flux: max {agree.max():.2e}  '
          f'median {np.median(agree):.2e}')

    # both approaches recover the truth; the residual errors are
    # dominated by the 6-gaussian exp model mismatch in these tight
    # blends and are common to the two approaches
    assert err_full.max() < 2.0e-2
    assert err_stamp.max() < 2.0e-2

    # the two approaches agree much more closely than either matches
    # truth
    assert agree.max() < 1.0e-3

    # structure also agrees between the approaches for the galaxies
    for cs, rf, rs in zip(scene, res_full['objects'], stamp_objects):
        if cs['kind'] == 'star':
            continue
        assert np.abs(rs['T'] / rf['T'] - 1) < 2.0e-3
        assert np.abs(rs['e1'] - rf['e1']) < 2.0e-3
        assert np.abs(rs['e2'] - rf['e2']) < 2.0e-3


if __name__ == '__main__':
    test_stamps_match_shared()
    test_deblend_stamps_errors()
    test_stamp_vs_full()
