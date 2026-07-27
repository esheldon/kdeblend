"""
tests for recentering: with recenter=True the centers join the
per-sweep updates, moving by the measured pull regularized toward
the detection position, correcting the sub-pixel errors of
detection centroids that otherwise distort blend-member fits
"""
import numpy as np

from kdeblend import deblend, deblend_stamps

from test_bdf_deblend import make_bdf_blend_obs, FWHM_SMOOTH

TOL = 1.0e-8


def test_recenter_single():
    """
    a single object with the input position off by ~0.1": without
    recentering the fit is conditioned on the wrong center and
    cen_pull reports the pull; with recentering the center is
    effectively free (noiseless data, so k ~ 1) and converges to
    the truth, matching the exact-position fit
    """
    comp = dict(v=0.0, u=0.0, e1=0.08, e2=-0.04, T=0.5,
                flux=150.0, fracdev=0.3, TdByTe=1.0)
    obs = make_bdf_blend_obs([comp], 0.9)

    def spec(v, u):
        return dict(v=v, u=u, type='bdf', Tguess=0.4, TdByTe=1.0)

    ref = deblend(
        obs, [spec(0.0, 0.0)], fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )['objects'][0]

    r0 = deblend(
        obs, [spec(0.08, -0.06)], fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )['objects'][0]
    assert np.allclose(r0['cen'], (0.08, -0.06))
    assert np.sqrt(np.sum(r0['cen_pull'] ** 2)) > 0.02

    r = deblend(
        obs, [spec(0.08, -0.06)], fwhm_smooth=FWHM_SMOOTH, tol=TOL,
        recenter=True,
    )['objects'][0]
    assert np.sqrt(np.sum(np.array(r['cen']) ** 2)) < 1.0e-3
    assert abs(r['T'] / ref['T'] - 1) < 1.0e-3
    assert abs(r['e1'] - ref['e1']) < 1.0e-3
    assert abs(r['flux'][0] / ref['flux'][0] - 1) < 1.0e-3


def test_recenter_pair():
    """
    a 2" pair with both input positions off by ~0.1 pixel in
    different directions: the miscentering distorts the blend
    members through the mutual subtraction; the free (noiseless)
    centers converge jointly and recover the exact-position fits
    """
    comps = [
        dict(v=0.0, u=-1.0, e1=0.05, e2=0.02, T=0.5, flux=250.0,
             fracdev=0.5, TdByTe=1.0),
        dict(v=0.0, u=1.0, e1=-0.06, e2=0.04, T=0.4, flux=120.0,
             fracdev=0.3, TdByTe=1.0),
    ]
    obs = make_bdf_blend_obs(comps, 0.9)

    def specs(offs):
        return [
            dict(v=c['v'] + dv, u=c['u'] + du, type='bdf',
                 Tguess=0.4, TdByTe=1.0)
            for c, (dv, du) in zip(comps, offs)
        ]

    exact = deblend(
        obs, specs([(0, 0), (0, 0)]),
        fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )['objects']

    rec = deblend(
        obs, specs([(0.07, -0.02), (-0.05, 0.06)]),
        fwhm_smooth=FWHM_SMOOTH, tol=TOL,
        recenter=True, maxiter=3000,
    )['objects']

    for c, e, r in zip(comps, exact, rec):
        assert np.sqrt(
            (r['cen'][0] - c['v']) ** 2 + (r['cen'][1] - c['u']) ** 2
        ) < 2.0e-3
        assert abs(r['T'] / e['T'] - 1) < 5.0e-3
        assert abs(r['e1'] - e['e1']) < 5.0e-3
        assert abs(r['flux'][0] / e['flux'][0] - 1) < 5.0e-3


def test_recenter_stamps():
    """
    recentering works in stamps mode.  There an object's phase
    center in its own stamps is the stamp jacobian center offset
    by (position - stated position), so the case to correct is an
    object sitting off its stamp center: the stated position picks
    up the offset
    """
    comp = dict(v=0.08, u=-0.06, e1=0.08, e2=-0.04, T=0.5,
                flux=150.0, fracdev=0.4, TdByTe=1.5)
    obs = make_bdf_blend_obs([comp], 0.9)

    res = deblend_stamps(
        [obs],
        [dict(v=0.0, u=0.0, type='bdf', Tguess=0.4, TdByTe=1.5)],
        fwhm_smooth=FWHM_SMOOTH, tol=TOL, recenter=True,
    )
    r = res['objects'][0]
    assert np.sqrt(
        (r['cen'][0] - comp['v']) ** 2
        + (r['cen'][1] - comp['u']) ** 2
    ) < 2.0e-3
    assert abs(r['flux'][0] / comp['flux'] - 1) < 5.0e-3
    assert abs(r['T'] / comp['T'] - 1) < 1.0e-2


def test_recenter_freeze():
    """
    cen_sigma0 = 0 freezes the centers at the detection positions:
    identical to recenter=False
    """
    comp = dict(v=0.0, u=0.0, e1=0.08, e2=-0.04, T=0.5,
                flux=150.0, fracdev=0.3, TdByTe=1.0)
    obs = make_bdf_blend_obs([comp], 0.9)
    spec = [dict(v=0.05, u=-0.04, type='bdf', Tguess=0.4,
                 TdByTe=1.0)]

    r0 = deblend(
        obs, spec, fwhm_smooth=FWHM_SMOOTH, tol=TOL,
    )['objects'][0]
    r = deblend(
        obs, spec, fwhm_smooth=FWHM_SMOOTH, tol=TOL,
        recenter=True, cen_sigma0=0.0,
    )['objects'][0]
    assert np.allclose(r['cen'], (0.05, -0.04))
    assert abs(r['T'] - r0['T']) < 1.0e-10
    assert abs(r['e1'] - r0['e1']) < 1.0e-10
    assert abs(r['flux'][0] - r0['flux'][0]) < 1.0e-8
