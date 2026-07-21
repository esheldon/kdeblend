import numpy as np
import pytest

from ngmix.moments import fwhm_to_T
from ngmix.prepsfadmom.prep import prep_epoch
from ngmix.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums

from kdeblend import deblend
from ngmix.prepsfadmom.models import cov_from_e, model_ksums

from _sims import make_blend_obs, make_blend_mbobs

FWHM_SMOOTH = 1.2
TSMOOTH = fwhm_to_T(FWHM_SMOOTH)


def test_closed_form_sums():
    """
    the closed-form model sums match the k-space sums measured from a
    rendered scene containing only the model
    """
    comp = dict(kind='gauss', e1=-0.05, e2=0.08, T=0.35, flux=5.0,
                v=0.2, u=1.5)
    obs = make_blend_obs([comp], 0.9)

    ep = prep_epoch(obs, band=0, fwhm_smooth=FWHM_SMOOTH, ap_rad=0)

    # measure with the weight centered on a different location
    vw, uw = -0.1, -1.5
    Sw = np.diag([(0.5 + TSMOOTH) / 2] * 2)
    esums = np.zeros(6)
    alpha, beta = get_phase_angles(ep, vw, uw)
    admom_ksums(
        ep['kim'], ep['iy'], ep['ix'], ep['dim'], alpha, beta,
        ep['kv'], ep['ku'], Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
        esums,
    )

    model = {
        'type': 'gauss',
        'cov_sm': cov_from_e(comp['e1'], comp['e2'], comp['T'])
        + np.diag([TSMOOTH / 2] * 2),
        'F': np.array([comp['flux']]),
    }
    csums = model_ksums(
        model, 0, comp['v'] - vw, comp['u'] - uw, Sw,
        ep['detAtinv'], TSMOOTH,
    )

    assert np.all(np.abs(esums / csums - 1) < 1.0e-3)


@pytest.mark.parametrize('sep', [2.5, 1.0])
def test_gauss_pair_multiband(sep):
    """
    a two-gaussian blend in three bands with different psfs: structure
    and per-band fluxes are recovered essentially exactly, down to
    separations well below the psf size
    """
    psf_fwhms = [1.1, 0.9, 0.8]
    objA = dict(kind='gauss', e1=0.15, e2=-0.08, T=0.5)
    objB = dict(kind='gauss', e1=-0.05, e2=0.10, T=0.3)
    fluxesA = np.array([2.0, 3.5, 4.5])
    fluxesB = np.array([6.0, 5.0, 4.0])
    posA = (0.05, -sep / 2)
    posB = (-0.05, sep / 2)

    comps_per_band = []
    for b in range(3):
        comps_per_band.append([
            dict(objA, flux=fluxesA[b], v=posA[0], u=posA[1]),
            dict(objB, flux=fluxesB[b], v=posB[0], u=posB[1]),
        ])
    mbobs = make_blend_mbobs(comps_per_band, psf_fwhms)

    res = deblend(
        mbobs,
        [
            dict(v=posA[0], u=posA[1], Tguess=0.4),
            dict(v=posB[0], u=posB[1], Tguess=0.4),
        ],
        fwhm_smooth=FWHM_SMOOTH,
    )

    for r, obj, fluxes in [
        (res['objects'][0], objA, fluxesA),
        (res['objects'][1], objB, fluxesB),
    ]:
        assert np.abs(r['e1'] - obj['e1']) < 1.0e-3
        assert np.abs(r['e2'] - obj['e2']) < 1.0e-3
        assert np.abs(r['T'] / obj['T'] - 1) < 1.0e-3
        assert np.all(np.abs(r['flux'] / fluxes - 1) < 1.0e-3)
        # centers are good, so the residual pull is small
        assert np.hypot(*r['cen_pull']) < 1.0e-3


def test_star_blends():
    """
    a bright star next to a faint galaxy, with the star modeled both
    as a fixed delta function and as a free gaussian; the free fit
    converges to zero size
    """
    sep = 1.5
    gal = dict(kind='gauss', e1=0.10, e2=-0.05, T=0.40, flux=3.5,
               v=0.05, u=-sep / 2)
    star = dict(kind='star', flux=105.0, v=-0.05, u=sep / 2)
    obs = make_blend_obs([gal, star], 0.9)

    for stype in ['star', 'gauss']:
        res = deblend(
            obs,
            [
                dict(v=gal['v'], u=gal['u'], Tguess=0.4),
                dict(v=star['v'], u=star['u'], type=stype, Tguess=0.4),
            ],
            fwhm_smooth=FWHM_SMOOTH,
        )
        rgal, rstar = res['objects']
        assert np.abs(rgal['e1'] - gal['e1']) < 1.0e-3
        assert np.abs(rgal['T'] / gal['T'] - 1) < 1.0e-2
        assert np.abs(rgal['flux'][0] / gal['flux'] - 1) < 1.0e-3
        assert np.abs(rstar['flux'][0] / star['flux'] - 1) < 1.0e-3
        if stype == 'gauss':
            # the star fit converges to essentially zero size
            assert np.abs(rstar['T']) < 1.0e-3

    # faint star next to a bright galaxy with the fixed model
    faint_star = dict(kind='star', flux=3.5, v=0.05, u=-sep / 2)
    bgal = dict(kind='gauss', e1=0.05, e2=0.02, T=0.6, flux=105.0,
                v=-0.05, u=sep / 2)
    obs = make_blend_obs([faint_star, bgal], 0.9)
    res = deblend(
        obs,
        [
            dict(v=faint_star['v'], u=faint_star['u'], type='star'),
            dict(v=bgal['v'], u=bgal['u'], Tguess=0.5),
        ],
        fwhm_smooth=FWHM_SMOOTH,
    )
    assert np.abs(res['objects'][0]['flux'][0] / faint_star['flux'] - 1) \
        < 1.0e-3
    assert np.abs(res['objects'][1]['flux'][0] / bgal['flux'] - 1) < 1.0e-3


def test_exp_bright_neighbor():
    """
    a faint gaussian near a bright exponential: the 6-gaussian exp
    model for the neighbor reduces the contamination of the faint
    object by more than an order of magnitude relative to a single
    gaussian neighbor model
    """
    sep = 1.5
    gal = dict(kind='gauss', e1=0.10, e2=-0.05, T=0.40, flux=3.5,
               v=0.05, u=-sep / 2)
    nbr = dict(kind='exp', e1=0.05, e2=0.02, hlr=0.5, flux=105.0,
               v=-0.05, u=sep / 2)
    obs = make_blend_obs([gal, nbr], 0.9)

    dff = {}
    for btype in ['exp', 'gauss']:
        res = deblend(
            obs,
            [
                dict(v=gal['v'], u=gal['u'], Tguess=0.4),
                dict(v=nbr['v'], u=nbr['u'], type=btype, Tguess=0.4),
            ],
            fwhm_smooth=FWHM_SMOOTH,
        )
        dff[btype] = np.abs(res['objects'][0]['flux'][0] / gal['flux'] - 1)

    assert dff['exp'] < 0.05
    assert dff['gauss'] > 0.2
    assert dff['exp'] < dff['gauss'] / 10


def test_noisy_smoke():
    """
    noisy multi-band blends run without failures and give sensible
    fluxes
    """
    ntrial = 20
    rng = np.random.RandomState(42)
    psf_fwhms = [1.1, 0.9, 0.8]
    sep = 2.0
    fluxesA = np.array([4.5, 3.5, 2.5])
    fluxesB = np.array([60.0, 105.0, 135.0])
    objA = dict(kind='gauss', e1=0.10, e2=-0.05, T=0.4)
    objB = dict(kind='gauss', e1=0.05, e2=0.02, T=0.6)
    posA = (0.05, -sep / 2)
    posB = (-0.05, sep / 2)

    # noise for faint-object r-band s/n ~ 15
    iso = make_blend_obs(
        [dict(objA, flux=fluxesA[1], v=posA[0], u=posA[1])], 0.9,
    )
    noise = np.sqrt(np.sum(iso.image ** 2)) / 15.0

    frs = []
    for trial in range(ntrial):
        comps_per_band = []
        for b in range(3):
            comps_per_band.append([
                dict(objA, flux=fluxesA[b], v=posA[0], u=posA[1]),
                dict(objB, flux=fluxesB[b], v=posB[0], u=posB[1]),
            ])
        mbobs = make_blend_mbobs(
            comps_per_band, psf_fwhms, noise=noise, rng=rng,
        )
        res = deblend(
            mbobs,
            [
                dict(v=posA[0], u=posA[1], Tguess=0.4),
                dict(v=posB[0], u=posB[1], Tguess=0.4),
            ],
            fwhm_smooth=FWHM_SMOOTH,
            tol=1.0e-6,
        )
        frs.append(res['objects'][0]['flux'][1] / fluxesA[1])

    frs = np.array(frs)
    assert np.abs(np.median(frs) - 1) < 0.15
    assert frs.std() < 0.3


def test_s2n():
    """
    the deblender flux s/n is consistent with its flux errors and
    matches the pre-psf admom fitter for an isolated object
    """
    from ngmix.prepsfadmom import run_prepsf_admom

    rng = np.random.RandomState(19)
    comp = dict(kind='gauss', e1=0.10, e2=-0.05, T=0.5, flux=10.0,
                v=0.1, u=-0.2)
    iso = make_blend_obs([comp], 0.9)
    noise = np.sqrt(np.sum(iso.image ** 2)) / 25.0
    obs = make_blend_obs([comp], 0.9, noise=noise, rng=rng)

    fres = run_prepsf_admom(
        obs, fwhm_smooth=FWHM_SMOOTH, ap_rad=0,
        rng=np.random.RandomState(2),
    )
    res = deblend(
        obs, [dict(v=comp['v'], u=comp['u'], Tguess=0.4)],
        fwhm_smooth=FWHM_SMOOTH,
    )
    r = res['objects'][0]

    assert np.all(np.isfinite(r['flux_err']))
    assert np.allclose(
        r['s2n'],
        np.sqrt(np.sum((r['flux'] / r['flux_err']) ** 2)),
    )
    assert np.abs(r['s2n'] / fres['s2n'] - 1) < 0.05


def test_flux_err_calibration():
    """
    the flux and structure errors from the sandwich over the moment
    matching conditions match the observed scatter for a deblended
    pair
    """
    ntrial = 100
    rng = np.random.RandomState(137)

    compA = dict(kind='gauss', e1=0.10, e2=-0.05, T=0.5, flux=10.0,
                 v=0.0, u=-1.25)
    compB = dict(kind='gauss', e1=-0.05, e2=0.08, T=0.4, flux=6.0,
                 v=0.0, u=1.25)
    objects = [
        dict(v=compA['v'], u=compA['u'], Tguess=0.4),
        dict(v=compB['v'], u=compB['u'], Tguess=0.4),
    ]

    iso = make_blend_obs([compA], 0.9)
    noise = np.sqrt(np.sum(iso.image ** 2)) / 20.0

    vals = {
        key: [[], []] for key in [
            'flux', 'flux_err', 'T', 'T_err', 'e1', 'e1_err',
        ]
    }
    for trial in range(ntrial):
        obs = make_blend_obs([compA, compB], 0.9, noise=noise, rng=rng)
        res = deblend(
            obs, objects, fwhm_smooth=FWHM_SMOOTH, tol=1.0e-6,
        )
        for i in range(2):
            robj = res['objects'][i]
            vals['flux'][i].append(robj['flux'][0])
            vals['flux_err'][i].append(robj['flux_err'][0])
            for key in ['T', 'T_err', 'e1', 'e1_err']:
                vals[key][i].append(robj[key])

    for i in range(2):
        for key in ['flux', 'T', 'e1']:
            ratio = (
                np.mean(vals[key + '_err'][i]) / np.std(vals[key][i])
            )
            assert np.abs(ratio - 1) < 0.25, (key, i, ratio)


def test_s2n_noise_image():
    """
    with correlated noise and use_noise_image=True, the deblender
    flux errors match the pre-psf admom fitter, and the white-noise
    assumption is seen to overestimate the s/n
    """
    from ngmix.prepsfadmom import run_prepsf_admom
    from scipy.ndimage import gaussian_filter
    import ngmix

    rng = np.random.RandomState(21)
    comp = dict(kind='gauss', e1=0.10, e2=-0.05, T=0.5, flux=10.0,
                v=0.1, u=-0.2)
    clean = make_blend_obs([comp], 0.9)

    # correlated noise: smoothed white, normalized to a pixel sigma
    sigma = np.sqrt(np.sum(clean.image ** 2)) / 25.0

    def corr_noise():
        nim = gaussian_filter(
            rng.normal(size=clean.image.shape), 1.0,
        )
        return nim * sigma / nim.std()

    obs = ngmix.Observation(
        clean.image + corr_noise(),
        weight=np.ones_like(clean.image) / sigma ** 2,
        jacobian=clean.jacobian,
        psf=clean.psf,
        noise=corr_noise(),
    )

    objects = [dict(v=comp['v'], u=comp['u'], Tguess=0.4)]

    fres = run_prepsf_admom(
        obs, fwhm_smooth=FWHM_SMOOTH, ap_rad=0,
        use_noise_image=True, rng=np.random.RandomState(2),
    )
    res = deblend(
        obs, objects, fwhm_smooth=FWHM_SMOOTH, use_noise_image=True,
    )
    r = res['objects'][0]

    assert np.all(np.isfinite(r['flux_err']))
    assert np.abs(r['s2n'] / fres['s2n'] - 1) < 0.05

    # the white assumption misses the low-k power concentration
    res_white = deblend(obs, objects, fwhm_smooth=FWHM_SMOOTH)
    assert res_white['objects'][0]['s2n'] > 1.5 * r['s2n']


def test_auto_smoothing():
    """
    the smoothing can be chosen automatically from the psfs
    """
    comp = dict(kind='gauss', e1=0.1, e2=-0.05, T=0.5, flux=3.5,
                v=0.0, u=0.0)
    obs = make_blend_obs([comp], 0.9)
    res = deblend(
        obs, [dict(v=0.0, u=0.0, Tguess=0.4)],
        rng=np.random.RandomState(5),
    )
    assert res['fwhm_smooth'] > 0.9
    assert np.abs(res['objects'][0]['flux'][0] / comp['flux'] - 1) < 1.0e-3
    assert np.abs(res['objects'][0]['T'] / comp['T'] - 1) < 1.0e-2


def test_failure_containment():
    """
    an unmodeled bright neighbor inflates the measured moments past
    the deweightable range; the containment must restart from the
    delta state and then demote the object to a fixed point source,
    quickly and without failing the group
    """
    from kdeblend import DEBLENDED_AS_PSF, RESTARTED

    for model in ['exp', 'gauss']:
        comps = [
            {'kind': 'gauss', 'flux': 100, 'v': 0, 'u': 0,
             'e1': 0.1, 'e2': 0.0, 'T': 1.5},
            # bright neighbor not present in the object list
            {'kind': 'gauss', 'flux': 10000, 'v': 0, 'u': 2.5,
             'e1': 0.0, 'e2': 0.0, 'T': 1.0},
        ]
        obs = make_blend_obs(comps, 0.9)
        objects = [
            {'v': 0.0, 'u': 0.0, 'type': model, 'Tguess': 1.5},
        ]
        res = deblend(obs, objects, tol=1.0e-6)
        obj = res['objects'][0]
        assert obj['deblend_flags'] & DEBLENDED_AS_PSF
        assert obj['deblend_flags'] & RESTARTED
        assert obj['type'] == 'star'
        assert np.all(np.isfinite(obj['flux']))
        assert obj['e_flags'] != 0
        # contained in ~2 NFAIL_LIMIT rounds, not the group limit
        assert res['numiter'] < 40

    # a clean pair is untouched by the containment
    comps = [
        {'kind': 'gauss', 'flux': 100, 'v': 0, 'u': -1.0,
         'e1': 0.05, 'e2': 0.0, 'T': 0.6},
        {'kind': 'gauss', 'flux': 150, 'v': 0, 'u': 1.0,
         'e1': -0.05, 'e2': 0.0, 'T': 0.8},
    ]
    obs = make_blend_obs(comps, 0.9)
    objects = [
        {'v': 0.0, 'u': -1.0, 'type': 'exp', 'Tguess': 0.6},
        {'v': 0.0, 'u': 1.0, 'type': 'exp', 'Tguess': 0.8},
    ]
    res = deblend(obs, objects, tol=1.0e-6)
    for obj in res['objects']:
        assert obj['deblend_flags'] == 0
        assert obj['type'] == 'exp'
        assert obj['e_flags'] == 0


def test_fixed_models():
    """
    the same unmodeled-bright-neighbor scene that triggers demotion
    in test_failure_containment becomes a clean measurement when the
    neighbor is supplied as a fixed external model, and the result
    matches an isolated fit of the same object
    """
    faint = {
        'kind': 'gauss', 'flux': 100, 'v': 0, 'u': 0,
        'e1': 0.1, 'e2': 0.0, 'T': 1.5,
    }
    bright = {
        'kind': 'gauss', 'flux': 10000, 'v': 0, 'u': 2.5,
        'e1': 0.0, 'e2': 0.0, 'T': 1.0,
    }
    objects = [{'v': 0.0, 'u': 0.0, 'type': 'exp', 'Tguess': 1.5}]

    # isolated reference
    obs = make_blend_obs([faint], 0.9)
    ref = deblend(obs, objects, tol=1.0e-6)['objects'][0]
    assert ref['deblend_flags'] == 0

    # blended, with the neighbor as a fixed external model at truth
    obs = make_blend_obs([faint, bright], 0.9)
    fixed = [{
        'v': 0.0, 'u': 2.5, 'type': 'gauss',
        'e1': 0.0, 'e2': 0.0, 'T': 1.0, 'flux': [10000.0],
    }]
    res = deblend(obs, objects, tol=1.0e-6, fixed_models=fixed)
    obj = res['objects'][0]

    assert obj['deblend_flags'] == 0
    assert obj['type'] == 'exp'
    assert obj['e_flags'] == 0
    # the closed-form subtraction recovers the isolated measurement
    assert abs(obj['T'] / ref['T'] - 1) < 0.01
    assert abs(obj['e1'] - ref['e1']) < 0.01
    assert abs(obj['e2'] - ref['e2']) < 0.01
    assert abs(obj['flux'][0] / ref['flux'][0] - 1) < 0.01

    # nonfinite fixed model parameters must raise
    bad = [{
        'v': 0.0, 'u': 2.5, 'type': 'gauss',
        'e1': 0.0, 'e2': 0.0, 'T': np.nan, 'flux': [10000.0],
    }]
    with pytest.raises(ValueError):
        deblend(obs, objects, tol=1.0e-6, fixed_models=bad)


def test_dev_blends():
    """
    per-object model types including 'dev': a dev + gauss pair
    recovers both objects nearly exactly on exact-mixture data, and
    a mixed dev + exp pair converges cleanly
    """
    # dev + gauss, both rendered exactly
    comps = [
        {'kind': 'dev', 'flux': 300, 'v': 0, 'u': -1.2,
         'e1': 0.1, 'e2': 0.0, 'T': 2.0},
        {'kind': 'gauss', 'flux': 100, 'v': 0, 'u': 1.2,
         'e1': -0.05, 'e2': 0.05, 'T': 0.5},
    ]
    obs = make_blend_obs(comps, 0.9)
    objects = [
        {'v': 0.0, 'u': -1.2, 'type': 'dev', 'Tguess': 2.0},
        {'v': 0.0, 'u': 1.2, 'type': 'gauss', 'Tguess': 0.5},
    ]
    res = deblend(obs, objects, tol=1.0e-8)
    o0, o1 = res['objects']
    for o in (o0, o1):
        assert o['deblend_flags'] == 0
        assert o['e_flags'] == 0
    assert abs(o0['T'] / 2.0 - 1) < 1.0e-2
    assert abs(o0['e1'] - 0.1) < 1.0e-2
    assert abs(o0['flux'][0] / 300 - 1) < 1.0e-2
    assert abs(o1['T'] / 0.5 - 1) < 1.0e-2
    assert abs(o1['flux'][0] / 100 - 1) < 1.0e-2

    # mixed dev + exp (exp data is a true exponential, so its fit
    # has the usual small profile-mismatch offsets; require clean
    # convergence and accurate dev recovery)
    comps = [
        {'kind': 'dev', 'flux': 300, 'v': 0, 'u': -1.2,
         'e1': 0.1, 'e2': 0.0, 'T': 2.0},
        {'kind': 'exp', 'flux': 100, 'v': 0, 'u': 1.2,
         'e1': -0.05, 'e2': 0.05, 'hlr': 0.4},
    ]
    obs = make_blend_obs(comps, 0.9)
    objects = [
        {'v': 0.0, 'u': -1.2, 'type': 'dev', 'Tguess': 2.0},
        {'v': 0.0, 'u': 1.2, 'type': 'exp', 'Tguess': 0.5},
    ]
    res = deblend(obs, objects, tol=1.0e-8)
    o0, o1 = res['objects']
    for o in (o0, o1):
        assert o['deblend_flags'] == 0
        assert o['e_flags'] == 0
    assert abs(o0['T'] / 2.0 - 1) < 2.0e-2
    assert abs(o0['e1'] - 0.1) < 1.0e-2
    assert abs(o0['flux'][0] / 300 - 1) < 2.0e-2
    assert abs(o1['e1'] - (-0.05)) < 1.0e-2


def test_gauss_shapes():
    """
    the gauss-estimator shapes from the converged weight: identical
    to the primary shapes for a gauss model, and lower noise than
    the family shapes for the mixture models
    """
    rng = np.random.RandomState(11)

    # gauss model: the two estimators coincide exactly
    comps = [
        {'kind': 'gauss', 'flux': 100, 'v': 0, 'u': -1.0,
         'e1': 0.05, 'e2': 0.0, 'T': 0.6},
        {'kind': 'gauss', 'flux': 150, 'v': 0, 'u': 1.0,
         'e1': -0.05, 'e2': 0.0, 'T': 0.8},
    ]
    obs = make_blend_obs(comps, 0.9)
    objects = [
        {'v': 0.0, 'u': -1.0, 'type': 'gauss', 'Tguess': 0.6},
        {'v': 0.0, 'u': 1.0, 'type': 'gauss', 'Tguess': 0.8},
    ]
    res = deblend(obs, objects, tol=1.0e-8)
    for o in res['objects']:
        assert np.allclose(o['gauss_e1'], o['e1'])
        assert np.allclose(o['gauss_e2'], o['e2'])
        assert np.allclose(o['gauss_T'], o['T'])
        assert np.allclose(o['gauss_e1_err'], o['e1_err'])
        assert o['gauss_e_flags'] == 0

    # exp model with noise: the gauss estimator is quieter than the
    # family estimator
    comps = [
        {'kind': 'exp', 'flux': 100, 'v': 0, 'u': 0,
         'e1': 0.1, 'e2': 0.0, 'hlr': 0.5},
    ]
    obs = make_blend_obs(comps, 0.9, noise=0.2, rng=rng)
    res = deblend(
        obs, [{'v': 0.0, 'u': 0.0, 'type': 'exp', 'Tguess': 0.5}],
        tol=1.0e-8,
    )
    o = res['objects'][0]
    assert o['gauss_e_flags'] == 0
    assert np.isfinite(o['gauss_e1'])
    assert o['gauss_e1_err'] < o['e1_err']
    assert o['gauss_e2_err'] < o['e2_err']


def test_gauss_s2n():
    """
    gauss-aperture flux and s2n: identical to the primary entries
    for a gauss model, finite and positive for the mixture models
    """
    rng = np.random.RandomState(21)
    comps = [
        {'kind': 'gauss', 'flux': 100, 'v': 0, 'u': 0,
         'e1': 0.05, 'e2': 0.0, 'T': 0.6},
    ]
    obs = make_blend_obs(comps, 0.9, noise=0.2, rng=rng)
    res = deblend(
        obs, [{'v': 0.0, 'u': 0.0, 'type': 'gauss', 'Tguess': 0.6}],
        tol=1.0e-8, use_noise_image=False,
    )
    o = res['objects'][0]
    assert np.allclose(o['gauss_flux'], o['flux'])
    assert np.allclose(o['gauss_flux_err'], o['flux_err'])
    assert np.allclose(o['gauss_s2n'], o['s2n'])

    res = deblend(
        obs, [{'v': 0.0, 'u': 0.0, 'type': 'exp', 'Tguess': 0.6}],
        tol=1.0e-8,
    )
    o = res['objects'][0]
    assert np.all(np.isfinite(o['gauss_flux']))
    assert np.all(o['gauss_flux_err'] > 0)
    assert o['gauss_s2n'] > 0
