"""
gate-1 tests for the ladder model type: single-object fixed
point and subtraction quality, pair accuracy, multiband
per-band amplitudes, state packing, and containment
"""
import numpy as np

from ngmix.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums

from kdeblend import deblend
from kdeblend.deblender import build_deblender
from kdeblend.ladder import (
    solve_group_amps, band_comps, _comps_flux_sum,
)
from kdeblend.flags import RESTARTED, DEBLENDED_AS_PSF

from _sims import make_blend_obs, make_blend_mbobs


def _data_flux_sum(ep, v, u, W):
    """measured flux sum under weight W at (v, u), scaled to be
    comparable to the closed-form model sums at detAtinv=1"""
    alpha, beta = get_phase_angles(
        ep, v - ep['vcen'], u - ep['ucen'],
    )
    sums = np.zeros(6)
    admom_ksums(
        ep['kim'], ep['iy'], ep['ix'], ep['dim'], alpha, beta,
        ep['kv'], ep['ku'], W[0, 0], W[0, 1], W[1, 1], ep['df2'],
        sums,
    )
    return sums[5] * ep['detAtinv']


def _model_flux_sum(model, Tsmooth, band, dv, du, W):
    Fb, S00, S01, S11 = band_comps(model, Tsmooth)
    return _comps_flux_sum(Fb[band], S00, S01, S11, dv, du, W)


def _fit_single(obs, t):
    deb, _ = build_deblender(
        obs, [dict(v=0.0, u=0.0, type=t, Tguess=0.5)],
    )
    res = deb.go()
    return deb, res


def test_ladder_single_fixed_point():
    """isolated sersic: the ladder converges cleanly and the
    converged amps are a fixed point of the scene-wide solve"""
    comp = dict(kind='sersic', n=3.0, hlr=0.8, flux=1.0,
                e1=0.0, e2=0.0, v=0.0, u=0.0)
    obs = make_blend_obs([comp], 0.8, dim=128)
    deb, res = _fit_single(obs, 'ladder')
    assert res['converged']
    o = res['objects'][0]
    assert o['deblend_flags'] == 0
    assert 'amps' in o and o['amps'].shape[0] == 1
    assert np.all(np.isfinite(o['amps']))

    da = solve_group_amps(deb)
    assert da < 1.0e-4


def test_ladder_beats_exp_subtraction():
    """the converged ladder model reproduces the data flux sum
    at a neighbor position far better than the exp model: the
    contamination result, through the deblender"""
    comp = dict(kind='sersic', n=3.0, hlr=0.8, flux=1.0,
                e1=0.0, e2=0.0, v=0.0, u=0.0)
    obs = make_blend_obs([comp], 0.8, dim=128)

    resid = {}
    for t in ('exp', 'ladder'):
        deb, res = _fit_single(obs, t)
        assert res['converged']
        ep = deb.epochs_per_obj[0][0]
        Twf = 0.2 / 2 + deb.Tsmooth / 2
        wf = np.diag([Twf, Twf])
        r = 0.0
        for d in (1.0, 2.0):
            tsum = _data_flux_sum(ep, d, 0.0, wf)
            msum = _model_flux_sum(
                deb.models[0], deb.Tsmooth, 0, -d, 0.0, wf,
            )
            r = max(r, abs(tsum - msum))
        resid[t] = r
    assert resid['ladder'] < resid['exp'] / 5


def test_ladder_pair_fluxes():
    """exp-truth bright neighbor with a faint gauss: the faint
    flux is recovered accurately with a ladder neighbor"""
    comps = [
        dict(kind='exp', hlr=0.6, flux=10.0, e1=0.05, e2=-0.03,
             v=0.0, u=-1.0),
        dict(kind='gauss', T=0.2, flux=0.5, e1=0.0, e2=0.0,
             v=0.0, u=1.0),
    ]
    obs = make_blend_obs(comps, 0.8)
    res = deblend(obs, [
        dict(v=0.0, u=-1.0, type='ladder', Tguess=0.7),
        dict(v=0.0, u=1.0, type='gauss', Tguess=0.2),
    ])
    assert res['converged']
    for o in res['objects']:
        assert o['deblend_flags'] == 0
    assert abs(res['objects'][1]['flux'][0] / 0.5 - 1) < 0.01


def test_ladder_multiband_perband_amps():
    """a color-gradient composite in two bands: the per-band
    amps differ and each band's model reproduces that band's
    aperture flux sums"""
    def comps(fd, fb):
        return [
            dict(kind='exp', hlr=0.8, flux=fd, e1=0.0, e2=0.0,
                 v=0.0, u=0.0),
            dict(kind='dev', T=0.3, flux=fb, e1=0.0, e2=0.0,
                 v=0.0, u=0.0),
        ]
    mbobs = make_blend_mbobs(
        [comps(0.7, 0.3), comps(0.4, 0.6)], [0.8, 0.8], dim=128,
    )
    deb, _ = build_deblender(
        mbobs, [dict(v=0.0, u=0.0, type='ladder', Tguess=0.5)],
    )
    res = deb.go()
    assert res['converged']
    m = deb.models[0]
    assert not np.allclose(m['amps'][0], m['amps'][1], rtol=0.02)

    eps = deb.epochs_per_obj[0]
    for af in (1.0, 4.0):
        W = af * np.asarray(deb.Sw[0])
        for ep in eps:
            band = ep['band']
            tsum = _data_flux_sum(ep, 0.0, 0.0, W)
            msum = _model_flux_sum(
                m, deb.Tsmooth, band, 0.0, 0.0, W,
            )
            assert abs(msum / tsum - 1) < 1.0e-2


def test_ladder_pack_roundtrip():
    """pack -> unpack -> pack is the identity for a mixed group
    containing a ladder object"""
    comps = [
        dict(kind='exp', hlr=0.6, flux=5.0, e1=0.0, e2=0.0,
             v=0.0, u=-1.0),
        dict(kind='gauss', T=0.2, flux=1.0, e1=0.0, e2=0.0,
             v=0.0, u=1.0),
    ]
    obs = make_blend_obs(comps, 0.8)
    deb, _ = build_deblender(obs, [
        dict(v=0.0, u=-1.0, type='ladder', Tguess=0.7),
        dict(v=0.0, u=1.0, type='gauss', Tguess=0.2),
    ])
    for _ in range(6):
        deb.isweep += 1
        deb._sweep()
    x = deb._pack_state()
    deb._unpack_state(x)
    x2 = deb._pack_state()
    assert np.allclose(x, x2, rtol=0, atol=1.0e-12)
    assert deb._state_valid()


def test_ladder_containment():
    """forced containment: restart resets the amps to the exp
    profile at the compact state; a second escalation demotes to
    star and drops the ladder state"""
    comp = dict(kind='sersic', n=2.0, hlr=0.5, flux=1.0,
                e1=0.0, e2=0.0, v=0.0, u=0.0)
    obs = make_blend_obs([comp], 0.8)
    deb, res = _fit_single(obs, 'ladder')
    assert res['converged']

    deb._contain_failure(0, force=True)
    m = deb.models[0]
    assert deb.dbflags[0] & RESTARTED
    assert m['type'] == 'ladder'
    assert np.all(np.isfinite(m['amps']))
    assert deb._state_valid()

    deb._contain_failure(0, force=True)
    assert deb.dbflags[0] & DEBLENDED_AS_PSF
    assert m['type'] == 'star'
    assert 'amps' not in m and 'rungs' not in m
    assert deb._state_valid()
    r = deb._get_object_result(0)
    assert r['type'] == 'star'


def test_ladder_pair_joint():
    """two ladder objects: the joint solve's cross-object blocks
    apportion an exp-truth pair accurately"""
    comps = [
        dict(kind='exp', hlr=0.6, flux=4.0, e1=0.0, e2=0.0,
             v=0.0, u=-1.0),
        dict(kind='exp', hlr=0.4, flux=1.0, e1=0.0, e2=0.0,
             v=0.0, u=1.0),
    ]
    obs = make_blend_obs(comps, 0.8)
    res = deblend(obs, [
        dict(v=0.0, u=-1.0, type='ladder', Tguess=0.7),
        dict(v=0.0, u=1.0, type='ladder', Tguess=0.4),
    ])
    assert res['converged']
    for o in res['objects']:
        assert o['deblend_flags'] == 0
        assert np.all(np.isfinite(o['amps']))
    # gauss-aperture capture of an exp profile is below total by
    # the wing miss; the deblended fluxes should match the
    # values measured for each object fit alone
    for i, c in enumerate(comps):
        iso = make_blend_obs([c], 0.8)
        riso = deblend(iso, [dict(
            v=c['v'], u=c['u'], type='ladder',
            Tguess=2 * c['hlr'] ** 2,
        )])
        assert riso['converged']
        assert np.allclose(
            res['objects'][i]['flux'][0],
            riso['objects'][0]['flux'][0],
            rtol=5.0e-3,
        )


def test_ladder_consistency_rows():
    """with the consistency row the converged ladder model
    reproduces the weighted size the deweight step consumed:
    deweight(model sums under Sw) matches Sw in T, for round and
    elliptical truths, much closer than without the row.  (The
    ellipticity is not made consistent: the rungs share the
    frame's deweighted ellipticity, see kdeblend.ladder)"""
    from ngmix.prepsfadmom import deweight
    from kdeblend.deblender import _moment_matrix, _shape_from_cov
    from kdeblend.ladder import _comps_sums
    import kdeblend.ladder as L

    def mismatch_T(obs, flag):
        saved = L.LADDER_MOMENT_ROWS
        L.LADDER_MOMENT_ROWS = flag
        try:
            deb, res = _fit_single(obs, 'ladder')
        finally:
            L.LADDER_MOMENT_ROWS = saved
        assert res['converged']
        m = deb.models[0]
        Sw = np.asarray(deb.Sw[0])
        S00, S01, S11 = m['rungs']
        msums = _comps_sums(
            m['amps'][0], S00, S01, S11, 0.0, 0.0, Sw,
        )
        newSw, flags = deweight(_moment_matrix(msums), Sw)
        assert flags == 0
        Tm = _shape_from_cov(newSw - deb.smooth_cov)[0]
        Tw = _shape_from_cov(Sw - deb.smooth_cov)[0]
        return abs(Tm / Tw - 1)

    for e1, e2 in ((0.1, -0.05), (0.0, 0.0)):
        comp = dict(kind='sersic', n=3.0, hlr=0.8, flux=1.0,
                    e1=e1, e2=e2, v=0.0, u=0.0)
        obs = make_blend_obs([comp], 0.8, dim=128)
        on = mismatch_T(obs, True)
        off = mismatch_T(obs, False)
        assert on < 1.0e-4
        assert on < off / 2
