"""
ladder-aware full (fixed-point) errors: chain vs finite
difference referee, application to a mixed group, and empirical
calibration
"""
import numpy as np
import pytest

from kdeblend.deblender import deblend, build_deblender
from kdeblend.full_errors import full_covariance

from test_full_errors import make_mbobs, NBAND

OFFSETS = [(-0.625, 0.0), (0.625, 0.0)]


def _objects(types, offsets=OFFSETS):
    return [
        {'v': dv, 'u': du, 'type': t, 'Tguess': 0.3}
        for (du, dv), t in zip(offsets, types)
    ]


@pytest.mark.parametrize('types', [
    ('ladder', 'gauss'), ('ladder', 'ladder'),
])
def test_ladder_errors_chain_vs_fd(types):
    """the chain assembly agrees with the full finite-difference
    referee for groups with ladder members"""
    rng = np.random.RandomState(31)
    mbobs = make_mbobs(rng, OFFSETS)
    deb, _ = build_deblender(
        mbobs, _objects(types), tol=1.0e-6, maxiter=2000,
        full_errors=True,
    rng=np.random.RandomState(1),
)
    res = deb.go()
    assert res['converged']

    cov_c, _, ex_c = full_covariance(deb, mbobs, use_chain=True)
    cov_f, _, ex_f = full_covariance(deb, mbobs, use_chain=False)
    dd = np.sqrt(np.diag(cov_c) / np.diag(cov_f))
    assert np.all(np.abs(dd - 1) < 1.0e-2), dd
    # off-diagonal entries relative to their natural scale
    # sqrt(C_ii C_jj): a pure relative test fails on near-zero
    # entries at rounding level
    sc = np.sqrt(np.outer(np.diag(cov_f), np.diag(cov_f)))
    assert np.all(np.abs(cov_c - cov_f) < 3.0e-2 * sc)
    for gc, gf in zip(ex_c['gauss_flux_cov'], ex_f['gauss_flux_cov']):
        assert np.allclose(
            np.sqrt(np.diag(gc)), np.sqrt(np.diag(gf)), rtol=1.0e-2,
        )


def test_ladder_errors_applied():
    """deblend(full_errors=True) applies the full errors to a
    ladder+gauss pair: finite, larger than the per-object
    values for the blended fluxes, and with the covariance
    entries"""
    rng = np.random.RandomState(7)
    mbobs = make_mbobs(rng, OFFSETS)
    objs = _objects(('ladder', 'gauss'))
    res0 = deblend(mbobs, objs, tol=1.0e-6, maxiter=2000, rng=np.random.RandomState(1))
    res = deblend(
        mbobs, objs, tol=1.0e-6, maxiter=2000, full_errors=True,
    rng=np.random.RandomState(1),
)
    assert res['converged'] and res['full_errors'] is True
    for r0, r in zip(res0['objects'], res['objects']):
        assert np.all(np.isfinite(r['flux_err']))
        assert r['flux_cov'].shape == (NBAND, NBAND)
        assert np.allclose(
            np.sqrt(np.diag(r['flux_cov'])), r['flux_err'],
            rtol=1.0e-6,
        )
        assert np.isfinite(r['s2n'])
        assert np.isfinite(r['T_err'])
        assert r['e_flags'] == 0
        assert np.isfinite(r['e1_err']) and np.isfinite(r['e2_err'])
        # the blend inflates the flux errors over the
        # deterministic-neighbor per-object values
        assert np.all(r['flux_err'] > 0.9 * r0['flux_err'])
    r = res['objects'][0]
    assert 'amps' in r
    for key in ('total_flux_err', 'fixed_flux_err'):
        assert r[key].shape == (NBAND,) and np.all(np.isfinite(r[key]))
        assert np.all(r[key] > 0)
    assert r['gradient_err'].shape == (NBAND - 1,)
    assert np.all(np.isfinite(r['gradient_err']))


@pytest.mark.parametrize('types', [
    ('ladder',), ('ladder', 'gauss'),
])
def test_ladder_errors_calibration(types):
    """the reported full errors match the empirical scatter over
    noise realizations for a ladder single and a ladder+gauss
    pair (fluxes, T and e1), at the precision of the ensemble"""
    nreal = 50
    offsets = OFFSETS[:len(types)]
    if len(types) == 1:
        offsets = [(0.0, 0.0)]
    objs = _objects(types, offsets)
    rng = np.random.RandomState(101)
    keys = ('flux', 'T', 'e1', 'total_flux', 'fixed_flux', 'gradient')
    vals = {k: [] for k in keys}
    errs = {k: [] for k in keys}
    nconv = 0
    for ir in range(nreal):
        mbobs = make_mbobs(rng, offsets)
        res = deblend(
            mbobs, objs, tol=1.0e-6, maxiter=2000, full_errors=True,
    rng=np.random.RandomState(1),
)
        if not (res['converged'] and res['full_errors'] is True):
            continue
        nconv += 1
        o = res['objects'][0]
        vals['flux'].append(o['flux'].copy())
        errs['flux'].append(o['flux_err'].copy())
        vals['T'].append(o['T'])
        errs['T'].append(o['T_err'])
        vals['e1'].append(o['e1'])
        errs['e1'].append(o['e1_err'])
        for k in ('total_flux', 'fixed_flux', 'gradient'):
            vals[k].append(np.atleast_1d(o[k]).copy())
            errs[k].append(np.atleast_1d(o[k + '_err']).copy())
    assert nconv >= 0.9 * nreal
    for k in keys:
        emp = np.std(np.array(vals[k]), axis=0)
        rep = np.mean(np.array(errs[k]), axis=0)
        ratio = rep / emp
        assert np.all(ratio > 0.75) and np.all(ratio < 1.35), (k, ratio)


def test_flux_kernel_and_dtheta_matches_ngmix():
    """the flux-only kernel row and derivative equal the flux
    rows of the full ngmix moment_kernels and dsums_dtheta"""
    from ngmix.prepsfadmom.full_errors import (
        moment_kernels, dsums_dtheta,
    )
    from kdeblend.full_errors import _flux_kernel_and_dtheta

    rng = np.random.RandomState(5)
    mbobs = make_mbobs(rng, OFFSETS)
    deb, _ = build_deblender(
        mbobs, _objects(('ladder', 'gauss')), tol=1.0e-6,
        maxiter=2000, full_errors=True,
    rng=np.random.RandomState(1),
)
    deb.go()
    for ep in deb.epochs_per_obj[0]:
        W = 2.7 * np.asarray(deb.Sw[0]) + np.array([[0.0, 0.03],
                                                     [0.03, 0.0]])
        v0 = deb.positions[0][0] - ep['vcen']
        u0 = deb.positions[0][1] - ep['ucen']
        G, D = _flux_kernel_and_dtheta(ep, W, v0, u0)
        Gref = moment_kernels(ep, W, v0, u0)[5]
        Dref = dsums_dtheta(ep, W, v0, u0)[5]
        assert np.allclose(G, Gref, rtol=1.0e-12, atol=0)
        assert np.allclose(D, Dref, rtol=1.0e-10, atol=0)


def test_flux_kernels_dyadic_match_per_aperture():
    """the fused dyadic aperture kernels and derivatives equal the
    per-aperture routine (hence the ngmix flux rows) for all
    eight apertures"""
    from kdeblend.full_errors import (
        _flux_kernel_and_dtheta, _flux_kernels_and_dtheta_dyadic,
    )
    from kdeblend.ladder import LADDER_AP_FACS

    rng = np.random.RandomState(5)
    mbobs = make_mbobs(rng, OFFSETS)
    deb, _ = build_deblender(
        mbobs, _objects(('ladder', 'gauss')), tol=1.0e-6,
        maxiter=2000, full_errors=True,
    rng=np.random.RandomState(1),
)
    deb.go()
    Sw = np.asarray(deb.Sw[0]) + np.array([[0.0, 0.03], [0.03, 0.0]])
    for ep in deb.epochs_per_obj[0]:
        v0 = deb.positions[0][0] - ep['vcen']
        u0 = deb.positions[0][1] - ep['ucen']
        G, D = _flux_kernels_and_dtheta_dyadic(ep, Sw, v0, u0)
        for j, af in enumerate(LADDER_AP_FACS):
            Gj, Dj = _flux_kernel_and_dtheta(ep, af * Sw, v0, u0)
            assert np.allclose(G[j], Gj, rtol=1.0e-10, atol=1e-300)
            assert np.allclose(D[j], Dj, rtol=1.0e-8, atol=0)


@pytest.mark.parametrize('types', [
    ('ladder', 'gauss'), ('ladder', 'ladder'),
])
def test_ladder_state_response_matches_fd(types):
    """the analytic state response (solve chain with micro-FD
    leaves) reproduces the finite-difference state loop: the
    neighbor-sum derivatives and the derived-functional
    responses"""
    from kdeblend import full_errors as fe
    from kdeblend.ladder import ladder_fixed_weight

    rng = np.random.RandomState(31)
    mbobs = make_mbobs(rng, OFFSETS)
    deb, _ = build_deblender(
        mbobs, _objects(types), tol=1.0e-6, maxiter=2000,
        full_errors=True,
    rng=np.random.RandomState(1),
)
    res = deb.go()
    assert res['converged']
    epochs = deb.epochs_per_obj[0]
    caches = [[fe._data_esums(deb, i, ep) for ep in epochs]
              for i in range(deb.nobj)]
    from ngmix.prepsfadmom.full_errors import dsums_dtheta
    Ds = [[dsums_dtheta(ep, deb.Sw[i], deb.positions[i][0] - ep['vcen'],
                        deb.positions[i][1] - ep['ucen'])
           for ep in epochs] for i in range(deb.nobj)]
    theta0s = [fe._theta_of(deb, i) for i in range(deb.nobj)]
    L = fe._ladder_setup(deb, epochs)
    fe._ladder_resolve(deb, L, caches, Ds, theta0s, {}, {})
    W2, s_star = ladder_fixed_weight(deb.Tsmooth)
    snap = fe._save_state(deb)
    x0 = deb._pack_state()
    cols = fe._column_map(deb)

    dNS_fd, ls_fd = fe._ladder_state_derivs(
        deb, snap, x0, L, caches, Ds, theta0s, cols, W2, s_star,
    )
    dNS_direct, _ = fe._model_sum_derivs(deb, cols)
    dNS_an, ls_an = fe._ladder_state_response(
        deb, snap, x0, L, caches, Ds, theta0s, cols, W2, s_star,
        dNS_direct,
    )
    # neighbor-sum derivatives: compare on the union of keys, in
    # units of the largest entry per column
    keys = set(dNS_fd) | set(dNS_an)
    for key in keys:
        a = dNS_an.get(key, np.zeros((deb.nband, 6)))
        f = dNS_fd.get(key, np.zeros((deb.nband, 6)))
        scale = max(np.abs(f).max(), np.abs(a).max(), 1e-30)
        assert np.abs(a - f).max() < 1.0e-3 * scale, (key, a, f)
    for name in ('G_fixed', 'G_total'):
        a, f = ls_an[name], ls_fd[name]
        scale = np.abs(f).max(axis=2, keepdims=True) + 1e-30
        assert np.all(np.abs(a - f) < 1.0e-3 * scale), name
    for name in ('f0_fixed', 'f0_total'):
        assert np.allclose(ls_an[name], ls_fd[name], rtol=1e-10)


def test_cov_sums_subsampled_matches_full_grid():
    """the influence kernels built on the coarser exact grids
    (every s-th mode) give the same Cov(S) as
    the full padded grid"""
    import kdeblend.full_errors as FE

    rng = np.random.RandomState(7)
    mbobs = make_mbobs(rng, OFFSETS)
    deb, _ = build_deblender(
        mbobs, _objects(('ladder', 'gauss')), tol=1.0e-6,
        maxiter=2000, full_errors=True,
    rng=np.random.RandomState(1),
)
    assert deb.go()['converged']
    obs_flat = [o for ol in mbobs for o in ol]
    epochs = deb.epochs_per_obj[0]
    # the test stamps must actually allow a coarser grid
    assert FE._subsample_factor(
        epochs[0]['dim'], obs_flat[0].image.shape, 20.0,
    ) >= 2
    L = FE._ladder_setup(deb, epochs)
    c_sub = FE._cov_sums(deb, obs_flat, epochs, L)
    c_full = FE._cov_sums(deb, obs_flat, epochs, L, subsample=False)
    sc = np.sqrt(np.outer(np.diag(c_full), np.diag(c_full)))
    assert np.all(np.abs(c_sub - c_full) < 1.0e-5 * sc)
    assert np.allclose(np.diag(c_sub), np.diag(c_full), rtol=1.0e-6)


@pytest.mark.parametrize('types', [
    ('ladder', 'gauss'), ('ladder', 'ladder', 'gauss'),
])
def test_model_sum_derivs_pairwise_matches_full(types):
    """the pairwise model-sum derivatives (FD on the perturbed
    object's own pair term) equal the full-set referee"""
    import kdeblend.full_errors as FE

    offsets = OFFSETS + [(0.0, 0.7)]
    offsets = offsets[:len(types)]
    rng = np.random.RandomState(11)
    mbobs = make_mbobs(rng, offsets)
    deb, _ = build_deblender(
        mbobs, _objects(types, offsets), tol=1.0e-6, maxiter=2000,
        full_errors=True,
    rng=np.random.RandomState(1),
)
    assert deb.go()['converged']
    cols = FE._column_map(deb)
    dA, pA = FE._model_sum_derivs(deb, cols)
    dB, pB = FE._model_sum_derivs_full(deb, cols)
    assert set(dA) == set(dB)
    assert set(pA) == set(pB)
    scale = max(np.abs(v).max() for v in dB.values())
    for key in dB:
        assert np.allclose(dA[key], dB[key], rtol=1.0e-7, atol=1.0e-7 * scale), key
    for key in pB:
        assert np.allclose(pA[key], pB[key], rtol=1.0e-9, atol=0)
