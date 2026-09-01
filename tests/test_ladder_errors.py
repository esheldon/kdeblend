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
    )
    res = deb.go()
    assert res['converged']

    cov_c, _, ex_c = full_covariance(deb, mbobs, use_chain=True)
    cov_f, _, ex_f = full_covariance(deb, mbobs, use_chain=False)
    dd = np.sqrt(np.diag(cov_c) / np.diag(cov_f))
    assert np.all(np.abs(dd - 1) < 1.0e-2), dd
    assert np.allclose(cov_c, cov_f, rtol=3.0e-2, atol=0)
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
    res0 = deblend(mbobs, objs, tol=1.0e-6, maxiter=2000)
    res = deblend(
        mbobs, objs, tol=1.0e-6, maxiter=2000, full_errors=True,
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
