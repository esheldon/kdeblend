"""
the multiplicative profile-prior width: the solve and the analytic
error chain agree with the finite-difference referee, the prior
center is unchanged, a faint object reverts to the exp projection
within the per-rung widths, and a bright object is still free to
leave it
"""
import numpy as np
import pytest

from kdeblend import ladder
from kdeblend.deblender import build_deblender
from kdeblend.full_errors import full_covariance
from kdeblend.ladder import (
    ladder_prior_lambda, ladder_context, LADDER_RUNGS,
)

from test_full_errors import make_mbobs

OFFSETS = [(-0.625, 0.0), (0.625, 0.0)]


def _objects(types, offsets=OFFSETS):
    return [
        {'v': dv, 'u': du, 'type': t, 'Tguess': 0.3}
        for (du, dv), t in zip(offsets, types)
    ]


@pytest.fixture
def multiplicative(monkeypatch):
    monkeypatch.setattr(ladder, 'LADDER_PRIOR_MODE', 'multiplicative')
    monkeypatch.setattr(ladder, 'LADDER_PRIOR_FLOOR', 0.05)


def test_prior_lambda_modes(monkeypatch):
    a0 = np.array([0.2, 0.25, 0.1, -0.02, 0.0])
    monkeypatch.setattr(ladder, 'LADDER_PRIOR_MODE', 'uniform')
    lam, dlam = ladder_prior_lambda(a0, 0.5)
    assert np.allclose(lam, 4.0)
    assert np.all(dlam == 0)

    monkeypatch.setattr(ladder, 'LADDER_PRIOR_MODE', 'multiplicative')
    monkeypatch.setattr(ladder, 'LADDER_PRIOR_FLOOR', 0.05)
    lam, dlam = ladder_prior_lambda(a0, 0.5)
    # width tau sqrt(a0^2 + floor^2): the near-zero rungs get the floor
    assert np.allclose(1 / np.sqrt(lam), 0.5 * np.sqrt(a0 ** 2 + 0.05 ** 2))
    # the derivative by finite difference
    h = 1.0e-6
    for k in range(a0.size):
        ap = a0.copy()
        ap[k] += h
        am = a0.copy()
        am[k] -= h
        fd = (ladder_prior_lambda(ap, 0.5)[0][k]
              - ladder_prior_lambda(am, 0.5)[0][k]) / (2 * h)
        assert np.isclose(dlam[k], fd, rtol=1.0e-5, atol=1.0e-8)

    monkeypatch.setattr(ladder, 'LADDER_PRIOR_MODE', 'bogus')
    with pytest.raises(ValueError):
        ladder_prior_lambda(a0, 0.5)


@pytest.mark.parametrize('types', [
    ('ladder', 'gauss'), ('ladder', 'ladder'),
])
def test_mult_errors_chain_vs_fd(types, multiplicative):
    """the chain assembly, with the moving prior precisions, agrees
    with the finite-difference referee in the multiplicative mode"""
    rng = np.random.RandomState(31)
    mbobs = make_mbobs(rng, OFFSETS)
    deb, _ = build_deblender(
        mbobs, _objects(types), tol=1.0e-6, maxiter=2000,
        full_errors=True, rng=np.random.RandomState(1),
    )
    res = deb.go()
    assert res['converged']

    cov_c, _, ex_c = full_covariance(deb, mbobs, use_chain=True)
    cov_f, _, ex_f = full_covariance(deb, mbobs, use_chain=False)
    dd = np.sqrt(np.diag(cov_c) / np.diag(cov_f))
    assert np.all(np.abs(dd - 1) < 1.0e-2), dd
    sc = np.sqrt(np.outer(np.diag(cov_f), np.diag(cov_f)))
    assert np.all(np.abs(cov_c - cov_f) < 3.0e-2 * sc)
    for gc, gf in zip(ex_c['gauss_flux_cov'], ex_f['gauss_flux_cov']):
        assert np.allclose(
            np.sqrt(np.diag(gc)), np.sqrt(np.diag(gf)), rtol=1.0e-2,
        )
    for i in ex_c['ladder']:
        for key in ('total_flux_cov', 'fixed_flux_cov'):
            assert np.allclose(
                np.sqrt(np.diag(ex_c['ladder'][i][key])),
                np.sqrt(np.diag(ex_f['ladder'][i][key])),
                rtol=2.0e-2,
            )


def _single(flux_scale, seed, mode, floor=0.05):
    """one isolated exponential at a flux scale; returns the amps in
    fraction units and the prior center"""
    import test_full_errors as tfe

    fluxes0 = list(tfe.FLUXES)
    tfe.FLUXES[:] = [f * flux_scale for f in fluxes0]
    try:
        rng = np.random.RandomState(seed)
        mbobs = make_mbobs(rng, [(0.0, 0.0)])
    finally:
        tfe.FLUXES[:] = fluxes0
    ladder.LADDER_PRIOR_MODE = mode
    ladder.LADDER_PRIOR_FLOOR = floor
    deb, _ = build_deblender(
        mbobs, _objects(('ladder',), [(0.0, 0.0)]), tol=1.0e-6,
        maxiter=2000, rng=np.random.RandomState(1),
    )
    res = deb.go()
    assert res['converged']
    assert res['objects'][0]['type'] == 'ladder'
    idx, aps, Sws, Tws, Fhat = ladder_context(deb)
    a0 = ladder.ladder_prior(deb, idx, Sws)
    x = res['objects'][0]['amps'] / Fhat[0][:, None]
    return x, a0, res


def test_faint_reverts_to_the_profile(monkeypatch):
    """at low s/n the multiplicative prior holds every rung near the
    exp projection within its width, while the uniform prior lets the
    outer rungs wander"""
    monkeypatch.setattr(ladder, 'LADDER_PRIOR_MODE', 'uniform')
    monkeypatch.setattr(ladder, 'LADDER_PRIOR_FLOOR', 0.05)
    x_u, a0, _ = _single(0.3, 7, 'uniform')
    x_m, a0m, _ = _single(0.3, 7, 'multiplicative')
    assert np.allclose(a0, a0m)
    lam, _ = ladder_prior_lambda(a0, ladder.LADDER_TAU0)
    width = 1 / np.sqrt(lam)
    dev_m = np.abs(x_m - a0[None, :]) / width[None, :]
    outer = LADDER_RUNGS >= 6.4
    # multiplicative: the outer rungs sit within a couple of widths
    # of the profile
    assert np.all(dev_m[:, outer] < 3.0), dev_m
    # and closer to the profile than the uniform solve leaves them
    assert np.median(np.abs(x_m[:, outer] - a0[outer])) \
        < np.median(np.abs(x_u[:, outer] - a0[outer]))


def test_bright_keeps_its_freedom(monkeypatch):
    """at high s/n the multiplicative prior leaves the data in charge:
    the total flux of a bright exponential matches the uniform-prior
    result and the truth"""
    import test_full_errors as tfe

    monkeypatch.setattr(ladder, 'LADDER_PRIOR_MODE', 'uniform')
    monkeypatch.setattr(ladder, 'LADDER_PRIOR_FLOOR', 0.05)
    _, _, res_u = _single(20.0, 3, 'uniform')
    _, _, res_m = _single(20.0, 3, 'multiplicative')
    truth = 20.0 * np.array(tfe.FLUXES)
    tu = res_u['objects'][0]['total_flux'] / truth
    tm = res_m['objects'][0]['total_flux'] / truth
    assert np.all(np.abs(tm - 1) < 0.05), tm
    assert np.all(np.abs(tm - tu) < 0.03), (tm, tu)
