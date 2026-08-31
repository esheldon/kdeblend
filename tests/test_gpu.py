"""
GPU fitter parity vs deb.go().  Skipped when cupy (or a device)
is unavailable.

The scenes are stable lanes (well-separated blends at moderate
noise, production tol=1e-5), where the fp64 kernel is expected
exact-discrete against the CPU fitter: identical iteration
counts, flags and classifications, float state at the 1e-13
level (asserted at 1e-8).  fp32 is ensemble-qualified only, so
it is checked for class-level agreement and loose floats.
"""
import numpy as np
import pytest

cp = pytest.importorskip('cupy')
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:
    pytest.skip('no CUDA device', allow_module_level=True)

import sys  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _sims import make_blend_obs  # noqa: E402

import kdeblend  # noqa: E402

FWHM_SMOOTH = 0.96
FIT_KW = dict(
    fwhm_smooth=FWHM_SMOOTH, maxiter=500, tol=1.0e-5,
    recenter=True, cen_sigma0=0.1,
)

DISCRETE = ('type', 'deblend_flags', 'e_flags', 'gauss_e_flags')


def _scenes():
    """(obs, objects) stable-lane battery: a single, two exp
    pairs, a triple with a star"""
    rng = np.random.RandomState(88)
    out = []

    comp = dict(kind='gauss', e1=0.1, e2=-0.05, T=0.4, flux=6.0,
                v=0.1, u=-0.2)
    obs = make_blend_obs([comp], 0.9, noise=0.005, rng=rng)
    out.append((obs, [dict(v=0.1, u=-0.2, Tguess=0.4,
                           type='gauss')]))

    for sep in (1.5, 2.0):
        a = dict(kind='exp', e1=0.1, e2=0.05, hlr=0.4, flux=6.0,
                 v=0.0, u=-sep / 2)
        b = dict(kind='exp', e1=-0.08, e2=0.12, hlr=0.3, flux=4.0,
                 v=0.1, u=sep / 2)
        obs = make_blend_obs([a, b], 0.9, noise=0.005, rng=rng)
        out.append((obs, [
            dict(v=0.0, u=-sep / 2, Tguess=0.4, type='exp'),
            dict(v=0.1, u=sep / 2, Tguess=0.3, type='exp'),
        ]))

    a = dict(kind='exp', e1=0.05, e2=0.0, hlr=0.35, flux=5.0,
             v=-1.0, u=-1.0)
    b = dict(kind='star', flux=3.0, v=1.2, u=0.0)
    c = dict(kind='exp', e1=0.0, e2=-0.1, hlr=0.3, flux=4.0,
             v=-0.2, u=1.4)
    obs = make_blend_obs([a, b, c], 0.9, noise=0.005, rng=rng)
    out.append((obs, [
        dict(v=-1.0, u=-1.0, Tguess=0.35, type='exp'),
        dict(v=1.2, u=0.0, Tguess=0.1, type='star'),
        dict(v=-0.2, u=1.4, Tguess=0.3, type='exp'),
    ]))
    return out


def _build_pair():
    """(cpu deblenders, gpu deblenders, identical construction)"""
    dc, dg = [], []
    for obs, objects in _scenes():
        for lst in (dc, dg):
            deb, _ = kdeblend.build_deblender(
                obs, [dict(o) for o in objects], **FIT_KW,
            )
            lst.append(deb)
    return dc, dg


def _compare(res_cpu, res_gpu, exact, ftol):
    if exact:
        assert res_gpu['numiter'] == res_cpu['numiter']
        assert res_gpu['nskip'] == res_cpu['nskip']
    assert res_gpu['converged'] == res_cpu['converged']
    for oc, og in zip(res_cpu['objects'], res_gpu['objects']):
        for k in DISCRETE:
            assert og.get(k) == oc.get(k), k
        for k, vc in oc.items():
            if k in DISCRETE or isinstance(vc, str) or vc is None:
                continue
            vg = og[k]
            vc = np.asarray(vc, dtype=float)
            vg = np.asarray(vg, dtype=float)
            # relative at ftol with a small absolute floor for
            # near-zero quantities (cen_pull on centered objects)
            assert np.allclose(
                vg, vc, rtol=ftol, atol=0.01 * ftol,
                equal_nan=True,
            ), (k, np.abs(vg - vc).max())


def test_fit_groups_fp64_exact():
    from kdeblend.gpu import fit_groups

    dc, dg = _build_pair()
    res_cpu = [d.go() for d in dc]
    res_gpu = fit_groups(dg, fp32=False)
    for rc, rg in zip(res_cpu, res_gpu):
        _compare(rc, rg, exact=True, ftol=1.0e-8)


def test_fit_groups_fp32_class():
    from kdeblend.gpu import fit_groups

    dc, dg = _build_pair()
    res_cpu = [d.go() for d in dc]
    res_gpu = fit_groups(dg, fp32=True)
    for rc, rg in zip(res_cpu, res_gpu):
        _compare(rc, rg, exact=False, ftol=5.0e-2)


def test_fp32_tol_guard():
    from kdeblend.gpu import fit_groups

    obs, objects = _scenes()[0]
    deb, _ = kdeblend.build_deblender(
        obs, objects, fwhm_smooth=FWHM_SMOOTH, tol=1.0e-8,
    )
    with pytest.raises(ValueError):
        fit_groups([deb], fp32=True)
