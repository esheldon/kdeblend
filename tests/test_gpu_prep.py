"""
Device-prep parity vs the CPU production path: prep_groups vs
ngmix prep_epoch (same stamps, same retained modes), the
init_sums kernel vs admom_ksums, and a full stub-construction
fit vs the host-pack fit.  Skipped without cupy/device.
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
# use_noise_image=True is the production deblending shape and the
# only path the device prep implements.  maxiter is generous:
# the fit-level gate compares CONVERGED states (localized to the
# tolerance envelope); a group at the cap has trajectory-time
# state that is not pinned, and which group sits at the cap
# varies with hardware (fp64 kernel details differ per arch)
FIT_KW = dict(
    fwhm_smooth=FWHM_SMOOTH, maxiter=2000, tol=1.0e-5,
    recenter=True, cen_sigma0=0.1, use_noise_image=True,
)


def _obs_battery():
    rng = np.random.RandomState(17)
    out = []
    for sep, noise in ((1.5, 0.005), (0.9, 0.02)):
        a = dict(kind='exp', e1=0.1, e2=0.05, hlr=0.4, flux=6.0,
                 v=0.0, u=-sep / 2)
        b = dict(kind='exp', e1=-0.08, e2=0.12, hlr=0.3,
                 flux=4.0, v=0.1, u=sep / 2)
        obs = make_blend_obs([a, b], 0.9, noise=noise, rng=rng)
        obs.noise = rng.normal(scale=noise, size=obs.image.shape)
        out.append((obs, [
            dict(v=0.0, u=-sep / 2, Tguess=0.4, type='exp'),
            dict(v=0.1, u=sep / 2, Tguess=0.3, type='exp'),
        ]))
    return out


def _make_req(obs, deb, aux, ep_s):
    jac = obs.jacobian
    dv = [p[0] for p in deb.positions]
    du = [p[1] for p in deb.positions]
    sw = np.array([[s[0, 0], s[0, 1], s[1, 1]] for s in deb.Sw])
    return dict(
        image=np.asarray(obs.image, dtype=np.float64),
        noise=np.asarray(obs.noise, dtype=np.float64),
        psf=np.asarray(obs.psf.image, dtype=np.float64),
        target_dim=aux['target_dim'],
        eff_pad_factor=aux['eff_pad_factor'],
        jac4=(jac.dvdrow, jac.dvdcol, jac.dudrow, jac.dudcol),
        Tsmooth=aux['Tsmooth'],
        drow=ep_s['drow'], dcol=ep_s['dcol'], df2=ep_s['df2'],
        dv=dv, du=du, sw=sw,
    )


def test_prep_parity():
    """device kim/err_fac2 match prep_epoch at fp64 fft level;
    the retained-mode geometry is bitwise"""
    from ngmix.prepsfadmom.prep import (
        prep_epoch, prep_epoch_scalars,
    )
    from kdeblend.gpu.prep import GeometryCache, prep_groups

    cache = GeometryCache()
    for obs, objects in _obs_battery():
        deb, _ = kdeblend.build_deblender(
            obs, [dict(o) for o in objects], **FIT_KW,
        )
        ep = prep_epoch(
            obs, band=0, fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
            use_noise_image=True,
        )
        ep_s, aux = prep_epoch_scalars(
            obs, band=0, fwhm_smooth=FWHM_SMOOTH,
        )
        slab = prep_groups(
            [_make_req(obs, deb, aux, ep_s)], cache,
        )
        m = slab.group_modes(0)
        assert slab.nm(0) == ep['kim'].size
        assert np.array_equal(cp.asnumpy(m['iy']), ep['iy'])
        assert np.array_equal(cp.asnumpy(m['ix']), ep['ix'])
        assert np.array_equal(cp.asnumpy(m['kv']), ep['kv'])
        assert np.array_equal(cp.asnumpy(m['ku']), ep['ku'])
        kim_d = cp.asnumpy(m['kim'])
        scale = np.abs(ep['kim']).max()
        assert np.abs(kim_d - ep['kim']).max() < 1e-11 * scale
        ef2_d = cp.asnumpy(m['ef2'])
        e2scale = np.abs(ep['err_fac2']).max()
        assert np.abs(ef2_d - ep['err_fac2']).max() \
            < 1e-11 * e2scale


def test_init_sums_parity():
    """the init_sums kernel matches admom_ksums at the guess
    weights to reduction-order level"""
    from ngmix.prepsfadmom.prep import (
        prep_epoch, prep_epoch_scalars,
    )
    from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums
    from ngmix.prepsfadmom.prepsfadmom import get_phase_angles
    from kdeblend.gpu.prep import GeometryCache, prep_groups

    cache = GeometryCache()
    for obs, objects in _obs_battery():
        deb, _ = kdeblend.build_deblender(
            obs, [dict(o) for o in objects], **FIT_KW,
        )
        ep = prep_epoch(
            obs, band=0, fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
            use_noise_image=True,
        )
        ep['vcen'] = 0.0
        ep['ucen'] = 0.0
        ep_s, aux = prep_epoch_scalars(
            obs, band=0, fwhm_smooth=FWHM_SMOOTH,
        )
        slab = prep_groups(
            [_make_req(obs, deb, aux, ep_s)], cache,
        )
        for i in range(deb.nobj):
            vi, ui = deb.positions[i]
            Sw = deb.Sw[i]
            alpha, beta = get_phase_angles(ep, vi, ui)
            sums = np.zeros(6)
            admom_ksums(
                ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                alpha, beta, ep['kv'], ep['ku'],
                Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                sums,
            )
            got = slab.esums[0][i]
            scale = np.abs(sums).max()
            assert np.abs(got - sums).max() < 1e-11 * scale, (
                i, got, sums,
            )


def test_stub_fit_matches_host_pack():
    """a full fit through the resident path (stub epochs +
    measured init sums + device mode assembly) matches the
    host-pack fit at the state level"""
    from ngmix.prepsfadmom.prep import prep_epoch_scalars
    from kdeblend.gpu import _core
    from kdeblend.gpu.prep import (
        GeometryCache, prep_groups, assemble_mode_fields,
    )

    battery = _obs_battery()

    # ---- reference: full CPU construction + host pack ----
    debs_ref = []
    for obs, objects in battery:
        deb, _ = kdeblend.build_deblender(
            obs, [dict(o) for o in objects], **FIT_KW,
        )
        debs_ref.append(deb)
    out_ref = _core.run_gpu(_core.to_gpu(
        _core.pack_host(debs_ref), fp32=False,
    ))

    # ---- resident path: stub construction + device prep ----
    cache = GeometryCache()
    debs = []
    reqs = []
    scalars = []
    for obs, objects in battery:
        ep_s, aux = prep_epoch_scalars(
            obs, band=0, fwhm_smooth=FWHM_SMOOTH,
        )
        ep_s['vcen'] = 0.0
        ep_s['ucen'] = 0.0
        deb, _ = kdeblend.build_deblender(
            obs, [dict(o) for o in objects],
            epochs=[ep_s], defer_flux_init=True, **FIT_KW,
        )
        debs.append(deb)
        reqs.append(_make_req(obs, deb, aux, ep_s))
        scalars.append(ep_s)
    slab = prep_groups(reqs, cache)
    for gi, deb in enumerate(debs):
        deb._measured_init_sums5 = (
            slab.esums[gi][:, 5].reshape(-1, 1)
        )
        deb._init_fluxes()

    # init fluxes must match the reference construction
    for deb, dref in zip(debs, debs_ref):
        for m, mr in zip(deb.models, dref.models):
            assert np.allclose(m['F'], mr['F'], rtol=1e-10)

    small = _core.pack_host_small(debs)
    modes = assemble_mode_fields(
        [(slab, gi) for gi in range(len(debs))], fp32=False,
    )
    moff = slab.moff
    packed = _core.to_gpu_device_modes(
        small, modes, moff, fp32=False,
    )
    out = _core.run_gpu(packed)

    # the device rfft perturbs kim/ef2 at ~1e-12, and the fitter's
    # converged state is only localized to the tol=1e-5 envelope
    # amplified by the slow collective mode (measured: a 1e-13
    # host-side perturbation moves numiter by tens of sweeps and
    # the state at the 1e-4..1e-2 level depending on hardware);
    # so the gate is: discrete outcomes equal, all groups
    # converged (maxiter is generous), state within the envelope
    assert np.all(out_ref['converged'] == 1)
    assert np.array_equal(out['converged'],
                          out_ref['converged'])
    assert np.array_equal(out['otype'], out_ref['otype'])
    assert np.array_equal(out['dbflags'], out_ref['dbflags'])
    for k in ('F', 'cov', 'sw', 'pos', 'objsums'):
        a, b = out[k], out_ref[k]
        scale = np.abs(b).max()
        assert np.abs(a - b).max() < 1e-2 * scale, (
            k, np.abs(a - b).max(), scale,
        )


def test_bigdim_fp32_scope():
    """a group whose padded dim exceeds the fp64 scope but fits
    the fp32 kernel: host-pack fp32 fit matches the CPU fitter at
    the class level; the fp64 upload path refuses it"""
    from kdeblend.gpu import (
        _core, DIM_MAX, DIM_MAX_FP64, pack_host, to_gpu,
        fit_groups, writeback, install_sum_overrides,
        result_from_state,
    )

    rng = np.random.RandomState(44)
    # a wide pair on a 280px stamp: target_dim 1120
    from _sims import make_blend_obs
    sep = 30.0
    a = dict(kind='exp', e1=0.1, e2=0.0, hlr=0.5, flux=8.0,
             v=0.0, u=-sep / 2 * 0.2)
    b = dict(kind='exp', e1=-0.05, e2=0.1, hlr=0.4, flux=6.0,
             v=0.5, u=sep / 2 * 0.2)
    obs = make_blend_obs([a, b], 0.9, noise=0.01, rng=rng,
                         dim=280)
    obs.noise = rng.normal(scale=0.01, size=obs.image.shape)
    objects = [
        dict(v=0.0, u=-sep / 2 * 0.2, Tguess=0.5, type='exp'),
        dict(v=0.5, u=sep / 2 * 0.2, Tguess=0.4, type='exp'),
    ]
    kw = dict(FIT_KW)
    kw['maxiter'] = 500

    deb_cpu, _ = kdeblend.build_deblender(
        obs, [dict(o) for o in objects], **kw)
    deb_gpu, _ = kdeblend.build_deblender(
        obs, [dict(o) for o in objects], **kw)
    ep = deb_gpu.epochs_per_obj[0][0]
    assert DIM_MAX_FP64 < ep['dim'] <= DIM_MAX, ep['dim']

    # fp64 path must refuse it up front
    with pytest.raises(ValueError):
        to_gpu(pack_host([deb_gpu]), fp32=False)

    res_cpu = deb_cpu.go()
    res_gpu = fit_groups([deb_gpu], fp32=True)[0]
    assert res_gpu['converged'] == res_cpu['converged']
    for oc, og in zip(res_cpu['objects'], res_gpu['objects']):
        assert og['type'] == oc['type']
        assert og['deblend_flags'] == oc['deblend_flags']
        assert abs(og['e1'] - oc['e1']) < 5.0e-3
        assert abs(og['T'] / oc['T'] - 1) < 5.0e-3
