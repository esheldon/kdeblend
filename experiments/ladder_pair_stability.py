"""
Close-pair stability study for the ladder: the scene-wide joint
amplitude solve vs the per-object iteration, as the pair
tightens.  Requirement 3 of the lensing case.

The feared failure mode of profile freedom is two overlapping
ladders trading flux shell-by-shell through their overlapping
rungs -- the K-DOF version of "who owns the overlap".  In the
deblender the amplitudes can be updated per object within the
Gauss-Seidel sweep (iterative, like every other update) or
solved scene-wide in one closed-form linear system (the
cross-object Gram blocks are product gaussians).  This toy
measures, for an equal-flux Sersic n=2 pair vs separation:

- the production baselines: gauss and exp pair fits with the
  actual deblender (converged, niter, exp flux errors)
- conditioning of the joint amp system (fraction units, rows
  noise-weighted at s2n=30, both objects' aperture sets), raw
  and with the standard prior tau0=1 toward the exp profile
- the spectral radius rho of the block Gauss-Seidel iteration
  on the priored system.  The system is SPD so rho < 1 always
  (the amp layer cannot diverge); the question is how slow it
  gets: sweeps-to-1e-8 ~ ln(1e-8)/ln(rho).  The joint solve is
  one step at any separation.
- noiseless joint-solve quality: flux attribution error
  |sum(a_i) - 1| per object and the subtraction residual under
  a faint probe weight (perpendicular at 1.5", and 2" outside
  the pair), per unit bright flux
- noisy (s2n=30 per object, fixed frames from the noiseless
  pair fit): scatter of the per-object flux attribution --
  does the tight pair trade flux noisily even when the mean is
  controlled?

Frames (weights) come from the noiseless gauss pair fit; frame
noise under recentering for tight blends is the separate known
issue tracked in TODO.md and is not re-measured here.

Run: python experiments/ladder_pair_stability.py [nreal]
"""
import sys
import os
import numpy as np
import ngmix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import ladder_contamination as lc  # noqa: E402
from _sims import make_blend_obs  # noqa: E402
from ngmix.prepsfadmom.models import model_comps  # noqa: E402
from kdeblend.deblender import build_deblender, _prep_epochs  # noqa: E402
from ngmix.observation import get_mb_obs  # noqa: E402

RUNGS = lc.RUNGS
AP_FACS = lc.AP_FACS
T_FAINT = lc.T_FAINT
PSF_FWHM = 0.8
DIM = 128
NSERSIC = 2.0
HLR = 0.5
SEPS = [3.0, 2.0, 1.5, 1.0, 0.75, 0.5]
TAU0 = 1.0
S2N_NOISY = 30.0
NREAL = 100
NNOISE = 150
SIGMA_CAL = 1.0e-3
SEED = 4242
K = RUNGS.size


def pair_comps(d):
    return [
        dict(kind='sersic', n=NSERSIC, hlr=HLR, flux=1.0,
             e1=0.0, e2=0.0, v=0.0, u=-d / 2),
        dict(kind='sersic', n=NSERSIC, hlr=HLR, flux=1.0,
             e1=0.0, e2=0.0, v=0.0, u=+d / 2),
    ]


def fit_pair(obs, t, d, maxiter=500):
    objs = [dict(v=0.0, u=-d / 2, type=t, Tguess=0.6),
            dict(v=0.0, u=+d / 2, type=t, Tguess=0.6)]
    deb, _ = build_deblender(obs, objs, maxiter=maxiter)
    return deb, deb.go()


def frame_of(Sw, Tsmooth):
    sm = Tsmooth / 2
    Sb = np.asarray(Sw) - np.diag([sm, sm])
    ev, evec = np.linalg.eigh(Sb)
    Sb = (evec * np.maximum(ev, 0.1 * sm)) @ evec.T
    return (RUNGS * Sb[0, 0] + sm, RUNGS * Sb[0, 1],
            RUNGS * Sb[1, 1] + sm)


def build_system(ep, us, rungs, aps, sig_row):
    """joint rows: for each object center i and aperture j, the
    data sum and the template row over both objects' rungs.
    Returns (Mw, dw, M_raw, dvec) in raw amp units"""
    nob = len(us)
    nap = len(aps[0])
    nrow = nob * nap
    M = np.zeros((nrow, nob * K))
    dvec = np.zeros(nrow)
    for i in range(nob):
        for j, w in enumerate(aps[i]):
            r = i * nap + j
            dvec[r] = lc.measure_sums(ep, 0.0, us[i], w)[5]
            for ip in range(nob):
                S00, S01, S11 = rungs[ip]
                for k in range(K):
                    M[r, ip * K + k] = lc.comp_flux_sum(
                        1.0, S00[k], S01[k], S11[k],
                        0.0, us[ip] - us[i], w,
                    )
    rw = np.concatenate([1.0 / sig_row] * nob)
    return M * rw[:, None], dvec * rw, M, dvec


def gs_rho(A):
    """spectral radius of the 2x2 block Gauss-Seidel iteration"""
    A11, A12 = A[:K, :K], A[:K, K:]
    A21, A22 = A[K:, :K], A[K:, K:]
    T = np.linalg.solve(A22, A21) @ np.linalg.solve(A11, A12)
    return np.abs(np.linalg.eigvals(T)).max() ** 0.5


def main():
    global NREAL
    if len(sys.argv) > 1:
        NREAL = int(sys.argv[1])
    rng = np.random.default_rng(SEED)

    # single-object s2n calibration
    sobs = make_blend_obs(
        [dict(kind='sersic', n=NSERSIC, hlr=HLR, flux=1.0,
              e1=0.0, e2=0.0, v=0.0, u=0.0)],
        PSF_FWHM, dim=DIM)
    cal_obs = ngmix.Observation(
        sobs.image, jacobian=sobs.jacobian,
        weight=np.ones_like(sobs.image) / SIGMA_CAL ** 2,
        psf=sobs.psf)
    deb_s, res_s = fit_pair(cal_obs, 'gauss', 0.0, maxiter=100) \
        if False else (None, None)
    deb1, _ = build_deblender(
        cal_obs, [dict(v=0.0, u=0.0, type='gauss', Tguess=0.6)],
        maxiter=100)
    r1 = deb1.go()
    s2n_cal = r1['objects'][0]['s2n']
    sigma_n = SIGMA_CAL * s2n_cal / S2N_NOISY
    print(f"single-object s2n at sigma={SIGMA_CAL:g}: {s2n_cal:.1f}"
          f" -> sigma_n(s2n={S2N_NOISY:g}) = {sigma_n:.2e}")

    lam0 = 1.0 / TAU0 ** 2

    for d in SEPS:
        obs = make_blend_obs(pair_comps(d), PSF_FWHM, dim=DIM)
        deb_g, res_g = fit_pair(obs, 'gauss', d)
        deb_e, res_e = fit_pair(obs, 'exp', d)
        Tsmooth = deb_g.Tsmooth
        sm = Tsmooth / 2
        ep = deb_g.epochs_per_obj[0][0]
        us = [-d / 2, d / 2]

        rungs = [frame_of(deb_g.Sw[i], Tsmooth) for i in (0, 1)]
        aps = [[af * np.asarray(deb_g.Sw[i]) for af in AP_FACS]
               for i in (0, 1)]

        # unit-noise aperture sigmas (object 1's set; symmetric)
        ap_sums = np.zeros((NNOISE, len(AP_FACS)))
        for i in range(NNOISE):
            nobs = ngmix.Observation(
                rng.normal(size=obs.image.shape),
                jacobian=obs.jacobian,
                weight=np.ones_like(obs.image), psf=obs.psf)
            epsn = _prep_epochs(
                get_mb_obs(nobs), fwhm_smooth=deb_g.fwhm_smooth,
                ap_rad=0.0, use_noise_image=False,
                vcen=0.0, ucen=0.0)
            for j, w in enumerate(aps[0]):
                ap_sums[i, j] = lc.measure_sums(
                    epsn[0], 0.0, us[0], w)[5]
        sig_row = ap_sums.std(axis=0) * sigma_n

        Mw, dw, M_raw, dvec = build_system(
            ep, us, rungs, aps, sig_row)

        # fraction units via the exp pair fluxes
        Fe = np.array([deb_e.models[i]['F'][0] for i in (0, 1)])
        Fe = np.where(np.isfinite(Fe) & (Fe > 0), Fe, 1.0e-6)
        cscale = np.repeat(Fe, K)
        Mwf = Mw * cscale[None, :]
        A = Mwf.T @ Mwf
        Ap = A + lam0 * np.eye(2 * K)

        # prior centers: exp profile of each object on its rungs
        a0 = np.zeros(2 * K)
        for i in (0, 1):
            fr, S00, S01, S11 = model_comps(
                deb_e.models[i], Tsmooth)
            nap = len(aps[i])
            d0 = np.array([
                lc.comp_flux_sum(fr, S00, S01, S11, 0.0, 0.0, w)
                for w in aps[i]])
            Mo = M_raw[i * nap:(i + 1) * nap, i * K:(i + 1) * K]
            MtM = Mo.T @ Mo
            a0[i * K:(i + 1) * K] = np.linalg.solve(
                MtM + 1e-8 * np.trace(MtM) / K * np.eye(K),
                Mo.T @ d0)

        conds = np.linalg.cond(A)
        condp = np.linalg.cond(Ap)
        rho = gs_rho(Ap)
        nsweep = np.log(1e-8) / np.log(rho) if rho < 1 else np.inf

        # noiseless joint solve
        z = np.linalg.solve(Ap, Mwf.T @ dw + lam0 * a0)
        amps = z * cscale
        fluxerr = [amps[i * K:(i + 1) * K].sum() - 1.0
                   for i in (0, 1)]

        # probe residuals
        Twf = T_FAINT / 2 + sm
        wf = np.diag([Twf, Twf])
        s_self = lc.comp_flux_sum(1.0, Twf, 0.0, Twf, 0.0, 0.0, wf)
        probes = [(1.5, 0.0), (0.0, d / 2 + 2.0)]
        cprobe = []
        for (pv, pu) in probes:
            t = lc.measure_sums(ep, pv, pu, wf)[5]
            m = 0.0
            for i in (0, 1):
                S00, S01, S11 = rungs[i]
                m += lc.comp_flux_sum(
                    amps[i * K:(i + 1) * K], S00, S01, S11,
                    -pv, us[i] - pu, wf)
            cprobe.append((t - m) / s_self)

        # noisy attribution scatter, fixed frames
        f1s, f2s = [], []
        for i in range(NREAL):
            nim = obs.image + rng.normal(scale=sigma_n,
                                         size=obs.image.shape)
            nobs = ngmix.Observation(
                nim, jacobian=obs.jacobian,
                weight=np.ones_like(nim) / sigma_n ** 2,
                psf=obs.psf)
            epsn = _prep_epochs(
                get_mb_obs(nobs), fwhm_smooth=deb_g.fwhm_smooth,
                ap_rad=0.0, use_noise_image=False,
                vcen=0.0, ucen=0.0)
            _, _, _, dv = build_system(
                epsn[0], us, rungs, aps, sig_row)
            rw = np.concatenate([1.0 / sig_row] * 2)
            zi = np.linalg.solve(
                Ap, Mwf.T @ (dv * rw) + lam0 * a0)
            ai = zi * cscale
            f1s.append(ai[:K].sum())
            f2s.append(ai[K:].sum())
        f1s, f2s = np.array(f1s), np.array(f2s)

        print(f"\n===== d={d:.2f} arcsec =====")
        print(f"  deblender: gauss conv={res_g['converged']}"
              f" niter={res_g['numiter']}"
              f" flags={[o['deblend_flags'] for o in res_g['objects']]}"
              f" | exp conv={res_e['converged']}"
              f" niter={res_e['numiter']}"
              f" Fexp-1={Fe[0] - 1:+.3f},{Fe[1] - 1:+.3f}")
        print(f"  joint amp system: cond(A)={conds:9.2e}"
              f"  cond(A+prior)={condp:9.2e}"
              f"  rho_GS={rho:7.4f}  sweeps(1e-8)={nsweep:6.1f}")
        print(f"  noiseless joint: flux-err ="
              f" {fluxerr[0]:+.4f}, {fluxerr[1]:+.4f}"
              f"  probe resid perp {cprobe[0]:+9.2e}"
              f" far {cprobe[1]:+9.2e}")
        print(f"  noisy s2n={S2N_NOISY:g} (N={NREAL}): flux"
              f" mean {f1s.mean() - 1:+.4f},{f2s.mean() - 1:+.4f}"
              f"  std {f1s.std():.4f},{f2s.std():.4f}"
              f"  corr {np.corrcoef(f1s, f2s)[0, 1]:+.3f}")


if __name__ == '__main__':
    main()
