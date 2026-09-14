"""
Noisy validation of per-band ladder amplitudes with a
cross-band prior.

The two-band color study showed that a color gradient in a
bright neighbor puts a shared-structure floor dc ~ 1.5-2e-2 on
the band-dependent leakage (all shared models alike), and that
free per-band amplitudes remove it (dc ~ 1e-5).  This study
asks what that freedom costs in *noise*: per-band amp vectors
double the amplitude DOF, and the color channel is exactly
where the faint object's photometry lives.

Per noise realization of the gradient-config bright composite
(g: disk .7/bulge .3, r: disk .4/bulge .6, same PSF):
- fit gauss and exp multiband with the deblender (shared
  structure, per-band fluxes)
- solve the joint 2K per-band amplitude system in fraction
  units (a_b = Fhat_b ubar_b, Fhat_b the realization's exp-fit
  per-band flux): noise-weighted aperture rows per band, a weak
  prior (width tau0=1) toward the exp profile, and the
  cross-band prior |ubar_g - ubar_r|^2 / taux^2.  taux -> 0 is
  the shared ladder, taux -> inf the free per-band one.
- measure the faint-position corrected flux sums per band, for
  each model and 'none'

Over realizations, per (s2n, d, model): the color-leakage bias
dc = mean(r_g - r_r)/s_self per unit same-band flux ratio, the
color noise inflation x_col = std(r_g - r_r) over the
no-subtraction floor, and the single-band flux inflation x_g.
The taux sweep maps the color bias-variance tradeoff and tests
whether a fixed taux gives gradient correction without noise
cost.

Run: python experiments/ladder_color_noise.py [nreal]
"""
import sys
import os
import numpy as np
import ngmix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import ladder_contamination as lc  # noqa: E402
import ladder_color_study as cs  # noqa: E402
from _sims import make_blend_mbobs  # noqa: E402
from ngmix.prepsfadmom.models import model_gauss_components  # noqa: E402
from kdeblend.deblender import build_deblender, _prep_epochs  # noqa: E402
from ngmix.observation import get_mb_obs  # noqa: E402

RUNGS = lc.RUNGS
AP_FACS = lc.AP_FACS
T_FAINT = lc.T_FAINT
DIM = cs.DIM
SEPS = [1.0, 2.0, 4.0]
TARGET_S2NS = [10.0, 30.0, 100.0, 1000.0]
TAU0 = 1.0
LADDERS = [('lad-sh', 1.0e-5), ('lad-x.03', 0.03),
           ('lad-x.1', 0.1), ('lad-x.3', 0.3),
           ('lad-free', 1.0e3)]
NREAL = 200
NNOISE = 200
MAXITER = 100
SIGMA_CAL = 1.0e-3
SEED = 2077

COMPS_PER_BAND = [cs.band_comps(0.7, 0.3), cs.band_comps(0.4, 0.6)]
PSFS = [0.8, 0.8]


def fit_mb(mbobs, t, fwhm_smooth=None, epochs=None):
    deb, _ = build_deblender(
        mbobs, [dict(v=0.0, u=0.0, type=t, Tguess=1.0)],
        maxiter=MAXITER, fwhm_smooth=fwhm_smooth, epochs=epochs,
    )
    return deb, deb.go()


def make_mbobs(ims, sigma, jac, psf_ims):
    mbobs = ngmix.MultiBandObsList()
    for im, pim in zip(ims, psf_ims):
        ol = ngmix.ObsList()
        ol.append(ngmix.Observation(
            im, jacobian=jac, weight=np.ones_like(im) / sigma ** 2,
            psf=ngmix.Observation(pim, jacobian=jac),
        ))
        mbobs.append(ol)
    return mbobs


def calibrate(rng):
    mb0 = make_blend_mbobs(COMPS_PER_BAND, PSFS, dim=DIM)
    ims0 = [ol[0].image.copy() for ol in mb0]
    psf_ims = [ol[0].psf.image.copy() for ol in mb0]
    jac = mb0[0][0].jacobian

    cal = make_mbobs(ims0, SIGMA_CAL, jac, psf_ims)
    deb_g, res_g = fit_mb(cal, 'gauss')
    deb_e, res_e = fit_mb(
        cal, 'exp', fwhm_smooth=deb_g.fwhm_smooth,
        epochs=deb_g.epochs_per_obj[0],
    )
    s2n_cal = res_g['objects'][0]['s2n']
    Tsmooth = deb_g.Tsmooth
    Sw_ref = deb_g.Sw[0].copy()
    apertures = [af * Sw_ref for af in AP_FACS]

    fr, S00, S01, S11 = model_gauss_components(deb_e.models[0], Tsmooth)
    d0 = np.array([
        lc.comp_flux_sum(fr, S00, S01, S11, 0.0, 0.0, w)
        for w in apertures
    ])

    ap_sums = np.zeros((NNOISE, len(apertures)))
    for i in range(NNOISE):
        nobs = ngmix.Observation(
            rng.normal(size=ims0[0].shape), jacobian=jac,
            weight=np.ones_like(ims0[0]),
            psf=ngmix.Observation(psf_ims[0], jacobian=jac),
        )
        eps = _prep_epochs(
            get_mb_obs(nobs), fwhm_smooth=deb_g.fwhm_smooth,
            ap_rad=0.0, use_noise_image=False, vcen=0.0, ucen=0.0,
        )
        for j, w in enumerate(apertures):
            ap_sums[i, j] = lc.measure_sums(eps[0], 0.0, 0.0, w)[5]
    sig_unit = ap_sums.std(axis=0)

    return dict(
        ims0=ims0, psf_ims=psf_ims, jac=jac, s2n_cal=s2n_cal,
        Tsmooth=Tsmooth, fwhm_smooth=deb_g.fwhm_smooth,
        apertures=apertures, d0=d0, sig_unit=sig_unit,
    )


def run_s2n(target_s2n, ref, rng):
    sigma_n = SIGMA_CAL * ref['s2n_cal'] / target_s2n
    Tsmooth = ref['Tsmooth']
    sm = Tsmooth / 2
    apertures = ref['apertures']
    sig = ref['sig_unit'] * sigma_n
    K = RUNGS.size

    Twf = T_FAINT / 2 + sm
    wf = np.diag([Twf, Twf])
    s_self = lc.comp_flux_sum(1.0, Twf, 0.0, Twf, 0.0, 0.0, wf)

    mnames = ['none', 'exp'] + [nm for nm, _ in LADDERS]
    res = {m: {d: [] for d in SEPS} for m in mnames}
    nconv = {'gauss': 0, 'exp': 0}
    nused = 0

    for i in range(NREAL):
        ims = [im + rng.normal(scale=sigma_n, size=im.shape)
               for im in ref['ims0']]
        mbobs = make_mbobs(ims, sigma_n, ref['jac'], ref['psf_ims'])
        try:
            deb_g, res_g = fit_mb(mbobs, 'gauss')
            deb_e, res_e = fit_mb(
                mbobs, 'exp', fwhm_smooth=deb_g.fwhm_smooth,
                epochs=deb_g.epochs_per_obj[0],
            )
        except Exception:
            continue
        nconv['gauss'] += res_g['converged']
        nconv['exp'] += res_e['converged']
        eps_b = deb_g.epochs_per_obj[0]

        models = {}
        fr, S00, S01, S11 = model_gauss_components(deb_e.models[0], Tsmooth)
        Fe = deb_e.models[0]['F']
        models['exp'] = [(Fe[b] * fr, S00, S01, S11) for b in (0, 1)]
        models['none'] = [(np.zeros(1), np.ones(1), np.zeros(1),
                           np.ones(1))] * 2

        Fhat = np.array([
            f if (np.isfinite(f) and f > 0) else 1.0e-6 for f in Fe
        ])
        Sbase = np.asarray(deb_g.Sw[0]) - np.diag([sm, sm])
        evals, evecs = np.linalg.eigh(Sbase)
        Sbase = (evecs * np.maximum(evals, 0.1 * sm)) @ evecs.T
        lS00 = RUNGS * Sbase[0, 0] + sm
        lS01 = RUNGS * Sbase[0, 1]
        lS11 = RUNGS * Sbase[1, 1] + sm
        M = np.zeros((len(apertures), K))
        for j, w in enumerate(apertures):
            for k in range(K):
                M[j, k] = lc.comp_flux_sum(
                    1.0, lS00[k], lS01[k], lS11[k], 0.0, 0.0, w)
        MtM = M.T @ M
        a0 = np.linalg.solve(
            MtM + 1e-8 * np.trace(MtM) / K * np.eye(K),
            M.T @ ref['d0'],
        )
        Mw = M / sig[:, None]
        dvecs = [
            np.array([lc.measure_sums(ep, 0.0, 0.0, w)[5]
                      for w in apertures])
            for ep in eps_b
        ]
        Ab = [Fhat[b] ** 2 * (Mw.T @ Mw) for b in (0, 1)]
        bb = [Fhat[b] * Mw.T @ (dvecs[b] / sig) for b in (0, 1)]
        lam0 = 1.0 / TAU0 ** 2
        Ik = np.eye(K)
        for nm, taux in LADDERS:
            lamx = 1.0 / taux ** 2
            A = np.block([
                [Ab[0] + (lam0 + lamx) * Ik, -lamx * Ik],
                [-lamx * Ik, Ab[1] + (lam0 + lamx) * Ik],
            ])
            rhs = np.concatenate([bb[0] + lam0 * a0,
                                  bb[1] + lam0 * a0])
            z = np.linalg.solve(A, rhs)
            models[nm] = [
                (Fhat[b] * z[b * K:(b + 1) * K], lS00, lS01, lS11)
                for b in (0, 1)
            ]

        nused += 1
        for d in SEPS:
            data = [lc.measure_sums(eps_b[b], d, 0.0, wf)[5]
                    for b in (0, 1)]
            for m in mnames:
                r = []
                for b in (0, 1):
                    F, S00m, S01m, S11m = models[m][b]
                    r.append(data[b] - lc.comp_flux_sum(
                        F, S00m, S01m, S11m, -d, 0.0, wf))
                res[m][d].append(r)

    print(f"\n===== target_s2n={target_s2n:g} sigma_n={sigma_n:.2e}"
          f"  nused={nused}/{NREAL}  converged: gauss"
          f" {nconv['gauss']} exp {nconv['exp']} =====")
    print("  dc = color-leakage bias (c_g - c_r) per unit ratio;"
          " x_col = color noise inflation; x_g = band-g flux"
          " inflation")
    for d in SEPS:
        arr = {m: np.array(res[m][d]) for m in mnames}
        col_floor = (arr['none'][:, 0] - arr['none'][:, 1]).std()
        g_floor = arr['none'][:, 0].std()
        print(f"  d={d:.1f}")
        for m in mnames:
            cg, cr = arr[m][:, 0], arr[m][:, 1]
            dc = (cg - cr).mean() / s_self
            xc = (cg - cr).std() / col_floor
            xg = cg.std() / g_floor
            print(f"    {m:9s} dc {dc:+9.2e}  x_col {xc:5.2f}"
                  f"  x_g {xg:5.2f}")


def main():
    global NREAL
    if len(sys.argv) > 1:
        NREAL = int(sys.argv[1])
    rng = np.random.default_rng(SEED)
    ref = calibrate(rng)
    print(f"##### gradient config: s2n at sigma={SIGMA_CAL:g} is"
          f" {ref['s2n_cal']:.1f}")
    for s2n in TARGET_S2NS:
        run_s2n(s2n, ref, rng)


if __name__ == '__main__':
    main()
