"""
Noise study for the free-amplitude ladder: does subtracting a
ladder-modeled bright neighbor inject more noise into a faint
neighbor's corrected sums than subtracting an exp model, and how
does the noise-scaled ridge control the bias-variance tradeoff?

This is requirement 2 of the lensing case (do not worsen
ellipticity noise) plus the lambda calibration.  Per noise
realization of an isolated bright Sersic:

- fit gauss and exp with the actual deblender machinery
- fit the ladder by the K-aperture solve with noise-weighted
  rows and a gaussian prior of width tau on the amplitude
  fractions, centered on the exp profile (the smooth
  revert-to-exp regularization; the data side is normalized by
  the realization's exp-fit flux so the prior carries no oracle
  flux information).  The rung frame comes from the
  realization's own noisy gauss fit (frame noise included); the
  measurement apertures are fixed
- measure the corrected sums a zero-flux faint object at
  separation d would get: (noisy data sums under the faint
  weight) minus (fitted model sums), for each model type and
  'none' (no subtraction)

Over realizations, the mean residual is the contamination bias
and the std is the total noise of the faint object's corrected
sums, including the shared-pixel correlation between the data
term and the fitted models.  Reported per channel:

- flux row: bias per unit bright/faint flux ratio (as in
  ladder_contamination) and the noise inflation factor
  std(model)/std(none) -- the factor by which the subtraction
  inflates the faint flux noise over the no-neighbor floor
- M1 row: the same inflation factor, the e1-noise proxy
  (requirement 2's number), and the M1 bias normalized by the
  faint object's own T sum (an e1 bias proxy per unit ratio)

The aperture-sum noise sigmas for the row weighting come from
pure-noise Monte Carlo at unit sigma, scaled to each noise
level; cross-aperture correlations are ignored in the weighting
(slightly suboptimal for the ladder, so conservative), but are
fully present in the reported empirical scatter.

Frame guard: at low S/N the noisy gauss weight can drop below
the smoothing; the base covariance eigenvalues are floored at
0.1 x Tsmooth/2.  In the deblender this becomes a smooth floor.

Run: python experiments/ladder_noise_study.py [nreal]
"""
import sys
import os
import numpy as np
import galsim
import ngmix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import ladder_contamination as lc  # noqa: E402
from _sims import make_blend_obs, SCALE, GSPARAMS  # noqa: E402
from ngmix.prepsfadmom.models import model_comps  # noqa: E402
from kdeblend.deblender import build_deblender, _prep_epochs  # noqa: E402
from ngmix.observation import get_mb_obs  # noqa: E402

# --- configuration ---------------------------------------------------

PSF_FWHM = lc.PSF_FWHM
DIM = lc.DIM
HLR = lc.HLR
T_FAINT = lc.T_FAINT
RUNGS = lc.RUNGS
AP_FACS = lc.AP_FACS

NS = [2.0, 4.0]
TARGET_S2NS = [10.0, 100.0, 1000.0]
SEPS = [1.0, 2.0, 4.0]
TAUS = [0.03, 0.1, 0.3, 1.0]
TAU_MAIN = 0.3
NREAL = 200
NNOISE = 200          # pure-noise draws for the aperture sigmas
MAXITER = 100
SIGMA_CAL = 1.0e-3    # noise level for the s2n calibration fit
SEED = 991


def make_obs(im, sigma, jac, psf_im):
    return ngmix.Observation(
        im, jacobian=jac,
        weight=np.ones_like(im) / sigma ** 2,
        psf=ngmix.Observation(psf_im, jacobian=jac),
    )


def fit_one(obs, t, fwhm_smooth=None, epochs=None):
    deb, _ = build_deblender(
        obs, [dict(v=0.0, u=0.0, type=t, Tguess=1.0)],
        maxiter=MAXITER, fwhm_smooth=fwhm_smooth, epochs=epochs,
    )
    res = deb.go()
    return deb, res


def floored_base(Sw, Tsmooth):
    """pre-smoothing base covariance with an eigenvalue floor"""
    sm = Tsmooth / 2
    Sb = Sw - np.diag([sm, sm])
    evals, evecs = np.linalg.eigh(Sb)
    evals = np.maximum(evals, 0.1 * sm)
    return (evecs * evals) @ evecs.T


def rung_covs(Sbase, Tsmooth):
    sm = Tsmooth / 2
    return (
        RUNGS * Sbase[0, 0] + sm,
        RUNGS * Sbase[0, 1],
        RUNGS * Sbase[1, 1] + sm,
    )


def template_matrix(So00, So01, So11, apertures):
    M = np.zeros((len(apertures), RUNGS.size))
    for j, w in enumerate(apertures):
        for k in range(RUNGS.size):
            M[j, k] = lc.comp_flux_sum(
                1.0, So00[k], So01[k], So11[k], 0.0, 0.0, w,
            )
    return M


def model_res_sums(F, S00, S01, S11, dsep, wf):
    """all 6 model sums at the faint position"""
    n = np.size(F)
    sums = np.zeros(6)
    from ngmix.prepsfadmom.models_nb import gauss_comps_ksums
    gauss_comps_ksums(
        np.atleast_1d(np.asarray(F, dtype='f8')),
        np.atleast_1d(np.asarray(S00, dtype='f8')),
        np.atleast_1d(np.asarray(S01, dtype='f8')),
        np.atleast_1d(np.asarray(S11, dtype='f8')),
        np.full(n, -dsep), np.full(n, 0.0),
        wf[0, 0], wf[0, 1], wf[1, 1], 1.0, sums,
    )
    return sums


def calibrate(n, rng):
    """noiseless reference: image, s2n scale, apertures, aperture
    noise sigmas at unit image noise, exp prior data vector"""
    comp = dict(kind='sersic', n=n, hlr=HLR, flux=1.0,
                e1=0.0, e2=0.0, v=0.0, u=0.0)
    obs0 = make_blend_obs([comp], PSF_FWHM, dim=DIM)
    im0 = obs0.image.copy()
    psf_im = obs0.psf.image.copy()
    jac = obs0.jacobian

    cal_obs = make_obs(im0, SIGMA_CAL, jac, psf_im)
    deb_g, res_g = fit_one(cal_obs, 'gauss')
    deb_e, res_e = fit_one(
        cal_obs, 'exp', fwhm_smooth=deb_g.fwhm_smooth,
        epochs=deb_g.epochs_per_obj[0],
    )
    s2n_cal = res_g['objects'][0]['s2n']
    Tsmooth = deb_g.Tsmooth
    fwhm_smooth = deb_g.fwhm_smooth
    Sw_ref = deb_g.Sw[0].copy()
    apertures = [af * Sw_ref for af in AP_FACS]

    # exp profile aperture sums at unit flux, for the prior
    fracs, eS00, eS01, eS11 = model_comps(deb_e.models[0], Tsmooth)
    d0 = np.array([
        lc.comp_flux_sum(fracs, eS00, eS01, eS11, 0.0, 0.0, w)
        for w in apertures
    ])

    # pure-noise aperture sigmas at unit sigma
    ap_sums = np.zeros((NNOISE, len(apertures)))
    for i in range(NNOISE):
        nim = rng.normal(size=im0.shape)
        nobs = make_obs(nim, 1.0, jac, psf_im)
        eps = _prep_epochs(
            get_mb_obs(nobs), fwhm_smooth=fwhm_smooth, ap_rad=0.0,
            use_noise_image=False, vcen=0.0, ucen=0.0,
        )
        for j, w in enumerate(apertures):
            ap_sums[i, j] = lc.measure_sums(eps[0], 0.0, 0.0, w)[5]
    sig_unit = ap_sums.std(axis=0)

    return dict(
        im0=im0, psf_im=psf_im, jac=jac, s2n_cal=s2n_cal,
        Tsmooth=Tsmooth, fwhm_smooth=fwhm_smooth,
        apertures=apertures, d0=d0, sig_unit=sig_unit,
    )


def run_config(n, target_s2n, ref, rng):
    sigma_n = SIGMA_CAL * ref['s2n_cal'] / target_s2n
    Tsmooth = ref['Tsmooth']
    apertures = ref['apertures']
    d0 = ref['d0']
    sig = ref['sig_unit'] * sigma_n

    Twf = T_FAINT / 2 + Tsmooth / 2
    wf = np.diag([Twf, Twf])
    self_sums = model_res_sums(1.0, Twf, 0.0, Twf, 0.0, wf)
    s_self, sT_self = self_sums[5], self_sums[4]

    mnames = ['none', 'gauss', 'exp'] + [f'lad{t:g}' for t in TAUS]
    res = {m: {d: [] for d in SEPS} for m in mnames}
    amps_kept = []
    nconv = {'gauss': 0, 'exp': 0}
    nused = 0

    for i in range(NREAL):
        im = ref['im0'] + rng.normal(scale=sigma_n,
                                     size=ref['im0'].shape)
        obs = make_obs(im, sigma_n, ref['jac'], ref['psf_im'])
        try:
            deb_g, res_g = fit_one(obs, 'gauss')
            deb_e, res_e = fit_one(
                obs, 'exp', fwhm_smooth=deb_g.fwhm_smooth,
                epochs=deb_g.epochs_per_obj[0],
            )
        except Exception:
            continue
        nconv['gauss'] += res_g['converged']
        nconv['exp'] += res_e['converged']
        ep = deb_g.epochs_per_obj[0][0]

        # models
        models = {}
        gf, gS00, gS01, gS11 = model_comps(deb_g.models[0], Tsmooth)
        models['gauss'] = (deb_g.models[0]['F'][0] * gf,
                          gS00, gS01, gS11)
        ef, eS00, eS01, eS11 = model_comps(deb_e.models[0], Tsmooth)
        Fhat = deb_e.models[0]['F'][0]
        models['exp'] = (Fhat * ef, eS00, eS01, eS11)
        models['none'] = (np.zeros(1), np.ones(1), np.zeros(1),
                          np.ones(1))
        if not np.isfinite(Fhat) or Fhat <= 0:
            Fhat = max(deb_g.models[0]['F'][0], 1.0e-6)

        # ladder
        Sbase = floored_base(deb_g.Sw[0], Tsmooth)
        lS00, lS01, lS11 = rung_covs(Sbase, Tsmooth)
        M = template_matrix(lS00, lS01, lS11, apertures)
        dvec = np.array([
            lc.measure_sums(ep, 0.0, 0.0, w)[5] for w in apertures
        ])
        # prior center: exp profile expressed on this frame
        MtM = M.T @ M
        eps_r = 1.0e-8 * np.trace(MtM) / RUNGS.size
        a0 = np.linalg.solve(MtM + eps_r * np.eye(RUNGS.size),
                             M.T @ d0)
        # noise-weighted rows, solved in fraction units:
        # model = Fhat * M @ abar, rows scaled by 1/sig
        Mwf = (M / sig[:, None]) * Fhat
        A = Mwf.T @ Mwf
        b = Mwf.T @ (dvec / sig)
        for tau in TAUS:
            lam = 1.0 / tau ** 2
            abar = np.linalg.solve(
                A + lam * np.eye(RUNGS.size), b + lam * a0,
            )
            models[f'lad{tau:g}'] = (Fhat * abar, lS00, lS01, lS11)
            if tau == TAU_MAIN:
                amps_kept.append(Fhat * abar)

        nused += 1
        for d in SEPS:
            data = lc.measure_sums(ep, d, 0.0, wf)
            for m in mnames:
                F, S00, S01, S11 = models[m]
                r = data - model_res_sums(F, S00, S01, S11, d, wf)
                res[m][d].append(r)

    # --- report ------------------------------------------------------
    print(f"\n===== n={n:.1f}  target_s2n={target_s2n:g}"
          f"  sigma_n={sigma_n:.3e}  nused={nused}/{NREAL}"
          f"  converged: gauss {nconv['gauss']} exp {nconv['exp']}"
          f" =====")
    stats = {}
    for m in mnames:
        stats[m] = {d: (np.mean(res[m][d], axis=0),
                        np.std(res[m][d], axis=0))
                    for d in SEPS}

    show = ['none', 'gauss', 'exp', f'lad{TAU_MAIN:g}']
    print("  flux row: bias per unit flux ratio (b), noise "
          "inflation over floor (x)")
    print("  " + " ".join([f"{'d':>5s}"] + [
        f"{m:>20s}" for m in show]))
    for d in SEPS:
        floor = stats['none'][d][1][5]
        row = [f"{d:5.1f}"]
        for m in show:
            mu, sd = stats[m][d]
            row.append(f"b{mu[5] / s_self:+9.1e} x{sd[5] / floor:5.2f}")
        print("  " + " ".join(row))

    print("  M1 row (e1 proxy): bias/sT_self (b), noise inflation (x)")
    for d in SEPS:
        floor = stats['none'][d][1][2]
        row = [f"{d:5.1f}"]
        for m in show:
            mu, sd = stats[m][d]
            row.append(f"b{mu[2] / sT_self:+9.1e} x{sd[2] / floor:5.2f}")
        print("  " + " ".join(row))

    print("  tau sweep at d=2: flux b / flux x / M1 x")
    parts = []
    for tau in TAUS:
        mu, sd = stats[f'lad{tau:g}'][2.0]
        fl = stats['none'][2.0][1]
        parts.append(
            f"t={tau:g}: {mu[5] / s_self:+8.1e} /"
            f" {sd[5] / fl[5]:5.2f} / {sd[2] / fl[2]:5.2f}"
        )
    print("    " + "   ".join(parts))
    if amps_kept:
        am = np.array(amps_kept)
        print(f"  amps(tau={TAU_MAIN}) mean "
              + np.array2string(am.mean(axis=0), precision=3)
              + "\n                 std  "
              + np.array2string(am.std(axis=0), precision=3))
    return {m: res[m] for m in mnames}, s_self, sT_self


def main():
    global NREAL
    if len(sys.argv) > 1:
        NREAL = int(sys.argv[1])
    np.set_printoptions(linewidth=200, suppress=False)
    out = {}
    for n in NS:
        rng = np.random.default_rng([SEED, int(n * 10)])
        ref = calibrate(n, rng)
        print(f"\n##### n={n:.1f}: s2n at sigma={SIGMA_CAL:g} is "
              f"{ref['s2n_cal']:.1f}")
        for s2n in TARGET_S2NS:
            out[(n, s2n)] = run_config(n, s2n, ref, rng)
    return out


if __name__ == '__main__':
    main()
