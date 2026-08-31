"""
A well-behaved total flux from the ladder.

Sum(amps) is the model total flux; free outer rungs make it
wing-dominated: noisy (the wings are below the noise) and
vulnerable to neighbor wing-stealing.  Every survey total flux
extrapolates the wings somehow (cmodel: from the exp/dev family,
hence its mismatch bias).  The ladder makes the extrapolation an
explicit, tunable prior: per-rung prior widths tau_k -- loose on
the inner, data-constrained rungs (free core, removing the exp
mismatch bias) and tight on the outer rungs toward the exp
completion (slaved wings, removing the variance and the
stealing).  The layer is linear, so the subtraction model
(loose prior, best bias) and the flux estimator (slaved wings)
are two solves of the same system at no extra data cost.

Part A (single object, Sersic n=2/4, s2n 10/30/100, N=200):
mean bias and std of the total-flux estimators vs truth (=1):
exp model flux (the cmodel-like baseline), and Sum(amps) for
uniform tau=1, 0.3, 0.1 and the slaved variant (inner tau=1,
outer tau=0.05).  The rendered stamp holds im0.sum() of the
true total; estimators extrapolate beyond it.

Part B (wings-on-compact 10:1 pair at d=1): the faint member's
total-flux attribution Sum(amps)/F_true - 1 per variant,
noiseless and noisy (bright s2n=100, N=100) -- does slaving the
wings kill the +145 percent attribution stealing seen with
tau=1?

Caveat: the slaved wings anchor to the exp-fit flux Fhat (a
linear formulation); anchoring to the fitted core scale would
be self-consistent but nonlinear (or one extra iteration).

Run: python experiments/ladder_total_flux.py [nreal]
"""
import sys
import os
import numpy as np
import ngmix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import ladder_contamination as lc  # noqa: E402
import ladder_noise_study as ns  # noqa: E402
import ladder_pair_stability as ps  # noqa: E402
import ladder_hetero_pairs as hp  # noqa: E402
from ngmix.prepsfadmom.models import model_comps  # noqa: E402
from kdeblend.deblender import build_deblender, _prep_epochs  # noqa: E402
from ngmix.observation import get_mb_obs  # noqa: E402

RUNGS = lc.RUNGS
K = RUNGS.size
NREAL = 200

VARIANTS = [
    ('t1', np.ones(K)),
    ('t0.3', np.full(K, 0.3)),
    ('t0.1', np.full(K, 0.1)),
    ('slave', np.array([1.0] * 7 + [0.05] * 3)),
]


def solve_variant(M, dvec, sig, Fhat, a0, taus):
    lam = 1.0 / taus ** 2
    Mwf = (M / sig[:, None]) * Fhat
    A = Mwf.T @ Mwf + np.diag(lam)
    z = np.linalg.solve(A, Mwf.T @ (dvec / sig) + lam * a0)
    return Fhat * z


def part_a():
    print("### Part A: single-object total flux estimators"
          " (bias, std vs truth=1)")
    for n in ns.NS:
        rng = np.random.default_rng([ns.SEED, int(n * 10), 7])
        ref = ns.calibrate(n, rng)
        print(f"\n--- n={n:.1f}: stamp holds"
              f" {ref['im0'].sum():.4f} of the total ---")
        for s2n in [10.0, 30.0, 100.0]:
            sigma_n = ns.SIGMA_CAL * ref['s2n_cal'] / s2n
            sig = ref['sig_unit'] * sigma_n
            tot = {nm: [] for nm, _ in VARIANTS}
            tot['core'] = []
            fexp = []
            for i in range(NREAL):
                im = ref['im0'] + rng.normal(
                    scale=sigma_n, size=ref['im0'].shape)
                obs = ns.make_obs(im, sigma_n, ref['jac'],
                                  ref['psf_im'])
                try:
                    deb_g, _ = ns.fit_one(obs, 'gauss')
                    deb_e, _ = ns.fit_one(
                        obs, 'exp', fwhm_smooth=deb_g.fwhm_smooth,
                        epochs=deb_g.epochs_per_obj[0])
                except Exception:
                    continue
                ep = deb_g.epochs_per_obj[0][0]
                Fhat = deb_e.models[0]['F'][0]
                if not np.isfinite(Fhat) or Fhat <= 0:
                    Fhat = 1e-6
                fexp.append(Fhat)
                Sb = ns.floored_base(deb_g.Sw[0], ref['Tsmooth'])
                lS00, lS01, lS11 = ns.rung_covs(Sb, ref['Tsmooth'])
                M = ns.template_matrix(
                    lS00, lS01, lS11, ref['apertures'])
                MtM = M.T @ M
                a0 = np.linalg.solve(
                    MtM + 1e-8 * np.trace(MtM) / K * np.eye(K),
                    M.T @ ref['d0'])
                dvec = np.array([
                    lc.measure_sums(ep, 0.0, 0.0, w)[5]
                    for w in ref['apertures']])
                for nm, taus in VARIANTS:
                    amps = solve_variant(M, dvec, sig, Fhat,
                                         a0, taus)
                    tot[nm].append(amps.sum())
                # core-anchored completion: scale the exp wing
                # fractions to the fitted inner core, then slave
                z1 = solve_variant(M, dvec, sig, Fhat, a0,
                                   np.full(K, 0.3)) / Fhat
                a0c = a0.copy()
                a0c[7:] *= z1[:7].sum() / a0[:7].sum()
                amps = solve_variant(
                    M, dvec, sig, Fhat, a0c,
                    np.array([0.3] * 7 + [0.05] * 3))
                tot['core'].append(amps.sum())
            fexp = np.array(fexp)
            line = (f"  s2n={s2n:5g}: exp {fexp.mean() - 1:+7.3f}"
                    f" +-{fexp.std():6.3f}")
            for nm in [v[0] for v in VARIANTS] + ['core']:
                t = np.array(tot[nm])
                line += (f"  | {nm} {t.mean() - 1:+7.3f}"
                         f" +-{t.std():6.3f}")
            print(line)


def part_b(rng):
    print("\n### Part B: wings-on-compact 10:1, d=1.0:"
          " faint total-flux attribution per variant")
    c1 = dict(n=4.0, hlr=0.8, flux=1.0)
    c2 = dict(n=1.0, hlr=0.3, flux=0.1)
    d = 1.0
    obs = hp.make_blend_obs(
        [hp.sersic(c1, -d / 2), hp.sersic(c2, +d / 2)],
        hp.PSF_FWHM, dim=hp.DIM)
    us = [-d / 2, d / 2]
    deb_g, _ = hp.fit_pair(obs, 'gauss', d)
    deb_e, _ = hp.fit_pair(obs, 'exp', d)
    Tsmooth = deb_g.Tsmooth
    ep = deb_g.epochs_per_obj[0][0]
    fwhm = deb_g.fwhm_smooth
    Sw = [np.asarray(deb_g.Sw[i]) for i in (0, 1)]
    rungs = [ps.frame_of(Sw[i], Tsmooth) for i in (0, 1)]
    aps = [[af * Sw[i] for af in lc.AP_FACS] for i in (0, 1)]

    # bright s2n=100 noise level
    iso1 = hp.make_blend_obs([hp.sersic(c1, us[0])], hp.PSF_FWHM,
                             dim=hp.DIM)
    debc, _ = build_deblender(
        ngmix.Observation(
            iso1.image, jacobian=iso1.jacobian,
            weight=np.ones_like(iso1.image) / hp.SIGMA_CAL ** 2,
            psf=iso1.psf),
        [dict(v=0.0, u=us[0], type='gauss', Tguess=0.6)],
        maxiter=100)
    sigma_n = hp.SIGMA_CAL * debc.go()['objects'][0]['s2n'] / 100.0

    apn = np.zeros((100, 2, len(lc.AP_FACS)))
    for i in range(100):
        nobs = ngmix.Observation(
            rng.normal(size=obs.image.shape), jacobian=obs.jacobian,
            weight=np.ones_like(obs.image), psf=obs.psf)
        epn = _prep_epochs(
            get_mb_obs(nobs), fwhm_smooth=fwhm, ap_rad=0.0,
            use_noise_image=False, vcen=0.0, ucen=0.0)[0]
        for o in (0, 1):
            for j, w in enumerate(aps[o]):
                apn[i, o, j] = lc.measure_sums(epn, 0.0, us[o], w)[5]
    sig_rows = [apn[:, o, :].std(axis=0) * sigma_n for o in (0, 1)]

    M, dvec, rw = hp.build_joint(ep, us, rungs, aps, sig_rows)
    Fe = np.zeros(2)
    a0 = np.zeros(2 * K)
    for i in (0, 1):
        fr, S00, S01, S11 = model_comps(deb_e.models[i], Tsmooth)
        F = deb_e.models[i]['F'][0]
        Fe[i] = F if (np.isfinite(F) and F > 0) else 1e-6
        nap = len(aps[i])
        d0 = np.array([
            lc.comp_flux_sum(fr, S00, S01, S11, 0.0, 0.0, w)
            for w in aps[i]])
        Mo = M[i * nap:(i + 1) * nap, i * K:(i + 1) * K]
        MtM = Mo.T @ Mo
        a0[i * K:(i + 1) * K] = np.linalg.solve(
            MtM + 1e-8 * np.trace(MtM) / K * np.eye(K), Mo.T @ d0)

    Ftrue = [1.0, 0.1]
    cscale = np.repeat(Fe, K)

    def core_solve(dv):
        lam1 = 1.0 / np.tile(np.full(K, 0.3), 2) ** 2
        Mwf = (M * rw[:, None]) * cscale[None, :]
        A1 = Mwf.T @ Mwf + np.diag(lam1)
        z1 = np.linalg.solve(A1, Mwf.T @ (dv * rw) + lam1 * a0)
        a0c = a0.copy()
        for i in (0, 1):
            s = (z1[i * K:i * K + 7].sum()
                 / a0[i * K:i * K + 7].sum())
            a0c[i * K + 7:(i + 1) * K] *= s
        lamc = 1.0 / np.tile(
            np.array([0.3] * 7 + [0.05] * 3), 2) ** 2
        Ac = Mwf.T @ Mwf + np.diag(lamc)
        return np.linalg.solve(
            Ac, Mwf.T @ (dv * rw) + lamc * a0c) * cscale

    for nm, taus in list(VARIANTS) + [('core', None)]:
        if nm == 'core':
            amps0 = core_solve(dvec)
            tots = {0: [], 1: []}
            for r in range(100):
                nim = obs.image + rng.normal(
                    scale=sigma_n, size=obs.image.shape)
                nobs = ngmix.Observation(
                    nim, jacobian=obs.jacobian,
                    weight=np.ones_like(nim) / sigma_n ** 2,
                    psf=obs.psf)
                epn = _prep_epochs(
                    get_mb_obs(nobs), fwhm_smooth=fwhm,
                    ap_rad=0.0, use_noise_image=False,
                    vcen=0.0, ucen=0.0)[0]
                _, dv, _ = hp.build_joint(
                    epn, us, rungs, aps, sig_rows)
                av = core_solve(dv)
                for i in (0, 1):
                    tots[i].append(av[i * K:(i + 1) * K].sum()
                                   / Ftrue[i] - 1.0)
            a1 = amps0[:K].sum() / Ftrue[0] - 1
            a2 = amps0[K:].sum() / Ftrue[1] - 1
            t1, t2 = np.array(tots[0]), np.array(tots[1])
            print(f"  {nm:6s} noiseless attr: bright {a1:+7.3f}"
                  f" faint {a2:+7.3f}   noisy: bright"
                  f" {t1.mean():+7.3f} +-{t1.std():6.3f}"
                  f"  faint {t2.mean():+7.3f} +-{t2.std():6.3f}")
            continue
        lam = 1.0 / np.tile(taus, 2) ** 2
        Mwf = (M * rw[:, None]) * cscale[None, :]
        A = Mwf.T @ Mwf + np.diag(lam)
        z0 = np.linalg.solve(A, Mwf.T @ (dvec * rw) + lam * a0)
        amps0 = z0 * cscale
        tots = {0: [], 1: []}
        for r in range(100):
            nim = obs.image + rng.normal(scale=sigma_n,
                                         size=obs.image.shape)
            nobs = ngmix.Observation(
                nim, jacobian=obs.jacobian,
                weight=np.ones_like(nim) / sigma_n ** 2,
                psf=obs.psf)
            epn = _prep_epochs(
                get_mb_obs(nobs), fwhm_smooth=fwhm, ap_rad=0.0,
                use_noise_image=False, vcen=0.0, ucen=0.0)[0]
            _, dv, _ = hp.build_joint(epn, us, rungs, aps, sig_rows)
            z = np.linalg.solve(A, Mwf.T @ (dv * rw) + lam * a0)
            av = z * cscale
            for i in (0, 1):
                tots[i].append(av[i * K:(i + 1) * K].sum()
                               / Ftrue[i] - 1.0)
        a1 = amps0[:K].sum() / Ftrue[0] - 1
        a2 = amps0[K:].sum() / Ftrue[1] - 1
        t1, t2 = np.array(tots[0]), np.array(tots[1])
        print(f"  {nm:6s} noiseless attr: bright {a1:+7.3f}"
              f" faint {a2:+7.3f}   noisy: bright {t1.mean():+7.3f}"
              f" +-{t1.std():6.3f}  faint {t2.mean():+7.3f}"
              f" +-{t2.std():6.3f}")


def main():
    global NREAL
    if len(sys.argv) > 1:
        NREAL = int(sys.argv[1])
    part_a()
    part_b(np.random.default_rng(313))


if __name__ == '__main__':
    main()
