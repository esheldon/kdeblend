"""
Heterogeneous-pair flux stealing study for the ladder.

The symmetric pair study probes degeneracy but hides stealing:
symmetric errors cancel antisymmetrically.  Here the pairs are
heterogeneous -- the dangerous cases being a compact object
sitting on a wing-heavy neighbor's profile (does the compact
object's ladder, whose outer rungs reach far, soak up the
neighbor's wing flux?) and large flux ratios.

Per config and separation, both objects get ladders and the
scene-wide joint amplitude solve (fraction units, noise-weighted
rows, tau0=1 priors toward each object's exp pair-fit profile).
The stealing metric is the production one: the member's
neighbor-corrected flux sum under its own adaptive weight,

    bias_i = (pair sum_i - neighbor model sum_i) / iso sum_i - 1

where iso sum_i is the same measurement on the member rendered
alone (same weight), so the bias isolates the neighbor
subtraction error.  Reported for the ladder and for the exp
pair fit (the production baseline), noiseless, plus the noisy
mean/std at bright-object s2n=100 (frames fixed from the
noiseless gauss pair fit; exp refit per realization; ladder
amps re-solved per realization).

attr_i = sum(amps_i)/F_true_i - 1 is also reported: the model
total flux attribution, the harshest (wing-dominated) metric.

Run: python experiments/ladder_hetero_pairs.py [nreal]
"""
import sys
import os
import numpy as np
import ngmix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import ladder_contamination as lc  # noqa: E402
import ladder_pair_stability as ps  # noqa: E402
from _sims import make_blend_obs  # noqa: E402
from ngmix.prepsfadmom.models import model_gauss_components  # noqa: E402
from kdeblend.deblender import build_deblender, _prep_epochs  # noqa: E402
from ngmix.observation import get_mb_obs  # noqa: E402

RUNGS = lc.RUNGS
AP_FACS = lc.AP_FACS
K = RUNGS.size
PSF_FWHM = 0.8
DIM = 128
SEPS = [2.0, 1.0, 0.75]
TAU0 = 1.0
S2N_BRIGHT = 100.0
NREAL = 100
NNOISE = 100
SIGMA_CAL = 1.0e-3
SEED = 777

CONFIGS = [
    ('wings-on-compact 10:1',
     dict(n=4.0, hlr=0.8, flux=1.0), dict(n=1.0, hlr=0.3, flux=0.1)),
    ('wings-on-compact 100:1',
     dict(n=4.0, hlr=0.8, flux=1.0), dict(n=1.0, hlr=0.3, flux=0.01)),
    ('diffuse vs compact',
     dict(n=1.0, hlr=1.0, flux=1.0), dict(n=4.0, hlr=0.3, flux=0.3)),
    ('equal-flux hetero',
     dict(n=4.0, hlr=0.8, flux=1.0), dict(n=1.0, hlr=0.3, flux=1.0)),
]


def sersic(c, u):
    return dict(kind='sersic', n=c['n'], hlr=c['hlr'],
                flux=c['flux'], e1=0.0, e2=0.0, v=0.0, u=u)


def fit_pair(obs, t, d, Tg=(0.6, 0.6), epochs=None, fwhm=None,
             maxiter=500):
    objs = [dict(v=0.0, u=-d / 2, type=t, Tguess=Tg[0]),
            dict(v=0.0, u=+d / 2, type=t, Tguess=Tg[1])]
    deb, _ = build_deblender(obs, objs, maxiter=maxiter,
                             epochs=epochs, fwhm_smooth=fwhm)
    return deb, deb.go()


def build_joint(ep, us, rungs, aps, sig_rows):
    nob = len(us)
    nap = len(aps[0])
    M = np.zeros((nob * nap, nob * K))
    dvec = np.zeros(nob * nap)
    for i in range(nob):
        for j, w in enumerate(aps[i]):
            r = i * nap + j
            dvec[r] = lc.measure_sums(ep, 0.0, us[i], w)[5]
            for ip in range(nob):
                S00, S01, S11 = rungs[ip]
                for k in range(K):
                    M[r, ip * K + k] = lc.comp_flux_sum(
                        1.0, S00[k], S01[k], S11[k],
                        0.0, us[ip] - us[i], w)
    rw = np.concatenate([1.0 / s for s in sig_rows])
    return M, dvec, rw


def solve_amps(M, dvec, rw, Fe, a0, lam0):
    cscale = np.repeat(Fe, K)
    Mwf = (M * rw[:, None]) * cscale[None, :]
    A = Mwf.T @ Mwf + lam0 * np.eye(2 * K)
    z = np.linalg.solve(A, Mwf.T @ (dvec * rw) + lam0 * a0)
    return z * cscale


def nbr_sum(amps_j, rungs_j, du, w):
    S00, S01, S11 = rungs_j
    return lc.comp_flux_sum(amps_j, S00, S01, S11, 0.0, du, w)


def run_cell(name, c1, c2, d, rng):
    obs = make_blend_obs(
        [sersic(c1, -d / 2), sersic(c2, +d / 2)], PSF_FWHM, dim=DIM)
    iso_obs = [
        make_blend_obs([sersic(c1, -d / 2)], PSF_FWHM, dim=DIM),
        make_blend_obs([sersic(c2, +d / 2)], PSF_FWHM, dim=DIM),
    ]
    us = [-d / 2, d / 2]
    Ftrue = [c1['flux'], c2['flux']]

    deb_g, res_g = fit_pair(obs, 'gauss', d)
    deb_e, res_e = fit_pair(obs, 'exp', d)
    Tsmooth = deb_g.Tsmooth
    ep = deb_g.epochs_per_obj[0][0]
    fwhm = deb_g.fwhm_smooth
    Sw = [np.asarray(deb_g.Sw[i]) for i in (0, 1)]
    rungs = [ps.frame_of(Sw[i], Tsmooth) for i in (0, 1)]
    aps = [[af * Sw[i] for af in AP_FACS] for i in (0, 1)]

    eps_iso = [
        _prep_epochs(get_mb_obs(o), fwhm_smooth=fwhm, ap_rad=0.0,
                     use_noise_image=False, vcen=0.0, ucen=0.0)[0]
        for o in iso_obs
    ]
    s_iso = [lc.measure_sums(eps_iso[i], 0.0, us[i], Sw[i])[5]
             for i in (0, 1)]

    # bright-object s2n calibration
    debc, _ = build_deblender(
        ngmix.Observation(
            iso_obs[0].image, jacobian=iso_obs[0].jacobian,
            weight=np.ones_like(iso_obs[0].image) / SIGMA_CAL ** 2,
            psf=iso_obs[0].psf),
        [dict(v=0.0, u=us[0], type='gauss', Tguess=0.6)],
        maxiter=100)
    rc = debc.go()
    sigma_n = SIGMA_CAL * rc['objects'][0]['s2n'] / S2N_BRIGHT

    # unit-noise aperture sigmas, both objects' sets
    ap_sums = np.zeros((NNOISE, 2, len(AP_FACS)))
    for i in range(NNOISE):
        nobs = ngmix.Observation(
            rng.normal(size=obs.image.shape), jacobian=obs.jacobian,
            weight=np.ones_like(obs.image), psf=obs.psf)
        epn = _prep_epochs(
            get_mb_obs(nobs), fwhm_smooth=fwhm, ap_rad=0.0,
            use_noise_image=False, vcen=0.0, ucen=0.0)[0]
        for o in (0, 1):
            for j, w in enumerate(aps[o]):
                ap_sums[i, o, j] = lc.measure_sums(
                    epn, 0.0, us[o], w)[5]
    sig_rows = [ap_sums[:, o, :].std(axis=0) * sigma_n
                for o in (0, 1)]

    lam0 = 1.0 / TAU0 ** 2

    def exp_pack(deb):
        mods, Fe, a0 = [], np.zeros(2), np.zeros(2 * K)
        M0, _, _ = build_joint(ep, us, rungs, aps, sig_rows)
        for i in (0, 1):
            fr, S00, S01, S11 = model_gauss_components(deb.models[i], Tsmooth)
            F = deb.models[i]['F'][0]
            mods.append((F * fr, S00, S01, S11))
            Fe[i] = F if (np.isfinite(F) and F > 0) else 1e-6
            nap = len(aps[i])
            d0 = np.array([
                lc.comp_flux_sum(fr, S00, S01, S11, 0.0, 0.0, w)
                for w in aps[i]])
            Mo = M0[i * nap:(i + 1) * nap, i * K:(i + 1) * K]
            MtM = Mo.T @ Mo
            a0[i * K:(i + 1) * K] = np.linalg.solve(
                MtM + 1e-8 * np.trace(MtM) / K * np.eye(K),
                Mo.T @ d0)
        return mods, Fe, a0

    exp_mods, Fe, a0 = exp_pack(deb_e)
    M, dvec, rw = build_joint(ep, us, rungs, aps, sig_rows)
    amps = solve_amps(M, dvec, rw, Fe, a0, lam0)

    def biases(ep_data, amps_v, emods):
        out = {}
        for i in (0, 1):
            j = 1 - i
            sp = lc.measure_sums(ep_data, 0.0, us[i], Sw[i])[5]
            nl = nbr_sum(amps_v[j * K:(j + 1) * K], rungs[j],
                         us[j] - us[i], Sw[i])
            Fj, S00, S01, S11 = emods[j]
            ne = lc.comp_flux_sum(Fj, S00, S01, S11,
                                  0.0, us[j] - us[i], Sw[i])
            out[i] = ((sp - nl) / s_iso[i] - 1.0,
                      (sp - ne) / s_iso[i] - 1.0)
        return out

    b0 = biases(ep, amps, exp_mods)
    attr = [amps[i * K:(i + 1) * K].sum() / Ftrue[i] - 1.0
            for i in (0, 1)]

    # noisy loop
    rows = {i: {'lad': [], 'exp': []} for i in (0, 1)}
    nconv_e = 0
    for r in range(NREAL):
        nim = obs.image + rng.normal(scale=sigma_n,
                                     size=obs.image.shape)
        nobs = ngmix.Observation(
            nim, jacobian=obs.jacobian,
            weight=np.ones_like(nim) / sigma_n ** 2, psf=obs.psf)
        epn = _prep_epochs(
            get_mb_obs(nobs), fwhm_smooth=fwhm, ap_rad=0.0,
            use_noise_image=False, vcen=0.0, ucen=0.0)
        try:
            deb_en, res_en = fit_pair(
                nobs, 'exp', d, epochs=epn, fwhm=fwhm, maxiter=100)
        except Exception:
            continue
        nconv_e += res_en['converged']
        emods_n, Fe_n, a0_n = exp_pack(deb_en)
        _, dv_n, _ = build_joint(epn[0], us, rungs, aps, sig_rows)
        amps_n = solve_amps(M, dv_n, rw, Fe_n, a0_n, lam0)
        bn = biases(epn[0], amps_n, emods_n)
        for i in (0, 1):
            rows[i]['lad'].append(bn[i][0])
            rows[i]['exp'].append(bn[i][1])

    print(f"\n===== {name}  d={d:.2f} =====")
    print(f"  gauss conv={res_g['converged']} niter={res_g['numiter']}"
          f" | exp conv={res_e['converged']} niter={res_e['numiter']}"
          f" | noisy exp conv {nconv_e}/{NREAL}")
    for i, tag in ((0, 'bright'), (1, 'second')):
        ls = np.array(rows[i]['lad'])
        es = np.array(rows[i]['exp'])
        print(f"  {tag:6s} F={Ftrue[i]:5.2f}: noiseless bias"
              f" lad {b0[i][0]:+9.2e}  exp {b0[i][1]:+9.2e}"
              f"  attr {attr[i]:+8.3f}")
        print(f"          noisy: lad {ls.mean():+9.2e} +-"
              f" {ls.std():8.2e}   exp {es.mean():+9.2e} +-"
              f" {es.std():8.2e}")


def main():
    global NREAL
    if len(sys.argv) > 1:
        NREAL = int(sys.argv[1])
    rng = np.random.default_rng(SEED)
    for name, c1, c2 in CONFIGS:
        for d in SEPS:
            run_cell(name, c1, c2, d, rng)


if __name__ == '__main__':
    main()
