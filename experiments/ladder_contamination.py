"""
Contamination experiment: how much of a bright Sersic neighbor's
light leaks into a faint neighbor's flux aperture after
subtracting each bright-object model type?

Motivation (README): unmodeled wings of a single-gaussian bright
neighbor contaminate faint neighbors at the tens-of-percent
level, and the exp model reduces that by 1-2 orders of
magnitude.  Those numbers were measured with gauss/exp/dev
truths, where 'exp' can be exact.  Real bright galaxies are not
exp; this measures the residual leakage on Sersic truths for the
current types (gauss/exp/dev/bdf) and for the proposed
free-amplitude concentric gaussian ladder, fit by the K-aperture
linear solve (the generalized bdf split).

Method, noiseless (pure model-mismatch systematics):

- render the bright Sersic alone (tests/_sims.py 'sersic' kind)
  and prep it into the standard pre-seeing smoothed k-space by
  building a deblender on it
- fit gauss/exp/dev/bdf with the actual machinery
  (single-object group, fixed center, adaptive moments)
- fit the ladder: measure the data's flux sums under K gaussian
  apertures (admom_ksums), build the closed-form template
  matrix (gauss_comps_ksums), solve the ridge least squares.
  The rung sizes are fixed multiples of the converged gauss
  pre-smoothing covariance; the ellipticity is inherited from
  the same fit, so only the radial profile is free
- contamination at separation d: the flux sum of (truth -
  model) under a faint object's gaussian weight centered at d;
  the truth side is measured from the prepped image with
  admom_ksums, the model side is closed form.  Dividing by the
  faint object's self flux sum s_self gives the fractional flux
  bias a unit-flux-ratio faint object would suffer; multiply by
  the actual bright/faint flux ratio for a real case.  The
  'none' column (no subtraction at all) sets the scale of what
  the models are removing.

Conventions (deblender._get_object_sums): data sums from
admom_ksums are multiplied by detAtinv to be comparable to
gauss_comps_ksums model sums evaluated at detAtinv=1; smoothed
component covariances are cT * Sfam + (Tsmooth / 2) I.

Caveats: this is the subtraction-quality proxy, not a full
deblend (the bright fit is not perturbed by the faint object;
second order for large flux ratios).  The stamp truncates the
Sersic wings at the box edge (as real stamps do), which mostly
affects the largest fitting apertures.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from _sims import make_blend_obs, SCALE  # noqa: E402

from ngmix.prepsfadmom import get_phase_angles  # noqa: E402
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums  # noqa: E402
from ngmix.prepsfadmom.models import model_gauss_components  # noqa: E402
from ngmix.prepsfadmom.models_nb import gauss_comps_ksums  # noqa: E402
from kdeblend.deblender import build_deblender  # noqa: E402

# --- configuration ---------------------------------------------------

PSF_FWHM = 0.8
DIM = 128                       # 32 arcsec box at SCALE=0.25
HLR = 0.8                       # bright object half light radius
NS = [1.0, 2.0, 3.0, 4.0]       # Sersic indices (n=1 is the exp sanity check)
SEPS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # arcsec
T_FAINT = 0.2                   # pre-psf T of the fiducial faint object

# ladder rungs: fixed multiples of the gauss-fit pre-smoothing
# covariance, geometric ratio 2
RUNGS = 0.05 * 2.0 ** np.arange(10)      # 0.05 ... 25.6
# fitting apertures: multiples of the converged gauss weight
AP_FACS = np.geomspace(0.25, 32.0, 16)
LAM_REL = 1.0e-8                # ridge, relative to trace(M^T M)/K

TYPES = ['gauss', 'exp', 'dev', 'bdf']


# --- sum helpers -----------------------------------------------------

def measure_sums(ep, v, u, w):
    """data moment sums under weight w centered at (v, u), scaled
    to be comparable to gauss_comps_ksums at detAtinv=1"""
    alpha, beta = get_phase_angles(ep, v - ep['vcen'], u - ep['ucen'])
    sums = np.zeros(6)
    admom_ksums(
        ep['kim'], ep['iy'], ep['ix'], ep['dim'], alpha, beta,
        ep['kv'], ep['ku'], w[0, 0], w[0, 1], w[1, 1], ep['df2'],
        sums,
    )
    return sums * ep['detAtinv']


def comp_flux_sum(F, So00, So01, So11, dv, du, w):
    """flux sum of gaussian components offset (dv, du) from the
    center of weight w"""
    n = np.size(F)
    sums = np.zeros(6)
    gauss_comps_ksums(
        np.atleast_1d(np.asarray(F, dtype='f8')),
        np.atleast_1d(np.asarray(So00, dtype='f8')),
        np.atleast_1d(np.asarray(So01, dtype='f8')),
        np.atleast_1d(np.asarray(So11, dtype='f8')),
        np.full(n, dv), np.full(n, du),
        w[0, 0], w[0, 1], w[1, 1], 1.0,
        sums,
    )
    return sums[5]


# --- fitting ---------------------------------------------------------

def fit_type(obs, t, Tguess=1.0):
    """fit a single object of the given type with the deblender"""
    o = dict(v=0.0, u=0.0, type=t, Tguess=Tguess)
    if t == 'bdf':
        o['TdByTe'] = 1.0
    deb, _ = build_deblender(obs, [o])
    res = deb.go()
    return deb, res


def fit_ladder(ep, Sbase, Sw, Tsmooth, rungs=RUNGS, ap_facs=AP_FACS,
               lam_rel=LAM_REL):
    """
    free-amplitude concentric ladder fit by the K-aperture linear
    solve.  Sbase is the pre-smoothing base covariance (sets the
    rung sizes and the shared ellipticity), Sw the aperture base.
    Rows are scaled by 1/|d_j| so every aperture counts equally
    in fractional terms.  Returns (amps, rung covs, rel residual)
    """
    sm = Tsmooth / 2
    So00 = rungs * Sbase[0, 0] + sm
    So01 = rungs * Sbase[0, 1]
    So11 = rungs * Sbase[1, 1] + sm

    nap = len(ap_facs)
    K = rungs.size
    d = np.zeros(nap)
    M = np.zeros((nap, K))
    for j, af in enumerate(ap_facs):
        w = af * Sw
        d[j] = measure_sums(ep, 0.0, 0.0, w)[5]
        for k in range(K):
            M[j, k] = comp_flux_sum(
                1.0, So00[k], So01[k], So11[k], 0.0, 0.0, w,
            )

    # relative row weighting
    rw = 1.0 / np.abs(d)
    Mw = M * rw[:, None]
    dw = d * rw

    MtM = Mw.T @ Mw
    lam = lam_rel * np.trace(MtM) / K
    amps = np.linalg.solve(MtM + lam * np.eye(K), Mw.T @ dw)
    resid = np.sqrt(np.mean((Mw @ amps - dw) ** 2))
    return amps, (So00, So01, So11), resid


# --- the experiment --------------------------------------------------

def run_one(n):
    comp = dict(
        kind='sersic', n=n, hlr=HLR, flux=1.0,
        e1=0.0, e2=0.0, v=0.0, u=0.0,
    )
    obs = make_blend_obs([comp], PSF_FWHM, dim=DIM)

    # fits with the production machinery
    models = {}
    notes = {}
    deb0 = None
    for t in TYPES:
        deb, res = fit_type(obs, t)
        if deb0 is None:
            deb0 = deb  # gauss first: epochs, Tsmooth, Sw base
        o = res['objects'][0]
        conv = res['converged'] and o['deblend_flags'] == 0
        fracs, So00, So01, So11 = model_gauss_components(deb.models[0], deb.Tsmooth)
        F = deb.models[0]['F'][0]
        models[t] = (F * fracs, So00, So01, So11)
        notes[t] = (
            f"converged={conv} niter={res['numiter']}"
            + (f" fracdev={deb.models[0]['fracdev']:.3f}" if t == 'bdf'
               else "")
        )

    ep = deb0.epochs_per_obj[0][0]
    Tsmooth = deb0.Tsmooth
    Sw = deb0.Sw[0].copy()                     # converged gauss weight
    Sbase = Sw - np.diag([Tsmooth / 2] * 2)    # pre-smoothing base
    amps, (lS00, lS01, lS11), resid = fit_ladder(ep, Sbase, Sw, Tsmooth)
    models['ladder'] = (amps, lS00, lS01, lS11)
    notes['ladder'] = (
        f"rel-resid={resid:.2e} amps="
        + np.array2string(amps, precision=3, floatmode='maxprec')
    )
    models['none'] = (np.zeros(1), np.ones(1), np.zeros(1), np.ones(1))

    # faint-object weight and self flux sum
    Twf = T_FAINT / 2 + Tsmooth / 2
    wf = np.diag([Twf, Twf])
    s_self = comp_flux_sum(1.0, Twf, 0.0, Twf, 0.0, 0.0, wf)

    names = TYPES + ['ladder', 'none']
    print(f"\n===== sersic n={n:.1f}  hlr={HLR}  psf_fwhm={PSF_FWHM}"
          f"  Tsmooth={Tsmooth:.3f}  T_gauss={Sbase.trace():.3f} =====")
    for t in names[:-1]:
        print(f"  {t:7s} {notes[t]}")
    print(f"\n  faint-object fractional flux bias per unit flux ratio"
          f"  (x bright/faint ratio for a real case)")
    print("  " + " ".join([f"{'d[asec]':>8s}"] + [
        f"{t:>10s}" for t in names
    ]))
    for dsep in SEPS:
        tflux = measure_sums(ep, dsep, 0.0, wf)[5]
        row = [f"{dsep:8.1f}"]
        for t in names:
            F, S00, S01, S11 = models[t]
            m = comp_flux_sum(F, S00, S01, S11, -dsep, 0.0, wf)
            row.append(f"{(tflux - m) / s_self:10.2e}")
        print("  " + " ".join(row))


def main():
    np.set_printoptions(linewidth=200)
    for n in NS:
        run_one(n)


if __name__ == '__main__':
    main()
