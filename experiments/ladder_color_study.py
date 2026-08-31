"""
Two-band color study: how much color bias does a bright
neighbor's subtraction residual inject into a faint neighbor,
per model type?  Requirement 1 of the lensing case.

The faint object's fractional flux bias in band b is
c_b x R_b, where c_b is the per-band leakage per unit same-band
flux ratio (the quantity tabulated here, per unit bright flux)
and R_b the bright/faint flux ratio in that band.  The color
bias is then

    d(g-r) = -1.086 (c_g R_g - c_r R_r) mag

so color bias comes from two channels: any nonzero leakage c
paired with different SEDs (R_g != R_r), and band-dependent
leakage (c_g != c_r) from a color gradient in the bright
object, which no shared-structure model can remove.

Bright object: bulge+disk composite (disk Sersic n=1 hlr=1.0,
bulge n=4 hlr=0.35), per-band component fluxes setting the
gradient, total flux 1 in each band.  Configs:

- control: same profile both bands, different PSFs.  In the
  common pre-seeing space the leakage should be band-independent
  (c_g = c_r) for every model, mismatched or not: aperture
  colors stay clean.
- gradient, same PSF: the irreducible shared-structure floor.
- gradient + different PSFs.

Models: gauss/exp/bdf fit with the deblender (shared structure,
per-band fluxes); the ladder with shared amplitudes and
per-band fluxes (alternating bilinear solve, mirroring the
deblender's flux/structure split); and a per-band-amplitude
ladder, the noiseless upper bound on what v2 per-band profile
freedom could buy.  Noiseless: color bias is a systematic.
Templates and apertures are band-independent by construction
(the pre-seeing space), which is what makes the shared solve
consistent across bands.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import ladder_contamination as lc  # noqa: E402
from _sims import make_blend_mbobs  # noqa: E402
from ngmix.prepsfadmom.models import model_comps  # noqa: E402
from kdeblend.deblender import build_deblender  # noqa: E402

T_FAINT = lc.T_FAINT
RUNGS = lc.RUNGS
AP_FACS = lc.AP_FACS
SEPS = [1.0, 2.0, 4.0]
DIM = 128
# fiducial bright/faint flux ratios for the worked color bias:
# the bright neighbor is 2x redder than the faint object
R_G, R_R = 30.0, 60.0

DISK = dict(kind='sersic', n=1.0, hlr=1.0, e1=0.0, e2=0.0,
            v=0.0, u=0.0)
BULGE = dict(kind='sersic', n=4.0, hlr=0.35, e1=0.0, e2=0.0,
             v=0.0, u=0.0)


def band_comps(fdisk, fbulge):
    return [dict(DISK, flux=fdisk), dict(BULGE, flux=fbulge)]


CONFIGS = [
    ('control: same profile, psf 0.9/0.75',
     [band_comps(0.55, 0.45), band_comps(0.55, 0.45)], [0.9, 0.75]),
    ('gradient: bulge frac 0.3/0.6, psf 0.8/0.8',
     [band_comps(0.7, 0.3), band_comps(0.4, 0.6)], [0.8, 0.8]),
    ('gradient + psf 0.9/0.75',
     [band_comps(0.7, 0.3), band_comps(0.4, 0.6)], [0.9, 0.75]),
]


def fit_type(mbobs, t):
    o = dict(v=0.0, u=0.0, type=t, Tguess=1.0)
    if t == 'bdf':
        o['TdByTe'] = 1.0
    deb, _ = build_deblender(mbobs, [o])
    res = deb.go()
    return deb, res


def ladder_solves(eps_by_band, apertures, M):
    """shared-amp (alternating) and per-band ladder solves;
    returns (F, abar) and per-band amp vectors"""
    K = RUNGS.size
    MtM = M.T @ M
    eps_r = 1.0e-8 * np.trace(MtM) / K
    dvecs = [
        np.array([lc.measure_sums(ep, 0.0, 0.0, w)[5]
                  for w in apertures])
        for ep in eps_by_band
    ]
    nband = len(dvecs)

    # per-band free amps
    a_pb = [
        np.linalg.solve(MtM + eps_r * np.eye(K), M.T @ d)
        for d in dvecs
    ]

    # shared amps, per-band fluxes: bilinear alternation
    F = np.ones(nband)
    abar = np.zeros(K)
    for it in range(20):
        A = (F ** 2).sum() * MtM
        rhs = M.T @ sum(F[b] * dvecs[b] for b in range(nband))
        abar_new = np.linalg.solve(A + eps_r * np.eye(K), rhs)
        mod = M @ abar_new
        F_new = np.array([
            (mod @ d) / (mod @ mod) for d in dvecs
        ])
        # fix the bilinear scale gauge
        s = abar_new.sum()
        abar_new = abar_new / s
        F_new = F_new * s
        dmax = np.abs(abar_new - abar).max()
        abar, F = abar_new, F_new
        if dmax < 1.0e-12:
            break
    return (F, abar), a_pb


def run_config(name, comps_per_band, psfs):
    mbobs = make_blend_mbobs(comps_per_band, psfs, dim=DIM)

    fits = {}
    notes = []
    deb0 = None
    for t in ('gauss', 'exp', 'bdf'):
        deb, res = fit_type(mbobs, t)
        if deb0 is None:
            deb0 = deb
        fits[t] = deb
        notes.append(f"{t} conv={res['converged']}"
                     f" niter={res['numiter']}")

    Tsmooth = deb0.Tsmooth
    eps_by_band = deb0.epochs_per_obj[0]
    assert [ep['band'] for ep in eps_by_band] == [0, 1]
    Sw = deb0.Sw[0].copy()
    sm = Tsmooth / 2
    Sbase = Sw - np.diag([sm, sm])
    lS00 = RUNGS * Sbase[0, 0] + sm
    lS01 = RUNGS * Sbase[0, 1]
    lS11 = RUNGS * Sbase[1, 1] + sm
    apertures = [af * Sw for af in AP_FACS]
    M = np.zeros((len(apertures), RUNGS.size))
    for j, w in enumerate(apertures):
        for k in range(RUNGS.size):
            M[j, k] = lc.comp_flux_sum(
                1.0, lS00[k], lS01[k], lS11[k], 0.0, 0.0, w)

    (Fl, abar), a_pb = ladder_solves(eps_by_band, apertures, M)

    # per-band model components for every model
    models = {}
    for t in ('gauss', 'exp', 'bdf'):
        fr, S00, S01, S11 = model_comps(fits[t].models[0], Tsmooth)
        Fb = fits[t].models[0]['F']
        models[t] = [(Fb[b] * fr, S00, S01, S11) for b in (0, 1)]
    models['ladder'] = [
        (Fl[b] * abar, lS00, lS01, lS11) for b in (0, 1)
    ]
    models['lad-pb'] = [
        (a_pb[b], lS00, lS01, lS11) for b in (0, 1)
    ]
    models['none'] = [
        (np.zeros(1), np.ones(1), np.zeros(1), np.ones(1))
    ] * 2

    Twf = T_FAINT / 2 + sm
    wf = np.diag([Twf, Twf])
    s_self = lc.comp_flux_sum(1.0, Twf, 0.0, Twf, 0.0, 0.0, wf)

    print(f"\n===== {name} =====")
    print("  " + "  ".join(notes)
          + f"  Tsmooth={Tsmooth:.3f}"
          + f"  ladder F=[{Fl[0]:.3f} {Fl[1]:.3f}]")
    print(f"  leakage per unit same-band flux ratio c_g, c_r; "
          f"dc = c_g - c_r;")
    print(f"  dmag = faint (g-r) bias for R_g={R_G:g}, R_r={R_R:g}"
          f" (bright 2x redder)")
    order = ['none', 'gauss', 'exp', 'bdf', 'ladder', 'lad-pb']
    for d in SEPS:
        tsums = [
            lc.measure_sums(eps_by_band[b], d, 0.0, wf)[5]
            for b in (0, 1)
        ]
        print(f"  d={d:.1f}")
        for m in order:
            cs = []
            for b in (0, 1):
                F, S00, S01, S11 = models[m][b]
                mod = lc.comp_flux_sum(F, S00, S01, S11,
                                       -d, 0.0, wf)
                cs.append((tsums[b] - mod) / s_self)
            dmag = -1.086 * (cs[0] * R_G - cs[1] * R_R)
            print(f"    {m:7s} c_g {cs[0]:+9.2e}  c_r {cs[1]:+9.2e}"
                  f"  dc {cs[0] - cs[1]:+9.2e}"
                  f"  dmag {dmag:+8.4f}")


def main():
    np.set_printoptions(linewidth=200)
    for name, cpb, psfs in CONFIGS:
        run_config(name, cpb, psfs)


if __name__ == '__main__':
    main()
