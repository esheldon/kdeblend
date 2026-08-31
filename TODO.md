# TODO

- Integrate the 'ladder' model type: the free-amplitude
  concentric gaussian ladder, to replace exp/dev/bdf once
  validated at field scale (evidence in the DONE 2026-08-31
  entry and experiments/).  Locked design choices: per-band
  amplitude vectors in flux units with a cross-band prior
  (taux ~ 0.1-0.3 in fraction units) and a noise-scaled prior
  toward the exp profile (tau0 ~ 0.3-1); the frame from the
  object's own data-driven adaptive weight (gauss-style
  deweight, no model moment matching -- the bdf size/profile
  feedback is cut structurally); a scene-wide joint amp solve
  every few sweeps (per-object amp iteration contracts at
  rho -> 1 for close pairs and is excluded); per-sweep flux
  rescaling of the amp vectors between solves.  Full errors:
  the amps join the state vector, the solve matrix is the
  closed-form response.  sum(amps) is never reported; totals
  come from the tau-dial (free core, prior-completed wings).
  Re-measure tight-blend totals after integration: the
  standalone frames came from gauss-corrected fits and swell
  under bright wings.
  Slice 1 landed (2026-08-31, branch ladder): the 'ladder'
  type in deblender.py + kdeblend/ladder.py -- gauss-path
  weight update, scene-wide amp solve every 2 sweeps (4
  defeats the rho estimate and destabilizes extrapolation; 1
  limit-cycles: the one-sweep lag is load-bearing), amps kept
  OUT of the packed state (extrapolating them injects noise
  the solve undoes) and the solve change metric in observable
  row space (raw-amp changes carry prior-dominated degenerate
  swings that never converge).  Gate-1 tests green
  (tests/test_ladder.py, 7), no regressions (78).  Cost 45-57
  ms/fit vs exp 11-21 after caching the aperture sigmas;
  remaining lever is the fused J-aperture ksums kernel.
  Consistency rows (2026-08-31): the T row under the object's
  own weight is in (soft, noise-weighted, admom_finalize
  variances cached with the flux-row sigmas):
  deweight(model sums) matches Sw in T to < 1e-4 (round and
  elliptical, vs 1e-4..4e-4 without).  The M1/M2 rows are
  excluded: the rungs share the frame's gaussian-deweighted
  ellipticity (0.079 for a true 0.10, sersic n=3), so they are
  unsatisfiable and forcing them corrupts the flux/T rows (T
  residual 4e-5 -> 2e-4, flux 7e-6 -> 1e-3).  Ellipticity
  consistency needs a frame-ellipticity degree of freedom (a
  two-parameter nonlinear correction of the rung frame, kept
  off the weight update); open, low priority: the reported
  shapes are the data-driven gauss estimator, and the
  inconsistency only enters neighbors' shape channels at the
  1e-3 x leakage level.
  Remaining slices: ladder-aware full errors (currently
  per-object via the gauss proxy; ladder groups return None
  from apply_full_errors), derived flux functionals,
  fixed-external and gpu support, field-scale validation.

DONE (2026-08-31): ladder experimental phase (branch ladder,
experiments/*.py, seven studies).  Findings and decisions:

- Contamination (sersic truths, noiseless): exp removes only
  30-55 percent of a bright neighbor's wing leakage at 1.5-2
  arcsec and none beyond 3; the K-aperture ladder solve is
  30-300x better than exp and ~10x better than bdf at all
  separations, and the win survives a ridge that makes the
  amplitude vector positive (monotone).  dev hits the
  500-sweep limit cycle on every default-box run; bdf is
  unconverged at n=4.

- Noise: the faint-neighbor corrected-sum noise inflation is
  <= exp everywhere (flux and M1 rows), insensitive to the
  prior width; shared-pixel correlation makes the subtraction
  *reduce* the noise below the no-neighbor floor (x ~ 0.7-0.8
  at 1 arcsec).  Ladder iterations are the gauss weight loop
  (median 11-14 sweeps vs exp 17-19); standalone cost 0.85x
  the exp path; an amp update is ~6 ms of aperture passes,
  amortizable.

- Colors: the pre-seeing control is exact (per-band leakage
  equal to 1e-9 under different psfs, for every model).  A
  color gradient puts a shared-structure floor dc ~ 1.5-2e-2
  on every shared model including the shared-amp ladder;
  per-band amplitudes with a cross-band prior remove it
  (dc < 1e-3 at s2n >= 30) with color-noise inflation <= 1
  everywhere, even fully free at s2n=10.  DECISION: v1 has
  per-band amps with the cross-band prior.

- Close pairs: the joint amp system's conditioning with the
  standard prior is flat in separation (1.9e5 -> 5.2e5 from
  d=3 to 0.5 arcsec; raw 1e16-1e17 at every d, the rung
  collinearity, not the pair); per-object amp Gauss-Seidel
  needs ~360 -> 13000 sweeps (SPD: slow, never divergent), so
  the scene-wide one-step solve is the design.  Equal-pair
  flux attribution +0.1-0.2 percent down to 0.75 arcsec,
  -+4 percent at 0.5.

- Flux stealing is a wings phenomenon: wings-onto-compact
  leaves +25 -> +83 percent of the faint flux with exp (10:1,
  d=2 -> 0.75 arcsec), whose noisy pair fit also converges
  only 24-56/100; the ladder residual is ~10x smaller and its
  gauss frames converged in all 1200 noisy fits.  The
  ladder's own sum(amps) soaks up neighbor wings (+145
  percent) while the aperture flux stays protected (+6
  percent): sum(amps) is never a reported quantity.

- Total flux: the tau-dial (free core, wings completed by the
  prior): n=2 +0.0 +- 6.6 percent at s2n=30 vs exp -17.5; n=4
  -7.7 +- 1.9 percent at s2n=100 vs exp -32.  Tight wing-blend
  totals are intrinsically ambiguous (member flux and neighbor
  wing degenerate below the curvature scale; the standalone
  frames also swell under wings, see the integration item).

- Apertures and calibration: adaptive gauss capture floors at
  0.81 for disks (the weight scales with the galaxy) and
  0.6-0.75 for dev wings at any size; over the wldb i<25
  population the median capture is 0.96 with 6.5 percent
  below 0.8, driven by bulge fraction, not size.  The fixed
  2-arcsec gaussian flux is exact from the ladder mixture
  (<= 0.1 percent, where gauss/exp mixtures err 1-16 and
  0-2.5 percent), star-free calibratable on the psf model
  (fwhm +0.5 percent -> +0.2 percent flux, k=0 protected), at
  0.86x adaptive S/N for the median object.  DECISION:
  adaptive colors are primary (S/N, blending, the
  matched-flux normalization cancels in ratios); the fixed-2
  flux and the color gradient are derived functionals;
  reduced outputs add gradient(+err) and drop the fixed
  aperture, reconstructable from the full outputs, which keep
  the amps.

- Consider bdf for the PAdmomFitter full errors (gauss/exp/dev
  done).

- Tight-blend (nn < 8 px) structure errors under recentering:
  the moving-anchor FD probe reads fam/gauss e and T 15-30
  percent low in the jumpy-anchor half of that bin (sep
  deblend-position instability under noise, a non-gaussian
  channel outside the linear anchor term; 65-row bin, EPS=1
  doubles the noise so this is an upper-ish estimate).
  Possible mitigations if it matters: inflate the anchor
  covariances for multi-peak segments, or flag unstable-anchor
  rows.

- Extend `bdf_joint_sandwich` with the cross-band flux
  covariance and lift the bdf/star guard in full_errors.


- Remaining cost levers, all bounded (full errors now 21 ms
  single / 2.0-2.6x fit, Cov(S)-dominated at 13.6 ms): a numba
  analytic model-sum derivative kernel next to
  gauss_comps_ksums (replacing the closed-form micro-FDs in
  _model_sum_derivs, ~2-4 ms), and Cov(S) itself.  Measured
  dead ends, do not revisit: chirp-z zoom kernels (exact but
  slower than the rfft path -- czt constant factor); micro-FD
  chain evaluations (superseded by the hand-differentiated
  algebra, which wins at every group size).

DONE (2026-07-29): recenter-on field validation (moving-anchor
FD probe, 64 wldb fields, arms re-run sep on the perturbed
detection coadd and hand the matched positions in as anchors,
so anchor noise and the anchor-sums cross-correlation enter).
All reported errors honest with recentering on: fluxes
0.97-1.12, e1/e2 0.95-1.01, colors 0.87-1.00, gauss entries
0.97-1.13.  The anchor term contributes at most 0.6 percent of
any reported error at production settings (data dominates the
centers at cen_sigma0 = 0.1), so the neglected anchor-sums
cross-correlation is bounded to irrelevance; sep errx2
underprices the actual anchor noise response about 2x (median
|da|/sep-err 2.0-2.5), immaterial at that share.  Residual:
tight-blend T/e excess, moved to the open list above.

DONE (2026-07-29): kdeblend gauss-estimator full errors from
the Sw rows: gauss_T/e errors from the weight block of the
state covariance, gauss fluxes as derived response rows
(direct data channel + own-kernel/neighbor/det-norm chain),
gauss_flux_cov added.  MC: sandwich emp/rep gauss_T 1.17
exp-single, 1.38 dev-single, 1.42 tight pair (pair flux 1.30);
full errors 1.01-1.04 everywhere.

DONE (2026-07-29): PAdmomFitter(full_errors=True) for
gauss/exp/dev (ngmix padmom_full_covariance): dev-truth MC
calibrated at 0.98 (T) / 0.99 (flux) where the sandwich reads
1.17 / 1.11; cost 4.2 -> 13.8 ms per fit.

DONE (2026-07-29, validated by unit tests, ensembles, mismatch
MC and field-scale FD probes; all errors honest 0.95-1.06 at
field scale): full (fixed-point) errors -- fluxes, cross-band
flux covariance, and T/e1/e2 -- for all objects including
singles, in kdeblend/full_errors.py on the
ngmix.prepsfadmom.full_errors building blocks (analytic
ktransfer in prep_epoch, closed-form kernels and theta
derivatives, rfft influence kernels, targeted state
save/restore), wired through simcoadd-mdet as the required
full_errors config option with sep anchor covariances and
flux_cov_{b1}_{b2} catalog columns.
