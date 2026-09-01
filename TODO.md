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
  (Tight-blend totals re-measured after integration: the
  standalone frame swelling was a harness artifact; see the
  remaining-slices note below.)
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
  Ladder-aware full errors (2026-08-31): the amps stay out of
  the state and are the closed-form response amps(x, S_ap,
  S_T) to the state, the aperture flux sums (new data modes,
  extra influence-kernel members of the same covariance
  construction) and the T-row moment sum.  One primitive --
  re-solve the amps from the linearized rows (dsums_dtheta at
  the aperture weights, chain factor the aperture multiple)
  at a perturbed state or data, recompute the neighbor sums --
  feeds the existing B_i for both J (replacing the model-sum
  micro-FD for ladder groups) and the new dFdS columns; the
  ladder's own update algebra is the gauss branch of
  _phi_healthy; the FD referee path re-solves inside
  _jacobi_block.  Validated: chain vs FD to 1 percent on
  diagonals for ladder+gauss and ladder+ladder pairs, and MC
  calibration of flux/T/e1 errors within the ensemble
  precision for a single and a pair (tests/test_ladder_errors.py).
  Finding on the way: the per-sweep amp rescale by the flux
  update (stale shape, fresh flux) drives overlapping ladders
  into a flux-trading limit cycle under noise (equal pair at
  1.25 arcsec unconverged at 600 sweeps; 29 without it, same
  fixed point) and made the amps depend on the flux history --
  removed; the amps now change only at the solve, exactly what
  the error model assumes.  Cost levers landed (2026-08-31): a batched closed-form
  kernel (gauss_pairs_sums: one call for all template, prior
  and others-subtraction entries; the 86k per-entry calls were
  pure Python overhead) and the analytic data-mode response
  (the rows enter the solve linearly, so d(amps)/d(row) is a
  column of A^-1 Mw^T/sigma; the neighbor sums respond through
  the unit rung sums).  Full errors 812 -> 335 ms pair, 227 ->
  146 single (exp 53 / 27); fit 100 -> 70 ms pair.  Then the
  flux-only aperture rows (_flux_kernel_and_dtheta: the flux
  row of moment_kernels and dsums_dtheta from one evaluation
  of the shared ingredients, exact against ngmix in the unit
  test) for the aperture members: full errors now add 143 ms
  on the pair (exp 38) and 69 ms on a single (exp 20); total
  fit+errors 213 / 84 ms vs exp 55 / 28.  The state-column
  response stays FD (32 re-solves, 40 ms total).
  Error-speed todo (pair, full_covariance 134 ms vs exp 35,
  profiled 2026-08-31): (1) Cov(S) 59 ms (exp 20): the 48
  aperture members triple the influence-kernel irfft2 count;
  batch the irfft2s in ngmix influence_kernels, and/or a
  k-space Gram fast path (G sqrt(ef2) products) exact for
  uniform weight maps without apodization; (2) _ladder_setup
  44 ms: _ingredients recomputes the phase and quadratic for
  each of the 12 apertures of an object, which share both --
  a shared-quadratic aperture routine (~34 -> 15 ms), plus the
  uncached exact variances (~10 ms); (3) the FD state-column
  response, 39 ms, replaced by the analytic d(amps)/dx
  (template and prior derivatives wrt the frame, most algebra,
  largest single item); (4) skip the prior recomputation in
  re-solves whose perturbed column is not that object's
  weight (~10 ms).  Projected floor ~60-70 ms, ~2x exp, which
  is itself Cov(S)-dominated (see the legacy cost item).  The
  fit's own 70 ms (12 aperture passes per ladder object every
  2 sweeps) is now the larger cost: the fused J-aperture
  admom_ksums kernel.
  The anchor response of ladder groups does not re-solve the
  amps (second order; open).
  Derived flux functionals (2026-08-31): ladder results carry
  total_flux (sum(amps) of a second joint solve of the same
  rows with the LADDER_TAU_TOTAL=0.3 prior), fixed_flux (the
  star-normalized model flux under a fixed LADDER_FIXED_FWHM=2
  gaussian in the smoothed plane, exact from the mixture) and
  gradient (fixed minus adaptive color per adjacent band pair,
  mag), with errors from the full errors: the direct data
  channel analytic through each solve's matrix, the state
  channel by FD of the re-solve chained through Tx, the
  gradient from the fixed response and the flux rows of Tx.
  MC calibration (N=60, err/emp): total 1.02-1.08, fixed
  0.94-1.07, gradient 1.09-1.10, alongside flux/T/e1 at
  0.92-1.15; the total recovers an exp truth to 1 percent on
  a single (694/1009 for 700/1000).  Errors now add 185 ms on
  the pair (78 single); the functional FD channel (2 tau x 2 x
  npars re-solves, ~40 ms) could share the subtraction
  re-solves of _ladder_derivs.
  Driver and fixed-external support (2026-08-31): kdeblend
  accepts ladder fixed externals (entry e1, e2, T are the
  gauss-estimator frame the amps were solved in, so the rungs
  reconstruct exactly; test_ladder_fixed_external) and the
  flux init dispatches them; simcoadd-mdet accepts model
  'ladder' (config validation, object types, fixed-external
  entries carry amps, catalog columns total_flux_{b},
  total_flux_err_{b}, fixed_flux_{b}, fixed_flux_err_{b},
  gradient_{b1}m{b2}(_err); example config
  example-wldb-random-coadd-ps-kdeblend-ladder-gri.yaml; the
  fit_model column widened from U5, which truncated 'ladder').
  Through the driver's two-blob scene over 30 realizations
  (s2n 40-50, gaussian truths): total_flux unbiased
  (mean/truth 0.98-1.00) with 7.5 percent scatter and err/emp
  0.90-1.03, adaptive flux 2.5 percent scatter, err/emp
  0.92-1.20, fixed flux err/emp 0.95-1.09
  (tests/test_mdet_detection.py::test_ladder_model).
  Pilot (2026-08-31, ~/data/simcoadd-mdet/runs/pilot-ladder-vs-exp,
  run_pilot_par.sh + compare.py): 20 wldb gri fields per config,
  same seeds so the rows match, group-replace with full errors,
  one core per process.  Health: ladder flags==0 0.976 vs exp
  0.970, deblend_flags!=0 6.1 vs 7.5 percent, demoted 83 vs 125,
  sweeps p50/p90 13/168 vs 14/202 (both hit 500 on a few
  groups).  Cost: 125 vs 55 cpu-s per field (whole pipeline),
  job max 461 vs 158 s.  Estimators, ladder vs exp on the same
  objects: family flux 2.0-2.5 percent lower at every s2n (the
  gauss-normalized aperture flux vs the exp model total: expected
  capture), colors identical (median 0, +-0.004 mag), gauss
  e1/e2 identical (median 0, +-0.003) with reported errors
  equal at the median and up to 5-7 percent larger at p84,
  gauss T lower by up to 0.02 for the p16 tail (less neighbor
  wing under the weight).  Derived columns sane: total/flux
  1.09-1.10, total err 1.5x flux err, fixed/flux 0.90, gradients
  consistent with zero at s/n ~ 1.  No failures.  The truth
  comparison (biases of fluxes, colors, sizes against the sim
  truth, and the metacal response) is the actual validation:
  Fused dyadic aperture kernel (2026-08-31, ladder_apsums): the
  apertures are 0.25 x 2^n, n=0..7 (J=8), so one pass per epoch
  gives every aperture flux sum from one exact exponential at
  a=1 with two square roots down and five squarings up (the
  squarings amplify relative error 2^n-fold; the table
  exponential's 2e-6 would become 2e-4 on the widest aperture,
  hence the exact root), plus the a=1 moment sums (the T row)
  and the noise variances of every row when the cache is
  stale.  Exact against the separate passes to 1e-6 (the table
  error).  Contamination with the production K=8 rungs: the
  dyadic-8 set matches the log-12 and log-16 sets to 10-20
  percent at every (n, d), all 30-1000x below exp.  One field:
  sweeps 47 -> 25 s (aperture passes 32 -> 8), errors 52 -> 47,
  mdet 108 -> 80 s vs exp 32 (2.5x, from 3.4x).  The error
  floor is now the functionals' FD state channel
  (_ladder_functional_covs, 21 s: two tau solves per column that
  rebuild rows and template already built by _ladder_derivs),
  then _ladder_derivs 12 s and Cov(S) 8 s; the fit's is the
  solve cadence on the grinders (gating on the weight change).
  Merged state loop (2026-08-31, _ladder_state_derivs): one
  restore/unpack and one row/template/prior build per state
  column serve the neighbor-sum derivatives, the fixed-flux and
  the total-flux responses (two cheap solves); _ladder_resolve
  is staged into _ladder_rows_at / _ladder_solve_at.  One
  field: errors 47 -> 29 s (state loop 14 s vs 12 + 21), mdet
  80 -> 62 s vs exp 32: 1.9x (3.4x at the pilot).  Remaining
  error floor: state loop 14 s (the analytic state response
  would remove it), Cov(S) 8 s (batched irfft2 / Gram fast
  path); fit: sweeps 24 s vs exp 12 (solve gating on the
  weight change for the grinders).
  Solve gating (2026-08-31): the solve is skipped when no
  packed-state component moved by more than 1e-4 (normalized)
  since the last solve, with the carried change retired and a
  final solve at the converged state; results identical to
  3e-5 in flux and 2e-5 in amps, fast groups skip a quarter of
  their solves, and no lag is introduced (it fires only on a
  static state).  It cannot touch the slow groups, which hold
  76 percent of the solves and keep moving > 1e-4 per two
  sweeps; an adaptive backoff of the cadence for them was
  measured and rejected (see LADDER_GATE_TOL).  Fit floor now:
  sweeps 23 s vs exp 12, of which the solves 13 s at ~2.7 ms
  each (~4700 per field); the remaining fit lever is per-solve
  cost (numba assembly, cached prior/template between the two
  bands), not solve count.
  Error levers (2026-08-31): the tau-independent assembly shared
  by the subtraction and total solves (ladder_assemble without
  the lam0 diagonal, ladder_solve_pieces adds it), the prior
  center cached against the weight (exact match in the error
  evaluations so the sw-column derivatives stay exact, 1e-3
  relative in the fit), single-batch prior and fixed-flux
  kernels, and the fused dyadic kernel rows and derivatives
  for the aperture members (_flux_kernels_and_dtheta_dyadic,
  exact against the per-aperture route).  One field: errors
  29 -> 23 s (state loop 14 -> 10, setup 2.6 -> 1.1), sweeps
  23 -> 21, mdet 59 s vs exp 32 (1.85x).  The error floor is
  now the FD state loop (10 s, ~1.1 ms per evaluation: rows +
  template + assembly 0.6, two solves 0.2, neighbor sums 0.25,
  save/restore 0.25) and Cov(S) 7.4 s (FFT-bound in the
  batched ngmix influence kernels; the a=1 aperture member
  duplicates the own flux row, ~10 percent).  The analytic
  state response (d(amps)/d(x) from the template and prior
  derivatives with respect to the frame) is the remaining
  structural lever for the state loop.
  Analytic state response (2026-08-31, _ladder_state_response):
  the amps' response to every packed state column through the
  solve chain dX = A^-1 (d rhs - dA X) at both prior widths,
  with the template, prior and subtracted-row derivatives from
  central micro-FD on the batched closed-form kernels (the
  _model_sum_derivs pattern), the data-row derivatives from
  the analytic kernel derivatives, the fixed-amps direct part
  of the neighbor sums from _model_sum_derivs (which now
  refreshes a ladder object's rungs with its weight column),
  and the amps channel through the unit rung sums; the fixed
  and total functionals likewise.  The FD state loop stays as
  the referee: agreement to 1e-3 on every derivative for the
  ladder+gauss and ladder+ladder pairs
  (test_ladder_state_response_matches_fd), chain vs the FD
  referee to 1e-4 on the covariance diagonals.  One field:
  errors 23 -> 15.4 s (state loop 10 -> 2.2), mdet 51 s vs exp
  32: 1.6x (3.4x at the pilot).  Remaining error floor: Cov(S)
  7.3 s (FFT-bound), the mode/functional/gauss-flux pieces
  ~4 s; fit sweeps 21 s vs 12.
  Pilot rerun (2026-08-31, par2, same 20 fields and seeds, all
  levers landed): cpu ratio of sums 1.30 (2.29 at the first
  pilot), per-job median 1.28, range 1.18-1.54 across the ten
  jobs (the exp baseline reproduces, 1099 vs 1097 s); the
  single-field profiles used seed 5003 throughout, a
  blend-heavier field than the median, and excluded the
  per-job startup, hence their 1.6x.  Health unchanged: flags0
  0.975 vs 0.970, deblend_flags 6.2 vs 7.5 percent, demoted 85
  vs 125, sweeps p90 141 vs 202.
  Per-solve cost (2026-09-01): kept -- the grid kernel
  (gauss_grid_sums: weights x components with per-pair offsets,
  no caller-side broadcasting; the template 170 -> 60 us, and
  every other batched closed-form call), and the fit-side prior
  cache tolerance 1e-2 (projections per solve 1.07 -> 0.83).
  Measured neutral or worse and reverted: one table
  exponential per aperture in the fused pass instead of the
  exact-root dyadic chain (~10 percent slower: the per-aperture
  branch costs more than the straight-line chain), and a
  jitted assembly (the 200 us is array handling around the
  call, not arithmetic).  Per solve 1.27 -> 1.06 ms on a
  median field; the fused data pass (0.62 ms per solve, ~0.2
  ms per object-epoch, ~5x a single admom_ksums pass for eight
  apertures plus the moment sums) is the floor, and the gpu
  port's kernel.
  Stage A memory (2026-09-01): Cov(S) built a group's influence
  kernels in one ngmix call, (nrows, dim, dim) on the padded fft
  grid plus the complex half plane, nrows = 6 nobj + 8 nlad per
  epoch; one wldb field reached 33 GB RSS (the other nine jobs
  4-7 GB).  Two fixes, both exact: _influence_kernels_chunked
  (16 rows per call, the same stacked real-space kernels) took
  that field to 19 GB; the rest was _ladder_setup holding every
  ladder object's (8, nmodes) complex kernel rows per epoch for
  the whole computation, plus _cov_sums stacking all rows into
  one (nrows, nmodes) complex array -- now _cov_sums rebuilds
  the dyadic kernels where it consumes them and streams the
  rows through the chunked builder.  That field (an 85-object
  group, heavy_field.py): ladder 2.4 GB, exp 0.7, wall 5:54 vs
  5:16 (noshear only).
  Total-flux background pickup (2026-09-01): in the
  truth-matched fields the ladder total ran +3-4 percent high on
  isolated bright small objects (exp model within 1) and +12-30
  in blends.  Noiselessly the total is exact (gauss/exp/dev
  1.0000, n=2 0.9995, n=4 0.952, wings past the largest rung)
  and a uniform background of 2 percent of the flux inside r=2
  arcsec raises it 14-37 percent through the a=32 aperture row
  (~100 arcsec^2 of whatever is unmodelled: undetected galaxies,
  sky residuals); the prior width does nothing there, the outer
  rows are many sigma on a bright object.  Lever added,
  LADDER_TOTAL_MAX_AP (largest aperture factor kept in the total
  solve; outer rows deweighted relative to noise and signal so
  the exp-projection prior completes beyond; default None,
  unchanged): at s2n ~100 the background response drops 2x at 8
  and 3x at 4, costing 1-3 percent on wide exp and up to 5 on
  n=4 wings.  Noiselessly any cap is ill-posed (the prior never
  engages against 1e18 row weights; the outer rungs become an
  extrapolation), so the cap only means something with real
  noise, and the noiseless derived-values test keeps cap None.
  Scan on 10 shared wldb fields (tauscan/, ~170 isolated and
  ~300 blended matches, same seeds): cap 4 takes the isolated
  total from +2.6/+4.6/+3.8/+1.9 percent (s2n 10-20/20-50/
  50-100/100+) to +1.1/+2.2/+2.2/+0.2 (exp model +1.6/-0.8/
  +0.4/-0.3), by size tracks the exp model within 2 percent
  and reads 1.007 where exp reads 0.964 (T > 1.2, the wings),
  cuts the low-s2n scatter 0.24 -> 0.14, halves the blended
  excess (+15/+13 -> +8/+8; exp +13/+4) and pulls 0.8-2.1 vs
  the exp model's 1.2-4.1; tau 0.1 adds nothing.  Adopted:
  LADDER_TOTAL_MAX_AP = 4, threaded through the full errors
  (var_total frozen in _ladder_setup, pieces_total in
  _ladder_rows_at, per-tau pieces in the analytic state
  response, per-functional assembly in the covariances); the
  chain-vs-FD, calibration and driver tests pass under it.
  Stage A (2026-09-01, truthphot-ladder-vs-exp, 10 jobs = 200
  wldb gri fields, exp and ladder on the same fields and fit
  seeds, noshear+1p, truth-matched, 18.3k matches each): health
  ladder flags==0 0.970 vs exp 0.966, deblend_flags==0 0.959 vs
  0.948, star demotions 459 vs 728.  Adaptive gauss flux (the
  same quantity for both): isolated identical (bias -0.2/-1.2/
  -2.7/-4.0 percent by s2n 10-20/20-50/50-100/100+ vs exp
  -0.5/-0.9/-2.8/-4.0, the aperture capture; scatter equal to 2
  percent; pulls 1.11/1.45/2.42/9.7 vs 1.12/1.53/2.64/10.6 --
  the high-s2n pulls are the capture-vs-total population
  scatter, not noise); bright neighbor (>= 1x within 15 px):
  +4.3/+0.5/-1.1 vs exp +11.1/+4.4/+0.5 percent with scatter
  0.26 vs 0.30 at s2n 10-20.  Colors g-r, r-i: medians within
  0.005 mag for both everywhere; bright-neighbor r-i at s2n
  10-20 +0.003 vs +0.013; isolated scatter 3-8 percent larger
  for the ladder (0.135 vs 0.131, 0.065 vs 0.060 mag: the
  per-band amps' freedom), bright-neighbor scatter smaller
  (0.164 vs 0.171, 0.162 vs 0.174); pulls 1.0-1.1 isolated for
  both, 1.2 vs 1.4 bright-neighbor.  The run's total_flux
  column is the uncapped solve (+7.4/+7.6/+5.0/+2.4 percent
  isolated, see above); the capped numbers are the scan's.
  Stage B (doshear differential m) and C (production scale)
  remain.
  Idea, for later (2026-09-01): the light the uncapped total
  absorbs is itself a measurement -- per object and band, the
  light in the 4-32 x Sw annuli that neither the object's inner
  profile nor the neighbor models explain (undetected galaxies,
  sky residuals, neighbor-model wings, psf-wing errors).  Uses:
  (1) an unrecognized-blend covariate for shear, sum(amps)
  uncapped minus capped, or the outer rows minus the capped
  model's prediction, measured on the sheared images too so a
  cut on it is a selection response, not a hidden bias; its
  per-band values give the color of the excess (a companion at
  another redshift, the photo-z blending failure); (2) a
  constant-surface-brightness nuisance column per band per
  group in the solve (row response b sum(W), growing with
  aperture area while any rung's fraction saturates: the c=25.6
  rung goes 0.24 -> 0.56 from a=8 to a=32, a constant 1 -> 4),
  which keeps the wings data-driven where cap 4 exp-completes
  them (dev-like galaxies) -- the principled version of the
  cap; (3) aggregated sky-residual / diffuse-light maps; (4) a
  group-level model-incompleteness flag.  Caveats: per object
  it is noisy (the a=32 row has ~6x the a=1 noise; uncapped
  minus capped scatters ~0.2 of the flux at s2n 10-20 against
  a few percent signal), and the flux rows cannot tell a
  uniform residual from an off-center companion from a
  neighbor's wing (outer-aperture moment rows from the fused
  pass would partly break that).  All linear functionals of
  measured rows, so errors come from the same chain.  Cheapest
  test: match objects across the tauscan capped/uncapped
  variants and correlate the excess with the truth catalog's
  undetected neighbors in the annulus.
  Remaining slices, in priority order: Stage B/C of the wldb
  validation against exp (doshear differential m, then
  production scale, with the capped total; Stage A photometry
  is done above; the gate for the exp/dev/bdf deletion
  decision, which also makes the bdf items below moot -- decide,
  do not do them); gpu support
  (the fused J-aperture admom_ksums kernel serves both the fit
  cost and the gpu port); the analytic state-column amp
  response.  (The tight-blend totals
  re-measurement is done, 2026-08-31: in the integrated
  deblender the 10:1 wings-on-compact faint member at d=1
  reads a total of -9.7 percent, not the +140-220 percent of
  the standalone harness whose frames came from
  gauss-corrected fits and swelled under the wings; the bright
  member -3.8 percent with its adaptive flux at -35.  The
  tau-dial total equals the adaptive flux times the fitted
  shape's aperture completion to 3 decimals in every case,
  see the LADDER_TAU_TOTAL comment.)

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
