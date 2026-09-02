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
  Stage B (2026-09-01, stageB-ladder-vs-exp): 100 seeds x 3
  wldb gri fields per file, exp and ladder on the same fields,
  g1 = +-0.02 for noise cancellation, types noshear/1p/1m,
  gauss shapes, Trat 0.5-6, shape-noise weights; 400 jobs on a
  20-wide pinned queue, 62 min.  The doshear-cancel estimate
  per model (own selection, bootstrap over files) is far too
  noisy at this scale: m_exp +0.002 +- 0.070, m_ladder +0.110
  +- 0.091, dm +0.11 +- 0.085 (s2n > 10; +0.20 +- 0.087 at
  s2n > 20), and its R11 differed 10 percent between models --
  an artifact: 12 percent of selected objects flip selection
  between the models (7 exp-only, 5 ladder-only: blends near
  the Trat/s2n thresholds).  The rows are the same detections
  in the same order, so the informative statistic is per
  object (stageB_paired.py): 61 percent of selected objects
  have bit-identical shapes under both models (no neighbor in
  reach) and contribute dm = 0 +- 0.0003; on the common
  selection dm = -0.019 +- 0.027 (s2n > 10), -0.009 +- 0.017
  (s2n > 20), dR = +0.002 +- 0.011 / -0.005 +- 0.010.  No
  evidence of a differential shear bias; the 1e-3 target needs
  ~300x the fields (Stage C, condor), and the own-selection
  differential (the metacal-relevant one, selection response
  included) needs more still.  Note for Stage C: bootstrap
  over files needs many small files, and the per-object
  pairing is what makes the difference measurable.
  Cost at Stage B scale (2026-09-01, three metacal types): the
  queue throughput gave 69 vs 47 s per field, ladder vs exp
  (1.5x, 20 concurrent pinned processes), against 1.3x in the
  pilot.  Typical fields (max group 4-9) are 1.3x whole
  pipeline including the ~7-8 s per-process numba compile
  (15.4/23.7/14.2 -> 19.9/31.0/18.2 s); a field profile (202
  groups) compile-free is 1.46x: sim 5.3 and metacal 4.2 s
  shared, deblend 19.3 vs 12.0 (sweeps 5.4 vs 4.4, full errors
  9.6 vs 4.3, of which Cov(S) 5.1 vs 2.3 -- the influence
  kernel irfft2 count scales with the 2.3x rows; now the top
  ladder lever, the k-space Gram fast path).  The run-level
  1.5x is the group-size tail: a field with a 39-object group
  runs 124 vs 239 s (1.9x) and costs 6-10 typical fields; 37 of
  100 files had a group > 10 (objects: p50/90/99/max group
  size 1/5/17/39; 0.7 percent in groups of 20-40).  The
  large-group cost is the production lever for both models.
  Ladder kernels now @njit(cache=True) (the compile was half of
  a single-field timing; ngmix's prepsfadmom kernels are not
  cached either).  Profile of that 39-object field (deblend
  138 s exp / 260 ladder): sweeps 67 / 160 -- both grind to
  the 500-sweep cap on the big group (5.1k sweeps, 40k / 52k
  object updates; _get_object_sums 55 / 87 s) and the ladder
  adds 2.4k joint solves at 21 ms each, all in
  ladder_measure_rows (57 s: the fused pass re-measures every
  member's rows although only a few still move -- the rows are
  data-only, so a per-object cache keyed on the aperture state
  is exact and would remove most of it); full errors 66 / 95:
  Cov(S) 33 / 50 (irfft2 20 / 31 s: ~32 ms per 16-row batch on
  the big cutout; the k-space Gram is the alternative there),
  _model_sum_derivs 18 / 15 (band_comps called 2.2M times: the
  per-call np.full/np.outer overhead, a vectorized group-level
  ns() would remove it for both models), _jacobi_block 5 / 11.
  Ranked: (1) per-object row cache in ladder_measure_rows
  (ladder only), (2) group-level vectorized model sums in
  _model_sum_derivs (both), (3) Cov(S) Gram path for large
  groups (both), (4) the 500-sweep grind itself (both).
  (1) landed (2026-09-01, LADDER_ROW_CACHE): a member whose
  position and weight are bit-identical to its last
  measurement reuses its rows (data-only, the neighbor
  subtraction is applied after), exact.  The choice of exact
  equality is measured: member motion between successive
  solves is bimodal -- on the 39-member grind 29 percent of
  member-solves are frozen below 1e-12 and the rest move by
  more than 1e-3 (a large-amplitude limit cycle, not a slow
  contraction); small groups 10 percent frozen, 30 below 1e-6,
  64 below 1e-3 -- so a tolerance up to 1e-9 adds no hits, and
  anything larger would inject discontinuities against
  DEFAULT_TOL = 1e-8 (the rho-capped projected residual treats
  any jump above ~1e-11 as non-convergence).  The FD referees
  never call ladder_measure_rows, so the cache cannot touch a
  derivative.  Measured (A/B, three types, catalogs identical to 1e-9 in
  every column): the 39-member field 245 -> 240 s (2 percent),
  a typical field 18.8 -> 18.8.  The frozen-member statistics
  above came from the noshear pass (12 solves of the big
  group); over all three types the frozen fraction is 17
  percent on the 39-member group (82 percent move by more
  than 1e-3), 7 on 6-19, 9 below 6.  Kept (exact, free).
  What the numbers actually say about the big-group cost: the
  group ran only ~85 solves across the three types, yet
  ladder_measure_rows took ~40 s of them -- ~0.5 s per solve,
  ~4 ms per member-epoch pass against 0.2 on a typical field:
  every per-object k-space pass runs over the whole group
  cutout's modes (a 600^2 padded grid is ~20x a stamp), and
  the same holds for the gauss loop's _get_object_sums (55 /
  87 s here, both models).  The structural large-group lever
  is per-object sub-grids (each member's sums on its own
  stamp, neighbors entering through the closed-form sums,
  which are grid-free) -- an architectural change shared by
  both models; short of that, (2) and (3) above.
  Correction (2026-09-01): stamps are out for the fit -- the
  data sums truncate bright neighbors at stamp edges and the
  edge rings through the deconvolution, which is why the
  group-replace cutout won -- and the ladder rows are data
  under apertures too.  The error stage is different in kind:
  the influence kernels are functions of the weights and psf
  only.  Prototype on the 39-member group (332 rows/epoch,
  image 187x205 on a 820^2 fft grid, i.e. 16x padding): a box
  of 4 sigma + 2 fwhm holds >= 99.993 percent of every row's
  energy (5 sigma + 3 fwhm 99.998), truncating to it perturbs
  Cov(S) by <= 1.6e-3 in correlation units; 41-53 percent of
  row pairs overlap in a group that dense.  Building the
  kernels from every s-th mode (a (dim/s)^2 grid, the kernel
  periodized with period dim/s) is exact on the image whenever
  dim/s >= image + extent: s=2 gives kernel errors 5e-7
  median, Cov(S) within 3e-6, irfft2 5x faster; s=4 (grid =
  image) fails for members within ~16 px of the cutout edge
  (they sit as close as 14 px), as the wrap condition says.
  Landed: KERNEL_SUBSAMPLE in full_errors -- per row the
  largest divisor of dim whose grid clears image + extent (5
  sigma + 3 smoothing fwhm), the widest ladder apertures on
  large objects staying on the full grid; unit test against
  the full grid at 1e-5; the 25 error tests pass 20 percent
  faster.  Fields (three types): the 39-member field exp 132
  -> 115 s, ladder 245 -> 221; typical exp 16.1 -> 15.6,
  ladder 20.2 -> 19.1.  The sweeps' cutout-area cost is
  untouched by this (and cannot be, per the fit finding).
  (2) landed (2026-09-01, MODEL_SUM_DERIVS_PAIRWISE): the
  model-sum derivatives take their micro-FD on the perturbed
  object's own pair term wherever it is a neighbor (the sums
  are additive, so the full-set difference is the pair
  difference exactly); the full set only under the object's
  own weight or center.  O(nobj^2) small kernel calls per group
  instead of O(nobj^3) model expansions; the full-set form is
  kept as the referee (_model_sum_derivs_full) and the
  equivalence test holds to 1e-7 on mixed two- and
  three-member groups.  Fields (three types): the 39-member
  field exp 114 -> 104 s, ladder 221 -> 211; typical unchanged;
  catalogs identical except error columns at <= 1.4e-5 (the
  full-set form's FD cancellation noise).  Today's two
  error-stage changes together: that field exp 132 -> 104,
  ladder 245 -> 211.  The remaining tail cost is the sweeps
  (67 / 160 s there) -- the grind.
  Grind diagnostic (2026-09-01, instrumented sweeps of the
  39-member group, both models, all three types; then the
  Stage B catalogs): no pass hits the 500 cap there (134-393
  sweeps) -- the sweeps go to the containment cascade: 46-56
  events per pass, 25-30 restarts then 21-26 demotions of the
  37-39 members (two thirds end as point sources), 560-1070
  rejected structure updates, each restart/demotion resetting
  the convergence history; not a limit cycle, no slow mode.
  Triggers visible in the movers: detection pairs 1.0-1.8 px
  apart (unresolvable at a 4 px psf) and weight runaways to
  T = 78-271 arcsec^2.  Run-wide (150 fields): the sweep
  budget is in groups of 3-10 (60 percent of object-sweeps;
  p90 sweeps 230-400), the giant groups are 2 percent; 2.6
  percent of objects end at the cap (both models); objects in
  a restarted/demoted state are 6-7.5 percent of the catalog
  and 19-21 percent of the sweeps (median 120 sweeps vs 12
  clean).  By nearest-detection distance: < 2 px is 12.8
  percent of detections but 43-46 percent of interventions
  and 27-34 of object-sweeps; < 3 px is 21 percent of
  detections, 60-63 of interventions, 43-47 of sweeps, with
  demotion 17-23 percent and 5-7.5 at the cap; beyond 8 px the
  intervention rate is 1-2 percent.  The ladder intervenes
  less than exp everywhere (demoted 4.6 vs 6.5 percent).
  Against the sim truth (three fields, pair_truth.py): all
  15 sub-2-px detection pairs are two detections of the same
  truth galaxy (0 real pairs, 0 spurious), i.e. split peaks.
  CORRECTION (2026-09-01): the split peaks are not the
  detector's (sep, as in production; fpdetect was never
  used).  The example gri configs used for Stage A/B and the
  timing had anull_extra_detections: true (the experimental
  adaptive-color-null extras, color_det = 1; off in
  production): they are 12 percent of the catalog with a 41
  percent sub-2-px-neighbor rate, intervene 42-50 percent of
  the time and hit the cap 8-9 percent; plain sep detections
  intervene 1-1.6 percent and cap at 1.8-1.9.  89-91 percent
  of all interventions and 52-55 of cap objects involve an
  injected detection or a sep detection within 2 px of one;
  with the injection off the same three fields have zero
  sub-2-px pairs (264 vs 305 detections).  So the grind
  measured here is mostly the injected extras fighting their
  hosts, not a deblender or detector pathology; the
  production-relevant numbers are the sep-only rows above,
  and the Stage A/B exp-vs-ladder comparisons stand as
  relative statements (same config for both) while their
  absolute health, sweep and timing-tail numbers carry the
  artifact.  Remaining, on sep-only detections: what drives
  the 1-1.6 percent interventions and 1.8 percent cap losses
  (the 433 sep-sep close pairs intervene at only ~1-2
  percent).  Diagnostic scripts: scratch grind_diag.py
  (per-sweep hooks), pair_truth.py.
  DECISION (Erin, 2026-09-01): when the extra detections are
  turned on in production, they get a de-duplication pass
  first (simcoadd-mdet).  What the data say it should do: the
  extras are injected with pinned centers to find missed
  objects, so an extra within the psf core of an existing sep
  detection is a duplicate by construction -- 41 percent of
  them sit within 2 px of one (all of the sub-2-px pairs on
  three fields were two detections of one truth galaxy), and
  those drive 89-91 percent of the restarts/demotions and
  half the cap losses.  Drop (or merge into the host) any
  extra within ~2-3 px, a fraction of the psf fwhm, of an
  existing detection; the detector's own sep-sep close pairs
  (433 of 12.9k) are not the problem (intervene at 1-2
  percent), so the pass is specific to the injected channel.
  Same for the s2 extras when they are used.
  Ladder cycle (2026-09-01, extras off, seed 8081 trial 2):
  with the extras off the same field has a 25-member group
  where exp converges in 117 sweeps (18 restarts, 14
  demotions) and the ladder's noshear pass runs to the cap
  (exp 49 s, ladder 150 s; typical fields 1.27x).  Per-sweep
  trace: a period-4 cycle locked to LADDER_SOLVE_EVERY = 2 --
  one faint member (F ~ 20-30) in the wings of two
  runaway-weight neighbors (T 30 and 81) alternates T = 11.2
  -> 38.2 -> 14.2 -> 47.4 -> 11.2 exactly, the amp solve (da
  9.9e-3 / 8.5e-3, amp sums 24.6 <-> 26.0) on every other
  sweep: the solve sets the amps under the current weight,
  the next weight update balloons with those amps subtracted,
  the next solve re-measures under the ballooned weight and
  collapses it.  The weight map and the amp map never reach a
  joint fixed point; exp has the same runaway neighbors but
  no group solve to couple to.  Nothing intervenes: no
  extrapolation is accepted (a period cycle has no
  contraction) and the non-contraction check keys on
  constrained steps, so accepted-but-oscillating steps are
  exempt by design.  Realization-sensitive: the noshear pass
  alone (different rng sequence) converges in 150 sweeps.
  Cadence: 1 cycles on all three passes (period 2, at the
  cap), 3 converges on all three (124/163/290 sweeps vs
  500/200/355 at 2); cadence is a fragile knob (4 defeated
  convergence in the pair studies).  Fixed damping of the amp
  update (amps <- 0.5 new + 0.5 old, prototype hook) is not
  it: the cycling pass converges (232 sweeps) and another
  speeds up (355 -> 228), but a pass that converged in 200
  now runs to the cap -- the group sits at the edge of
  stability in every pass and a fixed blend only moves which
  pass tips over; and it halves the step everywhere (median
  sweeps on the field's other groups 21 -> 43).  Open, design
  level: (a) alternation-triggered damping on the amps (damp
  only when successive row-space amp changes flip sign, like
  the extrapolation guard); (b) a joint weight+amp inner
  update for the offending member; (c) the underlying runaway
  weights themselves -- the cycle needs a faint member (F ~
  20) in the wings of neighbors whose weights ran to T = 30
  and 81 arcsec^2 (F 220-480); MAX_WEIGHT_SIGMA_FAC bounds the
  weight to half the cutout, which on a 200 px group cutout
  allows T of thousands, so a physically motivated bound
  (relative to the smoothing scale or the segment) would help
  both models, exp included (its own runaways reach T = 307).
  (c) landed (2026-09-01, WEIGHT_TMAX_FAC = 10): an object
  entry may carry Tdet, the observed second-moment size of its
  detection footprint (the driver passes sep's unclipped
  x2 + y2 in sky units, floored at TGUESS_RANGE[0]; injected
  extras get Tsmooth), and the weight is bounded at
  WEIGHT_TMAX_FAC (Tdet + Tsmooth) -- a floor of ten smoothing
  scales (~2.4 arcsec^2) for compact footprints, then
  proportional to the footprint; rejected updates are
  contained exactly like the stamp bound's (WEIGHT_BOUNDED).
  A per-pixel peak-surface-brightness criterion was
  considered and rejected: at fixed s2n the peak scales as
  1/sqrt(T), so the s2n ~ 30 runaways (peak/sigma_pix
  0.4-0.85 at T 30-100) sit exactly where legitimate faint
  small detections sit (T ~ 0.5, s2n ~ 5).  The 25-member
  field, extras off, three types: ladder cap 0.309 -> 0.000,
  max sweeps 500 -> 133, flags==0 0.69 -> 1.00, 150 -> 88 s;
  exp max sweeps 117 -> 88, 49 -> 45 s; no T > 15 weights
  remain (6 before); demoted 6 -> 17 percent (ladder), 19 ->
  26 (exp), 17-18 percent of the field's objects bounded --
  the runaways are now contained instead of absorbing the
  group.  A typical field is bit-identical.  Unit test
  test_tdet_bounds_the_weight; 102 kdeblend tests pass.
  Run-wide (10 fields, 901 objects, extras off, no errors):
  the bound touches 0 of 571 isolated objects for either
  model; 0.4-0.8 percent of objects bounded, all in groups
  (median size 4) and mostly demoted; T > 15 weights 3 -> 0
  (exp), 2 -> 0 (ladder); cap rate exp 0.55 -> 0.33 percent,
  ladder 0.67 -> 0.67 (the remaining cap objects are one
  group with a different failure, not a runaway); demoted
  exp 0.44 -> 0.89 percent, ladder 0 -> 0.22.  Adopted.
  The remaining cap cases (2026-09-01, the 10-field set
  reproduced with sweep hooks, all three passes: 16
  unconverged runs in 7 of 20 trials) are convergence
  machinery, not blending: (i) bright + faint pairs (F 329 vs
  25, 15 px; exp and ladder alike, every pass): the faint
  member's weight flips T = 1.47 <-> 0.75 every sweep with no
  constrained updates -- a period-2 cycle of the plain weight
  update, rho ~ -1, which the Steffensen guard (0.2 < rho <
  0.998) rejects and the non-contraction check (constrained
  steps only) exempts; (ii) isolated faint singles (s2n 6-10,
  both models) plus a triple and a quad: 125-150 ACCEPTED
  extrapolations in 500 sweeps, one every 3-4 sweeps, each
  boost overshooting and the next re-estimating -- the
  booster fighting the iteration.  Measured on the twenty
  reproductions: extrapolation off converges everything but
  the pairs (cap exp 0.33 -> 0.22 percent, ladder 0.67 ->
  0.22) at +9 percent object-sweeps for exp on typical groups
  (ladder -15, the cap runs outweigh it); accepting negative
  rho (the Aitken midpoint) does nothing for the pairs and
  adds failures -- their fixed point is unstable under the
  plain map, so after the midpoint the iteration walks off
  again: that case needs under-relaxation, not extrapolation.
  Landed (2026-09-01, EXTRAP_MAX_UNPRODUCTIVE = 3): a boost is
  productive when the plain sweep after it changes less than
  the sweep before it; three consecutive unproductive boosts
  retire the booster for the run (a fixed count would not do:
  healthy large groups accept 25-41 productive boosts, the
  failing singles 125-150 unproductive ones).  Ten fields,
  both models: cap rate exp 0.33 -> 0.22 percent, ladder 0.67
  -> 0.22, median/p90 sweeps unchanged (11/26, 11/23),
  object-sweeps -9 / -16 percent; the 25-member field exp 44.6
  -> 42.4 s, ladder 88.5 -> 93.7 (its big group 133 -> 191
  sweeps once the booster retired: the one cost seen); a
  typical field's slowest group 74/81 -> 36/35 sweeps.  What
  remains at the cap: the bright + faint pair (both models,
  every pass) and two singles in sheared passes.  Global
  under-relaxation of the weight update (0.5) was measured and
  rejected: cap 1.2 / 2.2 percent and +60-100 percent sweeps,
  the slow contraction everywhere costs far more than the
  pairs; if the pairs matter, a per-member under-relaxation
  triggered by sign-alternating structure changes (or the
  non-contraction check extended to alternation, demoting the
  faint member) is the targeted form.  102 kdeblend tests pass.
  Checked (2026-09-01): the pass exists, per channel.  s2
  extras: S2_EXTRA_MIN_SEP = 4 px vs sep and prior extras
  ("removes the near-degenerate pairs that destabilize the
  deblend").  anull extras: ANULL_EXTRA_MIN_SEP = 1 px vs sep,
  ANULL_EXTRA_DUP = 1.5 mutual, and a post-fit
  FLAG_DUPLICATE_EXTRA at R_DUP_FIT = 1.5 px that only runs
  with recenter on (off in production and in these runs, so
  it fired on 0 of 1811 extras).  The 1 px exclusion is what
  let 40 percent of the anull extras sit within 2 px of a sep
  detection (0.6 percent within 1 px, 60 within 3, 70 within
  4); the docstring notes the anull radii were validated on
  the cluster-core scenes, where extras near bright members
  are the point, so the radius is a policy choice: on random
  fields everything under ~2-3 px is a duplicate, and the
  same 4 px as s2 would drop 70 percent of the anull extras.
  DONE (2026-09-01, simcoadd-mdet): the radii are config keys,
  fitter.anull_extra_min_sep and fitter.s2_extra_min_sep
  (optional, default EXTRA_MIN_SEP_DEFAULT = 4 px, validated
  non-negative; the module constants remain the defaults for
  direct callers), passed through mdet to the two channel
  functions; the two gri example configs carry them
  explicitly with the rationale.  Three fields with the anull
  extras on at 4 px: 280 detections (305 at 1 px, 264 with
  the extras off), zero sub-2-px pairs.  Tests: the six
  extra-detection tests pass (one stub in
  test_single_fit_path_extra_detections learned the keyword).
  Refactor of the ladder code (2026-09-01, bit-identical
  catalogs on both reference fields for both models, all three
  types with errors; 102 tests; timing within noise).
  Removed dead ends: the unused imports (FASTEXP_MAX_CHI2,
  admom_ksums, admom_finalize) and constants (_NAP, _CHI2_CAP);
  the settled A/B switches LADDER_ROW_CACHE, KERNEL_SUBSAMPLE
  and MODEL_SUM_DERIVS_PAIRWISE (the referees stay as functions:
  _model_sum_derivs_full, _cov_sums(subsample=False),
  _ladder_state_derivs, _flux_kernel_and_dtheta); the
  never-passed parameters a0=, others=, write=, use_fd= and the
  unused aps argument of ladder_measure_rows; the _MOM_IDX
  "general moment rows" scaffolding that only ever supported
  the T row (now T_ROW_INDEX, row_layout(), t_row_indices());
  the second closed-form kernel gauss_pairs_sums/pairs_sums,
  whose three uses were all components under one weight at zero
  offset (unit_flux_sums on grid_sums).  Split: the row
  measurement into _measure_object_rows plus the two caches;
  the context -> rows -> subtract others -> template sequence
  shared by the fit, the derived functionals and the error
  setup into ladder_rows; the per-row flux-or-T selection into
  _row_values; the fixed-aperture routines onto unit_flux_sums;
  the amp-solve gate out of _sweep into _ladder_solve_step;
  _any_model_ksums through band_comps.  Test-only reference
  helpers (_comps_sums, _comps_flux_sum) moved to the tests.
  The cross-module names (idx, Sws, Tws, aps, Fhat, d, var,
  wsum, Mt, K, Z, nap, nrows) are defined once in the module
  docstring; opaque locals renamed.
  Reproducibility note (2026-09-01): deblend(rng=None) with the
  automatic smoothing choice is not run-to-run reproducible --
  choose_fwhm_smooth's psf fits use an unseeded generator, so
  Tsmooth differs at the 1e-6 level between calls (0.33369770 vs
  0.33369683 on an 8-galaxy blend) and the exp sweep path with
  it (99 vs 169 sweeps; the converged fluxes agree to 5e-6).
  With a seeded rng or an explicit fwhm_smooth the result is
  bitwise repeatable (the driver seeds it).  DECIDED (Erin):
  rng is required whenever fwhm_smooth is not sent
  (_get_smoothing raises); the 70 test call sites pass one, the
  driver already did; the experiments/ scripts (7) predate this
  and would need an rng to rerun.  The sweep count's
  sensitivity to a 1e-6 change in the smoothing is itself a
  measure of how near-critical the exp iteration is on such
  blends.
  Visual comparisons (2026-09-01, experiments/vis_*.py, figures
  in ~/data/simcoadd-mdet/runs/ladder-figures): a bulge (n=4,
  red) + disk (exp, blue) galaxy at s/n ~100, a five-galaxy
  blend, and a random eight-galaxy blend (s/n 9-120 after the
  full-error coupling), all gri with the same red-bulge /
  blue-disk colors, fit at the true positions with all-exp and
  all-ladder.  Exp leaves a negative core + positive ring on
  every bulge-dominated member (strongest in i where the red
  bulges dominate), reduced chi2 1.5-6 within 1.5 arcsec, flat
  color gradients, and on the blends runaway/demoted faint
  members in bright wings (with Tdet the bound demotes them);
  the ladder's residuals are noise in every band (chi2 0.9-1.2),
  it reproduces the color gradients, and its total_flux
  recovers the bright members to 0-8 percent, the faint n=4
  members under (-19/-31) and a faint disk in a bright wing over
  (+53 +- 16).  Colors compared the right way -- the fitted
  weight applied to the object's own noiseless truth in the
  smoothed plane, not the integrated color -- the ladder's
  aperture colors are within 1 sigma (0.013-0.02 mag) for the
  bright members; the earlier +0.2-0.35 offsets were entirely
  aperture-vs-integrated.  A blue disk's color error is smaller
  than a red bulge's at the same i flux because g is the
  limiting band.  render.py now draws ladder objects
  (render_model needs Tsmooth).
  Proposal under evaluation (2026-09-01): a new repo for the
  ladder deblender, kdeblend frozen as the reference.  Stripped
  scope: types ladder, star (demotion, point externals), gauss
  (the estimator); group-replace mode only (stamps dropped);
  fixed externals kept; full errors kept (the ladder's
  total/fixed/gradient errors exist only through them, though
  production runs full_errors false today); recentering to
  decide (small in the deblender, ~200 lines of anchor
  machinery in full_errors); gpu/ not carried (its kernel is
  the gauss/exp sweep; the ladder port is future work).
  Sizes: ladder.py 1150 stays; deblender 2600 -> ~1400 (bdf
  ~450 lines, mixture/damped/shrinkage ~250, stamps ~150);
  full_errors 2800 -> ~1700 (bdf joint sandwich, fracdev
  columns, dPS channel, mixture branches of _phi_healthy /
  _chain_pieces; the FD referees stay); render/vis/flags ~450;
  ~7200 -> ~4700 package lines, ngmix.prepsfadmom stays the
  engine dependency.  Method: extract with history, put an
  identity harness first (reproduce_field.py against the
  reference catalogs, bit-identical in every column, the
  refactor's test), then strip and re-prove identity after
  each removal so Stage A/B validation transfers by
  construction; a model-keyed dispatch in the driver runs both
  packages until Stage C signs off.  Alternative: strip on a
  kdeblend branch with the reference tagged.  Recommendation:
  the extraction.  The risky strip is full_errors' interleaved
  exp/bdf branches.  Next step offered: the AST inventory of
  the functions and branches that leave.
  Photometry-path run (2026-09-02,
  ~/data/simcoadd-mdet/runs/photwhite-ladder-vs-exp,
  run_photwhite.sh, analysis.txt): truth_photometry.py
  --photometry skips metacal entirely (run_photometry: the sim's
  own psf and noise), white noise, extras off, gri, full errors,
  exp and ladder on the same 5000 fields and fit seeds (20 jobs x
  250; 444k matched rows per model, 25x Stage A; ~1.5 h per job
  at 20 concurrent, the smoke rate was 3.5 s per field for both
  models).  Per field the photometry path costs 1.5 s exp / 2.9 s
  ladder against 6.6 / 10.6 for three metacal types, and the
  white sim 0.4 s against 2-3 for the coadd noise; detections
  and groups are the same (90 vs 87 on one field), psf T 15
  percent smaller than the metacal target.  Results reproduce
  Stage A at 20+ sigma: isolated flux_i bias exp +1.8/+1.1/+0.7/
  -0.2 percent vs ladder aperture -0.5/-1.9/-2.3/-4.2 by s2n bin
  (10-20/20-50/50-100/100+); blended exp +12.1/+8.0/+7.0/+5.3 vs
  ladder +7.8/+4.6/+3.4/+2.1; bright-neighbor exp +9.4/+4.7/+2.6/
  +0.8 vs ladder +3.5/-0.2/-1.8/-3.0.  The high-s2n ladder deficit
  is the aperture vs the total truth (exp's gauss_flux shows the
  same -4.1 percent), not a fit error: the analysis needs the
  same-aperture truth (as the figures used) before the aperture
  pulls mean anything (isolated pull 10 at s2n>100 for both
  aperture columns).  Ladder total_flux (cap 4): isolated +3.1/
  +3.5/+3.6/+1.4 percent, blended +14/+11/+12/+11 -- the total
  absorbs faint undetected neighbors and neighbor residue;
  as a total estimator it is worse than exp's model total in
  blends.  Colors: both unbiased to 6 mmag; isolated pulls exp
  1.02/1.03/1.13/1.35, ladder 1.01/1.02/1.11/1.32 (white noise,
  exact weight: the color errors are honest at s2n<50, the
  high-s2n excess is the aperture-vs-integrated color); ladder
  color scatter 5 percent larger than exp's.  Health: flags 0.996
  both, deblend_flags==0 0.977 vs 0.984, star demotions exp 2.0
  vs ladder 0.6 percent.  The isolated pulls match Stage A's
  metacal-noise values (1.10/1.35/1.95 vs 1.08/1.34/2.03), so the
  kernel-scale weight calibration is adequate for the gauss
  aperture quantities.  Next: same-aperture truth column in
  truth_photometry.py (render the true object with the fitted
  weight, as vis_blend8 does) so flux and color pulls test the
  errors; then the ladder's total in blends.
  Same-aperture truth (2026-09-02, simcoadd-mdet
  simcoadd_mdet/truthaper.py + truth_photometry.py): run_sim
  returns the per-band scene (return_scene), every true object
  is drawn as the sim drew it and prepped on its own stamp with
  the fit's fwhm_smooth (now a catalog column), and the fitted
  gauss weight (gauss frame + smoothing, at the fitted position)
  is applied to the object's own light (ap_true_flux), to the
  detected and undetected other objects (ap_nbr_det/undet_flux)
  and to the observed image itself (ap_data_flux, so gauss_flux
  - ap_data_flux is what the group subtraction removed); ap4_
  repeats it for 4 x Sw, the ladder total's largest aperture.
  Normalization is the gauss flux's (4 pi sqrt det Sw times the
  weighted sum, fs/ws with one epoch per band); the stamp prep
  matches a full-field prep to 1e-8 (k-space sums are exact),
  and data = own + neighbors within noise.  Also saved: dx, dy
  (fit - true, px), wldb true_bt/true_hlr_b/true_hlr_d/true_z,
  undet_nbr_ratio/sep (truth), ndet_nbr/det_nbr_sep (other
  detections within 15 px: shredding), psf_T.  Costs ~3 s per
  field on top of the 3.5 s fits.  Smoke (3 fields, white
  noise): isolated same-aperture pull scatter 1.00 (exp) / 0.89
  (ladder), bias -0.2/-0.4 percent, so the gauss flux errors are
  honest; the bright isolated outliers (pulls -6 to -25) are
  real: the group subtraction removes 1-5 percent of a bright
  object's light even with no detection within 15 px (other
  groups' models absorbing its wings), the wings-stealing
  phenomenon, now measurable per object.  Analysis script
  extended (same-aperture flux and color tables, neighbor-light
  fractions, the ladder total by undetected light in the 4x
  aperture).  Prepared, not launched (Erin reinstalls first):
  ~/data/simcoadd-mdet/runs/photwhite-ap-ladder-vs-exp/
  run_photwhite_ap.sh, 16 jobs x 250 fields, seeds 9101-9116
  (launched 2026-09-02 08:15 against the installed packages).
  lsst-mdet ladder catalog (2026-09-02, branch "ladder"): the
  minimum ladder row decided with Erin -- flux_{b} = total_flux
  (the exp run's flux is the model total, same meaning), colors
  from the gauss fluxes with gauss_flux_cov, gauss_flux_{b} kept,
  gradient_{b1}m{b2} (fixed minus adaptive color) with errors
  kept, fixed fluxes and amps dropped; fit_model and fwhm_smooth
  columns for every model; demoted stars fall back to the psf
  flux.  get_struct(model=) keys the ladder columns; the family
  and gauss shapes coincide for a ladder so one shape block
  stays.  Also: Tdet (sep x2+y2, floored) now reaches kdeblend
  from lsst-mdet (the footprint weight bound was inactive there),
  the hardcoded fit settings moved to DEBLEND_SETTINGS and are
  written to the meta table with the kdeblend/lsst-mdet versions
  (meta model widened to U8: 'ladder' did not fit U5).  Tests:
  the packer on synthetic ladder/star results, the meta round
  trip, and fit_deblend end to end on a two-band blob scene for
  exp and ladder (66 pass).  Noted for the total: on noiseless
  data the cap makes the outer rungs an unconstrained
  extrapolation (dev +44 percent, exp +10), a regime only above
  s2n ~5000; with noise the prior completes and dev comes out
  low (0.94 at s2n 170, 0.82 at 17), exp 0.99 at every s2n.
  Same-aperture run results (2026-09-02, photwhite-ap-ladder-vs-
  exp/analysis.txt: 4000 fields, 356k matched rows per model,
  2:07 per job at 16 concurrent).  Errors: gauss flux vs its own
  aperture truth, isolated, bias +1.4/+0.7/+0.1/-0.5 percent (exp)
  and +1.2/+0.4/0.0/-0.5 (ladder) by s2n bin, pulls 0.90/0.96/
  1.21/2.6 and 0.89/0.94/1.16/2.5: honest (slightly conservative)
  below s2n 50; above 100 the 1.5 percent scatter is not noise
  but the group subtraction (other groups' models absorb ~1
  percent of a bright object's wings, subtr/own median 0.94/0.98
  percent), same for both models.  Colors: both unbiased to 5
  mmag in every bin; the catalog (covariance-aware) color errors
  are honest isolated (pulls 1.01-1.05 at s2n<50, 1.27 at >100),
  the independent-band combination is conservative (0.85-0.95);
  ladder color scatter 5-6 percent larger than exp's.
  Contamination accounting in blends (nbr_ratio>=0.1, i band,
  medians of fractions of own light): neighbor light in the
  fitted aperture 9.9 percent of which undetected 2.9-3.1;
  the fit subtracts only 2.2 (exp) / 2.7 (ladder); so gauss -
  own truth = +8.6/+4.8/+2.9/+2.2 percent (exp), +7.7/+4.3/+2.7/
  +2.3 (ladder): roughly a third undetected light nothing can
  model and two thirds under-subtracted detected-neighbor light
  (the models under-predict the neighbor's light inside the
  target aperture; the ladder removes ~20 percent more than exp).
  Split by undetected light in the 4x aperture: objects with
  <1 percent are clean (gauss same-aperture bias +1.2/+0.1/-0.4/
  -0.7, pulls 0.87/0.88/0.99/2.3) and the >=5 percent class
  carries +16/+11/+8/+6.6.  Ladder total by the same split:
  <1 percent +2.9/+1.4/+0.9/+0.0 with pulls 1.07/1.18/1.33/1.98
  (nearly unbiased, honest errors), 1-5 percent +5/+4.6/+5.2/
  +4.2, >=5 percent +35/+27/+22/+18: the total absorbs the
  undetected light through its outer apertures, as anticipated
  (the earlier +3.5 percent 'isolated' excess was this mixed
  population).  Star demotions exp 1.9 vs ladder 0.6 percent.
  Next: (a) the neighbor under-subtraction is the dominant
  aperture bias in blends for both models and is a wings
  problem -- check it against nbr_sep and the neighbor's
  true_bt, and whether the ladder's cap/prior is what limits its
  wings; (b) the total needs an undetected-light guard or the
  unmodelled-light covariate (the parked idea below); (c) the
  same-aperture columns are the reference for any estimator
  change from here.
  lsst-mdet provenance (2026-09-02, branch "ladder"): fit_model
  column removed at Erin's request (the model is in meta, a
  demotion is DEBLENDED_AS_PSF); the meta table is now built by
  provenance.make_meta as plain columns: run options (repo,
  collections, patch_dir, gaia_file, gsub, apod_stars, cells),
  every stage's settings from the module constants (DETECT_ and
  METACAL_SETTINGS dicts now drive detect.py/metacal.py, plus
  deblend_, s2_, starsub_, mfrac, apodize, cell geometry,
  skymap), version_* from each package's __version__ (Erin keeps
  them meaningful), and date/hostname/command.
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
