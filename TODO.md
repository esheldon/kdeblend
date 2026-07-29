# TODO

- Consider bdf for the PAdmomFitter full errors (gauss/exp/dev
  done).


- Recenter-on field validation: anchor term with the sep
  covariances end-to-end, and the neglected anchor-sums
  cross-correlation (phase-randomized re-centroiding probe).

- Extend `bdf_joint_sandwich` with the cross-band flux
  covariance and lift the bdf/star guard in full_errors.

- kdeblend gauss-estimator full errors from the Sw rows (and
  gauss_flux_cov): now with measured motivation -- the gauss
  delta-method errors are low even on exp truth (T 1.14, flux
  1.06; dev 1.32/1.15) because every non-gaussian galaxy is
  mismatched for a gaussian-weight estimator; the PAdmomFitter
  gauss full errors fix this to 1.00-1.02 (MC 2026-07-29).

- Remaining cost levers, all bounded (full errors now 21 ms
  single / 2.0-2.6x fit, Cov(S)-dominated at 13.6 ms): a numba
  analytic model-sum derivative kernel next to
  gauss_comps_ksums (replacing the closed-form micro-FDs in
  _model_sum_derivs, ~2-4 ms), and Cov(S) itself.  Measured
  dead ends, do not revisit: chirp-z zoom kernels (exact but
  slower than the rfft path -- czt constant factor); micro-FD
  chain evaluations (superseded by the hand-differentiated
  algebra, which wins at every group size).

DONE (2026-07-29): PAdmomFitter(full_errors=True) for exp/dev
(ngmix padmom_full_covariance): dev-truth MC calibrated at
0.98 (T) / 0.99 (flux) where the sandwich reads 1.17 / 1.11;
cost 4.2 -> 13.8 ms per fit.

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
