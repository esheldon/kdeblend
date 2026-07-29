# TODO

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
