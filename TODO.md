# TODO

- Wire the group errors into simcoadd-mdet: a required config
  option on the kdeblend fitter passing
  `group_errors=True` (and `anchor_sigma` when recentering with
  detection anchors) through fit_deblend, plus catalog columns
  for the cross-band flux covariance (schema policy: only in
  modes that fill them).  Then the field-scale validation with
  the FD-replica probe harness (production fields, real sep
  detection, blendedness-binned).

- Extend `bdf_joint_sandwich` with the cross-band flux
  covariance (the model_sandwich extension covers gauss/exp/dev;
  bdf currently reports flux_cov=None on the joint path) and
  lift the bdf/star member guard in group_errors.

- gauss-estimator flux covariance (gauss_flux_cov) from the
  gauss sandwich call in _run_sandwiches, if the gauss-aperture
  colors are ever used downstream.

DONE (2026-07-29, on kdeblend-dual + ngmix kspace-admom,
validated by unit tests, the m=1 reduction, and 400-refit
ensembles): the group-coupled adjoint sandwich
(kdeblend/group_errors.py, deblend(group_errors=...,
anchor_sigma=...)) and the model_sandwich cross-band flux
covariance for singles (ngmix errors.py, result flux_cov).
