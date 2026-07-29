# TODO

- Investigate the isolated fam_T_err calibration: the field
  FD probe (2026-07-29) reads fam_T at 1.12 even for isolated
  objects (e1/e2 are honest there), so the per-object structure
  sandwich itself under-predicts T errors ~12 percent; fix
  before the group extension inherits it.

- Extend apply_group_errors to the structure errors: the group
  covariance already contains the cov/Sw rows; the field probe
  shows the per-object T/e errors miss 20-35 percent in the
  tight bin (e1 1.20, T 1.35) while the group fluxes are
  honest there (0.99).

- Recenter-on field validation: anchor term with the sep
  covariances end-to-end, and the neglected anchor-sums
  cross-correlation (phase-randomized re-centroiding probe).

- Extend `bdf_joint_sandwich` with the cross-band flux
  covariance (the model_sandwich extension covers gauss/exp/dev;
  bdf currently reports flux_cov=None on the joint path) and
  lift the bdf/star member guard in group_errors.

- gauss-estimator flux covariance (gauss_flux_cov) from the
  gauss sandwich call in _run_sandwiches, if the gauss-aperture
  colors are ever used downstream.

DONE (2026-07-29, validated by unit tests, the m=1 reduction
and 400-refit ensembles): the group-coupled adjoint sandwich
(kdeblend/group_errors.py; deblend(group_errors=...,
anchor_sigma=...) with scalar / per-object-sigma /
per-object-covariance anchors) and the model_sandwich
cross-band flux covariance for singles (ngmix kspace-admom,
result flux_cov).  Wired into simcoadd-mdet: required
group_errors bool on the kdeblend fitter, sep
errx2/erry2/errxy anchor covariances when recentering,
flux_cov_{b1}_{b2} catalog columns gated on the mode.
