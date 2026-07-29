"""
Multi-band deblending with pre-PSF adaptive moments in k-space.

Each band/epoch is deconvolved by its own PSF and smoothed by a common
round gaussian, placing all data in a common pre-seeing space (see
ngmix.prepsfadmom).  Objects have fixed centers and a per-object model
type; a Gauss-Seidel loop visits each object, measures its weighted
moment sums from the k-plane data, and subtracts the neighbor
contributions in closed form (see ngmix.prepsfadmom.models).  Structure is
common across bands; fluxes are per band with a common pre-seeing
aperture, so colors are independent of the per-band PSFs.

The data can be a single set of images shared by all objects
(deblend) or a postage stamp per object (deblend_stamps).  In stamp
mode each object is measured from its own stamps, while the neighbor
corrections use the neighbor models fit from their own stamps; only
the phase origin differs per object, so the two modes agree when
given the same pixels.

Notable iteration properties, established with the prototype tests:

- Centers are always fixed.  Free centers are unstable for close
  blends even with perfect models, and with model mismatch a faint
  object's center migrates onto its bright neighbor.  The implied
  centroid offset of the corrected data is recorded per object as
  'cen_pull'; a large value flags a bad local model or position.
- Fluxes are initialized by solving the per-band linear system at the
  guess structures; without this the first sweeps can assign the whole
  blend to one object and diverge for close pairs.
- On a failed structure update the previous structure is kept but the
  flux, which is linear and always well defined, is still updated.
  Without this a bright object with a bad initial structure can
  deadlock the blend.
- Persistent per-object failures are contained rather than fatal: at
  NFAIL_LIMIT consecutive failed structure updates the object is
  restarted from the compact delta state (where the neighbor
  contamination driving weight runaways is minimized), and if that
  also fails it is demoted to a fixed point source and marked
  DEBLENDED_AS_PSF in deblend_flags.  The group-level nskip limit
  remains only as a backstop.
- The exp/dev family state is the covariance matrix, not clipped
  (T, e1, e2) parameters, exactly as in ngmix PAdmomFitter._run_admom_mixture:
  proposed steps are damped against the model validity criterion
  rather than pinned at arbitrary parameter bounds, which lets faint
  families scatter through zero size instead of sticking at a clip
  and corrupting their neighbors' corrected sums.  Validity here is
  required for the smoothing alone (zero weight), which guarantees
  the model can be evaluated under every object's weight during
  neighbor subtraction.
"""
import numpy as np

import ngmix.flags
from ngmix.observation import get_mb_obs, ObsList, MultiBandObsList
from ngmix.moments import fwhm_to_T
from ngmix.prepsfadmom.prep import choose_fwhm_smooth, prep_epoch
from ngmix.prepsfadmom import get_phase_angles, deweight
from ngmix.prepsfadmom.errors import (
    model_sandwich, bdf_joint_sandwich, _mbasis_cov,
)
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums, admom_finalize
from ngmix.fastexp_nb import FASTEXP_MAX_CHI2

from ngmix.prepsfadmom.models import (
    det2, cov_from_e, model_ksums, model_comps, mixture_model_valid,
)
from ngmix.prepsfadmom.models_nb import gauss_comps_ksums

DEFAULT_TGUESS = 0.5
# past ~500 sweeps the surviving groups almost never converge
# (valid-step limit cycles; measured on 2000 wldb fields), so the
# default caps the grind and lets the caller cut on converged
DEFAULT_MAXITER = 500
DEFAULT_TOL = 1.0e-8

# zero weight for the scene-wide model validity rule: a model valid
# under the smoothing alone is valid under every object's weight
ZERO_WEIGHT = np.zeros((2, 2))

# deblend_flags bits: RESTARTED marks an object whose structure was
# restarted from the compact delta state after repeated failed
# updates; DEBLENDED_AS_PSF marks permanent demotion to a fixed
# point source after a restarted object failed again.
# EXTERNALS_SUBTRACTED is set by drivers that refit a group with
# fixed external models (the directed external subtraction scheme);
# it is defined here so all deblend_flags bits share one registry
DEBLENDED_AS_PSF = 2**0
RESTARTED = 2**1
EXTERNALS_SUBTRACTED = 2**2

# consecutive failed structure updates on one object before
# intervening
NFAIL_LIMIT = 10

# update the bdf flux split every this many sweeps: the split
# varies slowly compared to the structure, so intermediate sweeps
# can reuse it, saving the smoothing-aperture data pass.  The last
# update's change is carried in the convergence metric on the
# sweeps between updates, so a fit cannot converge with a stale
# split.  1 updates every sweep
BDF_SPLIT_EVERY = 2

# recentering: the maximum displacement of a center from its
# detection position, scaled by sqrt(Tsmooth), and the prior
# width in arcsec of the center regularization toward the
# detection position.  The per-sweep update is
# pos += k pull + (1 - k)(pos_det - pos) with
# k = sigma0^2/(sigma0^2 + sigma_pull^2): a free adaptive center
# for bright objects, frozen at the detection position when the
# pull is pure noise
RECENTER_CLIP_FAC = 0.5

# projected-residual stopping: the per-class contraction ratio
# is capped here; a sweep with no ratio estimate yet is treated
# at the cap, so early sweeps cannot stop spuriously.  The cap
# must exceed the slowest surviving plain-sweep contraction
# (the Steffensen boost handles ratios up to 0.998)
RHO_CAP = 0.999

# windowed non-contraction demotion: every WINDOW sweeps, an
# object whose windowed max change is above the structure
# tolerance, has not contracted by at least CONTRACT_FAC versus
# the previous window, and had a constrained structure update
# (rejected or boundary-damped) within the two windows is
# sawtoothing (constrained steps recur between free ones,
# threading the per-step nfail counter, which resets on
# accepted steps); the worst offender goes through
# the containment escalation immediately (forced restart, then
# forced demote -- the counter would just be threaded again).
# The rejected-step requirement protects legitimately slow
# contractions (rho^WINDOW > CONTRACT_FAC with every step
# valid).  One per window: coupled cycles often settle once
# the worst member is removed
NONCONTRACT_WINDOW = 50
NONCONTRACT_FAC = 0.7
DEFAULT_CEN_SIGMA0 = 0.1


def deblend(
    obs, objects,
    fwhm_smooth=None,
    smooth_fac=1.05,
    ap_rad=0.0,
    maxiter=DEFAULT_MAXITER,
    tol=DEFAULT_TOL,
    use_noise_image=False,
    rng=None,
    fixed_models=None,
    recenter=False,
    cen_sigma0=DEFAULT_CEN_SIGMA0,
    flux_tol=None,
    cen_tol=None,
    full_errors=False,
    anchor_sigma=0.0,
):
    """
    Deblend a set of objects with fixed centers.

    Parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The observation(s).  All epochs and bands are used jointly;
        each must have a psf set.
    objects: list of dicts
        One entry per object with entries
            v, u: float
                fixed center, as offsets from the image jacobian
                centers in sky coordinates
            type: str, optional
                'gauss' (default), 'star', 'exp', 'dev' or 'bdf'.
                Stars are pre-psf delta functions with only their
                fluxes fit.  The 'bdf' type is the composite exp
                plus dev model (shared center and ellipticity, dev
                size TdByTe times the exp size); the per-band flux
                split fracdev is fit from a two-aperture solve
                (the adaptive weight and the smoothing weight)
                interleaved with the structure updates, optionally
                regularized.
            TdByTe: float
                the dev to exp size ratio; required for 'bdf'
                objects (per object, mirroring the ngmix model
                spec dicts)
            fracdev0, fracdev_sigma0: float, optional
                sent together (or neither), regularize the bdf
                object's model flux split: the split that builds
                the composite is the inverse-variance blend of the
                measured split with the prior fracdev0 of width
                fracdev_sigma0.  The reported component fluxes and
                fracdev_gls stay the raw linear solutions;
                fracdev_sigma0=0 freezes the model split.
            Tguess: float, optional
                initial pre-psf T, default 0.5; ignored for stars
            fixcen: bool, optional
                keep this object's center fixed at (v, u) even
                when recenter is on (default False).  Useful for
                injected positions whose free centers would
                couple degenerately to nearby members
    fwhm_smooth: float, optional
        The common smoothing fwhm; chosen from the largest PSF if not
        sent (see ngmix.prepsfadmom).
    smooth_fac: float, optional
        Factor applied to the largest psf fwhm when choosing the
        smoothing automatically, default 1.05.
    ap_rad: float, optional
        Apodization radius in pixels for the stamps, default 0.  For
        deblending, apodization degrades the model subtraction near
        stamp edges and the smoothing already suppresses truncation
        leakage, so it is off by default.
    maxiter: int, optional
        Maximum number of Gauss-Seidel sweeps, default 500: small
        groups still unconverged there almost never converge
        later (valid-step limit cycles), and the result carries
        converged=False for the caller to cut on.  Sweeps to
        converge grow roughly linearly with group size, so
        callers fitting large groups should scale the cap with
        the member count (simcoadd-mdet does).
    tol: float, optional
        Structure (covariance/split) tolerance: the fit stops when
        the PROJECTED remaining distance to the fixed point,
        change * rho / (1 - rho) with rho the measured per-class
        contraction ratio, is below the class tolerance for every
        class.  This bounds closeness to the answer rather than
        the step size, uniformly across easy and strongly-coupled
        groups
    flux_tol: float, optional
        Flux-class tolerance, relative to each object's ratcheted
        historical flux scale; default 10 * tol
    cen_tol: float, optional
        Center-class tolerance, relative to the weight size;
        default 10 * tol
        per sweep, default 1e-8.
    use_noise_image: bool, optional
        If True, the per-mode noise power for the flux errors is
        measured from the noise realization attached to each
        observation (obs.noise) rather than assumed white at the
        weight-map level; use for correlated noise such as with
        metacal (see ngmix.prepsfadmom).  Default False.
    rng: np.random.RandomState, optional
        Used for psf fits when choosing the smoothing automatically.
    fixed_models: list of dicts, optional
        External sources whose light is subtracted in closed form
        but whose parameters are never updated, for contamination
        from outside the group (the directed external subtraction
        scheme).  Each entry has v, u (in the same frame as the
        objects), flux (array over bands), type ('gauss' default,
        'star', 'exp', or 'dev'), and for non-star types the pre-psf
        e1, e2, T, as reported in the objects entries of a previous
        deblend result; 'bdf' entries additionally carry fracdev
        and TdByTe.  Nonfinite parameters raise.
    recenter: bool, optional
        If True, the centers join the per-sweep updates, moving by
        the measured pull (the weighted centroid of the object's
        neighbor-corrected data, the same step the single-object
        adaptive-moments center update takes) regularized toward
        the detection position:

            pos += k pull + (1 - k)(pos_det - pos)

        with k = cen_sigma0^2/(cen_sigma0^2 + sigma_pull^2), where
        sigma_pull is the object's centroid noise.  Bright objects
        get a free adaptive center, faint ones stay at the
        detection position.  This removes the systematic sub-pixel
        errors of detection centroids (neighbor-pulled), which
        otherwise distort the fits of blend members.  The centers
        join the convergence metric and the sweep-map
        extrapolation, containment restarts reset them to the
        detection positions, and the displacement from the
        detection position is clipped to RECENTER_CLIP_FAC times
        sqrt(Tsmooth).  Default False (fixed centers).
    cen_sigma0: float, optional
        The prior width in arcsec of the center regularization,
        default DEFAULT_CEN_SIGMA0 = 0.1 (the scale of detection
        centroid errors).  Zero freezes the centers at the
        detection positions.  Unused with recenter=False.
    full_errors: bool, optional
        If True and the deblend converged (all members
        gauss/exp/dev), replace the per-object flux and
        structure errors with the full (fixed-point) values and
        fill flux_cov -- a full accounting of the errors.  For
        blend members this prices the neighbor-noise coupling
        the per-object sandwiches neglect (fluxes low by 10-30
        percent at 2 arcsec, T by 35 percent in tight blends);
        for every object including singles the structure errors
        avoid the model-consistency substitution of the
        per-object sandwich, which under-predicts T errors by
        ~12 percent under model mismatch (real morphologies fit
        with exp).  See full_errors.  Requires ap_rad=0.
        Default False
    anchor_sigma: float or array, optional
        With full_errors and recentering, the noise of the
        anchor (detection) positions: a scalar sigma in arcsec,
        an (nobj,) array of per-object sigmas, or an
        (nobj, 2, 2) array of per-object position covariances in
        arcsec^2 with (v, u) ordering (e.g. from the sep
        centroid error moments erry2/errxy/errx2 scaled to sky).
        The linear anchor response is added to the covariance
        (for tight pairs anchor noise at the detection-centroid
        scale can double the flux variance).  0 (default) leaves
        the errors conditional on the anchors

    Returns
    -------
    dict with entries
        objects: list of per-object dicts with type, T, e1, e2,
            their errors T_err, e1_err, e2_err from the sandwich
            over the moment matching conditions, e_flags (zero iff
            the ellipticities and their errors are usable),
            deblend_flags (RESTARTED when the structure was
            restarted from the compact delta state after repeated
            failed updates; DEBLENDED_AS_PSF when the object was
            demoted to a fixed point source, in which case type
            reports 'star' and the flux is the compact
            matched-aperture flux), flux and flux_err (arrays over
            bands), flux_cov (the (nband, nband) cross-band flux
            covariance from the shared family response, what
            honest color errors need; None on the star, bdf-joint
            and fallback paths), s2n (the flux s/n combined over
            bands in quadrature), cen, cen_pull.  Also gauss_T, gauss_e1,
            gauss_e2 with errors and gauss_e_flags: the
            gauss-estimator shapes from the converged weight, which
            is the adaptive-moments fixed point on the
            neighbor-corrected data.  These are lower noise than the
            family shapes (the family models still do the
            subtraction) and their response is calibrated by
            metacal like any estimator; the primary fluxes should
            come from the family models.  gauss_flux,
            gauss_flux_err and gauss_s2n are the gauss-aperture
            analogs, for selection studies
        fwhm_smooth, Tsmooth: the smoothing used
        numiter: number of sweeps
        nskip: total number of skipped structure updates
    """
    mbobs = get_mb_obs(obs)
    nband = len(mbobs)

    if full_errors and ap_rad != 0:
        raise ValueError(
            'full_errors requires ap_rad=0: the influence-kernel '
            'transfer assumes no apodization'
        )

    fwhm_smooth, Tsmooth = _get_smoothing(
        mbobs, fwhm_smooth, smooth_fac, rng,
    )

    epochs = _prep_epochs(
        mbobs, fwhm_smooth=fwhm_smooth, ap_rad=ap_rad,
        use_noise_image=use_noise_image, vcen=0.0, ucen=0.0,
        store_transfer=full_errors,
    )

    epochs_per_obj = [epochs] * len(objects)
    deb = _Deblender(
        epochs_per_obj, nband, objects, fwhm_smooth, Tsmooth,
        maxiter, tol, fixed_models=fixed_models,
        recenter=recenter, cen_sigma0=cen_sigma0,
        flux_tol=flux_tol, cen_tol=cen_tol,
    )
    res = deb.go()
    if full_errors:
        from .full_errors import apply_full_errors

        res['full_errors'] = apply_full_errors(
            deb, mbobs, res, anchor_sigma=anchor_sigma,
        )
    return res


def deblend_stamps(
    mbobs_list, objects,
    fwhm_smooth=None,
    smooth_fac=1.05,
    ap_rad=0.0,
    maxiter=DEFAULT_MAXITER,
    tol=DEFAULT_TOL,
    use_noise_image=False,
    rng=None,
    fixed_models=None,
    recenter=False,
    cen_sigma0=DEFAULT_CEN_SIGMA0,
    flux_tol=None,
    cen_tol=None,
):
    """
    Deblend a set of objects with fixed centers, with a postage stamp
    per object.

    Each object is measured from its own stamps; neighbor light in a
    stamp is subtracted in closed form using the neighbor models,
    which are themselves fit from the neighbors' own stamps.

    Parameters
    ----------
    mbobs_list: list of Observation, ObsList, or MultiBandObsList
        One entry per object holding the stamps for that object, with
        the jacobian of every stamp centered at the object position.
        All entries must have the same bands in the same order; each
        stamp must have a psf set.
    objects: list of dicts
        As for deblend.  The centers v, u are the object positions in
        a frame common to all objects, used for the neighbor offsets;
        the phase center of each object in its own stamps is its
        jacobian center.
    fwhm_smooth, smooth_fac, ap_rad, maxiter, tol, use_noise_image,
    rng, fixed_models, recenter, cen_sigma0: optional
        As for deblend.  The automatic smoothing choice uses the psfs
        of all stamps; with use_noise_image=True every stamp must
        carry its noise realization.  The fixed model centers v, u
        are in the same common frame as the object centers.
        Recentering shifts each object's position relative to its
        stamp centers.

    Returns
    -------
    dict as for deblend
    """
    nobj = len(objects)
    if len(mbobs_list) != nobj:
        raise ValueError(
            f'got {len(mbobs_list)} observation entries for '
            f'{nobj} objects'
        )
    mbobs_list = [get_mb_obs(m) for m in mbobs_list]

    nband = len(mbobs_list[0])
    for m in mbobs_list[1:]:
        if len(m) != nband:
            raise ValueError('all objects must have the same bands')

    # union of all stamps per band, for the smoothing choice
    union = MultiBandObsList()
    for band in range(nband):
        obslist = ObsList()
        for m in mbobs_list:
            for tobs in m[band]:
                obslist.append(tobs)
        union.append(obslist)
    fwhm_smooth, Tsmooth = _get_smoothing(
        union, fwhm_smooth, smooth_fac, rng,
    )

    epochs_per_obj = [
        _prep_epochs(
            m, fwhm_smooth=fwhm_smooth, ap_rad=ap_rad,
            use_noise_image=use_noise_image, vcen=o['v'], ucen=o['u'],
        )
        for m, o in zip(mbobs_list, objects)
    ]

    return _Deblender(
        epochs_per_obj, nband, objects, fwhm_smooth, Tsmooth,
        maxiter, tol, fixed_models=fixed_models,
        recenter=recenter, cen_sigma0=cen_sigma0,
        flux_tol=flux_tol, cen_tol=cen_tol,
    ).go()


def _get_smoothing(mbobs, fwhm_smooth, smooth_fac, rng):
    """
    the common smoothing fwhm, chosen from the largest psf when not
    sent (see ngmix.prepsfadmom), and its T
    """
    fwhm_smooth = choose_fwhm_smooth(
        mbobs, fwhm_smooth=fwhm_smooth, smooth_fac=smooth_fac, rng=rng,
    )
    Tsmooth = fwhm_to_T(fwhm_smooth) if fwhm_smooth > 0 else 0.0
    return fwhm_smooth, Tsmooth


def _prep_epochs(
    mbobs, fwhm_smooth, ap_rad, use_noise_image, vcen, ucen,
    store_transfer=False,
):
    """
    the prepared epochs for all bands (see
    ngmix.prepsfadmom.prep.prep_epoch), with the phase center
    entries stamped in.  In shared-image mode the phase origin is
    the jacobian center (vcen = ucen = 0); in stamp mode it is the
    object position, since each stamp jacobian is centered on its
    object
    """
    epochs = []
    for band, obslist in enumerate(mbobs):
        for tobs in obslist:
            ep = prep_epoch(
                tobs, band=band, fwhm_smooth=fwhm_smooth,
                ap_rad=ap_rad, use_noise_image=use_noise_image,
                store_transfer=store_transfer,
            )
            ep['vcen'] = vcen
            ep['ucen'] = ucen
            epochs.append(ep)
    return epochs


class _Deblender(object):
    """
    the Gauss-Seidel iteration over objects, with a per-object list
    of prepared epochs; in shared-image mode all objects have the
    same list

    The slow tail of the sweep iteration is a collective mode of the
    most blended objects, with their fluxes and structures locked in
    a single slowly decaying direction, so it is accelerated with a
    guarded Steffensen boost on the packed global state of all
    objects (see _extrapolate).  The packed state is a single
    normalized vector holding, for every object, its per-band fluxes
    and the entries of its model and weight covariance matrices;
    star structures are frozen so only their fluxes enter (see
    _pack_state). This single state vector, while unfortunately
    opaque, is needed for the Steffenson boost.

    Parameters
    ----------
    epochs_per_obj: list of lists of dicts
        For each object, the prepared epochs it is measured from
        (see ngmix.prepsfadmom.prep.prep_epoch), with the phase
        center entries vcen, ucen set
    nband: int
        The number of bands; the epoch band entries index this range
    objects: list of dicts
        As for deblend
    fwhm_smooth: float
        The common smoothing fwhm
    Tsmooth: float
        The T of the smoothing gaussian
    maxiter: int
        Maximum number of Gauss-Seidel sweeps
    tol: float
        Convergence tolerance on the maximum relative parameter
        change per sweep
    fixed_models: list of dicts, optional
        As for deblend
    """
    def __init__(
        self, epochs_per_obj, nband, objects, fwhm_smooth, Tsmooth,
        maxiter, tol, fixed_models=None, recenter=False,
        cen_sigma0=DEFAULT_CEN_SIGMA0, flux_tol=None, cen_tol=None,
    ):
        if len(objects) == 0:
            raise ValueError('no objects sent')

        self.recenter = bool(recenter)
        self.cen_sigma0 = cen_sigma0

        # per bdf object: the latest two-aperture component fluxes
        # (nband, 2), the raw split and its variance, and the
        # one-time aperture noise variances for the shrinkage;
        # the shrinkage parameters and the split init are read
        # from the object entries in _init_models
        self.bdf_info = {}
        self._bdf_noise_cache = {}
        self.fd_shrink = [None] * len(objects)
        self.fd_init = [0.5] * len(objects)
        self.bdf_last_dfd = np.zeros(len(objects))
        self.isweep = 0

        self.epochs_per_obj = epochs_per_obj
        self.nband = nband
        self.nobj = len(objects)
        self.fixcen = np.array(
            [bool(o.get('fixcen', False)) for o in objects]
        )
        self.fwhm_smooth = fwhm_smooth
        self.Tsmooth = Tsmooth
        self.maxiter = maxiter
        self.tol = tol
        # flux and center targets default to 10x the structure
        # tolerance: sizes/shapes carry the tightest systematic
        # requirement (weak-shear breakdown at m ~ 4e-4), fluxes
        # and centers are measured at far lower relative precision
        self.flux_tol = 10 * tol if flux_tol is None else flux_tol
        self.cen_tol = 10 * tol if cen_tol is None else cen_tol
        self._change_hist = {'flux': [], 'struct': [], 'cen': []}
        self.smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])

        self._init_models(objects)
        self.fpositions, self.fmodels = _convert_fixed_models(
            fixed_models, nband, Tsmooth,
        )

        self.nskip = 0
        # windowed per-object change maxima and constrained-step
        # (rejected or boundary-damped) counts for the
        # non-contraction demotion
        self._win_max = np.zeros(self.nobj)
        self._prev_win_max = None
        self._win_nfail = np.zeros(self.nobj, dtype='i4')
        self._prev_win_nfail = None
        # ratcheting per-object per-band flux scales for the
        # convergence metric (see _fchange)
        self._fscales = np.full(
            (self.nobj, nband), 1.0e-30,
        )
        self.cen_pull = [np.zeros(2) for _ in range(self.nobj)]
        # the detection positions: the recentering displacement
        # clip and regularization anchor to these
        self.det_positions = list(self.positions)
        # per-object pull noise for the recentering, computed
        # lazily at the first center update (negative marks unset)
        self._cen_sigma_sweep = np.full(self.nobj, -1.0)
        # scratch for the k-space sum kernels, overwritten per call
        self.esums = np.zeros(6)

        # sweep-map extrapolation history of normalized global states
        self.scales = None
        self.hist = []

        # per-object failure containment state
        self.nfail = np.zeros(self.nobj, dtype='i4')
        self.nrestart = np.zeros(self.nobj, dtype='i4')
        self.dbflags = np.zeros(self.nobj, dtype='i4')

        self._init_fluxes()

    def _init_models(self, objects):
        """
        the positions, models and weights at the guess structures
        """
        self.positions = []
        self.models = []
        self.Sw = []
        for o in objects:
            self.positions.append((o['v'], o['u']))
            otype = o.get('type', 'gauss')
            Tguess = o.get('Tguess', DEFAULT_TGUESS)
            m = {'type': otype, 'F': np.zeros(self.nband)}
            if otype == 'star':
                # pre-psf delta function: in the smoothed plane the
                # model and the matched weight are both the smoothing
                # gaussian
                self.Sw.append(self.smooth_cov.copy())
                m['cov_sm'] = self.smooth_cov.copy()
            elif otype == 'gauss':
                self.Sw.append(
                    np.diag([(Tguess + self.Tsmooth) / 2] * 2),
                )
                m['cov_sm'] = self.Sw[-1].copy()
            elif otype in ('exp', 'dev'):
                self.Sw.append(
                    np.diag([(Tguess + self.Tsmooth) / 2] * 2),
                )
                m['cov'] = cov_from_e(0.0, 0.0, Tguess)
            elif otype == 'bdf':
                if 'TdByTe' not in o:
                    raise ValueError(
                        "bdf objects require a 'TdByTe' entry"
                    )
                fd0 = o.get('fracdev0')
                sigma0 = o.get('fracdev_sigma0')
                if (fd0 is None) != (sigma0 is None):
                    raise ValueError(
                        'the fracdev shrinkage requires both '
                        'fracdev0 and fracdev_sigma0 (or neither)'
                    )
                if sigma0 is not None and sigma0 < 0:
                    raise ValueError(
                        'fracdev_sigma0 must be non-negative, '
                        f'got {sigma0}'
                    )
                i = len(self.models)
                if fd0 is not None:
                    self.fd_shrink[i] = (fd0, sigma0)
                    self.fd_init[i] = fd0
                self.Sw.append(
                    np.diag([(Tguess + self.Tsmooth) / 2] * 2),
                )
                m['cov'] = cov_from_e(0.0, 0.0, Tguess)
                m['fracdev'] = self.fd_init[i]
                m['TdByTe'] = o['TdByTe']
            else:
                raise ValueError(f"bad object type: '{otype}'")
            self.models.append(m)

    def _init_fluxes(self):
        """
        initialize the fluxes by solving the per-band linear system
        at the guess structures: the measured flux sums for each
        object are linear in all object fluxes with closed-form
        overlap coefficients.  The fixed external models are
        subtracted from the measured side
        """
        nobj = self.nobj

        # unit-flux copies of the guess models, for the closed-form
        # overlap coefficients
        unit_models = []
        for m in self.models:
            munit = dict(m)
            munit['F'] = np.ones(self.nband)
            unit_models.append(munit)

        for band in range(self.nband):
            A = np.zeros((nobj, nobj))
            bvec = np.zeros(nobj)
            for i in range(nobj):
                vi, ui = self.positions[i]
                Sw = self.Sw[i]
                for ep in self.epochs_per_obj[i]:
                    if ep['band'] != band:
                        continue
                    fac = ep['weight'] * ep['detAtinv']
                    alpha, beta = get_phase_angles(
                        ep, vi - ep['vcen'], ui - ep['ucen'],
                    )
                    admom_ksums(
                        ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                        alpha, beta, ep['kv'], ep['ku'],
                        Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                        self.esums,
                    )
                    bvec[i] += fac * self.esums[5]
                    for p, fm in zip(self.fpositions, self.fmodels):
                        bvec[i] -= fac * model_ksums(
                            fm, band, p[0] - vi, p[1] - ui,
                            Sw, ep['detAtinv'], self.Tsmooth,
                        )[5]
                    for j in range(nobj):
                        A[i, j] += fac * model_ksums(
                            unit_models[j], band,
                            self.positions[j][0] - vi,
                            self.positions[j][1] - ui,
                            Sw, ep['detAtinv'], self.Tsmooth,
                        )[5]
            fsol = np.linalg.solve(A, bvec)
            for i in range(nobj):
                self.models[i]['F'][band] = fsol[i]

    def go(self):
        """
        run the sweeps to convergence and package the results

        Returns
        -------
        dict as for deblend
        """
        converged = False
        for it in range(self.maxiter):
            self.isweep = it
            changes = self._sweep()
            if self._converged(changes):
                converged = True
                break
            if (it + 1) % NONCONTRACT_WINDOW == 0:
                self._check_noncontraction()
            self._extrapolate()

        return {
            'converged': converged,
            'objects': [
                self._get_object_result(i) for i in range(self.nobj)
            ],
            'fwhm_smooth': self.fwhm_smooth,
            'Tsmooth': self.Tsmooth,
            'numiter': it + 1,
            'nskip': self.nskip,
        }

    def _sweep(self):
        """
        one Gauss-Seidel sweep over the objects, returning the
        maximum relative parameter change per class (flux,
        structure, center)
        """
        self._sweep_changes = {'flux': 0.0, 'struct': 0.0, 'cen': 0.0}
        for i in range(self.nobj):
            ch = self._update_object(i)
            if ch > self._win_max[i]:
                self._win_max[i] = ch
        return dict(self._sweep_changes)

    def _converged(self, changes):
        """
        projected-residual stopping: from the per-class contraction
        ratio of consecutive sweeps, the remaining distance to the
        fixed point is ~ change * rho / (1 - rho); converged when
        that projection is below the class tolerance for EVERY
        class.  This bounds the distance to the answer rather than
        the step size, so the guarantee is uniform across easy and
        strongly-coupled groups, and stopping cannot freeze in a
        guess-side systematic.  A sweep without a ratio estimate
        (or with a growing change) is treated at RHO_CAP and
        cannot stop unless the change is already tiny; the history
        is reset wherever the sweep map is discontinuous
        (extrapolation jumps, restarts, demotions)
        """
        conv = True
        for cls, tol in (
            ('flux', self.flux_tol),
            ('struct', self.tol),
            ('cen', self.cen_tol),
        ):
            d = changes[cls]
            hist = self._change_hist[cls]
            if d > 0:
                if hist and hist[-1] > 0 and d < hist[-1]:
                    rho = min(d / hist[-1], RHO_CAP)
                else:
                    rho = RHO_CAP
                if d * rho / (1 - rho) >= tol:
                    conv = False
            hist.append(d)
            del hist[:-2]
        return conv

    def _check_noncontraction(self):
        """
        demote-or-restart the worst object whose windowed change
        is above tolerance and failed to contract versus the
        previous window; see NONCONTRACT_WINDOW
        """
        w = self._win_max.copy()
        prev = self._prev_win_max
        self._prev_win_max = w
        self._win_max[:] = 0.0
        wn = self._win_nfail.copy()
        prevn = self._prev_win_nfail
        self._prev_win_nfail = wn
        self._win_nfail[:] = 0
        if prev is None:
            return
        # a constrained structure update (rejected, or shortened
        # at the validity boundary) within the last two windows
        # separates a sawtooth (constrained steps recur; no fixed
        # point for the extended model; the cycle period can
        # exceed one window) from a legitimately slow contraction
        # (rho^WINDOW can exceed CONTRACT_FAC while every step is
        # free), which must not be touched
        bad = [
            i for i in np.flatnonzero(
                (w > self.tol) & (w > NONCONTRACT_FAC * prev)
                & (wn + prevn > 0)
            )
            if self.models[i]['type'] != 'star'
        ]
        if bad:
            i = max(bad, key=lambda k: w[k])
            self._contain_failure(i, force=True)
            # the comparison baseline is stale after the
            # intervention
            self._prev_win_max = None

    def _reset_change_hist(self):
        """the sweep map is discontinuous here; contraction ratios
        across the discontinuity are meaningless"""
        for hist in self._change_hist.values():
            del hist[:]

    def _note_change(self, cls, value):
        """record a parameter change in the per-sweep class maxima
        and return it, for the per-object bookkeeping"""
        if value > self._sweep_changes[cls]:
            self._sweep_changes[cls] = value
        return value

    def _flux_change(self, i, newF, oldF):
        """flux change against the ratcheted per-band scale"""
        self._fscales[i] = np.maximum(
            self._fscales[i], np.abs(newF),
        )
        return self._note_change(
            'flux', _fchange(newF, oldF, self._fscales[i]),
        )

    def _update_object(self, i):
        """
        update object i from its neighbor-corrected sums, returning
        its maximum relative parameter change.  On a failed structure
        update the previous structure is kept but the flux, which is
        linear and always well defined, is still updated, so a bad
        early structure state cannot deadlock the blend.

        With recentering the center update runs after the
        other updates, so within the sweep they all see the center
        the sums were measured at; the center lag vanishes at the
        fixed point like the other Gauss-Seidel lags
        """
        sums, fs, ws, pred, fs_pred = self._get_object_sums(i)
        m = self.models[i]

        if sums[5] > 0:
            self.cen_pull[i] = sums[0:2] / sums[5]

        if m['type'] == 'star':
            # structure frozen at the delta-function model; only
            # the linear flux is updated
            newF = _matched_flux(fs, ws, self.Sw[i], m['cov_sm'])
            change = self._flux_change(i, newF, m['F'])
            m['F'] = newF
        else:
            newSw = self._deweight_measured(i, sums)
            if newSw is None:
                change = self._skip_structure_update(
                    i, fs, ws, fs_pred,
                )
            elif m['type'] == 'gauss':
                change = self._update_gauss(i, newSw, fs, ws)
            else:
                change = self._update_mixture(
                    i, newSw, sums, pred, fs, fs_pred,
                )

        if (
            self.recenter and sums[5] > 0
            and not self.fixcen[i]
        ):
            change = max(change, self._update_center(i, sums))
        return change

    def _update_center(self, i, sums):
        """
        the regularized center update: move by the
        measured pull blended with a spring back to the detection
        position,

            pos += k pull + (1 - k)(pos_det - pos)

        with k = sigma0^2/(sigma0^2 + sigma_pull^2).  A bright
        object converges to its adaptive centroid, a faint one
        stays at the detection position; the pull noise is
        computed once per object at the first update (the weight
        evolves, so like the split shrinkage weight this is
        approximate and only sets the regularization strength).
        The displacement from the detection position is clipped.
        Returns the center change relative to sqrt(Twt)
        """
        if self._cen_sigma_sweep[i] < 0:
            covj = self._accumulate_error_sums(i)[2]
            var = (covj[0, 0] + covj[1, 1]) / sums[5] ** 2
            self._cen_sigma_sweep[i] = (
                np.sqrt(var) if var > 0 else 0.0
            )

        s0 = self.cen_sigma0
        sig = self._cen_sigma_sweep[i]
        if not np.isfinite(sig):
            return 0.0
        denom = s0 ** 2 + sig ** 2
        if denom == 0:
            return 0.0
        k = s0 ** 2 / denom

        p = self.cen_pull[i]
        v, u = self.positions[i]
        v0, u0 = self.det_positions[i]
        newv = v + k * p[0] + (1.0 - k) * (v0 - v)
        newu = u + k * p[1] + (1.0 - k) * (u0 - u)

        clip = RECENTER_CLIP_FAC * np.sqrt(self.Tsmooth)
        dvec = np.array([newv - v0, newu - u0])
        n = np.sqrt(dvec @ dvec)
        if n > clip:
            dvec *= clip / n
            newv = v0 + dvec[0]
            newu = u0 + dvec[1]

        dmax = max(abs(newv - v), abs(newu - u))
        self.positions[i] = (newv, newu)
        Twt = self.Sw[i][0, 0] + self.Sw[i][1, 1]
        return self._note_change('cen', dmax / np.sqrt(Twt))

    def _deweight_measured(self, i, sums):
        """
        the deweight update of object i's weight from the measured
        moment sums (single gaussian, all object types), or None if
        the sums do not admit one
        """
        if sums[5] > 0 and sums[4] > 0:
            newSw, flags = deweight(_moment_matrix(sums), self.Sw[i])
            if flags == 0:
                return newSw
        return None

    def _skip_structure_update(self, i, fs, ws, fs_pred):
        """
        a failed structure update: keep the previous structure but
        still update the linear flux, and count the failure toward
        containment.  Returns the relative change, always 1
        """
        m = self.models[i]
        self._count_skip(i)
        if m['type'] == 'gauss':
            newF = _matched_flux(fs, ws, self.Sw[i], m['cov_sm'])
            self._flux_change(i, newF, m['F'])
            m['F'] = newF
        elif np.all(fs_pred != 0):
            newF = m['F'] * fs / fs_pred
            self._flux_change(i, newF, m['F'])
            m['F'] = newF
        self._contain_failure(i)
        return self._note_change('struct', 1.0)

    def _update_gauss(self, i, newSw, fs, ws):
        """
        accept the weight update for a gauss object, whose smoothed
        model covariance is the weight, and update the matched flux.
        Returns the relative change
        """
        m = self.models[i]
        newF = _matched_flux(fs, ws, self.Sw[i], newSw)
        Twt = self.Sw[i][0, 0] + self.Sw[i][1, 1]
        change = max(
            self._note_change(
                'struct', np.abs(newSw - self.Sw[i]).max() / Twt,
            ),
            self._flux_change(i, newF, m['F']),
        )
        m['cov_sm'] = newSw
        m['F'] = newF
        self.nfail[i] = 0
        self.Sw[i] = newSw
        return change

    def _update_mixture(self, i, newSw, sums, pred, fs, fs_pred):
        """
        exp/dev: deweight-style update on the family covariance
        matrix, as in ngmix PAdmomFitter._run_admom_mixture.  Map
        both the measured and the model-predicted moments through
        the deweight transform and shift the family covariance by
        the difference.  For a single-gaussian family this is
        exactly the standard deweight update; for the mixture it has
        near-unit gain, unlike a plain Picard update on the weighted
        moments which converges at rate ~1/2.  The smoothing
        covariance cancels in the difference.  newSw is the deweight
        of the measured moments.  Returns the relative change
        """
        m = self.models[i]

        shift = self._mixture_shift(i, newSw, sums, pred)
        prop, shift, accepted, idamp = self._damped_step(i, shift)

        newF = m['F'] * fs / fs_pred
        change = self._flux_change(i, newF, m['F'])
        m['F'] = newF

        if not accepted:
            # no valid step: keep the previous structure and count a
            # failed update; the flux update above keeps the blend
            # from deadlocking
            self._count_skip(i)
            change = max(change, self._note_change('struct', 1.0))
            if self._contain_failure(i):
                # the weight was reset by the intervention
                return change
        elif idamp > 0:
            # a damped step can be small only because it was
            # shortened at the validity boundary, not because the
            # fit has settled; it also counts as a constrained
            # step for the windowed non-contraction check (a
            # boundary-hugging sawtooth may never take a fully
            # rejected step)
            change = max(change, self._note_change('struct', 1.0))
            m['cov'] = prop
            self.nfail[i] = 0
            self._win_nfail[i] += 1
        else:
            Twt = self.Sw[i][0, 0] + self.Sw[i][1, 1]
            change = max(change, self._note_change(
                'struct', np.abs(shift).max() / Twt,
            ))
            m['cov'] = prop
            self.nfail[i] = 0

        if m['type'] == 'bdf':
            # the split update runs under the pre-update weight so
            # the flux sums measured for the structure step can be
            # reused as its first aperture row; the one-sweep lag
            # vanishes at the fixed point like the other lags in
            # the iteration.  Between updates the last change is
            # carried so convergence waits for a settled split
            if self.isweep % BDF_SPLIT_EVERY == 0:
                self.bdf_last_dfd[i] = self._update_bdf_split(i, fs)
            change = max(change, self._note_change(
                'struct', self.bdf_last_dfd[i],
            ))

        self.Sw[i] = newSw
        return change

    def _update_bdf_split(self, i, fs1):
        """
        the per-sweep flux split update for a bdf object: a
        two-aperture linear solve for the component fluxes.  The
        apertures are the object's adaptive weight and the
        smoothing weight, whose different scales separate the exp
        and dev templates; the measured side is the
        neighbor-corrected flux sum under each aperture and the
        template side is closed form.  The first aperture's sums
        are reused from the structure step's measurement (fs1),
        so only the smoothing aperture needs a data pass.  The
        band-combined split is then optionally shrunk toward the
        prior (see the deblend docstring) before it updates the
        model; the component fluxes and the raw split are kept
        for the result.  Returns the absolute split change

        The shrinkage weight uses one-time per-aperture noise
        variances computed at the first call (the weight evolves
        during the fit, so this is approximate; it only sets the
        regularization strength)
        """
        m = self.models[i]
        Sfam = m['cov']
        vi, ui = self.positions[i]
        weights = [self.Sw[i], self.smooth_cov]
        parts = [
            {'type': 'exp', 'cov': Sfam,
             'F': np.ones(self.nband)},
            {'type': 'dev', 'cov': m['TdByTe'] * Sfam,
             'F': np.ones(self.nband)},
        ]

        ws = np.zeros(self.nband)
        for ep in self.epochs_per_obj[i]:
            ws[ep['band']] += ep['weight']

        # unit-flux template sums per aperture at detAtinv=1; the
        # per-band matrix is this times the band weight sum
        base = np.zeros((2, 2))
        for a, Sw in enumerate(weights):
            for c, part in enumerate(parts):
                base[a, c] = model_ksums(
                    part, 0, 0.0, 0.0, Sw, 1.0, self.Tsmooth,
                )[5]
        det = base[0, 0] * base[1, 1] - base[0, 1] * base[1, 0]
        if abs(det) < 1.0e-10 * abs(base[0, 0] * base[1, 1]):
            return 0.0

        # measured neighbor-corrected flux sums: the adaptive
        # aperture row is the reused structure-step measurement,
        # the smoothing aperture needs its own pass
        fs2 = np.zeros((2, self.nband))
        fs2[0] = fs1
        Sw = self.smooth_cov
        nsums = self._get_neighbor_sums(i, Sw=Sw)
        for ep in self.epochs_per_obj[i]:
            alpha, beta = get_phase_angles(
                ep, vi - ep['vcen'], ui - ep['ucen'],
            )
            admom_ksums(
                ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                alpha, beta, ep['kv'], ep['ku'],
                Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                self.esums,
            )
            csums = (
                self.esums - nsums[ep['band']] / ep['detAtinv']
            )
            fac = ep['weight'] * ep['detAtinv']
            fs2[1, ep['band']] += fac * csums[5]

        var2 = self._get_bdf_noise_vars(i, weights)

        F2 = np.zeros((self.nband, 2))
        fcovs = np.zeros((self.nband, 2, 2))
        for band in range(self.nband):
            Mb = base * ws[band]
            F2[band] = np.linalg.solve(Mb, fs2[:, band])
            Mbinv = np.linalg.inv(Mb)
            fcovs[band] = (
                Mbinv @ np.diag(var2[:, band]) @ Mbinv.T
            )

        E = F2[:, 0].sum()
        D = F2[:, 1].sum()
        S = E + D
        if S == 0:
            return 0.0
        fd_gls = D / S
        grad = np.array([-D, E]) / S ** 2
        fd_var = 0.0
        for band in range(self.nband):
            fd_var += grad @ fcovs[band] @ grad

        if self.fd_shrink[i] is None:
            newfd = fd_gls
        else:
            fd0, sigma0 = self.fd_shrink[i]
            if sigma0 == 0 or fd_var <= 0:
                newfd = fd0 if sigma0 == 0 else fd_gls
            else:
                w = 1.0 / fd_var
                w0 = 1.0 / sigma0 ** 2
                newfd = (fd_gls * w + fd0 * w0) / (w + w0)
        newfd = np.clip(newfd, -0.5, 1.5)

        change = abs(newfd - m['fracdev'])
        m['fracdev'] = newfd
        self.bdf_info[i] = {
            'F2': F2, 'fd_gls': fd_gls, 'fd_var': fd_var,
        }
        return change

    def _get_bdf_noise_vars(self, i, weights):
        """
        one-time per-aperture per-band noise variances of the flux
        sums, for the shrinkage weight (diagonal approximation:
        the cross covariance between the apertures is neglected,
        which underestimates their correlation but only affects
        the regularization strength)
        """
        if i not in self._bdf_noise_cache:
            vi, ui = self.positions[i]
            var2 = np.zeros((2, self.nband))
            fcov = np.zeros((6, 6))
            for a, Sw in enumerate(weights):
                for ep in self.epochs_per_obj[i]:
                    alpha, beta = get_phase_angles(
                        ep, vi - ep['vcen'], ui - ep['ucen'],
                    )
                    admom_finalize(
                        ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                        alpha, beta, ep['kv'], ep['ku'],
                        Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                        ep['err_fac2'],
                        self.esums, fcov,
                    )
                    fac = ep['weight'] * ep['detAtinv']
                    nfac = ep['df2'] ** 2
                    var2[a, ep['band']] += (
                        fac ** 2 * nfac * fcov[5, 5]
                    )
            self._bdf_noise_cache[i] = var2
        return self._bdf_noise_cache[i]

    def _bdf_split_response(self, i):
        """
        d fd_gls / d (M1, M2, T) of the family covariance at the
        model consistent point: the two-aperture solve of the
        converged model itself (closed form template sums on both
        sides), re-solved with the template matrix at perturbed
        structure.  Central differences over the mbasis.  The band
        weight sums cancel between the two sides, so they are
        omitted
        """
        m = self.models[i]
        info = self.bdf_info.get(i)
        Sfam = m['cov']
        Td = m['TdByTe']

        fam0 = np.array([
            Sfam[1, 1] - Sfam[0, 0], 2 * Sfam[0, 1],
            Sfam[0, 0] + Sfam[1, 1],
        ])
        h = 1.0e-6 * max((1.0 + Td) * fam0[2], 1.0e-3)

        def base_at(S):
            parts = [
                {'type': 'exp', 'cov': S,
                 'F': np.ones(self.nband)},
                {'type': 'dev', 'cov': Td * S,
                 'F': np.ones(self.nband)},
            ]
            base = np.zeros((2, 2))
            for a, Sw in enumerate((self.Sw[i], self.smooth_cov)):
                for c, part in enumerate(parts):
                    base[a, c] = model_ksums(
                        part, 0, 0.0, 0.0, Sw, 1.0, self.Tsmooth,
                    )[5]
            return base

        base0 = base_at(Sfam)
        # model-consistent measured side: the converged raw
        # components through the unperturbed template sums
        bmod = info['F2'] @ base0.T

        def split_at(famvec):
            bp = base_at(_mbasis_cov(*famvec))
            det = bp[0, 0] * bp[1, 1] - bp[0, 1] * bp[1, 0]
            if abs(det) < 1.0e-10 * abs(bp[0, 0] * bp[1, 1]):
                return None
            E = 0.0
            D = 0.0
            for band in range(self.nband):
                Fp = np.linalg.solve(bp, bmod[band])
                E += Fp[0]
                D += Fp[1]
            S = E + D
            return D / S if S != 0 else None

        G = np.zeros(3)
        for j in range(3):
            famp = fam0.copy()
            famm = fam0.copy()
            famp[j] += h
            famm[j] -= h
            fp = split_at(famp)
            fm = split_at(famm)
            if fp is None or fm is None:
                return None
            G[j] = (fp - fm) / (2 * h)
        return G

    def _bdf_error_terms(self, i, fvar_raw, fmcov):
        """
        the joint-sandwich inputs for a bdf object at the converged
        state: the split response G, the shrinkage factor k, the
        full noise variance of the raw split and its cross
        covariance with the object's moment and flux sums.

        The split noise is a linear functional of the same modes
        as the moment sums: eta = sum_band w_band . (dfs1, dfs2)
        with w_band the solve-gradient row and dfs1/dfs2 the flux
        sums under the adaptive and smoothing apertures.  The
        adaptive-aperture crosses are the fvar_raw/fmcov entries
        already accumulated; the smoothing aperture needs one
        cross-aperture kernel overlap pass (its kernel times the
        adaptive-weight moment kernels times the noise power).
        The same pass gives the aperture cross covariance, so the
        returned split variance is the full one, not the diagonal
        approximation used for the regularization strength.

        Returns (G, k, fd_var, eta_scov, eta_fcovs) or None when
        the terms cannot be evaluated
        """
        info = self.bdf_info.get(i)
        if info is None:
            return None

        F2 = info['F2']
        E = F2[:, 0].sum()
        D = F2[:, 1].sum()
        S = E + D
        if S == 0:
            return None
        grad = np.array([-D, E]) / S ** 2

        G = self._bdf_split_response(i)
        if G is None:
            return None

        Sw = self.Sw[i]
        Sm = self.smooth_cov

        # cross-aperture kernel overlaps: the smoothing-aperture
        # flux kernel against the adaptive-weight (M1, M2, T, flux)
        # kernels, and its own square, times the noise power
        X2 = np.zeros((self.nband, 4))
        var22 = np.zeros(self.nband)
        for ep in self.epochs_per_obj[i]:
            kv = ep['kv']
            ku = ep['ku']
            Sv = Sw[0, 0] * kv + Sw[0, 1] * ku
            Su = Sw[0, 1] * kv + Sw[1, 1] * ku
            chi2 = kv * Sv + ku * Su
            wk1 = np.exp(-0.5 * np.clip(chi2, 0, FASTEXP_MAX_CHI2))
            wk1[(chi2 > FASTEXP_MAX_CHI2) | (chi2 < 0)] = 0.0
            vvk = (Sw[0, 0] - Sv * Sv) * wk1
            vuk = (Sw[0, 1] - Sv * Su) * wk1
            uuk = (Sw[1, 1] - Su * Su) * wk1
            kern = (uuk - vvk, 2 * vuk, uuk + vvk, wk1)

            Sv2 = Sm[0, 0] * kv + Sm[0, 1] * ku
            Su2 = Sm[0, 1] * kv + Sm[1, 1] * ku
            chi22 = kv * Sv2 + ku * Su2
            wk2 = np.exp(
                -0.5 * np.clip(chi22, 0, FASTEXP_MAX_CHI2)
            )
            wk2[(chi22 > FASTEXP_MAX_CHI2) | (chi22 < 0)] = 0.0
            w2ef = wk2 * ep['err_fac2']

            fac2 = (
                (ep['weight'] * ep['detAtinv']) ** 2
                * ep['df2'] ** 2
            )
            band = ep['band']
            for c in range(4):
                X2[band, c] += fac2 * np.sum(w2ef * kern[c])
            var22[band] += fac2 * np.sum(w2ef * wk2)

        # the converged template matrix per band
        m = self.models[i]
        base = np.zeros((2, 2))
        parts = [
            {'type': 'exp', 'cov': m['cov'],
             'F': np.ones(self.nband)},
            {'type': 'dev', 'cov': m['TdByTe'] * m['cov'],
             'F': np.ones(self.nband)},
        ]
        for a, Swa in enumerate((Sw, Sm)):
            for c, part in enumerate(parts):
                base[a, c] = model_ksums(
                    part, 0, 0.0, 0.0, Swa, 1.0, self.Tsmooth,
                )[5]

        ws = np.zeros(self.nband)
        for ep in self.epochs_per_obj[i]:
            ws[ep['band']] += ep['weight']

        eta_scov = np.zeros(3)
        eta_fcovs = np.zeros(self.nband)
        fd_var = 0.0
        for band in range(self.nband):
            Mb = base * ws[band]
            try:
                wb = np.linalg.inv(Mb).T @ grad
            except np.linalg.LinAlgError:
                return None
            C2 = np.array([
                [fvar_raw[band], X2[band, 3]],
                [X2[band, 3], var22[band]],
            ])
            fd_var += wb @ C2 @ wb
            eta_scov += wb[0] * fmcov[band] + wb[1] * X2[band, :3]
            eta_fcovs[band] = (
                wb[0] * fvar_raw[band] + wb[1] * X2[band, 3]
            )
        if not fd_var > 0:
            return None
        info['fd_var_full'] = fd_var

        # the shrinkage factor the estimator actually applied,
        # from the same variance the update used
        shrink = self.fd_shrink[i]
        if shrink is None:
            k = 1.0
        else:
            _, sigma0 = shrink
            if sigma0 == 0:
                k = 0.0
            elif info['fd_var'] > 0:
                k = sigma0 ** 2 / (sigma0 ** 2 + info['fd_var'])
            else:
                k = 1.0
        return G, k, fd_var, eta_scov, eta_fcovs

    def _mixture_shift(self, i, newSw, sums, pred):
        """
        the proposed shift of the family covariance: the difference
        of the deweighted measured and predicted moments.  When the
        predicted moments do not admit a deweight, fall back to a
        gain-1 update on the weighted moment ratios, composed in
        matrix form: scale by the T ratio and shift the anisotropy
        by the ratio differences
        """
        Sp, pflags = deweight(_moment_matrix(pred), self.Sw[i])
        if pflags == 0:
            return newSw - Sp

        Sfam = self.models[i]['cov']
        Tp = pred[4] * (1.0 / pred[5])
        Tf = Sfam[0, 0] + Sfam[1, 1]
        fac = sums[4] / sums[5] / Tp
        de1 = sums[2] / sums[4] - pred[2] / pred[4]
        de2 = sums[3] / sums[4] - pred[3] / pred[4]
        return (fac - 1) * Sfam \
            + 0.5 * fac * Tf * np.array([
                [-de1, de2],
                [de2, de1],
            ])

    def _damped_step(self, i, shift):
        """
        the largest step from the family covariance, damping if
        needed, for which the smoothed components stay valid under
        the zero weight.  Returns (proposed, shift, accepted, idamp)
        """
        m = self.models[i]
        accepted = False
        for idamp in range(10):
            prop = m['cov'] + shift
            valid = mixture_model_valid(
                m, prop, ZERO_WEIGHT, self.Tsmooth,
            )
            if valid:
                accepted = True
                break
            shift = 0.5 * shift
        return prop, shift, accepted, idamp

    def _count_skip(self, i):
        """
        count a skipped structure update against the group-level
        backstop limit
        """
        self.nskip += 1
        if self.nskip > 100 * self.nobj:
            raise RuntimeError(
                f'too many failed structure updates, object {i}'
            )

    def _contain_failure(self, i, force=False):
        """
        count a consecutive failed structure update for object i.
        At NFAIL_LIMIT failures, restart the object from the compact
        delta state, where the neighbor contamination that drives
        weight runaways is minimized, so a transient runaway can
        recover and re-grow.  If a restarted object fails again,
        demote it permanently to a fixed point source, whose linear
        flux update is always well defined and which errs by
        under-subtracting wings rather than mis-subtracting a
        nonsense extended model.  With force the escalation is
        immediate, skipping the consecutive-failure count: the
        windowed non-contraction check uses this because
        sawtoothing objects take accepted steps between failed
        ones, resetting the counter, and the window of
        non-contraction is already the evidence of a cycle.
        Returns True when it intervened
        """
        self.nfail[i] += 1
        if not force:
            # a real rejected step, seen by the windowed
            # non-contraction check (the forced call is that
            # check itself, not a step)
            self._win_nfail[i] += 1
            if self.nfail[i] < NFAIL_LIMIT:
                return False
        self.nfail[i] = 0
        m = self.models[i]
        self.Sw[i] = self.smooth_cov.copy()
        if self.recenter:
            # the wandering center may be part of the runaway
            self.positions[i] = self.det_positions[i]
        if self.nrestart[i] == 0:
            self.nrestart[i] = 1
            self.dbflags[i] |= RESTARTED
            if m['type'] in ('exp', 'dev', 'bdf'):
                m['cov'] = np.zeros((2, 2))
                if m['type'] == 'bdf':
                    m['fracdev'] = self.fd_init[i]
                    self.bdf_last_dfd[i] = 1.0
            else:
                m['cov_sm'] = self.smooth_cov.copy()
            # the restart is a discontinuity in the sweep map
            self.hist = []
            self._reset_change_hist()
        else:
            self.dbflags[i] |= DEBLENDED_AS_PSF
            m['type'] = 'star'
            m.pop('cov', None)
            m['cov_sm'] = self.smooth_cov.copy()
            # the packed state layout changed
            self.hist = []
            self.scales = None
            self._reset_change_hist()
        return True

    def _extrapolate(self):
        """
        guarded Steffensen boost on the packed global state: three
        consecutive plain sweeps give the contraction ratio of the
        dominant mode and the remaining geometric series is applied
        in one step, rolled back if it leaves the valid region.
        Convergence is always decided by a subsequent plain sweep.
        See ngmix.prepsfadmom PAdmomFitter._run_admom_mixture for
        the Aitken/Steffensen/Sidi references
        """
        self.hist.append(self._pack_state())
        if len(self.hist) < 3:
            return

        d1 = self.hist[-2] - self.hist[-3]
        d2 = self.hist[-1] - self.hist[-2]
        denom = d1 @ d1
        rho = (d2 @ d1) / denom if denom > 0 else 0.0
        # accept ratios up to 0.998: the ultra-slow modes
        # (near-degenerate component pairs contract at
        # rho ~ 0.9975) are exactly the ones that need the
        # boost; the validity rollback guards the large jump
        if 0.2 < rho < 0.998:
            saved_models = [dict(m) for m in self.models]
            saved_Sw = [sw.copy() for sw in self.Sw]
            saved_pos = list(self.positions)
            # a full jump amplifies every component of the step by
            # rho/(1-rho); near-unit ratios can push a single
            # component (a clipped center, a covariance edge) out
            # of the valid region.  Backing off to a partial boost
            # still collapses most of the geometric tail, where a
            # plain rollback would retry the same doomed jump
            # every trio and crawl at the unboosted rate
            accepted = False
            for frac in (1.0, 0.5, 0.25):
                self._unpack_state(
                    self.hist[-1] + d2 * frac * rho / (1 - rho)
                )
                if self._state_valid():
                    accepted = True
                    break
                for m, sm in zip(self.models, saved_models):
                    m.update(sm)
                for k in range(self.nobj):
                    self.Sw[k] = saved_Sw[k]
                self.positions = saved_pos
            if accepted:
                # a fresh trio of plain sweeps is needed for the
                # next ratio estimate
                self.hist = []
                self._reset_change_hist()
        if len(self.hist) > 3:
            self.hist = self.hist[-3:]

    def _pack_state(self):
        """
        the global deblend state as a normalized vector, for the
        sweep map extrapolation.  The layout is the concatenation
        over objects, in order, of

            F[0], ..., F[nband-1]         per-band fluxes
            C[0, 0], C[0, 1], C[1, 1]     model covariance
            Sw[0, 0], Sw[0, 1], Sw[1, 1]  weight covariance

        where C is cov_sm for a gauss object and the family cov for
        exp/dev.  Star weights and covariances are frozen and only
        their fluxes enter, so the vector length depends on the
        current type of every object; a demotion changes the layout
        and resets the scales and history.  With recentering
        the center offsets from the detection positions follow
        the fluxes for every type, packed with a +1 offset so their
        scale is O(1) near zero.  Each component is divided by a
        per-component scale fixed on the first call, so the sweep
        map differences are comparable across fluxes and
        covariances
        """
        x = []
        for i, (m, sw) in enumerate(zip(self.models, self.Sw)):
            x.extend(m['F'])
            if self.recenter:
                v0, u0 = self.det_positions[i]
                x.extend([
                    self.positions[i][0] - v0 + 1.0,
                    self.positions[i][1] - u0 + 1.0,
                ])
            if m['type'] == 'gauss':
                x.extend([
                    m['cov_sm'][0, 0], m['cov_sm'][0, 1],
                    m['cov_sm'][1, 1],
                ])
            elif m['type'] in ('exp', 'dev'):
                x.extend([
                    m['cov'][0, 0], m['cov'][0, 1], m['cov'][1, 1],
                ])
            elif m['type'] == 'bdf':
                # the split is packed with an offset so its scale
                # is O(1) even when it converges near zero
                x.extend([
                    m['cov'][0, 0], m['cov'][0, 1], m['cov'][1, 1],
                    m['fracdev'] + 2.0,
                ])
            if m['type'] != 'star':
                x.extend([sw[0, 0], sw[0, 1], sw[1, 1]])
        x = np.array(x)
        if self.scales is None:
            self.scales = np.maximum(np.abs(x), 1.0e-10)
        return x / self.scales

    def _unpack_state(self, x):
        """
        write a packed state vector back into the models and
        weights, inverting the layout described in _pack_state
        """
        x = x * self.scales
        k = 0
        for i, m in enumerate(self.models):
            nband = m['F'].size
            m['F'] = x[k:k + nband].copy()
            k += nband
            if self.recenter:
                v0, u0 = self.det_positions[i]
                self.positions[i] = (
                    v0 + x[k] - 1.0, u0 + x[k + 1] - 1.0,
                )
                k += 2
            if m['type'] == 'gauss':
                m['cov_sm'] = np.array([
                    [x[k], x[k + 1]], [x[k + 1], x[k + 2]],
                ])
                k += 3
            elif m['type'] in ('exp', 'dev'):
                m['cov'] = np.array([
                    [x[k], x[k + 1]], [x[k + 1], x[k + 2]],
                ])
                k += 3
            elif m['type'] == 'bdf':
                m['cov'] = np.array([
                    [x[k], x[k + 1]], [x[k + 1], x[k + 2]],
                ])
                m['fracdev'] = x[k + 3] - 2.0
                k += 4
            if m['type'] != 'star':
                self.Sw[i] = np.array([
                    [x[k], x[k + 1]], [x[k + 1], x[k + 2]],
                ])
                k += 3

    def _state_valid(self):
        """
        every weight and model in the state gives well defined
        sums, and with recentering every center is inside
        its displacement clip box
        """
        for i, (m, sw) in enumerate(zip(self.models, self.Sw)):
            if sw[0, 0] <= 0 or sw[1, 1] <= 0 or det2(sw) <= 0:
                return False
            if m['type'] in ('exp', 'dev', 'bdf'):
                if not mixture_model_valid(
                        m, m['cov'], ZERO_WEIGHT,
                        self.Tsmooth):
                    return False
            elif m['type'] == 'gauss':
                if det2(m['cov_sm']) <= 0:
                    return False
            if self.recenter:
                v0, u0 = self.det_positions[i]
                d2 = (
                    (self.positions[i][0] - v0) ** 2
                    + (self.positions[i][1] - u0) ** 2
                )
                if d2 > (RECENTER_CLIP_FAC ** 2) * self.Tsmooth:
                    return False
        return True

    def _get_object_sums(self, i):
        """
        neighbor-corrected weighted moment sums for object i,
        accumulated over the object's epochs, plus the model's own
        predicted sums for exp/dev objects.  Returns
        (sums, fs, ws, pred, fs_pred) with fs, ws, fs_pred per band
        """
        vi, ui = self.positions[i]
        is_mix = self.models[i]['type'] in ('exp', 'dev', 'bdf')
        Sw = self.Sw[i]

        base_nsums = self._get_neighbor_sums(i)
        if is_mix:
            base_psums = self._get_predicted_sums(i)

        sums = np.zeros(6)
        fs = np.zeros(self.nband)
        ws = np.zeros(self.nband)
        pred = np.zeros(6)
        fs_pred = np.zeros(self.nband)

        for ep in self.epochs_per_obj[i]:
            alpha, beta = get_phase_angles(
                ep, vi - ep['vcen'], ui - ep['ucen'],
            )
            admom_ksums(
                ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                alpha, beta, ep['kv'], ep['ku'],
                Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                self.esums,
            )
            csums = (
                self.esums - base_nsums[ep['band']] / ep['detAtinv']
            )
            fac = ep['weight'] * ep['detAtinv']
            sums += fac * csums
            fs[ep['band']] += fac * csums[5]
            ws[ep['band']] += ep['weight']

            if is_mix:
                psums = base_psums[ep['band']] / ep['detAtinv']
                pred += fac * psums
                fs_pred[ep['band']] += fac * psums[5]

        return sums, fs, ws, pred, fs_pred

    def _get_neighbor_sums(self, i, Sw=None):
        """
        per-band weighted sums of the neighbor and fixed external
        models under object i's weight (or the given weight), at
        detAtinv=1.  The model sums scale exactly as 1/detAtinv,
        so expand the components once, run the kernel once per
        band, and rescale per epoch; the fixed externals are
        subtracted exactly like in-group neighbors and join the
        same kernel call
        """
        vi, ui = self.positions[i]
        if Sw is None:
            Sw = self.Sw[i]

        ncomps = []
        for j in range(self.nobj):
            if j == i:
                continue
            fracs, So00, So01, So11 = model_comps(
                self.models[j], self.Tsmooth,
            )
            ncomps.append((
                self.positions[j], self.models[j]['F'],
                fracs, So00, So01, So11,
            ))
        for p, fm in zip(self.fpositions, self.fmodels):
            fracs, So00, So01, So11 = model_comps(fm, self.Tsmooth)
            ncomps.append((p, fm['F'], fracs, So00, So01, So11))

        base_nsums = np.zeros((self.nband, 6))
        if ncomps:
            nSo00 = np.concatenate([c[3] for c in ncomps])
            nSo01 = np.concatenate([c[4] for c in ncomps])
            nSo11 = np.concatenate([c[5] for c in ncomps])
            ndv = np.concatenate([
                np.full(c[2].size, c[0][0] - vi) for c in ncomps
            ])
            ndu = np.concatenate([
                np.full(c[2].size, c[0][1] - ui) for c in ncomps
            ])
            for band in range(self.nband):
                nF = np.concatenate([
                    c[1][band] * c[2] for c in ncomps
                ])
                gauss_comps_ksums(
                    nF, nSo00, nSo01, nSo11, ndv, ndu,
                    Sw[0, 0], Sw[0, 1], Sw[1, 1], 1.0,
                    base_nsums[band],
                )
        return base_nsums

    def _get_predicted_sums(self, i):
        """
        per-band weighted sums predicted by object i's own model
        under its weight, at detAtinv=1
        """
        m = self.models[i]
        Sw = self.Sw[i]

        base_psums = np.zeros((self.nband, 6))
        fracs, So00, So01, So11 = model_comps(m, self.Tsmooth)
        zeros = np.zeros(fracs.size)
        for band in range(self.nband):
            gauss_comps_ksums(
                m['F'][band] * fracs, So00, So01, So11,
                zeros, zeros,
                Sw[0, 0], Sw[0, 1], Sw[1, 1], 1.0,
                base_psums[band],
            )
        return base_psums

    def _get_object_result(self, i):
        """
        the result dict for object i at the converged state; see
        deblend for the entries
        """
        m = self.models[i]

        sums_i, fs, ws, _, _ = self._get_object_sums(i)
        fvar_raw, fmcov, covj = self._accumulate_error_sums(i)
        fvar, fam_cov, fcov_raw, gfvar, gfam_cov, fd_var_tot = (
            self._run_sandwiches(
                i, sums_i, covj, fs, fvar_raw, fmcov,
            )
        )
        flux_err, s2n = _flux_errors(m['F'], fs, fvar)

        res = {
            'type': m['type'],
            'deblend_flags': int(self.dbflags[i]),
            'flux': m['F'].copy(),
            'flux_err': flux_err,
            'flux_cov': _flux_cov_phys(m['F'], fs, fcov_raw),
            's2n': s2n,
            'cen': np.array(self.positions[i]),
            'cen_pull': self.cen_pull[i],
        }
        self._set_shape(res, i, fam_cov)
        self._set_gauss_entries(res, i, fs, ws, gfvar, gfam_cov)

        if m['type'] == 'bdf':
            res['fracdev'] = m['fracdev']
            res['TdByTe'] = m['TdByTe']
            res['fracdev_err'] = (
                np.sqrt(fd_var_tot)
                if fd_var_tot is not None and fd_var_tot > 0
                else np.nan
            )
            info = self.bdf_info.get(i)
            if info is not None:
                # the full split noise variance from the error
                # pass when available, else the diagonal
                # approximation used for the regularization
                fdv = info.get('fd_var_full', info['fd_var'])
                res['fracdev_gls'] = info['fd_gls']
                res['fracdev_gls_err'] = (
                    np.sqrt(fdv) if fdv > 0 else np.nan
                )
                res['flux_exp'] = info['F2'][:, 0].copy()
                res['flux_dev'] = info['F2'][:, 1].copy()
            else:
                res['fracdev_gls'] = np.nan
                res['fracdev_gls_err'] = np.nan
                res['flux_exp'] = np.full(self.nband, np.nan)
                res['flux_dev'] = np.full(self.nband, np.nan)
        return res

    def _accumulate_error_sums(self, i):
        """
        noise propagation at the converged weight: the neighbor
        corrections are deterministic, so the raw kernel cross sums
        give the covariances of the corrected sums.  Returns the
        per-band flux variances, the per-band flux-structure
        covariances and the covariance of the combined sums
        """
        vi, ui = self.positions[i]
        Sw = self.Sw[i]

        fvar = np.zeros(self.nband)
        fmcov = np.zeros((self.nband, 3))
        covj = np.zeros((6, 6))
        fcov = np.zeros((6, 6))
        for ep in self.epochs_per_obj[i]:
            alpha, beta = get_phase_angles(
                ep, vi - ep['vcen'], ui - ep['ucen'],
            )
            admom_finalize(
                ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                alpha, beta, ep['kv'], ep['ku'],
                Sw[0, 0], Sw[0, 1], Sw[1, 1], ep['df2'],
                ep['err_fac2'],
                self.esums, fcov,
            )
            fac = ep['weight'] * ep['detAtinv']
            nfac = ep['df2'] ** 2
            band = ep['band']
            fvar[band] += fac ** 2 * nfac * fcov[5, 5]
            fmcov[band] += fac ** 2 * nfac * fcov[2:5, 5]
            covj += fac ** 2 * nfac * fcov
        return fvar, fmcov, covj

    def _run_sandwiches(self, i, sums_i, covj, fs, fvar_raw, fmcov):
        """
        for the weight-adaptive types the sandwich over the moment
        matching conditions (ngmix model_sandwich) gives the flux
        variances including the weight and family responses, plus
        the family covariance for the structure errors; for a gauss
        object it reduces exactly to the analytic delta method.
        Star weights are frozen, so the fixed weight flux variance
        is exact and there are no structure errors.  Also returns
        the gauss-estimator analogs under the same weight, for the
        gauss entries; for a gauss object the sandwiches coincide.

        For a bdf object the joint sandwich over the coupled
        (structure, split) estimating equations is used, including
        the cross covariance of the split noise with the moment
        sums; it also yields the total split variance.  The
        conditional model sandwich is the fallback when the joint
        terms cannot be evaluated

        Returns
        -------
        fvar, fam_cov, fcov_raw, gfvar, gfam_cov, fd_var_tot;
        fcov_raw is the full cross-band covariance of the flux
        sums from the shared family response (None on the star,
        bdf-joint and fallback paths)
        """
        m = self.models[i]

        fvar = fvar_raw
        fam_cov = None
        fcov_raw = None
        gfvar = None
        gfam_cov = None
        fd_var_tot = None
        if m['type'] != 'star' and sums_i[5] > 0:
            if m['type'] == 'gauss':
                mtype = 'gauss'
                Sfam = m['cov_sm'] - self.smooth_cov
            elif m['type'] == 'bdf':
                # the spec dict carries the split state
                mtype = m
                Sfam = m['cov']
            else:
                mtype = m['type']
                Sfam = m['cov']
            fvar = None
            if m['type'] == 'bdf':
                terms = self._bdf_error_terms(i, fvar_raw, fmcov)
                if terms is not None:
                    G, k, fdv, eta_scov, eta_fcovs = terms
                    fvar, fam_cov, fd_var_tot = bdf_joint_sandwich(
                        m, self.Sw[i], self.Tsmooth,
                        sums_i, covj, fs, fvar_raw, fmcov,
                        split_grad=G, shrink_k=k,
                        fd_var_data=fdv,
                        eta_scov=eta_scov, eta_fcovs=eta_fcovs,
                    )
            if fvar is None:
                fd_var_tot = None
                fvar, fam_cov, fcov_raw = model_sandwich(
                    mtype, Sfam, self.Sw[i], self.Tsmooth,
                    sums_i, covj, fs, fvar_raw, fmcov,
                )
            if fvar is None:
                # the sandwich could not be evaluated; fall back
                # to the fixed weight variances, with the
                # structure errors flagged downstream
                fvar = fvar_raw
                fam_cov = None
                fcov_raw = None
            if mtype == 'gauss':
                # the weight equals the gauss family covariance, so
                # the sandwiches coincide
                gfvar = fvar
                gfam_cov = fam_cov
            else:
                # gauss-estimator errors under the same weight, for
                # the low-noise shape entries
                gfvar, gfam_cov, _ = model_sandwich(
                    'gauss', self.Sw[i] - self.smooth_cov,
                    self.Sw[i], self.Tsmooth,
                    sums_i, covj, fs, fvar_raw, fmcov,
                )
                if gfvar is None:
                    gfvar = fvar_raw
                    gfam_cov = None
        return fvar, fam_cov, fcov_raw, gfvar, gfam_cov, fd_var_tot

    def _set_shape(self, res, i, fam_cov):
        """
        the family structure entries T, e1, e2 and their errors from
        the family covariance sandwich.  e_flags == 0 iff the
        ellipticities and their errors are usable, following the
        ngmix prepsfadmom convention
        """
        m = self.models[i]

        res['e_flags'] = 0
        res['e1'] = np.nan
        res['e2'] = np.nan
        if m['type'] == 'star':
            # a delta function has no shape by construction
            res['T'] = 0.0
            shape_ok = False
        else:
            if m['type'] == 'gauss':
                S = m['cov_sm'] - self.smooth_cov
            else:
                # the family covariance can scatter out of positive
                # definite, where the shape is undefined
                S = m['cov']
            res['T'], res['e1'], res['e2'], shape_ok = \
                _shape_from_cov(S)
        if not shape_ok:
            res['e_flags'] |= ngmix.flags.NONPOS_SIZE

        res['T_err'] = np.nan
        res['e1_err'] = np.nan
        res['e2_err'] = np.nan
        if fam_cov is not None:
            if fam_cov[2, 2] > 0:
                res['T_err'] = np.sqrt(fam_cov[2, 2])
            if shape_ok:
                res['e1_err'], res['e2_err'], eflags = _shape_errors(
                    res['e1'], res['e2'], res['T'], fam_cov,
                )
                res['e_flags'] |= eflags
        elif shape_ok:
            # no error propagation was possible for a shape that is
            # otherwise defined
            res['e_flags'] |= ngmix.flags.NONPOS_SHAPE_VAR

    def _set_gauss_entries(self, res, i, fs, ws, gfvar, gfam_cov):
        """
        gauss-estimator entries from the converged weight.  The
        weight iteration is exactly the adaptive-moments gauss fixed
        point on the neighbor-corrected data (the family state only
        enters through the matching conditions), so the lowest-noise
        gauss shape estimator is available for every model type at
        no extra fitting cost: the family models do the subtraction,
        the gauss weight does the measurement, and the metacal
        response calibrates the estimator.  The gauss-aperture
        fluxes and flux s/n are the analogs of the gauss-model
        deblender's outputs, for selection studies against the
        family quantities.  The primary fluxes should come from the
        family models; for a gauss object these equal the primary
        entries
        """
        m = self.models[i]

        res['gauss_T'] = np.nan
        res['gauss_e1'] = np.nan
        res['gauss_e2'] = np.nan
        res['gauss_T_err'] = np.nan
        res['gauss_e1_err'] = np.nan
        res['gauss_e2_err'] = np.nan
        res['gauss_e_flags'] = 0
        if m['type'] == 'star':
            res['gauss_e_flags'] |= ngmix.flags.NONPOS_SIZE
        else:
            Sgal_w = self.Sw[i] - self.smooth_cov
            res['gauss_T'], res['gauss_e1'], res['gauss_e2'], gok = \
                _shape_from_cov(Sgal_w)
            if not gok:
                res['gauss_e_flags'] |= ngmix.flags.NONPOS_SIZE
            if gfam_cov is not None:
                if gfam_cov[2, 2] > 0:
                    res['gauss_T_err'] = np.sqrt(gfam_cov[2, 2])
                if gok:
                    res['gauss_e1_err'], res['gauss_e2_err'], \
                        geflags = _shape_errors(
                            res['gauss_e1'], res['gauss_e2'],
                            res['gauss_T'], gfam_cov,
                        )
                    res['gauss_e_flags'] |= geflags
            elif gok:
                res['gauss_e_flags'] |= ngmix.flags.NONPOS_SHAPE_VAR

        res['gauss_flux'] = np.full(self.nband, np.nan)
        res['gauss_flux_err'] = np.full(self.nband, np.nan)
        res['gauss_s2n'] = np.nan
        if m['type'] != 'star' and gfvar is not None:
            Fg = fs / ws * 4 * np.pi * np.sqrt(det2(self.Sw[i]))
            res['gauss_flux'] = Fg
            res['gauss_flux_err'], res['gauss_s2n'] = _flux_errors(
                Fg, fs, gfvar,
            )


def _convert_fixed_models(fixed_models, nband, Tsmooth):
    """
    internal (positions, models) lists for the fixed external
    sources; entries carry v, u, type, flux and for non-star types
    the pre-psf e1, e2, T.  Nonfinite parameters raise: a poisoned
    fixed model would silently corrupt every subtraction
    """
    if not fixed_models:
        return [], []
    smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])
    positions = []
    models = []
    for f in fixed_models:
        F = np.atleast_1d(np.array(f['flux'], dtype='f8'))
        if F.size != nband:
            raise ValueError(
                f'fixed model flux has {F.size} bands, expected '
                f'{nband}'
            )
        ftype = f.get('type', 'gauss')
        if ftype == 'star':
            m = {'type': 'star', 'cov_sm': smooth_cov.copy(), 'F': F}
        elif ftype in ('gauss', 'exp', 'dev', 'bdf'):
            pars = [f['e1'], f['e2'], f['T']]
            if ftype == 'bdf':
                pars = pars + [f['fracdev'], f['TdByTe']]
            if not np.all(np.isfinite(pars)):
                raise ValueError(
                    f'nonfinite fixed model parameters: {f}'
                )
            cov = cov_from_e(f['e1'], f['e2'], f['T'])
            if ftype == 'gauss':
                m = {
                    'type': 'gauss',
                    'cov_sm': cov + smooth_cov,
                    'F': F,
                }
            elif ftype == 'bdf':
                m = {
                    'type': 'bdf', 'cov': cov, 'F': F,
                    'fracdev': f['fracdev'],
                    'TdByTe': f['TdByTe'],
                }
            else:
                m = {'type': ftype, 'cov': cov, 'F': F}
        else:
            raise ValueError(f"bad fixed model type: '{ftype}'")
        positions.append((f['v'], f['u']))
        models.append(m)
    return positions, models


def _moment_matrix(sums):
    """
    the 2x2 second moment matrix from the weighted moment sums
    """
    finv = 1.0 / sums[5]
    M1 = sums[2] * finv
    M2 = sums[3] * finv
    T = sums[4] * finv
    return np.array([
        [0.5 * (T - M1), 0.5 * M2],
        [0.5 * M2, 0.5 * (T + M1)],
    ])


def _matched_flux(fs, ws, Sigma, cov):
    """
    the matched-aperture flux from the per-band flux sums and weight
    normalizations, for a gaussian model covariance under a gaussian
    weight
    """
    return fs / ws * 2 * np.pi * np.sqrt(det2(Sigma + cov))


def _flux_cov_phys(F, fs, fcov_raw):
    """
    the physical cross-band flux covariance from the raw flux-sum
    covariance, with the same per-band normalization as
    _flux_errors (diag equals flux_err ** 2 where defined).
    None in, None out
    """
    if fcov_raw is None:
        return None
    scale = np.zeros(F.size)
    wgood = fs != 0
    scale[wgood] = F[wgood] / fs[wgood]
    return np.outer(scale, scale) * fcov_raw


def _flux_errors(F, fs, fvar):
    """
    per-band flux errors and the combined flux s/n from the flux
    sums and their variances.  Bands with no positive variance or a
    zero flux sum are nan, and the s/n is nan when no band is usable
    """
    flux_err = np.full(F.size, np.nan)
    wgood = (fvar > 0) & (fs != 0)
    flux_err[wgood] = np.abs(
        F[wgood] / fs[wgood],
    ) * np.sqrt(fvar[wgood])
    if np.any(wgood):
        s2n = np.sqrt(
            np.sum(fs[wgood] ** 2 / fvar[wgood]),
        )
    else:
        s2n = np.nan
    return flux_err, s2n


def _shape_from_cov(S):
    """
    T, e1, e2 and shape definedness from a covariance matrix.  The
    shape is defined for det > 0 with positive trace, which is
    |e| < 1: a positive-size object with a degenerate covariance has
    no defined shape.  e1 and e2 are nan when undefined
    """
    T = S[0, 0] + S[1, 1]
    ok = T > 0 and det2(S) > 0
    if ok:
        e1 = (S[1, 1] - S[0, 0]) / T
        e2 = 2 * S[0, 1] / T
    else:
        e1 = np.nan
        e2 = np.nan
    return T, e1, e2, ok


def _shape_errors(e1, e2, T, fam_cov):
    """
    delta-method errors of e1, e2 from the family covariance
    sandwich.  When either implied variance is not positive the
    errors are nan and NONPOS_SHAPE_VAR is returned in the flags
    """
    ev1 = (
        fam_cov[0, 0]
        - 2 * e1 * fam_cov[0, 2]
        + e1 ** 2 * fam_cov[2, 2]
    ) / T ** 2
    ev2 = (
        fam_cov[1, 1]
        - 2 * e2 * fam_cov[1, 2]
        + e2 ** 2 * fam_cov[2, 2]
    ) / T ** 2
    if ev1 > 0 and ev2 > 0:
        return np.sqrt(ev1), np.sqrt(ev2), 0
    return np.nan, np.nan, ngmix.flags.NONPOS_SHAPE_VAR


def _fchange(newF, oldF, scale):
    """
    maximum flux change relative to a FIXED per-band scale (the
    ratcheted historical maximum), not the current flux: a
    component converging toward or oscillating through zero flux
    would never satisfy a current-relative tolerance and would
    run the group to maxiter
    """
    return (np.abs(newF - oldF) / scale).max()
