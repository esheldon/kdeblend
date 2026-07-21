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
from ngmix.prepsfadmom.errors import model_sandwich
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums, admom_finalize

from ngmix.prepsfadmom.models import (
    det2, cov_from_e, model_ksums, model_comps, mixture_model_valid,
)
from ngmix.prepsfadmom.models_nb import gauss_comps_ksums

DEFAULT_TGUESS = 0.5
DEFAULT_MAXITER = 1000
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
                'gauss' (default), 'star', 'exp', or 'dev'.  Stars
                are pre-psf delta functions with only their fluxes
                fit.
            Tguess: float, optional
                initial pre-psf T, default 0.5; ignored for stars
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
        Maximum number of Gauss-Seidel sweeps, default 1000.
    tol: float, optional
        Convergence tolerance on the maximum relative parameter change
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
        deblend result.  Nonfinite parameters raise.

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
            bands), s2n (the flux s/n combined over bands in
            quadrature), cen, cen_pull.  Also gauss_T, gauss_e1,
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

    fwhm_smooth = choose_fwhm_smooth(
        mbobs, fwhm_smooth=fwhm_smooth, smooth_fac=smooth_fac, rng=rng,
    )
    Tsmooth = fwhm_to_T(fwhm_smooth) if fwhm_smooth > 0 else 0.0

    epochs = []
    for band, obslist in enumerate(mbobs):
        for tobs in obslist:
            ep = prep_epoch(
                tobs, band=band, fwhm_smooth=fwhm_smooth,
                ap_rad=ap_rad, use_noise_image=use_noise_image,
            )
            ep['vcen'] = 0.0
            ep['ucen'] = 0.0
            epochs.append(ep)

    epochs_per_obj = [epochs] * len(objects)
    return _deblend_core(
        epochs_per_obj, nband, objects, fwhm_smooth, Tsmooth,
        maxiter, tol, fixed_models=fixed_models,
    )


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
    rng, fixed_models: optional
        As for deblend.  The automatic smoothing choice uses the psfs
        of all stamps; with use_noise_image=True every stamp must
        carry its noise realization.  The fixed model centers v, u
        are in the same common frame as the object centers.

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
    fwhm_smooth = choose_fwhm_smooth(
        union, fwhm_smooth=fwhm_smooth, smooth_fac=smooth_fac, rng=rng,
    )
    Tsmooth = fwhm_to_T(fwhm_smooth) if fwhm_smooth > 0 else 0.0

    epochs_per_obj = []
    for m, o in zip(mbobs_list, objects):
        eps = []
        for band, obslist in enumerate(m):
            for tobs in obslist:
                ep = prep_epoch(
                    tobs, band=band, fwhm_smooth=fwhm_smooth,
                    ap_rad=ap_rad, use_noise_image=use_noise_image,
                )
                ep['vcen'] = o['v']
                ep['ucen'] = o['u']
                eps.append(ep)
        epochs_per_obj.append(eps)

    return _deblend_core(
        epochs_per_obj, nband, objects, fwhm_smooth, Tsmooth,
        maxiter, tol, fixed_models=fixed_models,
    )


def _pack_state(models, Sw, scales):
    """
    the global deblend state as a normalized vector, for the sweep
    map extrapolation.  Star weights and covariances are frozen and
    only their fluxes enter.  The scales are fixed on the first call
    """
    x = []
    for m, sw in zip(models, Sw):
        x.extend(m['F'])
        if m['type'] == 'gauss':
            x.extend([
                m['cov_sm'][0, 0], m['cov_sm'][0, 1],
                m['cov_sm'][1, 1],
            ])
        elif m['type'] in ('exp', 'dev'):
            x.extend([
                m['cov'][0, 0], m['cov'][0, 1], m['cov'][1, 1],
            ])
        if m['type'] != 'star':
            x.extend([sw[0, 0], sw[0, 1], sw[1, 1]])
    x = np.array(x)
    if scales is None:
        scales = np.maximum(np.abs(x), 1.0e-10)
    return x / scales, scales


def _unpack_state(x, scales, models, Sw):
    """write a packed state vector back into the models and weights"""
    x = x * scales
    k = 0
    for i, m in enumerate(models):
        nband = m['F'].size
        m['F'] = x[k:k + nband].copy()
        k += nband
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
        if m['type'] != 'star':
            Sw[i] = np.array([
                [x[k], x[k + 1]], [x[k + 1], x[k + 2]],
            ])
            k += 3


def _state_valid(models, Sw, Tsmooth):
    """every weight and model in the state gives well defined sums"""
    for m, sw in zip(models, Sw):
        if sw[0, 0] <= 0 or sw[1, 1] <= 0 or det2(sw) <= 0:
            return False
        if m['type'] in ('exp', 'dev'):
            if not mixture_model_valid(
                    m['type'], m['cov'], ZERO_WEIGHT, Tsmooth):
                return False
        elif m['type'] == 'gauss':
            if det2(m['cov_sm']) <= 0:
                return False
    return True


def _deblend_core(
    epochs_per_obj, nband, objects, fwhm_smooth, Tsmooth, maxiter, tol,
    fixed_models=None,
):
    """
    the Gauss-Seidel iteration over objects, with a per-object list
    of prepared epochs; in shared-image mode all objects have the
    same list

    The slow tail of the sweep iteration is a collective mode of the
    most blended objects, with their fluxes and structures locked in
    a single slowly decaying direction, so it is accelerated with a
    guarded Steffensen boost on the packed global state of all
    objects: three consecutive plain sweeps give the contraction
    ratio of the dominant mode and the remaining geometric series is
    applied in one step, rolled back if it leaves the valid region.
    Convergence is always decided by a subsequent plain sweep.  See
    ngmix.prepsfadmom PAdmomFitter._run_admom_mixture for the
    Aitken/Steffensen/Sidi references
    """
    nobj = len(objects)
    if nobj == 0:
        raise ValueError('no objects sent')

    positions = []
    models = []
    Sw = []
    for o in objects:
        positions.append((o['v'], o['u']))
        otype = o.get('type', 'gauss')
        Tguess = o.get('Tguess', DEFAULT_TGUESS)
        m = {'type': otype, 'F': np.zeros(nband)}
        if otype == 'star':
            # pre-psf delta function: in the smoothed plane the model
            # and the matched weight are both the smoothing gaussian
            Sw.append(np.diag([Tsmooth / 2, Tsmooth / 2]))
            m['cov_sm'] = np.diag([Tsmooth / 2, Tsmooth / 2])
        elif otype == 'gauss':
            Sw.append(np.diag([(Tguess + Tsmooth) / 2] * 2))
            m['cov_sm'] = Sw[-1].copy()
        elif otype in ('exp', 'dev'):
            Sw.append(np.diag([(Tguess + Tsmooth) / 2] * 2))
            m['cov'] = cov_from_e(0.0, 0.0, Tguess)
        else:
            raise ValueError(f"bad object type: '{otype}'")
        models.append(m)

    fpositions, fmodels = _convert_fixed_models(
        fixed_models, nband, Tsmooth,
    )

    _init_fluxes(
        epochs_per_obj, nband, positions, models, Sw, Tsmooth,
        fpositions=fpositions, fmodels=fmodels,
    )

    nskip = 0
    cen_pull = [np.zeros(2) for _ in range(nobj)]
    esums = np.zeros(6)

    # sweep-map extrapolation history of normalized global states
    scales = None
    hist = []

    # per-object failure containment state
    nfail = np.zeros(nobj, dtype='i4')
    nrestart = np.zeros(nobj, dtype='i4')
    dbflags = np.zeros(nobj, dtype='i4')

    def contain_failure(i):
        """
        count a consecutive failed structure update for object i.
        At NFAIL_LIMIT failures, restart the object from the compact
        delta state, where the neighbor contamination that drives
        weight runaways is minimized, so a transient runaway can
        recover and re-grow.  If a restarted object fails again,
        demote it permanently to a fixed point source, whose linear
        flux update is always well defined and which errs by
        under-subtracting wings rather than mis-subtracting a
        nonsense extended model.  Returns True when it intervened
        """
        nonlocal hist, scales
        nfail[i] += 1
        if nfail[i] < NFAIL_LIMIT:
            return False
        nfail[i] = 0
        m = models[i]
        smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])
        Sw[i] = smooth_cov.copy()
        if nrestart[i] == 0:
            nrestart[i] = 1
            dbflags[i] |= RESTARTED
            if m['type'] in ('exp', 'dev'):
                m['cov'] = np.zeros((2, 2))
            else:
                m['cov_sm'] = smooth_cov.copy()
            # the restart is a discontinuity in the sweep map
            hist = []
        else:
            dbflags[i] |= DEBLENDED_AS_PSF
            m['type'] = 'star'
            m.pop('cov', None)
            m['cov_sm'] = smooth_cov.copy()
            # the packed state layout changed
            hist = []
            scales = None
        return True

    for it in range(maxiter):
        maxchange = 0.0
        for i in range(nobj):
            sums, fs, ws, pred, fs_pred = _object_sums(
                epochs_per_obj[i], nband, positions, models, Sw,
                Tsmooth, i, esums,
                fpositions=fpositions, fmodels=fmodels,
            )
            m = models[i]

            if sums[5] > 0:
                cen_pull[i] = sums[0:2] / sums[5]

            if m['type'] == 'star':
                # structure frozen at the delta-function model; only
                # the linear flux is updated
                newF = fs / ws * 2 * np.pi * np.sqrt(
                    det2(Sw[i] + m['cov_sm'])
                )
                maxchange = max(maxchange, _fchange(newF, m['F']))
                m['F'] = newF
                continue

            # weight update (single gaussian, all object types); on
            # failure keep the previous structure but still update the
            # flux, which is linear and always well defined, so a bad
            # early structure state cannot deadlock the blend
            newSw = None
            if sums[5] > 0 and sums[4] > 0:
                finv = 1.0 / sums[5]
                M1 = sums[2] * finv
                M2 = sums[3] * finv
                T = sums[4] * finv
                M = np.array([
                    [0.5 * (T - M1), 0.5 * M2],
                    [0.5 * M2, 0.5 * (T + M1)],
                ])
                newSw, flags = deweight(M, Sw[i])
                if flags != 0:
                    newSw = None

            if newSw is None:
                nskip += 1
                if nskip > 100 * nobj:
                    raise RuntimeError(
                        f'too many failed structure updates, object {i}'
                    )
                maxchange = max(maxchange, 1.0)
                if m['type'] == 'gauss':
                    m['F'] = fs / ws * 2 * np.pi * np.sqrt(
                        det2(Sw[i] + m['cov_sm'])
                    )
                elif np.all(fs_pred != 0):
                    m['F'] = m['F'] * fs / fs_pred
                contain_failure(i)
                continue

            if m['type'] == 'gauss':
                newF = fs / ws * 2 * np.pi * np.sqrt(det2(Sw[i] + newSw))
                Twt = Sw[i][0, 0] + Sw[i][1, 1]
                maxchange = max(
                    maxchange,
                    np.abs(newSw - Sw[i]).max() / Twt,
                    _fchange(newF, m['F']),
                )
                m['cov_sm'] = newSw
                m['F'] = newF
                nfail[i] = 0
            else:
                # exp/dev: deweight-style update on the family
                # covariance matrix, as in ngmix
                # PAdmomFitter._run_admom_mixture.  Map both the measured
                # and the model-predicted moments through the deweight
                # transform and shift the family covariance by the
                # difference.  For a single-gaussian family this is
                # exactly the standard deweight update; for the
                # mixture it has near-unit gain, unlike a plain Picard
                # update on the weighted moments which converges at
                # rate ~1/2.  The smoothing covariance cancels in the
                # difference.  newSw is the deweight of the measured
                # moments, computed above.
                pinv = 1.0 / pred[5]
                Mp1 = pred[2] * pinv
                Mp2 = pred[3] * pinv
                Tp = pred[4] * pinv
                Mpred = np.array([
                    [0.5 * (Tp - Mp1), 0.5 * Mp2],
                    [0.5 * Mp2, 0.5 * (Tp + Mp1)],
                ])
                Sp, pflags = deweight(Mpred, Sw[i])

                Sfam = m['cov']
                if pflags == 0:
                    shift = newSw - Sp
                else:
                    # gain-1 fallback on the weighted moment ratios,
                    # composed in matrix form: scale by the T ratio
                    # and shift the anisotropy by the ratio
                    # differences
                    Tf = Sfam[0, 0] + Sfam[1, 1]
                    fac = sums[4] / sums[5] / Tp
                    de1 = sums[2] / sums[4] - pred[2] / pred[4]
                    de2 = sums[3] / sums[4] - pred[3] / pred[4]
                    shift = (fac - 1) * Sfam \
                        + 0.5 * fac * Tf * np.array([
                            [-de1, de2],
                            [de2, de1],
                        ])

                # accept the largest step, damping if needed, for
                # which the smoothed components stay valid under the
                # zero weight
                accepted = False
                for idamp in range(10):
                    prop = Sfam + shift
                    valid = mixture_model_valid(
                        m['type'], prop, ZERO_WEIGHT, Tsmooth,
                    )
                    if valid:
                        accepted = True
                        break
                    shift = 0.5 * shift

                newF = m['F'] * fs / fs_pred
                maxchange = max(maxchange, _fchange(newF, m['F']))
                m['F'] = newF

                if not accepted:
                    # no valid step: keep the previous structure and
                    # count a failed update; the flux update above
                    # keeps the blend from deadlocking
                    nskip += 1
                    if nskip > 100 * nobj:
                        raise RuntimeError(
                            'too many failed structure updates, '
                            f'object {i}'
                        )
                    maxchange = max(maxchange, 1.0)
                    if contain_failure(i):
                        # the weight was reset by the intervention
                        continue
                elif idamp > 0:
                    # a damped step can be small only because it was
                    # shortened at the validity boundary, not because
                    # the fit has settled
                    maxchange = max(maxchange, 1.0)
                    m['cov'] = prop
                    nfail[i] = 0
                else:
                    Twt = Sw[i][0, 0] + Sw[i][1, 1]
                    maxchange = max(
                        maxchange, np.abs(shift).max() / Twt,
                    )
                    m['cov'] = prop
                    nfail[i] = 0

            Sw[i] = newSw

        if maxchange < tol:
            break

        x, scales = _pack_state(models, Sw, scales)
        hist.append(x)
        if len(hist) >= 3:
            d1 = hist[-2] - hist[-3]
            d2 = hist[-1] - hist[-2]
            denom = d1 @ d1
            rho = (d2 @ d1) / denom if denom > 0 else 0.0
            if 0.2 < rho < 0.98:
                saved_models = [dict(m) for m in models]
                saved_Sw = [sw.copy() for sw in Sw]
                _unpack_state(
                    hist[-1] + d2 * rho / (1 - rho), scales,
                    models, Sw,
                )
                if _state_valid(models, Sw, Tsmooth):
                    # a fresh trio of plain sweeps is needed for the
                    # next ratio estimate
                    hist = []
                else:
                    for m, sm in zip(models, saved_models):
                        m.update(sm)
                    for k in range(nobj):
                        Sw[k] = saved_Sw[k]
            if len(hist) > 3:
                hist = hist[-3:]

    out_objects = []
    for i in range(nobj):
        m = models[i]

        # noise propagation at the converged weight; the neighbor
        # corrections are deterministic, so the raw kernel cross sums
        # give the covariances of the corrected sums.  For the
        # weight-adaptive types the sandwich over the moment matching
        # conditions (ngmix model_sandwich) gives the flux variances
        # including the weight and family responses, plus the family
        # covariance for the structure errors; for a gauss object it
        # reduces exactly to the analytic delta method.  Star weights
        # are frozen, so the fixed weight flux variance is exact and
        # there are no structure errors
        sums_i, fs, _, _, _ = _object_sums(
            epochs_per_obj[i], nband, positions, models, Sw,
            Tsmooth, i, esums,
            fpositions=fpositions, fmodels=fmodels,
        )
        vi, ui = positions[i]
        fvar = np.zeros(nband)
        fmcov = np.zeros((nband, 3))
        covj = np.zeros((6, 6))
        fcov = np.zeros((6, 6))
        for ep in epochs_per_obj[i]:
            alpha, beta = get_phase_angles(
                ep, vi - ep['vcen'], ui - ep['ucen'],
            )
            admom_finalize(
                ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                alpha, beta, ep['kv'], ep['ku'],
                Sw[i][0, 0], Sw[i][0, 1], Sw[i][1, 1], ep['df2'],
                ep['err_fac2'],
                esums, fcov,
            )
            fac = ep['weight'] * ep['detAtinv']
            nfac = ep['df2'] ** 2
            band = ep['band']
            fvar[band] += fac ** 2 * nfac * fcov[5, 5]
            fmcov[band] += fac ** 2 * nfac * fcov[2:5, 5]
            covj += fac ** 2 * nfac * fcov

        fam_cov = None
        gfam_cov = None
        gfvar = None
        smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])
        Sgal_w = Sw[i] - smooth_cov
        if m['type'] != 'star' and sums_i[5] > 0:
            if m['type'] == 'gauss':
                mtype = 'gauss'
                Sfam = m['cov_sm'] - smooth_cov
            else:
                mtype = m['type']
                Sfam = m['cov']
            fvar_raw = fvar
            fvar, fam_cov = model_sandwich(
                mtype, Sfam, Sw[i], Tsmooth,
                sums_i, covj, fs, fvar_raw, fmcov,
            )
            if mtype == 'gauss':
                # the weight equals the gauss family covariance, so
                # the sandwiches coincide
                gfvar = fvar
                gfam_cov = fam_cov
            else:
                # gauss-estimator errors under the same weight, for
                # the low-noise shape entries below
                gfvar, gfam_cov = model_sandwich(
                    'gauss', Sgal_w, Sw[i], Tsmooth,
                    sums_i, covj, fs, fvar_raw, fmcov,
                )

        flux_err = np.full(nband, np.nan)
        wgood = (fvar > 0) & (fs != 0)
        flux_err[wgood] = np.abs(
            m['F'][wgood] / fs[wgood],
        ) * np.sqrt(fvar[wgood])
        if np.any(wgood):
            s2n = np.sqrt(
                np.sum(fs[wgood] ** 2 / fvar[wgood]),
            )
        else:
            s2n = np.nan

        res = {
            'type': m['type'],
            'deblend_flags': int(dbflags[i]),
            'flux': m['F'].copy(),
            'flux_err': flux_err,
            's2n': s2n,
            'cen': np.array(positions[i]),
            'cen_pull': cen_pull[i],
        }
        # e_flags == 0 iff the ellipticities and their errors are
        # usable, following the ngmix prepsfadmom convention
        res['e_flags'] = 0
        res['e1'] = np.nan
        res['e2'] = np.nan
        if m['type'] == 'gauss':
            Sgal = m['cov_sm'] - np.diag([Tsmooth / 2, Tsmooth / 2])
            Tgal = Sgal[0, 0] + Sgal[1, 1]
            res['T'] = Tgal
            # det > 0 with positive trace is |e| < 1: a positive-size
            # gaussian with a degenerate covariance has no defined
            # shape, same rule as the exp family
            shape_ok = Tgal > 0 and det2(Sgal) > 0
            if shape_ok:
                res['e1'] = (Sgal[1, 1] - Sgal[0, 0]) / Tgal
                res['e2'] = 2 * Sgal[0, 1] / Tgal
        elif m['type'] == 'star':
            # a delta function has no shape by construction
            res['T'] = 0.0
            shape_ok = False
        else:
            Sfam = m['cov']
            Tgal = Sfam[0, 0] + Sfam[1, 1]
            res['T'] = Tgal
            # the family covariance can scatter out of positive
            # definite, where the shape is undefined
            shape_ok = Tgal > 0 and det2(Sfam) > 0
            if shape_ok:
                res['e1'] = (Sfam[1, 1] - Sfam[0, 0]) / Tgal
                res['e2'] = 2 * Sfam[0, 1] / Tgal
        if not shape_ok:
            res['e_flags'] |= ngmix.flags.NONPOS_SIZE

        # structure errors from the family covariance sandwich
        res['T_err'] = np.nan
        res['e1_err'] = np.nan
        res['e2_err'] = np.nan
        if fam_cov is not None:
            if fam_cov[2, 2] > 0:
                res['T_err'] = np.sqrt(fam_cov[2, 2])
            if shape_ok:
                e1 = res['e1']
                e2 = res['e2']
                ev1 = (
                    fam_cov[0, 0]
                    - 2 * e1 * fam_cov[0, 2]
                    + e1 ** 2 * fam_cov[2, 2]
                ) / Tgal ** 2
                ev2 = (
                    fam_cov[1, 1]
                    - 2 * e2 * fam_cov[1, 2]
                    + e2 ** 2 * fam_cov[2, 2]
                ) / Tgal ** 2
                if ev1 > 0 and ev2 > 0:
                    res['e1_err'] = np.sqrt(ev1)
                    res['e2_err'] = np.sqrt(ev2)
                else:
                    res['e_flags'] |= ngmix.flags.NONPOS_SHAPE_VAR
        elif shape_ok:
            # no error propagation was possible for a shape that is
            # otherwise defined
            res['e_flags'] |= ngmix.flags.NONPOS_SHAPE_VAR

        # gauss-estimator shapes from the converged weight.  The
        # weight iteration is exactly the adaptive-moments gauss
        # fixed point on the neighbor-corrected data (the family
        # state only enters through the matching conditions), so the
        # lowest-noise gauss shape estimator is available for every
        # model type at no extra fitting cost: the family models do
        # the subtraction, the gauss weight does the measurement,
        # and the metacal response calibrates the estimator.  Fluxes
        # should still come from the family models
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
            Tgw = Sgal_w[0, 0] + Sgal_w[1, 1]
            res['gauss_T'] = Tgw
            gok = Tgw > 0 and det2(Sgal_w) > 0
            if gok:
                res['gauss_e1'] = (Sgal_w[1, 1] - Sgal_w[0, 0]) / Tgw
                res['gauss_e2'] = 2 * Sgal_w[0, 1] / Tgw
            else:
                res['gauss_e_flags'] |= ngmix.flags.NONPOS_SIZE
            if gfam_cov is not None:
                if gfam_cov[2, 2] > 0:
                    res['gauss_T_err'] = np.sqrt(gfam_cov[2, 2])
                if gok:
                    ge1 = res['gauss_e1']
                    ge2 = res['gauss_e2']
                    gv1 = (
                        gfam_cov[0, 0]
                        - 2 * ge1 * gfam_cov[0, 2]
                        + ge1 ** 2 * gfam_cov[2, 2]
                    ) / Tgw ** 2
                    gv2 = (
                        gfam_cov[1, 1]
                        - 2 * ge2 * gfam_cov[1, 2]
                        + ge2 ** 2 * gfam_cov[2, 2]
                    ) / Tgw ** 2
                    if gv1 > 0 and gv2 > 0:
                        res['gauss_e1_err'] = np.sqrt(gv1)
                        res['gauss_e2_err'] = np.sqrt(gv2)
                    else:
                        res['gauss_e_flags'] |= (
                            ngmix.flags.NONPOS_SHAPE_VAR
                        )
            elif gok:
                res['gauss_e_flags'] |= ngmix.flags.NONPOS_SHAPE_VAR

        # gauss-aperture fluxes and flux s/n, the analogs of the
        # gauss-model deblender's outputs, for selection studies
        # against the family quantities.  For a gauss object these
        # equal the primary entries
        res['gauss_flux'] = np.full(nband, np.nan)
        res['gauss_flux_err'] = np.full(nband, np.nan)
        res['gauss_s2n'] = np.nan
        if m['type'] != 'star' and gfvar is not None:
            Fg = fs / ws * 4 * np.pi * np.sqrt(det2(Sw[i]))
            res['gauss_flux'] = Fg
            wg = (gfvar > 0) & (fs != 0)
            res['gauss_flux_err'][wg] = np.abs(
                Fg[wg] / fs[wg],
            ) * np.sqrt(gfvar[wg])
            if np.any(wg):
                res['gauss_s2n'] = np.sqrt(
                    np.sum(fs[wg] ** 2 / gfvar[wg]),
                )
        out_objects.append(res)

    return {
        'objects': out_objects,
        'fwhm_smooth': fwhm_smooth,
        'Tsmooth': Tsmooth,
        'numiter': it + 1,
        'nskip': nskip,
    }


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
        elif ftype in ('gauss', 'exp', 'dev'):
            pars = [f['e1'], f['e2'], f['T']]
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
            else:
                m = {'type': ftype, 'cov': cov, 'F': F}
        else:
            raise ValueError(f"bad fixed model type: '{ftype}'")
        positions.append((f['v'], f['u']))
        models.append(m)
    return positions, models


def _object_sums(epochs, nband, positions, models, Sw, Tsmooth, i,
                 esums, fpositions=(), fmodels=()):
    """
    neighbor-corrected weighted moment sums for object i, accumulated
    over the object's epochs, plus the model's own predicted sums for
    'exp' objects.  The fixed external models in fpositions/fmodels
    are subtracted exactly like in-group neighbors
    """
    vi, ui = positions[i]
    is_mix = models[i]['type'] in ('exp', 'dev')

    # the model sums scale exactly as 1/detAtinv, so expand the
    # components once, run the kernel once per band at detAtinv=1,
    # and rescale per epoch; the fixed externals join the same
    # kernel call
    ncomps = []
    for j in range(len(positions)):
        if j == i:
            continue
        fracs, So00, So01, So11 = model_comps(models[j], Tsmooth)
        ncomps.append(
            (positions[j], models[j]['F'], fracs, So00, So01, So11)
        )
    for p, fm in zip(fpositions, fmodels):
        fracs, So00, So01, So11 = model_comps(fm, Tsmooth)
        ncomps.append((p, fm['F'], fracs, So00, So01, So11))

    base_nsums = np.zeros((nband, 6))
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
        for band in range(nband):
            nF = np.concatenate([
                c[1][band] * c[2] for c in ncomps
            ])
            gauss_comps_ksums(
                nF, nSo00, nSo01, nSo11, ndv, ndu,
                Sw[i][0, 0], Sw[i][0, 1], Sw[i][1, 1], 1.0,
                base_nsums[band],
            )

    if is_mix:
        base_psums = np.zeros((nband, 6))
        fracs, So00, So01, So11 = model_comps(models[i], Tsmooth)
        zeros = np.zeros(fracs.size)
        for band in range(nband):
            gauss_comps_ksums(
                models[i]['F'][band] * fracs, So00, So01, So11,
                zeros, zeros,
                Sw[i][0, 0], Sw[i][0, 1], Sw[i][1, 1], 1.0,
                base_psums[band],
            )

    sums = np.zeros(6)
    fs = np.zeros(nband)
    ws = np.zeros(nband)
    pred = np.zeros(6)
    fs_pred = np.zeros(nband)

    for ep in epochs:
        alpha, beta = get_phase_angles(
            ep, vi - ep['vcen'], ui - ep['ucen'],
        )
        admom_ksums(
            ep['kim'], ep['iy'], ep['ix'], ep['dim'],
            alpha, beta, ep['kv'], ep['ku'],
            Sw[i][0, 0], Sw[i][0, 1], Sw[i][1, 1], ep['df2'],
            esums,
        )
        csums = esums - base_nsums[ep['band']] / ep['detAtinv']
        fac = ep['weight'] * ep['detAtinv']
        sums += fac * csums
        fs[ep['band']] += fac * csums[5]
        ws[ep['band']] += ep['weight']

        if is_mix:
            psums = base_psums[ep['band']] / ep['detAtinv']
            pred += fac * psums
            fs_pred[ep['band']] += fac * psums[5]

    return sums, fs, ws, pred, fs_pred


def _init_fluxes(epochs_per_obj, nband, positions, models, Sw, Tsmooth,
                 fpositions=(), fmodels=()):
    """
    initialize the fluxes by solving the per-band linear system at the
    guess structures: the measured flux sums for each object are linear
    in all object fluxes with closed-form overlap coefficients.  The
    fixed external models are subtracted from the measured side
    """
    nobj = len(positions)
    esums = np.zeros(6)
    for band in range(nband):
        A = np.zeros((nobj, nobj))
        bvec = np.zeros(nobj)
        for i in range(nobj):
            vi, ui = positions[i]
            for ep in epochs_per_obj[i]:
                if ep['band'] != band:
                    continue
                fac = ep['weight'] * ep['detAtinv']
                alpha, beta = get_phase_angles(
                    ep, vi - ep['vcen'], ui - ep['ucen'],
                )
                admom_ksums(
                    ep['kim'], ep['iy'], ep['ix'], ep['dim'],
                    alpha, beta, ep['kv'], ep['ku'],
                    Sw[i][0, 0], Sw[i][0, 1], Sw[i][1, 1], ep['df2'],
                    esums,
                )
                bvec[i] += fac * esums[5]
                for p, fm in zip(fpositions, fmodels):
                    bvec[i] -= fac * model_ksums(
                        fm, band, p[0] - vi, p[1] - ui,
                        Sw[i], ep['detAtinv'], Tsmooth,
                    )[5]
                for j in range(nobj):
                    munit = dict(models[j])
                    munit['F'] = np.ones(nband)
                    A[i, j] += fac * model_ksums(
                        munit, band,
                        positions[j][0] - vi, positions[j][1] - ui,
                        Sw[i], ep['detAtinv'], Tsmooth,
                    )[5]
        fsol = np.linalg.solve(A, bvec)
        for i in range(nobj):
            models[i]['F'][band] = fsol[i]


def _fchange(newF, oldF):
    """maximum relative flux change"""
    return (np.abs(newF - oldF) / (np.abs(oldF) + 1.0e-30)).max()
