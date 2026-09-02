"""
the weight bound: a deweight step whose weight exceeds the stamp
bound is rejected (a runaway), flagged, and counted; a normal step
is accepted.  Without the bound a runaway grows the weight by orders
of magnitude per sweep until it is numerically singular
"""
import numpy as np
import pytest

from kdeblend import build_deblender, WEIGHT_BOUNDED
from kdeblend.deblender import MAX_WEIGHT_SIGMA_FAC

from _sims import make_blend_obs, DIM, SCALE

FWHM_SMOOTH = 1.2


def make_deblender():
    comp = dict(kind='exp', e1=0.0, e2=0.0, hlr=0.4, flux=50.0,
                v=0.0, u=0.0)
    obs = make_blend_obs([comp], 0.9)
    objects = [dict(v=0.0, u=0.0, type='exp', Tguess=0.5)]
    dbl, _ = build_deblender(
        obs, objects, fwhm_smooth=FWHM_SMOOTH, ap_rad=0,
    rng=np.random.RandomState(1),
)
    return dbl


def sums_for_moments(M, flux=1.0):
    """
    weighted moment sums [v, u, M1, M2, T, flux] whose
    _moment_matrix is M
    """
    T = M[0, 0] + M[1, 1]
    M1 = M[1, 1] - M[0, 0]
    M2 = 2 * M[0, 1]
    return np.array([0.0, 0.0, M1 * flux, M2 * flux, T * flux, flux])


def test_bound_from_stamp():
    dbl = make_deblender()
    sigma_max = MAX_WEIGHT_SIGMA_FAC * DIM * SCALE
    assert dbl.Tw_max[0] == pytest.approx(2 * sigma_max ** 2)
    assert dbl.nbound[0] == 0


def test_deweight_rejects_runaway():
    dbl = make_deblender()
    Sw = dbl.Sw[0]

    # measured moments just below the weight: the deweight
    # (M^-1 - Sw^-1)^-1 is Sw (1 - eps) / eps, a thousand times
    # the weight, far beyond the stamp
    eps = 1.0e-3
    sums = sums_for_moments((1 - eps) * Sw)
    assert dbl._deweight_measured(0, sums) is None
    assert dbl.nbound[0] == 1
    assert dbl.dbflags[0] & WEIGHT_BOUNDED
    # a rejection, not an intervention: the failure count is left
    # to the caller's containment
    assert dbl.nfail[0] == 0

    # the weight itself is untouched by the rejection
    assert np.array_equal(dbl.Sw[0], Sw)

    # measured moments at half the weight deweight to the weight
    # itself, well within the bound
    sums = sums_for_moments(0.5 * Sw)
    newSw = dbl._deweight_measured(0, sums)
    assert newSw is not None
    assert np.allclose(newSw, Sw)
    assert dbl.nbound[0] == 1


def test_unbounded_without_entry():
    # epochs prepared without the stamp entry, e.g. by an external
    # device prep, leave the object unbounded
    dbl = make_deblender()
    epochs = dbl.epochs_per_obj[0]
    for ep in epochs:
        del ep['Tw_max']
    dbl2, _ = build_deblender(
        make_blend_obs(
            [dict(kind='exp', e1=0.0, e2=0.0, hlr=0.4, flux=50.0,
                  v=0.0, u=0.0)],
            0.9,
        ),
        [dict(v=0.0, u=0.0, type='exp', Tguess=0.5)],
        fwhm_smooth=FWHM_SMOOTH, ap_rad=0, epochs=epochs,
    rng=np.random.RandomState(1),
)
    assert not np.isfinite(dbl2.Tw_max[0])
    sums = sums_for_moments((1 - 1.0e-3) * dbl2.Sw[0])
    assert dbl2._deweight_measured(0, sums) is not None
    assert dbl2.nbound[0] == 0
