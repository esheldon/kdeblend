"""
fit_groups: run the sweep loop of a batch of constructed
_Deblender instances in one kernel launch and write each back so
it is indistinguishable from having run deb.go() on the CPU —
the untouched class result path then produces the results.

The natural unit of GPU work is the batch: a single group cannot
fill the device (one block), while independent groups batch
freely.  Callers with a streaming workload should batch as widely
as they can (hundreds of groups per launch); small batches are
tail-dominated (the launch wall is the deepest group).

Acceptance doctrine (measured; see the gpu-port document): in
fp64 the kernel is exact-discrete against the CPU fitter on
stable lanes (~95% at production widths); the remainder is
summation-order chaos on marginal boost-heavy blends, validated
at ensemble level (shared-seed A/B on m at the few-1e-4 level).
fp32 is the production configuration and is ensemble-qualified
only; per-lane discrete parity is gone by construction.

fp32 REQUIRES tol >= 1e-6: the fp32 change measures carry a
~1e-7 noise floor and the projected-residual convergence test
multiplies by rho/(1-rho) with rho capped at 0.999, so below
tol ~1e-6 convergence only fires by luck (the production tol is
1e-5).  fit_groups refuses fp32 below the floor.

The two per-object error mode passes of the result path (the
neighbor-corrected moment sums and the admom_finalize cross
sums) are computed by the kernel at the converged state;
fit_groups routes the class result path to them by installing
instance-attribute overrides of _get_object_sums and
_accumulate_error_sums on each written-back deblender.  The
class implementation is untouched; the overrides live only on
instances this module has processed.
"""
import numpy as np

from . import _core
from ..flags import SKIP_LIMIT

FP32_TOL_FLOOR = 1.0e-6

# the 13 unique nonzero entries of the symmetric 6x6 error cross
# matrix (odd x even vanish by parity): (row, col) per packed slot
_CA = (0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5)
_CB = (0, 1, 1, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5)


def writeback(deb, out, gi):
    """write the kernel's final state for group gi into the
    _Deblender exactly as the class updates would have left it"""
    o0 = int(out['ooff'][gi])
    cov3 = out['cov'].reshape(-1, 3)
    sw3 = out['sw'].reshape(-1, 3)
    pos2 = out['pos'].reshape(-1, 2)
    cp2 = out['cpull'].reshape(-1, 2)
    for k in range(deb.nobj):
        j = o0 + k
        m = deb.models[k]
        t = _core.TYPENAME[int(out['otype'][j])]
        c = cov3[j]
        cov = np.array([[c[0], c[1]], [c[1], c[2]]])
        s = sw3[j]
        deb.Sw[k] = np.array([[s[0], s[1]], [s[1], s[2]]])
        if t != m['type']:
            # demoted on device (exp/gauss -> star)
            m['type'] = 'star'
            m.pop('cov', None)
            m['cov_sm'] = cov
        elif t == 'exp':
            m['cov'] = cov
        else:
            # gauss (model is the weight) and star
            m['cov_sm'] = cov
        m['F'] = np.array([out['F'][j]])
        deb.positions[k] = (pos2[j][0], pos2[j][1])
        deb.cen_pull[k] = cp2[j].copy()
        deb.dbflags[k] = int(out['dbflags'][j])
        deb.nrestart[k] = int(out['nrestart'][j])
        deb.nfail[k] = 0
    deb.nskip = int(out['nskip'][gi])
    deb.skip_limit_hit = bool(out['err'][gi])
    if deb.skip_limit_hit:
        # the kernel stopped the group at the backstop limit on
        # skipped structure updates; flag as go() does
        deb.dbflags |= SKIP_LIMIT


def install_sum_overrides(deb, out, gi):
    """route the result path's two per-object mode passes to the
    kernel's final-state measurement outputs"""
    o0 = int(out['ooff'][gi])
    sums6 = out['objsums'].reshape(-1, 6)
    cov13 = out['objcov'].reshape(-1, 13)
    ep = deb.epochs_per_obj[0][0]
    weight = ep['weight']
    fac = ep['weight'] * ep['detAtinv']
    w = fac ** 2 * ep['df2'] ** 2

    def gos(i):
        s = sums6[o0 + i].copy()
        return (s, np.array([s[5]]), np.array([weight]),
                np.zeros(6), np.zeros(1))

    def aes(i):
        cr = np.zeros((6, 6))
        v = cov13[o0 + i]
        for k in range(13):
            cr[_CA[k], _CB[k]] = v[k]
            cr[_CB[k], _CA[k]] = v[k]
        return (np.array([w * cr[5, 5]]),
                (w * cr[2:5, 5]).reshape(1, 3),
                w * cr)

    deb._get_object_sums = gos
    deb._accumulate_error_sums = aes


def result_from_state(deb, out, gi):
    """the deb.go() return dict, from the written-back state"""
    return {
        'converged': bool(out['converged'][gi]),
        'objects': [
            deb._get_object_result(i) for i in range(deb.nobj)
        ],
        'fwhm_smooth': deb.fwhm_smooth,
        'Tsmooth': deb.Tsmooth,
        'numiter': int(out['numiter'][gi]),
        'nskip': int(out['nskip'][gi]),
    }


def fit_groups(debs, fp32=True, nt=_core.NT, results=True):
    """
    Fit a batch of constructed _Deblender instances on the GPU.

    Parameters
    ----------
    debs: sequence of _Deblender
        as returned by kdeblend.build_deblender; scope: single
        band and epoch, gauss/star/exp models, e_sigma0 == 0,
        group size <= 64, cutout dim <= 512
    fp32: bool
        run the per-mode math in single precision (production
        configuration; ensemble-qualified only).  Requires
        tol >= 1e-6 on every deblender.
    nt: int
        threads per block (power of 2)
    results: bool
        when True (default) run the class result path and return
        the list of deb.go()-shaped result dicts; when False only
        write the state back (callers doing their own result
        handling)

    Returns
    -------
    list of dict (results=True) or None; either way every deb has
    the converged state written back and its result-path sum
    overrides installed.
    """
    if len(debs) == 0:
        return [] if results else None
    if fp32:
        bad = [d.tol for d in debs if d.tol < FP32_TOL_FLOOR]
        if bad:
            raise ValueError(
                f'fp32 requires tol >= {FP32_TOL_FLOOR:g}: the '
                f'fp32 change measures have a ~1e-7 noise floor '
                f'and the projected-residual test amplifies it '
                f'by up to 1000x (rho cap); got tol {min(bad):g}'
            )
    packed = _core.pack_groups(debs, fp32=fp32)
    out = _core.run_gpu(packed, nt=nt)

    reslist = [] if results else None
    for gi, deb in enumerate(debs):
        writeback(deb, out, gi)
        install_sum_overrides(deb, out, gi)
        if results:
            reslist.append(result_from_state(deb, out, gi))
    return reslist
