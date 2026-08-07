"""
The kdeblend group-replace fitter as one CUDA kernel: whole-fit-per-block.

Boundary: construction on CPU (kd.build_deblender); the kernel runs
the complete sweep loop (member steps, gauss/star/exp updates,
damped mixture steps, skip/containment/restart/demote, matched and
predicted-ratio fluxes, ratcheted change classes, projected-residual
convergence, noncontraction window, Steffensen extrapolation with
validity rollback, regularized recentering with lazy pull noise) and
returns the final state for state-level comparison with the class.

Scope v1: gauss + star + exp, single band/epoch, e_sigma0=0, no
bdf/dev.  fp64 throughout; the kernel mirrors diffrig.py's my_*
functions (the bitwise-validated executable spec) line by line.
Sum reduction order differs from the serial CPU kernels, so parity
is expected at ~1e-13 state level with occasional threshold flips,
not bitwise.

fp32-mixed variant (get_kernel(fp32=True) + pack_groups(fp32=True)):
mode data stored fp32 (complex64 kim, float32 kv/ku/ef2), per-mode
math in fp32 with Kahan-compensated per-thread bins, tree reduction
and ALL state/decision logic unchanged in fp64.  Same source,
compiled with -DMODE_FP32; the fp64 path is byte-identical to the
validated kernel.

The CUDA sources live in cuda/ as real .cu/.cuh files (one
__global__ driver plus __device__ phase functions, all inlined by
nvrtc).  get_kernel concatenates them in fixed order behind a
generated #define prologue carrying the compile-time constants and
the python-derived exp tables; see _load_cuda_source for why the
#include lines are stripped rather than resolved by nvrtc.
"""
import os
import numpy as np
import cupy as cp

from ngmix.prepsfadmom.models import get_profile_comps

TGAUSS = 0
TSTAR = 1
TEXP = 2
TYPECODE = {'gauss': TGAUSS, 'star': TSTAR, 'exp': TEXP}
TYPENAME = {v: k for k, v in TYPECODE.items()}

_EXP_COMPS = get_profile_comps('exp')
NEXPC = len(_EXP_COMPS)

_EXPVALS = ', '.join(f'{v:.17e}' for v in np.exp(np.arange(-15, 1)))
_EXPVALSF = ', '.join(
    f'{v:.9e}f' for v in np.exp(np.arange(-15, 1)).astype(np.float32)
)
_EXPFRAC = ', '.join(f'{f:.17e}' for f, cT in _EXP_COMPS)
_EXPCT = ', '.join(f'{cT:.17e}' for f, cT in _EXP_COMPS)

GMAX = 64
SPO = 9         # packed slots per object (F,1 + cen,2 + cov,3 + Sw,3)
NT = 256
DIM_MAX = 512

_CUDA_DIR = os.path.join(os.path.dirname(__file__), 'cuda')

# concatenation order: this IS the include graph (the #include
# lines inside the files exist for editor tooling only)
_CUDA_FILES = (
    'common.cuh',
    'model.cuh',
    'state.cuh',
    'reduce.cuh',
    'update.cuh',
    'sweep.cuh',
    'measure.cuh',
    'deblend_groups.cu',
)


def _load_cuda_source():
    """read and concatenate the kernel sources, stripping the
    tooling-only #include/#pragma lines.  Everything lands in the
    one source string handed to cupy so its on-disk compile cache
    stays keyed on actual content; an #include resolved by nvrtc
    at compile time would not be hashed, and editing a header
    would silently reuse the stale cubin"""
    parts = []
    for fname in _CUDA_FILES:
        with open(os.path.join(_CUDA_DIR, fname)) as fobj:
            text = fobj.read()
        keep = [
            ln for ln in text.split('\n')
            if not ln.startswith('#include "')
            and not ln.startswith('#pragma once')
        ]
        parts.append('\n'.join(keep))
    return '\n'.join(parts)


def _prologue(nt):
    """the compile-time configuration and python-derived constant
    tables, as preprocessor defines prepended to the source"""
    return '\n'.join([
        f'#define NTHREADS_H {nt}',
        f'#define NEXPC_H {NEXPC}',
        f'#define DIM_MAX_H {DIM_MAX}',
        f'#define GMAX_H {GMAX}',
        f'#define EXPVALS_H {_EXPVALS}',
        f'#define EXPVALSF_H {_EXPVALSF}',
        f'#define EXPFRAC_H {_EXPFRAC}',
        f'#define EXPCT_H {_EXPCT}',
        '',
    ])


_KERNELS = {}


def get_kernel(fp32=False, nt=NT):
    nt = int(nt)
    assert nt >= 2 and (nt & (nt - 1)) == 0, 'nt must be power of 2'
    key = (bool(fp32), nt)
    if key not in _KERNELS:
        src = _prologue(nt) + _load_cuda_source()
        opts = ('-DMODE_FP32',) if key[0] else ()
        _KERNELS[key] = cp.RawKernel(
            src, 'deblend_groups', options=opts,
        )
    return _KERNELS[key]


def pack_host(debs):
    """pack constructed _Deblender instances into host numpy arrays
    (npz-serializable; convert with to_gpu, or use pack_groups)"""
    ng = len(debs)
    per = {k: [] for k in
           ['nobj', 'dim', 'df2', 'drow', 'dcol', 'jac', 'wt',
            'deti', 'tsm', 'tol', 'ftol', 'ctol', 'maxiter',
            'recenter', 'censig0']}
    modes = {k: [] for k in ['kim', 'iy', 'ix', 'kv', 'ku', 'ef2']}
    fixed = []
    state = {k: [] for k in
             ['otype', 'F', 'fscale', 'cov', 'sw', 'pos', 'dpos',
              'cpull', 'censig', 'nfail', 'nrestart', 'dbflags',
              'fixcen']}
    moff = [0]
    foff = [0]
    ooff = [0]
    soff = [0]

    for deb in debs:
        assert deb.nband == 1
        assert deb.e_sigma0 == 0.0
        eps = deb.epochs_per_obj[0]
        assert len(eps) == 1
        ep = eps[0]
        assert ep['dim'] <= DIM_MAX
        n = deb.nobj
        assert n <= GMAX

        per['nobj'].append(n)
        per['dim'].append(ep['dim'])
        per['df2'].append(ep['df2'])
        per['drow'].append(ep['drow'])
        per['dcol'].append(ep['dcol'])
        A = ep['Atinv']
        per['jac'].append([A[0, 0], A[0, 1], A[1, 0], A[1, 1]])
        per['wt'].append(ep['weight'])
        per['deti'].append(ep['detAtinv'])
        per['tsm'].append(deb.Tsmooth)
        per['tol'].append(deb.tol)
        per['ftol'].append(deb.flux_tol)
        per['ctol'].append(deb.cen_tol)
        per['maxiter'].append(deb.maxiter)
        per['recenter'].append(1 if deb.recenter else 0)
        per['censig0'].append(deb.cen_sigma0)

        modes['kim'].append(ep['kim'])
        modes['iy'].append(ep['iy'].astype(np.int32))
        modes['ix'].append(ep['ix'].astype(np.int32))
        modes['kv'].append(ep['kv'])
        modes['ku'].append(ep['ku'])
        modes['ef2'].append(ep['err_fac2'])
        moff.append(moff[-1] + ep['kim'].size)

        # fixed models: pre-expanded comps under any weight
        from ngmix.prepsfadmom.models import model_comps
        nf = 0
        for p, fm in zip(deb.fpositions, deb.fmodels):
            fracs, So00, So01, So11 = model_comps(fm, deb.Tsmooth)
            for c in range(fracs.size):
                fixed.append([
                    fm['F'][0] * fracs[c], So00[c], So01[c],
                    So11[c], p[0], p[1],
                ])
                nf += 1
        foff.append(foff[-1] + nf)

        for i in range(n):
            m = deb.models[i]
            state['otype'].append(TYPECODE[m['type']])
            state['F'].append(m['F'][0])
            state['fscale'].append(deb._fscales[i][0])
            c = m['cov_sm'] if m['type'] in ('gauss', 'star') \
                else m['cov']
            state['cov'].append([c[0, 0], c[0, 1], c[1, 1]])
            s = deb.Sw[i]
            state['sw'].append([s[0, 0], s[0, 1], s[1, 1]])
            state['pos'].append(list(deb.positions[i]))
            state['dpos'].append(list(deb.det_positions[i]))
            state['cpull'].append(list(deb.cen_pull[i]))
            state['censig'].append(deb._cen_sigma_sweep[i])
            state['nfail'].append(deb.nfail[i])
            state['nrestart'].append(deb.nrestart[i])
            state['dbflags'].append(deb.dbflags[i])
            state['fixcen'].append(1 if deb.fixcen[i] else 0)
        ooff.append(ooff[-1] + n)
        soff.append(soff[-1] + SPO * n)

    def hn(x, dt):
        return np.asarray(x, dtype=dt)

    return dict(
        ng=np.int64(ng),
        kim=hn(np.concatenate(modes['kim']), np.complex128),
        iy=hn(np.concatenate(modes['iy']), np.int32),
        ix=hn(np.concatenate(modes['ix']), np.int32),
        kv=hn(np.concatenate(modes['kv']), np.float64),
        ku=hn(np.concatenate(modes['ku']), np.float64),
        ef2=hn(np.concatenate(modes['ef2']), np.float64),
        moff=hn(moff, np.int64),
        fcomp=hn(np.array(fixed, dtype=np.float64).reshape(-1, 6)
                 if fixed else np.zeros((0, 6)), np.float64),
        foff=hn(foff, np.int64),
        nobj=hn(per['nobj'], np.int32),
        dim=hn(per['dim'], np.int32),
        df2=hn(per['df2'], np.float64),
        drow=hn(per['drow'], np.float64),
        dcol=hn(per['dcol'], np.float64),
        jac=hn(np.array(per['jac']).ravel(), np.float64),
        wt=hn(per['wt'], np.float64),
        deti=hn(per['deti'], np.float64),
        tsm=hn(per['tsm'], np.float64),
        tol=hn(per['tol'], np.float64),
        ftol=hn(per['ftol'], np.float64),
        ctol=hn(per['ctol'], np.float64),
        maxiter=hn(per['maxiter'], np.int32),
        recenter=hn(per['recenter'], np.int32),
        censig0=hn(per['censig0'], np.float64),
        otype=hn(state['otype'], np.int32),
        F=hn(state['F'], np.float64),
        fscale=hn(state['fscale'], np.float64),
        cov=hn(np.array(state['cov']).ravel(), np.float64),
        sw=hn(np.array(state['sw']).ravel(), np.float64),
        pos=hn(np.array(state['pos']).ravel(), np.float64),
        dpos=hn(np.array(state['dpos']).ravel(), np.float64),
        cpull=hn(np.array(state['cpull']).ravel(), np.float64),
        censig=hn(state['censig'], np.float64),
        nfail=hn(state['nfail'], np.int32),
        nrestart=hn(state['nrestart'], np.int32),
        dbflags=hn(state['dbflags'], np.int32),
        fixcen=hn(state['fixcen'], np.int32),
        ooff=hn(ooff, np.int64),
        soff=hn(soff, np.int64),
    )


MODE_FIELDS = ('kim', 'kv', 'ku', 'ef2', 'iy', 'ix')


def _mode_dtypes(fp32):
    kdt = np.complex64 if fp32 else np.complex128
    mdt = np.float32 if fp32 else np.float64
    return {'kim': kdt, 'kv': mdt, 'ku': mdt, 'ef2': mdt,
            'iy': np.int32, 'ix': np.int32}


def _alloc_scratch(packed, ns, ng, nobj_tot):
    packed['hist'] = cp.zeros(3 * ns, dtype=cp.float64)
    packed['scales'] = cp.zeros(ns, dtype=cp.float64)
    packed['saved'] = cp.zeros(ns, dtype=cp.float64)
    packed['xtmp'] = cp.zeros(ns, dtype=cp.float64)
    packed['objsums'] = cp.zeros(nobj_tot * 6, dtype=cp.float64)
    packed['objcov'] = cp.zeros(nobj_tot * 13, dtype=cp.float64)
    packed['numiter'] = cp.zeros(ng, dtype=cp.int32)
    packed['converged'] = cp.zeros(ng, dtype=cp.int32)
    packed['nskip'] = cp.zeros(ng, dtype=cp.int32)
    packed['err'] = cp.zeros(ng, dtype=cp.int32)


def to_gpu(h, fp32=False):
    """host pack -> device arrays + scratch/output allocations"""
    dts = _mode_dtypes(fp32)
    ng = int(h['ng'])
    packed = {'ng': ng, 'fp32': fp32}
    for k, v in h.items():
        if k == 'ng':
            continue
        if k in dts:
            packed[k] = cp.asarray(np.asarray(v, dtype=dts[k]))
        else:
            packed[k] = cp.asarray(v)
    _alloc_scratch(packed, int(h['soff'][-1]), ng,
                   int(h['ooff'][-1]))
    return packed


def to_gpu_multi(small_h, mode_list, fp32=False):
    """assemble a batch from a host-merged SMALL-array pack plus a
    list of per-submission mode-array dicts: the mode fields (the
    bulk of the bytes) are copied per submission straight into
    device slabs, no host merge"""
    dts = _mode_dtypes(fp32)
    ng = int(small_h['ng'])
    packed = {'ng': ng, 'fp32': fp32}
    nms = [int(m['kim'].size) for m in mode_list]
    nm_tot = sum(nms)
    for k, dt in dts.items():
        packed[k] = cp.empty(nm_tot, dtype=dt)
    off = 0
    for m, n in zip(mode_list, nms):
        for k, dt in dts.items():
            src = np.ascontiguousarray(np.asarray(m[k], dtype=dt))
            packed[k][off:off + n].set(src)
        off += n
    for k, v in small_h.items():
        if k == 'ng':
            continue
        packed[k] = cp.asarray(v)
    _alloc_scratch(packed, int(small_h['soff'][-1]), ng,
                   int(small_h['ooff'][-1]))
    return packed


def pack_groups(debs, fp32=False):
    """pack constructed _Deblender instances for the kernel"""
    return to_gpu(pack_host(debs), fp32=fp32)


def launch_gpu(packed, nt=NT):
    """launch the kernel and synchronize; no output fetch"""
    kern = get_kernel(fp32=packed.get('fp32', False), nt=nt)
    p = packed
    args = (
        p['kim'], p['iy'], p['ix'], p['kv'], p['ku'], p['ef2'],
        p['moff'], p['fcomp'], p['foff'],
        p['nobj'], p['dim'], p['df2'], p['drow'], p['dcol'],
        p['jac'], p['wt'], p['deti'], p['tsm'],
        p['tol'], p['ftol'], p['ctol'], p['maxiter'],
        p['recenter'], p['censig0'],
        p['otype'], p['F'], p['fscale'], p['cov'], p['sw'],
        p['pos'], p['dpos'], p['cpull'], p['censig'],
        p['nfail'], p['nrestart'], p['dbflags'], p['fixcen'],
        p['ooff'], p['hist'], p['scales'], p['saved'], p['xtmp'],
        p['soff'], p['objsums'], p['objcov'],
        p['numiter'], p['converged'], p['nskip'], p['err'],
    )
    kern((p['ng'],), (int(nt),), args)
    cp.cuda.Device().synchronize()


def fetch_out(packed):
    out = {}
    for k in ['otype', 'F', 'cov', 'sw', 'pos', 'cpull', 'dbflags',
              'nrestart', 'numiter', 'converged', 'nskip', 'err',
              'objsums', 'objcov']:
        out[k] = cp.asnumpy(packed[k])
    out['ooff'] = cp.asnumpy(packed['ooff'])
    return out


def run_gpu(packed, nt=NT):
    launch_gpu(packed, nt=nt)
    return fetch_out(packed)
