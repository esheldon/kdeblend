"""
Device epoch prep: the array half of ngmix prep_epoch on the gpu,
batched over groups, with the mode arrays staying resident.

The host half (dims, pad centers, weights: prep_epoch_scalars)
runs where the deblender is constructed; this module takes the
raw stamps (image pre-apodized by the caller when ap_rad > 0)
and produces the deconvolved, smoothed, folded kim and the noise
err_fac2 at the retained modes — the mode fields the deblend
kernel consumes — without them ever visiting the host.  The
retained-mode geometry (iy/ix/kv/ku/fold) is data-independent
(ngmix _get_kspace_grids, built on the CPU for bitwise-identical
selection and uploaded once per (dim, jacobian, Tsmooth)).

Per-object measured flux sums for the flux initialization
(_Deblender._init_fluxes) are computed with the init_sums kernel
(the device admom_ksums) and returned to the host: they are the
only construction-time consumer of the mode arrays.

Only the use_noise_image=True path is implemented: the batched
noise rfft is where the per-mode noise power comes from, exactly
as the production deblending path uses it.
"""
import numpy as np
import cupy as cp

from ._core import NT, DIM_MAX_FP64, get_init_sums_kernel

MIN_PSF_FRAC = 1.0e-5

# transient budget per batched rfft2 chunk (the (3B, D, D)
# stack): a byte bound rather than a group count, so dense
# fields and big dims are both bounded (at D=384 this is ~36
# groups per chunk, at D=1024 ~5)
DEFAULT_CHUNK_BYTES = 128 * 1024 ** 2


def _prep_class_chunk(reqs, gis, D, entry, kim_parts, ef2_parts):
    """pad+rfft one chunk of same-(dim, geometry) groups and fill
    their kim/ef2 slots (see prep_groups)"""
    B = len(gis)
    stack = cp.zeros((3 * B, D, D), dtype=cp.float64)
    effs = np.empty(B)
    for j, gi in enumerate(gis):
        r = reqs[gi]
        effs[j] = r['eff_pad_factor']
        for s, im in enumerate(
                (r['image'], r['noise'], r['psf'])):
            ny, nx = im.shape
            p0 = (D - ny) // 2
            p1 = (D - nx) // 2
            stack[3 * j + s, p0:p0 + ny, p1:p1 + nx] = (
                cp.asarray(im)
            )
    kflat = cp.fft.rfft2(stack).reshape(3 * B, -1)

    rows = np.arange(B)
    kim_s = kflat[3 * rows][:, entry.gidx]
    kno_s = kflat[3 * rows + 1][:, entry.gidx]
    kps_s = kflat[3 * rows + 2][:, entry.gidx]

    # deconvolution guard, matching ngmix
    # _deconvolve_im_psf_inplace op for op: modes with
    # 0 < |kpsf| <= min_amp are scaled to min_amp preserving
    # phase; exact zeros become min_amp
    max_amp = cp.abs(kflat[3 * rows + 2, 0])[:, None]
    min_amp = MIN_PSF_FRAC * max_amp
    aps = cp.abs(kps_s)
    low = aps <= min_amp
    kps_c = cp.where(low & (aps != 0),
                     kps_s / aps * min_amp, kps_s)
    kps_c = cp.where(low & (aps == 0),
                     min_amp.astype(kps_s.dtype), kps_c)

    kim_d = (kim_s / kps_c) * entry.fold[None, :]
    pnoise = cp.abs(kno_s) ** 2 \
        * cp.asarray(effs ** 2)[:, None]
    ef2_d = entry.fold2[None, :] * pnoise \
        / cp.abs(kps_c) ** 2

    for j, gi in enumerate(gis):
        kim_parts[gi] = kim_d[j]
        ef2_parts[gi] = ef2_d[j]


class _GeomEntry(object):
    def __init__(self, grids):
        half_iy = np.asarray(grids['iy'])
        half_ix = np.asarray(grids['ix'])
        self.nm = half_iy.size
        self.iy = cp.asarray(half_iy.astype(np.int32))
        self.ix = cp.asarray(half_ix.astype(np.int32))
        self.kv = cp.asarray(np.asarray(grids['kv']))
        self.ku = cp.asarray(np.asarray(grids['ku']))
        self.fold = cp.asarray(np.asarray(grids['fold']))
        self.fold2 = cp.asarray(np.asarray(grids['fold2']))
        self.Atinv = np.asarray(grids['Atinv'])
        self.detAtinv = float(grids['detAtinv'])


class GeometryCache(object):
    """device copies of the ngmix k-space grids, keyed like the
    ngmix lru: (dim, dvdrow, dvdcol, dudrow, dudcol, Tsmooth).
    The grids are built on the CPU with the exact production code
    so the retained-mode selection is bitwise the CPU path's,
    then uploaded once.

    LRU-bounded: group cutout dims vary freely, so an unbounded
    cache grows for the whole run (a few MB of device arrays per
    distinct geometry).  Eviction is safe at any time — live
    PrepSlabs hold direct references to their entries."""

    def __init__(self, maxsize=48):
        self._entries = {}
        self._maxsize = int(maxsize)

    def get(self, dim, jac4, Tsmooth):
        from ngmix.prepsfadmom.prep import _get_kspace_grids

        key = (int(dim),) + tuple(float(v) for v in jac4) \
            + (float(Tsmooth),)
        if key in self._entries:
            # refresh recency (dict preserves insertion order)
            self._entries[key] = self._entries.pop(key)
        else:
            grids = _get_kspace_grids(*key)
            entry = _GeomEntry(grids)
            # flat rfft half-plane index for the gathers
            half = key[0] // 2 + 1
            entry.gidx = (
                entry.iy.astype(cp.int64) * half
                + entry.ix.astype(cp.int64)
            )
            self._entries[key] = entry
            while len(self._entries) > self._maxsize:
                self._entries.pop(next(iter(self._entries)))
        return key, self._entries[key]


class PrepSlab(object):
    """the resident output of prep_groups: mode CSR device arrays
    in group order plus per-group geometry references and the
    host-side measured init sums"""

    def __init__(self, kim, ef2, moff, geom_keys, geoms, esums):
        self.kim = kim            # complex128 device, CSR
        self.ef2 = ef2            # float64 device, CSR
        self.moff = moff          # host int64, ngroup+1
        self.geom_keys = geom_keys
        self.geoms = geoms        # {key: _GeomEntry}
        self.esums = esums        # list of host (nobj, 6) f8

    @property
    def ngroup(self):
        return len(self.geom_keys)

    def nm(self, gi):
        return int(self.moff[gi + 1] - self.moff[gi])

    def group_modes(self, gi):
        """device views of group gi's mode data + its geometry"""
        g = self.geoms[self.geom_keys[gi]]
        sl = slice(int(self.moff[gi]), int(self.moff[gi + 1]))
        return {
            'kim': self.kim[sl],
            'ef2': self.ef2[sl],
            'iy': g.iy,
            'ix': g.ix,
            'kv': g.kv,
            'ku': g.ku,
        }


def prep_groups(reqs, geom_cache, nt=NT,
                chunk_bytes=DEFAULT_CHUNK_BYTES):
    """
    device prep for a batch of groups.

    Parameters
    ----------
    reqs: list of dict, one per group
        image, noise, psf: 2d float64 host arrays (the image
            already apodized by the caller when ap_rad > 0)
        target_dim: int (the padded fft dim; must be <= DIM_MAX)
        eff_pad_factor: float
        jac4: (dvdrow, dvdcol, dudrow, dudcol)
        Tsmooth: float
        drow, dcol, df2: float (epoch scalars)
        dv, du: (nobj,) float64 (positions minus vcen/ucen)
        sw: (nobj, 3) float64 guess weight matrices
    geom_cache: GeometryCache

    Returns
    -------
    PrepSlab
    """
    ngroup = len(reqs)

    geom_keys = []
    geoms = {}
    for r in reqs:
        # the init_sums kernel runs in the fp64 module, whose
        # phasor tables cap at DIM_MAX_FP64; bigger groups (up to
        # the fp32 DIM_MAX) go through CPU prep + the host-pack
        # path instead
        if r['target_dim'] > DIM_MAX_FP64:
            raise ValueError(
                f"target_dim {r['target_dim']} exceeds "
                f"DIM_MAX_FP64 {DIM_MAX_FP64}"
            )
        key, entry = geom_cache.get(
            r['target_dim'], r['jac4'], r['Tsmooth'],
        )
        geom_keys.append(key)
        geoms[key] = entry

    # ---- batched pad + rfft per (dim, geometry) class ----
    # rows 3*j + (0 image, 1 noise, 2 psf) within each class
    classes = {}
    for gi, r in enumerate(reqs):
        classes.setdefault((r['target_dim'], geom_keys[gi]),
                           []).append(gi)

    kim_parts = [None] * ngroup
    ef2_parts = [None] * ngroup
    for (D, gkey), all_gis in classes.items():
        entry = geoms[gkey]
        # chunk the batched ffts: a dense field can put tens of
        # groups in one class, and an unchunked (3B, D, D) stack
        # plus its transform spikes to a GB of transient device
        # memory that then sits cached in the feeder's pool
        per = max(1, int(chunk_bytes // (3 * D * D * 8)))
        for c0 in range(0, len(all_gis), per):
            gis = all_gis[c0:c0 + per]
            _prep_class_chunk(
                reqs, gis, D, entry, kim_parts, ef2_parts,
            )

    moff = np.zeros(ngroup + 1, dtype=np.int64)
    for gi in range(ngroup):
        moff[gi + 1] = moff[gi] + kim_parts[gi].size
    kim_cat = cp.concatenate(kim_parts)
    ef2_cat = cp.concatenate(ef2_parts)

    # ---- per-object measured init sums (device admom_ksums) ----
    distinct = list(dict.fromkeys(geom_keys))
    goff = np.zeros(len(distinct) + 1, dtype=np.int64)
    for k, key in enumerate(distinct):
        goff[k + 1] = goff[k] + geoms[key].nm
    iy_geo = cp.concatenate([geoms[k].iy for k in distinct])
    ix_geo = cp.concatenate([geoms[k].ix for k in distinct])
    kv_geo = cp.concatenate([geoms[k].kv for k in distinct])
    ku_geo = cp.concatenate([geoms[k].ku for k in distinct])
    geom_of_group = np.array(
        [distinct.index(k) for k in geom_keys], dtype=np.int32,
    )

    pair_group = []
    dv = []
    du = []
    sw = []
    for gi, r in enumerate(reqs):
        nobj = len(r['dv'])
        pair_group.extend([gi] * nobj)
        dv.extend(np.asarray(r['dv'], dtype=np.float64))
        du.extend(np.asarray(r['du'], dtype=np.float64))
        sw.append(np.asarray(r['sw'], dtype=np.float64
                             ).reshape(nobj, 3))
    npair = len(pair_group)
    sw = np.concatenate(sw).reshape(npair, 3)

    esums_d = cp.zeros(max(npair, 1) * 6, dtype=cp.float64)
    kern = get_init_sums_kernel(nt=nt)
    if npair:
        kern((npair,), (int(nt),), (
        kim_cat, cp.asarray(moff),
        iy_geo, ix_geo, kv_geo, ku_geo,
        cp.asarray(goff), cp.asarray(geom_of_group),
        cp.asarray(np.array([r['target_dim'] for r in reqs],
                            dtype=np.int32)),
        cp.asarray(np.array([r['df2'] for r in reqs])),
        cp.asarray(np.array([r['drow'] for r in reqs])),
        cp.asarray(np.array([r['dcol'] for r in reqs])),
        cp.asarray(np.concatenate(
            [np.asarray(geoms[k].Atinv).ravel()
             for k in geom_keys])),
        cp.asarray(np.array(pair_group, dtype=np.int32)),
        cp.asarray(np.array(dv)), cp.asarray(np.array(du)),
        cp.asarray(sw.ravel()),
        esums_d,
        ))
    esums_h = cp.asnumpy(esums_d)[:npair * 6].reshape(npair, 6)

    esums = []
    p0 = 0
    for r in reqs:
        nobj = len(r['dv'])
        esums.append(esums_h[p0:p0 + nobj].copy())
        p0 += nobj

    return PrepSlab(kim_cat, ef2_cat, moff, geom_keys, geoms,
                    esums)


def assemble_mode_fields(slab_refs, fp32):
    """
    concatenate the mode fields of the referenced groups into the
    kernel-ready device arrays (the device analog of the mode
    half of to_gpu_multi), casting to the fp32 dtypes when
    requested.

    Parameters
    ----------
    slab_refs: list of (PrepSlab, gi) in batch group order
    fp32: bool

    Returns
    -------
    mode_dev dict with kim/iy/ix/kv/ku/ef2 device arrays
    """
    kdt = cp.complex64 if fp32 else cp.complex128
    mdt = cp.float32 if fp32 else cp.float64

    parts = {k: [] for k in ('kim', 'iy', 'ix', 'kv', 'ku',
                             'ef2')}
    for slab, gi in slab_refs:
        m = slab.group_modes(gi)
        parts['kim'].append(m['kim'].astype(kdt))
        parts['ef2'].append(m['ef2'].astype(mdt))
        parts['kv'].append(m['kv'].astype(mdt))
        parts['ku'].append(m['ku'].astype(mdt))
        parts['iy'].append(m['iy'])
        parts['ix'].append(m['ix'])
    return {k: cp.concatenate(v) for k, v in parts.items()}
