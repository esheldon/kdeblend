# flake8: noqa
"""
GPU backend for the group-replace fitter (requires cupy and an
NVIDIA device).

The unit of work is a BATCH of groups: build deblenders with
kdeblend.build_deblender as usual, then

    from kdeblend.gpu import fit_groups
    reslist = fit_groups(debs, fp32=True)

runs every fit loop in one kernel launch, writes the converged
state back into each deblender (indistinguishable from deb.go())
and returns the same result dicts deblend() produces.  There is
deliberately no per-group entry point: one group cannot fill the
device.

For pipeline feeders that manage device transfers themselves the
lower-level pieces are exported: pack_host (host-side packing,
shared-memory friendly), to_gpu / to_gpu_multi (device upload,
the multi variant copies per-submission mode arrays straight
into preallocated slabs), launch_gpu / fetch_out, and the
writeback / install_sum_overrides / result_from_state trio.
"""

from ._core import (
    get_kernel, pack_host, pack_groups, to_gpu, to_gpu_multi,
    launch_gpu, fetch_out, run_gpu, MODE_FIELDS, TYPECODE,
    TYPENAME, GMAX, DIM_MAX, NT,
)
from .fit import (
    fit_groups, writeback, install_sum_overrides,
    result_from_state, FP32_TOL_FLOOR,
)
