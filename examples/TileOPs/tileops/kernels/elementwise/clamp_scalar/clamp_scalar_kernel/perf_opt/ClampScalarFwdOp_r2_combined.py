# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""R2 branch ``r2_combined``: native-dtype clamp + A4 static-extent pipelined tiles.

Two single-point changes carried together (each already screened separately
against ``r2-parent``):

1. ``r2-native``: clamp in the input dtype (fp16/fp32) instead of the
   ``vcast -> clamp -> vcast`` fp32 transit chain; bf16 keeps the fp32 transit
   (vector min/max have no bf16 support, see ``T.vclamp.md`` / ``T.vmax.md``
   §2.2.1).
2. ``r2-a4``: the task space is split into static-extent *full* tiles (hot
   loop, ``T.Pipelined(num_stages=2)``) plus one dynamic tail tile owned by
   core 0, so the hot loop no longer recomputes ``T.min`` / dynamic copy
   extents per tile.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse
import math
import struct

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

_STUCT_FMT = {"float16": "e", "float32": "f"}


def _round_bound(value, dtype_str):
    """Round a Python scalar bound onto the ``dtype_str`` grid (round-half-even)."""
    fmt = _STUCT_FMT.get(dtype_str)
    if fmt is None:
        return float(value)
    return float(struct.unpack(fmt, struct.pack(fmt, float(value)))[0])


@tilelang.jit(out_idx=[1], target="npuir")
def clamp_scalar_fwd_kernel(
    N, dtype, output_dtype=None, min_val=None, max_val=None, block_size=8192, num_cores=None
):
    """Clamp (scalar bounds) forward kernel: native clamp + A4 tile structure."""
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    native = dtype in ("float16", "float32") and out_dtype == dtype

    num_full = N // block_size
    tail = N - num_full * block_size
    num_logical = num_full + (1 if tail else 0)
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)

    has_min = min_val is not None
    has_max = max_val is not None
    nan_bound = (has_min and math.isnan(float(min_val))) or (
        has_max and math.isnan(float(max_val))
    )
    if native:
        min_scalar = _round_bound(min_val, dtype) if has_min else None
        max_scalar = _round_bound(max_val, dtype) if has_max else None
    else:
        min_scalar = float(min_val) if has_min else None
        max_scalar = float(max_val) if has_max else None

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            x32 = T.alloc_ub((block_size,), compute_dtype)
            buf1 = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            if num_full > 0:
                for i in T.Pipelined(num_local_tasks, num_stages=2):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size
                        T.copy(x[offset : offset + block_size], x_ub)

                        if nan_bound:
                            if native:
                                T.vmul(x_ub, 0.0, y_ub)
                                T.vdiv(y_ub, 0.0, y_ub)
                            else:
                                T.vcast(x_ub, x32, round_mode="rint")
                                T.vmul(x32, 0.0, buf1)
                                T.vdiv(buf1, 0.0, buf1)
                                T.vcast(buf1, y_ub, round_mode="rint")
                        elif native:
                            if has_min and has_max:
                                T.vmax(x_ub, min_scalar, y_ub)
                                T.vmin(y_ub, max_scalar, y_ub)
                            elif has_min:
                                T.vmax(x_ub, min_scalar, y_ub)
                            else:
                                T.vmin(x_ub, max_scalar, y_ub)
                        else:
                            T.vcast(x_ub, x32, round_mode="rint")
                            if has_min and has_max:
                                T.vmax(x32, min_scalar, buf1)
                                T.vmin(buf1, max_scalar, buf1)
                            elif has_min:
                                T.vmax(x32, min_scalar, buf1)
                            else:
                                T.vmin(x32, max_scalar, buf1)
                            T.vcast(buf1, y_ub, round_mode="rint")

                        T.copy(y_ub, y[offset : offset + block_size])

            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(x[offset : offset + tail], x_ub[0:tail])

                    if nan_bound:
                        if native:
                            T.vmul(x_ub, 0.0, y_ub)
                            T.vdiv(y_ub, 0.0, y_ub)
                        else:
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vmul(x32, 0.0, buf1)
                            T.vdiv(buf1, 0.0, buf1)
                            T.vcast(buf1, y_ub, round_mode="rint")
                    elif native:
                        if has_min and has_max:
                            T.vmax(x_ub, min_scalar, y_ub)
                            T.vmin(y_ub, max_scalar, y_ub)
                        elif has_min:
                            T.vmax(x_ub, min_scalar, y_ub)
                        else:
                            T.vmin(x_ub, max_scalar, y_ub)
                    else:
                        T.vcast(x_ub, x32, round_mode="rint")
                        if has_min and has_max:
                            T.vmax(x32, min_scalar, buf1)
                            T.vmin(buf1, max_scalar, buf1)
                        elif has_min:
                            T.vmax(x32, min_scalar, buf1)
                        else:
                            T.vmin(x32, max_scalar, buf1)
                        T.vcast(buf1, y_ub, round_mode="rint")

                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _r2_harness

    _r2_harness.main(lambda: clamp_scalar_fwd_kernel)
