# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""R2 branch ``r2_native``: clamp in the *input* dtype (drop the fp32 cast chain).

Single main change vs ``ClampScalarFwdOp_r2_parent.py``: the clamp comparison
runs natively on fp16 / fp32 buffers instead of the ``vcast -> clamp -> vcast``
fp32 transit chain.  bf16 keeps the fp32 transit because the NPU vector
compare/select ops do not support bf16
(``docs/Tilelang.language/比较操作/T.vclamp.md`` §2.2.1 and ``T.vmax.md`` /
``T.vmin.md`` §2.2.1 list bf16 as unsupported).

Trace-time fold: the scalar bound is pre-rounded to the compute dtype
(``struct`` round-trip, round-half-even) so the comparison constant is exactly
the dtype value the golden's final rounding would land on: rounding is
monotone, therefore ``rn(clamp(x, lo, hi)) == clamp(x, rn(lo), rn(hi))`` for
every input-dtype value ``x`` (differences are bounded by 1 ulp and vanish on
the dtype grid).  The task loop / persistent structure is untouched.
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
    if fmt is None:  # bf16 keeps the fp32 constant
        return float(value)
    return float(struct.unpack(fmt, struct.pack(fmt, float(value)))[0])


@tilelang.jit(out_idx=[1], target="npuir")
def clamp_scalar_fwd_kernel(
    N, dtype, output_dtype=None, min_val=None, max_val=None, block_size=8192, num_cores=None
):
    """Clamp (scalar bounds) forward kernel, native-dtype compare path."""
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    # Native compare path: fp16 / fp32 vector min/max support; bf16 does not.
    native = dtype in ("float16", "float32") and out_dtype == dtype

    num_logical = T.ceildiv(N, block_size)
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_logical, num_kernels)

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

            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    tail_size = T.min(block_size, shape - offset)

                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    if nan_bound:
                        if native:
                            T.vmul(x_ub, 0.0, y_ub)   # finite->0; +-inf/NaN->NaN
                            T.vdiv(y_ub, 0.0, y_ub)
                        else:
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vmul(x32, 0.0, buf1)
                            T.vdiv(buf1, 0.0, buf1)
                            T.vcast(buf1, y_ub, round_mode="rint")
                    elif native:
                        # One pass per present bound, straight into the output
                        # buffer (dst == src1 form only: no src2-aliasing tax).
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

                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _r2_harness

    _r2_harness.main(lambda: clamp_scalar_fwd_kernel)
