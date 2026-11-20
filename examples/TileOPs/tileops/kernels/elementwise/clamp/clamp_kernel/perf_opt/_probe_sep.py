"""Probe-only factory: non-in-place UB staging variants for the Tensor-bound
clamp kernel (NOT a deliverable, no gate evidence, no perf_records writes).

Motivation: the Round-1 in-place variant (`inp3`: `vmax`/`vmin` written back
into the input staging buffer) keeps `x_ub` live from load to store, which
appears to serialize the next tile's MTE2 prefetch -- its measured MTE2 read
bandwidth (1.57 TB/s @ bs8192) is far below the Stage 3 baseline's 8-buffer
structure (2.12 TB/s @ bs8192, mte2_time 44.25 us).  This file keeps the short
native chain but uses a SEPARATE output buffer so the staging buffers die right
after the vector op.

Variants (`variant=` factory kwarg):
  * ``sep``        -- 4 dtype buffers (x, lo, hi, out), `vclamp(x, out, lo, hi)`
                      (bf16: + 3 fp32 transit, clamp in place in the transit
                      buffer, rounded into the separate out buffer).
  * ``sep_static`` -- ``sep`` + static-extent full tiles and a constant-extent
                      remainder tile (PL-1.34).

Only the both-bounds path is implemented (the benchmark always carries both
Tensor bounds).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
from tilelang.utils.npu_utils import NPUUtils

_COMPUTE_DTYPE = "float32"


def _make_clamp_tensor_kernel(N, dtype, has_min, has_max, variant="sep"):
    if not (has_min and has_max):
        raise ValueError("probe variants implement only the both-bounds path")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported dtype {dtype!r}")
    if N <= 0:
        raise ValueError(f"N must be positive, got N={N}")

    vector_cores = NPUUtils.get().get_aicore_num() * 2
    native16 = dtype == "float16"
    transit = dtype == "bfloat16"  # mandatory fp32 transit (no bf16 v-op dtype)
    static = variant == "sep_static"

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(block_size):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        n_full = N // block_size
        tail_len = N - n_full * block_size
        has_tail = tail_len != 0
        num_logical = n_full + (1 if has_tail else 0)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        if transit and static:

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), dtype)
                    lo_ub = T.alloc_ub((block_size,), dtype)
                    hi_ub = T.alloc_ub((block_size,), dtype)
                    y_ub = T.alloc_ub((block_size,), dtype)
                    x32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    lo32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    hi32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)

                    for i in T.serial(num_local_tasks):
                        task = i * num_kernels + cid
                        if task < n_full:
                            off = task * block_size
                            T.copy(x[off : off + block_size], x_ub)
                            T.copy(lo[off : off + block_size], lo_ub)
                            T.copy(hi[off : off + block_size], hi_ub)
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(lo_ub, lo32, round_mode="rint")
                            T.vcast(hi_ub, hi32, round_mode="rint")
                            T.vclamp(x32, x32, lo32, hi32)
                            T.vcast(x32, y_ub, round_mode="rint")
                            T.copy(y_ub, y[off : off + block_size])
                        elif has_tail and task == n_full:
                            off = n_full * block_size
                            T.copy(x[off : off + tail_len], x_ub[0:tail_len])
                            T.copy(lo[off : off + tail_len], lo_ub[0:tail_len])
                            T.copy(hi[off : off + tail_len], hi_ub[0:tail_len])
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(lo_ub, lo32, round_mode="rint")
                            T.vcast(hi_ub, hi32, round_mode="rint")
                            T.vclamp(x32, x32, lo32, hi32)
                            T.vcast(x32, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail_len], y[off : off + tail_len])

            return main

        if transit:

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), dtype)
                    lo_ub = T.alloc_ub((block_size,), dtype)
                    hi_ub = T.alloc_ub((block_size,), dtype)
                    y_ub = T.alloc_ub((block_size,), dtype)
                    x32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    lo32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)
                    hi32 = T.alloc_ub((block_size,), _COMPUTE_DTYPE)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off = block_id * block_size
                            tail = T.min(block_size, N - off)
                            T.copy(x[off : off + tail], x_ub[0:tail])
                            T.copy(lo[off : off + tail], lo_ub[0:tail])
                            T.copy(hi[off : off + tail], hi_ub[0:tail])
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(lo_ub, lo32, round_mode="rint")
                            T.vcast(hi_ub, hi32, round_mode="rint")
                            T.vclamp(x32, x32, lo32, hi32)
                            T.vcast(x32, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off : off + tail])

            return main

        if static:

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), dtype)
                    lo_ub = T.alloc_ub((block_size,), dtype)
                    hi_ub = T.alloc_ub((block_size,), dtype)
                    y_ub = T.alloc_ub((block_size,), dtype)

                    for i in T.serial(num_local_tasks):
                        task = i * num_kernels + cid
                        if task < n_full:
                            off = task * block_size
                            T.copy(x[off : off + block_size], x_ub)
                            T.copy(lo[off : off + block_size], lo_ub)
                            T.copy(hi[off : off + block_size], hi_ub)
                            T.vclamp(x_ub, y_ub, lo_ub, hi_ub)
                            T.copy(y_ub, y[off : off + block_size])
                        elif has_tail and task == n_full:
                            off = n_full * block_size
                            T.copy(x[off : off + tail_len], x_ub[0:tail_len])
                            T.copy(lo[off : off + tail_len], lo_ub[0:tail_len])
                            T.copy(hi[off : off + tail_len], hi_ub[0:tail_len])
                            T.vclamp(x_ub, y_ub, lo_ub, hi_ub)
                            T.copy(y_ub[0:tail_len], y[off : off + tail_len])

            return main

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            lo: T.Tensor((N,), dtype),
            hi: T.Tensor((N,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                lo_ub = T.alloc_ub((block_size,), dtype)
                hi_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)

                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        off = block_id * block_size
                        tail = T.min(block_size, N - off)
                        T.copy(x[off : off + tail], x_ub[0:tail])
                        T.copy(lo[off : off + tail], lo_ub[0:tail])
                        T.copy(hi[off : off + tail], hi_ub[0:tail])
                        T.vclamp(x_ub, y_ub, lo_ub, hi_ub)
                        T.copy(y_ub[0:tail], y[off : off + tail])

        return main

    return kernel
