"""[repro] CONST-copy-floor-method -- pattern-library/constants.md

Methodology constant: copy-floor calibration probe. Build a probe with the
SAME persistent structure / SAME GM traffic mix as the target elementwise
multi-input kernel (3 loads + 1 store), replacing the transit chain with two
vadds (out = (a + b) + w): they keep all loads alive (DCE-proof) while
shrinking the vector chain to 2 passes, so the measured Task Duration
approximates the MTE2/DRAM floor of the traffic mix.

Insight (measured on lerp_tensor 2026-09-07): at >=16M elems per input the
full 7-pass fp32-transit chain measured within 1.1us of this 2-pass floor --
auto multi-buffer hides the vector chain inside the MTE2 window, so
compute-chain optimizations have ROI ~ 0 there; optimize the transfer
dimension instead. Below ~1M (mte2_ratio 0.70-0.84) the chain is exposed.

Assertion (constant class): the probe produces out == a + b + w exactly (the
probe itself is the calibrated artifact). Measurement (on demand, NOT run by
repro_runner): wrap in msprof op, compare vs the target kernel at the same
N/dtype -- delta < ~1.1us at >=16M means copy-floor bound.

First verified: lerp_tensor optimize 2026-09-07 (probe_copy.py, examples/
TileOPs/.../lerp_tensor_kernel/perf_opt/ -- provenance, may rot).
origin_task (provenance): lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
Re-verification log (append-only):
  2026-09-10 PASS runnable part (extracted & minimized from probe_copy.py).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401
from tilelang.utils.npu_utils import NPUUtils

DEFAULT_BLOCK = {"float32": 2048, "float16": 4096, "bfloat16": 4096}


def make_copy_floor_probe(N: int, dtype: str):
    """Same persistent structure / same 3:1 traffic mix as the lerp kernel;
    compute chain shrunk to two DCE-proof vadds."""
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    block_size = DEFAULT_BLOCK[dtype]

    @tilelang.jit(out_idx=[-1], target="npuir")
    def kernel(block_size=block_size):
        num_logical = T.ceildiv(N, block_size)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = T.ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            w: T.Tensor((N,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((block_size,), dtype)
                b_ub = T.alloc_ub((block_size,), dtype)
                w_ub = T.alloc_ub((block_size,), dtype)
                out_ub = T.alloc_ub((block_size,), dtype)
                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        t0 = block_id * block_size
                        tail = T.min(block_size, N - t0)
                        T.copy(a[t0 : t0 + tail], a_ub[0:tail])
                        T.copy(b[t0 : t0 + tail], b_ub[0:tail])
                        T.copy(w[t0 : t0 + tail], w_ub[0:tail])
                        T.vadd(a_ub, b_ub, out_ub)
                        T.vadd(out_ub, w_ub, out_ub)
                        T.copy(out_ub[0:tail], out[t0 : t0 + tail])

        return main

    return kernel


def main() -> None:
    N = 1 << 20  # runnable check uses a small N; measurement uses >=16M
    result = make_copy_floor_probe(N, "float16")()
    mk = lambda: torch.randn(N, dtype=torch.float16, device="npu")  # noqa: E731
    a, b, w = mk(), mk(), mk()
    res = result(a, b, w)
    out = res[0] if isinstance(res, tuple) else res
    torch.npu.synchronize()
    got = out.reshape(N).float().cpu()
    exp = a.float().cpu() + b.float().cpu() + w.float().cpu()
    max_diff = (got - exp).abs().max().item()
    assert max_diff <= 1e-2, "copy-floor probe must be numerically exact"
    print(
        f"CONST-copy-floor-method: PASS (probe exact, N={N}, max_diff={max_diff:.2e})"
    )


if __name__ == "__main__":
    main()
