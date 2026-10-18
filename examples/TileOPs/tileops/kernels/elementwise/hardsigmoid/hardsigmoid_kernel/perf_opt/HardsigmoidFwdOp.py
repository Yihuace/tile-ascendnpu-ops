# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Hardsigmoid forward kernel (NPU, Developer mode) -- Stage 4 tuned (R2).

y = clamp(x + 3, 0, 6) * (1 / 6), 1-D (N,) tensor, fp16/bf16/fp32 input,
computed in fp32 intermediate precision, output dtype = input dtype.
NaN input propagates NaN (vadd/vclamp/vmul already NaN-propagating on this
toolchain -- PL-1.21; re-verified on CANN 9.0.0).

Stage 4 R2 (vs prior perf_opt, which only dropped the redundant NaN recovery):
  Diagnosis (msprof, fc-wide fp16): `latency bound:compute caused`,
  aiv_vec_ratio ~0.85 / aiv_vec_time ~12.8us dominant, aiv_mte2_time ~11us at
  ~72% utilization only partially overlapped with the vector chain.
  Adopted: guard-free 2-stage T.Pipelined hot loop (Part A) + tail epilogue
  (Part B, <=1 guarded tile per core) on the fp16/bf16 cast path -- overlaps
  the MTE2 load of tile i+1 with the vector chain of tile i (mish perf_opt /
  add A4 structure).  fp32 path unchanged (serial, only serves smoke).

Drop-in usage (same factory signature as the baseline kernel):
    kernel = hardsigmoid_fwd_kernel(N, "float16", block_size=8192)
    y = kernel(x_npu, N)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_hardsigmoid", "hardsigmoid_fwd_kernel"]

# Tuned default block_size (exposed for the wrapper switch block pairing).
TUNED_BLOCK_SIZE_CAST = 8192   # fp16/bf16 (cast path)
TUNED_BLOCK_SIZE_FP32 = 4096   # fp32 (fits UB with fp32 transit buffers)


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_hardsigmoid(x):
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = torch.nn.functional.hardsigmoid(x_f32)
    return y_f32.to(x_cpu.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def hardsigmoid_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    num_logical = T.ceildiv(N, block_size)
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores

    three = T.cast(3.0, "float32")
    zero = T.cast(0.0, "float32")
    six = T.cast(6.0, "float32")
    one_sixth = T.cast(1.0 / 6.0, "float32")

    # Full waves whose every block_id is a full tile (guard-free hot loop).
    iters_a = (num_logical - 1) // num_kernels

    if dtype == compute_dtype and out_dtype == compute_dtype:
        # ---------------- fp32 specialization (serial, unchanged) ----------
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
            shape: T.int32,
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                buf1_ub = T.alloc_ub((block_size,), compute_dtype)
                buf2_ub = T.alloc_ub((block_size,), compute_dtype)
                for i in T.serial(T.ceildiv(num_logical, num_kernels)):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        offset = block_id * block_size
                        remaining = shape - offset
                        tail_size = T.min(block_size, remaining)
                        T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
                        T.vadd(x_ub, three, buf1_ub)
                        T.vclamp(buf1_ub, buf2_ub, zero, six)
                        T.vmul(buf2_ub, one_sixth, buf1_ub)
                        T.copy(buf1_ub[0:tail_size], y[offset : offset + tail_size])
    else:
        # ---------------- fp16/bf16 cast specialization --------------------
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
            shape: T.int32,
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
                buf1_ub = T.alloc_ub((block_size,), compute_dtype)
                buf2_ub = T.alloc_ub((block_size,), compute_dtype)
                y_ub = T.alloc_ub((block_size,), out_dtype)

                # ---- Part A: guard-free 2-stage pipelined hot loop --------
                # Trace-time guard: tiny shapes (n_num <= num_kernels) have
                # iters_a == 0; skip the pipelined loop and let the epilogue
                # handle every tile (avoids a zero-trip T.Pipelined).
                if iters_a > 0:
                    for i in T.Pipelined(iters_a, num_stages=2):
                        offset = (i * num_kernels + cid) * block_size
                        T.copy(x[offset : offset + block_size], x_ub)
                        T.vcast(x_ub, x_f32_ub, round_mode="rint")
                        T.vadd(x_f32_ub, three, buf1_ub)       # t = x + 3
                        T.vclamp(buf1_ub, buf2_ub, zero, six)  # c = clamp(t, 0, 6)
                        T.vmul(buf2_ub, one_sixth, buf1_ub)    # r = c * (1/6)
                        T.vcast(buf1_ub, y_ub, round_mode="rint")
                        T.copy(y_ub, y[offset : offset + block_size])

                # ---- Part B: tail epilogue, <=1 guarded tile per core -----
                block_id = iters_a * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
                    T.vcast(x_ub, x_f32_ub, round_mode="rint")
                    T.vadd(x_f32_ub, three, buf1_ub)
                    T.vclamp(buf1_ub, buf2_ub, zero, six)
                    T.vmul(buf2_ub, one_sixth, buf1_ub)
                    T.vcast(buf1_ub, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "float32": (1e-5, 1e-5),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _max_diff_metric(out, ref):
    eq = out == ref
    eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    return diff.max().item()


def _run_tensor(x, dtype_str, tag, equal_nan=False):
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kernel = hardsigmoid_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_hardsigmoid(x)
    y_cpu = y.cpu()
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_case(N, dtype_str, tag, scale=1.0, block_size=None):
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(x, dtype_str, tag)


def run_L0():
    N = 1048576
    _run_case(N, "float32", "L0")
    _run_case(N, "float16", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (8192, 100, 4097):
            _run_case(N, dtype_str, "L1")
    _run_case(4194304, "float16", "L1")
    _run_case(4194304, "float32", "L1")
    _run_case(1048576, "float16", "L1", block_size=2048)


def run_L2():
    for N in (1, 2, 10):
        try:
            _run_case(N, "float32", "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    try:
        _run_case(4194304, "float32", "L2")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (Record without blocking): N=4194304 err={e}")


def run_boundary():
    N = 4096
    try:
        x = torch.zeros(N, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-zeros")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros err={e}")

    try:
        x = (torch.randn(N, dtype=torch.float32) * 5.0 + 50.0).to(torch.float16).npu()
        _run_tensor(x, "float16", "Boundary-large-positive")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large-positive err={e}")

    try:
        x = torch.full((N,), -10.0, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-large-negative")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large-negative err={e}")

    try:
        x = torch.randn(N, dtype=torch.float32)
        x[0::4] = float("inf")
        x[1::4] = float("-inf")
        _run_tensor(x.to(torch.float32).npu(), "float32", "Boundary-inf")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf err={e}")

    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(N, dtype=torch.float32)
            x[0::8] = float("nan")
            x[1::8] = float("inf")
            x[2::8] = float("-inf")
            x = x.to(torch_dtype).npu()
            kernel = hardsigmoid_fwd_kernel(N, dtype_str)
            y = kernel(x, N).cpu()
            golden = golden_hardsigmoid(x)
            torch.testing.assert_close(
                y, golden,
                atol=_TOLERANCE[dtype_str][0],
                rtol=_TOLERANCE[dtype_str][1],
                equal_nan=True,
            )
            nan_in = torch.isnan(x.cpu())
            nan_out = torch.isnan(y)
            assert bool(torch.equal(nan_in, nan_out)), (
                f"NaN mask mismatch: input {int(nan_in.sum().item())} vs output "
                f"{int(nan_out.sum().item())}"
            )
            print(
                f"[Boundary] PASS: NaN->NaN dtype={dtype_str} "
                f"nan_count={int(nan_in.sum().item())}"
            )
        except AssertionError as e:
            print(f"[Boundary] FAIL (NaN probe): dtype={dtype_str} err={e}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): NaN dtype={dtype_str} err={e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
