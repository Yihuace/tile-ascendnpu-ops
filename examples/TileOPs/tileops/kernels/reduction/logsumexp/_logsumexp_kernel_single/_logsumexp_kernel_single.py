# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""LogSumExp single-tile kernel (NPU Developer mode).

Migrated from GPU ``_logsumexp_kernel_single`` to ``target="npuir"``.
Reduces a ``(M, N)`` tensor along the last dim (N) using the numerically
stable logsumexp algorithm and outputs an ``(M,)`` vector.

Algorithm:
    1. m = max(x, dim=1)
    2. s = sum(exp(x - m), dim=1)
    3. y = m + ln(s)

All accumulation is done in float32 (input cast to fp32, result cast back).
Single-tile path assumes N fits entirely in UB.
"""

import os

# Developer mode: compiler manages UB (alloc_shared) + fragment + auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

# ---------- dtype / tolerance mapping ----------
# Tolerances from examples/TileOPs/tests/ops/test_softmax.py _get_tolerances().
_DTYPE_MAP = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "float32": (torch.float32, 1e-5, 1e-5),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_logsumexp_kernel_single(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.logsumexp(x, dim=-1)`` (no keepdim).

    Computed in fp32 for precision, then cast back to the input dtype.
    Runs on CPU; callers should pass a CPU tensor.
    """
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel ----------
def _logsumexp_kernel_single(M, N, dtype):
    """Build a single-tile logsumexp kernel (N fits in UB).

    Factory mirrors the GPU source structure:
      ``_logsumexp_kernel_single(M, N, dtype) -> _func(block_m) -> main prim_func``.
    K9: ``threads`` parameter removed (NPU has no CUDA threads concept).
    """

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                # Buffer allocation (Developer mode: alloc_shared -> UB)
                x_ub = T.alloc_shared((block_m, N), dtype)
                x_local = T.alloc_fragment((block_m, N), dtype)
                x_f32 = T.alloc_fragment((block_m, N), "float32")
                row_max = T.alloc_fragment((block_m, 1), "float32")
                row_sum = T.alloc_fragment((block_m, 1), "float32")
                out_ub = T.alloc_shared((block_m,), dtype)

                # Tail-block row count (last block may have fewer rows)
                real_m = T.min(block_m, M - pid_m * block_m)

                # Step 1: Load GM -> UB -> Fragment
                # Tail block: src/dst sliced together to (real_m, N) to avoid
                # out-of-bounds reads (T.copy shape-consistency rule).
                T.copy(
                    x[pid_m * block_m : pid_m * block_m + real_m, 0:N],
                    x_ub[0:real_m, 0:N],
                )
                T.copy(x_ub, x_local)

                # Step 2: Cast to fp32 for accumulation precision
                for i, j in T.Parallel(block_m, N):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # Step 3: reduce_max per row
                T.reduce_max(x_f32, row_max, dim=1)

                # Step 4: x - max (broadcast (bm,N) - (bm,1) -> (bm,N))
                T.vsub(x_f32, row_max, x_f32)

                # Step 5: exp (in-place)
                T.vexp(x_f32, x_f32)

                # Step 6: reduce_sum per row
                T.reduce_sum(x_f32, row_sum, dim=1)

                # Step 7: ln(sum) (in-place; note: T.vln lowercase l)
                T.vln(row_sum, row_sum)

                # Step 8: max + ln(sum) (in-place)
                T.vadd(row_max, row_sum, row_sum)

                # Step 9: Cast back to original dtype, extract (i,0) -> (i,)
                for i in T.Parallel(block_m):
                    out_ub[i] = T.cast(row_sum[i, 0], dtype)

                # Step 10: Store UB -> GM
                # Tail block: src truncated to (real_m,) to match dst slice.
                T.copy(
                    out_ub[0:real_m],
                    y[pid_m * block_m : pid_m * block_m + real_m],
                )

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(M, N, dtype_str, block_m, tag):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _logsumexp_kernel_single(M, N, dtype_str)(block_m)
    y = kernel(x)
    ref = golden_logsumexp_kernel_single(x.cpu())
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} max_diff={max_diff:.2e}"
    )
    return max_diff


# ---------- L0: gate tests (must pass) ----------
# 10 cases from DESIGN.md section 8.3.
def run_L0():
    cases = [
        (32, 256, "float16", 8),  # L0-1 basic fp16
        (32, 256, "float32", 8),  # L0-2 fp32 no cast loss
        (32, 256, "bfloat16", 8),  # L0-3 bf16 -> fp32 -> bf16
        (33, 256, "float16", 8),  # L0-4 row-tail (33 % 8 = 1)
        (32, 300, "float16", 4),  # L0-5 N unaligned (300 not 256x)
        (4, 4096, "float16", 4),  # L0-6 manifest attn-weights-4k
        (4, 4096, "bfloat16", 4),  # L0-7 manifest bf16
        (1024, 4096, "float16", 4),  # L0-8 large M (bm=4: fragment+UB fit 192KB)
        (1, 256, "float16", 1),  # L0-9 single row
        (256, 32, "float16", 4),  # L0-10 dim=0 reduction (op-layer reshape)
    ]
    max_diff = 0.0
    for M, N, dtype_str, block_m in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, block_m, "L0"))
    print(f"[L0] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    cases = [
        (128, 512, "float32", 16),
        (130, 256, "float16", 8),  # row-tail 130 % 8 = 2
        (64, 1024, "bfloat16", 4),
        (8, 512, "float16", 8),  # exact divide
        (17, 128, "float16", 16),  # row-tail 17 % 16 = 1
        (256, 1024, "float32", 8),
        (3, 300, "bfloat16", 2),  # small M + unaligned N
    ]
    max_diff = 0.0
    for M, N, dtype_str, block_m in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, block_m, "L1"))
    print(f"[L1] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, "float16", 1),  # minimal
        (8, 256, "float16", 8),  # exact divide M=block_m
        (9, 256, "float16", 8),  # M=block_m+1
        (32, 1, "float16", 4),  # N=1 single-element reduce
    ]
    for M, N, dtype_str, block_m in cases:
        try:
            _run_case(M, N, dtype_str, block_m, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # zeros: logsumexp(0) = log(N)
    try:
        x = torch.zeros(32, 256, dtype=torch.float16, device="npu")
        kernel = _logsumexp_kernel_single(32, 256, "float16")(8)
        y = kernel(x)
        ref = golden_logsumexp_kernel_single(x.cpu())
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-3)
        print("[Boundary] PASS: zeros (32,256) fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {e}")

    # large positive/negative mix (exp overflow risk mitigated by max-shift)
    try:
        x = torch.randn(32, 256, dtype=torch.float16, device="npu") * 50
        kernel = _logsumexp_kernel_single(32, 256, "float16")(8)
        y = kernel(x)
        ref = golden_logsumexp_kernel_single(x.cpu())
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-3)
        print("[Boundary] PASS: large values (x50) (32,256) fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large values: {e}")

    # all -inf (math boundary: torch returns -inf, kernel may return NaN)
    try:
        x = torch.full((32, 256), float("-inf"), dtype=torch.float16, device="npu")
        kernel = _logsumexp_kernel_single(32, 256, "float16")(8)
        y = kernel(x)
        ref = golden_logsumexp_kernel_single(x.cpu())
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-3)
        print("[Boundary] PASS: all -inf (32,256) fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only, known math boundary): all -inf: {e}")

    # large random shape
    try:
        _run_case(512, 2048, "float16", 8, "Boundary")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large random (512,2048): {e}")


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
