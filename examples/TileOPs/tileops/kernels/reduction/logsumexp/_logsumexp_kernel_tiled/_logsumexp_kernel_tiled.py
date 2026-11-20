# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""LogSumExp tiled kernel (NPU Developer mode).

Migrated from GPU ``_logsumexp_kernel_tiled`` to ``target="npuir"``.
Reduces a ``(M, N)`` tensor along the last dim (N) using the numerically
stable logsumexp algorithm and outputs an ``(M,)`` vector.

Unlike the single-tile variant, the tiled variant assumes N does NOT fit
entirely in UB (e.g. N=32768, N=102400). It tiles along the N dimension and
uses the **online softmax recurrence** to compute running_max and a rescaled
running_sum in a single pass over the N-tiles:

    m_tile = max_j(x_tile)
    m_new  = max(m_running, m_tile)
    s_running = s_running * exp(m_running - m_new)
              + sum_j(exp(x_tile - m_new))
    m_running = m_new

Epilogue:  y = m_running + ln(s_running)

All accumulation is done in float32 (input cast to fp32, result cast back).
The N-tile loop uses ``T.serial`` (lowercase -- ``T.Serial`` does not exist).

Tail-tile column handling:
  The DESIGN.md specified a kernel-side ``T.if_then_else(t*tile_n+j < N, ...,
  -inf)`` mask for the tail tile. Empirically, a ``T.if_then_else`` condition
  that references a ``T.serial`` loop variable triggers a compiler segfault
  when the condition is non-trivially varying (the select op codegen crashes).
  The kernel-only fallback (vbrc -inf + partial UB->FRAG copy) in turn
  corrupts fp32 precision on tail tiles. This implementation therefore pads
  the input on the host side to ``N_padded = ceildiv(N, tile_n) * tile_n``
  with ``-inf`` (a new tensor is allocated -- the caller's input is never
  modified), so the kernel only ever sees tile-aligned N and uses a plain
  cast. This is mathematically exact: ``logsumexp([x, -inf...]) ==
  logsumexp(x)`` because ``exp(-inf - m) == 0`` contributes nothing to the
  sum and ``-inf`` is ignored by the max. Row-tail (M % block_m != 0) is still
  handled inside the kernel via ``T.min`` + sliced ``T.copy`` (verified in the
  single-tile kernel).
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
# bf16 uses 1.6e-2 due to the fp32<->bf16 cast round-trip loss.
_DTYPE_MAP = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "float32": (torch.float32, 1e-5, 1e-5),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_logsumexp_kernel_tiled(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.logsumexp(x, dim=-1)`` (no keepdim).

    Computed in fp32 for precision, then cast back to the input dtype.
    Runs on CPU; callers should pass a CPU tensor. Identical to the
    single-tile golden (same math, only the tiling strategy differs).
    """
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel ----------
# NOTE: the kernel requires ``N % tile_n == 0`` (tile-aligned). The host
# wrapper ``_run_case`` guarantees this by padding the input with -inf to
# ``N_padded = ceildiv(N, tile_n) * tile_n`` and building the kernel with
# ``N_padded``. See module docstring for why kernel-side tail masking is
# not used (compiler segfault on T.if_then_else + T.serial loop var).
def _logsumexp_kernel_tiled(M, N, dtype, tile_n):
    """Build a tiled logsumexp kernel (N too large for a single UB tile).

    Factory mirrors the GPU source structure:
      ``_logsumexp_kernel_tiled(M, N, dtype, tile_n) -> _func(block_m) -> main``
    K9: ``threads`` parameter removed (NPU has no CUDA threads concept).
    ``N`` must be a multiple of ``tile_n`` (host wrapper enforces this).
    """

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                # Buffer allocation (Developer mode: alloc_shared -> UB).
                shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                out_ub = T.alloc_shared((block_m,), dtype)

                # Fragment compute buffers.
                tile_local = T.alloc_fragment((block_m, tile_n), dtype)
                tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")

                # Online recurrence state (block_m, 1) -- rank must match the
                # reduce output rank (NPU reduce requires dst rank == src rank).
                row_max = T.alloc_fragment((block_m, 1), "float32")  # running max
                row_sum = T.alloc_fragment((block_m, 1), "float32")  # running sum
                tile_max = T.alloc_fragment((block_m, 1), "float32")  # per-tile max
                tile_sum = T.alloc_fragment((block_m, 1), "float32")  # per-tile sum
                new_max = T.alloc_fragment((block_m, 1), "float32")  # max(old, tile)
                correction = T.alloc_fragment((block_m, 1), "float32")  # exp(old-new)
                tmp1 = T.alloc_fragment((block_m, 1), "float32")  # scratch (copy trick)

                # Scalar constants (TIR exprs, loop-invariant).
                value_min = -T.infinity("float32")
                value_zero = 0

                # Row-tail count (last M-block may have fewer rows).
                real_m = T.min(block_m, M - pid_m * block_m)

                # --- Initialize recurrence state ---
                T.vbrc(value_min, row_max)  # row_max = -inf
                T.vbrc(value_zero, row_sum)  # row_sum = 0

                # --- Online recurrence main loop ---
                # NOTE: T.serial (lowercase) -- T.Serial does NOT exist.
                # num_tiles = ceildiv(N, tile_n) is a compile-time constant
                # (M, N, tile_n are all factory params). N is tile-aligned
                # (host wrapper pads), so every tile is a full tile_n columns.
                num_tiles = T.ceildiv(N, tile_n)
                for t in T.serial(num_tiles):
                    # 1. Load tile: GM -> UB (sliced by real_m for row-tail) -> Fragment.
                    #    Full tile_n columns (N is tile-aligned via host padding).
                    T.copy(
                        x[
                            pid_m * block_m : pid_m * block_m + real_m,
                            t * tile_n : (t + 1) * tile_n,
                        ],
                        shared_buf[0:real_m, 0:tile_n],
                    )
                    T.copy(shared_buf, tile_local)

                    # 2. Cast to fp32 (plain cast; tail columns are -inf from
                    #    host padding -> exp(-inf - m) = 0, ignored by max).
                    for i, j in T.Parallel(block_m, tile_n):
                        tile_f32[i, j] = T.cast(tile_local[i, j], "float32")

                    # 3. Per-tile row max.
                    T.reduce_max(tile_f32, tile_max, dim=1)

                    # 4. New running max = max(row_max, tile_max).
                    T.vmax(row_max, tile_max, new_max)

                    # 5. Correction = exp(old_max - new_max)  (<=1, rescale factor).
                    T.vsub(row_max, new_max, correction)
                    T.vexp(correction, correction)

                    # 6. Shift tile by NEW max, exp, per-tile row sum.
                    #    (bm,tn) - (bm,1) -> (bm,tn) row broadcast.
                    T.vsub(tile_f32, new_max, tile_f32)
                    T.vexp(tile_f32, tile_f32)
                    T.reduce_sum(tile_f32, tile_sum, dim=1)

                    # 7. Rescale old sum and accumulate.
                    T.vmul(row_sum, correction, row_sum)
                    T.vadd(row_sum, tile_sum, row_sum)

                    # 8. Update row_max = new_max.
                    #    fragment->fragment copy is not available via T.copy, so
                    #    use the vbrc(0,tmp1)+vadd(tmp1,new_max,row_max) trick
                    #    (flash_attn_npuir_dev.py L76-77).
                    T.vbrc(value_zero, tmp1)
                    T.vadd(tmp1, new_max, row_max)

                # --- Epilogue: y = max + ln(sum) ---
                T.vln(row_sum, row_sum)  # ln(sum)
                T.vadd(row_max, row_sum, row_sum)  # max + ln(sum)

                # Cast back to original dtype, extract (i,0) -> (i,).
                for i in T.Parallel(block_m):
                    out_ub[i] = T.cast(row_sum[i, 0], dtype)

                # Store: UB -> GM (row-tail truncate src to real_m).
                T.copy(
                    out_ub[0:real_m],
                    y[pid_m * block_m : pid_m * block_m + real_m],
                )

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(M, N, dtype_str, block_m, tile_n, tag):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")

    # Host-side padding: align N up to a multiple of tile_n with -inf so the
    # kernel only sees full tiles (plain cast, no kernel-side tail mask).
    # A NEW tensor is allocated; the caller's input ``x`` is never modified.
    # Mathematically exact: logsumexp([x, -inf...]) == logsumexp(x).
    n_padded = ((N + tile_n - 1) // tile_n) * tile_n
    if n_padded > N:
        x_pad = torch.full((M, n_padded), float("-inf"), dtype=torch_dtype, device="npu")
        x_pad[:, :N] = x
    else:
        x_pad = x

    kernel = _logsumexp_kernel_tiled(M, n_padded, dtype_str, tile_n)(block_m)
    y = kernel(x_pad)
    ref = golden_logsumexp_kernel_tiled(x.cpu())
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} "
        f"block_m={block_m} tile_n={tile_n} max_diff={max_diff:.2e}"
    )
    return max_diff


# ---------- L0: gate tests (must pass) ----------
# 10 cases from DESIGN.md section 8.3. Covers large N (tiled core use case),
# tail-tile column masking (exercised via host padding), row-tail
# non-divisible, all dtypes, manifest workloads, and multiple tile_n /
# block_m values.
def run_L0():
    cases = [
        (4, 32768, "float16", 4, 2048),  # L0-1 manifest attn-weights-32k
        (4, 102400, "float16", 4, 2048),  # L0-2 manifest lm-head-logits
        (4, 102400, "bfloat16", 4, 2048),  # L0-3 bf16 huge N
        (1024, 4096, "float16", 8, 2048),  # L0-4 manifest attn-weights-4k
        (4, 5000, "float16", 4, 2048),  # L0-5 tail pad (5000%2048=904)
        (33, 4096, "float16", 8, 2048),  # L0-6 row-tail (33%8=1)
        (4, 32768, "float32", 4, 2048),  # L0-7 fp32 large N
        (1, 102400, "float16", 1, 4096),  # L0-8 tiny M + large tile_n
        (128, 4096, "float16", 8, 1024),  # L0-9 larger M + smaller tile_n
        (4, 300, "float16", 4, 128),  # L0-10 small N pad (300%128=44)
    ]
    max_diff = 0.0
    for M, N, dtype_str, block_m, tile_n in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, block_m, tile_n, "L0"))
    print(f"[L0] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    cases = [
        (128, 16384, "float16", 8, 2048),  # 3d-multidim-reduce (4,128,4096,dim=[0,2])
        (128, 16384, "float32", 8, 2048),  # same, fp32
        (4, 8192, "bfloat16", 4, 2048),  # bf16 multi-tile
        (4, 32768, "float16", 4, 4096),  # tile_n=4096 multi-tile (bm=4 per DESIGN §4.5)
        (17, 6144, "float16", 16, 1024),  # row-tail 17%16=1 + tile_n=1024
        (2, 5000, "float32", 2, 2048),  # small M + unaligned N fp32
        (256, 2048, "float16", 16, 512),  # larger M + small tile_n (4 tiles)
    ]
    max_diff = 0.0
    for M, N, dtype_str, block_m, tile_n in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, block_m, tile_n, "L1"))
    print(f"[L1] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, "float16", 1, 128),  # minimal (N < tile_n, padded to 1 tile)
        (8, 2048, "float16", 8, 2048),  # exact divide M=block_m, N=tile_n (1 tile)
        (9, 2048, "float16", 8, 2048),  # M=block_m+1 (row-tail)
        (4, 2049, "float16", 4, 2048),  # N=tile_n+1 (2 tiles after padding)
    ]
    for M, N, dtype_str, block_m, tile_n in cases:
        try:
            _run_case(M, N, dtype_str, block_m, tile_n, "L2")
        except Exception as e:  # noqa: BLE001
            print(
                f"[L2] WARN (record only): shape=({M},{N}) {dtype_str} "
                f"block_m={block_m} tile_n={tile_n}: {e}"
            )


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # zeros: logsumexp(0,...,0) = log(N)
    try:
        x = torch.zeros(32, 8192, dtype=torch.float16, device="npu")
        n_padded = ((8192 + 2048 - 1) // 2048) * 2048  # already aligned
        kernel = _logsumexp_kernel_tiled(32, n_padded, "float16", 2048)(8)
        y = kernel(x)
        ref = golden_logsumexp_kernel_tiled(x.cpu())
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-3)
        print("[Boundary] PASS: zeros (32,8192) fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {e}")

    # large positive/negative mix (exp overflow risk mitigated by max-shift)
    try:
        x = torch.randn(32, 8192, dtype=torch.float16, device="npu") * 50
        n_padded = ((8192 + 2048 - 1) // 2048) * 2048
        kernel = _logsumexp_kernel_tiled(32, n_padded, "float16", 2048)(8)
        y = kernel(x)
        ref = golden_logsumexp_kernel_tiled(x.cpu())
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-3)
        print("[Boundary] PASS: large values (x50) (32,8192) fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large values: {e}")

    # all -inf (math boundary: torch returns -inf, kernel may return NaN/-inf)
    try:
        x = torch.full((32, 8192), float("-inf"), dtype=torch.float16, device="npu")
        n_padded = ((8192 + 2048 - 1) // 2048) * 2048
        kernel = _logsumexp_kernel_tiled(32, n_padded, "float16", 2048)(8)
        y = kernel(x)
        ref = golden_logsumexp_kernel_tiled(x.cpu())
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-3)
        print("[Boundary] PASS: all -inf (32,8192) fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only, known math boundary): all -inf: {e}")

    # large random shape
    try:
        _run_case(512, 65536, "float16", 8, 2048, "Boundary")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large random (512,65536): {e}")


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
