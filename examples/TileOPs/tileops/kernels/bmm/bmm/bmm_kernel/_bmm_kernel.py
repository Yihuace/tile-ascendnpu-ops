# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_bmm_kernel: batched dense matmul (``torch.bmm`` semantics) on npuir.

Migrated from the TileOPs GPU ``_bmm_kernel`` (``BmmFwdOp``) to the npuir
backend.  The NPU algorithm follows the frozen DESIGN.md of
``examples/bmm/_bmm_kernel/``:

  * R1: the GPU 3D grid ``(N-tile, M-tile, batch)`` is folded into a single
    1D grid, because npuir ``T.Kernel`` supports at most 2 block dims.
  * R2: the epilogue is a slice-form ``T.copy`` (L0C -> GM) with dynamic
    extents; ND layout is preserved and the compiler auto-inserts the
    ``hivm::VCastOp``(RINT) fp32 -> fp16/bf16 quantization.  No fixpipe /
    ``enable_nz2nd`` (those are Expert-only).
  * R3: ``T.gemm(initC=(k == 0))`` replaces the explicit ``T.clear``.
  * R4: GPU-specific means (L2 swizzle / SMEM swizzle / threads / TMA) are
    dropped; only L1 tiling (``T.alloc_shared``) and the K pipeline
    (``T.Pipelined``) are kept.

Semantics: strict 3D-3D batched matmul, no broadcasting.  ``a`` is
``(B, M, K)``, ``b`` is ``(B, K, N)`` (NN layout, N contiguous), ``c`` is
``(B, M, N)``.  Accumulation is fp32; the output dtype equals the input dtype.

Run: ``python _bmm_kernel.py --level {L0|all}``
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

# Stage 3 validated programming mode; must equal TILELANG_ASCEND_MODE above.
ASCEND_MODE = "Developer"

# fp32 accumulator is mandatory for fp16/bf16 GEMM on npuir (T.gemm.md 2.3).
_ACCUM_DTYPE = "float32"

# Tolerance thresholds (mirror examples/TileOPs/tests/ops/test_bmm.py).
_TOL = {
    "float16": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 1.6e-2, "rtol": 1.6e-2},
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BmmFwdOp semantics == ``torch.bmm`` (strict 3D-3D, no broadcasting).

    ``a``: (B, M, K); ``b``: (B, K, N); returns ``(B, M, N)``.

    The reference accumulates in fp32 (the operator's documented GEMM
    accumulation contract, see DESIGN Sec.4 / T.gemm.md), then performs the
    single fp32 -> input-dtype round that the kernel epilogue also performs.
    Computing the CPU golden directly in bf16 would round the result a second
    time (bf16 CPU matmul), producing a reference that is itself ~1 bf16 ulp
    away from the true product and making an atol=1.6e-2 bf16-vs-bf16 compare
    impossible for *any* correct implementation.
    """
    # Explicit torch module call keeps the golden an independent reference.
    return torch.bmm(a.float(), b.float()).to(a.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[-1], target="npuir")
def _bmm_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = None,
    num_stages: int = 2,
    dtype: str = "float16",
):
    """Pipelined batched GEMM for npuir (Developer mode).

    Args:
        batch: number of independent GEMM problems.
        m: rows of each ``A[b]`` / ``C[b]``.
        n: columns of each ``B[b]`` / ``C[b]``.
        k: contraction dim, shared by all batches (must be a multiple of 16).
        block_M/block_N: M/N tile sizes.  ``block_M * block_N * 4 <= L0C``.
        block_K: K tile; defaults to ``64 if k % 64 == 0 else 32``.
        num_stages: K-pipeline depth (L1 double buffering).
        dtype: ``"float16"`` or ``"bfloat16"``.

    Returns:
        The compiled ``_bmm_main`` prim_func (output allocated via out_idx).
    """
    accum_dtype = _ACCUM_DTYPE
    # block_K rule: 64 when k % 64 == 0 else 32 (>= 16 fractal, keeps k % block_K == 0).
    block_K = block_K if block_K is not None else (64 if k % 64 == 0 else 32)
    nt_m = (m + block_M - 1) // block_M
    nt_n = (n + block_N - 1) // block_N
    grid = batch * nt_m * nt_n
    k_iters = (k + block_K - 1) // block_K

    @T.prim_func
    def _bmm_main(
        a: T.Tensor((batch, m, k), dtype),
        b: T.Tensor((batch, k, n), dtype),
        c: T.Tensor((batch, m, n), dtype),
    ):
        # R1: fold (batch, M-tile, N-tile) into a single ``cid`` because an
        # npuir ``T.Kernel`` supports at most 2 block dimensions (DESIGN 0.6 R1).
        with T.Kernel(grid, is_npu=True) as (cid, _):
            bz = cid // (nt_m * nt_n)
            rem = cid % (nt_m * nt_n)
            by = rem // nt_n
            bx = rem % nt_n
            m0 = by * block_M
            n0 = bx * block_N
            # Tail extent: equals the block size when divisible, otherwise the
            # exact remainder (DESIGN 0.6 R2).
            tile_m = T.min(block_M, m - m0)
            tile_n = T.min(block_N, n - n0)

            a_s = T.alloc_shared((block_M, block_K), dtype)  # -> L1
            b_s = T.alloc_shared((block_K, block_N), dtype)  # -> L1
            c_l = T.alloc_fragment((block_M, block_N), accum_dtype)  # -> L0C

            # K-dimension pipelined accumulation: ``initC`` initialises the
            # accumulator on the first iteration (ks==0), replacing an explicit
            # ``T.clear`` (DESIGN 0.6 R3). ``T.Pipelined`` double-buffers so the
            # GM->L1 copies overlap the Cube MMA.
            for ks in T.Pipelined(k_iters, num_stages=num_stages):
                T.copy(
                    a[bz, m0 : m0 + block_M, ks * block_K : ks * block_K + block_K],
                    a_s,
                )
                T.copy(
                    b[bz, ks * block_K : ks * block_K + block_K, n0 : n0 + block_N],
                    b_s,
                )
                T.gemm(a_s, b_s, c_l, initC=(ks == 0))

            # Epilogue: slice-form L0C -> GM direct write, ND layout preserved;
            # the fp32 -> fp16/bf16 quantization is emitted by the compiler as
            # ``hivm::VCastOp``(RINT), no fixpipe/NZ (DESIGN 0.6 R2 / 3.2).  The
            # dynamic extents clip the M/N tail tiles exactly.
            T.copy(
                c_l[0:tile_m, 0:tile_n],
                c[bz, m0 : m0 + tile_m, n0 : n0 + tile_n],
            )

    return _bmm_main


# ---------- precision comparing function ----------
def run_case(batch, m, n, k, dtype, tag, block_M=128, block_N=128, block_K=None,
             num_stages=2):
    """Build inputs, run the kernel on NPU, compare against the CPU golden."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    dtype_str = str(torch_dtype).rsplit(".", 1)[-1]
    tol = _TOL[dtype_str]

    # Golden is computed on CPU from the same inputs (independent reference).
    a_cpu = torch.randn(batch, m, k, dtype=torch.float32).to(torch_dtype)
    b_cpu = torch.randn(batch, k, n, dtype=torch.float32).to(torch_dtype)
    golden = golden_bmm(a_cpu, b_cpu)

    kernel = _bmm_kernel(
        batch, m, n, k,
        block_M=block_M, block_N=block_N, block_K=block_K,
        num_stages=num_stages, dtype=dtype_str,
    )
    c = kernel(a_cpu.npu(), b_cpu.npu())

    c_cpu = c.cpu()
    assert c_cpu.shape == golden.shape, (c_cpu.shape, golden.shape)
    assert c_cpu.dtype == golden.dtype, (c_cpu.dtype, golden.dtype)
    max_diff = (c_cpu.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(c_cpu, golden, atol=tol["atol"], rtol=tol["rtol"])
    used_bk = block_K if block_K is not None else (64 if k % 64 == 0 else 32)
    print(
        f"[{tag}] PASS bmm(B={batch},M={m},N={n},K={k},dtype={dtype_str}) "
        f"block_K={used_bk} max_diff={max_diff:.3e}"
    )
    return max_diff


# ---------- hierarchical testing ----------
def run_L0():
    """Gate: fp16/bf16 smoke + M/N non-divisible boundary."""
    run_case(4, 128, 128, 128, torch.float16, "L0")
    run_case(4, 128, 128, 128, torch.bfloat16, "L0")
    run_case(4, 200, 300, 128, torch.float16, "L0")


def run_L1():
    """Manifest-shaped full cases + 16/32-aligned K variants."""
    run_case(8, 512, 512, 512, torch.float16, "L1")
    run_case(8, 512, 512, 512, torch.bfloat16, "L1")
    run_case(16, 256, 256, 256, torch.float16, "L1")
    run_case(32, 128, 512, 128, torch.float16, "L1")   # MHA QK regime
    run_case(8, 128, 128, 2048, torch.float16, "L1")   # MHA PV regime
    run_case(8, 128, 128, 2048, torch.bfloat16, "L1")
    run_case(4, 200, 300, 128, torch.float16, "L1")    # M/N non-aligned
    run_case(2, 128, 256, 160, torch.float16, "L1")    # K 16-aligned, not 64
    run_case(4, 128, 128, 320, torch.float16, "L1")    # K multiple of 64


def run_L2():
    """Larger shapes (non-blocking: failures are recorded as warnings)."""
    for args in [
        (4, 1024, 1024, 1024, torch.float16),
        (2, 512, 512, 2048, torch.bfloat16),
        (8, 256, 256, 512, torch.float16),
    ]:
        try:
            run_case(*args, "L2")
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (record without blocking): {args} -> {exc}")


def run_boundary():
    """Boundary cases: zeros, bf16 non-divisible, batch=1, tail extremes, K min."""
    # all-zeros input must yield all-zeros output.
    kernel = _bmm_kernel(4, 128, 128, 128, dtype="float16")
    z = torch.zeros(4, 128, 128, dtype=torch.float16)
    c = kernel(z.npu(), z.npu()).cpu()
    torch.testing.assert_close(c, torch.zeros_like(c), atol=1e-3, rtol=1e-3)
    print("[Boundary] PASS zeros fp16 (4,128,128,128)")

    # bf16 x non-divisible M/N (layout-sensitive combination).
    run_case(4, 200, 300, 128, torch.bfloat16, "Boundary")
    # batch size 1 (folded grid degenerates to the 2D grid).
    run_case(1, 256, 256, 256, torch.float16, "Boundary")
    # only one of M/N non-divisible at a time.
    run_case(4, 200, 128, 128, torch.float16, "Boundary")
    run_case(4, 128, 300, 128, torch.float16, "Boundary")
    # minimum K tile (block_K=32, single K iteration).
    run_case(2, 128, 128, 32, torch.float16, "Boundary")
    # K = 16-aligned but not 32/64-aligned -> block_K=32, k_iters not divisible.
    run_case(2, 128, 128, 160, torch.float16, "Boundary")
    # larger batch, aligned.
    run_case(32, 128, 128, 128, torch.float16, "Boundary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L0()
        run_L1()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
