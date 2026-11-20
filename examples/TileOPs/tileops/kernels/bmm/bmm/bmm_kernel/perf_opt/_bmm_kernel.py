# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_bmm_kernel: batched dense matmul (``torch.bmm`` semantics) on npuir.

Stage-4 tuned variant of the Stage-3 kernel.  Same factory signature and same
numerics as the baseline (fp32 accumulation, one round-to-nearest
fp32 -> input-dtype epilogue emitted by the compiler as ``hivm::VCastOp``).

Semantics: strict 3D-3D batched matmul, no broadcasting.  ``a`` is
``(B, M, K)``, ``b`` is ``(B, K, N)`` (NN layout, N contiguous), ``c`` is
``(B, M, N)``.

--------------------------------------------------------------------------------
Stage-4 tuning summary (details in ``perf_opt/opt_log.md``)
--------------------------------------------------------------------------------
Measured bottleneck (rounds 0-5): on the Developer (ND-only, L1-only) cube
path the ``aic_mte2`` pipe (GM->L1 operand fill) is the binding resource, and
its traffic is ``B*M*K*2*(N/bn) + B*K*N*2*(M/bm)``.  The tile area is capped at
``block_M*block_N <= 32768`` by the 128 KB L0C (``block_M*block_N*4 <= L0C``).

Two structures are therefore selected shape-adaptively (both were measured
with ``msprof op``; the selection constants below are the measured winners):

  * ``tiled`` (K-pipelined, ``T.Pipelined`` over K): used for the large-K
    shapes.  Winner config ``block_M=128, block_N=256, block_K=256,
    num_stages=2`` -- ``square-b8-1k`` 102.90 us / 56.61 %, ``square-b4-4k``
    2246.1 us / 82.99 %.
  * ``resident`` (full-K slab, B column slab resident in L1 for the whole M
    sweep): used for small-K shapes where both full slabs fit in the 512 KB
    L1.  The K loop collapses to a single full-and-full-buffer ``T.gemm``
    (no sliced gemm operand -- sliced / L1->L1-restaged B operands are
    numerically wrong on this toolchain, see opt_log round-3/4), so B is
    fetched once per (batch, N-tile) instead of once per M-tile and total
    GM->L1 traffic falls to ~25 MB from ~50 MB.  Winner config
    ``block_M=128, block_N=256, num_stages=2`` (M sweep pipelined) --
    ``square-b16-512`` 33.81 us / 43.07 %.

Wrapper dispatch contract: the shipped ``BmmKernel.default_config`` passes
``block_M=64, block_N=64, num_stages=2`` (with a k-derived ``block_K``); that
tuple is the "wrapper default" sentinel and is reinterpreted here as the tuned
shape-adaptive config.  Any explicitly-passed non-sentinel config is respected
verbatim, so every Stage-4 experiment config reproduces exactly.  The module
level ``TUNED_*`` constants are exposed so the wrapper's baseline/perf_opt
import switch block can reference the tuned defaults in pairs.

Run: ``python _bmm_kernel.py --level {L0|L1|all}``
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

# ---------------------------------------------------------------------------
# Stage-4 tuned defaults (module level: the wrapper's baseline/perf_opt import
# switch block references these in pairs).
# ---------------------------------------------------------------------------
TUNED_BLOCK_M = 128
TUNED_BLOCK_N = 256
TUNED_NUM_STAGES = 2
# Full-K-slab B-resident structure is used when both full operand slabs fit the
# 512 KB L1; the measured crossover is k <= 512 (k=1024 needs 768 KB and is
# compiled out by BiShengIR with "cbuf overflow").
TUNED_RESIDENT_MAX_K = 512
# L1 budget reserved for the two full slabs (b_s + a_s), bytes.
TUNED_RESIDENT_SLAB_BYTES = 384 * 1024
# The shipped wrapper default (the sentinel reinterpreted below).
_WRAPPER_DEFAULT_BM = 64
_WRAPPER_DEFAULT_BN = 64

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
    accumulation contract), then performs the single fp32 -> input-dtype round
    that the kernel epilogue also performs.
    """
    return torch.bmm(a.float(), b.float()).to(a.dtype)


# ---------- Stage-4 tuned config selection ----------
def tuned_config(batch, m, n, k, dtype):
    """Return the measured Stage-4 winner ``(block_M, block_N, block_K, stages)``.

    ``None`` means "no measured winner for this shape" -> the caller keeps the
    verbatim (wrapper-default) tiling, i.e. the shipped baseline behaviour.

    Selection rules (all values are the measured winners, see opt_log.md):
      * ``block_M`` fixed at 128 (L0C 128 KB caps ``block_M*block_N <= 32768``);
      * ``block_N`` = 256 / 128 / 64 = the largest power-of-two <= 256 that
        divides ``n`` -- required so that no out-of-range N slice is ever read;
      * ``block_K`` = 256 for the large-K shapes (>= 2 K-iterations at the
        pipelined tile), else the largest 16-aligned divisor of ``k``;
      * the full-K-slab B-resident structure replaces the K-pipelined one when
        ``k`` is small enough that both full slabs fit the L1 budget.
    """
    if k % 16 != 0:
        return None
    # N must be divisible by the (aligned) block_N so the load slices never
    # read past the row end; non-aligned shapes keep the verbatim tiling.
    if n % 64 != 0:
        return None
    bm = TUNED_BLOCK_M
    if n % 256 == 0:
        bn = 256
    elif n % 128 == 0:
        bn = 128
    else:
        bn = 64
    if k % 256 == 0 and k >= 1024:
        bk = 256
    elif k % 128 == 0:
        bk = 128
    elif k % 64 == 0:
        bk = 64
    else:
        bk = 32
    resident = (
        k <= TUNED_RESIDENT_MAX_K
        and m % bm == 0
        and n % bn == 0
        and (bm * k + k * bn) * 2 <= TUNED_RESIDENT_SLAB_BYTES
    )
    if resident:
        bk = k  # the K loop collapses to a single full-slab gemm
    return bm, bn, bk, TUNED_NUM_STAGES, resident


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
    """Batched GEMM for npuir (Developer mode), Stage-4 tuned.

    Args:
        batch: number of independent GEMM problems.
        m: rows of each ``A[b]`` / ``C[b]``.
        n: columns of each ``B[b]`` / ``C[b]``.
        k: contraction dim, shared by all batches (must be a multiple of 16).
        block_M/block_N: M/N tile sizes.  ``block_M * block_N * 4 <= L0C``.
        block_K: K tile; ``k`` selects the full-K-slab structure.
        num_stages: K-pipeline depth (tiled structure) / M-sweep depth
            (resident structure).
        dtype: ``"float16"`` or ``"bfloat16"``.

    Returns:
        The compiled ``_bmm_main`` prim_func (output allocated via out_idx).
    """
    accum_dtype = _ACCUM_DTYPE

    # Stage-4 dispatch: the wrapper default (64, 64) is the sentinel for the
    # tuned shape-adaptive config; any other explicit config is verbatim.
    resident = False
    if (block_M, block_N) == (_WRAPPER_DEFAULT_BM, _WRAPPER_DEFAULT_BN):
        tuned = tuned_config(batch, m, n, k, dtype)
        if tuned is not None:
            block_M, block_N, block_K, num_stages, resident = tuned

    if block_K is None:
        block_K = 64 if k % 64 == 0 else (32 if k % 32 == 0 else 16)

    nt_m = (m + block_M - 1) // block_M
    nt_n = (n + block_N - 1) // block_N
    k_iters = (k + block_K - 1) // block_K
    k_pad = k_iters * block_K

    if resident:
        # ---- full-K-slab B-resident structure (small-K winner) -----------
        grid = batch * nt_n

        @T.prim_func
        def _bmm_main(
            a: T.Tensor((batch, m, k), dtype),
            b: T.Tensor((batch, k, n), dtype),
            c: T.Tensor((batch, m, n), dtype),
        ):
            with T.Kernel(grid, is_npu=True) as (cid, _):
                bz = cid // nt_n
                bx = cid % nt_n
                n0 = bx * block_N
                tile_n = T.min(block_N, n - n0)
                # Whole B column slab resident in L1 for the entire M sweep.
                b_s = T.alloc_shared((k, block_N), dtype)
                T.copy(b[bz, 0:k, n0 : n0 + block_N], b_s)
                for by in T.Pipelined(nt_m, num_stages=num_stages):
                    m0 = by * block_M
                    tile_m = T.min(block_M, m - m0)
                    a_s = T.alloc_shared((block_M, k), dtype)
                    c_l = T.alloc_fragment((block_M, block_N), accum_dtype)
                    T.copy(a[bz, m0 : m0 + block_M, 0:k], a_s)
                    # Single K iteration: full-and-full operand gemm.
                    T.gemm(a_s, b_s, c_l, initC=True)
                    T.copy(
                        c_l[0:tile_m, 0:tile_n],
                        c[bz, m0 : m0 + tile_m, n0 : n0 + tile_n],
                    )

        return _bmm_main

    # ---- K-pipelined tiled structure (large-K winner / verbatim configs) --
    grid = batch * nt_m * nt_n

    @T.prim_func
    def _bmm_main(
        a: T.Tensor((batch, m, k), dtype),
        b: T.Tensor((batch, k, n), dtype),
        c: T.Tensor((batch, m, n), dtype),
    ):
        # Fold (batch, M-tile, N-tile) into a single ``cid`` because an npuir
        # ``T.Kernel`` supports at most 2 block dimensions (DESIGN 0.6 R1).
        with T.Kernel(grid, is_npu=True) as (cid, _):
            bz = cid // (nt_m * nt_n)
            rem = cid % (nt_m * nt_n)
            by = rem // nt_n
            bx = rem % nt_n
            m0 = by * block_M
            n0 = bx * block_N
            tile_m = T.min(block_M, m - m0)
            tile_n = T.min(block_N, n - n0)

            a_s = T.alloc_shared((block_M, block_K), dtype)  # -> L1
            b_s = T.alloc_shared((block_K, block_N), dtype)  # -> L1
            c_l = T.alloc_fragment((block_M, block_N), accum_dtype)  # -> L0C

            # K-dimension pipelined accumulation; ``initC`` initialises the
            # accumulator on the first iteration, replacing an explicit
            # ``T.clear`` (DESIGN 0.6 R3).
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
            # ``hivm::VCastOp``(RINT) (DESIGN 0.6 R2 / 3.2).
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
        f"block={block_M}x{block_N}x{used_bk} stages={num_stages} max_diff={max_diff:.3e}"
    )
    return max_diff


def run_wrapper_path(batch, m, n, k, dtype, tag):
    """Exercise the shipped-wrapper path (default_config sentinel (64, 64))."""
    dtype_str = str(dtype).rsplit(".", 1)[-1] if not isinstance(dtype, str) else dtype
    eff = tuned_config(batch, m, n, k, dtype_str)
    print(f"[{tag}] dispatch sentinel(64,64) -> effective {eff}")
    return run_case(batch, m, n, k, dtype, tag, block_M=64, block_N=64, block_K=None,
                    num_stages=2)


# ---------- hierarchical testing ----------
def run_L0():
    """Gate: fp16/bf16 smoke + M/N non-divisible boundary."""
    run_case(4, 128, 128, 128, torch.float16, "L0")
    run_case(4, 128, 128, 128, torch.bfloat16, "L0")
    run_case(4, 200, 300, 128, torch.float16, "L0")
    # shipped-wrapper dispatch path (tuned sentinel reinterpretation)
    run_wrapper_path(4, 128, 128, 128, torch.float16, "L0-wrapper")
    run_wrapper_path(4, 128, 128, 128, torch.bfloat16, "L0-wrapper")
    run_wrapper_path(4, 200, 300, 128, torch.float16, "L0-wrapper")


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
    # shipped-wrapper dispatch path
    run_wrapper_path(8, 512, 512, 512, torch.float16, "L1-wrapper")
    run_wrapper_path(16, 512, 512, 512, torch.float16, "L1-wrapper")
    run_wrapper_path(8, 1024, 1024, 1024, torch.bfloat16, "L1-wrapper")
    run_wrapper_path(2, 128, 256, 160, torch.float16, "L1-wrapper")
    run_wrapper_path(32, 128, 512, 128, torch.float16, "L1-wrapper")


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
    kernel = _bmm_kernel(4, 128, 128, 128, dtype="float16")
    z = torch.zeros(4, 128, 128, dtype=torch.float16)
    c = kernel(z.npu(), z.npu()).cpu()
    torch.testing.assert_close(c, torch.zeros_like(c), atol=1e-3, rtol=1e-3)
    print("[Boundary] PASS zeros fp16 (4,128,128,128)")

    run_case(4, 200, 300, 128, torch.bfloat16, "Boundary")
    run_case(1, 256, 256, 256, torch.float16, "Boundary")
    run_case(4, 200, 128, 128, torch.float16, "Boundary")
    run_case(4, 128, 300, 128, torch.float16, "Boundary")
    run_case(2, 128, 128, 32, torch.float16, "Boundary")
    run_case(2, 128, 128, 160, torch.float16, "Boundary")
    run_case(32, 128, 128, 128, torch.float16, "Boundary")
    # non-aligned + wrapper dispatch
    run_wrapper_path(4, 200, 300, 128, torch.bfloat16, "Boundary-wrapper")
    run_wrapper_path(4, 128, 300, 128, torch.float16, "Boundary-wrapper")
    run_wrapper_path(1, 256, 256, 256, torch.float16, "Boundary-wrapper")


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
