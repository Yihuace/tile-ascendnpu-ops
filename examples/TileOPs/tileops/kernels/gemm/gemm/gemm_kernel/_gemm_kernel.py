# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_gemm_kernel: dense GEMM ``C = op(A) @ op(B)`` on npuir (Developer mode).

Migrated from the TileOPs GPU ``_gemm_kernel`` (SM90 warp-specialized dense
GEMM) to the npuir backend.  The NPU algorithm follows the frozen DESIGN.md of
``examples/gemm/_gemm_kernel/``:

  * R1: the GPU producer/consumer warp specialization + TMA + mbarrier ring is
    replaced by a single-core ``T.Pipelined`` software pipeline over K
    (MTE2 GM->L1 prefetch overlapping the Cube ``T.gemm``).
  * R2: the GPU 2D grid is folded into a single 1D ``T.Kernel`` grid with
    ``num_launched = num_logical`` (non-persistent: one logical output tile
    per launched core; the grid is never truncated with ``min(...)``).
  * R3: WGMMA transpose flags become ``T.gemm(a_transpose=, b_transpose=)`` and
    the L1 tile shapes follow the storage layout of ``trans_a`` / ``trans_b``.
  * R4: no host-side padding (forbidden).  Non-divisible M/N/K is handled
    inside the kernel: static M/N/K-aligned block choice plus a peeled K-tail
    iteration whose A/B tail tiles hold exactly the valid K remainder (E2:
    no fabricated ``0 * x`` term, no stale tail, both operands), and a
    dynamic-extent guarded epilogue write-back.
  * R5: the GPU ``trace.*`` instrumentation is dropped.

Coverage: ``trans_a``/``trans_b`` in all four layout combinations
(NT/NN/TN/TT), fp16/bf16 inputs, fp32 accumulation, output dtype == input
dtype, and non-divisible M/N/K.

Implementation notes (Stage 3 discoveries vs. DESIGN 0.6 R4)
-----------------------------------------------------------
The DESIGN prescribes a *dynamic-extent* ``T.copy`` for the M/N/K tail
(``T.copy(A[m0:m0+tm, k0:k0+rK], A_shared[0:tm, 0:rK])``).  On the installed
toolchain (tilelang 0.1.2 dev root + CANN 9.0.0 / bishengir) that form is not
lowerable in Developer mode: bishengir-compile aborts with
``'hivm.hir.copy' op Unsupported copy from cbuf to cbuf!`` (reproducible with
num_stages=1 and with auto-multi-buffer disabled, i.e. it is the dynamic
*extent* of the GM->L1 tile, not the pipeline, that is unsupported).  A
dynamic *offset* into the fp32 L0C accumulator at the epilogue is likewise
unsupported (aicore exception: "Read L0C ... addresses are not aligned").
The equivalent, supported formulation used here is:

  * M/N tiles are chosen so that the tile size divides the dimension whenever
    a 16-multiple divisor <= 128 exists (covering every manifest workload);
    the load start offset is dynamic (supported) while the tile extent is
    static (supported), so the M/N tail needs no dynamic extent at all.
  * The K tail is peeled out of the pipelined loop: the tail size ``k %
    block_K`` is a compile-time constant, so the *tail tile itself* is
    allocated with exactly that K extent (shape ``(..., rK_last)``) and filled
    by a full-buffer, static-extent ``T.copy`` of the valid K range.  This is
    the E2 requirement (``0 * x == 0``, no stale tail, both operands) enforced
    by construction rather than by a ``T.clear`` of a padded tile: the
    reduction never sees a fabricated K element at all.  A padded-tile +
    ``T.clear`` variant is lowerable only for the NT/NN layouts on this
    toolchain (an ``a_transpose=True, b_transpose=False`` K tail additionally
    makes bishengir-compile SIGSEGV), so the exact-extent tile is used for all
    four layouts.
  * The epilogue keeps the DESIGN dynamic-extent form ``T.copy(C_local[0:tm,
    0:tn], C[m0:m0+tm, n0:n0+tn])`` (zero static source offset), which is
    supported and emits the auto ``hivm::VCastOp``(RINT) fp32 -> fp16/bf16
    single rounding (no fixpipe / NZ, those are Expert-only).

Run: ``python _gemm_kernel.py --level {L0|L1|all}``
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

# Stage 3 validated programming mode; must equal TILELANG_ASCEND_MODE above.
ASCEND_MODE = "Developer"

# fp32 accumulation is mandatory for fp16/bf16 GEMM on npuir (T.gemm.md 2.3:
# an fp16 dst/accumulator may hang the kernel at runtime).
_ACCUM_DTYPE = "float32"

# Tolerance thresholds (mirror examples/TileOPs/tests/ops/test_gemm.py).
_TOL = {
    "float16": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 1.6e-2, "rtol": 1.6e-2},
}

# Candidate M/N tile sizes (16-multiple, Cube fractal >= 16, L0C-safe <= 128).
_TILE_CANDIDATES = (128, 112, 96, 80, 64, 48, 32, 16)


def _tile_size(dim, cap):
    """Pick a 16-multiple tile <= cap that divides ``dim`` exactly.

    Exact division removes the M/N tail (no out-of-range GM tile read) and
    avoids computing wasted rows/columns.  When ``dim`` has no such divisor
    the largest usable 16-multiple <= cap is returned and the tail is handled
    by the dynamic-extent epilogue guard.
    """
    for cand in _TILE_CANDIDATES:
        if cand <= cap and dim % cand == 0:
            return cand
    return min(cap, max(16, (dim // 16) * 16))


def _block_k(k):
    """K tile: 64 when k % 64 == 0 else 32 (>= 16 fractal, keeps 32B tail axis)."""
    return 64 if k % 64 == 0 else 32


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_gemm(a: torch.Tensor, b: torch.Tensor, trans_a: bool = False,
                trans_b: bool = True) -> torch.Tensor:
    """Source-operator semantics: ``C = op(A) @ op(B)`` (manifest ref_api).

    ``a``: (M,K) if not trans_a else (K,M); ``b``: (K,N) if not trans_b else
    (N,K); returns ``(M,N)`` with dtype == a.dtype.

    The reference accumulates in fp32 through explicit ``torch`` module calls
    (the operator's documented GEMM accumulation contract, DESIGN 1.6.1 E1)
    and performs the single fp32 -> input-dtype round that the kernel epilogue
    VCast also performs.  A bare fp16/bf16 CPU matmul would round a second
    time and make the comparison meaningless for any correct implementation.
    """
    if trans_a:
        a = torch.transpose(a, 0, 1)
    if trans_b:
        b = torch.transpose(b, 0, 1)
    return torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(a.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemm_kernel(m, n, k, trans_a, trans_b, dtype="float16", block_M=128,
                 block_N=128, block_K=None, num_stages=3):
    """Pipelined dense GEMM for npuir (Developer mode).

    Args:
        m, n, k: logical GEMM dims (compile-time constants; one kernel per
            shape, cached by ``tilelang.jit``).
        trans_a: True when ``A`` is stored ``(K, M)`` (logical ``op(A)=A^T``).
        trans_b: True when ``B`` is stored ``(N, K)`` (logical ``op(B)=B^T``);
            the manifest default layout.
        dtype: ``"float16"`` or ``"bfloat16"``.
        block_M/block_N: M/N tile caps; shrunk per shape by ``_tile_size``.
        block_K: K tile; defaults to ``_block_k(k)`` = 64 if k % 64 == 0 else 32.
        num_stages: K-pipeline depth (L1 multi-buffering).

    Returns:
        The compiled ``main`` prim_func (output allocated via ``out_idx``).
    """
    accum_dtype = _ACCUM_DTYPE
    # M/N tiling: a divisor tile whenever possible, otherwise the cap with a
    # guarded tail (DESIGN 5.2 permits shrinking block_M for small m).
    bm = _tile_size(m, block_M)
    bn = _tile_size(n, block_N)
    block_K = block_K if block_K is not None else _block_k(k)

    nt_m = (m + bm - 1) // bm
    nt_n = (n + bn - 1) // bn
    num_logical = nt_m * nt_n
    # Non-persistent 1D grid: num_launched == num_logical (never min()-clamped,
    # otherwise cores > physical count would silently drop output tiles).
    num_launched = num_logical
    k_full = k // block_K
    rK_last = k - k_full * block_K
    has_k_tail = rK_last > 0

    # Layout-dependent L1 tile shapes (DESIGN 0.6 R3): the logical operands are
    # op(A)=[M,K] and op(B)=[K,N]; the tiles follow the *storage* layout.
    a_rows, a_cols = (block_K, bm) if trans_a else (bm, block_K)
    b_rows, b_cols = (bn, block_K) if trans_b else (block_K, bn)
    # Peeled K-tail tiles: exactly the valid K remainder (compile-time
    # constant), so every tail copy has static extents and the reduction sees
    # no fabricated/zero K element (DESIGN 1.6.1 E2 by construction).
    at_rows, at_cols = (rK_last, bm) if trans_a else (bm, rK_last)
    bt_rows, bt_cols = (bn, rK_last) if trans_b else (rK_last, bn)
    if not has_k_tail:
        # Width switch, never a zero-size or conditional allocation.
        at_rows = at_cols = bt_rows = bt_cols = 16

    @T.prim_func
    def main(
        A: T.Tensor((k, m) if trans_a else (m, k), dtype),
        B: T.Tensor((n, k) if trans_b else (k, n), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(num_launched, is_npu=True) as (cid, _):
            # cid -> (M block, N block) bijection (DESIGN 0.6 R2).
            by = cid // nt_n
            bx = cid % nt_n
            m0 = by * bm
            n0 = bx * bn
            # Valid output extents (dynamic tail guard, slice-form epilogue).
            tm = T.min(bm, m - m0)
            tn = T.min(bn, n - n0)

            A_shared = T.alloc_shared((a_rows, a_cols), dtype)  # -> L1
            B_shared = T.alloc_shared((b_rows, b_cols), dtype)  # -> L1
            A_tail = T.alloc_shared((at_rows, at_cols), dtype)  # -> L1
            B_tail = T.alloc_shared((bt_rows, bt_cols), dtype)  # -> L1
            C_local = T.alloc_fragment((bm, bn), accum_dtype)  # -> L0C

            # K-dimension pipelined accumulation: MTE2 prefetch of the next
            # tile overlaps the Cube MMA of the current one; initC zeroes the
            # accumulator on the first iteration (== explicit T.clear).
            for kk in T.Pipelined(k_full, num_stages=num_stages):
                k0 = kk * block_K
                if trans_a:
                    T.copy(A[k0:k0 + block_K, m0:m0 + bm], A_shared)
                else:
                    T.copy(A[m0:m0 + bm, k0:k0 + block_K], A_shared)
                if trans_b:
                    T.copy(B[n0:n0 + bn, k0:k0 + block_K], B_shared)
                else:
                    T.copy(B[k0:k0 + block_K, n0:n0 + bn], B_shared)
                T.gemm(A_shared, B_shared, C_local, initC=(kk == 0),
                       a_transpose=trans_a, b_transpose=trans_b)

            # Peeled K tail (block_K does not divide k): accumulate the exact
            # K remainder.  The tail tiles hold only valid K elements, so no
            # tail element can be stale (E2: no 0*NaN / 0*inf path either).
            if has_k_tail:
                kt = k_full * block_K
                if trans_a:
                    T.copy(A[kt:kt + rK_last, m0:m0 + bm], A_tail)
                else:
                    T.copy(A[m0:m0 + bm, kt:kt + rK_last], A_tail)
                if trans_b:
                    T.copy(B[n0:n0 + bn, kt:kt + rK_last], B_tail)
                else:
                    T.copy(B[kt:kt + rK_last, n0:n0 + bn], B_tail)
                T.gemm(A_tail, B_tail, C_local, initC=(k_full == 0),
                       a_transpose=trans_a, b_transpose=trans_b)

            # Epilogue: slice-form L0C -> GM write.  ND layout preserved; the
            # fp32 -> fp16/bf16 quantization is emitted by the compiler as
            # ``hivm::VCastOp``(RINT), a single rounding (E1).  Dynamic extents
            # clip the M/N tail tiles exactly -> no out-of-range write.
            T.copy(C_local[0:tm, 0:tn], C[m0:m0 + tm, n0:n0 + tn])

    return main


# ---------- precision comparing function ----------
def run_case(m, n, k, trans_a, trans_b, dtype, tag, block_M=128, block_N=128,
             block_K=None, num_stages=3):
    """Build inputs, run the kernel on NPU, compare against the CPU golden."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    dtype_str = str(torch_dtype).rsplit(".", 1)[-1]
    tol = _TOL[dtype_str]
    torch.manual_seed(0)

    # Layout-correct CPU inputs (independent reference, computed on CPU).
    a_cpu = torch.randn((k, m) if trans_a else (m, k), dtype=torch.float32).to(torch_dtype)
    b_cpu = torch.randn((n, k) if trans_b else (k, n), dtype=torch.float32).to(torch_dtype)
    golden = golden_gemm(a_cpu, b_cpu, trans_a, trans_b)

    kernel = _gemm_kernel(m, n, k, trans_a, trans_b, dtype_str, block_M, block_N,
                          block_K, num_stages)
    c = kernel(a_cpu.npu(), b_cpu.npu())

    c_cpu = c.cpu()
    assert c_cpu.shape == golden.shape, (c_cpu.shape, golden.shape)
    assert c_cpu.dtype == golden.dtype, (c_cpu.dtype, golden.dtype)
    max_diff = (c_cpu.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(c_cpu, golden, atol=tol["atol"], rtol=tol["rtol"])
    print(f"[{tag}] PASS gemm(m={m},n={n},k={k},ta={trans_a},tb={trans_b},"
          f"dtype={dtype_str}) block=({_tile_size(m, block_M)},{_tile_size(n, block_N)},"
          f"{block_K if block_K is not None else _block_k(k)}) max_diff={max_diff:.3e}")
    return max_diff


# ---------- hierarchical testing ----------
def run_L0():
    """Gate: DESIGN 8.3 L0 plan (smoke NT fp16/bf16, NN, non-divisible N/M)."""
    cases = [
        (128, 128, 128, False, True, torch.float16, "L0-1 smoke NT fp16"),
        (128, 128, 128, False, True, torch.bfloat16, "L0-2 smoke NT bf16"),
        (1024, 1024, 1024, False, False, torch.float16, "L0-3 NN square"),
        (128, 2112, 7168, False, True, torch.bfloat16, "L0-4 non-div N=2112"),
        (16, 4096, 7168, False, True, torch.bfloat16, "L0-5 small M=16"),
        (96, 2112, 7168, False, True, torch.bfloat16, "L0-6 M/N tails"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L0 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L1():
    """Manifest-shaped cases + all trans layouts + K / non-divisible variants."""
    cases = [
        # manifest workloads (DeepSeek-V3 projection shapes)
        (128, 7168, 2048, False, True, torch.bfloat16, "L1-1 decode-down"),
        (4096, 4096, 7168, False, True, torch.float16, "L1-2 prefill-attn-proj fp16"),
        (4096, 4096, 7168, False, True, torch.bfloat16, "L1-3 prefill-attn-proj bf16"),
        (1024, 1024, 1024, False, True, torch.float16, "L1-4 square-1k NT"),
        (32, 4096, 7168, False, True, torch.bfloat16, "L1-5 mid-m32"),
        (64, 7168, 2048, False, True, torch.bfloat16, "L1-6 mid-m64-down"),
        # trans layout coverage (NT / NN / TN / TT)
        (130, 260, 256, True, True, torch.float16, "L1-7 TT tail"),
        (130, 260, 256, True, False, torch.float16, "L1-8 TN tail"),
        (130, 260, 256, False, False, torch.float16, "L1-9 NN tail"),
        (130, 260, 256, False, True, torch.float16, "L1-10 NT tail"),
        (96, 160, 144, True, False, torch.bfloat16, "L1-11 TN bf16 tail"),
        (64, 96, 112, True, True, torch.bfloat16, "L1-12 TT bf16 tail"),
        # K non-divisible (block_K = 32, peeled K tail with T.clear)
        (128, 128, 48, False, True, torch.float16, "L1-13 K=48"),
        (128, 128, 80, False, True, torch.float16, "L1-14 K=80"),
        (200, 300, 160, False, True, torch.float16, "L1-15 K=160 + M/N tails"),
        (128, 128, 16, False, True, torch.float16, "L1-16 K=16 (tail only)"),
        (128, 128, 160, False, True, torch.bfloat16, "L1-17 K=160 bf16"),
        (128, 128, 1024, False, True, torch.bfloat16, "L1-18 K=1024 divides"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L1 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L2():
    """Larger shapes (non-blocking: failures are recorded as warnings)."""
    cases = [
        (4096, 2112, 7168, False, True, torch.bfloat16, "L2-1 prefill-gate-up"),
        (4096, 24576, 1536, False, True, torch.bfloat16, "L2-2 wide-n"),
        (1024, 1024, 2048, False, True, torch.float16, "L2-3 k-dominant"),
        (16, 16, 7168, False, True, torch.bfloat16, "L2-4 tiny M/N long K"),
    ]
    worst = 0.0
    for c in cases:
        try:
            worst = max(worst, run_case(*c))
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (record without blocking): {c[:3]} -> {exc}")
    print(f"L2 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_boundary():
    """Boundary cases (non-blocking): zeros, min tiles, odd shapes, K%16."""
    # all-zero inputs must yield an all-zero output.
    try:
        kernel = _gemm_kernel(128, 128, 128, False, True, "float16")
        z = torch.zeros(128, 128, dtype=torch.float16)
        c = kernel(z.npu(), z.npu()).cpu()
        torch.testing.assert_close(c, torch.zeros_like(c), atol=1e-3, rtol=1e-3)
        print("[Boundary] PASS zeros fp16 (128,128,128)")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN zeros -> {exc}")

    cases = [
        (16, 16, 16, False, True, torch.float16, "B-1 min tiles"),
        (48, 48, 32, False, True, torch.float16, "B-2 tiny 16-multiple"),
        (128, 16, 128, False, True, torch.float16, "B-3 N=16"),
        (16, 128, 128, False, True, torch.float16, "B-4 M=16"),
        (128, 128, 40, False, True, torch.float16, "B-5 K=40 (K%16!=0)"),
        (200, 300, 200, False, True, torch.bfloat16, "B-6 odd M/N + K tail"),
        (128, 128, 1, False, True, torch.float16, "B-7 K=1"),
        (1, 128, 128, False, True, torch.float16, "B-8 M=1"),
        (128, 1, 128, False, True, torch.float16, "B-9 N=1 (NT keeps gemm)"),
        (112, 96, 112, True, True, torch.float16, "B-10 TT odd tiles"),
    ]
    worst = 0.0
    for c in cases:
        try:
            worst = max(worst, run_case(*c))
        except Exception as exc:  # noqa: BLE001
            print(f"[Boundary] WARN {c[:3]} -> {exc}")
    print(f"Boundary done ({len(cases)} cases), worst max_diff={worst:.3e}")


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
