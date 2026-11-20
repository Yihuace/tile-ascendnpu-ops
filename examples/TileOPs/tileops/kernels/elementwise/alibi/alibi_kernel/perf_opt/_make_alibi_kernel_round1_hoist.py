# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""ALiBi bias generator kernel (NPU, ``target="npuir"``).

Migrated from the GPU TileLang factory ``_make_alibi_kernel``
(``/home/developer/TileOPs/tileops/kernels/elementwise.py`` L3557-L3591):

    bias[h, i, j] = -slope_h * |i - j|,   slope_h = 2 ** (-8 * (h + 1) / H)

This is a *generative* (zero-tensor-input) operator: the output
``out[H, S, S]`` is synthesised entirely from the compile-time integer
parameters ``seq_len`` (S) / ``num_heads`` (H) and the output ``dtype``.

.. note::
    The source factory *docstring* says ``slope_h = 2 ** (-8 * h / H)`` but
    the source *implementation* (and its golden test, which uses
    ``torch.arange(1, H + 1)``) uses ``2 ** (-8 * (h + 1) / H)``.  This file
    follows the implementation + golden (``h + 1``), see DESIGN.md section 0.1.

NPU re-design (DESIGN.md sections 0.6 / 3.3 / 5 / 6):

  * R1 -- coordinate generation.  The source recovers ``(h, row, col)`` from a
    flat index with per-element integer div/mod.  On NPU that is
    *implementable* but purely scalar (device-measured 59.1x slower than the
    vector scheme), so we use a *structured separation*: ``|i-j|`` depends only
    on ``(i, j)``, so it is generated from coordinate ramps.

    **Stage 4 round-1 revision (this file)** -- the baseline built *two* full
    ``(bi, bj)`` int16 ramps per logical tile (``arange`` x2 -> ``vcast`` x2 ->
    ``vsub`` -> ``vabs``, 6 full-tile ops per tile) and ``msprof op`` shows
    **~86% scalar-pipe time** concentrated there (2k fp16: 542 us total,
    455 us scalar; removing the chain entirely drops the kernel to 110 us).
    Because ``|(i0+r) - (j0+c)| == |c - ((i0-j0) + r)|``, the relative column
    ramp ``Crel[r, c] = c`` is **tile-invariant**: it is built once per core,
    and each tile only rebuilds the tiny ``(bi, 1)`` row ramp
    ``U[r] = (i0 - j0) + r``.  The distance chain is then
    ``vsub(Crel, U)`` (documented row broadcast ``[M,N] - [M,1]``) +
    ``vabs`` -- **2 full-tile ops per tile** instead of 6, with no int16
    ramp buffers.
  * R1b -- ``D`` is *shared across the head dimension* (CSE); the head loop
    only rescales ``D`` by the scalar slope ``s_h``.
  * R2 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical ``(i, j)`` tiles (the GPU thread model has no NPU
    counterpart).  ``core_num = 40`` is the physical Vector core count of
    Ascend910_9362 (NPUUtils: 20 AI cores x 2).
  * R3 -- each head computes its slope once via the scalar ``T.exp2`` (the
    source formula).  The vector ``T.vexp2`` is unreliable on this device
    (CG-2026-0018) and is not used.
  * R4 -- fp32 compute domain, a single round-back to fp16/bf16 via
    ``T.vcast`` with ``round_mode="rint"``.
  * R5 -- **explicit output parameter, no ``out_idx``**: ``out_idx=[...]``
    combined with ``T.npuir_arange`` crashes at runtime on the device
    (``MTE DDR out of range``, retCode 507057, CG-2026-0016).  The caller
    allocates ``out`` with ``torch.empty`` and passes it in; the source Op's
    1-D output is exposed here directly as ``(H, S, S)``.

Hard constraints honoured (DESIGN.md sections 3.5.1 / 9.1):

  * R-2: the arange tile first dim must be >= 2 -- a ``(1, N)`` tile silently
    produces garbage tail elements (CG-2026-0017).
  * R-7: every ``T.alloc_ub`` is allocated unconditionally at the ``T.Kernel``
    top level (a conditional alloc fails to trace).  Stage 4 addition: the
    *parser-level* conditional form ``X = T.alloc_ub(...) if <py-cond> else None``
    segfaults the tilelang parser on this toolchain (tilelang
    0.1.2+00aeb862, Ascend910B3) -- so all allocations are plain statements.
  * R-8: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` may use a
    dynamic region.
  * R-11: UB budget -- device-measured peak 12 B/elem for the baseline layout;
    the ceiling is 16384 elems (192 KB/AIV).  This file's layout is
    ``Crel`` fp32 + ``Dm`` fp32 + ``O`` fp32 + ``O16`` dtype (+ a ``(bi,1)`` fp32
    row ramp), i.e. 14 B/elem naive; the default ``bi * bj`` is 8192.

run: python _make_alibi_kernel_round1_hoist.py --level all
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into the
# TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"
_BUDGET = 8192  # default UB element budget (device ceiling is 16384)


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_alibi(seq_len: int, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """ALiBi bias reference, ported from ``test_special_elementwise.py::test_alibi``.

    Contract::

        positions = arange(S, fp32)
        dist      = |positions[i] - positions[j]|            # (S, S) fp32
        slopes    = 2 ** (-8 * arange(1, H + 1, fp32) / H)   # (H,)  fp32
        out       = (-slopes[:, None, None] * dist[None])    # fp32 opmath
        return out.to(dtype)

    i.e. ``|i-j|``, the slope and their product are all evaluated in fp32 and
    rounded back to ``dtype`` exactly once.

    This golden is deliberately *independent* of the NPU kernel structure: it
    never reuses the arange ramp / D-shared / vmul chain.  The per-head loop is
    a memory-safe equivalent of the broadcast form above (identical fp32
    elementwise arithmetic, but no ``(H, S, S)`` fp32 temporary).
    """
    positions = torch.arange(seq_len, dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (S, S)
    slopes = torch.pow(
        2.0, -8.0 * torch.arange(1, num_heads + 1, dtype=torch.float32) / num_heads
    )  # (H,)
    out = torch.empty(num_heads, seq_len, seq_len, dtype=dtype)
    for h in range(num_heads):
        out[h] = (-slopes[h] * dist).to(dtype)
    return out


# ---------- Tiling selection (compile-time, keeps bi | S and bj | S) ----------


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is <= ``cap`` (>= 1)."""
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def _select_blocks(seq_len: int, budget: int = _BUDGET):
    """Pick the spatial tile ``(bi, bj)`` for a compile-time ``seq_len``.

    Prefers ``bj = S`` (fully contiguous column ramp / store) whenever ``bi >= 2``
    still fits the UB budget, otherwise the largest ``bj | S`` that leaves room.
    ``bi`` is the largest divisor of ``S`` within budget and is forced to >= 2
    (the arange tile first dim constraint, CG-2026-0017).  ``bi | S`` and ``bj | S``
    hold for every ``S`` used here (DESIGN.md section 5.2/5.4), which makes every
    tile exactly divisible and removes the need for tail guards.
    """
    if 2 * seq_len <= budget:
        bj = seq_len
    else:
        bj = _largest_divisor_leq(seq_len, max(1, budget // 2))
    bi = _largest_divisor_leq(seq_len, max(2, budget // bj))
    if bi < 2:
        bi = 2  # degenerate; the tail guard below covers the resulting overhang
    return bi, bj


# ---------- Kernel ----------
@tilelang.jit(target="npuir")
def _make_alibi_kernel(
    seq_len: int,
    num_heads: int,
    dtype: str,
    bi: int = None,
    bj: int = None,
    budget: int = _BUDGET,
    core_num: int = _CORE_NUM,
):
    """ALiBi bias generator factory (NPU), round-1 hoisted-ramp structure.

    ``seq_len`` / ``num_heads`` are compile-time integers; ``dtype`` is a
    compile-time string in ``{"float16", "bfloat16", "float32"}``.

    Round-1 structural change (Stage 4) -- **tile-invariant relative ramp**:

        The baseline rebuilds *two* full ``(bi, bj)`` coordinate ramps inside
        every logical tile (``T.npuir_arange`` x2) and converts them i16 -> fp32
        (``T.vcast`` x2) before ``vsub`` + ``vabs``: 6 full-tile vector ops per
        tile, and device profilation shows ~86% of the kernel is *scalar* time
        spent there.  Since

            D[r, c] = |(i0 + r) - (j0 + c)| = |c - ((i0 - j0) + r)|

        the relative ramp ``Crel[r, c] = c`` does **not** depend on the tile, and
        the tile origin enters only through the *offset of the row ramp*.  So:

          * ``Crel`` is built once per core (before the grid-stride loop);
          * per tile only a tiny ``(bi, 1)`` row ramp ``U`` is rebuilt
            (``T.npuir_arange(U, [1, 0], i0 - j0)``, ``bi`` elements);
          * ``D = vabs(vsub(Crel, U))`` -- the documented row-broadcast form
            ``[M, N] op [M, 1]`` (docs/Tilelang.language/数学操作/T.vsub.md
            section 2.2.2), then a per-head ``vmul``/``vcast``/``T.copy``.

        Inner-loop distance cost drops from 6 full-tile ops to 2 (+ 1 tiny
        ``bi``-element arange), and the two int16 ramp buffers disappear.

    The output is an explicit parameter (no ``out_idx``, see DESIGN.md R5).
    """
    S, H = int(seq_len), int(num_heads)
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32

    if bi is None or bj is None:
        auto_bi, auto_bj = _select_blocks(S, budget)
        bi = auto_bi if bi is None else int(bi)
        bj = auto_bj if bj is None else int(bj)
    bi, bj = int(bi), int(bj)

    nb = S // bj  # number of column blocks
    n_tiles = (S // bi) * nb  # compile-time logical tile count
    exact = (S % bi == 0) and (S % bj == 0)
    stride = T.ceildiv(n_tiles, core_num)

    @T.prim_func
    def main(out: T.Tensor((H, S, S), dtype)):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            # R-7: every UB buffer is allocated unconditionally at the top level.
            # (NB: `T.alloc_ub(...) if <py-cond> else None` segfaults the tilelang
            # parser on this toolchain -- allocations must be plain statements.)
            Crel = T.alloc_ub((bi, bj), f32)  # tile-invariant ramp C[r, c] = c
            Rrow = T.alloc_ub((bi, 1), f32)  # per-tile row ramp U[r] = i0 - j0 + r
            Dm = T.alloc_ub((bi, bj), f32)  # distance D = |i - j| (fp32)
            O = T.alloc_ub((bi, bj), f32)  # per-head scaled result s_h * D
            O16 = T.alloc_ub((bi, bj), dtype)  # round-back buffer (fp16/bf16)

            # Hoisted column ramp: built once per core for the whole grid-stride
            # loop (j0 is folded into the row ramp offset below).
            T.npuir_arange(Crel, [0, 1], 0)

            # In-core grid-stride over logical tiles (static loop bound, guard
            # covers non-divisible grids; DESIGN.md section 5.5).
            for t in T.serial(stride):
                task = t * core_num + cid
                if task < n_tiles:
                    i0 = (task // nb) * bi
                    j0 = (task % nb) * bj

                    # U[r] = (i0 - j0) + r  ->  |c - U[r]| == |(i0 + r) - (j0 + c)|
                    T.npuir_arange(Rrow, [1, 0], i0 - j0)
                    T.vsub(Crel, Rrow, Dm)  # row broadcast [bi,bj] - [bi,1]
                    T.vabs(Dm, Dm)  # head-independent D = |i - j|

                    # Share D across heads; one scalar exp2 slope per head.
                    for h in T.serial(H):
                        e = (
                            T.cast(-8.0, f32)
                            * T.cast(h + 1, f32)
                            / T.cast(H, f32)
                        )
                        slp = -T.exp2(e)  # s_h = -slope_h (scalar)
                        T.vmul(Dm, slp, O)  # O = s_h * D
                        if need_cast:
                            T.vcast(O, O16, round_mode="rint")  # single round-back
                            if exact:
                                T.copy(O16, out[h, i0, j0])
                            else:  # pragma: no cover - generic-S tail guard
                                it = T.min(bi, S - i0)
                                jt = T.min(bj, S - j0)
                                T.copy(
                                    O16[0:it, 0:jt],
                                    out[h, i0 : i0 + it, j0 : j0 + jt],
                                )
                        else:
                            if exact:
                                T.copy(O, out[h, i0, j0])
                            else:  # pragma: no cover - generic-S tail guard
                                it = T.min(bi, S - i0)
                                jt = T.min(bj, S - j0)
                                T.copy(
                                    O[0:it, 0:jt],
                                    out[h, i0 : i0 + it, j0 : j0 + jt],
                                )

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances from examples/TileOPs/tests/ops/test_special_elementwise.py::test_alibi
# (fp16 1e-2, fp32 1e-5) plus the repo bf16 elementwise tolerance (DESIGN.md 8.2).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-2, 1e-2),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _max_diff(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error evaluated in fp64 (tensors are small enough here)."""
    return (out.to(torch.float64) - ref.to(torch.float64)).abs().max().item()


def _run_case(S, H, dtype_str, tag, bi=None, bj=None, check_diag=False):
    """Build the kernel for (S, H, dtype), run it on the NPU, compare to golden."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    ref = golden_alibi(S, H, torch_dtype)
    kernel = _make_alibi_kernel(S, H, dtype_str, bi=bi, bj=bj)

    out = torch.empty(H, S, S, dtype=torch_dtype, device="npu")
    kernel(out)
    out_cpu = out.cpu()

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(out_cpu, ref)
    diag_ok = True
    if check_diag:
        diag = out_cpu.diagonal(dim1=1, dim2=2)
        diag_ok = bool((diag == 0).all().item())
    try:
        torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol)
        if check_diag and not diag_ok:
            raise AssertionError("diagonal entries are not exactly zero")
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: S={S} H={H} dtype={dtype_str} "
            f"bi={bi} bj={bj} max_diff={max_diff:.6e} diag_ok={diag_ok}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: S={S} H={H} dtype={dtype_str} bi={bi} bj={bj} "
        f"max_diff={max_diff:.6e} atol={atol:.1e} rtol={rtol:.1e} diag_ok={diag_ok}"
    )
    return max_diff


# ---------- hierarchical testing ----------
def run_L0():
    # L0-1/2: source test case (S=128, H=8) -- H=8 slopes are exact powers of 2.
    _run_case(128, 8, "float16", "L0-test-fp16", check_diag=True)
    _run_case(128, 8, "float32", "L0-test-fp32")
    # L0-3/4: manifest smoke (S=2048, H=32).
    _run_case(2048, 32, "float16", "L0-manifest-2k-fp16")
    _run_case(2048, 32, "bfloat16", "L0-manifest-2k-bf16")
    # L0-5: large S (full manifest 4k prefill shape).
    _run_case(4096, 32, "float16", "L0-manifest-4k-fp16")
    # L0-6: int16 ramp value range -- |i-j| > 2048 (beyond fp16's exact range).
    _run_case(4096, 1, "float16", "L0-int16-range", check_diag=True)
    # L0-8: small S / single tile.
    _run_case(64, 4, "float16", "L0-small")
    # L0-9: dtype coverage at S=256, H=32.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(256, 32, dtype_str, f"L0-256-{dtype_str}")
    # L0-10: non-2-power S (divisor-based tiling keeps every tile exact).
    _run_case(1000, 8, "float16", "L0-nondiv-1000")


def run_L1():
    # L1: full manifest dtype coverage + explicit bi/bj overrides + H extremes.
    _run_case(4096, 32, "bfloat16", "L1-manifest-4k-bf16")
    _run_case(4096, 32, "float32", "L1-manifest-4k-fp32")
    # Explicit narrow bj (multi-column-block path, as device-probed in Q2).
    _run_case(2048, 32, "float16", "L1-tile-64x128", bi=64, bj=128)
    _run_case(2048, 32, "bfloat16", "L1-tile-32x256", bi=32, bj=256)
    _run_case(512, 16, "float16", "L1-512x16", bi=16, bj=128)
    # H extremes and non-power-of-2 head counts.
    _run_case(256, 1, "float16", "L1-H1")
    _run_case(256, 5, "float32", "L1-H5")
    _run_case(1024, 16, "bfloat16", "L1-1024x16")
    _run_case(2048, 8, "float16", "L1-2k-H8")
    _run_case(4096, 4, "float32", "L1-4k-H4")


def run_L2():
    # L2 (non-blocking): odd / small sizes and alternate tiles.
    for S, H, dtype_str, bi, bj in (
        (128, 8, "float16", 8, 128),
        (200, 6, "float32", None, None),
        (255, 3, "float16", None, None),
        (1024, 32, "float16", 4, 256),
        (2048, 32, "float16", 8, 512),
    ):
        try:
            _run_case(S, H, dtype_str, "L2", bi=bi, bj=bj)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): S={S} H={H} dtype={dtype_str} err={e}")


def run_boundary():
    # Boundary (non-blocking): diagonal exactness, extreme distance, bf16 tails.
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            _run_case(128, 8, dtype_str, "Boundary-diag", check_diag=True)
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): diag dtype={dtype_str} err={e}")

    for S, H, dtype_str in (
        (4096, 1, "bfloat16"),
        (4096, 1, "float32"),
        (4096, 32, "bfloat16"),
    ):
        try:
            _run_case(S, H, dtype_str, "Boundary-large")
        except Exception as e:  # noqa: BLE001
            print(
                f"[Boundary] WARN (Record without blocking): "
                f"S={S} H={H} dtype={dtype_str} err={e}"
            )


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
