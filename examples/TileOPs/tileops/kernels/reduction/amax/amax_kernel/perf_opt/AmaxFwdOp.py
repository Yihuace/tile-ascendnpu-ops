# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""AmaxFwdOp -- canonical Stage 4 perf_opt module (resident reduce factory).

Gate-4 contract artifact: ``perf_opt/AmaxFwdOp.py`` is the single deliverable
file that carries the real ``@T.prim_func`` definitions.  It holds the resident
factory ``_simple_reduce_kernel`` verbatim (content moved here from
``perf_opt/_simple_reduce_kernel.py``, which is kept as an import alias) and
re-exports the streaming factory ``_simple_reduce_kernel_tiled`` from its own
module (that module keeps its original name per the op's two-module layout).

Original module docstring follows.

_simple_reduce_kernel (AmaxFwdOp resident path) -- NPU Developer mode.

Stage 4 tuned drop-in for the Stage 3 baseline of the same name (identical
factory signature ``(M, N, op_kind, dtype) -> _func(block_m) -> main``).

Semantics (DESIGN.md section 0.1, frozen):
    y[i] = max_{0 <= j < N} x[i, j]          # torch.amax(x, dim=-1)
    x: (M, N) fp16/bf16/fp32  ->  out: (M,) same_as(x)

NPU redesign (DESIGN.md section 0.6, frozen) -- unchanged from Stage 3:
    R1 raw-N boundary; R2 persistent one-dimensional core split; R3 dtype
    domain (fp16/fp32 original domain, bf16 promoted to fp32); R4 UB budget.

Stage 4 scheduling changes (measured against the Stage 3 baseline):
    S1 **Full-core persistent grid** -- the grid size is the number of vector
       cores (``min(logical, 40)``) with the contiguous, statically balanced
       per-core range (``cid*nl//nk``), so no vector core is left idle.  (Round
       2 shipped the "largest divisor <= 40" grid, which exposed only 32 of 40
       cores on the 512-logical-block hidden-state shape.  A round-7
       interleaved 5x A/B on this op re-adjudicated the two grids and measured
       full-40 faster in 5/5 paired runs: bf16 15.80us vs 16.76us median,
       -5.7%; the round-2 "tie" was a run-state artefact of single-run
       sampling.)
    S2 **Contiguous, statically-balanced per-core row range** -- core ``cid``
       owns logical tasks ``[cid*L, (cid+1)*L)`` (or the balanced
       ``cid*nl//nk .. (cid+1)*nl//nk`` range in the fallback), so its output
       rows are contiguous.
    S3 **Batched output write** -- per-task maxima are staged in a flat UB
       buffer and flushed with exactly ONE UB->GM copy per core.  Stage 4
       profiles showed the per-task ``T.copy(out_ub[...], out[...])`` costs
       ~0.4-0.9 us of MTE3 busy time *regardless of payload*, which dominated
       the small-workload cases.
    S4 **Reduction destination slices** -- the row maxima of consecutive tasks
       land directly in consecutive slices of one ``(L*block_m, 1)`` buffer, so
       the (bm,1)->(bm,) shape contraction runs once per core instead of once
       per task.
    S5 **Single canonical perf_opt entry module** -- the op ships two factory
       functions in one ``perf_opt/`` directory (resident + streaming); this
       module re-exports the streaming factory (see the block below) so
       ``perf_opt/AmaxFwdOp.py`` (this file) is the single addressable entry
       module for the op.  Both files remain self-contained and keep their own
       ``--level all`` harness.
    S6 **Static row-block extent** -- when ``M % block_m == 0`` the per-task
       row slice is a compile-time constant, so the hot loop emits static
       ``T.copy(x[off_m:off_m+block_m, :], x_ub)`` instead of the dynamic
       ``real_m = min(block_m, M - off_m)`` form (the ``min`` / masked-DMA
       form is kept for non-divisible M).  Measured on hidden-state bf16:
       16.40us -> 15.83us median over 22 and 22 interleaved samples (-3.5%),
       by removing per-task scalar extent math from the critical loop.

Known toolchain constraints enforced here (DESIGN.md section 9.1):
    - ``T.reduce_max`` src/dst MUST be the same dtype (mixed dtype silently
      corrupts) -> bf16 uses a dedicated fp32 work buffer.
    - unused ``alloc`` is NOT dead-code-eliminated -> fp16/fp32 and bf16 use
      separate prim_funcs.
    - ``T.Parallel`` may not read GM -> all GM traffic goes through ``T.copy``.
    - ``T.vcast`` f32->f32 is non-identity -> same-dtype uses plain copy.
    - bf16 ``T.reduce_max`` DOES compile but its lowering materialises a 4x
      tile: "ub overflow, requires 2097152 bits while 1572864 bits available"
      at (64,512), and a vector-core runtime timeout (507034) at the size that
      does fit.  So bf16 keeps the explicit fp32 promotion.
    - int16/uint16 ``T.reduce_max/min`` are documented as supported and are
      bit-exact, but lower to a scalar-emulated reduction: per-core
      ``aiv_scalar_time`` 14.8us vs 4.66us for the fp32 reduce (measured,
      Round 8).  Not a performance option on this toolchain.
    - ``T.vmax`` IS supported for bf16 (docs mark it unsupported) and is
      element-wise bit-exact; there is however no bf16 cross-lane reduce, and a
      manual ``T.vmax`` fold is both slower and (for more than one fold)
      state-lossy -- see opt_log.md Round 8.
"""

import os

# Developer mode: compiler manages UB (alloc_shared) + fragment + auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# Manual UB budget per block (bytes) used for the block_m ladder / the tuned
# ``1 -> 2`` promotion (DESIGN.md section 4.5).
UB_MANUAL_BUDGET = 65536

# Large-buffer bytes per element (DESIGN.md section 4.5 / 3.2 dtype matrix):
# fp16 x_ub=2; fp32 x_ub=4; bf16 x_ub=2 + x_work fp32=4.
_B_ELEM = {"float16": 2, "float32": 4, "bfloat16": 6}

# ---------------------------------------------------------------------------
# Single canonical perf_opt entry module (Stage 4)
# ---------------------------------------------------------------------------
# TileOPs Stage 4 keeps ONE ``perf_opt/`` directory per op but the AmaxFwdOp
# migration unit ships TWO factory functions (resident + streaming).  The
# streaming factory is defined and self-tested in the sibling module
# ``_simple_reduce_kernel_tiled.py``; re-export it here so that
# ``perf_opt/_simple_reduce_kernel.py`` is the single addressable entry module
# for the op, exposing both factories.  The wrapper still imports each factory
# from its own module path (``perf_opt._simple_reduce_kernel`` /
# ``perf_opt._simple_reduce_kernel_tiled``) -- this import yields the *same*
# function object the wrapper binds, so both entry points stay consistent.
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
try:  # imported through the TileOPs package -> package-relative
    from ._simple_reduce_kernel_tiled import _simple_reduce_kernel_tiled  # noqa: E402,F401
except ImportError:  # standalone script / importlib file load
    from _simple_reduce_kernel_tiled import _simple_reduce_kernel_tiled  # noqa: E402,F401

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")

_DTYPE_MAP = {
    "float16": (torch.float16, 1e-2, 1e-2),
    "float32": (torch.float32, 1e-4, 1e-4),
    "bfloat16": (torch.bfloat16, 1e-2, 1e-2),
}


def _ceildiv(a, b):
    return (a + b - 1) // b


def _choose_grid(num_logical, vector_cores):
    """Return ``(nk, exact)`` (Stage 4 S1, tuned).

    Use the FULL vector-core count (``nk = min(num_logical, vector_cores)``)
    with the contiguous, statically-balanced per-core row range (S2 -- core
    ``cid`` owns ``[cid*nl//nk, (cid+1)*nl//nk)``), so no core is left idle and
    the slowest core's tail shrinks.  ``exact=True`` (static trip count, no
    dynamic guard) whenever the balanced range divides exactly.

    Round-7 re-adjudication against the divisor-exact grid (32 cores on the
    hidden-state shape): interleaved A/B/A/B x5 gave full-40 15.80us vs
    divisor-32 16.76us median for hidden-state bf16 (40 wins in 5/5 paired
    runs).  The round-2 "tie" was single-run run-state noise; the sibling
    AminFwdOp measured the same direction (+7.2% bf16, +2.8% fp16).
    """
    nk = min(num_logical, vector_cores)
    return nk, num_logical % nk == 0


# ---------- block_m ladder (DESIGN.md section 5.2) ----------
def select_block_m(M, N, dtype, ub_budget=UB_MANUAL_BUDGET):
    """Factory-period block_m ladder (harness helper; the benchmark path uses
    the block_m computed by the wrapper ``ReduceKernel.default_config``)."""
    b = _B_ELEM[dtype]
    cand = [p for p in (1, 2, 4, 8) if p * N * b <= ub_budget]
    block_m = max(cand) if cand else None
    if block_m is None:
        return None

    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if _ceildiv(M, block_m) > 2 * vector_cores:
        ext = [p for p in (16, 32, 64, 128, 256, 512, 1024, 2048) if p * N * b <= ub_budget]
        block_m = max(ext + [block_m])

    return min(block_m, M)


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_amax(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.amax(x, dim=-1)`` on CPU (fp32 then cast)."""
    return torch.amax(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel factory ----------
def _simple_reduce_kernel(M, N, op_kind, dtype):
    """Build the resident amax kernel factory.

    Mirrors the GPU source structure:
      ``_simple_reduce_kernel(M, N, op_kind, dtype) -> _func(block_m) -> main``.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if op_kind != "amax":
        raise ValueError(f"this unit implements op_kind='amax' only, got {op_kind!r}")
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")

    vector_cores = NPUUtils.get().get_aicore_num() * 2  # pure Vector cores x2
    is_bf16 = dtype == "bfloat16"
    b_elem = _B_ELEM[dtype]

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        # Stage 4: keep the caller-provided block_m (the wrapper's UB heuristic
        # already picks the measured-optimal value for the large-M cases), but
        # promote the degenerate ``1`` to ``2`` when the manual UB budget still
        # allows it -- this is the 3d-multidim (M small, N large) optimum.
        bm = block_m
        if bm == 1 and M >= 2 and 2 * N * b_elem <= UB_MANUAL_BUDGET:
            bm = 2
        bm = min(bm, M)

        num_logical = _ceildiv(M, bm)
        nk, exact = _choose_grid(num_logical, vector_cores)
        L = _ceildiv(num_logical, nk)
        # S6: when M is an exact multiple of bm the row-block extent is a
        # compile-time constant -- emit static ``T.copy`` slices instead of the
        # dynamic ``min(bm, M - off_m)`` form.  Removes per-task scalar extent
        # math / masked-DMA handling from the hot loop.
        static_m = (M % bm == 0)

        if is_bf16:
            # bf16 path: vcast to fp32 -> reduce_max(fp32) -> cast back to bf16.
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(nk, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((bm, N), dtype)
                    x_work = T.alloc_shared((bm, N), "float32")
                    m_all = T.alloc_shared((L * bm, 1), "float32")
                    out_acc = T.alloc_shared((L * bm,), dtype)

                    start = cid * L if exact else cid * num_logical // nk
                    end = start + L if exact else (cid + 1) * num_logical // nk

                    for s in T.serial(L):
                        task = start + s
                        if exact:
                            off_m = task * bm
                            if static_m:
                                T.copy(x[off_m : off_m + bm, 0:N], x_ub)
                            else:
                                real_m = T.min(bm, M - off_m)
                                T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.vcast(x_ub, x_work, round_mode="rint")
                            T.reduce_max(
                                x_work, m_all[s * bm : (s + 1) * bm, 0:1], dim=1, clear=True
                            )
                        else:
                            if task < end:
                                off_m = task * bm
                                if static_m:
                                    T.copy(x[off_m : off_m + bm, 0:N], x_ub)
                                else:
                                    real_m = T.min(bm, M - off_m)
                                    T.copy(
                                        x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N]
                                    )
                                T.vcast(x_ub, x_work, round_mode="rint")
                                T.reduce_max(
                                    x_work,
                                    m_all[s * bm : (s + 1) * bm, 0:1],
                                    dim=1,
                                    clear=True,
                                )

                    for i in T.Parallel(L * bm):
                        out_acc[i] = T.cast(m_all[i, 0], dtype)
                    base = T.min(start * bm, M)
                    valid = T.min(end * bm, M) - base
                    T.copy(out_acc[0:valid], out[base : base + valid])

        else:
            # fp16 / fp32 path: original-domain reduce_max, src/dst same dtype.
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(nk, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((bm, N), dtype)
                    m_all = T.alloc_shared((L * bm, 1), dtype)
                    out_acc = T.alloc_shared((L * bm,), dtype)

                    start = cid * L if exact else cid * num_logical // nk
                    end = start + L if exact else (cid + 1) * num_logical // nk

                    for s in T.serial(L):
                        task = start + s
                        if exact:
                            off_m = task * bm
                            if static_m:
                                T.copy(x[off_m : off_m + bm, 0:N], x_ub)
                            else:
                                real_m = T.min(bm, M - off_m)
                                T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.reduce_max(
                                x_ub, m_all[s * bm : (s + 1) * bm, 0:1], dim=1, clear=True
                            )
                        else:
                            if task < end:
                                off_m = task * bm
                                if static_m:
                                    T.copy(x[off_m : off_m + bm, 0:N], x_ub)
                                else:
                                    real_m = T.min(bm, M - off_m)
                                    T.copy(
                                        x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N]
                                    )
                                T.reduce_max(
                                    x_ub,
                                    m_all[s * bm : (s + 1) * bm, 0:1],
                                    dim=1,
                                    clear=True,
                                )

                    for i in T.Parallel(L * bm):
                        out_acc[i] = m_all[i, 0]
                    base = T.min(start * bm, M)
                    valid = T.min(end * bm, M) - base
                    T.copy(out_acc[0:valid], out[base : base + valid])

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(M, N, dtype_str, block_m, tag, equal_nan=False):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _simple_reduce_kernel(M, N, "amax", dtype_str)(block_m)
    y = kernel(x)
    ref = golden_amax(x.cpu())
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} max_diff={max_diff:.2e}"
    )
    return max_diff


def _run_case_from_values(x, dtype_str, tag, equal_nan=False):
    """Run a pre-built CPU tensor through the kernel; check against golden."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    M, N = x.shape
    block_m = min(select_block_m(M, N, dtype_str) or 1, M)
    x_npu = x.to(dtype=torch_dtype, device="npu")
    kernel = _simple_reduce_kernel(M, N, "amax", dtype_str)(block_m)
    y = kernel(x_npu)
    ref = golden_amax(x.to(dtype=torch_dtype))
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} max_diff={max_diff:.2e}")
    return max_diff


# ---------- L0: gate tests (must pass) ----------
def run_L0():
    max_diff = 0.0
    max_diff = max(max_diff, _run_case(32, 256, "float16", 8, "L0-1a"))
    max_diff = max(max_diff, _run_case(32, 256, "bfloat16", 8, "L0-1b"))
    max_diff = max(max_diff, _run_case(32, 256, "float32", 8, "L0-1c"))
    max_diff = max(max_diff, _run_case(2048, 4096, "float16", 8, "L0-2"))
    max_diff = max(max_diff, _run_case(2048, 4096, "bfloat16", 2, "L0-3"))
    max_diff = max(max_diff, _run_case(128, 16384, "float16", 2, "L0-4"))
    max_diff = max(max_diff, _run_case(128, 512, "float32", 8, "L0-5"))
    max_diff = max(max_diff, _run_case(128, 300, "float16", 8, "L0-6a"))
    max_diff = max(max_diff, _run_case(128, 300, "bfloat16", 8, "L0-6b"))
    max_diff = max(max_diff, _run_case(129, 512, "float16", 8, "L0-7"))
    max_diff = max(max_diff, _run_case(1, 512, "float16", 1, "L0-8"))
    for dtype_str in ("float16", "float32"):
        max_diff = max(max_diff, _run_boundary_rows(dtype_str, "L0-9"))
    print(f"[L0] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


def _boundary_rows(dtype_str):
    """Construct (8, 257) rows covering the E1/E3 corner set."""
    torch_dtype = _DTYPE_MAP[dtype_str][0]
    finfo = torch.finfo(torch_dtype)
    rows = [
        torch.full((257,), float("-inf")),
        torch.full((257,), float("inf")),
        torch.cat([torch.tensor([float("inf")]), torch.zeros(256)]),
        torch.cat([torch.tensor([float("-inf")]), torch.zeros(256)]),
        torch.cat([torch.full((256,), 3.0), torch.tensor([3.0])]),
        torch.arange(257, dtype=torch.float32) * 0.5,
        torch.zeros(257),
        torch.cat([torch.tensor([finfo.max]), torch.zeros(256)]),
    ]
    x = torch.stack(rows, dim=0).to(torch.float32)
    if dtype_str == "float16":
        x = torch.nan_to_num(x, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return x


def _run_boundary_rows(dtype_str, tag):
    x = _boundary_rows(dtype_str).to(_DTYPE_MAP[dtype_str][0])
    return _run_case_from_values(x, dtype_str, tag)


def _run_nan_case(M, N, dtype_str, tag):
    """NaN input: torch.amax propagates NaN; check pointwise equality on NPU."""
    torch_dtype = _DTYPE_MAP[dtype_str][0]
    x = torch.randn(M, N).to(torch_dtype)
    x[0, 1] = float("nan")
    x[2, 5] = float("nan")
    return _run_case_from_values(x, dtype_str, tag, equal_nan=True)


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    max_diff = max(max_diff, _run_case(130, 256, "float16", 8, "L1-1"))
    max_diff = max(max_diff, _run_case(64, 1024, "bfloat16", 4, "L1-2"))
    max_diff = max(max_diff, _run_case(8, 512, "float16", 8, "L1-3"))
    max_diff = max(max_diff, _run_case(256, 1024, "float32", 8, "L1-4"))
    max_diff = max(max_diff, _run_case(3, 300, "bfloat16", 2, "L1-5"))
    max_diff = max(max_diff, _run_case(32, 32, "float32", 8, "L1-6"))
    max_diff = max(max_diff, _run_case(255, 777, "float16", 4, "L1-7"))
    max_diff = max(max_diff, _run_nan_case(32, 256, "float16", "L1-8a-nan"))
    max_diff = max(max_diff, _run_nan_case(32, 256, "float32", "L1-8b-nan"))
    print(f"[L1] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, "float16", 1),
        (8, 256, "float16", 8),
        (9, 256, "float16", 8),
        (32, 1, "float16", 4),
    ]
    for M, N, dtype_str, block_m in cases:
        try:
            _run_case(M, N, dtype_str, block_m, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    try:
        x = torch.zeros(32, 256, dtype=torch.float16)
        x[:, 0] = -0.0
        _run_case_from_values(x, "float16", "Boundary-zeros-fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {e}")

    try:
        finfo = torch.finfo(torch.float16)
        x = torch.full((32, 256), finfo.tiny / 2, dtype=torch.float16)
        _run_case_from_values(x, "float16", "Boundary-subnormal-fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {e}")

    for val, name in ((float("-inf"), "all-neg-inf"), (float("inf"), "all-pos-inf")):
        try:
            x = torch.full((32, 256), val, dtype=torch.float16)
            _run_case_from_values(x, "float16", f"Boundary-{name}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): {name}: {e}")

    try:
        _run_case(1024, 2048, "float16", 4, "Boundary-large-M")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large M (1024,2048): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
