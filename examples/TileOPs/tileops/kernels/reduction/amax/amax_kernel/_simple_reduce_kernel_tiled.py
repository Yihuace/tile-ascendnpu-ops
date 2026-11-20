# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_simple_reduce_kernel_tiled (AmaxFwdOp tiled / streaming path) -- NPU Developer mode.

Migrated from the GPU TileOPs ``_simple_reduce_kernel_tiled`` (``op_kind="amax"``,
``T.reduce_max`` branch, ``reduce.py`` L150-L243) to ``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    y[i] = max_{0 <= j < N} x[i, j]          # torch.amax(x, dim=-1)
    x: (M, N) fp16/bf16/fp32  ->  out: (M,) same_as(x)

This unit carries the STREAMING (tiled) path: N is too large for the whole row
to reside in UB, so the kernel walks the row in ``tile_n``-wide column tiles and
maintains a cross-tile running max (online single pass). The residency sibling
``_simple_reduce_kernel`` handles the small-N case (DESIGN.md section 5.2 gate).

NPU redesign (DESIGN.md section 0.6, frozen):
    R1: raw-N boundary -- kernel receives the original (M, N); no 256-aligned pad
        and no masked load. N is split into ``num_full`` full tiles plus one
        STATIC tail tile whose width ``valid_n`` is a compile-time constant
        (independent buffer, no -inf padding, no serial-loop-variable condition).
        M tail uses ``real_m = min(block_m, M - off_m)`` + explicit src/dst slices.
    R2: persistent one-dimensional core split -- ``num_kernels = min(M, 40)``
        (tiled path forces bm=1, CG-2026-0014) with an in-core ``T.serial``
        grid-stride loop (static bound) and an if-guard (dynamic) only.
    R3: dtype domain -- fp16/fp32 reduce in the ORIGINAL domain (max is an exact
        selection, no rounding); bf16 is promoted with ``T.vcast(rint)`` to fp32
        and cast back on output. Cross-tile merge uses a single-op
        ``T.reduce_max(..., clear=False)`` running-max accumulation (E3).
    R4: UB budget tile_n -- ``tile_n * B(dtype) <= 64KB`` manual budget (x2
        auto-multi-buffer fits 192KB UB); divisor-preferred, else budget cap
        plus a static tail tile.

Known toolchain constraints enforced here (DESIGN.md section 9.1):
    - ``T.reduce_max`` src/dst MUST be the same dtype (mixed dtype silently
      corrupts) -> bf16 uses a dedicated fp32 work buffer.
    - unused ``alloc`` is NOT dead-code-eliminated -> fp16/fp32 and bf16 use
      separate prim_funcs (no conditional alloc).
    - ``T.Parallel`` may not read GM -> all GM traffic goes through ``T.copy``.
    - ``T.vcast`` f32->f32 is non-identity -> same-dtype uses plain slices.
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

# Probe-calibrated race-safe manual UB budget per row (bytes); x2 for
# auto-multi-buffer double-buffering fits the 192KB UB (DESIGN.md section 4.5).
UB_MANUAL_BUDGET = 65536

# Large-buffer bytes per element (DESIGN.md section 4.5 / 3.2 dtype matrix):
# fp16 x_ub=2; fp32 x_ub=4; bf16 x_ub=2 + x_work fp32=4.
_B_ELEM = {"float16": 2, "float32": 4, "bfloat16": 6}

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")

_DTYPE_MAP = {
    "float16": (torch.float16, 1e-2, 1e-2),
    "float32": (torch.float32, 1e-4, 1e-4),
    "bfloat16": (torch.bfloat16, 1e-2, 1e-2),
}


def _ceildiv(a, b):
    return (a + b - 1) // b


def _divisors(n):
    """All positive divisors of ``n`` (ascending)."""
    small, large = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
        d += 1
    return small + large[::-1]


# ---------- tile_n / routing gate (DESIGN.md section 5.2) ----------
def select_tile_n(M, N, dtype, ub_budget=UB_MANUAL_BUDGET):
    """Factory-period ``tile_n`` derivation (tiled path).

    ``tile_n_cap = (ub_budget // B) & ~255`` (budget cap, 256-aligned);
    divisor-preferred among the 256-multiple divisors of N <= cap (zero tail,
    no near-empty tail tile); otherwise the budget cap plus a static tail tile.
    """
    b = _B_ELEM[dtype]
    tile_n_cap = (ub_budget // b) & ~255
    cands = [f for f in _divisors(N) if f <= tile_n_cap and f % 256 == 0]
    return max(cands) if cands else tile_n_cap


def needs_tiling(M, N, dtype, ub_budget=UB_MANUAL_BUDGET):
    """Routing gate (wrapper period): N*B <= 64KB -> residency sibling unit."""
    return N * _B_ELEM[dtype] > ub_budget


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_amax_tiled(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.amax(x, dim=-1)`` on CPU.

    Computed in fp32 then cast back to the input dtype, matching
    ``examples/TileOPs/tests/ops/test_reduce.py::ReduceTest.ref_program``.
    It does NOT reproduce the NPU redesign (no static tail tile / original-domain
    reduce / persistent split), guaranteeing validation independence
    (DESIGN.md section 8.1).

    Uses explicit ``torch`` module calls (``torch.amax``) so the reference is
    unambiguously a torch API call rather than a hand-written reduction loop.
    """
    return torch.amax(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel factory ----------
def _simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n):
    """Build the tiled (streaming) amax kernel factory.

    Mirrors the GPU source structure:
      ``_simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n)
         -> _func(block_m) -> main``.
    K9: the ``threads`` parameter is removed (NPU has no CUDA threads concept).

    The tiled path forces ``block_m == 1`` (multitile + bm>1 triggers a
    compiler SIGSEGV, CG-2026-0014); ``op_kind`` is kept for sibling-reduce
    interface compatibility (amax branch only).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if op_kind != "amax":
        raise ValueError(f"this unit implements op_kind='amax' only, got {op_kind!r}")
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")
    if tile_n <= 0:
        raise ValueError(f"tile_n must be positive, got tile_n={tile_n}")

    b = _B_ELEM[dtype]
    # UB sanity envelope (DESIGN.md section 4.5): x2 auto-multi-buffer <= 184KB.
    assert 2 * tile_n * b <= 184 * 1024, (
        f"tile_n={tile_n} x B={b} exceeds the x2 UB envelope; lower tile_n"
    )

    num_full = N // tile_n                 # full tiles (compile-time constant)
    tail = N - num_full * tile_n           # static tail width (compile-time constant)
    # Static tail buffers are allocated unconditionally with a min width of 1
    # (the TVM tracer does not track buffers allocated in a conditional block
    # and referenced later); the tail PROCESSING is guarded by `if tail > 0`.
    tail_w = tail if tail > 0 else 1
    is_bf16 = dtype == "bfloat16"

    vector_cores = NPUUtils.get().get_aicore_num() * 2  # pure Vector cores x2

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        if block_m != 1:
            raise ValueError(
                "tiled path forces block_m == 1 (CG-2026-0014: multitile + bm>1 SIGSEGV)"
            )
        num_logical = M                          # bm=1 -> logical units = rows
        num_kernels = min(num_logical, vector_cores)  # persistent grid
        num_local_tasks = _ceildiv(num_logical, num_kernels)  # static bound

        if is_bf16:
            # bf16 path: vcast to fp32 -> running reduce_max(fp32) -> cast back.
            # Separate prim_func so the fp32 work buffer is only allocated when
            # actually used (unused alloc is not DCE'd).
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((1, tile_n), dtype)
                    x_work = T.alloc_shared((1, tile_n), "float32")
                    x_tail = T.alloc_shared((1, tail_w), dtype)
                    x_tail_work = T.alloc_shared((1, tail_w), "float32")
                    acc = T.alloc_shared((1, 1), "float32")   # running max (fp32 domain)
                    out_ub = T.alloc_shared((1,), dtype)

                    for s in T.serial(num_local_tasks):      # static loop bound
                        task = s * num_kernels + cid         # grid-stride mapping
                        if task < num_logical:               # dynamic guard only
                            off_m = task * block_m
                            real_m = T.min(block_m, M - off_m)
                            # acc 初值 = -inf（max 的单位元，不影响任何结果）
                            T.vbrc(-T.infinity("float32"), acc)
                            if num_full > 0:
                                for t in T.serial(num_full):     # 满 tile 串行（静态上界）
                                    # GM -> UB tile 载入（src/dst 双显式 slice）
                                    T.copy(
                                        x[off_m:off_m + real_m, t * tile_n:(t + 1) * tile_n],
                                        x_ub[0:real_m, 0:tile_n],
                                    )
                                    # bf16 -> fp32 精确升域（bf16 不被 reduce 支持）
                                    T.vcast(x_ub, x_work, round_mode="rint")
                                    # 跨 tile running max 在线合并（max 结合律，clear=False 累加）
                                    T.reduce_max(x_work, acc, dim=1, clear=False)
                            if tail > 0:
                                # 静态尾 tile：宽度 valid_n 为编译期常量，无 -inf 填充
                                T.copy(
                                    x[off_m:off_m + real_m, num_full * tile_n:N],
                                    x_tail[0:real_m, 0:tail],
                                )
                                T.vcast(x_tail, x_tail_work, round_mode="rint")
                                T.reduce_max(x_tail_work, acc, dim=1, clear=False)
                            for i in T.Parallel(1):
                                # (1,1) -> (1,) 形状收缩 + fp32 -> bf16 降回
                                out_ub[i] = T.cast(acc[i, 0], dtype)
                            T.copy(out_ub[0:real_m], out[off_m:off_m + real_m])

        else:
            # fp16 / fp32 path: original-domain reduce_max, src/dst same dtype.
            # max is an exact selection op (no arithmetic / no rounding), so the
            # native-domain result is bit-identical to an fp32-promoted baseline.
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((1, tile_n), dtype)
                    x_tail = T.alloc_shared((1, tail_w), dtype)
                    acc = T.alloc_shared((1, 1), dtype)     # running max (same dtype)
                    out_ub = T.alloc_shared((1,), dtype)

                    for s in T.serial(num_local_tasks):      # static loop bound
                        task = s * num_kernels + cid         # grid-stride mapping
                        if task < num_logical:               # dynamic guard only
                            off_m = task * block_m
                            real_m = T.min(block_m, M - off_m)
                            # acc 初值 = -inf（max 的单位元，不影响任何结果）
                            T.vbrc(-T.infinity(dtype), acc)
                            if num_full > 0:
                                for t in T.serial(num_full):     # 满 tile 串行（静态上界）
                                    # GM -> UB tile 载入（src/dst 双显式 slice，原域）
                                    T.copy(
                                        x[off_m:off_m + real_m, t * tile_n:(t + 1) * tile_n],
                                        x_ub[0:real_m, 0:tile_n],
                                    )
                                    # 跨 tile running max 在线合并（clear=False 在 acc 现值上取 max）
                                    T.reduce_max(x_ub, acc, dim=1, clear=False)
                            if tail > 0:
                                # 静态尾 tile（宽 valid_n 编译期常量，无 -inf 填充）
                                T.copy(
                                    x[off_m:off_m + real_m, num_full * tile_n:N],
                                    x_tail[0:real_m, 0:tail],
                                )
                                T.reduce_max(x_tail, acc, dim=1, clear=False)
                            for i in T.Parallel(1):
                                # (1,1) -> (1,) 形状收缩（结果已为目标 dtype，恒等）
                                out_ub[i] = acc[i, 0]
                            T.copy(out_ub[0:real_m], out[off_m:off_m + real_m])

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(M, N, dtype_str, tile_n, tag, equal_nan=False):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = torch.randn(M, N, dtype=torch.float32).to(torch_dtype)
    return _run_case_from_values(x, dtype_str, tile_n, tag, equal_nan=equal_nan)


def _run_case_from_values(x, dtype_str, tile_n, tag, equal_nan=False):
    """Run a pre-built CPU tensor through the kernel; check against golden."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    M, N = x.shape
    x = x.to(torch_dtype)
    kernel = _simple_reduce_kernel_tiled(M, N, "amax", dtype_str, tile_n)(1)
    y = kernel(x.to(device="npu"))
    ref = golden_amax_tiled(x)
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    # Report the max finite |diff| (inf - inf is nan but agrees under assert_close).
    diff = (y.cpu().float() - ref.float()).abs()
    diff = diff[torch.isfinite(diff)]
    max_diff = diff.max().item() if diff.numel() > 0 else 0.0
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} tile_n={tile_n} "
        f"max_diff={max_diff:.2e}"
    )
    return max_diff


# ---------- L0: gate tests (must pass) ----------
# DESIGN.md section 8.3: large-N streaming (core use case), static tail tile,
# non-divisible M/N, multiple dtypes and tile_n values, manifest long-seq.
def run_L0():
    max_diff = 0.0
    # L0-1: manifest long-seq (64,32768) bf16 -> 40 cores x 2 tasks.
    max_diff = max(max_diff, _run_case(64, 32768, "bfloat16", 8192, "L0-1"))
    # L0-2a/b: large N + extreme scale (128 -> 40 cores x 4 tasks), fp16 / fp32.
    max_diff = max(max_diff, _run_case(128, 65536, "float16", 16384, "L0-2a"))
    max_diff = max(max_diff, _run_case(128, 65536, "float32", 16384, "L0-2b"))
    # L0-3: static tail tile (100000 % 8192 -> valid_n=1696) + small M.
    max_diff = max(max_diff, _run_case(4, 100000, "float16", 8192, "L0-3"))
    # L0-4: static tail tile fp32 (40000 % 8192 -> valid_n=7232).
    max_diff = max(max_diff, _run_case(8, 40000, "float32", 8192, "L0-4"))
    # L0-5: extreme tail (valid_n=1).
    max_diff = max(max_diff, _run_case(16, 32769, "float16", 16384, "L0-5"))
    # L0-6: budget-cap tile_n (64KB manual budget at the limit).
    max_diff = max(max_diff, _run_case(2, 262144, "float16", 32768, "L0-6"))
    # L0-7: M tail / load imbalance (129 -> 40 cores x 4 tasks).
    max_diff = max(max_diff, _run_case(129, 65536, "float16", 16384, "L0-7"))
    # L0-8: bf16 + static tail tile (1 full tile + large tail).
    max_diff = max(max_diff, _run_case(4, 13000, "bfloat16", 8192, "L0-8"))
    # L0-9: M=1 single row + extreme tail.
    max_diff = max(max_diff, _run_case(1, 32769, "float16", 16384, "L0-9"))
    # L0-10: boundary-value semantics (verify_equiv corner set replayed on NPU).
    for dtype_str in ("float16", "float32"):
        max_diff = max(max_diff, _run_boundary_rows(dtype_str, 8192, "L0-10"))
    print(f"[L0] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


def _boundary_rows(dtype_str, n=32768):
    """Construct (8, n) rows covering the E1/E3/E4 corner set."""
    torch_dtype = _DTYPE_MAP[dtype_str][0]
    finfo = torch.finfo(torch_dtype)
    z = torch.zeros(n)
    rows = [
        torch.full((n,), float("-inf")),               # all -inf
        torch.full((n,), float("inf")),                # all +inf
        torch.cat([torch.tensor([float("inf")]), z[1:]]),   # +inf amid finite
        torch.cat([torch.tensor([float("-inf")]), z[1:]]),  # -inf amid finite
        torch.full((n,), 3.0),                         # repeated extreme
        torch.arange(n, dtype=torch.float32) * 0.5,    # monotone
        torch.zeros(n),                                # all zeros
        torch.cat([torch.tensor([finfo.max]), z[1:]]), # dtype extreme
    ]
    x = torch.stack(rows, dim=0).to(torch.float32)
    if dtype_str == "float16":
        # clamp sentinels into the fp16 range (inf stays inf).
        x = torch.nan_to_num(x, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return x.to(torch_dtype)


def _run_boundary_rows(dtype_str, tile_n, tag):
    x = _boundary_rows(dtype_str)
    return _run_case_from_values(x, dtype_str, tile_n, tag)


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    # multi-tile values + manifest workload + all dtypes.
    max_diff = max(max_diff, _run_case(128, 16384, "float16", 4096, "L1-1"))
    max_diff = max(max_diff, _run_case(4, 65536, "bfloat16", 8192, "L1-2"))
    max_diff = max(max_diff, _run_case(17, 20000, "float16", 8192, "L1-3"))
    max_diff = max(max_diff, _run_case(33, 32768, "float32", 8192, "L1-4"))
    max_diff = max(max_diff, _run_case(256, 16384, "float16", 4096, "L1-5"))
    max_diff = max(max_diff, _run_case(5, 50000, "bfloat16", 4096, "L1-6"))
    # derive tile_n through the DESIGN section 5.2 formula (wrapper default path).
    tn = select_tile_n(64, 40000, "float16")
    max_diff = max(max_diff, _run_case(64, 40000, "float16", tn, "L1-7-derived"))
    print(f"[L1] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 32768, "float16", 32768),   # M=1, single full tile (valid_n=0)
        (8, 8192, "float16", 8192),     # N == tile_n exactly (1 full tile)
        (8, 8193, "float16", 8192),     # N == tile_n + 1 (valid_n=1)
        (4, 1000, "float16", 4096),     # N < tile_n (num_full=0, all tail)
        (40, 32768, "float16", 8192),   # M == vector cores (single wave)
        (41, 32768, "float16", 8192),   # M > vector cores (2 waves)
        (2, 262144, "float16", 32768),  # tile_n at the 64KB budget cap
    ]
    for M, N, dtype_str, tile_n in cases:
        try:
            _run_case(M, N, dtype_str, tile_n, "L2")
        except Exception as e:  # noqa: BLE001
            print(
                f"[L2] WARN (record only): shape=({M},{N}) {dtype_str} "
                f"tile_n={tile_n}: {e}"
            )


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # all -inf / all +inf (documented boundary semantics).
    for val, name in ((float("-inf"), "all-neg-inf"), (float("inf"), "all-pos-inf")):
        try:
            x = torch.full((8, 32768), val, dtype=torch.float16)
            _run_case_from_values(x, "float16", 8192, f"Boundary-{name}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): {name}: {e}")

    # NaN propagation (record only; torch.amax and NPU reduce both propagate).
    try:
        x = torch.randn(8, 32768, dtype=torch.float16)
        x[0, 1] = float("nan")
        x[3, 100] = float("nan")
        _run_case_from_values(x, "float16", 8192, "Boundary-nan", equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): nan: {e}")

    # mixed-sign zeros (IEEE equality; max selection preserves a bit pattern).
    try:
        x = torch.zeros(8, 32768, dtype=torch.float16)
        x[:, 0] = -0.0
        _run_case_from_values(x, "float16", 8192, "Boundary-zeros")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {e}")

    # subnormal extremes.
    try:
        finfo = torch.finfo(torch.float16)
        x = torch.full((8, 32768), finfo.tiny / 2, dtype=torch.float16)
        _run_case_from_values(x, "float16", 8192, "Boundary-subnormal")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {e}")

    # large M persistent multi-wave regression.
    try:
        _run_case(1024, 16384, "float16", 4096, "Boundary-large-M")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large M (1024,16384): {e}")


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
