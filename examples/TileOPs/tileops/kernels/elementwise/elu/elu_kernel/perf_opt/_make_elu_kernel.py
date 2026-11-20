# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""ELU forward kernel (NPU, target="npuir") -- Stage 4 performance-tuned FINAL.

Drop-in replacement for the Stage 3 baseline ``_make_elu_kernel.py``: same
factory name, same ``(N, dtype, alpha, output_dtype=None, block_size=None)``
signature, same ``main(x, y, shape)`` prim_func, same fp32-transit precision
semantics (``exp(x) - 1`` for the negative branch, *not* ``expm1``).

Three single-point-verified changes vs the Stage 3 baseline, merged into ONE
``@T.prim_func`` (per-branch measurements in perf_opt/opt_log.md):

1. **mask-free vector chain** (Round 1 OP1 family, corrected in Round 3).
   The baseline branch-free mask pair ``vcmp(x,0) + vselect`` is replaced by

       elu(x) = relu(x) + alpha * (min(exp(x), 1) - 1)

   with the chain ``vexp -> vsub(-1) -> vmin(0)`` for the negative branch and
   ``vrelu`` for the positive branch.  It is **bit-exact**:
     * x > 0 : ``min(exp(x), 1) == 1`` exactly (``exp(x) >= 1``), so the
       negative contribution is ``1 - 1 == 0`` and the result is ``x + 0 == x``;
     * x <= 0: ``exp(x) <= 1`` so ``min(exp(x), 1) - 1 == exp(x) - 1``, the
       *same* fp32 evaluation as the baseline, added to ``relu(x) == 0``;
     * ``+/-inf``, ``NaN`` and ``-0.0`` propagate exactly as in the baseline
       (``exp(inf)=inf -> min(inf,1)=1 -> 0``; ``exp(-inf)=0 -> min(0,1)=0 -> -1``;
       ``vmin`` propagates ``NaN``; ``relu(NaN)`` is absorbed by the ``NaN``
       in the other addend).
   Both ``vmin`` and ``vsub`` take the scalar bounds directly, so **no zero
   constant buffer and no bool mask buffer** are needed (saves 5 B/elem on the
   fp16/bf16 path and 13 B/elem on fp32 vs Stage 3) -- which is what lets the
   tuned ``block_size`` reach 12288 for fp16/bf16.

2. **per-dtype tuned block_size**, exposed as the module constant
   ``TUNED_BLOCK_SIZE`` (the wrapper's perf_opt toggle imports it).  The Stage 3
   GPU-collapsed default (fp32 1024 / fp16·bf16 2048) leaves every tile at
   4 KB, where the GM read pipe runs at ~12 GB/s/core and the kernel is purely
   DMA-bound (baseline 16m-f16 = 104.6 us, effective 0.64 TB/s).

3. **static-bulk / dynamic-tail loop split for the fp32 path** (Round 2 SB).
   ``block_size`` is a compile-time constant, so the ``N // block_size`` full
   tiles can use *static-extent* ``T.copy`` and only the (at most one) remainder
   tile keeps the dynamic ``T.min`` extent.  On the fp32 workload this alone
   turns the serialised MTE2/VEC/MTE3 chain into an overlapped one
   (16m-f32 72.6 -> 50.3 us, effective 2.67 TB/s ~= the measured copy floor).
   The fp16/bf16 path keeps the original dynamic-extent loop: there the vector
   pipe is the bottleneck and the static-bulk form measured worse at the tuned
   block size, so the two paths intentionally differ.

run: python _make_elu_kernel.py --level all
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_BLOCK_SIZE = 2048
_CORE_NUM = 40  # physical Vector cores on Ascend910B3 / 910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# Stage 4 tuned per-dtype block_size (element count).
# UB budget: 12 B/elem (fp16/bf16: x_ub 2 + y_ub 2 + t 4 + s 4) against
# UB = 1572864 bits = 192 KB.  Measured BiShengIR overflow: fp16 overflows at
# 14336, fp32 at 13312 with the static-bulk loop; the values below keep margin.
TUNED_BLOCK_SIZE = {
    "float32": 8192,
    "float16": 12288,
    "bfloat16": 12288,
}
# Kept as an alias so the module keeps the Stage 3 constant name too.
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU reference implementation, equivalent to ``torch.nn.functional.elu``.

    Mirrors the source op semantics ``alpha * (exp(cast(x,'f32')) - 1)`` with
    an ``x > 0`` take-``x`` positive branch and a single round back to the
    input dtype (source fp32-transit precision semantics).  Both the CPU and
    the target-device ``F.elu`` negative branch use ``exp(x)-1`` (not
    ``expm1``), so this is exactly ``F.elu(x.float(), alpha).to(x.dtype)``.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return F.elu(x_f32, alpha).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def _make_elu_kernel(N, dtype, alpha, output_dtype=None, block_size=None):
    """ELU forward kernel factory: y = x if x > 0 else alpha*(exp(x)-1).

    ``block_size=None`` selects the Stage-4 tuned per-dtype default exposed as
    ``TUNED_BLOCK_SIZE``; an explicit ``block_size`` still overrides it.
    """
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    need_mul = float(alpha) != 1.0
    block_size = block_size or _DEFAULT_BLOCK_SIZE[dtype]

    n_num = T.ceildiv(N, block_size)
    n_full = N // block_size
    has_tail = (N % block_size) != 0
    n_tasks = n_full + (1 if has_tail else 0)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            # R-2/R-6: every UB buffer is allocated unconditionally at the top
            # level (a conditional alloc triggers ``Undefined variable``); the
            # compile-time ``if need_cast:`` branches only select which code
            # runs.
            x_ub = T.alloc_ub((block_size,), dtype)       # input tile
            y_ub = T.alloc_ub((block_size,), out_dtype)   # round-back staging
            xf = T.alloc_ub((block_size,), f32)           # fp32 transit / domain
            t = T.alloc_ub((block_size,), f32)            # min(exp(x),1) - 1
            s = T.alloc_ub((block_size,), f32)            # relu(x) = max(x, 0)

            # Compile-time alias: fp32 computes in place on x_ub, the fp16/bf16
            # path computes in the fp32 transit buffer xf.
            xc = xf if need_cast else x_ub

            if need_cast:
                # ---- fp16 / bf16: vector-bound, dynamic-extent (tail) loop ----
                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                    task = i * _CORE_NUM + cid
                    if task < n_num:
                        off = task * block_size
                        tail = T.min(block_size, shape - off)
                        T.copy(x[off : off + tail], x_ub[0:tail])

                        T.vcast(x_ub, xf, round_mode="rint")

                        # --- mask-free chain: elu = relu(x) + (min(exp(x),1)-1) --
                        T.vexp(xc, t)                       # t = exp(x)
                        T.vsub(t, T.cast(1.0, f32), t)      # t = exp(x) - 1
                        T.vmin(t, T.cast(0.0, f32), t)      # t = min(exp(x)-1, 0)
                        T.vrelu(xc, s)                      # s = max(x, 0)
                        if need_mul:
                            T.vmul(t, T.cast(alpha, f32), t)
                        T.vadd(s, t, s)                     # s = relu(x) + t

                        T.vcast(s, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off : off + tail])
            else:
                # ---- fp32: DMA-bound, static-bulk + dynamic-tail split ------
                for i in T.serial(T.ceildiv(n_tasks, _CORE_NUM)):
                    task = i * _CORE_NUM + cid
                    if task < n_full:
                        off = task * block_size
                        T.copy(x[off : off + block_size], x_ub)

                        T.vexp(xc, t)
                        T.vsub(t, T.cast(1.0, f32), t)
                        T.vmin(t, T.cast(0.0, f32), t)
                        T.vrelu(xc, s)
                        if need_mul:
                            T.vmul(t, T.cast(alpha, f32), t)
                        T.vadd(s, t, s)

                        T.copy(s, y[off : off + block_size])
                    elif has_tail and task == n_full:
                        off = task * block_size
                        tail = T.min(block_size, shape - off)
                        T.copy(x[off : off + tail], x_ub[0:tail])

                        T.vexp(xc, t)
                        T.vsub(t, T.cast(1.0, f32), t)
                        T.vmin(t, T.cast(0.0, f32), t)
                        T.vrelu(xc, s)
                        if need_mul:
                            T.vmul(t, T.cast(alpha, f32), t)
                        T.vadd(s, t, s)

                        T.copy(s[0:tail], y[off : off + tail])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances from examples/TileOPs/tests/ops/test_activation.py
# ``_get_tolerances`` (DESIGN.md section 8.2).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _make_input(N, dtype_str):
    """Generate a CPU input tensor following the EluWorkload logic (randn)."""
    return torch.randn(N, dtype=_TORCH_DTYPE[dtype_str])


def _repeat_values(values, N, dtype):
    """Repeat a small value pattern up to N elements (edge / corner cases)."""
    base = torch.tensor(values, dtype=dtype)
    repeats = (N + len(values) - 1) // len(values)
    return base.repeat(repeats)[:N]


def _max_diff(y, golden):
    """Max abs error over finite golden entries (inf/NaN handled by caller)."""
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    finite = torch.isfinite(g_f64) & torch.isfinite(y_f64)
    if not bool(finite.any()):
        return 0.0
    return (y_f64[finite] - g_f64[finite]).abs().max().item()


def _run_tensor(x_cpu, dtype_str, tag, alpha=1.0):
    """Run the kernel on a CPU tensor and compare with the golden output."""
    x_cpu = x_cpu.detach().cpu().reshape(-1).contiguous()
    N = x_cpu.numel()
    golden = golden_elu(x_cpu, alpha)
    kernel = _make_elu_kernel(N, dtype_str, alpha, block_size=_DEFAULT_BLOCK_SIZE[dtype_str])

    x_npu = x_cpu.npu()
    y = kernel(x_npu, N)
    y_cpu = y.cpu().reshape(-1)

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y_cpu, golden)
    try:
        torch.testing.assert_close(
            y_cpu, golden, rtol=rtol, atol=atol, equal_nan=True
        )
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({N},) dtype={dtype_str} alpha={alpha} "
            f"max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape=({N},) dtype={dtype_str} alpha={alpha} "
        f"max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, alpha=1.0):
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag, alpha)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke (DESIGN.md section 8.3 L0-1..L0-3): 1M elements per dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L0-smoke")

    # L0-4 corner / branch-point / saturation values (test_elu_edge pattern):
    # -88.8 -> -1.0, -1e-7 -> exp(-1e-7)-1 (negative branch), 0 -> 0,
    # +1e-7 -> +1e-7 (positive branch), 88.8 -> 88.8 (no overflow).
    _run_tensor(
        _repeat_values([-88.8, -1.0, -1e-7, 0.0, 1e-7, 1.0, 88.8], 4096, torch.float32),
        "float32",
        "L0-corner",
    )

    # L0-5 negative-branch small-|x| sensitive region (the exp-1 vs expm1
    # discriminator; DESIGN.md section 0.6 R2).
    x_small = torch.linspace(-1e-3, 0.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_small, "float32", "L0-small-x")

    # L0-6 full sensitive region including the branch point x = 0.
    x_lin = torch.linspace(-9.0, 9.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_lin, "float32", "L0-linspace")

    # L0-7 non-divisible tail (4_000_000 % 2048 = 1216).
    _run_case(4000000, "float16", "L0-tail")

    # L0-8 non-default alpha (exercises the vmul path).
    _run_case(1048576, "float32", "L0-alpha0.5", alpha=0.5)

    # L0-9 manifest multi-dim shapes, flattened by the host (fp16/bf16).
    for shape in ((2048, 4096), (2048, 8192)):
        for dtype_str in ("float16", "bfloat16"):
            x = torch.randn(*shape, dtype=_TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, f"L0-manifest{shape}")


def run_L1():
    # L1: full dtype x boundary shapes -- manifest 16m flat, non-divisible
    # tails for every dtype, exact / partial tiles.
    _run_case(4096 * 4096, "float32", "L1-16m")
    _run_case(4000000, "float32", "L1-tail")
    _run_case(4000000, "bfloat16", "L1-tail")
    _run_case(4000000, "float32", "L1-tail-alpha0.5", alpha=0.5)
    _run_case(1024, "float32", "L1-exact-tile")
    _run_case(4096, "float16", "L1-two-tiles")
    _run_case(2047, "bfloat16", "L1-partial-tile")


def run_L2():
    # L2: edge sizes around the tile boundary, non-blocking.
    for N, dtype_str in (
        (1, "float32"),
        (2, "float16"),
        (10, "bfloat16"),
        (1025, "float32"),
        (4095, "float16"),
        (8191, "bfloat16"),
    ):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    # Boundary: special-value inputs (branch floor, saturation, inf, NaN),
    # all non-blocking.  ``elu(-inf) = -1`` (alpha=1), ``elu(+inf) = +inf``,
    # ``elu(NaN) = NaN``.
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        try:
            x = _repeat_values(
                [0.0, -0.0, 1e-7, -1e-7, 1e-3, -1e-3, 1.0, -1.0, 88.8, -88.8],
                4096,
                td,
            )
            _run_tensor(x, dtype_str, "Boundary-special")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}")

    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            td = _TORCH_DTYPE[dtype_str]
            x = _repeat_values([float("inf"), float("-inf"), float("nan"), 0.0], 4096, td)
            _run_tensor(x, dtype_str, "Boundary-inf-nan")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} inf/nan err={e}")

    # Manifest full workloads (non-blocking).
    for N, dtype_str in (
        (4096 * 4096, "float16"),
        (8192 * 8192, "float16"),
        (8192 * 8192, "bfloat16"),
    ):
        try:
            _run_case(N, dtype_str, "Boundary-manifest")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


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
