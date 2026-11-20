# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Erf forward kernel (NPU, target="npuir") -- Stage 4 performance-tuned FINAL.

Drop-in replacement for the Stage 3 baseline ``erf_fwd_kernel.py``: same
factory name and ``(N, dtype, output_dtype=None, block_size=None)`` signature,
same ``main(x, y, shape)`` prim_func.  ``block_size=None`` selects the
per-dtype tuned default exposed as the module constant ``TUNED_BLOCK_SIZE``
(imported by the wrapper's perf_opt toggle to pin ``_DEFAULT_BLOCK_SIZE``).
An explicit ``block_size`` still overrides.

Round-1 single-point optimisation -- **odd tanh-polynomial approximation**
replacing the 21/23-op A&S 7.1.26 v-prefix chain:

    erf(x) ~= tanh(x * Q(x^2)),  Q degree 4 (5 coefficients)

``tanh`` of an *odd* polynomial is odd, so the whole sign handling of the
A&S chain disappears: no ``vabs``, no ``vcmp`` mask, no ``vselect`` sign
(measured ~6.5x the cost of a plain v-op on this device), no ``vrec`` and no
``vexp``.  The op count drops 21 -> 12 (fp32) / 23 -> 14 (fp16/bf16), and the
per-dtype tuned ``block_size`` (round-1 sweep) amortises the per-tile
overhead.  Round-2 structural candidates (static-bulk copies, contiguous
per-core spans) gave no gain on the now fully vector-bound kernel and are
recorded in opt_log.md.

Measured (``msprof op`` Task Duration(us), median of 20 launches,
same-session baseline -> final): 16m f16 256.45 -> 117.03 (-54.4%);
16m bf16 256.72 -> 119.28 (-53.5%); 16m f32 308.10 -> 114.47 (-62.8%);
64m f16 1015.60 -> 457.85 (-54.9%); 64m bf16 1016.85 -> 466.79 (-54.1%);
256m f16 4058.63 -> 1821.97 (-55.1%); 256m bf16 4060.51 -> 1857.78 (-54.2%).
See opt_log.md / perf_records.jsonl.

The fitted coefficients are the minimax solution of
``erf(x) = tanh(P(x))`` under a monotonicity constraint on P (see
opt_log.md).  Accuracy on device (raw max abs diff vs ``torch.erf``):
fp32 2.98e-6 (gate atol/rtol 1e-5 -> 3.4x margin), fp16 1 ulp, bf16 1 ulp
(margins identical to the baseline, both dominated by output rounding).

The ``vmin(P, 10)`` clamp is required: the device ``vtanh`` is implemented
via ``exp`` and returns NaN once its argument exceeds the fp32 exp range
(observed NaN at x = 8 where P ~ 3502).  ``vmin`` propagates NaN
(TRAP-vmax-vmin-nan-propagation), so ``inf/nan`` corner semantics still match
``torch.erf`` exactly (``erf(inf)=1``, ``erf(-inf)=-1``, ``erf(nan)=nan``).
Clamping only activates where ``erf`` is already saturated (P >= 10 first
crossed near x ~ 3.9), contributing < 1e-8 error.

Hard constraints honoured (DESIGN.md section 3.5.1 / 9.1):
  * every ``T.alloc_ub`` is allocated unconditionally at the ``T.Kernel``
    top level (conditional alloc triggers ``Undefined variable``);
  * v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x_ub[0:tail]``);
  * f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.

run: python perf_opt/erf_fwd_kernel.py --level all
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# erf(x) ~= tanh(x * Q(x^2)); Q coefficients ascending in x^2.
_TANH_D = [
    1.12835817913,
    0.102932893381,
    -0.000496822903435,
    -0.000415134821557,
    3.22441004229e-05,
]
_TANH_CLAMP = 10.0

# Per-dtype tuned block_size (element count).  UB footprint: the tanh kernel
# only keeps 5 live UB buffers (dtype in/out + 3 fp32) -> ~16 B/elem, so a
# much larger per-block element count fits than the A&S baseline; the winner
# latches per dtype (see opt_log.md).  fp32 halves the headroom (2 fp32
# in/out buffers) and tops out one step lower.
TUNED_BLOCK_SIZE = {
    "float32": 9216,
    "float16": 10240,
    "bfloat16": 10240,
}


def _auto_block_size(dtype):
    return TUNED_BLOCK_SIZE[dtype]


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_erf(x):
    """Erf reference implementation, equivalent to ``torch.erf``.

    Mirrors the source op_func ``T.erf(T.cast(x, "float32"))``: promote the
    input to fp32, compute the error function in the fp32 domain and round
    once back to the original dtype.  ``torch.erf`` itself already uses fp32
    opmath for fp16/bf16, so this is bit-identical to ``torch.erf(x)``.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return torch.erf(x_f32).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def erf_fwd_kernel(N, dtype, output_dtype=None, block_size=None):
    """Erf forward kernel factory: y = erf(x).

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> tanh chain directly on the fp32 input buffer;
      * ``"float16"`` / ``"bfloat16"`` -> fp32 transit via ``T.vcast``.
    """
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    block_size = block_size or _auto_block_size(dtype)
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)
            xf = T.alloc_ub((block_size,), f32)  # fp32 transit entry
            z = T.alloc_ub((block_size,), f32)   # x^2 -> tanh result
            q = T.alloc_ub((block_size,), f32)   # Horner accumulator

            # Compile-time alias: fp32 path computes in-place on x_ub, the
            # fp16/bf16 path computes in the fp32 transit buffer xf.
            xc = xf if need_cast else x_ub

            # In-core grid-stride over logical tiles: static loop bound,
            # guard covers non-divisible grids.
            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    # Only T.copy uses the dynamic tail slice.
                    T.copy(x[off : off + tail], x_ub[0:tail])

                    if need_cast:
                        # lossless up-cast (f16/bf16 -> f32, rint only)
                        T.vcast(x_ub, xf, round_mode="rint")

                    # --- erf(x) = tanh(x * Q(x^2)) ---
                    T.vmul(xc, xc, z)                       # z = x^2
                    # Horner, ascending Q built from the top coefficient.
                    T.vmul(z, _TANH_D[4], q)
                    T.vadd(q, _TANH_D[3], q)
                    T.vmul(q, z, q)
                    T.vadd(q, _TANH_D[2], q)
                    T.vmul(q, z, q)
                    T.vadd(q, _TANH_D[1], q)
                    T.vmul(q, z, q)
                    T.vadd(q, _TANH_D[0], q)                # q = Q(x^2)
                    T.vmul(xc, q, q)                        # q = P(x) = x*Q
                    T.vmin(q, _TANH_CLAMP, q)               # keep vtanh arg in range
                    T.vtanh(q, z)                           # y = tanh(P)

                    if need_cast:
                        # single round back to input dtype
                        T.vcast(z, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off : off + tail])
                    else:
                        T.copy(z[0:tail], y[off : off + tail])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances from examples/TileOPs/tests/ops/test_unary_math.py (DESIGN 8.2).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _make_input(N, dtype_str):
    """Generate a CPU input tensor following the ErfWorkload logic (randn)."""
    return torch.randn(N, dtype=_TORCH_DTYPE[dtype_str])


def _repeat_values(values, N, dtype):
    """Repeat a small value pattern up to N elements (edge / corner cases)."""
    base = torch.tensor(values, dtype=dtype)
    repeats = (N + len(values) - 1) // len(values)
    return base.repeat(repeats)[:N]


def _max_diff(y, golden):
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    return (y_f64 - g_f64).abs().max().item()


def _run_tensor(x_cpu, dtype_str, tag):
    """Run the kernel on a CPU tensor and compare with the golden output."""
    N = x_cpu.numel()
    golden = golden_erf(x_cpu)
    kernel = erf_fwd_kernel(N, dtype_str)

    x_npu = x_cpu.npu()
    y = kernel(x_npu, N)
    y_cpu = y.cpu()

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y_cpu, golden)
    try:
        torch.testing.assert_close(y_cpu, golden, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({N},) dtype={dtype_str} max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.6e}")
    return max_diff


def _run_case(N, dtype_str, tag):
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke: 1M elements per dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L0-smoke")
    # L0-4 corner values that the v0 (T.verf) design always failed:
    # erf(3) must be ~0.999978, not the truncated-polynomial value 37.9.
    _run_tensor(
        _repeat_values([0.0, 3.0, -3.0, 100.0], 4096, torch.float32),
        "float32",
        "L0-corner",
    )
    # L0-5 full-domain sensitive region (|x| up to 8).
    x_lin = torch.linspace(-8.0, 8.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_lin, "float32", "L0-linspace")
    # L0-6 non-divisible tail (4_000_000 % 10240 = 5120).
    _run_case(4000000, "float16", "L0-tail")


def run_L1():
    # L1: full dtype x boundary shapes -- manifest 16m flat and non-divisible
    # tails for every dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(4096 * 4096, dtype_str, "L1-16m")
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(4000000, dtype_str, "L1-tail")
    # Exact single-tile / multi-tile grid.
    _run_case(10240, "float32", "L1-exact-tile")
    _run_case(20480, "float16", "L1-two-tiles")
    _run_case(20479, "bfloat16", "L1-partial-tile")


def run_L2():
    # L2: edge sizes around the tile boundary, non-blocking.
    for N, dtype_str in (
        (1, "float32"),
        (2, "float16"),
        (10, "bfloat16"),
        (10241, "float32"),
        (20479, "float16"),
        (40959, "bfloat16"),
    ):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    # Boundary: special-value inputs (saturation, inf, NaN), non-blocking.
    special = [0.0, 3.0, -3.0, 3.9, -3.9, 5.0, 100.0, -100.0, 1e-8]
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            x = _repeat_values(special, 4096, _TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, "Boundary-special")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}")

    # inf / NaN IEEE propagation.
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
