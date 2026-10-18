# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GeluTanhAndMul forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``GeluTanhAndMulFwdKernel`` (TileOPs,
``elementwise/activations.py`` + ``FusedGatedKernel._get_effective_op_func``,
``_builders.make_fused_gated_explicit``) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    Input is a SINGLE tensor ``x`` of shape ``[M, 2*N]``.  Split along the
    last dim: ``gate = x[:, :N]``, ``value = x[:, N:]``.  Output ``y`` of
    shape ``[M, N]`` with ``y = gelu_tanh(gate) * value`` where ``gelu_tanh``
    is the tanh approximation (``F.gelu(approximate="tanh")``):
        gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 g^3)))
    dtype float16 | bfloat16 | float32; output same dtype.  The whole chain
    is evaluated in fp32 and rounded back once.

NPU redesign (DESIGN.md sections 0.6 / 3.3 / 5 / 6):
    R1: one-dim ``T.Kernel(num_cores, is_npu=True)`` with ``block_N`` folding
        the CUDA threads x npt two-level parallel; a row-major task order maps
        the 2-D (row, column-block) space to a 1-D sequence.
    R2: the toolchain has NO usable faithful vector tanh -- ``T.vtanh``
        dispatches by ``TILELANG_ASCEND_MODE`` to either a truncated 4-term
        Maclaurin series (default/Expert, valid only for ``|x| <~ 0.5``) or an
        exp identity that returns NaN for ``+inf`` / large ``|x|`` (Developer);
        ``T.tanh`` is a scalar TIR expression with no vector form.  Both are
        therefore DISABLED.  Using the exact identity
        ``0.5 g (1 + tanh(u)) = g * sigma(2u)``, the gelu_tanh factor is a
        single explicit fp32 sigmoid:
            u  = sqrt(2/pi) * (g + 0.044715 g^3)
            ex = exp(-2u)                (vexp, self-saturating: +inf -> 0)
            s  = 1 / (1 + ex)            (vdiv with a vbrc-filled ones buffer)
            gelu_tanh(g) = g * s
        y = gelu_tanh(gate) * value, full fp32 chain with a single T.vcast
        back to the output dtype.
    R3: persistent core split: ``num_cores = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and an
        if-guard for out-of-range tiles (static bounds).
    Gate/value are read as two contiguous column slices of the single
    ``[M, 2N]`` input (no host-side split, no materialization).

Boundary semantics (DESIGN.md section 0.1 / 8.3 / 9.3):
    gelu_tanh(+inf) = +inf, gelu_tanh(-inf) = NaN (source expression
    ``(-inf) * 0``), gelu_tanh(NaN) = NaN, gelu_tanh(-0.0) = -0.0.

Run the embedded hierarchical tests:
    python gelu_tanh_and_mul_fwd_kernel.py --level L0
    python gelu_tanh_and_mul_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual
# sync points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse
import math

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_gelu_tanh_and_mul", "gelu_tanh_and_mul_fwd_kernel"]


# ---------------------------------------------------------------------------
# Constants (DESIGN.md section 3.2 / 4.7)
# ---------------------------------------------------------------------------
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # sqrt(2/pi)
GELU_TANH_COEFF = 0.044715  # tanh-GELU cubic coefficient

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Precision tolerances (DESIGN.md section 8.2, identical to the source-repo
# test_gelu_tanh_and_mul.py ``_get_tolerances``).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-2, 1e-2),
    "bfloat16": (1.6e-2, 1.6e-2),
}

# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_gelu_tanh_and_mul(x):
    """GeluTanhAndMul reference: ``x [M, 2N] -> y [M, N]``.

    ``y = gelu_tanh(gate) * value`` with the tanh approximate GELU:
        gate  = x[:, :N]
        value = x[:, N:]
        inner = sqrt(2/pi) * (gate + 0.044715 * gate^3)
        gelu  = 0.5 * gate * (1 + tanh(inner))
        y     = gelu * value

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_gelu_tanh_and_mul.py: CPU explicit-tanh formula, fp32 opmath).
    Independent of the NPU algorithm (no explicit sigmoid / fp32 transit /
    persistent structure); only torch native tanh and multiply are used.
    Runs on CPU and does not require an NPU device.
    """
    x_cpu = x.detach().cpu()
    half = x_cpu.shape[-1] // 2
    gate = x_cpu[..., :half].float()
    value = x_cpu[..., half:].float()
    inner = SQRT_2_OVER_PI * (gate + GELU_TANH_COEFF * gate ** 3)
    gelu = 0.5 * gate * (1.0 + torch.tanh(inner))
    return (gelu * value).to(x_cpu.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def gelu_tanh_and_mul_fwd_kernel(M, N, dtype, output_dtype=None, block_N=2048, num_cores=None):
    """GeluTanhAndMul forward kernel: ``y = gelu_tanh(x[:, :N]) * x[:, N:]``.

    Compile-time params:
        M, N: input is ``[M, 2*N]``; output is ``[M, N]``.
        dtype: input dtype, "float16" | "bfloat16" | "float32".
        output_dtype: optional output dtype (defaults to ``dtype``).
        block_N: UB tile width along the last dim (DESIGN.md section 5.2).
        num_cores: optional explicit kernel-count override (else section 5.5).

    The returned ``main(x, y)`` is invoked as ``kernel(x)``; ``y`` is
    auto-allocated (``out_idx=[1]``).
    """
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    n_num = T.ceildiv(N, block_N)  # column blocks per row (compile-time)
    num_logical = M * n_num  # logical tiles = M * ceil(N / block_N)
    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op; re-queried per build so other devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores >= num_logical:
        num_cores = min(num_logical, vector_cores)
    tiles_per_core = T.ceildiv(num_logical, num_cores)  # static bound

    # fp32 scalar operands (DESIGN.md section 3.2/3.3).
    sqrt_2_over_pi = T.cast(SQRT_2_OVER_PI, "float32")
    neg2_sqrt_2_over_pi = T.cast(-2.0 * SQRT_2_OVER_PI, "float32")  # perf_opt: fold -2*sqrt(2/pi)
    coeff = T.cast(GELU_TANH_COEFF, "float32")

    @T.prim_func
    def main(
        x: T.Tensor((M, 2 * N), dtype),
        y: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            # UB buffers (DESIGN.md section 4.3).
            gate_ub = T.alloc_ub((block_N,), dtype)
            value_ub = T.alloc_ub((block_N,), dtype)
            g32 = T.alloc_ub((block_N,), compute_dtype)
            v32 = T.alloc_ub((block_N,), compute_dtype)
            g2 = T.alloc_ub((block_N,), compute_dtype)  # g^2
            g3 = T.alloc_ub((block_N,), compute_dtype)  # g^3
            cg3 = T.alloc_ub((block_N,), compute_dtype)  # c * g^3
            u32 = T.alloc_ub((block_N,), compute_dtype)  # u(g)
            vi32 = T.alloc_ub((block_N,), compute_dtype)  # -2u
            ex32 = T.alloc_ub((block_N,), compute_dtype)  # exp(-2u)
            den32 = T.alloc_ub((block_N,), compute_dtype)  # 1 + exp(-2u)
            one32 = T.alloc_ub((block_N,), compute_dtype)  # constant 1
            s32 = T.alloc_ub((block_N,), compute_dtype)  # sigmoid / gelu / result
            out_ub = T.alloc_ub((block_N,), out_dtype)

            # OP2: the constant-ones numerator buffer is loop-invariant; fill it
            # once per core (not once per tile) and reuse across all tiles.
            T.vbrc(T.cast(1.0, "float32"), one32)

            # Persistent in-core serial task loop (static bound, row-major
            # grid-stride mapping, if-guard masks out-of-range tiles).
            for i in T.serial(tiles_per_core):
                task = i * num_cores + cid
                if task < M * T.ceildiv(N, block_N):
                    m = task // T.ceildiv(N, block_N)  # row index
                    n0 = (task % T.ceildiv(N, block_N)) * block_N  # column start
                    tail = T.min(block_N, N - n0)  # tail handling (section 6.4)

                    # 1. Load the two contiguous column slices of the single
                    #    [M, 2N] input: gate at n0, value at N + n0.
                    T.copy(x[m, n0 : n0 + tail], gate_ub[0:tail])
                    T.copy(x[m, N + n0 : N + n0 + tail], value_ub[0:tail])

                    # 2. Upcast to fp32 (fp16/bf16) or same-dtype copy (fp32).
                    if dtype != compute_dtype:
                        T.vcast(gate_ub, g32, round_mode="rint")
                        T.vcast(value_ub, v32, round_mode="rint")
                    else:
                        T.copy(gate_ub, g32)
                        T.copy(value_ub, v32)

                    # 3. u(g) = sqrt(2/pi) * (g + 0.044715 g^3)   (fp32).
                    T.vmul(g32, g32, g2)                 # g^2
                    T.vmul(g2, g32, g3)                  # g^3
                    T.vmul(g3, coeff, cg3)               # c g^3
                    T.vadd(g32, cg3, u32)                # g + c g^3
                    # perf_opt: fold the sqrt(2/pi) scaling and the *(-2) into
                    # one multiply (was 2 vmuls) -- identical math.
                    T.vmul(u32, neg2_sqrt_2_over_pi, vi32)  # -2u = -2*sqrt(2/pi)*(g+c g^3)

                    # 4. Explicit sigmoid: gelu_tanh(g) = g * sigma(2u).
                    #    sigma(2u) = 1 / (1 + exp(-2u)).
                    T.vexp(vi32, ex32)                   # exp(-2u)
                    T.vadd(ex32, 1.0, den32)             # 1 + exp(-2u)
                    T.vdiv(one32, den32, s32)            # sigma(2u) = 1/den (ones hoisted)
                    T.vmul(g32, s32, s32)                # gelu_tanh(g) = g * sigma(2u)

                    # 5. Gated multiply: y = gelu_tanh(gate) * value.
                    T.vmul(s32, v32, s32)

                    # 6. Cast back to output dtype (fp16/bf16) or copy (fp32).
                    if out_dtype != compute_dtype:
                        T.vcast(s32, out_ub, round_mode="rint")
                    else:
                        T.copy(s32, out_ub)

                    # 7. Store UB -> GM (only valid elements copied out).
                    T.copy(out_ub[0:tail], y[m, n0 : n0 + tail])

    return main


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
def _max_diff_metric(out, ref):
    """Return the max absolute difference between out and ref.

    NaN/Inf aware: positions where both are NaN count as equal and contribute
    zero difference.
    """
    eq = out == ref
    eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    return diff.max().item()


def _record_stat(dtype_str, max_diff):
    if max_diff == max_diff:  # NaN guard
        if max_diff > _STATS_MAX_DIFF.get(dtype_str, 0.0):
            _STATS_MAX_DIFF[dtype_str] = max_diff


def _run_tensor(x, dtype_str, tag, block_size=None, equal_nan=False):
    """Run kernel on a pre-made ``[M, 2N]`` NPU tensor and compare vs golden.

    Raises AssertionError on tolerance violation.  Returns max_diff.
    """
    M, two_N = x.shape
    N = two_N // 2
    atol, rtol = _TOLERANCE[dtype_str]
    kwargs = {}
    if block_size is not None:
        kwargs["block_N"] = block_size
    kernel = gelu_tanh_and_mul_fwd_kernel(M, N, dtype_str, **kwargs)
    y_cpu = kernel(x).cpu()
    golden = golden_gelu_tanh_and_mul(x)
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    _record_stat(dtype_str, max_diff)
    print(
        f"[{tag}] PASS: M={M} N={N} dtype={dtype_str} "
        f"block_N={block_size} max_diff={max_diff:.3e}"
    )
    return max_diff


def _run_case(M, N, dtype_str, tag, block_size=None, scale=1.0):
    """Run kernel on random ``[M, 2N]`` data (gate/value independent randn)."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(M, 2 * N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(x, dtype_str, tag, block_size=block_size)


def _run_gate_case(M, N, dtype_str, tag, gate_scale=1.0):
    """Run with a large-magnitude gate half and a normal (scale-1) value half.

    Exercises the full tanh-sigmoid range (``|u|`` well past 0.5, the
    ``T.vtanh`` blind spot, and past the fp32 exp overflow point for the
    Developer tanh path) while keeping the gated multiply at the spec's O(1)
    value scale.
    """
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = torch.randn(M, 2 * N, dtype=torch.float32)
    x[:, :N] *= gate_scale
    return _run_tensor(x.to(torch_dtype).npu(), dtype_str, tag)


def _try_case(failures, case_id, fn, **kwargs):
    """Run a case, collecting (not re-raising) per-case failures."""
    try:
        fn(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


# ---------------------------------------------------------------------------
# Layer suites
# ---------------------------------------------------------------------------
def run_L0():
    """L0 gate: manifest workloads + smoke + non-divisible tail, all dtypes.

    DESIGN.md section 8.3: M=2048/N=14336 (prefill) fp16/bf16, M=1/N=14336
    (decode), smoke (16,128), plus fp32 and non-divisible N.  Returns the
    list of (case_id, exception) failures.
    """
    failures = []

    # L0-1: manifest prefill workload (M=2048, N=14336), narrow dtypes.
    for dtype_str in ("float16", "bfloat16"):
        _try_case(failures, f"L0-1/{dtype_str}", _run_case,
                  M=2048, N=14336, dtype_str=dtype_str, tag="L0-1")

    # L0-2: fp32 prefill workload.
    _try_case(failures, "L0-2/float32", _run_case,
              M=2048, N=14336, dtype_str="float32", tag="L0-2")

    # L0-3: decode workload (M=1, N=14336), all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-3/{dtype_str}", _run_case,
                  M=1, N=14336, dtype_str=dtype_str, tag="L0-3")

    # L0-4: smoke (16, 128), all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-4/{dtype_str}", _run_case,
                  M=16, N=128, dtype_str=dtype_str, tag="L0-4")

    # L0-5: non-divisible tail (N % block_N != 0), all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-5/{dtype_str}/N=1000", _run_case,
                  M=64, N=1000, dtype_str=dtype_str, tag="L0-5")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_N tail paths).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L1-1: qwen3 workload (32, 12288) + odd tails across all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L1-1/{dtype_str}/qwen3", _run_case,
                  M=32, N=12288, dtype_str=dtype_str, tag="L1-1")
        for N in (4097, 2050, 2047):
            _try_case(failures, f"L1-1/{dtype_str}/N={N}", _run_case,
                      M=8, N=N, dtype_str=dtype_str, tag="L1-1")

    # L1-2: non-divisible N with block_N smaller than N.
    for dtype_str in ("float16", "float32"):
        for N in (5000, 100):
            _try_case(failures, f"L1-2/{dtype_str}/N={N}", _run_case,
                      M=4, N=N, dtype_str=dtype_str, tag="L1-2", block_size=1024)

    # L1-3: small-M large-N decode variants.
    _try_case(failures, "L1-3/fp16/1x4096", _run_case,
              M=1, N=4096, dtype_str="float16", tag="L1-3")
    _try_case(failures, "L1-3/bf16/2x8192", _run_case,
              M=2, N=8192, dtype_str="bfloat16", tag="L1-3")

    # L1-4: large-|gate| full-range coverage (T.vtanh blind spot: |u| beyond
    #       0.5 and past the Developer exp-overflow point).  The value half
    #       stays at scale 1 so the sigmoid-gelu body is what is exercised.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L1-4/{dtype_str}/large", _run_gate_case,
                  M=64, N=2048, dtype_str=dtype_str, tag="L1-4", gate_scale=3.0)
        _try_case(failures, f"L1-4/{dtype_str}/sat", _run_gate_case,
                  M=32, N=2048, dtype_str=dtype_str, tag="L1-4", gate_scale=20.0)

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes + numerical stress (warn only, non-blocking)."""
    for M, N, dtype_str in ((1, 1, "float32"), (1, 2, "float16"), (3, 7, "bfloat16"),
                            (4, 4096, "float16")):
        try:
            _run_case(M, N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): M={M} N={N} dtype={dtype_str} err={e}")

    # Numerical stress: a large BOTH halves (gate *and* value of order 20).
    # Out of the spec's randn regime; recorded, not blocking.
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            _run_case(32, 2048, dtype_str, "L2-stress", scale=20.0)
        except Exception as e:  # noqa: BLE001
            print(f"[L2-stress] WARN (Record without blocking): dtype={dtype_str} err={e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking).

    DESIGN.md section 8.3 / 9.3: tanh-gelu saturation semantics.  Frozen
    boundary: ``gelu_tanh(+inf) = +inf`` and ``gelu_tanh(-inf) = NaN``
    (source expression ``(-inf) * 0``), matching the golden.  NaN input ->
    NaN output; ``-0.0`` -> ``-0.0``.
    """
    M, N = 8, 4096

    # NaN / +-inf gate and value, all dtypes, vs golden with equal_nan.
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(M, 2 * N, dtype=torch.float32)
            x[:, N] = 1.0                # default value = 1 for gate probes
            x[0, 0] = float("inf")       # gate +inf, value +1 -> +inf
            x[1, 0] = float("-inf")      # gate -inf -> -inf * 0 = NaN
            x[2, 0] = float("nan")       # gate NaN -> NaN
            x[3, 0] = 1.0                # gate 1, value +inf -> +inf
            x[3, N] = float("inf")
            x[4, 0] = 1.0                # gate 1, value -inf -> -inf
            x[4, N] = float("-inf")
            x[5, 0] = 1.0                # gate 1, value NaN -> NaN
            x[5, N] = float("nan")
            x[6, 0] = 0.0                # gate 0 -> 0
            x[7, 0] = 1e4                # large positive gate -> saturates to 1
            xt = x.to(torch_dtype).npu()
            kernel = gelu_tanh_and_mul_fwd_kernel(M, N, dtype_str)
            y_cpu = kernel(xt).cpu()
            golden = golden_gelu_tanh_and_mul(xt)
            torch.testing.assert_close(
                y_cpu, golden,
                atol=_TOLERANCE[dtype_str][0],
                rtol=_TOLERANCE[dtype_str][1],
                equal_nan=True,
            )
            # Explicit frozen-semantics probes.
            assert bool(torch.isnan(y_cpu[1, 0])), "gelu_tanh(-inf) must be NaN"
            assert bool(torch.isinf(y_cpu[0, 0]) and y_cpu[0, 0] > 0), "gelu_tanh(+inf) must be +inf"
            assert bool(torch.isnan(y_cpu[2, 0])), "gelu_tanh(NaN) must be NaN"
            assert bool(torch.isnan(y_cpu[5, 0])), "value NaN must propagate to NaN"
            _record_stat(dtype_str, _max_diff_metric(y_cpu, golden))
            print(f"[Boundary] PASS: inf/nan dtype={dtype_str}")
        except AssertionError as e:
            print(f"[Boundary] FAIL: inf/nan dtype={dtype_str} err={e}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): inf/nan dtype={dtype_str} err={e}")

    # tanh-gelu saturation: large positive gate => gelu_tanh ~ gate, large
    # negative gate => gelu_tanh ~ -0.0.
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(M, 2 * N, dtype=torch.float32)
            x[:, 0] = 50.0
            x[:, N] = 1.0
            xt = x.to(torch_dtype).npu()
            y_cpu = gelu_tanh_and_mul_fwd_kernel(M, N, dtype_str)(xt).cpu()
            g_pos = golden_gelu_tanh_and_mul(xt)
            torch.testing.assert_close(
                y_cpu, g_pos,
                atol=_TOLERANCE[dtype_str][0], rtol=_TOLERANCE[dtype_str][1],
            )
            x[:, 0] = -50.0
            xt = x.to(torch_dtype).npu()
            y_cpu = gelu_tanh_and_mul_fwd_kernel(M, N, dtype_str)(xt).cpu()
            g_neg = golden_gelu_tanh_and_mul(xt)
            torch.testing.assert_close(
                y_cpu, g_neg,
                atol=_TOLERANCE[dtype_str][0], rtol=_TOLERANCE[dtype_str][1],
            )
            _record_stat(dtype_str, max(_max_diff_metric(y_cpu, g_neg), 0.0))
            print(f"[Boundary] PASS: tanh-saturation dtype={dtype_str}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): saturation dtype={dtype_str} err={e}")

    # -0.0 / subnormal magnitudes (gate near zero, value finite).
    for dtype_str in ("float16", "float32"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = (torch.randn(M, 2 * N, dtype=torch.float32) * 1e-6).to(torch_dtype)
            x[0, 0] = 0.0
            x[1, 0] = -0.0
            _run_tensor(x.npu(), dtype_str, "Boundary-signedzero")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): signed-zero dtype={dtype_str} err={e}")

    # Large negative finite gate: gelu_tanh saturates toward -0.0.
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(M, 2 * N, dtype=torch.float32)
            x[:, 0] = -20.0
            _run_tensor(x.to(torch_dtype).npu(), dtype_str, "Boundary-large-negative")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): large-negative dtype={dtype_str} err={e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []
    failures += run_L0()

    if args.level == "all":
        failures += run_L1()
        run_L2()
        run_boundary()

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
