# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Lerp (scalar-weight) forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``LerpFwdKernel`` (TileOPs,
``elementwise/arithmetic.py``; the binary ``register_copy`` path with the
construction-time ``weight`` baked into the op body
``a + cast(weight, a.dtype) * (b - a)``) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    out = input + weight * (end - input)          (torch.lerp(input, end, weight))
    broadcast(out.shape) = broadcast_shapes(input.shape, end.shape)
on two pre-broadcast, flattened, contiguous same-shape 1-D ``(N,)``
vectors of dtype float16 | bfloat16 | float32; output ``(N,)`` same dtype.
``weight`` is a construction-time ``float`` scalar (default 0.5) baked into
the kernel (one specialization per distinct value).

Broadcast contract (DESIGN.md section 0.6 R2): the Op layer resolves numpy
broadcasting via ``broadcast_shapes`` + ``expand`` + flatten, so this
kernel only ever sees two same-shape contiguous 1-D vectors (the GPU
``register_copy`` case).  The kernel is broadcast-agnostic.

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` + v-prefix vector
        chain (vsub -> vmul -> vadd) replaces the CUDA threads x npt
        two-level parallel; ``block_size`` folds threads*npt.
    R2: fp32 transit -- fp16/bf16 have no native Vector arithmetic for
        fp32 opmath (bf16 has none at all), so both are upcast with
        ``T.vcast(rint)`` -> fp32 compute -> single ``T.vcast(rint)`` back,
        matching the golden's fp32 opmath rounding path (the same fix the
        sibling lerp_tensor kernel adopted).
    R3: the scalar ``weight`` is baked at trace time (a plain Python float
        operand, mirroring the GPU kernel's construction-time constant).
    R4: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and
        an if-guard for out-of-range tiles (static bounds).

Factory contract (wrapper E1/E2, see
examples/TileOPs/tileops/kernels/elementwise/lerp_scalar/lerp_scalar.py):
    _make_lerp_scalar_kernel(N, dtype, weight) -> kernel(block_size) -> main
    where ``main(a, b)`` returns ``out`` (``out_idx=[2]``).  The two-level
    factory defers JIT compilation to the first ``kernel(block_size)`` call,
    which the wrapper issues inside ``forward`` -- and ``forward`` runs
    under ``Kernel.__call__``'s ``scoped_ascend_mode("Developer")``
    (CG-2026-0010).  Compiling at construction time would instead pick up the
    ambient mode (``Expert`` when the elementwise package is imported) and
    overflow the UB budget for fp32 at ``block_size=8192``.

Run the embedded hierarchical tests:
    python _make_lerp_scalar_kernel.py --level L0
    python _make_lerp_scalar_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual
# sync points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["_make_lerp_scalar_kernel", "golden_lerp_scalar"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Per-dtype max block_size from the UB budget: the fp32-transit dtypes hold
# 3 input-dtype buffers + 2 fp32 buffers (a32/b32); fp32 holds 5 fp32 buffers.
#   fp32:       5 * 4 = 20 B/elem -> 9830  (kept at the family 8192 cap)
#   fp16/bf16:  3 * 2 + 2 * 4 = 14 B/elem -> 14043 (kept at the family cap)
_UB_MAX_BLOCK = {
    "float32": 8192,
    "float16": 8192,
    "bfloat16": 8192,
}

# Default tile sizes (elementwise family DESIGN section 5.2 value).
_DEFAULT_BLOCK = {
    "float32": 8192,
    "float16": 8192,
    "bfloat16": 8192,
}

# Precision tolerances (identical to the sibling lerp_tensor kernel's
# ``_TOLERANCE`` and the source-repo test_lerp_tensor.py from the same family).
_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-2, 1e-2),
    "float32": (1e-6, 1e-6),
}

# Per-(dtype, weight) worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_lerp_scalar(input, end, weight=0.5):
    """Scalar-weight lerp reference: ``out = input + weight * (end - input)``.

    Matches the source-repo test benchmark (examples/TileOPs/tests/
    test_lerp_scalar.py: ``ref = torch.lerp(input, end, weight)``).
    Independent of the NPU algorithm (no vcast / fp32 transit / persistent
    structure).  Runs on CPU and does not require an NPU device.
    """
    a_cpu = input.detach().cpu()
    b_cpu = end.detach().cpu()
    return torch.lerp(a_cpu, b_cpu, weight)


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
def _make_lerp_scalar_kernel(N, dtype, weight=0.5):
    """Build the scalar-weight lerp kernel factory (E1: ``(N, dtype, weight)``).

    Returns a JIT callable taking a single ``block_size`` (E2), which compiles
    and returns ``main``; ``main(a, b)`` returns ``out`` (``out_idx=[2]``).
    """
    if N <= 0:
        # DESIGN.md section 9.3: N == 0 never reaches the kernel (the Op
        # layer intercepts); reject defensively with a clear message.
        raise ValueError(f"N must be a positive int, got N={N}")
    if dtype not in _UB_MAX_BLOCK:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {sorted(_UB_MAX_BLOCK)}")

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; re-queried per factory call so deployments on other
    # devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    max_block = _UB_MAX_BLOCK[dtype]
    default_block = _DEFAULT_BLOCK[dtype]

    # Trace-time (Python) scalar: the NPU binary-op front end takes a raw
    # scalar operand, not a cast expression.
    weight_scalar = float(weight)

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=default_block, num_cores=None):
        # Host-side UB budget guard (DESIGN.md section 9.2).  ValueError is
        # used instead of a bare assert so the guard survives ``python -O``.
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for "
                f"{dtype} (max {max_block} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # Persistent core split (DESIGN.md section 5.5).  All quantities are
        # compile-time constants: N and block_size are Python ints and
        # T.ceildiv/min fold to IntImm at trace time.
        num_logical = T.ceildiv(N, block_size)  # logical tile count
        if num_cores is None or num_cores <= 0 or num_cores > num_logical:
            num_kernels = min(num_logical, vector_cores)
        else:
            num_kernels = num_cores
        num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # UB buffers (DESIGN.md section 4.3).  Allocated
                # unconditionally; a32 doubles as the in-place result chain.
                a_ub = T.alloc_ub((block_size,), dtype)
                b_ub = T.alloc_ub((block_size,), dtype)
                a32 = T.alloc_ub((block_size,), "float32")
                b32 = T.alloc_ub((block_size,), "float32")
                y_ub = T.alloc_ub((block_size,), dtype)

                # Persistent in-core serial task loop (static bound,
                # grid-stride mapping, if-guard masks out-of-range tiles).
                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        offset = block_id * block_size
                        tail_size = T.min(block_size, N - offset)

                        # 1. Load GM -> UB (only valid elements copied in).
                        T.copy(a[offset : offset + tail_size], a_ub[0:tail_size])
                        T.copy(b[offset : offset + tail_size], b_ub[0:tail_size])

                        # 2. Cast to fp32 (fp16/bf16) or copy (fp32).
                        if dtype != "float32":
                            T.vcast(a_ub, a32, round_mode="rint")
                            T.vcast(b_ub, b32, round_mode="rint")
                        else:
                            T.copy(a_ub, a32)
                            T.copy(b_ub, b32)

                        # 3. Compute in fp32: t = b - a; t = weight * t; y = a + t.
                        T.vsub(b32, a32, b32)  # t = b - a
                        T.vmul(b32, weight_scalar, b32)  # t = weight * t
                        T.vadd(a32, b32, a32)  # y = a + t

                        # 4. Cast back to the output dtype (fp16/bf16) or copy.
                        if dtype != "float32":
                            T.vcast(a32, y_ub, round_mode="rint")
                        else:
                            T.copy(a32, y_ub)

                        # 5. Store UB -> GM (only valid elements copied out).
                        T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
def _max_diff_metric(out, ref):
    """Return the max absolute difference between out and ref.

    NaN/Inf aware: positions where both are NaN (or both equal infinities)
    count as equal and contribute zero difference.
    """
    eq = out == ref
    eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    return diff.max().item()


def _record_stat(dtype_str, weight, max_diff):
    if max_diff == max_diff:  # NaN guard
        key = f"{dtype_str}/weight={weight}"
        if max_diff > _STATS_MAX_DIFF.get(key, 0.0):
            _STATS_MAX_DIFF[key] = max_diff


_FACTORY_CACHE = {}


def _get_factory(N, dtype_str, weight):
    """Cache factory instances (mirrors the wrapper's one-factory-per-op
    usage; the jit wrapper then caches compiled mains per block_size)."""
    key = (N, dtype_str, float(weight))
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _make_lerp_scalar_kernel(N, dtype_str, weight)
    return _FACTORY_CACHE[key]


def _run_tensor(a, b, dtype_str, tag, weight=0.5, block_size=None, equal_nan=False):
    """Run kernel on pre-made same-shape 1-D NPU tensors and compare vs golden.

    Raises AssertionError on tolerance violation.  Returns max_diff.
    """
    N = a.numel()
    atol, rtol = _TOLERANCE[dtype_str]
    if block_size is None:
        block_size = _DEFAULT_BLOCK[dtype_str]
    kernel = _get_factory(N, dtype_str, weight)(block_size)
    y_cpu = kernel(a, b).cpu()
    golden = golden_lerp_scalar(a, b, weight)
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    _record_stat(dtype_str, weight, max_diff)
    print(
        f"[{tag}] PASS: N={N} dtype={dtype_str} weight={weight} "
        f"block_size={block_size} max_diff={max_diff:.3e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, weight=0.5, block_size=None, scale=1.0):
    """Run kernel on random data of shape (N,) with the given dtype."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    a = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    b = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(a, b, dtype_str, tag, weight=weight, block_size=block_size)


def _run_broadcast_case(a_shape, b_shape, dtype_str, tag, weight=0.5):
    """Broadcast case: Op layer expands/flattens, kernel sees same-shape 1-D.

    Golden uses torch's native broadcast on the original shapes (independent
    of the NPU-side Op-layer flattening).
    """
    torch_dtype = _TORCH_DTYPE[dtype_str]
    out_shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
    a = torch.randn(a_shape, dtype=torch.float32).to(torch_dtype)
    b = torch.randn(b_shape, dtype=torch.float32).to(torch_dtype)
    golden = golden_lerp_scalar(a, b, weight)
    N = 1
    for d in out_shape:
        N *= d
    a_flat = a.expand(out_shape).contiguous().view(-1).npu()
    b_flat = b.expand(out_shape).contiguous().view(-1).npu()
    kernel = _get_factory(N, dtype_str, weight)(_DEFAULT_BLOCK[dtype_str])
    y_cpu = kernel(a_flat, b_flat).cpu().view(out_shape)
    atol, rtol = _TOLERANCE[dtype_str]
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol)
    max_diff = _max_diff_metric(y_cpu, golden)
    _record_stat(dtype_str, weight, max_diff)
    print(
        f"[{tag}] PASS: A={tuple(a_shape)} B={tuple(b_shape)} -> {out_shape} "
        f"N={N} dtype={dtype_str} weight={weight} max_diff={max_diff:.3e}"
    )
    return max_diff


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
_L0_N = 1048576  # 2^20, shape [1024, 1024] same-shape (DESIGN.md section 8.3)


def run_L0():
    """L0 gate: all dtypes x weight, non-divisible tail, broadcast path."""
    failures = []

    # L0-1: smoke, all dtypes, N=2^20 same-shape, default weight.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _try_case(failures, f"L0-1/{dtype_str}", _run_case, N=_L0_N, dtype_str=dtype_str, tag="L0-1")

    # L0-2: weight sweep (scalar multiply path), all dtypes.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for weight in (0.0, 0.3, 0.7, 1.0, -1.5):
            _try_case(
                failures,
                f"L0-2/{dtype_str}/weight={weight}",
                _run_case,
                N=_L0_N,
                dtype_str=dtype_str,
                tag="L0-2",
                weight=weight,
            )

    # L0-3: non-divisible tail (partial tile + guard).
    for dtype_str in ("float16", "float32"):
        for N in (10000, 4097):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/N={N}",
                _run_case,
                N=N,
                dtype_str=dtype_str,
                tag="L0-3",
                weight=0.3,
            )

    # L0-4: broadcast path (Op layer expand/flatten).
    _try_case(
        failures,
        "L0-4/fp32/[64,128]x[1,128]",
        _run_broadcast_case,
        a_shape=(64, 128),
        b_shape=(1, 128),
        dtype_str="float32",
        tag="L0-4",
    )
    _try_case(
        failures,
        "L0-4/fp16/[2,512,768]x[1,1,768]",
        _run_broadcast_case,
        a_shape=(2, 512, 768),
        b_shape=(1, 1, 768),
        dtype_str="float16",
        tag="L0-4",
        weight=0.5,
    )

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x weight x shape x block_size)."""
    failures = []

    # L1-1: weight sweep (incl. 0.0 and negative) x all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for weight in (0.0, -1.0, 0.5, 3.0):
            _try_case(
                failures,
                f"L1-1/{dtype_str}/weight={weight}",
                _run_case,
                N=1 << 16,
                dtype_str=dtype_str,
                tag="L1-1",
                weight=weight,
            )

    # L1-2: shape / tail coverage (exact multiple, tail==1-ish, odd tail).
    for dtype_str in ("float16", "bfloat16", "float32"):
        for N in (8192, 8191, 1, 100):
            _try_case(
                failures,
                f"L1-2/{dtype_str}/N={N}",
                _run_case,
                N=N,
                dtype_str=dtype_str,
                tag="L1-2",
            )

    # L1-3: non-default block_size (UB budget still holds).
    _try_case(
        failures,
        "L1-3/fp16/bs=2048",
        _run_case,
        N=1 << 20,
        dtype_str="float16",
        tag="L1-3",
        block_size=2048,
    )
    _try_case(
        failures,
        "L1-3/fp32/bs=4096",
        _run_case,
        N=1 << 20,
        dtype_str="float32",
        tag="L1-3",
        block_size=4096,
    )

    # L1-4: broadcast coverage across dtypes/shapes.
    _try_case(
        failures,
        "L1-4/bf16/[1,4096]x[1024,4096]",
        _run_broadcast_case,
        a_shape=(1, 4096),
        b_shape=(1024, 4096),
        dtype_str="bfloat16",
        tag="L1-4",
    )
    _try_case(
        failures,
        "L1-4/fp16/[2,512,768]x[2,512,1]",
        _run_broadcast_case,
        a_shape=(2, 512, 768),
        b_shape=(2, 512, 1),
        dtype_str="float16",
        tag="L1-4",
    )
    _try_case(
        failures,
        "L1-4/fp32/[4,1,8]x[1,8,1]",
        _run_broadcast_case,
        a_shape=(4, 1, 8),
        b_shape=(1, 8, 1),
        dtype_str="float32",
        tag="L1-4",
    )

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    for N, dtype_str in ((1, "float32"), (2, "float16"), (10, "bfloat16"), (1 << 24, "float16")):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking).

    Covers IEEE NaN/+-inf propagation and the weight=0 / weight=1 endpoints.
    """
    N = 4096

    # (a) weight=0 -> out == a; weight=1 -> out == b (within tolerance).
    for dtype_str in ("float32", "float16", "bfloat16"):
        for weight in (0.0, 1.0):
            try:
                _run_case(N, dtype_str, f"Boundary-weight={weight}", weight=weight)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[Boundary] WARN (Record without blocking): weight={weight} "
                    f"dtype={dtype_str} err={e}"
                )

    # (b) +-inf / NaN operands propagate (equal_nan tolerant).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            a = torch.randn(N, dtype=torch.float32)
            b = torch.randn(N, dtype=torch.float32)
            a[0::8] = float("inf")
            a[1::8] = float("-inf")
            a[2::8] = float("nan")
            b[3::8] = float("inf")
            b[4::8] = float("-inf")
            b[5::8] = float("nan")
            a[6::8] = float("inf")
            b[6::8] = float("-inf")
            a[7::8] = float("-inf")
            b[7::8] = float("inf")
            _run_tensor(
                a.to(torch_dtype).npu(),
                b.to(torch_dtype).npu(),
                dtype_str,
                "Boundary-infnan",
                weight=0.5,
                equal_nan=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[Boundary] WARN (Record without blocking): inf-nan "
                f"dtype={dtype_str} err={e}"
            )

    # (c) signed-zero / subnormal magnitudes (weight=0.5).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            a = (torch.randn(N, dtype=torch.float32) * 1e-6).to(torch_dtype)
            b = torch.randn(N, dtype=torch.float32).to(torch_dtype)
            a[0:4] = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=torch_dtype)
            _run_tensor(a.npu(), b.npu(), dtype_str, "Boundary-signedzero", weight=0.5)
        except Exception as e:  # noqa: BLE001
            print(
                f"[Boundary] WARN (Record without blocking): signed-zero "
                f"dtype={dtype_str} err={e}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per (dtype, weight): "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
