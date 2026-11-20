# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Add (scalar-alpha) forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``AddFwdKernel`` (TileOPs,
``elementwise/arithmetic.py`` -> ``AlphaScaledBinaryKernel`` register_copy
path, ``_add_fwd_kernels.py``) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = input + alpha * other          (torch.add(input, other, *, alpha=1))
    broadcast(y.shape) = broadcast_shapes(input, other)
on two pre-broadcast, flattened, contiguous same-shape 1-D ``(N,)``
vectors of dtype float16 | bfloat16 | float32; output ``(N,)`` same dtype.
``alpha`` is a construction-time ``int | float`` scalar (default 1) baked
into the kernel (one specialization per distinct value).

Broadcast contract (DESIGN.md section 0.6 R2): the Op layer resolves numpy
broadcasting via ``broadcast_shapes`` + ``expand`` + flatten, so this
kernel only ever sees two same-shape contiguous 1-D vectors (the GPU
``register_copy`` case).  The kernel is broadcast-agnostic.

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` + v-prefix vector
        chain (vcast -> vmul -> vadd -> vcast) replaces the CUDA
        threads x npt two-level parallel; ``block_size`` folds threads*npt.
    R2: fp32 transit -- fp16/bf16 have no native Vector arithmetic for
        fp32 opmath (bf16 has none at all), so both are upcast with
        ``T.vcast(rint)`` -> fp32 compute -> single ``T.vcast(rint)`` back,
        matching the golden's fp32 opmath rounding path.
    R3: ``alpha == 1`` identity fold at trace time (``1*b == b`` exactly for
        every representable value incl. NaN/+-inf/-0.0); ``alpha == 0`` is
        NOT folded (``0*inf = NaN``).
    R4: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and
        an if-guard for out-of-range tiles (static bounds).

Run the embedded hierarchical tests:
    python add_fwd_kernel.py --level L0
    python add_fwd_kernel.py --level all
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

__all__ = ["golden_add", "add_fwd_kernel"]


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

# Precision tolerances (DESIGN.md section 8.2, identical to the source-repo
# test_add.py ``_get_tolerances``).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}

# Per-(dtype, alpha) worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_add(input, other, alpha=1):
    """Add reference: ``y = input + alpha * other``, torch broadcast + fp32 opmath.

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_add.py: ``ref = torch.add(a.float(), b.float(), alpha=self.alpha)
    .to(a.dtype)``).  Independent of the NPU algorithm (no vcast / vmul /
    vadd fp32 chain / persistent structure / Op-layer flattening):
    broadcasting is handled natively by ``torch.add``.  Runs on CPU and does
    not require an NPU device.
    """
    a_cpu = input.detach().cpu()
    b_cpu = other.detach().cpu()
    y_f32 = torch.add(a_cpu.float(), b_cpu.float(), alpha=alpha)
    return y_f32.to(a_cpu.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[2], target="npuir")
def add_fwd_kernel(N, dtype, alpha=1, output_dtype=None, block_size=8192, num_cores=None):
    """Add (scalar-alpha) forward kernel: ``y = a + alpha * b``.

    Compile-time params:
        N: flattened element count (each input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "float16" | "bfloat16" | "float32".
        alpha: multiplier on ``b`` (int | float, default 1); baked in.
        output_dtype: optional output dtype (defaults to ``dtype``).
        block_size: UB tile width (folds GPU threads*npt; 8192 per section 5.2).
        num_cores: optional explicit kernel count override (else section 5.5).

    The returned ``main(a, b, y, shape)`` is invoked as ``kernel(a, b, N)``;
    ``y`` is auto-allocated (``out_idx=[2]``).
    """
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    # R1-A3: split the task space into static full tiles + one dynamic tail
    # tile.  The hot loop then has static extents (no per-tile T.min), which
    # lets the compiler keep the copy lengths compile-time and makes the loop
    # body pipeline-friendly.
    num_full = N // block_size           # static full tiles
    tail = N - num_full * block_size     # 0 <= tail < block_size
    num_logical = num_full + (1 if tail else 0)
    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op; re-queried per build so other devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)  # static bound (full tiles)

    # Trace-time (Python) branches: alpha == 1 identity fold; the scalar is a
    # plain Python float (the NPU binary-op front end takes a raw scalar
    # operand, not a cast expression).
    fold = alpha == 1
    alpha_scalar = float(alpha)

    @T.prim_func
    def main(
        a: T.Tensor((N,), dtype),
        b: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # UB buffers (DESIGN.md section 4.3).  All allocated
            # unconditionally; a32 doubles as the in-place result chain.
            a_ub = T.alloc_ub((block_size,), dtype)
            b_ub = T.alloc_ub((block_size,), dtype)
            a32 = T.alloc_ub((block_size,), compute_dtype)
            b32 = T.alloc_ub((block_size,), compute_dtype)
            # NOTE: an explicit y_ub is kept (R1-A2, aliasing y_ub=a_ub, was
            # rejected: the auto-multi-buffer pass versioned a_ub but not the
            # alias, breaking every L0 case).
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # Persistent full-tile loop: static extents, grid-stride mapping,
            # if-guard masks out-of-range tiles.  Only executed when N has at
            # least one full tile.
            if num_full > 0:
                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size

                        # 1. Load GM -> UB (full tile, static extent).
                        T.copy(a[offset : offset + block_size], a_ub)
                        T.copy(b[offset : offset + block_size], b_ub)

                        # 2. Cast to fp32 (fp16/bf16) or copy (fp32).
                        if dtype != compute_dtype:
                            T.vcast(a_ub, a32, round_mode="rint")
                            T.vcast(b_ub, b32, round_mode="rint")
                        else:
                            T.copy(a_ub, a32)
                            T.copy(b_ub, b32)

                        # 3. Compute in fp32: fold alpha==1, else scaled add.
                        if fold:
                            T.vadd(a32, b32, a32)  # y = a + b (identity fold)
                        else:
                            T.vmul(b32, alpha_scalar, b32)  # t = alpha * b
                            T.vadd(a32, b32, a32)           # y = a + t

                        # 4. Cast back to output dtype (fp16/bf16) or copy (fp32).
                        if out_dtype != compute_dtype:
                            T.vcast(a32, y_ub, round_mode="rint")
                        else:
                            T.copy(a32, y_ub)

                        # 5. Store UB -> GM (full tile, static extent).
                        T.copy(y_ub, y[offset : offset + block_size])

            # Tail epilogue: at most one partial tile, owned by core 0.
            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(a[offset : offset + tail], a_ub[0:tail])
                    T.copy(b[offset : offset + tail], b_ub[0:tail])
                    if dtype != compute_dtype:
                        T.vcast(a_ub, a32, round_mode="rint")
                        T.vcast(b_ub, b32, round_mode="rint")
                    else:
                        T.copy(a_ub, a32)
                        T.copy(b_ub, b32)
                    if fold:
                        T.vadd(a32, b32, a32)
                    else:
                        T.vmul(b32, alpha_scalar, b32)
                        T.vadd(a32, b32, a32)
                    if out_dtype != compute_dtype:
                        T.vcast(a32, y_ub, round_mode="rint")
                    else:
                        T.copy(a32, y_ub)
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


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


def _record_stat(dtype_str, alpha, max_diff):
    if max_diff == max_diff:  # NaN guard
        key = f"{dtype_str}/alpha={alpha}"
        if max_diff > _STATS_MAX_DIFF.get(key, 0.0):
            _STATS_MAX_DIFF[key] = max_diff


def _run_tensor(a, b, dtype_str, tag, alpha=1, block_size=None, equal_nan=False):
    """Run kernel on pre-made same-shape 1-D NPU tensors and compare vs golden.

    Raises AssertionError on tolerance violation.  Returns max_diff.
    """
    N = a.numel()
    atol, rtol = _TOLERANCE[dtype_str]
    kwargs = {}
    if block_size is not None:
        kwargs["block_size"] = block_size
    kernel = add_fwd_kernel(N, dtype_str, alpha=alpha, **kwargs)
    y_cpu = kernel(a, b, N).cpu()
    golden = golden_add(a, b, alpha)
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    _record_stat(dtype_str, alpha, max_diff)
    print(
        f"[{tag}] PASS: N={N} dtype={dtype_str} alpha={alpha} "
        f"block_size={block_size} max_diff={max_diff:.3e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, alpha=1, block_size=None, scale=1.0):
    """Run kernel on random data of shape (N,) with the given dtype."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    a = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    b = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(a, b, dtype_str, tag, alpha=alpha, block_size=block_size)


def _run_broadcast_case(a_shape, b_shape, dtype_str, tag, alpha=1):
    """Broadcast case: Op layer expands/flattens, kernel sees same-shape 1-D.

    Golden uses torch's native broadcast on the original shapes (independent
    of the NPU-side Op-layer flattening).
    """
    torch_dtype = _TORCH_DTYPE[dtype_str]
    out_shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
    a = torch.randn(a_shape, dtype=torch.float32).to(torch_dtype)
    b = torch.randn(b_shape, dtype=torch.float32).to(torch_dtype)
    golden = golden_add(a, b, alpha)
    N = 1
    for d in out_shape:
        N *= d
    a_flat = a.expand(out_shape).contiguous().view(-1).npu()
    b_flat = b.expand(out_shape).contiguous().view(-1).npu()
    kernel = add_fwd_kernel(N, dtype_str, alpha=alpha)
    y_cpu = kernel(a_flat, b_flat, N).cpu().view(out_shape)
    atol, rtol = _TOLERANCE[dtype_str]
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol)
    max_diff = _max_diff_metric(y_cpu, golden)
    _record_stat(dtype_str, alpha, max_diff)
    print(
        f"[{tag}] PASS: A={tuple(a_shape)} B={tuple(b_shape)} -> {out_shape} "
        f"N={N} dtype={dtype_str} alpha={alpha} max_diff={max_diff:.3e}"
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
    """L0 gate: all dtypes x alpha, non-divisible tail, broadcast path.

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L0-1: smoke, all dtypes, N=2^20 same-shape, default alpha.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _try_case(failures, f"L0-1/{dtype_str}", _run_case, N=_L0_N, dtype_str=dtype_str, tag="L0-1")

    # L0-2: nonzero alpha (scalar multiply path), all dtypes.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for alpha in (2.0, -1.5):
            _try_case(
                failures,
                f"L0-2/{dtype_str}/alpha={alpha}",
                _run_case,
                N=_L0_N,
                dtype_str=dtype_str,
                tag="L0-2",
                alpha=alpha,
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
                alpha=2.0,
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
        alpha=1.0,
    )

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x alpha x shape x block_size).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L1-1: alpha sweep (incl. 0.0 and negative) x all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for alpha in (0.0, -1.0, 0.5, 3):
            _try_case(
                failures,
                f"L1-1/{dtype_str}/alpha={alpha}",
                _run_case,
                N=1 << 16,
                dtype_str=dtype_str,
                tag="L1-1",
                alpha=alpha,
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

    Covers IEEE NaN/+-inf propagation, the alpha==0 no-fold requirement
    (``0*inf = NaN``), and the alpha==1 identity fold on special values.
    """
    N = 4096

    # (+-inf / NaN in a and b) x alpha={1 fold, 2.0 scale}: NaN/Inf propagate.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for alpha in (1, 2.0):
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
                    alpha=alpha,
                    equal_nan=True,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[Boundary] WARN (Record without blocking): inf-nan "
                    f"dtype={dtype_str} alpha={alpha} err={e}"
                )

    # alpha == 0 must NOT fold: b=inf -> 0*inf=NaN -> y=NaN (torch semantics).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            a = torch.randn(N, dtype=torch.float32)
            b = torch.randn(N, dtype=torch.float32)
            b[0::4] = float("inf")
            a = a.to(torch_dtype).npu()
            b = b.to(torch_dtype).npu()
            kernel = add_fwd_kernel(N, dtype_str, alpha=0.0)
            y_cpu = kernel(a, b, N).cpu()
            golden = golden_add(a, b, alpha=0.0)
            torch.testing.assert_close(y_cpu, golden, atol=0.0, rtol=0.0, equal_nan=True)
            print(
                f"[Boundary] PASS: alpha==0 no-fold (0*inf=NaN) dtype={dtype_str} "
                f"nan_out={int(torch.isnan(y_cpu).sum().item())}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): alpha=0 dtype={dtype_str} err={e}")

    # -0.0 / subnormal magnitudes (alpha=2.0).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            a = (torch.randn(N, dtype=torch.float32) * 1e-6).to(torch_dtype)
            b = torch.randn(N, dtype=torch.float32).to(torch_dtype)
            a[0:4] = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=torch_dtype)
            _run_tensor(a.npu(), b.npu(), dtype_str, "Boundary-signedzero", alpha=2.0)
        except Exception as e:  # noqa: BLE001
            print(
                f"[Boundary] WARN (Record without blocking): signed-zero "
                f"dtype={dtype_str} err={e}"
            )

    # Large magnitudes (alpha=-1.5, cancellation corners within tolerance).
    try:
        _run_case(1 << 20, "float16", "Boundary-large", alpha=-1.5, scale=100.0)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large err={e}")


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
        "Summary max_diff per (dtype, alpha): "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
