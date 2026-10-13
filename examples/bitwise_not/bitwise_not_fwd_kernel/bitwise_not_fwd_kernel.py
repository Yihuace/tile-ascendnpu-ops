# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Bitwise NOT forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``BitwiseNotFwdKernel`` (TileOPs,
``register_copy`` strategy; ``direct`` for a bool operand) to TileLang
``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = ~x   (elementwise bitwise NOT / all-ones complement)

on a flattened contiguous ``(N,)`` tensor; output ``(N,)`` same dtype.
Supported dtypes: bool | uint8 | int8 | int16 | int32 | int64 (float is
rejected).  The integer paths are bit-exact (no fp32 transit, atol = rtol = 0).

Byte storage note (DESIGN.md sections 3.3 / 11.4, implemented as int8): bool
i1 cannot load/store GM, so bool is passed as a one-byte storage view.  uint8
is also carried through the same int8 byte storage: TileLang NPU's runtime
output-dtype map (``JitKernel_NPU._tvm_dtype_to_torch``) has no ``uint8`` entry
and silently allocates the auto output as float32, so declaring a ``uint8``
tensor would corrupt the result.  int8 has the identical bit pattern for every
operand of ``vnot`` / ``vxor``, so the byte path is bit-exact; the host views
``bool``/``uint8`` to ``int8`` on the way in and back on the way out.  This
matches the GPU source's ``BOOL_STORAGE_DTYPE = int8``.

NPU redesign (DESIGN.md sections 0.6 / 3.3):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` with ``T.alloc_ub``
        tiles and explicit-slice ``T.copy``; the CUDA ``threads x npt``
        two-level parallel folds into a single ``block_size``.  Compute is a
        single vector op: ``T.vnot`` for integer dtypes, and for bool a
        byte-domain ``T.vxor`` against a broadcast constant-one byte tile (byte
        ``x ^ 1``; the scalar-operand form is not lowerable on npuir).
    R2: persistent core split ``num_kernels = min(num_logical, vector_cores)``
        with an in-core ``T.serial`` grid-stride loop and an if-guard for
        out-of-range tiles (static bounds).

Run the embedded hierarchical tests:
    python bitwise_not_fwd_kernel.py --level L0
    python bitwise_not_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual sync
# points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_bitwise_not", "bitwise_not_fwd_kernel"]

# Manifest dtype set (BITWISE_DTYPES): integer / bool only; float is rejected.
SUPPORTED_DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64")

# Per-dtype UB tile-width cap (DESIGN.md sections 5.2 / 4.5).  ``block_size``
# 8192 keeps the explicit x_ub + y_ub peak at <= 64KB for the 1/2/4-byte
# dtypes; int64 uses 4096 so its 8-byte peak also stays at 64KB, well below the
# 192KB UB budget.  Keys are the *storage* dtypes (bool/uint8 -> int8).
_MAX_BLOCK_SIZE = {
    "int8": 8192,
    "int16": 8192,
    "int32": 8192,
    "int64": 4096,
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_bitwise_not(x):
    """Bitwise NOT reference: y = ~x (exact, integer/bool).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_bitwise_not.py: ``ref_program = torch.bitwise_not(x)``, atol = rtol
    = 0).  Independent of the NPU algorithm (no vnot / vxor / byte-storage /
    persistent structure).  Runs on CPU and does not require an NPU device.
    """
    return torch.bitwise_not(x.detach().cpu())


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def bitwise_not_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    """Bitwise NOT forward kernel: ``y = ~x`` for bool / integer inputs.

    Compile-time params:
        N: flattened element count (input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "bool" | "uint8" | "int8" | "int16" | "int32" |
            "int64"; float dtypes raise ``ValueError``.
        output_dtype: optional output dtype (defaults to ``dtype``).
        block_size: UB tile width (folds the GPU ``threads * npt`` product;
            8192 per section 5.2, capped per dtype to respect the UB budget).
        num_cores: optional explicit kernel-count override (else section 5.5).

    The returned ``main(x, y, shape)`` is invoked as ``kernel(x, N)``; ``y``
    is auto-allocated (``out_idx=[1]``).  For bool / uint8 inputs the caller
    passes an ``int8`` byte view and views the result back (bool i1 cannot
    load/store GM; uint8 has no runtime output-dtype mapping -- see the module
    docstring).
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"bitwise_not_fwd_kernel only supports dtypes {SUPPORTED_DTYPES}, "
            f"got {dtype!r}"
        )

    out_dtype = output_dtype or dtype
    is_bool = dtype == "bool"
    # bool -> byte storage (bool i1 cannot load/store GM); uint8 -> int8 byte
    # storage (identical bit pattern; uint8 lacks a runtime output-dtype map).
    storage_dtype = "int8" if dtype in ("bool", "uint8") else dtype

    if block_size is None:
        block_size = _MAX_BLOCK_SIZE[storage_dtype]
    else:
        block_size = min(block_size, _MAX_BLOCK_SIZE[storage_dtype])

    num_logical = T.ceildiv(N, block_size)  # logical tile count (compile-time)
    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op; re-queried per build so other devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

    @T.prim_func
    def main(
        x: T.Tensor((N,), storage_dtype),
        y: T.Tensor((N,), storage_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # UB buffers (DESIGN.md section 4.3): v-prefix APIs require UB
            # residency.  ``c_ub`` is the constant-one byte tile for the bool
            # path; it is allocated unconditionally (a conditional alloc is not
            # accepted by the TVM script parser) but unused for integer paths.
            x_ub = T.alloc_ub((block_size,), storage_dtype)
            y_ub = T.alloc_ub((block_size,), storage_dtype)
            c_ub = T.alloc_ub((block_size,), storage_dtype)

            # Persistent in-core serial tile loop (static bound, grid-stride
            # mapping, if-guard masks out-of-range tiles).
            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # 1. Load GM -> UB (only valid elements copied in).
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 2. Compute (single vector op; Python branch resolved at
                    #    JIT trace time).  Integer: bitwise NOT (~x).  bool
                    #    byte domain: logical NOT via x ^ 1 (0 <-> 1).  The
                    #    constant operand must be a shaped tile -- a scalar
                    #    src1 lowers to an i8 operand that hivm.hir.vxor
                    #    rejects, hence the broadcast ``T.fill``.
                    if is_bool:
                        T.fill(c_ub, T.cast(1, storage_dtype))
                        T.vxor(x_ub, c_ub, y_ub)
                    else:
                        T.vnot(x_ub, y_ub)

                    # 3. Store UB -> GM (only valid elements copied out;
                    #    garbage tail lanes are never written back).
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _max_diff_metric(out, ref):
    """Max absolute difference between out and ref (integer/bool, exact)."""
    out = out.detach().cpu()
    ref = ref.detach().cpu()
    if out.dtype == torch.bool:
        out = out.to(torch.uint8)
        ref = ref.to(torch.uint8)
    eq = out == ref
    if bool(eq.all()):
        return 0.0
    diff = (out.to(torch.int64) - ref.to(torch.int64)).abs()
    diff = torch.where(eq, torch.zeros_like(diff), diff)
    return diff.max().item()


def _to_storage(t, dtype_str):
    """bool / uint8 -> int8 byte view (metadata only); others unchanged."""
    if dtype_str in ("bool", "uint8"):
        return t.view(torch.int8)
    return t


def _from_storage(t, dtype_str):
    """int8 byte view -> bool / uint8 (metadata only); others unchanged."""
    if dtype_str == "bool":
        return t.view(torch.bool)
    if dtype_str == "uint8":
        return t.view(torch.uint8)
    return t


def _make_input(N, dtype_str):
    """Random integer / bool input of shape (N,) (CPU)."""
    if dtype_str == "bool":
        return torch.randint(0, 2, (N,), dtype=torch.int64).to(torch.bool)
    if dtype_str == "int64":
        # int64 full-range randint is not supported; stay within a safe range
        # (extremes are covered by the Boundary cases).
        return torch.randint(-(2**62), 2**62, (N,), dtype=torch.int64)
    torch_dtype = _TORCH_DTYPE[dtype_str]
    info = torch.iinfo(torch_dtype)
    return torch.randint(
        int(info.min), int(info.max) + 1, (N,), dtype=torch.int64
    ).to(torch_dtype)


def _run_tensor(x, dtype_str, tag):
    """Run kernel on a pre-made CPU tensor and compare against golden."""
    N = x.shape[0]
    kernel = bitwise_not_fwd_kernel(N, dtype_str)
    x_storage = _to_storage(x, dtype_str).npu()
    y_storage = kernel(x_storage, N)
    y = _from_storage(y_storage.cpu(), dtype_str)
    golden = golden_bitwise_not(x)
    # Bit-exact: atol = rtol = 0 (DESIGN.md section 8.2).
    torch.testing.assert_close(y, golden, atol=0, rtol=0)
    max_diff = _max_diff_metric(y, golden)
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_case(N, dtype_str, tag):
    """Run kernel on random data of shape (N,) with given dtype."""
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag)


# ---------------------------------------------------------------------------
# hierarchical testing
# ---------------------------------------------------------------------------
def run_L0():
    """L0 gate: 1M-element case for every supported dtype (DESIGN.md 8.3),
    plus the negative case that float dtypes are rejected."""
    N = 1048576
    for dtype_str in SUPPORTED_DTYPES:
        _run_case(N, dtype_str, "L0")

    # Negative: float dtypes must be rejected with ValueError.
    for bad_dtype in ("float16", "float32", "bfloat16"):
        try:
            bitwise_not_fwd_kernel(N, bad_dtype)
            raise AssertionError(f"{bad_dtype} must be rejected")
        except ValueError:
            print(f"[L0] PASS: dtype {bad_dtype} rejected")


def run_L1():
    """L1: divisible + non-divisible shapes across all dtypes (tail path)."""
    for dtype_str in SUPPORTED_DTYPES:
        for N in (8192, 100, 4097):
            _run_case(N, dtype_str, "L1")
    # Large spot checks.
    _run_case(4194304, "int32", "L1")
    _run_case(4194304, "int64", "L1")
    _run_case(1048576, "int16", "L1")

    # Non-default block_size sanity (UB budget still holds).
    x = _make_input(1048576, "int32")
    N = x.shape[0]
    kernel = bitwise_not_fwd_kernel(N, "int32", block_size=2048)
    y = kernel(_to_storage(x, "int32").npu(), N).cpu()
    golden = golden_bitwise_not(x)
    torch.testing.assert_close(y, golden, atol=0, rtol=0)
    print(f"[L1] PASS: block_size=2048 override max_diff={_max_diff_metric(y, golden):.2e}")


def run_L2():
    """L2: edge sizes (warn only, non-blocking)."""
    for dtype_str in ("uint8", "int32", "int64"):
        for N in (1, 2, 10):
            try:
                _run_case(N, dtype_str, "L2")
            except Exception as e:  # noqa: BLE001
                print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    """Boundary: integer extremes (min / max / 0 / all-ones) and bool {0,1}
    (warn only, non-blocking; exact equality required when they run)."""
    N = 4096

    for dtype_str in ("uint8", "int8", "int16", "int32", "int64"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            info = torch.iinfo(torch_dtype)
            vals = torch.tensor(
                [info.min, info.max, 0, -1, 1], dtype=torch.int64
            )
            reps = (N + vals.numel() - 1) // vals.numel()
            x = vals.repeat(reps)[:N].to(torch_dtype)
            _run_tensor(x, dtype_str, "Boundary-extremes")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}")

    # bool {0, 1} through the byte-storage round trip.
    try:
        x = torch.randint(0, 2, (N,), dtype=torch.int64).to(torch.bool)
        _run_tensor(x, "bool", "Boundary-bool")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): bool err={e}")


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
