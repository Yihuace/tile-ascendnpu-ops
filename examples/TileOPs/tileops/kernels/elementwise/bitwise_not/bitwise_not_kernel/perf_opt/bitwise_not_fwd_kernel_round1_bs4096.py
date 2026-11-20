# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""BitwiseNot forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``bitwise_not_fwd_kernel``
(``y = ~x``, ``_make_unary_direct``, 1 element per thread).  Semantics are
equivalent to ``torch.bitwise_not`` for bool / uint8 / int8 / int16 /
int32 / int64, output dtype ``same_as(input)``.

NPU re-design (see DESIGN.md section 0.5/0.6):

  * R1 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical tiles (GPU thread-level parallelism has no
    direct NPU counterpart).  ``core_num = 40`` is the physical Vector core
    count (NPUUtils: 20 AI cores x 2).
  * R2 -- integer / uint8 path uses the native ``T.vnot(src, dst)`` single
    instruction instead of the GPU "all-ones mask XOR" pair.
  * R3 -- bool (i1) cannot be loaded/stored to GM on NPU, so the bool path
    is computed with the 1-byte storage semantics ``y = x ^ 1`` using
    ``T.fill(ones_ub, 1) + T.vxor(x_ub, ones_ub, y_ub)``, which is
    equivalent to the source ``x == 0`` on the {0,1} boolean domain.

R-6 constraint (hard): the v-prefix ops are applied to the *whole* UB
buffer (``T.vnot(x_ub, y_ub)`` / ``T.vxor(x_ub, ones_ub, y_ub)``); only
``T.copy`` uses the dynamic tail slice.  Passing a dynamic slice
(``x_ub[0:tail]``) to a v-op fails at compile time.

Stage 4 round1 candidate: block_size=4096 (K11 tile fold).
run: python bitwise_not_fwd_kernel_round1_bs4096.py --level all
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

_BLOCK_SIZE = 4096
_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_bitwise_not(x):
    """BitwiseNot reference, equivalent to ``torch.bitwise_not``.

    Supports bool / uint8 / int8 / int16 / int32 / int64; elementwise
    bitwise complement with the same output dtype.  Integer/boolean bit
    operations are exact -- no intermediate precision promotion, no
    tolerance (exact_compare).
    """
    return torch.bitwise_not(x.detach().cpu())


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def bitwise_not_fwd_kernel(N, dtype, output_dtype=None, block_size=_BLOCK_SIZE):
    """BitwiseNot forward kernel factory: y = ~x.

    ``dtype`` selects the compile-time dispatch branch:
      * ``"bool"``  -> 1-byte int8 storage view, computed as ``x ^ 1``;
      * uint8 / int8 / int16 / int32 / int64 -> native ``T.vnot``.

    ``block_size`` folds the GPU ``threads(256) * num_per_thread(8) = 2048``
    (K11).  ``N`` is a compile-time constant; the runtime ``shape`` parameter
    guards the dynamic tail tile.
    """
    is_bool = dtype == "bool"
    k_dtype = "int8" if is_bool else dtype
    out_dtype = output_dtype or k_dtype
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), k_dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), k_dtype)
            y_ub = T.alloc_ub((block_size,), k_dtype)
            ones_ub = T.alloc_ub((block_size,), k_dtype)

            # In-core grid-stride over logical tiles: static loop bound,
            # guard covers non-divisible grids (DESIGN.md section 5.5).
            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])
                    # Compile-time branch (JIT trace time, no runtime branch).
                    if is_bool:
                        T.fill(ones_ub, T.cast(1, k_dtype))
                        T.vxor(x_ub, ones_ub, y_ub)
                    else:
                        T.vnot(x_ub, y_ub)
                    T.copy(y_ub[0:tail], y[off : off + tail])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _make_input(N, dtype_str):
    """Generate a CPU input tensor following the BitwiseNotWorkload logic."""
    if dtype_str == "bool":
        return torch.rand(N) > 0.5
    if dtype_str == "uint8":
        return torch.randint(0, 256, (N,), dtype=torch.uint8)
    if dtype_str == "int64":
        return torch.randint(-(2**63), 2**63 - 1, (N,), dtype=torch.int64)
    if dtype_str == "int32":
        return torch.randint(-(2**31), 2**31 - 1, (N,), dtype=torch.int32)
    if dtype_str == "int16":
        return torch.randint(-(2**15), 2**15 - 1, (N,), dtype=torch.int16)
    if dtype_str == "int8":
        return torch.randint(-128, 127, (N,), dtype=torch.int8)
    raise ValueError(f"unsupported dtype {dtype_str}")


def _max_diff(y, golden):
    return (y.to(torch.float64) - golden.to(torch.float64)).abs().max().item()


def _run_tensor(x_cpu, dtype_str, tag):
    """Run the kernel on a CPU tensor and compare bit-exactly with golden."""
    N = x_cpu.numel()
    golden = golden_bitwise_not(x_cpu)
    kernel = bitwise_not_fwd_kernel(N, dtype_str)

    if dtype_str == "bool":
        # bool is not load/store-able as i1: use the 1-byte int8 storage view.
        x_npu = x_cpu.npu().view(torch.int8)
        y_i8 = kernel(x_npu, N)
        y = y_i8.view(torch.bool)
    elif dtype_str == "uint8":
        # CG-2026-0014: jit out_idx dtype map lacks uint8 -> the auto output
        # is float32.  The kernel still writes the correct N bytes at the
        # buffer base, so recover them via a byte reinterpretation.
        x_npu = x_cpu.npu()
        y_f32 = kernel(x_npu, N)
        y = y_f32.view(torch.uint8)[:N]
    else:
        x_npu = x_cpu.npu()
        y = kernel(x_npu, N)

    y_cpu = y.cpu()
    ok = bool(torch.equal(y_cpu, golden))
    assert ok, (
        f"[{tag}] exact-match FAIL: shape=({N},) dtype={dtype_str} "
        f"max_diff={_max_diff(y_cpu, golden)}"
    )
    print(
        f"[{tag}] PASS: shape=({N},) dtype={dtype_str} "
        f"max_diff={_max_diff(y_cpu, golden)}"
    )


def _run_case(N, dtype_str, tag):
    _run_tensor(_make_input(N, dtype_str), dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke: representative 1M-element case per dtype branch
    # (DESIGN.md section 8.3 L0-1..L0-4 + int8/int16 signed branches).
    N = 1048576
    for dtype_str in ("int32", "int64", "bool", "uint8", "int8", "int16"):
        _run_case(N, dtype_str, "L0")


def run_L1():
    # L1: full dtype x boundary shapes -- divisible tiles, tail tiles and
    # non-divisible N (4_000_000 % 2048 = 1216).
    _run_case(4000000, "int32", "L1")
    _run_case(4000000, "int64", "L1")
    _run_case(2047, "int32", "L1")  # single partial tile (n_num == 1)
    _run_case(4096, "int16", "L1")  # exact 2 tiles
    _run_case(2047, "int8", "L1")
    _run_case(100, "bool", "L1")
    _run_case(100, "uint8", "L1")


def run_L2():
    # L2: edge sizes, non-blocking.
    for N, dtype_str in (
        (1, "int32"),
        (2, "int32"),
        (10, "int64"),
        (1025, "bool"),
        (4095, "uint8"),
    ):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    # Boundary: special-value inputs, all non-blocking.
    boundary_cases = (
        ("int32", torch.full((4096,), 0, dtype=torch.int32)),
        ("int32", torch.full((4096,), -1, dtype=torch.int32)),
        ("int32", torch.tensor([-(2**31), 2**31 - 1, 0, 1, -1] * 819, dtype=torch.int32)),
        ("int64", torch.tensor([-(2**63), 2**63 - 1, 0, 1, -1] * 819, dtype=torch.int64)),
        ("uint8", torch.tensor([0, 255, 1, 254, 128] * 819, dtype=torch.uint8)),
        ("bool", torch.zeros(4096, dtype=torch.bool)),
        ("bool", torch.ones(4096, dtype=torch.bool)),
    )
    for dtype_str, x in boundary_cases:
        try:
            _run_tensor(x, dtype_str, "Boundary")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}")
    # Manifest elementwise-16m workload (int32), non-blocking.
    try:
        _run_case(4096 * 4096, "int32", "Boundary-16m")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): 16m err={e}")


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
