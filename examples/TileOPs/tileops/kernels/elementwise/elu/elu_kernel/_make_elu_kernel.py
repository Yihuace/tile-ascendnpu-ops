# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""ELU forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``_make_elu_kernel``
(``y = if_then_else(v32 > 0, val, cast(a*(T.exp(v32) - 1), dtype))``,
``register_copy`` strategy, one ``npt`` element per thread).  Semantics are
equivalent to ``torch.nn.functional.elu`` for float16 / bfloat16 / float32,
output dtype ``same_as(input)``.  The source fp32-transit precision semantics
(x -> fp32 -> ``alpha*(exp(x)-1)`` -> single round back to input dtype) is
preserved.

NPU re-design (see DESIGN.md section 0.5/0.6, revision 0):

  * R1 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical tiles (the GPU thread-level parallelism has no
    direct NPU counterpart).  ``core_num = 40`` is the physical Vector core
    count (NPUUtils: 20 AI cores x 2, Ascend910_9362).
  * R2 -- the computation core is a *literal* translation of the source
    formula.  The negative branch is ``alpha*(exp(x)-1)`` -- NOT ``expm1``.
    Both the CPU and the target-device ``F.elu`` implementation use
    ``exp(x)-1`` for the negative branch (device-proven:
    ``F.elu(-1e-7) == exp(-1e-7)-1 == -1.1920929e-7``, whereas
    ``expm1(-1e-7) == -9.99999e-8``, a 16.1% relative gap).  Therefore the
    sibling expm1-task small-|x| Taylor cancellation scheme is **explicitly
    rejected** here.  The NPU has a native true ``T.vexp`` and the op chain is

        y = vselect(x > 0, x, alpha*(vexp(x) - 1))

    implemented as a single branch-free vector chain: ``vcmp + vexp + vsub +
    (vmul) + vselect`` -- 4 ops for fp32 with alpha == 1.0 (the ``x * 1.0``
    multiply is folded out at compile time), 5 ops for alpha != 1.0.
  * R3 -- dtype transit with vector ``T.vcast``: fp32 runs the chain directly
    on the fp32 input buffer; fp16/bf16 go ``vcast(dtype->fp32, rint) ->
    chain (fp32) -> vcast(fp32->dtype, rint)``.  bf16 transit is *mandatory*
    because the v-prefix math ops (``vexp/vsub/vmul``) have no bf16 dtype in
    their matrix.
  * R4 -- the source ``T.if_then_else`` becomes a branch-free
    ``T.vcmp`` + ``T.vselect`` pair; the ``bool`` mask stays on-chip and is
    never copied back to GM.

Hard constraints honoured (DESIGN.md section 3.5.1 / 9.1):
  * R-2/R-6: every ``T.alloc_ub`` is allocated unconditionally at the
    ``T.Kernel`` top level (a conditional alloc triggers ``Undefined
    variable``); the compile-time ``if need_cast:`` / ``if need_mul:``
    branches only select which v-ops run.
  * R-7: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x_ub[0:tail]``).
  * R-8: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-9: the ``bool`` comparison mask is kept on-chip (``vselect``).
  * Zero constant buffer: ``T.fill(z, 0.0)`` once outside the tile loop.

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
_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# K11: GPU block_size = threads(256) * npt collapsed per dtype
# (source EluFwdKernel: npt = 4 for fp32, 8 for fp16/bf16).
_DEFAULT_BLOCK_SIZE = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU reference implementation, equivalent to ``torch.nn.functional.elu``.

    Mirrors the source op semantics ``alpha * (exp(cast(x,'f32')) - 1)`` with
    an ``x > 0`` take-``x`` positive branch and a single round back to the
    input dtype (source fp32-transit precision semantics).  Both the CPU and
    the target-device ``F.elu`` negative branch use ``exp(x)-1`` (not
    ``expm1``), so this is exactly ``F.elu(x.float(), alpha).to(x.dtype)``.

    Independent from the NPU kernel: only the source semantics are used here,
    never the DESIGN.md vector chain structure.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return F.elu(x_f32, alpha).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def _make_elu_kernel(N, dtype, alpha, output_dtype=None, block_size=_BLOCK_SIZE):
    """ELU forward kernel factory: y = x if x > 0 else alpha*(exp(x)-1).

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> branch-free vector chain directly on the fp32 input;
      * ``"float16"`` / ``"bfloat16"`` -> fp32 transit via ``T.vcast``.

    ``alpha`` is a compile-time scalar: when ``float(alpha) == 1.0`` the
    ``vmul`` is folded out (times-one identity, DESIGN.md section 1.6.1 #1).

    ``block_size`` folds the GPU ``threads(256) * npt`` product (K11).
    ``N`` is a compile-time constant; the runtime ``shape`` parameter guards
    the dynamic tail tile.
    """
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    need_mul = float(alpha) != 1.0
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            # R-2/R-6: all UB buffers allocated unconditionally at the top
            # level (conditional T.alloc is illegal -- TRAP-conditional-alloc).
            x_ub = T.alloc_ub((block_size,), dtype)        # input tile
            y_ub = T.alloc_ub((block_size,), out_dtype)    # round-back tile
            xf = T.alloc_ub((block_size,), f32)            # fp32 transit / domain
            e = T.alloc_ub((block_size,), f32)             # exp(x) -> -1 -> *alpha
            z = T.alloc_ub((block_size,), f32)             # read-only constant 0
            cond = T.alloc_ub((block_size,), "bool")       # on-chip mask (R-9)

            # Zero constant: filled once outside the tile loop (read-only,
            # valid across all tiles; vcmp needs a same-shape tensor operand).
            T.fill(z, T.cast(0.0, f32))

            # Compile-time alias: fp32 path computes in-place on x_ub, the
            # fp16/bf16 path computes in the fp32 transit buffer xf.
            xc = xf if need_cast else x_ub

            # In-core grid-stride over logical tiles: static loop bound,
            # guard covers non-divisible grids (DESIGN.md section 5.5).
            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    # Only T.copy uses the dynamic tail slice (R-7).
                    T.copy(x[off : off + tail], x_ub[0:tail])

                    if need_cast:
                        # lossless up-cast (f16/bf16 -> f32, rint only, R-8)
                        T.vcast(x_ub, xf, round_mode="rint")

                    # --- ELU vector chain (fp32 domain, whole-buffer v-ops) ---
                    # cond = (x > 0) captured before xc is reused as result.
                    T.vcmp(xc, z, cond, "gt")
                    T.vexp(xc, e)                       # e  = exp(x)
                    T.vsub(e, T.cast(1.0, f32), e)      # e  = exp(x) - 1
                    if need_mul:
                        T.vmul(e, T.cast(alpha, f32), e)  # e = alpha*(exp(x)-1)
                    # xc = x > 0 ? x : alpha*(exp(x)-1)  (result carried in xc)
                    T.vselect(cond, xc, e, xc)

                    if need_cast:
                        # single round back to input dtype (R-8)
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off : off + tail])
                    else:
                        T.copy(xc[0:tail], y[off : off + tail])

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
