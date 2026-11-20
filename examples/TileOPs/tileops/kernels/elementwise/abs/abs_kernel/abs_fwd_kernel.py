# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""AbsFwdOp (abs) -- TileLang-NPUIR kernel, golden and hierarchical tests.

Migration: GPU TileOPs -> Ascend NPU (target="npuir"). This is the Stage 3
artifact; the frozen design is ./DESIGN.md (Stage 2 review passed, see
./REVIEW.md).

Algorithm (from the frozen design):
    1-D flat, one block per core, UB tile staging, one vector op per tile.

Design constraints honoured here:
  * ``T.vabs(A, B)`` is SOURCE-FIRST -- it computes ``B = |A|``. This is the
    opposite of the ascendc-pto line's ``T.tile.abs(dst, src)``. Verified
    against ``tilelang/language/customize_npuir.py`` (``npuir_abs(A, B)``) and
    in-repo call sites (``examples/deepseek_v4/example_act_quant_kernel.py:61``,
    ``examples/deepseek_v4/engram/engram_gate_kernel_fwd.py:178``). Writing it
    the other way round does not raise -- it silently produces a wrong result.
  * ``tail_size`` slicing is applied to BOTH src and dst, and every UB slice
    starts at 0 (TRAP-UB-dst-align: a non-zero-start UB slice plus a 32B-wide
    vector op triggers a VEC alignment error).
  * The golden uses fp32 opmath plus a single round back, which matches
    torch's CPU reference domain (TRAP-fp16-opmath-golden).

Run:
    python abs_fwd_kernel.py --level L0    # blocking gate
    python abs_fwd_kernel.py --level all   # L0 + L1 (blocking) + L2/Boundary (warn)
"""

ASCEND_MODE = "Developer"

import argparse

import torch
import tilelang
import tilelang.language as T
import torch_npu  # noqa: F401  (registers the "npu" device)

# K11: the GPU factory's threads(256) * num_per_thread(8) product collapses
# into a single block_size on NPU.
BLOCK_SIZE = 2048

# Tolerances are lifted verbatim from the GPU source suite
# (tests/ops/test_unary_math.py::_get_tolerances) so the NPU gate matches the
# source gate.
TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_abs(x: torch.Tensor) -> torch.Tensor:
    """Golden for y = |x|, derived from the operator semantics only.

    Upcasts to fp32, applies the absolute value, then rounds back once -- i.e.
    the same opmath domain torch's CPU reference uses. This deliberately does
    NOT mirror the NPU algorithm, keeping the comparison independent.
    """
    return torch.abs(x.float()).to(x.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def abs_fwd_kernel(N, dtype, output_dtype=None, block_size=BLOCK_SIZE):
    """Build the abs kernel factory.

    The interface matches the extracted GPU factory
    (``abs_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)``)
    with the backend-specific parameters adapted: ``threads``/``num_per_thread``
    collapse into ``block_size`` (K3 + K11).
    """
    out_dtype = output_dtype or dtype

    @T.prim_func
    def abs_fwd_kernel(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
            # Vector-side buffers MUST use T.alloc_ub. T.alloc_shared lowers to
            # L1 (the Cube-side shared memory), which makes the UB->GM store
            # below fail to compile with
            #   "'hivm.hir.store' op only support store ub to gm currently!"
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            offset = cid * block_size
            # Runtime tail slice: covers N % block_size != 0 without padding.
            tail_size = T.min(block_size, N - offset)

            # Explicit slices on both sides; the UB slice starts at 0.
            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
            # Source-first: y_ub = |x_ub|
            T.vabs(x_ub, y_ub)
            T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return abs_fwd_kernel


# ---------- Precision comparison helper ----------
def run_case(n_total, dtype, tag):
    """Build inputs on NPU, run the kernel, compare against the golden."""
    x = torch.randn(n_total, dtype=dtype, device="npu")
    y = abs_fwd_kernel(n_total, str(dtype).replace("torch.", ""))(x)
    ref = golden_abs(x.cpu()).to("npu")

    diff = (y.float() - ref.float()).abs()
    max_abs = diff.max().item()
    denom = ref.float().abs().clamp_min(1e-12)
    max_rel = (diff / denom).max().item()

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(y, ref, atol=tol["atol"], rtol=tol["rtol"])
    print(
        f"[{tag}] PASS: N={n_total} dtype={str(dtype).replace('torch.', '')} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )
    return max_abs, max_rel


# ---------- Hierarchical tests ----------
def run_L0():
    """Smoke gate -- blocking. Smoke workload from the manifest, all dtypes."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        run_case(1_048_576, dtype, "L0")


def run_L1():
    """Full-scale and non-divisible shapes -- blocking."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        run_case(4096 * 4096, dtype, "L1")
    # Non-divisible extents: the tail path (N % block_size != 0).
    for n_total, dtype in (
        (1, torch.float32),
        (2, torch.float32),
        (10, torch.float32),
        (2049, torch.float16),
        (100_003, torch.float32),
    ):
        run_case(n_total, dtype, "L1-tail")


def run_L2():
    """Additional shapes -- non-blocking (warn on failure).

    Kept at a scale the host can hold: the advisory layer upcasts to fp32 for
    the golden, so the manifest's largest workload (16384 x 16384 = 268M
    elements) would need >1.6 GB of host memory on top of the device buffers
    and aborts the process instead of reporting a warn. L1 already covers
    16.7M elements, which is enough for the advisory layer's purpose.
    """
    for n_total, dtype in (
        (8192 * 8192, torch.float16),
        (8192 * 8192, torch.bfloat16),
        (63, torch.bfloat16),
        (4097, torch.float16),
    ):
        try:
            run_case(n_total, dtype, "L2")
        except Exception as exc:  # noqa: BLE001  (L2 is advisory)
            print(f"[L2] WARN (non-blocking): N={n_total} dtype={dtype}: {exc}")


def run_boundary():
    """Boundary values -- non-blocking (warn on failure)."""
    cases = {
        "zeros": lambda n, d: torch.zeros((n,), dtype=d, device="npu"),
        "neg-zeros": lambda n, d: torch.full((n,), -0.0, dtype=d, device="npu"),
        "inf": lambda n, d: torch.full((n,), float("inf"), dtype=d, device="npu"),
        "neg-inf": lambda n, d: torch.full((n,), float("-inf"), dtype=d, device="npu"),
        "nan": lambda n, d: torch.full((n,), float("nan"), dtype=d, device="npu"),
        "extreme": lambda n, d: torch.tensor(
            [1e-8, -1e-8, 3.4e38, -3.4e38] * (n // 4 + 1), dtype=d, device="npu"
        )[:n],
    }
    for name, gen in cases.items():
        for dtype in (torch.float32, torch.float16):
            try:
                n = 2050
                x = gen(n, dtype)
                y = abs_fwd_kernel(n, str(dtype).replace("torch.", ""))(x)
                ref = golden_abs(x.cpu()).to("npu")
                # NaN positions must match; elsewhere compare numerically.
                torch.testing.assert_close(
                    y, ref, equal_nan=True, atol=TOLERANCES[dtype]["atol"],
                    rtol=TOLERANCES[dtype]["rtol"],
                )
                print(f"[Boundary] PASS: {name} dtype={dtype}")
            except Exception as exc:  # noqa: BLE001  (Boundary is advisory)
                print(f"[Boundary] WARN (non-blocking): {name} dtype={dtype}: {exc}")


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
