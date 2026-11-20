# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""ExpFwdOp (exp) -- TileLang-NPUIR kernel, golden and hierarchical tests.

Migration: GPU TileOPs -> Ascend NPU (target="npuir"). This is the Stage 3
artifact; the frozen design is ./DESIGN.md (Stage 2 review passed, see
./REVIEW.md).

Algorithm (from the frozen design, section 1.4 / 3.3) -- the SIMPLEST member of
this batch, because npuir DOES expose a unary exp (`npuir_exp`):

    widen to fp32 -> T.vexp -> narrow to the output dtype

Unlike `ceil`/`floor`, which had to substitute `npuir_cast` round modes for a
missing instruction and therefore carry a saturation bypass plus a NaN
propagation path, exp needs NO compensation at all: npuir has `npuir_exp` and
it works on fp32, which is exactly the domain the GPU source computes in
(`op_func` is `T.exp(T.cast(x, "float32"))`).

Semantics worth pinning down (section 0.1): the compute domain is fp32 while the
output domain is the input dtype, so for fp16 `exp(20.0) -> +inf` even though
the fp32 reference is a finite ~4.85e8.  That is the SOURCE operator's existing
behaviour, not a defect, and it is asserted in L1.

HARD CONSTRAINTS measured on this toolchain -- getting either wrong fails
SILENTLY (no exception, just wrong numbers):

  * ``T.vexp(A, B)`` is SOURCE-FIRST (``B = exp(A)``); the opposite of the
    ascendc-pto line's ``T.tile.*(dst, src)``.
  * Vector buffers MUST use ``T.alloc_ub``; ``T.alloc_shared`` lowers to L1 and
    makes the UB->GM store fail to compile.

UNVERIFIED POINT (declared, not assumed): whether ``vexp`` propagates NaN and
+-inf at the instruction level has NOT been measured in this project.  The
`ceil`/`floor` migrations proved that a `vcast` turns NaN into 0, so special-value
propagation cannot be assumed.  NaN / +-inf / overflow / underflow / fp16-range
cases are therefore asserted in L1 (blocking); if they fail, the design calls for
returning [DESIGN_ERROR] and adding a C2-style bypass.

Run:
    python exp_fwd_kernel.py --level L0    # blocking gate
    python exp_fwd_kernel.py --level all   # L0 + L1 (blocking) + L2/Boundary (warn)
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

# Tolerances lifted verbatim from the GPU source suite.
TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_exp(x: torch.Tensor) -> torch.Tensor:
    """Golden for y = exp(x), derived from the operator semantics only.

    Computes in fp32 and rounds back once -- the same opmath domain the GPU
    source uses (`op_func` is ``T.exp(T.cast(x, "float32"))``).  It does NOT
    mirror the NPU implementation, keeping the comparison independent.
    """
    x_cpu = x.detach().cpu()
    return torch.exp(x_cpu.to(torch.float32)).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def exp_fwd_kernel(N, dtype, output_dtype=None, block_size=BLOCK_SIZE):
    """Build the exp kernel factory.

    The interface matches the extracted GPU factory
    (``exp_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)``)
    with the backend-specific parameters adapted: ``threads``/``num_per_thread``
    collapse into ``block_size`` (K3 + K11).
    """
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            f_ub = T.alloc_ub((block_size,), compute_dtype)
            e_ub = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            offset = cid * block_size
            remaining = shape - offset
            tail_size = T.min(block_size, remaining)

            # Explicit slices on both sides; the UB slice starts at 0.
            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

            # 1) widen to fp32 (fp32 goes straight through)
            if dtype == compute_dtype:
                T.copy(x_ub, f_ub)
            else:
                T.vcast(x_ub, f_ub, round_mode="rint")

            # 2) exp in the fp32 domain -- source-first: e_ub = exp(f_ub)
            T.vexp(f_ub, e_ub)

            # 3) narrow to the output dtype
            if out_dtype == compute_dtype:
                T.copy(e_ub, y_ub)
            else:
                T.vcast(e_ub, y_ub, round_mode="rint")

            T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


# ---------- Precision comparison helper ----------
def _max_diff(y, ref):
    """Max abs diff over non-NaN positions (NaN equality is asserted separately)."""
    d = (y.float() - ref.float()).abs()
    d = torch.nan_to_num(d, nan=0.0)
    return d.max().item()


def run_tensor(x, tag, equal_nan=False):
    """Run the kernel on an NPU tensor and compare against the golden."""
    n = x.shape[0]
    dtype = x.dtype
    kernel = exp_fwd_kernel(n, str(dtype).replace("torch.", ""))
    y = kernel(x, n)
    ref = golden_exp(x.cpu()).to("npu")

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(
        y.cpu(), ref.cpu(), equal_nan=equal_nan, atol=tol["atol"], rtol=tol["rtol"]
    )
    print(
        f"[{tag}] PASS: N={n} dtype={str(dtype).replace('torch.', '')} "
        f"max_diff={_max_diff(y.cpu(), ref.cpu()):.3e}"
    )


def run_random(n, dtype, tag, scale=3.0):
    x = (torch.randn(n, dtype=torch.float32) * scale).to(dtype).npu()
    run_tensor(x, tag)


def run_values(vals, n, dtype, tag, pad=0.25):
    """Build an n-length tensor whose head is `vals` (padded with `pad`)."""
    head = torch.tensor(vals, dtype=torch.float32)
    fill = torch.full((n - head.numel(),), pad, dtype=torch.float32)
    x = torch.cat([head, fill]).to(dtype).npu()
    run_tensor(x, tag, equal_nan=True)


# ---------- Hierarchical tests ----------
def run_L0():
    """Smoke gate -- blocking."""
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        run_random(1_048_576, dtype, "L0")


def run_L1():
    """Full-scale, non-divisible shapes AND every boundary semantic that
    section 0.1 declares must be preserved -- blocking."""
    # full-scale workloads from the manifest
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        run_random(4096 * 4096, dtype, "L1")

    # non-divisible extents (the tail path)
    for n, dtype in (
        (1, torch.float32),
        (2, torch.float32),
        (10, torch.float32),
        (2049, torch.float16),
        (100_003, torch.float32),
    ):
        run_random(n, dtype, "L1-tail")

    # NaN propagation -- vexp's special-value behaviour is UNVERIFIED here
    run_values([float("nan")] * 4, 4096, torch.float32, "L1-nan-fp32")
    run_values([float("nan")] * 4, 4096, torch.float16, "L1-nan-fp16")

    # +inf -> +inf and -inf -> +0.0
    run_values([float("inf"), float("-inf")] * 2, 4096, torch.float32, "L1-inf-fp32")
    run_values([float("inf"), float("-inf")] * 2, 4096, torch.float16, "L1-inf-fp16")

    # overflow / underflow in the fp32 compute domain
    run_values([100.0, -100.0, 90.0, -90.0], 4096, torch.float32, "L1-overflow-fp32")

    # fp16 output range: exp(20) ~ 4.85e8 exceeds fp16 max, so the result must be
    # +inf while the fp32 reference stays finite -- the source's own semantics.
    run_values([20.0, 11.1, -20.0, 0.0], 4096, torch.float16, "L1-fp16-range")

    # near zero: exp(+-1e-8) ~ 1.0
    run_values([0.0, -0.0, 1e-8, -1e-8], 4096, torch.float32, "L1-near-zero")


def run_L2():
    """Additional shapes -- non-blocking (warn on failure)."""
    for n, dtype in (
        (8192 * 8192, torch.float16),
        (8192 * 8192, torch.bfloat16),
        (63, torch.bfloat16),
        (4097, torch.float16),
    ):
        try:
            run_random(n, dtype, "L2")
        except Exception as exc:  # noqa: BLE001  (L2 is advisory)
            print(f"[L2] WARN (non-blocking): N={n} dtype={dtype}: {exc}")


def run_boundary():
    """Additional boundary values -- non-blocking (warn on failure)."""
    cases = {
        "zeros": lambda n, d: torch.zeros((n,), dtype=d, device="npu"),
        "neg-zeros": lambda n, d: torch.full((n,), -0.0, dtype=d, device="npu"),
        "large-neg": lambda n, d: torch.full((n,), -30.0, dtype=d, device="npu"),
    }
    for name, gen in cases.items():
        for dtype in (torch.float32, torch.float16):
            try:
                run_tensor(gen(2050, dtype), f"Boundary-{name}", equal_nan=True)
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
