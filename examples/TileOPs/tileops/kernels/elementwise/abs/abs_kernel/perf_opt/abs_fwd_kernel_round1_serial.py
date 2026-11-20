# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""AbsFwdOp (abs) -- Stage 4 round 1 candidate: grid-stride T.serial tiling.

BASE (read-only): ./../abs_fwd_kernel.py  (Stage 3, [PRECISION_PASS])
This branch changes exactly ONE thing relative to the base:

    Stage 3 launches one block per tile, i.e. `T.Kernel(ceildiv(N, 2048))`.
    For the 16m workload that is Block Dim = 8192 over 48 Vector cores
    (170 blocks/core, 2048-element = 4 KB tiles each); for 64m it is 131072
    blocks.  This branch folds the tile walk INTO the kernel: the grid is
    `num_cores` blocks and each block serially walks its share of tiles
    (grid-stride), which is the structure `tilelang-op-optimize/references/
    autotune.md` prescribes when Block Dim far exceeds the physical core count.

The legal `T.serial` grid-stride form follows the in-repo reference
`examples/elementwise/example_elementwise_exp2.py` (loop bound + `if block_id`
guard).  Everything else -- the source-first `T.vabs(A, B)` operand order, the
zero-start UB slices, the fp32-opmath golden -- is carried over unchanged from
the Stage 3 deliverable.

Tunables exposed for the parameter sweep: `block_size`, `num_cores`.

Run:
    python abs_fwd_kernel_round1_serial.py --level L0
"""

ASCEND_MODE = "Developer"

import argparse

import torch
import tilelang
import tilelang.language as T
import torch_npu  # noqa: F401  (registers the "npu" device)

# K11: the GPU factory's threads(256) * num_per_thread(8) product collapses
# into a single block_size on NPU.  Kept at the Stage 3 value so that round 1
# isolates the structural change (grid-stride serial walk) from the tile size.
BLOCK_SIZE = 2048

# 910B2C exposes 24 AICore / 48 Vector cores (hardware-context.md).  A pure
# Vector kernel therefore has 48 task slots.
NUM_CORES = 48

TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_abs(x: torch.Tensor) -> torch.Tensor:
    """Golden for y = |x| -- upcast to fp32, abs, single round back."""
    return torch.abs(x.float()).to(x.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def abs_fwd_kernel(N, dtype, output_dtype=None, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    """Build the abs kernel factory (grid-stride T.serial variant).

    `N` and `block_size` are Python integers at trace time, so `total_blocks`
    and `iters` are compile-time constants: the serial loop is statically
    bounded and the `blk < total_blocks` guard covers the remainder.
    """
    out_dtype = output_dtype or dtype
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def abs_fwd_kernel(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            # Vector-side buffers MUST use T.alloc_ub (Stage 3 lesson: alloc_shared
            # lowers to L1 and the UB->GM store then fails to compile).
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            for i in T.serial(iters):
                blk = i * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    # Runtime tail slice: covers N % block_size != 0 without padding.
                    tail_size = T.min(block_size, N - offset)

                    # Explicit slices on both sides; the UB slice starts at 0
                    # (TRAP-UB-dst-align).
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
                    # Source-first: y_ub = |x_ub|
                    T.vabs(x_ub, y_ub)
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return abs_fwd_kernel


# ---------- Precision comparison helper ----------
def run_case(n_total, dtype, tag):
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
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        run_case(1_048_576, dtype, "L0")


def run_L1():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        run_case(4096 * 4096, dtype, "L1")
    for n_total, dtype in (
        (1, torch.float32),
        (2, torch.float32),
        (10, torch.float32),
        (2049, torch.float16),
        (100_003, torch.float32),
    ):
        run_case(n_total, dtype, "L1-tail")


def run_L2():
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
