# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CeilFwdOp (ceil) -- Stage 4 tuned kernel (grid-stride T.serial + tuned tile).

BASE (read-only): ./../ceil_fwd_kernel.py  (Stage 3, [PRECISION_PASS])

Round 1 (structure).  Stage 3 launches one block per tile
(`T.Kernel(T.ceildiv(N, 2048))`): Block Dim = 8192 for 16m, 32768 for 64m and
131072 for 256m -- and 131072 exceeds the launch grid limit, so the Stage 3
structure cannot launch the 256m workload at all (ACL `107000` /
`Invalid_Argument(EL0003)`, child SIGABRT).  The tile walk is folded INTO the
kernel: grid = `num_cores`, each block walks its share with grid-stride
`T.serial`, and the `c_ub` constant fill is hoisted out of the tile loop.

Rounds 2-3 (tuning).  `block_size x num_cores` sweep on 16m fp16, then a
UB-ceiling probe and cross-dtype/shape confirmation:
    bs=2048 nc=48  106.12us    bs=4096 nc=48   70.62us  <- winner
    bs=4096 nc=96   70.64us    bs=4096 nc=192  71.92us
    bs=8192 nc=48  ❌ ub overflow (229,408 B > 192 KB)
`block_size` is capped at 4096 by the UB (28 bytes per element of block_size
across the ten buffers); 8192 is rejected by BiShengIR.

Measured effect (median of 15 msprof `op` launches): 16m fp16 342.07 -> 70.62us,
64m fp16 1363.01 -> 278.51us, 256m fp16 unlaunchable -> 1183.06us.

Run:
    python ceil_fwd_kernel.py --level L0

Algorithm (from the frozen design, section 1.4 / 3.3):

    The npuir build exposes NO unary ceil (verified: 53 npuir_* definitions,
    none of them ceil/floor/round/trunc), so ceil is expressed through the cast
    round modes.  A plain int32 round trip (called C1 in the design) is only
    correct for |x| < 2**31 and turns NaN into 0, so two correction paths are
    added on top:

      1. saturation bypass (C2)  -- |x| >= 2**31 -> take x itself
         (values that large are already integral in fp32, so ceil(x) == x)
      2. NaN propagation         -- NaN -> take x itself

HARD CONSTRAINTS measured on this toolchain -- getting either wrong fails
SILENTLY (no exception, just wrong numbers):

  * ``T.vabs(A, B)`` / ``T.vcast(A, B)`` are SOURCE-FIRST (``B = op(A)``).
  * ``vselect(Cond, A, B, Out)`` means ``Out = Cond ? A : B``.  The ``m_ord``
    mask is TRUE for ordered values (including +-inf) and FALSE for NaN, so the
    ceil result must be the TRUE branch:
        T.vselect(m_ord, tmp_ub, f_ub, ...)      # correct
        T.vselect(m_ord, f_ub, tmp_ub, ...)      # WRONG -- every ordinary
                                                 # value then bypasses the ceil
                                                 # (measured: 0.5 -> 0.5)
  * fp32 must go STRAIGHT to int32.  Routing it through a redundant
    fp32 -> fp32 ``vcast`` first makes the rounding silently fall back to
    truncation (0.5 -> 0).
  * Vector buffers MUST use ``T.alloc_ub``; ``T.alloc_shared`` lowers to L1 and
    makes the UB->GM store fail to compile.

Run:
    python ceil_fwd_kernel.py --level L0    # blocking gate
    python ceil_fwd_kernel.py --level all   # L0 + L1 (blocking) + L2/Boundary (warn)
"""

ASCEND_MODE = "Developer"

import argparse

import torch
import tilelang
import tilelang.language as T
import torch_npu  # noqa: F401  (registers the "npu" device)

# K11: the GPU factory's threads(256) * num_per_thread(8) product collapses
# into a single block_size on NPU.
#
# TUNED (Stage 4 rounds 1-3).  block_size is capped hard by the UB: this kernel
# holds ten block-sized UB buffers and the compiler measures the footprint at
# 28 bytes per element of block_size (bs=8192 -> 229,408 B, and BiShengIR
# rejects it with `ub overflow, requires 1835264 bits while 1572864 bits
# available`, i.e. 192 KB).  bs=4096 -> ~112 KB is the largest legal step and
# also the fastest measured.
# 8192 needs the two buffer eliminations below; at the round-1 structure it
# overflowed UB (measured: 1835264 bits required vs 1572864 available).
BLOCK_SIZE = 8192

# 910B2C exposes 24 AICore / 48 Vector cores (hardware-context.md).  The sweep
# showed 24 under-utilises the device and >=48 is flat within noise.
NUM_CORES = 48

# C2 saturation threshold.  2**31 is where the int32 round trip starts to
# overflow; using it (rather than the tighter 2**23, where fp32 loses its
# fractional part) is conservative and still exact.
SAT_MAG = 2.0 ** 31

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Tolerances lifted verbatim from the GPU source suite
# (tests/ops/test_unary_math.py::_get_tolerances).
TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_ceil(x: torch.Tensor) -> torch.Tensor:
    """Golden for y = ceil(x), derived from the operator semantics only.

    Computes in fp32 and rounds back once -- the same opmath domain the GPU
    source uses (its op_func is ``T.ceil(T.cast(x, "float32"))``).  It does NOT
    mirror the NPU int32 round trip, keeping the comparison independent.
    """
    x_cpu = x.detach().cpu()
    return torch.ceil(x_cpu.to(torch.float32)).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def ceil_fwd_kernel(N, dtype, output_dtype=None, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    """Build the ceil kernel factory (grid-stride T.serial variant).

    `N` and `block_size` are Python integers at trace time, so `total_blocks` and
    `iters` are compile-time constants: the serial loop is statically bounded and
    the `blk < total_blocks` guard covers the remainder.
    """
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    idx_dtype = "int32"
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            f_ub = T.alloc_ub((block_size,), compute_dtype)
            abs_ub = T.alloc_ub((block_size,), compute_dtype)
            c_ub = T.alloc_ub((1, 1), compute_dtype)   # constant only -> (1,1)
            t_ub = T.alloc_ub((block_size,), compute_dtype)
            # round 6: one uint8 predicate replaces the former m_ge + m_ord pair
            m_in = T.alloc_ub((block_size,), "uint8")
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # Constant buffer: vcmp requires BOTH operands to be buffers.
            # Hoisted out of the tile loop -- it does not depend on the tile.
            c_ub[0, 0] = SAT_MAG

            for i in T.serial(iters):
                blk = i * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # Explicit slices on both sides; the UB slice starts at 0.
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 1) widen to fp32 (fp32 goes straight through)
                    if dtype == compute_dtype:
                        T.copy(x_ub, f_ub)
                    else:
                        T.vcast(x_ub, f_ub, round_mode="rint")

                    # 2) C2 saturation bypass + NaN propagation
                    T.vabs(f_ub, abs_ub)                        # |x|
                    # round 6: the overflow test (|x| >= 2**31) and the NaN test
                    # (f == f) can be collapsed into ONE predicate, because both
                    # exceptions map to the same result (pass x through):
                    #   finite, in range -> |x| < c  is TRUE  -> use ceil(x)
                    #   |x| >= 2**31     -> |x| < c  is FALSE -> use x
                    #   NaN              -> any compare is FALSE (IEEE) -> use x
                    # This removes 1 vcmp and 1 vselect per tile.
                    T.vcmp(abs_ub, c_ub, m_in, "lt")            # |x| < 2**31 (NaN -> false)
                    # fp32 must go straight to int32; an extra fp32->fp32 vcast
                    # would silently turn the ceil into a truncation.
                    # round 5: a same-dtype vcast with round_mode="ceil" quantises to
                    # integer values in place, so the int32 round trip is unnecessary.
                    # (This is the same property that was a BUG in fused_add_rms_norm's
                    # fp32 branch -- here it is exactly the semantics we want.)
                    # msprof: 16m-fp16 50.562 -> 47.272 us (-6.5%), 64m-fp16 201.448 -> 191.318 (-5.0%)
                    T.vcast(f_ub, t_ub, round_mode="ceil")      # t = ceil(x) in fp32
                    # t_ub is dead once this select has read it, so the result is
                    # written back into it -- that removes one block_size-wide
                    # fp32 buffer and is what lets block_size reach 8192.
                    T.vselect(m_in, t_ub, f_ub, abs_ub)         # in-range -> ceil(x) ; overflow/NaN -> x

                    # 3) narrow to the output dtype
                    if out_dtype == compute_dtype:
                        T.copy(abs_ub, y_ub)
                    else:
                        T.vcast(abs_ub, y_ub, round_mode="rint")

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
    kernel = ceil_fwd_kernel(n, str(dtype).replace("torch.", ""))
    y = kernel(x, n)
    ref = golden_ceil(x.cpu()).to("npu")

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


def run_values(vals, n, dtype, tag):
    """Build an n-length tensor whose head is `vals` (padded with 1.25)."""
    head = torch.tensor(vals, dtype=torch.float32)
    pad = torch.full((n - head.numel(),), 1.25, dtype=torch.float32)
    x = torch.cat([head, pad]).to(dtype).npu()
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

    # ceil ties and sign crossing
    run_values([0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 0.0, -0.0], 4096, torch.float32, "L1-ties")

    # NaN propagation (must stay NaN)
    run_values([float("nan")] * 4, 4096, torch.float32, "L1-nan-fp32")
    run_values([float("nan")] * 4, 4096, torch.float16, "L1-nan-fp16")

    # +-inf propagation
    run_values([float("inf"), float("-inf")] * 2, 4096, torch.float32, "L1-inf-fp32")
    run_values([float("inf"), float("-inf")] * 2, 4096, torch.float16, "L1-inf-fp16")

    # beyond the int32 range: must return x itself, not +-2.147e9
    run_values([3.0e9, -3.0e9, 4.0e9, -4.0e9], 4096, torch.float32, "L1-beyond-int32")

    # around 2**23, where fp32 changes from fractional to already-integral
    run_values(
        [float(2 ** 23 - 1), float(2 ** 23), float(2 ** 23 + 1), float(2 ** 24 + 0.5)],
        4096,
        torch.float32,
        "L1-2p23",
    )


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
        "tiny": lambda n, d: torch.full((n,), 1e-8, dtype=d, device="npu"),
        "neg-tiny": lambda n, d: torch.full((n,), -1e-8, dtype=d, device="npu"),
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
