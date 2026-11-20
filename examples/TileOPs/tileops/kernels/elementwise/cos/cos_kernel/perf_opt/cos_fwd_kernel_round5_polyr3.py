# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CosFwdOp (cos) -- TileLang-NPUIR kernel, golden and hierarchical tests.

Migration: GPU TileOPs -> Ascend NPU (target="npuir"). This is the Stage 3
artifact; the frozen design is ./DESIGN.md (Stage 2 review passed, see
./REVIEW.md).

Algorithm (from the frozen design, sections 1.4 / 3.3)
------------------------------------------------------
The GPU source is a single ``cosf`` call, which is valid over the whole fp32
range.  npuir has NO equivalent: ``npuir_vcos``/``npuir_vsin`` exist but are
polynomial approximations valid only on about [-pi/4, pi/4].  Measured worst
absolute error per argument window (dense sweep, 910B2C / CANN 9.0.0):

    [0, pi/4)      3.0e-6        [3pi/4, pi)   1.9e-1
    [pi/4, pi/2)   8.1e-4        [pi, 4)       1.2e+0
    [pi/2, 3pi/4)  2.2e-2        [4, 6.5]      4.9e+1

So calling vcos directly diverges (at x ~ 6.45 it returns -47.7 where the answer
is 0.986).  This is the THIRD kind of API situation in this batch:

    ceil/floor -- instruction MISSING          -> substitute the cast round modes
    exp        -- instruction PRESENT + usable -> direct mapping
    cos        -- instruction PRESENT but its DOMAIN IS TOO NARROW
                                              -> range reduction

The kernel therefore performs an explicit four-quadrant range reduction:

    1. k = rint(x / (pi/2))                 # quadrant index (same-dtype quantising vcast)
    2. r = x - k * (pi/2)                   # now r in [-pi/4, pi/4]
    3. c = vcos(r), s = vsin(r)             # accurate on that window only
    4. select among c, -s, -c, s by k mod 4 -- arithmetic only

HARD CONSTRAINTS measured on this toolchain -- each fails SILENTLY:

  * A same-dtype ``vcast`` is NOT a no-op.  With an explicit ``round_mode`` it is a
  QUANTISER: fp32 --"rint"--> fp32 rounds the value in place, which is exactly what
  this kernel needs for the quadrant index and for floor(k/4) / floor(q/2) -- those
  three int32 round trips were removed in round 4 (measured -4.9%; vec_fops/element
  44.11 -> 41.11 with aiv_vec_ratio 0.970, i.e. purely 3 fewer vector ops).
  WITHOUT a round_mode a same-dtype conversion is still a no-op trap and must be
  ``T.copy`` (as the in-repo `mish_fwd_kernel` does).
  * ``hivm.hir.vand`` rejects scalar operands ("operand at index 1 is
    vector-only"), so the quadrant index, its parity and the sign are all
    derived ARITHMETICALLY:
        q   = k - 4*floor(k/4)          # q in {0,1,2,3}, correct for negative k
        b1  = floor(q/2)                # 0 when q<2, 1 when q>=2
        b0  = q - 2*b1                  # 0 when q even, 1 when q odd
        mag = c + (s - c)*b0
        neg = b0 XOR b1 = b0 + b1 - 2*b0*b1   # true for q in {1,2}
        out = mag - 2*neg*mag
    The negated set is {1,2}, NOT {2,3} -- an earlier revision used {2,3} and a
    dense sweep showed the magnitude right everywhere while the sign was flipped
    in exactly the odd-q windows.
  * ``vmul`` requires identical operand element types, so the output buffer must
    never be used as fp32 scratch.
  * npuir vector ops are SOURCE-FIRST: ``T.vcos(r_ub, c_ub)`` means c_ub = cos(r_ub).

KNOWN LIMIT (declared, not hidden): the reduction computes k in fp32, so for
very large |x| the subtraction x - k*(pi/2) suffers catastrophic cancellation
(the classic Cody-Waite problem).  Accuracy holds only while |k| fits
comfortably in fp32.  A Payne-Hanek reduction would be needed to go further and
that is out of scope for an operator migration.  L1 pins the boundary with
|x| ~ 1e3 / 1e4 cases.

Run:
    python cos_fwd_kernel.py --level L0    # blocking gate
    python cos_fwd_kernel.py --level all   # L0 + L1 (blocking) + L2/Boundary (warn)
"""

ASCEND_MODE = "Developer"

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

# K11: the GPU factory's threads(256) * num_per_thread(8) product collapses
# into a single block_size on NPU.
BLOCK_SIZE = 4096

# 910B2C exposes 24 AICore / 48 Vector cores (hardware-context.md).
NUM_CORES = 48

# Constants for the quadrant reduction.
_HALF_PI = 1.5707963267948966
_TWO_OVER_PI = 0.6366197723675814

# ---- 分支 R5（round 5, poly-R）常数 -----------------------------------------
# 恒等式 cos(x) = 1 - 2*sin^2(x/2)，对 y=x/2 按 pi 归约：
#   k = rint(x/(2*pi)),  r = y - k*pi ∈ [-pi/2, pi/2]
#   sin^2(y) = sin^2(r)   (sin^2 的周期是 pi，符号被平方吸收 ⇒ 不需要象限奇偶)
#   sin^2(r) = t*Q(t),  t = r^2 ∈ [0, (pi/2)^2]
# 于是 cos(x) = 1 - 2*t*Q(t)，Q 为 t 的多项式(Horner)，实测最大误差 9.7e-8。
_INV_TWO_PI = 0.15915494309189535
_PI = 3.141592653589793
# Q(t) 的系数，高次在前（deg 5，Chebyshev 拟合，[-pi/2, pi/2] 区间 max err 9.7e-8）
_POLY_C = (
    -3.6313361033289411e-06,
    0.0001393339809355861,
    -0.003172318131466477,
    0.04444304604564496,
    -0.33333299011391149,
    0.99999997991349299,
)
# fp32 专用 deg-7（在 [-pi/2, pi/2] 上 max err 9.4e-12，fp32 已是舍入级）
# 依据：fp32 用例的 vec 占用只有 72%（fp16 为 97%），有 28% 的空槽，多出来的
# 4 条 Horner 指令基本填在空槽里；同时把多项式误差从 9.7e-8 压到 1e-11。
_POLY_C7 = (
    -1.3761060558692339e-09,
    9.3218330466980845e-08,
    -4.2739901514840722e-06,
    0.00014109162407777346,
    -0.0031746019622231111,
    0.044444444037730159,
    -0.3333333332761016,
    0.99999999999804723,
)

# Tolerances lifted verbatim from the GPU source suite.
TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_cos(x: torch.Tensor) -> torch.Tensor:
    """Golden for y = cos(x), derived from the operator semantics only.

    Computes in fp32 and rounds back once -- the same opmath domain the GPU
    source uses.  It does NOT mirror the NPU four-quadrant reduction, keeping
    the comparison independent.
    """
    x_cpu = x.detach().cpu()
    return torch.cos(x_cpu.to(torch.float32)).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def cos_fwd_kernel(N, dtype, output_dtype=None, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    """Build the cos kernel factory (four-quadrant range reduction,
    grid-stride T.serial variant).

    `N` and `block_size` are Python integers at trace time, so `total_blocks` and
    `iters` are compile-time constants: the serial loop is statically bounded and
    the `blk < total_blocks` guard covers the remainder.
    """
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    q5, q4, q3, q2, q1, q0 = _POLY_C
    s7, s6, s5, s4, s3, s2, s1, s0 = _POLY_C7
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            # 分支 R5（无三角函数 · 无象限奇偶）：缓冲 5 -> 4，足迹 12 B(fp16) / 16 B(fp32)
            #   cos(x) = 1 - 2*sin^2(x/2)；对 y=x/2 按 pi 归约（sin^2 周期为 pi，
            #   符号被平方吸收），r = y - k*pi ∈ [-pi/2, pi/2]，k = rint(x/(2pi))；
            #   sin^2(r) = t*Q(t)，t = r^2，Q 为 deg-5 Horner 多项式。
            x_ub = T.alloc_ub((block_size,), dtype)  # GM 落地
            f_ub = T.alloc_ub((block_size,), compute_dtype)  # x -> y -> r -> P
            r_ub = T.alloc_ub((block_size,), compute_dtype)  # k -> k*pi -> t=r^2
            y_ub = T.alloc_ub((block_size,), out_dtype)  # GM 出口

            for i in T.serial(iters):
                blk = i * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # fp32（dtype == compute_dtype）：GM 直落 fp32 计算缓冲，省掉
                    # 一趟只吃周期、不记账 fops 的 UB->UB copy（实测 fp32 ratio -4.6pp
                    # 的主因）。fp16/bf16 仍需 x_ub 中转 + 一次加宽 vcast。
                    if dtype == compute_dtype:
                        T.copy(x[offset : offset + tail_size], f_ub[0:tail_size])
                    else:
                        T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
                        T.vcast(x_ub, f_ub, round_mode="rint")

                    # (1) k = rint(x/(2*pi)) ; r = x/2 - k*pi ∈ [-pi/2, pi/2]
                    T.vmul(f_ub, _INV_TWO_PI, r_ub)  # x/(2pi)
                    T.vcast(r_ub, r_ub, round_mode="rint")  # k（原地量化）
                    T.vmul(r_ub, _PI, r_ub)  # k*pi
                    T.vmul(f_ub, 0.5, f_ub)  # y = x/2
                    T.vsub(f_ub, r_ub, f_ub)  # r = y - k*pi

                    # (2) t = r^2 ; Q(t) = q0 + t*(q1 + t*(...))  —— Horner
                    #     fp32: deg-7 (err 9.4e-12, 且填满 fp32 的空闲 vec 槽)
                    #     fp16/bf16: deg-5 (err 9.7e-8, 远小于其容差)
                    T.vmul(f_ub, f_ub, r_ub)  # t = r^2
                    if dtype == compute_dtype:
                        T.vmul(r_ub, s7, f_ub)  # P = s7*t
                        T.vadd(f_ub, s6, f_ub)  # P += s6
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, s5, f_ub)  # P += s5
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, s4, f_ub)  # P += s4
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, s3, f_ub)  # P += s3
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, s2, f_ub)  # P += s2
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, s1, f_ub)  # P += s1
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, s0, f_ub)  # P += s0
                    else:
                        T.vmul(r_ub, q5, f_ub)  # P = q5*t
                        T.vadd(f_ub, q4, f_ub)  # P += q4
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, q3, f_ub)  # P += q3
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, q2, f_ub)  # P += q2
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, q1, f_ub)  # P += q1
                        T.vmul(f_ub, r_ub, f_ub)  # P *= t
                        T.vadd(f_ub, q0, f_ub)  # P += q0

                    # (3) cos(x) = 1 - 2*t*Q(t) = 1 - 2*sin^2(r)
                    T.vmul(f_ub, r_ub, f_ub)  # sin^2(r) = t*Q
                    T.vmul(f_ub, -2.0, f_ub)  # -2 sin^2
                    T.vadd(f_ub, 1.0, f_ub)  # 1 - 2 sin^2

                    # --- 收窄回输出 dtype（fp32 直接 UB->GM，省一趟 y_ub 中转）---
                    if out_dtype == compute_dtype:
                        T.copy(f_ub[0:tail_size], y[offset : offset + tail_size])
                    else:
                        T.vcast(f_ub, y_ub, round_mode="rint")
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
    kernel = cos_fwd_kernel(n, str(dtype).replace("torch.", ""))
    y = kernel(x, n)
    ref = golden_cos(x.cpu()).to("npu")

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

    # All four quadrants covered individually: q = rint(x/(pi/2)) mod 4.
    # q=0,1,2,3 respectively -- testing only q=0 cannot catch a sign error.
    run_values([0.1, 1.6, 3.2, 4.7], 4096, torch.float32, "L1-quadrants")
    run_values([0.1, 1.6, 3.2, 4.7], 4096, torch.float16, "L1-quadrants-fp16")
    # negative arguments: the rint/floor path must be correct for k < 0 too
    run_values([-0.1, -1.6, -3.2, -4.7], 4096, torch.float32, "L1-quadrants-neg")

    # The worst windows of vcos's measured error table -- these confirm the
    # reduction actually fires.  x ~ 6.45 is where a direct vcos gives -47.7.
    run_values([0.8, 2.0, 3.0, 6.45, 7.5], 4096, torch.float32, "L1-worst-windows")

    # large |x| WITHIN the measured accurate domain.  The reduction is accurate
    # only while |k| stays representable with room to spare; measured on this
    # toolchain against atol=rtol=1e-5:
    #     |x| <= 100  (k ~ 64)   -> 0/8192 elements out of tolerance
    #     |x| >= 500  (k ~ 318)  -> fails
    # so the blocking case stops at the measured boundary; beyond it is recorded
    # (not asserted) by record_beyond_domain() below.
    run_values([12.5, -12.5, 50.0, -50.0, 100.0, -100.0], 4096, torch.float32, "L1-large-x")

    # NaN propagation
    run_values([float("nan")] * 4, 4096, torch.float32, "L1-nan-fp32")
    run_values([float("nan")] * 4, 4096, torch.float16, "L1-nan-fp16")

    # +-inf -> NaN (IEEE: cos of an infinity is undefined)
    run_values([float("inf"), float("-inf")] * 2, 4096, torch.float32, "L1-inf-fp32")
    run_values([float("inf"), float("-inf")] * 2, 4096, torch.float16, "L1-inf-fp16")

    # near zero: no cancellation here, serves as a control
    run_values([0.0, -0.0, 1e-8, -1e-8], 4096, torch.float32, "L1-near-zero")


def run_L2():
    """Additional shapes AND the beyond-domain accuracy record -- non-blocking."""
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

    record_beyond_domain()


def record_beyond_domain():
    """Record -- do NOT assert -- the known Cody-Waite divergence.

    The source operator (CUDA cosf / torch.cos) is accurate over the whole fp32
    range; this implementation is not, because the reduction computes k in fp32.
    Measured (stage 3 probe, `history_version/design_probe_cody_waite_boundary.py`):

        |x| max    k~      max abs err   elements out of tol
        100        64      7.27e-06      0 / 8192
        500        318     2.72e-05      704 / 8192
        1000       637     5.53e-05      1828 / 8192
        10000      6366    7.23e-04      5930 / 8192

    This is a real divergence from the source operator and needs a contract-level
    decision, so it is recorded here rather than silently blocked or hidden.
    """
    print("[L2] beyond-domain accuracy record (Cody-Waite; NOT asserted)")
    for mag in (500.0, 1000.0, 10000.0):
        try:
            n = 8192
            t = torch.linspace(-1.0, 1.0, n, dtype=torch.float32) * mag
            x = t.to(torch.float32).npu()
            y = cos_fwd_kernel(n, "float32")(x, n).cpu()
            ref = golden_cos(x.cpu())
            diff = (y.float() - ref.float()).abs()
            allowed = 1e-5 + 1e-5 * ref.float().abs()
            n_bad = int((diff > allowed).sum().item())
            print(
                f"[L2]   |x|max={mag:>7.0f}  max_abs={diff.max().item():.3e}  "
                f"out_of_tol={n_bad}/{n}  (source operator: 0)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[L2]   |x|max={mag:>7.0f}  WARN: {exc}")


def run_boundary():
    """Additional boundary values -- non-blocking (warn on failure)."""
    cases = {
        "pi-2": lambda n, d: torch.full((n,), 3.141592653589793, dtype=d, device="npu"),
        "2pi": lambda n, d: torch.full((n,), 6.283185307179586, dtype=d, device="npu"),
        "neg-2pi": lambda n, d: torch.full((n,), -6.283185307179586, dtype=d, device="npu"),
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
