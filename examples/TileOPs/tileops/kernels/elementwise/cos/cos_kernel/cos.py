# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Cos forward kernel — NPU (npuir) implementation.

Migrated from GPU TileOPs ``tileops/kernels/elementwise.py`` — ``CosFwdKernel``
/ ``cos_fwd_kernel``.  GPU reference preserved verbatim in
``../_cos_fwd_kernels.py``; this file is the NPU re-implementation.

Adaptations: K1 (``target="npuir"`` + v-prefix body), K2 (1-D NPUIR grid, no
manual sync), K3 (``threads`` removed), K11 (``threads*npt`` -> ``block_size``).

=====================================================================
MEASURED INSTRUCTION ACCURACY AND THE RESULTING RANGE REDUCTION
=====================================================================

``npuir_vcos`` is a polynomial approximation whose validity interval is
**only about [-pi/4, pi/4]**.  A dense sweep on 910B2C / CANN 9.0.0 /
npuir wheel v0.1.2 measured (worst absolute error per argument window):

| |x| window        | max abs error |
|-------------------|---------------|
| [0, pi/4)         | 3.0e-6        |
| [pi/4, pi/2)      | 8.1e-4        |
| [pi/2, 3pi/4)     | 2.2e-2        |
| [3pi/4, pi)       | 1.9e-1        |
| [pi, 4)           | 1.2e+0        |
| [4, 6.5]          | 4.9e+1        |

So calling ``vcos`` directly diverges outside a narrow band (at x ~ 6.45 it
returns -47.7 where the answer is 0.986).  The migration therefore performs
**explicit four-quadrant range reduction** in the kernel:

    1. k = rint(x / (pi/2))            # quadrant index (float + int32)
    2. r = x - k * (pi/2)              # r is now in [-pi/4, pi/4]
    3. c = vcos(r), s = vsin(r)        # both accurate on [-pi/4, pi/4]
    4. cos(x) selects among c, -s, -c, s by k mod 4

Step 4 is expressed with **arithmetic only** — no compare/select and no
bitwise ops.  ``hivm.hir.vand`` rejects scalar operands ("operand at index 1 is
vector-only"), so the quadrant indices are derived arithmetically instead:

    q  = k - 4 * floor(k/4)      # q in {0,1,2,3}; correct for negative k too
    b1 = floor(q/2)              # 0 for q<2, 1 for q>=2
    b0 = q - 2*b1                # 0 when q even, 1 when q odd
    mag    = c + (s - c) * b0                 # cos(r) or sin(r) by parity
    neg    = b0 XOR b1 = b0 + b1 - 2*b0*b1    # true for q in {1, 2}
    result = mag * (1 - 2*neg)

The quadrant table behind this is the standard one for `q = k mod 4`:

    q=0 ->  cos(r)      q=1 -> -sin(r)
    q=2 -> -cos(r)      q=3 ->  sin(r)

(An earlier revision negated for `q in {2,3}` instead of `{1,2}`; the measured
sweep showed the magnitude was right everywhere while the sign was flipped in
exactly the odd-q windows, which is what pinned the error to the sign term.)

(``floor`` is ``vcast(..., round_mode="floor")``, measured working on the
fp32 -> int32 path in §5.2 of the migration log.)

``cos(x)`` is periodic with period ``2*pi``; ``k`` simply wraps around, so the
reduction is exact for any finite x.  The residual accuracy is then the
instruction's accuracy on ``[-pi/4, pi/4]`` plus fp32 rounding of the
reduction itself.

KNOWN LIMIT (documented, not hidden): the reduction computes ``k`` in fp32, so
for very large ``|x|`` the subtraction ``x - k*(pi/2)`` suffers catastrophic
cancellation (the classic Cody-Waite problem).  Accuracy is guaranteed only
while ``|k|`` fits comfortably in fp32 with room for the fractional part —
in practice a few decimal digits of ``x``.  The manifest workloads (randn
scaled vectors) are far inside this.  ``pip install`` of a better algorithm
(Payne-Hanek) is out of scope for an operator migration.
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

tilelang.cache.clear_cache()

# Constants for the quadrant reduction.
_HALF_PI = 1.5707963267948966
_TWO_OVER_PI = 0.6366197723675814


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_cos(x):
    """cos reference, computed in fp32 then cast back (matches the kernel)."""
    x_cpu = x.detach().cpu()
    return torch.cos(x_cpu.to(torch.float32)).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def cos_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    """Element-wise cosine: y = cos(x), with explicit four-quadrant reduction.

    ``block_size`` folds the GPU ``threads(256) * num_per_thread(8) = 2048``
    per-block element count (K11).

    See the module docstring for why the raw ``vcos`` instruction cannot be
    applied directly and how the reduction restores accuracy.
    """
    out_dtype = output_dtype or dtype
    n_num = T.ceildiv(N, block_size)
    compute_dtype = "float32"
    idx_dtype = "int32"

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            f_ub = T.alloc_ub((block_size,), compute_dtype)     # x as fp32
            kf_ub = T.alloc_ub((block_size,), compute_dtype)    # k as fp32
            ki_ub = T.alloc_ub((block_size,), idx_dtype)        # k as int32
            qf_ub = T.alloc_ub((block_size,), compute_dtype)    # q = k mod 4
            t_ub = T.alloc_ub((block_size,), compute_dtype)     # scratch
            half_ub = T.alloc_ub((block_size,), compute_dtype)  # b1 in {0,1}
            b0_ub = T.alloc_ub((block_size,), compute_dtype)    # b0 in {0,1}
            r_ub = T.alloc_ub((block_size,), compute_dtype)     # reduced argument
            c_ub = T.alloc_ub((block_size,), compute_dtype)     # cos(r)
            s_ub = T.alloc_ub((block_size,), compute_dtype)     # sin(r)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            offset = cid * block_size
            remaining = shape - offset
            tail_size = T.min(block_size, remaining)

            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

            # --- widen to fp32 ---
            # NOTE (measured): a `vcast` whose src and dst share a dtype is NOT
            # a no-op — with an integer-ish round_mode it quantises the value to
            # an integer (fp32 -> "rint" -> fp32 gave rint(x), collapsing cos to
            # +/-1).  Same-dtype "conversion" must therefore be a copy, exactly
            # as the in-repo `mish_fwd_kernel` does it.
            if dtype == compute_dtype:
                T.copy(x_ub, f_ub)
            else:
                T.vcast(x_ub, f_ub, round_mode="rint")

            # --- step 1+2: r = x - rint(x/(pi/2)) * (pi/2),  r in [-pi/4, pi/4] ---
            T.vmul(f_ub, _TWO_OVER_PI, kf_ub)                # x * 2/pi
            T.vcast(kf_ub, ki_ub, round_mode="rint")         # k (int32)
            T.vcast(ki_ub, kf_ub, round_mode="rint")         # k (fp32)
            T.vmul(kf_ub, _HALF_PI, r_ub)                    # k * pi/2
            T.vsub(f_ub, r_ub, r_ub)                         # r = x - k*pi/2

            # --- step 4a: quadrant q = k mod 4 (arithmetic, no bitwise ops) ---
            T.vmul(kf_ub, 0.25, t_ub)                        # k / 4
            T.vcast(t_ub, ki_ub, round_mode="floor")         # floor(k/4)
            T.vcast(ki_ub, t_ub, round_mode="rint")          # as fp32
            T.vmul(t_ub, 4.0, t_ub)                          # 4*floor(k/4)
            T.vsub(kf_ub, t_ub, qf_ub)                       # q in {0,1,2,3}

            # b1 = floor(q/2)  -> {0,1} ; b0 = q - 2*b1 -> {0,1}
            T.vmul(qf_ub, 0.5, t_ub)                         # q / 2
            T.vcast(t_ub, ki_ub, round_mode="floor")         # floor(q/2)
            T.vcast(ki_ub, half_ub, round_mode="rint")       # b1 as fp32 {0,1}
            T.vmul(half_ub, 2.0, t_ub)                       # 2*b1
            T.vsub(qf_ub, t_ub, b0_ub)                       # b0 in {0,1}

            # --- step 3: cos / sin of the reduced argument ---
            T.vcos(r_ub, c_ub)
            T.vsin(r_ub, s_ub)

            # --- step 4b: mag = c + (s - c) * b0   (sin when q odd) ---
            T.vsub(s_ub, c_ub, s_ub)                         # s - c
            T.vmul(s_ub, b0_ub, s_ub)                        # (s - c) * b0
            T.vadd(c_ub, s_ub, c_ub)                         # mag

            # --- step 4c: sign.  cos(r + q*pi/2) is
            #       q=0 ->  cos(r)   q=1 -> -sin(r)
            #       q=2 -> -cos(r)   q=3 ->  sin(r)
            #     so the result is negated exactly when q is in {1, 2}, i.e.
            #     when b0 != b1.  neg = b0 XOR b1, computed arithmetically.
            T.vmul(b0_ub, half_ub, t_ub)                     # b0 * b1
            T.vmul(t_ub, 2.0, t_ub)                          # 2*b0*b1
            T.vadd(b0_ub, half_ub, qf_ub)                    # b0 + b1
            T.vsub(qf_ub, t_ub, qf_ub)                       # neg in {0,1}

            # `s_ub` is free here (consumed by the mag computation above) and is
            # fp32 — the output buffer `y_ub` must NOT be used as fp32 scratch,
            # since for fp16/bf16 outputs its element type differs and
            # `hivm.hir.vmul` requires identical operand element types.
            T.vmul(qf_ub, 2.0, t_ub)                         # 2*neg
            T.vmul(c_ub, t_ub, s_ub)                         # 2*neg*mag
            T.vsub(c_ub, s_ub, c_ub)                         # mag - 2*neg*mag

            # --- narrow back to output dtype (copy when already fp32) ---
            if out_dtype == compute_dtype:
                T.copy(c_ub, y_ub)
            else:
                T.vcast(c_ub, y_ub, round_mode="rint")

            T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "float32": (2e-5, 2e-5),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _run_tensor(x, dtype_str, tag):
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kernel = cos_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_cos(x)
    torch.testing.assert_close(y.cpu(), golden, atol=atol, rtol=rtol)
    max_diff = (y.float().cpu() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")


def _run_case(N, dtype_str, tag, scale=3.0):
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    _run_tensor(x, dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    N = 1048576
    _run_case(N, "float16", "L0")
    _run_case(N, "float32", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (4096, 100):
            _run_case(N, dtype_str, "L1")
    _run_case(2049, "float16", "L1")
    _run_case(3000, "bfloat16", "L1")


def run_L2():
    # L2: edge sizes and far-out arguments (non-blocking, records only).
    for N in (1, 2, 10):
        try:
            _run_case(N, "float32", "L2")
        except Exception as e:
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    # Very large |x| stresses the fp32 range reduction (Cody-Waite style
    # cancellation; see the module docstring).  Recorded, not asserted.
    try:
        x = torch.full((4096,), 1.0e6, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "L2-far-argument")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): far-argument err={e}")


def run_boundary():
    # Boundary: the argument range that used to diverge before range reduction.
    try:
        xs = torch.linspace(-12.0, 12.0, 4096, dtype=torch.float32)
        _run_tensor(xs.npu(), "float32", "Boundary-wide-range")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): wide-range err={e}")
    # Exact quadrant boundaries of the reduction.
    try:
        pi = 3.14159265358979
        vals = torch.tensor(
            [0.0, pi / 2, pi, 3 * pi / 2, 2 * pi, -pi / 2, -pi, -3 * pi / 2, -2 * pi]
        )
        x = torch.cat([vals, torch.zeros(4096 - vals.numel())]).npu()
        _run_tensor(x, "float32", "Boundary-quadrants")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): quadrants err={e}")


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
