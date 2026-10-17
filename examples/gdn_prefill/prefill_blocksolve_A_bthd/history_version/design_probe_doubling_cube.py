"""Design probe (Stage 1, D-3): log-doubling triangular inverse on the Cube.

Target identity (N strictly lower 64x64, hence N^64 = 0):

    A = (I + S)^{-1} = sum_{j=0}^{63} N^j = prod_{i=0..5} (I + N^{2^i}),   N = -S

Recursion used by the design (T_0 = I, P_j = N^{2^j}):

    T_{j+1} = T_j + P_j T_j  ==  gemm(P_j, T_j, L0C, initC=False)   (L0C already holds T_j)

so the "+T_j" costs no extra op (``initC=False`` adds into the live accumulator) and the
whole inverse is 11 gemms on fp32 (HF32) operands with GM scratch feedback.

This probe runs **two** levels (T_2 = I+N+N^2+N^3) plus the squaring P_1 = N^2, on one
64x64 tile, and compares against an fp64 torch reference.  A second build times a
repeated 6-op level sequence so the design's roofline uses a measured constant.

Measured (2026-10-17, CANN 9.0.0 / tilelang dev root / Ascend910B3):
    T1 err ~3e-8,  T2 err ~1e-7  (fp32 chain),  per 6-op level ~1.5 us  (~0.26 us/op)

Run:  python design_probe_doubling_cube.py
"""

import os
import time

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 64
DT = "float32"


@tilelang.jit(out_idx=[5, 6], target="npuir")
def build_correct():
    @T.prim_func
    def main(
        n: T.Tensor((1, C, C), DT),
        eye: T.Tensor((1, C, C), DT),
        ws1: T.Tensor((1, C, C), DT),
        ws2: T.Tensor((1, C, C), DT),
        ws3: T.Tensor((1, C, C), DT),
        out_t1: T.Tensor((1, C, C), DT),
        out_t2: T.Tensor((1, C, C), DT),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1n = T.alloc_L1([C, C], DT)
                l1e = T.alloc_L1([C, C], DT)
                l1t = T.alloc_L1([C, C], DT)
                l1p = T.alloc_L1([C, C], DT)
                l1q = T.alloc_L1([C, C], DT)
                l0t = T.alloc_L0C([C, C], "float32")
                l0p = T.alloc_L0C([C, C], "float32")

                T.copy(n[0, 0:C, 0:C], l1n)
                T.copy(eye[0, 0:C, 0:C], l1e)

                # T_0 = I (L0C is seeded by a gemm: the only way an L0C gets a value)
                T.gemm(l1e, l1e, l0t, size=[C, C, C], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0t, ws1[0, 0:C, 0:C])
                T.copy(ws1[0, 0:C, 0:C], l1t)

                # level 0: T_1 = T_0 + N T_0 = I + N     (P_0 = N)
                T.gemm(l1n, l1t, l0t, size=[C, C, C], initC=False)
                with T.rs("PIPE_FIX"):
                    T.copy(l0t, ws2[0, 0:C, 0:C])  # T_1 (scratch, NOT an out_idx tensor)
                with T.rs("PIPE_FIX"):
                    T.copy(l0t, out_t1[0, 0:C, 0:C])

                # P_1 = N^2 (separate L1 operands: one buffer may not feed both slots)
                T.copy(n[0, 0:C, 0:C], l1p)
                T.copy(n[0, 0:C, 0:C], l1q)
                T.gemm(l1p, l1q, l0p, size=[C, C, C], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0p, ws3[0, 0:C, 0:C])

                # level 1: T_2 = T_1 + N^2 T_1
                T.copy(ws3[0, 0:C, 0:C], l1p)
                T.copy(ws2[0, 0:C, 0:C], l1t)
                T.gemm(l1p, l1t, l0t, size=[C, C, C], initC=False)
                with T.rs("PIPE_FIX"):
                    T.copy(l0t, out_t2[0, 0:C, 0:C])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_timing(reps: int):
    """Same 6-op level shape (gemm, store, load, gemm, store, load), repeated."""

    @T.prim_func
    def main(n: T.Tensor((1, C, C), DT), out: T.Tensor((1, C, C), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1a = T.alloc_L1([C, C], DT)
                l1b = T.alloc_L1([C, C], DT)
                l0a = T.alloc_L0C([C, C], "float32")
                l0b = T.alloc_L0C([C, C], "float32")
                T.copy(n[0, 0:C, 0:C], l1a)
                T.copy(n[0, 0:C, 0:C], l1b)
                for _ in T.serial(reps):
                    T.gemm(l1a, l1b, l0a, size=[C, C, C], initC=True)
                    with T.rs("PIPE_FIX"):
                        T.copy(l0a, out[0, 0:C, 0:C])
                    T.copy(out[0, 0:C, 0:C], l1b)
                    T.gemm(l1b, l1a, l0b, size=[C, C, C], initC=True)
                    with T.rs("PIPE_FIX"):
                        T.copy(l0b, out[0, 0:C, 0:C])
                    T.copy(out[0, 0:C, 0:C], l1a)

    return main


torch.manual_seed(0)
N = torch.tril(torch.randn(C, C, dtype=torch.float64), -1) * 0.25
EYE = torch.eye(C, dtype=torch.float64)
T1_ref = EYE + N
T2_ref = (EYE + N @ N) @ (EYE + N)

n_npu = N.float()[None].npu()
eye_npu = EYE.float()[None].npu()
ws = [torch.zeros(1, C, C, dtype=torch.float32).npu() for _ in range(3)]

try:
    t1, t2 = build_correct()(n_npu, eye_npu, *ws)
    torch.npu.synchronize()
    e1 = float((t1[0].cpu().double() - T1_ref).abs().max())
    e2 = float((t2[0].cpu().double() - T2_ref).abs().max())
    print(f"PROBE-CORRECT T1=I+N          max_abs_err={e1:.3e}  |ref|max={float(T1_ref.abs().max()):.3f}")
    print(f"PROBE-CORRECT T2=(I+N^2)(I+N) max_abs_err={e2:.3e}  |ref|max={float(T2_ref.abs().max()):.3f}")
except Exception as exc:  # noqa: BLE001
    print("PROBE-CORRECT FAIL")
    print(str(exc)[:1200])

try:
    r_small, r_big = 2, 202
    k_small = build_timing(r_small)
    k_big = build_timing(r_big)
    torch.npu.synchronize()

    def _time(k, iters=20):
        for _ in range(3):
            k(n_npu)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            k(n_npu)
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / iters

    t_small = _time(k_small)
    t_big = _time(k_big)
    per_level = (t_big - t_small) / (r_big - r_small)
    print(f"PROBE-TIMING kernel(reps={r_small})={t_small * 1e6:.1f}us  kernel(reps={r_big})={t_big * 1e6:.1f}us")
    print(f"PROBE-TIMING per 6-op level (gemm+store+load x2, 64x64 f32) = {per_level * 1e6:.3f}us")
except Exception as exc:  # noqa: BLE001
    print("PROBE-TIMING FAIL")
    print(str(exc)[:1200])
