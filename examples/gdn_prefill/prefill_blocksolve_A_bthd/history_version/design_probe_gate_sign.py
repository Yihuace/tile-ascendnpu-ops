"""Design probe #4 (Stage 1 revision 1, D-3): the two executable-level fixes.

REVIEW.md blocking issue 1 (broadcast direction) and issue 2 (lost minus sign) live
in the Vector phase; both were invisible to ``verify_equiv.py`` because that script
only re-derives the algebra.  This probe runs the **literal §3.3 op sequence** on
the device, so a wrong direction or a missing sign shows up as a numeric failure:

    K1  g column -> (64,1) UB, same GM region -> (1,64) UB   (two-load form, no T.transpose)
        gi = vbrc(g_col)   gj = vbrc(g_row)   d = vsub(gi, gj)   e = vexp(d)
        b32 = vcast(b16);  bn = vsub(zero, b32)          <-- the "-1" landing point
        t1 = vmul(vbrc(bn), e);  t2 = vmul(t1, G);  N = vselect(i>j, t2, 0)
        check: e == exp(g_i - g_j)  and  N == -beta_i*exp(g_i-g_j)*G on i>j, else 0

    K2  fp16-operand 2-level doubling chain **with A and B sharing one L1 buffer**
        T1 = I + N ; P = N@N (gemm(l1n, l1n, ...)) ; T2 = T1 + P@T1
        -> both the "can one L1 buffer feed A and B with b_transpose=False" question
           (REVIEW problem 9) and the real error of the fp16 scratch fallback
           (REVIEW problem 5: v0's V4 number came from a probe branch that never ran).

Run:  bash _probe_env.sh python3 design_probe_gate_sign.py
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 64
H = 4
HID = 2


# ---------------------------------------------------------------------------
# K1: Vector-phase literal expansion (direction + sign)
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[], target="npuir")
def build_gate():
    @T.prim_func
    def main(
        g: T.Tensor((1, C, H), "float32"),
        beta: T.Tensor((1, C, H), "float16"),
        G: T.Tensor((1, C, C), "float32"),
        mask01: T.Tensor((C, C), "float32"),
        out_e: T.Tensor((1, C, C), "float32"),
        out_n: T.Tensor((1, C, C), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Vector"):
                g_col = T.alloc_ub([C, 1], "float32")
                g_row = T.alloc_ub([1, C], "float32")
                b16 = T.alloc_ub([C, 1], "float16")
                b32 = T.alloc_ub([C, 1], "float32")
                bn = T.alloc_ub([C, 1], "float32")
                A = T.alloc_ub([C, C], "float32")   # gi -> d(clamped)
                B = T.alloc_ub([C, C], "float32")   # gj (spare)
                Cb = T.alloc_ub([C, C], "float32")  # e
                D = T.alloc_ub([C, C], "float32")   # G
                E = T.alloc_ub([C, C], "float32")   # bm -> t1
                F = T.alloc_ub([C, C], "float32")   # t2 -> N
                MK = T.alloc_ub([C, C], "float32")  # static strict-lower 0/1 tile

                with T.rs("PIPE_MTE2"):
                    T.copy(g[0, 0:C, HID:HID + 1], g_col)   # (64,1) column, strided GM
                    T.copy(g[0, 0:C, HID:HID + 1], g_row)   # same region, (1,64) landing
                    T.copy(beta[0, 0:C, HID:HID + 1], b16)
                    T.copy(G[0, 0:C, 0:C], D)
                    T.copy(mask01[0:C, 0:C], MK)

                T.vcast(b16, b32, round_mode="rint")
                T.vmul(b32, T.float32(-1.0), bn)         # bn = -beta  (issue 2 landing point)
                T.vbrc(g_col, A)                         # A[i,j] = g_i
                T.vbrc(g_row, B)                         # B[i,j] = g_j
                T.vsub(A, B, A)                          # A = g_i - g_j  (issue 1)
                T.vmin(A, T.float32(0.0), A)             # bound the discarded region (no inf)
                T.vexp(A, Cb)                            # Cb = exp(min(g_i-g_j, 0))
                T.vbrc(bn, E)                            # E = -beta_i
                T.vmul(E, Cb, E)                         # E = -beta_i * e
                T.vmul(E, D, F)                          # F = -beta_i * e * G
                T.vmul(F, MK, F)                         # F = N (strict lower, exactly 0 elsewhere)
                with T.rs("PIPE_MTE3"):
                    T.copy(Cb, out_e[0, 0:C, 0:C])
                    T.copy(F, out_n[0, 0:C, 0:C])

    return main


# ---------------------------------------------------------------------------
# K2: 2-level doubling chain, fp16 vs fp32 operands, A/B sharing one L1 buffer
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[], target="npuir")
def build_chain(dt: str):
    @T.prim_func
    def main(
        n: T.Tensor((1, C, C), dt),
        eye: T.Tensor((1, C, C), dt),
        ws: T.Tensor((1, C, C), dt),
        out: T.Tensor((1, C, C), dt),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1n = T.alloc_L1([C, C], dt)
                l1i = T.alloc_L1([C, C], dt)
                l1t = T.alloc_L1([C, C], dt)
                l1p = T.alloc_L1([C, C], dt)
                l0t = T.alloc_L0C([C, C], "float32")
                l0p = T.alloc_L0C([C, C], "float32")

                T.copy(n[0, 0:C, 0:C], l1n)
                T.copy(eye[0, 0:C, 0:C], l1i)
                T.gemm(l1i, l1i, l0t, size=[C, C, C], initC=True)      # T_0 = I
                T.gemm(l1n, l1i, l0t, size=[C, C, C], initC=False)     # T_1 = I + N
                with T.rs("PIPE_FIX"):
                    T.copy(l0t, ws[0, 0:C, 0:C])                       # f32 -> dt
                T.copy(ws[0, 0:C, 0:C], l1t)
                # one L1 buffer as BOTH gemm operands (b_transpose=False)
                T.gemm(l1n, l1n, l0p, size=[C, C, C], initC=True)      # P_1 = N^2
                with T.rs("PIPE_FIX"):
                    T.copy(l0p, ws[0, 0:C, 0:C])
                T.copy(ws[0, 0:C, 0:C], l1p)
                T.gemm(l1p, l1t, l0t, size=[C, C, C], initC=False)     # T_2 = T_1 + N^2 T_1
                with T.rs("PIPE_FIX"):
                    T.copy(l0t, out[0, 0:C, 0:C])

    return main


torch.manual_seed(0)
g64 = torch.cumsum(-torch.rand(C, dtype=torch.float64) - 0.01, 0)
beta64 = torch.rand(C, dtype=torch.float64) * 0.9
kk = torch.nn.functional.normalize(torch.randn(C, 64, dtype=torch.float64), dim=-1)
G64 = kk @ kk.T
mask = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)

g_npu = torch.stack([g64.float()] * H, dim=-1)[None].npu()            # [1,64,H]
beta_npu = torch.stack([beta64.float().half()] * H, dim=-1)[None].npu()
G_npu = G64.float()[None].npu()
out_e = torch.zeros(1, C, C, dtype=torch.float32).npu()
out_n = torch.zeros(1, C, C, dtype=torch.float32).npu()
mask01 = torch.tril(torch.ones(C, C, dtype=torch.float32), -1)

e_ref = torch.exp(g64[:, None] - g64[None, :])
wrong_dir = torch.exp(g64[None, :] - g64[:, None])                  # exp(g_j - g_i)
n_ref = torch.where(mask, -beta64.half().double()[:, None] * e_ref * G64,
                    torch.zeros((), dtype=torch.float64))
N64 = -torch.where(mask, beta64.half().double()[:, None] * e_ref * G64,
                   torch.zeros((), dtype=torch.float64))
T2_ref = (torch.eye(C, dtype=torch.float64) + N64 @ N64) @ (torch.eye(C, dtype=torch.float64) + N64)

try:
    build_gate()(g_npu, beta_npu, G_npu, mask01.npu(), out_e, out_n)
    torch.npu.synchronize()
    e_got = out_e[0].cpu().double()
    n_got = out_n[0].cpu().double()
    lo = mask
    e_ref_use = torch.exp(torch.minimum(g64[:, None] - g64[None, :],
                                        torch.zeros((), dtype=torch.float64)))
    dl = float((e_got - e_ref_use)[lo].abs().max())
    dw = float((e_got - torch.exp(g64[None, :] - g64[:, None]))[lo].abs().max())
    print(f"PROBE-GATE  e == exp(g_i-g_j)  (i>j region) : max_abs_err={dl:.3e}")
    print(f"PROBE-GATE  e vs exp(g_j-g_i)  (i>j region) : max_abs_err={dw:.3e}  "
          f"(must be LARGE -> direction is g_i-g_j)")
    print(f"PROBE-GATE  N == -b_i*e*G (i>j)    : max_abs_err={float((n_got - n_ref).abs().max()):.3e}  "
          f"|ref|max={float(n_ref.abs().max()):.3f}")
    print(f"PROBE-GATE  N vs +b_i*e*G (no sign): max_abs_err={float((n_got + n_ref).abs().max()):.3e}  "
          f"(must be LARGE -> the -1 is really applied)")
except Exception as exc:  # noqa: BLE001
    print("PROBE-GATE FAIL")
    print(str(exc)[-1800:])

for dt in ("float16", "float32"):
    try:
        n_npu = N64.float().to(getattr(torch, dt))[None].npu()
        eye_npu = torch.eye(C, dtype=torch.float64).float().to(getattr(torch, dt))[None].npu()
        ws = torch.zeros(1, C, C, dtype=getattr(torch, dt)).npu()
        out = torch.zeros(1, C, C, dtype=getattr(torch, dt)).npu()
        build_chain(dt)(n_npu, eye_npu, ws, out)
        torch.npu.synchronize()
        got = out[0].cpu().double()
        rel = float((got - T2_ref).abs().max()) / float(T2_ref.abs().max())
        print(f"PROBE-CHAIN {dt:8s} T2=(I+N^2)(I+N), A=B shared L1 : max_abs_err={float((got - T2_ref).abs().max()):.3e}  "
              f"rel={rel:.3e}  |ref|max={float(T2_ref.abs().max()):.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE-CHAIN {dt} FAIL")
        print(str(exc)[:1200])
