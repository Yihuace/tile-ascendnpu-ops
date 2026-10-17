"""CANN 9.0 re-probe of the two Cube-side limitations that blocked GDN C2 (Neumann).

From EXPERT_FINDINGS.md §10 (measured on CANN 8.5.2):
  L1. ``T.gemm(..., initC=False)`` does not accumulate across repeated gemms.
  L2. an L0C -> GM -> L1 round-trip (matmul output fed back as a matmul operand,
      loop-carried) gives wrong numbers (err ~193) even with explicit sync.

Both are tested here on 9.0 with the store discipline discovered in 08/09:
  - one L0C per gemm *result* that gets read out (no store-then-reuse);
  - every L0C -> GM store inside ``with T.rs("PIPE_FIX")``.

K1 (accumulation): M = I; then three ``gemm(P, I, M, initC=False)``  -> expect I + 3P.
K2 (feedback):     t = S@t three times, each via a GM round-trip   -> expect S^3@rhs.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C, DV = 16, 16


@tilelang.jit(out_idx=[2], target="npuir")
def build_k1():
    @T.prim_func
    def main(p: T.Tensor((1, C, C), "float16"), eye: T.Tensor((1, C, C), "float16"),
             m: T.Tensor((1, C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1p = T.alloc_L1([C, C], "float16")
                l1i = T.alloc_L1([C, C], "float16")
                l0m = T.alloc_L0C([C, C], "float32")
                T.copy(p[0, 0:C, 0:C], l1p)
                T.copy(eye[0, 0:C, 0:C], l1i)
                T.gemm(l1i, l1i, l0m, size=[C, C, C], initC=True)      # M = I
                T.gemm(l1p, l1i, l0m, size=[C, C, C], initC=False)     # M += P
                T.gemm(l1p, l1i, l0m, size=[C, C, C], initC=False)     # M += P
                T.gemm(l1p, l1i, l0m, size=[C, C, C], initC=False)     # M += P
                with T.rs("PIPE_FIX"):
                    T.copy(l0m, m[0, 0:C, 0:C])

    return main


@tilelang.jit(out_idx=[1, 2, 3], target="npuir")
def build_k2():
    @T.prim_func
    def main(sn: T.Tensor((1, C, C), "float16"), rh: T.Tensor((1, C, DV), "float16"),
             t1: T.Tensor((1, C, DV), "float32"), t2: T.Tensor((1, C, DV), "float32"),
             t3: T.Tensor((1, C, DV), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1s = T.alloc_L1([C, C], "float16")
                l1a = T.alloc_L1([C, DV], "float16")
                l1b = T.alloc_L1([C, DV], "float16")
                l1c = T.alloc_L1([C, DV], "float16")
                l0a = T.alloc_L0C([C, DV], "float32")
                l0b = T.alloc_L0C([C, DV], "float32")
                l0c = T.alloc_L0C([C, DV], "float32")
                T.copy(sn[0, 0:C, 0:C], l1s)
                T.copy(rh[0, 0:C, 0:DV], l1a)

                T.gemm(l1s, l1a, l0a, size=[C, C, DV], initC=True)     # t1 = S@rhs
                with T.rs("PIPE_FIX"):
                    T.copy(l0a, t1[0, 0:C, 0:DV])
                T.copy(t1[0, 0:C, 0:DV], l1b)                          # GM -> L1

                T.gemm(l1s, l1b, l0b, size=[C, C, DV], initC=True)     # t2 = S@t1
                with T.rs("PIPE_FIX"):
                    T.copy(l0b, t2[0, 0:C, 0:DV])
                T.copy(t2[0, 0:C, 0:DV], l1c)

                T.gemm(l1s, l1c, l0c, size=[C, C, DV], initC=True)     # t3 = S@t2
                with T.rs("PIPE_FIX"):
                    T.copy(l0c, t3[0, 0:C, 0:DV])

    return main


torch.manual_seed(0)
Sns = torch.tril(torch.randn(C, C), -1).half()
eyes = torch.eye(C).half()
rhs = torch.randn(C, DV).half()

# ---- K1: initC=False accumulation -----------------------------------------
try:
    m = build_k1()(Sns[None].npu(), eyes[None].npu())
    torch.npu.synchronize()
    ref_m = torch.eye(C) + 3 * Sns.float()
    print(f"K1 accum   OK   err={float((m[0].cpu() - ref_m).abs().max()):.3e}")
except Exception as exc:  # noqa: BLE001
    print("K1 accum   FAIL", str(exc).replace("\n", " ")[:110])

# ---- K2: L0C -> GM -> L1 feedback -----------------------------------------
try:
    t1, t2, t3 = build_k2()(Sns[None].npu(), rhs[None].npu())
    torch.npu.synchronize()
    S = Sns.float()
    r = rhs.float()
    for name, got, ref in (("t1", t1, S @ r), ("t2", t2, S @ S @ r), ("t3", t3, S @ S @ S @ r)):
        print(f"K2 {name}     OK   err={float((got[0].cpu() - ref).abs().max()):.3e}")
except Exception as exc:  # noqa: BLE001
    print("K2 feedback FAIL")
    print(str(exc))
