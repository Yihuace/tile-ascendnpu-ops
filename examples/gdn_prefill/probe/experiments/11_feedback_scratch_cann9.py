"""CANN 9.0: is the rejection caused by reading back a tensor that was stored to?

K2 in 10_feedback_and_acc_cann9.py stored into an ``out_idx`` tensor and then read
that same tensor back as a matmul operand -> "'hivm.hir.store' op only support
store ub to gm currently!".

Here the round-trip goes through a **separate scratch tensor passed as an input**
(not an out_idx), which is how GDN's C2 phase would carry the running product:

  step1: t = S @ rhs        -> scratch   (PIPE_FIX store)
  step2: t = S @ scratch    -> scratch   (PIPE_FIX store, re-read after)
  step3: t = S @ scratch    -> out       (PIPE_FIX store)

Expected: out == S^3 @ rhs.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C, DV = 16, 16


@tilelang.jit(out_idx=[3], target="npuir")
def build():
    @T.prim_func
    def main(sn: T.Tensor((1, C, C), "float16"), rh: T.Tensor((1, C, DV), "float16"),
             scratch: T.Tensor((1, C, DV), "float32"), out: T.Tensor((1, C, DV), "float32")):
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

                T.gemm(l1s, l1a, l0a, size=[C, C, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0a, scratch[0, 0:C, 0:DV])
                T.copy(scratch[0, 0:C, 0:DV], l1b)

                T.gemm(l1s, l1b, l0b, size=[C, C, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0b, scratch[0, 0:C, 0:DV])
                T.copy(scratch[0, 0:C, 0:DV], l1c)

                T.gemm(l1s, l1c, l0c, size=[C, C, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0c, out[0, 0:C, 0:DV])

    return main


torch.manual_seed(0)
Sns = torch.tril(torch.randn(C, C), -1).half()
rhs = torch.randn(C, DV).half()
scratch = torch.zeros(1, C, DV, dtype=torch.float32).npu()

try:
    out = build()(Sns[None].npu(), rhs[None].npu(), scratch)
    torch.npu.synchronize()
    S = Sns.float()
    ref = S @ S @ S @ rhs.float()
    print(f"K3 scratch feedback OK   err={float((out[0].cpu() - ref).abs().max()):.3e}")
except Exception as exc:  # noqa: BLE001
    print("K3 scratch feedback FAIL")
    print(str(exc))
