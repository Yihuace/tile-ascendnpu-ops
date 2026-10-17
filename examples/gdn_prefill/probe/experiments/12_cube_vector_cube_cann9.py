"""CANN 9.0: matmul-output feedback via an explicit Cube -> Vector -> Cube handoff.

10/11 showed that a GM round-trip written entirely inside a Cube scope fails:
the fp32 product must be cast to fp16 before it can be a matmul operand again, and
that cast (``hivm.hir.vcast``) is a **Vector** op -- it cannot live in the Cube
scope, so the store that follows gets mis-lowered and rejected.

This probe splits the round-trip into three explicit scopes:

  Cube   : t1 = S @ rhs          -> ws32 (fp32)            [sync_block_set(0)]
  Vector : ws32 -> ub32 -> vcast -> ub16 -> ws16 (fp16)    [sync_block_set(1)]
  Cube   : t2 = S @ ws16         -> out                    (validates the chain)

Chain t2 = S @ (S @ rhs) may be applied repeatedly to build S^j for Neumann.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C, DV = 16, 16


@tilelang.jit(out_idx=[4], target="npuir")
def build():
    @T.prim_func
    def main(sn: T.Tensor((1, C, C), "float16"), rh: T.Tensor((1, C, DV), "float16"),
             ws32: T.Tensor((1, C, DV), "float32"), ws16: T.Tensor((1, C, DV), "float16"),
             out: T.Tensor((1, C, DV), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1s = T.alloc_L1([C, C], "float16")
                l1a = T.alloc_L1([C, DV], "float16")
                l0a = T.alloc_L0C([C, DV], "float32")
                T.copy(sn[0, 0:C, 0:C], l1s)
                T.copy(rh[0, 0:C, 0:DV], l1a)
                T.gemm(l1s, l1a, l0a, size=[C, C, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0a, ws32[0, 0:C, 0:DV])
                    T.sync_block_set(0)

            with T.Scope("Vector"):
                ub32 = T.alloc_ub((C, DV), "float32")
                ub16 = T.alloc_ub((C, DV), "float16")
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)
                    T.copy(ws32[0, 0:C, 0:DV], ub32)
                T.vcast(ub32, ub16)
                with T.rs("PIPE_MTE3"):
                    T.copy(ub16, ws16[0, 0:C, 0:DV])
                    T.sync_block_set(1)

            with T.Scope("Cube"):
                l1s2 = T.alloc_L1([C, C], "float16")
                l1b = T.alloc_L1([C, DV], "float16")
                l0b = T.alloc_L0C([C, DV], "float32")
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(1)
                    T.copy(ws16[0, 0:C, 0:DV], l1b)
                T.copy(sn[0, 0:C, 0:C], l1s2)
                T.gemm(l1s2, l1b, l0b, size=[C, C, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0b, out[0, 0:C, 0:DV])

    return main


torch.manual_seed(0)
Sns = torch.tril(torch.randn(C, C), -1).half()
rhs = torch.randn(C, DV).half()
ws32 = torch.zeros(1, C, DV, dtype=torch.float32).npu()
ws16 = torch.zeros(1, C, DV, dtype=torch.float16).npu()

try:
    out = build()(Sns[None].npu(), rhs[None].npu(), ws32, ws16)
    torch.npu.synchronize()
    S = Sns.float()
    ref = S @ S @ rhs.float()
    print(f"K4 cube-vector-cube OK   err={float((out[0].cpu() - ref).abs().max()):.3e}")
except Exception as exc:  # noqa: BLE001
    print("K4 cube-vector-cube FAIL")
    print(str(exc))
