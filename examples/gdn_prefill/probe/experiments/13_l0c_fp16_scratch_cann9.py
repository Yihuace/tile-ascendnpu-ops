"""CANN 9.0: keep the matmul-output round-trip entirely inside one Cube scope.

Constraint found in 12: a kernel may contain only ONE Cube scope and ONE Vector
scope ("multiple entry functions detected ... main_mix_aic"), so Cube->Vector->Cube
interleaving is impossible.  And an explicit ``vcast`` inside a Cube scope is
rejected (10/11).

This probe avoids the cast altogether by letting **fixpipe** convert:
  L0C(fp32) --store(fixpipe)--> GM(fp16 scratch) --copy(MTE2)--> L1(fp16)
i.e. the cast happens during the store into a *fp16* scratch tensor.

With a strictly-lower-triangular ``Sn`` this chain is exactly the Neumann term
recurrence:  t_{j+1} = Sn @ t_j  (t_0 = rhs)  ->  out = Sn^3 @ rhs.
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
             ws16: T.Tensor((1, C, DV), "float16"), out: T.Tensor((1, C, DV), "float32")):
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

                T.gemm(l1s, l1a, l0a, size=[C, C, DV], initC=True)      # t1 = Sn@rhs
                with T.rs("PIPE_FIX"):
                    T.copy(l0a, ws16[0, 0:C, 0:DV])                     # fixpipe fp32->fp16
                T.copy(ws16[0, 0:C, 0:DV], l1b)

                T.gemm(l1s, l1b, l0b, size=[C, C, DV], initC=True)      # t2 = Sn@t1
                with T.rs("PIPE_FIX"):
                    T.copy(l0b, ws16[0, 0:C, 0:DV])
                T.copy(ws16[0, 0:C, 0:DV], l1c)

                T.gemm(l1s, l1c, l0c, size=[C, C, DV], initC=True)      # t3 = Sn@t2
                with T.rs("PIPE_FIX"):
                    T.copy(l0c, out[0, 0:C, 0:DV])

    return main


torch.manual_seed(0)
Sns = torch.tril(torch.randn(C, C), -1).half()
rhs = torch.randn(C, DV).half()
ws16 = torch.zeros(1, C, DV, dtype=torch.float16).npu()

try:
    out = build()(Sns[None].npu(), rhs[None].npu(), ws16)
    torch.npu.synchronize()
    S = Sns.float()
    ref = S @ S @ S @ rhs.float()
    print(f"K5 fixpipe-cast feedback OK   err={float((out[0].cpu() - ref).abs().max()):.3e}")
except Exception as exc:  # noqa: BLE001
    print("K5 fixpipe-cast feedback FAIL")
    print(str(exc))
