"""Mirror the exact sync skeleton of the delivered (9.0-working) conv3d MixCV kernel.

conv3d/_conv3d_kernel.py uses:
    Cube  : with T.rs("PIPE_FIX"): wait(2*slot+1); <store>; set(2*slot)
    Vector: with T.rs("PIPE_MTE2"): set(1); set(3)          # priming, before the loop
            with T.rs("PIPE_MTE2"): wait(2*slot); <load>
            with T.rs("PIPE_MTE3"): <store>; set(2*slot+1)
and both scopes live in @T.macro helpers called from the T.Kernel body.

The earlier probes 14/15/16 used "Cube set-first, Vector wait-first", which fails with
'multiple entry functions ... main_mix_aic'.  This probe keeps the conv3d ordering.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 16


@tilelang.jit(out_idx=[2], target="npuir")
def build(nk):
    @T.macro
    def _cube(x, ws, kernel_id):
        with T.Scope("Cube"):
            l1 = T.alloc_L1([C, C], "float16")
            l0 = T.alloc_L0C([C, C], "float32")
            T.copy(x[kernel_id, 0:C, 0:C], l1)
            T.gemm(l1, l1, l0, size=[C, C, C], b_transpose=True, initC=True)
            with T.rs("PIPE_FIX"):
                T.sync_block_wait(2 * 0 + 1)
                T.copy(l0, ws[kernel_id, 0:C, 0:C])
                T.sync_block_set(2 * 0)

    @T.macro
    def _epilogue(ws, out, kernel_id):
        with T.Scope("Vector"):
            ub16 = T.alloc_ub((C, C), "float16")
            ub32 = T.alloc_ub((C, C), "float32")
            with T.rs("PIPE_MTE2"):
                T.sync_block_set(1)
            with T.rs("PIPE_MTE2"):
                T.sync_block_wait(2 * 0)
                T.copy(ws[kernel_id, 0:C, 0:C], ub16)
            T.vcast(ub16, ub32)
            T.vmul(ub32, 2.0, ub32)
            T.vcast(ub32, ub16)
            with T.rs("PIPE_MTE3"):
                T.copy(ub16, out[kernel_id, 0:C, 0:C])
                T.sync_block_set(2 * 0 + 1)

    @T.prim_func
    def main(x: T.Tensor((nk, C, C), "float16"), ws: T.Tensor((nk, C, C), "float16"),
             out: T.Tensor((nk, C, C), "float16")):
        with T.Kernel(nk, is_npu=True) as (kernel_id, _):
            _cube(x, ws, kernel_id)
            _epilogue(ws, out, kernel_id)

    return main


torch.manual_seed(0)
nk = 4
x = torch.randn(nk, C, C, dtype=torch.float16).npu()
ws = torch.zeros(nk, C, C, dtype=torch.float16).npu()
try:
    out = build(nk)(x, ws)
    torch.npu.synchronize()
    xf = x.cpu().float()
    ref = (2 * (xf @ xf.transpose(1, 2))).half()
    print(f"conv3d-skeleton nk={nk}  OK   err={float((out.cpu().float() - ref.float()).abs().max()):.3e}")
except Exception as exc:  # noqa: BLE001
    msg = str(exc)
    key = ("multiple entry functions" if "multiple entry functions" in msg
           else msg.replace("\n", " ")[:110])
    print(f"conv3d-skeleton nk={nk}  FAIL {key}")
