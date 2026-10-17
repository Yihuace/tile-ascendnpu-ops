"""Does the repo's working MixCV form (Cube/Vector as @T.macro) compile on CANN 9.0?

The delivered conv3d/conv1d/ssd kernels (which pass on 9.0) structure MixCV as:

    @T.macro
    def _cube(...):
        with T.Scope("Cube"): ...

    @T.macro
    def _epilogue(...):
        with T.Scope("Vector"): ...

    @T.prim_func
    def main(...):
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
            _cube(...)
            _epilogue(...)

while placing `with T.Scope(...)` inline (probe 02 / 15) fails with
"multiple entry functions detected".

This probe reproduces the macro form with 1 and with several grid blocks.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 16


@tilelang.jit(out_idx=[2], target="npuir")
def build(nblocks):
    @T.macro
    def _cube(x, ws, cid):
        with T.Scope("Cube"):
            l1 = T.alloc_L1([C, C], "float16")
            l0 = T.alloc_L0C([C, C], "float32")
            T.copy(x[cid, 0:C, 0:C], l1)
            T.gemm(l1, l1, l0, size=[C, C, C], b_transpose=True, initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(l0, ws[cid, 0:C, 0:C])
                T.sync_block_set(0)

    @T.macro
    def _epilogue(ws, out, cid):
        with T.Scope("Vector"):
            ub = T.alloc_ub((C, C), "float32")
            with T.rs("PIPE_MTE2"):
                T.sync_block_wait(0)
                T.copy(ws[cid, 0:C, 0:C], ub)
            T.vmul(ub, 2.0, ub)
            with T.rs("PIPE_MTE3"):
                T.copy(ub, out[cid, 0:C, 0:C])

    @T.prim_func
    def main(x: T.Tensor((nblocks, C, C), "float16"), ws: T.Tensor((nblocks, C, C), "float32"),
             out: T.Tensor((nblocks, C, C), "float32")):
        with T.Kernel(nblocks, is_npu=True) as (cid, _):
            _cube(x, ws, cid)
            _epilogue(ws, out, cid)

    return main


torch.manual_seed(0)
for nb in (1, 4):
    x = torch.randn(nb, C, C, dtype=torch.float16).npu()
    ws = torch.zeros(nb, C, C, dtype=torch.float32).npu()
    try:
        out = build(nb)(x, ws)
        torch.npu.synchronize()
        xf = x.cpu().float()
        ref = 2 * (xf @ xf.transpose(1, 2))
        print(f"macro form nblocks={nb}  OK   err={float((out.cpu() - ref).abs().max()):.3e}")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        key = ("multiple entry functions" if "multiple entry functions" in msg
               else msg.replace("\n", " ")[:90])
        print(f"macro form nblocks={nb}  FAIL {key}")
