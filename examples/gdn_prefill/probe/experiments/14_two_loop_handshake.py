"""R0 probe: can Cube and Vector each run their own chunk loop with per-chunk
bidirectional handshakes, inside ONE kernel on CANN 9.0?

Structure under test (the one DESIGN.md §5 relies on):

    Cube  loop : for ck: produce -> set(0);  wait(1); consume
    Vector loop: for ck: wait(0); consume -> set(1)

Per chunk the Cube side computes G = x@x^T, the Vector side multiplies the stored
matrix by 2 and writes it back, and the Cube side consumes it again:

    out[ck] = (2G) @ (2G)^T      with G = x@x^T

Two chunks are used, and the chunk index is a real ``T.serial`` loop variable
(that is what the real kernel needs -- S/chunk is dynamic).
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

NCH, C = 2, 16


@tilelang.jit(out_idx=[2], target="npuir")
def build():
    @T.prim_func
    def main(x: T.Tensor((NCH, C, C), "float16"), ws: T.Tensor((NCH, C, C), "float16"),
             out: T.Tensor((NCH, C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1 = T.alloc_L1([C, C], "float16")
                lb = T.alloc_L1([C, C], "float16")
                l0a = T.alloc_L0C([C, C], "float32")
                l0b = T.alloc_L0C([C, C], "float32")
                for ck in T.serial(NCH):
                    T.copy(x[ck, 0:C, 0:C], l1)
                    T.gemm(l1, l1, l0a, size=[C, C, C], b_transpose=True, initC=True)
                    with T.rs("PIPE_FIX"):
                        T.copy(l0a, ws[ck, 0:C, 0:C])          # fixpipe fp32 -> fp16
                        T.sync_block_set(0)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(1)
                        T.copy(ws[ck, 0:C, 0:C], lb)           # consume vector result
                    T.gemm(lb, lb, l0b, size=[C, C, C], b_transpose=True, initC=True)
                    with T.rs("PIPE_FIX"):
                        T.copy(l0b, out[ck, 0:C, 0:C])

            with T.Scope("Vector"):
                ub16 = T.alloc_ub((C, C), "float16")
                ub32 = T.alloc_ub((C, C), "float32")
                for ck in T.serial(NCH):
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(0)
                        T.copy(ws[ck, 0:C, 0:C], ub16)
                    T.vcast(ub16, ub32)
                    T.vmul(ub32, 2.0, ub32)
                    T.vcast(ub32, ub16)
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub16, ws[ck, 0:C, 0:C])
                        T.sync_block_set(1)

    return main


torch.manual_seed(0)
x = torch.randn(NCH, C, C, dtype=torch.float16).npu()
ws = torch.zeros(NCH, C, C, dtype=torch.float16).npu()

try:
    out = build()(x, ws)
    torch.npu.synchronize()
    xf = x.cpu().float()
    G = xf @ xf.transpose(1, 2)                      # fp32 gemm result
    G16 = G.half().float()                            # stored to ws as fp16 (fixpipe)
    V = (2 * G16).half().float()                      # vector: *2 then cast to fp16
    ref16 = V @ V.transpose(1, 2)                     # expected kernel result
    ref = (2 * G) @ (2 * G).transpose(1, 2)            # naive all-fp32 reference
    got = out.cpu()
    print(f"R0 two-loop handshake OK   err(vs fp16-path ref)={float((got - ref16).abs().max()):.3e}  "
          f"err(vs all-fp32 ref)={float((got - ref).abs().max()):.3e}")
except Exception as exc:  # noqa: BLE001
    print("R0 two-loop handshake FAIL")
    print(str(exc))
