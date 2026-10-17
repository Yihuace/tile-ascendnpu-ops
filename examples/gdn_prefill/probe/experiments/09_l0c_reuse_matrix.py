"""CANN 9.0: workarounds for the "re-used L0C" store rejection.

Known (from 08): A (1 gemm) OK, B (3D slice) OK, C (same L0C used by a 2nd gemm
after a store) FAIL -> "'hivm.hir.store' op only support store ub to gm".

Variants:
  E. two distinct L0C buffers, store after each gemm      (no re-use)
  F. same L0C, two gemms, only the last result stored     (gemm overwrite, no interleaved store)
  G. same L0C, store, then re-gemm, then store, with a PIPE_FIX barrier
     inserted between the store and the re-use
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 16


@tilelang.jit(out_idx=[1, 2], target="npuir")
def build_e():
    @T.prim_func
    def main(k: T.Tensor((C, C), "float16"),
             out1: T.Tensor((C, C), "float32"), out2: T.Tensor((C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bk = T.alloc_L1([C, C], "float16")
                l0a = T.alloc_L0C([C, C], "float32")
                l0b = T.alloc_L0C([C, C], "float32")
                T.copy(k, bk)
                T.gemm(bk, bk, l0a, size=[C, C, C], b_transpose=True, initC=True)
                T.copy(l0a, out1)
                T.gemm(bk, bk, l0b, size=[C, C, C], b_transpose=True, initC=True)
                T.copy(l0b, out2)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_f():
    @T.prim_func
    def main(k: T.Tensor((C, C), "float16"), out: T.Tensor((C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bk = T.alloc_L1([C, C], "float16")
                lcc = T.alloc_L0C([C, C], "float32")
                T.copy(k, bk)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                T.copy(lcc, out)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_g():
    @T.prim_func
    def main(k: T.Tensor((C, C), "float16"), out: T.Tensor((C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bk = T.alloc_L1([C, C], "float16")
                lcc = T.alloc_L0C([C, C], "float32")
                T.copy(k, bk)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(lcc, out)
                with T.rs("PIPE_FIX"):
                    T.copy(bk, bk)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(lcc, out)

    return main


torch.manual_seed(0)
k = torch.randn(C, C, dtype=torch.float16).npu()
ref = k.cpu().float() @ k.cpu().float().T

for name, builder in (("E two distinct L0C", build_e),
                      ("F same L0C, store last only", build_f),
                      ("G barrier between store/reuse", build_g)):
    try:
        outs = builder()(k)
        torch.npu.synchronize()
        if not isinstance(outs, (tuple, list)):
            outs = (outs,)
        errs = [float((o.cpu().float() - ref).abs().max()) for o in outs]
        print(f"{name:30s} OK   err={max(errs):.2e}")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        key = "only support store ub to gm" if "only support store ub to gm" in msg else msg[:110]
        print(f"{name:30s} FAIL {key}")
