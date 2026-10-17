"""Isolate which L0C->GM store form CANN 9.0 accepts.

Variants (all Cube/Expert, C=16), each an explicit prim_func (no closure tricks):
  A. 2-D contiguous output, one gemm, one store          (baseline: known OK)
  B. 3-D output, store into out[0, 0:C, 0:C]             (sliced GM destination)
  C. 2-D output, two gemms re-using the same L0C, store between them
  D. 3-D output + two gemms + PIPE_FIX around each store
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 16


@tilelang.jit(out_idx=[1], target="npuir")
def build_a():
    @T.prim_func
    def main(k: T.Tensor((C, C), "float16"), out: T.Tensor((C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bk = T.alloc_L1([C, C], "float16")
                lcc = T.alloc_L0C([C, C], "float32")
                T.copy(k, bk)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                T.copy(lcc, out)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_b():
    @T.prim_func
    def main(k: T.Tensor((C, C), "float16"), out: T.Tensor((1, C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bk = T.alloc_L1([C, C], "float16")
                lcc = T.alloc_L0C([C, C], "float32")
                T.copy(k, bk)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                T.copy(lcc, out[0, 0:C, 0:C])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_c():
    @T.prim_func
    def main(k: T.Tensor((C, C), "float16"), out: T.Tensor((C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bk = T.alloc_L1([C, C], "float16")
                lcc = T.alloc_L0C([C, C], "float32")
                T.copy(k, bk)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                T.copy(lcc, out)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                T.copy(lcc, out)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_d():
    @T.prim_func
    def main(k: T.Tensor((C, C), "float16"), out: T.Tensor((1, C, C), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bk = T.alloc_L1([C, C], "float16")
                lcc = T.alloc_L0C([C, C], "float32")
                T.copy(k, bk)
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(lcc, out[0, 0:C, 0:C])
                T.gemm(bk, bk, lcc, size=[C, C, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(lcc, out[0, 0:C, 0:C])

    return main


torch.manual_seed(0)
k = torch.randn(C, C, dtype=torch.float16).npu()
ref = k.cpu().float() @ k.cpu().float().T

for name, builder in (("A 2D 1gemm", build_a), ("B 3D-slice 1gemm", build_b),
                      ("C 2D 2gemm reuse", build_c), ("D 3D-slice 2gemm PIPE_FIX", build_d)):
    try:
        out = builder()(k)
        torch.npu.synchronize()
        got = out.cpu().float()
        if got.ndim == 3:
            got = got[0]
        err = float((got - ref).abs().max())
        print(f"{name:28s} OK   err={err:.2e}")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        key = "only support store ub to gm" if "only support store ub to gm" in msg else msg[:110]
        print(f"{name:28s} FAIL {key}")
