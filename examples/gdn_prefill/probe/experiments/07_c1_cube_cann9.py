"""C1 (Cube) probe, CANN 9.0 variant (v2, WORKING form).

Two 9.0-specific constraints were isolated with 08/09:

1. An un-synchronised L0C -> GM copy lowers to a generic ``hivm.hir.store`` and is
   rejected: "'hivm.hir.store' op only support store ub to gm currently!".
   -> every L0C store is wrapped in ``with T.rs("PIPE_FIX")`` (as in 01_expert_gemm).
2. An L0C buffer that has been stored from must NOT be re-used as a gemm
   destination.  A fresh L0C is required per gemm result (variant E in 09).
   03_c1_cube.py / 07 v1 re-used two L0C buffers for four gemms -> rejected.

Also verified (08): a 3-D GM destination sliced as ``out[0, 0:C, 0:D]`` is fine.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C, D, DV = 16, 16, 16


@tilelang.jit(out_idx=[3, 4, 5, 6], target="npuir")
def build():
    @T.prim_func
    def main(
        q: T.Tensor((1, C, D), "float16"),
        k: T.Tensor((1, C, D), "float16"),
        s: T.Tensor((1, D, DV), "float16"),
        kk: T.Tensor((1, C, C), "float32"),
        qk: T.Tensor((1, C, C), "float32"),
        ks: T.Tensor((1, C, DV), "float32"),
        qs: T.Tensor((1, C, DV), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1k = T.alloc_L1([C, D], "float16")
                l1q = T.alloc_L1([C, D], "float16")
                l1s = T.alloc_L1([D, DV], "float16")
                # one L0C per gemm result (9.0 rule: no store-then-reuse)
                l0_kk = T.alloc_L0C([C, C], "float32")
                l0_qk = T.alloc_L0C([C, C], "float32")
                l0_ks = T.alloc_L0C([C, DV], "float32")
                l0_qs = T.alloc_L0C([C, DV], "float32")
                T.copy(k[0, 0:C, 0:D], l1k)
                T.copy(q[0, 0:C, 0:D], l1q)
                T.copy(s[0, 0:D, 0:DV], l1s)

                T.gemm(l1k, l1k, l0_kk, size=[C, D, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0_kk, kk[0, 0:C, 0:C])

                T.gemm(l1q, l1k, l0_qk, size=[C, D, C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0_qk, qk[0, 0:C, 0:C])

                T.gemm(l1k, l1s, l0_ks, size=[C, D, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0_ks, ks[0, 0:C, 0:DV])

                T.gemm(l1q, l1s, l0_qs, size=[C, D, DV], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0_qs, qs[0, 0:C, 0:DV])

    return main


torch.manual_seed(0)
q = torch.randn(1, C, D, dtype=torch.float16).npu()
k = torch.randn(1, C, D, dtype=torch.float16).npu()
s = torch.randn(1, D, DV, dtype=torch.float16).npu()
kk, qk, ks, qs = build()(q, k, s)


def e(a, b):
    return float((a.cpu()[0].float() - b).abs().max())


print("KK", e(kk, k[0].cpu().float() @ k[0].cpu().float().T))
print("QK", e(qk, q[0].cpu().float() @ k[0].cpu().float().T))
print("KS", e(ks, k[0].cpu().float() @ s[0].cpu().float()))
print("QS", e(qs, q[0].cpu().float() @ s[0].cpu().float()))
