#!/usr/bin/env python3
"""Minimal probe: does T.gemm accept a sliced L1 B operand (shifted N)? """
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

M, K, N = 64, 128, 128


@tilelang.jit(out_idx=[2], target="npuir")
def probe():
    @T.prim_func
    def kern(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N + 2), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            l1_a = T.alloc_L1([M, K], "float16")
            l1_b = T.alloc_L1([K, N + 2], "float16")
            l0_c = T.alloc_L0C([M, N], "float32")
            with T.rs("PIPE_MTE2"):
                T.copy(A[0:M, 0:K], l1_a[0:M, 0:K])
                T.copy(B[0:K, 0:N + 2], l1_b[0:K, 0:N + 2])
            with T.rs("PIPE_C"):
                # shifted slice of B (offset 1 along N)
                T.gemm(l1_a, l1_b[0:K, 1:1 + N], l0_c, initC=True,
                       size=[M, K, N])
            with T.rs("PIPE_FIX"):
                T.copy(l0_c[0:M, 0:N], C[0:M, 0:N])

    return kern


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    b = torch.randn(K, N + 2, dtype=torch.float16, device="npu")
    k = probe()
    c = k(a, b)
    torch.npu.synchronize()
    ref = (a.float() @ b.float()[:, 1:1 + N])
    diff = (c.cpu().float() - ref.cpu()).abs().max().item()
    print(f"PASS max_abs={diff:.4e}" if diff < 1e-1 else f"FAIL max_abs={diff:.4e}")
