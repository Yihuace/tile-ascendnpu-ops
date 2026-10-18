#!/usr/bin/env python3
"""TRAP-cube-gemm-sliced-operand — T.gemm 忽略 L1 操作数的切片偏移（ED-B 分级断言：self-contained + assert）。

断言：`T.gemm(l1_a, l1_b[0:K, 1:1+N], l0_c, size=[M,K,N])` 编译通过，但 gemm 按 l1_b 的
buffer 起点（offset 0）读取，忽略 `[1:1+N]` 的 N 偏移 → 结果与 `A @ B[:, 1:1+N]` 错位。

来源任务：conv3d_bias-Conv3dBiasFwdOp-20261017T220345Z（Stage 4）。
机制归因：gemm 的 `size=[a,b,c]` 只约束 K/N 扩展，src2 一律按 [0:b, 0:c] 起点读，
切片下标不参与 mmadL1 lowering ——「宽 B 块 + 平移 gemm」的 im2col 输入重复读削减因此不可用。
First verified: 2026-10-18, tilelang 0.1.2 dev root (tilelang-mlir-ascend/build) + CANN 9.0.0
  (bishengir 1.1.0) + Ascend910B3 (Chip 6, 20 AIC).
"""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

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
                # shifted slice of B (offset 1 along N) -- silently ignored
                T.gemm(l1_a, l1_b[0:K, 1:1 + N], l0_c, initC=True, size=[M, K, N])
            with T.rs("PIPE_FIX"):
                T.copy(l0_c[0:M, 0:N], C[0:M, 0:N])

    return kern


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    b = torch.randn(K, N + 2, dtype=torch.float16, device="npu")
    c = probe()(a, b)
    torch.npu.synchronize()
    ac = a.float().cpu()
    bc = b.float().cpu()
    cc = c.cpu().float()
    ref_shifted = (ac @ bc[:, 1:1 + N])
    ref_origin = (ac @ bc[:, 0:N])
    d_shift = (cc - ref_shifted).abs().max().item()
    d_origin = (cc - ref_origin).abs().max().item()
    # The slice offset [1:1+N] is ignored: output matches B[:, 0:N], not B[:, 1:N+1].
    assert d_origin < 1e-1, f"gemm does not even match origin slice: {d_origin}"
    assert d_shift > 1.0, f"slice offset unexpectedly honoured: {d_shift}"
    print(f"TRAP confirmed: origin_slice_err={d_origin:.3e} shifted_slice_err={d_shift:.3e}")
