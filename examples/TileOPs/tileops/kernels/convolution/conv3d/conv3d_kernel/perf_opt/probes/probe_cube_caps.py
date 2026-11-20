#!/usr/bin/env python3
"""Cube capability probes needed by the conv3d unet restructure.

P1 l0c-col-slice : T.copy(l0_c[0:M, off:off+N], gm) -- is the L0C column
                   offset honoured by the fixpipe lowering?  (The sliced-gemm
                   trap TRAP-cube-gemm-sliced-operand covers gemm *operands*;
                   this is the FIXPIPE source side.)
P2 b-transpose   : T.gemm(A, B, C, b_transpose=True) with B stored [N, K] in L1
                   (Expert mode, L1 operands) -- compiles and is numerically
                   correct?

Both are single-core correctness probes against a torch reference.
"""
import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

DT = "float16"
M, K, N = 64, 64, 128


@tilelang.jit(out_idx=[2], target="npuir")
def p1_l0c_slice(off):
    @T.prim_func
    def main(
        a: T.Tensor((M, K), DT),
        b: T.Tensor((K, N), DT),
        out: T.Tensor((M, N - 32), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (kid, _):
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([M, K], DT)
                l1_b = T.alloc_L1([K, N], DT)
                l0_c = T.alloc_L0C([M, N], "float32")
                with T.rs("PIPE_MTE2"):
                    T.copy(a, l1_a)
                    T.copy(b, l1_b)
                with T.rs("PIPE_C"):
                    T.gemm(l1_a, l1_b, l0_c, initC=True, size=[M, K, N])
                with T.rs("PIPE_FIX"):
                    T.copy(l0_c[0:M, off:off + (N - 32)], out)

    return main


@tilelang.jit(out_idx=[2], target="npuir")
def p2_b_transpose():
    @T.prim_func
    def main(
        a: T.Tensor((M, K), DT),
        bt: T.Tensor((N, K), DT),
        out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (kid, _):
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([M, K], DT)
                l1_b = T.alloc_L1([N, K], DT)
                l0_c = T.alloc_L0C([M, N], "float32")
                with T.rs("PIPE_MTE2"):
                    T.copy(a, l1_a)
                    T.copy(bt, l1_b)
                with T.rs("PIPE_C"):
                    T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True,
                           size=[M, K, N])
                with T.rs("PIPE_FIX"):
                    T.copy(l0_c[0:M, 0:N], out)

    return main


def check_p1(off=32):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    b = torch.randn(K, N, dtype=torch.float16, device="npu")
    ker = p1_l0c_slice(off)
    out = ker(a, b).cpu().float()
    ref = (a.cpu().float() @ b.cpu().float())[:, off:off + (N - 32)]
    diff = (out - ref).abs().max().item()
    shifted0 = (out - (a.cpu().float() @ b.cpu().float())[:, :(N - 32)]).abs().max().item()
    print(f"[P1 l0c col-slice off={off}] max_abs_vs_ref={diff:.4e} "
          f"max_abs_vs_offset0={shifted0:.4e} -> "
          f"{'OFFSET HONOURED' if diff < 1e-2 else 'OFFSET IGNORED/OTHER'}")
    return diff < 1e-2


def check_p2():
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    bt = torch.randn(N, K, dtype=torch.float16, device="npu")
    ker = p2_b_transpose()
    out = ker(a, bt).cpu().float()
    ref = a.cpu().float() @ bt.cpu().float().T
    diff = (out - ref).abs().max().item()
    print(f"[P2 b_transpose] max_abs_vs_ref={diff:.4e} -> "
          f"{'OK' if diff < 1e-2 else 'WRONG'}")
    return diff < 1e-2


if __name__ == "__main__":
    ok1 = ok2 = None
    try:
        ok1 = check_p1(32)
    except Exception as exc:  # noqa: BLE001
        print(f"[P1] FAILED: {str(exc).splitlines()[0][:300]}")
    try:
        ok2 = check_p2()
    except Exception as exc:  # noqa: BLE001
        print(f"[P2] FAILED: {str(exc).splitlines()[0][:300]}")
    print(f"SUMMARY l0c_col_slice={ok1} b_transpose={ok2}")
