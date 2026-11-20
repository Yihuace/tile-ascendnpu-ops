"""Probe: chunk-major L0C epilogue via T.unroll + single part buffer + vadd.

If this compiles and runs correctly it validates the S3 head-chunk structure
that keeps ``acc_ub`` at block_Q*hp_sub*block_N (frees UB for larger tiles).
"""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

BQ, HP, BN, D = 16, 32, 64, 64
HP_SUB = 8
CH = BQ * HP_SUB
NCH = HP // HP_SUB


@tilelang.jit(target="npuir")
def build():
    dt = "bfloat16"
    acc_dt = "float32"

    @T.prim_func
    def main(
            Q: T.Tensor([BQ * HP, D], dt),  # type: ignore
            K: T.Tensor([BN, D], dt),  # type: ignore
            W: T.Tensor([BQ * HP], acc_dt),  # type: ignore
            O: T.Tensor([BQ, BN], acc_dt),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            q_ub = T.alloc_shared([BQ * HP, D], dt)
            k_ub = T.alloc_shared([BN, D], dt)
            acc = T.alloc_fragment([BQ * HP, BN], acc_dt)
            acc_ub = T.alloc_shared([CH, BN], acc_dt)
            w_ub = T.alloc_shared([BQ * HP, 1], acc_dt)
            part_ub = T.alloc_shared([BQ, BN], acc_dt)
            out_ub = T.alloc_shared([BQ, BN], acc_dt)

            T.copy(Q, q_ub)
            T.copy(K, k_ub)
            T.copy(W, w_ub)
            T.gemm(q_ub, k_ub, acc, b_transpose=True, initC=True)
            T.vbrc(T.float32(0.0), out_ub)
            for hg in T.unroll(NCH):
                T.copy(acc[hg * CH:(hg + 1) * CH, :], acc_ub)
                T.vrelu(acc_ub, acc_ub)
                T.vmul(acc_ub, w_ub[hg * CH:(hg + 1) * CH, 0:1], acc_ub)
                for q in T.serial(BQ):
                    T.reduce_sum(acc_ub[q * HP_SUB:(q + 1) * HP_SUB, :],
                                 part_ub[q:q + 1, :], dim=0, clear=True)
                T.vadd(out_ub, part_ub, out_ub)
            T.copy(out_ub, O)

    return main


torch.manual_seed(0)
k = build()
q = torch.randn(BQ * HP, D, dtype=torch.bfloat16, device="npu")
kk = torch.randn(BN, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(BQ * HP, dtype=torch.float32, device="npu")
o = torch.empty(BQ, BN, dtype=torch.float32, device="npu")
k(q, kk, w, o)
ref = torch.zeros(BQ, BN, dtype=torch.float32, device="npu")
sc = (q.float() @ kk.float().t()).relu() * w[:, None]
for qq in range(BQ):
    for hg in range(NCH):
        ref[qq] += sc[hg * CH + qq * HP_SUB:hg * CH + (qq + 1) * HP_SUB].sum(0)
d = (o - ref).abs().max().item()
print(f"max_abs={d:.4e}")
assert d < 5e-2, d
print("PASS")
