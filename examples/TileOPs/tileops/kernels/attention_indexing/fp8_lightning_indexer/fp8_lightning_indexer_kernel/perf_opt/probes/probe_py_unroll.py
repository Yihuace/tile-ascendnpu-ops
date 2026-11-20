"""Probe: can a Python-level (meta) loop unroll in the npuir kernel body?

Checks two forms:
  A) `for hg in [0, 1]:` over a literal Python list  -> Python unroll?
  B) a literal list of two allocated buffers
and whether a *static* L0C row-range copy + accumulate reduce compiles.
"""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T

BQ, HP, BN, D = 8, 32, 64, 64
HP_SUB = 16
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
            O: T.Tensor([BQ, BN], acc_dt),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            q_ub = T.alloc_shared([BQ * HP, D], dt)
            k_ub = T.alloc_shared([BN, D], dt)
            parts = [T.alloc_shared([BQ, BN], acc_dt), T.alloc_shared([BQ, BN], acc_dt)]
            acc = T.alloc_fragment([BQ * HP, BN], acc_dt)
            acc_ub = T.alloc_shared([CH, BN], acc_dt)
            out_ub = T.alloc_shared([BQ, BN], acc_dt)

            T.copy(Q, q_ub)
            T.copy(K, k_ub)
            T.gemm(q_ub, k_ub, acc, b_transpose=True, initC=True)
            for hg in [0, 1]:
                T.copy(acc[hg * CH:(hg + 1) * CH, :], acc_ub)
                T.vrelu(acc_ub, acc_ub)
                for q in T.serial(BQ):
                    T.reduce_sum(acc_ub[q * HP_SUB:(q + 1) * HP_SUB, :],
                                 parts[hg][q:q + 1, :], dim=0, clear=True)
            T.vadd(parts[0], parts[1], out_ub)
            T.copy(out_ub, O)

    return main


k = build()
print("COMPILE_OK")
