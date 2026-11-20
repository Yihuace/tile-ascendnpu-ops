"""Alternative vectorisation route: TIR-level T.vectorized loop instead of the
npuir custom NpuLoopVectorize pass (which requires bare-Var store indices).

V1: for g in T.serial(ng): for i in T.vectorized(h): <4 affine stores>
V2: 4 separate T.serial(g)/T.vectorized(i) loops, one store each
V3: single T.Parallel(ng,h) loop (baseline form) -- control
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

N = 4096
NH = N // 2


def build(variant):
    @tilelang.jit(out_idx=[2], target="npuir")
    def f(variant=variant):
        @T.prim_func
        def main(x: T.Tensor((1, N), "float32"),
                 lut: T.Tensor((N - 1,), "float32"),
                 y: T.Tensor((1, N), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                B = T.alloc_shared((N,), "float32")
                T.copy(x[cid, 0:N], A)
                h = NH // 4          # a mid stage: ng=4, h=512
                ng = N // (2 * h)
                lb = h - 1
                if variant == "V1_serial_vec_4store":
                    for g in T.serial(ng):
                        for i in T.vectorized(h):
                            lo = A[g * 2 * h + i]
                            hi = A[g * 2 * h + h + i]
                            B[g * h + i] = lo + hi
                            B[g * h + i + NH] = (lo - hi) * lut[lb + i]
                elif variant == "V2_serial_vec_split":
                    for g in T.serial(ng):
                        for i in T.vectorized(h):
                            B[g * h + i] = (A[g * 2 * h + i]
                                            + A[g * 2 * h + h + i])
                    for g in T.serial(ng):
                        for i in T.vectorized(h):
                            B[g * h + i + NH] = ((A[g * 2 * h + i]
                                                  - A[g * 2 * h + h + i])
                                                 * lut[lb + i])
                elif variant == "V3_parallel":
                    for g, i in T.Parallel(ng, h):
                        lo = A[g * 2 * h + i]
                        hi = A[g * 2 * h + h + i]
                        B[g * h + i] = lo + hi
                        B[g * h + i + NH] = (lo - hi) * lut[lb + i]
                T.copy(B, y[cid, 0:N])
        return main
    return f()


for v in ("V1_serial_vec_4store", "V2_serial_vec_split", "V3_parallel"):
    try:
        k = build(v)
        ir = k.mlir_content
        print(f"{v}: COMPILE OK scf.for={ir.count('scf.for')} "
              f"vector={ir.count('vector')} npuir_add={ir.count('npuir_add')} "
              f"npuir_mul={ir.count('npuir_mul')}")
    except Exception as e:  # noqa: BLE001
        print(f"{v}: FAIL {str(e).splitlines()[-1][:200]}")
