"""Which loop form makes NpuLoopVectorize emit npuir vector ops?

Dumps mlir_content (post npuir lowering) and greps for vector op markers.
No NPU execution needed -> fast.
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import re

import tilelang
import tilelang.language as T

N = 4096
H = N // 2


def build(variant):
    @tilelang.jit(out_idx=[2], target="npuir")
    def f(variant=variant):
        @T.prim_func
        def main(
            x: T.Tensor((1, N), "float32"),
            lut: T.Tensor((N - 1,), "float32"),
            y: T.Tensor((1, N), "float32"),
        ) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                Bc = T.alloc_shared((N,), "float32")
                T.copy(x[cid, 0:N], A)
                if variant == "A_flat_4store":
                    for g, i in T.Parallel(1, H):
                        Bc[g * N + i] = A[g * N + i] + A[g * N + H + i]
                    for g, i in T.Parallel(1, H):
                        Bc[g * N + i + H] = (A[g * N + i] - A[g * N + H + i]) * lut[i]
                elif variant == "B_flat_1store_twoloops":
                    for i in T.Parallel(H):
                        Bc[i] = A[i] + A[H + i]
                    for i in T.Parallel(H):
                        Bc[H + i] = (A[i] - A[H + i]) * lut[i]
                elif variant == "C_3d_direct":
                    A3 = T.alloc_shared((1, 2, H), "float32")
                    D3 = T.alloc_shared((2, 1, H), "float32")
                    T.copy(x[cid, 0:N], A3[0, 0, 0:H])
                    T.copy(x[cid, 0:N], A3[0, 1, 0:H])
                    for g, i in T.Parallel(1, H):
                        D3[0, g, i] = A3[g, 0, i] + A3[g, 1, i]
                    for g, i in T.Parallel(1, H):
                        D3[1, g, i] = (A3[g, 0, i] - A3[g, 1, i]) * lut[i]
                    T.copy(D3, Bc)
                elif variant == "D_2d_contig":
                    A2 = T.alloc_shared((2, H), "float32")
                    D2 = T.alloc_shared((2, H), "float32")
                    T.copy(x[cid, 0:H], A2[0, 0:H])
                    T.copy(x[cid, H:N], A2[1, 0:H])
                    for i in T.Parallel(H):
                        D2[0, i] = A2[0, i] + A2[1, i]
                    for i in T.Parallel(H):
                        D2[1, i] = (A2[0, i] - A2[1, i]) * lut[i]
                    T.copy(D2, Bc)
                T.copy(Bc, y[cid, 0:N])

        return main

    return f()


VEC_MARKERS = ["hivm.hir.vadd", "hivm.hir.vsub", "hivm.hir.vmul", "VAddOp",
               "vadd", "vsub", "vmul", "npuir_add", "npuir_sub", "npuir_mul",
               "hivm.hir.v"]

for v in ("A_flat_4store", "B_flat_1store_twoloops", "C_3d_direct", "D_2d_contig"):
    try:
        k = build(v)
        ir = k.mlir_content
        found = sorted({m for m in VEC_MARKERS if m in ir})
        nloops = ir.count("scf.for")
        nextract = ir.count("tensor.extract")
        print(f"{v}: OK vec_markers={found} scf.for={nloops} tensor.extract={nextract}")
    except Exception as e:  # noqa: BLE001
        print(f"{v}: ERR {str(e).splitlines()[-1][:220]}")
