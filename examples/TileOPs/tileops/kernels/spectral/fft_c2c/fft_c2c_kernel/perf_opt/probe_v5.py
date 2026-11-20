"""Precise per-form vectorisation census on the real FFT-shaped stage."""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import collections

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
                if variant == "S1_var_store":
                    for i in T.Parallel(H):
                        Bc[i] = A[i] + A[H + i]
                elif variant == "S2_cmp_store":
                    for i in T.Parallel(H):
                        Bc[H + i] = A[i] + A[H + i]
                elif variant == "S3_four_store":
                    for i in T.Parallel(H):
                        Bc[i] = A[i] + A[H + i]
                        Bc[H + i] = A[i] - A[H + i]
                elif variant == "S4_two_loops_cmp":
                    for i in T.Parallel(H):
                        Bc[i] = A[i] + A[H + i]
                    for i in T.Parallel(H):
                        Bc[H + i] = (A[i] - A[H + i]) * lut[i]
                elif variant == "S5_two_loops_sepbuf":
                    Bd = T.alloc_shared((H,), "float32")
                    for i in T.Parallel(H):
                        Bc[i] = A[i] + A[H + i]
                    for i in T.Parallel(H):
                        Bd[i] = (A[i] - A[H + i]) * lut[i]
                    T.copy(Bd, Bc[H:N])
                T.copy(Bc, y[cid, 0:N])

        return main

    return f()


for v in ("S1_var_store", "S2_cmp_store", "S3_four_store", "S4_two_loops_cmp",
          "S5_two_loops_sepbuf"):
    try:
        k = build(v)
        ir = k.mlir_content
        ops = collections.Counter(
            __import__("re").findall(r"hivm\.hir\.[a-z_0-9]+", ir))
        print(f"{v}: OK scf.for={ir.count('scf.for')} "
              f"tensor.extract={ir.count('tensor.extract')} "
              f"arith.addf={ir.count('arith.addf')} "
              f"arith.mulf={ir.count('arith.mulf')} hivm={dict(ops)}")
    except Exception as e:  # noqa: BLE001
        print(f"{v}: ERR {str(e).splitlines()[-1][:200]}")
