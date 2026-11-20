"""Does T.reshape(src, dst) (npuir_reshape) act as a free alias, a copy, or
fail, on npuir shared buffers?

R1: A flat (8,) filled with x; reshape A -> B (2,4); write B[i,j]=B[i,j]+1 in a
    T.Parallel loop; then read A back out.  If A reflects the +1 -> alias.
R2: same but read B back out (sanity).
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

N = 8


def r1(read_back_a):
    @tilelang.jit(out_idx=[1], target="npuir")
    def f(read_back_a=read_back_a):
        @T.prim_func
        def main(x: T.Tensor((N,), "float32"),
                 y: T.Tensor((N,), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                B = T.alloc_shared((N // 2, 2), "float32")
                T.copy(x[0:N], A)
                T.reshape(A, B)
                for i, j in T.Parallel(N // 2, 2):
                    B[i, j] = B[i, j] + 1.0
                if read_back_a:
                    T.copy(A, y[0:N])
                else:
                    T.copy(B, y[0:N // 2, 0:2])
        return main
    return f()


for name, rb in (("R1_read_A", True), ("R2_read_B", False)):
    try:
        k = r1(rb)
        print(f"{name}: COMPILE OK")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAIL {str(e).splitlines()[-1][:200]}")
