"""Can a 1-D Parallel loop with a plain-Var store but non-affine source index
still vectorise?"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import collections
import re

import tilelang
import tilelang.language as T

N = 4096
NH = N // 2


def build(variant):
    @tilelang.jit(out_idx=[3], target="npuir")
    def f(variant=variant):
        @T.prim_func
        def main(
            x: T.Tensor((1, N), "float32"),
            lut: T.Tensor((N - 1,), "float32"),
            s2: T.Tensor((2, NH), "float32"),
            y: T.Tensor((1, N), "float32"),
        ) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                S2 = T.alloc_shared((2, NH), "float32")
                D2 = T.alloc_shared((2, NH), "float32")
                T.copy(x[cid, 0:N], A)
                T.copy(s2[0:2, 0:NH], S2)
                if variant == "T1_nonaffine_read":
                    # h = NH (stage 0, ng=1): m//h == 0, m%h == m -> affine here.
                    # emulate a later stage (ng=4, h=NH//4=512) to get the
                    # piecewise case.
                    h = NH // 4
                    for m in T.Parallel(NH):
                        D2[0, m] = A[(m // h) * 2 * h + m % h] + A[(m // h) * 2 * h + h + m % h]
                    for m in T.Parallel(NH):
                        D2[1, m] = (
                            A[(m // h) * 2 * h + m % h] - A[(m // h) * 2 * h + h + m % h]
                        ) * lut[m % h]
                elif variant == "T2_contig_read":
                    for m in T.Parallel(NH):
                        D2[0, m] = S2[0, m] + S2[1, m]
                    for m in T.Parallel(NH):
                        D2[1, m] = (S2[0, m] - S2[1, m]) * lut[m]
                T.copy(D2, y[cid, 0:N])

        return main

    return f()


for v in ("T1_nonaffine_read", "T2_contig_read"):
    try:
        k = build(v)
        ir = k.mlir_content
        ops = collections.Counter(re.findall(r"hivm\.hir\.[a-z_0-9]+", ir))
        print(f"{v}: OK scf.for={ir.count('scf.for')} addf={ir.count('arith.addf')} "
              f"mulf={ir.count('arith.mulf')} subf={ir.count('arith.subf')} hivm={dict(ops)}")
    except Exception as e:  # noqa: BLE001
        print(f"{v}: ERR {str(e).splitlines()[-1][:200]}")
