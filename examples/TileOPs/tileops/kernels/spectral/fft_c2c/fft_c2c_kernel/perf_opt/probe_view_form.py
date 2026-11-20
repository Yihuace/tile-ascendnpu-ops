"""Minimal localisation probe: which T.view construction form survives the
TileLang parser + TVM SSA check when a macro chain creates a fresh view of the
same shared ping-pong buffer at every FFT stage?

Forms:
  F1  bound name, same identifier reused across stages   (round1 -- expected FAIL)
  F2  inline T.view(...)[idx], never bound to a name
  F3  helper python fn returning BufferLoad, call result inline
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

N = 8
T_STAGES = 3


def make_chain_factory(form):
    if form == "F1":
        def make_chain(s):
            if s == T_STAGES:
                @T.macro
                def done(S, D):
                    pass
                return done
            nxt = make_chain(s + 1)
            h = 1 << (T_STAGES - 1 - s)
            ng = N // (2 * h)

            @T.macro
            def stage(S, D):
                S3 = T.view(S, [ng, 2, h])
                D3 = T.view(D, [2, ng, h])
                for g, i in T.Parallel(ng, h):
                    D3[0, g, i] = S3[g, 0, i] + S3[g, 1, i]
                for g, i in T.Parallel(ng, h):
                    D3[1, g, i] = S3[g, 0, i] - S3[g, 1, i]
                nxt(D, S)
            return stage
        return make_chain

    if form == "F2":
        def make_chain(s):
            if s == T_STAGES:
                @T.macro
                def done(S, D):
                    pass
                return done
            nxt = make_chain(s + 1)
            h = 1 << (T_STAGES - 1 - s)
            ng = N // (2 * h)

            @T.macro
            def stage(S, D):
                for g, i in T.Parallel(ng, h):
                    T.view(D, [2, ng, h])[0, g, i] = (
                        T.view(S, [ng, 2, h])[g, 0, i]
                        + T.view(S, [ng, 2, h])[g, 1, i])
                for g, i in T.Parallel(ng, h):
                    T.view(D, [2, ng, h])[1, g, i] = (
                        T.view(S, [ng, 2, h])[g, 0, i]
                        - T.view(S, [ng, 2, h])[g, 1, i])
                nxt(D, S)
            return stage
        return make_chain

    if form == "F3":
        def vload(buf, shape, idx):
            return T.view(buf, shape)[idx]

        def make_chain(s):
            if s == T_STAGES:
                @T.macro
                def done(S, D):
                    pass
                return done
            nxt = make_chain(s + 1)
            h = 1 << (T_STAGES - 1 - s)
            ng = N // (2 * h)

            @T.macro
            def stage(S, D):
                for g, i in T.Parallel(ng, h):
                    T.view(D, [2, ng, h])[0, g, i] = (
                        vload(S, [ng, 2, h], (g, 0, i))
                        + vload(S, [ng, 2, h], (g, 1, i)))
                for g, i in T.Parallel(ng, h):
                    T.view(D, [2, ng, h])[1, g, i] = (
                        vload(S, [ng, 2, h], (g, 0, i))
                        - vload(S, [ng, 2, h], (g, 1, i)))
                nxt(D, S)
            return stage
        return make_chain

    raise ValueError(form)


def make_kernel(form):
    @tilelang.jit(out_idx=[1], target="npuir")
    def f(form=form):
        chain = make_chain_factory(form)(0)

        @T.prim_func
        def main(x: T.Tensor((1, N), "float32"),
                 y: T.Tensor((1, N), "float32")) -> None:
            with T.Kernel(1, is_npu=True) as (cid, _):
                A = T.alloc_shared((N,), "float32")
                Bc = T.alloc_shared((N,), "float32")
                T.copy(x[cid, 0:N], A)
                chain(A, Bc)
                T.copy(Bc, y[cid, 0:N])

        return main

    return f()


for form in ("F1", "F2", "F3"):
    try:
        k = make_kernel(form)
        ir = k.mlir_content
        print(f"{form}: COMPILE OK  scf.for={ir.count('scf.for')} "
              f"vector={ir.count('vector')} npuir_add={ir.count('npuir_add')} "
              f"npuir_sub={ir.count('npuir_sub')}")
    except Exception as e:  # noqa: BLE001
        tail = str(e).splitlines()[-1][:220]
        print(f"{form}: FAIL {tail}")
