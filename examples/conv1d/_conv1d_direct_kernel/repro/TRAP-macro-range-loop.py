# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: TRAP-macro-range-loop

Phenomenon: A python ``for ... in range(...)`` loop placed inside a
``@T.macro`` that is called from a ``@T.prim_func`` fails to compile with a
*misleading* parser diagnostic::

    error: Expect the for loop to be one of the following: range, T.serial,
    T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding

The diagnostic names ``range`` as allowed, but the TVMScript macro parser does
not resolve the builtin ``range`` inside a ``@T.macro`` body (it does resolve it
directly inside a ``@T.prim_func`` body).  The reported source location is
bogus (it points at a line *after* the offending function), so the error is hard
to localise from the message alone.

Fix / workaround: inside ``@T.macro`` bodies use ``T.serial(...)`` (or
``T.unroll``) for loops.  If a python-int loop index is needed to index a python
list at trace time, move the loop out of the macro (inline it in the
``@T.prim_func`` body) or rewrite the index arithmetic as TIR expressions on the
loop variable.

First verified: 2026-10-14, conv1d _conv1d_direct_kernel Stage 3.
Toolchain: tilelang 0.1.2+417dae99 + CANN 8.5.x + Ascend 910B3
(get_aicore_num=20), Expert mode.

Run: python repro/TRAP-macro-range-loop.py
"""
import os
import subprocess
import sys
import textwrap

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

N = 16
_HERE = os.path.dirname(os.path.abspath(__file__))


@tilelang.jit(out_idx=[1], target="npuir")
def build_serial_macro(n):
    """Control: macro with T.serial loop -- compiles fine."""

    @T.macro
    def body(a, b):
        with T.Kernel(1, is_npu=True) as (kid, _):
            with T.Scope("Vector"):
                u = T.alloc_ub([n], "float16")
                for i in T.serial(2):
                    T.copy(a[0:n], u)
                T.copy(u, b[0:n])

    @T.prim_func
    def main(a: T.Tensor((n,), "float16"), b: T.Tensor((n,), "float16")):
        body(a, b)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def build_range_in_prim(n):
    """Control: `range` loop directly in the prim_func body -- compiles fine."""

    @T.prim_func
    def main(a: T.Tensor((n,), "float16"), b: T.Tensor((n,), "float16")):
        with T.Kernel(1, is_npu=True) as (kid, _):
            with T.Scope("Vector"):
                u = T.alloc_ub([n], "float16")
                for i in range(2):
                    T.copy(a[0:n], u)
                T.copy(u, b[0:n])

    return main


# The failing variant must run in a subprocess: the parser diagnostic aborts
# the process, so it cannot be caught in-process.
_FAILING_SRC = textwrap.dedent(
    """
    import os
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")
    import tilelang
    import tilelang.language as T

    @tilelang.jit(out_idx=[1], target="npuir")
    def build(n):
        @T.macro
        def body(a, b):
            with T.Kernel(1, is_npu=True) as (kid, _):
                with T.Scope("Vector"):
                    u = T.alloc_ub([n], "float16")
                    for i in range(2):
                        T.copy(a[0:n], u)
                    T.copy(u, b[0:n])
        @T.prim_func
        def main(a: T.Tensor((n,), "float16"), b: T.Tensor((n,), "float16")):
            body(a, b)
        return main

    build(16)
    print("BUILD_OK")
    """
)


def run():
    build_serial_macro(N)
    build_range_in_prim(N)
    print("control A PASS: @T.macro with T.serial compiles")
    print("control B PASS: range directly in @T.prim_func compiles")

    tmp = os.path.join(_HERE, "_failing_variant.py")
    with open(tmp, "w") as fh:
        fh.write(_FAILING_SRC)
    try:
        proc = subprocess.run(
            [sys.executable, tmp], capture_output=True, text=True, timeout=600
        )
    finally:
        os.remove(tmp)
    err = proc.stdout + proc.stderr
    assert proc.returncode != 0, "failing variant unexpectedly succeeded"
    assert "Expect the for loop" in err, f"unexpected failure output:\n{err[-2000:]}"
    print("failing variant FAILED as expected: 'Expect the for loop' diagnostic")
    print("[TRAP-macro-range-loop] PASS: @T.macro + range fails; T.serial / prim_func range work")


if __name__ == "__main__":
    run()
