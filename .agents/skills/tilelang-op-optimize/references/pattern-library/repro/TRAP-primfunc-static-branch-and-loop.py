# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-primfunc-static-branch-and-loop -- ``T.prim_func`` bodies that look like
plain Python yet the parser rejects:
  BAD-1 ``if 0 <= idx < L:`` -> "Literal value N exceeds maximum of bool"
        (``for p in range(k)`` is NOT unrolled -- p is a Var, so ``idx`` is
        symbolic and the chained comparison is lowered as a bool cast)
  BAD-2 ``for p, idx in taps:`` (Python tuple) -> "Expect the for loop to be one
        of the following: range, T.serial, T.grid, T.parallel, ..."
GOOD = fold the guard *semantics* at factory level, keep a static trip count and
an always-legal affine row index: legal taps are the suffix p in [p_start, w-2]
with p_start = max(0, w-1-L//dil).  Hit in a dilated-causal-conv tap loop whose
DESIGN asked for a "compile-time Python branch".
Kind: trap.  Assertions: both BAD forms fail with the quoted diagnostics (in a
subprocess -- the readable text goes to the C++ diagnostic stream, the Python
exception only carries a source-map render error); GOOD is bit-exact for 2
shapes incl. the guarded ``L < dil*(w-1)`` case.
First verified: 2026-10-31, tilelang 0.1.2+3ed7c914 / CANN 9.0.0 / Ascend910B3,
TILELANG_ASCEND_MODE=Developer.  origin_task: engram_decode_mcore (Stage 3).
Re-verify: python repro/TRAP-primfunc-static-branch-and-loop.py
"""

import os
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

F32 = "float32"
D = 32


def _build(mode, L=12, w=4, dil=1):
    p_start = max(0, w - 1 - L // dil)
    n = (w - 1) - p_start
    taps = tuple((p, L - (w - 1 - p) * dil) for p in range(p_start, w - 1))
    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def main(x: T.Tensor((L, D), F32), y: T.Tensor((D,), F32)):
            with T.Kernel(1, is_npu=True) as (bid, _):
                acc = T.alloc_shared((D,), F32)
                row = T.alloc_shared((D,), F32)
                T.clear(acc)
                if mode == "bad1":
                    for p in range(n):                  # p is a symbolic Var
                        idx = L - (w - 1 - p) * dil
                        if 0 <= idx < L:                # BAD-1: symbolic `if`
                            T.copy(x[idx, :], row)
                            for j in T.Parallel(D):
                                acc[j] += row[j]
                elif mode == "bad2":
                    for _p, idx in taps:                # BAD-2: not a loop form
                        T.copy(x[idx, :], row)
                        for j in T.Parallel(D):
                            acc[j] += row[j]
                else:
                    for t in T.serial(n):
                        p = t + p_start
                        T.copy(x[L - (w - 1 - p) * dil, :], row)
                        for j in T.Parallel(D):
                            acc[j] += row[j]
                T.copy(acc, y)

        return main
    return factory()


if __name__ == "__main__":
    if len(sys.argv) > 1:                      # --bad1 / --bad2 child mode
        _build(sys.argv[1][2:])
        sys.exit(0)
    for label, flag, needle in (
        ("BAD-1 symbolic if", "--bad1", "exceeds maximum of bool"),
        ("BAD-2 list iteration", "--bad2", "Expect the for loop"),
    ):
        proc = subprocess.run([sys.executable, __file__, flag],
                              capture_output=True, text=True, timeout=1800)
        assert proc.returncode != 0, f"{label} compiled -- trap changed"
        blob = proc.stdout + proc.stderr
        assert needle in blob, f"{label}: diagnostic changed:\n{blob[-300:]}"
        print(f"[ok] {label} rejected: {needle!r}")

    torch.manual_seed(0)
    # 12/4/3 -> taps 3,6,9 ; 2/4/1 -> L < dil*(w-1) so the guarded p_start=1
    # drops the p=0 tap (row -1): exactly what the removed guard used to do.
    for L, w, dil in ((12, 4, 3), (2, 4, 1)):
        ps = max(0, w - 1 - L // dil)
        rows = [L - (w - 1 - p) * dil for p in range(ps, w - 1)]
        x = torch.randn(L, D, dtype=torch.float32).npu()
        y = torch.zeros(D, dtype=torch.float32).npu()
        _build("good", L, w, dil)(x, y)
        torch.npu.synchronize()
        err = (y.cpu() - x.cpu()[rows].sum(0)).abs().max().item()
        assert err < 1e-6, f"GOOD form wrong (L={L},w={w},dil={dil}): {err}"
        print(f"[ok] GOOD L={L} w={w} dil={dil}: taps={rows} max_err={err:.1e}")
    print("OVERALL: TRAP_PASS")
