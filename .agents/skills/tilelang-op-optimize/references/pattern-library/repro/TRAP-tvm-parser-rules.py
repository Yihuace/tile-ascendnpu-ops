"""[repro] TRAP-tvm-parser-rules -- pattern-library/traps-compiler.md

Phenomenon: the TVM script parser has four hard rules (violating them raises
Undefined variable / trace error): (1) `if` statements always generate TIR
runtime ifs (no folding; assignment scope trapped inside the if body), so
trace-time selection must use scalar ternaries or Python-level precompute;
(2) ternaries evaluate BOTH arms (a buffer-slice arm cannot use a sentinel
tensor -> unconditional full-size alloc); (3) conditional T.alloc is illegal
(sentinel only allowed in shape ternary, and that buffer must not be
referenced inside a runtime if body); (4) locals inside `with T.rs()` blocks
are invisible outside (hoist scalar constants to Scope top level).

Assertion (trap, legal-form-pass): the legal forms below compile and produce
correct results in BOTH factory-time modes (FLAG on/off): shape-ternary
sentinel alloc, ternary inside copy call args (trace-time select), and
trace-time zero-bound serial loop with valid-shaped body.

Usage: python TRAP-tvm-parser-rules.py [on|off]   (default runs both modes)

First verified: tilelang dev root build 2026-09-07 (HEAD 21586b5) + CANN 8.5.0
+ Ascend910B2C (debug_log D4).
origin_task (provenance, may rot): multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
Re-verification log (append-only):
  2026-09-10 PASS (formalized from /tmp/opencode/probe_forms.py, both modes).
"""

import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

D = 16


def run_mode(flag: bool) -> bool:
    @tilelang.jit(out_idx=[-1], target="npuir")
    def forms_probe(n):
        guard = flag

        @T.prim_func
        def main(
            a: T.Tensor((8, D), "float32"),
            out: T.Tensor((8, D), "float32"),
        ):
            with T.Kernel(2, is_npu=True) as (cid, vid):
                with T.Scope("Vector"):
                    # legal form 1: shape-ternary sentinel alloc
                    ub = T.alloc_ub([4, D] if guard else [1, 1], "float32")
                    ub2 = T.alloc_ub([4, D], "float32")
                    value = 3.0
                    with T.rs("PIPE_V"):
                        T.vbrc(value, ub2)
                        # legal form 2: ternary inside call args
                        T.copy(a[0:4, 0:D] if guard else a[4:8, 0:D], ub2[0:4, 0:D])
                    # legal form 3: trace-time zero-bound loop, valid body
                    for _gi in T.serial(T.ceildiv(2, 1) if guard else T.ceildiv(0, 1)):
                        with T.rs("PIPE_V"):
                            T.vbrc(value, ub[0:4, 0:D])
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub2[0:4, 0:D], out[0:4, 0:D])

        return main

    f = forms_probe(8)
    a = torch.arange(8 * D, dtype=torch.float32, device="npu").reshape(8, D)
    r = f(a)
    o = r[0] if isinstance(r, tuple) else r
    torch.npu.synchronize()
    got = o[0, :2].cpu().numpy()
    exp = (a[0, :2] if flag else a[4, :2]).cpu().numpy()
    ok = bool(np.allclose(got, exp))
    print(f"FLAG={flag} ternary-select correct: {ok}")
    return ok


def main():
    modes = [True, False]
    if len(sys.argv) > 1:
        modes = [sys.argv[1] == "on"]
    results = {m: run_mode(m) for m in modes}
    assert all(results.values()), f"legal parser forms must pass: {results}"
    print(f"TRAP-tvm-parser-rules: PASS (modes={modes})")


if __name__ == "__main__":
    main()
