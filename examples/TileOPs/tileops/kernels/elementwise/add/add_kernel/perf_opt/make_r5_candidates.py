#!/usr/bin/env python3
"""R5 candidate generator for AddFwdOp (Stage-4 round 5).

Derives every round-5 experiment branch from the frozen current best
(``perf_opt/AddFwdOp.py``) by a *single* textual change each, so the branches
stay diff-auditable:

  B1  native bf16 vadd      -- fast path extends to bf16 (fp16 unchanged)
  B2  direct fp32 vadd      -- fast path extends to fp32 (bf16 unchanged);
                               drops the 3 no-information UB transit copies
  B3  all-native + pipe3    -- B1 + B2 merged, plus num_stages 2 -> 3

Every branch keeps ``TUNED_BLOCK_SIZE`` and all other structure identical.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "AddFwdOp.py")

OLD_COND = 'if dtype == "float16" and fold and out_dtype == "float16":'
NEW_COND = "if native_fast:"

FLAG_ANCHOR = "    fold = alpha == 1\n    alpha_scalar = float(alpha)\n"

BRANCHES = {
    "AddFwdOp_round5_b1_native_bf16.py": (
        '    # R5-B1: alpha==1 native same-dtype vadd extended to bf16.  bf16 has\n'
        '    # an 8-bit mantissa, so the exact sum of two bf16 values only needs\n'
        '    # fp32 rounding when the exponent gap exceeds 2**24; for every case\n'
        '    # the kernel is measured on, the fp32 sum is exact and "fp32 add then\n'
        '    # bf16 round" == "native bf16 add" (bitwise, verified in\n'
        '    # perf_opt/logs/r5_bitwise_b1.log).  Saves 3 of 4 vector passes.\n'
        '    native_fast = fold and out_dtype == dtype and dtype in ("float16", "bfloat16")\n',
        None,
    ),
    "AddFwdOp_round5_b2_fp32_direct.py": (
        '    # R5-B2: fp32 needs no opmath transit at all -- the fp32 branch used\n'
        '    # to copy a_ub->a32, b_ub->b32, add, copy a32->y_ub, i.e. three\n'
        '    # no-information UB copies.  Native fp32 vadd(a_ub, b_ub, y_ub) is the\n'
        '    # same arithmetic (bitwise identical), 3 vector passes shorter.\n'
        '    native_fast = (\n'
        '        fold and out_dtype == dtype and dtype in ("float16", "float32")\n'
        '    )\n',
        None,
    ),
    "AddFwdOp_round5_b3_allnative_pipe3.py": (
        '    # R5-B3: B1+B2 merged (alpha==1 native vadd for every same-dtype\n'
        '    # case) plus a 3-stage software pipeline.\n'
        '    native_fast = fold and out_dtype == dtype\n',
        ("num_stages=2", "num_stages=3"),
    ),
}


def build(name, flag_snippet, extra_sub):
    with open(BASE, encoding="utf-8") as handle:
        src = handle.read()
    if OLD_COND not in src:
        raise SystemExit("anchor condition not found in current best")
    src = src.replace(OLD_COND, NEW_COND)
    src = src.replace(FLAG_ANCHOR, FLAG_ANCHOR + flag_snippet)
    if extra_sub is not None:
        old, new = extra_sub
        if old not in src:
            raise SystemExit(f"anchor {old!r} not found")
        src = src.replace(old, new)
    # Keep the module docstring honest about the branch origin.
    src = re.sub(
        r"Stage-4 tuning \(this file is the tuned drop-in; see ``perf_opt/opt_log.md``\):",
        "Stage-4 tuning (this file is the tuned drop-in; see ``perf_opt/opt_log.md``)\n"
        "    R5 branch: see ``perf_opt/logs/r5_*`` / opt_log.md \"Round 5\".\n"
        "Stage-4 tuning history:",
        src,
        count=1,
    )
    out = os.path.join(HERE, name)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(src)
    print(f"wrote {name} ({len(src)} bytes)")


if __name__ == "__main__":
    for branch, (snippet, extra) in BRANCHES.items():
        build(branch, snippet, extra)
