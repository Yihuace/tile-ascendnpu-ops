"""Targeted correctness screen for a Stage-4 candidate (fast pre-L0 gate).

Runs a small, adversarial subset of the kernel's own ``_run_case`` checks
(golden = independent fp32 CPU reference in the candidate module):

  * causal smoke (GQA groups=4, D=64)
  * n_kv == 1 (single KV block -> no cross-block atomic contribution)
  * n_kv == 4 causal (concurrent dq row-segment atomics)
  * non-causal (loop_st = 0, mask path fully skipped)
  * D == 128 manifest shape (BN = 32 branch), bf16

Usage: python _check_candidate.py --impl <candidate.py> [--short]
"""

import argparse
import importlib.util
import sys


def _load(path):
    spec = importlib.util.spec_from_file_location("gqa_cand", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gqa_cand"] = mod
    spec.loader.exec_module(mod)
    return mod


SHORT = [
    (1, 128, 8, 2, 64, True, "float16", "screen-smoke-causal"),
    (1, 256, 4, 2, 64, True, "float16", "screen-nkv4-causal"),
    (1, 1024, 8, 4, 64, False, "float16", "screen-noncausal"),
]

FULL_EXTRA = [
    (1, 64, 4, 2, 64, True, "float16", "screen-nkv1"),
    (2, 512, 32, 8, 128, True, "bfloat16", "screen-D128-bf16"),
    (1, 512, 8, 4, 128, True, "float16", "screen-D128-fp16"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--short", action="store_true")
    args = ap.parse_args()
    mod = _load(args.impl)
    cases = SHORT if args.short else SHORT + FULL_EXTRA
    worst = 0.0
    for B, S, H, H_kv, D, causal, dt, tag in cases:
        worst = max(worst, mod._run_case(B, S, H, H_kv, D, causal, dt, tag))
    print(f"SCREEN PASS worst_max_diff={worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
