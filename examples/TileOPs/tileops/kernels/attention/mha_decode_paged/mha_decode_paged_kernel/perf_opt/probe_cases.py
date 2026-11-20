#!/usr/bin/env python3
"""Targeted precision probe for a single candidate module (fast L0/L1 subset)."""

import argparse
import importlib.util
import os
import sys

import torch

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")


def load(path):
    name = "probe_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = {
    # (B, H, S_q, N_kv, D, page, causal, dtype), kwargs
    "full-page-d64": ((1, 8, 1, 1024, 64, 256, False, torch.float16), dict(seed=2)),
    "full-page-d128": ((1, 16, 1, 512, 128, 128, False, torch.float16), dict(seed=0)),
    "partial-200": ((1, 8, 1, 512, 64, 256, False, torch.float16),
                    dict(real_kv_list=[200], seed=6)),
    "partial-shuffled": ((1, 8, 1, 512, 64, 256, False, torch.float16),
                         dict(real_kv_list=[300], shuffle_pages=True, seed=9)),
    "zero-len": ((1, 8, 1, 512, 64, 256, False, torch.float16),
                 dict(real_kv_list=[0], seed=7, expect_zero=True)),
    "causal": ((1, 8, 1, 512, 64, 256, True, torch.float16), dict(seed=14)),
    "nsplit2": ((1, 8, 1, 512, 64, 256, False, torch.float16), dict(num_split=2, seed=15)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--cases", default="full-page-d64,full-page-d128,partial-200,partial-shuffled,zero-len")
    args = ap.parse_args()
    mod = load(args.module)
    failed = []
    for tag in [c.strip() for c in args.cases.split(",") if c.strip()]:
        shape, kw = CASES[tag]
        try:
            mod._run_case(*shape, tag, **kw)
        except AssertionError as exc:
            print(f"  [{tag}] FAIL: {str(exc)[:160]}")
            failed.append(tag)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{tag}] ERROR: {type(exc).__name__}: {str(exc)[:160]}")
            failed.append(tag)
    print(f"[probe] {os.path.basename(args.module)}: "
          f"{'PASS' if not failed else 'FAIL ' + ','.join(failed)}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
