#!/usr/bin/env python3
"""Precision check for ONE general-conv1d shape through a kernel module.

Usage:
  python3 prec_case.py <module.py> <case> [<case> ...]

``case`` is one of the named benchmark shapes (or ``all``).  Each case runs the
module's own ``_run_case`` (golden = fp32 F.conv1d), i.e. exactly the L0 check,
and prints ``PREC <case> PASS|FAIL max_abs=...``.
"""
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# name -> (n, c_in, l_in, c_out, k, stride, pad, dilation, has_bias)
CASES = {
    "encodec-init":      (1, 1, 24000, 32, 7, 1, 0, 1, False),
    "encodec-init-bias": (1, 1, 24000, 32, 7, 1, 0, 1, True),
    "wav2vec2":          (1, 1, 16000, 512, 10, 5, 0, 1, False),
    "wav2vec2-bias":     (1, 1, 16000, 512, 10, 5, 0, 1, True),
    "smoke":             (1, 8, 128, 16, 3, 1, 0, 1, False),
    "smoke-bias":        (1, 8, 128, 16, 3, 1, 0, 1, True),
    "nondiv":            (1, 50, 1000, 100, 7, 1, 0, 1, False),
    "batch3-bias":       (3, 96, 300, 192, 3, 1, 0, 1, True),
    "dilation-bias":     (1, 50, 1000, 100, 3, 1, 3, 2, True),
    "stride3":           (2, 64, 512, 128, 4, 3, 2, 1, False),
    # regression-triplet cases from the report's correctness run
    "pad-wider":         (1, 17, 5, 7, 3, 1, 5, 2, False),
    "dilation2":         (1, 64, 512, 128, 3, 1, 1, 2, False),
}


def main():
    mod_path = sys.argv[1]
    names = sys.argv[2:]
    if not names or names == ["all"]:
        names = list(CASES)
    spec = importlib.util.spec_from_file_location("cand", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok_all = True
    for name in names:
        n, c_in, l_in, c_out, k, s, p, d, bias = CASES[name]
        for dt in (torch.float16, torch.bfloat16):
            tag = f"{name}-{'bias' if bias else 'nobias'}-{dt}"
            ok, mx = mod._run_case(n, c_in, l_in, c_out, k, s, p, p, d, bias,
                                   dt, tag, seed=0)
            print(f"PREC {tag} {'PASS' if ok else 'FAIL'} max_abs={mx:.4e}")
            ok_all &= ok
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
