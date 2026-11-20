#!/usr/bin/env python3
"""Bench entry for ab_test.py: launch a candidate kernel N times (no msprof).

ab_test.py wraps this in ``msprof op``; this script only issues the launches.

Usage:
  python bench_entry.py --kernel-file <path.py> --N <n> --dtype int32 \
      [--launch-count 20] [--warm-up 5] [--device-index 0]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

TORCH_DTYPE = {
    "bool": torch.bool, "uint8": torch.uint8, "int8": torch.int8,
    "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-file", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", required=True, choices=tuple(TORCH_DTYPE))
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--device-index", type=int, default=0)
    args = ap.parse_args()

    torch.npu.set_device(args.device_index)
    spec = importlib.util.spec_from_file_location(
        "bench_cand", os.path.abspath(args.kernel_file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_cand"] = mod
    spec.loader.exec_module(mod)

    x = torch.randint(-128, 128, (args.N,), device="npu",
                      dtype=TORCH_DTYPE[args.dtype])
    kernel = mod.bitwise_not_fwd_kernel(args.N, args.dtype)
    torch.npu.synchronize()
    for _ in range(args.warm_up):
        y = kernel(x, args.N)
    torch.npu.synchronize()
    for _ in range(args.launch_count):
        y = kernel(x, args.N)
    torch.npu.synchronize()
    print("done", y.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
