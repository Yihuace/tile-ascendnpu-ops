#!/usr/bin/env python3
"""Correctness check of a candidate at its *tuned* block_size.

The embedded `--level L0` suite runs at the module default block_size, so a
candidate that changes block_size needs this extra gate.  Runs one same-shape
case per dtype at the requested block size (alpha=1 fold + alpha=2 scale) and
raises on tolerance violation.

    python check_block_size.py <kernel.py> <block_size>
"""

import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def load(path):
    name = "bs_check_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    path = os.path.abspath(sys.argv[1])
    block_size = int(sys.argv[2])
    module = load(path)
    for dtype_str in ("float16", "bfloat16", "float32"):
        for alpha in (1, 2):
            module._run_case(
                1 << 20,
                dtype_str,
                f"bs-check-{block_size}",
                alpha=alpha,
                block_size=block_size,
            )
    # Non-divisible tail at the tuned block size.
    module._run_case(10000 + 7, "float16", f"bs-tail-{block_size}", block_size=block_size)
    print(f"[bs-check] PASS block_size={block_size} ({path})")


if __name__ == "__main__":
    main()
