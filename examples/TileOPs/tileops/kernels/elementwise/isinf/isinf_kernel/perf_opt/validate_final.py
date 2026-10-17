#!/usr/bin/env python3
"""Final-version acceptance for SubFwdOp: tuned block size + smoke broadcast case.

Loads the final kernel module by path and:
  1. runs the smoke workload's exact shapes (same-shape or broadcast as recorded
     in workload_inventory.json) through the module's own golden comparison;
  2. runs the linear-bias fp16/bf16 same-shape case at the tuned block size.

    python validate_final.py AddFwdOp.py --block-size 8192
"""

import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    name = "final_validate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--block-size", type=int, default=8192)
    args = ap.parse_args()

    module = load(os.path.abspath(args.module))
    inventory = json.load(open(os.path.join(HERE, "workload_inventory.json")))
    smoke = [w for w in inventory["workloads"] if w["kind"] == "smoke"]
    if not smoke:
        raise SystemExit("no smoke workload in inventory")

    for item in smoke:
        shape = item["shape"]
        a_shape = tuple(shape["input"])
        b_shape = tuple(shape["other"])
        if a_shape == b_shape:
            module._run_case(
                item["N"], item["dtype"], f"smoke/{item['workload_id']}",
                alpha=item["params"].get("alpha", 1),
                block_size=args.block_size,
            )
        else:
            module._run_broadcast_case(
                a_shape, b_shape, item["dtype"], f"smoke/{item['workload_id']}",
                alpha=item["params"].get("alpha", 1),
            )

    for dtype_str in ("float16", "bfloat16"):
        module._run_case(
            1024 * 4096, dtype_str, "linear-bias", alpha=1, block_size=args.block_size
        )
    print(f"[final-validate] PASS module={args.module} block_size={args.block_size}")


if __name__ == "__main__":
    main()
