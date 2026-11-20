#!/usr/bin/env python3
"""Dump the compiled device source / npuir IR for one conv3d candidate + workload.

Usage:
  python probe_dump_ir.py <module.py> <workload> <dtype> <out.txt> [--block-m N] ...
"""
import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Expert"
os.environ.setdefault("TILELANG_DUMP_IR", "FALSE")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils import NPUUtils  # noqa: E402

WORKLOADS = {
    "smoke-k3-s1-bias": (1, 8, 8, 16, 16, 16, 3, 3, 3, 1, 1, 1),
    "r3d-stem-bias": (1, 3, 16, 112, 112, 64, 3, 3, 3, 1, 1, 1),
    "video-down-bias": (1, 64, 8, 56, 56, 128, 3, 3, 3, 2, 1, 1),
    "unet-encoder-bias": (1, 32, 32, 64, 64, 64, 3, 3, 3, 1, 1, 1),
}
DT = {"float16": "float16", "bfloat16": "bfloat16"}


def load_module(path):
    name = "probe_mod_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("workload")
    ap.add_argument("dtype")
    ap.add_argument("out")
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--block-k", type=int, default=128)
    a = ap.parse_args()

    mod = load_module(os.path.abspath(a.module))
    (n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
     stride, pad, dil) = WORKLOADS[a.workload]
    fn = mod._conv3d_kernel(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
                            stride, pad, dil, True, DT[a.dtype])
    ker = fn(a.block_m, a.block_n, a.block_k, 1)
    src = ker.get_kernel_source()
    with open(a.out, "w") as f:
        f.write(src if isinstance(src, str) else str(src))
    print(f"[probe] kernel source -> {a.out} ({len(src)} bytes)")
    # host source (python-side launch stub) for reference
    try:
        hs = ker.get_host_source()
        with open(a.out + ".host", "w") as f:
            f.write(hs)
        print(f"[probe] host source -> {a.out}.host")
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] host source unavailable: {exc}")


if __name__ == "__main__":
    main()
