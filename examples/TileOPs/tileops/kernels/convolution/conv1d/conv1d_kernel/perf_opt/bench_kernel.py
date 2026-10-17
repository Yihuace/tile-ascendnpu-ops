#!/usr/bin/env python3
"""Standalone kernel driver for msprof (Stage 4 workflow-compliant measurement).

Runs ONE benchmark workload directly against the Stage-3 kernel file or a
``perf_opt/`` candidate file -- no Stage-5 wrapper, no benchmark harness.  The
process launches the target kernel ``--launches`` times so that

    msprof op --kernel-name=<name> --launch-count=<n> --warm-up=<m>

can collect a stable ``Task Duration(us)``.

Usage:
  python perf_opt/bench_kernel.py --module <path-to-kernel .py> \
      --workload whisper-conv1 --dtype float16 --bias 0 --launches 25

The msprof command is printed to stderr so the log records the actual file and
workload (workflow requirement: the command must identify the candidate).
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # force; .bashrc exports Developer

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# label -> (n, c_in, l_in, c_out, k, stride, pad, dilation, has_bias)
WORKLOADS = {
    "smoke-k3-s1":       (1, 8, 128, 16, 3, 1, 0, 1, False),
    "whisper-conv1":     (1, 80, 3000, 1280, 3, 1, 0, 1, False),
    "wav2vec2-layer1":   (1, 1, 16000, 512, 10, 5, 0, 1, False),
    "encodec-init":      (1, 1, 24000, 32, 7, 1, 0, 1, False),
    "encodec-deep":      (1, 128, 600, 256, 10, 1, 0, 1, False),
    "whisper-conv1-bias":   (1, 80, 3000, 1280, 3, 1, 0, 1, True),
    "wav2vec2-layer1-bias": (1, 1, 16000, 512, 10, 5, 0, 1, True),
    "encodec-init-bias":    (1, 1, 24000, 32, 7, 1, 0, 1, True),
    "encodec-deep-bias":    (1, 128, 600, 256, 10, 1, 0, 1, True),
}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
BLOCK = (64, 128, 128)  # module defaults: block_m, block_n, block_k
TAPS = 1
SLOTS = 2


def load_module(path):
    name = "perf_candidate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="kernel .py file to measure")
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    ap.add_argument("--bias", type=int, default=-1, help="-1 = workload default")
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    n, c_in, l_in, c_out, k, stride, pad, dil, wl_bias = WORKLOADS[args.workload]
    has_bias = wl_bias if args.bias < 0 else bool(args.bias)
    dtype = DTYPES[args.dtype]

    if stride != 1 or k <= 1 or c_in < 16:
        sys.exit(
            f"workload {args.workload} dispatches to the general conv1d kernel; "
            "this driver currently measures the unit_stride kernel only"
        )

    mod = load_module(args.module)
    factory = mod._conv1d_unit_stride_kernel
    plan = mod._plan
    fn = factory(n, c_in, l_in, c_out, k, stride, pad, pad, dil, has_bias,
                 args.dtype)
    kernel = fn(*BLOCK, TAPS, 1)

    torch.manual_seed(0)
    x = torch.randn(n, c_in, l_in, dtype=dtype, device="npu")
    w = torch.randn(c_out, c_in, k, dtype=dtype, device="npu")
    bias = torch.randn(c_out, dtype=dtype, device="npu") if has_bias else None

    l_pad, c_in_pad, _ = plan(l_in, c_in, pad, pad)
    w_flat = torch.zeros((k, c_out, c_in_pad), dtype=dtype, device="npu")
    x_pad = torch.zeros((n, c_in_pad, l_pad), dtype=dtype, device="npu")
    nk = __import__("tilelang.utils", fromlist=["NPUUtils"]).NPUUtils.get().get_aicore_num()

    ws = torch.empty((nk, SLOTS) + BLOCK, dtype=torch.float32, device="npu") if has_bias else None

    print(f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
          f"dtype={args.dtype} bias={has_bias} launches={args.launches}",
          file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            if has_bias:
                kernel(x, w, bias, w_flat, x_pad, ws)
            else:
                kernel(x, w, w_flat, x_pad)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
