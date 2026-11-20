#!/usr/bin/env python3
"""Standalone kernel driver for msprof (Stage 4 workflow-compliant measurement).

Runs ONE benchmark workload directly against the Stage-3 kernel file or a
``perf_opt/`` candidate file -- no Stage-5 wrapper, no benchmark harness.  The
process launches the target kernel ``--launches`` times so that

    msprof op --kernel-name=<name> --launch-count=<n> --warm-up=<m>

can collect a stable ``Task Duration(us)``.

Usage:
  python perf_opt/bench_kernel.py --module <path-to-kernel .py> \
      --workload r3d-stem-bias --dtype float16 --launches 25
"""

import argparse
import importlib.util
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # force; .bashrc exports Developer

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils import NPUUtils  # noqa: E402

# label -> (n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw, stride, pad, dil, bias)
WORKLOADS = {
    "smoke-k3-s1-bias":   (1, 8, 8, 16, 16, 16, 3, 3, 3, 1, 1, 1, True),
    "r3d-stem-bias":      (1, 3, 16, 112, 112, 64, 3, 3, 3, 1, 1, 1, True),
    "video-down-bias":    (1, 64, 8, 56, 56, 128, 3, 3, 3, 2, 1, 1, True),
    "unet-encoder-bias":  (1, 32, 32, 64, 64, 64, 3, 3, 3, 1, 1, 1, True),
}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
BLOCK = (64, 128, 128)  # module defaults: block_m, block_n, block_k
NUM_STAGES = 1
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
    ap.add_argument("--launches", type=int, default=25)
    ap.add_argument("--block-m", type=int, default=None)
    ap.add_argument("--block-n", type=int, default=None)
    ap.add_argument("--block-k", type=int, default=None)
    args = ap.parse_args()
    block = list(BLOCK)
    if args.block_m: block[0] = args.block_m
    if args.block_n: block[1] = args.block_n
    if args.block_k: block[2] = args.block_k

    (n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
     stride, pad, dil, wl_bias) = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]

    mod = load_module(args.module)
    factory = mod._conv3d_kernel
    host_prep = mod._host_prep
    ws_dims = mod._ws_dims

    # stride/pad/dil are ints here; the kernel's _as3 broadcasts to 3-tuples.
    fn = factory(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
                 stride, pad, dil, wl_bias, args.dtype)
    kernel = fn(*block, NUM_STAGES)

    sd = sh = sw = stride
    pd = ph = pw = pad
    dd = dh = dw = dil
    out_d = (d_in + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    out_h = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    d_pad = d_in + 2 * pd
    h_pad = h_in + 2 * ph
    need_poly, w_pad, mdec, c_in_pad = ws_dims(w_in, c_in, pw, sw)

    torch.manual_seed(0)
    x = torch.randn(n, c_in, d_in, h_in, w_in, dtype=dtype, device="npu")
    weight = torch.randn(c_out, c_in, kd, kh, kw, dtype=dtype, device="npu")
    bias = torch.randn(c_out, dtype=dtype, device="npu")

    x_pad, xdec, w_flat = host_prep(
        x, weight, c_in_pad, d_pad, h_pad, w_pad, sd, sh, sw, pd, ph, pw,
        d_in, h_in, w_in, kd, kh, kw, need_poly, mdec, dtype,
    )

    nk = NPUUtils.get().get_aicore_num()
    ws = torch.empty(
        (nk, SLOTS, block[0], block[1]), dtype=torch.float32, device="npu"
    )

    print(f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
          f"dtype={args.dtype} bias={wl_bias} launches={args.launches}",
          file=sys.stderr)
    with torch.no_grad():
        for _ in range(args.launches):
            kernel(x_pad, w_flat, bias, xdec, ws)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
