# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PROBE-ew-block-size -- sweep `block_size` for the elementwise NPU kernels.

Why: the delivered elementwise kernels hard-code ``block_size = 2048`` elements
per block, and each block holds three UB buffers of that width.  For fp32 that is
3 x 2048 x 4 = 24 KB out of the 192 KB UB, so the block is ~8x smaller than the
buffers could be.  The measured effective bandwidth is only ~0.19 TB/s (10.7% of
the 1.8 TB/s peak) for the 16M/64M cases, and the bottleneck is the number of
GM<->UB tile round trips -- i.e. exactly what a larger block reduces.

This probe times the real kernel with several block sizes so the tuning decision
is data-driven instead of guessed.

Run:
    env -u PYTHONPATH <venv>/bin/python -u PROBE-ew-block-size.py --op abs
    ... --op cos --n 1048576
"""

import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

# This probe lives deep inside the package tree, so the TileOPs root must be put
# on sys.path explicitly (running the file by path does not do it).
_TILEOPS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             *([os.pardir] * 6)))
if _TILEOPS_ROOT not in sys.path:
    sys.path.insert(0, _TILEOPS_ROOT)

import argparse
import time

import torch
import torch_npu  # noqa: F401

from tileops.kernels.elementwise.abs.abs_kernel.abs import abs_fwd_kernel
from tileops.kernels.elementwise.ceil.ceil_kernel.ceil import ceil_fwd_kernel
from tileops.kernels.elementwise.cos.cos_kernel.cos import cos_fwd_kernel
from tileops.kernels.elementwise.exp.exp_kernel.exp import exp_fwd_kernel
from tileops.kernels.elementwise.floor.floor_kernel.floor import floor_fwd_kernel

FACTORIES = {
    "abs": abs_fwd_kernel,
    "ceil": ceil_fwd_kernel,
    "cos": cos_fwd_kernel,
    "exp": exp_fwd_kernel,
    "floor": floor_fwd_kernel,
}

BLOCK_SIZES = [2048, 4096, 8192, 16384, 32768]
DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


def time_kernel(fn, x, n, iters=12, warmup=3):
    for _ in range(warmup):
        fn(x, n)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, n)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6       # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="abs", choices=sorted(FACTORIES))
    ap.add_argument("--n", type=int, default=1048576)
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    a = ap.parse_args()

    dt = DTYPES[a.dtype]
    x = torch.randn(a.n, dtype=dt, device="npu")
    bytes_moved = 2 * a.n * dt.itemsize

    print(f"--- {a.op}  N={a.n}  dtype={a.dtype}  "
          f"(bytes moved = {bytes_moved / 1e6:.2f} MB) ---", flush=True)
    print(f"    {'block':>7}  {'latency':>10}  {'bandwidth':>11}  {'% of 1.8TB/s':>13}",
          flush=True)
    base = None
    for bs in BLOCK_SIZES:
        try:
            k = FACTORIES[a.op](a.n, a.dtype, block_size=bs)
        except Exception as exc:  # noqa: BLE001
            print(f"    {bs:>7}  build FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:70]}", flush=True)
            continue
        if a.n % bs:
            print(f"    {bs:>7}  skipped (N not a multiple)", flush=True)
            continue
        lat = time_kernel(k, x, a.n)
        bw = bytes_moved / (lat * 1e-6) / 1e12
        if base is None:
            base = lat
        print(f"    {bs:>7}  {lat:>9.2f}us  {bw:>9.3f}TB/s  {bw / 1.8 * 100:>12.1f}%"
              f"  ({base / lat:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
