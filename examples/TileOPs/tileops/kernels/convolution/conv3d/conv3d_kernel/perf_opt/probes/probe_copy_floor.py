#!/usr/bin/env python3
"""GM->L1 copy-floor microbenchmark for the conv3d im2col copy shape.

Question it answers: for the Cube-side `T.copy(GM 2D tile -> L1)` used by the
conv3d implicit GEMM, is the cost driven by (a) the number of source rows
(segment count), (b) the source row stride, or (c) bytes?  That decides whether
"wider N per task" can pay off.

Each config builds one Cube-only kernel:
    for it in range(iters):
        T.copy(src[0:rows, 0:row_len], l1_b)        # the measured copy
        T.gemm(l1_a, l1_b, l0_c, size=[16, rows, row_len])
    T.copy(l0_c[0:16, 0:16], out)                   # keep the chain live

Reported throughput = bytes(iters * rows * row_len * 2) / median launch time.

Usage:
  python probe_copy_floor.py            # full sweep, wall-clock median
  python probe_copy_floor.py --configs "64,64,179520"   # one config
"""
import argparse
import os
import statistics
import time

os.environ["TILELANG_ASCEND_MODE"] = "Expert"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from tilelang.utils import NPUUtils  # noqa: E402

ITERS = 64
LAUNCHES = 20
WARMUP = 5
CORES = 20


def build(rows, row_len, stride, dtype="float16"):
    @tilelang.jit(out_idx=[1], target="npuir")
    def factory():
        @T.prim_func
        def main(
            src: T.Tensor((rows, stride), dtype),
            out: T.Tensor((16, 16), "float32"),
        ):
            with T.Kernel(CORES, is_npu=True) as (kid, _):
                with T.Scope("Cube"):
                    l1_a = T.alloc_L1([16, rows], dtype)
                    l1_b = T.alloc_L1([rows, row_len], dtype)
                    l0_c = T.alloc_L0C([16, row_len], "float32")
                    with T.rs("PIPE_MTE2"):
                        T.copy(src[0:16, 0:rows], l1_a)
                    for it in T.serial(ITERS):
                        with T.rs("PIPE_MTE2"):
                            T.copy(src[0:rows, 0:row_len], l1_b)
                        with T.rs("PIPE_C"):
                            T.gemm(l1_a, l1_b, l0_c, initC=True,
                                   b_transpose=False,
                                   size=[16, rows, row_len])
                    with T.rs("PIPE_FIX"):
                        T.copy(l0_c[0:16, 0:16], out)

        return main

    return factory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None,
                    help="semicolon list rows,row_len,stride")
    a = ap.parse_args()
    if a.configs:
        cfgs = [tuple(int(x) for x in c.split(",")) for c in a.configs.split(";")]
    else:
        cfgs = [
            # (rows, row_len, stride)   stride 179520 ~ unet conv3d channel plane
            (32, 64, 179520),     # today's unet B copy (32 x 128B, huge stride)
            (32, 160, 179520),    # 2 padded rows
            (32, 480, 179520),    # 6 padded rows
            (32, 960, 179520),    # 12 padded rows
            (32, 64, 1584),       # small stride (video-down-like)
            (32, 480, 1584),      # small stride + wide rows
            (32, 64, 128),        # fully packed stride
            (32, 480, 128),       # packed + wide
            (16, 112, 262656),    # r3d-stem B copy (16 x 224B)
            (64, 64, 179520),     # rows doubled
        ]
    print(f"{'rows':>5} {'row_len':>8} {'stride':>8} {'bytes/iter':>11} "
          f"{'median_us':>10} {'GB/s':>8}")
    for (rows, row_len, stride) in cfgs:
        try:
            ker = build(rows, row_len, stride)
            src = torch.randn(rows, stride, dtype=torch.float16, device="npu")
            for _ in range(WARMUP):
                ker(src)
            torch.npu.synchronize()
            ts = []
            for _ in range(LAUNCHES):
                t0 = time.perf_counter()
                ker(src)
                torch.npu.synchronize()
                ts.append((time.perf_counter() - t0) * 1e6)
            med = statistics.median(ts)
            nbytes = ITERS * rows * row_len * 2
            gbs = nbytes / (med * 1e-6) / 1e9
            print(f"{rows:5d} {row_len:8d} {stride:8d} {nbytes:11d} "
                  f"{med:10.2f} {gbs:8.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{rows:5d} {row_len:8d} {stride:8d}  FAILED: "
                  f"{str(exc).splitlines()[0][:120]}")


if __name__ == "__main__":
    main()
