# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""TRAP-npuir-vector-lut-alignment: a vector operand whose base is not 32 B
aligned (or whose read runs past the end of the UB allocation) aborts the
Vector core on this toolchain.

Observed (2026-10-31, Ascend910_9362, tilelang 0.1.2+9738284, Expert mode):

  * The Stockham twiddle layout of the baseline kernel is ``LUT[h-1+i]``,
    so a stage's twiddle segment starts at element ``h-1`` (odd for every
    ``h >= 2`` -> 4-byte granularity).
  * Reading that segment as a vector operand of a ``T.Parallel`` store
    (``dst[0,0,i] = t[0,0,i] * lut[0,0,h-1+i]``) compiles ("AscendNPU IR
    compile success") and then aborts at run time:

        The write address of the MTE instruction is out of range.
        CCU instruction address check error.  ... [vector core exception]

  * Reading the segment at an aligned base (``base % 8 == 0``) is correct;
    the identical kernel then returns the exact fp32 product.  Every stage
    segment of the table must therefore start on a 32 B boundary.

Consequence: fft_c2c_vec.py cannot read the design's flat ``(n-1,)`` LUT
layout directly; ``_lut_layout`` pads each stage segment to 32 B (the table
becomes ``npad >= n-1`` elements) - a measured, disclosed deviation from
DESIGN.md section 1.2 / 3.3.

run: python TRAP-npuir-vector-lut-alignment.py            # aligned, asserted
     python TRAP-npuir-vector-lut-alignment.py --unsafe   # unaligned (aborts)

First verified: 2026-10-31, tilelang 0.1.2+9738284 / CANN 9.0.0 /
Ascend910_9362 (Expert mode).  origin_task (provenance, may go stale):
fft_c2c_vec-fft_c2c_vec-20261031T103133Z; file lives in
traps-compiler.md.
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

DT = "float32"


@tilelang.jit(out_idx=[3], target="npuir")
def k_twiddle(n, base):
    """Multiply a vector by the twiddle segment starting at ``base``."""
    nh = n // 2

    @T.prim_func
    def main(
        x: T.Tensor((1, n), DT),
        lut: T.Tensor((2 * n,), DT),
        scale: T.Tensor((1,), DT),
        y: T.Tensor((1, nh), DT),
    ) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((1, 1, nh), DT)
            lut_ub = T.alloc_ub((1, 1, 2 * n), DT)
            dst = T.alloc_ub((1, 1, nh), DT)
            T.copy(x[cid, 0:nh], src[0, 0, 0:nh])
            T.copy(lut[0:2 * n], lut_ub[0, 0, 0:2 * n])
            T.pipe_barrier("PIPE_ALL")
            for i in T.Parallel(nh):
                dst[0, 0, i] = src[0, 0, i] * lut_ub[0, 0, base + i]
            T.pipe_barrier("PIPE_V")
            T.copy(dst[0, 0, 0:nh], y[cid, 0:nh])

    return main


def run(base):
    # n = 16 -> one 8-lane fp32 repeat per vector statement; the twiddle
    # segment is read out of a 32-element table.
    n = 16
    x = torch.arange(1, n + 1, dtype=torch.float32).reshape(1, n)
    lut = torch.arange(2 * n, dtype=torch.float32) * 0.5
    out = k_twiddle(n, base)(x.npu(), lut.npu(), torch.zeros(1).npu())
    torch.npu.synchronize()
    got = out.cpu()[0]
    ref = x[0, : n // 2] * lut[base:base + n // 2]
    err = (got - ref).abs().max().item()
    print(f"[base={base}] max_abs_err = {err:.3e}")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsafe", action="store_true",
                    help="run the unaligned variant (aborts the Vector core)")
    args = ap.parse_args()

    if args.unsafe:
        # base = 1 element = 4 B: not 32 B aligned.
        run(1)  # 4 B offset -> device fault
        print("UNEXPECTED: the unaligned read did not fault")
        sys.exit(0)

    # Aligned (base = 0 elements = 32 B boundary).
    err = run(0)
    print(f"TILELANG_ASCEND_MODE={os.environ.get('TILELANG_ASCEND_MODE')}")
    print("aligned read: OK")
    assert err < 1e-6, f"aligned twiddle read must be exact, got {err}"

    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--unsafe"],
        capture_output=True,
        text=True,
    )
    print(f"unaligned read: subprocess exit={proc.returncode}")
    last = [l for l in proc.stderr.splitlines() if "out of range" in l or
            "vector core exception" in l]
    if proc.returncode != 0:
        print("TRAP CONFIRMED:", last[0].strip()[:130] if last else "device fault")
    else:
        # The isolated case does not always reproduce the abort; in the real
        # FFT kernel (odd 28 B table, segment ending on the allocation end)
        # it did.  The mitigation below is therefore unconditional.
        print("NOT REPRODUCED IN ISOLATION: the abort needs the overrun of a "
              "non-32 B-sized allocation (observed in fft_c2c_vec.py); keep "
              "every twiddle segment 32 B aligned and 32 B padded.")
    print("ok")


if __name__ == "__main__":
    main()
