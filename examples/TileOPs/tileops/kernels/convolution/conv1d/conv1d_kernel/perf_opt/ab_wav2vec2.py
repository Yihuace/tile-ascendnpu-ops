#!/usr/bin/env python3
"""A/B: does the ``wav2vec2-layer1-fp16`` bench case crash with BOTH kernels?

Runs the Op layer (``Conv1dFwdOp``, stride=5) for the wav2vec2 shape and lets the
caller pick which general-kernel module supplies ``_conv1d_kernel``:

  --kernel perf_opt   -> the Stage-4 tuned module (wrapper default)
  --kernel baseline   -> the Stage-3 module, injected via a module-attribute
                         monkeypatch of ``tileops.kernels.convolution.conv1d``
                         (no file is modified)

Prints ``AB <kernel> PASS|CRASH|EXC`` so a "partial" report can be attributed to
a pre-existing compiler defect instead of this round's change.

Usage: python3 ab_wav2vec2.py {perf_opt|baseline}
"""
import argparse
import importlib
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["perf_opt", "baseline"], required=True)
    args = ap.parse_args()

    wrapper = importlib.import_module("tileops.kernels.convolution.conv1d")
    if args.kernel == "baseline":
        base_mod = importlib.import_module(
            "tileops.kernels.convolution.conv1d.conv1d_kernel._conv1d_kernel"
        )
        wrapper._conv1d_kernel = base_mod._conv1d_kernel
        tag = "baseline"
    else:
        tag = "perf_opt"

    from tileops.ops.convolution import Conv1dFwdOp

    op = Conv1dFwdOp(stride=5, padding=0, dilation=1, groups=1)
    x = torch.randn(1, 1, 16000, dtype=torch.float16, device="npu")
    w = torch.randn(512, 1, 10, dtype=torch.float16, device="npu")
    out = op(x, w)
    torch.npu.synchronize()
    ref = torch.nn.functional.conv1d(x.cpu().float(), w.cpu().float(), stride=5)
    err = float((out.cpu().float() - ref).abs().max().item())
    print(f"AB {tag} PASS max_abs={err:.4e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"AB {sys.argv[-1]} EXC {type(exc).__name__}: {str(exc)[:200]}")
        sys.exit(2)
