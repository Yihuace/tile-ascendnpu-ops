# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Single-workload launcher for the engram_decode Stage 4 profiling runs.

Launches one manifest workload of a *file-selected* kernel module under
``msprof op`` (kernel-only Task Duration is the sole latency metric).

Usage:
    python bench_driver.py --module <kernel .py> --case <workload_id> [--iters N]

The workload parameters are the 4 official manifest workloads expanded by
``benchmarks/ops/bench_engram_decode.py::_engram_decode_params``; they are
reproduced here verbatim so the driver never depends on the TileOPs op layer.

The module is imported by path, so experiment candidates
(``_engram_decode_kernel_round2_*.py``) can be profiled without touching the
baseline or the wrapper.
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401  enables .npu() tensor placement

# (label, batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, marks)
CASES = {
    "decode-smoke-b1-dmem256-d256-fp16": (1, 256, 256, 9, 4, 3, "float16", "smoke"),
    "decode-b1-dmem512-d256-fp16": (1, 512, 256, 12, 4, 3, "float16", "full"),
    "decode-b4-dmem1024-d512-fp16": (4, 1024, 512, 20, 4, 5, "float16", "full"),
    "decode-b8-dmem512-d256-bf16": (8, 512, 256, 18, 4, 3, "bfloat16", "full"),
}

STATE_MODULES = ("perf_opt",)


def _load_module(path):
    path = os.path.abspath(path)
    name = "engram_cand_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check", action="store_true",
                    help="also verify against the module golden once")
    args = ap.parse_args()

    batch, d_mem, d, L, w, dil, dtype_s, _marks = CASES[args.case]
    dtype = getattr(torch, dtype_s)
    mod = _load_module(args.module)

    eps = 1e-6
    kernel = mod._engram_decode_kernel(batch, d_mem, d, L, w, dil, eps, dtype_s)
    ins = mod._make_inputs(batch, d_mem, d, L, w, dtype, 0)
    e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w = ins

    d_pad = mod._align_up(d, mod.ALIGNMENT)
    e_n = e_t.npu()
    h_n = h_t.npu()
    cs_n = mod._pad_state(cs, L).npu()
    WK_n, WV_n = W_K.npu(), W_V.npu()
    wh_n, wv_n = rms_w_h.npu(), rms_w_v.npu()
    cw_n = conv_w.npu()
    pad = d_pad - d
    if pad:
        h_n = mod._pad_last(h_n, d_pad)
        cs_n = mod._pad_last(cs_n, d_pad)
        WK_n = mod._pad_last(WK_n, d_pad)
        WV_n = mod._pad_last(WV_n, d_pad)
        wh_n = mod._pad_last(wh_n, d_pad)
        wv_n = mod._pad_last(wv_n, d_pad)
        cw_n = mod._pad_last(cw_n, d_pad)

    argsx = (e_n, h_n, cs_n, WK_n, WV_n, wh_n, wv_n, cw_n)

    if args.check:
        y, ns = kernel(*argsx)
        y_ref, ns_ref = mod.golden_engram_decode(
            e_t, h_t, cs, W_K, W_V, rms_w_h, rms_w_v, conv_w, L, dil, eps)
        yv = y.float().cpu()[:, :d]
        nsv = ns.float().cpu()[:, :, :d]
        yd = (yv - y_ref.float()).abs().max().item()
        sd = (nsv - ns_ref.float()).abs().max().item()
        print(f"[check] {args.case} y_maxdiff={yd:.3e} st_maxdiff={sd:.3e}")
        torch.npu.synchronize()

    for _ in range(args.iters):
        kernel(*argsx)
    torch.npu.synchronize()
    print(f"[done] {args.module} :: {args.case} iters={args.iters}")


if __name__ == "__main__":
    main()
