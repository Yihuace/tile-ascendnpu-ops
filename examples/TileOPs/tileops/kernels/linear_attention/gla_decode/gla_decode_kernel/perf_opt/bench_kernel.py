#!/usr/bin/env python3
"""Standalone msprof driver for GLADecodeOp Stage 4 (5-input GLA single-token decode).

The measured op is the target TileLang kernel itself (``main``); no Stage-5
wrapper is involved.  Workload shapes / scale / dtypes are the manifest
``GLADecodeOp`` entries expanded by ``bench_gla_decode.py``; inputs are
generated exactly like ``GLADecodeWorkload.gen_inputs``.

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload serving-b1-h8 --params k_tile=32 --launches 25

``--params`` carries compile-time knobs (``k_tile``, ``dv_split``, ...); only the
keys accepted by the candidate's jitted program are forwarded, so the same
driver measures the untouched baseline and every experiment branch.
"""

import argparse
import importlib.util
import inspect
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# manifest `GLADecodeOp.workloads` (tune cases + smoke), keyed by workload label.
WORKLOADS = {
    "smoke-b2-h4-d64": dict(B=2, H=4, DK=64, DV=64, scale=0.125),
    "serving-b1-h8": dict(B=1, H=8, DK=128, DV=128, scale=0.08838834764831845),
    "serving-b1-h32": dict(B=1, H=32, DK=128, DV=128, scale=0.08838834764831845),
    "serving-b1-h64": dict(B=1, H=64, DK=128, DV=128, scale=0.08838834764831845),
    "serving-b8-h32": dict(B=8, H=32, DK=128, DV=128, scale=0.08838834764831845),
    "serving-b8-h64": dict(B=8, H=64, DK=128, DV=128, scale=0.08838834764831845),
}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_module(path):
    name = "perf_candidate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_params(text):
    params = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        params[key.strip()] = int(value)
    return params


def gen_inputs(w, dtype, device):
    B, H, DK, DV = w["B"], w["H"], w["DK"], w["DV"]
    q = (torch.randn(B, H, DK, device=device, dtype=dtype) * 0.5).contiguous()
    k = (torch.randn(B, H, DK, device=device, dtype=dtype) * 0.5).contiguous()
    v = (torch.randn(B, H, DV, device=device, dtype=dtype) * 0.5).contiguous()
    gk = (-torch.rand(B, H, DK, device=device, dtype=dtype)).contiguous()
    state = torch.randn(B, H, DK, DV, device=device, dtype=dtype).contiguous()
    return q, k, v, gk, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--params", default="")
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    w = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]
    mod = load_module(args.module)
    factory = getattr(mod, "_gla_decode_kernel")
    program = factory(w["B"], w["H"], w["DK"], w["DV"], args.dtype, w["scale"])

    knobs = parse_params(args.params)
    accepted = set(inspect.signature(program).parameters)
    forwarded = {key: value for key, value in knobs.items() if key in accepted}
    dropped = sorted(set(knobs) - set(forwarded))
    if dropped:
        print(f"[driver] ignored knob(s) not in program signature: {dropped}", file=sys.stderr)
    prog = program(**forwarded)

    device = "npu"
    inputs = gen_inputs(w, dtype, device)
    print(
        f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
        f"dtype={args.dtype} B={w['B']} H={w['H']} DK={w['DK']} DV={w['DV']} "
        f"params={forwarded} launches={args.launches}",
        file=sys.stderr,
    )
    with torch.no_grad():
        for _ in range(args.launches):
            prog(*inputs)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
