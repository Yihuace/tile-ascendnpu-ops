#!/usr/bin/env python3
"""Standalone msprof driver for mha_decode_paged (split-KV) Stage 4.

The measured program is the TileLang kernel file itself: the driver replicates
``MHADecodePagedKernel.forward``'s split path (Phase A ``split_partial`` ->
workspace, Phase B ``combine`` -> Output) so each launch issues exactly the two
kernel launches the benchmark path issues.

    msprof op --kernel-name=split_partial --launch-count=20 --warm-up=5 \
        python perf_opt/bench_kernel.py --module <kernel.py> \
        --workload longer-cache --launches 25

``--params`` carries the compile-time knobs (``num_split`` / ``num_stages`` /
``block_M`` / ``block_N``); only the keys accepted by the candidate factory's
inner ``_func`` are forwarded, so the same driver measures the untouched
baseline and every experiment branch.

Input generation mirrors ``tileops.workloads.attention.MhaDecodePagedWorkload``
(identity block table, ``real_seqlen_kv`` == full physical cache length) and the
wrapper's fp32 ``glse`` / ``Output_partial`` workspace allocation.
"""

import argparse
import importlib.util
import inspect
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# manifest `MultiHeadAttentionDecodePagedWithKVCacheFwdOp.workloads`
WORKLOADS = {
    "single-token-page128": dict(B=1, H=16, D=128, S_kv=512, page=128),
    "batch2-page256": dict(B=2, H=8, D=64, S_kv=1024, page=256),
    "longer-cache": dict(B=1, H=8, D=64, S_kv=1024, page=256),
    "shorter-cache": dict(B=1, H=8, D=64, S_kv=512, page=256),
}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# wrapper ``default_config`` for the split path (K9: no ``threads``).
WRAPPER_CONFIG = dict(block_M=128, num_split=4, num_stages=2)


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


def build_inputs(w, dtype, num_split, device="npu"):
    """Mirror MhaDecodePagedWorkload.gen_inputs + wrapper workspace alloc."""
    B, H, D, S_kv, page = w["B"], w["H"], w["D"], w["S_kv"], w["page"]
    seqlen_q = 1
    num_pages = S_kv // page
    q = torch.randn(B, seqlen_q, H, D, device=device, dtype=dtype).contiguous()
    k = torch.randn(S_kv, H, D, device=device, dtype=dtype).contiguous()
    v = torch.randn(S_kv, H, D, device=device, dtype=dtype).contiguous()
    real = torch.full((B,), S_kv, dtype=torch.int32, device=device).contiguous()
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1).contiguous()
    glse = torch.zeros(B, H, num_split, seqlen_q, dtype=torch.float32, device=device)
    out_partial = torch.zeros(B, seqlen_q, H, num_split, D, dtype=torch.float32, device=device)
    acc_len = torch.zeros(B, num_split, dtype=torch.int32, device=device)
    return q, k, v, real, block_table, glse, out_partial, acc_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    ap.add_argument("--params", default="")
    ap.add_argument("--launches", type=int, default=25)
    args = ap.parse_args()

    w = WORKLOADS[args.workload]
    dtype = DTYPES[args.dtype]
    dtype_str = {v: k for k, v in DTYPES.items()}[dtype]
    mod = load_module(args.module)
    factory = getattr(mod, "_mha_decode_split_kernel")
    program = factory(w["B"], w["H"], 1, w["S_kv"], w["D"], w["page"], False, dtype_str)

    knobs = dict(WRAPPER_CONFIG)
    knobs["block_N"] = 64 if w["D"] <= 128 else 32
    knobs.update(parse_params(args.params))
    accepted = set(inspect.signature(program).parameters)
    call_kwargs = {}
    forwarded = {}
    for key, value in knobs.items():
        target = key if key in accepted else f"{key}_in"
        if target in accepted:
            call_kwargs[target] = value
            forwarded[key] = value
    dropped = sorted(set(knobs) - set(forwarded))
    if dropped:
        print(f"[driver] ignored knob(s) not in factory signature: {dropped}", file=sys.stderr)
    prog = program(**call_kwargs)

    num_split = forwarded.get("num_split", 4)
    inputs = build_inputs(w, dtype, num_split)
    print(
        f"[driver] module={os.path.abspath(args.module)} workload={args.workload} "
        f"dtype={dtype_str} B={w['B']} H={w['H']} D={w['D']} N_kv={w['S_kv']} "
        f"page={w['page']} params={forwarded} launches={args.launches}",
        file=sys.stderr,
    )
    with torch.no_grad():
        for _ in range(args.launches):
            prog(*inputs)
        torch.npu.synchronize()
    print("[driver] done", file=sys.stderr)


if __name__ == "__main__":
    main()
