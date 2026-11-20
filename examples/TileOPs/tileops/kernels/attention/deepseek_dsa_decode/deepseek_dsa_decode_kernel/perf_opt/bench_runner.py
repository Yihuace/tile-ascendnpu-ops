#!/usr/bin/env python
"""Stage 4 workload runner for `_sparse_mla_kernel` (msprof op target).

Runs one benchmark workload of one kernel file (baseline or an experiment
branch) with warm-up + launch-count repeats.  The kernel-only timing is taken
externally by `msprof op --kernel-name=...` (the only measurement contract of
Stage 4); this script only drives the launches and, with `--check`, the L0
precision gate.

Usage:
  python bench_runner.py --file <kernel.py> --case mainstream [--repeat 20]
  python bench_runner.py --file <kernel.py> --case l0 --check
"""

import argparse
import importlib.util
import os
import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))

# Benchmark workloads (manifest `attention.yaml`, op
# DeepSeekSparseAttentionDecodeWithKVCacheFwdOp), first workload is smoke.
# fields: batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv,
#         q_start_index_s, dtype
CASES = {
    "smoke-f16": (1, 128, 128, 256, 512, 64, 256, 1, 128, "float16"),
    "smoke-bf16": (1, 128, 128, 256, 512, 64, 256, 1, 128, "bfloat16"),
    "mainstream": (1, 128, 1024, 2048, 512, 64, 2048, 1, 1024, "float16"),
    "longer-kv": (1, 128, 512, 4096, 512, 64, 1024, 1, 512, "float16"),
    "long-kv": (1, 256, 128, 8192, 512, 64, 512, 1, 256, "float16"),
}

_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def load_module(path):
    name = "dsa_kernel_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--block-heads", type=int, default=None)
    ap.add_argument("--block-n", type=int, default=None)
    ap.add_argument("--num-stages", type=int, default=None)
    ap.add_argument("--gform", type=int, default=None)
    args = ap.parse_args()

    mod = load_module(os.path.abspath(args.file))
    batch, heads, sq, skv, dim, dt, topk, stride_kv, qstart, dt_s = CASES[args.case]
    dtype = _TORCH_DTYPE[dt_s]

    kw = {}
    if args.block_heads is not None:
        kw["block_heads"] = args.block_heads
    if args.block_n is not None:
        kw["block_N"] = args.block_n
    if args.num_stages is not None:
        kw["num_stages"] = args.num_stages
    if args.gform is not None:
        kw["gform"] = args.gform

    q, kv, indices = mod._gen_inputs(batch, heads, sq, skv, dim, dt, topk, stride_kv,
                                     qstart, dtype, seed=0)
    kernel = mod._sparse_mla_kernel(batch, sq, skv, heads, dim, dt, topk, stride_kv,
                                    qstart, kv_group=1, sm_scale=None, is_causal=True,
                                    cp0=False, dtype=dt_s, **kw)
    if os.environ.get("FLAT_KV"):
        kv = kv.reshape(-1, kv.shape[-1])
    for _ in range(args.warmup):
        out = kernel(q, kv, indices)
    torch.npu.synchronize()

    for _ in range(args.repeat):
        out = kernel(q, kv, indices)
    torch.npu.synchronize()

    if args.check:
        ref = mod.golden_sparse_mla_decode(q.cpu(), kv.cpu(), indices.cpu(), dim, dt,
                                           stride_kv, qstart, sm_scale=None, dtype=dtype)
        md, strict_ok, loose_ok, finite_ok = mod._compare(out, ref)
        print(f"[check] {args.case}: max_diff={md:.3e} loose={'PASS' if loose_ok else 'FAIL'} "
              f"strict={'PASS' if strict_ok else 'note'} finite={finite_ok}")
        if not (loose_ok and finite_ok):
            raise SystemExit(2)
    print(f"[run] {args.case} file={os.path.basename(args.file)} "
          f"warmup={args.warmup} repeat={args.repeat} kw={kw}")


if __name__ == "__main__":
    main()
