"""Standalone msprof `op` probe for the GQA-backward kernel pipeline (Stage 4).

Loads the kernel factory from an arbitrary candidate file so that the exact
artifact named on the command line is the one measured.  Only the TileLang
kernel is launched -- the TileOPs wrapper / benchmark is never used.

The three pipeline stages are selected with ``--kernel``:

  * ``main``        -> ``_gqa_bwd_kernel`` (primary, gate-4 accounting object)
  * ``preprocess``  -> ``_flashattn_bwd_preprocess_kernel``
  * ``postprocess`` -> ``_flashattn_bwd_postprocess_kernel``

Usage:
  python _bench_kernel.py --impl <candidate.py> --kernel main \
      --B 4 --S 512 --H 32 --H_kv 8 --D 128 --dtype float16 --causal 1 --iters 20
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401

_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _load_impl(path):
    spec = importlib.util.spec_from_file_location("gqa_bwd_candidate_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gqa_bwd_candidate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _main_inputs(B, S, H, H_kv, D, dtype_str, seed=0):
    td = _TORCH_DTYPE[dtype_str]
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=gen).to(td)
    k = torch.randn(B, S, H_kv, D, generator=gen).to(td)
    v = torch.randn(B, S, H_kv, D, generator=gen).to(td)
    do = torch.randn(B, S, H, D, generator=gen).to(td)
    o = torch.randn(B, S, H, D, generator=gen).to(td)
    lse = (6.0 + 0.05 * torch.randn(B, H, S, generator=gen)).float()
    delta = torch.sum(o.float() * do.float(), dim=-1).transpose(1, 2).contiguous()
    return q, k, v, do, o, lse, delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--kernel", default="main",
                    choices=("main", "preprocess", "postprocess"))
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--H_kv", type=int, default=0)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--dtype", required=True, choices=("float16", "bfloat16"))
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    mod = _load_impl(args.impl)
    B, S, H, D = args.B, args.S, args.H, args.D
    H_kv = args.H_kv or H
    dtype_str = args.dtype

    if args.kernel == "main":
        factory = mod._gqa_bwd_kernel
        q, k, v, do, o, lse, delta = _main_inputs(B, S, H, H_kv, D, dtype_str)
        q, k, v, do = q.npu(), k.npu(), v.npu(), do.npu()
        lse, delta = lse.npu(), delta.npu()
        dq = torch.zeros(B, S, H, D, dtype=torch.float32).npu()
        dk = torch.zeros(B, S, H_kv, D, dtype=torch.float32).npu()
        dv = torch.zeros(B, S, H_kv, D, dtype=torch.float32).npu()
        kernel = factory(B, H, H_kv, S, D, bool(args.causal), dtype_str)
        for _ in range(args.iters):
            dq.zero_()
            kernel(q, k, v, do, lse, delta, dq, dk, dv)
    elif args.kernel == "preprocess":
        factory = mod._flashattn_bwd_preprocess_kernel
        td = _TORCH_DTYPE[dtype_str]
        gen = torch.Generator().manual_seed(0)
        o = torch.randn(B, S, H, D, generator=gen).to(td).npu()
        do = torch.randn(B, S, H, D, generator=gen).to(td).npu()
        kernel = factory(B, H, S, D, dtype_str)
        for _ in range(args.iters):
            kernel(o, do)
    else:
        factory = mod._flashattn_bwd_postprocess_kernel
        gen = torch.Generator().manual_seed(0)
        dq = torch.randn(B, S, H, D, generator=gen).npu()
        kernel = factory(B, H, S, D, dtype_str)
        for _ in range(args.iters):
            kernel(dq)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
