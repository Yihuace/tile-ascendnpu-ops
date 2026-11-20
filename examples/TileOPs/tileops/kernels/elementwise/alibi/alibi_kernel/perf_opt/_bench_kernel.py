"""Standalone msprof op probe for the ALiBi kernel (one workload per process).

Loads the kernel factory from an arbitrary candidate file so that the exact
artifact under measurement is the one named on the command line.  Only the
kernel is launched here -- the Stage 5 wrapper / benchmark is never used.

ALiBi is *generative* (zero tensor inputs): the caller allocates
``out (H, S, S)`` with ``torch.empty`` and passes it in (DESIGN §0.6 R5).

Usage:
  python _bench_kernel.py --impl <candidate.py> --S <int> --H <int> --dtype <str>
                          [--bi <int>] [--bj <int>] [--budget <int>]
                          [--core_num <int>] [--iters <int>]
"""

import argparse
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _load_impl(path: str):
    spec = importlib.util.spec_from_file_location("alibi_candidate_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alibi_candidate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--bi", type=int, default=None)
    ap.add_argument("--bj", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--core_num", type=int, default=None)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    mod = _load_impl(args.impl)
    factory = mod._make_alibi_kernel

    torch_dtype = _TORCH_DTYPE[args.dtype]
    out = torch.empty(args.H, args.S, args.S, dtype=torch_dtype, device="npu")

    kwargs = {}
    if args.bi is not None:
        kwargs["bi"] = args.bi
    if args.bj is not None:
        kwargs["bj"] = args.bj
    if args.budget is not None:
        kwargs["budget"] = args.budget
    if args.core_num is not None:
        kwargs["core_num"] = args.core_num

    kernel = factory(args.S, args.H, args.dtype, **kwargs)

    for _ in range(args.iters):
        kernel(out)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
