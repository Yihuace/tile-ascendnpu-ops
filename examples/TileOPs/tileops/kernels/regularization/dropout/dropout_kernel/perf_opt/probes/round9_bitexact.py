#!/usr/bin/env python3
"""R9 bit-exactness harness: candidate output must equal the pre-R9 output bit for bit.

For every case the *same* input tensor is fed to the base module and to each
candidate module; ``torch.equal`` must hold on the raw output.  For the all-ones
input the 0/1 mask is recovered as ``y / scale`` and compared exactly as well,
so both "hash value" and "mask" are shown unchanged.
"""
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, x, dtype_str, p, seed, bs):
    fn = mod._make_dropout_kernel(x.numel(), dtype_str, float(p), int(seed))(bs)
    return fn(x.npu()).cpu()


def main():
    base_path = sys.argv[1]
    cand_paths = sys.argv[2:]
    base = load(base_path, "r9_base")
    cands = [(p, load(p, f"r9_c{i}")) for i, p in enumerate(cand_paths)]

    cases = [
        ("W1-fp16-4M", 4_194_304, "float16", 0.5, 42, 8192),
        ("W2-bf16-10M", 10_485_760, "bfloat16", 0.5, 42, 8192),
        ("W3-fp32-4M", 4_194_304, "float32", 0.5, 42, 8192),
        ("fp16-p0.75-native-nonsigned", 1_048_576, "float16", 0.75, 7, 8192),
        ("fp16-tail-4.0M", 4_000_000, "float16", 0.5, 42, 8192),
    ]
    all_ok = True
    for tag, N, dt, p, seed, bs in cases:
        tdt = base._TORCH_DTYPE[dt]
        x_ones = torch.ones(N, dtype=tdt)
        x_rand = torch.randn(N, dtype=tdt)
        for kind, x in (("ones", x_ones), ("randn", x_rand)):
            yb = run(base, x, dt, p, seed, bs)
            scale = 1.0 / (1.0 - p)
            mb = torch.round(yb.to(torch.float32) / scale)
            for path, mod in cands:
                yc = run(mod, x, dt, p, seed, bs)
                out_eq = bool(torch.equal(yb, yc))
                mc = torch.round(yc.to(torch.float32) / scale)
                mask_eq = bool(torch.equal(mb, mc))
                rate = float((mb == 0).to(torch.float32).mean())
                ok = out_eq and mask_eq
                all_ok &= ok
                print(f"[{'PASS' if ok else 'FAIL'}] {tag:30s} {kind:5s} "
                      f"cand={path.split('/')[-1]:22s} out_equal={out_eq} "
                      f"mask_equal={mask_eq} ones_drop_rate={rate:.6f}", flush=True)
    print("ALL_BIT_EXACT=", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
