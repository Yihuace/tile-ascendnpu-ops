#!/usr/bin/env python3
"""R5 bitwise probe: native bf16/fp32 vadd vs the fp32-transit baseline.

Compares, element by element in raw bit form, the output of the R5 candidate
kernels against the frozen current best (``perf_opt/AddFwdOp.py``) on the
*exact benchmark workload tensors* (N = 4,194,304 bf16 for ``linear-bias``)
plus adversarial exponent-gap data that is the only theoretical source of a
double-rounding difference.

    python3 r5_bitwise_probe.py [--module-a AddFwdOp.py --module-b <cand.py>]

Writes a JSON summary to stdout; the caller tees it into logs/r5/.
"""

import argparse
import hashlib
import importlib.util
import json
import os

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))

# linear-bias: input [1024, 4096], other [1, 4096] -> broadcast N
N_LINEAR_BIAS = 1024 * 4096


def load_module(path):
    tag = hashlib.md5(path.encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(f"probe_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits_equal(x, y):
    """True when every element matches bit-for-bit in its own dtype."""
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    ix = x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)
    iy = y.contiguous().view(torch.int16 if y.element_size() == 2 else torch.int32)
    return bool(torch.equal(ix, iy)), int((ix != iy).sum().item())


def run_pair(mod_a, mod_b, dtype, a, b, alpha=1, block_size=8192):
    ka = mod_a.add_fwd_kernel(a.numel(), dtype, alpha=alpha, block_size=block_size)
    kb = mod_b.add_fwd_kernel(a.numel(), dtype, alpha=alpha, block_size=block_size)
    ya = ka(a, b, a.numel()).clone()
    yb = kb(a, b, a.numel()).clone()
    return ya, yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-a", default=os.path.join(HERE, "AddFwdOp.py"))
    ap.add_argument("--module-b", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    mod_a = load_module(args.module_a)
    mod_b = load_module(args.module_b)
    torch_dtype = getattr(torch, args.dtype)
    results = []

    # 1. Workload-shaped random data, several seeds (linear-bias N).
    for seed in range(4):
        torch.manual_seed(seed)
        a = torch.randn(N_LINEAR_BIAS, dtype=torch.float32).to(torch_dtype).npu()
        b = torch.randn(N_LINEAR_BIAS, dtype=torch.float32).to(torch_dtype).npu()
        ya, yb = run_pair(mod_a, mod_b, args.dtype, a, b)
        eq, n_diff = bits_equal(ya.cpu(), yb.cpu())
        results.append(
            {"case": f"linear-bias/randn/seed={seed}", "n": N_LINEAR_BIAS,
             "bitwise_equal": eq, "n_diff": n_diff}
        )
        print(f"[bitwise] linear-bias seed={seed}: equal={eq} n_diff={n_diff}", flush=True)

    # 2. Adversarial exponent gaps: the only place fp32-then-round (double
    #    rounding) could differ from a single native rounding.
    torch.manual_seed(1234)
    n_adv = 1 << 20
    cases = {}
    big = (torch.rand(n_adv, dtype=torch.float32) * 2 + 1) * 2.0 ** 100
    tiny = torch.rand(n_adv, dtype=torch.float32) * 2.0 ** -100
    cases["gap=200"] = (big, tiny)
    big2 = (torch.rand(n_adv, dtype=torch.float32) * 2 + 1) * 2.0 ** 10
    tiny2 = torch.rand(n_adv, dtype=torch.float32) * 2.0 ** -60
    cases["gap=70"] = (big2, tiny2)
    # near-exact-tie construction: a + b where the exact sum sits on a bf16
    # rounding midpoint (b = mid - a with 12 extra mantissa bits).
    base = (torch.randint(-1000, 1000, (n_adv,), dtype=torch.int32)).float()
    mid = torch.round(base * 2.0) / 2.0 + 1.0 / 256.0
    cases["near-tie"] = (base, mid - base)
    for name, (a_f, b_f) in cases.items():
        a = a_f.to(torch_dtype).npu()
        b = b_f.to(torch_dtype).npu()
        ya, yb = run_pair(mod_a, mod_b, args.dtype, a, b)
        eq, n_diff = bits_equal(ya.cpu(), yb.cpu())
        results.append({"case": f"adversarial/{name}", "n": n_adv,
                         "bitwise_equal": eq, "n_diff": n_diff})
        print(f"[bitwise] adversarial {name}: equal={eq} n_diff={n_diff}", flush=True)

    all_eq = all(r["bitwise_equal"] for r in results)
    print(json.dumps({"module_a": args.module_a, "module_b": args.module_b,
                      "dtype": args.dtype, "all_bitwise_equal": all_eq,
                      "cases": results}, indent=2))
    if not all_eq:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
