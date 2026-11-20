#!/usr/bin/env python3
"""L0/L1 precision regression for a GLADecodeOp Stage-4 candidate file.

The candidate kernel module is loaded by path and injected into the Stage-3
wrapper module globals (``tileops.kernels.linear_attention.gla_decode.gla_decode``),
so the *real* wrapper path (``GLADecodeOp`` -> ``GLADecodeKernel`` -> factory)
is exercised without touching any file outside ``perf_opt/``.

Case list mirrors ``tests/ops/test_gla_decode.py`` (the regression entry):
fp16/bf16/fp32, manifest serving shapes, non-tile shapes, custom scale.

    python validate_final.py --module GLADecodeOp.py --level L0
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

HERE = os.path.dirname(os.path.abspath(__file__))
TILEOPS_ROOT = os.path.abspath(os.path.join(HERE, "../../../../../../.."))
sys.path.insert(0, TILEOPS_ROOT)

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

_F16 = torch.float16
_BF16 = torch.bfloat16
_F32 = torch.float32

# (B, H, DK, DV, dtype, scale, id) -- same list as tests/ops/test_gla_decode.py
CASES = [
    (2, 4, 64, 64, _F16, -1.0, "smoke-d64-fp16"),
    (2, 4, 64, 64, _BF16, -1.0, "smoke-d64-bf16"),
    (1, 8, 128, 128, _BF16, -1.0, "serving-b1-h8"),
    (1, 32, 128, 128, _BF16, -1.0, "serving-b1-h32"),
    (8, 32, 128, 128, _BF16, -1.0, "serving-b8-h32"),
    (1, 2, 32, 16, _F32, -1.0, "f32-untiled"),
    (2, 3, 16, 16, _F16, -1.0, "small-k16"),
    (1, 4, 128, 64, _F16, -1.0, "rect-d128x64"),
    (2, 3, 64, 64, _BF16, 0.25, "custom-scale"),
]
LEVELS = {"L0": CASES, "L1": CASES, "all": CASES}


def load(path):
    name = "perf_validate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--level", default="L0", choices=sorted(LEVELS))
    args = ap.parse_args()

    module = load(os.path.abspath(args.module))
    import tileops.kernels.linear_attention.gla_decode.gla_decode as wrapper_mod
    from tileops.ops.linear_attention.gla_decode import GLADecodeOp
    from tileops.workloads.linear_attention import GLADecodeWorkload, gla_decode_tolerance

    wrapper_mod._gla_decode_kernel = module._gla_decode_kernel
    wrapper_mod._ceil_ktile_fit = module._ceil_ktile_fit

    failures = 0
    for B, H, DK, DV, dtype, scale, case_id in LEVELS[args.level]:
        test = GLADecodeWorkload(B, H, DK, DV, dtype, scale)
        q, k, v, gk, state = test.gen_inputs()
        op = GLADecodeOp(B, H, DK, DV, scale=scale, dtype=dtype)
        o, ns = op(q, k, v, gk, state)
        ref_o, ref_ns = test.ref_program(q, k, v, gk, state)
        tol = gla_decode_tolerance(dtype)
        err_o = float((o.cpu().float() - ref_o).abs().max().item())
        err_ns = float((ns.cpu().float() - ref_ns).abs().max().item())
        try:
            torch.testing.assert_close(o.cpu().float(), ref_o, atol=tol["atol"], rtol=tol["rtol"])
            torch.testing.assert_close(ns.cpu().float(), ref_ns, atol=tol["atol"], rtol=tol["rtol"])
            print(f"[{case_id}] PASS err_o={err_o:.3e} err_ns={err_ns:.3e} tol={tol['atol']:.1e}")
        except AssertionError as exc:
            failures += 1
            print(f"[{case_id}] FAIL err_o={err_o:.3e} err_ns={err_ns:.3e} tol={tol['atol']:.1e}")
            print(str(exc).splitlines()[0])

    if failures:
        print(f"[{args.level}] {failures} FAILED module={args.module}")
        return 1
    print(f"[{args.level}] ALL PASS module={args.module}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
