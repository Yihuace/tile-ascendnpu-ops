"""Round-12 minimal repro: dim=64 nqb=2 L0A address fault.

Fault: `aicore exception: The operation address of L0A exceeds the maximum
range of L0A. The write address of the mte load3d instruction is out of range`
(errcode 0x800000001000 / mte 0x600000043).  Round-11 found the correlation
`dim == 64 and kv_len % bn_eff(256) == 0`; this driver isolates ONE shape per
process (a fault aborts the whole process, so shapes cannot share a run).

Usage:
  python _gqa_sw_fwd_varlen_kernel_round12_l0a_repro.py --case t256
  python ... --case k128 --impl bn128 --variant bn128   # workaround shape

`--impl base` loads the frozen round-11 nqb2 probe (known-faulting kernel);
`--impl bn128` loads `_gqa_sw_fwd_varlen_kernel_round12_l0a_bn128.py`, whose
`GQA_R12_VARIANT` env selects a trace-time ablation:
  bn128    : dim=64 -> bn_eff clamped to GQA_R12_BN64 (default 128)
  mclamp   : pass-1 fused gemm M capped at bm (diagnostic, wrong results)
  p1tnc192 : pass-1 gemm/store N capped at 192 (diagnostic, wrong results)
  p2tnc192 : pass-2 PV gemm N capped at 192 (diagnostic, wrong results)

Exit code 0 = LAUNCH_OK, 2 = device fault.  No precision gate here (see the
bn128 file's own `run_L0` for the gated pass evidence).
"""

import argparse
import importlib.util
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
IMPLS = {
    "base": "_gqa_sw_fwd_varlen_kernel_round11_nqb2_probe.py",
    "bn128": "_gqa_sw_fwd_varlen_kernel_round12_l0a_bn128.py",
}

# name -> (batch, seqlens_q, seqlens_k, heads, heads_kv, dim)
CASES = {
    # fault set (dim=64, kv_len % 256 == 0)
    "m1": (1, [128], [256], 8, 2, 64),
    "t256": (2, [128, 128], [256, 256], 8, 2, 64),
    "t512": (1, [64], [512], 8, 2, 64),
    "c3": (2, [64, 128], [256, 512], 8, 2, 64),
    "c4": (3, [128, 256, 384], [128, 256, 384], 8, 2, 64),
    "kv256_512": (2, [128, 128], [256, 512], 8, 2, 64),
    # counterexamples (dim=64, kv_len % 256 != 0)
    "k128": (2, [128, 128], [128, 128], 8, 2, 64),
    "t384": (1, [384], [384], 8, 2, 64),
    "q128_k384": (1, [128], [384], 8, 2, 64),
    # dim=128 real-workload controls
    "d128k256": (1, [128], [256], 8, 2, 128),
    "d128k512": (2, [128, 128], [256, 512], 8, 2, 128),
}


def _load(name):
    path = os.path.join(HERE, IMPLS[name])
    spec = importlib.util.spec_from_file_location(f"r12_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--impl", default="base", choices=sorted(IMPLS))
    ap.add_argument("--variant", default=None,
                    help="GQA_R12_VARIANT for --impl bn128")
    ap.add_argument("--golden", action="store_true",
                    help="also run the module's golden gate (slow, CPU torch)")
    args = ap.parse_args()
    if args.variant:
        os.environ["GQA_R12_VARIANT"] = args.variant

    batch, sq, sk, heads, heads_kv, dim = CASES[args.case]
    mod = _load(args.impl)
    total_q, total_k = sum(sq), sum(sk)
    cu_q = torch.tensor([0] + list(torch.tensor(sq).cumsum(0).tolist()),
                        dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(sk).cumsum(0).tolist()),
                        dtype=torch.int32)
    torch.manual_seed(0)
    dt = torch.float16
    q = torch.randn((total_q, heads, dim), dtype=dt, device="npu")
    k = torch.randn((total_k, heads_kv, dim), dtype=dt, device="npu")
    v = torch.randn((total_k, heads_kv, dim), dtype=dt, device="npu")
    fn = mod._gqa_sw_fwd_varlen_kernel(batch, heads, heads_kv, total_q,
                                       total_k, dim, True, -1, -1, "float16")
    wrapped = fn(64, 64, 1)
    print(f"[repro] case={args.case} impl={args.impl} variant={args.variant} "
          f"B={batch} sq={sq} sk={sk} D={dim}", flush=True)
    try:
        out, lse = wrapped(q, k, v, cu_q.npu(), cu_k.npu(), max(sq))
        torch.npu.synchronize()
        print(f"LAUNCH_OK out={tuple(out.shape)} finite="
              f"{bool(torch.isfinite(out).all().item())} "
              f"lse_finite={bool(torch.isfinite(lse).all().item())}", flush=True)
    except Exception:
        print("LAUNCH_FAULT", flush=True)
        traceback.print_exc()
        return 2
    if args.golden:
        ok = mod._run_case(batch, sq, sk, heads, heads_kv, dim, True, -1, -1,
                           dt, tag=f"repro-{args.case}", seed=0)
        print("GOLDEN", "PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
