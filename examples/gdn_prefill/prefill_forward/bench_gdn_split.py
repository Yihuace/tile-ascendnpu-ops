"""Wall-clock benchmark for the GDN in-scope split path vs reference fallback.

This is NOT an msprof/SOL benchmark. It exists so the numbers in
OPTIMIZE_REPORT.md are reproducible from a checked-in file.

Run:
  source /home/developer/tilelang-mlir-ascend/env_cann90.sh
  TILELANG_ASCEND_MODE=Developer python3 bench_gdn_split.py
"""

import os
import time

import torch
import torch_npu  # noqa: F401

from tileops.ops.linear_attention.gdn_prefill import GatedDeltaNetPrefillFwdOp

CASES = [
    # S, H, K
    (64, 1, 128),
    (256, 4, 128),
    (1024, 16, 128),
    (4096, 16, 128),
]


def run_once(S, H, K, force_fallback):
    os.environ["GDN_FORCE_FALLBACK"] = "1" if force_fallback else "0"
    B, HV, V, dtype = 1, H, K, torch.float16
    scale = K ** -0.5
    q = torch.randn(B, S, H, K, dtype=dtype, device="npu")
    k = torch.nn.functional.normalize(
        torch.randn(B, S, H, K, dtype=dtype, device="npu"), dim=-1
    )
    v = torch.randn(B, S, HV, V, dtype=dtype, device="npu")
    g = -torch.rand(B, S, HV, device="npu") * 0.2
    beta = torch.rand(B, S, HV, device="npu") * 0.1
    state = torch.randn(B, HV, K, V, device="npu") * 0.1

    op = GatedDeltaNetPrefillFwdOp(
        B=B, S=S, H=H, HV=HV, K=K, V=V, dtype=dtype, scale=scale
    )
    op(q, k, v, g, beta, state)  # warmup / JIT compile
    torch.npu.synchronize()
    t0 = time.perf_counter()
    op(q, k, v, g, beta, state)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3


def main():
    for S, H, K in CASES:
        split = run_once(S, H, K, False)
        fallback = run_once(S, H, K, True)
        print(
            f"S={S} H={H} K={K} split={split:.1f}ms "
            f"fallback={fallback:.1f}ms speedup={fallback / split:.2f}x"
        )


if __name__ == "__main__":
    main()
