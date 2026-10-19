"""Per-kernel msprof aggregation for GDN (multi-kernel op).

Each TileLang stage has a unique kernel name.  Profile one forward per name,
sum all Task Duration(us) rows, then compute an op-level roofline ratio from
GatedDeltaNetPrefillFwdOp.eval_roofline().

This is the per-kernel fallback for msprof versions that cannot aggregate a
multi-kernel op in a single capture.

Run:
  source /home/developer/tilelang-mlir-ascend/env_cann90.sh
  TILELANG_ASCEND_MODE=Developer python3 bench_gdn_perkernel.py 4096
"""

import os
import sys

import torch
import torch_npu  # noqa: F401

from tileops.benchmark.msprof import bench_kernel_msprof, _parse_op_basic_info
from tileops.ops.linear_attention.gdn_prefill import GatedDeltaNetPrefillFwdOp

BATCHED_NAMES = [
    "k1_cumsum_main",
    "gram_main",
    "build_n_main",
    "seg_cube_main",
    "seg_add_main",
    "bmm_main",
    "vec_rhs_main",
    "vec_p_main",
    "vec_vp_main",
    "vec_o_main",
    "vec_state_main",
]
PERHEAD_NAMES = [
    "k1_cumsum_main",
    "gram_main",
    "build_n_main",
    "seg_cube_main",
    "seg_add_main",
    "mm_single_main",
    "vec_rhs_single_main",
    "vec_p_single_main",
    "vec_vp_single_main",
    "vec_o_single_main",
    "vec_state_single_main",
]


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    B, HV, V, dt = 1, H, K, torch.float16
    q = torch.randn(B, S, H, K, dtype=dt, device="npu")
    k = torch.nn.functional.normalize(torch.randn(B, S, H, K, dtype=dt, device="npu"), dim=-1)
    v = torch.randn(B, S, HV, V, dtype=dt, device="npu")
    g = -torch.rand(B, S, HV, device="npu") * 0.2
    beta = torch.rand(B, S, HV, device="npu") * 0.1
    state = torch.randn(B, HV, K, V, device="npu") * 0.1
    op = GatedDeltaNetPrefillFwdOp(B=B, S=S, H=H, HV=HV, K=K, V=V, dtype=dt)

    perhead = os.environ.get("GDN_PERHEAD") == "1"
    names = PERHEAD_NAMES if perhead else BATCHED_NAMES
    launch_count = int(os.environ.get("GDN_MSPROF_LAUNCH", "5000" if perhead else "500"))
    total_us = 0.0
    for name in names:
        try:
            _, out = bench_kernel_msprof(
                op, (q, k, v, g, beta, state),
                kernel_name=name, launch_count=launch_count, warm_up=0, keep_output=False,
            )
            durs, _, _ = _parse_op_basic_info(out)
            s = sum(durs)
            total_us += s
            print(f"{name}: rows={len(durs)} sum_us={s:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAIL {str(exc)[:200]}", flush=True)

    flops, nbytes = op.eval_roofline()
    perf_tops = flops / (total_us * 1e-6) / 1e12 if total_us else 0.0
    ai = flops / nbytes if nbytes else 0.0
    # 910B3: Cube fp16 peak = 4096 MAC/cycle * 40 AIC * 1.8 GHz;
    # GM bandwidth from msprof roofline (1.75 TB/s).
    peak_tops = min(ai * 1.75, 4096 * 40 * 1800 * 1e6 / 1e12)
    ratio = 100.0 * perf_tops / peak_tops if peak_tops else 0.0
    print(
        f"TOTAL_US={total_us:.3f} FLOPS={flops} BYTES={nbytes} "
        f"AI={ai:.3f} PERF_TOPS={perf_tops:.6f} PEAK_TOPS={peak_tops:.3f} "
        f"RATIO_PCT={ratio:.4f}"
    )


if __name__ == "__main__":
    main()
