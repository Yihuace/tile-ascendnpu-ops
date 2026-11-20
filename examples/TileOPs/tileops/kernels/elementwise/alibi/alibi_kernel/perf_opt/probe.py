"""Lightweight msprof op probe (no L0, no perf_records) for ALiBi diagnostics.

Usage (inside perf_opt/):
  python probe.py --impl _probe_x.py --workload llama-prefill-2k-fp16 --tag x_2kfp16
                  [--bi N --bj N --budget N --core_num N]
"""

import argparse
import os
import sys

import run_experiments as R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bi", type=int, default=None)
    ap.add_argument("--bj", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--core_num", type=int, default=None)
    ap.add_argument("--S", type=int, default=None, help="override the workload's seq_len")
    ap.add_argument("--H", type=int, default=None, help="override the workload's num_heads")
    ap.add_argument("--dtype", default=None, help="override the workload's dtype")
    args = ap.parse_args()

    if args.workload not in R.WORKLOADS:
        print(f"unknown workload {args.workload}; known: {list(R.WORKLOADS)}")
        return 2
    if args.S is not None or args.H is not None or args.dtype is not None:
        S, H, dt = R.WORKLOADS[args.workload]
        R.WORKLOADS[args.workload] = (
            args.S if args.S is not None else S,
            args.H if args.H is not None else H,
            args.dtype if args.dtype is not None else dt,
        )
    if not os.path.isfile(args.impl):
        print(f"missing impl {args.impl}")
        return 2

    out_dir = os.path.join("profiles", "probe", R.safe_component(args.tag),
                           R.safe_component(args.workload))
    prof = R.run_msprof(args.impl, args.workload, out_dir, args)
    if "error" in prof:
        print(f"PROBE {args.tag} {args.workload}: FAILED {prof['error']} (log {prof.get('log')})")
        return 1
    print(
        f"PROBE {args.tag} | {args.workload} | TaskDur={prof['duration_us']}us | "
        f"vec={prof['aiv_vec_ratio']} mte2={prof['aiv_mte2_ratio']} "
        f"mte3={prof['aiv_mte3_ratio']} scalar={prof['aiv_scalar_ratio']} | "
        f"vec_t={prof['aiv_vec_time_us']} mte3_t={prof['aiv_mte3_time_us']} "
        f"scalar_t={prof['aiv_scalar_time_us']} (sgl={prof.get('aiv_scalar_single_time', '-')} "
        f"dual={prof.get('aiv_scalar_dual_time', '-')}) | "
        f"aiv_t={prof['aiv_time_us']} mte3bw={prof['aiv_mte3_active_bw_gbs']} | "
        f"{prof['msprof_raw_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
