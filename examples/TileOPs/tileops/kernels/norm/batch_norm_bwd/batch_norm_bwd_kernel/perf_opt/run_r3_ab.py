#!/usr/bin/env python3
"""Same-session paired measurement: R2 best (round-3 start) vs R3 final candidate.

Run-to-run device state shifts ~7% on the small wide shapes (PL-1.22 /
TRAP-BENCH-CONFIG-CALIBRATION), which is the same order as the expected gain.
This driver therefore measures BOTH candidates for every tune workload inside
one NPU session, one after the other, so the reported ratio is state-clean.
Absolute durations are appended to perf_records.jsonl as the `final` phase for
the final candidate (W2S3 merged module).

Usage: python run_r3_ab.py [--reps 1]
"""

import argparse
import datetime
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiments as rx  # noqa: E402  (sets up the toolchain env on import)

BEST = os.path.join(rx.PERF_OPT_DIR, "_batch_norm_bwd_wide_kernel.py")   # r3 W2 wide
BEST_SPLIT = os.path.join(rx.PERF_OPT_DIR, "_batch_norm_bwd_split_kernel.py")  # r3 S3 split
FINAL = os.path.join(rx.PERF_OPT_DIR, "BatchNormBwdOp.py")

# R2 current best (round-3 start) as shipped before this round.
R2_WIDE = os.path.join(rx.PERF_OPT_DIR, "_batch_norm_bwd_wide_kernel_round1_op1.py")
R2_SPLIT = os.path.join(rx.PERF_OPT_DIR, "_batch_norm_bwd_split_kernel_round1_op1.py")

CANDIDATES = [
    ("r2best", {"_batch_norm_bwd_wide_kernel": R2_WIDE,
                "_batch_norm_bwd_split_kernel": R2_SPLIT}),
    ("r3final", {"_batch_norm_bwd_wide_kernel": FINAL,
                 "_batch_norm_bwd_split_kernel": FINAL}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--phase-final", default="final")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    by_kernel = rx.load_inventory()
    plan = []
    for kernel_id, items in by_kernel.items():
        if kernel_id not in rx.KERNEL_TYPE:
            continue
        for item in items:
            if item.get("kind") == "tune":
                plan.append((kernel_id, item))

    results = {}
    for rep in range(args.reps):
        for kernel_id, item in plan:
            ktype = rx.KERNEL_TYPE[kernel_id]
            for label, modules in CANDIDATES:
                module = modules[kernel_id]
                tag = f"ab-r{rep}-{label}-{item['workload_id']}"
                out_dir = os.path.join(rx.PERF_OPT_DIR, "profiles", rx.safe(tag))
                log_path = os.path.join(rx.PERF_OPT_DIR, "logs", "r3", rx.safe(tag) + ".log")
                res = rx.measure(tag, module, ktype, item["label"], item["dtype"],
                                 out_dir, log_path)
                if res is None:
                    print(f"[ab] FAIL {tag}")
                    continue
                print(f"  {tag}: {res['duration_us']:.3f}us ratio={res['ratio_pct']}% "
                      f"ai={res['ai']}")
                results.setdefault((label, kernel_id, item["workload_id"]), []).append(
                    (res["duration_us"], module, out_dir, res["ratio_pct"]))
                if label == "r3final" and rep == 0:
                    rx.append_record({
                        "round": 3,
                        "candidate_id": "r3_final_w2s3",
                        "parent_id": "r3_s2_applyfold" if kernel_id.endswith("split_kernel")
                        else "r3_w2_loadsfirst",
                        "kernel_id": kernel_id,
                        "workload_id": item["workload_id"],
                        "phase": args.phase_final,
                        "artifact_path": module,
                        "artifact_sha256": rx.sha256(module),
                        "duration_us": res["duration_us"],
                        "l0_pass": True,
                        "msprof_raw_path": out_dir,
                        "timestamp": datetime.datetime.now().astimezone().isoformat(
                            timespec="seconds"),
                        "ratio_pct": res["ratio_pct"],
                    })

    print("\n| workload | r2best us | r3final us | delta% |")
    print("|---|---:|---:|---:|")
    for (label, kernel_id, wid) in list(results):
        if label != "r2best":
            continue
        a = statistics.median([r[0] for r in results[(label, kernel_id, wid)]])
        b = statistics.median([r[0] for r in results[("r3final", kernel_id, wid)]])
        print(f"| {wid} | {a:.3f} | {b:.3f} | {100.0 * (b - a) / a:+.1f} |")


if __name__ == "__main__":
    main()
