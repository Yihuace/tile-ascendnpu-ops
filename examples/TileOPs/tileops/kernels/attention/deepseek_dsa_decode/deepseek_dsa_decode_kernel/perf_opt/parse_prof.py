#!/usr/bin/env python
"""Parse msprof `op` output for `main_mix_aic` and print a compact summary.

Usage: python parse_prof.py <profile_dir> [more dirs...]
Prints: task duration (us) per launch, mean/min, and selected pipe metrics.
"""
import csv
import glob
import os
import sys

KERNEL = "main_mix_aic"

# metric columns we care about (PipeUtilization csv)
INTEREST = [
    "aic_time(us)", "aic_total_cycles", "aic_cube_time(us)", "aic_cube_ratio",
    "aic_mte1_time(us)", "aic_mte2_time(us)", "aic_fix_time(us)",
    "aic_scalar_time(us)", "aic_scalar_wait_id1_time(us)",
    "aic_scalar_wait_id2_time(us)", "aiv_time(us)", "aiv_vec_time(us)",
    "aiv_vec_ratio", "aiv_scalar_time(us)", "aiv_mte2_time(us)",
    "aiv_mte3_time(us)", "aiv_scalar_wait_id2_time(us)",
]


def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def summarize(d):
    op_files = sorted(glob.glob(os.path.join(d, "**", KERNEL, "*", "OpBasicInfo_*.csv"), recursive=True))
    if not op_files:
        print(f"  !! no OpBasicInfo under {d}")
        return None
    durs = []
    for f in op_files:
        rows, fields = read_csv(f)
        for row in rows:
            key = [k for k in fields if k and "Task Duration" in k]
            if key:
                try:
                    durs.append(float(row[key[0]]))
                except (TypeError, ValueError):
                    pass
    pipe_files = sorted(glob.glob(os.path.join(d, "**", KERNEL, "*", "PipeUtilization_*.csv"), recursive=True))
    agg = {}
    if pipe_files:
        rows, fields = read_csv(pipe_files[0])
        cube = [r for r in rows if r.get("op_type") == "cube0" or r.get("Task Type") == "cube0"]
        for name in INTEREST:
            col = [k for k in fields if k and k.startswith(name.rstrip("(us)"))]
            if not col:
                continue
            vals = []
            for r in cube:
                try:
                    vals.append(float(r[col[0]]))
                except (TypeError, ValueError):
                    pass
            if vals:
                agg[name] = sum(vals) / len(vals)
    if durs:
        print(f"  {os.path.basename(d)}: n={len(durs)} mean={sum(durs)/len(durs):.2f}us "
              f"min={min(durs):.2f} max={max(durs):.2f}")
    for k, v in agg.items():
        print(f"     {k:34s} {v:.6f}")
    return sum(durs) / len(durs) if durs else None


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d)
        summarize(d)
