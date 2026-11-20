#!/usr/bin/env python3
"""Aggregate msprof op CSV metrics for one profile directory.

Prints the median over launches of the key PipeUtilization / ArithmeticUtilization
fields per (block_id, sub_block_id) group, plus the Task Duration distribution.
Read-only diagnostic helper (no NPU access).
"""

import argparse
import csv
import glob
import os
import statistics

PIPE_FIELDS = [
    ("aiv_time(us)", "aiv_time"),
    ("aiv_vec_ratio", "vec"),
    ("aiv_mte2_ratio", "mte2"),
    ("aiv_mte2_active_bw(GB/s)", "mte2_bw"),
    ("aiv_mte3_ratio", "mte3"),
    ("aiv_mte3_active_bw(GB/s)", "mte3_bw"),
    ("aiv_scalar_ratio", "scalar"),
    ("aiv_scalar_mte2_stall_time(us)", "sc_mte2_stall"),
    ("aiv_scalar_vector_stall_time(us)", "sc_vec_stall"),
    ("aiv_scalar_wait_time(us)", "sc_wait"),
    ("aiv_scalar_stall_by_ub_time(us)", "sc_ub_stall"),
    ("aiv_vec_time(us)", "vec_time"),
    ("aiv_mte2_time(us)", "mte2_time"),
    ("aiv_mte3_time(us)", "mte3_time"),
]


def read_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as handle:
            rows.extend(list(csv.DictReader(handle)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    args = ap.parse_args()
    root = args.profile_dir

    basic = read_rows(os.path.join(root, "**", "OpBasicInfo*.csv"))
    durs = sorted(float(r["Task Duration(us)"]) for r in basic if r.get("Task Duration(us)"))
    if durs:
        print(f"Task Duration(us): median={statistics.median(durs):.3f} "
              f"min={durs[0]:.3f} max={durs[-1]:.3f} n={len(durs)}")
        first = basic[0]
        print(f"Block Dim={first.get('Block Dim')} Mix Block Dim={first.get('Mix Block Dim')}")

    rows = read_rows(os.path.join(root, "**", "PipeUtilization*.csv"))
    groups = {}
    for row in rows:
        key = (row["block_id"], row["sub_block_id"])
        groups.setdefault(key, []).append(row)
    for key, items in sorted(groups.items()):
        if not items[0].get("aiv_time(us)") or items[0]["aiv_time(us)"] == "NA":
            continue
        parts = []
        for field, short in PIPE_FIELDS:
            vals = [float(r[field]) for r in items if r.get(field) not in (None, "NA", "")]
            if vals:
                parts.append(f"{short}={statistics.median(vals):.3f}")
        print(f"block={key[0]} {key[1]}: " + " ".join(parts))

    arows = read_rows(os.path.join(root, "**", "ArithmeticUtilization*.csv"))
    agroups = {}
    for row in arows:
        agroups.setdefault(row["sub_block_id"], []).append(row)
    for sub, items in sorted(agroups.items()):
        if items[0].get("aiv_vec_fops") in (None, "NA"):
            continue
        parts = []
        for field in ("aiv_vec_fp32_ratio", "aiv_vec_fp16_ratio", "aiv_vec_misc_ratio",
                      "aiv_vec_fops", "aiv_vec_fp32_instructions", "aiv_vec_fops_ratio"):
            vals = [float(r[field]) for r in items if r.get(field) not in (None, "NA", "")]
            if vals:
                parts.append(f"{field}={statistics.median(vals):.4f}")
        print(f"arith {sub}: " + " ".join(parts))


if __name__ == "__main__":
    main()
