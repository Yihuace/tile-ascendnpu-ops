#!/usr/bin/env python3
"""Summarise one or more msprof-op profile directories for the conv1d general kernel.

Usage:
  python3 prof_summary.py <prof_root> [<prof_root> ...]

Each root is searched recursively for ``OpBasicInfo.csv`` (Task Duration) and
``PipeUtilization.csv`` (per-pipe times).  Prints one line per input root with
the duration plus the max-over-blocks pipe breakdown, which is the evidence
table used in ``opt_log.md`` (B2 structured feedback).
"""
import csv
import glob
import os
import sys

KEYS = [
    "aic_time(us)", "aic_cube_time(us)", "aic_mte1_time(us)", "aic_mte2_time(us)",
    "aic_mte3_time(us)", "aic_fixpipe_time(us)", "aic_scalar_time(us)",
    "aic_scalar_wait_id4_time(us)", "aiv_time(us)", "aiv_vec_time(us)",
    "aiv_mte2_time(us)", "aiv_mte3_time(us)", "aiv_scalar_time(us)",
    "aiv_scalar_wait_time(us)",
]


def duration(root):
    for f in glob.glob(os.path.join(root, "**", "OpBasicInfo*.csv"), recursive=True):
        for row in csv.DictReader(open(f)):
            v = row.get("Task Duration(us)")
            if v:
                try:
                    return float(v), row.get("Block Dim")
                except ValueError:
                    pass
    return None, None


def pipes(root):
    agg = {}
    for f in glob.glob(os.path.join(root, "**", "PipeUtilization*.csv"), recursive=True):
        for row in csv.DictReader(open(f)):
            for k, v in row.items():
                if k not in KEYS:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                agg[k] = max(agg.get(k, 0.0), v)
    return agg


def main():
    for root in sys.argv[1:]:
        dur, bdim = duration(root)
        ap = pipes(root)
        print(f"## {root}")
        print(f"   Task Duration(us) = {dur}  Block Dim = {bdim}")
        for k in KEYS:
            if k in ap:
                print(f"   {k:30s} {ap[k]:9.2f}")


if __name__ == "__main__":
    main()
