#!/usr/bin/env python3
"""Summarize perf_records.jsonl: per (candidate, workload) task duration / SOL."""

import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "perf_records.jsonl")


def main():
    rounds = {int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--round=")} or None
    groups = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        if rounds and rec["round"] not in rounds:
            continue
        groups[(rec["workload_id"], rec["candidate_id"], rec["round"])].append(rec)
    print("| workload | candidate | round | n | min | median | max | SOL%%(median) | phase |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for key in sorted(groups):
        wid, cid, rnd = key
        recs = groups[key]
        durs = sorted(r["duration_us"] for r in recs)
        ratios = [r["ratio_pct"] for r in recs if r.get("ratio_pct") is not None]
        print(
            f"| {wid} | {cid} | {rnd} | {len(durs)} | {durs[0]:.3f} | "
            f"{statistics.median(durs):.3f} | {durs[-1]:.3f} | "
            f"{statistics.median(ratios) if ratios else 'NA'} | {recs[0]['phase']} |"
        )


if __name__ == "__main__":
    main()
