#!/usr/bin/env python3
"""Summarise ``perf_records.jsonl`` into the tables used by opt_log/Final Summary.

    python summarize_r2.py            # phase/candidate medians per workload
    python summarize_r2.py --final    # baseline vs final table (5-column rows)
"""

import argparse
import collections
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "perf_records.jsonl")


def load():
    rows = []
    with open(RECORDS, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    rows = load()

    if not args.final:
        groups = collections.defaultdict(list)
        for rec in rows:
            if rec["phase"] == "final":
                continue
            groups[(rec["candidate_id"], rec["workload_id"])].append(rec)
        print("| candidate | workload | n | median us | ratio% | ratio range |")
        print("|---|---|---:|---:|---:|---|")
        for (cand, wid), recs in sorted(groups.items()):
            d = [r["duration_us"] for r in recs]
            ratios = [r.get("ratio_pct") for r in recs if r.get("ratio_pct")]
            rng = f"{min(ratios):.1f}-{max(ratios):.1f}" if ratios else "-"
            med_ratio = f"{statistics.median(ratios):.2f}" if ratios else "-"
            print(f"| {cand} | {wid} | {len(d)} | {statistics.median(d):.3f} | "
                  f"{med_ratio} | {rng} |")
        return

    base, fin = {}, {}
    for rec in rows:
        key = rec["workload_id"]
        if rec["phase"] == "baseline":
            base.setdefault(key, []).append(rec)
        elif rec["phase"] == "final":
            fin.setdefault(key, []).append(rec)
    print("| kernel_id | workload_id | baseline_us | final_us | candidate_id |")
    print("|---|---|---:|---:|---|")
    for key in sorted(fin):
        b = statistics.median([r["duration_us"] for r in base.get(key, [])]) \
            if base.get(key) else float("nan")
        f = statistics.median([r["duration_us"] for r in fin[key]])
        cand = fin[key][0]["candidate_id"]
        print(f"| {fin[key][0]['kernel_id']} | {key} | {b:.3f} | {f:.3f} | {cand} |")
    print("\n| workload_id | base_ratio% | final_ratio% | delta_pct |")
    print("|---|---:|---:|---:|")
    for key in sorted(fin):
        br = statistics.median([r["ratio_pct"] for r in base.get(key, [])
                               if r.get("ratio_pct")]) if base.get(key) else float("nan")
        fr = statistics.median([r["ratio_pct"] for r in fin[key] if r.get("ratio_pct")])
        dp = (statistics.median([r["duration_us"] for r in fin[key]])
              / statistics.median([r["duration_us"] for r in base[key]]) - 1) * 100
        print(f"| {key} | {br:.2f} | {fr:.2f} | {dp:+.2f} |")


if __name__ == "__main__":
    main()
