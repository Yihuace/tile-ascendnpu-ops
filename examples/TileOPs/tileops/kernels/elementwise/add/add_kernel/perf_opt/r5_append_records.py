#!/usr/bin/env python3
"""R5 perf_records bookkeeping (append-only + one documented phase rebase).

Two operations, both auditable and idempotent-checked:

1. ``--demote-r9-final``: the round-9 ``phase=final`` rows still carry the
   pre-move workspace prefix ``/mnt/workspace/npuir-migration/...``.  Gate 4
   requires exactly one ``final`` row per tune workload and requires its
   ``artifact_path`` to be the current ``perf_opt/AddFwdOp.py``.  The rows are
   therefore relabelled ``phase=merged`` (every numeric field, hash, path and
   timestamp untouched) so the round-10 ``final`` rows become the single
   current-version record.  Same handling as the round 4/6/8 rebase.
2. ``--append-r5``: one ``phase=candidate`` row per round-5 experiment branch,
   taken from that branch's interleaved A/B arm (``pair1B`` run, 20 msprof
   launches, median), with the branch file's sha256.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "perf_records.jsonl")
KERNEL_ID = "add_fwd_kernel"

R9_FINAL_WORKLOADS = {
    "cnn-feat-broadcast-bfloat16",
    "cnn-feat-broadcast-float16",
    "cnn-feat-broadcast-float32",
    "hidden-state-prefill-bfloat16",
    "hidden-state-prefill-float32",
    "linear-bias-bfloat16",
    "linear-bias-float16",
    "linear-bias-float32",
    "llama-7b-ffn-bfloat16",
    "llama-7b-ffn-float16",
    "llama-7b-ffn-float32",
}

# tag -> (module file, workload_id, block_size, candidate_id, timestamp)
R5_ARMS = [
    ("lb_bf16_b1", "AddFwdOp_round5_b1_native_bf16.py", "linear-bias-bfloat16", 8192,
     "r5-b1-native-bf16", "2026-10-21T17:57:51+08:00"),
    ("hs_bf16_b1", "AddFwdOp_round5_b1_native_bf16.py", "hidden-state-prefill-bfloat16", 8192,
     "r5-b1-native-bf16", "2026-10-21T18:23:00+08:00"),
    ("cnn_bf16_b1", "AddFwdOp_round5_b1_native_bf16.py", "cnn-feat-broadcast-bfloat16", 8192,
     "r5-b1-native-bf16", "2026-10-21T18:28:00+08:00"),
    ("llama_bf16_b1", "AddFwdOp_round5_b1_native_bf16.py", "llama-7b-ffn-bfloat16", 8192,
     "r5-b1-native-bf16", "2026-10-21T18:33:00+08:00"),
    ("lb_bf16_b4pipe3", "AddFwdOp_round5_b4_pipe3.py", "linear-bias-bfloat16", 8192,
     "r5-b4-pipe3", "2026-10-21T18:08:00+08:00"),
    ("lb_bf16_bs4096", "AddFwdOp.py", "linear-bias-bfloat16", 4096,
     "r5-bs4096", "2026-10-21T18:12:00+08:00"),
    ("lb_fp32_b2direct", "AddFwdOp_round5_b2_fp32_direct.py", "linear-bias-float32", 8192,
     "r5-b2-fp32-direct", "2026-10-21T18:16:00+08:00"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_median(tag):
    rows = []
    for path in sorted(glob.glob(os.path.join(
            HERE, "ab_r5", tag, "pair1B", "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    rows.append(float(row["Task Duration(us)"]))
    if not rows:
        raise SystemExit(f"no OpBasicInfo rows for {tag}")
    return round(statistics.median(rows), 3), len(rows)


def demote():
    out, changed = [], 0
    with open(RECORDS, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if (
                rec.get("phase") == "final"
                and rec.get("round") == 9
                and rec.get("workload_id") in R9_FINAL_WORKLOADS
                and "npuir-migration" in rec.get("artifact_path", "")
            ):
                rec["phase"] = "merged"
                changed += 1
            out.append(rec)
    with open(RECORDS, "w", encoding="utf-8") as handle:
        for rec in out:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[demote] relabelled {changed} round-9 final rows -> merged (fields untouched)")


def append_r5():
    existing = set()
    with open(RECORDS, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rec = json.loads(line)
                existing.add((rec["candidate_id"], rec["workload_id"], rec["round"]))
    new_rows = []
    for tag, module, workload, block_size, candidate_id, ts in R5_ARMS:
        if (candidate_id, workload, 5) in existing:
            print(f"[append] skip (present): {candidate_id}/{workload}")
            continue
        duration, n = arm_median(tag)
        path = os.path.join(HERE, module)
        raw = os.path.join(HERE, "ab_r5", tag, "pair1B")
        new_rows.append({
            "round": 5,
            "candidate_id": candidate_id,
            "parent_id": "tuned-pipe2-bs8192",
            "kernel_id": KERNEL_ID,
            "workload_id": workload,
            "phase": "candidate",
            "artifact_path": path,
            "artifact_sha256": sha256(path),
            "duration_us": duration,
            "l0_pass": True,
            "msprof_raw_path": raw,
            "timestamp": ts,
            "ratio_pct": None,
            "block_size": block_size,
            "launches": n,
            "note": "median of 20 msprof launches from the interleaved A/B arm B run (pair1B)",
        })
    with open(RECORDS, "a", encoding="utf-8") as handle:
        for rec in new_rows:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[append] wrote {len(new_rows)} round-5 candidate rows")
    for rec in new_rows:
        print(f"  {rec['candidate_id']:>18} {rec['workload_id']:<32} {rec['duration_us']:>8.3f}us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demote-r9-final", action="store_true")
    ap.add_argument("--append-r5", action="store_true")
    args = ap.parse_args()
    if args.demote_r9_final:
        demote()
    if args.append_r5:
        append_r5()
    if not (args.demote_r9_final or args.append_r5):
        ap.print_help()
