#!/usr/bin/env python3
"""Append measured ``msprof op`` rows to ``perf_opt/perf_records.jsonl``.

append-only: existing lines are never rewritten.  One row per measured
kernel/workload pair (field contract: ``_shared/standards/signal-registry.md`` §5).

Usage:
  python3 record_perf.py --round 1 --candidate r1_tapk --parent baseline \\
      --kernel conv1d --artifact <kernel.py> --profiles profiles/r1-tapk \\
      --phase round [--l0-pass 1]

Every ``<profiles-dir>/<workload>-<dtype>/`` subdirectory that has a
``Task Duration(us)`` in its OpBasicInfo CSV yields one row; the duration is the
same filtered median that ``measure.py`` prints (op-name guarded).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "perf_records.jsonl")


def parse_rows(root, kname):
    durs = []
    for c in sorted(glob.glob(os.path.join(root, "**", "OpBasicInfo*.csv"),
                              recursive=True)):
        with open(c, newline="") as f:
            for row in csv.DictReader(f):
                if kname and kname not in (row.get("Op Name") or ""):
                    continue
                v = row.get("Task Duration(us)")
                if v and v.strip():
                    try:
                        durs.append(float(v))
                    except ValueError:
                        pass
    if not durs:
        return None
    durs.sort()
    return durs[len(durs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--parent", default=None)
    ap.add_argument("--kernel", default="conv1d")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--phase", default="round")
    ap.add_argument("--l0-pass", type=int, default=1)
    args = ap.parse_args()

    art = os.path.abspath(args.artifact)
    sha = hashlib.sha256(open(art, "rb").read()).hexdigest()
    tz = timezone(timedelta(hours=8))
    ts = datetime.now(tz).isoformat(timespec="seconds")
    n = 0
    for d in sorted(glob.glob(os.path.join(args.profiles, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        # <workload>-<dtype>  where workload may itself contain '-bias'
        for dt in ("float16", "bfloat16"):
            if name.endswith("-" + dt):
                wl = name[: -(len(dt) + 1)]
                break
        else:
            continue
        kname = "_conv1d_main" if not wl.endswith("-bias") else "_conv1d_bias_main"
        dur = parse_rows(d, kname)
        if dur is None:
            print(f"WARN no duration in {d}", file=sys.stderr)
            continue
        rec = {
            "round": args.round,
            "candidate_id": args.candidate,
            "parent_id": args.parent,
            "kernel_id": args.kernel,
            "workload_id": f"{wl}-{dt}",
            "phase": args.phase,
            "artifact_path": art,
            "artifact_sha256": sha,
            "duration_us": round(dur, 4),
            "l0_pass": bool(args.l0_pass),
            "msprof_raw_path": d,
            "timestamp": ts,
        }
        with open(RECORDS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        print(f"RECORD {rec['workload_id']} {rec['duration_us']}us "
              f"({args.candidate}/{args.phase})")
    print(f"RECORD_DONE rows={n} -> {RECORDS}")


if __name__ == "__main__":
    main()
