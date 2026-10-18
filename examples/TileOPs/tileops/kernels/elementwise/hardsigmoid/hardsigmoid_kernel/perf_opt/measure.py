#!/usr/bin/env python3
"""Compact msprof measurement for HardsigmoidFwdOp Stage 4.

Runs msprof op on one (module, workload, dtype), parses Task Duration(us) median
and GM Read+Write SOL ratio from visualize_data.bin.  Appends to perf_records.jsonl.

Usage (inside perf_opt/):
    python measure.py --module hardsigmoid_fwd_kernel.py --workload fc-wide \
        --dtype float16 --phase baseline --candidate baseline --round 0
"""

import argparse
import csv
import datetime
import glob
import json
import os
import shlex
import statistics
import subprocess
import sys
import hashlib

PERF_OPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILEOPS_ROOT = os.path.abspath(os.path.join(PERF_OPT_DIR, "../../../../../.."))
sys.path.insert(0, TILEOPS_ROOT)
BENCH = os.path.join(PERF_OPT_DIR, "bench_kernel.py")
RECORDS = os.path.join(PERF_OPT_DIR, "perf_records.jsonl")
KERNEL_ID = "hardsigmoid_fwd_kernel"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
LAUNCHES = 25
MSPROF_METRICS = "Default,Roofline"


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()


def safe(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def parse_ratio(raw_dir):
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file
        paths = sorted(glob.glob(os.path.join(raw_dir, "**", "visualize_data.bin"), recursive=True))
        if not paths:
            return None, None, None
        data = parse_bin_file(paths[0])
        for e in data.get("roofline_entries", []):
            if e.get("bw_name") == "GM Read + Write":
                return (round(float(e["ratio"]) * 100.0, 3),
                        round(float(e.get("AI", 0.0)), 4),
                        e.get("computility_name"))
    except Exception as exc:
        print(f"[ratio] warn: {exc}")
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--phase", default="candidate")
    ap.add_argument("--candidate", default="cand")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    module = os.path.join(PERF_OPT_DIR, args.module)
    tag = args.tag or f"{args.phase}-r{args.round}-{args.candidate}-{args.workload}-{args.dtype}"
    out_dir = os.path.join(PERF_OPT_DIR, "profiles", safe(tag))
    log_path = os.path.join(PERF_OPT_DIR, "logs", safe(tag) + ".log")
    os.makedirs(os.path.join(PERF_OPT_DIR, "logs"), exist_ok=True)
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)

    cmd = ["msprof", "op", f"--kernel-name={KERNEL_NAME}",
           f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}",
           f"--aic-metrics={MSPROF_METRICS}", f"--output={out_dir}",
           sys.executable, BENCH, "--module", module, "--workload", args.workload,
           "--dtype", args.dtype, "--launches", str(LAUNCHES)]
    if args.block_size:
        cmd += ["--block-size", str(args.block_size)]

    with open(log_path, "w") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        log.write(proc.stdout or "")
        log.write(proc.stderr or "")
    if proc.returncode != 0:
        print(f"[measure] FAIL rc={proc.returncode} -> {log_path}")
        sys.exit(1)

    durations = []
    for p in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        print(f"[measure] FAIL no OpBasicInfo rows -> {log_path}")
        sys.exit(1)
    ratio, ai, computility = parse_ratio(out_dir)
    dur = round(statistics.median(durations), 3)
    rec = {
        "round": args.round,
        "candidate_id": args.candidate,
        "parent_id": args.parent,
        "kernel_id": KERNEL_ID,
        "workload_id": f"{args.workload}-{args.dtype}",
        "phase": args.phase,
        "artifact_path": module,
        "artifact_sha256": sha256(module),
        "duration_us": dur,
        "l0_pass": True,
        "msprof_raw_path": out_dir,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "ratio_pct": ratio,
        "ai": ai,
        "computility_name": computility,
    }
    with open(RECORDS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[measure] {tag}: duration={dur}us ratio={ratio}% ai={ai} computility={computility}")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
