#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Stage 4 experiment batch runner for the dropout kernel.

For each experiment row (candidate file + workload + optional block_size):
  1. run the L0 precision regression of the candidate (subprocess),
  2. run ``msprof op`` on the candidate for that workload (subprocess),
  3. parse ``OpBasicInfo.csv`` / ``PipeUtilization.csv`` / ``Memory*.csv``,
  4. append one structured row to ``perf_records.jsonl`` (append-only).

Usage:
    python run_experiments.py --list experiments.json [--only ID ...]
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(HERE, "dropout_entry.py")
RECORDS = os.path.join(HERE, "perf_records.jsonl")


def _run(cmd, log_path, timeout=2400):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           cwd=HERE, timeout=timeout)
    return p.returncode


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _last_csv(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    return files[-1] if files else None


def _num(d, key, default=0.0):
    try:
        return float(d.get(key, default) or default)
    except ValueError:
        return default


def collect_metrics(out_dir):
    """Aggregate the msprof output dir into a metrics dict."""
    info = sorted(glob.glob(os.path.join(out_dir, "**", "OpBasicInfo_*.csv"),
                            recursive=True))
    if not info:
        return None
    durs = []
    captured = None
    for f in info:
        rows = _read_csv(f)
        if not rows:
            continue
        captured = rows[0].get("Op Name")
        durs.append(_num(rows[0], "Task Duration(us)"))
    if not durs:
        return None
    m = {
        "task_duration_us": round(sum(durs) / len(durs), 4),
        "launch_samples": len(durs),
        "captured_op_name": captured,
        "task_duration_min_us": round(min(durs), 4),
        "task_duration_max_us": round(max(durs), 4),
    }
    pipe = _last_csv(os.path.join(out_dir, "**", "PipeUtilization_*.csv"))
    if pipe:
        rows = _read_csv(pipe)
        if rows:
            r = rows[0]
            m["aiv_vec_ratio"] = round(_num(r, "aiv_vec_ratio"), 4)
            m["aiv_mte2_ratio"] = round(_num(r, "aiv_mte2_ratio"), 4)
            m["aiv_mte3_ratio"] = round(_num(r, "aiv_mte3_ratio"), 4)
            m["aiv_scalar_ratio"] = round(_num(r, "aiv_scalar_ratio"), 4)
            m["aiv_vec_time_us"] = round(_num(r, "aiv_vec_time(us)"), 4)
            m["aiv_mte2_time_us"] = round(_num(r, "aiv_mte2_time(us)"), 4)
            m["aiv_mte3_time_us"] = round(_num(r, "aiv_mte3_time(us)"), 4)
            m["aiv_scalar_time_us"] = round(_num(r, "aiv_scalar_time(us)"), 4)
            m["aiv_mte2_active_bw"] = round(_num(r, "aiv_mte2_active_bw(GB/s)"), 4)
    for pat, tag in (("Memory_*.csv", "mem"), ("MemoryUB_*.csv", "ub")):
        f = _last_csv(os.path.join(out_dir, "**", pat))
        if not f:
            continue
        rows = _read_csv(f)
        if not rows:
            continue
        r = rows[0]
        if "ub_read_bw(GB/s)" in r:
            m[f"{tag}_read_bw"] = round(_num(r, "ub_read_bw(GB/s)"), 4)
            m[f"{tag}_write_bw"] = round(_num(r, "ub_write_bw(GB/s)"), 4)
        if "aic_ub_read_bw(GB/s)" in r:
            m["ub_read_bw"] = round(_num(r, "aic_ub_read_bw(GB/s)"), 4)
            m["ub_write_bw"] = round(_num(r, "aic_ub_write_bw(GB/s)"), 4)
    return m


def profile_row(exp, tag):
    """Run L0 + msprof for one experiment row; return the records dict."""
    cand = os.path.join(HERE, exp["candidate"])
    names = exp.get("name", os.path.basename(exp["candidate"]))
    wid = exp["workload_id"]
    bs = exp.get("bs")
    log_dir = os.path.join(HERE, "logs", exp.get("round", "r0"))
    os.makedirs(log_dir, exist_ok=True)

    l0_ok = None
    l0_reason = None
    if exp.get("run_l0", True):
        rc = _run([sys.executable, cand, "--level", exp.get("l0_level", "L0")],
                  os.path.join(log_dir, f"L0_{names}_{wid}.log"))
        l0_ok = rc == 0
        if not l0_ok:
            l0_reason = f"L0 exit={rc}"
            print(f"[{names}/{wid}] L0 FAIL ({l0_reason})", flush=True)
            return None

    out_root = os.path.join(HERE, "profiles", exp.get("phase", "round0"),
                            wid, f"{names}_bs{bs}")
    os.makedirs(out_root, exist_ok=True)
    os.chmod(out_root, 0o700)
    cmd = ["msprof", "op", "--kernel-name=main", f"--output={out_root}",
           f"--launch-count={exp.get('launch_count', 20)}",
           f"--warm-up={exp.get('warmup', 5)}", "--dump=off",
           f"--aic-metrics={exp.get('aic_metrics', 'Default')}",
           sys.executable, ENTRY, "--kernel", cand,
           "--N", str(exp["N"]), "--dtype", exp["dtype"],
           "--p", str(exp.get("p", 0.5)), "--seed", str(exp.get("seed", 42)),
           "--warmup", str(exp.get("warmup", 5)),
           "--runs", str(exp.get("launch_count", 20))]
    if bs:
        cmd += ["--bs", str(bs)]
    rc = _run(cmd, os.path.join(log_dir, f"prof_{names}_{wid}.log"))
    if rc != 0:
        print(f"[{names}/{wid}] msprof FAIL rc={rc}", flush=True)
        return None
    opprof = sorted(glob.glob(os.path.join(out_root, "OPPROF_*")))
    if not opprof:
        print(f"[{names}/{wid}] no OPPROF dir", flush=True)
        return None
    m = collect_metrics(opprof[-1])
    if not m:
        print(f"[{names}/{wid}] profile invalid (no OpBasicInfo)", flush=True)
        return None
    rec = {
        "kernel_id": "dropout_main",
        "workload_id": wid,
        "candidate_id": names,
        "parent_id": exp.get("parent", "baseline_stage3"),
        "block_size": bs if bs else exp.get("bs_default"),
        "phase": exp.get("phase", "round"),
        "round": exp.get("round", "r0"),
        "target_kernel_name": "main",
        "profile_status": "valid",
        "l0_result": "pass" if l0_ok else "not_run",
        "raw_profile_dir": os.path.relpath(opprof[-1], HERE),
        "command": " ".join(cmd),
        "primary_metric": "msprof_task_duration",
        **m,
    }
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"[{names}/{wid}] bs={rec['block_size']} dur={m['task_duration_us']}us "
          f"vec={m.get('aiv_vec_ratio')} mte2={m.get('aiv_mte2_ratio')} "
          f"mte3={m.get('aiv_mte3_ratio')} L0={'pass' if l0_ok else 'skip'}",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    with open(os.path.join(HERE, args.list)) as fh:
        exps = json.load(fh)
    results = []
    for e in exps:
        if args.only and e.get("name") not in args.only:
            continue
        r = profile_row(e, e.get("name"))
        results.append(r)
    print("\n=== summary ===")
    for r in results:
        if r is None:
            print("  FAILED row")
        else:
            print(f"  {r['candidate_id']:38s} {r['workload_id']:26s} "
                  f"bs={r['block_size']} {r['task_duration_us']}us")


if __name__ == "__main__":
    main()
