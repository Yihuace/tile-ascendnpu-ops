#!/usr/bin/env python3
"""run_experiments.py -- Stage 4 experiment batch runner for BitwiseNotFwdOp.

Adapted from the skill template:
  references/run_experiments_template.py

Per experiment branch, in order:
  1. runs the L0 precision regression (branch file `--level L0`);
  2. runs msprof op kernel-only profiling via scripts/profile_kernel.py
     (which generates a launch script, runs msprof op, parses
     OpBasicInfo*.csv Task Duration(us) -- the single latency metric);
  3. appends one line per measured tune workload to perf_records.jsonl
     (append-only);
  4. prints the candidate-vs-current-best summary table.

Usage (inside perf_opt/):
  python run_experiments.py --phase baseline --all
  python run_experiments.py --round 1 --workload elementwise-256m-int32 --only <opt_id>
  python run_experiments.py --phase final --all
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys

# --------------------------------------------------------------- CONFIG
OP = "bitwise_not_fwd_kernel"
BASELINE_KERNEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", OP + ".py")
)
KERNEL_ID = (
    "examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/"
    "bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main"
)
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
TIMEOUT_S = 1200
CURRENT_BEST = f"{OP}.py"
CURRENT_BEST_ID = "baseline"
# --------------------------------------------------------------- /CONFIG

PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"
PROFILER = os.path.join(os.path.dirname(__file__), "scripts", "profile_kernel.py")

# ----------------------------------------------------------- EXPERIMENTS
EXPERIMENTS = [
    # Baseline: the read-only Stage 3 kernel (profiled by absolute path).
    {"file": BASELINE_KERNEL, "opt_id": "baseline", "parent_id": None},
]
# ------------------------------------------------------------------ /EXPERIMENTS


def safe_component(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def run_l0(branch_file: str) -> bool:
    proc = subprocess.run(
        [sys.executable, branch_file, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    return proc.returncode == 0


def _workload_n_dtype() -> dict:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)
    out = {}
    for item in inventory["workloads"]:
        shape = item["shape"]
        n = math.prod(shape) if isinstance(shape, list) else shape
        out[item["workload_id"]] = {
            "N": int(n),
            "dtype": item["dtype"],
            "kind": item["kind"],
        }
    return out


def run_profile(branch_file: str, workload_id: str, spec: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    cmd = [
        sys.executable,
        PROFILER,
        "--kernel-file",
        os.path.abspath(branch_file),
        "--N",
        str(spec["N"]),
        "--dtype",
        spec["dtype"],
        "--out",
        out_dir,
        "--launch-count",
        str(LAUNCH_COUNT),
        "--warm-up",
        str(WARM_UP),
        "--kernel-name",
        KERNEL_NAME,
    ]
    log = os.path.join(log_dir, safe_component(out_dir) + ".log")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n" + proc.stdout + proc.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith("PROFILE_RESULT "):
            res = json.loads(line[len("PROFILE_RESULT "):])
            res["msprof_raw_path"] = out_dir
            return res
    return {"error": (proc.stdout + proc.stderr)[-400:], "log": log}


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tune_workload_ids() -> list[str]:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)
    return [
        item["workload_id"]
        for item in inventory["workloads"]
        if item.get("kernel_id") == KERNEL_ID and item.get("kind") == "tune"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--only", help="run a single opt_id")
    ap.add_argument("--workload", help="one non-smoke benchmark workload ID")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--phase", choices=("baseline", "candidate", "merged", "final"),
                    default="candidate")
    ap.add_argument("--exp", action="append", default=[],
                    help="override EXPERIMENTS: file:opt_id[:parent_id]")
    args = ap.parse_args()

    global EXPERIMENTS
    if args.exp:
        EXPERIMENTS = []
        for spec in args.exp:
            parts = spec.split(":")
            assert len(parts) in (2, 3), spec
            EXPERIMENTS.append({
                "file": parts[0],
                "opt_id": parts[1],
                "parent_id": parts[2] if len(parts) == 3 else None,
            })

    if args.round is None:
        args.round = 0 if args.phase == "baseline" else 1
    if args.phase == "baseline" and args.round != 0:
        ap.error("baseline measurements must use round 0")

    available = tune_workload_ids()
    if args.phase == "candidate":
        if args.all or not args.workload:
            ap.error("candidate measurements require exactly one --workload")
        selected = [args.workload]
    else:
        if not args.all or args.workload:
            ap.error("baseline/merged/final measurements require --all and no --workload")
        selected = list(available)
    unknown = [w for w in selected if w not in available]
    if unknown or not selected:
        ap.error(f"unknown or empty tune workload selection: {unknown or selected}")

    todo = [e for e in EXPERIMENTS if not args.only or e["opt_id"] == args.only]
    if not todo:
        print("no experiments configured")
        return 2
    if args.phase in ("baseline", "merged", "final") and len(todo) != 1:
        ap.error("baseline/merged/final must use exactly one candidate per kernel")

    specs = _workload_n_dtype()
    results = []
    for exp in todo:
        branch = exp["file"]
        print(f"=== {exp['opt_id']}: {branch} ===")
        if not os.path.isfile(branch):
            results.append({**exp, "status": "missing_file"})
            continue
        if not run_l0(branch):
            results.append({**exp, "status": "l0_fail", "l0_pass": False})
            print("  L0 FAIL")
            continue
        for workload_id in selected:
            out_dir = os.path.join(
                "profiles", args.phase, f"round{args.round}",
                safe_component(exp["opt_id"]), safe_component(workload_id),
            )
            prof = run_profile(branch, workload_id, specs[workload_id], out_dir)
            if "error" in prof:
                results.append({**exp, "workload_id": workload_id,
                                "status": "msprof_fail", "l0_pass": True})
                print(f"  {workload_id}: msprof FAILED: {prof['error']}")
                continue
            append_record({
                "round": args.round,
                "candidate_id": exp["opt_id"],
                "parent_id": exp.get("parent_id"),
                "kernel_id": KERNEL_ID,
                "workload_id": workload_id,
                "phase": args.phase,
                "artifact_path": os.path.basename(branch) if args.phase == "final" else branch,
                "artifact_sha256": file_sha256(branch),
                "duration_us": prof["median_us"],
                "l0_pass": True,
                "msprof_raw_path": prof["msprof_raw_path"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
            })
            results.append({**exp, "workload_id": workload_id, "status": "ok",
                            "l0_pass": True, "duration_us": prof["median_us"],
                            "captured": prof.get("captured_op_names")})
            print(f"  {workload_id}: L0 pass, median Task Duration = "
                  f"{prof['median_us']} us ({prof.get('samples')} samples)")

    bases = {}
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if r.get("kernel_id") == KERNEL_ID and r.get("candidate_id") == CURRENT_BEST_ID:
                bases[r.get("workload_id")] = r["duration_us"]
    print("\n| branch | workload_id | l0 | Task Duration(us) | vs current best | status |")
    print("|---|---|---|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        base = bases.get(r.get("workload_id"))
        rel = f"{(base - dur) / base * 100:+.2f}%" if (dur is not None and base) else ""
        print(f"| {r['opt_id']} | {r.get('workload_id', '-')} | {r.get('l0_pass', '-')} | "
              f"{dur if dur is not None else '-'} | {rel} | {r['status']} |")
    print(f"\n(current merged best candidate: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
