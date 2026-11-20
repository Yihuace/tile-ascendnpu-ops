"""Stage 4 experiment batch runner for the erf kernel (T-4).

Runs, per candidate file: L0 precision regression -> msprof op kernel-only
measurement (the single latency metric) -> append one perf_records.jsonl line
per measured tune workload -> print a candidate-vs-current-best table.

Usage (inside perf_opt/):
  python run_experiments.py --phase baseline --impl erf_fwd_kernel.py --opt-id baseline --all
  python run_experiments.py --phase candidate --impl erf_fwd_kernel_round1_bs4096.py \
         --opt-id r1_bs4096 --parent-id baseline --round 1 --workload elementwise-16m-float16
  python run_experiments.py --phase final --impl erf_fwd_kernel.py --opt-id rN_merged --all
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys

OP = "erf_fwd_kernel"
KERNEL_ID = (
    "examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/"
    "erf_fwd_kernel.py::erf_fwd_kernel::main"
)
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
TIMEOUT_S = 1800
CURRENT_BEST_ID = "r0_current"
PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"
MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)

# workload_id -> (N, dtype_str)
WORKLOADS = {
    "smoke-1m-float32": (1048576, "float32"),
    "smoke-1m-float16": (1048576, "float16"),
    "smoke-1m-bfloat16": (1048576, "bfloat16"),
    "elementwise-16m-float16": (16777216, "float16"),
    "elementwise-16m-bfloat16": (16777216, "bfloat16"),
    "elementwise-16m-float32": (16777216, "float32"),
    "elementwise-64m-float16": (67108864, "float16"),
    "elementwise-64m-bfloat16": (67108864, "bfloat16"),
    "elementwise-256m-float16": (268435456, "float16"),
    "elementwise-256m-bfloat16": (268435456, "bfloat16"),
}


def safe_component(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_l0(branch_file: str) -> bool:
    proc = subprocess.run(
        [sys.executable, branch_file, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
    return proc.returncode == 0


def _csv_rows(pattern: str):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _f(row, key):
    val = row.get(key)
    if val is None or val.strip() == "" or val.strip() == "NA":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def collect_metrics(out_dir: str) -> dict:
    metrics = {}
    op_rows = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    durs = [_f(r, "Task Duration(us)") for r in op_rows if r.get("Op Name") == KERNEL_NAME]
    durs = [d for d in durs if d is not None]
    if not durs:
        return {"error": "no OpBasicInfo rows for kernel"}
    metrics["duration_us"] = round(statistics.median(durs), 4)
    metrics["n_launch"] = len(durs)

    pipe = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv"))
    def avg(key):
        vals = [v for r in pipe if (v := _f(r, key)) is not None]
        return round(statistics.fmean(vals), 6) if vals else None

    metrics["aiv_vec_ratio"] = avg("aiv_vec_ratio")
    metrics["aiv_mte2_ratio"] = avg("aiv_mte2_ratio")
    metrics["aiv_mte3_ratio"] = avg("aiv_mte3_ratio")
    metrics["aiv_scalar_ratio"] = avg("aiv_scalar_ratio")
    metrics["aiv_mte2_active_bw_gbs"] = avg("aiv_mte2_active_bw(GB/s)")
    metrics["aiv_mte3_active_bw_gbs"] = avg("aiv_mte3_active_bw(GB/s)")
    metrics["aiv_time_us"] = avg("aiv_time(us)")
    metrics["aiv_scalar_wait_time_us"] = avg("aiv_scalar_wait_time(us)")

    arith = _csv_rows(
        os.path.join(out_dir, "OPPROF_*", "*", "*", "ArithmeticUtilization_*.csv")
    )
    fops = [v for r in arith if (v := _f(r, "aiv_vec_fops")) is not None]
    metrics["aiv_vec_fops"] = round(statistics.fmean(fops), 2) if fops else None

    mem = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "Memory_*.csv"))
    if mem:
        metrics["mem_rows"] = len(mem)
    return metrics


def run_msprof(branch_file: str, workload_id: str, out_dir: str, block_size, extra):
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "posix":
        os.chmod(out_dir, 0o700)
    N, dtype = WORKLOADS[workload_id]
    inner = [
        sys.executable,
        os.path.abspath("_bench_kernel.py"),
        "--impl",
        branch_file,
        "--N",
        str(N),
        "--dtype",
        dtype,
    ]
    if block_size is not None:
        inner += ["--block_size", str(block_size)]
    cmd = [
        "msprof",
        "op",
        f"--kernel-name={KERNEL_NAME}",
        f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}",
        f"--warm-up={WARM_UP}",
        "--dump=off",
        f"--aic-metrics={MSPROF_METRICS}",
        *inner,
    ]
    log = os.path.join("logs", safe_component(out_dir) + ".log")
    os.makedirs("logs", exist_ok=True)
    with open(log, "w") as f:
        f.write(f"# {shlex.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"error": f"msprof rc={proc.returncode}", "log": log}
    metrics = collect_metrics(out_dir)
    if "error" in metrics:
        return {"error": metrics["error"], "log": log}
    metrics["msprof_raw_path"] = out_dir
    metrics["command"] = shlex.join(cmd)
    return metrics


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tune_workload_ids() -> list:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)
    return [
        item["workload_id"]
        for item in inventory["workloads"]
        if item.get("kernel_id") == KERNEL_ID and item.get("kind") == "tune"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("baseline", "candidate", "merged", "final"),
                    default="candidate")
    ap.add_argument("--impl", required=True, help="candidate file (relative to perf_opt/)")
    ap.add_argument("--opt-id", required=True)
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--workload", default=None)
    ap.add_argument("--workloads", default=None,
                    help="comma-separated workload ids (single L0 run, several msprof runs)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--skip-l0", action="store_true")
    args = ap.parse_args()

    if args.round is None:
        args.round = 0 if args.phase == "baseline" else 1

    available = tune_workload_ids()
    if args.all:
        if args.workload or args.workloads:
            ap.error("--all and --workload/--workloads are mutually exclusive")
        selected = list(available)
    elif args.workloads:
        if args.workload:
            ap.error("--workload and --workloads are mutually exclusive")
        selected = [w.strip() for w in args.workloads.split(",") if w.strip()]
    else:
        if not args.workload:
            ap.error("candidate phase requires --workload, --workloads or --all")
        selected = [args.workload]
    unknown = [w for w in selected if w not in available]
    if unknown:
        ap.error(f"unknown/ non-tune workload: {unknown}")

    if not os.path.isfile(args.impl):
        print(f"missing impl file: {args.impl}")
        return 2

    l0_pass = True
    if not args.skip_l0:
        print(f"=== L0 regression: {args.impl} ===")
        l0_pass = run_l0(args.impl)
        print(f"L0 {'PASS' if l0_pass else 'FAIL'}")
        if not l0_pass:
            print("L0 failed -> no performance measurement")
            return 1

    sha = file_sha256(args.impl)
    results = []
    for workload_id in selected:
        out_dir = os.path.join(
            "profiles", args.phase, f"round{args.round}",
            safe_component(args.opt_id), safe_component(workload_id),
        )
        print(f"=== {args.opt_id} | {workload_id} | block_size={args.block_size} ===")
        prof = run_msprof(args.impl, workload_id, out_dir, args.block_size, None)
        if "error" in prof:
            print(f"  msprof FAILED: {prof['error']} (log {prof.get('log')})")
            results.append({"opt_id": args.opt_id, "workload_id": workload_id,
                            "status": "msprof_fail"})
            continue
        append_record({
            "round": args.round,
            "candidate_id": args.opt_id,
            "parent_id": args.parent_id,
            "kernel_id": KERNEL_ID,
            "workload_id": workload_id,
            "phase": args.phase,
            "artifact_path": args.impl,
            "artifact_sha256": sha,
            "duration_us": prof["duration_us"],
            "l0_pass": l0_pass,
            "msprof_raw_path": prof["msprof_raw_path"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        })
        print(f"  Task Duration = {prof['duration_us']} us  "
              f"vec={prof['aiv_vec_ratio']} mte2={prof['aiv_mte2_ratio']} "
              f"mte3={prof['aiv_mte3_ratio']} scalar={prof['aiv_scalar_ratio']}")
        results.append({"opt_id": args.opt_id, "workload_id": workload_id,
                        "status": "ok", **prof, "l0_pass": l0_pass})

    # candidate vs current best table
    bases = {}
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kernel_id") == KERNEL_ID and r.get("candidate_id") == CURRENT_BEST_ID:
                    bases[r.get("workload_id")] = r["duration_us"]
    print("\n| candidate | workload_id | L0 | TaskDur(us) | vec | mte2 | vs best | status |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        base = bases.get(r.get("workload_id"))
        rel = f"{(base - dur) / base * 100:+.1f}%" if (dur and base) else "-"
        print(f"| {r['opt_id']} | {r['workload_id']} | {r.get('l0_pass', '-')} | "
              f"{dur if dur is not None else '-'} | {r.get('aiv_vec_ratio', '-')} | "
              f"{r.get('aiv_mte2_ratio', '-')} | {rel} | {r['status']} |")
    print(f"\n(current best candidate id: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
