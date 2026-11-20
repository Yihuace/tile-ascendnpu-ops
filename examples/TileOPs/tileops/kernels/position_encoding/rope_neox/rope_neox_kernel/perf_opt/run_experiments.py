"""Stage 4 experiment batch runner for the shared RoPE (neox-family) kernels.

Covers BOTH shared kernel files of the family:
  * `_make_rope_neox_1d` (RopeNeoxOp / RopeLongRopeOp / RopeLlama31Op, layout=1d)
  * `_make_rope_neox_2d` (same three ops, layout=2d)

Per candidate file this runner, in order:
  1. runs the L0 precision regression (`<file> --level L0`);
  2. wraps `msprof op` (kernel-only `Task Duration(us)`, the single latency
     metric) and parses PipeUtilization / ArithmeticUtilization;
  3. appends one line per measured tune workload to perf_records.jsonl
     (append-only, contract: _shared/standards/signal-registry.md #5);
  4. prints a candidate-vs-current-best table.

Usage (inside perf_opt/):
  python run_experiments.py --kernel 1d --phase baseline --impl ../_make_rope_neox_1d.py \
         --opt-id baseline --all
  python run_experiments.py --kernel 1d --phase candidate --impl _make_rope_neox_1d_r1_balance.py \
         --opt-id r1_balance --parent-id baseline --round 1 --all
  python run_experiments.py --kernel 1d --phase candidate --impl _probe.py --opt-id p \
         --all --tile 52 --no-record
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

KERNELS = {
    "1d": {
        "kernel_id": (
            "examples/TileOPs/tileops/kernels/position_encoding/rope_neox/"
            "rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main"
        ),
        "baseline_impl": "../_make_rope_neox_1d.py",
        "workloads": {
            "neox-1d-2k-d64-fp16": {"layout": "1d", "S": 2048, "d": 64, "dtype": "float16"},
            "neox-1d-4k-d128-bf16": {"layout": "1d", "S": 4096, "d": 128, "dtype": "bfloat16"},
            "longrope-1d-8k-d128-bf16": {"layout": "1d", "S": 8192, "d": 128, "dtype": "bfloat16"},
            "longrope-1d-2k-d128-fp16": {"layout": "1d", "S": 2048, "d": 128, "dtype": "float16"},
        },
    },
    "2d": {
        "kernel_id": (
            "examples/TileOPs/tileops/kernels/position_encoding/rope_neox/"
            "rope_neox_kernel/_make_rope_neox_2d.py::_make_rope_neox_2d::main"
        ),
        "baseline_impl": "../_make_rope_neox_2d.py",
        "workloads": {
            "neox-2d-b2-s2k-h32-d128-fp16": {
                "layout": "2d", "B": 2, "S": 2048, "H": 32, "d": 128, "dtype": "float16"},
            "longrope-2d-b1-s8k-h32-d128-fp16": {
                "layout": "2d", "B": 1, "S": 8192, "H": 32, "d": 128, "dtype": "float16"},
        },
    },
}

KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
TIMEOUT_S = 2400
PERF_RECORDS = "perf_records.jsonl"
MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)


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
        capture_output=True, text=True, timeout=TIMEOUT_S,
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
    durs = [d for r in op_rows if r.get("Op Name") == KERNEL_NAME
            if (d := _f(r, "Task Duration(us)")) is not None]
    if not durs:
        names = sorted({r.get("Op Name") for r in op_rows})
        return {"error": f"no OpBasicInfo rows for kernel {KERNEL_NAME!r}; seen {names}"}
    metrics["duration_us"] = round(statistics.median(durs), 4)
    metrics["n_launch"] = len(durs)

    pipe = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv"))

    def avg(key):
        vals = [v for r in pipe if (v := _f(r, key)) is not None]
        return round(statistics.fmean(vals), 6) if vals else None

    for key in ("aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_scalar_ratio",
                "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)",
                "aiv_scalar_time(us)", "aiv_time(us)", "aiv_mte2_active_bw(GB/s)",
                "aiv_mte3_active_bw(GB/s)", "aiv_vec_total_cycles",
                "aiv_vec_total_issue_cycles", "aiv_vec_total_execute_cycles"):
        metrics[key.replace("(", "_").replace(")", "").replace("/", "_")] = avg(key)

    arith = _csv_rows(
        os.path.join(out_dir, "OPPROF_*", "*", "*", "ArithmeticUtilization_*.csv"))
    fops = [v for r in arith if (v := _f(r, "aiv_vec_fops")) is not None]
    metrics["aiv_vec_fops"] = round(statistics.fmean(fops), 2) if fops else None
    return metrics


def run_msprof(branch_file: str, workload_id: str, out_dir: str, args, spec: dict):
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "posix":
        os.chmod(out_dir, 0o700)
    inner = [sys.executable, os.path.abspath("_bench_kernel.py"),
             "--impl", os.path.abspath(branch_file),
             "--layout", spec["layout"],
             "--S", str(spec["S"]), "--d", str(spec["d"]), "--dtype", spec["dtype"]]
    if spec["layout"] == "2d":
        inner += ["--B", str(spec["B"]), "--H", str(spec["H"])]
    for flag in ("tile", "core_num", "block_size"):
        val = getattr(args, flag, None)
        if val is not None:
            inner += [f"--{flag}", str(val)]
    cmd = ["msprof", "op", f"--kernel-name={KERNEL_NAME}", f"--output={out_dir}",
           f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}", "--dump=off",
           f"--aic-metrics={MSPROF_METRICS}", *inner]
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


def tune_workload_ids(kernel_id: str) -> list:
    # Per-kernel inventory split (records-only change, see opt_log §9): the
    # 1d kernel is reconciled by gate 4 (`kernel_opt_py_path`), so its workload
    # set stays in workload_inventory.json; the 2d kernel's set lives in
    # workload_inventory_2d.json. Fall back across both to stay reproducible.
    ids = []
    seen = set()
    for name in ("workload_inventory.json", "workload_inventory_2d.json"):
        try:
            with open(name, encoding="utf-8") as handle:
                inventory = json.load(handle)
        except OSError:
            continue
        for item in inventory["workloads"]:
            if item.get("kernel_id") != kernel_id or item.get("kind") != "tune":
                continue
            if item["workload_id"] in seen:
                continue
            seen.add(item["workload_id"])
            ids.append(item["workload_id"])
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=("1d", "2d"))
    ap.add_argument("--phase", choices=("baseline", "candidate", "merged", "final"),
                    default="candidate")
    ap.add_argument("--impl", required=True, help="candidate file (relative to perf_opt/)")
    ap.add_argument("--opt-id", required=True)
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--workload", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--core_num", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--best-id", default="baseline")
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    if args.round is None:
        args.round = 0 if args.phase == "baseline" else 1

    spec_all = KERNELS[args.kernel]
    kernel_id = spec_all["kernel_id"]
    available = tune_workload_ids(kernel_id)
    if args.all:
        if args.workload:
            ap.error("--all and --workload are mutually exclusive")
        selected = list(available)
    else:
        if not args.workload:
            ap.error("candidate phase requires --workload or --all")
        selected = [args.workload]
    unknown = [w for w in selected if w not in available]
    if unknown:
        ap.error(f"unknown/non-tune workload: {unknown}")

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
        spec = spec_all["workloads"][workload_id]
        out_dir = os.path.join("profiles", args.phase, f"round{args.round}",
                               safe_component(args.opt_id), safe_component(workload_id))
        print(f"=== {args.opt_id} | {workload_id} ===")
        prof = run_msprof(args.impl, workload_id, out_dir, args, spec)
        if "error" in prof:
            print(f"  msprof FAILED: {prof['error']} (log {prof.get('log')})")
            results.append({"opt_id": args.opt_id, "workload_id": workload_id,
                            "status": "msprof_fail"})
            continue
        if not args.no_record:
            append_record({
                "round": args.round, "candidate_id": args.opt_id,
                "parent_id": args.parent_id, "kernel_id": kernel_id,
                "workload_id": workload_id, "phase": args.phase,
                "artifact_path": args.impl, "artifact_sha256": sha,
                "duration_us": prof["duration_us"], "l0_pass": l0_pass,
                "msprof_raw_path": prof["msprof_raw_path"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
            })
        print(f"  Task Duration = {prof['duration_us']} us  "
              f"vec={prof['aiv_vec_ratio']} mte2={prof['aiv_mte2_ratio']} "
              f"mte3={prof['aiv_mte3_ratio']} scalar={prof['aiv_scalar_ratio']}")
        results.append({"opt_id": args.opt_id, "workload_id": workload_id,
                        "status": "ok", **prof, "l0_pass": l0_pass})

    bases = {}
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kernel_id") == kernel_id and r.get("candidate_id") == args.best_id:
                    bases[r.get("workload_id")] = r["duration_us"]
    print("\n| candidate | workload_id | L0 | TaskDur(us) | vec | mte2 | mte3 | scalar | vs best | status |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        base = bases.get(r.get("workload_id"))
        rel = f"{(base - dur) / base * 100:+.1f}%" if (dur and base) else "-"
        print(f"| {r['opt_id']} | {r['workload_id']} | {r.get('l0_pass', '-')} | "
              f"{dur if dur is not None else '-'} | {r.get('aiv_vec_ratio', '-')} | "
              f"{r.get('aiv_mte2_ratio', '-')} | {r.get('aiv_mte3_ratio', '-')} | "
              f"{r.get('aiv_scalar_ratio', '-')} | {rel} | {r['status']} |")
    print(f"\n(current best candidate id: {args.best_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
