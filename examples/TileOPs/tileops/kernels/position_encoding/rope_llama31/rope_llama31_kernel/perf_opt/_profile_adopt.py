"""Shared-kernel adoption profiler (RopeLlama31Op, r3 1d kernel).

Measures the *adopted* artifact (`perf_opt/_make_rope_neox_1d.py`) with
`msprof op` on this op's two tune workloads, so the final records in
`perf_records.jsonl` are backed by a raw profile produced here (no record is
copied from another directory).  Kernel-only `Task Duration(us)` is the only
metric (median over the launches dumped by `--launch-count`).

Usage:
  python perf_opt/_profile_adopt.py <artifact.py> [--only <workload_id>] \
      [--out-subdir final/round4/r3_adopt_c6098298]
"""

import argparse
import csv
import glob
import json
import os
import shlex
import statistics
import subprocess
import sys

KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
TIMEOUT_S = 1800
MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,"
    "L2Cache,ResourceConflictRatio"
)

WORKLOADS = {
    "llama31-1d-8k-d128-bf16": {"S": 8192, "d": 128, "dtype": "bfloat16"},
    "llama31-1d-2k-d128-fp16": {"S": 2048, "d": 128, "dtype": "float16"},
}


def _csv_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="") as handle:
            rows.extend(list(csv.DictReader(handle)))
    return rows


def _f(row, key):
    val = row.get(key)
    if val is None or str(val).strip() in ("", "NA"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def collect_metrics(out_dir):
    metrics = {}
    op_rows = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    durs = [
        d
        for row in op_rows
        if row.get("Op Name") == KERNEL_NAME
        if (d := _f(row, "Task Duration(us)")) is not None
    ]
    if not durs:
        names = sorted({row.get("Op Name") for row in op_rows})
        return {"error": f"no OpBasicInfo rows for kernel {KERNEL_NAME!r}; seen {names}"}
    metrics["duration_us"] = round(statistics.median(durs), 4)
    metrics["n_launch"] = len(durs)
    metrics["launch_durations_us"] = [round(d, 4) for d in durs]

    pipe = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv"))

    def avg(key):
        vals = [v for row in pipe if (v := _f(row, key)) is not None]
        return round(statistics.fmean(vals), 6) if vals else None

    for key in (
        "aiv_vec_ratio",
        "aiv_mte2_ratio",
        "aiv_mte3_ratio",
        "aiv_scalar_ratio",
        "aiv_mte2_active_bw(GB/s)",
        "aiv_mte3_active_bw(GB/s)",
    ):
        metrics[key.replace("(", "_").replace(")", "").replace("/", "_")] = avg(key)
    return metrics


def run_msprof(artifact, workload_id, spec, root, log_label):
    out_dir = os.path.join(root, workload_id)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    inner = [
        sys.executable,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "_bench_kernel.py")),
        "--impl",
        os.path.abspath(artifact),
        "--layout",
        "1d",
        "--S",
        str(spec["S"]),
        "--d",
        str(spec["d"]),
        "--dtype",
        spec["dtype"],
    ]
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
    logs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", "r3_adopt"
    )
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"msprof_{log_label}_{workload_id}.log")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("# " + shlex.join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        handle.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"workload_id": workload_id, "error": f"msprof rc={proc.returncode}",
                "log": log_path}
    metrics = collect_metrics(out_dir)
    metrics.update({"workload_id": workload_id, "msprof_raw_path": out_dir,
                    "command": shlex.join(cmd), "log": log_path})
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out-subdir", default="final/round4/r3_adopt_c6098298",
                    help="profile dir under perf_opt/profiles/")
    args = ap.parse_args()
    root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "profiles",
        *args.out_subdir.split("/"),
    )
    log_label = args.out_subdir.split("/")[-1]
    results = []
    for workload_id, spec in WORKLOADS.items():
        if args.only and args.only != workload_id:
            continue
        print(f"[msprof] {workload_id} ...", flush=True)
        result = run_msprof(args.artifact, workload_id, spec, root, log_label)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
