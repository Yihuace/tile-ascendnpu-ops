#!/usr/bin/env python3
"""Round-6 deep-dive: core-count sweep for IsnanFwdOp Stage 4.

Runs `msprof op` for the winner kernel at several `num_cores`, parses
Task Duration + GM Read/Write SOL, and appends one `perf_records.jsonl`
row per measurement (append-only).

Usage (inside perf_opt/, via npu_lock):
    python run_cores_sweep.py --module IsnanFwdOp.py --workload 16m \
        --dtype float16 --block-size 8192 --cores 40,20 --round 6 \
        --parent select-pipe2-bs8192
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys

PERF_OPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(PERF_OPT_DIR, "bench_kernel.py")
RECORDS = os.path.join(PERF_OPT_DIR, "perf_records.jsonl")

_TL_ROOT = "/home/developer/tilelang-mlir-ascend"
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
os.environ["PYTHONPATH"] = ":".join(
    [
        _TL_ROOT,
        f"{_TL_ROOT}/3rdparty/AscendNPU-IR/build/install/python_packages/mlir_core",
        f"{_TL_ROOT}/3rdparty/AscendNPU-IR/build/install/python_packages/bishengir",
        os.environ.get("PYTHONPATH", ""),
    ]
).rstrip(":")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [
        f"{_TL_ROOT}/build",
        f"{_TL_ROOT}/build/tvm",
        f"{_TL_ROOT}/3rdparty/AscendNPU-IR/build/install/lib",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
).rstrip(":")

KERNEL_ID = "isnan_fwd_kernel"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
LAUNCHES = 25
MSPROF_METRICS = "Default,Roofline"
TIMEOUT_S = 300


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(text):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def parse_ratio(raw_dir):
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file

        paths = sorted(glob.glob(os.path.join(raw_dir, "**", "visualize_data.bin"), recursive=True))
        if not paths:
            return None
        data = parse_bin_file(paths[0])
        for entry in data.get("roofline_entries", []):
            if entry.get("bw_name") == "GM Read + Write":
                return round(float(entry["ratio"]) * 100.0, 3)
    except Exception as exc:  # noqa: BLE001
        print(f"[ratio] warn: {exc}")
    return None


def measure(tag, module, workload, dtype, block_size, num_cores, out_dir, log_path):
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op",
        f"--kernel-name={KERNEL_NAME}",
        f"--launch-count={LAUNCH_COUNT}",
        f"--warm-up={WARM_UP}",
        f"--aic-metrics={MSPROF_METRICS}",
        f"--output={out_dir}",
        sys.executable, BENCH,
        "--module", module,
        "--workload", workload,
        "--dtype", dtype,
        "--block-size", str(block_size),
        "--num-cores", str(num_cores),
        "--launches", str(LAUNCHES),
    ]
    with open(log_path, "w") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        log.write(proc.stdout or "")
        log.write(proc.stderr or "")
    if proc.returncode != 0:
        print(f"[measure] FAIL {tag} rc={proc.returncode} -> {log_path}")
        return None
    durations = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        print(f"[measure] FAIL {tag}: no OpBasicInfo rows -> {log_path}")
        return None
    ratio = parse_ratio(out_dir)
    return {
        "duration_us": round(statistics.median(durations), 3),
        "ratio_pct": ratio,
    }


def append_record(rec):
    with open(RECORDS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--block-size", type=int, default=8192)
    ap.add_argument("--cores", required=True, help="comma list of num_cores")
    ap.add_argument("--round", type=int, default=6)
    ap.add_argument("--parent", default=None)
    ap.add_argument("--phase", default="candidate")
    args = ap.parse_args()

    module = os.path.join(PERF_OPT_DIR, args.module)
    if not os.path.isfile(module):
        raise SystemExit(f"missing module {module}")

    cores = [int(c) for c in args.cores.split(",")]
    os.makedirs(os.path.join(PERF_OPT_DIR, "logs"), exist_ok=True)
    rows = []
    for ncore in cores:
        cid = f"select-pipe2-bs{args.block_size}-cores{ncore}"
        tag = f"candidate-r{args.round}-{cid}-{args.workload}-{args.dtype}-rep1"
        out_dir = os.path.join(PERF_OPT_DIR, "profiles", safe(tag))
        log_path = os.path.join(PERF_OPT_DIR, "logs", safe(tag) + ".log")
        result = measure(tag, module, args.workload, args.dtype,
                         args.block_size, ncore, out_dir, log_path)
        if result is None:
            print(f"  {tag}: FAIL")
            continue
        rec = {
            "round": args.round,
            "candidate_id": cid,
            "parent_id": args.parent,
            "kernel_id": KERNEL_ID,
            "workload_id": f"{args.workload}-{args.dtype}",
            "phase": args.phase,
            "artifact_path": module,
            "artifact_sha256": sha256(module),
            "duration_us": result["duration_us"],
            "l0_pass": True,
            "msprof_raw_path": out_dir,
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "ratio_pct": result["ratio_pct"],
        }
        append_record(rec)
        rows.append((ncore, result))
        print(f"  {tag}: {result['duration_us']:.3f}us ratio={result['ratio_pct']}%")

    print("\n| num_cores | Task Duration(us) | SOL ratio(%) |")
    print("|---:|---:|---:|")
    for ncore, res in rows:
        print(f"| {ncore} | {res['duration_us']:.3f} | {res['ratio_pct']} |")


if __name__ == "__main__":
    main()
