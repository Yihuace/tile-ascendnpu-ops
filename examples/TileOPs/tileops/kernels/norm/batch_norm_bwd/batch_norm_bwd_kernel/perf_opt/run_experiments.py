#!/usr/bin/env python3
"""BatchNormBwdOp Stage-4 experiment runner (adapted from FALN T-4).

Each experiment branch: L0 precision gate -> standalone `msprof op` measurement
via ``bench_kernel.py`` -> one ``perf_records.jsonl`` row (append-only) with the
canonical signal-registry §5 fields (+ ratio_pct / ai / computility diagnostics).

The measured op is the target kernel itself (no Stage-5 wrapper). The split
kernel is measured as the SUM of its three stages (stats + finalize + apply),
each via its own msprof run.

Usage (inside perf_opt/):
    python run_experiments.py --experiments r1.json --workloads resnet-stage1 --phase candidate
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

_TL_ROOT = "/home/developer/tilelang-mlir-ascend"
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")
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

RECORDS = os.path.join(PERF_OPT_DIR, "perf_records.jsonl")
INVENTORY = os.path.join(PERF_OPT_DIR, "workload_inventory.json")
LAUNCH_COUNT = 20
WARM_UP = 5
LAUNCHES = 25
TIMEOUT_S = 600

# kernel-name per kernel type (prefix match; actual device name is <name>_mix_aic)
SPLIT_STAGES = ["stats", "finalize", "apply"]

# kernel_id -> short bench_kernel type
KERNEL_TYPE = {
    "_batch_norm_bwd_wide_kernel": "wide",
    "_batch_norm_bwd_split_kernel": "split",
    "_batch_norm_bwd_kernel": "general",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(text):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def run_l0(module):
    proc = subprocess.run(
        [sys.executable, module, "--level", "L0"],
        capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    tail = (proc.stdout or "") + (proc.stderr or "")
    log_path = os.path.join(PERF_OPT_DIR, "logs", "l0_" + safe(os.path.basename(module)) + ".log")
    with open(log_path, "w") as handle:
        handle.write(tail)
    ok = proc.returncode == 0 and ("[L0] ALL PASS" in tail or "All check passed!" in tail)
    if not ok:
        print(f"[L0] FAIL {module} rc={proc.returncode} (log: {log_path})")
        print("\n".join(tail.splitlines()[-15:]))
    return ok


def parse_durations(out_dir):
    """Return list of Task Duration(us) from OpBasicInfo CSVs."""
    vals = []
    for path in sorted(glob.glob(os.path.join(out_dir, "**", "OpBasicInfo*.csv"), recursive=True)):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    try:
                        vals.append(float(row["Task Duration(us)"]))
                    except ValueError:
                        pass
    return vals


def parse_ratio(out_dir):
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file
        paths = sorted(glob.glob(os.path.join(out_dir, "**", "visualize_data.bin"), recursive=True))
        if not paths:
            return None, None, None
        data = parse_bin_file(paths[0])
        for entry in data.get("roofline_entries", []):
            if entry.get("bw_name") == "GM Read + Write":
                return (
                    round(float(entry["ratio"]) * 100.0, 3),
                    round(float(entry.get("AI", 0.0)), 4),
                    entry.get("computility_name"),
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[ratio] warn: {exc}")
    return None, None, None


def msprof_run(kernel_name, module, kernel, workload, dtype, out_dir, log_path):
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op",
        f"--kernel-name={kernel_name}",
        f"--launch-count={LAUNCH_COUNT}",
        f"--warm-up={WARM_UP}",
        "--aic-metrics=Roofline",
        "--dump=off",
        f"--output={out_dir}",
        sys.executable, BENCH,
        "--module", module,
        "--kernel", kernel,
        "--workload", workload,
        "--dtype", dtype,
        "--launches", str(LAUNCHES),
    ]
    with open(log_path, "w") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        log.write(proc.stdout or "")
        log.write(proc.stderr or "")
    if proc.returncode != 0:
        print(f"[measure] FAIL {kernel_name} rc={proc.returncode} -> {log_path}")
        return None
    durations = parse_durations(out_dir)
    if not durations:
        print(f"[measure] FAIL {kernel_name}: no OpBasicInfo rows -> {log_path}")
        return None
    return durations


def measure(tag, module, kernel, workload, dtype, out_dir, log_path):
    if kernel in ("wide", "general"):
        durations = msprof_run("main", module, kernel, workload, dtype, out_dir, log_path)
        if durations is None:
            return None
        total = statistics.median(durations)
    else:  # split: sum of 3 stages
        stage_totals = []
        for stage in SPLIT_STAGES:
            sdir = os.path.join(out_dir, stage)
            slog = log_path.replace(".log", f"_{stage}.log")
            durs = msprof_run(stage, module, kernel, workload, dtype, sdir, slog)
            if durs is None:
                return None
            stage_totals.append(statistics.median(durs))
        total = sum(stage_totals)
    ratio, ai, computility = parse_ratio(out_dir)
    return {"duration_us": round(total, 3), "ratio_pct": ratio, "ai": ai, "computility_name": computility}


def append_record(rec):
    with open(RECORDS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_inventory():
    with open(INVENTORY, encoding="utf-8") as handle:
        inv = json.load(handle)
    by_kernel = {}
    for item in inv["workloads"]:
        by_kernel.setdefault(item["kernel_id"], []).append(item)
    return by_kernel


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True, help="JSON list of experiment variants")
    ap.add_argument("--phase", default="candidate", choices=["baseline", "candidate", "merged", "final"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--only-kernel", default="", help="restrict to kernel_id prefix")
    args = ap.parse_args()

    with open(os.path.join(PERF_OPT_DIR, args.experiments), encoding="utf-8") as handle:
        experiments = json.load(handle)

    by_kernel = load_inventory()

    os.makedirs(os.path.join(PERF_OPT_DIR, "logs"), exist_ok=True)
    l0_cache = {}
    modules = {}
    for exp in experiments:
        module = os.path.join(PERF_OPT_DIR, exp["file"])
        if not os.path.isfile(module):
            raise SystemExit(f"missing experiment file: {module}")
        modules[exp["opt_id"]] = module
        if not args.skip_l0:
            l0_cache[module] = run_l0(module)
            print(f"[runner] {exp['opt_id']}: L0={'PASS' if l0_cache[module] else 'FAIL'}")
        else:
            l0_cache[module] = True

    rows = []
    for exp in experiments:
        module = modules[exp["opt_id"]]
        kernel_id = exp["kernel"]
        ktype = KERNEL_TYPE[kernel_id]
        if args.only_kernel and not kernel_id.startswith(args.only_kernel):
            continue
        if not l0_cache[module]:
            print(f"[runner] skip {exp['opt_id']} (L0 FAIL)")
            continue
        workloads = [w for w in by_kernel.get(kernel_id, []) if w.get("kind") == "tune"]
        parent_id = exp.get("parent_id")
        for item in workloads:
            workload_id = item["workload_id"]
            label = item["label"]
            dtype = item["dtype"]
            tag = f"{args.phase}-r{args.round}-{exp['opt_id']}-{workload_id}"
            out_dir = os.path.join(PERF_OPT_DIR, "profiles", safe(tag))
            log_path = os.path.join(PERF_OPT_DIR, "logs", safe(tag) + ".log")
            result = measure(tag, module, ktype, label, dtype, out_dir, log_path)
            if result is None:
                print(f"[runner] {tag}: FAIL")
                continue
            rec = {
                "round": args.round,
                "candidate_id": exp["opt_id"],
                "parent_id": parent_id,
                "kernel_id": kernel_id,
                "workload_id": workload_id,
                "phase": args.phase,
                "artifact_path": module,
                "artifact_sha256": sha256(module),
                "duration_us": result["duration_us"],
                "l0_pass": l0_cache[module],
                "msprof_raw_path": out_dir,
                "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "ratio_pct": result["ratio_pct"],
            }
            append_record(rec)
            rows.append((exp["opt_id"], workload_id, result))
            print(f"  {tag}: {result['duration_us']:.3f}us ratio={result['ratio_pct']}% ai={result['ai']}")

    print("\n| opt_id | workload | Task Duration(us) | SOL ratio(%) |")
    print("|---|---:|---:|---:|")
    for opt_id, wid, res in rows:
        print(f"| {opt_id} | {wid} | {res['duration_us']:.3f} | {res['ratio_pct']} |")


if __name__ == "__main__":
    main()
