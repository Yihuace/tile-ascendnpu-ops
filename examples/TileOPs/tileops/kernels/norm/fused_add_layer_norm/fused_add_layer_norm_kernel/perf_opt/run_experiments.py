#!/usr/bin/env python3
"""AddFwdOp Stage-4 experiment batch runner (T-4 adapted).

Each experiment branch: L0 precision gate -> standalone `msprof op` measurement
via ``bench_kernel.py`` -> one ``perf_records.jsonl`` row (append-only) with the
canonical signal-registry §5 fields (+ ``ratio_pct`` diagnostic extra).

The measured op is the target kernel itself (``main``), no Stage-5 wrapper.

Usage (inside perf_opt/):
    python run_experiments.py --experiments r1.json --workloads linear-bias \
        --dtypes float16,bfloat16 --reps 3
    python run_experiments.py --experiments final.json --all-tune --phase final
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
TILEOPS_ROOT = os.path.abspath(os.path.join(PERF_OPT_DIR, "../../../../../.."))
BENCH = os.path.join(PERF_OPT_DIR, "bench_kernel.py")

# Toolchain environment (dev-root tilelang + CANN 9.0.0).  Non-interactive
# shells do not source ~/.bashrc, so export it here before spawning msprof.
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

RECORDS = os.path.join(PERF_OPT_DIR, "perf_records.jsonl")
INVENTORY = os.path.join(PERF_OPT_DIR, "workload_inventory.json")
KERNEL_ID = "_fused_add_layer_norm_kernel"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
LAUNCHES = 25
TIMEOUT_S = 300
MSPROF_METRICS = "Default,Roofline"

sys.path.insert(0, TILEOPS_ROOT)


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
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    tail = (proc.stdout or "") + (proc.stderr or "")
    log_dir = os.path.join(PERF_OPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, "l0_" + safe(os.path.basename(module).replace(".py", "")) + ".log"
    )
    with open(log_path, "w") as handle:
        handle.write(tail)
    ok = proc.returncode == 0 and ("[L0] ALL PASS" in tail or "All check passed!" in tail)
    if not ok:
        print(f"[L0] FAIL {module} rc={proc.returncode} (log: {log_path})")
        print("\n".join(tail.splitlines()[-15:]))
    return ok


def parse_ratio(raw_dir):
    """GM/L2 `GM Read + Write` SOL ratio (%) from visualize_data.bin."""
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file

        paths = sorted(glob.glob(os.path.join(raw_dir, "**", "visualize_data.bin"), recursive=True))
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


def measure(tag, module, workload, dtype, block_size, out_dir, log_path, block_m=0):
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
        "--block-m", str(block_m),
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
    ratio, ai, computility = parse_ratio(out_dir)
    return {
        "duration_us": round(statistics.median(durations), 3),
        "ratio_pct": ratio,
        "ai": ai,
        "computility_name": computility,
    }


def append_record(rec):
    with open(RECORDS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_tune_workloads():
    with open(INVENTORY, encoding="utf-8") as handle:
        inv = json.load(handle)
    return {
        item["workload_id"]: item
        for item in inv["workloads"]
        if item.get("kind") == "tune"
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True, help="JSON list of experiment variants")
    ap.add_argument("--workloads", default="", help="comma list of workload labels (e.g. linear-bias)")
    ap.add_argument("--dtypes", default="float16,bfloat16")
    ap.add_argument("--all-tune", action="store_true")
    ap.add_argument("--phase", default="candidate", choices=["baseline", "candidate", "merged", "final"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--order", default="variant", choices=["variant", "workload"],
                    help="measurement order: variant-major or workload-major (interleaves variants)")
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--include-missing-dtypes", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(PERF_OPT_DIR, args.experiments), encoding="utf-8") as handle:
        experiments = json.load(handle)

    tune = load_tune_workloads()
    if args.all_tune:
        workload_ids = sorted(tune)
    else:
        wanted = {w.strip() for w in args.workloads.split(",") if w.strip()}
        dtypes = {d.strip() for d in args.dtypes.split(",") if d.strip()}
        workload_ids = [
            wid for wid in sorted(tune)
            if tune[wid]["label"] in wanted
            and (tune[wid]["dtype"] in dtypes or args.include_missing_dtypes)
        ]
    if not workload_ids:
        raise SystemExit("no tune workloads selected")
    print(f"[runner] phase={args.phase} round={args.round} reps={args.reps} "
          f"workloads={len(workload_ids)} experiments={len(experiments)}")

    os.makedirs(os.path.join(PERF_OPT_DIR, "logs"), exist_ok=True)
    l0_cache = {}
    rows = []

    # L0 precision gate first (once per branch file), then interleave variants
    # within each rep so slow-machine drift hits all candidates equally.
    modules = {}
    for exp in experiments:
        module = os.path.join(PERF_OPT_DIR, exp["file"])
        if not os.path.isfile(module):
            raise SystemExit(f"missing experiment file: {module}")
        modules[exp["opt_id"]] = module
        l0_ok = l0_cache.get(module)
        if l0_ok is None and not args.skip_l0:
            l0_ok = run_l0(module)
            l0_cache[module] = l0_ok
        l0_cache[module] = bool(l0_ok) if not args.skip_l0 else True
        print(f"[runner] {exp['opt_id']}: L0={'PASS' if l0_cache[module] else 'FAIL'} "
              f"module={exp['file']} bs={exp.get('block_size', 8192)}")

    def measure_one(rep, exp, workload_id):
        module = modules[exp["opt_id"]]
        block_size = exp.get("block_size", 8192)
        opt_id = exp["opt_id"]
        parent_id = exp.get("parent_id")
        phase = exp.get("phase", args.phase)
        l0_ok = l0_cache[module]
        item = tune[workload_id]
        label, dtype = item["label"], item["dtype"]
        tag = f"{phase}-r{args.round}-{opt_id}-bs{block_size}-{workload_id}-rep{rep}"
        out_dir = os.path.join(PERF_OPT_DIR, "profiles", safe(tag))
        log_path = os.path.join(PERF_OPT_DIR, "logs", safe(tag) + ".log")
        result = measure(tag, module, label, dtype, block_size, out_dir, log_path, exp.get("block_m", 0))
        if result is None:
            return
        rec = {
            "round": args.round,
            "candidate_id": opt_id,
            "parent_id": parent_id,
            "kernel_id": KERNEL_ID,
            "workload_id": workload_id,
            "phase": phase,
            "artifact_path": module,
            "artifact_sha256": sha256(module),
            "duration_us": result["duration_us"],
            "l0_pass": l0_ok,
            "msprof_raw_path": out_dir,
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "ratio_pct": result["ratio_pct"],
        }
        append_record(rec)
        rows.append((opt_id, block_size, workload_id, rep, result))
        print(f"  {tag}: {result['duration_us']:.3f}us ratio={result['ratio_pct']}%")

    for rep in range(1, args.reps + 1):
        if args.order == "workload":
            for workload_id in workload_ids:
                for exp in experiments:
                    if not l0_cache[modules[exp["opt_id"]]]:
                        continue
                    measure_one(rep, exp, workload_id)
        else:
            for exp in experiments:
                if not l0_cache[modules[exp["opt_id"]]]:
                    continue
                for workload_id in workload_ids:
                    measure_one(rep, exp, workload_id)

    print("\n| opt_id | block_size | workload | rep | Task Duration(us) | SOL ratio(%) |")
    print("|---|---:|---|---:|---:|---:|")
    for opt_id, bs, wid, rep, res in rows:
        print(f"| {opt_id} | {bs} | {wid} | {rep} | {res['duration_us']:.3f} | {res['ratio_pct']} |")


if __name__ == "__main__":
    main()
