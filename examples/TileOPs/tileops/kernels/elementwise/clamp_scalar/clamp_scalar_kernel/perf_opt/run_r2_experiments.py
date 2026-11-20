#!/usr/bin/env python3
"""ClampScalarFwdOp / ClampMinFwdOp Stage-4 R2 experiment batch runner.

Per branch: L0 gate -> standalone ``msprof op`` measurement via
``bench_kernel.py`` -> one ``perf_records.jsonl`` row (append-only, canonical
signal-registry §5 fields + diagnostic extras).

The measured op is the target kernel itself (``main``), not the Op wrapper.

Usage (inside perf_opt/):
    python run_r2_experiments.py --experiments r2_screen.json \
        --workloads 16M-float16,16M-bfloat16 --phase candidate --round 1
    python run_r2_experiments.py --experiments r2_final.json --all-tune \
        --phase final --round 3
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

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

RECORDS = os.path.join(PERF_OPT_DIR, "perf_records.jsonl")
INVENTORY = os.path.join(PERF_OPT_DIR, "workload_inventory.json")
KERNEL_ID = "clamp_scalar_fwd_kernel"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
LAUNCHES = 25
TIMEOUT_S = 600
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
        capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    tail = (proc.stdout or "") + (proc.stderr or "")
    log_dir = os.path.join(PERF_OPT_DIR, "logs", "r2")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, "l0_" + safe(os.path.basename(module).replace(".py", "")) + ".log"
    )
    with open(log_path, "w") as handle:
        handle.write(tail)
    ok = proc.returncode == 0 and ("All check passed!" in tail)
    if not ok:
        print(f"[L0] FAIL {module} rc={proc.returncode} (log: {log_path})")
        print("\n".join(tail.splitlines()[-15:]))
    return ok


def check_workload(module, label, dtype, block_size):
    """Per-workload golden check in the exact measured configuration."""
    cmd = [sys.executable, os.path.join(PERF_OPT_DIR, "check_workload.py"),
           "--module", module, "--workload", label, "--dtype", dtype,
           "--block-size", str(block_size)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    tail = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and "[workload-precision] PASS" in tail
    if not ok:
        print(f"[precision] FAIL {module} {label}/{dtype} bs={block_size}")
        print("\n".join(tail.splitlines()[-8:]))
    return ok


def parse_ratio(raw_dir):
    """GM Read + Write SOL ratio (%) from visualize_data.bin."""
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file

        paths = sorted(glob.glob(os.path.join(raw_dir, "**", "visualize_data.bin"),
                                recursive=True))
        if not paths:
            return None, None
        data = parse_bin_file(paths[0])
        for entry in data.get("roofline_entries", []):
            if entry.get("bw_name") == "GM Read + Write":
                return round(float(entry["ratio"]) * 100.0, 3), round(
                    float(entry.get("AI", 0.0)), 4)
    except Exception as exc:  # noqa: BLE001
        print(f"[ratio] warn: {exc}")
    return None, None


def measure(tag, module, workload_label, dtype, block_size, out_dir, log_path):
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
        "--workload", workload_label,
        "--dtype", dtype,
        "--block-size", str(block_size),
        "--launches", str(LAUNCHES),
    ]
    with open(log_path, "w") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT_S)
        log.write(proc.stdout or "")
        log.write(proc.stderr or "")
    if proc.returncode != 0:
        print(f"[measure] FAIL {tag} rc={proc.returncode} -> {log_path}")
        return None

    durations = []
    for path in sorted(glob.glob(os.path.join(
            out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        print(f"[measure] FAIL {tag}: no OpBasicInfo rows -> {log_path}")
        return None
    ratio, ai = parse_ratio(out_dir)
    return {"duration_us": round(statistics.median(durations), 3),
            "ratio_pct": ratio, "ai": ai}


def append_record(rec):
    with open(RECORDS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_tune_workloads():
    with open(INVENTORY, encoding="utf-8") as handle:
        inv = json.load(handle)
    return {item["workload_id"]: item for item in inv["workloads"]
            if item.get("kind") == "tune"}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--workloads", default="")
    ap.add_argument("--all-tune", action="store_true")
    ap.add_argument("--phase", default="candidate",
                    choices=["baseline", "candidate", "merged", "final"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--order", default="workload",
                    choices=["variant", "workload"])
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--precision-mode", default="l0", choices=["l0", "workload"],
                    help="l0: aggregated --level L0 gate; workload: per-workload "
                         "golden check in the measured configuration")
    args = ap.parse_args()

    with open(os.path.join(PERF_OPT_DIR, args.experiments), encoding="utf-8") as h:
        experiments = json.load(h)

    tune = load_tune_workloads()
    if args.all_tune:
        workload_ids = sorted(tune)
    else:
        wanted = {w.strip() for w in args.workloads.split(",") if w.strip()}
        workload_ids = [wid for wid in sorted(tune) if wid in wanted]
    if not workload_ids:
        raise SystemExit("no tune workloads selected")
    print(f"[runner] phase={args.phase} round={args.round} reps={args.reps} "
          f"workloads={len(workload_ids)} experiments={len(experiments)}")

    os.makedirs(os.path.join(PERF_OPT_DIR, "logs", "r2"), exist_ok=True)
    l0_cache, modules = {}, {}
    for exp in experiments:
        module = os.path.join(PERF_OPT_DIR, exp["file"])
        if not os.path.isfile(module):
            raise SystemExit(f"missing experiment file: {module}")
        modules[exp["opt_id"]] = module
        if args.precision_mode == "workload":
            l0_cache[module] = True
            print(f"[runner] {exp['opt_id']}: precision=per-workload "
                  f"module={exp['file']}")
            continue
        if module not in l0_cache:
            l0_cache[module] = True if args.skip_l0 else run_l0(module)
        print(f"[runner] {exp['opt_id']}: L0="
              f"{'PASS' if l0_cache[module] else 'FAIL'} module={exp['file']}")

    rows = []

    def measure_one(rep, exp, workload_id):
        module = modules[exp["opt_id"]]
        opt_id = exp["opt_id"]
        item = tune[workload_id]
        label, dtype = item["label"], item["dtype"]
        # an explicit experiment-level block_size overrides the workload default
        block_size = exp.get("block_size") or item.get("block_size", 8192)
        phase = exp.get("phase", args.phase)
        tag = f"{phase}-r{args.round}-{opt_id}-{workload_id}-rep{rep}"
        out_dir = os.path.join(PERF_OPT_DIR, "profiles", "r2", safe(tag))
        log_path = os.path.join(PERF_OPT_DIR, "logs", "r2", safe(tag) + ".log")
        if args.precision_mode == "workload":
            l0_ok = check_workload(module, label, dtype, block_size)
            if not l0_ok:
                return
        else:
            l0_ok = l0_cache[module]
        result = measure(tag, module, label, dtype, block_size, out_dir, log_path)
        if result is None:
            return
        append_record({
            "round": args.round,
            "candidate_id": opt_id,
            "parent_id": exp.get("parent_id"),
            "kernel_id": KERNEL_ID,
            "workload_id": workload_id,
            "phase": phase,
            "artifact_path": module,
            "artifact_sha256": sha256(module),
            "duration_us": result["duration_us"],
            "l0_pass": l0_ok,
            "msprof_raw_path": out_dir,
            "timestamp": datetime.datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "ratio_pct": result["ratio_pct"],
            "ai": result["ai"],
            "block_size": block_size,
        })
        rows.append((opt_id, workload_id, rep, result))
        print(f"  {tag}: {result['duration_us']:.3f}us "
              f"ratio={result['ratio_pct']}%")

    for rep in range(1, args.reps + 1):
        if args.order == "workload":
            # interleave variants per workload: A/B/A/B inside one batch
            for workload_id in workload_ids:
                ordered = experiments if rep % 2 else list(reversed(experiments))
                for exp in ordered:
                    if l0_cache[modules[exp["opt_id"]]]:
                        measure_one(rep, exp, workload_id)
        else:
            for exp in experiments:
                if not l0_cache[modules[exp["opt_id"]]]:
                    continue
                for workload_id in workload_ids:
                    measure_one(rep, exp, workload_id)

    print("\n| opt_id | workload | rep | Task Duration(us) | SOL ratio(%) |")
    print("|---|---|---:|---:|---:|")
    for opt_id, wid, rep, res in rows:
        print(f"| {opt_id} | {wid} | {rep} | {res['duration_us']:.3f} | "
              f"{res['ratio_pct']} |")


if __name__ == "__main__":
    main()
