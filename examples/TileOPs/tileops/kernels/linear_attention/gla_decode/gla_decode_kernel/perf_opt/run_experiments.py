#!/usr/bin/env python3
"""GLADecodeOp Stage-4 experiment batch runner (T-4 adapted).

Each experiment branch: L0 precision regression (``validate_final.py``, the
same 9 cases as ``tests/ops/test_gla_decode.py``) -> standalone ``msprof op``
measurement via ``bench_kernel.py`` -> one ``perf_records.jsonl`` row
(append-only) with the canonical signal-registry §5 fields (+ ``ratio_pct`` /
``ai`` / ``computility_name`` / ``sol_pct`` diagnostics).

The measured op is the target TileLang kernel itself (``main``), no Stage-5
wrapper.

Usage (inside perf_opt/):
    python run_experiments.py --experiments r1.json --workloads serving-b1-h8-bfloat16 \\
        --reps 2
    python run_experiments.py --experiments final.json --all-tune --phase final \\
        --order workload --reps 3
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys

PERF_OPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILEOPS_ROOT = os.path.abspath(os.path.join(PERF_OPT_DIR, "../../../../../.."))
BENCH = os.path.join(PERF_OPT_DIR, "bench_kernel.py")
VALIDATE = os.path.join(PERF_OPT_DIR, "validate_final.py")
OP_DIR = os.path.dirname(PERF_OPT_DIR)

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
KERNEL_ID = "gla_decode_kernel"
KERNEL_NAME = "_gla_decode_transit_main"
LAUNCH_COUNT = 20
WARM_UP = 5
LAUNCHES = 25
TIMEOUT_S = 900
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
    tag = safe(os.path.basename(module).replace(".py", ""))
    log_path = os.path.join(PERF_OPT_DIR, "logs", "l0_" + tag + ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [sys.executable, "-u", VALIDATE, "--module", module, "--level", "L0"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            cwd=OP_DIR,
        )
        log.write(proc.stdout or "")
        log.write(proc.stderr or "")
    tail = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and "[L0] ALL PASS" in tail
    if not ok:
        print(f"[L0] FAIL {module} rc={proc.returncode} (log: {log_path})")
        print("\n".join([ln for ln in tail.splitlines() if "FAIL" in ln or "Error" in ln][-15:]))
    return ok


def parse_ratio(raw_dir):
    """GM Read + Write SOL ratio / AI / advice from visualize_data.bin."""
    out = {
        "ratio_pct": None,
        "ai": None,
        "computility_name": None,
        "sol_pct": None,
        "advice": None,
        "block_dim": None,
        "freq_mhz": None,
    }
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file

        paths = sorted(glob.glob(os.path.join(raw_dir, "**", "visualize_data.bin"), recursive=True))
        if not paths:
            return out
        data = parse_bin_file(paths[0])
        base = data.get("base_info") or {}
        out["advice"] = data.get("advice") or None
        out["block_dim"] = base.get("block_dim")
        out["freq_mhz"] = round(float(base["freq"]) / 1e6, 1) if base.get("freq") else None
        ratios = []
        for entry in data.get("roofline_entries", []):
            ratio = entry.get("ratio")
            if not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)):
                continue
            ratios.append(float(ratio))
            if entry.get("bw_name") == "GM Read + Write":
                out["ratio_pct"] = round(float(ratio) * 100.0, 3)
                out["ai"] = round(float(entry.get("AI") or 0.0), 4)
                out["computility_name"] = entry.get("computility_name")
        # SOL% = best roofline ratio (memory or compute, whichever binds).
        if ratios:
            out["sol_pct"] = round(max(ratios) * 100.0, 3)
    except Exception as exc:  # noqa: BLE001
        print(f"[ratio] warn: {exc}")
    return out


def measure(tag, module, workload_label, dtype, params, out_dir, log_path, launches, kernel_name):
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op",
        f"--kernel-name={kernel_name}",
        f"--launch-count={LAUNCH_COUNT}",
        f"--warm-up={WARM_UP}",
        f"--aic-metrics={MSPROF_METRICS}",
        f"--output={out_dir}",
        sys.executable, BENCH,
        "--module", module,
        "--workload", workload_label,
        "--dtype", dtype,
        "--params", ",".join(f"{k}={v}" for k, v in params.items()),
        "--launches", str(launches),
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
    result = {"duration_us": round(statistics.median(durations), 3), "n": len(durations)}
    result.update(parse_ratio(out_dir))
    return result


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
    ap.add_argument("--workloads", default="", help="comma list of workload ids")
    ap.add_argument("--dtypes", default="bfloat16,float16,float32")
    ap.add_argument("--all-tune", action="store_true")
    ap.add_argument("--phase", default="candidate", choices=["baseline", "candidate", "merged", "final"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--order", default="variant", choices=["variant", "workload"])
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--launches", type=int, default=25)
    ap.add_argument("--allow-multi-final", action="store_true",
                    help="write every final-phase rep as phase=final (gate 4 wants exactly one)")
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
            if wid in wanted and tune[wid]["dtype"] in dtypes
        ]
    if not workload_ids:
        raise SystemExit("no tune workloads selected")
    print(f"[runner] phase={args.phase} round={args.round} reps={args.reps} "
          f"workloads={len(workload_ids)} experiments={len(experiments)} launches={args.launches}")

    os.makedirs(os.path.join(PERF_OPT_DIR, "logs"), exist_ok=True)
    l0_cache = {}
    rows = []
    written_final = set()

    modules = {}
    for exp in experiments:
        module = os.path.join(PERF_OPT_DIR, exp["file"])
        if not os.path.isfile(module):
            raise SystemExit(f"missing experiment file: {module}")
        modules[exp["opt_id"]] = module
        if args.skip_l0:
            l0_cache[module] = True
        elif module not in l0_cache:
            l0_cache[module] = run_l0(module)
        print(f"[runner] {exp['opt_id']}: L0={'PASS' if l0_cache[module] else 'FAIL'} "
              f"module={exp['file']} params={params_of(exp)}")

    def measure_one(rep, exp, workload_id):
        module = modules[exp["opt_id"]]
        if not l0_cache[module]:
            return
        params = params_of(exp)
        opt_id = exp["opt_id"]
        parent_id = exp.get("parent_id")
        phase = exp.get("phase", args.phase)
        if phase == "final" and not args.allow_multi_final:
            if (workload_id, opt_id) in written_final:
                phase = "candidate"
        item = tune[workload_id]
        label, dtype = item["label"], item["dtype"]
        tag = f"{phase}-r{args.round}-{opt_id}-{workload_id}-rep{rep}"
        out_dir = os.path.join(PERF_OPT_DIR, "profiles", safe(tag))
        log_path = os.path.join(PERF_OPT_DIR, "logs", safe(tag) + ".log")
        result = measure(tag, module, label, dtype, params, out_dir, log_path, args.launches,
                         exp.get("kernel_name", KERNEL_NAME))
        if result is None:
            return
        if phase == "final":
            written_final.add((workload_id, opt_id))
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
            "l0_pass": True,
            "msprof_raw_path": out_dir,
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "ratio_pct": result["ratio_pct"],
            "ai": result["ai"],
            "computility_name": result["computility_name"],
            "sol_pct": result["sol_pct"],
            "advice": result["advice"],
            "block_dim": result["block_dim"],
            "freq_mhz": result["freq_mhz"],
            "params": params,
        }
        append_record(rec)
        rows.append((opt_id, params, workload_id, rep, result))
        print(f"  {tag}: {result['duration_us']:.3f}us "
              f"GM={result['ratio_pct']}% AI={result['ai']} SOL={result['sol_pct']}%")

    for rep in range(1, args.reps + 1):
        if args.order == "workload":
            for workload_id in workload_ids:
                for exp in experiments:
                    measure_one(rep, exp, workload_id)
        else:
            for exp in experiments:
                for workload_id in workload_ids:
                    measure_one(rep, exp, workload_id)

    print("\n| opt_id | params | workload | rep | Task Duration(us) | GM SOL% | AI | SOL% |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for opt_id, params, wid, rep, res in rows:
        print(f"| {opt_id} | {params} | {wid} | {rep} | {res['duration_us']:.3f} | "
              f"{res['ratio_pct']} | {res['ai']} | {res['sol_pct']} |")


def params_of(exp):
    params = dict(exp.get("params") or {})
    if "block_size" in exp:
        params.setdefault("k_tile", exp["block_size"])
    return params


if __name__ == "__main__":
    main()
