"""Stage-4 experiment runner for the GQA-backward kernel pipeline.

One process per (candidate, kernel, workload) measurement:

  1. launches ``_bench_kernel.py`` under ``msprof op`` (kernel-only
     ``Task Duration(us)`` -- the only latency yardstick used here),
  2. parses ``OpBasicInfo`` / ``PipeUtilization`` / ``ArithmeticUtilization`` /
     ``Memory`` CSVs from the raw profile,
  3. appends one row to the matching append-only ``perf_records*.jsonl``.

Usage:
  python run_experiments.py --impl <file.py> --kernel main \
      --workload llama-3.1-8b-short-float16 --candidate baseline \
      --round 0 --phase baseline --records perf_records.jsonl
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
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.dirname(HERE)

KERNEL_IDS = {
    "main": (
        "examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/"
        "grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main"
    ),
    "preprocess": (
        "examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/"
        "grouped_query_attention_bwd_kernel/_flashattn_bwd_preprocess_kernel.py::"
        "_flashattn_bwd_preprocess_kernel::flash_bwd_prep"
    ),
    "postprocess": (
        "examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/"
        "grouped_query_attention_bwd_kernel/_flashattn_bwd_postprocess_kernel.py::"
        "_flashattn_bwd_postprocess_kernel::flash_bwd_post"
    ),
}

# target kernel name handed to ``msprof op --kernel-name``
KERNEL_NAMES = {
    "main": "_gqa_bwd_main_mix_aic",
    "preprocess": "flash_bwd_prep",
    "postprocess": "flash_bwd_post",
}

# workload_id -> (B, S, H, H_kv, D, dtype, causal)
WORKLOADS = {
    "gqa-bwd-smoke-float16": (1, 128, 8, 2, 64, "float16", 1),
    "llama-3.1-8b-short-float16": (4, 512, 32, 8, 128, "float16", 1),
    "llama-3.1-8b-short-bfloat16": (4, 512, 32, 8, 128, "bfloat16", 1),
    "llama-3.1-8b-long-float16": (2, 2048, 32, 8, 128, "float16", 1),
    "llama-3.1-8b-long-bfloat16": (2, 2048, 32, 8, 128, "bfloat16", 1),
    "llama-3.1-70b-short-float16": (2, 512, 64, 8, 128, "float16", 1),
    "llama-3.1-70b-short-bfloat16": (2, 512, 64, 8, 128, "bfloat16", 1),
    "llama-3.1-70b-long-float16": (1, 2048, 64, 8, 128, "float16", 1),
    "llama-3.1-70b-long-bfloat16": (1, 2048, 64, 8, 128, "bfloat16", 1),
}

LAUNCH_COUNT = 20
WARM_UP = 5
MSPROF_METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline"
TIMEOUT_S = 2400


def _csv_rows(pattern):
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


def _avg(rows, key):
    vals = [v for r in rows if (v := _f(r, key)) is not None]
    return round(statistics.fmean(vals), 6) if vals else None


def collect_metrics(out_dir, kernel_name):
    op_rows = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    matched = [r for r in op_rows if (r.get("Op Name") or "").startswith(kernel_name)]
    durs = [d for r in matched if (d := _f(r, "Task Duration(us)")) is not None]
    if not durs:
        seen = sorted({r.get("Op Name") for r in op_rows})
        return {"error": f"no OpBasicInfo rows for kernel {kernel_name}", "seen_ops": seen}
    metrics = {
        "duration_us": round(statistics.median(durs), 4),
        "duration_min_us": round(min(durs), 4),
        "duration_max_us": round(max(durs), 4),
        "n_launch": len(durs),
    }
    pipe = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv"))
    for key, label in (
        ("aiv_vec_ratio", "aiv_vec_ratio"),
        ("aiv_scalar_ratio", "aiv_scalar_ratio"),
        ("aiv_mte2_ratio", "aiv_mte2_ratio"),
        ("aiv_mte3_ratio", "aiv_mte3_ratio"),
        ("aiv_vec_time(us)", "aiv_vec_time_us"),
        ("aiv_scalar_time(us)", "aiv_scalar_time_us"),
        ("aiv_mte2_time(us)", "aiv_mte2_time_us"),
        ("aiv_mte3_time(us)", "aiv_mte3_time_us"),
        ("aiv_mte2_active_bw(GB/s)", "aiv_mte2_active_bw_gbs"),
        ("aiv_mte3_active_bw(GB/s)", "aiv_mte3_active_bw_gbs"),
        ("aic_mac_ratio", "aic_mac_ratio"),
        ("aic_mte2_ratio", "aic_mte2_ratio"),
        ("aic_mte1_ratio", "aic_mte1_ratio"),
        ("aic_fixpipe_ratio", "aic_fixpipe_ratio"),
    ):
        metrics[label] = _avg(pipe, key)
    arith = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "ArithmeticUtilization_*.csv"))
    metrics["aiv_vec_fops"] = _avg(arith, "aiv_vec_fops")
    metrics["aic_mac_fp16"] = _avg(arith, "aic_mac_fp16")
    mem = _csv_rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "Memory_*.csv"))
    metrics["ub_read_bw"] = _avg(mem, "ub_read_bw(GB/s)")
    metrics["ub_write_bw"] = _avg(mem, "ub_write_bw(GB/s)")
    metrics["main_mem_bw"] = _avg(mem, "main_mem_read_bw(GB/s)")
    return metrics


def run_msprof(impl, kernel, workload_id, out_dir, launch_count=LAUNCH_COUNT):
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    B, S, H, H_kv, D, dtype, causal = WORKLOADS[workload_id]
    inner = [
        sys.executable,
        os.path.join(HERE, "_bench_kernel.py"),
        "--impl", os.path.abspath(impl),
        "--kernel", kernel,
        "--B", str(B), "--S", str(S), "--H", str(H), "--H_kv", str(H_kv),
        "--D", str(D), "--dtype", dtype, "--causal", str(causal),
        "--iters", str(launch_count + WARM_UP),
    ]
    cmd = [
        "msprof", "op",
        f"--kernel-name={KERNEL_NAMES[kernel]}",
        f"--output={out_dir}",
        f"--launch-count={launch_count}",
        f"--warm-up={WARM_UP}",
        "--dump=off",
        f"--aic-metrics={MSPROF_METRICS}",
        *inner,
    ]
    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, os.path.basename(out_dir) + ".log")
    with open(log, "w") as f:
        f.write(f"# {shlex.join(cmd)}\n")
        f.flush()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"error": f"msprof timeout after {TIMEOUT_S}s", "log": log}
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"error": f"msprof rc={proc.returncode}", "log": log}
    metrics = collect_metrics(out_dir, KERNEL_NAMES[kernel])
    if "error" in metrics:
        metrics["log"] = log
        return metrics
    metrics["msprof_raw_path"] = os.path.relpath(out_dir, HERE)
    metrics["command"] = shlex.join(cmd)
    return metrics


def _sha256(path):
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--kernel", default="main", choices=tuple(KERNEL_IDS))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--parent", default=None)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--phase", default="candidate",
                    choices=("baseline", "candidate", "merged", "final"))
    ap.add_argument("--records", default="perf_records.jsonl")
    ap.add_argument("--launch-count", type=int, default=LAUNCH_COUNT)
    ap.add_argument("--l0-pass", type=int, default=1)
    ap.add_argument("--out-subdir", default=None)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    sub = args.out_subdir or f"{args.phase}/round{args.round}"
    out_dir = os.path.join(
        HERE, "profiles", sub, args.kernel,
        f"{args.workload}_{args.candidate}",
    )
    metrics = run_msprof(args.impl, args.kernel, args.workload, out_dir,
                         args.launch_count)
    if "error" in metrics:
        print(json.dumps({"status": "error", "workload": args.workload,
                          "candidate": args.candidate, **metrics}, ensure_ascii=False))
        return 1
    if not args.no_record:
        record = {
            "round": args.round,
            "candidate_id": args.candidate,
            "parent_id": args.parent,
            "kernel_id": KERNEL_IDS[args.kernel],
            "workload_id": args.workload,
            "phase": args.phase,
            "artifact_path": (
                os.path.basename(args.impl)
                if os.path.dirname(os.path.abspath(args.impl)) == HERE
                else os.path.relpath(os.path.abspath(args.impl), HERE)
            ),
            "artifact_sha256": _sha256(args.impl),
            "duration_us": metrics["duration_us"],
            "l0_pass": bool(args.l0_pass),
            "msprof_raw_path": metrics["msprof_raw_path"],
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(os.path.join(HERE, args.records), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "workload": args.workload,
                      "candidate": args.candidate, **metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
