"""Diagnostic probe harness for the Hardtanh kernel (NOT part of gate evidence,
no perf_records writes).

Runs an arbitrary (impl, workload, block_size, extra) matrix under `msprof op`
and prints Task Duration + AIV PipeUtilization, plus a derived effective
bandwidth.  Used to (a) calibrate the GM read/write floor for this kernel and
(b) sweep structural parameters before promoting a candidate to a real
experiment branch.

Usage:
  python probe_sweep.py --impl ../_make_hardtanh_kernel.py \
      --workload elementwise-16m-float16 --bs 2048 4096 8192 --tag base
"""

import argparse
import csv
import glob
import hashlib
import os
import re
import shlex
import statistics
import subprocess
import sys

WORKLOADS = {
    "smoke-1m-float32": (1048576, "float32"),
    "smoke-1m-float16": (1048576, "float16"),
    "smoke-1m-bfloat16": (1048576, "bfloat16"),
    "bounded-hidden-float16": (8388608, "float16"),
    "bounded-hidden-bfloat16": (8388608, "bfloat16"),
    "bounded-conv-feat-float16": (12845056, "float16"),
    "bounded-conv-feat-bfloat16": (12845056, "bfloat16"),
    "elementwise-16m-float16": (16777216, "float16"),
    "elementwise-16m-bfloat16": (16777216, "bfloat16"),
    "elementwise-16m-float32": (16777216, "float32"),
}
ELEM_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}

MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5


def safe_component(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def _rows(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p, newline="") as f:
            out.extend(list(csv.DictReader(f)))
    return out


def _f(row, key):
    v = row.get(key)
    if v is None or v.strip() in ("", "NA"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def measure(impl, workload_id, block_size, extra, out_dir, inner_script="_bench_kernel.py"):
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "posix":
        os.chmod(out_dir, 0o700)
    N, dtype = WORKLOADS[workload_id]
    inner = [sys.executable, os.path.abspath(inner_script),
             "--impl", os.path.abspath(impl), "--N", str(N), "--dtype", dtype]
    if block_size is not None:
        inner += ["--block_size", str(block_size)]
    for it in extra or []:
        inner += ["--kw", it]
    cmd = ["msprof", "op", f"--kernel-name={KERNEL_NAME}", f"--output={out_dir}",
           f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}", "--dump=off",
           f"--aic-metrics={MSPROF_METRICS}", *inner]
    os.makedirs("logs/probe", exist_ok=True)
    log = os.path.join("logs/probe", safe_component(out_dir) + ".log")
    with open(log, "w") as f:
        f.write("# " + shlex.join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"status": "msprof_fail", "rc": proc.returncode, "log": log}
    op_rows = _rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    durs = [d for r in op_rows if r.get("Op Name") == KERNEL_NAME
            if (d := _f(r, "Task Duration(us)")) is not None]
    if not durs:
        # mprof rc=0 but no dump rows (BiShengIR UB overflow often surfaces here
        # via an empty/failed run); surface the tail of the log for diagnosis.
        return {"status": "no_rows", "log": log}
    pipe = _rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv"))

    def avg(key):
        vals = [v for r in pipe if (v := _f(r, key)) is not None]
        return round(statistics.fmean(vals), 4) if vals else None
    dur = round(statistics.median(durs), 3)
    nbytes = 2 * N * ELEM_BYTES[dtype]
    return {
        "status": "ok", "duration_us": dur, "n": len(durs),
        "vec": avg("aiv_vec_ratio"), "mte2": avg("aiv_mte2_ratio"),
        "mte3": avg("aiv_mte3_ratio"), "scalar": avg("aiv_scalar_ratio"),
        "mte2_time": avg("aiv_mte2_time(us)"), "mte3_time": avg("aiv_mte3_time(us)"),
        "mte2_bw": avg("aiv_mte2_active_bw(GB/s)"),
        "mte3_bw": avg("aiv_mte3_active_bw(GB/s)"),
        "eff_tb_s": round(nbytes / dur / 1e6, 4) if dur else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--bs", type=int, nargs="*", default=[None])
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--out-root", default="profiles/probe")
    ap.add_argument("--inner-script", default="_bench_kernel.py")
    args = ap.parse_args()

    print("| tag | bs | TaskDur(us) | vec | mte2 | mte3 | scalar | "
          "mte2_time | mte3_time | mte2_bw | mte3_bw | eff_TB/s | status |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for bs in args.bs:
        tag = f"{args.tag}_bs{bs}"
        out_dir = os.path.join(args.out_root, safe_component(tag), safe_component(args.workload))
        r = measure(args.impl, args.workload, bs, args.extra, out_dir, args.inner_script)
        if r["status"] != "ok":
            print(f"| {tag} | {bs} | - | - | - | - | - | - | - | - | - | - | {r['status']} |")
            continue
        print(f"| {tag} | {bs} | {r['duration_us']} | {r['vec']} | {r['mte2']} | "
              f"{r['mte3']} | {r['scalar']} | {r['mte2_time']} | {r['mte3_time']} | "
              f"{r['mte2_bw']} | {r['mte3_bw']} | {r['eff_tb_s']} | ok |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
