"""Multi-kernel msprof-op measurement for the round-20 3-pass GQA backward.

The split path launches three device kernels per op call
(``_gqa_bwd_sdw`` -> ``_gqa_bwd_pds`` -> ``_gqa_bwd_mul``).  ``msprof op``
reports one ``Task Duration(us)`` row per kernel, so the aggregate latency of
one op call is the **sum of the per-kernel medians** (the same yardstick the
r11 MHA reference used; see
``examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md`` 10.4).

This runner therefore profiles without ``--kernel-name`` (capturing every
kernel of the call), groups the ``OpBasicInfo`` rows by ``Op Name`` and prints
the per-pass medians plus their sum.  It also reports PipeUtilization per pass
so the round table can carry the AICore/vector utilisation.

Usage:
  python run_split_measure.py --impl <file.py> --B 4 --S 512 --H 32 --H_kv 8 \
      --D 128 --dtype float16 --iters 15 --out <profile dir>
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

HERE = os.path.dirname(os.path.abspath(__file__))
PASS_NAMES = ("_gqa_bwd_sdw", "_gqa_bwd_pds", "_gqa_bwd_mul")
MSPROF_METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline"
TIMEOUT_S = 2400


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


def _median(vals):
    return round(statistics.median(vals), 4) if vals else None


def collect(out_dir):
    op_rows = _rows(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    by_op = {}
    for r in op_rows:
        name = (r.get("Op Name") or "").strip()
        dur = _f(r, "Task Duration(us)")
        if name and dur is not None:
            by_op.setdefault(name, []).append(dur)
    per_pass = {}
    for name, durs in by_op.items():
        entry = {"n": len(durs), "median_us": _median(durs),
                 "min_us": round(min(durs), 4), "max_us": round(max(durs), 4)}
        # PipeUtilization has no `Op Name` column; the op name is the
        # directory component: OPPROF_*/<op_name>/<launch>/PipeUtilization_*.csv
        pr = _rows(os.path.join(out_dir, "OPPROF_*", name, "*", "PipeUtilization_*.csv"))
        for key, label in (
            ("aic_cube_ratio", "cube_ratio"),
            ("aic_mac_ratio", "mac_ratio"),
            ("aiv_vec_ratio", "vec_ratio"),
            ("aiv_scalar_ratio", "scalar_ratio"),
            ("aiv_mte2_ratio", "mte2_ratio"),
            ("aiv_mte3_ratio", "mte3_ratio"),
        ):
            vals = [v for r in pr if (v := _f(r, key)) is not None]
            entry[label] = round(statistics.fmean(vals), 4) if vals else None
        per_pass[name] = entry
    total = 0.0
    found = []
    for p in PASS_NAMES:
        hits = [n for n in by_op if n.startswith(p)]
        if not hits:
            continue
        n = hits[0]
        total += per_pass[n]["median_us"]
        found.append(n)
    return {
        "per_kernel": per_pass,
        "pass_order": found,
        "n_kernels_per_call": len(found),
        "duration_us": round(total, 4),
        "captured_ops": sorted(by_op),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--H_kv", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--dtype", required=True, choices=("float16", "bfloat16"))
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--out", required=True)
    # Log sub-directory under perf_opt/logs/.  Defaults to the historical
    # round20 bucket; later rounds pass their own (e.g. round21_adopt) so that
    # earlier evidence is never overwritten.
    ap.add_argument("--log-dir", default="round20")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    inner = [
        sys.executable,
        os.path.join(HERE, "_bench_kernel.py"),
        "--impl", os.path.abspath(args.impl),
        "--kernel", "main",
        "--B", str(args.B), "--S", str(args.S), "--H", str(args.H),
        "--H_kv", str(args.H_kv), "--D", str(args.D),
        "--dtype", args.dtype, "--causal", str(args.causal),
        "--iters", str(args.iters + args.warm_up),
    ]
    cmd = [
        "msprof", "op",
        f"--output={out_dir}",
        f"--launch-count={3 * args.iters}",
        f"--warm-up={args.warm_up}",
        "--dump=off",
        f"--aic-metrics={MSPROF_METRICS}",
        *inner,
    ]
    log_dir = os.path.join(HERE, "logs", args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, os.path.basename(out_dir) + ".log")
    with open(log, "w") as f:
        f.write(f"# {shlex.join(cmd)}\n")
        f.flush()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(json.dumps({"status": "error", "error": "msprof timeout", "log": log}))
            return 1
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(json.dumps({"status": "error", "rc": proc.returncode, "log": log}))
        return 1
    res = collect(out_dir)
    res.update({
        "status": "ok" if res["n_kernels_per_call"] == 3 else "incomplete",
        "workload": f"B{args.B}-S{args.S}-H{args.H}-Hkv{args.H_kv}-D{args.D}-{args.dtype}",
        "impl": os.path.basename(args.impl),
        "msprof_raw_path": os.path.relpath(out_dir, HERE),
        "log": os.path.relpath(log, HERE),
        "cmd": shlex.join(cmd),
    })
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
