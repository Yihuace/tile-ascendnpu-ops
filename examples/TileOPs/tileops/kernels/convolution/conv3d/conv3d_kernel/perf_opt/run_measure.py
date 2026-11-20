#!/usr/bin/env python3
"""Stage-4 batch msprof-op measurement runner for conv3d (adapted T-4 runner).

Runs `measure.py` (msprof op, kernel-only Task Duration) for a list of
(tag, module, workload, dtype) entries, stores raw profiles under
``perf_opt/profiles/<workload>/<phase>/``, appends one structured line per
measurement to ``perf_opt/perf_records.jsonl`` (append-only) and prints a
summary table.

Usage (must be wrapped in .agents/tools/npu_lock.sh):
  python3 run_measure.py --phase baseline --tag stage3 \
      --items r3d-stem-bias:float16:_conv3d_kernel.py:...
Only the shapes used by the historical rounds are supported; see ITEMS presets.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)  # conv3d_kernel/
STAGE3 = os.path.join(KDIR, "_conv3d_kernel.py")
PERFOPT = os.path.join(HERE, "Conv3dBiasFwdOp.py")
RECORDS = os.path.join(HERE, "perf_records.jsonl")

TUNE = [
    ("r3d-stem-bias", "float16"),
    ("video-down-bias", "float16"),
    ("unet-encoder-bias", "bfloat16"),
]
SMOKE = [("smoke-k3-s1-bias", "float16"), ("smoke-k3-s1-bias", "bfloat16")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one(module, workload, dtype, outdir, launches, extra=()):
    os.makedirs(outdir, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "measure.py"), module, workload,
           dtype, "--out", outdir, "--launches", str(launches)]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    tail = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        return None, tail
    m = re.search(r"duration_us=([0-9.]+) samples=(\d+)", r.stdout)
    if not m:
        return None, tail
    return (float(m.group(1)), int(m.group(2))), tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    help="baseline|r1|r2|...|merged|final")
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--module", default=PERFOPT)
    ap.add_argument("--items", default="tune",
                    help="tune|smoke|all|comma list workload:dtype")
    ap.add_argument("--launches", type=int, default=20)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--block-m", type=int, default=None)
    ap.add_argument("--block-n", type=int, default=None)
    ap.add_argument("--block-k", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.items == "tune":
        items = TUNE
    elif args.items == "smoke":
        items = SMOKE
    elif args.items == "all":
        items = TUNE + SMOKE
    else:
        items = [(w, d) for w, d in (x.split(":") for x in args.items.split(","))]

    extra = []
    for name, val in (("--block-m", args.block_m), ("--block-n", args.block_n),
                      ("--block-k", args.block_k)):
        if val:
            extra += [name, str(val)]

    module = os.path.abspath(args.module)
    digest = sha256(module)
    results = []
    for workload, dtype in items:
        outdir = os.path.join(HERE, "profiles", workload, args.phase)
        os.makedirs(outdir, exist_ok=True)
        for stale in os.listdir(outdir):
            p = os.path.join(outdir, stale)
            if os.path.isdir(p):
                subprocess.run(["rm", "-rf", p])
        res, tail = run_one(module, workload, dtype, outdir, args.launches, extra)
        logdir = os.path.join(HERE, "logs", args.phase)
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, f"{workload}_{dtype}.log"), "w") as f:
            f.write(tail)
        if res is None:
            print(f"[run_measure] FAIL {workload} {dtype} (see logs/{args.phase})")
            print(tail[-2000:])
            results.append((workload, dtype, None, 0))
            continue
        dur, n = res
        results.append((workload, dtype, dur, n))
        if not args.no_record:
            rec = {
                "round": args.round,
                "candidate_id": args.candidate_id,
                "parent_id": args.parent_id,
                "kernel_id": "conv3d",
                "workload_id": f"{workload}-{dtype}",
                "phase": args.phase,
                "artifact_path": module,
                "artifact_sha256": digest,
                "duration_us": dur,
                "l0_pass": True,
                "msprof_raw_path": outdir,
                "timestamp": __import__("datetime").datetime.now().astimezone().isoformat(
                    timespec="seconds"),
            }
            with open(RECORDS, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n## summary  phase=%s candidate=%s module=%s sha=%s%s"
          % (args.phase, args.candidate_id, os.path.basename(module), digest[:12],
             (" tag=" + args.tag) if args.tag else ""))
    print("| workload | dtype | Task Duration(us) | samples |")
    print("|---|---|---:|---:|")
    for workload, dtype, dur, n in results:
        print(f"| {workload} | {dtype} | "
              f"{'FAIL' if dur is None else f'{dur:.2f}'} | {n} |")


if __name__ == "__main__":
    main()
