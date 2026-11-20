# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Round-3 experiment batch runner (T-4 template, adapted).

Runs, for one candidate module:
  1. L0 precision regression (``python <module> --level L0``)
  2. ``msprof op`` Task Duration per tune workload (via ``run_prof.sh``)
  3. appends one ``perf_records.jsonl`` row per workload (append-only)

Usage:
  python run_exp.py --module <cand.py> --cand-id <id> --parent <pid> --round 3 \
      [--stage round3] [--workloads b1,b4,b8] [--launch-count 10] [--skip-l0]

The workload parameters come from ``bench_driver.CASES`` (official manifest
expansion).  No new injection interface is introduced: the module is imported
by path exactly like the Stage 3 kernel is.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KERNEL_ID = (
    "examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/"
    "engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main"
)

SHORT = {
    "b1": "decode-b1-dmem512-d256-fp16",
    "b4": "decode-b4-dmem1024-d512-fp16",
    "b8": "decode-b8-dmem512-d256-bf16",
    "smoke": "decode-smoke-b1-dmem256-d256-fp16",
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--cand-id", required=True)
    ap.add_argument("--parent", default=None)
    ap.add_argument("--round", type=int, default=3)
    ap.add_argument("--phase", default="candidate")
    ap.add_argument("--stage", default="round3")
    ap.add_argument("--workloads", default="b1,b4,b8")
    ap.add_argument("--launch-count", default="10")
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--only-prof", action="store_true")
    args = ap.parse_args()

    module = os.path.abspath(args.module)
    logdir = os.path.join(HERE, "logs", args.stage)
    os.makedirs(logdir, exist_ok=True)
    profdir = os.path.join(HERE, "profiles", args.stage)
    os.chmod(profdir, 0o700) if os.path.isdir(profdir) else None

    l0_pass = None
    if not args.skip_l0:
        lg = os.path.join(logdir, f"L0_{args.cand_id}.log")
        with open(lg, "w", encoding="utf-8") as f:
            rc = subprocess.run([sys.executable, module, "--level", "L0"],
                                stdout=f, stderr=subprocess.STDOUT).returncode
        l0_pass = rc == 0
        print(f"[L0] {args.cand_id} {'PASS' if l0_pass else 'FAIL'} ({lg})",
              flush=True)
        if not l0_pass:
            print(f"[skip] {args.cand_id} L0 failed -> no perf capture")
            return 1

    rows = []
    for short in args.workloads.split(","):
        case = SHORT[short]
        tag = f"{args.cand_id}_{short}"
        cmd = [os.path.join(HERE, "run_prof.sh"), profdir, tag,
               args.launch_count,
               sys.executable, os.path.join(HERE, "bench_driver.py"),
               "--module", module, "--case", case, "--iters", "20"]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout.strip().splitlines()
        print(f"[prof] {args.cand_id} {short} {out[-1] if out else p.stderr[-300:]}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        pj = os.path.join(profdir, tag, "prof.json")
        if p.returncode != 0 or not os.path.isfile(pj):
            continue
        d = json.load(open(pj, encoding="utf-8"))
        if "duration_us" not in d:
            continue
        row = {
            "round": args.round,
            "candidate_id": args.cand_id,
            "parent_id": args.parent,
            "kernel_id": KERNEL_ID,
            "workload_id": case,
            "phase": args.phase,
            "artifact_path": os.path.basename(module),
            "artifact_sha256": sha256(module),
            "duration_us": round(d["duration_us"], 3),
            "l0_pass": bool(l0_pass) if l0_pass is not None else True,
            "msprof_raw_path": os.path.join("profiles", args.stage, tag),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        rows.append((row, d))
        with open(os.path.join(HERE, "perf_records.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    print("\n| candidate | workload | dur_us | ratio% | vec_t | mte2_t | scalar_t |"
          " vec_r | mte2_r | scalar_r | mte2_bw |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row, d in rows:
        def g(k, nd=2):
            v = d.get(k)
            return "—" if v is None else round(v, nd)
        print(f"| {args.cand_id} | {row['workload_id']} | {row['duration_us']} |"
              f" {g('ratio_pct', 3)} | {g('aiv_vec_time(us)')} |"
              f" {g('aiv_mte2_time(us)')} | {g('aiv_scalar_time(us)')} |"
              f" {g('aiv_vec_ratio', 3)} | {g('aiv_mte2_ratio', 3)} |"
              f" {g('aiv_scalar_ratio', 3)} | {g('aiv_mte2_active_bw')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
