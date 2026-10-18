#!/usr/bin/env python3
"""msprof-op Task Duration measurement for one conv1d kernel workload.

Runs the standalone bench_kernel.py driver under ``msprof op`` and prints the
median Task Duration(us) parsed from the OpBasicInfo CSV.  This mirrors
tileops.benchmark.msprof.bench_kernel_msprof but drives a candidate .py module
directly (bypassing the Stage-5 wrapper, which still imports the baseline).

Usage:
  measure.py <module.py> <workload> <dtype> <bias(0/1)> [--out DIR]

Workload names match bench_kernel.py WORKLOADS (unit_stride) plus general-kernel
names handled by the extended driver.
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = "/mnt/workspace/npuir-migration"
TILEOPS = os.path.join(REPO, "examples/TileOPs")
KDIR = os.path.join(TILEOPS, "tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt")
MSPROF = "/home/developer/Ascend/cann-9.0.0/bin/msprof"

# kernel-name filter per (workload has_bias) -> resolved @T.prim_func name.
def kernel_name(workload, has_bias, is_general):
    base = "conv1d" if is_general else "conv1d_unit_stride"
    suffix = "_bias" if has_bias else ""
    return f"_{base}{suffix}_main"


def parse_duration(outdir):
    csvs = sorted(glob.glob(os.path.join(outdir, "**", "OpBasicInfo*.csv"), recursive=True))
    dur = []
    for c in csvs:
        with open(c, newline="") as f:
            for row in csv.DictReader(f):
                v = row.get("Task Duration(us)")
                if v and v.strip():
                    try:
                        dur.append(float(v))
                    except ValueError:
                        pass
    if not dur:
        return None, None
    dur.sort()
    return dur[len(dur) // 2], len(dur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("workload")
    ap.add_argument("dtype", choices=["float16", "bfloat16"])
    ap.add_argument("bias", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--launches", type=int, default=25)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--warm-up", type=int, default=5)
    args = ap.parse_args()

    module = os.path.abspath(args.module)
    # general kernel dispatch: wav2vec2 / encodec-init / smoke
    is_general = args.workload.startswith(("wav2vec2", "encodec-init", "smoke"))

    kname = kernel_name(args.workload, bool(args.bias), is_general)

    outdir = args.out or tempfile.mkdtemp(prefix="msprof_conv1d_")
    os.makedirs(outdir, exist_ok=True)
    os.chmod(outdir, 0o700)

    cmd = [
        MSPROF, "op",
        f"--kernel-name={kname}",
        f"--output={outdir}",
        f"--launch-count={args.launch_count}",
        f"--warm-up={args.warm_up}",
        "--aic-metrics=Roofline",
        "--dump=off",
        sys.executable,
        os.path.join(KDIR, "bench_kernel.py"),
        "--module", module,
        "--workload", args.workload,
        "--dtype", args.dtype,
        "--bias", str(args.bias),
        "--launches", str(args.launches),
    ]
    env = dict(os.environ)
    env["TILELANG_ASCEND_MODE"] = "Expert"
    print(f"[measure] {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    if r.returncode != 0:
        print(f"[measure] FAILED rc={r.returncode}", file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    median, nsamp = parse_duration(outdir)
    if median is None:
        # retry without --kernel-name (unfiltered)
        cmd2 = [a for a in cmd if not a.startswith("--kernel-name=")]
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=900, env=env)
        median, nsamp = parse_duration(outdir)
    if median is None:
        print("[measure] no OpBasicInfo CSV", file=sys.stderr)
        sys.exit(1)
    print(f"MEASURE workload={args.workload} dtype={args.dtype} bias={args.bias} "
          f"duration_us={median:.4f} samples={nsamp}")
    if not args.out:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
