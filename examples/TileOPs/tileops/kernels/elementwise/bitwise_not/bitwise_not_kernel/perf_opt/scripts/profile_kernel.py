#!/usr/bin/env python3
"""Kernel-only profiler for Stage 4 tuning (msprof op Task Duration, us).

Generates a standalone launch script that imports a candidate kernel module
*by file path* (so experiments never touch the read-only baseline or wrapper),
builds the workload input on the NPU, and runs the kernel ``warm_up +
launch_count`` times under ``msprof op``.  Parses ``OpBasicInfo*.csv`` and
reports the median Task Duration(us).

Only measurement entry point: ``msprof op``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _parse_durations(output_dir: str) -> tuple[list[float], list[str]]:
    csv_files = sorted(
        glob.glob(os.path.join(output_dir, "**", "OpBasicInfo*.csv"), recursive=True)
    )
    durations: list[float] = []
    op_names: set[str] = set()
    for csv_path in csv_files:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("Task Duration(us)")
                if val is not None and val.strip():
                    durations.append(float(val))
                name = row.get("Op Name", "")
                if name and name.strip():
                    op_names.add(name.strip())
    return durations, sorted(op_names)


TORCH_DTYPE = {
    "bool": "torch.bool",
    "uint8": "torch.uint8",
    "int8": "torch.int8",
    "int16": "torch.int16",
    "int32": "torch.int32",
    "int64": "torch.int64",
}


def _gen_script(kernel_file: str, N: int, dtype_str: str, warm_up: int, launch_count: int,
                device_index: int) -> str:
    return f'''\
# Auto-generated msprof profiling script (Stage 4 candidate kernel)
import os, sys, importlib.util
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import warnings
warnings.filterwarnings("ignore")
import torch
import torch_npu  # noqa: F401
torch.npu.set_device({device_index})

KERNEL_FILE = {kernel_file!r}
spec = importlib.util.spec_from_file_location("cand_kernel", KERNEL_FILE)
mod = importlib.util.module_from_spec(spec)
sys.modules["cand_kernel"] = mod
spec.loader.exec_module(mod)

N = {N}
DTYPE = {TORCH_DTYPE[dtype_str]}
x = torch.randint(-128, 128, (N,), device="npu", dtype=DTYPE)
kernel = mod.bitwise_not_fwd_kernel(N, {dtype_str!r})
torch.npu.synchronize()

# Warm up (compiles the kernel; result discarded)
for _ in range({warm_up}):
    y = kernel(x, N)
torch.npu.synchronize()

# Measured launches
for _ in range({launch_count}):
    y = kernel(x, N)
torch.npu.synchronize()
print("done", y.shape)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-file", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", required=True, choices=tuple(TORCH_DTYPE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--kernel-name", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    msprof = shutil.which("msprof")
    if msprof is None:
        print(json.dumps({"error": "msprof not found"}))
        return 2

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    # msprof refuses group/world-writable output (and its parent) dirs.
    os.chmod(out_dir, 0o700)
    parent = os.path.dirname(out_dir)
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass
    # clean stale contents
    for entry in os.listdir(out_dir):
        p = os.path.join(out_dir, entry)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)

    tmp_dir = tempfile.mkdtemp(prefix="stage4_prof_")
    script_path = os.path.join(tmp_dir, "prof_script.py")
    with open(script_path, "w") as f:
        f.write(_gen_script(args.kernel_file, args.N, args.dtype, args.warm_up,
                            args.launch_count, args.device_index))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")

    candidates = [args.kernel_name] if args.kernel_name else [None, None]
    last_err = ""
    for kn in candidates:
        cmd = [msprof, "op"]
        if kn:
            cmd.append(f"--kernel-name={kn}")
        cmd += [
            f"--output={out_dir}",
            f"--launch-count={args.launch_count}",
            f"--warm-up={args.warm_up}",
            "--aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,L2Cache",
            "--dump=off",
            sys.executable,
            script_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        if proc.returncode != 0:
            last_err = proc.stderr.strip()[-800:]
            continue
        try:
            durations, op_names = _parse_durations(out_dir)
        except Exception as e:  # noqa: BLE001
            last_err = f"{e}"
            continue
        if not durations:
            last_err = "no durations"
            continue
        durations.sort()
        n = len(durations)
        result = {
            "kernel_file": args.kernel_file,
            "N": args.N,
            "dtype": args.dtype,
            "kernel_name_filter": kn,
            "samples": n,
            "min_us": round(durations[0], 4),
            "median_us": round(durations[n // 2], 4),
            "max_us": round(durations[-1], 4),
            "captured_op_names": op_names,
            "out_dir": out_dir,
        }
        print("PROFILE_RESULT " + json.dumps(result))
        if not args.keep:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return 0

    print("PROFILE_FAIL " + json.dumps({"error": last_err, "out_dir": out_dir}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
