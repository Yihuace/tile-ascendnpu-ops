#!/usr/bin/env python3
"""msprof op driver for the Cube-GEMV row-reduce probe.

For each config it runs ONE python process that alternates the Vector (A) and
Cube (B) kernels, and two `msprof op` invocations over that same process (one
per kernel name), so A and B are measured from an identical device state and
in the same session.

  python run_probe.py --configs base,kc512

Timing discipline: `msprof op --launch-count=20 --warm-up=5`, median Task
Duration.  Raw profiles + logs land next to this file.
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
PROBE = os.path.join(HERE, "probe_cube_gemv.py")
PROFILES = os.path.join(HERE, "profiles")
LOGS = os.path.join(HERE, "logs")

KERNELS = {"vec": "vec_reduce_main", "cube": "cube_gemv_main"}
KERNELS_MULTI = {"vec": "vec_reduce_multi_main",
                 "cube": "cube_gemv_multi_main"}

# label -> [M, N, PN, KC, bt]
CONFIGS = {
    "base": {"2x4096": [2, 4096, 1, 0, 0],
             "1x8192": [1, 8192, 1, 0, 0]},
    "bt1": {"2x4096": [2, 4096, 1, 0, 1],
            "1x8192": [1, 8192, 1, 0, 1]},
    "pad16": {"2x4096": [2, 4096, 16, 0, 1],
              "1x8192": [1, 8192, 16, 0, 1]},
    "kc512": {"2x4096": [2, 4096, 1, 512, 1],
              "1x8192": [1, 8192, 1, 512, 1]},
}


def measure(tag, shape, cfg, which):
    M, N, PN, KC, BT = cfg
    out_dir = os.path.join(PROFILES, f"{tag}-{shape}-{which}")
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    log_path = os.path.join(LOGS, f"{tag}-{shape}-{which}.log")
    os.makedirs(LOGS, exist_ok=True)

    cmd = [
        "msprof", "op",
        f"--kernel-name={KERNELS[which]}",
        "--launch-count=20",
        "--warm-up=5",
        f"--output={out_dir}",
        sys.executable, PROBE,
        "--mode", "bench",
        "--iters", "26",
        "--shapes", f"{M}x{N}",
        "--pn", str(PN),
        "--kc", str(KC),
        "--bt", str(BT),
        "--vec", "1",
        "--cube", "1",
    ]
    with open(log_path, "w") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        log.write(proc.stdout or "")
        log.write(proc.stderr or "")
    if proc.returncode != 0:
        print(f"[measure] FAIL {tag}/{shape}/{which} rc={proc.returncode} "
              f"-> {log_path}")
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        for line in tail:
            print("   ", line[:200])
        return None

    durations = []
    names = set()
    for path in sorted(glob.glob(os.path.join(
            out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    names.add(row.get("Kernel Name", row.get("Op Name", "?")))
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        print(f"[measure] FAIL {tag}/{shape}/{which}: no OpBasicInfo rows "
              f"-> {log_path}")
        return None
    return {
        "tag": tag, "shape": shape, "which": which,
        "M": M, "N": N, "PN": PN, "KC": KC, "bt": BT,
        "n": len(durations),
        "median_us": round(statistics.median(durations), 3),
        "min_us": round(min(durations), 3),
        "max_us": round(max(durations), 3),
        "kernel_names": sorted(names),
        "raw": out_dir,
    }


def measure_multi(tag, N, bm, ntasks, which, kc=0, iters=26,
                  vec_flag=True):
    out_dir = os.path.join(PROFILES, f"{tag}-{bm}x{N}-{which}")
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    log_path = os.path.join(LOGS, f"{tag}-{bm}x{N}-{which}.log")
    os.makedirs(LOGS, exist_ok=True)
    cmd = [
        "msprof", "op",
        f"--kernel-name={KERNELS_MULTI[which]}",
        "--launch-count=20",
        "--warm-up=5",
        f"--output={out_dir}",
        sys.executable, PROBE,
        "--mode", "multi",
        "--iters", str(iters),
        "--shapes", f"1x{N}",
        "--bm", str(bm),
        "--ntasks", str(ntasks),
        "--kc", str(kc),
        "--vec", "1" if vec_flag else "0",
        "--cube", "1",
    ]
    with open(log_path, "w") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        log.write(proc.stdout or "")
        log.write(proc.stderr or "")
    if proc.returncode != 0:
        print(f"[multi] FAIL {tag}/{bm}x{N}/{which} rc={proc.returncode} "
              f"-> {log_path}")
        for line in (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]:
            print("   ", line[:200])
        return None
    durations, names = [], set()
    for path in sorted(glob.glob(os.path.join(
            out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    names.add(row.get("Kernel Name", row.get("Op Name", "?")))
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        print(f"[multi] FAIL {tag}/{bm}x{N}/{which}: no rows -> {log_path}")
        return None
    med = statistics.median(durations)
    M = bm * ntasks
    return {
        "tag": tag, "N": N, "bm": bm, "ntasks": ntasks, "kc": kc,
        "which": which,
        "M": M, "n": len(durations),
        "median_us": round(med, 3),
        "per_tile_us": round(med / ntasks, 4),
        "per_elem_ps": round(med * 1e6 / (M * N), 4),
        "min_us": round(min(durations), 3),
        "max_us": round(max(durations), 3),
        "kernel_names": sorted(names),
        "raw": out_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="base,kc512")
    ap.add_argument("--mode", default="single", choices=["single", "multi"])
    args = ap.parse_args()

    records = []
    if args.mode == "multi":
        for tag, N, bm, ntasks, kc in (
                ("multi-bm2-full", 4096, 2, 128, 0),
                ("multi-bm2-kc512", 4096, 2, 128, 512),
                ("multi-bm1-kc512", 8192, 1, 128, 512),
                ("multi-bm1-kc1024", 8192, 1, 128, 1024),
                ("multi-bm16-full", 4096, 16, 8, 0),
                ("multi-bm16-kc512", 8192, 16, 8, 512),
                ("multi-bm4-full", 4096, 4, 32, 0),
                ("multi-bm8-full", 4096, 8, 16, 0)):
            whiches = ("vec", "cube") if bm <= 8 else ("cube",)
            for which in whiches:
                rec = measure_multi(tag, N, bm, ntasks, which, kc,
                                    vec_flag=(bm <= 8))
                if rec:
                    records.append(rec)
                    print(f"{tag}/{bm}x{N}/{which}: median={rec['median_us']}us "
                          f"per_tile={rec['per_tile_us']}us n={rec['n']}")
        with open(os.path.join(HERE, "probe_results_multi.json"), "w") as handle:
            json.dump(records, handle, indent=2)
        print("\n| config | bm | N | ntasks | path | median us | per-tile us |")
        print("|---|---|---|---|---|---|---|")
        for rec in records:
            print(f"| {rec['tag']} | {rec['bm']} | {rec['N']} | "
                  f"{rec['ntasks']} | {rec['kc']} | {rec['which']} | "
                  f"{rec['median_us']} | {rec['per_tile_us']} |")
        return

    for tag in args.configs.split(","):
        for shape, cfg in CONFIGS[tag].items():
            for which in ("vec", "cube"):
                rec = measure(tag, shape, cfg, which)
                if rec:
                    records.append(rec)
                    print(f"{tag}/{shape}/{which}: median={rec['median_us']}us "
                          f"n={rec['n']} names={rec['kernel_names']}")

    with open(os.path.join(HERE, "probe_results.json"), "w") as handle:
        json.dump(records, handle, indent=2)

    print("\n| config | shape | path | median us | min | max | n |")
    print("|---|---|---|---|---|---|---|")
    for rec in records:
        print(f"| {rec['tag']} | {rec['M']}x{rec['N']} | {rec['which']} | "
              f"{rec['median_us']} | {rec['min_us']} | {rec['max_us']} | "
              f"{rec['n']} |")


if __name__ == "__main__":
    main()
