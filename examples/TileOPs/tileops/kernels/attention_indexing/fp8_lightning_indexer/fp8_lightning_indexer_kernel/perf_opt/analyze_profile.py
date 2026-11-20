"""Aggregate one msprof op capture into the roofline-model terms.

Mirrors ``tileops.benchmark.msprof_roofline.compute_from_csvs`` (same formulas,
same SoC constants) but tolerates the timestamped CSV names and averages over
all ``launch`` sub-directories.  Prints the ops/bytes decomposition that decides
the official ``Ratio(%)`` interpretation.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

CUBE_PEAK_PPC = 4096          # MAC/cycle/core (SoC table, Ascend910B3)
CUBE_ROOFLINE_MULTIPLIER = 384
NUM_CORES = 48
CACHELINE = 128
RATED_GHZ = 1.5213            # back-solved from the bin's 299.094 TOPS


def _rows(root: str, stem: str) -> list:
    out = []
    for path in glob.glob(os.path.join(root, "**", f"{stem}_*.csv"), recursive=True):
        with open(path, newline="") as f:
            out.extend(list(csv.DictReader(f)))
    return out


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    root = sys.argv[1]
    basic = _rows(root, "OpBasicInfo")
    au = _rows(root, "ArithmeticUtilization")
    l2 = _rows(root, "L2Cache")
    pipe = _rows(root, "PipeUtilization")
    mem = _rows(root, "Memory")

    durs = sorted(_f(r, "Task Duration(us)") for r in basic if _f(r, "Task Duration(us)") > 0)
    dur = durs[len(durs) // 2]
    freq = _f(basic[0], "Current Freq")
    block_dim = int(_f(basic[0], "Block Dim"))

    cube_rows = [r for r in au if (r.get("sub_block_id") or "").startswith("cube")]
    vec_rows = [r for r in au if (r.get("sub_block_id") or "").startswith("vector")]
    cube_fops_csv = sum(_f(r, "aic_cube_fops") for r in cube_rows)
    cube_fops = cube_fops_csv * CUBE_ROOFLINE_MULTIPLIER
    vec_fops = sum(_f(r, "aiv_vec_fops") for r in vec_rows)

    miss = sum(_f(r, "aiv_r0_read_cache_miss_allocate") + _f(r, "aiv_r1_read_cache_miss_allocate")
               for r in l2)
    miss += sum(_f(r, "aic_r0_read_cache_miss_allocate") + _f(r, "aic_r1_read_cache_miss_allocate")
                for r in l2)
    total_bytes = miss * CACHELINE

    total_fops = cube_fops + vec_fops
    ratio_peak = 299.0943603515625
    res = dict(
        duration_us=dur, samples=len(durs), min_us=durs[0], max_us=durs[-1],
        freq_mhz=freq, block_dim=block_dim,
        cube_fops=cube_fops, cube_fops_csv=cube_fops_csv,
        vec_fops=vec_fops, total_fops=total_fops,
        vec_frac=vec_fops / total_fops if total_fops else 0.0,
        total_bytes_MB=total_bytes / 1e6,
        AI=total_fops / total_bytes if total_bytes else 0.0,
        perf_TOPS=total_fops / (dur * 1e-6) / 1e12,
        ratio_pct=total_fops / (dur * 1e-6) / ratio_peak * 100,
        ideal_us=total_fops / ratio_peak * 1e6,
        pipe={k: round(sum(_f(r, k) for r in pipe) / max(1, len(pipe)), 4) for k in
              ("aic_cube_ratio", "aic_fixpipe_ratio", "aic_scalar_ratio", "aic_mte1_ratio",
               "aic_mte2_ratio", "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio",
               "aiv_mte3_ratio")},
        mem={k: round(sum(_f(r, k) for r in mem) / max(1, len(mem)), 2) for k in
             ("read_main_memory_datas(KB)", "write_main_memory_datas(KB)")},
    )
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
