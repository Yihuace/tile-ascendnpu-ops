"""Compute the official roofline Ratio from one `msprof op` OPPROF directory.

This is a read-only helper that replicates `tileops/benchmark/msprof_roofline.py`
CSV path (lines 200-300) so a Stage-4 candidate can be scored with the same
metric without touching the (read-only) Stage-5 wrapper.  It is validated
against the harness number: the C3 baseline run must reproduce Ratio = 53.211%.

Usage:
    python3 ratio_from_opprof.py <OPPROF_dir> [<OPPROF_dir> ...]
"""

import csv
import glob
import os
import statistics as st
import sys

SOC = {"num_cores": 48, "cacheline_size": 128, "gm_bw": 1.8, "l2_bw": 8.0}


def _rows(op_dir, kind):
    out = []
    for f in sorted(glob.glob(os.path.join(op_dir, "main", "*", f"{kind}_*.csv"))):
        with open(f, encoding="utf-8") as handle:
            out.extend(csv.DictReader(handle))
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def score(op_dir):
    base = _rows(op_dir, "OpBasicInfo")
    ar = _rows(op_dir, "ArithmeticUtilization")
    l2 = _rows(op_dir, "L2Cache")
    if not base or not ar:
        raise SystemExit(f"no CSVs under {op_dir}")

    duration = st.median([_f(r["Task Duration(us)"]) for r in base])
    freq = st.median([_f(r["Current Freq"]) for r in base])
    cores = SOC["num_cores"]

    # per-launch medians (the CSV holds one row per launch; summing would scale by N)
    vec_fops = st.median([_f(r["aiv_vec_fops"]) for r in ar])
    sum_fp32 = st.median([_f(r["aiv_vec_fp32_ratio"]) * _f(r["aiv_total_cycles"]) for r in ar])
    sum_misc = st.median([_f(r["aiv_vec_misc_ratio"]) * _f(r["aiv_total_cycles"]) for r in ar])
    sum_vec = st.median([_f(r["aiv_vec_ratio"]) * _f(r["aiv_total_cycles"]) for r in ar])
    fp32_prop = sum_fp32 / sum_vec if sum_vec else 0.0
    misc_prop = sum_misc / sum_vec if sum_vec else 0.0
    comp = (64 * fp32_prop + 32 * misc_prop) * freq * 1e6 * cores

    miss = (
        st.median(
            [
                _f(r.get("aiv_r0_read_cache_miss_allocate"))
                + _f(r.get("aiv_r1_read_cache_miss_allocate"))
                + _f(r.get("aic_r0_read_cache_miss_allocate"))
                + _f(r.get("aic_r1_read_cache_miss_allocate"))
                for r in l2
            ]
        )
        if l2
        else 0.0
    )
    total_bytes = miss * SOC["cacheline_size"]

    perf = vec_fops / (duration * 1e-6) if duration else 0.0
    ai = vec_fops / total_bytes if total_bytes else 0.0
    peak = min(ai * SOC["gm_bw"] * 1e12, comp)
    ratio = perf / peak if peak else 0.0
    return {
        "dir": os.path.basename(os.path.dirname(os.path.dirname(op_dir.rstrip("/")))),
        "duration_us": duration,
        "freq_mhz": freq,
        "fops": vec_fops,
        "fp32_prop": fp32_prop,
        "misc_prop": misc_prop,
        "computility": comp / 1e12,
        "AI": ai,
        "perf": perf / 1e12,
        "ratio_pct": ratio * 100.0,
        "branch": "compute" if ai * SOC["gm_bw"] * 1e12 > comp else "memory",
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        r = score(d)
        print(
            f"{r['dir']:32s} dur={r['duration_us']:9.2f}us fops={r['fops']:12.0f} "
            f"fp32_prop={r['fp32_prop']:.3f} misc={r['misc_prop']:.3f} "
            f"Comp={r['computility']:.3f}TOps/s AI={r['AI']:.2f} ({r['branch']}) "
            f"Perf={r['perf']:.3f} -> Ratio={r['ratio_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
