"""Re-parse an existing `msprof op` profile dir -> one JSON line of pipe/memory metrics.

Usage: python parse_profile.py <profile_dir> [<profile_dir> ...]

Reads OpBasicInfo*.csv (Task Duration) and the first PipeUtilization*/Memory*/
MemoryL0* CSVs (aic/aiv pipe times & ratios, GM<->L1 / L0C->GM byte counts) from
the deepest directory that contains them.  Same caliber as the Stage-4 rule:
Task Duration median over the 20 encoded launches.
"""

import glob
import json
import os
import statistics
import sys


def _csv_rows(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    return [dict(zip(header, ln.split(","))) for ln in lines[1:]]


def _find(d, pattern):
    """Deepest-first match (per-launch CSVs live in the leaf op dir)."""
    hits = glob.glob(os.path.join(d, "**", pattern), recursive=True)
    return sorted(hits, key=len)[-1] if hits else None


def _find_all(d, pattern):
    return sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True))


def parse(d):
    rows = []
    for op in _find_all(d, "OpBasicInfo*.csv"):
        rows.extend(_csv_rows(op))
    durs = sorted(float(r["Task Duration(us)"]) for r in rows if r.get("Task Duration(us)"))
    out = {
        "dir": d,
        "n": len(durs),
        "median_us": durs[len(durs) // 2] if durs else None,
        "min_us": durs[0] if durs else None,
        "max_us": durs[-1] if durs else None,
        "mean_us": round(statistics.mean(durs), 2) if durs else None,
        "block_dim": rows[0].get("Block Dim") if rows else None,
    }
    pipe = _find(d, "PipeUtilization*.csv")
    if pipe:
        r = _csv_rows(pipe)[0]
        for k in ("aic_time(us)", "aic_cube_time(us)", "aic_cube_ratio",
                  "aic_mte1_time(us)", "aic_mte1_ratio",
                  "aic_mte2_time(us)", "aic_mte2_ratio",
                  "aic_fixpipe_time(us)", "aic_fixpipe_ratio",
                  "aic_scalar_time(us)", "aic_scalar_ratio",
                  "aic_scalar_wait_time(us)",
                  "aiv_vec_time(us)", "aiv_vec_ratio",
                  "aiv_scalar_ratio", "aiv_scalar_wait_time(us)"):
            v = r.get(k)
            out[k] = None if v in (None, "NA", "") else float(v)
        # per-event-id cross-engine wait, AIC side
        waits = {k: float(v) for k, v in r.items()
                 if k.startswith("aic_scalar_wait_id") and v not in ("NA", "")}
        out["aic_wait_id_sum(us)"] = round(sum(waits.values()), 1)
    mem = _find(d, "Memory_*.csv")
    if mem:
        r = _csv_rows(mem)[0]
        for k in ("GM_to_L1_datas(KB)", "GM_to_L1_bw_usage_rate(%)",
                  "L0C_to_GM_datas(KB)", "L0C_to_GM_bw_usage_rate(%)",
                  "GM_to_UB_datas(KB)", "aic_mte2_instructions",
                  "aic_mte1_instructions", "read_main_memory_datas(KB)",
                  "write_main_memory_datas(KB)"):
            v = r.get(k)
            out[k] = None if v in (None, "NA", "") else float(v)
    l0 = _find(d, "MemoryL0_*.csv")
    if l0:
        r = _csv_rows(l0)[0]
        for k in ("aic_l1_read_bw(GB/s)", "aic_l1_write_bw(GB/s)",
                  "aic_l0a_read_bw(GB/s)", "aic_l0b_read_bw(GB/s)",
                  "aic_l0c_read_bw_cube(GB/s)", "aic_l0c_write_bw_cube(GB/s)"):
            v = r.get(k)
            out[k] = None if v in (None, "NA", "") else float(v)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(parse(d), ensure_ascii=False))
