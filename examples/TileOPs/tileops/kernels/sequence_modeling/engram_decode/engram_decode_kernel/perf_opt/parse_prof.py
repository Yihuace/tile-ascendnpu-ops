# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Parse one ``msprof op`` output tree into a flat JSON summary.

Emits (stdout, one JSON object):
  duration_us   median of task durations across all captured launches
  block_dim, op_type, soc, perf_top
  aiv_time/vec/mte2/mte3/scalar/ratios (median over blocks & launches)
  read_kb / write_kb / gm2ub_kb
  roofline      GM/L2 "GM Read + Write": bw, AI, performance, computility, ratio
  ideal_us      duration_us * ratio   (roofline-limited ideal time)

Usage: python parse_prof.py <profile_dir>
"""

import glob
import json
import os
import statistics
import sys

# --- bootstrap: locate the TileOPs package root (for the msprof bin parser) ---
_here = os.path.dirname(os.path.abspath(__file__))
_d = _here
for _ in range(12):
    if os.path.isfile(os.path.join(_d, "tileops", "benchmark", "msprof_roofline.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _d = os.path.dirname(_d)

CSV_PREFIX = {
    "basic": "OpBasicInfo_",
    "pipe": "PipeUtilization_",
    "mem": "Memory_",
    "arith": "ArithmeticUtilization_",
}
NUM_COLS = {
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_vec_fp32_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte2_active_bw(GB/s)",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_main_mem_read_bw(GB/s)",
    "read_main_memory_datas(KB)",
    "write_main_memory_datas(KB)",
    "GM_to_UB_datas(KB)",
    "UB_to_GM_datas(KB)",
    "aiv_vec_fops",
}


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n").rstrip(",") for ln in f if ln.strip()]
    header = lines[0].split(",")
    rows = []
    for ln in lines[1:]:
        vals = ln.split(",")
        rows.append(dict(zip(header, vals)))
    return rows


def _aggregate(paths, keys):
    vals = {k: [] for k in keys}
    for p in paths:
        for row in _read_csv(p):
            for k in keys:
                v = row.get(k)
                if v in (None, "", "NA"):
                    continue
                try:
                    vals[k].append(float(v))
                except ValueError:
                    pass
    return {k: (statistics.median(v) if v else None) for k, v in vals.items()}


def main():
    root = sys.argv[1]
    out = {}
    basics = sorted(glob.glob(os.path.join(root, "**", CSV_PREFIX["basic"] + "*.csv"),
                              recursive=True))
    pipes = sorted(glob.glob(os.path.join(root, "**", CSV_PREFIX["pipe"] + "*.csv"),
                             recursive=True))
    mems = sorted(glob.glob(os.path.join(root, "**", CSV_PREFIX["mem"] + "*.csv"),
                            recursive=True))
    if not basics:
        print(json.dumps({"error": "no OpBasicInfo csv under " + root}))
        return 1
    durs, blocks, ops, socs = [], [], set(), set()
    for p in basics:
        for row in _read_csv(p):
            durs.append(float(row["Task Duration(us)"]))
            blocks.append(row.get("Block Dim"))
            ops.add(row.get("Op Type"))
            socs.add(row.get("soc", ""))
    out["duration_us"] = statistics.median(durs)
    out["n_launches"] = len(durs)
    out["duration_min_us"] = min(durs)
    out["block_dim"] = blocks[0]
    out["op_type"] = ",".join(sorted(ops))
    out["soc"] = ",".join(sorted(s for s in socs if s))
    ariths = sorted(glob.glob(os.path.join(root, "**", CSV_PREFIX["arith"] + "*.csv"),
                             recursive=True))
    out.update(_aggregate(pipes, NUM_COLS))
    out.update(_aggregate(ariths, {"aiv_vec_fops", "aiv_vec_fp32_ratio"}))
    out.update(_aggregate(mems, {"read_main_memory_datas(KB)", "write_main_memory_datas(KB)",
                                "GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)",
                                "aiv_main_mem_read_bw(GB/s)"}))

    roofline = None
    if len(sys.argv) > 2 and sys.argv[2] == "--no-roofline":
        pass
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file
    except Exception:  # noqa: BLE001
        parse_bin_file = None
    bins = sorted(glob.glob(os.path.join(root, "**", "visualize_data.bin"), recursive=True))
    if parse_bin_file and bins:
        # one bin per captured launch; take the median-performing entry
        entries = []
        for b in bins:
            try:
                data = parse_bin_file(b)
            except Exception:  # noqa: BLE001
                continue
            for e in data.get("roofline_entries", []):
                if e.get("title") == "GM/L2" and "GM Read + Write" in e.get("bw_name", ""):
                    entries.append({"bw": e["bw"], "AI": e["AI"],
                                    "performance": e["performance"],
                                    "computility": e["computility"],
                                    "ratio": e["ratio"]})
        if entries:
            roofline = {k: statistics.median([e[k] for e in entries])
                        for k in ("bw", "AI", "performance", "computility", "ratio")}
    out["roofline"] = roofline
    if roofline:
        out["ratio_pct"] = roofline["ratio"] * 100.0
        out["ideal_us"] = out["duration_us"] * roofline["ratio"]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
