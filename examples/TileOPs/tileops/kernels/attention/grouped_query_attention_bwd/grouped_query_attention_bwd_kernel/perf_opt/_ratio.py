"""Extract the official msprof roofline metric for a raw `msprof op` profile dir.

The official workload Ratio is defined in
``examples/TileOPs/tileops/benchmark/benchmark_base.py::_parse_roofline``: from
``visualize_data.bin`` take the ``roofline_entries`` row with
``title == "GM/L2"`` and ``bw_name`` containing ``"GM Read + Write"`` and report
``Ratio(%) = ratio * 100``.

This helper reports that row plus the derived quantities that make the number
auditable:

    ratio          = performance / min(AI * bw, computility)
    performance    = counted_flops / duration      (TOps/s)
    counted_flops  = performance * duration

Usage:
  python _ratio.py <profile_dir> [<profile_dir> ...]
"""

import glob
import json
import os
import statistics
import sys

sys.path.insert(0, "/mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs")
from tileops.benchmark.msprof_roofline import parse_bin_file  # noqa: E402


def extract(profile_dir):
    bins = sorted(glob.glob(os.path.join(profile_dir, "**", "visualize_data.bin"),
                            recursive=True))
    out = []
    for b in bins:
        d = parse_bin_file(b)
        gm = [e for e in d["roofline_entries"]
              if e.get("title") == "GM/L2" and "GM Read + Write" in e.get("bw_name", "")]
        if not gm:
            continue
        e = gm[0]
        bi = d["base_info"]
        out.append({
            "duration_us": bi.get("duration"),
            "soc": bi.get("soc"),
            "freq_mhz": bi.get("freq"),
            "op_type": bi.get("op_type"),
            "ratio_pct": e["ratio"] * 100.0,
            "perf_tops": e["performance"],
            "computility_tops": e["computility"],
            "ai_ops_per_byte": e["AI"],
            "bw_tbs": e["bw"],
            "counted_gflop": e["performance"] * bi.get("duration", 0.0) * 1e-3,
            "mem_bound_tops": e["AI"] * e["bw"],
        })
    if not out:
        return None
    agg = {}
    for k in ("duration_us", "ratio_pct", "perf_tops", "counted_gflop"):
        agg[k] = round(statistics.median([o[k] for o in out]), 4)
    agg.update({
        "computility_tops": out[0]["computility_tops"],
        "ai_ops_per_byte": out[0]["ai_ops_per_byte"],
        "bw_tbs": out[0]["bw_tbs"],
        "mem_bound_tops": out[0]["mem_bound_tops"],
        "soc": out[0]["soc"],
        "op_type": out[0]["op_type"],
        "n_bins": len(out),
        "binding": "compute" if out[0]["mem_bound_tops"] > out[0]["computility_tops"] else "memory",
    })
    return agg


def main():
    for d in sys.argv[1:]:
        r = extract(d)
        print(json.dumps({"dir": d, **(r or {"error": "no roofline bin"})},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
