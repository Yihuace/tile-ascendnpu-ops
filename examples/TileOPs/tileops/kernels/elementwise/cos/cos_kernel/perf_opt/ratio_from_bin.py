"""Print the official roofline Ratio from an OPPROF dir (bin path, Method 1).

The package benchmark (`ManifestBenchmark.profile`) reads
`visualize_data.bin` and filters the `GM/L2` + `GM Read + Write` roofline entry,
so that is exactly what this helper does -- same parser, same filter.  The Stage-4
candidate can therefore be scored with the official number without touching the
read-only Stage-5 wrapper.

Requires `msprof op --aic-metrics=Roofline` (that is what produces the bin).

Usage:
    python3 ratio_from_bin.py <OPPROF_dir> [...]
"""

import glob
import json
import os
import sys

from tileops.benchmark.msprof_roofline import parse_bin_file


def score(prof_dir):
    bins = sorted(glob.glob(os.path.join(prof_dir, "**", "visualize_data.bin"), recursive=True))
    if not bins:
        return None
    best = None
    for path in bins:
        data = parse_bin_file(path)
        for entry in data.get("roofline_entries", []):
            if entry.get("title") == "GM/L2" and "GM Read + Write" in entry.get("bw_name", ""):
                if best is None or entry["ratio"] > best["ratio"]:
                    best = dict(entry)
                    best["_dir"] = os.path.basename(os.path.dirname(os.path.dirname(path)))
                    best["_duration"] = data.get("base_info", {}).get("duration")
    return best


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        hit = score(d)
        if hit is None:
            print(f"{d}: no roofline entry (need --aic-metrics=Roofline)")
            continue
        print(
            f"{hit['_dir']:34s} dur={float(hit['_duration']):9.2f}us "
            f"fops={hit['fops']:12.0f} bytes={hit.get('bytes', 0):12.0f} "
            f"AI={hit['AI']:7.3f} Perf={hit['performance']:7.4f} "
            f"Comp={hit['computility']:7.4f} -> Ratio={hit['ratio'] * 100:.4f}%"
        )
        if os.environ.get("RATIO_JSON"):
            print(json.dumps(hit, default=str))


if __name__ == "__main__":
    main()
