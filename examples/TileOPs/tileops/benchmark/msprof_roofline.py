"""
Compare Roofline metrics: visualize_data.bin (Method 1) vs CSV formulas (Method 2).

Supports vector, cube, and mix (cube+vector) kernel types.

Usage:
    python3 compare_roofline.py <OPPROF_directory>

Example:
    python3 compare_roofline.py /tmp/tileops_msprof_2fun3kfv/msprof_output/OPPROF_20260813122049_IDYAVJINHLQBOPMQ
"""

import csv
import json
import os
import struct
import sys
from collections import defaultdict

# ---- SoC configuration ----
SOC_CONFIG = {
    "Ascend910B2C": {
        "num_cores": 48,
        "cacheline_size": 128,
        "vec_peak_ppc": {
            "FP32": 64,
            "MISC": 32,
            "FP16": 128,
            "INT32": 64,
            "INT16": 128,
            "S32": 64,
            "S16": 128,
        },
        "cube_peak_ppc": 4096,
        "cube_roofline_multiplier": 384,
        "gm_bw": 1.8,
        "l2_bw": 8.0,
    },
    "Ascend910B1": {
        "num_cores": 48,
        "cacheline_size": 128,
        "vec_peak_ppc": {
            "FP32": 64,
            "MISC": 32,
            "FP16": 128,
            "INT32": 64,
            "INT16": 128,
            "S32": 64,
            "S16": 128,
        },
        "cube_peak_ppc": 4096,
        "cube_roofline_multiplier": 384,
        "gm_bw": 1.8,
        "l2_bw": 8.0,
    },
    "Ascend910B": {
        "num_cores": 48,
        "cacheline_size": 128,
        "vec_peak_ppc": {
            "FP32": 64,
            "MISC": 32,
            "FP16": 128,
            "INT32": 64,
            "INT16": 128,
            "S32": 64,
            "S16": 128,
        },
        "cube_peak_ppc": 4096,
        "cube_roofline_multiplier": 384,
        "gm_bw": 1.8,
        "l2_bw": 8.0,
    },
    "Ascend910A": {
        "num_cores": 32,
        "cacheline_size": 64,
        "vec_peak_ppc": {
            "FP32": 64,
            "MISC": 32,
            "FP16": 128,
            "INT32": 64,
            "INT16": 128,
            "S32": 64,
            "S16": 128,
        },
        "cube_peak_ppc": 4096,
        "cube_roofline_multiplier": 384,
        "gm_bw": 1.2,
        "l2_bw": 6.0,
    },
}
DEFAULT_CONFIG = {
    "num_cores": 48,
    "cacheline_size": 128,
    "vec_peak_ppc": {
        "FP32": 64,
        "MISC": 32,
        "FP16": 128,
        "INT32": 64,
        "INT16": 128,
        "S32": 64,
        "S16": 128,
    },
    "cube_peak_ppc": 4096,
    "cube_roofline_multiplier": 384,
    "gm_bw": 1.8,
    "l2_bw": 8.0,
}

DATA_TYPE_BASE_INFO = 5
DATA_TYPE_COMPUTE_LOAD_TABLE = 7
DATA_TYPE_MEMORY_TABLE = 9
DATA_TYPE_ROOFLINE = 13

VEC_INSTR_MAPPING = {
    "Vector FP32": "FP32",
    "Vector FP16": "FP16",
    "Vector INT32": "INT32",
    "Vector INT16": "INT16",
    "Vector S32": "S32",
    "Vector S16": "S16",
    "Vector Misc": "MISC",
}


def parse_bin_blocks(data):
    blocks = defaultdict(list)
    pos = 0
    while pos + 12 <= len(data):
        data_size = struct.unpack_from("<Q", data, pos)[0]
        data_type = data[pos + 8]
        padding_len = data[pos + 9]
        data_start = pos + 12
        if data_size > len(data):
            break
        data_end = data_start + data_size
        actual_end = data_end - padding_len
        blocks[data_type].append(data[data_start:actual_end])
        pos = data_end
    return blocks


def safe_float(val):
    if val is None or val == "NA" or val == "":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def parse_bin_file(bin_path):
    with open(bin_path, "rb") as f:
        data = f.read()
    blocks = parse_bin_blocks(data)

    base_info = {}
    if DATA_TYPE_BASE_INFO in blocks:
        j = json.loads(blocks[DATA_TYPE_BASE_INFO][0].decode("utf-8"))
        base_info = {
            "duration": float(j["duration"]),
            "freq": float(j["cur_freq"]),
            "soc": j.get("soc", ""),
            "block_dim": int(j.get("block_dim", 0)),
            "op_type": j.get("op_type", ""),
        }

    soc = base_info.get("soc", "")
    config = SOC_CONFIG.get(soc, DEFAULT_CONFIG)

    roofline_entries = []
    if DATA_TYPE_ROOFLINE in blocks:
        j = json.loads(blocks[DATA_TYPE_ROOFLINE][0].decode("utf-8"))
        for graph in j.get("multiple_rooflines", []):
            title = graph.get("title", "")
            for r in graph.get("rooflines", []):
                bw = float(r["bw"])
                ai = float(r["point"][0])
                perf = float(r["point"][1])
                computility = float(r["computility"])
                ratio = float(r["ratio"])
                bw_name = r.get("bw_name", "")
                computility_name = r.get("computility_name", "")
                roofline_entries.append(
                    {
                        "title": title,
                        "bw_name": bw_name,
                        "bw": bw,
                        "AI": ai,
                        "performance": perf,
                        "computility": computility,
                        "ratio": ratio,
                        "computility_name": computility_name,
                    }
                )

    return {
        "base_info": base_info,
        "roofline_entries": roofline_entries,
        "config": config,
    }


def compute_from_csvs(dir_path, config):
    op_basic_path = os.path.join(dir_path, "OpBasicInfo.csv")
    with open(op_basic_path) as f:
        row = next(csv.DictReader(f))
        duration = safe_float(row["Task Duration(us)"])
        freq = safe_float(row["Current Freq"])
        block_dim = int(safe_float(row["Block Dim"]))
        op_type = row.get("Op Type", "vector")

    au_path = os.path.join(dir_path, "ArithmeticUtilization.csv")
    with open(au_path) as f:
        au_rows = list(csv.DictReader(f))

    cube_rows = [r for r in au_rows if r["sub_block_id"].startswith("cube")]
    vec_rows = [r for r in au_rows if r["sub_block_id"].startswith("vector")]

    # ---- Cube fops ----
    cube_fops_csv = sum(safe_float(r.get("aic_cube_fops")) for r in cube_rows)
    cube_roofline_multiplier = config["cube_roofline_multiplier"]
    cube_fops = cube_fops_csv * cube_roofline_multiplier if cube_rows else 0

    # ---- Cube computility ----
    # cube_ratio (fp16_ratio, int8_ratio) already represents the proportion
    # For Cube, computility = cube_peak_ppc * freq * num_cores (proportion is 100% if active)
    cube_peak_ppc = config["cube_peak_ppc"]
    num_cores = config["num_cores"]

    sum_cube_cycles = sum(safe_float(r.get("aic_total_cycles")) for r in cube_rows)
    sum_cube_ratio = sum(safe_float(r.get("aic_cube_ratio")) for r in cube_rows)

    cube_computility = 0
    if cube_rows and sum_cube_cycles > 0:
        cube_computility = cube_peak_ppc * freq * 1e6 * num_cores

    # ---- Vector fops ----
    vec_fops = sum(safe_float(r.get("aiv_vec_fops")) for r in vec_rows)

    # ---- Vector computility ----
    sum_fp32_cycles = sum(
        safe_float(r.get("aiv_vec_fp32_ratio")) * safe_float(r.get("aiv_total_cycles"))
        for r in vec_rows
    )
    sum_misc_cycles = sum(
        safe_float(r.get("aiv_vec_misc_ratio")) * safe_float(r.get("aiv_total_cycles"))
        for r in vec_rows
    )
    sum_vec_cycles = sum(
        safe_float(r.get("aiv_vec_ratio")) * safe_float(r.get("aiv_total_cycles")) for r in vec_rows
    )

    fp32_prop = sum_fp32_cycles / sum_vec_cycles if sum_vec_cycles > 0 else 0
    misc_prop = sum_misc_cycles / sum_vec_cycles if sum_vec_cycles > 0 else 0

    vec_peak_ppc_weighted = 64 * fp32_prop + 32 * misc_prop
    vec_computility = vec_peak_ppc_weighted * freq * 1e6 * num_cores if vec_rows else 0

    # ---- Combined ----
    total_fops = cube_fops + vec_fops
    combined_computility = cube_computility + vec_computility

    # ---- total_bytes (L2 read miss * cacheline_size) ----
    l2_path = os.path.join(dir_path, "L2Cache.csv")
    with open(l2_path) as f:
        l2_rows = list(csv.DictReader(f))

    total_l2_read_miss = sum(
        safe_float(r.get("aiv_r0_read_cache_miss_allocate"))
        + safe_float(r.get("aiv_r1_read_cache_miss_allocate"))
        for r in l2_rows
    )
    total_l2_read_miss += sum(
        safe_float(r.get("aic_r0_read_cache_miss_allocate"))
        + safe_float(r.get("aic_r1_read_cache_miss_allocate"))
        for r in l2_rows
    )

    cacheline_size = config["cacheline_size"]
    total_bytes = total_l2_read_miss * cacheline_size

    # ---- Final metrics (GM Read + Write path) ----
    gm_bw = config["gm_bw"]
    performance = total_fops / (duration * 1e-6) if duration > 0 else 0
    ai = total_fops / total_bytes if total_bytes > 0 else 0
    perf_tops = performance / 1e12
    comp_tops = combined_computility / 1e12
    roofline_peak = min(ai * gm_bw, comp_tops)
    ratio = perf_tops / roofline_peak if roofline_peak > 0 else 0

    # ---- L2 Read + Write path ----
    l2_bw = config["l2_bw"]
    l2_peak = min(ai * l2_bw, comp_tops)
    l2_ratio = perf_tops / l2_peak if l2_peak > 0 else 0

    return {
        "op_type": op_type,
        "duration": duration,
        "freq": freq,
        "block_dim": block_dim,
        "cube_fops_csv": cube_fops_csv,
        "cube_fops": cube_fops,
        "cube_roofline_multiplier": cube_roofline_multiplier,
        "cube_peak_ppc": cube_peak_ppc,
        "cube_computility": cube_computility / 1e12,
        "sum_cube_cycles": sum_cube_cycles,
        "sum_cube_ratio": sum_cube_ratio,
        "vec_fops": vec_fops,
        "vec_peak_ppc_weighted": vec_peak_ppc_weighted,
        "fp32_prop": fp32_prop,
        "misc_prop": misc_prop,
        "vec_computility": vec_computility / 1e12,
        "sum_vec_cycles": sum_vec_cycles,
        "total_fops": total_fops,
        "combined_computility": comp_tops,
        "total_l2_read_miss": total_l2_read_miss,
        "total_bytes": total_bytes,
        "cacheline_size": cacheline_size,
        "num_cores": num_cores,
        "AI": ai,
        "performance": perf_tops,
        "computility": comp_tops,
        "bw": gm_bw,
        "l2_bw": l2_bw,
        "ratio": ratio,
        "l2_ratio": l2_ratio,
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <OPPROF_directory>")
        sys.exit(1)

    opprof_dir = sys.argv[1]
    bin_path = os.path.join(opprof_dir, "visualize_data.bin")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found")
        sys.exit(1)

    print("=" * 90)
    print(
        "  Roofline Metrics Comparison: visualize_data.bin (Method 1)  vs  CSV Formulas (Method 2)"
    )
    print("=" * 90)
    print(f"\n  OPPROF Directory: {opprof_dir}")

    bin_data = parse_bin_file(bin_path)
    base_info = bin_data["base_info"]
    config = bin_data["config"]
    op_type = base_info.get("op_type", "vector")

    print("\n[Base Info]")
    print(f"  SoC            : {base_info.get('soc', 'N/A')}")
    print(f"  Op Type        : {op_type}")
    print(f"  Task Duration  : {base_info.get('duration', 0):.9f} us")
    print(f"  AI Core Freq   : {base_info.get('freq', 0):.0f} MHz")
    print(f"  Block Dim      : {base_info.get('block_dim', 0)}")
    print(f"  Num Cores      : {config['num_cores']}")
    print(f"  Cacheline Size : {config['cacheline_size']} B")
    print(f"  Cube peak ppc  : {config['cube_peak_ppc']}")
    print(f"  Cube multiplier: {config['cube_roofline_multiplier']}x")

    # ---- Method 1: bin file ----
    print(f"\n{'-' * 90}")
    print("[Method 1] Extract from visualize_data.bin")
    print("-" * 90)
    print("\n  Roofline entries (pre-computed by msopprof):")
    header = f"  {'Title':<22} {'BW Name':<22} {'BW':>8} {'AI':>14} {'Perf':>12} {'Computility':>12} {'Ratio':>10}"
    print(header)
    print(f"  {'-' * 22} {'-' * 22} {'-' * 8} {'-' * 14} {'-' * 12} {'-' * 12} {'-' * 10}")
    for e in bin_data["roofline_entries"]:
        print(
            f"  {e['title']:<22} {e['bw_name']:<22} {e['bw']:>8.3f} {e['AI']:>14.6f} {e['performance']:>12.6f} {e['computility']:>12.6f} {e['ratio']:>10.6f}"
        )

    # ---- Method 2: CSV ----
    print(f"\n{'-' * 90}")
    print("[Method 2] Compute from CSV files using verified formulas")
    print("-" * 90)
    csv_m = compute_from_csvs(opprof_dir, config)

    print("\n  [Raw data from CSVs]")
    print("  OpBasicInfo.csv:")
    print(f"    Op Type                     : {csv_m['op_type']}")
    print(f"    Task Duration               : {csv_m['duration']:.9f} us")
    print(f"    AI Core Freq                : {csv_m['freq']:.0f} MHz")

    if csv_m["cube_fops_csv"] > 0:
        print("\n  ArithmeticUtilization.csv (Cube):")
        print(f"    sum(aic_cube_fops)          : {csv_m['cube_fops_csv']:.0f}")
        print(f"    cube_roofline_multiplier    : {csv_m['cube_roofline_multiplier']}x")
        print(f"    cube_fops (csv x multiplier): {csv_m['cube_fops']:.0f}")
        print(f"    cube_peak_ppc               : {csv_m['cube_peak_ppc']}")
        print(f"    cube_computility            : {csv_m['cube_computility']:.6f} TOps/s")
        print(f"      = {csv_m['cube_peak_ppc']} x {csv_m['freq']:.0f}MHz x {csv_m['num_cores']}")

    if csv_m["vec_fops"] > 0:
        print("\n  ArithmeticUtilization.csv (Vector):")
        print(f"    sum(aiv_vec_fops)           : {csv_m['vec_fops']:.0f}")
        print(
            f"    fp32 proportion             : {csv_m['fp32_prop']:.9f}  ({csv_m['fp32_prop'] * 100:.6f}%)"
        )
        print(
            f"    misc proportion             : {csv_m['misc_prop']:.9f}  ({csv_m['misc_prop'] * 100:.6f}%)"
        )
        print(f"    vec_peak_ppc (weighted)     : {csv_m['vec_peak_ppc_weighted']:.6f}")
        print(f"    vec_computility             : {csv_m['vec_computility']:.6f} TOps/s")

    print("\n  Combined:")
    print(f"    total_fops (cube + vec)     : {csv_m['total_fops']:.0f}")
    print(f"    combined_computility        : {csv_m['combined_computility']:.6f} TOps/s")

    print("\n  L2Cache.csv:")
    print(f"    total L2 read miss          : {csv_m['total_l2_read_miss']:.0f}")
    print(f"    total_bytes (miss x {csv_m['cacheline_size']})   : {csv_m['total_bytes']:.0f}")

    print("\n  [Computed metrics — GM Read + Write path]")
    print(f"    Bandwidth (theoretical)     : {csv_m['bw']:.3f} TB/s")
    print(f"    fops                        : {csv_m['total_fops']:.0f}")
    print(f"    total_bytes                 : {csv_m['total_bytes']:.0f}")
    print(f"    Arithmetic Intensity        : {csv_m['AI']:.9f} Ops/Byte")
    print(f"    Performance                 : {csv_m['performance']:.9f} TOps/s")
    print(f"    Computility                 : {csv_m['computility']:.9f} TOps/s")
    rl_peak = min(csv_m["AI"] * csv_m["bw"], csv_m["computility"])
    print(
        f"    Roofline peak               : min({csv_m['AI']:.6f}x{csv_m['bw']}, {csv_m['computility']:.6f}) = {rl_peak:.9f}"
    )
    print(f"    Performance Ratio           : {csv_m['ratio']:.9f}  ({csv_m['ratio'] * 100:.6f}%)")

    # ---- Comparison ----
    print(f"\n{'=' * 90}")
    print("[Comparison] Method 1 (bin) vs Method 2 (CSV)")
    print("=" * 90)

    # ---- Compare GM and L2 paths ----
    bin_gm_entries = [
        e
        for e in bin_data["roofline_entries"]
        if e.get("title") == "GM/L2" and "GM Read + Write" in e.get("bw_name", "")
    ]
    bin_l2_entries = [
        e
        for e in bin_data["roofline_entries"]
        if e.get("title") == "GM/L2" and "L2 Read + Write" in e.get("bw_name", "")
    ]

    if not bin_gm_entries:
        bin_gm_entries = [
            e
            for e in bin_data["roofline_entries"]
            if "GM" in e.get("bw_name", "")
            and ("GM/L2" in e.get("title", "") or "Memory Unit" in e.get("title", ""))
        ]

    # CSV metrics dict (GM path)
    csv_gm = {
        "bw": csv_m["bw"],
        "fops": csv_m["total_fops"],
        "total_bytes": csv_m["total_bytes"],
        "AI": csv_m["AI"],
        "performance": csv_m["performance"],
        "computility": csv_m["computility"],
        "ratio": csv_m["ratio"],
    }

    comparisons = []
    if bin_gm_entries:
        gm = bin_gm_entries[0]
        bin_fops = gm["performance"] * 1e12 * base_info["duration"] * 1e-6
        bin_total_bytes = bin_fops / gm["AI"] if gm["AI"] > 0 else 0
        comparisons.append(("GM Read + Write", gm, csv_gm, bin_fops, bin_total_bytes))

    if bin_l2_entries:
        l2 = bin_l2_entries[0]
        bin_fops_l2 = l2["performance"] * 1e12 * base_info["duration"] * 1e-6
        bin_total_bytes_l2 = bin_fops_l2 / l2["AI"] if l2["AI"] > 0 else 0
        # L2 total_bytes uses all L2 traffic (hit+miss), not just read misses
        # This cannot be reliably derived from CSV for mix kernels, so mark as N/A
        csv_l2 = {
            "bw": csv_m["l2_bw"],
            "fops": csv_m["total_fops"],
            "total_bytes": None,  # Cannot derive from CSV
            "AI": None,  # Depends on total_bytes
            "performance": csv_m["performance"],
            "computility": csv_m["computility"],
            "ratio": None,  # Depends on AI
        }
        comparisons.append(("L2 Read + Write", l2, csv_l2, bin_fops_l2, bin_total_bytes_l2))

    for _, bin_entry, csv_vals, bin_fops, bin_total_bytes in comparisons:
        print(f"\n  Path: {bin_entry['title']} / {bin_entry['bw_name']}")

        bin_rows = [
            ("Bandwidth (TB/s)", bin_entry["bw"]),
            ("fops", float(bin_fops)),
            ("Computility (TOps/s)", bin_entry["computility"]),
            ("Performance (TOps/s)", bin_entry["performance"]),
        ]
        if bin_total_bytes is not None and not isinstance(bin_total_bytes, str):
            bin_rows.insert(2, ("total_bytes", float(bin_total_bytes)))
            bin_rows.insert(4, ("Arithmetic Intensity", bin_entry["AI"]))
        bin_rows.append(("Performance Ratio", bin_entry["ratio"]))

        key_map = {
            "Bandwidth (TB/s)": "bw",
            "fops": "fops",
            "total_bytes": "total_bytes",
            "Arithmetic Intensity": "AI",
            "Performance (TOps/s)": "performance",
            "Computility (TOps/s)": "computility",
            "Performance Ratio": "ratio",
        }

        print(
            f"  {'Metric':<25} {'Bin (Method 1)':<22} {'CSV (Method 2)':<22} {'Match':<8} {'Rel.Diff':<12}"
        )
        print(f"  {'-' * 25} {'-' * 22} {'-' * 22} {'-' * 8} {'-' * 12}")
        all_match = True
        for name, bv in bin_rows:
            csv_key = key_map.get(name, "")
            cv = csv_vals.get(csv_key, None)
            if cv is None:
                print(f"  {name:<25} {bv:<22.9f} {'N/A (CSV)':<22} {'N/A':<8} {'N/A':<12}")
                continue
            rel = abs(bv - cv) / abs(bv) if abs(bv) > 0 else abs(bv - cv)
            match = "YES" if rel < 1e-3 else "NO"
            if match == "NO":
                all_match = False
            print(f"  {name:<25} {bv:<22.9f} {cv:<22.9f} {match:<8} {rel:<12.2e}")

        verdict = "ALL MATCH" if all_match else "MISMATCH"
        print(f"  -> {verdict}")

    # ---- All bin entries for reference ----
    print(f"\n{'-' * 90}")
    print("[Reference] All roofline entries from bin file")
    print("-" * 90)
    for e in bin_data["roofline_entries"]:
        print(
            f"  {e['title']:<22} / {e['bw_name']:<22}:  bw={e['bw']:.6f}  AI={e['AI']:.6f}  "
            f"Perf={e['performance']:.6f}  Comp={e['computility']:.6f}  ratio={e['ratio']:.6f}  ({e['computility_name']})"
        )

    print(f"\n{'=' * 90}")
    print("Done.")
    print("=" * 90)


if __name__ == "__main__":
    main()
