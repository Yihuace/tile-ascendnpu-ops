#!/usr/bin/env python3
"""Re-evaluate operator Ratio against MEASURED achievable peaks.

The framework's ``Ratio`` divides by a theoretical peak (bw = 1.75 TB/s,
Computility = 4.61 TOp/s for fp32). That denominator is physically unreachable,
so ``Ratio`` alone cannot answer "how much head-room is left".

This tool keeps the framework Ratio intact (for comparability with historical
reports) and adds a second column:

    达成率 = Ratio / 该 shape 档位（或该 dtype）的实测 Ratio 天花板

The ceilings come from ``tools/calibration_910B3.json``, which is produced by
probing this machine (see docs/op_performance_methodology.md §3).

Binding term is decided per workload, exactly as the framework does:

    bandwidth-bound  if AI * nominal_bw < Computility   -> tier ceiling (by element count)
    compute-bound    otherwise                          -> dtype ceiling

Usage
-----
    python3 tools/ratio_vs_peak.py --log <profile_run.log> [--label OP]
    python3 tools/ratio_vs_peak.py --batch          # the standard 24-op set
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "calibration_910B3.json")


def load_calibration(path: str = CALIB_PATH) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _elements(shape_text: str) -> int | None:
    """Parse a shape cell like '(4096, 4096)' / '[1048576]' / '[2, 8, 4096, 128]'."""
    if not shape_text:
        return None
    digits = []
    token = ""
    for char in shape_text:
        if char.isdigit():
            token += char
        else:
            if token:
                digits.append(int(token))
                token = ""
    if token:
        digits.append(int(token))
    if not digits:
        return None
    product = 1
    for value in digits:
        product *= value
    return product


def read_op_side(path: str) -> list[dict]:
    """Read the operator-side table (first section) of a profile_run log."""
    rows: list[dict] = []
    columns = None
    section = 0
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if "Ratio (%)" in cells:
                section += 1
                if section == 1:
                    def idx(name):
                        return cells.index(name) if name in cells else None

                    columns = {
                        "label": idx("Label") or 0,
                        "ratio": idx("Ratio (%)"),
                        "ai": idx("AI(Ops/Byte)"),
                        "computility": idx("Computility(TOps/s)"),
                        "perf": idx("Perf(TOps/s)"),
                        "dtype": idx("dtype"),
                        # families disagree on the name: elementwise uses "shape",
                        # clamp/pool use "input_shape", attention has no shape at all.
                        "shape": next(
                            (idx(name) for name in ("shape", "input_shape") if idx(name) is not None),
                            None,
                        ),
                        "latency": idx("Latency (us)"),
                    }
                continue
            if section != 1 or columns is None or set(line.strip()) <= set("|-: "):
                continue

            def cell(key):
                pos = columns[key]
                return cells[pos] if pos is not None and len(cells) > pos else ""

            ratio = _float(cell("ratio"))
            if not ratio:
                continue
            rows.append(
                {
                    "label": cell("label"),
                    "dtype": cell("dtype"),
                    "latency_us": _float(cell("latency")),
                    "ratio": ratio,
                    "ai": _float(cell("ai")),
                    "computility": _float(cell("computility")),
                    "perf": _float(cell("perf")),
                    "elements": _elements(cell("shape")),
                }
            )
    return rows


def _dtype_key(dtype: str) -> str:
    token = (dtype or "").lower()
    if "bf" in token or "bfloat" in token:
        return "bfloat16"
    if "16" in token:
        return "float16"
    if "32" in token:
        return "float32"
    return "float32"


def ceiling_for(row: dict, calib: dict) -> tuple[float | None, str, float | None]:
    """Return (reach_pct, binding, ceiling_abs).

    Head-room is derived from ABSOLUTE achievable peaks, never from a nominal
    percentage: the framework's ``Computility`` is weighted per kernel, so a
    fixed nominal would make 达成率 meaningless (it produced >100% values).

    Binding is decided in three classes, because the denominator differs:
      * ``bandwidth``      -- AI*bw < Computility and Computility is vector-scale
      * ``vector-compute`` -- vector peak dominates (Computility <= ~50 TOp/s)
      * ``cube-compute``   -- Computility is Cube-scale (~300 TOp/s). There is
                              NO Cube-side calibration yet, so 达成率 is N/A
                              rather than a fabricated number.
    """
    nominal_bw = calib["nominal"]["gm_bandwidth_TBs"]
    ai = row.get("ai")
    computility = row.get("computility")
    perf = row.get("perf")

    if ai is None or computility is None or perf is None:
        return None, "unknown", None

    cube_scale = 50.0        # vector peaks are <= ~9.2 TOp/s; Cube peaks ~300
    if computility > cube_scale:
        return None, "cube-compute", None

    bandwidth_term = ai * nominal_bw
    if bandwidth_term < computility:
        elements = row.get("elements")
        if elements is None:
            return None, "bandwidth", None
        ceiling = None
        for tier in calib["achievable_bandwidth_TBs"]["tiers"]:
            limit = tier["max_elements"]
            if limit is None or elements <= limit:
                ceiling = tier["achievable_TBs"]
                break
        if ceiling is None:
            return None, "bandwidth", None
        achieved_bw = perf / ai       # TOp/s ÷ Ops/Byte = TB/s
        return achieved_bw / ceiling * 100, "bandwidth", ceiling

    ceiling = calib["achievable_compute_TOpss"].get(_dtype_key(row["dtype"]))
    if not ceiling:
        return None, "vector-compute", None
    return perf / ceiling * 100, "vector-compute", ceiling


def evaluate(rows: list[dict], calib: dict) -> list[dict]:
    out = []
    for row in rows:
        reach, binding, ceiling_abs = ceiling_for(row, calib)
        out.append({**row, "ceiling_abs": ceiling_abs, "binding": binding, "reach": reach})
    return out


def print_table(label: str, rows: list[dict], calib: dict) -> None:
    threshold = calib["pass_threshold_pct"]
    print(f"\n=== {label} ({len(rows)} workloads) ===")
    print(f"  {'workload':<34}{'dtype':<10}{'Ratio%':>8}{'绑定':>7}"
          f"{'实测':>8}{'可达':>8}{'达成率%':>9}")
    for row in rows:
        unit = "TB/s" if row["binding"] == "bandwidth" else "TOp/s"
        if row["binding"] == "bandwidth" and row.get("ai"):
            actual = row["perf"] / row["ai"]
        else:
            actual = row.get("perf")
        actual_text = f"{actual:.2f}" if actual else "-"
        ceiling_text = f"{row['ceiling_abs']:.2f}" if row["ceiling_abs"] else "-"
        reach_text = f"{row['reach']:.1f}" if row["reach"] is not None else "-"
        print(f"  {row['label'][:32]:<34}{row['dtype'][:9]:<10}"
              f"{row['ratio']:>8.2f}{row['binding']:>7}"
              f"{actual_text:>8}{ceiling_text:>8}{reach_text:>9}")

    ratios = [r["ratio"] for r in rows]
    reaches = [r["reach"] for r in rows if r["reach"] is not None]
    print(f"  算术平均 Ratio {statistics.mean(ratios):.2f}%"
          f"   (阈值 {threshold:.0f}% -> {'达标' if statistics.mean(ratios) >= threshold else '未达标'})")
    if reaches:
        print(f"  算术平均 达成率 {statistics.mean(reaches):.1f}%"
              f"   最低 {min(reaches):.1f}%   最高 {max(reaches):.1f}%"
              f"   (实测 / 可达，>100% 说明该流量比超出了标定探针)")


BATCH = [
    ("BitwiseNotFwdOp", None),
    ("ErfFwdOp", "profile_run_msprof_20261029_000919.log"),
    ("Expm1FwdOp", "/tmp/bench_rest/expm1.profile.log"),
    ("EluFwdOp", "/tmp/bench_rest/elu.profile.log"),
    ("LeakyReluFwdOp", "/tmp/bench_rest/leaky_relu.profile.log"),
    ("AlibiFwdOp", "/tmp/bench_rest/alibi.profile.log"),
    ("HardtanhFwdOp", "/tmp/bench_rest/hardtanh.profile.log"),
    ("ClampFwdOp", "/tmp/bench_rest/clamp.profile.log"),
    ("LerpTensorFwdOp", "/tmp/bench_rest/lerp_tensor.profile.log"),
    ("FFTC2COp", "/tmp/bench_rest/fft.profile.log"),
    ("ArgmaxFwdOp", "/tmp/bench_rest/argmax.profile.log"),
    ("DropoutOp", "/tmp/bench_rest/dropout.profile.log"),
    ("RopeNeoxOp", "/tmp/bench_zero/rope_neox.profile.log"),
    ("RopeLongRopeOp", "/tmp/bench_zero/rope_longrope.profile.log"),
    ("RopeLlama31Op", "/tmp/bench_zero/rope_llama31.profile.log"),
    ("DSA-Decode", "/tmp/bench_zero/deepseek_dsa_decode.profile.log"),
    ("FP8LightningIndexer", "/tmp/bench_zero/fp8_lightning_indexer.profile.log"),
    ("MHA-DecodePaged", "/tmp/bench_zero/mha_decode_paged.profile.log"),
    ("EngramDecode", "/tmp/bench_zero/engram_decode.profile.log"),
    ("EngramGateConvFwd", "/tmp/bench_zero/engram_gate_conv_fwd.profile.log"),
    ("EngramGateConvBwd", "/tmp/bench_zero/engram_gate_conv_bwd.profile.log"),
    ("GQA-SlidingWindow", "/tmp/bench_zero/grouped_query_attention_sliding_window.profile.log"),
    ("GQA-SWVarlen", "/tmp/bench_zero/grouped_query_attention_sliding_window_varlen.profile.log"),
    ("GQA-Bwd", "/tmp/bench_zero/grouped_query_attention_bwd.profile.log"),
]


def bitwise_not_rows(calib: dict) -> list[dict]:
    """BitwiseNotFwdOp has no profile_run log; rebuild its rows from the bins."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tileops.benchmark.msprof_roofline import parse_bin_file  # noqa: PLC0415

    spec = [
        ("prof_bn", "op_i32_1m", "smoke-1m", "int32", 1048576),
        ("prof_bn2", "i64_1m", "smoke-1m", "int64", 1048576),
        ("prof_bn", "op_i32_16m", "elementwise-16m", "int32", 16777216),
        ("prof_bn2", "i64_16m", "elementwise-16m", "int64", 16777216),
        ("prof_bn", "op_i32_64m", "elementwise-64m", "int32", 67108864),
        ("prof_bn2", "i32_256m", "elementwise-256m", "int32", 268435456),
    ]
    nominal_bw = calib["nominal"]["gm_bandwidth_TBs"]
    rows = []
    for base, tag, label, dtype, elements in spec:
        import glob

        pattern = os.path.expanduser(f"~/{base}/{tag}/OPPROF_*")
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            continue
        sub = os.path.join(dirs[-1], "main", "5")
        if not os.path.isdir(sub):
            candidates = sorted(glob.glob(os.path.join(dirs[-1], "main", "*")))
            if not candidates:
                continue
            sub = candidates[0]
        bin_path = os.path.join(sub, "visualize_data.bin")
        if not os.path.exists(bin_path):
            continue
        parsed = parse_bin_file(bin_path)
        entry = next(
            (e for e in parsed["roofline_entries"]
             if e["title"] == "GM/L2" and "GM Read + Write" in e["bw_name"]),
            None,
        )
        if not entry:
            continue
        rows.append(
            {
                "label": label,
                "dtype": dtype,
                "latency_us": parsed["base_info"]["duration"],
                "ratio": entry["ratio"] * 100,
                "ai": entry["AI"],
                "computility": entry["computility"],
                "perf": entry["performance"],
                "elements": elements,
                "binding_hint": "bandwidth" if entry["AI"] * nominal_bw < entry["computility"] else "compute",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", help="profile_run log to evaluate")
    parser.add_argument("--label", default="")
    parser.add_argument("--batch", action="store_true", help="evaluate the standard 24-op set")
    parser.add_argument("--calib", default=CALIB_PATH)
    args = parser.parse_args()

    calib = load_calibration(args.calib)

    if args.log:
        rows = evaluate(read_op_side(args.log), calib)
        print_table(args.label or os.path.basename(args.log), rows, calib)
        return 0

    if not args.batch:
        parser.error("pass --log or --batch")

    summary = []
    for name, path in BATCH:
        if path is None:
            raw = bitwise_not_rows(calib)
        elif os.path.exists(path):
            raw = read_op_side(path)
        else:
            print(f"{name}: log missing ({path})", file=sys.stderr)
            continue
        if not raw:
            continue
        rows = evaluate(raw, calib)
        print_table(name, rows, calib)
        ratios = [r["ratio"] for r in rows]
        reaches = [r["reach"] for r in rows if r["reach"] is not None]
        summary.append((name, len(rows), statistics.mean(ratios),
                        statistics.mean(reaches) if reaches else None))

    print("\n\n" + "=" * 78)
    print("汇总（按 Ratio 降序）")
    print(f"{'算子':<24}{'用例':>4}{'Ratio均%':>10}{'达成率均%':>11}{'Ratio判定':>11}")
    for name, count, ratio, reach in sorted(summary, key=lambda x: -x[2]):
        verdict = "达标" if ratio >= calib["pass_threshold_pct"] else "-"
        reach_text = f"{reach:.1f}" if reach is not None else "-"
        print(f"{name:<24}{count:>4}{ratio:>10.2f}{reach_text:>11}{verdict:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
