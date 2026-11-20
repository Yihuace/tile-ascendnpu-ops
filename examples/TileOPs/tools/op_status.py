#!/usr/bin/env python3
"""Assess a TileLang NPU op's optimization status from msprof artifacts.

The manual 7-layer checklist collapses into two mechanised questions:

  1. Facts per shape   -- latency, roofline Ratio, dominant pipeline,
                          core balance, binding term.
  2. Verdict by rules  -- no per-op calibrated ceiling required.

Two data sources are supported:

  --opprof DIR   an msprof ``OPPROF_*`` directory (full facts: needs the
                 PipeUtilization/OpBasicInfo CSVs next to visualize_data.bin)
  --log FILE     a ``profile_run_msprof_*.log`` (latency + Ratio only; the
                 pipeline CSVs are gone because msprof profiles under /tmp)

Verdict rules
-------------
dominant = the largest of {mte2, vec, scalar} ratio (share of AIV time)

  SATURATED  dominant >= 0.85 and main-core spread <= 0.30
             -> the bottleneck engine is already pegged; structural tuning
                has little left. Stop.
  HEADROOM   dominant <  0.65
             -> no engine is saturated, so time is going somewhere else
                (launch/wave quantisation/scalar). Worth investigating.
  CHECK      anything else -> needs a human look (e.g. 0.65..0.85 boundary).

"spread" ignores cores whose aiv_time is below half the maximum: tail-wave
leftovers and partial records show up there and would otherwise dominate the
statistic.

Examples
--------
    python3 tools/op_status.py --opprof ~/prof_bn/op_i32_16m/OPPROF_xxx \\
        --label BitwiseNotFwdOp/16m/int32

    python3 tools/op_status.py --log /tmp/bench_zero/rope_neox.profile.log \\
        --label RopeNeoxOp
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys

SATURATED_RATIO = 0.85
HEADROOM_RATIO = 0.65
SPREAD_LIMIT = 0.30


def _f(value):
    """Parse a CSV cell to float, or None when it is NA/empty."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_rows(directory: str, stem: str) -> list[dict] | None:
    matches = glob.glob(os.path.join(directory, f"{stem}_*.csv"))
    if not matches:
        return None
    with open(matches[0], newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def read_opprof(directory: str) -> dict:
    """Collect facts from one OPPROF_* directory (uses launch 5 when present)."""
    sub = os.path.join(directory, "main", "5")
    if not os.path.isdir(sub):
        subs = sorted(glob.glob(os.path.join(directory, "main", "*")))
        if not subs:
            raise SystemExit(f"no main/ sub-directory under {directory}")
        sub = subs[0]

    facts: dict = {"source": os.path.basename(directory)}

    basic = _csv_rows(sub, "OpBasicInfo")
    if basic:
        row = basic[0]
        facts["duration_us"] = _f(row.get("Task Duration(us)"))
        facts["block_dim"] = row.get("Block Dim")
        facts["freq"] = row.get("Current Freq")

    pipe = _csv_rows(sub, "PipeUtilization") or []
    for key, column in (
        ("mte2", "aiv_mte2_ratio"),
        ("mte3", "aiv_mte3_ratio"),
        ("vec", "aiv_vec_ratio"),
        ("scalar", "aiv_scalar_ratio"),
        ("vec_wait", "aiv_vec_wait_ratio"),
        ("mte2_wait", "aiv_mte2_wait_ratio"),
    ):
        facts[key] = _mean([v for v in (_f(r.get(column)) for r in pipe) if v is not None])

    times = [v for v in (_f(r.get("aiv_time(us)")) for r in pipe) if v]
    if times:
        peak = max(times)
        main = [t for t in times if t >= peak * 0.5]
        facts["cores"] = len(times)
        facts["cores_main"] = len(main)
        facts["core_time_min"] = min(main)
        facts["core_time_max"] = peak
        facts["spread"] = (peak - min(main)) / statistics.mean(main) if main else None

    l2 = _csv_rows(sub, "L2Cache") or []
    if l2:
        reads = sum(
            v
            for v in (
                _f(r.get("aiv_r0_read_cache_miss_allocate")) for r in l2
            )
            if v
        ) + sum(
            v
            for v in (
                _f(r.get("aiv_r1_read_cache_miss_allocate")) for r in l2
            )
            if v
        )
        facts["dram_read_cachelines"] = reads or None

    bin_path = os.path.join(sub, "visualize_data.bin")
    if os.path.exists(bin_path):
        try:
            from tileops.benchmark.msprof_roofline import parse_bin_file

            parsed = parse_bin_file(bin_path)
            entry = next(
                (
                    e
                    for e in parsed["roofline_entries"]
                    if e["title"] == "GM/L2" and "GM Read + Write" in e["bw_name"]
                ),
                None,
            )
            if entry:
                facts["ratio"] = entry["ratio"] * 100
                facts["ai"] = entry["AI"]
                facts["computility"] = entry["computility"]
                facts["binding"] = (
                    "bandwidth"
                    if entry["AI"] * 1.75 < entry["computility"]
                    else "compute"
                )
        except Exception as exc:  # pragma: no cover - optional dependency
            facts["bin_error"] = str(exc)[:80]

    return facts


def read_log(path: str) -> list[dict]:
    """Read the operator side (first table section) of a profile_run log."""
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
                    columns = {
                        "label": cells.index("Label") if "Label" in cells else 0,
                        "ratio": cells.index("Ratio (%)"),
                        "latency": cells.index("Latency (us)") if "Latency (us)" in cells else None,
                        "dtype": cells.index("dtype") if "dtype" in cells else None,
                    }
                continue
            if section != 1 or columns is None or set(line.strip()) <= set("|-: "):
                continue
            ratio = _f(cells[columns["ratio"]]) if len(cells) > columns["ratio"] else None
            if not ratio:
                continue
            rows.append(
                {
                    "label": cells[columns["label"]],
                    "dtype": cells[columns["dtype"]] if columns["dtype"] is not None else "",
                    "latency_us": (
                        _f(cells[columns["latency"]])
                        if columns["latency"] is not None and len(cells) > columns["latency"]
                        else None
                    ),
                    "ratio": ratio,
                }
            )
    return rows


def verdict(facts: dict) -> tuple[str, str]:
    """Return (verdict, reason) from the collected facts."""
    ratios = {
        "mte2": facts.get("mte2"),
        "vec": facts.get("vec"),
        "scalar": facts.get("scalar"),
    }
    ratios = {k: v for k, v in ratios.items() if v is not None}
    if not ratios:
        return "UNKNOWN", "no pipeline ratios (log-only source)"
    engine = max(ratios, key=ratios.get)
    share = ratios[engine]
    spread = facts.get("spread")

    if share >= SATURATED_RATIO and (spread is None or spread <= SPREAD_LIMIT):
        return "SATURATED", f"{engine} engine pegged at {share:.2f}; little structural head-room"
    if share < HEADROOM_RATIO:
        return "HEADROOM", f"no engine saturated (max {engine} {share:.2f}); time goes elsewhere"
    return "CHECK", f"borderline: {engine} at {share:.2f}"


def _fmt(value, pattern="{:.2f}"):
    return pattern.format(value) if value is not None else "-"


def report_shape(label: str, facts: dict) -> None:
    name, _reason = verdict(facts)
    print(f"\n=== {label} ===")
    print(f"  latency       : {_fmt(facts.get('duration_us'))} us"
          f"   block_dim {facts.get('block_dim', '-')}   freq {facts.get('freq', '-')}")
    print(f"  roofline Ratio: {_fmt(facts.get('ratio'))} %"
          f"   AI {_fmt(facts.get('ai'), '{:.4f}')}"
          f"   binding {facts.get('binding', '-')}")
    print(f"  pipelines     : mte2 {_fmt(facts.get('mte2'))}"
          f"  vec {_fmt(facts.get('vec'))}"
          f"  scalar {_fmt(facts.get('scalar'))}")
    if facts.get("spread") is not None:
        print(f"  core balance  : {facts.get('cores_main')}/{facts.get('cores')} main cores"
              f"  {_fmt(facts.get('core_time_min'), '{:.1f}')}..{_fmt(facts.get('core_time_max'), '{:.1f}')} us"
              f"  spread {_fmt(facts.get('spread') * 100, '{:.0f}')}%")
    if facts.get("dram_read_cachelines"):
        print(f"  DRAM reads    : {facts['dram_read_cachelines']:.0f} cachelines")
    print(f"  VERDICT       : {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opprof", help="msprof OPPROF_* directory (full facts)")
    parser.add_argument("--log", help="profile_run_msprof_*.log (latency + Ratio)")
    parser.add_argument("--label", default="", help="operator / shape label")
    args = parser.parse_args()

    if not args.opprof and not args.log:
        parser.error("one of --opprof / --log is required")

    if args.opprof:
        facts = read_opprof(args.opprof)
        label = args.label or os.path.basename(os.path.dirname(args.opprof.rstrip("/")))
        report_shape(label, facts)
        name, reason = verdict(facts)
        print(f"  reason        : {reason}")
        return 0

    rows = read_log(args.log)
    if not rows:
        print(f"no operator-side rows in {args.log}", file=sys.stderr)
        return 1
    label = args.label or os.path.basename(args.log).replace(".profile.log", "")
    values = [r["ratio"] for r in rows]
    print(f"\n=== {label} ({len(rows)} workloads, log-only source) ===")
    print(f"  {'workload':<44}{'dtype':<11}{'latency us':>12}{'Ratio %':>9}")
    for row in rows:
        print(f"  {row['label']:<44}{row['dtype']:<11}"
              f"{_fmt(row['latency_us']):>12}{row['ratio']:>9.2f}")
    print(f"  arithmetic mean {statistics.mean(values):.2f}%"
          f"   min {min(values):.2f}%   max {max(values):.2f}%")
    print("  VERDICT       : log-only source (no pipeline data); judge on Ratio / mean only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
