#!/usr/bin/env python3
"""Per-shape optimization summary chart for BitwiseNotFwdOp.

Reads the stage tables produced by ``pipe_util_report.py``
(``perf_opt/pipe_util_summary.csv``) + ``perf_opt/workload_inventory.json`` and
renders a two-panel comparison:

  * left  : Task Duration, baseline (Stage 3) vs perf_opt (Stage 4), per shape,
            annotated with the absolute values and the % speed-up.
  * right : achieved effective bandwidth, baseline vs perf_opt, per shape,
            with the machine's single-die HBM theory (1.64 TB/s) as a reference
            line and the % of that theory annotated.

Effective bandwidth = declared traffic / measured latency, where the declared
traffic is the manifest roofline ``2 * N * elem_bytes`` (read once + write once).

  NOTE: shapes whose traffic fits in L2 (the 16M cases here) can show
  achieved > HBM theory -- that is an L2-residency effect, not a real HBM
  exceedance; those bars are tagged "(L2)".

Usage:
  cd <...>/bitwise_not_kernel/perf_opt
  python scripts/perf_summary_chart.py
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)
HBM_TBPS = 1.64          # single-die (device 0) HBM theoretical
EB = {"int32": 4, "int64": 8, "uint8": 1, "int8": 1, "int16": 2,
      "bool": 1, "float16": 2, "float32": 4, "bfloat16": 2}


def _shape_str(shape):
    n = 1
    for s in shape:
        n *= s
    return f"[{'×'.join(map(str, shape))}] N={n/1e6:.1f}M"


def main():
    inv = {w["workload_id"]: w
           for w in json.load(open(os.path.join(PERF_OPT, "workload_inventory.json")))["workloads"]}
    rows = list(csv.DictReader(open(os.path.join(PERF_OPT, "pipe_util_summary.csv"))))

    base, fin = {}, {}
    for r in rows:
        wl, ph, dur = r["workload"], r["phase"], r["duration_us"]
        if wl not in inv or not dur:
            continue
        d = float(dur)
        if ph == "baseline":
            base[wl] = d
        elif ph == "final" or (ph == "merged" and wl not in fin):
            fin[wl] = d

    wls = [w for w in ("elementwise-16m-int32", "elementwise-16m-int64",
                       "elementwise-64m-int32", "elementwise-256m-int32")
           if w in base and w in fin]

    labels, bdur, fdur, bbw, fbw, pct156, l2flag = [], [], [], [], [], [], []
    for wl in wls:
        iv = inv[wl]
        n = 1
        for s in iv["shape"]:
            n *= s
        byt = 2 * n * EB[iv["dtype"]]
        ab = byt * 1e-12 / (base[wl] * 1e-6)     # TB/s
        af = byt * 1e-12 / (fin[wl] * 1e-6)
        labels.append(f"{wl.replace('elementwise-', '')}\n{_shape_str(iv['shape'])}")
        bdur.append(base[wl]); fdur.append(fin[wl])
        bbw.append(ab); fbw.append(af)
        pct156.append(100.0 * af / HBM_TBPS)
        l2flag.append(af > HBM_TBPS or ab > HBM_TBPS)

    # ---- markdown summary --------------------------------------------------
    print("## BitwiseNotFwdOp 优化前后（本机 Ascend910_9362 单die，HBM 理论 1.64 TB/s）\n")
    print("| shape | dtype | baseline(us) | final(us) | baseline BW(TB/s) | final BW(TB/s) | 带宽提升% | final ÷ 1.64 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for i, wl in enumerate(wls):
        iv = inv[wl]
        tag = " (L2)" if l2flag[i] else ""
        bwgain = 100.0 * (fbw[i] - bbw[i]) / bbw[i]
        print(f"| {_shape_str(iv['shape'])} | {iv['dtype']} | {bdur[i]:.2f} | {fdur[i]:.2f} | "
              f"{bbw[i]:.3f} | {fbw[i]:.3f} | **{bwgain:+.1f}%** | "
              f"**{pct156[i]:.1f}%**{tag} |")

    # ---- chart -------------------------------------------------------------
    x = list(range(len(wls)))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    # left: duration
    ax1.bar([i - w / 2 for i in x], bdur, w, label="baseline (Stage 3)", color="#9aa0a6")
    ax1.bar([i + w / 2 for i in x], fdur, w, label="perf_opt (Stage 4)", color="#1a73e8")
    for i in x:
        ax1.annotate(f"{bdur[i]:.0f}", (i - w / 2, bdur[i]), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, color="#5f6368")
        ax1.annotate(f"{fdur[i]:.0f}\n-{100*(bdur[i]-fdur[i])/bdur[i]:.0f}%", (i + w / 2, fdur[i]),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                     color="#1a73e8", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Task Duration (us)   [lower = faster]")
    ax1.set_title("① Optimization result: baseline vs perf_opt (per shape)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    # right: achieved BW vs HBM theory
    ax2.bar([i - w / 2 for i in x], bbw, w, label="baseline", color="#9aa0a6")
    ax2.bar([i + w / 2 for i in x], fbw, w, label="perf_opt", color="#1a73e8")
    ax2.axhline(HBM_TBPS, ls="--", lw=1.5, color="#d93025")
    ax2.text(len(wls) - 0.5, HBM_TBPS + 0.02, f"single-die HBM theory = {HBM_TBPS} TB/s",
             color="#d93025", fontsize=8, ha="right")
    for i in x:
        tag = " (L2)" if l2flag[i] else ""
        bgain = 100.0 * (fbw[i] - bbw[i]) / bbw[i]
        ax2.annotate(f"{fbw[i]:.2f}\n+{bgain:.0f}% BW\n({pct156[i]:.0f}% of 1.64{tag})",
                     (i + w / 2, fbw[i]), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5, color="#1a73e8", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("achieved effective bandwidth (TB/s)")
    ax2.set_title("② achieved BW vs machine theory (1.64 TB/s) + BW gain vs baseline")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.16, wspace=0.23)
    out = os.path.join(PERF_OPT, "optimization_summary.png")
    fig.savefig(out, dpi=130)
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
