#!/usr/bin/env python3
"""Stage-4 msprof PipeUtilization aggregator + bottleneck/improvement report.

What it does
------------
Walks ``<perf_opt>/profiles/**`` and, for every profiled case
(``phase/round/candidate/workload``), aggregates the raw msprof CSVs:

  * ``duration_us``  <- ``OpBasicInfo.csv``      ("Task Duration(us)")
  * ``vec%``         <- ``PipeUtilization.csv``  ("aiv_vec_ratio")   算力单元
  * ``scalar%``      <- ``PipeUtilization.csv``  ("aiv_scalar_ratio")标量单元
  * ``mte2%``        <- ``PipeUtilization.csv``  ("aiv_mte2_ratio")  GM->UB 搬运
  * ``mte3%``        <- ``PipeUtilization.csv``  ("aiv_mte3_ratio")  UB->GM 搬运
  * ``mte2_bw``      <- ``PipeUtilization.csv``  ("aiv_mte2_active_bw(GB/s)")

(On Ascend vector cores MTE1 is Cube-only, so it is absent here by design.)

Outputs
-------
  * ``perf_opt/pipe_util_summary.csv``   -- one row per case
  * ``perf_opt/optimization_report.png`` -- matrix layout:
        rows = shapes (vertical), cols = metrics (horizontal),
        + a duration bar panel (baseline vs final).               (图证)
  * markdown tables on stdout            -- per-case table + per-workload
                                            baseline->final delta

Usage
-----
  cd <...>/bitwise_not_kernel/perf_opt
  python scripts/pipe_util_report.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)
PROFILES = os.path.join(PERF_OPT, "profiles")
HASH = re.compile(r"_[0-9a-f]{8}$")
PHASE_ORDER = {"baseline": 0, "candidate": 1, "merged": 2, "final": 3, "_smoke_test": 9}


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _strip_hash(name: str) -> str:
    return HASH.sub("", name)


def load_inventory():
    """workload_id -> {shape, dtype} (from workload_inventory.json)."""
    path = os.path.join(PERF_OPT, "workload_inventory.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return {w["workload_id"]: {"shape": w.get("shape"), "dtype": w.get("dtype")}
            for w in data.get("workloads", [])}


def _shape_str(shape):
    """[4096, 4096] -> '[4096×4096] N=16.8M'."""
    if not shape:
        return "-"
    n = 1
    for s in shape:
        n *= s
    if n >= 1_000_000:
        nn = f"{n / 1e6:.1f}M"
    elif n >= 1_000:
        nn = f"{n / 1e3:.0f}K"
    else:
        nn = str(n)
    return f"[{'×'.join(str(s) for s in shape)}] N={nn}"


def _pipe(dur, ratio, nd=1):
    """忙时(µs)/占比(%)  -- e.g. ``84.4/90.4%`` (= dur_us x ratio%)."""
    if dur is None or ratio is None:
        return "-"
    return f"{dur * ratio / 100.0:.{nd}f}/{ratio:.{nd}f}%"


def _pipe_pair(db, rb, df, rf):
    """''prev' -> 'new'  as 忙时/占比 (shows both absolute drop and ratio)."""
    return f"{_pipe(db, rb)} → {_pipe(df, rf)}"


def collect():
    """-> { (phase, round, candidate, workload): {metrics} }"""
    cases: dict[tuple, dict] = {}
    for opprof in glob.glob(os.path.join(PROFILES, "**", "OPPROF_*"), recursive=True):
        case_dir = os.path.dirname(opprof)
        rel = os.path.relpath(case_dir, PROFILES).split(os.sep)
        if len(rel) >= 4:
            phase, rnd, cand, wl = rel[0], rel[1], _strip_hash(rel[2]), _strip_hash(rel[3])
        elif len(rel) == 3:
            phase, rnd, cand, wl = rel[0], rel[1], "-", _strip_hash(rel[2])
        elif len(rel) == 2:
            phase, rnd, cand, wl = rel[0], "-", "-", _strip_hash(rel[1])
        else:
            phase, rnd, cand, wl = "(root)", "-", "-", _strip_hash(rel[-1])
        key = (phase, rnd, cand, wl)

        vec, sca, m2, m3, bw, dur = [], [], [], [], [], []
        for c in glob.glob(os.path.join(opprof, "**", "PipeUtilization_*.csv"), recursive=True):
            for r in _rows(c):
                vec.append(_fnum(r.get("aiv_vec_ratio")))
                sca.append(_fnum(r.get("aiv_scalar_ratio")))
                m2.append(_fnum(r.get("aiv_mte2_ratio")))
                m3.append(_fnum(r.get("aiv_mte3_ratio")))
                bw.append(_fnum(r.get("aiv_mte2_active_bw(GB/s)")))
        for c in glob.glob(os.path.join(opprof, "**", "OpBasicInfo_*.csv"), recursive=True):
            for r in _rows(c):
                dur.append(_fnum(r.get("Task Duration(us)")))

        pct = lambda xs: (100.0 * _mean(xs)) if _mean(xs) is not None else None  # noqa: E731
        cases[key] = {
            "duration_us": _mean(dur),
            "vec_pct": pct(vec),
            "scalar_pct": pct(sca),
            "mte2_pct": pct(m2),
            "mte3_pct": pct(m3),
            "mte2_bw": _mean(bw),
        }
    return cases


def _fmt(v, nd=2, suffix=""):
    return "-" if v is None else f"{v:.{nd}f}{suffix}"


def _dominant(row):
    cand = {k: row[k] for k in ("vec_pct", "scalar_pct", "mte2_pct", "mte3_pct") if row[k] is not None}
    if not cand:
        return "-"
    name = {"vec_pct": "Vector", "scalar_pct": "Scalar", "mte2_pct": "MTE2(GM->UB)", "mte3_pct": "MTE3(UB->GM)"}
    return name[max(cand, key=cand.get)]


def main():
    cases = collect()
    inv = load_inventory()
    if not cases:
        print("no profiles found under", PROFILES)
        return
    order = sorted(cases, key=lambda k: (PHASE_ORDER.get(k[0], 5), str(k[1]), k[2], k[3]))

    # ---- 1. one table per stage (phase / round / candidate) ----------------
    base, final = {}, {}
    for k, m in cases.items():
        if k[0] == "baseline" and m["duration_us"]:
            base[k[3]] = (m["duration_us"], m)
        if m["duration_us"] and k[0] in ("final", "merged"):
            if k[3] not in final or k[0] == "final":
                final[k[3]] = (m["duration_us"], m)

    groups: dict[tuple, list] = {}
    for k in order:
        if k[3] not in inv:            # only manifest-inventory workloads
            continue
        groups.setdefault((k[0], k[1], k[2]), []).append(k)

    for (phase, rnd, cand), ks in groups.items():
        parts = [phase] + ([rnd] if rnd not in (None, "-") else []) + \
                ([cand] if cand not in (None, "-") else [])
        print(f"\n### {' · '.join(parts)}\n")
        print("| workload | shape | dur(us) | Vector 忙时/占比 | Scalar 忙时/占比 | MTE2 忙时/占比 | MTE3 忙时/占比 | 主导单元 | MTE2 bw(GB/s) |")
        print("|---|---|---:|---|---|---|---|---|---:|")
        for k in sorted(ks, key=lambda x: x[3]):
            m = cases[k]
            wl = k[3]
            print(
                f"| {wl} | {_shape_str(inv.get(wl, {}).get('shape'))} | {_fmt(m['duration_us'])} | "
                f"{_pipe(m['duration_us'], m['vec_pct'])} | {_pipe(m['duration_us'], m['scalar_pct'])} | "
                f"{_pipe(m['duration_us'], m['mte2_pct'])} | {_pipe(m['duration_us'], m['mte3_pct'])} | "
                f"**{_dominant(m)}** | {_fmt(m['mte2_bw'],1)} |"
            )

    deltas = []
    for wl in sorted(set(base) & set(final)):
        b, bm = base[wl]
        f, fm = final[wl]
        deltas.append((wl, b, f, 100.0 * (b - f) / b))

    # ---- 3. write CSV ------------------------------------------------------
    out_csv = os.path.join(PERF_OPT, "pipe_util_summary.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "round", "candidate", "workload", "duration_us",
                    "vec_us", "vec_pct", "scalar_us", "scalar_pct",
                    "mte2_us", "mte2_pct", "mte3_us", "mte3_pct", "mte2_bw_gbs", "dominant"])
        for k in order:
            m = cases[k]
            d = m["duration_us"]

            def _t(p):
                return None if (d is None or m[p] is None) else d * m[p] / 100.0

            w.writerow([*k, d, _t("vec_pct"), m["vec_pct"], _t("scalar_pct"), m["scalar_pct"],
                        _t("mte2_pct"), m["mte2_pct"], _t("mte3_pct"), m["mte3_pct"],
                        m["mte2_bw"], _dominant(m)])
    print(f"\n[ok] wrote {out_csv}")

    # ---- 4. chart: rows = shapes (vertical), cols = metrics (horizontal) ---
    if not deltas:
        print("[warn] no baseline/final overlap; skip chart")
        return
    wls = [d[0] for d in deltas]
    short = [f"{w.replace('elementwise-', '')}\n{_shape_str(inv.get(w, {}).get('shape'))}" for w in wls]
    metrics = [("vec_pct", "Vector\n(compute)"), ("scalar_pct", "Scalar"),
               ("mte2_pct", "MTE2\n(GM->UB)"), ("mte3_pct", "MTE3\n(UB->GM)")]

    def mat(which):
        src = base if which == "baseline" else final
        return [[(src[w][1][k] or 0.0) for k, _ in metrics] for w in wls]

    fig = plt.figure(figsize=(16, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.28)

    for i, (which, tag) in enumerate((("baseline", "baseline (Stage 3)"),
                                      ("final", "perf_opt (Stage 4)"))):
        ax = fig.add_subplot(gs[0, i])
        data = mat(which)
        ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([n for _, n in metrics], fontsize=8)
        ax.set_yticks(range(len(wls)))
        ax.set_yticklabels(short if i == 0 else [""] * len(wls), fontsize=9)
        ax.set_title(f"{tag}\npipe busy %  (rows=shape / cols=metric)", fontsize=10)
        for r in range(len(wls)):
            for c in range(len(metrics)):
                v = data[r][c]
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="black" if v < 60 else "white")

    ax = fig.add_subplot(gs[0, 2])
    y = list(range(len(wls)))
    h = 0.36
    ax.barh([i + h / 2 for i in y], [base[w][0] for w in wls], h,
            label="baseline", color="#9aa0a6")
    ax.barh([i - h / 2 for i in y], [final[w][0] for w in wls], h,
            label="perf_opt", color="#1a73e8")
    xs = [max(base[w][0], final[w][0]) for w in wls]
    for i, d in enumerate(deltas):
        ax.text(xs[i] * 1.02, i - h / 2, f"{d[2]:.0f}us  (-{d[3]:.0f}%)",
                va="center", fontsize=8, color="#1a73e8", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(wls))
    ax.invert_yaxis()
    ax.set_xlim(0, max(xs) * 1.35)
    ax.set_xlabel("Task Duration (us)")
    ax.set_title("baseline vs perf_opt  duration (us)\n(rows=shape, aligned with left panels)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.14)
    out_png = os.path.join(PERF_OPT, "optimization_report.png")
    fig.savefig(out_png, dpi=130)
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()
