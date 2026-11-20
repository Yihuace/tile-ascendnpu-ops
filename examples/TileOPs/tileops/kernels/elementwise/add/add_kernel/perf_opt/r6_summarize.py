#!/usr/bin/env python3
"""R6 A/B summariser: join two ``profile_run_msprof_*.json`` reports (base/final)
and print the per-workload table used by ``REPORT_ADD_BCAST.md``.

Columns
    base_us / final_us   msprof ``op`` Task Duration (the only latency metric)
    delta                final vs base (%)
    ratio_h_*            the harness ``Ratio(%)``: msprof GM **read** bytes /
                         (t * 1.75 TB/s).  Drops whenever redundant reads are
                         removed, so it is NOT comparable across byte models.
    ratio_m_*            ``eval_roofline()`` bytes / (t * PEAK) with one byte
                         model per arm (base 3N, final 2N+B) -> comparable.
                         PEAK is calibrated so the base arm reproduces the
                         harness ratio (R5: 3N / (69.17% * 14.05us) = 2.59 TB/s).

    python3 r6_summarize.py --base BASE.json --final FINAL.json
"""

import argparse
import json
import os
import sys
from math import prod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))
from tileops.kernels.elementwise.add import plan_broadcast  # noqa: E402

PEAK_TBS = 2.59  # calibrated on the R5 base arm; see module docstring


def load(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    out = {}
    for rec in data.get("records", []):
        params = rec.get("params", {})
        key = (rec.get("label"), str(params.get("dtype")))
        out[key] = {
            "us": rec["result"]["latency_us"],
            "ratio_h": rec["result"].get("Ratio(%)"),
            "in_shape": tuple(params.get("input_shape") or ()),
            "other_shape": tuple(params.get("other_shape") or ()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--final", required=True)
    args = ap.parse_args()
    base, final = load(args.base), load(args.final)

    hdr = (
        "| workload | dtype | base µs | final µs | Δ | ratio_h base | ratio_h final "
        "| ratio_m base | ratio_m final |"
    )
    print(hdr)
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for key in sorted(base, key=lambda k: (k[0], k[1])):
        if key not in final:
            continue
        b, f = base[key], final[key]
        label, dtype = key
        es = 4 if dtype == "float32" else 2
        n = prod(b["in_shape"]) if b["in_shape"] else 1
        blen = prod(b["other_shape"]) if b["other_shape"] else 1
        buy = 3 * n * es
        # The final arm's byte model follows the real dispatch decision: the
        # reuse plan when it matches, else the flat path (3N).
        plan = plan_broadcast(b["in_shape"], b["other_shape"], es)
        fuy = 3 * n * es if plan is None else (2 * n + blen) * es
        delta = (f["us"] - b["us"]) / b["us"] * 100
        bm = buy / (b["us"] * 1e-6) / PEAK_TBS / 1e12 * 100
        fm = fuy / (f["us"] * 1e-6) / PEAK_TBS / 1e12 * 100
        rows.append((label, dtype, b["us"], f["us"], delta, b["ratio_h"], f["ratio_h"], bm, fm))
        print(
            f"| {label} | {dtype} | {b['us']:.3f} | {f['us']:.3f} | {delta:+.1f}% "
            f"| {b['ratio_h']:.1f} | {f['ratio_h']:.1f} | {bm:.1f} | {fm:.1f} |"
        )
    tot_b = sum(r[2] for r in rows)
    tot_f = sum(r[3] for r in rows)
    print(f"\ntotal: base {tot_b:.3f} us, final {tot_f:.3f} us, Δ {(tot_f-tot_b)/tot_b*100:+.2f}%")


if __name__ == "__main__":
    main()
