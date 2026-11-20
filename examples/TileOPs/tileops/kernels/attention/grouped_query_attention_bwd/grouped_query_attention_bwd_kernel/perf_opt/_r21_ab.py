"""Round 21 interleaved 3-run A/B (msprof op Task Duration, same session).

Arms: current best (`_gqa_bwd_kernel.py`, 3-pass BM=BN=64) vs
`_gqa_bwd_kernel_r21_bn128_hpds.py` (BN=128 for sdw/mul + workspace slot,
pass-2 at a 64-row UB-capped sub-tile).

Each pair launches the two arms back to back and alternates the order
(pair1 base->cand, pair2 cand->base, pair3 base->cand) so that any drift in the
device state is shared.  The per-workload comparator is the median of the three
aggregate 3-pass durations per arm (sum of per-pass `Task Duration` medians).

Usage: python _r21_ab.py [--pairs 3] [--iters 15]
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "_gqa_bwd_kernel.py"
CAND = "_gqa_bwd_kernel_r21_bn128_hpds.py"

WORKLOADS = [
    ("llama-3.1-8b-short-float16", 4, 512, 32, 8, 128, "float16"),
    ("llama-3.1-8b-long-float16", 2, 2048, 32, 8, 128, "float16"),
]


def run(impl, w, B, S, H, Hk, D, dtype, out, iters):
    cmd = [sys.executable, os.path.join(HERE, "run_split_measure.py"),
           "--impl", os.path.join(HERE, impl), "--B", str(B), "--S", str(S),
           "--H", str(H), "--H_kv", str(Hk), "--D", str(D), "--dtype", dtype,
           "--iters", str(iters), "--warm-up", "5", "--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    line = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not line:
        return {"status": "error", "rc": proc.returncode,
                "tail": (proc.stdout + proc.stderr)[-800:]}
    return json.loads(line[-1])


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--workloads", default="")
    args = ap.parse_args()
    wl = WORKLOADS
    if args.workloads:
        keep = set(args.workloads.split(","))
        wl = [w for w in WORKLOADS if w[0] in keep]

    summary = {}
    for wid, B, S, H, Hk, D, dtype in wl:
        rec = {"base": [], "cand": [], "per_pass": {}}
        for k in range(args.pairs):
            order = [("base", BASE), ("cand", CAND)]
            if k % 2 == 1:
                order.reverse()
            for arm, impl in order:
                out = os.path.join(HERE, "profiles", "round21", "ab",
                                   f"{wid}_{arm}_r{k + 1}")
                res = run(impl, wid, B, S, H, Hk, D, dtype, out, args.iters)
                if res.get("status") != "ok":
                    rec[f"{arm}_error"] = res
                    print(f"[AB] {wid} {arm} r{k + 1} ERROR {res}", flush=True)
                    continue
                rec[arm].append(res["duration_us"])
                rec["per_pass"].setdefault(arm, []).append(
                    {name: v["median_us"] for name, v in res["per_kernel"].items()
                     if name.startswith("_gqa_bwd")})
                print(f"[AB] {wid} {arm} r{k + 1} n_kernels={res['n_kernels_per_call']} "
                      f"total={res['duration_us']:.2f}us", flush=True)
        if rec["base"] and rec["cand"]:
            mb, mc = median(rec["base"]), median(rec["cand"])
            rec.update({"median_base_us": mb, "median_cand_us": mc,
                        "ratio_cand_over_base": round(mc / mb, 4),
                        "delta_pct": round(100.0 * (mc - mb) / mb, 2),
                        "all_pairs_same_direction": all(c < b for c, b in
                                                        zip(rec["cand"], rec["base"]))})
            print(f"[AB] {wid} MEDIAN base={mb:.2f} cand={mc:.2f} "
                  f"delta={rec['delta_pct']:+.2f}% same_dir={rec['all_pairs_same_direction']}",
                  flush=True)
        summary[wid] = rec
    path = os.path.join(HERE, "logs", "round21", "ab_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print("[AB] summary ->", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
