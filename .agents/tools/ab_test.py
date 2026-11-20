"""ab_test.py -- interleaved A/B multi-run measurement protocol (T-2).

Implements the BP_run_state_bimodality countermeasure: same-kernel cross-run
bimodality (+-3~5%) makes single msprof runs unreliable for <5% candidate
differences. This tool runs an interleaved A/B/A/B... sequence (>= 3 pairs,
alternating execution order per pair to break slow drift), extracts the
per-run Task Duration median from msprof OpBasicInfo CSVs, and reports the
merged median difference plus an exact paired sign test.

Usage:
  python3 .agents/tools/ab_test.py \
      --a-cmd "python bench.py --impl baseline --dtype float16 --N 16777216 --check" \
      --b-cmd "python bench.py --impl final    --dtype float16 --N 16777216 --check" \
      --workdir perf_opt/ab_20260910 [--pairs 3] [--launch-count 15] \
      [--kernel-name main] [--warm-up 5] [--noise 0.03] [--timeout 900]

Each arm command is a bench entry (it must launch the target kernel); the
tool wraps it in `msprof op` itself. Exit code 0 on success (verdict printed
as JSON), 2 on usage/environment error, 1 on measurement failure.
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

DEFAULT_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)


def run_msprof(
    cmd: str,
    out_dir: str,
    kernel: str,
    launch_count: int,
    warm_up: int,
    timeout: int,
    log_path: str,
) -> list:
    """Run one msprof op measurement; return the list of Task Durations."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    msprof_prefix = (
        f"msprof op --kernel-name={kernel} --output={out_dir} "
        f"--launch-count={launch_count} --warm-up={warm_up} --dump=off "
        f"--aic-metrics={DEFAULT_METRICS}"
    )
    shell_cmd = f"{msprof_prefix} {cmd}"
    with open(log_path, "w") as log:
        log.write(f"# {shell_cmd}\n")
        log.flush()
        proc = subprocess.run(
            shell_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log.write(proc.stdout)
        log.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"msprof run failed (rc={proc.returncode}), see {log_path}")
    rows = []
    for path in sorted(
        glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    ):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    rows.append(float(row["Task Duration(us)"]))
    if not rows:
        raise RuntimeError(f"no OpBasicInfo rows under {out_dir}")
    return rows


def sign_test_paired(diffs: list) -> float:
    """Exact two-sided paired sign test p-value over non-zero diffs."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    p = 2.0 * tail / (2**n)
    return min(1.0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--a-cmd", required=True, help="baseline arm bench command (shell line)"
    )
    ap.add_argument(
        "--b-cmd", required=True, help="candidate arm bench command (shell line)"
    )
    ap.add_argument(
        "--workdir", default="perf_opt/ab_test", help="output dir for profiles and logs"
    )
    ap.add_argument(
        "--pairs", type=int, default=3, help="number of A/B pairs (>= 3 recommended)"
    )
    ap.add_argument("--launch-count", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--kernel-name", default="main")
    ap.add_argument(
        "--noise",
        type=float,
        default=0.03,
        help="noise threshold for the tie/adopt verdict",
    )
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    if shutil.which("msprof") is None:
        print(json.dumps({"error": "msprof not found in PATH"}))
        return 2

    runs = {"A": [], "B": []}
    pair_diffs = []
    try:
        for i in range(args.pairs):
            # alternate which arm runs first within each pair (drift guard)
            order = ("A", "B") if i % 2 == 0 else ("B", "A")
            medians = {}
            for arm in order:
                cmd = args.a_cmd if arm == "A" else args.b_cmd
                out_dir = os.path.join(args.workdir, f"pair{i + 1}{arm}")
                log_path = os.path.join(args.workdir, "logs", f"pair{i + 1}{arm}.log")
                durs = run_msprof(
                    cmd,
                    out_dir,
                    args.kernel_name,
                    args.launch_count,
                    args.warm_up,
                    args.timeout,
                    log_path,
                )
                med = statistics.median(durs)
                medians[arm] = med
                runs[arm].append(med)
                print(
                    f"pair{i + 1} {arm}: median={med:.3f}us (n={len(durs)})",
                    file=sys.stderr,
                )
            pair_diffs.append(medians["B"] - medians["A"])
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(json.dumps({"error": str(e)}))
        return 1

    med_a = statistics.median(runs["A"])
    med_b = statistics.median(runs["B"])
    rel = (med_a - med_b) / med_a if med_a else 0.0
    p_value = sign_test_paired(pair_diffs)
    # consistent direction = every pair agrees with the merged median sign
    consistent = (
        all((d > 0) == (sum(pair_diffs) > 0) and d != 0 for d in pair_diffs)
        if pair_diffs
        else False
    )
    # For small n the exact sign test cannot reach p<0.1 (n=3 floor is
    # 0.25), so direction consistency is the deciding criterion there
    # (documented protocol: merged median + stable pairing order); the
    # p-value gates the verdict only once n >= 5.
    significance = consistent and (len(pair_diffs) < 5 or p_value < 0.1)
    if abs(rel) < args.noise:
        verdict = "tie"
    elif rel > 0 and significance:
        verdict = "B_better"
    elif rel < 0 and significance:
        verdict = "A_better"
    else:
        verdict = "inconclusive"

    result = {
        "protocol": "interleaved A/B multi-run (BP_run_state_bimodality)",
        "pairs": args.pairs,
        "launch_count": args.launch_count,
        "a_medians_us": runs["A"],
        "b_medians_us": runs["B"],
        "a_merged_median_us": round(med_a, 3),
        "b_merged_median_us": round(med_b, 3),
        "b_rel_change": round(-rel, 4),
        "pair_diffs_us": [round(d, 3) for d in pair_diffs],
        "sign_test_p": round(p_value, 4),
        "direction_consistent": consistent,
        "noise_threshold": args.noise,
        "verdict": verdict,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
