"""Interleaved A/B driver for the RopeLlama31Op r3 adoption sync.

Runs `msprof op` alternately on the previous adopted version (r2_contig,
archived in `perf_opt/history/`) and the adopted canonical r3 winner
(`perf_opt/_make_rope_neox_1d.py`), N rounds x 2 workloads, and reports the
per-candidate median of the per-run `Task Duration(us)` medians.

Kernel-only `msprof op Task Duration(us)` is the only metric; no NPU-event or
end-to-end timing is collected.  Profiles land under
`perf_opt/profiles/{baseline,final}/round*/...`, logs under `perf_opt/logs/r3_adopt/`.

Usage:
  python perf_opt/_ab_adopt.py [--rounds 5]
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_profile_mod():
    spec = importlib.util.spec_from_file_location(
        "rope_llama31_profile_adopt", os.path.join(HERE, "_profile_adopt.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CANDIDATES = (
    (
        "r2_contig",
        "history/_make_rope_neox_1d_r2_contig.py",
        "796840bcee4d5f696be72c787667a1fa41c5f5da1cee8fe7cc0331c275272897",
        "baseline/round0",
    ),
    (
        "r3_adopt",
        "_make_rope_neox_1d.py",
        "c6098298b94beb664c4ff670f33a2b15926838a207a9a6b74b5a1bfb1a504f4a",
        "final/round4",
    ),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    import hashlib

    prof = _load_profile_mod()
    results = {}
    for run in range(1, args.rounds + 1):
        for cand_id, rel_artifact, expect_sha, prefix in CANDIDATES:
            artifact = os.path.join(HERE, rel_artifact)
            with open(artifact, "rb") as handle:
                sha = hashlib.sha256(handle.read()).hexdigest()
            if sha != expect_sha:
                print(f"[abort] {cand_id} sha256 {sha} != {expect_sha}", flush=True)
                return 2
            root = os.path.join(
                HERE, "profiles", *prefix.split("/"),
                f"{cand_id}_{sha[:8]}_run{run}",
            )
            for workload_id, spec in prof.WORKLOADS.items():
                print(f"[msprof] run{run} {cand_id} {workload_id} ...", flush=True)
                res = prof.run_msprof(
                    artifact, workload_id, spec, root,
                    f"{cand_id}_{sha[:8]}_run{run}",
                )
                if "error" in res:
                    print(json.dumps(res, ensure_ascii=False), flush=True)
                    return 1
                results.setdefault((cand_id, workload_id), []).append(
                    res["duration_us"]
                )
                print(
                    f"  -> {res['duration_us']} us (n_launch={res['n_launch']})",
                    flush=True,
                )

    summary = {"rounds": args.rounds, "unit": "msprof op Task Duration(us)", "runs": {}, "median": {}}
    for key, values in results.items():
        cand_id, workload_id = key
        summary["runs"].setdefault(f"{cand_id}|{workload_id}", []).extend(values)
        summary["median"][f"{cand_id}|{workload_id}"] = round(
            statistics.median(values), 4
        )
    out = os.path.join(HERE, "logs", "r3_adopt", "ab_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[summary] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
