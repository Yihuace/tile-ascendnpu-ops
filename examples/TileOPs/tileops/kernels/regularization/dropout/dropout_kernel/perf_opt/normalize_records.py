#!/usr/bin/env python3
"""Normalize ``perf_records.jsonl`` to the signal-registry section 5 contract.

The first collection pass (this same task, same session) wrote rows with
locally-invented field names (``task_duration_us`` / ``l0_result`` /
``raw_profile_dir`` / ``phase=rN``).  Those rows are migrated **in place** to
the contract (`duration_us` / `l0_pass` / `msprof_raw_path` / legal ``phase``
enum) with every measured value preserved; the pre-normalization file is kept
verbatim at ``logs/r0/perf_records_preschema.jsonl`` for audit.  No measurement
is dropped, added or edited by this step.
"""

import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "perf_records.jsonl")
FROZEN = os.path.join(HERE, "logs", "r0", "perf_records_preschema.jsonl")

KERNEL_ID = ("examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/"
             "perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main")

CAND_FILE = {
    "baseline": "../_make_dropout_kernel.py",
    "u5_diet": "_u5_diet.py",
    "u6_vclamp": "_u6_vclamp.py",
    "c1_sbulk": "_make_dropout_kernel_round2_staticbulk.py",
    "u7_sb_diet": "_u7_sb_diet.py",
    "hashC": "_v_hashC.py",
    "f16native": "_w_f16native.py",
    "seedfold": "_w_seedfold.py",
    "nat_seedfold": "_w_native_seedfold.py",
    "x_final": "_x_final.py",
    "final": "_make_dropout_kernel.py",
}
WL = {"W1": "tokens-1k-hidden-4k-fp16-float16",
      "W2": "tokens-1k-hidden-10k-bf16-bfloat16",
      "W3": "tokens-1k-hidden-4k-fp32-float32"}
PHASE = {"r0": "baseline", "r1": "candidate", "baseline": "baseline", "r2": "candidate", "r3": "candidate",
         "r4": "candidate", "r5": "merged", "final": "final"}
ROUND = {"r0": 0, "r1": 1, "baseline": 0, "r2": 2, "r3": 3, "r4": 4, "r5": 5, "final": 5}


def sha256(rel_path):
    with open(os.path.join(HERE, rel_path), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def ts_from_raw(raw_dir):
    m = re.search(r"OPPROF_(\d{8})(\d{6})", raw_dir)
    if not m:
        return "2026-10-17T00:00:00Z"
    d, t = m.groups()
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}Z"


def main():
    rows = [json.loads(line) for line in open(SRC)]
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    if not os.path.exists(FROZEN):
        with open(SRC) as a, open(FROZEN, "w") as b:
            b.write(a.read())

    out = []
    for r in rows:
        cand = r["candidate_id"]
        rel = CAND_FILE[cand]
        rd = r.get("raw_profile_dir", "")
        row = {
            "round": ROUND[r["round"]],
            "candidate_id": cand,
            "parent_id": r["parent_id"],
            "kernel_id": KERNEL_ID,
            "workload_id": WL[r["workload_id"]],
            "phase": PHASE[r["round"]],
            "artifact_path": rel,
            "artifact_sha256": sha256(rel),
            "duration_us": r["task_duration_us"],
            "l0_pass": r["l0_result"] == "pass",
            "msprof_raw_path": rd,
            "timestamp": ts_from_raw(rd),
            # diagnostics (non-contract, kept for the opt_log comparison tables)
            "block_size": r["block_size"],
            "captured_op_name": r["captured_op_name"],
            "launch_samples": r["launch_samples"],
            "aiv_vec_ratio": r.get("aiv_vec_ratio"),
            "aiv_mte2_ratio": r.get("aiv_mte2_ratio"),
            "aiv_mte3_ratio": r.get("aiv_mte3_ratio"),
            "aiv_scalar_ratio": r.get("aiv_scalar_ratio"),
            "aiv_mte2_time_us": r.get("aiv_mte2_time_us"),
            "aiv_mte3_time_us": r.get("aiv_mte3_time_us"),
            "profile_status": r["profile_status"],
            "command": r["command"],
        }
        out.append(row)
    with open(SRC, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"normalized {len(out)} rows -> {SRC}")
    print(f"pre-normalization copy: {FROZEN}")


if __name__ == "__main__":
    main()
