"""Stage 4 batch runner for TopkSelectorOp (T-4: script runs the experiments).

One invocation = one phase (baseline / candidate / final) over all tune workloads.
For each workload it runs `msprof op` twice -- once for the count kernel, once for
the select kernel -- and accounts the op latency as the number of count launches
the delivered algorithm needs times the measured count launch:

    op_us = n_count_launches * count_us + select_us

`n_count_launches` comes from the candidate's own bench output (`--mode count`
prints it): the Stage 3 host-bisection design needs ROUNDS (=40) launches, an
in-kernel-bisection candidate needs 1.

Ratio (user-specified口径): bandwidth-at-peak = manifest roofline bytes / latency
/ 1.8 TB/s, with `bytes = 4*M*N + 4*M*k` (attention_indexing.yaml).

Usage (from perf_opt/):
    python run_experiments.py --phase baseline --branch ../_topk_selector_kernel.py
    python run_experiments.py --phase candidate --round 1 \
        --branch _topk_selector_kernel_round1_inbranch.py --candidate-id r1_inbranch
    python run_experiments.py --phase final  --branch _topk_selector_kernel.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_RECORDS = os.path.join(HERE, "perf_records.jsonl")
WORKLOAD_INVENTORY = os.path.join(HERE, "workload_inventory.json")
MSPROF = os.environ.get("MSPROF_BIN", "/usr/local/Ascend/cann-9.0.0/bin/msprof")

KERNEL_ID = (
    "tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernel/"
    "_topk_selector_kernel.py::count+select::main"
)
MSPROF_METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,L2Cache,ResourceConflictRatio"
LAUNCH_COUNT = 15
WARM_UP = 5
TIMEOUT_S = int(os.environ.get("S4_TIMEOUT_S", "2400"))
PEAK_BW_TB_S = 1.8  # SOC_CONFIG["Ascend910B2C"].gm_bw in tileops/benchmark/msprof_roofline.py

DIRECT_CMD = [
    sys.executable,
    os.path.join(HERE, "bench.py"),
    "--impl",
    "{impl}",
    "--case",
    "{workload_id}",
    "--mode",
    "{mode}",
]


# ---------------------------------------------------------------- environment
def child_env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.setdefault("TILELANG_ASCEND_MODE", "Developer")
    return env


def safe_component(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------- inventory
def load_inventory() -> dict:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        return json.load(handle)


def tune_workloads() -> list[dict]:
    inv = load_inventory()
    return [item for item in inv["workloads"] if item.get("kind") == "tune"]


def smoke_workloads() -> list[dict]:
    inv = load_inventory()
    return [item for item in inv["workloads"] if item.get("kind") == "smoke"]


def roofline_ratio(workload: dict, duration_us: float) -> float | None:
    """bytes / latency / 1.8 TB/s, in percent (user-specified Stage 4口径)."""
    if not duration_us or duration_us <= 0:
        return None
    m = int(workload["params"]["m"])
    n = int(workload["params"]["n"])
    k = int(workload["params"]["k"])
    nbytes = 4 * m * n + 4 * m * k
    tb_s = nbytes / (duration_us * 1e-6) / 1e12
    return round(tb_s / PEAK_BW_TB_S * 100.0, 4)


# --------------------------------------------------------------- L0 regression
def run_l0(branch_file: str) -> tuple[bool, str]:
    """Run the candidate's own L0+L1 suite; PASS iff it reports PASS and no FAIL.

    The Stage 3 harness prints `[tag] ... -> PASS|FAIL` and does not exit nonzero
    on failure, so the verdict is parsed from stdout, not from the return code.
    """
    proc = subprocess.run(
        [sys.executable, branch_file, "--level", "all"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=child_env(),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    passes = out.count("-> PASS")
    fails = out.count("-> FAIL")
    ok = proc.returncode == 0 and fails == 0 and passes > 0
    tail = "\n".join(out.strip().splitlines()[-6:])
    return ok, f"rc={proc.returncode} pass={passes} fail={fails}\n{tail}"


# ---------------------------------------------------------------------- msprof
def run_msprof(branch_file: str, workload_id: str, mode: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "posix":
        os.chmod(out_dir, 0o700)
    cmd = [
        MSPROF,
        "op",
        f"--kernel-name=main",
        f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}",
        f"--warm-up={WARM_UP}",
        "--dump=off",
        f"--aic-metrics={MSPROF_METRICS}",
        *[part.format(impl=branch_file, workload_id=workload_id, mode=mode) for part in DIRECT_CMD],
    ]
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = os.path.join("logs", safe_component(out_dir) + ".log")
    with open(os.path.join(HERE, log), "w") as handle:
        handle.write(f"# {shlex.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=child_env())
        handle.write((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode != 0:
        return {"error": f"msprof rc={proc.returncode}", "log": log}

    durs: list[float] = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    try:
                        durs.append(float(row["Task Duration(us)"]))
                    except ValueError:
                        pass
    if not durs:
        return {"error": "no OpBasicInfo rows", "log": log}

    n_launches = None
    for line in reversed((proc.stdout or "").strip().splitlines()):
        if line.startswith("{"):
            try:
                n_launches = int(json.loads(line).get("n_launches_per_op"))
            except (ValueError, TypeError):
                pass
            break

    return {
        "duration_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "n_launches_per_op": n_launches,
        "msprof_raw_path": os.path.relpath(out_dir, HERE),
        "log": log,
    }


# --------------------------------------------------------------------- records
def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def recorded(phase: str, candidate_id: str) -> set[str]:
    if not os.path.isfile(PERF_RECORDS):
        return set()
    seen = set()
    with open(PERF_RECORDS) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == phase and row.get("candidate_id") == candidate_id:
                seen.add(row.get("workload_id"))
    return seen


def latest_phase_map(phase: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not os.path.isfile(PERF_RECORDS):
        return out
    with open(PERF_RECORDS) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == phase:
                out[row["workload_id"]] = row
    return out


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["baseline", "candidate", "final"])
    ap.add_argument("--branch", required=True, help="kernel file under measurement")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--candidate-id", default=None)
    ap.add_argument("--only", action="append", default=None, help="restrict to these workload_ids")
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-measure even if a row exists")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    candidate_id = args.candidate_id or ("baseline" if args.phase == "baseline" else f"r{args.round}")
    branch = os.path.abspath(os.path.join(HERE, args.branch)) if not os.path.isabs(args.branch) else args.branch
    if not os.path.isfile(branch):
        raise SystemExit(f"no such branch file: {branch}")
    sha = file_sha256(branch)

    l0_pass = True
    l0_detail = "skipped"
    if not args.skip_l0:
        t0 = time.time()
        l0_pass, l0_detail = run_l0(branch)
        # same gate applies to every row of this phase
        print(f"[l0] {os.path.basename(branch)} -> {'PASS' if l0_pass else 'FAIL'} ({time.time()-t0:.0f}s)")
        print("     " + l0_detail.replace("\n", "\n     "))

    already = recorded(args.phase, candidate_id)
    rows = []
    for wl in tune_workloads():
        wid = wl["workload_id"]
        if args.only and wid not in args.only:
            continue
        if wid in already and not args.force:
            print(f"[skip] {wid}: phase={args.phase} candidate={candidate_id} already recorded")
            continue

        base_dir = os.path.join(HERE, "profiles", args.phase, f"round{args.round}", f"{candidate_id}_{sha[:8]}", safe_component(wid))
        t0 = time.time()
        count = run_msprof(branch, wid, "count", os.path.join(base_dir, "count"))
        if "error" in count:
            print(f"  {wid}: count msprof FAILED: {count['error']} (log {count.get('log')})")
            rows.append({"workload_id": wid, "status": "msprof_fail", "error": count["error"]})
            continue
        sel = run_msprof(branch, wid, "select", os.path.join(base_dir, "select"))
        if "error" in sel:
            print(f"  {wid}: select msprof FAILED: {sel['error']} (log {sel.get('log')})")
            rows.append({"workload_id": wid, "status": "msprof_fail", "error": sel["error"]})
            continue

        n_count = count.get("n_launches_per_op") or 1
        op_us = round(n_count * count["duration_us"] + sel["duration_us"], 3)
        ratio = roofline_ratio(wl, op_us)
        record = {
            "round": args.round,
            "candidate_id": candidate_id,
            "parent_id": None if args.phase == "baseline" else "baseline",
            "kernel_id": KERNEL_ID,
            "workload_id": wid,
            "phase": args.phase,
            "artifact_path": os.path.relpath(branch, HERE),
            "artifact_sha256": sha,
            "duration_us": op_us,
            "l0_pass": bool(l0_pass),
            "msprof_raw_path": count["msprof_raw_path"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count_us": count["duration_us"],
            "select_us": sel["duration_us"],
            "n_count_launches": n_count,
            "ratio_percent": ratio,
            "m": wl["params"]["m"],
            "n": wl["params"]["n"],
            "k": wl["params"]["k"],
        }
        append_record(record)
        rows.append({**record, "status": "ok"})
        print(
            f"  {wid}: count {count['duration_us']:.3f} us x{n_count} + select {sel['duration_us']:.3f} us"
            f" = {op_us:.3f} us  ratio={ratio}%  ({time.time()-t0:.0f}s)"
        )

    if args.phase == "final" and not args.skip_smoke:
        smoke = smoke_workloads()
        if smoke:
            ok, detail = run_l0(branch)
            print(f"[smoke/final] L0+L1 precision on {os.path.basename(branch)} -> {'PASS' if ok else 'FAIL'}")
            print("     " + detail.replace("\n", "\n     "))

    base = latest_phase_map("baseline")
    print("\n| candidate | workload_id | l0 | op_us | vs baseline | ratio % | status |")
    print("|---|---|---|---:|---:|---:|---|")
    for row in rows:
        wid = row["workload_id"]
        ref = (base.get(wid) or {}).get("duration_us")
        gain = ""
        if ref and row.get("duration_us"):
            gain = f"{(row['duration_us'] / ref - 1) * 100:+.1f}%"
        print(
            f"| {candidate_id} | {wid} | {'PASS' if row.get('l0_pass') else '-'} | "
            f"{row.get('duration_us', '-')} | {gain or '-'} | {row.get('ratio_percent') or '-'} | {row['status']} |"
        )
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
