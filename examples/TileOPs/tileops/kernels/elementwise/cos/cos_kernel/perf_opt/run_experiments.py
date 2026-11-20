"""run_experiments.py -- Stage 4 experiment batch runner for AbsFwdOp (T-4).

Adapted from tilelang-op-optimize/references/run_experiments_template.py.
Adaptation points vs the template:
  * CONFIG: AbsFwdOp kernel identity; the target kernel name reported by msprof
    is `abs_fwd_kernel` (NOT `main`, which is what the Stage 5 report records);
  * build_bench_cmd(): launches through ./bench.py, which runs the candidate
    kernel directly (never the Stage 5 wrapper benchmark);
  * subprocesses run WITHOUT PYTHONPATH and with TILELANG_ASCEND_MODE=Developer,
    matching the delivered kernel family's build path.

Usage (inside perf_opt/):
  python run_experiments.py --phase baseline --all
  python run_experiments.py --round 2 --workload elementwise-16m-float16
  python run_experiments.py --phase final --all
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys

# --------------------------------------------------------------- CONFIG (ADAPT)
OP = "cos_fwd_kernel"
KERNEL_ID = (
    "tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py"
    "::cos_fwd_kernel::main"
)
KERNEL_NAME = "main"  # ADAPT: cos's prim_func is named `main`
MSPROF = os.environ.get("MSPROF_BIN", "/usr/local/Ascend/cann-9.0.0/bin/msprof")
DIRECT_CMD = [
    sys.executable,
    "bench.py",
    "--impl",
    "{impl}",
    "--case",
    "{workload_id}",
]  # ADAPT: direct kernel argv; never the Stage 5 wrapper benchmark
LAUNCH_COUNT = 15
WARM_UP = 5
TIMEOUT_S = 900
CURRENT_BEST = "../cos_fwd_kernel.py"  # ADAPT: round base (Stage 3 deliverable)
CURRENT_BEST_ID = "baseline"  # ADAPT: ID of the current merged kernel version
# --------------------------------------------------------------- /CONFIG

PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"
MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)

# ----------------------------------------------------------- EXPERIMENTS (ADAPT)
# Each entry: {file, opt_id, parent_id}
EXPERIMENTS = [
    {"file": CURRENT_BEST, "opt_id": "baseline", "parent_id": None},
    # round 1: fold the tile walk into the kernel (grid = num_cores, grid-stride
    # T.serial inside).  Single main change; also removes the Block Dim overflow
    # that makes the 256m workload unlaunchable under the Stage 3 structure.
    {
        "file": "cos_fwd_kernel_round1_serial.py",
        "opt_id": "r1_serial48",
        "parent_id": "baseline",
    },
]

# round 2 coarse sweep on the round-1 structure: block_size x num_cores.
#   * block_size is tightly bounded here: this kernel holds ten block-sized UB
#     buffers totalling ~30 bytes per element of block_size (x/y 2+2, five fp32
#     20, int32 4, two uint8 2), so 2048 costs ~60 KB and 4096 ~120 KB against
#     the 192 KB UB -- 8192 would be ~240 KB and is expected to fail to compile.
#     The 4096 row is therefore a UB-ceiling probe, not just a perf point.
#   * num_cores is the grid width; 24 under-utilises the 48 Vector cores.
for _bs in (2048, 4096):
    for _nc in (24, 48, 96, 192):
        EXPERIMENTS.append(
            {
                "file": "cos_fwd_kernel_round1_serial.py",
                "opt_id": f"r2_bs{_bs}_nc{_nc}",
                "parent_id": "r1_serial48",
                "skip_l0": True,
                "extra": [
                    "--block-size", str(_bs),
                    "--num-cores", str(_nc),
                    "--check",
                ],
            }
        )
# ------------------------------------------------------------------ /EXPERIMENTS

# round 3: UB-ceiling probe (bs=8192 -- the ~240 KB estimate says it should NOT
# compile, and 4096 proving legal means the ~1.7x multibuffer inflation does not
# apply to this kernel) plus cross-dtype / cross-shape confirmation of the
# round-2 winner (bs=4096 / nc=48).
for _wid, _bs, _nc in (
    ("elementwise-16m-float16", 8192, 48),
    ("elementwise-16m-float32", 4096, 48),
    ("elementwise-16m-bfloat16", 4096, 48),
    ("elementwise-64m-float16", 4096, 48),
    ("elementwise-256m-float16", 4096, 48),
):
    _label = "-".join(_wid.split("-")[:2])
    EXPERIMENTS.append(
        {
            "file": "cos_fwd_kernel_round1_serial.py",
            "opt_id": f"r3_{_label}_{_wid.split('-')[-1]}_bs{_bs}_nc{_nc}",
            "parent_id": "r2_bs4096_nc48",
            "skip_l0": True,
            "workload": _wid,
            "extra": [
                "--block-size", str(_bs),
                "--num-cores", str(_nc),
                "--check",
            ],
        }
    )
# ------------------------------------------------------------------ /EXPERIMENTS

# The Stage 4 deliverable.  Final-phase measurements must come from this single
# candidate for every tune workload (contract: _shared/standards/signal-registry.md §5).
# round 3: b0/b1 come straight from the integer k with bitwise ops.  `q = k mod 4`
# was a DEAD intermediate -- only b0 and b1 are ever consumed downstream, so the
# 10-instruction fp32 chain that built q and immediately decomposed it collapses to
# 6.  The two int32 constants are (1,1) so the change stays UB-neutral, and ki_ub is
# reused in place (k is recomputed from kf_ub).
EXPERIMENTS.append(
    {
        "file": "cos_fwd_kernel_round3_intquad.py",
        "opt_id": "r3_intquad",
        "parent_id": "r2_bs4096_nc48",
        "skip_l0": False,
    }
)


FINAL_CANDIDATE = {"file": "cos_fwd_kernel.py", "opt_id": "final-bs4096-nc48", "parent_id": "r2_bs4096_nc48"}
EXPERIMENTS.append(FINAL_CANDIDATE)

# Baseline completion for the two 256m workloads.
# The Stage 3 structure launches one block per tile, so the grid is N/block_size.
# cos holds twelve block-sized UB buffers totalling ~44 bytes per element of
# block_size, which makes the two constraints MUTUALLY EXCLUSIVE at 256m:
#   * UB:  block_size * 44 <= 192 KB  =>  block_size <= 4468
#   * grid: N / block_size <= limit   =>  block_size >= 6144 (4096 gives grid
#           65536, which the driver rejects)
# There is therefore NO legal Stage 3 configuration at block_size >= 4468.  The
# only launchable reference is a block_size inside the narrow window that keeps
# the grid just under the limit: 4200 -> grid 63914, UB 180.5 KB (both legal).
# This is recorded as the baseline with that caveat, never as a default-config
# measurement.
for _wid in ("elementwise-256m-float16", "elementwise-256m-bfloat16"):
    EXPERIMENTS.append(
        {
            "file": CURRENT_BEST,
            "opt_id": f"baseline-launchable-{_wid.split('-')[1]}-{_wid.rsplit('-', 1)[-1]}",
            "parent_id": None,
            "skip_l0": True,
            "workload": _wid,
            "extra": ["--block-size", "4200"],
        }
    )


def child_env() -> dict:
    """Environment for every kernel subprocess.

    PYTHONPATH is dropped (the isolated venv must not inherit the harness path)
    and the Vector/Developer lowering path is pinned before the candidate module
    is imported.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.setdefault("TILELANG_ASCEND_MODE", "Developer")
    return env


def build_bench_cmd(impl_path: str, workload_id: str, extra=()) -> list[str]:
    base = [part.format(impl=impl_path, workload_id=workload_id) for part in DIRECT_CMD]
    return base + [str(item) for item in extra]


def safe_component(value: str) -> str:
    """Keep benchmark IDs out of path traversal and avoid sanitized collisions."""
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def run_l0(branch_file: str) -> bool:
    proc = subprocess.run(
        [sys.executable, branch_file, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=child_env(),
    )
    return proc.returncode == 0


def run_msprof(branch_file: str, workload_id: str, out_dir: str, extra=()) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "posix":
        os.chmod(out_dir, 0o700)
    cmd = [
        MSPROF,
        "op",
        f"--kernel-name={KERNEL_NAME}",
        f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}",
        f"--warm-up={WARM_UP}",
        "--dump=off",
        f"--aic-metrics={MSPROF_METRICS}",
        *build_bench_cmd(branch_file, workload_id, extra),
    ]
    log = os.path.join("logs", safe_component(out_dir) + ".log")
    os.makedirs("logs", exist_ok=True)
    with open(log, "w") as f:
        f.write(f"# {shlex.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=child_env()
        )
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"error": f"msprof rc={proc.returncode}", "log": log}
    durs = []
    for path in sorted(
        glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    ):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    if not durs:
        return {"error": "no OpBasicInfo rows", "log": log}
    return {
        "duration_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "msprof_raw_path": out_dir,
    }


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_workloads(phase: str = "baseline") -> set:
    """workload_ids that already have a row for `phase` in perf_records.jsonl."""
    if not os.path.isfile(PERF_RECORDS):
        return set()
    seen = set()
    with open(PERF_RECORDS) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kernel_id") == KERNEL_ID and row.get("phase") == phase:
                seen.add(row.get("workload_id"))
    return seen


def tune_workload_ids() -> list[str]:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)
    if not isinstance(inventory, dict) or not isinstance(
        inventory.get("workloads"), list
    ):
        raise ValueError("workload_inventory.json must contain a workloads array")
    return [
        item["workload_id"]
        for item in inventory["workloads"]
        if item.get("kernel_id") == KERNEL_ID and item.get("kind") == "tune"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--only", help="run a single opt_id")
    ap.add_argument("--workload", help="one non-smoke benchmark workload ID")
    ap.add_argument(
        "--all",
        action="store_true",
        help="measure every tune workload (baseline/final/merged)",
    )
    ap.add_argument(
        "--phase",
        choices=("baseline", "candidate", "merged", "final"),
        default="candidate",
    )
    args = ap.parse_args()

    if args.round is None:
        args.round = 0 if args.phase == "baseline" else 1
    if args.phase == "baseline" and args.round != 0:
        ap.error("baseline measurements must use round 0")

    if args.phase == "candidate":
        if args.all or not args.workload:
            ap.error("candidate measurements require exactly one --workload")
    elif args.phase in ("merged", "final"):
        if not args.all or args.workload:
            ap.error("merged/final measurements require --all and no --workload")
    else:  # baseline
        # --all is the normal path.  --workload exists only to COMPLETE a
        # baseline set that was interrupted (this run was), so that each tune
        # workload keeps exactly one baseline row in the append-only record.
        if not args.all and not args.workload:
            ap.error("baseline requires --all or --workload <id>")
        if args.all and args.workload:
            ap.error("baseline accepts either --all or --workload <id>, not both")
    if args.phase == "baseline":
        already = [w for w in (tune_workload_ids() if args.all else [args.workload])
                   if w in recorded_workloads()]
        if already:
            ap.error(
                "baseline already recorded for "
                f"{already}; perf_records.jsonl is append-only and must keep one "
                "baseline row per workload"
            )
    available = tune_workload_ids()
    selected = list(available) if args.all else [args.workload]
    unknown = [workload_id for workload_id in selected if workload_id not in available]
    if unknown or not selected:
        ap.error(f"unknown or empty tune workload selection: {unknown or selected}")

    todo = [e for e in EXPERIMENTS if not args.only or e["opt_id"] == args.only]
    if not todo:
        print("no experiments configured (edit EXPERIMENTS)")
        return 2

    if args.phase in ("baseline", "merged", "final") and len(todo) != 1:
        ap.error(
            "baseline/merged/final measurements must use exactly one candidate per kernel"
        )
    results = []
    for exp in todo:
        branch = exp["file"]
        print(f"=== {exp['opt_id']}: {branch} ===")
        if not os.path.isfile(branch):
            results.append({**exp, "status": "missing_file"})
            continue
        if args.phase != "baseline" and not exp.get("skip_l0"):
            l0 = run_l0(branch)
            if not l0:
                results.append({**exp, "status": "l0_fail", "l0_pass": False})
                continue
        for workload_id in ([exp["workload"]] if exp.get("workload") else selected):
            prof = run_msprof(
                branch,
                workload_id,
                os.path.join(
                    "profiles",
                    args.phase,
                    f"round{args.round}",
                    safe_component(exp["opt_id"]),
                    safe_component(workload_id),
                ),
                exp.get("extra", ()),
            )
            if "error" in prof:
                results.append(
                    {
                        **exp,
                        "workload_id": workload_id,
                        "status": "msprof_fail",
                        "l0_pass": run_l0(branch) if args.phase != "baseline" else None,
                    }
                )
                print(f"  {workload_id}: msprof FAILED: {prof['error']}")
                continue
            append_record(
                {
                    "round": args.round,
                    "candidate_id": exp["opt_id"],
                    "parent_id": exp.get("parent_id"),
                    "kernel_id": KERNEL_ID,
                    "workload_id": workload_id,
                    "phase": args.phase,
                    "artifact_path": branch,
                    "artifact_sha256": file_sha256(branch),
                    "duration_us": prof["duration_us"],
                    "l0_pass": True,
                    "msprof_raw_path": prof["msprof_raw_path"],
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
            results.append(
                {
                    **exp,
                    "workload_id": workload_id,
                    "status": "ok",
                    "l0_pass": True,
                    "duration_us": prof["duration_us"],
                }
            )
            print(
                f"  {workload_id}: median Task Duration = {prof['duration_us']} us "
                f"(n={prof['n']})"
            )

    # summary table: candidates vs current best (B2 structured backflow)
    bases = {}
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if (
                r.get("kernel_id") == KERNEL_ID
                and r.get("candidate_id") == CURRENT_BEST_ID
            ):
                bases[r.get("workload_id")] = r["duration_us"]
    print(
        "\n| branch | workload_id | l0 | Task Duration(us) | vs current best | status |"
    )
    print("|---|---|---|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        rel = ""
        base = bases.get(r.get("workload_id"))
        if dur is not None and base:
            rel = f"{(base - dur) / base * 100:+.1f}%"
        print(
            f"| {r['opt_id']} | {r.get('workload_id', '-')} | {r.get('l0_pass', '-')} | "
            f"{dur if dur is not None else '-'} | {rel} | {r['status']} |"
        )
    print(f"\n(current merged best candidate: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
