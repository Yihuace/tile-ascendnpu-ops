"""repro_runner.py -- batch executor for the KB repro suite (ED-D).

The repro suite turns knowledge entries from "facts that were once verified"
into "facts that can be re-verified at any time":

  - toolchain upgrade re-verification: kb_stale_check lists stale entries ->
    repro_runner --filter stale re-runs their repros -> FAIL means the entry
    is overturned (negate/deprecate delta), PASS refreshes the verified
    stamp (re-verification goes through the evolver update delta);
  - apply pre-validation (E-4) and Tier 1 fast-track confirmation (E-3:
    a repro re-run in a different context counts as the second independent
    evidence) reuse the same suite;
  - CI may run syntax-level checks; full NPU execution is triggered on
    demand (each repro is a real NPU probe).

Usage:
  python3 .agents/tools/repro_runner.py                      # all repros
  python3 .agents/tools/repro_runner.py --filter traps       # TRAP-* files
  python3 .agents/tools/repro_runner.py --filter constants   # CONST-* files
  python3 .agents/tools/repro_runner.py --filter stale       # entries listed
                                                             # by kb_stale_check
  python3 .agents/tools/repro_runner.py --filter TRAP-load-nd2nz-strided
  python3 .agents/tools/repro_runner.py --list
Exit code 0 if every executed repro passed, 1 otherwise, 2 on usage error.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 2)))
REPRO_DIR = os.path.join(
    REPO_ROOT,
    ".agents",
    "skills",
    "tilelang-op-optimize",
    "references",
    "pattern-library",
    "repro",
)
STALE_CHECK = os.path.join(REPO_ROOT, ".agents", "tools", "kb_stale_check.py")
DEFAULT_TIMEOUT = 900


def discover() -> list:
    return sorted(glob.glob(os.path.join(REPRO_DIR, "*.py")))


def stale_ids() -> set:
    """Entry ids listed as stale by kb_stale_check (JSON output)."""
    try:
        proc = subprocess.run(
            [sys.executable, STALE_CHECK, "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        data = json.loads(proc.stdout)
        return {e["id"] for e in data.get("stale", [])}
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return set()


def select(files: list, flt: str) -> list:
    if not flt or flt == "all":
        return files
    if flt == "stale":
        ids = stale_ids()
        return [f for f in files if os.path.splitext(os.path.basename(f))[0] in ids]
    if flt in ("traps", "trap"):
        return [f for f in files if os.path.basename(f).startswith("TRAP-")]
    if flt in ("constants", "constant"):
        return [f for f in files if os.path.basename(f).startswith("CONST-")]
    if flt in ("patterns", "pattern"):
        return [f for f in files if os.path.basename(f).startswith("PATT-")]
    return [f for f in files if os.path.splitext(os.path.basename(f))[0] == flt]


def run_one(path: str, timeout: int) -> dict:
    rid = os.path.splitext(os.path.basename(path))[0]
    started = time.time()
    # delta-form optimization-point repros (PATT-*) carry a code skeleton and
    # an insight summary; per ED-B they are syntax-checked, not executed.
    if rid.startswith("PATT-"):
        try:
            import ast

            with open(path, encoding="utf-8") as f:
                ast.parse(f.read())
            return {
                "id": rid,
                "status": "PASS",
                "duration_s": 0.0,
                "tail": "delta-form: syntax check only (ED-B)",
            }
        except SyntaxError as e:
            return {
                "id": rid,
                "status": "FAIL",
                "duration_s": 0.0,
                "tail": f"syntax error: {e}",
            }
    try:
        proc = subprocess.run(
            [sys.executable, os.path.basename(path)],
            cwd=REPRO_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "PASS" if proc.returncode == 0 else "FAIL"
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        if status == "FAIL":
            err = (proc.stderr.strip().splitlines() or [""])[-1]
            tail = f"{tail} | {err}"[:300]
        return {
            "id": rid,
            "status": status,
            "duration_s": round(time.time() - started, 1),
            "tail": tail[:300],
        }
    except subprocess.TimeoutExpired:
        return {
            "id": rid,
            "status": "ERROR",
            "duration_s": round(time.time() - started, 1),
            "tail": f"timeout after {timeout}s",
        }
    except OSError as e:
        return {"id": rid, "status": "ERROR", "duration_s": 0.0, "tail": str(e)[:300]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--filter",
        default="all",
        help="all / traps / constants / patterns / stale / <entry_id>",
    )
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    files = discover()
    if args.list:
        for f in files:
            print(os.path.splitext(os.path.basename(f))[0])
        return 0
    chosen = select(files, args.filter)
    if not chosen:
        print(f"repro_runner: no repros matched filter '{args.filter}'")
        return 2

    results = [run_one(f, args.timeout) for f in chosen]
    passed = sum(1 for r in results if r["status"] == "PASS")
    summary = {
        "filter": args.filter,
        "executed": len(results),
        "passed": passed,
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "errors": sum(1 for r in results if r["status"] == "ERROR"),
        "results": results,
        "verdict": "ALL_PASS" if passed == len(results) else "HAS_FAILURES",
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"{'id':44s} {'status':8s} {'secs':>7s}  tail")
        for r in results:
            print(
                f"{r['id']:44s} {r['status']:8s} {r['duration_s']:7.1f}  "
                f"{r['tail'][:100]}"
            )
        print(
            f"\nrepro_runner[{args.filter}]: {passed}/{len(results)} PASS "
            f"-> {summary['verdict']}"
        )
    return 0 if summary["verdict"] == "ALL_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
