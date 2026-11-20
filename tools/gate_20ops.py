#!/usr/bin/env python3
"""Per-operator submission gate for the 20-op (19 + Add/Argmax/GLA) submission tree.

Why this exists
---------------
`team/scripts/check_op_submission.py` validates a single-operator *git diff*
against `origin/integration/npu-ops`.  This snapshot has no upstream history
(the clone is a file snapshot, origin returns 401), so there is no per-op
diff to feed it.  This runner applies the *same rules* to the explicit file
set of each operator instead:

  * branch naming / protected branch          -> checked from progress YAML
  * forbidden transient artifacts             -> check_op_submission patterns
  * out-of-scope paths                        -> same allow-list
  * >5 MB files, secret patterns              -> same thresholds/patterns
  * six areas (Manifest/Workload/Op/Kernel/Test/Bench)
  * evidence set (DESIGN/REVIEW/integration_log/run.json/report.md/RETROSPECTIVE)
  * optimized ops need perf_records.jsonl + opt_log.md
  * progress entry: dev_stage / branch / pr

Exit code is 1 when any operator fails.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TILEOPS = ROOT / "examples" / "TileOPs"
INVENTORY = ROOT / "submission_inventory.json"
OWNER = "郭益华"
PROGRESS = f"team/progress/{OWNER}.yaml"
MAX_MB = 5.0


def load_checker():
    path = ROOT / "team" / "scripts" / "check_op_submission.py"
    spec = importlib.util.spec_from_file_location("check_op_submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = load_checker()
AREAS = {
    "Manifest": "/tileops/manifest/",
    "Workload": "/tileops/workloads/",
    "Op 实现": "/tileops/ops/",
    "Kernel 实现": "/tileops/kernels/",
    "精度测试": "/tests/ops/",
    "Benchmark": "/benchmarks/ops/",
}


def rel(path: str) -> str:
    return f"examples/TileOPs/{path}"


def check_op(op: dict, progress: dict, owner_progress: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []

    areas = op["areas"]
    files: list[str] = [rel(p) for p in areas.values() if isinstance(p, str)]
    for key in ("workload",):
        files += [rel(p) for p in areas.get(key) or []]
    files += [rel(p) for p in op["evidence_paths"].values()]
    files += [rel(p) for p in op["perf_files"].values()]
    report_dir = (
        TILEOPS / "reports" / "tileops" / (op.get("report", {}).get("dir") or "")
    )
    if report_dir.is_dir():
        files += [
            rel(str(p.relative_to(TILEOPS)))
            for p in report_dir.rglob("*")
            if p.is_file()
        ]
    files += [owner_progress]
    files = sorted(set(f for f in files if f))

    # 1. forbidden artifacts / scope / size / secrets
    for path in files:
        if any(p.search(path) for p in CHECKER.FORBIDDEN_PATH_PATTERNS):
            errors.append(f"禁止提交临时或大体积文件：{path}")
        if not CHECKER.is_allowed_path(path, owner_progress):
            errors.append(f"文件不在允许的算子提交范围内：{path}")
        full = ROOT / path
        if not full.exists():
            errors.append(f"文件不存在：{path}")
            continue
        if full.stat().st_size / (1024 * 1024) > MAX_MB:
            errors.append(f"文件超过 {MAX_MB:g} MB：{path}")
        if full.is_file() and full.stat().st_size <= 1024 * 1024:
            try:
                text = full.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            if any(p.search(text) for p in CHECKER.SECRET_PATTERNS):
                errors.append(f"疑似包含密钥或密码：{path}")

    # 2. six areas
    for label, marker in AREAS.items():
        if not any(marker in path for path in files):
            errors.append(f"缺少{label}改动")

    # 3. evidence
    basenames = {Path(path).name for path in files}
    for label, candidates in CHECKER.REQUIRED_EVIDENCE.items():
        if not basenames.intersection(candidates):
            errors.append(f"缺少验收证据：{label}")

    # 4. optimized evidence
    if any("/perf_opt/" in path for path in files):
        for name in ("perf_records.jsonl", "opt_log.md"):
            if name not in basenames:
                errors.append(f"优化算子缺少{name}")

    # 5. progress entry
    branch = progress.get("branch", "")
    pr = progress.get("pr", "")
    stage = progress.get("dev_stage", "")
    if branch in CHECKER.PROTECTED_BRANCHES or not branch.startswith("op/"):
        errors.append(f"进度文件 branch 非法：{branch!r}")
    if stage not in CHECKER.DEV_STAGES:
        errors.append(f"dev_stage 取值非法：{stage!r}")
    elif stage != "已提交审核":
        errors.append("提交 PR 时 dev_stage 必须为“已提交审核”")
    if not pr:
        errors.append("进度文件缺少 pr 地址")

    if not errors:
        notes.append(f"{len(files)} 个文件 / branch={branch} / pr=...{pr[-3:]}")
    return errors, notes


def parse_progress() -> tuple[dict[str, dict], dict[str, str]]:
    """Return {op: record} and {op: owner} across every progress YAML."""
    import re

    records: dict[str, dict] = {}
    owners: dict[str, str] = {}
    for path in sorted((ROOT / "team" / "progress").glob("*.yaml")):
        owner = path.stem
        cur = None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*-\s*op_name:\s*(\S+)", line)
            if m:
                cur = m.group(1)
                records[cur] = {}
                owners[cur] = owner
                continue
            m = re.match(r"\s+(\w+):\s*(.*)", line)
            if m and cur:
                records[cur][m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return records, owners


def main() -> int:
    ops = json.loads(INVENTORY.read_text(encoding="utf-8"))
    progress, owners = parse_progress()
    failures = 0
    blockers = 0
    results = []
    print("== 20 算子提交闸门（规则同 team/scripts/check_op_submission.py） ==")
    print(f"owner: {OWNER}   tree: {ROOT}\n")
    for op in ops:
        name = op["op"]
        if op.get("status") == "blocker":
            print(f"[BLOCKER] {name}  ({op.get('notes','')})")
            results.append({"op": name, "status": "BLOCKER", "errors": [op.get("notes", "")]})
            blockers += 1
            continue
        owner = owners.get(name) or OWNER
        errors, notes = check_op(op, progress.get(name, {}), f"team/progress/{owner}.yaml")
        status = "PASS" if not errors else "FAIL"
        if errors:
            failures += 1
        results.append({"op": name, "status": status, "errors": errors})
        print(f"[{status}] {name}" + (f"  ({notes[0]})" if notes else ""))
        for e in errors:
            print(f"        - {e}")
    (ROOT / "submission_gate.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ready = len(ops) - blockers
    print(f"\n汇总：{ready - failures}/{ready} PASS, {failures} FAIL, {blockers} BLOCKER")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
