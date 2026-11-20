#!/usr/bin/env python3
"""Validate a single-operator submission before commit, push, or review.

The script intentionally uses only the Python standard library so every team
member can run it in a minimal environment.

Examples:
    python team/scripts/check_op_submission.py \
        --op-name BitwiseNotFwdOp --owner 吕盼豪

    python team/scripts/check_op_submission.py \
        --op-name BitwiseNotFwdOp --owner 吕盼豪 --require-pr
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROTECTED_BRANCHES = {"main", "integration/npu-ops"}
DEV_STAGES = {
    "未开始",
    "迁移中",
    "已迁移",
    "优化中",
    "已优化",
    "已提交审核",
    "已阻塞",
}
FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"(^|/)full\.json$"),
    re.compile(r"(^|/)\.stage_state\.json$"),
    re.compile(r"(^|/)\.migration_state\.json$"),
    re.compile(r"(^|/)\.task_timeline\.jsonl$"),
    re.compile(r"(^|/)history_version(/|$)"),
    re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.ruff_cache|build|dist)(/|$)"),
    re.compile(r"\.(msprof|db|sqlite|trace|prof)$", re.IGNORECASE),
    re.compile(r"\.log$", re.IGNORECASE),
    re.compile(r"(^|/)\.env($|\.)"),
    re.compile(r"\.(pem|key)$", re.IGNORECASE),
]
SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key|password|passwd)\b"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+\-]{12,}"
    ),
    re.compile(r"\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{20,})\b"),
]
REQUIRED_EVIDENCE = {
    "DESIGN.md": ("DESIGN.md", "_DESIGN.md"),
    "REVIEW.md": ("REVIEW.md", "_REVIEW.md"),
    "integration_log.md": ("integration_log.md",),
    "run.json": ("run.json",),
    "report.md": ("report.md",),
    "RETROSPECTIVE.md": ("RETROSPECTIVE.md",),
}


def git(root: Path, *args: str, check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return proc.stdout


def git_text(root: Path, *args: str, check: bool = True) -> str:
    return git(root, *args, check=check).decode("utf-8", errors="replace").strip()


def git_names(root: Path, *args: str) -> list[str]:
    raw = git(root, *args)
    return [item for item in raw.decode("utf-8", errors="replace").split("\0") if item]


def collect_changed_files(root: Path, base: str) -> list[str]:
    files: set[str] = set()
    files.update(git_names(root, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRT"))
    files.update(git_names(root, "diff", "--name-only", "-z", "--diff-filter=ACMRT"))
    files.update(git_names(root, "ls-files", "--others", "--exclude-standard", "-z"))

    if not files:
        if not git_text(root, "rev-parse", "--verify", base, check=False):
            raise RuntimeError(
                f"找不到基准分支 {base}，请先运行 git fetch origin，或使用 --base 指定基准分支。"
            )
        files.update(
            git_names(root, "diff", "--name-only", "-z", "--diff-filter=ACMRT", f"{base}...HEAD")
        )
    return sorted(path.replace("\\", "/") for path in files)


def parse_progress(path: Path, op_name: str) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(rf"^\s*-\s*op_name:\s*{re.escape(op_name)}\s*$", line):
            start = index
            break
    if start is None:
        raise ValueError(f"在 {path} 中找不到算子 {op_name}")

    values: dict[str, str] = {"op_name": op_name}
    for line in lines[start + 1 :]:
        if re.match(r"^\s*-\s*op_name:", line):
            break
        match = re.match(r"^\s+([A-Za-z_]+):\s*(.*?)\s*$", line)
        if match:
            key, value = match.groups()
            values[key] = value.strip().strip('"').strip("'")
    return values


def is_allowed_path(path: str, owner_progress: str) -> bool:
    if path == owner_progress:
        return True
    if path.startswith("team/knowledge/"):
        return True
    if path.startswith("examples/"):
        return True
    if path.startswith("reports/"):
        return True
    if path.startswith("evidence/"):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="检查单个算子的提交范围和必要证据")
    parser.add_argument("--op-name", required=True, help="Manifest 中的算子名称，例如 BitwiseNotFwdOp")
    parser.add_argument("--owner", required=True, help="负责人姓名，例如 吕盼豪")
    parser.add_argument("--base", default="origin/integration/npu-ops", help="对比的集成分支")
    parser.add_argument("--require-pr", action="store_true", help="提交 PR 后检查 PR 地址已填写")
    parser.add_argument("--optimized", action="store_true", help="强制要求优化证据")
    parser.add_argument("--max-file-mb", type=float, default=5.0, help="单个文件大小限制")
    parser.add_argument("--repo-root", default=".", help="仓库根目录，默认当前目录")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    if not (root / ".git").exists():
        print(f"[FAIL] 当前目录不是 Git 仓库：{root}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    branch = git_text(root, "rev-parse", "--abbrev-ref", "HEAD")
    owner_progress = f"team/progress/{args.owner}.yaml"

    print("== 算子提交检查 ==")
    print(f"分支：{branch}")
    print(f"算子：{args.op_name}")
    print(f"负责人：{args.owner}")

    if branch in PROTECTED_BRANCHES:
        errors.append(f"不能从受保护分支 {branch} 提交算子，请使用 op/<family>/<op>-<owner> 分支")
    elif not branch.startswith("op/"):
        errors.append(f"当前分支 {branch} 不是算子分支，必须以 op/ 开头")
    else:
        print("[PASS] 当前分支是算子开发分支")

    try:
        changed = collect_changed_files(root, args.base)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if not changed:
        errors.append("没有检测到任何改动")
    else:
        print(f"检测到 {len(changed)} 个改动文件：")
        for path in changed:
            print(f"  - {path}")

    for path in changed:
        if any(pattern.search(path) for pattern in FORBIDDEN_PATH_PATTERNS):
            errors.append(f"禁止提交临时或大体积文件：{path}")
        if not is_allowed_path(path, owner_progress):
            errors.append(f"文件不在允许的算子提交范围内：{path}")
        file_path = root / path
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > args.max_file_mb:
                errors.append(
                    f"文件超过 {args.max_file_mb:g} MB：{path} ({size_mb:.2f} MB)"
                )
            if file_path.stat().st_size <= 1024 * 1024:
                try:
                    text = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = ""
                for pattern in SECRET_PATTERNS:
                    if pattern.search(text):
                        errors.append(f"疑似包含密钥或密码：{path}")
                        break

    required_code = {
        "Manifest": "/tileops/manifest/",
        "Workload": "/tileops/workloads/",
        "Op 实现": "/tileops/ops/",
        "Kernel 实现": "/tileops/kernels/",
        "精度测试": "/tests/ops/",
        "Benchmark": "/benchmarks/ops/",
    }
    for label, marker in required_code.items():
        if any(marker in path for path in changed):
            print(f"[PASS] 已提交{label}")
        else:
            errors.append(f"缺少{label}改动")

    basenames = {Path(path).name for path in changed}
    for label, candidates in REQUIRED_EVIDENCE.items():
        if basenames.intersection(candidates):
            print(f"[PASS] 已提交 {label}")
        else:
            errors.append(f"缺少验收证据：{label}")

    optimized = args.optimized or any("/perf_opt/" in path for path in changed)
    if optimized:
        for label, name in (("性能记录", "perf_records.jsonl"), ("优化日志", "opt_log.md")):
            if name in basenames:
                print(f"[PASS] 已提交{label}")
            else:
                errors.append(f"优化算子缺少{label}：{name}")

    progress_path = root / owner_progress
    if owner_progress not in changed:
        errors.append(f"必须更新自己的进度文件：{owner_progress}")
    elif not progress_path.exists():
        errors.append(f"进度文件不存在：{owner_progress}")
    else:
        try:
            record = parse_progress(progress_path, args.op_name)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            stage = record.get("dev_stage", "")
            if stage not in DEV_STAGES:
                errors.append(f"dev_stage 取值非法：{stage!r}")
            elif args.require_pr and stage != "已提交审核":
                errors.append("提交 PR 时 dev_stage 必须为“已提交审核”")
            else:
                print(f"[PASS] dev_stage = {stage}")

            progress_branch = record.get("branch", "")
            if not progress_branch.startswith("op/"):
                errors.append("进度文件中的 branch 必须填写为 op/ 开头的分支名")
            else:
                print(f"[PASS] branch = {progress_branch}")

            pr = record.get("pr", "")
            if args.require_pr and not pr:
                errors.append("提交 PR 时必须在进度文件中填写 pr 地址")
            elif pr:
                print(f"[PASS] pr = {pr}")

    if errors:
        print("\n检查结果：FAIL")
        for error in errors:
            print(f"  [FAIL] {error}")
    else:
        print("\n检查结果：PASS")
    for warning in warnings:
        print(f"  [WARN] {warning}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())