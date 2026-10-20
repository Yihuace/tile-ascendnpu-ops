#!/usr/bin/env python3
"""数据交付面 lint：守住「9.0 分支的数据面只有 9.0.0 一套口径」。

两个检查：

1. **无历史版本号**：数据交付面不得出现上一代工具链的版本号字面量。面 = `perf_data/**`、
   `MIGRATION_INDEX.md`、`STATUS.md`、`versions.txt`、`examples/TileOPs/reports/optimize/**`、
   `examples/TileOPs/reports/tileops/**`（生成报告）、`examples/TileOPs/profile_run_msprof_*.log`、
   `examples/TileOPs/tileops/kernels/**/perf_opt/perf_records.jsonl`（机器数据记录）、
   以及仓库外的交付件 `../19_operators_ratio.xlsx`。
   过程叙事（`perf_opt/opt_log.md`、`*_DESIGN.md`、`RETROSPECTIVE.md`、`docs/**`、`.agents/**`）
   按约定保留历史版本号，不在本 lint 范围内。
   正则刻意写成 ``(?<!\\d)8\\.5(?!\\d)``：只匹配「8.5」这个独立 token，
   因此 ``8.54e-04``（max_abs_err 数值）、``68.5%``（ratio）都不会误报，
   而 ``8.5``、``8.5.2`` 会被抓住。

2. **文档数字 == SSOT**：交付表（``FINAL_DELIVERY_20261018.md`` §2）与台账
   （``MIGRATION_INDEX.md`` §A）里的 ratio 区间与达标数必须和 ``perf_data/canonical.json``
   一致（每个算子恰好一条记录）。STATUS / 总报告 / STAGE4_SUMMARY 只放结论与索引。

用法::

    python3 .agents/tools/data_surface_lint.py            # 校验，有问题返回 1
    python3 .agents/tools/data_surface_lint.py --list     # 只打印扫描面
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 数据交付面
SURFACE_FILES = (
    "MIGRATION_INDEX.md",
    "STATUS.md",
    "versions.txt",
    "examples/TileOPs/reports/迁移与优化总报告.md",
)
SURFACE_DIRS = (
    "perf_data",
    "examples/TileOPs/reports/optimize",
    "examples/TileOPs/reports/tileops",  # 生成的 9.0 报告目录
)
SURFACE_GLOBS = (
    "examples/TileOPs/profile_run_msprof_*.log",
    # 机器数据记录（每算子恰好一份）：是数据面，不允许带历史版本号
    "examples/TileOPs/tileops/kernels/**/perf_opt/perf_records.jsonl",
)

# 历史工具链版本号 token（上一代 = 8.5.x）
FORBIDDEN = re.compile(r"(?<!\d)8\.5(?!\d)")

FINAL_DELIVERY = "examples/TileOPs/reports/FINAL_DELIVERY_20261018.md"
# ratio 一致性：这些文档里只要某算子行上出现 ratio，就必须能溯源到 SSOT
NUMBER_DOCS = (FINAL_DELIVERY, "MIGRATION_INDEX.md")
# 强制出现：这些文档必须给出每个已交付算子的 ratio 区间
REQUIRED_DOCS = (FINAL_DELIVERY, "MIGRATION_INDEX.md")
# 达标数 a/b：汇总表强制；状态快照表只校验「写出来的那一行」
COUNT_DOCS = (FINAL_DELIVERY, "MIGRATION_INDEX.md")

# 令牌 / 私钥特征（命中只报位置，绝不回显内容）
SECRET_PATTERNS = (
    ("GitHub fine-grained PAT", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("私钥", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
)

SSOT = REPO / "perf_data" / "canonical.json"
# 交付件（仓库外）：19 算子比率表
XLSX = REPO.parent / "19_operators_ratio.xlsx"

RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[–\-~]\s*(\d+(?:\.\d+)?)\s*%")
SINGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
FRACTION_RE = re.compile(r"\b\d+\s*/\s*\d+")
TOL = 0.15  # 文档按 1 位小数四舍五入，允许的最大偏差

# 交付表用短名/缩写时补别名（FINAL_DELIVERY 用全名）
ALIASES: dict[str, tuple[str, ...]] = {
    "FusedAddLayerNormFwdOp": ("FALN",),
    "GatedDeltaNetPrefillFwdOp": ("GDN",),
    "GLADecodeOp": ("GLA",),
}


def _names(op: str) -> tuple[str, ...]:
    """算子在文档里可能出现的写法：全名 + 去掉 Op / Fwd 后缀 + 别名。

    §2 汇总表用的短名保留 ``Bwd``（如 ``BatchNormBwd``），只去掉 ``Op`` / ``Fwd``。
    """
    short = re.sub(r"Op$", "", op)
    short = re.sub(r"Fwd$", "", short)
    out = [op]
    if short and short != op:
        out.append(short)
    out.extend(ALIASES.get(op, ()))
    return tuple(dict.fromkeys(out))


def _mentions(line: str, names: tuple[str, ...]) -> bool:
    """名字后面不能紧跟字母/数字/下划线，避免 ``Add`` 命中 ``AddLayerNorm``。"""
    for name in names:
        for m in re.finditer(re.escape(name), line):
            nxt = line[m.end() : m.end() + 1]
            if not (nxt.isalnum() or nxt == "_"):
                return True
    return False


def surface_paths() -> list[Path]:
    paths: list[Path] = []
    for name in SURFACE_FILES:
        p = REPO / name
        if p.exists():
            paths.append(p)
    for name in SURFACE_DIRS:
        d = REPO / name
        if d.is_file():
            paths.append(d)
        elif d.is_dir():
            paths.extend(sorted(f for f in d.rglob("*") if f.is_file()))
    for pattern in SURFACE_GLOBS:
        paths.extend(sorted(REPO.glob(pattern)))
    return paths


def check_forbidden() -> list[str]:
    problems: list[str] = []
    for path in surface_paths():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in FORBIDDEN.finditer(line):
                problems.append(
                    f"{path.relative_to(REPO)}:{lineno}: 出现历史版本号 "
                    f"“{m.group(0)}” -> {line.strip()[:120]}"
                )
    return problems


def _canonical() -> dict:
    if not SSOT.exists():
        raise SystemExit(f"[lint] 缺少 SSOT：{SSOT.relative_to(REPO)}")
    return json.loads(SSOT.read_text(encoding="utf-8"))


def _ratio_range(record: dict, phase: str = "optimized") -> tuple[float, float] | None:
    summary = record.get("summary") or {}
    block = summary.get(phase) or {}
    rng = block.get("ratio_pct")
    if rng and len(rng) == 2 and block.get("n"):
        return float(rng[0]), float(rng[1])
    return None


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def _match(rng: tuple[float, float], line: str) -> bool:
    lo, hi = rng
    for a, b in RANGE_RE.findall(line):
        if _near(float(a), lo) and _near(float(b), hi):
            return True
    for a in SINGLE_RE.findall(line):
        v = float(a)
        if lo - TOL <= v <= hi + TOL:
            return True
    return False


def check_numbers(canon: dict) -> list[str]:
    problems: list[str] = []
    ops = canon.get("operators") or {}
    if len(ops) != 19:
        problems.append(f"perf_data/canonical.json: 期望 19 个算子，实到 {len(ops)} 个")

    needed = sorted(set(NUMBER_DOCS) | set(COUNT_DOCS))
    doc_texts: dict[str, str] = {}
    for name in needed:
        p = REPO / name
        doc_texts[name] = p.read_text(encoding="utf-8") if p.exists() else ""
        if not doc_texts[name]:
            problems.append(f"{name}: 数字一致性检查所需的文档缺失")

    for op, record in ops.items():
        opt = _ratio_range(record, "optimized")
        mig = _ratio_range(record, "migrated")
        summary = record.get("summary") or {}
        block = summary.get("optimized") or {}
        names = _names(op)
        adjustable = summary.get("adjustable_shapes") or block.get("n")

        # 1) ratio 区间必须能溯源到 SSOT
        if opt is not None or mig is not None:
            for name in NUMBER_DOCS:
                lines = [
                    ln for ln in doc_texts[name].splitlines() if _mentions(ln, names)
                ]
                seen = any(RANGE_RE.search(ln) or SINGLE_RE.search(ln) for ln in lines)
                matched = any(
                    _match(rng, ln)
                    for ln in lines
                    for rng in (opt, mig)
                    if rng is not None
                )
                if matched:
                    continue
                if seen:
                    problems.append(
                        f"{name}: {op} 行上的 ratio 无法溯源到 SSOT"
                        f"（优化态 {opt}，迁移态 {mig}）"
                    )
                elif name in REQUIRED_DOCS:
                    problems.append(
                        f"{name}: 缺少 {op} 的 ratio（期望优化态 {opt}，迁移态 {mig}）"
                    )

        # 2) 达标数 a/b（状态快照表只校验已经写出 a/b 的行，避免误伤「时延收益」列）
        if opt is not None and adjustable and record.get("target_pct") is not None:
            want = f"{int(block.get('meeting', 0))}/{int(adjustable)}"
            for name in COUNT_DOCS:
                lines = [
                    ln for ln in doc_texts[name].splitlines() if _mentions(ln, names)
                ]
                if name == FINAL_DELIVERY:
                    ok = any(want in ln for ln in lines)
                else:
                    quoted = any(FRACTION_RE.search(ln) for ln in lines)
                    ok = (not quoted) or any(want in ln for ln in lines)
                if not ok:
                    problems.append(f"{name}: {op} 的达标数应为 {want}，文档里没找到")
    return problems


def check_secrets() -> list[str]:
    """仓库内（git 跟踪文件）不得出现令牌/私钥；命中只报文件+行号，不回显内容。"""
    problems: list[str] = []
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True
    ).stdout
    for raw in out.split(b"\0"):
        if not raw:
            continue
        path = REPO / raw.decode("utf-8", "replace")
        try:
            if path.stat().st_size > 2 * 1024 * 1024:  # 大二进制跳过
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, pat in SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                lineno = text[: m.start()].count("\n") + 1
                problems.append(
                    f"{path.relative_to(REPO)}:{lineno}: 检出疑似{label}"
                    "（内容不回显；请移除该文件中的凭据并轮换令牌）"
                )
    return problems


def check_totals(canon: dict) -> list[str]:
    """shape 级达标汇总必须按「可调 shape」作分母（微型 shape 豁免，不计入）。"""
    ops = canon.get("operators") or {}
    meet = adj = 0
    for rec in ops.values():
        if rec.get("target_pct") is None:  # 未定义达标线（GDN）
            continue
        sm = rec.get("summary") or {}
        opt = sm.get("optimized") or {}
        adj += sm.get("adjustable_shapes") or opt.get("n") or 0
        meet += opt.get("meeting", 0)
    if not adj:
        return []
    doc = REPO / FINAL_DELIVERY
    if not doc.exists():
        return [f"{FINAL_DELIVERY}: 缺失"]
    text = doc.read_text(encoding="utf-8")
    want = f"{meet}/{adj}"
    if want in text or f"{meet} / {adj}" in text:
        return []
    return [
        f"{FINAL_DELIVERY}: shape 级达标汇总应为 {want}"
        f"（可调 shape 分母；微型 shape 豁免不计入），文档里没找到"
    ]


def check_xlsx(canon: dict) -> list[str]:
    """交付件比率表：无历史版本号 + 优化 ratio 与 SSOT 一致（由 gen_ratio_xlsx.py 生成）。"""
    if not XLSX.exists():
        print(f"[lint] 提示：未找到交付件 {XLSX}，跳过其一致性检查")
        return []
    try:
        import openpyxl
    except ImportError:  # pragma: no cover
        print("[lint] 提示：未安装 openpyxl，跳过交付件检查")
        return []

    problems: list[str] = []
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb.worksheets[0]
    for row in ws.iter_rows(values_only=True):
        for cell in row:
            if isinstance(cell, str):
                for m in FORBIDDEN.finditer(cell):
                    problems.append(
                        f"{XLSX.name}: 出现历史版本号 “{m.group(0)}” -> {cell.strip()[:120]}"
                    )

    ops = canon.get("operators") or {}
    seen = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not isinstance(row[1], str):
            continue
        name = row[1].strip()
        op = next((o for o in ops if name == o or name.startswith(o)), None)
        if op is None:
            continue
        seen += 1
        for phase, col, label in (("migrated", 2, "迁移"), ("optimized", 3, "优化")):
            expected = _ratio_range(ops[op], phase)
            if expected is None:
                continue
            text = str(row[col] or "")
            if not _match(expected, text):
                problems.append(
                    f"{XLSX.name}: {name} 的{label} ratio “{text}” 与 SSOT {expected} 不一致"
                )
    if seen != 19:
        problems.append(f"{XLSX.name}: 期望 19 行算子，实到 {seen} 行")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="只打印扫描面")
    args = ap.parse_args(argv)

    paths = surface_paths()
    if args.list:
        print(f"数据交付面（{len(paths)} 个文件）：")
        for p in paths:
            print("  ", p.relative_to(REPO))
        return 0

    canon = _canonical()
    problems = (
        check_forbidden() + check_secrets() + check_numbers(canon) + check_xlsx(canon)
    )
    if problems:
        print(f"[lint] 发现 {len(problems)} 个问题：")
        for problem in problems:
            print("  ❌", problem)
        return 1
    print(
        f"[lint] 通过：数据面 {len(paths)} 个文件无历史版本号，"
        "文档 ratio/达标数与 perf_data/canonical.json 一致"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
