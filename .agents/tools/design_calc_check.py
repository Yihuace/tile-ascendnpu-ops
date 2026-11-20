"""design_calc_check.py -- mechanical arithmetic checks on DESIGN.md (D-5).

Best-effort mechanical re-computation of the machine-checkable parts of a
Stage 1 design document. Output is an advisory report for the Stage 2
reviewer (dimension 2/3/8 support) -- it is NOT a gate: any check that
cannot be parsed from the document is reported as "skip" with the reason.

Checks:
  1. ub_budget   -- sum the buffer sizes declared in the memory plan section
                    (mixed B/KB/MB units are normalized; VP-2026-0009) and
                    compare against the per-AIV UB capacity (192KB, with the
                    documented auto-multi-buffer inflation factor of ~1.7x
                    reported alongside).
  2. l0c_budget  -- for GEMM designs, block_M * block_N * accum_bytes must
                    fit the 128KB L0C.
  3. core_split  -- recompute logical cores ceil(M/bm) * ceil(N/bn) from the
                    numbers the design declares and cross-check the declared
                    logical-core count.
  4. r3_metrics  -- the research complexity table must cover the four
                    required metrics (FLOPs / access bytes / scan passes /
                    intermediate buffer peak).

Usage:
  python3 .agents/tools/design_calc_check.py --design examples/{p}/{op}/DESIGN.md
Exit code 0 always (advisory); parse the JSON "checks" list.
"""

import argparse
import json
import math
import re

UB_CAPACITY_KB = 192.0
UB_INFLATION = 1.7
L0C_CAPACITY_KB = 128.0
UNIT_KB = {"b": 1.0 / 1024, "kb": 1.0, "mb": 1024.0}


def _section(text: str, marker: str) -> str:
    """Return the body of the first heading containing the marker."""
    lines = text.splitlines()
    body, inside, depth = [], False, 0
    for ln in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            if inside and len(m.group(1)) <= depth:
                break
            if marker in m.group(2):
                inside, depth = True, len(m.group(1))
                continue
        if inside:
            body.append(ln)
    return "\n".join(body)


SIZE_CHAIN_RE = re.compile(r"((?:\d+(?:\.\d+)?\s*\+\s*)+\d+(?:\.\d+)?)\s*(B|KB|MB)\b")
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KB|MB)\b")
LEVEL_CAPACITY_KB = {
    "UB": 192.0,
    "L1": 512.0,
    "L0C": 128.0,
    "L0": 128.0,
    "L0A": 64.0,
    "L0B": 64.0,
}
TOTAL_MARK = re.compile(r"合计|总占用|total", re.I)
CAPACITY_MARK = re.compile(r"容量|佐证|参考|≤")


def _parse_sizes(s: str) -> list:
    """Parse size expressions in a string; chains like '16 + 16 + 8KB'
    count as one summed entry (mixed byte-unit trap, VP-2026-0009)."""
    out = []
    covered = []
    for m in SIZE_CHAIN_RE.finditer(s):
        total = sum(float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1)))
        out.append(total * UNIT_KB[m.group(2).lower()])
        covered.append(m.span())
    for m in SIZE_RE.finditer(s):
        if any(a <= m.start() < b for a, b in covered):
            continue
        out.append(float(m.group(1)) * UNIT_KB[m.group(2).lower()])
    return out


def check_ub_budget(text: str) -> dict:
    """Sum per-level buffer tables in the memory plan and compare against
    the level capacity; recompute any declared 合计 row."""
    sec = _section(text, "4.5") or _section(text, "内存") or ""
    level = None
    buckets = {}
    declared_totals = {}
    for ln in sec.splitlines():
        m = re.search(r"\*\*\s*(UB|L1|L0C|L0A|L0B)\b", ln) or re.search(
            r"^#{2,6}.*\b(UB|L1|L0C)\b", ln
        )
        if m:
            level = m.group(1).upper()
            buckets.setdefault(level, [])
            continue
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        joined = " ".join(cells)
        if set(joined.replace(" ", "")) <= {"-", " "}:
            continue
        if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
            continue  # header rows
        if level is None:
            continue
        if TOTAL_MARK.search(joined) and not CAPACITY_MARK.search(joined):
            vals = _parse_sizes(joined)
            if vals:
                declared_totals[level] = sum(vals)
            continue
        vals = _parse_sizes(joined)
        if vals:
            buckets[level].extend(vals)
    checks = []
    for lv, sizes in buckets.items():
        if not sizes:
            continue
        total = sum(sizes)
        cap = LEVEL_CAPACITY_KB.get(lv)
        detail = (
            f"{lv}: {len(sizes)} buffer entries sum = {total:.1f} KB"
            + (
                f" (declared 合计 {declared_totals.get(lv):.1f} KB, "
                f"recompute matches: "
                f"{abs(declared_totals.get(lv, total) - total) < 1.0})"
                if lv in declared_totals
                else ""
            )
            + f"; capacity = {cap} KB"
            + (
                f"; x{UB_INFLATION} auto-multi-buffer inflation budget = "
                f"{cap / UB_INFLATION:.0f} KB"
                if lv == "UB"
                else ""
            )
        )
        status = "fail" if cap and total > cap else "pass"
        checks.append(
            {
                "check": f"{lv.lower()}_budget",
                "status": status,
                "detail": detail,
                "within_inflation_budget": (
                    total <= cap / UB_INFLATION if lv == "UB" else None
                ),
            }
        )
    if not checks:
        return {
            "check": "ub_budget",
            "status": "skip",
            "detail": "no per-level buffer tables parsed from the memory "
            "plan (advisory checker; verify manually)",
        }
    return checks


def check_l0c_budget(text: str) -> dict:
    # L0C only exists on the Cube path: pure Vector/elementwise designs have
    # no GEMM tiles, and any MxN pair we parse there is a tensor shape.
    # Use affirmative API tokens only (negations like "无 matmul" don't count).
    if not re.search(r"T\.gemm|load_nd2nz|store_fixpipe|Scope\(\s*[\"']Cube", text):
        return {
            "check": "l0c_budget",
            "status": "skip",
            "detail": "no affirmative Cube-path API (T.gemm/load_nd2nz/"
            "store_fixpipe/Scope(Cube)) in this design",
        }
    sec = _section(text, "Tiling") or text
    bm = bn = None
    m = re.search(r"(?:block_?[MN]|bm)\s*[=×x*]\s*(\d+)", sec, re.I)
    if m:
        bm = int(m.group(1))
    m = re.search(r"(?:block_?[NM]|bn)\s*[=×x*]\s*(\d+)", sec, re.I)
    if m:
        bn = int(m.group(1))
    if bm is None or bn is None:
        # try "bm x bn" adjacent pattern
        m = re.search(
            r"\b(\d{2,4})\s*[x×*]\s*(\d{2,4})\b.*?(?:L0C|累加|accum|fp32)", text
        )
        if m:
            bm, bn = int(m.group(1)), int(m.group(2))
    if bm is None or bn is None:
        return {
            "check": "l0c_budget",
            "status": "skip",
            "detail": "no block_M x block_N values parsed",
        }
    accum_bytes = 4  # fp32 accumulation is the design default
    need_kb = bm * bn * accum_bytes / 1024.0
    ok = need_kb <= L0C_CAPACITY_KB
    return {
        "check": "l0c_budget",
        "status": "pass" if ok else "fail",
        "detail": (
            f"block {bm}x{bn} x fp32 accum = {need_kb:.1f} KB vs "
            f"L0C {L0C_CAPACITY_KB:.0f} KB"
        ),
    }


def check_core_split(text: str) -> dict:
    sec = _section(text, "Tiling") or _section(text, "5") or text
    nums = {}
    for key, pats in (
        ("M", [r"\bM\s*=\s*(\d+)", r"seq.*?(\d{3,6})\b"]),
        ("N", [r"\bN\s*=\s*(\d+)"]),
        ("bm", [r"\b(?:bm|block_?m)\s*=\s*(\d+)"]),
        ("bn", [r"\b(?:bn|block_?n)\s*=\s*(\d+)"]),
    ):
        for p in pats:
            m = re.search(p, sec, re.I)
            if m:
                nums[key] = int(m.group(1))
                break
    declared = None
    m = re.search(r"逻辑核数[^\d]*(\d+)", sec) or re.search(
        r"逻辑核数.*?=\s*(\d+)", sec
    )
    if m:
        declared = int(m.group(1))
    if not all(k in nums for k in ("M", "bm")):
        return {
            "check": "core_split",
            "status": "skip",
            "detail": f"insufficient parsed dims {nums}; declared logical "
            f"cores = {declared}",
        }
    n = nums.get("N", 1)
    bn = nums.get("bn", 1)
    logical = math.ceil(nums["M"] / nums["bm"]) * math.ceil(n / bn)
    if declared is None:
        return {
            "check": "core_split",
            "status": "pass",
            "detail": f"recomputed logical cores = ceil({nums['M']}/"
            f"{nums['bm']}) x ceil({n}/{bn}) = {logical} "
            f"(design declares no explicit count)",
        }
    ok = declared == logical
    return {
        "check": "core_split",
        "status": "pass" if ok else "fail",
        "detail": f"recomputed ceil({nums['M']}/{nums['bm']}) x "
        f"ceil({n}/{bn}) = {logical} vs declared {declared}",
    }


def check_r3_metrics(text: str) -> dict:
    sec = _section(text, "1.6.0") or text
    metrics = {
        "FLOPs": bool(re.search(r"FLOPs?|浮点", sec, re.I)),
        "access_bytes": bool(re.search(r"访存|Bytes|字节", sec)),
        "scan_passes": bool(re.search(r"扫描遍数|遍数|扫描次数", sec)),
        "buffer_peak": bool(re.search(r"中间缓冲|缓冲峰值|buffer", sec, re.I)),
    }
    missing = [k for k, v in metrics.items() if not v]
    if missing:
        return {
            "check": "r3_metrics",
            "status": "fail",
            "detail": f"complexity table missing metrics: {missing}",
        }
    return {
        "check": "r3_metrics",
        "status": "pass",
        "detail": "all four R3 metrics present "
        "(FLOPs/access bytes/scan passes/buffer peak)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True, help="path to DESIGN.md")
    args = ap.parse_args()
    try:
        with open(args.design, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(json.dumps({"error": str(e)}))
        return 0
    ub = check_ub_budget(text)
    if not isinstance(ub, list):
        ub = [ub]
    report = {
        "design": args.design,
        "capacity_constants": {
            "ub_kb": UB_CAPACITY_KB,
            "ub_inflation": UB_INFLATION,
            "l0c_kb": L0C_CAPACITY_KB,
            "source": "pattern-library/constants.md CONST-capacity-910B2C",
        },
        "checks": [
            *ub,
            check_l0c_budget(text),
            check_core_split(text),
            check_r3_metrics(text),
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
