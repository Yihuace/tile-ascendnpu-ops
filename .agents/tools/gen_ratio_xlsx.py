#!/usr/bin/env python3
"""从 ``perf_data/canonical.json`` 重建 19 算子比率表 xlsx（交付件）。

交付件在仓库外：``/mnt/workspace/19_operators_ratio.xlsx``（可用第 1 个参数覆盖）。

为什么要有这个脚本：这张表原先**混合口径**——ClampScalar / Hardswish / ClampMin / Lerp 四行用的是
上一代工具链的衔接值，其余行是 9.0 的值，注释里还写死了版本号，读者无法判断哪一格可信。
现在两列统一到 **CANN 9.0.0 交付态**，ratio 一律从 SSOT 取（迁移态 / 优化态），
备注只写「数据怎么来的 + 达标情况」，不再出现历史版本号。

用法::

    python3 .agents/tools/gen_ratio_xlsx.py [目标.xlsx]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

REPO = Path(__file__).resolve().parents[2]
SSOT = REPO / "perf_data" / "canonical.json"
DEFAULT_XLSX = Path("/mnt/workspace/19_operators_ratio.xlsx")

# 与 FINAL_DELIVERY §2 一致的呈现顺序（序号 1..19）
ORDER = [
    "HardsigmoidFwdOp",
    "ClampScalarFwdOp",
    "HardswishFwdOp",
    "AddFwdOp",
    "SubFwdOp",
    "GeluAndMulFwdOp",
    "GeluTanhAndMulFwdOp",
    "IsnanFwdOp",
    "IsinfFwdOp",
    "IsfiniteFwdOp",
    "FusedAddLayerNormFwdOp",
    "BatchNormBwdOp",
    "Conv1dFwdOp",
    "ClampMinFwdOp",
    "LerpFwdOp",
    "Conv1dBiasFwdOp",
    "Conv3dBiasFwdOp",
    "GLADecodeOp",
    "GatedDeltaNetPrefillFwdOp",
]


def _fmt(lo: float, hi: float) -> str:
    """区间文本：下界 <1% 时用 2 位小数（卷积族的 0.03% 不能被四舍五入成 0.0%），否则 1 位。"""
    digits = 2 if lo < 1 else 1
    return f"{lo:.{digits}f}%–{hi:.{digits}f}%"


def span(record: dict, phase: str) -> str:
    block = (record.get("summary") or {}).get(phase) or {}
    rng = block.get("ratio_pct")
    if not (rng and len(rng) == 2):
        return "—"
    lo, hi = rng
    digits = 2 if lo < 1 else 1
    text = f"{lo:.{digits}f}%" if abs(lo - hi) < 1e-9 else _fmt(lo, hi)
    if block.get("n") == 1:
        text += "（单点）"
    if phase == "optimized" and record.get("micro_shapes"):
        text += f"（含{record['micro_shapes']}微型）"
    if phase == "optimized" and record["operator"] == "BatchNormBwdOp":
        text += "（不可比）"
    return text


ROWS = [
    (
        "HardsigmoidFwdOp",
        "6 shape（4 个微型豁免）；达标 0/2；compute-bound，判据失配 → 采纳 tie。区间含微型 shape",
    ),
    (
        "ClampScalarFwdOp",
        "11 个可调 shape；达标 1/11；9.0 上相对上一代工具链有回退（非优化可解，不触发重优化）",
    ),
    (
        "HardswishFwdOp",
        "6 个可调 shape；达标 2/6；compute-bound，wrapper 未切（迁移态 ≡ 交付态）",
    ),
    ("AddFwdOp", "11 个可调 shape；达标 10/11（min 69.37% 贴线未过）"),
    ("SubFwdOp", "15 个可调 shape；**达标 15/15**（唯一全 shape 严格达标）"),
    (
        "GeluAndMulFwdOp",
        "达标 0/4；dtype 分化 erf 阶数，prefill −43.6%",
    ),
    (
        "GeluTanhAndMulFwdOp",
        "达标 0/4；fold + hoist `vbrc`，prefill −8.2%",
    ),
    ("IsnanFwdOp", "达标 1/5；MTE2 混合读写带宽地板，未采纳 perf_opt"),
    ("IsinfFwdOp", "达标 1/5；bf16 `vand` 6→4 pass，bf16 −19.4~−19.9%"),
    ("IsfiniteFwdOp", "达标 1/5；bf16 `vand` 4-pass，bf16 −21.2%"),
    (
        "FusedAddLayerNormFwdOp",
        "5 个可调 shape（2 微型豁免）；达标 0/5；8b prefill −23.2%",
    ),
    (
        "BatchNormBwdOp",
        "达标 0/6；subid 去双计后 ratio 与优化前不可比；以 Task Duration 为准（−33~−46%）",
    ),
    (
        "Conv1dFwdOp",
        "15 个可调 shape；达标 0/15；通用 kernel wav2vec2 17.7×、encodec-init 4.2×；"
        "unit_stride 路径 whisper 15×、encodec-deep 3.6×；bench 仍有 flaky unit_stride 用例",
    ),
    (
        "ClampMinFwdOp",
        "7 个可调 shape；达标 0/7；复用 ClampScalar 的 perf_opt 内核，本算子未优化",
    ),
    ("LerpFwdOp", "11 个可调 shape；达标 9/11；未优化（迁移态 ≡ 交付态）"),
    (
        "Conv1dBiasFwdOp",
        "8 个可调 shape；达标 0/8；bias 路径 8.3× / 1.15×；bench 仍有 flaky 用例",
    ),
    (
        "Conv3dBiasFwdOp",
        "3 个可调 shape；达标 0/3；迁移 0.76–6.29% → 交付 1.68–20.43%；"
        "r3d-stem 2.36×、unet 21.3×（Task Duration）",
    ),
    (
        "GLADecodeOp",
        "5 个可调 shape；达标 0/5；latency-bound，k16→k32 相对收益 −24.2~35.4%",
    ),
    (
        "GatedDeltaNetPrefillFwdOp",
        "in-scope 主路径已交付（K1 + K2 split + K6 batched），test_gdn **10/10**；K5/CP 与小 shape fallback 保留。"
        "**无 msprof ratio**：per-kernel msprof 汇总 + op-level 理论 SOL 0.027% → 0.357%（约 13.3×，墙钟 4.4×），"
        "与其它行不可横比；瓶颈 `vec_p_main` 占 62%",
    ),
]

CALIBER_ROWS = [
    (
        "数据口径",
        "两列都是 **CANN 9.0.0**：**迁移态** = 优化前（Stage-4 baseline 等），**交付态** = 2026-10-18 端到端实测。"
        "唯一事实源 `perf_data/canonical.json`；交付态取 `perf_data/delivery.json`"
        "（逐行转录自 `examples/TileOPs/reports/FINAL_DELIVERY_20261018.md` §2；该批 per-run 报告目录未入库）。",
    ),
    (
        "达标统计",
        "全 shape 严格达标 **1 个**：Sub（15/15）；贴线/部分达标 4 个：Add（10/11）、Lerp（9/11）、"
        "Hardswish（2/6）、ClampScalar（1/11）；已采纳真实时延收益但未达标 12 个；未采纳 1 个（Isnan）；"
        "shape 级 **40/128（31.2%）**。8 个微型 shape 已豁免，**不计入分母**"
        "（交付表此前写的 40/136 把豁免项也算进了分母，已更正）。",
    ),
    (
        "迁移态列",
        "四类来源：① Stage-4 `perf_records.jsonl` 的 `phase=baseline`（11 个算子）；"
        "② 卷积族优化前 `bench_all.json`（Conv1d 5 项 / Conv1dBias 7 项）；"
        "③ 恒等（Hardswish / Lerp：无 perf_opt 内核、wrapper 未切）；"
        "④ clamp 迁移态补测（ClampScalar / ClampMin：临时切 Stage-3 baseline 跑 benchmark-only，"
        "见 `reports/tileops/*_migrated/PROVENANCE.md`）；"
        "⑤ Conv3dBias 官方迁移态 run `20261018_101545`（baseline `_conv3d_kernel`，17/17，ratio 0.76–6.29%）。"
        "早期 0.25% 自算 SOL 已废弃。",
    ),
    (
        "区间含微型 shape",
        "Hardsigmoid / FusedAddLayerNorm / GeluAndMul / GeluTanhAndMul 的交付态区间**含微型 shape**"
        "（≤5µs、物理不可达，故下限只有 0.4%~4%）。这些 case 在交付表里没有逐 case 数据"
        "（10-18 那批 per-run 目录未入库、工作区已删除），**无法事后剔除**，且按用户口径不接受重测，"
        "故保留原区间并显式标注。**判定/达标一律只用可调 shape**。",
    ),
    (
        "正确性",
        "**19/19 接口**：Conv1d / Conv1dBias 为 **27 passed + 1 skipped、0 failed**"
        "（此前的 CG-2026-0003 失败已不再复现）；GDN `test_gdn.py` 10/10。"
        "端到端报告 17 passed + 1 partial（Conv1d 的 bench 有 flaky `unit_stride` 用例）。",
    ),
    (
        "GDN",
        "in-scope 主路径交付（K=V∈{64,128}，含尾 chunk）；小 shape（K<16）与 K5/CP 走 fallback。"
        "性能为 **per-kernel msprof Task Duration 汇总 + op-level roofline**，"
        "**不是 msprof op 的 ratio**；迁移态 0.027% 由 per-head H1×16 重建，非直测。",
    ),
    (
        "BatchNormBwd",
        "ratio 因 subid 去双计而下降（15.3–28.2% → 11.91–24.44%），**与优化前不可比**；"
        "实际时延 −33~−46%（几何 −40.3%），以 Task Duration 为准。",
    ),
    (
        "统一 70% / 60% 口径",
        "按各算子自身达标线（访存 70 / 算力 60）统计，见「达标统计」。"
        "上一代工具链的数字一律作废，不与本表横比。",
    ),
]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    target = Path(argv[0]).expanduser() if argv else DEFAULT_XLSX
    canon = json.loads(SSOT.read_text(encoding="utf-8"))["operators"]

    wb = openpyxl.load_workbook(target)
    ws = wb["Ratio汇总"]
    ws.cell(row=1, column=3, value="迁移后 ratio（9.0.0）")
    ws.cell(row=1, column=4, value="优化 ratio（9.0.0 交付态）")

    for i, (op, note) in enumerate(ROWS, start=2):
        rec = canon[op]
        name = "LerpFwdOp（标量）" if op == "LerpFwdOp" else op
        if rec.get("status") in ("not_delivered", "missing_9_0_data"):
            mig = opt = "—"
        else:
            mig, opt = span(rec, "migrated"), span(rec, "optimized")
        ws.cell(row=i, column=1, value=i - 1)
        ws.cell(row=i, column=2, value=name)
        ws.cell(row=i, column=3, value=mig)
        ws.cell(row=i, column=4, value=opt)
        ws.cell(row=i, column=5, value=note)

    ws2 = wb["口径说明"]
    fill = PatternFill("solid", fgColor="DDEBF7")
    for i, (key, val) in enumerate(CALIBER_ROWS, start=2):
        ws2.cell(row=i, column=1, value=key).font = Font(bold=True)
        ws2.cell(row=i, column=1).fill = fill
        cell = ws2.cell(row=i, column=2, value=val)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for i in range(len(CALIBER_ROWS) + 2, ws2.max_row + 1):
        for j in (1, 2):
            ws2.cell(row=i, column=j).value = None

    wb.save(target)
    print(f"已重建 {target}（{len(ROWS)} 行算子 + {len(CALIBER_ROWS)} 行口径说明）")
    for i, (op, _) in enumerate(ROWS, start=2):
        print(
            f"   {i - 1:>2} {ws.cell(row=i, column=2).value:<28} "
            f"{ws.cell(row=i, column=3).value:<18} {ws.cell(row=i, column=4).value}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
