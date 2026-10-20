#!/usr/bin/env python3
"""Rebuild ``perf_data/canonical.json`` -- the single trusted 9.0 dataset per operator.

Why this exists
---------------
The 19-operator numbers used to be hand-copied into a dozen documents
(``STAGE4_SUMMARY.md`` §2, ``MIGRATION_INDEX.md`` §A, ``STATUS.md``, the ratio
xlsx, per-op ``OPTIMIZE_REPORT.md``).  They drifted, and several rows were still
labelled CANN 8.5.2.  This script derives every ratio from the two machine
records that actually hold one, applies one uniform rule, and writes the result
to a single file that the documents are checked against:

  1. Stage-4 record   ``perf_opt/perf_records.jsonl``   (one file per operator)
         ``phase == "baseline"`` -> 迁移态
         ``phase == "final"``    -> 优化态        (last round per workload)
  2. CANN 9.0.0 official report   ``reports/tileops/<ts>_<Op>/run.json``
         used only when the operator has no usable Stage-4 record
         (ClampScalar / Hardswish / ClampMin / Lerp)
  3. CANN 9.0.0 benchmark-only run ``reports/tileops/<ts>_<Op>_benchonly/benchmark.json``
         for operators whose correctness gate cannot pass stably (the
         convolution family, CG-2026-0003): the官方 run.json carries no ratio,
         so the benchmark stage is run on its own (see PROVENANCE.md). Still the
         same msprof Roofline ratio.

Definitions follow ``STAGE4_SUMMARY.md``:
  可调 shape = measured case with Task Duration > 5 µs  (微型 shape 豁免)
  达标       = ratio >= 达标线 (memory 70 / compute 60 / mix 40)

Usage
-----
    python3 .agents/tools/gen_perf_data.py            # write perf_data/canonical.json
    python3 .agents/tools/gen_perf_data.py --check    # non-zero exit if the file is stale
    python3 .agents/tools/gen_perf_data.py --table    # print the §2-style table
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TILEOPS = os.path.join(REPO, "examples", "TileOPs")
OUT = os.path.join(REPO, "perf_data", "canonical.json")
MANUAL = os.path.join(REPO, "perf_data", "manual.json")
DELIVERY = os.path.join(REPO, "perf_data", "delivery.json")

# 达标线 per STAGE4_SUMMARY §2 (访存 70 / 算力 60 / Mix 40)
TARGET = {
    "AddFwdOp": 70,
    "SubFwdOp": 70,
    "ClampScalarFwdOp": 70,
    "LerpFwdOp": 70,
    "ClampMinFwdOp": 70,
    "HardswishFwdOp": 60,
    "HardsigmoidFwdOp": 70,
    "IsnanFwdOp": 70,
    "IsinfFwdOp": 70,
    "IsfiniteFwdOp": 70,
    "GeluTanhAndMulFwdOp": 60,
    "GeluAndMulFwdOp": 60,
    "FusedAddLayerNormFwdOp": 60,
    "BatchNormBwdOp": 60,
    "GLADecodeOp": 60,
    "Conv1dFwdOp": 60,
    "Conv1dBiasFwdOp": 60,
    "Conv3dBiasFwdOp": 60,
    "GatedDeltaNetPrefillFwdOp": 60,  # 未定义达标线，仅占位
}
STEM = {
    "HardsigmoidFwdOp": "hardsigmoid",
    "ClampScalarFwdOp": "clamp_scalar",
    "HardswishFwdOp": "hardswish",
    "AddFwdOp": "add",
    "SubFwdOp": "sub",
    "GeluAndMulFwdOp": "gelu_and_mul",
    "GeluTanhAndMulFwdOp": "gelu_tanh_and_mul",
    "IsnanFwdOp": "isnan",
    "IsinfFwdOp": "isinf",
    "IsfiniteFwdOp": "isfinite",
    "FusedAddLayerNormFwdOp": "fused_add_layer_norm",
    "BatchNormBwdOp": "batch_norm_bwd",
    "Conv1dFwdOp": "conv1d",
    "Conv1dBiasFwdOp": "conv1d",
    "Conv3dBiasFwdOp": "conv3d",
    "GLADecodeOp": "gla_decode",
}
FAMILY = {
    "AddFwdOp": "elementwise",
    "SubFwdOp": "elementwise",
    "ClampScalarFwdOp": "elementwise",
    "LerpFwdOp": "elementwise",
    "ClampMinFwdOp": "elementwise",
    "HardswishFwdOp": "elementwise",
    "HardsigmoidFwdOp": "elementwise",
    "IsnanFwdOp": "elementwise",
    "IsinfFwdOp": "elementwise",
    "IsfiniteFwdOp": "elementwise",
    "GeluTanhAndMulFwdOp": "elementwise",
    "GeluAndMulFwdOp": "elementwise",
    "FusedAddLayerNormFwdOp": "norm",
    "BatchNormBwdOp": "norm",
    "GLADecodeOp": "linear_attention",
    "Conv1dFwdOp": "convolution",
    "Conv1dBiasFwdOp": "convolution",
    "Conv3dBiasFwdOp": "convolution",
    "GatedDeltaNetPrefillFwdOp": "linear_attention",
}
ORDER = list(FAMILY)
MICRO_US = 5.0  # Task Duration <= 5 µs 豁免

CALIBER = {
    "cann": "9.0.0",
    "soc": "Ascend910B3",
    "ratio_definition": "msprof op Roofline GM Read+Write SOL / 实测 Task Duration",
    "exemption": f"Task Duration <= {MICRO_US} us",
    "target_line": "访存 70 / 算力 60 / Mix 40",
}


def _jsonl_path(stem: str) -> str | None:
    for f in glob.glob(
        os.path.join(TILEOPS, "tileops/kernels/*/*/*/perf_opt/perf_records.jsonl")
    ):
        if (
            os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
            == stem
        ):
            return f
    return None


def _last_round(rows: list[dict], phase: str) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows:
        if r.get("phase") != phase or r.get("ratio_pct") is None:
            continue
        w = r.get("workload_id")
        if w not in best or (r.get("round") or 0) >= (best[w].get("round") or 0):
            best[w] = r
    return best


def _official(op: str) -> list[dict]:
    out = []
    for f in sorted(
        glob.glob(os.path.join(TILEOPS, f"reports/tileops/*_{op}/run.json"))
    ):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        env = (j.get("setup") or {}).get("environment") or {}
        if env.get("cann") != CALIBER["cann"]:
            continue
        cases = []
        for c in (j.get("performance") or {}).get("cases") or []:
            if (
                not c.get("profiler_valid")
                or c.get("roofline_utilization_percent") is None
            ):
                continue
            cases.append(
                {
                    "label": c.get("label"),
                    "dtype": (c.get("params") or {}).get("dtype"),
                    "ratio_pct": round(c["roofline_utilization_percent"], 4),
                    "latency_us": round(c["latency_us"], 4)
                    if c.get("latency_us")
                    else None,
                }
            )
        if not cases:
            # 失败的 run（正确性没过、benchmark 被跳过）不携带任何 ratio，不能算数据源
            continue
        meta = j.get("metadata") or {}
        out.append(
            {
                "run_id": os.path.basename(os.path.dirname(f)),
                "path": os.path.relpath(f, REPO),
                "stamp": {
                    "cann": env.get("cann"),
                    "soc": env.get("npu"),
                    "tilelang": env.get("tilelang"),
                    "git_commit": meta.get("git_commit"),
                    "generated_at": meta.get("generated_at"),
                },
                "cases": cases,
            }
        )
    return out


def _benchonly(op: str) -> list[dict]:
    """benchmark-only 报告：正确性闸门被 CG-2026-0003 阻断的算子只能单独跑 benchmark。

    官方 ``tileops-report run`` 硬编码了「正确性 passed 才跑 benchmark」，而这几个算子的
    失败用例集合每次漂移，静态 xfail 列表兜不住，因此产出的是 ``*_benchonly/benchmark.json``
    （见该目录的 ``PROVENANCE.md``）。ratio 仍是同一个 msprof Roofline 口径。
    """
    out = []
    pattern = os.path.join(TILEOPS, f"reports/tileops/*_{op}_benchonly/benchmark.json")
    for f in sorted(glob.glob(pattern)):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        cases = []
        for r in j.get("records") or []:
            res = r.get("result") or {}
            ratio = res.get("Ratio(%)")
            if ratio is None:
                continue
            lat = res.get("latency_us")
            cases.append(
                {
                    "label": r.get("label"),
                    "dtype": (r.get("params") or {}).get("dtype"),
                    "ratio_pct": round(float(ratio), 4),
                    "latency_us": round(float(lat), 4) if lat is not None else None,
                }
            )
        if not cases:
            continue
        env = " ".join(j.get("environment") or [])
        m = re.search(r"NPU model\*\*:\s*(\S+)", env)
        out.append(
            {
                "run_id": os.path.basename(os.path.dirname(f)),
                "path": os.path.relpath(f, REPO),
                "stamp": {
                    "cann": CALIBER["cann"],
                    "soc": m.group(1) if m else CALIBER["soc"],
                    "generated_at": j.get("generated_at"),
                },
                "cases": cases,
            }
        )
    return out


def _tunable(cases: list[dict]) -> list[dict]:
    adj = [
        c
        for c in cases
        if c.get("ratio_pct") is not None and (c.get("latency_us") or 0) > MICRO_US
    ]
    return sorted(adj, key=lambda c: c["ratio_pct"])


def _summary(cases: list[dict], target: int) -> dict:
    rs = [c["ratio_pct"] for c in cases]
    return {
        "n": len(cases),
        "ratio_pct": [min(rs), max(rs)] if rs else None,
        "meeting": sum(1 for c in cases if c["ratio_pct"] >= target),
    }


def build() -> dict:
    manual = {}
    migrated_supp: dict = {}
    if os.path.exists(MANUAL):
        doc = json.load(open(MANUAL, encoding="utf-8")) or {}
        manual = doc.get("operators") or {}
        migrated_supp = doc.get("migrated") or {}
    # 最新 9.0 实测交付表（权威：交付态一律取它）
    delivery: dict = {}
    if os.path.exists(DELIVERY):
        delivery = (json.load(open(DELIVERY, encoding="utf-8")) or {}).get(
            "operators"
        ) or {}
    ops: dict[str, dict] = {}
    for op in ORDER:
        if op in manual:
            ops[op] = manual[op]
            continue
        rec: dict = {
            "operator": op,
            "family": FAMILY[op],
            "status": "delivered",
            "target_pct": TARGET[op],
        }
        stem = STEM.get(op)
        path = _jsonl_path(stem) if stem else None
        if path and os.path.getsize(path) > 0:
            rows = []
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            finals = _last_round(rows, "final")
            if finals:
                bases = _last_round(rows, "baseline")
                rec["source"] = {
                    "kind": "stage4_records",
                    "path": os.path.relpath(path, REPO),
                }
                rec["stamp"] = {
                    "cann": "9.0.0",
                    "basis": "all rows timestamped after the CANN 9.0.0 re-baseline; "
                    "the jsonl itself carries no toolchain stamp",
                    "soc": CALIBER["soc"],
                }
                rec["optimized_raw"] = [
                    {
                        "label": w,
                        "ratio_pct": r["ratio_pct"],
                        "latency_us": r.get("duration_us"),
                    }
                    for w, r in sorted(finals.items())
                ]
                if bases:
                    rec["migrated_raw"] = [
                        {
                            "label": w,
                            "ratio_pct": r["ratio_pct"],
                            "latency_us": r.get("duration_us"),
                        }
                        for w, r in sorted(bases.items())
                    ]
        if "optimized_raw" not in rec and op not in delivery:
            kind = "official_report"
            reps = _official(op)
            if not reps:
                kind = "benchmark_only"
                reps = _benchonly(op)
            if not reps:
                rec["status"] = "missing_9_0_data"
                ops[op] = rec
                continue
            r = sorted(reps, key=lambda x: x["run_id"])[-1]
            rec["source"] = {"kind": kind, "run_id": r["run_id"], "path": r["path"]}
            if kind == "benchmark_only":
                rec["source"]["note"] = (
                    "官方 run 的正确性闸门被 CG-2026-0003 阻断（失败用例集合非确定），"
                    "benchmark 阶段单独执行；ratio 仍为 msprof Roofline 口径。"
                )
            rec["stamp"] = r["stamp"]
            rec["optimized_raw"] = r["cases"]
        for key in ("migrated", "optimized"):
            raw = rec.pop(f"{key}_raw", None)
            if raw is None:
                continue
            tunable = _tunable(raw)
            rec[key] = tunable
            rec.setdefault("summary", {})[key] = _summary(tunable, TARGET[op])
        # 交付态一律取【最新 9.0 实测】交付表（覆盖 jsonl / 旧报告推导值）
        dl = delivery.get(op)
        if dl is not None:
            sm = rec.setdefault("summary", {})
            rec["optimized"] = []
            rec["source"] = {
                "kind": "delivery_20261018",
                "path": "examples/TileOPs/reports/FINAL_DELIVERY_20261018.md §2",
                "note": "2026-10-18 端到端重测表；per-run 报告目录未入库，故以该表为快照",
            }
            rec["stamp"] = {
                "cann": "9.0.0",
                "soc": CALIBER["soc"],
                "measured_at": "2026-10-18",
            }
            if dl.get("self_sol"):
                self_sol = dl["self_sol"]
                rec["target_pct"] = None  # 未定义达标线
                sm["optimized"] = {
                    "n": 1,
                    "ratio_pct": [self_sol["optimized_pct"]] * 2,
                    "meeting": 0,
                }
                rec["migrated"] = []
                sm["migrated"] = {
                    "n": 1,
                    "ratio_pct": [self_sol["migrated_pct"]] * 2,
                    "meeting": 0,
                }
                rec["migrated_source"] = {
                    "kind": "op_level_roofline",
                    "path": "examples/TileOPs/reports/optimize/GatedDeltaNetPrefillFwdOp/OPTIMIZE_REPORT.md §3",
                    "basis": self_sol["note"],
                }
            else:
                sm["optimized"] = {
                    "n": dl["adjustable"],
                    "ratio_pct": dl["ratio_pct"],
                    "meeting": dl["meeting"],
                }
            rec["correctness"] = dl.get("correctness")
            rec["effective_kernel"] = dl.get("kernel")
            if dl.get("note"):
                rec["note"] = dl["note"]
            if dl.get("perf_note"):
                rec["perf_note"] = dl["perf_note"]

        # 迁移态补充（机器记录里没有 baseline 的算子）：只补缺口，不覆盖交付态
        supp = migrated_supp.get(op)
        if supp and "migrated" not in rec:
            copied = False
            if supp.get("alias_of") == "optimized" and not rec.get("optimized"):
                sm2 = rec.setdefault("summary", {})
                opt = sm2.get("optimized") or {}
                if opt.get("ratio_pct"):
                    sm2["migrated"] = {
                        "n": opt.get("n"),
                        "ratio_pct": list(opt["ratio_pct"]),
                        "meeting": opt.get("meeting", 0),
                    }
                    rec["migrated"] = []
                    copied = True
            if not copied:
                if supp.get("alias_of") == "optimized":
                    cases = [dict(c) for c in rec.get("optimized", [])]
                else:
                    cases = [dict(c) for c in (supp.get("cases") or [])]
                tunable = _tunable(cases)
                rec["migrated"] = tunable
                rec.setdefault("summary", {})["migrated"] = _summary(
                    tunable, TARGET[op]
                )
            rec["migrated_source"] = {
                "kind": supp.get("kind"),
                "path": supp.get("path"),
                "basis": supp.get("basis"),
            }
        ops[op] = rec
    return {"caliber": CALIBER, "operators": ops}


def _fmt(v) -> str:
    if not v:
        return "—"
    lo, hi = v
    return f"{lo:.1f}%" if abs(hi - lo) < 0.05 else f"{lo:.1f}–{hi:.1f}%"


def render_table(doc: dict) -> str:
    rows = [
        f"{'算子':<26}{'可调':<6}{'迁移态':<14}{'优化态':<14}{'达标线':<8}{'达标':<8}{'来源'}",
        "-" * 96,
    ]
    for op in ORDER:
        r = doc["operators"][op]
        if r["status"] != "delivered":
            rows.append(f"{op:<26}{'':<6}{'':<14}{'':<14}{'':<8}{'':<8}{r['status']}")
            continue
        sm = r.get("summary", {})
        ov = sm.get("optimized", {})
        adj = sm.get("adjustable_shapes") or ov.get("n", "?")
        rows.append(
            f"{op:<26}{adj:<6}{_fmt(sm.get('migrated', {}).get('ratio_pct')):<14}"
            f"{_fmt(ov.get('ratio_pct')):<14}{r['target_pct']:<8}"
            f"{str(ov.get('meeting', '?')) + '/' + str(adj):<8}{r['source']['kind']}"
        )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="fail if canonical.json is stale"
    )
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    doc = build()
    if args.table:
        print(render_table(doc))
        return 0
    text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current != text:
            print(
                "perf_data/canonical.json is stale -- run gen_perf_data.py",
                file=sys.stderr,
            )
            return 1
        print("perf_data/canonical.json is up to date")
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
