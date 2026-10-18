# 优化报告：GLADecodeOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `GLADecodeOp`；单 kernel / 单 func；Vector（AIV）；语义 `single-token gated linear attention decode`（DK=DV=128，bf16 输入，fp32 累加中转） |
| kernel（基准，只读） | `tileops/kernels/linear_attention/gla_decode/gla_decode_kernel/_gla_decode_kernel.py`（k_tile 默认 16） |
| 调优产物 | `.../gla_decode_kernel/perf_opt/GLADecodeOp.py`（k_tile 默认 32） |
| 口径 | `msprof op Task Duration(us)`（唯一时延口径）+ `--aic-metrics=Default,Roofline` 的 GM Read+Write SOL；kernel-name 用 transit 变体名 `_gla_decode_transit_main` |
| **工具链版本戳** | tilelang `0.1.2` dev root（`7a4f77e184`）/ **CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| 达标线 | 访存受限 **≥70%**（AI≈3.64 Ops/Byte，`AI×GM_bw=6.4TOps/s` 远低于 Vector 算力） |
| benchmark | `benchmarks/ops/bench_gla_decode.py::test_gla_decode_manifest_bench`（5 tune + 2 smoke） |
| 过程日志 | `perf_opt/opt_log.md`、`perf_records.jsonl`（20 行）、`profiles/`、`run_experiments.py` |
| 前置报告 | 本算子为 direct-implementation 迁移，无 v0 优化报告 |

## 1. 结论

- 一句话：**唯一有效杠杆是 k_tile 16→32**（把串行 K-tile 循环从 8 次降到 4 次），5 个 serving workload 全部提速 **21.6–28.3%**；SOL 由 2.52–13.45% 升到 3.18–18.25%，但仍远低于 70% 线，根因是单 token decode 的 latency-bound（并行度 = B×H、逐头 state 串行递归）。
- **达标数**：全 shape **0/5**、可调 shape **0/5**、豁免 **0**（5 个 tune workload 均 ≥6.8µs，不触发微型 shape 豁免；2 个 smoke 仅精度回归，不计入性能分母）。
- 关键收益（Task Duration）：

| workload | base(k16) µs | final(k32) µs | 提速 |
|---|---:|---:|---:|
| serving-b1-h8-bf16 | 9.01 | 6.82 | 24.3% |
| serving-b1-h32-bf16 | 11.03 | 8.63 | 21.8% |
| serving-b1-h64-bf16 | 18.77 | 14.71 | 21.6% |
| serving-b8-h32-bf16 | 57.59 | 41.31 | 28.3% |
| serving-b8-h64-bf16 | 106.31 | 76.44 | 28.1% |

- **是否建议采纳 wrapper**：**是，且已采纳**——`gla_decode.py` 已 import `perf_opt.GLADecodeOp`；9.0 端到端 report `20261018_000805_527785_GLADecodeOp` 正确性 11/11、`status=passed`。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| AI | ≈3.64 Ops/Byte（GM Read+Write） | `AI×GM_bw(1.75TB/s)=6.4TOps/s` 远低于 Vector 算力 ⇒ 访存型，线 70% |
| 瓶颈 | `advice=latency bound:pipeline caused`（全 workload） | 不是带宽打满，而是并行度/流水填充不足 |
| 并行度 | 每 (b,h) 一个核；decode 阶段 B×H 有限（8~64） | 20 AIC / 40 AIV 下多数 workload 欠订 |
| 固定开销 | 逐头 state 递归 + 小 per-head 工作集 | 串行依赖限制流水深度 |
| 编译期参数 | k_tile 是唯一可调参数；旧 `_ceil_ktile_fit` 用 150KB 预算 | 实际 UB=192KB，k_tile=32 可编译，旧启发式过保守 |
| kernel-name | bf16 的 jitted 名为 `_gla_decode_transit_main`，不是 `main` | msprof 过滤必须用 transit 变体名 |

### 2.2 方法（做了什么、为什么这样做）

1. **k_tile 16 → 32**：`_ceil_ktile_fit` 的 UB 预算从经验值 150KB 改为实测 **192KB**（910B3 UB 容量），使 DK=DV=128 自动选 32。K-tile 循环 8→4 次，减少每 tile 的 reduce/barrier 与 loop 开销。
   - 正确性依据：纯 tiling 变化，数学语义不变；L0/L1/L2/boundary 9 案例全过。
   - 容量依据：162KB 手工 + ~1.10–1.12× BishengIR 膨胀 < 192KB；k_tile=64 需要 294KB+，超 UB，已否决。
2. **暴露模块常量 `TUNED_K_TILE=32`**：供 wrapper 切换块成对引用，避免 wrapper 与 perf_opt 配置漂移。
3. **补齐内嵌分层测试**：`run_L0/L1/L2/boundary` + `main --level`，满足 gate 4 的 `S4-TEST-LEVELS`。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| 1 | ktile8 | k_tile=8 | 5 workload 全部回退 +43~47% | ❌ 否决 |
| 1 | ktile32 | k_tile=32 | 5 workload 全部胜出 +23~27% | ✅ 采纳为 current best |
| 2 | winner_k32 | 合并 winner + 192KB 预算 + 测试 | final 复测 21.6~28.3% | ✅ 收束 |
| — | persistent 分核 | `grid=min(B*H,40)` | 未执行：超发 shape 理论 10–20%，欠订 shape 会反噬，需按 shape 分派 | ❌ 放弃（收益/风险不划算） |
| — | transit 路径消除 | bf16/fp16 直接运算 | 未执行：bf16 无 Vector 算术，fp32 累加是精度要求 | ❌ 不可行 |

## 3. 优化结果（msprof 对比表）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---:|---:|---:|---:|---:|:---:|
| serving-b1-h8-bf16 | 9.01 | 6.82 | −24.3% | 2.52 | 3.18 | ❌ |
| serving-b1-h32-bf16 | 11.03 | 8.63 | −21.8% | 8.23 | 10.15 | ❌ |
| serving-b1-h64-bf16 | 18.77 | 14.71 | −21.6% | 9.53 | 11.68 | ❌ |
| serving-b8-h32-bf16 | 57.59 | 41.31 | −28.3% | 12.30 | 16.77 | ❌ |
| serving-b8-h64-bf16 | 106.31 | 76.44 | −28.1% | 13.45 | 18.25 | ❌ |

> 口径：相邻交错、各 1 次独立 `msprof op`；final 与 `perf_records.jsonl` `phase=final` round 3 一致，全部来自同一候选 `winner_k32`。9.0 端到端 report 的 SOL 范围 3.16–18.24%，与上表一致（±0.1pp 采样差）。

## 4. 精度

- `--level all`：L0/L1/L2/boundary **9 案例全过**，`precision_pass=true`。
- 2 个 smoke（`smoke-b2-h4-d64-float16/bfloat16`）仅做精度回归，不采性能。
- 关键边界：bf16 输入、fp32 累加中转；具体 max_diff 原文未记录。
- 9.0 端到端 report：正确性 **11/11**。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 起止 | 2026-10-17 16:15 UTC → 2026-10-17 23:08 UTC |
| 墙钟 | **24,794 s（≈6.9h）**；含上一轮会话中断/恢复 |
| token（input/output/reasoning） | 原文未记录 |
| cost | 原文未记录 |

## 6. 产物

| 项 | 路径/状态 |
|---|---|
| final kernel | `perf_opt/GLADecodeOp.py`（k_tile 默认 32） |
| 调优记录 | `perf_opt/opt_log.md`、`perf_records.jsonl`（20 行）、`profiles/` |
| gate 4 | `pass`（`stage_status[4]=completed`，`phase=DONE`） |
| 算子状态 | `DONE`（optimize） |
| wrapper 切换 | **已切** `perf_opt.GLADecodeOp` |
| 端到端 report | `reports/tileops/20261018_000805_527785_GLADecodeOp/`（passed，11/11） |
| pattern-library 回写 | 由 evolver 蒸馏：CONST-capacity-910B3 UB=192KB + K-tile 启发式反例、CASE-linear-attention-gla-decode |

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：无（5 个 tune workload 均非微型；2 个 smoke 仅精度，不计入分母）。
- **ceiling 证据**：全部 workload `advice=latency bound:pipeline caused`；并行度 = B×H（8–64），40 AIV 下多数欠订；逐头 state 递归串行；k_tile=32 已到 UB 上限（k_tile=64 需 294KB+）。
- **贴线风险**：无（最高 18.25%，距 70% 线很远）。
- **能力缺口**：无新增；transit 消除受 bf16 无 Vector 算术限制，persistent 分核受 shape 分派复杂度限制。
- **不产出 [DESIGN_LIMIT]**：触发条件「结构性加速 >2×」无实测/文献依据支撑。

## 8. 建议

1. **维持采纳**：k_tile=32 是当前参数级最优，5/5 workload 正收益，无回退。
2. **后续可选**：persistent 分核 `grid=min(B*H,40)` 仅对超发 shape 有 10–20% 理论空间，需运行时 shape guard，收益/风险不划算；如要做，先按 `b8-h32`/`b8-h64` 两个超发 shape 做单点验证。
3. **升级工具链后复核**：k_tile=32 的 UB 容量结论绑定 CANN 9.0.0；若 UB 容量或 BishengIR 膨胀系数变化，需重测 k_tile=32/64。
4. **未解决项**：单 token decode 的 SOL 天花板（~18%）属算法/并行度结构限制，非 kernel 参数可解。
