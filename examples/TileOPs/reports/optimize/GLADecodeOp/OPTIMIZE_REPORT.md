# 优化报告：GLADecodeOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `GLADecodeOp`；单 kernel / 单 func；Vector（AIV）；语义 `single-token gated linear attention decode`（DK=DV=128，bf16 输入，fp32 累加中转） |
| kernel（基准，只读） | `tileops/kernels/linear_attention/gla_decode/gla_decode_kernel/_gla_decode_kernel.py`（k_tile 默认 16） |
| 调优产物 | `.../gla_decode_kernel/perf_opt/GLADecodeOp.py`（k_tile 默认 32 + persistent 分核 `TUNED_NCORE=40`；round 4 采纳） |
| 口径 | `msprof op Task Duration(us)`（唯一时延口径）+ `--aic-metrics=Default,Roofline` 的 GM Read+Write SOL；kernel-name 用 transit 变体名 `_gla_decode_transit_main` |
| **工具链版本戳** | tilelang `0.1.2` dev root（`7a4f77e184`）/ **CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| 达标线 | 访存受限 **≥70%**（AI≈3.64 Ops/Byte，`AI×GM_bw=6.4TOps/s` 远低于 Vector 算力） |
| benchmark | `benchmarks/ops/bench_gla_decode.py::test_gla_decode_manifest_bench`（5 tune + 2 smoke） |
| 过程日志 | `perf_opt/opt_log.md`、`perf_records.jsonl`（round 0–4，共 97 行）、`profiles/`、`run_experiments.py` |
| 前置报告 | 本算子为 direct-implementation 迁移，无 v0 优化报告 |

## 1. 结论

- 一句话：两个有效杠杆——**k_tile 16→32**（串行 K-tile 循环 8→4 次）+ **persistent 分核**（超发 shape 把 block 数从 B×H 收到物理核数、欠订 shape 保持 flat），5 个 serving workload 相对 Stage-3 基线提速 **24.6–35.2%**；SOL 由 2.52–13.45% 升到 3.24–20.25%，但仍远低于 70% 线，根因是单 token decode 的 latency-bound（并行度 = B×H、逐头 state 串行递归 ~5.3–6.8µs/头且与核数无关）。
- **达标数**：全 shape **0/5**、可调 shape **0/5**、豁免 **0**（5 个 tune workload 均 ≥6.8µs，不触发微型 shape 豁免；2 个 smoke 仅精度回归，不计入性能分母）。
- 关键收益（Task Duration）：

| workload | base(k16) µs | final(k32) µs | final(k32+persistent) µs | 相对 base | 相对上轮 k32 |
|---|---:|---:|---:|---:|---:|
| serving-b1-h8-bf16 | 9.01 | 6.82 | 6.79 | −24.6% | −0.4%（A/B tie 带内） |
| serving-b1-h32-bf16 | 11.03 | 8.63 | 8.22 | −25.5% | −4.8% |
| serving-b1-h64-bf16 | 18.77 | 14.71 | 12.53 | −33.2% | −14.8% |
| serving-b8-h32-bf16 | 57.59 | 41.31 | 38.41 | −33.3% | −7.0% |
| serving-b8-h64-bf16 | 106.31 | 76.44 | 68.92 | −35.2% | −9.8% |

- **是否建议采纳 wrapper**：**是，且已采纳**——`gla_decode.py` 已 import `perf_opt.GLADecodeOp`；round 3 端到端 report `20261018_000805_527785_GLADecodeOp` 正确性 11/11、`status=passed`；round 4（persistent）端到端 report 见 `reports/tileops/20261018_215742_239583_GLADecodeOp/`（正确性与 benchmark 结果见 §3b.3）。

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
| 3 | persist40 | `T.Kernel(min(BH,40))` + 核内 grid-stride serial | 超发 shape −6.1~−10.9%；欠订 shape A/B tie | ✅ 采纳（作为超发档形态） |
| 3 | persist32 | 同上，ncore=32（整除数） | b1-h64 **−15.5%**；b8-h32/b8-h64 因 32/40 核容量损失落败 | ⚠️ 仅 `BH=64` 档采纳 |
| 3 | persist20 | 同上，ncore=20 | b8-h64 **+71%**（131.7µs） | ❌ 否决（核数不足） |
| 3 | pipe2 | persistent + `T.Pipelined(2)` | 全部落在噪声内（68.75 vs 68.61µs 等） | ❌ 否决（family_no_gain） |
| 3 | dispatch | 编译期分派：`BH≤40` 原 flat 体 / 否则 persistent + 2-iter 整除规则 | 3 个超发 shape −7.4~−14.3%，2 个欠订 shape 0 回退 | ✅ 采纳为最终版本 |
| — | transit 路径消除 | bf16/fp16 直接运算 | 未执行：bf16 无 Vector 算术，fp32 累加是精度要求 | ❌ 不可行 |
| — | 每核 2 头交错 | 2 头 interleave 复用 1 条依赖链 | 未执行：UB 192KB 只允许 k_tile=16+2 头，而 k_tile=16 单头已慢 25%，估算净收益 <5% | ❌ 未执行（<噪声） |
| — | DV 切分 | 按 DV 列分核提升欠订并行度 | 未执行：不缩短 K-tile DMA 往返关键路径，且与 persistent 方向相反 | ❌ 未执行 |

## 3. 优化结果（msprof 对比表）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---:|---:|---:|---:|---:|:---:|
| serving-b1-h8-bf16 | 9.01 | 6.82 | −24.3% | 2.52 | 3.18 | ❌ |
| serving-b1-h32-bf16 | 11.03 | 8.63 | −21.8% | 8.23 | 10.15 | ❌ |
| serving-b1-h64-bf16 | 18.77 | 14.71 | −21.6% | 9.53 | 11.68 | ❌ |
| serving-b8-h32-bf16 | 57.59 | 41.31 | −28.3% | 12.30 | 16.77 | ❌ |
| serving-b8-h64-bf16 | 106.31 | 76.44 | −28.1% | 13.45 | 18.25 | ❌ |

> 口径：相邻交错、各 1 次独立 `msprof op`；final 与 `perf_records.jsonl` `phase=final` round 3 一致，全部来自同一候选 `winner_k32`。9.0 端到端 report 的 SOL 范围 3.16–18.24%，与上表一致（±0.1pp 采样差）。

> 上表为 **round 1–2（参数级 k_tile 16→32）** 的结论，保留不变；round 3（persistent 分核）见 §3b。

## 3b. 第 3 轮增量：persistent 分核（round 4）

### 3b.1 诊断（为什么这一轮改分核而不是改参数）

| 观测 | 数值 | 推论 |
|---|---|---|
| 并行度 | 每 (b,h) 一个 block，block_dim = B×H = 8/32/64/256/512 | 40 AIV 下：8、32 欠订；64=1.6 wave；256=6.4 wave；512=12.8 wave |
| wave 空槽 | 64→2 wave（尾 wave 只 24 核）；256→7（24 槽空）；512→13（8 槽空） | 超发 shape 每 wave 都要重走 block 进出 + UB 装配 |
| 单头时延 | ~5.3–6.8µs/头，**几乎不随核数变化**（b1-h8 单头 6.8µs，b8-h64 512 头/40 核 = 68.4µs ⇒ 5.3µs/头） | 单头内部串行、无并行；只能跨头并行 |
| advice | 全 workload `latency bound:pipeline caused` | 需判别：wave 调度开销 vs 搬运/计算未重叠 |

### 3b.2 候选 vs current best（`msprof op Task Duration(us)` 2 rep 中位数）

| workload | current best k32 µs | persist40 | persist32 | pipe2 | dispatch（采纳形态） | block_dim | GM SOL%（k32→dispatch） |
|---|---:|---:|---:|---:|---:|:---:|---:|
| serving-b1-h8（8） | 6.87 | 6.88 | 6.81 | 6.82 | 6.86 | 8 → 8 | 3.18 → 3.08 |
| serving-b1-h32（32） | 8.35 | 8.45 | 8.63 | – | 8.68（A/B tie +0.94%） | 32 → 32 | 10.15 → 10.44 |
| serving-b1-h64（64） | 14.80 | 13.89 | **12.50** | 14.02 | **12.68** | 64 → 32 | 11.68 → 13.46 |
| serving-b8-h32（256） | 41.73 | **38.54** | 39.69 | 38.60 | **38.84** | 256 → 40 | 16.77 → 17.79 |
| serving-b8-h64（512） | 77.04 | **68.61** | 75.85 | 68.75 | **68.44** | 512 → 40 | 18.25 → 20.17 |

- **采纳形态**：编译期分派——`B*H ≤ 40` 走原 flat 体（源码逐字节相同，A/B 判 tie，回退 0）；否则 persistent grid-stride，且 `ceil(BH/40)==2` 时取 `BH` 在 `[ceil(BH/2),40]` 内的最大整除核数（b1-h64 → 32 核 × 2 iter，比 40 核 13.89µs 再快 10%）。
- **否决**：`persist20` 在 b8-h64 +71%（核数不足）；`pipe2` 全噪声内（见 §7 ceiling 证据）。
- **过程中发现的真 bug（已修）**：persistent 形态套在 **fp32 直拷路径**（`T.copy(GM→fp32 UB)`，无 vcast 中转）上会**静默算错**（B8-H32-D128 fp32 实测 max diff 5.12 / 43.6，同 shape 的 flat 与 bf16 transit 路径均正确）。最终版本把 persistent **只作用于 transit（sub-fp32）路径**，fp32 永远走验证过的 flat 体；5 个 serving workload 全 bf16，性能不受影响。该结论已写入 opt_log 复盘（供 pattern-library 收 D 类条目）。注：`perf_opt/GLADecodeOp_round3_persist.py` / `..._round3_pipe.py` 是**修复前的实验分支字节**（保留用于复现该 bug 与对应测量记录），只有最终 `GLADecodeOp.py`（= `..._round3_dispatch.py`）带修复。
- **记录纪律**：round 3 的 5 条 `phase=final` 记录已 supersede 为 `phase=merged`（数值保留，未删除），以使 gate 4 「每 workload 恰一条当前 final」成立。

### 3b.3 端到端 report（采纳前门禁）

- `python3 -m tileops.reporting.cli run --op GLADecodeOp --prof-mode msprof` → 最终字节上的 run_id **`20261018_215742_239583_GLADecodeOp`**（前一次 `20261018_214543_514969` 测的是修 fp32 分支前的字节，已被取代）。
- 正确性 **11/11（100%）**、`status=passed`、0 failed（smoke 允许 skip，无 failed）。
- 该 report（wrapper 路径、单次采样）逐 workload 时延与 §3b.2 dispatch 列一致（±3% 采样差）：

| workload | round-3 report µs（k32） | round-4 report µs（k32+persistent） | Δ | Ratio%（GM SOL） |
|---|---:|---:|---:|---:|
| serving-b1-h8 | 6.86 | 7.00 | +2.0%（flat 同码，双态噪声） | 3.10 |
| serving-b1-h32 | 8.52 | 8.34 | −2.1%（flat 同码，噪声） | 10.39 |
| serving-b1-h64 | 14.90 | 12.50 | −16.1% | 13.87 |
| serving-b8-h32 | 41.56 | 37.88 | −8.9% | 18.29 |
| serving-b8-h64 | 76.04 | 68.72 | −9.6% | 20.16 |
| smoke-b2-h4-d64 | 3.80 | 4.06 | 精度通过（不采性能结论） | 1.55 |


## 4. 精度

- `--level all`：L0/L1/L2/boundary **9 案例全过**，`precision_pass=true`（round 4 的最终文件同样全过，含 persistent 路径覆盖的 b8-h32=256 头/7 iter 案例与 `B*H ≤ 40` 的 flat 案例）。
- 2 个 smoke（`smoke-b2-h4-d64-float16/bfloat16`）仅做精度回归，不采性能。
- 关键边界：bf16 输入、fp32 累加中转；persistent 只改「哪个核处理哪个 (b,h)」，数学语义与累加序不变（每头仍在自己的 fp32 序列上累加）。
- 9.0 端到端 report：round 3 `20261018_000805_527785` 正确性 11/11；round 4 `20261018_215742_239583`（最终字节）正确性 **11/11**、status=passed。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 起止 | round 1–2：2026-10-17 16:15 UTC → 2026-10-17 23:08 UTC；round 3：2026-10-18 20:40 → 2026-10-18 22:00 |
| 墙钟 | round 1–2 **24,794 s（≈6.9h）**（含会话中断/恢复）；round 3 ≈1h20m 设备时间（5 批 msprof 扫描 + final + 端到端 report） |
| token（input/output/reasoning） | 原文未记录 |
| cost | 原文未记录 |

## 6. 产物

| 项 | 路径/状态 |
|---|---|
| final kernel | `perf_opt/GLADecodeOp.py`（k_tile 默认 32 + persistent 分核 `TUNED_NCORE=40`） |
| 实验分支 | `perf_opt/GLADecodeOp_round3_persist.py`（persist）、`..._round3_pipe.py`（Pipelined 探针）、`..._round3_dispatch.py`（= 最终形态） |
| 调优记录 | `perf_opt/opt_log.md`、`perf_records.jsonl`（round 0–4，97 行；含 `phase=merged` 的 supersede 行）、`profiles/` |
| gate 4 | round 1–2：`pass`（`stage_status[4]=completed`，`phase=DONE`）；round 3 由主会话统一跑 gate/状态 |
| 算子状态 | `DONE`（optimize） |
| wrapper 切换 | **已切** `perf_opt.GLADecodeOp` |
| 端到端 report | round 3 `reports/tileops/20261018_000805_527785_GLADecodeOp/`（passed，11/11）；round 4（最终字节）`reports/tileops/20261018_215742_239583_GLADecodeOp/`（passed，11/11，Ratio 3.10–20.16%） |
| pattern-library 回写 | 由 evolver 蒸馏：CONST-capacity-910B3 UB=192KB + K-tile 启发式反例、CASE-linear-attention-gla-decode |

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：无（5 个 tune workload 均非微型；2 个 smoke 仅精度，不计入分母）。
- **ceiling 证据（round 3 加厚）**：全部 workload `advice=latency bound:pipeline caused`；并行度 = B×H（8–512），40 AIV；逐头 state 递归串行，**单头 ~5.3–6.8µs 且几乎不随核数变化**；k_tile=32 已到 UB 上限（k_tile=64 需 294KB+）；wave 调度开销已在本轮吃掉（block_dim 512→40/−10.9%、256→40/−7.3%、64→32/−14.3%）；`T.Pipelined(num_stages=2)` 实测**零收益**（68.75 vs 68.61µs）⇒ advice 的 "pipeline caused" 指向的是 wave/依赖链而非 copy-compute 未重叠。
- **贴线风险**：无（最高 20.16%，距 70% 线很远）。
- **能力缺口**：无新增；transit 消除受 bf16 无 Vector 算术限制；persistent 分核已实现（编译期分派，无需运行时分派）。
- **不产出 [DESIGN_LIMIT]**：触发条件「结构性加速 >2×」无实测/文献依据支撑。

## 8. 建议

1. **维持采纳**：k_tile=32 + persistent 分核（`B*H > TUNED_NCORE` 时），5/5 workload 无回退，3/5 超发 shape 再获 −6.9~−14.3%。
2. **persistent 适用结论（回答"哪些 shape 适合"）**：以 `B*H > 40` 为门槛——`B*H ∈ (40,80]`（单 token 多头的 b1-h64 档）用「2-iter 整除核数」规则；`B*H ≥ 96`（大 batch serving）直接用满 40 核；`B*H ≤ 40`（b1-h8/b1-h32 类欠订）保持 flat，不要 persistent。
3. **升级工具链后复核**：k_tile=32 的 UB 容量结论绑定 CANN 9.0.0；若 UB 容量或 BishengIR 膨胀系数变化，需重测 k_tile=32/64 与 persistent 核数曲线。
4. **未解决项**：单 token decode 的 SOL 天花板（~20%）属算法/并行度结构限制；下一步若继续攻，唯一有机制依据的方向是「每核 2 头交错」（需 k_tile=16 + 2 头的工作集 147KB），但按现有单头时延估算净收益 <5%，需先做 k_tile=16 双头原型实测再定。
