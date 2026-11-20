# 优化报告：GeluAndMulFwdOp（Stage-4 optimize）

> **口径（2026-10-20 修订）**：本文件是**过程记录**——第 3 节的逐轮测量沿用**当时工具链**，**不参与交付口径**。交付数据一律取 `perf_data/canonical.json`（CANN 9.0.0，每算子恰好一条记录），其中同时给出**迁移态（优化前）**与**交付态（当前交付）**两列；本文数字与之不一致时以 SSOT 为准。同 case 的前后对比见 `examples/TileOPs/reports/迁移与优化总报告.md` 的汇总表。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `GeluAndMulFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = gelu(x[:,:N]) * x[:,N:]`（精确 erf GELU） |
| kernel（基准，只读） | `tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/`（`gelu_and_mul_fwd_kernel.py`：deg-20 多项式 + NaN 恢复） |
| 调优产物 | `.../gelu_and_mul_kernel/perf_opt/GeluAndMulFwdOp.py`（winner = deg-10 + vsub） |
| 口径 | `msprof op --kernel-name=main` 的 **Task Duration(us)**（唯一时延口径）+ `visualize_data.bin` GM Read+Write SOL；数据来自 `perf_opt/perf_records.jsonl`（final 行） |
| **工具链版本戳** | tilelang 0.1.2 dev-root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0）/ torch 2.7.1 / torch_npu 2.7.1.post4 / **910B3**（20 AIC / 40 AIV）/ 2026-10-18 |
| 达标线 | 原文按**判据失配改判向量利用率**：算子 compute-bound（`aiv_vec_ratio=0.943`），SOL 60% 目标不适用；硬底线 = Task Duration 必须变短 |
| benchmark | `benchmarks/ops/bench_gelu_and_mul.py::test_gelu_and_mul_manifest_bench`（5 tune + 3 smoke） |
| 过程日志 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`（29 条）、`workload_inventory.json`、`profiles/`、`logs/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md` |

## 1. 结论

- 一句话：**主杠杆是「dtype 分化多项式阶数」（fp16/bf16 用 deg-10 minimax 替代 deg-20），把 Horner 从 40 条向量 op 减到 20 条，prefill 打到 −43.6% / −43.1%；再叠加 loop-hoisted 常数 1 + 单条 vsub 的子阈值改进 −2.2%。**
- **达标数（双分母）**：全 shape = **4/5 达标 + 1 豁免**（decode 微型豁免，不计分母）；可调 shape = **4/4 达标**（prefill fp16/bf16 + swiglu fp16/bf16，Task Duration 全部显著下降）；豁免 **1**，依据：decode bf16 Task Duration 3.00µs 属微型 shape。
- 硬底线「Task Duration 必须变短」满足：prefill fp16 **877.20→494.38µs**。
- 关键收益：prefill fp16 **−43.6%**、prefill bf16 **−43.1%**、swiglu fp16 **−35.4%**、swiglu bf16 **−33.2%**。
- **SOL 60% 目标不适用**：算子为 compute-bound（`aiv_vec_ratio=0.943`，GM SOL 仅 ~47%），按用户口径豁免规则 3 改判向量利用率。
- **是否建议采纳 wrapper**：**是**——有净收益且无回退（decode 为微型豁免），精度全绿。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 瓶颈 pipe | `aiv_vec_ratio = 0.943`、`aiv_mte2_ratio = 0.246`、`aiv_scalar_ratio = 0.147`（prefill fp16，PipeUtilization） | **compute-bound**：Vector FP32 单元是唯一瓶颈（上一轮记 0.879，本轮复测 0.943，结论未变） |
| GM SOL | prefill GM Read+Write SOL ≈ 43–48%（表内 final 46.86 / 47.70） | **非正确判据**：实测向量利用率 ≥0.9 → 按豁免规则 3 改判向量利用率 |
| 链长构成 | fp16/bf16：2 vcast + erf 段 + gelu 链 + 1 vcast；主瓶颈 = deg-20 Horner erf（40 条 vmul/vadd） | 优化目标 = 减向量 op 数，而不是搬流量 |
| 精度需求 vs 实际阶数 | fp16/bf16 输出只需 ~1e-3 erf 精度（输出容差 1e-2 / 1.6e-2），远不需要 deg-20 的 2.98e-7 精度 | **deg-20 严重过剩** → 阶数是可动杠杆 |
| 残余链长（win 后） | deg-10 多项式 erf 在 fp32 域展开为 ~34 条向量 op | 已是该精度契约下的 op 数下限附近 |
| 设计估算 vs 实测 | DESIGN 估算 prefill 下界 ≈ 226µs（带宽口径），实测 baseline 877.2µs | **判据失配**：把 MTE2 带宽当地板，实为向量 FP32 打满（估算偏差来自设计假设，opt_log 记 ≈3.9×） |
| 噪声阈 | 3%（A/B tie 判据） | 子阈值改进（−2.2%）只作为结构改进采纳，不记为 >3% winner |

### 2.2 方法（做了什么、为什么这样做）

本轮在 CANN 9.0.0 重采 + 追加；上一轮（工具链升级前）已合入的「删除冗余 NaN 恢复」（−9.3%）作为起点（current best `nan_drop`）。

1. **T1：dtype 分化多项式阶数（主杠杆）**
   - 改了什么：fp16/bf16 用 **deg-10 minimax** 多项式替代 deg-20（系数 |c|≤309、fp32 求值、max erf err 1.4e-4），**fp32 保持 deg-20**；分支在 trace-time 按 dtype 展开（非运行时条件路径）。
   - 为什么有效：compute-bound 下时延 ∝ 向量 op 数；Horner 从 40 条 vmul/vadd 减到 20 条（整链 ~56 → ~36 量级），prefill 800.08→505.64µs（**−36.8%**），对 Stage-3 基线 **−42%**。
   - 正确性依据：fp16/bf16 结果最终要舍入回 fp16/bf16（容差 1e-2 / 1.6e-2），deg-10 的 erf err 1.4e-4 远小于容差；`--level all` 全绿（L0/L1/L2/Boundary），max_diff fp32 2.86e-6 / fp16 7.81e-3 / bf16 1.25e-1。
   - 溯源：该方向是 DESIGN §5.4 显式列出的 Stage 4 实验项（「窄 dtype 走 deg10 多项式」），属执行既定优化，非设计冻结天花板。

2. **T2：loop-hoisted 常数 1 + 单条 `vsub`**
   - 改了什么：`w = 1 − u²` 原先用 `vmul`-negate + `vadd` 两条，改为循环外 `vbrc(1.0)` 建常数 buffer + 单条 `T.vsub(one_ub, w, w)`。
   - 为什么有效：在 94% vec 占用的 compute-bound 下，少 1 条向量 op ≈ 2.8%（opt_log）；A/B 交错 3 对方向一致（−2.24%，`sign_test_p=0.25`，verdict=tie），故作为**子阈值结构改进**采纳，不记为 >3% winner。
   - 正确性依据：`1 − u²` 与 `vsub(1, u²)` 数学严格等价；L0 全绿，无精度风险。

**被否决的方向（保留）**：
- **deg-12（op12）**：比 deg-10 多 4 条 Horner op、erf err 9.7e-6 属精度过剩，实测 564.521µs（prefill fp16）比 deg-10 的 505.640µs **更慢** → ❌ 否决。
- **fold `1/√2` 进 erf 系数**：把 `1/√2`、`1/4` 折叠为 `1/(4√2)` 会**破坏 erf 的精确饱和**——原式 `c0=1/4` 与 `t=±4` 都是 fp32 精确值、`t·c0=±1` 精确，故 `gelu(-inf)` 为 NaN（L1-4/Boundary 通过）；折叠后 `c0'`、`t'` 非精确，`clip(±1)` 不再保证 ±1，`gelu(-inf)` 变 `-inf` → **L1-4/Boundary 全部失败** → ❌ 否决（`c0=1/A` 精确是不可妥协的设计约束）。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | `nan_drop`（上一轮，工具链升级前） | 删除冗余 NaN 恢复（−9.3%），本轮在 9.0.0 重采为起点 | prefill fp16 800.076µs / bf16 804.756µs / swiglu 13.850 / 13.910µs | ✅ 起点 current best |
| R1 | `op10`（deg-10） | fp16/bf16 用 deg-10 minimax，fp32 保持 deg-20 | prefill 505.640 / 510.370µs（**−36.8% / −36.6%**）、swiglu 9.750 / 10.040µs，L0 PASS | ✅ winner（主杠杆） |
| R1 | `op12`（deg-12） | fp16/bf16 用 deg-12 minimax | prefill 564.521 / 569.381µs、swiglu 10.740 / 10.810µs，L0 PASS | ❌ 否决：精度过剩、op 更多 → 更慢 |
| R2 | `fold` | 折叠 `z=gate/√2` 与 `u=t/4` 为 `clamp(gate,±4√2)·1/(4√2)` | **精度 FAIL**：破坏精确饱和，`gelu(-inf)` 变 `-inf`，L1-4/Boundary 失败 | ❌ 否决：`c0=1/A` 精确性不可破坏 |
| R2 | `vsub` | loop-hoisted 常数 1 buffer + 单条 `T.vsub`（替代 vmul-negate+vadd） | 494.210 / 498.990µs；A/B 3 对方向一致 −2.24%（p=0.25，tie），L0 PASS | ✅ 采纳（子阈值结构改进，不记 >3% winner） |
| R3 | merged | deg-10 + vsub | prefill 494.380 / 499.190µs、swiglu 9.670 / 9.900µs、decode 3.000µs，`--level all` 全绿 | ✅ final |

**判据/边界说明**：`fold` 属「测试失败」类否决（精度失败），`op12` 属「测试通过但更慢」类否决；`vsub` 属「方向一致但子阈值」类采纳——三类分开记录，不混为一谈。

## 3. 优化结果（msprof 对比表）

| workload | dtype | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---|---:|---:|---:|---:|---:|---|
| llama-8b-ffn-prefill | fp16 | 877.198 | **494.380** | **−43.6%** | 43.20 | 46.86 | ❌ |
| llama-8b-ffn-prefill | bf16 | 877.188 | **499.190** | **−43.1%** | 44.05 | 47.70 | ❌ |
| qwen3-8b-swiglu-b32 | fp16 | 14.980 | 9.670 | −35.4% | 34.41 | 32.88 | ❌ |
| qwen3-8b-swiglu-b32 | bf16 | 14.810 | 9.900 | −33.2% | 35.28 | 32.75 | ❌ |
| llama-8b-ffn-decode | bf16 | 3.860 | 3.000 | 微型豁免 | 4.89 | 3.67 | 豁免 |

smoke（16×128，fp32/fp16/bf16）：仅精度回归，不做性能采集。

**测量口径与噪声**：每点独立 `msprof op --kernel-name=main` 进程；base SOL% 由 `perf_records.jsonl` baseline 行回填；表中「是否达标」按 SOL≥60% 判定，时延改善见 Δ 列。decode 3.00µs ≤ 5µs → 微型 shape 豁免；swiglu-b32（9.67/9.90µs）属 small-shape（<20µs），差异按交错 A/B 纪律处理，winner 由 prefill 主判 −36.8%（远超 3% 噪声阈）决定。

## 4. 精度

- `python perf_opt/GeluAndMulFwdOp.py --level all` **全绿**（exit 0）：L0/L1/L2/Boundary 全过。
- 覆盖：erf 饱和（±50 / 1e4 / ±inf）、大负值（−20）、signed-zero、NaN 传播、L1-4 大 |gate| 全域（gate_scale 6/20）。
- max_diff 汇总：fp32 **2.86e-6**、fp16 **7.81e-3**（L1-4 大 gate）、bf16 **1.25e-1**（rtol 支配）。
- 唯一 WARN 为 L2-stress（scale-20 both halves）：设计 §9.1 明言 out-of-spec、非阻塞；deg-20 的 fp32 本身亦 WARN。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（UTC+8） | ≈ 2026-10-18 01:35 → 02:30 |
| 墙钟时长 | **≈ 55 min**（含等待前序算子 GeluTanhAndMul benchmark 释放 NPU ~10 min） |
| 实测阶段 | baseline(r0) + 2 候选(r1) + fold/vsub(r2) + A/B + final(r3) + L0 全量 ≈ 45 min |
| token input/output/reasoning | 原文未记录 |
| cache-read / cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `perf_opt/GeluAndMulFwdOp.py`（final kernel + golden + L0/L1/L2/Boundary + main --level） |
| wrapper 兼容同步 | `perf_opt/gelu_and_mul_fwd_kernel.py`（同内容同步，保持当前 wrapper 的 perf_opt import 可运行） |
| 调优记录 | `perf_opt/opt_log.md`（逐轮候选 vs current best 对比表 + Final Performance Test Data + Skill Retrospective） |
| 结构化记录 | `perf_opt/perf_records.jsonl`（29 条，signal-registry §5 字段契约；旧 schema 4 条归档于 `logs/legacy_perf_records_prior_run.jsonl`） |
| 采集工件 | `perf_opt/workload_inventory.json`、`profiles/`（raw msprof）、`logs/`、`run_experiments.py`、`bench_kernel.py`、`env.sh`、`experiments_r*.json`、实验分支 `*_round*.py` |
| 能力缺口 | 原文未记录 |
| gate 4 | 原文未记录 |
| 算子状态 | 原文未记录 |
| wrapper | 未翻转（建议采纳，由 conductor 执行） |
| 端到端 report | 原文未记录 |
| pattern-library 回写 | 原文未记录（VP-2026-0033/0034/0035 的 `repro/` 最小形态待回填，标 repro-missing） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项与依据**：
  1. **decode bf16**：Task Duration 3.00µs ≤ 5µs → 微型 shape 豁免。
  2. **swiglu-b32**（9.67/9.90µs）：small-shape（<20µs），差异按交错 A/B 纪律处理。
  3. **判据失配**：`aiv_vec_ratio=0.943 ≥ 0.9` → compute-bound，按豁免规则 3 改用向量利用率判据；硬底线是 Task Duration 变短（已满足）。
  4. **L2-stress WARN**：deg-10 的 erf err 1.4e-4 在 gate/value 同时 scale-20 时被 0.5·|gate|·|value| 放大到超 fp16 atol，属设计 §9.1 明言的 out-of-spec regime（recorded, not blocking）。
- **ceiling 证据（compute-bound 地板）**：`aiv_vec_ratio=0.943`（prefill fp16）——deg-10 多项式 erf 在 fp32 域展开为 ~34 条向量 op，把 Vector FP32 单元打满；GM SOL 仅 47% 非瓶颈。残余 op 数是「全 fp32 域 + 单次 cast + 精确 erf（无可用原生 erf）」精度契约下的 floor（fp16/bf16 的 deg-10 已达该 dtype 容差下的 op 数下限；fp32 仍需 deg-20）。
- **结构性加速估计 ~1.8×**（已实现的 deg-20→deg-10 + vsub）；进一步（deg-8 精度不足、tanh 近似语义错误、exp 型 minimax 引入 vexp/vdiv 更多 op）**远未达 2×** → 不触发 `[DESIGN_LIMIT]` 逆向反馈（perf-feedback 两项触发条件未同时满足）。
- **贴线风险**：无（判据已改判向量利用率，不存在 SOL 贴线项）。
- **缺口**：VP-2026-0033/0034/0035 的 `repro/` 最小形态待回填（时间受限，标 repro-missing）。

## 8. 建议

1. **采纳 `perf_opt/GeluAndMulFwdOp.py`**（翻转 wrapper 的 perf_opt import 块）：prefill 主判 −43.6% / −43.1%，swiglu −35.4% / −33.2%，L0–Boundary 全绿。
2. 后续动作：若工具链引入原生 erf 或 fp16 向量 erf 指令，需重新评估 deg 阶数下限（当前 ~34 条向量 op 的 floor 会移动）；工具链升级后复核 L2-stress（scale-20）档的 WARN 行为。
3. **未解决项与原因**：fp32 路径仍需 deg-20（精度契约），无法共享 deg-10 收益；VP-2026-0033/0034/0035 的 repro 尚未回填。
