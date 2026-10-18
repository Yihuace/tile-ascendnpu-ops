# 优化报告：SubFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `SubFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = a - alpha * b` |
| kernel（基准，只读） | `tileops/kernels/elementwise/sub/sub_kernel/sub_fwd_kernel.py` |
| 调优产物 | `.../sub_kernel/perf_opt/SubFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op --kernel-name=main` 的 `Task Duration(us)` + `visualize_data.bin` 的 GM Read+Write SOL ratio；直跑 kernel（`perf_opt/bench_kernel.py`），不经 wrapper |
| **工具链版本戳** | tilelang `0.1.2` dev root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir-compile 1.1.0 + hivmc 0.2.0）/ torch 2.7.1 / torch_npu 2.7.1.post4 / Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| 达标线 | 访存受限 **≥70% SOL**（原文按逐元素/访存受限 70% 线判定；AI 数值原文未记录） |
| benchmark | `benchmarks/ops/bench_sub.py::test_sub_manifest_bench`（15 tune + 1 smoke） |
| 过程日志 | `.../sub_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/`；实验清单 `experiments_sub_r1.json` / `experiments_sub_final.json` |
| 前置报告 | 本文件 v0 归档 `history_version/OPTIMIZE_REPORT_v0.md`（内容相同，仅结构未归一） |

## 1. 结论

- 一句话：把 AddFwdOp 的 Stage-4 winner 结构同构移植到 Sub（满块静态 extent + 单尾块 epilogue +
  `T.Pipelined(num_stages=2)` + grid-stride 40 核），再叠加 **fp16(alpha==1) 原生 vsub** 快路径，
  **15 个 tune case 中 14 个 ≥70% SOL**。
- **达标数**：全 shape **14/15**、可调 shape **14/15**、豁免 **0**（15 个 tune case 原文均未声明豁免）。
- 关键收益：
  - 两个此前未过的 case 现在稳过：`linear-bias fp16` **73.97%**（12.96µs，−6.76%）、
    `llama-7b-hidden fp16` **73.29%**（12.99µs，−7.08%）；
  - 全部 15 case 时延和 **506.151 → 504.791 µs（−0.27%）**；其余稳定收益 `phi-2-ffn-bias −2.72%`、
    `phi-2-ffn −2.03%`、`llama-7b-ffn-bias fp16 −1.55%`。
- 唯一未过 `linear-bias bf16` 66.76%：该 case 的 bf16 代码路径与 baseline 完全相同（A6 只改 fp16），
  5 对交错 A/B 显示 tuned **14.000→13.770µs（快 1.64%，verdict=tie）** ⇒ 官方值是跨轮漂移样本，不是回退。
- 方法论修正：首轮 baseline 处于慢态，曾把 `llama-7b-hidden` 伪造成 −13.5%；相邻交错复测只剩 −0.6%。
- **是否建议采纳 wrapper**：原文未记录明确建议；原文记「基准 `sub_fwd_kernel.py` 与 wrapper 未改」
  （即 wrapper 切换块未翻，仍 import baseline）。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 基线达标分布 | 14/15 ≥70%；唯一线下 `llama-7b-hidden fp16` **68.09%**（N=4.19M，最小 shape） | 目标是让最小/贴线 shape 跨线，不必全局重构 |
| 算子族同构性 | 与 Add 同一 `AlphaScaledBinary` fp32 中转链，仅 `vadd→vsub`；DESIGN 分块/UB 预算一致 | 采用「已证结构迁移」而非从零枚举参数 |
| 计算链 | fp32 中转链（`vcast→vsub→vcast`，4 条向量 pass） | `alpha==1` 且 fp16 时存在语义等价快路径（单次 vsub） |
| 跨进程噪声 | 单进程双态 **±5%**（fp16 13.4~14.9 µs 之间摆动） | <5% 差异必须相邻交错或 A/B，否则会把机器双态误读成优化收益 |
| 首轮 baseline 状态 | R1 baseline 处于慢态：`llama-hidden` 15.68µs vs 同配置 final 13.98µs（~12%） | R1 的大收益不可直接采信，必须复测 |

### 2.2 方法（做了什么、为什么这样做）

1. **满块静态 extent**：`num_full = N // block_size` 的热循环用编译期常量边界，去掉每 tile 的
   `T.min` 与动态 extent（省标量开销、利于向量化）。
2. **单尾块 epilogue**：尾块单独处理一次，不进热循环（热循环保持纯静态）。
3. **`T.Pipelined(num_local_tasks, num_stages=2)`**：用下一 tile 的 MTE2 装载覆盖当前 tile 的 VEC 链，
   把访存与计算重叠。
4. **grid-stride 40 核**：20 AIC / 40 AIV 下按 AIV 数铺满，避免核间负载不均。
5. **`TUNED_BLOCK_SIZE = 8192` 提为模块级常量**：供 wrapper 成对引用。
   以上 1–5 为结构杠杆，预期收益 1–3%，单独不足以稳定跨 70% 线。
6. **A6：fp16 原生 vsub**（`dtype=="float16" and alpha==1` 时直接 `T.vsub(a_ub, b_ub, y_ub)`）：
   向量链由 4 条 pass 压到 1 条。
   - 为什么有效：省掉两次 `vcast` 与一次 fp32 域运算，减少向量 pipe 占用与 UB 流量。
   - 正确性依据：两个 fp16 的**精确差**可被 fp32 精确表示 ⇒「fp32 算完舍回」与「原生 fp16 vsub」
     都是单次正确舍入，**逐位相等**；`--level all` **63 用例全 PASS，fp16 max_diff=0**。
7. **bf16 保持 fp32 中转**（不启用原生 bf16 vsub）：原生版虽可编译且 max_diff=0，但性能方向不一致
   （Add 上探针 −2%、官方 +3.8%），不采纳（见 2.3）。
   产物只落 `perf_opt/`；基准 kernel 与 wrapper 未动。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | 结构移植（A4） | 满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核 | 首轮出现 `llama-7b-hidden fp16` −13.5%、`phi-2-ffn-bias` −8.2% 等大收益；14/15 候选 ≥70%（仅 linear-bias fp16 69.67%） | ⚠️ 暂不采信：baseline 明显慢态，转 A/B 复核 |
| A/B | A4 vs baseline | 5 对/例，交替先后，launch-count=20 | `hidden-state-prefill bf16` 25.891→25.051（−3.2%，推翻 R1 的 +5.8%）；`cnn-feat-broadcast fp32` 71.011→72.521（+2.1%）；`linear-bias fp16` 13.880→13.880（tie） | ✅ R1 大收益判为噪声假象；`cnn fp32 +2.1%` 记噪声，不做 shape 特判 |
| R2 | 官方 final 全量复测 | baseline/final 相邻交错、各 1 次独立 msprof | R1 收益回落到噪声内（`llama-hidden` 15.68→13.56 的 −13.5% 在复测中为 13.98→13.90，即 −0.6%）；真实收益 `llama-7b-ffn fp16 −3.06%`、`llama-7b-ffn-bias fp16 −2.77%`、`linear-bias fp32 −2.33%` | ✅ 采纳为 current best（无 >5% 可信回退） |
| R3（A6） | fp16 原生 vsub | `dtype=="float16" and alpha==1` 走 `T.vsub`，4 条向量 pass → 1 条 | `linear-bias fp16` 13.90→12.96µs（−6.76%）、`llama-7b-hidden fp16` 13.98→12.99µs（−7.08%），两个此前未过的 case 跨线 | ✅ 采纳 |
| 探针 | bf16 原生 vsub | bf16 直接 `T.vsub` | 可编译、max_diff=0，但性能方向不一致 | ❌ 否决：保留 fp32 中转（其代码路径与 baseline 相同） |
| 复核 | 合并版 final（round 5） | 全量 15 case | `linear-bias bf16` 官方值 13.41→14.08µs（66.76%），但 A/B tuned 快 1.64%（verdict=tie） | 判为漂移样本，不改 kernel、不做 shape 特判 |

## 3. 优化结果（msprof，baseline round 2 / final round 5）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 36.971 | 37.151 | +0.49% | 78.50 | 79.09 | ✅ |
| cnn-feat-broadcast-float16 | 36.841 | 37.091 | +0.68% | 79.78 | 78.80 | ✅ |
| cnn-feat-broadcast-float32 | 70.651 | 71.341 | +0.98% | 82.96 | 83.64 | ✅ |
| hidden-state-prefill-bfloat16 | 25.111 | 25.091 | -0.08% | 76.76 | 75.61 | ✅ |
| hidden-state-prefill-float32 | 46.561 | 46.971 | +0.88% | 83.40 | 83.91 | ✅ |
| linear-bias-bfloat16 | 13.410 | 14.080 | +5.00% | 71.44 | 66.76 | ❌ |
| linear-bias-float16 | 13.900 | 12.960 | -6.76% | 71.12 | 73.97 | ✅ |
| linear-bias-float32 | 24.900 | 24.580 | -1.29% | 76.69 | 76.33 | ✅ |
| llama-7b-ffn-bfloat16 | 32.711 | 32.851 | +0.43% | 79.18 | 78.69 | ✅ |
| llama-7b-ffn-bias-float16 | 33.621 | 33.101 | -1.55% | 77.09 | 79.13 | ✅ |
| llama-7b-ffn-float16 | 33.621 | 33.941 | +0.95% | 75.42 | 77.79 | ✅ |
| llama-7b-ffn-float32 | 62.881 | 63.101 | +0.35% | 82.39 | 84.72 | ✅ |
| llama-7b-hidden-float16 | 13.980 | 12.990 | -7.08% | 68.09 | 73.29 | ✅ |
| phi-2-ffn-bias-float16 | 30.511 | 29.681 | -2.72% | 78.58 | 79.52 | ✅ |
| phi-2-ffn-float16 | 30.481 | 29.861 | -2.03% | 77.46 | 79.73 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

**测量口径**：baseline 取 round 2 同会话记录，final 为 round 5 独立 msprof（20:12）；round 2 的 15 条
final 已降级为 `phase=merged` 保留，未删除。

**噪声裁定（单列）**：

| case | base µs | 候选 µs | 判定 |
|---|---:|---:|---|
| hidden-state-prefill bf16 | 25.891 | 25.051（−3.2%） | inconclusive（推翻 R1 的 +5.8%） |
| cnn-feat-broadcast fp32 | 71.011 | 72.521（+2.1%） | inconclusive（噪声，不做特判） |
| linear-bias fp16 | 13.880 | 13.880 | tie |
| linear-bias bf16 | 14.000 | 13.770（−1.6%） | tie（官方 66.76% 是漂移） |

## 4. 精度

- `python perf_opt/SubFwdOp.py --level all` → **L0 / L1 / L2 / Boundary 全 PASS**（`All check passed!`，
  max_diff 最大 **9.537e-07**）。
- `python perf_opt/validate_final.py SubFwdOp.py --block-size 8192` → smoke
  `hidden-state-prefill fp16`（N=8,388,608）PASS；`linear-bias` fp16/bf16 PASS，**max_diff=0**。
- A6 依据：fp16 精确差可被 fp32 精确表示 → 原生 vsub 与 fp32 中转同为单次正确舍入（逐位相等）。
- 边界用例逐项清单（NaN/±inf/−0/subnormal）：原文未记录（仅记 Boundary 全 PASS）。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 17:48 → 20:35 |
| 墙钟时长 | **≈ 167 min**（含 A6 原生路径与 bf16 证伪、两次 final 复测） |
| token input/output/reasoning | 88,528 / 75,076 / 50,407 |
| cache-read / cost | 78,880,000 / **$0.3252** |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../sub_kernel/perf_opt/SubFwdOp.py`（sha256 见 round 5 final 行） |
| 调优记录 | `.../perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 实验清单 | `perf_opt/experiments_sub_r1.json`、`experiments_sub_final.json` |
| 迁移期基线报告 | `reports/tileops/20261013_170400_275674_SubFwdOp/` |
| 端到端 report | 原文未记录（原文未给出 perf_opt 版本的单算子 report 路径/状态） |
| gate 4 | `pass` |
| 算子状态 | `phase=DONE`（原文记；`perf_iteration.count=2`、`last_improvement=3.06`） |
| wrapper 切换 | 未切换（原文记「基准 `sub_fwd_kernel.py` 与 wrapper 未改」，wrapper 仍 import baseline） |
| pattern-library 回写 | PL-1.22 增补「同构算子迁移不复现首轮大收益」的边界（原文 Skill Retrospective） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项：无**（15 个 tune case 原文均未声明豁免；smoke 仅精度回归，不计入性能分母）。
- **贴线风险**：`linear-bias fp16` 73.97% 与 `llama-7b-hidden fp16` 73.29% 距 70% 线约 3–4pp；
  `linear-bias bf16` 官方 66.76% 在线下（距 70% 线约 3pp），但经 5 对 A/B 判为跨轮漂移样本（tuned 快 1.64%，tie），
  不因此改 kernel。
- **ceiling 证据**：原文未记录 roofline advice / pipe ratio（诊断杠杆来自同构迁移与噪声实测，
  非 pipe 拆解）；本算子未声明结构上限。
- **行为回退**：`cnn-feat-broadcast` 三档 +0.49%/+0.68%/+0.98%、`llama-7b-ffn fp16` +0.95%、
  `linear-bias bf16` +5.00% 均在 ±5% 噪声带或漂移样本范围内（A/B 已复核），不计为真实回退。
- **能力缺口**：原文未记录 CG 编号；bf16 原生路径「可编译、max_diff=0 但时延方向不一致」属未证伪的
  性能不确定性，未记入能力缺口表。

## 8. 建议

1. **wrapper 采纳建议：原文未记录**（原文未给出采纳/不采纳建议；实际 wrapper 未切换）。若采纳，
   需先把 `sub.py` 的双 import 切换块翻到 `perf_opt` 并重跑 smoke 与 15 个 tune case。
2. 后续任何 <5% 的结论必须用相邻交错或 A/B 支撑（原文方法论：首轮 baseline 慢态会把机器双态
   误读为优化收益，本案 `llama-hidden` 假 −13.5%）。
3. 贴线项复核顺序：`linear-bias bf16` / `linear-bias fp16` / `llama-7b-hidden fp16` 需多对 A/B 后再改判。
4. 未解决项：bf16 原生 vsub 的收益未证实（原文否决原因：方向不一致），留待工具链/多轮复测再评。
