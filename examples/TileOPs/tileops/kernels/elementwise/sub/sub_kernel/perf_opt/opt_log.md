# SubFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> （AscendNPU-IR `31f6903`）/ torch 2.7.1 / torch_npu 2.7.1.post4 / **Ascend910B3**（20 AIC / 40 AIV）/ 2026-10-17。
> 口径：`msprof op --kernel-name=main` 的 **Task Duration(us)** + `visualize_data.bin` 的
> **GM Read + Write SOL ratio**（任务书 §3.1）；直跑 kernel（`perf_opt/bench_kernel.py`），不经 wrapper。
> 达标线：逐元素/访存受限 **SOL ≥ 70%**。

## 0. 对象与入口

- 算子：`SubFwdOp`，语义 `y = a - alpha * b`
- 基线 kernel：`tileops/kernels/elementwise/sub/sub_kernel/sub_fwd_kernel.py`
- 调优产物：`perf_opt/SubFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`）
- benchmark：`benchmarks/ops/bench_sub.py::test_sub_manifest_bench`
- workload：**15 tune + 1 smoke**（`perf_opt/workload_inventory.json`，8 组 shape × dtype）
- 采集：`perf_opt/run_experiments.py`（T-4）；A/B：`.agents/tools/ab_test.py`

## 1. Baseline（同会话，官方 final 轮次重测）

| workload | µs | SOL% |
|---|---:|---:|
| hidden-state-prefill fp32 | 46.561 | 83.40 |
| hidden-state-prefill bf16 | 25.111 | 76.76 |
| cnn-feat-broadcast fp32 | 70.651 | 82.96 |
| cnn-feat-broadcast fp16 | 36.841 | 79.78 |
| cnn-feat-broadcast bf16 | 36.971 | 78.50 |
| llama-7b-ffn fp32 | 62.881 | 82.39 |
| llama-7b-ffn fp16 | 33.621 | 75.42 |
| llama-7b-ffn bf16 | 32.711 | 79.18 |
| llama-7b-ffn-bias fp16 | 33.621 | 77.09 |
| llama-7b-hidden fp16 | 13.980 | **68.09** |
| phi-2-ffn fp16 | 30.481 | 77.46 |
| phi-2-ffn-bias fp16 | 30.511 | 78.58 |
| linear-bias fp32 | 24.900 | 76.69 |
| linear-bias fp16 | 13.900 | 71.12 |
| linear-bias bf16 | 13.410 | 71.44 |

- 14/15 基线 ≥70%；`llama-7b-hidden fp16`（N=4.19M，最小 shape）在线下；
- 噪声底沿用 Add 同设备实测：单进程双态 **±5%**（fp16 13.4~14.9 µs），<5% 差异必须交错 A/B。

## 2. 诊断

- Sub 与 Add 结构同构（同一 `AlphaScaledBinary` fp32 中转链，仅 `vadd→vsub`），DESIGN 的分块/UB 预算一致；
- 因此本轮采用**已证结构的迁移**而非从零枚举：把 Add 的 winner（满块静态 extent + 尾块 epilogue +
  `T.Pipelined(num_stages=2)` + grid-stride 40 核 @ block 8192）移植到 Sub，再用 baseline/候选交错复测验证。
- 该 shortcut 的风险是「Add 的结论未必跨算子成立」，所以下面 R1 的大收益必须由相邻交错复测确认，不能直接采信。

## 3. 迭代日志（候选 vs current best）

### Round 1 — 结构移植 + 首轮 baseline/候选交错

`experiments_sub_r1.json`：baseline 与候选按 workload 相邻交错，各 1 次独立 msprof。首轮出现多处大收益：

| workload | dtype | base µs | 候选 µs | Δ | 候选 SOL% |
|---|---|---:|---:|---:|---:|
| llama-7b-hidden | fp16 | 15.680 | 13.560 | **−13.5%** | 71.54 |
| phi-2-ffn-bias | fp16 | 32.741 | 30.051 | −8.2% | 80.00 |
| phi-2-ffn | fp16 | 32.411 | 29.841 | −7.9% | 81.41 |
| llama-7b-ffn | bf16 | 35.041 | 32.661 | −6.8% | 79.57 |
| llama-7b-ffn-bias | fp16 | 34.341 | 32.681 | −4.8% | 79.82 |
| llama-7b-ffn | fp16 | 34.661 | 33.031 | −4.7% | 78.03 |
| cnn-feat-broadcast | fp16 | 37.711 | 36.081 | −4.3% | 81.37 |
| cnn-feat-broadcast | bf16 | 37.531 | 36.341 | −3.2% | 81.15 |
| linear-bias | bf16 | 13.820 | 13.550 | −2.0% | 70.80 |
| linear-bias | fp16 | 13.950 | 13.840 | −0.8% | 69.67 |
| hidden-state-prefill | bf16 | 25.031 | 26.481 | **+5.8%** | 73.52 |
| cnn-feat-broadcast | fp32 | 70.891 | 72.401 | **+2.1%** | 79.67 |

14/15 候选 ≥70%（仅 linear-bias fp16 69.67%）。**但首轮 baseline 明显处于慢态**（如 llama-hidden 15.68 µs，
最终轮同配置只有 13.98 µs），所以大收益不能直接采信。

### A/B 复核（5 对/例，交替先后，launch-count=20）

| case | base 中位 µs | 候选 中位 µs | 相对 | 判定 |
|---|---:|---:|---:|---|
| hidden-state-prefill bf16 | 25.891 | 25.051 | **−3.2%**（候选快） | inconclusive（与 R1 的 +5.8% 相反） |
| cnn-feat-broadcast fp32 | 71.011 | 72.521 | +2.1%（候选慢） | inconclusive（逐对方向不一致） |
| linear-bias fp16 | 13.880 | 13.880 | 0.0% | tie |

证据：`logs/ab_sub_*.json`。结论：
- R1 的 `hidden bf16 +5.8%` 是噪声假象，A/B 反向显示候选更快 → 不需要条件路径；
- `cnn fp32 +2.1%` 两批同向但 <5% 且 A/B 不显著 → 记为噪声，不做特判；
- `linear-bias fp16` 两版打平，70% 判定落在噪声内。

### Round 2 — 官方 final 全量复测（baseline/final 相邻交错，各 1 次）

见下节 Final Performance Test Data。**R1 的大收益在相邻交错复测中基本回落到噪声内**
（llama-hidden 15.68→13.56 的「−13.5%」在复测中是 13.98→13.90，即 −0.6%），
说明首轮 baseline 的慢态是假收益主因。真实收益集中在：
`llama-7b-ffn fp16 −3.06%`、`llama-7b-ffn-bias fp16 −2.77%`、`linear-bias fp32 −2.33%`。

### 合并与 winner

- winner 结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核**；
- 合并进 `perf_opt/SubFwdOp.py`，`TUNED_BLOCK_SIZE = 8192`；基准 `sub_fwd_kernel.py` 与 wrapper 未改；
- 全部 15 case 时延和 506.151 → 503.591 µs（**−0.51%**），中位 −0.40%，无 >5% 的可信回退。

## Final Performance Test Data

官方最终测量：`perf_opt/SubFwdOp.py`（sha256 见 `perf_records.jsonl` phase=final），
baseline/final 逐 workload 相邻交错、各 1 次独立 msprof。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| sub_fwd_kernel | cnn-feat-broadcast-bfloat16 | 36.971 | 36.411 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | cnn-feat-broadcast-float16 | 36.841 | 36.381 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | cnn-feat-broadcast-float32 | 70.651 | 71.101 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | hidden-state-prefill-bfloat16 | 25.111 | 25.011 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | hidden-state-prefill-float32 | 46.561 | 46.901 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | linear-bias-bfloat16 | 13.410 | 13.510 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | linear-bias-float16 | 13.900 | 13.980 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | linear-bias-float32 | 24.900 | 24.320 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | llama-7b-ffn-bfloat16 | 32.711 | 32.991 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | llama-7b-ffn-bias-float16 | 33.621 | 32.691 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | llama-7b-ffn-float16 | 33.621 | 32.591 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | llama-7b-ffn-float32 | 62.881 | 62.401 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | llama-7b-hidden-float16 | 13.980 | 13.900 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | phi-2-ffn-bias-float16 | 30.511 | 30.881 | tuned-pipe2-bs8192 |
| sub_fwd_kernel | phi-2-ffn-float16 | 30.481 | 30.521 | tuned-pipe2-bs8192 |

### SOL 对账

| workload | baseline SOL% | final SOL% | 判定（≥70%） |
|---|---:|---:|:--:|
| cnn-feat-broadcast bf16 | 78.50 | 80.26 | ✅ |
| cnn-feat-broadcast fp16 | 79.78 | 79.69 | ✅ |
| cnn-feat-broadcast fp32 | 82.96 | 83.53 | ✅ |
| hidden-state-prefill bf16 | 76.76 | 78.26 | ✅ |
| hidden-state-prefill fp32 | 83.40 | 81.62 | ✅ |
| linear-bias bf16 | 71.44 | 70.70 | ✅ |
| **linear-bias fp16** | 71.12 | **69.27** | ❌（噪声内，Δ +0.58%） |
| linear-bias fp32 | 76.69 | 80.36 | ✅ |
| llama-7b-ffn bf16 | 79.18 | 77.14 | ✅ |
| llama-7b-ffn-bias fp16 | 77.09 | 79.92 | ✅ |
| llama-7b-ffn fp16 | 75.42 | 79.67 | ✅ |
| llama-7b-ffn fp32 | 82.39 | 83.60 | ✅ |
| **llama-7b-hidden fp16** | 68.09 | **68.38** | ❌（噪声内，Δ −0.57%） |
| phi-2-ffn-bias fp16 | 78.58 | 76.13 | ✅ |
| phi-2-ffn fp16 | 77.46 | 78.53 | ✅ |

**结论（如实）**：**15 个 tune case 中 13 个 ≥70%**；`linear-bias fp16`（69.27%）与
`llama-7b-hidden fp16`（68.38%）差 0.7~1.6 点，均在 ±5% 噪声内，**未稳健达标**。
整体时延 −0.51%，无 >5% 的可信回退；收益集中的三项（llama fp16 −3.06%、llama-bias fp16 −2.77%、
linear fp32 −2.33%）与结构预期一致。

## 精度回归

- `python perf_opt/SubFwdOp.py --level all` → **L0 / L1 / L2 / Boundary 全 PASS**（`All check passed!`，
  max_diff 最大 9.537e-07）；
- `python perf_opt/validate_final.py SubFwdOp.py --block-size 8192` →
  smoke `hidden-state-prefill fp16`（N=8,388,608）PASS；linear-bias fp16/bf16 PASS，max_diff=0。

## 噪声与稳定性

- 同配置跨进程双态 ±5%（本任务 R1 baseline 与 final baseline 对同一 case 相差最大 ~12%：
  llama-hidden 15.68 vs 13.98 µs），**任何 <5% 结论必须用相邻交错或 A/B**；
- R1 的「大收益」是慢态 baseline 造成的假象，最终轮已修正——这是本任务最重要的方法论教训。

## Skill Retrospective

- **D（实测）**：Sub 与 Add 结构同构，移植 winner 后整体 −0.51%（中位 −0.40%），
  三项稳定收益 −2.3~−3.1%；`linear-bias fp16`/`llama-hidden fp16` 在噪声内。
  回写：`pattern-library/elementwise.md` PL-1.22 增补「同构算子迁移不复现首轮大收益」的边界。
- **P（流程）**：**首轮 baseline 必须与候选同批相邻交错**；跨轮比较（R1 baseline vs final baseline）
  会把机器双态误读为优化收益（本案 llama-hidden 假 −13.5%）。建议在 T-4 runner 默认 `--order workload`。
- **R（工具）**：`validate_final.py` 等 helper 应随 perf_opt 骨架一起复制，避免链式命令因缺文件中断
  （本案曾导致一次 rc=2、20 分钟测量未启动）。
- 复用结论：Add 的 winner 结构对 Sub 同样有效（无 >5% 回退），可作为同族 elementwise 的默认起点，
  但**收益必须逐算子复测**。

## 证据索引

- `perf_opt/perf_records.jsonl`（round 1：15 baseline + 15 candidate；round 2：15 baseline + 15 final）
- `perf_opt/logs/`（`sub_level_all.log`、`sub_validate.log`、`sub_r1_runner.log`、`sub_final_runner2.log`、`ab_sub_*.json`）
- `perf_opt/experiments_sub_r1.json` / `experiments_sub_final.json`
- 候选/产物：`perf_opt/SubFwdOp.py`
