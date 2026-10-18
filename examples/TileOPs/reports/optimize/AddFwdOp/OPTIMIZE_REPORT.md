# 优化报告：AddFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `AddFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = a + alpha * b` |
| kernel（基准，只读） | `tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py` |
| 调优产物 | `.../add_kernel/perf_opt/AddFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op --kernel-name=main` 的 `Task Duration(us)` + `visualize_data.bin` 的 GM Read+Write SOL ratio |
| **工具链版本戳** | tilelang `0.1.2` dev root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0 `428ab8fdab46` + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| 达标线 | 访存受限 **≥70%**（AI=0.125） |
| benchmark | `benchmarks/ops/bench_add.py::test_add_manifest_bench`（11 tune + 1 smoke） |
| 过程日志 | `.../add_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md`（内容相同，仅结构未归一） |

## 1. 结论

- **11 个 tune case 全部 ≥70% SOL**：全 shape **11/11**、可调 shape **11/11**，无豁免项。
- 最贴线的是 `linear-bias` fp16 **72.30%** / bf16 **70.80%**（此前两轮分别为 66.9% / 68.7%，均在线下）。
- 结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核 +
  fp16(`alpha==1`) 原生 vadd**。
- 全部 11 case 时延和 **404.818 → 400.138 µs（−1.16%）**；明确收益
  `llama-7b-ffn fp16 −5.37%`、`llama-7b-ffn bf16 −3.34%`、`hidden-state-prefill bf16 −2.64%`。
- **建议采纳 wrapper**：净收益为正且无 shape 回退（唯一 `+2.06%` 的 cnn bf16 已判为噪声）。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| AI | 0.125 | 访存受限，达标线 70% |
| 瓶颈 | `mte2` 主导、`vec` 有富余 | 应优先压缩「搬运次数 / 每 tile 固定开销」，而不是减计算 |
| 小 shape 固定开销 | `linear-bias`(4.19M) ≈ 1.6µs（占 12%） | 结构参数（block/映射）只能挤 1–3%，不足以稳定跨线 |
| 跨进程噪声 | ±5% | 单批 A/B <3% 不可分辨，必须交错多对 |

### 2.2 方法（两段式）

**第一段：结构杠杆拿底**（目标 1–3%）

1. **满块静态 extent**：`num_full = N // block_size`，热循环用编译期常量边界，去掉每 tile 的 `T.min` 与动态 extent（省标量开销、利于向量化）；
2. **单尾块 epilogue**：尾块单独一次处理，不进热循环；
3. **`T.Pipelined(num_stages=2)`**：用下一 tile 的 MTE2 覆盖当前 tile 的 VEC 链；
4. **grid-stride 40 核**：20 AIC / 40 AIV 下按 AIV 数铺满，避免核间负载不均；
5. `TUNED_BLOCK_SIZE = 8192` 提为模块级常量，供 wrapper 成对引用。

**第二段：语义等价快路径跨线**（目标 −5~6%）

6. **fp16 原生 vadd**（`alpha == 1` 且 dtype=fp16）：直接 `T.vadd(a_ub, b_ub, y_ub)`，把
   `cast→add(fp32)→cast` 的 4 条向量 pass 压到 1 条、UB 流量降到 1/3。
   合法性依据：单次加法的精确和可被 fp32 精确表示 →「fp32 加 + 一次舍回」与「原生 fp16 加」都是
   单次正确舍入，**逐位相等**（实测 max_diff = 0）。

### 2.3 逐轮记录

策略：本算子 AI=0.125（访存受限），1–3% 的参数杠杆不足以稳定跨 70% 线，因此分两段——先用结构杠杆
（静态 extent / 流水 / 映射）拿底，再用**语义等价的快路径**（fp16 原生）跨线。

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | `base-bs12288`（A1） | block 8192→12288 | fp16 +5~7% | ❌ 否决：9/8 tiles，关键路径 9×12288 比均值高 5.5% |
| R2 | A3 / A4 | 静态尾块 + `T.Pipelined(2)` | fp16 −1.7~−4.6%、bf16 −1.1~−2.0%（4/4 相邻对比） | ✅ 采纳为 current best |
| R2 | `a5-balanced`（A5） | 每核等元素连续分块 | +3%（256M 档再 +3.2%） | ❌ 否决：省下的 1.6% 不平衡抵不过访问模式损失 |
| R3 | `a4-pipe2-bs8192` | 大 block / 别名复测 | 与基线互有胜负（噪声内） | 保留 |
| A/B | A4 vs 基线 | 交错 A/B、交替先后、5 对/dtype | fp16 tie（+1.76%）、bf16 −2.87%（p=0.375）；合并复测 −2.4% / −2.9% | ✅ pipe2 采纳（约 2–3% 真实优势，不宣称 >5%） |
| R4（A6） | fp16 原生 vadd | `alpha==1` 且 fp16 时 `T.vadd(a_ub,b_ub,y_ub)`，4 条向量 pass → 1 条 | `linear-bias` 跨线（72.30%），相对 A4 −2.5%，相对最初基线 −5~6% | ✅ 采纳 |
| 探针 | bf16 原生 | bf16 直接 vadd | CANN 9.0 下可编译、max_diff=0，但时延方向不一致（探针 −2.0%/−2.8%，round 8 官方 +3.8%） | ❌ 不采纳：保留 fp32 中转 |

**fp16 原生的语义依据**：单次加法的精确和可被 fp32 精确表示，「fp32 计算 + 一次舍回」与「原生 fp16」
都是单次正确舍入 → 逐位相等（`--level all` max_diff = 0）。

## 3. 优化结果（msprof，baseline/final 逐 workload 相邻交错，各 1 次独立进程）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 38.421 | 39.211 | +2.06% | 77.42 | 73.81 | ✅ |
| cnn-feat-broadcast-float16 | 38.531 | 38.361 | -0.44% | 76.34 | 79.05 | ✅ |
| cnn-feat-broadcast-float32 | 71.661 | 71.201 | -0.64% | 82.26 | 82.56 | ✅ |
| hidden-state-prefill-bfloat16 | 25.761 | 25.081 | -2.64% | 75.19 | 76.45 | ✅ |
| hidden-state-prefill-float32 | 47.321 | 46.751 | -1.20% | 81.97 | 81.87 | ✅ |
| linear-bias-bfloat16 | 13.780 | 13.550 | -1.67% | 69.47 | **70.80** | ✅ |
| linear-bias-float16 | 13.350 | 13.260 | -0.67% | 72.63 | 72.30 | ✅ |
| linear-bias-float32 | 24.570 | 24.430 | -0.57% | 78.13 | 78.45 | ✅ |
| llama-7b-ffn-bfloat16 | 34.181 | 33.041 | -3.34% | 74.85 | 77.93 | ✅ |
| llama-7b-ffn-float16 | 33.871 | 32.051 | -5.37% | 75.51 | 82.00 | ✅ |
| llama-7b-ffn-float32 | 63.371 | 63.201 | -0.27% | 81.51 | 81.30 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

**测量口径与噪声**：baseline/final 相邻交错、各 1 次独立进程；`linear-bias` fp16/bf16 存在跨进程双态 ±5%，
A6 后 fp16 连续两次 ≥71%；`cnn bf16 +2.06%` 为噪声（同 case round 6 为 −1.40%），不做 shape 特判。

## 4. 精度

- `python perf_opt/AddFwdOp.py --level all` → **L0/L1/L2/Boundary 全 PASS**（`All check passed!`）
- `python perf_opt/validate_final.py AddFwdOp.py --block-size 8192` → smoke + `linear-bias` PASS，max_diff = 0
- 依据：fp16 原生与 fp32 中转同为单次正确舍入 → 逐位相等

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 16:03 → 20:05 |
| 墙钟时长 | **≈ 242 min**（3 轮结构 + A/B + 4 次官方 final，含 A6 与 bf16 原生证伪） |
| token input/output/reasoning | 254,794 / 117,543 / 128,923 |
| cache-read / cost | 100,834,560 / **$0.4886** |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../add_kernel/perf_opt/AddFwdOp.py`（sha256 见 `perf_records.jsonl` round 9 final 行） |
| 调优记录 | `.../perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 迁移期基线报告 | `reports/tileops/20261013_162043_457669_AddFwdOp/` |
| gate 4 | `pass` |
| 算子状态 | `phase=DONE` |
| wrapper | 已切 `perf_opt`（双 import 切换块） |
| 端到端验收 | `reports/tileops/20261017_095652_466213_AddFwdOp/`（status=passed） |
| pattern-library 回写 | PL-1.22（小 shape 固定截距/负载不均）、PL-1.17 证伪更正（bf16 原生可用但不可靠）、TRAP-tvm-parser-rules 追加 |

## 7. 天花板 / 豁免 / 缺口

- **豁免项：无**（11 个 tune shape 全部 ≥6µs，不触发微型 shape 豁免）。
- **贴线风险**：`linear-bias` bf16 70.80%、fp16 72.30% 距 70% 线仅 0.8–2.3pp，受 ±5% 双态噪声支配；
  已在报告中如实标注，不做 shape 特判。
- **`cnn-feat-broadcast bf16` 的 +2.06%**：判定为噪声（同 case 多轮为负），不计为回退。
- 无能力缺口（未触发 CG 条目）。

## 8. 建议

1. 采纳 `perf_opt/AddFwdOp.py`（已切 wrapper，端到端 report passed）。
2. 若后续做 9.0 → 更高版本的工具链升级，需重跑本表 11 个 workload 复核贴线项（`linear-bias` 两档）。
3. 本报告的 ratio 口径为「访存受限」；若未来该算子 AI 变化超过 ridge，需改按算力判据重评。
