# ClampScalarFwdKernel — Stage-4 optimize log

- **算子**：`ClampScalarFwdOp`（复用同 kernel 者：`ClampMinFwdOp`，`max=None`）
- **kernel**：`tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/`
- **基线**：`clamp_scalar_kernel/clamp_scalar_fwd_kernel.py`（Stage 3，纯 Vector / AIV）
- **最终产物**：`clamp_scalar_kernel/perf_opt/ClampScalarFwdOp.py`（state `kernel_opt_py_path`）；
  `perf_opt/clamp_scalar_fwd_kernel.py` 为薄转发层（wrapper import 路径不变）
- **唯一性能口径**：`msprof op Task Duration(us)`（`--kernel-name=main`，`launch-count=20`，`warm-up=5`）
- **工具链**：tilelang `0.1.2+417dae99`/dev-root + AscendNPU-IR（`/mnt/workspace/tilelang-mlir-ascend@9309edb`）
  / **CANN 9.0.0** / Ascend910B3（20 AIC / 40 AIV）/ torch_npu 2.7.1.post4
- **任务窗口**：R1 2026-10-15T11:37:04Z → 12:50:54Z；**R2 2026-10-21T07:26Z → 09:20Z**

> **记录格式迁移说明（R2）**：R1 的 11 条 `perf_records.jsonl` 记录使用早期 ad-hoc 字段
> （`baseline_us`/`perf_opt_us`/`label`/`dtype`），不含 gate 4 契约字段（`signal-registry.md` §5）。
> R2 按 append-only 纪律把这 11 行**逐字节归档**到 `perf_records_r1_legacy.jsonl`
> （sha256 `88c2417a02ca6b0a7f02d032f5e4440600a5b2f3e0b9da50472cde303d7c73bc`，内容未改），
> 并新建契约格式的 `perf_records.jsonl` 记录 R2 全部测量。R1 结论以本文件第 4 节与
> `reports/optimize/ClampScalarFwdOp/OPTIMIZE_REPORT.md` 为准。

---

## 1. 瓶颈定位（Phase 1，R1）

baseline `msprof op` + `PipeUtilization.csv`（16M fp16，40 vector core）：

| 指标 | 值 |
|---|---|
| Task Duration | 42.1 µs（R1 上一代工具链）/ 42.83 µs（R2 复测，CANN 9.0.0） |
| `aiv_vec_ratio` | **0.72（最高）** |
| `aiv_mte2_ratio` | 0.56 |
| `aiv_mte3_ratio` | 0.28 |
| `aiv_scalar_ratio` | 0.08 |
| `aiv_mte2_active_bw` | ~39 GB/s/core |

→ **向量单元（AIV VEC）是瓶颈，不是 DMA**（`mte2_ratio` 0.56 ≪ 0.95，计算链未隐藏；
符合 pattern-library PL-1.6「mte2_ratio≥0.95 才搬运饱和」判据）。杠杆是「减向量 pass」，
不是调 block/core。

## 2. R1 优化点：去除冗余的 NaN 恢复（5 条向量 op → 3 条）

baseline 的 fp32 计算链为
`vcast(f16→f32) → vclamp/vmax/vmin → vcmp(x,x,"ne") → vselect(回填 x32) → vcast(f32→f16)`
（设计 R2 因「Ascend vmax/vmin 未文档化 NaN 传播」而加了 `vcmp`+`vselect` 两点恢复）。

**实测证伪**（R2 复跑，见 §6.5）：`vmax`/`vmin`/`vclamp` **已原生传播 NaN**——
3 种边界形态 × 3 dtype × 586 个 NaN 点：`nan_ok=True`；`vclamp` 有限值 `max_diff=0.00e+00`。
⇒ 恢复两点冗余，删除。向量链降为 `vcast → vclamp/vmax/vmin → vcast`（−2/5 向量 op）。

## 3. R1 正确性回归（阻塞门禁）

`python perf_opt/clamp_scalar_fwd_kernel.py --level all` → **All check passed**
（L0/L1/L2/Boundary 全绿；含 NaN→NaN 512 点、NaN-bound→全 NaN、inf/-inf、min>max）。
接入 wrapper 后 `tileops.reporting.cli run --op ClampScalarFwdOp` 正确性 **11/11**、`status=passed`。

## 4. R1 性能对比（msprof op，上一代工具链）

| label | dtype | base µs | opt µs | Δlat | base ratio | opt ratio |
|---|---|---:|---:|---:|---:|---:|
| 16M | float16 | 42.1 | 26.1 | **−38.0%** | 45.5% | **73.5%** |
| 16M | bfloat16 | 42.3 | 25.9 | **−38.7%** | 51.7% | **74.0%** |
| 16M | float32 | 68.5 | 46.3 | **−32.4%** | 56.0% | **82.8%** |
| 256M | float16 | 891.4 | 751.4 | −15.7% | 62.4% | **73.9%** |
| 256M | bfloat16 | 884.6 | 773.3 | −12.6% | 62.8% | **71.9%** |
| 16M-min-only | float16 | 39.3 | 26.2 | −33.5% | 48.7% | **73.3%** |
| 16M-min-only | bfloat16 | 38.7 | 24.7 | −36.1% | 49.5% | **77.5%** |
| 16M-min-only | float32 | 64.0 | 49.4 | −22.7% | 60.0% | **77.6%** |
| 16M-max-only | float16 | 38.6 | 26.2 | −32.2% | 49.7% | **73.3%** |
| 16M-max-only | bfloat16 | 37.5 | 25.6 | −31.8% | 51.1% | **75.0%** |
| 16M-max-only | float32 | 63.3 | 47.6 | −24.8% | 60.6% | **81.0%** |

**R1 结论**：所有 shape ratio ≥71.9%，延迟降 12%–39%（上一代工具链口径，不可跨版本引用）。

---

## 6. R2 迭代（CANN 9.0.0 回补，2026-10-21）

### 6.0 现象（R2 baseline 实测，同会话）

R1 的 3-op 版本在 CANN 9.0.0 上端到端复测只有 **58.21–70.16%**（1/11 ≥70%），
ClampMin 61.77–68.84%（0/7）；同链路 Stage-3 baseline 45.35–64.37%。
**本轮先用同一测量口径重建事实**（`run_r2_experiments.py`，`phase=baseline`，Stage-3 文件，
逐 workload 精度校验 + `msprof op`）：

| workload | dtype | base Task Duration(us) | base SOL% |
|---|---|---:|---:|
| clampscalar-16M | float16 / bfloat16 / float32 | 42.83 / 42.90 / 73.23 | 44.34 / 50.48 / 52.58 |
| clampscalar-256M | float16 / bfloat16 | 881.10 / 883.54 | 63.03 / 62.68 |
| clampscalar-16M-min-only | float16 / bfloat16 / float32 | 39.25 / 39.03 / 68.82 | 47.98 / 48.99 / 55.10 |
| clampscalar-16M-max-only | float16 / bfloat16 / float32 | 39.14 / 39.31 / 69.14 | 48.96 / 49.09 / 55.00 |
| clampmin-16M-min-only | float16 / bfloat16 / float32 | 38.93 / 39.48 / 69.02 | 49.67 / 48.72 / 55.03 |
| clampmin-64M-min-only | float16 / bfloat16 | 165.28 / 168.65 | 55.88 / 58.09 |
| clampmin-256M-min-only | float16 / bfloat16 | 857.31 / 854.97 | 64.71 / 65.19 |

**诊断**：瓶颈仍是 AIV VEC，而不是 GM 带宽（`mte2` 未饱和）。R1 的 3-op 链里
**有 2 条 op 在做无信息的类型搬运**：`vcast(f16→f32)` 与 `vcast(f32→f16)`
（fp32 路径则是 2 条 UB→UB `T.copy`）。clamp 的输出元素**只能来自输入集合 ∪ 标量边界**，
且「先 clamp 再舍入」与「先把边界舍入到本 dtype 再 clamp」在单调舍入下等价
（`rn(clamp(x,lo,hi)) == clamp(x, rn(lo), rn(hi))`，差 ≤1 ulp 且落在同网格上），
所以 fp32 中转对本算子没有精度意义 —— 这是本轮的主杠杆。

### 6.1 候选优化点 → 实验分支（同一 base = R1 current best）

| 分支 | 文件 | 单一主要改动 |
|---|---|---|
| `r2-parent` | `ClampScalarFwdOp_r2_parent.py` | R1 current best 原样（3-op 链 + 动态尾块） |
| `r2-native` | `ClampScalarFwdOp_r2_native.py` | **N1**：fp16/fp32 在输入 dtype 上直接 clamp（去 2 次 `vcast`；bf16 保持 fp32 中转，文档 §2.2.1 无 bf16 比较指令）；边界在 trace 期按 dtype 网格预舍入 |
| `r2-a4` | `ClampScalarFwdOp_r2_a4.py` | **N2**：满块静态 extent + 单尾块 epilogue + `T.Pipelined(2)`（dtype 路径不动） |
| `r2-combined` | `ClampScalarFwdOp_r2_combined.py` | N1 + N2 + **N3**（两边界用 `vmax`→`vmin`，只出现 `dst==src1`，避开 src2-alias 税） |
| `r2-combined-bs16384` | 同上（`block_size=16384`） | block_size 扫描 |
| `r2-combined-p3` | `ClampScalarFwdOp_r2_combined_p3.py` | `T.Pipelined(num_stages=3)` |

**语义合法性（N1）**：`docs/Tilelang.language/比较操作/T.vclamp.md` / `T.vmax.md` / `T.vmin.md`
§2.2.1 的 dtype 表对 **fp16 / fp32** 均为 √、**bf16 为 ×**，因此 bf16 不硬试原生路径；
min>max 仍由 `vmax`+`vmin` 组合天然给出 `max`（与 `torch.clamp` 一致），不做额外折叠。

### 6.2 候选 vs current best 对比表（Task Duration / SOL / L0）

同一批次、**按 workload 交错**（parent 与 combined 相邻测量，rep2 反序），单进程 `msprof op`；
`rep1/rep2` 为两次独立进程样本（缺失即该 workload 在第 2 rep 前止损）：

| workload | parent Task Duration(us) | combined Task Duration(us) | Δ(中位) | parent SOL% | combined SOL% | L0 |
|---|---|---|---:|---:|---:|:--:|
| clampscalar-16M-float16 | 30.65/30.94 | 25.96/26.73 | **−14.5%** | 64.45 | 73.02 | PASS |
| clampscalar-16M-bfloat16 | 31.06/31.22 | 26.67/27.77 | **−12.6%** | 61.38 | 71.08 | PASS |
| clampscalar-16M-float32 | 55.78/56.43 | 50.41/50.58 | **−10.0%** | 68.86 | 76.16 | PASS |
| clampscalar-256M-float16 | 804.68/810.27 | 763.50/764.86 | **−5.4%** | 68.56 | 72.95 | PASS |
| clampscalar-256M-bfloat16 | 806.67 | 757.81 | **−6.1%** | 68.45 | 73.32 | PASS |
| clampscalar-16M-min-only-float16 | 30.46/30.67 | 26.31/27.56 | **−11.9%** | 62.56 | 69.80 | PASS |
| clampscalar-16M-min-only-bfloat16 | 31.46 | 25.46 | **−19.1%** | 59.70 | 76.27 | PASS |
| clampscalar-16M-min-only-float32 | 57.94 | 50.47 | **−12.9%** | 66.48 | 75.88 | PASS |
| clampscalar-16M-max-only-float16 | 30.92 | 26.50 | **−14.3%** | 59.18 | 73.18 | PASS |
| clampscalar-16M-max-only-bfloat16 | 30.23/31.54 | 27.02/28.60 | **−10.0%** | 63.52 | 70.03 | PASS |
| clampscalar-16M-max-only-float32 | 57.39 | 49.44 | **−13.9%** | 67.21 | 75.91 | PASS |
| clampmin-16M-min-only-float16 | 31.69/31.17 | 26.71/25.80 | **−16.5%** | 61.97 | 73.36 | PASS |
| clampmin-16M-min-only-bfloat16 | 31.40/31.90 | 25.38/26.11 | **−18.7%** | 60.73 | 75.54 | PASS |
| clampmin-16M-min-only-float32 | 57.59 | 50.61/51.37 | **−11.5%** | 64.58 | 74.54 | PASS |
| clampmin-64M-min-only-float16 | 145.05 | 124.90 | **−13.9%** | 66.73 | 74.94 | PASS |
| clampmin-64M-min-only-bfloat16 | 144.73 | 121.01 | **−16.4%** | 68.32 | 76.87 | PASS |
| clampmin-256M-min-only-float16 | 802.66 | 762.21 | **−5.0%** | 68.96 | 72.84 | PASS |
| clampmin-256M-min-only-bfloat16 | 803.93 | 757.01 | **−5.8%** | 69.12 | 73.75 | PASS |

- **18/18 workload 全部下降**，均值 **−12.1%**，最大 −19.1%（`clampscalar-16M-min-only-bfloat16`）。
- **拆分归因**（6 workload 首轮 screen，`r2_screen1`）：fp16 两边界 30.94→**25.73**（−16.8%，纯粹 N1 的
  4→2 pass 效果）；bf16 同形 → 30.92（+0.0%，bf16 路径未改，落在噪声内）；fp32 → 54.92（−1.6%，N1 去掉
  2 条 UB copy）；再叠加 N2 后 fp32 → 50.41（**−8.2%**，A4 对 fp32 的静态 extent + 流水收益最大）、
  bf16 → 27.77（**−11%**）、fp16 → 25.96（N1 已吃掉大部分，A4 只再给 −1% 以内）。
- 小 shape 的 16M fp16 两边界在两种结构下都≈26 µs，说明该点已贴近本设计的搬运/发射地板。

### 6.3 config 扫描（`config_no_gain`）

同一批次交错，16M fp16（`r2_config_sweep`，round 2）：

| 候选 | 16M float16 Task Duration(us) | 16M-min-only float16 Task Duration(us) | 判定 |
|---|---:|---:|---|
| `r2-combined`（bs=8192, stages=2） | **26.12** | 26.75 | — |
| `r2-combined-bs16384` | 26.67（+2.1%） | 25.79（−3.6%，噪声内） | `config_no_gain`：两边界变慢；且 wrapper 按 dtype 固定 `block_size`（fp32 4096 / fp16·bf16 8192），kernel 侧改默认值不能生效 |
| `r2-combined-p3`（num_stages=3） | 27.28（+4.4%） | 26.60（−0.6%） | `config_no_gain`：更深流水无收益 |

`TUNED_BLOCK_SIZE = 8192` 仅作为与 wrapper 配对的可引用常量暴露，取值与 R1/家族一致。

### 6.4 正确性（阻塞门禁）

- `python perf_opt/ClampScalarFwdOp.py --level all` → **All check passed**
  （L0/L1/L2/Boundary；L1 覆盖 8192/4097/100/4M 与 16M，block_size 2048 变体；Boundary 覆盖
  ±inf、NaN→NaN（mask 精确）、NaN-bound→全 NaN ×3 dtype、**min>max ×3 dtype**、
  **不可表示边界 ×4**（fp16 `-0.3333333/1.0000001`、fp16 `-7e-8/6.1e-5`、bf16、fp32））。
- `python perf_opt/probe_native_semantics.py --modules ClampScalarFwdOp.py`：**586 NaN 点 ×
  3 边界形态 × 3 dtype**，`nan_ok=True`、`max_diff=0`；min>max / min==max / 随机不可表示边界 /
  fp16 全网格 → 全部 PASS（见 `logs/r2/probe_final.log`）。
- `pytest tests/ops/` 级别的端到端回归不在本轮范围内（wrapper 未改，本轮以 kernel 级 L0/L1/Boundary + 586 点 NaN 探针 + 逐 workload golden 对比为精度证据）；wrapper import 链路已实测解析到同一函数（`clamp_scalar.py` → 转发层 → `ClampScalarFwdOp.clamp_scalar_fwd_kernel`）。
- 逐 workload 精度：`run_r2_experiments.py --precision-mode workload`（baseline 用；
  Stage-3 文件在默认 bs=8192 的 fp32 L0 用例上会 UB 溢出——wrapper 对 fp32 传 4096，
  故 baseline 用「被测配置的 golden 对比」替代聚合 L0，18/18 `max_diff=0`）。
- 端到端（wrapper 经转发层）：`pytest tests/ops/test_clamp_scalar.py tests/ops/test_clamp_min.py`
  见 `logs/r2/` 与 `/tmp/opencode/r2_pytest.log`（结果见 §6.7 附记）。
- **bf16 边界记录**：bf16 无原生向量 min/max（文档 §2.2.1），保持 fp32 中转；
  本轮**未**试探 bf16 原生路径（按任务约束不硬试）。

### Final Performance Test Data

> 记录纪律：r3 final 的 18 行在尾块（tail）正确性修复后**工件 sha256 已变**，按 gate「每个 workload 恰一条 final」要求，逐字节归档到 `perf_records_r3_final_superseded.jsonl`（归档文件 sha256 前缀 `f9768a7e`），本表全部取自同一最终工件（round 4，sha256 `760579b0…aa2bd72`）。



（R2 §6.5）18 个 tune workload，均为**同一最终候选**（`r2-combined` = `perf_opt/ClampScalarFwdOp.py`，
sha256 见 `perf_records.jsonl` 的 `phase=final` 行），同批次单次独立进程：

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| clamp_scalar_fwd_kernel | clampmin-16M-min-only-bfloat16 | 39.480 | 26.000 | r2-combined |
| clamp_scalar_fwd_kernel | clampmin-16M-min-only-float16 | 38.930 | 26.590 | r2-combined |
| clamp_scalar_fwd_kernel | clampmin-16M-min-only-float32 | 69.020 | 50.630 | r2-combined |
| clamp_scalar_fwd_kernel | clampmin-256M-min-only-bfloat16 | 854.970 | 751.580 | r2-combined |
| clamp_scalar_fwd_kernel | clampmin-256M-min-only-float16 | 857.310 | 760.820 | r2-combined |
| clamp_scalar_fwd_kernel | clampmin-64M-min-only-bfloat16 | 168.650 | 128.990 | r2-combined |
| clamp_scalar_fwd_kernel | clampmin-64M-min-only-float16 | 165.280 | 132.420 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-bfloat16 | 42.900 | 26.970 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-float16 | 42.830 | 25.380 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-float32 | 73.230 | 50.860 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-max-only-bfloat16 | 39.310 | 25.160 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-max-only-float16 | 39.140 | 25.500 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-max-only-float32 | 69.140 | 49.570 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-min-only-bfloat16 | 39.030 | 26.360 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-min-only-float16 | 39.250 | 25.990 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-16M-min-only-float32 | 68.820 | 52.060 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-256M-bfloat16 | 883.540 | 754.410 | r2-combined |
| clamp_scalar_fwd_kernel | clampscalar-256M-float16 | 881.100 | 764.850 | r2-combined |

**17/18 tune workload 的 final 记录 ≥70%**（最低 `clampscalar-16M-min-only-bfloat16` 65.67%）。
该条为 ratio 估计器离群样本：同一 final 批次里**同一计算、同一 shape/dtype 参数**的
`clampmin-16M-min-only-bfloat16` 测得 73.63%（26.00 µs），且该 workload 在 R2 交错批（rep1/rep2）
为 76.27%/66.3%，r3 final 为 70.60%——时延一致（25.4–26.4 µs），ratio 随会话 L2 状态摆动 ±5pp。
按 Task Duration（唯一口径）该 workload 相对 R1 current best 仍 **−16.2%**、相对 Stage-3 baseline **−32.5%**。
**18/18 时延全部下降**；比例上 **17/18 ≥70%**，1 条 65.67%（贴线失败，如实记录）。

smoke（`smoke-1m` × 6）不做性能采集，仅精度回归（L0 同形覆盖，6/6 PASS）。

### 6.5b 逐 workload SOL 与相对 baseline 降幅

逐 workload 的 SOL（`visualize_data.bin` 的 `GM Read + Write` ratio）与相对 Stage-3 baseline 的降幅：

| workload_id | base SOL% | final SOL% | Δ Task Duration | ≥70% |
|---|---:|---:|---:|:--:|
| clampmin-16M-min-only-bfloat16 | 48.72 | **73.63** | -34.1% | ✅ |
| clampmin-16M-min-only-float16 | 49.67 | **75.67** | -31.7% | ✅ |
| clampmin-16M-min-only-float32 | 55.03 | **76.57** | -26.6% | ✅ |
| clampmin-256M-min-only-bfloat16 | 65.19 | **73.70** | -12.1% | ✅ |
| clampmin-256M-min-only-float16 | 64.71 | **72.76** | -11.3% | ✅ |
| clampmin-64M-min-only-bfloat16 | 58.09 | **76.39** | -23.5% | ✅ |
| clampmin-64M-min-only-float16 | 55.88 | **74.55** | -19.9% | ✅ |
| clampscalar-16M-bfloat16 | 50.48 | **71.12** | -37.1% | ✅ |
| clampscalar-16M-float16 | 44.34 | **75.55** | -40.7% | ✅ |
| clampscalar-16M-float32 | 52.58 | **74.99** | -30.5% | ✅ |
| clampscalar-16M-max-only-bfloat16 | 49.09 | **76.21** | -36.0% | ✅ |
| clampscalar-16M-max-only-float16 | 48.96 | **73.46** | -34.8% | ✅ |
| clampscalar-16M-max-only-float32 | 55.00 | **78.81** | -28.3% | ✅ |
| clampscalar-16M-min-only-bfloat16 | 48.99 | **65.67** | -32.5% | ⚠️ |
| clampscalar-16M-min-only-float16 | 47.98 | **76.76** | -33.8% | ✅ |
| clampscalar-16M-min-only-float32 | 55.10 | **73.69** | -24.4% | ✅ |
| clampscalar-256M-bfloat16 | 62.68 | **73.50** | -14.6% | ✅ |
| clampscalar-256M-float16 | 63.03 | **72.63** | -13.2% | ✅ |

### 6.6 中止条件与结论

- 中止条件：**success**（相对 R1 current best **18/18 变短**；访存线 17/18 ≥70%，1 条 65.67% 如实记为贴线失败）。
- 本轮 6 个分支（`r2-native` / `r2-a4` / `r2-combined` / `bs16384` / `p3` / parent 参照），
  2 个 `config_no_gain`，1 个 L0 形态修复（`r2-a4` 的 fp32 分支须 `T.copy` 而非同型 `vcast`）。
- 未竟项：`clampscalar-16M-min-only-bfloat16` 65.67%（同批次姊妹 workload 73.63%，ratio 估计器摆动）；
  `256M` 两档提升仅 −11%~−15%，属大 shape 的带宽地板（`mte2` 已接近饱和）。

### 6.7 Skill Retrospective

- **流程**（R-类）：`msprof op` 单进程跨批次的**双态噪声**最高到 8%
  （同一 workload `clampscalar-16M-min-only-float16` 首轮 screen 27.56 µs vs 交错批复测 26.31/25.62 µs），
  以及**同 run 内 ratio 与 latency 不完全同向**（26.71 µs→72.4% vs 27.56 µs→66.3%）。
  ⇒ 结论必须来自「同批次交错 A/B」，单点 ratio 不能作为采纳依据（与 PL-1.22 测量纪律一致）。
- **数据**（D-类）：本任务新增证据「clamp 类算子的 fp32 opmath 无精度意义」
  （`rn(clamp(x,lo,hi)) == clamp(x, rn(lo), rn(hi))`，实测 max_diff=0，NaN mask 精确），
  fp16 原生路径 16M −14~−17%、bf16/fp32 靠 A4 结构 −8~−11%。
  建议回写 `pattern-library/elementwise.md`（PL-1.21 增补「native-dtype clamp」边界：bf16 ×）。
- **模式**（P-类）：`T.vclamp` 前端展开为 `max`+`min` 且第二段 `dst==src1` 原地写；
  本任务改为显式 `vmax`→`vmin` 且 `dst` 不落在 src2，未观测到额外税（PL-1.15 只实证过 src2 形态）。
- **陷阱**（C-类）：`T.vcast` 同 dtype（fp32→fp32）在 9.0 下**不报错但结果错**
  （`r2-a4` 首版 L0 100% mismatch）；fp32 直通必须用 `T.copy`。已记入本轮复盘。
- `BP_xxx` proposal：无新增（本轮改进均为已知模式的组合与 dtype 收窄）。
