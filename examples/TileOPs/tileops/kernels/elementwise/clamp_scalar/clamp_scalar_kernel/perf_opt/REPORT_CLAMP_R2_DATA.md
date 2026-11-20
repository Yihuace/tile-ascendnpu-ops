# ClampScalarFwdOp / ClampMinFwdOp — Stage-4 R2 数据报告（CANN 9.0.0）

> 本文件是 **R2 单轮实验数据报告**，与 `reports/optimize/ClampScalarFwdOp/OPTIMIZE_REPORT.md`（R1 汇总）
> 和 `perf_opt/opt_log.md`（全过程日志）互补。口径唯一：`msprof op Task Duration(us)`。

## 0. 身份与工具链

| 项 | 值 |
|---|---|
| 算子 | `ClampScalarFwdOp`（11 tune workload）+ `ClampMinFwdOp`（7 tune workload，复用同一 kernel） |
| family | elementwise / 单 kernel / 纯 Vector（AIV，20 AIC × 2 = 40 AIV） |
| 基准 kernel（只读） | `clamp_scalar_kernel/clamp_scalar_fwd_kernel.py`（Stage-3，5 条向量 op） |
| R1 current best（本轮 parent） | `perf_opt/ClampScalarFwdOp_r2_parent.py`（3 条向量 op） |
| **R2 final** | `perf_opt/ClampScalarFwdOp.py`（state `kernel_opt_py_path`） |
| wrapper 入口（未改） | `perf_opt/clamp_scalar_fwd_kernel.py` → 薄转发 `from .ClampScalarFwdOp import clamp_scalar_fwd_kernel` |
| 工具链 | tilelang `0.1.2+417dae99` dev-root（`/mnt/workspace/tilelang-mlir-ascend@9309edb`）/ **CANN 9.0.0** / Ascend910B3 / torch_npu 2.7.1.post4 / 2026-10-21 |
| benchmark | `benchmarks/ops/bench_clamp_scalar.py::test_clamp_scalar_bench`（11 tune + 3 smoke）、`benchmarks/ops/bench_clamp_min.py::test_clamp_min_bench`（7 tune + 3 smoke） |
| 达标线 | 访存 SOL **≥70%**（`visualize_data.bin` 的 `GM Read + Write` ratio） |
| gate 4 | `pass: true`（`statectl gate 4 --dir .../clamp_scalar_kernel`，2026-10-21T09:20Z（本报告生成时）） |

## 1. 诊断（为什么 9.0 上掉到 58–70%）

R1 的 3-op 链在 9.0 上的构成（fp16/fp32）：

```
T.copy(GM→UB) → vcast(f16→f32) → vclamp/vmax/vmin(fp32) → vcast(f32→f16) → T.copy(UB→GM)
                 \__ 2 条无信息类型搬运 __/
```

- profile 事实：AIV `vec_ratio` 0.72 ≫ `mte2_ratio` 0.56 ⇒ **不是 GM 带宽瓶颈，是向量 pass 数**；
  `aiv_mte2_active_bw ≈ 39 GB/s/core`，远未饱和。
- 语义事实：clamp 的输出只可能是**输入值或标量边界**，fp32 opmath 不产生新的中间值；
  且舍入单调 ⇒ `rn(clamp(x, lo, hi)) == clamp(x, rn(lo), rn(hi))` 在 dtype 网格上成立。
  ⇒ 两条 `vcast` 是**纯开销**（R1 只删掉了 NaN 恢复的 2 条，没有删 cast）。
- 结构事实：R1 每个 tile 都重算 `T.min(block_size, shape-offset)` 与动态 copy extent，
  热循环拿不到编译期常量长度，`T.Pipelined` 无法形成干净的 2 段流水。

## 2. 方法（3 个单点改动 + 2 个 config 扫描）

| 编号 | 改动 | 文件 | 依据 |
|---|---|---|---|
| **N1** | fp16/fp32 在**输入 dtype 上直接 clamp**（去 2 条 `vcast`）；边界在 trace 期按 dtype 网格预舍入（`struct` round-half-even） | `ClampScalarFwdOp_r2_native.py` | `T.vclamp.md`/`T.vmax.md`/`T.vmin.md` §2.2.1：fp16/fp32 √、**bf16 ×** ⇒ bf16 保持 fp32 中转 |
| **N2** | 满块静态 extent + 单尾块 epilogue（core 0）+ `T.Pipelined(num_stages=2)` | `ClampScalarFwdOp_r2_a4.py` | pattern-library PL-1.22 |
| **N3** | 两边界用 `vmax`→`vmin`（只出现 `dst==src1`，无 `dst==src2`） | 合入 combined | PL-1.15（src2-alias 税 +17~20%，src1 形态未观测到税） |
| config | `block_size` 16384、`num_stages=3` | `r2_config_sweep.json` | 均为 `config_no_gain` |

**为什么 N1 合法（精度）**：clamp 是选择器，不是算术；两次舍入与一次舍入在单调映射下同结果。
实测 `--level all` 与 586-NaN 探针 `max_diff = 0.00e+00`（逐位相等），NaN mask 完全一致。
**为什么 bf16 不硬试**：文档 §2.2.1 对 bf16 明示不支持；本轮按要求保持 fp32 中转。

## 3. 逐 workload 数据（两套 bench，同一最终候选）

`msprof op --kernel-name=main --launch-count=20 --warm-up=5`，单进程、同批次；
baseline = Stage-3 文件，final = `perf_opt/ClampScalarFwdOp.py`（sha256 `760579b0…aa2bd72`）。

### 3.1 ClampScalar bench（11 tune）

| workload | dtype | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---|---:|---:|---:|---:|---:|:--:|
| 16M | float16 | 42.830 | 25.380 | **−40.7%** | 44.34 | **75.55** | ✅ |
| 16M | bfloat16 | 42.900 | 26.970 | **−37.1%** | 50.48 | **71.12** | ✅ |
| 16M | float32 | 73.230 | 50.860 | **−30.5%** | 52.58 | **74.99** | ✅ |
| 256M | float16 | 881.100 | 764.850 | −13.2% | 63.03 | **72.63** | ✅ |
| 256M | bfloat16 | 883.540 | 754.410 | −14.6% | 62.68 | **73.50** | ✅ |
| 16M-min-only | float16 | 39.250 | 25.990 | **−33.8%** | 47.98 | **76.76** | ✅ |
| 16M-min-only | bfloat16 | 39.030 | 26.360 | **−32.5%** | 48.99 | **65.67** | ⚠️ |
| 16M-min-only | float32 | 68.820 | 52.060 | −24.4% | 55.10 | **73.69** | ✅ |
| 16M-max-only | float16 | 39.140 | 25.500 | **−34.8%** | 48.96 | **73.46** | ✅ |
| 16M-max-only | bfloat16 | 39.310 | 25.160 | **−36.0%** | 49.09 | **76.21** | ✅ |
| 16M-max-only | float32 | 69.140 | 49.570 | −28.3% | 55.00 | **78.81** | ✅ |

### 3.2 ClampMin bench（7 tune，同 kernel，`max=None`）

| workload | dtype | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---|---:|---:|---:|---:|---:|:--:|
| 16M-min-only | float16 | 38.930 | 26.590 | **−31.7%** | 49.67 | **75.67** | ✅ |
| 16M-min-only | bfloat16 | 39.480 | 26.000 | **−34.1%** | 48.72 | **73.63** | ✅ |
| 16M-min-only | float32 | 69.020 | 50.630 | −26.6% | 55.03 | **76.57** | ✅ |
| 64M-min-only | float16 | 165.280 | 132.420 | −19.9% | 55.88 | **74.55** | ✅ |
| 64M-min-only | bfloat16 | 168.650 | 128.990 | −23.5% | 58.09 | **76.39** | ✅ |
| 256M-min-only | float16 | 857.310 | 760.820 | −11.3% | 64.71 | **72.76** | ✅ |
| 256M-min-only | bfloat16 | 854.970 | 751.580 | −12.1% | 65.19 | **73.70** | ✅ |

**合计**：18/18 时延下降（−11.3% ~ −40.7%，均值 **−27.0%**）；SOL **17/18 ≥70%**（1 条 65.67%，见 §6）。

### 3.3 相对 R1 current best 的交错 A/B（同一 base 派生，同批次相邻测量）

| workload | parent µs | final µs | Δ |
|---|---:|---:|---:|
| clampscalar-16M-float16 | 30.65/30.94 | 25.96/26.73 | −14.5% |
| clampscalar-16M-bfloat16 | 31.06/31.22 | 26.67/27.77 | −12.6% |
| clampscalar-16M-float32 | 55.78/56.43 | 50.41/50.58 | −10.0% |
| clampscalar-256M-float16 | 804.68/810.27 | 763.50/764.86 | −5.4% |
| clampscalar-256M-bfloat16 | 806.67 | 757.81 | −6.1% |
| clampscalar-16M-min-only-float16 | 30.46/30.67 | 26.31/27.56 | −11.9% |
| clampscalar-16M-min-only-bfloat16 | 31.46 | 25.46 | −19.1% |
| clampscalar-16M-min-only-float32 | 57.94 | 50.47 | −12.9% |
| clampscalar-16M-max-only-float16 | 30.92 | 26.50 | −14.3% |
| clampscalar-16M-max-only-bfloat16 | 30.23/31.54 | 27.02/28.60 | −10.0% |
| clampscalar-16M-max-only-float32 | 57.39 | 49.44 | −13.9% |
| clampmin-16M-min-only-float16 | 31.69/31.17 | 26.71/25.80 | −16.5% |
| clampmin-16M-min-only-bfloat16 | 31.40/31.90 | 25.38/26.11 | −18.7% |
| clampmin-16M-min-only-float32 | 57.59 | 50.61/51.37 | −11.5% |
| clampmin-64M-min-only-float16 | 145.05 | 124.90 | −13.9% |
| clampmin-64M-min-only-bfloat16 | 144.73 | 121.01 | −16.4% |
| clampmin-256M-min-only-float16 | 802.66 | 762.21 | −5.0% |
| clampmin-256M-min-only-bfloat16 | 803.93 | 757.01 | −5.8% |

**归因**（首轮 screen，`r2_screen1`，6 workload）：fp16 两边界 30.94→25.73（−16.8%，纯 N1）；
bf16 同形 30.92（N1 不适用，噪声内）；fp32 54.92（−1.6%，N1 只去掉 2 条 UB copy）；
叠加 N2 后 fp32→50.41（再 −8.2%）、bf16→27.77（−11%）、fp16→25.96（N1 已吃掉大部分）。

## 4. 精度

| 项 | 结果 | 证据 |
|---|---|---|
| `python perf_opt/ClampScalarFwdOp.py --level all` | **All check passed** | `logs/r2/l0_all_final.log`（L0/L1/L2/Boundary 全绿；L1 含 8192/4097/100/4M/16M 与 bs=2048 变体） |
| NaN 传播（586 点 × 3 形态 × 3 dtype，final + parent 各一轮） | `nan_ok=True`、`max_diff=0.00e+00` | `logs/r2/probe_final.log` |
| ±inf / NaN-bound（全 NaN）/ min>max / min==max / 不可表示边界 / fp16 全网格扫描 | 全部 PASS | 同上（`probe failures: 0`） |
| 逐 workload golden 对比（18 个 tune 配置，含 baseline） | `max_diff = 0`，18/18 PASS | `check_workload.py`，见 `logs/r2/` |
| smoke（`smoke-1m` × 6） | PASS（L0 同形 1M × 3 dtype × 两套 bound 形式） | `inventory.workloads[*].precision_pass=true` |

关键语义说明：fp16 原生路径与 fp32 opmath golden **逐位相等**（`max_diff = 0`），
不是"容差内相等"；bf16 仍走 fp32 中转，因此保持 R1 的舍入路径不变。

## 5. 产物与 sha256

| 类别 | 路径 | sha256 |
|---|---|---|
| **final kernel** | `perf_opt/ClampScalarFwdOp.py` | `760579b07434ea91f4659e6a693fe7139df09e0fac2fbf8e056fd1b31aa2bd72` |
| wrapper 转发层 | `perf_opt/clamp_scalar_fwd_kernel.py` | `dae319bc9c92b7dd994cc3c07af07c3ba2c75512abdfdbcbd508cf83bd779f4b` |
| R2 parent（R1 best） | `perf_opt/ClampScalarFwdOp_r2_parent.py` | 见 `perf_records.jsonl` 的 `r2-parent` candidate 行 |
| 分支 | `perf_opt/ClampScalarFwdOp_r2_{native,a4,combined,combined_p3}.py` | 同上 |
| 结构化记录 | `perf_opt/perf_records.jsonl`（101 行：18 baseline + 65 candidate + 18 final） | — |
| R1 记录归档 | `perf_opt/perf_records_r1_legacy.jsonl`（11 行，逐字节） | `88c2417a…7c73bc` |
| r3 final 归档 | `perf_opt/perf_records_r3_final_superseded.jsonl`（18 行，被尾块修复作废） | `f9768a7e…` |
| workload 清单 | `perf_opt/workload_inventory.json`（18 tune + 6 smoke） | — |
| raw profile | `perf_opt/profiles/r2/**`（msprof 原始输出） | — |
| 过程日志 | `perf_opt/logs/r2/**` | — |
| 调优日志 | `perf_opt/opt_log.md`（R2 章节 + Final 表） | — |

## 6. gate / 未竟项 / 天花板

- **gate 4**：`pass: true`（S4-EXISTS / S4-JIT / S4-GOLDEN / S4-TEST-LEVELS / S4-PLACEHOLDER /
  S4-OPTLOG-COMPTABLE / S4-WORKLOAD-INVENTORY / S4-PERF-RECORDS-SCHEMA / -COVERAGE / -RECON 全过）。
- **未竟项 1**：`clampscalar-16M-min-only-bfloat16` final 记录 65.67%（26.36 µs）。
  同一 final 批次中**同计算同参数**的 `clampmin-16M-min-only-bfloat16` 为 73.63%（26.00 µs），
  该 workload 在交错批为 76.27% / 66.30%，r3 final 为 70.60%：**时延稳定在 25.4–26.4 µs，
  ratio 估计器随会话 L2 状态摆动 ±5pp**。按唯一口径（Task Duration）它相对 R1 best 仍 −16.2%。
  结论：该点属"贴线未过"，不掩盖、不特判。
- **未竟项 2**：bf16 两档 16M 只到 71.1%/76.2%（含 bf16 必须的 fp32 中转 2 条 pass）。
  原生 bf16 比较无文档支持（§2.2.1 ×），本轮按要求不硬试。
- **未竟项 3**：`block_size` 与 `num_stages` 已扫（16384 / stages=3），均 `config_no_gain`；
  且 wrapper 按 dtype 固定 block_size（fp32 4096、fp16/bf16 8192），kernel 侧改默认值不生效。
- **天花板判断**：不触发 `[DESIGN_LIMIT]`。理由：本轮已用同 API 把 16M fp16 从 58.8%（R1 9.0 口径）
  抬到 75.6%，**结构性加速 >2x 或"实测与设计矛盾"两项均不成立**；剩余差距（256M −11~−15%、
  bf16 贴线）属带宽/指令可得性边界，非设计层假设错误，故不产出 `perf_feedback.md`。
- **R1 记录迁移说明**：R1 的 `perf_records.jsonl` 使用早期 ad-hoc 字段，与 gate 4 契约
  （`signal-registry.md` §5）不兼容；本轮按 append-only 纪律逐字节归档为
  `perf_records_r1_legacy.jsonl`，并新建契约格式记录。R1 的 11 条结论仍在
  `reports/optimize/ClampScalarFwdOp/OPTIMIZE_REPORT.md` 与 `opt_log.md` §1–§4。
