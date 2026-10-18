# BatchNormBwdOp Stage 4 优化报告（OPTIMIZE_REPORT）

- 算子：`BatchNormBwdOp`（batch norm 反向）
- 目录：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/`
- benchmark：`bench_batch_norm_bwd.py::test_batch_norm_bwd_bench`
- 环境：Ascend910B3（20 AIC / 40 AIV）、CANN 9.0.0、tilelang 0.1.2 dev-root、`msprof op Task Duration` 唯一时延口径

## 结论

- **全 shape 达标数：0 / 8**（8 = 6 tune + 3 smoke 共 8 案例；smoke 不采性能，按精度回归计 3/3 通过）
- **可调 shape 达标数：0 / 6**（6 个 tune workload 的算力受限 60% SOL 均未达标）
- **硬底线（Task Duration 变短）：6 / 6 达标**（−33.1% ~ −45.9%）

一句话：SOL 未达 60% 算力受限线（结构性天花板，见 §ceiling/缺口），但 Task Duration 硬底线全 workload 显著下降，**建议采纳 wrapper 切换**（有净收益）。

## 判据选择依据

本算子三 kernel 均为 **Vector-only**：单 `T.Scope("Vector")`、无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/L1/L0C；msprof `ArithmeticUtilization` 实测 `aic_cube_ratio=0`。roofline computility 归属 `Vec_FP32(65%) + Vec_MISC(35%)`。**不满足 Cube 主导混合类（Mix ≥40%）**，故按**算力受限 60%** 判（非 40%）。

## 改动

三个 kernel 统一一个优化点：**subid 分片（PL-1.13-aiv-dup-subid-split）**。

- 现象：`T.Kernel(n, is_npu=True) + T.Scope("Vector")` 的 Vector 程序默认在每 block 双 AIV 上**重复执行**（Mix Block Dim = 2×Block Dim；判据：PipeUtilization 中 `vector0`/`vector1` 子块 vec/mte ratio 完全相同而非各半）。Stage-3 的 `num_kernels = min(C, 40)`（设计 §5.5 "AI cores ×2"）实际导致 40 block × 双 AIV 冗余 = 2x 产能浪费。
- 改法（每个 kernel）：
  - `num_kernels = min(C, aicore_num)`（aicore_num = `get_aicore_num()` = 20）
  - `total_slots = num_kernels * 2`
  - 通道/任务循环 `c = i * total_slots + cid * 2 + subid`（`subid` 取 `T.Kernel(...) as (cid, subid)` 第二元）
  - split kernel 三段（stats/finalize/apply）同步改（stats 任务索引 `t = i*(KS*2)+cid*2+subid`，finalize/apply 通道索引同 wide）
- 签名不变（drop-in）：三 kernel 工厂签名、`out_idx`、wrapper 调用约定均未改。

## msprof 对比表（Task Duration，唯一口径）

| workload_id | shape | dtype | baseline_us | final_us | 提升 |
|---|---|---:|---:|---:|
| resnet-stage1-float16 | 8,64,1024 | fp16 | 21.080 | 13.460 | −36.2% |
| resnet-stage1-bfloat16 | 8,64,1024 | bf16 | 20.400 | 13.640 | −33.1% |
| resnet-stage2-float16 | 4,128,1024 | fp16 | 25.351 | 15.810 | −37.6% |
| resnet-stage3-float16 | 4,256,784 | fp16 | 42.241 | 23.530 | −44.3% |
| resnet-stage3-float32 | 4,256,784 | fp32 | 43.181 | 24.340 | −43.6% |
| large-spatial-float16 | 4,128,262144 | fp16 | 5365.027 | 2903.328 | −45.9% |

SOL（msprof roofline `GM Read+Write` ratio）：wide 14.0~15.5%、split 25.5%（60% 未达标）。注：subid 分片后 ratio 下降是**正确现象**（roofline `performance=counted_flops/duration` 去重复执行的双计），不用于跨 subid 前后对比。

## 精度

- L0：三 kernel 全 PASS（wide 14 items、split 13 items、general 15 items）。
- smoke（smoke-fc [32,64,1] 三 dtype）：float32/float16/bfloat16 全 PASS（max_abs_err ≤ 1.9e-6）。
- 无精度回归。

## 耗时

- 总 wall-clock ≈ 55 min（含 baseline 采数、1 轮候选、final 复测、L0/smoke 精度回归），在 180min 止损内。
- 实验分支：1 轮 2 分支（wide subid-split + split subid-split），每分支单优化点；`msprof op` 全局串行，每 workload launch-count=20 / warm-up=5。

## 产物

- `perf_opt/_batch_norm_bwd_wide_kernel.py`（subid 分片，sha256 3a6e5fb5…）
- `perf_opt/_batch_norm_bwd_split_kernel.py`（subid 分片，sha256 cffb859b…）
- `perf_opt/_batch_norm_bwd_kernel.py`（subid 分片，no_tunable_workload，sha256 ffb95322…）
- `perf_opt/opt_log.md`（含 `Final Performance Test Data` 表 + 逐轮对比表 + Skill Retrospective）
- `perf_opt/perf_records.jsonl`（18 行，append-only，每 (kernel, workload) 恰一条 baseline/candidate/final）
- `perf_opt/workload_inventory.json`（分类完整，tuning_status=winner_merged / precision_pass=true）
- `perf_opt/profiles/`（raw msprof）、`perf_opt/logs/`（过程日志）

## ceiling / 缺口

- **SOL 60% 未达标的缺口归因**（roofline + pipe 证据）：
  1. fp32 中转（正确性硬约束）：fp16/bf16 → fp32 计算（bf16 无 Vector 算术、fp16 对齐 fp32 golden），ALU 吞吐减半。
  2. Vec_MISC 35%：vbrc 广播（逐通道 5 次）+ vcast，占向量工作 35%，封顶 ~65% SOL。
  3. 标量 + 内存：wide `aiv_scalar 23~33%` + `aiv_mte2 12~18%`；split `aiv_mte2/mte3 31%/10%`（5-pass 流量）→ 向量引擎占空比 37~57%。
- **不触发 [DESIGN_LIMIT]**：无被设计冻结而可 >2x 加速的备选结构（fp32 中转/vbrc/内存为正确性或算法固有；"归约走 Cube"为 M=1/K=L/N=1 极瘦 GEMM，无实测先例，标未实证假设）；subid 分片这一结构性修复已在 Stage 4 内完成。
- **建议采纳 wrapper**：有净收益（−33~−46%），精度无回归，签名 drop-in，建议 conductor 在回归通过后翻转 wrapper 的 baseline/perf_opt 双 import 切换块。
