# 优化报告：AddFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `AddFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = a + alpha * b` |
| kernel | `tileops/kernels/elementwise/add/add_kernel/` |
| 调优产物 | `.../add_kernel/perf_opt/AddFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op --kernel-name=main` 的 `Task Duration(us)` + `visualize_data.bin` 的 GM Read+Write SOL ratio |
| 工具链 | tilelang `0.1.2` dev root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| benchmark | `benchmarks/ops/bench_add.py::test_add_manifest_bench`（11 tune + 1 smoke） |
| 过程日志 | `.../add_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/` |

## 1. 结论

- **11 个 tune case 全部 ≥70% SOL**；`linear-bias` fp16 **72.30%**、bf16 **70.80%**（此前两轮分别
  为 66.9% / 68.7%，均在线下）。
- 结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核
  + fp16(alpha==1) 原生 vadd**。
- 全部 11 case 时延和 **404.818 → 400.138 µs（−1.16%）**；明确收益
  `llama-7b-ffn fp16 −5.37%`、`llama-7b-ffn bf16 −3.34%`、`hidden-state-prefill bf16 −2.64%`。
- fp16 原生的语义依据：单次加法的精确和可被 fp32 精确表示，"fp32 计算 + 一次舍回" 与 "原生 fp16"
  都是单次正确舍入 → 逐位相等（`--level all` max_diff = 0）。
- bf16 原生：CANN 9.0.0 下可编译且 max_diff = 0，但时延方向不一致（探针 −2%、官方 +3.8%），
  按“不采纳不可靠收益”保留 fp32 中转。

## 2. 所做改动

仅在 `perf_opt/` 内（基准 kernel 与 wrapper 未动）：

1. 满块静态 extent（`num_full = N // block_size`）+ 单尾块 epilogue，去掉热循环 `T.min`/动态 extent；
2. `T.Pipelined(num_local_tasks, num_stages=2)` 重叠下一 tile 的 MTE2 与当前 tile 的 VEC 链；
3. **A6**：fp16 且 `alpha==1` 时直接 `T.vadd(a_ub,b_ub,y_ub)`（4 条向量 pass → 1 条，UB 流量 1/3）；
4. `TUNED_BLOCK_SIZE = 8192` 模块级常量，供 Stage-5 wrapper 成对引用。

## 3. 优化结果（msprof，baseline/final 逐 workload 相邻交错，各 1 次独立进程）

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 38.421 | 39.211 | +2.06% | 77.42 | 73.81 | ✅ |
| cnn-feat-broadcast-float16 | 38.531 | 38.361 | -0.44% | 76.34 | 79.05 | ✅ |
| cnn-feat-broadcast-float32 | 71.661 | 71.201 | -0.64% | 82.26 | 82.56 | ✅ |
| hidden-state-prefill-bfloat16 | 25.761 | 25.081 | -2.64% | 75.19 | 76.45 | ✅ |
| hidden-state-prefill-float32 | 47.321 | 46.751 | -1.20% | 81.97 | 81.87 | ✅ |
| linear-bias-bfloat16 | 13.780 | 13.550 | -1.67% | 69.47 | 70.80 | ✅ |
| linear-bias-float16 | 13.350 | 13.260 | -0.67% | 72.63 | 72.30 | ✅ |
| linear-bias-float32 | 24.570 | 24.430 | -0.57% | 78.13 | 78.45 | ✅ |
| llama-7b-ffn-bfloat16 | 34.181 | 33.041 | -3.34% | 74.85 | 77.93 | ✅ |
| llama-7b-ffn-float16 | 33.871 | 32.051 | -5.37% | 75.51 | 82.00 | ✅ |
| llama-7b-ffn-float32 | 63.371 | 63.201 | -0.27% | 81.51 | 81.30 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

### A/B 与噪声

- `linear-bias` fp16/bf16 的跨进程双态 ±5%；本轮 A6 后 fp16 连续两次 ≥71%，bf16 70.8%（贴线）；
- `cnn bf16 +2.06%` 为噪声（同 case round 6 为 −1.40%），不做 shape 特判。

### 精度

- `python perf_opt/AddFwdOp.py --level all` → **L0/L1/L2/Boundary 全 PASS**（`All check passed!`）；
- `python perf_opt/validate_final.py AddFwdOp.py --block-size 8192` → smoke + linear-bias PASS，max_diff=0。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 16:03 → 20:05 |
| 墙钟时长 | **≈ 242 min**（3 轮结构 + A/B + 4 次官方 final，含 A6 与 bf16 原生证伪） |
| token input/output/reasoning | 254,794 / 117,543 / 128,923 |
| cache-read / cost | 100,834,560 / **$0.4886** |

## 5. 产物

- `.../add_kernel/perf_opt/AddFwdOp.py`（final，sha256 见 `perf_records.jsonl` round 9 final 行）
- `.../add_kernel/perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/`
- 基线报告（迁移期）：`reports/tileops/20261013_162043_457669_AddFwdOp/`
- gate 4：`pass`；state `phase=DONE`
- pattern-library 回写：PL-1.22（小 shape 固定截距/负载不均）、PL-1.17 证伪更正（bf16 原生可用但不可靠）、
  TRAP-tvm-parser-rules 追加
