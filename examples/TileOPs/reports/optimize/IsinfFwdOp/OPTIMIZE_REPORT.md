# 优化报告：IsinfFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsinfFwdOp`；纯 Vector，`y = isinf(x)`，bool 经 int8 字节存储 |
| kernel | `tileops/kernels/elementwise/isinf/isinf_kernel/` |
| 调优产物 | `.../isinf_kernel/perf_opt/IsinfFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | msprof op Task Duration(us) + GM Read+Write SOL ratio |
| 工具链 | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 / 2026-10-17 |
| benchmark | `benchmarks/ops/bench_isinf.py::test_isinf_bench`（7 tune + 3 smoke） |

## 1. 结论

- **7 个 tune case 中 1 个 ≥70%**（16m fp32 76.03%）；fp16 57–63%、bf16 42–46%。
- fp16/fp32 改用 **4-pass vabs 链**（`|x| == +inf`），相对基线 6-pass：fp16 **−11.9~−12.2%**、256m fp16 −5.6%、fp32 −4.2%；
- bf16：`T.vabs` 无 bf16 dtype 边；`vselect` 合并实测 +6.6~8.6%（更慢），回退基线链（−0.2~−0.9%，中性）；
- 未达 70% 属 predicate 族结构上限（**CG-2026-0027**）：无 `i1→i8`、无 bool `vnot`、`vselect` 无 int8 输出。

## 2. 所做改动

仅在 `perf_opt/` 内：静态尾块 + `T.Pipelined(2)` + grid-stride；fp16/fp32 的 vabs 4-pass 链；
bf16 保留基线 6-pass 合并；`TUNED_BLOCK_SIZE = 8192`。

## 3. 优化结果（round 4 相邻交错）

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 57.071 | 56.671 | -0.70% | 41.80 | 42.37 | ❌ |
| 16m-float16 | 38.561 | 33.841 | -12.24% | 50.22 | 58.07 | ❌ |
| 16m-float32 | 52.761 | 50.561 | -4.17% | 71.92 | 76.03 | ✅ |
| 16m-half-bfloat16 | 57.251 | 56.761 | -0.86% | 41.83 | 42.65 | ❌ |
| 16m-half-float16 | 37.951 | 33.421 | -11.94% | 50.11 | 57.37 | ❌ |
| 256m-bfloat16 | 917.428 | 915.318 | -0.23% | 46.05 | 46.29 | ❌ |
| 256m-float16 | 709.204 | 669.443 | -5.61% | 59.76 | 62.94 | ❌ |

### 精度

`python perf_opt/IsinfFwdOp.py --level all` → `All check passed!`，max_diff = 0。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 21:05 → 21:35 |
| 墙钟时长 | ≈ 30 min |
| token input/output/reasoning | 4,952 / 5,691 / 2,002 |
| cache-read / cost | 7,877,632 / $0.0290 |

## 5. 产物

- `.../isinf_kernel/perf_opt/IsinfFwdOp.py`、`opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/`
- 能力缺口：CG-2026-0027
- gate 4 pass；state `phase=DONE`
