# 优化报告：IsfiniteFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsfiniteFwdOp`；纯 Vector，`y = isfinite(x)`，bool 经 int8 存储 |
| kernel | `tileops/kernels/elementwise/isfinite/isfinite_kernel/` |
| 产物 | `perf_opt/IsfiniteFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 / 工具链 | msprof op Task Duration + SOL ratio / CANN 9.0.0 / Ascend910B3 / 2026-10-17 |
| benchmark | `benchmarks/ops/bench_isfinite.py`（7 tune + 3 smoke） |

## 1. 结论

- **7 tune 中 1 个 ≥70%**（16m fp32 76.33%）；fp16 57–63%、bf16 42–46%。
- fp16/fp32 改用 **`|x| < +inf` 4-pass 链**（`vabs → vcmp(lt) → vcast → vcast`），相对基线 6-pass：
  fp16 **−12.4~−13.8%**、256m fp16 −5.6%、fp32 −3.1%。
- bf16：`T.vabs` 无 bf16 边，保留基线两条 mask 合并（−2.2~−2.8%，结构收益）。
- 未达 70% 属 predicate 族结构上限（CG-2026-0027）。

## 2. 所做改动
静态尾块 + `T.Pipelined(2)` + grid-stride；fp16/fp32 4-pass vabs 链；bf16 基线链；`TUNED_BLOCK_SIZE = 8192`。

## 3. 优化结果（round 2 相邻交错）

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 58.431 | 56.811 | -2.77% | 40.74 | 42.21 | ❌ |
| 16m-float16 | 38.091 | 33.371 | -12.39% | 48.71 | 57.44 | ❌ |
| 16m-float32 | 52.081 | 50.481 | -3.07% | 73.58 | 76.33 | ✅ |
| 16m-half-bfloat16 | 58.081 | 56.781 | -2.24% | 41.45 | 42.08 | ❌ |
| 16m-half-float16 | 38.631 | 33.321 | -13.75% | 48.59 | 58.89 | ❌ |
| 256m-bfloat16 | 931.809 | 911.728 | -2.16% | 45.31 | 46.47 | ❌ |
| 256m-float16 | 710.714 | 670.963 | -5.59% | 59.05 | 63.10 | ❌ |

### 精度
`python perf_opt/IsfiniteFwdOp.py --level all` → `All check passed!`，max_diff = 0。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束 | 2026-10-17 21:25 → 21:45 (+08:00) |
| 墙钟 | ≈ 20 min |
| tokens (in/out/reasoning) | 10,023 / 9,417 / 3,682 |
| cache-read / cost | 12,182,144 / $0.0459 |

## 5. 产物
`perf_opt/IsfiniteFwdOp.py`、`opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/`；
CG-2026-0027；gate 4 pass；`phase=DONE`。
