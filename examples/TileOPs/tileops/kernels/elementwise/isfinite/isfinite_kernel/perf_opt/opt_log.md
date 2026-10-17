# IsfiniteFwdOp — Stage 4 调优日志

> CANN 9.0.0 / tilelang 0.1.2 dev root（AscendNPU-IR 31f6903）/ Ascend910B3 / 2026-10-17。
> 口径：msprof op Task Duration(us) + GM Read+Write SOL ratio；AI ≈ 2.0 → 访存受限，线 70%。

## 0. 对象与入口
- `IsfiniteFwdOp`（`y = isfinite(x)`，bool 经 int8）；基线 `isfinite_kernel/isfinite_fwd_kernel.py`；
- 产物 `perf_opt/IsfiniteFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`）；benchmark `bench_isfinite.py`（7 tune + 3 smoke）。

## 1. Baseline 与诊断
基线链 6 pass：`vcmp(gt -inf) + vcmp(lt +inf) + vcast + vcast + vmul + vcast`；
16m fp32 73.58%、fp16 48.7%、bf16 40.7%；256m fp16 59.1%、bf16 45.3%。

## 2. 迭代日志（候选 vs current best，Task Duration）

### Round 1 — `isfinite(x) = |x| < +inf`（fp16/fp32，4 pass）
`vabs → vcmp(lt) → vcast(bool→f16) → vcast(f16→i8)`；NaN 的有序 `lt` 为 false，天然排除。

| workload | baseline µs | abs µs | Δ |
|---|---:|---:|---:|
| 16m fp16 | 38.72 | 33.22 | **−14.20%** |
| 16m-half fp16 | 38.47 | 33.36 | **−13.28%** |
| 256m fp16 | 704.86 | 669.75 | −4.98% |

bf16：`T.vabs` 无 bf16 dtype 边，保留基线两条 mask 链（+静态尾块/流水），−2.0~−2.9%。

## Final Performance Test Data

官方最终测量：`perf_opt/IsfiniteFwdOp.py`（round 2，baseline/final 相邻交错，各 1 次独立 msprof）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| isfinite_fwd_kernel | 16m-bfloat16 | 58.431 | 56.811 | abs-pipe2-bs8192 |
| isfinite_fwd_kernel | 16m-float16 | 38.091 | 33.371 | abs-pipe2-bs8192 |
| isfinite_fwd_kernel | 16m-float32 | 52.081 | 50.481 | abs-pipe2-bs8192 |
| isfinite_fwd_kernel | 16m-half-bfloat16 | 58.081 | 56.781 | abs-pipe2-bs8192 |
| isfinite_fwd_kernel | 16m-half-float16 | 38.631 | 33.321 | abs-pipe2-bs8192 |
| isfinite_fwd_kernel | 256m-bfloat16 | 931.809 | 911.728 | abs-pipe2-bs8192 |
| isfinite_fwd_kernel | 256m-float16 | 710.714 | 670.963 | abs-pipe2-bs8192 |

### SOL 对账

| workload | baseline SOL% | final SOL% | 判定（≥70%） |
|---|---:|---:|:--:|
| 16m-bfloat16 | 58.431 | 56.811 | -2.77% | 40.74 | 42.21 | ❌ |
| 16m-float16 | 38.091 | 33.371 | -12.39% | 48.71 | 57.44 | ❌ |
| 16m-float32 | 52.081 | 50.481 | -3.07% | 73.58 | 76.33 | ✅ |
| 16m-half-bfloat16 | 58.081 | 56.781 | -2.24% | 41.45 | 42.08 | ❌ |
| 16m-half-float16 | 38.631 | 33.321 | -13.75% | 48.59 | 58.89 | ❌ |
| 256m-bfloat16 | 931.809 | 911.728 | -2.16% | 45.31 | 46.47 | ❌ |
| 256m-float16 | 710.714 | 670.963 | -5.59% | 59.05 | 63.10 | ❌ |

**结论（如实）**：
- **7 个 tune case 中 1 个 ≥70%**（16m fp32 76.33%）；fp16 57–63%、bf16 42–46%；
- fp16 收益 −5.0~−13.8%（4-pass vabs）；bf16 −2.2~−2.8%（仅结构收益）；
- 与 Isnan/Isinf 同源上限（CG-2026-0027），按实际 ceiling 记录，不宣称达标。

## 精度回归
`python perf_opt/IsfiniteFwdOp.py --level all` → **All check passed!**，max_diff=0（NaN/±inf/−0 全覆盖）。

## Skill Retrospective
- **P**：`isfinite = |x| < +inf` 把 6 pass 压到 4 pass，且 NaN 自动排除——predicate 族的可复用形态。
- **R**：bf16 仍受 `vabs` dtype 边限制，只能拿结构收益；不假设与 fp16 同构。
- 证据：`logs/l0_abs.log`、`logs/final_level_all.log`、`candidates/isfinite_fwd_kernel_abs.py`。

## 证据索引
`perf_records.jsonl`（round 1 baseline/candidate 7+7；round 2 baseline/final 7+7）。
