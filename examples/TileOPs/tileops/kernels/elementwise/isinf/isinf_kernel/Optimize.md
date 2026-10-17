# IsinfFwdOp Stage-4 调优摘要（Optimize.md）

> 摘要自 `perf_opt/opt_log.md`。CANN 9.0.0 / Ascend910B3 / 2026-10-17。

## 结论

- **7 个 tune case 中 1 个 ≥70%**（16m fp32 76.03%）；fp16 57–63%，bf16 42–46%。
- 采纳：fp16/fp32 用 **`vabs → vcmp(eq +inf) → vcast → vcast`（4 pass，vs 基线 6 pass）**；
  bf16 保持基线两条 mask 合并（`vabs` 无 bf16 边，`vselect` 实测更慢）。
- 收益：16m-half fp16 **−11.94%**、16m fp16 **−12.24%**、256m fp16 **−5.61%**、16m fp32 −4.17%。
- 未达 70% 属 predicate 族结构上限（CG-2026-0027），非分块可解。

## 最终对账

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 57.071 | 56.671 | -0.70% | 41.80 | 42.37 | ❌ |
| 16m-float16 | 38.561 | 33.841 | -12.24% | 50.22 | 58.07 | ❌ |
| 16m-float32 | 52.761 | 50.561 | -4.17% | 71.92 | 76.03 | ✅ |
| 16m-half-bfloat16 | 57.251 | 56.761 | -0.86% | 41.83 | 42.65 | ❌ |
| 16m-half-float16 | 37.951 | 33.421 | -11.94% | 50.11 | 57.37 | ❌ |
| 256m-bfloat16 | 917.428 | 915.318 | -0.23% | 46.05 | 46.29 | ❌ |
| 256m-float16 | 709.204 | 669.443 | -5.61% | 59.76 | 62.94 | ❌ |

## 精度

`python perf_opt/IsinfFwdOp.py --level all` → `All check passed!`，max_diff=0。

## 证据

`perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/`
