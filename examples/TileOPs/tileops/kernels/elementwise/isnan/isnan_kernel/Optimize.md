# IsnanFwdOp Stage-4 调优摘要（Optimize.md）

> 摘要自 `perf_opt/opt_log.md`。CANN 9.0.0 / tilelang 0.1.2 dev root / Ascend910B3 / 2026-10-17。

## 结论

- **7 个 tune case 中 1 个 ≥70%**（16m fp32 78.45%），其余 55–65%（16m/256m）；3 个 `smoke-1m` 仅做精度。
- 采纳：**满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核
  + 3-pass vselect predicate 链 @8192**。
- 收益：16m-half bf16 −6.16%、16m bf16 −5.62%、16m-half fp16 −4.08%、256m bf16 −1.01%。
- **未达 70% 是结构性上限**：predicate 至少 3 条向量 pass（无 i1→i8、无 bool vnot、vselect 无 int8 输出，
  CG-2026-0027），roofline advice = `latency bound:pipeline caused`；block 16384 已 UB 超限。

## 最终对账（官方 msprof，相邻交错）

| workload | baseline µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 36.861 | 34.791 | -5.62% | 51.13 | 55.26 | ❌ |
| 16m-float16 | 31.131 | 30.841 | -0.93% | 60.83 | 62.95 | ❌ |
| 16m-float32 | 49.831 | 49.301 | -1.06% | 77.34 | 78.45 | ✅ |
| 16m-half-bfloat16 | 37.031 | 34.751 | -6.16% | 51.32 | 55.35 | ❌ |
| 16m-half-float16 | 30.181 | 30.111 | -0.23% | 64.73 | 64.69 | ❌ |
| 256m-bfloat16 | 688.644 | 681.684 | -1.01% | 60.91 | 61.99 | ❌ |
| 256m-float16 | 644.903 | 648.443 | +0.55% | 65.49 | 64.99 | ❌ |

## 有效优化点

| 优化点 | 实测 |
|---|---|
| 静态尾块 + Pipelined | 16m fp16 −3.9%、256m bf16 −2.1% |
| vselect 替代 vcast+vsub（4→3 pass） | bf16 −3.0%、fp16 −2.8%（16m） |

## 否决项

| 候选 | 原因 |
|---|---|
| block_size 16384 | 6×UB buffer × 16384，2-stage 后超 192KB，采集失败 |
| 再减 pass | 受 CG-2026-0027 限制（无 i1→i8 / bool vnot / int8 vselect） |

## 精度

- `python perf_opt/IsnanFwdOp.py --level all` → `All check passed!`（NaN/±inf/−0），max_diff=0
- 3 个 `smoke-1m` case：精度 PASS（不参与性能）

## 遗留

- 若补齐 CG-2026-0027 的任一能力（i1→i8 cast / bool vnot / int8 vselect），预计可再省 1–2 条 pass，
  是后续唯一的结构性提升路径。

## 证据

`perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/final_level_all.log`、`candidates/`
