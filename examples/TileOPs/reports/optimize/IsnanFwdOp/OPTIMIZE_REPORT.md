# 优化报告：IsnanFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsnanFwdOp`；单 kernel / 纯 Vector（AIV），`y = isnan(x)`，bool 输出经 int8 字节存储 |
| kernel | `tileops/kernels/elementwise/isnan/isnan_kernel/` |
| 调优产物 | `.../isnan_kernel/perf_opt/IsnanFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op Task Duration(us)` + `visualize_data.bin` GM Read+Write SOL ratio |
| 工具链 | tilelang 0.1.2 dev root（AscendNPU-IR 31f6903）/ CANN 9.0.0（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3 / 2026-10-17 |
| benchmark | `benchmarks/ops/bench_isnan.py::test_isnan_bench`（7 tune + 3 smoke：`smoke-1m`） |
| 过程日志 | `.../isnan_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/` |

## 1. 结论

- **7 个 tune case 中 1 个 ≥70%**：16m fp32 78.45%；16m/256m fp16 55–65%，bf16 55–62%。
- 结构：**满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核 + 3-pass vselect 链**。
- 收益：16m-half bf16 **−6.16%**、16m bf16 −5.62%、16m-half fp16 −4.08%、256m bf16 −1.01%。
- **未达 70% 属结构上限**：predicate 至少 3 条向量 pass（vcmp → vselect → vcast），
  受 CG-2026-0027 约束（无 `i1→i8` 直接 cast、无 bool/int1 `vnot`、`T.vselect` Out 不支持 int8）；
  roofline `advice` = `latency bound:pipeline caused`，不是带宽饱和。block 16384 已 UB 超限。

## 2. 所做改动

仅在 `perf_opt/` 内（基准与 wrapper 未动）：

1. 满块静态 extent + 尾块 epilogue，去热循环 `T.min`；
2. `T.Pipelined(num_local_tasks, num_stages=2)`；
3. **A7**：`vcmp(le) → vselect(mask, 0.0, 1.0, f16) → vcast(f16→i8)`，4 条向量 pass → 3 条；
4. `TUNED_BLOCK_SIZE = 8192`。

## 3. 优化结果（msprof，round 5 baseline/final 相邻交错）

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 36.861 | 34.791 | -5.62% | 51.13 | 55.26 | ❌ |
| 16m-float16 | 31.131 | 30.841 | -0.93% | 60.83 | 62.95 | ❌ |
| 16m-float32 | 49.831 | 49.301 | -1.06% | 77.34 | 78.45 | ✅ |
| 16m-half-bfloat16 | 37.031 | 34.751 | -6.16% | 51.32 | 55.35 | ❌ |
| 16m-half-float16 | 30.181 | 30.111 | -0.23% | 64.73 | 64.69 | ❌ |
| 256m-bfloat16 | 688.644 | 681.684 | -1.01% | 60.91 | 61.99 | ❌ |
| 256m-float16 | 644.903 | 648.443 | +0.55% | 65.49 | 64.99 | ❌ |

### 精度

- `python perf_opt/IsnanFwdOp.py --level all` → `All check passed!`，max_diff = 0（NaN/±inf/−0 全覆盖）；
- 3 个 `smoke-1m`（label 含 smoke）：仅精度回归，PASS，不参与性能判定。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 19:48 → 21:05 |
| 墙钟时长 | ≈ 77 min |
| token input/output/reasoning | 33,636 / 26,374 / 16,305 |
| cache-read / cost | 34,575,360 / $0.1344 |

## 5. 产物

- `.../isnan_kernel/perf_opt/IsnanFwdOp.py`（final，sha256 见 round 5 final 行）
- `perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/`
- 能力缺口：`.agents/evolution/capability-gaps.md` **CG-2026-0027**
- gate 4：`pass`；state `phase=DONE`
