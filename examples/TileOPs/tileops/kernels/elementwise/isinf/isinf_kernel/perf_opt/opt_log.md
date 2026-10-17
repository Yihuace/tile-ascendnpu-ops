# IsinfFwdOp — Stage 4 调优日志

> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root（AscendNPU-IR 31f6903）·
> Ascend910B3（40 AIV）· 2026-10-17。口径：msprof op Task Duration(us) + GM Read+Write SOL ratio。
> 判据：AI ≈ 2.0 → 访存受限，线 70%。

## 0. 对象与入口

- `IsinfFwdOp`（`y = isinf(x)`，bool 经 int8 字节存储）；基线 `isinf_kernel/isinf_fwd_kernel.py`；
- 产物 `perf_opt/IsinfFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`）；benchmark `bench_isinf.py`（7 tune + 3 smoke）。

## 1. Baseline（round 1，tune）

| workload | µs | SOL% |
|---|---:|---:|
| 16m fp32 | 52.77 | 73.94 |
| 16m fp16 | 38.98 | 50.19 |
| 16m bf16 | 57.38 | 42.37 |
| 16m-half fp16 | 38.34 | 50.46 |
| 16m-half bf16 | 57.23 | 42.45 |
| 256m fp16 | 711.45 | 59.01 |
| 256m bf16 | 918.79 | 46.02 |

基线链 6 条向量 pass：`vcmp(eq +inf) + vcmp(eq -inf) + vcast + vcast + vadd + vcast`。

## 2. 迭代日志

### Round 2 — vabs 合并正负 inf（fp16/fp32；候选 vs current best，Task Duration）

`isinf(x) = (|x| == +inf)`：`vabs → vcmp(eq) → vcast(bool→f16) → vcast(f16→i8)`，**6 pass → 4 pass**。

| workload | baseline µs | abs µs | Δ |
|---|---:|---:|---:|
| 16m fp16 | 38.02 | 33.52 | **−11.84%** |
| 16m-half fp16 | 38.83 | 33.71 | **−13.19%** |
| 256m fp16 | 706.20 | 671.25 | −4.95% |

- **bf16 不适用**：`T.vabs` dtype 表无 bf16。先试 `vselect` 合并两条 mask，实测 **+6.6~8.6%（更慢）**，
  已回退为基线的 `2×vcmp + 2×vcast + vadd + vcast`（Round 3 复测 −0.2~−0.9%，中性）。

### Round 3 — bf16 回退验证

| workload | baseline µs | 候选 µs | Δ |
|---|---:|---:|---:|
| 16m bf16 | 57.411 | 57.161 | −0.44% |
| 256m bf16 | 918.268 | 915.058 | −0.35% |

### 合并与 winner

- fp16/fp32：**4-pass vabs 链**；bf16：**基线 6-pass 两条 mask 合并**；两者共用静态尾块 + Pipelined 结构；
- 合并进 `perf_opt/IsinfFwdOp.py`，基准与 wrapper 未改。

## Final Performance Test Data

官方最终测量：`perf_opt/IsinfFwdOp.py`（round 4，baseline/final 相邻交错，各 1 次独立 msprof）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| isinf_fwd_kernel | 16m-bfloat16 | 57.071 | 56.671 | abs-pipe2-bs8192 |
| isinf_fwd_kernel | 16m-float16 | 38.561 | 33.841 | abs-pipe2-bs8192 |
| isinf_fwd_kernel | 16m-float32 | 52.761 | 50.561 | abs-pipe2-bs8192 |
| isinf_fwd_kernel | 16m-half-bfloat16 | 57.251 | 56.761 | abs-pipe2-bs8192 |
| isinf_fwd_kernel | 16m-half-float16 | 37.951 | 33.421 | abs-pipe2-bs8192 |
| isinf_fwd_kernel | 256m-bfloat16 | 917.428 | 915.318 | abs-pipe2-bs8192 |
| isinf_fwd_kernel | 256m-float16 | 709.204 | 669.443 | abs-pipe2-bs8192 |

### SOL 对账

| workload | baseline SOL% | final SOL% | 判定（≥70%） |
|---|---:|---:|:--:|
| 16m-bfloat16 | 57.071 | 56.671 | -0.70% | 41.80 | 42.37 | ❌ |
| 16m-float16 | 38.561 | 33.841 | -12.24% | 50.22 | 58.07 | ❌ |
| 16m-float32 | 52.761 | 50.561 | -4.17% | 71.92 | 76.03 | ✅ |
| 16m-half-bfloat16 | 57.251 | 56.761 | -0.86% | 41.83 | 42.65 | ❌ |
| 16m-half-float16 | 37.951 | 33.421 | -11.94% | 50.11 | 57.37 | ❌ |
| 256m-bfloat16 | 917.428 | 915.318 | -0.23% | 46.05 | 46.29 | ❌ |
| 256m-float16 | 709.204 | 669.443 | -5.61% | 59.76 | 62.94 | ❌ |

**结论（如实）**：

- **7 个 tune case 中 1 个 ≥70%**（16m fp32 76.03%）；fp16 57–63%（较基线 +5.6~+12.2%）、bf16 42–46%；
- 收益集中在 fp16（4-pass vabs 链）；bf16 因无 vabs 边、且 vselect 更慢，只能保持 6-pass，收益 ~0；
- 与 Isnan 同源的结构上限（CG-2026-0027）：无 `i1→i8` 直连、无 bool `vnot`、`vselect` 无 int8 输出，
  predicate 最少 3–4 条向量 pass；report 按实际 ceiling 写，不宣称达标。

## 精度回归

- `python perf_opt/IsinfFwdOp.py --level all` → **All check passed!**（NaN/±inf/−0 全覆盖，max_diff=0）。

## 证据

`perf_records.jsonl`（round 1 baseline 7；round 2/3 候选；round 4 baseline+final 各 7）、
`logs/l0_abs.log`、`logs/final_level_all.log`、`candidates/isinf_fwd_kernel_abs.py`。

## Skill Retrospective

- **D（实测）**：`isinf(x)=|x|==+inf` 的 vabs 4-pass 链让 fp16 −12%，是 predicate 族目前最大的单点收益；
  bf16 无 vabs 边、vselect 更慢，收益为 0。回写：CG-2026-0027 的 workaround 字段已覆盖 4-pass 形态。
- **R（流程）**：同一 predicate 族的 bf16 路径不能假设与 fp16 同构——`vselect` 在 bf16 上慢 6–9%；
  任何 bf16 候选必须先做相邻 A/B。
- **P（模式）**：`isfinite` 可用 `|x| < +inf` 同一 4-pass 形态（NaN 有序比较为 false，天然排除）。

## 证据索引

`perf_records.jsonl`、`logs/l0_abs.log`、`logs/final_level_all.log`、`candidates/isinf_fwd_kernel_abs.py`
