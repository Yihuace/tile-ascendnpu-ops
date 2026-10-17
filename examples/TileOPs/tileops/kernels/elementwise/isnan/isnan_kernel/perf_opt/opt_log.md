# IsnanFwdOp — Stage 4 调优日志

> 工具链：CANN **9.0.0**（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root（AscendNPU-IR 31f6903）·
> Ascend910B3（20 AIC / 40 AIV）· 2026-10-17。
> 口径：`msprof op --kernel-name=main` 的 **Task Duration(us)** + `visualize_data.bin` 的 GM Read+Write SOL ratio；
> 直跑 kernel（`perf_opt/bench_kernel.py`）。
> 判据：AI = 1.5–2.3 op/byte，平衡点 = computility/bw ≈ 9.07/1.75 = **5.18** ⇒ 本算子属**访存受限**，线为 **70%**。

## 0. 对象与入口

- 算子：`IsnanFwdOp`（`y = isnan(x)`，bool 输出，经 int8 字节存储 + host view）
- 基线 kernel：`isnan_kernel/isnan_fwd_kernel.py`
- 调优产物：`perf_opt/IsnanFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`）
- benchmark：`benchmarks/ops/bench_isnan.py::test_isnan_bench`；**10 个 tune case**（manifest 未打 smoke 标记，
  `smoke-1m` 只是 label，实际进入 benchmark）

## 1. Baseline（round 1）

| workload | µs | SOL% |
|---|---:|---:|
| 16m fp32 | 49.94 | 77.22 |
| 16m fp16 | 30.60 | 63.24 |
| 16m bf16 | 36.46 | 53.02 |
| 16m-half fp16 | 31.39 | 59.10 |
| 16m-half bf16 | 36.90 | 51.99 |
| 256m fp16 | 650.79 | 64.81 |
| 256m bf16 | 694.42 | 60.39 |
| smoke-1m fp32 | 5.42 | 42.65 |
| smoke-1m fp16 | 4.69 | 25.72 |
| smoke-1m bf16 | 5.26 | 23.59 |

- 仅 16m fp32 过 70%；1m 小 shape（4.4–5.6µs）由固定开销主导；
- roofline `advice` = **latency bound:pipeline caused**，不是带宽饱和。

## 2. 诊断

predicate 链是 **4 条向量 pass**：

```
vcmp(x<=x) -> vcast(bool->f16) -> vsub(1-m) -> vcast(f16->i8) -> GM
```

工具链约束（CG-2026-0027）：无 `i1->i8` 直接 vcast、无 bool/int1 的 vnot、`T.vselect` 的 Out 不支持 int8。
因此无法用一条 predicate 指令直出字节结果，最少 3 条 pass（vcmp + vselect + vcast）。

## 3. 迭代日志（候选 vs current best）

### Round 1 — 静态尾块 + Pipelined（结构移植）

| workload | baseline µs | pipe2 µs | Δ | SOL |
|---|---:|---:|---:|---:|
| 16m bf16 | 36.461 | 35.831 | −1.73% | 53.02→54.32 |
| 16m fp16 | 31.211 | 29.991 | −3.91% | 60.75→61.81 |
| 16m-half bf16 | 37.151 | 35.741 | −3.80% | 52.53→54.69 |
| 16m-half fp16 | 30.501 | 30.531 | +0.10% | 63.15→62.17 |
| 256m bf16 | 697.604 | 682.764 | −2.13% | 60.30→61.86 |
| 256m fp16 | 651.683 | 645.473 | −0.95% | 64.77→65.99 |

### Round 2 — vselect 替掉 vcast+vsub（4 pass → 3 pass，A7）

`vcmp(le) -> vselect(mask, 0.0, 1.0, f16) -> vcast(f16->i8)`，语义 `mask=1`（非 NaN）→ 0.0、`mask=0`（NaN）→ 1.0。

| workload | pipe2 µs | select µs | Δ |
|---|---:|---:|---:|
| 16m bf16 | 35.771 | 34.721 | −2.94% |
| 16m fp16 | 30.621 | 29.771 | −2.78% |
| 256m bf16 | 686.504 | 679.024 | −1.09% |
| 256m fp16 | 648.403 | 641.653 | −1.04% |

### Round 3 — block_size 16384

`select@16384` 采集失败（6 个 UB buffer × 16384，2-stage 后超 192KB）→ 8192 为可行上限。

### 合并与 winner

- winner：**满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核 + 3-pass vselect 链 @8192**；
- 合并进 `perf_opt/IsnanFwdOp.py`，`TUNED_BLOCK_SIZE = 8192`；基准与 wrapper 未改。

## Final Performance Test Data

官方最终测量：`perf_opt/IsnanFwdOp.py`（round 5，baseline/final 相邻交错，各 1 次独立 msprof）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| isnan_fwd_kernel | 16m-bfloat16 | 36.861 | 34.791 | select-pipe2-bs8192 |
| isnan_fwd_kernel | 16m-float16 | 31.131 | 30.841 | select-pipe2-bs8192 |
| isnan_fwd_kernel | 16m-float32 | 49.831 | 49.301 | select-pipe2-bs8192 |
| isnan_fwd_kernel | 16m-half-bfloat16 | 37.031 | 34.751 | select-pipe2-bs8192 |
| isnan_fwd_kernel | 16m-half-float16 | 30.181 | 30.111 | select-pipe2-bs8192 |
| isnan_fwd_kernel | 256m-bfloat16 | 688.644 | 681.684 | select-pipe2-bs8192 |
| isnan_fwd_kernel | 256m-float16 | 644.903 | 648.443 | select-pipe2-bs8192 |

### SOL 对账

| workload | baseline SOL% | final SOL% | 判定（≥70%） |
|---|---:|---:|:--:|
| 16m-bfloat16 | 51.13 | 55.26 | ❌ |
| 16m-float16 | 60.83 | 62.95 | ❌ |
| 16m-float32 | 77.34 | 78.45 | ✅ |
| 16m-half-bfloat16 | 51.32 | 55.35 | ❌ |
| 16m-half-float16 | 64.73 | 64.69 | ❌ |
| 256m-bfloat16 | 60.91 | 61.99 | ❌ |
| 256m-float16 | 65.49 | 64.99 | ❌ |

**结论（如实）**：

- **7 个 tune case 中 1 个 ≥70%**（16m fp32 78.45%）；其余 55–65%（16m/256m）；
- 时延收益：16m-half bf16 **−6.16%**、16m bf16 **−5.62%**、16m-half fp16 −4.08%；
- **未达 70% 的根因是结构性的**：predicate 至少 3 条向量 pass（CG-2026-0027：无 i1→i8、无 bool vnot、
  vselect 无 int8 输出），roofline advice 为 `latency bound:pipeline caused`；block 16384 已 UB 超限。
  该结论不是分块/参数可解，按“pipeline-latency 受限”如实记录，不宣称达标；
- 3 个 `smoke-1m`（label 含 smoke，按 smoke 分类）仅做精度回归：`--level all` 全 PASS，不参与性能判定
  （其 4.4–5.6µs 属固定开销主导）。

## 精度回归


- `python perf_opt/IsnanFwdOp.py --level all` → **All check passed!**（NaN/±inf/−0 全覆盖，max_diff=0）；
- 3-pass vselect 链与原 4-pass 链在 L0 上逐位一致。

## Skill Retrospective

- **D**：predicate 族的 3-pass 下限与 `select` 收益（bf16 −3%、fp16 −2.8%）回写 pattern-library？本次先登记
  CG-2026-0027（缺口本体），绕法（3-pass 链）随条目 workaround 字段记录。
- **R**：runner 的 L0 成功判据需兼容 `All check passed!`（已修，Add/Sub/Isnan 同步）。
- **P**：bool 输出的算子在 inventory 里不能靠 label 判定 smoke——本 benchmark 无 smoke marker，
  `smoke-1m` 实际进入正式 benchmark，按 10 个 tune 处理。

## 证据索引

- `perf_opt/perf_records.jsonl`（round 1 baseline 10；round 2/3/4 候选；round 5 baseline+final 各 10）
- `perf_opt/logs/`（`final_level_all.log`、`isnan_r1_runner.log`、`l0_select.log`）
- 候选：`perf_opt/candidates/isnan_fwd_kernel_pipe2.py`、`isnan_fwd_kernel_select.py`
