# ClampScalarFwdKernel — Stage-4 optimize log

- **算子**：`ClampScalarFwdOp`（复用同 kernel 者：`ClampMinFwdOp`）
- **kernel**：`tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/`
- **基线**：`clamp_scalar_kernel/clamp_scalar_fwd_kernel.py`（Stage 3，纯 Vector / AIV）
- **优化版**：`clamp_scalar_kernel/perf_opt/clamp_scalar_fwd_kernel.py`
- **唯一性能口径**：`msprof op Task Duration(us)`
- **工具链**：tilelang `0.1.2+417dae99`(dev-root) / **CANN 8.5.2** / Ascend910B3 / torch_npu 2.7.1.post4
- **任务窗口**：2026-10-15T11:37:04Z → 2026-10-15T12:50:54Z（≈73.8 min）

## 1. 瓶颈定位（Phase 1）

baseline `msprof op` + `PipeUtilization.csv`（16M fp16，40 vector core）：

| 指标 | 值 |
|---|---|
| Task Duration | 42.1 µs |
| `aiv_vec_ratio` | **0.72（最高）** |
| `aiv_mte2_ratio` | 0.56 |
| `aiv_mte3_ratio` | 0.28 |
| `aiv_scalar_ratio` | 0.08 |
| `aiv_mte2_active_bw` | ~39 GB/s/core |

→ **向量单元（AIV VEC）是瓶颈，不是 DMA**（`mte2_ratio` 0.56 ≪ 0.95，计算链未隐藏；符合 pattern-library PL-1.6「mte2_ratio≥0.95 才搬运饱和」判据）。

## 2. 优化点：去除冗余的 NaN 恢复（5 条向量 op → 3 条）

baseline 的 fp32 计算链为
`vcast(f16→f32) → vclamp/vmax/vmin → vcmp(x,x,"ne") → vselect(回填 x32) → vcast(f32→f16)`
（设计 R2 因「Ascend vmax/vmin 未文档化 NaN 传播」而加了 `vcmp`+`vselect` 两点恢复）。

**实测证伪**（本任务，本工具链）：`vmax`/`vmin`/`vclamp` 在 fp32 域**已传播 NaN**——
- 3 种边界形态（min-only / max-only / both）× fp16，586 个 NaN 点：`nan_ok=True`；
- `vclamp` 有限值 `max_diff=0.00e+00`；`inf/-inf` 与 torch 一致。

⇒ 恢复两点冗余，删除。向量链降为 `vcast → vclamp/vmax/vmin → vcast`（−2/5 向量 op）。
（另：`nan-bound` 分支的 `vdiv` 改原地写 `buf1`，`buf2` 不再使用。）

## 3. 正确性回归（阻塞门禁）

`python perf_opt/clamp_scalar_fwd_kernel.py --level all` → **All check passed**
（L0/L1/L2/Boundary 全绿；含 NaN→NaN 512 点、NaN-bound→全 NaN、inf/-inf、min>max）。
接入 wrapper 后 `tileops.reporting.cli run --op ClampScalarFwdOp` 正确性 **11/11**、`status=passed`。

## 4. 性能对比（msprof op，baseline → perf_opt）

| label | dtype | base µs | opt µs | Δlat | base ratio | opt ratio |
|---|---|---:|---:|---:|---:|---:|
| 16M | float16 | 42.1 | 26.1 | **−38.0%** | 45.5% | **73.5%** |
| 16M | bfloat16 | 42.3 | 25.9 | **−38.7%** | 51.7% | **74.0%** |
| 16M | float32 | 68.5 | 46.3 | **−32.4%** | 56.0% | **82.8%** |
| 256M | float16 | 891.4 | 751.4 | −15.7% | 62.4% | **73.9%** |
| 256M | bfloat16 | 884.6 | 773.3 | −12.6% | 62.8% | **71.9%** |
| 16M-min-only | float16 | 39.3 | 26.2 | −33.5% | 48.7% | **73.3%** |
| 16M-min-only | bfloat16 | 38.7 | 24.7 | −36.1% | 49.5% | **77.5%** |
| 16M-min-only | float32 | 64.0 | 49.4 | −22.7% | 60.0% | **77.6%** |
| 16M-max-only | float16 | 38.6 | 26.2 | −32.2% | 49.7% | **73.3%** |
| 16M-max-only | bfloat16 | 37.5 | 25.6 | −31.8% | 51.1% | **75.0%** |
| 16M-max-only | float32 | 63.3 | 47.6 | −24.8% | 60.6% | **81.0%** |

**所有 shape ratio ≥71.9%（均超 60% 目标）**，延迟降 12%–39%。

报告：`reports/tileops/20261015_195138_601192_ClampScalarFwdOp/`（baseline）/ `reports/tileops/20261015_202825_275014_ClampScalarFwdOp/`（perf_opt）。

## 5. 结论与沿用

- 采纳（wrapper 切换块已翻至 perf_opt）。**ClampMinFwdOp 复用同 kernel，同步受益**。
- 该优化点（**去除被证伪的冗余 NaN 恢复**）机制清晰：纯 Vector 逐元素算子若 NaN 已由 `vmax/vmin/vclamp` 原生传播，则 `vcmp+vselect` 恢复是纯开销。
- 建议：作为 pattern-library 候选回写（kind: pattern，family: elementwise，token 待 distill）。
