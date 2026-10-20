# 优化报告：ClampScalarFwdOp（Stage-4 optimize）

> 本报告记录**优化方法、优化过程、优化结果与产物**。目标算子族中的微型 shape 按用户决定**豁免**并标注「物理不可达」（见 §6）。

## 0. 身份与范围

| 项 | 值 |
|---|---|
| 算子 | `ClampScalarFwdOp`（复用同 kernel：`ClampMinFwdOp`） |
| family | elementwise |
| 算子类型 | **单 kernel / 单 func / 纯 Vector（AIV）**，非 Mixed |
| kernel 目录 | `examples/TileOPs/tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/` |
| 性能口径 | `msprof op Task Duration(us)`（唯一） |
| 工具链 | tilelang `0.1.2+417dae99` / **上一代工具链**（版本见 `docs/cann9_adaptation.md` §13 两代对照）/ Ascend910B3 / torch_npu 2.7.1.post4 |

## 1. 优化方法

**方法：去除被实测证伪为冗余的 NaN 恢复。**

1. **瓶颈定位**（baseline `msprof` + `PipeUtilization.csv`，16M fp16）：
   `aiv_vec_ratio = 0.72`（最高）、`mte2_ratio = 0.56`、`mte3_ratio = 0.28`
   → **向量单元是瓶颈，不是 DMA**（`mte2_ratio ≪ 0.95`，计算链未隐藏，符合 pattern-library PL-1.6 判据）。
2. **根因**：fp32 计算链为
   `vcast→vclamp/vmax/vmin→vcmp(NaN)→vselect(NaN)→vcast`（5 条向量 op）。
   其中 NaN 恢复（`vcmp`+`vselect`）占 2/5，是设计 R2 因「vmax 未文档化 NaN 传播」而加的。
3. **证伪实验**：实测本工具链上 `vmax`/`vmin`/`vclamp`（fp32 域）**已传播 NaN**——
   3 种边界形态 × fp16 × 586 个 NaN 点全部 `nan_ok=True`；`vclamp` 有限值 `max_diff=0`；`inf/-inf` 与 torch 一致。
4. **改动**：删除 `vcmp`+`vselect`（向量链 5→3），`nan-bound` 分支 `vdiv` 改原地写。
   改动**只落在 `perf_opt/`，基线文件未动**；产物仅 `perf_opt/clamp_scalar_fwd_kernel.py`。

## 2. 优化过程（阶段记录）

| 阶段 | 动作 | 结果 |
|---|---|---|
| 预检 | 读 `conductor-scenarios/optimize.md`、`tilelang-op-optimize` skill、pattern-library INDEX/hardware-context/elementwise.md | 确定口径与手法 |
| Baseline | `tileops-report run --op ClampScalarFwdOp` + 单例 `PipeUtilization.csv` | 16M fp16 = 42.1µs / 45.5%；向量瓶颈 |
| 微基准 | block_size∈{2048,4096,8192} × num_cores∈{auto,20,40} 扫描 | 默认已最优，config 调参无收益 |
| NaN 证伪 | 3 形态 NaN/inf 传播实测 | NaN 已原生传播 → 恢复冗余 |
| 变体 | 写 `perf_opt/`（去恢复） | — |
| 回归 | `perf_opt/...py --level all` | 全绿（含 Boundary NaN/inf） |
| 采纳 | 翻 wrapper 切换块注释 → perf_opt | — |
| 验收 | `tileops-report run --op ClampScalarFwdOp` | 正确性 11/11，`status=passed` |

## 3. 优化结果（msprof op，baseline → perf_opt）

| label | dtype | base µs | opt µs | Δlat | base ratio | opt ratio |
|---|---|---:|---:|---:|---:|---:|
| 16M | float16 | 42.1 | 26.1 | **−38.0%** | 45.5% | **73.5%** |
| 16M | bfloat16 | 42.3 | 25.9 | **−38.7%** | 51.7% | **74.0%** |
| 16M | float32 | 68.5 | 46.3 | **−32.4%** | 56.0% | **82.8%** |
| 256M | float16 | 891.4 | 751.4 | −15.7% | 62.4% | **73.9%** |
| 256M | bfloat16 | 884.6 | 773.3 | −12.6% | 62.8% | **71.9%** |
| 16M-min-only | fp16/bf16/fp32 | 39.3 / 38.7 / 64.0 | 26.2 / 24.7 / 49.4 | −33.5% / −36.1% / −22.7% | 48.7 / 49.5 / 60.0 | **73.3 / 77.5 / 77.6** |
| 16M-max-only | fp16/bf16/fp32 | 38.6 / 37.5 / 63.3 | 26.2 / 25.6 / 47.6 | −32.2% / −31.8% / −24.8% | 49.7 / 51.1 / 60.6 | **73.3 / 75.0 / 81.0** |

**结论：所有 shape 的带宽利用率由 45–63% 提升至 71.9%–82.8%，全部 ≥60% 目标；延迟降 12%–39%。**

## 4. 时间与 token 消耗（本次优化过程）

| 项 | 值 | 来源 |
|---|---|---|
| 开始（UTC） | 2026-10-15T11:37:04Z | `reports/optimize/ClampScalarFwdOp/.start_utc` |
| 结束（UTC） | 2026-10-15T12:50:54Z | `date -u` |
| 墙钟时长 | **≈ 73.8 min** | 相减 |
| token input | 23,391 | `~/.local/share/opencode/opencode.db`（`message.data.tokens`，窗口内求和） |
| token output | 7,638 | 同上 |
| token reasoning | 20,067 | 同上 |
| token 合计 | **51,096** | input+output+reasoning |
| cost | $0.0510 | 同上 |

> 取数窗口 = 开始时刻→结束时刻；`time_created` 为 epoch ms。token 数为该窗口内 `message` 表求和，受 db 记录粒度影响，供量级参考。

## 5. 优化产物

| 产物 | 路径 |
|---|---|
| 优化 kernel | `tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/clamp_scalar_fwd_kernel.py` |
| 过程日志 | `tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/opt_log.md` |
| 结构化记录 | `tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/perf_records.jsonl` |
| baseline 报告 | `reports/tileops/20261015_195138_601192_ClampScalarFwdOp/` |
| perf_opt 报告 | `reports/tileops/20261015_202825_275014_ClampScalarFwdOp/`（status=passed） |
| wrapper 切换 | `clamp_scalar.py` 的 baseline/perf_opt import 块已翻至 perf_opt |

## 6. 微型 shape 豁免说明（物理不可达）

用户已同意：**微型 shape 豁免 60% 目标，标注「物理不可达」**。

判据（以 Hardsigmoid `mbv3-se-gate`、shape `[32,240,1,1]` fp16 为例）：
- 数据量 ≈ 30 KB；按设备地板 1.26 TB/s 理论搬运 ≈ **24 ns**；实测 kernel ≈ **2.86 µs**；
- 一个 kernel 有与数据量无关的**固定成本**（派发/调度 + 流水填充排空 + 收尾同步）≈ 2 µs 级；
- 即便数据搬运瞬间完成，利用率上限 ≈ `30KB/(2µs×1.75TB/s) ≈ 0.86%`。
⇒ 此类 shape **60% 在数学上不可达**；其正确指标是**绝对延迟（2–4 µs）**，不是带宽利用率。

本算子（ClampScalar/ClampMin）**无微型 shape**，故本报告不涉及豁免项；此节为族内通用政策记录。

## 7. 复核入口

```bash
cd examples/TileOPs
# 正确性
python -m pytest tests/ops/test_clamp_scalar.py tests/ops/test_clamp_min.py -q
# 分层回归（perf_opt）
python tileops/kernels/elementwise/clamp_scalar/clamp_scalar_kernel/perf_opt/clamp_scalar_fwd_kernel.py --level all
# 端到端报告
python -m tileops.reporting.cli run --op ClampScalarFwdOp --prof-mode msprof
```
