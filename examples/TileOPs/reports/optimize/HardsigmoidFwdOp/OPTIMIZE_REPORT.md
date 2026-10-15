# 优化报告：HardsigmoidFwdOp（Stage-4 optimize）

## 0. 身份与范围

| 项 | 值 |
|---|---|
| 算子 | `HardsigmoidFwdOp` |
| 类型 | 单 kernel / 单 func / 纯 Vector（AIV） |
| kernel 目录 | `tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/` |
| 性能口径 | `msprof op Task Duration(us)` |
| 工具链 | tilelang `0.1.2+417dae99` / **CANN 8.5.2** / Ascend910B3 |

## 1. 优化方法（两项）

1. **去除冗余 NaN 恢复**：baseline 计算链为
   `vcast→vadd(x+3)→vclamp(0,6)→vmul(1/6)→vbrc(NaN 常量)+vcmp(NaN 检测)+vselect(NaN 回填)→vcast`。
   实测本工具链上 `vadd+vclamp+vmul` **已传播 NaN**（NaN→NaN）→ 恢复的 3 条 op 冗余，删除（6→3 条向量 op）。
2. **修 fp32 UB 溢出（原 baseline bug）**：`_DEFAULT_BLOCK_SIZE` 原为标量 8192，fp32 在 Expert 多缓冲下需 ~224KB>192KB → `smoke-1m-float32` 编译失败，**baseline 报告本就 `partial`**。改为按 dtype：fp32=4096、fp16/bf16=8192；且 `__init__` 构造时即传该 block_size。

产物仅落 `perf_opt/`（基线 kernel 文件未动）；wrapper import 切换块翻至 perf_opt。

## 2. 优化过程

| 阶段 | 动作 | 结果 |
|---|---|---|
| 基线 | `tileops-report run --op HardsigmoidFwdOp`（20261015_113947） | status=**partial**（fp32 编译失败） |
| NaN 证伪 | perf_opt 去恢复后跑 `--level all` | 全绿（含 NaN→NaN、inf） |
| fp32 修复 | block_size 按 dtype | 编译通过 |
| 验收 | `tileops-report run --op HardsigmoidFwdOp`（20261015_214729） | status=**passed**，正确性 7/7 |

## 3. 优化结果（msprof op，baseline → perf_opt）

| label | dtype | base µs | opt µs | Δlat | base ratio | opt ratio |
|---|---|---:|---:|---:|---:|---:|
| fc-wide | float16 | 28.98 | 17.2 | **−40.6%** | 50.5% | 55.7% |
| fc-wide | bfloat16 | 28.88 | 16.9 | **−41.5%** | 57.0% | **64.6%** |
| mbv3-se-gate | float16 | 2.86 | 2.48 | −13.3% | 0.5% | 0.4%（豁免） |
| mbv3-se-gate-deep | bfloat16 | 2.68 | 2.06 | −23.1% | 2.4% | 2.1%（豁免） |

> **有效 shape（fc-wide）延迟降 40%+**；bf16 达 64.6%（≥60%）。fc-wide fp16 实测 55.7%（两次测量在 55.7%–60.2% 间波动，贴近但偶低于 60%，属测量方差 + 该 kernel 仍含 3 条计算 op 的残差）。
> `mbv3-se-gate*` 为 2–3µs 微型 shape，**物理不可达**（固定启动开销主导），按用户决定豁免。

## 4. 时间与 token 消耗

| 项 | 值 | 来源 |
|---|---|---|
| 开始（UTC） | 2026-10-15T13:30:31Z | `reports/optimize/HardsigmoidFwdOp/.start_utc` |
| 结束（UTC） | 2026-10-15T14:03:29Z | `date -u` |
| 墙钟时长 | **≈ 33.0 min** | 相减 |
| token input / output / reasoning | 5,577 / 2,786 / 3,201 | `~/.local/share/opencode/opencode.db` 窗口求和 |
| token 合计 / cost | **11,564** / $0.0212 | 同上 |

> token 取数为 db 窗口求和，受记录粒度影响，供量级参考。

## 5. 优化产物

| 产物 | 路径 |
|---|---|
| 优化 kernel | `.../hardsigmoid_kernel/perf_opt/hardsigmoid_fwd_kernel.py` |
| 过程日志/记录 | `.../perf_opt/opt_log.md`（见下）、`perf_opt/perf_records.jsonl` |
| baseline 报告 | `reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/` |
| perf_opt 报告 | `reports/tileops/20261015_214729_104006_HardsigmoidFwdOp/`（passed） |

## 6. 复核入口

```bash
cd examples/TileOPs
python -m pytest tests/ops/test_hardsigmoid.py -q
python tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/perf_opt/hardsigmoid_fwd_kernel.py --level all
python -m tileops.reporting.cli run --op HardsigmoidFwdOp --prof-mode msprof
```
