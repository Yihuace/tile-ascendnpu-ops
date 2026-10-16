# 优化报告：Conv1dFwdOp（unit_stride 路径，Stage-4）

## 方法
**把 in-kernel Vector prelude 移到 host**。诊断：`aic_cube_ratio=0.0015`（Cube 99.85% 空转）→ 瓶颈是
Cube 前串行执行的权重重排 + 输入填充（`_prelude`，Cube 阻塞于 `sync_block_wait(4)`）。
perf_opt kernel 删除 `_prelude` 与同步，`w_flat`/`x_pad` 由 host 预填。

## 结果（msprof op）
| shape | base µs | opt µs | 加速 | base SOL% | opt SOL% |
|---|---:|---:|---:|---:|---:|
| whisper-conv1 | 5457 | **53.2** | **~100×** | 0.115 | **11.68** |
| encodec-deep | 1848 | **22.9** | **~80×** | 0.072 | **5.70** |

正确性：`pytest tests/ops/test_conv1d.py` **28/28 passed**（宿主布局）。

## 时间/token
- 本项与本批其它工作同会话完成，未单独计时；方法与数据见 `perf_opt/opt_log.md`。

## 产物
- `conv1d_kernel/perf_opt/_conv1d_unit_stride_kernel.py`（去 prelude）
- `conv1d.py`（wrapper：unit_stride 切 perf_opt + host 填 `w_flat`/`x_pad`）
- 报告：`reports/tileops/20261016_142945_037507_Conv1dFwdOp/`（passed）

## 待续（未达 MixCV ≥40%）
- 剩余 5.7%–11.7% 属 **Cube gemm tiling/overlap** 空间；且 general/pointwise 等 kernel 未改。
