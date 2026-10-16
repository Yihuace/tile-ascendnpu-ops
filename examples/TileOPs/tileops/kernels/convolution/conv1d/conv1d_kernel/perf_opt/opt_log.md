# Conv1dFwdOp(unit_stride) — Stage-4 optimize log

- 算子 `Conv1dFwdOp`（kernel：`_conv1d_unit_stride_kernel`）；口径 `msprof op Task Duration` + Roofline SOL。
- 工具链：tilelang 0.1.2+417dae99 / CANN 8.5.2 / Ascend910B3。

## 诊断
whisper-conv1 `PipeUtilization`：**aic_cube_ratio=0.0015**、其它 pipe ≤0.8% → **Cube 99.85% 空转**（被卡住）。
根因：Cube 前的 **in-kernel Vector prelude**（权重重排 `w_flat` + 输入填充 `x_pad`）**串行**执行，
Cube 阻塞在 `sync_block_wait(4)` 上等 prelude；对大 `C_out`（whisper 1280）代价巨大。

## 优化：prelude 移到 host
- perf_opt kernel **删除 `_prelude`**（及 `sync_block_wait/set(4)`）→ `w_flat`/`x_pad` 由 **host** 预填：
  - `x_pad[:, :C_in, pL:pL+L] = x`；`w_flat[:, :, :C_in] = weight.permute(2,0,1)`。
- 设备 kernel 只剩 Cube 主循环 → Cube 立即开工。

## 结果（msprof）
| shape | base µs | opt µs | 加速 | base% | opt% |
|---|---:|---:|---:|---:|---:|
| whisper-conv1 (C_out 1280) | 5457 | **53.2** | **~100×** | 0.115 | **11.68** |
| encodec-deep (C_out 256) | 1848 | **22.9** | **~80×** | 0.072 | **5.70** |
| wav2vec2-layer1 | 619 | 617 | tie | 0.31 | 0.32 |
| encodec-init | 262 | 262 | tie | 0.44 | 0.44 |

（wav2vec2/encodec-init 走 general/pointwise 等其它 kernel，本次未改。）

## 结论
- **去 in-kernel prelude 是数量级收益**（whisper 100×、encodec-deep 80×）。
- 剩余（ratio 5.7%–11.7%，未达 MixCV ≥40%）：是 **Cube gemm tiling/overlap** 的空间，属下一步。

## 续：general kernel（`_conv1d_kernel`）同样去 prelude
对 general 路径（wav2vec2 等）套用同一手法：perf_opt 删 `_prelude`；wrapper 在 host 预填
`w_flat`(permute) + `x_pad`(pad) + `xdec`(多项式抽取)。

| shape | base µs | opt µs | 加速 | base% | opt% |
|---|---:|---:|---:|---:|---:|
| wav2vec2-layer1 (C_in=1,K=10) | 619 | **218** | **~2.8×** | 0.31 | 0.81 |
| encodec-init | 262 | **175** | ~1.5× | 0.44 | 0.66 |
| whisper-conv1 | 5457 | 53 | ~100× | 0.115 | 11.6 |
| encodec-deep | 1848 | 21 | ~88× | 0.072 | 6.2 |

正确性 `pytest tests/ops/test_conv1d.py` 28/28。注：wav2vec2 `K=C_in·kW=10` 极小 → gemm 天生喂不饱 Cube，
其 ratio 上限低（非 prelude 问题）。
