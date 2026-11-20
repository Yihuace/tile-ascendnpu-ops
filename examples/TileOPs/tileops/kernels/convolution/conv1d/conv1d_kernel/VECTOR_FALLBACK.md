# Conv1d Vector-only 回退说明（CANN 9.0）

## 为什么需要

CANN 9.0 的 `bishengir-compile` / `hivmc 0.2.0` 在 Conv1d 的 Cube implicit-GEMM 路径上存在两层问题：

1. `HFusionToHIVM` 对 buffer-form `MmadL1Op` 取 `getResult(0)` 越界，触发 SIGSEGV；
2. 即使加上 guard，部分 shape 生成的 HIVM IR 仍会在 `hivmc` 阶段失败：
   ```text
   Failed to compile BiShengLIR to binary
   ```

在不升级 CANN 的前提下，继续修编译器不可行，因此提供一条**纯 Vector** 执行路径。

## 实现

- 复用 `_conv1d_direct_kernel` 的 tap 循环、xpad/xdec/polyphase 逻辑；
- 去掉 `c_in == c_out` 限制，增加 `groups / c_in_g`：
  - 每个 output channel 按其 group 取输入通道；
  - 对每个输入通道做 kernel-window 的向量乘加，累加到 `acc`；
- 不使用 `T.gemm`、`mmadL1`、Cube/HFusion；
- 不使用 `T.sync_block_set` / `sync_block_wait`。

## 调度

`Conv1dFwdOp._select_kernel_key` 仍返回原有的 `unit-stride / pointwise / generic / group / depthwise` key，
以保持 dispatch 测试和接口语义不变；`_get_kernel` 在 CANN 9.0 回退模式下统一构造
`Conv1dVectorKernel` 执行。

## 验证

- `pytest tests/ops/test_conv1d.py`：**27 passed, 1 xpassed**（28 个用例）
- `pytest benchmarks/ops/bench_conv1d.py`：**18 passed**
- 端到端 report：`20261026_180218_350993_Conv1dFwdOp`，status=passed，28/28，ratio 1.52–2.92%

## 代价

- 大 channel / 大 kernel 的 shape 会比 Cube 版本慢；
- 性能以本次 report 的 Task Duration 为准，PR 中会如实给出 ratio。

## L-tile 调优（2026-10-28）

纯 Vector direct kernel 的唯一 tile 旋钮是 `block_n`。固定 128 时单 task 的固定开销
（weight/x copy、循环）占比过高；改为按 shape 自适应：

- `kernel_l <= 7`：最大 2048；
- `kernel_l >= 8`：最大 1024（k=10 时 2048 会超 UB）；
- 小 `out_l`：保持 128。

实测（`Conv1dFwdOp`，fp16，Ascend910B3，同进程 median）：

| shape | block_n=128 | 调优后 |
|---|---:|---:|
| whisper-conv1-bias | 59.74 ms | **7.12 ms**（block_n=2048） |
| encodec-deep-bias | 15.45 ms | **4.81 ms**（block_n=1024） |
| encodec-init-bias | 1.12 ms | **0.67 ms**（block_n=2048） |

正确性：`tests/ops/test_conv1d.py` **28/28 passed**。

## Cube 通路在 CANN 9.0.0 上的恢复（2026-10-28）

此前结论「CANN 9.0.0 下 Cube 不可编译」需要修正：CG-2026-0003 是
`convert-hfusion-to-hivm` 里的**非确定性 UB**（对 0 result 的 `hivm.hir.mmadL1` 取
`getResult(0)`），同一份 IR 重复编译约 **21%** 成功（3/14），关 ASLR 后 6/6 必崩。
前端已加**有界重试**（`TILELANG_NPU_COMPILE_RETRIES`，默认 20），Cube 因此可用。

`Conv1dUnitStrideKernel`（fp16，has_bias，Expert MIX，空 cache）实测：

| shape | Cube（重试次数） | Vector 回退 | 结论 |
|---|---:|---:|---|
| whisper (1,80,3000→1280,k=3) | **0.899 ms**（0） | 6.20 ms | Cube 快 ≈6.9× |
| encodec-deep (1,128,600→256,k=10) | **0.993 ms**（1–3） | 4.14 ms | Cube 快 ≈4.2× |
| small (1,64,256→64,k=3) | 1.73 ms（1） | — | 小 shape 不占优 |
| encodec-init (1,1,24000→32,k=7) | 0.814 ms（0） | **0.088 ms** | 瘦 channel 仍走 Vector |

四个 shape 均与 torch 参考对拍通过；`pytest tests/ops/test_conv1d.py` **29 passed**。
因此路由保持「按 shape 选路」：`c_in` 大且 `out_l` 长 → Cube；`c_in` 极小的瘦形状 → Vector。

## Cube 路由开关（2026-10-28）

`Conv1dFwdOp` 默认仍走纯 Vector（`_CUBE_KEYS` 里的三个 key 被重定向到
`Conv1dVectorKernel`），原因只有一个：Cube 依赖前端 `bishengir-compile` 的重试兜底，
而该补丁尚未进入上游（见 `tilelang/jit/jit_npu.py` 的 `TILELANG_NPU_COMPILE_RETRIES`）。

设 `TILEOPS_CONV1D_CUBE=1` 可切到 Cube：

- `conv1d` / `conv1d_unit_stride` / `conv1d_pointwise` → 各自的 Cube kernel；
- `group_conv1d` / `depthwise_conv1d` → 始终用自己的（Vector-direct）kernel，不再被误重定向；
- 若某个 shape 的 Cube 编译抛 `NPU IR opt failed`（CG-2026-0003 的确定性崩溃形状），
  自动回退到 Vector 并把该 shape 的决策记入 kernel cache。

实测（fp16，has_bias，空 cache）：

| 场景 | 结果 |
| --- | --- |
| `TILEOPS_CONV1D_CUBE=1` + `pytest tests/ops/test_conv1d.py` | **29 passed**（4m22s，含回退） |
| 默认（纯 Vector） | 29 passed |
| whisper Cube | 0.899 ms（Vector 6.20 ms） |
| encodec-deep Cube | 0.993 ms（Vector 4.14 ms） |
| encodec-init Cube | 0.814 ms（Vector **0.088 ms**，故瘦 channel 仍应留 Vector） |
