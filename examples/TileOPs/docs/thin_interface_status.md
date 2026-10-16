# 薄接口算子 harness 状态说明

任务书 §2 R2 要求 harness 场景算子走 Stage 0→5；以下 4 个算子**未走 harness**（无 `extract/integrate` 产物），现补录状态与理由。它们是"底层复用既有 kernel"或"直接新写 kernel"的**独立 Op 接口**（任务书 §5 要求独立暴露）。

| 算子 | 交付性质 | kernel | 说明 |
|---|---|---|---|
| `ClampMinFwdOp` | 薄接口（复用） | `ClampScalarFwdKernel`（`kernels/elementwise/clamp_scalar`） | `torch.clamp_min` = `torch.clamp(min=·, max=None)`；无新 kernel、无 extract。op：`ops/elementwise/clamp_min.py`；测试 `tests/ops/test_clamp_min.py`。 |
| `Conv1dBiasFwdOp` | 薄接口（复用） | `Conv1dKernel` 等（`kernels/convolution/conv1d`） | bias 为必填的 conv1d 变体；复用 conv1d 的 `has_bias` 路径；无新 kernel。op：`ops/convolution.py::Conv1dBiasFwdOp`；测试 `tests/ops/test_conv1d_bias.py`。 |
| `Conv3dBiasFwdOp` | **新 kernel**（直接实现） | `Conv3dKernel`（`kernels/convolution/conv3d`） | 新写 dense NCDHW implicit-GEMM + MixCV bias；未走 harness（`.migration_meta.json` 已补于 `conv3d/`，`mode=direct-implementation`）。 |
| `GLADecodeOp` | **新 kernel**（直接实现） | `GLADecodeKernel`（`kernels/linear_attention/gla_decode`） | 新写 Vector K-tile decode；未走 harness（`.migration_meta.json` 已补于 `gla_decode/`，`mode=direct-implementation`）。 |

> 后续（若要严格对齐 R2）：可为上述算子补 Stage 1 DESIGN.md / Stage 2 REVIEW.md / Stage 5 三护栏日志；**kernel 与测试/报告均已存在**，补的是流程证据。
