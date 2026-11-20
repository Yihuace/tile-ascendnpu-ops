## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 只把 wrapper 的 `from ._f_p8_quant_kernels import ...` 改写成 baseline/perf_opt 选择块，不同步适配 wrapper 对 kernel 的**调用签名**。当 Stage 3 kernel 采用显式输出参数（`(X, Scale, OutU8)`，无 `out_idx`，见 CG-2026-0019）时，改写后的 wrapper 首次调用即 `ValueError: Expected 3 inputs, got 1`，必须人工补胶水层。 | `examples/TileOPs/tileops/kernels/quantization/fp8_quant/fp8_quant.py`（集成前）；复现见下方 P 类价值点 repro；`examples/TileOPs/tileops/kernels/quantization/fp8_quant/fp8_quant_kernel/integration_log.md#调试历史` | 在 add-npu-op skill 的集成适配要点/已知修复目录中补一条：kernel 若显式传输出张量，wrapper 需 host 侧分配输出载体（并 `.view` 还原 dtype）；或在 integrate_kernel.py 冒烟后追加一次 forward 签名核对警告 | R |
| Integrate-flow | 集成脚本的 import 冒烟只做 `hasattr` 检查（`smoke-ok`），不触发一次真实 forward，因此签名不匹配类缺陷在冒烟阶段不可见，只能等到 report 首例才暴露。 | integrate_kernel.py L398-422 (`smoke_test` 仅 import + hasattr)；本次冒烟 pass 后 direct forward 才报 `Expected 3 inputs, got 1` | 集成脚本冒烟可考虑对最小 shape 追加一次 `kernel.forward` 形状/参数核对（保持 device 可用时可选），或至少在文档中声明「import 冒烟不覆盖调用签名」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir 显式输出签名 kernel 的 TileOPs wrapper 胶水适配：host 分配输出载体 + reshape + `.view` 还原 dtype | P | `examples/TileOPs/tileops/kernels/quantization/fp8_quant/fp8_quant_kernel/integration_log.md#调试历史`；修复后文件 `fp8_quant.py` `_fp8_quant_wrapped_kernel`；备份 `fp8_quant_kernel/history_version/fp8_quant_s5_attempt1.py` | `cd examples/TileOPs && PYTHONPATH=/workspace/tilelang-mlir-ascend python -c "from tileops.kernels.quantization.fp8_quant.fp8_quant import FP8QuantKernel; ..."`（集成前报 `ValueError: Expected 3 inputs, got 1`；修复后 cos≈1.0） | tilelang build=/workspace/tilelang-mlir-ascend/build, git=8b16bccacfd9fa17b09ebd4313bd4501fee03760；Ascend910B3 x1；CANN 9.0.0 | pattern-library |
| 集成后单算子 report 全绿：FP8QuantOp 8/8 正确性 + 3/3 msprof benchmark（~238 us，roofline 1.57–1.62%） | D | `examples/TileOPs/reports/tileops/20261014_171628_509361_FP8QuantOp/run.json` (`summary.correctness_passed=true`, `benchmark_passed=true`, `invalid_profile_count=0`) | `cd examples/TileOPs && PYTHONPATH=/workspace/tilelang-mlir-ascend python -m tileops.reporting.cli run --op FP8QuantOp --prof-mode msprof` | tilelang build=/workspace/tilelang-mlir-ascend/build, git=8b16bccacfd9fa17b09ebd4313bd4501fee03760；Ascend910B3 x1；CANN 9.0.0 | pattern-library / bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- 当 Stage 3 kernel 采用**显式输出张量签名**（无 `out_idx`；uint8/fp8 载体场景常见）时，TileOPs wrapper 的 custom_op 实现体必须在 host 侧分配输出张量、以全部参数调用 kernel，并用 `reshape + .view(dtype)` 还原对外 dtype，不能沿用「单输入 + 返回值」旧模板。
- `integrate_kernel.py` 的 import 冒烟只验证可导入/可属性访问，**不覆盖调用签名**；集成后应至少对最小 shape 手动跑一次 wrapper forward，再进入耗时 report。
