# RETROSPECTIVE -- bitwise_not (BitwiseNotFwdOp)

## Stage 5 (integrate) Retrospective — attempt 2 — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 0 脚手架为多函数算子生成的 test 子类漏掉 workload `__init__` 适配器，导致 fixture 构造即崩（`TypeError: 'int' object is not iterable`）；Stage 5 被禁止修改 `tests/`，只能把失败交回 conductor 重调度一次 | `tests/ops/test_bitwise_not.py#L47-L51`（attempt 1 缺失 `BitwiseNotTest.__init__`）vs `tileops/workloads/elementwise.py#L58`（`tuple(shape)`）；同仓先例 `tests/ops/test_hardswish.py::HardswishTest`、`tests/ops/test_clamp_scalar.py::ClampScalarTest` 均已带 `__init__` 适配器 | 在 Stage 0 scaffolding 阶段对「workload `__init__(shape, dtype)` + fixture 以 `n_total:int` 调用」的算子统一注入 `def __init__(self, n_total, dtype): super().__init__((n_total,), dtype)` 适配器（可并入 add-npu-op 脚手架生成器或新增 Tier 1 结构校验断言） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| NPU 端 bool/uint8 逐元素算子统一走 int8 字节存储 + 元数据 view 往返，bit-exact 且绕过 i1 不可 GM load/store 与 uint8 无运行时输出 dtype 映射两个限制 | P | `tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py#L15-L23`（byte storage 说明）、`bitwise_not.py#L157-L184`（forward 的 `.view(torch.int8)` / `.view(torch.bool)` / `.view(torch.uint8)` 往返） | `python -m tileops.reporting.cli run --op BitwiseNotFwdOp --prof-mode msprof --kernel-name main`（从 `examples/TileOPs/` 运行）：正确性 11/11、max_abs_err=0.0，bool/uint8/int8/int16/int32/int64 全部 exact | tilelang build root `/home/developer/tilelang-mlir-ascend/build`；Ascend910B3 x1；CANN 8.5.2；driver 25.5.0；pytorch_npu 2.7.1.post4 | pattern-library |
| TileLang NPU 运行时输出 dtype 映射缺 `uint8` 条目，声明 uint8 输出会被静默分配为 float32 并损坏结果 | D | `bitwise_not_fwd_kernel.py#L17-L22`；wrapper 端 uint8 后置 `.view(torch.uint8)` 才使 `assert_close` 恢复（attempt 1 未加时 uint8 数值+dtype 双不符） | 见上（同一 report 的 `test_bitwise_not_op[...uint8]` 用例） | 同上 | bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- NPU 上逐元素算子凡输入含 `bool` 或 `uint8`：`bool`（i1）不能 GM load/store，`uint8` 在 TileLang NPU 运行时输出 dtype 映射中无条目（会被静默分配成 float32），两者都应通过 `int8` 字节张量读写；host 侧用 `.view(torch.int8)` 进入、`.view(torch.bool)`/`.view(torch.uint8)` 返回，`.view` 是纯元数据重解释、字节不变，结果 bit-exact。
- 逐元素 bool 的「取反」若 `T.vnot` 不可用于该存储形态，可在字节域用 `x ^ 1`（`T.vxor`）实现；常量操作数必须是 shaped tile（`T.fill` 广播），标量 src1 会 lower 成 `i8` 操作数并被 `hivm.hir.vxor` 拒绝。
- Stage 0 脚手架生成的测试子类若继承「`__init__(shape: tuple, dtype)`」的 workload，而被 fixture 以 `n_total:int` 调用，必须自带 `__init__(n_total, dtype) -> super().__init__((n_total,), dtype)` 适配器；缺失时 Stage 5 无法在禁止修改的 `tests/` 内修复，只能整轮重调度。
