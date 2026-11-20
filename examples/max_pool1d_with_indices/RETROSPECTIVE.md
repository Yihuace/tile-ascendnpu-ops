# RETROSPECTIVE — max_pool1d_with_indices (migration-harness)

> Op-level retrospective log (canonical schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`).
> Append-only; one section per Stage execution. Stage 5 集成复盘为 op 级单份（见 schema §1）。

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 失败分类表「NPU reference 后端能力缺口」子类只以「异常 + `AclNN_*_Error`」为识别信号（并要求「op 尚未执行、堆栈落在 `torch.nn.functional`」）。本次暴露第二形态：后端**不抛异常、照常返回**，但 `F.max_pool1d(..., return_indices=True)` 的 indices 是 dtype/shape 全错的垃圾 buffer（`int8 (N,C,k,L_pad)`），错误直到测试 harness 的 tensor 比较处才以 `RuntimeError: size mismatch` 浮出。仅按「异常信号」判类会误把该形态归到 shape/runtime 类而漏掉 CPU-golden 修复。 | `examples/TileOPs/reports/tileops/20261014_122210_526209_MaxPool1dIndicesFwdOp/run.json`（attempt 1，`correctness_passed=false`，7 用例失败于 `test_base.py:record_check_result`）；`integration_log.md#调试历史`；设备直探输出 `torch.int8 (1,8,3,64)` | 在 tilelang-op-integrator 失败分类表的「NPU reference 后端能力缺口」行，把识别信号扩为「异常 `AclNN_*_Error` **或** 后端返回的 tensor dtype/shape 与算子契约不符（静默损坏）」，并注明「stack 落在 ref_program/比较 harness、op 未被执行判据」均可判为该子类。 | R |
| Integrate-flow | 本子类预置的 CPU-golden 回退（Stage 0 脚手架里为 `dilation != 1` 预埋）覆盖面不足：NPU `return_indices=True` 的 indices 损坏对**所有** dilation 成立，而非仅受限参数。脚手架按「受限参数」预埋回退，遇到「全参数区间静默损坏」时仍会失败。 | `examples/TileOPs/tests/ops/test_pool.py#MaxPool1dIndicesTest.ref_program`（原 `needs_cpu_golden = dilation != 1`）；`integration_log.md#调试历史` | 在 add-npu-op 的 reference 模板里，把「纯选择类算子的 NPU golden 不可用 → 直接走 CPU golden」作为默认（而非仅在受限参数分支），并加注释说明需实测 `return_indices`/多输出后端支持。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| torch_npu 的 `F.max_pool1d(..., return_indices=True)` 在 NPU 上 indices 输出静默损坏：`dilation==1` 时返回 `torch.int8 (N,C,kernel_size,L_pad)` 内部 mask buffer（如 `(1,8,3,64)`，含 0/1 噪声）而非 `int64 (N,C,L_out)`；`dilation!=1` 另抛 `AclNN_Parameter_Error(EZ1001)`。values 输出正常。 | D | `examples/TileOPs/reports/tileops/20261014_122210_526209_MaxPool1dIndicesFwdOp/run.json`（7 用例失败）；`integration_log.md#调试历史` 直探输出 | `python -c "import torch,torch_npu,torch.nn.functional as F; x=torch.randn(1,8,32,dtype=torch.float16,device='npu:0'); v,i=F.max_pool1d(x,3,2,1,return_indices=True); print(i.shape,i.dtype)"` → `(1,8,3,64) torch.int8` | tilelang dev build @ git 892e2c18d80c37795727236b5ebbe145626a8313；Ascend910B3 x1；CANN 9.0.0；torch 2.9.0+cpu / torch_npu 2.9.0.post8 | pattern-library / bottleneck-patterns |
| 纯选择类算子（max/min/argmax/排序，无累加）的「NPU reference 后端缺 `return_indices`/多输出支持」可用 CPU golden 回退修复且逐位一致：`F.max_pool1d(input.cpu(), **kwargs)` 后 `.to(input.device)`，值与 int64 indices 与迁移算子 `val_max_diff=0.000e+00`、`torch.equal(idx)=True`，`atol=rtol=0` 全程保持；修复后 per-op 47/47。 | P | `examples/TileOPs/tests/ops/test_pool.py#MaxPool1dIndicesTest.ref_program`（改为无条件 CPU golden）；`examples/TileOPs/reports/tileops/20261014_122557_915676_MaxPool1dIndicesFwdOp/run.json`（`operators[0]={cases:47,passed:47,max_abs_err:0.0}`） | `cd examples/TileOPs && python -m tileops.reporting.cli run --op MaxPool1dIndicesFwdOp --prof-mode msprof` | tilelang dev build @ git 892e2c18d80c37795727236b5ebbe145626a8313；Ascend910B3 x1；CANN 9.0.0；torch 2.9.0+cpu / torch_npu 2.9.0.post8 | pattern-library / verification |
| 双输出（value + int64 indices）max-pool 的 Stage 5 集成案例：Developer 模式 kernel 免改直接集成，per-op 47/47 correctness（`max_abs_err=0.0`）、benchmark 36/36 pass（24 valid msprof，`invalid_profile_count=0`），6 条 bench（sincnet 3192.84us / textcnn 138.52us / ecg 1429.55us；Ratio 3.40–5.70%）。 | D | `examples/TileOPs/reports/tileops/20261014_122557_915676_MaxPool1dIndicesFwdOp/run.json`（status=passed）、`.../benchmark.json`（6 条 MaxPool1dIndicesFwdOp records） | `cd examples/TileOPs && python -m tileops.reporting.cli run --op MaxPool1dIndicesFwdOp --prof-mode msprof` | tilelang dev build @ git 892e2c18d80c37795727236b5ebbe145626a8313；Ascend910B3 x1；CANN 9.0.0 | pattern-library / bottleneck-patterns |
| NaN-window 冒烟测试若用 `torch.testing.assert_close` 比较「选出窗口值的 NaN 切片」而漏传 `equal_nan=True`（默认 False），该断言与后端无关、恒不可满足（NaN≠NaN）；兄弟测试 `test_adaptive_max_pool2d_indices_nan_window` 通过「只断言 isnan + 比 idx、不比 NaN 值」规避。 | C | `examples/TileOPs/tests/ops/test_pool.py#test_max_pool1d_indices_nan_window`（原 `assert_close(val.float()[nan_mask], ref_val[nan_mask], atol=0, rtol=0)`）；`integration_log.md#调试历史` attempt 1 | 修复前 `python -m pytest tests/ops/test_pool.py -k "max_pool1d_indices_nan_window" -q` → `Mismatched elements: 2 / 2`（greatest abs diff nan） | tilelang dev build @ git 892e2c18；Ascend910B3 x1；CANN 9.0.0；torch 2.9.0+cpu | bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- 用 `torch.nn.functional.*(..., return_indices=True)`（或任何多输出 torch API）作 NPU golden 时，**不要只看它跑没跑通**：torch_npu 可能不抛异常却返回 dtype/shape 全错的 indices（本次为 `int8 (N,C,k,L_pad)` 内部 mask buffer）。参考实现落回 CPU 前，先实测 indices 的 `dtype`/`shape` 是否符合算子契约。
- 「NPU reference 后端能力缺口」有两种形态，判类方式不同：①后端抛 `AclNN_*_Error`（堆栈在 `F.*` 调用处）；②后端静默返回损坏 tensor（异常直到测试 harness 的广播/相减处才出现，形似 shape/runtime 失败）。两者都属该子类，处理都是「纯选择类算子 → reference 走 CPU golden」。
- 纯选择类算子（max/min/argmax/排序，无累加）的 CPU golden 与正确 NPU 结果逐位一致，`atol=rtol=0` 与 int64 索引精确比较可全程保留；含累加算子（sum/avg/conv/gemm）不得照搬该回退。
- NaN 语义测试不要用默认的 `torch.testing.assert_close` 去比 NaN 切片（默认 `equal_nan=False`，恒失败）：要么传 `equal_nan=True`（等价于 repo 内 `allclose_compare` 的口径），要么像 adaptive 兄弟用例那样只断言 `isnan` 并单独精确比 indices。
- 修改被授权范围内的测试 reference 路径时：先 `cp` 备份到集成包 `history_version/`（本包 `test_pool_s5_attempt1.bak`），只动 reference 计算路径，保留全部用例、`atol=rtol=0` 与索引判据；`integrate_kernel.py` 只跑一次（其聚合 `__init__.py` 生成在 wrapper 已改写后不可幂等复现），集成后手工核对聚合 `__init__.py` 与相对导入。
