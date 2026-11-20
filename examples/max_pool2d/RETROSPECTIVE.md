# RETROSPECTIVE — max_pool2d (migration-harness)

## Stage 5 (integrate) Retrospective — attempt 2 — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 5 的「已知修复目录」/失败分类表未覆盖「NPU reference 后端不支持某配置」这一独立子类（本次为 `aclnnMaxPool2dWithMask` 不支持 `dilation!=1`）。attempt 1 据此判为「测试/用例自身问题」→ `[INTEGRATE_FAIL]`，但该子类的正确处理是「在受控闭环内修复脚手架测试用例的 reference 计算路径」，而非无条件回交。 | `examples/TileOPs/reports/tileops/20261014_071437_810771_MaxPool2dFwdOp/run.json`（attempt 1，`correctness_failed=1`）；`tileops/kernels/pool/max_pool2d/max_pool2d_kernel/integration_log.md#调试历史` attempt 1 行 | 在 tilelang-op-integrator 的失败分类表中新增子类「NPU reference 后端能力缺口」，处理列写明「若失败根因是 torch/torch_npu 的 NPU golden 不支持该配置（异常在 ref_program 内、op 未执行），可在受控闭环内将该用例 reference 回退 CPU 计算（保留 atol=rtol=0、保留用例覆盖），仅当调度方未授权改 tests 时回交」 | R |
| Integrate-flow | `--op {op_name}` 实际执行整份共享测试/bench 文件，文件级 `summary.correctness_tests` 混合了兄弟算子用例；判定本算子必须读 `run.json.operators[]` per-operator rollup，此口径在流程描述里出现过但易被误用（attempt 1 曾以 `35/36` 文件级计数汇报）。 | `examples/TileOPs/reports/tileops/20261014_072101_015168_MaxPool2dFwdOp/run.json`（`summary.correctness_tests=36` vs `operators[0].{passed:22,failed:0}`） | 在 tilelang-op-integrator 的 report 判定口径处加粗强调「共享文件场景一律以 `operators[]` rollup 为准，文件级 summary 仅作旁证」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| torch_npu 的 `F.max_pool2d` golden 经 `aclnnMaxPool2dWithMask` 只支持 `dilation==1`，`dilation!=1` 抛 `AclNN_Parameter_Error(EZ1001): dilation only support 1` | D | `examples/TileOPs/tests/ops/test_pool.py#MaxPool2dTest.ref_program`；`reports/tileops/20261014_071437_810771_MaxPool2dFwdOp/run.json`；`integration_log.md#调试历史` | `cd examples/TileOPs && python -m pytest "tests/ops/test_pool.py::test_max_pool2d[full-default-stride-dilation-fp16]" -q`（修复前 FAILED，异常来自 ref_program，op 未执行） | tilelang commit 0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c；Ascend910B3 x1；CANN 9.0.0；torch 2.9.0 / torch_npu 2.9.0.post8 | pattern-library / bottleneck-patterns |
| 「纯选择类算子（max/min/argmax 等无累加）在 NPU reference 后端缺某配置时，用 CPU golden 回退」可保证逐位一致（atol=rtol=0 不变），是既能恢复用例又不动 kernel 的通用手法 | P | `examples/TileOPs/tests/ops/test_pool.py#MaxPool2dTest.ref_program`（`F.max_pool2d(input.cpu(), **kwargs).to(input.device)`）；`reports/tileops/20261014_072101_015168_MaxPool2dFwdOp/run.json`（`operators[0].max_abs_err=0.0`，22/22 通过） | `cd examples/TileOPs && python -m pytest "tests/ops/test_pool.py::test_max_pool2d[full-default-stride-dilation-fp16]" -q`（修复后 1 passed） | tilelang commit 0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c；Ascend910B3 x1；CANN 9.0.0 | pattern-library |
| Stage 5 集成-调试闭环的合法编辑范围可包含「本任务 Stage 0 新追加的脚手架测试段落」（需调度方显式授权），但仅限 reference 计算路径，不得改 op/kernel/判据/容差 | C | `examples/TileOPs/tileops/kernels/pool/max_pool2d/max_pool2d_kernel/history_version/test_pool.py.pre_attempt2.bak`（md5 `4364b26d...`→`f6a1dcbb...`）；`integration_log.md#Changes made this attempt` | `diff examples/TileOPs/tileops/kernels/pool/max_pool2d/max_pool2d_kernel/history_version/test_pool.py.pre_attempt2.bak examples/TileOPs/tests/ops/test_pool.py` | tilelang commit 0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c；CANN 9.0.0 | 待定 |

### Transferable Lessons（可迁移教训）

- NPU 上跑 reference/golden 时，若异常出现在 `ref_program` 内部且 op 尚未执行（堆栈落在 `torch.nn.functional` 且报 `AclNN_*_Error`），先判定为 **NPU 后端能力缺口**而非 kernel 缺陷：用 CPU golden 复核该配置与迁移输出的数值一致性，再决定修复路径。
- 纯选择类算子（max/min/argmax/排序，无累加）的参考实现可安全地退到 CPU 计算再搬回设备——结果逐位一致（`atol=rtol=0` 可保持），不引入浮点误差；含累加的算子（sum/avg/conv/gemm）不可照搬此回退。
- 修复脚手架测试时只改 reference 计算路径，保留原用例（含 `dilation>1`）与其严格判据；禁止 skip/xfail/删除用例或放宽容差换绿。
- `--op {name}` 在共享测试文件下会连带执行兄弟算子用例；本算子结论一律以 `run.json.operators[]` per-operator rollup 为准，不要数文件级 `summary.correctness_tests`。
