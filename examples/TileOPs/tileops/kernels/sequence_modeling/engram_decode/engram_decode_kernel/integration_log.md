# Integration Log -- EngramDecodeOp

- meta: examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/.migration_meta.json
- op_slug / family: engram_decode / sequence_modeling
- integrated: `_engram_decode_kernel` -> `tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py`（源自 `examples/engram_decode/_engram_decode_kernel/_engram_decode_kernel.py`）
- design_docs: `_engram_decode_kernel` -> `tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel_DESIGN.md`
- wrapper: `tileops/kernels/sequence_modeling/engram_decode/engram_decode.py`（import 改写为 baseline/perf_opt 选择块；`ascend_mode = "Developer"` 保持不变）
- ascend_mode: Developer（从 Stage 3 产物字面量 `ASCEND_MODE = "Developer"` 读取；与 wrapper 既有值一致，未修改）
- attempts: 1（1 次 wrapper 胶水修复，非失败重试）
- 集成脚本摘要（`[json]`）: `wrapper_rewritten=true`, `smoke=pass`, `reexported=["ALIGNMENT","_align_up","_engram_decode_kernel"]`, `ascend_mode=Developer`

## 集成核对（Stage 5 pre-check）

- **调用约定核对（关键）**：Stage 3 工厂 `_engram_decode_kernel(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype, BK=128)` 直接以 `@tilelang.jit(out_idx=[8,9], target="npuir")` 装饰，调用后**返回可直接以张量调用的 `tilelang.jit.jit_npu.JitKernel_NPU`**（实测 `type(_engram_decode_kernel(1,256,256,9,4,3,1e-6,'float16')) == JitKernel_NPU`），即约定为 `factory(params)(tensors)`。
  - Stage 0 脚手架 wrapper 残留 GPU 约定 `factory(params)()(tensors)`（多一层空 `()`，对应已移除的 `threads` 内层闭包）→ 集成后 import smoke **通过但运行即 `TypeError`**。本次在 wrapper 胶水层删除多余 `()`，改为 `factory(params)(tensors)`（属允许修改的 glue 层）。
- 多输入契约核对：`EngramDecodeOp` 为 8 tensor 输入算子；bench 使用本地 `_engram_decode_params()` 读 `load_workloads("EngramDecodeOp")`（Stage 0 已按 skill 多输入指引生成），未使用 `workloads_to_params`。
- 对齐/padding 契约核对：kernel 以 `d_padded = align_up(d, 256)` 建 shape；全部 workload 的 `d ∈ {256, 512}` 本就 256 对齐，wrapper `pad` 分支不触发。`conv_state` 在 workload 中为满长 `(B, max_conv_len, d)`，`state_pad=0`。wrapper 的 pad/trim 逻辑与 op 层契约一致。
- 输出约定核对：kernel `out_idx=[8, 9]`，与 wrapper fake 注册返回 `(y_t, new_conv_state)` 两输出、op/测试的两输出契约一致。
- 已知能力缺口 CG-2026-0022（设计的 R1 Cube `T.gemm` GEMV 在现工具链不可实现，回退为等价 fp32 向量 GEMV）：集成以 conductor 产物实际实现为准，未做改动；性能现象见下。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261015_063432_610459_EngramDecodeOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_063432_610459_EngramDecodeOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261015_063432_610459_EngramDecodeOp/report.html`
- run_id: `20261015_063432_610459_EngramDecodeOp`
- report_status: passed（`run.json.status=passed`；`warnings=[]`）
- correctness: pass（5/5 cases；`correctness_passed=true`、`correctness_failed=0`、`errors=0`、`skipped=0`；用例 = 4 个 manifest/fixture 参数化（含 2 smoke + 2 full）+ 1 个 `test_engram_decode_multi_step`；avg max abs err = 8.77e-03）
- benchmark: pass（`benchmark_passed=true`；4/4 workloads；`msprof_case_count=4`、`invalid_profile_count=0`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`；每个 workload `captured_op_count=1`、`op_name='main'`）

### benchmark 数值（来自 run.json / benchmark.md；latency 单位 us）

| workload | dtype | batch | d_mem | d | max_conv_len | dilation | latency (us) |
|---|---|---|---|---|---|---|---|
| decode-smoke-b1-dmem256-d256-fp16 | float16 | 1 | 256 | 256 | 9 | 3 | 2996.1599 |
| decode-b1-dmem512-d256-fp16 | float16 | 1 | 512 | 256 | 12 | 3 | 5988.0396 |
| decode-b4-dmem1024-d512-fp16 | float16 | 4 | 1024 | 512 | 20 | 5 | 23667.8613 |
| decode-b8-dmem512-d256-bf16 | bfloat16 | 8 | 512 | 256 | 18 | 3 | 5987.2397 |

> 说明（仅记录，不修复）：4 档 latency 均远高于该算子的 roofline 期望（report 的 `Ratio` ≈ 0.01%，相对单 die HBM 理论 1.64 TB/s）；同 batch 不同长的 b1-L12（5988us）与 b8-L18（5987us）几乎相等，b1-L9（2996us）≈ 其 1/2，呈固定量级台阶，疑似 host 侧/调度或固定开销主导，而非数据规模。属性能现象记录，未触发修复（Stage 5 性能只报告）。CG-2026-0022 的 R1 Cube GEMV 未落地，GEMV 为 fp32 向量 FMA 链，也是该档位的固有落差来源之一。

## 调试历史

- attempt 1（wrapper 胶水，非失败重试）：
  - 症状：集成后 import smoke 通过；直接以 wrapper/op 运行会因调用约定不符报错（工厂返回 `JitKernel_NPU`，wrapper 多调一层空 `()`）。
  - 分类：wrapper 胶水错误（GPU↔NPU 工厂调用约定差异，等价于 argmax 集成 VP-2026-0134）。
  - 定位：实测 `type(_engram_decode_kernel(1,256,256,9,4,3,1e-6,'float16')) == JitKernel_NPU`，且 Stage 3 自测 harness `_run_op` 使用 `kernel(tensors)` 一级调用，确证约定为 `factory(params)(tensors)`。
  - 修复：`engram_decode.py` 中 `_engram_decode_kernel(...)()(tensors)` -> `_engram_decode_kernel(...)(tensors)`（仅胶水层）。
  - 备份：`engram_decode_kernel/history_version/engram_decode_wrapper_s5_attempt1.py`。
  - 结果：`pytest -m smoke` 3 passed；完整 report `status=passed`（correctness 5/5、benchmark 4/4）。

## issues

- `integrate_kernel.py` 生成的选择块中，`--` 分隔注释保留了脚手架旧注释（wrapper L66-L70 的 GPU 提取说明）紧邻新块，形成两段 `# ----` 头；功能无害（import 生效、smoke 通过），记为 Integrate-flow 流程缺陷（见 `examples/engram_decode/RETROSPECTIVE.md`）。
- 选择块的 perf_opt 注释行为「每个 re-export 名一行」，因此辅助符号 `ALIGNMENT`/`_align_up` 也各生成一行 `# from .engram_decode_kernel.perf_opt._engram_decode_kernel import ...`（`ALIGNMENT`、`_align_up`、`_engram_decode_kernel` 共 3 行），略冗余；不影响运行，记录不修复。
- 本次 harness 迁移 Stage 4 跳过，无 perf_opt 产物；perf_opt import 行仅为占位注释。
