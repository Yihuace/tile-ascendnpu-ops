# Integration Log -- EngramGateConvFwdOp

- meta: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_fwd/.migration_meta.json`
- op_name / op_slug / family: `EngramGateConvFwdOp` / `engram_gate_conv_fwd` / `sequence_modeling`
- integrated: `_engram_gate_conv_fwd_kernel` -> `tileops/kernels/sequence_modeling/engram_gate_conv_fwd/engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py`（源自 `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py`）
- design_docs: `_engram_gate_conv_fwd_kernel` -> `tileops/kernels/sequence_modeling/engram_gate_conv_fwd/engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel_DESIGN.md`（无 missing）
- wrapper: `tileops/kernels/sequence_modeling/engram_gate_conv_fwd/engram_gate_conv_fwd.py`（import 改写为 baseline/perf_opt 选择块；`ascend_mode = "Developer"` 保持）
- ascend_mode: Developer（从 Stage 3 产物字面量 `ASCEND_MODE = "Developer"` 读取；与 wrapper 既有值一致，`wrapper_mode_updated=false`）
- attempts: 1（**无任何修复**；集成脚本一次通过 + report 一次通过）
- 集成脚本摘要（`[json]`）: `wrapper_rewritten=true`, `wrapper_mode_updated=false`, `smoke=pass`, `reexported=["ALIGNMENT","_align_up","_engram_gate_conv_fwd_kernel"]`, `ascend_mode=Developer`

## 集成核对（Stage 5 pre-check）

- **调用约定核对（关键，任务指明的最高优先核查项）**：Stage 3 工厂 `_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype)` 直接以 `@tilelang.jit(out_idx=[6,7,8,9,10,11], target="npuir")` 装饰；实测 `type(_engram_gate_conv_fwd_kernel(1,32,256,1e-6,'float16')) == tilelang.jit.jit_npu.JitKernel_NPU` 且 `callable=True`。
  - **结论：本算子 Stage 0 脚手架的调用约定已是 NPU 单级约定 `factory(params)(tensors)`，未残留 `factory(params)()(tensors)`**。即 `EngramDecodeOp` 已修复的 wrapper 调用约定问题在本算子**不需要再次修复**（同一 Stage 0 生成器，本次产物已是修复后形态）。集成脚本仅改写 import 与（未变的）ascend_mode，不触碰调用点，故该调用点为 Stage 0 原状。
  - 佐证：wrapper `_engram_gate_conv_fwd_wrapped` 内为 `_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype_str)(H, k, v, rms_w_h, rms_w_v, conv_w)` —— 一级 `factory(params)` + 一级 `(tensors)`。未做任何修改。
- 多输入契约核对：`EngramGateConvFwdOp` 为 **6 tensor 输入 / 6 输出**（Y/vhat/alpha/rrms_h/rrms_k/rrms_v）算子；bench 使用本地参数化 helper `_engram_gate_conv_fwd_params()` 读 `load_workloads("EngramGateConvFwdOp")`（Stage 0 已按 skill 多输入指引生成），未使用 `workloads_to_params`。
- 对齐/padding 契约核对：kernel 以 `d_padded = _align_up(d, ALIGNMENT=256)` 建 shape；三个 full workload 的 `d ∈ {256,512}` 均 256 对齐，`fwd-smoke` 的 `d=256` 亦对齐，wrapper 的 `pad` 分支不触发（`d_padded - d == 0`）。wrapper 的 pad/trim 逻辑与 op 层契约一致。
- 输出约定核对：kernel `out_idx=[6,7,8,9,10,11]`（Y, vhat_buf, alpha, rrms_h, rrms_k, rrms_v），wrapper fake 注册返回 6 个同序张量，与 op/测试的 6 输出契约一致。
- `source.test` / `source.bench` 核对：manifest `source.test=tests/ops/test_engram_gate_conv_fwd.py`、`source.bench=benchmarks/ops/bench_engram_gate_conv_fwd.py`，与 `.migration_meta.json` 的 `test_path` / `bench_path` 一致。
- Stage 3 已知非阻塞告警（Boundary `seq_len=1` 越界、`T.vbrc` 拒绝空广播维）在 manifest 契约外，未进入本次 report 用例集（smoke/full 共 4 例均含 `seq_len>=16`）。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261015_153050_186944_EngramGateConvFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_153050_186944_EngramGateConvFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261015_153050_186944_EngramGateConvFwdOp/report.html`
- run_id: `20261015_153050_186944_EngramGateConvFwdOp`
- report_status: **passed**（`run.json.status=passed`；`warnings=[]`；CLI `status=passed`）
- correctness: **pass（4/4 cases）**；`correctness_passed=true`、`correctness_failed=0`、`errors=0`、`skipped=0`；`exit_codes.correctness=0`
  - 用例 = 4 个 manifest/fixture 参数化（2 smoke：`[1-32-256-fp16]`、`[1-32-256-bf16]`；2 full：`[2-64-512-fp16]`、`[2-16-256-bf16]`）
  - max abs err：1.87e-03（1-32-256-fp16）/ 2.01e-02（1-32-256-bf16）/ 3.39e-03（2-64-512-fp16）/ 2.01e-02（2-16-256-bf16）；`avg_max_abs_err=1.1365e-02`、`max_abs_err=0.0201`
- benchmark: **pass（4/4 workloads）**；`benchmark_passed=true`、`benchmark_failed=0`；`msprof_case_count=4`、`invalid_profile_count=0`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`；每个 workload `captured_op_count=1`、`op_name='main'`、`profiler_valid=true`
  - 4 档全部走 msprof 真实 profile（`prof_mode=msprof`），无 fallback。

### benchmark 数值（来自 run.json / report.md；latency 单位 us）

| workload | dtype | M | seq_len | d | latency (us) | Ratio (%) (=实测÷单die HBM 理论 1.64) | AchievedBW (TB/s) |
|---|---|---|---|---|---|---|---|
| fwd-smoke-b1-s16-d256-fp16 | float16 | 1 | 16 | 256 | 94.0000 | 0.04 | 1.7500 |
| fwd-b1-s32-d256-fp16 | float16 | 1 | 32 | 256 | 93.1400 | 0.07 | 1.7500 |
| fwd-b2-s64-d512-fp16 | float16 | 2 | 64 | 512 | 134.5800 | 0.47 | 1.7500 |
| fwd-b1-s128-d256-bf16 | bfloat16 | 1 | 128 | 256 | 411.7200 | 0.55 | 1.7500 |

> 说明（仅记录，不修复，Stage 5 性能只报告）：4 档 latency 均远高于该 memory-bound 算子的 roofline 期望（report 的 `Ratio` 0.04%–0.55%，`AchievedBW` 恒为 1.75 TB/s 的理论参考值）；最小档（smoke s16，94us）与 s32 档（93.14us）几乎相等、且与数据规模不成比例，呈固定量级台阶，疑似 host 侧/固定开销（含每次 profile 的单次采样 `sample_count=1`）主导。s128-bf16 档（411.72us）随序列长度近似线性增长，b2-s64-d512（134.58us）随 batch×d 增大。属性能现象记录，未触发修复。

## 调试历史

- 无。集成脚本一次通过（import smoke pass），单算子 report 一次通过（correctness 4/4、benchmark 4/4、warnings=[]），未进入调试闭环，未做任何 kernel/wrapper 修改，故 `history_version/` 为空（无备份需求）。

## issues

- **Integrate-flow（继承自 engram_decode）**：`integrate_kernel.py` 生成的选择块之前保留了 Stage 0 脚手架的旧注释（`# Part A: GPU TileLang kernel factory ...` 后的两行 `# ----` 分隔头），与新块合并后形成两段 `# ----` 头（wrapper L61-L63）。功能无害（import 生效、smoke 与 report 均通过），记为流程缺陷，不修复。
- **Integrate-flow**：选择块的 perf_opt 注释行为「每个 re-export 名一行」，辅助符号 `ALIGNMENT` / `_align_up` 也各生成一行 `# from .engram_gate_conv_fwd_kernel.perf_opt._engram_gate_conv_fwd_kernel import ...`（共 3 行），略冗余；不影响运行，记录不修复。
- 本次 harness 迁移 Stage 4 跳过，无 perf_opt 产物；perf_opt import 行仅为占位注释（baseline 默认激活）。
- 任务输入预判「本算子 wrapper 很可能残留 `factory(params)()(tensors)`」**与本算子实际不符**：实测 Stage 0 脚手架即为 `factory(params)(tensors)`（NPU 单级约定），未做修复。见「集成核对」段。
