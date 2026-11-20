# Integration Log -- EngramGateConvBwdOp

- meta: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/.migration_meta.json`
- op_name / op_slug / family: `EngramGateConvBwdOp` / `engram_gate_conv_bwd` / `sequence_modeling`
- integrated: `_engram_gate_conv_bwd_kernel` -> `tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py`（源自 `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py`）
- design_docs: `_engram_gate_conv_bwd_kernel` -> `tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel_DESIGN.md`（无 missing）
- wrapper: `tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd.py`（import 改写为 baseline/perf_opt 选择块；`ascend_mode = "Developer"` 保持）
- ascend_mode: Developer（从 Stage 3 产物字面量 `ASCEND_MODE = "Developer"` 读取；与 wrapper 既有值一致，`wrapper_mode_updated=false`）
- attempts: 1（**无失败重试**；1 次 wrapper 胶水对齐 = 跨阶段移交项 #1，随后集成脚本一次通过 + report 一次通过）
- 集成脚本摘要（`[json]`）: `wrapper_rewritten=true`, `wrapper_mode_updated=false`, `smoke=pass`, `reexported=["ALIGNMENT","CONV_KERNEL_SIZE","_align_up","_engram_gate_conv_bwd_kernel"]`, `ascend_mode=Developer`, `design_docs` 全部命中

## 集成核对（Stage 5 pre-check）

### 跨阶段移交项 #1：`register_fake` 8→6 对齐（已按 DESIGN §3.4 / R-13 处理）

- **现象**：Stage 0 脚手架 wrapper 的 `@register_fake` 返回 **8** 个占位张量（6 契约输出 + `dvhat_buf`/`dvhat_out` 两个 workspace）；而 Stage 3 闭包只返回 **6** 个契约输出（顺序 `dH, dk, dv, drms_w_h, drms_w_v, dconv_w`，与源 `out_idx=[12..17]` 一致；workspace 由闭包内 `torch.empty` 内部分配、不返回）。
- **修复**：删除 fake 中末两行 `dvhat_buf`/`dvhat_out`，并在其上补契约注释（DESIGN §3.4 / R-13）。属允许的 wrapper 胶水层修改。
- **验证（FakeTensorMode 实测）**：`fake output count: 6`，shapes/dtypes = `[(1,32,256)/fp16, (1,32,256)/fp16, (1,32,256)/fp16, (256,)/fp32, (256,)/fp32, (4,256)/fp32]` —— 与 op/测试的 6 输出契约逐项一致（见「调试历史」attempt 1）。
- wrapper `forward` 侧 `results[:6]` 与 crop 逻辑（`d_padded` 对齐）无需改动，与 6 输出契约一致。

### 跨阶段移交项 #2：wrapper 调用约定核查（**无需修复**）

- Stage 3 工厂 `_engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype)` 以 `@functools.lru_cache` 装饰，**返回一个 Python 闭包** `_run(dY,H,k,v,rms_w_h,rms_w_v,conv_w,vhat,alpha,rrms_h,rrms_k,rrms_v)`（**非** `JitKernel_NPU`——DESIGN §3.4 已显式指出 `TRAP-outidx-call-convention` 的零成本判定在此不适用）。
- server 实测：`factory type = functools._lru_cache_wrapper`；`type(f(1,32,256,1e-6,'float16')) == function`；`callable=True`；`inspect.signature = (dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v)`。
- wrapper 调用点（集成脚本不触碰调用点，为 Stage 0 原状）为**单级** `_engram_gate_conv_bwd_kernel(M,seq_len,d,eps,dtype_str)(dY,H,...,rrms_v)`，与该闭包签名**逐参对齐**，即 NPU 单级约定 `factory(params)(tensors)`。
- **结论**：本算子 Stage 0 脚手架**未**残留 GPU 双级约定 `factory(params)()(tensors)`（decode 曾残留、fwd 未残留），本次**不需要**该胶水修复。这与同族 fwd 的结论一致（同一 Stage 0 生成器，本产物已是修复后形态）。

### 其它契约核对

- 多输入多输出契约：`EngramGateConvBwdOp` 为 **12 tensor 输入 / 6 输出**算子；bench 使用本地参数化 helper `_engram_gate_conv_bwd_params()` 读 `load_workloads("EngramGateConvBwdOp")`（Stage 0 已按 skill 多输入指引生成，未用 `workloads_to_params`）。
- 对齐/padding 契约：kernel 以 `d_padded = _align_up(d, ALIGNMENT=256)` 建 shape；全部 workload 的 `d ∈ {256, 512}` 均 256 对齐，`bwd-smoke` 的 `d=256` 亦对齐，wrapper 的 `pad` 分支不触发（`d_padded - d == 0`）。wrapper pad/trim 逻辑与 op 层契约一致。
- 输出约定：kernel 闭包返回 6，fake 现返回 6（同序），op/测试的返回契约为 6，三处一致（DESIGN §3.4 要求）。
- `source.test` / `source.bench` 核对：manifest `source.test=tests/ops/test_engram_gate_conv_bwd.py`、`source.bench=benchmarks/ops/bench_engram_gate_conv_bwd.py`，与 `.migration_meta.json` 的 `test_path` / `bench_path` 一致。
- Stage 3 已知非阻塞告警（Boundary `seq_len=1` 退化 shape 触发 npuir canonicalization size mismatch）在 manifest/fixture 契约外（最小 `seq_len=16`），未进入本次 report 用例集。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261016_002822_275836_EngramGateConvBwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261016_002822_275836_EngramGateConvBwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261016_002822_275836_EngramGateConvBwdOp/report.html`
- benchmark_json / benchmark_md: `examples/TileOPs/reports/tileops/20261016_002822_275836_EngramGateConvBwdOp/benchmark.json` / `benchmark.md`
- run_id: `20261016_002822_275836_EngramGateConvBwdOp`
- report_status: **passed**（`run.json.status=passed`；`warnings=[]`；CLI `status=passed`）
- correctness: **pass（5/5 cases）**；`correctness_passed=true`、`correctness_failed=0`、`errors=0`、`skipped=0`
  - 用例 = 5 个 manifest/fixture 参数化（2 smoke：`[1-32-256-fp16]`、`[1-32-256-bf16]`；3 full：`[2-64-512-fp16]`、`[2-16-256-bf16]`、`[2-512-256-fp16]`）
  - max abs err：2.72e-04 / 2.26e-03 / 6.23e-04 / 2.06e-03 / 1.30e-03；report `Avg Max Abs Error = 1.30e-03`
  - 每例均含**确定性双跑**校验（`torch.equal(run1, run2)` on dH/dk/dv）并全部通过 —— 继承 Stage 3 的确定性两段式归约（R5）在集成路径下保持确定性。
- benchmark: **pass（4/4 workloads）**；`benchmark_passed=true`、`benchmark_failed=0`、`case_count=4`；`msprof_case_count=4`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`、`invalid_profile_count=0`
  - 4 档全部走 msprof 真实 profile（`prof_mode=msprof`），无 fallback；每档 `captured_op_count=1`。
  - **profiler 记录口径**：闭包按 K1→K2→K3 顺序发射三个 jit kernel，msprof 捕获到 `op_name='k1'`（`kernel_name_resolved='k1'`），故 report 表中 Kernel 列均为 `k1`、`sample_count=1`。记录的 latency 为本次 profile 采样值。

### benchmark 数值（来自 run.json / benchmark.json；latency 单位 us）

| workload | dtype | M | seq_len | d | latency (us) | Ratio (%) (=实测÷单die HBM 理论 1.64) | AchievedBW (TB/s) | AI (Ops/Byte) |
|---|---|---|---|---|---|---|---|---|
| bwd-smoke-b1-s16-d256-fp16 | float16 | 1 | 16 | 256 | 50.9200 | 0.0615 | 1.7500 | 17.7049 |
| bwd-b1-s32-d256-fp16 | float16 | 1 | 32 | 256 | 51.0400 | 0.1228 | 1.7500 | 23.2258 |
| bwd-b2-s64-d512-fp16 | float16 | 2 | 64 | 512 | 59.6000 | 0.8561 | 1.7500 | 32.0578 |
| bwd-b1-s128-d256-bf16 | bfloat16 | 1 | 128 | 256 | 225.6400 | 0.5905 | 1.7500 | 171.1079 |

> 说明（仅记录，不修复，Stage 5 性能只报告）：4 档 latency 均远高于该 memory-bound 算子的 roofline 期望（report 的 `Ratio` 0.06%–0.86%，`AchievedBW` 恒为 1.75 TB/s 的理论参考值）。最小两档（s16，50.92us；s32，51.04us）几乎相等，呈固定量级台阶，疑似 host 侧/固定开销（含每次 profile 单次采样 `sample_count=1`）主导；s128-bf16 档（225.64us）随序列长度近似线性增长，b2-s64-d512（59.60us）随 batch×d 增大。属性能现象记录，未触发修复。msprof 只捕获固定 kernel 名 `k1`，多 kernel 闭包的总时延未被逐 kernel 分解，亦为口径细节。

## 调试历史

- attempt 1（wrapper 胶水对齐，**非失败重试**）：
  - 症状：集成脚本 import smoke 通过；但 wrapper `register_fake` 声明 8 输出、Stage 3 闭包只返回 6 输出，fake/real 契约不一致（DESIGN R-13 已预警：集成期 fake/real 不一致）。
  - 分类：wrapper 胶水错误（输出契约数 6 vs 8；DESIGN §3.4 / R-13）。
  - 定位：DESIGN §3.4「output 契约对齐（K7）」+ R-13 明确要求 Stage 3/集成期把 `register_fake` 由 8 对齐到 6；Stage 3 闭包 `_run` 实测返回 6（源码 L594-597）。
  - 修复：`engram_gate_conv_bwd.py` 的 `register_fake` 删除 `dvhat_buf`/`dvhat_out` 两行，返回 6 个契约占位张量（仅胶水层）。
  - 备份：`engram_gate_conv_bwd_kernel/history_version/engram_gate_conv_bwd_wrapper_s5_attempt1.py`（修复前快照）。
  - 结果：FakeTensorMode 实测 fake 输出 6 个（shape/dtype 逐项对齐契约）；`pytest -m smoke` 2 passed；完整 report `status=passed`（correctness 5/5、benchmark 4/4、warnings=[]）。
- 跨阶段移交项 #2（调用约定）核查后**判定无需修复**，故无对应 attempt。

## issues

- **Integrate-flow**：`integrate_kernel.py` 生成的选择块之前保留了 Stage 0 脚手架的旧注释（`# Part A: GPU TileLang kernel factory ...` 后的两行 `# ----` 分隔头），与新块合并后形成两段 `# ----` 头（wrapper L60-L67）。功能无害（import 生效、smoke 与 report 均通过），记为流程缺陷，不修复（继承同族 decode/fwd）。
- **Integrate-flow**：选择块的 perf_opt 注释行为「每个 re-export 名一行」，辅助符号 `ALIGNMENT` / `CONV_KERNEL_SIZE` / `_align_up` 也各生成一行 `# from ...perf_opt... import ...`（共 4 行，其中 3 行为噪声）；不影响运行，记录不修复。
- 本次 harness 迁移 Stage 4 跳过，无 perf_opt 产物；perf_opt import 行仅为占位注释（baseline 默认激活）。
- 任务输入预判「wrapper 残留 `factory(params)()(tensors)`」**与本算子实际不符**：实测 Stage 0 脚手架即为单级 `factory(params)(tensors)`（工厂返回闭包，非 `JitKernel_NPU`）；未做修复。见「集成核对」段。
- **profiler 口径**：多 kernel 闭包（K1/K2/K3）在 msprof 下只记录到固定 kernel 名 `k1`，report 的 latency 为单次采样值，未覆盖全算子总时延；记录不修复。
