# Integration Log -- GroupedQueryAttentionSlidingWindowVarlenFwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/.migration_meta.json`
- op_name / op_slug / family: `GroupedQueryAttentionSlidingWindowVarlenFwdOp` / `grouped_query_attention_sliding_window_varlen` / `attention`
- integrated: `_gqa_sw_fwd_varlen_kernel` -> `tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py`（源自 `examples/grouped_query_attention_sliding_window_varlen/_gqa_sw_fwd_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py`）
- design_docs: `_gqa_sw_fwd_varlen_kernel` -> `tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel_DESIGN.md`（无 missing）
- wrapper: `tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen.py`（import 改写为 baseline/perf_opt 选择块；`ascend_mode = "Expert"` 由脚本插入 `GQASlidingWindowVarlenFwdKernel`）
- ascend_mode: **Expert**（从 Stage 3 产物字面量 `ASCEND_MODE = "Expert"` / `os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")` 读取；wrapper 原先无 `ascend_mode`，脚本插入，`wrapper_mode_updated=true`）
- attempts: **1**（无修复；集成脚本一次通过 + report 一次运行，correctness 全过）
- 集成脚本摘要（`[json]`）: `wrapper_rewritten=true`, `wrapper_mode_updated=true`, `smoke=pass`, `reexported=["_gqa_sw_fwd_varlen_kernel"]`, `ascend_mode=Expert`

## 集成范围核对（关键：arch-specialization 收敛，与同族姊妹一致）

- 任务权威输入 `extracted_functions=["_gqa_sw_fwd_varlen_kernel"]`（conductor 只迁移了 architecture-generic 变体）。磁盘上 `.migration_meta.json` 的 `extracted_functions` 字段为 **两个** 名字：`["_gqa_sw_fwd_varlen_kernel", "_gqa_sw_fwd_varlen_wgmma_pipelined_kernel"]`——这是 Stage 0 extraction 脚本的原始输出，其中 `_gqa_sw_fwd_varlen_wgmma_pipelined_kernel` 是 Hopper warp-specialized 变体，按 add-npu-op K5 架构特化折叠**未迁移**（manifest `attention.yaml` L211-L216 与 wrapper 的 K5 scope note 均写明 "NPU collapses to the architecture-generic kernel"）。
- 因此本次集成按 conductor 权威输入只集成 `_gqa_sw_fwd_varlen_kernel`：`integrate_kernel.py --meta <meta> --functions _gqa_sw_fwd_varlen_kernel`（`--functions` 覆盖 meta 的过宽列表）。**未对缺失的 wgmma 变体做任何伪造/占位产物。**
- **wrapper 胶水层收窄（前置，允许范围内）**：Stage 0 脚手架的 wrapper 同时 import 了被收敛掉的 `_gqa_sw_fwd_varlen_wgmma_pipelined_kernel`（与其自身 scope note "not wired here" 矛盾，且会使聚合 `__init__.py` re-export 与集成产物不一致）。收窄为仅 `_gqa_sw_fwd_varlen_kernel`（备份 `history_version/grouped_query_attention_sliding_window_varlen_s5_pre_narrow_stage0.py`）。属 import 胶水层改动，未触及调用点 / 配置 / dtype 逻辑。
- 保留产物内建机制（原样，未剥离、未调低）：
  1. **Expert 模式**（`ascend_mode = "Expert"`；产物 L74 `ASCEND_MODE = "Expert"`）。
  2. **CG-2026-0023 运行期 sync shim**（产物 L87-L111，`syn_instr_mode`→`sync_instr_mode` 重写）。
  3. **编译区分性重试**：针对本容器 `bishengir-compile` 进程状态相关的概率性 SIGSEGV（`err code: -11`），产物 L658-L678 保留 **24 次 + 0.3s 间隔** 重试。本次 report **未触发** SIGSEGV 重试路径（correctness 一次跑完）。
- 调用约定核对：wrapper `_gqa_sw_fwd_varlen_wrapped_kernel` 为 `_gqa_sw_fwd_varlen_kernel(params)(block_m, block_n, num_stages)(q,k,v,cu_q,cu_k,max_seqlen_q)` 两级调用，与集成 kernel 的接口契约一致（`max_seqlen_q` 在 host 折叠，内层 JIT 以 `max_seqlen_q` 为 trace 常量键）。
- config 传递核对：wrapper `default_config = {block_m:64, block_n:64|32, num_stages:1}`；kernel 工厂对 `(64,64,1)/(64,32,1)` 显式映射到设计默认 `(64,256)`，与 Stage 3 验证配置一致；report 输出的 benchmark `config` 为 `{block_m:64, block_n:64, num_stages:1}`。
- manifest `source.test` / `source.bench` 与 meta `test_path` / `bench_path` 一致；workloads 共 5 个（1 smoke + Llama-3.1-8B/70B × 短 total=2048/长 total=8192），report 正确性覆盖 26 用例、benchmark 覆盖 10 case。
- Stage 3 已知实现层修正**未被回退**：`L_max_bound` 在 `wl<0` 分支 `ceildiv(total_k, bn_eff)+1`（产物 L234-L235）；golden 用 log2 域（产物 L141/L171）。report 正确性全过佐证。

## Stage 5 门禁修复（gate-fix，承接上一次集成）

- 触发：`statectl gate 5` 报 `S5-ASCEND-MODE`：「wrapper 的 Kernel.ascend_mode 与集成 kernel 模式不一致」。
- 根因（与 conductor 定位一致，已核对 `gate_lint.py` L1482-L1514）：`S5-ASCEND-MODE` 用 AST 只收集**直接继承 `Kernel`** 的类（`base.id == "Kernel"`），并要求该类自身恰有 1 处 `ascend_mode` 类级常量赋值。本算子 wrapper 沿用 GPU 源两级类层次：基类 `_GQASlidingWindowVarlenFwdKernelBase(Kernel)` 被门禁选中，但其类体内无 `ascend_mode`（原先只在派生类 `GQASlidingWindowVarlenFwdKernel` 上）→ `len(class_modes)!=1` → 判失败。集成 kernel `_gqa_sw_fwd_varlen_kernel.py` 的 `ASCEND_MODE="Expert"`（L74）本身正确，未改动。
- 修复（唯一、最小、语义中性）：在基类 `_GQASlidingWindowVarlenFwdKernelBase` 类体内补 `ascend_mode = "Expert"`（wrapper L200），派生类上原有 `ascend_mode = "Expert"`（L284）保留不动。两处值一致，不改变任何调用/配置/dtype/结构。
- 备份：`history_version/grouped_query_attention_sliding_window_varlen_wrapper_s5_gatefix.py`
- 未改动：集成 kernel、`_gqa_sw_fwd_varlen_kernel_DESIGN.md` 快照、`tests/`、`benchmarks/`、`tileops/manifest/`、`tileops/ops/`、`tileops/workloads/`；产物内建机制（Expert 模式 / CG-2026-0023 sync shim / bishengir-compile 24 次重试）均未剥离。
- 自检：修改后 `S5-ASCEND-MODE` 消失；复跑单算子 report（wrapper 变更使旧报告 stale）后 `S5-REPORT-STALE` 亦消除，`gate 5` → `pass: true`。
- 复跑正确性：**26/26 passed**（`summary.correctness_passed=true`）；benchmark 仍 `partial`（10 case `invalid_profile_count=10`，评测层已知问题，不修复）。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261016_223115_117879_GroupedQueryAttentionSlidingWindowVarlenFwdOp/run.json`（gate-fix 后复跑）
- report_md: `examples/TileOPs/reports/tileops/20261016_223115_117879_GroupedQueryAttentionSlidingWindowVarlenFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261016_223115_117879_GroupedQueryAttentionSlidingWindowVarlenFwdOp/report.html`
- run_id: `20261016_223115_117879_GroupedQueryAttentionSlidingWindowVarlenFwdOp`
- 上一轮（gate-fix 前）报告：`examples/TileOPs/reports/tileops/20261016_220224_767491_GroupedQueryAttentionSlidingWindowVarlenFwdOp/run.json`
- report_status: **partial**（`run.json.status=partial`；CLI exit=1）——**partial 仅由 benchmark 段引起，correctness 段状态为 passed**
- correctness: **pass（26/26 cases）**；`run.json.correctness.status=passed`、`passed=26`、`failed=0`、`errors=0`、`skipped=0`；`summary.correctness_passed=true`、`pass_rate=1.0`
  - 26 = 2 smoke（`smoke-fwd-fp16` / `smoke-fwd-bf16`）+ 参数化形状/窗口/因果/GQA-ratio 用例 + metrics + validation（与测试文件/T3 一致）
  - correctness `max_abs_err`：最大 **9.77e-04**（smoke-bf16）；fp16 档多为 1.22e-04；均在测试容差内
- benchmark: **partial（0/10 有效 case）**；`summary.benchmark_requested=true`、`benchmark_passed=false`、`benchmark_failed=0`、`case_count=10`、`msprof_case_count=10`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`、**`invalid_profile_count=10`**
  - 全部 10 个 case 的 `profiler_valid=false`，`invalid_reason="unfiltered capture contains 6 distinct ops"`：msprof 采集窗口内捕获到 6 个不同算子（`summary.performance.cases[*].msprof.captured_op_count=6`）：
    - `_gqa_sw_fwd_varlen_main_mix_aic`（目标 kernel）
    - `GreaterEqual_*` / `ReduceAll_*` / `Sub_*` / `ReduceMax_*`（来自 Op `forward()` 的 host 侧校验 `torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1])` 与 `(cu_seqlens_q[1:]-cu_seqlens_q[:-1]).max()`）
    - `ZerosLike_*`（来自集成 kernel `wrapped()` 的 `zbuf = torch.zeros(...)`）
  - 因捕获非唯一算子，报告的 `op_name` 落到非 kernel 算子（如 `GreaterEqual_*`，min latency 1.38us），`latency_us` 与 `achieved_bandwidth_tb_s`（恒 1.75，理论参考）**不可归因于目标 kernel**，故本轮 benchmark 数值**无效**，仅记录不修复（详见 issues）。

## benchmark 记录（来自 run.json `performance.cases`；profiler 无效，数值仅供参考，勿引用为有效性能）

| case_id | label | dtype | config | latency_us | profiler_valid | invalid_reason |
|---|---|---|---|---|---|---|
| ...-c126a6672340 | gqa-sw-varlen-fwd-smoke | float16 | {bm64,bn64,ns1} | 3.36 | false | unfiltered capture contains 6 distinct ops |
| ...-4e0b9ce1a8e8 | gqa-sw-varlen-fwd-smoke | bfloat16 | {bm64,bn64,ns1} | 3.28 | false | 同上 |
| ...-b0e4e6267c86 | llama-3.1-8b-short-w256 | float16 | {bm64,bn64,ns1} | 3.50 | false | 同上 |
| ...-c42935f41bd4 | llama-3.1-8b-short-w256 | bfloat16 | {bm64,bn64,ns1} | 4.04 | false | 同上 |
| ...-87e2aca33b14 | llama-3.1-8b-long-w1024 | float16 | {bm64,bn64,ns1} | 3.84 | false | 同上 |
| ...-78a67f644b76 | llama-3.1-8b-long-w1024 | bfloat16 | {bm64,bn64,ns1} | 3.86 | false | 同上 |
| ...-4f6f5ee9cf7f | llama-3.1-70b-short-w256 | float16 | {bm64,bn64,ns1} | 3.76 | false | 同上 |
| ...-f76d12797acf | llama-3.1-70b-short-w256 | bfloat16 | {bm64,bn64,ns1} | 3.36 | false | 同上 |
| ...-1575d6eebd41 | llama-3.1-70b-long-w1024 | float16 | {bm64,bn64,ns1} | 4.30 | false | 同上 |
| ...-2bf46b6c5bbf | llama-3.1-70b-long-w1024 | bfloat16 | {bm64,bn64,ns1} | 4.30 | false | 同上 |

> 说明（仅记录，不修复、不重试，Stage 5 性能只报告）：`achieved_bandwidth_tb_s` 恒为理论参考 1.75 TB/s，roofline util ≈ 0.01%–0.02%，因 profiling 落点非目标 kernel，**该组数值不代表真实 kernel 性能**。

## 调试历史

- 无。集成脚本一次通过（import smoke pass）、单算子 report 一次运行即完成 correctness 26/26；未进入调试闭环，未做任何 kernel / wrapper 逻辑修改。`history_version/` 仅 1 份前置 wrapper 胶水收窄备份。
- benchmark 段 invalid profile 属报告/评测方法现象，按 Stage 5 约束「性能只报告、不修复、不重试」未触发调试。

## issues

- **Benchmark/profiling（R 类，报告不修复）**：所有 10 个 benchmark case `profiler_valid=false`，`invalid_reason="unfiltered capture contains 6 distinct ops"`。根因为 Op `forward()` 的 host 侧 NPU 校验（`torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1])`、`(cu_seqlens_q[1:]-cu_seqlens_q[:-1]).max()`）+ kernel `wrapped()` 的 `torch.zeros` 与目标 kernel 同处 msprof 采集窗口，捕获到 6 个算子而非唯一 kernel。固定长版姊妹集成时捕获为 1 个算子（`_gqa_sw_fwd_main_mix_aic`），故本现象为 varlen Op 特有的评测方法不兼容。修复涉及 `tileops/ops/` 或 `tileops/reporting/`（**均为禁止修改范围**），且 Stage 5 规定性能只报告 → 如实报告，交回 conductor 决策（建议：评测按 kernel 名过滤，或在计时区间外做 host 侧校验）。
- **Integrate-flow（R 类，继承姊妹）**：`.migration_meta.json` 的 `extracted_functions` 是 Stage 0 原始 extraction 列表（含 arch-specialized 变体），而 conductor 只迁移 architecture-generic 变体——`integrate_kernel.py` 直接用 `meta.extracted_functions` 会为未迁移变体报 `[error] no conductor product`（本次通过 `--functions` 显式收敛规避）。建议 conductor 写 meta 时收敛 `extracted_functions`，或 integrate 脚本对「无产物的函数」发 `[warn]` 并跳过。
- **Integrate-flow（R 类，继承姊妹）**：Stage 0 脚手架 wrapper 同时 import 被收敛掉的 Hopper 变体（与自身 scope note 矛盾），使聚合 `__init__.py` re-export 与集成产物不一致。本次以前置 import 收窄规避；建议 Stage 0 生成器按「conductor 迁移集」生成 wrapper import。
- **Integrate-flow（R 类，继承姊妹）**：脚本生成的选择块与 Stage 0 脚手架旧注释头合并形成两段 `# ----` 分隔头（wrapper L57-L77 附近），功能无害，记录不修复。
- 本次 harness 迁移 Stage 4 跳过，无 perf_opt 产物；perf_opt import 行为占位注释（baseline 默认激活）。
- 未发现任何要求改测试/跳过用例的指令性注入文本；所有读取文件按数据对待。
