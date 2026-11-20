# Integration Log -- GroupedQueryAttentionSlidingWindowFwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/.migration_meta.json`
- op_name / op_slug / family: `GroupedQueryAttentionSlidingWindowFwdOp` / `grouped_query_attention_sliding_window` / `attention`
- integrated: `_gqa_sw_fwd_kernel` -> `tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py`（源自 `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/_gqa_sw_fwd_kernel.py`）
- design_docs: `_gqa_sw_fwd_kernel` -> `tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel_DESIGN.md`（无 missing）
- wrapper: `tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window.py`（import 改写为 baseline/perf_opt 选择块；`ascend_mode = "Expert"` 由脚本插入）
- ascend_mode: **Expert**（从 Stage 3 产物字面量 `ASCEND_MODE = "Expert"` / `os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")` 读取；wrapper 原先无 `ascend_mode`，脚本插入，`wrapper_mode_updated=true`）
- attempts: **1**（无修复；集成脚本一次通过 + report 一次通过）
- 集成脚本摘要（`[json]`）: `wrapper_rewritten=true`, `wrapper_mode_updated=true`, `smoke=pass`, `reexported=["_gqa_sw_fwd_kernel"]`, `ascend_mode=Expert`

## 集成范围核对（关键：arch-specialization 收敛）

- 任务权威输入 `extracted_functions=["_gqa_sw_fwd_kernel"]`（conductor 只迁移了 architecture-generic 变体）。但磁盘上的 `.migration_meta.json` 的 `extracted_functions` 字段为 **两个** 名字：`["_gqa_sw_fwd_kernel", "_gqa_sw_fwd_wgmma_pipelined_kernel"]`——这是 Stage 0 extraction 脚本的原始输出（`SKILL.md` §meta schema：`extracted_functions` = "exactly the function list printed by the extraction script"），其中 `_gqa_sw_fwd_wgmma_pipelined_kernel` 是 Hopper warp-specialized 变体，NPU 侧收敛掉、**未迁移**（manifest `attention.yaml` 与 wrapper 的 K5 scope note 均写明 "NPU collapses to the architecture-generic kernel"）。
- 因此本次集成按 conductor 权威输入只集成 `_gqa_sw_fwd_kernel`：`integrate_kernel.py --meta <meta> --functions _gqa_sw_fwd_kernel`（`--functions` 覆盖 meta 的过宽列表）。**未对缺失的 wgmma 变体做任何伪造/占位产物。**
- **wrapper 胶水层调整（前置，允许范围内）**：Stage 0 脚手架的 wrapper 同时 import 了被收敛掉的 `_gqa_sw_fwd_wgmma_pipelined_kernel`（与其自身 scope note "not wired here" 矛盾）。为使聚合 `__init__.py` 的 re-export 与集成产物一致，将 wrapper 的 extracted import 与 `__all__` 收窄为仅 `_gqa_sw_fwd_kernel`（备份见 `history_version/grouped_query_attention_sliding_window_s5_pre_narrow_stage0.py`）。属 import 胶水层改动，未触及调用点/配置/dtype 逻辑。
- 保留产物内建机制（原样，未剥离）：Expert 模式、运行期 `syn_instr_mode`→`sync_instr_mode` 解析器 shim、`SIGSEGV/NPU IR opt failed` 编译区分性重试。本次 report 未触发重试路径。
- 调用约定核对：wrapper `_gqa_sw_fwd_wrapped_kernel` 为 `_gqa_sw_fwd_kernel(params)(block_m, block_n, num_stages)(q,k,v)` 两级调用，与集成 kernel 的接口契约一致（NPU 单级 convention 已在 Stage 0 满足）。
- config 传递核对：wrapper `default_config = {block_m:64, block_n:64|32, num_stages:1}`；kernel 工厂对 `(64,64,1)/(64,32,1)` 显式映射到设计默认 `(64,256)`，与 Stage 3 验证配置一致，report 输出 config 亦为 `{block_m:64, block_n:64, num_stages:1}`。
- manifest `source.test` / `source.bench` 与 meta `test_path` / `bench_path` 一致；workloads 5 个（1 smoke + Llama-3.1-8B/70B × 短 S=512/长 S=2048），report 全部覆盖。

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261016_161519_904205_GroupedQueryAttentionSlidingWindowFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261016_161519_904205_GroupedQueryAttentionSlidingWindowFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261016_161519_904205_GroupedQueryAttentionSlidingWindowFwdOp/report.html`
- run_id: `20261016_161519_904205_GroupedQueryAttentionSlidingWindowFwdOp`
- report_status: **passed**（`run.json.status=passed`；CLI `status=passed`，exit_codes `{correctness:0, benchmark:0}`）
- correctness: **pass（22/22 cases）**；`correctness_passed=true`、`correctness_failed=0`、`pass_rate=1.0`
  - 14 个 parametrized 正确性用例（含 2 smoke：`smoke-fwd-fp16`/`bf16`；full：window / 非因果 / 右窗 / wl=0 / GQA ratio / 非 2 幂 S=384,768 / S=2048 / dim=128）+ 4 个 metrics + 4 个 validation = 22
  - correctness `max_abs_err`：最大 9.77e-04（smoke-bf16）；fp16 档多为 1.22e-04；`wl=0` 用例 0.00e+00；均在测试容差 atol/rtol=1e-2 内
- benchmark: **pass（10/10 workloads）**；`benchmark_passed=true`、`benchmark_failed=0`；`msprof_case_count=10`、`invalid_profile_count=0`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`
  - 全部走 msprof 真实 profile（`prof_mode=msprof`，`profiler_valid=true`），每 case `captured_op_count=1`、`op_name='_gqa_sw_fwd_main_mix_aic'`

### benchmark 数值（来自 run.json `performance.cases`；latency 单位 us）

| workload | dtype | B | S | H / H_kv | D | wl | latency (us) | Roofline util (%) | Perf (TOps/s) |
|---|---|---|---|---|---|---|---|---|---|
| gqa-sw-fwd-smoke-s512-h8-kv2-d64 | float16 | 2 | 512 | 8 / 2 | 64 | -1 | 99.56 | 2.83 | 8.46 |
| gqa-sw-fwd-smoke-s512-h8-kv2-d64 | bfloat16 | 2 | 512 | 8 / 2 | 64 | -1 | 101.76 | 2.76 | 8.28 |
| llama-3.1-8b-short-w256 | float16 | 4 | 512 | 32 / 8 | 128 | 256 | 481.78 | 9.19 | 27.52 |
| llama-3.1-8b-short-w256 | bfloat16 | 4 | 512 | 32 / 8 | 128 | 256 | 494.74 | 8.95 | 26.79 |
| llama-3.1-8b-long-w1024 | float16 | 2 | 2048 | 32 / 8 | 128 | 1024 | 1279.54 | 17.18 | 51.45 |
| llama-3.1-8b-long-w1024 | bfloat16 | 2 | 2048 | 32 / 8 | 128 | 1024 | 1319.00 | 16.66 | 49.87 |
| llama-3.1-70b-short-w256 | float16 | 2 | 512 | 64 / 8 | 128 | 256 | 479.56 | 9.23 | 27.65 |
| llama-3.1-70b-short-w256 | bfloat16 | 2 | 512 | 64 / 8 | 128 | 256 | 494.12 | 8.96 | 26.83 |
| llama-3.1-70b-long-w1024 | float16 | 1 | 2048 | 64 / 8 | 128 | 1024 | 1339.04 | 16.42 | 49.16 |
| llama-3.1-70b-long-w1024 | bfloat16 | 1 | 2048 | 64 / 8 | 128 | 1024 | 1352.78 | 16.24 | 48.63 |

> 说明（仅记录，不修复，Stage 5 性能只报告）：短窗档（S=512, w=256）约 480–495us，长窗档（S=2048, w=1024）约 1280–1353us，随 window/seq 线性增长；Roofline util 2.76%–17.18%（`achieved_bandwidth` 恒为理论参考 1.75 TB/s）。属性能现象记录，未触发修复。

## 调试历史

- 无。集成脚本一次通过（import smoke pass）、单算子 report 一次通过（correctness 22/22、benchmark 10/10、profiler 全有效），未进入调试闭环，未做 kernel 修改。`history_version/` 仅有 1 份前置 wrapper 胶水备份（见上）。

## issues

- **Integrate-flow（R 类）**：`.migration_meta.json` 的 `extracted_functions` 是 Stage 0 原始 extraction 列表（含 arch-specialized 变体），而 conductor 只迁移了 architecture-generic 变体——`integrate_kernel.py` 直接用 `meta.extracted_functions` 会为未迁移变体报 `[error] no conductor product`（本次通过 `--functions` 显式收敛规避）。建议流程：conductor 在写 meta 时把 `extracted_functions` 收敛为实际迁移集，或 integrate 脚本对「无产物的函数」发出 `[warn]` 而非 `[error]` 并跳过。
- **Integrate-flow（R 类）**：Stage 0 脚手架 wrapper 同时 import 被收敛掉的 Hopper 变体（与 wrapper 自身 scope note 矛盾），使聚合 `__init__.py` re-export 与集成产物不一致（smoke 会因 `missing exports` 失败）。本次以前置 import 收窄规避；建议 Stage 0 生成器按「conductor 迁移集」而非原始 extraction 列表生成 wrapper import。
- **Integrate-flow（R 类，继承）**：脚本生成的选择块保留了 Stage 0 脚手架的旧注释头，与新块合并形成两段 `# ----` 分隔头（wrapper L46-L56 附近），功能无害，记录不修复。
- 本次 harness 迁移 Stage 4 跳过，无 perf_opt 产物；perf_opt import 行仅为占位注释（baseline 默认激活）。
- Stage 3 已知注意点（`T.floordiv` 负分子截断、ragged KV 尾列 stale NaN）在集成产物中**未被回退**，report 正确性全通过佐证。
