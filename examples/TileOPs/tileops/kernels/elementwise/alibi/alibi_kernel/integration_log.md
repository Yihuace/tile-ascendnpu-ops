# Integration Log -- AlibiFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/alibi/.migration_meta.json
- op_slug: alibi
- family: elementwise
- integrated:
  - `_make_alibi_kernel` -> `tileops/kernels/elementwise/alibi/alibi_kernel/_make_alibi_kernel.py`
- design_docs:
  - `_make_alibi_kernel` -> `tileops/kernels/elementwise/alibi/alibi_kernel/_make_alibi_kernel_DESIGN.md`
  - missing: none
- ascend_mode: Developer (read from the Stage 3 product `ASCEND_MODE` literal;
  wrapper `Kernel.ascend_mode` was already `"Developer"` -> unchanged)
- attempts: 1

## 集成步骤

1. `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta \
   tileops/kernels/elementwise/alibi/.migration_meta.json`
   （workdir = `examples/TileOPs/`）-> import smoke `pass`。
   - 复制 kernel + `_make_alibi_kernel_DESIGN.md`；
   - 生成聚合 `__init__.py`（re-export `_make_alibi_kernel`）；
   - wrapper import 改写为 baseline/perf_opt 切换块（baseline 默认激活，
     `# from .alibi_kernel.perf_opt._make_alibi_kernel import _make_alibi_kernel`
     注释占位）；
   - `[warn] no DESIGN.md ...`：无（Stage 1 交付件齐全）。

2. Report（workdir = `examples/TileOPs/`）：
   `python -m tileops.reporting.cli run --op AlibiFwdOp --prof-mode msprof`

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261013_173535_898248_AlibiFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_173535_898248_AlibiFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_173535_898248_AlibiFwdOp/report.html
- correctness: **pass** (2/2 cases; `test_alibi[128-8-dtype0]` fp16,
  `test_alibi[128-8-dtype1]` fp32, tol fp16 1e-2 / fp32 1e-5)
- benchmark: **pass** (4/4 cases, msprof; no profiler fallback / kernel-filter
  fallback / invalid profile):
  | label | latency (us) | ratio (%) | achieved BW (TB/s) |
  |---|---:|---:|---:|
  | llama-prefill-2k-fp16 | 598.7720 | 27.34 | 0.4483 |
  | llama-prefill-2k-bf16 | 608.0522 | 26.92 | 0.4415 |
  | llama-prefill-4k-fp16 | 2320.4863 | 28.21 | 0.4627 |
  | llama-prefill-4k-bf16 | 2391.2478 | 27.38 | 0.4490 |
- report_status: passed
- run.json summary: `correctness_passed=true`, `benchmark_passed=true`,
  `msprof_case_count=4`, `profiler_fallback_count=0`,
  `kernel_filter_fallback_count=0`, `invalid_profile_count=0`

## 调试历史

- **attempt 1** (integration package as shipped by Stage 0/Stage 3):
  - 症状: 正确性直接 SIGABRT（exit code -6，correctness.xml missing）。
    设备报 `aivec error ... The DDR address of the MTE instruction is out of
    range`，retCode 507057。Python 栈: `alibi.py:156 init_config` ->
    `jit_npu.py:1333 __call__`。
  - 分类: **wrapper 胶水错误**（host 集成层；非 pass/codegen/runtime kernel 问题）。
  - 根因: Stage 0 移植的 wrapper 仍按 GPU `K11` 假设写成
    `self.kernel = _make_alibi_kernel(S, H, dtype)` 后再
    `self._compiled_fn = self.kernel(self.config["block_size"])`。但 Stage 3
    NPU 工厂的签名是
    `_make_alibi_kernel(seq_len, num_heads, dtype, bi=None, bj=None, budget, core_num)`
    且按 DESIGN §0.6 R5 用**显式输出参数**（无 `out_idx`，CG-2026-0016）。
    因此 `self.kernel(2048)` 把整数 `2048` 当作 `out` 张量传入设备 -> DDR
    越界崩溃；且 `forward()` 原本无参调用 `self._compiled_fn()`，也没有分配
    输出张量的路径。
  - 修复（仅 wrapper 胶水层，符合例外授权；DESIGN §0.6 R5 / 第 568 行明确要求）:
    `tileops/kernels/elementwise/alibi/alibi.py`
    - 移除 GPU-only `_DEFAULT_BLOCK_SIZE`；`default_config` 改为
      `{"bi": None, "bj": None}`（NPU 工厂按 UB 预算自动选 `(bi,bj)`，可被
      Stage 4 覆盖）；
    - `init_config` 不再用 `block_size` 调用已编译 kernel，而是
      `self.kernel = _make_alibi_kernel(seq_len, num_heads, dtype_str, **tile)`
      （`tile` 只含非 None 的 `bi`/`bj`）；
    - `forward` 在 `get_device_backend().name` 上 `torch.empty(H,S,S,...)`
      分配输出并 `self._compiled_fn(out)`，返回 `out`。
    - `ASCEND_MODE` / import 切换块未被手工改动（由集成脚本管理）。
  - 备份: `tileops/kernels/elementwise/alibi/alibi_kernel/history_version/alibi_s5_attempt1.py`
  - 结果: 修正后完整 report `status=passed`（正确性 2/2，benchmark 4/4）。
- 未修改 `tests/` / `benchmarks/` / `manifest/` / `ops/` / `workloads/` /
  `_make_alibi_kernel_DESIGN.md`。

## Issues

- report.md 附 4 条 `missing tileops candidate record` warning（每 workload
  一条）。report 整体 `status=passed`，正确性与 benchmark 均有效，非阻断；
  仅记录。
