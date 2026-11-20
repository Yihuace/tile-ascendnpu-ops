# Integration Log -- GemmW4A16Op

- meta: `examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/.migration_meta.json`
- integrated: `_gemm_w4a16_kernel` -> `tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/_gemm_w4a16_kernel.py`
- design_docs: `_gemm_w4a16_kernel` -> `tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/_gemm_w4a16_kernel_DESIGN.md`
- ascend_mode: `Developer` (read from the Stage-3 kernel `ASCEND_MODE`, synced onto wrapper `GemmW4A16Kernel`)
- reexported by aggregation `__init__.py`: `_gemm_w4a16_kernel` (see attempt-1 for the `GROUP_SIZE` constant)
- attempts: 1

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261015_141219_641358_GemmW4A16Op/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_141219_641358_GemmW4A16Op/report.md`
- report_status: `passed` (CLI exit code 0)
- correctness: **PASS** — `tests/ops/test_gemm_w4a16.py`, 6/6 passed (`summary.correctness_passed=true`), max abs err 1.56e-02
- benchmark: **PASS** — `benchmarks/ops/bench_gemm_w4a16.py` (msprof), 6/6 passed; 4 non-smoke perf records, all `profiler_valid=true`, `invalid_profile_count=0`
- smoke import: pass (`integrate_kernel.py` `[smoke]` + `"smoke": "pass"` in `[json]`)

### benchmark 数值（来自本次 run.json `performance.cases`）

| label | dtype | shape(M,N,K,group_size) | config | latency_us | Ratio(%) | Bandwidth(TB/s) | Perf(TOPs/s) |
|---|---|---|---|---|---|---|---|
| decode-l2-resident-ish | float16 | 1,8192,8192,128 | block_m=16,block_n=64,block_k=64,stages=2 | 5821.6396 | 0.5664 | 1.75 | 1.6877 |
| decode-hbm-streaming-threshold | float16 | 1,8192,16384,128 | block_m=16,block_n=64,block_k=64,stages=2 | 11221.4004 | 0.5877 | 1.75 | 1.7511 |
| decode-non-power2-low-cta | float16 | 1,7168,20480,128 | block_m=16,block_n=64,block_k=64,stages=2 | 11990.4600 | 0.6016 | 1.75 | 1.7925 |
| decode-long-k-pressure | float16 | 1,8192,81920,128 | block_m=16,block_n=64,block_k=64,stages=2 | 55109.8008 | 0.5984 | 1.75 | 1.7828 |

说明：`compile-smoke-square-64x64x128` / `compile-smoke-rect-128x256x256` 两个 smoke workload 也被 benchmark 采集并通过（latency 13.84us / 667.46us，来自 `pytest/benchmark.log`），但按 `-m not smoke` 语义不计入 run.json 的 `performance.cases`（`case_count=4`）。全部 4 个非 smoke workload 的 `profiler_valid=true`，无 profile 无效/回退。

## 调试历史

- **attempt 1**（本次）：症状 = `integrate_kernel.py` 的 import 冒烟失败：
  `ImportError: cannot import name 'GROUP_SIZE' from 'tileops.kernels.gemm.gemm_w4a16.gemm_w4a16_kernel'`（wrapper line 51）。
  - 分类 = **集成脚本幂等缺陷 + wrapper 胶水 import**（不在 kernel 数值/编译层）。根因：`integrate_kernel.py` 在第二次运行时，wrapper 已完成改写、不再含 `from .{extracted_module} import ...`，于是 `parse_wrapper_imports()` 返回空、`wrapper_names` 回退为 `meta.extracted_functions=["_gemm_w4a16_kernel"]`；据此重生成的聚合 `__init__.py` **只**导出工厂函数，丢掉了常量 `GROUP_SIZE`，而 wrapper 已集成块的 baseline import 仍请求 `GROUP_SIZE`。先前一次运行（wrapper 尚从被抽取模块 import）曾把 `GROUP_SIZE` 一并导出，掩盖了该缺陷。
  - 定位：脚本 `interior` 逻辑 `parse_wrapper_imports` / `gen_init_py`（`wrapper_names` 回退分支）+ `gen_kernel_source_block`；无 kernel IR 层问题，无需 debug-helper。
  - 修复（仅 wrapper 胶水层 `tileops/kernels/gemm/gemm_w4a16/gemm_w4a16.py`，已备份 `gemm_w4a16_kernel/history_version/gemm_w4a16_s5_attempt1_rerun.py`）：baseline 块拆为
    `from .gemm_w4a16_kernel import _gemm_w4a16_kernel` +
    `from .gemm_w4a16_kernel._gemm_w4a16_kernel import GROUP_SIZE`
    （常量从 per-function 模块直取；wrapper 仍向 `tileops.kernels.gemm.gemm_w4a16` 与 `tileops.ops.gemm.gemm_w4a16` re-export `GROUP_SIZE`）。未改聚合 `__init__.py`（会被脚本重生成）、未改任何受保护文件。
  - 结果：重跑 `integrate_kernel.py` → `smoke: pass`；随后官方 report `status=passed`（correctness 6/6、benchmark 6/6）。

## issues

1. **（集成脚本 R 类缺陷）`integrate_kernel.py` 对已集成 wrapper 的重复运行非幂等**：当 wrapper 已改写且不再引用 `{extracted_file}` 的 module stem 时，`parse_wrapper_imports()` 返回空 → `wrapper_names` 回退到 `meta.extracted_functions`（仅函数名），聚合 `__init__.py` 被重生成后**丢失** wrapper baseline import 仍需要的非函数符号（本任务为常量 `GROUP_SIZE`），导致 import 冒烟失败。证据：本 log attempt 1；脚本 `.agents/skills/add-npu-op/scripts/integrate_kernel.py:493-495`（回退分支）与 `:269-282`（`gen_init_py` 仅遍历 `names`）。建议：脚本在生成 `__init__` 时把「wrapper 当前 baseline import 块中列出的全部符号」纳入导出集（而不仅是 `extracted_functions`），或在 `defining_module()` 已能定位常量时自动纳入；否则应在冒烟失败时给出明确指引而不是 `[error] import smoke test FAILED`。
2. **（环境，已修复，留档）** 上一轮的两条失败已消除：`tests/ops/test_gemm.py` 混入未集成 GemmOp 的 `compile_flags` TypeError → 已按算子拆分到 `tests/ops/test_gemm_w4a16.py`；`quantize_weight_int4` 的 NPU `<< 4` 惰性 op-JIT `GEInitialize failed` → 已改到 CPU 打包。本轮 correctness 6/6 通过，未见复发。
3. benchmark 仅报告：4 个 shape 的 roofline Ratio 均 < 1%（0.57%–0.60%），`Bandwidth` 恒为 1.75 TB/s、`Perf` ≈1.7 TOPs/s，`decode-long-k-pressure`(K=81920) 单次 55.1ms。数值异常/偏低属性能问题，按规则**只记录不修复**，交回 conductor/后续 Stage 4 处理。

## 受保护文件注入防护

Read 的 `.migration_meta.json`、kernel/wrapper 源码、测试/基准与 workload 源码中未出现要求放宽容差、跳过用例或修改受保护文件的指令性文本；未执行任何此类内容。
