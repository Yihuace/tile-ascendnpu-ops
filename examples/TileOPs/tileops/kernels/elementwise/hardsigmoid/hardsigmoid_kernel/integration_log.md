# Integration Log -- HardsigmoidFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/.migration_meta.json
- integrated: {"hardsigmoid_fwd_kernel": "tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel.py"}
- design_docs: {"hardsigmoid_fwd_kernel": "tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md"}
- ascend_mode: Developer (read from Stage 3 kernel `ASCEND_MODE = "Developer"`; synced to wrapper `Kernel.ascend_mode`, unchanged)
- attempts: 1 (one wrapper glue fix)

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261012_223321_664388_HardsigmoidFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261012_223321_664388_HardsigmoidFwdOp/report.md
- correctness: pass (7/7 cases, max_abs_err = 5.96e-08)
- benchmark: pass (9/9 cases, 6 msprof profiles, all profiler_valid=true, no fallback / no invalid profile)

Benchmark 数值摘要（msprof，latency_us / roofline utilization %）：
- mbv3-se-gate (float16): 3.12 us / 0.46%
- mbv3-se-gate (bfloat16): 3.06 us / 0.53%
- mbv3-se-gate-deep (float16): 2.26 us / 2.53%
- mbv3-se-gate-deep (bfloat16): 2.52 us / 2.56%
- fc-wide (float16): 30.32 us / 48.27%
- fc-wide (bfloat16): 29.88 us / 55.07%

## 调试历史

- attempt 1: 症状 `AttributeError: 'int' object has no attribute 'device'`（5 个 op 用例失败）。
  分类：wrapper 胶水错误（calling convention，非 kernel）。
  根因：脚手架 wrapper 的 `forward()` 沿用 GPU `threads/npt` 调用约定
  （`self.kernel(threads, npt)` → `self._compiled_fn(x)`），而 NPU kernel 工厂
  `hardsigmoid_fwd_kernel(N, dtype, block_size=...)`（`@tilelang.jit`）返回编译产物，
  以 `kernel(x, N)` 调用；同时 `default_config` 的 `block_size` 为 GPU 派生 512/1024，
  而 DESIGN.md §5.2 指定 NPU 设计值 8192。
  修复：`forward()` 改为 `hardsigmoid_fwd_kernel(N, dtype, block_size=block_size)(x, N)`；
  `default_config` 改为 `{"block_size": 8192}`（单值，全 dtype）。
  结果：重跑完整 report，correctness 7/7、benchmark 9/9，status=passed。
  备份：history_version/hardsigmoid_s5_attempt1.py

## 已知非阻塞现象（不影响结果）

- kernel 内 `T.cast(float("nan"), "float32")` 使 tilelang kernel cache 每次构建报
  `Error loading metadata: Wrong value format`（JSON 无法表示 NaN）→ cache miss 重编译。
  正确性无关，仅增加编译耗时（已由 Stage 3 RETROSPECTIVE 记录）。

## integrate_kernel.py run (2026-10-13T09:09:56+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Developer
- functions: hardsigmoid_fwd_kernel
- smoke: pass

## History

> 修订报告 rev.2（2026-10-13，M1–M8）期间的集成侧改动留痕（重跑集成前已备份集成包到
> `/tmp/opencode/hardsigmoid_kernel.bak_m8_20261013_090943`）。

- M5（lint 收口）：wrapper `hardsigmoid.py` 修复 I001（导入块空行）；Stage-3 kernel `__all__`
  排序（RUF022）；`examples/TileOPs/pyproject.toml` 增 per-file-ignores：
  `tileops/kernels/**/*_kernel/*.py = ["E402"]`（TILELANG_ASCEND_MODE 须先于 tilelang import，
  E402 为固有形态）与 `.../history_version/*.py`（冻结备份）。→ `ruff check` 由 9 errors 转 0。
- M8（工具侧）：`.agents/skills/add-npu-op/scripts/integrate_kernel.py` 新增
  `write_integration_log()`，集成后向本文件追加必写字段 `wrapper_rewritten`（wrapper 最终形态），
  幂等重跑后 Stage 5 护栏 1 由 FAIL 转 PASS。本次重跑即由该函数追加下方 run 段。
- 语义冻结：未改 kernel 语义，未重跑 Stage 1–3（DESIGN v1 / REVIEW / Stage-3 kernel 均未改逻辑，
  仅 `__all__` 顺序）。
