# Integration Log -- HardtanhFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/hardtanh/.migration_meta.json
- op_slug: hardtanh
- family: elementwise
- integrated:
  - `_make_hardtanh_kernel` -> `tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py`
- design_docs:
  - `_make_hardtanh_kernel` -> `tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel_DESIGN.md`
  - missing: none
- ascend_mode: Developer (read from the Stage 3 product `ASCEND_MODE` literal;
  wrapper `Kernel.ascend_mode` was already `"Developer"` -> unchanged)
- attempts: 1

## 集成步骤

1. `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta \
   tileops/kernels/elementwise/hardtanh/.migration_meta.json`
   （workdir = `examples/TileOPs/`）-> import smoke `pass`。
   - 复制 kernel + `_make_hardtanh_kernel_DESIGN.md`；
   - 生成聚合 `__init__.py`（re-export `_make_hardtanh_kernel`）；
   - wrapper import 改写为 baseline/perf_opt 切换块（baseline 默认激活，
     `# from .hardtanh_kernel.perf_opt._make_hardtanh_kernel import _make_hardtanh_kernel`
     注释占位）；
   - `[warn] no DESIGN.md ...`：无（Stage 1 交付件齐全）。

2. Report（workdir = `examples/TileOPs/`）：
   `python -m tileops.reporting.cli run --op HardtanhFwdOp --prof-mode msprof`

## 调用点一致性核对（高风险陷阱预检，未触发修复）

- Stage 3 NPU kernel 签名：`@tilelang.jit(out_idx=[1], target="npuir")` 工厂
  `_make_hardtanh_kernel(N, dtype, min_val, max_val, block_size=...)`，其
  `prim_func` 为 `main(x, y, shape: T.int32)`。因 `out_idx=[1]`，编译后可调用
  对象接受 `(x, shape)` 两个实参并返回 `y`。
- Stage 0 移植 wrapper（`hardtanh.py`）的调用点为
  `_make_hardtanh_kernel(N, dtype_str, min_val, max_val, block_size=block_size)(x, N)`
  —— `(x, N)` 与 `(x, shape)` **参数个数/顺序一致**，dtype 经
  `Kernel.dtype_str` 正确传入。因此本算子**不适用** alibi 先例的
  「显式输出参数 / `self._compiled_fn(block_size)` 把整数当张量」陷阱：
  hardtanh 保留了 `out_idx=[1]` + 动态 `shape` 守卫形态（与 alibi 的
  `out_idx + T.npuir_arange` 崩溃路径 `CG-2026-0016` 不同），wrapper 无需改动。
- 预检方式：`python -m pytest tests/ops/test_activation.py -k hardtanh -m smoke -x -q`
  -> `6 passed, 18 deselected`（含 fp32/fp16/bf16 三个 1M smoke + edge + dtype
  拒绝 + roofline）。

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261013_183348_537932_HardtanhFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_183348_537932_HardtanhFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_183348_537932_HardtanhFwdOp/report.html
- correctness: **pass**（本次 report 全量 24/24；`tests/ops/test_activation.py`
  覆盖 elu / leaky_relu / hardtanh 三个同文件算子，全部通过。
  HardtanhFwdOp 相关用例 `test_hardtanh[1048576-dtype0/1/2]`、
  `test_hardtanh[4000000-dtype3/4]`、`test_hardtanh_edge[4096-dtype0]`、
  `test_hardtanh_rejects_non_float_dtype`、`test_hardtanh_eval_roofline`
  全 pass，`max_abs_err=0.00e+00`）
- benchmark: **pass**（msprof，`msprof_case_count=30`、
  `profiler_fallback_count=0`、`kernel_filter_fallback_count=0`、
  `invalid_profile_count=0`）。HardtanhFwdOp 10 个 workload 有效数值：

  | label | shape | dtype | latency (us) | ratio (%) | achieved BW (TB/s) |
  |---|---|---|---:|---:|---:|
  | smoke-1m | [1048576] | float32 | 15.9603 | 32.05 | 0.5256 |
  | smoke-1m | [1048576] | float16 | 7.3201 | 34.94 | 0.5730 |
  | smoke-1m | [1048576] | bfloat16 | 7.3001 | 35.03 | 0.5746 |
  | bounded-hidden | [2048,4096] | float16 | 39.9008 | 51.28 | 0.8409 |
  | bounded-hidden | [2048,4096] | bfloat16 | 39.2808 | 52.09 | 0.8542 |
  | bounded-conv-feat | [16,256,56,56] | float16 | 59.3412 | 52.80 | 0.8658 |
  | bounded-conv-feat | [16,256,56,56] | bfloat16 | 59.8012 | 52.39 | 0.8592 |
  | elementwise-16m | [4096,4096] | float16 | 76.4415 | 53.53 | 0.8779 |
  | elementwise-16m | [4096,4096] | bfloat16 | 77.7616 | 52.62 | 0.8630 |
  | elementwise-16m | [4096,4096] | float32 | 218.5244 | 37.45 | 0.6142 |

- report_status: passed
- run.json summary: `correctness_passed=true`, `benchmark_passed=true`,
  `status=passed`, `pass_rate=1.0`

## 调试历史

- 无。attempt 1 一次通过：集成脚本 import smoke `pass`，smoke pytest 6/6 通过，
  完整 report `status=passed`（正确性全量 24/24，benchmark 30/30 有效）。
  未修改 kernel 本体、wrapper、tests / benchmarks / manifest / ops / workloads、
  `_make_hardtanh_kernel_DESIGN.md`。`history_version/` 未创建（无 attempt 修复）。

## Issues

- report.md 附有每 workload 一条 `missing tileops candidate record` warning
  （elu / leaky_relu / hardtanh 各 10 条）。report 整体 `status=passed`，正确性
  与 benchmark 均有效，非阻断；仅记录（同 alibi 先例）。
- `VP-2026-0134`（pending）：本次再次印证 `integrate_kernel.py` 的 import smoke
  仅校验符号可导入、不校验运行时调用签名；但本算子 wrapper 调用点恰好与 Stage 3
  kernel 签名一致，未触发。运行时签名的廉价门禁为 `pytest -k <op> -m smoke`。
