# Integration Log -- GeluAndMulFwdOp

- meta: tileops/kernels/elementwise/gelu_and_mul/.migration_meta.json
- integrated: `gelu_and_mul_fwd_kernel` -> `gelu_and_mul_kernel/gelu_and_mul_fwd_kernel.py`
- design_docs: `gelu_and_mul_fwd_kernel` -> `gelu_and_mul_kernel/gelu_and_mul_fwd_kernel_DESIGN.md`
- attempts: 0 (attempt 1 root cause lay in a frozen test/reference; no allowed-file change could
  affect it).  **Attempt 2** (this run, `attempt_index=2`): reference fix applied by conductor at
  Stage 0; re-ran integration + report; no new debug iteration consumed.

## Stage 5 attempt 2 — result (`run_id=20261013_215524_720318_GeluAndMulFwdOp`)

- verdict: **INTEGRATE_COMPLETED**
- report_run_json: `examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/report.md`
- report status: **passed**
- correctness: **pass 11/11** (`summary.correctness_passed=true`, `pass_rate=1.0`)
- benchmark: **pass 7/7** (`summary.benchmark_passed=true`); `msprof_case_count=5`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`
- integration script re-run (idempotent): `wrapper_rewrite_action=already integrated`,
  `ascend_mode=Developer`, smoke `pass`.

### Per-shape perf table (from `run.json.performance.cases`, msprof)

| label | params (M, N, dtype) | latency_us | roofline_util % | bandwidth TB/s | perf TOPS |
|---|---|---:|---:|---:|---:|
| llama-8b-ffn-prefill | 2048, 14336, fp16 | 878.76 | 43.21 | 1.75 | 1.964 |
| llama-8b-ffn-prefill | 2048, 14336, bf16 | 877.64 | 43.99 | 1.75 | 2.000 |
| llama-8b-ffn-decode | 1, 14336, bf16 | 3.96 | 4.77 | 1.75 | 0.217 |
| qwen3-8b-swiglu-b32 | 32, 12288, fp16 | 15.12 | 33.64 | 1.75 | 1.529 |
| qwen3-8b-swiglu-b32 | 32, 12288, bf16 | 14.94 | 34.62 | 1.75 | 1.574 |

### Correctness per-case `max_abs_err`

| case | shape / dtype | max_abs_err |
|---|---|---:|
| test_gelu_and_mul_op[16-128-dtype0] | 16x128 fp32 | 1.43e-06 |
| test_gelu_and_mul_op[16-128-dtype1] | 16x128 fp16 | 3.81e-06 |
| test_gelu_and_mul_op[16-128-dtype2] | 16x128 bf16 | 0.00e+00 |
| test_gelu_and_mul_op[2048-14336-dtype3] | 2048x14336 fp16 | 3.91e-03 |
| test_gelu_and_mul_op[2048-14336-dtype4] | 2048x14336 bf16 | 3.12e-02 |
| test_gelu_and_mul_op[1-14336-dtype5] | 1x14336 bf16 | 6.10e-05 |

Operator aggregate `max_abs_err=0.0312` (bf16 full shape; within `atol+rtol*|expected|`).

## Guardrails (§2.1) — raw results (attempt 2)

```
$ grep -n "wrapper_rewritten" tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/integration_log.md
88:- wrapper_rewritten: true
96:- wrapper_rewritten: true
$ grep -n "already integrated\|wrapper_rewrite_action" .../integration_log.md
89:- wrapper_rewrite_action: rewritten
97:- wrapper_rewrite_action: already integrated
$ python -c "import importlib; w=importlib.import_module('tileops.kernels.elementwise.gelu_and_mul.gelu_and_mul'); print(w.gelu_and_mul_fwd_kernel.__module__)"
tileops.kernels.elementwise.gelu_and_mul.gelu_and_mul_kernel.gelu_and_mul_fwd_kernel
```

- Guardrail 1 `wrapper_rewritten`: **true**  (PASS)
- Guardrail 2 idempotent-rerun marker: `already integrated` hit — normal signal for an idempotent
  re-run; the script also reports `wrapper_rewrite_action: already integrated`.  Judged jointly with
  Guardrail 3.  (PASS)
- Guardrail 3 baseline lineage: `__module__` =
  `tileops.kernels.elementwise.gelu_and_mul.gelu_and_mul_kernel.gelu_and_mul_fwd_kernel`;
  `perf_opt` **not** in the path -> baseline (not perf_opt) lineage.  (PASS)

> Note: the §2.1 command imports `tileops.kernels.<family>.<op_slug>` and reads
> `w.<func>.__module__`; here the function is re-exported through the `_kernel` subpackage, so the
> literal command raises `AttributeError`.  The intended check is run against the wrapper module
> (shown above) — still resolves to the baseline module.  No glue/package change made.

## integrate_kernel.py run

- wrapper rewritten -> baseline/perf_opt kernel-source selection block (baseline active).
- ascend_mode: Developer (unchanged; wrapper already declared `ascend_mode = "Developer"`).
- design doc copied to `gelu_and_mul_fwd_kernel_DESIGN.md`.
- import smoke: pass.
- no wrapper glue fix needed: the scaffolded wrapper calls
  `gelu_and_mul_fwd_kernel(self.M, self.N, self.dtype_str)(x)` and keeps
  `default_config={}` (K11), which matches the Stage-3 factory
  `gelu_and_mul_fwd_kernel(M, N, dtype, output_dtype=None, block_N=2048, num_cores=None)`.

## 验证结果 (attempt 1 — historical; superseded by attempt 2 above)

- report_run_json: `examples/TileOPs/reports/tileops/20261013_213708_656857_GeluAndMulFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261013_213708_656857_GeluAndMulFwdOp/report.md`
- correctness: **fail** (10/11 passed).  Single failing case
  `tests.ops.test_gelu_and_mul::test_gelu_and_mul_op[16-128-dtype0]` (M=16, N=128, **float32**).
  run.json `summary`: `correctness_passed=false`, `correctness_tests=11`,
  `correctness_failed=1`, `max_abs_err=7.72e-4` (atol=rtol=1e-5).
- benchmark: **未跑** (report skips benchmark because correctness did not pass).
  run.json `summary.benchmark_requested=false`, `benchmark_tests=0`,
  `msprof_case_count=0`, `profiler_fallback_count=0`, `invalid_profile_count=0`.

## 根因分析（归属：测试侧在设备上求参考实现 / 框架 gelu 近似）

失败**不是** kernel 缺陷。三路定量归因（同一 `x`，`examples/TileOPs` 下实测）：

| 对比 | max abs | 结论 |
|---|---:|---|
| kernel vs 精确 GELU（双精度 erf 参考）| **4.92e-7** | kernel 精度达标（fp32） |
| kernel vs CPU `F.gelu` | 9.54e-7 | 与精确 erf 一致 |
| kernel vs NPU `F.gelu` | **1.20e-3** | 与设备参考不符 |
| NPU `F.gelu` vs CPU `F.gelu` | 1.20e-3 | 设备参考本身偏差 |
| NPU `F.gelu` vs 精确 GELU | 1.20e-3 | 设备参考不精确 |
| CPU `F.gelu` vs 精确 GELU | 4.77e-7 | CPU 参考正确 |
| NPU `F.gelu(approximate="none")` vs CPU `F.gelu(approximate="tanh")` | **2.38e-7** | 设备 `none` 实际返回 tanh 近似 |

要点：

- `tests/ops/test_gelu_and_mul.py::GeluAndMulTest.ref_program` 在**设备张量**上执行
  `torch.nn.functional.gelu(gate)`（`tileops/testing/test_base.py:118` 用设备输入调用
  `ref_program`），即参考实现落在 NPU 上。
- 本机 `torch_npu 2.7.1.post4` 的 `F.gelu(approximate="none")` 结果与 tanh 近似一致
  （差异 2.4e-7），与精确 erf GELU 最大偏差 4.7e-4；乘以 `value` 后误差放大到 ~1.2e-3。
- fp32 用例容差为 `(1e-5, 1e-5)`（`test_gelu_and_mul.py:_get_tolerances`），无法吸收
  设备参考自身的 ~1.2e-3 偏差，故唯一的 fp32 用例必然失败。
- fp16/bf16 用例容差为 `1e-2` / `1.6e-2`，足以吸收该设备参考偏差，故全部通过。
- Stage-3 内嵌 L0/L1 全部通过（其 golden `golden_gelu_and_mul` 在 **CPU** 上算，见
  `gelu_and_mul_fwd_kernel.py:122`），与本次测试的设备侧参考形成对照，进一步证明 kernel 正确。

结论：kernel 设计/实现正确（对精确 GELU 达 4.9e-7）；失败根因在 **Stage-0 冻结测试的
设备侧参考实现** 与 **torch_npu `F.gelu` 近似行为**。按 integrator 失败分类表属
「测试/用例自身问题」（不修改、不消耗 attempt、如实上报）。

## 调试历史

- attempt（未消耗）: 症状 = fp32 用例 vs 设备 `F.gelu` 超差。
  分类 = 测试/参考（非 kernel、非编译/运行/shape、非 wrapper 胶水）。
  证据 = 上表三路归因（kernel 对精确 GELU 4.9e-7；设备 `F.gelu` 偏差 1.2e-3；设备 `none`≡tanh）。
  结论 = 无允许范围内的修复（kernel 已精确；wrapper 仅做 import/调用，不影响测试参考），
  交回 conductor 决策。

## 建议（供 conductor 路由）

1. 若以「数学正确 GELU」为准：修正 Stage-0 测试参考，使其在 host/CPU 上计算精确 GELU
   （例如复用 Stage-3 的 `golden_gelu_and_mul`），而非在 NPU 上调用 `F.gelu`。
2. 或为 fp32 用例调整容差以覆盖平台 `F.gelu` 的设备近似误差；但**不得**为过测试而弱化
   kernel 语义（kernel 现与精确 GELU 一致）。
3. 不改 kernel / wrapper：当前产物对精确 GELU 达 fp32 精度，属正确实现。

## integrate_kernel.py run (2026-10-13T21:36:41+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: gelu_and_mul_fwd_kernel
- smoke: pass

## integrate_kernel.py run (2026-10-13T21:54:09+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Developer
- functions: gelu_and_mul_fwd_kernel
- smoke: pass
