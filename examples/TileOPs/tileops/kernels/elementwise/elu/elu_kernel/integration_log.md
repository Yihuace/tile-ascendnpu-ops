# Integration Log -- EluFwdOp

- meta: `examples/TileOPs/tileops/kernels/elementwise/elu/.migration_meta.json`
- op: EluFwdOp (op_slug=elu, family=elementwise)
- integrated: `_make_elu_kernel` -> `tileops/kernels/elementwise/elu/elu_kernel/_make_elu_kernel.py`
- design_docs: `_make_elu_kernel` -> `tileops/kernels/elementwise/elu/elu_kernel/_make_elu_kernel_DESIGN.md`
- wrapper: `tileops/kernels/elementwise/elu/elu.py` (import rewritten to baseline/perf_opt selection block; `ascend_mode="Developer"` synced)
- ascend_mode: Developer (read from final Stage 3 kernel `ASCEND_MODE` literal; wrapper unchanged = already Developer)
- attempts: 1 (no debug fixes needed)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261013_043305_313351_EluFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261013_043305_313351_EluFwdOp/report.md`
- report_status: passed
- correctness: pass (8/8 cases, max_abs_err = 0.00e+00)
  - test_elu[1048576-dtype0/1/2] (fp32/fp16/bf16)
  - test_elu[4000000-dtype3/4]
  - test_elu_rejects_non_float_dtype
  - test_elu_eval_roofline
  - test_elu_edge[4096-dtype0]
- benchmark: pass (10/10, msprof valid, profiler_fallback_count=0, invalid_profile_count=0)
  - smoke-1m float32 18.2204us (12.79%) / float16 9.3402us (16.28%) / bfloat16 9.3602us (18.89%)
  - mlp-hidden float16 54.7211us (21.93%) / bfloat16 54.8211us (25.51%)
  - mlp-hidden-wide float16 108.2422us (22.15%) / bfloat16 108.8622us (25.67%)
  - elementwise-16m float16 106.7221us (22.47%) / bfloat16 107.4821us (26.00%) / float32 256.1851us (15.69%)
  - Ratio range 12.79% – 26.00%, bandwidth 1.80 TB/s (BW-bound roofline)

## 集成备注

- 集成脚本首跑即 `smoke=pass`，wrapper `ascend_mode=Developer` 与最终 kernel 声明一致（`unchanged`）。
- wrapper custom_op 调用点实参为 `(x, N)`（= 2），与 kernel `prim_func main(x, y, shape)` 的 `total_params(3) - len(out_idx)(1) = 2` 一致，未触发已知的 `out_idx + 动态 shape` 调用实参缺参缺陷。
- 集成包内 wrapper 顶部存在重复的 kernel-source 选择注释块（scaffolder 原注释 + integrate 生成块），仅注释冗余、不影响行为；见 op 级 RETROSPECTIVE.md 的 Integrate-flow 记录。

## 调试历史

- 无（attempt 1 直接通过全量正确性与 benchmark）。
