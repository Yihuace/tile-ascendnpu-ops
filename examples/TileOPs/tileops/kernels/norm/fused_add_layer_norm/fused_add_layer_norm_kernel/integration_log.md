# Integration Log -- FusedAddLayerNormFwdOp

- meta: examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/.migration_meta.json
- integrated: _fused_add_layer_norm_kernel -> fused_add_layer_norm_kernel/_fused_add_layer_norm_kernel.py
- design_docs: _fused_add_layer_norm_kernel -> fused_add_layer_norm_kernel/_fused_add_layer_norm_kernel_DESIGN.md
- attempts: 1

## integrate_kernel.py run (2026-10-14T11:53:46+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: _fused_add_layer_norm_kernel
- smoke: pass

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261014_115534_389335_FusedAddLayerNormFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_115534_389335_FusedAddLayerNormFwdOp/report.md
- correctness: pass (14/14 cases; 0 failed)
- benchmark: pass (10/10 cases; 7 msprof-profiled, profiler_fallback_count=0, invalid_profile_count=0)
- summary: correctness_passed=true, benchmark_passed=true, case_count=7, msprof_case_count=7
- max_abs_err: 1.56e-02 (bfloat16 2048x4096, tolerance 1.6e-2); avg_max_abs_err=3.29e-03

## 调试历史

- attempt 1:
  - 症状: 集成后运行时 TypeError: `_fused_add_layer_norm_kernel.<locals>._func() takes from 0 to 1 positional arguments but 4 were given`。
  - 分类: wrapper 胶水错误（参数传递）。
  - 根因: 迁移后 NPU 工厂契约是 `_fused_add_layer_norm_kernel(M,N,eps,dtype) -> _func(block_m=default)`，需再调用一次选择 block_m 才得到编译后的 `main`；wrapper `forward` 直接以 4 个张量调用工厂返回值（缺省 block_m 选择层），且漏接 `()(tensors)` 形态（GPU 原签名 `_func(block_m, threads)` 被移植后未随 NPU 契约更新）。
  - 修复: 仅改 wrapper 胶水 `fused_add_layer_norm.py` 的 `forward`，把 `...)(rows, residual, weight, bias)` 改为 `...)()(rows, residual, weight, bias)`，使用工厂自身默认 block_m（K11：不烘焙执行参数）。Op 层/两输出处理（`(y, residual_out)` reshape 回原 shape）本就正确，未改动。
  - 备份: history_version/fused_add_layer_norm_s5_attempt1.py
  - 结果: 修复后小形状冒烟（fp32/fp16/bf16，双输出 max_err ≤4.8e-7/0）通过；完整单算子 report 全量正确性 14/14 + benchmark 10/10 通过。

## 护栏（§2.1 三护栏）

- guardrail 1 (wrapper_rewritten == true): PASS
- guardrail 2 (grep "already integrated"): PASS（未命中）
- guardrail 3 (_fused_add_layer_norm_kernel.__module__ 指向 baseline): PASS -> tileops.kernels.norm.fused_add_layer_norm.fused_add_layer_norm_kernel._fused_add_layer_norm_kernel
