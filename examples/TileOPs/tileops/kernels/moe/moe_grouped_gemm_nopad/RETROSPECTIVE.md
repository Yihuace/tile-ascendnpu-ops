# RETROSPECTIVE — moe_grouped_gemm_nopad

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | integrate_kernel.py 的 wrapper import 改写不做"集成包内是否存在该名"的过滤（元数据/流程已知缺陷）；本次因 wrapper 导入的名恰好在集成模块顶层都定义而未触发，但缺陷本身仍在 | `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` `rewrite_wrapper_import()` (L367-391) 直接用 `parse_wrapper_imports` 返回的全部名生成 baseline import，无 `defining_module` 判空过滤 | 在 `rewrite_wrapper_import` 里对 `names` 调 `defining_module(name, integrated_files)` 过滤或告警未命中项 | R |
| Integrate-flow | mixed-mode 的 `__init__.py` 私有别名对 `_` 前缀名生成 `__name_impl` 双下划线，可读性差但功能正确 | `moe_grouped_gemm_nopad_kernel/__init__.py` L9-10 | 改用无歧义别名（如 `_impl_<name>`） | R |
| Integrate-flow | 并行集成多个同 family 算子时，`tileops/kernels/{family}/__init__.py` 会 eager import 所有兄弟算子，任一兄弟处于集成中间态（坏 import）会让本 op 的 import smoke 假失败 | 本任务 10:21:29 复跑 smoke 失败 traceback 指向兄弟 `moe_permute_nopad`（`ImportError: _make_scan_kernel_nopad_ep`），等待其完成后再跑 3 次全过 | 允许 smoke 只导入本 op 叶子包（延迟/绕过父 package eager import），或在文档中说明需串行集成同 family | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| mixed Expert+Developer 双模式算子可被 integrate_kernel.py 原生集成（per-function scoped re-export，无需手改 wrapper） | P | `moe_grouped_gemm_nopad_kernel/__init__.py` L7-33；integrate_kernel.py `gen_init_py()` mixed 分支 L265-313；日志 `[mode] mixed programming modes ... emitting per-function scoped re-exports` | `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta <meta>` | tilelang dev root /mnt/workspace/tilelang-mlir-ascend/build；device 1；CANN 版本未记录（本次未跑 report） | pattern-library |
| 混合模式下 wrapper Kernel.ascend_mode 取首个函数模式安全：per-function scoped 包装在 factory+callable 两个编译点 pin 模式，类级属性被 defer | P | wrapper `ascend_mode="Expert"`（由 functions[0] `_moe_grouped_gemm_kernel` 决定）；integrate_kernel.py L562；`_pin_ascend_mode` 在 factory 与返回 callable 两处 `scoped_ascend_mode` | 同上 + import smoke | 同上 | pattern-library |

### Transferable Lessons（可迁移教训）

- 一个 op 的多个 kernel 函数混用 Expert/Developer 时，直接跑 `integrate_kernel.py` 即可：脚本会读每个函数源文件的字面量 `ASCEND_MODE`，生成 per-function scoped re-export 的 `__init__.py`，并给 wrapper Kernel 类填首个函数的模式（运行时 per-function scope 覆盖类级属性）。不要手动给类填单一模式，也不要给整个类强行统一模式。
- 集成后先跑 import smoke 确认 `moe_grouped_gemm_nopad_kernel.__all__` 全部可 import；wrapper 从 `._*_fwd_kernels` 导入的名必须都能在各集成模块顶层找到，否则会生成坏 import（integrate_kernel.py 不过滤未命中名）。
- 集成期不要改动 wrapper 上测试/集成层需要的既有属性（本 op 的 `msprof_kernel_name = "main_mix_aic"` 是 msprof kernel-name 过滤所必需）。
