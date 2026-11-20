# Integration Log -- MoeGroupedGemmNopadFwdOp

- meta: examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/.migration_meta.json
- op_slug: moe_grouped_gemm_nopad | family: moe | attempts: 0 (no debug loop needed)
- scope: integration-only (report NOT run per task instruction)
- integrated:
  - `_moe_grouped_gemm_kernel` -> `moe_grouped_gemm_nopad_kernel/_moe_grouped_gemm_kernel.py` (ASCEND_MODE = "Expert")
  - `_tile_scheduler_kernel`  -> `moe_grouped_gemm_nopad_kernel/_tile_scheduler_kernel.py`  (ASCEND_MODE = "Developer")
- design_docs:
  - `_moe_grouped_gemm_kernel` -> `moe_grouped_gemm_nopad_kernel/_moe_grouped_gemm_kernel_DESIGN.md` (present)
  - `_tile_scheduler_kernel`  -> `moe_grouped_gemm_nopad_kernel/_tile_scheduler_kernel_DESIGN.md` (present)
  - missing: none
- mixed_modes: true (Expert + Developer)

## integrate_kernel.py 摘要

```
[mode] mixed programming modes across functions {'_moe_grouped_gemm_kernel': 'Expert', '_tile_scheduler_kernel': 'Developer'}; emitting per-function scoped re-exports
[wrapper] ... rewritten -> baseline/perf_opt kernel-source selection block
[mode] ... ascend_mode = Expert (updated)  # wrapper Kernel class defers to first function's mode; per-function scope wins at compile time
[smoke] pass
```
- `[json]` summary: wrapper_rewritten=true, mixed_modes=true, function_modes={Expert, Developer}, reexported=[_moe_grouped_gemm_kernel, _tile_scheduler_kernel], smoke=pass

## import 改写（before/after）

**before**（wrapper 第 46-49 行）:
```python
from ._moe_grouped_gemm_nopad_fwd_kernels import (
    _moe_grouped_gemm_kernel,
    _tile_scheduler_kernel,
)
```

**after**（baseline 激活；perf_opt 注释保留）:
```python
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_grouped_gemm_nopad_kernel import _moe_grouped_gemm_kernel, _tile_scheduler_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .moe_grouped_gemm_nopad_kernel.perf_opt._moe_grouped_gemm_kernel import _moe_grouped_gemm_kernel
# from .moe_grouped_gemm_nopad_kernel.perf_opt._tile_scheduler_kernel import _tile_scheduler_kernel
```
- 另插入 `ascend_mode = "Expert"`（wrapper `MoeGroupedGemmNopadKernel` 类体）。
- wrapper 既有 `msprof_kernel_name = "main_mix_aic"` **未改动**。

**mixed-mode scoped 重导出**（`moe_grouped_gemm_nopad_kernel/__init__.py`，脚本自动生成）:
```python
from tileops.utils import scoped_ascend_mode as _scoped_ascend_mode
from ._moe_grouped_gemm_kernel import _moe_grouped_gemm_kernel as __moe_grouped_gemm_kernel_impl
from ._tile_scheduler_kernel import _tile_scheduler_kernel as __tile_scheduler_kernel_impl
# _pin_ascend_mode 在每个函数的 factory 调用与返回 callable 调用两处 pin TILELANG_ASCEND_MODE
_moe_grouped_gemm_kernel = _pin_ascend_mode(__moe_grouped_gemm_kernel_impl, "Expert", "_moe_grouped_gemm_kernel")
_tile_scheduler_kernel  = _pin_ascend_mode(__tile_scheduler_kernel_impl,  "Developer", "_tile_scheduler_kernel")
```

## 已知集成脚本缺陷核查（"不过滤集成包里不存在的名"）

- 未复现：wrapper 从 `._moe_grouped_gemm_nopad_fwd_kernels` 导入的 2 个名
  (`_moe_grouped_gemm_kernel`, `_tile_scheduler_kernel`) 均在各集成模块顶层定义，
  生成的 baseline import 与 `__all__` 全部可用，无坏 import，故无需在 wrapper 胶水层修补。

## 验证结果

- report_run_json: not run (task scope = integration only)
- report_md: not run
- correctness: not run (per task instruction)
- benchmark: not run (per task instruction)
- smoke: pass
  - `tileops.kernels.moe.moe_grouped_gemm_nopad.moe_grouped_gemm_nopad_kernel.__all__` = [_moe_grouped_gemm_kernel, _tile_scheduler_kernel]; missing = []
  - `MoeGroupedGemmNopadKernel` importable; `msprof_kernel_name == "main_mix_aic"`; `ascend_mode == "Expert"`
  - re-verified 3x consecutively (10:2x) → pass each time.

## 调试历史

- none (integration + smoke passed on first run; no attempt consumed)

## issues

- **并发干扰（非本 op 缺陷，未修改）**：初次集成 smoke 通过后（10:20:56），10:21:29 复跑自己的 smoke 曾失败一次，traceback 指向**兄弟算子** `tileops.kernels.moe.moe_permute_nopad`：
  `ImportError: cannot import name '_make_scan_kernel_nopad_ep' from ...moe_permute_nopad_kernel`。
  根因：`tileops/kernels/moe/__init__.py` 会 eager import 所有兄弟算子，而兄弟算子正被另一并发写入者集成（其 wrapper/`__init__.py` 在 10:21:25 前后处于中间态）。
  该失败与本 op 集成产物无关；等待并发写入完成后连续 3 次复跑 smoke 全部通过。未触碰任何兄弟算子或公共文件。
