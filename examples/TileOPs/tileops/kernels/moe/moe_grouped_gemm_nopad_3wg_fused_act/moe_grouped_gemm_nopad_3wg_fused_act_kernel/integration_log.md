# Integration Log -- MoeGroupedGemmNopad3WGFusedActFwdOp

- meta: `examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad_3wg_fused_act/.migration_meta.json`
- op: `MoeGroupedGemmNopad3WGFusedActFwdOp` (op_slug=`moe_grouped_gemm_nopad_3wg_fused_act`, family=`moe`)
- scope: **integration only** -- report/benchmark intentionally NOT run (per task instruction)
- device: `ASCEND_RT_VISIBLE_DEVICES=1`
- integrated: `_fused_act_kernel` -> `moe_grouped_gemm_nopad_3wg_fused_act_kernel/_fused_act_kernel.py`
- design_docs: `_fused_act_kernel` -> `_fused_act_kernel_DESIGN.md` (byte-identical to `examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/DESIGN.md`)
- kernel source: `examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/_fused_act_kernel.py` (byte-identical copy)
- wrapper: `tileops/kernels/moe/moe_grouped_gemm_nopad_3wg_fused_act/moe_grouped_gemm_nopad_3wg_fused_act.py`
- attempts: 1 (one integration pass + one glue fix; 2 script invocations total)

## 集成结果

`integrate_kernel.py` 摘要（最后一次成功运行，脚本 stdout `[json]` 行）：

```json
{"functions": ["_fused_act_kernel"], "reexported": ["_gemm_gate_up", "_act_packed"],
 "ascend_mode": "Expert", "function_modes": {"_fused_act_kernel": "Expert"},
 "mixed_modes": false, "wrapper_rewritten": true, "wrapper_mode_updated": true,
 "smoke": "pass"}
```

- 编程模式：Stage 3 产物 `ASCEND_MODE = "Expert"`（`_fused_act_kernel.py` L89），脚本同步 `Kernel.ascend_mode = "Expert"`。GEMM 核为 Expert 单 Cube scope；Vector 激活核在同一 Expert 进程模式下编译（RETROSPECTIVE.md 实证 D）。
- 聚合 `__init__.py`：
  ```python
  from ._fused_act_kernel import _gemm_gate_up
  from ._fused_act_kernel import _act_packed
  __all__ = ["_gemm_gate_up", "_act_packed"]
  ```

## Wrapper import 改写

**改写前（Stage-0 scaffold，占位工厂名）**：
```python
from ._moe_grouped_gemm_nopad3_w_g_fused_act_fwd_kernels import (
    _fused_act_kernel,
)
```

**首次脚本运行生成（坏 import，已知缺陷复现）**：
```python
from .moe_grouped_gemm_nopad_3wg_fused_act_kernel import _fused_act_kernel
```
脚本未过滤"集成包里不存在的名"：Stage-3 产物实际导出 `_gemm_gate_up` / `_act_packed`，
不导出 `_fused_act_kernel`；生成的 `__init__.py` 随之带 `# [warn] _fused_act_kernel: not found in integrated modules`。

**最终（glue 修复后，canonical baseline/perf_opt 切换块，baseline 激活）**：
```python
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_grouped_gemm_nopad_3wg_fused_act_kernel import _gemm_gate_up, _act_packed
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .moe_grouped_gemm_nopad_3wg_fused_act_kernel.perf_opt._fused_act_kernel import _gemm_gate_up
# from .moe_grouped_gemm_nopad_3wg_fused_act_kernel.perf_opt._fused_act_kernel import _act_packed
```

修复手法（**仅改 wrapper**，未改 `integrate_kernel.py`，未改 `__init__.py`）：
把 wrapper 的 import glue 改为 Stage-3 产物的真实导出名后重跑脚本 —— `integrate_kernel.py`
从该 import 语句提取名字用于生成 block 与 `__init__.py`，因此一次运行即同时产出正确的
baseline/perf_opt 块与正确的聚合 `__init__.py`（`[warn]` 消失）。
同时把 wrapper 内两处 `_fused_act_kernel(...)` 调用点（`__init__` 的 `self.kernel` lambda 与
`forward`）改写为按 `run_case` 已验证路径串联调用 `_gemm_gate_up` + `_act_packed`。

## import smoke 结果

脚本自带 smoke：`pass`（`reexported = ["_gemm_gate_up", "_act_packed"]`）。

**注意：脚本 smoke 存在假阳性风险（本任务已复现）**：首次（坏 import）运行脚本仍报 `smoke: pass`，
因为 `from .pkg import _fused_act_kernel` 在名字缺失时回落到导入**同名子模块**（文件 stem == 名字），
`hasattr(pkg, "_fused_act_kernel")` 为 True，但该对象是 module：
`TypeError: 'module' object is not callable`（复现见下）。

附加（人工）smoke 验证最终状态：

```
pkg __all__       : ['_gemm_gate_up', '_act_packed']
missing exports   : []
Kernel class      : True
_gemm_gate_up type: function  callable= True
_act_packed  type : function  callable= True
ascend_mode       : Expert
op.kernel         : MoeGroupedGemmPersistent3WGFusedActKernel   (Op 构造成功)
SMOKE-OK
```

## 补充正确性证据（非 report）

- `_build_tile_schedule`（wrapper 内 host 侧调度表）与 Stage-3 `_fused_act_kernel.build_schedule`
  在全部 test/benchmark shape 上逐元素一致（ids/rows/total 全等），且按 `max_tiles` 补 `(-1, 0)`
  sentinel（GEMM 核静态张量形状要求）：
  `64x4x128x64 (mt=4,total=4)` / `256x8x256x128 (mt=10,total=8)` /
  `4096x256x2048x7168 (mt=288,total=256)` / `32768x256x2048x7168 (mt=512,total=256)` /
  `4096x8x768x2048 (mt=40,total=32)` / `32768x8x768x2048 (mt=264,total=256)`。
- **未运行 forward/report/benchmark**（按任务要求）；因此端到端数值正确性与性能均未在本步验证。

## issues / 风险（交回 conductor，不修复）

1. **集成脚本缺陷（已按约束在 wrapper 内规避）**：`integrate_kernel.py` 的 wrapper import 改写不过滤
   "集成包里不存在的名"。根因是 `.migration_meta.json` 的 `extracted_functions=["_fused_act_kernel"]`
   与 Stage-3 实际导出（`_gemm_gate_up`/`_act_packed`）不一致，而脚本以此推断名字。
2. **脚本幂等性缺口（重要）**：当前 wrapper 的 import 已指向 `.{op_slug}_kernel`，脚本再次运行会命中
   "already integrated" 分支，但 `__init__.py` 仍会用 `parse_wrapper_imports(...) or functions` 回落到
   `["_fused_act_kernel"]` 重新生成 → 覆盖为坏 `__init__.py` 并使 smoke 因 ImportError 失败。
   **建议 conductor 不要重复运行 `integrate_kernel.py`**（如需重跑，附加
   `--functions _gemm_gate_up _act_packed`）。
3. **msprof 单算子解析缺口**：本 op 每次 `forward` 启动两个 device kernel（GEMM + Vector），
   msprof 无 kernel-name filter 时无法解析单一算子、Roofline Ratio 会丢失（与胞兄
   `moe_grouped_gemm_nopad` 同类问题）。需在 report/bench 阶段实测 device kernel 名后，为
   Kernel 类补 `msprof_kernel_name`（本步未猜写，避免写错名）。
4. **flaky 编译（CG-2026-0033）**：Stage-3 harness 用 `_run_kernel`（重试预算 20）规避
   bishengir-compile 间歇 SIGSEGV；wrapper `forward` 未内建该重试。本 op 的 test/manifest shape
   的 ffn 均可被 block_n=128 整除，且多数 shape `num_local_tasks==1`（低风险），但 DeepSeek
   prefill 类大 shape（`num_pid_n>1` 且 `num_local_tasks>1`）仍可能触发间歇崩溃 → report/bench
   阶段若出现 `NPU IR opt failed` / SIGSEGV，应视为该已知工具链问题而非数值错误。
5. **host 同步**：wrapper 每调用一次 `true_sizes` 需 `.cpu()` 取回以构造调度表（Stage-3 尚无
   on-device `_tile_scheduler_kernel`）。胞兄落地上位后应替换为设备侧调度（DESIGN §4.1）。

## 调试历史

- attempt 1：症状 = 首次脚本运行生成坏 import（`_fused_act_kernel` 非集成包导出；`__init__.py` 带
  `[warn]`；脚本 smoke 假阳性 pass）。分类 = wrapper 胶水错误 / 集成脚本缺陷。修复 = 仅改 wrapper
  的 import glue 为真实导出名 `_gemm_gate_up`/`_act_packed`，并同步 `__init__`/`forward` 两处调用点
  到两个 Stage-3 kernel；重跑脚本一次 → 生成 canonical block + 正确 `__init__.py`。结果 = smoke pass，
  导出名齐全且可调用，Op 可构造。
