# Integration Log -- MoePermuteNopadFwdOp

- meta: `examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/.migration_meta.json`
- op: `MoePermuteNopadFwdOp` (op_slug=`moe_permute_nopad`, family=`moe`)
- scope: **integration-only re-run** (per dispatch: integrate 3 functions + rewire wrapper EP
  path + import smoke; report NOT run)
- integrated (3 functions -> files):
  - `_make_gather_kernel_nopad` -> `tileops/kernels/moe/moe_permute_nopad/moe_permute_nopad_kernel/_make_gather_kernel_nopad.py`
  - `_make_scan_kernel_nopad` -> `tileops/kernels/moe/moe_permute_nopad/moe_permute_nopad_kernel/_make_scan_kernel_nopad.py`
  - `_make_scan_kernel_nopad_ep` -> `tileops/kernels/moe/moe_permute_nopad/moe_permute_nopad_kernel/_make_scan_kernel_nopad_ep.py`  **(NEW this run)**
- design_docs (Stage 1 snapshot -> `{func}_DESIGN.md`) -- all 3 present, byte-identical to source:
  - `_make_gather_kernel_nopad` -> `_make_gather_kernel_nopad_DESIGN.md`
  - `_make_scan_kernel_nopad` -> `_make_scan_kernel_nopad_DESIGN.md`
  - `_make_scan_kernel_nopad_ep` -> `_make_scan_kernel_nopad_ep_DESIGN.md`  **(NEW this run)**
- attempts: 1 integration + 1 manual wrapper-glue edit (EP rewire; script guard does not
  rewrite the wrapper on re-run)
- ascend_mode: `Developer` (all 3 functions declare `ASCEND_MODE = "Developer"`; no mixed modes)

## 集成命令

```
cd examples/TileOPs
python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/moe/moe_permute_nopad/.migration_meta.json \
  --functions _make_gather_kernel_nopad _make_scan_kernel_nopad _make_scan_kernel_nopad_ep
```

Result: `[ok] integration complete`; script smoke `pass`; `wrapper_rewritten=false`
(`rewrite_wrapper_import` short-circuits because the wrapper already contains a
`from .moe_permute_nopad_kernel import` block), `mixed_modes=false`, `wrapper_mode_updated=false`.

Note: previously `--functions` was required to **exclude** `_make_scan_kernel_nopad_ep`
(no conductor product). Its Stage-3 product now exists at
`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/`, so all 3 `extracted_functions`
from the meta are integrated and `--functions` is now a plain enumeration.

## 验证结果

- report_run_json: **not produced** (report intentionally not run; dispatch scope = integration only)
- report_md: not produced
- correctness: not run (out of scope)
- benchmark: not run (out of scope)
- import smoke: **pass** (see below)

## Wrapper import 改写

### 改写前 (this run, pre-edit)

```python
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_permute_nopad_kernel import (
    _make_gather_kernel_nopad,
    _make_scan_kernel_nopad,
)
from .moe_permute_nopad_kernel._make_scan_kernel_nopad import (
    VECTOR_CORES as _SCAN_VECTOR_CORES,
)
# [S5 glue] `_make_scan_kernel_nopad_ep` (the EP / expert_map scan variant) was
# NOT imported here any more -- the EP path is rejected in ``__init__`` ...
```

(The `integrate_kernel.py` script did **not** touch this block: see "已知缺陷复现" below.)

### 改写后 (this run, final)

```python
# --- baseline (Stage 3) ------------------------------------------------------
from .moe_permute_nopad_kernel import (
    _make_gather_kernel_nopad,
    _make_scan_kernel_nopad,
    _make_scan_kernel_nopad_ep,
)
from .moe_permute_nopad_kernel._make_scan_kernel_nopad import (
    VECTOR_CORES as _SCAN_VECTOR_CORES,
)
from .moe_permute_nopad_kernel._make_scan_kernel_nopad_ep import (
    VECTOR_CORES as _SCAN_EP_VECTOR_CORES,
)
```

## 集成包 `__init__.py` 改写

- 脚本无条件重写了 `moe_permute_nopad_kernel/__init__.py`。因 3 个函数均已迁移，本次生成物
  **干净**（不再需要 Stage-0 fallback 胶水）：

```python
from ._make_gather_kernel_nopad import _make_gather_kernel_nopad
from ._make_scan_kernel_nopad import _make_scan_kernel_nopad
from ._make_scan_kernel_nopad_ep import _make_scan_kernel_nopad_ep

__all__ = [
    "_make_gather_kernel_nopad",
    "_make_scan_kernel_nopad",
    "_make_scan_kernel_nopad_ep",
]
```

- 上一 attempt 的手工胶水（`from .._moe_permute_nopad_fwd_kernels import
  _make_scan_kernel_nopad_ep`）随本次重跑被覆盖，**且不再需要**（见下）。

## EP 路径重接（本次任务核心）

对照已修好的非 EP 路径（`_make_scan_kernel_nopad(numel, num_experts, top_k)` ->
`run(flat_ids, bh, true_offsets, true_sizes, efo, slot_base, run_buf, fwd_idx, permuted_idx)`），
NPU EP 工厂的真实签名是：

- 工厂：`_make_scan_kernel_nopad_ep(numel: int, e_global: int, e_local: int, top_k: int)`
- run 可调用：`run(flat_ids, expert_map, bh, true_offsets, true_sizes, efo, slot_base, run_buf, fwd_idx, permuted_idx)`
  （比非 EP 多一个 `expert_map`，位于第 2 个位置参数）

改动（仅 wrapper 胶水层 `moe_permute_nopad.py`）：

1. **import**：baseline block 从集成包导入新增 `_make_scan_kernel_nopad_ep`（不再指向
   Stage-0 GPU 模块 `._moe_permute_nopad_fwd_kernels`）；并按 EP 模块绑定其自身的
   `VECTOR_CORES` -> `_SCAN_EP_VECTOR_CORES`（scratch NB 必须与工厂 grid 一致）。
2. **`__init__` EP 分支**：删除 `raise NotImplementedError(...)`，改为
   `self._scan_fn = _make_scan_kernel_nopad_ep(self.numel, num_experts, self.num_experts_local, top_k)`
   （`num_experts` = e_global，`self.num_experts_local` = e_local）。
3. **`forward`**：按 `self.expert_map is not None` 分流——EP 分支用 `_SCAN_EP_VECTOR_CORES`
   定 scratch NB，并将 `expert_map = self.expert_map.flatten().contiguous()` 作为 run 的
   第 2 个位置参数传入；非 EP 分支保持原调用不变。
4. **module docstring**：更新为「3 个工厂均为已迁移的 NPU 产物」+ 说明两条 scan 分派均已
   绑定集成包工厂、forward 按 `VECTOR_CORES` 定 scratch、EP 变体走 `expert_map`。

`MoePermuteNopadFwdOp.forward`（Op 层，未改）在 `_get_kernel` 中把 `self.expert_map` 传给
kernel 构造函数（`tileops/ops/moe/moe_permute_nopad.py:127`），因此 EP 路径从 Op 层可达且
不再抛 `NotImplementedError`。

## 已知缺陷复现（integrate_kernel.py）

**症状（上一 attempt，本次未复现）**：当 wrapper 引用了集成包内不存在的导出名时，脚本
`gen_init_py` / `gen_kernel_source_block` 不过滤该名，生成坏 import，smoke
`ImportError: cannot import name '_make_scan_kernel_nopad_ep'`。

**本次状态**：3 个函数全部迁移并集成，生成物无缺失名，smoke 直接通过。**上一 attempt 的
手工胶水（绑定到 Stage-0 模块）不再需要且已被覆盖**。缺陷本身仍存在于脚本中（若将来出现
新的未迁移名会再次触发），根治仍须由 conductor 在 `integrate_kernel.py` 侧过滤。

**另一处脚本行为（本次观察到）**：`rewrite_wrapper_import` 用
`re.search(r"from \.moe_permute_nopad_kernel(\.\w+)?\s+import", text)` 做短路——wrapper 已有
该 import 时直接返回 `already integrated`，**不会**把新集成的函数名补进 wrapper 的 baseline
block。因此「增量重集成一个函数」时 wrapper import 必须人工补齐（本次已人工补
`_make_scan_kernel_nopad_ep` 及其 VECTOR_CORES）。

## Smoke 结果

```
python - <<'PY'   # cd examples/TileOPs; ASCEND_RT_VISIBLE_DEVICES=1
import tileops.kernels.moe.moe_permute_nopad.moe_permute_nopad_kernel as pkg
assert set(pkg.__all__) == {"_make_gather_kernel_nopad","_make_scan_kernel_nopad","_make_scan_kernel_nopad_ep"}
import tileops.kernels.moe.moe_permute_nopad.moe_permute_nopad as w
assert hasattr(w, "MoePermuteNopadKernel")
assert w.MoePermuteNopadKernel.ascend_mode == "Developer"
# EP rewire: must resolve to the integrated NPU module, not the Stage-0 GPU extract
assert pkg._make_scan_kernel_nopad_ep.__module__.endswith("moe_permute_nopad_kernel._make_scan_kernel_nopad_ep")
assert w._make_scan_kernel_nopad_ep is pkg._make_scan_kernel_nopad_ep
assert w._SCAN_EP_VECTOR_CORES == 40 and w._SCAN_VECTOR_CORES == 40
assert "NotImplementedError" not in inspect.getsource(w.MoePermuteNopadKernel.__init__)
PY
```

- `smoke-ok ['_make_gather_kernel_nopad', '_make_scan_kernel_nopad', '_make_scan_kernel_nopad_ep']`
- `EP module: tileops.kernels.moe.moe_permute_nopad.moe_permute_nopad_kernel._make_scan_kernel_nopad_ep`
- `EP factory signature: (numel: int, e_global: int, e_local: int, top_k: int)`
- `MoePermuteNopadKernel.ascend_mode = "Developer"`
- `py_compile` OK for wrapper + EP kernel.

## 备份 (history_version/)

- `moe_permute_nopad_wrapper_pre_s5rein.py` (wrapper, before this re-integration)
- `pkg_kernel_init_pre_s5rein.py` (`moe_permute_nopad_kernel/__init__.py`, before script rerun)
- `pkg_init_pre_s5rein.py` (op-level `__init__.py`, before script rerun)
- (prior attempt backups retained: `moe_permute_nopad_wrapper_pre_s5.py`,
  `pkg_kernel_init_post_integrate_s5.py`, `moe_permute_nopad_wrapper_post_integrate_s5.py`,
  `moe_permute_nopad_s5_attempt1.py`, `pkg_init_pre_s5.py`)

## Issues

- `integration_report.json` not written: no `run.json` exists (report intentionally not run) —
  writing a path would fabricate a result.
- `integrate_kernel.py`: (a) import-name filter defect persists for any future unmigrated name;
  (b) `rewrite_wrapper_import` short-circuit means incremental re-integration does not add the
  new function to the wrapper baseline block (worked around manually in op glue this run).
- EP scan path is now wired but **not executed** (report/benchmark out of scope); runtime
  correctness of the EP variant on NPU remains to be validated by a future report run.
