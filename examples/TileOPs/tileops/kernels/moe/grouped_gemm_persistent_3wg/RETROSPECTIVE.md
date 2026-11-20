# RETROSPECTIVE -- grouped_gemm_persistent_3wg

> Append-only log. One section per Stage execution. Do not edit history.

## Stage 5 (integrate) Retrospective — kernel-only / not-applicable — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 调度 prompt 断言本单元 `wrapper_path=null`，但 `.migration_meta.json` 实际为非空路径；适用性判定若沿用 prompt 断言会误判，必须以 meta 的 `scope`/`op_carrier`/`test_path` 为准 | `tileops/kernels/moe/grouped_gemm_persistent_3wg/.migration_meta.json`（`wrapper_path` 非空、`scope=kernel_only`、`test_path=null`）vs 本次 Stage-5 调度 prompt 的 `wrapper_path=null` 断言 | conductor 的 Stage-5 调度 prompt 改为从 meta 派生适用性前提（由 `scope`/`op_carrier` 决定），不硬编码 `wrapper_path=null` | R |
| Integrate-flow | `integrate_kernel.py` 无 "kernel-only / 无 op-carrier" 分支：无条件要求 wrapper 并重写其 import；对 `scope=kernel_only` 单元会把 scaffolded Kernel 类接到运行期契约不匹配的 Stage-3 工厂上，产出一个 smoke 绿、运行期不可调用的伪集成 | 重写逻辑 `integrate_kernel.py#L367-L391`；smoke 仅 `hasattr` 校验 `integrate_kernel.py#L394-L418`；wrapper 调用点 `grouped_gemm_persistent_3wg.py#L146-L213`（`(numel,num_experts,N,K,dtype_str,sm_count,block_k)` + `(block_m,block_n,block_k,num_stages,group_size_m)`）vs Stage-3 工厂 `_persistent_grouped_gemm_v2_kernel.py#L174`（`(numel,max_tiles,num_experts,N,K,dtype,block_m,block_n,block_k,num_stages)`，prim_func 另需 3 张 GM 调度表） | `integrate_kernel.py` 增加 `--kernel-dir-only` / 依 `meta.scope=="kernel_only"` 跳过 wrapper 重写；wrapper 重写前做工厂签名契约校验 | R |
| Integrate-flow | family 级 kernel 包 import smoke 会先执行父包 `tileops/kernels/moe/__init__.py` 的聚合导入；并发集成的兄弟单元处于中间态时，本单元 smoke 会因无关的 `ImportError` 失败（本次实测：10:21:36 失败 → 10:22:38 兄弟单元补齐后通过） | 本次执行日志：`ImportError: cannot import name '_make_scan_kernel_nopad_ep' from '...moe_permute_nopad.moe_permute_nopad_kernel'`（父包 `tileops/kernels/moe/__init__.py#L7`）；兄弟单元 `moe_permute_nopad_kernel/__init__.py` 含 `# [warn] _make_scan_kernel_nopad_ep: not found in integrated modules` | Stage-5 集成 skill 增加"隔离 smoke"指引（绕过父包聚合，逐子包 import）并提示并发单元噪声 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| kernel-only 迁移单元（`scope=kernel_only` / `op_carrier=none`）不做 wrapper 重写：只落 Stage-3 内核 + DESIGN 快照 + `_kernel/__init__.py`；import-only 重写会因 NPU 工厂改变运行期输入契约而产生伪集成 | P | `tileops/kernels/moe/grouped_gemm_persistent_3wg/.migration_meta.json`；`grouped_gemm_persistent_3wg.py#L146-L213`；`.../grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py#L174`；对照同族可重写单元 `moe_grouped_gemm_nopad/moe_grouped_gemm_nopad.py#L58,L182-L196`（Stage-3 保留了工厂签名） | `cd examples/TileOPs && python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/moe/grouped_gemm_persistent_3wg/.migration_meta.json --dry-run`（观察其重写目标为 stale wrapper） | tilelang 0.1.2+8875071d（HEAD 8875071d, tvm c2921fdaf, AscendNPU-IR 31f690369）；Ascend910；CANN 9.0.0；torch_npu 2.7.1.post4 / py3.11.4 | pattern-library |
| Kernel 包 import smoke 会触发 family `__init__.py` 聚合；并发集成兄弟单元的中间态可导致与本单元无关的瞬时 `ImportError`，需重跑或隔离后再判定 | C | 本次执行日志（10:21:36 失败 / 10:22:38 通过）；`tileops/kernels/moe/__init__.py#L7` | `export ASCEND_RT_VISIBLE_DEVICES=1 && cd examples/TileOPs && python -c "import tileops.kernels.moe.grouped_gemm_persistent_3wg"` | 同上 | pattern-library / bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- migration-harness 的 **kernel-only 单元**（meta `scope=kernel_only` / `op_carrier=none`）无 manifest op / test / bench，Stage 5 的单算子 report 门禁不存在；集成仅限于把 Stage-3 内核文件 + `{func}_DESIGN.md` 快照落到 `{op_slug}_kernel/` 并由 `gen_init_py` 生成 `__init__.py` 重导出。
- 适用性判定一律以 `.migration_meta.json` 为准（`scope` / `op_carrier` / `test_path` / `wrapper_path`），不要沿用调度 prompt 中可能过时或模板化的断言。
- 在把 scaffolded Kernel 类接到 Stage-3 工厂前先比对**运行期调用契约**（工厂参数列表 + 返回可调用签名 + `prim_func` 的运行时张量入参）。本单元 GPU 脚手架（`(numel,num_experts,N,K,dtype,sm_count,block_k)`）与 NPU 工厂（多出 `max_tiles`，且需 `tile_expert_ids`/`tile_row_offsets`/`total_tiles` 三张调度表）不一致——仅重写 import 会让只做 `hasattr` 的 smoke 变绿，属于伪集成。
- 同族 `moe_grouped_gemm_nopad` 可直接重写 wrapper，是因为其 Stage-3 内核**保留了原工厂签名**；本单元 NPU 内核改动了运行期输入契约，因此其 wrapper 属显式 stale 脚手架，不能机械套用同族做法。
- Kernel 包 import smoke 会执行父包 `<family>/__init__.py` 的聚合导入。并发集成场景下，兄弟单元的中间态会污染本单元 smoke 结论；遇 `ImportError` 先确认失败符号是否属于本单元，必要时重跑或做隔离 import，再下判定。
