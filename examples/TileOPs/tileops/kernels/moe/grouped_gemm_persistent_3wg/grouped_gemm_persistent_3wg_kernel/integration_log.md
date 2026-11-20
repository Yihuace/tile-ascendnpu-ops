# Integration Log -- GroupedGemmPersistent3WGKernel

- meta: `tileops/kernels/moe/grouped_gemm_persistent_3wg/.migration_meta.json` (`scope=kernel_only`, `op_carrier=none`)
- integrated:
  - `_persistent_grouped_gemm_v2_kernel` -> `grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py`
- design_docs:
  - `_persistent_grouped_gemm_v2_kernel` -> `grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel_DESIGN.md`
- attempts: 0 (no verification/debug cycle possible; kernel-only unit, report not run)

## 适用性判定（Stage 5）

| 判据 | 结果 |
|------|------|
| manifest op 条目 | 无（`op_class_name=null`，非独立 op/wrapper） |
| test / bench | `test_path=null`, `bench_path=null` |
| `source.kernel_map` 引用 | 由 3 个 MoE fusion op 以键 `moe_grouped_gemm_kernel` 引用 `GroupedGemmPersistent3WGKernel`（见 `fused_moe_experts_nopad_persistent_3wg/.migration_meta.json#composed_kernel_map`、`fused_moe/.migration_meta.json#composed_kernel_map`） |
| `integrate_kernel.py` | **不适用（harness 语义）** —— 见下 |

`integrate_kernel.py --meta ... --dry-run` 机械上可运行（已实测：会 copy kernel + DESIGN 并生成 `_kernel/__init__.py`），但其 wrapper 重写步骤不适配本单元：

1. 本单元无 manifest op / test / bench，Stage 5 的核心门禁（单算子 report）不存在。
2. meta 的 `wrapper_path` 指向 scaffolded Kernel 类文件 `grouped_gemm_persistent_3wg.py`，其调用契约是 **Pre-Stage-3 的 GPU 形状**：
   - wrapper 调用 `_persistent_grouped_gemm_v2_kernel(numel, num_experts, N, K, dtype_str, sm_count, block_k)` 并再以 `(block_m, block_n, block_k, num_stages, group_size_m)` 调用返回值；
   - Stage-3 NPU 工厂签名是 `_persistent_grouped_gemm_v2_kernel(numel, max_tiles, num_experts, N, K, dtype, block_m, block_n, block_k, num_stages)`，其 `prim_func` 额外要求 3 张 GM 调度表（`tile_expert_ids` / `tile_row_offsets` / `total_tiles`）。
   - 仅重写 import 会让 `integrate_kernel.py` 的 smoke（只做 `hasattr` 检查）变绿，但类在运行期不可调用 —— 属伪集成。要真正接通还需新增 host/producer 调度表构造代码，超出 Stage-5 胶水范围且无从验证（无 test/bench），即任务禁写的"伪造 wrapper 接线"。
   - 注：同族 `moe_grouped_gemm_nopad` 的 wrapper 之所以可直接重写，是因为其 Stage-3 内核保留了工厂签名（`(numel, max_tiles, ...)`）且 wrapper 调用点已同步；本单元的 NPU 内核改变了运行期输入契约，wrapper 属显式 stale 脚手架。
3. 调度 prompt 断言的 `wrapper_path=null` 与 meta 事实不符（meta 为 `tileops/kernels/moe/grouped_gemm_persistent_3wg/grouped_gemm_persistent_3wg.py`）；判定按 meta 的 `scope` / `op_carrier` / `test_path` 而非 prompt 断言。

**结论**：`INTEGRATE_COMPLETED`，`scope=not-applicable`。只落盘 Stage-3 内核 + DESIGN 快照 + `_kernel/__init__.py` 重导出；**未**修改 wrapper / 未伪造 op；**未**运行 report。

## 落盘产物

- `grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py`（Stage-3 内核，整文件拷贝自 `examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/_persistent_grouped_gemm_v2_kernel.py`，`ASCEND_MODE = "Expert"`）
- `grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel_DESIGN.md`（Stage-1 交付件 DESIGN.md 快照）
- `grouped_gemm_persistent_3wg_kernel/__init__.py`（由 `integrate_kernel.py::gen_init_py` 生成：`from ._persistent_grouped_gemm_v2_kernel import _persistent_grouped_gemm_v2_kernel`，`__all__=["_persistent_grouped_gemm_v2_kernel"]`）
- `grouped_gemm_persistent_3wg.py`（wrapper）：**未修改**（`git diff` 为空）

## 验证结果

- report_run_json: none —— report **未运行**（无 manifest op，任务明确要求不跑）
- report_md: none
- import smoke（命令 `export ASCEND_RT_VISIBLE_DEVICES=1; python -c "import tileops.kernels.moe.grouped_gemm_persistent_3wg"`）: **PASS**
  - `smoke-ok ['GroupedGemmPersistent3WGKernel']`
- import smoke（`import tileops.kernels.moe.grouped_gemm_persistent_3wg.grouped_gemm_persistent_3wg_kernel`，即落盘的内核包）: **PASS**
  - `smoke-ok ['_persistent_grouped_gemm_v2_kernel']`
- correctness: n/a（无 test）
- benchmark: n/a（无 bench）

### 并发环境观测（非本单元问题）

首次 smoke（`2026-10-16 10:21:36`）失败，根因为**同族并发单元 `moe_permute_nopad` 处于中间态**：其 `moe_permute_nopad.py` 已改为 `from .moe_permute_nopad_kernel import _make_scan_kernel_nopad_ep`，但该 `_kernel/__init__.py` 只导出 `_make_gather_kernel_nopad` / `_make_scan_kernel_nopad`（且带 `# [warn] _make_scan_kernel_nopad_ep: not found in integrated modules`）。因 `import tileops.kernels.moe.grouped_gemm_persistent_3wg` 必先执行父包 `tileops/kernels/moe/__init__.py` 的聚合导入，遂报与本单元无关的 `ImportError`。约 1 分钟后（`10:22:38`）该兄弟单元补齐导出，重跑 smoke 通过。本单元未修改任何公共/兄弟文件。

## 调试历史

- 无（未进入调试闭环；`INTEGRATE_COMPLETED`/`not-applicable` 路径）
