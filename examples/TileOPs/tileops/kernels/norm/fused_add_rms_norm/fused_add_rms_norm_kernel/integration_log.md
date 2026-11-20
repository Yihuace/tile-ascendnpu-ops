# Integration Log — `FusedAddRMSNormFwdOp`

- **meta**:`examples/TileOPs/tileops/kernels/norm/fused_add_rms_norm/.migration_meta.json`
- **integrated**:`_fused_add_rms_norm_kernel` → `tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py`(源自 `examples/fused_add_rms_norm/_fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py`)
- **design_docs**:`_fused_add_rms_norm_kernel` → `tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel_DESIGN.md`(源自 `examples/fused_add_rms_norm/_fused_add_rms_norm_kernel/DESIGN.md`;**未缺失**)
- **wrapper**:`tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm.py` —— **已重写**为 `from .fused_add_rms_norm_kernel import _fused_add_rms_norm_kernel`(`wrapper_rewritten: true`)
- **ascend_mode**:`Developer`(**以环境变量形式生效**)
- **attempts**:1(集成侧首次即完成;**上游 Stage 3 经历了一次精度回退**,见下)
- **smoke**:pass

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261016_024415_885666_FusedAddRMSNormFwdOp/run.json`
- **report_status**:**`passed`** —— ⭐ **本批第一个 `passed`**(而非 `partial`)
- **correctness**:**12 / 12** 通过(`summary.correctness_passed = true`)
- **benchmark**:`benchmark_failed = 0`

> **为什么是 `passed` 而非 `partial`**:norm 族的 manifest **不含 268M 量级 workload**(其最大为 `(2048, 8192)`),故逐元素族那两条采集侧失败(`aclrtEventElapsedTime … 507000` / `event recorder null`)**不会出现**。这从正面进一步印证:那 2 条失败**源于 workload 规模**而非算子。

## 本次集成的实质变更

### 1. 交付件取代了既有手工实现(wrapper 已重写指向新包)

- 新包:`tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/`
- ⚠️ **残留孤儿目录**:`tileops/kernels/norm/fused_add_rms_norm/_fused_add_rms_norm_kernel/`(手工版)。集成后**已无引用**,但**未删除**(避免误伤未知引用)。**须由 conductor 决定是否清理。**

### 2. 上游 Stage 3 的精度回退(须在集成日志留痕)

交付件的形态是**两次 attempt** 的产物:

| attempt | 结果 | 根因 |
|---|---|---|
| 1 | **精度失败**(L0 0/3;`abs diff 2.72` vs 容差 `1e-5`) | **漏实现 fp32 分支**:`use_transit` 路径对 fp32 输入会执行 `T.vcast(x_tile, s_f32)` —— **两侧同 dtype ⇒ 量化成整数**(**E8** 陷阱) |
| 2 | **`[PRECISION_PASS]`** | 补上 fp32 分支:该分支在 dtype 域直接算,**全程不做同 dtype cast** |

⇒ 见 `examples/fused_add_rms_norm/_fused_add_rms_norm_kernel/RETROSPECTIVE.md` 的 Stage 3 章节。

### 3. 集成侧测试覆盖缺口

`tests/ops/test_fused_add_rms_norm.py`(12 项)不含**非 16 对齐的 N**、**全零行触发 `eps`**、**`rstd` 精度**等类别 —— 而 manifest 的 N **全是 16 的倍数**,对齐尾部路径在集成侧**从未被覆盖**。**`tests/` 不在本 Agent 的可改范围内**,故该缺口登记于 op 级复盘,交 conductor 决策。

### 4. 已知未交付项

无。本算子**未采用任何偏差声明**(与 `vector_norm` 不同,它没有"永不被触发的分支"需要裁剪)。

## 调试历史

集成侧无调试(attempt 1 即完成)。**上游 Stage 3 的调试链**见上表。

## 环境

- Ascend 910B2C,npu-smi `26.0.rc1`;CANN 9.0.0;torch `2.10.0+cpu`;torch_npu `2.10.0`
- tilelang wheel `0.1.2+npuir+7fc4a3c8`;profiling mode:msprof

## 已知未执行项(如实披露)

- **`SESSION_TIMING_ANALYSIS.md` 未生成**:harness.md §7.1 要求基于 `opencode export <session_id>` 生成,但本任务运行在 **AtomCode** 而非 OpenCode,**没有 `opencode` CLI**。按"无法确认时如实标注、不得伪造"的口径,**未伪造任何导出数据**,交由 conductor 处置。
