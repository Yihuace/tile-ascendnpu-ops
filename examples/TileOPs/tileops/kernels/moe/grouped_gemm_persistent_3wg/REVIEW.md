# _persistent_grouped_gemm_v2_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/DESIGN.md
- 任务类型: 迁移（harness；kernel-only 单元——无 manifest op 条目 / 无 Op 类 / 无 test/bench）
- 源算子: /tmp/opencode/TileOPs-ref/tileops/kernels/grouped_gemm/grouped_gemm_persistent_3wg.py
  （`GroupedGemmPersistent3WGKernel` + `_persistent_grouped_gemm_v2_kernel` 工厂 + `_make_pingpong_kernel` / `_make_cooperative_kernel`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/verify_equiv.py`
  → exit 0，`OVERALL: ALL EQUIV_PASS (guard diverges as intended)`。执行输出与 §1.6.1 内嵌结果表**逐项一致**：
  - float16 M1 `max_ulp=0.000e+00 viol=0/5 EQUIV_PASS`；M2 `0.000e+00 0/5 EQUIV_PASS`
  - bfloat16 M1 `0.000e+00 0/5 EQUIV_PASS`；M2 `0.000e+00 0/5 EQUIV_PASS`
  - guard `max_abs_diff=1.218e+03`(fp16) / `1.216e+03`(bf16) `viol=5/5`（发散，非采纳项）
  结论：10/10 EQUIV_PASS，且 guard 确认 `T.gemm(size=)` 归约界限为负荷项，与设计自述一致。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design <DESIGN.md>` JSON checks 摘要：
  - `ub_budget`: **skip** — "no per-level buffer tables parsed"。**人工复核结论**：§4.3/§4.5 本核无 UB 分配（调度索引段已外移，§0.6 R1），UB 预算 = 0，不构成预算项 → 无风险。
  - `l0c_budget`: **pass** — "block 64x64 x fp32 accum = 16.0 KB vs L0C 128 KB"（解析器取到 64×64 粒度）。人工复核 §4.5 真实值 `block_m×block_n×4 = 64×256×4 = 64KB ≤ 128KB` ✓。
  - `core_split`: **skip** — "insufficient parsed dims {'N':128,'bm':64,'bn':256}"。**人工复核结论**：§5.5 分核三要素齐全且自洽——① 逻辑核数（4 代表 shape 逐一复算：5 / 5120 / 12288 / 21504，本检视独立复算一致）；② 物理核数 = `NPUUtils.get().get_aicore_num()` 实查 20（查询代码 + 返回值 + 探针日志均在）；③ 规模判定（smoke 5≤20 无需适配 / 其余 5120–21504≫20 固定核数 20 + 核内 `T.serial`，边界静态）。`core_split` 失败条件未触发。
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。人工复算 §1.6.0 R3：FLOPs `2·numel·N·K = 1.924e12` ✓；A `numel·K·2 = 0.47 GB` ✓；B `E·N·K·2 = 15.03 GB` ✓；C `numel·N·2 = 0.27 GB` ✓；合计 ≈15.77 GB ✓。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源算子全文（1056 行，`/tmp/opencode/TileOPs-ref/.../grouped_gemm_persistent_3wg.py`），与 §0.1–§0.7 逐项核对。

- **语义理解正确性 — pass**。§0.1 数学语义 `C[o_e+r,:]=A[o_e+r,:]·B[e]^T`（等价 `C[e]=A[e]@B[e]^T`）与源 docstring L183（`C[e] = A[e] @ B[e]^T` for every expert）逐字一致。dtype 语义核对：源 `accum_dtype="float"`（L274）、WGMMA fp32 累加器、epilogue cast 回落输入 dtype（L529/L612 `T.copy(C_local, C_local_cast)`）——§0.1 描述准确。边界语义核对：源 K 对齐为硬前置（L202-203 `if self.K % block_k != 0: raise ValueError("K-aligned only")`，L1029-1034 工厂同样硬校验）、N 对齐（L204-205）——§0.1 的「K 对齐 / N 对齐」表述与源码一致；M 尾由 TMA OOB 零填充（L247-250 注释 + `A_shape` L290/L681 注释）+ epilogue full/partial 守卫处理，§0.1 亦如实写明。规约语义：K 维 `T.wgmma_gemm(clear_accum=(k==0))` 顺序分块累加（L512-519）——§0.1 表述一致。
- **I/O 核对 — pass**。§0.2 表 A`(numel,K)`/B`(E,N,K)`/true_sizes`(E,)`/true_offsets`(E,)`/C`(numel,N)` 与源码 `forward` 签名（L182-198）及 prim_func 签名（L292-299 等）一致；工厂签名 `_persistent_grouped_gemm_v2_kernel(numel,num_experts,N,K,dtype,sm_count,block_k)`（L1019-1021）与内层 `_func(block_m,block_n,block_k,num_stages,threads,group_size_m)`（L1044）一致；`out_idx=[]` 如实记录（L1037）。host 侧语义（`sm_count←multi_processor_count` L75-78、`torch.empty` 无 memset L223-224、无 host pad）逐条与源码一致。
- **算法解读完整性 — pass**。§0.3 步骤表 P0-a…P4 覆盖源码全部计算语句：P0-a 前缀扫描与源 L368-371 逐符号一致；P1-a static-wave `flat_id=2*(sm_count*w+pid)` 与源 L379 / L789 一致；P1-d 二分 `mid=(lo+hi)>>1; if s_cum[mid+1]<=m_tile` 与源 L390-394 一致；P2-b wgmma 与源 L512-519 一致；P3-b/P3-c full/partial 双路与源 L529-554 / L613-628 一致。host 逻辑（L145-155）亦列出。无遗漏、无臆造步骤。数据流图覆盖 GM/SMEM/fragment/TMA 全部 buffer。
- **优化手段覆盖完整性 — pass**。§0.4 的 O1–O14 全部在源码有对应实体：O1 `alloc_shared` ring（L302-305）、O2 `wgmma_gemm`（L512）、O3 `tma_copy`+`alloc_barrier`+`tx<128` 分流（L432-471）、O4 `make_swizzled_layout`（L335-342）、O5 static-wave（L379）、O6 `_swizzle_decode`（L698-708）、O7 双模板分派（L1045-1054）、O8 split-A/shared-B（L719-722）、O9 TMA store/谓词 STG 双路（L530-554）、O10 二分（L389-395）、O11 `dec/inc_max_nreg`（L365/L477）、O12 `warpgroup_fence_operand`（L521）、O13 persistent grid=`sm_count`（L300）、O14 `lru_cache`（L1019）。未把实现层优化误判为算法层（O5/O10 正确标为「算法+优化」）。§0.4 末行「无静默丢弃」的自我约束成立。
- **耦合性判定合理性 — pass**。§0.5 每项四态处置（保留/等价替换/重新设计/舍弃）与依据齐备：O3（warp-spec/TMA/mbarrier）→**重新设计**（NPU 无 warp/warpgroup 概念，保留「搬运-计算重叠」意图）；O4/O11/O12 → **舍弃**（NPU 无 SMEM bank conflict / 无 setmaxnreg / 无 WGMMA fence）；O7/O8 → **舍弃**（Hopper SMEM 228KB 预算产物，NPU L1 512KB 不同）；O1/O2/O9 → **等价替换**；O5/O6/O10/O13 → **重新设计**。舍弃项均写明理由，无「warp shuffle 判保留」式错判。依据与 `tilelang-op-design/references/ascend-constraints.md`、`docs/Tilelang.language/...` 不矛盾。
- **NPU 重设计可行性 — pass，且实测背书**。§0.6 R0–R8 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素：
  - R0 编程模式：源 CUDA 单 kernel → Expert 单 Cube scope。依据 CG-2026-0010（亲自核对 `.agents/evolution/capability-gaps.md` L382-433：Developer persistent+`T.gemm`+整型索引段 lowering 期 segfault，Expert 同结构正常）。**本检视独立核对成立**。
  - R1 调度外移：内嵌 static-wave 调度器 → 前导 kernel 产 3 张 GM int32 表。依据 CG-2026-0010 / CG-2026-0027（L1205-1242：`T.cumsum` 产 UB 前缀被标量消费不可同步）；且 CG-2026-0026（L1138-1202）的 workaround 明确写着「Expert Cube + 外部调度表 + 动态 extent gemm max_diff 9.5e-7」——**正是本设计的底座**。前导 kernel 的当前版本（`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` v2，两 kernel GM 交接）已规避 CG-2026-0027，且其产表契约（`tile_expert_ids`/`tile_row_offsets`/`total_tiles`）与本设计 §4.1 一致。
  - R2 warp-spec→单核 K 循环、R3 双模板/split-A→单模板、R4 `sm_count`→AI core 数、R5 TB swizzle→专家聚簇、R6 双路 epilogue→单路 `store_fixpipe`、R7 TMA OOB→动态 extent：语义保持论证到位（累加顺序不变 / 仅片上组织与调度变化 / 越界项数学恒 0）。R8 明示矩阵乘算法族无替换。
- **§1–§7 与迁移决策一致性 — pass**。§1.4 / §3.1 / §3.3 / §4 / §5 / §6 / §7 描述的均为 §0.5/§0.6 决策后的 Expert 单 Cube scope + 读 GM 表 + 动态 extent + 单路 `store_fixpipe` 形态；未见「§0 判重设计、§3 照抄源方案」的脱节（源的 producer/consumer 分流、mbarrier、双模板、swizzle 在 §3.3 中均无残留）。
- **golden 独立性 — pass（如实评估：该单元无 op 承载）**。§8.1 golden `golden_persistent_grouped_gemm` 以 §0.1 源算子语义实现（逐专家 NT matmul、fp32 累加后回落输入 dtype），**未复刻** §0.6 的 NPU 数据流（不含表读 / 动态 extent / `store_fixpipe`）。物理核对：其形态与源 op 测试 `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py::_ref_grouped_gemm_nopad`（L29-59，同样 `a_e.to(float32) @ b_e.to(float32).T` 后 cast）逐语句同构。本单元为 kernel-only（`.migration_meta.json` 标记 `scope=kernel_only`、`op_carrier=none`），无 manifest op 条目 / Op 类 / test 承载；golden 草案落在 §8.1、实际测试从等价 op（`MoeGroupedGemmNopadFwdOp`）借用——该事实已在 §0.7 / §8.1 / §10.2 如实披露，非缺陷。

### 1. API 可行性 — pass

- **API 存在性/佐证逐条核对**：`T.gemm`、`T.copy`、`T.store_fixpipe`、`T.min`、`T.alloc_L1`/`T.alloc_L0C`、`T.Kernel(n,is_npu=True)`、`T.serial` 均在本仓 examples 与 `tilelang/language/` 有佐证。
- **关键静态核对（亲自执行）**：DESIGN §3.4 声称 `T.gemm`「无 `transpose_B` 形参」。核对 `tilelang/language/__init__.py` L102-103：`npuir_dot, # noqa` 且 `npuir_dot as gemm`——即 `T.gemm` 实为 `npuir_dot` 别名，签名 `(A,B,C,size=[],initC=False,a_transpose=False,b_transpose=False)`（`customize_npuir.py` L542-589），**确无 `transpose_B`**；`gemm.py::gemm`（含 `transpose_B`）非本命名空间导出。**该声明成立**，设计统一用 `b_transpose` 正确。
- **模式匹配核对**：`T.gemm(...)` 于 `T.Scope("Cube")` 内使用，与先例 `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/_moe_grouped_gemm_kernel.py` L187/L222（Expert 单 Cube scope + 动态 extent `T.gemm`）及 D-3 探针同形，且该先例有 Stage 3 实测（max_diff 9.537e-07）。Expert 形态与 `T.store_fixpipe`（文档标题即「Expert 模式」）匹配。
- **v-prefix 优先**：本核为纯 Cube，无浮点向量运算，§3.2 说明恰当地澄清 v-prefix 不适用（不违反 AGENTS.md API Convention）。§3.3 注入的 `T.copy` 语义（GM→L1 ND2NZ、动态尾块）与 `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 第 2 条 / §2.3 第 6 条一致。
- 明确标注不可用 API：`T.sync_block_set/wait`（§3.4 末行，CG-2026-0026），本设计不使用——避坑正确。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出 `GM[A/B] --T.copy 切片(ND2NZ)--> L1[a_l1/b_l1] --> L0C[c_l0c] --store_fixpipe--> GM[C]`，无 GM→L0 越级路径。
- **UB 预算**：本核无 UB 分配（§4.3/§4.5），调度已外移 → UB = 0 ≤ 192KB。
- **L0C 容量**：`64×256×4 = 65536 B = 64KB ≤ 128KB` ✓（`design_calc_check.l0c_budget` pass，人工复核一致）。
- **L1 预算**：`(a_l1 8192 + b_l1 32768) × 2 stages = 81920 B = 80KB ≤ 512KB` ✓；§4.5 标注 num_stages 可达 4（160KB）。
- **Buffer 层级标注**：§4.3 每 buffer 均标 L1/L0C/GM。`pass_configs` 关闭 auto-multi-buffer，§4.5 据此说明不适用 ×1.7 膨胀系数（与 `constants.md` CONST-capacity-910B2C 的系数适用条件说明一致）。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 给出 `block_m=64/block_n=256/block_k=64` 及理由（单 Cube scope 下 L0C 须容纳全 `block_m×block_n`；源 cooperative 劈半粒度等价；L0C 余量 64KB）。
- **对齐约束**：M/N≥16、K≥32、32B 对齐 ✓（§5.3）。
- **非整除处理**：§5.4 三方向动态 extent；M 尾 `tile_m=T.min(block_m,s_e−row)` 且 `store_fixpipe` 只写有效区，杜绝越界写坏下一专家（no-pad 关键约束）；N/K 尾同法。
- **分核策略三要素（重点）**：§5.5 齐全——① 逻辑核数计算（`(numel//block_m+E)×ceil(N/block_n)`，本检视对 4 代表 shape 独立复算 5 / 5120 / 12288 / 21504，与设计一致）；② 物理核数依据为 `NPUUtils.get().get_aicore_num()` **实查 20**（查询代码 + 返回值 + 探针日志 `design_probe_persistent_gemm.log` L6 `[probe] NPUUtils.get().get_aicore_num() = 20`），Cube 算子直接用返回值不翻倍——符合 `core-split-strategy.md` §1；③ 规模判定：smoke（5≤20）无需适配、其余（≫20）固定核数=20 + `T.serial` 串行，`num_local_tasks` 边界静态。核数与 block 取值自洽。**核对标准**：符合 `.agents/skills/_shared/standards/core-split-strategy.md` §1/§2.2。

### 4. 技术约束检测 — pass

- 三维 Kernel：不用（一维 `T.Kernel(nums_kernel)`，tile 坐标由 `bx/by` 整除取模得），符合约束。
- GEMM 非整除：已在 §5.4 动态 extent 处理（`design_calc_check` 无 fail）。
- L0C 溢出：64KB ≤ 128KB（已在维度 2 处理）。
- §3.5.1 逐条列出本项目已知限制的涉及情况与处理方案，覆盖 host 侧禁止改动输入、跨引擎交接、Cube scope 维护调度状态等。

### 5. 循环与同步 — pass

- **循环结构**：§6.1 明确核级 `T.Kernel` / 核内任务 `T.serial(num_local_tasks)`（静态边界）/ K 维 `T.serial(ceildiv(K,block_k))`（Stage 4 可试 `T.Pipelined`）/ 元素级无。选择清晰。
- **同步与模式匹配**：§7.1/§7.2 为 Expert 单 Cube scope：跨 kernel 无显式同步（同 stream 顺序 launch）、L0C 写回前 `T.rs("PIPE_FIX")`、不使用 `T.sync_block_*`（CG-2026-0026）。与单 scope 形态自洽。
- **尾块处理**：§6.4 三方向动态 extent；死 tile 守卫 `if bx < total_tiles[0]` 位于 `T.serial` 体内、K 循环之外（规避 CG-2026-0001）。
- 死 tile 守卫 + 动态列偏移的 flaky 编译风险（CG-2026-0033）已在 §9.1/§9.2/§9.4 R-7 明确（抬升 `RLIMIT_STACK` + 有界重试），并已按 Gap Rule 登记（`.agents/evolution/capability-gaps.md` CG-2026-0033，occurrences=3、status=recurring，明确含本任务）。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 含 PyTorch 参考草案（逐专家 NT matmul、fp32 累加后 cast），与源 op 测试参考实现同构。
- **L0 测试计划**：§8.3 给出 6 用例（含 K 对齐、N<block_n、偏斜、奇偶 shape、代表大 shape、两 kernel 协同），dtype 覆盖 fp16/bf16，验证点明确；并说明 L0 单测用 host-built 表 fixture（只构造辅助输入、不改 A/B 真值）。
- **精度标准**：§8.2 `fp16/bf16 atol=rtol=1e-2`，与源 op 测试 `torch.testing.assert_close(..., atol=1e-2, rtol=1e-2)`（`test_moe_grouped_gemm_nopad.py` L101-102）一致；D-3 探针 worst max_diff 7.629e-06 远优于门限。

### 7. 完整性与一致性 — pass

- **章节齐全**：§0（含 0.1–0.7）–§11 全部存在。
- **无占位符**：全文经检查无占位符标记（模板槽位、待办标记、待补内容标记均未出现；grep 命中的花括号均为数学公式文本）。
- **同类引用真实**：§3.5.3 / §0.7 / §11 引用的 examples 路径经 Glob 逐一核对**全部存在**（`_moe_grouped_gemm_kernel.py`、`TRAP-SYNCBLOCK-ATTR-MISMATCH.py`、`_tile_scheduler_kernel/DESIGN.md`、`fp8_lighting_indexer.py`、`matmul_dynamic_shape.py`、`mixcv_mixkernel.py`、`example_gemm.py`、manifest `moe.yaml`、`test_moe_grouped_gemm_nopad.py`、`bench_moe_grouped_gemm_nopad.py`）。
- **内部一致**：§3 API 映射、§4 内存规划、§5 Tiling 三处对 `block_m=64/block_n=256/block_k=64`、L1/L0C 预算、`T.gemm(size=,b_transpose,initC)` 的描述相互一致，无矛盾；§3.3 伪代码与 §6 循环结构一致。

### 8. 算法优化分析 — pass

- **§1.6 章节存在**：§1.6 含 1.6.0 算法调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代分析 / 1.6.3 向量化轴与布局决策四子节。
- **算法调研存在性与完整性**：四问齐全——R1 候选表**含基线 C0** 且 C1–C8 逐项初判/收益/纳入列；R2 在线算法**明确结论「有」**（矩阵乘 K 维分块累加天然单遍流式，O(1) running 累加器，跨专家无依赖——结构判据，本检视认可）；R3 复杂度表**四口径齐全**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线 C0 在表、口径脚注明确；R4 逐候选六维亲和评估（单元匹配/容量/对齐/静态边界/流水/工具链可用性）。调研范围写明（`algorithm-candidates.md` 无 MoE grouped-GEMM 专条；参考 ALG-gemv；pattern-library；CG 缺口）。调研深度与矩阵类复杂度匹配。
- **调研结论正确性（独立复核）**：
  - ① 等价性：选定 C1 的候选等价性推演成立（见下 M1/M2/M3）；表内容与源逐元素一致（同 `cum` 定义、同 searchsorted 落点、同 `(-1,0)` 哨兵）承接 R1。
  - ② 复杂度算术**独立复算一致**（维度 6 段已列：FLOPs 1.924e12、A 0.47G、B 15.03G、C 0.27G、合计 15.77G）；中间缓冲峰值仅 L0C 64KB + L1 80KB，C7 split-K 的 partial 缓冲**已计入 GM 往返**（未漏计），口径自洽。
  - ③ 亲和性淘汰依据核对：容量/带宽淘汰引用 `constants.md` CONST 条目（`CONST-mte2-degradation` ~1.26 TB/s 混合流量地板、`CONST-vector-launch-overhead`、`CONST-capacity-910B2C`）——经核对条目存在且描述匹配。
  - ④ **负向断言主动复核**：R2「有在线形态」与「不存在需多遍物化的候选」经结构复核成立（矩阵乘非 softmax/attention 类）；R3 弃选 C2（工具链不可实现——亲自核对 CG-2026-0010/0026/0027 原文成立）、C5（死 tile 暴增）、C6（pad 膨胀）、C7（split-K 访存×nsplit + 归并）、C8（CPU 同步）；无与参考表/源码证据矛盾之负向断言。候选表与 `algorithm-candidates.md` 命中行核对无漏掉的已知候选（该文件确无 grouped-GEMM 专条，ALG-gemv 的 split-K 已被 C7 覆盖）。
  - ⑤ 未引用互联网候选，无来源记录缺失问题。
- **调研→设计一致性**：§1.6.0 选定 C1 实际驱动 §1.4（两 kernel 形态）、§3.1（公式拆解以 §1.6.1 优化后公式为输入）、§6（单核 K 循环 + 核内 `T.serial` + 守卫位置）。**迁移优化意图承接**：§1.6.0 末段对 §0.4 的 O1–O14 逐项给出承接/舍弃说明（O3→框架同步、O5→R1 外移、O6→R5、O9→R6、O11/O12 舍弃…），无静默丢弃；与 §0.5/§0.6 同口径。
- **数学等价优化完整性（四要素逐项）**：
  - M1 动态 extent + 动态 K 归约范围：原式（整块取数 + 谓词/守卫）→ 只取/只写有效区（`tile_m/tile_n/tile_k`）+ `T.gemm(size=[tile_m,tile_k,tile_n])` → 论证（越界项恒 0，0·x=0；归约范围=有效 tile_k 为前提）→ 收益量化（消 TMA OOB 零填充 + 合并 full/partial 双路）。**本检视独立推演**：`T.copy` 只写缓冲的 `[0,tk)` 区，`size=` 把归约上限设为 `tk`，故求和覆盖 `[0,K)`，与零填充全算后裁剪在数学上恒等（被裁项恒 0）；`stale_tail_guard` 证实缺 `size=` 时缓冲 `[tk,block_k)` 陈旧值进入累加（发散 ≈1.218e3）。**推演成立**。
  - M2 `initC=(k==0)`：原式（清零后累加）→ 首块覆盖初始化 → 等式（IEEE-754 `0+x==x` 精确，含 ±inf/NaN/subnormal）→ 收益（省 64×256×4=64KB/tile 清零写）。**独立推演成立**。
  - M3 NT 形式（`b_transpose=True`）：原式（显式转置后 `b_transpose=False`）→ 直接用 B[N,K] → 数学恒等 `(AB^T)`，源即 `transpose_B=True` → 收益（B 为最大张量 15GB，免全量重排）。**独立推演成立**。
  - M4 明示无进一步化简（无公共子表达式/高代价超越函数/除法可倒数化），结论与依据齐备。
  - **等价性论证正确性**：三项均为精确恒等或「删除恒零项/仅改搬运与归约范围」，不改变实际参与累加的浮点运算 → 不引入额外 fp16/bf16 舍入误差；与 §8.2 精度标准一致。verify_equiv 重跑与内嵌表一致、10/10 EQUIV_PASS。
- **向量化替代覆盖完整性**：§1.6.2 盘点 8 个循环/标量点，与 §3.3 伪代码、§6 循环结构**交叉核对无遗漏**——第 1/2 项（K 累加、A/B 装载）由 Cube `T.gemm`/`T.copy` 承担；第 3 项（输出写回+cast）由单路 `store_fixpipe` **整体消除**逐元素守卫与 C_shared staging；第 4 项（M 尾零填充）由动态 extent 取代；第 5–8 项（tile 坐标/表标量读/动态 extent/守卫）保留为 block 级标量。§6.1 元素级行「无（已消除）」与之一致，无「§6 仍有逐元素标量循环而 §1.6.2 未涵盖」的情形。
- **不可替代理由充分性**：第 5–8 项理由具体到本算子（block 级索引 / 每 task O(1) 标量 GM 读、`bx` 为运行时值、非逐元素热点、每核 O(num_local_tasks) 次相对 15GB B 访存可忽略），非泛泛而谈。
- **替代 API 可行性**：`T.gemm`/`T.copy`/`T.store_fixpipe` 佐证见维度 1；本核无 v-prefix 浮点向量 API 需求。
- **向量化轴与布局决策完整性**：§1.6.3 按 Cube 类枚举候选矩阵 L1–L6，覆盖「fractal-NZ/load_nd2nz 即时重排（L1）vs 显式转置（L2 host / L3 核内 T.transpose）vs epilogue 归属侧（L4 Vector / L5 Cube）」类别内必选候选；明确本核无 Vector 轴（Cube-only）的豁免说明。逐候选有量化取舍。**弃选论证前提核对（亲自打开文档/条目）**：L2 引用 `ascend-constraints.md` §4「host 侧禁止改动输入真实内容」——已核对原文存在（L60-62）；L3 引用 TRAP-transpose-epilogue-poison（`traps-compiler.md` L178-192「活跃源 transpose 毒化整 kernel 2.6x」）——存在；`docs/Tilelang.language/创建操作/T.transpose.md` 确认 `T.transpose` 为核内 UB 级（`hivm.hir.vtranspose`、仅两轴交换）——与 L3 的「核内转置链」代价机制描述一致，非 GM 级误解；L4 引用 CONST-store-fixpipe-gm-only（`constants.md` L140）——存在。无负向论断与文档/实测矛盾。**结论**：主选 L1+L5（NZ 分形 + epilogue 归 Cube `store_fixpipe`），不触发实验裁决三件套（NT+NZ 为文档指定通路，判定不依赖未实证常数）。
- **布局决策与实现一致**：§1.6.3 选定的 NZ 布局/epilogue 归属与 §3.3 伪代码（`T.copy` 切片 GM→L1 ND2NZ、`store_fixpipe`）、§4 内存规划（buffer 形状与 L1 预算）、§6 循环结构（无 Vector 索引循环）三方一致。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入（步骤 5 动态 extent、步骤 6 `T.gemm(size=)` + initC、步骤 7 动态 extent store）；§6 与 §1.6.2 一致（无被判向量化的逐元素标量循环残留）。

## 检视过程小结（非问题项，记录已完成的关键核对）

- 维度 0 的源码核对基准：源算子全文 1056 行亲读；§0.1/§0.2/§0.3/§0.4 与源码逐语句核对；§0.5/§0.6 处置内核级缺口经 `.agents/evolution/capability-gaps.md` CG-2026-0010/0026/0027/0033 原文核对成立。
- 选定底座（Expert 单 Cube scope + 外置调度表 + 动态 extent + 单路 `store_fixpipe`）的**可实现性**由 D-3 探针设备实测背书（`design_probe_persistent_gemm.py` 4 用例全 PASS，log 记录 cores=20、local_tasks≤3、`num_pid_n`>1，worst max_diff 7.629e-06），探针源码形态与 §3.3 伪代码同构；避开了 CG-2026-0010（Expert 非 Developer）、CG-2026-0026（单 Cube scope 无跨引擎交接）、CG-2026-0027（无「向量产出→标量消费」）。CG-2026-0026 原文 workaround 即本设计底座（max_diff 9.5e-7）。
- Hopper warp-spec/TMA/wgmma/双模板/split-A/O4 swizzle/O11 nreg/O12 fence 的处置均标注硬件依赖并有依据，无静默丢弃。
- `sm_count → AI core 数（实查 20）` 语义在 §0.6 R4 明确（无 SM 概念、Cube 直接用 `get_aicore_num()` 返回值、persistent grid = 该值），§5.5 附查询代码与返回值，探针日志印证。
- M1/M2/M3 等价性经独立推演 + `verify_equiv.py` 重跑双重确认（0 ulp，10/10 EQUIV_PASS；guard 发散 1.218e3 确认 `size=` 为负荷项）。
