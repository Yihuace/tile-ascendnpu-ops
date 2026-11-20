# _moe_grouped_gemm_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md（**revision 2**，Stage 3 `[DESIGN_ERROR]` 后的架构修订）
- 任务类型: 迁移（harness）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/moe/moe_grouped_gemm_nopad.py`（`_moe_grouped_gemm_kernel` 419 行；配套 `_tile_scheduler_kernel`）
- 上一版: `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/history_version/design_v1.md`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2
- 本轮专项复核: v2 是否把「调度内联 + Developer persistent」不可实现底座改为「前导 kernel 独立产调度表（GM）+ Expert 单 Cube scope 读表」

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）**: **PASS**。`python examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/verify_equiv.py` 执行输出与 §1.6.1 内嵌结果表**逐项一致且全 EQUIV_PASS**：
  - `M1 integer-ceil-identity`: max_metric 0.000e+00，0 / 24612 违反，**EQUIV_PASS**（基线 = 源整数 `(s+bm-1)//bm`，源 L95）
  - `M2 valid-region==padded`: max_metric **1.003e-11**，0 / 4 违反，**EQUIV_PASS**（含 K 分块 + 陈旧尾值注入）
  - `M3 NT==B^T form`: max_metric 0.000e+00，0 / 1 违反，**EQUIV_PASS**
  - `stale-tail guard`（漏传 `size=` 的错误形态）: max_abs_diff = **1.218e+03**（发散），与 §1.6.1 guard 行一致 → 证明 M2 的等价性前提 load-bearing
  - 实跑数字（0.000e+00/24612、1.003e-11/4、0.000e+00/1、1.218e+03）与 §1.6.1 内嵌表逐一吻合。
- **design_calc_check（D-5）**: `python3 .agents/tools/design_calc_check.py --design examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md` JSON checks 摘要：
  - `l0c_budget`: **pass** — detail 报 "block 64x64 x fp32 accum = 16.0 KB"（解析器把 block_k 当 block_n 误解析）。**人工复核（机械跳过口径）**：真实 L0C = block_m×block_n×4 = 64×256×4 = **64 KB ≤ 128 KB** ✓（结论不变）。
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"）。**人工复核**：本核**无 UB 分配**（§4.3/§4.5，调度阶段已外移至前导 kernel），预算 trivially 满足；`a_l1(8KB)+b_l1(32KB)` × num_stages(2) = 80 KB ≤ L1 512KB ✓。
  - `core_split`: **skip**（"insufficient parsed dims {'N':128,'bm':64,'bn':256}"）。**人工复核**：§5.5 三要素齐全且算术自洽——逻辑核数 = max_tiles×num_pid_n：smoke(64,4,128,64) 5×1=**5**；decode gate-up(4096,256,4096,7168) 320×16=**5120**；prefill gate-up(32768,256,4096,7168) 768×16=**12288**；prefill down(32768,256,7168,2048) 768×28=**21504**（独立复算通过）；物理核数本机实查 `NPUUtils.get().get_aicore_num()` = **20**（与 capability-gaps CG-2026-0010 第二证 "Ascend910B2C(aicore 20)" 一致）；`nums_kernel=min(logical,20)` 与 `num_local_tasks=ceildiv(...)` 均编译期常量，循环边界静态 ✓。
  - `r3_metrics`: **pass**（四口径 FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值齐全）。

## 本轮专项复核（v2 架构修订：不可实现底座 → 可实现架构）

| # | 复核点 | 结论 | 证据 |
|---|--------|------|------|
| ① | **撤回并重估 v1「融合单核」收益** | **通过** | §0.6 R1「v1 方案（已撤回）」给出两条独立阻断（CG-2026-0010 Developer lowering segfault；CG-2026-0026 `syn_instr_mode` 属性名不匹配 + 官方 mixcv 样例复现）；§1.6.0 R1 表 C2 标「❌ 撤回：工具链不可实现」；§1.6.1「优化结论」**收益重估**把「少 3 张表 GM 往返 + 少一次 launch」撤回，改记本形态多出的 launch + 表往返(≈6.1 KB) + 跨 kernel barrier 合计 < 1 ms（≪ B 访存 15.03 GB ≈ 11.9 ms）。收益主张不再纸面保留 |
| ② | **两函数对接契约明确** | **通过** | §0.6 R1「对接契约（本核消费）」+ §4.1 六条契约：① `max_tiles = numel//block_m + E` 与 `block_m` 为两函数共享编译期常量；② 表 dtype 全 int32；③ `tile_row_offsets` 单位=行（已乘 block_m，与源 L113 `(tile_id - s_cum[lo]) * block_m` 一致）；④ 死 tile 哨兵 `(-1,0)`；⑤ `total_tiles ≤ max_tiles`；⑥ 同 stream 顺序 launch。§9.4 R-5 显式要求胞弟据此修订其设计，避免两设计互相矛盾 |
| ③ | **Expert 单 Cube scope 无 in-block 跨引擎同步** | **通过** | §2.1「单 Cube scope，无 Vector 段、无跨引擎同步」；§3.3 伪代码仅 `T.Scope("Cube")`，无 `T.sync_block_set/wait`；§7.2 同步表明确「前导 kernel→本核=无显式同步（同 stream 自然顺序）」，唯一同步为 Cube scope 内 `T.rs("PIPE_FIX")`。§3.4 把 `T.sync_block_set/wait` 标「**不使用**」。Stage 3 正对照 `repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` 的 `positive_control()` 为同形态（实读确认：单 `T.Scope("Cube")`、动态 extent、`T.gemm(size=)`、`store_fixpipe`），repro.log 实测 **max_diff 9.537e-07** |
| ④ | **保留项完好** | **通过** | 动态 extent（§0.6 R2/§5.4/§3.3 `tile_m/tile_n/tile_k=T.min(...)`）；`T.gemm(size=[tile_m,tile_k,tile_n])` 限定归约范围（§3.3 注 1 + §1.6.1 M2 + §5.4）；NT `b_transpose=True`（§1.6.1 M3）；专家聚簇任务序（§0.6 R4，flat `tile_id → bx=tile_id//num_pid_n, by=tile_id%num_pid_n`）；B 访存主导下界（§1.6.0 roofline，B 15.03 GB ≈ 11.9 ms 占 95%+）。四项与 v1 一致，未被架构修订波及 |

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对基准**：已亲自 Read 源算子全文（`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/moe_grouped_gemm_nopad.py`，419 行）与源测试 `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py`（`_ref_grouped_gemm_nopad`）。以下逐项附源码语句/位置。

- **语义理解正确性 — pass**。§0.1 的 $C[o_e+r,:]=A[o_e+r,:]B[e]^T$ 与源码一致（源 L140 `A[numel,K] × B[E,N,K]^T → C[numel,N]`）；规约轴=K 完整求和、fp32 分块累加（源 L149 `accum_dtype="float"`、L191 `C_local = T.alloc_fragment([block_m, block_n], accum_dtype)`）；输出回落输入 dtype（源 L158 `C_shape=(numel,N)` + L152 `out_idx=[2]`）；M/N/K 尾块语义与源一致（源 L218-220 `actual_rows/actual_cols`、L281-287 `if_then_else(...,0)`、L233-235/L290-292 epilogue 守卫）；no-pad 紧致布局「M-tile 不跨专家、尾块多余行属下一专家」的判断与源 L199-211 的 tile 映射一致。host `F.pad(A,(0,0,0,block_m))` 被判「实现手段而非语义」正确（源 L403-404 `if self.K % block_k == 0` 分支 + L394-402 注释「Pad with block_m zero rows so the T.copy lands in valid memory」）。
- **算法解读完整性 — pass**。§0.3 步骤表逐句覆盖源码：P1-a 前缀扫描（源 L92-98，`s_cum[e+1]=s_cum[e]+(true_sizes[e]+block_m-1)//block_m`，thread 0 串行）；P1-b 二分查找（源 L106-113，`mid=(lo+hi)>>1`、`(tile_id - s_cum[lo])*block_m`）；P2-1 `bx/by`（源 L201-203）；P2-2（源 L214-217 `expert_id/m_start/n_start`）；P2-3（源 L218-220）；P2-4 取数（源 L227-230 快路 / L280-287 慢路）；P2-5 `T.gemm(transpose_B=True)`（源 L231/L288）；P2-6 epilogue（源 L233-235/L290-292）。无遗漏、无臆造。host 侧三条逻辑（F.pad、`max_tiles = numel//block_m + E`（源 L342-343）、两 kernel 顺序调用（源 L407-418））已列出。
- **优化手段覆盖完整性 — pass**。§0.4 的 O1–O10 与源码逐项印证：O1 SMEM tiling（源 L189-190 + L192-195 `make_swizzled_layout`）、O2 `T.gemm`/tensor core、O3 TMA+`T.Pipelined(num_stages)`+WarpSpecialized（源 L223-225 注释 + L226）、O4 swizzle、O5 GROUP_SIZE_M 重排（源 L197-211 + L133-138 注释）、O6 死 CTA 早退 + 保守 `max_tiles`（源 L213 + L4-5 docstring）、O7 tile 调度器（源 L55-120）、O8 `_b_copy_ok` 编译期分派（源 L163-168）、O9 谓词零填充（源 L281-287）、O10 epilogue 守卫；host `F.pad` 亦识别。**未把实现层优化误判为算法层**（swizzle/GROUP_SIZE_M 判为硬件相关优化，谓词零填充判为可移植算法）。
- **耦合性判定合理性 — pass**。§0.5 每条含四态处置 + 依据，与 migration-analysis.md §5.3 GPU→NPU 映射表不矛盾：O2 → Cube（`hivm.hir.mmadL1`，分形 M/N≥16 K≥32）；O3 → MTE2/ND2NZ + `T.Pipelined`（stage 数按 L1 重算）；O4 swizzle「舍弃」（NPU 无 SMEM bank conflict，与映射表「swizzle 防 bank conflict → 舍弃」一致）；host `F.pad`「舍弃」引 ascend-constraints.md §4（host 禁止改动输入真实内容）；O6「重新设计」引 core-split-strategy.md §1 要素③（一维核超发被串行调度）；O7「保留（独立产出 GM 表）」引 CG-2026-0026。舍弃项（O4、host pad、核内调度折叠）理由均已写明。
- **NPU 重设计可行性 — pass**。§0.6 R0–R5 逐项有「源方案 → NPU 新算法 → 语义保持论证」三要素：
  - R0（编程模式 Developer→Expert）：语义保持论证「编程模式不改变数值语义」成立，附 Stage 3 正对照 max_diff 9.537e-07。
  - R1（调度承载）：新算法三步（前导 kernel 产 3 张 GM 表 / GEMM 核读表 / 同 stream 顺序 launch 无 in-block 同步）；语义保持论证「表内容逐位一致（同一 cum 定义、同一落点、同一哨兵）、变化仅在求值时机」成立（与源 L74-116 调度器一致）。
  - R2（动态 extent）：语义保持论证「越界贡献恒为 0，前提=归约范围=tile_k」并以 verify_equiv M2 + `stale_tail_guard`（1.218e3 发散）支撑；no-pad 关键边界（只写 `[0,tile_m)` 杜绝越界写坏邻专家）论证正确；dtype 路径（`store_fixpipe` fp32→fp16/bf16，源 L158 同 dtype 输出）一致。
  - R3（固定核数 + 核内串行）、R4（专家聚簇任务序）：均论证「只改调度不改数值」。
  - 重设计满足 NPU 约束：一维 Kernel、动态 extent 对齐、L0C 64KB ≤128KB、循环边界静态。
- **设计与决策一致性 — pass**。§1–§7 描述的是 §0.5/§0.6 决策后的 C1 算法（两 kernel：前导 kernel 独立产 GM 表 + Expert Cube-only 读表 + 动态 extent + 专家聚簇 + 固定核数核内串行），无「§0 判重设计、§3 照抄源方案」脱节：§3.5.2 差异表逐行对应 §0.6 R1–R4；§1.4/§1.5/§4.4 的数据流均为「调度表外部产出→本核读表」，与 §0.6 R1 一致。**v1 的「核内调度内联」形态已在全文清除**（§1.1/§2.1/§3.5.1/§3.5.2/§5.1/§7.2 均显式标注「已撤回」）。
- **golden 独立性 — pass**。§8.1 `golden_moe_grouped_gemm_nopad` 以 §0.1 语义实现（逐专家 `a[off:off+s].float() @ b[e].float().T` 后回落输入 dtype），与源测试 `_ref_grouped_gemm_nopad` 同构；**未复刻 §0.6 的 NPU 算法**（无调度表、无动态 extent、无 NZ/ND2NZ、无 `store_fixpipe`），验证独立 ✓。

### 1. API 可行性 — pass

- **API 存在性 — pass**。本检视亲自复核 `inspect.signature(T.gemm)` = `(A, B, C, size=None, initC=False, a_transpose=False, b_transpose=False)`（`tilelang/language/__init__.py` L103 `npuir_dot as gemm`）——**无 `transpose_B`**，设计全部用 `b_transpose=True`（§3.3、§3.4、§1.6.1 M3）。`size=[a,b,c]` 语义经 `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1 核对（src1 [a,b] 或 [b,a]，src2 [b,c] 或 [c,b]，dst [a,c]）——本核 src1=A_l1[tile_m,tile_k]、src2=B_l1[tile_n,tile_k] 且 `b_transpose=True`，故 `size=[tile_m,tile_k,tile_n]` 正确。其余 API 均有本仓 examples/docs 佐证（本检视实读）：`T.Kernel(nums_kernel, is_npu=True)`+核内 `T.serial` 单比较守卫（`examples/deepseek_v32/fp8_lighting_indexer.py` L53-57 结构同形）、`T.copy` 切片动态 extents（`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 第 2 条 + §2.4 示例 4；`examples/gemm/matmul_dynamic_shape.py` L28-44）、`T.min`（`docs/.../比较操作/T.min.md`）、`T.store_fixpipe` 动态 extent（`docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.4；`mixcv_mixkernel.py` L57）、`T.alloc_L1/alloc_L0C`（mixcv_mixkernel.py L40-46）、`T.rs("PIPE_FIX")`（fp8_lighting_indexer.py L83）。
- **API 与模式匹配 — pass**。Expert 用 `T.rs` 管线同步（不误用 Developer 自动同步 API）；手动跨引擎同步**不使用**（§7.1/§7.2），与单 Cube scope 形态一致。
- **v-prefix 优先 — pass（正确豁免）**。§2.1/§3.2/§5.1 明确本核数据通路 Cube 主导、无 v-prefix 浮点向量 op；按 AGENTS.md MixCV 触发规则（Cube `T.gemm` + 至少一个 v-prefix op），本核严格不满足判据，归为纯 Cube（Expert）。属**正确豁免**而非遗漏；§2.1 并披露调度 prompt 的「混合 Cube+Vector」标签与实测数据通路的差异。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**。§4.4 路径：`GM[A/B] → L1[a_l1/b_l1]（ND2NZ）→ L0C[c_l0c]（T.gemm）→ GM[C]（store_fixpipe nz2nd）`，调度表 `GM → 标量读`；无 GM→L0 越级直连。
- **UB 预算 — pass**。本核**无 UB 分配**（§4.3「本核无 UB 分配」），预算 trivially 满足（机械 `ub_budget` skip，人工已核）。
- **L0C 容量 — pass**。block_m×block_n×4 = 64×256×4 = **64 KB ≤ 128 KB** ✓（机械 `l0c_budget` pass，解析器 64×64 为误解析，人工复算 64 KB）。
- **Buffer 层级标注 — pass**。§4.3 每个 buffer 标注存储层级（a_l1/b_l1=L1，c_l0c=L0C，调度表=GM）。

### 3. Tiling 策略 — pass

- **Block 划分明确 — pass**。§5.2 给出 block_m=64 / block_n=256 / block_k=64 及理由（分形 M/N≥16、K≥32；L0C 64KB 预算；专家行数通常 ≥ 数十）。
- **对齐约束 — pass**。三方向均 ≥ 分形下限且 32B 对齐 ✓（§5.3）。
- **非整除处理 — pass**。M/行方向 `tile_m=T.min(block_m, s_e−row)`（`row` 来自调度表已乘 block_m）+ 动态 extent 写回（杜绝越界写坏邻专家）；N 方向 `tile_n=T.min(block_n,N−n_start)`；**K 方向 `tile_k=T.min(block_k,K−k·bk)` 显式传入 `T.gemm(size=[tile_m,tile_k,tile_n])`** 限定归约范围（§5.4，消除 `[tile_k,block_k)` 陈旧尾值污染；正确范式 `matmul_dynamic_shape.py` L31-41）。
- **分核策略（物理核数适配）— pass**。三要素齐全且自洽（对照 core-split-strategy.md §1）：
  - ① **逻辑核数计算**：`num_logical_kernels = max_tiles × num_pid_n`（§5.5）；代表值 smoke 5 / decode 5120 / prefill gate-up 12288 / prefill down 21504，独立复算无误。
  - ② **物理核数依据（实查）**：§5.5 记录 `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（Ascend910B2C，2026-10-13）；本算子为 Cube 算子直接用返回值 20（纯 Vector 才翻倍）。与 `capability-gaps.md` CG-2026-0010 第二证记录的 "Ascend910B2C(aicore 20)" 相互印证（旧 pattern-library 常数 CONST-aicore-910B2C 记 24 属更早工具链/CANN 8.5.0 口径，本机 CANN 9.0.0 实查为 20，取实查值正确）。
  - ③ **规模判定与分核方案**：smoke 5 ≤ 20 →「无需适配」；代表 shape 5120–21504 远超 20 → **极大规模**，固定 `nums_kernel = min(logical, 20)` + 核内 `T.serial(num_local_tasks)`，`num_local_tasks = ceildiv(logical, nums_kernel)` 编译期常量（**静态边界**），块连续分配以利专家聚簇，负载均衡说明完备。
  - 机械 `core_split` skip（解析器缺 N 维信息），人工已核如上。

### 4. 技术约束检测 — pass

- **三维 Kernel — pass**。源为展平一维 grid（源 L188 `T.Kernel(_total_ctas, threads=threads)`），设计改一维 `T.Kernel(nums_kernel, is_npu=True)` + 核内 `bx/by` 计算，无三维 Kernel 残留（§3.5.1、§4.6）。
- **GEMM 非整除 — pass**。已在维度 3 处理（动态 extent + `T.gemm(size=)`），§3.5.1「GEMM M/N 需为 block 整数倍」行明确处理方案。
- **L0C 溢出 — pass**。已在维度 2 处理（64 KB ≤ 128 KB），§3.5.1 L0C 行。
- **host 侧禁止改动输入 — pass**。源 `F.pad(A)` 明确「舍弃」（§0.5、§3.5.1、§3.5.2），改动态 extent 解决。

### 5. 循环与同步 — pass（1 条建议级观察）

- **循环结构 — pass**。§6.1 明确核级（`T.Kernel(nums_kernel)` 固定核数）、核内任务（`T.serial(num_local_tasks)` 静态边界）、死 tile 守卫（单比较，位于 `T.serial` 体内、`T.Pipelined` 之外）、K 维（`T.serial` 或 `T.Pipelined`）、元素级（无，§1.6.2 已整体消除 epilogue 守卫）。
- **同步与模式匹配 — pass**。§7.2 同步表：前导 kernel→本核=无显式同步（同 stream 自然顺序/kernel 边界隐式全局顺序）；L0C 写回前 `T.rs("PIPE_FIX")`；A/B GM→L1 与 `T.gemm` 之间=框架/`T.rs` 单引擎（MTE2→Cube）管线同步；`T.sync_block_set/wait` 显式标「不使用」（CG-2026-0026）。与单 Cube scope 形态完全匹配，**无跨引擎同 block 交接**。
- **尾块处理 — pass**。§6.4 三方向统一动态 extent，K 方向显式把 `tile_k` 传入 `T.gemm(size=)`；死 tile 守卫 `if` 置于 `T.serial` 体内、`T.Pipelined` 之外（规避 CG-2026-0001）。
- **建议级观察（不阻塞）**：K 循环在 §3.3 伪代码用 `T.serial`，而 §3.2/§6.1/§6.3 并列表述为 `T.serial` 或 `T.Pipelined(num_stages)`（§9.4 R-2 已自认该选择待 Stage 3/4 实测裁决）。二者均为合法且正对照以 `T.serial` 跑通，属**性能形态选择**而非语义正确性，建议 Stage 3 据实测在 §6.3 收敛为单一形态并同步 §3.3。

### 6. 验证方案 — pass

- **Golden 函数 — pass**。§8.1 含 PyTorch 参考实现草案（§8.1 代码块），与 §0.1 语义、源测试 `_ref_grouped_gemm_nopad` 一致（fp32 累加后回落输入 dtype），显式声明「不得复刻 §0.6 的 NPU 算法」。
- **L0 测试计划 — pass**。§8.3 含 6 条门槛用例（L0-1 K 对齐+N<block_n smoke、L0-2 偏斜+fp16、L0-3 K 非整除 bf16、L0-4 中等偏斜奇偶 shape、L0-5 代表大 shape、**L0-6 两 kernel 协同/对接契约**），每条含具体 shape/dtype/dist 与验证点；L0-3（K=96）/L0-4（K=70）覆盖 K 尾动态 extent 与 `T.gemm(size=)`；L0-6 专验两函数对接契约（表内容、`total_tiles` 全程设备侧、无 CPU 同步）。
- **精度标准 — pass**。§8.2 fp16/bf16 atol=rtol=1e-2，与源测试 `torch.testing.assert_close(..., atol=1e-2, rtol=1e-2)` 一致；附 Stage 3 正对照 max_diff 9.537e-07（远优于门限）。

### 7. 完整性与一致性 — pass

- **章节齐全 — pass**。§0（0.1–0.7 全）与 §1–§11 均存在；含「修订说明（revision 2）」表。
- **无占位符 — pass**。全文占位符标记检索 0 命中（未出现任何未填充占位标记）。
- **同类引用真实 — pass**。§3.5.3 引用的 `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`、`examples/deepseek_v32/fp8_lighting_indexer.py`、`examples/gemm/matmul_dynamic_shape.py`、`examples/mixcv/mixcv_mixkernel.py`、`testing/autotune/anneal/moe_grouped_gemm_anneal.py`、`examples/gemm/example_gemm.py` 经 Glob 逐一确认**存在**；§11 的 `examples/TileOPs/...` 路径亦存在。
- **内部一致 — pass**。API 映射（`b_transpose` + `T.gemm(size=)`）、内存规划（无 UB / L1 80KB / L0C 64KB）、Tiling（动态 extent）三处描述相互一致；v2 修订说明表 6 项与正文（§0.5/§0.6/§1.6.0/§1.6.1/§2.1/§4.1/§7）逐项对应，无遗留 v1 内联形态表述。

### 8. 算法优化分析 — pass（1 条建议级观察）

- **§1.6 章节存在性 — pass**。含 §1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与数据布局决策四子节。
- **算法调研存在性与完整性 — pass**。§1.6.0 四问齐全：
  - **R1** 候选表含**基线候选 C0（源算法）**并评估 C0–C8（含 C2 融合单核撤回、C3 动态 extent、C4 NT、C5 无搜索映射、C6 专家内 pad、C7 split-K、C8 host 调度）。
  - **R2** 在线算法有**明确结论「有，且是本题天然形态」**并给结构判据（输出不含跨 tile 全局依赖、M/N 可任意分块、K 是唯一规约维可单遍 running 累加；源算法已用「分块 K 累加 + 前缀扫描」即直接证据）。
  - **R3** 四口径齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线在表，脚注写清口径（含「不含 L1/L2 复用命中」「中间缓冲以片上为准」「访存含调度表 GM 往返」）。
  - **R4** 逐候选六项亲和性评估（单元匹配/片上容量/对齐整除/静态边界/流水融合/跨核）。
  - 含选定算法（C1）与关键依据、「调研范围」（algorithm-candidates.md 无 MoE grouped-GEMM 专条 + ALG-gemv + pattern-library + 源码 + 本仓 examples）。
- **调研结论正确性（独立复核）— pass**：
  - ① **等价性**：C1 与原语义等价性独立推演成立——C1 相对 C0 的实质变化是「host pad + 谓词零填充 + epilogue 守卫 → 动态 extent」「超发 grid → 固定核数」「GROUP_SIZE_M → 专家聚簇」「调度内联(已撤回) → 独立前导 kernel 产表」；前三项为删恒零项/纯调度重排，第四项「表内容逐位一致」经源 L74-116 与 §4.1 契约核对成立。容差内等价（M2）已评估 fp16/bf16 舍入：三项均不改变实际参与累加的浮点项。**注意**：C1 的动态 extent 等价性**强依赖**「归约范围=tile_k」，设计已通过 `size=` 落实，且以 `stale_tail_guard`（陈旧尾值注入，1.218e3 发散）证明该前提不可省——复核成立。
  - ② **复杂度算术独立复算**（shape numel=32768, E=256, N=4096, K=7168, bf16 2B）：FLOPs=2·32768·4096·7168=**1.924e12** ✓；B=256·4096·7168·2=**15.03 GB** ✓；A=32768·7168·2=**0.47 GB** ✓；C=32768·4096·2=**0.27 GB** ✓；表 `(2·max_tiles+1)·4`，max_tiles=numel//64+256=768 → 6148 B ≈ **6.1 KB** ✓；总 ≈ **15.77 GB** ✓；roofline 15.77 GB ÷ 1.26 TB/s（CONST-mte2-degradation 混合流量地板，实读存在）= **12.5 ms** ✓，B 单遍 15.03 GB ≈ **11.9 ms** ✓。**中间缓冲无漏计 GM 往返**：C1 无中间张量落 GM（L0C 片上累加），调度表仅 ~6 KB；C7 split-K 的 partial GM 往返已如实计入。口径自洽（含/不含项写明）。
  - ③ **亲和性判断**与 ascend-constraints §1/§4、constants.md 实测条目（CONST-capacity-910B2C：L1 512KB/L0C 128KB；CONST-mte2-degradation：混合流量地板 ~1.26 TB/s）不矛盾；C5/C6/C7/C8 淘汰有量化依据（死 tile ~64× / FLOPs 与 A 访存膨胀 / B 访存 ×nsplit + 跨核归并 / CPU 同步违反零同步意图）。
  - ④ **负向断言主动复核**：**本检视亲自读** `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`——grep `moe|grouped` **0 命中**，确无 MoE grouped-GEMM 专条；ALG-gemv 存在（含 R1「Cube 分形路径 vs Vector 规约累加」、R2「split-K 分块累加」），设计「参考 ALG-gemv 与 split-K 条目」属实且已把 split-K 作为 C7 评估。**关键负向断言「C2 融合单核不可实现」**经独立复核成立：`examples/mixcv/mixcv_mixkernel.py`（官方 MixCV 样例）使用 `T.sync_block_wait/set`，而 `repro/repro.log` 实读确证同报错 `error: custom op 'syn_instr_mode' is unknown`（`hivm.hir.sync_block_set[...] flag = 0 syn_instr_mode = ...`），与 capability-gaps CG-2026-0026 记录一致。**未发现被漏掉的已知候选**（GEMM 族无在线多遍候选；split-K 已评估）。
  - ⑤ **并列候选裁决**：C0 vs C1 裁决有明确依据（超发 grid 在 NPU 被串行调度 + host pad 被禁 + 跨引擎交接不可编译）；无「判定裕度依赖未实证常数」的轴决策（NT+NZ 为文档指定通路），§1.6.3 正确判定不触发实验裁决三件套。
  - ⑥ **调研→设计一致性**：§1.4/§3.1/§6 描述的是 §1.6.0 选定 C1；迁移侧 §0.5/§0.6 处置与调研选族一致；**源算法优化手段意图逐项承接**（§1.6.0 承接段：O1→L1 分块、O2→Cube `T.gemm`、O3→MTE+`T.Pipelined`、O4 **舍弃**、O5→R4 专家聚簇、O6→R3 固定核数+核内串行、O7→R1 **保留独立调度 kernel + GM 表**、O8/O9→R2 动态 extent、O10→R2 写回、host pad→**舍弃**），无静默丢弃。
- **数学等价优化完整性 — pass**。§1.6.1 对选定 C1 系统审视，M1/M2/M3 均有「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；M4 对「无进一步公式化简」给出结论与依据（无公共子表达式、无高代价超越函数、无除法可倒数化）。M1 原式为源整数 `(s_e+block_m-1)//block_m`（源 L95），收益如实标注「≈0，属无损整理」。
- **等价性论证独立推演 — pass**。逐项推演：M1 `(s+bm-1)>>log2(bm) ≡ (s+bm-1)//bm`（bm=2^k、s≥0）**成立**（24612 例 0 违反）；M3 `A@B^T`（B[N,K] 行主序）与物化 B^T 后相乘**恒等**（0 违反）；M2 有效区 `[0:tile_m)×[0:tile_k)` + `size=[tile_m,tile_k,tile_n]` 归约 vs 零填充整块——**独立推演成立**（零填充区乘 0 贡献恒 0），前提「归约范围=tile_k」由 `size` 落实，`stale_tail_guard` 以陈旧尾值注入证明漏删前提发散（1.218e3）。fp16/bf16 舍入影响评估成立（不改变参与累加的浮点项集合）。**verify_equiv 重跑结果与内嵌表逐项一致**（见机械复核）。
- **向量化替代覆盖完整性 — pass**。§1.6.2 盘点**本 GEMM 核全部循环/标量点**（8 项），与 §3.3 伪代码、§6 循环结构交叉核对**无遗漏**：第 1 项 K 累加循环→Cube `T.gemm`（已替代）；第 2 项 A/B 取数→`T.copy` 切片（已替代）；第 3 项 epilogue 逐元素守卫→动态 extent 写回（**整体消除**）；第 4 项 host `F.pad`→删除（已替代）；第 5 项 tile 坐标/基址、第 6 项调度表标量读、第 7 项动态 extent `T.min`、第 8 项死 tile 守卫→保留为 block 级标量。判定向量化的点（1/2/3/4）在 §6 未再出现对应逐元素标量循环 ✓。调度阶段的向量化点已按 §0.6 R1 外移至前导 kernel，§1.6.2 注明由胞弟 §1.6.2 承载（范围划分清晰）。
- **不可替代理由充分性 — pass**。第 5/6/7/8 项均给出具体到本算子的理由（block 级索引/任务映射、O(1) 运行时标量 GM 读（数据来自 GM 表，正是源调度器设计目的）、block 级 `T.min`、block 级单比较守卫），且明确「非逐元素热点（每核 O(num_local_tasks) 次标量，相对 B 访存 15 GB 量级可忽略）」。无「实现简单/照源码写」式空泛理由。
- **替代 API 可行性 — pass**。替代实现的 `T.gemm(..., size=[tile_m,tile_k,tile_n], b_transpose=True, initC=(k==0))`、`T.copy` 切片动态 extents、`T.store_fixpipe(size=[tile_m,tile_n])` 与 §3.2/§3.4 一致，形参名与 `inspect.signature` 及本仓 examples/doc 佐证一致（与维度 1 同口径）。
- **向量化轴与布局决策完整性 — pass**。§1.6.3 按 Cube/MixCV 类别给出候选矩阵 L1–L6，覆盖类别必选「fractal-NZ/load_nd2nz 即时重排 vs 显式转置/epilogue 归属侧」；逐候选有量化取舍；**弃选均附文档/实测依据**：L2（host 侧 B 转置）引 ascend-constraints §4 + 15 GB 全量转置量级；L3（核内 `T.transpose`）引 `T.gemm.md` §2.1（NZ 通路已支持 NT）+ pattern-library `TRAP-transpose-epilogue-poison`（活跃源 transpose 毒化 2.6×，本检视确认条目存在）；L4（epilogue 归 Vector）引 `CONST-store-fixpipe-gm-only`（L0C 跨引擎强制 GM 往返，实读存在）；L6（核内调度内联）引 CG-2026-0026/CG-2026-0010。无「把 I/O 契约当内部布局依据」；弃选代价机制（核内 UB 级转置、fixpipe 强制 GM 往返）与文档一致。本核为 Cube-only 无 Vector 轴，§1.6.3 明确豁免单轴评分并说明理由。
- **布局决策与实现一致 — pass**。选定 L1（`T.copy` 切片 + 自动 ND2NZ + `T.gemm(b_transpose=True)`）+ L5（epilogue 归 Cube `store_fixpipe` 动态 extent）；§3.3/§4（buffer 形状与预算）/§6（循环结构）与之一致；累加内层为 Cube K 维，无 Vector 二次处理。
- **与下游章节一致 — pass**。§3.1 公式拆解以 §1.6.1 优化后公式（含动态 $t_m/t_n/t_k$）为输入，第 5–7 步即动态 extent + `T.gemm(size=)` + 动态写回；§6 循环结构与 §1.6.2 结论一致，无未论证的标量循环。
- **建议级观察（不阻塞）**：§1.6.2「向量化结论」段文字写「仅保留 3 处 block 级标量点（第 5/6/7/8 项…）」，实际列举 4 项（5/6/7/8）。计数与枚举不符，属措辞小瑕，不影响结论（4 项均为合法保留点）。

## 检视总结

- **v2 专项复核通过**：把不可实现底座（调度内联 + Developer persistent）替换为可实现架构（前导 kernel 独立产 GM 调度表 + Expert 单 Cube scope 读表），① v1「融合单核」收益已整项撤回并重估；② 两函数对接契约显式化（§0.6 R1/§4.1）；③ Expert 单 Cube scope 无 in-block 跨引擎同步（规避 CG-2026-0026）；④ 四项保留结论（动态 extent、`T.gemm(size=)`、NT `b_transpose`、专家聚簇、B 访存下界）完好。
- **维度 0 亲自读源码核对**：语义/I/O/算法步骤/优化手段/耦合性/重设计/一致性/golden 独立性均与源码一致，无偏差。
- **9 个维度全部 pass，无阻塞级问题**；机械复核（verify_equiv 重跑 + design_calc_check）与文档内嵌表一致，skip 项已人工复核。
- 2 条建议级观察（K 循环 `T.serial` vs `T.Pipelined` 表述待 Stage 3 收敛；§1.6.2「3 处」应为「4 处」措辞），均不构成本阶段阻塞。
- 设计具备进入 Stage 3 的条件。R-1（两 kernel 固定开销）、R-2（Pipelined vs serial）、R-3（动态 extent 分形对齐）、R-4（L2 对 B 复用率）、R-5（调度表对接契约集成核对）、R-6（死 tile 比例）为已登记且有兜底的实现/性能风险，由 Stage 3/4 实测裁决。
