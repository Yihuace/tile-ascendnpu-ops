# `_sparse_mla_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md（修订版 revision 1）
- 任务类型: 迁移（scenario=migration, migration_mode=harness）
- 源算子: `/home/developer/TileOPs/tileops/kernels/attention/deepseek_dsa_decode.py`（函数 `_sparse_mla_kernel`，L17–401）；Op 类 `tileops/ops/attention/deepseek_dsa.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0（上一轮 1 阻塞 + 4 建议，本轮逐条复核均已真正修复）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**。执行 `python3 examples/deepseek_dsa_decode/_sparse_mla_kernel/verify_equiv.py`（torch CPU，cases=76，tol atol=rtol=1e-2），输出与 §1.6.1 内嵌结果表**逐项一致**，采纳集合全部 `EQUIV_PASS`，`OVERALL: EQUIV_PASS`（exit 0）：
  - fp16：OPT-1 `max_abs=4.883e-04 / ulp 531.00`；OPT-2 `9.766e-04 / ulp 3.04`；OPT-3 `0.0`；采纳集合 `9.766e-04 / ulp 530.00`；违反率均 `0.000e+00`。
  - bf16：OPT-1 `1.953e-03 / ulp 715.34`；OPT-2 `9.766e-04 / ulp 27.64`；OPT-3 `0.0`；采纳集合 `1.953e-03 / ulp 715.34`；违反率均 `0.000e+00`。
  - 被否决臂 `OPT-X fold scale into Q (REJECTED)`：fp16 `8.179e-03` PASS、**bf16 `9.766e-02 / 违反率 4.550e-02 → EQUIV_FAIL`** —— 与 §1.6.1 OPT-X 否决记录一致，机器诊断成立。
  - `[baseline vs fp64 truth]` fp16 `9.813e-04` / bf16 `7.807e-03`（sanity）；`[OPT-3 sentinel] PASS`；`[OPT-2 Q-rounding] max_rel 1.595e-02(fp16)/2.591e-03(bf16)`（非空臂）；`[OPT-R rejected] 1.192e-07`。**无采纳项 FAIL**。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design …/DESIGN.md`）JSON checks 摘要：
  - `l0c_budget`: **pass** — "block 64x64 x fp32 accum = 16.0 KB vs L0C 128 KB"（人工复核：主选 `acc_s(32×64)f32=8,192 + acc_o(32×512)f32=65,536 = 73,728 B = 72 KB ≤ 128 KB` ✓）。
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan"。**机械跳过，人工已核**：§4.5 分层表存在；L0C 72KB ✓、L1 184KB ✓、UB 物化态逐行求和 `32,768+65,536+65,536+9,728 = 173,568 B ≈ 169.5 KB ≤ 192 KB`（原始口径在限内）；设计已显式取两档膨胀（`~1.7x` ⇒ 288KB 超限不可编译 / `×1.10–1.12` ⇒ 190KB 贴限）并**把削减方案改为必需**（削减态 ≈41.5KB，×1.7≈71KB ✓）——不再有「小计 vs 所列行」不自洽。
  - `core_split`: **skip** — "insufficient parsed dims {M:32,N:64,bn:64}; declared logical cores = 1"（解析器未识出 `num_logical` 派生表达式）。**机械跳过，人工已核**：§5.5 三要素齐全（逻辑核 4096 = `B·S·(H/32)`、物理核 20 实查、非持久波次），且本轮**亲自实查** `NPUUtils.get().get_aicore_num() → 20`、`get_aivector_core_num() → 40`、`get_arch() → 'Ascend910B3'`（2026-10-15），与 §5.5 记录一致。

## 检视详情

### 0. 源算子理解与迁移分析 — **pass**

**已亲自 Read 源算子代码全文**（`deepseek_dsa_decode.py` L1–620），并以抽取副本 + manifest（`examples/TileOPs/tileops/manifest/attention.yaml` L136–191）交叉核对。逐项结论：

- **语义理解正确性 — pass**：§0.1 数学语义（`o[b,s,h,:]=Σ_j softmax_j((q·kv)/√(D+Dt))·kv[:D]`，`Dqk=D+Dt=576`）、规约语义（online over topk，累加顺序随块变）、dtype 语义（gemm fp32 累加 `accum_dtype="float"`（源码 L81）、softmax 全程 fp32、`P` 经 `T.copy(acc_s, s_shared)` 落回输入 dtype（L260））、边界语义（`idx>max_kv_i`→`-inf`→`exp=0`（L234-235/L247）；`m_i` init `-2**30` 避 `inf-inf`（L225））与源码一致。§0.2 I/O（q `(1,1024,128,576)`、kv `(1,2048,1,576)`、indices `(1,1024,1,2048)` int32、**输出 `(B,S,H,D)=(1,1024,128,512)`**）与源码 L105-108 及 manifest L156-159 逐一相符；`sm_scale=(1/√576)·LOG2E`（源码 L76）、`cp0=(q_start==0)=False`（§0.2）、去重语义差异（源逐条累加 vs 参考实现 `scatter` 去重；manifest `supply_prog` L583 `randperm` 保证索引互异）核对无误。
- **算法解读完整性 — pass（上一轮阻塞项已修复）**：§0.3 现**逐行引用源码行号并显式代入本任务实参**：源码 L78 `head_kv = heads // kv_group`；L111 `padded_h = max(next_pow2(head_kv),16)`；L130-137 `if head_kv>64: replicate_h = head_kv//64; h_per_block = padded_h if replicate_h==1 else 64`；L164-168 `T.Kernel((seq_len-kv_stride+1 if cp0 else seq_len)*replicate_h, batch, kv_group, threads)`；L204-210 `s_i = bx//replicate_h`、`h0 = g_i*padded_h + (bx % replicate_h)*64`。本任务 `heads=128`（manifest q_shape `[1,1024,128,576]`）、`kv_group=H_kv=1` ⇒ **`head_kv=128>64` ⇒ `replicate_h=2`、`h_per_block=64`、源 grid x = `S·replicate_h = 2048`**（§0.3 L123-129 结论与源码逐字一致）。v0 的「`replicate_h=1` / 全头单 block」误读已彻底更正，且经全局检索确认**无残留错误读数**（仅剩历史修订说明中作为 v0 勘误引用）。§0.3 步骤 1–13（含 host 侧逻辑、buffer 复用 `o_shared_l=q_shared_l`（L178-179）、三维 grid + 384 线程三段角色、warp-specialized 内层 `T.serial(ceildiv(n_i,2))`）覆盖源码全部计算语句，无遗漏/无臆造。
- **优化手段覆盖完整性 — pass（上一轮阻塞项已修复）**：§0.4 现列 **O1–O17**，新增 **O17「head 复制拆分（`replicate_h=head_kv//64`）」，机制注明为 Hopper register-pressure 驱动**（使单 block 寄存器态可容纳），并标注硬件耦合性（NPU 侧动机等价 = L0C/UB 容量）。O11 与 O17 的关系被明确区分（padding 恒等 ≠ head 拆分），不再以「padding 恒等」掩盖 head 拆分事实。O1–O16 原有识别完整、未把实现层优化误判为算法层。
- **耦合性判定合理性 — pass**：§0.5 每项四态处置（保留 / 等价替换 / 重新设计 / 舍弃）+ 依据；O17 已纳入（**重新设计 → R4**，依据 L0C 容量重定 `block_heads=32`）；O1/O4/O15/O16/`threads`/`is_hopper` 判「舍弃」均写明 NPU 无对应能力（单程序模型 / 无命名 mbarrier / 一维 Kernel / CUDA flags / host 门禁）；与 GPU→NPU 能力映射不矛盾。**无「不可判定」项**。
- **NPU 重设计可行性 — pass**：R1–R8 每项三要素齐全（源方案 → NPU 新算法 → 语义保持论证）：R1 `T.gemm`（K=576 一次 + PV 前缀列 `size=[BH,block_N,D]`，有 `docs/.../T.gemm.md` + `sparse_mla_fwd_dynamic_shape.py` `CubePVMatmul` 先例）；R2 单程序 Developer 自动 CV-split（`tilelang-mixcv-skill`，先例充分）；R3 `T.Pipelined(num_stages=2)` 替代 mbarrier；R4 L0C 预算 `(BH·(D+block_N)·4 ≤ 128KB ⇒ BH≤56)` 取 BH=32 / 承接 O17（`H%BH==0` 断言）；R5 单次 576 维 gemm（NPU 无 WGMMA K 维寄存器压力）；R6 一维 grid + `cid` 反解（非持久，避 `TRAP-DEVMODE-PERSIST-GEMM`）；R7 移除 `threads`；R8 clamp + 列掩码。重设计**未依赖三维 Kernel / 未照搬源硬件方案**，全部落在 NPU 约束内。**关键结构（L0C fragment 累加器 + 每块 Vector rescale + `T.gemm` 累加）有直接先例** `examples/deepseek_v4/example_sparse_mla_fwd_kernel.py`（`acc_o=alloc_fragment`、`T.vmul(acc_o, correction, acc_o)`（L135）、`T.gemm(scores_cast, KV_shared, acc_o, initC=False)`（L139）、`T.vdiv`（L141）），可行性成立。
- **设计与决策一致性 — pass（上一轮阻塞项已修复）**：§0.6 末段「源算法优化意图承接」逐项映射 O1→R2、O2→R1、O3/O4/O5→R3、O6→R4、**O17→R4**、O7→R5、O9→R8、O10/O12→OPT-1/OPT-3、保留项 O8/O11/O13/O14、舍弃项 O15/O16/`threads`/`is_hopper`——**无静默丢弃**。§1.4 第 1 点、§1.6.0 R3/R4（候选 0）、§3.5.2 差异表均已按更正后的 `replicate_h=2 / h_per_block=64 / 源寄存器态 2×[64,256]f32=128KB / 源 KV 遍历 2 遍·5.12GB` 重算，与 §0.6 R4/R6 自洽，**无「§0 判重设计、§3 照抄源方案」脱节**。
- **golden 独立性 — pass**：§8.1 golden 逐字移植 `examples/TileOPs/tests/ops/test_deepseek_dsa_decode.py::ref_program`（全头 `einsum` + `scatter` 去重 mask + `masked_fill(-inf).softmax` 两遍式），并显式列出 5 条独立要素（全头 / 去重 / 两遍式 softmax / 单次 einsum / 无 exp2·LOG2E 常数）——**结构上不复刻** §0.6 的 NPU 算法（head 分块 / 576 维单次 gemm / 自然 exp 域），独立验证成立。

### 1. API 可行性 — **pass**

- **API 存在性 — pass**：§3.2/§3.4 所列 `T.copy`（切片/索引 `size=`）、`T.gemm(src1,src2,dst,size,initC,a_transpose,b_transpose)`、`T.reduce_max/sum(dim=1,clear=)`、`T.vmax/vmul/vsub/vadd/vdiv/vcast/vbrc/vexp`、`T.Pipelined`、`T.Kernel(n,is_npu=True)`、`T.alloc_shared/alloc_fragment` 均在 `docs/Tilelang.language/` 有文档、在 `examples/` 有先例（`sparse_mla_fwd.py`、`example_sparse_mla_fwd_kernel.py`、`sparse_mla_fwd_dynamic_shape.py` L135 `size=[...]`）。`T.vexp.md` 文档已亲自打开核对（见下）。
- **API 与模式匹配 — pass**：Developer 自动同步（§7），无手动 `set_flag/wait_flag/sync_block_*`；`T.Kernel(..., is_npu=True)` 不带 `threads`（`TRAP-threads-kwarg-noop`）。
- **v-prefix 优先 — pass**：全部采用 v-prefix（`T.vexp/vmax/vsub/...`），未用 `npuir_xxx`。
- **上一轮建议 2（`T.vexp` bf16 论断）已修复并复核**：§1.6.1 OPT-1 收益栏、§0.5 O10、§3.5.2 exp 行、§9.1 R-13、§8.3 L0-2 均已更正为「**两 API 文档 dtype 矩阵（`T.vexp.md`/`T.vexp2.md` §2.2.1）均标 bf16 ✗**；本设计 softmax 的 exp 在 **fp32 域**执行（`acc_s` 累加、`tmp` 均 fp32），故**不触发**该缺口」，并引用 `TRAP-bf16-vop-doc-matrix-divergence`（文档矩阵与设备实测可能分歧，逐 op micro-probe 为准）。**亲自打开 `docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1 核对：fp16 √ / fp32 √ / bf16 × / fp64 ×** —— 与修订陈述一致，不再有「覆盖 bf16」的正向错误论断。

### 2. 内存层级规划 — **pass**

- **搬运路径完整 — pass**：§4.4 给出 GM→L1/UB→L0C 完整路径（Q/KV/P 操作数 L1；idx/状态/O_ub UB；acc_s/acc_o L0C），无 GM→L0 越级；无 `store_fixpipe` GM 往返。
- **L0C 容量 — pass**：`acc_s(32×64)f32=8KB + acc_o(32×512)f32=64KB = 72KB ≤ 128KB` ✓；`BH·(D+block_N)·4 ≤ 128KB ⇒ BH≤56`，取 32 —— 算术独立复算成立。
- **UB 预算 — pass（上一轮建议 3 已修复）**：§4.5 小计现**逐行求和统一为 `173,568 B ≈ 169.5 KB`**（`O_ub 32,768 + alpha_mat 65,536 + l_mat 65,536 + tmp/行状态/idx/pen 9,728`），独立复算一致（9,728 = tmp 8,192 + 8×32×4 行状态 1,024 + idx_ub 256 + pen 256）；原始口径 ≤ 192KB ✓。设计显式取 `CONST-capacity-910B2C` **两档膨胀**：`~1.7x` ⇒ **288KB > 192KB ⇒ 物化态不可编译**、`×1.10–1.12` ⇒ 190KB 贴限，并据此把**削减方案改为必需**（行广播不物化 `alpha_mat`/`l_mat`；削减态 ≈41.5KB，×1.7 ≈71KB ✓）。三处数字不自洽问题已消除。
- **L1 容量 — pass**：Q 36,864 + KV 147,456（双 stage）+ P 4,096 = 188,416 B ≈ 184KB ≤ 512KB ✓。
- **Buffer 层级标注 — pass**：§4.3 逐 buffer 标注 UB/L1/L0C。
- **备注（非阻塞）**：§3.3 设计级伪代码仍物化 `alpha_mat`/`l_mat`（`[BH,D]`），与 §4.5「削减方案为必需」存在措辞张力；但 §4.5/§9 R-2/R-3 已明确 Stage 3 落地形态（广播不物化，或 `T.Parallel` 带缓冲变量）且列为风险，不构成阻塞。

### 3. Tiling 策略 — **pass**

对照 `.agents/skills/_shared/standards/core-split-strategy.md §1/§2.1` 核对 §5.5 三要素：
- **① 逻辑核数计算**：`num_logical = B·S·(H/block_heads)` = 4096（mainstream）/512（smoke）/2048（longer-kv）/1024（long-kv），算术正确。
- **② 物理核数依据（实查）**：§5.5 记录 `NPUUtils.get().get_aicore_num() → 20`、`get_aivector_core_num() → 40`、`get_arch() → 'Ascend910B3'`（含查询代码）；混合算子直接取 20。**本轮亲自实查复核**（2026-10-15）：`get_aicore_num()=20`、`get_aivector_core_num()=40`、`get_arch()='Ascend910B3'` —— 与设计记录一致，符合「实查、禁止假设」。
- **③ 规模判定与分核方案**：`4096 ≫ 20` 判**极大规模**；标准第三档（核内 `T.serial` persistent）因 `TRAP-DEVMODE-PERSIST-GEMM`（verified，910B2C）不可用 → 走「无法对齐兜底」的**非持久一维 grid + 硬件波次**，`4096 = 20×204+16`（204.8 波次，残余 ≈1/205≈0.5%）、smoke `512=20×25+12`（≈4%），**显式披露偏差与核启动开销**，并给出备选 BH=64 把逻辑核降到 2048。
- **自洽性**：block 取值（BH=32/block_N=64）与核数、L0C/L1/UB 预算自洽；核内无依赖动态 shape 的串行循环；核内串行边界静态。（`design_calc_check` 的 `core_split` skip 已人工复核。）

### 4. 技术约束检测 — **pass**

- **三维 Kernel**：源为 3D，§3.5.1 明确改用一维 `T.Kernel(num_logical, is_npu=True)` + `cid` 反解（R6）。
- **GPU 专用 API**：逐一列出（`T.wgmma_gemm`/`T.wait_wgmma`/`T.alloc_barrier`/`cp_async_barrier_noinc`/`T.set_max_nreg`/`T.get_thread_binding`/`threads=`/CUDA `compile_flags`）并全部处置（§0.5 四态）。
- **GEMM 非整除**：不涉及（`block_heads|H`、`block_N|topk` 构造期断言，源同款 L119-125）；§3.5.1/§5.3/§5.4 说明。
- **L0C 溢出**：已在维度 2 处理（72KB ≤ 128KB）。

### 5. 循环与同步 — **pass**

- **循环结构 — pass**：§6 明确 `T.Kernel`（非持久 block 并行）/`T.Pipelined(topk//block_N, num_stages=2)`/`T.serial(block_N)`（gather 行）/`T.Parallel(BH,block_N)`（掩码）/`T.reduce_max/sum(dim=1)`；每类有选择理由。
- **同步与模式匹配 — pass**：Developer 自动同步，§7 无手动同步点（不使用 `set/wait_flag`/`sync_block_*`/`pipe_barrier`），与模式一致；并记录 `PL-1.16` 作为切 Expert 的判据。
- **尾块处理 — pass**：§6.4 说明无 shape 尾块（整除），`indices` 掩码尾由单次 clamp + 列掩码处理（数据有效性，非 shape 非整除）。
- **备注（非阻塞）**：§3.3 伪代码出现的 `tmp1_row` 未在 §4.3/§3.3 的 buffer 列表中显式声明（属设计级伪代码的临时行向量，Stage 3 按 §9 R-3 落地），不影响循环/同步结构判定。

### 6. 验证方案 — **pass**

- **Golden 函数 — pass**：§8.1 含 PyTorch 参考实现草案（移植源仓 `ref_program`），独立于 NPU 算法（维度 0 已核）。
- **L0 测试计划 — pass**：§8.3 列 6 例（smoke fp16/bf16、full mainstream/longer-kv、boundary 掩码、全掩码退化），规则 shape 满足整除契约，dtype 覆盖 fp16+bf16；bf16 用例已注明 exp 在 fp32 域执行（不触发文档矩阵缺口）。
- **精度标准 — pass**：§8.2 给出 `atol=3e-4 / rtol=1e-5`（源自 `test_deepseek_dsa_decode.py` L109 断言）及 fallback 1e-2，并如实标注「严于 fp16 单 ulp@O(1)」（R-7）。

### 7. 完整性与一致性 — **pass**

- **章节齐全 — pass**：11 章齐全；迁移类含 §0 且 0.1–0.7 小节齐全（§0.7 含源路径 / Op 类 / 参考实现 / 抽取副本 / NPU wrapper / 同族先例）。
- **无占位符 — pass**：全文未见模板占位符或未完成编辑标记（§10.2 的 `⬜ 待实现` 为 Stage 3 交付物状态列，非设计内容占位）。
- **同类引用真实 — pass**：§3.5.3 与 §1.6.0 引用的 7 个本仓先例均**经 Glob/ls 核对真实存在**：`examples/deepseek_v32/sparse_mla_fwd.py`、`sparse_mla_fwd_dynamic_shape.py`、**`sparse_mla_fwd_exp.py`（存在）**、`examples/deepseek_v4/example_sparse_mla_fwd_kernel.py`、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_fwd.py`、`examples/indexer/indexer_fwd.py`、`examples/mha_decode_paged/`；抽取副本与测试路径同样存在。
- **内部一致 — pass**：§0.3/§0.4/§0.5/§0.6/§1.4/§1.6.0/§3.5.2 的 `replicate_h/h_per_block/源寄存器态/KV 遍数` 已一致更正为源码事实；API 映射（§3.2）、内存规划（§4）、Tiling（§5）三处相互一致；§4.5 小计与所列行自洽。

### 8. 算法优化分析 — **pass**

- **§1.6 章节存在 — pass**：含 1.6.0（算法调研）/ 1.6.1（数学等价优化）/ 1.6.2（向量化替代分析）/ 1.6.3（向量化轴与数据布局决策）。
- **算法调研存在性与完整性 — pass**：§1.6.0 四问齐全——**R1** 候选表**含基线候选**（候选 0 = 源 online-softmax flash over topk；本任务 `replicate_h=2` ⇒ 整体 2 遍 + 5.12GB），并覆盖 `algorithm-candidates.md` 命中行（`ALG-attention` R1 的「scale 融入 Q/K」=OPT-X、「softmax 分母倒数乘法化」=OPT-R，均已评估；R2 flash；`ALG-softmax`「减 max / exp2·log2e / online」；`ALG-gemv`「K 分块累加」；`ALG-reduction`「分块归约+合并」）；**R2** 在线算法有**明确结论**（"有，且为基线所采用，且本问题必需"，结构判据 + 源码证据 L246-255 + 容量判据 1MB/行）；**R3** 复杂度**四口径**齐全（FLOPs / 访存 Bytes / KV 扫描遍数 / 中间缓冲峰值；基线在表，口径写清含 gather 实际字节 vs manifest 理想下界）；**R4** 硬件亲和逐候选评估（计算单元 / 片上容量 / 对齐整除 / 静态边界 / 流水融合）。「无更优替代」写明调研范围。
- **调研结论正确性（独立复核）— pass**：
  - **① 等价性**：OPT-1（换底 `exp2(x·log2e)=exp(x)`）、OPT-2（分配律重结合 + `m·s` CSE，与 OPT-1 合成）、OPT-3（`exp(-1e38)=0` 精确）独立推演成立，容差内；`verify_equiv.py` 重跑逐项一致（见机械复核）。
  - **② 复杂度独立复算**：FLOPs `2·B·S·H·topk·(2D+Dt) = 2·1024·128·2048·1088 = 5.84e11` ✓（与 manifest roofline L181 `2*B*S*H*topk*(2*dim+dim_tail)` 同式）；gather/遍 `1·1024·2048·576·2 = 2.42G` ✓，主选 `×4 = 9.66G` ✓；Q `151M`、idx `8.4M`、O `134M` ✓；主选总量 `≈9.95GB` ✓；**源候选 0 已按更正后 `replicate_h=2` 重算为 `2.42G×2 = 4.83G ⇒ 5.12GB / 2 遍`** ✓（v0 的 `[128,512]f32=256KB` 与 1 遍读数均已消除）。roofline 流量项 `9.95GB / 1.06TB/s ≈ 9.4ms`、`/2.49TB/s ≈ 4.0ms` ⇒ `[4.0, 9.4]ms` ✓；中间缓冲**未漏计 GM 往返**（主选无 store_fixpipe GM 往返，容量项 ≈0）。
  - **③ 负向断言主动复核**：对照 `algorithm-candidates.md` 的 `ALG-attention`/`ALG-softmax`/`ALG-gemv`/`ALG-reduction` 命中行逐条核对——源算法**已用 online**（与"存在在线变体"一致，非矛盾）；候选 1（两遍 materialize）因 `S=[128,2048]f32=1MB/行 ≫ UB/L0C` 容量不可行 + 遍数↑；候选 3（workspace 物化）访存 14.5GB 最大；**未发现被漏掉的已知候选**（softmax 族 online 变体、attention flash 结构均已评估并选定）。负向断言成立。
  - **④ 并列候选裁决**：主选 BH=32 / 备选 BH=64 属并列（`block_heads` = KV 重读倍数 ↔ L0C/UB 容量的权衡），**有 §1.6.3 实验裁决三件套**（主选结构 + 备选达 Stage 4 深度（buffer 形状 / 循环结构 / 分核三要素按 BH=64 重算 / 膨胀阈值 1.26x 分支）+ 裁决计划含代表 shape / 指标（msprof Task Duration + 反解 gather 带宽）/ 判定阈值 0.90 / 回写路径 / shape 分派评估）——不纸面拍板。
- **调研→设计一致性 — pass**：§1.4 算法描述、§3.1 公式拆解（优化后公式）、§6 循环结构与 §1.6.0 选定算法（head 分块 + online flash 单遍分块）**同一口径**；§3.1 以 §1.6.1 优化后公式为输入 ✓；§6 无被判向量化仍留标量循环的点 ✓；迁移侧 §0.5/§0.6 处置与调研选族一致（WGMMA→`T.gemm`、mbarrier→`T.Pipelined`、head 复制拆分→R4 均在 §1.4/§3–§7 落地），源优化意图 O1–O17 逐项承接、**无静默丢弃**。
- **数学等价优化完整性 — pass**：OPT-1/2/3 四要素齐全（原式 → 优化后公式 → 等价性论证 → 收益量化，如 OPT-2「每块省 1 个 `[BH,block_N]` vmul ⇒ 每任务省 32×[32,64]=65,536 次乘」）；无优化空间项（OPT-R 归一化乘倒数、OPT-9 索引去重）有结论与依据。
- **等价性论证正确性 — pass**：逐项独立推演成立，与 §8.2 精度标准一致；被否决 OPT-X 的 bf16 违反率 4.55% 有独立机器臂佐证。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 16 个计算点（点 1–10 已向量化、点 11/16 n/a、点 12–15 保留），**与 §3.3 伪代码、§6 循环结构交叉核对无遗漏**（§3.3/§6 中不存在 §1.6.2 未涵盖的逐元素标量循环）。
- **不可替代理由充分性 — pass**：保留点理由具体到本算子——点 12（KV 行 gather：数据依赖行地址 + 行连续 DMA，NPU 缺跨步 gather 向量指令 `TRAP-C10`，可用形态 = 串行 `T.copy` 连续行，先例 `example_sparse_mla_fwd_kernel.py` L115-118）；点 13（block 级单次 clamp，`TRAP-scalar-min-chain-offset` 推荐形态）；点 14（block 级 `cid` 反解）；点 15（tile 级 online 递推，`T.Pipelined` 是调度原语非逐元素循环）。**无「实现简单 / 照源码写」类理由**。
- **替代 API 可行性 — pass**：替代 API 均有 `docs/` 或 `examples/` 佐证；`T.vexp` 的 bf16 论断已更正（见维度 1）。
- **向量化轴与布局决策完整性 — pass（上一轮建议 4 已修复）**：§1.6.3 含轴质量评分表（整除性 / 尾 lane 浪费 / 累加链形态 / repack / UB 影响）+ **5 候选矩阵 + 量化弃选理由**（#2 K/V 拆缓冲 gather +89%、#3 `T.transpose` 无收益（UB 级 ~µs）、#4 host permute 净亏 ~百 µs（`PL-1.3`）+ 违反 host 只读、#5 BH=64 备选）。**新增 softmax 归约 lane 映射候选表**（L1 水平归约 `T.reduce_*(dim=1)`（block_N=64=8×8，0% 尾 lane 浪费，无 repack）vs **L2 垂直扫描**（64 步串行链 × 发射 ×64 + 列主序 repack，净亏）→ 选定 L1），与 §1.6.2 点 2/6、§3.3/§6 的 `T.reduce_*(dim=1)` 互相印证。MixCV 类别已覆盖（`b_transpose` 即时消费 vs 显式转置）。弃选前提经核对（`T.transpose` 核内 UB 级、host permute ~百 µs 均与 `docs/`/pattern-library 实测一致）。
- **实验裁决模式完整性 — pass**：判定依赖两个未实证常数（910B3 核→GM/L2 有效读带宽、KV cache L2 驻留程度），三件套齐全：主选 BH=32（进 Stage 3）、备选 BH=64（结构达 Stage 4 深度 + 膨胀阈值 `≤1.26x` 判定分支）、裁决计划（代表 shape / 指标 / 阈值 0.90 / 回写路径 / shape 特化分派评估）。
- **布局决策与实现一致 — pass**：主选 #1（I/O 原生 BSHD、KV tile `[block_N, Dqk]` 行连续、向量轴 `block_N`/`D`、无 repack）与 §3.3 伪代码、§4 内存规划（buffer 形状与预算）、§6 循环结构三方一致。
- **与下游章节一致 — pass**：§3.1 使用 §1.6.1 优化后公式；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）。
