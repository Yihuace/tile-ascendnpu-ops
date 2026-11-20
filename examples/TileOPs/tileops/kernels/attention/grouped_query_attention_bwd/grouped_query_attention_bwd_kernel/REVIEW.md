# `_gqa_bwd_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/grouped_query_attention_bwd/_gqa_bwd_kernel/DESIGN.md（修订版 revision_index=1，v0 备份 `history_version/design_v0.md`）
- 任务类型: 迁移（scenario=migration / migration_mode=harness / 函数 3/3）
- 源算子: /home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py（`_gqa_bwd_kernel` L514–621 / `GQABwdKernel` L622–665）；Op 编排 `tileops/ops/attention/gqa.py` L1330–1389
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4（均为非阻塞残余项，见「检视附注」）

> 本轮为**修订后重检**。上一轮（1 阻塞 + 5 建议，见 history 中的 v0 REVIEW）已逐条回源码复核修复效果：**6 项全部真正修复**（阻塞项 1；建议项 2–6）。复核方式：亲自 Read 源码 `gqa_bwd.py` L480–665（含 L521–526 常量、L575–621 主体、L649–656 default_config）、`indexer_bwd.py`、`sparse_mla_bwd.py`，并亲自打开 `docs/Tilelang.language/` 相关 API 文档。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/grouped_query_attention_bwd/_gqa_bwd_kernel/verify_equiv.py` —— **exit 0**。10 个用例（fp16×5 + bf16×5）判定量 `A_vs_B`（掩码融合）与 `C_vs_B`（掩码融合 + 区间裁剪）**全部 = 0.000e+00**，全部 `EQUIV_PASS`，**未通过项计数 = 0**，`finite_ok=True`（band：fp16 1e-05 / bf16 1e-04）。执行输出与 DESIGN.md §1.6.1 内嵌结果表**逐行一致**（含 `base~fp64` 信息量列：7.419e-05 / 1.028e-03 / 3.521e-04 / 1.074e-04 / 4.289e-05 / 6.727e-04 / 2.175e-05 等）。脚本候选式已使用实现常数 `NEG=-3.0e38`（`verify_equiv.py` L40–41 含 `assert NEG == -3.0e38`），与 §3.2/§3.3 实现逐位一致（上一轮问题 5 已修复）。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/grouped_query_attention_bwd/_gqa_bwd_kernel/DESIGN.md`）JSON checks 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed…")——**人工复核结论**：§4.5 表内各行显式加和 = `16384+16384+8192+8192+256+8192+8192+8192+8192+16384+8192 = 106752 B ≈ 104.25 KB`（含 `T.vexp2` 的 `tmp`）；不含 `tmp` = `98560 B ≈ 96.25 KB`。两者均 < 192 KB；按 x1.7 自动多缓冲口径 `104.25×1.7 = 177.2 KB < 192 KB` 亦成立。**容量结论成立**（上一轮问题 2 已订正）。skip 原因为表首列反引号包名（`` `k_s` ``）致解析器判为 header。
  - `l0c_budget`: **pass**（解析为 `64×64×fp32 = 16.0 KB ≤ 128 KB`；与 §4.5 独立口径「五累加器 `dk_f+dv_f+qkt_f+dp_f+dq_f = 24576 ≤ 32768` 元素」并行成立）。
  - `core_split`: **fail → 判定为解析器假阳性（人工已核，非算术错误）**。checker 用 2D 启发式 `ceil(M/bm)×ceil(N/bn)`，对**一维展平 grid** 不适用：其 `M=128` 误取自 §5.4 边界句「`seq_len in {128,512,2048}`」、`bn=64` 误取自 §5.2 的 `BN=64`、`declared` 被 `逻辑核数[^\d]*(\d+)` 误匹配为 `0`（文档从未声明 0）。**人工按 §5.5 逐 workload 复算** `num_logical_kernels = B·H_kv·ceil(S/BM)`：`4 / 256 / 512 / 128 / 256` 与文档声明**完全一致**；物理核 20、`512/20 = 25.6` 波（临界 26 波，+1.6%）复算一致 ⇒ 维度 3 判 **pass**（详见维度 3）。
  - `r3_metrics`: **pass**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值四口径均在 R3 表）。
  > 说明：机械复核为「可机检子集」，与维度 8 的独立推演互不替代；`core_split` 的 fail 经人工复核确认为**公式不适用**的解析假阳性（与 v0 REVIEW 判定一致），`ub_budget` 的 skip 已人工复算。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源码 `gqa_bwd.py` L480–665（含 `make_dq_layout` 被引处、`_gqa_bwd_func`/`_gqa_bwd_main` 主体 L533–621、`GQABwdKernel` L622–665）与 Op 编排，逐项比对 DESIGN §0。

- **语义理解正确性 — pass**（源码逐行核对）：
  - `sm_scale = (1.0/dim)**0.5`（源码 L521）、`scale = (1.0/dim)**0.5 * LOG2E`（L522）——§0.1 「`scale` L522 / `sm_scale` L521」行号**已订正且正确**（上一轮问题 6(i) 已修复）；`heads % heads_kv` 校验 L523、`groups = heads // heads_kv` L525 与 §0.1 一致。
  - `qkt = K Q^T`：源码 L586 `T.gemm(k_shared, q_frag, qkt, transpose_B=True)`（`k_shared[bm,D]` KV 行、`q_frag[bn,D]` Q 行）↔ §0.1 `qkt[i,j]=Σ_d k[kv0+i]·q[q0+j]` ✓（`i`=KV 位置、`j`=Q 位置）。
  - `P = exp2(qkt*scale − lse)`：源码 L588–589 `qkt[i,j]=T.exp2(qkt[i,j]*scale - lse_shared[j])` ✓（L587 读 `lse[bz,bx,...]`）。
  - `dp = V dO^T`（L596–597）、`ds = P⊙(dp−delta)*sm_scale`（L603–604）、`dv += P^T dO`（L599 `gemm(qkt_cast, do_shared, dv_frag)`）、`dk += ds^T Q`（L605 `gemm(dst_cast, q_frag, dk_frag)`）、`dq = ds K`（L608–609 `gemm(dst_shared, k_shared, dq_frag, transpose_A=True)`）↔ §0.1 五式**逐条一致** ✓。
  - **I/O 契约**（§0.1 表/§0.2）：源 L535–548 声明 `q/do:(B,S,H,D)`、`k/v:(B,S,H_kv,D)`、`lse/delta:(B,H,S)` fp32、`dq:(B,S,H,D)`/`dk/dv:(B,S,H_kv,D)` `accum_dtype="float"`（L526）✓；`@tilelang.jit` L528–533 **无 `out_idx`**、`dq/dk/dv` in-place ✓（与 §0.1/§4.7 一致）。
  - **dtype 语义**：`accum_dtype="float"`（L526）；P/ds 显式降 dtype 再喂 Cube（`qkt_cast` L598、`dst_cast` L604/607）↔ §0.1/§0.4 #13 ✓。
  - **累加语义**：`dk/dv` 核内寄存器跨 q-block 累加（L599/605）+ 跨 group head 原子（L613/618）；`dq` 核内跨 q-block 累加 + 跨 kv block 原子（L611）↔ §0.1 ✓。
  - **边界语义**：`loop_st = T.floordiv(by*block_m, block_n) if is_causal else 0`（源码 **L580**，非设计所写 L581——见检视附注）；block 内 `if is_causal:` 才施加掩码（L590–593）；`S_q==S_kv`；`heads % heads_kv != 0` raise（L523）↔ §0.1 ✓。
  - **host 侧**：`default_config = {block_m:64, block_n:64 if dim<=64 else 32, num_stages:1, threads:128}`（L649–656）、`forward(*inputs)` in-place（L662–663）↔ §0.3/§0.5 #16 ✓。
- **算法解读完整性 — pass**：§0.3 步骤 1–23 覆盖源码 L540–621 的全部计算语句（含三处 `T.atomic_add` L611/613/618、`annotate_layout` L569–573、`T.clear` L577–578/595/608），数据流图覆盖全部 buffer。**次要遗漏（非阻塞）**：源码 L585 `T.clear(qkt)` 未单列于 §0.3 表（设计 §3.3 用 `initC=True` 已等效覆盖，不影响语义），见检视附注。
- **优化手段覆盖完整性 — pass**：§0.4 的 17 项逐项与源码对照成立——`T.GemmWarpPolicy.FullRow` 现为 **4 处（L586/597/599/605）**、L609 的 `dq` gemm 无 `policy`（上一轮问题 6(iii) 已订正，源码 L586/597/599/605/609 亲核一致）；`make_dq_layout`+`annotate_layout`（L569–571）、`make_swizzled_layout`（L571–572）、`loop_st`（L580）、`exp2`+log2 域（L521–522）、`sm_scale` 折进 ds（L604）、P/ds 显式 cast（L598/604/607）、`T.Pipelined`（L583）、`threads`（L545）、`compile_flags`（L532）/`pass_configs`（L531）、`lru_cache`（L514）均被正确识别，GPU 专属项未被误列保留。
- **耦合性判定合理性 — pass（抽检 + 文档亲核）**：四态处置逐条有依据——
  - `T.gemm` 五 gemm：**等价替换**，`docs/Tilelang.language/线性代数操作/T.gemm.md` L8 签名确为 `T.gemm(src1,src2,dst,size=[],initC=False,a_transpose=False,b_transpose=False)`，**无 `policy`**（亲核 ✓，支持 §0.5 #5 舍弃 `FullRow`）；
  - `T.atomic_add`：**能力保留 + 结构重设计**，`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.1 表亲核 Ascend `fp16/fp32/bf16` 均 √、§3 降级 `hivm::StoreOp`、§2.4 `size=` 形态示例 ✓；
  - 三维 grid → 重新设计（一维）、`make_dq_layout`/`make_swizzled_layout` → 舍弃（注释 L47 原文确证仅服务 Ampere `atomicAdd` 向量化）、`threads`/`compile_flags`/`pass_configs` → 舍弃；
  - §0.5 统计口径**已订正**：保留 7 项（语义 + #3/#10/#11/#12/#13/#17）、等价替换 3 项（#1/#4/**#9**）、重新设计 2 项（#2、#6）、舍弃 6 项（#5/#7/#8/#14/#15/#16）；#9 **不再重复计数**（上一轮问题 6(ii) 已修复）。
- **NPU 重设计可行性 — pass**：R1/R2/R3/R4 均有「源方案 → NPU 新算法 → 语义保持论证」三要素。
  - **R1（GQA head 合并 + grid 首维改 KV head + 数学等价性）** 独立推演成立：新 grid `nblk = B·H_kv·n_kv`，核内 `for g in range(groups)` × `T.Pipelined(loop_st, n_q)` 覆盖「全部 q 位置 × group 内全部 head」；`dk/dv` 求和集合 = 源「寄存器累加 + 跨 group 原子汇总」的求和集合，且每元素**唯一核写入、零原子**（写序确定：`g` 升序、`i` 升序，fp32 下不劣于源）；`dq` 仍跨 kv block 原子加，求和集合与源一致。边界 `H % H_kv == 0` 保证 `h = kh*groups+g` 恰枚举全部 q_head（源 L523 已 raise 非整除）。
  - **R2（tile 重算）**：D=64→BN=64 / D=128→BN=32 与源 `default_config`（L653）一致；`kv0=jb*BM`、`q0=i*BN` 一切片覆盖全部位置（`S%BM==0`、`S%BN==0`）。
  - **R3（归约结构）**：`dq` 改为 `T.serial(BN)` 行循环 + D 段整宽原子；`dk/dv` 单次 `T.copy` 写。语义保持论证成立（唯一写入 ⇒ 确定顺序；不改变任何求和集合）。
  - **R4（无 `annotate_layout`）**：`T.Layout` 只改排列不改数学内容；NPU 无 warp-fragment、`T.atomic_add` 具备 `size=` 形态 ✓。
- **设计与决策一致性 — pass（上一轮阻塞项已真正修复）**：源码 L590 `if is_causal:` 的**条件守卫现已在 DESIGN 全链路复现**——§3.3 L596–604（`if is_causal:` 掩码融合 / `else:` 无掩码）+ L579 `loop_st = (kv0 // BN) if is_causal else 0`；§1.4 步骤 4、§1.6.1 #1、§3.1 步骤 5、§3.2 步骤 5、§6.2、§8.2、§9.2、§0.6 R1 均同步。两分支核对：
  - `is_causal=True`：扫描 `i ∈ [kv0//BN, n_q)` + 块内 `where(kv0+ii<=q0+jj, x, NEG)` —— 与源码 L580+L590–593 逐元素一致 ✓；
  - `is_causal=False`：`loop_st=0` 全区间 + **不施加掩码** `qkt_ub = qkt_ub*scale − lse_s` —— 与源码 `else 0` 分支及 L590 守卫一致 ✓。
  §0.1/§8.1 golden（`if is_causal:` 才 `torch.where`）与本设计三处一致，不再自相矛盾。
- **golden 独立性 — pass**：§8.1 主路径 `golden_gqa_bwd` 按 §0.1 语义**逐 head 显式公式**实现（`torch.matmul`/`torch.exp2`/`torch.where`/`torch.tril`），`delta = torch.sum(of*dof, dim=-1)` 由 `o⊙do` 行和**独立算出**，未复刻 NPU 数据流（无 `for g in range(groups)` 核内展开、无行段原子）；交叉路径 `golden_gqa_bwd_sdpa` 走 `F.scaled_dot_product_attention` 反向互证。函数体内直接出现 `torch.*` 调用 ✓；
  独立推演 golden 公式：`qk=[S,D]@[D,S]` 行=Q、列=KV；`tril`（`i>=j` ⇒ `q>=kv`）保留 `q>=kv`、置零 `q<kv`，与源码「`kv0+i>q0+j` → 0」一致；`dv += p^T@do`、`dk += ds^T@q`、`dq += ds@k` 形状与源码三 gemm 一致 ✓。

### 1. API 可行性 — pass

- §3.2/§3.4 列出的 API **全部有真实佐证**：`T.Kernel(n, is_npu=True)` 一维（`indexer_bwd.py` L41、`sparse_mla_bwd.py` L155）、`T.Pipelined(start,stop,num_stages=)`（`sparse_mla_bwd.py` L199）、`T.gemm(...,initC=,a_transpose=,b_transpose=)`（`T.gemm.md` L8，亲核无 `policy`）、`T.atomic_add(dst,src,[size])`（`T.atomic_add.md` §2.2.1/§2.4）、`T.vcast(...,round_mode="rint")`（`indexer_bwd.py` L102、`docs/Tilelang.language/数据类型转换操作/`）、`T.Parallel`+`T.if_then_else`、`T.serial`、slice 形态 `T.copy`（`_sparse_mla_kernel.py` L221）。
- **`T.vexp2` 主选（上一轮问题 4 已修复并复核）**：`docs/Tilelang.language/数学操作/T.vexp2.md` **存在**，签名 `T.vexp2(input, output, tmpBuffer)`，三 tensor 形状一致，fp16/fp32 支持；`examples/deepseek_v32/sparse_mla_bwd.py` **L234** 确为 `T.vexp2(acc_p, acc_p, tmp)`（同族 flash-attn 反向，in-place 形态先例）；`docs/Tilelang.language/数学操作/` **无 `T.exp2.md`**（亲核 `ls` 确认，仅有 `T.vexp.md`/`T.vexp2.md`）——§3.4 的「docs 无 `T.exp2.md`」论断成立。§3.3 已补 `tmp: T.alloc_shared([BM,BN], accum_dtype)` 并计入 §4.5。
- v-prefix 优先：设计给出 v 形态等价链（`T.vmul/T.vsub/T.vexp2/T.vcast/T.vbrc`），符合 AGENTS.md「新设计优先 v-prefix」。Developer 模式用 `T.Parallel`/自动同步，无手动 flag ✓。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：`GM→UB→L1(cbuf)→L0A/L0B→L0C→UB→GM`（+ 原子写），无 GM→L0 越级；每个 buffer 标注层级（§4.3）✓。
- **UB 预算（独立复算，上一轮问题 2 已订正）**：D=128/BM=64/BN=32 逐行 `k_s 16384 + v_s 16384 + q_s 8192 + do_s 8192 + lse_s/delta_s 256 + qkt_ub 8192 + dp_ub 8192 + ds_ub 8192 + tmp 8192 + p16/ds16 8192 + dq_ub 16384 = 106752 B ≈ 104.25 KB`（含 `tmp`）；不含 `tmp` = `98560 B ≈ 96.25 KB`。文中「小计 106752 ≈ 104.25 KB」「不含 tmp 98560 ≈ 96.25 KB」与表内加和**一致**。x1.7 口径 `104.25×1.7 = 177.2 KB < 192 KB` ✓。D=64 情形复算 `131584 B ≈ 128.5 KB` ✓。
- **L0C 容量（独立复算）**：`dk_f+dv_f = 2·64·128 = 16384`、`qkt_f+dp_f = 2·64·32 = 4096`、`dq_f = 32·128 = 4096` ⇒ **24576 ≤ 32768** 元素（75%）✓；D=64 时 20480 ✓。D=128 取 `BN=32`（取 64 则 32768 = 100% 无余量）的推理成立。

### 3. Tiling 策略 — pass

- Block 划分明确（§5.2）：`BM=64`（L0C/整除/可比性）、`BN=64(D≤64)/32(D=128)`（L0C 反推）✓；对齐（§5.3）D=64/128 行宽 128/256B ≥ 32B、fp16 x8 / fp32 x4 整除 ✓；非整除（§5.4）manifest `S∈{128,512,2048}` 被 64/32 整除 ⇒ 无尾块，越界给 `gcd` 静态兜底 ✓。
- **分核策略三要素（对照 `core-split-strategy.md` §2.1/§2.2）齐全且自洽**：
  - ① 逻辑核数：`num_logical_kernels = B·H_kv·ceil(S/BM)`，逐 workload **独立复算与声明完全一致**：`[1,128,8,64]/H_kv=2 → 4`、`[4,512,32,128]/H_kv=8 → 256`、`[2,2048,32,128] → 512`、`[2,512,64,128] → 128`、`[1,2048,64,128] → 256` ✓；
  - ② 物理核数依据：`NPUUtils.get().get_aicore_num()` **实查 = 20**（记录查询代码与返回值，设备 Ascend910B3），本算子含 Cube ⇒ 按标准 §1 要素②「Cube/混合直接用返回值」**不翻倍** ✓；
  - ③ 规模判定与方案：判**极大规模**（512 = 25.6×20），因 `TRAP-DEVMODE-PERSIST-GEMM` 禁核内 persistent+gemm，走标准 §1「**无法对齐兜底**」——非持久一维 grid + 硬件多波次，**显式记录残留**（512 非 20 整数倍，临界 26 波 vs 均值 25.6，+1.6%）+ causal 三角负载不均（主导项）缓解（`jb'=n_kv-1-jb` 逆序优先调度）✓；核内循环边界（`groups`/`n_q`/`T.serial(BN)`）均编译期静态，唯一运行期标量 `loop_st` 仅作 `T.Pipelined` 下界、非数据依赖动态边界 ✓。
- 机械 `core_split` fail 为解析器假阳性（2D 公式 + `declared=0` 误匹配，对 1D 展平 grid 不适用），已在机械复核段说明并人工按一维 grid 复算通过。

### 4. 技术约束检测 — pass

- 三维 Kernel：源 3D grid（L550–551）已展开为一维 + `(bz,kh,jb)` 静态分解（§0.6 R1/§3.5.1）✓。
- GEMM 非整除 / L0C 溢出：§5.4 无尾块、§4.5/§5.3/§9.2 给出 `BN=32` 防溢出（`cc overflow` 对策）✓。
- 陷阱清单（§9.2）覆盖 `TRAP-DEVMODE-PERSIST-GEMM`、`TRAP-C10-strided-gather`（禁 `T.load_nd2nz`）、`TRAP-if_then_else-runtime-scalar-cmp`、`TRAP-loopvariant-vop-mask`、隐式跨 dtype `T.copy` 舍入、原子目标跨步被当连续（静默写错）✓。

### 5. 循环与同步 — pass

- 循环结构（§6.1/§6.2）：Python `for g in range(groups)` 静态展开 + `T.Pipelined(loop_st, n_q, num_stages=1)`（q-block 扫描，嵌套于 group 循环内）+ `T.serial(BN)`（`dq` 行原子，不含 gemm）；`T.Persistent` 主动规避 ✓。上一轮的掩码条件化 warn 已随问题 1 修复（§6.2 现含 `if is_causal:/else:` 两分支）✓。
- 同步（§7）：Developer 自动同步；跨核仅 `dq` 的 GM 原子（无需跨核 flag/workspace 握手）；`cv_split` 由编译器插同步，与 `sparse_mla_bwd.py`/`indexer_bwd.py` 先例一致 ✓。
- 尾块（§6.4）：无尾块，切片恒在界内；`nblk` 余数由硬件分波吸收 ✓。

### 6. 验证方案 — pass

- Golden：§8.1 含显式逐 head 公式 + SDPA 两条互证路径，均直接 `torch.*`、独立算 `delta` ✓（见维度 0 golden 独立性）。
- L0 门槛测试计划（§8.2）：覆盖全部 manifest workloads、GQA `groups=1/2/4/8` 专项、边界（`seq_len<BM`/非 2 幂/`is_causal=False` 显式对拍 smoke `(1,1024,8,4,64,False,...)`）、数值边界、`dq` 原子归约 `n_kv=1 vs >1` 对拍 ✓。
- 精度标准按 dtype 分档：fp16 → atol 1e-3/rtol 1e-4、bf16 → 5e-3/1e-3，并说明与源端到端 `atol=5e-3/rtol=1e-5`（cast 回输入 dtype 后）的关系 ✓。

### 7. 完整性与一致性 — warn

- 章节齐全：§0（0.1–0.7）+ §1–§12 齐全，**无未填充占位符**（全文扫描无命中）✓。
- §3.5.3 引用的 `examples/` 路径**全部真实存在**（`examples/deepseek_v32/sparse_mla_bwd.py`、`examples/indexer/indexer_bwd.py`、`.../multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`.../deepseek_dsa_decode_kernel/_sparse_mla_kernel.py`、`examples/flash_attention/flash_attn_npuir_dev.py`、`examples/grouped_query_attention_bwd/_flashattn_bwd_preprocess_kernel/DESIGN.md`，均已 `ls` 亲核）✓；§3.2 API 佐证文件（`T.gemm.md`/`T.atomic_add.md`/`T.vexp2.md`）存在 ✓。
- 内部一致性：上一轮的掩码/UB 小计/统计三处偏差**已全部消除**；§3.1/§3.2/§4/§6 相互一致 ✓。
- **warn（残余引用精度，非阻塞，见检视附注）**：部分 `examples/indexer/indexer_bwd.py` 的行号引用为旧修订（`L109–111`/`L106`/`L100`/`L53` 实为 `L117–119`/`L117`/`L102`/`L41`，同一先例形态真实存在）；源码 `loop_st` 在 §0.1/§0.3/§0.4 标为 `L581`（实为 `L580`）；§5.5/§9.3 的 causal 逆序映射（`jb'=n_kv-1-jb`）未落入 §3.3/§6.2 伪代码。以上仅为引用/文档一致性的细节偏差，**不影响 Stage 3 可实现性与正确性**。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局决策四子节齐全。
- **调研存在性与完整性 — pass**：R1 候选表**含基线 #0**（源算法）且列 8 个候选（#0 基线、#1 GQA 合并、#2 两遍、#3 q 维网格、#4 代数展开、#5 sm_scale 外提、#6 store-attention、#7 causal 剪枝、#8 掩码融合）；R2 给出**明确「无在线算法」结论 + 结构依据**；R3 表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径说明完整）；R4 逐候选六维评估；调研范围明确（`algorithm-candidates.md` 的 `ALG-attention`/`ALG-softmax` + pattern-library + examples），并声明未用互联网检索。
- **调研结论正确性（独立复核）— pass**：
  - **R2 负向断言「无在线变体」复核成立**：对照 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-attention` 条（第 35–41 行，R2 原文为「flash 结构——online softmax + 分块 KV」）与 `ALG-softmax` 条（第 8–14 行，R2 为「online softmax（分块 running max/sum + 重缩放）」），在线变体的**前提是存在跨 tile running 统计量并需重缩放**；本反向主核的 `lse`（log2 域行 logsumexp）与 `delta` 已由 preprocess **预先算好并作为输入**（源码 L544–545 `lse`/`delta` 入参、`gqa.py` L1377 `delta = self.prep_kernel(o, do)`），`P=exp2(qkt*scale−lse)` 无 running 量、`ds`/`dv/dk/dq` 为无重缩放纯求和 ⇒ `ALG-attention` 的 online 属**前向**结构，不适用于本反向核。**负向断言未被推翻** ✓。候选库条目名 `ALG-attention` **已订正为实际条目名**（上一轮问题 6(iv) 已修复）。
  - **复杂度算术独立复算一致**：`10·B·H·S²·D`（=5 gemm × `2·B·H·S²·D`；causal /2）；代表 shape `[2,2048,32,128]` fp16（H_kv=8）：`10·2·32·2048²·128/2 = 1.718e11 ≈ 171.8 GFLOP` → `/112 TFLOP/s ≈ 1.53 ms`；访存 `B(3H+4H_kv)·S·D·e = 2·128·2048·128·2 = 134.2 MB` → `/2.49 TB/s ≈ 53.9 µs`；`dq` 原子元素写 `B·H·S·D·(S/BM) = 2·32·2048·128·32 = 5.37e8` ×4B = **2.15 GB** → `≈0.86 ms`；容量 UB 104.25 KB < 192KB、L0C 24576 ≤ 32768。**均与设计数字一致** ✓（未见中间缓冲 GM 往返漏计——本算法无中间 GM 暂存，原子流量单列）。
  - **亲和性淘汰依据核对**：`#6 store-attention` 的 `2·B·H·S²·e`（S=2048 → +6.9 GB）远超本体 134.2 MB，否决成立；容量/带宽淘汰引用 `CONST-capacity-910B2C`（`constants.md` L8）/`CONST-cube-gemm-l0c-traffic-floor`（L218）/`CONST-elu-910B3-bw-floor`（L325，2.49 TB/s）**均亲核存在** ✓。
  - **并列候选裁决路径**：#2（两遍零原子）为**结构完整备选**（buffer/循环/分核三要素），其取舍依赖未实测常数（原子与 Cube 重叠率），交 §9.3 实验裁决（代表 shape、msprof `TaskDur`、翻转阈值 0.95、回写路径），**未纸面拍板** ✓。
  - **#3 原子量倍数独立重推（上一轮问题 3 已修复）**：`#3 dk/dv 原子 ≈ 2·B·H_kv·S·D·groups·(S/BN)`、`#1 dq 原子 ≈ B·H·S·D·(S/BM)`，比值 `= 2·(H_kv/H)·groups·(BM/BN) = 2·BM/BN`（`H=H_kv·groups` 消元）= **4**（BM=64/BN=32）。设计现给出同式推导并标 **4x**，复算成立 ✓（v0 的 16x 系多乘 `groups`）。
- **调研→设计一致性 — pass**：选定族 #1（+采纳 #7/#8）实际驱动 §1.4 算法描述、§3.1 公式拆解（输入为 §1.6.1 优化后公式）、§6 循环结构；迁移任务 §0.5/§0.6 处置与调研选族一致；源优化手段（#7 `loop_st`、#8 掩码、#4 gemm、#6 atomic 结构重设计、#1 tiling、#2 grid）**逐项承接说明，无静默丢弃** ✓。
- **数学等价优化完整性 — pass**：§1.6.1 两项采纳（#1 掩码融合进 `exp2`、#2 causal 区间裁剪）四要素齐全；「无其它优化空间」有逐条依据（逐元素链已最小、`exp2` 为 log2 域最廉价、CSE/访存/归约均下界）。
- **等价性论证独立推演 — pass**：
  - #1 `P=exp2(where(mask,x,NEG))` vs 基线 `P=where(mask,exp2(x),0)`：mask 真两式同为 `exp2(x)`；mask 假候选取 `exp2(NEG)`（`NEG=-3.0e38` 直接替换 `x`，无 `NEG*scale` 溢出），IEEE-754 fp32 下精确下溢为 `+0`，与基线 `0` 一致 ⇒ **精确等价**（独立推演成立）；`is_causal=False` 时两式均不加掩码 ⇒ 恒等。
  - #2 只扫描 `i ∈ [kv0/BN, S/BN)`：对 `i < kv0/BN`，因 `(i+1)·BN ≤ kv0`，块内全部 `(ii,jj)` 满足 `kv0+ii > q0+jj` ⇒ 掩码恒假 ⇒ `P≡0` ⇒ `dv/dk/dq` 该块贡献恒 0 ⇒ **删的是数学零项**（独立推演成立）；`is_causal=False` 时 `loop_st=0` 不裁剪。
  - **机器验证一致**：verify_equiv 重跑 10/10 全 `EQUIV_PASS`、判定量恒 0、与内嵌表逐行一致（见机械复核段）。
- **向量化替代完整性 — pass**：§1.6.2 覆盖方案全部循环/标量点（与 §3.3 伪代码、§6.1/§6.2 循环结构交叉核对**无遗漏**），10 点中 **7 点**已向量替代（#1 P、#2 掩码融合、#3 ds、#4 P/ds 降精度、#5 `dq` 行段原子、#9 group 结构展开、#10 epilogue 单次 `T.copy`）；**保留 3 点**（#6 `dq` 行循环、#7 block 索引分解、#8 `loop_st`），理由逐项具体到本算子。
- **不可替代理由充分性 — pass**：#6（`dq[b,q,h,:]` 为 BSHD 固定 head 的跨步行区、`T.atomic_add` 的 `size=` 覆盖连续区、`indexer_bwd.py` 同类退化为 `T.serial` 行循环），#7（block 级标量任务号非张量）、#8（block 级循环界），无「实现简单」类空话 ✓。
- **替代 API 可行性 — pass**：`T.vexp2(input,output,tmpBuffer)`、`T.vcast`、`T.vmul/vsub/vbrc`、`T.Parallel` 均有文档/先例；主选已由 `T.exp2` 改为 `T.vexp2`（上一轮问题 4 已修复）。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 轴质量评分（整除性/尾 lane/收缩维/repack/容量/原子目标形态）+ 候选矩阵（#1–#5）逐候选量化；#1 选定（I/O 原生 BSHD、向量轴=D）、#2（两遍零原子）为**结构完整备选**并交 §9.3 实验裁决、#3/#4/#5 给量化否决理由；未把 I/O 契约当内部布局依据；弃选前提（`T.atomic_add` `size=` 连续区语义、host permute 约束）已对照 `T.atomic_add.md` §2.4 与 `ascend-constraints.md` 亲核 ✓（`load_nd2nz` 误用风险在 §3.5.1/§9.2 有依据）。
- **实验裁决模式完整性 — pass**：判定裕度依赖未实测常数（原子与 Cube 重叠率、910B3 fp16 峰值）时，给出主选+备选（#2 两遍：buffer/循环/分核三要素）+ §9.3 裁决计划（代表 shape、msprof `TaskDur`、翻转阈值 0.95、回写）三件套；§9.4 逐项登记未实测假设（含 vexp2 KB 告警回退链）✓。
- **布局决策与实现一致 — pass**：§1.6.3 选定轴=D、布局=原生 BSHD 连续行块，与 §3.3 伪代码、§4.3 buffer 形状、§6 循环（向量维在内层）三方一致；无 repack。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入 ✓；§6 无 §1.6.2 未论证的标量循环 ✓；§3.3/§6.2 掩码现已按 `is_causal` 条件化（上一轮阻塞项已修复）。

## 检视附注：非阻塞残余项（建议级，不影响结论）

> 以下为可改进的细节，**均不阻塞 Stage 3**，供 Stage 1/Stage 3 参考。

1. **`indexer_bwd.py` 行号引用为旧修订**：§0.6 R3、§1.6.2 #6、§1.6.2 结论、§1.6.3 弃选证据、§3.2 步骤 11、§3.4 多处引用 `examples/indexer/indexer_bwd.py L109–111`（及 `L106`/`L100`/`L53`）。亲核该文件：`T.Kernel` 在 **L41**、`T.serial` 行循环在 **L117**、`T.atomic_add(dIndexK[...], dk_temp_ub[j,0], [head_dim])` 在 **L119**、`T.vcast` 在 **L102**。**引用形态真实存在、结论不受影响**，仅行号需同步至当前修订。
2. **源码 `loop_st` 行号偏差**：§0.1/§0.3 步骤 5/§0.4 #12 引用 `loop_st` 为 `L581`，源码实为 **L580**（L581 是 `loop_ed`）。属细节，不影响语义判定。
3. **§0.3 步骤表漏列源码 L585 `T.clear(qkt)`**：其余 `T.clear`（L577–578/595/608）已列。因 §3.3 NPU 实现用 `initC=True` 等效覆盖，无正确性影响。
4. **causal 逆序调度未落入伪代码**：§5.5③/§9.3 将 `jb' = n_kv-1-jb` 列为**主选**缓解措施，但 §3.3/§6.2 伪代码的 `jb = rem % n_kv` 未体现该映射。建议 Stage 1 在 §3.3 补一行（或在 §5.5 标注「属 Stage 4 调度级，不入 §3.3 功能伪代码」），以消除「主选措施 vs 伪代码」的表面不一致。仅影响任务调度顺序，**不影响正确性**。

### Stage 3 残余风险提示（沿用，供下游关注）

- **`dq` 行段原子的并发安全**：同一 `dq[b,q,h,:]` 行被同 `kh` 下不同 `jb` 的多个 program 并发 `T.atomic_add`（`size=[D]`，降级 `hivm::StoreOp`），并发度 = `S/BM`。有 `indexer_bwd.py` L117–119 先例，Stage 3 须以 §8.2 用例 5（`n_kv=1` vs `>1` 对拍）验证无丢失/重复。
- **`T.Pipelined` 运行期下界**：`loop_st = kv0//BN` 非 0 起始，§9.4 已列待确认并给回退（0 起始 + 全程掩码，正确性不变）。
- **向量 `T.vexp2` 跨设备 KB 告警**（`CG-2026-0018`/`VP-2026-0172`，Ascend910_9362）：§9.2/§9.4 已登记为 Stage 3 首测项（对 `P` 做 `isfinite` 断言）+ 回退链（`T.vexp`/标量 `T.exp2`）。
- **bf16 路径 Cube 操作数 dtype 组合**：按 `_gqa_prefill_fwd_kernel` 注记留待 Stage 3 试，L0 覆盖。
