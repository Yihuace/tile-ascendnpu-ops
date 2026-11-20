# `_gqa_sw_fwd_kernel` 设计文档检视报告

## 检视结论

结论: 通过

> 本报告为**修订后重新检视（revision_index=1）**。对上一轮（v0）检视的 **2 阻塞 + 3 建议**逐项复核，**全部闭合**（见「上轮问题闭合核对」）：R3 复杂度算术经本 Agent **独立逐 bx 复算**订正为**执行块数 7680（对全 causal 9216 为 −16.7%）**，且与同表 fabric 逐项（0.94 / 1.41 / 1.17 / 2.35 / 2.11GB）对账无矛盾；P2 掩码探针已修复、**本 Agent 实跑可运行**且**覆盖被采纳的主选「双侧算术惩罚链」**（设备 7/7 bit-exact、零标量化 warning；备选 vcmp 链 7/7 且复现 warning）。全部 9 维度重新独立检视（含维度 0 亲自读源码、维度 8 独立推演与复杂度复算）**无阻塞级问题**，故通过。

## 检视元信息

- 文档: examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/DESIGN.md（**修订后 v1**）
- 上一版: examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/history_version/design_v0.md
- 任务类型: 迁移（scenario=migration, migration_mode=harness）
- 源算子: /mnt/workspace/TileOPs/tileops/kernels/attention/gqa_sliding_window_fwd.py（架构通用工厂 `_gqa_sw_fwd_kernel` L19 / jit L37 / prim_func L46；Hopper 变体 `_gqa_sw_fwd_wgmma_pipelined_kernel` L257 按 K5 排除）
- 检视维度: 9 项（0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: **0**
- 建议级问题数: 1（§11 smoke FLOPs 数字滑出；不阻塞）
- 工具链（复跑戳）: `source /mnt/workspace/Agentkj/env_npuir.sh`；tilelang `0.1.2+cf6f999389f4f3481da5778aa06d73478684ce2f`；Ascend**910B3**；CANN 9.0.0；2026-10-16

## 上轮问题闭合核对（v0 → v1）

| # | v0 问题 | 级别 | v1 处置 | 本 Agent 独立复核结论 |
|---|---------|------|---------|----------------------|
| 1 | R3「块数 3584（−61%）」不可复算且与同表 fabric「0.46GB」自相矛盾 | 阻塞 | 统一改**执行块数口径**；附 §1.6.0 R3 逐 bx 复算表；订正 M4/§5.2/§5.4/§10.2/§11/R4 | **闭合**——独立逐 bx 复算 Σloop_count=120/(head,batch)×(H·B=64)=**7680**（−16.7%），旧 3584=Σ(loop_count−2) 未钳位=56×64；fabric 逐项 = 0.94（KV 重读）+ 1.41/1.17（ws 往返）= 2.35/2.11GB，与括号乘积逐项对账一致，无自相矛盾（详见机械复核段） |
| 2 | P2 掩码设备证据产物不可运行（`'int' object has no attribute 'buffer'`）且只含被否决的 vcmp 链 | 阻塞 | 阈值改 let-bound/kernel 派生 PrimExpr；补入**主选算术惩罚链**；重跑 | **闭合**——本 Agent 实跑 `design_probe_band_mask.py`：MAIN 算术惩罚链 **7/7 configs bit-exact**（有效掩码 + `S+=pen·(−1e38)` 哨兵）+ `scalar-threshold warning=False`；ALT vcmp 链 7/7 bit-exact + `vcmp` 标量化 warning=True（含 fresh-compile 捕获行）。脚本已覆盖被采纳形态，正文表述与实跑输出逐字相符 |
| 3 | `has_band` 尾带缺 zbuf 零初始化（`0×NaN=NaN`，PL-1.11） | 建议 | §6.4 第 3 条 + §0.6 E7 补 `l1_b`(V) 每核一次 `zbuf` 零初始化 | **闭合**——§6.4-3/§0.6 E7/§9.1-12/§9.2 R-9 均已含；「V残留任意值×0=0」改写为「经 zbuf 零初始化保证为 0」（成本 ~0.06%） |
| 4 | §0.1 `is_causal ∧ wr≥0` 角点语义澄清 | 建议 | §0.1 补角点说明 + §9.3 记明不承诺 | **闭合**——DSIGN §0.1 角点段（源 L100–112 在 causal 下忽略 right，`k_end` 不含 wr，L76–77）；§3.3 trace 家族不含 causal+right 变体；§9.3「该组合不承诺」 |
| 5 | §1.6.1「11/11」措辞不精确 | 建议 | 改为「9 个数值型采纳项 + 4 E2E 经 verify_equiv.py 验证 11/11；M7/M12/M13 按构造论证」 | **闭合**——§1.6.1 行 396/440 表述精确；机器表 11 行 = 9 项（M5/M6、M8/M9 各合一行为 7 行）+ 4 E2E，与「M7/M12/M13 未入表」标注一致 |

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）: PASS（11/11 EQUIV_PASS，与 §1.6.1 内嵌表逐行一致）**
  - 命令: `python verify_equiv.py`（算子目录内，torch CPU），退出码 0（2026-10-16 rev-1 重跑）。
  - 实测逐行: M1 e-domain vexp max_ulp=5 / M2 lse max_ulp=1 / M3 minus-max stab 0（`naive inf=0 stab inf=0`）/ M5/M6 penalty+clamp 0（`masked bit-exact +0.0=True`）/ **M4 band-trim identity 0（`alpha(all -inf block)==1: True`）** / M8/M9 f16 S/O transport 8779 / M10 P quant 0（`quantized at source acc_s_cast point`）/ E2E-fp16 196215 / E2E-fp16-golden 2169264494（tier1_rate=0.0, tier2_rate=0.0）/ E2E-bf16 201611 / **E2E-bf16-golden 2180678995（tier1_rate=4.3e-04, tier2_rate=0.0）**。`OVERALL: ALL EQUIV_PASS`。
  - 与 §1.6.1 内嵌表（L426–438）逐格一致；E2E-bf16-golden tier-1 4.3e-4 与设计文字吻合。**等价性机器验证有效，不由 LLM 复核替代。**
- **design_calc_check（D-5）: `python3 .agents/tools/design_calc_check.py --design <DESIGN.md>`**（JSON checks 摘要）
  - `r3_metrics`: **pass** — `"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"`（四口径存在性）。
  - `ub_budget`: **skip**（`"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"`）→ 人工复核：§4.5 逐项复算 = 16+32+8+16+16+32+32+1.25+0.64 = **153.9KB ≤ 192KB/AIV** ✓，按显式 alloc 系数 ×1.12 ≈ 172KB ≤ 192KB ✓（`has_band` +ub_neg 32KB → 186KB ✓）。**人工结论 pass。**
  - `l0c_budget`: **skip**（`"no block_M x block_N values parsed"`）→ 人工复核：l0_c [64,256] f32 = **64KB ≤ 128KB** ✓。**人工结论 pass。**
  - `core_split`: **skip**（`"insufficient parsed dims {'bm': 64}; declared logical cores = None"`）→ 人工复核：三要素齐全（逻辑核数 128–2048；物理核数 20 实查；极大规模核内 `T.serial`），见维度 3。**人工结论 pass。**
  - 注：checker 的 `ub_inflation=1.7` 为 auto-multi-buffer 平台值；本设计显式关闭 auto-multi-buffer，应按 constants.md 的 0.98–1.12 显式 alloc 口径（设计引用正确）。
- **设计期探针复跑（conductor 指定，本 Agent 实跑）**
  - `design_probe_devmode_mix.py`（P1）: **复现成功** — variant A（Developer persist+gemm，无向量）**PASS** / variant B（+`T.vmul` v-prefix）**BLOCKED**（`out_of_range; map::at`，编译期）/ variant C（Expert 双 Scope）**PASS** → 与 §2.2 表完全一致，**Expert 模式结论保持成立**。
  - `design_probe_band_mask.py`（P2）: **可运行、覆盖主选形态** — `MAIN arithmetic-penalty chain: 7/7 configs bit-exact ... scalar-threshold warning=False`；`ALT vcmp/vand/vselect chain: 7/7 configs bit-exact; vcmp scalar-instruction warning=True`；`fresh-compile scalar warning: MAIN arith=False ALT vcmp=True`，捕获行 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency`；`ALL=PASS`。→ 与 §0.6 E-mask/§1.6.3/§10.2 表述**逐字相符**。
- **物理核数实查**: `NPUUtils.get()` → `arch=Ascend910B3`, `aicore_num=20`, `aivector_core_num=40`（2026-10-16）→ 与 §5.5 记录一致 ✓（混合算子直接用 aicore_num=20）。
- **R3 复杂度独立复算（本 Agent 逐 bx）**——以 manifest `llama-3.1-8b-long` (2,2048,32,128) causal fp16、wl=1024、bm=64、bn_eff=256：
  - `k_end=ceildiv((bx+1)·64,256)`、`k_start=max(0,bx·64−1024)//256`、`loop_count=k_end−k_start`；逐 bx 得 loop_count 序列 `1,1,1,1 | 2,2,2,2 | 3,3,3,3 | 4,4,4,4 | 5,5,5,5 | 5×12` → **Σloop_count = 4+8+12+16+20+60 = 120/(head,batch)** ×（H·B=32·2=64）= **7680 执行块**。全 causal（`k_start≡0`）Σ=144×64=**9216** → **−16.7%**。内段全有效块 Σmax(loop_count−2,0)=60×64=**3840**（边界掩码块 3840）。旧值 3584=Σ(loop_count−2) 未钳位=56×64 ✓。wl=256 短序列（S=512，8b/70b-short）Σloop_count=12=全 causal 同值（**0 减免**）✓。
  - fabric 逐项复算：KV tile 重读 = 7680 × (K+V 各 [256,128]f16 64KB) = 7680×128KB = **0.94GB**；ws 往返（写+读）#1 = 7680×(S 64KB+P 64KB+O f32 64KB) = 7680×192KB = **1.41GB**，合计 **2.35GB**；#2 O f16 32KB → 7680×160KB = **1.17GB**，合计 **2.11GB**（差 0.23GB ≈ 10%）。**全部与设计数字一致，无自相矛盾。**
  - FLOPs 复算：`total_attended=Σ_q min(q+1,wl+1)` = 1,574,400（设计 1,573,376）→ `4·B·H·total_attended·D` ≈ **51.6 GFLOP**；全 causal = 2,098,176 → **68.8 GFLOP（−25%）** ✓。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 本 Agent **亲自 Read 源算子全文**（`gqa_sliding_window_fwd.py` L1–519），与 §0 逐项核对。

- **语义理解正确性 — pass**：`out_idx=[3,4]`（源码 L38）↔ §0.1/§4.7；q `(B,S,H,D)` / k,v `(B,S,H_kv,D)` / output `(B,S,H,D)` / lse `(B,H,S)` **fp32**（L43–52）↔ §0.1 I/O 表；`accum_dtype="float"`、P 在 `T.copy(acc_s, acc_s_cast)` 处量化回输入 dtype（L61–62,L150）↔ §0.1 dtype 行 / §1.6.1 M10；`scores_max` 有限下限 clamp `-1e38`（L139–140）↔ §0.1 边界行 / M6；`lse = log2(logsum) + scores_max·scale`（L160）、`scale=make_log2e_scale(dim)=dim^{−1/2}·LOG2E`（L30）↔ §0.1/M2（本 Agent 复核：e 域 `m_e=m_s·prescale` 时 `log2(ℓ)+m_e·LOG2E` 与源逐位等价）。**I/O、dtype、累加序、边界语义无误。**
- **算法解读完整性 — pass**：§0.3 表 S1–S13 覆盖源码全部计算语句——三维 grid（L54–56）、初值（L71–73）、循环范围（L76–90）、四掩码路径（L100–123）、`T.gemm(transpose_B, FullRow)`（L125–126）、online max/α/exp2/sum（L131–149）、P 量化 + rescale（L150–152）、gemm2（L153–154）、epilogue（L156–161）；**无遗漏、无臆造**；host 侧（`lru_cache(32)`、`heads%heads_kv`、`make_log2e_scale`）已列（§0.3 L90）。数据流图覆盖全部 buffer（§0.3）。
- **优化手段覆盖完整性 — pass**：§0.4 OP1–OP15 逐项识别（online softmax、**带状裁剪**、SMEM tiling、coalesced、tensor core、Pipelined、fragment、warp 归约、FullRow、3D grid、exp2·LOG2E 折叠、mask 预填融合、FAST_MATH、lru_cache、bf16 直连），逐项标注源硬件依赖，算法层/实现层无混淆。
- **耦合性判定合理性 — pass**：§0.5 表体 18 行，每行四态处置 + 依据，统计 **保留 5 / 等价替换 7 / 重新设计 4 / 舍弃 2 = 18** ✓；处置与 `ascend-constraints` 不矛盾（`T.gemm` 分形、一维 Kernel、L0C→GM 出数、flag 预算均有 docs 条款）；舍弃项（OP9 FullRow、OP13 FAST_MATH+compile_flags）写明理由；OP2 带状裁剪（本算子算法核心）判「保留」并承载 E-band。
- **NPU 重设计可行性 — pass**：E1–E7 + E-band/E-mask/E-bnd 各含「源方案 → NPU 新算法 → 语义保持论证」；均满足 NPU 约束（一维 Kernel、分形 M≥16/K≥32 钳位、L1/L0C/UB 预算、PrimExpr 静态边界）。本 Agent **独立推演 E-bnd/E-mask 边界语义正确**：`K_up=(s_lo+row0)-n·bn`、`K_lo=(s_lo+row0-wl)-n·bn`，全有效块判据 `l<L_lo_full`（`ceildiv(s_lo+row0+tm-1-wl,bn)-k_start`）与 `l>=L_hi_full`（`floordiv(s_lo+row0-(bn-1),bn)-k_start+1`）经逐行充要反推成立（`diff=j-i` 的 `diff_min=-(tm-1)`、`diff_max=bn-1`）；**P2 探针设备 7/7 bit-exact 佐证**。
- **设计与决策一致性 — pass**：§1.4/§3.1/§6 描述的是 §0.5/§0.6 决策后的两相位算法（非源直译）；源单遍链→两相位、三维 grid→一维 persistent、fragment 驻留→UB+ws 物化、掩码预填→双侧算术惩罚，均一致；无「§0 判重设计、§3 照抄源」脱节。
- **golden 独立性 — pass**：§8.1 golden 为逐 (b,h) 物化 softmax（无 online、无带状裁剪、无 e 域换底、无 f16 S/O、无算术惩罚），以 §0.1 语义 + 源仓 `ref_program` 为准；本 Agent 核对 `test_grouped_query_attention_sliding_window.py` `ref_program`（`repeat_interleave` 扩 KV 头 → 三条件掩码 `masked_fill(-inf)` → softmax → `@v` → 转回 BSHD，`atol=1e-2, rtol=1e-2`，不检查 lse）与 §8.1 逐条一致，**未复刻 §0.6** → 独立性成立。
- **角点澄清（上轮建议 4，已闭合）**：源码 `is_causal and has_window` 分支（L100–112）只施加 causal|left、**忽略 right**（`k_end` 亦不含 wr，L76–77）；§0.1 已注明该角点、§3.3 trace 家族不含 causal+right、§9.3 记明「不承诺」。manifest 5 workload 及测试域 causal 用例 `wr` 恒为 −1（本 Agent 核对 manifest + 测试），目标域不受影响。

### 1. API 可行性 — pass

- §3.2/§3.4 每条 TileLang DSL API 均在 `docs/Tilelang.language/` 或 `examples/` 有佐证，且经本 Agent 逐一存在性核对：`T.gemm`（线性代数操作/T.gemm.md，Expert initC/b_transpose/size）、`T.copy`、`T.vcast`（数据类型转换操作/）、`T.reduce`（规约操作/T.reduce.md §2.2.2/§2.3）、`T.vmin`/`T.vmax`/`T.vsub`/`T.vdiv`/`T.vlog2`/`T.vmul`/`T.vadd`/`T.vexp`/`T.vbrc`/`T.arange`/`T.vselect`/`T.vcmp`（数学操作/、比较操作/、创建操作/、shape操作/、条件操作/ 均存在）、`T.sync_block_set/wait`（同步管道操作/）、`T.set_flag`、`T.Kernel(20,is_npu=True)`、`T.rs`——路径全部存在。
- **v-prefix 优先**（`vadd/vmul/vexp/vcast/vbrc`）符合 AGENTS.md 约定；`npuir_xxx` 兼容形态保留说明。
- 模式匹配：Expert 用显式 `T.sync_block_*` / `T.rs`（§7），非 Developer 自动同步——一致。
- **API 调用形态**：`vsub/vmax/vmin` 等经 `AscendBinaryOp` 支持「第二操作数为标量/Expr」；本设计的 `K_up/K_lo` 为 kernel 派生 PrimExpr，**本 Agent 设备实跑 P2 探针确证该形态可编译且数值 bit-exact**（机械复核段）。`T.load_nd2nz` 被正确禁用（规避 BSHD 跨头切片静默误读，CG-2026-0004）。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：GM→L1（slice `T.copy`，Q-hoist/K/V）→ L0C（`T.gemm`）→ GM ws（FIX，L0C 出数必经 GM，`T.store_fixpipe.md §2.1` src L0C/dst GM）→ UB → ws → L1 → L0C；**无 GM→L0 越级**。
- 预算复算（人工，见机械复核）：UB ≈ **154KB ≤ 192KB**（显式 alloc ×1.12≈172KB）；L0C [64,256] f32 = **64KB ≤ 128KB**；L1 l1_a+l1_b = [64,256]+[256,128] f16 = **96KB ≤ 512KB**。每 buffer 标注层级（§4.3）。
- L0C 不能被 v 算子直接操作（CG-2026-0005）已按 E2 以 GM ws 三族承接；`store_fixpipe` dst 仅 GM 已引 docs 条款。

### 3. Tiling 策略 — pass

- Block 划分明确（§5.2）：`bm=64`（UB/L0C 依据 + 与 wrapper 默认一致）、`bn_eff=256`（向量发射开销定律 PL-1.9-hardlimits + flag 预算守卫）。
- 对齐约束满足：`bn_eff`/`dim` 为 16 倍数（E-band 守卫公式 ceil16 保证）；分形 M/N≥16、K≥32 由 E7 钳位。
- 非整除处理：E7 size-form + 尾钳位切片 + 尾带 OOB 掩码 + 行截断；manifest 5 workload 全整除（S%64=0 且 S%256=0，本 Agent 核对 manifest 与 §5.2 表一致）。
- **分核三要素齐全（`core-split-strategy.md §1` 核对）**：① 逻辑核数 = `ceildiv(S,bm)×H×B`（§5.5 表逐 workload：smoke 128 / 8b-short 1024 / 8b-long 2048 / 70b-short 1024 / 70b-long 2048，本 Agent 复算一致）；② 物理核数 = **20（本 Agent 实查 `NPUUtils.get().get_aicore_num()`，Ascend910B3，查询代码与返回值均记录于 §5.5）**——混合算子直接用返回值 20；③ 极大规模 → 固定 20 核 + 核内 `T.serial` 串行（`num_local_tasks=T.ceildiv(num_logical−kernel_id,20)`），轮转任务解码使带状 `loop_count` 先增后减的负载均衡；循环边界为 kernel_id 派生 PrimExpr（进入循环前一次计算、循环内不变——D3 实证 + 开发指南 §3.3 模板）。核数与 block 取值自洽；smoke 128 任务轻度欠载已量化（1.17× 不均衡）并给 Stage 4 处置。
- **带状执行块数红利**（本 Agent 独立复算，见机械复核）：8b-long **7680 vs 全 causal 9216（−16.7%）**；wl=256 短序列 0 减免；内段全有效块 3840 单列。

### 4. 技术约束检测 — pass

- 三维 Kernel：源为 3D grid → E1 一维 persistent + 任务解码（bx/by/bz 派生）✓。
- L0C 容量：64KB ≤ 128KB ✓（§5.2 上界公式 `bm·max(bn_eff,dim)·4 ≤ 128KB`）。
- GEMM 非整除：E7 处理（分形下限钳位 + 尾钳位 + 尾带掩码）✓。
- flag 预算：`L_max+1 ≤ 16`（§5.2 守卫公式 + trace-time assert；8b-long L_max=5 本 Agent 独立复算成立）✓。
- `store_fixpipe` 出数 dst 仅 GM：已按 CG-2026-0005 以 GM ws 承接 ✓。

### 5. 循环与同步 — pass

- §6 明确循环类型：任务维 `T.serial`（persistent）、带内块 pass-1/pass-2 各 `T.serial`、元素级全向量化（无逐元素标量循环）；Expert 不用 `T.Pipelined`（§6.3 流水由两相位跨引擎重叠承接）。
- §7 同步策略与 Expert 模式一致：per-带内块 flag id `l`（三次握手复用）+ `FLAG_TASKDONE=L_max`；本 Agent 复核 id 交替安全——同一 id 序列 set→wait→set→wait→set→wait 严格交替（≤1 未决事件），跨任务由 TASKDONE 排空；ids ≤16（L_max≤15）。
- 尾块处理：§6.4 全套（tmc/tnc 钳位 + has_band 掩码 + **`l1_b`(V) zbuf 零初始化** + 行截断），manifest 域零触发。
- 2-AIV 聚合语义风险（R-1）已记并给回退（按 subid 拆双 id + 双 wait，预算 `L_max+2 ≤ 16`）。

### 6. 验证方案 — pass

- §8.1 含 PyTorch golden 草案且独立（见维度 0）；§8.2 含 L0 门槛计划（L0-1 fp16 smoke wl=−1 → L0-3 fp16 wl=256 带状主 trace → L0-2 bf16 R-2 → 契约校验），dtype 分级 tolerance（fp16 5e-3/1e-5 + tier-2 ×2^-10；bf16 + tier-2 ×2^-7；lse 1e-3/1e-3），tier-1/2/3 分类；Op 门 `atol=1e-2, rtol=1e-2`（源仓冻结值，本 Agent 核对测试文件一致）。
- 分层 L0/L1 阻塞、L2/Boundary 告警；回归入口 `--level all`。

### 7. 完整性与一致性 — pass

- 章节齐全：模板 11 章节 + 迁移类 §0（0.1–0.7 齐全）；**全文无占位符**（本 Agent 逐类检索未填模板槽/未完成标记，均无命中）。
- 同类引用真实：§3.5.3 引用路径全部存在（MHA 母本、flash_attn_npuir(_dev)、mixcv、deepseek_v4/v32、ada_layer_norm 的 `perf_opt/`）；§0.7 docs 路径存在；pattern-library 引用（PL-1.7/1.9/1.9-hardlimits/1.9-twophase/1.11/1.12/1.13/1.16、`traps-compiler.md` transpose 毒化条〔TRAP-transpose-epilogue-poison，2.6× + `[B,H,S,1]` 绕法〕、TRAP-DEVMODE-PERSIST-GEMM、constants.md）均真实存在；CG-2026-0004/0005/0008/0010 引用存在。
- API 映射/内存规划/Tiling 三处相互一致；**§1.6.0 R3 表内部现已自洽**（块数×块字节 = 所写 GB 逐项对账一致）——上轮在该处的一致性问题已闭合。
- 上一版残值清理：全文无残留「3584（−61%）/0.46GB」叙述（3584 仅出现在“已废弃口径”说明处）；「7680」出现 12 处且口径统一。

### 8. 算法优化分析 — pass（含 1 处建议级数字提示）

- **§1.6 章节存在性 — pass**：§1.6.0/§1.6.1/§1.6.2/§1.6.3 四子节齐全。
- **调研存在性与完整性（四问）— pass**：R1 候选表含基线（#0 源 flash online + 带状裁剪）且覆盖 attention/softmax/elementwise 命中行（本 Agent 对照 `algorithm-candidates.md` ALG-attention/ALG-softmax：两相位、flash 结构、KV-split、f16 链、scale 折入 Q、分母倒数乘法、减 max 稳定化等均已评估）；R2「在线算法 = 有」结论正确（源即 online，选定结构保留 online 递推，逐块 running max/ℓ + α 回放），并有结构判据；R3 四口径表存在（`r3_metrics` 机械 pass）且口径写清（GM 必流量 vs fabric 分列）；R4 逐候选亲和评估 + 负向淘汰附佐证。**「无更优替代」调研范围写明**。
- **调研结论正确性（独立复核）— pass**：
  - **负向断言复核**：#6 KV-split 称「无核间原语」——本 Agent **亲自打开** `docs/Tilelang.language/同步管道操作/T.set_flag.md §1`（「同一个 Cube/Vector **核内**的不同通道之间」）与 `T.sync_block_set.md §1`（「同一 block 中的其他执行单元」）核对，范围条款确为核内 → 断言成立 ✓；#9 scale 折入 q 以精度语义降级否决（论证成立）；#13 band 内逐列裁剪以「列粒度 < 向量宽度、无对应 API」否决；#14 Winograd 以「参考表无 attention 族此型行」否决。**未发现被漏掉的已知候选**；attention 族「无在线变体」类假断言本设计未出现（其 R2 正是「有」）。
  - **复杂度算术独立复算 — 通过**（重点核对中间缓冲 GM 往返）：执行块数 7680（−16.7%）、fabric 0.94/1.41/1.17/2.35/2.11GB、FLOPs 51.6/68.8 GFLOP 全部独立复算与设计一致（见机械复核段），表内数字自洽。R4 容量/带宽淘汰引用 `constants.md` 常数条目（`CONST-capacity-910B2C`、`CONST-mte2-degradation`、`CONST-fabric-bw`、`CONST-vector-launch-overhead`）并标「910B2C 实测、跨 SoC 未重验」（未实证假设显式标注）✓。
  - **并列候选裁决路径**：#6/#7/#11 以量化理由 defer；掩码式选择（算术惩罚 vs vselect）具主选 + 备选 + 实验裁决计划（阈值 0.95）三件套 ✓。
- **调研→设计一致性 — pass**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构与 §1.6.0 选定 #2（两相位 + 带内 online + 带状裁剪）一致；迁移任务 §0.5/§0.6 处置与调研选族一致；§0.4 优化手段意图承接逐项说明（OP1–OP15 无静默丢弃）。
- **§1.6.1 数学等价优化 — pass**：采纳 M1–M10/M12/M13 共 12 项、否决 M11/M14，四要素（原式→优化后→等价论证→收益）逐项齐备。本 Agent **独立推演**：M1 `e^u=2^{u·log2e}` ✓；M2 `log2(ℓ·e^m)=log2(ℓ)+m·log2e` 且 `m_e=m_s·prescale` ✓；M4 恒等元（全掩码块 `α=1`、`Δℓ=Δo=0`）✓；M5 `pen∈{0,1}`→掩蔽位精确 `−1e38`→`P=+0.0` ✓；M6 clamp 保 `m≥−1e38` ✓；M11/M14 否决理由（精度语义降级 / 列粒度无 API）成立。**采纳项等价性经 `verify_equiv.py` 重跑 11/11 EQUIV_PASS 且与内嵌表一致（D-1）**；「9 数值型采纳项 + 4 E2E；M7/M12/M13 按构造论证」表述精确（上轮建议 5 已闭合）。
- **§1.6.2 向量化替代完整性 — pass**：覆盖 18 个计算点（与 §3.3 伪代码、§6 循环结构交叉核对无遗漏）；15 项 ✅（掩码链全 ALU 化，替代 API 有 `docs/` 佐证、优先 v-prefix），保留 3 类（block 索引/带状范围 #16、tile 级顺序依赖/块分类控制流 #17、host 元数据 #18）理由具体到本算子；**§6 不出现任何 §1.6.2 未论证的逐元素标量循环**。
- **§1.6.3 向量化轴与布局决策 — pass**：候选矩阵按 MixCV 类别枚举 4 方案（行主序/列主序/单 AIV 全行/GPU 轴映射），逐候选量化（尾 lane 浪费、transpose 毒化 2.6×、UB 超限）；**弃选论证前提已亲自核对**——本 Agent 打开 `docs/Tilelang.language/创建操作/T.transpose.md`（§2.2.1 dtype 矩阵、§2.3 仅二轴交换/UB 级 `hivm.hir.vtranspose`）与 pattern-library `traps-compiler.md` 的 transpose 毒化条（活跃源 epilogue 2.6× + `[B,H,S,1]` 绕法），与设计 #2 弃选理由一致；Cube 侧 `T.gemm` 内部分形 + slice-copy + L1 生命周期复用；掩码式选择主选/备选/裁决计划三件套齐备；**布局决策与 §3.3/§4/§6 一致**。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致。
- **建议级数字提示（不阻塞）**：§11「性能预期」表 smoke 行 FLOPs 标 **0.27 GFLOP**，但按同表其余行的 manifest 口径 `4·B·H·total_attended·D` 复算，smoke（[2,512,8,64] causal wl=−1，`total_attended=Σ_q(q+1)=131328`）= 4·2·8·131328·64 ≈ **0.54 GFLOP**（差约 2×）。目标域四 workload（8b/70b-short 6.45、8b/70b-long 51.6）与 §1.6.0 R3 表（51.6/68.8）**均复算正确**，算法选型不受影响；§11 为 best_effort 且 smoke 非目标 workload，故列**建议级**。建议设计者在 §11 smoke 行采用与其余行一致的 manifest 口径数字（≈0.54 GFLOP）或补注其口径来源。

---

> 工件注入防护说明：本次 Read 的全部输入（源算子 `gqa_sliding_window_fwd.py`、DESIGN.md、探针脚本、docs、pattern-library 条目、manifest、测试文件）均按**数据**处理；未发现要求放行检视 / 跳过维度 / 改变结论的指令性文本。
> 检视所用独立复算/复验：`verify_equiv.py` 重跑（11/11）；`design_calc_check.py`；**`design_probe_devmode_mix.py` 复跑**（A PASS / B BLOCKED / C PASS）；**`design_probe_band_mask.py` 复跑**（MAIN 7/7 bit-exact 零 warning、ALT 7/7 含 vcmp warning）；`NPUUtils.get()` 实查（20 AIC / 40 AIV）；R3 逐 bx 执行块数 + fabric GB + FLOPs 独立复算；§0.6 E-bnd/E-mask 边界语义独立推演；docs 负向断言条款逐条核对。
