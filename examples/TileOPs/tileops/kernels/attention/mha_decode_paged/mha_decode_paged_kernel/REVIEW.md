# `_mha_decode_no_split_kernel` 设计文档检视报告

## 检视结论

结论: 通过

> 本轮为「修订后重检」（DESIGN v1 ⟵ 上一版 v0：2 阻塞 + 3 建议）。上一轮 2 项阻塞（尾部全掩码块稳定化、online 状态初始化顺序）与 3 项建议（§3.3 掩码算子、`verify_equiv.py` OPT-2 臂、§7.2 `block_table` 间接地址 stage 依赖）**均已逐条核实修复**，且本 Agent 独立推演/复跑确认修复正确、无新引入语义偏差。

## 检视元信息

- 文档: examples/mha_decode_paged/_mha_decode_no_split_kernel/DESIGN.md（修订版 v1）
- 上一版备份: examples/mha_decode_paged/_mha_decode_no_split_kernel/history_version/design_v0.md
- 上一轮结论: `结论: 不通过`（2 阻塞 + 3 建议）——本轮逐条复核已修复
- 任务类型: 迁移（migration / migration_mode=harness）
- 源算子: `/home/developer/TileOPs/tileops/kernels/attention/mha_decode_paged.py`（`_mha_decode_no_split_kernel`，L17–110）
- 抽取副本: `examples/TileOPs/tileops/kernels/attention/mha_decode_paged/_multi_head_attention_decode_paged_with_k_v_cache_fwd_kernels.py`（helper L23–89 + kernel L93–185）
- NPU wrapper: `examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged.py`
- 规格: `examples/TileOPs/tileops/manifest/attention.yaml`（MultiHeadAttentionDecodePagedWithKVCacheFwdOp）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

**D-1 verify_equiv 重跑**（`python3 examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py`，本 Agent 实跑，EXIT=0）：

```
== verify_equiv.py (D-1): paged MHA-decode no-split online-softmax chain ==
candidate                                     max_abs   max_ulp       viol  verdict
OPT-1 (fold scale into Q)                   9.766e-04   4129.00  0.000e+00  EQUIV_PASS
OPT-2 (reassociated rescale, raw-domain)    0.000e+00      0.00  0.000e+00  EQUIV_PASS
OPT-3 (natural-exp domain)                  3.052e-05      4.48  0.000e+00  EQUIV_PASS
OPT-1+2+3 (adopted set)                     9.766e-04     95.05  0.000e+00  EQUIV_PASS

[OPT-2 factor-level (raw-domain, fp32)] rs = exp2(m_p*s - m_c*s) vs exp2((m_p - m_c)*s): max_abs=2.384e-07 max_rel=2.905e-07 over 33 blocks -> non-vacuous (rounds to 0 at fp16 output)
[MASK-CHECK (blocking-1)] static-bound + same-domain identity guard vs source dynamic bound (real_kv<N_kv + real_kv=0): max_abs=0.000e+00 nan/inf=False zero_len_nonzero=False -> PASS

vs fp64 truth (D=128,Skv=512): base_max_err=6.104e-05 cand_max_err=1.221e-04

OVERALL: EQUIV_PASS
```

- 与 §1.6.1 内嵌结果表**逐格一致**（OPT-1 9.766e-04 / OPT-2 0.0 / OPT-3 3.052e-05 / adopted 9.766e-04，违反率均 0，全部 EQUIV_PASS）。
- **OPT-2 臂非空核验（建议4 修复）**：`opt2_factor_check` 在 **raw 域 fp32** 对照源式 `exp2(m_p·s−m_c·s)` 与重结合式 `exp2((m_p−m_c)·s)`，得 `max_abs=2.384e-07 / max_rel=2.905e-07`（33 块）——两分支确为**不同表达式**，v0 的 `if/else` 同值同义反复已消除。
- **MASK-CHECK（阻塞1 修复的机器证据）**：静态界 + 同域恒等守卫 vs 源动态界，`real_kv∈{0,3,8,9,17,24,32}`，`max_abs=0.000e+00`、无 NaN/inf、`real_kv=0` 输出 0 → **PASS**。

**D-5 design_calc_check**（`python3 .agents/tools/design_calc_check.py --design <DESIGN.md>`，EXIT=0）逐项 JSON checks 摘要：

- `ub_budget`: **pass** — UB 11 buffer 项合计 53.8 KB；×1.7 auto-multi-buffer 膨胀预算 113 KB < 192 KB。
- `l1_budget`: **pass** — L1 3 buffer 项合计 132.0 KB < 512 KB（解析器附注 `declared 合计 644.0 KB, recompute matches: False`；**人工复核**：§4.5 L1 = 4+64+64 = 132 KB，含 `num_stages=2` K/V 双槽 ≈266 KB < 512 KB，成立；解析器「declared 644」系把容量行 512KB 误计入口径）。
- `l0c_budget`: **pass**（两检查）— 2 buffer 项 24 KB < 128 KB；`16×16×4 = 1 KB` 单块校验亦过（`16×256=4096 ≤ 32768` 元素）。
- `core_split`: **skip**（`insufficient parsed dims {'bm': 16}`）→ **人工复核**：`num_logical = 1×H×B ∈ {16,16,8,8}`（manifest 4 workload）≤ 物理核 20（本 Agent 实查，见维度 3），结论「无需适配」成立。
- `r3_metrics`: **pass** — R3 四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）均存在。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自 Read 源码**（`/home/developer/TileOPs/tileops/kernels/attention/mha_decode_paged.py` L17–110）+ helper 源（`online_softmax.py` + 抽取副本 L23–89），与 §0 逐项核对，**确认修订未引入新的语义偏差**：

- **语义理解（§0.1/§0.2）— pass**：
  - 数学语义 `softmax(qKᵀ/√D)V`、S_q=1 decode、BSHD、`Output` 与 Q 同 shape（源 L38–44 `Output: T.Tensor(shape_q, dtype)`）✓；
  - `scale = make_log2e_scale(dim) = (1/√dim)·log2e`（源 L20 + 抽取副本 L31）✓；
  - `accum_dtype="float"` fp32 累加（源 L21）✓；
  - exp2 域 `acc_s=exp2(acc_s*scale−max*scale)`（抽取副本 L65）✓；
  - P 量化点 `T.copy(acc_s, acc_s_cast)` 到输入 dtype（源 L98）✓；
  - **`scores_max` 取当前块 max 而非 running max**：源 helper `T.fill(scores_max,-inf)` + `T.reduce_max(clear=False)`（抽取副本 L60–61）——§0.1 该断言与源码一致 ✓（注意：`flash_attn_npuir_dev.py` 用 monotonic-max 形态，本设计**刻意**复刻源的非单调形态以保语义等价）；
  - `scores_scale=exp2(m_p·s−m_c·s)`、`acc_o*=scale`、`logsum=logsum*scale+sum`（抽取副本 L63/68 + L86–87）✓；
  - `logsum==0→0` 守卫（源 L102）✓；死代码 lse（源 L104–106）✓；非因果 OOB 掩码写在 gemm **之前**、由 gemm 累加语义生效（源 L86–88 → L89）✓。
  - **源侧守卫缺口的解读正确**：源用 `make_online_softmax`（**非** `make_online_softmax_with_mask_guard`，源 L34），该 helper 无 `-inf−(-inf)` NaN 特判（§0.1 所述）；源动态循环界 `ceildiv(real_seqlen_kv, block_N)`（源 L67–69）**从不迭代整块全掩码**，故源侧不触发。§0.1 结论与源码一致 ✓。
- **算法解读完整性（§0.3）— pass**：8 步分解逐语句覆盖源 L46–106（三维 grid、acc 初始化、动态 `loop_range`、paged `page_idx/block_idx_in_page/blockn_num_offset`（源 L74–77）、双 `T.gemm`、online softmax、rescale、归一化、lse 死代码）；host 侧 `forward` dispatch（源 L543–578）已列出，无遗漏/臆造。
- **优化手段覆盖（§0.4）— pass**：O1–O12 + 死代码，覆盖 online softmax / exp2 域 / SMEM tiling / tensor core MMA / fragment / `T.Pipelined` / 三维 grid / `threads` / P 量化 / paged 寻址 / `-inf` 掩码 / 融合 epilogue；硬件依赖逐项标注（O1 纯算法层、O3–O8 硬相关），未把实现层误判为算法层。
- **耦合性判定（§0.5）— pass**：保留 4 / 等价替换 7 / 重新设计 1（O7）/ 舍弃 2（O8、死代码）——计数与逐条依据核对一致；处置与 GPU→NPU 映射一致；舍弃项均写明理由。
- **NPU 重设计（§0.6）— pass（阻塞1 已修复）**：
  - R1（三维→一维非持久 grid）、R2（block_M=16 分形 pad + 行隔离）、R3（slice 形态 paged 装载）、R5（v-prefix online 链）、R6（移除 threads）三要素齐备；
  - **R4（动态界→静态界 + 尾部全掩码块恒等稳定化）— 已修复且本 Agent 独立推演确认正确**：
    - 源界动态（源 L67–69），静态化后**新到达**尾部全掩码块的枚举成立（非因果 OOB 有效列恒为前缀 ⟹ 全掩码块必在尾部；`real_kv=0` 全块掩码）——§0.6-R4.2 论证正确；
    - 有限大数哨兵 `-1e38` + **同域**守卫 `all_masked = m_cur ≤ -1e30`（哨兵 `-1e38 ≤ -1e30`）→ 全掩码块触发守卫 ✓（v0 的 `== -inf` 与哨兵跨域、永不触发，已弃）；
    - **`rs=1`（恒等）独立推演**：全掩码块 `m_cur_safe=0`、`rs=all_masked?1:exp(m_prev−m_cur_safe)`、`P=exp(-1e38−0)=0`、`s_sum=0` ⟹ `logsum←logsum·1+0`、`acc_o←acc_o·1+P@V`——**零贡献且不清零已累积状态**。因超界块均属尾部（其后无有效块），**`rs=0` 会错误清零已累积 `logsum/acc_o`**，故不采纳上一轮 REVIEW「置零」措辞、取恒等 `rs=1` **是正确的**（本 Agent 独立确认，即 REVIEW 原建议对尾部块不成立，设计论证更严密）；`real_kv=0` 全掩码 → `logsum=0, acc_o=0` → epilogue 守卫输出 0 ✓；
    - 与 §0.6-R4/§6.4/§5.4/§9.3「输出 0」表述**一致**（统一为「有限哨兵 + 同域守卫 + 恒等」）✓；
    - 机器证据：`MASK-CHECK`（`real_kv∈{0,3,8,9,17,24,32}`）`max_abs=0.000e+00`、无 NaN/inf、`real_kv=0` 输出 0 → PASS ✓。
- **设计与决策一致性（§1–§7）— pass（阻塞2 已修复）**：
  - §3.3 伪代码 online 状态顺序改为「`m_prev` init `-inf` → `reduce_max→m_cur` → `rs=exp(m_prev−m_cur_safe)` → … → 循环体**末尾** `T.copy(m_cur, m_prev)`」；
  - **独立演算**：首迭代 `m_prev=-inf`、`m_cur=块0 max` ⟹ `rs=exp(-inf−m_cur)=0`（fp32 exp(-inf)=0，无 NaN）⟹ `logsum=0·0+s_sum=s_sum`、`acc_o=P@V` ✓；第二迭代 `m_prev=块0 max`、`rs=exp(块0max−块1max)` ✓——与源 helper「prev=old max、new=block max」（抽取副本 L59–63）逐项等价；
  - 四处同步核对一致：§0.6-R5（L223）、§1.6.1（L383–387 优化后公式）、§3.3（L584/605–614）、§7.4（L940/945）；§3.3 明示不变量「被读缓冲先 init」✓。
- **golden 独立性（§8.1）— pass**：golden 移植源仓 `test_mha_decode_paged.py::ref_program`（paged 重组逻辑 KV + `F.scaled_dot_product_attention`），依据 §0.1 源语义，未复刻 §0.6 的 NPU 数据流（无静态界/哨兵/守卫/v-prefix 痕迹）✓。

### 1. API 可行性 — pass

§3.2/§3.4 列出的 API 均有本地佐证（本 Agent 亲自打开文档核对）：`T.gemm(initC/b_transpose)`（`docs/Tilelang.language/线性代数操作/T.gemm.md`；签名与 §3.2 一致，M/N 分形下限确为未文档化假设、设计已如实标注）；`T.reduce_max/sum(clear=True)`（`规约操作/T.reduce.md`：`clear=False` 在未初始化 buffer 上静默错误 → 一律 `clear=True` 正确；src[M,N]→dst[M,1] rank 一致、单维为 1，§3.3 形态合法）；`T.vcast(round_mode="rint")`（`数据类型转换操作/T.vcast.md`：f32→f16/bf16 支持 rint，src/dst shape 须一致）；`T.vexp`（`数学操作/T.vexp.md`：fp16/fp32，无 bf16 → §9.1-R-8 已给 bf16 经 fp32 中转绕法）；`T.vadd/vmul`（**明确支持 src1/src2 为标量**，`数学操作/T.vadd.md §2.3-4`、`T.vmul.md §2.3-4`；原地 dst==src 支持）——§3.3 `T.vmul(Q_shared, s_nat, Q_shared)` 对 `alloc_shared` 施加标量乘法**有多个先例**（`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py:266` 等 20+ 处 `vmul/vadd/vsub(*_shared,...)`），Developer 模式自动分层可行；`T.vbrc`（`shape操作/T.vbrc.md`：标量→任意 shape 无限制；`[1,N]→[M,N]` 合法；bool 尾轴须按 16 对齐——哨兵/可见性缓冲尾轴 128/256 满足）；`T.vselect`（`条件操作/T.vselect.md`：Cond/A/B/Out shape 一致，bool 条件可用）；`T.arange`（`创建操作/T.arange.md`，`dst/strides/offset` 形态）；`T.vcmp`（`比较操作/T.vcmp.md`：A/B 同 shape、输出 bool 片上使用，不可写回 GM）；`T.copy` 必须 slice 形态（`TRAP-load-nd2nz-strided` repro 实存）；`T.Pipelined`/`T.alloc_shared`/`T.alloc_fragment`（`flash_attn_npuir_dev.py` Developer 先例）。Developer 模式用自动同步，未误用手动 `sync_block_set/wait`。v-prefix 优先 ✓。

> 说明（非阻塞）：§3.3 掩码链 `T.arange(jpos,[1],0)` / `T.vcmp(jpos, real_kv, ...)`（`real_kv` 为标量，vcmp 文档要求 A/B 同 shape）/ `T.vselect(dst==srcB)` 的**逐条 arange 形态与 bool 广播细节**，设计已在 §3.3 尾注声明「以 Stage 3 实测为准（`TRAP-vbrc-empty-broadcast`）并给出算术惩罚等价替代」——该缓释在本设计边界内可接受。

### 2. 内存层级规划 — pass（含 1 项建议）

§4.4 搬运路径完整（GM→L1→L0A/L0B→L0C，`acc_s`/`acc_o` 驻 UB fragment，无 GM→L0 越级）；§4.3 逐 buffer 标注层级（L1/UB/L0C）。机械复算：UB 53.8 KB（×1.7 ≈92 KB，整除域 ≈56 KB）< 192 KB；L1 132 KB（双槽 ≈266 KB）< 512 KB；L0C 24 KB < 128 KB 且 `16×256=4096 ≤ 32768` 元素。均通过。

> 建议（非阻塞）：§3.3 epilogue 复用 `acc_s_cast`（声明 shape `(16, page_size)`）承接 `T.vcast(acc_o[16,dim], acc_s_cast)`。`T.vcast.md §2.2.2` 要求 src/dst shape 一致，故 `dim ≠ page_size` 时（manifest 的 batch2-page256 / longer-cache / shorter-cache 三例均 `D=64, page=256`）该行形态不成立。§4.3 宜补一个专用输出 cast buffer（如 `O_cast (16, dim)`，对齐 `flash_attn_npuir_dev.py:84` 先例），或在 vcast 时对 dst 显式取 `acc_s_cast[0:block_M, 0:dim]` 切片。

### 3. Tiling 策略 — pass

§5.2 给出 `block_M=16`（分形下限 + `S_q=1` 仅 1 真实行）、`block_N=page_size∈{128,256}` 及理由；§5.3 对齐约束（fp16 尾轴 256B、`dim,page_size` 均 16 倍数、`dim≥32`）；§5.4 非整除处理（pad+行隔离、OOB 掩码、写回 `real_m` 裁剪、**尾整块无效 → 同域守卫恒等**、`real_kv=0 → 输出 0`）。
**分核策略三要素齐备**（对照 `.agents/skills/_shared/standards/core-split-strategy.md §1`，本 Agent **实查复核**）：
- ① `num_logical_kernels = ceil(S_q/block_M)×H×B = 1×H×B`（§5.5），manifest 4 workload 为 `{16,16,8,8}`（与 §5.2 表逐一核对，`attention.yaml` workload `q_shape`/`kv_shape`/`page_size` 实查一致）✓；
- ② 物理核数**实查**——本 Agent 独立复跑 `NPUUtils.get().get_aicore_num()` = **20**、`NPUUtils.get().get_aivector_core_num()` = **40**，与 §5.5 记录及 `CONST-aicore-910_9362`（9362 单 die 20 AIC / 40 AIV）一致；Cube/混合算子直用 20 ✓；
- ③ 逻辑核 ≤ 物理核 → 「无需适配」，非持久一维 grid、每核 1 任务，无重复任务（`16/20=0.8 波次`，4 核空闲但无负载不均）✓；边界情形 `H·B>20` 已给分核方案（改走 split-KV dispatch）✓。
核数与 block 取值自洽；未引入核内串行（`TRAP-DEVMODE-PERSIST-GEMM` 已规避）。机械 `core_split` skip，**人工已核**如上。

### 4. 技术约束检测 — pass

三维 Kernel 已由 §0.6-R1 转一维（`ascend-constraints §1`）；GEMM 非整除在 §5.4 处理（M pad + 行裁剪、`dim`/`page_size` 16 倍数）；L0C 溢出在 §5.3/§4.5 处理（`16×256=4096` 元素 ✓）。无遗留约束。

### 5. 循环与同步 — pass

§6.1 明确 `T.Pipelined(nkv, num_stages=2)` + 元素级 `T.Parallel`/v-prefix（无逐元素标量循环）；§6.3 复算流水 L1 占用（smoke 128KB、batch2 128KB ✓）；§7 采用 Developer 自动同步并给出 Expert 回退路径。
**建议5 已修复**：§7.2 同步点表新增「`block_table` 间接寻址的 `T.copy` 源地址（`phys=block_table[bz,k]`，`k` 为流水循环变量）」条目——指出地址计算属该 stage 标量前导、须在 MTE2 装载前完成，多缓冲槽位选择是**独立于数据缓冲**的 stage 依赖（R-2 未文档化能力），并给出退路（`num_stages=1` 或 UB 行缓存 `bt_ub` + `T.alloc_local` 标量读），与 §9.1-R-2 一致，要求 Stage 3 首验专项检查。叙事完整。

### 6. 验证方案 — pass

§8.1 golden 为独立 PyTorch 参考实现（移植源仓 `ref_program`）；§8.2 按 dtype 给 atol/rtol=1e-2 且附 `cos>0.99` 双判；§8.3 L0 计划含 4 个 manifest shape（与 §5.2 表一致）+ paged 契约用例（`block_table` 非恒等、输出 shape/dtype == q）。
**建议 已修复（阻塞1 专用拦截用例）**：新增
- **L1-1**（`real_seqlen_kv=200, page=256 → 块 0 部分掩码、块 1 全掩码`）：判据 max_diff<atol + cos>0.99 且**输出非 0**、无 NaN/±inf；
- **L1-2**（`real_seqlen_kv=0` 全块掩码）：判据输出 0、无 NaN/±inf（源零迭代行为）；
- **L1-3**（`real_seqlen_kv=300` 末块部分掩码）：`all_masked=false` 走正常 softmax。
§8.3 尾注明确 L1-1/L1-2「v0 方案在此二例下会产出 inf/错误非零值」为拦截用例——用例与阻塞1 缺陷一一对应，可有效拦截回归。

### 7. 完整性与一致性 — pass

11 章节齐全（迁移类含 §0，0.1–0.7 小节齐全）；新增 §0.0 修订说明表（2 阻塞 + 3 建议逐条对照）；无占位符残留；§3.5.3 引用的 examples 路径**逐一实查存在**（`flash_attention/flash_attn_npuir_dev.py`、`flash_attention/flash_attn_npuir.py`、`gemv/example_gemv.py`、`.../multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`vector_add/vector_add/`）；pattern-library/repro 引用**逐一实查存在**（`CONST-capacity-910B2C`、`CONST-vector-launch-overhead`、`CONST-ascend910-9362-profile`、`CONST-aicore-910_9362`、`CONST-store-fixpipe-gm-only`、`CONST-cube-gemm-l0c-traffic-floor`、`PL-1.11-causal-mask-scalartrap`、`repro/TRAP-load-nd2nz-strided.py`、`repro/TRAP-DEVMODE-PERSIST-GEMM.py`、`repro/TRAP-T-copy-region-semantics.py`、`repro/TRAP-vbrc-empty-broadcast.py`）；API 映射/Tiling/内存三处数值自洽（§3.1 以 §1.6.1 优化后公式为输入、§3.3 掩码算子已落地、§6 与 §1.6.2 一致）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：含 1.6.0/1.6.1/1.6.2/1.6.3 四子节。
- **调研存在性与完整性（§1.6.0）— pass**：R1 候选表含**基线候选 0** + 5 个候选（scale 折入 Q、自然 exp 域、重缩放重结合、分母倒数、P 不量化、materialize），覆盖 `algorithm-candidates.md` `ALG-attention` R1（scale 融入 Q/K、分母倒数乘法化）与 `ALG-softmax` R1（减 max 稳定化、exp→exp2）命中行；R2 明确「**有**，且为基线所采用」，以源 `online_softmax` helper **已用 online** 为直接结构证据（符合 `algorithm-research.md §3-R2③`）：扫描 1 遍、中间缓冲 O(tile)、可 UB 驻留；R3 四口径表齐（FLOPs / 访存 Bytes / 遍数 / 中间缓冲峰值，基线在表，口径写明）；R4 逐候选亲和评估（单元匹配/容量/对齐/静态边界/流水融合）。
- **调研结论正确性（独立复核）— pass**：
  - **复杂度算术独立复算**：baseline FLOPs `4·B·H·S_q·S_kv·D = 4·1·16·1·512·128 = 4.194 MF`（×16 M-pad ≈67 MF）✓；GM 流量 `(q+2kv+o)·2 + meta`，smoke `q=2048, kv=1048576, meta=20` → `(2048+2097152+2048)·2+20 = 4,202,516 ≈ 4.20 MB` ✓（与 manifest `attention.yaml` roofline `bytes: "(q_elems + 2*kv_elems + q_elems)*elem_bytes + metadata_bytes"` 及 `metadata_bytes = B*4 + B*ceil(N_kv/page)*4 = 20` 一致）；roofline 流量项 `4.20MB / 1.06 TB/s ≈ 3.96µs`（`CONST-ascend910-9362-profile` 核→GM ≈1.06 TB/s，实存条目）✓；发射项 `45 op/核 × 0.5µs ≈ 22µs`（`CONST-vector-launch-overhead`，9362 未重验已如实标注「待重验」）✓；**中间缓冲未漏计 GM 往返**（本设计 `acc_s`/`acc_o` 驻 UB、无 `store_fixpipe` GM 往返，容量项 ≈0）✓；候选 6 materialize 的中间缓冲 GM 往返 `2·16·S_kv·N_kv_blocks` 已计入并据此淘汰 ✓；
  - **负向断言复核**：设计断言「无访存量或遍数更低的候选（KV 必须各读一次、算术强度 <1、无重算空间）」——对照 `ALG-attention`（flash O(N) 缓冲 vs materialize O(N²)）/ `ALG-softmax`（单遍 online）/ `ALG-gemv`（Cube 分形 vs Vector 规约）与结构判据成立，**未发现漏掉的已知候选**（split-KV 属 wrapper 的 split 路径，本任务 no-split 范围外，§5.5 已说明）；候选 2（纯 Vector GEMV）淘汰（发射项爆炸）与候选 6（materialize）淘汰（访存↑）均有量化依据 ✓；
  - **并列候选裁决路径**：主选（源定向）/备选（KV-as-M GEMV）指向 §1.6.3 实验裁决（含阈值 `备选 latency < 主选 × 0.9 才翻转`），非纸面拍板 ✓；
  - **亲和性淘汰依据**：容量引用 `CONST-capacity-910B2C`、发射成本引用 `CONST-vector-launch-overhead`（本 Agent 亲查 constants.md 条目实存）✓。
- **调研→设计一致性 — pass**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均以选定算法族（Cube 双 GEMM + Vector online softmax，per-(b,h) 单 CTA 整段 KV 归约）为对象；§0.5/§0.6 处置与调研选族一致；§0.4 优化意图逐项承接（§0.6 末段，无静默丢弃，唯一舍弃 = 死代码 lse 与 `threads`，均写明理由）✓。
- **数学等价优化完整性（§1.6.1）— pass**：采纳 OPT-1/2/3，每项具「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素。
- **等价性论证正确性（逐项独立推演）— pass**：
  - OPT-1 `(Q·s)Kᵀ=(QKᵀ)·s` 实域恒等；fp16 重结合误差由 D-1 实测 max_abs 9.77e-4 < atol 1e-2、违反率 0 覆盖 ✓；
  - OPT-2 源 `exp2(m_p·s−m_c·s)`，`s=log2e/√D` ⟹ `2^{(m_p−m_c)·log2e/√D}=e^{(m_p−m_c)/√D}`；设计自然域 `exp(m̃_p−m̃_c)=e^{(m_p−m_c)/√D}` ⟹ **逐位等价** ✓；raw 域 fp32 因子级对照 max_abs 2.384e-07 证臂非空 ✓；
  - OPT-3 换底 `exp2(x·log2e)=exp(x)` 恒等 ✓；三项均为精确恒等 + D-1 机器验证（违反率 0）✓。
- **向量化替代完整性（§1.6.2）— pass**：15 个计算点（含 #9b）覆盖 §3.3/§6 全部循环/标量点，**11 点已向量化**（#1–#10 + #9b）、**4 点合理保留**（#11/#12 block 索引与任务映射、#13 依赖动态 shape 的写回 extent、#14 tile 级顺序依赖），理由具体到本算子、非「实现简单/照源码写」；**与 §3.3 伪代码、§6 循环结构交叉核对一致**（§3.3 掩码链已落地 `T.arange/T.vcmp/T.vbrc/T.vselect`，epilogue 行级守卫为 #10，无 §1.6.2 未论证的逐元素标量循环）。替代 API 均有 `docs/` 或 `examples/` 佐证。
- **§1.6.3 向量化轴与布局决策 — pass**：候选矩阵 4 项（源定向主选 / KV-as-M 备选 / host permute / 核内 transpose）+ 量化评分（整除性、尾 lane 浪费率 0%、累加链形态、repack 代价、UB 影响）；弃选量化理由成立（#3 host permute 引 `PL-1.3-host-permute` 百 µs 级净亏、#4 `T.transpose` 无收益且已对照 `docs/.../创建操作/T.transpose.md`）；主选 #1 + 备选 #2（`examples/gemv` 先例，达 Stage 4 可实现深度）+ 裁决计划（代表 shape / msprof 指标 / 阈值 0.9 / 回写路径）三件套齐备；对照了源码隐式轴选择（注明 GPU 轴为输入非结论）✓。布局决策与 §3.3/§4（buffer 形状与 UB 预算）/§6 三方一致（累加循环内层向量维 = 连续尾轴）✓。
- **与下游章节一致性 — pass**：§3.1 以 §1.6.1 优化后公式为输入 ✓；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）✓。

---

## 复核结论摘要

| 上一轮条目 | 严重度 | 本轮核实结果 |
|-----------|--------|-------------|
| 阻塞1：尾部全掩码块稳定化（哨兵/守卫跨域） | 阻塞 | **已修复**——同域守卫 `all_masked = m_cur ≤ -1e30` + 恒等 `rs=1/P=0/s_sum=0`；独立推演正确（尾部块 `rs=1` 而非 0）；MASK-CHECK 机器通过；§0.6-R4/§6.4/§5.4/§9.3 表述统一 |
| 阻塞2：§3.3 online 状态初始化顺序 | 阻塞 | **已修复**——`m_prev` init `-inf` → 先算 `m_cur` → 末尾 `copy(m_cur,m_prev)`；首/次迭代独立演算正确；四处（§0.6-R5/§1.6.1/§3.3/§7.4）同步 |
| 建议3：§3.3 缺 OOB 掩码算子 | 建议 | **已修复**——补入 `T.arange+T.vadd+T.vcmp+T.vbrc+T.vselect` 哨兵链，与 §6.2/§6.4/§1.6.2-#9 一致 |
| 建议4：`verify_equiv.py` OPT-2 臂同义反复 | 建议 | **已修复**——改为 raw 域真对照 + 因子级指标（max_abs 2.384e-07 证非空） |
| 建议5：§7.2 未覆盖 `block_table` 间接地址 stage 依赖 | 建议 | **已修复**——§7.2 增补该路径同步条目 + 与 R-2 一致的退路 |

**无残留阻塞级问题。**2 项建议级（非阻塞）：① §3.3 epilogue `T.vcast` 复用 `acc_s_cast`（dim≠page_size 时 shape 不一致，宜补专用 `O_cast (16,dim)`）；② 掩码链 micro-API（`T.vcmp` 标量操作数 / `T.vselect` dst==srcB / `T.vbrc` bool 广播）细节——设计已声明 Stage 3 实测并以算术惩罚为等价替代，缓释充分。
