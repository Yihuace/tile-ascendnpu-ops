# `_mha_decode_no_split_kernel` 算子设计文档（Paged MHA Decode, no-split, NPU 迁移）

> 任务：迁移 GPU TileLang 算子 `MultiHeadAttentionDecodePagedWithKVCacheFwdOp` 的 JIT 工厂函数
> `_mha_decode_no_split_kernel`（源仓 `/home/developer/TileOPs/tileops/kernels/attention/mha_decode_paged.py` L18–116）
> 到 `target="npuir"`（华为昇腾 Ascend910_9362）。编程模式 = Developer。
> 本设计仅覆盖 **no-split**（单 CTA 完成整段 KV 归约）路径，split/combine 变体不在本任务范围。

---

## 0.0 本版修订说明（revision 1 ⟵ REVIEW.md：2 阻塞 + 3 建议）

> 本版为 Stage 2 检视**不通过**后的受控修订。**保留**首次设计的全部有效结论（Developer MixCV、一维非持久 grid、`block_M=16`、`block_N=page_size`、OPT-1/2/3、§0 源算子三问解读、§1.6 算法调研 + roofline 下界行、§5 分核三要素），**仅**按 REVIEW.md 逐条修复缺陷，未推翻已验证部分。

**相对上一版的关键调整**：

| # | REVIEW 条目 | 上一版（v0） | 本版（v1）修正 | 为何不会再犯同一错误 |
|---|-------------|-------------|----------------|---------------------|
| 阻塞1 | 尾部全掩码块稳定化自相矛盾 | 掩码用有限哨兵 `-1e38`，守卫却写 `m_cur == -inf`（**不同域**）→ 守卫永不触发；`rs=exp(m_prev−m_cur)=+inf`、`P=exp(0)=1`、`s_sum=block_N`、`logsum` 被错误累加 | 守卫与哨兵**同域**（`all_masked = m_cur ≤ -1e30`，与哨兵 `-1e38` 同域）；全掩码块置 **`rs=1`（恒等，保状态）、`P=0`、`s_sum=0`**；`m_cur` 取安全值后再算 `P`；统一 §0.6-R4/§6.4 表述为「有限哨兵 + 同域守卫」；§8.3 增补 `real_kv<N_kv`（尾整块无效）与 `real_kv=0` 用例 | 本版把「动态界→静态界后**新到达**的块」升为一等边界情形（§0.6-R4 新增「新引入块枚举」子段），并把守卫写成**与哨兵同域**；不再把 causal 场景的 `==-inf` 守卫照搬到 OOB 全掩码场景 |
| 阻塞2 | §3.3 online 状态初始化顺序错配 | 伪代码 `T.copy(m_cur, m_prev)` 在 `reduce_max` **之前**，首迭代 `m_prev` 取未初始化 `m_cur` | 统一为「**先算新 `m_cur`，再用旧 `m_prev`，末尾存**」：`m_prev` init `-inf` → `reduce_max→m_cur` → `rs=exp(m_prev−m_cur)` → … → 循环体末尾 `T.copy(m_cur, m_prev)`；§1.6.1/§0.6-R5/§7.4 同步 | 初始化缓冲与 copy 顺序绑定为一条不变量（「谁被读就先 init」，§3.3 内注明），并与 §1.6.1 公式顺序逐字对齐 |
| 建议3 | §3.3 缺 OOB 掩码实际算子（仅注释） | 仅注释 | 补入向量化掩码链（`T.arange`+`T.vcmp`+`T.vbrc`+`T.vselect` 哨兵替换；算术惩罚为等价替代），与 §6.2/§6.4/§1.6.2-#9 三处一致 | 伪代码与 §6.2/§6.4/§1.6.2 交叉对齐（掩码算子显式可见、可实施） |
| 建议4 | `verify_equiv.py` OPT-2 臂同义反复 | `if/else` 两分支同值（`m_prev*1.0 - m_cur*1.0`） | OPT-2 臂改为 **raw 域**真对照：源式 `exp2(m_p·s − m_c·s)` vs 重结合式 `exp2((m_p−m_c)·s)`；并新增 fp32 **因子级**对照行（max_abs 2.384e-07）证其非空 | 每个 D-1 臂的 baseline/candidate 必为不同表达式（`else` 分支不再是同值） |
| 建议5 | §7 同步点未覆盖 `block_table` 间接地址 stage 依赖 | §7.2 未列该路径 | §7.2 增补「循环变量寻址的 `T.copy` 源地址（`phys=block_table[bz,k]`）」的流水 stage 依赖条目 + 与 R-2 一致的退路 | 把「地址也随循环变量变化」的装载纳入 stage 依赖分析，不再只讨论数据缓冲的 stage 冲突 |

> 未采纳项：无（2 阻塞 + 3 建议全部修复）。REVIEW「阻塞1 修改建议1」中「对 `rs` 置零」的措辞未逐字采纳——**超界块均为尾部块**，`rs=0` 会把前面已累积的 `logsum/acc_o` 清零（输出错误归零），正确恒等取值为 `rs=1`（§0.6-R4 论证）；等效目标是「本块零贡献且不清零已累积状态」。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架（0.1–0.4）→ 硬件耦合性判定（0.5）→ NPU 算法重设计（0.6）」组织（方法论见
> `.agents/skills/tilelang-op-design/references/migration-analysis.md`）。本章结论驱动 §1–§11 的全部设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：paged KV-cache 多头注意力 **decode**（BSHD 布局，无 split-KV）。对每个 batch $b$、head $h$、query 位置：

$$
o[b,:,h,:]=\operatorname{softmax}\!\Big(\frac{q[b,:,h,:]\,K_{b,h}^{\top}}{\sqrt{D}}\Big)\,V_{b,h}
$$

其中 $K_{b,h},V_{b,h}$ 是**逻辑 KV 序列**（长度 `real_seqlen_kv[b]`），物理上以 **paged（block-table 寻址）** 形式存放于 `[N_kv, H, D]`。decode 场景 `S_q=1`（单 query token）。

**I/O 契约**（源 `_mha_decode_no_split_kernel` 签名，L38–45）：

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `Q` | `[batch, seqlen_q, heads, dim]` | float16 / bfloat16 | query（BSHD） |
| `K` | `[seqlen_kv, heads, dim]` | 同 Q | **物理** paged KV cache（`N_kv=seqlen_kv`） |
| `V` | `[seqlen_kv, heads, dim]` | 同 Q | 物理 paged cache |
| `real_seqlen_kv` | `[batch]` | int32 | 每 batch 的**逻辑** KV 长度 |
| `block_table` | `[batch, seqlen_kv // page_size]` | int32 | 逻辑页 → 物理页映射 |
| `Output` | `[batch, seqlen_q, heads, dim]` | 同 Q | **输出 shape = 输入 q 形状**（迁移必需字段 `(B, S_q, H, D)`），非转置布局 |

**规约语义**：
- 分数 $s_{j}=q\!\cdot\!K_j$ 在 **`dim`（D）轴**上做点积（GEMM 的 K 维，累加顺序 = 源 `T.gemm` 内部顺序，fp32 累加）。
- softmax 在 **KV 逻辑长度轴**上做（max / sum 归约，逐行 dim=1）。
- 输出 $o_d=\sum_j p_j V_{j,d}$：对 **KV 轴**归约（GEMM 的 K 维 = `block_N` 分块累加）。
- **`online softmax` 的累加分组是语义的一部分**：源按 KV-block 分块做 online 更新（`scores_max` 取**当前块** max 而非 running max，`scores_scale=exp2((max_prev−max)·scale)`，`acc_o*=scores_scale`，`logsum=logsum*scores_scale+scores_sum`）。该分块重缩放是**精确的在线等价**（不是近似），迁移必须逐项保持（重设计并行度不得改变 `scores_scale` 语义）。

**dtype 语义**：
- 输入/输出 fp16 或 bf16；**中间累加 accum_dtype = "float"（fp32）**（源 L21）。
- softmax 在 **exp2 域**计算：`scale = make_log2e_scale(dim) = (1/√dim)·log2(e)`（源 L20 + helper L31），`acc_s = exp2(acc_s*scale − max*scale)`。
- `acc_s` 在送入第二个 GEMM 前被 **cast 回输入 dtype**（`T.copy(acc_s, acc_s_cast)`，L98）——即 **P 矩阵在输入 dtype 上量化**，这是源算法的语义点（golden 与 kernel 的 P 量化噪声来源）。
- q/k/v 张量本身是 fp16/bf16；GEMM 输入为输入 dtype，累加 fp32。

**边界语义**：
- **OOB（非因果）**：`acc_s[i,j]=0 if k*block_N+j < seqlen_kv else -inf`（L87–88，`seqlen_kv=real_seqlen_kv[bz]` 动态）；掩码位置 `exp2(-inf)=0`，对 softmax 分子/分母均不贡献。
- **零长度 / 短 KV**：源 KV 循环界为**动态** `ceildiv(real_seqlen_kv, block_N)`（L67–69）；`real_seqlen_kv=0` 时 `loop_range=0`（**零迭代**）→ `logsum=0, acc_o=0` → 输出 0；`real_seqlen_kv` 非 `block_N` 整数倍时，最后一（些）块**从不被迭代**。即源**从不处理「整块被全掩码」的块**——这是迁移静态化后必须自行补齐的边界（§0.6-R4）。
- **全掩码行**：`logsum==0` 时输出 0（`T.if_then_else(logsum[i]==0, 0, acc_o/logsum)`，L102）。源中该守卫只覆盖「零迭代 / 逻辑长度 0」路径（源无全掩码块）。
- **causal**（源默认参数，manifest 工作负载全为 `False`）：`q_pos >= kv_pos`（`bx*block_M+i >= k*block_N+j`，L83）；causal 时 KV 循环上界收缩为 `ceildiv((bx+1)*block_M, block_N)`。
- **NaN/inf 传播**：源掩码用 `-T.infinity(fp32)`；`online_softmax` helper（L63）未对 `-inf − (-inf)` 做特判 → **整块全掩码**时 `exp2(-inf-(-inf)) = exp2(NaN) = NaN`（源侧因动态界不迭代全掩码块而未触发；**迁移静态化会新到达这类块**）。迁移侧改用**有限大数哨兵 `-1e38`**（规避 NaN，`PL-1.11`）+ **同域守卫**（`m_cur <= -1e30`）→ 全掩码块按**恒等**处理（`rs=1, P=0, s_sum=0`）（见 §0.6-R4 / §6.4 / O11）。
- **非整除**：`seqlen_kv % block_N`、`seqlen_q % block_M`、`dim` 与分形对齐；源用运行时掩码 + `ceildiv` 处理。

**语义保持基线**：§8.1 的 golden 函数以**本节语义**为唯一依据实现（优先移植源仓测试的 `ref_program`：paged 重组 + SDPA），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `Q` | `(B, S_q, H, D)` | fp16/bf16 | BSHD query |
| `K` | `(N_kv, H, D)` | fp16/bf16 | 物理 paged cache |
| `V` | `(N_kv, H, D)` | fp16/bf16 | 物理 paged cache |
| `real_seqlen_kv` | `(B,)` | int32 | 逻辑 KV 长度 |
| `block_table` | `(B, num_pages)` | int32 | 逻辑页→物理页 |
| `Output` | `(B, S_q, H, D)` | fp16/bf16 | 输出（**与 Q 同 shape / 同布局**） |

工厂外部签名（源 L18）：`_mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype) -> _func(block_M, block_N, num_stages, threads)`。

### 0.3 实现算法解读（怎么算）

**Kernel 结构**：`T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=threads)`（**三维 grid**，L46–47）——block 索引 `(bx=query-block, by=head, bz=batch)`。

**计算步骤分解**（源 `mha_decode_no_split` L37–106，每语句归类）：

| 步 | 计算 | 输入 | 输出 | 对应语义 |
|----|------|------|------|----------|
| 1 | `seqlen_kv = real_seqlen_kv[bz]`（标量读） | real_seqlen_kv | 标量 | 动态逻辑长度 |
| 2 | `T.copy(Q[bz, bx*bm:(bx+1)*bm, by, :], Q_shared)` | Q | Q_shared `[bm,D]` | 载入 query tile |
| 3 | `T.clear(acc_o)` / `T.clear(logsum)` / `T.fill(scores_max, -inf)` | — | 累加器初始化 | online 状态初始化 |
| 4 | `loop_range = ceildiv((bx+1)*bm, bn) if causal else ceildiv(seqlen_kv, bn)` | seqlen_kv | 标量 | KV 循环上界（动态） |
| 5 | KV 循环（`k in T.Pipelined(loop_range, num_stages)`, L73）： | | | |
| 5a | `page_idx=k//nbpp; off=block_table[bz,page_idx]*nbpp + k%nbpp` | block_table | 标量 | **paged 寻址** |
| 5b | `T.copy(K[off*bn:(off+1)*bn, by, :], K_shared)` | K | K_shared `[bn,D]` | 载入 K tile |
| 5c | `acc_s[i,j] = if_then_else(可见, 0, -inf)`（`T.Parallel(bm,bn)`） | 索引 | acc_s `[bm,bn]` fp32 | causal/OOB 掩码 |
| 5d | `T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)` | Q_shared,K_shared | acc_s | $S=QK^{\top}$（**Cube**） |
| 5e | `T.copy(V[...], V_shared)` | V | V_shared `[bn,D]` | 载入 V tile |
| 5f | `online_softmax(acc_s, ...)`（helper L57–70） | acc_s | acc_s, scores_max/scale/sum, logsum | online softmax（见下） |
| 5g | `T.copy(acc_s, acc_s_cast)` | acc_s fp32 | acc_s_cast fp16/bf16 | **P 量化到输入 dtype** |
| 5h | `rescale(acc_o, scores_scale)`（helper L84–89） | acc_o, scale | acc_o | 旧累加器重缩放 |
| 5i | `T.gemm(acc_s_cast, V_shared, acc_o)` | acc_s_cast,V_shared | acc_o | $O\mathrel{+}=PV$（**Cube**） |
| 6 | `acc_o[i,j] = if_then_else(logsum[i]==0, 0, acc_o[i,j]/logsum[i])` | acc_o,logsum | acc_o | 归一化 |
| 7 | `T.copy(acc_o, Output[bz, bx*bm:..., by, :])` | acc_o | Output | 写出 |
| 8 | （死代码）`logsum = log2(logsum_safe) + scores_max*scale`（L105–106） | — | — | 计算 **lse 但未写出**（源副作用残留） |

`online_softmax`（helper）内部 = 5 个语句：`copy(max→max_prev)`；`fill(max,-inf)`；`reduce_max(acc_s, max, dim=1)`；`scores_scale=exp2(max_prev*s − max*s)`；`acc_s=exp2(acc_s*s − max*s)`；`reduce_sum(acc_s, sum, dim=1)`；`logsum=logsum*scores_scale+sum`。

**数据流与内存访问模式**（源硬件视角）：

```
GM[Q] --copy--> SMEM[Q_shared]                                 (每 block 一次)
GM[K] --copy--> SMEM[K_shared] --gemm--> accum[acc_s f32]      (每 KV block)
acc_s --reduce/exp2--> acc_s (P, f32) --copy--> acc_s_cast f16 (每 KV block)
GM[V] --copy--> SMEM[V_shared] --gemm--> accum[acc_o f32]      (每 KV block)
acc_o --/logsum--> GM[Output]                                  (写出)
logsum/scores_max/scores_scale: fragment[row] 级在线状态
```

**循环与并行结构**：grid = `(ceildiv(S_q,bm), H, B)`；KV 维 = `T.Pipelined(loop_range, num_stages)`；元素级 = `T.Parallel` + fragment/T.macro。分块：`block_M`（query 行，默认 128，dim≤128 时 `block_N=64`，否则 32）。

**host 侧逻辑**（`MHADecodePagedKernel.forward` L543–578）：仅 no-split / split 的**dispatch 选择**（`real_max < num_split*block_N` 走 no-split）与 split 路径的 workspace 分配；**不改动输入张量内容**。本任务只实现 no-split 工厂，host 分发逻辑由 NPU 脚手架继承。

**编译配置**：`@tilelang.jit(out_idx=[-1], pass_configs={TL_ENABLE_FAST_MATH: True}, compile_flags=["-O3","-DENABLE_BF16"])`（L23–28）。

### 0.4 优化手段解读（为什么快）

> 识别不出来 ≠ 不存在；未识别的优化会在迁移中被静默丢弃。

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **online softmax（flash 结构）** | 消除 materialize O(S_kv²) 分数矩阵，中间缓冲 O(S_kv)→O(tile) | 分块 running max/sum + 重缩放 | **无（纯算法层）** | **可移植**（保留） |
| O2 | exp2 域 softmax（log2e 折入 scale） | 用廉价 exp2 替代 exp | `scale=(1/√D)·log2e`；`exp2(x*scale−m*scale)` | 无（算法层） | 可移植（等价替换） |
| O3 | SMEM tiling（`alloc_shared`） | 减少 GM 重复访问，喂 Cube | Q/K/V tile 驻留共享内存 | shared memory | **硬件强相关**（→ L1/UB） |
| O4 | Tensor Core MMA（`T.gemm`） | 矩阵算力 | `Q@K^T`、`P@V` | tensor core | **硬件强相关**（→ Cube 等价） |
| O5 | fragment 累加器（`alloc_fragment`） | 数据驻留寄存器 | `acc_s/acc_o` fragment | 大寄存器堆 | 硬件强相关（→ L0C/UB 等价） |
| O6 | 多级异步流水（`T.Pipelined(num_stages)`） | 访存/计算重叠 | KV 循环流水 | 异步拷贝引擎 | 硬件强相关（→ MTE 等价） |
| O7 | 三维 grid 并行 | 任务划分 `(query-block, head, batch)` | `T.Kernel` 三维 | — | **重新设计**（NPU 仅一维 Kernel） |
| O8 | `threads` 参数 | CUDA 线程块配置 | — | CUDA 线程模型 | **舍弃**（NPU 无对应） |
| O9 | P 矩阵量化到输入 dtype（`acc_s_cast`） | 减半 PV-GEMM 的 K 维操作数字节 | `T.copy(acc_s, acc_s_cast)` | — | 可移植（保留，语义点） |
| O10 | paged 块寻址（block_table 间接） | 支持不连续物理页 KV cache | `block_table` 间接取页 | 无（算法/寻址层） | 可移植（保留，装载形态需适配） |
| O11 | `-inf` 掩码 + `logsum==0` 守卫 | 处理 OOB/空行 | `if_then_else` | — | 可移植（保留，需稳定化） |
| O12 | 融合 epilogue（归一化写回） | 免中间落 GM | `acc_o/logsum` 直接写 Output | — | 可移植（保留） |

**算法层（可移植）**：O1、O2、O9、O10、O11、O12。**实现层（需适配）**：O3–O8。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 §1.6.0 算法调研结论）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| **语义层：数学映射 / I/O 契约 / dtype 语义 / 边界语义** | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| O1 online softmax | 算法 | 无 | — | **保留** | 纯算法层；§0.6 保持 `scores_scale` 精确在线语义 |
| O2 exp2/log2e scale | 算法 | 无 | 有（T.vexp2 / T.vexp） | **等价替换** | `exp2(x·log2e·s0) ≡ exp(x·s0)`（恒等）；见 §1.6.1 OPT-3。`docs/Tilelang.language/数学操作/T.vexp.md`、`T.vexp2.md` |
| O3 SMEM tiling | 优化 | shared memory | 有（L1 512KB / UB 192KB，32B 对齐） | **等价替换** | Cube 操作数 → L1（`alloc_shared`）；Vector 数据 → UB（`alloc_fragment`/`alloc_ub`）。CONST-capacity-910B2C；migration-analysis §5.3 |
| O4 Tensor Core MMA | 优化 | tensor core | 有（Cube `T.gemm` + L0A/L0B/L0C） | **等价替换** | `T.gemm`（Developer 模式，`docs/Tilelang.language/线性代数操作/T.gemm.md`）；分形 M/N≥16、K≥32，L0C 128KB |
| O5 fragment 累加器 | 优化 | 大寄存器堆 | 有（L0C fp32 / UB 中间缓冲） | **等价替换** | `T.alloc_fragment`（Developer）；累加顺序变化需评估（见 §0.6 语义保持） |
| O6 多级异步流水 | 优化 | 异步拷贝引擎 | 有（MTE + `T.Pipelined`） | **等价替换** | `T.Pipelined(num_stages=2)`；stage 数按 L1/UB 容量重算（§4.5） |
| O7 三维 grid | 优化 | block×thread 组织 | **无直接等价** | **重新设计** | NPU 仅一维 `T.Kernel`（ascend-constraints §1）→ §0.6 重设计项 **R1** |
| O8 `threads` 参数 | 优化 | CUDA 线程模型 | 无（NPU 无数值对应） | **舍弃** | 脚手架 K9 已约定从 NPU 工厂签名移除 `threads`；无 NPU 语义 |
| O9 P 量化到输入 dtype | 算法 | 无 | — | **保留** | `T.vcast(..., round_mode="rint")` 复现同一量化点（§1.6.2） |
| O10 paged 块寻址 | 算法 | 无 | 有（slice-form `T.copy` + 标量索引） | **等价替换** | 跨步/间接 GM 区域装载**必须**用 slice 形态 `T.copy`（`TRAP-load-nd2nz-strided`，repro `repro/TRAP-load-nd2nz-strided.py`）；block_table 间接索引 → §0.6 **R3** |
| O11 `-inf` 掩码 + 空行守卫 | 算法 | 无 | 有 | **等价替换（+稳定化）** | 掩码用有限大数哨兵 `-1e38`（`PL-1.11`）；**守卫与哨兵同域**：`all_masked = m_cur <= -1e30` → 全掩码块 `rs=1`（恒等，保状态）、`P=0`、`s_sum=0`；`logsum==0 → 0` epilogue 守卫保留（§0.6-R4 / §6.4） |
| O12 融合 epilogue | 算法 | 无 | — | **保留** | Vector epilogue（`T.vdiv` / `T.vmul` + `T.vcast`） |
| **死代码 lse 计算（L104–106）** | 实现 | 无 | — | **舍弃** | 源计算 `logsum` 的 lse 但**未写出**（无 `Output_lse` 参数），对本算子输出无贡献；迁移不实现（避免无谓 Vector op） |

> 每项处置依据已注明（映射表条目 / docs 条目 / pattern-library 条目）。**无「不可判定」项**。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目，逐项给出 NPU 新算法；其余「等价替换」项在实现中承接源意图。

**重设计项 R1：三维 grid → 一维非持久 grid（任务映射重排）**

- **源方案**：`T.Kernel(ceildiv(S_q,bm), heads, batch)` 三维 grid，`(bx,by,bz)` 直接给出 query-block/head/batch；性能意图 = 每个 (query-block, head, batch) 任务独立、无跨块通信。
- **NPU 新算法**：一维 `T.Kernel(num_logical, is_npu=True) as (cid, subid)`，`num_logical = ceildiv(S_q, block_M) * heads * batch`；由 `cid` 反解任务：`bx = cid % nq; by = (cid // nq) % heads; bz = cid // (nq*heads)`（`nq=ceildiv(S_q,block_M)`）。**非持久**（每核一任务，不引入核内 `T.serial` 多任务循环）——依据 `TRAP-DEVMODE-PERSIST-GEMM`（Developer + persistent + gemm 运行时崩溃，repro `repro/TRAP-DEVMODE-PERSIST-GEMM.py`），Developer 模式须保持非持久网格。
- **语义保持论证**：任务→(bx,by,bz) 映射是一一满射；每个任务的计算内容与源逐字相同，任务间独立无共享写；输出索引 `Output[bz, bx*bm+i, by, :]` 不变。**边界**：`num_logical` 与三维乘积相等，无遗漏/重复任务。

**重设计项 R2：decode M 维退化（S_q=1）下的 GEMM 分形适配**

- **源方案**：`block_M`（如 128）覆盖 query 行；`T.gemm(Q_shared[bm,D], K_shared[bn,D], acc_s[bm,bn], transpose_B=True)` 与 `T.gemm(acc_s_cast[bm,bn], V_shared[bn,D], acc_o[bm,D])`。
- **NPU 新算法**：NPU Cube 分形硬约束 **M≥16 / N≥16 / K≥32**（`ascend-constraints.md §1`、`docs/.../T.gemm.md §2.2.2`：src1/src2/dst 均 2 维，未文档化分形下限——分形下限为 **未文档化假设**，依据 `examples/gemv/example_gemv.py` 与迁移 prefill `_gqa_prefill_fwd_kernel` 的 `tmc=max(16,ceil16(real))` 实践）。取 **`block_M = 16`**（分形下限）覆盖 query 行；decode `S_q=1` 时 **仅第 0 行是真实 query**，行 1..15 为 padding。**K/V 分块轴仍为 `block_N`（≥16，取 page_size）**，两个 GEMM 的 `M` 维恒为 16（≥16 ✓），`N`/`K` 维满足分形。padding 行经「逐行 softmax + 只写回第 0 行」与真实行**完全隔离**（softmax/归约/重缩放均按行 dim=1，行间无耦合），不污染输出。
- **语义保持论证**：`block_M=16 ⊇ S_q=1`；只有真实行参与写回。真实行的 `S=Q̃K^T`、online softmax、`O=PV` 与源逐项一致；padding 行为垃圾但被行隔离。**边界**：`S_q` 不被 block_M 整除时（manifest 全为 1 整除）写回按 `min(block_M, S_q−bx*block_M)` 行裁剪（`TRAP-T-copy-region-semantics` 规则②：src/dst 显式同 extent 切片）。**代价**：Cube M 维 16× padding（有用行 1/16），但本算子 memory-bound（§1.6.0 R3/roofline），Cube FLOPs 非瓶颈。
- **备选（§1.6.3 实验裁决）**：KV-as-M 的 GEMV 形态（`scores^T=K_tile@q`，N=1）——`examples/gemv/example_gemv.py` 的 `naive_gemv_high_perf` 先例；主选 vs 备选见 §1.6.3。

**重设计项 R3：paged KV 寻址 + 跨步/间接装载形态**

- **源方案**：`blockn_num_offset = block_table[bz, page_idx]*num_blockn_in_page + k%num_blockn_in_page`，`T.copy(K[off*bn:(off+1)*bn, by, :], K_shared)`（GPU `T.copy` 自动处理跨步）。
- **NPU 新算法**：
  1. **令 `block_N = page_size`**，则 `num_blockn_in_page = 1`，一个 KV tile 恰为**一个物理页**：`phys_off = block_table[bz, k]`（`k` = KV 循环下标）。避免「一个 tile 跨多个物理页」带来的非连续 gather。
  2. 装载用 **slice 形态** `T.copy(K[phys_off*page_size : (phys_off+1)*page_size, by, 0:dim], K_shared[0:page_size, 0:dim])`——**必须** slice 形态（`TRAP-load-nd2nz-strided`：base+size 形态与 `load_nd2nz` 对跨步区域静默平坦误读，slice 形态 bit-exact）。K/V 的 `(rows, dim)` 区域对固定 head 是「行 stride = H·D、列连续」的跨步区域。
  3. `real_seqlen_kv[bz]` 与 `block_table[bz, :]` 为 GM 标量/行读，**只在循环头读一次**（`block_table` 物理页号）——**开列为风险项**：npuir 核内 GM 标量索引（用运行时循环变量寻址）为**未文档化能力**，Stage 3 首验；绕法 = 预先把 `block_table` 行拷入 UB（`T.copy(block_table[bz,:], bt_ub)`）后以 `T.alloc_local` 标量读取（见 §9 风险 R-2）。
- **语义保持论证**：`block_N=page_size` 时逻辑页↔物理页映射与原公式等价（`num_blockn_in_page=1`）。slice `T.copy` 的地址算术与源 `blockn_num_offset*bn` 逐位一致（仅装载 API 形态不同，值相同）。

**重设计项 R4：动态 KV 长度 → 静态循环界 + 掩码 + 全掩码块恒等稳定化**

- **源方案**：`loop_range = T.ceildiv(seqlen_kv, block_N)`，`seqlen_kv = real_seqlen_kv[bz]` **运行时**读取（动态 trip count）——源**从不迭代**满足 `k*block_N >= real_seqlen_kv` 的块，故源**从不遇到整块全掩码**。
- **NPU 新算法**：
  1. KV 循环界取**编译期静态**值 `nkv = ceildiv(N_kv_physical, page_size)`（`N_kv` 来自 `shape_kv=[seqlen_kv,H,D]`，编译期常量）；块 `k` 内有效列为 `k*block_N + j < real_seqlen_kv[bz]`（掩码；`real_seqlen_kv[bz]` 循环头读一次为标量）。
  2. **新引入块枚举（本版新增，阻塞1 修复）**：静态界比源动态界**多迭代**的块恰为尾部 `k ∈ [ceildiv(real_kv, block_N), nkv)`。对非因果 OOB 与 causal 掩码，有效列集合恒为**前缀** ⟹ 超界块**必为尾部整块全掩码**（`real_kv=0` 时全部 `nkv` 块均为全掩码块）。**源正确 ≠ 静态化后仍正确**，必须显式处理。
  3. **掩码（哨兵与守卫同域）**：无效列用**有限大数哨兵 `-1e38`**（`PL-1.11`；`exp(−1e38 − m)=0` 精确零，规避 `−inf−(−inf)=NaN`）经 `T.vselect` 值替换（NaN 免疫）。
  4. **全掩码块恒等稳定化**（守卫与哨兵**同域**）：`all_masked = (m_cur <= -1e30)`（与哨兵 `-1e38` 同域阈值）→
     - `m_cur_safe = all_masked ? 0 : m_cur`（先取安全 max，避免 `exp(+1e38)` 的 inf 瞬态）；
     - `rs = all_masked ? 1 : exp(m_prev − m_cur_safe)`（**恒等**：`logsum·1 + 0 = logsum`、`acc_o·1 + 0 = acc_o`，**保留已累积状态**）；
     - `P = exp(S̃ − m_cur_safe)` ⟹ 全掩码块所有列 `exp(−1e38 − 0)=0`；
     - ⟹ `s_sum = 0`，本块对 `logsum / acc_o` **零贡献**。
     > **`rs` 取 1 而非 0**：超界块均为**尾部**块（前缀性质），其后不会再出现有效块；若令 `rs=0` 会把前面已累积的 `logsum/acc_o` 清零 → 输出错误归零。**恒等（`rs=1`）是唯一既保状态又零贡献的取值**。`real_kv=0`（全块掩码）时同样得 `logsum=0, acc_o=0` → 输出 0。
- **语义保持论证**：超界块经 (4) 后对 `logsum / acc_o` **恒等零贡献**，与源「不迭代该块」输出等价；`real_seqlen_kv < N_kv` 与 `real_seqlen_kv = 0` 均落入同一恒等路径（不再依赖失效的 `==-inf` 守卫）。**边界**：`real_kv=0` → `logsum=0` → 输出 0（与源零迭代一致）；`real_kv` 非 `page_size` 整数倍 → 末块**部分**掩码（`m_cur` 有限，`all_masked=false`）走正常 softmax。**收益**：循环界静态，规避动态 shape 的 trip-count 限制并允许 `T.Pipelined`。

**重设计项 R5：在线 softmax 由 T.macro/T.Parallel → v-prefix 向量链（Developer MixCV）**

- **源方案**：`online_softmax` / `rescale` 为 GPU 侧 `T.macro`，内部用 `T.Parallel` 逐元素循环；`T.reduce_max/sum` 归约。
- **NPU 新算法**：Developer 模式下以 **v-prefix 向量 API** 表达同一数学：`T.vsub`/`T.vmul`/`T.vadd`/`T.vexp`/`T.vcast`/`T.vdiv` + `T.reduce_max`/`T.reduce_sum`（dim=1）作用于 fragment；广播用 `T.vbrc`。参照 `examples/flash_attention/flash_attn_npuir_dev.py`（Developer 模式 Cube `T.gemm` + v-prefix online softmax 的**已验证形态**）。
- **语义保持论证**：逐项对应，无数学改变；**状态顺序与初始化绑定**（阻塞2 修复）——`m_prev` init `-inf`，循环内「**先 `reduce_max → m_cur`**，再 `rs=exp(m_prev−m_cur)`（含同域守卫），最后 **循环体末尾 `T.copy(m_cur, m_prev)`**」；`acc_o *= scores_scale`、`logsum = logsum*scale + sum`、`P = exp((x−max)·s)` 顺序与量化点（`T.vcast(rint)`）保持一致（§3.3 伪代码逐字遵循该顺序）。

**重设计项 R6：`threads` 参数移除（继承脚手架 K9）**

- **源方案**：`_func(block_M, block_N, num_stages, threads)`；`T.Kernel(..., threads=threads)`。
- **NPU 新算法**：工厂签名改为 `_func(block_M, block_N, num_stages)`（脚手架 `mha_decode_paged.py` 已按 K9 以无 `threads` 形参调用，见该文件 L92–95）；`T.Kernel(..., is_npu=True)` 不带 `threads`。
- **语义保持论证**：`threads` 无 NPU 数值语义（O8 舍弃），移除不改变计算。

**源算法优化意图承接（§0.4 → §0.6/§1.6.0）**：O1–O12 的意图（省中间缓冲、省算、减访存、Cube 算力、流水重叠、paged 寻址）全部由 R1–R6 与 §1.6 承接；**唯一舍弃** = 死代码 lse（无输出贡献）与 `threads`（无 NPU 语义）。无静默丢弃。

### 0.7 标杆实现

- **源算子路径**：`/home/developer/TileOPs/tileops/kernels/attention/mha_decode_paged.py`（函数 `_mha_decode_no_split_kernel`）。
- **源仓参考实现 / 测试基准**：`examples/TileOPs/tests/ops/test_mha_decode_paged.py` 的 `ref_program`（paged 重组逻辑 KV + `F.scaled_dot_product_attention`）——§8.1 golden 直接移植该参考（**独立于** §0.6 的 NPU 算法）。
- **抽象副本（含 helper）**：`examples/TileOPs/tileops/kernels/attention/mha_decode_paged/_multi_head_attention_decode_paged_with_k_v_cache_fwd_kernels.py`（`make_online_softmax` / `make_rescale` / `make_log2e_scale`）。
- **NPU 集成目标（接口契约）**：`examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged.py`（`MHADecodePagedKernel`，`default_config = {block_M:128, block_N:64/32, num_split:4, num_stages:2, threads(已移除)}`）。
## 1. 概述

### 1.1 算子名称

`_mha_decode_no_split_kernel`（paged KV-cache MHA **decode**，no-split 单 CTA 整段 KV 归约）。

### 1.2 功能描述

对每个 `(batch, head, query-token)`，从 **paged（block-table 寻址）** KV cache 中取出逻辑 KV 序列，计算 `softmax(q·Kᵀ/√D)·V`；decode 场景 `S_q=1`，整个 KV 归约在**单个 CTA** 内完成（不跨 CTA 拆分 KV）。

### 1.3 数学公式

$$
o[b,:,h,:]=\frac{\sum_{j<L_b}\exp\!\big(q[b,:,h,:]\cdot K_{b,h,j}/\sqrt{D}\big)\,V_{b,h,j}}
{\sum_{j<L_b}\exp\!\big(q[b,:,h,:]\cdot K_{b,h,j}/\sqrt{D}\big)},\qquad L_b=\texttt{real\_seqlen\_kv}[b]
$$

其中 $K_{b,h,j}=K[\;\texttt{block\_table}[b,j/\!\downarrow\!p]\cdot p+(j\bmod p),\,h,\,:]$（$p$=`page_size`）。数值实现按 §1.6.1 的**优化后公式**（自然 exp 域 + scale 折入 Q）。

### 1.4 算法描述

> 本节描述**迁移决策（§0.5/§0.6）与算法调研（§1.6.0）之后**的 NPU 侧算法，**不是**源算法直译。

**单 kernel、非持久 1D grid、Developer MixCV**：一维 grid 上每核一个任务 `(query-block, head, batch)`（`S_q=1` 时 `query-block` 恒为 0），任务内：

1. 由 `cid` 反解 `(bx, by, bz)`（§0.6 R1）；读 `real_seqlen_kv[bz]` 与 `page_size` 元数据。
2. 载入 query tile `Q[bz, bx*block_M:(bx+1)*block_M, by, :]` → `Q_shared`（`block_M=16`，仅第 0 行真实，其余清零；§0.6 R2），并按 §1.6.1 OPT-1 将 `Q_shared *= (1/√D)` 一次。
3. **KV 循环**（`k in T.Pipelined(nkv, num_stages=2)`，`nkv=ceildiv(N_kv_physical, page_size)` 静态，§0.6 R4）：
   - **Cube**：按 `block_table[bz, k]` 得到物理页号 → slice-form `T.copy` 载入 `K_shared`（`[page_size, D]`，§0.6 R3）；`T.gemm(Q_shared, K_shared, acc_s, b_transpose=True, initC=True)` → `acc_s[16, block_N]` fp32（`block_N=page_size`）。
   - **Vector**：对 `acc_s` 施加 OOB/causal 掩码（**有限大数哨兵 `-1e38`**，值替换 `vselect`；§0.6 R4 / §6.4）；**在线 softmax**（§1.6.1 OPT-2/OPT-3）：**先 `reduce_max → m_cur`**，`scores_scale = exp(m_prev−m_cur)`，`acc_s = exp(acc_s − m_cur_safe)`，`reduce_sum`，`acc_o *= scores_scale`，`logsum = logsum*scores_scale + s_sum`，**末尾 `T.copy(m_cur, m_prev)`**；`acc_s` `vcast(rint)` → `acc_s_cast`（P 量化到输入 dtype，O9）。**全掩码块同域守卫**（§0.6-R4 / §6.4）：`all_masked = m_cur ≤ -1e30` → `scores_scale = 1`（恒等，保状态）、`P = 0`、`s_sum = 0`。
   - **Cube**：slice-form 载入 `V_shared`；`T.gemm(acc_s_cast, V_shared, acc_o, initC=False)` → `acc_o[16, D]` fp32。
4. **epilogue（Vector）**：`acc_o /= logsum`（`logsum==0 → 0`），`vcast(rint)` → 输出 dtype；slice 裁剪只写真实行 `Output[bz, bx*block_M : bx*block_M+real_m, by, :]`。

**与源算法的结构差异**（来源标注）：
- 三维 grid → 一维非持久 grid（§0.6 **R1**）。
- `block_M` 由 128 缩到 **16**（分形下限；`S_q=1` 时仅 1 真实行）（§0.6 **R2**）。
- `block_N = page_size`（1 tile = 1 物理页），装载用 **slice 形态** `T.copy`（§0.6 **R3**）。
- KV 循环界改为**静态**（`ceildiv(N_kv, page_size)`）+ OOB 掩码 + **尾部全掩码块同域恒等稳定化**（§0.6 **R4**，源为动态 trip 且从不迭代全掩码块）。
- softmax 由 GPU `T.macro/T.Parallel` → **v-prefix 向量链**（§0.6 **R5**）。
- 移除 `threads`（§0.6 **R6**），移除死代码 lse（§0.5 舍弃项）。
- 数学等价优化：scale 折入 Q、自然 exp 域、重缩放因子重结合（§1.6.1 OPT-1/2/3）。

### 1.5 数据流图

```
GM[Q](1,D) --T.copy--> L1[Q_shared(16,D)] --*scale(OPT-1)--> L1[Q_shared]
GM[block_table] --scalar--> phys_page
GM[K|V] --slice T.copy--> L1[K_shared(page_size,D) | V_shared(page_size,D)]
L1[Q_shared] --T.gemm(b_transpose)--> UB[acc_s(16,page_size) f32]      (Cube)
UB[acc_s] --mask / online-softmax(v-prefix)--> UB[acc_s] / UB[acc_s_cast f16] / row-state (Vector)
L1[V_shared] + UB[acc_s_cast] --T.gemm(initC=False)--> UB[acc_o(16,D) f32]   (Cube)
UB[acc_o] --/logsum, vcast--> GM[Output(bz, real_m, by, :)]            (Vector epilogue)
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先做算法调研四问（§1.6.0，输入公式/源算法只是基线候选之一）→ 公式级等价优化（§1.6.1，含 D-1 机器验证）→ 循环/标量向量化替代（§1.6.2）→ 向量化轴与布局决策（§1.6.3）。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = 同一数学语义（paged MHA decode，单 CTA 整段 KV 归约）的**算法族**。方法论与深度分级见
> `.agents/skills/tilelang-op-design/references/algorithm-research.md`；本算子为「矩阵/多步/融合」类 → **完整调研**。
> 深度匹配依据：含 2 个矩阵乘 + online 归约 + 融合 epilogue（非单步逐元素）。

**R1 等价化简公式候选**（基线候选必在表内；正式论证在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|--------|
| 0 | **基线**：源算法（Cube 双 GEMM + exp2 域 online softmax，scale 后置） | `S=(QKᵀ)·(log2e/√D)`；`P=exp2(S−m_scaled)` | — | — | ✅（基线） |
| 1 | scale 折入 Q（OPT-1） | `Q̃=Q·(1/√D)`；`S̃=Q̃Kᵀ` | 数学恒等；fp16 重结合（微 ulp） | 省每 KV 块 `[16,bn]` vmul | ✅ |
| 2 | 自然 exp 域（OPT-3） | `exp2(x·log2e) ≡ exp(x)` | 数学恒等（换底） | 去 log2e 因子 + `T.vexp2` 的 tmpBuffer，可用 `T.vexp` | ✅ |
| 3 | 重缩放因子重结合（OPT-2） | `exp2(m_p·s−m_c·s)` → `exp(m_p−m_c)`（scaled 域） | 数学恒等 | 每块省 1 个 `[16]` mul | ✅ |
| 4 | 归一化乘倒数 | `o=acc_o·(1/logsum)` | 数学恒等 | **不采纳**：向量域 `[16,D]/[16,1]` 已是 1 个 vdiv op；改 vrec+vmul = 2 op，**更贵**（`CONST-vector-launch-overhead` 每 op 固定） | ❌ |
| 5 | P 不量化（`acc_s_cast` 用 fp32 送 gemm2） | 去掉 P 的 fp16 量化点 | **非等价**（改变数值路径） | 反而增加 gemm2 的 K 维字节 | ❌（违反语义 O9） |
| 6 | 全量 materialize softmax（两遍：先存全 S 再 softmax） | O(S_kv) 中间矩阵 → GM | 数学恒等（tolerance） | **反向**：中间缓冲 O(tile)→O(N_kv)，访存↑ | ❌（见 R3） |

**R2 在线算法**：**有，且为基线所采用**（结构依据 + 源证据）。online softmax（分块 running max/sum + 重缩放）是 flash 结构（`ALG-attention` R2、`ALG-softmax` R2），源算法 `online_softmax` helper 已用该手段——**源算法已用 online 是「存在在线算法」的直接证据**（algorithm-research.md §3 R2③）。收益口径：扫描遍数 1 遍、中间缓冲 O(tile)（`acc_s[16,page_size]`，非 O(S_kv)）、UB 驻留可行（§4.5）。候选 6（materialize）在 R3 被淘汰。

**R3 复杂度对比**（四口径；口径：FLOPs 只计乘加=2/次；访存按 GM 字节；遍数 = 同一 KV 数据的 GM 读取次数；中间缓冲峰值 = 片上最大常驻 + GM 往返量）：

| 算法候选 | FLOPs（含分形 padding） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| **0 基线**（Cube 双 GEMM + online，M 分形 pad 到 16） | `4·B·H·S_q·S_kv·D` = 4.19 MF（smoke，**×16 M-pad ≈ 67 MF**） | `(q+2kv+o)·2 + meta` = **4.20 MB**（smoke） | **1**（KV 各读一次） | 片上 `acc_s[16,page_size]f32`+`acc_o[16,D]f32`；无 GM 往返 | logical = `B·H`（smoke 16）；无跨核同步（no-split） |
| 1 Cube-GEMV（KV-as-M, N=1） | 同阶（N 分形 pad 16） | 同基线 | 1 | 同基线 | 同基线；`examples/gemv` 先例 |
| 2 纯 Vector GEMV（外层积累加） | 同阶 | 同基线 | 1 | 同基线 | **发射项爆炸**：`o_d=Σ_j p_j V_{j,d}` 退化为 O(block_N) 个 `[D]` 向量 op（smoke：4×128×D/8 ≈ 8192 op/bh ⇒ 4000+µs，`CONST-vector-launch-overhead`）→ **淘汰** |
| 6 materialize 两遍 softmax | 同阶 | `+` S 中间矩阵 GM 往返 `2·16·S_kv·N_kv_blocks` | **KV 读 1 + S 写/读 2** | O(S_kv) GM 缓冲 | 同基线但访存↑ |

> 该算子**带宽受限**（算术强度 ≈ 2 FLOP/byte ≪ 机器平衡点），故**访存/遍数是主导项**：候选 0/1/2 访存量相同（差异只在单元匹配），候选 6 劣。

**R4 硬件亲和性评估**（逐候选对照检查清单，定性 + 定量 roofline；负向淘汰附依据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水/融合 | 结论 |
|---------|-------------|---------|------------|---------|----------|------|
| **0 基线（Cube 双 GEMM + Vector online）** | ✅ Cube MAC（`T.gemm`）+ Vector v-prefix；M 维 pad 到 16 但 FLOPs 非瓶颈 | ✅ L1 512KB（K/V tile ≤ 2×64KB）；UB（acc_s ≤ 16KB） | ✅ `page_size∈{128,256}` 是 16 倍数；D∈{64,128} 是 16 倍数 | ✅ 循环界静态（§0.6 R4）；静态界**新引入的尾部全掩码块**由同域守卫恒等化（§6.4） | ✅ `T.Pipelined`；Cube/Vector 同 kernel 融合（MixCV） | **✅ 选定** |
| 1 Cube-GEMV（KV-as-M,N=1） | ✅ Cube；N=1 分形 pad | ✅ 更小 L0C | ✅ | ✅ | ✅ | ✅ 备选（§1.6.3 实验裁决） |
| 2 纯 Vector GEMV | ❌ 输出累加退化为 O(bn) 外层积向量链 | ✅ | ✅ | ✅ | ⚠️ | ❌ 淘汰（发射项，`CONST-vector-launch-overhead`） |
| 6 materialize 两遍 | ✅ | ⚠️ O(S_kv) GM 缓冲 | ✅ | ✅ | ⚠️ 多一趟 | ❌ 淘汰（访存↑） |

**R4 定量 roofline（D-2）**：dispatch 代表 workload = smoke `(B=1,H=16,S_q=1,N_kv=512,D=128,page=128)`（其余 workload 同阶或更小）。
- **流量项** = GM 总字节 ÷ 有效带宽。GM 流量 = `(q+2·kv+o)·2 + meta` = `(2048 + 2·1048576 + 2048)·2 + 20` ≈ **4.20 MB**。带宽按 Ascend910_9362 单 die **核→GM ≈1.06 TB/s**（`CONST-ascend910-9362-profile`，device=910_9362，2026-10-13）→ **3.96 µs**。〔注：混合读写禁用峰值；本条按单 die 主口径。〕
- **发射项** = 向量 op 数 × 单 op 成本。每 (b,h) 的 KV 循环 4 次 × ≈10 v-op（掩码/online-softmax/cast）+ epilogue ≈5 op ≈ **45 op/核**（每核 1 任务）；按 `CONST-vector-launch-overhead`（**910B2C 实测 ~0.5µs/op，9362 未重验——待重验条目**）→ **≈22 µs**。
- **容量项** = 容量决定的 tile 循环数 × 跨引擎往返代价。本设计**无 `store_fixpipe` GM 往返**（acc_s/acc_o 驻 UB fragment，Developer 自动 CV split），容量项 ≈ 0；L0C（`16×256×4=16KB`）与 UB（≈40KB）均在限内（`CONST-capacity-910B2C`）。
- **估算下界**：`估算下界: max(3.96, 22, ~0) ≈ 22 µs（流量 3.96µs + 发射 22µs + 容量 ~0；引用 CONST-ascend910-9362-profile / CONST-vector-launch-overhead / CONST-capacity-910B2C）`。**不确定性**：发射项依赖的 0.5µs/op 为 910B2C 常数、9362 单 die 未重验（`CONST-vector-launch-overhead` 的 9362 适用性注记）——下界真实值区间约 **[4µs, 22µs]**，属**未实测常数**区间 → 进 §1.6.3 实验裁决（决定"块宽摊减"是否为主杠杆）。

**调研结论**：选定 **算法族 = 基线结构（Cube 双 GEMM + Vector online softmax，per-(b,h) 单 CTA 整段 KV 归约）+ 本文 §0.6 的 NPU 重设计**。关键依据：R3 三候选访存量同阶（带宽受限），R4 中候选 0 的 Cube/Vector 分工匹配 NPU（候选 2 纯 Vector GEMV 发射项爆炸、候选 6 materialize 访存劣）。与基线结构差异：仅实现层（grid/M-pad/静态界/装载形态/v-prefix）+ 数学等价优化（§1.6.1），算法族不变。**静态界新引入的尾部全掩码块**由 §0.6-R4 的**同域守卫恒等化**处理（`rs=1, P=0, s_sum=0`，零贡献且不清零已累积状态；§6.4），不改算法族。**调研范围**（无「更优替代」的负向断言依据）：查过 `algorithm-candidates.md` 的 `ALG-attention` / `ALG-softmax` / `ALG-gemv` / `ALG-reduction` 命中行、`attention.md` 的 `PL-1.7/1.9/1.16`、`examples/gemv`、`examples/flash_attention/flash_attn_npuir_dev.py`、迁移 prefill `_gqa_prefill_fwd_kernel`，并做 R3/R4 结构分析——**无访存量或遍数更低的候选**（KV 必须各读一次，算术强度 <1，无重算空间）。**源算法优化意图承接**：§0.4 O1–O12 全部由本选定算法 + §0.6 承接（无静默丢弃，见 §0.6 末段）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定算法。**等价性论证 = 数学恒等 + D-1 机器验证**（`verify_equiv.py`，结果表见下）。
> baseline 式 = 源 `online_softmax`/`rescale`；候选式 = 采纳优化后的 NPU 公式。

| # | 优化项 | 原式（源） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|-----------|-----------|-----------|---------|
| OPT-1 | **scale 折入 Q** | `S=(QKᵀ)·s`，`s=log2e/√D`（每 KV 块 1 个 `[16,bn]` vmul） | `Q̃=Q·s`（每 (b,h) 1 个 `[16,D]` vmul）；`S̃=Q̃Kᵀ` | 数学恒等：`(Q·s)Kᵀ=(QKᵀ)·s`。fp16 重结合：`s` 在 fp16 施加于 Q（1 ulp）vs fp32 后置——**D-1 实测 max_abs 9.77e-4 ≪ atol 1e-2，违反率 0** | 每 (b,h) 由 `nkv` 个 `[16,bn]` vmul 降为 **1 个 `[16,D]`**（smoke: 4→1，省 3 op/核） |
| OPT-3 | **自然 exp 域** | `exp2(x)`，`s=log2e/√D` | `exp(x)`，`s=1/√D` | 数学恒等（换底）：`exp2(x·log2e)=exp(x)`；故 `exp2((QKᵀ)·log2e/√D)=exp((QKᵀ)/√D)`。日志 D-1 实测 max_abs 3.05e-5 | 去 log2e 因子、去 `T.vexp2` 的 `tmpBuffer`（`T.vexp2.md §1` 需额外 tmp 且**不支持 bf16**），改用 `T.vexp`（fp16/fp32） |
| OPT-2 | **重缩放因子重结合** | `scores_scale=exp2(m_p·s − m_c·s)`（2 mul + 1 sub on `[16]`） | `scores_scale=exp((m_p − m_c)·s)`（1 sub + 1 mul） | 数学恒等（分配律 + 换底）：`m_p·s − m_c·s ≡ (m_p − m_c)·s`（实数域）；OPT-1 后 `m̃=m·s`（scaled 域）即 `exp(m̃_p − m̃_c)`。**D-1 raw 域真对照（v1 修复，替代 v0 同义反复臂）**：源式 `exp2(m_p·s−m_c·s)` vs 重结合式 `exp2((m_p−m_c)·s)`，fp32 因子级 max_abs **2.384e-07** / max_rel 2.905e-07（见下表下注）；fp16 输出级 max_abs **0.0**（重结合误差 < fp16 分辨率） | KV 循环每块省 1 个 `[16]` mul（nkv 个） |

**D-1 等价性机器验证结果**（`examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py`，torch CPU，
fp16 mixed-precision 候选式 vs 基线式，随机张量 + 角点「单可见 token」，D∈{64,128}×S_kv∈{1,4,64,512,1024}×3 seed；
对照基准 = 同 dtype 舍入路径的基线式；门限 atol=rtol=1e-2）：

| 采纳项 | 最大绝对差 (max_abs) | 最大 ulp 差 | 违反率 | 结论 |
|--------|---------------------|-------------|--------|------|
| OPT-1（fold scale into Q） | 9.766e-04 | 4129.00* | 0.000e+00 | **EQUIV_PASS** |
| OPT-2（reassociated rescale, raw-domain） | 0.000e+00 | 0.00 | 0.000e+00 | **EQUIV_PASS** |
| OPT-3（natural-exp domain） | 3.052e-05 | 4.48 | 0.000e+00 | **EQUIV_PASS** |
| OPT-1+2+3（adopted set） | 9.766e-04 | 95.05 | 0.000e+00 | **EQUIV_PASS** |

> \* OPT-1 单独项的 max_ulp 较大源于近零元素的 ULP 归一化（分母钳到 fp16 最小正规数 2⁻¹⁴），绝对差 9.77e-4 远小于 atol=1e-2；采纳集合同口径 max_ulp=95.05。所有配置 **违反率 0**，且候选式对 fp64 真值的误差（cand 1.22e-4）与基线式（base 6.10e-5）同阶——**等效性成立且无精度劣化**。**无未通过（FAIL）项**。
>
> **OPT-2 臂有效性（v1 修复，阻塞4/建议4）**：fp16 输出级 max_abs = 0.0（重结合误差低于 fp16 分辨率，故输出同值）。为证明该臂**非空**（v0 的 `if/else` 同值同义反复已消除），脚本额外输出 **raw 域 fp32 因子级对照**：源式 `exp2(m_p·s − m_c·s)` vs 重结合式 `exp2((m_p − m_c)·s)` → `max_abs=2.384e-07, max_rel=2.905e-07`（33 块）。两式为**真实不同的表达式**，数值差异在 fp32 舍入级、在 fp16 输出级归零；数学恒等由分配律独立成立。
>
> **掩码稳定化机器对照（阻塞1 附加证据）**：`verify_equiv.py` 的 `MASK-CHECK` 段用同一 fp16 单行 online 链，对照「源**动态界**」与「**静态界 + 同域恒等守卫**」两轨，覆盖 `real_kv ∈ {0,3,8,9,17,24,32}`（含 `page_size=8` 的整除 / 非整除 / 零长度）：`max_abs=0.000e+00`、无 NaN/inf、`real_kv=0` 输出为 0 → **PASS**。**反证**：同链下 v0 方案（哨兵 `-1e38` + `== -inf` 守卫）在 `real_kv=3/8/17` 产出 NaN、`real_kv=0` 得非零输出（与 REVIEW 诊断一致）。

**优化结论**：采纳 **3 项**（OPT-1/2/3，全部 EQUIV_PASS）。**优化后公式**（供 §3.1 使用）：

```
s = log2e 已去除：s_nat = 1/√D
Q̃ = Q · s_nat                                    # 每 (b,h) 一次
S̃ = Q̃ Kᵀ                                         # Cube, fp32 累加
S̃ = mask(S̃)                                      # OOB：无效列 → 哨兵 -1e38（值替换）
m_cur = rowmax(S̃)                                # 块 max（全掩码块 → -1e38）
all_masked = (m_cur ≤ -1e30)                      # 同域守卫（与哨兵同域）
m_cur_safe = all_masked ? 0 : m_cur
rs = all_masked ? 1 : exp(m_prev − m_cur_safe)    # OPT-2；全掩码 → 恒等 1（保状态）
P = exp(S̃ − m_cur_safe)                          # 全掩码块 → exp(-1e38) = 0
logsum = logsum·rs + rowsum(P)                    # 全掩码块：·1 + 0（零贡献）
acc_o = acc_o·rs + (cast_f16(P) @ V)              # O9 量化点保持
m_prev ← m_cur                                    # 循环体末尾：先算后存
o = acc_o / logsum                                # logsum==0 → 0
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案全部循环与标量计算点；可替代的用 v-prefix / `T.reduce` / `T.Parallel` 替代，不可替代的逐项给充分理由。
> API 存在性均以 `docs/Tilelang.language/` 与 `examples/` 佐证。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | 行最大 `m_cur=rowmax(S̃)` | 标量逐元素扫描 | `T.reduce_max(acc_s, m, dims=[1], clear=True)`（`docs/Tilelang.language/规约操作/T.reduce.md`） | ✅ | — （`clear=True` 强制，`clear=False` 为静默错误形，`T.reduce.md §2.3`） |
| 2 | 行和 `rowsum(P)` | 标量累加 | `T.reduce_sum(acc_s, sum, dims=[1], clear=True)` | ✅ | — |
| 3 | 重缩放因子 `rs` | 逐元素 | `T.vsub` + `T.vmul` + `T.vexp`（`docs/Tilelang.language/数学操作/`） | ✅ | — |
| 4 | `P=exp(S̃−m)` | 标量 `exp` | `T.vsub` + `T.vexp`（`docs/Tilelang.language/数学操作/T.vexp.md`） | ✅ | — |
| 5 | `acc_o*=rs` | 标量循环 | `T.vmul(acc_o, rs_bcast, acc_o)` + `T.vbrc` 广播 | ✅ | — |
| 6 | `logsum=logsum*rs+sum` | 标量 | `T.vmul` + `T.vadd`（`[16]` 行向量） | ✅ | — |
| 7 | P 量化 `acc_s_cast` | 逐元素 cast | `T.vcast(acc_s, acc_s_cast, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/`） | ✅ | — |
| 8 | 归一化 `acc_o/logsum` | 逐元素除 | `T.vdiv(acc_o, logsum_bcast, acc_o)`（广播除，1 op） | ✅ | — |
| 9 | OOB/causal 掩码 | `T.Parallel` `if_then_else` 写 -inf | 有限大数哨兵 `-1e38` + `T.vselect`（值替换，NaN 免疫）/算术惩罚（`PL-1.11`；`docs/Tilelang.language/{比较操作/T.vcmp.md,条件操作/T.vselect.md}`；列索引 `T.arange` + `T.vbrc` 广播，`shape操作/T.vbrc.md` 示例2 `(1,N)→(M,N)`） | ✅ | — |
| 9b | **全掩码块守卫** `all_masked=(m_cur≤-1e30)` → `rs=1,P=0,s_sum=0` | 行级条件 `m==-inf→…`（v0 与哨兵**不同域**、失效） | `T.vcmp` + `T.vselect`（`[block_M,1]` 行向量；守卫与哨兵 `-1e38` **同域**，`PL-1.11`）；`m_cur_safe` 经 `T.vselect` 置 0 使 `P=exp(-1e38)=0` | ✅ | — （阻塞1 修复；替代 v0 的失效 `==-inf` 守卫） |
| 10 | `logsum==0 → 0` 守卫 | 标量条件 | `T.if_then_else`（`[block_M,1]` 行，非逐元素热点） | ✅（保留为行级条件） | — |
| 11 | `cid → (bx,by,bz)` 任务反解 | 标量整除/取模 | 无 | ❌ | **block 级索引 / 任务映射**：每核 1 次（`S_q=1` 时 `bx≡0`），非逐元素热点，无逐元素等价 API |
| 12 | `phys = block_table[bz,k]` 物理页寻址 | 标量 GM 读 | 无（`T.copy` 地址需标量） | ❌ | **block 级索引**：每 KV 块 1 次标量，用于构造 `T.copy` 地址，不可向量化（API 层面地址必须标量）；见 §9 风险 R-2 |
| 13 | `real_m = min(block_M, S_q − bx·block_M)` | 标量 | 无 | ❌ | **依赖动态 shape 的边界**（尾块行裁剪）；写回 extent 必须标量（`TRAP-T-copy-region-semantics` 规则②），非逐元素热点 |
| 14 | KV 循环体（`T.Pipelined`） | block 级循环 | 保留（非逐元素） | ❌（合理保留） | **tile 级顺序依赖**：online 更新跨 tile 携带依赖（`logsum`/`acc_o`/`m_prev`），无归约/双缓冲可消除 |

**向量化结论**：**逐元素/行级计算已全部向量化**（第 1–10 项及 9b，替代 API 均有 `docs/` 或 `examples/` 佐证）；保留 **4 处**标量/循环，理由类别：**block 索引/任务映射（#11, #12）**、**依赖动态 shape 的边界（#13）**、**tile 级顺序依赖（#14）**。无「实现简单/照源码写」类保留。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（BSHD 进 / BSHD 出，**不重排 GM 布局**）；核内布局与向量化轴是**设计变量**。
> 本算子含 ≥2 非 batch 维（`block_M` 行 × `block_N` KV 列 / `dim` 列），须枚举候选并量化。

**轴质量评分**（候选"源定向"方案：`scores[16, block_N]`，reduce dim=1；`acc_o[16, D]`）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | 尾轴长度是否向量宽度整数倍（fp16 ×8） | `block_N=page_size∈{128,256}` ✓（8 倍数）；`dim∈{64,128}` ✓ |
| 尾 lane 浪费率 | `(ceil(len/8)·8 − len)/len` | 0%（128/256/64 均 8 整除） |
| 累加链形态 | reduce 轴 stride=1（连续，走快速向量路径）还是跨步 | `scores` reduce dim=1 = 连续尾轴 ✓（水平归约）；`acc_o` 逐元素沿连续 `dim` ✓ |
| repack 代价 | 布局重排 | **无 repack**（I/O 原生 BSHD；核内 fragment 由编译器布局） |
| UB 容量影响 | 驻留缓冲尺寸 | `acc_s[16,256]f32=16KB` + `acc_o[16,128]f32=8KB` + 小 buffer ≈40KB ≪ 192KB ✓ |

**候选矩阵**（枚举 + 弃选量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **I/O 原生 BSHD + 源定向**（`scores[16,bn]` reduce dim=1；`acc_o[16,D]`） | 连续尾轴（`bn` / `dim`） | 无 | 基线；尾 lane 0%；reduce 走水平（连续）路径 | **✅ 主选** |
| 2 | KV-as-M 定向（`scores[bn,1]` reduce dim=0；`acc_o[D,1]`；GEMV N=1） | 连续尾轴 | 无（但 reduce 沿 dim=0 跨步） | 变 Cube M/N 定向；`examples/gemv` 先例；reduce dim=0 为垂直（跨步）路径，效率未知 | ✅ **备选**（§1.6.3 实验裁决） |
| 3 | host permute 进出（BSHD↔BHSD） | `dim` | 两次 host permute | repack ~百 µs 级（`PL-1.3-host-permute`）> 算子本体流量（~4µs），**净亏** | ❌ |
| 4 | 核内 transpose 到 `[D, page_size]` 让归约沿连续轴 | `D` | `T.transpose`（UB 级 ~µs，`docs/Tilelang.language/创建操作/T.transpose.md`；`ALG-transpose`） | 无必要：主选 reduce 已在连续轴；额外转置只增向量管线开销与 UB | ❌（无收益） |

> 迁移提示：源 GPU 的 `block×thread` 轴选择是**输入而非结论**（GPU warp/thread 轴 ≠ NPU Vector 轴）；本表独立评估。

**实验裁决模式（判定裕度落在未实证常数 → 强制三件套）**：主选（#1）与备选（#2）的取舍涉及**未实证常数**——`CONST-vector-launch-overhead`（0.5µs/op）在 9362 未重验，且 NPU `T.reduce(dim=0)` 跨步归约效率无实测（pattern-library 未覆盖）。故：
- **主选方案（进 Stage 3）**：#1（源定向 + 源分块；§3/§4/§6 只承载主选）。
- **备选方案（≤1 个）**：#2（KV-as-M GEMV 定向）——buffer 形状 `K_shared[page_size,D]`/`q_col[D,1]`/`p_col[page_size,1]`/`o_col[D,1]`；reduce 沿 dim=0；分核三要素不变（逻辑核 = `B·H`，物理核实查 20）；`T.gemm(..., a_transpose=True)` 出 `o_col`。达 Stage 4 可直接实现深度。
- **裁决计划**：代表 shape = smoke `(1,16,1,512,128,128)` 与 batch2-page256 `(2,8,1,1024,64,256)`；指标 = msprof op Task Duration（median≥20）+ 反解单 op 发射成本（对比 `CONST-vector-launch-overhead`）；**判定阈值** = 备选 latency < 主选 × 0.9 才翻转；结果回写 §1.6.3。**shape 特化**：工厂 `lru_cache`，可按 `page_size`/`D` 分派主选/备选。

**布局决策结论**：主选 **#1 = 核内保持源定向（`scores[block_M, block_N]`、reduce dim=1；`acc_o[block_M, D]`），向量化轴 = 连续尾轴（`block_N` / `dim`），无 repack**；备选 #2 由 Stage 4 实验裁决。该结论同步落入 §3.3 伪代码与 §6（累加循环内层向量维 = 连续尾轴）。
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Developer**（含 Cube `T.gemm` + Vector v-prefix 的 **MixCV** 形态）。

### 2.2 选型理由

- 任务指定编程模式 = `developer`；脚手架 `mha_decode_paged.py`（K9）也按 Developer 形态调用（无 `threads`、无显式 `Scope`）。
- 本 kernel **同时**含 Cube 侧 `T.gemm`（2 处）与 Vector 侧 v-prefix 算子（`T.vsub/vexp/vmul/vcast/vdiv/vbrc`），满足 `tilelang-mixcv-skill` 的 **Developer 模式 MixCV 判定规则**（AGENTS.md「Developer-mode MixCV trigger rule」）→ 按 `tilelang-mixcv-skill` 设计。
- 参照已上库的 Developer MixCV 先例 `examples/flash_attention/flash_attn_npuir_dev.py`（`T.Kernel(n, is_npu=True)` 非持久网格 + `alloc_shared`/`alloc_fragment` + `T.gemm` + v-prefix online softmax + `T.Pipelined`）。
- **非持久网格**：规避 `TRAP-DEVMODE-PERSIST-GEMM`（Developer + persistent + gemm 运行时崩溃，repro `repro/TRAP-DEVMODE-PERSIST-GEMM.py`）——本设计每核 1 任务（`B·H ≤ 20`，见 §5.5），无需核内 `T.serial` 多任务。
- 不选 Expert：Expert 双 Scope 的手动 `sync_block_set/wait` + L1/L0C 显式管理复杂度高，且本算子任务数少（≤20）、KV 块数少（≤8），跨引擎块级流水的收益不足以抵其复杂度（`PL-1.7-expert-persistent-boundary` 短 KV 形态回退 ~1.31×）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| Kernel 声明 | `T.Kernel(num_logical, is_npu=True) as (cid, subid)`（一维、非持久） |
| 内存分配 | `T.alloc_shared`（Cube 操作数 Q/K/V → L1）；`T.alloc_fragment`（acc_s/acc_o/行向量 → UB/寄存器） |
| 计算方式 | `T.gemm`（Cube） + v-prefix 向量链 + `T.reduce_max/sum`（Vector），编译器自动 CV split |
| 同步 | **自动同步**（Developer 模式；`T.Pipelined` 自动插同步）——无手动 `sync_block_set/wait` |
| `pass_configs` | 沿用源 `TL_ENABLE_FAST_MATH=True`；`threads` 移除 |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 的**优化后公式**（自然 exp 域 + scale 折入 Q + 重结合重缩放）。步骤优先映射 v-prefix。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `Q̃ = Q·(1/√D)` | scale 折入 Q（OPT-1），每 (b,h) 一次 |
| 2 | `S̃ = Q̃ Kᵀ` | score GEMM（Cube），fp32 累加 |
| 3 | `S̃ ← mask(S̃)` | OOB/causal 掩码（**有限大数哨兵 `-1e38`**，值替换） |
| 4 | `m_cur = rowmax(S̃)` | 行最大归约（块 max） |
| 5 | `all_masked = m_cur ≤ -1e30`；`m_cur_safe = all_masked ? 0 : m_cur`；`rs = all_masked ? 1 : exp(m_prev − m_cur_safe)` | **同域守卫** + online 重缩放因子（OPT-2 + **全掩码恒等**，保状态） |
| 6 | `P = exp(S̃ − m_cur_safe)` | 指数（OPT-3 自然 exp；全掩码 → 0） |
| 7 | `s_sum = rowsum(P)`；`logsum = logsum·rs + s_sum` | 行和 + online 累加（全掩码块：·1+0） |
| 8 | `acc_o = acc_o·rs`；`m_prev ← m_cur` | 旧累加器重缩放 + **末尾存旧 max（先算后存）** |
| 9 | `acc_o += cast(P)·V` | PV GEMM（Cube），P 量化到输入 dtype（O9） |
| 10 | `o = acc_o / logsum`（`logsum==0 → 0`）；`o ← cast(o)` | 归一化 epilogue（融合写回） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 任务反解 / 元数据 | `cid→(bx,by,bz)`；`real_seqlen_kv[bz]` | 标量算术 / BufferLoad | — | Developer |
| Q 载入 | `Q[bz, bx·bm:(bx+1)·bm, by, :] → Q_shared` | `T.copy`（**slice 形态**） | src/dst 双显式 slice | Developer |
| Q scale | `Q_shared *= s` | `T.vmul(Q_shared, s_bcast, Q_shared)` 或 `T.vbrc`+乘法 | 广播标量 | Developer |
| K 载入 | `K[phys·p:(phys+1)·p, by, :] → K_shared` | `T.copy`（slice 形态） | `phys=block_table[bz,k]` | Developer |
| score GEMM | `Q̃ Kᵀ` | `T.gemm(Q_shared, K_shared, acc_s, b_transpose=True, initC=True)` | fp16×fp16→fp32（`T.gemm.md §2.2.1/§2.3`） | Developer（Cube） |
| 掩码 | OOB/causal | `T.arange`（列索引）+ `T.vcmp` + `T.vbrc`（广播）+ `T.vselect`（哨兵 `-1e38` 值替换）；等价替代 = 算术惩罚 `T.vclamp`+`T.vmul`+`T.vadd` | 见 §6.4 | Developer（Vector） |
| 行最大 | `rowmax` | `T.reduce_max(acc_s, m_cur, dims=[1], clear=True)` | `T.reduce.md` | Developer（Vector） |
| 全掩码守卫 | `all_masked` | `T.vcmp(m_cur, -1e30, all_masked, "le")` + `T.vselect(all_masked, one_vec, rs, rs)`（恒等）+ `T.vselect(..., zero_vec, m_cur, m_cur_safe)` | `[block_M,1]`；与哨兵**同域** | Developer（Vector） |
| 重缩放因子 | `rs` | `T.vsub` + `T.vexp`（全掩码 `vselect` → 1） | `[block_M,1]` 行向量 | Developer（Vector） |
| P | `exp(S̃−m_cur_safe)` | `T.vsub` + `T.vexp` | inplace 或新 buffer | Developer（Vector） |
| 行和 / logsum | `rowsum(P)`；online | `T.reduce_sum(..., clear=True)` + `T.vmul` + `T.vadd` | — | Developer（Vector） |
| 重缩放 acc_o | `acc_o·rs` | `T.vmul(acc_o, rs_bcast, acc_o)` | `T.vbrc` 广播 `[16]→[16,D]` | Developer（Vector） |
| P 量化 | `cast(P)` | `T.vcast(acc_s, acc_s_cast, round_mode="rint")` | — | Developer（Vector） |
| V 载入 | `V[phys·p:(phys+1)·p, by, :] → V_shared` | `T.copy`（slice 形态） | — | Developer |
| PV GEMM | `acc_o += P·V` | `T.gemm(acc_s_cast, V_shared, acc_o, initC=False)` | fp16×fp16→fp32 | Developer（Cube） |
| 归一化 | `acc_o/logsum` | `T.vdiv(acc_o, logsum_bcast, acc_o)` | 广播除 | Developer（Vector） |
| 写出 | `acc_o → Output` | `T.vcast` + `T.copy`（局部行切片） | `TRAP-T-copy-region-semantics` ② | Developer（Vector） |
| KV 流水 | 循环调度 | `T.Pipelined(nkv, num_stages=2)` | 静态界 | Developer |

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型作为**第二位置参数**传入（不用 `dtype=` 关键字），避免 `name 'in_dtype' is not defined` 假告警。以下为主选方案（§1.6.3 #1）。

```python
@functools.lru_cache(maxsize=32)
def _mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size,
                                is_causal, dtype):
    accum_dtype = "float"
    block_M = 16                       # 分形下限（§0.6 R2）；S_q=1 时 1 真实行
    block_N = page_size                # 1 tile = 1 物理页（§0.6 R3）
    num_q  = ceildiv(seqlen_q, block_M)
    num_logical = num_q * heads * batch
    nkv = ceildiv(seqlen_kv, page_size)          # 静态 KV 块数（§0.6 R4）
    s_nat = 1.0 / (dim ** 0.5)                   # 自然 exp 域 scale（OPT-1/3）

    @tilelang.jit(out_idx=[-1], target="npuir",
                  pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _func():  # 注意：工厂签名已移除 threads（§0.6 R6）
        @T.prim_func
        def mha_decode_no_split(
            Q: T.Tensor([batch, seqlen_q, heads, dim], dtype),
            K: T.Tensor([seqlen_kv, heads, dim], dtype),
            V: T.Tensor([seqlen_kv, heads, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], "int32"),
            block_table: T.Tensor([batch, ceildiv(seqlen_kv, page_size)], "int32"),
            Output: T.Tensor([batch, seqlen_q, heads, dim], dtype),
        ):
            with T.Kernel(num_logical, is_npu=True) as (cid, subid):
                # ---- 任务反解（§0.6 R1；标量，每核一次）----
                bx = cid % num_q
                by = (cid // num_q) % heads
                bz = cid // (num_q * heads)
                real_kv = real_seqlen_kv[bz]        # 标量读（风险 R-2）
                real_m  = T.min(block_M, seqlen_q - bx * block_M)

                # ---- Cube 操作数 / Vector 累加器 ----
                Q_shared   = T.alloc_shared([block_M, dim], dtype)
                K_shared   = T.alloc_shared([block_N, dim], dtype)
                V_shared   = T.alloc_shared([block_N, dim], dtype)
                acc_s      = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o      = T.alloc_fragment([block_M, dim], accum_dtype)
                # 行状态统一用 [block_M, 1]（对齐 flash_attn_npuir_dev.py 的 [bm,1] 广播形态）
                m_prev     = T.alloc_fragment([block_M, 1], accum_dtype)
                m_cur      = T.alloc_fragment([block_M, 1], accum_dtype)
                m_cur_safe = T.alloc_fragment([block_M, 1], accum_dtype)
                rs         = T.alloc_fragment([block_M, 1], accum_dtype)
                s_sum      = T.alloc_fragment([block_M, 1], accum_dtype)
                logsum     = T.alloc_fragment([block_M, 1], accum_dtype)
                one_vec    = T.alloc_fragment([block_M, 1], accum_dtype)
                zero_vec   = T.alloc_fragment([block_M, 1], accum_dtype)
                all_masked = T.alloc_fragment([block_M, 1], "bool")   # 同域守卫
                # 掩码临时（列位置 / 可见性 / 哨兵）
                jpos       = T.alloc_fragment([1, block_N], "int32")
                vis        = T.alloc_fragment([1, block_N], "bool")
                vis2d      = T.alloc_fragment([block_M, block_N], "bool")
                sent_buf   = T.alloc_fragment([block_M, block_N], accum_dtype)

                OOB_SENT = -1e38                                   # 有限大数哨兵（PL-1.11）
                NEG_THR  = -1e30                                   # 同域守卫阈值（与哨兵同域）

                # ---- 初始化 ----
                T.clear(Q_shared)                              # padding 行清零
                T.copy(Q[bz, bx*block_M:bx*block_M+real_m, by, 0:dim],
                       Q_shared[0:real_m, 0:dim])              # slice 形态
                T.vmul(Q_shared, s_nat, Q_shared)              # OPT-1
                T.clear(acc_o); T.clear(logsum)
                T.fill(m_prev, -T.infinity(accum_dtype))       # ★ init 被读的 m_prev（先算后存）
                T.vbrc(1.0, one_vec); T.vbrc(0.0, zero_vec)
                T.vbrc(OOB_SENT, sent_buf)                     # 哨兵缓冲

                # ---- KV 循环（静态界 + 流水）----
                for k in T.Pipelined(nkv, num_stages=2):
                    phys = block_table[bz, k]                  # 物理页（R-2）
                    T.copy(K[phys*block_N:(phys+1)*block_N, by, 0:dim],
                           K_shared[0:block_N, 0:dim])         # slice 形态
                    T.gemm(Q_shared, K_shared, acc_s, initC=True, b_transpose=True)

                    # ---- OOB 掩码（向量化；哨兵与守卫同域，§6.4）----
                    # 列 j 有效当且仅当 k*block_N + j < real_kv；无效列 → 哨兵 -1e38
                    T.arange(jpos, [1], 0)                          # jpos = [0,1,...,bn-1]
                    T.vadd(jpos, k * block_N, jpos)                 # 全局 KV 位置 k*bn + j
                    T.vcmp(jpos, real_kv, vis, "lt")                # vis = (jpos < real_kv)
                    T.vbrc(vis, vis2d)                             # [1,bn] → [bm,bn]（vbrc 文档示例2）
                    T.vselect(vis2d, acc_s, sent_buf, acc_s)        # 无效列 → -1e38（值替换，NaN 免疫）
                    # 等价替代（算术惩罚）：pen = clamp(jpos-real_kv+1, 0, 1)·OOB_SENT; T.vadd(acc_s, pen, acc_s)

                    # ---- online softmax：★先算新 m_cur，再用旧 m_prev，末尾存 ----
                    T.reduce_max(acc_s, m_cur, dims=[1], clear=True)     # 块 max（全掩码 → -1e38）
                    T.vcmp(m_cur, NEG_THR, all_masked, "le")             # ★同域守卫（替 v0 失效的 ==-inf）
                    T.vselect(all_masked, zero_vec, m_cur, m_cur_safe)   # 全掩码 → 安全 max=0（无 inf 瞬态）
                    T.vsub(m_prev, m_cur_safe, rs); T.vexp(rs, rs)       # OPT-2: exp(m_prev−m_cur_safe)
                    T.vselect(all_masked, one_vec, rs, rs)               # 全掩码 → rs=1（恒等，保状态）
                    T.vsub(acc_s, m_cur_safe, acc_s); T.vexp(acc_s, acc_s)  # P=exp(S̃−m_cur_safe)（全掩码→0）
                    T.reduce_sum(acc_s, s_sum, dims=[1], clear=True)     # 全掩码 → 0
                    T.vmul(acc_o, rs, acc_o)                             # 重缩放（[bm,1] 广播）
                    T.vmul(logsum, rs, logsum); T.vadd(logsum, s_sum, logsum)
                    T.copy(m_cur, m_prev)                                # ★末尾更新旧 max（先算后存）
                    T.vcast(acc_s, acc_s_cast, round_mode="rint")        # O9 量化点

                    T.copy(V[phys*block_N:(phys+1)*block_N, by, 0:dim],
                           V_shared[0:block_N, 0:dim])
                    T.gemm(acc_s_cast, V_shared, acc_o, initC=False)

                # ---- epilogue：归一化 + 写回（只写真实行）----
                for i in T.Parallel(block_M):
                    logsum[i, 0] = T.if_then_else(logsum[i, 0] == 0, 1, logsum[i, 0])  # 空行守卫
                T.vdiv(acc_o, logsum, acc_o)
                T.vcast(acc_o, acc_s_cast, round_mode="rint")
                T.copy(acc_s_cast[0:real_m, 0:dim],
                       Output[bz, bx*block_M:bx*block_M+real_m, by, 0:dim])
        return mha_decode_no_split
    return _func
```

> 说明：伪代码为**主选方案**的结构骨架，用于固化 buffer 形状、向量化轴（连续尾轴）、循环与同步分层。**不变量（阻塞2）**：被读的缓冲必须先 init——`m_prev` 在循环外在先 init `-inf`，循环内**先算 `m_cur` 再用 `m_prev`、`m_cur` 在循环体末尾才写回 `m_prev`**。**掩码链（建议3）**：`T.arange`/`T.vcmp`/`T.vbrc`/`T.vselect` 的具体 arange 形态（`T.arange(dst, strides, offset)`）与 `[1,bn]→[bm,bn]` bool 广播以 Stage 3 实测为准（`TRAP-vbrc-empty-broadcast`）；算术惩罚为文档化等价替代。**同域守卫（阻塞1）**：`all_masked` 与哨兵 `-1e38` 同域，替代 v0 的 `== -inf`。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|----------|------|
| `T.gemm`（`b_transpose`/`initC`/`size`） | `docs/Tilelang.language/线性代数操作/T.gemm.md`；`examples/flash_attention/flash_attn_npuir_dev.py`；`examples/gemv/example_gemv.py` | ✅ 文档 + 先例（M/N 分形下限为「未文档化假设」） |
| `T.reduce_max` / `T.reduce_sum`（`dims`/`clear`） | `docs/Tilelang.language/规约操作/T.reduce.md`（`clear=False` 静默错误 → 一律 `clear=True`） | ✅ 文档 |
| `T.vexp` / `T.vsub` / `T.vmul` / `T.vadd` / `T.vdiv` / `T.vcast` / `T.vbrc` | `docs/Tilelang.language/数学操作/`、`数据类型转换操作/`；`flash_attn_npuir_dev.py` | ✅ 文档 + 先例 |
| `T.vexp2`（备选，若坚持 exp2 域） | `docs/Tilelang.language/数学操作/T.vexp2.md`（需 `tmpBuffer`；**不支持 bf16**） | ✅ 文档（本设计改用 `T.vexp`，见 OPT-3） |
| `T.copy`（**slice 形态**） | `TRAP-load-nd2nz-strided` / `TRAP-T-copy-region-semantics`（repro 存在）；`_gqa_prefill_fwd_kernel` 先例 | ✅（**必须 slice 形态**） |
| `T.Pipelined` | `flash_attn_npuir_dev.py`；`_gqa_prefill_fwd_kernel` | ✅ 先例 |
| `T.alloc_shared` / `T.alloc_fragment` | `flash_attn_npuir_dev.py`（Developer 先例） | ✅ 先例 |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源 `T.Kernel(m,n,b)`） | §0.6 R1：一维 `T.Kernel(num_logical)` + cid 反解 |
| GPU 专用 API | **Yes**（`threads=`、`T.GemmWarpPolicy`、`T.Pipelined` 的 CUDA 语义） | `threads`/`GemmWarpPolicy` 舍弃；`T.Pipelined` 用 NPU 语义（无 `threads`） |
| GEMM 非整除风险 | **Yes**（`S_q=1` 不被 `block_M=16` 整除；`dim`/`page_size` 分形对齐） | R2：`block_M=16` 覆盖（1 行有效）+ padding 行隔离 + 写回按 `real_m` 裁剪；`dim,page_size ∈ {64,128,256}` 均 16 倍数 ✓ |
| L0C 溢出（`bm×bn×4 > 128KB`） | **No**（`16×256×4 = 16KB`，且 `16×256 ≤ 32768`） | `CONST-cube-gemm-l0c-traffic-floor`（L0C fp32 容量 32768 元素）✓ |
| 分核策略缺失/不适配 | **No**（逻辑核 `B·H ≤ 16 ≤ 20`） | §5.5 分核三要素（实查 20） |
| Host 侧改动输入内容 | **No**（host 仅 dispatch；派生 workspace 不改输入内容） | 见 §4.4 |

### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(S_q,bm), H, B)` 三维 | 一维 `T.Kernel(N, is_npu=True)` | cid 反解（§0.6 R1） |
| 线程配置 | `threads=` 参数 | 无 | 移除（§0.6 R6 / O8 舍弃） |
| 内存层级 | `alloc_shared`/`alloc_fragment` 自动映射 | L1（Cube）/ UB（Vector） | 沿用 `alloc_shared`/`alloc_fragment`，Developer 编译器分层（§4.4） |
| softmax 表达 | `T.macro` + `T.Parallel` + `T.reduce` | v-prefix 链 + `T.reduce` | §0.6 R5 |
| Paged 装载 | `T.copy` 自动跨步 | **slice 形态** `T.copy` | §0.6 R3（`TRAP-load-nd2nz-strided`） |
| KV 循环界 | 动态 `ceildiv(real_kv, bn)`（从不迭代全掩码块） | 静态 `ceildiv(N_kv, page)` + 掩码 + **尾部全掩码块恒等稳定化** | §0.6 R4 |
| 掩码哨兵 | `-T.infinity(fp32)` | 有限大数哨兵 `-1e38`（值替换 `vselect`）+ **同域守卫** → 恒等 `rs=1, P=0` | §6.4 |
| GEMM policy | `T.GemmWarpPolicy.FullRow` | 无对应 | 舍弃 |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/flash_attention/flash_attn_npuir_dev.py` | **高度相似** | Developer 模式 `T.Kernel(n,is_npu=True)` + `T.gemm` + v-prefix online softmax + `T.Pipelined` 的**已验证形态**（本设计直接参照） |
| `examples/flash_attention/flash_attn_npuir.py` | 高 | Expert 双 Scope flash；跨引擎流水（本设计不用，但佐证 attention 结构） |
| `examples/gemv/example_gemv.py` | 高 | GEMV `T.gemm`（M=BLOCK_N/N=1 与 M=1 形态均可编译运行；`naive_gemv_high_perf` 用 `T.gemm(B_shared,A_shared,C_shared)`） |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 中高 | attention 族迁移先例（分形 clamp `max(16,ceil16)`、slice 形态装载、`T.reduce(clear=True)`、行裁剪写回） |
| `examples/vector_add/vector_add/` | 低 | 设计/检视文档格式参照 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Q` | `(B, S_q, H, D)` | fp16/bf16 | BSHD（`S_q=1`） |
| `K` | `(N_kv, H, D)` | fp16/bf16 | 物理 paged cache |
| `V` | `(N_kv, H, D)` | fp16/bf16 | 物理 paged cache |
| `real_seqlen_kv` | `(B,)` | int32 | 逻辑 KV 长度 |
| `block_table` | `(B, N_kv/page_size)` | int32 | 逻辑页→物理页 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Output` | `(B, S_q, H, D)` | fp16/bf16 | 与 Q 同 shape/同布局（BSHD，非转置） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 层级 | 用途 |
|-----------|-------|-------|------|------|
| Q_shared | `(16, D)` | fp16/bf16 | L1 | Q tile（scale 折入后） |
| K_shared | `(page_size, D)` | fp16/bf16 | L1 | K 物理页 tile |
| V_shared | `(page_size, D)` | fp16/bf16 | L1 | V 物理页 tile |
| acc_s | `(16, page_size)` | fp32 | UB(fragment) | 分数 / P |
| acc_s_cast | `(16, page_size)` | fp16/bf16 | UB(fragment) | P 量化（送 gemm2） |
| acc_o | `(16, D)` | fp32 | UB(fragment) | 输出累加器 |
| `m_prev/m_cur/m_cur_safe/rs/s_sum/logsum` | `(16, 1)` | fp32 | UB(fragment) | online 状态行向量（`[bm,1]` 广播形态，对齐 `flash_attn_npuir_dev.py`） |
| `one_vec/zero_vec` | `(16, 1)` | fp32 | UB(fragment) | 全掩码恒等守卫常数（`rs=1`/`m_cur_safe=0`） |
| `all_masked` | `(16, 1)` | bool | UB(fragment) | 全掩码块同域守卫（与哨兵 `-1e38` 同域） |
| 掩码临时 `jpos/vis/vis2d/sent_buf` | `(1,bn)/(1,bn)/(16,bn)/(16,bn)` | int32/bool/fp32 | UB(fragment) | 列索引 / 可见性 / 广播 / 哨兵缓冲（§6.4） |
| `bt_ub`（风险 R-2 绕法） | `(page_size 数,)` | int32 | UB | block_table 行缓存 |

### 4.4 内存搬运路径

```
纯 Vector 段与 Cube 段由 Developer 编译器自动分层：

GM[Q] --slice T.copy--> L1[Q_shared(16,D)] --v-mul(*s_nat, Vector)--> L1[Q_shared]
GM[block_table] --标量/UB 行缓存--> phys_page
GM[K] --slice T.copy--> L1[K_shared(page,D)]
   L1[Q_shared] + L1[K_shared] --T.gemm(b_transpose, Cube: L1→L0A/L0B→L0C)--> UB[acc_s(16,page) f32]
   UB[acc_s] --Vector: mask→reduce_max→exp→reduce_sum→vcast--> UB[acc_s_cast f16]  (P)
GM[V] --slice T.copy--> L1[V_shared(page,D)]
   UB[acc_s_cast] + L1[V_shared] --T.gemm(initC=False, Cube)--> UB[acc_o(16,D) f32]
UB[acc_o] --Vector: /logsum, vcast(rint)--> GM[Output(bz, real_m, by, :)]
```

> **Host 侧约束**：host 只做 dispatch 与 workspace 分配（split 路径），**不改动输入张量内容**（`ascend-constraints.md §4`）。no-split 路径无 host 预处理。

### 4.5 UB / L1 / L0C 内存预算

**UB 层**（Vector 侧，容量 192KB/AIV = 196,608 Bytes）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| acc_s | (16,256) | fp32 | 16384 B |
| acc_s_cast | (16,256) | fp16 | 8192 B |
| acc_o | (16,128) | fp32 | 8192 B |
| 行向量 ×6（`m_prev/m_cur/m_cur_safe/rs/s_sum/logsum`） | (16,1) | fp32 | 384 B |
| 恒等守卫常数 `one_vec/zero_vec` | (16,1) | fp32 | 128 B |
| `all_masked` | (16,1) | bool | 16 B |
| `sent_buf`（掩码哨兵缓冲，§6.4） | (16,256) | fp32 | 16384 B |
| `vis2d`（掩码可见性，§6.4） | (16,256) | bool | 4096 B |
| `jpos`（列索引） | (1,256) | int32 | 1024 B |
| `vis`（列可见性） | (1,256) | bool | 256 B |
| `bt_ub`（R-2 绕法才启用） | (16,) | int32 | 64 B |
| **合计** | | | **≈ 54 KB** / 196,608（192KB）✓（整除域可省掩码 ⇒ ≈33 KB） |

**L1 层**（Cube 操作数，容量 512KB/核 = 524,288 Bytes）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| Q_shared | (16,128) | fp16 | 4096 B |
| K_shared | (256,128) | fp16 | 65536 B |
| V_shared | (256,128) | fp16 | 65536 B |
| **合计**（不含流水双缓冲） | | | **≈ 132 KB** / 524,288（512KB）✓ |
| 含 `num_stages=2` K/V 双槽 | | | ≈ 266 KB ✓ |

**L0C 层**（Cube 累加，容量 128KB = 131,072 Bytes；逐块 32768 元素上限）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| gemm1 `acc_s` | (16,256) | fp32 | 16384 B（`16×256 = 4096 ≤ 32768` 元素 ✓） |
| gemm2 `acc_o` | (16,128) | fp32 | 8192 B |

> auto-multi-buffer 膨胀系数（`CONST-capacity-910B2C` ~1.7× / ×1.10–1.12 结构依赖）下 UB ≈ 92KB（含掩码临时）/ ≈56KB（整除域省掩码）、L1 ≈ 300KB，均在限内。**容量项 ≈0**（见 §1.6.0 roofline）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `real_seqlen_kv[bz]` | 标量 BufferLoad（运行时值） | `0 ~ N_kv`（编译期 `N_kv` 固定）；本设计**不影响循环界**（静态 `nkv` + 掩码，§0.6 R4） |
| `block_table[bz,k]` | 标量 BufferLoad | `0 ~ num_pages-1`（页号动态，影响 `T.copy` 地址） |

> `B/S_q/H/D/N_kv/page_size` 均为编译期常量（`shape_q`/`shape_kv` 由工厂参数决定）；仅 KV 逻辑长度与页映射为运行时数据。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-1],                                   # Output 自动分配
    target="npuir",
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: **混合（MixCV）** — Cube（2 个 `T.gemm`）+ Vector（online softmax / epilogue）。

**判定依据**: 算子含 `S̃=Q̃Kᵀ` 与 `acc_o+=PV` 两个矩阵乘（Cube）+ 逐元素 exp/归约/归一化（Vector），构成 Cube↔Vector 数据依赖 → CV 融合流水；满足 `tilelang-mixcv-skill` Developer MixCV 判定。

### 5.2 Block 划分

```python
block_M = 16        # 分形下限；S_q=1 → 1 真实行（§0.6 R2）
block_N = page_size # 1 tile = 1 物理页，block_table 1:1（§0.6 R3）；∈{128,256}
num_logical = ceildiv(S_q, block_M) * heads * batch   # = heads*batch (S_q=1)
nkv = ceildiv(N_kv, page_size)                        # KV 迭代次数（静态）
```

| workload | B | H | S_q | N_kv | page | block_M | block_N | nkv | num_logical |
|----------|---|---|-----|------|------|---------|---------|-----|-------------|
| single-token-page128 | 1 | 16 | 1 | 512 | 128 | 16 | 128 | 4 | 16 |
| batch2-page256 | 2 | 8 | 1 | 1024 | 256 | 16 | 256 | 4 | 16 |
| longer-cache | 1 | 8 | 1 | 1024 | 256 | 16 | 256 | 4 | 8 |
| shorter-cache | 1 | 8 | 1 | 512 | 256 | 16 | 256 | 2 | 8 |

> 说明：`block_N = page_size` 是为使「1 个 KV tile = 1 个物理页」，令 `block_table` 间接寻址退化为每块 1 次页号读取（§0.6 R3）。若实现期发现 `page_size` 过大（如 256）导致 L1/UB 压力或流水不利，可退化为 `block_N = page_size/2`（则 2 块/页，`num_blockn_in_page=2`，同样支持）。

### 5.3 约束分析

- **对齐约束**: `block_N ∈ {128,256}`（32B 尾轴：128·2B=256B ✓）；`dim ∈ {64,128}`（16 倍数，分形 K≥32 ✓，`dim=64,128≥32` ✓）；`block_M=16`（分形 M≥16 ✓）。
- **UB 容量**: ≈54KB（含掩码临时；含膨胀 ~92KB；整除域省掩码 ≈33KB）< 192KB ✓。
- **L1 容量**: ≈266KB（含双槽）< 512KB ✓。
- **L0C 容量**: `16×256×4 = 16KB < 128KB`，且 `16×256 = 4096 ≤ 32768` 元素 ✓。

### 5.4 注意事项（非整除处理策略）

- **`S_q` 不被 `block_M` 整除**：`S_q=1 < block_M=16` → `num_q=1`，仅 1 真实行；padding 行（1..15）经 `Q_shared` 清零 + 逐行 softmax + 写回裁剪 `real_m = min(block_M, S_q−bx·block_M)` **完全隔离**，不污染输出。
- **`N_kv` 不被 `page_size` 整除**：`nkv=ceil(N_kv/page)`，末块列 `j ≥ real_kv` 由 OOB 掩码置哨兵 `-1e38` → `exp` 为 0（零贡献）。
- **`real_seqlen_kv < N_kv`（尾整块无效，阻塞1）**：静态界多迭代的尾部块**整块被掩码**；用**同域守卫**（`all_masked = m_cur ≤ -1e30`）置 `rs=1`（**恒等**）、`P=0`、`s_sum=0`——本块对已累积 `logsum/acc_o` **零贡献且不清零**（§0.6 R4 / §6.4）。**`rs` 不取 0**：尾部块之后不再有有效块，`rs=0` 会错误清零已累积状态。
- **`real_seqlen_kv = 0`**：全部 `nkv` 块均为全掩码块 → 恒等路径不累积 → `logsum=0, acc_o=0` → epilogue 守卫输出 **0**（与源「零迭代」一致）。
- **写回 extent 匹配**：`T.copy` src/dst 双显式切片同 extent（`TRAP-T-copy-region-semantics` ②），避免越界写坏相邻 GM。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：
  `num_logical_kernels = ceil(S_q/block_M) × H × B = 1 × H × B`（`S_q=1`，`block_M=16`）。
  Manifest 工作负载：`16 / 16 / 8 / 8`（见 §5.2 表）。
- **② 物理核数依据**（**实查，非假设**）：本设计期在目标设备执行
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()         # 实查返回: 20
  NPUUtils.get().get_aivector_core_num()  # 实查返回: 40  (= 20 × 2)
  ```
  → **物理核数 = 20 AI Core（Ascend910_9362 单 die；AIV = 40）**。本算子为 **Cube/混合**算子 → 直接使用 `get_aicore_num()` 返回值 **20**（不翻倍；`CONST-aicore-910_9362`：9362 单 die = 20 AIC / 40 AIV）。
- **③ 规模判定与分核方案**：**逻辑核数 ≤ 物理核数 → 无需适配**。
  依据：manifest 全 workload `num_logical = H·B ∈ {8,16} ≤ 20`（见 §5.2 表），一核一任务、无非整数倍负载不均（`16/20 = 0.8 波次`，4 核空闲但无重复任务）。**方案 = 非持久一维 grid，每核 1 任务**（不引入核内 `T.serial` 多任务循环——`TRAP-DEVMODE-PERSIST-GEMM`）。
  - **边界情形**（超出 manifest 的更大 `H·B`）：若 `H·B > 20`（如 H=32），逻辑核超发会被运行时串行调度。届时按标准要素③选**极大规模核内串行**（固定 20 核，`num_local_tasks = ceildiv(num_logical − cid, 20)`，**循环边界静态**）——但该形态落在 `TRAP-DEVMODE-PERSIST-GEMM` 风险区（Developer+persistent+gemm），故**推荐改走 split-KV dispatch**（本任务范围外，由 wrapper 的 split 路径承接）。本设计对 manifest 不引入核内串行。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务（`bx,by,bz`） | **block 级并行** | `T.Kernel(num_logical, is_npu=True)` | 每核 1 任务（§5.5 无需适配） |
| KV 维 | **tile 级迭代 + 流水** | `T.Pipelined(nkv, num_stages=2)` | online 状态跨 tile 携带依赖；流水重叠 K/V 装载与计算 |
| 元素级（masked/行操作） | **向量化** | `T.Parallel(block_M)` / v-prefix fragment 操作 | §1.6.2 已向量化；无逐元素标量循环 |

### 6.2 循环伪代码

```python
with T.Kernel(num_logical, is_npu=True) as (cid, subid):
    bx, by, bz = decode_cid(cid)            # §0.6 R1
    for k in T.Pipelined(nkv, num_stages=2):   # 静态界（§0.6 R4）
        # Cube: score
        T.copy(K[block_table[bz,k]*block_N:(...)+block_N, by, :], K_shared)
        T.gemm(Q_shared, K_shared, acc_s, initC=True, b_transpose=True)
        # Vector: mask + online softmax
        mask_out_of_range(acc_s, k, real_kv)          # 有限大数哨兵
        online_softmax(acc_s, acc_o, logsum, m_prev, m_cur, rs, s_sum)  # v-prefix
        T.vcast(acc_s, acc_s_cast, round_mode="rint")
        # Cube: output accumulate
        T.copy(V[block_table[bz,k]*block_N:(...)+block_N, by, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, initC=False)
    # epilogue
    normalize_and_store(acc_o, logsum, Output, bz, by, bx, real_m)   # v-prefix
```

### 6.3 流水线优化

- `T.Pipelined(nkv, num_stages=2)`：K/V 物理页装载（MTE2）与 Cube/Vector 计算重叠。
- **stage 数**：`num_stages=2` 使 L1 的 K/V 双槽 ≈ 2×`(page·D·2)`；smoke（page=128,D=128）= 128KB ✓；batch2（page=256,D=64）= 128KB ✓。若追求更深流水需复算 L1（≤512KB）——交由 Stage 4。
- KV 迭代次数少（`nkv ∈ {2,4}`），流水收益有限；`num_stages` 为保留调优旋钮（Stage 4）。

### 6.4 尾块 / 掩码处理

- **OOB 掩码（非因果，通用）**：列 `j`（全局 KV 位置 `k·block_N + j`）有效当且仅当 `< real_seqlen_kv[bz]`。用**有限大数哨兵 `-1e38`**（`PL-1.11`）经 `T.vselect`（**值替换，NaN 免疫**）置换无效列：`P = exp(−1e38 − m) → 0`（精确零），规避 `−inf − (−inf) = NaN`（`T.arange` 构造列索引 + `T.vcmp` + `T.vbrc` 广播 + `T.vselect`）。**哨兵与守卫必须同域**（v0 的 `== -inf` 守卫与 `-1e38` 哨兵不同域，永不触发——本版修正）。当 `N_kv % page_size == 0` 且 `real_kv == N_kv`（manifest 情形）时**无实际掩码列**，trace-time 可省该链。
- **尾部全掩码块（静态界新引入；本版新增，阻塞1）**：`real_kv` 为有效**前缀**长度 ⟹ 静态界比源动态界多迭代的块**必为尾部整块全掩码**（`real_kv=0` 时全部块）。统一处理（`all_masked = m_cur ≤ -1e30`，与哨兵 `-1e38` **同域**）：
  - `rs = all_masked ? 1 : exp(m_prev − m_cur)`（**恒等**，保状态——尾部块之后不再有有效块，`rs=0` 会错误清零已累积的 `logsum/acc_o`）；
  - `m_cur_safe = all_masked ? 0 : m_cur` ⟹ `P = exp(S̃ − m_cur_safe) = exp(−1e38) = 0`；`s_sum = 0`。
- **causal 掩码**（源支持，manifest 全 `False`）：`q_pos = bx·block_M + i`，`kv_pos = k·block_N + j`，可见当 `q_pos ≥ kv_pos`。用算术惩罚形 `pen = vmin(q_pos − kv_pos, 0)·PEN`（`PEN=1e38`，`PL-1.11` 符号约定变体）加到分数上。（causal 下有效列同为前缀，尾部全掩码块走同一恒等路径。）
- **全掩码行 / 块**：`all_masked` → `rs=1, P=0, s_sum=0`（见上）；总 `logsum==0`（如 `real_kv=0`）→ epilogue 守卫输出 0。
- **行裁剪**：写回 `real_m` 行（`S_q=1 → 1` 行）。
## 7. 同步策略

### 7.1 同步模式

**模式**: **自动同步**（Developer 模式）。

### 7.2 同步点说明

Developer 模式下，`T.gemm`（Cube）与 v-prefix 向量链（Vector）之间的跨引擎依赖、`T.Pipelined` 的 stage 间依赖由编译器（auto CV-split + pipe 调度）**自动插入同步**，无需手动 `sync_block_set/wait`。本设计**不使用** `T.Scope("Cube")/T.Scope("Vector")` 或 `T.rs("PIPE_*")`（那是 Expert 形态）。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| Cube `T.gemm` → Vector softmax | 自动（编译器 CV split 插同步） | score/P 跨引擎依赖 |
| Vector P → Cube `T.gemm`（gemm2） | 自动 | P 跨引擎依赖 |
| KV 循环 stage 间（`T.Pipelined`） | 自动（多缓冲 + 同步） | K/V 装载与计算重叠 |
| **`block_table` 间接寻址的 `T.copy` 源地址**（`phys=block_table[bz,k]`，`k` 为流水循环变量） | 自动（多缓冲 + 同步）；**但地址本身随循环变量变化**——地址计算属该 stage 的标量前导 | 地址取值（GM 标量读）须在对应 stage 的装载（MTE2）**之前**完成，装载写入的多缓冲槽与计算读取的槽须由编译器按 stage 配平；`k` 的地址依赖与 `num_stages=2` 的槽位选择是**独立于数据缓冲**的 stage 依赖（R-2 未文档化能力）。**退路**：`num_stages=1`，或循环头把 `block_table[bz,:]` 拷入 UB（`bt_ub`）+ `T.alloc_local` 标量读（§9.1-R-2）；Stage 3 首验专项检查 |
| epilogue 写回 | 自动 | 累加器→GM |

> 若 Stage 3 发现自动同步在 `T.Pipelined` + 动态地址（`block_table` 间接）下不足，可回退把 KV 循环改为 `T.serial` + 显式双缓冲（仍为 Developer 自动同步）或评估 `T.Pipelined` 的 num_stages=1。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,   # 沿用源（L26）
}
```

> 不设 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` / `NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`——那两项是 **Expert** 形态为保护手动 Scope/workspace 而设（`PL-1.16-expert-dualscope-bypass`）；本设计为 Developer，保留编译器自动多缓冲（有利于流水）。

### 7.4 CV 融合设计（MixCV）

本 kernel 是 **Cube + Vector 融合**（Developer 自动 CV split），KV 循环体内每块的引擎依赖链：

```
(Cube) gemm1: acc_s = Q̃ Kᵀ        [K_shared 就绪 → L0C]
  │  S/P 跨引擎依赖（自动同步）
(Vector) mask → reduce_max→m_cur → rs=exp(m_prev−m_cur)/全掩码恒等守卫 → P=exp(S̃−m_cur_safe)
         → reduce_sum → logsum/acc_o 重缩放（·rs）→ **末尾 m_prev←m_cur**   [acc_s 在 UB]
  │  P 跨引擎依赖（自动同步）
(Cube) gemm2: acc_o += cast(P) V  [acc_s_cast 就绪 + V_shared 就绪 → L0C]
```

> **状态顺序（阻塞2）**：`m_prev` init `-inf`；每块**先算新 `m_cur`，再用旧 `m_prev` 算 `rs`，最后（gemm2 之前或之后、下一块之前）写回 `m_prev←m_cur`**。v0 的「先 `copy(m_cur,m_prev)` 再 `reduce_max`」形态首迭代读到未初始化 `m_cur`，已废弃。

- **侧归属**：`T.gemm` 归 Cube（L1 操作数 → L0A/L0B → L0C fp32）；mask/softmax/cast/normalize 归 Vector（UB fragment + v-prefix）。`acc_s`（P）是跨引擎中间量，驻 **UB**（Developer 自动分层，**不经 `store_fixpipe` GM 往返**——这是相对 Expert 形态的一个优势：`CONST-store-fixpipe-gm-only` 的 GM 往返在本设计不出现）。
- **流水**：`T.Pipelined(num_stages=2)` 让下一块的 K 装载（MTE2）与当前块计算重叠；跨引擎同步由编译器自动插入。
- **边界**：本设计**不**使用 `T.Scope("Cube")/T.Scope("Vector")`、`T.rs("PIPE_*")`、`T.sync_block_set/wait`（Expert 专属，见 §7.1）；若 Stage 3 实测自动 CV split 在本结构下失效（R-1），回退路径 = 按 `tilelang-mixcv-skill` 的 Expert 双 Scope 形态重设计（跨引擎 GM workspace + per-slot flag），并遵循 `PL-1.16` 的 Expert 硬边界（关闭 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION` 等）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（**优先移植源仓参考实现/测试基准**），**不得复刻 §0.6 的 NPU 算法**——保证验证独立性。
> 唯一依据 = `examples/TileOPs/tests/ops/test_mha_decode_paged.py::MhaDecodePagedTest.ref_program`（paged 重组逻辑 KV + `F.scaled_dot_product_attention`）。

```python
import math
import torch
import torch.nn.functional as F

def golden_mha_decode_paged(q, k, v, real_seqlen_kv, block_table, page_size):
    """Paged MHA decode reference (移植源仓 ref_program).

    q: (B, S_q, H, D); k/v: (N_kv, H, D) fp16/bf16;
    real_seqlen_kv: (B,) int32; block_table: (B, num_pages) int32.
    返回 (B, S_q, H, D)。独立于 NPU 算法：paged 重组到逻辑布局后走 SDPA。
    """
    batch, seqlen_q, heads, dim = q.shape
    seqlen_kv = k.shape[0]
    out_list = []
    for i_b in range(batch):
        k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
        v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
        num_pages = math.ceil(real_seqlen_kv[i_b].item() / page_size)
        for i_paged in range(num_pages):
            start_pos = block_table[i_b, i_paged].item() * page_size
            end_pos = min(start_pos + page_size, seqlen_kv)
            page_len = end_pos - start_pos
            k_logical[i_paged*page_size:i_paged*page_size+page_len] = k[start_pos:end_pos]
            v_logical[i_paged*page_size:i_paged*page_size+page_len] = v[start_pos:end_pos]
        L = real_seqlen_kv[i_b].item()
        k_b = k_logical[:L].unsqueeze(0).transpose(1, 2)   # (1,H,L,D)
        v_b = v_logical[:L].unsqueeze(0).transpose(1, 2)
        q_b = q[i_b:i_b+1].transpose(1, 2)                 # (1,H,S_q,D)
        o = F.scaled_dot_product_attention(q_b, k_b, v_b).transpose(1, 2)
        out_list.append(o.contiguous())
    return torch.cat(out_list, dim=0)
```

**可比性/比对方法**（移植源测试的 `_maxdiff_cosine_compare`）：`max_diff < atol` 且 `cosine_similarity(reshape(B,-1), dim=-1).min() > 0.99`。

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-2 | 1e-2 |

（源测试用 `atol=0.001` + cosine>0.99；本设计沿用 `atol=1e-2/rtol=1e-2` 与 `cos>0.99` 双判。**§1.6.1 OPT 项已由 `verify_equiv.py` 机器验证违反率 0**，D-1 结果见 §1.6.1 表。）

### 8.3 L0 门槛测试计划（blocking）

| 用例 | 配置（来自 manifest） | 目的 | 通过判据 |
|------|---------------------|------|----------|
| L0-1 | `(1,16,1,512,128,128)` fp16 smoke | 主路径（4 KV 块，D=128） | max_diff<atol + cos>0.99 |
| L0-2 | `(2,8,1,1024,64,256)` fp16 | 多 batch + page 256 + D=64（8 核任务） | 同上 |
| L0-3 | `(1,8,1,1024,64,256)` fp16 | page 256、`num_logical=8` | 同上 |
| L0-4 | `(1,8,1,512,64,256)` fp16 | 短 cache（`nkv=2`） | 同上 |
| L0-5（契约） | 输出 shape/dtype == q；`block_table` 非恒等（打乱页序） | paged 寻址正确性 | shape/dtype 一致 + 精度门 |
| **L1-1（边界·尾整块无效）** | `(1,8,1,512,64,256)` fp16 + `real_seqlen_kv=200`（`page=256` → 块 0 部分掩码、**块 1 全掩码**） | **尾部全掩码块恒等稳定化**（阻塞1 拦截） | max_diff<atol + cos>0.99（输出 == 同配置 golden，**非 0**）；无 NaN/±inf |
| **L1-2（边界·零逻辑长度）** | `(1,8,1,512,64,256)` fp16 + `real_seqlen_kv=0`（**全部块全掩码**） | 全掩码块 → 输出 0 | `max_diff == 0`、`cos` 未定义时按绝对差判；**无 NaN/±inf**（源零迭代行为） |
| **L1-3（边界·末块部分掩码）** | 同上 + `real_seqlen_kv=300`（块 1 部分掩码） | 部分掩码（`all_masked=false`）走正常 softmax | max_diff<atol + cos>0.99 |

> 分层套件（L1/L2/Boundary、bf16 覆盖、causal、空行）交由 `tilelang-op-develop` 按 §9 风险清单枚举。**L1-1/L1-2 为阻塞1 缺陷的专用拦截用例**：v0 方案在此二例下会产出 inf / 错误非零值。

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 | 类别 | 影响 | 缓解 / 依据 |
|---|------|------|------|------------|
| R-1 | **Developer + 非持久 1D grid + `T.gemm`** 的支持边界 | 编译/运行时 | 若不支持则需切 Expert | 先例 `flash_attn_npuir_dev.py`（Developer MixCV 已验证）；**不引入 persistent**（`TRAP-DEVMODE-PERSIST-GEMM`）；Stage 3 首编译验证 |
| R-2 | **核内 GM 标量索引**（`real_seqlen_kv[bz]`、`block_table[bz,k]`，`k` 为运行时循环变量）在 npuir 的支持 | 前端能力（**未文档化**） | 装载地址构造失败 / 静默错读 | **Stage 3 首验**；绕法 = 循环头把整行 `block_table`/`real_seqlen_kv` 拷入 UB（`T.copy(block_table[bz,:], bt_ub)`）→ `T.alloc_local` 标量读取（`bt_ub_local[0] = bt_ub[k]`）。极端绕法 = host 预派生物理页偏移 workspace（**仅派生元数据，不改输入内容**，符合 `ascend-constraints.md §4`）。**标注为「未文档化假设 + 估算依据」，不得作为既定结论** |
| R-3 | **GEMM M/N 分形下限**（M=16 pad、`S_q=1`）为**未文档化假设** | 前端能力 | 编译失败 / 结果错 | 依据 `docs/.../T.gemm.md §2.2.2`（2 维，未标分形数值）+ `examples/gemv` + prefill `tmc=max(16,ceil16)`；`block_M=16` 满足假设；**plan B** = §1.6.3 备选 #2（KV-as-M GEMV）或把 `block_M` 提到 32/64（多 pad，仍正确） |
| R-4 | `T.copy` 跨步/间接 GM 区域装载 | 语义陷阱 | 静默乱值 | **必须 slice 形态**（`TRAP-load-nd2nz-strided`，repro 存在）；`load_nd2nz`/base+size 形态禁用 |
| R-5 | 掩码用 `-inf` 触发 `-inf-(-inf)=NaN`；**且静态界新引入尾部全掩码块**（源动态界从不迭代） | 数值 | NaN / 输出错误（±inf 或错误归零） | 有限大数哨兵 `-1e38`（`PL-1.11`）+ **同域守卫** `all_masked=m_cur ≤ -1e30` → 恒等 `rs=1, P=0, s_sum=0`（§0.6-R4/§6.4）；`real_kv<N_kv`/`real_kv=0` 由 L1-1/L1-2 用例拦截（§8.3） |
| R-6 | `T.reduce(clear=False)` 静默错误 | 数值 | 静默错值 | 全部 `clear=True`（`T.reduce.md §2.3`；prefill M14 先例） |
| R-7 | 动态 KV 长度 / KV 循环界 | 前端能力 | trip-count 限制 | 静态界 `nkv` + 掩码（§0.6 R4）；**规避动态 trip count** |
| R-8 | `T.vexp2` 不支持 bf16 + 需 tmpBuffer | 精度/dtype | bf16 失败 | 改用 `T.vexp`（OPT-3，fp16/fp32）；bf16 经 `vcast` fp32 中转链（`TRAP-fp16-opmath-golden` / `ALG-elementwise` PL-1.17） |
| R-9 | 发射项常数（0.5µs/op）在 9362 **未重验** | 性能估计 | roofline 不确定 | 标注「未实测假设」（`CONST-vector-launch-overhead` 9362 注记）；§1.6.3 实验裁决 |
| R-10 | padding 行（1..15）在 gemm/softmax 中的垃圾值 | 数值 | 若行隔离失效则污染 | 行级 reduce/重缩放/写回天然隔离 + `Q_shared` 清零；L0 用全零/随机输入双验 |
| R-11 | `bf16` 输出经 `vcast(rint)` 舍回 | 数值 | ~2 ulp | 沿用 fp32 中转链；门禁 atol=1e-2 覆盖（manifest decode 仅 fp16，bf16 为附加覆盖） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block 过大 | 编译失败 | `block_M=16`、`block_N=page_size` 已控；按 `CONST-capacity-910B2C` 复算 |
| L0C 溢出 | `bm×bn>32768` | segfault | `16×256=4096` ✓ |
| MTE DDR fault | `T.copy` src/dst extent 不匹配 | 越界写坏相邻 GM | 双显式同 extent 切片（`TRAP-T-copy-region-semantics`） |
| 装载乱值 | 非 slice 形态跨步装载 | 静默错值 | 强制 slice 形态 |
| NaN | `-inf-(-inf)` / 静态界新引入的尾部全掩码块 | NaN / ±inf / 错误非零输出 | 有限大数哨兵 `-1e38` + **同域守卫** `all_masked` → 恒等 `rs=1, P=0, s_sum=0`（§6.4） |

### 9.3 特殊场景处理

- **非整除**：`S_q<block_M`（pad+隔离）、`N_kv%page_size≠0` 或 `real_kv<N_kv`（OOB 掩码，尾部整块全掩码 → 同域守卫恒等）、`page_size%block_N≠0`（退化 `block_N=page_size/2`）。
- **极小 shape**：`N_kv<page_size`（`nkv=1`，掩码）；`real_kv=0`（全掩码块 → 恒等路径 → 输出 0）。
- **混合精度**：fp16 主；bf16 经 fp32 中转链。
- **causal**（manifest 外）：算术惩罚掩码（§6.4）。
- **超大 `H·B>20`**（超出 manifest）：走 split-KV dispatch（本任务外）。

### 9.4 NPU 能力缺口（候选，未确证）

以下为设计期识别、**尚未确证**的能力不确定项（Stage 3 首验后若确证为缺口，按 `AGENTS.md`「Compiler Capability Gap Rule」登记到 `.agents/evolution/capability-gaps.md`）：R-2（核内 GM 标量索引）、R-3（GEMM M/N=16 分形下限）、R-7（动态 trip count）。**本设计不预先登记为缺口**（无确证证据，避免投机条目）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mha_decode_paged/_mha_decode_no_split_kernel/
├── DESIGN.md                 # 本设计文档
├── verify_equiv.py           # D-1 等价性机器验证脚本（§1.6.1）
├── REVIEW.md                 # Stage 2 检视（后续阶段产出）
├── RETROSPECTIVE.md          # Stage 1 复盘（本 Agent 产出）
└── _mha_decode_no_split_kernel.py   # 待实现（Stage 3）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档（含 §0 迁移分析、§1.6 算法调研与优化） |
| `verify_equiv.py` | ✅ 已完成 | D-1 等价验证（3 采纳项全部 EQUIV_PASS） |
| `_mha_decode_no_split_kernel.py` | ⬜ 待实现 | 算子实现（Stage 3） |

### 10.3 命名规范

- 项目目录：`mha_decode_paged`（snake_case）
- 算子目录：`_mha_decode_no_split_kernel`（源 JIT 工厂函数名）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ D-1 等价验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1，移植源仓 ref_program）
3. ⬜ 算子实现（`_mha_decode_no_split_kernel.py`）+ L0 精度
