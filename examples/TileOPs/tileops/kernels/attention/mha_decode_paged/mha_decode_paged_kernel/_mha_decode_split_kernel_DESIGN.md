# `_mha_decode_split_kernel` 算子设计文档（Paged MHA Decode, split-KV + LSE combine, NPU 迁移）

> 任务：迁移 GPU TileLang 算子 `MultiHeadAttentionDecodePagedWithKVCacheFwdOp` 的 JIT 工厂函数
> `_mha_decode_split_kernel`（源仓 `/home/developer/TileOPs/tileops/kernels/attention/mha_decode_paged.py`
> L117–379，含 `_mha_decode_split` 与 `combine` 两个 `@T.macro`）到 `target="npuir"`（华为昇腾
> Ascend910_9362 单 die）。编程模式 = Developer。
> 本设计仅覆盖 **split-KV**（分段 online softmax + 段级 LSE 合并）路径；no-split 变体为同任务前序函数
> （`examples/mha_decode_paged/_mha_decode_no_split_kernel/`，已完成 Stage 1–3），本设计复用其已验证的
> MixCV 结构、paged slice 装载与静态界掩码结论。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架（0.1–0.4）→ 硬件耦合性判定（0.5）→ NPU 算法重设计（0.6）」组织（方法论见
> `.agents/skills/tilelang-op-design/references/migration-analysis.md`）。本章结论驱动 §1–§11 的全部设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：paged KV-cache 多头注意力 **decode**（BSHD 布局），KV 序列被切成 `num_split` 个**连续段**，
每段独立做 online-softmax 得到归一化的段内输出 $o_i$ 与段级 log-sum-exp $l_i$，再按段级方案合并：

$$
o[b,:,h,:]=\sum_{i=0}^{n_s-1}\exp\!\big(l_i-\operatorname{LSE}\big)\,o_i,\qquad
o_i=\frac{\sum_{j\in\text{seg}_i}\exp(s_j)\,V_j}{\sum_{j\in\text{seg}_i}\exp(s_j)},\quad
l_i=\ln\!\!\sum_{j\in\text{seg}_i}\exp(s_j),\quad
\operatorname{LSE}=\ln\!\!\sum_{i}\exp(l_i)
$$

其中 $s_j=q[b,:,h,:]\cdot K_{b,h,j}/\sqrt{D}$，$K_{b,h}$ 是**逻辑 KV 序列**（长度 `real_seqlen_kv[b]`），
物理上以 **paged（block-table 寻址）** 形式存放于 `[N_kv, H, D]`。合并式是**精确恒等**（分段 softmax 的重组），
与 no-split 单段结果一致到 fp 舍入。decode 场景 `S_q=1`。$n_s$=`num_split`（源 host 侧默认 4）。

**I/O 契约**（源 `_mha_decode_split` / `combine` 签名，L203–212 / L300–304）：

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `Q` | `[batch, seqlen_q, heads, dim]` | float16 / bfloat16 | query（BSHD） |
| `K` / `V` | `[seqlen_kv, heads, dim]` | 同 Q | **物理** paged KV cache（`N_kv=seqlen_kv`） |
| `real_seqlen_kv` | `[batch]` | int32 | 每 batch 的**逻辑** KV 长度 |
| `block_table` | `[batch, seqlen_kv // page_size]` | int32 | 逻辑页 → 物理页映射 |
| `glse` | `[batch, heads, num_split, seqlen_q]` | **同 Q（输入 dtype）** | 段级 LSE（源为 log2 域） |
| `Output_partial` | `[batch, seqlen_q, heads, num_split, dim]` | 同 Q | 段级归一化部分输出 |
| `split_length` | `[batch, num_split]` | int32 | **累积**段边界（host 传入 `acc_split_length`） |
| `Output` | `[batch, seqlen_q, heads, dim]` | 同 Q | 最终输出（**与 Q 同 shape/同布局**，非转置） |

**规约语义**：
- 分段内部与 no-split 逐字相同（同 `Softmax` 宏语义）：分数在 `dim` 轴点积（Cube 的 K 维，fp32 累加）；
  softmax 在段内 KV 轴做（行 max/sum，逐行 dim=1）；输出对段内 KV 轴归约（GEMM 的 K 维 = 块累加）。
- **段内 `online softmax` 的累加分组是精确在线等价**（重缩放 `scores_scale=exp2(max_prev·scale−max_cur·scale)`，
  取**当前块** max 而非 running max），必须逐项保持。
- **合并语义（关键）**：段级 LSE 与部分输出按 `exp2(l_i − LSE)` 加权求和。LSE 由 `combine` 计算：
  `lse_max = max_i l_i`；`lse_logsum = Σ_i exp2(l_i − lse_max)`；`LSE = log2(lse_logsum) + lse_max`。
  这是**两阶段归约（max→sum→加权和）**，不是单次可结合归约。

**dtype 语义**：
- 输入/输出 fp16/bf16；中间累加 `accum_dtype="float"`（fp32）。
- softmax 在 **exp2 域**：`scale = make_log2e_scale(dim) = (1/√D)·log2(e)`。
- `glse` 以**输入 dtype 存储**（源 `glse: T.Tensor([batch,heads,num_split,seqlen_q], dtype)`）——段级 LSE
  经一次 fp16/bf16 量化往返，是源算法的语义点（合并权重噪声来源）。
- `Output_partial` 同样以输入 dtype 存储（段内归一化输出量化一次）。
- `acc_s`（P）在送第二个 GEMM 前 cast 回输入 dtype（`T.copy(acc_s, acc_s_cast)`）——**P 量化点**保持。

**边界语义**：
- **OOB（非因果）**：段内块中列 `j` 全局位置 `k_global*block_N + j < real_seqlen_kv` 为有效，否则 `-inf`
  （`MMA0` L157–159，`seqlen_kv=real_seqlen_kv[bid]` 动态）。
- **分段为空 / 超界段**：源用**动态** `loop_range = blocks_in_split`（L267；注意 L256–264 的
  `loop_range` 计算被 L267 **逐字覆盖，是死代码**）。源注释明确：只迭代 `real_seqlen_kv` 内的块，
  这样短 batch 不会产生**空段**（全 -inf → combine 中 NaN）。即源**从不迭代全掩码块**。
- **段级 `Softmax` 的 `-inf` 守卫**：`scores_scale = if scores_max == -inf then 0 else exp2(...)`（L261–263）
  ——源的全掩码块守卫取 `rs=0`；因动态界不迭代全掩码块，该分支实际为死代码。
- **空段 LSE**：`logsum_safe = if logsum==0 then 1`；`glse = log2(1) + scores_max*scale`。若全段空，
  `scores_max=-inf` → `glse=-inf`（NaN 风险路径，源靠动态界规避）。
- **causal**（源支持，manifest 全 `False`）：源 split 的 causal 分支带「Handle causal split case」占位注释
  （L151）——`MMA0` 用 `mid*block_M+i >= k_global*block_N+j`（全局位置），但未闭环验证，属**源实现未完成**项。
- **NaN/inf 传播**：源掩码用 `-T.infinity(fp32)`；`-inf − (-inf) = NaN`。迁移侧改用**有限大数哨兵** +
  **同域守卫**（见 §0.6 / §6.4）。

**语义保持基线**：§8.1 的 golden 以**本节语义**为唯一依据实现（优先移植源仓测试的 `ref_program`：
paged 重组 + SDPA），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `Q` | `(B, S_q, H, D)` | fp16/bf16 | BSHD query |
| `K` / `V` | `(N_kv, H, D)` | fp16/bf16 | 物理 paged cache |
| `real_seqlen_kv` | `(B,)` | int32 | 逻辑 KV 长度 |
| `block_table` | `(B, num_pages)` | int32 | 逻辑页→物理页 |
| `glse` | `(B, H, num_split, S_q)` | fp16/bf16 | **工作区（输入）**：段级 LSE |
| `Output_partial` | `(B, S_q, H, num_split, D)` | fp16/bf16 | **工作区（输入）**：段级部分输出 |
| `split_length` | `(B, num_split)` | int32 | **工作区（输入）**：累积段边界 |
| `Output` | `(B, S_q, H, D)` | fp16/bf16 | 输出（与 Q 同 shape/同布局） |

工厂外部签名（源 L117）：`_mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
-> _func(block_M, block_N, num_split, num_stages, threads)`。NPU 脚手架（接口契约）
`examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged.py` L106–117 已按 K9
以无 `threads` 形式调用：`(...)(block_M, block_N, num_split, num_stages)(Q,K,V,real_seqlen_kv,block_table,glse,Output_partial,acc_split_length)`。

### 0.3 实现算法解读（怎么算）

**Phase A —— `_mha_decode_split` 宏**（源 L202–297）：`T.Kernel(ceildiv(S_q, block_M), heads*batch, num_split, threads=128)`
（**三维 grid**，`(bx=query-block, by=heads*batch 合并维, bz=split-id)`）。

**Phase B —— `combine` 宏**（源 L299–357）：`T.Kernel(ceildiv(S_q, block_M), heads, batch, threads=128)`
（**三维 grid**，`(bx, by=head, bz=batch)`）。两宏在同一 `@T.prim_func` 内**顺序调用**（L373–375）——
GPU 上产生**两次 kernel launch**，其间的隐式全局同步构成「所有段就绪」屏障。

**计算步骤分解**（每语句归类）：

| 步 | 计算 | 输入 | 输出 | 对应语义 |
|----|------|------|------|----------|
| A1 | `split_length_shared ← split_length[bid,:]`（累积边界，`disable_tma`） | split_length | `[num_split]` | 段划分元数据 |
| A2 | `start_block_sid = sid>0 ? cum[sid-1]//block_N : 0` | cum | 标量 | 段起始块 |
| A3 | `blocks_in_split = sid>0 ? ceildiv(cum[sid]−cum[sid-1], bn) : ceildiv(cum[0], bn)` | cum | 标量 | **段长（块数）** |
| A4 | `offset = sid>0 ? cum[sid-1]//bn : 0`；`k_global = k + offset` | cum | 标量 | 段内→全局块号 |
| A5 | `T.copy(Q[bid, mid*bm:(mid+1)*bm, hid, :], Q_shared)`（`disable_tma`） | Q | Q_shared | 载入 query tile |
| A6 | `page_idx=k_global//nbpp; boff=block_table[bid,page_idx]*nbpp + k_global%nbpp` | block_table | 标量 | **paged 寻址** |
| A7 | `T.copy(K[boff*bn:(boff+1)*bn, hid, :], K_shared)` | K | K_shared | 载入 K tile |
| A8 | `acc_s[i,j] = if_then_else(可见, 0, -inf)`（`T.Parallel`） | 索引 | acc_s | causal/OOB 掩码 |
| A9 | `T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)` | Q,K | acc_s | $S=QK^{\top}$（Cube） |
| A10 | `Softmax(acc_s, ...)`（宏）：保存 `max→max_prev`；`reduce_max`；`scores_scale=if max==-inf then 0 else exp2(max_prev·s−max·s)`；`acc_s=exp2(acc_s·s−max·s)`；`reduce_sum`；`logsum=logsum·scale+s_sum`；`copy(acc_s,acc_s_cast)` | acc_s | acc_s/sum/scale/logsum | 段内 online softmax |
| A11 | `rescale(acc_o, scores_scale)` | acc_o,scale | acc_o | 旧累加器重缩放 |
| A12 | `T.copy(V[boff*bn:(boff+1)*bn, hid, :], V_shared)`；`T.gemm(acc_s_cast, V_shared, acc_o)` | V | acc_o | $O\mathrel{+}=PV$（Cube） |
| A13 | `acc_o = if logsum==0 then 0 else acc_o/logsum` | acc_o,logsum | acc_o | **段内归一化** |
| A14 | `logsum = log2(logsum_safe) + scores_max*scale` | — | glse 值 | **段级 LSE（log2 域）** |
| A15 | `copy(logsum, glse[bid,hid,sid,rows])`；`copy(acc_o, Output_partial[bid,rows,hid,sid,:])` | — | glse/Output_partial | **段结果落 GM** |
| B1 | `copy(glse[bz,by,:,rows], lse_shared)`；`copy(lse_shared, lse_local)` | glse | `[num_split,bm]` | 载入段级 LSE |
| B2 | `lse_max_local = reduce_max(lse_local, dim=0)` | lse_local | `[bm]` | 段间 max |
| B3 | `lse_logsum += Σ_i exp2(lse_i − lse_max)`；`lse_logsum = log2(lse_logsum)+lse_max` | — | LSE | 段间 logsumexp |
| B4 | 循环 i：`copy(Output_partial[bz,rows,by,i,:], po_shared)`；`scale=exp2(lse_i − LSE)`；`o_accum += po*scale` | Output_partial | o_accum | **加权合并** |
| B5 | `T.copy(o_accum_local, o_shared)`；`copy(o_shared, Output[bz,rows,by,:])` | — | Output | 写出 |

**数据流与内存访问模式**（源硬件视角）：

```
Phase A（每 (query-block, batch*head, split) 任务）:
GM[Q,K,V] --copy--> SMEM --gemm--> frag[S/O] --Softmax--> frag
frag[O] --/logsum--> SMEM --> GM[Output_partial]     ; frag[lse] --> GM[glse]
Phase B（每 (query-block, head, batch) 任务）:
GM[glse] --copy--> SMEM/frag --reduce--> LSE
GM[Output_partial] --copy--> SMEM/frag --加权累加--> frag[O] --> GM[Output]
```

**host 侧逻辑**（`MHADecodePagedKernel.forward`，源 L543–578）：
1. **dispatch**：`real_max = real_seqlen_kv.max()`；`threshold = num_split*block_N`；
   `real_max < threshold` → **no-split** 路径，否则 **split** 路径。
2. split 路径派生元数据：`chunk_size = real_max // (num_split*block_N) * block_N`；
   `split_length[:, :] = chunk_size`（末列 = `real_max − (num_split−1)*chunk_size`）；
   `acc_split_length = cumsum(split_length, dim=1)`（**累积**边界传入 kernel 的 `split_length` 形参）。
3. 分配 `glse`/`Output_partial`（`torch.zeros`，`dtype=self.dtype`）**工作区**并传入。
   > **关键观测**：manifest 的 `gen_inputs` 令 `real_seqlen_kv ≡ seqlen_kv`（=512/1024），
   > `threshold = num_split*block_N = 4*64 = 256` ⟹ **manifest 全部 workload 走 split 路径**
   > （no-split 仅在 `real_max < 256` 时生效）。故本函数是 manifest 的主路径。

**编译配置**：`@tilelang.jit(out_idx=[-1], pass_configs={TL_ENABLE_FAST_MATH: True}, compile_flags=["-O3","-DENABLE_BF16"])`（L121–126）。

### 0.4 优化手段解读（为什么快）

> 识别不出来 ≠ 不存在；未识别的优化会在迁移中被静默丢弃。

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **split-KV 两级分解 + LSE 合并** | 长 KV 并行度（GPU 数千 SM） | 段间独立 online softmax + `glse`/`Output_partial` GM 工作区 + 第二 kernel 合并 | 多 CTA + 两次 launch 全局屏障 | **重新设计**（NPU 20 核 + 一维 grid + 单 prim_func 单 kernel；见 §0.5/§0.6-R5） |
| O2 | online softmax（段内 flash 结构） | 消 materialize O(S²)，中间缓冲 O(tile) | 分块 running max/sum + 重缩放 | **无（纯算法层）** | 可移植（保留） |
| O3 | exp2 域 softmax（log2e 折入 scale） | 廉价 exp2 替代 exp | `scale=(1/√D)·log2e` | 无 | 可移植（等价替换，§1.6.1 OPT-3） |
| O4 | SMEM tiling（`alloc_shared`） | 减少 GM 重复访问，喂 Cube | Q/K/V/O tile 驻留 | shared memory | 硬件强相关（→ L1/UB） |
| O5 | Tensor Core MMA（`T.gemm`） | 矩阵算力 | `Q@K^T`、`P@V` | tensor core | 硬件强相关（→ Cube） |
| O6 | fragment 累加器（`alloc_fragment`） | 数据驻留寄存器 | `acc_s/acc_o` | 大寄存器堆 | 硬件强相关（→ L0C/UB） |
| O7 | 多级异步流水（`T.Pipelined(num_stages=2)`） | 访存/计算重叠 | KV 循环流水 | 异步拷贝引擎 | 硬件强相关（→ MTE） |
| O8 | **三维 grid**（A: `(q, H·B, split)`；B: `(q, H, B)`） | 任务划分 | `T.Kernel` 三维 | — | **重新设计**（NPU 单 kernel ≤2D block） |
| O9 | `threads` 参数 | CUDA 线程块配置 | — | CUDA 线程模型 | **舍弃**（NPU 无对应） |
| O10 | P 量化到输入 dtype（`acc_s_cast`） | 减半 PV-GEMM 的 K 维字节 | `T.copy(acc_s, acc_s_cast)` | — | 可移植（保留，语义点） |
| O11 | paged 块寻址（block_table 间接） | 不连续物理页 KV cache | `block_table` 间接取页 | 无 | 可移植（保留；装载形态需适配） |
| O12 | `-inf` 掩码 + `logsum==0` 守卫 + 段内 `max==-inf → rs=0` 守卫 | 处理 OOB/空段 | `if_then_else` | — | 可移植（保留，需**稳定化**：哨兵与守卫同域） |
| O13 | 融合 epilogue（段内归一化写回 `Output_partial`） | 免中间落 GM（段内） | `acc_o/logsum` 直接写 `Output_partial` | — | 可移植（保留） |
| O14 | `combine` 的 swizzled layout（`T.annotate_layout` + `make_swizzled_layout`）+ `forward_thread_fn` | 消 SMEM bank conflict（GPU） | 手动 swizzle + thread 映射 | SMEM bank 结构 / warp 映射 | **舍弃**（NPU 无 bank conflict，thread 映射无意义） |
| O15 | `disable_tma=True`（A5/A15/B4/B5） | 规避 TMA barrier 在 padded `seq_q` 上的问题 | 退回 SIMT copy | CUDA TMA | **舍弃**（NPU 无 TMA，`T.copy` 即 SIMT/MTE） |

**算法层（可移植）**：O1（两级分解的**数学结构**）、O2、O3、O10、O11、O12、O13。
**实现层（需适配/舍弃）**：O4–O9、O14、O15。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 §1.6.0 算法调研结论）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| **语义层：数学映射 / I/O 契约 / dtype 语义 / 边界语义** | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| O1 split-KV 两级分解 + LSE 合并（**数学结构**） | 算法 | 无（分段重组是恒等） | 有 | **保留** | 段划分 + `l_i` 加权合并的**数学结构**保留（精确恒等）；见 §1.6.1 OPT-S |
| O1 的**跨核并行 + 二次 launch 屏障**（实现形态） | 优化 | 多 CTA + 全局屏障 | **部分**（npuir 一个 prim_func 一个 kernel；无核内全局屏障） | **重新设计** | §0.6 **R5**：拆为两个 `@tilelang.jit` kernel，由 host 侧 Python closure 顺序 launch（`glse`/`Output_partial` 即接口工作区）——先例 `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`（preprocess→bwd→postprocess 顺序 launch） |
| O2 online softmax | 算法 | 无 | — | **保留** | 纯算法层；§0.6 保持段内 `scores_scale` 精确在线语义 |
| O3 exp2/log2e scale | 算法 | 无 | 有（`T.vexp2`/`T.vexp`） | **等价替换** | `exp2(x·log2e·s0) ≡ exp(x·s0)`；见 §1.6.1 OPT-3 |
| O4 SMEM tiling | 优化 | shared memory | 有（L1 512KB / UB 192KB，32B 对齐） | **等价替换** | Cube 操作数→L1；Vector 数据→UB（`alloc_shared`/`alloc_fragment`）；`CONST-capacity-910B2C` |
| O5 Tensor Core MMA | 优化 | tensor core | 有（Cube `T.gemm` + L0A/L0B/L0C） | **等价替换** | `T.gemm`（Developer，`docs/Tilelang.language/线性代数操作/T.gemm.md`）；分形 M/N≥16、K≥32，L0C 128KB |
| O6 fragment 累加器 | 优化 | 大寄存器堆 | 有（L0C fp32 / UB） | **等价替换** | `T.alloc_fragment`（Developer） |
| O7 异步流水 | 优化 | 异步拷贝引擎 | 有（MTE + `T.Pipelined`） | **等价替换** | `T.Pipelined(num_stages=2)`；stage 数按 L1/UB 容量重算（§4.5） |
| O8 三维 grid | 优化 | block×thread 组织 | **无直接等价**（NPU 单 kernel ≤2D，且本设计拆两 kernel） | **重新设计** | 一维 `T.Kernel(num_logical, is_npu=True)` + cid 反解（§0.6 **R1**） |
| O9 `threads` | 优化 | CUDA 线程模型 | 无 | **舍弃** | 脚手架 K9 已从 NPU 工厂签名移除；无 NPU 语义（§0.6 R8） |
| O10 P 量化 | 算法 | 无 | — | **保留** | `T.vcast(round_mode="rint")` 复现同一量化点（§1.6.2） |
| O11 paged 寻址 | 算法 | 无 | 有（slice-form `T.copy` + 标量索引） | **等价替换** | 跨步/间接 GM 装载**必须** slice 形态（`TRAP-load-nd2nz-strided`）；block_table 间接 → §0.6 **R3** |
| O12 `-inf` 掩码 + 守卫 | 算法 | 无 | 有 | **等价替换（+稳定化）** | 有限大数哨兵 `-1e38`（`PL-1.11`）+ **同域守卫**；源段内 `max==-inf → rs=0` 守卫**不可照搬**（静态界下会清零已累积状态）→ §0.6 **R4** |
| O13 段内融合 epilogue | 算法 | 无 | — | **保留** | Vector epilogue（`T.vdiv`/`T.vcast`） |
| O14 swizzle + thread_fn | 实现 | SMEM bank / warp | 无 | **舍弃** | NPU 无 bank conflict、无 warp 映射（migration-analysis §5.3） |
| O15 `disable_tma` | 实现 | CUDA TMA | 无 | **舍弃** | NPU 无 TMA；`T.copy` 默认 MTE/SIMT |

> 每项处置依据已注明。**无「不可判定」项**；O1 拆为「数学结构（保留）」与「跨核实现（重新设计）」两项判定。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目，逐项给出 NPU 新算法；其余「等价替换」项在实现中承接源意图。

**重设计项 R1：三维 grid → 一维非持久 grid（两个 kernel 各一维）**

- **源方案**：Phase A `T.Kernel(ceildiv(S_q,bm), heads*batch, num_split)`；Phase B `T.Kernel(ceildiv(S_q,bm), heads, batch)`（均三维，GPU 允许）。
- **NPU 新算法**：两个独立 kernel，各自一维：Phase A `num_logical_A = num_q · B · H · num_split_eff`；
  Phase B `num_logical_B = num_q · B · H`。由 `cid` 反解：
  - A：`sid = cid % ns; r = cid // ns; hid = r % H; r2 = r // H; bz = r2 % B; mid = r2 // B`（`ns=num_split_eff`，`num_q=ceildiv(S_q,block_M)`）。
  - B：`mid = cid % num_q; r = cid // num_q; hid = r % H; bz = r // H`。
  **非持久**（每核一任务，无核内 `T.serial` 多任务循环）——依据 `TRAP-DEVMODE-PERSIST-GEMM`（Developer + persistent + gemm 运行时崩溃）。
- **语义保持论证**：任务→(mid,hid,bz,sid) 映射是一一满射；A 的每段计算与源逐字相同且段间无共享写；B 的合并式与源 `combine` 逐字对应。**边界**：`num_logical_A()`/`num_logical_B()` 与源三维乘积相等。

**重设计项 R2：decode M 维退化（S_q=1）下的 GEMM 分形适配（承接 no-split R2）**

- **源方案**：`block_M`（默认 128）覆盖 query 行；两个 `T.gemm`。
- **NPU 新算法**：NPU Cube 分形硬约束 **M≥16 / N≥16 / K≥32**（`ascend-constraints.md §1`；`docs/.../T.gemm.md §2.2.2` 未标分形数值，属**未文档化假设**，依据 `examples/gemm/example_gemm.py` 与 prefill `_gqa_prefill_fwd_kernel` 的 `tmc=max(16,ceil16(real))` 实践）。取 **`block_M = 16`**；decode `S_q=1` 时仅第 0 行真实，行 1..15 为 padding。两个 GEMM 的 M 维恒 16（≥16 ✓）；`N`/`K` 维满足分形。padding 行经「逐行 softmax + 只写回真实行」完全隔离。
- **语义保持论证**：`block_M=16 ⊇ S_q=1`；只有真实行参与写回；真实行的 `S`/online-softmax/`O` 与源逐项一致。**边界**：`S_q` 不被 `block_M` 整除时按 `real_m = min(block_M, S_q − mid·block_M)` 裁剪写回（`TRAP-T-copy-region-semantics` 规则②）。

**重设计项 R3：paged KV 寻址 + 跨步/间接装载形态（承接 no-split R3）**

- **源方案**：`page_idx = k_global // nbpp`；`boff = block_table[bid, page_idx]*nbpp + k_global % nbpp`；`T.copy(K[boff*bn:(boff+1)*bn, hid, :], K_shared)`。
- **NPU 新算法**：令 **`block_N = page_size`** ⟹ `nbpp=1`，一个 KV tile 恰为**一个物理页**：
  `phys = block_table[bz, pg]`（`pg` = 该段内全局页号）。装载用 **slice 形态**
  `T.copy(K[phys*page:(phys+1)*page, hid, 0:dim], K_shared[0:page, 0:dim])`——**必须** slice 形态
  （`TRAP-load-nd2nz-strided`）。`block_N = page_size` 时源的 `page_idx/`%nbpp` 算术退化为恒等
  （`page_idx = pg`，`k_global % 1 = 0`），地址与源**逐位一致**。
- **语义保持论证**：`nbpp=1` 时逻辑页↔物理页映射与原公式等价；slice `T.copy` 的地址算术与源 `boff*bn` 相同（仅装载 API 形态不同，值相同）。**实测复核（承接 no-split）**：核内 GM 标量索引 `block_table[bz,pg]`（`pg` 为流水循环变量的表达式）在 Developer 模式 `num_stages=1/2` 均**编译+运行正确**（`repro/TRAP-paged-dynamic-index-sliceload.py`）。

**重设计项 R4：段划分重设计 + 静态段内循环界 + 同域恒等稳定化（split 特有）**

- **源方案**：段边界来自 host 的 `acc_split_length`——段长 `chunk_size = floor(real_max/(num_split·block_N))·block_N`（**block_N 倍数，不保证 page 对齐**），段内循环界 `blocks_in_split` 为**运行时标量**（动态 trip count）。源靠动态界**从不迭代全掩码块**。
- **NPU 新算法**（**段划分准则重设计**）：
  1. **编译期页划分**：令 `nkv = ceildiv(N_kv, page)`（编译期常量），`num_split_eff = min(num_split, nkv)`，
     `P = ceildiv(nkv, num_split_eff)`（**编译期常量**）。段 `sid` 覆盖页 `[sid·P, (sid+1)·P)`。
     **段内循环界 = P（静态）**，规避动态 trip-count 限制并保留 `T.Pipelined`。
     > **等价性关键**：分段 softmax + LSE 合并的**结果与段划分方式无关**（任意把 KV 划分成连续段，
     > `Σ_i exp(l_i−LSE)·o_i = Σ_j exp(s_j−LSE)·V_j` 恒成立）——故用**页对齐的编译期划分**替换源的
     > host token 级划分**不改变输出**，只改变内部工作区内容（`glse`/`Output_partial`），而 `glse`/`Output_partial`
     > 是**内部工作区**（契约只约束 `Output`；§8.1 测试只比 `Output`）。`acc_split_length` 形参因此**不被使用**
     > （保留形参以维持接口契约）。**MANIFEST 复核**：`nkv=4/num_split=4 → P=1`；`nkv=2/num_split=4 →
     > num_split_eff=2, P=1`（无空段）。
  2. **页号钳位（避免越界）**：段 `sid` 的页号 `pg = sid·P + k`，可能 ≥ `nkv`（末段）。**装载用钳位页号**
     `pgc = min(pg, nkv−1)`；正确性由列掩码保证：`pg ≥ nkv ⟹ global_pos = pg·page + j ≥ nkv·page = N_kv ≥ real_kv` ⟹ 该块**整块被掩码**。故无需运行时分支。
  3. **列掩码（全局位置）**：`global_pos = pg·page + j < real_kv` 为有效，否则**有限大数哨兵 `-1e38`**
     （`PL-1.11`；`exp(−1e38 − m)=0` 精确零，规避 `−inf−(−inf)=NaN`）经 `T.vselect` 值替换。
  4. **全掩码块同域恒等稳定化**：`all_masked = (m_cur ≤ −1e30)`（与哨兵 `−1e38` **同域**）→
     `m_cur_safe = all_masked ? 0 : m_cur`；**`rs = all_masked ? 1 : exp(m_prev − m_cur_safe)`（恒等）**；
     `P = exp(S̃ − m_cur_safe)` ⟹ 全掩码块 `P=0`、`s_sum=0` ⟹ 零贡献且**不清零已累积状态**。
     > **为何不用源的 `rs=0`**：源 `Softmax` 宏的全掩码守卫取 `scores_scale=0`（L261–263）。在**动态界**下
     > 该分支不触发；**静态界**下会触发，而超界块恒为**段尾块**（有效列是前缀），`rs=0` 会把前面已累积的
     > `logsum/acc_o` 清零 → 输出错误归零。**恒等（`rs=1`）是唯一既零贡献又保状态的取值**（no-split 结论沿用，
     > `repro/…` 与设备端 L1-1/L1-2 已证）。
  5. **段级 LSE 的参考 max（`m_ref`）**：段内 online 累加的不变量是 `logsum = Σ_j exp(s̃_j − m_ref)`，
     `m_ref` = **最后一个非全掩码块的 max**（全掩码块 `rs=1` 不改参考）。段 LSE `l_i = ln(logsum) + m_ref`。
     全掩码段（`logsum=0`）→ `l_i = LSE_SENT`。
  6. **段级 LSE 哨兵 `LSE_SENT = −30000`（fp16 可表示）**：源用 `-inf`；因 **`glse` 以输入 dtype 存储**
     （fp16 min ≈ −65504），`-1e38` 会 cast 为 `-inf`，导致 combine 中全空段 `exp(−inf−(−inf))=NaN`。
     取 `−30000`（fp16 可表示）后 combine 全程走光滑公式无 NaN（§6.4 已机器验证）。
- **语义保持论证**：段划分重设计不改变输出（恒等式，§1.6.1 OPT-S，D-1 机器验证 `max_abs 3.66e-04`、违反率 0）；
  钳位页号 + 全局位置掩码与源 `k_global*block_N+j < real_seqlen_kv` 逐位等价；同域恒等稳定化对全掩码段尾块零贡献。
  **边界**：`real_kv=0` → 全段全掩码 → `logsum=0`、`l_i=LSE_SENT`、`o_i=0` → combine 输出 **0**（与源零迭代一致）；
  `N_kv < num_split·page`（`shorter-cache`）→ `num_split_eff=min(num_split,nkv)` 无空段。

**重设计项 R5：单 prim_func 内两次 `T.Kernel`（二次 launch）→ 两个 `@tilelang.jit` kernel + host closure 编排**

- **源方案**：同一 `@T.prim_func` 内顺序 `_mha_decode_split(...)` 与 `combine(...)`（两个 `@T.macro`，各含一个 `T.Kernel`）——GPU 编译为两次 launch，其间隐式全局屏障。
- **NPU 新算法**：npuir **一个 prim_func 编译为一个 device kernel**（`T.Kernel` 一维/≤2D；无 prim_func 级多 launch 支持；
  `tilelang/jit/jit_npu.py` 的 `_launch` 单次 `rtKernelLaunchWithFlagV2`）。故拆成**两个独立 `@tilelang.jit` kernel**：
  `_split_partial`（Phase A，写 `glse`/`Output_partial`）与 `_combine`（Phase B，读工作区、`out_idx=[-1]` 产 `Output`），
  由工厂返回的 **Python closure `wrapped(...)`** 顺序 launch：
  ```python
  def wrapped(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial, acc_split_length):
      k_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial)   # Phase A
      return k_combine(glse, Output_partial)                                # Phase B -> Output
  ```
  **先例**：`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`（`preprocess→bwd→postprocess` 三个 kernel
  顺序 launch）；`_gqa_prefill_fwd_kernel.py`（内层工厂返回 Python closure `wrapped(q,k,v)`）。
  `glse`/`Output_partial` 由脚手架 host **预分配并传入**——正是接口为此设计的工作区，无需额外分配。
- **语义保持论证**：两次 launch 的顺序（A 完成 → B 读）与源两次 launch 的隐式屏障**语义相同**（同为 stream 序）；
  计算内容逐字不变。**边界**：`glse`/`Output_partial` 由 host `torch.zeros` 初始化，本设计**写满**所用 `sid∈[0,num_split_eff)`
  分片（未用分片 `[num_split_eff, num_split)` 由 combine 循环界排除，不读）。
  > **风险标注**：脚手架 `@torch.library.custom_op(..., mutates_args=())` 把 `glse`/`Output_partial` 列为输入却**被写入**——
  > 该不一致**继承自 GPU 源算子**（源亦如此），非本设计引入；列入 §9 风险 R-9。

**重设计项 R6：段内 online softmax 由 `T.macro`/`T.Parallel` → v-prefix 向量链（承接 no-split R5）**

- **源方案**：`Softmax`/`rescale` 为 GPU 侧 `@T.macro`，内部 `T.Parallel` 逐元素 + `T.reduce_max/sum`。
- **NPU 新算法**：Developer 模式以 v-prefix 向量 API 表达同一数学：`T.vsub/vmul/vadd/vexp/vcast/vdiv` +
  `T.reduce_max/sum(dim=1)` + `T.vbrc` 广播。参照 `examples/flash_attention/flash_attn_npuir_dev.py`。
- **语义保持论证**：逐项对应，无数学改变；状态顺序与初始化绑定（`m_prev` init `-inf`；循环内**先 `reduce_max→m_cur`**、
  再算 `rs`、最后**末尾 `T.copy(m_cur,m_prev)`**）；`O10` 量化点（`T.vcast(rint)`）保持。

**重设计项 R7：combine 重设计（Vector-only kernel，去 swizzle / thread_fn / dim=0 reduce）**

- **源方案**：`combine` 用 `T.annotate_layout`（swizzled）+ `T.reduce_max(lse_local, lse_max_local, dim=0)` +
  `T.Pipelined(num_split)` 循环 + `T.Parallel(bm,dim)` 逐元素累加。
- **NPU 新算法**：Vector-only 单 kernel（无 Cube，无 swizzle、无 thread_fn）。`num_split_eff` 小（≤8，编译期常量）
  ⟹ **段维循环展开为 `T.serial(num_split_eff)`**，段间 max/logsum/加权累加全部用**行向量 v-op**（`T.vmax`/`T.vsub`/`T.vexp`/
  `T.vadd`/`T.vln`）作用在 `[block_M,1]` / `[block_M,D]` 上——**规避 `T.reduce(dim=0)`**（跨步归约为未实测路径）；
  `LSE = ln(Σ_i exp(l_i − m)) + m`（自然对数域，与 §1.6.1 OPT-3 一致）；`o = Σ_i o_i·exp(l_i − LSE)`。
- **语义保持论证**：与源 `combine` 逐项同构（`lse_max`→`m`、`lse_logsum`→`Σ exp`、`scale=exp2(l_i−LSE)`→`exp(l_i−LSE)`），
  仅换底（自然对数）与归约实现；NaN 免疫由 `LSE_SENT` 保证。`Output_partial`/`glse` 经输入 dtype 往返（与源相同）。

**重设计项 R8：`threads` 参数移除（承接脚手架 K9）**

- **源方案**：`_func(block_M, block_N, num_split, num_stages, threads)`。
- **NPU 新算法**：工厂签名 `_func(block_M, block_N, num_split, num_stages)`（脚手架 L114–117 已按此调用）；
  `T.Kernel(..., is_npu=True)` 不带 `threads`。
- **语义保持论证**：`threads` 无 NPU 数值语义（O9 舍弃），移除不改变计算。

**源算法优化意图承接（§0.4 → §0.6/§1.6.0）**：O1–O13 的意图（长 KV 并行度、省中间缓冲、省算、减访存、Cube 算力、
流水重叠、paged 寻址、段级融合）全部由 R1–R8 与 §1.6 承接；**舍弃** = O9（threads）、O14（swizzle/thread_fn）、O15（disable_tma）
——三者均无 NPU 语义。无静默丢弃。

### 0.7 标杆实现

- **源算子路径**：`/home/developer/TileOPs/tileops/kernels/attention/mha_decode_paged.py`（函数 `_mha_decode_split_kernel`，L117 起）。
- **源仓参考实现 / 测试基准**：`examples/TileOPs/tests/ops/test_mha_decode_paged.py::MhaDecodePagedTest.ref_program`
  （paged 重组逻辑 KV + `F.scaled_dot_product_attention`）——§8.1 golden 直接移植（**独立于** §0.6 的 NPU 算法）。
- **抽象副本（含 helper）**：`examples/TileOPs/tileops/kernels/attention/mha_decode_paged/_multi_head_attention_decode_paged_with_k_v_cache_fwd_kernels.py`。
- **NPU 集成目标（接口契约）**：`examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged.py`
  （`MHADecodePagedKernel`，`default_config = {block_M:128, block_N:64/32, num_split:4, num_stages:2}`）。
- **同任务前序函数（强烈参照）**：`examples/mha_decode_paged/_mha_decode_no_split_kernel/`
  （DESIGN.md + `_mha_decode_no_split_kernel.py`，已通过全部门禁）——本设计复用其 MixCV 结构、slice 装载、
  静态界掩码、同域恒等稳定化与 v-prefix 链。
## 1. 概述

### 1.1 算子名称

`_mha_decode_split_kernel`（paged KV-cache MHA **decode**，split-KV：分段 online softmax + 段级 LSE 合并）。

### 1.2 功能描述

对每个 `(batch, head, query-token)`，从 **paged（block-table 寻址）** KV cache 取出逻辑 KV 序列，按
**连续段**切分；段内做 online softmax 得归一化部分输出与段级 LSE，再按 LSE 加权合并成最终输出：
`softmax(q·Kᵀ/√D)·V`；decode 场景 `S_q=1`。**输出与 no-split 单段计算一致**（分段重组是精确恒等）。

### 1.3 数学公式

$$
o[b,:,h,:]=\sum_{i=0}^{n_s-1}w_i\,o_i,\qquad
o_i=\frac{\sum_{j\in\mathcal S_i}\exp(s_j)V_{b,h,j}}{\sum_{j\in\mathcal S_i}\exp(s_j)},\qquad
w_i=\frac{\exp(l_i)}{\sum_{i'}\exp(l_{i'})},\qquad
l_i=\ln\!\!\sum_{j\in\mathcal S_i}\exp(s_j)
$$

其中 $s_j=q[b,:,h,:]\cdot K_{b,h,j}/\sqrt{D}$，$\{\mathcal S_i\}$ 是逻辑 KV 轴的**连续划分**
（`num_split_eff` 段），$K_{b,h,j}=K[\,\texttt{block\_table}[b,j/\!\!\downarrow\!p]\cdot p+(j\bmod p),\,h,\,:]$
（$p$=`page_size`）。数值实现按 §1.6.1 的**优化后公式**（自然 exp 域 + 段内 scale 折入 Q + 编译期页划分 + 段级 LSE 加权合并）。

### 1.4 算法描述

> 本节描述**迁移决策（§0.5/§0.6）与算法调研（§1.6.0）之后**的 NPU 侧算法，**不是**源算法直译。

**两个 Developer MixCV kernel + host closure 编排**（§0.6 R5）：

**Phase A —— `_split_partial`**（一维非持久 grid，`num_logical_A = num_q·B·H·num_split_eff`）：
1. 由 `cid` 反解 `(mid, hid, bz, sid)`（§0.6 R1）；`sid` 段覆盖页 `[sid·P, (sid+1)·P)`，`P=ceildiv(nkv, num_split_eff)` 编译期常量。
2. 载入 `Q[bz, mid·bm:…, hid, :] → Q_shared`（`block_M=16`，仅第 0 行真实，其余清零；§0.6 R2）；`Q_shared *= (1/√D)`（OPT-1）。
3. **段内 KV 循环**（`k in T.Pipelined(P, num_stages=2)`，**静态界**，§0.6 R4）：
   - **Cube**：`pg = sid·P + k`，`pgc = min(pg, nkv−1)`（钳位）→ `phys = block_table[bz, pgc]` → slice-form `T.copy` 载入 `K_shared`；`T.gemm(Q_shared, K_shared, acc_s, b_transpose=True, initC=True)`。
   - **Vector**：按**全局位置** `pg·page + j < real_kv` 掩码（有限哨兵 `-1e38`，值替换 `vselect`）；**段内在线 softmax**（OPT-2/OPT-3，**先 `reduce_max→m_cur`**，`rs=exp(m_prev−m_cur_safe)`，`P=exp(S̃−m_cur_safe)`，`acc_o*=rs`，`logsum=logsum·rs+s_sum`，**末尾 `T.copy(m_cur,m_prev)`**，跟踪 `m_ref`）；`acc_s` `vcast(rint)→acc_s_cast`（O10）。**全掩码块同域恒等**（`rs=1,P=0,s_sum=0`，§0.6 R4）。
   - **Cube**：slice-form 载入 `V_shared`；`T.gemm(acc_s_cast, V_shared, acc_o, initC=False)`。
4. **段内 epilogue（Vector）**：`o_sid = acc_o/logsum`（`logsum==0 → 0`），`vcast(rint)` → 写 `Output_partial[bz, rows, hid, sid, :]`（行裁剪 `real_m`）；`l_i = ln(logsum) + m_ref`（空段 → `LSE_SENT=-30000`）→ 写 `glse[bz, hid, sid, rows]`。

**Phase B —— `_combine`**（一维非持久 grid，`num_logical_B = num_q·B·H`，**纯 Vector**）：
1. 反解 `(mid, hid, bz)`；逐段读 `glse[bz,hid,i,rows]`（`i ∈ [0,num_split_eff)`）求 `m = max_i l_i`（行向量 `vmax`）。
2. `LSE = ln(Σ_i exp(l_i − m)) + m`（`vsub/vexp/vadd/vln`；`vlog` 域无 NaN，由 `LSE_SENT` 保证）。
3. 逐段读 `Output_partial[bz,rows,hid,i,:]`，`acc += o_i · exp(l_i − LSE)`（`vcast/vmul/vadd`）。
4. `vcast(rint)` → `T.copy` 写 `Output[bz, rows, hid, :]`（行裁剪）。

**与源算法的结构差异**（来源标注）：
- 三维 grid → 一维非持久 grid（§0.6 **R1**）。
- 两个 `T.Kernel`（同 prim_func 二次 launch）→ **两个 `@tilelang.jit` kernel + host closure 编排**（§0.6 **R5**，npuir 一 prim_func 一 kernel）。
- **段划分准则重设计**：host token 级（block_N 倍数、非 page 对齐、动态 trip）→ **编译期页划分** `P=ceildiv(nkv,num_split_eff)` + 静态界 + 页号钳位（§0.6 **R4**，OPT-S）。
- `block_M=16`（分形下限）、`block_N=page_size`、slice-form 装载（§0.6 **R2/R3**）。
- 段内 `max==-inf → rs=0` 守卫 → **同域恒等 `rs=1`**（§0.6 **R4**）。
- `combine` 去 swizzle/thread_fn、去 `dim=0` reduce（改行向量 v-op 展开）→ §0.6 **R7**。
- 移除 `threads`（§0.6 **R8**）、`disable_tma`（O15 舍弃）、swizzle（O14 舍弃）。
- 数学等价优化：段内 scale 折入 Q、自然 exp 域、重缩放因子重结合、段划分页对齐（§1.6.1 OPT-1/2/3/OPT-S）。

### 1.5 数据流图

```
Phase A（每 (mid,hid,bz,sid) 任务）:
GM[Q](1,D) --slice T.copy--> L1[Q_shared(16,D)] --*s_nat--> L1[Q_shared]
GM[block_table] --scalar--> phys (clamped pg)
GM[K|V] --slice T.copy--> L1[K_shared(page,D) | V_shared(page,D)]
L1[Q_shared] + L1[K_shared] --T.gemm(b_transpose)--> UB[acc_s(16,page) f32]
UB[acc_s] --mask / online-softmax(v-prefix)--> UB[acc_s]/UB[acc_s_cast f16]/row-state
L1[V_shared] + UB[acc_s_cast] --T.gemm(initC=False)--> UB[acc_o(16,D) f32]
UB[acc_o] --/logsum,vcast--> GM[Output_partial(bz,rows,hid,sid,:)]
row-state --ln+ m_ref--> GM[glse(bz,hid,sid,rows)]

Phase B（每 (mid,hid,bz) 任务）:
GM[glse] --copy--> UB[lse(16,1)] --vmax/vexp/vln--> LSE(16,1)
GM[Output_partial] --copy--> UB[po(16,D) f16] --vcast/vmul(sc)/vadd--> UB[acc_o(16,D) f32]
UB[acc_o] --vcast(rint)--> GM[Output(bz,rows,hid,:)]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先做算法调研四问（§1.6.0，输入公式/源算法只是基线候选之一）→ 公式级等价优化（§1.6.1，含 D-1 机器验证）→ 循环/标量向量化替代（§1.6.2）→ 向量化轴与布局决策（§1.6.3）。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = 同一数学语义（paged MHA decode，`S_q=1`）的**算法族**（含「是否分段 / 如何合并」的决策）。
> 方法论与深度分级见 `.agents/skills/tilelang-op-design/references/algorithm-research.md`；本算子为
> 「矩阵/多步/融合」类 → **完整调研**（含 2 个矩阵乘 + online 归约 + 段级两阶段合并）。
> 参考表命中行：`algorithm-candidates.md` 的 `ALG-attention`（flash 结构 R2、MixCV R4）、`ALG-softmax`
> （减 max、exp2、online）、`ALG-gemv`（split-K 分块累加）、`ALG-reduction`（分块归约 + 合并）。

**R1 等价化简公式候选**（基线候选必在表内；正式论证在 §1.6.1）：

| # | 候选（算法族） | 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|---------------|------|-----------|---------|--------|
| 0 | **基线：源 split 链**（分段 online softmax + 段级 LSE 合并；`glse`/`Output_partial` 落 GM；两次 launch） | 段内 flash + 段间两阶段归约 | — | — | ✅（基线） |
| 1 | 单遍（不分段；= 候选 0 的 `num_split_eff=1` 退化，同任务 no-split 算法） | 单段 online softmax | 精确恒等（分段重组恒等式） | 去段间合并/工作区；丧失 KV 并行度 | ✅ |
| 2 | 核内分段累加 + 核内合并（单 kernel；`num_split` 组累加器在寄存器/UB 合并） | 段内 flash × nsplit + 寄存器合并 | 精确恒等 | 去 `glse`/`Output_partial` GM 往返与二次 launch；仍无跨核 KV 并行 | ✅（备选，§1.6.3） |
| 3 | 段内 materialize S 矩阵到 GM（两遍：先存 S 再 softmax） | O(段内 S_kv) GM 缓冲 | 容差内等价 | **反向**：多一趟 GM 往返 + 去 online 缓冲收益 | ✅（见 R3） |
| 4 | scale 折入 Q / 自然 exp 域 / 重缩放重结合 | 段内公式变形 | 数学恒等 | 省段内 op（OPT-1/2/3） | ✅（§1.6.1） |
| 5 | 归一化乘倒数（`o=acc_o·(1/logsum)`） | 段内公式变形 | 数学恒等 | **不采纳**：向量域 `[16,D]/[16,1]` 已是 1 个 `vdiv` op；改 `vrec+vmul` = 2 op，更贵（`CONST-vector-op-cost-910_9362` 每 op 固定） | ❌ |
| 6 | P 不量化（P 以 fp32 送 gemm2） | 去 P 量化点 | **非等价**（改数值路径） | 反增 gemm2 K 维字节 | ❌（违反 O10） |
| 7 | 合并式改写为「单次可结合归约」（如原子加） | 段间 combine 变原子累加 | **非等价**（LSE 需 max→sum 两阶段，非单原子可结合） | — | ❌（需全局 max 才能稳定，见下） |

> **候选 7 的负向论证（结构依据）**：合并权重 `w_i = exp(l_i)/Σexp(l_{i'})` 依赖**全局** `max_i l_i`
> （数值稳定所需），且 `o = Σ_i o_i·w_i` 的 `w_i` 在算出 LSE 前不可得 ⟹ 存在「先全局 max+sum、后加权」的
> **两阶段依赖**，单次原子加/单遍可结合归约无法表达（`ALG-reduction` R2 的分块归约 + 合并即此两阶段）。
> 故不为等价化简候选。

**R2 在线算法**：**有，且为基线所采用**（结构依据 + 源证据）。段内 online softmax（分块 running max/sum + 重缩放）
是 flash 结构（`ALG-attention` R2、`ALG-softmax` R2）；源 `Softmax` 宏已用该手段——**源算法已用 online 是「存在在线算法」的直接证据**（algorithm-research.md §3 R2③）。段**间**合并是分块归约 + 合并（`ALG-reduction` R2），
两阶段（max→sum）不可在线化（见候选 7）。收益口径：段内扫描 1 遍、中间缓冲 O(tile)（`acc_s[16,page]`）；
段间无数据扫描（只读 `glse`/`Output_partial` 小工作区）。候选 3（materialize）在 R3 被淘汰。

**R3 复杂度对比**（四口径；口径：FLOPs 只计乘加=2/次；访存按 GM 字节；遍数 = 同一 KV 数据的 GM 读取次数；
中间缓冲峰值 = 片上最大常驻 + GM 工作区往返量。**dispatch 代表 workload = smoke `(B=1,H=16,S_q=1,N_kv=512,D=128,page=128,nsplit=4)`**）：

| 算法候选 | FLOPs（含 M 分形 pad 16×） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核归并代价 |
|---------|--------------------------|---------------|---------|-------------|------------------------|
| **0 基线（源 split 链，两 kernel）** | `4·B·H·S_q·S_kv·D` = 4.19 MF（**×16 M-pad ≈ 67 MF**） | `(q+2kv+o)·2 + glse + Output_partial + meta` ≈ **4.22 MB** | KV 1（各段读一次） | 片上 `acc_s[16,page]f32`+`acc_o[16,D]f32`；GM 工作区 `glse`128B + `Output_partial`16KB（写+读） | A: `B·H·nsplit=64` / B: `B·H=16`；跨核归并 = 二次 launch（stream 序屏障，≈0 同步指令） |
| 1 单遍（`num_split_eff=1`，無段间合并） | 同基线 | ≈ **4.20 MB**（无工作区） | KV 1 | 片上同基线；**无 GM 工作区** | `B·H=16`；**无跨核归并** |
| 2 核内分段 + 核内合并（单 kernel） | 同基线 | ≈ **4.20 MB**（无 GM 工作区） | KV 1 | 片上 + `nsplit` 组段累加器（`nsplit×[16,D]f32` ≈ 32KB） | `B·H=16`；**无跨核归并** |
| 3 materialize 段内 S 到 GM | 同基线 | `+` S 写/读 `4·B·H·S_q·S_kv·2` ≈ **+32 KB** | KV 1 + S 写/读 **2** | O(段内 S_kv) GM 缓冲 | 同基线但访存↑ |
| 7 原子单遍合并 | — | — | — | — | ❌ 结构不可行（R1 候选 7 论证） |

> 该算子**带宽受限**（算术强度 ≈ 2 FLOP/byte ≪ 机器平衡点），故**访存/遍数是主导项**：候选 0/1/2 访存量同阶
> （工作区仅 16KB 级，占比 <0.4%），候选 3 多一趟。差异主要在**发射项（op 数 × 单 op 成本）**与**并行度/归并代价**。

**R4 硬件亲和性评估**（逐候选对照检查清单，定性 + 定量 roofline；负向淘汰附依据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水/融合 | 结论 |
|---------|-------------|---------|------------|---------|----------|------|
| **0 基线（分段 Cube 双 GEMM + Vector 段内 online + 段间合并）** | ✅ Cube MAC + Vector v-prefix；M 维 pad 16 但 FLOPs 非瓶颈 | ✅ L1（K/V tile ≤2×64KB）；UB（acc_s ≤16KB） | ✅ `page∈{128,256}` 16 倍数；D∈{64,128} 16 倍数 | ✅ 段内界静态（§0.6 R4）；新引入的全掩码尾块由同域恒等守卫处理 | ✅ `T.Pipelined`；段内 Cube/Vector 融合（MixCV）；段间二次 launch | **✅ 选定** |
| 1 单遍 | ✅ 同基线 | ✅ | ✅ | ✅ | ✅ | ✅ **= 候选 0 的 `num_split_eff=1` 退化**（合入候选 0；段数由 §1.6.3 归一化） |
| 2 核内分段 + 核内合并 | ✅ 同基线（合并段改行向量 v-op） | ✅（+32KB 段累加器） | ✅ | ✅ | ✅ 单 launch | ✅ **备选**（§1.6.3 实验裁决） |
| 3 materialize | ✅ | ⚠️ O(S_kv) GM 缓冲 | ✅ | ✅ | ⚠️ 多一趟 | ❌ 淘汰（访存↑，遍数↑） |

**R4 定量 roofline（D-2）**：dispatch 代表 workload = smoke `(B=1,H=16,S_q=1,N_kv=512,D=128,page=128,nsplit=4)`（`P=1`；其余 manifest workload 同阶或更小）。
- **流量项** = GM 总字节 ÷ 有效带宽。候选 0 GM 流量 = `(q+2·kv+o)·2 + glse + Output_partial + meta` = `(4096 + 2·2097152 + 4096)·2 + 128 + 16384 + 20` ≈ **4.22 MB**。带宽按 Ascend910_9362 单 die **核→GM ≈1.06 TB/s**（`CONST-ascend910-9362-profile`，device=910_9362，2026-10-13）→ **3.98 µs**。
- **发射项** = 向量 op 数 × 单 op 成本。候选 0：Phase A 每任务（`P=1`）≈ 5（前导）+ 10（掩码/online/cast）+ 4（epilogue: vdiv/vcast/vln/vadd）= **19 op**；`64` 任务 ÷ 20 核 ≈ 4 任务/核 ⇒ **≈76 op/核**。Phase B 每任务 ≈ 4（段内 max/sum）+ 4（加权累加）+ 4（epilogue）≈ **22 op**，`16` 任务 ÷ 20 核 ⇒ **≈18 op/核**。合计 ≈ **94 op/核** × `CONST-vector-launch-overhead`（**910B2C 实测 ~0.5µs/op，9362 未重验——待重验条目**）→ **≈47 µs**。
  > 对照（同口径）：候选 1（单遍）= 16 任务 ×（5+4×10+4）/20 = **≈20 µs**；候选 2 = 16 任务 ×（5+4×12+12+4）/20 ≈ **≈28 µs**。
- **容量项** = 容量决定的 tile 循环数 × 跨引擎往返代价。本设计**无 `store_fixpipe` GM 往返**（`acc_s/acc_o` 驻 UB fragment，Developer 自动 CV split）；GM 工作区 `glse`+`Output_partial` 16.4KB 由设计**主动**读写（非容量强制）；L0C（`16×256×4=16KB`）与 UB（≈90KB）在限内（`CONST-capacity-910B2C`）。容量项 ≈ 0。
- **估算下界**：`估算下界: max(3.98, 47, ~0) ≈ 47 µs（流量 3.98µs + 发射 47µs + 容量 ~0；引用 CONST-ascend910-9362-profile / CONST-vector-launch-overhead / CONST-vector-op-cost-910_9362 / CONST-capacity-910B2C）`。
  **不确定性（重要）**：发射项依赖的 0.5µs/op 为 910B2C 常数、9362 未重验，且 `CONST-vector-op-cost-910_9362`
  实测表明**单 op 边际代价随 block_size 大幅变化**（16m fp16 普通 v-op ≈3.4–7.8µs 而非固定值，小张量受发射开销主导），
  故本下界真实区间约 **[4 µs, 47 µs]**，属**未实测常数区**；且候选 0/1/2 的排序（47/20/28）在该区间内**可能反转**
  （尤其候选 0 的 per-task 前导/epilogue 开销是否真的按 op 计）。→ **强制进 §1.6.3 实验裁决模式**（决定段数归一化与主选/备选）。

**调研结论**：选定 **算法族 = 分段 online softmax + 段级 LSE 两阶段合并（源 split 链的数学结构）**，
以**两个 NPU kernel（Phase A 分段 / Phase B 合并）**落地，段划分按**编译期页对齐**重设计（OPT-S），
段数 `num_split_eff` 按物理核预算归一化（§1.6.3）。
- **关键依据**：R3 三候选访存量同阶（带宽受限，工作区占比 <0.4%），差异在发射项与并行度；R4 中 Cube/Vector
  分工匹配 NPU（候选 3 访存劣淘汰）；R1 负向候选 5/6/7 均有据淘汰。
- **与基线（源算法）的结构差异**：仅实现层（grid/段划分/静态界/装载形态/两 kernel 编排/v-prefix）+ 数学等价优化（§1.6.1），算法族不变。
- **候选 1 vs 候选 0 的关系**：候选 1（单遍）**是候选 0 在 `num_split_eff=1` 时的退化**，二者不是竞争算法族而是
  **同一族的不同段数**——段数是本设计的自由变量（§1.6.3 归一化 + 实验裁决），故候选 1 不单列为「更优替代」，而是并入候选 0。
- **候选 2 的定位**：核内分段 + 核内合并省去 GM 工作区往返与二次 launch，是**同族的结构变体**（§1.6.3 备选）。
- **「无更优替代」的调研范围**：查过 `algorithm-candidates.md` 的 `ALG-attention`/`ALG-softmax`/`ALG-gemv`/`ALG-reduction`
  命中行、`attention.md` 的 `PL-1.7-expert-persistent-boundary`/`PL-1.9-twophase`/`PL-1.11-causal-mask-scalartrap`、
  `examples/flash_attention/flash_attn_npuir_dev.py`（Developer MixCV 先例）、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`
  （多 kernel host 编排先例）、迁移 prefill `_gqa_prefill_fwd_kernel`（两相位注意力迁移先例），并做 R3/R4 结构分析——
  **无访存量或遍数更低的候选**（KV 必须各读一次，算术强度 ≈2，无重算空间；段级合并的两阶段性已证不可约）。
- **源算法优化意图承接**：§0.4 O1–O15 全部由本选定算法 + §0.6 承接（无静默丢弃，见 §0.6 末段）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定算法。**等价性论证 = 数学恒等 + D-1 机器验证**（`verify_equiv.py`，结果表见下）。
> baseline 式 = 源 `Softmax`/`rescale`/`combine`；候选式 = 采纳优化后的 NPU 公式。

| # | 优化项 | 原式（源） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|-----------|-----------|-----------|---------|
| OPT-1 | **段内 scale 折入 Q** | `S=(QKᵀ)·s`，`s=log2e/√D`（每段内每块 1 个 `[16,page]` vmul） | `Q̃=Q·s`（每任务 1 个 `[16,D]` vmul）；`S̃=Q̃Kᵀ` | 数学恒等：`(Q·s)Kᵀ=(QKᵀ)·s`。fp16 重结合：`s` 在 fp16 施加于 Q vs fp32 后置——**D-1 max_abs 2.441e-04 ≪ atol 1e-2，违反率 0** | 每任务由 `P` 个 `[16,page]` vmul 降为 **1 个 `[16,D]`**（smoke P=1：持平；长 KV 显著省） |
| OPT-2 | **重缩放因子重结合** | `scores_scale=exp2(m_p·s−m_c·s)`（2 mul + 1 sub on `[16]`） | `scores_scale=exp((m_p−m_c)·s)`（1 sub + 1 mul） | 数学恒等（分配律 + 换底）。**D-1 raw 域真对照**：源式 `exp2(m_p·s−m_c·s)` vs 重结合 `exp2((m_p−m_c)·s)`，fp32 因子级 **max_abs 4.768e-07 / max_rel 3.122e-07**（28 块）证非空；fp16 输出级 max_abs **0.0** | 段内每块省 1 个 `[16]` mul（`P` 个/任务） |
| OPT-3 | **自然 exp 域** | `exp2(x)`，`s=log2e/√D` | `exp(x)`，`s=1/√D` | 数学恒等（换底）：`exp2(x·log2e)=exp(x)`。**D-1 max_abs 3.052e-05** | 去 log2e 因子、去 `T.vexp2` 的 `tmpBuffer`（`T.vexp2.md §1` 需额外 tmp 且**不支持 bf16**），改用 `T.vexp`（fp16/fp32） |
| OPT-S | **段划分页对齐 + 静态界 + 钳位 + 恒等稳定化 + 有限 LSE 哨兵** | host token 级划分（block_N 倍数、非 page 对齐、**动态 trip**、`glse` 用 `-inf`/`rs=0` 守卫） | 编译期 `P=ceildiv(nkv,num_split_eff)` 页划分 + `T.Pipelined(P)` 静态界 + `pgc=min(pg,nkv−1)` 钳位 + 全局位置掩码 + 全掩码块 `rs=1,P=0` + `LSE_SENT=-30000` | **分段重组是精确恒等**（`Σ_i exp(l_i−LSE)·o_i = Σ_j exp(s_j−LSE)V_j`，任意连续划分）；钳位页号 + 全局位置掩码与源 `k_global·block_N+j<real_kv` 逐位等价（`pg≥nkv ⟹ 全掩码`）；`rs=1` 对**段尾**全掩码块零贡献且不清零（段尾之后无有效块）；`LSE_SENT=-30000` 是 fp16 可表示有限值，规避全空段 `exp(−inf−(−inf))=NaN`。**D-1：OPT-S vs 单遍 max_abs 3.815e-04（源公式）/ 3.662e-04（+OPT-1/2/3），违反率 0；MASK-CHECK max_abs 9.766e-04、无 NaN/inf、`real_kv=0` 输出 0** | 段内循环界静态（规避动态 trip count）；段划分与 host 一致化；防 NaN/错误归零 |

**D-1 等价性机器验证结果**（`examples/mha_decode_paged/_mha_decode_split_kernel/verify_equiv.py`，torch CPU；
fp16 mixed-precision 候选式 vs 基线式；D∈{64,128} × page∈{128,256} × N_kv∈{512,1024} × nsplit∈{2,4} ×
real_kv∈{N_kv, N_kv−1, N_kv/2+3, page, 1, 0} × 2 seed；对照基准 = 同 dtype 舍入路径的基线式；门限 atol=rtol=1e-2）：

| 采纳项 | 最大绝对差 (max_abs) | 最大 ulp 差 | 违反率 | 结论 |
|--------|---------------------|-------------|--------|------|
| OPT-1（fold scale into Q） | 2.441e-04 | 743.00 | 0.000e+00 | **EQUIV_PASS** |
| OPT-2（reassociated rescale, raw-domain） | 0.000e+00 | 0.00 | 0.000e+00 | **EQUIV_PASS** |
| OPT-3（natural-exp domain） | 3.052e-05 | 0.76 | 0.000e+00 | **EQUIV_PASS** |
| OPT-1+2+3（段内采纳集合） | 2.441e-04 | 904.29 | 0.000e+00 | **EQUIV_PASS** |
| **OPT-S**（split+combine vs 单遍，源公式） | 3.815e-04 | 1322.35 | 0.000e+00 | **EQUIV_PASS** |
| **OPT-S + OPT-1+2+3（全采纳集合）** | 3.662e-04 | 1712.16 | 0.000e+00 | **EQUIV_PASS** |

> **无未通过（FAIL）项**。`max_ulp` 偏大源于近零元素的 ULP 归一化（分母钳到 fp16 最小正规数 2⁻¹⁴），
> 绝对差 ≪ atol=1e-2；所有配置**违反率 0**。
> **OPT-2 臂有效性**：fp16 输出级 max_abs = 0.0（重结合误差低于 fp16 分辨率），脚本额外输出 **raw 域 fp32 因子级对照**
> `exp2(m_p·s−m_c·s)` vs `exp2((m_p−m_c)·s)` → `max_abs=4.768e-07 / max_rel=3.122e-07`（28 块）——两式为**真实不同的表达式**，非空臂。
> **MASK-CHECK（OPT-S 附加证据）**：同链对照「源**动态段界**」与「**静态页划分 + 钳位 + 同域恒等守卫 + `LSE_SENT`**」，
> 覆盖 `real_kv ∈ {0,3,8,9,17,24,32}`（含整除/非整除/零长度 × nsplit∈{2,4}）：`max_abs=9.766e-04`、**无 NaN/inf**、
> `real_kv=0` 输出为 0 → **PASS**。

**优化结论**：采纳 **4 项**（OPT-1/2/3/OPT-S，全部 EQUIV_PASS）。**优化后公式**（供 §3.1 使用）：

```
# 段划分（编译期）
nkv = ceildiv(N_kv, page)                      # 物理页数
num_split_eff = min(num_split, nkv)            # 无空段
P = ceildiv(nkv, num_split_eff)                # 每段页数（静态界）
s_nat = 1/sqrt(D)
# --- Phase A：段 sid ∈ [0,num_split_eff)，页 [sid*P, (sid+1)*P) ---
Q̃ = Q · s_nat                                  # 每任务一次（OPT-1）
S̃ = Q̃ Kᵀ                                       # Cube, fp32 累加
S̃ = mask(S̃)                                    # 全局位置 pg*page+j < real_kv 之外 → 哨兵 -1e38
m_cur = rowmax(S̃)                              # 全掩码块 → -1e38
all_masked = (m_cur ≤ -1e30)                   # 同域守卫
m_cur_safe = all_masked ? 0 : m_cur
rs = all_masked ? 1 : exp(m_prev − m_cur_safe) # OPT-2 + 全掩码恒等（保状态）
P_ = exp(S̃ − m_cur_safe)                       # OPT-3；全掩码块 → exp(-1e38) = 0
logsum = logsum·rs + rowsum(P_)                # 全掩码块：·1 + 0
acc_o = acc_o·rs + cast_f16(P_) @ V            # O10 量化点保持
m_ref  = all_masked ? m_ref : m_cur            # 段 LSE 的参考 max
m_prev ← m_cur                                 # 循环体末尾（先算后存）
o_sid = acc_o / logsum                         # logsum==0 → 0
l_sid = (logsum==0) ? LSE_SENT : ln(logsum) + m_ref
# --- Phase B：段间合并（自然对数域）---
m  = max_i l_i                                 # LSE_SENT=-30000（全空段 → m=LSE_SENT）
LSE = ln( Σ_i exp(l_i − m) ) + m               # 无 NaN（m 有限）
w_i = exp(l_i − LSE)
o  = Σ_i o_i · w_i                             # → cast → Output
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案全部循环与标量计算点；可替代的用 v-prefix / `T.reduce` / `T.Parallel` 替代，不可替代的逐项给充分理由。
> API 存在性均以 `docs/Tilelang.language/` 与 `examples/` 佐证。

**Phase A（段内）**：与 no-split 结构同构，逐元素/行级计算点（1–10）与保留点（11–13）逐条与
`_mha_decode_no_split_kernel/DESIGN.md §1.6.2` 相同（掩码链、online softmax、P 量化、归一化、`logsum==0` 守卫、
`cid→(mid,hid,bz,sid)` 任务反解、`phys=block_table[bz,pgc]`、`real_m` 裁剪、`T.Pipelined`；`clear=True` 强制）。
**Phase A 新增点**：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| A-S1 | 页号钳位 `pgc = min(pg, nkv−1)` | 标量 | 无 | ❌ | **block 级索引**：每块 1 次标量，用于构造 `T.copy` 地址与 `vbrc(col_lim)` 的标量源；非逐元素热点 |
| A-S2 | 全局列限 `col_lim = real_kv − pg·page` | 标量 | `T.vbrc(col_lim, lim_mat)` 广播后矩阵-矩阵 `T.vcmp`（承接 no-split `TRAP-vcmp-scalar-operand` 实测：运行时标量直接作 `T.vcmp` 第二操作数令 npuir 编译器段错误） | ✅ | — |
| A-S3 | 段级 LSE `l_sid = ln(logsum) + m_ref` | 标量 | `T.vln`（`docs/Tilelang.language/数学操作/T.vln.md`，fp16/fp32）+ `T.vadd` + `T.vselect`（空段→哨兵） | ✅ | — |
| A-S4 | `m_ref` 跟踪 | `if all_masked then m_ref else m_cur` | `T.vselect(all_masked, m_ref, m_cur, m_ref)`（`[16,1]` 行向量） | ✅ | — |

**Phase B（段间合并）**：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| B1 | 段间 max `m = max_i l_i` | `T.reduce_max(dim=0)`（源） | `T.serial(num_split_eff)` + `T.vmax`（`[16,1]` 行向量）——**规避 `T.reduce(dim=0)` 跨步归约未实测路径** | ✅ | — （`num_split_eff` 小且编译期常量，展开成本可忽略） |
| B2 | 段间 `Σ exp(l_i−m)` | `T.Pipelined(num_split)` 逐元素 | 展开 `T.serial` + `T.vsub`/`T.vexp`/`T.vadd`（`[16,1]`） | ✅ | — |
| B3 | `LSE = ln(Σexp)+m` | `log2`+`pipelined`（源） | `T.vln` + `T.vadd`（`[16,1]`） | ✅ | — |
| B4 | 加权合并 `o = Σ o_i·w_i` | `T.Parallel(bm,dim)` 逐元素 | `T.vcast`(po→f32) + `T.vbrc`(w `[16,1]`→`[16,D]`) + `T.vmul` + `T.vadd`（`T.vbrc.md` 示例2 广播） | ✅ | — |
| B5 | 最终 cast + 行裁剪写出 | 逐元素 | `T.vcast(rint)` + `T.copy`（双显式同 extent 切片，`TRAP-T-copy-region-semantics` ②） | ✅ | — |
| B6 | `cid→(mid,hid,bz)` 任务反解 | 标量整除/取模 | 无 | ❌ | **block 级索引 / 任务映射**：每核 1 次，非逐元素热点 |
| B7 | 段循环 `T.serial(num_split_eff)` | block 级循环 | 保留（非逐元素；编译期常量展开） | ❌（合理保留） | 段间为**两阶段归约**（max→加权），非单遍可结合；且 `num_split_eff` 小 |

**向量化结论**：Phase A/B 的**逐元素与行级计算已全部向量化**（Phase A 点 1–10 + A-S2/S3/S4，Phase B 点 B1–B5，
替代 API 均有 `docs/` 或 `examples/` 佐证）；保留的标量/循环点理由类别：**block 级索引/任务映射**（A-S1、B6）、
**tile 级两阶段归约依赖**（B7）、**依赖动态 shape 的边界**（`real_m`，同 no-split）。无「实现简单/照源码写」类保留。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（BSHD 进 / BSHD 出，**不重排 GM 布局**；`glse`/`Output_partial` 是接口工作区，布局固定）。
> 核内布局与向量化轴是**设计变量**。

**Phase A**：布局决策与 no-split 完全同构（`scores[block_M, block_N]` reduce dim=1；`acc_o[block_M, D]`；
向量化轴 = 连续尾轴 `block_N`/`dim`；**无 repack**）——见 `_mha_decode_no_split_kernel/DESIGN.md §1.6.3`
（主选 #1 源定向；候选 2 备选 KV-as-M GEMV；候选 3/4 host permute / 核内 transpose 弃选）。本设计沿用同一结论。

**Phase B（合并）候选矩阵**（枚举 + 弃选量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **I/O 原生**（`lse[16,1]` 行向量；`po[16,D]` 原布局；段维展开） | 连续尾轴 `dim`；段间用行向量 v-op | 无 | 基线；无 `dim=0` 跨步归约、无 transpose | **✅ 主选** |
| 2 | 源式：`lse[num_split,16]` + `T.reduce(dim=0)` | 段维跨步 | 无 | `T.reduce(dim=0)` 跨步归约效率**未实测**（pattern-library 未覆盖）；`num_split` 小（≤8），展开更稳 | ❌（弃选，无收益且引入未实测路径） |
| 3 | 核内 transpose `lse→[16,num_split]` 使归约沿连续轴 | `num_split` | `T.transpose`（UB 级 ~µs，`docs/Tilelang.language/创建操作/T.transpose.md`；`ALG-transpose`） | 无必要：主选已用行向量展开规避跨步；额外转置只增向量管线开销与 UB | ❌（无收益） |
| 4 | host permute `Output_partial` 进出 | `dim` | 两次 host permute | repack ~百 µs 级（`PL-1.3-host-permute`）> 算子本体流量（~4µs），**净亏** | ❌ |

**Phase B 布局决策结论**：主选 **#1 = I/O 原生 + 段维行向量展开**（无 repack、无 `dim=0` 归约、无 transpose）。
该结论同步落入 §3.3 伪代码与 §6（合并循环的内层向量维 = 连续尾轴 `dim`）。

**实验裁决模式（强制三件套）**：§1.6.0 roofline 显示候选 0/1/2 的发射项排序依赖 **`CONST-vector-launch-overhead`
（0.5µs/op，**910B2C 实测、9362 未重验**）与 `CONST-vector-op-cost-910_9362`（单 op 边际代价随 block_size 大幅变化）
——两者均为**未在 9362 实证的常数**，且决定「段数 `num_split_eff` 是否应归一化 / 是否退化为单遍」。故强制三件套：
- **主选方案（进 Stage 3）**：#0 = 两 kernel 分段 + 段级合并，段划分按编译期页划分，`num_split_eff = min(num_split, nkv)`
  （§3/§4/§6 只承载主选）。**段数归一化作为保留旋钮**：§6/§7 的分核方案给出 `num_split_eff` 的两种取值口径
  （`min(num_split, nkv)` 与「按核预算归一化 `min(num_split, nkv, max(1, floor(AIC/(num_q·B·H))))`」），默认前者。
- **备选方案（≤1 个）**：#2 = **单 kernel 核内分段 + 核内合并**（Design S）——buffer：段累加器 `seg_o[num_split][16,D]f32`
  （≈32KB UB）、段 LSE `seg_l[num_split][16,1]f32`；结构：grid `num_q·B·H`（一核一 (b,h)，**天然对齐物理核**），
  段循环内做 4 组独立 online（同 §3.3 段内链），核内用 §1.6.2 B1–B5 合并后写 `Output`；**不使用** `glse`/`Output_partial`
  GM 工作区、**单次 launch**；分核三要素按新粒度重算（逻辑核 = `B·H ∈ {8,16} ≤ 20`）。达 Stage 4 可直接实现深度。
- **裁决计划**：代表 shape = smoke `(1,16,1,512,128,128,nsplit=4)` 与 `(1,8,1,1024,64,256,nsplit=4)` 与
  `(1,8,1,512,64,256,nsplit=4)`（B*H=8 档）；指标 = msprof op Task Duration（median≥20）+ 反解单 op 发射成本
  （对比 `CONST-vector-launch-overhead`）；**判定阈值** = 备选 latency < 主选 × 0.9 才翻转（全局或**按 shape 分片**）；
  结果回写 §1.6.3 并触发设计修订回写路径。**shape 特化**：工厂 `lru_cache`（per-shape），可按 `B·H`/`nkv` 分派
  主选/备选，或对主选按 shape 取不同 `num_split_eff`（含 `=1` 退化为单遍）。
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Developer**（两个 kernel 均为 Cube `T.gemm` + Vector v-prefix 的 **MixCV** 形态；
`combine` 为纯 Vector）。host 侧由一个 Python closure 顺序 launch 两个 Developer kernel。

### 2.2 选型理由

- 任务指定编程模式 = `developer`；脚手架 `mha_decode_paged.py`（K9）按 Developer 形态调用（无 `threads`、无显式 `Scope`）。
- **Phase A** 同时含 Cube 侧 `T.gemm`（2 处）与 Vector 侧 v-prefix（`T.vsub/vexp/vmul/vcast/vdiv/vbrc/vln`），
  满足 `tilelang-mixcv-skill` 的 **Developer 模式 MixCV 判定规则**（AGENTS.md「Developer-mode MixCV trigger rule」）；
  参照已上库先例 `examples/flash_attention/flash_attn_npuir_dev.py`。
- **Phase B** 无 `T.gemm`，为纯 Vector 归约/加权 kernel（`T.Kernel(num_logical, is_npu=True)`）。
- **非持久网格**：规避 `TRAP-DEVMODE-PERSIST-GEMM`（Developer + persistent + gemm 运行时崩溃）——每核 1 任务。
- **两 kernel 由 host closure 编排**（§0.6 R5）：npuir 一 prim_func 一 kernel，故 Phase A/B 拆为两个
  `@tilelang.jit` kernel，由工厂返回的 Python closure 顺序 launch；先例 `tilelang_sparse_mla_bwd.py` /
  `_gqa_prefill_fwd_kernel.py`（内层工厂返回 closure）。
- 不选 Expert：Expert 双 Scope 手动 `sync_block_set/wait` 用于**同 kernel 内**跨引擎流水，本设计的两相位是
  **跨 kernel 的段级合并**（stream 序即屏障），无需 Expert 硬边界；且 Expert 硬边界（无 fragment/`T.Pipelined`）
  会使段内 online softmax 表达复杂化。

### 2.3 模式影响

| 维度 | Phase A（`_split_partial`） | Phase B（`_combine`） |
|------|----------------------------|----------------------|
| Kernel 声明 | `T.Kernel(num_logical_A, is_npu=True) as (cid, _)`（一维、非持久） | `T.Kernel(num_logical_B, is_npu=True)` |
| 内存分配 | `T.alloc_shared`（Q/K/V → L1）；`T.alloc_fragment`（acc_s/acc_o/行向量 → UB） | `T.alloc_fragment`/`T.alloc_shared`（lse/po/acc → UB） |
| 计算方式 | `T.gemm`（Cube）+ v-prefix 段内 online softmax（Vector），编译器自动 CV split | 纯 v-prefix（`vmax/vsub/vexp/vadd/vln/vmul/vcast/vbrc`） |
| 同步 | **自动同步**（Developer；`T.Pipelined` 自动插同步）——无手动 `sync_block_set/wait` | 自动同步 |
| 跨 kernel | host closure 顺序 launch（stream 序）；`glse`/`Output_partial` 为 GM 工作区 | 读同一工作区 |
| `pass_configs` | 不设（见 §7.3） | 不设 |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 的**优化后公式**（段内自然 exp 域 + scale 折入 Q + 重结合重缩放 + 段划分页对齐 + 段级 LSE 合并）。

**Phase A（段 sid 内）**

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `pg = sid·P + k`；`pgc = min(pg, nkv−1)`；`phys = block_table[bz,pgc]` | 编译期页划分 + 钳位寻址 |
| 2 | `Q̃ = Q·(1/√D)` | 段内 scale 折入 Q（OPT-1），每任务一次 |
| 3 | `S̃ = Q̃ Kᵀ` | score GEMM（Cube），fp32 累加 |
| 4 | `S̃ ← mask(S̃)` | 全局位置 `pg·page+j < real_kv` 之外 → 哨兵 `-1e38`（值替换） |
| 5 | `m_cur = rowmax(S̃)`；`all_masked = m_cur ≤ -1e30`；`m_cur_safe = all_masked ? 0 : m_cur` | 行 max + **同域**守卫 |
| 6 | `rs = all_masked ? 1 : exp(m_prev − m_cur_safe)`；`m_ref ← all_masked ? m_ref : m_cur` | 重缩放（OPT-2）+ 全掩码恒等 + 参考 max |
| 7 | `P_ = exp(S̃ − m_cur_safe)`；`s_sum = rowsum(P_)`；`logsum = logsum·rs + s_sum` | 指数（OPT-3）+ 段内在线累加 |
| 8 | `acc_o = acc_o·rs`；`m_prev ← m_cur` | 累加器重缩放 + **末尾存旧 max** |
| 9 | `acc_o += cast(P_)·V` | PV GEMM（Cube），P 量化到输入 dtype（O10） |
| 10 | `o_sid = acc_o/logsum`（`logsum==0 → 0`） | 段内归一化 |
| 11 | `l_sid = (logsum==0) ? LSE_SENT : ln(logsum) + m_ref` | 段级 LSE（自然对数域） |
| 12 | 写 `Output_partial[bz,rows,hid,sid,:] = cast(o_sid)`；`glse[bz,hid,sid,rows] = cast(l_sid)` | 段结果落接口工作区 |

**Phase B（段间）**

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 13 | `m = max_i l_i`（`i ∈ [0,num_split_eff)`） | 段间 max（`vmax` 展开） |
| 14 | `LSE = ln(Σ_i exp(l_i − m)) + m` | 段间 logsumexp（自然对数域，无 NaN） |
| 15 | `o = Σ_i o_i · exp(l_i − LSE)` | 加权合并 |
| 16 | `o ← cast(o)`；写 `Output[bz,rows,hid,:]` | 归一化 cast + 行裁剪写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 任务反解 | A:`cid→(mid,hid,bz,sid)` / B:`cid→(mid,hid,bz)` | 标量算术 | — | Developer（host closure launch） |
| 页钳位/寻址 | `pgc=min(pg,nkv−1)`；`phys=block_table[bz,pgc]` | 标量 `T.min` + BufferLoad | — | Developer |
| Q 载入/scale | `Q[...]→Q_shared`；`*s_nat` | `T.copy`（slice）+ `T.vmul` | src/dst 双显式 slice | Developer |
| K/V 载入 | `K|V[phys·page:(phys+1)·page, hid, :]→K_shared/V_shared` | `T.copy`（**slice 形态**） | `TRAP-load-nd2nz-strided` | Developer |
| score GEMM | `Q̃ Kᵀ` | `T.gemm(Q_shared, K_shared, acc_s, b_transpose=True, initC=True)` | fp16×fp16→fp32 | Developer（Cube） |
| 掩码 | 全局位置 OOB | `T.arange`（列索引）+ `T.vbrc`（`col_lim` 广播）+ `T.vcmp` + `T.vselect`（哨兵 `-1e38` 值替换） | 见 §6.4；`TRAP-vcmp-scalar-operand`（标量须先 vbrc） | Developer（Vector） |
| 行 max / 守卫 | `rowmax`；`all_masked` | `T.reduce_max(acc_s, m_cur, dim=1, clear=True)` + `T.vcmp(m_cur, thr_vec, all_masked, "le")` + `T.vselect` | `[16,1]` | Developer（Vector） |
| 重缩放/参考 max | `rs`；`m_ref` | `T.vsub`+`T.vexp`+`T.vselect`（`rs`）；`T.vselect`（`m_ref`） | `[16,1]` | Developer（Vector） |
| P / 行和 / logsum | `exp(S̃−m)`；`rowsum`；online | `T.vsub`+`T.vexp`；`T.reduce_sum(clear=True)`；`T.vmul`+`T.vadd` | — | Developer（Vector） |
| 重缩放 acc_o | `acc_o·rs` | `T.vmul(acc_o, rs, acc_o)`（`[16,1]` 行广播） | — | Developer（Vector） |
| P 量化 | `cast(P)` | `T.vcast(acc_s, acc_s_cast, round_mode="rint")` | src/dst 同 shape | Developer（Vector） |
| PV GEMM | `acc_o += P·V` | `T.gemm(acc_s_cast, V_shared, acc_o, initC=False)` | fp16×fp16→fp32 | Developer（Cube） |
| 段内归一化 | `acc_o/logsum` | `T.vselect`（`logsum==0→1`）+ `T.vdiv` | 广播除 | Developer（Vector） |
| 段级 LSE | `ln(logsum)+m_ref` | `T.vln` + `T.vadd` + `T.vselect`（空段哨兵） | `docs/.../数学操作/T.vln.md` | Developer（Vector） |
| 段结果写出 | `→ glse / Output_partial` | `T.vcast` + `T.copy`（局部行切片） | `TRAP-T-copy-region-semantics` ② | Developer（Vector） |
| 段内流水 | 循环调度 | `T.Pipelined(P, num_stages=2)` | 静态界 | Developer |
| 段间 max/sum | `m=max l_i`；`Σ exp(l_i−m)` | `T.serial(num_split_eff)` + `T.vmax` / `T.vsub`+`T.vexp`+`T.vadd` | `[16,1]`（规避 `dim=0` reduce） | Developer（Vector） |
| 段间 log | `LSE = ln(Σ)+m` | `T.vln` + `T.vadd` | `[16,1]` | Developer（Vector） |
| 加权合并 | `o += o_i·exp(l_i−LSE)` | `T.vcast`(po) + `T.vbrc`(w `[16,1]`→`[16,D]`) + `T.vmul` + `T.vadd` | `T.vbrc.md` 示例2 | Developer（Vector） |
| 输出写回 | `→ Output` | `T.vcast(rint)` + `T.copy`（行裁剪） | 双显式同 extent 切片 | Developer（Vector） |

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型作为**第二位置参数**传入（不用 `dtype=` 关键字）；`vbrc/vcmp` 的**标量源必须 let-bound**
> （`TRAP-vcmp-scalar-operand`）。以下为主选方案（§1.6.3 #0/#1）。

```python
@functools.lru_cache(maxsize=32)
def _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype):
    accum_dtype = "float"
    block_M   = 16                                 # 分形下限（§0.6 R2）
    block_N   = page_size                          # 1 tile = 1 物理页（§0.6 R3）
    num_q     = (seqlen_q + block_M - 1) // block_M
    nkv       = (seqlen_kv + page_size - 1) // page_size   # 物理页数（编译期）
    s_nat     = 1.0 / (dim ** 0.5)                 # 自然 exp 域（OPT-1/OPT-3）
    OOB_SENT  = -1e38                              # 分数域哨兵（非 fp16 存储）
    NEG_THR   = -1e30                              # 同域守卫阈值
    LSE_SENT  = -30000.0                           # glse 哨兵（fp16 可表示！）

    @tilelang.jit(target="npuir")
    def _func(block_M_in, block_N_in, num_split, num_stages):
        ns_eff = min(num_split, nkv)               # 无空段（§0.6 R4）
        P      = (nkv + ns_eff - 1) // ns_eff      # 段内页数（静态界）
        num_logical_A = num_q * batch * heads * ns_eff
        num_logical_B = num_q * batch * heads

        # ---------------- Phase A: split partial ----------------
        @T.prim_func
        def split_partial(
            Q: T.Tensor([batch, seqlen_q, heads, dim], dtype),
            K: T.Tensor([seqlen_kv, heads, dim], dtype),
            V: T.Tensor([seqlen_kv, heads, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], "int32"),
            block_table: T.Tensor([batch, nkv], "int32"),
            glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),
            Output_partial: T.Tensor([batch, seqlen_q, heads, num_split, dim], dtype),
        ):
            with T.Kernel(num_logical_A, is_npu=True) as (cid, _):
                # 任务反解（§0.6 R1）
                sid = cid % ns_eff
                r   = cid // ns_eff
                hid = r % heads
                r2  = r // heads
                bz  = r2 % batch
                mid = r2 // batch
                real_kv = real_seqlen_kv[bz]
                real_m  = T.min(block_M, seqlen_q - mid * block_M)

                Q_shared   = T.alloc_shared([block_M, dim], dtype)
                K_shared   = T.alloc_shared([block_N, dim], dtype)
                V_shared   = T.alloc_shared([block_N, dim], dtype)
                O_cast     = T.alloc_shared([block_M, dim], dtype)
                acc_s      = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o      = T.alloc_fragment([block_M, dim], accum_dtype)
                m_prev     = T.alloc_fragment([block_M, 1], accum_dtype)
                m_cur      = T.alloc_fragment([block_M, 1], accum_dtype)
                m_cur_safe = T.alloc_fragment([block_M, 1], accum_dtype)
                m_ref      = T.alloc_fragment([block_M, 1], accum_dtype)
                rs         = T.alloc_fragment([block_M, 1], accum_dtype)
                s_sum      = T.alloc_fragment([block_M, 1], accum_dtype)
                logsum     = T.alloc_fragment([block_M, 1], accum_dtype)
                lse        = T.alloc_fragment([block_M, 1], accum_dtype)
                one_vec    = T.alloc_fragment([block_M, 1], accum_dtype)
                zero_vec   = T.alloc_fragment([block_M, 1], accum_dtype)
                thr_vec    = T.alloc_fragment([block_M, 1], accum_dtype)
                sent_lse   = T.alloc_fragment([block_M, 1], accum_dtype)
                all_masked = T.alloc_fragment([block_M, 1], "bool")
                is_zero    = T.alloc_fragment([block_M, 1], "bool")
                colmat     = T.alloc_fragment([block_M, block_N], "int32")   # j
                lim_mat    = T.alloc_fragment([block_M, block_N], "int32")
                vis        = T.alloc_fragment([block_M, block_N], "bool")
                sent_buf   = T.alloc_fragment([block_M, block_N], accum_dtype)

                T.vbrc(OOB_SENT, sent_buf); T.vbrc(1.0, one_vec); T.vbrc(0.0, zero_vec)
                T.vbrc(NEG_THR, thr_vec);   T.vbrc(LSE_SENT, sent_lse)
                T.arange(colmat, [0, 1], offset=0)           # colmat[i,j] == j

                # prologue（先 init 被读的缓冲）
                T.clear(Q_shared)
                T.copy(Q[bz, mid*block_M:mid*block_M+real_m, hid, 0:dim],
                       Q_shared[0:real_m, 0:dim])
                T.vmul(Q_shared, s_nat, Q_shared)                     # OPT-1
                T.clear(acc_o); T.clear(logsum)
                T.vbrc(-T.infinity(accum_dtype), m_prev)              # m_prev init -inf
                T.vbrc(-T.infinity(accum_dtype), m_ref)

                for k in T.Pipelined(P, num_stages=num_stages):       # 静态界（§0.6 R4）
                    pg  = sid * P + k
                    pgc = T.min(pg, nkv - 1)                          # 钳位页号
                    phys = block_table[bz, pgc]
                    T.copy(K[phys*block_N:(phys+1)*block_N, hid, 0:dim],
                           K_shared[0:block_N, 0:dim])
                    T.gemm(Q_shared, K_shared, acc_s, initC=True, b_transpose=True)
                    # 列掩码：全局位置 pg*block_N + j < real_kv
                    col_lim = real_kv - pg * block_N
                    T.vbrc(col_lim, lim_mat)                          # 标量→矩阵（vcmp 标量会段错误）
                    T.vcmp(colmat, lim_mat, vis, "lt")
                    T.vselect(vis, acc_s, sent_buf, acc_s)
                    # 段内 online softmax（先算 m_cur，末尾存 m_prev）
                    T.reduce_max(acc_s, m_cur, dim=1, clear=True)
                    T.vcmp(m_cur, thr_vec, all_masked, "le")
                    T.vselect(all_masked, zero_vec, m_cur, m_cur_safe)
                    T.vsub(m_prev, m_cur_safe, rs); T.vexp(rs, rs)     # OPT-2
                    T.vselect(all_masked, one_vec, rs, rs)            # 全掩码恒等
                    T.vsub(acc_s, m_cur_safe, acc_s); T.vexp(acc_s, acc_s)  # OPT-3
                    T.reduce_sum(acc_s, s_sum, dim=1, clear=True)
                    T.vmul(acc_o, rs, acc_o)
                    T.vmul(logsum, rs, logsum); T.vadd(logsum, s_sum, logsum)
                    T.vselect(all_masked, m_ref, m_cur, m_ref)        # m_ref 跟踪
                    T.copy(m_cur, m_prev)                             # 末尾（先算后存）
                    T.vcast(acc_s, acc_s_cast, round_mode="rint")     # O10
                    T.copy(V[phys*block_N:(phys+1)*block_N, hid, 0:dim],
                           V_shared[0:block_N, 0:dim])
                    T.gemm(acc_s_cast, V_shared, acc_o, initC=False)

                # epilogue：段内归一化 + 段级 LSE
                T.vcmp(logsum, zero_vec, is_zero, "eq")
                T.vselect(is_zero, one_vec, logsum, logsum)           # logsum==0 -> 1
                T.vdiv(acc_o, logsum, acc_o)
                T.vcast(acc_o, O_cast, round_mode="rint")
                T.copy(O_cast[0:real_m, 0:dim],
                       Output_partial[bz, mid*block_M:mid*block_M+real_m, hid, sid, 0:dim])
                T.vln(logsum, lse); T.vadd(lse, m_ref, lse)           # l = ln(logsum)+m_ref
                T.vselect(is_zero, sent_lse, lse, lse)               # 空段 -> LSE_SENT
                T.copy(lse[0:real_m, 0], glse[bz, hid, sid, mid*block_M:mid*block_M+real_m])

        # ---------------- Phase B: combine ----------------
        @tilelang.jit(out_idx=[-1], target="npuir")
        def _combine():
            @T.prim_func
            def combine(
                glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),
                Output_partial: T.Tensor([batch, seqlen_q, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, seqlen_q, heads, dim], dtype),
            ):
                with T.Kernel(num_logical_B, is_npu=True) as (cid, _):
                    mid = cid % num_q; r = cid // num_q
                    hid = r % heads;   bz = r // heads
                    real_m = T.min(block_M, seqlen_q - mid * block_M)
                    g    = T.alloc_fragment([block_M, 1], accum_dtype)
                    gcol = T.alloc_fragment([block_M, 1], accum_dtype)
                    mval = T.alloc_fragment([block_M, 1], accum_dtype)
                    ssum = T.alloc_fragment([block_M, 1], accum_dtype)
                    w    = T.alloc_fragment([block_M, 1], accum_dtype)
                    acc  = T.alloc_fragment([block_M, dim], accum_dtype)
                    po   = T.alloc_fragment([block_M, dim], dtype)
                    po_f = T.alloc_fragment([block_M, dim], accum_dtype)
                    O_cast = T.alloc_shared([block_M, dim], dtype)

                    T.vbrc(-T.infinity(accum_dtype), mval); T.vbrc(0.0, ssum); T.clear(acc)
                    # 段间 max
                    for i in T.serial(ns_eff):
                        T.copy(glse[bz, hid, i, mid*block_M:mid*block_M+real_m], g[0:real_m, 0])
                        T.vmax(mval, g, mval)
                    # 段间 logsumexp（自然对数域）
                    for i in T.serial(ns_eff):
                        T.copy(glse[bz, hid, i, mid*block_M:mid*block_M+real_m], g[0:real_m, 0])
                        T.vsub(g, mval, g); T.vexp(g, g); T.vadd(ssum, g, ssum)
                    T.vln(ssum, g); T.vadd(g, mval, g)                    # LSE
                    # 加权合并
                    for i in T.serial(ns_eff):
                        T.copy(Output_partial[bz, mid*block_M:mid*block_M+real_m, hid, i, 0:dim],
                               po[0:real_m, 0:dim])
                        T.copy(glse[bz, hid, i, mid*block_M:mid*block_M+real_m], gcol[0:real_m, 0])
                        T.vsub(gcol, g, w); T.vexp(w, w)                  # w_i = exp(l_i - LSE)
                        T.vcast(po, po_f, round_mode="rint")
                        T.vmul(po_f, w, po_f); T.vadd(acc, po_f, acc)
                    T.vcast(acc, O_cast, round_mode="rint")
                    T.copy(O_cast[0:real_m, 0:dim],
                           Output[bz, mid*block_M:mid*block_M+real_m, hid, 0:dim])

        k_split = split_partial                       # Phase A（写 glse/Output_partial）
        k_comb  = _combine()                          # Phase B（out_idx=[-1] -> Output）

        def wrapped(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial, acc_split_length):
            # 接口契约：8 输入 -> Output；glse/Output_partial 为 host 预分配工作区
            # （acc_split_length 不被使用——段划分为编译期页划分，OPT-S）
            k_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial)
            return k_comb(glse, Output_partial)

        return wrapped

    return _func
```

> 说明：伪代码为**主选方案**的结构骨架，固化 buffer 形状、向量化轴（连续尾轴）、循环与同步分层。
> **不变量**：被读缓冲先 init（`m_prev`/`m_ref` 循环外 init `-inf`；循环内**先算 `m_cur` 再用 `m_prev`**、
> `m_cur` 在末尾才写回 `m_prev`）。**掩码链**：`T.arange`/`T.vcmp`/`T.vbrc`/`T.vselect` 的具体形态以 Stage 3 实测为准
> （`TRAP-vbrc-empty-broadcast`）。**同域守卫**：`all_masked`（`≤ -1e30`）与哨兵 `-1e38` 同域。
> **`LSE_SENT` 必须 fp16 可表示**（`-30000`，非 `-1e38`）——`glse` 以输入 dtype 存储。`ns_eff`/`P` 为编译期常量，
> 故 `T.serial(ns_eff)` 与 `T.Pipelined(P)` 均为静态界。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|----------|------|
| `T.gemm`（`b_transpose`/`initC`） | `docs/Tilelang.language/线性代数操作/T.gemm.md`；`flash_attn_npuir_dev.py`；`examples/gemm` | ✅ 文档 + 先例（M/N 分形下限为「未文档化假设」） |
| `T.reduce_max`/`T.reduce_sum`（`dim`/`clear`） | `docs/Tilelang.language/规约操作/T.reduce.md`（`clear=False` 静默错误 → 一律 `clear=True`；bf16 **不支持**，本设计全程 fp32 累加器） | ✅ 文档 |
| `T.vexp`/`T.vsub`/`T.vmul`/`T.vadd`/`T.vdiv`/`T.vcast`/`T.vbrc`/`T.vmax`/`T.vselect`/`T.vcmp` | `docs/Tilelang.language/{数学操作,数据类型转换操作,shape操作,比较操作,条件操作,逻辑操作}/`；`flash_attn_npuir_dev.py`；`_gqa_prefill_fwd_kernel.py` | ✅ 文档 + 先例 |
| `T.vln` | `docs/Tilelang.language/数学操作/T.vln.md`（fp16/fp32，1–5D，同 shape src/dst） | ✅ 文档 |
| `T.vexp2`（备选，若坚持 exp2 域） | `docs/Tilelang.language/数学操作/T.vexp2.md`（需 `tmpBuffer`；**不支持 bf16**） | ✅ 文档（本设计改用 `T.vexp`，OPT-3） |
| `T.copy`（**slice 形态**） | `TRAP-load-nd2nz-strided` / `TRAP-T-copy-region-semantics`（repro 存在）；`_gqa_prefill_fwd_kernel` | ✅（**必须 slice 形态**） |
| `T.Pipelined` | `flash_attn_npuir_dev.py`；`_gqa_prefill_fwd_kernel` | ✅ 先例 |
| `T.alloc_shared`/`T.alloc_fragment` | `flash_attn_npuir_dev.py` | ✅ 先例 |
| `T.arange` | `docs/Tilelang.language/创建操作/T.arange.md`；no-split 实测 | ✅ 文档 + 先例 |
| 多 kernel 顺序 launch（Python 编排） | `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` | ✅ 先例 |
| 工厂返回 Python closure | `_gqa_prefill_fwd_kernel.py`（`wrapped`） | ✅ 先例 |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源 A `(q, H·B, ns)`；B `(q, H, B)`） | §0.6 R1：两个一维 `T.Kernel(num_logical)` + cid 反解 |
| 单 prim_func 多 kernel launch | **Yes**（源同 prim_func 两次 launch） | §0.6 R5：拆两个 `@tilelang.jit` kernel + host closure 顺序 launch（先例 `tilelang_sparse_mla_bwd.py`） |
| GPU 专用 API | **Yes**（`threads=`、`T.GemmWarpPolicy`、`disable_tma`、`make_swizzled_layout`、`annotate_layout`/`forward_thread_fn`） | `threads`/`GemmWarpPolicy`/`disable_tma`/swizzle/thread_fn 全部舍弃（O9/O14/O15） |
| GEMM 非整除风险 | **Yes**（`S_q=1` 不被 `block_M=16` 整除；`dim`/`page_size` 分形对齐） | R2：`block_M=16` + padding 行隔离 + 写回按 `real_m` 裁剪；`dim,page ∈ {64,128,256}` 均 16 倍数 ✓ |
| L0C 溢出（`bm×bn×4 > 128KB`） | **No**（`16×256×4 = 16KB`，且 `16×256 ≤ 32768`） | `CONST-cube-gemm-l0c-traffic-floor`（L0C fp32 容量 32768 元素）✓ |
| 分核策略缺失/不适配 | **Yes**（Phase A 逻辑核 64 > 物理核 20） | §5.5「无法对齐兜底」：非持久超发 grid + 记录残留负载不均 + `num_split_eff` 归一化/分派（§1.6.3） |
| 动态 shape / trip count | **Yes**（源段内界为运行时标量） | §0.6 R4：编译期页划分 `P` 静态界 |
| `glse` 工作区精度 | **Yes**（输入 dtype 存储段级 LSE） | §9 风险 R-6（D-1 实测 OPT-S 往返 max_abs 3.7e-04） |

### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度/数量 | 2 个三维 kernel（同 prim_func 两次 launch） | 2 个一维非持久 kernel | §0.6 R1/R5 |
| 段划分 | host token 级（block_N 倍数、非 page 对齐） | 编译期页划分 `P=ceildiv(nkv,ns_eff)` | §0.6 R4 / OPT-S |
| 段内循环界 | 动态 `blocks_in_split` | 静态 `P` + 页号钳位 | §0.6 R4 |
| 页寻址 | `block_table[b, k_global//nbpp]*nbpp + k_global%nbpp` | `block_table[b, min(pg,nkv−1)]`（`block_N=page`） | §0.6 R3 |
| 掩码/守卫 | `-inf` + `max==-inf → rs=0` | 有限哨兵 `-1e38` + **同域**守卫 + 恒等 `rs=1` | §0.6 R4 |
| 段级 LSE 存储 | log2 域、输入 dtype | 自然对数域、输入 dtype、`LSE_SENT=-30000` | §0.6 R7 / OPT-3 |
| 合并 | swizzled layout + `reduce(dim=0)` + `T.Pipelined` | 行向量 v-op 展开、无 swizzle、无 `dim=0` reduce | §0.6 R7 |
| softmax 表达 | `T.macro` + `T.Parallel` | v-prefix 链 + `T.reduce` | §0.6 R6 |
| 线程/TMA | `threads=`、`disable_tma` | 无 | 舍弃（O9/O15） |
| GEMM policy | `T.GemmWarpPolicy.FullRow` | 无对应 | 舍弃 |
| `pass_configs` | `TL_ENABLE_FAST_MATH=True` | **不设**（npuir `PassConfigKey` 无该成员，照搬即 `AttributeError`——no-split Stage 3 实测） | §7.3 |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/mha_decode_paged/_mha_decode_no_split_kernel/`（同任务前序函数） | **最高** | Developer MixCV 已验证形态、paged slice 装载、静态界同域恒等守卫、v-prefix 链、分核三要素写法（**本设计直接复用**） |
| `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` | **高** | **多 kernel（3 个）host 顺序编排**（preprocess→bwd→postprocess）——两 kernel 编排的先例 |
| `examples/flash_attention/flash_attn_npuir_dev.py` | 高 | Developer MixCV：`T.Kernel(n,is_npu=True)` + `T.gemm` + v-prefix online softmax + `T.Pipelined` |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高 | attention 族迁移先例（分形 clamp、slice 装载、`T.reduce(clear=True)`、行裁剪写回、**内层工厂返回 Python closure**） |
| `examples/flash_attention/flash_attn_npuir.py` | 中 | Expert 双 Scope 两相位 flash（本设计不用，但佐证分块 KV + 归约结构） |
| `examples/gemv/example_gemv.py` | 中 | GEMV `T.gemm` 形态（§1.6.3 备选参考） |
## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Q` | `(B, S_q, H, D)` | fp16/bf16 | BSHD（`S_q=1`） |
| `K` / `V` | `(N_kv, H, D)` | fp16/bf16 | 物理 paged cache |
| `real_seqlen_kv` | `(B,)` | int32 | 逻辑 KV 长度 |
| `block_table` | `(B, nkv)` | int32 | 逻辑页→物理页（`nkv = N_kv/page_size`） |
| `glse` | `(B, H, num_split, S_q)` | fp16/bf16 | **工作区（输入）**：段级 LSE |
| `Output_partial` | `(B, S_q, H, num_split, D)` | fp16/bf16 | **工作区（输入）**：段级部分输出 |
| `acc_split_length` | `(B, num_split)` | int32 | **工作区（输入）**：累积段边界（**本设计不使用**——编译期页划分，OPT-S；保留形参维持接口） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Output` | `(B, S_q, H, D)` | fp16/bf16 | 与 Q 同 shape/同布局（BSHD，非转置）；Phase B `out_idx=[-1]` 自动分配 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 层级 | 用途 |
|-----------|-------|-------|------|------|
| **Phase A** | | | | |
| Q_shared | `(16, D)` | fp16/bf16 | L1 | Q tile（scale 折入后） |
| K_shared | `(page, D)` | fp16/bf16 | L1 | K 物理页 tile |
| V_shared | `(page, D)` | fp16/bf16 | L1 | V 物理页 tile |
| O_cast | `(16, D)` | fp16/bf16 | L1/UB | 段内输出 cast（写 `Output_partial`） |
| acc_s | `(16, page)` | fp32 | UB | 分数 / P（段内） |
| acc_s_cast | `(16, page)` | fp16/bf16 | UB | P 量化（送 gemm2） |
| acc_o | `(16, D)` | fp32 | UB | 段内输出累加器 |
| m_prev/m_cur/m_cur_safe/m_ref/rs/s_sum/logsum/lse | `(16, 1)` | fp32 | UB | 段内在线状态行向量 + 段级 LSE |
| one_vec/zero_vec/thr_vec/sent_lse | `(16, 1)` | fp32 | UB | 恒等守卫 / 同域阈值 / LSE 哨兵常数 |
| all_masked/is_zero | `(16, 1)` | bool | UB | 同域守卫 / 空段判定 |
| colmat/lim_mat | `(16, page)` | int32 | UB | 列索引 / 列限（掩码链） |
| vis | `(16, page)` | bool | UB | 列可见性 |
| sent_buf | `(16, page)` | fp32 | UB | 哨兵缓冲 |
| **Phase B** | | | | |
| g/gcol/mval/ssum/w | `(16, 1)` | fp32 | UB | 段间 max/logsumexp/权重 |
| acc | `(16, D)` | fp32 | UB | 合并累加器 |
| po | `(16, D)` | fp16/bf16 | UB | 段部分输出 tile |
| po_f | `(16, D)` | fp32 | UB | 段部分输出 fp32 中转 |
| O_cast | `(16, D)` | fp16/bf16 | L1/UB | 输出 cast |

### 4.4 内存搬运路径

```
Phase A（Developer 编译器自动分层）：
GM[Q] --slice T.copy--> L1[Q_shared(16,D)] --v-mul(*s_nat)--> L1[Q_shared]
GM[block_table] --标量--> pgc = min(sid*P+k, nkv-1); phys = block_table[bz,pgc]
GM[K] --slice T.copy--> L1[K_shared(page,D)]
   L1[Q_shared] + L1[K_shared] --T.gemm(b_transpose, Cube)--> UB[acc_s(16,page) f32]
   UB[acc_s] --Vector: mask→reduce_max→exp→reduce_sum→vcast--> UB[acc_s_cast f16]
GM[V] --slice T.copy--> L1[V_shared(page,D)]
   UB[acc_s_cast] + L1[V_shared] --T.gemm(initC=False, Cube)--> UB[acc_o(16,D) f32]
UB[acc_o] --Vector: /logsum, vcast--> GM[Output_partial(bz,rows,hid,sid,:)]
UB[logsum/m_ref] --Vector: vln+vadd+vselect--> GM[glse(bz,hid,sid,rows)]
--- host closure：stream 序（Phase A 全部完成）---
Phase B（纯 Vector）：
GM[glse] --copy--> UB --vmax/vexp/vln--> LSE(16,1)
GM[Output_partial] --copy--> UB[po f16] --vcast/vmul(w)/vadd--> UB[acc f32]
UB[acc] --vcast(rint)--> GM[Output(bz,rows,hid,:)]
```

> **Host 侧约束**：host 只做 dispatch 与工作区分配；**不改动输入张量内容**（`ascend-constraints.md §4`）。
> `glse`/`Output_partial` 由 host 预分配（`torch.zeros`），Phase A 写满所用分片。

### 4.5 UB / L1 / L0C 内存预算

**Phase A — UB 层**（Vector 侧，容量 192KB/AIV = 196,608 B；上界列取 `page=256, D=128`）

| Buffer | Shape | dtype | 大小 (B) |
|--------|-------|-------|---------|
| acc_s | (16,256) | fp32 | 16384 |
| acc_s_cast | (16,256) | fp16 | 8192 |
| acc_o | (16,128) | fp32 | 8192 |
| O_cast | (16,128) | fp16 | 4096 |
| 行向量 ×8（m_prev/m_cur/m_cur_safe/m_ref/rs/s_sum/logsum/lse） | (16,1) | fp32 | 512 |
| 常数 ×4（one/zero/thr/sent_lse） | (16,1) | fp32 | 256 |
| all_masked/is_zero | (16,1) | bool | 32 |
| colmat/lim_mat | (16,256) | int32 | 32768 |
| vis | (16,256) | bool | 4096 |
| sent_buf | (16,256) | fp32 | 16384 |
| **合计** | | | **≈ 91 KB** / 192 KB ✓ |
| 含 auto-multi-buffer 膨胀（×1.10–1.7，`CONST-capacity-910B2C`） | | | ≈ 100–155 KB ✓ |

> 降额手段：`colmat`/`lim_mat` 改 int16（`page≤256`、`real_kv≤N_kv` 均在 int16 域）→ 合计算 ≈ 74KB。

**Phase A — L1 层**（Cube 操作数，512KB/核）

| Buffer | Shape | dtype | 大小 (B) |
|--------|-------|-------|---------|
| Q_shared | (16,128) | fp16 | 4096 |
| K_shared | (256,128) | fp16 | 65536 |
| V_shared | (256,128) | fp16 | 65536 |
| **合计**（不含流水双缓冲） | | | **≈ 135 KB** ✓ |
| 含 `num_stages=2` K/V 双槽 | | | ≈ 266 KB ✓ |

**Phase A — L0C 层**（128KB；逐块 32768 元素上限）：gemm1 `acc_s` `16×256 = 4096` 元素（16KB）✓；gemm2 `acc_o` `16×128 = 2048` 元素（8KB）✓。

**Phase B — UB 层**：g/gcol/mval/ssum/w（5×64）+ acc 8192 + po 4096 + po_f 8192 + O_cast 4096 ≈ **25 KB** ✓（含膨胀 ≈ 43KB）。

> **容量项 ≈0**（见 §1.6.0 roofline）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `real_seqlen_kv[bz]` | 标量 BufferLoad | `0 ~ N_kv`（编译期 `N_kv` 固定）；**不影响循环界**（静态 `P` + 掩码，§0.6 R4） |
| `block_table[bz,pgc]` | 标量 BufferLoad | `0 ~ nkv−1`（页号动态，影响 `T.copy` 地址） |

> `B/S_q/H/D/N_kv/page_size/num_split` 均为编译期常量（工厂参数）；`ns_eff`/`P` 由它们编译期导出。

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir")            # Phase A: 写 glse/Output_partial（无 out_idx）
def _func(block_M, block_N, num_split, num_stages): ...

@tilelang.jit(out_idx=[-1], target="npuir")   # Phase B: Output 自动分配
def _combine(): ...
```

> **不设 `pass_configs`**：源 `TL_ENABLE_FAST_MATH` 在 npuir `PassConfigKey` 中**不存在**（no-split Stage 3 实测
> `AttributeError`），照搬即失败；沿用 Developer 默认（与已验证先例一致）。**不设** `NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`
> 等 Expert 专用开关（本设计为 Developer，保留自动多缓冲以利流水）。

---

## 5. Tiling 策略

### 5.1 计算类型

**Phase A**: 混合（MixCV）— Cube（2 个 `T.gemm`）+ Vector（段内 online softmax / epilogue）。
**Phase B**: 纯 Vector（段间两阶段归约 + 加权合并）。

### 5.2 Block 划分

```python
block_M = 16        # 分形下限；S_q=1 -> 1 真实行（§0.6 R2）
block_N = page_size # 1 tile = 1 物理页，block_table 1:1（§0.6 R3）；∈{128,256}
num_q   = ceildiv(S_q, block_M)                 # S_q=1 -> 1
nkv     = ceildiv(N_kv, page_size)              # 物理页数（编译期）
ns_eff  = min(num_split, nkv)                   # 段数（无空段，§0.6 R4）
P       = ceildiv(nkv, ns_eff)                  # 段内页数（静态界）
num_logical_A = num_q * B * H * ns_eff          # Phase A 逻辑核
num_logical_B = num_q * B * H                   # Phase B 逻辑核
```

| workload | B | H | S_q | N_kv | page | nkv | num_split | ns_eff | P | num_logical_A | num_logical_B |
|----------|---|---|-----|------|------|-----|-----------|--------|---|---------------|---------------|
| single-token-page128 | 1 | 16 | 1 | 512 | 128 | 4 | 4 | 4 | 1 | 64 | 16 |
| batch2-page256 | 2 | 8 | 1 | 1024 | 256 | 4 | 4 | 4 | 1 | 64 | 16 |
| longer-cache | 1 | 8 | 1 | 1024 | 256 | 4 | 4 | 4 | 1 | 32 | 8 |
| shorter-cache | 1 | 8 | 1 | 512 | 256 | 2 | 4 | **2** | 1 | 32 | 8 |

> 说明：`block_N = page_size` 使「1 个 KV tile = 1 物理页」，paged 间接寻址退化为每块 1 次页号读取（§0.6 R3）。
> `shorter-cache` 的 `nkv=2 < num_split=4` → `ns_eff=2`（不产生空段）。若需降低 Phase A 超发，可按
> §1.6.3 归一化 `ns_eff`（`min(num_split, nkv, max(1, floor(AIC/(num_q·B·H))))`）——Stage 4 旋钮。

### 5.3 约束分析

- **对齐约束**: `block_N ∈ {128,256}`（32B 尾轴：128·2B=256B ✓）；`dim ∈ {64,128}`（16 倍数，分形 K≥32 ✓）；`block_M=16`（分形 M≥16 ✓）。
- **UB 容量**: Phase A ≈91KB（含膨胀 ≤155KB）< 192KB ✓；Phase B ≈25KB ✓。
- **L1 容量**: Phase A ≈266KB（含双槽）< 512KB ✓。
- **L0C 容量**: `16×256×4 = 16KB < 128KB`，且 `16×256 = 4096 ≤ 32768` 元素 ✓。
- **GM 工作区**: `glse` = `B·H·num_split·S_q·2`（smoke 128B）；`Output_partial` = `B·S_q·H·num_split·D·2`（smoke 16KB，占比 <0.4%）。

### 5.4 注意事项（非整除处理策略）

- **`S_q` 不被 `block_M` 整除**：`S_q=1 < block_M=16` → `num_q=1`，仅 1 真实行；padding 行经 `Q_shared` 清零 + 逐行 softmax + 写回裁剪 `real_m = min(block_M, S_q−mid·block_M)` **完全隔离**。
- **`N_kv` 不被 `page_size` 整除**：`nkv=ceil(N_kv/page)`，末页列由全局位置掩码置哨兵 `-1e38` → `exp` 为 0。
- **`real_kv < N_kv`（段尾整块无效）**：静态界多迭代的尾部块**整块被掩码**；用**同域守卫**（`all_masked = m_cur ≤ -1e30`）置 `rs=1`（**恒等**）→ 本块对已累积 `logsum/acc_o` **零贡献且不清零**（§0.6 R4）。
- **`real_kv = 0`**：全部段全掩码 → `logsum=0`、`l_i=LSE_SENT`、`o_i=0` → 合并输出 **0**（与源零迭代一致）。
- **`nkv < num_split`**：`ns_eff=min(num_split,nkv)` 无空段；`num_logical_A` 相应缩小。
- **`P·ns_eff > nkv`（末段越界）**：页号 `pg ≥ nkv` 时钳位 `pgc=min(pg,nkv−1)` 只影响**装载地址**（读重复页），正确性由列掩码（`pg·page+j ≥ N_kv ≥ real_kv` ⟹ 全掩码）保证。
- **写回 extent 匹配**：`T.copy` src/dst 双显式切片同 extent（`TRAP-T-copy-region-semantics` ②）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：
  - **Phase A**：`num_logical_A = num_q × H × B × ns_eff = 1 × H × B × ns_eff`。
    Manifest：`64 / 64 / 32 / 32`（见 §5.2 表）。
  - **Phase B**：`num_logical_B = num_q × H × B = 1 × H × B`。Manifest：`16 / 16 / 8 / 8`。
- **② 物理核数依据**（**实查，非假设**）：本设计期在目标设备执行
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()         # 实查返回: 20
  NPUUtils.get().get_aivector_core_num()  # 实查返回: 40  (= 20 × 2)
  ```
  → **物理核数 = 20 AI Core（Ascend910_9362 单 die；AIV = 40）**。本算子内核为 **Cube/混合（Phase A）+ 纯 Vector（Phase B）**
  ——统一按**核粒度**（每核一任务）分核，直接使用 `get_aicore_num()` 返回值 **20**（不翻倍；
  `CONST-aicore-910_9362`：9362 单 die = 20 AIC / 40 AIV）。
- **③ 规模判定与分核方案**：
  - **Phase B**：`num_logical_B ∈ {8, 16} ≤ 20` → **逻辑核数 ≤ 物理核数 → 无需适配**（一核一任务，无非整数倍负载不均）。
    方案 = 非持久一维 grid，每核 1 任务。
  - **Phase A**：`num_logical_A ∈ {32, 64} > 20` → **超出物理核数**（1.6–3.2 波次）。
    **不能**用标准要素③的「极大规模核内 `T.serial` 串行（固定 20 核）」——该形态要求
    `persistent grid + 核内 T.serial + gemm`，落在 `TRAP-DEVMODE-PERSIST-GEMM`（Developer + persistent + gemm 运行时崩溃）
    **风险区**。故取标准要素③的**「无法对齐兜底」**分支：**保持非持久一维 grid，内核数 = 逻辑核数**（运行时按波次调度），
    并**显式记录残留负载不均与临界路径惩罚**：
    - `64` 逻辑核 / `20` 物理核 = **3.2 波次**；`64 = 3×20 + 4` → 末波仅 `4/20` 核忙碌 → 负载不均；
      临界路径相对完美均衡（`64/20` 波）的惩罚 ≈ `4/3.2 = +25%`。
    - `32 / 20 = 1.6` 波次；末波 `12/20` 忙碌 → 惩罚 ≈ `2/1.6 = +25%`。
    - **缓解**：§1.6.3 的 **`ns_eff` 归一化**（按 shape 取 `min(num_split, nkv, max(1, floor(20/(num_q·B·H))))`）可
      把 `num_logical_A` 压到 ≤20 的整数倍；Stage 4 实验裁决决定是否启用（含 `ns_eff=1` 退化为单遍）。
    - **不引入核内 `T.serial` 多任务循环**（`TRAP-DEVMODE-PERSIST-GEMM`）。
  - **跨 kernel**：Phase A → Phase B 为**两次独立 launch**（host closure，stream 序），各自按下发逻辑核调度；
    无核内跨任务依赖。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Phase A 任务 `(mid,hid,bz,sid)` | **block 级并行** | `T.Kernel(num_logical_A, is_npu=True)` | 每核 1 任务（非持久） |
| Phase A 段内 KV 维 | **tile 级迭代 + 流水** | `T.Pipelined(P, num_stages=2)` | **静态界**；段内在线状态跨 tile 携带依赖；流水重叠 K/V 装载与计算 |
| Phase A 元素级（掩码/行操作） | **向量化** | v-prefix fragment 操作 / `T.Parallel(block_M)` | §1.6.2 已向量化；无逐元素标量循环 |
| Phase B 任务 `(mid,hid,bz)` | **block 级并行** | `T.Kernel(num_logical_B, is_npu=True)` | 每核 1 任务 |
| Phase B 段维 | **编译期展开的小循环** | `T.serial(ns_eff)`（`ns_eff` 编译期常量） | 段间两阶段归约（max→sum→加权），非逐元素热点 |
| Phase B 元素级 | **向量化** | 行向量 v-op（`vmax/vsub/vexp/vadd/vln/vmul/vcast/vbrc`） | §1.6.2 B1–B5 |

### 6.2 循环伪代码

```python
# Phase A
with T.Kernel(num_logical_A, is_npu=True) as (cid, _):
    sid, hid, bz, mid = decode_A(cid)                  # §0.6 R1
    for k in T.Pipelined(P, num_stages=2):             # 静态界（§0.6 R4）
        pgc = T.min(sid * P + k, nkv - 1)              # 钳位页号
        phys = block_table[bz, pgc]
        T.copy(K[phys*block_N:(phys+1)*block_N, hid, :], K_shared)
        T.gemm(Q_shared, K_shared, acc_s, initC=True, b_transpose=True)     # Cube
        mask_and_online_softmax(acc_s, ...)            # Vector: 有限哨兵 + 同域恒等守卫
        T.vcast(acc_s, acc_s_cast, round_mode="rint")
        T.copy(V[phys*block_N:(phys+1)*block_N, hid, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, initC=False)                    # Cube
    segment_epilogue(acc_o, logsum, m_ref, ...)        # Vector: o_sid + l_sid -> 工作区

# Phase B
with T.Kernel(num_logical_B, is_npu=True) as (cid, _):
    mid, hid, bz = decode_B(cid)
    combine_segments(glse, Output_partial, Output, mid, hid, bz)            # Vector
```

### 6.3 流水线优化

- `T.Pipelined(P, num_stages=2)`：K/V 物理页装载（MTE2）与 Cube/Vector 计算重叠。
- **stage 数**：`num_stages=2` 使 L1 的 K/V 双槽 ≈ 2×`(page·D·2)`；smoke（page=128,D=128）= 128KB ✓；
  page=256,D=128 = 256KB ✓（<512KB）。KV 迭代次数少（manifest `P=1`），流水收益有限；`num_stages` 为保留调优旋钮（Stage 4）。
- **段数与流水的关系**：减小 `ns_eff` 增大 `P` 可显著摊薄每段前导/epilogue 开销并提升流水深度——是 §1.6.3 实验裁决的主杠杆之一。

### 6.4 尾块 / 掩码处理

- **OOB 掩码（非因果，通用）**：列 `j` 的全局 KV 位置为 `pg·block_N + j`；有效当且仅当 `< real_kv`。
  用**有限大数哨兵 `-1e38`**（`PL-1.11`）经 `T.vselect`（**值替换，NaN 免疫**）置换无效列：
  `P = exp(−1e38 − m) → 0`（精确零），规避 `−inf − (−inf) = NaN`。掩码链：`col_lim = real_kv − pg·block_N` →
  `T.vbrc(col_lim, lim_mat)`（**标量先广播为矩阵**——`TRAP-vcmp-scalar-operand`：运行时标量直接作 `T.vcmp` 第二操作数
  令 npuir 编译器段错误）→ `T.vcmp(colmat, lim_mat, vis, "lt")` → `T.vselect(vis, acc_s, sent_buf, acc_s)`。
- **页号越界（新引入，split 特有）**：末段 `pg ≥ nkv`。**装载钳位** `pgc = min(pg, nkv−1)`；正确性由列掩码保证
  （`pg·page + j ≥ nkv·page = N_kv ≥ real_kv` ⟹ 该块**整块全掩码**）。
- **全掩码块恒等稳定化**（守卫与哨兵**同域**；§0.6 R4）：`all_masked = (m_cur ≤ -1e30)`（与哨兵 `-1e38` 同域）→
  `m_cur_safe = all_masked ? 0 : m_cur`；`rs = all_masked ? 1 : exp(m_prev − m_cur_safe)`（**恒等**，保状态）；
  `P = exp(S̃ − m_cur_safe) = exp(−1e38) = 0`；`s_sum = 0`。**`rs` 取 1 而非 0**：超界块恒为**段尾**块（有效列是前缀），
  其后不再有有效块；`rs=0` 会把前面已累积的 `logsum/acc_o` 清零 → 输出错误归零。
- **段级 LSE 哨兵**：`l_sid = (logsum==0) ? LSE_SENT : ln(logsum)+m_ref`。`LSE_SENT = -30000.0`（**必须 fp16 可表示**：
  `glse` 以输入 dtype 存储，`-1e38 → fp16 -inf` 会导致全空段 `exp(−inf−(−inf))=NaN`）。`-30000` 下：
  全空段 `m=lse_max=-30000`、`Σexp(0)=ns_eff`、`LSE=ln(ns_eff)-30000`、`w=exp(-ln ns_eff)=1/ns_eff`、`o=Σ0=0` ✓ 无 NaN。
- **causal 掩码**（源支持，manifest 全 `False`）：源 split 的 causal 分支带「Handle causal split case」占位注释（L151），
  属**源实现未闭环**项。本设计按**全局位置**给出正确泛化：`q_pos = mid·block_M + i`，`kv_pos = pg·block_N + j`，
  可见当 `q_pos ≥ kv_pos`（与 no-split 同口径；有效列同为前缀，段尾块走同一恒等路径）。列入 §9 风险 R-7。
## 7. 同步策略

### 7.1 同步模式

**模式**: **自动同步**（Developer 模式）；跨 kernel 由 **host closure 的 stream 序**保证。

### 7.2 同步点说明

- **Phase A 内**：`T.gemm`（Cube）与 v-prefix 向量链（Vector）之间的跨引擎依赖、`T.Pipelined` 的 stage 间依赖
  由编译器（auto CV-split + pipe 调度）**自动插同步**，无需手动 `sync_block_set/wait`。
- **Phase A → Phase B**：两次 `@tilelang.jit` kernel 在同一 stream 顺序 launch ⟹ Phase B 读取
  `glse`/`Output_partial` 时 Phase A 已全部完成（**等价于源同 prim_func 两次 launch 的隐式全局屏障**）。
- **Phase B 内**：`T.serial(ns_eff)` 段循环内为行向量 v-op，数据依赖由编译器在 UB 内自动排序。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| A: Cube `T.gemm` → Vector softmax | 自动（编译器 CV split 插同步） | score/P 跨引擎依赖 |
| A: Vector P → Cube `T.gemm`（gemm2） | 自动 | P 跨引擎依赖 |
| A: KV 循环 stage 间（`T.Pipelined`） | 自动（多缓冲 + 同步） | K/V 装载与计算重叠 |
| A: `block_table[bz,pgc]` 间接寻址的 `T.copy` 源地址（`pgc=min(sid·P+k, nkv−1)`，`k` 为流水循环变量） | 自动（多缓冲 + 同步）；地址计算属该 stage 的标量前导 | 同 no-split（R-2 已实测可用：`num_stages=1/2` 均正确） |
| A → B 跨 kernel | **host closure stream 序** | 段级合并需全部段结果就绪（源两次 launch 的屏障语义） |
| B: 段循环内 v-op | 自动 | UB 内数据依赖 |

> 本设计**不使用** `T.Scope("Cube")/T.Scope("Vector")`、`T.rs("PIPE_*")`、`T.sync_block_set/wait`（Expert 专属）。
> 母函数 Phase A 是标准 Developer MixCV（no-split 已验证形态）。

### 7.3 pass_configs 配置

```python
# 不设 pass_configs（npuir PassConfigKey 无 TL_ENABLE_FAST_MATH；照搬源配置即 AttributeError）
```

> 不设 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` / `NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`——那两项是
> **Expert** 形态为保护手动 Scope/workspace 而设（`PL-1.16-expert-dualscope-bypass`）；本设计为 Developer，保留自动多缓冲。

### 7.4 CV 融合设计（Phase A）

Phase A 是 **Cube + Vector 融合**（Developer 自动 CV split），段内每块的引擎依赖链：

```
(Cube) gemm1: acc_s = Q̃ Kᵀ            [K_shared 就绪 → L0C]
  │ S/P 跨引擎依赖（自动同步）
(Vector) mask → reduce_max→m_cur → rs=exp(m_prev−m_cur_safe) + 全掩码恒等守卫 → P=exp(S̃−m_cur_safe)
         → reduce_sum → logsum/acc_o 重缩放 → m_ref 跟踪 → **末尾 m_prev←m_cur**
  │ P 跨引擎依赖（自动同步）
(Cube) gemm2: acc_o += cast(P) V       [acc_s_cast + V_shared 就绪 → L0C]
```

- **状态顺序**：`m_prev` init `-inf`；每块**先算新 `m_cur`，再用旧 `m_prev` 算 `rs`，最后写回 `m_prev←m_cur`**。
- **侧归属**：`T.gemm` 归 Cube（L1 操作数 → L0A/L0B → L0C fp32）；mask/softmax/cast/normalize 归 Vector（UB）。
  `acc_s`（P）是跨引擎中间量，驻 **UB**（**不经 `store_fixpipe` GM 往返**——相对 Expert 形态的优势）。
- **边界**：若 Stage 3 实测自动 CV split 失效，回退路径 = 按 `tilelang-mixcv-skill` 的 Expert 双 Scope 重设计
  （§9 R-1）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（**优先移植源仓参考实现/测试基准**），**不得复刻 §0.6 的 NPU 算法**。
> 唯一依据 = `examples/TileOPs/tests/ops/test_mha_decode_paged.py::MhaDecodePagedTest.ref_program`
> （paged 重组逻辑 KV + `F.scaled_dot_product_attention`）——**与 split/no-split 无关**（输出相同），保证独立性。

```python
import math
import torch
import torch.nn.functional as F

def golden_mha_decode_paged(q, k, v, real_seqlen_kv, block_table, page_size):
    """Paged MHA decode reference (移植源仓 ref_program；独立于 NPU 算法).

    q: (B, S_q, H, D); k/v: (N_kv, H, D) fp16/bf16;
    real_seqlen_kv: (B,) int32; block_table: (B, num_pages) int32.
    返回 (B, S_q, H, D)。paged 重组到逻辑布局后走 SDPA（无分段/无哨兵/无恒等守卫）。
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

**可比性/比对方法**（移植源测试的 `_maxdiff_cosine_compare`）：`max_diff < atol` 且
`cosine_similarity(reshape(B,-1), dim=-1).min() > 0.99`。

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-2 | 1e-2 |

> 源测试用 `atol=1e-3` + `cos>0.99`；本设计沿用 `atol=1e-2/rtol=1e-2`（与 no-split 一致）+ `cos>0.99` 双判。
> **段级 `glse`/`Output_partial` 的输入 dtype 往返**是 split 相对 no-split 的额外精度点：**D-1 实测
> OPT-S 往返 `max_abs=3.815e-04`（源公式）/`3.662e-04`（全采纳集），违反率 0**，远小于 atol=1e-2
> （甚至低于源测试的 atol=1e-3）。**§1.6.1 OPT 项全部由 `verify_equiv.py` 机器验证违反率 0**。

### 8.3 L0 门槛测试计划（blocking）

| 用例 | 配置（来自 manifest + 边界） | 目的 | 通过判据 |
|------|----------------------------|------|----------|
| L0-1 | `(1,16,1,512,128,page=128)` fp16 smoke（`nkv=4,ns_eff=4,P=1`） | 主路径（4 段各 1 页，B*H=16 → Phase A 64 核 / Phase B 16 核） | max_diff<atol + cos>0.99 |
| L0-2 | `(2,8,1,1024,64,page=256)` fp16（`nkv=4`） | 多 batch + page 256 + D=64 | 同上 |
| L0-3 | `(1,8,1,1024,64,page=256)` fp16 | page 256、B*H=8（Phase A 32 核） | 同上 |
| L0-4 | `(1,8,1,512,64,page=256)` fp16（`nkv=2 < num_split=4`） | **`ns_eff=min(num_split,nkv)=2`**（截断段数） | 同上 |
| L0-5（契约） | 输出 shape/dtype == q；`block_table` **打乱页序**（非恒等） | paged 寻址 + 两 kernel 编排正确性 | shape/dtype 一致 + 精度门 |
| L0-6（工作区契约） | 上述任一配置：`glse`/`Output_partial` 由 host 预分配传入 | **两 kernel 工作区接口**（Phase A 写 / Phase B 读） | 精度门（工作区未被 combine 循环外读取） |
| **L1-1（边界·段尾整块无效）** | `(1,8,1,512,64,page=256)` fp16 + `real_seqlen_kv=200`（页 0 部分掩码、**页 1 全掩码；段 sid≥1 全空**) | **段尾全掩码块恒等稳定化 + 空段 LSE 哨兵** | max_diff<atol + cos>0.99（输出 == golden，**非 0**）；无 NaN/±inf |
| **L1-2（边界·零逻辑长度）** | 同上 + `real_seqlen_kv=0`（**全段全空**） | 全空段 → `LSE_SENT` → 合并输出 0 | `max_diff == 0`；**无 NaN/±inf**（源零迭代行为） |
| **L1-3（边界·末段越界钳位）** | `(1,8,1,512,64,page=256)` + `nkv=2, num_split=4`（`ns_eff=2,P=1`，无越界）；另测 `nkv=5,num_split=2`（`P=3`，末段 `pg=5,6` 越界钳位） | **页号钳位 + 全掩码** | max_diff<atol + cos>0.99；无越界读（无 MTE fault） |
| **L1-4（边界·打乱页表 + 部分长度）** | `(1,8,1,512,64,page=256)` + `real_seqlen_kv=300` + shuffle 页表 | paged 间接寻址 + 掩码联合 | 同上 |

> 分层套件（L1/L2/Boundary、bf16 覆盖、causal、空行）交由 `tilelang-op-develop` 按 §9 风险清单枚举。
> **L1-1/L1-2 为 OPT-S 缺陷的专用拦截用例**（源 `rs=0` 守卫在静态界下会错误归零 / `-inf` LSE 会 NaN）。
> **L0-6 为两 kernel 工作区接口的专用用例。**

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 | 类别 | 影响 | 缓解 / 依据 |
|---|------|------|------|------------|
| R-1 | **两 kernel + host closure** 相对源「同 prim_func 两次 launch」的形态差异：若 Stage 3 发现工厂返回 Python closure 不被目标 harness 接受（要求单一 `@tilelang.jit` kernel 返回物） | 实施/接口 | 实现受阻 | 先例：`tilelang_sparse_mla_bwd.py`（3 kernel host 编排）、`_gqa_prefill_fwd_kernel.py`（内层工厂返回 closure）；**退路** = §1.6.3 备选 #2（单 kernel 核内合并，天然单返回物） |
| R-2 | **Developer + 非持久 1D grid + `T.gemm`** 的支持边界 | 编译/运行时 | 若不支持则需切 Expert | 先例 `flash_attn_npuir_dev.py`；**不引入 persistent**（`TRAP-DEVMODE-PERSIST-GEMM`）；Stage 3 首编验证 |
| R-3 | **核内 GM 标量索引**（`real_seqlen_kv[bz]`、`block_table[bz,pgc]`） | 前端能力 | 装载地址构造失败/静默错读 | **no-split 已实测可用**（`repro/TRAP-paged-dynamic-index-sliceload.py`，`num_stages=1/2` 均正确）——本设计沿用 |
| R-4 | **GEMM M/N 分形下限**（M=16 pad、`S_q=1`）为**未文档化假设** | 前端能力 | 编译失败/结果错 | 依据 `docs/.../T.gemm.md §2.2.2` + `examples/gemm` + prefill `tmc=max(16,ceil16)`；`block_M=16`；**plan B** = §1.6.3 备选或把 `block_M` 提到 32/64 |
| R-5 | `T.copy` 跨步/间接 GM 区域装载 | 语义陷阱 | 静默乱值 | **必须 slice 形态**（`TRAP-load-nd2nz-strided`）；`load_nd2nz`/base+size 禁用 |
| R-6 | **`glse` 以输入 dtype 存储**（段级 LSE 经 fp16 量化往返） | 精度 | 合并权重相对误差 ~0.4% | 源算法固有（`glse` 声明为输入 dtype）；**D-1 实测 OPT-S 往返 max_abs 3.82e-04 / 违反率 0**；若 L0 在 atol=1e-3 下不达标，启用 §1.6.3 备选 #2（核内合并，无 glse 往返） |
| R-7 | **causal split** 源实现未闭环（源 L151 占位注释） | 语义 | causal 场景正确性 | 本设计按**全局位置**给出正确泛化（§6.4）；manifest 全 `False`；列 L2/风险项，Stage 3 覆盖 causal 用例 |
| R-8 | **Phase A 逻辑核超发**（64/20 = 3.2 波次，末波 4/20） | 性能 | 残留负载不均（临界路径 +25%） | §5.5「无法对齐兜底」记录；`ns_eff` 归一化（§1.6.3）为 Stage 4 旋钮；不超过 `TRAP-DEVMODE-PERSIST-GEMM` 风险区 |
| R-9 | **`mutates_args=()` 与工作区写入不一致**：脚手架 `custom_op(..., mutates_args=())` 把 `glse`/`Output_partial` 列为输入却被 Phase A 写入 | 接口契约 | torch.compile 图语义不一致 | **继承自 GPU 源算子**（源亦如此），非本设计引入；若 harness 严格要求，退路 = §1.6.3 备选 #2（不使用传入工作区，不 mutate） |
| R-10 | **发射项常数（0.5µs/op）9362 未重验** + 单 op 边际代价随 block_size 变化 | 性能估计 | roofline 不确定（[4,47]µs） | 标注「未实测假设」（`CONST-vector-launch-overhead` 9362 注记 + `CONST-vector-op-cost-910_9362`）；§1.6.3 实验裁决 |
| R-11 | padding 行（1..15）在 gemm/softmax 中的垃圾值 | 数值 | 若行隔离失效则污染 | 行级 reduce/重缩放/写回天然隔离 + `Q_shared` 清零；L0 用全零/随机输入双验 |
| R-12 | `T.vcmp` 运行时标量操作数段错误 | 编译器 | segfault | 标量先 `T.vbrc` 广播为矩阵（`repro/TRAP-vcmp-scalar-operand.py`） |
| R-13 | `T.reduce(clear=False)` 静默错误 | 数值 | 静默错值 | 全部 `clear=True`（`T.reduce.md §2.3`） |
| R-14 | `bf16` 输出经 `vcast(rint)` 舍回 + `glse` bf16 往返 | 数值 | ~2 ulp | 沿用 fp32 中转链；门禁 atol=1e-2 覆盖（manifest 仅 fp16，bf16 为附加覆盖） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block 过大 | 编译失败 | `block_M=16`、`block_N=page_size` 已控；按 `CONST-capacity-910B2C` 复算（Phase A ≈91KB） |
| L0C 溢出 | `bm×bn>32768` | segfault | `16×256=4096` ✓ |
| MTE DDR fault | `T.copy` src/dst extent 不匹配 | 越界写坏相邻 GM | 双显式同 extent 切片（`TRAP-T-copy-region-semantics`） |
| 装载乱值 | 非 slice 形态跨步装载 | 静默错值 | 强制 slice 形态 |
| NaN | `-inf-(-inf)` / 全空段 `-inf` LSE / 静态界新引入的段尾全掩码块 | NaN / ±inf / 错误输出 | 有限哨兵 `-1e38` + 同域守卫 `rs=1` + `LSE_SENT=-30000`（fp16 可表示） |
| 编译器段错误 | `T.vcmp(tensor, runtime_scalar)` | exit 139 | 标量先 `T.vbrc` 广播为矩阵（R-12） |
| 段数空段 | `nkv < num_split` | 空段产生无效 LSE | `ns_eff = min(num_split, nkv)` |

### 9.3 特殊场景处理

- **非整除**：`S_q<block_M`（pad+隔离）、`N_kv%page_size≠0` 或 `real_kv<N_kv`（全局位置掩码，段尾整块全掩码 → 同域守卫恒等）、`P·ns_eff>nkv`（页号钳位）。
- **极小 shape**：`N_kv<page_size`（`nkv=1`，掩码）；`nkv<num_split`（`ns_eff` 截断）；`real_kv=0`（全段全空 → 输出 0）。
- **混合精度**：fp16 主；bf16 经 fp32 中转链 + `glse` bf16 往返。
- **causal**（manifest 外）：全局位置算术掩码（§6.4；源侧未闭环，本设计补全）。
- **超大 `H·B`**（超出 manifest）：Phase A/B 逻辑核随之增长，超发波次增加 → `ns_eff` 归一化或 split/no-split 分派（wrapper 层）。

### 9.4 NPU 能力缺口（候选，未确证）

以下为设计期识别、**尚未确证**的能力不确定项（Stage 3 首验后若确证为缺口，按 `AGENTS.md`「Compiler Capability Gap Rule」登记到
`.agents/evolution/capability-gaps.md`）：
1. **一个 prim_func 内多 `T.Kernel`（多 launch）支持**——本设计以「两 kernel + host closure」绕开，但该能力是否存在/是否支持在本工具链**未确证**（`tilelang/jit/jit_npu.py` 的 `_launch` 为单次 `rtKernelLaunchWithFlagV2`；未见多 launch 支持）。
2. **核内 GM 标量索引**——**已由 no-split 实测确证可用**（非缺口，此处仅记录）。
3. **GEMM M/N=16 分形下限**——**未文档化假设**（无确证证据）。
本设计**不预先登记为缺口**（无确证证据，避免投机条目）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mha_decode_paged/_mha_decode_split_kernel/
├── DESIGN.md                 # 本设计文档
├── verify_equiv.py           # D-1 等价性机器验证脚本（§1.6.1；4 采纳项 + MASK-CHECK 全 PASS）
├── REVIEW.md                 # Stage 2 检视（后续阶段产出）
├── RETROSPECTIVE.md          # Stage 1 复盘（本 Agent 产出）
└── _mha_decode_split_kernel.py   # 待实现（Stage 3）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档（含 §0 迁移分析、§1.6 算法调研与优化） |
| `verify_equiv.py` | ✅ 已完成 | D-1 等价验证（OPT-1/2/3/OPT-S 全部 EQUIV_PASS + MASK-CHECK PASS） |
| `_mha_decode_split_kernel.py` | ⬜ 待实现 | 算子实现（Stage 3；两个 Developer MixCV kernel + host closure） |

### 10.3 命名规范

- 项目目录：`mha_decode_paged`（snake_case）
- 算子目录：`_mha_decode_split_kernel`（源 JIT 工厂函数名）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ D-1 等价验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1，移植源仓 ref_program）
3. ⬜ Phase A（`_split_partial`）实现 + L0 精度（先单测 Phase A 写出的 `glse`/`Output_partial` 与 golden 段划分对比）
4. ⬜ Phase B（`_combine`）实现 + host closure 编排 + L0 端到端精度
5. ⬜ §8.3 边界用例（L1-1..L1-4）与两 kernel 工作区契约（L0-6）
