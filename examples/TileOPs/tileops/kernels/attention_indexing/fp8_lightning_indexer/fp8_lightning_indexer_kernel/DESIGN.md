# _fp8_lightning_indexer_kernel 算子设计文档（迁移）

> **项目**：`fp8_lightning_indexer` ｜ **算子目录**：`examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_fp8_lightning_indexer_kernel`（`/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py` L19–L149）重设计为 NPU（`target="npuir"`）实现。同文件的 `clean_logits_`（L152–L184）是**另一个迁移单元**（`.migration_state.json` 中 `clean_logits_` status=pending），本设计**只覆盖主 indexer kernel**，但显式定义与它的接口契约（§0.2 / §5.4）。
>
> **本版最关键的三项决策**：
>
> 1. **FP8 不可用（能力缺口，非设计选择）**：源工厂的 `dtype = T.float8_e4m3fn` + fp8 `T.gemm` + Op 层设备端 fp8 量化，在 npuir 上三步同时不可行（详见 §0.5 条目 1 与 §3.5.1）。设计改用 **bf16 / fp16 index 载体 + fp32 累加 + fp32 向量 epilogue**，把 fp8 的“反量化缩放”语义以可选 fp32 `index_k_scale` 乘子承接。已按项目根 `AGENTS.md` Compiler Capability Gap Rule 登记 **CG-2026-0022**（含逐层证据），fp8 workload 标为受限（§9.1）。
> 2. **算法级优化 E1（scale 提出归约，已机器验证）**：源公式 `Σ_h (relu(s_h)·w_h)·scale` → `scale · Σ_h (relu(s_h)·w_h)`。fp64 有理数精确算术证明**代数恒等**（200 组随机有理数 diff=0，`verify_equiv.py` L1）；fp32 契约域 13/13 `EQUIV_PASS`（§1.6.1）。这是本算子矢量发射项的主要削减点（每次 S 元素少 1 次乘 → 约 −1.2 ms，见 §1.6.0 估算）。
> 3. **结构对齐 NPU 同族先例**：开发者模式非持久 1D grid + Cube `T.gemm` + **L0C→UB 直拷**（`T.copy(acc_frag, acc_ub)`，省掉 Expert 的 GM workspace 往返，`examples/indexer/indexer_fwd.py` 已验证形态），Vector 侧 `T.vrelu` → `T.vmul(w 行因子)` → 逐行 `T.reduce_sum` → `T.vbrc(scale)+T.vmul`。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)）：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py`

- 主 kernel 工厂：`_fp8_lightning_indexer_kernel(batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits=True)`（L19–L149，`@functools.lru_cache(maxsize=32)`），内部 `@tilelang.jit(pass_configs={TL_ENABLE_FAST_MATH: True})` 的 `_fp8_lightning_indexer_func(block_N=256, num_stages=3, threads=512, block_Q=None)`（L31–L147）。
- 后处理 kernel（**非本单元**）：`clean_logits_(threads=512)`（L152–L184）。
- 集成 wrapper（NPU 侧接口契约）：`examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer.py`（`npub::fp8_lightning_indexer_wrapped_kernel`；调用 `_fp8_lightning_indexer_kernel(batch, seq_len, heads, index_dim, seq_len_kv, kv_group)(block_N, num_stages, block_Q)(IndexQ.view(...), IndexK, IndexKScale, Logits, Weights, CuSeqLenKS, CuSeqLenKE)`，K9 已去掉 `threads`，K5 `supported_archs=None`，K8 去 autotune）。

### 0.1 源算子语义（做什么）

**数学语义**（在 `[cu_seqlen_ks[s], cu_seqlen_ke[s])` 界定的可见区间内，对 index 缓存打分）：

$$
\text{Logits}[b,s,t,g] \;=\; \sum_{h=0}^{H_p-1} \Big( \mathrm{relu}\big(\text{Sq}[b,s,\,g\!\cdot\!H_p+h,\,t]\big)\cdot W[s,\,g\!\cdot\!H_p+h] \Big)\cdot \text{Scale}[b,t,g],
\qquad
\text{Sq}[b,s,h,t]=\sum_{d=0}^{D-1} Q[b,s,h,d]\cdot K[b,t,g,d]
$$

其中 $H_p = \lceil H/G \rceil$（源码 `heads_per_group = heads // kv_group`，L64），$g=\lfloor h/H_p\rfloor$。

- **规约语义**：唯一归约轴是 **head 轴**（在组内 $H_p$ 上求和，`T.reduce_sum(s_reshaped, logits, dim=-2, clear=True)` L140）；**kv 轴不归约**——每个 `(s,t,g)` 输出独立，kv 方向是纯 streaming（扫一遍写一遍）。归约累加顺序 = 每组内 $h$ 升序（`s_reshaped[bn,bq,h,g]` 的 `dim=-2`）；累加 dtype = **fp32**（`accum_dtype = T.float32`，L43）。
- **dtype 语义**：IndexQ/IndexK 输入 `T.float8_e4m3fn`（L42）；`weights` / `index_k_scale` / `logits` 全 `fp32`；`cu_seqlen_ks/ke` `int32`。中间得分 `s` 为 fp32 fragment（`alloc_fragment(..., accum_dtype)` L68），**relu/乘权/乘 scale 全部在 fp32 域**，只在最终写 `Logits` 时已是 fp32（输出即 fp32，无末端截断）。
- **窗口/边界语义**：主 kernel 只负责 `[cu\_k\_s\_min, cu\_k\_e\_max)`（对组内 block_Q 行取 `min(ks)` / `max(min(ke,Skv))` 的并集，L82–L91）的**块对齐超集**；该超集内落在各行 `[ks,ke)` 之外的值由独立 `clean_logits_` 覆写为 `-inf`。即：**主 kernel 不做掩码，掩码是后处理单元的契约**。
- **极值语义**：`relu` 消负；`scale` 来自 `absmax/448`（`ops/fp8_lightning_indexer.py` L152–L160）恒正，但外部 `index_k_scale` 路径（`tl_quant_forward`）未约束符号；`-inf` 仅由 `clean_logits_` 写入。NaN/Inf 传播无特殊处理。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（源仓 `examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py`（`ref_program`），见 §0.7）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `IndexQ` | `[batch, seq_len * heads, index_dim]`（由 `[batch, seq_len, heads, index_dim]` view 而来） | float8_e4m3fn | 查询（kernel 内按 `seq_len_i*heads` 起始，`[block_Q*heads, index_dim]` tile） |
| `IndexK` | `[batch, seq_len_kv, kv_group, index_dim]` | float8_e4m3fn | 键缓存（按 `[block_N, kv_group, index_dim]` tile） |
| `IndexKScale` | `[batch, seq_len_kv, kv_group]` | float32 | 键反量化 scale（可选，Op 层决定） |
| `Weights` | `[seq_len, heads]` | float32 | head 权重 |
| `CuSeqLenKS` | `[seq_len]` | int32 | 每 query 可见窗口下界 |
| `CuSeqLenKE` | `[seq_len]` | int32 | 每 query 可见窗口上界 |
| `Logits`（**输出**） | `[batch, seq_len, seq_len_kv, kv_group]` | float32 | 打分输出（迁移必需字段；`Logits` 为显式入参、非 out_idx） |

`heads % kv_group == 0`（Op 层校验，`ops/fp8_lightning_indexer.py` L96）。`Logits` 输出 shape 与源 `source_output_shape` 一致：`(batch, seq_len, seq_len_kv, kv_group)`。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句都归入某步骤；括号内为源码行）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | `cu_k_s_min = min_bq min(CuSeqLenKS[...], seq_len_kv)`；`cu_k_e_max = max_bq min(CuSeqLenKE[...], seq_len_kv)`（L88–L91，两次 `T.serial(block_Q)` + `T.min/T.max`） | CuSeqLenKS/KE | 两个 int32 标量（`T.alloc_var`） | 窗口并集（调度参数，非公式项） |
| 2 | `T.copy(IndexQ[b_i, seq_len_i*heads, 0], index_q_shared)`（L93）；`T.copy(Weights[seq_len_i, 0], weights)`（L94） | IndexQ / Weights | `index_q_shared[bq*H, D]`、`weights[bq, H]` | Q tile / 权重的每核一次装载 |
| 3 | kv 主循环 `T.Pipelined(ceildiv(cu_k_e_max-cu_k_s_min, block_N), num_stages)`（L96–L144） | — | — | kv streaming |
| 3a | `T.copy(IndexKScale[b_i, k0, 0], index_k_scale_fragment)`（L100） | IndexKScale | `[block_N, G]` fp32 | 反量化 scale tile |
| 3b | **G==1**：`T.copy(IndexK[b_i, k0:k0+bn, 0, 0:D], index_k_group_shared)`（L104–L106）+ `T.gemm(K, Q, s, transpose_B=True, clear_accum=True)`（L107–L114） | K tile, Q tile | `s[bn, bq*H]` fp32 | `Sq` 矩阵乘（M=bn, N=bq·H, K=D） |
| 3b' | **G>1**：`g` 串行，`T.Parallel` gather 出 `index_k_group_shared[bn,D]` 与 `index_q_group_shared[bq*hpg,D]`（L117–L121），逐组 `T.gemm → s_tmp`（L122–L129），散写回 `s_reshaped[...,g]`（L130–L132） | 同 3b | `s_tmp[bn, bq*hpg]` | `Sq`（分组） |
| 3c | `s_reshaped[bn,bq,h,g] = max(s,0) * weights[bq, g*hpg+h] * index_k_scale_fragment[bn,g]`（L134–L138，`T.Parallel`） | `s`, weights, scale | `s_reshaped[bn,bq,hpg,G]` fp32 | `relu(Sq)·W·Scale`（**scale 在本式内**） |
| 3d | `T.reduce_sum(s_reshaped, logits, dim=-2, clear=True)`（L140） | `s_reshaped` | `logits[bn, bq, G]` | `Σ_h`（组内 head 归约） |
| 3e | `Logits[b_i, seq_len_i+bq_i, k0+bn_i, g] = logits[bn_i,bq_i,g]`（L142–L144，`T.Parallel`） | `logits` | GM `Logits` | 输出写回 |

**数据流与内存访问模式**（源硬件视角，CUDA SM90）：

```
GM[IndexQ] --T.copy--> SMEM[index_q_shared] --┐
GM[IndexK] --T.copy--> SMEM[index_k_group_shared] --T.gemm(TensorCore)--> frag s (L0C/reg, fp32)
GM[IndexKScale] --T.copy--> frag[bn,G] --┐
frag s + weights + scale --T.Parallel(relu·mul·mul)--> s_reshaped (frag)
s_reshaped --T.reduce_sum(dim=-2)--> logits (frag) --T.Parallel--> GM[Logits]
```

- `T.Pipelined(num_stages)` 在 CUDA 上是 **cp.async 软件流水**（K/scale 的 GM→SMEM 与 TensorCore gemm 重叠）；`index_q_shared` / `weights` 被 hoist 到 kv 循环外（每核只搬一次）。
- 每个 `(block_Q, batch)` block 把整个窗口的所有 `block_N` tile 走完；`Logits` 为块对齐超集直接写（无掩码）。

**循环与并行结构**：`T.Kernel(ceildiv(seq_len, block_Q), batch, threads=threads) as (bx, by)` —— **2D grid**（bx=q-block, by=batch）× **512 线程**；`block_Q = max(1, 128//heads)`（L37–L38，heads=32 时 =4；heads≥65 时 =1）；kv 方向 `T.Pipelined`（动态 trip count = `ceildiv(cu_k_e_max-cu_k_s_min, block_N)`）；block 内元素级并行靠 `T.Parallel` / `T.reduce_sum`（warp 级）。

**host 侧逻辑**：本单元 kernel 无 host 侧预处理/后处理；`IndexQ.view(batch, seq_len*heads, index_dim)`（wrapped L198）为纯 view（不改内容）。wrapped 在 `clean_logits=True` 时追加调用 `clean_logits_`（**另一个迁移单元**）。Op 层 `per_custom_dims_cast_to_fp8`（`ops/fp8_lightning_indexer.py` L152–L160）在**调用 kernel 前**做设备端 fp8 量化（cast 输入、算 scale）——属 fp8 路径，见 §0.5。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **TensorCore GEMM 承载 `Sq`**（`T.gemm(..., transpose_B=True)`，`GemmWarpPolicy.FullCol`） | 把 O(S·Skv·H·D) 的 MAC 密集段交给矩阵单元 | MMA 指令 + `transpose_B` 免显式转置 | SM90 4th/5th-gen TensorCore（**fp8 mma**） | **强相关**（fp8 mma 是 SM90 特性；NPU 无 fp8 mma → 见 §0.5） |
| O2 | **block_Q 行合并**（`block_Q = max(1,128//heads)`，把多 query 行拼进 GEMM 的 N 维 `bq*H`） | 提高 gemm N 维整除性/利用率，摊薄 K 装载 | 一个 q-block 摊一次 K/scale 装载 | 与硬件弱相关（tiling 层） | **可移植**（NPU tiling 变量，见 §5） |
| O3 | **Q / Weights hoist**（L93–L94 提到 kv 循环外） | 每核只搬一次 Q 与 weights，消除每 kv-tile 重读 | 循环不变量外提 | 无（纯算法/调度层） | **可移植** |
| O4 | **软件流水 `T.Pipelined(num_stages)`**（L96，K/scale 预取重叠 gemm） | 隐藏 GM→SMEM 搬运延迟 | cp.async 多级缓冲 | CUDA 软件流水 + `__syncthreads` 屏障 | **等价替换**（NPU `T.Pipelined` = MTE2 双缓冲，机制不同） |
| O5 | **`_MIN_PIPELINED_BLOCK_Q` 死锁守卫**（L13–L15, L40–L41：`block_Q < 2 → num_stages = 0`） | 规避 `block_Q==1` 时部分 warp 永不达流水屏障的死锁 | CUDA warp 级流水屏障语义 | CUDA warp/barrier 语义 | **舍弃/重设计**（NPU 无 warp 级流水屏障，见 §0.5 条目 4） |
| O6 | **fp32 fragment 累加 + 组内 head 归约**（`s`/`s_reshaped` fp32，`reduce_sum`） | 归约精度（避免 fp8/fp16 累加溢出/误差） | register/L0C fp32 累加 | 通用 | **可移植**（NPU fp32 accum + `T.reduce_sum` fp32） |
| O7 | **scale 在归约内乘**（L136–L138 把 `index_k_scale_fragment` 乘进每个 head 项） | 源实现写法，便于与 `weights` 同式表达 | 逐元素 mul | 无 | **可移植但可优化**（→ E1 提出归约，§1.6.1） |
| O8 | **SMEM tiling + 块对齐超集写 + 独立 clean pass** | 主 kernel 不判边界、写满 tile；掩码交给后处理 | shared memory tile + 独立 kernel | shared memory / 独立 grid | **等价替换**（NPU UB/L1 tile + `clean_logits_` 仍是独立迁移单元） |
| O9 | **`TL_ENABLE_FAST_MATH`**（L29） | 放宽数学函数精度换吞吐 | 编译 pass | CUDA fast-math | **舍弃/重评**（NPU 无对应；epilogue 只有 relu/mul，无超越函数，收益≈0） |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 1 **计算语义**（relu(Σq·k)·w 求和 + 窗口） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1）；§8.1 golden 以此为准 |
| 2 **fp8 数据类型 + fp8 GEMM（O1）** | 算法/优化 | SM90 fp8 TensorCore mma | **无** | **舍弃 + 重新设计（载体替换）** | npuir 前端无 `T.float8_e4m3fn`、codegen `Cannot convert type e4m3_float8`、`T.gemm`/`T.vcast` dtype 矩阵无 fp8、torch_npu 设备端 fp8 cast 失败（错码 561103）→ **CG-2026-0022**。改 bf16/fp16 载体（`T.gemm` §2.2.1 支持）+ fp32 累加；权重 `w_h` 乘子与 `index_k_scale` 乘子承接 fp8 反量化语义。→ §0.6 R1 |
| 3 **TensorCore GEMM（矩阵乘骨架）** | 算法 | SM90 MMA | **有**（Cube `T.gemm`→`hivm.hir.mmadL1`） | **等价替换** | Cube GEMM 承接 `Sq`（`docs/Tilelang.language/线性代数操作/T.gemm.md`；`examples/indexer/indexer_fwd.py` L68 同形先例）；分形 16 对齐由框架处理 |
| 4 **CUDA 软件流水 `T.Pipelined`（O4）** | 优化 | cp.async + warp 屏障 | **有**（语义不同） | **等价替换** | NPU `T.Pipelined(num_stages)` = MTE2/Cube 双缓冲调度，block 级（`examples/indexer/indexer_fwd.py` L64 同形）；`num_stages` 保留为缓冲深度旋钮 |
| 5 **`_MIN_PIPELINED_BLOCK_Q` 死锁守卫（O5）** | 优化 | warp 级流水屏障 | **无对应物** | **舍弃** | NPU 无 warp/线程级软件流水屏障（`T.Kernel(is_npu=True)` 无 `threads` 语义，`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3；NPU 先例 `indexer_fwd.py` 无此守卫）→ 该守卫是 CUDA 专属，NPU 上 `block_Q` 不受 ≥2 约束 |
| 6 **`T.get_thread_binding()` + `threads` 线程索引** | 实现 | threadIdx/warp | **无对应物** | **重新设计** | 见 §0.6 R2。`clean_logits_` 的 `tx = T.get_thread_binding()` / `T.serial(ceildiv(cu_k_s, threads))` 无 NPU 对应；本单元不涉及，但接口契约需一致（§5.4） |
| 7 **Q/Weights hoist（O3）** | 优化 | 无 | 有 | **保留** | 循环不变量外提，NPU 同样适用（`indexer_fwd.py` L59–L62 同形） |
| 8 **fp32 fragment 累加 + head 归约（O6）** | 算法 | 通用 | 有 | **保留** | `T.reduce_sum` fp32 支持（`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1）；`T.reduce` 支持 `dims=[0]` 垂直归约（`indexer_fwd.py` L74–L80 先例） |
| 9 **scale 在归约内乘（O7）** | 优化 | 无 | 有 | **重新设计（数学等价化简）** | → E1：`scale·Σ_h(relu·w)`（§1.6.1） |
| 10 **SMEM tiling / 块对齐超集写（O8）** | 优化 | shared memory | 有（UB/L1） | **等价替换** | UB tile + L0C/L1（§4）；块对齐超集写语义保留（掩码仍属 `clean_logits_` 迁移单元） |
| 11 **`TL_ENABLE_FAST_MATH`（O9）** | 优化 | CUDA fast-math pass | **无对应** | **舍弃** | 无超越函数调用（epilogue 仅 relu+mul），收益≈0；`PassConfigKey` 在 npuir 无意义 |
| 12 **`GemmWarpPolicy.FullCol`** | 优化 | CUDA warp 划分策略 | **无对应**（NPU gemm 无 warp 概念） | **舍弃** | `T.gemm` npuir 签名无 `policy`（`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1） |

> 依据：`migration-analysis.md` §5.3 能力映射 + 本仓 `examples/` 同族先例 + 逐条 `docs/Tilelang.language/` 条款。舍弃项理由均见上表“NPU 对应方案/依据”列。

### 0.6 NPU 算法重设计

对 §0.5 中判定为「重新设计」的条目逐项给出 NPU 新算法。

**重设计项 R1: fp8 载体 → bf16/fp16 载体（承接 O1/O2/O6 的性能意图）**

- **源方案**：IndexQ/IndexK 为 `float8_e4m3fn`，用 SM90 fp8 TensorCore 做 `Sq` 矩阵乘（1 字节/元素，fp32 accum），Op 层 `per_custom_dims_cast_to_fp8` 做设备端 absmax 量化并产出 `index_k_scale`。
- **NPU 新算法**：index 载体改为 **bf16（默认，匹配 manifest workload）/ fp16（可选）**，`T.gemm(K_tile, Q_tile, acc, b_transpose=True)` 走 Cube `hivm.hir.mmadL1`，**fp32 累加**；`index_k_scale` 退化为**可选 fp32 乘子**（`has_scale=False` 时编译期跳过该乘，即 identity）。fp8 量化本体不下沉到 kernel（设备端 cast 不可用，CG-2026-0022）。
- **语义保持论证**：公式骨架（`relu(Σ_d q·k)·w` 再对 head 求和、窗口由 `clean_logits_` 覆写）**完全不变**；变化只在 index 载体的数值精度（fp8→bf16），不改变算子的数学定义。`has_scale=False` 的 identity 特化经 `verify_equiv.py` E3 证明与「乘 1.0」逐位等价（`fp64_fwd_ulp=0`，corr_diff=0）。**边界语义核对**：`-inf` 仍由 `clean_logits_` 写入（本单元不写）；NaN/Inf 在 relu/mul 下传播与源一致（E1 角点用例 `nonfinite_eq=True`）。

**重设计项 R2: 线程索引（`get_thread_binding`/`threads`）→ block 级 UB 向量化**

- **源方案**：CUDA 用 `threads=512` + 线程索引把元素映射到 lane（`clean_logits_` 里 `tx = T.get_thread_binding()` + `T.serial(ceildiv(n, threads))` + `if idx < n`）。
- **NPU 新算法**：NPU 无 threadIdx（`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3：`is_npu=True` 只到 2D block，无线程绑定）；把线程展开的逐元素工作改为 **UB tile 上的 v-prefix / `T.Parallel` 整 buffer 操作**。本单元（主 indexer）本就用 `T.Parallel` + `T.reduce_sum`，无需 thread 索引，天然可移植；§5.4 明确 `clean_logits_` 迁移单元的同一重表达契约（block_N tile + `T.vmin/T.vmax` 得到窗口边界，逐 tile `T.vselect`/边界 v-prefix 写 `-inf`）。
- **语义保持论证**：线程-lane 与 UB-tile 的映射不同但语义等价（同一集合的逐元素映射，无跨 lane 依赖）；`clean_logits_` 的 `if idx < n` 由 UB 尾块切片（零起点 + `T.min` extent）等价承接（PL-1.22）。

**重设计项 R3: 软件流水死锁守卫（O5）→ 去掉守卫、以缓冲深度表达**

- **源方案**：`block_Q < 2` 时强制 `num_stages = 0`，规避 CUDA 软件流水在 `block_Q==1` 下「部分 warp 永不达屏障」的死锁。
- **NPU 新算法**：NPU `T.Pipelined` 是 block 级 MTE2 双缓冲（无 warp 屏障），不存在该死锁形态；`_MIN_PIPELINED_BLOCK_Q` 守卫**去掉**，`num_stages` 仅作缓冲深度旋钮（受 UB 预算约束）。`block_Q` 由 tiling 决定（§5），不受 ≥2 约束。
- **语义保持论证**：流水调度不改变计算结果（仅影响预取时机）；数值路径与顺序无关。**风险**：该结论基于「NPU 无 warp 屏障」的结构判断（`T.Kernel.md` §2.3 + `indexer_fwd.py` 无守卫先例），非文档化直述——列入 §9.1 风险点，Stage 3 首编验证。

**其余 (§0.5 保留/等价替换/舍弃项)** 无需重设计，但在 §1–§6 按 NPU 能力落地（Cube 分形、UB/L1 预算、`T.Pipelined` 缓冲、Developer 自动同步）。

### 0.7 标杆实现

- **源算子路径标注**（golden 来源）：`/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py`；源仓测试 golden：`examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py`（`FP8LightningIndexerTest.ref_program`）（L67–L94，einsum → relu → 分组加权求和 → 窗口 `masked_fill(-inf)`；比较器 `_validate_tensor_match` 用 **correlation**，容差 1e-3，L42–L65）。
- **迁移 golden 策略**：§8.1 直接移植该 `ref_program`（改 device 解析），**不复刻** §0.6 的 NPU 算法（保证验证独立性）。
- **同族 NPU 先例**：`examples/indexer/indexer_fwd.py`（Developer 非持久 indexer，结构与本设计同形）；`examples/deepseek_v32/fp8_lighting_indexer.py`（Expert 两 Scope indexer，`dtype="float16"` 佐证 NPU 用 fp16 载体）。

---

## 1. 概述

### 1.1 算子名称

`_fp8_lightning_indexer_kernel`（Lightning Indexer 前向打分 kernel，迁移自 GPU TileOPs `FP8LightningIndexerOp`）。

### 1.2 功能描述

在每条 query 的可见窗口 `[cu_seqlen_ks[s], cu_seqlen_ke[s])` 内，对 index 键缓存逐 `(kv_group, kv 位置)` 计算「组内各 head 的 `relu(q·k)·权重` 之和，再乘键反量化 scale」的 fp32 打分，输出 `[batch, seq_len, seq_len_kv, kv_group]`。

### 1.3 数学公式

$$
\text{Logits}[b,s,t,g]=\sum_{h=0}^{H_p-1}\Big(\mathrm{relu}\big(\textstyle\sum_{d=0}^{D-1}Q[b,s,\,gH_p{+}h,d]\cdot K[b,t,g,d]\big)\cdot W[s,\,gH_p{+}h]\Big)\cdot \text{Scale}[b,t,g]
$$

$H_p = H/G$。**§1.6.1 优化后（采纳 E1，scale 提出归约）**：

$$
\text{Logits}[b,s,t,g]=\text{Scale}[b,t,g]\cdot\sum_{h=0}^{H_p-1}\mathrm{relu}\big(\textstyle\sum_{d}Q\cdot K\big)\cdot W[s,\,gH_p{+}h]
$$

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6；与源算法结构差异逐条注明）：

1. **Grid**：非持久 **1D** grid `num_q_blocks × batch`（源为 2D grid + 512 线程；差异来源 §0.6 R2 —— NPU 无线程绑定，q-block 与 batch 拼成一维逻辑任务）。每个逻辑 block 处理 `block_Q` 条 query 的**全部可见 kv 窗口**。
2. **窗口并集（调度参数）**：block 内对 `block_Q` 条 query 求 `cu_k_s_min`/`cu_k_e_max`（源 L88–L91 同义；NPU 用编译期展开的 `block_Q` 次标量 `T.min/T.max`，属 block 级索引计算，§1.6.2 #3）。
3. **hoist 装载**：Q tile `[block_Q*H, D]` 与 Weights `[block_Q*H, 1]` 每核只搬一次（承接源 O3）。
4. **kv streaming + Cube GEMM**：对窗口内每个 `block_N` 键块，`T.gemm(K_tile, Q_tile, acc)`（`b_transpose=True`）得 `acc[block_Q*H_p, block_N]` fp32（承接源 O1/O2/O6，载体改 bf16/fp16，§0.6 R1）。**核内布局**：acc 以 `[block_Q*H_p, block_N]` 方阵布局（M=head 行、N=kv 列），kv 轴连续。
5. **Vector epilogue（fp32 域）**：`acc → UB`（`T.copy`，L0C→UB 直拷，免 GM workspace 往返，见 §4.4）；`T.vrelu`；`T.vmul(acc_ub, w_ub)`（`w_ub [block_Q*H_p, 1]` 行因子广播）；逐 query 行 `T.reduce_sum(acc_ub[q*H_p:(q+1)*H_p, :], out_ub[q:q+1, :], dim=0, clear=True)` 完成组内 head 归约；`T.vbrc(scale_ub, brc_ub)` + `T.vmul(out_ub, brc_ub, out_ub)` 施加反量化 scale（**E1：scale 在归约外乘**，与源 O7 差异，§1.6.1）。
6. **写回**：`out_ub [block_Q, block_N]` → `Logits[b, seq_len_i+bq, k0+kv, g]`。**块对齐超集写**（承接源 O8）：窗口外/超集内的值不由本 kernel 掩码，由独立 `clean_logits_` 迁移单元覆写 `-inf`。
7. **G>1**：对每组 `g` 依次执行 4–6（Q 取 `[g*H_p,(g+1)*H_p)` 行、K 取 `g` 组），结果写 `Logits[...,g]`（源 L115–L132 的分组 gemm 语义）。

与源算法的差异来源：① 载体 fp8→bf16/fp16（§0.6 R1）；② 2D grid+threads → 1D grid（§0.6 R2）；③ 去流水守卫（§0.6 R3）；④ scale 提出归约（§1.6.1 E1）。

### 1.5 数据流图

```
GM[IndexQ] ─copy─┐                        (每核一次, hoist)
GM[Weights]─copy─┼─> UB/L1 [q_ub, w_ub] ─┐
GM[IndexK] ─copy─> UB/L1 [k_ub] ──> Cube T.gemm ──> L0C acc (fp32) ─copy─> UB acc_ub
GM[IndexKScale]─copy─> UB [scale_ub] ────────────────────────────────────────┐
                                                                             v
                       acc_ub ─vrelu─> ·w(row) ─reduce_sum(h)─> out_ub ─vbrc+v×scale─> out_ub
                                                                             │
                                                              copy ─> GM[Logits] (块对齐超集)
[另一迁移单元] clean_logits_: window 外 GM[Logits] := -inf
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（1.6.0 调研四问）选算法族；再数学等价优化公式（1.6.1）；再把计算交给向量单元（1.6.2）；最后定向量化轴与核内布局（1.6.3）。本节结论是 §3.1 与 §6 的输入。分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

算子类别：**矩阵 + 窗口 + 规约 + 多步融合**（GEMM → 逐元素 → 归约 → 写回），按 `algorithm-research.md` §2 走**完整调研**。
调研范围：`algorithm-candidates.md` 命中行 **ALG-attention / ALG-reduction / ALG-elementwise / ALG-gemv**（本算子无直接命中条目，取同族候选）；本仓 `examples/indexer/`、`examples/deepseek_v32/`、`pattern-library/attention.md`（PL-1.9/PL-1.11）、`pattern-library/elementwise.md`（PL-1.17/PL-1.27/PL-1.29）；结构分析（§0.3 的归约轴 = head、kv 轴 streaming）。

**R1 等价化简公式候选**：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|--------|
| 0 | **基线：源算法** | `Σ_h (relu(Sq_h)·w_h)·s` | — | — | ✅（基线） |
| 1 | **E1 scale 提出归约**（采纳，§1.6.1） | `s·Σ_h(relu(Sq_h)·w_h)` | 代数恒等（s 对 h 为常量；relu 作用于未缩放 Sq）——fp64 精确、200 例有理数 diff=0 | 每次 S 元素少 1 次乘（−≈2.0e9 elem-mul） | ✅ |
| 2 | **E2 relu 与 ·w 合并为单链**（无数学变化，仅 op 组织） | `vmul(vrelu(acc), w)` | 逐位等价 | 不省 op（2 op 不可融合），但避免中间 `s` 物化回 GM | ✅（作为实现形态） |
| 3 | **E3 identity-scale 特化**（采纳） | `has_scale=False` 时去掉 `·s` | 与 `·1.0` 逐位等价（verify_equiv E3：ulp=0） | 无 scale 路径省 1 `vbrc`+1 `vmul` | ✅ |
| 4 | **scale 折进 K（gemm 前）** | `relu(s·Sq)` 等价，需 s>0 | **非普遍等价**：s<0 时 `relu(s·Sq)=s·relu(Sq)` 不成立（verify_equiv 诊断 corr_diff=0.997） | — | ❌（负向，诊断见 §1.6.1） |
| 5 | **把 head 归约折进 Cube**（weighted-sum 用矩阵乘） | `Σ_h relu(Sq_h)·w_h` 视为 `relu(Sq)·w` 的 matvec | **不成立**：relu 在求和之前、非线性，无法折进线性算子 | — | ❌（数学不可行） |
| 6 | **融合 clean_logits 进主 kernel** | 主 kernel 内做窗口掩码、省后处理 pass | 语义可行 | 省后处理 pass，但把掩码成本并入主循环且需覆盖窗口外区域 | ❌（**范围/契约理由**：`clean_logits_` 是独立迁移单元，融合会与该单元重复并改变其契约；且源已把掩码外置，见 §0.1） |
| 7 | **两遍/物化 S 矩阵**（先算全 S 落 GM，再归约） | 类似 `deepseek_v32/fp8_lighting_indexer.py` 的 workspace 写法 | 等价 | **负向**：S 全量 = 2.1e9 elem × 2B = 4.3GB GM 往返 | ❌（访存爆炸，见 R3） |

**R2 在线算法**：**无（不需要）**。结构依据：唯一归约轴是**组内 head 轴**，完全在 tile 内（`block_Q*H_p` 行），**kv 轴无归约**——每个 `(s,t,g)` 输出独立、kv 方向是 embarrasingly-parallel 的块流式写。因此不存在「跨块 running 统计量」需要 online 重缩放（对比 online-softmax 因为 softmax 分母跨 kv 归约）。源算法也未用 online 手段（§0.4 无；R2 判据 `algorithm-research.md` §3 R2 命中「逐元素/无跨元素依赖 → 单遍即天然在线」）。结论：**单遍 streaming 即为最优扫描形态**，无 online 变体可采。

**R3 复杂度对比**（四口径；代表 workload = `batch1/seq8192/heads32/index_dim64/seq_len_kv8192/kv_group1`，bf16 载体 eb=2，口径 = HBM 无 L2 复用、block_Q=8/block_N=64）：

| 候选 | FLOPs | 访存量 (Bytes, HBM 口径) | 扫描遍数 | 中间缓冲峰值 | 可并行度/跨核 |
|------|-------|--------------------------|---------|-------------|--------------|
| **0 基线（源）** | MAC `2·B·S·Skv·G·H·D` = **275 GFLOP**；epilogue mul ≈ `2·B·S·Skv·G·H`(relu+w)+`B·S·Skv·G·H`(×scale) | Q `B·S·H·D·eb`=33.5MB(1×) + K **重读** `B·Skv·G·D·eb·⌈S/bq⌉`=**1.07GB**(×1024) + out `B·S·Skv·G·4`=**268MB** + weights 1MB + scale 32KB ≈ **1.37GB**（L2 全驻留时 ≈303MB） | Q 1 / K **⌈S/bq⌉** / Logits 1 | `s_reshaped` tile `bq·H·bn·G·4`=64KB(L0C)+fragment | 无跨核归约；逻辑核 `B·⌈S/bq⌉` |
| **1 E1（采纳）** | MAC 同基线；scale 乘子 `B·S·Skv·G`(1×而不是 H×) | 同基线 | 同基线 | 同基线 | 同基线 |
| **2 E2（实现形态）** | 同 1 | 同 1 | 同 1 | 同 1 | 同 1 |
| **3 E3（identity-scale）** | MAC 同基线；**无** scale 乘子 | 同基线 | 同基线 | 同基线 | 同基线 |
| **7 物化 S（否决）** | 同基线 | + `S 全量`写 4.3GB + 读 4.3GB = **+8.6GB** | +2（S 写/读） | `B·S·Skv·H·eb`=4.3GB(GM) | 同基线 |

口径说明：FLOPs 含 GEMM MAC×2 与 epilogue 逐元素乘（relu 按 1 cmp，未折算超越函数）；访存量按 HBM 无 L2 复用（K 被每个 q-block 重读，最坏口径），**L2 驻留时 K 读 ≈1MB 一次**（K 全量仅 1.05MB，远小于 Ascend L2）——两口径并列，均入 §1.6.0 估算。**不含** `clean_logits_` 单元的流量（另单元）。

**R4 硬件亲和性评估**：

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水/融合 | 结论 |
|------|-------------|---------|-----------|---------|-----------|------|
| 0/1/2/3 | Cube GEMM（MAC 密集 → `T.gemm` madL1）+ Vector epilogue（逐元素/归约 → v-prefix）。**匹配佳** | acc tile `bq·H_p×bn` fp32 ≤ 32768 elem（L0C 128KB，`CONST-cube-gemm-l0c-traffic-floor`）；UB 预算 §4.5 ✓ | `bn=64` fp32 尾轴 256B ✓；`D=64` 分形 ✓ | kv trip count 动态（窗口并集，运行时值）——`T.Pipelined` 动态 trip 支持（`indexer_fwd.py` L64 先例） | Cube/Vector 在**同一 AI Core 上顺序执行**（Developer 自动 CV-split）；`T.Pipelined` 双缓冲 K/scale | ✅ **选定** |
| 4 scale 折进 K | Vector 在 gemm 前多一遍 K tile 乘 | — | — | — | — | ❌ 非普遍等价（s<0 破坏） |
| 5 head 归约折进 Cube | — | — | — | — | — | ❌ 数学不可行（relu 非线性） |
| 6 融合 clean_logits | 需覆盖窗口外 `[0,ks)∪[ke,Skv)`，但主循环只覆盖并集窗口 → 需扩到全 `Skv` 或另写掩码路径 | — | — | — | 掩码成本并入主循环 | ❌ 范围/契约（独立迁移单元）+ 收益不明确 |
| 7 物化 S | — | — | — | — | — | ❌ 访存 +8.6GB（R3）→ 违反带宽量级（`CONST-mte2-degradation` 混合读写 ~1.26TB/s 地板下 ≈+6.8ms） |

**调研结论**：选定**候选 1+2+3 的组合**（`scale·Σ_h relu(Sq_h)·w_h`，scale 可编译期省略），结构 = **Cube GEMM + Vector fp32 epilogue（L0C→UB 直拷）+ 块流式无在线**。与基线的结构差异仅两点：① `scale` 从「归约内逐 head 乘」移到「归约外一次乘」（E1，代数恒等）；② 载体 fp8→bf16/fp16（§0.6 R1，非算法层）。**源优化意图承接**：O1/O2/O6 → NPU Cube tiling + fp32 accum（本结论）；O3 → 保留 hoist；O4 → `T.Pipelined` 等价替换；O5 → 舍弃（§0.6 R3）；O7 → E1 重设计；O8 → UB tile + 独立 clean 单元；O9 → 舍弃（无超越函数）。负向候选（4/5/6/7）依据见 R1/R4 表。

**设计期估算（D-2）**：代表 workload（bf16, block_Q=8, block_N=64）：

- **流量项**：HBM 口径 ≈ 1.37GB（K 重读为主）/ 核→GM 带宽。设备带宽常数（`CONST-ascend910-9362-profile`：单 die 核→GM ≈1.06 TB/s、HBM ≈1.64 TB/s，2026-10-13）**绑定 9362**，本机为 **Ascend910B3**（`torch.npu.get_device_name(0)`）——「未实测假设 + 估算依据」：取 ~1.06 TB/s ⇒ **≈1.29ms**；若 K 全驻 L2（K=1.05MB ≪ L2）则 HBM 口径 ≈303MB ⇒ **≈0.29ms**。
- **发射项**：S 矩阵元素 `B·S·Skv·H = 2.147e9`；epilogue ≈ 3 遍（vrelu / vmul·w / reduce）+ 少量 scale 乘 ⇒ ≈6.4e9 elem-op。逐元素速率取 `CONST-vector-op-cost-910_9362`（9.15µs/16m fp16 → ≈1.75e12 elem-op/s；**同为 9362 实测，910B3 未实测**——未实测假设）⇒ **≈3.7ms**。（E1 前基线多 1 遍 ×scale ≈ +1.2ms ⇒ E1 是主要削减项。）
- **容量项**：Developer 走 **L0C→UB 直拷**（`T.copy(acc_frag, acc_ub)`，`docs/Tilelang.language/内存操作/T.copy.md` §2.1/§3「L0C fragment↔UB 局部拷贝」），**不引入 GM 往返**（`CONST-store-fixpipe-gm-only` 的约束适用于 `store_fixpipe`/Expert；Developer 通用 copy 路径不经 fixpipe）⇒ 容量项 ≈ 0。
- **估算下界** = max(1.29 / 0.29, 3.7, 0) ≈ **3.7 ms（发射项主导）**，引用 `CONST-vector-op-cost-910_9362` + `CONST-ascend910-9362-profile`（均为 **9362 绑定、910B3 未重验**，标注未实测假设）。
  估算下界: ≈3.7 ms（流量 0.29–1.29 ms + 发射 3.7 ms + 容量 ≈0，引用 CONST-vector-op-cost-910_9362 / CONST-ascend910-9362-profile / CONST-store-fixpipe-gm-only）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法。

| # | 优化项 | 原式（源） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|-----------|-----------|-----------|---------|
| **E1** | **scale 提出 head 归约** | `L = Σ_h (relu(Sq_h)·w_h)·s` | `L = s·Σ_h (relu(Sq_h)·w_h)` | **代数恒等**：`s` 对归约轴 `h` 为常量，分配律 `Σ_h a_h·s = s·Σ_h a_h`；relu 作用于**未缩放的** `Sq`，故对任意（含负/零）有限 `s` 成立。**机器验证**：`verify_equiv.py` L1 精确有理数算术 200 组随机有理输入 `max_diff=0`；L2 fp32 契约域 13/13 `EQUIV_PASS`（随机 fp64 前向误差 0.85 ulp、`corr_diff` 6.7e-16、`viol_1e-3=0`、`nonfinite_eq=True`） | 每次 S 元素少 1 次乘：`B·S·Skv·G·(H−1)` ≈ **2.0e9 elem-mul**（S 元素 2.147e9、overhead 修正后 ≈−1.2ms，占发射项 ≈32%）。**NPU 先例**：`deepseek_v32/fp8_lighting_indexer.py` L143/L146（`vmul(·w)` 在 `reduce` 前、`vmul(·k_s)` 在 `reduce` 后）即为该形态 |
| **E3** | **identity-scale 编译期特化** | `L = Σ_h(relu·w)·s`（s 恒 1） | `has_scale=False` 时 `L = Σ_h(relu·w)`（去掉 `·s`） | 与 `s≡1.0` 逐位等价（`x·1.0 ≡ x`）；`verify_equiv.py` E3 `fp64_fwd_ulp=0, corr_diff=0` | 无外部 scale 的 bf16 路径省 1 `vbrc` + 1 `vmul`/tile（`PL-1.28-const-scalar-param-fold` 型特化，类比可迁移） |
| — | **relu 融合**（不构成公式变化） | `max(s,0)` | `T.vrelu(acc_ub, acc_ub)` | 逐位等价（relu 是逐元素精确运算），仅从标量 `T.max` 换向量指令 | 与 §1.6.2 #1 同项 |
| **否决** | scale 折进 K（gemm 前 `relu(Sq·s)`） | — | — | **非普遍等价**：`relu(x·s) = s·relu(x)` 仅当 `s≥0`；`s<0` 时破坏（`verify_equiv.py` 诊断：`corr_diff=0.997`）；且会把 1 次归约后乘换成 `bn·D` 次 K 元素乘（更贵） | 否决（负向，机器诊断留痕） |
| **否决** | head 归约折进 Cube（`Σ_h relu(Sq_h)·w_h` 视为线性 matvec） | — | — | **数学不可行**：`relu` 在归约之前、非线性，无法折进线性矩阵乘 | 否决 |

**优化结论**：采纳 **2 项**（E1 + E3）。**优化后公式**（供 §3.1 使用）：

```
acc[qh, n] = Σ_d K[n,d]·Q[qh,d]                     # Cube gemm, fp32 accum
out[q, n]  = Scale[n] · Σ_{h} relu(acc[q*H_p+h, n]) · W[q, h]     # Vector epilogue
             (has_scale=False 时省略 Scale 因子)     # E3
```

**D-1 机器验证结果表**（`verify_equiv.py`，torch CPU；脚本存于算子目录）：

| 项 | 用例 | fp64 前向误差（ULP@conditioning） | corr_diff | 违反率(1e-3) | 非有限一致 | 结论 |
|----|------|----------------------------------|-----------|--------------|-----------|------|
| E1 | 随机（scale∈[1e-4,448]） | 0.848 | 6.66e-16 | 0 | True | **EQUIV_PASS** |
| E1 | scale=0.0 / −0.0 / 1e-4 / 1.0 / −1.0 / 448 / 1e4 | 0 / 0 / 0.755 / 0 / 0 / 0.636 / 0.889 | ≤8.9e-16 | 0 | True | **EQUIV_PASS** |
| E1 | scores/weights 含 ±inf/NaN（scale=1e-4/1/448） | ≤0.755 | ≤8.9e-16 | 0 | True | **EQUIV_PASS** |
| E1 | subnormal / −0.0 scores | 0.848 | 7.77e-16 | 0 | True | **EQUIV_PASS** |
| E3 | identity-scale skip | 0 | 0 | 0 | True | **EQUIV_PASS** |
| 诊断(非门禁) | 折进 K，scale>0 | 0 | 7.8e-16 | 0 | True | DIAGNOSTIC（数学等价但不省 op） |
| 诊断(非门禁) | 折进 K，scale<0 | 3.85e+15 | 0.997 | 1.0 | True | DIAGNOSTIC（**证明否决必要**） |
| **L1 精确** | 200 组随机有理数（`fractions.Fraction`） | **代数 diff = 0** | — | — | — | **EQUIV_PASS** |

`ALL_GATE_ITEMS EQUIV_PASS`（L1 exact PASS，L2 13/13）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§1.4/§3.3）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态（源） | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------------|-------------|---------|------------------------------|
| 1 | relu(Sq) | `s_reshaped[...] = T.max(s, 0)`（`T.Parallel`，源 L136） | `T.vrelu(acc_ub, acc_ub)` | ✅ | — （API：`docs/Tilelang.language/数学操作/T.vrelu.md`，fp16/fp32 支持；bf16 ✗） |
| 2 | ×weights（行因子） | `T.Parallel` 逐元素 `* weights[bq,g*hpg+h]`（源 L137） | `T.vmul(acc_ub, w_ub, acc_ub)`，`w_ub [bq*H_p,1]` 行广播 | ✅ | — （行因子 `[M,N]*[M,1]` 是文档化合法形，`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2；先例 `indexer_fwd.py` L72） |
| 3 | 组内 head 归约 | `T.reduce_sum(s_reshaped, logits, dim=-2, clear=True)`（源 L140） | 逐 query `T.reduce_sum(acc_ub[q*Hp:(q+1)*Hp,:], out_ub[q:q+1,:], dim=0, clear=True)`（垂直扫描） | ✅ | — （`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2 合法形 `(M,N,K)→(1,N,K)`；`reduce_sum` fp16/fp32 支持，**bf16 ✗**；先例 `indexer_fwd.py` L74–L80） |
| 4 | ×scale（列因子 `[1,bn]`） | `s_reshaped[...] * index_k_scale_fragment[bn,g]`（源 L138，逐元素） | `T.vbrc(scale_ub, brc_ub)` 把 `[1,bn]` 展成 `[bq,bn]` + `T.vmul(out_ub, brc_ub, out_ub)`（E3 无 scale 时整段编译期省略） | ✅ | — （**不可用 `[M,N]*[1,N]` 直乘**：`T.vmul` 文档化仅四形、不含列因子 —— queue **VP-2026-0087** / `T.vmul.md` §2.2.2；故先 `T.vbrc` 展开，`T.vbrc` 支持 `(1,N)→(M,N)`，`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2） |
| 5 | Q tile 装载 | `T.copy(IndexQ[b_i, seq_len_i*heads, 0], index_q_shared)`（源 L93） | `T.copy(IndexQ[b, seq_len_i*H, 0], q_ub, size=[bq*H, D])` | ✅ | — （`docs/Tilelang.language/内存操作/T.copy.md` §2.1 索引+size 形） |
| 6 | Weights 装载 | `T.copy(Weights[seq_len_i, 0], weights)`（源 L94） | `T.copy(Weights[seq_len_i, 0], w_ub, size=[bq, H])` | ✅ | — |
| 7 | **窗口并集 `cu_k_s_min`/`cu_k_e_max`** | 源 `T.serial(block_Q)`×2 + `T.alloc_var` + `T.min/T.max`（L82–L91） | 无逐元素等价 API | ❌ | **block 级索引/调度计算**（非逐元素热点）：只需 `block_Q`(=8) 次标量比较得到 2 个调度标量，无法也无需向量化（`T.reduce_min/max` 需先物化 UB 数组，收益为负）。编译期展开 `block_Q` 次即可 |
| 8 | 输出写回 | `T.Parallel` 逐元素（源 L142–L144） | `T.copy(out_ub, Logits[b, seq_len_i, k0], size=[bq, bn])`（整 buffer 拷贝，隐式向量化；`G>1` 时按 `g` 切片写） | ✅ | — （`docs/Tilelang.language/内存操作/T.copy.md` §2.1） |
| 9 | 逻辑任务解码（`cid → (by, bx)`） | 源 2D grid 自带 | 无 | ❌ | **block 级任务映射**：每核 1 次整数除/模，非逐元素热点（`T.Kernel.md` §2.4 例 1 同形） |
| 10 | kv 循环 trip count（`ceildiv(cu_k_e_max−cu_k_s_min, bn)`） | 源 `T.Pipelined` 动态上界 | 无 | ❌ | **依赖动态 shape 的边界处理**：上界是运行时值（窗口并集），不可静态化（padding 到全 `Skv` 会把 gemm 工作量放大到最坏窗口）；`T.Pipelined` 支持动态 trip（`indexer_fwd.py` L64） |
| 11 | `G>1` 组 gather（源 L117–L121 的 `T.Parallel` 带下标变换） | `T.Parallel` gather（源） | 无逐元素等价（`T.Parallel` **禁止下标变换**，`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3）；改用 block 级 `T.copy` 切片（K 组：`IndexK[b,t0:t0+bn,g,0:D]` 跨步切片；Q 组：逐 query `T.copy` 连续 `H_p` 行） | ❌（部分） | **block 级拷贝替代逐元素 gather**：`T.Parallel` 下标不可变换（API 限制，附文档），故用 `T.copy` 切片 + `T.serial(bq)` 承载。**仅 G>1 触发**（无 manifest workload），Stage 3 需验证跨步切片 |

**向量化结论**：逐元素计算点（#1/#2/#3/#4/#5/#6/#8）**全部向量化**，替代 API 均有 `docs/`/`examples/` 佐证；保留 5 处非逐元素标量/循环（#7 block 索引、#9 block 任务映射、#10 动态边界、#11 API 限制下的 block 级拷贝），逐项理由见上表，无「实现简单」类理由。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

I/O layout 是**契约**（`IndexQ[B,S,H,D]` / `IndexK[B,Skv,G,D]` / `Logits[B,S,Skv,G]`），核内布局是**设计变量**。算子非 batch 维 ≥2（S / H / D / Skv），逐候选枚举：

| # | 核内布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|-------------|---------|------------|--------------|------|
| 1 | **acc `[bq·H_p, bn]`（M=head 行、N=kv 列），kv 轴连续** | **`bn`（kv）** —— 最内连续轴，fp32 ×4 | 无（gemm 原生写 `[M,N]`；`T.copy(acc_frag, acc_ub)` 保形） | 基线；`bn=64` fp32 尾轴 256B 对齐 ✓；head 归约走**垂直扫描**（每 query 行内 `dims=[0]` reduce） | ✅ **主选** |
| 2 | acc `[bn, bq·H_p]`（源 `s` 朝向，N=head） | `bq·H_p`（head） | 无 | 归约轴变最内维（水平归约 lane 映射到 head）——`H_p=32` fp32 尾轴 128B ✓，但 reduce 语义要求跨 lane；且 KV 列变非连续轴，`Logits` 写回需重排 | ❌：与 `indexer_fwd.py`（L51/L68 已验证 `[bq·H, bn]`）相反，无收益、增加写回重排；`T.reduce` 合法形为垂直（`dims` 维置 1） |
| 3 | host permute `IndexK→[B,D,Skv,G]` | 任意 | host 重排（~百 µs，`PL-1.3-host-permute`） | repack 代价 ≫ 收益，且**违反 `ascend-constraints.md` §4**（host 不得改输入内容/物理排布） | ❌ |
| 4 | 核内 `T.transpose` 重排 K tile 到 `[D, bn]` | D | 核内融合转置（~µs 级，`PL-1.1-transpose-chain`） | 无收益：`T.gemm` 已支持 `b_transpose=True`（`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1），且 K tile 的 `D` 轴（64）本就是分形对齐的连续尾轴，无需重排 | ❌（转置纯属多余） |

**布局决策结论**：选定 **#1**：核内布局 acc = `[bq·H_p, bn]`（M=head 行、N=kv 列），**向量化轴 = `bn`（kv）**，repack 路径 = **无**（cube 原生 `[M,N]` 写 + 保形 L0C→UB 拷贝）；head 归约用 `bn` 连续行上的**垂直扫描**（逐 query `T.reduce_sum(..., dims=[0])`）。该结论同步落入 §3.3 伪代码与 §6 循环结构：acc/out 的缓冲形状与向量维与本节一致（`bn` 为最内连续维）。轴质量评分：`bn=64` fp32（向量宽 4）→ 整除 ✓、尾 lane 浪费 0%；归约轴 head（`H_p=32`）作垂直串行维（非向量维），不阻碍向量化。**弃选理由均为量化/契约性**（#3 host permute 净亏且违约束；#4 无收益；#2 无收益且增重排），非「未考虑」。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（非持久，单 kernel 内 Cube + Vector 顺序书写，由编译器自动 CV-split）。

### 2.2 选型理由

- 算子含 **Cube `T.gemm`**（`Sq` 矩阵乘）+ **Vector epilogue**（relu/加权/归约），是典型混合算子，Developer 模式在同族已有**可运行先例**：`examples/indexer/indexer_fwd.py`（`@tilelang.jit(target="npuir")`，`T.gemm` + `T.copy(acc_frag, acc_ub)` + `T.vrelu`/`T.vmul`/`T.reduce_sum`）——与本设计结构同形。
- 脚手架 wrapper 已设 `ascend_mode = "Developer"`（K9 去 `threads`），且 `op_requirements.programming_mode = developer`。
- **约束核对**：CG-2026-0010 指出 **Developer 模式对 persistent + gemm + vector 混排不支持**（运行时非法指令），**仅适用于非 persistent 网格**——本 kernel 为**非持久** 1D grid（同 `indexer_fwd.py`），落在 Developer 适用域内。
- 相对 Expert：免手写 `T.Scope("Cube")/("Vector")` + `sync_block_set/wait` + `alloc_L1/alloc_L0C`，且 **L0C→UB 直拷**省掉 Expert `store_fixpipe` 强制的 GM workspace 往返（`CONST-store-fixpipe-gm-only`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB/L1 由编译器按用途放置）、`T.alloc_fragment`（acc，Cube 累加）；不显式 `alloc_L0C/L1` |
| 计算方式 | `T.gemm`（Cube）+ v-prefix 向量 API + `T.reduce_sum`（Vector）；C→V 交接由 `T.copy(acc_frag, acc_ub)` |
| 同步 | **自动同步**（Developer 由编译器插入 CV 交接/流水同步），无手写 flag |
| 流水 | `T.Pipelined(num_stages)` 双缓冲（MTE2 预取 K/scale） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 **优化后公式**。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `k0 = cu_k_s_min + nbn·bn`；`ks=k0`（块起点） | 窗口并集 + kv 块游标 |
| 2 | `K_t ← IndexK[b, k0:k0+bn, g, 0:D]` | Cube 操作数 B（`[bn, D]`） |
| 3 | `Q_t ← IndexQ[b, seq_len_i·H : (seq_len_i+bq)·H, 0:D]` | Cube 操作数 A（`[bq·H_p, D]`，hoist） |
| 4 | `acc[qh, n] = Σ_d K_t[n,d]·Q_t[qh,d]`（`Sq`） | Cube gemm（`b_transpose=True`），fp32 accum |
| 5 | `r = relu(acc)` | 向量 relu |
| 6 | `r = r · W_t[qh]`（行因子） | 向量乘（权重广播） |
| 7 | `out[q, n] = Σ_{h} r[q·H_p+h, n]` | 组内 head 垂直归约 |
| 8 | `out[q, n] = out[q, n] · Scale[n]`（E1；E3 无 scale 时省略） | 列因子 scale（vbrc+vmul） |
| 9 | `Logits[b, seq_len_i+q, k0+n, g] = out[q,n]` | 输出写回（块对齐超集） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| grid | 逻辑任务 | `T.Kernel(B*⌈S/bq⌉, is_npu=True)` | 1D 非持久 grid；`cid → (by=batch, bx=q-block)` | Developer |
| 2/3 | tile 装载 | `T.copy(IndexK[b, k0:k0+tail, g, 0:D], k_ub)` / `T.copy(IndexQ[b, seq_len_i*H, 0], q_ub, size=[bq*H, D])` | 索引+size / 切片；`tail=T.min(bn, Skv−k0)` 动态尾 | Developer |
| 4 | `Sq = K·Qᵀ` | `T.gemm(k_ub, q_ub, acc, b_transpose=True, initC=True)` | M=`bq·H_p`, N=`bn`, K=`D`；acc `alloc_fragment` fp32 | Developer |
| — | L0C→UB | `T.copy(acc, acc_ub)` | acc(fp32 fragment) → acc_ub(fp32 UB)，保形 | Developer |
| 5 | `relu` | `T.vrelu(acc_ub, acc_ub)` | 原地；fp32 支持（bf16 ✗） | Developer |
| 6 | `·w`（行因子） | `T.vmul(acc_ub, w_ub, acc_ub)` | `w_ub [bq*H_p, 1]` fp32 行因子 | Developer |
| 7 | `Σ_h` | `T.reduce_sum(acc_ub[q*Hp:(q+1)*Hp, :], out_ub[q:q+1, :], dim=0, clear=True)`（q 循环） | 垂直归约，`dim=0` | Developer |
| 8 | `·Scale`（列因子） | `T.vbrc(scale_ub, brc_ub)` + `T.vmul(out_ub, brc_ub, out_ub)` | `scale_ub [1,bn]` → `brc_ub [bq,bn]`；E3 时整段省略 | Developer |
| 9 | 写回 | `T.copy(out_ub, Logits[b, seq_len_i, k0], size=[bq, bn])` | `G>1` 时按 `g` 分量切片写 | Developer |
| 流水 | kv 双缓冲 | `T.Pipelined(nkv, num_stages=num_stages)` | `nkv = T.ceildiv(cu_k_e_max−cu_k_s_min, bn)` 动态上界 | Developer |

### 3.3 计算伪代码

> 依 `design-template.md` 提示：`T.Tensor` 的 dtype 直接作第二位置参数传入（不用 `dtype=`）。

```python
@tilelang.jit(target="npuir")
def _fp8_lightning_indexer_func(block_N=64, num_stages=2, block_Q=8):
    index_dtype = "bfloat16"          # 契约 bf16（或 "float16"）；fp8 不可用（CG-2026-0022）
    accum_dtype = "float32"
    index_dtype_i = "int32"

    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    @T.prim_func
    def _fp8_lightning_indexer_main(
        IndexQ: T.Tensor([batch, seq_len * heads, index_dim], index_dtype),
        IndexK: T.Tensor([batch, seq_len_kv, kv_group, index_dim], index_dtype),
        IndexKScale: T.Tensor([batch, seq_len_kv, kv_group], accum_dtype),   # 可选语义：has_scale
        Logits: T.Tensor([batch, seq_len, seq_len_kv, kv_group], accum_dtype),
        Weights: T.Tensor([seq_len, heads], accum_dtype),
        CuSeqLenKS: T.Tensor([seq_len], index_dtype_i),
        CuSeqLenKE: T.Tensor([seq_len], index_dtype_i),
    ):
        heads_per_group = heads // kv_group
        num_q_blocks = T.ceildiv(seq_len, block_Q)
        with T.Kernel(batch * num_q_blocks, is_npu=True) as (cid, _):
            by = cid // num_q_blocks              # batch
            bx = cid % num_q_blocks               # q-block
            seq_len_i = bx * block_Q

            q_ub = T.alloc_shared([block_Q * heads_per_group, index_dim], index_dtype)  # hoist
            w_ub = T.alloc_shared([block_Q * heads_per_group, 1], accum_dtype)          # hoist
            k_ub = T.alloc_shared([block_N, index_dim], index_dtype)
            scale_ub = T.alloc_shared([1, block_N], accum_dtype)      # has_scale
            brc_ub = T.alloc_shared([block_Q, block_N], accum_dtype)  # has_scale
            acc = T.alloc_fragment([block_Q * heads_per_group, block_N], accum_dtype)  # L0C
            acc_ub = T.alloc_shared([block_Q * heads_per_group, block_N], accum_dtype)
            out_ub = T.alloc_shared([block_Q, block_N], accum_dtype)

            # 窗口并集（block 级标量，编译期展开 block_Q 次；§1.6.2 #7）
            cu_k_s_min = ...   # min over block_Q of min(CuSeqLenKS[s], seq_len_kv)
            cu_k_e_max = ...   # max over block_Q of min(CuSeqLenKE[s], seq_len_kv)

            # hoist 装载（O3）
            T.copy(IndexQ[by, seq_len_i * heads, 0], q_ub, size=[block_Q * heads_per_group, index_dim])
            T.copy(Weights[seq_len_i, 0], w_ub, size=[block_Q * heads_per_group, 1])

            for g in T.serial(kv_group):                     # G==1：单次
                n_kv = T.ceildiv(cu_k_e_max - cu_k_s_min, block_N)
                for nbn in T.Pipelined(n_kv, num_stages=num_stages):
                    k0 = cu_k_s_min + nbn * block_N
                    tail = T.min(block_N, seq_len_kv - k0)
                    T.copy(IndexK[by, k0:k0 + tail, g, 0:index_dim], k_ub[0:tail, :])
                    if has_scale:
                        T.copy(IndexKScale[by, k0, g], scale_ub, size=[1, tail])
                    # Cube GEMM: acc[qh, n] = sum_d k[n,d] * q[qh,d]
                    T.gemm(k_ub, q_ub, acc, b_transpose=True, initC=True)
                    # Vector epilogue（fp32 域）
                    T.copy(acc, acc_ub)
                    T.vrelu(acc_ub, acc_ub)
                    T.vmul(acc_ub, w_ub, acc_ub)
                    for q in T.serial(block_Q):
                        T.reduce_sum(acc_ub[q * heads_per_group:(q + 1) * heads_per_group, :],
                                     out_ub[q:q + 1, :], dim=0, clear=True)
                    if has_scale:                              # E3：无 scale 时编译期省略
                        T.vbrc(scale_ub, brc_ub)
                        T.vmul(out_ub, brc_ub, out_ub)         # E1：scale 在归约外乘
                    T.copy(out_ub, Logits[by, seq_len_i, k0, g], size=[block_Q, tail])
    return _fp8_lightning_indexer_main
```

> **接口契约（不改变）**：工厂签名 `(block_N, num_stages, block_Q)`（无 `threads`，K9）；`prim_func` 7 个张量参数顺序与源一致；`Logits` 显式入参（非 out_idx）；`block_Q` 由 wrapper 的 `default_config` 传入。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.gemm(..., b_transpose=, initC=)` | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.4；`examples/indexer/indexer_fwd.py` L68 | 文档+先例（fp16；bf16 见 `testing/npuir/bf16_support_ops/test_gemm_bf16.py`） |
| `T.copy`（索引+size / 切片；L0C fragment↔UB） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2/§3 表 | 文档；L0C→UB 先例 `indexer_fwd.py` L70 |
| `T.vrelu` | `docs/Tilelang.language/数学操作/T.vrelu.md` §2.2.1（fp16/fp32 ✓，bf16 ✗） | 文档+先例 `indexer_fwd.py` L71 |
| `T.vmul`（行因子） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2（`[M,N]*[M,1]` ✓；列因子 ✗） | 文档+先例 `indexer_fwd.py` L72 |
| `T.reduce_sum(dim=0, clear=True)` | `docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2 / `T.reduce_sum.md`；先例 `indexer_fwd.py` L74–L80 | 文档+先例 |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（`(1,N)→(M,N)` ✓） | 文档（Stage 3 首编验证） |
| `T.Pipelined(动态 trip, num_stages)` | `examples/indexer/indexer_fwd.py` L64；`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.4 | 先例 |
| `T.symbolic` / `T.ceildiv` / `T.min` | `indexer_fwd.py` L29–L31；`docs/Tilelang.language/数学操作/T.ceildiv.md`、`比较操作/T.min.md` | 文档+先例 |
| `T.alloc_shared` / `T.alloc_fragment` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；`allocate.py` | 文档+先例 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **否** | NPU grid ≤2D；本设计用 **1D** grid（`T.Kernel(batch*num_q_blocks, is_npu=True)`），未用第三维 |
| GPU 专用 API | **是** | `T.get_thread_binding`/`threads`/`GemmWarpPolicy`/`TL_ENABLE_FAST_MATH`/`T.float8_e4m3fn` 均不可用 → §0.5 处置（R2/舍弃/§0.6） |
| **FP8 dtype / fp8 GEMM** | **是（阻塞，已登记能力缺口）** | 源 `dtype=T.float8_e4m3fn`、fp8 `T.gemm`、设备端 fp8 量化全不可用 → **CG-2026-0022**；设计改 bf16/fp16 载体（§0.5 条目 2 / §0.6 R1），fp8 workload 标受限（§9.1） |
| GEMM 非整除（M/N 非 block 整数倍） | **部分** | M=`bq·H_p`(=256，整除 ✓)；N=`bn`=64 整除 ✓；**kv 尾块非整除**（窗口并集起点非 `bn` 对齐 + `seq_len_kv` 非 `bn` 整数倍）→ 动态 `tail=T.min(bn, Skv−k0)` + 零起点 UB 切片（PL-1.22 形态）+ `k_ub` 循环前零初始化（防 NaN 污染，PL-1.11 zbuf 形态）；`seq_len` 尾块：`ceildiv` + 行守卫 |
| L0C 溢出（`block_M×block_N×4 > 131072`） | **否（恰在界内）** | acc `bq·H_p × bn = 8·32 × 64 = 256×64 = 16384` elem = 64KB ≤ 128KB ✓（约束 `bm·bn ≤ 32768`，`CONST-cube-gemm-l0c-traffic-floor`） |
| 分核策略缺失/不适配 | **是** | 见 §5.5（逻辑核 1024、物理核实查 20、“无法对齐兜底”并记录残留负载不均） |
| Host 侧改输入内容（`ascend-constraints.md` §4） | **否** | kernel 内完成全部计算；host 仅做 `IndexQ.view`（纯元数据）；不使用 host 元数据张量（窗口并集在 kernel 内算，避免 `indexer_fwd.py` 的 host `prepare_metadata` 形态） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | 2D `T.Kernel(⌈S/bq⌉, batch, threads=512)` | 1D `T.Kernel(batch*num_q_blocks, is_npu=True)` | cid 解码（§0.6 R2；`T.Kernel.md` §2.4 例 1） |
| 线程模型 | 512 线程 + `T.Parallel`/warp 归约 | 无线程；v-prefix + `T.reduce_sum` | §0.6 R2 |
| GEMM | fp8 TensorCore `T.gemm(..., policy=FullCol)` | bf16/fp16 Cube `T.gemm(b_transpose=True)` | §0.6 R1；policy 无对应（舍弃） |
| 数据类型 | `T.float8_e4m3fn` | bfloat16 / float16 | CG-2026-0022 |
| 内存分配 | `alloc_shared`/`alloc_fragment`（SMEM/reg） | `alloc_shared`(UB/L1) / `alloc_fragment`(L0C) | Developer 自动放置 |
| C→V 交接 | fragment 直读 | `T.copy(acc, acc_ub)`（L0C→UB） | `T.copy.md` §3；`indexer_fwd.py` L70 |
| 流水 | `T.Pipelined(num_stages=3)` + 死锁守卫 | `T.Pipelined(num_stages=2)`，无守卫 | §0.6 R3 |
| 掩码 | 主 kernel 块对齐超集 + 独立 `clean_logits_` | **同**（掩码属另一迁移单元） | §5.4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/indexer/indexer_fwd.py` | **极高**（同族 NPU indexer） | Developer 1D 非持久 grid、`T.gemm(q,k,acc,b_transpose=True)`、`T.copy(acc_frag→acc_ub)`、`T.vrelu`/`T.vmul` 行因子、逐行 `T.reduce_sum(dim=0)`、窗口块范围迭代 |
| `examples/deepseek_v32/fp8_lighting_indexer.py` | **高**（同算子 NPU 先例） | `dtype="float16"` 载体（fp8 不可用佐证）、Expert 两 Scope 结构、`store_fixpipe(pre_relu_mode="relu")`、`vmul(·w)` 在 reduce 前 / `vmul(·k_s)` 在 reduce 后（E1 先例）、FFTS flag 预算 |
| `examples/deepseek_v4/example_fp8_gemm_kernel.py` | 中 | NPU “fp8” GEMM 实为 fp16 载体（L35/L184）、`T.gemm(..., b_transpose=, size=)` npuir 签名 |
| `examples/flash_attention/flash_attn_npuir_dev.py` | 中 | Developer 模式 Cube+Vector 非持久混排形态 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel_DESIGN.md` | 中 | 分核三要素写法（`get_aicore_num` 实查记录）与 UB 预算表范式 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `IndexQ` | `[batch, seq_len * heads, index_dim]` | **bfloat16（默认）/ float16** | 由 `[batch, seq_len, heads, index_dim]` view；fp8 不可用（CG-2026-0022） |
| `IndexK` | `[batch, seq_len_kv, kv_group, index_dim]` | bfloat16 / float16 | 键缓存 |
| `IndexKScale` | `[batch, seq_len_kv, kv_group]` | float32 | 键反量化 scale（可选；`has_scale` 编译期开关） |
| `Weights` | `[seq_len, heads]` | float32 | head 权重 |
| `CuSeqLenKS` | `[seq_len]` | int32 | 窗口下界 |
| `CuSeqLenKE` | `[seq_len]` | int32 | 窗口上界 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Logits` | `[batch, seq_len, seq_len_kv, kv_group]` | float32 | 打分输出（显式入参，非 out_idx）；块对齐超集写，掩码由 `clean_logits_` 覆写 |

### 4.3 中间缓冲区（主选 block_Q=8, block_N=64, heads=32, H_p=32, D=64, 载体 bf16）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `q_ub` | `[256, 64]` | bfloat16 | UB/L1 | Q tile（hoist，每核一次） |
| `w_ub` | `[256, 1]` | float32 | UB | 权重行因子（hoist） |
| `k_ub` | `[64, 64]` | bfloat16 | UB/L1 | K tile（每 kv 块；循环前零初始化） |
| `scale_ub` | `[1, 64]` | float32 | UB | 反量化 scale 行（`has_scale`） |
| `brc_ub` | `[8, 64]` | float32 | UB | scale 列因子广播体（`has_scale`；E3 时省略） |
| `acc` | `[256, 64]` | float32 | **L0C**（`alloc_fragment`） | Cube 累加器 |
| `acc_ub` | `[256, 64]` | float32 | UB | L0C→UB 的 epilogue 工作区 |
| `out_ub` | `[8, 64]` | float32 | UB | 归约输出（block_Q 行） |

### 4.4 内存搬运路径

```
GM[IndexQ]  --T.copy--> UB/L1 q_ub (hoist, 仅一次)
GM[Weights] --T.copy--> UB   w_ub (hoist, 仅一次)
GM[IndexK]  --T.copy--> UB/L1 k_ub ──> T.gemm(k_ub, q_ub, acc, b_transpose=True) ──> L0C acc (fp32)
L0C acc --T.copy(保形)--> UB acc_ub --T.vrelu--> --T.vmul(w_ub)--> --T.reduce_sum(dim=0, 逐q)--> UB out_ub
GM[IndexKScale] --T.copy--> UB scale_ub --T.vbrc--> brc_ub --T.vmul--> UB out_ub   (has_scale)
UB out_ub --T.copy--> GM[Logits] (块对齐超集)
[另一单元] clean_logits_: GM[Logits] 窗口外 := -inf
```

**关键点**：Developer 通用 copy 路径支持 **L0C fragment↔UB**（`T.copy.md` §2.1 表 / §3 分发），故 `acc`(L0C) **直接拷入 UB**，**不经 GM workspace**——省掉 Expert `store_fixpipe` 强制的 GM 往返（`CONST-store-fixpipe-gm-only`）。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `q_ub` | (256, 64) | bfloat16 | 32768 |
| `w_ub` | (256, 1) | float32 | 1024 |
| `k_ub` | (64, 64) | bfloat16 | 8192 |
| `scale_ub` | (1, 64) | float32 | 256 |
| `brc_ub` | (8, 64) | float32 | 2048 |
| `acc_ub` | (256, 64) | float32 | 65536 |
| `out_ub` | (8, 64) | float32 | 2048 |
| 标量/边界临时 | — | int32/fp32 | 64 |
| **手工小计** | | | **≈111.9 KB** |
| **×auto-multi-buffer 膨胀 ≈1.10**（`CONST-capacity-910B2C`，910B2C 结构实测；本机 910B3 未重验——未实测假设） | | | **≈123 KB** |
| **总计** | | | **≈123 KB / 192 KB（UB 上限）** ✅ |

**L0C**：`acc` = 256×64×4 = 64KB ≤ 128KB（且 16384 elem ≤ 32768，`CONST-cube-gemm-l0c-traffic-floor`）✅。**L1**：`q_ub`+`k_ub` 若落 L1 共 40KB ≪ 512KB ✅。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `seq_len` | `T.symbolic("seq_len")` | manifest 1024 / 4096 / 8192（约束 `seq_len % block_Q` 由尾块守卫处理） |
| `seq_len_kv` | `T.symbolic("seq_len_kv")` | 1024 / 8192 |

`batch` / `heads` / `index_dim` / `kv_group` 为编译期常量（工厂 `lru_cache` 键，与源一致）。

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir")   # 无 out_idx（Logits 为显式入参）；无 pass_configs（无超越函数）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**混合（Cube GEMM + Vector epilogue）**。判定依据：含矩阵乘 `Sq = K·Qᵀ`（MAC 密集 → Cube）+ 逐元素/归约 epilogue（→ Vector）。

### 5.2 Block 划分

```python
block_Q = 8    # query 行数（kv 复用单位；越大 K 重读越少，但 q_ub/L0C 越大）
block_N = 64   # kv 列数（向量化轴；fp32 尾轴 256B 对齐）
num_logical = batch * ceildiv(seq_len, block_Q)   # 逻辑核数
```

**选择理由**：约束 `(block_Q·H_p)·block_N ≤ 32768`（L0C fp32 容量）。候选矩阵（§1.6.3 布局 #1 下）：

| (block_Q, block_N) | L0C elem | L0C | UB 估算(×1.10) | K 重读次数（S=8192） | 备注 |
|--------------------|---------|-----|----------------|---------------------|------|
| (8, 64) | 16384 | 64KB | ≈123KB ✓ | 1024 | **主选**（平衡） |
| (16, 32) | 16384 | 64KB | ≈152KB ✓ | 512 | **备选**（K 读减半，`bn` 变窄） |
| (4, 128) | 16384 | 64KB | ≈110KB ✓ | 2048 | K 读翻倍 → 弃选 |
| (16, 64) | 32768 | 128KB(满) | >192KB ✗ | 512 | UB 超容 → 弃选 |
| (8, 128) | 32768 | 128KB(满) | >192KB ✗ | 1024 | UB 超容 → 弃选 |

**实验裁决模式（D-2/§1.6.3）**：主选与备选的判定裕度依赖两个**未实测常数**（本机 910B3 的 Cube 吞吐比与 L2 对 K 的驻留率），且 §1.6.0 估算显示算子**发射项主导**（tiling 对发射项几乎无影响、仅影响流量项）→ 不纸面拍板：
- **① 主选**：(block_Q=8, block_N=64)（本设计正文/§3.3 承载）。
- **② 备选**：(block_Q=16, block_N=32)，结构同上、仅 tile 尺寸不同（buffer 形状等比例改），分核三要素按新粒度重算（逻辑核 512）。
- **③ 裁决计划**：Stage 4 在代表 shape（smoke `s1k-h8`、`bf16-s8k-h32`、`bf16-s4k-h32`）A/B 实测 msprof op Task Duration + 记录 `aic_mte2_ratio`/`aic_cube_ratio`；**判定阈值**：备选 latency < 主选 × (1 − 5%) 才翻转；结果回写 §1.6.3/§5.2 并产出 `perf_feedback.md`（若翻转）。**shape 特化**：因 `seq_len` 影响逻辑核数（smoke 的 128 核负载不均 +9.4% vs full 的 +1.6%），裁决须评估**按 shape 分派主选/备选**的可行性。

### 5.3 约束分析

- **对齐约束**：`block_N=64` fp32 尾轴 64×4=256B（≥32B ✓）；`index_dim=64` 分形对齐 ✓；`block_Q*H_p=256` 分形 M 对齐 ✓。
- **UB 容量**：手工 ≈111.9KB，膨胀后 ≈123KB < 192KB ✓（§4.5）。
- **L0 容量**：acc 16384 elem = 64KB ≤ 128KB ✓。
- **32B 尾轴**：`out_ub [8,64]` fp32 尾轴 256B ✓；`scale_ub [1,64]` ✓。

### 5.4 注意事项（非整除 / 尾块 / 与 clean_logits_ 的接口契约）

- **kv 块起点非对齐**：`cu_k_s_min` 为运行时值，可能非 `block_N` 整数倍 → 用 `tail = T.min(block_N, seq_len_kv − k0)` 动态 extent 切片 + 零起点 UB 切片 `k_ub[0:tail, :]`（PL-1.22 形态）；`k_ub` 在 kv 循环前**零初始化一次**，杜绝尾块陈旧/未初始化行经 gemm 产生 NaN 污染（PL-1.11 zbuf 形态；`relu` 无法清除 NaN）。
- **seq_len 尾块**：`ceildiv(seq_len, block_Q)`，末 block 行数不足时 `T.copy` 用动态 extent 切片并 `if seq_len_i + q < seq_len` 守卫写回。manifest shapes 均被 block_Q=8 整除（1024/4096/8192 ✓），尾块逻辑为通用性保留。
- **窗口并集可能空**：若 `cu_k_e_max ≤ cu_k_s_min`（全行无可见 kv），`n_kv=0`，主循环不执行 → 不写任何 Logits（与源一致；该区域由 `clean_logits_` 覆写 `-inf`）。
- **块对齐超集写（与 `clean_logits_` 的契约）**：主 kernel **只保证写入** `[cu_k_s_min, cu_k_s_min + n_kv·block_N)` 的块对齐区域（≥ 并集窗口）；该区域内**但落在各行 `[ks,ke)` 之外**的值**不保证**为 `-inf`（可能是任意计算值），必须由 `clean_logits_` 迁移单元覆写。反之主 kernel **不写** `[0, cu_k_s_min)` 与 `[cu_k_e_max_rounded, seq_len_kv)` —— `clean_logits_` 必须覆盖 `[0,ks)∪[ke,Skv)` 全区间（含这些）。此契约与源 `clean_logits_` 的行为一致（源 L170–L182），**本设计不得单方面改变**。
- **`G>1`**：分组循环 + K 组跨步切片 + Q 组逐行 `T.copy`（§1.6.2 #11）；无 manifest workload 覆盖，Stage 3 需专项验证（列 §9.2）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`../../../.agents/skills/_shared/standards/core-split-strategy.md` §1/§2.1（依据 `docs/开发指南.md` §3.3）。

- **① 物理核数依据（实查）**：

  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()          # 实查返回: 20  （本机 Ascend910B3）
  NPUUtils.get().get_aivector_core_num()   # 实查返回: 40  （= 20 × 2）
  ```

  → **物理核数 = 20 AI Core**（本算子为 **Cube+Vector 混合算子** → 直接使用 `get_aicore_num()` 返回值，**不翻倍**；翻倍仅适用于**纯 Vector** 算子）。查询时间：2026-10-15，工具链 `tilelang 0.1.2+a2357253662babf317ada194080078f928d155b3`（dev root build）/ CANN 9.0.0。**禁止文档假设/经验值**；本机设备名为 `Ascend910B3`（`torch.npu.get_device_name(0)`），核数 20 与 `CONST-aicore-910_9362` 一致但 SoC 名不同——常数表未含 910B3 条目，此为**实查值**记录。

- **② 逻辑核数计算**：`num_logical_kernels = batch × ceildiv(seq_len, block_Q)`
  - smoke `s1k-h8`：1 × ⌈1024/8⌉ = **128**
  - full `bf16-s4k-h32`：1 × ⌈4096/8⌉ = **512**
  - full `bf16-s8k-h32` / `fp8-s8k-h32`：1 × ⌈8192/8⌉ = **1024**

- **③ 规模判定与分核方案**：逻辑核数（128–1024）**远大于**物理核数（20）→ 属**极大规模**；但核心结构为 **Developer + gemm + vector 混排**，**核内串行 persistent 不可用**（CG-2026-0010：Developer 对 persistent 混排运行时崩溃，仅 Expert 支持），无法用「固定启动核数 = 物理核数 + 核内 `T.serial` 多任务」摊薄 → 落到标准 §1 要素③的 **「无法对齐兜底」**：
  - `seq_len` 为 2 的幂（1024/4096/8192 = 2¹⁰/²¹³），`block_Q` 取 2 的幂 → 逻辑核数恒为 2 的幂；物理核数 **20 = 2²·5** 含非 2 因子 5；`gcd(2ⁿ, 20) = 4` → 数学上**不存在**任何 2 的幂逻辑核数能被 20 整除 → **无法对齐任意整数倍**。
  - **方案**：取最近的物理核倍数调度（由运行时按 20 物理核分波执行），并**显式记录残留负载不均**：full shape 1024 = 20×51 + 4 → **4 核执行 52 个任务、16 核执行 51 个**，临界路径惩罚 ≈ 52/51.2 − 1 = **+1.6%**；smoke 128 = 20×6 + 8 → 8 核 7 任务、12 核 6 任务，惩罚 ≈ 7/6.4 − 1 = **+9.4%**（smoke 短任务，绝对影响小）。
  - **不采用核内串行 persistent 的原因**（附依据）：CG-2026-0010（Developer 模式 persistent + gemm + v-prefix 混排运行时崩溃、仅 Expert 可用）——若改 Expert 可解锁 persistent 串行，但会引入 `store_fixpipe` 的 GM workspace 往返（`CONST-store-fixpipe-gm-only`）与手写双 Scope 同步，权衡后**维持 Developer 非持久**（§2.2）。该取舍列入 §9.1，Stage 4 可作可选优化项复评。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑任务（batch × q-block） | block 级并行 | `T.Kernel(batch*num_q_blocks, is_npu=True)` | 1D 非持久 grid；每核处理一个 q-block 的全部 kv 窗口 |
| `kv_group`（G） | tile 级顺序 | `T.serial(kv_group)` | G>1 分组 gemm（G=1 退化为单次）；组间独立、无跨组依赖 |
| kv 窗口 | tile 级流水迭代 | `T.Pipelined(n_kv, num_stages)` | 动态 trip（窗口并集）；双缓冲预取 K/scale |
| 组内 head 归约 | 向量垂直归约 | `T.reduce_sum(..., dim=0)`（`T.serial(block_Q)` 内） | 每 query 行归约 `H_p` 行；block_Q 次（≤8） |
| 元素级（relu/乘/写） | 向量化 | `T.vrelu` / `T.vmul` / `T.copy` 整 buffer | 依 §1.6.2 已全部向量化，无逐元素标量循环 |
| 逐元素并行循环 | **不用 `T.Parallel`** | — | 源用 `T.Parallel` 表达的逐元素式（relu/乘权/乘 scale/写回）已由 v-prefix 整 buffer 操作 + `T.copy` 等价承接（§1.6.2 #1/#2/#4/#8）；`T.Parallel` 仅在两处受限（下标不可变换、`G>1` gather，§1.6.2 #11），无需要用它的计算点 |
| 核内 persistent 串行多任务 | **不用 `T.Persistent`**（也不手写 `T.serial` 核内多任务持久化） | — | Developer 非持久（CG-2026-0010）；核启动开销取舍见 §5.5，Stage 4 可复评 |

### 6.2 循环伪代码

```python
with T.Kernel(batch * num_q_blocks, is_npu=True) as (cid, _):
    by = cid // num_q_blocks; bx = cid % num_q_blocks; seq_len_i = bx * block_Q
    # hoist 装载（O3）
    T.copy(IndexQ[by, seq_len_i*heads, 0], q_ub, size=[block_Q*heads_per_group, index_dim])
    T.copy(Weights[seq_len_i, 0], w_ub, size=[block_Q*heads_per_group, 1])
    # 窗口并集（编译期展开 block_Q 次；§1.6.2 #7）
    cu_k_s_min = min over block_Q (min(CuSeqLenKS[...], seq_len_kv))
    cu_k_e_max = max over block_Q (min(CuSeqLenKE[...], seq_len_kv))
    T.clear(k_ub) if has_tail_guard else None          # 防尾块 NaN（PL-1.11）
    for g in T.serial(kv_group):                        # G==1 → 单次
        n_kv = T.ceildiv(cu_k_e_max - cu_k_s_min, block_N)
        for nbn in T.Pipelined(n_kv, num_stages=num_stages):
            k0 = cu_k_s_min + nbn * block_N
            tail = T.min(block_N, seq_len_kv - k0)
            T.copy(IndexK[by, k0:k0+tail, g, 0:index_dim], k_ub[0:tail, :])
            if has_scale: T.copy(IndexKScale[by, k0, g], scale_ub, size=[1, tail])
            T.gemm(k_ub, q_ub, acc, b_transpose=True, initC=True)
            T.copy(acc, acc_ub)
            T.vrelu(acc_ub, acc_ub)
            T.vmul(acc_ub, w_ub, acc_ub)
            for q in T.serial(block_Q):
                T.reduce_sum(acc_ub[q*heads_per_group:(q+1)*heads_per_group, :],
                             out_ub[q:q+1, :], dim=0, clear=True)
            if has_scale:
                T.vbrc(scale_ub, brc_ub); T.vmul(out_ub, brc_ub, out_ub)   # E1
            T.copy(out_ub, Logits[by, seq_len_i, k0, g], size=[block_Q, tail])
```

### 6.3 流水线优化

- **`T.Pipelined(n_kv, num_stages=2)`**：KV 方向双缓冲，MTE2 预取 `k_ub`/`scale_ub` 与 Cube gemm / Vector epilogue 重叠。
- **`num_stages` 语义重定义**（§0.6 R3）：源 `num_stages` 是 CUDA 软件流水深度（受 `_MIN_PIPELINED_BLOCK_Q` 守卫）；NPU 上仅作**缓冲深度旋钮**（受 UB 预算），**无 block_Q≥2 约束**。工厂签名保留 `num_stages`（接口契约），默认 2。
- **buffer 管理**：`q_ub`/`w_ub` 为循环不变量（hoist，不参与流水）；`k_ub`/`scale_ub`/`acc`/`acc_ub`/`out_ub` 在 kv 循环内，逐 iterator 复用。

### 6.4 尾块处理

见 §5.4：kv 尾块（动态 `tail` + 零起点 UB 切片 + `k_ub` 预清零）、`seq_len` 尾块（行守卫）、空窗口（`n_kv=0` 不写）。manifest shapes 全部整除，尾块逻辑为通用性保留。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式由编译器插入 CV 交接与流水同步指令，用户不手写 flag）。

### 7.2 同步点说明

无手写同步点。Developer 自动处理：GM→UB/L1（MTE2）→ Cube（MTE1/M）→ Vector（V）→ GM 的依赖链，以及 `T.Pipelined` 的双缓冲期次同步。**若 Stage 3 遇到同步相关崩溃**（如 CV 交接竞态），回退手段见 §9.2（转 Expert 显式 `sync_block_set/wait`）。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # 无。源 TL_ENABLE_FAST_MATH 无 NPU 对应（O9，epilogue 无超越函数）
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据，**直接移植源仓参考实现**（`examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py`（`ref_program`），L67–L94），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。输入域为原始 index 张量（bf16/fp16，未量化）；反量化 scale 语义由可选 `index_k_scale` 参数承接。

```python
def golden_fp8_lightning_indexer(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke,
                                 index_k_scale=None):
    """Lightning-indexer logits. index_q [B,S,H,D]; index_k [B,Skv,G,D];
    weights [S,H] fp32; cu_seqlen_ks/ke [S] int32; returns [B,S,Skv,G] fp32."""
    q = index_q.float()
    k = index_k.float()
    B, S, H, D = q.shape
    Skv, G = k.shape[1], k.shape[2]
    Hp = H // G
    k = k.view(B, Skv, G, D)
    q = q.view(B, S, G, Hp, D)
    score = torch.einsum("bsghd,bngd->bghsn", q, k)          # [B,G,Hp,S,Skv]
    w = weights.view(S, G, Hp).permute(1, 2, 0)              # [G,Hp,S]
    score = score.relu() * w.unsqueeze(0).unsqueeze(-1)
    logits = score.sum(dim=2)                                # 组内 head 求和
    logits = logits.permute(0, 2, 3, 1)                      # [B,S,Skv,G]
    if index_k_scale is not None:
        logits = logits * index_k_scale.view(B, 1, Skv, G)
    # 窗口掩码（-inf）
    ar = torch.arange(0, Skv, device=q.device)
    mask = (ar[None, :] >= cu_seqlen_ks[:, None]) & (ar[None, :] < cu_seqlen_ke[:, None])
    logits = logits.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
    return logits
```

**比较器**（沿用源：`examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py`（`_validate_tensor_match`），L41–L65）：先比对 `isfinite` 掩码一致、非有限值逐位一致，再在有限域用 **correlation** `1 − 2·Σab/Σ(a²+b²) ≤ 1e-3`。

### 8.2 精度标准

| 域 | 判据 | 容差 |
|----|------|------|
| 有限域 | correlation difference `1 − corr` | ≤ **1e-3**（源测试契约） |
| 非有限域 | `isfinite` 掩码逐位 + NaN 一致 | **精确** |
| 窗口内元素 | （可选辅助）rtol/atol | 1e-2 / 1e-2（bfloat16 载体，参考源 bench） |

### 8.3 L0 门槛测试计划（本阶段规划，交 `tilelang-op-develop`）

| 编号 | 用例 | 目的 |
|------|------|------|
| L0-1 | smoke `batch1/s1024/h8/d64/skv1024/g1`，`clean_logits=False`（无 scale） | 最小正确性（E3 路径）、编译通过 |
| L0-2 | full `b1/s8192/h32/d64/skv8192/g1` bf16 | 契约主 shape |
| L0-3 | full `b1/s4096/h32/d64/skv8192/g1` bf16 | 非方形（S≠Skv）+ 窗口非全可见 |
| L0-4 | `b1/s4096/h32/d64/skv8192/g2` bf16 | **G>1 分组路径**（§1.6.2 #11 专项） |
| L0-5 | 外部 `index_k_scale` 提供（`has_scale=True`） | E1 scale 路径 |
| L0-6 | 非整除/尾块（`seq_len_kv` 非 block_N 整数倍、`ks` 非对齐） | 尾块 + 零初始化守卫 |
| Boundary | 空窗口（`ke≤ks`）、全 `-inf` 区域（配 `clean_logits_`） | 边界语义 |

（完整 L1/L2/Boundary 套件交 `tilelang-op-develop`，此处不枚举。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **FP8 不可用（能力缺口 CG-2026-0022，最高风险）**：fp8 index 载体 / fp8 gemm / 设备端 fp8 量化在 npuir 全链路不可用。**受限范围**：manifest 的 `lightning-indexer-fp8-s8k-h32-d64` workload 在缺口修复前**不纳入 NPU 交付契约**；交付契约为 **bf16/fp16 index 路径**（manifest 4 个 workload 中 3 个为 bf16）。集成层需相应放宽 `FP8LightningIndexerKernel` 的 `dtype == torch.float8_e4m3fn` 校验（脚手架注释已明示 wrapper 将被 `integrate_kernel.py` 重新同步）。
2. **估算常数绑定 9362、本机 910B3 未实测**（§1.6.0 估算下界/§4.5 UB 膨胀系数）：带宽/发射常数取 `CONST-ascend910-9362-profile` / `CONST-vector-op-cost-910_9362`，标注「未实测假设 + 估算依据」；Stage 4 须回填实测对账（`hardware-cost-model.md` §3）。
3. **Developer 非持久约束（CG-2026-0010）**：本 kernel 必须保持非持久；若误改 persistent 会运行时崩溃。取舍（放弃 persistent 串行摊薄核启动开销）见 §5.5。
4. **流水守卫结论的依据缺口（§0.6 R3）**：「NPU 无 warp 级流水屏障、故 `block_Q<2` 无死锁」是**结构判断**（`T.Kernel.md` §2.3 + `indexer_fwd.py` 无守卫先例），非文档直述 → Stage 3 首编须验证 `num_stages>0` 下无死锁。
5. **`G>1` 路径未实测**：K 组跨步切片 `IndexK[b,t0:t0+bn,g,0:D]` 与 Q 组逐行 `T.copy` 的组合未见同族先例（`examples/` 无 G>1 indexer），列入 §9.2。
6. **bf16 累加精度**：Cube bf16×bf16→fp32；golden 在 fp32 域计算。载体由源 fp8 变 bf16——对 golden 的一致性需 Stage 3 L0 实测（bf16 mantissa 8 bit vs fp8 3 bit，精度**优于**源 fp8 路径，方向有利）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出（`ub overflow`） | block 过大 / auto-multi-buffer 膨胀 | 编译失败 | 减小 `block_N` 或 `block_Q`（§5.2 候选矩阵已按 ×1.10 预算） |
| L0C 溢出（`cc overflow`） | `block_Q*H_p*block_N > 32768` | 编译失败/segfault | 维持 §5.2 约束 `(block_Q·H_p)·block_N ≤ 32768` |
| CV 交接非法指令 / 非对齐 UUB | Developer persistent 混排 | 运行时崩溃 | 保持**非持久**（CG-2026-0010）；或转 Expert |
| `T.copy does not support element type casting`（Expert） | 跨 dtype copy（Expert 通用路径） | 编译报错 | Developer 支持跨 dtype（自动 VCast）；若转 Expert 需显式 `T.vcast`（VP-2026-0086 / `T.copy.md` §2.3） |
| `vrelu`/`vmul` bf16 编译失败 | 向量链用 bf16 | 编译失败 | **epilogue 一律 fp32**（`T.vrelu`/`T.vmul`/`T.reduce_sum` 的 bf16 均 ✗，各 API §2.2.1）；载体只用于 gemm 输入 |
| `T.Parallel` 下标变换报错 | G>1 用 `T.Parallel` gather | 前端报错 | 改 block 级 `T.copy` 切片 + `T.serial`（§1.6.2 #11；`T.Parallel.md` §2.3） |
| 尾块 NaN 污染 | `k_ub` 未初始化 + 动态尾块 | 结果错（NaN 经 gemm 扩散） | `k_ub` 循环前零初始化（PL-1.11）+ 动态 extent 切片 |
| `[M,N]*[1,N]` 静默算错 | 直接对 `out_ub [bq,bn]` 乘 `scale [1,bn]` | M==N 时静默错 | 先 `T.vbrc` 展开为 `[bq,bn]`（VP-2026-0087） |

### 9.3 特殊场景处理

- **极小 shape**：smoke `s1k-h8`（逻辑核 128，负载不均 +9.4%，§5.5）。
- **混合精度**：索引 bf16/fp16 + 累加/权重/scale/输出 fp32（固化路径）。
- **窗口全不可见 / 全可见**：`n_kv=0` 或满窗口；均由 `clean_logits_` 与块对齐超集写契约承接（§5.4）。
- **`heads` 不是 32 的整数倍**：`block_Q*H_p` M 维需分形 16 对齐；smoke `h8`（M=8×8=64 ✓）。若 `H_p<16`，需 `block_Q` 提升 M（Stage 3 按 shape 补 guard）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # D-1 等价性机器验证脚本（已产出并执行）
├── RETROSPECTIVE.md        # Stage 1 复盘
├── _fp8_lightning_indexer_kernel.py   # 算子实现（待 Stage 3）
└── history_version/        # 设计期探针/版本备份（如产生）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行（`ALL_GATE_ITEMS EQUIV_PASS`） | D-1 等价验证 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `_fp8_lightning_indexer_kernel.py` | ⬜ 待实现 | NPU kernel 实现（Stage 3） |

### 10.3 命名规范

- 项目目录：`fp8_lightning_indexer`；算子目录：`_fp8_lightning_indexer_kernel`；实现文件：`_fp8_lightning_indexer_kernel.py`（与 `.stage_state.json::kernel_py_path` 一致）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ 等价性验证（verify_equiv.py）
3. ⬜ Golden 函数（§8.1）+ NPU kernel 实现（依 §3.3 伪代码）
4. ⬜ L0 门槛测试（§8.3）→ 集成（wrapper `integrate_kernel.py` 同步 `ascend_mode="Developer"`）

---

## 11. 参考与工件索引

| 类型 | 路径 / 条目 |
|------|------------|
| 源算子 | `/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py`（L19–L149） |
| 源 Op 类 | `/home/developer/TileOPs/tileops/ops/fp8_lightning_indexer.py` |
| 抽取副本 | `examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/_f_p8_lightning_indexer_kernels.py` |
| NPU 接口契约 | `examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer.py` |
| 规格 manifest | `examples/TileOPs/tileops/manifest/attention_indexing.yaml` |
| 精度用例 | `examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py` |
| 性能用例 | `examples/TileOPs/benchmarks/ops/bench_fp8_lightning_indexer.py` |
| 同族先例 | `examples/indexer/indexer_fwd.py`；`examples/deepseek_v32/fp8_lighting_indexer.py`；`examples/deepseek_v4/example_fp8_gemm_kernel.py`；`examples/flash_attention/flash_attn_npuir_dev.py` |
| 能力缺口 | **CG-2026-0022**（`.agents/evolution/capability-gaps.md`） |
| 模式库引用 | `PL-1.22-persistent-1d-grid-stride`、`PL-1.11-causal-mask-scalartrap`、`PL-1.17-subfp32-fp32-transit`、`PL-1.27-same-family-precedent-precondition`、`PL-1.9-twophase`、`constants.md`（`CONST-cube-gemm-l0c-traffic-floor` / `CONST-vector-op-cost-910_9362` / `CONST-capacity-910B2C` / `CONST-store-fixpipe-gm-only` / `CONST-ascend910-9362-profile`）、`queue.md`（VP-2026-0086 / VP-2026-0228 / VP-2026-0087） |
| 共享标准 | `.agents/skills/_shared/standards/{algorithm-research.md, core-split-strategy.md, negative-claim-evidence.md, hardware-cost-model.md}` |
| 文档条款 | `docs/Tilelang.language/{线性代数操作/T.gemm.md, 内存操作/T.copy.md, 数学操作/T.vrelu.md, 数学操作/T.vmul.md, 规约操作/T.reduce.md, shape操作/T.vbrc.md, 数据类型转换操作/T.vcast.md, 编译器提示操作/T.Kernel.md, 编译器提示操作/T.Parallel.md, 条件操作/T.vselect.md}` |
