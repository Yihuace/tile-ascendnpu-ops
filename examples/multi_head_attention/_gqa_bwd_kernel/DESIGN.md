# `_gqa_bwd_kernel` 算子设计文档

> **算子**：`_gqa_bwd_kernel`（承载 `GQABwdKernel`，MultiHeadAttentionBwdOp backward 三件套的**主 kernel**）
> **项目**：`multi_head_attention`（NPU 侧追加进既有 `multi_head_attention` 包，不改动 fwd 产物与同批另两个函数）
> **迁移来源**：`/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`（函数定义在第 514 至 619 行，Kernel 类 `GQABwdKernel` 在第 622 至 663 行）
> **目标后端**：`target="npuir"`（Ascend910B3；物理 AI Core **20**（本机 `NPUUtils.get().get_aicore_num()` 实查）、UB **196608 B**（`AscendArch().ub_cap` 实查）、L1 512 KB、L0C 128 KB）
> **编程模式**：Developer（`op_requirements.programming_mode`；含 Expert 双 Scope 备用形态，见 §2）
> **冻结版本**：本文件为 Stage 1 交付物（**revision 1**——修订 §0.6 R3 的原子形态、§1.6.0 探针证据链与发射项记账，见文末「修订说明」）；
> `verify_equiv.py` 已执行（§1.6.1 结果表，**110 案例 O1 全覆盖**），
> `history_version/design_probe_*.py` **三枚**设计期探针已执行（§1.6.0 / §1.6.4）

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

**数学语义**：BSHD 布局的 **flash-attention 反向传播主 kernel**（GQA 通用语义，MHA = `heads_kv == heads` 的特例）。
给定前向输出统计量 `lse`（**base-2 / log2 域**的逐行 log-sum-exp）与 `delta = rowsum(o ∘ do)`，重算注意力概率并求三个梯度：

$$
\begin{aligned}
P[b,h,i,j] &= 2^{\,q[b,i,h,:]\cdot k[b,j,g(h),:]\;\cdot\;\texttt{scale}\;-\;\texttt{lse}[b,h,i]},\qquad \texttt{scale} = D^{-1/2}\log_2 e \\
dv[b,j,g(h),:] &\mathrel{+}= \sum_i P[b,h,i,j]\cdot do[b,i,h,:] \\
ds[b,h,i,j] &= P[b,h,i,j]\cdot\Big(\big(v[b,j,g(h),:]\cdot do[b,i,h,:]\big) - \delta[b,h,i]\Big)\cdot \texttt{sm\_scale},\qquad \texttt{sm\_scale} = D^{-1/2}\\
dk[b,j,g(h),:] &\mathrel{+}= \sum_i ds[b,h,i,j]\cdot q[b,i,h,:] \\
dq[b,i,h,:] &\mathrel{+}= \sum_j ds[b,h,i,j]\cdot k[b,j,g(h),:]
\end{aligned}
$$

其中 `i` 为 query 位置、`j` 为 KV 位置、`g(h) = h // groups`（`groups = heads // heads_kv`）、`(b,h)` 逐头独立。
等价地：`P = softmax_j(q·kᵀ/√D)`；`dv = PᵀdO`、`ds = P∘(dO·Vᵀ − δ)·sm_scale`、`dk = dsᵀQ`、`dq = ds·K`。

| 语义要素 | 解读（源码证据） | 是否必须保持 |
|----------|------------------|--------------|
| 数学映射 | 逐 `(b,h)` 独立的「重算 score → 三梯度累加」；`dv/dk` 沿 **i（query 位置）** 归约、`dq` 沿 **j（KV 位置）** 归约 | ✅ 必须 |
| I/O 契约 | `q/do`：`(B,S,H,D)` dtype（BSHD）；`k/v`：`(B,S,H_kv,D)` dtype；`lse/delta`：`(B,H,S)` **fp32**；`dq`：`(B,S,H,D)` **fp32**、`dk/dv`：`(B,S,H_kv,D)` **fp32**。**9 个参数全为入参**（工厂**无 `out_idx`**），`dq/dk/dv` 由 op 层 `torch.zeros(..., dtype=torch.float32)` 预置后传入（`tileops/ops/attention/gqa.py` 第 1377 至 1383 行） | ✅ 必须 |
| dtype 语义 | 输入 fp16/bf16；**乘积与累加在 fp32**（`qkt`/`dst`/`dv_frag`/`dk_frag`/`dq_frag` 均为 `accum_dtype="float"`）；P 与 ds 在送入后两个 GEMM 前**显式回落到输入 dtype**（`qkt_cast` / `dst_cast`，源码第 558/559/598/604 行） | ✅ 必须（P/dS 的输入-dtype 量化是源数值契约的一部分） |
| 规约语义 | `dv/dk` 沿 query 块（`k_idx`）累加、**循环结束后一次性写回**（`T.atomic_add`，源码第 612 至 615 行）；`dq` 每迭代清空后**立即原子累加**（第 608 至 611 行）。累加顺序 = `k_idx` 升序 | 容差内保持（见 §1.6.1 P2 / §8.2） |
| 边界语义 | `is_causal` 时 KV 块 `by` 只与 query 块 `k_idx ≥ floordiv(by·block_m, block_n)` 配对（第 580 行），且块内按 `by·block_m + i ≤ k_idx·block_n + j` 逐元素置 **P=0**（第 590 至 593 行，**置零的对象是 P 而非 score**）。`S` 非 `block_n` 整除时由 `T.ceildiv` 超发 + `T.copy` 区域裁剪承担（源未显式裁剪） | ✅ 必须（尾块须显式处理，§0.6 R6） |
| 空/极值 | 全零输入 ⇒ 全零梯度；`lse = +inf` ⇒ `exp2(-inf) = +0.0`；`lse = -inf`（非法）⇒ 上溢。无 `eps`、无 `softcap`、无 dropout | ✅ 必须 |
| host 侧语义 | **无 host 预处理**：`forward(*inputs)` 直接调用 kernel；`GQABwdKernel` 只做 `self.kernel(**self.config)(*inputs)`（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention.py` 第 476 至 506 行） | ✅ 保留 |
| `lse` 域 | **base-2（log2）域**：源用 `T.exp2(qkt·scale − lse)` 且 `scale = D^{-1/2}·LOG2E`。与本包 fwd 的 lse 契约自洽——`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` 的 golden 用 `torch.logsumexp(scores, dim=-1) / math.log(2.0)`（log2 域），其 kernel 注释明示「lse = log2-domain per-row log-sum-exp」 | ✅ 必须（**契约自洽性已在设计期核对**） |

> **语义解读完成标准核对**：§8.1 的 golden 完全按上表实现（fp32 物化 softmax + `exp2(qk·scale − lse)` + 三个矩阵乘），
> **不引用任何 NPU 侧算法结构**（不引用分块循环、不引用 tile 形状、不引用原子累加策略），验证独立性成立。
> **歧义项**：无。`lse` 域、P/dS 的 dtype 回落点、`dq`/`dv`/`dk` 的归约轴均由源码与 op 层交叉核对确定。

### 0.2 源算子输入输出规格

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `q` | 输入 | `(batch, seq_len, heads, dim)` | float16 / bfloat16 | BSHD |
| `k` | 输入 | `(batch, seq_len, heads_kv, dim)` | 同上 | GQA 时 `heads_kv < heads` |
| `v` | 输入 | `(batch, seq_len, heads_kv, dim)` | 同上 | 同上 |
| `do` | 输入 | `(batch, seq_len, heads, dim)` | 同上 | attention 输出梯度 |
| `lse` | 输入 | `(batch, heads, seq_len)` | **float32** | base-2 域逐行 log-sum-exp |
| `delta` | 输入 | `(batch, heads, seq_len)` | **float32** | `rowsum(o ∘ do)`，由已迁移的 `_flashattn_bwd_preprocess_kernel` 产出 |
| `dq` | 输入/累加 | `(batch, seq_len, heads, dim)` | **float32** | 预置 0，本 kernel 累加，随后由 `_flashattn_bwd_postprocess_kernel` 转存储 dtype |
| `dk` | 输入/累加 | `(batch, seq_len, heads_kv, dim)` | **float32** | 预置 0，本 kernel 累加，wrapper 侧 `.to(dtype)` |
| `dv` | 输入/累加 | `(batch, seq_len, heads_kv, dim)` | **float32** | 同上 |

工厂签名（源）：`_gqa_bwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype="float16")`，
经 `@functools.lru_cache(maxsize=32)` 包装；返回 `_gqa_bwd_func(block_m, block_n, num_stages, threads)`，
其 `@tilelang.jit(pass_configs={TL_ENABLE_FAST_MATH: True}, compile_flags=["-O3", "-DENABLE_BF16"])` 装饰器（第 528 至 532 行）
在 NPU 侧**不适用**（npuir 的 `tilelang.jit` 无 `compile_flags` 形参）。NPU wrapper（Stage 0 冻结）调用
`_gqa_bwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype_str)`，`default_config` 为
`{"block_m": 64, "block_n": 64 if dim <= 64 else 32, "num_stages": 1}`（**已按 K9 删除 `threads`**，K8 删除 autotune）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源码第 550 至 615 行，逐语句归位，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|------|------|------|------|----------|
| S0 | 三维 grid `T.Kernel(heads, ceildiv(seq_len, block_m), batch) as (bx,by,bz)`，`bx`=**query head**、`by`=**KV 块**、`bz`=batch | — | block 索引 | 并行结构（非数学语义） |
| S1 | `T.copy(k[bz, by·bm:(by+1)·bm, bx//groups, :], k_shared)`；`v` 同形 → `v_shared` | `k`,`v` | L1/共享缓冲 | 读本任务的 K/V 块 |
| S2 | `T.clear(dv_frag)`；`T.clear(dk_frag)` | — | `dv=0`,`dk=0` | 累加初值 |
| S3 | `loop_st = floordiv(by·bm, bn) if is_causal else 0`；`loop_ed = ceildiv(S, bn)` | — | — | **causal 下跳过全掩码 query 块**（FLOPs 减半） |
| S4 | `for k_idx in T.Pipelined(loop_st, loop_ed, num_stages)`：`T.copy(q[...], q_frag)` | `q` | `q_frag [bn,dim]` | 读 query 块 |
| S5 | `T.clear(qkt)`；`T.gemm(k_shared, q_frag, qkt, transpose_B=True)` | `k_shared`,`q_frag` | `qkt [bm,bn]` fp32 | `qkt = K·Qᵀ`（0.1 的 `q·kᵀ` 转置形态） |
| S6 | `qkt[i,j] = T.exp2(qkt[i,j]·scale − lse_shared[j])` | `qkt`,`lse` | `qkt`（=**P**） | **base-2 域概率重算** |
| S7 | `if is_causal:` `qkt[i,j] = if_then_else(by·bm+i ≤ k_idx·bn+j, qkt[i,j], 0)` | — | `qkt` | **因果掩码（置 P=0）** |
| S8 | `T.copy(do[...], do_shared)`；`T.clear(dst)`；`T.gemm(v_shared, do_shared, dst, transpose_B=True)` | `v`,`do` | `dst [bm,bn]` fp32 | `dst = V·dOᵀ` |
| S9 | `T.copy(qkt, qkt_cast)`；`T.gemm(qkt_cast, do_shared, dv_frag)` | P,`do` | `dv_frag [bm,dim]` fp32 | `dv += Pᵀ·dO`（**P 回落到输入 dtype**） |
| S10 | `T.copy(delta[...], delta_shared)`；`dst_cast[i,j] = qkt[i,j]·(dst[i,j]−delta_shared[j])·sm_scale` | P,`dst`,`delta` | `dst_cast [bm,bn]` dtype | **ds**（并回落到输入 dtype） |
| S11 | `T.gemm(dst_cast, q_frag, dk_frag)` | ds,`q` | `dk_frag [bm,dim]` fp32 | `dk += dsᵀ·Q` |
| S12 | `T.copy(dst_cast, dst_shared)`；`T.clear(dq_frag)`；`T.gemm(dst_shared, k_shared, dq_frag, transpose_A=True)` | ds,`k` | `dq_frag [bn,dim]` fp32 | `dq = ds·K` |
| S13 | `T.atomic_add(dq[bz, k_idx·bn+i, bx, j], dq_frag[i,j])`（`T.Parallel(bn, dim)` 逐元素） | `dq_frag` | `dq` | **dq 跨 KV 块累加** |
| S14 | `T.copy(dv_frag, dv_shared)`；`T.atomic_add(dv[bz, by·bm:(by+1)·bm, bx//groups, :], dv_shared)` | — | `dv` | dv 一次性写回 |
| S15 | 同 S14，`dk_frag → dk_shared → T.atomic_add(dk[...])` | — | `dk` | dk 一次性写回 |

**数据流图（源硬件视角，CUDA）**：

```
GM[k,v (B,S,Hkv,D)] --T.copy(跨步 slice, 行 stride=Hkv·D)--> shared[k_shared,v_shared (bm,dim) dtype]
GM[q,do (B,S,H,D)]  --T.copy(跨步 slice, 行 stride=H·D)-----> shared[q_frag/do_shared (bn,dim) dtype]
   k_shared × q_fragᵀ --T.gemm(mma, fp32 累加)--> frag[qkt (bm,bn) fp32]
   qkt --exp2(qkt·scale − lse)--> frag[P]  --(causal: if_then_else 置 0)--> frag[P]
   v_shared × do_sharedᵀ --T.gemm--> frag[dst (bm,bn) fp32]
   P --T.copy(cast 到 dtype)--> frag[qkt_cast] --T.gemm--> frag[dv_frag (bm,dim) fp32]   (跨 k_idx 累加)
   P,dst,delta --T.Parallel(逐元素)--> frag[dst_cast (bm,bn) dtype] --T.gemm--> frag[dk_frag] (跨 k_idx 累加)
   dst_cast --T.gemm(transpose_A)--> frag[dq_frag (bn,dim) fp32] --T.atomic_add--> GM[dq]
   GM[dv] / GM[dk] <--T.atomic_add(一次性)--- dv_frag / dk_frag
```

**循环与并行结构**：grid = `(heads, ceildiv(S,block_m), batch)` 三维；`threads` 默认 128（Kernel 类 `default_config`，本函数调用时由 wrapper 传入）；
block 内 `T.Parallel(block_m,block_n)` 做逐元素链、`T.Parallel(block_n,dim)` 做 dq 的原子散写；
query 块循环由 `T.Pipelined(start, stop, num_stages)` 承载（runtime 界，causal 时 `start` 依赖 `by`）。
**无跨 block 协作、无 workspace、无核间 flag。**

**host 侧逻辑**：无。op 层在调用前只对 `do` 显式调 `.contiguous()`，`dq/dk/dv` 由 `torch.zeros_like(..., dtype=torch.float32)` 预置；`q/k/v` 的连续性由调用方（fwd 路径）保证（`tileops/ops/attention/gqa.py` 第 1377 至 1383 行）。
**`make_dq_layout`**：模块级 helper（`_multi_head_attention_bwd_kernels.py` 末尾逐字保留），
`T.annotate_layout({dq: make_dq_layout(dq), ...})`（第 569 至 573 行）——它是 **Ampere 8×8 fragment 重排**
（GPU 源文件 `gqa_bwd.py` 第 99 至 103 行的 helper 注释原文：「atomicAdd cannot be vectorized on Ampere,
need to reorder dq to match the 8x8 gemm fragment」），**不是 host 逻辑**，是 GPU 硬件强相关的实现层手段（§0.5 舍弃）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 代码特征 | 目的 | 依赖的源硬件特性 | 迁移初判 |
|---|----------|----------|------|------------------|----------|
| 1 | **SMEM tiling + 跨 k_idx 复用** | `k_shared`/`v_shared` 每任务装载一次、在 query 块循环外 | K/V 只从 GM 读一次 | CUDA shared memory | 硬件强相关（NPU 等价 = L1） |
| 2 | **fragment 上 fp32 累加器** | `dv_frag`/`dk_frag` fp32，跨 k_idx 循环累加 | 低精度域不反复舍入 | 大寄存器堆 | 硬件强相关（NPU = UB fp32 载体） |
| 3 | **单趟融合三梯度（不重算）** | 同一 query 块循环内同时算 dv/dk/dq | **避免第二遍重读 Q/K/V/dO**（FA2 的 dk/dv 与 dq 两个 kernel 需重算 P） | 无（算法层） | **保留意图**（省 FLOPs 与读流量，是本设计主选的核心依据，§1.6.0 R3） |
| 4 | **Tensor Core mma** | 5 处 `T.gemm`（含 `transpose_B` / `transpose_A`） | 矩阵算力 | tensor core | 等价替换（NPU Cube，§0.5） |
| 5 | **异步流水 / 多级拷贝** | `T.Pipelined(loop_st, loop_ed, num_stages)` | 搬运与计算重叠 | `cp.async` | 等价替换（NPU = MTE2 + 自动多缓冲） |
| 6 | **causal 块级跳过** | `loop_st = floordiv(by·bm, bn)` | 全掩码块不参与计算 | 无（算法层） | **保留** |
| 7 | **块内 causal 掩码** | `if_then_else(by·bm+i ≤ k_idx·bn+j, qkt, 0)`（`T.Parallel` 逐元素） | 对角块置 P=0 | CUDA 线程级谓词 | **重新设计**（NPU 逐元素条件选择会标量化，§0.6 R4） |
| 8 | **P/dS 回落到输入 dtype** | `T.copy(qkt, qkt_cast)` / `dst_cast` | 让后两个 GEMM 用 tensor core 的 fp16/bf16 输入 | tensor core 输入必须是 fp16 | **保留**（数值契约，NPU Cube 同样要求低精度操作数） |
| 9 | **原子累加（`T.atomic_add`）** | dq 每迭代逐元素原子加；dv/dk 任务末一次性原子加 | 跨 block 归约免第二遍 kernel | 全局原子 + `make_dq_layout` 向量化 | **等价替换**（NPU 用**逐行 1-D 行原子**，探针三实测正确；二维区域形态不可用、`make_dq_layout` 舍弃，§0.5/§0.6 R3） |
| 10 | **三维 grid 并行** | `T.Kernel(heads, ⌈S/bm⌉, batch)` | 任务级并行 | 无 | **重新设计**（NPU 一维 Kernel + persistent，§0.6 R1） |
| 11 | **swizzle / `make_swizzled_layout`** | `annotate_layout` 于 `dv_shared`/`dk_shared` | 消 SMEM bank conflict | SMEM bank 结构 | **舍弃**（NPU 无 bank conflict 概念；cube skill 明示 NZ 路径为 Expert-only，Developer 保持 ND） |
| 12 | **`threads` / `T.Parallel` 元素级并行** | `T.Parallel(block_m,block_n)`、`T.Parallel(block_n,dim)` | 元素级并行 | CUDA 线程 | **等价替换**（NPU = 整 tile v-prefix 向量 op，§1.6.2） |

> **未识别 ≠ 不存在**：本工厂源码 66 行已逐语句归类（S0 至 S15 + 上表 12 项）。与文件内其它工厂相比，本函数**没有**
> wgmma、**没有** `T.load_nd2nz`、**没有** `T.store_fixpipe`、**没有** workspace/flag 同步、
> **没有** online softmax 的 running max/rescale（`lse` 已外部给定）——这些不属于本函数的优化手段，迁移时不得误搬。

### 0.5 硬件耦合性分析与 NPU 适配决策

三层判定模型（`migration-analysis.md` §5.1）+ 四态处置（§5.2）。判定输入含 §1.6.0 调研结论。
逐项依据见「依据」列（映射表条目 / `examples/` / `docs/` / pattern-library / 本设计期探针）。

| 项 | 层 | 源方案 | 处置 | 依据 |
|----|----|--------|------|------|
| 数学映射 / I/O 契约 / fp32 累加域 / P·dS 的输入-dtype 回落 / 边界语义 | 语义层 | §0.1 | **保留** | `migration-analysis.md` §5.1（语义层无条件保留） |
| `lse` 的 base-2 域与 `scale = D^{-1/2}·LOG2E` | 语义层 | 源码第 521 至 522 行 | **保留** | 与本包 fwd 的 lse 契约自洽（`_gqa_prefill_fwd_kernel.py` golden 的 `logsumexp/log(2)`） |
| **单趟融合 dv/dk/dq（不重算 P）** | 算法层 | 同一 query 循环内三梯度 | **保留** | §1.6.0 R3：两趟（FA2 式）方案的迭代数与向量 pass 数均为本方案的 2 倍，而本算子落在**发射受限**区（§1.6.0 设计期估算） |
| causal 块级跳过（`loop_st`） | 算法层 | 全掩码块不计 | **保留** | 纯数学分块、与硬件无关 |
| 三维 grid `T.Kernel(heads, ⌈S/bm⌉, batch)` | 优化层 | 三维 block | **重新设计** | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1（`is_npu` 最多 2 维 block）+ 映射表「三维 Kernel → 一维展开」+ 分核标准 §1 要素③（极大规模须固定核数 + 核内 `T.serial`） |
| `T.alloc_fragment` 计算缓冲 | 优化层 | CUDA 寄存器/fragment | **等价替换** | 映射表「register 累加 → UB 中间缓冲」；`docs/Tilelang.language/数学操作/T.vmul.md` §2.3 第 1 条（v 算子操作数必须在 UB） |
| `threads=128` | 优化层 | CUDA 线程数 | **舍弃** | K9 已从 wrapper config 删除；`TRAP-threads-kwarg-noop`（`T.Kernel(threads=)` 在 npuir 无效） |
| `T.gemm`（5 处，含 tanspose_A / transpose_B） | 优化层 | tensor core mma | **等价替换** | 映射表「Tensor Core mma → Cube 单元（`T.gemm` + L0A/L0B/L0C）」；Developer 形态先例 `examples/deepseek_v32/sparse_mla_bwd.py`（`T.gemm(..., a_transpose=True/b_transpose=True)` + `T.vexp2` + `T.atomic_addx4`，`target="npuir"` + `is_npu=True`） |
| `T.alloc_shared`（k/v/q/do 装载目标） | 优化层 | SMEM | **等价替换** | `docs/Tilelang.language/内存操作/T.alloc_shared.md` §2.4 示例 1（`T.alloc_shared` 用于 matmul ⇒ 「申请了 Ascend L1 内存」）；cube skill「Developer mode 用 `T.alloc_shared`」 |
| `make_swizzled_layout`（dv_shared/dk_shared） | 优化层 | 消 bank conflict | **舍弃** | 映射表「swizzle 防 bank conflict → NZ 分形布局，手动 swizzle 无意义」；cube skill「NZ format path is Expert mode only；In Developer mode kernels, keep ND layout」 |
| `make_dq_layout`（Ampere 8×8 fragment 重排）+ `annotate_layout(dq)` | 优化层 | 让 CUDA 逐元素 `atomicAdd` 向量化 | **舍弃** | 该 layout 是 CUDA fragment 的 8×8 重排（GPU 源 `gqa_bwd.py` 第 99 至 103 行的 helper 注释明示「atomicAdd cannot be vectorized on Ampere」）；NPU 侧该角色由**逐行 1-D 行原子**承担（`T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r, 0:dim], [1, dim])`；`history_version/design_probe_row_atomic_bshd.py` 实测：设计几何下目标区元素全中、区外 0、20 并发同址精确 20.0） |
| `T.Pipelined` 异步流水 | 优化层 | `cp.async` 多级流水 | **等价替换** | 映射表「cp.async → `T.Pipelined` + MTE 异步搬运」；stage 数须按 UB 容量重算（§4.5） |
| 跨 k_idx 的 fragment 累加（`dv_frag`/`dk_frag`，`initC=False`） | 优化层 | register 累加 | **保留** | 本工具链已验证形态：`examples/deepseek_v32/sparse_mla_bwd.py` 的 `acc_dq`（`alloc_shared` + 循环外 `T.clear` + 循环内 `T.gemm(initC=False)`） |
| dq 逐元素 `T.atomic_add`（`T.Parallel(bn,dim)` 标量原子） | 优化层 | 逐元素全局原子 | **重新设计** | NPU 用**逐行 1-D 行原子** `T.atomic_add(dst_row, src_row, [1, dim])`（`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.2「对形状无特殊限制」+ §2.4 区域形态；生产先例 `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` 第 257 行 `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])`）；逐元素形态会标量化（§1.6.2）。**二维区域形态（rank-reduced 4-D 目标）实测不可用**（REVIEW.md 问题 1；`design_probe_row_atomic_bshd.py` 正面验证修正形态） |
| 因果掩码 `if_then_else`（`T.Parallel` 逐元素） | 优化层 | CUDA 线程谓词 | **重新设计** | NPU 逐元素条件选择形态实测标量化（`TRAP-parallel2d-bm-ge2-scalarize`，含 `T.if_then_else` 融合条件选择）；改用指数域惩罚（`attention.md` PL-1.11，§0.6 R4） |
| swizzle `annotate_layout` 之外的 `fragment` 直取 | — | 本函数**未使用** | 不适用 | §0.3 |
| `pass_configs={TL_ENABLE_FAST_MATH}` / `compile_flags` | 宿主层 | CUDA 编译开关 | **舍弃** | npuir 的 `tilelang.jit` 无 `compile_flags` 形参；本设计不使用 `pass_configs`（保持与同批前作一致的最小面） |
| `supported_archs=[80,89,90]` | 宿主层 | CUDA SM 列表 | **舍弃** | wrapper 已按 K5 改写 `supported_archs = None`（`multi_head_attention.py` 第 453 至 454 行） |

**判定统计**（上表逐行计数，共 **18** 行）：保留 **5** 项（语义层 2：数学映射/I-O 契约/fp32 域/边界语义 + `lse` 的 base-2 域；算法层 2：单趟融合三梯度 + causal 块级跳过；优化层 1：跨 `k_idx` 的 `fragment` 累加形态）、
等价替换 **4** 项（`alloc_fragment`/`T.gemm`/`alloc_shared`/`T.Pipelined`）、重新设计 **3** 项（三维 grid / dq 逐元素原子 / 因果掩码）、
舍弃 **5** 项（`threads` / `make_swizzled_layout` / `make_dq_layout` / `pass_configs`+`compile_flags` / `supported_archs`）、不适用 **1** 项。
**舍弃项均有理由**（见上表）；**无静默丢弃**（§0.6 末尾逐项承接核对）。

### 0.6 NPU 算法重设计

对 §0.5 判为「重新设计」的 3 项逐项给出「源方案 → NPU 新算法 → 语义保持论证」；另给出 NPU 适配必需的 4 项配套设计（R4 掩码 / R5 原子 / R6 尾块 / R7 向量化）。
重设计直接承接 §1.6.0 调研结论（**单趟融合三梯度 + UB fp32 累加 + 逐行 1-D 行原子 + 一维 persistent 分核**）。

**R1（三维 grid → 一维 persistent）**

- 源方案：`T.Kernel(heads, ceildiv(S,bm), batch)`，每个 block 一个 `(query head, KV 块, batch)` 任务。
- NPU 新算法：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(num_logical, 20)`；
  逻辑任务 `uid = i·num_kernels + cid`（`i` 遍历 `T.serial(num_local_tasks)`，静态界），
  `if uid < num_logical` 守卫；`uid` 用整除/取模分解回 `(bz, bx, by)`（§5.5）。
- 语义保持论证：仅改变**任务到核的映射**；每个任务内部的 `(b,h,KV 块)` 计算与累加顺序完全不变；
  `(b,h)` 之间、KV 块之间相互独立（`dq` 的跨 KV 块归约由原子完成，与源同构）⇒ 任务顺序无关。
  `uid → (bz,bx,by)` 为双射，无重复/遗漏（`uid = i·num_kernels + cid` 覆盖 `[0, num_logical)` 全部取值，越界由守卫跳过）。
  **探针证据**：`history_version/design_probe_persist_and_strided.py` 的 Q3 在本工具链（910B3 + CANN 9.0.0 + Developer）
  实测「固定核数 `T.Kernel(n_kernels)` + 核内 `T.serial(n_tasks)` + `T.gemm` + `v` 前缀算子弹」**编译通过且数值正确**
  （`n_kernels=4/n_tasks=8` 与 `n_kernels=20/n_tasks=4` 两档，`max_abs_err = 4.578e-05`）；
  ⇒ pattern-library `TRAP-DEVMODE-PERSIST-GEMM`（910B2C + CANN 8.5.0，2026-09-17）在**本工具链不成立**
  （按证伪协议 §2「工具链变更后自动待重验」，本设计以本机探针为准，并把该重验结论如实登记，见 §9 R7）。

**R2（tile 形态 → UB 预算驱动）**

- 源方案：`(block_m, block_n)` 方形 tile（`k_shared`/`v_shared` `(bm,dim)`、`q_frag`/`do_shared` `(bn,dim)`、
  `qkt`/`dst` `(bm,bn)` fp32、`dv_frag`/`dk_frag` `(bm,dim)` fp32、`dq_frag` `(bn,dim)` fp32），
  `default_config = (64, 64|32, 1)`。
- NPU 新算法：**沿用源 tile 形态但改为 UB 预算驱动**——`bm = block_m`（KV 行数）、`bn = block_n`（query 行数），
  按 §4.5 逐 buffer 复算 UB；`(bm, bn) = (64, 64)`（`dim ≤ 64`）或 `(64, 32)`（`dim = 128`）即 wrapper 默认值，
  另有 `(64,64)@dim=128` 上探档（§1.6.3 参数级 A/B）。
- 语义保持论证：tile 只切分 **query 轴** 与 **KV 轴** 的位置，不改变任何元素的计算与分组：
  每个 `P[b,h,i,j]` 仍是同一次 `q_i·k_j` 点积 + 同一次 `exp2`；`dv/dk` 沿 query 的累加集合与源相同（同一 `(b,h,j)` 行对全部 i 求和）；
  `dq` 沿 KV 的累加集合相同（原子顺序无关）。**`dim` 轴不切分**（整行驻留）⇒ GEMM 的 K 维（`dim`）与归约集合与源一致。
  fp32 累加域不变。`bm/bn` 变化只改变累加**结合序**（§8.2 容差内，与源同为 fp32 累加链）。

**R3（原子累加形态 → 逐行 1-D 行原子 + 工厂期分派）**

- 源方案：`dq` 用 `T.Parallel(block_n, dim)` 的**逐元素** `T.atomic_add(dq[bz, i, bx, j], dq_frag[i,j])`，
  并用 `make_dq_layout` 重排 dq 的 fragment 布局让 CUDA 的逐元素原子能向量化；`dv/dk` 用 `T.copy` 到 shared 后
  `T.atomic_add(区域, src)`。
- NPU 新算法（**revision 1 修订形态**）：统一为**逐行 1-D 行原子**（`r` 沿区域行步进）：
  `for r in T.serial(real_n): T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r, 0:dim], [1, dim])`
  （`dv`/`dk` 的行数为 `real_m`）；目标行以**标量**索引（`bz`/`c0+r`/`bx`）+ 尾轴切片构成 1-D 区域，
  与生产先例 `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` 第 257 行同形态。
  `groups == 1`（MHA）时 `dk/dv` 每元素只有一个写者，在**工厂期 trace-time 分派**为普通区域 `T.copy` 存储（免 RMW），
  `groups > 1` 时保持逐行原子。
  **为什么不用二维区域形态（revision 0 的形态）**：本机实测该 rank-reduced 4-D 形态**不可落地**——
  ① 动态 extent（`real_n` 为 runtime Var）⇒ 前端 `DiagnosticError: int() argument must be ... not 'Var'`，**首编即失败**；
  ② 静态 extent ⇒ 目标被当作**连续平铺块**处理（行 stride = 区域宽 `dim` 而非张量真实 stride `H·D`）⇒ **静默写错区域**
  （设计自身 manifest 几何 (1,128,8,64) 目标区仅写 12.5%、区外写入 1792 cell；(1,64,16,128) 仅 6.2% / 1920 cell），
  且 `bx = H−1` 时连续块越过张量尾部（越界写相邻 GM 分配）；
  ③ rank-4 `size` 变体触发编译器内部断言。见 REVIEW.md 问题 1 与本设计的 §1.6.0 探针表。
- 语义保持论证：逐行行原子与逐元素原子是**同一个数学操作**（同目标元素集合、同加数、同一 fp32 加），
  只是把 `bn`（或 `bm`）个行操作串行化；`groups == 1` 时用普通 store 等价，依据是「每个 KV 块只被一个任务写」——
  任务的 `by` 与 `bx//groups` 唯一确定 `dk[bz, by·bm:(by+1)·bm, bx//groups, :]` 的归属，
  `groups==1 ⇒ bx//groups = bx` 与 `by` 一起构成双射，
  而 op 层已把 `dk/dv` 预置为 0（`torch.zeros`）⇒ 「对 0 的单写者原子加」≡「普通写」。
- **探针证据（revision 1 更换证据源，证据链与工件形态逐条对齐）**：
  ① `history_version/design_probe_atomic_vs_store.py`：二维区域 `T.atomic_add`（fp32，**连续** 2-D 区域）
  数值逐位正确且耗时 **0.97×** 同体积普通 store（126.60 vs 130.02 µs / 8.4 MB，2026-10-16 重跑）——该结论**仅适用于连续 2-D 区域 /
  rank-2 双侧显式切片**，不外推到 BSHD rank-reduced 形态；
  ② `history_version/design_probe_row_atomic_bshd.py`（本机 910B3 + CANN 9.0.0 + Developer，**revision 1 新增**）：
  **修正形态（逐行 1-D 行原子）在 BSHD 目标上的正面验证**——目标区元素计数全中、区外元素计数 0（含 `h = H−1` 档）、
  `T.serial(T.min(bn, S−row0))` 动态行数正确、**20 个 block 并发同址写同一批行 ⇒ 每 cell 精确 20.0**；
  ③ `history_version/design_probe_persist_and_strided.py` Q1/Q2：**rank-2 张量双侧显式切片**的跨步区域
  `T.copy`/`T.atomic_add` 逐位正确（revision 1 已修正其期望值判据，见 §1.6.0）——**不能外推到 BSHD rank-reduced 形态**
  （这正是 revision 0 的漏检根因，见 §9 R1）。
  ⇒ 「原子 = 读-改-写 = 2× store」的纸面模型被实测推翻（降级为 `hivm::StoreOp`，见 `T.atomic_add.md` §3）。

**R4（因果掩码 → 指数域惩罚 + 掩码平面循环不变量提升）**

- 源方案：`qkt[i,j] = T.if_then_else(by·bm + i ≤ k_idx·bn + j, qkt[i,j], 0)`（`T.Parallel` 逐元素条件选择）。
- NPU 新算法：把「掩码」从 **P 的逐元素条件选择**改为 **指数域惩罚**：
  `P = exp2(qkt·scale − lse + pen)`，其中 `pen = min(−PEN·(j − i − d0), 0)`（**单一定义源 = §3.3 伪代码**；§1.4/§3.2 引用同一行，临时缓冲 `t ≡ dsum`）、
  `d0 = k_idx·bn − by·bm`（每迭代一个标量）、`(j − i)` 平面**每任务物化一次**（循环不变量提升，CSE）；
  `PEN = 1e30`（有限大数）。
- 语义保持论证：`valid ⟺ j − i ≤ d0`（与源条件 `by·bm+i ≤ k_idx·bn+j` 代数等价：两边同减 `by·bm+i` 得 `0 ≤ k_idx·bn − by·bm + j − i`）；
  valid 时 `pen = 0` ⇒ **逐位恒等**；invalid 时 `pen ≤ −1e30` ⇒ `exp2(有限负数)` 在 fp32 下**下溢为精确 +0.0**，
  与源「置 0」逐位一致。**D-1 机器验证**：30 案例（含 `(bm,bn) ∈ {(64,64),(64,32),(32,32),(16,64),(1,1),(7,5)}` × 5 种分数分布）
  **最大 ulp 差 0、超容差 0**（§1.6.1 O1）。
  **边界语义差异（诚实登记）**：若某 masked 位置的 **score 为非有限（NaN）**，惩罚式会传播 NaN 而源的
  `if_then_else` 仍给 0。该差异域在**有限输入下为空**（q/k/v/do 有限 ⇒ 点积有限；`lse` 由前向 logsumexp 得有限），
  故不影响语义；§8.3 的边界用例把该差异显式登记为「记录不断言」。备选精确形态（`vcmp` + `vselect`）见 §1.6.4。
- **循环不变量提升依据**：`(j − i)` 与 `k_idx` 无关 ⇒ 每任务 1 次物化取代每迭代 1 次构造，
  迭代数 `S/bn`（manifest 下 32 至 64）倍的索引构造被消除（§1.6.2 #7）。

**R5（`groups` 的分头语义）**

- 源方案：grid 的 `bx` 是 **query head**，`dk/dv` 用 `bx // groups` 索引 KV head ⇒ GQA 下多个 query head 写同一 `dk/dv` 元素（原子）。
- NPU 新算法：保留 `bx`=query head 的映射（**与源同构**，`dq`/`lse`/`delta` 都按 query head 索引）；
  `dk/dv` 的冲突由 R3 的工厂期 `groups` 分派处理（MHA 普通 store / GQA 原子）。
- 语义保持论证：分头索引 `g(h) = h // groups` 与源逐字一致；`groups == 1` 时退化为 MHA 语义（manifest 全部是 MHA）。

**R6（尾块：非整除 `S`）**

- 源方案：依赖 `ceildiv` 超发 + `T.copy` 区域裁剪（隐式），未显式裁剪。
- NPU 新算法：**显式动态边界**——对 KV 轴 `real_m = T.min(bm, S − by·bm)`、对 query 轴 `real_n = T.min(bn, S − k_idx·bn)`；
  装载用双显式 slice（`src[..., 0:real_m, ...]` 与 `dst[0:real_m, ...]`）；
  GEMM 用 `size=[real_m, real_n, dim]` 形式的尾块裁剪（或保留完整 tile 计算、写出时裁剪）；
  `dq` 的**逐行行原子行数**用 `real_n`（`T.serial(real_n)` 的界）、`dv/dk` 的写出区域用 `real_m`（`T.copy` 裁剪或逐行行原子行数）。
- 语义保持论证：输出元素集合与源一致（`[by·bm, min((by+1)·bm, S))` 与 `[k_idx·bn, min((k_idx+1)·bn, S))`）；
  垃圾行/列永不落 GM；GEMM 的尾块裁剪由 `size=` 的 fractal 下界处理（`TRAP-L1-band-dst-tail-overrun` 的教训：
  src/dst 列宽须同步裁剪）。与同族前作同款形态（`examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/DESIGN.md` §0.6 R5）。

**R7（元素级并行 → v-prefix 向量链）**

- 源方案：`T.Parallel(block_m,block_n)` 逐元素链与 `T.Parallel(block_n,dim)` 逐元素原子散写。
- NPU 新算法：全部替换为**整 tile v-prefix 算子**（`T.vmul` / `T.vsub` / `T.vmax` / `T.vexp` / `T.vcast` / `T.atomic_add` 区域形态）；
  盘点与逐项理由见 §1.6.2。
- 语义保持论证：逐元素映射不变（无跨元素依赖；`lse`/`delta` 是**行广播** `[bn,1]`→`[bn,bm]`，
  `T.vmul`/`T.vsub` 文档化的行广播形态 `[M,N]*[M,1]` 直接支持，无需 `vbrc` 展开）；
  唯一形态变化是把「逐元素条件选择」改为「指数域惩罚」与「逐元素原子」改为「**逐行 1-D 行原子**」（R3/R4 已论证）。

**源算法优化手段承接核对（§0.4 逐项，无静默丢弃）**：

| §0.4 项 | 意图 | 承接方式（→ 本设计） |
|---------|------|---------------------|
| #1 SMEM tiling 跨 k_idx 复用 | K/V 每任务只读一次 | 承接：`k_s`/`v_s` 在任务内装载一次、置于 L1（gemm 操作数），query 循环只重读 Q/dO（§4.3） |
| #2 fragment fp32 累加 | 低精度不反复舍入 | 承接：`dv_acc`/`dk_acc`/`dq_blk`/`qkt`/`dsum` 全 fp32 UB 载体（§4.3） |
| #3 单趟融合三梯度（不重算） | 省 FLOPs 与读流量 | **原样保留**（§1.6.0 R3 选定本方案、否决两趟方案） |
| #4 Tensor Core mma | 矩阵算力 | 等价替换为 Cube `T.gemm`（5 处），`transpose_B`/`transpose_A` 改名为 `b_transpose`/`a_transpose`（§3.2） |
| #5 异步流水 | 搬运/计算重叠 | 等价替换为 `T.Pipelined(..., num_stages=ns)`，stage 数按 UB 重算（§4.5、§6） |
| #6 causal 块级跳过 | FLOPs 减半 | **原样保留**（`loop_st`） |
| #7 块内 causal 掩码 | 对角块置 P=0 | **重新设计**（R4：指数域惩罚 + 循环不变量提升） |
| #8 P/dS 回落输入 dtype | tensor core 低精度操作数 | **原样保留**（R7 的 `T.vcast(..., round_mode="rint")`，与源 fragment cast 同为就近舍入） |
| #9 原子累加 | 跨 block 归约免第二遍 kernel | **等价替换**（R3：**逐行 1-D 行原子**；MHA 分派为普通区域 store） |
| #10 三维 grid 并行 | 任务级并行 | **重新设计**（R1：一维 persistent） |
| #11 swizzle | 消 bank conflict | **舍弃**（NPU 无该概念，§0.5） |
| #12 线程级元素并行 | 元素级并行 | 等价替换为 v-prefix 向量链（R7） |

### 0.7 标杆实现（迁移参考）

| 参考 | 相似度 | 关键参考点 |
|------|--------|-----------|
| `examples/deepseek_v32/sparse_mla_bwd.py` | **极高（同数学语义 + 同编程模式的 npuir 实现）** | `preprocess()`/`bwd()`：`@tl.jit(target="npuir")` + `T.Kernel(N, is_npu=True)`（**无 `T.Scope`、Developer 模式**）+ `T.alloc_shared` 承载 gemm 操作数与累加器 + `T.gemm(..., a_transpose=True/b_transpose=True)` + `T.vexp2(src,dst,tmp)` + `T.reshape` + `T.atomic_addx4(区域, ...)` + `acc_dq` 跨 `T.Pipelined` 循环累加。**证实本设计全部原语组合在 `target="npuir"` 上可编译可跑** |
| `examples/deepseek_v4/example_sparse_attn_bwd_kernel.py` | 极高（同语义，Developer） | 同款 attention-bwd 形态（`T.gemm` ×5 + `T.serial` + `T.atomic_add` 行散写）；bwd 族的最小可用骨架 |
| `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` | 高（同族，Developer） | `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])`（**动态下标的 1-D 行原子**）+ `T.vadd`/`T.vmul`/`T.vexp2` 链 + `T.serial(split_store)` 分片降寄存器压力 |
| `examples/elementwise/atomic_add.py` / `atomic_add_dev.py` | 高（**二维区域原子**） | Developer 形态 `T.atomic_add(B[bx, by], A_shared, [tile_M, tile_N])` 的官方示例（连续 2-D 区域） |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 极高（**同包 fwd，lse 契约 + 闭包工厂先例**） | ① lse 为 **log2 域**（golden `logsumexp/log(2)`）；② 工厂**返回 `wrapped(...)` 闭包**（不是裸 jit callable）——本设计的 §1.6.4 备选形态有先例；③ `pass_configs` 关闭 buffer 重排（Expert 侧） |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention.py` | 极高（**wrapper 契约**） | `GQABwdKernel` 构造（7 位置参数）与 `forward` = `self.kernel(**self.config)(*inputs)`；`default_config = {block_m, block_n, num_stages}`（K9 已删 `threads`）；`supported_archs = None` |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/_multi_head_attention_bwd_kernels.py` | 极高（**Stage 0 提取的迁移源工件**） | 第 128 至 233 行为本函数（与 GPU 源逐字一致）+ 末尾 `make_dq_layout` 逐字副本；作为 §0.1/§0.3 的对照基准 |
| `examples/TileOPs/tests/ops/test_multi_head_attention.py` | 高（**Stage 5 验收口径**） | `MhaBwdFixture` 的 manifest 四档 shape（§5.5/§8.3 直接采用）；op 级容差 `atol=5e-3, rtol=1e-5`；roofline 断言 `flops = 10·B·H·S²·D`（causal 减半）、`bytes = B·7·H·S·D·elem_bytes` |
| `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/DESIGN.md` | 高（**同批前作设计文档**） | §0 迁移分析结构、§1.6 调研/优化写法、§4.5 UB 双口径、§5.5 分核三要素、§8 L0 计划；其「Transferable Lessons」见 §4.5/§3.3 与 `RETROSPECTIVE.md` |
| `/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py` | —（**源算子**） | 第 514 至 619 行（工厂 + kernel）、第 622 至 663 行（Kernel 类）；`tileops/ops/attention/gqa.py` 第 1377 至 1383 行（三件套调用链 + fp32 预置） |

---

## 1. 概述

### 1.1 算子名称

`_gqa_bwd_kernel`（承载 `GQABwdKernel`：MultiHeadAttentionBwdOp 的 backward 主 kernel，GQA 通用语义、MHA 为 `heads_kv == heads` 特化）

### 1.2 功能描述

在 BSHD 布局下重算注意力概率 `P = exp2(q·kᵀ·scale − lse)`（`lse` 为 base-2 域）并求三个梯度：
`dv = Pᵀ·dO`、`dk = dsᵀ·Q`、`dq = ds·K`，其中 `ds = P∘(dO·Vᵀ − delta)·sm_scale`；
`dq/dk/dv` 为 **fp32 累加器**（由 wrapper 预置 0 后传入，`dq` 由已迁移的
`_flashattn_bwd_postprocess_kernel` 转回存储 dtype），`dv/dk` 由 wrapper 侧 `.to(dtype)`。
**NPU 侧实现与源算法的差异**（§0.6 / §1.6.0）：数学内核与单趟融合结构不变，改的是
**并行结构**（三维 grid → 一维 persistent）、**tile 尺寸**（UB 预算驱动）、
**原子形态**（逐元素原子 → **逐行 1-D 行原子**；MHA 分派为普通区域 store）、
**因果掩码**（逐元素条件选择 → 指数域惩罚 + 循环不变量提升）、
**元素级实现**（`T.Parallel` 逐元素链 → 整 tile v-prefix 向量链）。

### 1.3 数学公式

$$
\begin{aligned}
g(h) &= \lfloor h/\text{groups}\rfloor,\quad \text{groups} = H/H_{kv},\quad \texttt{scale}=D^{-1/2}\log_2 e,\quad \texttt{sm\_scale}=D^{-1/2}\\
P[b,h,i,j] &= 2^{\,q[b,i,h,:]\cdot k[b,j,g(h),:]\cdot\texttt{scale}\;-\;\texttt{lse}[b,h,i]}\quad(\text{causal}: $j>i$ gives $P=0$)\\
dv[b,j,g(h),:] &= \textstyle\sum_i P[b,h,i,j]\,do[b,i,h,:]\\
ds[b,h,i,j] &= P[b,h,i,j]\cdot\big(\textstyle\sum_d do[b,i,h,d]\,v[b,j,g(h),d] - \delta[b,h,i]\big)\cdot\texttt{sm\_scale}\\
dk[b,j,g(h),:] &= \textstyle\sum_i ds[b,h,i,j]\,q[b,i,h,:]\\
dq[b,i,h,:] &= \textstyle\sum_j ds[b,h,i,j]\,k[b,j,g(h),:]
\end{aligned}
$$

公式与 §0.1 语义逐项对应；`lse` 由前向产出、按 **base-2 域**（`logsumexp(scores)/ln2`）输入，与本包 fwd 契约自洽。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

与 §1.6.0 调研选定算法（**单趟融合三梯度 + UB fp32 累加 + 逐行 1-D 行原子 + 一维 persistent 分核**）一致。
每个逻辑任务处理 `(bz, bx, by)` = `(batch, query head, KV 块)`，块内循环 query 块 `kq`：

1. **任务装载**：`T.copy(k[bz, r0:r0+real_m, bx//groups, :], k_s[0:real_m,0:dim])`、`v` 同形 → `v_s`（L1，任务内只读一次）；`T.clear(dv_acc)`、`T.clear(dk_acc)`；causal 时物化掩码平面 `dplane[i,j] = j − i`（**每任务一次**）；
2. **query 循环** `for kq in T.Pipelined(loop_st, loop_ed, num_stages=ns)`（causal 时 `loop_st = by·bm//bn`，块级跳过全掩码区）：
   1. 装载 `q_s[0:real_n,0:dim] ⟵ q[bz, c0:c0+real_n, bx, 0:dim]`；`lse_s[0:real_n] ⟵ lse[bz, bx, c0:c0+real_n]`（→ `vmul(lse_s, LN2)` 得到自然指数域的行偏置，`[bn,1]` 微缓冲）；
   2. `T.gemm(q_s, k_s, qkt, b_transpose=True, initC=True, size=[real_n, real_m, dim])` ⇒ `qkt = Q·Kᵀ`（`[bn,bm]` fp32）；
   3. `T.vmul(qkt, sm_scale, qkt)` ⇒ 分数域；
   4. `T.vsub(qkt, lse_nat, qkt)`（**行广播** `[bn,bm] − [bn,1]`）；
   5. causal：`T.vsub(dplane, d0, dsum)` → `T.vmul(dsum, -PEN, dsum)` → `T.vmin(dsum, 0.0, dsum)` → `T.vadd(qkt, dsum, qkt)`（`d0 = kq·bn − by·bm` 标量，`PEN = 1e30`；**单一定义源 = §3.3**，掩码临时缓冲 `t ≡ dsum`；**不用 `T.vmax`**——其 src 不接受标量）；
   6. `T.vexp(qkt, qkt)` ⇒ `P`（fp32，掩码位下溢为精确 +0.0）；
   7. `T.vcast(qkt, pcast, round_mode="rint")` ⇒ `P` 回落输入 dtype；
   8. 装载 `do_s[0:real_n,0:dim] ⟵ do[bz, c0:c0+real_n, bx, 0:dim]`；
      `T.gemm(do_s, v_s, dsum, b_transpose=True, initC=True, size=[real_n, real_m, dim])` ⇒ `dsum = dO·Vᵀ`（`[bn,bm]` fp32）；
   9. `T.gemm(pcast, do_s, dv_acc, a_transpose=True, size=[real_m, dim, real_n])` ⇒ `dv += Pᵀ·dO`（**跨 kq 累加**，`initC=False`）；
   10. 装载 `delta_s[0:real_n] ⟵ delta[bz, bx, c0:c0+real_n]`；`T.vmul(delta_s, sm_scale, delta_s)`（`[bn,1]` 微缓冲）；
   11. `T.vsub(dsum, delta_s, dsum)` → `T.vmul(qkt, dsum, qkt)`（**dS 原位写回 `qkt`**，O2）→ `T.vmul(qkt, sm_scale, qkt)` ⇒ `ds`（fp32）；
   12. `T.vcast(qkt, dscast, round_mode="rint")` ⇒ `ds` 回落输入 dtype（**复用 `pcast` 的 scratch**，O3）；
   13. `T.gemm(dscast, q_s, dk_acc, a_transpose=True, size=[real_m, dim, real_n])` ⇒ `dk += dsᵀ·Q`（跨 kq 累加）；
   14. `T.gemm(dscast, k_s, dq_blk, initC=True, size=[real_n, dim, real_m])` ⇒ `dq = ds·K`；
   15. `for r in T.serial(real_n): T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r, 0:dim], [1, dim])` ⇒ **dq 跨 KV 块累加**（**逐行 1-D 行原子**；`groups` 无关）；
3. **任务写回**：`groups == 1`（MHA）时 `T.copy(dv_acc[0:real_m,0:dim], dv[bz, r0:r0+real_m, hkv, 0:dim])`（**普通区域存储**，每元素单写者）；`dk` 同形。`groups > 1` 时改**逐行行原子** `for r in T.serial(real_m): T.atomic_add(dv[bz, r0+r, hkv, 0:dim], dv_acc[r,0:dim], [1, dim])`（工厂期 trace-time 分派，§3.3）。

标量/循环盘点（逐项理由见 §1.6.2）：kernel 内**无逐元素标量循环**；保留的循环/标量为
`real_m`/`real_n`（动态边界，每任务/每迭代 1 次）、逐行行原子的行步进 `T.serial(real_n)`/`T.serial(real_m)`（**结构性行步进，行体为向量宽 1-D op**，§1.6.2 #9）、
任务映射索引算术（block 级）、`d0`（causal 每迭代 1 次）。

**与源算法的差异及来源**：一维 persistent 分核（§0.6 R1）；tile 由 UB 预算驱动（R2）；**逐行 1-D 行原子**（R3，revision 1 修订）；
指数域惩罚掩码（R4）；显式尾块（R6）；v-prefix 向量链（R7）。

### 1.5 数据流图

```
GM[q (B,S,H,D) dtype]  --T.copy(跨步 slice, 行 stride=H·D)--> L1[q_s (bn,dim) dtype]
GM[k,v (B,S,Hkv,D)]    --T.copy(跨步 slice, 行 stride=Hkv·D)--> L1[k_s,v_s (bm,dim) dtype]   (任务内一次)
GM[do (B,S,H,D)]       --T.copy(同上)-----------------------> L1[do_s (bn,dim) dtype]
GM[lse,delta (B,H,S) fp32] --T.copy--------------------------> UB[lse_s,delta_s (bn,1) fp32]

Cube 段（L1 → L0A/L0B → L0C，5 次 T.gemm）
  k_sᵀ/b 转置 + q_s --T.gemm--> UB[qkt (bn,bm) fp32]        (= Q·Kᵀ)
  do_s + v_s --T.gemm--> UB[dsum (bn,bm) fp32]              (= dO·Vᵀ)
  pcast + do_s --T.gemm(a_transpose)--> UB[dv_acc (bm,dim) fp32]   跨 kq 累加
  dscast + q_s --T.gemm(a_transpose)--> UB[dk_acc (bm,dim) fp32]   跨 kq 累加
  dscast + k_s --T.gemm--> UB[dq_blk (bn,dim) fp32]

Vector 段（全部在 UB 内闭环；行广播 [bn,1] → [bn,bm]）
  qkt --vmul(sm_scale)--> vsub(lse_nat)--> [causal: vsub/vmax/vmul/vadd 惩罚]--> vexp --> P (fp32)
  P --vcast(rint)--> UB[pcast (bn,bm) dtype] --(供 g3)-->
  dsum --vsub(delta·sm_scale)--> vmul(P)--> vmul(sm_scale)--> ds (fp32, **原位写回 qkt**)
  ds --vcast(rint)--> UB[dscast (同一 scratch)] --(供 g4/g5)-->

UB --> GM（结果）
  dv_acc --T.copy(跨步区域；groups==1) / 逐行行原子(groups>1)--> GM[dv (B,S,Hkv,D) fp32]
  dk_acc --同上--> GM[dk fp32]
  dq_blk --逐行 1-D 行原子 [1,dim] × real_n 行--> GM[dq (B,S,H,D) fp32]   (每 kq 一次)

数据层级：GM ↔ L1（Cube 操作数）↔ L0A/L0B/L0C（GEMM）↔ UB（Vector 计算 + fp32 累加器）；
无中间结果落回 GM（除三个输出）、无 workspace、无核间 flag 同步（跨核归约由硬件原子承担）。
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：同一数学语义（**flash-attention backward 的 dq/dk/dv 重算 + 跨块累加**）的算法族——
> 源算法（三维 grid + 单趟融合 + 逐元素原子 dq）只是**基线候选之一**。
> **调研深度**：**完整调研**（矩阵/多步/融合类，按 `references/algorithm-research.md` §2 分级：R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。
> **信息源**：`algorithm-candidates.md` 的 `ALG-attention` 命中行（必查清单）与 pattern-library `attention.md`
> （PL-1.7 / PL-1.8 / PL-1.9-blockwidth / PL-1.9-hardlimits / PL-1.11 / PL-1.12 / PL-1.13 / PL-1.16 / PL-1.18）；
> 本仓同类 npuir 实现（`examples/deepseek_v32/sparse_mla_bwd.py`、`examples/deepseek_v4/example_sparse_attn_bwd_kernel.py`、
> `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`、`examples/elementwise/atomic_add.py`）；
> `constants.md`（CONST-capacity-910B2C / CONST-aicore-910B2C / CONST-vector-launch-overhead / CONST-mte2-degradation /
> CONST-copy-floor-dma-granularity / CONST-aiv-time-decomposition / CONST-msprof-task-floor-910B3 / CONST-L1-port-bw / CONST-fabric-bw）；
> `traps-compiler.md`（TRAP-parallel2d-bm-ge2-scalarize / TRAP-inner-axis-reduce-scalarize / TRAP-tcv-parser-rules 族 /
> TRAP-scf-for-row-offset-silent-miscompile / TRAP-primfunc-compile-time-if-buffer-def / TRAP-vcast-same-dtype-rounds /
> TRAP-C12-copy-dtype-cast / TRAP-vexp2-not-exact-pow2 / TRAP-threads-kwarg-noop）；
> `traps-runtime.md`（TRAP-DEVMODE-PERSIST-GEMM / TRAP-DEVEMODE 相关 / TRAP-T-copy-region-semantics / TRAP-UB-dst-align /
> TRAP-load-nd2nz-strided / TRAP-L1-band-dst-tail-overrun）；
> 源码 §0.3/§0.4；**本任务三枚设计期探针**（§1.6.0 末）。
> **未使用互联网检索**：本地源（参考表命中行 + 三个同语义 npuir 实现 + pattern-library 实测条目 + 源码 + 三枚本机探针）
> 已覆盖四问所需的全部候选与常数，无未命中缺口；本算子族（attention bwd）的候选空间是公开已知的
> 「单趟 vs 两趟、materialize vs 重算、原子 vs 无原子」三类，本地证据已足以排序。

> **证据路径（负向断言与弃选论证所指向的条目，设计期已逐条打开核对）**：
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（CONST-capacity-910B2C、CONST-aicore-910B2C、CONST-vector-launch-overhead、CONST-mte2-degradation、CONST-copy-floor-dma-granularity、CONST-aiv-time-decomposition、CONST-msprof-task-floor-910B3、CONST-L1-port-bw、CONST-fabric-bw、CONST-reduce-ub-per-element）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/attention.md`（PL-1.7、PL-1.8、PL-1.9-blockwidth、PL-1.9-hardlimits、PL-1.11、PL-1.12、PL-1.13、PL-1.16、PL-1.18）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md`（PL-1.1、PL-1.5-quickref、PL-1.15、PL-1.21、PL-1.23）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`（TRAP-parallel2d-bm-ge2-scalarize、TRAP-inner-axis-reduce-scalarize、TRAP-scf-for-row-offset-silent-miscompile、TRAP-primfunc-compile-time-if-buffer-def、TRAP-vcast-same-dtype-rounds、TRAP-C12-copy-dtype-cast、TRAP-vexp2-not-exact-pow2、TRAP-threads-kwarg-noop、TRAP-transpose-ub-gm-roundtrip）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md`（TRAP-DEVMODE-PERSIST-GEMM、TRAP-T-copy-region-semantics、TRAP-UB-dst-align、TRAP-load-nd2nz-strided、TRAP-L1-band-dst-tail-overrun）
> - `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`、`.agents/skills/tilelang-op-design/references/algorithm-research.md`、`.agents/skills/tilelang-op-design/references/migration-analysis.md`
> - `.agents/skills/_shared/standards/core-split-strategy.md`、`.agents/skills/_shared/standards/hardware-cost-model.md`、`.agents/skills/_shared/standards/negative-claim-evidence.md`
> - `docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/原子操作/T.atomic_add.md`、`docs/Tilelang.language/原子操作/T.atomic_addx4.md`、`docs/Tilelang.language/数学操作/T.vmul.md`、`docs/Tilelang.language/数学操作/T.vsub.md`、`docs/Tilelang.language/数学操作/T.vexp.md`、`docs/Tilelang.language/数学操作/T.vexp2.md`、`docs/Tilelang.language/比较操作/T.vmax.md`、`docs/Tilelang.language/数据类型转换操作/T.vcast.md`、`docs/Tilelang.language/内存操作/T.copy.md`、`docs/Tilelang.language/内存操作/T.alloc_shared.md`、`docs/Tilelang.language/编译器提示操作/T.Kernel.md`、`docs/Tilelang.language/比较操作/T.min.md`
> - `.agents/evolution/capability-gaps.md`（CG-2026-0004 load_nd2nz 跨步、CG-2026-0010 Developer persistent+gemm）

**R1 等价化简公式候选**：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | **基线**（GPU 源算法） | 三维 grid `(H,⌈S/bm⌉,B)`；`qkt=K·Qᵀ` 单趟算 P/ds/dv/dk/dq；`if_then_else` 逐元素因果掩码；dq 逐元素 `T.atomic_add` + `make_dq_layout` | — | — | ✅（基线） |
| 1 | **单趟融合 + 指数域惩罚掩码 + 逐行 1-D 行原子**（本设计主选） | 同源结构；掩码改 `P = exp(x + pen)`、`pen=min(−PEN·(dplane−d0),0)` 且 `(j−i)` 平面循环不变量提升；dq 改**逐行 1-D 行原子** `for r: T.atomic_add(dq[bz,c0+r,bx,0:dim], dq_blk[r,0:dim],[1,dim])`；`exp2` 换 `vexp` 并重标定参数 | **数学恒等**（掩码位：valid 时 `pen=0` 逐位恒等；invalid 时 `exp(有限负数)` 下溢为精确 +0.0，与源置 0 逐位一致）。`vexp` 换形为**容差内等价**（§1.6.1 O4，机器验证 16 ulp / 0 超容差）。原子形态替换为**同一数学操作**（同目标元素、同加数、同 fp32 加；探针三正面验证） | 少算什么：整 tile 向量 op −1（O4）/ 掩码路径由逐元素标量化变 4 个整 tile op；少搬什么：无 GM 变化（流量 A/C 与基线相同） | ✅（主选） |
| 2 | **两趟原子自由**（FA2 式：dk/dv kernel + dq kernel） | 两个 kernel 各自「owner 维完整、另一维循环」，两趟都重算 P 与 ds | 数学恒等（fp32 累加结合序不同，容差内） | 少搬什么：无原子放大；**多算什么**：GEMM 由 5 个变 7 个（1.4×）、向量 pass 数 ×2；跨核归并代价归零 | ✅（R3 对比；弃选论证见下） |
| 3 | **`mb` 分块单趟**（每任务持 `mb` 个 KV 块，dq 原子放大 ÷`mb`） | 单趟 + 片上同时持 `mb` 份 `dk/dv` 累加器 | 数学恒等 | 少搬什么：dq 原子放大 `⌈S/bm⌉/mb` | ✅（R3/R4） |
| 4 | **转置 P 布局 `[kv,q]`**（源原样） + 列广播 / `vbrc` 展开 | `qkt` 声明 `(bm,bn)`，`lse/delta` 变成列因子 | 数学恒等 | 无 GM 收益；`lse/delta` 需 `vbrc` 展开或吃未文档化列广播税 | ✅（R4） |
| 5 | **`dq`-owner 单趟**（grid 过 query 块，循环 KV 块） | dq 片上累加、无原子；dk/dv 原子放大 `⌈S/bn⌉` | 数学恒等 | 少搬什么：dq 无原子放大 | ✅（R3） |
| 6 | **scale 融入 K**（`k'=dtype(scale·k)` 后 GEMM） | 把 `scale` 从 `[bn,bm]` 的 vmul 移到 `[bm,dim]` 的装载 | **否**——K 存于输入 dtype，预乘后回落的量化改变了 GEMM 操作数值 | 省 1 个 `[bn,bm]` vmul | ❌（数值否决，§1.6.1 R1 机器证据：max_rel 3.25） |
| 7 | **`lse` 折进 GEMM 的增广维**（`[Q,−lse/scale]·[K,1]ᵀ`） | 用 GEMM 的累加吸收 `−lse` | **否**（容差内也难）——K 维变 `dim+1` 破坏 fractal 对齐（K≥32 且 16 对齐），且 `−lse` 进入 GEMM 累加序改变舍入 | 省 1 个 `[bn,bm]` vsub | ❌（结构否决 + 舍入路径变化） |
| 8 | **`PEN` 由 `1e38` 改 `1e30`** | 惩罚常量 | 容差内等价（均给出精确 +0.0） | 避免 `t·PEN` 中间量上溢为 `−inf` | ✅（采纳为 O1 的一部分） |

> 注：本算子在**代数式子层面**的化简空间很窄——`exp2`/`sm_scale`/`scale` 的常数折算已被源做尽
> （`scale` 已把 `log2 e` 折进，`sm_scale` 单独保留用于 ds），除法/归一化/公共子表达式在源中均不存在；
> R1 的候选主要来自**计算域重排（掩码域、指数基域）、数据布局与累加结构**三个方向（上表 1–7）。

**R2 在线算法**：**有**在线（流式）变体，但**本算子不需要，且源算法已是本族的正确形态**。
依据（结构判据，非无据否定）：
- 在线 softmax 的收益口径是「running max/sum + 重缩放 ⇒ 免 O(N) 中间缓冲、免两遍扫描」。
  本算子的 `lse` 与 `delta` 由**前序 kernel 预先给出**（`_flashattn_bwd_preprocess_kernel` 与 fwd），
  `P = exp2(qk·scale − lse)` 是**闭式重算**，**不存在 running max、不存在重缩放、不存在两遍扫描需求**——
  在线化的收益口径在本算子**不适用**（与源一致：FA2 的 backward 同样非在线）。
- **已在线化的部分**：`dv/dk` 沿 query 块的跨迭代累加（`dv_acc`/`dk_acc` 跨 `kq` 循环累加）本身就是
  「单遍流式 running 累加」形态；`dq` 沿 KV 块的单遍累加由硬件原子承接。即本算子的累加结构**已是在线式**。
- **可在线化的余量**：无。`P` 无法避免逐元素物化（`dv` 与 `dk` 需要 P 与 ds 的**整块**参与两个不同的 GEMM）。

**R3 复杂度对比**（四口径；口径定义：**FLOPs** 以「每 (query 块 × KV 块) 对的 GEMM MAC 数」为单位（`1 单位 = bm·bn·dim` MAC），
向量 pass 数单列；**访存 Bytes** 分三口径（**口径 A**：L2 吸收跨任务重读与原子放大的 **HBM 工作假设口径**；**口径 B′**：**仅**原子写放大、不含重读；**口径 C**：**统一总线口径** = A + 循环轴重读 + 原子放大，供 §1.6.4 裁决）；
**扫描遍数** = 同一 GM 元素的重复读取/写入次数；**中间缓冲峰值** = UB/片上驻留）：

| 算法候选 | GEMM 单位 | 向量 pass / 迭代（revision 1 口径：整 tile + 微 op + 行原子） | 访存口径 A（HBM，`E=B·H·S·D`） | 访存口径 B′（**仅**原子写放大，不含重读） | 访存口径 C（统一总线：A + 循环轴重读 + 原子放大） | 扫描遍数（Q/dO / K/V / dq） | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------|---------------|------------------------------|--------------------------|---------------------------|-------------|--------------------|
| 0 基线（GPU 源：三维 grid + 逐元素原子） | **5** | 10（含逐元素条件选择 + 逐元素原子的标量化） | `E·(4·eb+12)` = full-fp16 **1.34 GB** | `(⌈S/bm⌉−1)·E·4` = **8.32 GB** | **17.98 GB**（= 1.34 + 原子 8.32 + q/dO 重读 `31·2·E·eb` = 8.32） | `⌈S/bm⌉` / 1 / `⌈S/bm⌉` | 5×`(bm·bn)` fp32 + 3×`(bm/bn·dim)` fp32（CUDA fragment 无 UB 上限概念） | 逻辑核 `B·H·⌈S/bm⌉`；dq 靠硬件原子跨核归并 |
| **1 主选：单趟 + 指数域惩罚 + 逐行行原子** | **5** | **8 整 tile + 2 微 + `real_n` 行原子**（非 causal）/ 12 + 2 + `real_n`（causal） | **1.34 GB（= 基线）** | 8.32 GB（同基线） | **17.98 GB**（同基线；流量不因原子形态改变而改变） | `⌈S/bm⌉` / 1 / `⌈S/bm⌉` | 声明 **100–136 KB**（§4.5 逐 buffer 复算） | 逻辑核 `B·H·⌈S/bm⌉`（full = 8192）→ persistent 20 核（§5.5） |
| 2 两趟原子自由 | **7（1.4×）** | **10 每趟（8 整 tile + 2 微）/ 两趟合计 20** | `E·(8·eb+12)` = **1.88 GB** | **0**（无原子） | **27.1 GB**（= 1.88 + pass A q/dO 重读 8.32 + pass B k/v 重读 `63·2·E·eb` = 16.91） | 1 / 1 / 1（各趟 owner 维完整） | 每趟更小（单趟的子集，无 dq+dk/dv 同时驻留） | 逻辑核 ×2；**无跨核归并**；但两次 launch |
| 3 `mb` 分块单趟 | 5 | 8 + 2 + `real_n` 行原子 | 1.34 GB | `(⌈S/bm⌉−1)/mb·E·4`（mb=2 ⇒ 4.16 GB） | **13.8 GB**（= 1.34 + 4.16 + 8.32） | `⌈S/bm⌉` / 1 / `⌈S/bm⌉÷mb` | **翻倍**（`mb` 份 `dk/dv` 累加器：mb=2 时 +96 KB ⇒ 超预算） | 同主选 |
| 4 转置 P 布局（源原样）+ 列广播 | 5 | 8 + 2 + `real_n` 行原子 + 2·`vbrc` | 1.34 GB | 8.32 GB | 17.98 GB | 同主选 | 主选 + 2 个 `[bm,bn]` 广播平面 | 同主选 |
| 5 `dq`-owner 单趟 | 5 | 8 + 2 + `real_m` 行原子（dk/dv 侧） | 1.34 GB | `(⌈S/bn⌉−1)·2·E·4` = **33.8 GB** | **52.1 GB**（= 1.34 + 33.8 + k/v 重读 16.91） | 1 / `⌈S/bn⌉` / 1 | 主选（累加器角色互换） | 同主选 |
| 6 scale 融入 K | 5 | 7 整 tile + 2 微 + `real_n` 行原子（主选 − 1 整 tile op） | 1.34 GB | 8.32 GB | 17.98 GB | 同主选 | 主选 − 0（省一个整 tile op） | 数值否决（§1.6.1 R1） |
| 7 `lse` 折进增广维 | 5（K 维 `dim+1`） | 7 整 tile + 2 微 + `real_n` 行原子（主选 − 1 整 tile op） | 1.34 GB | 8.32 GB | 17.98 GB | 同主选 | fractal 对齐破坏 | 结构否决 |

> **口径注（revision 1 重定义，REVIEW.md 问题 3）**：
> ① **口径 A（HBM 工作假设）**：「输入一次读入 + 三个输出一次写出」= `E·(4·eb+12)` B 的**不可消除 I/O**，
> 与 manifest 的 `bytes = B·7·H·S·D·elem_bytes` 同量级（后者含 `lse`/`delta` 与 f32 输出计权）；
> 假设跨任务重读与原子放大被 L2 吸收——**未文档化假设 + 估算依据**：单头工作集 0.5–1 MB 小于 pattern-library
> 反复观测到的 L2 命中规模（PL-1.18 记 `cube read_hit 91% / AIV 99%`）。
> ② **口径 B′（仅原子写放大）**：**只计入原子写放大、不计循环轴重读**。revision 0 曾把这一列命名为
> 「无 L2 吸收的总线口径」却只加了原子项、漏掉同口径下同样不被 L2 吸收的重读项（记账不自洽）。
> 原子放大因子按各候选的**实际写者数**取：候选 0/1/3 的 `dq` 被写 `⌈S/bm⌉` 次；
> 候选 5 的 `dk/dv` 被写 `⌈S/bn⌉` 次 ⇒ `(⌈S/bn⌉−1)·2·E·4`。
> ③ **口径 C（统一总线口径，新增，供 §1.6.4 裁决）**：`A + 循环轴重读 + 原子放大`。
> 重读按各候选循环轴的实际发生次数计：候选 0/1/3 的 `q/dO` 重读 `(⌈S/bm⌉−1)·2·E·eb`；
> 候选 2 的 pass A `q/dO` + pass B `k/v` 两轴都重读；候选 5 的 `k/v` 重读 `(⌈S/bn⌉−1)·2·E·eb`。
> **在统一口径 C 下候选 2 并不占优（27.1 GB vs 主选 17.98 GB）** ⇒ revision 0 的
> 「若口径 B 成立则备选反超（2.01 vs 9.66 GB）」论断**作废**——其前提是「只给备选计零原子、却给主选计原子放大，
> 而两者都不计重读」，属口径不对称（见 §1.6.4）。
> ④ **候选 0 与候选 1 的口径 A/C 完全相同**——主选的收益不在流量，而在 ① **发射形态**：候选 0 的掩码与
> dq 原子均为逐元素形态，在 NPU 上被标量化（`TRAP-parallel2d-bm-ge2-scalarize`）；与 ② **形态可落地性**：
> 候选 0 的逐元素原子 + `make_dq_layout` 在 NPU 无对应（§0.5/§0.6 R3）。
> ⑤ 本算子的排序**主要由「发射项（整 tile 向量 pass 数 + 逐行行原子发射数）」与「原子放大是否被 L2 吸收」两个维度决定**，
> 不是纸面并列（本节末尾的设计期估算把两个维度都定量给出；行原子发射单价的实测标定见探针三）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰均附证据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| 0 基线（GPU 逐元素形态直译） | ⚠ 5 个 GEMM 落 Cube ✅，但掩码用 `T.Parallel`+`T.if_then_else`（NPU 实测标量化）、dq 原子逐元素 + `make_dq_layout`（CUDA fragment 布局） | ❌ CUDA fragment 尺寸无 UB 对应，须按 §4.5 重算 | ✅ | ✅ | ✅ | 三维 grid ❌（NPU ≤2 维） | ❌ **直译不可用**（§0.5/§0.6 重设计） |
| **1 主选：单趟 + 惩罚掩码 + 逐行行原子** | ✅ Cube `T.gemm`（M/N≥16、K≥32：`dim∈{64,128}`、`bm,bn≥32`）+ Vector v-prefix 链 | ✅ 声明 100–136 KB ≤ 192 KB（双口径，§4.5） | ✅ 尾轴 `dim ∈ {64,128}` 均为 8 的整数倍（fp16 向量宽）；`bm,bn` 为 32 的整数倍（fractal + 向量宽）；行原子目标 = `dst[bz, c0+r, bx, 0:dim]`（**尾轴 `dim` 连续段，逐行 1-D**，行 stride = 张量真实 `H·D`，由标量行索引表达） | ✅ `B/S/H/D/groups/is_causal` 全为工厂期常量；`T.serial(num_local_tasks)` 界为 host 折叠常量；行原子行数 `real_n`/`real_m` 为 runtime 标量（`T.serial` 接受） | ✅ `T.Pipelined` 承载 query 循环（`num_stages` 可调）；auto multi-buffer 交编译器 | ✅ dq 跨核归并用**逐行 1-D 行原子**（探针三实测：目标区全中、区外 0、20 并发同址精确 20.0）；`groups==1` 时 dk/dv 无跨核写者 | ✅ **主选** |
| 2 两趟原子自由 | ✅ Cube + Vector（每趟 3–4 个 GEMM） | ✅ 每趟更小 | ✅ | ⚠ causal 时 pass B 的上界为 runtime 表达式 | ✅ | ✅ 无跨核归并 | ❌/⚠ **并列候选（交 §1.6.4 实验裁决）**：GEMM `5→7`、元素吞吐 ×2（地板档 15.2 ms vs 主选 9.5 ms）；但**零原子** ⇒ 免疫同址并发不确定性，且**聚合 issued-op 10.5 M < 主选 22.0 M**（上界档 131–262 ms vs 主选 275–550 ms）。**两趟的 pass A `groups>1` 分支在本设计中同样只用逐行 1-D 行原子**（不用区域形态）。口径 A 下多 0.55 GB，**口径 C 下更差（27.1 vs 17.98 GB）** ⇒ 备选理据为**正确性 + 发射项**（非流量），裁决计划见 §1.6.4 |
| 3 `mb` 分块单趟 | ✅ | ❌ `mb` 份 `dk/dv` 累加器（`[mb·bm, dim]` fp32 ×2）在 `mb=2, bm=64, dim=128` 时 +96 KB ⇒ 总 ~232 KB 超 UB；`mb=2` 仅能把 dq 原子放大 32→16 | ✅ | ✅ | ✅ | ✅ | ❌ **域内不可行**（UB 上限）⇒ 留作超预算 fallback 说明（§8.3） |
| 4 转置 P 布局（源原样）+ 列广播 | ✅ | ⚠ +2 个 `[bm,bn]` 广播平面 | ⚠ 列因子靠 `vbrc` 展开（`T.vbrc` 的 region/同形陷阱族：TRAP-npuir-vbrc-region-dropped / TRAP-vbrc-same-shape-empty-broadcast） | ✅ | ⚠ 每迭代 +2 op | ✅ | ❌ **弃选（可免费避免的开销）**：改成 `[q,kv]` 声明后 `lse/delta` 是**文档化的行广播** `[M,N]*[M,1]`（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2），零额外 op、零额外缓冲；源原样的 `[kv,q]` 会强制列因子走 `vbrc` 或吃 `PL-1.15` 记的 +12~16% 未文档化列广播税 |
| 5 `dq`-owner 单趟 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠ dk/dv 放大 `(⌈S/bn⌉−1)·2·E·4`：统一口径 C 下 **52.1 GB**（主选 1.8× 以上） | ❌（R3 统一口径 C 更差；口径 A 下与主选同流量但换 dq 无原子为 dk/dv 原子 ⇒ 无净收益） |
| 6 scale 融入 K | ✅ | ✅ | ⚠ 预乘后回落输入 dtype 改变 GEMM 操作数 | ✅ | ✅ 省 1 op | ✅ | ❌ 数值否决（§1.6.1 R1） |
| 7 `lse` 折进增广维 | ✅ | ✅ | ❌ K 维 `dim+1` 破坏 16 对齐与 K≥32 的 fractal 下界 | ✅ | ✅ 省 1 op | ✅ | ❌ 结构 + 舍入否决 |

**调研结论**：选定 **候选 1「单趟融合三梯度 + 指数域惩罚掩码（含循环不变量提升）+ 逐行 1-D 行原子 + 一维 persistent 分核 + UB fp32 累加」**
（= 候选 0 的数学内核与单趟结构 + §1.6.1 的 O1/O2/O3/O4；原子形态为 revision 1 修订后的逐行行原子）。
关键依据：**R3** 中该候选的 GEMM 单位（5）与口径 A/C 流量（1.34 / 17.98 GB）为全候选最小/并列最小，
发射形态为 8 整 tile + 2 微 op + `real_n` 行原子（候选 2 为每趟 10、两趟合计 20；候选 0 的掩码与原子为标量化形态）；
**R4** 全项亲和（Cube+Vector 分工明确、声明 100–136 KB ≤ 192 KB、`dim∈{64,128}` 满足 fractal 与向量宽、
静态边界、逐行行原子承接跨核归并——探针三已验证），
且候选 3/5 分别因「UB 超限」「统一口径 C 更差（52.1 GB）」被淘汰，候选 4/6/7 分别因「可免费避免的列广播开销」「数值否决」「结构否决」被淘汰；
候选 2（两趟原子自由）**未被纸面淘汰，而是降为 §1.6.4 的结构级并列候选**——其零原子正确性优势与 revision 1 重算后的聚合 issued-op 优势（10.5 M vs 22.0 M）在串行单价档下足以反超，而元素吞吐档下主选仍占优（两档给出相反排序）⇒ 交 Stage 4 实测「行原子单价」裁决（Phase 2 第 3 项：判定裕度落在未实测常数值域内时不得纸面单选）。
**与基线的结构差异一句话**：数学内核（单趟、5 个 GEMM、fp32 累加、P/ds 回落输入 dtype）**完全不变**，
改的是**并行结构**（三维 grid → 一维 persistent）、**tile 尺寸**（UB 预算驱动）、**掩码实现**（逐元素条件选择 → 指数域惩罚 + 循环不变量提升）、
**原子形态**（逐元素 → **逐行 1-D 行原子**；`groups==1` 的 dk/dv 分派为普通区域 store）、**元素级实现**（`T.Parallel` → v-prefix 整 tile 算子）。
**调研范围**（供「无更优替代」类断言核对）：`algorithm-candidates.md` 的 `ALG-attention` 命中行；
本仓三个同语义 npuir 实现 + 两个同族已交付迁移（`examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/DESIGN.md`、
`examples/group_norm/_group_norm_kernel/DESIGN.md`）；pattern-library 四个条目文件（attention/layout/constants/traps-compiler/traps-runtime）；
源码 §0.3/§0.4；R2 结构判据（`lse`/`delta` 外部给定 ⇒ 无在线化收益）；本任务**三枚**设计期探针（本节末）。
**源算法优化手段承接**见 §0.6 末尾核对表（12 项逐项落实，**无静默丢弃**）。

**设计期探针（D-3；**三枚**，均落 `history_version/`，本机 910B3 + CANN 9.0.0 + Developer）**：

| 探针 | 测什么（未知常数/可行性） | 实测结论（2026-10-16 重跑） | 用途 / 适用范围 |
|------|--------------------------|---------|------|
| `history_version/design_probe_atomic_vs_store.py` | ① `T.atomic_add`（fp32，**连续** 2-D 区域）相对普通 `T.copy` store 的**单价比**；② **rank-2 显式切片**的跨步目标（`dst[m0:m0+BM, BN:2BN]`）可用性 | ① **原子 / store = 0.97×**（126.60 vs 130.02 µs / 8.4 MB，median of 5×30）、**降级为 `hivm::StoreOp`（非 RMW 内存往返）**；② 跨步 band 原子与 store **逐位精确**（band[128:256] 全 4.0 / 全 1.0、区外 0）——**revision 1 已修正其期望值判据**（原 `expect=0.0` 恒为 False） | 支撑 R3 的「原子放大是**流量**问题而非单价问题」；**适用范围仅限连续 2-D 区域 / rank-2 双侧显式切片**，不外推到 BSHD rank-reduced 4-D 形态 |
| `history_version/design_probe_persist_and_strided.py` | ① rank-2 张量双侧显式切片的跨步 2-D `T.copy` 存储数值正确性；② 同形态跨步 2-D `T.atomic_add`；③ **Developer + 固定核数 + 核内 `T.serial` + `T.gemm` + v-prefix** 是否可用（TRAP-DEVMODE-PERSIST-GEMM 重验） | ①/② **逐位正确**（band[1.0,1.0] expect 1、区域外 max 0.0）——**revision 1 已修正 `n_write` 判据**（原 `n_write = NBM` 与实际「每 cell 写 1 次」的构造不符，恒为 False）；③ **编译通过且数值正确**（`n_kernels=4/n_tasks=8` 与 `20/4` 两档，`max_abs_err=4.578e-05`）⇒ 该 trap 在**本工具链不成立** | ③ 支撑 §5.5 的 persistent 分核方案（否则须切 Expert，§2）；①/② **仅覆盖 rank-2 形态**，是 revision 0「BSHD 形态逐位正确」错误断言的根因（见 §9 R1） |
| `history_version/design_probe_row_atomic_bshd.py`（**revision 1 新增**） | **修正形态（逐行 1-D 行原子）在 BSHD `[B,S,H,D]` 目标上的三类断言**：① 目标区元素计数；② 区外元素计数；③ 并发同址总量；另标定行原子发射单价 | ① 设计几何 (1,128,8,64) rows0:32 h3 ⇒ **2048/2048 cell == 1.0**；(1,64,16,128) rows0:16 h5 ⇒ 2048/2048 == 1.0；② 区外 **65536 / 131072 cell 全 0**（含 `h = H−1` 档）；③ **20 个 block 并发同址 ⇒ 每 cell 精确 20.0**、8 并发 ⇒ 8.0；④ 动态行数 `T.serial(T.min(bn, S−row0))`（S=100, row0=64 ⇒ nrow=36）**2304/2304 == 3.0、区外 0**；⑤ 计时：`real_n=32`@dim=128 的行原子环 **0.99×** 单次同体积区域 store、`real_n=64`@dim=64 **0.79×**（该测试床有 ~120 µs/launch 地板，**未能解析每行原子单价**） | **正面确立修订形态的可落地性与正确性**（REVIEW.md 问题 1/2 的修复证据）；⑤ 作为 §1.6.4 裁决计划中「行原子单价」未解析的诚实登记 |

**设计期估算（D-2 roofline；协议 `_shared/standards/hardware-cost-model.md` §2；常数带版本戳）**：
估算下界 = max(流量项, 发射项, 容量项)。按 manifest 四档（§5.5 给出 `bm/bn`/逻辑核数/核内轮数）：
`E = B·H·S·D`；带宽锚点 **1.06 TB/s**（`CONST-copy-floor-dma-granularity`：910B3 + CANN 9.0.0，16 KB 粒度、40 AIV 聚合纯拷贝实测 1.06–1.23 TB/s）；
发射项按**迭代内 issued-op 清点**（revision 1 重算，REVIEW.md 问题 1 建议 5）：
`N_ops/迭代 = 8 整 tile + 2 微（[bn,1]）+ real_n 行原子`（非 causal；causal 追加 4 个整 tile op）。
单价双档：**上界档 0.25–0.5 µs/op**（`CONST-vector-launch-overhead` + `CONST-msprof-task-floor-910B3`，串行模型，把行原子也按整 op 计——**保守偏悲观**）
与**计算地板档**（元素吞吐模型：沿用 v0 的「36 ns / 2048 fp32 元素 op」常数 ⇒ **17.6 ps/元素**；每迭代元素数 `N_elem = 8·bn·bm + 2·bn + real_n·dim`）。

| workload（manifest） | (bm,bn,ns) | 逻辑核数 / 核内轮数 | `N_ops`/迭代 | 流量项（口径 A / 口径 C） | 发射项（上式 ÷ 20 核 × 单价） | 容量项 | 估算下界 |
|----------------------|-----------|-------------------|--------------|--------------------------|--------------------------------------|--------|----------|
| `smoke-bwd-fp16` (1,1024,8,64) | (64,64,1) | 128 / 7 | 8+2+64 = **74** | 10.5 MB / 73 MB → **10 µs / 69 µs** | 2048 iters → 102 iter/核 × 74 = 7578 op → **1.9–3.8 ms（上界档）/ 67 µs（地板档，36 992 元素/迭代）** | 0 | **≈67 µs – 3.8 ms（发射受限）** |
| `smoke-bwd-bf16` 同形 | 同上 | 同上 | 同上 | 同上 | 同上 | 0 | 同上 |
| `full-bwd-fp16-large` (16,2048,16,128) | (64,32,1) | 8192 / 410 | 8+2+32 = **42** | **1.34 GB / 17.98 GB** → **1.27 ms / 17.0 ms** | 524288 iters → 26214 iter/核 × 42 = 1 101 005 op → **275–550 ms（上界档）/ 9.5 ms（地板档，20 544 元素/迭代）** | 0 | **≈9.5–550 ms（发射受限）；流量项 1.27 ms 为可信地板** |
| `full-bwd-bf16` (4,4096,16,128) | (64,32,1) | 4096 / 205 | 8+2+32 = **42** | 671 MB / 17.58 GB → **0.63 ms / 16.6 ms** | 262144 iters → 13107 iter/核 × 42 = 550 512 op → **138–275 ms（上界档）/ 4.7 ms（地板档）** | 0 | **≈4.7–275 ms** |

> **估算下界（汇总）**：代表 workload `full-bwd-fp16-large` 的**流量项 1.27 ms 是可信地板**（口径 A；见 R3 口径注①），
> 发射项在 **9.5 ms（元素吞吐地板档）～550 ms（issued-op 串行上界档）** 之间。
> ⇒ **本算子仍落在「发射受限」区**，发射项 ∝ `迭代数 × N_ops/迭代`（后者 = `10 + real_n`，`real_n ∈ {32,64}`）。
> **revision 1 重算的净效应**：每迭代 issued-op 由 revision 0 记账的 **9**（其中写出为 1 个区域原子）变为 **`10 + real_n`**（写出为 `real_n` 个行原子）
> ⇒ **上界档发射项放大约 4.7×**（`42/9`）；**元素/流量口径不变**（写出元素数、读写字节数、GEMM 数、整 tile 向量 op 数完全不变）。
> 行原子是 UB→GM 的 **store 类 op**（非向量 ALU op），其单价未在本工具链解析（探针三⑤：32 个行原子 ≈ 1 个同体积区域 store，
> 被测试床 ~120 µs/launch 地板掩盖）⇒ 上界档把行原子按整 op 价计属**保守偏悲观**（未文档化假设 + 估算依据：store 类 op 无 UB 读写往返）。
> **模型可靠性声明（诚实标注）**：发射项的单价档分别来自
> ① `CONST-vector-launch-overhead`（910B2C + CANN 8.5.0，~0.5 µs/op，**跨设备跨代引用**）与
> ② `CONST-msprof-task-floor-910B3`（910B3 + CANN 9.0.0，0.23–0.3 µs/pass，**同设备**，但其语境是 fp8 软件编码链的小 tile 串行链）；
> ③ 地板档为**元素吞吐模型**（向量宽度落地价，未实测）。
> 三者相差一个量级 ⇒ **设计期不据此在参数/结构层面拍板**，而是给出参数级 A/B（§1.6.3）与结构级裁决计划（§1.6.4），由 Stage 4 实测标定。
> **参数敏感方向（指导 Stage 4）**：发射项 ∝ `迭代数 × N_ops/迭代`。dim=128 档由 `bn=32` 上探到 `bn=64` 可把迭代数 524288 → 262144
> 且 `real_n` 由 32 → 64（`N_ops` 42 → 74）⇒ 上界档发射项 **≈ 0.88×**（迭代数减半、每迭代 op 数近翻倍，净小幅下降）；
> **元素吞吐地板档对 tile 尺寸不变**（总元素工作量守恒：`20544×524288 = 41088×262144`）⇒ 上探档只压低「发射/launch」维度，不改变计算量。
> UB 声明 100 → 136 KB（§4.5 双口径与 §1.6.3 参数级 A/B）；dim≤64 档已取 `bn=64`。
> **常数适用性声明**：`CONST-mte2-degradation` / `CONST-vector-launch-overhead` 绑定 **910B2C + CANN 8.5.0**，
> 本设备为 **910B3 + CANN 9.0.0** ⇒ 按「未实测假设：沿用同代 910B 曲线量级」引用；
> `CONST-copy-floor-dma-granularity` / `CONST-msprof-task-floor-910B3` / `CONST-aicore-910B2C` 的 910B3 数据点为本工具链同设备实测（可信档）。
> 核数（20）与 UB 容量（196608 B）**已本机实查复核**（§5.5），无假设成分。Stage 4 须按 `hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行，
> 并**优先标定两个新未知量**：① 行原子单价（`aiv_mte3_time` / store-op 计数）；② 逐行行原子环对 `Task Duration` 的实际贡献。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 调研选定的算法（单趟 + 指数域惩罚掩码 + 逐行 1-D 行原子 + 一维 persistent + UB fp32 累加）。
> 逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）。
> **R1 已结论**：本算子的**代数化简空间很窄**——源的 `scale` 已把 `log2 e` 折进指数、`sm_scale` 单独保留、
> 无除法、无归一化、无公共子表达式；§1.6.0 R1 的 8 个候选里 6/7 被数值/结构否决（见 R1 表）。
> 故本文的采纳项集中在**计算域重排**（掩码域、指数基域）与**缓冲生命周期**两类。

| # | 优化项 | 原式（源 / 基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|------------------|-----------|-----------|---------|
| **O1** | **因果掩码：逐元素条件选择 → 指数域惩罚 + 掩码平面循环不变量提升** | `qkt[i,j] = if_then_else(by·bm+i ≤ k_idx·bn+j, qkt[i,j], 0)`，在 `T.Parallel(bm,bn)` 内逐元素（NPU 实测标量化），且 `(i,j)` 索引比较**每迭代重算** | `P = exp(qkt·sm_scale − lse·ln2 + pen)`，`pen = min(−PEN·(j − i − d0), 0)`（**单一定义源 = §3.3**；`t ≡ dsum`，见 §0.6/§1.4 引用行），`d0 = k_idx·bn − by·bm`（每迭代 1 个标量），`(j−i)` 平面**每任务物化 1 次**（与 `k_idx` 无关的不变量），`PEN = 1e30` | **数学恒等 + 逐位一致**：`by·bm+i ≤ k_idx·bn+j ⟺ j − i ≤ d0`（两边同减 `by·bm+i`）；valid 时 `pen ≡ 0` ⇒ 与原式逐位恒等；invalid 时 `exp(有限负数 ≤ −1e30)` 在 fp32 下**下溢为精确 +0.0**，与源「置 0」逐位一致。**D-1 机器验证**：**110 案例**（`(bm,bn) ∈ {(64,64),(64,32),(32,32),(16,64),(1,1),(7,5)}` × **4 个掩码偏移 `d0`** × rand/large/zero/tiny/neg 五种分数分布；**80/110 案例含 masked 位置、共 61 740 个 masked 元素，6 个 tile 形状全部覆盖**）**最大 ulp 差 0、超容差元素 0 ⇒ EQUIV_PASS**；且**两种等价写法**（`clamp(diff−d0,0,1)·(−PEN)` 与 §3.3 伪代码的 `min(−PEN·(diff−d0), 0)`）**彼此及与源 select 三者互比仍为 0 ulp**（REVIEW.md 问题 5 已闭环）。**边界注**：非有限 score 的 masked 位会传播 NaN（源的 select 仍给 0）——该差异域在有限输入下为空（§0.6 R4 已登记） | 掩码路径由「每迭代 `(bm×bn)` 次标量条件选择」变为「每任务 1 次平面物化 + 每迭代 4 个整 tile 向量 op（causal 专属）」；且 `PEN` 由 `1e38` 降为 `1e30` 避免 `t·PEN` 中间量上溢为 `−inf`（仍给出精确 +0.0）。非 causal（**manifest 全部四档**）**零成本**：整条掩码链在 trace 期被 `if is_causal` 剔除 |
| **O2** | **`ds` 原位写回 `qkt` 载体（省一块 `[bn,bm]` fp32）** | 源：`dst_cast[i,j] = qkt[i,j]·(dst[i,j]−delta[j])·sm_scale` 写入**独立**的 `dst_cast` 缓冲；迁移直译需 3 块 `[bn,bm]` fp32（`qkt`(P) + `dsum` + 独立 ds） | `T.vsub(dsum, delta_scaled, dsum)` → `T.vmul(qkt, dsum, qkt)` → `T.vmul(qkt, sm_scale, qkt)`，即 ds **原位覆盖 P 载体 `qkt`** | 逐元素同一 fp32 减/乘/乘、同一操作数、同一目标元素；P 在 `qkt` 中的最后消费点是第 11 步（`T.vcast(qkt, pcast)` 已在第 7 步把 P 复制到 `pcast`，且 `pcast` 只被 g3 使用、g3 在第 9 步完成）⇒ 无读写别名冲突。**D-1 机器验证：逐位恒等（`torch.equal` = True）⇒ EQUIV_PASS** | `[bn,bm]` fp32 声明量 **3 块 → 2 块**（dim=128/bn=32 时 8 KB；bn=64 时 16 KB） |
| **O3** | **P 与 ds 的 dtype 转换 scratch 串行复用** | 源用 `qkt_cast` 与 `dst_cast` 两块 dtype 缓冲 | 一块 `[bn,bm]` dtype scratch：先 `T.vcast(qkt, scratch, rint)`（供 g3），g3 完成后再 `T.vcast(qkt, scratch, rint)`（此时 `qkt` 已是 ds，供 g4/g5） | 时序上两次写入不重叠（g3 在第 9 步、ds 在第 11 步之后）；每次内容与独立缓冲逐位相同。**D-1 机器验证：逐位恒等 ⇒ EQUIV_PASS** | dtype 缓冲 **2 块 → 1 块**（dim=128/bn=32 时省 4 KB） |
| **O4** | **指数基域重标定：`exp2(x)` → `exp(x')`（算子降代换，省 1 个整 tile 向量 op）** | `P = T.vexp2(qkt·scale − lse)`；`T.vexp2` 的 lowering 是 `VMul(x, ln2)` + `VExp`（`docs/Tilelang.language/数学操作/T.vexp2.md` §1 明示「底层不支持硬件级别 exp2，实际使用 `exp(A·ln2)`」），即**参数需先乘 ln2**（2 op） | `P = T.vexp(qkt·sm_scale − lse·ln2)`，`sm_scale = D^{-1/2}`、`lse·ln2` 在 `[bn,1]` 微缓冲上做 1 次（每迭代 1 个小 op，非整 tile）；`T.vexp` 为单 op `VExp` | **数学恒等**：`exp2(a) = exp(a·ln2)`，且 `(qkt·scale − lse)·ln2 = qkt·sm_scale − lse·ln2`（`scale·ln2 = D^{-1/2}·LOG2E·ln2 = D^{-1/2}·1.0000000000000002`，相对偏差 2e-16，远小于 fp32 舍入）。两条路径各有两次 fp32 舍入，属**容差内等价**。**D-1 机器验证**：60 案例（`(bm,bn)` 六档 × 5 分布去重后 × `D ∈ {64,128}`，`qkt` 为真实量级的 GEMM 输出、`lse = logsumexp(scores)/ln2`）**最大 ulp 差 16、超容差元素 0 ⇒ EQUIV_PASS**；且候选 vs fp64 oracle 的偏差（13 ulp）**不劣于**基线（13 ulp） | 整 tile 向量 op **9 → 8**、causal 档 **13 → 12**。**计数口径**（revision 1 统一，见 §1.6.0 R3 表头）：只计 `[bn,bm]` 规模的整 tile 向量 op，**不含** 2 个 `[bn,1]` 微 op、**不含**写出侧的行原子/区域 store。按 §1.6.0 的发射受限判定，等于**整 tile 向量 op 部分的发射项 −11%**（非整迭代发射项的 −11%——revision 1 后每迭代还含 `real_n` 个行原子，见 §1.6.0 估算表） |

**设计所依赖的数值等价性主张（同样以机器验证固化，供 Stage 2 复核）**：

| # | 主张 | 内容 | 机器验证结论 |
|---|------|------|--------------|
| **P1** | fp16/bf16 → fp32 上抛与 fp32 乘积**逐位精确** | `fp32(a)·fp32(b) ≡ a·b`（fp64 精确乘积），当 `a,b` 为 fp16/bf16：fp16 11+11 = 22 位 ≤ 24；bf16 8+8 = 16 位 ≤ 24 | 各 65536 元素随机样例：fp16 最大 ulp 差 **0**、bf16 **0** ⇒ 本设计的 fp32 负载链不引入额外误差 |
| **P2** | 累加**结合序**变化在容差内 | 源序 = 沿 `k_idx` 升序的 fp32 累加 + `T.atomic_add`；本设计序 = 同一升序累加（`Pipelined` 迭代序不变）+ **逐行 1-D 行原子**（`dq` 的跨 KV 块原子顺序由硬件定） | fp32 加法对 32 项（`⌈S/bm⌉`，manifest 档）的**最坏相对误差上界** `≈ (n−1)·2^-24 ≈ 1.8e-6`，远小于 op 级容差 `atol=5e-3, rtol=1e-5`（`examples/TileOPs/tests/ops/test_multi_head_attention.py` 第 251 行）。**边界注（诚实标注）**：`dq` 的原子顺序不可控 ⇒ **无法逐位复现**；L0 精度门禁以「与 fp64 参照的偏差不劣于源序」为准（§8.2） |
| **P3** | `P`/`ds` 回落到输入 dtype（就近舍入）与源的 fragment cast 同义 | 源 `T.copy(qkt, qkt_cast)` 在 CUDA 上是 fragment 内 fp32→fp16 RN 转换；本设计 `T.vcast(..., round_mode="rint")` 为就近偶数舍入 | 二者都是 IEEE 就近舍入（RN = RINT）；本设计**显式**指定 RINT 而非依赖跨 dtype `T.copy`（后者不可定制舍入模式，`TRAP-C12-copy-dtype-cast`） |
| **R1** | **否决候选证据**：`scale` 融入 K（`k' = dtype(scale·k)`） | 候选式 `q16 @ dtype(scale·k)ᵀ` vs 基线 `(q16 @ k16ᵀ)·scale` | 512×128 随机样例：**最大 ulp 差 2.52e9、最大相对偏差 3.25** ⇒ **数值否决，禁止采纳**（§1.6.0 R1 候选 6）：K 存于输入 dtype，预乘后回落输入 dtype 会量化 GEMM 操作数 |

**D-1 等价性机器验证结果表**（脚本 `examples/multi_head_attention/_gqa_bwd_kernel/verify_equiv.py`，2026-10-16 设计期执行，
torch CPU、fp64 参照 + 角点值 + 90 案例；判定双准则：**(a)** 候选式 vs 基线式在目标容差内（`atol = rtol = 1e-4`，超容差元素数 = 0）；
**(b)** 候选式对 fp64 精确数学的偏差不超过基线式偏差 + 一个容差预算）：

| 采纳 / 主张项 | 案例数 | 最大 ulp 差 | 超容差元素数（准则 a） | 准则 (b) 判否案例数 | 结论 |
|--------------|-------|------------|----------------------|-------------------|------|
| **O1 指数域惩罚掩码 + 平面不变量提升** | **110**（80 含 masked、6 tile 全覆） | **0** | **0** | 0 | **EQUIV_PASS**（逐位相等；两种惩罚写法互比同为 0 ulp） |
| **O2 `ds` 原位写回 `qkt` 载体** | 65536 元素 | **0** | **0** | 0 | **EQUIV_PASS**（逐位相等，`torch.equal`=True） |
| **O3 P/ds 的 dtype scratch 串行复用** | 2×8192 元素 | **0** | **0** | 0 | **EQUIV_PASS**（逐位相等） |
| **O4 `exp2(x)` → `exp(x')` 基域重标定** | 60 | **16** | **0** | 0（候选 13 ulp vs 基线 13 ulp，**不劣**） | **EQUIV_PASS** |
| P1 上抛 + 乘积精确性 | 2×65536 | **0** | **0** | 0 | **EQUIV_PASS** |
| P2 `dq` 原子结合序（设计序 vs 源序） | 分析性上界 | — | 上界 1.8e-6 ≪ 容差 | — | **EQUIV_PASS（分析性，见上表）** |
| P3 RINT 回落同义 | 分析性（IEEE RN） | — | — | — | **EQUIV_PASS（分析性）** |
| R1 否决候选（scale 融入 K） | 512×128 | 2543873248 | 最大相对偏差 **2.782** | — | **判否（未采纳）** |

形态覆盖（revision 1 扩）：`(bm,bn) ∈ {(64,64),(64,32),(32,32),(16,64),(1,1),(7,5)}` × **4 个掩码偏移 `d0`**、
分数分布 rand/large(σ=300)/zero/tiny(1e-6)/neg、`D ∈ {64,128}`、fp16 与 bf16 两 dtype；
O1 的掩码分支覆盖 80/110 案例（61 740 个 masked 元素），**每种 tile 形状都含 masked 位置**（REVIEW.md 问题 5 的修复）。
角点值覆盖：超容差边界（σ=300 的大分数）、全零、极小数、全负、`(1,1)`/`(7,5)` 退化 tile；
非有限域（NaN 分数在 masked 位的传播差异）单独以 `check_o1_nonfinite()` 记录为**域外说明**（不构成 FAIL）。
脚本已随算子目录留存，Stage 2 可直接重跑复核（`python verify_equiv.py`，无需 NPU）。

**优化结论**：采纳 **O1（指数域惩罚掩码 + 掩码平面循环不变量提升，逐位恒等）/ O2（`ds` 原位，逐位恒等）/
O3（dtype scratch 串行复用，逐位恒等）/ O4（指数基域重标定，容差内等价且不劣于基线）** 四项，
并以机器验证固化 **P1/P2/P3** 三项数值等价性主张、以 **R1** 的机器证据否决「scale 融入 K」候选。
**优化后公式（§3.1 的唯一输入）**：

$$
\begin{aligned}
x' &= \texttt{fp32}\big(q\cdot k^{\!\top}\big)\cdot \texttt{sm\_scale} \;-\; \texttt{lse}\cdot\ln 2
\qquad (\text{O4: base-}e \text{ exponent domain})\\
P &= \exp\big(x' + \texttt{pen}\big),\qquad
\texttt{pen} = \min\big(-\texttt{PEN}\cdot(j - i - d_0),\;0\big),\;\; \texttt{PEN}=10^{30}
\qquad (\text{O1})\\
ds &= P\cdot\big(\texttt{fp32}(do\cdot v^{\!\top}) - \delta\cdot\texttt{sm\_scale}\big)\cdot\texttt{sm\_scale}
\qquad (\text{O2: } ds \text{ written in place over the } P \text{ carrier})\\
dv &\mathrel{+}= \texttt{dtype}(P)^{\!\top}\cdot do,\qquad
dk \mathrel{+}= \texttt{dtype}(ds)^{\!\top}\cdot q,\qquad
dq \mathrel{+}= \texttt{dtype}(ds)\cdot k
\qquad (\text{O3: one shared dtype-conversion scratch})
\end{aligned}
$$

**否决项及依据**（负向断言举证，`negative-claim-evidence.md` §1；不满足条件者一律不采纳）：

| 否决候选 | 否决理由 | 证据 |
|----------|----------|------|
| C1 `scale` 融入 K（省 1 个 `[bn,bm]` vmul） | K 存于输入 dtype，预乘后回落输入 dtype **量化了 GEMM 操作数**，与「分数域缩放」不等价 | **本任务 D-1 机器证据**（最大相对偏差 3.25）；与 fwd 的 `prescale` 形态的差别在 fwd 是**先回落再算**（无二次量化） |
| C2 `lse` 折进 GEMM 增广维 | ① K 维变 `dim+1` 破坏 16 对齐与 K≥32 的 fractal 下界；② `−lse` 进入 GEMM 累加序改变舍入路径 | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.3（`fp16×fp16` 场景 dst 必须 fp32；未列 K 非 16 对齐的支持）；§1.6.0 R1 候选 7 |
| C3 跨 dtype `T.copy` 当 cast（省 `vcast`） | ① 不减向量 pass（编译器仍插 `hivm::VCastOp`）；② 舍入模式不可控（RINT 不可定制） | `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条 + `TRAP-C12-copy-dtype-cast`；同族先例 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/DESIGN.md` §1.6.1 C2 |
| C4 同 dtype `T.vcast`（fp32→fp32）当搬运 | 会退化为「四舍五入到整数」静默坏值 | `TRAP-vcast-same-dtype-rounds`（`f32→f32` `vcast == round(src) != src`）；本设计的 fp32 载体一律用 `T.copy` 或原位 v-op |
| C5 用 `T.vexp2` 保留源形态（省一次重标定） | 不是「不行」而是**多 1 个整 tile vmUL**（`vexp2` = VMul+VExp） | `docs/Tilelang.language/数学操作/T.vexp2.md` §1（「底层不支持硬件级别 exp2，实际使用 `exp(A·ln2)`」）；O4 已给出等价替代 |
| C6 逐元素 `T.atomic_add`（源逐字形态）+ `make_dq_layout` | ① 逐元素原子在 NPU 是 `(bn×dim)` 次标量散写，`make_dq_layout` 是 CUDA 专有 fragment 重排、无 NPU 对应；② **逐行 1-D 行原子**实测正确且 20 并发同址精确 | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.2/§2.4；`history_version/design_probe_row_atomic_bshd.py`（目标区全中/区外 0/20 并发精确 20.0）；`history_version/design_probe_atomic_vs_store.py`（**连续**区域原子 0.97× store） |
| C7 逐元素 `T.if_then_else` 因果掩码（源逐字形态） | 带 `T.if_then_else` 融合条件选择的 2D `T.Parallel` 在本工具链触发标量化 | `TRAP-parallel2d-bm-ge2-scalarize`（含该触发条件的适用域收窄记录）；`attention.md` PL-1.11（`vcmp`/`vselect` 掩码链在 causal 域实测 236 µs/链 ⇒ 改算术惩罚后 14.2×） |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点范围：本设计实现方案（§3.3 伪代码）中的**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
API 佐证均指向 `docs/Tilelang.language/` 下的具体文档（已逐份打开核对 dtype 表与 shape 规则）。

| # | 计算点 | 源实现形态（GPU） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|-------------------|--------------------------|---------|--------------|
| 1 | `q/k/v/do` 装载 | `T.copy(q[bz, s-slice, bx, :], q_frag)` | `T.copy(q[bz, c0:c0+real_n, bx, 0:dim], q_s[0:real_n,0:dim])`（**双显式 slice**，规避 `TRAP-T-copy-region-semantics` 的标量基址形态；dst 零起点规避 `TRAP-UB-dst-align`；slice 形态规避 `TRAP-load-nd2nz-strided`/CG-2026-0004） | ✅（形态替换） | —（MTE 搬运本身非标量计算） |
| 2 | `lse`/`delta` 行偏置预处理 | 源直接用 `lse_shared[j]` 参与逐元素式 | `T.vmul(lse_s, LN2, lse_s)` / `T.vmul(delta_s, sm_scale, delta_s)`（`[bn,1]` 微缓冲，1 op/迭代） | ✅ | — |
| 3 | **P 的分数域缩放 + 偏置**（B1） | `qkt[i,j]·scale − lse_shared[j]`（`T.Parallel` 逐元素） | `T.vmul(qkt, sm_scale, qkt)` + `T.vsub(qkt, lse_nat, qkt)`（**行广播** `[bn,bm] ∘ [bn,1]`，`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2「行广播 `[M,N]*[M,1]`」+ `T.vsub.md` 同款） | ✅ | — |
| 4 | **因果掩码**（B2） | `T.if_then_else(...)` 逐元素条件选择 | 指数域惩罚（O1）：`T.vsub(dplane,d0,dsum)` + `T.vmul(dsum,-PEN,dsum)` + `T.vmin(dsum,0,dsum)` + `T.vadd(qkt,dsum,qkt)`（4 个整 tile op，**全向量化**）；平面 `dplane` 每任务物化 1 次。**单一定义源 = §3.3**（`t ≡ dsum`；不用 `T.vmax`，其 src 不接受标量） | ✅（域重排） | — |
| 5 | **`exp2`**（B3） | `T.exp2(...)` 逐元素 | `T.vexp(qkt, qkt)`（单 op；O4 已把基域重标定到 `e`） | ✅ | —（换 `T.vexp2` 则为 2 op，见 §1.6.1 C5） |
| 6 | **P 回落输入 dtype**（B4） | `T.copy(qkt, qkt_cast)`（fragment cast） | `T.vcast(qkt, scratch, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1：`f32→f16` rint ✓、`f32→bf16` rint ✓；§2.2.2 shape 须一致） | ✅ | — |
| 7 | **`ds` 计算**（B5） | `dst_cast[i,j] = qkt[i,j]·(dst[i,j]−delta_shared[j])·sm_scale`（`T.Parallel` 逐元素） | `T.vsub(dsum, delta_scaled, dsum)` + `T.vmul(qkt, dsum, qkt)` + `T.vmul(qkt, sm_scale, qkt)`（3 个整 tile op；O2 原位） | ✅ | — |
| 8 | **`ds` 回落输入 dtype**（B6） | `dst_cast` fragment | `T.vcast(qkt, scratch, round_mode="rint")`（O3 复用 scratch） | ✅ | — |
| 9 | **`dq` 逐行累加**（B7，revision 1 修订形态） | `for i, j in T.Parallel(bn,dim): T.atomic_add(dq[bz, k_idx·bn+i, bx, j], dq_frag[i,j])`（**逐元素原子**）+ `make_dq_layout` | `for r in T.serial(real_n): T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r, 0:dim], [1, dim])`（**逐行 1-D 行原子**：`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.2「对形状无特殊限制」+ §2.4 区域形态；生产先例 `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` 第 257 行同款 `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])`；本任务探针三正面验证：目标区全中、区外 0、20 并发同址精确）。**行体（`dim` 连续段）本身是向量宽的 1-D op**；外层 `T.serial(real_n)` 是把跨步行区域分解为逐行（唯一可用形态，见下方结论的保留项） | ✅（形态替换：逐元素散写 → 逐行 1-D 行原子，标量散写 op 数 `(bn×dim)` → `real_n`） | — |
| 10 | **`dv`/`dk` 写回**（B8） | `T.copy(dv_frag, dv_shared)` + `T.atomic_add(dv[区域], dv_shared)` | `groups == 1`：`T.copy(dv_acc[0:real_m,0:dim], dv[bz, r0:r0+real_m, h_kv, 0:dim])`（**普通区域存储**，单写者；rank-reduced 4-D 目标由 `T.copy` 正确写出，探针二 ①② 与 `T.copy` 语义一致）；`groups > 1`：**逐行 1-D 行原子** `for r in T.serial(real_m): T.atomic_add(dv[bz, r0+r, h_kv, 0:dim], dv_acc[r, 0:dim], [1, dim])`（`dk` 同形） | ✅（工厂期分派 + 逐行形态） | — |
| 11 | `dv`/`dk` 的**跨 kq 累加**（B9） | `dv_frag`/`dk_frag` fragment 累加 | `T.gemm(..., initC=False)` 累加进 UB 载体（跨 `T.Pipelined` 迭代） | ✅（本工具链已验证形态） | —（非标量计算点：累加由 Cube 的 L0C 累加器承担，无向量开销） |
| 12 | **query 块循环**（B10） | `T.Pipelined(loop_st, loop_ed, num_stages)` | `T.Pipelined(loop_st, loop_ed, num_stages=ns)` | ✅（保留并重设 stage） | — |
| 13 | 尾块界 `real_m` / `real_n` | 源无显式谓词（靠 `ceildiv` 超发 + 隐式裁剪） | `real_m = T.min(bm, S − by·bm)`、`real_n = T.min(bn, S − c0)`（标量，每任务/每迭代 1 次，`docs/Tilelang.language/比较操作/T.min.md` 标量语义 ✓） | ❌ 保留 | **依赖动态边界的边界处理**：`S % bm/bn ≠ 0` 时尾任务的有效行数（同时也是探针三的**动态行数** `T.serial(T.min(bn, S−row0))` 的上界）；每迭代 1 次标量 `min` + 两次切片界，`CONST-aiv-time-decomposition` 的判据句「`vec + scalar > mte2` ⇒ 计算受限」在本处不成立（标量开销被 MTE2 搬运算覆盖）⇒ 不构成热点；静态化需为尾块复制整份 kernel 体，复杂度不值 |
| 14 | 任务映射索引（`uid → (bz,bx,by)` 整除/取模 + grid-stride） | 源为三维 grid（无核内映射） | `for i in T.serial(num_local_tasks)` + `if uid < num_logical` + 整除/取模分解 | ❌ 保留 | **block 级索引 / 任务映射计算**：每任务数个标量索引（`uid`/`r0`/`c0`），非逐元素热点；该 `T.serial` 循环是 §0.6 R1 persistent 分核的载体，无逐元素等价的向量 API；同族先例 `examples/deepseek_v32/sparse_mla_bwd.py`（`cid` 分解）与 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/DESIGN.md` §1.6.2 #8 |
| 15 | causal 掩码平面 `dplane[i,j] = j − i` 物化 | 源无（每迭代重算索引比较） | `T.Parallel(bn,bm)` 内的纯整型/浮点算术赋值（**每任务 1 次**，非逐迭代） | ❌ 保留（每任务 1 次） | **依赖动态边界的 tile 级初始化**：`dplane` 的物化是每任务一次（迭代数的 `1/(S/bn)` 倍），且 `T.Parallel` 体内是单一算术表达式（`TRAP-parallel2d-bm-ge2-scalarize` 的适用域收窄记录明示：**普通逐元素 2D loop 正常向量化**，触发条件是 `T.if_then_else` 融合条件选择——本点不含条件）。备选：`T.arange` 派生（`CONST-arange-scalar-materialize` 记其按标量执行 ~1 ns/元素，反而更贵）⇒ 保留 `T.Parallel` 算术形态 |
| 16 | causal 惩罚标量 `d0` | 源无（掩码常量内联） | 每迭代 1 个标量 `d0 = kq·bn − by·bm` | ❌ 保留 | **block 级索引算术**（每迭代 1 次标量，非逐元素热点） |
| 17 | `bm`/`bn`/`num_logical`/`num_kernels`/`num_local_tasks`/`loop_ed` | host Python（源为 `default_config`） | host 层编译期常量（trace 期折叠为 `IntImm`） | ❌（非 kernel 内） | **host 侧 shape 元数据计算**：不在 kernel 内，无向量化问题 |
| 18 | 标量广播点（bias / per-row scale） | 源的 `lse`/`delta` 逐元素引用 | 已由 #2 的 `[bn,1]` 微缓冲 + 文档化行广播承接 | ✅ | — |

**向量化结论**：全部**逐元素/归约/条件**计算点（#3–#10，共 8 个）**已全部向量化**，
替代 API 均在 `docs/Tilelang.language/` 有 dtype 与 shape 佐证（上表）；其中 **#4 掩码**与 **#9 原子**是源逐字形态会在 NPU 标量化的两处
（`TRAP-parallel2d-bm-ge2-scalarize` / 逐元素散写），已改为指数域惩罚与**逐行 1-D 行原子**。
保留的 **6 类**循环/标量为：**动态边界 `T.min`（#13）**、**block 级任务映射（#14）**、**每任务一次的平面物化（#15）**、
**每迭代一个掩码标量（#16）**、**host 侧元数据（#17）**、以及 **revision 1 新增的「逐行行原子的行步进 `T.serial(real_n)`」（#9 的行循环）**
——最后一项的理由：BSHD 目标的连续行 stride = `H·D ≠ dim`，**跨步行区域无法用单条区域 op 表达**（rank-reduced 4-D 区域 op 实测
编译失败或静默写错，见 §0.6 R3）；逐行分解是唯一可用形态，且**行体本身是向量宽的 1-D op**（`dim` 连续段，
dim ∈ {64,128} ≥ 向量宽），故该 `T.serial` 是把向量化 op 逐行发射的**结构性循环**、非逐元素标量热点。
逐项理由如上（均为「每任务/每迭代 1 次」/「block 级」/「kernel 外」/「结构性行步进」，
**无「实现简单」类理由**）。§6 循环结构与本表一致：kernel 内**不存在**逐元素标量循环
（源 S6/S7/S10/S13 的 `T.Parallel` 元素循环与逐元素原子已全部消除）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

本算子属 **Cube / MixCV 类**（§1.6.0 选定）。I/O layout（BSHD 的 `q/k/v/do/dq/dk/dv`、`(B,H,S)` 的 `lse/delta`）是**契约**，不在候选之列；
按本类别必选候选枚举：**P/O tile 的轴向声明**、**fractal/NZ 布局路径（load_nd2nz vs 显式转置）**、**epilogue 归属侧（Cube vs Vector）**、
**行/列偏置的广播轴**；另枚举「装载粒度 / 布局重排」方向的候选。

**轴质量评分**（主选：向量化轴 = 连续尾轴 `dim`；P tile 声明为 `[q 行, kv 列]`，行偏置沿第 0 轴广播）：

| 评分项 | 判定 |
|--------|------|
| 对齐 / 整除性 | 尾轴 `dim ∈ {64,128}`：fp16/bf16 向量宽 ×8 ⇒ `64 = 8×8` ✓、`128 = 8×16` ✓；`bm = 64`、`bn ∈ {32,64}` 均为 32 的整数倍（同时满足 Cube 的 M/N≥16、K≥32 与向量宽） |
| 尾 lane 浪费率 | manifest 两档 **0%**；`dim % 8 != 0` 的防御性微型用例按工厂 guard 拒绝（§5.3） |
| 累加链形态 | GEMM 的 K 维分别为 `dim`（g1/g2）、`bn`（g3/g4）、`bm`（g5），均 ≥32 且 16 对齐；向量链的归约/广播无跨步（`lse/delta` 是**行广播**，非列因子） |
| repack 代价 | **无**（I/O 原生 BSHD 序 = 核内布局；`dplane` 是新数据不是 repack） |
| UB 容量影响 | `E = max(bm·bn·2, bm·dim, bn·dim)` 元素；manifest 下声明 100–136 KB（§4.5） |
| 装载粒度 | 每行 burst = `dim·eb`（128 B @D=64 / 256 B @D=128），行间 stride = `H(· Hkv)·D·eb`；与 attention 族主 kernel 的装载形态一致（§3.5.3） |

**候选矩阵**（弃选项均给量化理由与证据）：

| # | 布局方案 | 向量化轴 / lane 语义 | repack / 结构路径 | 预估收益 / 代价 | 采纳？ |
|---|---------|---------------------|------------------|-----------------|--------|
| 1 | **源轴向（`qkt = K·Qᵀ`，声明 `[kv 行, q 列]`）+ `lse/delta` 列因子** | lane → 尾轴 `q 列` | 无 repack | `lse/delta` 是**列因子** ⇒ 需 `vbrc` 展开成 `[bm,bn]` 平面（+2 个全尺寸缓冲 +2 op/迭代）或吃 `PL-1.15` 记的**未文档化列广播税 +12~16%** | ❌（可免费避免：候选 2 把同一信息变成文档化行广播） |
| 2 | **转置 P tile 声明 `[q 行, kv 列]`（`qkt = Q·Kᵀ`）+ `lse/delta` 行因子** | lane → 尾轴 `kv 列`；行偏置 → 第 0 轴广播 | 无 repack（**仅声明秩/轴序变化，不移动任何元素**） | `lse/delta` 命中 `T.vmul`/`T.vsub` **文档化行广播** `[M,N] ∘ [M,1]`；零额外缓冲、零额外 op；且与同包 fwd 的 `S = Q·Kᵀ` 轴向一致（族内一致） | ✅ **主选** |
| 3 | 显式 NZ / fractal 布局（`T.alloc_L1` + `T.load_nd2nz` + `T.store_fixpipe`） | Cube MAC | Expert-only 通路 | 需要切换 Expert 编程模式 | ❌（cube skill 明示「NZ format path is Expert mode only；In Developer mode kernels, keep ND layout and use T.copy-based data movement」；本任务编程模式为 Developer） |
| 4 | 经 `T.transpose` 做核内重排换取「归约轴非最内」 | — | 核内二轴交换链 | 本算子**不存在归约轴**（无 row-reduce）；纯开销 | ❌（`TRAP-transpose-ub-gm-roundtrip`：核内转置在 3D/非向量宽整除形态会降级为 UB→GM→UB 往返，实测 5.6 → 138.2 µs） |
| 5 | H-collapse（把 `H` 折进 tile，连续大 burst 装载） | lane → `dim` | 无 repack | 读 burst 变连续 `bm·H·(Hkv)·D·eb`；代价：累加器与 P tile 全部按 `H` 放大（`dv_acc [bm·H, dim]` fp32 在 `H=16` 时 = 512 KB ⇒ **UB 结构性超限**）；且 `dq` 的原子区域变成 `[bn·H, dim]` 的跨 head 非连续段 | ❌（UB 结构性不可行；同族先例 `_flashattn_bwd_preprocess_kernel` §1.6.3 候选 3 在 `H·D` 大时的同类退化） |
| 6 | `dim` 分块（K-split 的 GEMM） | lane → `dim_c` | 需 K 维分段累加 | 累加器变小，但 **P 链（掩码/exp2/cast）会按 `dim` 分块数重复**（P 与 `dim` 无关）⇒ 向量 pass 数 ×分块数 | ❌（发射项反增；§1.6.0 判定本算子发射受限） |
| 7 | epilogue 交 Cube 侧（`store_fixpipe` 直写 GM） | — | L0C→GM | 本算子的 dq/dk/dv 需要**跨任务累加**（原子），`store_fixpipe` 只支持 L0C→GM 且**不带原子语义** | ❌（`CONST-store-fixpipe-gm-only`：store_fixpipe 仅支持 L0C→GM，无累加能力；且本设计的 dk/dv 累加器在 UB 而非 L0C） |

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：源 kernel 把 `qkt` 声明为 `(block_m, block_n)` = `(kv, query)`，
并把 `lse_shared[j]`（`j` 为 query 下标）作为**列因子**参与逐元素式——这是 CUDA 的产物：
线程网格 `T.Parallel(block_m, block_n)` 让 coalescing 沿最内轴，且 CUDA 上列因子与行因子的代价**没有区别**。
NPU 上两者代价**不对称**（行广播文档化、列因子需 `vbrc` 或吃未文档化税），故**独立评估**后选择
`[q 行, kv 列]` 声明（候选 2）；同时 lane 语义与源不同（源在 256 线程块内并行，本设计在 20 个核内按任务切分，§5.5）。

**布局决策结论**：选定 **候选 2——核内布局 = I/O 原生 BSHD（跨步行块装载）+ P tile 声明 `[bn(q) 行, bm(kv) 列]`（`Q·Kᵀ` 轴向）
- `lse/delta` 为文档化行广播 `[bn,1]` + 向量化轴 = 连续尾轴 `dim`/`kv 列` + repack = 无**。
该结论同步落入 §3.3 伪代码、§4.3 缓冲表与 §6 循环结构（所有整 tile v-op 的 buffer 形状 = `[bn,bm]`、
`dv_acc`/`dk_acc` = `[bm,dim]`、`dq_blk` = `[bn,dim]`）；弃选候选的量化理由见上表（#1 可免费避免、#3 模式不符、#4 无归约轴、#5 UB 超限、#6 发射项反增、#7 无原子语义）。

**参数级 A/B（参数级，故不触发结构级实验裁决三件套）**：

- **主选（默认实现，进 Stage 3）**：`(block_m, block_n, num_stages)` = wrapper `default_config` **原样**
  = `(64, 64, 1)`（`dim ≤ 64`）/ `(64, 32, 1)`（`dim > 64`）。无需任何 trace-time 分派表（**与冻结 wrapper 零摩擦**），
  声明 UB 100 KB（dim=128）/ 88 KB（dim=64）**在两个 UB 口径（×1.12 与 ×1.7）下均可行**（§4.5）。
- **参数级备选（Stage 4 可切换，不改结构）**：`dim > 64` 时把 `block_n` 由 32 上探到 **64**
  — 迭代数 `B·H·(S/bm)·(S/bn)` 减半（full 档 524288 → 262144），但每迭代 issued-op 数由 `10+32 = 42` 升到 `10+64 = 74`
  （revision 1 重算，行原子数随 `real_n` 增大）⇒ 上界档发射项 **≈0.88×**（不是 revision 0 记账下的「减半」）；
  **元素吞吐地板档不随 tile 尺寸变化**（总元素工作量守恒，`20544×524288 = 41088×262144`，见 §1.6.0 估算表）；
  代价：声明 UB 100 → 136 KB ⇒ ×1.12 口径 152 KB ✓、**×1.7 口径 231 KB ✗** ⇒ 该档**要求 auto multi-buffer 的实际膨胀系数 < 1.4**
  （`CONST-capacity-910B2C` 记默认档 ~1.7x、显式 alloc 结构 ~1.10–1.12，本设计属后者形态但无实测）。
  **可行条件**：编不过（`ub overflow`）即自动落回主选档（工厂 guard + 阶梯，§4.5/§8.3）。
- **裁决计划**：代表 shape = manifest 四档；指标 = msprof op `Task Duration`（每档 ≥20 次采样、去首 5 次预热）
  ＋该档的 UB 编译需求（`ub overflow` 报文反解，反解**真实 auto multi-buffer 膨胀系数**）
  ＋`aiv_vec_time`/`aiv_mte2_time`/`aiv_mte3_time` 分解（反解**真实向量 op 单价**与**逐行行原子单价**——
  后者是 revision 1 新增的未解析常数，见 §1.6.4）；
  **判定阈值**：备选在该档 `Task Duration < 主选 × 0.95`（且 L0 精度门禁不回归）才翻转该档；
  **按 shape 分派**（per-`dim` 表）允许不同档各自选择；裁决结果经附录补记回写本节，反解出的常数同步作为 pattern-library 候选常数。
- **不启动设计期探针（D-3）**：本决策只影响**参数**（tile 面积 / `block_n`），不改结构；**三枚**设计期探针已用尽于
  「原子单价」「persistent+跨步正确性」「修正形态的行原子正确性与单价」三个**结构级前提**上（§1.6.0）。

#### 1.6.4 备选方案（结构级，供 Stage 4 A/B；主选胜出则仅存档）

> 依据 Phase 2 第 3 项：备选 = §1.6.0 R3 候选 2「**两趟原子自由**（FA2 式 dk/dv 与 dq 分 kernel）」。
> **revision 1 重述其成因（REVIEW.md 问题 3/6）**：备选的理据**不再是**「口径 B 下备选反超」
> （该论断在统一口径 C 下不成立：主选 17.98 GB vs 备选 27.1 GB，见 §1.6.0 R3 口径注③），
> 而是**两个当前未解析的量同时朝同一方向拉**：
> ① **零原子** ⇒ 免疫「同址并发原子」的正确性/性能不确定性（对应 §9 R1）；
> ② **revision 1 的行原子发射项**：主选每迭代 issued-op `10 + real_n`（=42 @ dim=128），
> 备选每迭代 `10` 但迭代数 ×2 ⇒ **聚合 issued-op 10.5 M vs 主选 22.0 M**（上界档单价下备选 ≈ 主选的 0.48×）。
> 两趟的代价（GEMM 7 vs 5、元素吞吐 ×2）在**元素吞吐档**下反过来使主选占优（9.5 ms vs 15.2 ms）⇒ **两个价格档给出相反排序**
> ⇒ 按实验裁决模式给出三件套，由 Stage 4 实测**行原子单价**决胜。

| 维度 | 主选（单趟 + 逐行行原子） | 备选 B（两趟原子自由） |
|------|-----------------------|----------------------|
| 工厂返回 | `@tilelang.jit(target="npuir")` 工厂直接返回编译后的 callable（与源同构） | **闭包**：工厂返回 `wrapped(q,k,v,do,lse,delta,dq,dk,dv)`，内部顺序调用两个 npuir kernel（**先例**：同包 `_gqa_prefill_fwd_kernel.py` 返回 `wrapped(q,k,v)` 闭包；op 层本就串行调用 prep/main/post 三次 launch） |
| 任务划分 | 逻辑任务 `(bz, bx, by)`，**每次迭代**把 `dq_blk` 逐行行原子累加到 `dq` | pass A：`(bz, bx, by)`（KV 块 owner）逐 query 块累加 `dv_acc/dk_acc`，任务末**普通区域存储**（两趟 owner 维完整 ⇒ 单写者，`groups` 无关）；pass B：`(bz, bx, by')`（query 块 owner）逐 KV 块累加 `dq_acc`，任务末**普通区域存储** |
| GEMM | `qkt=Q·Kᵀ`、`dsum=dO·Vᵀ`、`dv += PᵀdO`、`dk += dsᵀQ`、`dq = ds·K`（**5**） | pass A：`qkt`、`dsum`、`dv`、`dk`（**4**）；pass B：`qkt`、`dsum`、`dq`（**3**）；合计 **7** |
| issued-op / 迭代 | `10 + real_n` = **42**（非 causal，dim=128） | pass A `10` + pass B `10` ⇒ 每趟 10；**聚合**（×2 迭代数）= 10.5 M vs 主选 22.0 M |
| 原子 | dq：每迭代 `real_n` 个逐行行原子；dk/dv：`groups==1` 普通 store | **无**（两趟的 owner 维完整 ⇒ 单写者；dk/dv/dq 全走 `T.copy` 区域存储） |
| dq 的 dtype 回落 | 每迭代 `vcast` | 每迭代 `vcast`（pass B 仍需 ds 回落输入 dtype 才能用 fp16 GEMM） |
| UB 声明（dim=128, (64,32)） | **100 KB**（含 `dq_blk`） | pass A **84 KB**（−`dq_blk`）/ pass B **36 KB**（−`dv_acc`/`dk_acc`；causal 档 +`dplane` 8 KB）——各自独立预算，更松 |
| 分核三要素重算（manifest） | full：逻辑核 8192 → 20 核 / 410 轮；smoke：128 → 20 核 / 7 轮 | pass A：同主选（8192 → 410 轮）；pass B：`B·H·⌈S/bn⌉` = 16·16·64 = 16384 逻辑核 → 20 核 / **820 轮**（每轮迭代数与 pass A 相同 ⇒ 两趟合计每核迭代数 ×2） |
| 优势 | GEMM 5（元素吞吐档 **9.5 ms** 胜）、口径 A/C 流量最小（1.34 GB）、单次 launch、UB 最省 | **零原子**（免疫同址并发不确定性）；上界档聚合 issued-op **0.48×** ⇒ 上界档 **131–262 ms** 胜主选 **275–550 ms** |
| 代价 | `dq` 由跨核并发行原子写（正确性已由探针三正面覆盖）；上界档被 `real_n` 行原子发射项拉高 | GEMM 1.4×、元素吞吐 ×2（地板档 15.2 ms vs 主选 9.5 ms）；两次 launch；闭包形态偏离源的「单 jit callable」 |

**实验裁决计划**：代表 shape = manifest 四档；指标 = msprof op `Task Duration`（每档 ≥20 次采样、去首 5 次预热）
＋ `aiv_vec_time` / `aiv_mte2_time` / **`aiv_mte3_time`**（反解**逐行行原子单价**——revision 1 新增的未解析常数，
即解答「32 个行原子 ≈ 1 个区域 store 是否只是在探针地台下不可分辨」，见 §1.6.0 探针三⑤）
＋ `GM_to_GM_datas` / L2 命中计数（反解「dq 原子放大的 L2 吸收率」）；
**判定阈值**：备选在该档 `Task Duration < 主选 × 0.95` 且 L0 精度不回归 ⇒ 翻转该档（按 shape 分派，允许仅大 shape 档走备选）；
裁决结果回写本节并作为常数候选（含工具链版本戳）提交 pattern-library。
**不启动设计期探针**：两个待定量（行原子单价、L2 吸收率）都需要**真实 kernel 的稳态画像**才能反解
（探针的孤立微基准无法代表——探针三⑤即为该局限的实证：其 ~120 µs/launch 地板掩盖了每 op 单价），
且三枚探针预算已用尽于结构级前提；主选的先验证据是 pattern-library 在同类 attention kernel 上反复观测到的高 L2 命中率
（PL-1.18：`cube read_hit 91% / AIV 99%`）。
**结构级替代之二的精确掩码形态**：若 §8.3 的边界用例要求 masked 位的 NaN 语义与源逐位一致，
把 O1 的惩罚式替换为 `vcmp(dplane, d0, cond, "gt")` + `vselect(cond, 0, P, P)`
（**值替换语义**，NaN 也归 0；`attention.md` PL-1.11 记该链在 causal 域有 int16 标量化陷阱 ⇒ 须以 fp32 平面驱动 `vcmp`，
且按 `TRAP-hivm-vsel-cond-slice` 用专用 bool 缓冲而非切片）。该替换只影响 causal 档，属**局部形态**而非结构级备选。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（`op_requirements.programming_mode = developer`），附 **Expert 双 Scope 备用形态**（同结构，仅显式化引擎边界与同步，切换条件见 §2.3）。

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | **MixCV**：Cube 侧 5 处 `T.gemm`；Vector 侧 `T.vmul`/`T.vsub`/`T.vmin`/`T.vadd`/`T.vexp`/`T.vcast`/`T.atomic_add` 等 v-prefix 算子（满足 `tilelang-mixcv-skill` 的 Developer 模式 MixCV 判据） | Developer 的 auto CV split 可直接承接，无需手写双 Scope |
| 计算缓冲 | 全部用 `T.alloc_shared`（Developer 映射 UB/L1，`docs/Tilelang.language/内存操作/T.alloc_shared.md` §2.4 示例 1 明示 matmul 场景下申请 L1）；gemm 操作数与 UB 计算缓冲由编译器按用途分派 | Developer 形态即可，无需 `T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub`（后三者为 Expert-only） |
| NZ / fractal 布局 | cube skill 明示「NZ format path is Expert mode only；In Developer mode kernels, keep ND layout and use T.copy-based data movement」 | Developer 保持 ND，与 §1.6.3 候选 2 一致 |
| 同步 | 单 kernel 内的 Cube→Vector→Cube 依赖链；dq 的跨核归并由硬件原子承担 | Developer 自动同步即可，无需 `sync_block_set`/`sync_block_wait`/`pipe_barrier` |
| 流水 | query 块循环有可重叠的 K 维（`T.Pipelined` 承载） | Developer 的自动多缓冲可承接 |
| 先例 | 三个同语义 npuir 实现（`examples/deepseek_v32/sparse_mla_bwd.py`、`examples/deepseek_v4/example_sparse_attn_bwd_kernel.py`、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`）**全部是 Developer + `T.Kernel(n, is_npu=True)`（无 `T.Scope`）+ `T.gemm` + v-prefix 混排**，本设计原语组合与之逐项同族 | Developer 已验证可行，无阻塞证据 |
| persistent + gemm 的旧阻塞 | pattern-library `TRAP-DEVMODE-PERSIST-GEMM`（910B2C + CANN 8.5.0）称该结构在 Developer 下运行期崩溃 | **本任务设计期探针二 Q3 在本工具链（910B3 + CANN 9.0.0）复验：编译通过且数值正确** ⇒ 该 trap 按证伪协议「工具链变更后自动待重验」已被本机证据推翻，**不构成切换 Expert 的理由**（§1.6.0 探针表） |
| 用户指定 | `op_requirements.programming_mode = developer` | 一致 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译期按用途分派 UB / L1；v 算子操作数落 UB，`docs/Tilelang.language/数学操作/T.vmul.md` §2.3 第 1 条） |
| 计算方式 | Cube：5 处 `T.gemm`（`b_transpose`/`a_transpose` + `size=` 尾块裁剪）；Vector：v-prefix 整 tile 算子 + **逐行 1-D 行原子**（`T.atomic_add(dst_row, src_row, [1, dim])`，跨步 BSHD 目标的唯一可用形态） |
| 同步 | Developer 自动同步（**无任何手动同步原语**） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维；`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1：`is_npu` 最多 2 维 block） |
| 输出分配 | **无 `out_idx`**（源契约：9 个参数全为入参，`dq/dk/dv` 由 op 层预置 0 后传入） |
| 环境变量 | `ASCEND_MODE = "Developer"`（文件顶层写死并 `os.environ.setdefault("TILELANG_ASCEND_MODE", ASCEND_MODE)`；同批前作与本包 fwd 同款纪律） |
| `pass_configs` / `compile_flags` | **均不使用**（源工厂的 `TL_ENABLE_FAST_MATH` 与 `compile_flags` 在 npuir 无对应；本任务要求 `target="npuir"` 下不加 `compile_flags`/`pass_configs`） |
| **若 Developer 路径在本环境仍失败** | 按 `attention.md` PL-1.16 的「Expert 双 Scope 绕法」切换：同结构、同算法，仅显式化 `T.Scope("Cube")`（`T.alloc_L1`/`T.alloc_L0C` + `T.gemm`）/`T.Scope("Vector")`（`T.alloc_ub` + v 链）+ 跨引擎数据经 GM workspace + per-slot flag；wrapper 契约不变（**闭包工厂先例**见同包 `_gqa_prefill_fwd_kernel.py`）。触发条件：Stage 3 首编/首跑命中 `TRAP-DEVMODE-PERSIST-GEMM` 或 CG-2026-0001（Pipelined 体内条件构造崩溃）类 |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 的优化后公式（fp32 计算域，O1/O2/O3/O4 已采纳）。步骤编号与 §3.3 伪代码一一对应：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `uid = it·num_kernels + cid`；`(bz,bx,by) = unflatten(uid)`；`hkv = bx//groups`；`r0 = by·bm`；`real_m = min(bm, S−r0)` | 任务映射与尾块界（标量） |
| 2 | `k_s ⟵ k[bz, r0:r0+real_m, hkv, :]`；`v_s` 同形 | 任务级 K/V 装载（L1，任务内一次） |
| 3 | `dv_acc = 0`；`dk_acc = 0`；causal 时 `dplane[i,j] = j − i` | 累加初值与掩码平面（每任务一次） |
| 4 | `c0 = kq·bn`；`real_n = min(bn, S−c0)`；`q_s ⟵ q[bz, c0:c0+real_n, bx, :]`；`lse_s ⟵ lse[bz,bx,c0:…]`；`lse_s ← lse_s·ln2` | query 块装载与自然指数域行偏置 |
| 5 | `qkt = q_s · k_sᵀ`（`[bn,bm]` fp32） | GEMM g1 |
| 6 | `qkt ← qkt·sm_scale − lse_s`（行广播） | B1：分数域缩放与偏置 |
| 7 | causal：`t = dplane − d0`；`t ← t·(−PEN)`；`t ← min(t, 0)`；`qkt ← qkt + t`（`d0 = c0 − by·bm` 标量） | B2：掩码（O1） |
| 8 | `qkt ← exp(qkt)` | B3：概率（O4） |
| 9 | `cast ← rint_dtype(qkt)`（P 回落输入 dtype） | B4 |
| 10 | `do_s ⟵ do[bz, c0:c0+real_n, bx, :]`；`dsum = do_s · v_sᵀ` | GEMM g2 |
| 11 | `dv_acc ← dv_acc + castᵀ · do_s` | GEMM g3（跨 kq 累加） |
| 12 | `dl_s ⟵ delta[bz,bx,c0:…]`；`dl_s ← dl_s·sm_scale`；`dsum ← dsum − dl_s`；`qkt ← qkt·dsum`；`qkt ← qkt·sm_scale` | B5：ds（O2 原位） |
| 13 | `cast ← rint_dtype(qkt)`（ds 回落，O3 复用 scratch） | B6 |
| 14 | `dk_acc ← dk_acc + castᵀ · q_s` | GEMM g4（跨 kq 累加） |
| 15 | `dq_blk = cast · k_s`；`for r: dq[bz, c0+r, bx, :] += dq_blk[r, :]` | GEMM g5 + B7 **逐行 1-D 行原子** |
| 16 | `groups==1`：`dv[bz, r0:r0+real_m, hkv, :] ← dv_acc`（`dk` 同形，普通区域存储）；否则 `for r: dv[bz, r0+r, hkv, :] += dv_acc[r, :]`（`dk` 同形，逐行行原子） | B8 写回（工厂期分派） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数与形态 | 模式 | 依据 |
|------|----------|-------------|-----------|------|------|
| 工厂 | — | `@functools.lru_cache(maxsize=32)` 外层 + 内层 `@tilelang.jit(target="npuir")` builder | 外层签名与源一致（7 位置参数 + `dtype` 默认值）；内层 `(block_m, block_n, num_stages)`（**K9 无 `threads`**） | Developer | 源同款外层形态；内层 npuir 形态先例 `examples/deepseek_v32/sparse_mla_bwd.py`、`examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/_flashattn_bwd_preprocess_kernel.py` |
| 核数实查 | — | `NPUUtils.get().get_aicore_num()` | 本机实查 **20**（910B3；**混/ Cube 算子直接用返回值**，不翻倍） | Developer | `.agents/skills/_shared/standards/core-split-strategy.md` §1 要素② |
| Kernel | — | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` | 一维（≤2 维） | Developer | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1 |
| 任务循环 | `it` 遍历 | `for it in T.serial(num_local_tasks)` + `if uid < num_logical` | **静态界**（host 编译期常量折叠） | Developer | 分核标准 §1 要素③；`PL-1.21-vector-persistent-split`；同族先例 §3.5.3 |
| 装载 `k/v/q/do` | GM → L1 | `T.copy(k[bz, r0:r0+real_m, hkv, 0:dim], k_s[0:real_m, 0:dim])` | **双显式 slice** + dst 零起点 + 同 dtype | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2 规则 1/2/4 + §2.4；规避 `TRAP-T-copy-region-semantics` 与 `TRAP-UB-dst-align`、`TRAP-load-nd2nz-strided` |
| 装载 `lse/delta` | GM → UB | `T.copy(lse[bz, bx, c0:c0+real_n], lse_s[0:real_n, 0])` | 1-D 连续段 → `[bn,1]` 微缓冲 | Developer | 同族先例 `examples/deepseek_v32/sparse_mla_bwd.py` 第 59 至 60 行的同类 1-D 段拷贝 |
| 行偏置预处理 | `lse·ln2` / `delta·sm_scale` | `T.vmul(lse_s, LN2, lse_s)` | `[bn,1]` 微缓冲，标量 src2 | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` §2.1（src2 支持 scalar）+ §2.2.2（标量广播） |
| GEMM g1/g2 | `qkt = Q·Kᵀ`；`dsum = dO·Vᵀ` | `T.gemm(q_s, k_s, qkt, b_transpose=True, initC=True, size=[real_n, real_m, dim])` | 非累加 ⇒ `initC=True`；`size` 承载尾块 fractal 裁剪 | Developer | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.3（dst 必须 fp32）+ §2.4；`b_transpose` 先例 `examples/deepseek_v32/sparse_mla_bwd.py` 第 215 行 |
| GEMM g3/g4 | `dv += PᵀdO`；`dk += dsᵀQ` | `T.gemm(cast, do_s, dv_acc, a_transpose=True, size=[real_m, dim, real_n])` | **累加** ⇒ `initC=False`（跨 `T.Pipelined` 迭代） | Developer | `a_transpose` 先例 `examples/deepseek_v32/sparse_mla_bwd.py` 第 258 行；**跨循环累加 + UB 载体的已验证形态**：同文件 `acc_dq`（循环外 `T.clear` + 循环内 `T.gemm(initC=False)`） |
| GEMM g5 | `dq = ds·K` | `T.gemm(cast, k_s, dq_blk, initC=True, size=[real_n, dim, real_m])` | 非累加 ⇒ `initC=True` | Developer | 同 §3.2 g1 |
| B3 概率 | `exp(x')` | `T.vexp(qkt, qkt)` | 单 op（O4） | Developer | `docs/Tilelang.language/数学操作/T.vexp.md`（src/dst 同 shape、fp32 ✓） |
| B1 缩放/偏置 | `x·sm_scale − lse` | `T.vmul(qkt, sm_scale, qkt)` + `T.vsub(qkt, lse_s, qkt)` | **行广播** `[bn,bm] ∘ [bn,1]` | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2「行广播 `[M,N]*[M,1]`」；`T.vsub` 同族（scalar/广播先例 `examples/engram/engram_bwd.py`） |
| B2 掩码（O1） | `dsum = min(−PEN·(dplane − d0), 0)`（**单一定义源 = §3.3**，`t ≡ dsum`） | `T.vsub(dplane, d0, dsum)` → `T.vmul(dsum, -PEN, dsum)` → `T.vmin(dsum, 0.0, dsum)` → `T.vadd(qkt, dsum, qkt)` | 4 个整 tile op；用 `dsum` 作**临时缓冲**（此时它尚未被 g2 写入） | Developer | `T.vmin` 的 src2 支持 scalar（`docs/Tilelang.language/比较操作/T.vmin.md` §2.1）；`T.vmin` + 标量形态先例见 `attention.md` PL-1.11（`pen = vmin(i−j, 0)·PEN`）；**不用 `T.vmax`**（其 §2.1 的 src1/src2 只标 `tensor`，而 `vmin` 明标「tensor or scalar」） |
| B4/B6 dtype 回落 | `rint_dtype(·)` | `T.vcast(qkt, cast, round_mode="rint")` | `[bn,bm]` 同 shape；P 与 ds **串行复用同一 scratch**（O3） | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f32→f16` rint ✓、`f32→bf16` rint ✓）+ §2.2.2（shape 须一致）；规避 `TRAP-vcast-same-dtype-rounds`（本设计两处都是真 dtype 变化）与 `TRAP-C12-copy-dtype-cast` |
| B5 ds（O2） | `P·(dsum−δ')·sm_scale` | `T.vsub(dsum, dl_s, dsum)` → `T.vmul(qkt, dsum, qkt)` → `T.vmul(qkt, sm_scale, qkt)` | 原位：ds 覆盖 `qkt`（P 已复制到 scratch） | Developer | `T.vmul` §2.3 第 3 条（dst 可与 src 同 buffer）；O2 已机器验证逐位恒等 |
| B7 dq 累加 | `dq += ds·K` | `for r in T.serial(real_n): T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r, 0:dim], [1, dim])` | **逐行 1-D 行原子**（逐行分解跨步 BSHD 目标；动态行数由 `T.serial(real_n)` 界承担） | Developer | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.2（「对形状无特殊限制」）+ §2.4；生产先例 `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` 第 257 行 `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])`；**正确性由本任务探针三正面实测**（目标区全中 / 区外 0 / 20 并发精确 20.0 / 动态行数正确）；fp32 ✓（§2.2.1）。**rank-reduced 4-D 区域形态实测不可用**（§0.6 R3） |
| B8 dv/dk 写回 | `dv/dk ← / += acc` | `groups==1`：`T.copy(dv_acc[0:real_m,0:dim], dv[bz, r0:r0+real_m, hkv, 0:dim])`；否则 `for r in T.serial(real_m): T.atomic_add(dv[bz, r0+r, hkv, 0:dim], dv_acc[r, 0:dim], [1, dim])`（`dk` 同形） | **工厂期 trace-time 分派**（`groups` 是编译期常量）；`groups>1` 走**逐行 1-D 行原子**（跨步目标，同 B7） | Developer | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.2/§2.4；分派形态先例 `attention.md` PL-1.9-blockwidth「工厂级多 `@T.prim_func` 变体分派」；`T.copy` 的 rank-reduced 4-D 区域形态**可用**（`T.copy` 语义不同，探针二 ①② 支撑，§0.6 R6） |
| 尾块界 | `real_m`/`real_n` | `T.min(bm, seq_len - r0)` / `T.min(bn, seq_len - c0)` | 标量（每任务/每迭代 1 次） | Developer | `docs/Tilelang.language/比较操作/T.min.md` |
| 编译配置 | — | 无 `pass_configs`、无 `compile_flags`、无 `threads` | — | Developer | 任务约束（`target="npuir"` 不放编译开关）；`TRAP-threads-kwarg-noop` |

### 3.3 计算伪代码

> **接口保持**：外层工厂签名与源**逐字一致**（`(batch, heads, heads_kv, seq_len, dim, is_causal, dtype="float16")`，`functools.lru_cache` 保留）；
> 内层 builder 签名 `(block_m, block_n, num_stages)`（K9 删除 `threads`），与冻结 wrapper 的 `default_config` 键**完全对齐**
> （**不做任何 trace-time 配置替换**——主选档就是 wrapper 默认档，见 §1.6.3）。
> `_gqa_bwd_main(q, k, v, do, lse, delta, dq, dk, dv)` 的参数名/顺序与源一致；**无 `out_idx`**。
> `num_kernels` / `num_local_tasks` 为 host 编译期常量（trace 期折叠），不进入运行时。

```python
import functools
import os

import tilelang
import tilelang.language as T
from tilelang.utils.npu_utils import NPUUtils

ASCEND_MODE = "Developer"
os.environ.setdefault("TILELANG_ASCEND_MODE", ASCEND_MODE)

LOG2E = 1.44269504          # 与源一致（`tileops/kernels/online_softmax.py` 的 LOG2E 常量语义）
LN2 = 0.6931471805599453
_PEN = 1e30                 # O1：有限大数惩罚（避免 t*PEN 上溢为 -inf；仍给出精确 +0.0）
_SUPPORTED_DTYPES = ("float16", "bfloat16")


@functools.lru_cache(maxsize=32)
def _gqa_bwd_kernel(batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                    is_causal: bool, dtype: str = "float16"):
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim < 32 or dim % 16 != 0:
        raise ValueError("dim must be a multiple of 16 and >= 32 (fractal K>=32)")
    if min(batch, heads, heads_kv, seq_len, dim) < 1:
        raise ValueError("all shape dims must be positive")

    groups = heads // heads_kv
    sm_scale = dim ** -0.5

    def _gqa_bwd_func(block_m: int, block_n: int, num_stages: int):
        # ---- 编译期下界检查（fractal M/N>=16；**K>=32**：g3/g4 的 K 维 = bn、g5 的 K 维 = bm ⇒ 两者均须 >= 32；
        #      见 `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.3）----
        if block_m < 32 or block_n < 32:
            raise ValueError("block_m/block_n must be >= 32 (Cube fractal M/N>=16 and K>=32 for g3/g4/g5)")
        # ---- host 编译期常量（trace 期折叠为 IntImm）----
        n_kv = (seq_len + block_m - 1) // block_m
        n_q = (seq_len + block_n - 1) // block_n
        num_logical = batch * heads * n_kv
        cores = NPUUtils.get().get_aicore_num()          # 本机实查 20（§5.5）
        num_kernels = min(num_logical, cores)            # 混合/Cube 算子不翻倍
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels   # 静态界
        atomic_dk = groups > 1                           # trace-time 分派（§0.6 R3）
        causal = bool(is_causal)

        @tilelang.jit(target="npuir")
        def _builder(bm: int, bn: int, ns: int):
            @T.prim_func
            def _gqa_bwd_main(
                q: T.Tensor((batch, seq_len, heads, dim), dtype),          # type: ignore
                k: T.Tensor((batch, seq_len, heads_kv, dim), dtype),       # type: ignore
                v: T.Tensor((batch, seq_len, heads_kv, dim), dtype),       # type: ignore
                do: T.Tensor((batch, seq_len, heads, dim), dtype),         # type: ignore
                lse: T.Tensor((batch, heads, seq_len), "float32"),         # type: ignore
                delta: T.Tensor((batch, heads, seq_len), "float32"),       # type: ignore
                dq: T.Tensor((batch, seq_len, heads, dim), "float32"),     # type: ignore
                dk: T.Tensor((batch, seq_len, heads_kv, dim), "float32"),  # type: ignore
                dv: T.Tensor((batch, seq_len, heads_kv, dim), "float32"),  # type: ignore
            ) -> None:
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    k_s = T.alloc_shared((bm, dim), dtype)         # L1（仅作 gemm 操作数）
                    v_s = T.alloc_shared((bm, dim), dtype)         # L1
                    q_s = T.alloc_shared((bn, dim), dtype)         # L1
                    do_s = T.alloc_shared((bn, dim), dtype)        # L1
                    qkt = T.alloc_shared((bn, bm), "float32")      # UB：P → 原位 ds（O2）
                    dsum = T.alloc_shared((bn, bm), "float32")     # UB：dO·Vᵀ；兼作掩码临时缓冲
                    cast = T.alloc_shared((bn, bm), dtype)         # UB：dtype scratch（O3 串行复用）
                    dv_acc = T.alloc_shared((bm, dim), "float32")  # UB：跨 kq 累加
                    dk_acc = T.alloc_shared((bm, dim), "float32")  # UB：跨 kq 累加
                    dq_blk = T.alloc_shared((bn, dim), "float32")  # UB：每迭代
                    lse_s = T.alloc_shared((bn, 1), "float32")     # UB 微缓冲
                    dl_s = T.alloc_shared((bn, 1), "float32")      # UB 微缓冲
                    dplane = T.alloc_shared((bn, bm), "float32")   # UB：掩码平面（causal 专用）

                    for it in T.serial(num_local_tasks):           # 静态界（分核标准要素③）
                        uid = it * num_kernels + cid
                        if uid < num_logical:
                            bz = uid // (heads * n_kv)
                            r1 = uid % (heads * n_kv)
                            bx = r1 // n_kv                        # query head
                            by = r1 % n_kv                         # KV 块
                            hkv = bx // groups                     # KV head
                            r0 = by * bm
                            real_m = T.min(bm, seq_len - r0)

                            T.copy(k[bz, r0:r0 + real_m, hkv, 0:dim], k_s[0:real_m, 0:dim])
                            T.copy(v[bz, r0:r0 + real_m, hkv, 0:dim], v_s[0:real_m, 0:dim])
                            T.clear(dv_acc)
                            T.clear(dk_acc)
                            if causal:
                                for i0, j0 in T.Parallel(bn, bm):
                                    dplane[i0, j0] = T.Cast("float32", j0) - T.Cast("float32", i0)

                            for kq in T.Pipelined((by * bm) // bn if causal else 0,
                                                  n_q, num_stages=ns):
                                c0 = kq * bn
                                real_n = T.min(bn, seq_len - c0)

                                T.copy(q[bz, c0:c0 + real_n, bx, 0:dim], q_s[0:real_n, 0:dim])
                                T.copy(lse[bz, bx, c0:c0 + real_n], lse_s[0:real_n, 0])
                                T.vmul(lse_s, LN2, lse_s)                       # 自然指数域行偏置

                                T.gemm(q_s, k_s, qkt, b_transpose=True, initC=True,
                                       size=[real_n, real_m, dim])              # g1: Q·Kᵀ
                                T.vmul(qkt, sm_scale, qkt)                      # B1
                                T.vsub(qkt, lse_s, qkt)                         # B1（行广播）
                                if causal:                                      # B2: O1 掩码
                                    d0 = c0 - by * bm
                                    T.vsub(dplane, d0, dsum)
                                    T.vmul(dsum, -_PEN, dsum)
                                    T.vmin(dsum, 0.0, dsum)
                                    T.vadd(qkt, dsum, qkt)
                                T.vexp(qkt, qkt)                                # B3: O4 概率
                                T.vcast(qkt, cast, round_mode="rint")            # B4: P → dtype

                                T.copy(do[bz, c0:c0 + real_n, bx, 0:dim], do_s[0:real_n, 0:dim])
                                T.gemm(do_s, v_s, dsum, b_transpose=True, initC=True,
                                       size=[real_n, real_m, dim])              # g2: dO·Vᵀ
                                T.gemm(cast, do_s, dv_acc, a_transpose=True,
                                       size=[real_m, dim, real_n])              # g3: dv += PᵀdO

                                T.copy(delta[bz, bx, c0:c0 + real_n], dl_s[0:real_n, 0])
                                T.vmul(dl_s, sm_scale, dl_s)
                                T.vsub(dsum, dl_s, dsum)                        # B5
                                T.vmul(qkt, dsum, qkt)                          # B5（原位 O2）
                                T.vmul(qkt, sm_scale, qkt)                      # B5
                                T.vcast(qkt, cast, round_mode="rint")            # B6（O3 复用）

                                T.gemm(cast, q_s, dk_acc, a_transpose=True,
                                       size=[real_m, dim, real_n])              # g4: dk += dsᵀQ
                                T.gemm(cast, k_s, dq_blk, initC=True,
                                       size=[real_n, dim, real_m])              # g5: dq = ds·K
                                # B7: dq 跨 KV 块逐行 1-D 行原子（跨步 BSHD 目标的唯一可用形态；
                                #     行数 real_n 为 runtime 标量，T.serial 接受）
                                for r in T.serial(real_n):
                                    T.atomic_add(dq[bz, c0 + r, bx, 0:dim],
                                                 dq_blk[r, 0:dim], [1, dim])

                            if atomic_dk:                                       # B8: groups>1 ⇒ 逐行 1-D 行原子
                                for r in T.serial(real_m):
                                    T.atomic_add(dv[bz, r0 + r, hkv, 0:dim],
                                                 dv_acc[r, 0:dim], [1, dim])
                                    T.atomic_add(dk[bz, r0 + r, hkv, 0:dim],
                                                 dk_acc[r, 0:dim], [1, dim])
                            else:                                               # groups==1 ⇒ 普通区域存储（单写者）
                                T.copy(dv_acc[0:real_m, 0:dim],
                                       dv[bz, r0:r0 + real_m, hkv, 0:dim])
                                T.copy(dk_acc[0:real_m, 0:dim],
                                       dk[bz, r0:r0 + real_m, hkv, 0:dim])

            return _gqa_bwd_main

        return _builder(block_m, block_n, num_stages)

    return _gqa_bwd_func
```

> **尾块说明**：装载用 `real_m`/`real_n` 显式裁剪（垃圾行/列不入有效区）；GEMM 用 `size=` 的 fractal 下界裁剪
> （`mmadL1` 按 `size` 取真实范围）；逐行行原子的行数用同一 `real_n`/`real_m`（**动态行数，探针三已实测无越界**）、
> `T.copy` 的写出区域用同一 `real_*` 裁剪 ⇒ 与源的输出元素集合一致（§0.6 R6）。
> **缓冲复用**：`qkt`（P → ds，O2）、`cast`（P 的 dtype 载体 → ds 的 dtype 载体，O3）、
> `dsum`（dO·Vᵀ → 掩码临时缓冲 → ds 的中间载体，时序不重叠），与 §4.3 表一致。
> **K 维下界**：g3/g4 的 K 维 = `bn`、g5 的 K 维 = `bm` ⇒ 工厂 guard 要求 `block_m, block_n ≥ 32`（§4.5 阶梯同步收窄）。
> **若编译报 `ub overflow`**：按 §8.3 的降级阶梯把 `block_n` 降到 **32**（dim=128 档，已是下界）或 `block_m` 降到 32，功能不变。
> **若 `T.vmin` 的标量操作数形态在前端被拒**（`T.vbrc` 族有裸标量 rank 断言前科，见 `TRAP-npuir-vbrc-region-dropped` §「标量 operand 形态」）：
> 改用 typed 标量 `T.float32(0.0)` 与 `T.float32(-_PEN)`；若仍被拒，退回 `vcmp`+`vselect` 精确形态（§1.6.4 末段）。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `@tilelang.jit(target="npuir")` + `functools.lru_cache` 外层 | `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/_flashattn_bwd_preprocess_kernel.py`、`examples/deepseek_v32/sparse_mla_bwd.py` | ✅ 佐证（同批与同族已跑通） | 本设计**无** `compile_flags`/`pass_configs`/`out_idx` |
| `T.Kernel(一维, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1 | ✅ 已核对 | `is_npu` 最多 2 维；本设计一维 + `(cid, _)` |
| `T.serial` + `if` 守卫（静态界。persistent） | `PL-1.21-vector-persistent-split`；`examples/deepseek_v32/sparse_mla_bwd.py`（`cid` 分解） | ✅ 佐证 | `if` 体内只写预分配 buffer |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md` §2.1/§2.2.2/§2.4 | ⚠ 已核对（含限制） | Developer 映射 UB/L1；§2.2.2 标 3D~5D「有待充分验证」⇒ 本设计**全部用 2D 声明**（规避该风险，见 §4.3） |
| `T.copy`（双显式 slice + 尾块动态 extent） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2/§2.4 | ✅ 已核对 | 规则 1（静态 1 维折叠）、规则 2/4（尾部借用 extents）；切片与 `size` 互斥 |
| `T.gemm`（`b_transpose` / `a_transpose` / `initC` / `size`） | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1–§2.4；`examples/deepseek_v32/sparse_mla_bwd.py` 第 215/258 行 | ✅ 已核对 + 佐证 | fp16/bf16/fp32 ✓（§2.2.1）；dst 必须 fp32（§2.3）；`bf16×bf16→fp32` 建议 fp32（实测可用，PL-1.8）；`size=[a,b,c]` 语义 ✓ |
| `T.vmul` / `T.vsub` / `T.vadd` | `docs/Tilelang.language/数学操作/T.vmul.md` §2.1–§2.3、`T.vsub.md`、`T.vadd.md` | ✅ 已核对 | 行广播 `[M,N]*[M,1]` ✓；标量 src2 ✓；原位 ✓；操作数须在 UB ✓；fp16/fp32 ✓、**bf16 ×**（本设计的 v 链全在 fp32 域，无 bf16 算术 ⇒ 免疫） |
| `T.vmin`（scalar src2） | `docs/Tilelang.language/比较操作/T.vmin.md` §2.1（「tensor or scalar」） | ✅ 已核对 | 仅用于掩码惩罚的 `min(t,0)` |
| `T.vexp` | `docs/Tilelang.language/数学操作/T.vexp.md` | ✅ 已核对 | fp32 ✓；单 op（对比 `T.vexp2` 的 2 op，§1.6.1 O4/C5） |
| `T.vcast`（`f32→f16/bf16`，rint） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1/§2.2.2 | ✅ 已核对 | `f32→f16` rint ✓、`f32→bf16` rint ✓；shape 须一致；src 须为 UB |
| `T.atomic_add`（**逐行 1-D 行原子**，跨步 BSHD 目标） | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.1/§2.2.2/§2.4；`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` 第 257 行（同形态生产先例）；**本任务探针一/二/三** | ✅ 已核对 + **实测** | fp32 ✓（§2.2.1）；行原子形态 `T.atomic_add(dst[bz, row, h, 0:dim], src_row, [1, dim])` 由**探针三**正面实测（目标区全中 / 区外 0 / 20 并发精确 / 动态行数正确）；连续区域形态单价 0.97× store（探针一）。**rank-reduced 4-D 二维区域形态实测不可用**（§0.6 R3）⇒ 本设计只用逐行形态 |
| `T.atomic_addx4` | `docs/Tilelang.language/原子操作/T.atomic_addx4.md` §1 | ✅ 已核对（**不采用**） | 文档明示「当前实现与 `T.atomic_add` 完全等价（生成相同的 `tl.npuir_atomic_add` TIR），x4 专用硬件语义暂未接入」⇒ 用 `T.atomic_add` 即可，无额外收益 |
| `T.min`（标量） | `docs/Tilelang.language/比较操作/T.min.md` | ✅ 已核对 | 仅用于 `real_m`/`real_n` |
| `T.Cast` / `T.Parallel`（掩码平面物化） | `docs/Tilelang.language/编译器提示操作/T.Parallel.md`；`TRAP-parallel2d-bm-ge2-scalarize` 的适用域收窄记录 | ✅ 已核对（含限制） | 该 `T.Parallel` 体内是**单一算术表达式**（无条件选择），属「普通逐元素 2D loop」⇒ 不触发标量化；仅 causal 档执行 |
| `NPUUtils.get().get_aicore_num()` | `.agents/skills/_shared/standards/core-split-strategy.md` §1 要素② | ✅ 已实查 | 本机返回 **20**（§5.5 记录查询代码与返回值） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 是否涉及 | 处理方案 |
|------|---------|---------|
| 不支持三维 Kernel | **是** | 源为三维 grid ⇒ 一维 `T.Kernel(num_kernels, is_npu=True)` + 整除/取模任务分解（§0.6 R1） |
| GPU 专用 API（`threads=` / `T.Parallel` 元素循环 / `alloc_fragment` / wgmma / swizzle / `make_dq_layout`） | **是** | 本工厂未用 `threads`/wgmma/`load_nd2nz`；`T.Parallel` 元素循环 → v-prefix 整 tile 算子（§1.6.2）；`alloc_fragment` → `alloc_shared`（§0.5）；swizzle 与 `make_dq_layout` **舍弃**（§0.5）；`T.Parallel` 仅保留掩码平面物化（每任务一次，§1.6.2 #15） |
| GEMM 分形 / 非整除 | **是** | 无量化类约束；M/N ≥ 16、K ≥ 32 由工厂 guard 保证（`block_m, block_n ≥ 32`、`dim ≥ 32` 且 16 对齐——g3/g4 的 K 维 = `bn`、g5 的 K 维 = `bm`）；`S` 非整除由 `real_m`/`real_n` + `size=` 裁剪（§0.6 R6） |
| L0C 容量（128KB） | **是** | 单次 GEMM 的最大输出为 `dk_acc`/`dv_acc` 的 `[bm,dim]` fp32（`bm=64, dim=128` ⇒ **32 KB**）≤ 128 KB ✓；5 个 GEMM 的输出在 L0C 的驻留由编译器串行复用（源与同族先例同款） |
| L1 容量（512KB） | **是** | 任务级 K/V（`[bm,dim]` dtype ×2 = 32 KB@dim=128）＋ query 级 Q/dO（`[bn,dim]` ×2 = 16 KB@bn=32）＝ **48 KB**（主选）/ 64 KB（`bn=64` 备选档）≤ 512 KB ✓（§4.5） |
| 物理核数适配 | **是** | §5.5 三要素（实查 20，persistent 核内串行，静态界） |
| bf16 无 Vector 算术 | **是**（bf16 档） | 本设计的整条向量链（缩放/偏置/掩码/exp/ds）**全部在 fp32 域**（`qkt`/`dsum` 为 fp32 载体），仅在**送入 GEMM 前**用 `T.vcast` 回落到 bf16 ⇒ 不依赖 bf16 的 v-op（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 的 bf16 列为 ×）⇒ 免疫 |
| UB 多缓冲膨胀 / 归约链单价 | **是** | §4.5 双口径逐 buffer 复算 + 工厂 guard + 降级阶梯（§8.3） |
| 三维 UB buffer 的文档状态 | **否** | 本设计**全部 2D 声明**（§4.3），不触及 `T.alloc_shared.md` §2.2.2「3D~5D 有待充分验证」的风险面 |
| host 侧不得改写输入张量内容 | **否** | host 侧只做工厂期常量推导与 `functools.lru_cache`；op 层的 `contiguous()` 是元数据视图（允许） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5 / §0.6）

| 差异项 | GPU 源 | 本项目（Ascend NPU） | 转换方案 |
|--------|--------|---------------------|----------|
| Kernel 维度 | `T.Kernel(heads, ⌈S/bm⌉, batch)`（三维） | 一维，最多 2 维 block | 一维 + persistent 任务分解（§0.6 R1） |
| `threads` | `T.Kernel(..., threads=128)` | 无该概念 | K9 已从 wrapper config 删除；本设计不传（`TRAP-threads-kwarg-noop`） |
| `T.alloc_fragment` 计算域 | CUDA 寄存器 / fragment | v 算子要求 UB 操作数 | 全部 `T.alloc_shared`（Compiler 按用途分派 UB/L1） |
| GEMM 转置参数名 | `transpose_A` / `transpose_B` | `a_transpose` / `b_transpose` | 改名（`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1） |
| GEMM 尾块 | 无 `size`，靠 fragment 尺寸 | `size=[a,b,c]` 显式声明真实范围 | 传 `size=[real_n, real_m, dim]` 等（fractal 下界裁剪） |
| P tile 轴向 | `qkt` = `(block_m, block_n)` = `(kv, query)`；`lse` 是列因子 | 行广播文档化、列因子需 `vbrc` | 改为 `(query, kv)` 声明（§1.6.3 候选 2） |
| 因果掩码 | `if_then_else` 逐元素（`T.Parallel`） | 该形态标量化 | 指数域惩罚 + 平面不变量提升（§0.6 R4 / §1.6.1 O1） |
| `exp2` | `T.exp2`（硬件 exp2） | `T.vexp2` 实为 `exp(x·ln2)`（2 op） | `T.vexp` + 基域重标定（§1.6.1 O4） |
| dtype 回落 | `T.copy(qkt, qkt_cast)`（fragment cast） | 跨 dtype `T.copy` 舍入模式不可控 | 显式 `T.vcast(..., round_mode="rint")`（§1.6.1 C3） |
| 原子累加 | 逐元素 `T.atomic_add` + `make_dq_layout` | 跨步 BSHD 目标的二维区域原子**不可用**（编译失败 / 静默写错） | **逐行 1-D 行原子** `T.atomic_add(dst_row, src_row, [1, dim])`（§0.6 R3，探针三实测正确） |
| `dv/dk` 写回 | 恒用 `T.atomic_add` | MHA（`groups==1`）时每元素单写者 | 工厂期分派：`groups==1` → `T.copy` 区域存储；否则 → **逐行 1-D 行原子** |
| swizzle | `make_swizzled_layout(dv_shared/dk_shared)` | NPU 无 bank conflict | 舍弃（cube skill：Developer 保持 ND） |
| `pass_configs` / `compile_flags` | `TL_ENABLE_FAST_MATH` / `-O3 -DENABLE_BF16` | npuir 无 `compile_flags` 形参 | 均不使用 |
| `supported_archs=[80,89,90]` | CUDA SM 列表 | 无意义 | wrapper 已按 K5 置 `None`（本 kernel 不改 wrapper） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/deepseek_v32/sparse_mla_bwd.py` | **极高（同数学语义 + 同编程模式的 npuir 实现）** | `bwd()`：`@tl.jit(target="npuir")` + `T.Kernel(S*B*(vg*NH), is_npu=True)`（**无 `T.Scope`、Developer**）+ `T.alloc_shared` 承载 gemm 操作数与累加器 + `T.gemm(..., b_transpose=True/a_transpose=True, initC=...)` + `T.vexp2(src,dst,tmp)` + `T.reshape` + `T.atomic_addx4(区域, src, [1,4])` + `acc_dq` 跨 `T.Pipelined` 循环累加。**证实本设计的原语组合在 `target="npuir"` 上可编译可跑** |
| `examples/deepseek_v4/example_sparse_attn_bwd_kernel.py` | 极高（同语义，Developer） | 同款 attention-bwd 形态（`T.gemm` ×5 + `T.serial` + `T.atomic_add` 行散写）；**本设计的最小可用骨架** |
| `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` | 高（同族，Developer） | `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])`（动态下标的 1-D 行原子）+ `T.vadd`/`T.vmul`/`T.vexp2` 链 + `T.serial(split_store)` 分片降 UB 压力 |
| `examples/elementwise/atomic_add.py` / `atomic_add_dev.py` | 高（**二维区域原子**） | Developer 形态 `T.atomic_add(B[bx, by], A_shared, [tile_M, tile_N])` 的官方示例 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 极高（**同包 fwd：lse 契约 + 闭包工厂 + 本包分核先例**） | ① lse = **log2 域**（golden `logsumexp/log(2)`）；② 工厂返回 `wrapped(...)` 闭包；③ `T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub` + `T.gemm(..., size=[tmc, tnc, dim])` + `sync_block_set/wait`（Expert 形态，本设计不采用但其 `size=` 用法同源）；④ `NUM_KERNELS = 24` 为 910B2C 实查值（本机 910B3 = 20） |
| `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/_flashattn_bwd_preprocess_kernel.py` | 高（**同批前作的可运行实现**） | 一维 persistent 分核 + `T.min` 尾块 + 1-D 连续段写出 + 工厂 UB guard；其 `RETROSPECTIVE.md` 的「Transferable Lessons」已逐条吸收（§4.5 / §3.3 / §8.3） |
| `examples/TileOPs/tests/ops/test_multi_head_attention.py` | 高（**Stage 5 验收口径**） | `MhaBwdFixture` 四档 shape（§5.5/§8.3 采用）、op 级容差 `atol=5e-3, rtol=1e-5`、roofline 断言公式 |
| `/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py` | —（**源算子**） | 第 514 至 619 行（工厂 + kernel）、第 622 至 663 行（Kernel 类） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `q` | `(batch, seq_len, heads, dim)` | float16 / bfloat16 | BSHD；`dim` 最内连续（stride 1），seq 轴 stride = `heads·dim` |
| `k` / `v` | `(batch, seq_len, heads_kv, dim)` | 同上 | seq 轴 stride = `heads_kv·dim` |
| `do` | `(batch, seq_len, heads, dim)` | 同上 | 同 `q` |
| `lse` | `(batch, heads, seq_len)` | **float32** | **base-2（log2）域**；seq 轴最内连续，head 轴 stride = `seq_len`；由同包 fwd 产出 |
| `delta` | `(batch, heads, seq_len)` | **float32** | `rowsum(o∘do)`；由已迁移的 `_flashattn_bwd_preprocess_kernel` 产出 |
| `dq` / `dk` / `dv` | 见 §4.2 | float32 | **入参**（op 层预置 `torch.zeros`），本 kernel 累加 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dq` | `(batch, seq_len, heads, dim)` | **float32** | `dq` 由 `_flashattn_bwd_postprocess_kernel` 转回存储 dtype（**本 kernel 不转**） |
| `dk` | `(batch, seq_len, heads_kv, dim)` | **float32** | wrapper 侧 `.to(self.dtype)` |
| `dv` | `(batch, seq_len, heads_kv, dim)` | **float32** | 同上 |

输出 shape/dtype 与 `source_output_shape` 逐项一致（`dq:(B,S,H,D) fp32, dk/dv:(B,S,H_kv,D) fp32`）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 作用（角色复用时序） | 声明字节/元素 |
|-----------|-------|-------|----------|---------------------|--------------|
| `k_s` | `(bm, dim)` | 输入 dtype | **L1**（仅作 gemm 操作数） | 任务级 K 块 | 2 |
| `v_s` | `(bm, dim)` | 输入 dtype | **L1** | 任务级 V 块 | 2 |
| `q_s` | `(bn, dim)` | 输入 dtype | **L1** | query 块 Q（每迭代） | 2 |
| `do_s` | `(bn, dim)` | 输入 dtype | **L1** | query 块 dO（每迭代） | 2 |
| `qkt` | `(bn, bm)` | float32 | UB | ① `Q·Kᵀ` → P（B1/B3 原位）→ ② **ds（O2 原位覆盖）** | 4 |
| `dsum` | `(bn, bm)` | float32 | UB | ① **掩码临时缓冲**（causal，在 g2 之前）→ ② `dO·Vᵀ` → ③ ds 的中间载体（B5 原位） | 4 |
| `cast` | `(bn, bm)` | 输入 dtype | UB | ① P 的 dtype 载体 → ② ds 的 dtype 载体（**O3 串行复用**） | 2 |
| `dv_acc` | `(bm, dim)` | float32 | UB | `dv` 的跨 kq 累加器（任务末写回） | 4 |
| `dk_acc` | `(bm, dim)` | float32 | UB | `dk` 的跨 kq 累加器 | 4 |
| `dq_blk` | `(bn, dim)` | float32 | UB | 每迭代的 `dq` 块（g5 输出 → **逐行 1-D 行原子**） | 4 |
| `lse_s` | `(bn, 1)` | float32 | UB | `lse` 行偏置（装载后原位乘 `ln2`） | ≈0 |
| `dl_s` | `(bn, 1)` | float32 | UB | `delta` 行偏置（装载后原位乘 `sm_scale`） | ≈0 |
| `dplane` | `(bn, bm)` | float32 | UB | 掩码平面 `j − i`（**每任务物化一次**；causal 专用） | 4 |

**驻留元素数口径**：UB 侧声明量
`= (2·bn·bm + (bm + bm + bn)·dim) · 4 B + bn·bm·2 B`（主选 `bm=64, dim=128, bn=32` ⇒ **100 KB**；
`bm=64, dim=64, bn=64` ⇒ **88 KB**；causal 追加 `bn·bm·4 B` = 8 KB / 16 KB）。
**全部 2D 声明**（规避 `T.alloc_shared.md` §2.2.2 的 3D+ 未充分验证面）。
**无**乘积专用缓冲（O2 已消除）、**无**第二块 dtype 缓冲（O3 已消除）。

### 4.4 内存搬运路径

```
GM[q/k/v/do (BSHD, 跨步行块)] --T.copy(双显式 slice, dst 零起点)--> L1[k_s/v_s/q_s/do_s]
GM[lse/delta (B,H,S) 连续段]   --T.copy(1-D)---------------------> UB[lse_s/dl_s (bn,1) fp32]

Cube 段（L1 → L0A/L0B → L0C）：
  L1[q_s] × L1[k_s]ᵀ  --T.gemm(b_transpose)--> UB[qkt (bn,bm) fp32]
  L1[do_s] × L1[v_s]ᵀ --T.gemm(b_transpose)--> UB[dsum (bn,bm) fp32]
  UB[cast] × L1[do_s] --T.gemm(a_transpose)--> UB[dv_acc (bm,dim) fp32]   跨 kq 累加
  UB[cast] × L1[q_s]  --T.gemm(a_transpose)--> UB[dk_acc (bm,dim) fp32]   跨 kq 累加
  UB[cast] × L1[k_s]  --T.gemm--------------> UB[dq_blk (bn,dim) fp32]

Vector 段（全部在 UB 内闭环）：
  qkt --vmul(sm_scale)--> vsub(lse_s: 行广播)--> [causal: vsub/vmul/vmin/vadd]--> vexp --> P (fp32)
  P --vcast(rint)--> cast (dtype)                            （供 g3）
  dsum --vsub(dl_s)·--vmul(qkt)·--vmul(sm_scale)--> ds (fp32, **原位覆盖 qkt**)   （O2）
  ds --vcast(rint)--> cast（同一 scratch）                    （供 g4/g5）（O3）

UB --> GM（结果）：
  dq_blk --逐行 1-D 行原子 [1,dim] × real_n 行（跨步 BSHD 目标）--> GM[dq]   （每 kq 一次）
  dv_acc/dk_acc --T.copy(groups==1 普通区域存储) / 逐行 1-D 行原子(groups>1)--> GM[dv/dk]
```

**关键约束**：v 算子操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3 第 1 条）；
GEMM 操作数经 L1、累加器落 L0C（单次最大 32 KB ≤ L0C 128 KB）；
**无中间结果落回 GM**（除三个输出）、**无 workspace**、**无跨引擎 S/P 往返**（⇒ §1.6.0 roofline 的容量项 = 0）。

### 4.5 UB 内存预算

**预算口径（双口径并列，遵循 `hardware-cost-model.md` §5 与 `CONST-capacity-910B2C`）**：

1. **口径 A（声明量 + 显式-alloc 结构的小膨胀）**：`CONST-capacity-910B2C` 的 2026-09-15 重校与 2026-09-16 r9 反例
   实测「逐 buffer 显式 alloc 结构」的真实需求 ≈ 手工核算 × **1.10–1.12**（r9 结构实测 **0.981**，低于手工）
   ⇒ 主选 `100 KB × 1.12 = 112 KB ≤ 192 KB` ✓。
2. **口径 B（auto multi-buffer 默认档的保守膨胀）**：同条目记默认档膨胀 **~1.7x**
   ⇒ 主选 `100 KB × 1.7 = 170 KB ≤ 192 KB` ✓（**主选在双口径下均可行**）。
3. **取双口径的交集作为默认档；上探档只依赖口径 A**（§1.6.3 参数级 A/B）。

**逐 workload 复算（主选档 = wrapper `default_config` 原样）**：

| workload（manifest） | (B,S,H,D) | dtype | `(bm,bn,ns)` | `⌈S/bm⌉`（`num_logical`） | UB 声明量 | 口径 A（×1.12） | 口径 B（×1.7） | L1 声明量 | ≤192 KB / 512 KB |
|----------------------|-----------|-------|--------------|--------------------------|-----------|-----------------|----------------|-----------|------------------|
| `smoke-bwd-fp16` | (1,1024,8,64) | fp16 | (64,64,1) | 16（128） | **88 KB** | 99 KB | 150 KB | 32 KB | ✓ / ✓ |
| `smoke-bwd-bf16` | (1,1024,8,64) | bf16 | (64,64,1) | 16（128） | 88 KB | 99 KB | 150 KB | 32 KB | ✓ / ✓ |
| `full-bwd-fp16-large` | (16,2048,16,128) | fp16 | (64,32,1) | 32（8192） | **100 KB** | 112 KB | 170 KB | 48 KB | ✓ / ✓ |
| `full-bwd-bf16` | (4,4096,16,128) | bf16 | (64,32,1) | 64（4096） | 100 KB | 112 KB | 170 KB | 48 KB | ✓ / ✓ |
| causal 档（追加 `dplane`） | 任一 | 任一 | 同上 | 同上 | +8 / +16 KB | 121 / 130 KB | 184 / 197 KB | 同上 | ⚠ dim=128+bn=64 档在口径 B 下 197 KB 超限 ⇒ 该组合由工厂 guard 降档（§8.3） |
| 上探档（§1.6.3 备选） | full 档 | 任一 | (64,64,1) | 32（8192） | **136 KB** | 152 KB | **231 KB ✗** | 64 KB | 仅口径 A 可行 |
| 域上界（超即拦截） | — | — | — | — | `(2·bm·bn + (2·bm+bn)·dim)·4 + bm·bn·2 (+causal`bm·bn·4`)` ×1.7 > 196 608 B **或** `min(bm,bn) < 32`（g3/g4/g5 的 K 维下界） | 工厂 guard 拒绝 | — | — | ⚠ 见 §8.3 |

**降级阶梯（工厂 guard + Stage 4 反解；revision 1 已把 `(64,16)` 兜底档删除——`bn=16` 使 g3/g4 的 K 维 = 16 < 32，违反全篇声明的 K ≥ 32 下界，且旧 guard 只校验 ≥16、会静默接受该档）**：

| 档 | `(bm,bn)` @dim=128 | UB 声明 | 可行条件 | 迭代数与 issued-op（full 档） |
|----|-------------------|---------|----------|------------------------|
| 上探档 | (64,64) | 136 KB | auto multi-buffer 实际膨胀 < 1.4（口径 A 成立） | 迭代 `262 144`、`N_ops/迭代 = 74` ⇒ 上界档发射项 ≈ **0.88×** 默认档；**地板档不变**（总元素工作量守恒） |
| **默认档** | **(64,32)** | **100 KB** | **口径 A 与口径 B 均成立（= wrapper 默认值）** | 迭代 `524 288`、`N_ops/迭代 = 42` |
| 兜底档 | **(32,32)** | 58 KB | 任何膨胀 ≤ 3.3 | 迭代 `1 048 576`（`bm` 减半 ⇒ 迭代数 ×2）、`N_ops/迭代 = 42` |

> **UB guard**：工厂层校验 `UB_declared(bm, bn, dim, causal) ≤ 196608 / 1.7`（口径 B 保守值）**且 `min(bm,bn) ≥ 32`**（K 维下界）；
> 编译期若报 `ub overflow`，按阶梯降档（无需改代码，只改传入的 `block_n`），功能与精度完全不变（§8.3）。
> **诚实标注**：口径 B 的 1.7x 取自 pattern-library 的默认档反推（大 block_size 探索场景），
> 本设计属「逐 buffer 显式 alloc 且无大残差缓冲」结构（口径 A 的 1.10–1.12 适用域），
> **两者的实际取值需 Stage 4 以 `ub overflow` 报文反解**；主选档在**双边最保守**下仍可行，故不构成阻塞。

### 4.6 动态轴定义

无运行时动态 shape：`batch`/`heads`/`heads_kv`/`seq_len`/`dim`/`is_causal`/`dtype` 均为工厂期常量（per-build JIT 特化，与源工厂语义一致）。
kernel 内唯一的运行时标量是尾块界 `real_m = T.min(bm, S − r0)`、`real_n = T.min(bn, S − c0)` 与掩码标量 `d0 = c0 − by·bm`。
`seq_len` 在**训练场景**下可能随 step 变化 ⇒ 由 `functools.lru_cache(maxsize=32)` 的 per-shape 特化承接（与源一致，缓存 32 档）。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=32)                 # 源同款：per-shape 特化缓存
def _gqa_bwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype="float16"):
    ...
    def _gqa_bwd_func(block_m, block_n, num_stages):
        ...
        @tilelang.jit(target="npuir")            # 无 compile_flags / pass_configs / out_idx
        def _builder(bm, bn, ns):
            @T.prim_func
            def _gqa_bwd_main(...): ...
            return _gqa_bwd_main
        return _builder(block_m, block_n, num_stages)
    return _gqa_bwd_func
```

- **无 `pass_configs`**（源工厂的 `TL_ENABLE_FAST_MATH` 在 npuir 无对应；本任务约束不放编译开关）。
- **无 `compile_flags`**（npuir 的 `tilelang.jit` 无形参）。
- **无 `out_idx`**（源契约：9 个参数全为入参）。
- **无 `threads`**（K9）。
- 运行时环境变量：`TILELANG_ASCEND_MODE=Developer`（文件顶层 `os.environ.setdefault` 写死）。
- msprof 过滤：device kernel 名为 `_gqa_bwd_main`；wrapper 的 `msprof_kernel_name` 若与自动推导不符可由集成侧显式声明（§10）。

---

## 5. Tiling 策略

### 5.1 Block 划分与 Tile Shape（总表）

| 形状 | 取值（主选） | 说明 |
|------|-------------|------|
| `block_m`（KV 轴行数） | **64** | 决定 `dv_acc`/`dk_acc`（`[bm,dim]` fp32）与 g5 的 K 维 |
| `block_n`（query 轴行数） | **64**（`dim ≤ 64`）/ **32**（`dim > 64`） | 决定 query 循环迭代数与逐行行原子的行数（`real_n`）；= wrapper `default_config` |
| `num_stages` | **1** | `T.Pipelined` 的 stage 数；Stage 4 可上探 2/3（UB 允许时） |
| 逻辑任务 | `(bz, bx, by)` = `(batch, query head, KV 块)` | `num_logical = B·H·⌈S/bm⌉` |
| 任务内循环 | query 块 `kq ∈ [loop_st, ⌈S/bn⌉)` | causal 时 `loop_st = ⌊by·bm/bn⌋`（块级跳过全掩码区） |
| GEMM 分块 | g1/g2：`M=bn, N=bm, K=dim`；g3/g4：`M=bm, N=dim, K=bn`；g5：`M=bn, N=dim, K=bm` | 全部满足 M/N ≥ 16、K ≥ 32（`dim≥32`、**`bm,bn≥32`——g3/g4 的 K=bn、g5 的 K=bm**） |

### 5.2 Tile 尺寸的量化依据

- **Tile 面积越大 ⇒ 发射项越低**：发射项 ∝ 迭代数 `= B·H·(S/bm)·(S/bn)`（§1.6.0 估算表）。`bm` 受 `dv_acc+dk_acc = 2·bm·dim·4` 约束（`dim=128` 时 `bm=64 ⇒ 64 KB`；`bm=128 ⇒ 128 KB` 已无空间给 P tile）⇒ `bm=64` 是 dim=128 档的实际上界。
- **`bn` 受 P 链缓冲与 dq_blk 约束**：`(2·bn·bm·4) + (bn·bm·2) + bn·dim·4`。`bn=32` 时为 `8+8+4+16=36 KB`（dim=128），`bn=64` 时为 `16+16+8+32=72 KB` ⇒ `bn=64` 需 UB 口径 A 才成立（§4.5）。
- **`dim` 不切分**：切分会使 P 链（与 `dim` 无关）按分块数重复（§1.6.3 候选 6）。
- **对齐**：`dim ∈ {64,128}` 均为 8 的整数倍（fp16/bf16 向量宽）且 16 对齐（Cube K 维）；`bm=64`、`bn∈{32,64}` 均为 32 的整数倍（同时满足 Cube M/N≥16、K≥32 与向量宽）；`dq_blk`/`dv_acc` 的尾轴 = `dim` 为连续段 ⇒ **逐行行原子/区域存储的行体尾轴 32B 对齐** ✓。

### 5.3 非整除场景处理策略

| 场景 | 策略 |
|------|------|
| `S % block_m ≠ 0`（KV 轴尾块） | `real_m = T.min(bm, S − by·bm)`；装载 `T.copy(k[bz, r0:r0+real_m, hkv, 0:dim], k_s[0:real_m, 0:dim])`（**双显式 slice，两侧同裁剪**）；GEMM 用 `size=[·, real_m, ·]`；写回用 `[0:real_m, 0:dim]` ⇒ 输出元素集合与源一致 |
| `S % block_n ≠ 0`（query 轴尾块） | `real_n = T.min(bn, S − c0)`；同款双显式 slice + `size=[real_n, ·, ·]`；`dq` 的逐行行原子行数用 `real_n`（探针三已验证动态行数无越界） |
| `heads % heads_kv ≠ 0` | 工厂期 `raise ValueError("heads must be divisible by heads_kv")`（与源一致，第 523 至 524 行） |
| `dim < 32` 或 `dim % 16 ≠ 0` | 工厂期拒绝（Cube K ≥ 32 且 16 对齐；`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` 亦断言 `dim % 16 == 0`） |
| `num_logical < num_kernels`（如 `B·H=1`、`S ≤ bm`） | `num_kernels = min(num_logical, cores)` ⇒ 空闲核不启动；`if uid < num_logical` 守卫兜底 |
| causal 且 `by·bm` 非 `bn` 整数倍 | `loop_st = T.floordiv(by*bm, bn)` 为运行时表达式（`T.Pipelined` 的 start 支持 runtime 界，源同款）；块内掩码由 `d0 = c0 − by·bm` 的惩罚式承担 |

> **零 block 风险**：`S < bm` 时 `n_kv = 1`、`real_m = S` ⇒ 每任务仍处理一次，无零 block。
> **尾块与 fractal 的冲突**：GEMM 的 M/N/K 由 `size=` 显式声明，`mmadL1` 按真实范围取数
> （同包 fwd 的 `size=[tmc, tnc, dim]` 与之同款）；L1 侧残留的 stale 行/列被 `size` 排除在 GEMM 之外 ⇒ 不污染结果。

### 5.4 Tiling 与内存 / 循环的耦合

- `T.Pipelined(loop_st, n_q, num_stages=ns)`：stage 数 × query 级缓冲（`q_s`/`do_s`/`lse_s`/`dl_s`）构成 UB/L1 的流水占用；
  主选 `ns=1`（与 wrapper 默认一致）⇒ 无额外多缓冲压力。
- 任务级（`k_s`/`v_s`）与 query 级（`q_s`/`do_s`）分离：任务级只装载一次（源 #1 的意图），query 级每迭代重载。
- `dv_acc`/`dk_acc` 是**任务级**累加器（跨 `kq`），其生命周期覆盖整个 query 循环 ⇒ 不能被流水缓冲复用。

### 5.5 分核策略（三要素）⭐

**① 逻辑核数计算**

$$
\texttt{num\_logical} = \text{num\_logical} = B \cdot H \cdot \lceil S/\texttt{block\_m}\rceil \quad \text{(one task per (batch, query head, KV block))}
$$

| workload | `num_logical` |
|----------|---------------|
| `smoke-bwd-fp16/bf16` (1,1024,8,64)，`bm=64` | `1·8·16 =` **128** |
| `full-bwd-fp16-large` (16,2048,16,128)，`bm=64` | `16·16·32 =` **8192** |
| `full-bwd-bf16` (4,4096,16,128)，`bm=64` | `4·16·64 =` **4096** |
| 测试用极端小档 (1,64,1,64) | `1·1·1 =` **1** |

**② 物理核数依据（实查，禁止假设值）**

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()          # 查询代码
# 实际返回值（2026-10-16，Ascend910B3 + CANN 9.0.0）：20
```
交叉核对：`from tilelang.utils.npu_arch import AscendArch; a = AscendArch()` → `a.aivector_core_num = 40`、`a.ub_cap = 196608`（互证 `20 × 2 = 40`）。
**本算子是混合（Cube + Vector）算子 ⇒ 直接使用返回值 20**（分核标准 §1 要素②：「Cube/混合算子直接使用返回值；纯 Vector 算子核数翻倍」）。

**③ 规模判定与分核方案**

判定：`num_logical ∈ {128, 4096, 8192} ≫ 20` ⇒ **极大规模**（无法通过调整分块把核数缩到物理核数的 1×/2×/3×——
`B·H` 最小为 8、`⌈S/bm⌉` 最小为 1，核数下界 `B·H ≥ 8`，manifest 档为 128–8192）。
⇒ 采用标准要素③的「固定启动内核数 = 物理核数 + 核内 `T.serial` 串行」方案：

```python
num_kernels      = min(num_logical, cores)                       # = 20
num_local_tasks  = (num_logical + num_kernels - 1) // num_kernels # 静态界（host 折叠）
for it in T.serial(num_local_tasks):        # ← 循环边界为静态值
    uid = it * num_kernels + cid
    if uid < num_logical:                   # 尾轮守卫
        ...
```

| workload | `num_kernels` | `num_local_tasks`（静态） | 说明 |
|----------|---------------|--------------------------|------|
| smoke (1,1024,8,64) | 20 | **7** | 128 逻辑任务 / 20 核 = 6.4 ⇒ 7 轮（尾轮仅 8 个有效核，`uid < num_logical` 守卫） |
| full-large (16,2048,16,128) | 20 | **410** | 8192 / 20 = 409.6 ⇒ 410 |
| full-bf16 (4,4096,16,128) | 20 | **205** | 4096 / 20 = 204.8 ⇒ 205 |
| 极端小档 (1,64,1,64) | **1** | 1 | `num_logical=1 < 20` ⇒ 只启动 1 个核（无超发） |

**为何不选「逻辑核超发」**：标准 §3 的失败信号表明示「逻辑核数远超物理核数致串行调度开销剧增」属 `[DESIGN_ERROR]`；
本算子逻辑核数达物理核数的 6.4–410 倍，若直接下发 `num_logical` 个内核，超出物理核数的部分会被**串行调度**并引入核启动开销。
**为何 persistent 在本编程模式下可行**：`TRAP-DEVMODE-PERSIST-GEMM`（Developer + persistent + gemm 运行期崩溃）
在**本工具链已被本任务设计期探针二 Q3 复验推翻**（`n_kernels=4/n_tasks=8` 与 `20/4` 两档编译通过、`max_abs_err = 4.578e-05`），
故无须为此改切 Expert（§1.6.0 探针表 / §2.2）。**兜底**：若 Stage 3 出现该 trap 的等价症状，按 §2.3 切 Expert 双 Scope（同结构），
分核三要素不变（仅 `T.Scope` 与同步显式化）。

**核数与 block 取值的自洽性**：`num_kernels = min(num_logical, 20)` 与 `num_local_tasks = ⌈num_logical/20⌉` 自洽；
`bm=64` 使 `n_kv = ⌈S/64⌉`，从而 `num_logical = B·H·n_kv` 与 §5.5① 表一致。

---

## 6. 循环与调度结构

| 层级 | 结构 | 选择依据 |
|------|------|---------|
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维，`num_kernels = min(num_logical, 20)`） | 分核标准 §1 要素③（极大规模：固定核数）；`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1（≤2 维） |
| 任务循环 | `for it in T.serial(num_local_tasks)` + `if uid < num_logical` | **静态界**（host 编译期常量折叠）；persistent grid-stride；`PL-1.21-vector-persistent-split` |
| query 块循环 | `for kq in T.Pipelined(loop_st, n_q, num_stages=ns)` | 有可重叠的搬运算（`q_s`/`do_s`/`lse_s`/`dl_s` 装载）与计算 ⇒ 交给编译器做多缓冲；stage 数按 UB 容量取（主选 `ns=1`） |
| 迭代内顺序 | 装载 → g1 → B1 → [B2] → B3 → B4 → 装载 do → g2 → g3 → 装载 delta → B5 → B6 → g4 → g5 → B7 | 与源逐语句同序（§0.3 步骤 S4–S13）；g3 必须在 B4 之后（用 P 的 dtype 载体），g4/g5 必须在 B6 之后（用 ds 的 dtype 载体） |
| **B7 行原子循环** | `for r in T.serial(real_n): T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r, 0:dim], [1, dim])` | 跨步 BSHD 目标的**结构性行步进**（行体为向量宽 1-D op）；行数 `real_n` 为 runtime 标量（§1.6.2 #9，探针三实测） |
| **不使用** `T.Parallel` 逐元素链 | 源 S6/S7/S10/S13 的 `T.Parallel` 元素循环全部以整 tile v-prefix 算子替代 | §1.6.2（逐元素条件选择与逐元素原子在 NPU 标量化） |
| **不使用** `T.Persistent` | 本工具链 `tilelang.language` 无该原语（`grep -rn "Persistent" docs/Tilelang.language/ tilelang/language/__init__.py` 无命中） | 以 `T.serial` + 静态界实现等价语义（映射表「persistent kernel → `T.Persistent`」在本工具链以 `T.serial` 落地） |
| 仅有的 `T.Parallel` | `for i0, j0 in T.Parallel(bn, bm): dplane[i0, j0] = j0 − i0`（**每任务一次**，causal 专用） | 体内是单一算术表达式（无 `T.if_then_else` 融合条件选择）⇒ 属 `TRAP-parallel2d-bm-ge2-scalarize` 适用域收窄记录中的「正常向量化」形态 |

**与 §1.6.2 的一致性**：§1.6.2 判定为「已向量化」的 8 个计算点（#3–#10）在本节结构中**均以整 tile v-prefix 算子出现**，
kernel 内**不存在**对应的逐元素标量循环；保留的 5 类标量点（#13–#17）在结构中分别对应 `real_m`/`real_n`、`uid` 分解、
`dplane` 物化、`d0`、host 常量。

**调度与流水的量化约束**：`num_local_tasks × (per-iteration work)` 是每核的串行工作量（§1.6.0 估算表的发射项即由此得出）；
stage 数上探会按 `ns × (query 级缓冲)` 增加 L1/UB 占用（`q_s`/`do_s` 在 L1，`lse_s`/`dl_s` 在 UB，合计增量很小）。

---

## 7. 同步策略与 CV 融合设计

**模式**：**Developer ⇒ 自动同步**（不出现任何手动同步原语：无 `T.sync_block_set`/`T.sync_block_wait`、无 `T.set_flag`/`T.wait_flag`、无 `T.pipe_barrier`、无 `T.rs("PIPE_*")`）。

依赖链与编译器职责（说明为何自动同步充分）：

| 依赖 | 生产者 → 消费者 | 同步性质 | 承接者 |
|------|----------------|----------|--------|
| 装载 → GEMM | MTE2（`T.copy` 写 L1）→ Cube（读 L1 到 L0A/L0B） | 跨引擎 RAW | auto CV split 插入的引擎间同步 |
| GEMM → Vector | Cube（写 L0C）→ Vector（读 UB 载体） | 跨引擎 RAW（含 L0C→UB 的数据搬运） | 同上 |
| Vector → GEMM | Vector（写 `cast` 的 dtype）→ Cube（读 `cast`） | 同引擎但跨 pipe（VEC→MTE1/MTE2）；**且 `cast` 在一个迭代内被写两次**（P 与 ds）⇒ 存在 RAW/WAR | 编译器按 buffer 依赖插入同步；两处写在同一依赖链上顺序执行 |
| `qkt` 的原位链（P → ds） | Vector 内部串行 | 同引擎、同 buffer 的 RAW | 程序序（v-算子按发射序，同 buffer 天然串行） |
| `dsum` 的角色切换（掩码临时 → g2 输出 → ds 中间） | Vector ↔ Cube 交替 | WAR（掩码写入后 g2 覆盖） | 同上；**causal 档**须保证掩码消费（`T.vadd`）先于 g2 的写入 ⇒ 程序序保证 |
| `dv_acc`/`dk_acc` 的跨迭代累加 | Cube 的 L0C 累加器 → UB 载体（每迭代读出再累加） | 循环携带依赖（RAW/WAR 跨 `T.Pipelined` 迭代） | 编译器的多缓冲/依赖分析；**已验证形态**见 `examples/deepseek_v32/sparse_mla_bwd.py` 的 `acc_dq` |
| `dq` 的跨任务原子 | Vector（**逐行 1-D 行原子**写 GM）→ 其他核 | **跨核**；由**硬件原子**保证 | 无软件同步需求（原子语义；探针三实测 20 并发同址精确） |
| `dq_blk` 的每迭代重写 | g5（写）→ 原子（读） | 同迭代 RAW | 程序序 |
| `num_local_tasks` 轮之间 | 任务 T 的尾（`dv/dk` 写回、`dq` 原子）→ 任务 T+1 的头部（装载覆盖 `k_s`/`v_s`） | 跨任务 WAR（L1 缓冲复用） | 程序序 + 编译器插入的引擎间同步（**注意**：`k_s`/`v_s` 被任务 T+1 覆盖前，任务 T 的 g3/g4 必须已读完 —— 但 `dv_acc`/`dk_acc` 的写回发生在 q 循环之后、任务 T+1 的装载之前，程序序天然闭合） |

**为何不需要 Expert 的手动同步**：本算子的跨引擎数据流**全部经由预分配的 UB/L1 缓冲**（不经 GM workspace），
不存在「跨引擎同 buffer 无同步原语」的流水分片问题（`attention.md` PL-1.12 的 r7f 反向教训适用于手工分片流水的 Vec 侧）；
唯一的跨核交互是 `dq` 的原子累加，由硬件承接。若 Stage 3 切换 Expert（§2.3），
则须显式给出：Cube 侧 `T.sync_block_wait/set` 的 flag 编号（≤15/核，`CONST-flag-id-budget`）与 Vector 侧的对应握手。

---

### 7.1 融合结构（引擎分工）

本算子是**迭代内紧耦合的 MixCV**（不是跨 kernel 融合）：每个 query 块迭代内
Cube 与 Vector **交替 5 次**（g1→v链→g2→g3→v链→g4→g5→原子），数据全部经 UB/L1 交接，**无 GM workspace 往返**。

| 段 | 引擎 | 计算 | 输入 / 输出 | 关键约束 |
|----|------|------|-------------|---------|
| 装载 | MTE2 | `T.copy` GM→L1（`k_s`/`v_s`/`q_s`/`do_s`）、GM→UB（`lse_s`/`dl_s`） | — | 跨步 slice 形态（`TRAP-load-nd2nz-strided` 的绕法）；双显式 slice |
| g1 | Cube | `qkt = Q·Kᵀ`（`b_transpose`） | L1[q_s] × L1[k_s] → L0C → UB[qkt] | M=bn≥16, N=bm≥16, K=dim≥32 |
| g2 | Cube | `dsum = dO·Vᵀ`（`b_transpose`） | L1[do_s] × L1[v_s] → L0C → UB[dsum] | 同上 |
| g3 | Cube | `dv_acc += Pᵀ·dO`（`a_transpose`，累加） | UB[cast] × L1[do_s] → L0C（累加） → UB[dv_acc] | 跨迭代累加（`initC=False`） |
| g4 | Cube | `dk_acc += dsᵀ·Q`（`a_transpose`，累加） | UB[cast] × L1[q_s] → L0C（累加） → UB[dk_acc] | 同上 |
| g5 | Cube | `dq_blk = ds·K` | UB[cast] × L1[k_s] → L0C → UB[dq_blk] | 每迭代重置（`initC=True`） |
| v 链 A（B1/B3/B4） | Vector | `qkt·sm_scale − lse·ln2` → [`pen` 惩罚] → `exp` → `rint_dtype` → `cast` | UB → UB | 行广播 `[bn,1]`；全程 fp32 |
| v 链 B（B5/B6） | Vector | `dsum − δ·sm_scale` → `·P` → `·sm_scale` → `rint_dtype` → `cast` | UB → UB | O2 原位；O3 scratch 复用 |
| 结果写出 | MTE3 / 原子 | `dq_blk` **逐行 1-D 行原子**（每迭代 `real_n` 行）、`dv_acc`/`dk_acc` 普通区域存储（`groups==1`）或逐行行原子（`groups>1`） | UB → GM | 跨步 BSHD 目标（行 stride = `H·D`；探针三实测正确） |

### 7.2 数据交接与容量

| 交接点 | 载体 | 容量 | 说明 |
|--------|------|------|------|
| Cube → Vector | `qkt` / `dsum` / `dv_acc` / `dk_acc` / `dq_blk`（UB，fp32） | 8/8/32/32/16 KB（dim=128, bn=32） | L0C → UB 的数据搬运由编译器插入；**无 GM 往返**（`CONST-store-fixpipe-gm-only` 只约束 `store_fixpipe` 直写 L1/UB 的缺失，本设计不经 fixpipe） |
| Vector → Cube | `cast`（UB，dtype） | 4 KB | 一个迭代内先承载 P、后承载 ds（O3） |
| 跨核 | `dq`（GM，fp32） | — | 硬件原子（探针三：**逐行 1-D 行原子**，目标区全中/区外 0/20 并发精确；连续区域形态单价 0.97× store，探针一） |
| 引擎内工作集 | Cube：L1 48–64 KB / L0C ≤32 KB 单次；Vector：UB 100 KB（dim=128 主选） | 见 §4.5 | 均为「单块在片」无流水槽需求 |

### 7.3 同步点（Developer 自动）

见 §7 表。**本设计的融合是「迭代内交替」而非「跨引擎块级流水」**，故不涉及 per-slot flag、
不需要 GM workspace 多槽、不需要 `TASKDONE` 屏障 ⇒ 规避了 `attention.md` PL-1.12 记录的一整类同步陷阱
（跨引擎同 buffer WAR、flag id 预算、单槽/双槽流水深度）。

### 7.4 已知边界

- **无 `store_fixpipe`**：dq/dk/dv 需要跨任务累加（原子），而 `store_fixpipe` 只支持 L0C→GM 且无累加语义（`CONST-store-fixpipe-gm-only`）⇒ 累加器在 UB、由 Vector/MTE3 侧写出。
- **无 NZ/fractal 显式布局**：Developer 保持 ND（cube skill）；L0A/L0B 的 NZ 转换由后端隐式处理（同族先例实测 gemm 结果正确，CG-2026-0004 的 workaround 记录）。
- **`T.alloc_shared` 的 UB/L1 归属由编译器决定**：本设计的 §4.5 按「全量计 UB」给出保守上界，按「gemm 操作数落 L1」给出实际口径；两者均在预算内（主选档）。

---

## 8. 验证方案

### 8.1 Golden 函数草案（PyTorch 参考实现，独立于 NPU 算法）

**唯一依据 = §0.1 的语义**：fp32 物化 softmax（无 online softmax、无分块、无 tile 形状、无原子累加策略），
从给定的 `(q,k,v,do,lse,delta)` 直接算出 `(dq,dk,dv)`；`lse` 按 **base-2 域**消费。逐 `(b,h)` 循环以控制峰值显存
（同包 fwd 的 golden 同款做法）。

```python
import math
import torch

LOG2E = 1.4426950408889634


def golden_gqa_bwd(q, k, v, do, lse, delta, is_causal):
    """BSHD GQA/MHA backward reference (fp32 materialized softmax).

    q/do: (B,S,H,D) fp16/bf16 ; k/v: (B,S,Hkv,D) ; lse/delta: (B,H,S) fp32 (lse 为 base-2 域)
    returns dq:(B,S,H,D) fp32, dk:(B,S,Hkv,D) fp32, dv:(B,S,Hkv,D) fp32
    """
    batch, seq_len, heads, dim = q.shape
    heads_kv = k.shape[2]
    groups = heads // heads_kv
    scale2 = (dim ** -0.5) * LOG2E        # base-2 域的 scale（与源一致）
    sm_scale = dim ** -0.5

    qf = torch.transpose(q.to(torch.float32), 1, 2)     # (B,H,S,D)
    kf = torch.transpose(k.to(torch.float32), 1, 2)     # (B,Hkv,S,D)
    vf = torch.transpose(v.to(torch.float32), 1, 2)
    dof = torch.transpose(do.to(torch.float32), 1, 2)
    dqf = torch.zeros_like(qf)
    dkf = torch.zeros_like(kf)
    dvf = torch.zeros_like(kf)

    qi = torch.arange(seq_len).view(seq_len, 1)         # query 位置
    kj = torch.arange(seq_len).view(1, seq_len)         # KV 位置
    valid = torch.le(kj, qi)                            # causal: kv <= q

    for b in range(batch):
        for h in range(heads):
            g = h // groups
            kt = torch.transpose(kf[b, g], 0, 1)                      # (D,S)
            scores = torch.matmul(qf[b, h], kt)                       # (S,S) 自然域
            scores = torch.mul(scores, scale2)                        # → base-2 指数域
            p = torch.exp2(scores - torch.unsqueeze(lse[b, h], -1))   # P（S,S）
            if is_causal:
                p = torch.mul(p, valid.to(torch.float32))
            dw = torch.matmul(dof[b, h], torch.transpose(vf[b, g], 0, 1))   # dO·Vᵀ
            ds = torch.mul(p, dw - torch.unsqueeze(delta[b, h], -1))
            ds = torch.mul(ds, sm_scale)
            dvf[b, g] = dvf[b, g] + torch.matmul(torch.transpose(p, 0, 1), dof[b, h])
            dkf[b, g] = dkf[b, g] + torch.matmul(torch.transpose(ds, 0, 1), qf[b, h])
            dqf[b, h] = torch.matmul(ds, kf[b, g])
    return (torch.transpose(dqf, 1, 2), torch.transpose(dkf, 1, 2), torch.transpose(dvf, 1, 2))


def golden_lse_delta(q, k, v, do, is_causal):
    """(可选) 自洽提供 lse（base-2 域）与 delta，使 kernel 级用例自包含。"""
    qf = torch.transpose(q.to(torch.float32), 1, 2)
    kf = torch.transpose(k.to(torch.float32), 1, 2)
    vf = torch.transpose(v.to(torch.float32), 1, 2)
    dof = torch.transpose(do.to(torch.float32), 1, 2)
    heads_kv = k.shape[2]
    groups = q.shape[2] // heads_kv
    qi = torch.arange(q.shape[1]).view(q.shape[1], 1)
    kj = torch.arange(k.shape[1]).view(1, k.shape[1])
    lse = torch.empty(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
    delta = torch.empty_like(lse)
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            g = h // groups
            scores = torch.mul(torch.matmul(qf[b, h], torch.transpose(kf[b, g], 0, 1)),
                               q.shape[3] ** -0.5)
            if is_causal:
                scores = scores.masked_fill(torch.logical_not(torch.le(kj, qi)), float("-inf"))
            lse[b, h] = torch.div(torch.logsumexp(scores, dim=-1), math.log(2.0))  # base-2 域
            o = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype).to(torch.float32),
                             vf[b, g])
            delta[b, h] = torch.sum(torch.mul(o, dof[b, h]), dim=-1)
    return lse, delta
```

**独立性核对**：golden 只依赖 §0.1 的数学定义与 `lse`（base-2 域）契约，
**不引用**分块循环、tile 形状、v-prefix API、原子策略、UB 预算等任何 NPU 侧实现要素 ⇒ 验证独立性成立。

### 8.2 精度标准

| 项 | 标准 | 依据 |
|----|------|------|
| 主门禁（tier-1） | `dq/dk/dv` 对 golden：`atol = 5e-3`、`rtol = 1e-5` | manifest op 级契约（`examples/TileOPs/tests/ops/test_multi_head_attention.py` 第 251 行） |
| tier-2（2-ulp 放宽） | 在 tier-1 未过时启用 `atol + (rtol + 2·ulp_rel)·|ref|` | 同包 fwd 的双门禁口径（`_TWO_ULP_REL = {fp16: 2**-9, bf16: 2**-6}`） |
| tier-3（噪声分类，**须显式记录**） | 失败元素 ≤ 0.01% 且每个失败元素 ≤ `2·flip_rel·|dO|absmax`（`flip_rel = {fp16: 2**-11, bf16: 2**-8}`），且 `fp32` 累加器相关项不受影响 ⇒ 归类为 **P/dS 输入-dtype 量化的放大噪声**（源算法固有：源的 `qkt_cast`/`dst_cast` 也把 P/dS 回落输入 dtype） | 同包 fwd 的 tier-3 先例（`_check_output`，含 `|kernel−true| < |golden−true|` 的逐元素实测证据）；本设计的 P/dS 量化点与源**逐点对齐** |
| `dq` 的结合序 | 允许 `dq` 的原子累加顺序与源不同（P2：上界 1.8e-6 ≪ 容差）；**门禁不断言 bit 一致**，但**断言与 fp64 参照的偏差不劣于源序** | §1.6.1 P2 |
| 自洽性检查 | `delta` 与 `lse` 必须来自同一前向（`golden_lse_delta` 提供）；若外部传入，须校验 `lse` 为 base-2 域（本包 fwd 的 golden 口径） | §0.1 「`lse` 域」行 |

### 8.3 L0 门槛测试计划（Stage 3 必须全过）

> 分层套件：本节只定义 **L0（门槛）**；L1（泛化）/ L2（鲁棒）/ Boundary（边界）由 `tilelang-op-develop` 扩展，不在此枚举。

| # | 用例 | shape | dtype | 目的 |
|---|------|-------|-------|------|
| L0-1 | `smoke`（manifest） | (1,1024,8,64) | fp16 | 非 causal 主路径；`num_logical=128` → 20 核 / 7 轮 |
| L0-2 | `smoke` bf16（manifest） | (1,1024,8,64) | bf16 | bf16 路径（v 链仍在 fp32；只有装载与 `cast` 是 bf16） |
| L0-3 | `full`（manifest） | (16,2048,16,128) | fp16 | `dim=128`、`bn=32`、`num_logical=8192`（410 轮）；**发射受限主战场** |
| L0-4 | `full` bf16（manifest） | (4,4096,16,128) | bf16 | `S=4096`、`⌈S/bm⌉=64`（205 轮）；与 L0-3 构成 `S` 与 `H` 的交叉 |
| L0-5 | causal 主路径 | (1,256,8,64) | fp16 | `is_causal=True`：`loop_st` runtime 界 + 掩码惩罚链 + 平面物化 |
| L0-6 | causal + 非整除 | (1,300,8,64) | fp16 | `loop_st` + `real_m`/`real_n` 同时生效（`300 % 64 ≠ 0`、`300 % 64 ≠ 0`） |
| L0-7 | 非整除（非 causal） | (1,1000,8,64) | fp16 | `real_m`/`real_n` 绑定：`k_s`/`q_s` 装载裁剪 + `size=` fractal 裁剪 + 写出裁剪 |
| L0-8 | **GQA** | (1,256,8,2,64)（`heads=8, heads_kv=2`） | fp16 | `groups=4`：`hkv = bx//groups` 映射 + `dk/dv` 的**原子**分支（§0.6 R3 的 `groups>1` 路径） |
| L0-9 | GQA + causal + 非整除 | (1,300,8,2,64) | bf16 | 三个边界条件的组合（bf16 × 非整除 × 分组原子的笛卡尔覆盖，`TRAP-L1-band-dst-tail-overrun` 的教训） |
| L0-10 | 退化小档 | (1,64,1,64) | fp16 | `num_logical = 1 < 20` ⇒ `num_kernels=1`（无超发）；`H=1` |
| L0-11 | `dim=64` + causal | (2,128,4,64) | fp16 | `bn=64`（`dim≤64` 档）与 `bm=64` 的组合 |
| L0-12 | 数值角点 | (1,64,2,64) | fp16 | ① 全零 `q/k/v/do` ⇒ 全零梯度（`lse` 给 0）；② `lse = +inf` 的整行 ⇒ `P=0` ⇒ 该 query 行梯度为 0；③ 大分数（`q,k ~ N(0,4)`）⇒ `P` 饱和到 1/0 的稳定性。**±inf/NaN 输入按 INFO 记录，不断言**（§0.6 R4 的差异域） |
| L0-13 | 输入校验 | — | `dtype="float32"` / `heads=8, heads_kv=3` / `dim=24` | 工厂期 `raise ValueError`（§5.3） |
| L0-14 | `dq` 的确定性记录 | (1,256,8,64) | fp16 | 同输入连跑 3 次，记录 `dq` 的 `torch.equal` 结果与最大差（**只记录不断言 bit 一致**，P2） |
| L0-15 | UB 编译阶梯 | L0-3 形状 | fp16 | 以 `(bn=32)` 主选档编译；再以 `(bn=64)`（上探档）尝试，记录是否 `ub overflow` ⇒ **反解真实 auto multi-buffer 膨胀系数**（§4.5 双口径的实测校准） |
| **L0-16** | **H>1 跨步写出的区域完整性（revision 1 新增，REVIEW.md 问题 1 建议 4）** | (2,300,16,4,128)（`heads=16, heads_kv=4`，`groups=4`） | fp16 | **专门检测「逐元素区域写错位置」类缺陷**：`H=16 > 1`、`dim=128 ≠ H·D`、`S=300` 非整除 ⇒ 逐 head（尤其 `h ∈ {0, 7, 15}`）核对 `dq/dk/dv` 的目标区逐元素正确性 + 其它 head/行未被污染（revision 0 的 rank-reduced 区域形态在此几何下会丢写目标区并越界污染，见 §0.6 R3）；同时覆盖 GQA 的 `groups>1` 逐行行原子分支 |

**测试装置要求**：`dq/dk/dv` 必须由测试侧 `torch.zeros(...).float()` **显式置零**后传入（§9 R5）；
`lse` 必须来自与 `delta` 同一前向（本套件用 §8.1 的 `golden_lse_delta` 生成，base-2 域）。
**L0 全过即可进入 Stage 4**；tier-3 命中时须在测试输出中打印分类依据（元素数占比 + 放大包络）。

---

## 9. 风险点与注意事项

> 技术约束检测结论（三维 Kernel / GPU 专用 API / GEMM 非整除 / L0C 容量 / UB 预算 / bf16 无向量算术 / 物理核数适配）
> 已逐项在 §3.5.1 给出处理方案；本节列出**残余风险**与设计期已识别的不确定项。

| # | 风险 | 影响 | 证据 / 状态 | 处置 |
|---|------|------|------------|------|
| **R1** | `dq` 的跨核/跨任务**同址并发原子**的可靠性（**revision 1 重定性**） | 原判为头号风险；revision 1 已**关闭硬件/原子语义维度**，转为「形态正确性」已被正面覆盖 | **revision 1 实测证据**：① `T.atomic_add` 是**真正的 fetch-add**（非覆盖 store）；② **20 个 block 并发同址写同一批行 ⇒ 每 cell 精确 20.0、区外 0**（`history_version/design_probe_row_atomic_bshd.py`，910B3 + CANN 9.0.0）；⇒「并发同址丢更新」假设在本工具链**不成立**。**revision 0 的「1/4 异常」根因已定位**：既非丢更新、也非探针期望值，而是探针一的**跨步 case 期望值写错**（`expect=0.0`，实际 band 应 += `NBLK_N`；revision 1 已修正并重跑，band[128:256] 全 4.0、区外 0）+ revision 0 把 rank-2 显式切片探针结论**外推**到未测的 rank-reduced 4-D BSHD 形态（该形态实测**编译失败或静默写错区域**） | **残留风险 = 修正形态（逐行 1-D 行原子）的正确性**，由 L0-1/L0-3/L0-8/**L0-16** 的逐元素门禁正面覆盖（失败是响亮的逐元素失败）；若 Stage 3 仍复现并发出错 ⇒ 按 §1.6.4 切「两趟原子自由」零原子备选。**旁证**：三个同族 npuir 实现（`examples/deepseek_v32/sparse_mla_bwd.py`、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`、`examples/indexer/indexer_bwd.py`）均依赖并发原子散写并在生产使用 |
| **R2** | 发射项单价常数不确定（±1 个量级；revision 1 新增「行原子单价」未解析） | 设计期估算下界区间宽（full 档 **9.5–550 ms**，revision 1 重算）；若落在串行上界档，本算子远慢于流量地板 1.27 ms | §1.6.0 估算表 + 「模型可靠性声明」（单价档来源与版本戳已标注；行原子单价为**未文档化假设**：探针三⑤ 32 行原子 ≈ 1 区域 store，被测试床 ~120 µs/launch 地板掩盖） | Stage 4 以 L0-15 与 `aiv_vec_time`/**`aiv_mte3_time`** 反解真实单价；第一杠杆 = 单迭代 tile 面积（§1.6.3 参数级 A/B 的 `bn=64` 上探档）+ 结构级 A/B（§1.6.4 两趟） |
| **R3** | UB 膨胀系数（×1.12 vs ×1.7）未知 | 上探档 `(64,64)@dim=128` 在口径 B 下超限；主选档双边均可行（无阻塞） | §4.5 双口径（`CONST-capacity-910B2C` 的默认档 ~1.7x 与显式-alloc 结构 1.10–1.12/0.981 两组实测） | 工厂 guard + 阶梯降档（§4.5）；L0-15 反解 |
| **R4** | causal 档的 `T.Pipelined(loop_st, n_q, ns)` 使用 **runtime start** | 若 npuir 不支持 runtime 界 ⇒ causal 档编译失败（**仅影响 causal**；manifest 四档全为非 causal） | 源 GPU 代码即用 runtime 界（第 580 至 583 行）；同族 npuir 先例（`examples/deepseek_v32/sparse_mla_bwd.py`）用的是静态界 ⇒ **未文档化假设** | 兜底形态：`loop_st = 0` + 保留掩码惩罚链（**结构不变**，只是失去块级跳过 ⇒ causal 的 FLOPs 减半优化失效）；该兜底在 §8.3 的 L0-5/L0-6/L0-9 中实测筛选 |
| **R5** | `dq` 依赖调用方**预置 0** | 若 `dq` 未置零，原子加会把旧值累加进去（**源同款契约**） | op 层 `torch.zeros_like(q, dtype=torch.float32)`（`tileops/ops/attention/gqa.py` 第 1378 行）；§0.6 R3 的语义论证亦以「预置 0 + 单写者 ⇒ 等价普通写」为前提 | §8.3 测试装置显式置零；wrapper 契约不变（不改 wrapper） |
| **R6** | `T.vmin`/`T.vsub`/`T.vmul` 的 **bare 标量操作数**可能触发前端 rank 断言 | causal 档编译失败 | `T.vmin` §2.1 明标「tensor or scalar」✓；但 `TRAP-npuir-vbrc-region-dropped` §「标量 operand 形态」记录**裸 Python 数**会命中前端 rank 断言（需 typed `T.float32(1.0)`）——该记录针对 `vbrc`，`vsub`/`vmul` 的标量形态另有生产先例（`attention.md` PL-1.11：`vmin(idx,0,pen)` → `vmul(pen,PEN,pen)` → `vadd(diff,pen,diff)`） | 首选 typed 标量 `T.float32(0.0)` / `T.float32(-_PEN)`；仍失败 ⇒ 退 `vcmp`+`vselect` 精确形态（§1.6.4 末段） |
| **R7** | `TRAP-DEVMODE-PERSIST-GEMM`（Developer + persistent + gemm） | 若在本工具链复现 ⇒ 全部 20 核路径崩溃 | **本任务探针二 Q3 已在本工具链复验：编译通过 + `max_abs_err=4.578e-05`** ⇒ 按证伪协议「工具链变更后待重验」该 trap 结论在本工具链**不成立**；本设计据此保留 Developer + persistent | 若 Stage 3 复现同症状 ⇒ 按 §2.3 切 Expert 双 Scope（同结构、同分核三要素）；重验结论已写入 `RETROSPECTIVE.md` 作为 pattern-library 更新候选（含工具链版本戳） |
| **R8** | P/dS 的输入-dtype 量化噪声 vs 不量化 golden | 可能出现少量超 tier-1 的元素（**源算法固有**） | 本设计的量化点与源**逐点对齐**（`qkt_cast`/`dst_cast`）；同包 fwd 有同类 tier-3 先例与逐元素证据 | tier-3 分类（§8.2）须显式打印依据；同时提供「源忠实」golden 变体（P/dS 也回落输入 dtype）作为 tier-2 对照 |
| **R9** | `groups > 1`（GQA）的 `dk/dv` **逐行行原子**分支（revision 1 修订形态） | GQA 正确性 | §0.6 R3 的语义论证（`groups>1` ⇒ 多写者 ⇒ 必须原子）；逐行行原子形态由 L0-8/L0-9/**L0-16** 正面覆盖；探针三的并发同址精确结论对该形态成立 | L0-8/L0-9/L0-16 覆盖；若 R1 的修正形态仍出错则 GQA 一并走 §1.6.4 零原子备选 |
| **R10** | `num_stages=1` 的流水重叠有限 | 搬运/计算可能不完全重叠（流量项被放大） | wrapper 默认 `num_stages=1`；`T.Pipelined` 仍需 stage≥2 才双缓冲 | Stage 4 参数级上探（UB 允许时，`ns ∈ {2,3}`）；与 §1.6.3 的 tile A/B 联合扫描 |
| **R11** | 掩码惩罚在**非有限 score** 下与源的语义差异（NaN 传播 vs 置 0） | 仅影响含 NaN/Inf 输入的用例 | §0.6 R4 已登记；差异域在有限输入下为空 | §8.3 L0-12 按 INFO 记录；需要精确语义时切 `vcmp`+`vselect` 形态（§1.6.4 末段） |
| **R12** | 首次落盘的三件套 ABI（`q,k,v,do,lse,delta,dq,dk,dv`）与 wrapper 的调用点 | ABI 不符 ⇒ wrapper 调用失败 | wrapper `GQABwdKernel.forward` = `self.kernel(**self.config)(*inputs)`，`self.config = {block_m, block_n, num_stages}`（`multi_head_attention.py` 第 476 至 506 行） | 本设计的 builder 签名 `(block_m, block_n, num_stages)` 与之**逐键对齐**；**不做 trace-time 配置替换**（§3.3） |

**设计期已识别但**不构成风险**的项（供复核）**：`make_dq_layout`（舍弃，§0.5）、`make_swizzled_layout`（舍弃）、
`pass_configs`/`compile_flags`（不使用）、`threads`（不使用）、3D UB 声明（不使用，全 2D）、
`T.atomic_addx4`（文档明示与 `T.atomic_add` 等价 ⇒ 无额外收益）。

---

## 10. 交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 设计文档 | `examples/multi_head_attention/_gqa_bwd_kernel/DESIGN.md` | ✅ 本文件 |
| D-1 等价性验证脚本 | `examples/multi_head_attention/_gqa_bwd_kernel/verify_equiv.py` | ✅ 已执行（§1.6.1 结果表；全部采纳项 EQUIV_PASS） |
| 设计期探针 1（原子单价） | `examples/multi_head_attention/_gqa_bwd_kernel/history_version/design_probe_atomic_vs_store.py` | ✅ 已执行并**修正期望值判据后重跑**（连续区域原子/store = **0.97×**；rank-2 跨步 band 原子与 store 逐位精确） |
| 设计期探针 2（persistent 可行性 + rank-2 跨步区域正确性） | `examples/multi_head_attention/_gqa_bwd_kernel/history_version/design_probe_persist_and_strided.py` | ✅ 已执行并**修正 `n_write` 判据后重跑**（persistent+gemm 可用；rank-2 显式切片跨步 store/atomic 逐位正确） |
| **设计期探针 3（revision 1 新增：修正形态的逐行 1-D 行原子）** | `examples/multi_head_attention/_gqa_bwd_kernel/history_version/design_probe_row_atomic_bshd.py` | ✅ 已执行（目标区全中 / 区外 0 / 20 并发精确 20.0 / 动态行数正确；行原子单价未解析） |
| 复盘 | `examples/multi_head_attention/_gqa_bwd_kernel/RETROSPECTIVE.md` | ✅ 本 Stage 写入（含 revision 1 更正条目） |
| Stage 3 待产出（本算子目录下的实现文件） | `_gqa_bwd_kernel.py`（工厂 + §8.1 golden + `__main__` 的 L0 套件；文件顶层写死 `ASCEND_MODE = "Developer"` 并 `os.environ.setdefault("TILELANG_ASCEND_MODE", ASCEND_MODE)`） | ⏳ 交 Stage 3 |
| Stage 4 待产出 | `perf_opt/_gqa_bwd_kernel.py` + perf records（§1.6.3 参数级 A/B：`bn` 上探档 / `ns` 上探档；§1.6.4 两趟备选的实验裁决，含**行原子单价**标定） | ⏳ 交 Stage 4 |
| Stage 5 待产出 | TileOPs 集成（不改 wrapper、不改 fwd 产物、不改同批另两个函数） | ⏳ 交 Stage 5 |
| 常数候选（供 evolver 蒸馏） | 910B3 + CANN 9.0.0 的「fp32 **连续**区域原子 / 普通 store 单价比 = 0.97」；「**逐行 1-D 行原子在 BSHD 目标上正确且 20 并发同址精确**」；「Developer + persistent + gemm 在此工具链可用」（推翻 `TRAP-DEVMODE-PERSIST-GEMM`）；「**rank-reduced 4-D 目标**的区域 `T.atomic_add` 不可用（动态 extent 编译失败 / 静态 extent 静默写错）」 | 📝 见 `RETROSPECTIVE.md` 的 Value Point Proposals（Stage 1 的「BSHD 形态逐位正确」条目已更正） |

---

## 11. 修订说明（revision 1，`mode=revision`）⭐

> 触发：Stage 2 检视结论**不通过**（阻塞 2 条 + 建议 5 条，见 `REVIEW.md`）。本版**原地重写** `DESIGN.md`（不新建 `design_v1.md`），
> 保留全部已通过的调研 / 优化 / 等价性结论（§1.6.1 的 O1/O2/O3/O4 与 P1/P2/P3 全部 EQUIV_PASS 项），仅在被本修订影响处更新。
> **未改动项**：§0.1/§0.2/§0.3/§0.4（源算子语义/算法/优化手段解读）、§1.6.1 的 O1–O4 采纳项与 §1.6.2 的 #3–#8 向量化项、
> §1.6.3 的布局决策（P tile `[q,kv]` 轴向 + 行广播）、§2 编程模式（Developer）、§5.5 分核三要素、§8.1 golden（独立于 NPU 算法）。
> **§8 编号契约保持**（`## 8. 验证方案` 含 Golden 函数说明与 L0 计划）；**§5 分核三要素保持**（① 逻辑核数计算 / ② 物理核数 `NPUUtils.get().get_aicore_num()` 实查记录 / ③ 规模判定与分核方案）。

### 11.1 相对上一版（v0）的关键调整

| # | 对应检视问题 | v0（被回退） | v1（本版） | 为什么不会再犯同一错误 |
|---|-------------|-------------|-----------|----------------------|
| 1 | **问题 1（阻塞）** | §0.6 R3 / §3.2 B7/B8 / §3.3 / §4.4 / §7.1 / §1.6.2 #9–#10 用 **rank-reduced 4-D 二维区域 `T.atomic_add(dq[bz, c0:c0+real_n, bx, 0:dim], dq_blk, [real_n, dim])`**（跨步 BSHD 目标）——本机实测动态 extent 编译失败、静态 extent 静默写错区域 | 改为**逐行 1-D 行原子**：`for r in T.serial(real_n): T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r, 0:dim], [1, dim])`；`groups>1` 的 dk/dv 同款；`groups==1` 仍走 `T.copy` 区域存储 | **形态三要素核对纪律**（张量 rank / 索引方式（标量 vs 切片 vs runtime Var）/ 目标行 stride 关系）：新形态 rank=1、索引全为标量 + 尾轴切片、行 stride 由标量行索引表达 ⇒ 不再与 docs 示例形态（rank-2 + `len(size)==rank`）混淆；且由**探针三在 BSHD 几何上正面实测**（落盘可复核） |
| 2 | **问题 2（阻塞）** | §1.6.0 探针表把 rank-2 显式切片探针证据**外推**为「BSHD `dk[b,s,h,:]` 形态逐位正确」；两枚旧探针的 `exact` 判据写错（`expect=0.0` / `n_write=NBM`）⇒ 期望值恒 False，只能靠人读原始量解释 | ① 探针表逐条注明**实际被测形态**（rank-2 双侧显式切片）并删除外推；② **修正两枚旧探针的期望值判据后重跑**；③ **新增探针三**专测修正形态（断言：目标区元素计数 / 区外元素计数 / 并发同址总量 / 动态行数），附工具链版本戳 | **证据链 = 工件形态 + 自证判据**：每条「已实测」断言必须能打开探针源码核对被测形态与期望值构造式；探针自带 `exact` 判据并打印「期望 vs 实测」两列 |
| 3 | **问题 3（建议）** | §1.6.0 R3 的「口径 B（无 L2 吸收的总线口径）」只计原子写放大、漏计同口径的循环轴重读 ⇒ 口径不自洽；§1.6.4「口径 B 下备选反超（2.01 vs 9.66 GB）」量化前提不成立 | 口径 B **改名「口径 B′（仅原子写放大）」并写明定义**；**新增口径 C（统一总线：A + 重读 + 原子放大）**并给候选 0/1/2/3/5 的正确数字（17.98 / 27.1 / 13.8 / 52.1 GB）；§1.6.4 的备选理据改为**正确性驱动 + 发射项重算**，删除未成立的性能论断 | **每个口径先写定义句（含什么/不含什么）**，同一口径下所有候选使用同一记账公式；跨口径比较前先对齐记账项 |
| 4 | **问题 4（建议）** | §4.5 兜底档 `(64,16)` 违反全篇声明的 K ≥ 32（g3/g4 的 K = `bn` = 16），工厂 guard 只校验 `≥16` | 兜底档改为 **`(32,32)`**；工厂 guard 改为 **`block_m, block_n ≥ 32`**（K 维下界）；§4.5 阶梯表与「域上界」行、§3.3 注、§5.1/§5.2/§3.5.1 同步 | **声明即校验**：凡正文声明为「下界」的量（M/N ≥ 16、K ≥ 32）必须在工厂 guard 中有对应判据，且**按每个 GEMM 逐一列出其 K 维来源**（= `dim`/`bn`/`bm`） |
| 5 | **问题 5（建议）** | `verify_equiv.py` 的 O1 掩码分支仅被 `(16,64)` 一档覆盖（另 5 档 masked = 0）；脚本只验 clamp 形式 | `_cases()` 扫 **4 个掩码偏移 `d0`**（110 案例、80 含 masked、6 个 tile 全覆、61 740 个 masked 元素）；`check_o1` **三向互比**（clamp 形式 / §3.3 的 `min(−PEN·(diff−d0),0)` 形式 / 源 select）并打印掩码覆盖统计；§1.6.1 表与 O1 行同步 | **角点/分支覆盖显式可见**：覆盖率统计随结果表打印；「机器验证固化的公式形态」与「伪代码形态」必须**逐形态断言** |
| 6 | **问题 6（建议）** | §9 R1 把「并发同址区域原子丢更新」列为头号风险且旁证推理不成立；§9 R2 用旧发射项 | §9 R1 **重定性**：`T.atomic_add` 为真 fetch-add、20 并发同址精确 ⇒ 硬件/原子语义维度**关闭**；异常根因定位为「探针期望值写错 + 形态外推」；残留风险 = 修正形态正确性，由 L0-1/L0-3/L0-8/L0-16 正面覆盖。§9 R2 按**新发射项**重算（9.5–550 ms） | **结论与证据同源**：每条风险状态必须写「实测证据 + 工具链戳」或「未文档化假设 + 估算依据」；不再用降级名/纸面模型作旁证 |
| 7 | **问题 7（建议）** | 内部一致性小项：O1 三种等价写法与命名漂移、pass 计数 9 vs 8、候选 2 的 2.01 vs 1.88 GB、候选 5 的 17.6 GB、`make_dq_layout` 行号（43–47，实为 GPU 源 99–103）、§0.3 的 contiguous 声称、§1.6.2 #13 常数引用 | 逐一修正：**单一定义源**（以 §3.3 的 `vmin(·,0)` 形式为准）；**统一计数口径**（整 tile 向量 op 8/12 + 2 个 `[bn,1]` 微 op + `real_n` 行原子）；候选 2 = `E·(8·eb+12)` = 1.88 GB、候选 3 重算、候选 5 重算；`make_dq_layout` 行号改指 GPU 源第 99 至 103 行；§0.3 改为「只对 `do` 显式 `contiguous()`」；§1.6.2 #13 改引 `CONST-aiv-time-decomposition` 的判据句 | **单一事实源 + 表头声明口径**：任何在 ≥2 处出现的公式/计数/常量只在「定义处」写公式，其余处引用；跨节数字互校前先核对口径注 |

### 11.2 本版新增 / 修改的工件

- `history_version/design_probe_row_atomic_bshd.py`（**新增**）：修正形态在 BSHD 几何上的三类断言 + 单价标定。
- `history_version/design_probe_atomic_vs_store.py`（**修正期望值 + 重跑**：跨步 band 期望 `NBLK_N` 而非 0；新增 strided store 精确性断言；docstring 标明适用范围）。
- `history_version/design_probe_persist_and_strided.py`（**修正判据 + 重跑**：`n_write = 1`；docstring 标明被测形态为 rank-2 显式切片）。
- `verify_equiv.py`（**扩覆盖 + 重跑**：O1 扫 4 个 `d0`、110 案例、三向互比、打印掩码覆盖；O4 按 `(bm,bn)` 去重保持 60 案例）。
- `RETROSPECTIVE.md`（追加 Stage 1 revision 章节，并**更正** Stage 1 `first_design` 的「BSHD 形态逐位正确」蒸馏条目）。

### 11.3 重算后的发射项（revision 1 的核心量化变化）

每迭代 issued-op 清点（非 causal）：**8 整 tile 向量 op + 2 个 `[bn,1]` 微 op + `real_n` 个逐行行原子**
（v0 记账为 9，其中写出为 1 个区域原子；causal 档追加 4 个整 tile op）。`real_n` 随 tile 档变化（dim=128/bn=32 ⇒ 32；dim≤64/bn=64 ⇒ 64）。
代表 workload `full-bwd-fp16-large`：迭代 `524 288`（每核 26 214）× 42 op = **1 101 005 op/核**
⇒ 上界档 **275–550 ms**、元素吞吐地板档 **9.5 ms**；流量项（口径 A）**1.27 ms** 不变。
**流量/元素口径与 GEMM 数完全不变**（写出字节数、整 tile 向量 op 数、GEMM 5 个均不变）——本修订只抬高 issued-op 记账，
且新引入的「行原子单价」被探针三⑤ 标定为**未解析常数**（保守按整 op 计价）。
