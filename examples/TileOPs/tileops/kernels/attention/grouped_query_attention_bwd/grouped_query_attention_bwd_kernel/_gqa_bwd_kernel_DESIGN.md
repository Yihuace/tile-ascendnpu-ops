# `_gqa_bwd_kernel` 算子设计文档

> 迁移任务（scenario=migration / migration_mode=harness）：将 GPU TileOPs `GroupedQueryAttentionBwdOp` 的**第 3 个（主）TileLang 函数** `_gqa_bwd_kernel`
> （源：`/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py` L514–621，类 `GQABwdKernel` L622–665）迁移到 `target="npuir"`。
> 编程模式：Developer（用户指定）。目标设备：**Ascend910B3**（`NPUUtils.get().get_aicore_num()` 实查 = 20 AI Core，UB 192KB）。
> 同任务前两个函数（preprocess / postprocess）已完成 Stage 1–3，本算子复用其骨架与排坑结论。

---

## 修订说明（revision v1，相对 `history_version/design_v0.md`）

> 本版为 Stage 2 检视不通过（1 阻塞 + 5 建议）后的设计修订（revision_index=1）。**首次设计的全部有效结论均保留**（MixCV 一维非持久 grid、GQA head 合并 + `dk/dv` 零原子、`dq` 行段原子、tiling `BM=64/BN=64(32)`、`T.Pipelined num_stages=1`、Hopper 专属项处置、采纳的 2 项优化），仅按 REVIEW 逐条修正下列问题。

| # | 问题（严重度） | 本版关键调整 | 为何不会再犯同一错误 |
|---|---------------|-------------|---------------------|
| 1 | 因果掩码未条件化（**阻塞**） | §3.3 伪代码把掩码 pass 包进编译期 `if is_causal:` 守卫（`else` 走不加掩码 + `loop_st=0`），并同步 §1.4 步骤 4、§1.6.1 采纳项、§3.1/§3.2、§6.2、§8.2、§9.2、§0.6 R1 | 「掩码/裁剪/循环下界」三处均改为同一 `is_causal` 守卫，逐处交叉核对；golden（§8.1 早已条件化）、契约、§8.2 边界用例陈述一致 |
| 2 | §4.5 UB 小计与表内加和不符（建议） | 逐行复算：不含 `tmp` 为 98560 B≈96.25 KB；含 `T.vexp2` 的 `tmp` 为 106752 B≈**104.25 KB**（x1.12→116.8 KB）；全文 106KB/118.7KB 引用同步订正 | 小计改为「表内各行显式加和」的写法，并列出加和算式，杜绝声明值与表内数值脱节 |
| 3 | #3 原子量倍数 16x 复算不成立（建议） | 改为 `2*BM/BN` = **4x**（推导：`2·B·H_kv·S·D·groups·(S/BN) ÷ [B·H·S·D·(S/BM)] = 2·(H_kv/H)·groups·(BM/BN) = 2·BM/BN`），同步 §1.6.0 R3/R4、调研结论、§1.6.3 | 量化数字一律写出完整推导式再代入；`H=H_kv·groups` 的消元在推导中显式体现 |
| 4 | `T.exp2` NPU 先例引用不实（建议） | 主选改为 `T.vexp2(src,dst,tmpBuffer)`（补 `tmp` UB 缓冲并计入 §4.5）；§3.4 佐证改为 `T.vexp2.md` + `sparse_mla_bwd.py` L234；同步 §3.2/§3.3/§6/§9.2 | 引用的先例逐行核对 API 名（`sparse_mla_bwd` 用的是 `T.vexp2` 非 `T.exp2`）；并在 §9.4 采纳知识库 `CG-2026-0018`/`VP-2026-0172` 的向量 vexp2 告警 + 回退链 |
| 5 | verify_equiv sentinel 与实现不一致（建议） | `verify_equiv.py` 候选式改用实现常数 `NEG=-3.0e38`（并 `assert NEG==-3.0e38`），重跑回写结果表 | 机器验证的数值口径与实现规格逐位对齐，不再使用 `-inf` |
| 6 | 细节引用/统计偏差（建议） | 修正 §0.1 行号（`scale`/`sm_scale`→L521–522、`bx//groups`→L575）、§0.5 统计（#9 从保留清单移除；等价替换 3 项=#1/#4/#9）、`FullRow` 处数（5→**4** 处 L586/597/599/605）、候选库条目名（`ALG-attention`） | 行号/统计口径逐处回源核对；`H=H_kv·groups` 相关计数与 §0.4 表逐项对齐 |

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `tilelang-op-design` skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：FlashAttention 风格的 **GQA（grouped-query attention）反向主 kernel**。给定 `q/k/v/do` 与前序 preprocess 产出的 `lse`、`delta`，重算注意力概率并求出三个梯度 `dq`（query 侧）、`dk`/`dv`（key/value 侧）。**GQA 语义**：`H % H_kv == 0`（源码 L523 显式校验），`groups = H // H_kv` 个 query head 共享同一个 KV head（`kv_head = q_head // groups`，对应源码 L575 的 `bx // groups`）。

源码逐行读出（L534–621）的精确公式（**不以 spec 的 `math_formula` 推测为准，以源码为准**）：

```
scale    = D^(-1/2) * log2(e)          # 源码 L522（LOG2E 折进 scale）
sm_scale = D^(-1/2)                    # 源码 L521（L523-524 为 heads % heads_kv 校验）

qkt[i,j] = sum_d k[b, kv0+i, h_kv, d] * q[b, q0+j, h, d]        # = K Q^T (gemm L586)
P[i,j]   = exp2( qkt[i,j]*scale - lse[b, h, q0+j] )             # L588-589
P[i,j]  <- 0   若 (kv0+i) > (q0+j)                              # L590-593 (causal)
dp[i,j]  = sum_d v[b, kv0+i, h_kv, d] * do[b, q0+j, h, d]       # = V dO^T (gemm L597)
ds[i,j]  = P[i,j] * ( dp[i,j] - delta[b, h, q0+j] ) * sm_scale  # L603-604

dv[b, kv0+i, h_kv, d] = sum_{h in group} sum_j P[i,j]  * do[b, q0+j, h, d]   # = P^T dO   (L599)
dk[b, kv0+i, h_kv, d] = sum_{h in group} sum_j ds[i,j] * q[b, q0+j, h, d]    # = ds^T Q   (L605)
dq[b, q0+j, h, d]     = sum_i ds[i,j] * k[b, kv0+i, h_kv, d]                 # = ds K     (L609)
```

其中 `i` 为 **KV 位置**（块内行，`kv0 = by*block_m`）、`j` 为 **query 位置**（块内列，`q0 = k_idx*block_n`）。

**I/O 契约（源码 L534–542）**：

| 参数 | Shape | dtype | 角色 |
|------|-------|-------|------|
| `q` | `(B,S,H,D)` BSHD | fp16/bf16 | 输入 |
| `k`,`v` | `(B,S,H_kv,D)` BSHD | same | 输入 |
| `do` | `(B,S,H,D)` BSHD | same | 输入 |
| `lse` | `(B,H,S)` | **fp32** | 输入（log2 域行 logsumexp） |
| `delta` | `(B,H,S)` | **fp32** | 输入（`= rowsum(o odot do)`，preprocess 产出） |
| `dq` | `(B,S,H,D)` | **fp32 累加器** | 输出（in-place 写） |
| `dk`,`dv` | `(B,S,H_kv,D)` | **fp32 累加器** | 输出（in-place 写） |

**输出参数位置**：`_gqa_bwd_kernel` 的 `@tilelang.jit`（L530–533）**无 `out_idx`**，`dq/dk/dv` 作为入参 in-place 累加（与 preprocess 的 `out_idx=[2]` 不同）。Op 侧（`gqa.py` L1381–1389）先 `torch.zeros_like(..., dtype=torch.float32)` 把三者清零，再调用本 kernel，最后 host 侧把 `dq`（非 Hopper 走 postprocess kernel）与 `dk/dv`（直接 `.to(dtype)`）转回存储 dtype。

**dtype 语义**：所有 gemm 累加、`P/dp/ds` 中间量均为 **fp32**（`accum_dtype="float"`，L525）；只有喂给 Cube 的 P/ds 被显式降到 `dtype`（`qkt_cast`/`dst_cast`，L598/L604/L607）。`lse`/`delta` 是 fp32 契约。

**累加语义**：`dk/dv` 在**核内寄存器跨 q-block 循环累加**（L599/L605），再**跨 group 的 query head 原子累加**（L613/L618）；`dq` 在核内跨 q-block（j 方向）累加，再**跨 kv block 原子累加**（L611）。所有累加均 fp32；原子加的顺序不确定 ⇒ **fp32 累加分组不确定**（§1.6.0 需处理）。

**边界语义**：causal 由 `loop_st = floordiv(by*block_m, block_n)`（L581）整块跳过对角以下的 q block + 块内 `if_then_else` 掩码（L590–593），无 nan/inf 特殊处理。`S_q == S_kv`（`gqa.py` L1346 源码注释标注为「尚未支持 `s_q != s_kv`」）。`heads % heads_kv != 0` 在 factory 内 raise（L523）。

**host 侧语义**：`do = do.contiguous()`（L1376）；`dq/dk/dv` 分配为 fp32 zeros；kernel 调用后 `dq` 走 postprocess kernel（非 Hopper）/`dk,dv` 走 `.to(dtype)`（L1387–1388）。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（SDPA 反向参考 / 上述公式直算），**不复刻** §0.6 的 NPU 算法结构。

### 0.2 源算子输入输出

见 §0.1 表。补充说明：

- 输出 shape 依据源码 L537–542（`dq`/`dk`/`dv` 的 `T.Tensor` 声明），与 manifest `shape_rules`（`examples/TileOPs/tileops/manifest/attention.yaml` L233–235）一致。
- `dq` 与 `q`/`do` 同 shape（`(B,S,H,D)`），`dk`/`dv` 与 `k`/`v` 同 shape（`(B,S,H_kv,D)`）。
- **布局关键点**：三个输出均为 BSHD，固定 `(b,h)` 时沿 `S` 的行是**跨步行**（行内 `D` 连续、行间 stride `H*D` 或 `H_kv*D`）——这决定了 NPU 侧读写的搬运形态（§1.6.3）与 `dq` 原子写的形态（§0.6 R3）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码 L540–621，逐语句归档，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.Kernel(heads, ceildiv(S,block_m), batch)`（L544–545） | — | 三维网格 `(bx=q_head, by=kv_block, bz=batch)` | 任务划分（非计算） |
| 2 | `T.copy(k[bz, by*bm:(by+1)*bm, bx//groups, :], k_shared)`（L576） | `k` GM | `k_shared[bm,D]` | 读 KV 块（整程序常驻） |
| 3 | `T.copy(v[...], v_shared)`（L577） | `v` GM | `v_shared[bm,D]` | 同上 |
| 4 | `T.clear(dv_frag); T.clear(dk_frag)`（L578–579） | — | fp32 累加器清零 | 累加初始化 |
| 5 | `loop_st = floordiv(by*bm, bn) if causal else 0`（L581） | — | 循环下界（causal 剪枝） | 跳过恒零 q block |
| 6 | `for k_idx in T.Pipelined(loop_st, ceildiv(S,bn), num_stages)`（L583） | — | q-block 循环（软件流水） | 沿 query 位置扫描 |
| 7 | `T.copy(q[bz, k_idx*bn:(k_idx+1)*bn, bx, :], q_frag)`（L584） | `q` GM | `q_frag[bn,D]` | 读 Q 块 |
| 8 | `T.gemm(k_shared, q_frag, qkt, transpose_B=True, policy=FullRow)`（L586） | `k_shared`,`q_frag` | `qkt[bm,bn]` fp32 | `qkt = K Q^T` |
| 9 | `T.copy(lse[bz,bx,k_idx*bn:...], lse_shared)`（L587） | `lse` GM | `lse_shared[bn]` fp32 | 读 lse |
| 10 | `qkt[i,j] = exp2(qkt[i,j]*scale - lse_shared[j])`（L588–589，`T.Parallel(bm,bn)`） | `qkt`,`lse_shared` | `qkt` fp32 | `P = exp2(...)` |
| 11 | `qkt[i,j] = if_then_else(by*bm+i <= k_idx*bn+j, qkt[i,j], 0)`（L590–593） | `qkt` | `qkt`（掩码） | causal 掩码 |
| 12 | `T.copy(do[bz, k_idx*bn:..., bx, :], do_shared)`（L594） | `do` GM | `do_shared[bn,D]` | 读 dO 块 |
| 13 | `T.clear(dst); T.gemm(v_shared, do_shared, dst, transpose_B=True, FullRow)`（L595–597） | `v_shared`,`do_shared` | `dst[bm,bn]` fp32 | `dp = V dO^T` |
| 14 | `T.copy(qkt, qkt_cast)`（L598） | `qkt` fp32 | `qkt_cast[bm,bn]` dtype | P 降精度供 Cube |
| 15 | `T.gemm(qkt_cast, do_shared, dv_frag, FullRow)`（L599） | `qkt_cast`,`do_shared` | `dv_frag[bm,D]` 累加 | `dv += P^T dO` |
| 16 | `T.copy(delta[bz,bx,k_idx*bn:...], delta_shared)`（L601） | `delta` GM | `delta_shared[bn]` fp32 | 读 delta |
| 17 | `dst_cast[i,j] = qkt[i,j]*(dst[i,j] - delta_shared[j])*sm_scale`（L603–604） | `qkt`,`dst`,`delta_shared` | `dst_cast[bm,bn]` dtype | `ds = P odot (dp-delta) * sm_scale` |
| 18 | `T.gemm(dst_cast, q_frag, dk_frag, FullRow)`（L605） | `dst_cast`,`q_frag` | `dk_frag[bm,D]` 累加 | `dk += ds^T Q` |
| 19 | `T.copy(dst_cast, dst_shared)`（L607） | `dst_cast` | `dst_shared[bm,bn]` dtype | ds 落 UB 供下次 gemm |
| 20 | `T.clear(dq_frag); T.gemm(dst_shared, k_shared, dq_frag, transpose_A=True)`（L608–609） | `dst_shared`,`k_shared` | `dq_frag[bn,D]` | `dq_partial = ds K` |
| 21 | `T.atomic_add(dq[bz, k_idx*bn+i, bx, j], dq_frag[i,j])`（L610–611，`T.Parallel(bn,D)`） | `dq_frag` | `dq` GM 原子累加 | 跨 kv block 归约 dq |
| 22 | `T.copy(dv_frag, dv_shared); T.atomic_add(dv[bz, by*bm:, bx//groups, :], dv_shared)`（L612–613） | `dv_frag` | `dv` GM 原子累加 | 跨 group head 归约 dv |
| 23 | `T.copy(dk_frag, dk_shared); T.atomic_add(dk[bz, by*bm:, bx//groups, :], dk_shared)`（L615–618） | `dk_frag` | `dk` GM 原子累加 | 跨 group head 归约 dk |

**数据流与内存访问模式**（源硬件视角，GPU/CUDA）：

```
GM[k] --T.copy--> SMEM[k_shared(bm,D) f16] --+   (整程序常驻)
GM[v] --T.copy--> SMEM[v_shared(bm,D) f16] --+
                                              |
for k_idx in Pipelined(loop_st, S/bn):
  GM[q]   --> SMEM[q_frag(bn,D)] --> gemm(Cube) --> FRAG[qkt(bm,bn) f32]
  GM[lse] --> SMEM[lse_shared(bn)] --> T.Parallel: qkt = exp2(qkt*scale-lse) --> mask
  GM[do]  --> SMEM[do_shared(bn,D)] --> gemm(Cube) --> FRAG[dst(bm,bn) f32]   (= V dO^T)
  FRAG[qkt] --cast--> FRAG[qkt_cast f16] --> gemm --> FRAG[dv_frag(bm,D)]  +=   (= P^T dO)
  GM[delta] --> SMEM[delta_shared(bn)]
  FRAG[dst],FRAG[qkt],SMEM[delta_shared] --T.Parallel--> FRAG[dst_cast f16]     (= ds)
  FRAG[dst_cast] --> gemm --> FRAG[dk_frag(bm,D)]  +=                           (= ds^T Q)
  FRAG[dst_cast] --> SMEM[dst_shared] --> gemm --> FRAG[dq_frag(bn,D)]
  FRAG[dq_frag] --T.Parallel atomic--> GM[dq]     (跨 kv block 归约)   <-- 原子写
FRAG[dv_frag] --> SMEM[dv_shared] --atomic--> GM[dv]  (跨 group head 归约) <-- 原子写
FRAG[dk_frag] --> SMEM[dk_shared] --atomic--> GM[dk]  (跨 group head 归约) <-- 原子写
```

- 所有 `*_frag`（`qkt/dst/dq_frag/dk_frag/dv_frag`）是 **register fragment**（GPU 视角），`*_shared` 是 SMEM。
- **关键观察 1**：一个程序（逻辑核）处理 `(q_head, kv_block, batch)`，**内层循环扫过全部 q block**；`dk/dv` 的 kv 维**在核内寄存器全量累加**，`dq` 的 kv 维**跨程序原子累加**（`dq` 位置只依赖 `(bx, q_idx)` 与本程序负责的 kv block `by`）。**GQA 的 `groups` 个 query head 各自算出 `dk/dv` 的同一元素**，靠 `atomic_add` 汇总（L613/L618）。
- **关键观察 2**：`T.annotate_layout`（L569–573）给 `dq` 施加 `make_dq_layout`（L46–50：按 8x8 gemm fragment 重排 `(d%2, 4*(length%8)+(d%8)//2)`），给 `dv_shared/dk_shared` 施加 `make_swizzled_layout` —— 前者是 **Ampere 上让 `atomicAdd` 向量化的布局技巧**（L47 注释原文 "atomicAdd cannot be vectorized on Ampere, need to reorder dq to match the 8x8 gemm fragment"），后者是 **SMEM bank conflict 规避**。二者均为 GPU 专属（→ §0.5 舍弃）。

**循环与并行结构**：grid = `(H, ceil(S/bm), B)` 三维；块内 q-block 循环 `T.Pipelined(loop_st, ceil(S/bn), num_stages)`；元素级并行 `T.Parallel(bm,bn)` / `T.Parallel(bn,D)`；`T.Pipelined` 提供软件流水（`num_stages` 默认 1）。

**host 侧逻辑**：`GQABwdKernel.forward`（L664–665）= `self.kernel(**self.config)(*inputs)`（`dq/dk/dv` in-place）；`default_config` = `{block_m:64, block_n: 64 if dim<=64 else 32, num_stages:1, threads:128}`（L645–652）；`autotune_configs` = `tile_stage_thread_configs()`（L654–656）。Op 侧编排（`gqa.py` L1376–1389）见 §0.1。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | fragment/SMEM 分级 tiling（L548–566 一整套 `alloc_shared`/`alloc_fragment`） | 数据驻留片上、减少 GM 往返 | SMEM 装 gemm 操作数、register fragment 装累加器 | 大寄存器堆 + SMEM（GPU） | **部分可移植**（NPU 有 UB/L1/L0C，容量不同需重算） |
| 2 | 三维 block 网格（L544–545） | 天然三维任务映射，免索引分解 | 硬件提供 3D grid | CUDA 3D grid | **硬件强相关**（NPU 一维 Kernel） |
| 3 | `T.Pipelined` 软件流水（L583） | q-block 循环与 DMA/gemm 重叠 | num_stages 双缓冲 | 通用（NPU 亦支持） | **可移植** |
| 4 | `T.gemm` Tensor Core matmul（L586/597/599/605/609） | 5 个矩阵乘走 Cube | Tensor Core / MMA | GPU Tensor Core | **等价可移植**（NPU Cube + `T.gemm`） |
| 5 | `T.GemmWarpPolicy.FullRow`（L586 等） | 指定 warp 级 M/N 划分 | warp-level fragment 分布 | **warp 概念（CUDA 专属）** | **硬件强相关 → 舍弃** |
| 6 | `T.atomic_add`（L611/613/618） | 跨程序/跨 head 的梯度归约（dq 跨 kv block；dk/dv 跨 group head） | 全局原子加 | 全局原子单元（NPU 亦有） | **能力可移植**（NPU 支持，见 `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.1 fp32 ✓）；**归约结构需重设计**（§0.6 R3） |
| 7 | `make_dq_layout` + `T.annotate_layout`（L569–571, L46–50） | 让 Ampere 的 `atomicAdd` 向量化（8x8 fragment 重排） | fragment 布局重排 | Ampere `atomicAdd` 吞吐限制 | **GPU 专属 → 舍弃** |
| 8 | `make_swizzled_layout`（L571–572） | 规避 SMEM bank conflict | swizzle 地址映射 | SMEM bank 结构 | **GPU 专属 → 舍弃** |
| 9 | `lse/delta` 落 shared + 列广播（L587/601 + L589/604） | 复用归约辅助量，避免重复读 GM | SMEM 广播 | 通用 | **可移植** |
| 10 | `sm_scale` 折进 `ds`（L604） | 让 `dk`/`dq` 两个 gemm **共享同一份**已缩放 ds | 代数结合（标量外提） | 无 | **可移植**（纯代数） |
| 11 | `exp2` + log2 域 lse（L522 `scale = sm_scale*LOG2E`） | 用 `exp2` 代替 `exp`，省一次底数换算 | 预先把 `LOG2E` 折进 scale 与 lse 域 | 无 | **可移植**（纯代数） |
| 12 | causal 循环下界剪枝 `loop_st`（L581） | 跳过对角以下整块恒零的 q block | 结构性删零项 | 无 | **可移植**（纯代数） |
| 13 | P/ds 显式降 fp16 再喂 Cube（L598/604->L607） | 半精度 Tensor Core 吞吐 + 减 SMEM 带宽 | 显式 cast | fp16 MMA | **等价可移植** |
| 14 | `compile_flags=["-O3","-DENABLE_BF16"]`（L532） | 编译优化 + 启用 bf16 | nvcc 标志 | CUDA 工具链 | **GPU 专属 → 舍弃** |
| 15 | `pass_configs={TL_ENABLE_FAST_MATH: True}`（L531） | CUDA fast-math | pass 配置 | CUDA 后端 | **GPU 专属 → 舍弃** |
| 16 | `threads=128`（L545） | CUDA block 线程数 | 线程块配置 | CUDA 执行模型 | **GPU 专属 → 舍弃** |
| 17 | `functools.lru_cache`（L514） | 缓存 kernel factory | host 侧 | 无 | **可移植（host 语义）** |

> 识别不出来 ≠ 不存在：以上 17 项为源码全部优化点（含 3 个 GPU 专属装饰/布局项与 3 个 CUDA 编译项）；未列出的优化不会在迁移中被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义（GQA 分组、ds 公式、fp32 累加、BSHD I/O、causal） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| `T.gemm` 五个矩阵乘（#4） | 算法 | Tensor Core | 有（Cube） | **等价替换** | `docs/Tilelang.language/线性代数操作/T.gemm.md`（Developer 模式 `T.gemm(src1,src2,dst,size=[],initC,a_transpose,b_transpose)`）；先例 `examples/indexer/indexer_bwd.py` L74/87/95 |
| `T.atomic_add` 归约（#6） | 算法 | 全局原子单元 | 有（fp32 ✓） | **能力保留 + 结构重新设计** | 能力：`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.1（fp32/fp16/bf16 ✓，降级为 `hivm::StoreOp`）；结构：→ §0.6 R3 |
| `T.Pipelined`（#3） | 优化 | 通用 | 有 | **保留** | 先例 `examples/deepseek_v32/sparse_mla_bwd.py` L199（`T.Pipelined(NS, num_stages=...)`） |
| causal 循环下界 `loop_st`（#12） | 优化 | 无 | 有 | **保留** | 纯代数删零；NPU `T.Pipelined(loop_st, n)` 允许运行期下界 |
| `exp2` + log2 域 scale（#11） | 优化 | 无 | 有（`T.vexp2`） | **保留** | `docs/Tilelang.language/数学操作/T.vexp2.md`；先例 sparse_mla_bwd L234 |
| `sm_scale` 折进 ds（#10） | 优化 | 无 | 有 | **保留** | 纯代数 |
| P/ds 显式 cast 喂 Cube（#13） | 优化 | fp16 MMA | 有 | **保留** | `T.vcast(..., round_mode="rint")`（前序函数教训：`T.copy` 隐式跨 dtype cast 不可控） |
| SMEM/register 分级 tiling（#1） | 优化 | SMEM + 大寄存器堆 | 有（UB/L1/L0C）但容量小 | **等价替换** | `T.alloc_shared`->UB、`T.gemm` 累加器 `T.alloc_fragment`->L0C；**tile 尺寸必须按 UB 192KB / L0C 32768 元素重算**（§4.5/§5.3） |
| 三维 block 网格（#2） | 优化 | CUDA 3D grid | **无** | **重新设计** | → §0.6 R1（一维 `T.Kernel` + block 索引静态分解） |
| `lse/delta` shared 广播（#9） | 优化 | SMEM 广播 | 有（UB + vbrc/reshape 广播） | **等价替换** | 载入 UB 后按列广播；先例 sparse_mla_bwd L195/206（`T.reshape` 广播）/ PL-1.36 |
| `T.GemmWarpPolicy.FullRow`（#5） | 优化 | warp 划分 | **无**（NPU 无 warp 概念） | **舍弃** | `T.gemm` 无 `policy` 参数（`T.gemm.md` §2.1 参数表）；NPU 由编译器决定切分 |
| `make_dq_layout` + `annotate_layout`（#7） | 优化 | Ampere `atomicAdd` 吞吐 | **无（且不需要）** | **舍弃** | 该布局的唯一目的是让 Ampere 的标量 `atomicAdd` 向量化；NPU `T.atomic_add` 有 `size=` 形态与 `hivm::StoreOp` 降级，无需 fragment 重排（`T.atomic_add.md` §2.4 示例即直用 2D `size`） |
| `make_swizzled_layout`（#8） | 优化 | SMEM bank 结构 | **无（且不需要）** | **舍弃** | NPU L1/UB 的 bank/对齐由编译器与 MTE 管理；先例 `_gqa_prefill_fwd_kernel` 等全部 NPU kernel 均无 `annotate_layout` |
| `threads=128`（#16） | 优化 | CUDA 执行模型 | 无 | **舍弃** | NPU 无显式线程配置（wrapper 已按 K9 从 config 移除） |
| `compile_flags` / `pass_configs`（#14/#15） | 优化 | CUDA 工具链 | 无 | **舍弃** | `-O3/-DENABLE_BF16`、`TL_ENABLE_FAST_MATH` 均为 CUDA 后端标志；`examples/` 全部 NPU kernel 均不携带 |
| `lru_cache`（#17） | 优化 | 无 | 有 | **保留** | host 语义，与 preprocess/postprocess 一致 |

**统计**：保留 7 项（语义 + #3/#10/#11/#12/#13/#17）、等价替换 3 项（#1/#4/#9）、重新设计 2 项（#2、#6 的归约结构；#6 同时「能力保留」）、舍弃 6 项（#5/#7/#8/#14/#15/#16；若把 #14–16 三个 CUDA 编译项合并计则为 4 组）。**无「静默丢弃」**：每个 GPU 专属项均在 §0.4/本表给出舍弃依据。

### 0.6 NPU 算法重设计

> §0.5 判定为「重新设计」的 2 类（R1 并行结构、R3 归约结构）+ tile 重算（R2）+ 布局注解舍弃（R4）。**无算法族替换**——§1.6.0 调研确认基线（单遍 tiled flash-bwd）即为最优候选族；重设计只做**并行/归约结构**与 **tile 形状**层面的落地，不改数学。

**重设计项 R1：并行结构（三维 Kernel -> 一维 Kernel + task 索引静态分解）**

- **源方案**：`T.Kernel(H, ceil(S/bm), B)` 三维 grid，逻辑核数 = `H * ceil(S/bm) * B`；意图「零索引开销的三维任务映射 + 充分并行」，且第 0 维取 **query head**（GQA 的 `groups` 个 head 各自独立算 `dk/dv` 后再原子汇总）。
- **NPU 新算法**：`T.Kernel(nblk, is_npu=True) as (cid, _)` 一维，`nblk = B * H_kv * ceil(S/BM)`；`cid` 静态分解为 `(bz, kh, jb)`（§3.3）。**关键结构改动**：第 0 维由 **query head** 改为 **KV head**，并把 GQA 的 `groups` 个 query head 变成**核内静态展开的循环**（`for g in range(groups)`，Python 层展开，编译期常量）——于是 `dk/dv` 的跨 head 归约**从原子加变为核内寄存器累加**（见 R3）。
- **语义保持论证**：任务划分只改变「谁算哪一块」，所有输出元素的求和集合不变（`dq` 仍对全部 kv 位置求和、`dk/dv` 仍对全部 query 位置与 group 内全部 head 求和）；`dk/dv` 每个元素恰被**唯一一个**逻辑核写一次（不再有跨核写），`dq` 的跨核归约语义与源一致（原子加，fp32 分组差异，§1.6.0 R3 说明）。GQA 分组公式 `kv_head = q_head // groups` 与源 L575 完全一致。**因果掩码在 NPU 侧仍严格按源 L590 的 `if is_causal:` 编译期守卫施加**：`is_causal=True` 时 `loop_st = kv0//BN` + 块内掩码；`is_causal=False` 时 `loop_st = 0` 且**不施加掩码**（全区间扫描），与源非因果路径逐元素一致（§3.3）。
- **意图承接**：源方案「零开销三维映射」->「少量 block 索引标量分解 + 一维 grid」；源 `dk/dv` 跨 head 原子汇总 -> 核内 head 循环（**附带 GQA 收益**：K/V 块每核只从 GM 读一次、被 `groups` 个 head 复用，源方案下同一 K/V 块被 `groups` 个程序各读一次）。

**重设计项 R2：tile 形状（GPU fragment 尺寸 -> UB/L0C 容量约束下的 tile）**

- **源方案**：`k_shared/v_shared [bm,D]`、`q_frag/do_shared [bn,D]`、`qkt/dst [bm,bn]` fp32、`dq/dk/dv_frag [*,D]` fp32（`bm=64`、`bn=64/32`、`threads=128` 下由 warp 分摊）。意图：一次搬入、片内完成。
- **NPU 新算法**：保留 `BM=64`、`BN= 64 (D<=64) / 32 (D=128)`（与源 `default_config` 一致），但显式按**两条硬上限**核算：
  - **L0C fp32 累加容量 = 32768 元素（128KB）**：同时驻留的累加器 `dk_f + dv_f = 2*BM*D`、`qkt + dp_f = 2*BM*BN`、`dq_f = BN*D`。D=128/BM=64/BN=32 -> `2*8192 + 2*2048 + 4096 = 24576 <= 32768` ✓（75%）；若取 BN=64 则 32768（100%，无余量）-> 选 BN=32。
  - **UB = 192KB**：`k_s/v_s/q_s/do_s/p16/ds16/qkt_ub/dp_ub/ds_ub/dq_ub` 合计 ≈ 104.25 KB（含 tmp，§4.5）✓。
- **语义保持论证**：形状重排不改变任何矩阵乘的收缩维与逐元素运算的数学定义；`kv0=jb*BM` 的切片与 `q0=i*BN` 的切片一一覆盖全部位置（`S % BM == 0`、`S % BN == 0`，§5.4 无尾块）。
- **意图承接**：保持「一次搬入、片内完成」的原意；把 GPU 大 fragment 换成按 NPU 容量核算的 UB/L0C 组合。

**重设计项 R3：归约结构（dq 原子写形态 + dk/dv 去原子）**

- **源方案**：
  - `dq`：`T.Parallel(bn, D)` 内**逐元素** `T.atomic_add(dq[bz,q,bx,j], dq_frag[i,j])`（L610–611），并靠 `make_dq_layout`（#7）让汇编层向量化——**逐元素原子 + GPU 专属布局**。
  - `dk/dv`：核内寄存器跨 q-block 累加（L599/L605）后，靠 `T.atomic_add` 做**跨 group head** 汇总（L613/L618）。
- **NPU 新算法**：
  - `dk/dv`：R1 的 head 循环使其**完全核内累加**，epilogue 直接从 L0C 经 UB 用 `T.copy` **写一次**（`dk[bz, kv0:kv0+BM, kh, 0:D]`，切片 strided 写），**零原子**。
  - `dq`：`dq` 是 BSHD 固定 head 的**跨步行区**（行内 D 连续、行间 stride `H*D`），因此用 `for r in T.serial(BN): T.atomic_add(dq[bz, q0+r, h, 0], dq_ub[r, 0], [D])` —— **按行对 D 长连续段整体原子加**，而非源方案的逐元素原子。形态与先例 `examples/indexer/indexer_bwd.py` L109–111 完全一致。
- **语义保持论证**：`dk/dv` 每个元素唯一写入 ⇒ 与源「寄存器累加 + 原子汇总」的**求和集合完全相同**，且因不再跨核，累加顺序变为**确定**（`g` 由小到大、`i` 由小到大）——这是**精度上更优**的确定性顺序（fp32 下与源的任意顺序差异属 §8.2 容差内，见 §1.6.1 验证）。`dq` 仍为跨核原子加，求和集合与源一致；原子加的先后不确定 ⇒ fp32 分组不确定（同源，§1.6.0 R3 已量化）。

**重设计项 R4：布局注解（`make_dq_layout` / `make_swizzled_layout` 舍弃）**

- **源方案**：`T.annotate_layout({dq: make_dq_layout(dq), dv_shared: make_swizzled_layout(...), dk_shared: ...})`（L569–573）。
- **NPU 新算法**：**不施加任何 `annotate_layout`**。
- **语义保持论证**：`T.Layout` 只影响片上数据的**排列方式**（GPU fragment 布局 / SMEM 寻址），不改变任何张量的数学内容与形状。NPU 无 warp-fragment 布局概念、SMEM bank 由编译器管理，且 `T.atomic_add` 已具备 `size=` 形态与 `hivm::StoreOp` 降级（`T.atomic_add.md` §2.4），故舍弃后语义与正确性不受影响。**依据**：`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.4（无需 fragment 重排即可 2D 原子加）；先例 `examples/indexer/indexer_bwd.py`（同形态 `T.atomic_add`，无 `annotate_layout`）。

**§0.4 优化意图承接核查**：#1->R2（UB/L0C 等价替换）；#2->R1（一维重设计）；#3->保留；#4->等价替换；#5->舍弃（无 warp）；#6->R3（能力保留、结构重设计）；#7->R4；#8->R4；#9->等价替换（UB + 列广播）；#10/#11/#12/#13->保留；#14/#15/#16->舍弃（CUDA 专属）；#17->保留。**无优化意图静默丢弃**。

### 0.7 标杆实现

- 源算子路径：`/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py`（`_gqa_bwd_kernel` L514–621 / `GQABwdKernel` L622–665）。
- 源仓参考实现（golden 依据）：`gqa_bwd.py` L521–522/L588/L603 的语义（§0.1 公式）；测试基准（NPU 仓）：`examples/TileOPs/tests/ops/test_grouped_query_attention_bwd.py` L121–122（`GroupedQueryAttentionBwdOp` + SDPA 参考，`atol=5e-3, rtol=1e-5`）。
- Op 侧整体编排（迁移约束上下文）：`/home/developer/TileOPs/tileops/ops/attention/gqa.py` L1330–1389。
- 迁移抽取副本：`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/_grouped_query_attention_bwd_kernels.py` L46–50（`make_dq_layout`）/ L85–218（`_gqa_bwd_kernel`）。
- NPU wrapper/集成目标（接口契约）：`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd.py` L121–218（`GQABwdKernel`：`supported_archs=None`、`default_config` 已去 `threads`、`forward(*inputs)` in-place 派发）。
- **golden 以 §0.1 语义为唯一依据**，不复刻 §0.6 的 NPU 算法。

---

## 1. 概述

### 1.1 算子名称

`_gqa_bwd_kernel`（GQA backward 主 kernel，NPU 侧实现文件名 `_gqa_bwd_kernel.py`）。

### 1.2 功能描述

FlashAttention 风格 GQA 反向主核：重算注意力概率 `P`，据 `do`/`lse`/`delta` 求出 query 梯度 `dq`、key 梯度 `dk`、value 梯度 `dv`。`groups = H/H_kv` 个 query head 共享同一 KV head。

### 1.3 数学公式

```
P[i,j]  = exp2( sum_d k[kv0+i] * q[q0+j] * scale - lse[q0+j] )    (causal 掩码置 0),  scale = D^(-1/2)*log2(e)
ds[i,j] = P[i,j] * ( sum_d v[kv0+i] * do[q0+j] - delta[q0+j] ) * D^(-1/2)
dv = sum_{h in group} P^T dO ,   dk = sum_{h in group} ds^T Q ,   dq = ds K
```

输入 fp16/bf16；`P/dp/ds` 与全部累加在 **fp32**；喂 Cube 的 P/ds 显式降 fp16；`dq/dk/dv` 以 fp32 写出。

### 1.4 算法描述

> 本节描述**迁移决策后（§0.6）的 NPU 侧算法**，与 §1.6.0 调研选定算法一致。

1. **任务映射**：输出侧按 `(batch b, kv_head kh, kv_block jb)` 展平为 `nblk = B * H_kv * ceil(S/BM)` 个逻辑块；每块负责 KV 区间 `[kv0, kv0+BM)`，把 `dk/dv` 的**全部求和**（含 group 内 `groups` 个 query head）在核内完成。
2. **常驻搬入**：读入 `k[b, kv0:kv0+BM, kh, :]`、`v[b, kv0:kv0+BM, kh, :]` 到 UB，全程序生命周期复用（GQA 复用：被 `groups` 个 head 共享）。
3. **head x q-block 扫描**：`for g in range(groups)`（静态展开）-> `for i in T.Pipelined(loop_st, S/BN)`：
   - 载入 `q/do` 的 `[BN,D]` 块、`lse/delta` 的 `[BN]` 段（均在当前 head `h = kh*groups + g`）。
   - `qkt = K Q^T`（Cube）-> 得 `[BM,BN]` fp32 分数。
   - `dp = V dO^T`（Cube）-> `[BM,BN]` fp32。
4. **概率与残差（向量化）**：`P = exp2(qkt*scale - lse)`（`T.vexp2`）；**因果掩码仅在 `is_causal=True` 时融合进同一逐元素 pass**（越界元素取 sentinel `NEG=-3.0e38` -> `exp2` 精确给 0，§1.6.1 #1；`is_causal=False` 时**不施加掩码**，直接 `P=exp2(qkt*scale-lse)`，与源 L590 守卫一致）；`ds = P odot (dp - delta) * sm_scale`；P/ds 显式 `vcast` 到 fp16。
5. **三个梯度 gemm**：`dv_f += P^T dO`、`dk_f += ds^T Q`（**跨 head 与 q-block 核内累加**）、`dq_partial = ds K`（每 q-block 重算，`initC=True`）。
6. **dq 归约（原子）**：`dq_partial` 落 UB 后按行 `T.atomic_add` 到 `dq[b, q0+r, h, :]`（跨 kv block 归约，与源语义一致）。
7. **epilogue**：`dk_f/dv_f` 经 UB 一次性 `T.copy` 写回 `dk/dv[b, kv0:kv0+BM, kh, :]`（**无原子**）。

**与源算法的结构差异**（来源编号）：三维 grid（首维 q_head）-> 一维 grid（首维 kv_head）（§0.6 R1）；GPU fragment -> UB/L0C 核算 tile（R2）；`dk/dv` 从「寄存器累加 + 跨 head 原子」->「核内全量累加 + 单次写」，`dq` 从「逐元素原子 + fragment 重排」->「按行段原子」（R3）；舍弃全部 `annotate_layout`（R4）。**数学语义与 fp32 精度路径不变**。

### 1.5 数据流图

```
GM[k (B,S,H_kv,D) f16] --slice T.copy--> UB[k_s [BM,D] f16] --+  (常驻，group 复用)
GM[v (B,S,H_kv,D) f16] --slice T.copy--> UB[v_s [BM,D] f16] --+
                                                              |
for g in range(groups):  for i in Pipelined(loop_st, S/BN):
  GM[q  (B,S,H,D)] --slice--> UB[q_s  [BN,D] f16] --gemm--> L0C[qkt_f [BM,BN] f32] --> UB[qkt_ub]
  GM[do]           --slice--> UB[do_s [BN,D] f16] --gemm--> L0C[dp_f  [BM,BN] f32] --> UB[dp_ub]
  GM[lse]/[delta] ----------> UB[lse_s/delta_s [BN] f32]
  UB[qkt_ub],UB[lse_s] --T.Parallel(exp2 + 掩码融合[仅 is_causal])--> UB[p_ub] --vcast--> UB[p16 f16]
  UB[p_ub],UB[dp_ub],UB[delta_s] --T.Parallel--> UB[ds_ub] --vcast--> UB[ds16 f16]
  UB[p16]  @ UB[do_s] --gemm--> L0C[dv_f [BM,D] f32]  +=             (= sum_h P^T dO)
  UB[ds16] @ UB[q_s]  --gemm--> L0C[dk_f [BM,D] f32]  +=             (= sum_h ds^T Q)
  UB[ds16] @ UB[k_s]  --gemm--> L0C[dq_f [BN,D] f32] --> UB[dq_ub]
  UB[dq_ub] --按行 T.serial + T.atomic_add(D 段)--> GM[dq (B,S,H,D) f32]   <-- 原子（跨 kv block）
L0C[dk_f/dv_f] --> UB[dk_ub/dv_ub] --slice T.copy--> GM[dk/dv (B,S,H_kv,D) f32]  <-- 无原子
```

（Cube 路径：GM->UB->L1->L0A/L0B->L0C；Vector 路径：UB。Developer 模式由编译器 `cv_split` 切分并自动插同步。）

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**同一数学语义（flash-attention backward：重算 softmax 后求 dq/dk/dv）的算法族**。基线 = §0.3 源算法。本算子为**多步融合 / 矩阵 + 规约类**，按 `algorithm-research.md` §3 做**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。

**R1 等价化简公式候选**（含基线）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法）**：单遍 tiled flash-bwd，grid `(q_head, kv_block, batch)`，`dq` 原子、`dk/dv` 寄存器 + 跨 head 原子 | §0.3 步骤 1–23 | — | — | ✓（基线） |
| 1 | **GQA head 合并 + grid 改 kv_head 维**（§0.6 R1）：核内展开 `groups` 个 head，K/V 块核内复用 | 同 #0 的公式，仅重排任务维与内层循环 | 数学等价（求和集合不变，K/V 与 head 无关） | 省 K/V 的 GM 读 `(groups-1)/groups`；`dk/dv` 跨 head 原子 -> 0；写入顺序确定 | ✓（**选定**，见 §0.6 R1/R3） |
| 2 | **两遍拆分（pass A: dk/dv；pass B: dq）**，两条 KV 归约各落在各自网格维 | 同 #0 公式，拆成两遍扫描 | 数学等价（部分和重组） | **零原子**；代价：`KQ^T`/`VdO^T` 重算 ⇒ FLOPs **1.4x**、Q/dO/K/V 再读一遍 | ✓（**备选**，实验裁决 §9.3） |
| 3 | **grid 改 q_head/q_block 维**（sparse_mla_bwd 形态）：`dq` 核内全量累加免原子，`dk/dv` 原子 | 同 #0 公式 | 数学等价 | `dk/dv` 原子量 ∝ `groups*(S/BN)`，比 #1 的 `dq` 原子量（∝ `S/BM`）差 `2*groups*BM/BN` 倍 | ✗（R3/R4 否决） |
| 4 | 代数展开重写 `ds` | `P odot (dp-d) * s = P odot dp * s - P odot d * s` | 数学等价 | FLOPs 不变（多一次逐元素乘减），无收益 | ✗（无收益） |
| 5 | 把 `sm_scale` 从 `ds` 外提到 gemm 输出 | `ds' = P odot (dp-d)`，`dk = s*(ds'^T Q)`，`dq = s*(ds' K)` | 数学等价 | FLOPs 不变；需两次额外逐元素缩放，改变 fp32 舍入点 | ✗（无收益 + 精度路径变差） |
| 6 | 由前向存 `P`（store-attention）避免重算 | 直接读回 `P` | 数学等价 | GM 存/读 `B*H*S*S` 元素 ⇒ 对 S=2048 是 **3.4e9 元素 = 6.9 GB(fp16)**，远超本算子全部流量（134 MB） | ✗（容量/流量否决） |
| 7 | causal 循环下界剪枝（#12） | 跳过对角以下整块 | 数学等价（被删项恒为 0） | FLOPs 减半（causal） | ✓（**采纳**，见 §1.6.1） |
| 8 | 掩码融合进 `exp2`（sentinel，仅 `is_causal=True`） | `P = exp2(where(mask, x, NEG))` 代替 `where(mask, exp2(x), 0)`（`NEG=-3.0e38`） | 精确等价（`exp2(-3.0e38)=0`） | 省一个 `[BM,BN]` 逐元素 pass | ✓（**采纳**，见 §1.6.1） |

**R2 在线算法**：**无**（结论有结构依据）。在线（online）变体的适用前提是「存在**跨 tile 的 running 统计量**且其估计值随扫描更新时需对已累积结果做**重缩放**」——典型是前向 flash-attention 的 online softmax（`m`/`l` 未定，需 rescale `acc_o`）。反向主核**不需要**：`lse`（log2 域行 logsumexp）与 `delta` 已由 preprocess **先行算好并作为输入**（`gqa_bwd.py` L537–538），因此 `P = exp2(qkt*scale - lse)` 中不含任何随扫描变化的 running 量，`ds` 与 `dv/dk/dq` 的累加是**无重缩放的纯求和**。对照 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-attention` 条：在线变体只在**输入缺少 lse 而必须与分数扫描同步估计归一化**时才有意义；本算子不属于该情形。**调研范围**：`ALG-attention` / `ALG-softmax` 条目 + 源算子输入契约（L537–538 证明 `lse/delta` 为输入）+ 结构分析。

**R3 复杂度对比**（四口径；基线在表；该算子**算力受限**（见下 roofline），故 FLOPs 与扫描遍数为主导，同时列访存）：

| 算法候选 | FLOPs（mul+add） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 跨核代价 / 原子量 |
|---------|-----------------|---------------|---------|-------------|------------------|
| **#0 基线（源结构）** | `10*B*H*S^2*D`（causal 减半） | 读 `B(3H+4H_kv)*S*D*e` + 写 `B(3H+4H_kv)*S*D*4` | 1 | 片上 fragment（GPU） | `dq` 原子 ≈ `B*H*S*D*(S/bm)` 次元素写；`dk/dv` 原子 ≈ `2*B*H_kv*S*D*groups` 次 |
| **#1 选定（§0.6）** | 同 #0 | 同 #0，**但 K/V 读量 / groups** | 1 | UB ≤104.25 KB + L0C ≤24576 元素（§4.5/§5.3） | `dq` 原子 ≈ `B*H*S*D*(S/BM)`；**`dk/dv` 原子 = 0** |
| #2 两遍 | **1.4x #0**（`KQ^T`、`VdO^T` 各多算一遍） | 同 #0 + Q/do/K/V 再读一遍（≈ +1.5x 读量） | **2** | 同 #1（每遍更小） | **原子 = 0** |
| #3 全核内（q 维网格） | 同 #0 | 同 #0 | 1 | 同 #1 | `dq` 原子 = 0；`dk/dv` 原子 ≈ `2*B*H_kv*S*D*groups*(S/BN)`（**比 #1 的 `dq` 原子量 `B*H*S*D*(S/BM)` 差 `2*BM/BN` 倍**：比值 = `2*B*H_kv*S*D*groups*(S/BN) ÷ [B*H*S*D*(S/BM)]` = `2*(H_kv/H)*groups*(BM/BN)` = `2*BM/BN`，BM=64/BN=32 ⇒ **4x**） |
| #6 store-attention | ≈ `5*B*H*S^2*D`（省 `KQ^T`/`VdO^T` 重算） | **+`2*B*H*S^2*e` 存取**（S=2048 时 +6.9 GB） | 1（IO 爆炸） | 需 GM 暂存 `P` | — |
| #7 causal 剪枝（并入 #1） | `5*B*H*S^2*D`（causal） | 同 #0 | 1 | 同 #1 | 同 #1 |

> 口径说明：FLOPs 只计 gemm 的 mul+add（`qkt`/`dp`/`dv`/`dk`/`dq` 各 `2*BM*BN*D` ⇒ `10*B*H*S^2*D`，causal 由 #7 减半），**不含** exp2/逐元素缩放与索引计算；访存量含输入读 + 输出写，**不含**原子流量与 TLB/指令开销；`e` = 输入元素字节（2）。**原子流量单列**：`dq` 原子元素写 ≈ `B*H*S*D*(S/BM)`，代表 shape `[2,2048,32,128]`/BM=64 时为 `16.8M * 32 = 5.37e8` 次 x4B ≈ **2.15 GB**（换算见 roofline 行）。

**R4 硬件亲和性评估**（逐候选对照）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 选定（基线 + GQA 合并）** | Cube `T.gemm` x5 + Vector 逐元素/广播，天然匹配 | UB ≤104.25 KB ≤192KB ✓；L0C 24576 ≤32768 元素 ✓ | D=64/128、BM=64、BN=32/64 均整除 S 且行宽 ≥128B ✓ | `nblk`、`groups`、`BM/BN` 全为编译期常量 ✓ | 单遍，`T.Pipelined` 可 overlap；`dq` 原子为唯一跨核写 | **选定** |
| #2 两遍 | 同 #1，但 **Cube 工作量 1.4x**、需两遍扫描 | 同 #1 | 同 #1 | 同 #1 | 无原子；但多一遍扫描与再读 | 备选（原子代价 vs 1.4x FLOPs 的裁决见 §9.3） |
| #3 全核内（q 维网格） | 同 #1 | 同 #1 | 同 #1 | 同 #1 | **`dk/dv` 原子量比 #1 差 `2*BM/BN` = 4x**（BM=64/BN=32），且 GQA 下同一 KV 元素被 group 内多 head 争抢 | 否决：原子争用与流量显著劣化（量化见 R3 表） |
| #4 代数展开 | Vector 匹配 | 同 #1 | — | — | FLOPs 不变、多一次逐元素乘减 | 否决：零收益 |
| #5 sm_scale 外提 | 同 #1 | 同 #1 | — | — | 额外逐元素 pass x2，且改变 fp32 舍入点 | 否决：负收益 |
| #6 store-attention | 省 2 个 gemm | **GM 需 `2*B*H*S^2*e`**（+6.9 GB） | — | — | GM 容量/带宽不可行 | 否决：流量爆炸（`S^2` 项），远超算子本体流量 134 MB |
| #7 causal 剪枝 | 同 #1 | 同 #1 | 同 #1 | `loop_st` 为运行期标量但仅作循环下界（非数据依赖分支） | 同 #1 | **采纳**（并入 #1；纯删零项） |
| #8 掩码融合 exp2 | 同 #1（Vector 少一个 pass） | 同 #1 | 同 #1 | 同 #1 | 同 #1 | **采纳** |

**调研结论**：**选定算法族 #1 =「单遍 tiled flash-bwd + GQA head 合并（grid 取 kv_head 维、group 核内展开）」**，并采纳 #7（causal 剪枝）、#8（掩码融合）两个代数等价改写。关键依据：R3 表显示 #1 与基线 FLOPs/流量同阶而**消去 `dk/dv` 的全部原子量、且 K/V 读量 / groups**；#2 以 `1.4x` FLOPs 换取零原子（作为备选交 §9.3 实验裁决）；#3 原子量恶化 `2*BM/BN` 倍（4x）被 R3/R4 否决；#4/#5 零收益或负收益；#6 流量爆炸。故对「等价化简公式」问题，**除 #7/#8 两个删零/融合改写外无更优替代**。与基线的结构差异见 §0.6 R1–R4。**调研范围**：`tilelang-op-design/references/algorithm-candidates.md`（`ALG-attention` / `ALG-softmax`，其中 `ALG-attention` 含反向注意力语义）+ `tilelang-op-optimize/references/pattern-library/{attention.md, cases.md}`（PL-1.9-twophase / PL-1.9-hardlimits / PL-1.7 / CASE-ref-flash-attn-npuir）+ `examples/` 同类（`sparse_mla_bwd.py`、`indexer_bwd.py`、`_gqa_prefill_fwd_kernel.py`、`flash_attn_npuir*.py`）+ 源码结构分析；**未使用互联网检索**（本地源已覆盖该语义的全部候选；该算子是标准 flash-attn 反向，无本地未覆盖的公开候选）。

**设计期 roofline 估算下界行**（协议 `_shared/standards/hardware-cost-model.md` §2；常数取 `pattern-library/constants.md`，含版本戳）：

| 项 | 代表 shape `[2,2048,32,128]` fp16（H_kv=8, causal） | 口径与常数 |
|----|---------------------------------------------------|-----------|
| 流量项 | 读 `B(3H+4H_kv)S*D*e = 134.2 MB` -> `134.2MB / 2.49TB/s ≈ 53.9 us` | 常数 `CONST-elu-910B3-bw-floor`（**同设备 Ascend910B3**，逐元素搬运地板聚合 R+W ≈ 2.49 TB/s，toolchain tilelang 0.1.2+75e21fe37 / CANN 9.0.0 / 2026-10-15）。**下限** |
| 计算项（保守估计，**非严格下界**） | `10*B*H*S^2*D/2 = 171.8 GFLOP` -> `/ 112 TFLOP/s ≈ 1.53 ms` | 常量 `CONST-cube-gemm-l0c-traffic-floor`（910_9362 实测 **≥112 TFLOP/s** 的 fp16 GEMM 吞吐；因该实测本身 MTE2-bound，112 是**可达吞吐下界 ⇒ 1.53 ms 是时间上偏估计**。**未取得 910B3 fp16 峰值实测常数** -> 列入 §9.4 待测） |
| 原子项（单列，§1.6.3 裁决用） | `dq` 原子元素写 `B*H*S*D*(S/BM) = 5.37e8` x4B = **2.15 GB** -> `/2.49TB/s ≈ 0.86 ms`（若与 Cube 不重叠则叠加） | 同上带宽常数；`CONST-cube-gemm-l0c-traffic-floor` 口径不含原子 |
| 容量项 | UB ≈ 104.25 KB < 192KB ✓；L0C 24576 ≤ 32768 元素 ✓ | `CONST-capacity-910B2C`（UB=192KB，9362/910B3 数值一致注记） |
| **估算下界** | **max(53.9 us, ≈1.53 ms) ≈ 1.5 ms -> 算力受限**（流量项有 28x 裕度）；原子项 0.86 ms 与计算项同阶，是否与 Cube 重叠决定 #1/#2 裁决（§9.3） | 取计算项为主导 |

> 未实测假设：① `CONST-cube-gemm-l0c-traffic-floor` 的 112 TFLOP/s 绑定 910_9362，与 910B3 同核数（20 AIC）但**未在 910B3 重验**（已标注）；② 原子项与 Cube 的重叠程度无实测常数 -> 交 §9.3 实验裁决。二者均不改变「算力受限」结论。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（#1 + 采纳 #7/#8）。逐项给出四要素。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | **causal 掩码融合进 exp2**（R1 候选 #8；**仅 `is_causal=True`**） | `P = exp2(x)`；随后 `P = where(mask, P, 0)`（两次 `[BM,BN]` 逐元素 pass，源码 L588–593，掩码在 `if is_causal:` 内） | `P = exp2(where(mask, x, NEG))`，`x = qkt*scale - lse`，`NEG=-3.0e38`（一次 pass） | **精确等价**：`mask=真` 时两式均取 `exp2(x)`；`mask=假` 时基线取 `0`，候选取 `exp2(-3.0e38)=+0.0`（IEEE-754：足够小的有限负实参在 fp32 下 `exp2` 精确下溢为 `+0`），且 sentinel 直接替换 `x`（不做 `NEG*scale`）故无溢出。`is_causal=False` 时**两式均不施加掩码**，候选退化为恒等式 `P=exp2(x)` | 每 tile 省 1 个 `[BM,BN]` 逐元素 pass + 1 次读改写；代表 shape 约省 `B*H*(S/BM)*(S/BN)*BM*BN = B*H*S^2` 次元素操作（causal 减半） |
| 2 | **causal q-block 区间裁剪**（R1 候选 #7，源码 L581 已有；**仅 `is_causal=True`**） | 对全部 `S/BN` 个 q block 扫描并在块内掩码 | 只扫描 `i in [floor(kv0/BN), S/BN)`；被跳过块的所有 `P` 恒为 0 | **精确等价**：`i < floor(kv0/BN)` 的块内所有位置满足 `kv0+ii > q0+jj`（因 `q0+BN-1 < kv0`），故 `P ≡ 0` ⇒ `dv/dk/dq` 的该块贡献**恒等于 0**，删除的是数学上的零项。`is_causal=False` 时 `loop_st=0`（不裁剪，扫描全区间），与源 `else 0` 分支一致 | causal 下扫描块数约减半 ⇒ FLOPs 由 `10*B*H*S^2*D` 降至 `5*B*H*S^2*D` |

**优化结论**：采纳 **2 项**（均为精确等价/删零，非容差近似）。**优化后公式**（供 §3.1）：

```
causal=True :  P[i,j] = exp2( if_then_else(kv0+i <= q0+j, qkt[i,j]*scale - lse[q0+j], NEG) )   # NEG = -3.0e38
causal=False:  P[i,j] = exp2( qkt[i,j]*scale - lse[q0+j] )                                     # 不加掩码（源 L590 守卫）
ds[i,j] = P[i,j] * (dp[i,j] - delta[q0+j]) * sm_scale
dv = sum_h P^T dO ,  dk = sum_h ds^T Q ,  dq = ds K
（causal=True 时扫描 i in [floor(kv0/BN), S/BN)；causal=False 时扫描 i in [0, S/BN)）
```

**无其它优化空间的依据**：① 该算子的逐元素链只有「`exp2` 一次 + `ds` 一次」，已各占一个 pass（`ds` 为 3 项乘减，无法再减：`P`、`(dp-delta)`、`sm_scale` 均是必需因子）；② 无除法/开方/超越函数可做降代价代换（`exp2` 已是 log2 域最廉价指数，源码 L522 已把 `LOG2E` 折进 scale/lse）；③ 公共子表达式已识别：`P` 同时服务 `dv` 与 `ds`、`ds` 同时服务 `dk` 与 `dq`、`sm_scale` 已外提到 ds 让两 gemm 共享（源码 L604 已有）；④ 访存已是下界（`q/do/k/v` 各读一次、`dq/dk/dv` 各写一次）；⑤ 归约无合并空间（`dq/dk/dv` 三个归约的求和轴互不相同）。故除上述 2 项外**无采纳项**。

**等价性机器验证（D-1，结果表）**：两项均为**精确等价**（#1 靠 IEEE-754 `exp2(-3.0e38)=+0`——实现实际使用的 sentinel 常数，见下；#2 靠被删项恒零），其唯一数值风险是 #2 在 fp32 下的**累加顺序/分组变化**（跳过恒零块后，非零块的累加顺序不变，故理论差异为 0）。脚本 `verify_equiv.py`（本算子目录，torch CPU）对**候选式（A：掩码融合；C：掩码融合 + 区间裁剪）vs 基线式（B：先 exp2 后掩码、全区间扫描）**在随机 + 角点张量上按 kernel 同口径（fp16 输入 -> fp32 计算 -> P/ds 落 fp16 -> fp32 gemm）做数值对照；判定量 = 候选 vs 基线最大相对差（须 ≤ §8.2 rtol 的 1/10），并以 fp64 参照给出信息量偏差。

> **sentinel 常数与实现一致（问题 5 修复）**：脚本候选式使用 `NEG = -3.0e38`（与 §3.2/§3.3 实现的 `NEG` 常量逐位一致），而非 `-inf`；同时脚本内 `assert NEG == -3.0e38`。数学上二者经 `exp2` 均精确给 `+0`，但机器验证现已覆盖实现实际使用的常数。

执行命令：`python3 verify_equiv.py`（exit 0 = 全部 `EQUIV_PASS`）。实测结果：

| dtype | case | A_vs_B（判定量） | C_vs_B（判定量） | base~fp64（信息量） | finite_ok | 结论 |
|-------|------|----------------|----------------|-------------------|-----------|------|
| float16 | random | 0.000e+00 | 0.000e+00 | 7.42e-05 | ✓ | `EQUIV_PASS` |
| float16 | wide_dynamic_range | 0.000e+00 | 0.000e+00 | 1.03e-03 | ✓ | `EQUIV_PASS` |
| float16 | all_zero | 0.000e+00 | 0.000e+00 | 0.00e+00 | ✓ | `EQUIV_PASS` |
| float16 | dtype_extremes | 0.000e+00 | 0.000e+00 | 3.52e-04 | ✓ | `EQUIV_PASS` |
| float16 | subnormal_neg_shift | 0.000e+00 | 0.000e+00 | 1.07e-04 | ✓ | `EQUIV_PASS` |
| bfloat16 | random | 0.000e+00 | 0.000e+00 | 4.29e-05 | ✓ | `EQUIV_PASS` |
| bfloat16 | wide_dynamic_range | 0.000e+00 | 0.000e+00 | 6.73e-04 | ✓ | `EQUIV_PASS` |
| bfloat16 | all_zero | 0.000e+00 | 0.000e+00 | 0.00e+00 | ✓ | `EQUIV_PASS` |
| bfloat16 | dtype_extremes | 0.000e+00 | 0.000e+00 | 0.00e+00 | ✓ | `EQUIV_PASS` |
| bfloat16 | subnormal_neg_shift | 0.000e+00 | 0.000e+00 | 2.18e-05 | ✓ | `EQUIV_PASS` |

**全部 10 个用例、两个采纳项均为 `EQUIV_PASS`，未通过项计数 = 0**（候选与基线的相对差**恒为 0**，与「精确等价」的解析结论一致；`base~fp64` 列为信息量，度量分块/量化相对 fp64 参照的归一化绝对残差）。角点覆盖：全零 / dtype 极值量级 / subnormal+负移 / 大动态范围 / 随机；两类 dtype 各一遍。**本表为 sentinel 常数 `NEG=-3.0e38`（与实现 §3.2/§3.3 一致，问题 5 修复后）重跑产出，与上一版（`-inf`）逐行一致**。采纳项成立。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 分数->概率 `P = exp2(qkt*scale - lse)`（+掩码融合） | `[BM,BN]` 逐元素 | `T.Parallel(BM,BN)`（编译器向量化）或 `T.vmul` + `T.vsub`(列广播 `T.vbrc`) + `T.vexp2` | 是 | — |
| 2 | 因果掩码 | `[BM,BN]` 逐元素选择 | 融合进 #1 的 `T.if_then_else`（同 pass） | 是 | — |
| 3 | 残差 `ds = P odot (dp-delta) * sm_scale` | `[BM,BN]` 逐元素 | `T.Parallel(BM,BN)` 或 `T.vsub` + `T.vmul` + 列广播 | 是 | — |
| 4 | P/ds 降 fp16 | 逐元素 cast | `T.vcast(p_ub, p16, round_mode="rint")` | 是 | — |
| 5 | `dq` 写回 `dq[b, q0+r, h, :]` | 逐元素 `atomic_add`（源 L610–611） | `T.serial(BN)` 行循环 + `T.atomic_add(dq[b,q0+r,h,0], dq_ub[r,0], [D])`（行内 D 段整体原子） | 是（**已向量化到「行段」粒度**） | — |
| 6 | `dq` 行循环本身（#5 的外层） | — | 无「一次覆盖整块」的向量形态 | 否 | **API 缺失（已核）**：`dq` 是 BSHD 固定 head 的**跨步行区**（行间 stride `H*D`），而 `T.atomic_add` 的 `size=` 形态覆盖的是**连续区域**（`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.4 示例为连续 2D tile；先例 `examples/indexer/indexer_bwd.py` L109–111 对同类跨步目标同样退化为 `T.serial` 行循环）。**残留标量：仅 BN 次行循环的循环控制**（原子本体已是 D 段整宽） |
| 7 | 逻辑任务索引分解 `(bz,kh,jb) <- cid` | block 级标量 div/mod | 无 | 否 | **block 级索引 / 任务映射**：每逻辑任务仅 2 次 div + 2 次 mod，发生在 block 入口（非逐元素热点），无逐元素等价向量 API（对象是标量任务号，非张量） |
| 8 | causal 循环下界 `loop_st = kv0 // BN` | block 级标量算 | 无 | 否 | **block 级循环界**（同一 block 内所有 lane 共享同一值），无逐元素语义；用于 `T.Pipelined` 下界（结构性，非逐元素计算） |
| 9 | `groups` head 展开 | 源为独立逻辑核（grid 维） | 编译期 Python `range(groups)` 展开 | 是（结构展开，非向量化） | — |
| 10 | epilogue `dk/dv` 写回 | 逐元素原子（源 L613/618） | `T.copy(dk_ub, dk[b, kv0:kv0+BM, kh, 0:D])`（切片形态，编译器 MTE 批量搬运） | 是 | — |

**向量化结论**：**逐元素计算全部向量化**（#1–#4、#10，替代 API 佐证：`T.Parallel` 见 `examples/indexer/indexer_bwd.py` L92–94 邻域；`T.vexp2(src,dst,tmp)` 见 `docs/Tilelang.language/数学操作/T.vexp2.md` 与 `examples/deepseek_v32/sparse_mla_bwd.py` L234；`T.vcast(round_mode=...)` 见 indexer_bwd L100；`T.vmul`/`T.vsub`/`T.vbrc` 见 `docs/Tilelang.language/数学操作/`）。**保留 3 处标量/循环**（#6/#7/#8），理由类别分别为 **API 缺失（跨步目标的原子 size 形态）+ 逐元素残留行循环**、**block 级索引 / 任务映射**、**block 级循环界**，逐项见上表——**无「实现简单」类理由**。其中 #6 是唯一真正的「逐元素原子」残留，已通过「行内 D 段整宽原子」把它压缩到 BN 次原子发射。

#### 1.6.3 向量化轴与数据布局决策（阻塞级）

> I/O layout 是契约（`q/do/dq` BSHD、`k/v/dk/dv` 的 KV 版 BSHD、`lse/delta` BHS），核内布局是自由变量。本算子 >= 2 个非 batch 维度（S、D），完成本节分析。

**轴质量评分**：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | D 轴为向量宽度整数倍；行宽 ≥128B | D=64/128：fp16 行宽 = D*2 = 128/256 B ✓；fp16 向量宽 x8、fp32 x4，D%8=0/D%4=0 ✓ |
| 尾 lane 浪费率 | `(ceil(D/vw)*vw - D)/D` | 0%（D=64/128 均为 8 的整数倍） |
| 收缩维形态 | `T.gemm` 的收缩维须有足够长度且对齐 | `KQ^T`/`VdO^T` 收缩 D（64/128 ≥32）✓；`P^T dO`/`ds^T Q`/`ds K` 收缩 BN（32/64 ≥16）✓ |
| repack 代价 | 改布局的重排代价 | 主选方案无需 repack |
| UB/L0C 容量影响 | 驻留缓冲尺寸 | 主选：UB ≈104.25 KB、L0C 24576 元素（§4.5/§5.3）✓ |
| 原子目标形态 | `dq` 的行是否连续 | `dq[b,q,h,:]` 行内 D 连续（可整段原子），行间跨步 -> 需行循环（§1.6.2 #6） |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 / 归约组织 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|-------------------|------------|--------------|---------|
| 1 | **I/O 原生 BSHD + KV-head 网格 + group 核内展开**（§0.6 R1/R3） | D（逐元素）；归约轴 = KV（`dk/dv`，核内寄存器）/ query（`dq`，跨核原子） | 无 | K/V 读量 / groups；`dk/dv` 零原子；`dq` 原子量 `B*H*S*D*(S/BM)`（2.15 GB） | **选定（主选）** |
| 2 | 两遍拆分（pass A：KV 网格，`dk/dv` 核内全量；pass B：q 网格，`dq` 核内全量） | 同 #1 | 无 | 零原子（省 2.15 GB）；代价 FLOPs x1.4 + Q/dO/K/V 再读一遍 | **备选（§9.3 实验裁决）** |
| 3 | q 维网格（`dq` 核内全量，`dk/dv` 原子） | 同 #1 | 无 | `dk/dv` 原子量比 #1 的 `dq` 原子量差 `2*BM/BN` 倍（= **4x** @BM=64/BN=32） | 否选：原子争用/流量量化劣化（§1.6.0 R3/R4） |
| 4 | `dq` 先写 GM workspace、再用第二 kernel 跨 kv-block 求和 | 同 #1 | 需 GM workspace `B*H*S*D*(S/BM)*4B`（=`dq` 的 32 倍 = 2.1 GB）+ 第二次读 | ❌ 否选：GM 容量/流量爆炸，等价于把原子流量变成两倍 GM 流量（无收益） |
| 5 | host 侧把 `dk/dv` 改成按 `(kv_head, q_head)` 拆分再 host 求和 | — | host permute + 额外 kernel | 否选：违反 host 约束（仅允许 stride/shape 元数据视图），且引入额外 GM 往返 |

**布局决策结论**：**选定方案 #1**（主选）——核内布局 = 原生 BSHD 的连续行块（`k_s/v_s/q_s/do_s` 的 D 最内连续），**向量化轴 = D**；归约组织 = `dk/dv` 走 KV-head 网格 + group 核内展开（**零原子**）、`dq` 走跨核原子（行内 D 段整宽）。**#2（两遍零原子）作为备选**，其取舍（省 2.15 GB 原子流量 vs FLOPs x1.4）依赖「原子流量能否与 Cube 重叠」这一**未知常数**，按 §9.3 实验裁决，不在纸面拍板。该结论同步落入 §3.3 伪代码与 §6.1 循环结构。

> 弃选论证证据：`T.atomic_add` 的 `size=` 语义与降级（`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.4 + §3「降级为 `hivm::StoreOp`」）——据此判定 #1 的 `dq` 原子可行、#4 无收益；跨步目标需行循环的实证见 `examples/indexer/indexer_bwd.py` L109–111；host permute 约束见 `.agents/skills/tilelang-op-design/references/ascend-constraints.md` §4（未凭空断言）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（用户指定；且与算子特征一致——Developer 模式支持 `T.gemm` 与 v 前缀/`T.Parallel` 混排，由编译器 `cv_split` 自动切分 CV 并插同步）。

### 2.2 选型理由

- 算子为 **MixCV**（同一 kernel 内既有 Cube 的 `T.gemm` x5，又有 Vector 的 `exp2`/逐元素乘减/`vcast`/广播）-> Developer 模式可写 `T.gemm` + `T.Parallel`/v-op 并自动切分，无需 Expert 的手动 L1/L0 分配与 `sync_block_set/wait`。
- 强先例：`examples/deepseek_v32/sparse_mla_bwd.py`（flash-attn 反向，Developer + `T.gemm` x5 + `vexp2` + `T.atomic_addx4`）与 `examples/indexer/indexer_bwd.py`（Developer + `T.gemm` + `T.atomic_add`）与本算子结构同族。
- 遵循 AGENTS.md「Developer-mode MixCV trigger rule」-> 参照 `tilelang-mixcv-skill`。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`->UB（`k_s/v_s/q_s/do_s/qkt_ub/dp_ub/ds_ub/p16/ds16/dq_ub/dk_ub/dv_ub`）；`T.alloc_fragment`->L0C（`qkt_f/dp_f/dq_f/dk_f/dv_f`） |
| 计算方式 | `T.gemm`（Cube）x5 + `T.Parallel`/v-op 逐元素（Vector） |
| 同步 | 自动同步（§7.1）；**无手动 flag**（Developer 不支持 Expert 的 `sync_block_set/wait`；跨核原子累加是 GM 语义，不需要核间 flag） |
| 流控 | `T.Pipelined`（q-block 扫描）+ Python `range(groups)` 静态展开；**无核内 persistent 任务循环**（遵守 `TRAP-DEVEMODE-PERSIST-GEMM`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（含掩码融合与区间裁剪）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(bz,kh,jb) = decode(cid)`；`kv0 = jb*BM` | block 索引静态分解 |
| 2 | `k_s <- k[bz, kv0:kv0+BM, kh, :]`；`v_s <- v[同切片]` | 常驻搬入（group 复用） |
| 3 | `qkt[ii,jj] = sum_d k_s[ii,d]*q_s[jj,d]` | `K Q^T`（Cube） |
| 4 | `dp[ii,jj] = sum_d v_s[ii,d]*do_s[jj,d]` | `V dO^T`（Cube） |
| 5 | `P[ii,jj] = exp2(where(kv0+ii <= q0+jj, qkt*scale - lse_s[jj], NEG))` **（仅 `is_causal=True`；`False` 时 `P=exp2(qkt*scale-lse_s)`，不加掩码）** | 概率 + 掩码融合（Vector，`NEG=-3.0e38`） |
| 6 | `ds[ii,jj] = P[ii,jj]*(dp[ii,jj] - delta_s[jj])*sm_scale` | 残差（Vector） |
| 7 | `dv_f += sum_{ii->d} P16[ii,jj]*do_s[jj,d]` | `P^T dO`（Cube，累加） |
| 8 | `dk_f += sum_{ii->d} ds16[ii,jj]*q_s[jj,d]` | `ds^T Q`（Cube，累加） |
| 9 | `dq_f[jj,d] = sum_{ii} ds16[ii,jj]*k_s[ii,d]` | `ds K`（Cube，每块重算） |
| 10 | `dq[bz, q0+r, h, :] += dq_ub[r, :]` | 跨 kv block 原子归约 |
| 11 | `dk[bz, kv0:kv0+BM, kh, :] <- dk_ub`；`dv[同] <- dv_ub` | 单次写回（无原子） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 任务索引分解 | 标量算术（`cid`、`//`、`%`） | `bz=cid//(H_kv*n_kv)`、`kh=(cid%(H_kv*n_kv))//n_kv`、`jb=cid%n_kv` | Developer |
| 2 | 常驻搬入 | `T.copy(k[bz, kv0:kv0+BM, kh, 0:dim], k_s)` | **slice 形态**（禁用 `T.load_nd2nz`，见 §9.2） | Developer |
| 3 | `K Q^T` | `T.gemm(k_s, q_s, qkt_f, b_transpose=True, initC=True)` | b_transpose（转置 B） | Developer |
| 4 | `V dO^T` | `T.gemm(v_s, do_s, dp_f, b_transpose=True, initC=True)` | 同上 | Developer |
| 5 | 概率 + 掩码 | **`if is_causal:`**（编译期 Python 守卫，源 L590）`T.Parallel(BM,BN)` 写参数 `qkt_ub[ii,jj] = T.if_then_else(cond, qkt_ub[ii,jj]*scale - lse_s[jj], NEG)`；**`else:`** `T.Parallel(BM,BN)` 写 `qkt_ub[ii,jj] = qkt_ub[ii,jj]*scale - lse_s[jj]`（无掩码）。随后 `T.vexp2(qkt_ub, qkt_ub, tmp)` 原地求 `P=exp2(x)` | `NEG = -3.0e38` sentinel；`tmp` 为 `[BM,BN]` fp32 UB 缓冲（§4.5） | Developer |
| 6 | 残差 | `T.Parallel(BM,BN)`：`ds_ub[ii,jj] = p_ub[ii,jj]*(dp_ub[ii,jj] - delta_s[jj])*sm_scale` | 列广播 | Developer |
| 7 | 降精度 | `T.vcast(p_ub, p16, round_mode="rint")`；`T.vcast(ds_ub, ds16, round_mode="rint")` | 显式 cast（不用隐式 `T.copy` 转换） | Developer |
| 8 | `P^T dO` | `T.gemm(p16, do_s, dv_f, initC=False)` | 累加进 L0C | Developer |
| 9 | `ds^T Q` | `T.gemm(ds16, q_s, dk_f, initC=False)` | 累加 | Developer |
| 10 | `ds K` | `T.gemm(ds16, k_s, dq_f, a_transpose=True, initC=True)` | a_transpose（转置 A） | Developer |
| 11 | `dq` 原子归约 | `for r in T.serial(BN): T.atomic_add(dq[bz, q0+r, h, 0], dq_ub[r, 0], [dim])` | 行内 D 段整宽；先例 `indexer_bwd.py` L109–111 | Developer |
| 12 | `dk/dv` 写回 | `T.copy(dk_ub, dk[bz, kv0:kv0+BM, kh, 0:dim])`（dv 同） | slice 形态 strided 写 | Developer |
| 13 | L0C->UB | `T.copy(qkt_f, qkt_ub)` 等 | 同 dtype 搬运 | Developer |
| 14 | q-block 循环 | `T.Pipelined(loop_st, n_q, num_stages=1)` | `loop_st` 运行期下界（causal） | Developer |
| 15 | group 展开 | Python `for g in range(groups)` | 编译期常量，trace 期展开 | — |

### 3.3 计算伪代码

> 注意：`T.Tensor` 的 dtype 直接作为第二个位置参数传入（不用 `dtype=` 关键字）；`dim` 为 Python 编译期常量。

```python
@tilelang.jit(target="npuir")
def _gqa_bwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype="float16"):
    accum_dtype = "float"
    LOG2E = 1.4426950408889634
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * LOG2E
    groups = heads // heads_kv              # 编译期常量；heads % heads_kv == 0（源 L523 校验）
    BM = 64                                 # KV 侧 tile 行数
    BN = 64 if dim <= 64 else 32            # Q 侧 tile 行数（源 default_config：L649）
    n_kv = seq_len // BM                    # 编译期整除（全部 workload 的 S 为 2 的幂）
    n_q  = seq_len // BN
    nblk = batch * heads_kv * n_kv
    NEG = -3.0e38                           # exp2 sentinel（越界 => 精确 0，§1.6.1 #1）

    @T.prim_func
    def _gqa_bwd_main(
        q: T.Tensor((batch, seq_len, heads, dim), dtype),
        k: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
        v: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
        do: T.Tensor((batch, seq_len, heads, dim), dtype),
        lse: T.Tensor((batch, heads, seq_len), accum_dtype),
        delta: T.Tensor((batch, heads, seq_len), accum_dtype),
        dq: T.Tensor((batch, seq_len, heads, dim), accum_dtype),
        dk: T.Tensor((batch, seq_len, heads_kv, dim), accum_dtype),
        dv: T.Tensor((batch, seq_len, heads_kv, dim), accum_dtype),
    ) -> None:
        with T.Kernel(nblk, is_npu=True) as (cid, _):
            bz = cid // (heads_kv * n_kv)
            rem = cid % (heads_kv * n_kv)
            kh = rem // n_kv                     # KV head（R1：网格首维由 q_head 改为 kv_head）
            jb = rem % n_kv
            kv0 = jb * BM

            k_s = T.alloc_shared([BM, dim], dtype)          # 常驻（group 复用）
            v_s = T.alloc_shared([BM, dim], dtype)
            q_s = T.alloc_shared([BN, dim], dtype)
            do_s = T.alloc_shared([BN, dim], dtype)
            lse_s = T.alloc_shared([BN], accum_dtype)
            delta_s = T.alloc_shared([BN], accum_dtype)
            qkt_ub = T.alloc_shared([BM, BN], accum_dtype)   # 分数 -> 原地复用为 P
            dp_ub = T.alloc_shared([BM, BN], accum_dtype)
            ds_ub = T.alloc_shared([BM, BN], accum_dtype)
            tmp = T.alloc_shared([BM, BN], accum_dtype)      # vexp2 的 tmpBuffer
            p16 = T.alloc_shared([BM, BN], dtype)
            ds16 = T.alloc_shared([BM, BN], dtype)
            dq_ub = T.alloc_shared([BN, dim], accum_dtype)
            dk_ub = T.alloc_shared([BM, dim], accum_dtype)   # epilogue
            dv_ub = T.alloc_shared([BM, dim], accum_dtype)

            qkt_f = T.alloc_fragment([BM, BN], accum_dtype)  # L0C
            dp_f = T.alloc_fragment([BM, BN], accum_dtype)
            dq_f = T.alloc_fragment([BN, dim], accum_dtype)
            dk_f = T.alloc_fragment([BM, dim], accum_dtype)
            dv_f = T.alloc_fragment([BM, dim], accum_dtype)

            T.copy(k[bz, kv0:kv0 + BM, kh, 0:dim], k_s)
            T.copy(v[bz, kv0:kv0 + BM, kh, 0:dim], v_s)
            T.clear(dk_f)
            T.clear(dv_f)

            loop_st = (kv0 // BN) if is_causal else 0        # causal 区间裁剪（§1.6.1 #2）

            for g in range(groups):                          # 静态展开（GQA head 合并，R1）
                h = kh * groups + g
                for i in T.Pipelined(loop_st, n_q, num_stages=1):
                    q0 = i * BN
                    T.copy(q[bz, q0:q0 + BN, h, 0:dim], q_s)
                    T.copy(do[bz, q0:q0 + BN, h, 0:dim], do_s)
                    T.copy(lse[bz, h, q0:q0 + BN], lse_s)
                    T.copy(delta[bz, h, q0:q0 + BN], delta_s)

                    T.gemm(k_s, q_s, qkt_f, b_transpose=True, initC=True)     # K Q^T
                    T.gemm(v_s, do_s, dp_f, b_transpose=True, initC=True)     # V dO^T
                    T.copy(qkt_f, qkt_ub)
                    T.copy(dp_f, dp_ub)

                    # P = exp2(x)；因果掩码仅在 is_causal 时施加（编译期守卫，源 L590）
                    if is_causal:
                        for ii, jj in T.Parallel(BM, BN):
                            qkt_ub[ii, jj] = T.if_then_else(
                                kv0 + ii <= q0 + jj,
                                qkt_ub[ii, jj] * scale - lse_s[jj],
                                NEG)                       # 越界 => sentinel
                    else:
                        for ii, jj in T.Parallel(BM, BN):
                            qkt_ub[ii, jj] = qkt_ub[ii, jj] * scale - lse_s[jj]
                    T.vexp2(qkt_ub, qkt_ub, tmp)               # P = exp2(x)
                    # ds = P odot (dp - delta) * sm_scale
                    for ii, jj in T.Parallel(BM, BN):
                        ds_ub[ii, jj] = qkt_ub[ii, jj] * (dp_ub[ii, jj] - delta_s[jj]) * sm_scale
                    T.vcast(qkt_ub, p16, round_mode="rint")
                    T.vcast(ds_ub, ds16, round_mode="rint")

                    T.gemm(p16, do_s, dv_f, initC=False)                      # dv += P^T dO
                    T.gemm(ds16, q_s, dk_f, initC=False)                      # dk += ds^T Q
                    T.gemm(ds16, k_s, dq_f, a_transpose=True, initC=True)     # dq_block = ds K
                    T.copy(dq_f, dq_ub)
                    for r in T.serial(BN):                                    # 跨 kv block 原子归约
                        T.atomic_add(dq[bz, q0 + r, h, 0], dq_ub[r, 0], [dim])

            T.copy(dk_f, dk_ub)
            T.copy(dv_f, dv_ub)
            T.copy(dk_ub, dk[bz, kv0:kv0 + BM, kh, 0:dim])       # 单次写（无原子）
            T.copy(dv_ub, dv[bz, kv0:kv0 + BM, kh, 0:dim])

    return _gqa_bwd_main
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(n, is_npu=True)` 一维 | `examples/indexer/indexer_bwd.py` L53；`examples/deepseek_v32/sparse_mla_bwd.py` L155 | 现有先例 |
| `T.alloc_shared` / `T.alloc_fragment` | `docs/Tilelang.language/内存操作/`；indexer_bwd L36–56 | 文档化 + 先例 |
| `T.gemm(..., b_transpose=, a_transpose=, initC=)` | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1（Developer 签名 `gemm(src1,src2,dst,size=[],initC=False,a_transpose=False,b_transpose=False)`） | 文档化 + 先例（sparse_mla_bwd L219–254） |
| `T.atomic_add(dst, src, size)` | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.4（fp32 ✓、降级 `hivm::StoreOp`）；indexer_bwd L109–111（1D 段） | 文档化 + 先例 |
| `T.vcast(src, dst, round_mode="rint")` | `docs/Tilelang.language/数据类型转换操作/`；indexer_bwd L100 | 文档化 + 先例 |
| `T.vexp2(src, dst, tmpBuffer)` | `docs/Tilelang.language/数学操作/T.vexp2.md`（签名 `vexp2(src, dst, tmp)`；**docs 无 `T.exp2.md`**）；`examples/deepseek_v32/sparse_mla_bwd.py` L234（`T.vexp2(acc_p, acc_p, tmp)`，同族 flash-attn 反向） | 文档化 + 先例。⚠ 知识库 `CG-2026-0018` / `VP-2026-0172`（**另一设备 Ascend910_9362**，2026-10-13）报告向量 `T.vexp2` 在 `(8,64)` fp32 tile 偶发 `inf`、而标量 `T.exp2` 可靠 → 列为 §9.4 待确认项，回退链见 §9.2 |
| `T.Parallel` 逐元素（含 `T.if_then_else`） | indexer_bwd L92–94 邻域；sparse_mla_bwd L214–220 | 现有先例 |
| `T.Pipelined(start, stop, num_stages=)` | `examples/deepseek_v32/sparse_mla_bwd.py` L199 | 现有先例 |
| `T.copy` slice 形态（BSHD strided tile） | `_sparse_mla_kernel.py` L221；sparse_mla_bwd L172–174 | 现有先例 |
| `T.serial`（行原子循环） | indexer_bwd L106 | 现有先例 |
| `T.vbrc` / `T.vmul` / `T.vsub`（列广播替代） | `docs/Tilelang.language/数学操作/`；PL-1.36 | 文档化 + 先例 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是（源 3D grid，L544）** | 一维 `T.Kernel(nblk)` + `(bz,kh,jb)` 静态分解（§0.6 R1） |
| GEMM 非整除 / L0C 溢出 | **是（D=128 时 BN 需收紧）** | `S % BM == S % BN == 0`（§5.4 无尾块）；`BN=32`(D=128) 使 L0C 24576 ≤ 32768（§5.3） |
| Developer + persistent + `T.serial` 多任务 + `T.gemm` 混排（`TRAP-DEVMODE-PERSIST-GEMM`） | **否（主动规避）** | 网格为**非持久**（一逻辑核一任务）；核内 `T.Pipelined` 是 q-block 扫描（非持久任务循环）；`T.serial` 仅用于 `dq` 原子行循环（不含 gemm） |
| 尾轴对齐（32B） | 否 | D=64/128 -> 行宽 128/256B ≥32B ✓ |
| strided tile 误用 `T.load_nd2nz`（`TRAP-C10-strided-gather` 同族） | **是（风险）** | 一律 slice 形态 `T.copy`（禁 `load_nd2nz`，§9.2） |
| 隐式跨 dtype `T.copy` 舍入不可控 | **是** | P/ds 用显式 `T.vcast(round_mode="rint")`（§3.2 步骤 7） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | `T.Kernel(heads, ceil(S/bm), batch)` 三维（L544） | 一维 `T.Kernel(batch*H_kv*n_kv)` + 索引分解 | §0.6 R1 |
| 网格首维 | **query head** | **KV head**（group 在核内展开） | §0.6 R1（GQA 合并） |
| `dk/dv` 归约 | 核内累加 + 跨 group head `atomic_add` | 核内全量累加 + 单次 `T.copy` 写 | §0.6 R3 |
| `dq` 归约 | `T.Parallel(bn,D)` 逐元素 `atomic_add` | `T.serial(BN)` 行循环 + D 段整宽 `atomic_add` | §0.6 R3 |
| 布局注解 | `make_dq_layout` + `make_swizzled_layout` | 无 `annotate_layout` | §0.6 R4（舍弃） |
| gemm policy | `policy=T.GemmWarpPolicy.FullRow` | 不传 policy | §0.5（舍弃 #5） |
| 编译配置 | `pass_configs={TL_ENABLE_FAST_MATH}` + `compile_flags=["-O3","-DENABLE_BF16"]` | 不传 | §0.5（舍弃 #14/#15） |
| 线程配置 | `threads=128` | 不传 | §0.5（舍弃 #16） |
| tile 尺寸 | `bm=64, bn=64/32`（GPU fragment 视角） | `BM=64, BN=64/32`（按 UB/L0C 核算） | §0.6 R2 |
| 梯度累加 dtype | fp32（`accum_dtype="float"`） | fp32（同） | 保留 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/deepseek_v32/sparse_mla_bwd.py` | **最高** | flash-attn 反向全流程：`T.gemm` x5（`b_transpose`/`a_transpose`）、`vexp2`、`ds = P odot (dp-Delta) * sm_scale`（L232–242）、`T.atomic_addx4` 原子归约、`T.Pipelined` 扫描 |
| `examples/indexer/indexer_bwd.py` | 高 | Developer 模式 MixCV 骨架；`T.gemm` + `T.alloc_fragment`/`T.alloc_shared`；`T.atomic_add(dst, ub_slice, [D])` 行段原子（L109–111）；`T.vcast` |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 中高 | GQA head 分组与 KV-head 视角的任务分解（前向）；NPU 上 `vcast`/`vbrc`/掩码 sentinel 的工程细节 |
| `examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py` | 中 | slice 形态 `T.copy` 读 BSHD strided tile（L221）；`T.reduce_sum` 用法 |
| `examples/grouped_query_attention_bwd/_flashattn_bwd_preprocess_kernel/DESIGN.md` | 中高 | 同任务前序函数的设计骨架、排坑结论（`size=` 写回、`gcd` 静态 tiling、guard 兼容性） |
| `examples/flash_attention/flash_attn_npuir_dev.py` | 中 | NPU flash-attention 的 Developer 模式组织与 `lse` 处理 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `q` | `(B,S,H,D)` | float16 / bfloat16 | BSHD，D 连续 |
| `k` | `(B,S,H_kv,D)` | 同 | BSHD（KV head 数） |
| `v` | `(B,S,H_kv,D)` | 同 | 同 |
| `do` | `(B,S,H,D)` | 同 | 同 `q` |
| `lse` | `(B,H,S)` | **float32** | log2 域行 logsumexp（上游产出） |
| `delta` | `(B,H,S)` | **float32** | `rowsum(o odot do)`（同任务 preprocess kernel 产出） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dq` | `(B,S,H,D)` | **float32** | 跨 kv block 原子累加；Op 侧后续 cast 回存储 dtype（走 postprocess kernel） |
| `dk` | `(B,S,H_kv,D)` | **float32** | 单次写；Op 侧 `.to(dtype)` |
| `dv` | `(B,S,H_kv,D)` | **float32** | 单次写；Op 侧 `.to(dtype)` |

> 三个输出由 Op 侧预先 `torch.zeros`（fp32），kernel in-place 写；无 `out_idx`（与 preprocess 不同）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `k_s` / `v_s` | `[BM, D]` | fp16/bf16 | UB | KV 块常驻（group 复用） |
| `q_s` / `do_s` | `[BN, D]` | fp16/bf16 | UB | Q/dO 块 |
| `lse_s` / `delta_s` | `[BN]` | fp32 | UB | 辅助量（列广播） |
| `qkt_ub` | `[BM, BN]` | fp32 | UB | 分数 -> 原地复用为 P |
| `dp_ub` | `[BM, BN]` | fp32 | UB | `V dO^T` |
| `ds_ub` | `[BM, BN]` | fp32 | UB | ds |
| `p16` / `ds16` | `[BM, BN]` | fp16/bf16 | UB | Cube 操作数 |
| `dq_ub` | `[BN, D]` | fp32 | UB | dq 块（原子源） |
| `dk_ub` / `dv_ub` | `[BM, D]` | fp32 | UB | epilogue 写回（可复用前者空间） |
| `qkt_f` / `dp_f` | `[BM, BN]` | fp32 | **L0C** | Cube 输出 |
| `dq_f` | `[BN, D]` | fp32 | **L0C** | dq 块 |
| `dk_f` / `dv_f` | `[BM, D]` | fp32 | **L0C** | 核内累加器（跨整个 group x q-block） |

### 4.4 内存搬运路径

```
GM[q/k/v/do/lse/delta] --slice T.copy--> UB(k_s,v_s,q_s,do_s,lse_s,delta_s)
UB 操作数 --> L1(cbuf) --> L0A/L0B --Cube T.gemm--> L0C(qkt_f,dp_f,dq_f,dk_f,dv_f)
L0C --T.copy--> UB(qkt_ub,dp_ub,ds_ub,...) --Vector(T.Parallel/v-op)--> UB
UB(dq_ub) --T.serial 行循环 + T.atomic_add--> GM[dq]        (原子)
UB(dk_ub/dv_ub) --slice T.copy--> GM[dk/dv]                  (普通写)
```

Developer 模式由编译器自动完成 `GM->UB->L1->L0` 的搬运与 `L0C->UB` 回读，并插同步。

### 4.5 UB / L0C 内存预算

**主选 D=128, BM=64, BN=32**：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `k_s` | (64,128) | fp16 | 16384 |
| `v_s` | (64,128) | fp16 | 16384 |
| `q_s` | (32,128) | fp16 | 8192 |
| `do_s` | (32,128) | fp16 | 8192 |
| `lse_s` / `delta_s` | (32,) x2 | fp32 | 256 |
| `qkt_ub` | (64,32) | fp32 | 8192 |
| `dp_ub` | (64,32) | fp32 | 8192 |
| `ds_ub` | (64,32) | fp32 | 8192 |
| `tmp`（`T.vexp2` 的 tmpBuffer） | (64,32) | fp32 | 8192 |
| `p16` / `ds16` | (64,32) x2 | fp16 | 8192 |
| `dq_ub` | (32,128) | fp32 | 16384 |
| **小计（循环内常驻，含 tmp）** | | | **106752 ≈ 104.25 KB** |
| `dk_ub` + `dv_ub`（epilogue，可复用循环内缓冲） | (64,128) x2 | fp32 | 65536（复用则不额外计） |
| **峰值** | | | **≈ 104.25 KB / 192 KB** |

> 表内各行加和 = `16384+16384+8192+8192+256+8192+8192+8192+8192+8192+16384 = 106752 B ≈ 104.25 KB`（不含 `tmp` 时为 **98560 B ≈ 96.25 KB**）；`tmp` 仅存活于 `T.vexp2` 一步，其生命期与 `ds_ub`（vexp2 之后才写）不重叠，BishengIR 生命周期复用可使其复用 `ds_ub` 空间，故实际峰值 ≤ 本表口径（本表按「全部常驻」保守求和，与 v0 的 108544 声明无关，v0 该数字系笔误，已订正）。

- 余量 `192 - 104.25 = 87.75 KB`；按 `CONST-capacity-910B2C` 的 x1.12 上偏口径 -> **116.8 KB < 192 KB** ✓（该系数为 910B2C 结构实测，910B3 未重验，已保守采用）；即便按 x1.7 自动多缓冲口径 -> 177.2 KB < 192 KB ✓。
- **L0C（fp32 累加容量 32768 元素 / 128KB，`CONST-cube-gemm-l0c-traffic-floor`）**：`dk_f+dv_f = 2*64*128 = 16384`、`qkt_f+dp_f = 2*64*32 = 4096`、`dq_f = 32*128 = 4096` -> **24576 ≤ 32768** ✓（75%）。若取 BN=64 则为 32768（100%，无余量）-> 故 **D=128 时 BN=32**。
- **D=64, BM=64, BN=64**：UB `k_s/v_s` 8192x2 + `q_s/do_s` 8192x2 + `lse/delta` 512 + `qkt_ub/dp_ub/ds_ub/tmp` 4×16384 + `p16/ds16` 2×8192 + `dq_ub` 16384 = **131584 B ≈ 128.5 KB**；L0C `2*64*64+2*64*64+64*64 = 20480 ≤ 32768` ✓；x1.12 -> 143.9 KB < 192 KB ✓。

### 4.6 动态轴定义

**无动态轴**。`batch/heads/heads_kv/seq_len/dim/is_causal` 均为编译期常量（源 factory 签名 L514–521）；`groups`、`n_kv`、`n_q`、`nblk` 均在编译期求得；`loop_st` 依赖 `cid`，为 **block 级标量**（非张量动态轴）。

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir")
def _gqa_bwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype="float16"):
    ...
```

无 `out_idx`（与源一致，in-place 写 `dq/dk/dv`）；**无 `pass_configs` / `compile_flags` / `threads`**（源为 CUDA 专属，已舍弃，§0.5）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**MixCV（Cube + Vector）**。
**判定依据**：kernel 内同时含 Cube 的 `T.gemm`（5 处）与 Vector 的逐元素/广播/类型转换（`exp2`、乘减、`vcast`）-> 触发 AGENTS.md 的 Developer-mode MixCV 规则；数据流为 `GM->UB->L1->L0A/L0B->L0C->UB->GM`（+ 原子写）。

### 5.2 Block 划分

```python
groups = heads // heads_kv          # 编译期
BM = 64                             # KV 侧行块
BN = 64 if dim <= 64 else 32        # Q 侧行块（源 default_config）
n_kv = seq_len // BM                # KV 块数（编译期整除）
n_q  = seq_len // BN                # Q 块数
nblk = batch * heads_kv * n_kv      # 逻辑核数（一维 grid）
loop_st = kv0 // BN if is_causal    # 每核的 causal 起始 q block
```

- `BM=64` 选择理由：① L0C 三条累加器共存可行性（§4.5）；② 整除全部 workload 的 `S`（128/512/2048）；③ 与源 `default_config` 一致，行为可比。
- `BN=64 (D<=64) / 32 (D=128)`：由 L0C 容量（`dk_f+dv_f` 占大头）反推，见 §4.5。
- **量化影响（U1）**：`dq` 原子量 ∝ `S/BM`，`BM` 增大可线性减少原子次数；但 `BM` 受 L0C（`dk_f+dv_f = 2*BM*D`）约束，D=128 时 `BM ≤ 128`（且需为其他 fragment 留空间）-> 实测裁决见 §9.3。

### 5.3 约束分析

- **对齐约束**：D=64/128，行宽 128/256B ≥ 32B ✓；fp16 向量宽 x8（D%8=0）、fp32 x4（D%4=0）✓。
- **L0C 容量**：24576/32768 = 75%（D=128,BN=32）✓；D=64 时 20480/32768 = 62.5% ✓。
- **UB 容量**：104.25 KB（x1.12 -> 116.8 KB）< 192KB ✓。
- **L1（cbuf）容量**：由编译器按 `T.gemm` 操作数管理；`k_s/v_s/q_s/do_s` 均为 `[64,128]` 级 fp16 tile，L1 512KB 余量充足 ✓（`CONST-capacity-910B2C` L1 上限）。

### 5.4 注意事项（非整除处理）

- **编译期消解尾块**：manifest 全部 workload 的 `seq_len in {128, 512, 2048}`，`BM=64`、`BN in {32,64}` 均整除 `S` -> **无 padding、无动态边界、无尾块掩码**（`64 | S` 恒成立）。`loop_st = kv0 // BN` 亦为整除结果。
- **若遇非整除 S**（超出 manifest 的边界用例）：取 `BM = gcd(seq_len, 64)`、`BN = gcd(seq_len, 64 or 32)`（编译期静态），或对最后一次块做 slice 长度钳制 + sentinel 掩码（`T.min(BN, S-q0)` 形态，先例 `T.atomic_add.md` §2.4）。本设计主选编译期静态路径。
- `seq_len < BM` -> `n_kv=1`，KV 全覆盖；`nblk = B*H_kv`。

### 5.5 分核策略（物理核数适配）

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 `docs/开发指南.md` §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录；**本算子为 MixCV（Cube+Vector）-> 直接使用实查返回值（不翻倍）**。

**① 逻辑核数计算**

```
num_logical_kernels = batch * heads_kv * ceil(seq_len / BM) = batch * heads_kv * n_kv
```

逐 workload（`BM=64`，`is_causal=True`）：

| workload（q_shape / kv_shape） | H_kv | n_kv = S/64 | num_logical_kernels |
|---|---|---|---|
| `[1,128,8,64]` / `[1,128,2,64]`（smoke） | 2 | 2 | `1*2*2 = 4` |
| `[4,512,32,128]` / `[4,512,8,128]` | 8 | 8 | `4*8*8 = 256` |
| `[2,2048,32,128]` / `[2,2048,8,128]` | 8 | 32 | `2*8*32 = 512` |
| `[2,512,64,128]` / `[2,512,8,128]` | 8 | 8 | `2*8*8 = 128` |
| `[1,2048,64,128]` / `[1,2048,8,128]` | 8 | 32 | `1*8*32 = 256` |

**② 物理核数依据（实查）**

- 查询代码：`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()`
- **实际返回值：`20`**（本设计期实查，2026-10-15，输出 `aicore_num = 20`；设备 Ascend910B3）。
- 本算子为 **MixCV**（含 Cube `T.gemm`）-> 按标准 §1 要素②「Cube/混合算子直接使用返回值」-> **物理核数 = 20**。记 `N_PHYS = 20`。（**未**翻倍；纯 Vector 算子才翻倍为 40。）

**③ 规模判定与分核方案**

- **规模判定：极大规模**。最大 `num_logical_kernels = 512`（`[2,2048,32,128]`），为物理核数 20 的 **25.6 倍**；且**无法通过增大 `BM` 缩减到接近 20**（`BM` 受 L0C 硬上限约束，D=128 时 `BM≤128` -> 最多降到 `2*8*16=256`，仍为 12.8 倍）。
- **关键结构约束**：标准 §1 要素③ 的「核内 `T.serial` 串行 persistent」路径**在本算子不可用**——`TRAP-DEVMODE-PERSIST-GEMM`（Developer 模式下 persistent 网格 + 核内 `T.serial` 多任务 + `T.gemm` 混排会导致运行时崩溃）明确禁止。故按标准 §1 要素③「**无法对齐兜底**」处理。
- **分核方案（兜底路径）**：**固定以「一逻辑核一任务」的非持久一维 grid 下发 `nblk = num_logical_kernels`**，由硬件按 20 个物理核分波（wave）调度；**显式记录残留负载不均与临界路径惩罚**：
  - **波次不均**：512 / 20 = **25.6 波**，临界路径 26 波 vs 均值 25.6 -> **+1.6%**（512 非 20 的整数倍；`nblk` 由 shape 与 `BM` 决定，无法在不破坏 L0C 约束的前提下钉到 20 的倍数）。
  - **causal 三角负载不均（主导项）**：每核工作量 ∝ 其 q-block 扫描数 `n_q - loop_st`，随 `jb` 增大而增大（首块最少、末块最多）-> 若不干预，末块（最重）可能被最后调度而拖长临界路径。**缓解措施（主选）**：把 `cid` 的 `jb` 维**逆序映射**（`jb' = n_kv - 1 - jb`），使重负载块取得较小 `cid` 从而**优先调度**，显著压缩尾效应（该映射只改任务顺序、不改结果）。
  - 备选 `BM=128`（D=128 需 L0C 复核）可把最大逻辑核数降到 256（12.8 波），但会提高单核工作量与 L0C 压力 -> 列入 §9.3 实验裁决。
- 核内循环边界均为**编译期静态值**：`groups`（Python 展开）、`n_q`（编译期）、`T.serial(BN)`（编译期）；唯一运行期标量 `loop_st` 只作 `T.Pipelined` 的**下界**，不构成数据依赖的动态边界。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| GQA head（group 内） | 编译期静态展开 | Python `for g in range(groups)` | 编译期常量；展开后 `dk/dv` 累加变为核内确定性顺序（§0.6 R1/R3） |
| q-block 扫描 | 软件流水 | `T.Pipelined(loop_st, n_q, num_stages=1)` | 与 DMA 重叠；运行期下界支持 causal 裁剪（先例 sparse_mla_bwd L199） |
| 逐元素概率/残差 | 向量化 | `T.Parallel(BM, BN)`（等价 v-op 链） | 最内维 = BN（连续），§1.6.3 选定 |
| `dq` 原子行循环 | 任务内串行 | `T.serial(BN)` | 跨步 GM 目标的原子写需按行（§1.6.2 #6，API 缺失） |
| 逻辑任务索引分解 | block 级标量 | `//`、`%` | 非逐元素热点（§1.6.2 #7） |

> 无逐元素标量循环；保留的 3 处标量点（`dq` 行循环、索引分解、`loop_st`）均在 §1.6.2 有充分理由。
>
> **`T.Persistent` 不采用**：本算子是 Developer 模式 MixCV（含 `T.gemm`），`TRAP-DEVMODE-PERSIST-GEMM` 禁止 persistent 网格 + 核内 `T.serial` 多任务 + `T.gemm` 混排（运行时崩溃）；分核改用「非持久一维 grid + 多波次 + causal 逆序调度」兜底（理由与残留量化见 §5.5③）。

### 6.2 循环伪代码

见 §3.3（`_gqa_bwd_main`）。结构摘要：

```python
with T.Kernel(nblk, is_npu=True) as (cid, _):
    <alloc + 常驻 k_s/v_s + clear dk_f/dv_f>                 # §4.3
    for g in range(groups):                                   # 静态展开（GQA 合并）
        for i in T.Pipelined(loop_st, n_q, num_stages=1):     # q-block 流水扫描
            <载入 q_s/do_s/lse_s/delta_s>
            <2 gemm -> qkt_f/dp_f -> UB>
            if is_causal: <T.Parallel: 参数 = mask ? (x) : NEG>   # 掩码融合，§1.6.1 #1
            else:         <T.Parallel: 参数 = x>                  # 非因果：不加掩码
            <T.vexp2(qkt_ub, qkt_ub, tmp) -> P = exp2(x)>
            <T.Parallel: ds = P odot (dp - delta) * sm_scale>
            <vcast -> p16/ds16>
            <3 gemm -> dv_f/dk_f（累加）, dq_f>
            <T.copy dq_f->dq_ub; T.serial(BN): atomic_add->dq>  # 跨 kv block 归约
    <T.copy dk_f/dv_f -> UB -> slice T.copy -> GM dk/dv>          # 单次写，无原子
```

### 6.3 流水线优化

**主选 `num_stages=1`**（与源默认一致）：UB 常驻集 104.25 KB，`T.Pipelined` 的双缓冲只作用于 `q_s/do_s/lse_s/delta_s` 等**每迭代换新**的小缓冲（≈16.4KB/套），`num_stages=2` 需 +16.4KB ≈ 120.7KB（x1.12 -> 135.2 KB）仍在 192KB 内 -> **Stage 4 可试 `num_stages=2/3`**（§9.3）。`k_s/v_s` 为程序级常驻，不参与流水。

### 6.4 尾块处理

**无尾块**：`BM | S`、`BN | S`（manifest 全部 shape 成立，§5.4）；切片 `k[bz, kv0:kv0+BM, kh, :]`、`q[bz, q0:q0+BN, h, :]` 恒在界内；`nblk` 与物理核数的整除余数由硬件分波吸收（无「部分 tile」，§5.5）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式默认）。无手动 flag / 无 `sync_block_set/wait`（Developer 模式不暴露 Expert 的手动同步原语）。核间通信仅通过 GM 的 `T.atomic_add`（`dq`），原子语义本身保证读改写原子性，**无需跨核 flag 或 workspace 握手**。

### 7.2 同步点说明

- **核内**：`T.copy`(MTE2)/`T.gemm`(Cube)/`T.Parallel`(Vector)/`T.vcast`(Vector)/`T.copy`(MTE3) 之间的 RAW/WAR/WAW 依赖由编译器自动插同步（含 `T.Pipelined` 迭代间缓冲复用）。
- **核间**：仅 `dq` 的 `T.atomic_add`（GM 原子，`hivm::StoreOp`），不依赖任何跨核事件；`dk/dv` 无跨核写（每个元素唯一写者）。
- **跨引擎（Cube<->Vector，MixCV）**：由 Developer 模式的 `cv_split` 自动插入（对齐先例 `sparse_mla_bwd.py` / `indexer_bwd.py`，二者均无手动同步代码）。

### 7.3 pass_configs 配置

```python
pass_configs = {}      # 无
```

源 GPU 的 `pass_configs={TL_ENABLE_FAST_MATH: True}` 与 `compile_flags=["-O3","-DENABLE_BF16"]`（CUDA 编译标志）**不迁移**（§0.5 #14/#15）。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据，**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。优先采用源仓/NPU 仓参考路径：`torch.nn.functional.scaled_dot_product_attention` 的反向（manifest `ref_api`），并以**显式公式实现**交叉校验。注意 `delta` 来自 preprocess 语义（`rowsum(o odot do)`），故 golden 以 `o` 为显式入参独立算出 `delta`，**不复用本 kernel 的算法结构**。函数体内**直接出现 `torch.<fn>(...)` 调用**（满足 `S3-GOLDEN-TORCH` 口径）。

```python
def golden_gqa_bwd(q, k, v, do, o, lse, is_causal=True):
    """独立参考实现（fp32，逐 head 显式公式）。

    q/do: (B,S,H,D); k/v: (B,S,H_kv,D); o: (B,S,H,D); lse: (B,H,S) fp32。
    返回 (dq, dk, dv)（fp32）；dq 与 q 同 shape，dk/dv 与 k/v 同 shape。
    delta 按 preprocess 语义由 o⊙do 行和独立算出（不使用本 kernel 的任何结构）。
    """
    B, S, H, D = q.shape
    H_kv = k.shape[2]
    groups = H // H_kv
    sm_scale = (1.0 / D) ** 0.5
    scale = sm_scale * 1.4426950408889634
    qf = q.float(); kf = k.float(); vf = v.float(); dof = do.float(); of = o.float()
    delta = torch.sum(of * dof, dim=-1)                       # [B,S,H] fp32（preprocess 语义）
    dq = torch.zeros(B, S, H, D, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, S, H_kv, D, dtype=torch.float32, device=q.device)
    dv = torch.zeros(B, S, H_kv, D, dtype=torch.float32, device=q.device)
    tril = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
    for b in range(B):
        for h in range(H):
            kh = h // groups
            qk = torch.matmul(qf[b, :, h, :], kf[b, :, kh, :].transpose(0, 1)) * scale  # [S,S]
            p = torch.exp2(qk - lse[b, h, :].unsqueeze(0))                            # [S,S] (kv,q)
            if is_causal:
                p = torch.where(tril, p, torch.zeros_like(p))
            dp = torch.matmul(dof[b, :, h, :], vf[b, :, kh, :].transpose(0, 1))       # [S,S] (kv,q)
            ds = p * (dp - delta[b, :, h].unsqueeze(0)) * sm_scale
            dv[b, :, kh, :] += torch.matmul(p.transpose(0, 1), dof[b, :, h, :])
            dk[b, :, kh, :] += torch.matmul(ds.transpose(0, 1), qf[b, :, h, :])
            dq[b, :, h, :] += torch.matmul(ds, kf[b, :, kh, :])
    return dq, dk, dv


def golden_gqa_bwd_sdpa(q, k, v, do, o, lse, is_causal=True):
    """交叉校验路径：用 torch 的 SDPA 反向作为独立参照（manifest ref_api）。"""
    import torch.nn.functional as F
    qq = q.transpose(1, 2).float().detach().requires_grad_(True)   # -> (B,H,S,D)
    kk = k.transpose(1, 2).float().detach().requires_grad_(True)
    vv = v.transpose(1, 2).float().detach().requires_grad_(True)
    ref = F.scaled_dot_product_attention(qq, kk, vv, is_causal=is_causal, scale=(1.0 / q.shape[-1]) ** 0.5)
    grads = torch.autograd.grad(ref, (qq, kk, vv), do.transpose(1, 2).float())
    dq, dk, dv = (g.transpose(1, 2).contiguous() for g in grads)
    return dq, dk, dv
```

> 两条路径互证；主用显式公式实现（可逐 head 对拍、定位到具体 head/位置），SDPA 路径作为第三方参照。

### 8.2 精度标准

| dtype（输入） | 输出 | atol | rtol | 说明 |
|-------|------|------|------|------|
| float16 | float32 | 1e-3 | 1e-4 | `P/ds` 落 fp16 引入主要量化；gemm 累加 fp32 |
| bfloat16 | float32 | 5e-3 | 1e-3 | bf16 尾数 8 位，量化更粗 |

> 上游 op 端到端测试 `test_grouped_query_attention_bwd.py` L122 用 `atol=5e-3, rtol=1e-5`（对已 cast 回输入 dtype 的 `dq/dk/dv`）；本 kernel 单独验证先比 fp32 输出（上表），再由 postprocess/cast 后比端到端阈值。

**L0 门槛测试计划**：
1. **主用例**（覆盖全部 manifest workloads）：`[1,128,8,64]`(H_kv=2)、`[4,512,32,128]`、`[2,2048,32,128]`、`[2,512,64,128]`、`[1,2048,64,128]`（H_kv=8），fp16 与 bf16 各一遍，与 §8.1 golden 比对（`dq/dk/dv` 三方全 assert）。
2. **GQA 分组专项**：`groups=1`（MHA 退化）、`groups=2`、`groups=4`、`groups=8`（验证 head 合并与 `dk/dv` 核内求和）。
3. **边界用例**：`seq_len < BM`（如 64，`n_kv=1`）；`B=1,H_kv=1`；非 2 的幂 `seq_len`（验证 `gcd` 兜底路径）；`is_causal=False`（**非因果路径：`loop_st=0` 且不施加掩码**，必须与 golden 的 `is_causal=False` 分支逐元素对拍——迁移 smoke/packaging 用例 `(1,1024,8,4,64,False,fp16/bf16)` 即走该路径）。
4. **数值边界**：全零输入（`dq/dk/dv == 0`）；`lse` 取极值（掩码列 `P -> 0`）；`delta` 含 0/负值。
5. **原子归约专项**：`dq` 与「单核串行版」逐元素对拍，确认跨 kv block 原子累加无丢失/重复（可用 `n_kv=1` 与 `n_kv>1` 两配置对照）。
6. Layer 1（单 kernel L0 精度+契约）、Layer 2 由 `tilelang-op-develop` 承接；完整 L1/L2/Boundary 套件交下游 skill。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **L0C 硬上限 32768 元素**：D=128 时 `dk_f+dv_f` 即占 16384，`BN` 不能取 64（否则 100% 无余量）-> `BN=32`。
2. **UB 192KB**：常驻 104.25 KB（x1.12 -> 116.8 KB），余量 ~39%；`BM` 不可再大一档到 128 而不复核 L0C。
3. **一维 Kernel + 非持久网格**：源三维 grid 已展开；`TRAP-DEVMODE-PERSIST-GEMM` 禁止 persistent+gemm，故走多波次（25.6 波）+ causal 逆序调度缓解。
4. **原子归约量**：`dq` 原子元素写 5.37e8（2.15 GB，约 0.86 ms 若与 Cube 不重叠）-> 与计算项同阶，是 #1/#2 裁决的核心（§9.3）。
5. **物理核数 20 与逻辑核数非整数倍**（512/20 = 25.6，残留 +1.6%，§5.5）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C 溢出 | D=128 时取 `BN=64`（`dk_f+dv_f+qkt+dp_f+dq_f = 32768`） | 编译失败（`cc overflow ...`） | `BN=32`（本文档主选）；或 `BM=32,BN=64` 变体 |
| UB 溢出 | `BM=128`（D=128） | 编译失败（`ub overflow ...`） | `BM<=64`（本文档主选） |
| `T.copy` rank 不匹配 | `[BN,D]` fragment 直接 copy 到 `dq[bz, q0, h, 0]`（需 2D 目标） | 编译期 shape 报错 | 用 slice 形态 `T.copy(dq_ub, dq[bz, q0:q0+BN, h, 0:D])` 或 `size=` 形态；原子写用 §3.2 步骤 11 |
| 原子目标跨步被当成连续 | 误以为 `T.atomic_add(dq[bz,q0,h,0], dq_ub, [BN,D])` 可整块原子加 | **静默写错**（BSHD 行间 stride `H*D`） | 按 §3.2 步骤 11 逐行 `T.serial(BN)` + `size=[D]` |
| strided GM 读被误用 `T.load_nd2nz` | 误用 `T.load_nd2nz` 读 BSHD 固定 head tile | 静默读错（`_gqa_prefill_fwd_kernel` 实施注记 1） | 恒用 slice 形态 `T.copy`（§3.2 步骤 2） |
| `T.reduce_sum` 未初始化 | 若改用 reduce 且 `clear=False` | 静默数值错误（`T.reduce_sum.md` §2.3） | 本算子不用 `reduce_sum`；若引入恒传 `clear=True` |
| 隐式跨 dtype `T.copy` | 直接 `T.copy(qkt_ub(fp32), p16(fp16))` | 隐藏 VCastOp、舍入不可控 | 显式 `T.vcast(..., round_mode="rint")`（§3.2 步骤 7） |
| Developer + persistent + gemm 混排 | 为降低波次而引入核内持久任务循环 + `T.gemm` | **运行时崩溃**（`TRAP-DEVMODE-PERSIST-GEMM`） | 保持非持久一维 grid（§5.5/§0.6） |
| `T.if_then_else` 内运行期标量比较 | `T.Parallel` 内用 `cid` 派生标量做比较（`TRAP-if_then_else-runtime-scalar-cmp`） | 编译/语义问题 | 掩码比较对象全为编译期求值的索引表达式（`kv0+ii <= q0+jj` 中 `kv0/q0` 为 block 级 let-bound 标量、`ii/jj` 为 lane 索引），与 trap 形态不同；若 Stage 3 报错，改用「先写 sentinel 再条件覆盖」两段式 |
| `loopvariant v-op mask`（`TRAP-loopvariant-vop-mask`） | v-op 的 mask 依赖循环变量 | 语义错误 | 本设计 v-op mask 仅由 `T.Parallel` lane 索引与 block 常量构成，不依赖 `T.Pipelined` 迭代变量 |
| bf16 路径 gemm 操作数 dtype | 参考实现 bf16 的 Cube 输入需一致 | 编译/精度问题 | `p16/ds16` 的 dtype 随 `dtype` 参数取 bf16，与 `k_s/v_s/q_s/do_s` 一致；L0 测试覆盖 bf16 |
| 因果掩码无条件施加（非因果语义丢失） | 把掩码 pass 写在 `if is_causal:` 之外 | **`is_causal=False` 结果错误**（本应保留的 `kv>q` 贡献被错误置零） | 掩码 pass 与 `loop_st` 均包进**编译期 `if is_causal:` 守卫**：`True` 时 sentinel 掩码 + `loop_st=kv0//BN`；`False` 时**不加掩码** + `loop_st=0`（源 L590 语义，§3.3）；L0 必测 `is_causal=False` |
| 向量 `T.vexp2` 在 fp32 tile 偶发 `inf`（知识库 `CG-2026-0018` / `VP-2026-0172`，另一设备） | Developer 模式向量 `2^x` 路径 | P 出现 `inf` -> `dq/dk/dv` 全污染 | L0 首测即对 `P` 做 `isfinite` 断言；若命中，回退 **标量 `T.exp2` 链**或逐元素 `T.vmul(x, LN2)` + `T.vexp`（见 §9.4） |

### 9.3 特殊场景处理 / 待裁决项

- **主选 vs 备选实验裁决（`dq` 原子 vs 两遍零原子）**：主选 §3.3（一次扫描 + `dq` 原子，原子流量 2.15 GB）；备选 §1.6.0 #2（两遍拆分，零原子，FLOPs x1.4）。**裁决计划**：对 `[2,2048,32,128]` 与 `[1,2048,64,128]` 跑 msprof op 实测 `TaskDur` 并读取原子/MTE 时间列；**若备选 latency < 主选 x 0.95 则翻转**（备选代价明确，需显著优于主选才翻转）；裁决数据经附录补记回写本节。该裁决的核心未知常数 = 「原子流量与 Cube 的重叠率」（§1.6.0 roofline 未实测假设 ②）。
- **`BM` 取值的分核/原子权衡（U1）**：`BM=64`（主选）vs `BM=128`（D=128 需先复核 L0C：`dk_f+dv_f = 2*128*128 = 32768` 已占满，需降 `BN` 到 16 或改双累加器分时复用 -> 风险高）。**裁决计划**：仅当主选 profile 显示原子项主导时试 `BM=128` 变体；判定阈值同上。
- **`num_stages` 裁决**：主选 1，备选 2/3（UB 余量允许，§6.3）；`[2,2048,32,128]` 上实测，超过 3% 提升则采纳。
- **causal 逆序调度**：主选启用 `jb' = n_kv-1-jb`（§5.5）；备选不启用。实测 `TaskDur` 决定；预期收益在末端 kv 块重负载较大时明显。
- **极小 shape**：smoke `[1,128,8,64]` -> `nblk = 4`、`n_q = 2`、每核 `groups=4` 个 head、每 head 2 个 q block；无持久化开销，可直接跑。

### 9.4 待确认 / 未实测假设（举证责任登记）

| 项 | 状态 | 依据 / 估算 |
|----|------|------------|
| 910B3 fp16 Cube 峰值吞吐常数 | **未实测假设** | 借用 `CONST-cube-gemm-l0c-traffic-floor`（910_9362，112 TFLOP/s 可达下界）；910B3 未重验 -> roofline 计算项标为「上偏估计」 |
| 原子流量与 Cube 的重叠率 | **未实测假设** | 无 pattern-library 实测条目 -> §9.3 实验裁决 |
| `T.Pipelined` 支持运行期下界 `loop_st`（非 0 起始） | **待确认** | 源 GPU 支持（L583）；NPU 先例 sparse_mla_bwd 用 0 起始 -> 若 Stage 3 编译失败，回退「0 起始 + 全程掩码」（正确性不变，causal 下多算一倍） |
| `T.Parallel` 内直接读 L0C fragment（`dp_f[ii,jj]`） | **待确认** | 本设计已规避（先 `T.copy` 到 UB 再算），符合 indexer_bwd 的 UB-first 惯例 |
| bf16 路径的 Cube 操作数 dtype 组合 | **待确认** | `_gqa_prefill_fwd_kernel` 实施注记 2（fp16：f16/f16/f16；bf16：保守 f32/bf16/f32）-> Stage 3 按该注记试，L0 覆盖 |
| 向量 `T.vexp2` 可靠性（P=exp2(x)） | **待确认（KB 实测告警，跨设备）** | 知识库 `CG-2026-0018` / `VP-2026-0172`（toolchain tilelang 0.1.2 dev / CANN 9.0.0 / **Ascend910_9362**，2026-10-13）：向量 `T.vexp2` 在 `(8,64)` fp32 tile 6/512 元素产 `inf`（非输入溢出），标量 `T.exp2` 对照 `torch.pow` 1 ulp 内可靠；本设计为 910B3（未重验），且输入域 `x=qkt*scale-lse ≤ 0`（输出 ∈ (0,1]，与告警域 [-2,13.75] 不同）-> 列为 Stage 3 首测项；**回退链**：逐元素 `T.vmul(x, LN2)` + `T.vexp`，或标量 `T.exp2` 链（源算子原式） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_query_attention_bwd/_gqa_bwd_kernel/
├── DESIGN.md                        # 本设计文档
├── verify_equiv.py                  # §1.6.1 采纳项 #1/#2 的 D-1 机器验证（全 EQUIV_PASS）
├── _gqa_bwd_kernel.py               # 算子实现 + L0 测试（DEVELOP 阶段）
└── RETROSPECTIVE.md                 # Stage 1 复盘（自进化钩子）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成 | §1.6.1 采纳项 #1（掩码融合）/#2（区间裁剪）的 D-1 等价性机器验证脚本（实测全 `EQUIV_PASS`，exit 0） |
| `_gqa_bwd_kernel.py` | 待实现 | 算子实现（DEVELOP 阶段） |
| `RETROSPECTIVE.md` | 已追加 | Stage 1 复盘 |

### 10.3 命名规范

- 项目目录：`grouped_query_attention_bwd`（snake_case）；算子目录/实现文件：`_gqa_bwd_kernel`。

### 10.4 实现顺序

1. 设计文档（DESIGN.md）
2. Golden 函数（§8.1，两条互证路径）
3. 算子实现（`_gqa_bwd_kernel.py`，按 §3.3）+ L0 精度比对
4. 主选/备选（#1 vs #2）与 `BM/num_stages/causal 逆序` 的 A/B 裁决（§9.3，Stage 4）

---

## 11. 迁移专属：Hopper/CUDA 专属项逐项处置

> 调度 prompt 要求的逐项处置结论（与 §0.4/§0.5 一致）。

| Hopper/CUDA 专属项 | 源位置 | 处置 | 依据 |
|---------------------|--------|------|------|
| `T.atomic_add`（逐元素 + `T.Parallel(bn,D)`） | L610–611/613/618 | **能力保留 + 结构重新设计** | NPU `T.atomic_add` fp32 ✓（`T.atomic_add.md` §2.2.1）；`dq` 改为按行 D 段整宽原子（§0.6 R3）；`dk/dv` 改为核内累加 + 单次写（去原子） |
| `make_dq_layout`（+ `T.annotate_layout`） | L46–50 / L569–571 | **舍弃** | 唯一目的为 Ampere 标量 `atomicAdd` 向量化（L47 注释）；NPU 无需 fragment 重排（§0.6 R4） |
| `make_swizzled_layout`（+ `T.annotate_layout`） | L571–572 | **舍弃** | SMEM bank conflict 规避；NPU 无此暴露面，L1/UB 由编译器与 MTE 管理 |
| `compile_flags=["-O3","-DENABLE_BF16"]` | L532 | **舍弃** | CUDA/nvcc 后端标志 |
| `pass_configs={TL_ENABLE_FAST_MATH: True}` | L531 | **舍弃** | CUDA 后端 pass 配置；NPU 无对应语义 |
| `T.GemmWarpPolicy.FullRow`（**4 处 `policy=`**：L586/597/599/605；L609 的 `dq` gemm 无 `policy`） | L586/597/599/605 | **舍弃** | NPU 无 warp 概念；`T.gemm` 签名不含 `policy`（`T.gemm.md` §2.1） |
| `threads=128` | L545 | **舍弃** | CUDA 执行模型；wrapper 已按 K9 从 config 移除 |
| `transpose_A/transpose_B` 命名 | L586/609 等 | **等价替换** | NPU `T.gemm` 用 `a_transpose`/`b_transpose`（`T.gemm.md` §2.1） |
| `T.alloc_fragment` 大 tile（register 视角） | L548–566 | **等价替换** | -> L0C（`T.alloc_fragment`）+ UB（`T.alloc_shared`），尺寸按 L0C 32768 元素 / UB 192KB 重算（§0.6 R2/§4.5） |
| `T.Pipelined` | L583 | **保留** | NPU 支持（sparse_mla_bwd L199） |
| `functools.lru_cache` | L514 | **保留** | host 语义 |
| 三维 `T.Kernel` grid | L544–545 | **重新设计** | -> 一维 `T.Kernel(nblk)` + `(bz,kh,jb)` 静态分解（§0.6 R1） |

---

## 12. 附录：本次相对同任务前序函数的一致性

- 复用前序 preprocess/postprocess 的落盘约定、`verify_equiv.py` 的 D-1 结果表形态、`§5.5` 分核三要素实查口径（`NPUUtils.get().get_aicore_num()`）。
- 排坑结论沿用：slice 形态 `T.copy` 读 BSHD strided tile（禁 `T.load_nd2nz`）、编译期 `gcd` 消解非整除、`T.copy` 隐式跨 dtype cast 不可控须显式 `T.vcast`、Developer 模式禁 persistent+gemm 混排。
- 与前序函数的接口对齐：本 kernel 消费 preprocess 产出的 `delta[B,H,S] fp32`、`lse[B,H,S] fp32`，产出 `dq/dk/dv`（fp32，in-place），与 `grouped_query_attention_bwd.py` 的 `GQABwdKernel.forward(*inputs)` 契约一致（`dq` 由 postprocess 承接 cast）。

---

## 性能目标（Stage 4 / optimize 场景；由 conductor 追加）

- 性能目标类型：`best_effort`（用户未指定目标类型 / 数值 / baseline）
- 性能 workload 来源：`examples/TileOPs/benchmarks/ops/bench_grouped_query_attention_bwd.py::test_gqa_bwd_bench` 的 manifest 驱动参数化案例（完整集合由 optimizer 展开并分类，conductor 不写死 tune 用例与数量）。已知 workload：smoke `q[1,128,8,64]/kv[1,128,2,64]` f16 + `llama-3.1-8b-short`、`8b-long`、`70b-short`、`70b-long`（f16/bf16，causal）。
- 本目录含 **3 个 kernel**（`_flashattn_bwd_preprocess_kernel` / `_gqa_bwd_kernel` / `_flashattn_bwd_postprocess_kernel`），`_gqa_bwd_kernel` 为 **primary**（`kernel_opt_py_path` 对账对象）。`gate 4` 的 final-artifact 对账为单文件口径，**多 kernel 时须按 kernel 拆分 `perf_records`/`inventory` 并保留不可变审计副本**（Rope 任务已验证该做法）。
- 回归入口：`python {kernel_dir}/perf_opt/{func}.py --level all`（三个函数各自；L0/L1 失败阻塞）；采纳后从 `examples/TileOPs/` 运行 `python -m tileops.reporting.cli run --op GroupedQueryAttentionBwdOp --prof-mode msprof`
- 关注 shape：未指定（完整 workload 集合参与调优）

---

## 性能目标（Stage 4 / optimize 第 2 轮；由 conductor 追加）

- 性能目标类型：`best_effort`
- **背景**：第 1 轮已把 primary 从 140.7 ms 降到 39.9 ms（3.47–3.54×），但 ratio 仍只有 0.03–1.54%（计算受限 C）。第 1 轮 profile 给出的剩余瓶颈：
  - `dq` 行段原子写 ≈ **34%**（探针：禁用原子 5883 → 3876 µs，≈1.5×）；
  - `MTE2` ≈ **33%**；
  - 「每 program 融合 2 个 KV block 使 dq 原子减半」为**未验证假设**（第 1 轮预算未实现）；
  - **split-N（4 核 → 40 核）未实现**：需 GM workspace / 核间 flag，而当前 drop-in 契约签名仅 `(x, out)`、`out_idx=[1]`，无 workspace 参数（第 1 轮判定不可落地）。
- **本轮目标**：在不破坏 drop-in 契约的前提下，实现「融合 KV block / 减少 dq 原子」「改善 MTE2」等结构改进；若确需扩展契约（workspace），请给出**量化收益 + 改动面 + 可回退方案**，交由 conductor 决策，**不得**擅自改 wrapper/签名。
- 性能 workload：`benchmarks/ops/bench_grouped_query_attention_bwd.py::test_gqa_bwd_bench` 的 manifest 全量参数化案例（由 optimizer 展开）。性能目标 `best_effort`，关注 shape 未指定。
- 回归入口：`python {kernel_dir}/perf_opt/_gqa_bwd_kernel.py --level all`（L0/L1 阻塞）；采纳后 `--op GroupedQueryAttentionBwdOp` 的 msprof report。
- ⚠️ 正确性契约不变：`is_causal` 两分支、GQA head 合并（dk/dv 零原子）、dq 行段原子、`P/ds` 存储 dtype 折算（`CG-2026-0025` 已证 fp32 A 操作数会静默错）、不得改 tests/benchmarks/manifest/wrapper。

---

## 附录 A — Stage 4 `[DESIGN_LIMIT]` 逆向反馈补记（2026-10-31，路由 = 附录补记）

> 完整反馈见 `perf_opt/perf_feedback.md`；此处仅登记设计层结论，不改动本设计任何技术选型。

- **实测天花板**：本 kernel 经本轮 `dq` rank-2 区域原子 + `BN 32→64` 后，8 档 tune 几何均值 22966.3 → 8001.4 µs（−65.9%），官方 Ratio 1.4610% → **4.2375%**（官方 report 复测 4.2451%）。
- **40% 不可达**：msprof roofline 绑定 compute（`AI·bw = 2069 TOps/s ≫ computility 299.36 TOps/s`），要 40% 需 cube 占比 ≥53%，而当前仅 5.5%（剩余墙钟 70% 为 `aiv_scalar`）。
- **上界参照**：同仓同算子最优实现 `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §10.4 的 3-pass 单引擎相位拆分（Cube-only → Vector-only → Mix + GM workspace + `by0` 标量入参 + `T.pipe_barrier`）实测 8b-short 1016.8 µs ⇒ **Ratio ≈8.2%**，仍为目标 40% 的 1/4.9。
- **所需结构性前提**（超出 Stage 4 边界）：kernel 数量 1→3、新增 GM workspace 与标量 chunk 入参、跨 pass 交接；同时会改变官方单 op 采集口径（`--kernel-name` 单 op 捕获 vs 多 kernel 时延之和）。
- **对本设计假设的修正**：§4.5「`BN=64` 静态 UB 240 KB 不可行」经实测被推翻（分配器按生命周期复用，实测 155 648 B/block 可编译）；§0.6 R3「`dq` 逐行原子成本可接受」被实测否定（占 51% 墙钟）。
