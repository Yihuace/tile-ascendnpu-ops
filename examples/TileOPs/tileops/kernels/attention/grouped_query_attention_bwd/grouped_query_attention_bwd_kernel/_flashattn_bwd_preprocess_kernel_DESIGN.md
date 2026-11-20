# `_flashattn_bwd_preprocess_kernel` 算子设计文档

> 迁移任务（scenario=migration / migration_mode=harness）：将 GPU TileOPs `GroupedQueryAttentionBwdOp` 的第 1 个 TileLang 函数 `_flashattn_bwd_preprocess_kernel`
> （源：`/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py` L26–53，类 `FlashAttnBwdPreprocessKernel` L56–105）迁移到 `target="npuir"`。
> 编程模式：Developer（用户指定）。目标设备：**Ascend910B3**（实查 20 AI core / 40 Vector core，UB 192KB）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `tilelang-op-design` skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：FlashAttention 反向的预处理步骤，逐行计算输出梯度与注意力输出之间的点积之和

$$\text{delta}[b,h,s] \;=\; \sum_{d=0}^{D-1} o[b,s,h,d]\cdot do[b,s,h,d]$$

`delta` 是 softmax 分母（行和）对输出的梯度辅助量（`D_i = rowsum(o_i ⊙ do_i)`），供主反向 kernel 计算 `dp = (do @ vᵀ − delta)` 使用。

**I/O 契约（源码 L30/L37 逐行读出）**：
- 输入布局 **BSHD**（`shape = (batch, seq_len, heads, dim)`，L30），`o` / `do` 同 shape 同 dtype；`o[b,s,h,:]` 沿 D 连续。
- 输出 `delta: [batch, heads, seq_len]`（L37），**dtype = float32**（`accum_dtype = "float"`，L29）——输出布局是 **BHS**，与输入的 (s,h) 顺序相反（这是一次隐式转置）。
- `@tilelang.jit(out_idx=[2])`（L26）——kernel 自行分配并返回第 2 个参数 `delta`。

**规约语义**：沿最后一维 D 求和（`T.reduce_sum(acc, delta_frag, 1)`，L50），累加在 **fp32** 完成（`acc`/`delta_frag` 为 `accum_dtype="float"`，L42–43）。源码先在 fp32 `acc[i,j]` 中累乘（L48–49），最后沿 dim=1 归约 —— 乘法与累加均为 fp32。累加顺序：dim 维按 `k` chunk（`T.ceildiv(dim, blk)`，L45）分段，再在 reduce 内归约；映射到 NPU 后累加分组可变（fp32 下容差内等价，见 §0.6）。

**dtype 语义**：输入 fp16/bf16（`dtype` 参数），中间 `o⊙do` 与累加均 fp32，输出 fp32。由于 fp16×fp16 的乘积恰可用 fp32 精确表示（≤22 位有效位 < 24），**先提升到 fp32 再相乘与 golden 完全一致**。

**边界语义**：无 nan/inf 特殊处理（直接传播）；无空维度场景（D ≥ 64）；`o`/`do` 形状相同、无广播。

**host 侧语义**：无 host 预处理/后处理（Op 侧只做 `do = do.contiguous()`，`gqa.py` L1376）。输出 delta 被主反向 kernel 直接消费。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`(o.float()*do.float()).sum(-1)` + BSH→BHS 转置），**不复刻** §0.6 的 NPU 算法结构。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `o` | `(batch, seq_len, heads, dim)` = BSHD | `float16` \| `bfloat16` | 注意力前向输出 |
| `do` | `(batch, seq_len, heads, dim)` = BSHD | 同 `o` | 输出梯度 d(out) |
| `delta` | `(batch, heads, seq_len)` = **BHS** | **float32** | 输出（迁移必需字段；`out_idx=[2]`，源码 L37）。**布局与输入不同（s/h 交换）** |

> 输出 shape 依据：源码 L37 `delta: T.Tensor([batch, heads, seq_len], accum_dtype)`；op 契约见 `gqa.py` L1353/L1377（`prep_kernel(o, do)` → 传入主 kernel 的 `delta`，主 kernel 声明 `[batch, heads, seq_len]`，`_grouped_query_attention_bwd_kernels.py` L116）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码 L39–51，逐语句归档，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.Kernel(heads, ceildiv(seq_len,256), batch)`（L39）| — | 三维 block 网格 | 任务划分（非计算） |
| 2 | `T.copy(o[bz, by*blk:(by+1)*blk, bx, k*blk:(k+1)*blk], o_frag)`（L46）| `o` GM | `o_frag[256,256]` fp16 | 读取 o 的 (s,dim) tile（head=bx 固定） |
| 3 | `T.copy(do[...同切片...], do_frag)`（L47）| `do` GM | `do_frag[256,256]` fp16 | 读取 do 同位置 tile |
| 4 | `acc[i,j] += o_frag[i,j]*do_frag[i,j]`（L48–49，`T.Parallel(256,256)`）| `o_frag`,`do_frag` | `acc[256,256]` **fp32** | 逐元素乘积，按 dim chunk 累加 |
| 5 | `T.reduce_sum(acc, delta_frag, 1)`（L50）| `acc` | `delta_frag[256]` fp32 | 沿 D 求和 |
| 6 | `T.copy(delta_frag, delta[bz,bx,by*blk:...])`（L51）| `delta_frag` | `delta` GM (BHS) | 写回 |

**数据流与内存访问模式**（源硬件视角，GPU/CUDA）：

```
GM[o]  --T.copy--> fragment(o_frag)   ┐
GM[do] --T.copy--> fragment(do_frag)  ├─ T.Parallel(256,256) --乘法--> fp32 acc[256,256]
                                       ┘                                   │
                                                      T.reduce_sum(dim=1) ─┘
                                                              │
                                                    delta_frag[256] fp32 --T.copy--> GM[delta] (BHS)
```
- `o_frag`/`do_frag` 是 register fragment（GPU 视角）；`acc` 是 fp32 register 累加器。
- **关键观察**：`o_frag`/`do_frag`/`acc` 三个 fragment 各 `[256,256]`，`acc` fp32 即 **256KB**，三者合计 > 384KB —— 这是 **GPU register/SMEM 视角的产物**，直接搬到 NPU UB（192KB）**必然溢出**（→ §0.5 重新设计）。

**循环与并行结构**：grid = `(heads, ceil(seq_len/256), batch)` 三维；block 内 `k` 循环 `T.ceildiv(dim, blk)`（dim=64/128 时仅 1 次，只用 tile 的前 dim 列）；元素级并行 `T.Parallel(256,256)`；归约 `T.reduce_sum`。

**host 侧逻辑**：无。`FlashAttnBwdPreprocessKernel.forward`（L95–96）仅 `return self.kernel(o, do)`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | fragment/register tiling（`alloc_fragment [256,256]`）| 数据驻留寄存器，减少 GM 往返 | 一次性搬入大 tile 在片内完成乘加 | 大寄存器堆（GPU） | **硬件强相关**（NPU register 远小；UB 192KB） |
| 2 | 三维 block 网格（L39）| 天然二维/三维任务映射，免索引分解 | 硬件提供 3D grid 维度 | CUDA grid 三维 | **硬件强相关**（NPU 一维 Kernel） |
| 3 | 归约融合（`T.reduce_sum` 直接作用于 acc，L50）| 免中间落 GM | fragment 内归约 | warp shuffle / 片内归约 | **部分可移植**（NPU 有 `VReduceOp`） |
| 4 | dim 维 chunk 累加循环（L45）| 支持 dim > blk 的通用性 | 分段累加进 acc | — | **纯结构，可移植**（dim ≤ 128 时退化为单次） |
| 5 | 固定 `blk=256` 启动几何（L31）| 消除 autotune（类 docstring L58–62）| 编译期固定 tile | — | **可移植（模式）**，但 tile 取值受 UB 限制需重算 |
| 6 | coalesced 尾轴连续读（`o[b,s,h,:]` D 连续）| 访存带宽 | 尾轴 256B（D=128 fp16）连续段 | memory coalescing | **可移植**（NPU MTE 亦以连续段计） |
| 7 | 输出 `out_idx=[2]` 自动分配（L26）| 免去 host 侧 zeros 分配 | jit 装饰器 | — | **纯 host 语义，保留** |

> 识别不出来 ≠ 不存在：以上 7 项为源码全部优化点；未列出的优化不会在迁移中被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `delta = rowsum(o⊙do)`、fp32 输出、BHS 布局 | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| dim 维 chunk 累加（L45）| 算法 | 无 | — | **保留** | 纯数学分组；NPU 侧可退化为单遍（dim ≤ 128） |
| fp32 累加（`acc`/`delta_frag`）| 算法 | 无 | — | **保留** | NPU `T.reduce_sum` 支持 fp32（`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1） |
| fragment/register tiling（#1）| 优化 | GPU register 堆 | 有（UB） | **等价替换** | `T.alloc_fragment`/`T.alloc_shared` 映射 UB（192KB）；tile 尺寸须重算 |
| 三维 block 网格（#2）| 优化 | CUDA 3D grid | **无** | **重新设计** | → §0.6 R1（一维 `T.Kernel` + block 索引分解；本项目不支持三维 Kernel，见 `ascend-constraints.md`） |
| `[256,256]` 大 tile（#5）| 优化 | 大寄存器堆 | 有（UB）但容量小 | **重新设计** | → §0.6 R2（UB 192KB 硬上限，tile 改 `[R,D]` 行块） |
| 归约融合 `T.reduce_sum`（#3）| 优化 | warp shuffle | 有（`VReduceOp`） | **等价替换** | `T.reduce_sum(buf,out,dim=1,clear=True)`，先例 `examples/indexer/indexer_bwd.py` L94 / `_sparse_mla_kernel.py` L268 |
| coalesced 尾轴连续读（#6）| 优化 | memory coalescing | 有（MTE2 连续段） | **等价替换** | slice 形态 `T.copy`（行 256B 连续段 ≥128B，非 load_nd2nz） |
| `out_idx=[2]`（#7）| 优化 | 无 | 有 | **保留** | jit `out_idx` 装饰器语义 |

> 依据：映射表 migration-analysis.md §5.3（shared→UB、Tensor Core→Cube、warp shuffle→Vector 归约重新设计、三维 Kernel→一维重新设计）。**无「舍弃」项**（本算子未使用 swizzle / warp shuffle / cp.async 等 GPU 专属手法）。

### 0.6 NPU 算法重设计

> §0.5 判定为「重新设计」的 2 项（R1 并行结构、R2 tile 形状）+ 1 项源算法结构的直接承接（R3 dim-chunk 循环）如下。**无算法族替换**——Phase R（§1.6.0）调研确认基线（乘后归约）即为最优候选，故重设计只做**并行结构 / tile 形状**层面的落地，不改数学。源算法优化意图（#1–#7）承接见每项末。

**重设计项 R1：并行结构（三维 Kernel → 一维 Kernel + 持久化分核）**

- **源方案**：`T.Kernel(heads, ceildiv(seq_len,256), batch)` 三维 grid，硬件直接给出 `(bx,by,bz)`，逻辑核数 = `heads × ceil(S/256) × batch`（如 workload3 = `32×8×2 = 512`）。意图：零索引开销的三维任务映射 + 充分并行。
- **NPU 新算法**：`T.Kernel(nblk, is_npu=True) as (cid, _)` 一维；`nblk = 40`（物理 Vector 核数，§5.5）；核内 `T.serial(iters)`（`iters = ceildiv(total_tiles, 40)`，编译期静态）轮转处理逻辑行块任务；每个逻辑任务 `t = cid + it*40` 分解为 `(bz,bx,by)`（§3.3）。意图承接：把源方案「零开销三维映射」替换为「少量 block 索引标量分解 + 一维 grid」，并按 §5.5 摊薄核启动开销。
- **语义保持论证**：任务划分只影响「谁算哪一行」，不影响任何一行的计算结果（每行独立，无跨任务依赖）；`delta` 每行恰被一个逻辑任务写一次（`t < total_tiles` 守卫）。累加分组与源一致（每行独立求和），fp32 下无舍入差异。

**重设计项 R2：tile 形状（`[256,256]` fragment → `[R,D]` 行块）**

- **源方案**：`o_frag/do_frag/acc` 各 `[blk,blk]=[256,256]`（fp32 `acc` 单块即 256KB），blk=256 同时充当「seq 分块」与「dim 分块」两个量纲。意图：大 tile 一次搬入、片内完成。
- **NPU 新算法**：tile 改为 `[R, D]` 行块（R = 行数、D = head dim 全宽），`R = gcd(seq_len, 128)`（编译期静态；manifest 各 shape 均为 2 的幂 → R=128）。UB buffer：`o_s/do_s [R,D]` fp16 + `prod_s [R,D]` fp32 + `delta_s [R,1]` fp32。意图承接：保持「一次搬入、片内完成」的原始意图，同时把 fp32 中间量从 `[256,256]`（256KB）压到 `[R,D]`（D=128,R=128 时 64KB），**总 UB 128.5KB < 192KB**（§4.5）。
- **语义保持论证**：形状重排不改变逐元素乘与沿 D 求和；`o[b,a,bx,:]`（a∈[s0,s0+R)）的切片行恰对应 `delta[b,bx,a]`，一一映射无遗漏。边界：`R | seq_len` 由构造保证（`R = gcd(seq_len,128)`），**无尾块 / 无 padding**（§5.4）。nan/inf 传播与源一致（无特殊处理）。

**重设计项 R3：归约结构（dim-chunk 累加 → 单遍乘+归约）**

- **源方案**：`for k in range(ceildiv(dim,blk))`（L45）分块累加进 `acc`，再 `T.reduce_sum`。意图：支持 dim > blk 的通用 tile 尺寸（blk=256 时 dim ≤ 256 都走 1 次循环）。
- **NPU 新算法**：tile 已取全宽 D（D ≤ 128），chunk 循环恒为 1 次 → 直接 `T.Parallel(R,D)` 逐元素乘入 `prod_s`（fp32），再 `T.reduce_sum(prod_s, delta_s, dim=1, clear=True)` 单遍归约。意图承接：R2 的 tile 取全宽 D 后源 chunk 循环的「通用性意图」自然消失（不再是舍弃，是结构上无需），归约融合意图（#3）由 `VReduceOp` 承接。
- **语义保持论证**：`Σ_d` 分组由「(chunk 内累加) 再 (chunk 间归约)」变为「单次归约」，数学上同为 Σ_d；fp32 下不同分组仅引入 ≤1ulp 级差异（D=128 项，远小于 §8.2 容差）。`clear=True` 显式置位（规避 `T.reduce_sum` 未初始化静默错误，`docs/.../T.reduce_sum.md` §2.4 + VP-2026-0221 文档偏差提示）。

**§0.4 优化意图承接核查**：#1→R2（UB 等价替换）；#2→R1（一维重设计）；#3→R3（VReduceOp）；#4→R3（tile 取全宽后自然消解）；#5→R2（tile 重算）；#6→等价替换（slice copy 256B 连续段）；#7→保留。**无优化意图静默丢弃**。

### 0.7 标杆实现

- 源算子路径：`/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py`（`_flashattn_bwd_preprocess_kernel` L26–53 / `FlashAttnBwdPreprocessKernel` L56–105）。
- 源仓参考实现（golden 依据）：`gqa_bwd.py` L30/L37/L50 语义 —— `delta[b,h,s] = Σ_d o[b,s,h,d]·do[b,s,h,d]`，fp32。测试基准（NPU 仓）：`examples/TileOPs/tests/ops/test_grouped_query_attention_bwd.py`（SDPA 反向参考，间接覆盖本 kernel 的 `delta`）。
- 迁移抽取副本：`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/_grouped_query_attention_bwd_kernels.py` L54–81。
- NPU wrapper/集成目标：`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd.py` L71–118（`FlashAttnBwdPreprocessKernel.forward` 契约：`(o,do) -> delta[B,H,S] fp32`）。
- golden **以 §0.1 语义为唯一依据**，不复刻 §0.6 NPU 算法。

---

## 1. 概述

### 1.1 算子名称

`_flashattn_bwd_preprocess_kernel`（GQA backward preprocess，NPU 侧实现文件名 `_flashattn_bwd_preprocess_kernel.py`）。

### 1.2 功能描述

对注意力输出 `o` 与其梯度 `do` 逐行取点积，得到 fp32 辅助量 `delta`（softmax 分母梯度），供主反向 kernel 使用。

### 1.3 数学公式

$$
\text{delta}[b,h,s] = \sum_{d=0}^{D-1} o[b,s,h,d]\cdot do[b,s,h,d],\qquad b\in[0,B),\ h\in[0,H),\ s\in[0,S)
$$

输入 fp16/bf16，**乘法与累加均在 fp32**，输出 fp32。

### 1.4 算法描述

> 本节描述**迁移决策后（§0.6）的 NPU 侧算法**，与 §1.6.0 调研选定算法一致。

1. **任务映射**：把输出 `delta[B,H,S]` 按 `(b, h, s 行块)` 展平为 `total_tiles = B×H×ceil(S/R)` 个逻辑行块，每个行块 = 固定 `(b,h)` 下连续的 `R` 个 `s`（R = `gcd(S,128)`）。
2. **搬入**：对逻辑任务 `t=(b,h,s0)`，从 GM 读入 `o[b, s0:s0+R, h, :]` 与 `do[b, s0:s0+R, h, :]` 两个 `[R,D]` tile（slice 形态，行连续 256B）。
3. **逐元素乘**：`prod[i,j] = fp32(o[i,j]) · fp32(do[i,j])`（`T.Parallel`，向量化）。
4. **行归约**：`delta_s[i,0] = Σ_j prod[i,j]`（`T.reduce_sum(dim=1, clear=True)`，fp32）。
5. **搬出**：`delta[b, h, s0:s0+R] ← delta_s`（连续写）。
6. **持久化**：一维 grid 启动 40 个 block，各核 `T.serial` 轮转处理 26 个逻辑行块（§5.5）。

**与源算法的结构差异**（来源编号）：三维 grid → 一维持久化（§0.6 R1）；`[256,256]` fragment → `[R,D]` 行块（R2）；dim-chunk 循环 + 尾部归约 → 单遍乘 + 归约（R3）。**数学语义与 fp32 精度路径不变**。

### 1.5 数据流图

```
GM[o  (B,S,H,D) fp16] ──slice T.copy──> UB[o_s  [R,D] fp16]  ┐
                                                              ├─ T.Parallel: prod_s = f32(o_s)*f32(do_s)
GM[do (B,S,H,D) fp16] ──slice T.copy──> UB[do_s [R,D] fp16]  ┘        → UB[prod_s [R,D] fp32]
                                                                              │ T.reduce_sum(dim=1)
                                                                              ▼
                                                     UB[delta_s [R,1] fp32] ──T.copy──> GM[delta (B,H,S) fp32]
```
（纯 Vector 路径：无 L1/L0A/L0B/L0C 参与。）

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**同一数学语义（逐行 `Σ_d o·do` 点积）的算法族**。基线 = §0.3 源算法；本算子为**规约类**，按 `algorithm-research.md` §3 做**完整调研**（R1 候选表 + R3 对比表 + R4 逐候选评估）。

**R1 等价化简公式候选**（含基线）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法）**：逐元素乘 + 沿 D 归约 | `Σ_d o_d·do_d`（fp32 乘加） | — | — | ✅（基线） |
| 1 | 对角化 GEMM：`o[R,D] @ do[R,D]ᵀ` 取对角 | `diag(o·doᵀ)` | 数学等价（对角元 = 行点积） | FLOPs **R 倍放大**、需 Cube/L1/L0C 全链路 | ❌（R3 表已量化否决） |
| 2 | `‖o+do‖²/2 − ‖o‖²/2 − ‖do‖²/2`（极化恒等式） | 3 次平方和 | 数学等价 | FLOPs **3×**、访存不变 | ❌（更贵，且引入抵消误差） |
| 3 | 转置（d-major）后在 D 轴串行累加 | `Σ_d o_d·do_d`（lane=R） | 数学等价 | 无减少，且读取退化为跨步 | ❌（R4 否决） |
| 4 | 逐元素乘 + 单遍规约（= 基线的 chunk 循环消解版） | 同 #0，去 chunk 循环 | 数学等价（fp32 分组差异 ≤1ulp） | op/循环数减少（实现层） | ✅（= §0.6 R3，采纳，论证见 1.6.1） |

**R2 在线算法**：**无**（结论有结构依据）。在线（online）变体适用于「存在跨 tile 的 running 统计量（max/sum）且需要重缩放」的扫描类算子（如 online softmax / 前缀扫描）。本算子的每一输出行 `delta[b,h,s]` 仅依赖 O 与 do 中**同一行**的 D 个元素，行与行之间**无任何递归依赖**，是天然单遍、可完全并行的一次性归约（`o⊙do` 的 D 维求和）；不存在需要跨 tile 传递的 running 状态，故不存在「online 化」的语义空间。对照 `tilelang-op-design/references/algorithm-candidates.md`「规约类」条：仅当输出是**扫描/前缀**型（依赖前序结果）时才存在在线变体。

**R3 复杂度对比**（四口径；基线在表；带宽受限算子以访存/遍数为主导）：

| 算法候选 | FLOPs（mul+add） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------|---------------|---------|-------------|---------------------|
| **#0 基线（向量乘+归约）** | `2·N`，N=B·S·H·D | `读 2·N·e + 写 B·H·S·4`（e=elem bytes；无中间 GM 往返） | 1 | UB 级 `[R,D]` fp32（§4.5，≤64KB） | 行内全并行；行间独立，无跨核归并 |
| #1 对角 GEMM | `2·R·N`（R 倍） | `读 2·N·e + B·H·S·4`（含 L0C→GM 对角抽取） | 1（但 Cube 链路 L1→L0 多级） | L1 512KB + L0C 128KB 占用 | 需 Cube；L0C 仅存 `[R,R]` 对角抽取 |
| #2 极化恒等式 | `6·N`（3×） | `读 2·N·e + 写 B·H·S·4`（同基线） | 1 | 需 3 路平方和累加器 | 同基线，但 FLOPs 3× 且数值抵消误差 |
| #4 基线去 chunk 循环版 | `2·N`（同基线） | 同基线 | 1 | 同基线（更少 buffer） | 同基线 |

> 口径说明：FLOPs 只计 mul+add（不含 index 计算）；访存量含输入读 + 输出写，**不含** TLB/指令开销；三类候选的 GM 流量完全相同（`o`/`do` 各读一次、`delta` 写一次），差异只在 FLOPs 与片上结构。

**R4 硬件亲和性评估**（逐候选对照）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#0 基线** | Vector 逐元素乘 + `VReduceOp` 归约，天然匹配 | UB 128.5KB ≤ 192KB ✓ | D=64/128，fp16 行宽 128/256B（≥128B）；向量宽 fp32 ×4，D%4=0 ✓ | R=gcd(S,128) 静态 ✓ | 单遍，无中间 GM 往返，可直接与主反向 kernel 融合（本任务不做） | ✅ **选定** |
| #1 对角 GEMM | 需 Cube（本算子无 matmul 需求），且只取对角 → 算力浪费 R 倍 | 需 L1+L0C（512+128KB）额外占用 | 分形 M/N≥16 满足，但 R×R 输出仅对角有用 | — | 引入 CV 融合与跨引擎同步，复杂度剧增 | ❌ 否决：FLOPs×R、容量/同步代价，收益为负 |
| #2 极化恒等式 | Vector 匹配但 3 路平方和 | 3 个累加器 | 同基线 | 同基线 | 同基线 | ❌ 否决：FLOPs 3×、且 `‖o+do‖²` 大数相减引入灾难性抵消（fp32 下相对误差↑） |
| #3 转置串行累加 | lane=R、D 为串行累加维（跨步系数 = D ≠ 1）→ 阻碍向量化 | 需核内转置（T.transpose repack 代价） | D 连续但读写退化为跨步 | — | 无收益 | ❌ 否决：跨步访问 + repack 代价，**无任何流量/FLOP 收益**（依据 §1.6.3 候选矩阵） |

**调研结论**：**选定基线算法族 #0（逐元素乘 + 沿 D 单遍规约，采纳 #4 的 chunk 消解形式）**。关键依据：R3 表中三候选 GM 流量完全相同（该算子**天然带宽受限**，FLOPs 非瓶颈），而 #1 使 FLOPs ×R、#2 使 FLOPs ×3 且引入数值抵消，#3 无收益却引入跨步访问——故「无更优替代」。与基线的结构差异仅在于实现层：chunk 循环消解（§0.6 R3）、tile 改 `[R,D]` 行块（R2）、一维持久化 grid（R1）。**调研范围**：`algorithm-candidates.md` 规约类条目 + 恒等变形自查（极化恒等式/对角 GEMM）+ `pattern-library/attention.md`（PL-1.7/1.8/1.9/1.36/1.37）+ `examples/` 同类（`indexer_bwd.py`、`_sparse_mla_kernel.py`、`_gqa_prefill_fwd_kernel.py`）+ 结构分析；未使用互联网检索（本地源已覆盖该简单语义的全部候选）。

**设计期 roofline 估算下界行**（协议 `_shared/standards/hardware-cost-model.md` §2；常数取 `pattern-library/constants.md`，含版本戳）：

| 项 | 代表 shape `[2,2048,32,128]` fp16 | 口径与常数 |
|----|-----------------------------------|-----------|
| 流量项 | 读 `2·(2·2048·32·128)·2B = 67.1 MB` + 写 `2·32·2048·4B = 0.52 MB` = **67.6 MB** → `67.6MB / 2.49TB/s ≈ 27.2 µs` | 常数 `CONST-elu-910B3-bw-floor`（**同设备 Ascend910B3**，逐元素 copy 地板聚合 R+W ≈ 2.49 TB/s，toolchain tilelang 0.1.2+75e21fe37 / CANN 9.0.0 / 2026-10-15）。**下限** |
| 发射项 | 每 tile 约 3–4 个向量 op（vcast×2 + vmul + vreduce）× 1024 tile；按 `CONST-vector-op-cost-910_9362`「普通 v-op ≈3.4–7.8 µs / 16m-elem（bs7168）」外推 16.78M 元素 ×4 op ≈ **13 µs**（< 流量项） | 常数 `CONST-vector-op-cost-910_9362` 第二数据点（**跨 SoC 外推，已标注**：910_9362 ↔ 910B3 同为 20 AI/40 Vector 核但发射常数未在 910B3 重验） |
| 容量项 | UB 128.5 KB < 192 KB → 不构成约束 | `CONST-capacity-910B2C`（UB=192KB，9362/910B3 数值一致注记） |
| **估算下界** | **max(27.2, ~13, —) ≈ 27 µs**（小 tile 退化时段：按 `CONST-elu-910B3-bw-floor` bs 曲线 1.18–1.48 TB/s 折算上界 ≈ 46–57 µs） | 取流量项为主导 |

> 未实测假设：`CONST-vector-op-cost-910_9362` 的发射常数在 910B3 上未重验（标注「未实测外推」）；发射项占比 <50%，不改变下界结论。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线 #0 + chunk 消解 #4）。逐项给出四要素。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | dim-chunk 循环消解（R1 候选 #4） | `for k: acc[i,j] += o[i,j]·do[i,j]`（chunk 累加）后 `Σ` | `prod[i,j] = o[i,j]·do[i,j]`；`delta[i] = Σ_j prod[i,j]`（单次归约） | **数学恒等**：有限项求和的结合律重排，`Σ_d` 不变；fp32 下仅改变累加分组，D=128 项、每项 ≤2^24 量级，分组差异 ≤1 ulp（远小于 §8.2 容差） | 消除 chunk 循环控制与 `[R,D]` fp32 累加器的读改写；省 1 个 UB 级 buffer、每条 tile 省 `R·D` 次写回读 |

**优化结论**：采纳 **1 项**（chunk 循环消解，即 §0.6 R3；属实现层循环消除、数学恒等）。**优化后公式**（供 §3.1）：`delta[i] = Σ_{j=0}^{D-1} fp32(o[i,j])·fp32(do[i,j])`。

**无其它优化空间的依据**：该算子是「单次逐元素乘 + 单次归约」的最小结构——① 无除法/开方/超越函数可做降代价代换（R1 候选 #2 极化恒等式反而更贵且更不稳）；② 无公共子表达式（`o⊙do` 仅出现一次）；③ 两个输入各读一次、输出写一次，已是流量下界（§1.6.0 R3）；④ 无归约合并空间（只有一个归约）。故除 chunk 循环消解外**无采纳项**。

**等价性机器验证（D-1，结果表）**：采纳项 #1 为求和的结合律重排（数学恒等），其唯一数值风险是 **fp32 累加分组改变**。脚本 `verify_equiv.py`（本算子目录，torch CPU）对**候选式（单遍 fp32 顺序累加）vs 基线式（dim-chunk 分段 fp32 累加，chunk=64）**在随机 + 角点张量上做数值对照：判定量 = 两式相对差（须 ≤ §8.2 rtol 的 1/10，即 fp16 1e-5 / bf16 1e-4），并以 fp64 精确和为参照给出偏差；角点覆盖 0 / dtype 极值 / subnormal / 全零 / 大动态范围 / 正负抵消。

执行命令：`python3 verify_equiv.py`（exit 0 = 全部 EQUIV_PASS）。实测结果：

| dtype | case | cand_vs_base（判定量）| cand_vs_fp64 | base_vs_fp64 | finite_ok | 结论 |
|-------|------|----------------------|--------------|--------------|-----------|------|
| float16 | random_D128 | 3.13e-06 | 1.60e-06 | 1.86e-06 | ✓ | `EQUIV_PASS` |
| float16 | random_D64 | 1.78e-06 | 2.27e-06 | 2.27e-06 | ✓ | `EQUIV_PASS` |
| float16 | wide_dynamic_range | 1.55e-06 | 2.56e-06 | 1.32e-06 | ✓ | `EQUIV_PASS` |
| float16 | all_zero / dtype_max / subnormal / single_nonzero / cancellation | ≤1.79e-07 | ≤1.79e-07 | 0 | ✓ | `EQUIV_PASS` |
| bfloat16 | random_D128 | 5.78e-07 | 2.88e-06 | 2.88e-06 | ✓ | `EQUIV_PASS` |
| bfloat16 | random_D64 | 3.11e-05 | 1.55e-05 | 1.55e-05 | ✓ | `EQUIV_PASS`（band 1e-4）|
| bfloat16 | wide_dynamic_range | 1.48e-06 | 9.27e-07 | 5.56e-07 | ✓ | `EQUIV_PASS` |
| bfloat16 | all_zero / dtype_max / subnormal / single_nonzero / cancellation | ≤0 | ≤0 | 0 | ✓ | `EQUIV_PASS` |

**全部 16 个用例均为 `EQUIV_PASS`，未通过项计数为 0**（bf16 `random_D64` 的 3.1e-5 为 bf16 量化放大下的 fp32 累加噪声，两式对 fp64 的偏差同为 1.55e-5，仍低于 §8.2 bf16 rtol=1e-3 的 1/10）。采纳项 #1 成立。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素乘 `o⊙do` | 标量循环 / 逐元素赋值 | `T.Parallel(R, D)`（编译为 `vcast`+`vmul` 向量链）| ✅ | — |
| 2 | 行归约 `Σ_j prod[i,j]` | 标量累加循环 | `T.reduce_sum(prod_s, delta_s, dim=1, clear=True)` → `VReduceOp` | ✅ | — |
| 3 | 逻辑任务索引分解 `(bz,bx,by) ← t` | block 级标量乘除/取模 | 无 | ❌ | **block 级索引 / 任务映射计算**：每逻辑任务仅 2 次 div + 2 次 mod，发生在 block 入口（非逐元素热点），无逐元素等价向量 API（索引分解的对象是标量任务号，非张量） |
| 4 | 持久化轮转 `t = cid + it*40` | block 级标量乘加（`T.serial` 内） | 无 | ❌ | 同 #3：block 级循环变量计算，每迭代 1 次乘加；此处的 `T.serial` 是**任务级调度循环**（§6.1），非逐元素循环，不属向量化对象 |
| 5 | 尾块 / 越界守卫 `if t < total_tiles` | 标量比较 | 无 | ❌ | **block 级控制流**（任务级守卫，见 §0.6 R1）；对象是标量任务号，且 R 由构造整除 S（§5.4）故无逐元素边界分支 |
| 6 | 输出写 `delta_s → GM` | 逐元素标量写 | `T.copy(delta_s, delta[b,h,s0], size=[R])` | ✅ | — |

**向量化结论**：**逐元素计算全部向量化**（#1/#2/#6，替代 API 佐证：`T.Parallel` 见 `docs/Tilelang.language/` 及 `examples/indexer/indexer_bwd.py` L92–94；`T.reduce_sum` 见 `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.4 + `_sparse_mla_kernel.py` L268；`T.copy` size 形态见 `indexer_bwd.py` L132）。**保留 3 处标量/循环**（#3/#4/#5），理由类别均为 **block 级索引 / 任务映射**（含任务级调度循环与守卫），逐项见上表，无「实现简单」类理由。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`o`/`do` 为 BSHD，`delta` 为 BHS），核内布局是自由变量。本算子 ≥2 个非 batch 维度（S、D），完成本节分析。

**轴质量评分**：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | D 轴为向量宽度整数倍 | D=64/128：fp16 行宽 = D·2 = 128/256 B（≥128B 连续段 ✓）；向量宽 fp16 ×8（D%8=0 ✓）、fp32 ×4（D%4=0 ✓） |
| 尾 lane 浪费率 | `(ceil(D/vw)·vw − D)/D` | 0%（D=64/128 均为 8 的整数倍） |
| 累加链形态 | D 为归约维（stride=1，连续）| 归约维在连续轴上 → `VReduceOp` 直接作用；**若 lane 映射到 R，对 o[:,d] 的访问 stride=D（跨步系数≠1）会退化向量化** |
| repack 代价 | 改布局的重排代价 | 方案 1/2 均无需 repack；方案 3 需核内 `T.transpose`（UB 级向量管线开销） |
| UB 容量影响 | 驻留缓冲尺寸 | 方案 1/2：`[R,D]`×2 + `[R,D]`fp32 + `[R,1]` = 128.5KB（D=128,R=128），≤192KB ✓ |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **I/O 原生 BSHD，tile = 固定 (b,h) 的连续 s 行块 `[R,D]`**（读 strided 行、写连续）| **D（归约维，连续）** | 无 | 读：行连续 256B 段（2·N·e 流量）；写：`delta` 连续 [R] 段（B·H·S·4 B）；基线 | ✅ **选定** |
| 2 | 输入序展平 tile（`(b,s,h)` 连续行块 `[R,D]`，读全连续）| D | 无（读侧）| 读侧多行合并为 1 段连续（略优于方案 1 的 strided 段）；**但写侧 `delta[b,h,s]` 变成 4B 跨步散写**（B·H·S 次 4B 事务）| ❌ 否选：写侧散写事务数 = 2^17 量级，DMA 事务开销远超读侧段合并收益；且 `delta` 布局契约无法改 |
| 3 | 核内转置为 d-major（`[D,R]`），lane=R、D 串行 | R | 核内 `T.transpose`（2 次二轴交换链）| 无流量/FLOP 收益，额外付转置向量管线开销 + UB 双驻留 | ❌ 否选：**纯增代价零收益**；跨步系数 D≠1 使累加退化（见轴评分第 3 行） |
| 4 | host 侧 permute 进/出 | — | host permute（内容拷贝）| 违反 host 约束（仅允许 stride/shape 元数据视图，`ascend-constraints.md` §4），且 ~百 µs 级代价 ≫ 算子本体 27µs | ❌ 否选 |

**布局决策结论**：**选定方案 #1** —— 核内布局 = 固定 `(b,h)` 的连续 s 行块 `[R,D]`（D 最内连续），**向量化轴 = D（归约维）**，无 repack。理由：D 为连续轴且整除向量宽度、归约沿 D 直接命中 `VReduceOp`；写侧 `delta` 保持 BHS 连续段（契约布局）；方案 2/3/4 的量化理由见上表。该结论同步落入 §3.3 伪代码（`T.Parallel(R,D)` 最内维 = D）与 §6.1 循环结构。

> 弃选论证证据：`T.transpose` 依赖项已核 `docs/Tilelang.language/创建操作/`（UB 级两轴交换链，非 GM 级重排——未把核内重排写成 GM 级稻草人）；host permute 约束见 `ascend-constraints.md` §4（本设计未凭空断言，引用具体条款）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（用户指定；且与算子特征一致）。

### 2.2 选型理由

- **纯 Vector 算子**：仅逐元素乘 + 单遍归约，无 matmul → 不需要 Cube/L0C，不需要手动 L1/L0 分配与 `sync_block_set/wait` 跨引擎同步（Expert 模式的收益点）。
- **无跨引擎流水**：数据流为 `GM→UB→GM` 单链，Developer 的自动同步 / 自动 buffer 管理足够。
- 同类 NPU 先例 `examples/indexer/indexer_bwd.py`（Developer，`reduce_sum` + `T.copy size=` 写回）与本算子结构高度一致。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared([R,D], dtype)` / `T.alloc_shared([R,D], "float")`（UB 级），`T.alloc_shared([R,1], "float")` |
| 计算方式 | `T.Parallel(R,D)` 逐元素向量化 + `T.reduce_sum(dim=1)` |
| 同步 | 自动同步（§7.1），无手动 flag |
| 流控 | `T.serial` 任务级循环（§6.1），无 `T.Pipelined`（UB 预算不允许双缓冲，§6.3） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `t = cid + it·N_PHYS`, `(bz,bx,by) = decode(t)` | 逻辑任务号 → (b,h,s 行块) |
| 2 | `o_s ← o[bz, by·R:(by+1)·R, bx, :]`；`do_s ← do[同切片]` | 搬入两个 `[R,D]` tile |
| 3 | `prod[i,j] = fp32(o_s[i,j])·fp32(do_s[i,j])` | 逐元素乘（fp32 提升） |
| 4 | `delta_s[i,0] = Σ_j prod[i,j]` | 沿 D 单遍归约 |
| 5 | `delta[bz, bx, by·R:(by+1)·R] ← delta_s` | 连续写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 任务索引分解 | 标量算术（`cid`、`//`、`%`）| `bz = cid // (heads*n_s)` 等 | Developer |
| 2 | 搬入 | `T.copy(o[bz, s0:s0+R, bx, 0:D], o_s)` | slice 形态（行连续 256B）| Developer |
| 3 | fp32 提升 + 乘 | `T.cast(x, "float")` + `T.Parallel(R,D)` 内的 `*` | — | Developer |
| 4 | 行归约 | `T.reduce_sum(prod_s, delta_s, dim=1, clear=True)` | `dim=1`（末轴 D）| Developer |
| 5 | 搬出 | `T.copy(delta_s, delta[bz,bx,s0], size=[R])` | `size=` 形态（[R,1]→GM 连续 [R]）| Developer |
| — | 任务级循环 | `T.serial(iters)` | `iters` 编译期静态值 | Developer |

### 3.3 计算伪代码

> 注意：`T.Tensor` 的 dtype 直接作为第二个位置参数传入（不用 `dtype=` 关键字），避免 `name 'in_dtype' is not defined` 类 false alarm。

```python
@tilelang.jit(out_idx=[2])
def _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype):
    accum_dtype = "float"
    R = math.gcd(seq_len, 128)              # 编译期静态；整除 seq_len
    n_s = seq_len // R
    total_tiles = batch * heads * n_s
    N_PHYS = 40                              # 纯 Vector 物理核数（实查 20×2）
    nblk = min(N_PHYS, total_tiles)
    iters = (total_tiles + nblk - 1) // nblk  # 静态循环边界

    @T.prim_func
    def flash_bwd_prep(
        o:   T.Tensor((batch, seq_len, heads, dim), dtype),
        do:  T.Tensor((batch, seq_len, heads, dim), dtype),
        delta: T.Tensor((batch, heads, seq_len), accum_dtype),
    ) -> None:
        with T.Kernel(nblk, is_npu=True) as (cid, _):
            o_s     = T.alloc_shared([R, dim], dtype)
            do_s    = T.alloc_shared([R, dim], dtype)
            prod_s  = T.alloc_shared([R, dim], accum_dtype)
            delta_s = T.alloc_shared([R, 1], accum_dtype)

            for it in T.serial(iters):
                t = cid + it * nblk
                if t < total_tiles:
                    bz = t // (heads * n_s)
                    bx = (t % (heads * n_s)) // n_s     # head
                    by = t % n_s                        # seq 行块
                    s0 = by * R
                    # 2. 数据搬入（slice 形态：行连续 D 段）
                    T.copy(o[bz, s0:s0 + R, bx, 0:dim], o_s)
                    T.copy(do[bz, s0:s0 + R, bx, 0:dim], do_s)
                    # 3. fp32 提升 + 逐元素乘（向量化）
                    for i, j in T.Parallel(R, dim):
                        prod_s[i, j] = T.cast(o_s[i, j], accum_dtype) \
                                     * T.cast(do_s[i, j], accum_dtype)
                    # 4. 沿 D 单遍归约
                    T.reduce_sum(prod_s, delta_s, dim=1, clear=True)
                    # 5. 连续写回（[R,1] -> delta[bz,bx,s0:s0+R]）
                    T.copy(delta_s, delta[bz, bx, s0], size=[R])

    return flash_bwd_prep
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(n, is_npu=True)` 一维 | `docs/Tilelang.language/` + `examples/indexer/indexer_bwd.py` L53 邻域 | 现有先例 |
| `T.alloc_shared` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.4 示例 | 文档化 |
| `T.copy` slice 形态（含混合索引/slice）| `_sparse_mla_kernel.py` L221（`T.copy(Q[b_i,s_i,h0:h0+bh,0:Dqk], Q_shared)`）| 现有先例 |
| `T.copy` `size=` 形态写回 | `examples/indexer/indexer_bwd.py` L132（`T.copy(dw_acc_ub, dWeights[idx,0], size=[nh])`）| 现有先例 |
| `T.Parallel` + `T.cast` | `examples/indexer/indexer_bwd.py` L92–94 邻域 / sparse_mla L222 | 现有先例 |
| `T.reduce_sum(buf, out, dim=1, clear=True)` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.4；`_sparse_mla_kernel.py` L268 | 文档化 + 先例 |
| `T.serial` 静态边界 | `docs/Tilelang.language/` + 多处 examples | 现有先例 |
| `math.gcd`（host 侧编译期）| Python 标准库 | — |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是（源为 3D grid，L39）** | 展开为一维 `total_tiles`；block 索引静态分解（§0.6 R1） |
| GEMM 非整除 / L0C 溢出 | 否 | 无 Cube 计算 |
| 三维 Kernel / GPU 专用 API | 是（`T.Kernel(m,n,b)`、fragment tiling）| 已重设计（§0.5/§0.6） |
| 尾轴对齐（32B）| 否 | D=64/128 → 行 128/256B ≥32B ✓ |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | `T.Kernel(heads, ceil(S/256), batch)` 三维（L39）| 一维 `T.Kernel(nblk)` + 索引分解 | §0.6 R1 |
| tile 形状 | `[256,256]` fragment（fp32 acc 256KB）| `[R,D]`（R=128, ≤128.5KB UB）| §0.6 R2 |
| 累加器 | register fragment `acc` | UB `prod_s`（fp32）| 等价替换 |
| 归约 | `T.reduce_sum(acc, delta_frag, 1)` | `T.reduce_sum(prod_s, delta_s, dim=1, clear=True)` | 等价替换（`VReduceOp`）|
| chunk 循环 | `for k in range(ceildiv(dim,256))` | 消解（tile 取全宽 D）| §0.6 R3 |
| 输出写 | `T.copy(delta_frag, delta[bz,bx,s0:s0+R])` | `T.copy(delta_s, delta[bz,bx,s0], size=[R])` | Developer 写回形态 |
| 分核 | 3D grid → 512 逻辑核（SMs 自动调度）| 固定 40 block + `T.serial` | §5.5 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/indexer/indexer_bwd.py` | 高 | `T.reduce_sum(dim=1, clear=...)` 用法；`T.copy(..., size=[n])` 写回连续段；Developer 纯 Vector buffer 形态 |
| `examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py` | 中高 | `T.reduce_sum(acc_s, loc_sum, dim=1, clear=True)`；slice 形态 `T.copy` 读 BSHD strided tile（L221）|
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 中 | 1-D `T.Kernel` 任务分解与持久化结构（Expert；本设计仅参考任务映射思想）|

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `o` | `(B, S, H, D)` | float16 / bfloat16 | BSHD，D 连续 |
| `do` | `(B, S, H, D)` | float16 / bfloat16 | BSHD，D 连续 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `delta` | `(B, H, S)` | float32 | BHS（与输入 s/h 顺序不同）；`out_idx=[2]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `o_s` | `[R, D]` | fp16/bf16 | UB | o tile |
| `do_s` | `[R, D]` | fp16/bf16 | UB | do tile |
| `prod_s` | `[R, D]` | fp32 | UB | `o⊙do`（fp32）|
| `delta_s` | `[R, 1]` | fp32 | UB | 行归约结果 |

### 4.4 内存搬运路径

```
GM[o  BSHD]  --T.copy(slice)--> UB[o_s [R,D] fp16]   ┐
GM[do BSHD]  --T.copy(slice)--> UB[do_s [R,D] fp16]  ├─ T.Parallel → UB[prod_s [R,D] fp32]
                                                      ┘        │ T.reduce_sum(dim=1)
                                                               ▼
                                          UB[delta_s [R,1] fp32] --T.copy(size=[R])--> GM[delta BHS]
```
纯 Vector 路径，无 L1/L0A/L0B/L0C；单级 GM↔UB 搬运，无中间落 GM。

### 4.5 UB 内存预算

**D=128, R=128**（worst case）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `o_s` | (128,128) | fp16 | 32768 |
| `do_s` | (128,128) | fp16 | 32768 |
| `prod_s` | (128,128) | fp32 | 65536 |
| `delta_s` | (128,1) | fp32 | 512 |
| **总计** | | | **131584 ≈ 128.5 KB** / 192 KB（Ascend910B3 实测上限） |

- 余量：`192KB − 128.5KB = 63.5KB`；按 `CONST-capacity-910B2C` 的 UB 膨胀系数上偏口径 ×1.12 → 147.4KB < 192KB ✓（该系数为 910B2C 结构实测，910B3 未重验，已保守采用）。
- **D=64, R=128**：`o_s/do_s` 各 16384 + `prod_s` 32768 + `delta_s` 512 = **66.05KB**（远低于上限）。
- 保守变体（若 Stage 3 遇 UB 膨胀）：R=64 → D=128 时总计 64.3KB；本文档主选 R=128。

### 4.6 动态轴定义

**无动态轴**。`batch / heads / seq_len / dim` 均为编译期常量（源 factory 签名 L27）；`R = gcd(seq_len,128)`、`total_tiles`、`iters` 均在编译期求得。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2])
def _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype):
    ...
```
（沿用源 `out_idx=[2]`；无 `pass_configs`/`compile_flags` —— 源 `-O3 -DENABLE_BF16` 为 CUDA 编译标志，NPU 不适用，已舍弃。）

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。
**判定依据**：仅含逐元素乘（`T.Parallel`）与单遍归约（`T.reduce_sum`），无 matmul → 无需 Cube/L1/L0/L0C，数据流 `GM→UB→GM`。

### 5.2 Block 划分

```python
R = math.gcd(seq_len, 128)      # 行块大小（编译期静态，整除 seq_len）
n_s = seq_len // R              # s 方向行块数
total_tiles = batch * heads * n_s
N_PHYS = 40                     # 物理 Vector 核数
nblk = min(N_PHYS, total_tiles)
iters = ceildiv(total_tiles, nblk)   # 核内串行迭代数（静态）
```
- `R = 128` 选择理由：① 整除全部 manifest/测试 shape 的 `seq_len`（128/512/1024/2048）；② D=128 时 UB 128.5KB（含膨胀后 147.4KB）< 192KB；③ tile 内元素数 = 16384，DMA 段 ≥ 每行 256B，避开 `CONST-elu-910B3-bw-floor` 记录的 bs=2048（4KB tile）退化区（0.64 TB/s）。

### 5.3 约束分析

- **对齐约束**：D=64/128，行宽 128/256B ≥ 32B ✓；向量宽 fp16 ×8 / fp32 ×4，D%8=0 / D%4=0 ✓。
- **UB 容量**：128.5KB（含 ×1.12 → 147.4KB）< 192KB ✓。
- **L0 容量**：无 Cube 计算，不适用。

### 5.4 注意事项（非整除处理）

- **本算子以「编译期选 R 整除 seq_len」彻底消除尾块**：`R = gcd(seq_len, 128)` 由构造保证 `seq_len % R == 0`，因此**无 padding、无动态边界、无尾块掩码**（对照源算法 `blk=256` 固定、`by*blk` 可能越界的隐患，本设计更安全）。manifest 全部 shape 给出 R=128。
- `seq_len < 128` 时 R = seq_len（degrade 到 1 个行块）；`seq_len` 非 2 的幂时 R 取 `gcd`（如 1000 → R=8），仍静态、无尾块。
- `total_tiles < 40`（smoke: 8）时 `nblk = total_tiles`，`iters = 1`（§5.5 ①）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 `docs/开发指南.md` §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录；纯 Vector 算子核数翻倍。

**① 逻辑核数计算**

```
num_logical_kernels = batch × heads × ceil(seq_len / R) = batch × heads × n_s
```
逐 workload（R=128）：

| workload（B,S,H,D）| total_tiles | 备注 |
|---|---|---|
| `[1,128,8,64]`（smoke）| `1×8×1 = 8` | < 40 |
| `[4,512,32,128]` | `4×32×4 = 512` | |
| `[2,2048,32,128]` | `2×32×16 = 1024` | 最大 |
| `[2,512,64,128]` | `2×64×4 = 512` | |
| `[1,2048,64,128]` | `1×64×16 = 1024` | 最大 |

**② 物理核数依据（实查）**

- 查询代码：`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()`
- **实际返回值：`20`**（设备 `torch_npu.npu.get_device_name(0)` = **Ascend910B3**；`npu-smi info` 同证）。
- 本算子为**纯 Vector**（无 Cube）→ **物理核数 = 20 × 2 = 40**（AIV）。记 `N_PHYS = 40`。*（禁止文档假设/经验值替代——此处为实查 20 后按标准翻倍。）*

**③ 规模判定与分核方案**

- **规模判定：极大规模**。最大 `num_logical_kernels = 1024`，为物理核数 40 的 **25.6 倍**；且无法通过增大 R 缩减到接近 40（R 受 UB 硬上限约束：D=128 时 R≤128 已使 UB 达 128.5KB；即便取 R=160 也仅降到 819，仍 ≫40）。→ 适用标准 §1 要素③ **「极大规模——核内 `T.serial` 串行」**。
- **分核方案**：**固定启动内核数 `nblk = min(40, total_tiles)`；核内 `T.serial(iters)` 轮转处理多个逻辑行块**：
  ```
  iters = ceildiv(total_tiles, nblk)     # 编译期静态边界（最大 workload: ceil(1024/40)=26）
  for it in T.serial(iters):
      t = cid + it * nblk                # 轮转（round-robin）任务映射
      if t < total_tiles: <处理任务 t>    # 尾轮守卫
  ```
  - 循环边界 `iters` 为**静态编译期常量**（满足标准「核内串行循环边界必须为静态值」）。
  - 效果：把 25.6 波次摊薄为 1 波（每核 26 个串行 tile），消除 1024 次 block 启动开销中的绝大部分。
  - smoke（8 tiles）：`nblk=8`、`iters=1` → 属「逻辑核数 ≤ 物理核数」的无需适配情形（8 ≤ 40）。
- **残留负载不均（显式记录）**：`total_tiles` 恒为 2 的幂（8/512/1024），而 `nblk = 40 = 2³×5` 含非 2 因子，数学上**无法对齐任何整数倍**（与标准 §1 要素③「无法对齐兜底」的同源约束）。残留代价：1024 tiles 时 26×40 = 1040，**16 个核多处理 1 个 tile**（临界路径 26 vs 均值 25.6 → **+1.6%**）；512 tiles 时 13×40 = 520，8 核多 1 个（13 vs 12.8 → +1.6%）。该残留 < 2%，可接受；**是否改用 `nblk=32`（512/1024 的公约数，完美均衡但闲置 8 核）留待 Stage 4 实验裁决**（§9.3）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑行块（持久化轮转）| 任务级串行 | `T.serial(iters)` | 极大规模核内串行，摊薄核启动开销（§5.5 ③）；静态边界 |
| 元素级乘 `o⊙do` | 向量化 | `T.Parallel(R, dim)` | 逐元素并行（最内维 = D，§1.6.3 选定轴）|
| D 维归约 | 向量化 | `T.reduce_sum(dim=1, clear=True)` | `VReduceOp`（§1.6.2 #2）|
| 逻辑任务索引分解 | block 级标量 | `//`、`%` | 非逐元素热点（§1.6.2 #3）|

> 无逐元素标量循环；保留的 3 处标量点均在 §1.6.2 有充分理由（block 级索引 / 任务映射 / 任务级守卫）。

### 6.2 循环伪代码

```python
with T.Kernel(nblk, is_npu=True) as (cid, _):
    o_s, do_s, prod_s, delta_s = <alloc>          # §3.3
    for it in T.serial(iters):                     # 任务级串行（静态边界）
        t = cid + it * nblk
        if t < total_tiles:                        # 任务级守卫（§1.6.2 #5）
            (bz, bx, by) = decode(t)               # 索引分解（§1.6.2 #3）
            T.copy(o [...], o_s); T.copy(do [...], do_s)
            for i, j in T.Parallel(R, dim):        # 向量化逐元素乘
                prod_s[i, j] = T.cast(o_s[i,j],"float") * T.cast(do_s[i,j],"float")
            T.reduce_sum(prod_s, delta_s, dim=1, clear=True)
            T.copy(delta_s, delta[bz, bx, s0], size=[R])
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**（主选）。理由：UB 预算下 `o_s+do_s+prod_s+delta_s` = 128.5KB，双缓冲需 ~257KB > 192KB；且 `T.Pipelined` 与内核内 `if t < total_tiles` 守卫的组合行为未经本仓验证（风险，§9.2）。
**Stage 4 可选**：若 profile 显示 MTE2 与 Vector 未重叠导致低效，改 R=64（单套 64.3KB）在任务级循环上启用 `T.Pipelined(iters, num_stages=2)`（2×64.3=128.6KB < 192KB）。

### 6.4 尾块处理

**无尾块**：`R = gcd(seq_len,128)` 整除 `seq_len`（§5.4），切片 `o[bz, s0:s0+R, bx, :]` 恒在界内；`total_tiles` 与 `nblk` 的余数由 `if t < total_tiles` 守卫处理（跳过多余任务，非部分 tile）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式默认）。无手动 flag / 无 `sync_block_set/wait`（单核内单链 `GM→UB→算→GM`，无跨引擎、无跨核依赖）。

### 7.2 同步点说明

无手动同步点。`T.copy`（MTE3/MTE2）、`T.Parallel`/`T.reduce_sum`（Vector）、`T.copy`（MTE3 出）之间的依赖由编译器自动插入同步（Developer 模式标准行为）；`T.serial` 迭代间因 buffer 复用（WAR）亦由自动同步保证。

### 7.3 pass_configs 配置

```python
pass_configs = {}      # 与同步相关的 pass 配置：无
```
源 GPU 的 `pass_configs={TL_ENABLE_FAST_MATH: True}` 与 `compile_flags=["-O3","-DENABLE_BF16"]`（CUDA 编译标志）**不迁移**（NPU 不适用；fp32 乘法无 fast-math 需求）。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据（`delta[b,h,s] = Σ_d o[b,s,h,d]·do[b,s,h,d]`，fp32），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。

```python
def golden_flashattn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """o/do: (B,S,H,D) fp16/bf16 -> delta: (B,H,S) fp32."""
    # 行内点积：沿最后一维 D 求和；乘法与累加均以 fp32 进行（对齐源码 accum fp32 语义）
    delta_bsh = (o.float() * do.float()).sum(dim=-1)   # [B,S,H] fp32
    return delta_bsh.transpose(1, 2).contiguous()       # [B,S,H] -> [B,H,S]
```

### 8.2 精度标准

| dtype（输入）| 输出 | atol | rtol | 说明 |
|-------|------|------|------|------|
| float16 | float32 | 1e-3 | 1e-4 | 乘积在 fp32 下精确（fp16×fp16 ≤22 位有效位），差异仅来自 fp32 累加顺序 |
| bfloat16 | float32 | 5e-3 | 1e-3 | bf16 输入本身有舍入，但本 kernel 不引入 |

> 上游 op 端到端测试 `test_grouped_query_attention_bwd.py` 用 `atol=5e-3, rtol=1e-5`；本 kernel 单独验证用上表更严值。

**L0 门槛测试计划**：
1. **主用例**（覆盖全部 manifest workloads）：`[1,128,8,64]`、`[4,512,32,128]`、`[2,2048,32,128]`、`[2,512,64,128]`、`[1,2048,64,128]`，fp16 与 bf16 各一遍，与 §8.1 golden 比对（assert 全通过）。
2. **边界用例**：`S < R`（如 `seq_len=64, R=64`）；`B=1,H=1`；非 2 的幂 `seq_len=1000`（R=8，验证 gcd 选 R 逻辑）。
3. **数值边界**：`o/do` 含 0、±极值（fp16 max）、全零（delta=0）。
4. Layer 1（单 kernel L0 精度+契约）、Layer 2 由 `tilelang-op-develop` 承接；完整 L1/L2/Boundary 套件交下游 skill。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **UB 硬上限 192KB**：D=128,R=128 时 128.5KB（×1.12 → 147.4KB），余量 ~23%；R 不可再增（R=160 → 164.5KB 逼近上限）。
2. **一维 Kernel**：源三维 grid 已展开；索引分解为 block 级标量（不可向量化，§1.6.2）。
3. **物理核数 40 与 2 的幂 tile 数无法整除对齐**（残留 +1.6%，§5.5）。
4. **`T.Pipelined` 未启用** → MTE2 与 Vector 是否重叠依赖编译器自动调度（若未重叠，性能上界受 §1.6.0 roofline 高端 46–57µs 约束）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | R 取 256（D=128）→ 257KB | 编译失败（`ub overflow ... 1572864 bits`）| R≤128（本文档主选）|
| `T.copy` rank 不匹配 | `[R,1]` fragment 直接 copy 到 `delta[bz,bx,s0:s0+R]`（1D）| 编译期 shape 报错 | 用 `size=[R]` 形态（§3.2，先例 `indexer_bwd.py` L132）|
| `T.reduce_sum` 未初始化 | `clear=False` 且 `delta_s` 未初始化 | **静默数值错误**（`T.reduce_sum.md` §2.3 / VP-2026-0221）| 恒用 `clear=True`（§3.3）|
| 尾轮越界 | `t ≥ total_tiles` 仍执行切片 | 读到界外/写坏 `delta` | `if t < total_tiles` 守卫（§6.2）|
| `if t < N` 标量守卫编译问题 | 任务级动态控制流（`TRAP-if_then_else-runtime-scalar-cmp` 为 `T.Parallel` 内标量比较，形态不同）| 编译失败 | 备选：`t = min(t, total_tiles-1)` 无分支钳制（幂等重算，见 §9.3）|
| strided GM 读被误用 load_nd2nz | 若误用 `T.load_nd2nz` 读 BSHD 固定 head tile | 静默读错（`_gqa_prefill_fwd_kernel` 实施注记 1）| 恒用 slice 形态 `T.copy`（§3.2）|

### 9.3 特殊场景处理 / 待裁决项

- **分核备选（Stage 4 实验裁决）**：主选 `nblk=40`（残留 +1.6%）；备选 `nblk=32`（512/1024 的公约数，完美均衡，闲置 8 核）。裁决计划：代表 shape `[2,2048,32,128]`/`[1,2048,64,128]` 上 msprof op 实测 `TaskDur`，**若备选 latency < 主选 × 0.98 则翻转**；裁决数据经附录补记回写本节。
- **守卫备选**：`if t < total_tiles`（主选）vs `t = min(t, total_tiles-1)`（无分支幂等重算，多算 ≤16 个 tile ≈ +1.6% 流量）。若主选编译失败则启用备选。
- **极小 shape**：smoke `[1,128,8,64]` → 8 tiles、`nblk=8`、`iters=1`，无持久化开销。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_query_attention_bwd/_flashattn_bwd_preprocess_kernel/
├── DESIGN.md                        # 本设计文档
├── verify_equiv.py                  # §1.6.1 采纳项 #1 的 D-1 机器验证（EQUIV_PASS）
├── _flashattn_bwd_preprocess_kernel.py   # 算子实现 + L0 测试
└── RETROSPECTIVE.md                 # Stage 1 复盘（自进化钩子）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档 |
| `_flashattn_bwd_preprocess_kernel.py` | ⬜ 待实现 | 算子实现（DEVELOP 阶段）|
| `verify_equiv.py` | ✅ 已完成 | §1.6.1 采纳项 #1 的 D-1 等价性机器验证脚本（实测全 `EQUIV_PASS`，exit 0）|
| `RETROSPECTIVE.md` | ✅ 已追加 | Stage 1 复盘 |

### 10.3 命名规范

- 项目目录：`grouped_query_attention_bwd`（snake_case）；算子目录：`_flashattn_bwd_preprocess_kernel`；实现文件同名 `.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ 算子实现（`_flashattn_bwd_preprocess_kernel.py`）+ L0 精度比对
4. ⬜ 分核备选 A/B 裁决（§9.3，Stage 4）
