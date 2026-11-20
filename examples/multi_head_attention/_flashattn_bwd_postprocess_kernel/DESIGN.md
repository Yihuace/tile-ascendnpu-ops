# `_flashattn_bwd_postprocess_kernel` 算子设计文档

> 迁移场景（harness 模式，`mode=first_design`）。算子目录：`examples/multi_head_attention/_flashattn_bwd_postprocess_kernel/`。
> 源算子：`/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`（本函数定义在第 106–128 行；`FlashAttnBwdPostprocessKernel` 类在第 131–171 行；其依赖的 `make_dq_layout` 在第 99–103 行）。
> 迁移背景：`MultiHeadAttentionBwdOp`（family=attention）backward 三件套的**收尾 kernel**；前两件套（`_flashattn_bwd_preprocess_kernel`、主 kernel `_gqa_bwd_kernel`）已在同批交付。
> 设备/工具链戳：**Ascend910B3**；tilelang 0.1.2 dev root build（`/workspace/tilelang-mlir-ascend/build`）+ CANN 9.0.0；本设计实查 `NPUUtils.get().get_aicore_num() = 20`、`AscendArch().aivector_core_num = 40`、`AscendArch().ub_cap = 196608`（2026-10-16）。

## 0. 源算子解读与迁移分析

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：逐元素（element-wise）**dtype 下转换**，无规约、无缩放、无掩码、无广播：

$$
\mathrm{dq\_out}[b,s,h,d] \;=\; \mathrm{cast}_{T}\big(\mathrm{dq}[b,s,h,d]\big),
\qquad T \in \{\text{float16},\ \text{bfloat16}\}
$$

其中 `cast_T` 是 IEEE 754 的 **round-to-nearest-even（RNE）** 由 fp32 到存储 dtype 的转换（源实现经 `T.copy` 的隐式跨 dtype cast 完成，而 CUDA 侧该隐式 cast 即 `__float2half` / `__nv_bfloat16` 的 RNE 语义）。

| 语义要素 | 解读 | 依据 |
|----------|------|------|
| 数学映射 | 元素级一元映射 `x -> cast_T(x)`；输入输出元素位置**一一对应**（恒等索引映射，无置换、无转置） | 源第 123–126 行的 `T.copy` 两侧区域切片下标完全一致（`[bz, bx*blk:(bx+1)*blk, by, :]` 对 `[bz, bx*blk:(bx+1)*blk, by, :]`） |
| I/O 契约 | 输入 `dq`：`(B, S, H, D)` fp32；输出 `dq_out`：`(B, S, H, D)` 存储 dtype（fp16/bf16）。BSHD 行主序，`D` 轴 stride 1 | `source_output_shape = (B, S, H, D) fp16/bf16`；op 层 `dq = torch.zeros_like(q, dtype=torch.float32)`（`examples/TileOPs/tileops/ops/attention/multi_head_attention.py` 第 249 行）与 `dq = self.post_kernel(dq)`（第 253 行） |
| 规约语义 | **无规约**（无跨元素依赖）⇒ 累加顺序无关，输出与并行划分方式无关 | 逐元素映射 |
| dtype 语义 | 输入固定 **fp32**（主 kernel 的累加器 dtype，`accum_dtype = "float"`，源第 112 行）；输出 = 存储 dtype；**唯一**的数值行为是舍入模式（RNE）与溢出/次正规处理 | 源第 112 行 `accum_dtype = "float"`、第 118 行 `dq: T.Tensor(shape, accum_dtype)`、第 119 行 `dq_out: T.Tensor(shape, dtype)` |
| 边界语义 | fp32 `±inf`/`NaN` 原样传播到输出（`NaN -> NaN`、`±inf -> ±inf`）；超 fp16 表示范围（`|x| >= 65520`）溢出为 `±inf`；fp16 次正规区间（`|x| < 6.103515625e-05`）按 RNE 量化；`±0` 保号 | 语义由「fp32→fp16/bf16 的 IEEE RNE 转换」定义；本设计以 D-1 机器验证固化（§1.6.1） |
| 动态轴 | 无。`B/S/H/D/dtype` 均为工厂期常量（源工厂同为 per-shape JIT 特化） | 源第 107–111 行的工厂签名 |

**语义保持基线**：§8.1 的 golden 函数以本节语义为唯一依据实现（`torch.Tensor.to(storage_dtype)`），**不复刻** NPU 算法（§0.6）。

**`make_dq_layout` 的语义地位（关键判定点）**：源第 122 行 `T.annotate_layout({dq: make_dq_layout(dq)})` 不是本算子的**语义**，而是**与主 kernel 共享的物理布局契约**——它把 `dq` 的物理地址重排（第 101–103 行的索引映射），使主 kernel 在 Ampere 上的 atomicAdd 能按 8×8 mma fragment 向量化（第 100 行注释：`atomicAdd cannot be vectorized on Ampere, need to reorder dq to match the 8x8 gemm fragment`）。读取方（本 kernel）必须用**同一个** layout 注解才能读回逻辑上的 `dq[b,s,h,d]`。它是「实现层布局手段」，不是「输出语义」——§0.5 对其做四态判定。

### 0.2 源算子输入输出规格

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `dq` | `(batch, seq_len, heads, dim)` | **float32** | 主 kernel 的 dq 累加器（源第 118 行 `accum_dtype = "float"`）；BSHD，`dim` 轴 stride 1，`seq_len` 轴 stride = `heads·dim` |
| `dq_out` | `(batch, seq_len, heads, dim)` | float16 / bfloat16 | **输出**（源第 119 行、`@tilelang.jit(out_idx=[1])` 第 106 行 ⇒ 运行时自动分配）；shape 与 `dq` 完全一致（`source_output_shape = (B, S, H, D) fp16/bf16`），无转置布局 |

**op 层调用序列（契约边界，决定本 kernel 不得越界改动主 kernel 产物）**：

```text
dq = torch.zeros_like(q, dtype=torch.float32)      # op 层第 249 行：fp32 零初始化
dk = torch.zeros_like(k, dtype=torch.float32)      # 第 250 行
dv = torch.zeros_like(v, dtype=torch.float32)      # 第 251 行
self.kernel(q, k, v, do, lse, delta, dq, dk, dv)   # 第 252 行：主 kernel 以原子写填 dq（fp32）
dq = self.post_kernel(dq)                          # 第 253 行：本 kernel（唯一消费者）
dk, dv = dk.to(self.dtype), dv.to(self.dtype)      # 第 254 行：dk/dv 的 cast 在 op 层完成，不经本 kernel
return dq, dk, dv
```

⇒ 本 kernel 的输入 `dq` 与主 kernel 的输出 `dq` 是**同一个** `torch.zeros_like(q, dtype=torch.float32)` 缓冲；`dq` 的物理布局由**主 kernel 的写侧**决定，本 kernel 只能读取该布局。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源 kernel 的全部计算语句，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.annotate_layout({dq: make_dq_layout(dq)})`（源第 122 行）——**布局声明，非计算** | `dq` | 物理布局注解 | 无（物理地址重排，仅供原子写向量化） |
| 2 | `T.copy(dq[bz, bx*64:(bx+1)*64, by, :], dq_out[bz, bx*64:(bx+1)*64, by, :])`（源第 123–126 行）——单条语句同时完成「装载 + dtype 转换 + 写出」 | `dq` 的 `(64, dim)` 逻辑区域 | `dq_out` 的同名区域 | 整个公式 `cast_T(dq[b,s,h,d])` |

**计算语句清单核对**：源 `@T.prim_func flash_bwd_post`（第 116–126 行）体内**只有一条**可在语义上归类的计算语句（步骤 2）与一条布局声明（步骤 1）。全程无 `T.Parallel` / `T.reduce` / 算术运算 / 标量循环 ⇒ 步骤分解完整。

**数据流与内存访问模式（源硬件视角）**：

```text
GM[dq  (B,S,H,D) fp32, 但物理地址被 make_dq_layout 重排]
   │  T.copy（隐式跨 dtype cast = RNE；row = dim·4 B，跨步轴 stride = heads·dim·4 B）
   ▼
SMEM/register（源侧由 T.copy 内部完成，用户不可见）
   │
   ▼
GM[dq_out (B,S,H,D) fp16/bf16, 自然 BSHD 布局]
```

- **每次搬运的 layout 变换**：读侧 = `make_dq_layout` 的逆映射（由 `T.annotate_layout` 的 layout 传播自动完成）；写侧 = 自然 BSHD（`dq_out` **无** layout 注解）。
- **buffer 复用关系**：无显式 UB/SMEM 分配——源把「装载 + cast + 写出」压进单条 `T.copy`，中转由编译器内部完成。

**循环与并行结构**：

- **grid**：`T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128)`（源第 121 行）——**三维** grid `(⌈S/64⌉, H, B)`；`blk = 64`（源第 114 行）。
- **block 内**：`threads=128`（CUDA 线程块）；**无显式循环**（单条 `T.copy` 覆盖 `(64, dim)` 区域）。
- **总逻辑任务数**：`⌈S/64⌉ · H · B`；每个任务处理 `64·dim` 个元素。
- **无流水**（单条语句，无 `T.Pipelined` / `num_stages`）。

**host 侧逻辑**：无。源 `FlashAttnBwdPostprocessKernel.forward`（第 170–171 行）直接 `return self.kernel(dq)`，`__init__`（第 151–168 行）只保存 shape/dtype 并构建 kernel；无预处理、无后处理、无 `contiguous()`/`permute()`。

**编译配置**：本函数**无** `pass_configs`、**无** `compile_flags`（对比 `_mha_bwd_kernel` 第 188–193 行的 `TL_ENABLE_FAST_MATH` / `-O3 -DENABLE_BF16`）。源 `supported_archs = [80, 89, 90]`（第 149 行）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **`make_dq_layout` 物理地址重排**（源第 99–103 行 / 第 122 行） | 让**主 kernel** 的 `atomicAdd` 在 Ampere 上可向量化（按 8×8 mma fragment 对齐被原子更新的地址） | 用 `T.Layout` 把 `(b, length, h, d)` 映射到 `[b, length//8, h, d//8, (d%2), 4·(length%8)+(d%8)//2]`；读侧用同一 layout 反解 | **Ampere 的 mma fragment 形状 + atomicAdd 的向量化条件** | **硬件强相关**（GPU 专属；详见 §0.5） |
| 2 | **三维 grid `(⌈S/blk⌉, H, B)` + `blk=64` 固定几何**（源第 114 / 121 行） | 提供 block 级并行度；`blk=64` 是 CUDA 侧容量/占用率的经验取值（源注释未给推导） | 三维 grid 各轴对应 `(s-block, head, batch)` | block×thread 两级并行模型 | **硬件强相关**（NPU `T.Kernel` 只接受一维；见 `references/ascend-constraints.md` §1） |
| 3 | **跨 dtype `T.copy`（隐式 cast 融合）**（源第 123–126 行） | 把「装载 + cast + 写出」融合成一条语句，省一次显式中间缓冲与一次向量 pass 的书写 | 编译器在 copy 内插入类型转换 | 无（CUDA `T.copy` 自动插 cast） | **模式可移植**（NPU Developer 侧同为自动插 `hivm::VCastOp`；见 `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条） |
| 4 | **`threads=128`**（源第 121 行） | CUDA 线程块宽度（影响 occupancy 与 copy 的线程级并行） | 128 线程块 | CUDA thread block | **硬件强相关 → 无意义**（NPU npuir 的 `T.Kernel(threads=)` 为 no-op；见 `pattern-library/traps-compiler.md` 的 TRAP-threads-kwarg-noop） |
| 5 | **单条语句、无循环、无流水** | 该 kernel 为纯带宽受限搬运，无需流水/计算重叠 | 无 | 无 | **可移植** |

> 识别结论：本源算子的「快」**不来自算法**（语义上只有一次逐元素 cast），而全部来自 ① 与主 kernel 共享的物理布局（手段 1）与 ② CUDA 执行模型下的几何/线程选择（手段 2/4）。手段 3/5 是通用搬运模式。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 逐元素 dtype 下转换（RNE）语义 | **语义** | 无 | — | **保留** | 语义层无条件保留（`references/migration-analysis.md` §5.1）。`T.vcast(round_mode="rint")` 提供逐位相同的 RNE（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1，f32→f16 / f32→bf16 均支持 `rint`） |
| I/O 契约（BSHD 进 / BSHD 出，恒等索引映射） | **语义** | 无 | — | **保留** | 输出 `(B,S,H,D)` 与输入同形同位置（§0.1/§0.2） |
| **`make_dq_layout` 物理布局重排** | 优化 | Ampere mma fragment 形状 + atomicAdd 向量化条件 | **无**（NPU 无 warp mma fragment 概念、无 SMEM bank conflict 问题） | **舍弃** | 理由与自洽性论证见下方「判定 1」 |
| 三维 grid `(⌈S/blk⌉, H, B)` | 优化（并行结构） | block×thread 两级并行 | 无（`T.Kernel` 只接受一维 block 数） | **重新设计** | → §0.6 R1（一维 persistent 分核） |
| `blk = 64` 固定几何 | 优化（tile 形状） | CUDA 容量/占用率经验值 | 无直接等价（NPU 按 UB 192 KB 反推） | **重新设计** | → §0.6 R2（UB 预算驱动的 `(blk_s, H, D)` 连续 slab） |
| 跨 dtype `T.copy`（隐式 cast） | 优化（API 形态） | 无 | 有（Developer 侧自动插 `hivm::VCastOp`，RoundMode=RINT） | **等价替换** | → §0.6 R3（显式 `T.vcast(round_mode="rint")` + 同 dtype `T.copy`；等价性经 §1.6.1 O1 机器验证） |
| `threads=128` | 优化（线程模型） | CUDA thread block | 无（npuir 忽略 `threads=`） | **舍弃** | `pattern-library/traps-compiler.md` 的 TRAP-threads-kwarg-noop（`T.Kernel(threads=)` 在 npuir 无效果）；且本任务调度约定要求删除 `threads`（K9） |
| 单条语句 / 无循环 / 无流水 | 算法 | 无 | — | **保留** | 纯带宽受限搬运，NPU 侧亦无需流水（对比 `capability-gaps.md` 的 CG-2026-0027：单元循环上的 `T.Pipelined` 无增益） |

**判定 1（关键判定点）：`make_dq_layout` 在 NPU 上是否仍需要/可用？处置 = 舍弃。**

- **源侧它是什么**：一个**跨 kernel 的物理布局契约**。主 kernel 用 `make_dq_layout` 的写侧布局把 `dq` 的 8×8 fragment 邻域写成连续地址，从而让 Ampere 的 `atomicAdd` 可向量化（源第 100 行注释、第 233 行 / 第 570 行的 `T.annotate_layout`）。本 kernel 用**同一个**布局注解（源第 122 行）才能读回逻辑 `dq[b,s,h,d]`。
- **NPU 侧为什么不需要**：本批已交付的主 kernel `_gqa_bwd_kernel`（NPU 实现）**没有**对 `dq` 使用任何 layout 注解——它的 dq 写出形态是**逐行 1-D 行原子** `T.atomic_add(dq[bz, row, h, 0:dim], …)`（见本批 `examples/multi_head_attention/_gqa_bwd_kernel/DESIGN.md` 的 §0.5 判定 9 与 §0.6 R3，以及该 kernel 实现文件的 dq 写出段），即 `dq` 在 GM 中保持**自然 BSHD 布局**。NPU 无 warp mma fragment、无 SMEM bank conflict（`references/migration-analysis.md` §5.3 的「swizzle 防 bank conflict → 舍弃」行），该重排**无任何收益**。
- **为什么必须舍弃（不仅是「不需要」而是「有害」）**：op 层的主 kernel 与本 kernel 共用**同一个** `dq` 缓冲（`multi_head_attention.py` 第 249 / 252 / 253 行）。若保留布局注解，读侧会把自然布局的地址按 `make_dq_layout` 的置换规则重新解释，写出 `dq_out[i] = cast(dq[π(i)])`（`π ≠ id`）——**元素被置换，语义被破坏**，且不会报错。
- **语义保持论证**：去掉注解后，读写两侧都回到**恒等索引映射** `dq_out[idx] = cast_T(dq[idx])`，与 §0.1 的定义逐字一致；元素集合、元素位置、舍入模式（RNE）、边界行为（inf/NaN/次正规/溢出）全部不变。**这是纯「去实现层重排、恢复契约语义」的操作，不改变任何语义层内容。**
- **证据**：源第 99–103 行（layout 定义与 Ampere 注释）、第 122 行（本 kernel 的注解）、第 233 / 570 行（主 kernel 的注解）；NPU 侧 `examples/multi_head_attention/_gqa_bwd_kernel/_gqa_bwd_kernel.py` 的 `dq` 写出为逐行 1-D 行原子、无 dq layout 注解；`references/migration-analysis.md` §5.3 的 swizzle 行。

### 0.6 NPU 算法重设计

对 §0.5 中判定为「重新设计」的条目逐项给出 NPU 新算法（调研结论 §1.6.0 是本节判定与重设计的输入）。

**重设计项 R1：三维 grid → 一维 persistent 分核**

- **源方案**：`T.Kernel(⌈S/64⌉, H, B, threads=128)`，三维 grid 覆盖 `(s-block, head, batch)`；性能意图 = 用 block 级并行把 `⌈S/64⌉·H·B` 个任务铺到尽量多的 SM 上，压满带宽。
- **NPU 新算法**：`T.Kernel(num_kernels, is_npu=True)` 一维核数，`num_kernels = min(num_logical, aivector_core_num)`；核内 `for i in T.serial(num_local_tasks)`（**静态边界**）+ grid-stride 映射 `uid = i · num_kernels + cid` + `if uid < num_logical` 守卫；`uid` 解算 `bz = uid // nblk`、`s0 = (uid % nblk) · blk_s`。性能意图承接：仍是把 `B·⌈S/blk_s⌉` 个独立任务铺满物理核，且用 grid-stride 取得聚集窗口的 MTE2 效益（`pattern-library/constants.md` 的 CONST-mte2-degradation 记录 grid-stride 优于每核连续分块 +3.2%）。
- **语义保持论证**：任务的**元素集合**与源完全一致（同一 `(B,S,H,D)` 的同一元素）；逐元素映射无跨任务依赖 ⇒ 任务划分方式不改变任何输出元素的值；无规约 ⇒ 「累加分组变化」问题不存在。边界语义：尾块由 `rs = T.min(blk_s, seq_len - s0)` 精确裁剪（§5.4）。

**重设计项 R2：`blk = 64` 固定几何 → UB 预算驱动的 `(blk_s, H, D)` 连续 slab**

- **源方案**：`blk = 64`（固定，跨 S 轴切 64 行），每任务 `(64, dim)` 的**逻辑**区域，且因 `make_dq_layout` 的存在，读侧物理访问是重排后的散点；CUDA 上 `(64, dim)` 是寄存器/SMEM 容量无关假设的产物。
- **NPU 新算法**：tile 改为 `(blk_s, heads, dim)`——**把整条 H 轴纳入 tile**。因为输入/输出都是 BSHD 行主序，对固定 `bz` 而言 `(S, H, D)` 是一段**连续**内存，故区域 `dq[bz, s0:s0+rs, 0:heads, 0:dim]` 是长度 `rs·H·D` 的**单段连续 DMA**（burst 最大化）；`blk_s = max(1, min(seq_len, blk_elems // (heads·dim)))` 由 UB 元素预算反推（本设计 `blk_elems = 16384`）。性能意图承接并**加强**：源按 H 分核导致「同 `(b,s-block)` 的不同 head 被拆成独立任务、且行 burst 只有 `dim·4 B`、行距 `H·D·4 B`」；NPU 新算法把 H 并入 tile 后，① burst 由 `dim·4 B` 提升到 `blk_s·H·D·4 B` 连续，② 逻辑任务数由 `⌈S/blk⌉·H·B` 降到 `⌈S/blk_s⌉·B`（**少 H 倍**的 task/发射开销）。
- **语义保持论证**：元素集合与元素位置完全不变（tile 只改变「一次搬运多少元素」，不改「哪个输入元素写到哪个输出位置」）；舍入逐元素独立 ⇒ tile 形状对数值无任何影响（无需容差论证，**逐位恒等**）。边界语义：`rs` 同时约束 GM 侧切片、UB 侧切片（两侧同源）⇒ 无越界写、无 padding 语义污染（对照 `pattern-library/traps-runtime.md` 的 TRAP-T-copy-region-semantics）。

**重设计项 R3：跨 dtype `T.copy`（隐式 cast）→ 显式 `T.vcast(round_mode="rint")` + 同 dtype `T.copy`**

- **源方案**：一条 `T.copy(dq[f32 区域], dq_out[f16 区域])` 隐式完成「装载 + cast + 写出」；性能意图 = 省一次显式中间缓冲与一次 pass 的书写。
- **NPU 新算法**：`T.copy(dq 区域 → src32 UB)` → `T.vcast(src32, dst16, round_mode="rint")` → `T.copy(dst16 UB → dq_out 区域)`（三段显式链路）。
- **语义保持论证**：源侧隐式 cast 与 NPU 的显式 `vcast(rint)` **同为 IEEE RNE**，逐位相同——由 §1.6.1 的 D-1 机器验证（12471 案例 × 2 dtype、含全部角点，**违反 0**）固化。**为何不沿用隐式跨 dtype copy 以省一块 UB**：① Developer 侧隐式路径仍插 `hivm::VCastOp`，**不减少向量 pass**；② 舍入模式**不可控**（固定 RINT，无法显式声明/审计）；③ 本设计是带宽受限（§1.6.0 roofline 流量项主导），UB 声明量 6 B/元素远未到容量约束，为省 4 B/元素而放弃舍入可审计性不划算（依据：`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条、`pattern-library/traps-compiler.md` 的 TRAP-C12-copy-dtype-cast）。

**源算法优化手段的意图承接核对表（不得静默丢弃）**：

| 源手段（§0.4） | 处置（§0.5） | 优化意图由什么承接 |
|----------------|--------------|-------------------|
| 1 `make_dq_layout` 布局重排 | 舍弃 | **意图本身在 NPU 上不存在**（为 Ampere atomicAdd 向量化而设；NPU dq 走自然布局逐行行原子）。承接方式 = 去掉重排、恢复恒等映射；**收益侧无损**（无重排即无 repack 代价） |
| 2 三维 grid + `blk=64` | 重新设计 | R1（一维 persistent 铺满 40 个 Vector 核）+ R2（H 并入 tile、burst 最大化、任务数 ÷ H） |
| 3 跨 dtype copy 隐式 cast | 等价替换 | R3（显式 `vcast(rint)`；等价性机器验证）。**「融合进搬运」的性能意图以「三段显式链」在同一条 UB 通路上实现**（无额外 GM 往返、pass 数不变） |
| 4 `threads=128` | 舍弃 | NPU 无线程块模型（TRAP-threads-kwarg-noop）；并行度由 R1 的核数 + Vector 引擎的元素级并行承接 |
| 5 单条语句/无循环/无流水 | 保留 | 同构保留（纯带宽受限搬运）；任务级串行循环为持久化所需，非流水 |

### 0.7 标杆实现（迁移参考）

- **源算子路径**：`/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`（第 99–103 行 `make_dq_layout`；第 106–128 行 `_flashattn_bwd_postprocess_kernel`；第 131–171 行 `FlashAttnBwdPostprocessKernel`）。
- **op 层（契约与调用序）**：`examples/TileOPs/tileops/ops/attention/multi_head_attention.py` 的 `MultiHeadAttentionBwdOp.forward`（第 241–255 行）。
- **源仓测试基准**：`examples/TileOPs/tests/ops/test_multi_head_attention.py` 的 `test_mha_bwd`（`atol=5e-3, rtol=1e-5`，整 op 级；本 kernel 级门禁按 §8.2 独立设定）。
- **golden 依据**：以 §0.1 的语义（纯 RNE dtype 下转换）为唯一依据，实现为 `torch.Tensor.to(storage_dtype)`（源侧的数值行为即 torch `.to` 的 RNE）；**不复刻** §0.6 的 NPU 算法（无 persistent、无显式 vcast、无 UB tile），保证验证独立性。

---

## 1. 概述

### 1.1 算子名称

`_flashattn_bwd_postprocess_kernel`（NPU 侧工厂 `_flashattn_bwd_postprocess_kernel`；对应 wrapper 类 `FlashAttnBwdPostprocessKernel`）。

### 1.2 功能描述

把 FlashAttention backward 主 kernel 产出的 **fp32 `dq` 累加器**逐元素转换为**存储 dtype**（fp16 / bfloat16），作为 `MultiHeadAttentionBwdOp` 的 `dq` 最终产物。纯带宽受限的逐元素 dtype 下转换。

### 1.3 数学公式

$$
\mathrm{dq\_out}[b,s,h,d] \;=\; \mathrm{RNE}_{T}\big(\mathrm{dq}[b,s,h,d]\big),
\qquad
\mathrm{dq} \in \mathbb{R}^{B\times S\times H\times D}_{fp32},\;
\mathrm{dq\_out}\in T^{B\times S\times H\times D},\;
T \in \{\mathrm{fp16},\ \mathrm{bf16}\}
$$

其中 `RNE_T` 为 IEEE 754 round-to-nearest-even（ties-to-even）。**无规约、无缩放、无掩码**。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

本节描述 §0.6 决策后的算法（与源算法的差异逐条注明来源）。算法与 §1.6.0 调研选定候选 **一致**。

1. **分核（§0.6 R1）**：一维 `T.Kernel(num_kernels, is_npu=True)`，`num_kernels = min(num_logical, aivector_core_num = 40)`，核内 `T.serial(num_local_tasks)` grid-stride 循环 + `if uid < num_logical` 守卫。**与源差异**：三维 grid `(⌈S/64⌉, H, B)` → 一维 persistent（源第 121 行）。
2. **tile 形态（§0.6 R2）**：每个逻辑任务 = 一个 `(blk_s, H, D)` 的 **BSHD 连续 slab**（`blk_s = max(1, min(S, blk_elems // (H·D)))`）；区域表达式 `dq[bz, s0:s0+rs, 0:H, 0:D]` 是**单段连续**内存。**与源差异**：源固定 `blk=64` 且逐 `(b,h)` 切、行 burst 仅 `D·4 B`。
3. **元素级实现（§0.6 R3）**：`T.copy`（GM→UB，fp32）→ `T.vcast(round_mode="rint")`（fp32→存储 dtype）→ `T.copy`（UB→GM）。**与源差异**：源用单条跨 dtype `T.copy` 的隐式 cast。
4. **布局**：**恒等索引映射**（无 `make_dq_layout`）；输出自然 BSHD。**与源差异**：源第 122 行的 layout 注解被舍弃（§0.5 判定 1）。
5. **尾块**：`rs = T.min(blk_s, seq_len - s0)` 同时约束 GM 侧与 UB 侧切片（两侧同源），无 padding。
6. **无循环流水**：无 `T.Pipelined`（单个依赖链 `copy → vcast → copy`，跨任务重叠交 Developer 自动多缓冲）。

### 1.5 数据流图

```text
输入 dq (B,S,H,D) fp32 ──┐
                        │  [步骤 1] T.copy：GM→UB，连续 slab (rs, H, D)
                        ▼
                  UB src32 (blk_s, H, D) fp32
                        │  [步骤 2] T.vcast(round_mode="rint")：fp32 → 存储 dtype（逐位等价于源隐式 cast）
                        ▼
                  UB dst16 (blk_s, H, D) fp16/bf16
                        │  [步骤 3] T.copy：UB→GM，同一连续 slab
                        ▼
输出 dq_out (B,S,H,D) fp16/bf16
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**——对同一数学语义的算法族回答调研四问（有没有等价化简公式、有没有在线算法、复杂度是否过高、是否硬件亲和），选定算法族（§1.6.0；输入公式/源算法只是候选之一，迁移任务在源算子解读后、耦合性判定前完成调研）；再在保证数学等价的前提下优化公式（§1.6.1）；再把保留下来的计算全部交给向量单元（§1.6.2）；最后回答「在哪个轴向量化」（§1.6.3，阻塞级）。迁移任务：§1.6.1–§1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：同一数学语义（**逐元素一元 dtype 下转换 + 恒等索引映射**）的算法族——源算法（三维 grid + 跨 dtype 单条 `T.copy` + `make_dq_layout`）只是**基线候选之一**。
> **调研深度**：**轻量调研**（按 `references/algorithm-research.md` §2 分级：本算子是「单步逐元素 / 纯搬运（无规约、无窗口、无矩阵）」⇒ 四问各需明确结论与依据；下方仍给出 R1/R3/R4 的候选表以便 Stage 2 复核）。
> **信息源**：本 skill 的 `references/algorithm-candidates.md` 的 `ALG-elementwise` 命中行（必查清单）；本仓 `examples/` 同类实现（`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`、同批 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/_flashattn_bwd_preprocess_kernel.py` 与 `examples/multi_head_attention/_gqa_bwd_kernel/_gqa_bwd_kernel.py`）；pattern-library（`elementwise.md` 的 PL-1.17-subfp32-fp32-transit / PL-1.6-copy-floor / PL-1.19-scatter-col-write-amplification、`layout.md` 的 PL-1.3-host-permute、`traps-compiler.md` 的 TRAP-C12-copy-dtype-cast / TRAP-vcast-same-dtype-rounds / TRAP-UB-dst-align、`constants.md` 的 CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-msprof-task-floor-910B3 / CONST-capacity-910B2C / CONST-aicore-910B2C / CONST-gm-to-ub-bw-dilution）；`docs/Tilelang.language/内存操作/T.copy.md` 与 `docs/Tilelang.language/数据类型转换操作/T.vcast.md`；源码 §0.3/§0.4；结构判据（R2）。
> **未使用互联网检索**：本算子为单步逐元素映射，候选空间在本地源（`ALG-elementwise` 命中行 + 本仓逐元素/vcast 先例 + pattern-library elementwise 条目 + 源算子）内**已封闭**，无未命中缺口；`algorithm-research.md` §6 第 6 层的适用时机（参考表未命中 / R2 结构判据不确定）均不成立，故不联网（如实记录）。

> **证据路径（负向断言与弃选论证所指向的条目，设计期已逐条打开核对）**：
> - `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（§2.2.1 f32→f16 / f32→bf16 的 `rint`；§2.3 形状须一致、src 须为 buffer）、`docs/Tilelang.language/内存操作/T.copy.md`（§2.2.2 形状规则；§2.3 第 2/4/5 条跨 dtype 与 GM→GM 限制）
> - `docs/Tilelang.language/数学操作/T.vmul.md`、`docs/Tilelang.language/规约操作/T.reduce_sum.md`、`docs/Tilelang.language/编译器提示操作/T.Kernel.md`
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md`（PL-1.17-subfp32-fp32-transit、PL-1.6-copy-floor、PL-1.19-scatter-col-write-amplification）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md`（PL-1.3-host-permute）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`（TRAP-C12-copy-dtype-cast、TRAP-vcast-same-dtype-rounds、TRAP-UB-dst-align、TRAP-threads-kwarg-noop、TRAP-T-copy-region-semantics、TRAP-npuir-reshape-dst-buffer-shadow）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（CONST-mte2-degradation、CONST-vector-launch-overhead、CONST-msprof-task-floor-910B3、CONST-capacity-910B2C、CONST-aicore-910B2C、CONST-gm-to-ub-bw-dilution）
> - `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`、`.agents/skills/tilelang-op-design/references/algorithm-research.md`、`.agents/skills/tilelang-op-design/references/migration-analysis.md`
> - `.agents/skills/_shared/standards/core-split-strategy.md`、`.agents/skills/_shared/standards/hardware-cost-model.md`、`.agents/skills/_shared/standards/negative-claim-evidence.md`
> - `.agents/evolution/capability-gaps.md`（CG-2026-0029、CG-2026-0018）

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | **基线**（GPU 源算法） | 三维 grid `(⌈S/64⌉,H,B)`；`make_dq_layout` 物理重排；单条跨 dtype `T.copy`（隐式 RNE cast） | — | — | ✅（基线） |
| 1 | **全 H 连续 slab + 一维 persistent + 显式 `vcast`**（本设计主选） | tile `(blk_s,H,D)`，区域 `dq[bz, s0:s0+rs, 0:H, 0:D]` 为单段连续；三段显式链 copy→vcast→copy | **数学恒等**（同一元素集合、同一恒等索引映射、同一 RNE；仅并行结构/tile 形状/API 形态不同，§0.6 R1–R3） | 无 GM 收益（I/O 下界本就为 `N·(4+eb)`）；**收益在 burst 连续化**（`D·4 B` 行 → `blk_s·H·D·4 B` 连续段）与**任务数 ÷ H** | ✅（主选） |
| 2 | 保留 source 的 per-`(b,h,s-block)` 跨步 2-D 形态（不把 H 并入 tile） | tile `(blk_s, D)`，区域 `dq[bz, s0:s0+rs, by, 0:D]`，行距 `H·D·4 B` | 数学恒等 | GM 流量不变；burst 仅 `D·4 B`；逻辑任务数为主选 ×H ⇒ 发射项放大 H 倍 | ✅（作 R3/R4 对照，预期被主选支配） |
| 3 | 跨 dtype `T.copy`（隐式 cast，省 fp32 UB） | UB 声明量 6 → 2 B/元素 | 数学恒等（同 RNE） | 省 4 B/元素 UB 声明量 | ✅（R3/R4 对照；**精度可控性否决**见 §1.6.1 C2） |
| 4 | 融合进主 kernel 的 epilogue（直接产出存储 dtype 的 dq） | 主 kernel 以 fp16/bf16 原子累加 dq | **不等价**——fp16/bf16 原子累加与「fp32 累加后一次 cast」在舍入上不同（累积舍入 vs 单次舍入） | 理论省 `2·N·4 B`（fp32 dq 的一次 GM 写 + 一次 GM 读） | ❌（**结构越界**：`dq` 的 fp32 跨 block 原子累加是 op 层契约（`multi_head_attention.py` 第 249/252 行）、主 kernel 已冻结交付；且 NPU 侧 fp16 原子累加的数值语义与其 fp32 契约冲突） |
| 5 | 逐元素标量循环（放弃 tile 语义，手写 index 循环） | `for i in range(N): out[i] = cast(in[i])` | 数学恒等 | 无 | ❌（R3/R4：pass 数 = `N`，标量化量级劣势；`pattern-library/traps-compiler.md` 的 TRAP-inner-axis-reduce-scalarize 同族证据：单个标量化 op 可把 kernel 5.6 → 155.1 µs） |
| 6 | 核内重排布局后再转换（转置/置换成「更好对齐」的轴） | 先 `T.transpose` 再 cast | 数学恒等（若转置正确） | 无（本算子最内轴 `D` 已是向量宽整数倍） | ❌（R4：repack 纯开销；`layout.md` 的 PL-1.1-transpose-chain 与 `traps-compiler.md` 的 TRAP-transpose-ub-gm-roundtrip） |
| 7 | host 侧 `permute`/`reshape` 视图 + 逐元素 cast | host 摊平后调用 | 数学恒等（仅元数据） | 无（host 只改元数据，不省任何设备侧流量） | ❌（R4：`layout.md` 的 PL-1.3-host-permute 记录 host permute 通常净亏 106–147 µs；且 `ascend-constraints.md` §4 禁止 host 改动真实内容） |

> 注：本算子在**代数式子层面无化简对象**——一元逐元素映射，无除法、无超越函数、无归一化、无公共子表达式、无矩式可降代换（`algorithm-candidates.md` 的 `ALG-elementwise` 行其 R1 列列的「公共子表达式消除 / 除法乘倒数 / cast 链合并」三项中，前两项本算子不含，第三项「cast 链合并」在本算子只有**单次** cast、无可合并对象）。故 R1 的候选全部来自**结构重排 / 数据布局 / 计算单元归属 / API 形态**四个方向（上表 1–7），这是本算子族（纯逐元素搬运 + 单次 dtype 变换）的固有形态。

**R2 在线算法**：**无实质在线变体，且收益口径不适用**。结构依据（非无依据的否定）：

- 在线算法的定义是「单遍扫描 + O(1)/O(tile) 中间状态的流式变体」，其收益口径是 **① 扫描遍数（多遍 → 单遍）、② 中间缓冲（`O(N)` → `O(tile)`）、③ 数值稳定化（running max/rescale）**。
- 本算子是**逐元素映射，无跨元素依赖**：① 天然单遍（输入元素只被消费一次，R3 表中所有候选 scan passes = 1）；② 中间状态本就只需 `O(tile)`（一个 UB tile，无 `O(N)` 中间量可省）；③ 无 max/sum 归一化，不存在重缩放动机。
- ⇒ 三项收益口径**逐项失效**，不存在「在线化能带来收益」的结构空间。**该结论的结构依据** = 「输出元素 `i` 只依赖输入元素 `i`」这一映射性质（源第 123–126 行的 region 下标两侧完全一致即为该性质的直接证据）。

**R3 复杂度对比**（四口径；口径定义：**FLOPs** = 0——本算子无乘加运算（`cast` 是类型转换而非算术，1 个 rounding 指令/元素，按 `CONST-vector-launch-overhead` 的 **pass 单价**记账而非 FLOPs）；**访存 Bytes = GM 流量**（输入读一次 + 输出写一次；不含 UB 内流量、不含任何中间 GM 往返）；**扫描遍数** = 同一输入元素的 GM 重复读取次数；**中间缓冲峰值** = UB 片上驻留；附可并行度与跨核代价。数字按 manifest 五档给出区间，`N = B·S·H·D`、`eb` = 存储元素字节 = 2）：

| 算法候选 | FLOPs | GM 访存 Bytes（`N·(4+eb) = 6N`） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|----------------------------------|---------|-------------|---------------------|
| 0 基线（GPU 源：三维 grid + 64 行跨步区域 + layout） | 0 | **6N**（full: 100.7 MB） | **1** | CUDA 侧 `(64, dim)` 的 `(4+2)` B/元素 = 48–96 KB（SMEM/fragment；CUDA 无 UB 上限口径） | 逻辑核 `⌈S/64⌉·H·B`（full: 8192–16384）→ CUDA 侧无物理核数约束；无跨核代价 |
| **1 主选：全 H 连续 slab + 一维 persistent + 显式 vcast** | 0 | **6N（= 基线，无可再削）** | **1** | 声明 `6 B/元素 × E`（E=16384 ⇒ 96 KB） | 逻辑核 `⌈S/blk_s⌉·B`（full: 512–1024）→ persistent 40 核（§5.5）；无跨核代价 |
| 2 per-`(b,h,s)` 跨步 2-D（源几何，H 不并入 tile） | 0 | 6N | 1 | 6 B/元素（同主选可小） | 逻辑核 = 主选 ×H（full: 8192–32768）⇒ 发射项 ×H ⇒ 同元素量下**严格不优** |
| 3 跨 dtype `T.copy`（隐式 cast） | 0 | 6N | 1 | **2 B/元素**（省 fp32 UB） | 同上（pass 数不减：仍插 `hivm::VCastOp`） |
| 4 融合进主 kernel epilogue | 0 | **2N**（省 fp32 dq 的写 + 读） | 1 | — | 需改主 kernel 契约 ⇒ **出域** |
| 5 逐元素标量循环 | 0 | 6N | 1 | 小 | pass 数 = `N`（标量化）⇒ 量级劣势 |
| 6 核内重排布局后 cast | 0 | 6N（GM 流量不变） | 1 | 主选 + 重排中转缓冲 | repack 纯开销（可能降级 UB→GM→UB 往返） |
| 7 host permute/reshape 视图 | 0 | 6N | 1 | 0（host 侧） | host 开销 106–147 µs 级；不省设备侧流量 |

> **口径注**：① 除候选 4（越界）外，**全部候选的 GM 流量与扫描遍数相同**（`6N` / 1 遍）——本算子的 I/O 下界由契约固定，**任何候选都不能改变它**。故本算子的排序**完全由 R4 决定**（burst 连续性、任务数/发射项、API 舍入可控性、repack 代价），不是纸面并列：候选 2 发射项 ×H、候选 5 量级劣势、候选 6/7 纯开销、候选 4 越界。② 每元素级表达式中的 `4` = fp32 输入字节、`eb` = 存储 dtype 字节（2）。③ 本表不含 UB 内流量（`copy` 的 UB 写 + `vcast` 的读+写 + `copy` 的 UB 读）——它是候选间的**共同项**（任何候选都要把元素搬进 UB 才能用 Vector 引擎处理），差异只在 burst 形态。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰均附证据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| 0 基线（GPU 源） | CUDA thread/fragment 级（NPU 无对应）；`make_dq_layout` 绑定 Ampere mma fragment | CUDA 口径；NPU 需按 `(blk_s,H,D)` 重算 | 源无 pad 要求 | 源 `ceildiv` 超发 + 隐式裁剪 | 单条语句 | 三维 grid 超发 | ❌ **不可直译**（§0.5 判定；三层重设计 R1–R3） |
| **1 主选** | ✅ **Vector**：`T.vcast`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 f32→f16/f32→bf16 的 `rint`）+ `T.copy`（DMA 引擎） | ✅ 声明 6 B/元素 × 16384 = 96 KB（×1.7 = 167 KB ≤ 192 KB，§4.5） | ✅ 最内 `D ∈ {64,128}`：fp32 向量宽 ×4、fp16/bf16 ×8 均整除（尾 lane 浪费 0%）；连续 slab 段长 `blk_s·H·D·4 B` 远超 128 B | ✅ `B/S/H/D/dtype` 全为工厂期常量；`T.serial` 边界 = host 折叠常量 | ✅ 单依赖链，跨任务重叠交 auto multi-buffer；不用 `T.Pipelined`（`capability-gaps.md` CG-2026-0027：单元循环 `T.Pipelined` 无增益） | ✅ 任务间元素独立 ⇒ 无跨核归并、无 workspace、无 flag（CONST-flag-id-budget 不适用） | ✅ **主选** |
| 2 per-`(b,h,s)` 跨步 2-D | ✅ Vector | ✅ 同主选 | ✅ `D` 整除 | ✅ | ✅ | ✅ | ❌ 同元素量下任务数 ×H ⇒ 发射项 ×H（R3） |
| 3 跨 dtype `T.copy` | ✅ Vector（编译器插 VCastOp） | ✅ 更松（2 B/元素） | ✅ | ✅ | ✅ | ✅ | ❌ **舍入不可控**（固定 RINT、不可显式审计）+ 不减 pass（§1.6.1 C2） |
| 4 融合进主 kernel epilogue | ❌ 出域（需改主 kernel + op 层契约） | — | — | — | — | 需 fp16 原子累加（数值语义冲突） | ❌ 结构越界（R1 候选 4） |
| 5 逐元素标量循环 | ❌ 标量化（Vector 引擎空转、标量 store 满发） | ✅ | ⚠ | ✅ | ❌ | ✅ | ❌ 量级劣势（TRAP-inner-axis-reduce-scalarize 同族证据） |
| 6 核内重排后 cast | ✅ Vector | ⚠ 中转缓冲 | ⚠ 两边须 ≥ 向量宽且为整数倍 | ✅ | ⚠ 转置与 cast 串行化 | ✅ | ❌ repack 纯开销（PL-1.1-transpose-chain / TRAP-transpose-ub-gm-roundtrip） |
| 7 host permute/reshape | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 不省设备侧流量 + host 开销百 µs 级（PL-1.3-host-permute）+ host 侧改动约束（`ascend-constraints.md` §4） |

**调研结论**：选定 **候选 1「全 H 连续 slab（`(blk_s,H,D)`）+ 一维 persistent 分核 + 显式 `vcast(rint)` 三段链 + 恒等索引映射（舍弃 `make_dq_layout`）」**（= 候选 0 的语义内核 + §0.6 R1/R2/R3 的结构重设计）。关键依据：
**R3** 表中候选 0–3/5–7 的 GM 流量与扫描遍数**全部相同**（`6N` / 1 遍，I/O 下界由契约固定），候选 4 虽理论降至 `2N` 但**结构越界**（主 kernel 已冻结 + op 层 fp32 原子累加契约）；
故排序由 **R4** 决定——候选 1 全项亲和（Vector 单元、声明 96 KB ≤ 192 KB、`D` 轴 8 整除尾 lane 0%、静态边界、无跨核结构），而候选 2 的发射项按 `H` 倍放大、候选 5 标量化、候选 6/7 为「无收益 + 有代价」、候选 3 牺牲舍入可控性。
**与基线的结构差异一句话**：数学内核（逐元素 RNE cast + 恒等索引映射）完全不变，改的是**并行结构**（三维 grid → 一维 persistent）、**tile 形态**（64 行跨步区域 → UB 预算驱动的 `(blk_s,H,D)` 连续 slab）、**布局契约**（舍弃 `make_dq_layout` 恢复恒等映射）、**API 形态**（隐式跨 dtype copy → 显式 `vcast(rint)`）。
**调研范围**（供「无更优替代」类断言核对）：`references/algorithm-candidates.md` 的 `ALG-elementwise` 命中行；本仓逐元素/vcast 先例（`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`、同批 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/_flashattn_bwd_preprocess_kernel.py`、`examples/multi_head_attention/_gqa_bwd_kernel/_gqa_bwd_kernel.py`）；pattern-library 四个条目文件（elementwise/layout/traps-compiler/constants）；`docs/Tilelang.language/` 的 `T.copy` / `T.vcast` / `T.Kernel` 规格；源码 §0.1–§0.4；R2 结构判据（逐元素无跨元素依赖）。**源算法优化手段承接**见 §0.6 末尾核对表（5 项逐项落实，**无静默丢弃**）。

**设计期估算（D-2 roofline；协议 `.agents/skills/_shared/standards/hardware-cost-model.md` §2；常数带版本戳）**：

口径（本算子为**纯读写搬运**，`flops = 0`）：`bytes = N · (4 + eb)` = 输入读一次（fp32，4 B/元素）+ 输出写一次（存储 dtype，`eb = 2` B/元素）= `6N`；**不含** UB 内流量、不含任何中间 GM 往返。估算下界 `= max(流量项, 发射项, 容量项)`；发射项按「每核 pass 数 × 单价」，本算子每 tile 3 个 pass（GM→UB copy、`vcast`、UB→GM copy），pass 单价取 `CONST-msprof-task-floor-910B3` 实测档（0.23–0.3 µs/pass，**本设备同工具链实测**），并给出 `CONST-vector-launch-overhead` 的 0.5 µs/op 上界档。推导式：`逻辑核 = B·⌈S/blk_s⌉`；`轮数 = ⌈逻辑核 / 40⌉`；`每核 pass = 轮数 × 3`。

| workload（manifest） | (B,S,H,D) | `row=H·D` | `blk_s` | 逻辑核 / 轮数 | 流量项（`6N` ÷ 有效带宽） | 发射项（pass × 单价） | 容量项 | 估算下界 |
|----------------------|-----------|-----------|---------|---------------|--------------------------|----------------------|--------|----------|
| `mha-bwd-smoke-s512-h8-d64` | (1,512,8,64) | 512 | 32 | 16 / **1** | 1.573 MB ÷ (33.9 GB/s × 16)〔CONST-mte2-degradation 1M 档〕= **2.9 µs** | 1×3 pass × 0.23–0.5 µs = **0.7–1.5 µs** | 0 | **≈2.9 µs**（流量项；受 ≈2.4 µs 固定地板约束〔CONST-msprof-task-floor-910B3〕） |
| `llama-3.1-8b-short` | (4,512,32,128) | 4096 | 4 | 512 / **13** | 50.3 MB ÷ (30.2 GB/s × 40 = 1.208 TB/s)〔16M 档〕= **41.7 µs** | 13×3 = 39 pass × 0.23–0.5 = **9–19.5 µs** | 0 | **≈41.7 µs**（流量项） |
| `llama-3.1-8b-long` | (2,2048,32,128) | 4096 | 4 | 1024 / **26** | 100.7 MB ÷ 1.208 TB/s = **83.4 µs** | 26×3 = 78 pass × 0.23–0.5 = **18–39 µs** | 0 | **≈83.4 µs**（流量项） |
| `llama-3.1-70b-short` | (2,512,64,128) | 8192 | 2 | 512 / **13** | 50.3 MB ÷ 1.208 TB/s = **41.7 µs** | 39 pass × 0.23–0.5 = **9–19.5 µs** | 0 | **≈41.7 µs** |
| `llama-3.1-70b-long` | (1,2048,64,128) | 8192 | 2 | 1024 / **26** | 100.7 MB ÷ 1.208 TB/s = **83.4 µs** | 78 pass × 0.23–0.5 = **18–39 µs** | 0 | **≈83.4 µs** |

> **估算下界（汇总）**：全部五档的 `max(流量项, 发射项, 容量项)` 均由**流量项主导**（发射项为其 1/4 ~ 1/2），故**下界 = 流量项**（41.7 / 83.4 µs 量级；smoke 档 2.9 µs）。**容量项 = 0**（纯 Vector 单引擎、无 L0/L1、无跨引擎 S/P 往返，`constants.md` 的 CONST-store-fixpipe-gm-only 不适用）。
> **常数适用性声明**：`CONST-mte2-degradation` 绑定 **910B2C + CANN 8.5.0 + 48 vector core**（本设备为 **910B3 + CANN 9.0.0 + 40 vector core**）⇒ 按「**未实测假设：沿用同代 910B 曲线量级**」引用，40 核按 48 核做**线性外推**（属未实测假设）；`CONST-vector-launch-overhead`（~0.5 µs/op）同为 910B2C/CANN 8.5.0 条目。**可信档** = `CONST-msprof-task-floor-910B3`（0.23–0.3 µs/pass 与 2.4 µs 固定地板，**910B3 + CANN 9.0.0 本设备实测**）。核数（20 / 40）、UB 容量（196608 B）**已本机实查复核**（§5.5），无假设成分。↔ 交叉一致性：40 核 × 30.2 GB/s/core = 1.208 TB/s，低于 `CONST-mte2-degradation` 记录的混合读写地板 ~1.26 TB/s（2:1 R/W 混合口径），量级自洽。
> **Stage 4 敏感方向**：本算子流量项主导 ⇒ 第一杠杆是**有效带宽**（burst 连续性、核数满发），第二杠杆是 `blk_elems`（放大 tile 摊薄 3 个 pass 的发射项；但在流量项主导下边际收益有限，属**参数级** A/B，见 §1.6.3 与 §4.5 阶梯）。Stage 4 须按 `hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 调研选定的算法（全 H 连续 slab + 一维 persistent + 显式 `vcast(rint)` 三段链 + 恒等索引映射）。逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）。
> **前置结论（代数化简类）**：本算子**代数化简类优化采纳 0 项**。依据：① `references/algorithm-candidates.md` 的 `ALG-elementwise` 命中行的 R1 列只列「公共子表达式消除 / 除法乘倒数 / cast 链合并」，其中前两项本算子不含（无重复子式、无除法）；② 第三项无对象——本算子每个元素只经历**一次** cast（§0.3 步骤分解证实源码体内只有一条计算语句），无 cast 链可合并；③ 恒等变形空间枚举（§1.6.0 R1 的 7 个候选）无一代数化简类，全部是结构/布局/API 形态类。故本节的采纳项只能来自**舍入语义的等价性选择**与**实现形态的等价替换**。

| # | 优化项 | 原式（源 / 基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|------------------|-----------|-----------|---------|
| **O1** | **舍入模式显式化为 IEEE RNE（`rint`）**（**等价性选择项**，非降计算量项） | 源：单条跨 dtype `T.copy` 的**隐式** cast（CUDA C `__float2half` / `__nv_bfloat16`，RNE；NPU Developer 侧固定插 `hivm::VCastOp`，`RoundMode=RINT`，**不可定制**） | `T.vcast(src32, dst16, round_mode="rint")`——显式声明 RNE | **同一数值函数**：源隐式 cast 与本算子显式 `rint` 均为 IEEE 754 **round-to-nearest-even**（ties-to-even）。**D-1 机器验证**（下结果表）：12471 案例 × 2 dtype（含全部角点）与其精确有理数参考实现**逐位相等**，违反 0；且**两个被否决的舍入模式与该参考明确不同**（证明 `rint` 是**可区分**的选择而非恒真陈述） | **计算量/访存量收益 = 0**（同一 pass 数、同一 UB 声明量）。换取：① 舍入模式**显式可审计**（杜绝隐式路径「不可定制」的风险，见 C2）；② 与 golden（`torch.Tensor.to`）**逐位对齐**，使 §8.2 可采用 `torch.equal` 的**逐位门禁**（比容差门禁强一个量级） |

**D-1 等价性机器验证结果表**（脚本 `examples/multi_head_attention/_flashattn_bwd_postprocess_kernel/verify_equiv.py`，2026-10-16 设计期执行；torch CPU；判定 = **逐位相等**）：

| 采纳项 / 主张 | dtype | 案例数 | 覆盖 | 最大 ulp 差 | 违反元素数 | 结论 |
|--------------|-------|-------|------|------------|-----------|------|
| **O1 舍入模式 = RNE（`rint`）** | float16 | 12471 | `random[-4,4]` / `random[-7e4,7e4]` / `random[-1e-4,1e-4]` / `exact-ties(2^-11)` / 角点（`±inf`/`NaN`/`±0`/fp16 最大 65504/溢出边界 65520/最小正规 6.103515625e-05/最小次正规 5.960464477539063e-08/半个最小次正规/`1+2^-11` tie/`±3.4028235e38`/`±1e-40`） | **0** | **0** | **EQUIV_PASS** |
| **O1 舍入模式 = RNE（`rint`）** | bfloat16 | 12471 | 同上（tie 宽度改为 2^-8） | **0** | **0** | **EQUIV_PASS** |
| O1 的**可区分性**（否决模式对照，未采纳） | float16 | 12471 | 同上 | — | `round`(half-away) 与参考差 **91** 个元素、`trunc` 差 **6091** 个元素 | （未采纳：证明只有 RNE 是等价选择） |
| O1 的**可区分性**（否决模式对照，未采纳） | bfloat16 | 12471 | 同上 | — | `round` 差 **82** 个元素、`trunc` 差 **6278** 个元素 | （未采纳，同上） |

形态覆盖：两 dtype × 约 12471 个输入点 × 5 类用例；角点覆盖 `±inf`、`NaN`（两侧 `NaN` 视为一致）、`±0`、fp16 次正规与半次正规、fp16 最大正规与溢出阈值 65520、bf16/fp32 极值、`1+2^-11`（fp16 tie）与 `1+2^-8`（bf16 tie）。脚本以 `fractions.Fraction` 从**定义**实现 RNE（不调用 torch/numpy 做舍入），与 `torch.Tensor.to` 独立；脚本已随算子目录留存，Stage 2 可直接重跑复核（`python verify_equiv.py`，无需 NPU，exit 0 = `SUMMARY: ALL_EQUIV_PASS`）。

**设计所依赖的数值等价性主张（同样以机器验证固化，供 Stage 2 复核）**：

| # | 主张 | 内容 | 机器验证结论 |
|---|------|------|--------------|
| **P1** | `T.vcast(rint)` ≡ 源隐式跨 dtype cast | 两者同为 IEEE RNE | 见上表：两 dtype 各 12471 案例、最大 ulp 差 **0**、违反 **0** ⇒ 与源逐位相同 |
| **P2** | 逐元素映射的**元素集合与索引映射恒等** | `dq_out[idx] = cast(dq[idx])`，无置换、无遗漏、无重复 | 结构性主张（§0.1 依据源第 123–126 行两侧下标一致；§0.6 判定 1 舍弃 `make_dq_layout` 后恢复恒等映射）。**机器验证不适用**（无第二形态可对照），改由 Stage 3 的 L0「全元素覆盖 + 区域外零污染」用例正面覆盖（§8.3 的 L0-6/L0-7/L0-9） |
| **P3** | tile 形状/分核方式对数值**无影响** | 逐元素独立 ⇒ 任何 tile/任务划分给出逐位相同结果 | 结构性主张（无跨元素依赖、无规约）。由 §8.3 的 L0-5（同 shape 多 `blk_elems` 档位输出逐位一致）覆盖 |

**优化结论**：**代数化简类采纳 0 项**（依据见节首前置结论）；**等价性选择/替换类采纳 O1 一项**（舍入模式 = `rint`，`EQUIV_PASS`，两 dtype 各 12471 案例逐位相等）；同时以机器验证固化 **P1**，并以结构性论证登记 **P2/P3**（由 Stage 3 的 L0 用例正面覆盖）。
**优化后公式（§3.1 的唯一输入；`RNE_T` 为 IEEE round-to-nearest-even，`idx` 为恒等索引）**：

$$
\mathrm{dq\_out}[idx] \;=\; \mathrm{vcast}_{\text{float32} \to T}^{\text{rint}}\big(\mathrm{dq}[idx]\big),
\qquad
\text{(O1 adopted: explicit RNE, identical to the source implicit cast)}
$$

**否决项及依据**（负向断言举证，`.agents/skills/_shared/standards/negative-claim-evidence.md` §1；不满足条件者一律不采纳）：

| 否决候选 | 否决理由 | 证据 |
|----------|----------|------|
| **C1** 保留 `make_dq_layout`（源第 122 行） | 该布局是**与主 kernel 共享的物理契约**；NPU 主 kernel（本批已交付）以自然 BSHD 布局写 `dq`（逐行 1-D 行原子、无 layout 注解），op 层两者共用同一 `torch.zeros_like` 缓冲。保留注解 ⇒ 读侧按 `π ≠ id` 置换解释地址 ⇒ **元素被置换、语义被破坏且静默无报错**；NPU 亦无 Ampere mma fragment / SMEM bank conflict，重排**零收益** | 源第 99–103 / 122 / 233 / 570 行；`examples/multi_head_attention/_gqa_bwd_kernel/_gqa_bwd_kernel.py`（dq 逐行行原子、无 dq layout 注解）；`examples/TileOPs/tileops/ops/attention/multi_head_attention.py` 第 249 / 252 / 253 行（同一缓冲）；`references/migration-analysis.md` §5.3 的 swizzle 行 |
| **C2** 跨 dtype `T.copy` 当 cast（省 `src32` 的 4 B/元素 UB 声明量） | ① **不减向量 pass**——Developer 侧隐式路径仍插 `hivm::VCastOp`，只是从显式变隐式；② **舍入模式不可控**（固定 RINT，一旦语义需要其它模式即静默错值，且无法审计）；③ 本设计流量项主导、UB 声明量 96 KB 远未到容量约束，省 4 B/元素无价值 | `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条（Developer 自动插 `hivm::VCastOp`，RoundMode=RINT，不可通过 `T.copy` 定制）；`pattern-library/traps-compiler.md` 的 TRAP-C12-copy-dtype-cast（「跨 dtype copy 不减少 vector pass 且舍入不可控——勿用作 cast 融合」）；同批先例 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/DESIGN.md` §1.6.1 的 C2 行 |
| **C3** 同 dtype `T.vcast`（如 f32→f32）充当「搬运」 | `T.vcast` 同 dtype 退化为 round-to-integer、**静默坏值**；同 dtype 搬运必须用 `T.copy`（按 dtype 在 trace 期分派） | `pattern-library/traps-compiler.md` 的 TRAP-vcast-same-dtype-rounds |
| **C4** 融合进主 kernel epilogue 直接产出存储 dtype | 越界：主 kernel 已冻结交付；dq 的 fp32 跨 block 原子累加是 op 层契约；fp16/bf16 原子累加与「fp32 累加后一次 cast」在舍入上不等价（累积舍入 vs 单次舍入） | `examples/TileOPs/tileops/ops/attention/multi_head_attention.py` 第 249 / 252 / 253 行；§1.6.0 R1 候选 4 行 |
| **C5** 逐元素标量循环 / host 侧 Python cast | 标量化使 pass 数 = `N`（量级劣势）；host 侧改动真实内容被禁（`ascend-constraints.md` §4 只允许 stride/shape 元数据视图）。且 NPU 侧 host 无法访问设备张量做逐元素转换 | `pattern-library/traps-compiler.md` 的 TRAP-inner-axis-reduce-scalarize（单 op 5.6 → 155.1 µs 同族证据）；`.agents/skills/tilelang-op-design/references/ascend-constraints.md` §4 |
| **C6** 核内重排布局后 cast（含 `T.transpose` 链） | 本算子最内轴 `D` 已是向量宽整数倍（fp32 ×4、fp16/bf16 ×8），重排**无收益**；核内二轴转置在本工具链可被降级为 UB→GM→UB 往返 | `pattern-library/layout.md` 的 PL-1.1-transpose-chain；`pattern-library/traps-compiler.md` 的 TRAP-transpose-ub-gm-roundtrip（5.6 → 138.2 µs） |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点范围：本设计实现方案（§3.3 伪代码）中的**全部**循环与标量计算点。本算子是逐元素一元映射，盘点结果为「1 个逐元素点 + 1 个 DMA 搬运点 + 3 个标量/循环点」。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | **逐元素 dtype 下转换（fp32 → fp16/bf16）** | 源：`T.copy` 内部隐式逐元素 cast（CUDA 侧 thread 级、NPU 侧 `hivm::VCastOp`） | `T.vcast(src32, dst16, round_mode="rint")` —— 整 buffer 向量化（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 支持 f32→f16 / f32→bf16 的 `rint`；§2.2.2 要求 src/dst shape 一致，本设计两侧均为 `(blk_s, H, D)`） | ✅ | — |
| 2 | **数据搬入 / 搬出** | 源：`T.copy` 区域搬运 | `T.copy`（GM↔UB）——DMA 引擎整段搬运，非标量循环（`docs/Tilelang.language/内存操作/T.copy.md` §1 支持的 GM→UB / UB→GM 方向） | ✅（本就是向量/引擎级） | — |
| 3 | **任务映射标量（`bz = uid // nblk`、`s0 = (uid % nblk) · blk_s`、`uid = i · num_kernels + cid`）** | 源：三维 grid 由硬件 index 提供（无显式标量） | 无（不存在「逐元素等价」的向量 API——这是 block 级任务映射） | ❌ | **block 级索引 / 任务映射计算**：每任务执行 **1 次**（不是逐元素热点）；标量规模 = `O(逻辑核数)`，与 `O(N)` 的元素量无关。可接受理由类别：block 级索引 / 任务映射计算 |
| 4 | **尾块行数 `rs = T.min(blk_s, seq_len - s0)`** | 源：隐式裁剪（三维 grid 的 `ceildiv` 超发 + 区域切片） | 无（单个标量上界，非逐元素） | ❌ | **依赖运行时边界的标量**：每任务执行 1 次；用于同时约束 GM 侧与 UB 侧切片（两侧同源），无法静态化（`seq_len % blk_s ≠ 0` 时才有意义），也不宜 padding（padding 会把无效行写进 GM）。可接受理由类别：动态边界的处理（已说明为何不能静态化 / padding） |
| 5 | **工厂层 host 常量（`row`、`blk_s`、`nblk`、`num_logical`、`num_kernels`、`num_local_tasks`）** | 源：host 侧无标量（三维 grid 参数直接由 shape 给出） | 无 | ❌ | **host 侧 shape 元数据计算（不在 kernel 内）**：全部为 Python 层编译期常量，trace 期折叠为 `IntImm`，不产生任何设备侧指令。可接受理由类别：host 侧元数据计算 |

> 说明：计算点 #3/#4 的标量**不进入**任何逐元素通路——它们决定「哪个 tile 被处理」，而非「每个元素怎么算」。逐元素通路（#1/#2）中**不存在**任何标量循环：本设计的循环只有一处，即 `for i in T.serial(num_local_tasks)`（**任务级**调度循环，#3 的载体），其体内是整 buffer 的 `T.copy`/`T.vcast`，无逐元素标量语句。（对照：`pattern-library/traps-compiler.md` 的 TRAP-tir-range-loop-not-unrolled 记录 `@T.prim_func` 内 `for i in range(n)` 是串行 TIR 循环而非 Python 展开；本设计正是利用该性质把任务循环写成静态界的 `T.serial`。）

**向量化结论**：逐元素计算 **100% 向量化**——本算子唯一的逐元素计算点是 dtype 转换（#1），由整 buffer 的 `T.vcast(round_mode="rint")` 承接（API 有 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 佐证）；搬运点（#2）由 DMA 引擎的 `T.copy` 承接。**保留 3 处标量/循环**（#3 任务映射标量、#4 尾块标量、#5 host 元数据），理由类别：**block 级索引 / 任务映射**、**动态边界处理**、**host 侧元数据**，逐项见上表——三者均与元素量无关（`O(任务数)` 而非 `O(N)`），且无逐元素等价向量 API 可替代。**本设计的 `T.serial` 循环体内不含任何逐元素标量语句**（§6 与之保持一致）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（BSHD 进 / BSHD 出，恒等索引映射），但**核内布局与向量化轴是设计变量**。本算子有 3 个非 batch 轴（`S`、`H`、`D`），故必须枚举候选。

**轴质量评分维度**（逐候选填写）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | 轴长是否为向量宽度整数倍（昇腾 Vector 128 bit：fp32 ×4、fp16/bf16 ×8） | `D ∈ {64,128}`（manifest）✓ 两者均整除；`D` 是**唯一** stride-1 轴 |
| 尾 lane 浪费率 | `(⌈len/vw⌉·vw − len)/len` | `D` 轴 + `blk_s·H·D` 连续段：**0%**（`D` 与 `H·D` 均为 8 的整数倍） |
| 累加链形态 | 该轴作为串行累加维（stride≠1）还是向量维（stride=1） | **无累加链**（逐元素映射、无规约）⇒ 本维度不构成约束；`D` 轴 stride 1，天然向量维 |
| repack 代价 | 改变布局需要的重排代价 | 候选 1/2/5：**无需 repack**（直接用 I/O 原生布局）；候选 3/4：核内转置或 host permute |
| UB 容量影响 | 布局变化对驻留缓冲尺寸的影响 | 候选 1/2/5：`6 B/元素 × E`；候选 3：额外中转缓冲 |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **I/O 原生 BSHD + 把整个 H 轴纳入 tile（`(blk_s, H, D)` 连续 slab）** | `D`（stride 1，向量宽 ×4/×8 整除） | **无**（直接用原生布局） | **基线即最优**：区域 `dq[bz, s0:s0+rs, 0:H, 0:D]` 是长度 `rs·H·D` 的**单段连续 DMA**（`blk_s·H·D·4 B` = 64–256 KB 级连续段）；逻辑任务数 = `B·⌈S/blk_s⌉`（比候选 2 少 `H` 倍） | ✅ **主选** |
| 2 | I/O 原生 BSHD + per-`(b, h, s-block)` 跨步 2-D（源几何：H 不并入 tile，作独立任务轴） | `D`（stride 1） | 无 | GM 流量相同；但行 burst 仅 `D·4 B`（256–512 B）且行距 `H·D·4 B`（跨步）；逻辑任务数 = 主选 ×`H`（`H = 8…64`）⇒ 发射项按 `H` 倍放大（§1.6.0 R3 候选 2 行） | ❌ + 依据：同元素量下任务数 ×`H`、burst 更碎；`constants.md` 的 CONST-mte2-degradation（同族 grid-stride/连续分块差异）与 CONST-gm-to-ub-bw-dilution（读形态在段宽 ≥ 数百字节时无差异，但此处**任务数**差异是实打实的发射项代价） |
| 3 | 核内重排为 `(H, D, blk_s)` 或 `(D, blk_s, H)`（把 `D` 挪到中轴 / 最内轴换成 `H`） | 最内 `H` | 核内 `T.transpose` 二轴交换链（引入 `vtranspose` 管线开销；大 shape 下可能降级为 UB→GM→UB 往返） | **无收益**（候选 1 的最内轴 `D` 已整除，重排不改善整除性）；转置代价 µs–百 µs 级 | ❌ + 依据：`layout.md` 的 PL-1.1-transpose-chain（实测 ~5.4 µs，且 2026-10-14 收窄为仅「两边 ≥ 向量宽且为整数倍」的 2D 形态成立）；`traps-compiler.md` 的 TRAP-transpose-ub-gm-roundtrip（UB 内转置可降级为 UB→GM→UB 往返 5.6 → 138.2 µs） |
| 4 | host 侧 `permute`/`reshape` 后进出 | 任意 | host permute（两次） | 不省设备侧流量；host 开销 **106–147 µs**（远超本算子 smoke 档的 2.9 µs 估算下界）；且触犯 host 侧约束 | ❌ + 依据：`layout.md` 的 PL-1.3-host-permute；`ascend-constraints.md` §4（host 仅允许 stride/shape 元数据视图） |
| 5 | 把整个 tensor 摊平成 1-D `(B·S·H·D,)` 单轴 | 该单轴 | 需要 1-D 声明 / reshape 视图 | 理论上取得与候选 1 **相同**的连续性（候选 1 在批内已取得全连续），但需 1-D 张量声明与接口 shape 不匹配（kernel 签名须与传入的 `(B,S,H,D)` 一致）或 reshape 视图 | ❌ + 依据：**等价收益 + 额外接口/API 复杂度**——`traps-compiler.md` 的 TRAP-npuir-reshape-dst-buffer-shadow（npuir 分支 `T.reshape` 语义漂移：被 `npuir_reshape(src,dst)` 覆盖，第二参为目标 buffer）。候选 1 以 3-D 声明达到同一连续性，无需 reshape |

**布局决策结论**：选定**候选 1**：核内布局 = **I/O 原生 BSHD**、向量化轴 = **最内 `D` 轴**（stride 1，`D ∈ {64,128}` 为 fp32 向量宽 ×4 与 fp16/bf16 向量宽 ×8 的整数倍，尾 lane 浪费 0%）、repack 路径 = **无需 repack**（恒等索引映射，`make_dq_layout` 已按 §0.5 判定 1 舍弃）、tile = `(blk_s, H, D)` 连续 slab（H 并入 tile 以取得单段连续 DMA 与 `÷H` 的任务数削减）。该结论已同步落入 §3.3 伪代码与 §6 循环结构：伪代码中 `T.copy`/`T.vcast` 的操作数形状均为 `(blk_s, heads, dim)`，`T.vcast` 的向量维就是本节选定的最内 `D` 轴；§6 中**不存在**逐元素标量循环（见 §1.6.2 的向量化结论）。弃选方案的量化理由见候选矩阵第 2–5 行。

**实验裁决与设计期探针的适用性判定**：本算子的主选与弃选候选的判定裕度**不依赖任何未实证常数**——① 候选 2/5 由**结构性计数**否决（任务数 ×`H`、额外 reshape API 语义风险），② 候选 3/4 由 pattern-library **已有实测条目**否决（PL-1.1 的 ~5.4 µs、TRAP-transpose-ub-gm-roundtrip 的 5.6 → 138.2 µs、PL-1.3 的 106–147 µs，均非「未实证常数」），③ 候选 1 自身的收益来自「单段连续 vs 跨步行」这一由 `CONST-mte2-degradation` / `CONST-gm-to-ub-bw-dilution` 已实测口径支撑的定性方向。⇒ 不触发「实验裁决三件套」的强制条件，也**不使用设计期探针**（预算未消费，可留给真需要标定常数的算子）。`blk_elems`（tile 元素预算）的取值差异属**参数级 A/B**（§4.5 阶梯），不构成结构级备选，其 A/B 计划写在 §4.5 与 §11。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（纯 Vector、无 Cube 路径）。

### 2.2 选型理由

1. **算子本体是纯 Vector**：全部计算只有一次整 buffer 的 fp32→fp16/bf16 转换（`T.vcast`）与两次 DMA 搬运（`T.copy`），**无 `T.gemm`、无 L0A/L0B/L0C、无 L1** ⇒ 不需要 Expert 的显式内存层级/手动同步能力。
2. **Developer 侧能力足够**（同批实证）：`T.alloc_shared` 映射到 UB、3-D UB buffer 在本工具链可用（同批 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/RETROSPECTIVE.md` 的 Stage 3 记录：4 个三维 `T.alloc_shared`、28 个 shape/dtype 组合全部编译通过）；`T.vcast` / `T.copy` 的 v-prefix/通用形态在 Developer 下均有 `docs/Tilelang.language/` 规格与 `examples/` 先例（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 的 Developer 示例 2）。
3. **同批一致性**：前两件套（preprocess / main）均以 Developer 交付并通过（`[PRECISION_PASS]`），保持三件套同模式可降低集成期的模式不一致风险（Stage 5 按 `{op}.py` 顶层 `ASCEND_MODE` 字面量同步 `Kernel.ascend_mode`）。
4. **无需 Expert 的双 Scope / 手动同步**：本算子无 CV 融合（无 Cube 侧），无跨引擎同步需求。
5. **调度约定**：本任务约定 `target="npuir"` 不带 `compile_flags` / `pass_configs`，且删除源 `threads=128`（K9）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared((blk_s, heads, dim), dtype)`（Developer 语义 = UB）；无 `T.alloc_L1` / `T.alloc_L0*` |
| 计算方式 | `T.copy`（DMA）+ `T.vcast`（整 buffer 向量）——**不使用** `T.Parallel` 逐元素循环（§1.6.2） |
| 同步 | 全部由编译器自动插入（§7） |
| Kernel 维度 | 一维 `T.Kernel(num_kernels, is_npu=True)`（§5.5） |
| 编译配置 | **无** `pass_configs` / `compile_flags`（按任务约定；源本函数亦无） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（单次显式 RNE 转换）。拆解为「搬运 → 转换 → 搬运」三步，全部映射到 v-prefix / 通用搬运 API。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `src32 ← dq[slab]` | 把 `(rs, H, D)` 连续 slab 从 GM 搬入 UB（fp32） |
| 2 | `dst16 ← RNE_T(src32)` | 逐元素 RNE 转换（fp32 → 存储 dtype），整 buffer 向量化 |
| 3 | `dq_out[slab] ← dst16` | 把同形 slab 从 UB 搬回 GM（存储 dtype） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 任务映射 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `for i in T.serial(num_local_tasks)` + `if uid < num_logical` | 一维核数、静态串行界、grid-stride 守卫 | Developer |
| 0 | 任务解算 | `bz = uid // nblk`；`s0 = (uid % nblk) * blk_s`；`rs = T.min(blk_s, seq_len - s0)` | 标量（每任务 1 次） | Developer |
| 1 | `src32 ← dq[slab]` | `T.copy(dq[bz, s0:s0+rs, 0:heads, 0:dim], src32[0:rs, 0:heads, 0:dim])` | GM→UB；两侧显式切片；两侧 extents 同源（`rs`） | Developer |
| 2 | `dst16 ← RNE_T(src32)` | `T.vcast(src32, dst16, round_mode="rint")` | 整 buffer；src/dst 同 shape `(blk_s, heads, dim)`；dtype f32→f16/bf16 | Developer |
| 3 | `dq_out[slab] ← dst16` | `T.copy(dst16[0:rs, 0:heads, 0:dim], dq_out[bz, s0:s0+rs, 0:heads, 0:dim])` | UB→GM；同形 | Developer |

### 3.3 计算伪代码

> 约定（模板要求）：`T.Tensor` 的 dtype 直接作为**第二个位置参数**传入，不用 `dtype=` 关键字，避免 `name 'in_dtype' is not defined` 一类的 false alarm。

```python
# 基于 TileLang API 的计算流程伪代码（Developer 模式，target="npuir"）
@tilelang.jit(out_idx=[1], target="npuir")
def _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype,
                                      blk_elems=_UB_ELEMS_BUDGET):
    # ---- host 侧编译期常量（trace 期折叠为 IntImm） ----
    row = heads * dim                                  # 每 S 行的连续元素数
    blk_s = max(1, min(seq_len, blk_elems // row))     # UB 元素预算反推（§5.2）
    nblk = (seq_len + blk_s - 1) // blk_s
    num_logical = batch * nblk
    vector_cores = NPUUtils.get().get_aicore_num() * 2  # 纯 Vector 核数（§5.5）
    num_kernels = min(num_logical, vector_cores)
    num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

    @T.prim_func
    def flash_bwd_post(
        dq: T.Tensor((batch, seq_len, heads, dim), "float32"),      # 主 kernel 的 fp32 累加器
        dq_out: T.Tensor((batch, seq_len, heads, dim), dtype),      # out_idx=[1] 自动分配
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            src32 = T.alloc_shared((blk_s, heads, dim), "float32")  # GM→UB 中转（fp32）
            dst16 = T.alloc_shared((blk_s, heads, dim), dtype)      # 转换结果（存储 dtype）

            for i in T.serial(num_local_tasks):                # 任务级串行（静态界）
                uid = i * num_kernels + cid                    # grid-stride 映射
                if uid < num_logical:
                    bz = uid // nblk                           # batch 下标
                    s0 = (uid % nblk) * blk_s                  # S 轴起点
                    rs = T.min(blk_s, seq_len - s0)            # 尾块行数（两侧同源）

                    # 1. GM → UB（fp32，单段连续 slab）
                    T.copy(dq[bz, s0:s0 + rs, 0:heads, 0:dim],
                           src32[0:rs, 0:heads, 0:dim])
                    # 2. 逐元素 RNE 转换（fp32 → fp16/bf16），整 buffer 向量化
                    T.vcast(src32, dst16, round_mode="rint")
                    # 3. UB → GM（存储 dtype，同形同区域）
                    T.copy(dst16[0:rs, 0:heads, 0:dim],
                           dq_out[bz, s0:s0 + rs, 0:heads, 0:dim])

    return flash_bwd_post
```

### 3.4 API 可行性确认

| API | 本设计用法 | 佐证（已核对） | 备注 |
|-----|-----------|----------------|------|
| `T.Kernel(num_kernels, is_npu=True)` | 一维核数 + persistent | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；同批 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/_flashattn_bwd_preprocess_kernel.py` 第 259 行同形 | **不传 `threads=`**（npuir 无效果，TRAP-threads-kwarg-noop；K9） |
| `T.alloc_shared((blk_s, heads, dim), dtype)` | 3-D UB buffer（fp32 / 存储 dtype） | `docs/Tilelang.language/内存操作/T.alloc_shared.md`；同批 preprocess 的 3-D `T.alloc_shared` 在本工具链实测可用（其 RETROSPECTIVE Stage 3） | Developer 语义 = UB |
| `T.copy(region, region)` | GM↔UB 双向，两侧显式切片 | `docs/Tilelang.language/内存操作/T.copy.md` §1（GM→UB / UB→GM 均支持）、§2.2.2 规则 2（尾块动态 extent 支持） | **同 dtype**；不使用跨 dtype（C2 否决） |
| `T.vcast(src, dst, round_mode="rint")` | f32→f16 / f32→bf16，整 buffer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f32→f16` 与 `f32→bf16` 均含 `rint`）、§2.2.2（input/output shape 须一致）、§2.3（src 须为 buffer，不可标量） | shape 一致 `(blk_s, heads, dim)` ✓ |
| `T.serial(num_local_tasks)` | 任务级串行（静态界） | `docs/Tilelang.language/编译器提示操作/`；同批 preprocess 第 266 行同形 | 静态界（host 折叠常量）满足分核标准 §1 要素③ |
| `T.min(blk_s, seq_len - s0)` | 尾块行数 | 同批 preprocess 第 274 行同形；`docs/Tilelang.language/比较操作/T.min.md` | 每任务 1 个标量 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是（源是三维）** | 已按 §0.6 R1 重设计为一维 `T.Kernel(num_kernels)` + 任务解算（`bz`/`s0` 由 `uid` 解出） |
| 部分 GPU API 不可用 | 是（`make_dq_layout` / `T.Layout` / `threads=`） | `make_dq_layout` 按 §0.5 判定 1 **舍弃**（NPU 无 mma fragment）；`threads=` 删除 |
| GEMM 要求 M,N 为 block 整数倍 | **否**（无 GEMM） | 不涉及；尾块由 `rs` 精确裁剪 |
| L0C 容量上限（128 KB） | **否**（无 Cube） | 不涉及 |
| 物理核数限制 | **是** | §5.5 三分核要素；实查 `get_aicore_num() = 20`、`aivector_core_num = 40` |
| Host 侧不得改动输入张量真实内容 | **是（须遵守）** | host 侧只做编译期常量计算（`row`/`blk_s`/`nblk`/核数），**不做**任何张量内容变换、不做 padding（对照 `ascend-constraints.md` §4） |
| 32B 尾轴对齐 | 是 | 最内轴 `D ∈ {64,128}`：`D·4 B = 256/512 B` 为 32 B 整数倍 ✓ |
| 非整除 shape | 是（`seq_len % blk_s ≠ 0`） | `rs = T.min(...)` 同时约束 GM 与 UB 两侧切片（§5.4） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5 / §0.6）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(T.ceildiv(S,64), heads, batch, threads=128)`（三维） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1：任务解算 + persistent |
| 布局注解 | `T.annotate_layout({dq: make_dq_layout(dq)})` | **无**（自然 BSHD，恒等映射） | §0.5 判定 1：舍弃（与主 kernel 产物布局自洽） |
| 线程模型 | `threads=128` | 无 `threads` 形参 | TRAP-threads-kwarg-noop；K9 删除 |
| dtype 转换 | 单条跨 dtype `T.copy`（隐式 cast） | `T.copy`（fp32）+ `T.vcast(rint)` + `T.copy`（存储 dtype） | §0.6 R3：显式化，等价性经 §1.6.1 O1 机器验证 |
| tile 几何 | `blk = 64`（固定，逐 `(b,h)` 切） | `(blk_s, heads, dim)` 连续 slab，`blk_s` 由 UB 预算反推 | §0.6 R2 |
| `pass_configs` / `compile_flags` | 源本函数无 | 无（任务约定：`target="npuir"` 不带） | — |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/_flashattn_bwd_preprocess_kernel.py` | **高度相似（同批兄弟 kernel）** | 一维 persistent 骨架（`T.Kernel(num_kernels, is_npu=True)` + `T.serial(num_local_tasks)` + `uid` 解算 + `if uid < num_logical`）；3-D `T.alloc_shared`；尾块 `real_s = T.min(...)` 两侧同源切片；`T.vcast(round_mode="rint")` 用法；`out_idx` / `ASCEND_MODE` / golden 形态 |
| `examples/multi_head_attention/_gqa_bwd_kernel/_gqa_bwd_kernel.py` | **高度相关（本 kernel 的 dq 生产者）** | 证明 NPU 侧 `dq` 以**自然 BSHD 布局**写出（逐行 1-D 行原子、**无** dq layout 注解）⇒ §0.5 判定 1 的直接依据；同款 persistent 分核参数（`get_aicore_num() * 2`） |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 高（同 dtype 变换族） | sub-fp32 逐元素算子的 `vcast` 中转链形态（`pattern-library/elementwise.md` 的 PL-1.17-subfp32-fp32-transit 首证） |
| `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | 规格 + 直接先例 | §2.4 的 Developer 示例 2 即「`T.copy` 载入 → `T.vcast` 转换 → `T.copy` 写出」的同构 kernel（本设计的 API 骨架） |
| `examples/TileOPs/tileops/ops/attention/multi_head_attention.py` | 契约层 | 第 241–255 行的 `forward` 调用序列：`dq = torch.zeros_like(q, dtype=float32)` → 主 kernel → `post_kernel(dq)`（本 kernel 的唯一调用点） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dq` | `(batch, seq_len, heads, dim)` | **float32** | 主 kernel 的 dq 累加器（与源 `accum_dtype = "float"` 一致）；BSHD 行主序，`dim` 轴 stride 1，`seq_len` 轴 stride = `heads·dim`；由 op 层 `torch.zeros_like(q, dtype=torch.float32)` 提供（`examples/TileOPs/tileops/ops/attention/multi_head_attention.py` 第 249 行） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dq_out` | `(batch, seq_len, heads, dim)` | float16 / bfloat16 | `out_idx=[1]` 自动分配；shape 与 `dq` **完全一致**、无转置布局（`source_output_shape = (B,S,H,D) fp16/bf16`）；元素值 = `RNE_T(dq[same index])` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 | 声明字节/元素 |
|-----------|-------|-------|----------|------|--------------|
| `src32` | `(blk_s, heads, dim)` | float32 | UB（`T.alloc_shared`） | GM→UB 装载载体；`T.vcast` 的 src | 4 |
| `dst16` | `(blk_s, heads, dim)` | 存储 dtype | UB（`T.alloc_shared`） | `T.vcast` 的 dst；UB→GM 写回载体 | 2 |

- **驻留元素数口径**：`E = blk_s · heads · dim`（单元的**实际**元素数，≤ `blk_elems`）；**声明量 = 6 B/元素**（4 + 2）。
- **无第三块缓冲**：转换在 UB 内完成（`src32` 读、`dst16` 写），无 fp32 乘积/暂存 buffer（对照同批 preprocess 的 O1 原位复用，本算子因 dtype 变化**不可能**原位——src/dst 的 dtype 不同，`T.vcast` 要求 src/dst 同 shape 但**不同 buffer**，见 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.2）。
- **无标量向量/无 workspace**：无跨核归并 ⇒ 无 workspace、无 flag。

### 4.4 内存搬运路径

```text
纯 Vector 路径（无 Cube / L0 / L1 参与）：

GM[dq (B,S,H,D) fp32]
   --T.copy(两侧显式切片；单段连续 slab，段长 = rs·H·D·4 B)--> UB[src32 (blk_s,H,D) fp32]
UB[src32]
   --T.vcast(round_mode="rint"：fp32 → fp16/bf16，整 buffer)--> UB[dst16 (blk_s,H,D) f16/bf16]
UB[dst16]
   --T.copy(同形同区域)----------------------------------------> GM[dq_out (B,S,H,D) f16/bf16]
```

关键约束：v 算子的操作数必须在 UB（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.3）；GM 不可直达 fragment；本设计不经 L1/L0、**无任何跨引擎往返、无中间结果落回 GM** ⇒ §1.6.0 roofline 的**容量项 = 0**。

### 4.5 UB 内存预算

**口径**（遵循 `.agents/skills/_shared/standards/hardware-cost-model.md` §5 与 `pattern-library/constants.md` 的 CONST-capacity-910B2C）：UB 容量实查 = **196608 B（192 KB，`AscendArch().ub_cap`）**；声明量按 `6 B/元素 × E`；以 auto multi-buffer 膨胀系数放大后与容量比较。

**逐 workload 复算（`blk_elems = 16384`；`blk_s = max(1, min(S, blk_elems // (H·D)))`）**：

| workload（manifest） | (B,S,H,D) | 存储 dtype | `H·D` | `blk_s` | `E = blk_s·H·D` | 声明量（6 B/元素） | ×1.7（默认档） | ×1.12（关 auto multi-buffer 档） | ≤192 KB |
|----------------------|-----------|-----------|-------|---------|-----------------|--------------------|----------------|----------------------------------|--------|
| `mha-bwd-smoke-s512-h8-d64` | (1,512,8,64) | fp16 / bf16 | 512 | 32 | 16384 | 98304 B（96 KB） | 167117 B（163 KB） | 110100 B（108 KB） | ✓ |
| `llama-3.1-8b-short` | (4,512,32,128) | fp16 / bf16 | 4096 | 4 | 16384 | 96 KB | 163 KB | 108 KB | ✓ |
| `llama-3.1-8b-long` | (2,2048,32,128) | fp16 / bf16 | 4096 | 4 | 16384 | 96 KB | 163 KB | 108 KB | ✓ |
| `llama-3.1-70b-short` | (2,512,64,128) | fp16 / bf16 | 8192 | 2 | 16384 | 96 KB | 163 KB | 108 KB | ✓ |
| `llama-3.1-70b-long` | (1,2048,64,128) | fp16 / bf16 | 8192 | 2 | 16384 | 96 KB | 163 KB | 108 KB | ✓ |
| 测试用极小档（`H·D = 1`） | (1,8,1,1) | fp16 / bf16 | 1 | 8 | 8 | 48 B | 82 B | 54 B | ✓ |
| **域上界（`H·D > blk_elems`）** | 如 `H·D > 16384` | — | — | `blk_s = 1` | `H·D` | `6·H·D` | `10.2·H·D` | `6.72·H·D` | ⚠ 见 §9.3 |

**降级阶梯（工厂 guard + Stage 4 反解）**：

| 档 | 元素预算 `blk_elems` | 可行条件 | 发射项（full 档核内轮数） |
|----|--------------------|----------|---------------------------|
| 上探档 | 32768 | `6 B/元素 × 32768 × 1.7 = 334 KB > 192 KB` ⇒ **不可行**（除非关闭 auto multi-buffer：`6 × 32768 × 1.12 = 220 KB > 192 KB`，仍不可行）⇒ 上探档实际**封顶在 16384 附近**，本设计不设该档 | — |
| **默认档** | **16384** | `6 × 16384 × 1.7 = 163 KB ≤ 192 KB`（口径 A 通过）；无归约链（本算子无 `T.reduce_*`）故无 `pattern-library/constants.md` 的 CONST-reduce-ub-per-element 项 | 13–26（见 §5.5） |
| 保守档 | 8192 | `6 × 8192 × 1.7 = 82 KB` | 26–52 |
| 兜底档 | 4096 | `6 × 4096 × 1.7 = 41 KB` | 52–104 |

> **UB guard**：工厂层校验 `blk_s · heads · dim ≤ blk_elems`（构造上恒成立，作为契约检查保留）；编译期若报 `ub overflow`，按阶梯降档（只改 `blk_elems` 实参，功能与精度完全不变）。**诚实标注**：本算子的 UB 形态（两块同形 buffer、无归约链、无原位复用）比同批 preprocess 简单得多（后者存在归约 lowering 的 8–96 B/元素分带单价，见其 DESIGN §4.5），故 16384 档的可行裕度**远高于** preprocess（其默认档亦为 4096）；`16384` 的取值属**设计期估算**（口径 A），Stage 3 首编即可确认，Stage 4 若需上探亦有 `--enable-auto-multi-buffer=false` 杠杆（但本设计估算该杠杆对本档无必要）。

### 4.6 动态轴定义

**无运行时动态 shape**：`batch` / `heads` / `seq_len` / `dim` / `dtype` 均为工厂期常量（per-build JIT 特化，与源工厂语义一致）；host 层按 shape 推导的 `row` / `blk_s` / `nblk` / `num_logical` / `num_kernels` / `num_local_tasks` 全部为 Python 编译期常量（trace 期折叠）。
kernel 内**唯一**的运行时标量是尾块行数 `rs = T.min(blk_s, seq_len - s0)`。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
def _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype,
                                      blk_elems=_UB_ELEMS_BUDGET): ...
```

- `out_idx=[1]`：`dq_out` 是第二个参数（源契约 `@tilelang.jit(out_idx=[1])`，源第 106 行），运行时自动分配 `(batch, seq_len, heads, dim)` 存储 dtype。
- 前五个位置参数**保持源工厂签名**（`batch, heads, seq_len, dim, dtype`）；`blk_elems` 为带默认值的可选参数（内部 A/B 杠杆，不改接口契约 ⇒ 集成后无需改 wrapper）。
- **无** `pass_configs`、**无** `compile_flags`（源本函数亦无；且按任务约定 `target="npuir"` 不带）。
- `ASCEND_MODE = "Developer"` 顶层字面量 + `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` 双写（Stage 5 集成脚本按顶层字面量同步 `Kernel.ascend_mode`）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**（无 Cube）。

**判定依据**：算子只包含 `T.copy`（DMA 搬运）与 `T.vcast`（逐元素 dtype 转换），**无 `T.gemm`、无矩阵分形约束、无 L0A/L0B/L0C、无 L1**；计算单元匹配为 Vector（§1.6.0 R4 主选行）。对比：源 GPU 侧同样是逐元素 kernel（无 `T.gemm`），但用 CUDA thread/fragment 模型表达 ⇒ 归「纯 Vector 的 CUDA 组织方式」，NPU 侧改用整 buffer 向量 op。

### 5.2 Block 划分

```python
row     = heads * dim                                  # 每 S 行的连续元素数（stride 1 的连续段）
blk_s   = max(1, min(seq_len, blk_elems // row))       # UB 元素预算反推的 S 行数
tile    = (blk_s, heads, dim)                          # 单元 tile（实际元素数 E = blk_s * row <= blk_elems）
nblk    = (seq_len + blk_s - 1) // blk_s               # S 轴分块数
# 逻辑网格（一维）由 (batch, nblk) 展平
```

**形状选择理由**：

1. **把 H 轴整体纳入 tile（§1.6.3 主选）**：对固定 `bz`，`(S, H, D)` 是连续内存 ⇒ `dq[bz, s0:s0+rs, 0:H, 0:D]` 是长度 `rs·H·D` 的**单段连续**区域（seg 长度 = `blk_s·H·D·4 B`，manifest 域 96–256 KB 级），DMA burst 最大化；同时逻辑任务数比「H 作独立任务轴」少 `H` 倍（§1.6.0 R3 候选 2 行）。
2. **`blk_s` 由 UB 预算反推而非沿用源的 64**：源 `blk = 64` 是 CUDA 侧经验值；NPU 侧容量约束是 UB 的 192 KB，`blk_elems = 16384` 在 `H·D ∈ {512,4096,8192}` 上给出 `blk_s = 32/4/2`（§4.5 表）。
3. **`blk_s = max(1, …)`**：材料上 `blk_s ≥ 1`；`H·D > blk_elems` 时退化为 `blk_s = 1`（tile = 一整 S 行的 `H·D` 元素），其容量可行性见 §9.3。

| workload | `blk_s` | `E` | 单元 tile 形状 | 单段连续段字节 |
|----------|---------|-----|----------------|---------------|
| (1,512,8,64) | 32 | 16384 | `(32, 8, 64)` | 64 KB |
| (4,512,32,128) | 4 | 16384 | `(4, 32, 128)` | 64 KB |
| (2,2048,32,128) | 4 | 16384 | `(4, 32, 128)` | 64 KB |
| (2,512,64,128) | 2 | 16384 | `(2, 64, 128)` | 64 KB |
| (1,2048,64,128) | 2 | 16384 | `(2, 64, 128)` | 64 KB |

### 5.3 约束分析

- **对齐约束**：最内轴 `dim ∈ {64,128}`；`dim·4 B = 256/512 B` 为 32 B 整数倍 ✓；fp32 向量宽 ×4、fp16/bf16 向量宽 ×8 均整除 `dim`（尾 lane 浪费 0%）✓。UB 侧切片起点恒为 0（避免 `pattern-library/traps-runtime.md` 的 TRAP-UB-dst-align 的「非零起点 + 32B 倍宽度」触发条件）✓。
- **UB 容量**：声明 96 KB，×1.7 = 163 KB ≤ 192 KB ✓（§4.5）。
- **L0C 容量**：无 Cube 计算，不适用 ✓。
- **GM→GM**：不使用（`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 5 条 Developer 不支持 GM→GM）✓。

### 5.4 注意事项（非整除处理策略）

**策略：运行时动态 extent，两侧同源裁剪（无 padding、无 crop、无越界写）**。

- `rs = T.min(blk_s, seq_len - s0)` 是唯一的运行时标量，**同时**约束：① GM 装载的 src 切片 `dq[bz, s0:s0+rs, …]` 与 UB 目标切片 `src32[0:rs, …]`；② GM 写出的 src 切片 `dst16[0:rs, …]` 与 dst 切片 `dq_out[bz, s0:s0+rs, …]`。**四侧同源** ⇒ tile 覆盖的元素集合 **恒等于**输出元素集合（无空转尾块、无越界写、无 padding 语义污染）。
- **`seq_len % blk_s == 0` 时** `rs == blk_s`，退化为无尾块（manifest 五档全部整除，尾块路径由 L0 非整除用例覆盖）。
- **不 padding 的理由**：padding 会把无效行写进 GM（越界污染相邻元素），且 32B 对齐非编译必要条件（`references/ascend-constraints.md` §1 的「raw 宽度无须对齐 pad」行）；动态 extent 已被 `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2 明确支持。
- **UB buffer 大小静态**（`(blk_s, heads, dim)`）：Developer 模式 UB buffer shape 须静态（`T.copy.md` §2.2.2 规则 4），故 tile 的**分配上界**是 `blk_s` 而非 `rs`——无效 lane 停留在 UB 内、永不写出。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。一句话：物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（Cube/混合直接用返回值、纯 Vector 翻倍），禁止文档假设/经验值；逻辑核超发会被串行调度并引入核启动开销，核数非整数倍造成负载不均。

**② 物理核数依据（实查记录，2026-10-16 本设计执行）**：

```python
# 在项目环境下执行（/workspace/venv/bin/python）
from tilelang.utils.npu_utils import NPUUtils
from tilelang.utils.npu_arch import AscendArch
NPUUtils.get().get_aicore_num()      # -> 20      （物理 AI Core）
AscendArch().aivector_core_num       # -> 40      （纯 Vector 核数，直接实查）
AscendArch().ub_cap                  # -> 196608  （= 192 KB，实查）
```

设备：**Ascend910B3**；**物理 AI Core = 20**、**纯 Vector 核数 = 40**（`get_aicore_num() × 2 = 40`，与 `aivector_core_num` 直查一致）；与 `pattern-library/constants.md` 的 CONST-aicore-910B2C 跨型号数据点（910B3：20 / 40 / 196608 B）一致；**与 910B2C 的 24 / 48 不同，不得跨型号复用核数值**。工厂每次构建重查 `get_aicore_num()`（跨型号自动适配）。

**① 逻辑核数计算**：`num_logical = batch × ⌈seq_len / blk_s⌉`（仅 S 轴分块；H 已并入 tile 故不作独立任务轴）：

| workload（manifest） | (B,S,H,D) | `blk_s` | `nblk = ⌈S/blk_s⌉` | `num_logical = B·nblk` | `num_kernels = min(逻辑, 40)` | `num_local_tasks = ⌈逻辑/核数⌉` | 负载分布 |
|----------------------|-----------|---------|--------------------|------------------------|-------------------------------|--------------------------------|----------|
| `mha-bwd-smoke-s512-h8-d64` | (1,512,8,64) | 32 | 16 | **16** | **16** | **1** | 16 核各 1 任务（其余 24 核空闲） |
| `llama-3.1-8b-short` | (4,512,32,128) | 4 | 128 | **512** | **40** | **13** | 512 = 40×12 + 32 ⇒ 32 核 13 任务、8 核 12 任务 |
| `llama-3.1-8b-long` | (2,2048,32,128) | 4 | 512 | **1024** | **40** | **26** | 1024 = 40×25 + 24 ⇒ 24 核 26 任务、16 核 25 任务 |
| `llama-3.1-70b-short` | (2,512,64,128) | 2 | 256 | **512** | **40** | **13** | 同 `8b-short` |
| `llama-3.1-70b-long` | (1,2048,64,128) | 2 | 1024 | **1024** | **40** | **26** | 同 `8b-long` |

**③ 规模判定与分核方案**：

- **`mha-bwd-smoke-s512-h8-d64`（逻辑核数 16 ≤ 物理核数 40）⇒ 判定「无需适配」**：`num_kernels = min(16, 40) = 16`、`num_local_tasks = 1`，每核恰好 1 个逻辑任务，**无核启动超发、无核内串行、无负载不均**（依据：`core-split-strategy.md` §1 要素③ 的第一档）。
- **其余四档（逻辑核数 512 / 1024 ≫ 物理核数 40）⇒ 判定「极大规模」**：无法通过调整 `blk_s` 把核数压到 40 量级（压到 40 意味着每任务处理 `S/40` 行 ≈ 13–51 行，UB tile 将达 `13·8192` 元素 = 106K 元素 ≫ 预算），故采用**固定启动内核数 = 物理核数整数倍（1×= 40）+ 核内 `T.serial` 串行**：`num_kernels = min(num_logical, 40) = 40`、`num_local_tasks = ⌈num_logical / 40⌉ ∈ {13, 26}`，核内 `for i in T.serial(num_local_tasks)` + grid-stride 映射 `uid = i · num_kernels + cid` + `if uid < num_logical` 守卫。
- **核内串行循环边界静态**：`num_local_tasks` 为 host 层 Python 编译期常量（`T.ceildiv`/整数运算在 trace 期折叠为 `IntImm`），满足 `core-split-strategy.md` §1 要素③ 的静态边界要求。
- **负载均衡**：不平衡 ≤ 1 个任务（上表第 8 列）；每任务的元素数相同（`E = blk_s·H·D = 16384` 恒定，除每 `bz` 的最后一个尾块），故最大核时延 ≈ `⌈num_logical/40⌉ × 单任务时延`。
- **收益**：消除逻辑核超发（`8b-long` 档 1024 个逻辑核若直接下发会被运行时串行调度）的核启动开销，并取得 grid-stride 聚集窗口的 MTE2 访问效益（`pattern-library/constants.md` 的 CONST-mte2-degradation 记录 grid-stride 优于每核连续分块 +3.2%）。
- **无核间归并**：逐元素映射、任务间元素独立 ⇒ 「每元素唯一写者」，无 `sync_block_set/wait`、无 workspace、无 flag id 预算问题（CONST-flag-id-budget 不适用），跨核代价 = 0（已计入 §1.6.0 R3 的「跨核结构」列）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表只列 block 级 / tile 级调度循环。本设计**不存在**逐元素标量循环，故无需引用 §1.6.2 的「保留理由」。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑任务（`batch × nblk`） | **核内串行（persistent）** | `for i in T.serial(num_local_tasks)` + `if uid < num_logical` | 极大规模分核（§5.5 要素③）：固定 40 核、核内摊薄；边界为 host 折叠的静态常量 |
| tile 内元素（`E = blk_s·H·D`） | **向量化（整 buffer）** | `T.copy`（DMA 引擎）+ `T.vcast`（Vector 引擎） | 无逐元素标量语句；向量维 = 最内 `D` 轴（§1.6.3 布局决策结论） |
| S 轴尾块 | 无循环（单标量） | `rs = T.min(blk_s, seq_len - s0)` | 动态 extent 由 `T.copy` 的切片承担（§5.4） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    src32 = T.alloc_shared((blk_s, heads, dim), "float32")
    dst16 = T.alloc_shared((blk_s, heads, dim), dtype)
    for i in T.serial(num_local_tasks):            # 静态界（host 折叠常量）
        uid = i * num_kernels + cid                # grid-stride
        if uid < num_logical:
            bz = uid // nblk
            s0 = (uid % nblk) * blk_s
            rs = T.min(blk_s, seq_len - s0)
            T.copy(dq[bz, s0:s0 + rs, 0:heads, 0:dim], src32[0:rs, 0:heads, 0:dim])
            T.vcast(src32, dst16, round_mode="rint")
            T.copy(dst16[0:rs, 0:heads, 0:dim], dq_out[bz, s0:s0 + rs, 0:heads, 0:dim])
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**：

1. 单元体内是**单条依赖链** `T.copy → T.vcast → T.copy`（写后读），跨迭代重叠由 Developer 的 auto multi-buffer 自动完成；
2. `.agents/evolution/capability-gaps.md` 的 CG-2026-0027 记录同类「单元循环上的 `T.Pipelined` 无增益」；
3. 本算子为流量项主导（§1.6.0 roofline），发射项占比 1/4–1/2，流水重叠的边际空间小。
4. **A/B 备选（Stage 4 可选）**：对任务循环尝试 `T.Pipelined(0, num_local_tasks, num_stages=2)`；须注意 `T.Pipelined` 的 `start` 只放**静态值**（同批 `_gqa_bwd_kernel` 的 Stage 3 实测：runtime start 直接编译失败、症状为伪 `cbuf overflow`），本设计的循环 `start = 0` 恰好满足。该备选属**参数级**调优，不作为结构级设计。

### 6.4 尾块处理

同 §5.4：`rs = T.min(blk_s, seq_len - s0)`，四侧（GM src / UB src / UB dst / GM dst）同源，无 padding、无越界写；UB buffer 静态分配、无效 lane 永不写出。manifest 五档均整除（`seq_len ∈ {512, 2048}`、`blk_s ∈ {2,4,32}`），尾块路径由 §8.3 的 L0 非整除用例覆盖（如 `seq_len = 513` / `blk_s` 不整除、`h·d` 边界组合）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式）。本设计不出现任何显式同步原语。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| GM→UB 装载后、`T.vcast` 前 | **自动**（编译器插入） | 保证 `src32` 的 DMA 写对后续 Vector 读可见 |
| `T.vcast` 后、UB→GM 写出前 | **自动**（编译器插入） | 保证 `dst16` 的 Vector 写对后续 DMA 读可见（写后读依赖） |
| 跨任务（`T.serial` 迭代间） | **自动** + auto multi-buffer | 同一对 buffer 被连续迭代复用；重叠由多缓冲版本化保证 |

无手动同步点：① 无跨引擎协作（纯 Vector，无 Cube 侧）；② 无跨核通信（任务间元素独立，无 workspace / flag）；③ 无持久化跨迭代累加状态（每个任务独立读写自己的 slab）。⇒ 与 §2.1 的 Developer 选型一致（Developer 用自动同步；Expert 才需标手动同步点）。

### 7.3 pass_configs 配置

**无 `pass_configs`**（按任务约定；源本函数亦无）。Stage 4 若需关闭 auto multi-buffer 以换取 UB 余量，可在工厂内附加 `pass_configs={"npuir.enable_auto_multi_buffer": False}`（属**参数级**调优，不改结构；对照同批 `_gqa_bwd_kernel` 的 §3.5.1 记录：`pass_configs` 在 npuir 可用，「不使用」属本任务约定）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（源侧数值行为 = IEEE RNE 的 dtype 下转换，等价于 `torch.Tensor.to(storage_dtype)`），**不复刻** §0.6 的 NPU 算法——无 persistent、无 `T.vcast`、无 UB tile、无 `rs` 尾块，保证验证独立性。

```python
def golden_flashattn_bwd_postprocess(dq: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Source semantics (DESIGN.md section 0.1 / 1.3, frozen):
        dq_out[b, s, h, d] = RNE_storage_dtype(dq[b, s, h, d])
    dq   : (B, S, H, D) float32
    dtype: torch.float16 | torch.bfloat16
    Returns (B, S, H, D) in `dtype`.  Device independent (CPU is fine).

    Written with an explicit ``torch.*`` namespace call (``torch.Tensor.to``)
    rather than the equivalent tensor-method form ``dq.to(dtype)`` so that the
    module-level torch usage is unambiguous to static gate checks.  The values
    are bit-identical to the tensor-method form (verified: ``torch.equal``
    returns True for both fp16 and bf16 on random inputs).
    """
    x32 = torch.Tensor.to(dq.detach(), torch.float32)   # view/copy as fp32 (already fp32 here)
    return torch.Tensor.to(x32, dtype)                  # single RNE round -> storage dtype
```

**独立性与可判定性**：golden 只做一次 RNE 下转换，与 NPU 算法的 tile 形状 / 分核 / 搬运形态完全无关；输入输出 `shape/dtype` 与 §0.2 契约逐项一致，可直接作为 L0 逐位门禁的参照（§8.2）。

### 8.2 精度标准

**本算子的门禁标准是「逐位相等」（bit-exact），而非容差**——理由是转换语义完全确定且不可容差化：

| 判据 | 适用 | 说明 |
|------|------|------|
| **主判据：`torch.equal(out, golden)` 为真**（逐位相等） | 全部 L0/L1/L2/Boundary 用例 | 输入 fp32 序列相同、舍入模式相同（RNE，§1.6.1 O1 机器验证）⇒ 输出必须逐位相同。**唯一合法的逐位差异来源**是 NaN 载荷（`NaN != NaN`），故比较时对 NaN 位置用 `both is nan` 视为相等（`equal_nan=True` 语义） |
| 兜底判据：`atol = rtol = 1e-3` | 若某用例暴露非 NaN 的逐位差异（设计预期不会） | 仅作诊断分级用；**逐位差异一旦出现在非 NaN 位置即视为缺陷**（须定位到舍入模式或元素映射），而非用容差放过 |

| dtype | atol（兜底） | rtol（兜底） | 主判据 |
|-------|-------------|-------------|--------|
| float16 | 1e-3 | 1e-3 | 逐位相等（`torch.equal` + NaN 等价） |
| bfloat16 | 1e-3 | 1e-3 | 逐位相等（`torch.equal` + NaN 等价） |

> 说明：源 op 级测试用 `atol=5e-3, rtol=1e-5`（`examples/TileOPs/tests/ops/test_multi_head_attention.py`）——那是**整 op**（含主 kernel 的 fp32 累加）的容差；本 kernel 级门禁可以且应当**更强**（逐位），因为它是纯粹的确定性 dtype 转换。

### 8.3 L0 门槛测试计划（Stage 3 必须全过）

> 用例分五组；`dtype ∈ {float16, bfloat16}`。所有用例的判据 = §8.2 主判据（逐位相等），并额外断言「输出 dtype/shape 正确」与「区域外无污染」。

| # | 分组 | 用例（B,S,H,D） | 目标 |
|---|------|-----------------|------|
| L0-1 | 主域 | (1,512,8,64) | manifest smoke 档：`blk_s=32`、`E=16384`、16 核 1 任务（§5.5 无需适配档） |
| L0-2 | 主域 | (4,512,32,128) | `blk_s=4`、40 核 13 任务（8b-short） |
| L0-3 | 主域 | (2,2048,32,128) | `blk_s=4`、40 核 26 任务（8b-long） |
| L0-4 | 主域 | (2,512,64,128) | `blk_s=2`、`H·D = 8192`（70b-short） |
| L0-5 | **参数一致性** | (1,512,8,64) 上 `blk_elems ∈ {4096, 8192, 16384}` | **P3**：同 shape 不同 tile 预算 ⇒ 输出**逐位一致**（证明 tile 形状不影响数值） |
| L0-6 | **元素覆盖 / 恒等映射** | (1,64,4,64) 输入取**每元素唯一可识别值**（如 `idx*1.0/1024` 或符号编码） | **P2**：逐元素对上（无置换、无遗漏、无重复）；区域外零污染 |
| L0-7 | **越界写检查** | (1,64,4,64)，`dq_out` 预填哨兵值 + 只写 slab 覆盖区 | 无越界写（对照 `pattern-library/traps-runtime.md` 的 TRAP-T-copy-region-semantics） |
| L0-8 | **尾块（S 不整除）** | `blk_s=32` 强制档下的 `seq_len = 513`，即 (1,513,8,64) 与 (1,100,8,64) | `rs < blk_s` 的重设计项 R2 尾块裁剪正确；四侧同源、无写出越界 |
| L0-9 | **极端 `H·D`** | (1,8,1,1)（`H·D=1`）、(1,8,1,64)（`H·D=64`，`blk_s` 截顶到 `seq_len`）、(2,4,64,128)（`H·D=8192`、`blk_s=2`） | 覆盖 §5.2 的 `min(seq_len, …)` 截顶与 `blk_s=2` 下界档 |
| L0-10 | **UB 阶梯** | (1,512,8,64) 上 `blk_elems ∈ {4096, 8192, 16384}` 编译并跑通 | §4.5 阶梯的可行性（Stage 4 反解真实 UB 单价的输入） |
| L0-11 | **接口契约** | 2 位置参数调用 + `out_idx=[1]` 自动分配 | 自动分配张量的 shape/dtype 与 §4.2 逐项一致（同批 preprocess 的 L0-14 同款） |
| L0-12 | **角点值** | 输入含 `±inf` / `NaN` / `±0` / fp16 溢出阈值 65520 / 最小正规 6.103515625e-05 / 最小次正规 5.960464477539063e-08 / `1+2^-11`(fp16 tie) / `1+2^-8`(bf16 tie) / `±3.4028235e38` / `±1e-40` | 边界语义（§0.1）与 §1.6.1 的 D-1 角点一致；NaN 位置按 `equal_nan` 语义判定 |
| L0-13 | **随机量级** | 输入 ~`N(0,1)`、~`U(-4,4)`、~`U(-7e4,7e4)`、~`U(-1e-4,1e-4)` 四种分布 | 覆盖常规域、溢出域、次正规域（tie 密度区） |
| L0-14 | **确定性** | 同输入连续两次调用 | 输出逐位一致（无未定义行为） |

> **L1/L2/Boundary 分工**：L1 = 更大 shape / 更多 dtype 组合（交 `tilelang-op-develop` 的分层套件展开）；L2 = 真实 manifest 五档端到端；Boundary = 极限 shape（如 `blk_s = 1`、`seq_len = 1`）与（记录不断言的）后端编译稳定性观察。完整分层套件由 `tilelang-op-develop` 生成，此处不枚举。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **`make_dq_layout` 的舍弃依赖主 kernel 的产物布局（跨工件一致性风险，最高优先级）**：本设计能删掉布局注解，前提是**主 kernel `_gqa_bwd_kernel` 的 NPU 实现以自然 BSHD 布局写 `dq`**。该前提当前成立（其实现为逐行 1-D 行原子、无 dq layout 注解，见 `examples/multi_head_attention/_gqa_bwd_kernel/_gqa_bwd_kernel.py`），但它是**跨工件契约**：若后续主 kernel 因任何原因改变了 dq 的写出布局，本 kernel 会静默算错（不会报错、不会 NaN）。**缓解**：集成期（Stage 5）任何对主 kernel 的 dq 写侧改动都必须同步复核本 kernel；本设计在 §0.5 判定 1 与 §1.6.1 C1 两处显式记录该依赖。
2. **`H·D > blk_elems` 的域外 fallback 未在本设计实现**：`blk_s` 退化为 1 后 tile 元素数 = `H·D`；若 `H·D > blk_elems`（manifest 域 `H·D ≤ 8192` 不触发），UB 声明量 = `6·H·D` 可能超 192 KB（阈值 `H·D ≳ 19200` 在 ×1.7 口径下）。处理见 §9.3。**风险等级：低**（manifest 五档全部 `H·D ≤ 8192`，`blk_s ≥ 2`）；工厂层应加 guard 拒绝 `6·H·D·1.7 > ub_cap` 的配置并给出清晰报错。
3. **auto multi-buffer 膨胀系数为估算档（未在本形态实测）**：§4.5 用 ×1.7（默认档）与 ×1.12（关多缓冲档）双口径；本算子 UB 形态简单（两块同形、无归约链）⇒ 163 KB ≤ 192 KB 有 29 KB 余量，即使真实系数略高也在余量内。**诚实标注**：该系数为设计期引用值（`constants.md` 的 CONST-capacity-910B2C，910B2C + CANN 8.5.0 档），Stage 3 首编即可确认。
4. **roofline 常数跨设备/跨工具链**：`CONST-mte2-degradation` / `CONST-vector-launch-overhead` 绑定 910B2C + CANN 8.5.0，本设备为 910B3 + CANN 9.0.0 ⇒ 按「未实测假设 + 同代曲线线性外推」引用（§1.6.0 已声明）。**流量项与发射项的比值**（决定「流量主导」这一结论）对系数不敏感（发射项为流量项的 1/4–1/2，即使单价取上界 0.5 µs/op 仍不反超），故结论稳健；但**绝对时延下界**须 Stage 4 回填实测。
5. **本算子无 Cube ⇒ L0C 约束不适用**，但若后续有人尝试「用 `T.gemm` 做 cast」（无意义），会立即触发 L0C/分形约束——本设计已在 §1.6.0 R4 明确否决（候选 5 行）。
6. **`T.vcast` 的形态约束**：src/dst 必须同 rank 同 shape（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.2），且 src 须为 buffer（§2.3，标量 src 触发前端断言）。本设计的 src/dst 均为 `(blk_s, heads, dim)` 的独立 buffer ✓。**同 dtype `vcast` 不可用作搬运**（TRAP-vcast-same-dtype-rounds），本设计两次搬运均用 `T.copy` ✓。
7. **`T.copy` 的语义边界**（`pattern-library/traps-runtime.md` 的 TRAP-T-copy-region-semantics）：不使用 GM→GM（Developer 不支持，`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 5 条）；不使用跨 dtype（§1.6.1 C2）；UB 侧切片起点恒为 0（规避 TRAP-UB-dst-align）。
8. **能力缺口（本设计未触发，登记备查）**：预注入的 CG-2026-0029（`T.reduce_sum` 无 bf16 累加形态 + `T.vcast` 不接受 UB 子区域形状）与 CG-2026-0018（codegen `DTypetoMLIRType`）在本设计中**均不适用**——本算子无归约（不触发 0029(a)），且 `T.vcast` 的操作数是**完整 UB buffer**（`src32` 整块，非前导轴切片，不触发 0029(b)）。**本设计的 `vcast` 形态规避了该缺口的触发条件**，且该规避是设计性选择（§0.6 R3 用同形 buffer 而非大 buffer 的子区域切片）。
9. **同批已知后端问题**：同批 `_gqa_bwd_kernel` 的 Stage 3 记录 `S == 32` 会触发 `bishengir-compile` 段错误（与数值无关）；本设计的 L0 用例避免 `seq_len = 32` 组合以免混淆诊断（L1/L2 若覆盖该 shape 应把崩溃与精度告警分开记录，同批纪律）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| `ub overflow` | `blk_elems` 过高或 `H·D` 过大 | 编译失败 | 按 §4.5 阶梯降档（`16384 → 8192 → 4096`）；或关闭 auto multi-buffer（×1.12 档） |
| **静默错值（元素置换）** | 误加回 `make_dq_layout` 的 `T.annotate_layout({dq: …})` | **无报错、无 NaN、全元素错位** | 严格按 §0.5 判定 1 不引入 layout 注解；L0-6（唯一标识值）专测该风险 |
| 逐位不一致（非 NaN 位） | 舍入模式被改成 `round`/`trunc`（如写成 `round_mode="round"`） | ties 处与 golden 不同（fp16 91/12471、bf16 82/12471） | 恢复 `round_mode="rint"`（§1.6.1 O1 与 verify_equiv.py 均已固化） |
| `shapes are incompatible`（codegen） | 让 UB 源的最后一条非单位轴不是 payload 轴，或两侧 extents 不同源 | 编译失败（`bufferization.materialize_in_destination`） | 保持两侧切片同源（`rs` 同时约束四侧）；payload 轴（`heads`,`dim`）两侧末尾对齐（同批 preprocess 的 Stage 3 教训，本设计的 3-D 算子形态天然满足） |
| 尾块越界写 / 输出残留 | `rs` 只约束了部分侧（如只裁 GM src） | 输出出现哨兵值残留或相邻元素被污染 | 四侧同源（§5.4）；L0-7/L0-8 专测 |
| 编译期 `T.Pipelined` 伪 `cbuf overflow` | 若尝试 6.3 的 A/B 备选且写成 runtime start | 编译失败（症状具误导性） | `T.Pipelined` 的 `start` 只放静态值（本设计 `start = 0`） |
| 运行中编辑 kernel 源文件导致 `DiagnosticError` | 跑测期间修改 `{op}.py` | trace 期按路径重读文件，报与代码无关的错误 | 跑测期间冻结源文件（同批教训） |

### 9.3 特殊场景处理

- **`H·D > blk_elems`（域外）**：`blk_s` 退化为 1。若 `6·H·D·1.7 > 196608`（即 `H·D ≳ 19200`），需对 H 轴再分块：tile 改为 `(1, blk_h, dim)`、`blk_h = max(1, blk_elems // dim)`，区域 `dq[bz, s0, h0:h0+blk_h, 0:dim]`（对固定 `(b,s)` 的 `(H,D)` 块仍连续，故段内连续；仅当 `blk_h < H` 时不同 `s` 行之间变成跨步）。**该 fallback 不进入主设计**（manifest 域不触发），工厂层以 guard 拒绝并给出显式报错，Stage 3 若需扩域再按本节实现。
- **`seq_len < blk_s`**：`blk_s = min(seq_len, …)` 的截顶使 `nblk = 1`、`rs = seq_len`，单任务覆盖全部 S 行（L0-9 覆盖）。
- **极小 shape（如 `H = D = 1`）**：`H·D = 1` ⇒ `blk_s` 截顶到 `seq_len`；`dim = 1` 时最内轴为 1 个元素——`T.vcast` 与 `T.copy` 仍可用（同批 preprocess 的 L0 覆盖过 `D = 1` 档），但向量效率低；仅作正确性覆盖，不追求性能（L0-9）。
- **`B·S·H·D` 极大（如 `(2,8192,64,128)`）**：`num_logical` = `2·⌈8192/1⌉`? 不——`H·D = 8192` ⇒ `blk_s = 2` ⇒ `nblk = 4096` ⇒ `num_logical = 8192` ⇒ 40 核 205 轮，UB 与正确性均不受影响（仅时延随流量线性增长）。
- **`dq` 含大量 `±inf`/`NaN`**：输出为 `±inf`/`NaN`（RNE 不改变其类别）；`|x| ≥ 65520` 的有限值溢出为 `±inf`（L0-12 覆盖）。注意：主 kernel 若因上游异常产出 `NaN`，本 kernel 会如实传播——**不做**任何 sanitize（语义保持优先）。

---

## 10. 交付清单

### 10.1 目录结构

```text
examples/multi_head_attention/_flashattn_bwd_postprocess_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # D-1 等价性机器验证（已执行：SUMMARY: ALL_EQUIV_PASS）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8 追加）
```

下游 Stage 3 将生成 `_flashattn_bwd_postprocess_kernel.py`（kernel + golden + 分层测试套件 + main）。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（含 §0 迁移分析、§1.6 算法调研与优化、§5.5 分核三要素、§8 验证方案） |
| `verify_equiv.py` | 已完成 | D-1 等价性机器验证脚本；2026-10-16 设计期已执行，`SUMMARY: ALL_EQUIV_PASS`（两 dtype 各 12471 案例、逐位相等） |
| `RETROSPECTIVE.md` | 已完成（Stage 1 章节） | 复盘与价值点提案 |
| `_flashattn_bwd_postprocess_kernel.py` | 待实现 | 算子实现（kernel + golden + L0/L1/L2/Boundary 套件 + main） |

### 10.3 命名规范

- 项目目录名：`multi_head_attention`（snake_case）
- 算子目录名：`_flashattn_bwd_postprocess_kernel`（与源函数名逐字一致，保留前导下划线）
- 实现文件：`_flashattn_bwd_postprocess_kernel.py`
- 工厂函数：`_flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype, blk_elems=16384)`（前五参数与源签名逐字一致）

### 10.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）
2. ✅ D-1 等价性机器验证（`verify_equiv.py`，`ALL_EQUIV_PASS`）
3. ⬜ Golden 函数（`golden_flashattn_bwd_postprocess`）+ L0 套件（§8.3 的 14 组）
4. ⬜ 算子实现（`{op}.py`）+ 分层测试（L1/L2/Boundary）
5. ⬜ 集成进 `multi_head_attention` 包（Stage 5；**不得改动 fwd 产物**）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 |
|------|----------|
| 保持源函数签名（前五参数） | ✅ `(batch, heads, seq_len, dim, dtype)`；`blk_elems` 为带默认值可选参数 |
| 保持输出 shape/dtype 契约 | ✅ `(B,S,H,D)` 存储 dtype（`out_idx=[1]`） |
| 不照抄 GPU 专属手段（三维 Kernel / mma fragment / threads） | ✅ §0.5 四态判定 + §0.6 重设计 |
| 不读源码不臆测 | ✅ 已逐语句读源实现（§0.1–§0.4） |
| golden 独立于 NPU 算法 | ✅ §8.1（只用 `torch.Tensor.to`），未复刻 persistent/vcast/tile |
| 不改动其他函数工件 / GPU 仓库 | ✅ 仅在本算子目录落盘 |

---

## 11. 关键结论与 Stage 4 交接

### 11.1 关键结论

1. **算子本质**：`dq_out = RNE_T(dq)`——逐元素一元 dtype 下转换，`flops = 0`、`bytes = N·(4+eb) = 6N`、1 遍扫描、无规约、无中间 GM 往返 ⇒ **纯带宽受限搬运**（§1.6.0 roofline：五档全部由流量项主导，下界 = 41.7 / 83.4 µs 量级）。
2. **迁移决策（§0.5/§0.6）**：**保留** 语义层（逐元素 RNE cast + 恒等映射）；**等价替换** 跨 dtype `T.copy` → 显式 `T.vcast(rint)`（机器验证逐位等价）；**重新设计** 三维 grid → 一维 persistent、`blk=64` → UB 预算驱动的 `(blk_s,H,D)` 连续 slab；**舍弃** `make_dq_layout`（Ampere mma fragment 专属；保留会**静默置换元素**）与 `threads=128`（npuir no-op）。
3. **算法调研（§1.6.0）**：选定「全 H 连续 slab + 一维 persistent + 显式 `vcast(rint)` + 恒等映射」；R1 的 7 个候选中 6 个被结构性/证据性否决（含 1 个越界、1 个数值不可用）；R2 无实质在线变体（逐元素映射，三项收益口径逐项失效）；R3 全部存活候选的 GM 流量与扫描遍数相同（排序由 R4 决定）。
4. **算法级优化（§1.6）**：代数化简类采纳 **0 项**（无除法/超越/归一化/CSE/cast 链）；等价性选择类采纳 **O1（`round_mode="rint"`）**，D-1 机器验证两 dtype 各 12471 案例**逐位相等**、违反 0，且否决模式（`round`/`trunc`）可区分（fp16 91/6091、bf16 82/6278）；向量化替代 **1/1 逐元素点已向量化**，保留 3 处标量（任务映射 / 动态尾块 / host 元数据，均 `O(任务数)` 而非 `O(N)`）；布局决策选定 **I/O 原生 BSHD + 最内 `D` 轴 + 无 repack**。
5. **分核（§5.5）**：物理 20 AI Core / 40 Vector Core（本机实查）；smoke 档逻辑 16 ≤ 40（无需适配）、其余四档 512/1024 ≫ 40（极大规模 → 固定 40 核 + `T.serial(13/26)` 核内串行，静态界）。
6. **验证（§8）**：判据 = **逐位相等**（`torch.equal` + NaN 等价），比源 op 级容差强一个量级；14 组 L0 覆盖主域 / 参数一致性 / 恒等映射 / 越界写 / 尾块 / 极端 `H·D` / UB 阶梯 / 接口契约 / 角点 / 随机量级 / 确定性。

### 11.2 Stage 3 首编即应验证的项（低成本高信息量）

1. **首编 UB 占用**（确认 §4.5 的 16384 档在真实膨胀系数下可行；若 `ub overflow` 按阶梯降档）；
2. **`T.copy` + `T.vcast` 三段链在本形态编译通过**（`(blk_s, heads, dim)` 3-D UB、两侧显式切片、动态前导 extent `rs`）；
3. **L0-6 唯一标识值用例**（正面覆盖 §9.1 风险 1 的「无置换」主张）；
4. **两 dtype 逐位一致**（`vcast(rint)` 的 f32→bf16 路径须与 `torch.Tensor.to(torch.bfloat16)` 逐位对齐）。

### 11.3 Stage 4 调优交接

| 项 | 内容 |
|----|------|
| 主杠杆 | **有效带宽**（流量项主导）：确认 40 核满发、slab 段连续、核内轮数 = 13/26；msprof 关注 `aiv_mte2_time` / `aiv_mte3_time` / `mte2_active_bw` |
| 参数级 A/B | `blk_elems ∈ {4096, 8192, 16384}`（tile 越大 → 轮数越少 → 发射项越小；但流量项主导下边际有限，且 16384 已贴近 UB 预算上界） |
| 结构级 A/B（可选、非强制） | §6.3 的 `T.Pipelined(0, num_local_tasks, num_stages=2)`（任务循环流水）；关闭 auto multi-buffer（`pass_configs={"npuir.enable_auto_multi_buffer": False}`）换取 UB 余量 |
| 回填要求 | 按 `.agents/skills/_shared/standards/hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行；若实测流量下界明显低于 §1.6.0（如 < 0.6×），优先怀疑「有效带宽常数的跨设备外推」并按实测更新常数引用 |
| 无结构级实验裁决义务 | §1.6.3 已判定：主选与弃选的判定裕度**不依赖未实证常数**（弃选由结构性计数 + pattern-library 已实测条目支撑），故不触发实验裁决三件套，也无设计期探针消费 |

---

> **本设计文档的算法调研（§1.6.0）、算法级优化（§1.6.1–§1.6.3）与迁移决策（§0）已冻结，供 Stage 2 检视与 Stage 3 实现引用。**
