# `_flashattn_bwd_postprocess_kernel` 算子设计文档

> 迁移任务（scenario=migration / migration_mode=harness）：将 GPU TileOPs `GroupedQueryAttentionBwdOp` 的第 2 个 TileLang 函数 `_flashattn_bwd_postprocess_kernel`
> （源：`/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py` L106–128，类 `FlashAttnBwdPostprocessKernel` L131–176）迁移到 `target="npuir"`。
> 编程模式：Developer（用户指定）。目标设备：**Ascend910B3**（实查 20 AI core / 40 Vector core，UB 192KB）。
> 同任务第 1 个函数 `_flashattn_bwd_preprocess_kernel` 已通过 Stage 1–3，其实现骨架为本函数的最佳参照。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `tilelang-op-design` skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：FlashAttention 反向流水线的收尾步骤——把主反向 kernel 以 fp32 累加器写出的 `dq`（**未缩放**）**逐元素转换（cast）回存储 dtype**，得到最终输出 `dq_out`。

$$\text{dq\_out}[b,s,h,d] \;=\; \mathrm{cast}_{\text{storage\,dtype}}\big(\text{dq}[b,s,h,d]\big),\qquad b\in[0,B),\ s\in[0,S),\ h\in[0,H),\ d\in[0,D)$$

转换按 IEEE **round-to-nearest-even（RNE）** 进行（源实现由 `T.copy` 的隐式跨 dtype 转换完成，CUDA 侧为 RNE）。

> ⚠️ **关键语义澄清（纠 op_requirements 的推测）**：调度 prompt 的 `math_formula` 推测本 kernel 会做 `dq_final = dq_acc * delta`（或除 delta）的缩放。**源码逐行核对后确认：本 kernel 不做任何缩放、不读取 `delta`、不做除法** —— 它只有一个参数 `dq`（L118）与一个输出 `dq_out`（L119），函数体仅一次 `T.copy(dq[...], dq_out[...])`（L123–126），`dq` 为 `accum_dtype="float"`（fp32），`dq_out` 为 `dtype`（fp16/bf16），故该 `T.copy` 的全部语义就是 **隐式 dtype 转换（cast）**。本设计严格以此为准（缩放/`delta` 在源码中不存在，不得添加）。

**I/O 契约（源码 L113/L118–119 逐行读出）**：
- `dq`：**BSHD**（`shape = (batch, seq_len, heads, dim)`，L113），**dtype = float32**（`accum_dtype = "float"`，L112）。
- `dq_out`：**BSHD，与 `dq` 同 shape 同内存顺序**，dtype = `dtype` 参数（fp16/bf16）。
- 输入与输出布局**完全相同**（BSHD），算子为**纯逐元素映射**（element-wise，一对一，无跨元素依赖）。
- `@tilelang.jit(out_idx=[1])`（L106）——kernel 自行分配并返回第 2 个参数 `dq_out`。

**dtype 语义**：唯一数值操作是 `fp32 → fp16/bf16` 的 RNE 转换（`T.vcast.md` §2.2.1：`f32→f16`、`f32→bf16` 均支持，`round_mode` 可选）。无乘法/加法/累加，故无累加顺序语义（**与 preprocess 的规约语义不同**）。

**边界语义**：无 nan/inf 特殊处理（直接按 IEEE 转换传播：fp32 溢出到 fp16 ≥ 65520 → ±inf，符合 IEEE 与 PyTorch `.to()` 语义）；无空维度场景；`dq`/`dq_out` 形状相同、无广播。

**host 侧语义**：无 host 预处理/后处理。Op 侧 `gqa.py` L1382 在非 Hopper 路径上调用 `self.post_kernel(dq)`；`dq` 由 `torch.zeros_like(q, dtype=torch.float32)`（L1378）分配、主 kernel 原地累加填充。`FlashAttnBwdPostprocessKernel.forward`（L170–171）仅 `return self.kernel(dq)`。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`dq.to(storage_dtype)`，RNE），**不复刻** §0.6 的 NPU 算法结构。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `dq` | `(batch, seq_len, heads, dim)` = **BSHD** | **float32** | fp32 `dq` 累加器（主反向 kernel 就地累加产出） |
| `dq_out` | `(batch, seq_len, heads, dim)` = **BSHD** | `float16` \| `bfloat16` | 输出（迁移必需字段；`out_idx=[1]`，源码 L119/L106）。**布局与 `dq` 完全一致** |

> 输出 shape 依据：源码 L113 `shape = (batch, seq_len, heads, dim)`（`dq`/`dq_out` 共用，L118–119）；op 契约见 `gqa.py` L1382（`dq = self.post_kernel(dq)`，返回后直接作为 op 的 `dq` 输出，shape 为 `(B,S,H,D)`）；manifest `dq.shape == (B, S, H, D)`（`examples/TileOPs/tileops/manifest/attention.yaml` L233）。
> **非 Hopper 分支**：Hopper（SM90）走 `dq.to(self.dtype)` 并跳过本 kernel（`gqa.py` L1357/L1382）；NPU 无 WGMMA，等价于非 Hopper 分支 → 本 kernel 在 NPU 上始终参与。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码 L116–128，逐语句归档，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.Kernel(ceildiv(seq_len,64), heads, batch, threads=128)`（L121）| — | 三维 block 网格 | 任务划分（非计算） |
| 2 | `T.annotate_layout({dq: make_dq_layout(dq)})`（L122）| — | 声明 `dq` 的 GM 布局（重排） | **无计算语义**（纯内存布局声明，见 §0.4 #3） |
| 3 | `T.copy(dq[bz, bx*64:(bx+1)*64, by, :], dq_out[bz, bx*64:(bx+1)*64, by, :])`（L123–126）| `dq` GM (fp32) | `dq_out` GM (fp16/bf16) | 逐元素 cast（搬运内隐式转换） |

**数据流与内存访问模式**（源硬件视角，GPU/CUDA）：

```
GM[dq  BSHD fp32]  ──T.copy（隐式 RNE cast）──>  GM[dq_out BSHD fp16/bf16]
        （固定 head=by，s ∈ [bx·64, (bx+1)·64)，全 D）
```

- 无 register fragment、无 shared memory 中转、无归约、无累加 —— **源实现是 GM→GM 的直接拷贝（含隐式 cast）**。
- **关键观察**：源 step 3 是 **GM→GM** 拷贝。NPU Developer 模式的 `T.copy` **显式不支持 GM→GM**（`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 5 条：「Developer 模式不支持 GM→GM 拷贝……需经 UB 中转」）→ **必须引入 UB 中转**（→ §0.5 重新设计）。

**循环与并行结构**：grid = `(ceildiv(seq_len,64), heads, batch)` 三维（共 `ceil(S/64)·H·B` 个 block）；block 内无循环、无并行指令（单次 coalesced 拷贝）；`threads=128`。

**host 侧逻辑**：无。`FlashAttnBwdPostprocessKernel.forward`（L170–171）仅 `return self.kernel(dq)`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 三维 block 网格（L121）| 天然三维任务映射，免索引分解 | 硬件直接给出 `(bx,by,bz)` | CUDA grid 三维 | **硬件强相关**（NPU 一维 `T.Kernel`） |
| 2 | 固定 `blk=64` 启动几何（L114）| 消除 autotune（类 docstring L133–137）| 编译期固定 tile | — | **可移植（模式）**，但 tile 取值受 NPU UB/搬运带宽约束需重算 |
| 3 | `make_dq_layout` 布局声明 + `T.annotate_layout`（L122，helper L99–103）| 让 `dq` GM 以 8×8 gemm fragment 友好的重排顺序存储，**使主 kernel 的 `T.atomic_add` 可向量化**（helper 注释 L100：「atomicAdd cannot be vectorized on Ampere, need to reorder dq to match the 8x8 gemm fragment」）| 声明 `dq` 的逻辑→物理布局映射 `[b, len//8, h, d//8, d%2, 4·(len%8)+(d%8)//2]` | Ampere 的 atomicAdd 向量化限制 + GPU fragment 布局 | **硬件强相关（GPU 专属）**（NPU 无该 fragment 布局、无对应 atomicAdd 约束）|
| 4 | `out_idx=[1]` 自动分配输出（L106）| 免 host 侧 zeros 分配 | jit 装饰器 | — | **纯 host 语义，保留** |
| 5 | coalesced 连续尾轴访问（L123–126）| 访存带宽 | `dq[b,s,h,:]` 沿 D 连续（fp32 D·4 = 256/512B 段）| memory coalescing | **可移植**（NPU MTE 亦按连续段计） |
| 6 | 隐式跨 dtype 拷贝即 cast（L123–126）| 把「读 + 转 + 写」融为一次拷贝 | GPU `T.copy` 允许 src/dst dtype 不同，拷贝内完成 RNE 转换 | CUDA 拷贝语义 | **部分可移植**（NPU 亦有隐式转换，但**另见 §1.6.1 的 TRAP-C12 否决**）|

> 识别不出来 ≠ 不存在：以上 6 项为源码全部优化点；未列出的优化不会在迁移中被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（逐元素 fp32→storage cast、RNE）| 语义 | 无 | — | **保留** | 语义层无条件保留（`references/migration-analysis.md` §5.4 规则 1）；RNE 由 `T.vcast(round_mode="rint")` 承接 |
| BSHD→BSHD 同形同序布局 | 语义 | 无 | — | **保留** | 输入输出布局不变（与 preprocess 的 BSHD↔BHS 隐式转置不同） |
| 逐元素映射（无归约/无累加）| 算法 | 无 | — | **保留** | 唯一算法结构 |
| 三维 block 网格（#1）| 优化 | CUDA 3D grid | **无** | **重新设计** | → §0.6 R1（一维 `T.Kernel` + block 索引分解；本项目不支持三维 Kernel，见 `references/ascend-constraints.md`）|
| 固定 `blk=64` tile（#2）| 优化 | 无 | 有（UB）但容量/带宽约束不同 | **重新设计** | → §0.6 R2（按 UB 与搬运带宽膝点重算 tile）|
| `make_dq_layout` 布局声明（#3）| 优化 | **GPU 专属**（Ampere atomicAdd 向量化 + fragment 布局）| **无** | **舍弃** | NPU 无 8×8 fragment 布局、无对应 atomicAdd 约束；NPU 主 kernel 不使用该布局（`docs/Tilelang.language/` 无 `annotate_layout` 条目，NPU 侧无此语义需求）→ **丢弃不影响本 kernel 语义**（布局声明无计算语义，见 §0.3 step 2）|
| GM→GM 直接拷贝（算法结构）| 算法 | GPU 允许 | **无** | **重新设计** | → §0.6 R3（经 UB 中转：GM→UB→GM；依据 `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 5 条）|
| `out_idx=[1]`（#4）| 优化 | 无 | 有 | **保留** | jit `out_idx` 装饰器语义 |
| coalesced 尾轴连续读（#5）| 优化 | memory coalescing | 有（MTE 连续段）| **等价替换** | 连续 slab 拷贝（`T.copy` slice 形态，见 §1.6.3）|
| 隐式跨 dtype 拷贝 cast（#6）| 优化 | CUDA 拷贝语义 | 部分（Developer 支持，但见 TRAP-C12）| **等价替换** | 改用**显式 `T.vcast(round_mode="rint")`**（可控舍入、可核验）；依据 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例 2（Developer）+ `.../traps-compiler.md` TRAP-C12（否决把隐式 cast 当融合优化）|

> 依据：映射表 `references/migration-analysis.md` §5.3（shared→UB、三维 Kernel→一维重新设计）。**「舍弃」项 1 个**（#3 `make_dq_layout`，GPU 专属且无计算语义）；**「重新设计」项 3 个**（#1 并行结构、#2 tile、GM→GM 结构）；**「保留」3 项**；**「等价替换」2 项**。

### 0.6 NPU 算法重设计

> §0.5 判定为「重新设计」的 3 项（R1 并行结构、R2 tile、R3 UB 中转）如下。**无算法族替换**——Phase R（§1.6.0）调研确认基线（显式 UB 中转的逐元素 cast）即为最优候选，故重设计只做**并行结构 / 搬运路径 / tile 形状**层面的落地，不改数学（唯一数值操作仍是 RNE cast）。源算法优化意图（#1–#6）承接见每项末。

**重设计项 R1：并行结构（三维 Kernel → 一维 Kernel + 持久化分核）**

- **源方案**：`T.Kernel(ceildiv(S,64), heads, batch)` 三维 grid，硬件直接给出 `(bx,by,bz)`，逻辑核数 = `ceil(S/64)·H·B`（如 [2,2048,32,128] = `32·32·2 = 2048`）。意图：零索引开销的三维任务映射 + 充分并行。
- **NPU 新算法**：`T.Kernel(nblk, is_npu=True) as (cid, _)` 一维；`nblk = min(40, total_tiles)`（物理 Vector 核数，§5.5）；核内 `T.serial(iters)`（`iters = ceildiv(total_tiles, 40)`，编译期静态）轮转处理逻辑 tile；每个逻辑任务 `t = cid + it*40` 分解为 `(bz, ss)`（§3.3）。意图承接：把源方案「零开销三维映射」替换为「少量 block 索引标量分解 + 一维 grid」，并按 §5.5 摊薄核启动开销。
- **语义保持论证**：任务划分只影响「谁搬哪一块」，不影响任何一块的转换结果（每块独立，元素一一映射，无跨任务依赖）；`dq_out` 每个元素恰被一个逻辑任务写一次（`t < total_tiles` 守卫）。逐元素映射无累加顺序，故无舍入差异（与源逐元素一致）。

**重设计项 R2：tile 形状（`[blk=64, dim]` 逐块 → 每批连续 slab `[R_S, H, D]`）**

- **源方案**：每 block 搬 `dq[bz, bx·64:(bx+1)·64, by, :]`（固定 head `by`，s 方向 64 行、全 D）——tile 是「固定 head 的 strided 区域」。意图：三维 grid 下每 thread block 负责一片 (s, d) 矩形。
- **NPU 新算法**：**利用「输入输出同形、算子逐元素」这一事实，把 tile 改为每 batch 内一段完全连续的 slab** `dq[bz, ss:ss+R_S, 0:H, 0:D]`（形状 `[R_S, H, D]`）。因 BSHD 内存序为 `(b,s,h,d)` row-major、且 slab 取全 H/D，该区域在 GM 中**物理连续** → 一次长连续 DMA（`R_S·H·D·4` 字节读 / `·2` 字节写）。`R_S` 编译期静态、整除 `seq_len`（§5.4）。意图承接：源方案「固定 head 的 64 行矩形」是为三维 grid 服务的，NPU 一维化后无此约束；改用连续 slab 保留「一次大块搬运」的原始意图并**改善合并访存**（长连续段 vs D·4=512B 的 strided 行段）。
- **语义保持论证**：slab 覆盖的 (s,h,d) 集合是全部 `s∈[s0,s0+R_S)`、全部 `h`、全部 `d`，与逐元素映射一一对应、无遗漏无重复；形状重排不改变任何元素的 cast。边界：`R_S | seq_len` 由构造保证（§5.4），**无尾块 / 无 padding**。nan/inf 传播与源一致（无特殊处理）。

**重设计项 R3：搬运路径（GM→GM → GM→UB→GM 三段）**

- **源方案**：`T.copy(dq_slice, dq_out_slice)` 是 GM→GM 直连（GPU 允许）。意图：一次拷贝完成读+转+写。
- **NPU 新算法**：Developer 模式 `T.copy` 不支持 GM→GM（`T.copy.md` §2.3 第 5 条），必须经 UB 中转，拆为三段：(a) `T.copy(dq[slab], dq_f32_ub)`（GM→UB，fp32→fp32）；(b) `T.vcast(dq_f32_ub, dq_f16_ub, round_mode="rint")`（UB→UB，显式 RNE cast）；(c) `T.copy(dq_f16_ub, dq_out[slab])`（UB→GM，fp16→fp16）。意图承接：源「一次完成读+转+写」的意图在 NPU 上以「UB 中转 + 显式转换」实现（UB 中转是硬约束；显式 `T.vcast` 出于 TRAP-C12 的可控/可核验考虑，见 §1.6.1）。
- **语义保持论证**：三段串联对每个元素执行 `GM读 → 恒等(UB) → RNE cast → GM写`，与源 GM→GM cast 的逐元素结果**完全相同**（唯一数值操作都是 RNE fp32→storage 转换；`T.vcast(round_mode="rint")` 与源 CUDA 拷贝的 RNE 一致，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1）。UB 中转是恒等搬运，不引入任何数值变化。边界与 nan/inf 传播不变。

**§0.4 优化意图承接核查**：#1→R1（一维重设计）；#2→R2（tile 重算）；#3→**舍弃**（GPU 专属、无计算语义，丢弃不损语义）；#4→保留；#5→R2（连续 slab 改善合并访存）；#6→**等价替换**为显式 `T.vcast`（R3(b)）。**无优化意图静默丢弃**。

### 0.7 标杆实现

- 源算子路径：`/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py`（`_flashattn_bwd_postprocess_kernel` L106–128 / `FlashAttnBwdPostprocessKernel` L131–176）。
- 源仓参考实现（golden 依据）：`gqa_bwd.py` L113/L118–119/L123–126 语义 —— `dq_out[b,s,h,d] = cast_storage(dq[b,s,h,d])`，RNE；Op 侧 `gqa.py` L1382。
- 迁移抽取副本：`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/_grouped_query_attention_bwd_kernels.py` L193–215。
- NPU wrapper/集成目标：`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd.py` L211–257（`FlashAttnBwdPostprocessKernel.forward` 契约：`dq (B,S,H,D) fp32 -> (B,S,H,D) fp16/bf16`）。
- 同任务前序实现（最佳参照、已真机通过）：`examples/grouped_query_attention_bwd/_flashattn_bwd_preprocess_kernel/DESIGN.md` 与同目录 `_flashattn_bwd_preprocess_kernel.py`（1-D persistent + `T.serial` + `if t<total_tiles` 守卫骨架）。
- golden **以 §0.1 语义为唯一依据**，不复刻 §0.6 NPU 算法。

---

## 1. 概述

### 1.1 算子名称

`_flashattn_bwd_postprocess_kernel`（GQA backward postprocess：fp32 `dq` 累加器转存储 dtype，NPU 侧实现文件名 `_flashattn_bwd_postprocess_kernel.py`）。

### 1.2 功能描述

把注意力反向主 kernel 产出的 fp32 `dq` 累加器**逐元素转换为存储 dtype（fp16/bf16）**，作为 GQA 反向的最终 `dq` 输出。**不含缩放、不含 `delta`、不含除法**（见 §0.1 语义澄清）。

### 1.3 数学公式

$$
\text{dq\_out}[b,s,h,d] \;=\; \mathrm{RNE}_{\text{storage\_dtype}}\big(\text{dq}[b,s,h,d]\big),\qquad b\in[0,B),\ s\in[0,S),\ h\in[0,H),\ d\in[0,D)
$$

输入 fp32、输出 fp16/bf16，逐元素一对一映射（无归约、无累加、无缩放）。

### 1.4 算法描述

> 本节描述**迁移决策后（§0.6）的 NPU 侧算法**，与 §1.6.0 调研选定算法一致。

1. **任务映射**：把 BSHD 张量按 batch 与 s 方向展平为 `total_tiles = B × (S/R_S)` 个逻辑 slab（每 slab = 固定 `b` 下连续的 `R_S` 个 s、全部 H、全部 D，共 `R_S·H·D` 个连续元素；`R_S = gcd(S, ceil-target)`，编译期静态且整除 S）。
2. **搬入**：对逻辑任务 `t=(b, s0)`，从 GM 读连续 slab `dq[b, s0:s0+R_S, 0:H, 0:D]` 到 UB fp32 buffer（长连续段）。
3. **转换**：UB 内显式 RNE cast `dq_f16 = rint(dq_f32)`（fp32→fp16/bf16）。
4. **搬出**：UB fp16 buffer 写回连续 slab `dq_out[b, s0:s0+R_S, 0:H, 0:D]`。
5. **持久化**：一维 grid 启动 `nblk = min(40, total_tiles)` 个 block，各核 `T.serial(iters)` 轮转处理逻辑 slab（§5.5）。

**与源算法的结构差异**（来源编号）：三维 grid → 一维持久化（§0.6 R1）；固定 head 的 64 行矩形 → 每批连续 slab `[R_S,H,D]`（R2）；GM→GM 直连 → GM→UB→GM 三段（R3）；隐式 cast → 显式 `T.vcast(rint)`（R3(b)）；`make_dq_layout` 布局声明**舍弃**（#3）。**数学语义（逐元素 RNE cast）与 dtype 语义不变**。

### 1.5 数据流图

```
GM[dq  (B,S,H,D) fp32] ──T.copy（连续 slab，slice 形态）──> UB[dq_f32 [R_S,H,D] fp32] ┐
                                                                                     │ T.vcast(round_mode="rint")
                                                                                     ▼
                                                    UB[dq_f16 [R_S,H,D] fp16/bf16] ──T.copy──> GM[dq_out (B,S,H,D) fp16/bf16]
```
（纯 Vector 路径：无 L1/L0A/L0B/L0C 参与；无 Cube。）

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**同一数学语义（fp32→storage dtype 的逐元素 RNE 转换，BSHD→BSHD）的算法族**。基线 = §0.3 源算法（GM→GM 隐式 cast）；本算子按 `references/algorithm-research.md` §2 属**单步逐元素 / 纯搬运类** → **轻量调研**（四问各一行结论 + 依据），但候选表与依据仍完整列出。

**R1 等价化简公式候选**（含基线；「等价」指数学等价或容差内等价，「化简」指更少的计算/访存/结构）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法）**：GM→GM 隐式 cast（NPU 落地：GM→UB→GM + 显式 `T.vcast(rint)`）| `dq_out = RNE(dq)`，1 次点态映射 | — | — | ✅（基线） |
| 1 | cast 融合：用**跨 dtype `T.copy`**（隐式 `hivm::VCastOp`）替代「显式 vcast + fp32 暂存」| 同 #0（RINT） | 数值等价（RINT=RNE） | 省 1 个 fp32 UB 暂存 buffer + 1 个 op（名义上）| ❌ **否决**：`.../pattern-library/traps-compiler.md` TRAP-C12（status=verified）——Developer GM→UB 的隐式 cast 仍发隐藏 `VCastOp`（**不减少 vector pass**）且**舍入不可控**，不得用作融合优化；与 `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条（隐式转换固定 RINT）一致 |
| 2 | 把 cast 折进主反向 kernel 的 epilogue（消除 fp32 `dq` 的落盘与回读）| 主 kernel 累加 `dq` 时顺带 cast | **不等价**（结构上不可行）| 理论省 `4N`(dq 写) + `4N`(dq 读) bytes | ❌ 否决：`dq` 由主 kernel 的 `T.atomic_add` 跨多个 (head,s-block) program 累加（`_grouped_query_attention_bwd_kernels.py` L182），**任一 program 都不拥有完整的 `dq` 元素**，cast 必须待全部累加完成 → 需跨 program 全局同步；GPU 源亦因此拆为独立 kernel（`gqa.py` L1381–1382）|
| 3 | 位截断替代 RNE（直接截断尾数位 / bitcast 直通）| `dq_out = trunc(dq)` | **不等价**（丢 RNE，引入 ≤1 ulp 系统性偏差）| 无流量收益 | ❌ 否决：与 §8.1 golden（RNE）不符 |
| 4 | 跳过本 kernel（恒等）| 无 | **不等价**（dtype 必须改变）| — | ❌ 不适用 |

**R2 在线算法**：**无**（结论有结构依据）。在线（online）变体适用于「存在跨 tile 的 running 统计量（max/sum）且需重缩放」的扫描/规约类算子（`references/algorithm-candidates.md`「规约类」条目仅对扫描/前缀型给出在线变体）。本算子是**纯点态映射**：每个输出元素 `dq_out[e]` 仅依赖**同一位置**的 `dq[e]`，元素间**无任何依赖**，不存在可增量维护的 running 状态，亦无两遍/多遍结构可供单遍化 → 不存在「online 化」的语义空间。（对照：已迁移的 preprocess 有行内规约，本 postprocess 连规约都没有。）

**R3 复杂度对比（四口径；基线在表；带宽受限算子以访存 / 遍数为主导）**：

| 算法候选 | FLOPs（cast op 数） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|--------------------|---------------|---------|-------------|---------------------|
| **#0 基线（UB 中转 + 显式 cast）** | `N` 次点态转换（N=B·S·H·D），无乘加 | `读 4N + 写 2N = 6N`（N 元素；无中间 GM 往返）| 1 | UB 级 `[R_S,H,D]` fp32 + fp16 = `6·R_S·H·D` ≤ 96KB（§4.5）| 元素级全并行；slab 间独立，无跨核归并 |
| #1 跨 dtype copy 融合 | `N`（隐藏 VCast 仍执行）| `6N`（同基线）| 1 | UB 级 fp16 `[R_S,H,D]`（省 fp32 暂存）| 同基线；**但舍入不可控（TRAP-C12）** |
| #2 折进主 kernel epilogue | — | 全局省 `8N`，但需跨 program 同步 | — | — | **需跨 program 全局同步**（本仓单 kernel 内无此机制）|
| #3 位截断 | `N` | `6N`（同基线）| 1 | 同基线 | 同基线；**数值不等价** |

> 口径说明：FLOPs 只计逐元素转换 op（无乘加）；访存量含输入读 + 输出写，**不含** TLB/指令开销；#0/#1/#3 的 GM 流量完全相同（`dq` 读一次、`dq_out` 写一次），差异只在 UB 驻留与舍入可控性。

**R4 硬件亲和性评估**（逐候选对照）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#0 基线** | Vector `VCastOp`（逐元素转换），天然匹配 | UB 96KB ≤ 192KB ✓ | slab 连续；D=64/128：fp32 行 256/512B、fp16 行 128/256B（≥32B ✓）；向量宽 fp32 ×4 / fp16 ×8，D%8=0 ✓ | `R_S | seq_len` 编译期静态 ✓ | 单遍，GM→UB→GM 链；`T.Pipelined` 该 UB 预算下可选（§6.3）| ✅ **选定** |
| #1 跨 dtype copy 融合 | Vector（隐藏 VCast）| UB 少 48KB | 同基线 | 同基线 | 同基线 | ❌ 否决：TRAP-C12（无 op 收益 + 舍入不可控，已实测证伪作融合用）|
| #2 折进主 kernel | 需跨 program 全局同步（NPU 单 kernel 无此机制）| — | — | — | 破坏主 kernel 的 atomic 累加结构 | ❌ 否决：结构不可行（见 R1 #2）|
| #3 位截断 | Vector（`vbitcast`/截位）| 同基线 | 同基线 | 同基线 | 同基线 | ❌ 否决：与 golden RNE 不等价 |

**调研结论**：**选定基线算法族 #0（UB 中转 + 显式 `T.vcast(round_mode="rint")` 的逐元素 cast）**。关键依据：R3 表中 #0/#1/#3 的 GM 流量完全相同（该算子**天然带宽受限**，FLOPs 非瓶颈），而 #1 被 TRAP-C12 证伪为无 op 收益且舍入不可控、#2 结构不可行、#3 不等价 —— 故「**无更优替代**」。与基线的结构差异仅在于实现层：并行结构一维化（§0.6 R1）、tile 改连续 slab（R2）、搬运拆为 GM→UB→GM（R3）。**调研范围**：`references/algorithm-candidates.md`（单步逐元素/纯搬运类条目）+ 恒等变形自查（cast 融合 / epilogue 融合 / 位截断）+ `pattern-library/`（traps-compiler.md TRAP-C12、traps-runtime.md TRAP-load-nd2nz-strided、constants.md `CONST-elu-910B3-bw-floor`）+ `docs/Tilelang.language/`（T.copy / T.vcast）+ `examples/` 同类（`examples/indexer/indexer_bwd.py`、`examples/elu/_make_elu_kernel/_make_elu_kernel.py`、`examples/TileOPs/.../_gqa_prefill_fwd_kernel.py`）+ 源算子结构分析；未使用互联网检索（本地源已覆盖该点态语义的全部候选）。

**设计期 roofline 估算下界行**（协议 `.agents/skills/_shared/standards/hardware-cost-model.md` §2；常数取 `.../pattern-library/constants.md`，含版本戳）：

| 项 | 代表 shape `[2,2048,32,128]` fp32→fp16（N = 16.78M 元素）| 口径与常数 |
|----|-------------------------------------------------------|-----------|
| 流量项 | 读 `16.78M × 4B = 67.1 MB` + 写 `16.78M × 2B = 33.6 MB` = **100.7 MB** → `100.7MB / 2.49TB/s ≈ 40.4 µs` | 常数 `CONST-elu-910B3-bw-floor`（**同设备 Ascend910B3**，逐元素聚合 R+W 地板 ≈ 2.49 TB/s；toolchain tilelang 0.1.2+75e21fe37 / CANN 9.0.0 / 2026-10-15）。**下限** |
| 发射项 | 每 slab 1 次 `vcast`（+2 次 DMA）；按 `CONST-vector-op-cost-910_9362`「普通 v-op ≈3.4–7.8 µs / 16m-elem（bs7168）」外推 16.78M 元素 ×1 op ≈ **3.4–7.8 µs**（< 流量项）| 常数 `CONST-vector-op-cost-910_9362`（**跨 SoC 外推，已标注**：910_9362 ↔ 910B3 同为 20 AI/40 Vector 核，但发射常数未在 910B3 重验）|
| 容量项 | UB 96 KB < 192 KB → 不构成约束 | `CONST-capacity-910B2C`（UB=192KB，9362/910B3 数值一致注记）|
| **估算下界** | **max(40.4, ~7.8, —) ≈ 40.4 µs**（若 slab 落入小 tile 退化区则上界更高：按 `CONST-elu-910B3-bw-floor` bs 曲线 bs=4096 ≈ 1.0–1.2 TB/s → ~84–100 µs）| 取流量项为主导 |

> 未实测假设：`CONST-vector-op-cost-910_9362` 的发射常数在 910B3 上未重验（标注「未实测外推」）；发射项占比 <20%，不改变下界结论。本设计将 slab 目标元素数定在搬运带宽膝点（§5.2），以避开小 tile 退化区。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线 #0）。逐项给出四要素；无采纳项时给出结论与依据。

| # | 候选优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 | 是否采纳 |
|---|-----------|------|-----------|-----------|---------|---------|
| 1 | cast 融合（跨 dtype `T.copy` 取代显式 `T.vcast` + fp32 暂存）| `dq_out = RNE(dq)`（三段：load-cast-store）| `dq_out = RNE(dq)`（两段：fused-loadcast / store）| 数值等价（RINT=RNE），但**舍入不可控** | 名义省 48KB UB、省 1 op | ❌ **不采纳**：`.../pattern-library/traps-compiler.md` TRAP-C12（verified）实测——隐式 cast 仍发 `VCastOp`（**不减少 vector pass**），且舍入固定不可控；`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条同口径。**无实际收益** |
| 2 | 位截断替代 RNE | `RNE(dq)` | `trunc(dq)` | **不等价**（丢 RNE） | 无流量收益 | ❌ 不采纳（数值不等价）|

**优化结论**：**无优化空间**（no optimization space）。依据：本算子是「单一逐元素 RNE 转换」的**最小结构** —— ① 无乘/加/除/开方/超越函数，不存在降代价代换（如除法转倒数、rsqrt）空间；② `RNE(dq)` 仅出现一次，无公共子表达式；③ 输入读一次、输出写一次，已是**不可消除的 I/O 下界**（§1.6.0 R3）；④ 无归约，无归约合并空间；⑤ 唯一可能省 buffer/op 的候选（#1 cast 融合）经 TRAP-C12 实测否决（不减少 vector pass）。故**无数学等价优化项被采纳**，§3.1 直接采用原式 `dq_out = RNE(dq)`。

> **等价性机器验证（D-1）说明**：本节**无采纳优化项** → 按 gate 规则（`S1-EQUIV-EXEC`：§1.6.1 写明「无优化空间」时豁免）本算子**不产出 `verify_equiv.py`**（无可对照的「候选式 vs 基线式」）。舍入语义的正确性由 §8.1 golden（PyTorch `.to()`，RNE）在 Stage 3 精度测试中直接覆盖。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 cast `dq_out = RNE(dq)` | 标量逐元素赋值 | `T.vcast(dq_f32_ub, dq_f16_ub, round_mode="rint")`（`hivm::VCastOp`，向量逐元素）| ✅ | — |
| 2 | GM→UB 搬入 | 逐元素标量读 | `T.copy(dq[bz, s0:s0+R_S, 0:H, 0:D], dq_f32_ub)`（MTE2，连续段）| ✅ | — |
| 3 | UB→GM 搬出 | 逐元素标量写 | `T.copy(dq_f16_ub, dq_out[bz, s0:s0+R_S, 0:H, 0:D])`（MTE3，连续段）| ✅ | — |
| 4 | 逻辑任务索引分解 `(bz, s0) ← t` | block 级标量除/取模 | 无 | ❌ | **block 级索引 / 任务映射计算**：每逻辑任务仅 1 次 div + 1 次 mod，发生在 block 入口（非逐元素热点），无逐元素等价向量 API（分解对象是标量任务号，非张量）|
| 5 | 持久化轮转 `t = cid + it*40` | block 级标量乘加（`T.serial` 内）| 无 | ❌ | 同 #4：block 级循环变量计算，每迭代 1 次乘加；此处的 `T.serial` 是**任务级调度循环**（§6.1），非逐元素循环，不属向量化对象 |
| 6 | 尾块 / 越界守卫 `if t < total_tiles` | 标量比较 | 无 | ❌ | **block 级控制流**（任务级守卫，见 §0.6 R1）；对象是标量任务号；且 `R_S | seq_len`（§5.4）故无逐元素边界分支 |

**向量化结论**：**唯一逐元素计算点（#1 cast）已完全向量化**（`T.vcast`，佐证 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例 1/2；搬入搬出 #2/#3 由 MTE 向量化搬运完成，佐证 `docs/Tilelang.language/内存操作/T.copy.md` §2.4）。**保留 3 处标量/循环**（#4/#5/#6），理由类别均为 **block 级索引 / 任务映射**（含任务级调度循环与守卫），逐项见上表，无「实现简单」类理由。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`dq`/`dq_out` 均为 BSHD，且**同形同序**）；核内布局是自由变量。本算子非 batch 维度有 3 个（S、H、D），但**算子逐元素且输入输出同布局** —— 故可把 (S,H,D) 视为可自由切分的连续空间。

**轴质量评分**：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | 最内连续轴对向量宽度 | D=64/128：fp32 行宽 = D·4 = 256/512B、fp16 行宽 = D·2 = 128/256B（≥32B ✓）；向量宽 fp32 ×4（D%4=0 ✓）、fp16 ×8（D%8=0 ✓）|
| 尾 lane 浪费率 | `(ceil(D/vw)·vw − D)/D` | 0%（D=64/128 均为 8 的整数倍）|
| 累加链形态 | 无累加（逐元素点态映射）| 无累加链 → 无「跨步系数阻碍向量化」问题（与 preprocess 不同）|
| repack 代价 | 改布局的重排代价 | 方案 1/2 均**无需 repack**（输入输出同布局，逐元素）|
| UB 容量影响 | 驻留缓冲尺寸 | 方案 1：`fp32 [R_S,H,D] + fp16 [R_S,H,D]` = 96KB（R_S·H·D=16384）；≤192KB ✓ |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **每批连续 slab `[R_S,H,D]`**（取全 H、全 D；GM 中物理连续）| 最内连续 D（实际为整块连续 (H,D) 展平）| 无 | 读：单段长连续 `R_S·H·D·4` B（如 64KB）；写：单段长连续 `R_S·H·D·2` B（32KB）；落在搬运带宽膝点（§5.2）| ✅ **选定** |
| 2 | 固定 head 的 strided `[R,D]`（preprocess 风格）| D | 无 | GM 流量**完全相同**，但读为 `D·4=512B`/行的 strided 段、写为 `256B`/行 → 更多 MTE 描述符，长连续段收益丢失；且本算子**无 head 归约需求**（逐元素），无采用该形态的理由 | ❌ 弃选：无收益、段更短（对照 `CONST-elu-910B3-bw-floor` 的 bs↔带宽曲线：tile 越大有效带宽越高）|
| 3 | 展平为 1-D `(N,)` 核（最自然的点态形态）| 全连续 | 无 | 与方案 1 的 GM 连续性等价（slab 本身连续）| ❌ 弃选：**需改 kernel 签名/契约**（wrapper `forward(self, dq)` 传 4-D 张量，`grouped_query_attention_bwd.py` L255–257），且相对方案 1 无额外收益 → 不引入接口风险 |
| 4 | 核内转置 / 布局重排（`T.transpose`）| — | 核内重排 | 逐元素算子无需转置，纯增代价 | ❌ 弃选：零收益；`T.transpose` 为 UB 级两轴交换链（`docs/Tilelang.language/创建操作/T.transpose.md`），此处无目标 |
| 5 | host 侧 permute / reshape 进 kernel | — | host 拷贝 | 违反 host 约束（仅允许 stride/shape 元数据视图，`references/ascend-constraints.md` §4），且 ~百 µs 级 ≫ 算子本体 40 µs | ❌ 弃选 |

**布局决策结论**：**选定方案 #1** —— 核内布局 = **每批连续 slab `[R_S,H,D]`**（BSHD 中固定 `b`、连续 `R_S` 个 s、全 H/D，物理连续），**向量化/搬运轴 = 最内连续 D（实为整块连续 (H,D) 展平）**，无 repack。理由：(a) 算子输入输出同布局且逐元素 → (S,H,D) 可自由切分，取连续 slab 得**长连续 DMA 段**（优于方案 2 的短 strided 行段）；(b) 无累加链，无轴选择的数值风险；(c) 保持 4-D 契约（方案 3 需改接口）。(4) 该结论同步落入 §3.3 伪代码（`T.copy` slab 形态）与 §6.1 循环结构。

> 弃选论证证据：方案 2 的「段更短」依据 `CONST-elu-910B3-bw-floor`（同设备 bs↔带宽实测曲线，tile 越大有效带宽越高）；方案 3 的接口约束依据 wrapper `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd.py` L255–257；方案 4 依据 `docs/Tilelang.language/创建操作/T.transpose.md`（核内两轴交换链，非 GM 级重排）；方案 5 依据 `references/ascend-constraints.md` §4（未凭空断言）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（用户指定；且与算子特征一致）。

### 2.2 选型理由

- **纯 Vector 算子**：仅逐元素 cast + 搬运，无 matmul → 不需要 Cube/L0C，不需要手动 L1/L0 分配与 `sync_block_set/wait` 跨引擎同步（Expert 模式的收益点）。
- **无跨引擎流水**：数据流为 `GM→UB→GM` 单链，Developer 的自动同步 / 自动 buffer 管理足够。
- 同任务前序函数 `_flashattn_bwd_preprocess_kernel`（Developer，1-D persistent + `T.serial` + 守卫）已在真机验证可编译可跑，本函数结构更简单（无归约）→ 同模式最省风险。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared([R_S,H,D], "float")`（fp32 中转）+ `T.alloc_shared([R_S,H,D], dtype)`（fp16/bf16 输出暂存）——UB 级 |
| 计算方式 | `T.vcast(dq_f32_ub, dq_f16_ub, round_mode="rint")`（`hivm::VCastOp`）|
| 同步 | 自动同步（§7.1），无手动 flag |
| 流控 | `T.serial` 任务级循环（§6.1），无 `T.Pipelined`（UB 预算下可选，§6.3）|

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的（无优化空间的）原式 `dq_out = RNE(dq)`。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `t = cid + it·N_PHYS`, `(bz, s0) = (t // n_s, (t % n_s)·R_S)` | 逻辑任务号 → (b, s-slab) |
| 2 | `dq_f32_ub ← dq[bz, s0:s0+R_S, 0:H, 0:D]` | 搬入连续 slab（GM→UB，fp32）|
| 3 | `dq_f16_ub ← RNE(dq_f32_ub)` | 逐元素 RNE cast（fp32→fp16/bf16）|
| 4 | `dq_out[bz, s0:s0+R_S, 0:H, 0:D] ← dq_f16_ub` | 搬出连续 slab（UB→GM）|

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 任务索引分解 | 标量算术（`cid`、`//`、`%`）| `bz = t // n_s`、`ss = (t % n_s) * R_S` | Developer |
| 2 | 搬入 | `T.copy(dq[bz, ss:ss+R_S, 0:H, 0:D], dq_f32_ub)` | slice 形态（连续 slab）| Developer |
| 3 | RNE cast | `T.vcast(dq_f32_ub, dq_f16_ub, round_mode="rint")` | 同 shape、fp32→fp16/bf16 | Developer |
| 4 | 搬出 | `T.copy(dq_f16_ub, dq_out[bz, ss:ss+R_S, 0:H, 0:D])` | slice 形态（连续 slab）| Developer |
| — | 任务级循环 | `T.serial(iters)` | `iters` 编译期静态值 | Developer |

### 3.3 计算伪代码

> 注意：`T.Tensor` 的 dtype 直接作为第二个位置参数传入（不用 `dtype=` 关键字），避免 `name 'in_dtype' is not defined` 类 false alarm。

```python
@tilelang.jit(out_idx=[1], target="npuir")
def _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype="float16"):
    accum_dtype = "float"
    _TILE_ELEMS = 16384                       # 目标 slab 元素数（搬运带宽膝点，§5.2）
    _rs_max = max(1, _TILE_ELEMS // (heads * dim))
    R_S = math.gcd(seq_len, _rs_max)          # 编译期静态；整除 seq_len
    n_s = seq_len // R_S
    total_tiles = batch * n_s
    N_PHYS = 40                               # 纯 Vector 物理核数（实查 20×2）
    nblk = min(N_PHYS, total_tiles)
    iters = (total_tiles + nblk - 1) // nblk  # 静态循环边界

    @T.prim_func
    def flash_bwd_post(
        dq: T.Tensor((batch, seq_len, heads, dim), accum_dtype),      # fp32 BSHD
        dq_out: T.Tensor((batch, seq_len, heads, dim), dtype),        # storage dtype BSHD
    ) -> None:
        with T.Kernel(nblk, is_npu=True) as (cid, _):
            dq_f32 = T.alloc_shared([R_S, heads, dim], accum_dtype)   # fp32 中转
            dq_f16 = T.alloc_shared([R_S, heads, dim], dtype)         # 输出暂存
            for it in T.serial(iters):
                t = cid + it * nblk
                if t < total_tiles:
                    bz = t // n_s                    # batch 索引
                    ss = (t % n_s) * R_S             # s slab 起点
                    # 2. 搬入连续 slab（GM -> UB）
                    T.copy(dq[bz, ss:ss + R_S, 0:heads, 0:dim], dq_f32)
                    # 3. 逐元素 RNE cast（向量）
                    T.vcast(dq_f32, dq_f16, round_mode="rint")
                    # 4. 搬出连续 slab（UB -> GM）
                    T.copy(dq_f16, dq_out[bz, ss:ss + R_S, 0:heads, 0:dim])

    return flash_bwd_post
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(n, is_npu=True)` 一维 | `docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 2；`examples/indexer/indexer_bwd.py` L53 邻域 | 现有先例 |
| `T.alloc_shared` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例 2（Developer，`T.alloc_shared`）| 文档化 |
| `T.copy` slice 形态（4-D 张量、2 标量索引 + 2 slice）| `docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 1/2；`_gqa_prefill_fwd_kernel.py`（BSHD slice copy）| 文档化 + 先例 |
| `T.vcast(src, dst, round_mode="rint")`（fp32→fp16/bf16）| `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f32→f16`/`f32→bf16` 支持 rint）+ §2.4 示例 1/2 | 文档化 |
| `T.serial` 静态边界 | `docs/Tilelang.language/` + `examples/elu/_make_elu_kernel/_make_elu_kernel.py` L151–157 | 现有先例 |
| `math.gcd`（host 侧编译期）| Python 标准库 | — |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是（源为 3D grid，L121）** | 展开为一维 `total_tiles`；block 索引静态分解（§0.6 R1）|
| GM→GM 拷贝不支持（Developer）| **是（源 step 3 为 GM→GM，L123–126）** | 经 UB 中转：GM→UB→GM（§0.6 R3；依据 `T.copy.md` §2.3 第 5 条）|
| Expert 模式通用路径禁止跨 dtype copy | 否（**Developer 模式**，且改用显式 `T.vcast`，不依赖隐式跨 dtype copy）| — |
| GEMM 非整除 / L0C 溢出 | 否 | 无 Cube 计算 |
| 尾轴对齐（32B）| 否 | D=64/128 → 行 128–512B ≥ 32B ✓ |
| 动态 shape | 否 | 全部 shape 编译期常量（源 factory 签名 L107–111）|

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | `T.Kernel(ceil(S/64), heads, batch)` 三维（L121）| 一维 `T.Kernel(nblk)` + 索引分解 | §0.6 R1 |
| tile 形状 | 固定 head 的 `[64, dim]` strided 矩形 | 每批连续 slab `[R_S,H,D]`（R_S·H·D=16384）| §0.6 R2 |
| 搬运路径 | GM→GM 直连（L123–126）| GM→UB→GM 三段（UB 中转硬约束）| §0.6 R3 |
| 转换方式 | 隐式跨 dtype 拷贝（CUDA RNE）| 显式 `T.vcast(round_mode="rint")` | §0.6 R3(b)（TRAP-C12）|
| 布局声明 | `T.annotate_layout({dq: make_dq_layout(dq)})`（L122）| **舍弃**（NPU 无该 fragment 布局需求）| §0.5 #3 |
| 分核 | 3-D grid → 2048/4096 逻辑核（SM 自动调度）| 固定 ≤40 block + `T.serial` | §5.5 |
| `threads=128` | 显式（L121）| **移除**（NPU 执行细节）| K9 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/grouped_query_attention_bwd/_flashattn_bwd_preprocess_kernel/DESIGN.md` | 高（同任务前序）| 1-D persistent + `T.serial` + `if t<total_tiles` 守卫骨架；`gcd` 静态 tiling；slice 形态 `T.copy` |
| `examples/elu/_make_elu_kernel/_make_elu_kernel.py` | 高（迁移件）| 整块 `T.vcast(src,dst,round_mode=...)` 张量 dtype 提升/转换的惯用形态；`T.serial` 静态持久化 + 守卫先例（L151–157）|
| `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | 高（API）| fp32→fp16/bf16 的 `T.vcast` Developer 用法（§2.4 示例 2）|
| `examples/indexer/indexer_bwd.py` | 中 | Developer 纯 Vector buffer 形态；连续写回 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dq` | `(B, S, H, D)` | **float32** | BSHD，D 连续；主反向 kernel 的 fp32 累加器 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dq_out` | `(B, S, H, D)` | float16 / bfloat16 | BSHD，与 `dq` 同形同序；`out_idx=[1]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `dq_f32` | `[R_S, H, D]` | fp32 | UB | 输入 slab 中转 |
| `dq_f16` | `[R_S, H, D]` | fp16/bf16 | UB | cast 结果暂存 |

### 4.4 内存搬运路径

```
GM[dq  BSHD fp32]  --T.copy(slice)--> UB[dq_f32 [R_S,H,D] fp32]  ┐
                                                                  │ T.vcast(round_mode="rint")
                                                                  ▼
                                        UB[dq_f16 [R_S,H,D] fp16/bf16] --T.copy(slice)--> GM[dq_out BSHD]
```
纯 Vector 路径，无 L1/L0A/L0B/L0C；单级 GM↔UB 搬运，无中间落 GM。

### 4.5 UB 内存预算

**slab 元素数 = R_S·H·D = 16384（各 manifest workload 一致，§5.2）**：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `dq_f32` | (R_S,H,D)，R_S·H·D=16384 | fp32 | 65536 |
| `dq_f16` | (R_S,H,D)，R_S·H·D=16384 | fp16 | 32768 |
| **总计** | | | **98304 ≈ 96 KB** / 192 KB（Ascend910B3 实测上限）|

- 余量：`192KB − 96KB = 96KB`；按 `CONST-capacity-910B2C` 的 UB 膨胀系数上偏口径 ×1.12 → 107.5KB < 192KB ✓（该系数为 auto-multi-buffer 关闭时的实测，见 `.../pattern-library/traps-compiler.md` TRAP-UB-multibuffer-inflation 第四证；910B3 未单独重验，已保守采用）。
- 保守变体（若 Stage 3 遇 UB 膨胀）：把 `_TILE_ELEMS` 降到 12288 → slab 12288 元素、UB 72KB（本设计主选 16384）。

### 4.6 动态轴定义

**无动态轴**。`batch / heads / seq_len / dim` 均为编译期常量（源 factory 签名 L107–111）；`R_S`、`total_tiles`、`iters` 均在编译期求得。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
def _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype="float16"):
    ...
```
（沿用源 `out_idx=[1]`；**移除** `threads=128`（K9，CUDA 执行细节）；无 `pass_configs`/`compile_flags` —— 源无额外编译标志。）

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。
**判定依据**：仅含逐元素转换（`T.vcast`）与搬运（`T.copy`），无 matmul → 无需 Cube/L1/L0/L0C，数据流 `GM→UB→GM`。

### 5.2 Block 划分

```python
_TILE_ELEMS = 16384                       # 目标 slab 元素数
_rs_max = max(1, _TILE_ELEMS // (heads * dim))
R_S = math.gcd(seq_len, _rs_max)          # s 方向 slab 行数（编译期静态，整除 seq_len）
n_s = seq_len // R_S                      # s 方向 slab 数
total_tiles = batch * n_s                 # 逻辑核数
N_PHYS = 40                               # 物理 Vector 核数
nblk = min(N_PHYS, total_tiles)
iters = ceildiv(total_tiles, nblk)        # 核内串行迭代数（静态）
```
- `R_S` 选择理由：① 由 `gcd` 构造保证整除全部 shape 的 `seq_len`（→ 无尾块，§5.4）；② 使 slab 元素数 ≈ 16384，落在 `CONST-elu-910B3-bw-floor` 实测搬运带宽膝点（bs 10240→1.48、12288→1.53 TB/s；再增大收益递减），避开 bs=2048（4KB tile → 0.64 TB/s）退化区；③ UB 96KB（含 ×1.12 → 107.5KB）< 192KB。

逐 workload（slab 元素数恒为 16384）：

| workload（B,S,H,D）| H·D | `_rs_max` | `R_S` | n_s | total_tiles | UB (KB) |
|---|---|---|---|---|---|---|
| `[1,128,8,64]`（smoke）| 512 | 32 | `gcd(128,32)=32` | 4 | 4 | 96 |
| `[4,512,32,128]` | 4096 | 4 | `gcd(512,4)=4` | 128 | 512 | 96 |
| `[2,2048,32,128]` | 4096 | 4 | `gcd(2048,4)=4` | 512 | 1024 | 96 |
| `[2,512,64,128]` | 8192 | 2 | `gcd(512,2)=2` | 256 | 512 | 96 |
| `[1,2048,64,128]` | 8192 | 2 | `gcd(2048,2)=2` | 1024 | 1024 | 96 |

### 5.3 约束分析

- **对齐约束**：D=64/128，fp32 行宽 256/512B、fp16 行宽 128/256B ≥ 32B ✓；向量宽 fp32 ×4 / fp16 ×8，D%4=0 / D%8=0 ✓。
- **UB 容量**：96KB（含 ×1.12 → 107.5KB）< 192KB ✓。
- **L0 容量**：无 Cube 计算，不适用。

### 5.4 注意事项（非整除处理）

- **本算子以「编译期选 R_S 整除 seq_len」彻底消除尾块**：`R_S = gcd(seq_len, _rs_max)` 由构造保证 `seq_len % R_S == 0`，因此**无 padding、无动态边界、无尾块掩码**。manifest 全部 shape 的 total_tiles 均为整数、无残块。
- `seq_len < _rs_max` 时 `R_S = seq_len`（退化为 1 个 slab）；`seq_len` 非 2 的幂时 `R_S` 取 `gcd`（如 1000 → 视 `_rs_max` 取 1/2/4/8），仍静态、无尾块。
- slab 的 (s,h,d) 覆盖为「固定 b、连续 R_S 个 s、全 H、全 D」，与元素一一映射，无跨界。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 `docs/开发指南.md` §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录；纯 Vector 算子核数翻倍。

**① 逻辑核数计算**

```
num_logical_kernels = batch × ceil(seq_len / R_S) = batch × n_s
```
逐 workload（§5.2 表）：smoke `4`、`[4,512,32,128]`→`512`、`[2,2048,32,128]`→`1024`、`[2,512,64,128]`→`512`、`[1,2048,64,128]`→`1024`。

**② 物理核数依据（实查）**

- 查询代码：`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()`
- **实际返回值：`20`**（设备 `torch_npu.npu.get_device_name(0)` = **Ascend910B3**；`npu-smi info` 同证）。
- 本算子为**纯 Vector**（无 Cube）→ **物理核数 = 20 × 2 = 40**（AIV）。记 `N_PHYS = 40`。*（禁止文档假设/经验值替代——此处为实查 20 后按标准翻倍。）*

**③ 规模判定与分核方案**

- **规模判定：极大规模**。最大 `num_logical_kernels = 1024`，为物理核数 40 的 **25.6 倍**；且无法通过增大 `R_S` 缩减到接近 40（`R_S` 受 UB 硬上限与搬运带宽膝点约束：slab 元素数已取 16384，再增大收益递减且逼近 UB 上限）。→ 适用标准 §1 要素③ **「极大规模——核内 `T.serial` 串行」**。
- **分核方案**：**固定启动内核数 `nblk = min(40, total_tiles)`；核内 `T.serial(iters)` 轮转处理多个逻辑 slab**：
  ```
  iters = ceildiv(total_tiles, nblk)     # 编译期静态边界（最大 workload: ceil(1024/40)=26）
  for it in T.serial(iters):
      t = cid + it * nblk                # 轮转（round-robin）任务映射
      if t < total_tiles: <处理 slab t>   # 尾轮守卫
  ```
  - 循环边界 `iters` 为**静态编译期常量**（满足标准「核内串行循环边界必须为静态值」）。
  - 效果：把 25.6 波次摊薄为 1 波（每核 ≤26 个串行 slab），消除 1024 次 block 启动开销中的绝大部分。
  - smoke（4 tiles）：`nblk = 4`、`iters = 1` → 属「逻辑核数 ≤ 物理核数」的无需适配情形（4 ≤ 40）。
- **残留负载不均（显式记录）**：`total_tiles` 恒为 2 的幂（4/512/1024），而 `nblk = 40 = 2³×5` 含非 2 因子，数学上**无法对齐任何整数倍**（与标准 §1 要素③「无法对齐兜底」的同源约束）。残留代价：1024 slabs 时 26×40 = 1040，**16 个核多处理 1 个 slab**（临界路径 26 vs 均值 25.6 → **+1.6%**）；512 时 13×40 = 520，8 核多 1 个（+1.6%）。该残留 < 2%，可接受；**是否改用 `nblk=32`（512/1024 的公约数，完美均衡但闲置 8 核）留待 Stage 4 实验裁决**（§9.3）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑 slab（持久化轮转）| 任务级串行 | `T.serial(iters)` | 极大规模核内串行，摊薄核启动开销（§5.5③）；静态边界 |
| 逐元素 cast | 向量化 | `T.vcast(round_mode="rint")` | 单次整块向量转换（§1.6.2 #1）|
| slab 搬入 / 搬出 | 向量化搬运 | `T.copy`（slice 形态）| MTE2/MTE3 连续段（§1.6.3 方案 #1）|
| 逻辑任务索引分解 | block 级标量 | `//`、`%` | 非逐元素热点（§1.6.2 #4）|

> 无逐元素标量循环；保留的 3 处标量点均在 §1.6.2 有充分理由（block 级索引 / 任务映射 / 任务级守卫）。

### 6.2 循环伪代码

```python
with T.Kernel(nblk, is_npu=True) as (cid, _):
    dq_f32, dq_f16 = <alloc>                       # §3.3
    for it in T.serial(iters):                     # 任务级串行（静态边界）
        t = cid + it * nblk
        if t < total_tiles:                        # 任务级守卫（§1.6.2 #6）
            bz = t // n_s; ss = (t % n_s) * R_S    # 索引分解（§1.6.2 #4）
            T.copy(dq[bz, ss:ss+R_S, 0:heads, 0:dim], dq_f32)
            T.vcast(dq_f32, dq_f16, round_mode="rint")
            T.copy(dq_f16, dq_out[bz, ss:ss+R_S, 0:heads, 0:dim])
```

### 6.3 流水线优化

**主选不使用 `T.Pipelined`**。理由：本算子是纯搬运链（MTE2 → Vector → MTE3），`T.serial` 下单 slab 内三段天然可被编译器流水化调度；`T.serial` + 守卫的组合已在前序函数真机验证。**Stage 4 可选**：若 profile 显示 MTE2/VEC/MTE3 未重叠导致低效，可在任务级循环上启用 `T.Pipelined(iters, num_stages=2)`（双缓冲需 2×96KB = 192KB = UB 上限，风险高；若启用应先把 `_TILE_ELEMS` 降到 8192 → 2×48KB = 96KB）。

### 6.4 尾块处理

**无尾块**：`R_S = gcd(seq_len, _rs_max)` 整除 `seq_len`（§5.4），slab 切片恒在界内；`total_tiles` 与 `nblk` 的余数由 `if t < total_tiles` 守卫处理（跳过多余任务，非部分 slab）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式默认）。无手动 flag / 无 `sync_block_set/wait`（单核内单链 `GM→UB→算→GM`，无跨引擎、无跨核依赖）。

### 7.2 同步点说明

无手动同步点。`T.copy`（MTE2）、`T.vcast`（Vector）、`T.copy`（MTE3）之间的依赖由编译器自动插入同步（Developer 模式标准行为）；`T.serial` 迭代间因 buffer 复用（WAR）亦由自动同步保证。

### 7.3 pass_configs 配置

```python
pass_configs = {}      # 与同步相关的 pass 配置：无
```
源无额外 pass 配置；无 CUDA 编译标志需舍弃。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据（逐元素 RNE cast），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。

```python
def golden_flashattn_bwd_postprocess(dq: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """dq: (B,S,H,D) fp32 -> (B,S,H,D) out_dtype（逐元素 RNE 转换）。"""
    # 纯逐元素 cast（与源 T.copy 的隐式转换语义一致，RNE）；以 torch API 为权威参考
    dq_cpu = torch.as_tensor(dq.detach().cpu())
    return torch.as_tensor(dq_cpu.to(torch.float32)).to(out_dtype)   # [B,S,H,D] 同形
```

### 8.2 精度标准

| dtype（输出）| atol | rtol | 说明 |
|-------|------|------|------|
| float16 | 1e-3 | 1e-3 | 纯 cast，两侧均 RNE；理论逐位一致，容差覆盖极值/舍入边界 |
| bfloat16 | 1e-2 | 1e-2 | bf16 量化粒度更大，同上 |

> 上游 op 端到端测试 `examples/TileOPs/tests/ops/test_grouped_query_attention_bwd.py` L122 用 `atol=5e-3, rtol=1e-5`；本 kernel 单独验证用上表。

**L0 门槛测试计划**：
1. **主用例**（覆盖全部 manifest workloads）：`[1,128,8,64]`、`[4,512,32,128]`、`[2,2048,32,128]`、`[2,512,64,128]`、`[1,2048,64,128]`，fp16 与 bf16 各一遍，与 §8.1 golden 比对（assert 全通过）。
2. **边界用例**：`seq_len=1000`（非 2 的幂，验证 gcd 选 `R_S`）；`B=1,H=1`；`D=64`。
3. **数值边界**：`dq` 含 `0`、`±0`、fp32 极值、fp16 溢出阈值附近值（`±65504`、`±65520`→±inf）、subnormal、全零。
4. Layer 1（单 kernel L0 精度+契约）、Layer 2 由 `tilelang-op-develop` 承接；完整 L1/L2/Boundary 套件交下游 skill。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **UB 硬上限 192KB**：slab 16384 元素时 96KB（×1.12 → 107.5KB），余量 ~44%。
2. **一维 Kernel**：源三维 grid 已展开；索引分解为 block 级标量（不可向量化，§1.6.2）。
3. **GM→GM 不支持**：必须 UB 中转（§0.6 R3），故 UB 带宽（读取 + 回写各一次 slab）也是片上开销，但 GM 流量与源相同。
4. **物理核数 40 与 2 的幂 tile 数无法整除对齐**（残留 +1.6%，§5.5）。
5. **`T.Pipelined` 未启用** → MTE2/VEC/MTE3 重叠依赖编译器自动调度。
6. **smoke shape 逻辑核数偏少**：`[1,128,8,64]` → 4 tiles（`R_S=32`），仅用 4 核；smoke 为正确性用例，不影响性能评测。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | slab 元素数取 32768（fp32 128KB+fp16 64KB=192KB 顶格）→ 膨胀后溢出 | 编译失败（`ub overflow ... 1572864 bits`）| `_TILE_ELEMS ≤ 16384`（本文档主选）|
| GM→GM 拷贝 | 直接 `T.copy(dq_slice, dq_out_slice)`（照抄源 L123–126）| 编译期报错（Developer 不支持 GM→GM）| 经 UB 中转（§0.6 R3）|
| 隐式跨 dtype copy 当融合用 | 用 `T.copy(dq_fp32, dq_out_fp16)` | 未必报错，但舍入不可控且无 op 收益（TRAP-C12）| 显式 `T.vcast(round_mode="rint")`（§3.3）|
| `T.serial` + `if t < N` 守卫 | 任务级动态控制流 | 若编译失败 | 备选：`t = min(t, total_tiles-1)` 无分支幂等重算（前序函数已证主选可编译可跑，本项为兜底）|
| 尾轮越界 | `t ≥ total_tiles` 仍执行切片 | 读到界外/写坏 `dq_out` | `if t < total_tiles` 守卫（§6.2）|

### 9.3 特殊场景处理 / 待裁决项

- **分核备选（Stage 4 实验裁决）**：主选 `nblk=40`（残留 +1.6%）；备选 `nblk=32`（512/1024 的公约数，完美均衡，闲置 8 核）。裁决计划：代表 shape `[2,2048,32,128]`/`[1,2048,64,128]` 上 msprof op 实测 `TaskDur`，**若备选 latency < 主选 × 0.98 则翻转**；裁决数据经附录补记回写本节。
- **slab 元素数备选（Stage 4）**：主选 `_TILE_ELEMS=16384`（UB 96KB）；备选 12288（UB 72KB）或 8192（UB 48KB）——若实测大 slab 未带来带宽收益或与 `T.Pipelined` 冲突，可下调。
- **极小 shape**：smoke `[1,128,8,64]` → 4 slabs、`nblk=4`、`iters=1`，无持久化开销。

### 9.4 能力缺口

- 未识别到阻塞性能力缺口。设计用到的 API（`T.copy` slice、`T.vcast`、`T.serial`、`T.alloc_shared`）均有 `docs/Tilelang.language/` 或本仓 `examples/` 佐证。GM→GM 限制已由 UB 中转绕开（非缺口）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_query_attention_bwd/_flashattn_bwd_postprocess_kernel/
├── DESIGN.md                              # 本设计文档
├── _flashattn_bwd_postprocess_kernel.py   # 算子实现 + L0 测试（DEVELOP 阶段）
└── RETROSPECTIVE.md                       # Stage 1 复盘（自进化钩子）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档 |
| `_flashattn_bwd_postprocess_kernel.py` | ⬜ 待实现 | 算子实现（DEVELOP 阶段）|
| `RETROSPECTIVE.md` | ✅ 已追加 | Stage 1 复盘 |
| `verify_equiv.py` | — 不适用 | §1.6.1 结论为「无优化空间」（gate 豁免），无候选式可对照（§1.6.1 说明）|

### 10.3 命名规范

- 项目目录：`grouped_query_attention_bwd`（snake_case）；算子目录：`_flashattn_bwd_postprocess_kernel`；实现文件同名 `.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ 算子实现（`_flashattn_bwd_postprocess_kernel.py`）+ L0 精度比对
4. ⬜ 分核 / slab 备选 A/B 裁决（§9.3，Stage 4）
