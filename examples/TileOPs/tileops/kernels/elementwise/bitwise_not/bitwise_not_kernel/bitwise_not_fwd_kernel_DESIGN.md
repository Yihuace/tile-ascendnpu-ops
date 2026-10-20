# BitwiseNotFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/bitwise.py` 的 `BitwiseNotFwdKernel`，复用 `UnaryKernel` / `FloatUnaryKernel` 的 `register_copy` 策略及 `_builders.py` 的 `make_unary_regcopy` / `make_unary_direct`）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `bitwise_not_fwd_kernel`，`target="npuir"`。
> 迁移工作区：`examples/bitwise_not/bitwise_not_fwd_kernel/`（project=`bitwise_not`，op=`bitwise_not_fwd_kernel`，scenario=migration，mode=harness）。
> 本设计按 skill `tilelang-op-design` 的三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出；本文档为 `first_design` 基线（无 revision 段）。
> **整数 dtype 头条结论**：本算子为 **INTEGER / BOOL 精确按位运算**，无浮点中转、无浮点容差；计算路径 = 直接在整型 dtype 上做**向量按位取反**（`T.vnot`，见 `docs/Tilelang.language/逻辑操作/T.vnot.md`），bool 走字节存储域（`T.vxor`，见 `docs/Tilelang.language/逻辑操作/T.vxor.md`）。`SUPPORTED_DTYPES = {bool, uint8, int8, int16, int32, int64}`（六种整型），**拒绝 float**。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织。先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：`torch.bitwise_not(input)` 的逐元素按位取反（all-ones 补）：

$$
y = \sim x \quad(\text{element-wise bitwise NOT})
$$

源实现按 dtype 分支（`BitwiseNotFwdKernel.op_func`，GPU `bitwise.py` 约 L100-106）：

```python
if x.dtype == "bool":
    return x == T.cast(0, "bool")              # 逻辑 NOT
if x.dtype == "uint8":
    return T.bitwise_xor(x, T.cast(255, "uint8"))   # 全 1 字节
return T.bitwise_xor(x, T.cast(-1, x.dtype))        # 该 dtype 的全 1
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | `y = ~x`：对整型逐位取反（等价 `x XOR 全 1`）；对 bool 为逻辑 NOT。三种 dtype 分支结果都是**该元素的全 1 补**（bool 在字节存储域为 `0<->1`） |
| 规约语义 | **无规约**，纯逐元素，无跨元素依赖、无累加顺序问题 |
| dtype 语义 | 输入/输出同 dtype，`T ∈ {bool, uint8, int8, int16, int32, int64}`（`BITWISE_DTYPES`）；**无浮点中转、无精度提升**——整型按位运算 bit-exact。源码 `int4` 注释说明其用 XOR-with-all-ones 只是为规避 CUDA `T.bitwise_not` 在向量化 `int4` 上的缺陷，**不是**浮点语义 |
| 边界语义 | 无 NaN/±inf 概念（整型）；空维度按展平后处理；非整除 shape 由尾块处理。bool 在 NPU 上不可直接 load/store GM（见 0.5），须以字节存储传递 |
| host 侧语义 | `flat(x)` 展平（视图）、`_restore_output_dtype(...).reshape(x.shape)` 回塑（视图）；`_bool_via_int8` 时 bool 结果以 int8 字节缓冲存储再 `view(torch.bool)`。均为 stride/shape/dtype 视图操作，不改输入张量内容 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.bitwise_not(x)`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input` | `[*S]`（任意，展平为 1-D `N=numel`） | `bool \| uint8 \| int8 \| int16 \| int32 \| int64` | 输入张量；kernel 收到 `flat(x)` 的 1-D 视图 |
| `output` | 与 input 相同 shape `[*S]` | 与 input 相同 dtype（`same_as(input)`） | 输出张量；`forward` 内 `reshape(x.shape)` 回塑（输出为输入别名，非转置布局） |

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host 展平：`flat(x)` → 1-D `(N,)` | `input [*S]` | `x_flat (N,)` | 前置（host 视图操作，不改数据） |
| 2 | host bool 存储适配：`x.view(torch.uint8)`（`_bool_via_int8` 路径） | `x_flat bool` | `x_flat uint8` | bool 字节存储（0.1 边界语义） |
| 3 | 分块装载：`T.copy(x[bx*bs:(bx+1)*bs], x_reg)` | `x_flat` | `x_reg (bs,)` fragment | 数据搬入 |
| 4 | 逐元素计算 `op_func(x_reg[i])`（dtype-branched：bool→`x==0`；uint8→`xor(x,255)`；else→`xor(x,-1)`） | `x_reg` | `y_reg` | `y = ~x` |
| 5 | 分块写回：`T.copy(y_reg, y[bx*bs:(bx+1)*bs])` | `y_reg` | `y_flat` | 数据搬出 |
| 6 | host 回塑：`_restore_output_dtype(result).reshape(x.shape)` | `y_flat` | `output [*S]` | 后置（stride/shape/dtype 视图） |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x] --T.copy--> fragment/x_reg(寄存器/SMEM) --逐元素 op_func--> fragment/y_reg --T.copy--> GM[y]
```

`BitwiseNotFwdKernel` 继承 `UnaryKernel`（`DEFAULT_STRATEGY = "register_copy"`；bool operand 由 `choose_unary_strategy` 选 `direct`）。输出 dtype 与输入相同，`_narrows(dtype, out_dtype)` 为假，`make_unary_regcopy` 的 `contiguous`（窄化逐线程连续 run）分支**不触发**，非 bool 路径走 fragment 全块：`T.copy` 装载到 `x_reg`，`T.Parallel(block_size)` 逐元素计算写入 `y_reg`，`T.copy` 整块写回；bool 路径走 `make_unary_direct`（`x==0` 返回 bool，`store_unary_bool_as_int8` 落 int8 字节）。

**循环与并行结构**：grid 一维 `T.Kernel(T.ceildiv(N, block_size), threads=128)`，block 内 `T.Parallel(block_size)` 元素级并行；`block_size = threads × npt`（`register_copy` 下 `npt = max(min_npt, bytes_per_thread // elem_bytes)` 再取 `register_copy_floor`）。

**host 侧逻辑**：`forward` 中 `flat(x)`（展平，视图）→ `_compiled_fn(...)` → `_restore_output_dtype(result).reshape(x.shape)`（回塑）。Op 层（`BitwiseNotFwdOp` 经 `UnaryOp`）为 `input.contiguous().reshape(-1)` → kernel → reshape。全部为 stride/shape/dtype 视图操作，**不改输入张量内容**（满足 ascend-constraints 的 host 侧约束）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | register_copy（fragment 分块装载/写回） | 减少 GM 重复访问、放大访存粒度 | 分块 `T.copy` 到 fragment，逐元素计算后整块写回 | CUDA SMEM/寄存器 | 硬件强相关 |
| 2 | coalesced 向量化访问 | 尾轴连续读写打满带宽 | fragment 块 `T.copy` 连续搬运 | memory coalescing | 硬件强相关（NPU 有 MTE 等价） |
| 3 | `threads × npt` 两级并行执行模型 | 每线程处理 `npt` 元素放大吞吐 | CUDA thread/warp 组织 + `T.Parallel` | CUDA 线程模型 | 硬件强相关 |
| 4 | dtype 分支的标量逐元素 `op_func` | 一套 body 覆盖三种位宽语义 | `T.Parallel` 内标量 `xor` / `==`，由编译器向量化 | CUDA SIMT | 硬件强相关 |
| 5 | XOR-with-all-ones 代替 `bitwise_not` | 规避 CUDA `T.bitwise_not` 在向量化 `int4` 上的缺陷 | `xor(x, -1)`（有符号）/ `xor(x, 255)`（uint8） | 无（纯算法层 API 选择） | 可移植（NPU 有直接 `T.vnot`，见 §1.6.1） |
| 6 | bool 字节存储（`BOOL_STORAGE_DTYPE = int8`） | bool 无向量化 CUDA 类型，按字节存储 | bool 视图为 uint8，结果 `x==0` 后落 int8，再 `view(bool)` | 无（算法/存储层） | 可移植 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。识别不出来 ≠ 不存在——上表为源码 `op_func` + `UnaryKernel` / `make_unary_regcopy` / `make_unary_direct` 的逐项核对结果。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y = ~x`（整型 all-ones 补 / bool 逻辑 NOT） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| dtype 分支（bool / uint8 / 有符号整型） | 算法 | 无 | — | **保留** | 分支语义保留；NPU 侧 bool 走字节域、uint8/整型走 `vnot`（§0.6 / §1.6.1） |
| bool 字节存储（`x==0` + int8 存储 + `view(bool)`） | 算法/存储 | 无（`BOOL_STORAGE_DTYPE` 是后端事实） | 有（NPU bool i1 同样不可 load/store GM，须字节存储） | **等价替换** | NPU bool 输入/输出以 `uint8` 字节视图传递（`x.view(torch.uint8)` / `result.view(torch.bool)`，均为视图操作不改内容）；判据见 `docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3「bool（i1）无法 load/store GM，仅可作片上中间结果」 |
| XOR-with-all-ones（`xor(x,-1)` / `xor(x,255)`） | 算法 | 无 | 有（NPU `T.vnot` 直接支持全部整型） | **等价替换** | `T.vnot(x)` = 按位取反 = XOR 全 1；NPU `VNotOp` 原生支持 uint8/int8/int16/int32/int64（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1），无需常量操作数（§1.6.1 项 1） |
| `threads × npt` 两级并行 + `T.Parallel` 标量循环 | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel） | **重新设计** | → §0.6 R1（block_size 折叠 + 一维 `T.Kernel` + UB 向量化） |
| fragment 分块装载/写回（SMEM/寄存器） | 优化 | SMEM/寄存器 | 有（UB） | **等价替换** | `T.alloc_ub` + `T.copy`（GM→UB→GM），ascend-constraints 已知限制 / decision-tree §3 |
| coalesced 连续访问 | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | `T.copy` 连续 slice 搬运（双显式 slice 形态，`docs/Tilelang.language/内存操作/T.copy.md`） |
| swizzle / mma / warp shuffle | 优化 | SMEM bank / Tensor Core / warp | 不涉及（源未使用） | **舍弃（不适用）** | 源算子为纯逐元素，未使用这些手段，无迁移对象 |

> 每项处置均写明依据（migration-analysis.md §5.3 映射表 / `examples/` 佐证 / `docs/Tilelang.language/` 条目）。**无「舍弃理由待补」项**——唯一判「舍弃（不适用）」的 swizzle/mma/shuffle 本就不在源算法中。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（`threads × npt` → 一维 Kernel + UB 向量化）**

- **源方案**：`T.Kernel(T.ceildiv(N, block_size), threads=128)` + `T.Parallel(block_size)` 两级并行；`block_size = threads × npt`；fragment 装载/写回（`register_copy` / bool 的 `direct`）。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；GPU 的 `threads × npt` 折叠为单一 `block_size`（ascend-constraints 已知限制「仅支持一维 Kernel」；GPU `threads=` kwarg 在 NPU 无效果，见 `TRAP-threads-kwarg-noop`）；元素级并行交给 Vector 单元（v-prefix 逐元素 API 对整块 UB buffer 计算，天然向量化，无需显式 `T.Parallel` 标量循环）；分块装载/写回用 `T.copy`（GM→UB→GM，32B 对齐连续 slice）。`block_size` 取值见 §5.2。计算链为单 op：整型 `T.vnot(x_ub)`；bool 字节域 `T.vxor(x_ub, 1, y_ub)`。
- **语义保持论证**：逐元素映射不变（每元素独立计算，无跨元素依赖、无规约），`threads/npt` 到 `block_size/核` 的任务映射变化不改变任何输出值——每个 `out[i]` 的结果与源码逐步一致（整型 `~x`；bool `x==0`）。整型按位运算 bit-exact，无舍入。边界语义：尾块（非整除）仅 copy 有效元素，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 fragment 尾块行为等价；bool 字节视图不改内容。

> **本算子只有 R1 一项重设计**：其余 §0.5 条目为保留 / 等价替换。源优化手段（§0.4）的意图承接：#1/#2 的「放大访存粒度、整块搬移」意图由 UB `T.copy` 分块搬移承接；#3 的「元素级并行放大吞吐」意图由 Vector 单元整块 v-prefix 计算承接；#4 的 dtype 分支标量 `op_func` 由 §1.6.2 的 `vnot`/`vxor` 向量化承接；#5 的 XOR-with-all-ones 由 NPU 原生 `T.vnot` 直接承接（并升级为 §1.6.1 项 1 的等价化简）；#6 的 bool 字节存储由 NPU bool 字节视图等价承接。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/bitwise.py`（`BitwiseNotFwdKernel`，`BITWISE_DTYPES` 在 `.../_dtype.py`）、`.../_base.py`（`UnaryKernel` / `FloatUnaryKernel`）、`.../_builders.py`（`make_unary_regcopy` / `make_unary_direct`）、`.../_op_body.py`（`store_unary_bool_as_int8`）。
- NPU 侧工件：wrapper `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not.py`、抽取 GPU 参考 `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/_bitwise_not_fwd_kernels.py`、op `examples/TileOPs/tileops/ops/elementwise/bitwise_not.py`、test `examples/TileOPs/tests/ops/test_bitwise_not.py`、bench `examples/TileOPs/benchmarks/ops/bench_bitwise_not.py`、manifest `examples/TileOPs/tileops/manifest/elementwise.yaml`。
- 源仓参考实现 / 测试基准：`torch.bitwise_not(x)`（`ref_program = torch.bitwise_not(x)`，见 test 文件）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`torch.bitwise_not`），**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

BitwiseNotFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `bitwise_not_fwd_kernel`。

### 1.2 功能描述

逐元素按位取反：`y = ~x`（all-ones 补），用于 bool / 整型张量；输入展平为 1-D 后逐元素计算，整型 bit-exact、无浮点中转，输出与输入同 shape/dtype。

### 1.3 数学公式

$$
y_i = \sim x_i
\quad\Longleftrightarrow\quad
\begin{cases}
y_i = \lnot x_i, & x\ \text{dtype} = \texttt{bool}\ (\text{logical NOT})\\
y_i = x_i \oplus 0\text{xFF}, & x\ \text{dtype} = \texttt{uint8}\\
y_i = x_i \oplus 0\text{xFF}\ldots\text{FF}, & x\ \text{dtype} \in \{\texttt{int8},\texttt{int16},\texttt{int32},\texttt{int64}\}
\end{cases}
$$

### 1.4 算法描述

单步逐元素映射：展平为 1-D `N` 后，对每个元素独立求按位取反。整型直接 `vnot`（`~x`）；bool 在字节存储域做逻辑 NOT。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**；整型 bit-exact，**无 fp32 中转**（这是本算子区别于同族浮点 elementwise 迁移的关键——见 §1.6.1 整数 dtype 契约）。本算法与 §1.6.0 调研选定算法一致。与源算法结构差异：仅并行组织（GPU `threads×npt` → NPU 一维 Kernel + `block_size` 折叠，§0.6 R1）与「XOR-全 1 → 原生 `vnot`」的算符形态（§1.6.1 项 1）不同；逐元素语义与源码逐步一致。

### 1.5 数据流图

```
输入 input [*S] --host 展平--> x_flat (N,) --[bool? view(uint8)]--> x_bytes (N,)
  --T.copy--> UB[x_ub] --(int: vnot / bool: vxor(·,1))--> UB[y_ub]
  --T.copy--> y_flat (N,) --[bool? view(bool)]--> 输出 [*S]
```

### 1.6 算法调研与优化分析

> 设计第一优先级：先调研四问（§1.6.0），再数学等价优化公式（§1.6.1），再向量化替代分析（§1.6.2），最后向量化轴/布局决策（§1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。本节分析对象为 §1.6.0 选定、经 §0.6 R1 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：bitwise-NOT 的算法族（所有能算出相同结果的算法结构）。源算法/输入公式只是基线候选之一。本算子为**单步逐元素按位运算**，按 `algorithm-research.md` §2 判为**轻量调研**（四问各一行结论 + 依据），但四问不可跳过。对照 `algorithm-candidates.md` 的 elementwise 命中行（R1：算子降代换 / 公共子表达式消除；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返——本算子无中间 GM 往返）。

**R1 等价化简公式候选**（基线候选在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源公式：dtype 分支 XOR/`==`） | `xor(x,-1)` / `xor(x,255)` / `x==0` | — | — | ✅（基线） |
| 1 | 原生一元取反 `vnot` | `T.vnot(x)`（整型，含 uint8） | 数学恒等（XOR 全 1 ≡ 按位取反），bit-exact | 二元 op（带常量操作数）→ 一元 op（少常量物化/广播） | ✅ |
| 2 | bool 字节域 `x ^ 1` | `T.vxor(x_ub, 1, y_ub)` | 数学恒等（字节 ∈ {0,1}：`x==0` ⇔ `x^1`），bit-exact | 比较+掩码物化（2 op）→ 单 `vxor`（1 op） | ✅ |
| 3 | `1 - x`（bool 字节域） | `T.vsub(1, x_ub)` | 数学恒等（0→1，1→0） | 与候选 2 同 op 数（1 op） | ❌（与候选 2 等价且无更优点数，`vxor` 与语义更贴合） |
| 4 | 查表 / 位翻转其它形态 | — | — | 无收益 | ❌（无结构差异） |

**R2 在线算法**：**无「在线/离线」之分**（天然单遍）——bitwise-NOT 是逐元素映射，每个输出元素只依赖对应输入元素，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（`algorithm-research.md` §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导，不得只比 FLOPs）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 `xor(x, all-ones)` / `x==0` | N（1 op/elem，manifest 口径 `prod(S)`） | 2N·elem_bytes（读 N + 写 N，无中间 GM 往返） | 1 | O(block_size)（UB 驻留，无 GM 中间缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1 `vnot(x)` | N | 2N·elem_bytes | 1 | O(block_size) | 完全并行 |
| 候选 2 bool `vxor(x,1)` | N | 2N·elem_bytes | 1 | O(block_size) | 完全并行 |
| 候选 3 bool `vsub(1,x)` | N | 2N·elem_bytes | 1 | O(block_size) | 完全并行 |

**结论**：候选间 FLOPs / 访存 / 遍数 / 中间缓冲四口径全同（带宽受限，访存主导），基线即为最优复杂度量级；候选 1/2 在 R4 因**算子形态更省**（一元 vs 二元带常量、单 op vs 比较+掩码）胜出，无并列争议。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰须附本地 docs / examples 目录佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线 | 逐元素按位 → Vector，**不落 Cube**，与元素级算力正确匹配 | 工作集 = O(block_size) UB 驻留，≤64KB < 192KB（§4.5） | 尾轴连续，block_size 为 32B 整数倍且整除对应向量宽（int8/uint8/bool ×16、int16 ×8、int32 ×4、int64 ×2）✓ | 循环边界 `ceildiv(N, block_size)` / `ceildiv(num_logical, num_kernels)` 均为编译期常量（N/block_size 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ✅ 采纳 |
| 候选 1 `vnot` | 同上；NPU `VNotOp` 原生一元指令 | 同上 | 同上 | 同上 | 同上 | ✅ 采纳（取代基线整型路径） |
| 候选 2 bool `vxor` | 同上；`VXorOp` 原生支持 uint8 | 同上 | 同上 | 同上 | 同上 | ✅ 采纳（取代基线 bool 路径） |
| 候选 3 bool `vsub` | 同上；`vsub` 对 uint8 减可能下溢风险（非 bool 值域） | 同上 | 同上 | 同上 | 同上 | ❌ 淘汰（`vsub` 语义不如 `vxor` 明确；无收益） |

**定量 roofline（D-2）**：基线容量/带宽/发射均匹配 Vector 单元（工作集 < UB、尾轴 32B 对齐、约束可静态化），无候选因违反容量/带宽量级被淘汰。引用 `constants.md` 实测常数条目（`CONST-capacity-910B2C` UB 192KB、`CONST-vector-launch-overhead` 发射 ~0.5µs/op、`CONST-mte2-degradation` MTE2 带宽档）计算估算下界（见本节末「设计期估算」）。

**调研结论**：选定算法族 = **逐元素单遍直接映射 + 算符降代换**——整型用原生一元 `T.vnot`（取代 `xor(x, all-ones)`），bool 用字节域单 `T.vxor(x,1)`（取代 `x==0` + 掩码物化）。关键依据：R1 候选 1/2 数学恒等且 bit-exact、候选 3/4 无更优点数；R2 天然单遍；R3 四口径全同（带宽受限，访存主导）；R4 纯 Vector 亲和（UB 驻留、32B 对齐、静态边界）。与基线结构差异：**逐元素计算由「二元 XOR-with-all-ones / 比较」降为「一元 vnot / 单 vxor」**（§1.6.1 完成等价论证）。「无更优替代（就复杂度量级而言）」调研范围：`algorithm-candidates.md` elementwise 命中行、恒等变形自查（R1 候选 3/4）、pattern-library elementwise 同族先例（`CASE-elementwise-mish-opt`、`examples/TileOPs/tileops/kernels/elementwise/hardswish/hardswish_kernel/hardswish_fwd_kernel_DESIGN.md`）、源码 §0.3/§0.4。源优化手段（§0.4）中 #5「XOR-with-all-ones」被调研升级为原生 `vnot`（意图承接见 §0.6/§11.2），其余无被替换。

**设计期估算**（D-2，轻量调研豁免容量项，流量项必算）：

```
估算下界(manifest 代表 workload，流量项主导):
  16M int32 [4096,4096]: max(流量 ≈ 111µs, 发射 ≈ 0.5µs) ≈ 111µs
  16M int64 [4096,4096]: max(流量 ≈ 222µs, 发射 ≈ 0.5µs) ≈ 222µs
  256M int32 [16384,16384]: max(流量 ≈ 2917µs, 发射 ≈ 0.5µs) ≈ 2.92ms
  流量项: 2 × N × elem_bytes ÷ 有效带宽；有效带宽 = CONST-mte2-degradation 档位 × 40 向量核
    16M 档 30.2 GB/s/core × 40 ≈ 1.208 TB/s；256M 档 18.4 GB/s/core × 40 ≈ 0.736 TB/s
  发射项: 计算链 ≈ 1 个向量 op（vnot 或 vxor）× 0.5µs/op ≈ 0.5µs（CONST-vector-launch-overhead）
  容量项: 轻量调研豁免（单步逐元素，无 fixed-loop 跨引擎往返；CONST-store-fixpipe-gm-only 不适用——无 Cube）
  未实测假设: CONST-mte2-degradation 曲线绑定 fp16 / 48 vector core / 4-set 轮换；本设计 40 向量核（实查 20×2）
    与整型 dtype，档位按「总流量档」近似引用（int32/int64 元素字节已计入字节口径），标注为未实测外推；L2 驻留
    workload（smoke-1m 复用同输入）会高估时延（该条目 L2 口径注：遇同张量复用带宽偏高）
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（经 §0.6 R1 重设计落地）。采纳项经 `verify_equiv.py` 机器验证（结果表见下）。**整数 dtype 契约**：整型按位运算 bit-exact，**无 fp32 中转、无容差**；候选式与基线式在整型域逐位相等。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | 算符降代换：XOR-全 1 → 原生一元取反（整型，含 uint8） | `y = bitwise_xor(x, 0xFF…FF)`（有符号 `-1`；uint8 `255`） | `y = T.vnot(x)` | 数学恒等：对任一固定位宽整型，`x ⊕ 全1 = ~x`（按位取反的定义），bit-exact、无舍入。`verify_equiv.py` 项 1：uint8/int8/int16/int32/int64 五 dtype 随机 + 角点（min/max/0/1/全1）共 5×4101 元素 mismatches=0，EQUIV_PASS | 二元 op（一个张量 + 一个常量操作数，需常量物化/广播）→ 一元 op（单操作数）；向量管线少一次操作数读取/广播，op 语义更直接 |
| 2 | bool 逻辑 NOT 的字节域化简 | `y = (x_byte == 0)`（比较 → 掩码 → 落字节，源 `store_unary_bool_as_int8` 路径） | `y = T.vxor(x_byte, 1)` | 数学恒等：字节存储 bool 取值 ∈ {0,1}，`(x==0)` ⇔ `(x ⊕ 1)`（0→1，1→0），bit-exact。`verify_equiv.py` 项 2：bool 4098 元素（随机 + {False,True} 角点）mismatches=0，EQUIV_PASS | 比较 + 掩码物化（≥2 op：`vcmp` + cast/select）→ 单 `vxor`（1 op）；消去 bool 掩码中间缓冲 |

**机器验证结果表**（`verify_equiv.py`，torch CPU 精确整型/布尔对照 + 随机/角点，已执行）：

| 采纳项 | 违反数（mismatches） | 结论 |
|--------|---------------------|------|
| 1. `xor(x, all-ones)` → `vnot(x)`（uint8/int8/int16/int32/int64） | 0 | EQUIV_PASS |
| 2. `(x == 0)` → `(x ^ 1)`（bool 字节存储） | 0 | EQUIV_PASS |

逐 dtype 结果：uint8 = 0 / int8 = 0 / int16 = 0 / int32 = 0 / int64 = 0 / bool = 0（全部 EQUIV_PASS）。

**优化结论**：采纳 2 项。优化后公式为（供 §3.1 公式拆解使用）：

$$
y_i =
\begin{cases}
\texttt{bool}: & y_i = x^{byte}_i \oplus 1 \quad(\text{byte storage domain},\ x^{byte}\in\{0,1\})\\
\texttt{uint8/int8/int16/int32/int64}: & y_i = \operatorname{vnot}(x_i) = \sim x_i
\end{cases}
$$

无其他可采纳优化项：单次逐元素映射，无公共子表达式（`x` 为输入，非可消冗余）、无超越函数、无 cast 链（整型 bit-exact，**不动 fp16/bf16 中转**）、无中间 GM 往返。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | 整型按位取反（uint8/int8/int16/int32/int64） | `T.Parallel` 逐元素标量 `xor` | `T.vnot(x_ub, y_ub)`（`docs/Tilelang.language/逻辑操作/T.vnot.md`，`VNotOp`，DataType 表 uint8/int8/int16/int32/int64 均 √） | ✅ | — |
| 2 | bool 逻辑 NOT（字节存储域） | 逐元素 `x == 0`（比较 + 掩码物化） | `T.vxor(x_ub, 1, y_ub)`（`docs/Tilelang.language/逻辑操作/T.vxor.md`，src1 可为 Python 标量 int 并广播；uint8 √） | ✅ | — |
| 3 | 数据搬入 GM→UB | 逐元素标量 load | `T.copy(x[offset:offset+tail], x_ub[0:tail])`（双显式 slice，`docs/Tilelang.language/内存操作/T.copy.md`） | ✅ | — |
| 4 | 数据搬出 UB→GM | 逐元素标量 store | `T.copy(y_ub[0:tail], y[offset:offset+tail])`（双显式 slice） | ✅ | — |
| 5 | block 起始索引 `offset = block_id * block_size` | 标量乘加 | 无向量 API | ❌ | block 级任务映射索引，每 block 少量标量，非逐元素热点（可接受理由类别①：block 级索引 / 任务映射计算） |
| 6 | 尾块大小 `tail_size = T.min(block_size, remaining)` | 标量 min | 无（`T.min` 标量形） | ❌ | 依赖动态 shape 的边界处理，每 block 一次，非逐元素热点（可接受理由类别④：动态边界；同族 mish / hardsigmoid / hardswish 同款尾块处理） |

**向量化结论**：逐元素计算（整型 `vnot` / bool `vxor`）与数据搬入搬出（`T.copy`）已**全部向量化**（替代 API 均有 `docs/Tilelang.language/逻辑操作/`、`docs/Tilelang.language/内存操作/` 佐证）。保留 2 处标量：block 起始索引（block 级任务映射）与尾块大小（动态边界），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策

**单轴豁免说明**：本算子输入展平为 1-D `(N,)`，仅一个非 batch 维（展平轴），无 ≥2 个非 batch 维度可作向量化轴选择——逐元素按位运算在**唯一的展平轴**上向量化（昇腾 Vector 128bit 寄存器：int8/uint8/bool(字节) 每 op ×16、int16 ×8、int32 ×4、int64 ×2），不存在广播轴、窗口轴、归约轴或重排布局候选。按 SKILL.md Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」。

- **向量化轴** = 展平轴（stride=1 连续，无累加链、无跨步系数）。
- **整除性**：block_size 为 32B 整数倍且整除对应向量宽（§5.2：int32/int16/1-byte 用 8192，int64 用 4096；8192 = 2048×4 = 1024×8 = 512×16，4096 = 2048×2）✓；尾块 `tail_size` 动态，仅 copy 有效元素，垃圾 lane 不写回（与 mish / hardswish 同款）。
- **repack 代价**：无（输入即 1-D 连续，无需转置/重排；bool 的 `view(uint8)` 为元数据视图，非数据重排）。
- **UB 容量影响**：驻留缓冲 = O(block_size)，预算见 §4.5（峰值 ≤64KB < 192KB）。
- **迁移对照**：GPU 源算子的并行轴选择（`threads` 线程 × `npt` 每线程元素）是 CUDA 线程模型，与 NPU Vector 展平轴不对应——本设计独立按展平轴向量化（§0.6 R1）。

**布局决策结论**：选定「I/O 原生 1-D 连续布局 + 展平轴向量化」，无 repack、无重排候选（单轴豁免）。该结论同步落入 §3.3 伪代码与 §6 循环结构（向量维 = 展平轴，buffer 均为 1-D）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（迁移默认）。

### 2.2 选型理由

纯 Vector 逐元素算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。算子体仅 1 个向量 op（`vnot` / `vxor`），无需 Expert 的手动同步与多级 buffer 控制。与同族 elementwise（mish / hardsigmoid / hardswish）先例一致；NPU wrapper `BitwiseNotFwdKernel` 声明 `ascend_mode = "Developer"`（见 `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not.py`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix API 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（`T.vnot` / `T.vxor`），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 set_flag/wait_flag） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_bytes = x`（整型）或 `x.view(uint8)`（bool） | bool 以字节存储传递（`view` 为元数据视图，不改内容） |
| 2 | `y = vnot(x_bytes)`（uint8/int8/int16/int32/int64） | 整型按位取反（all-ones 补），bit-exact |
| 3 | `y = vxor(x_bytes, 1)`（bool） | bool 字节域逻辑 NOT（0↔1） |
| 4 | `y_out = y`（整型）或 `y.view(bool)`（bool） | bool 结果视图回 bool |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 数据搬入 GM→UB | `T.copy(x[offset:offset+tail], x_ub[0:tail])` | 双显式 slice（`TRAP-T-copy-region-semantics` 推荐形态） | Developer |
| 1 | 整型按位取反 `~x` | `T.vnot(x_ub, y_ub)` | `T.vnot(src, dst)`，同 shape/dtype | Developer |
| 2 | bool 字节域逻辑 NOT | `T.vxor(x_ub, 1, y_ub)` | `T.vxor(src0, src1, dst)`，src1 为 Python 标量 int（广播） | Developer |
| 3 | 数据搬出 UB→GM | `T.copy(y_ub[0:tail], y[offset:offset+tail])` | 双显式 slice | Developer |

> 无 `vcast`（整型 bit-exact，**无浮点中转**，这是本算子区别于同族浮点 elementwise 迁移的关键）；无 `vbrc` / `vcmp` / `vselect`（无 NaN、无掩码物化）。

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型以**第二个位置参数**传入（不用 `dtype=` 关键字），避免产生 `name 'in_dtype' is not defined` 的 false alarm。bool 在 kernel 内以 `uint8` 字节存储处理（bool i1 不可 load/store GM，见 `docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3）。

```python
@tilelang.jit(out_idx=[1], target="npuir")
def bitwise_not_fwd_kernel(N, dtype, output_dtype=None, block_size=None, num_cores=None):
    out_dtype = output_dtype or dtype
    is_bool = (dtype == "bool")
    # bool -> uint8 byte storage (bool i1 cannot load/store GM); others native.
    storage_dtype = "uint8" if is_bool else dtype
    if block_size is None:
        block_size = 4096 if storage_dtype == "int64" else 8192   # see section 5.2

    num_logical = T.ceildiv(N, block_size)                        # compile-time
    # Physical vector cores (section 5.5: AI cores x 2 for a pure Vector op).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_logical, num_kernels)         # static bound

    one_u8 = T.cast(1, "uint8")

    @T.prim_func
    def main(
        x: T.Tensor((N,), storage_dtype),
        y: T.Tensor((N,), storage_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # 1. UB buffers (v-prefix API requires UB residency). Two buffers only.
            x_ub = T.alloc_ub((block_size,), storage_dtype)
            y_ub = T.alloc_ub((block_size,), storage_dtype)

            # 2. Persistent in-core serial tile loop (static bound, grid-stride).
            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # 3. Load GM -> UB (valid elements only).
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 4. Compute (single vector op; Python branch at trace time).
                    if is_bool:
                        T.vxor(x_ub, one_u8, y_ub)   # bool byte-domain logical NOT
                    else:
                        T.vnot(x_ub, y_ub)           # integer bitwise NOT (~x)

                    # 5. Store UB -> GM (valid elements only).
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main
```

> 注：`T.Tensor((N,), storage_dtype)` 用位置参数传 dtype（避免 false alarm，见本节开头约定）。bool 的 host 侧视图：Op 层 `x.view(torch.uint8)` 进入、`result.view(torch.bool)` 返回（均为元数据视图，内容不变；见 §11.4）。整型无 `vcast`（bit-exact）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vnot` | `docs/Tilelang.language/逻辑操作/T.vnot.md`（`T.vnot(src, dst)`，`VNotOp`；DataType 表 uint8/int8/int16/int32/int64 均 √；bool ×） | ✅ 已验证（docs 读） |
| `T.vxor` | `docs/Tilelang.language/逻辑操作/T.vxor.md`（`T.vxor(src0, src1, dst)`，src1 可为 Python 标量 int 并广播；uint8 √；1-5D） | ✅ 已验证（docs 读） |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md` + 同族先例（`TRAP-T-copy-region-semantics`：src/dst 双显式 slice 为推荐通用形态） | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/_bitwise_not_fwd_kernels.py` / 同族迁移 `examples/TileOPs/tileops/kernels/elementwise/hardswish/hardswish_kernel/hardswish_fwd_kernel_DESIGN.md` | ✅ 已验证 |
| `T.alloc_ub` | `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not.py` / 同族迁移先例 | ✅ 已验证 |
| `T.ceildiv` / `T.min`（标量） | `docs/Tilelang.language/数学操作/T.ceildiv.md` + 同族尾块先例 | ✅ 已验证 |

> 无 `vcast` / `vbrc` / `vcmp` / `vselect`（整数 bit-exact，无浮点中转、无 NaN 专用链、无掩码物化——与同族浮点激活设计的关键差异）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 `T.Kernel(ceildiv(N, block_size))`） | 一维 `T.Kernel(num_kernels, is_npu=True)`，无三维 |
| GPU 专用 API | Yes（`T.get_thread_binding` / `T.vectorized` / `T.unroll` CUDA 语义；`T.bitwise_xor` / `T.bitwise_not` GPU 形态；bool i1 GM 存取） | 全部改写为 NPU v-prefix API + UB 路径（§0.5/§0.6 R1）；bool 走字节存储（§3.3） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(N, block_size), threads=128)`（block+thread 两级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(block_size)` 显式元素循环 + 标量 `op_func` | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| 逐元素算符 | `bitwise_xor(x, all-ones)` / `x == 0`（标量，编译器向量化） | `T.vnot(x)` / `T.vxor(x, 1)`（原生向量 op） | §1.6.1 项 1/2 |
| bool 处理 | `BOOL_STORAGE_DTYPE = int8`，`store_unary_bool_as_int8` | `uint8` 字节存储 + `view(bool)` | §0.5 等价替换 / §3.3 |
| fp32 中转 | 无（整型算子本就走整型） | 无（bit-exact） | 语义层保留 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/hardswish/hardswish_kernel/hardswish_fwd_kernel_DESIGN.md` | 高度相似（同族迁移设计；本算子为其「整数 / 无浮点中转」变体） | 一维 Kernel + block_size 折叠、分核三要素、UB 预算口径、persistent 尾块 |
| `examples/TileOPs/tileops/kernels/elementwise/hardswish/hardswish_kernel/hardswish_fwd_kernel.py` | 高度相似（同族迁移实现骨架） | `T.alloc_ub`、`T.copy` 尾块双显式 slice、`T.Kernel(is_npu=True)`、`NPUUtils.get().get_aicore_num()` |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 相似（同族 Developer elementwise 迁移） | Developer 模式、persistent 核内串行、UB 分块 |
| `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/_bitwise_not_fwd_kernels.py` | 相似（迁移对象抽取件） | dtype 分支 `op_func` 的语义与调用约定（CPU reference，需 NPU 重写） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `input` | `[*S]`（展平 1-D `N`） | `bool \| uint8 \| int8 \| int16 \| int32 \| int64` | 输入张量，host 展平视图；bool 以 uint8 字节视图入 kernel |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | 与 input 相同 | 与 input 相同 dtype | 输出张量，host 回塑；bool 以 uint8 字节视图出 kernel 后 `view(bool)` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | 存储 dtype（bool→uint8） | UB | 输入 tile（GM→UB） |
| `y_ub` | `(block_size,)` | 存储 dtype（bool→uint8） | UB | 输出 tile（UB→GM） |

### 4.4 内存搬运路径

```
GM[x] --T.copy--> UB[x_ub] --vnot / vxor--> UB[y_ub] --T.copy--> GM[y]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `x_ub` | (8192,) 或 (4096,) | 1B/2B/4B/8B 存储 dtype | 见下 |
| `y_ub` | (8192,) 或 (4096,) | 同 dtype | 见下 |
| **uint8/int8/bool（block_size=8192, 1B）** | | | **16384（16KB）** |
| **int16（block_size=8192, 2B）** | | | **32768（32KB）** |
| **int32（block_size=8192, 4B）** | | | **65536（64KB）** |
| **int64（block_size=4096, 8B）** | | | **65536（64KB）** |

对照目标平台 UB 容量 **192KB**（`constants.md` `CONST-capacity-910B2C`：UB=192KB/AIV）：

- 显式 alloc 结构按该条目 ×1.10–1.12 上偏系数：峰值 64KB ×1.12 ≈ 72KB < 192KB ✓，余量 ~63%。
- 即便按 auto-multi-buffer 保守膨胀（`TRAP-UB-multibuffer-inflation` ~1.7x）：64KB ×1.7 ≈ 109KB < 192KB ✓。
- int64 选 block_size=4096 而非 8192，正是为使 8B dtype 的峰值与 int32 一致（64KB），避免 8192×8B×2 = 128KB 在膨胀下逼近上限。全部 dtype 路径均不溢出，**无需进一步调小 block_size**。

### 4.6 动态轴定义

无动态轴。`N` 为 JIT 编译期参数（`bitwise_not_fwd_kernel(N, dtype)` 每 shape 一特化），`shape`（`T.int32`）为运行时传入的实际元素数（用于尾块 `remaining = shape - offset`，与同族迁移同款）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素按位运算（`vnot` / `vxor`），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_size = 4096 if storage_dtype == "int64" else 8192   # per-dtype
num_logical = T.ceildiv(N, block_size)                    # logical tile count
```

- **选择理由**：8192 为同族 elementwise（mish / hardsigmoid / hardswish）实测最优 block_size；32B 对齐（8192 × 1/2/4B 均为 32B 整数倍；4096 × 8B = 32KB 亦为）。int64 取 4096 以约束 UB 峰值（§4.5）。宽块摊薄向量发射开销（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。
- **dtype → block_size**：bool/uint8/int8/int16/int32 → 8192；int64 → 4096。

### 5.3 约束分析

- **对齐约束**：block_size=8192 时 1B 尾轴 8KB、2B 16KB、4B 32KB 均 32B 对齐且整除向量宽（1B ×16、2B ×8、4B ×4）✓；int64 block_size=4096 尾轴 32KB、整除 ×2 ✓。
- **UB 容量**：峰值 64KB < 192KB（§4.5）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾块**：`tail_size = T.min(block_size, remaining)`，仅 `T.copy` 有效元素进/出；v-prefix 对全 buffer 计算，尾部垃圾元素不写回（与同族迁移同款）。
- **极小 shape**（如 N < block_size）：`num_logical=1`，`num_kernels=1`，单核单 tile 处理，无 oversubscription。
- **bool 字节视图**：bool 输入/输出在 Op 层与 kernel 之间以 uint8 视图传递，kernel 内不出现 i1 GM 访问。

### 5.5 分核策略（物理核数适配）

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（Cube/混合直接用返回值、纯 Vector 翻倍），禁止文档假设/经验值。

- **物理核数（实查）**：`NPUUtils.get().get_aicore_num()` → **20**（2026-10-13 于本任务环境实查）。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。查询代码与实际返回值记录于此：

```python
from tilelang.utils.npu_utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 翻倍：40
```

（注：`constants.md` `CONST-aicore-910B2C` 记录 910B2C 为 24 核，本环境设备实查返回 20——设备型号差异，一律以实查为准；同现象在本工作区 hardsigmoid / hardswish 任务复现。）

- **逻辑核数**：`num_logical = ceil(N / block_size)`，随 workload（NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `BitwiseNotFwdOp` 条目）变化：

| workload | shape | dtype | N = prod(S) | block_size | 逻辑核数 ceil(N/bs) | 每核串行 tile 数 ceil(num_logical/40) |
|----------|-------|-------|-------------|-----------|---------------------|--------------------------------------|
| smoke-1m | [1048576] | int32 | 1,048,576 | 8192 | 128 | 4 |
| 16m | [4096,4096] | int32 | 16,777,216 | 8192 | 2048 | 52 |
| 16m | [4096,4096] | int64 | 16,777,216 | 4096 | 4096 | 103 |
| 256m | [16384,16384] | int32 | 268,435,456 | 8192 | 32768 | 820 |

- **规模判定与分核方案**：workload 跨越「中等规模」到「极大规模」两档。设计采用**统一 persistent 结构**一次覆盖全部档位（成熟 elementwise 形态）：

  - **小规模（逻辑核数 ≤ 40，即 N ≤ 327680）**：`num_kernels = num_logical`，每核 ≤1 tile，`num_local_tasks=1`，无串行循环、无负载不均——结论「**无需适配**」（逻辑核数 ≤ 物理向量核数）。
  - **中等/极大规模（逻辑核数 > 40，如 smoke-1m 128、16m int32 2048、16m int64 4096、256m 32768）**：固定启动内核数 = 40（= 物理向量核数），核内 `T.serial` **串行**处理 `num_local_tasks = ceil(num_logical/40)` 个 tile（smoke-1m：4；16m int32：52；16m int64：103；256m：820），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均；循环边界 `num_local_tasks` 为编译期常量（N / block_size / 40 均编译期已知），满足「**核内串行**边界静态」。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（`vnot` / `vxor`） | 块内逐元素由 Vector 单元并行（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # UB buffer 分配 ...
    for i in T.serial(num_local_tasks):        # 核内串行 tile 循环（静态边界）
        block_id = i * num_kernels + cid
        if block_id < num_logical:             # 尾 tile guard
            # 装载 -> 单向量 op（vnot / vxor）-> 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对计算链（单向量 op）采用 guard-free 2-stage `T.Pipelined(num_local_tasks, num_stages=2)` 双缓冲流水，将向量 op 隐藏于 MTE2 窗口（该算子计算链仅 1 op，`PL-1.6-copy-floor` 表明 ≥16M 档计算链被完全隐藏，预计收益有限）。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 shape：`tail_size = T.min(block_size, remaining)`；`T.copy` 仅装载/写回 `[0:tail_size]` 有效元素；v-prefix 对全 block_size buffer 计算，垃圾尾部不写回。`block_id < num_logical` guard 覆盖最后一波不完全 tile。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer）。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器自动管理 GM↔UB 搬运与向量计算间的依赖（`T.copy` 与后续 v-prefix op 之间的数据依赖由自动同步保证），无跨核协作（逐元素无跨元素依赖，各核独立）。

### 7.3 pass_configs 配置

Developer 模式默认配置，无需额外 `pass_configs`。环境变量 `TILELANG_ASCEND_MODE = "Developer"`（可选，默认即 Developer；NPU wrapper 声明 `ascend_mode = "Developer"`）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（`torch.bitwise_not`，精确整型/布尔），移植源仓参考实现，**不复刻 §0.6 的 NPU 算法**。

```python
import torch

def golden_bitwise_not(x):
    """Bitwise NOT reference: y = ~x (exact, integer/bool).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_bitwise_not.py: ref_program = torch.bitwise_not(x), atol = rtol = 0).
    Independent of the NPU algorithm (no vnot / vxor / byte-storage / persistent
    structure).  Runs on CPU and does not require an NPU device.
    """
    return torch.bitwise_not(x.detach().cpu())
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| bool / uint8 / int8 / int16 / int32 / int64 | 0.0 | 0.0 |

（与源仓测试 `_get_tolerances` 一致：`torch.bitwise_not` 为精确运算，比较用精确相等，max_abs_err 必须为 0。候选式与基线式 bit-exact，§1.6.1 机器验证 mismatches=0。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`N=1048576`，六 dtype（bool / uint8 / int8 / int16 / int32 / int64）各一例，随机整型/布尔输入，vs golden（`torch.bitwise_not`）以精确相等（atol=rtol=0，`torch.equal`）比对；另含拒绝 float dtype 的负向用例。完整分层套件（L1 非整除 / 多 shape、L2 边界 size、Boundary 特殊值）交由 `tilelang-op-develop` 规划，本设计不枚举。

**特别说明（边界用例）**：整型无 NaN/±inf 语义；Boundary 用例应覆盖整型极值（min / max / 0 / 全 1）与 bool 的 {0,1}，全部要求精确相等。bool 用例须经字节存储路径验证（`view(uint8)` / `view(bool)` 往返）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| bool 不可直接 load/store GM | bool i1 无法 load/store GM（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3），必须经 uint8 字节存储；Op 层 `view(uint8)` / `view(bool)` 为元数据视图，内容不变（§11.4） |
| `T.vnot` bool 不支持 | `T.vnot` 的 bool/int1 列为 ×，故 bool 走 `T.vxor`（uint8 支持），不可对 bool 直接 `vnot` |
| uint8 与 bool 路径不同 | uint8 按字节位取反（`vnot`），bool 字节域逻辑 NOT（`vxor·,1`）；二者语义不同，不可混用 |
| UB 预算 | 峰值 64KB（int32/int64）< 192KB；显式 alloc ×1.12 ≈ 72KB、保守 ×1.7 ≈ 109KB，均安全 |
| 分核 | 物理核数实查 20（纯 Vector ×2 = 40）；与 `constants.md` `CONST-aicore-910B2C`（24）不一致——以实查为准 |
| 无浮点中转 | 整型 bit-exact，**禁止**引入 fp16/bf16/fp32 中转（会破坏按位语义） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bool 直接作为 GM buffer dtype | `T.Tensor((N,), "bool")` / bool UB 做算术 | 编译失败 / 值错误（i1 不可 GM 存取、vnot/vxor bool ×） | bool 走 uint8 字节存储 + `view(bool)`（§3.3/§11.4） |
| 用 `vnot` 处理 bool 字节 | 对字节存储 bool 调 `T.vnot` | 结果 255/254 而非 0/1（逻辑错误） | bool 路径用 `T.vxor(x, 1)`（§1.6.1 项 2） |
| 对 uint8 误用逻辑 NOT | 把 uint8 当 bool 用 `vxor(x,1)` | uint8 语义错误（应 `vnot`） | uint8 用 `T.vnot`（§3.1 步骤 2） |
| 尾块越界 | `T.copy` 用 base+size 形态写尾块 | MTE 越界写坏相邻 tensor | 双显式 slice `T.copy(src[0:tail], dst[0:tail])`（`TRAP-T-copy-region-semantics`） |
| `dtype=` 关键字 false alarm | `T.Tensor((N,), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（§3.3 约定） |
| int64 UB 溢出 | int64 用 block_size=8192 | 128KB×膨胀逼近上限 | int64 用 block_size=4096（§4.5/§5.2） |

### 9.3 特殊场景处理

- **非整除 shape**（如 N 奇数）：尾块 `tail_size` + `block_id < num_logical` guard（§6.4）。
- **极小 shape**（N < block_size）：`num_logical=1`，单核单 tile。
- **bool 输入/输出**：字节视图往返（§11.4）。
- **整型极值**：min/max/全 1 由 `vnot` 精确处理（bit-exact）。
- **float dtype**：Op / Kernel 层显式拒绝（`ValueError`，见 test 文件的负向用例）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/bitwise_not/bitwise_not_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，结果内嵌）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档 |
| `verify_equiv.py` | 已完成 | §1.6.1 等价性验证（2 项 EQUIV_PASS） |
| `bitwise_not_fwd_kernel.py` | 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_bitwise_not.py` | 待实现（Stage 3） | 精度测试 |

### 10.3 命名规范

- 项目目录名：`bitwise_not`；算子目录名：`bitwise_not_fwd_kernel`；接口函数：`bitwise_not_fwd_kernel`（manifest 键 `BitwiseNotFwdOp`）。

### 10.4 实现顺序

1. 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. Golden 函数（§8.1）
3. 算子实现（`bitwise_not_fwd_kernel`）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）

---

## 11. 迁移决策与设计一致性核对

> 本节逐项核对 §0 迁移决策与 §1–§10 设计的一致性，并说明源优化手段的意图承接（防止「§0 说重设计、§3 又照抄源方案」的脱节）。

### 11.1 §0 → §1–§10 一致性

| §0 决策 | 后续落地 | 一致？ |
|---------|---------|--------|
| 语义层 `y = ~x`（整型 all-ones 补 / bool 逻辑 NOT） | §1.3/§1.4 公式与算法描述、§3.1 拆解、§8.1 golden | ✅ |
| §0.5 R1「并行结构重设计」 | §3.3 伪代码（一维 `T.Kernel(num_kernels, is_npu=True)` + block_size 折叠 + v-prefix）、§5.2/§5.5、§6.1 | ✅（未照抄 GPU `threads×npt`） |
| §0.5「fragment → UB」等价替换 | §4.3/§4.4（`T.alloc_ub` + `T.copy` GM↔UB） | ✅ |
| §0.5「coalesced → MTE 连续搬运」等价替换 | §3.2 步骤 0/3（双显式 slice `T.copy`） | ✅ |
| §0.5「XOR-with-all-ones → 原生 vnot」等价替换 | §1.6.1 项 1（机器验证）+ §3.2 步骤 1 | ✅ |
| §0.5「bool int8 存储 → uint8 字节存储」等价替换 | §3.3 bool 分支 + §11.4 视图契约 | ✅ |
| §0.5「swizzle/mma/shuffle 舍弃（不适用）」 | §3.5.1（源未使用，无迁移对象） | ✅（不涉及） |

### 11.2 源优化手段（§0.4）意图承接

| §0.4 手段 | 承接结构 | 是否静默丢弃 |
|-----------|---------|-------------|
| #1 register_copy fragment 分块装载/写回 | UB `T.copy` 分块搬移（§4.4） | 否 |
| #2 coalesced 向量化访问 | UB 连续 slice `T.copy` + Vector 整块 v-prefix（§3.2/§6） | 否 |
| #3 `threads × npt` 两级并行 | §0.6 R1 重设计为一维 Kernel + block_size，元素级并行交给 Vector | 否（意图由 R1 承接） |
| #4 dtype 分支标量 `op_func` | §1.6.2（整型 `vnot` / bool `vxor`）向量化承接 | 否 |
| #5 XOR-with-all-ones（规避 CUDA int4 缺陷） | §1.6.1 项 1 升级为 NPU 原生 `T.vnot`（NPU 无该缺陷） | 否（升级为等价化简） |
| #6 bool 字节存储（int8） | §0.5 等价替换为 uint8 字节存储 + `view(bool)`（§11.4） | 否 |

### 11.3 整数 dtype 契约

- **支持集**：`{bool, uint8, int8, int16, int32, int64}`（六种整型，与 `BITWISE_DTYPES` 一致）；Op / Kernel 层显式拒绝 float（`ValueError`）。
- **计算路径**：整型直接在整型 dtype 上做按位取反（`T.vnot`）；**无 fp16/bf16/fp32 中转**、**无浮点容差**（atol=rtol=0）。
- **依据**：`T.vnot` 的 DataType 表覆盖 uint8/int8/int16/int32/int64（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1）；按位语义不涉及浮点舍入，任何浮点中转都会破坏精确性。

### 11.4 bool 字节存储契约

- kernel 始终以 `uint8` 字节缓冲读写 bool 数据；Op 层在进入 kernel 前 `flat(x).view(torch.uint8)`、离开 kernel 后 `.view(torch.bool)`（均为 stride/shape/dtype 元数据视图操作，**不改输入张量真实内容**，满足 ascend-constraints host 侧约束）。
- 依据：bool i1 无法 load/store GM（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3 / `T.vxor.md` §2.3）；与 GPU 源 `BOOL_STORAGE_DTYPE`（int8）的字节存储策略同源等价。
- bool 逐元素计算：`y_byte = x_byte ^ 1`（字节 ∈ {0,1}），即 §1.6.1 项 2。

