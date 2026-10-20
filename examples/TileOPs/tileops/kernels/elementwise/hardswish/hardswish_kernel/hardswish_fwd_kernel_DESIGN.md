# HardswishFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/activations.py` 的 `HardswishFwdKernel`，复用 `FloatUnaryKernel` / `UnaryKernel` 的 register_copy 策略及其 builder `make_unary_regcopy`）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `hardswish_fwd_kernel`，`target="npuir"`。
> 迁移工作区：`examples/hardswish/hardswish_fwd_kernel/`（project=`hardswish`，op=`hardswish_fwd_kernel`，scenario=migration，mode=harness）。
> 本设计按 skill `tilelang-op-design` 的三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出；本文档为 `first_design` 基线（无 revision 段）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织。先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：`torch.nn.functional.hardswish` 的逐元素激活

$$
y = x \cdot \operatorname{clamp}(x + 3,\ 0,\ 6) \cdot \frac{1}{6}
= \frac{x \cdot \operatorname{relu6}(x + 3)}{6}
$$

源实现（`HardswishFwdKernel.op_func`，`activations.py` 约 L160-166）：

```python
three = T.cast(3.0, "float32")
six = T.cast(6.0, "float32")
zero = T.cast(0.0, "float32")
one_sixth = T.cast(1.0 / 6.0, "float32")
clamped = T.min(T.max(x + three, zero), six)   # 普通 min/max（非 nan_min/nan_max）
return x * clamped * one_sixth
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | `y = x * clamp(x+3, 0, 6) * (1/6)`；缩放以乘倒数 `one_sixth = 1/6` 表达（等价 `x * relu6(x+3) / 6`） |
| 规约语义 | **无规约**，纯逐元素，无跨元素依赖、无累加顺序问题 |
| dtype 语义 | 输入 `float16 \| bfloat16 \| float32`；输出同 dtype（`same_as(input)`）；**中间计算在 fp32**（常数 `3/0/6/1/6` 均 `T.cast(..., "float32")`，`x + three` 提升为 fp32；`x * clamped` 亦在 fp32，最后按输出 dtype 截断） |
| 边界语义 | **有限输入契约 + 无专用 NaN 处理**：op requirement 明确「finite inputs only」。NaN/±inf 行为是普通 `min/max` 与末尾 `x * ...` 乘法**自然涌现**的（非专用链）：`x=NaN` 时 `x*clamped = NaN*0 = NaN` → 输出 NaN（NaN 经 `x` 因子传播，而非经 clamp 传播）；`x=+inf` → `clamp=6` → `+inf*6*(1/6)=+inf`；`x=-inf` → `clamp=0` → `-inf*0=NaN`。源码用普通 `T.min/T.max`（**非** `nan_min/nan_max`），见 §0.4 #6 |
| 空维度 / 非整除 | 输入展平为 1-D `N=numel`，任意 shape 均按展平后处理；非整除由尾块处理 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（中间 fp32、输出同 dtype、`F.hardswish` 语义），**不复刻** §0.6 的 NPU 算法。

**为何 NaN 处理不同于同族 Hardsigmoid（源依据）**：`HardsigmoidFwdKernel.op_func` 的输出是 `clamp(x+3,0,6)*(1/6)`——**没有** `x *` 因子；若用普通 `min/max`，`x=NaN` 会在 clamp 中被 `maxNum` 语义丢弃（`max(NaN,0)=0`）→ 输出 0，**不等价于 torch 的 NaN→NaN**，故它必须用专用 `nan_min/nan_max` 逐层恢复 NaN。`HardswishFwdKernel` 的输出含 `x * ...` 因子，NaN 由该乘法天然传播（`NaN*0=NaN`），普通 `min/max` 已足够，源码因而**刻意选用普通 `T.min/T.max`**（见 `activations.py` 两者注释差异）。本算子沿用源选择：**无 NaN 单点恢复链**（对照 Hardsigmoid 的 §0.6 R2），§1.6.1 亦无对应优化项。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input` | `[*S]`（任意，展平为 1-D `N=numel`） | `float16 \| bfloat16 \| float32` | 输入张量；kernel 收到 `flat(x)` 的 1-D 视图 |
| `output` | 与 input 相同 shape `[*S]` | 与 input 相同 dtype | 输出张量；`forward` 内 `result.reshape(x.shape)` 回塑（输出为输入别名，非转置布局） |

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host 展平：`flat(x)` → 1-D `(N,)` | `input [*S]` | `x_flat (N,)` | 前置（host 视图操作，不改数据） |
| 2 | 分块装载：`T.copy(x[bx*bs:(bx+1)*bs], x_reg)`（fragment） | `x_flat` | `x_reg (bs,)` fragment | 数据搬入 |
| 3 | 逐元素 `t = x + 3`（fp32 常数 three） | `x_reg` | `t`（fp32） | `x + 3` |
| 4 | 逐元素 `c1 = max(t, 0)`（普通 max） | `t` | `c1` | 下界 clamp |
| 5 | 逐元素 `c2 = min(c1, 6)`（普通 min） | `c1` | `c2` | 上界 clamp |
| 6 | 逐元素 `y = x * c2 * one_sixth`（fp32 常数，两连乘） | `x_reg, c2` | `y` | `x * clamp(...) * (1/6)` |
| 7 | 分块写回：`T.copy(y_reg, y[bx*bs:(bx+1)*bs])` | `y_reg` | `y_flat` | 数据搬出 |
| 8 | host 回塑：`result.reshape(x.shape)`；`inplace=True` 时 `input.copy_(result)` | `y_flat` | `output [*S]` | 后置（stride/shape 视图操作） |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x] --T.copy--> fragment/x_reg(寄存器/SMEM) --逐元素 op_func--> fragment/y_reg --T.copy--> GM[y]
```

`HardswishFwdKernel` 继承 `FloatUnaryKernel`（`DEFAULT_STRATEGY = "register_copy"`）。由于输出 dtype 与输入相同，`_narrows(dtype, out_dtype)` 为假，`make_unary_regcopy` 的 `contiguous`（窄化逐线程连续 run）分支**不触发**，一律走 fragment 全块路径：`T.copy` 装载到 `x_reg`，`T.Parallel(threads, npt)` 逐元素计算写入 `y_reg`，`T.copy` 整块写回。

**循环与并行结构**：grid 一维 `T.Kernel(T.ceildiv(N, block_size), threads=128)`，block 内 `threads × npt` 两级并行（`T.Parallel(threads, npt)`）；`block_size = threads × npt`（默认 `threads=128`，`npt = max(min_npt, bytes_per_thread // elem_bytes)`，`register_copy` 下再取 `register_copy_floor`）。

**host 侧逻辑**：`forward` 中 `flat(x)`（展平，视图）→ `_compiled_fn(...)` → `_restore_output_dtype(result).reshape(x.shape)`（回塑）。Op 层（`HardswishFwdOp` 经 `_ParamFreeActivationOp`/`_UnaryActivationMixin`）为 `input.contiguous()` → kernel → reshape；`inplace=True` 时 `input.copy_(result)` 并返回 `input`。全部为 stride/shape 元数据视图操作，**不改输入张量内容**（满足 ascend-constraints 的 host 侧约束）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | register_copy（fragment 分块装载/写回） | 减少 GM 重复访问、放大访存粒度 | 分块 `T.copy` 到 fragment，逐元素计算后整块写回 | CUDA SMEM/寄存器 | 硬件强相关 |
| 2 | coalesced 向量化访问 | 尾轴连续读写打满带宽 | fragment 块 `T.copy` 连续搬运（窄化分支用 `T.vectorized`） | memory coalescing | 硬件强相关（NPU 有 MTE 等价） |
| 3 | 除法转乘倒数（`× one_sixth`） | 除法 → 乘法（更廉价 ALU op） | `1/6` 预计算常数，逐元素乘法 | 无（纯算法层） | 可移植 |
| 4 | 常数上抛 fp32 + 中间 fp32 计算 | 窄 dtype 下保持精度、对齐 torch fp32 opmath | `T.cast(3.0,"float32")` 等；`x+3` 提升 fp32 | 无（纯算法层） | 可移植 |
| 5 | `threads × npt` 两级并行执行模型 | 每线程处理 `npt` 元素放大吞吐 | CUDA thread/warp 组织 | CUDA 线程模型 | 硬件强相关 |
| 6 | NaN 免处理 clamp（普通 `T.min/T.max`） | 依靠 `x *` 因子天然传播 NaN/±inf，省去专用 NaN 检测/回填 | 普通 min/max（`maxNum/minNum` 语义丢 NaN）+ 末尾乘法传播 | 无（算法层选择，不依赖特定 intrinsic） | 可移植 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。识别不出来 ≠ 不存在——上表为源码 `op_func` + `make_unary_regcopy` + `_policy.default_launch_config` 的逐项核对结果。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y=x*clamp(x+3,0,6)*(1/6)`、中间 fp32 | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| 除法转乘倒数（`×1/6`） | 算法 | 无 | — | **保留** | 纯算法层恒等变形，等价性经 verify_equiv.py 机器验证（§1.6.1 项 1） |
| 常数上抛 fp32 + 中间 fp32 中转 | 算法 | 无 | — | **保留** | 对齐 torch golden（fp32 opmath），见 pattern-library 的 PL-1.17-subfp32-fp32-transit（`constants.md` 同目录 elementwise.md） |
| NaN 免处理 clamp（普通 `T.min/T.max`，靠 `x*` 传播） | 算法 | `fminf/fmaxf` 的 `maxNum/minNum` 语义 + 乘法传播 | 有（NPU `vmax/vmin` 同为 `maxNum/minNum` 语义、丢 NaN；乘法传播一致） | **保留** | §0.1 语义论证 + §0.6 反证：hardswish 无需 NaN 单点恢复（不同于 Hardsigmoid/R2）。NPU `vmax/vmin` 丢 NaN 见 `constants.md` `CONST-vcmp-ne-nan-unordered`（「vclamp(vmax/vmin 丢 NaN)」） |
| `threads × npt` 两级并行 + `T.Parallel(threads, npt)` | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel） | **重新设计** | → §0.6 R1（block_size 折叠 + 一维 `T.Kernel` + UB 向量化） |
| fragment 分块装载/写回（SMEM/寄存器） | 优化 | SMEM/寄存器 | 有（UB） | **等价替换** | `T.alloc_ub` + `T.copy`（GM→UB→GM），ascend-constraints 已知限制 / decision-tree §3 |
| coalesced 连续访问 | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | `T.copy` 连续 slice 搬运（双显式 slice 形态） |

> 每项处置均写明依据（migration-analysis.md §5.3 映射表 / `examples/` 佐证 / `constants.md` 条目）。无「舍弃」项。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（`threads × npt` → 一维 Kernel + UB 向量化）**

- **源方案**：`T.Kernel(T.ceildiv(N, block_size), threads=128)` + `T.Parallel(threads, npt)` 两级并行；`block_size = threads × npt`；fragment 装载/写回（`register_copy` 策略）。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；GPU 的 `threads × npt` 折叠为单一 `block_size`（ascend-constraints 已知限制「仅支持一维 Kernel」；GPU `threads=` kwarg 在 NPU 无效果）；元素级并行交给 Vector 单元（v-prefix 逐元素 API 对整块 UB buffer 计算，天然向量化，无需显式 `T.Parallel`）；分块装载/写回用 `T.copy`（GM→UB→GM，32B 对齐连续 slice）。`block_size` 取值见 §5.2（8192，承接 mish 同族元素级实测）。
- **语义保持论证**：逐元素映射不变（每元素独立计算，无跨元素依赖、无规约），`threads/npt` 到 `block_size/核` 的任务映射变化不改变任何输出值——每个 `out[i]` 的表达式与求值顺序与源码逐步一致（`x+3 → max(·,0) → min(·,6) → x*clamped → ×(1/6)`，全程 fp32）。fp32 中转与源算子舍入路径一致（源常数即 fp32 上抛）。边界语义：尾块（非整除）仅 copy 有效元素，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 fragment 尾块行为等价。

> **本算子无需 NaN 重设计项**（对照 Hardsigmoid 的 R2）：源 `HardswishFwdKernel.op_func` 用普通 `T.min/T.max`（非 `nan_min/nan_max`），NaN/±inf 由末尾 `x * ...` 因子天然传播（§0.1），NPU `vclamp`（`vmax/vmin`）与 CUDA `fminf/fmaxf` 同为 `maxNum/minNum` 语义，配合乘法传播得到一致结果。故 §0.5「NaN 免处理 clamp」判定为**保留**，不产生重设计项。op requirement 亦明确 finite inputs only。
>
> §0.5 全部「重新设计」条目（仅 R1）已重设计；其余为保留/等价替换。源优化手段（§0.4 #1/#2/#5）的意图承接：register_copy/coalesced 的「放大访存粒度、整块搬移」意图由 UB `T.copy` 分块搬移承接；`threads×npt` 的「元素级并行放大吞吐」意图由 Vector 单元整块 v-prefix 计算承接；除法转乘倒数（#3）与 fp32 上抛（#4）为算法层直接保留；NaN 免处理（#6）为算法层直接保留。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`HardswishFwdKernel.op_func`）、`.../elementwise/_base.py`（`FloatUnaryKernel`/`UnaryKernel`）、`.../elementwise/_builders.py`（`make_unary_regcopy`）、`.../elementwise/_policy.py`（默认 launch config）。
- 源仓参考实现 / 测试基准：`torch.nn.functional.hardswish`（`ref_program = F.hardswish(x.float()).to(x.dtype)`，见 `examples/TileOPs/tests/ops/test_hardswish.py`）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`F.hardswish`），**不复刻 NPU 算法**（保证验证独立性）。
---

## 1. 概述

### 1.1 算子名称

HardswishFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `hardswish_fwd_kernel`。

### 1.2 功能描述

逐元素 HardSwish 激活：`y = x * clamp(x + 3, 0, 6) * (1/6)`（等价 `x * relu6(x+3) / 6`），用于 MobileNetV3 等轻量网络的非线性激活；输入展平为 1-D 后逐元素计算，中间 fp32，输出与输入同 shape/dtype。

### 1.3 数学公式

$$
y = x \cdot \operatorname{clamp}(x + 3,\ 0,\ 6) \cdot \frac{1}{6}
$$

### 1.4 算法描述

单步逐元素映射：展平为 1-D `N` 后，对每个元素独立计算 `x * min(max(x+3, 0), 6) * (1/6)`，中间 fp32。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**。本算法与 §1.6.0 调研选定算法一致（逐元素直接映射，无更优算法族）。与源算法结构差异：仅并行组织（GPU `threads×npt` → NPU 一维 Kernel + `block_size` 折叠，§0.6 R1）不同；计算表达式本身（`x+3 → clamp → x*clamped → ×1/6`，fp32）与源 `op_func` 逐步一致。**无 NaN 单点恢复步**（不同于 Hardsigmoid），NaN/±inf 由 `x *` 因子天然传播（§0.1）。

### 1.5 数据流图

```
输入 input [*S] --host 展平--> x_flat (N,) --T.copy--> UB[x_ub] --vcast--> UB[x_f32]
  --vadd(+3)--> UB[buf1] --vclamp[0,6]--> UB[buf2] --vmul(x_f32 × buf2)--> UB[buf1]
  --vmul(×1/6)--> UB[buf2] --vcast--> UB[y_ub] --T.copy--> y_flat (N,) --host 回塑--> 输出 [*S]
```

### 1.6 算法调研与优化分析

> 设计第一优先级：先调研四问（§1.6.0），再数学等价优化公式（§1.6.1），再向量化替代分析（§1.6.2），最后向量化轴/布局决策（§1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：hardswish 的算法族（所有能算出相同结果的算法结构）。源算法/输入公式只是基线候选之一。本算子为**单步逐元素激活**，按 `algorithm-research.md` §2 判为**轻量调研**（四问各一行结论 + 依据），但四问不可跳过。对照 `algorithm-candidates.md` 的 elementwise 命中行（R1：公共子表达式消除 / 除法乘倒数 / cast 链合并；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返——本算子无中间 GM 往返）。

**R1 等价化简公式候选**（基线候选在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源公式） | `x * clamp(x+3,0,6) * (1/6)` | — | — | ✅（基线） |
| 1 | relu6 原生式 | `x * min(max(x+3,0),6) / 6` | 数学恒等（与基线仅除法/乘法形之差） | 无（同 op 数） | ❌（无结构差异） |
| 2 | 除法转乘倒数 | `x * clamp(x+3,0,6) * (1/6)` | 容差内等价（1/6 倒数舍入，≤1 ulp） | N 次除法 → 1 次编译期倒数 + N 次乘法（源已采纳，基线即此式） | ✅（基线即此式） |
| 3 | 分段 / 条件式 | `x`（x≥3）或 `0`（x≤-3）或 `x*(x+3)/6` | 数学恒等（有限域），但引入逐元素条件分支 | 无（op 数不降，破坏向量化；±inf/NaN 边界与乘法式不同） | ❌（R4 更差，不纳入） |

**结论**：无更优于基线的化简公式。基线（`x * clamp(x+3,0,6) * (1/6)`）已是最小形式：1 add + 1 two-sided clamp + 2 mul，无公共子表达式、无超越函数、无 cast 链冗余、无中间 GM 往返。候选 3 的分段式虽 flop 数相近但引入条件分支，反而破坏向量化（§1.6.2 / §1.6.3），不予采纳。

**R2 在线算法**：**无「在线/离线」之分**（即天然单遍）——hardswish 是逐元素映射，每个输出元素只依赖对应输入元素，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（`algorithm-research.md` §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导，不得只比 FLOPs）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 `x * clamp(x+3,0,6) * (1/6)` | 4N（add 1 + two-sided clamp 1 + mul 1 + 缩放 1，manifest 口径 `4 * prod(S)`） | 2N·elem_bytes（读 N + 写 N，无中间 GM 往返） | 1 | O(block_size)（UB 驻留，无 GM 中间缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1（relu6 原生式） | 4N | 2N·elem_bytes | 1 | O(block_size) | 完全并行 |
| 候选 3（分段/条件式） | 4N（条件分支不增加 flop，但破坏 SIMD 一致性） | 2N·elem_bytes | 1 | O(block_size) | 完全并行但 lane 内分支发散 |

**结论**：候选间 FLOPs / 访存 / 遍数 / 中间缓冲四口径全同，基线即为最优（无中间 GM 往返、单遍、O(block) 驻留），无并列争议；候选 3 在 R4 因条件分支失分。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰须附 docs/ 或 examples/ 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线 | 逐元素 → Vector（vadd/vclamp/vmul），**不落 Cube**，与元素级算力正确匹配 | 工作集 = O(block_size) UB 驻留，128–160KB < 192KB（§4.5） | 尾轴连续，block_size=8192 为 32B 整数倍且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓ | 循环边界 `ceildiv(N, block_size)` / `ceildiv(num_logical, num_kernels)` 均为编译期常量（N/block_size 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ✅ 采纳 |
| 候选 1 | 同上 | 同上 | 同上 | 同上 | 同上 | ✅（与基线无差异，并入基线） |
| 候选 3 | 逐元素条件分支 → Vector，但 `T.if_then_else`/条件域在 npuir 上会**标量化**（破坏 lane 级向量化；见 pattern-library traps 中「条件构造标量化」类条目） | 同上 | 同上 | 同上 | 分支阻碍向量链融合 | ❌ 淘汰（破坏向量化，无收益） |

**定量 roofline（D-2）**：基线容量/带宽/发射均匹配 Vector 单元（工作集 < UB、尾轴 32B 对齐、约束可静态化），无候选因违反容量/带宽量级被淘汰。采用 `constants.md` 实测常数条目：

**调研结论**：选定算法族 = **逐元素直接映射（单遍，`x * clamp(x+3,0,6) * (1/6)`，fp32 中转，无 NaN 专用链）**。关键依据：R1 无更优化简公式（基线已最小，分段式反破坏向量化）、R2 天然单遍、R3 四口径全同（带宽受限，访存主导）、R4 纯 Vector 亲和（UB 驻留、32B 对齐、静态边界）。与基线结构差异：**无**（调研确认源算法即最优，仅需按 §0.6 R1 重设计其并行组织为 NPU 形态）。「无更优替代」调研范围：`algorithm-candidates.md` 的 elementwise 命中行、恒等变形自查（R1 候选 1/3）、pattern-library elementwise 的 PL-1.6-copy-floor / PL-1.17-subfp32-fp32-transit 同类实现、源码 §0.3/§0.4 的算法与优化手段。源优化手段（§0.4）在调研后无被替换，无需意图承接中继。

**设计期估算**（D-2，轻量调研豁免容量项，流量项必算）：

```
估算下界(smoke-1m fp16): max(流量 ≈ 3.1µs, 发射 ≈ 3.0µs) ≈ 3.1µs
  流量项: 2 × 1048576 × 2B = 4MB；MTE2 有效带宽按 CONST-mte2-degradation 的 1M 档 33.9 GB/s/core × 40 向量核 ≈ 1.36 TB/s（注：1M 档含 L2 驻留成分，实际可能更高）→ 4MB / 1.36TB/s ≈ 3.1µs
  发射项: 向量链 vcast + vadd + vclamp + vmul + vmul + vcast ≈ 6 op × 0.5µs/op（CONST-vector-launch-overhead，f16≈f32）≈ 3.0µs
  容量项: 轻量调研豁免（单步逐元素，无 fixed-loop 跨引擎往返；CONST-store-fixpipe-gm-only 不适用——无 Cube）
  参考: 同族 mish 实测 smoke-1m fp16 ≈ 7.0µs（Stage 4）；本算子链更短（6 op vs mish ~9 op），估算更低合理
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线）。采纳项经 verify_equiv.py 机器验证（结果表见下）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | 除法转乘倒数 | `x * clamp(x+3,0,6) / 6`（torch `F.hardswish` 语义） | `x * clamp(x+3,0,6) * (1/6)`（`one_sixth` 编译期常数，源 `op_func` 形式） | 容差内等价：`1/6` 在 fp32 舍入，`(x*clamped)*(1/6)` 与 `(x*clamped)/6` 在 fp64 域逐元素差 ≤1 ulp；公共因子 `(x*clamped)` 与乘法次序不变，仅缩放算子形态改变。verify_equiv.py 项 1：max 1 ulp / 0 违反，三 dtype 舍入路径均 EQUIV_PASS | N 次除法 → 1 次编译期倒数（常数折叠）+ N 次乘法（除法在硬件上更昂贵）；此式与源公式一致（源已采纳） |

**机器验证结果表**（`verify_equiv.py`，fp64 参照 + 随机/角点 + 逐 dtype 舍入，已执行）：

| 采纳项 | max ulp 差 | 违反率 | 结论 |
|--------|-----------|--------|------|
| 1. 除法转乘倒数（`x*clamp(...)/6` → `x*clamp(...)*(1/6)`） | 1.000 | 0 | EQUIV_PASS |

逐 dtype 舍入路径对照（fp32 中转后按输出 dtype 舍入）：float32 = 0 ulp / 0 违反 / EQUIV_PASS；float16 = 0 ulp / 0 违反 / EQUIV_PASS；bfloat16 = 0 ulp / 0 违反 / EQUIV_PASS。

**优化结论**：采纳 1 项。优化后公式为：

$$
y = x \cdot \operatorname{clamp}(x + 3,\ 0,\ 6) \cdot \frac{1}{6}
$$

（fp32 中转，输出同 dtype；供 §3.1 公式拆解使用）。无其他可采纳优化项：单次逐元素映射，无公共子表达式（`x+3` 只用一次；`x` 为输入，非可消冗余）、无超越函数、无 cast 链冗余、无中间 GM 往返。本算子**无 NaN 单点恢复优化项**——源 `HardswishFwdKernel.op_func` 用普通 `T.min/T.max`，NaN/±inf 由 `x *` 因子天然传播（§0.1），无需 Hardsigmoid 的逐层/单点 NaN 恢复链。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | 逐元素 `x + 3` | `T.Parallel` 逐元素标量 `x_reg[i]+3` | `T.vadd(x_f32_ub, 3.0, buf1_ub)`（`docs/Tilelang.language/数学操作/T.vadd.md` + mish 佐证） | ✅ | — |
| 2 | 逐元素 clamp `min(max(t,0),6)` | 逐元素 `T.min(T.max(...))` | `T.vclamp(buf1_ub, buf2_ub, 0.0, 6.0)`（`docs/Tilelang.language/比较操作/T.vclamp.md`：`T.vclamp(src, dst, min_val, max_val)`）或 `T.vmax`+`T.vmin` | ✅ | — |
| 3 | 逐元素 `x * clamped` | 逐元素 `x_reg * c2` | `T.vmul(x_f32_ub, buf2_ub, buf1_ub)`（`docs/Tilelang.language/数学操作/T.vmul.md`，tensor×tensor 同 rank） | ✅ | — |
| 4 | 逐元素 `× 1/6` | 逐元素 `* one_sixth` | `T.vmul(buf1_ub, one_sixth, buf2_ub)`（`T.vmul` 支持 src2 为标量） | ✅ | — |
| 5 | dtype cast（fp16/bf16 ↔ fp32） | 隐式类型提升 | `T.vcast(src, dst, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`：f16→f32 仅 rint；f32→f16/bf16 含 rint） | ✅ | — |
| 6 | block 起始索引 `offset = block_id * block_size` | 标量乘加 | 无向量 API | ❌ | block 级任务映射索引，每 block 少量标量，非逐元素热点（可接受理由类别①） |
| 7 | 尾块大小 `tail_size = T.min(block_size, remaining)` | 标量 min | 无（`T.min` 标量形） | ❌ | 依赖动态 shape 的边界处理，每 block 一次，非逐元素热点（可接受理由类别④；mish 同款尾块处理） |

**向量化结论**：逐元素计算（`x+3` / clamp / `x*clamped` / `×(1/6)` / cast）已**全部向量化**（替代 API 均有 docs 与 examples 佐证：vadd / vclamp / vmul / vcast）。保留 2 处标量：block 起始索引（block 级任务映射）与尾块大小（动态边界），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴豁免说明**：本算子输入展平为 1-D `(N,)`，仅一个非 batch 维（展平轴），无 ≥2 个非 batch 维度可作向量化轴选择——逐元素激活在**唯一的展平轴**上向量化（Vector 单元 128bit：fp16/bf16 ×8、fp32 ×4），不存在广播轴、窗口轴、归约轴或重排布局候选。按 SKILL.md Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」。

- **向量化轴** = 展平轴（stride=1 连续，无累加链、无跨步系数）。
- **整除性**：block_size=8192，fp16/bf16 向量宽 ×8、fp32 ×4 均整除（8192 = 1024×8 = 2048×4）✓；尾块 `tail_size` 动态，仅 copy 有效元素，垃圾 lane 不写回（与 mish 同款）。
- **repack 代价**：无（输入即 1-D 连续，无需转置/重排）。
- **UB 容量影响**：驻留缓冲 = O(block_size)，预算见 §4.5（128–160KB < 192KB）。
- **迁移对照**：GPU 源算子的并行轴选择（`threads` 线程 × `npt` 每线程元素）是 CUDA 线程模型，与 NPU Vector 展平轴不对应——本设计独立按展平轴向量化（§0.6 R1）。

**布局决策结论**：选定「I/O 原生 1-D 连续布局 + 展平轴向量化」，无 repack、无重排候选（单轴豁免）。该结论同步落入 §3.3 伪代码与 §6 循环结构（向量维 = 展平轴，buffer 均为 1-D）。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（迁移默认）。

### 2.2 选型理由

纯 Vector 逐元素算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。Developer 模式由编译器自动管理同步与 buffer 分配，与 mish / hardsigmoid 同族 elementwise 先例一致（NPU wrapper `HardswishFwdKernel` 声明 `ascend_mode = "Developer"`，见 `examples/TileOPs/tileops/kernels/elementwise/hardswish/hardswish.py`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix API 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（vadd / vclamp / vmul / vcast），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 set_flag/wait_flag） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_f32 = cast(x)`（fp16/bf16 时） | 输入升 fp32（fp32 输入跳过），对齐 torch golden 的 fp32 opmath |
| 2 | `t = x_f32 + 3.0` | 加法 |
| 3 | `c = clamp(t, 0.0, 6.0)` | 双侧 clamp（等价 `min(max(t,0),6)`） |
| 4 | `m = x_f32 * c` | 乘输入（NaN/±inf 由此因子天然传播） |
| 5 | `y = m * one_sixth`（`one_sixth = 1/6`） | 缩放（乘法倒数） |
| 6 | `y_out = cast(y)`（fp16/bf16 时） | 降回输出 dtype |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 数据搬入 GM→UB | `T.copy(x[offset:offset+tail], x_ub[0:tail])` | 双显式 slice（TRAP-T-copy-region-semantics 推荐形态） | Developer |
| 1 | cast | `T.vcast(x_ub, x_f32_ub, round_mode="rint")` | fp16/bf16 → fp32 | Developer |
| 2 | +3 | `T.vadd(x_f32_ub, 3.0, buf1_ub)` | 标量广播 3.0 | Developer |
| 3 | clamp | `T.vclamp(buf1_ub, buf2_ub, 0.0, 6.0)` 或 `T.vmax(buf1_ub, 0.0, buf2_ub); T.vmin(buf2_ub, 6.0, buf2_ub)` | 标量广播 0.0/6.0 | Developer |
| 4 | x × clamped | `T.vmul(x_f32_ub, buf2_ub, buf1_ub)` | tensor×tensor（同 rank 1-D） | Developer |
| 5 | ×1/6 | `T.vmul(buf1_ub, one_sixth, buf2_ub)` | 标量 `one_sixth`（fp32 常数） | Developer |
| 6 | cast 回 | `T.vcast(buf2_ub, y_ub, round_mode="rint")` | fp32 → fp16/bf16 | Developer |
| 7 | 数据搬出 UB→GM | `T.copy(y_ub[0:tail], y[offset:offset+tail])` | 双显式 slice | Developer |

> fp32 输入路径跳过步骤 1/6（无 vcast，走同 dtype `T.copy`）。常数 `three/zero/six/one_sixth` 均 `T.cast(..., "float32")` 上抛 fp32，与源 `op_func` 一致。**无 vbrc / vcmp / vselect**（本算子无 NaN 专用链）。

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型以**第二个位置参数**传入（不用 `dtype=` 关键字），避免产生 `name 'in_dtype' is not defined` 的 false alarm。

```python
@tilelang.jit(out_idx=[1], target="npuir")
def hardswish_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    num_logical = T.ceildiv(N, block_size)            # 逻辑 tile 数（编译期常量）
    # 物理核数 = NPUUtils.get().get_aicore_num() * 2（纯 Vector 翻倍，实查见 §5.5）
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_logical, num_kernels)   # 编译期常量，静态边界

    three = T.cast(3.0, "float32")
    zero = T.cast(0.0, "float32")
    six = T.cast(6.0, "float32")
    one_sixth = T.cast(1.0 / 6.0, "float32")

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # 1. 分配 UB buffer（v-prefix API 要求 UB 驻留）
            x_ub = T.alloc_ub((block_size,), dtype)
            x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
            buf1_ub = T.alloc_ub((block_size,), compute_dtype)
            buf2_ub = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # 2. persistent 串行 tile 循环（核内串行，边界静态）
            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # 3. 数据搬入（仅有效元素）
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 4. cast 到 fp32（fp16/bf16）或 copy（fp32）
                    if dtype != compute_dtype:
                        T.vcast(x_ub, x_f32_ub, round_mode="rint")
                    else:
                        T.copy(x_ub, x_f32_ub)

                    # 5. 计算链：x * clamp(x+3, 0, 6) * (1/6)
                    T.vadd(x_f32_ub, three, buf1_ub)       # t = x + 3
                    T.vclamp(buf1_ub, buf2_ub, zero, six)  # c = clamp(t, 0, 6)
                    T.vmul(x_f32_ub, buf2_ub, buf1_ub)     # m = x * c（buf1 的 t 已死）
                    T.vmul(buf1_ub, one_sixth, buf2_ub)    # y = m * (1/6)（buf2 的 c 已死）

                    # 6. cast 回输出 dtype（fp16/bf16）或 copy（fp32）
                    if out_dtype != compute_dtype:
                        T.vcast(buf2_ub, y_ub, round_mode="rint")
                    else:
                        T.copy(buf2_ub, y_ub)

                    # 7. 数据搬出（仅有效元素）
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main
```

> 注：`T.Tensor((N,), dtype)` 用位置参数传 dtype（避免 false alarm，见本节开头约定）。NaN/±inf **不**做特殊处理——`x_ub`/`x_f32_ub` 中的 NaN 经步骤 5 的 `T.vmul` 天然传播（§0.1）。fp32 路径（`dtype==compute_dtype and out_dtype==compute_dtype`）可走更简 buffer 链：`x_f32_ub` 复用 `x_ub`（输入即 fp32，无需 vcast）、`y_ub` 复用 `buf2_ub`（输出即 fp32，`T.copy(buf2_ub, y)` 直接回 GM），驻留从 160KB 降至 96KB（§4.5），Stage 3 实现时细化。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vadd` | `docs/Tilelang.language/数学操作/T.vadd.md`（`T.vadd(src0, src1, dst)`，src1 可标量；fp16/fp32 ✓、bf16 ×）+ `customize_npuir.py` `npuir_add` | ✅ 已验证（docs + 源码读） |
| `T.vmul` | `docs/Tilelang.language/数学操作/T.vmul.md`（`T.vmul(src1, src2, dst)`，仅 src2 可标量；dst 可原地；fp16/fp32 ✓、bf16 ×） | ✅ 已验证（docs 读） |
| `T.vclamp` | `docs/Tilelang.language/比较操作/T.vclamp.md`（`T.vclamp(src, dst, min_val, max_val)`）+ `customize_npuir.py` `npuir_clamp` | ✅ 已验证（docs + 源码读） |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（f16→f32 仅 rint；f32→f16/bf16 含 rint）+ `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`（`round_mode="rint"`） | ✅ 已验证 |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md` + mish 尾块先例（TRAP-T-copy-region-semantics：src/dst 双显式 slice 为推荐通用形态） | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` / `examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md` 同族先例 | ✅ 已验证 |
| `T.alloc_ub` | `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | ✅ 已验证 |
| `T.ceildiv` / `T.min`（标量） | `docs/Tilelang.language/数学操作/T.ceildiv.md` + mish 尾块 `T.min` 先例 | ✅ 已验证 |

> 无 vbrc / vcmp / vselect / isnan（本算子无 NaN 专用链，与 Hardsigmoid 设计的关键差异）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 `T.Kernel(ceildiv(N, block_size))`） | 一维 `T.Kernel(num_kernels, is_npu=True)`，无三维 |
| GPU 专用 API | Yes（`T.get_thread_binding` / `T.vectorized` / `T.unroll` CUDA 语义；bf16 不可用原生向量算术） | 全部改写为 NPU v-prefix API + UB 路径（§0.5/§0.6 R1）；bf16 经 fp32 中转（§3.1/§3.4） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(N, block_size), threads=128)`（block+thread 两级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(threads, npt)` 显式两级循环 | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| `threads×npt` | 128×npt（default threads=128） | `block_size=8192` 折叠 | §5.2 |
| NaN 处理 | 普通 `T.min/T.max`（靠 `x*` 传播，源码即如此） | 普通 `vclamp`（`vmax/vmin` 丢 NaN）+ `x*` 传播 | 无需专用链（与 Hardsigmoid 不同，§0.1/§0.6） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 高度相似（同族 unary 激活迁移） | Developer 模式、`T.alloc_ub`、`T.copy` 尾块、v-prefix 链、fp32 中转、`T.Kernel(is_npu=True)` |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/mish.py` | 高度相似（Stage 4 调优） | persistent 核内串行 + 2-stage 流水 + block_size=8192 |
| `examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md` | 高度相似（同族迁移设计，hardsigmoid/hardswish/hardtanh/relu6 族） | 一维 Kernel + block_size 折叠、fp32 中转、分核三要素、UB 预算口径（本算子为其「无 NaN 链」变体） |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 相似（同族 elementwise） | fp32 中转模式（PL-1.17）、`T.min` 尾块 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `input` | `[*S]`（展平 1-D `N`） | `float16 \| bfloat16 \| float32` | 输入张量，host 展平视图 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | 与 input 相同 | 与 input 相同 dtype | 输出张量，host 回塑 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | input dtype | UB | 输入 tile（GM→UB） |
| `x_f32_ub` | `(block_size,)` | float32 | UB | fp32 中转输入（vcast 目标；末尾 `x*clamped` 用） |
| `buf1_ub` | `(block_size,)` | float32 | UB | 中间结果（t / `x*clamped` 复用） |
| `buf2_ub` | `(block_size,)` | float32 | UB | 中间结果（clamp 输出 / 最终结果复用） |
| `y_ub` | `(block_size,)` | output dtype | UB | 输出 tile（UB→GM） |

### 4.4 内存搬运路径

```
GM[x] --T.copy--> UB[x_ub] --vcast--> UB[x_f32_ub]
  --vadd(+3)--> UB[buf1_ub] --vclamp[0,6]--> UB[buf2_ub]
  --vmul(x_f32 × buf2)--> UB[buf1_ub] --vmul(×1/6)--> UB[buf2_ub]
  --vcast--> UB[y_ub] --T.copy--> GM[y]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `x_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| `x_f32_ub` | (8192,) | fp32 | 32768 |
| `buf1_ub` | (8192,) | fp32 | 32768 |
| `buf2_ub` | (8192,) | fp32 | 32768 |
| `y_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| **总计（fp16/bf16 路径）** | | | **131072（128KB）** |
| **总计（fp32 路径，全 buffer 显式）** | | | **163840（160KB）** |
| **总计（fp32 路径，复用 `x_f32_ub`→`x_ub` + `y_ub`→`buf2_ub`）** | | | **98304（96KB）** |

对照目标平台 UB 容量 **192KB**（`constants.md` `CONST-capacity-910B2C`：UB=192KB/AIV）。按该条目记录的 ×1.10–1.12 上偏系数（`--enable-auto-multi-buffer=false` 显式 alloc 结构下 BishengIR 实际需求 ≈ 手工预算 ×1.10–1.12）：

- **fp16/bf16 路径**：128KB ×1.12 ≈ 143KB < 192KB ✓，余量 ~25%。
- **fp32 路径（全 buffer 显式）**：160KB ×1.12 ≈ 179KB < 192KB ✓，余量 ~7%。
- **fp32 路径（复用方案，推荐）**：fp32 输入即 fp32，`x_f32_ub` 与 `x_ub` 等价，复用 `x_ub` 直接计算（省 32KB → 128KB）；输出即 fp32，`buf2_ub` 结果直接 `T.copy` 回 GM（`y_ub` 可省，再省 32KB → 96KB，×1.12 ≈ 108KB，余量 ~44%）。

结论：全部 dtype 路径均不溢出（最保守的 fp32 全 buffer 显式 160KB ×1.12 ≈ 179KB 仍在 192KB 之内），**无需单独调小 block_size**。Stage 3 实现时按 fp32 路径走复用链（`x_f32_ub`→`x_ub`、`y_ub`→`buf2_ub`），预算与分配一致。

### 4.6 动态轴定义

无动态轴。`N` 为 JIT 编译期参数（`hardswish_fwd_kernel(N, dtype)` 每 shape 一特化），`shape`（`T.int32`）为运行时传入的实际元素数（用于尾块 `remaining = shape - offset`，与 mish / hardsigmoid 同款）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素运算（add / clamp / mul / mul / cast），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_size = 8192   # 同族 mish / hardsigmoid Stage 4 实测最优
num_logical = T.ceildiv(N, block_size)   # 逻辑 tile 数
```

- **选择理由**：8192 为 mish 同族 elementwise Stage 4 实测最优 block_size（`mish_kernel/perf_opt/mish.py`），hardsigmoid 迁移设计同取 8192；32B 对齐（8192 元素 × 2/4B 均为 32B 整数倍）；UB 预算 128–160KB < 192KB（§4.5，fp32 复用后 ≤96KB）；块宽足够使向量链发射开销摊薄（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。

### 5.3 约束分析

- **对齐约束**：block_size=8192，fp16/bf16 尾轴 8192×2B=16KB、fp32 8192×4B=32KB，均 32B 对齐 ✓；整除向量宽（fp16/bf16 ×8、fp32 ×4）✓。
- **UB 容量**：128–160KB < 192KB（§4.5；fp32 全 buffer 160KB ×1.12 ≈ 179KB 余量 ~7%，复用后 128KB ×1.12 ≈ 143KB 余量 ~25%）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾块**：`tail_size = T.min(block_size, remaining)`，仅 `T.copy` 有效元素进/出；v-prefix 对全 buffer 计算，尾部垃圾元素不写回（与 mish 同款）。
- **极小 shape**（如 N < 8192）：`num_logical=1`，`num_kernels=1`，单核单 tile 处理，无 oversubscription。

### 5.5 分核策略（物理核数适配）

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（Cube/混合直接用返回值、纯 Vector 翻倍），禁止文档假设/经验值。

- **物理核数（实查）**：`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-13 于本任务环境实查）。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。查询代码与实际返回值记录于此：

```python
from tilelang.utils.npu_utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 翻倍：40
```

（注：`constants.md` `CONST-aicore-910B2C` 记录 910B2C 为 24 核，本环境设备实查返回 20，设备型号差异——一律以实查为准。）

- **逻辑核数**：`num_logical = ceil(N / block_size)`，随 workload（NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `HardswishFwdOp` 条目）变化：

| workload | shape | N = prod(S) | 逻辑核数 ceil(N/8192) | 每核串行 tile 数 ceil(num_logical/40) |
|----------|-------|-------------|----------------------|-----------------------------------|
| smoke-1m | [1048576] | 1,048,576 | 128 | 4 |
| mbv3-stage2 | [32,96,56,56] | 9,633,792 | 1176 | 30 |
| mbv3-stage3 | [32,240,28,28] | 6,021,120 | 735 | 19 |
| fc-wide | [2048,4096] | 8,388,608 | 1024 | 26 |

- **规模判定与分核方案**：workload 跨越「无需适配」到「极大规模」两档。设计采用**统一 persistent 结构**一次覆盖全部档位（成熟 elementwise 形态，mish perf_opt / VP-2026-0003 persistent 模板）：

  - **小规模（逻辑核数 ≤ 40，即 N ≤ 327680）**：`num_kernels = num_logical`，每核 ≤1 tile，`num_local_tasks=1`，无串行循环、无负载不均——结论「**无需适配**」（逻辑核数 ≤ 物理向量核数）。
  - **中等/极大规模（逻辑核数 > 40，如 smoke-1m 128、mbv3-stage3 735、fc-wide 1024、mbv3-stage2 1176）**：固定启动内核数 = 40（= 物理向量核数），核内 `T.serial` **串行**处理 `num_local_tasks = ceil(num_logical/40)` 个 tile（smoke-1m：4；mbv3-stage3：19；fc-wide：26；mbv3-stage2：30），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均；循环边界 `num_local_tasks` 为编译期常量（N/block_size/40 均编译期已知），满足「**核内串行**边界静态」。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（vadd / vclamp / vmul / vcast） | 块内逐元素由 Vector 单元并行（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # UB buffer 分配 ...
    for i in T.serial(num_local_tasks):        # 核内串行 tile 循环（静态边界）
        block_id = i * num_kernels + cid
        if block_id < num_logical:             # 尾 tile guard
            # 装载 → v-prefix 计算链 → 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对 cast 路径采用 guard-free 2-stage `T.Pipelined(num_local_tasks, num_stages=2)` 双缓冲流水（mish perf_opt 形态），将向量链隐藏于 MTE2 窗口（PL-1.6-copy-floor：≥16M 档计算链被完全隐藏）。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 shape：`tail_size = T.min(block_size, remaining)`；`T.copy` 仅装载/写回 `[0:tail_size]` 有效元素；v-prefix 对全 block_size buffer 计算，垃圾尾部不写回（与 mish 同款）。`block_id < num_logical` guard 覆盖最后一波不完全 tile。

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

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（中间 fp32、输出同 dtype、`F.hardswish` 语义），移植源仓参考实现（`torch.nn.functional.hardswish`），**不复刻 §0.6 的 NPU 算法**。

```python
import torch
import torch.nn.functional as F

def golden_hardswish(x):
    """HardSwish 参考实现：y = x * clamp(x + 3, 0, 6) / 6（torch 语义，fp32 opmath）。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_hardswish.py 的
    ref_program = F.hardswish(x.float()).to(x.dtype)）。
    独立于 NPU 算法（不含 vclamp/vmul fp32 链/persistent 结构），仅依赖 torch。
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = F.hardswish(x_f32)   # = x * clamp(x + 3, 0, 6) / 6
    return y_f32.to(x_cpu.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |
| float32 | 1e-5 | 1e-5 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓测试 `_get_tolerances` 一致。kernel 用 `×(1/6)` vs golden 语义差 ≤1 ulp，落在容差内；fp32 中转链对齐 torch fp32 opmath golden，见 PL-1.17-subfp32-fp32-transit。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`N=1048576`，`float32` / `float16` / `bfloat16` 三 dtype，`randn`（有限值）输入，vs golden（`F.hardswish`）在对应容差下 `torch.testing.assert_close`。完整分层套件（L1 非整除 / 多 shape、L2 边界 size、Boundary 特殊值）交由 `tilelang-op-develop` 规划，本设计不枚举。

**特别说明（边界用例）**：op requirement 为 **finite inputs only**，L0 用有限 `randn`。若 Stage 3 规划 Boundary 用例，须注意本算子**无 NaN 专用链**——NaN 由 `x*clamped` 乘法天然传播（`NaN*0=NaN`），`+inf`→`+inf`、`-inf`→`NaN`；这与 Hardsigmoid 的 NaN 单点恢复链（`vcmp`+`vselect`）是**不同**语义路径，验证时应以 `F.hardswish` golden 为准，不得套用 hardsigmoid 的 NaN 断言。
---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| bf16 无原生 v-prefix 算术 | 官方 dtype 表对 `T.vadd` / `T.vmul` / `T.vclamp` 的 bf16 列标 ×（`docs/Tilelang.language/数学操作/T.vadd.md` / `T.vmul.md`），故 bf16 输入必须经 `T.vcast(bf16→fp32, rint)` 中转、算完再 `T.vcast(fp32→bf16)`；这也顺带对齐 torch fp32 opmath golden（PL-1.17-subfp32-fp32-transit） |
| UB 预算 | 128KB（fp16/bf16）/ 160KB（fp32 全 buffer 显式）< 192KB；fp32 全 buffer ×1.12 ≈ 179KB 余量 ~7%（复用 `x_f32_ub`→`x_ub` + `y_ub`→`buf2_ub` 后 ≤96KB）。Stage 3 复用 buffer 进一步放宽 |
| 分核 | 物理核数实查 20（纯 Vector ×2 = 40），与 `constants.md` `CONST-aicore-910B2C`（24）不一致——本环境设备返回 20，一律以实查为准 |
| NaN/±inf 由 `x*` 乘法传播 | 使用普通 `vclamp`（`vmax/vmin` 丢 NaN），不引入 NaN 专用链：`x=NaN`→输出 NaN（`NaN*0`），`x=+inf`→`+inf`，`x=-inf`→NaN（`-inf*0`），与 `F.hardswish` 一致。op requirement 为 finite inputs only，L0 用有限输入；此性质为源码既有语义（普通 `T.min/T.max`），非本设计新增风险 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_size 过大（>8192×fp32 全 buffer 链） | 编译失败（`ub overflow`） | 维持 block_size=8192；fp32 路径复用 buffer 减少驻留 |
| bf16 直接进 v-prefix 算术 | 忘记 fp32 中转，直接对 bf16 buffer 调 vadd/vmul | 编译失败（bf16 ×） | bf16 输入先 `vcast` 升 fp32（§3.1 步骤 1） |
| 尾块越界 | `T.copy` 用 base+size 形态写尾块 | MTE 越界写坏相邻 tensor | 双显式 slice `T.copy(src[0:tail], dst[0:tail])`（TRAP-T-copy-region-semantics） |
| `dtype=` 关键字 false alarm | `T.Tensor((N,), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（§3.3 约定） |
| inplace 语义丢失 | Op 层未在 `inplace=True` 时拷贝回输入 | 调用方拿不到原地写回 | Op 层 `input.copy_(result)` 并返回 `input`（§0.3；沿用 NPU wrapper `HardswishFwdOp.forward`） |

### 9.3 特殊场景处理

- **非整除 shape**（如 N 奇数）：尾块 `tail_size` + `block_id < num_logical` guard（§6.4）。
- **极小 shape**（N < block_size）：`num_logical=1`，单核单 tile。
- **NaN / ±inf 输入**：由 `x*clamped` 乘法天然传播（§9.1），无专用链；L0 契约用有限输入。
- **混合精度**：fp16/bf16 走 fp32 中转（vcast），fp32 直算；输出 dtype 与输入一致。
- **inplace 契约**：Op 保留 `inplace`（bool, kw_only, default false）；`inplace=True` 时前向原地写回输入并返回输入张量（见 §11 的契约说明）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/hardswish/hardswish_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，结果内嵌）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成 | §1.6.1 等价性验证（1 项 EQUIV_PASS） |
| `hardswish_fwd_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_hardswish.py` | ⬜ 待实现（Stage 3） | 精度测试 |

### 10.3 命名规范

- 项目目录名：`hardswish`；算子目录名：`hardswish_fwd_kernel`；接口函数：`hardswish_fwd_kernel`（manifest 键 `HardswishFwdOp`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`hardswish_fwd_kernel`）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）

---

## 11. 迁移决策与设计一致性核对

> 本节逐项核对 §0 迁移决策与 §1–§10 设计的一致性，并说明源优化手段的意图承接（防止「§0 说重设计、§3 又照抄源方案」的脱节）。

### 11.1 §0 → §1–§10 一致性

| §0 决策 | 后续落地 | 一致？ |
|---------|---------|--------|
| 语义层 `y=x*clamp(x+3,0,6)*(1/6)`、中间 fp32 | §1.3/§1.4 公式与算法描述、§3.1 拆解、§8.1 golden | ✅ |
| §0.5 R1「并行结构重设计」 | §3.3 伪代码（一维 `T.Kernel(num_kernels, is_npu=True)` + block_size 折叠 + v-prefix）、§5.2/§5.5、§6.1 | ✅（未照抄 GPU `threads×npt`） |
| §0.5「fragment → UB」等价替换 | §4.3/§4.4（`T.alloc_ub` + `T.copy` GM↔UB） | ✅ |
| §0.5「coalesced → MTE 连续搬运」等价替换 | §3.2 步骤 0/7（双显式 slice `T.copy`） | ✅ |
| §0.5「NaN 免处理 clamp」保留 | §3.2 无 vbrc/vcmp/vselect；§1.6.1 无 NaN 优化项 | ✅（与 Hardsigmoid 的 R2 明确区分） |
| §0.5 除法转乘倒数 / fp32 上抛 保留 | §1.6.1 项 1（机器验证）+ §3.1 步骤 5 | ✅ |

### 11.2 源优化手段（§0.4）意图承接

| §0.4 手段 | 承接结构 | 是否静默丢弃 |
|-----------|---------|-------------|
| #1 register_copy fragment 分块装载/写回 | UB `T.copy` 分块搬移（§4.4） | 否 |
| #2 coalesced 向量化访问 | UB 连续 slice `T.copy` + Vector 整块 v-prefix（§3.2/§6） | 否 |
| #3 除法转乘倒数 | §1.6.1 项 1 直接保留（编译期常数 `one_sixth`） | 否 |
| #4 常数上抛 fp32 + 中间 fp32 | §3.1 步骤 1/6（vcast 中转）+ §3.2 常数 fp32 | 否 |
| #5 `threads × npt` 两级并行 | §0.6 R1 重设计为一维 Kernel + block_size，元素级并行交给 Vector | 否（意图由 R1 承接） |
| #6 NaN 免处理 clamp（普通 min/max） | §3.2 普通 `vclamp`；NaN 经 `x*` 乘法传播 | 否 |

### 11.3 inplace 契约说明

源 manifest（GPU `spec/elementwise_unary_activation.yaml` 与 NPU `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `HardswishFwdOp`）声明 `params: inplace: {type: bool, default: false, kw_only: true}`，且 `input: {mutated: inplace}` / `output: {alias: input}`。Op 层（NPU `HardswishFwdOp`，沿用 GPU `_ParamFreeActivationOp`/`_UnaryActivationMixin` 语义）在 `inplace=True` 时先算 `result = _eager_forward(input)`，再 `input.copy_(result.reshape(input.shape))` 并返回 `input`（张量身份保持 `y is x`）；`inplace=False` 时返回新张量。**kernel 本身与 inplace 无关**：kernel 始终写入独立的输出 buffer（`out_idx=[1]`），原地语义完全在 Op 层用 `copy_` 实现（不改变 kernel 契约），符合 host 侧「不改输入张量内容」约束中「Op 层可写回输出」的语义。
