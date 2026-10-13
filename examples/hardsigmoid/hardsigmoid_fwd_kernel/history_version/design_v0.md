# HardsigmoidFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/`，`HardsigmoidFwdKernel` + `FloatUnaryKernel`/`UnaryKernel` register_copy 策略）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `hardsigmoid_fwd_kernel`，`target="npuir"`。
> 本设计按 skill `tilelang-op-design` 三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：

$$
y = \operatorname{clamp}(x + 3,\ 0,\ 6) \cdot \frac{1}{6} = \frac{\operatorname{relu6}(x+3)}{6}
$$

逐元素激活，与 `torch.nn.functional.hardsigmoid` 一致。源实现为：

```python
# HardsigmoidFwdKernel.op_func（activations.py L177-183）
three = T.cast(3.0, "float32"); six = T.cast(6.0, "float32")
zero = T.cast(0.0, "float32"); one_sixth = T.cast(1.0/6.0, "float32")
return nan_min(nan_max(x + three, zero), six) * one_sixth
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | `y = clamp(x+3, 0, 6) * (1/6)`（除法以乘法倒数 `one_sixth = 1/6` 表达，等价 `relu6(x+3)/6`） |
| 规约语义 | **无规约**，纯逐元素，无跨元素依赖、无累加顺序问题 |
| dtype 语义 | 输入 `float16 \| bfloat16 \| float32`；输出与输入同 dtype（`same_as(input)`）；**中间计算在 fp32**（常数 `3/0/6/1/6` 均 `T.cast(..., "float32")`，`x + three` 提升为 fp32，结果最后按输出 dtype 截断） |
| 边界语义 | **NaN 传播**：输入 NaN → 输出 NaN（源 docstring「A NaN input gives NaN, as in torch」）。`±inf` 经 clamp 收敛到 `0/6`（`clamp(+inf+3,0,6)=6 → y=1`；`clamp(-inf+3,0,6)=0 → y=0`），无上溢/下溢风险 |
| 空维度 / 非整除 | 输入展平为 1-D `N`（`N=numel`），任意 shape 均按展平后处理；非整除由尾块处理 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（NaN → NaN、中间 fp32、输出同 dtype）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input` | `[*S]`（任意，展平为 1-D `N=numel`） | `float16 \| bfloat16 \| float32` | 输入张量；kernel 收到 `flat(x)` 的 1-D 视图 |
| `output` | 与 input 相同 shape | 与 input 相同 dtype | 输出张量；`forward` 内 `_restore_output_dtype(result).reshape(x.shape)` 回塑 |

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host 展平：`flat(x)` → 1-D `(N,)` | `input [*S]` | `x_flat (N,)` | 前置（host 侧视图操作，不改数据） |
| 2 | 分块装载：`T.copy(x[bx*bs:(bx+1)*bs], x_reg)`（fragment） | `x_flat` | `x_reg (bs,)` fragment | 数据搬入 |
| 3 | 逐元素 `t = x + 3.0`（fp32） | `x_reg` | `t`（fp32） | `x+3` |
| 4 | 逐元素 `c1 = nan_max(t, 0.0)` | `t` | `c1` | 下界 clamp |
| 5 | 逐元素 `c2 = nan_min(c1, 6.0)` | `c1` | `c2` | 上界 clamp |
| 6 | 逐元素 `y = c2 * one_sixth`（fp32 常数） | `c2` | `y` | 缩放 `×1/6` |
| 7 | 分块写回：`T.copy(y_reg, y[bx*bs:(bx+1)*bs])` | `y_reg` | `y_flat` | 数据搬出 |
| 8 | host 回塑：`result.reshape(x.shape)` | `y_flat` | `output [*S]` | 后置（host 侧视图操作） |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x] --T.copy--> fragment/x_reg(寄存器/SMEM) --逐元素 op_func--> fragment/y_reg --T.copy--> GM[y]
```

register_copy 策略（`make_unary_regcopy`，`_builders.py` L85-121）：block_size = `threads × npt`，分块 `T.copy` 装载到 fragment，`T.Parallel(threads, npt)` 逐元素计算，`T.copy` 写回。整数倍时走 per-thread 连续 run（`T.vectorized` 装载/写回 + `T.unroll` 计算），否则走 fragment 全块路径。

**循环与并行结构**：grid 一维 `T.Kernel(ceildiv(N, block_size))`，block 内 `threads × npt` 两级并行（`T.Parallel(threads, npt)`）。block_size = `threads(128) × npt(4 fp32 / 8 fp16/bf16)` = 512/1024。

**host 侧逻辑**：`forward` 中 `flat(x)`（展平，视图）→ `_compiled_fn(...)` → `_restore_output_dtype(result).reshape(x.shape)`（回塑）。Op 层（`HardsigmoidFwdOp._eager_forward`）为 `input.contiguous().reshape(-1) → kernel → reshape(orig_shape)`；`inplace=True` 时 `input.copy_(result)`。全部为 stride/shape 元数据视图操作，**不改输入张量内容**（满足 ascend-constraints.md §4）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | register_copy（fragment 分块装载/写回） | 减少 GM 重复访问、放大访存粒度 | 分块 `T.copy` 到 fragment，逐元素计算后整块写回 | CUDA SMEM/寄存器 | 硬件强相关 |
| 2 | coalesced 向量化访问 | 尾轴连续读写打满带宽 | `T.vectorized(npt)` 连续 run 装载/写回 | memory coalescing | 硬件强相关（NPU 有 MTE 等价） |
| 3 | 除法转乘倒数（`× one_sixth`） | 除法 → 乘法（更廉价的 ALU op） | `1/6` 预计算常数，逐元素乘法 | 无（纯算法层） | 可移植 |
| 4 | NaN 传播 min/max（`nan_min`/`nan_max`） | 匹配 torch 的 NaN→NaN 语义 | fp16/bf16 用 `tl.max_nan/tl.min_nan`（CUDA `__hmax_nan/__hmin_nan`）单指令；fp32 用 `isnan + Select` 回填 NaN | CUDA 内建 intrinsic | 硬件强相关 |
| 5 | 常数上抛 fp32 | 窄 dtype 下保持精度、避免 fp16 常数舍入 | `T.cast(3.0,"float32")` 等 | 无（纯算法层） | 可移植 |
| 6 | `threads × npt` 两级并行执行模型 | 每线程处理 `npt` 元素放大吞吐 | CUDA thread/warp 组织 | CUDA 线程模型 | 硬件强相关 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y=clamp(x+3,0,6)/6`、NaN→NaN | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| 除法转乘倒数（`×1/6`） | 算法 | 无 | — | **保留** | 纯算法层恒等变形，等价性经 verify_equiv.py 机器验证（§1.6.1 项 2） |
| 常数上抛 fp32 + 中间 fp32 中转 | 算法 | 无 | — | **保留** | 对齐 torch golden（fp32 opmath），见 PL-1.17-subfp32-fp32-transit（两任务实证） |
| `threads × npt` 两级并行 + `T.Parallel(threads, npt)` | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel） | **重新设计** | → §0.6 R1（block_size 折叠 + 一维 `T.Kernel` + UB 向量化） |
| fragment 分块装载/写回（SMEM/寄存器） | 优化 | SMEM/寄存器 | 有（UB） | **等价替换** | `T.alloc_ub` + `T.copy`（GM→UB→GM），ascend-constraints.md §1 / decision-tree.md §3 |
| coalesced 向量化访问（`T.vectorized`） | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | `T.copy` 连续 slice 搬运（TRAP-T-copy-region-semantics 双显式 slice 形态） |
| NaN 传播 min/max（CUDA `tl.max_nan/tl.min_nan` intrinsic） | 优化 | CUDA intrinsic | 无同款 intrinsic（Ascend vmax/vmin NaN 语义未文档化） | **重新设计** | → §0.6 R2（`isnan` + `vselect` 单点恢复） |

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（`threads × npt` → 一维 Kernel + UB 向量化）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=128)` + `T.Parallel(threads, npt)` 两级并行；block_size = `threads×npt` = 512(fp32)/1024(fp16/bf16)；fragment 装载/写回（`T.vectorized` + `T.unroll`）。
- **NPU 新算法**：一维 `T.Kernel(num_cores, is_npu=True) as (cid, _)`；GPU 的 `threads×npt` 折叠为单一 `block_size`（K11 适配，ascend-constraints 已知限制「仅支持一维 Kernel」）；元素级并行交给 Vector 单元（v-prefix 逐元素 API 对整块 UB buffer 计算，天然向量化，无需显式 `T.Parallel`）；分块装载/写回用 `T.copy`（GM→UB→GM，32B 对齐连续 slice）。`block_size` 取值见 §5（8192，承接 mish 同族元素级实测）。
- **语义保持论证**：逐元素映射不变（每元素独立计算，无跨元素依赖、无规约），`threads/npt` 到 `block_size/核` 的任务映射变化不改变任何输出值——数学上每个 `out[i]` 的计算表达式与求值顺序与源码逐步一致（`x+3 → clamp → ×1/6`，fp32 域）。fp32 中转与源算子舍入路径一致（源常数即 fp32 上抛）。边界语义：尾块（非整除）仅 copy 有效元素，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 fragment 尾块行为等价。

**重设计项 R2：NaN 传播语义重设计（CUDA `tl.max_nan/tl.min_nan` → NPU `isnan`+`vselect` 单点恢复）**

- **源方案**：fp16/bf16 用 `tl.max_nan`/`tl.min_nan`（CUDA `__hmax_nan`/`__hmin_nan`，NaN 传播单指令）；fp32 用 `isnan(a)||isnan(b)` 守卫 + `Select` 回填 NaN（`_nan.py` `_propagate_nan`）。每次 min/max 各一次 isnan 检测 + select。
- **NPU 新算法**：Ascend `vmax`/`vmin`（`tl.npuir_max/min`，`vclamp` 内部即 vmax+vmin）的 NaN 特殊值语义**未在 Ascend C Max/Min 文档「约束说明」中明确**（已检索 CANN 9.0.0-beta1 Max/Min 文档，仅载明地址对齐/重叠约束，无特殊值处理说明）。为保证语义层 NaN→NaN 不变，采用**单点恢复**：用普通 `vclamp`（vmax+vmin）做 clamp，末尾用 `T.isnan(x)` 得到 NaN 掩码 + `T.vselect` 将 NaN 位回填 NaN。由于 NaN 只能源自输入 x（常数 3/0/6/1/6 均有限），逐层传播（源）≡ 单点恢复（新），一次检测即可覆盖。
- **语义保持论证**：数学恒等——NaN 唯一来源是输入 x；`x` 有限时两式给出完全相同的 `clamp(x+3,0,6)*(1/6)`；`x=NaN` 时源式经逐层 nan_min/nan_max 输出 NaN、新式经单点 vselect 输出 NaN，位级一致。已由 `verify_equiv.py` 机器验证（§1.6.1 项 1，0 ulp 差 / 0 违反 / EQUIV_PASS，含 ±inf/NaN/subnormal 角点）。若 Stage 3 实测 Ascend vmax/vmin **本身传播 NaN**，则该 isnan+vselect 可作可选优化移除（见 §9 风险点，不影响正确性）。

> §0.5 全部「重新设计」条目（R1/R2）已重设计；其余为保留/等价替换，无需额外 NPU 算法重设计。源优化手段（§0.4 #1/#2/#4/#6）的意图承接：register_copy/coalesced 的「放大访存粒度、整块搬移」意图由 UB `T.copy` 分块搬移承接；`threads×npt` 的「元素级并行放大吞吐」意图由 Vector 单元整块 v-prefix 计算承接；NaN 传播意图由 R2 单点恢复承接；除法转乘倒数与 fp32 上抛为算法层直接保留。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`HardsigmoidFwdKernel.op_func`）、`_base.py`、`_builders.py`（`make_unary_regcopy`）、`_nan.py`（`nan_min/nan_max`）。
- 源仓参考实现 / 测试基准：`torch.nn.functional.hardsigmoid`（`F.hardsigmoid(x.float()).to(x.dtype)`，见 `examples/TileOPs/tests/ops/test_hardsigmoid.py` 的 `ref_program`）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`F.hardsigmoid`），**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

HardsigmoidFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `hardsigmoid_fwd_kernel`。

### 1.2 功能描述

逐元素 HardSigmoid 激活：`y = clamp(x+3, 0, 6) * (1/6)`，等价 `relu6(x+3)/6`，用于轻量级 gating（SE-block 等），NaN 输入传播 NaN。

### 1.3 数学公式

$$
y = \operatorname{clamp}(x + 3,\ 0,\ 6) \cdot \frac{1}{6}, \quad \text{NaN} \to \text{NaN}
$$

### 1.4 算法描述

单步逐元素映射：展平为 1-D `N` 后，对每个元素独立计算 `(min(max(x+3, 0), 6)) * (1/6)`，中间 fp32。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**。本算法与 §1.6.0 调研选定算法一致（逐元素直接映射，无更优化简公式）。与源算法结构差异：仅并行组织（GPU `threads×npt` → NPU 一维 Kernel + block_size 折叠，§0.6 R1）与 NaN 处理（CUDA intrinsic → isnan+vselect 单点恢复，§0.6 R2）不同，计算表达式本身（逐元素 `x+3 → clamp → ×1/6`，fp32）完全一致。

### 1.5 数据流图

```
输入 input [*S] --host 展平--> x_flat (N,) --T.copy--> UB[x_ub] --vcast--> UB[x_f32]
  --vadd(+3)--> UB[buf1] --vclamp[0,6]--> UB[buf2] --vmul(×1/6)--> UB[buf1]
  --isnan+vselect(NaN 恢复)--> UB[buf1] --vcast--> UB[y_ub] --T.copy--> y_flat (N,) --host 回塑--> 输出 [*S]
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：调研四问 → 数学等价优化 → 向量化替代 → 向量化轴/布局。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：hardsigmoid 的算法族（所有能算出相同结果的算法结构）。源算法/输入公式只是基线候选之一。本算子为**单步逐元素激活**，按 algorithm-research.md §2 判为**轻量调研**（四问各一行结论 + 依据）。对照 algorithm-candidates.md `ALG-elementwise` 命中行（R1：公共子表达式消除/除法乘倒数/cast 链合并；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返——本算子无中间 GM 往返）。

**R1 等价化简公式候选**（基线在表；正式论证在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|--------|
| 0 | 基线（源公式） | `clamp(x+3,0,6) * (1/6)`，NaN 逐层传播 | — | — | ✅ 基线 |
| 1 | relu6 原生式 | `min(max(x+3,0),6) / 6` | 数学恒等（与基线仅除法/乘法、NaN 处理位置之差） | 无（同 3 op + 1 缩放） | ❌（无结构差异） |
| 2 | 除法转乘倒数 | `clamp(x+3,0,6) * (1/6)`（基线已含） | 容差内等价（1/6 倒数舍入，≤1 ulp） | N 次除法 → 1 次倒数 + N 次乘法（基线已采纳） | ✅（基线即此式） |
| 3 | 恒等变形 | `6 - min(6, max(0, 6-(x+3)))` | 数学恒等 | 无（op 数反而增多） | ❌（更差） |

**结论**：无更优于基线的化简公式。基线（`clamp(x+3,0,6)*(1/6)`）已是最小形式（1 add + 1 two-sided clamp + 1 mul，无公共子表达式、无超越函数、无 cast 链冗余）。

**R2 在线算法**：**无「在线/离线」之分**（也即天然单遍）——hardsigmoid 是逐元素映射，每个输出元素只依赖对应输入元素，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（algorithm-research.md §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 `clamp(x+3,0,6)*(1/6)` | 3N（add 1 + two-sided-clamp 1 + mul 1，manifest 口径） | 2N·elem_bytes（读 N + 写 N，无中间 GM 往返） | 1 | O(block_size)（UB 驻留，无 GM 中间缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1（relu6 原生式） | 3N | 2N·elem_bytes | 1 | O(block_size) | 完全并行 |

**结论**：候选间 FLOPs / 访存 / 遍数 / 中间缓冲四口径全同，基线即为最优（无中间 GM 往返、单遍、O(block) 驻留），无并列争议。

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水/融合 | 结论 |
|---------|-------------|---------|----------|---------|----------|------|
| 基线 | 逐元素 → Vector（vadd/vclamp/vmul），**不落 Cube**，与元素级算力正确匹配 | 工作集 = O(block_size) UB 驻留，~128KB < 192KB（§4.5） | 尾轴连续，block_size 为 32B 整数倍 ✓ | 循环边界 `ceildiv(N, block_size)` / `ceildiv(n_num, num_cores)` 均为编译期常量（N/block_size 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ✅ 采纳 |
| 候选 1 | 同上 | 同上 | 同上 | 同上 | 同上 | ✅（与基线无差异，并入基线） |

**调研结论**：选定算法族 = **逐元素直接映射（单遍，`clamp(x+3,0,6)*(1/6)`，fp32 中转）**。关键依据：R1 无更优化简公式（基线已最小）、R2 天然单遍、R3 四口径全同（带宽受限，访存主导）、R4 纯 Vector 亲和（UB 驻留、32B 对齐、静态边界）。与基线结构差异：**无**（调研确认源算法即最优，仅需按 §0.6 重设计其并行组织与 NaN 处理为 NPU 形态）。「无更优替代」调研范围：algorithm-candidates.md `ALG-elementwise` 命中行、恒等变形自查（R1 候选 3）、pattern-library `PL-1.6-copy-floor` / `PL-1.17-subfp32-fp32-transit` 同类实现、源码 §0.3/§0.4 算法与优化手段；结论为沿用基线（无算法级替代）。

**设计期估算**（D-2，轻量调研豁免容量项，流量项必算）：

```
估算下界(smoke-1m fp16): max(流量 4MB / ~1.36TB/s ≈ 2.9µs, 发射 ~8 op × 0.5µs ≈ 4µs) ≈ 4µs
  流量项: 2×1048576×2B = 4MB；MTE2 有效带宽按 CONST-mte2-degradation 1M 档 33.9GB/s/core × 40 核 ≈ 1.36TB/s（注：1M 档含 L2 驻留成分，实际更高）
  发射项: 向量链 vcast+vadd+vclamp(2)+vmul+isnan+vselect+vcast ≈ 8 op × 0.5µs/op（CONST-vector-launch-overhead）
  参考：同族 mish 实测 smoke-1m fp16 ≈ 7.0µs（Stage 4），本算子链更短（~6 op vs mish ~9 op）
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线）。采纳项经 verify_equiv.py 机器验证（结果表见下）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | NaN 传播化简 | `nan_min(nan_max(x+3,0),6) * (1/6)`（逐层 nan_min/nan_max 各一次 isnan+select，共 2 检测/2 select） | `clamp(x+3,0,6) * (1/6)` 后 `select(isnan(x), NaN, y)`（单点恢复，1 检测/1 select） | 数学恒等：NaN 唯一来源是输入 x（常数 3/0/6/1/6 均有限），逐层传播 ≡ 单点恢复。`x` 有限时两式同值；`x=NaN` 时两式均输出 NaN。位级一致（verify_equiv.py 项 1：0 ulp / 0 违反） | isnan 检测与 select 各减 1 次（2→1），向量链从 ~10 op 降至 ~8 op |
| 2 | 除法转乘倒数 | `clamp(x+3,0,6) / 6`（torch `F.hardsigmoid` 形式） | `clamp(x+3,0,6) * (1/6)`（`one_sixth` 预计算常数） | 容差内等价：`1/6` 在 fp64/fp32 中舍入，`×1/6` 与 `/6` 差 ≤1 ulp（verify_equiv.py 项 2：max 1 ulp / 0 违反，容差内 EQUIV_PASS）。此式与源公式一致（源已采纳） | N 次除法 → 1 次倒数（编译期常数）+ N 次乘法（除法在硬件上更昂贵） |

**机器验证结果表**（`verify_equiv.py`，fp64 参照 + 随机/角点 + 逐 dtype 舍入，已执行）：

| 采纳项 | max ulp 差 | 违反率 | 结论 |
|--------|-----------|--------|------|
| 1. NaN 传播化简（逐层 nan_min/nan_max → 单点 isnan 恢复） | 0.000 | 0 | EQUIV_PASS |
| 2. 除法转乘倒数（`/6` → `×(1/6)`） | 1.000 | 0 | EQUIV_PASS |

**优化结论**：采纳 2 项。优化后公式为：

$$
y = \begin{cases} \operatorname{NaN}, & x \text{ is NaN} \\ \operatorname{clamp}(x+3,\ 0,\ 6) \cdot (1/6), & \text{otherwise} \end{cases}
$$

（即 `clamp(x+3,0,6)*(1/6)` + 单点 NaN 恢复，fp32 中转）。无进一步优化空间：单次逐元素映射，无公共子表达式（`x+3` 只用一次）、无超越函数、无 cast 链冗余、无中间 GM 往返。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | 逐元素 `x+3` | `T.Parallel` 逐元素标量 `x_reg[i]+3` | `T.vadd(buf, 3.0, out)`（api-quickref.md / mish.py 佐证） | ✅ | — |
| 2 | 逐元素 clamp `min(max(t,0),6)` | 逐元素 `nan_min(nan_max(...))` | `T.vclamp(t, 0.0, 6.0, out)` 或 `T.vmax`+`T.vmin`（customize_npuir.py `npuir_clamp` 佐证） | ✅ | — |
| 3 | 逐元素 `×1/6` | 逐元素 `c2 * one_sixth` | `T.vmul(buf, one_sixth, out)`（api-quickref.md 佐证） | ✅ | — |
| 4 | dtype cast（fp16/bf16 ↔ fp32） | 隐式类型提升 | `T.vcast(src, dst, round_mode="rint")`（mish.py / PL-1.17 佐证） | ✅ | — |
| 5 | NaN 掩码 | 逐层 `isnan(a)||isnan(b)` | `T.isnan(x, dst)` + `T.vselect(mask, nan, y, y)`（attention `_gqa_prefill_fwd_kernel.py` vselect/vcmp 用法佐证） | ✅ | — |
| 6 | block 起始索引 `offset = block_id * block_size` | 标量乘加 | 无向量 API | ❌ | block 级任务映射索引，每 block 少量标量，非逐元素热点（quality-checklist 可接受理由类别①） |
| 7 | 尾块大小 `tail_size = T.min(block_size, remaining)` | 标量 min | 无（T.min 标量形） | ❌ | 依赖动态 shape 的边界处理，每 block 一次，非逐元素热点（可接受理由类别④；mish.py 同款尾块处理） |

**向量化结论**：逐元素计算（x+3 / clamp / ×1/6 / cast / NaN 恢复）已**全部向量化**（替代 API 均有 examples 与 docs 下的源码佐证：vadd/vclamp/vmul/vcast/isnan+vselect）。保留 2 处标量：block 起始索引（block 级任务映射）与尾块大小（动态边界），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴豁免说明**：本算子输入展平为 1-D `(N,)`，仅一个非 batch 维（展平轴），无 ≥2 个非 batch 维度可作向量化轴选择——逐元素激活在**唯一的展平轴**上向量化（Vector 单元 128bit：fp16/bf16 ×8、fp32 ×4），不存在广播轴、窗口轴、归约轴或重排布局候选。按 SKILL.md Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」。

- **向量化轴** = 展平轴（stride=1 连续，无累加链、无跨步系数）。
- **整除性**：block_size=8192，fp16/bf16 向量宽 ×8、fp32 ×4 均整除（8192 = 1024×8 = 2048×4）✓；尾块 `tail_size` 动态，仅 copy 有效元素，垃圾 lane 不写回（与 mish.py 同款）。
- **repack 代价**：无（输入即 1-D 连续，无需转置/重排）。
- **UB 容量影响**：驻留缓冲 = O(block_size)，预算见 §4.5。

**布局决策结论**：选定「I/O 原生 1-D 连续布局 + 展平轴向量化」，无 repack、无重排候选（单轴豁免）。该结论同步落入 §3.3 伪代码与 §6 循环结构（向量维 = 展平轴，buffer 均为 1-D）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（迁移默认）。

### 2.2 选型理由

纯 Vector 逐元素算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。Developer 模式由编译器自动管理同步与 buffer 分配，与 mish/lerp_tensor 同族 elementwise 先例一致（`mish_kernel/mish.py` 即 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`）。源算子 wrapper 亦声明 `ascend_mode = "Developer"`（`hardsigmoid.py` L89）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix API 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（vadd/vclamp/vmul/vcast/isnan/vselect），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 `set_flag/wait_flag`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_f32 = cast(x)`（fp16/bf16 时） | 输入升 fp32（fp32 输入跳过），对齐 torch golden 的 fp32 opmath |
| 2 | `t = x_f32 + 3.0` | 加法 |
| 3 | `c = clamp(t, 0.0, 6.0)` | 双侧 clamp（等价 `min(max(t,0),6)`） |
| 4 | `r = c * one_sixth`（`one_sixth = 1/6`） | 缩放（乘法倒数） |
| 5 | `y = select(isnan(x), NaN, r)` | NaN 单点恢复（§1.6.1 项 1） |
| 6 | `y_out = cast(y)`（fp16/bf16 时） | 降回输出 dtype |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 数据搬入 GM→UB | `T.copy(x[offset:offset+tail], x_ub[0:tail])` | 双显式 slice（TRAP-T-copy-region-semantics 推荐形态） | Developer |
| 1 | cast | `T.vcast(x_ub, x_f32_ub, round_mode="rint")` | fp16/bf16 → fp32 | Developer |
| 2 | +3 | `T.vadd(x_f32_ub, 3.0, buf1_ub)` | 标量广播 3.0 | Developer |
| 3 | clamp | `T.vclamp(buf1_ub, buf2_ub, 0.0, 6.0)` 或 `T.vmax(buf1_ub, 0.0, buf2_ub); T.vmin(buf2_ub, 6.0, buf2_ub)` | 标量广播 0.0/6.0 | Developer |
| 4 | ×1/6 | `T.vmul(buf2_ub, one_sixth, buf1_ub)` | 标量 `one_sixth`（fp32 常数） | Developer |
| 5 | NaN 恢复 | `T.isnan(x_f32_ub, nan_mask); T.vselect(nan_mask, nan_val, buf1_ub, buf1_ub)` | `nan_val` = NaN 广播 | Developer |
| 6 | cast 回 | `T.vcast(buf1_ub, y_ub, round_mode="rint")` | fp32 → fp16/bf16 | Developer |
| 7 | 数据搬出 UB→GM | `T.copy(y_ub[0:tail], y[offset:offset+tail])` | 双显式 slice | Developer |

> fp32 输入路径跳过步骤 1/6（无 vcast）。常数 `three/six/zero/one_sixth` 均 `T.cast(..., "float32")` 上抛 fp32，与源 `op_func` 一致。

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[1], target="npuir")
def hardsigmoid_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    n_num = T.ceildiv(N, block_size)
    # 物理核数 = NPUUtils.get().get_aicore_num() * 2（纯 Vector 翻倍，实查见 §5.5）
    if num_cores is None or num_cores <= 0 or num_cores >= n_num:
        num_cores = n_num
    tiles_per_core = T.ceildiv(n_num, num_cores)   # 编译期常量，静态边界

    one_sixth = T.cast(1.0 / 6.0, "float32")

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            # 1. 分配 UB buffer（v-prefix API 要求 UB 驻留）
            x_ub = T.alloc_ub((block_size,), dtype)
            x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
            buf1_ub = T.alloc_ub((block_size,), compute_dtype)
            buf2_ub = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)
            nan_mask = T.alloc_ub((block_size,), "int8")  # isnan 掩码

            # 2. persistent 串行 tile 循环（核内串行，边界静态）
            for i in T.serial(tiles_per_core):
                block_id = i * num_cores + cid
                if block_id < n_num:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # 3. 数据搬入（仅有效元素）
                    T.copy(x[offset:offset + tail_size], x_ub[0:tail_size])

                    # 4. cast 到 fp32（fp16/bf16）或 copy（fp32）
                    if dtype != compute_dtype:
                        T.vcast(x_ub, x_f32_ub, round_mode="rint")
                    else:
                        T.copy(x_ub, x_f32_ub)

                    # 5. 计算链：clamp(x+3, 0, 6) * (1/6)
                    T.vadd(x_f32_ub, 3.0, buf1_ub)      # t = x + 3
                    T.vclamp(buf1_ub, buf2_ub, 0.0, 6.0)  # c = clamp(t, 0, 6)
                    T.vmul(buf2_ub, one_sixth, buf1_ub)  # r = c * (1/6)

                    # 6. NaN 单点恢复（§1.6.1 项 1）
                    T.isnan(x_f32_ub, nan_mask)
                    T.vselect(nan_mask, nan_val_f32, buf1_ub, buf1_ub)

                    # 7. cast 回输出 dtype（fp16/bf16）或 copy（fp32）
                    if out_dtype != compute_dtype:
                        T.vcast(buf1_ub, y_ub, round_mode="rint")
                    else:
                        T.copy(buf1_ub, y_ub)

                    # 8. 数据搬出（仅有效元素）
                    T.copy(y_ub[0:tail_size], y[offset:offset + tail_size])

    return main
```

> 注：`T.Tensor((N,), dtype)` 用位置参数传 dtype（避免 `dtype=` 关键字 false alarm，design-template §3.3 约定）。`nan_val_f32` 为 NaN 常数广播（fp32）。fp32 路径（`dtype==compute_dtype and out_dtype==compute_dtype`）可走更简 buffer 链（复用 `x_f32_ub` 作 `x_ub`），Stage 3 实现时细化。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vadd/vmul/vdiv/vmax/vmin/vclamp` | `tilelang/language/customize_npuir.py`（`npuir_add/mul/div/max/min/clamp`）+ `.agents/skills/tilelang-vector-skill/references/api-quickref.md` | ✅ 已验证（源码读） |
| `T.vcast` | api-quickref.md + `mish_kernel/mish.py`（`round_mode="rint"`）+ PL-1.17-subfp32-fp32-transit | ✅ 已验证 |
| `T.isnan` | `tilelang.language` 导出（`dir(T)` 含 `isnan`） | ✅ 已验证 |
| `T.vselect/vcmp` | `multi_head_attention/_gqa_prefill_fwd_kernel.py`（`T.vselect(cond, a, b, out)` 用法） | ✅ 已验证 |
| `T.copy`（双显式 slice） | TRAP-T-copy-region-semantics（src/dst 双显式 slice 为推荐通用形态） | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | `mish_kernel/mish.py` / `lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | ✅ 已验证 |
| `T.alloc_ub` | `mish_kernel/mish.py` | ✅ 已验证 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 `T.Kernel(ceildiv(N, block_size))`） | 一维 `T.Kernel(num_cores, is_npu=True)`，无三维 |
| GPU 专用 API | Yes（`T.get_thread_binding` / `tl.max_nan` / `tl.min_nan` / `T.vectorized`/`T.unroll` CUDA 语义） | 全部改写为 NPU v-prefix API + UB 路径（§0.5/§0.6） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(N, block_size), threads=128)`（block+thread 两级） | 一维 `T.Kernel(num_cores, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(threads, npt)` 显式两级循环 | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | `T.alloc_fragment` / `T.alloc_local`（SMEM/寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| NaN min/max | `tl.max_nan/tl.min_nan`（CUDA intrinsic） | `isnan` + `vselect` 单点恢复 | §0.6 R2 |
| `threads×npt` | 512/1024 | `block_size=8192` 折叠（K11） | §5.2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 高度相似（同族 unary 激活迁移） | Developer 模式、`T.alloc_ub`、`T.copy` 尾块、v-prefix 链、fp32 中转、`T.Kernel(is_npu=True)` |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/mish.py` | 高度相似（Stage 4 调优） | persistent 核内串行 + 2-stage 流水 + block_size=8192 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 相似（同族 elementwise） | fp32 中转模式（PL-1.17）、NaN/Inf 传播、`T.min` 尾块 |

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
| `x_f32_ub` | `(block_size,)` | float32 | UB | fp32 中转输入（vcast 目标） |
| `buf1_ub` | `(block_size,)` | float32 | UB | 中间结果（t / r ping-pong） |
| `buf2_ub` | `(block_size,)` | float32 | UB | 中间结果（clamp 输出） |
| `nan_mask` | `(block_size,)` | int8/bool | UB | NaN 掩码（isnan 输出） |
| `y_ub` | `(block_size,)` | output dtype | UB | 输出 tile（UB→GM） |

### 4.4 内存搬运路径

```
GM[x] --T.copy--> UB[x_ub] --vcast--> UB[x_f32_ub]
  --vadd--> UB[buf1_ub] --vclamp--> UB[buf2_ub] --vmul--> UB[buf1_ub]
  --isnan+vselect--> UB[buf1_ub] --vcast--> UB[y_ub] --T.copy--> GM[y]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `x_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| `x_f32_ub` | (8192,) | fp32 | 32768 |
| `buf1_ub` | (8192,) | fp32 | 32768 |
| `buf2_ub` | (8192,) | fp32 | 32768 |
| `nan_mask` | (8192,) | int8 | 8192 |
| `y_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| **总计（fp16/bf16 路径）** | | | **≈ 139264（136KB）** |
| **总计（fp32 路径）** | | | **≈ 155648（152KB）** |

对照目标平台 UB 容量 **192KB**（`CONST-capacity-910B2C`：UB=192KB/AIV）。按 `CONST-capacity-910B2C` 的 ×1.10–1.12 上偏系数（auto-multi-buffer 实际需求 ≈ 手工预算 ×1.10–1.12）：fp32 路径 152KB ×1.12 ≈ 170KB < 192KB ✓。留有 ~12% 余量（逐 buffer 显式分配结构实测 ratio ≈0.981，见 constants.md 第六轮 r9 反例——显式 alloc 结构 ~12% 余量即足）。Stage 3 实现时可进一步复用 buffer（fp32 路径 `x_f32_ub` 复用 `x_ub`，省 32KB），预算更宽裕。

### 4.6 动态轴定义

无动态轴。`N` 为 JIT 编译期参数（`hardsigmoid_fwd_kernel(N, dtype)` 每 shape 一特化），`shape`（`T.int32`）为运行时传入的实际元素数（用于尾块 `remaining = shape - offset`，与 mish 同款）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素运算（add/clamp/mul/cast/select），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_size = 8192   # 同族 mish Stage 4 实测最优（BLOCK_SIZE_CAST/F32 = 8192）
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数
```

- **选择理由**：8192 为 mish 同族 elementwise Stage 4 实测最优 block_size（cast 路径与 fp32 路径均为 8192，见 `mish_kernel/perf_opt/mish.py`）；32B 对齐（8192 元素 × 2/4B 均为 32B 整数倍）；UB 预算 ~136–152KB < 192KB（§4.5）；块宽足够使向量链发射开销摊薄（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。

### 5.3 约束分析

- **对齐约束**：block_size=8192，fp16/bf16 尾轴 8192×2B=16KB、fp32 8192×4B=32KB，均 32B 对齐 ✓。
- **UB 容量**：~136–152KB < 192KB（§4.5，含 ×1.12 上偏后仍有余量）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾块**：`tail_size = T.min(block_size, remaining)`，仅 `T.copy` 有效元素进/出；v-prefix 对全 buffer 计算，尾部垃圾元素不写回（与 mish.py 同款）。
- **极小 shape**（如 mbv3-se-gate N=7680 < 8192）：`n_num=1`，`num_cores=1`，单核单 tile 处理，无 oversubscription。

### 5.5 分核策略（物理核数适配）⭐

- **物理核数（实查）**：`from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-12 实查，本机设备；与 CONST-aicore-910B2C 记录的 24 不同——该条目为特定 910B2C 设备，本环境设备返回 20，以实查为准）。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。查询代码与实际返回值记录于此（禁止文档假设/经验值替代）：

```python
from tilelang.utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 翻倍：40
```

- **逻辑核数**：`num_logical_kernels = ceil(N / block_size)`，随 workload 变化：
  - mbv3-se-gate（N=7680）：1；mbv3-se-gate-deep（N=30720）：4；smoke-1m（N=1048576）：128；fc-wide（N=8388608）：1024。

- **规模判定**：workload 跨越「无需适配」到「极大规模」多档。设计采用**统一 persistent 结构**一次覆盖全部档位（成熟 elementwise 形态，mish perf_opt + PL-1.6-copy-floor「grid-stride 48 核聚集窗口实测优于每核连续分块 +3.2%」）：

- **分核方案**：`num_cores = min(ceil(N/block_size), 40)`，核内 `T.serial(tiles_per_core)` 串行处理多个逻辑块（`tiles_per_core = T.ceildiv(n_num, num_cores)`，编译期常量，**循环边界静态**）：
  - **小规模（逻辑核数 ≤ 40，如 N ≤ 327680）**：`num_cores = n_num`，每核 ≤1 tile，`tiles_per_core=1`，无串行循环、无负载不均——结论「无需适配」（逻辑核数 ≤ 物理核数）。
  - **中等/极大规模（逻辑核数 > 40，如 smoke-1m 128、fc-wide 1024）**：固定启动内核数 = 40（= 物理向量核数），核内串行处理 `tiles_per_core = ceil(n_num/40)` 个 tile（smoke-1m：4；fc-wide：26），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均；循环边界 `tiles_per_core` 为编译期常量（N/block_size/40 均编译期已知），满足「核内串行边界静态」。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_cores, is_npu=True) as (cid, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(tiles_per_core)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（vadd/vclamp/vmul/vcast/isnan/vselect） | 块内逐元素由 Vector 单元并行（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_cores, is_npu=True) as (cid, _):
    # UB buffer 分配 ...
    for i in T.serial(tiles_per_core):          # 核内串行 tile 循环（静态边界）
        block_id = i * num_cores + cid
        if block_id < n_num:                    # 尾 tile guard
            # 装载 → v-prefix 计算链 → 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对 cast 路径采用 guard-free 2-stage `T.Pipelined(tiles_per_core_a, num_stages=2)` 双缓冲流水（mish perf_opt 形态），将向量链隐藏于 MTE2 窗口（PL-1.6-copy-floor：≥16M 档计算链被完全隐藏）。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 shape：`tail_size = T.min(block_size, remaining)`；`T.copy` 仅装载/写回 `[0:tail_size]` 有效元素；v-prefix 对全 block_size buffer 计算，垃圾尾部不写回（与 mish.py 同款，见 mish DESIGN.md 尾块处理节）。`block_id < n_num` guard 覆盖最后一波不完全 tile。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer）。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器自动管理 GM↔UB 搬运与向量计算间的依赖（`T.copy` 与后续 v-prefix op 之间的数据依赖由自动同步保证），无跨核协作（逐元素无跨元素依赖，各核独立）。

### 7.3 pass_configs 配置

Developer 模式默认配置，无需额外 `pass_configs`。环境变量 `TILELANG_ASCEND_MODE = "Developer"`（可选，默认即 Developer；wrapper 声明 `ascend_mode = "Developer"`）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（NaN → NaN、fp32 opmath、输出同 dtype），移植源仓参考实现（`torch.nn.functional.hardsigmoid`），**不复刻 §0.6 的 NPU 算法**。

```python
import torch
import torch.nn.functional as F

def golden_hardsigmoid(x):
    """Hardsigmoid 参考实现：y = clamp(x+3, 0, 6) / 6（NaN 传播，torch 语义）。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_hardsigmoid.py 的
    ref_program = F.hardsigmoid(x.float()).to(x.dtype)）。
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = F.hardsigmoid(x_f32)   # = clamp(x+3, 0, 6) / 6
    return y_f32.to(x_cpu.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |
| float32 | 1e-5 | 1e-5 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓测试 `_get_tolerances` 一致。kernel 用 `×(1/6)` vs golden 用 `/6` 差 ≤1 ulp，落在容差内；fp32 中转链对齐 torch fp32 opmath golden，差 ≤1 ulp fp16/bf16，见 PL-1.17-subfp32-fp32-transit。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`N=1048576`，`float32` / `float16` / `bfloat16` 三 dtype，`randn` 输入，vs golden（`F.hardsigmoid`）在对应容差下 `torch.testing.assert_close`。完整分层套件（L1 非整除/多 shape、L2 边界 size、Boundary 特殊值含 NaN/±inf）交由 `tilelang-op-develop` 规划，本设计不枚举。

**特别提醒（NaN 语义验证）**：L0 之外的 Boundary 用例必须覆盖 NaN 输入 → NaN 输出（§0.1 语义层要求），验证 §0.6 R2 的 isnan+vselect 单点恢复是否按预期工作，并据此确认 Ascend `vmax`/`vmin` 的实际 NaN 行为（§9 风险点）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| Ascend `vmax`/`vmin` NaN 语义未文档化 | Ascend C Max/Min 文档（CANN 9.0.0-beta1）「约束说明」仅载明地址对齐/重叠，无特殊值处理说明。本设计用 `isnan`+`vselect` 单点恢复兜底，保证语义正确；Stage 3 实测确认硬件行为后，若 vmax/vmin 本身传播 NaN，可移除该步（可选优化，不影响正确性） |
| UB 预算 | ~136–152KB < 192KB，含 ×1.12 上偏系数仍有 ~12% 余量；Stage 3 可复用 buffer 进一步放宽 |
| 分核 | 物理核数实查 20（×2=40），与 CONST-aicore-910B2C（24）不一致——本环境设备返回 20，以实查为准 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_size 过大（>8192×fp32 全 buffer 链） | 编译失败（`ub overflow`） | 维持 block_size=8192；复用 buffer 减少驻留 |
| NaN 未传播 | 直接 `vclamp` 而 vmax/vmin 丢 NaN | 输出 0/6 而非 NaN，违反语义 | §0.6 R2 的 isnan+vselect 单点恢复 |
| 尾块越界 | `T.copy` 用 base+size 形态写尾块 | MTE 越界写坏相邻 tensor | 双显式 slice `T.copy(src[0:tail], dst[0:tail])`（TRAP-T-copy-region-semantics） |
| `dtype=` 关键字 false alarm | `T.Tensor((N,), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（design-template §3.3） |

### 9.3 特殊场景处理

- **非整除 shape**（N=7680、奇数 N 等）：尾块 `tail_size` + `block_id < n_num` guard（§6.4）。
- **极小 shape**（N < block_size）：`n_num=1`，单核单 tile。
- **NaN/±inf 输入**：NaN → NaN（§0.6 R2）；±inf → clamp 收敛到 0/6，无上溢。
- **混合精度**：fp16/bf16 走 fp32 中转（vcast），fp32 直算；输出 dtype 与输入一致。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/hardsigmoid/hardsigmoid_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，结果内嵌）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成 | §1.6.1 等价性验证（2 项 EQUIV_PASS） |
| `hardsigmoid_fwd_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 测试 |
| `test_hardsigmoid.py` | ⬜ 待实现（Stage 3） | 精度测试 |

### 10.3 命名规范

- 项目目录名：`hardsigmoid`；算子目录名：`hardsigmoid_fwd_kernel`；接口函数：`hardsigmoid_fwd_kernel`（manifest 键 `HardsigmoidFwdOp`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`hardsigmoid_fwd_kernel`）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）
