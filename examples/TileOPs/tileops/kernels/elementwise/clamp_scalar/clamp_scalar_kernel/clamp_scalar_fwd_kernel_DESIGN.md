# ClampScalarFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/clamp.py` 的 `ClampFwdKernel` + `_base.py` 的 `ScalarParamUnaryKernel`/`UnaryKernel` register_copy 策略 + `_builders.py` 的 `make_unary_regcopy` + `_nan.py` 的 `nan_max`/`nan_min`）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `clamp_scalar_fwd_kernel`，`target="npuir"`。
> 本设计按 skill `tilelang-op-design` 三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出；programming_mode = `developer`，target = `npuir`。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 skill `tilelang-op-design` 的 references/migration-analysis.md）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：

$$
y_i = \operatorname{clamp}(x_i,\ \text{min},\ \text{max}) = \min\bigl(\max(x_i,\ \text{min}),\ \text{max}\bigr)
$$

其中 `min` / `max` 为 `Number | None`（keyword-only，默认 `null`），**至少一个 present**；absent 的一侧不施加（即 `present(min)` 且 `present(max)`=clamp；仅 min=clamp_min；仅 max=clamp_max）。与 `torch.clamp(input, min=None, max=None)` 一致。

源 op body（`ClampFwdKernel._make_op_func`，clamp.py L44-55）：

```python
def op_func(x):
    wide = T.cast(x, "float32")
    if min_val is not None:
        wide = nan_max(wide, T.cast(min_val, "float32"))
    if max_val is not None:
        wide = nan_min(wide, T.cast(max_val, "float32"))
    return T.Cast(x.dtype, wide)          # fp32 中间计算，cast 回输入 dtype
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | 逐元素 `clamp(x, min, max)`；absent bound 不施加；**fp32 中间计算后按输入 dtype 截断** |
| 规约语义 | **无规约**，纯逐元素，无跨元素依赖、无累加顺序问题 |
| dtype 语义 | 输入 ∈ {float16, bfloat16, float32}；输出 `same_as(input)`；`wide = T.cast(x,"float32")` → **fp32 中转**；bound 亦 `T.cast(bound,"float32")`；末尾 `T.Cast(x.dtype, wide)` 截断回输入 dtype |
| 边界语义（NaN） | **NaN 传播**：源 `nan_max`/`nan_min`（`_nan.py`）为 NaN-propagating —— `nan_max(a,b)=Select(isnan(a) or isnan(b), NaN, max(a,b))`，故输入 x 为 NaN **或** present bound 为 NaN → 输出 NaN（源 docstring「A NaN input or bound gives NaN, as in torch.clamp」）。torch 2.7.1 实测：`clamp(x, min=NaN, ...)` / `clamp(x, ..., max=NaN)` 对**整个输出**返回 NaN（本机 `torch.clamp` 验证） |
| 边界语义（±inf） | `clamp(+inf, min, max) = max`、`clamp(-inf, min, max) = min`（收敛，无溢出）；bound=±inf 为恒等侧（`width` 表示需 present-val 有限，见 manifest shape_rules `representable(min.value, T)`） |
| 边界语义（min > max） | 与 torch 一致（`min(max(x,min),max)` 组合逻辑，结果 = max）；manifest 与 workload 均 min<max |
| 空维度 / 非整除 | 输入展平为 1-D `N = prod(S)`（`N` 为运行时 numel），非整除 shape（如 N=4_000_000）由尾块处理 |
| **optional min/max 契约** | `min`/`max` 为**构造期标量参数**（非常量 tensor）：GPU 侧经 `_param_key` 把 `(min_val, max_val)` 编入 op body 名（每组合一特化，lru_cache）；absent 侧在 trace 时整段跳过。NPU 侧同一契约：`min`/`max` 作为 Op 参数（kw_only，default null）→ kernel factory → **作为标量操作数传给 v-prefix 向量 API**（vmax/vmin/vclamp 的标量广播），absent 侧同样 trace-time 跳过。**开发可见契约差异**：GPU 把 bound 烘进闭包（`T.cast(min_val,"float32")` 作为算术常量），NPU 把它作为标量 operand 传入向量指令——两者均为编译期常量特化，数值语义一致；但 NPU 需为 absent 侧选择单边 API（vmax 或 vmin），且 NaN bound 需编译期退化分支（见 §0.6 R3）。 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.clamp(x.float(), min, max).to(dtype)`，NaN → NaN、fp32 opmath、输出同 dtype），**不复刻 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input` | `[*S]`（任意，展平为 1-D `N=prod(S)`） | `float16 \| bfloat16 \| float32` | 输入张量；kernel 收到 `flat(x)` 的 1-D 视图（Op 层 `input.contiguous().reshape(-1)`） |
| `min` | 标量（`Number`），非 tensor | — | 下界（可选，构造期参数，kw_only） |
| `max` | 标量（`Number`），非 tensor | — | 上界（可选，构造期参数，kw_only） |
| `output` | 与 input 相同 shape | 与 input 相同 dtype | 输出张量；Op 层 `reshape(orig_shape)` 回塑 |

> 本算子（`ClampScalarFwdOp`）与 Tensor-bound 的 `ClampTensorFwdOp` 是**两个 manifest 条目**：前者的 bound 是 op 参数（Number），后者是 tensor 输入。本设计只迁移 `ClampScalarFwdOp`。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host 展平：`flat(x)` → 1-D `(N,)` | `input [*S]` | `x_flat (N,)` | 前置（host 视图操作，不改数据） |
| 2 | 分块装载：`T.copy(x[bx*bs:(bx+1)*bs], x_reg)`（fragment） | `x_flat` | `x_reg (bs,)` fragment | 数据搬入 |
| 3 | 升精度：`wide = T.cast(x, "float32")` | `x_reg` | `wide`（fp32） | fp32 中间域 |
| 4 | 下界（若 present(min)）：`wide = nan_max(wide, T.cast(min,"float32"))` | `wide`, `min` | `wide` | `max(x, min)` + NaN 传播 |
| 5 | 上界（若 present(max)）：`wide = nan_min(wide, T.cast(max,"float32"))` | `wide`, `max` | `wide` | `min(·, max)` + NaN 传播 |
| 6 | 降精度：`y = T.Cast(x.dtype, wide)` | `wide` | `y` | cast 回输入 dtype |
| 7 | 分块写回：`T.copy(y_reg, y[bx*bs:(bx+1)*bs])` | `y_reg` | `y_flat` | 数据搬出 |
| 8 | host 回塑：`result.reshape(x.shape)` | `y_flat` | `output [*S]` | 后置（host 视图操作） |

absent bound：步骤 4（或 5）整段不生成——`_make_op_func` 的 `if min_val is not None` / `if max_val is not None` 为 trace-time 分支。

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x] --T.copy--> fragment/x_reg(寄存器) --cast+op_func(nan_max/nan_min)--> fragment/y_reg --T.copy--> GM[y]
```

`register_copy` 策略（`make_unary_regcopy`，`_builders.py` L85-121）：`block_size = threads × npt`，分块 `T.copy` 装载到 fragment，`T.Parallel(threads, npt)` 逐元素计算，`T.copy` 写回。整数倍且 result 变窄时走 per-thread 连续 run（`T.vectorized` 装载/写回 + `T.unroll` 计算），否则走 fragment 全块路径。**本算子输入输出同宽（不 narrow），走 fragment 全块路径**（`contiguous = _narrows(dtype,out_dtype) and N % block_size == 0`，clamp 下 `_narrows(...,...)` 为 False）。

**循环与并行结构**：grid 一维 `T.Kernel(ceildiv(N, block_size))`；block 内 `threads × npt` 两级并行（`T.Parallel(threads, npt)`）。`ScalarParamUnaryKernel.DEFAULT_THREADS = 256`，`npt` 来自 `default_launch_config`（16 bytes/thread：fp32 npt=4，fp16/bf16 npt=8）→ `block_size = 256×4 = 1024`（fp32）/ `256×8 = 2048`（fp16/bf16）。

**host 侧逻辑**：Op 层 `ClampScalarFwdOp._eager_forward`：`flat = input.contiguous().reshape(-1)` → `kernel(flat)` → `reshape(orig_shape)`；Kernel 层 `forward`：`flat(x)` → `_compiled_fn(...)` → `_restore_output_dtype(result).reshape(x.shape)`。全部为 stride/shape 元数据视图操作，**不改输入张量真实内容**（满足 ascend-constraints.md §4）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | register_copy（fragment 分块装载/写回） | 减少 GM 重复访问、放大访存粒度 | 分块 `T.copy` 到 fragment，逐元素计算后整块写回 | CUDA SMEM/寄存器 | 硬件强相关 |
| 2 | coalesced 向量化访问 | 尾轴连续读写打满带宽 | `T.vectorized(npt)` 连续 run 装载/写回 | memory coalescing | 硬件强相关（NPU 有 MTE 等价） |
| 3 | fp32 中间计算 + 末尾 cast | 窄 dtype 下保持比较精度 | `T.cast(x,"float32")` 上抛，`T.Cast(dtype, wide)` 截断 | 无（纯算法/精度策略） | 可移植 |
| 4 | NaN 传播 min/max（`nan_max`/`nan_min`） | 匹配 torch 的 NaN→NaN 语义 | fp16/bf16 用 `tl.max_nan/tl.min_nan`（CUDA `__hmax_nan/__hmin_nan` 单指令）；fp32 用 `isnan(a)||isnan(b)` 守卫 + `Select` 回填 NaN（`_propagate_nan`） | CUDA intrinsic / tirx.Select | 硬件强相关 |
| 5 | `threads × npt` 两级并行执行模型 | 每线程处理 npt 元素放大吞吐 | CUDA thread/warp 组织 | CUDA 线程模型 | 硬件强相关 |
| 6 | bound absent 跳过（trace-time 分支） | 单边 clamp 省一步 | Python `if min_val is not None` 在 trace 时决定 op body | 无（编译期特化） | 可移植 |
| 7 | `_param_key` lru_cache 特化 | 每个 (min,max) 组合独立编译缓存 | `@lru_cache` + op body 名含参数 | 无（host 缓存机制） | 可移植 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。识别不出来 ≠ 不存在：以上 7 项为源码显式优化/机制，全部纳入 §0.5 判定。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `clamp(x,min,max)`、NaN→NaN、fp32 中转、输出同 dtype | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| bound absent trace-time 跳过（#6） | 算法 | 无 | — | **保留** | 编译期特化，NPU 侧同款（absent 侧不生成 vmax/vmin） |
| fp32 中间计算 + 末尾 cast（#3） | 算法 | 无 | — | **保留** | 对齐 torch golden（fp32 opmath）；PL-1.17-subfp32-fp32-transit 实证（bf16 v-prefix 缺失 + fp16 golden 对齐双触发） |
| `threads × npt` 两级并行 + `T.Parallel(threads,npt)`（#5） | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel） | **重新设计** | → §0.6 R1（block_size 折叠 + 一维 `T.Kernel` + UB 向量化，K11 适配） |
| fragment 分块装载/写回（SMEM/寄存器，#1） | 优化 | SMEM/寄存器 | 有（UB） | **等价替换** | `T.alloc_ub` + `T.copy`（GM→UB→GM），ascend-constraints.md §1 / decision-tree.md §3 |
| coalesced 向量化访问（`T.vectorized`，#2） | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | `T.copy` 连续 slice 搬运（TRAP-T-copy-region-semantics 双显式 slice 形态） |
| NaN 传播 min/max（CUDA `tl.max_nan/tl.min_nan` intrinsic + 逐层 isnan+Select，#4） | 优化 | CUDA intrinsic | 无同款 NaN-propagating intrinsic（Ascend vmax/vmin NaN 语义未文档化，PL-1.11 P8 指示为 NaN-dropping） | **重新设计** | → §0.6 R2（普通 vclamp/vmax/vmin + `vcmp(x,x,"ne")` + `vselect` 单点恢复） |
| NaN bound（present bound 为 NaN → torch 全 NaN） | 算法/边界 | 无 | 无直接等价（vmax/vmin 丢 NaN） | **重新设计** | → §0.6 R3（trace-time 退化分支：任一 present bound 为 NaN → 全 NaN 输出） |
| `_param_key` lru_cache 特化（#7） | 优化 | 无 | 有 | **等价替换** | factory 层 `@functools.lru_cache(maxsize=32)`（NPU 侧同款，explicit-alloc 可见性见 CONST-capacity-910B2C） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目（SMEM→UB、warp→Vector、CUDA intrinsic→无等价重设计）+ examples/ 佐证（hardsigmoid 同族迁移）+ docs 条目（`docs/Tilelang.language/` v-prefix API）。舍弃项：无（未发现 NPU 上无意义/有害条目；`T.vectorized`/`T.Parallel` 的 CUDA 语义不迁移，但被等价替换而非舍弃）。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（`threads × npt` → 一维 Kernel + UB 向量化）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=256)` + `T.Parallel(threads, npt)` 两级并行；`block_size = threads × npt` = 1024(fp32)/2048(fp16/bf16)；fragment 装载/写回（`T.vectorized` + `T.unroll`）。
- **NPU 新算法**：一维 `T.Kernel(num_cores, is_npu=True) as (cid, _)`；GPU 的 `threads×npt` 折叠为单一 `block_size`（K11 适配，ascend-constraints.md 已知限制「仅支持一维 Kernel」）；元素级并行交给 Vector 单元（v-prefix 逐元素 API 对整块 UB buffer 计算，天然向量化，无需显式 `T.Parallel`）；分块装载/写回用 `T.copy`（GM→UB→GM，32B 对齐连续 slice）；极大规模（256M → 32768 逻辑块）用核内 `T.serial` persistent 串行（§5.5）。`block_size` 取值见 §5.2（8192，承接 hardsigmoid/mish 同族元素级实测）。
- **语义保持论证**：逐元素映射不变（每元素独立计算，无跨元素依赖、无规约），`threads/npt` 到 `block_size/核` 的任务映射变化不改变任何输出值——每个 `out[i]` 的计算表达式与求值顺序与源码逐步一致（`cast→clamp→cast`，fp32 域）。fp32 中转与源舍入路径一致。边界语义：尾块（非整除）仅 copy 有效元素，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 fragment 尾块行为等价。

**重设计项 R2：NaN 传播语义重设计（CUDA `tl.max_nan/tl.min_nan` + 逐层 isnan/Select → NPU 普通 clamp + 单点恢复）**

- **源方案**：fp16/bf16 用 `tl.max_nan`/`tl.min_nan`（CUDA `__hmax_nan`/`__hmin_nan`，NaN 传播单指令）；fp32 用 `isnan(a)||isnan(b)` 守卫 + `Select` 回填 canonical NaN（`_propagate_nan`）。每次 min/max 各一次 isnan 检测 + select（两个 bound 共 2 检测/2 select）。
- **NPU 新算法**：Ascend `vmax`/`vmin`（`tl.npuir_max/min`，`vclamp` 内部即 vmax+vmin）为 **NaN-dropping**（fmax/fmin 语义：丢 NaN；PL-1.11-causal-mask-scalartrap P8 探针结论「vmax/vmin 钳位只治 ±Inf」；Ascend C Max/Min 文档「约束说明」仅载地址对齐/重叠，无特殊值处理说明）。为保持 NaN→NaN，采用**单点恢复**：普通 clamp（vclamp/vmax/vmin，丢 NaN）后，用 `T.vcmp(x32, x32, nan_mask, "ne")` 自比较得 NaN 掩码（IEEE 无序比较：NaN≠NaN→true，±inf/有限→false）+ `T.vselect(nan_mask, x32, buf1, buf2)` **回选原始 x32**（其 NaN 位模式直接成为输出 NaN），**不构造 canonical NaN 常数**（规避 VP-2026-0142 的 NaN FloatImm cache-metadata 陷阱）。
- **语义保持论证**：数学恒等——finite bound 下 NaN 唯一来源是输入 x；`x` 有限时两式给出完全相同的 `clamp(x,min,max)`；`x=NaN` 时源式经逐层 nan_max/nan_min 输出 NaN、新式经单点 vselect 回选 x 输出 NaN，NaN 语义一致。逐层传播（源）≡ 单点恢复（新）。`vcmp(x,x,"ne")` 的 NaN 检测依赖 **IEEE 无序比较语义**（掩码仅命中 NaN，不误伤 ±inf——区别于 `0·x` 算术掩码在 ±inf 位产生 NaN 的缺陷）；该语义已由 `verify_equiv.py` 在数学层机器验证（§1.6.1，0 ulp / 0 违反 / EQUIV_PASS）；Ascend `npuir_cmp "ne"` 硬件是否按 IEEE 无序实现需 Stage 3 NaN Boundary 实测标定（§9.1，若证伪回退标量 `T.isnan` 兜底）。

**重设计项 R3：NaN-bound 退化分支（torch 全 NaN 语义保持）**

- **源方案**：present bound 为 NaN 时，源 `nan_max(x, NaN)`/`nan_min(x, NaN)` 返回 NaN（`_propagate_nan` 的 `isnan(b)` 分支）→ 整个输出 NaN；torch 2.7.1 同（`clamp(x, min=NaN)` / `clamp(x, max=NaN)` 全 NaN，本机实测）。
- **NPU 新算法**：`min_val`/`max_val` 为**构造期 Python 标量**，NaN bound 是完全确定的编译期退化特化。在 factory 内以 `math.isnan(b)`（trace-time）检测：任一 present bound 为 NaN → op body 短路为「全 NaN 输出」，不经 vmax/vmin（因其丢 NaN 会给出错误结果）。全 NaN 的生成避免 NaN 常数：`T.vmul(x32, 0.0, buf1)`（有限 x→0，±inf·0→NaN，NaN→NaN）后 `T.vdiv(buf1, 0.0, buf2)`（0/0→NaN，NaN/0→NaN）→ 全 NaN；仅用有限常数 0.0（仍规避 VP-2026-0142）。manifest 与 workload 均为有限 bound（-0.5/0.5），此分支不被 dispatch，仅保证契约完备。
- **语义保持论证**：NaN bound → torch 全 NaN，新分支输出全 NaN，位/语义一致（`verify_equiv.py` 覆盖 min=NaN / max=NaN 两组合，3 dtype，EQUIV_PASS）；finite bound 时该分支不触发，退化为 R2 路径。

> §0.5 判定为「重新设计」的条目（R1/R2/R3）已全部重设计；其余为保留/等价替换。**源优化手段（§0.4）意图承接**：#1 register_copy 与 #2 coalesced 的「放大访存粒度、整块搬移」意图由 UB `T.copy` 分块搬移承接；#5 `threads×npt` 的「元素级并行放大吞吐」意图由 Vector 单元整块 v-prefix 计算承接；#4 NaN 传播意图由 R2 单点恢复 + R3 退化分支承接；#3 fp32 中转与 #6 absent 跳过为算法层直接保留；#7 lru_cache 特化由 factory `@lru_cache` 等价替换承接。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/clamp.py`（`ClampFwdKernel._make_op_func`）；`_base.py`（`ScalarParamUnaryKernel`/`UnaryKernel`/`_build_kernel`）、`_builders.py`（`make_unary_regcopy`）、`_nan.py`（`nan_max`/`nan_min`）、`ops/elementwise/clamp.py`（`ClampScalarFwdOp`）。
- 源 manifest 条目：`/mnt/workspace/TileOPs-main/src/tileops/manifest/spec/elementwise_unary_activation.yaml`（`ClampScalarFwdOp`，~L323）。
- NPU 侧提取参考（合并 factory）：`examples/TileOPs/tileops/kernels/elementwise/clamp_scalar/_clamp_scalar_fwd_kernels.py`（factory `clamp_scalar_fwd_kernel(N, dtype, output_dtype=None, min_val=None, max_val=None, threads=128, num_per_thread=8)`）。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_clamp_scalar.py` 的 `ref_program = torch.clamp(x.float(), min, max).to(x.dtype)`；tolerances 见同文件 `_get_tolerances`。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现，**不复刻 NPU 算法**（保证验证独立性）。

---
## 1. 概述

### 1.1 算子名称

ClampScalarFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `clamp_scalar_fwd_kernel`。

### 1.2 功能描述

逐元素标量边界裁剪：`y = clamp(x, min, max)`，`min`/`max` 为 `Number | None`（至少一个 present）；fp32 中间计算后 cast 回输入 dtype；NaN 输入或 bound 传播 NaN（同 `torch.clamp`）。用于数值范围裁剪/激活前裁剪。

### 1.3 数学公式

$$
y_i =
\begin{cases}
\mathrm{NaN}, & x_i=\mathrm{NaN}\ \text{or present bound}=\mathrm{NaN} \\
\min\bigl(\max(x_i,\ \text{min}),\ \text{max}\bigr), & \text{otherwise}
\end{cases}
$$

（absent bound 侧不施加；中间 fp32 计算，输出 dtype 与输入相同。）

### 1.4 算法描述

单步逐元素映射：展平为 1-D `N` 后，对每个元素独立计算 `clamp(x, min, max)`，中间 fp32。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**。本算法与 §1.6.0 调研选定算法一致（逐元素直接映射，无更优化简公式）。与源算法结构差异：仅并行组织（GPU `threads×npt` → NPU 一维 Kernel + `block_size` 折叠，§0.6 R1）与 NaN 处理（CUDA NaN-propagating intrinsic → NPU 普通 clamp + 单点恢复 + NaN-bound 退化分支，§0.6 R2/R3）不同；计算表达式本身（逐元素 `cast→clamp→cast`，fp32 域）与源完全一致。

### 1.5 数据流图

```
输入 input [*S] --host 展平--> x_flat (N,) --T.copy--> UB[x_ub] --vcast--> UB[x32]
  --clamp(vclamp | vmax | vmin, 标量 bound)--> UB[buf1]
  --vcmp(x32,x32,"ne")+vselect(回选 x32)--> UB[buf2] --vcast--> UB[y_ub] --T.copy--> y_flat (N,) --host 回塑--> 输出 [*S]
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：调研四问 → 数学等价优化 → 向量化替代 → 向量化轴/布局。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：clamp（标量边界）的算法族——所有能算出相同结果的算法结构。源算法/输入公式只是基线候选之一。本算子为**单步逐元素**，按 algorithm-research.md §2 判为**轻量调研**（四问各一行结论 + 依据），但四问逐问回答。对照 algorithm-candidates.md `ALG-elementwise` 命中行（R1：公共子表达式消除 / 除法转乘倒数 / cast 链合并；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返——本算子无中间 GM 往返）。

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|--------|
| 0 | 基线（源公式） | `nan_min(nan_max(cast(x,f32), cast(min,f32)), cast(max,f32))` → `cast(dtype)`，NaN 逐层传播 | — | — | ✅ 基线 |
| 1 | 单点 NaN 恢复 | 普通 `clamp(x,min,max)`（丢 NaN）+ `select(isnan(x), x, clamp)` | NaN 唯一来源为 x（finite bound）⇒ 与基线数学恒等 | 少 1 次 NaN 检测 + 1 次 select（两 bound → 单点）；免 canonical NaN 常数 | ✅ |
| 2 | 中点/绝对值恒等重排 | `clamp(x,m,M)=(|x-m|−|x-M|)/2+(m+M)/2` 或 `m+min(max(x−m,0),M−m)` | 数学恒等 | **负收益**（op 数增加：多 abs/sub/add） | ❌（更差） |
| 3 | 单边降代换 | 仅 min：`max(x,min)`；仅 max：`min(x,max)` | 数学恒等（absent 侧不施加） | 两 bound→单 op；基线同款 trace-time 跳过 | ✅（基线已含） |

**R2 在线算法**：**无「在线/离线」之分**（也即天然单遍）——clamp 是逐元素映射，每个输出元素只依赖对应输入元素，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（algorithm-research.md §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 `clamp`（fp32 中转，NaN 逐层） | 1·N（compare-and-select，manifest 口径 `flops: prod(S)`） | 2·N·elem_bytes（读 N + 写 N，无中间 GM 往返） | 1 | O(block_size)（UB 驻留，无 GM 中间缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1（单点恢复） | 1·N（FLOPs 口径同；实际 v-op 数更少） | 2·N·elem_bytes | 1 | O(block_size) | 完全并行 |
| 候选 2（中点/绝对值） | ~5·N | 2·N·elem_bytes | 1 | O(block_size) | 完全并行 |

**结论**：候选间访存 / 遍数 / 中间缓冲四口径全同（带宽受限，访存主导），基线即为最优访存；候选 2 op 数更多（淘汰），候选 1 在不增加访存的前提下减少 v-op 数与 NaN 检测数（采纳）。无并列争议（候选 1 严格不劣于基线）。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线 | 逐元素 → Vector（vcast/vmax/vmin/vclamp/vcmp/vselect），**不落 Cube**，与元素级算力正确匹配 | 工作集 = O(block_size) UB 驻留，~136–168KB < 192KB（§4.5） | 尾轴连续，block_size 为 32B 整数倍 ✓ | `ceildiv(N, block_size)` / `ceildiv(n_num, num_cores)` 均编译期常量（N/block_size 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ✅ 采纳 |
| 候选 1（单点恢复） | 同上（v-op 更少） | 同上 | 同上 | 同上 | 同上 | ✅ 采纳（并入选定算法） |
| 候选 2（中点/绝对值） | Vector（但多 3+ v-op） | 同上 | 同上 | 同上 | 同上 | ❌（发射开销 ×CONST-vector-launch-overhead，无收益） |

**调研结论**：选定算法族 = **逐元素直接映射（单遍，fp32 中转，普通 clamp + 单点 NaN 恢复 + NaN-bound 退化分支）**。关键依据：R1 无更优化简公式（基线已最小；单点恢复为唯一正收益项，见 §1.6.1 项 1）、R2 天然单遍、R3 四口径基线最优（带宽受限，访存主导）、R4 纯 Vector 亲和（UB 驻留、32B 对齐、静态边界）。与基线结构差异：**计算表达式不变**（`cast→clamp→cast`），仅 NaN 处理（逐层 → 单点恢复 + 退化分支）与并行组织（§0.6 R1）不同。「无更优替代」调研范围：algorithm-candidates.md `ALG-elementwise` 命中行、恒等变形自查（R1 候选 2/3）、pattern-library `PL-1.6-copy-floor` / `PL-1.17-subfp32-fp32-transit` / `CONST-vcmp-ne-nan-unordered` 同类实现、源码 §0.3/§0.4 算法与优化手段；结论为沿用基线 + 单点恢复。

**设计期估算**（D-2，单步逐元素，容量项豁免，流量项必算；常数引用 pattern-library/constants.md，版本戳：`tilelang 0.1.2+ed787bb`（2026-09-07 build）/ CANN 8.5.0 / Ascend910B2C）：

```
估算下界(256M fp16 both-bounds) = max(流量 852µs, 发射 ~3.5µs) ≈ 852µs
  流量项: 2 × 268435456 × 2B = 1073741824B ÷ ~1.26TB/s ≈ 852µs
          （CONST-mte2-degradation，256M 档 3:1 R/W 混合流量地板 ~1.26TB/s；
           外推前核对：该曲线绑定 4-set 轮换 fp16 访问）
  发射项: 链长 ~7 v-op × 0.5µs/op ≈ 3.5µs（CONST-vector-launch-overhead；
          vcast+vclamp(2)+vcmp+vselect+vcast = 6–7 op，按单波固定发射口径）
  容量项: 豁免（单步逐元素，无跨引擎中间矩阵往返，CONST-store-fixpipe-gm-only 不适用）
估算下界(16M fp32 both-bounds) = max(流量 111µs, 发射 ~3µs) ≈ 111µs
  流量项: 2 × 16777216 × 4B = 134217728B ÷ ~1.208TB/s ≈ 111µs
          （CONST-mte2-degradation，16M 档 30.2GB/s/core × 40 核）
备注: 若发射成本不可跨 tile 重叠（串行），32768 逻辑 tile 的发射项可达 ~2.9ms > 流量项——
      block_size 增大 / T.Pipelined 为 Stage 4 首要调优轴（§6.3 / §9.1）。
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线 + 单点恢复）。采纳项经 verify_equiv.py 机器验证（结果表见下，脚本已执行）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | NaN 传播化简（单点恢复） | `nan_min(nan_max(wide,min),max)`（逐层：两 bound 共 2 次 `isnan(a)||isnan(b)` 检测 + 2 次 Select 回填 NaN） | 普通 `clamp(x,min,max)`（vclamp/vmax/vmin，丢 NaN）后 `select(isnan(x), x, clamp)`（1 次检测 + 1 次 select；NPU 检测 = `vcmp(x,x,"ne")` 自比较，见 §1.6.2 #5） | 数学恒等：finite bound 下 NaN 唯一来源为输入 x（bound 为有限常数），逐层传播 ≡ 单点恢复。`x` 有限 → 两式同值；`x=NaN` → 两式均 NaN。位/语义一致。NaN-bound 情形由项 2 退化分支覆盖（verify_equiv.py：all finite+Nan-bound 组合 0 ulp / 0 违反） | NaN 检测与 select 各由 2 次降为 1 次；v-op 链由 ~9 降至 ~7（省 1 vcmp + 1 vselect） |
| 2 | NaN-bound 全 NaN 退化分支 | `nan_max(x, NaN)=NaN` / `nan_min(x, NaN)=NaN`（逐层分支，源 & torch 均全 NaN） | trace-time `math.isnan(bound)` 检测 → 全 NaN 输出分支（`vmul(x,0.0)` + `vdiv(·,0.0)` 构造，仅用有限常数） | 语义恒等：present bound=NaN 时 torch.clamp / 源式均对整个输出返回 NaN（torch 2.7.1 实测）；退化分支输出全 NaN 位/语义一致。finite bound 时不触发 | 正确性保持分支（NPU vmax/vmin 丢 NaN，无此分支则 NaN bound 结果错误）；仅用有限常数，规避 VP-2026-0142 NaN 常数 cache 陷阱 |
| 3 | NaN 值来源无常数化 | 回填 canonical NaN：`T.cast(float("nan"),"float32")` 经 `vbrc` 广播 | 直接 `vselect(mask, x32, buf1, buf2)` 回选输入 x32（其 NaN 位模式即输出 NaN） | NaN 语义等价（两者均为 NaN，`torch.testing.assert_close(equal_nan=True)` 与 NaN 位对齐判定均相等）；省去 NaN 常数构造与广播 | 省 1 次 `vbrc`；规避 VP-2026-0142（`T.cast(float("nan"))` NaN FloatImm 破坏 kernel cache JSON 元数据 → 每次 miss 重编译） |

**机器验证结果表**（`verify_equiv.py`，fp64 参照 + 随机/角点 + 逐 dtype 舍入，已执行；bound 组合 both / min-only / max-only / min=NaN / max=NaN）：

| 采纳项 | 覆盖范围 | max ulp 差 | 违反率 | 结论 |
|--------|---------|-----------|--------|------|
| 1. NaN 传播化简（逐层 nan_max/nan_min → 普通 clamp + 单点恢复） | both / min-only / max-only finite × {fp32,fp16,bf16}（9 行） | 0.000 | 0 / 0（NaN 失配 0） | EQUIV_PASS |
| 2. NaN-bound 全 NaN 退化分支 | min=NaN / max=NaN × {fp32,fp16,bf16}（6 行） | 0.000 | 0 / 0（NaN 失配 0） | EQUIV_PASS |
| 3. NaN 值来源无常数化（canonical NaN vs 回选 x 的 NaN 位模式） | 角点 NaN 混合（1 行） | 0.000 | 0 / 0（NaN 失配 0） | EQUIV_PASS |

> 全部 16 行 `EQUIV_PASS`，无未通过项（无 FAIL），可采纳。脚本：`examples/clamp_scalar/clamp_scalar_fwd_kernel/verify_equiv.py`（Stage 2 维度 8 重跑复核）。

**优化结论**：采纳 3 项（2 个数学/边界等价项 + 1 个实现形态项；其数学内核即「NaN 传播化简 + NaN-bound 退化分支」）。优化后公式为：

$$
y_i = \begin{cases}
\text{NaN}, & \text{present bound}=\text{NaN} \\
x_i, & x_i=\text{NaN} \\
\operatorname{clamp}(x_i,\ \text{min},\ \text{max}) = \min(\max(x_i,\text{min}),\text{max}), & \text{otherwise}
\end{cases}
$$

（即普通 clamp + 单点 NaN 回选 + NaN-bound 退化，全程 fp32 中转）。无进一步优化空间：单次逐元素映射，无公共子表达式（x 复用一次于 clamp 与 NaN 恢复）、无除法/超越函数、无 cast 链冗余、无中间 GM 往返。优化后公式供 §3.1 公式拆解使用。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 升精度 cast（fp16/bf16→fp32） | 逐元素 `T.cast(x[i],"float32")` | `T.vcast(x_ub, x32, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；mish.py 佐证） | ✅ | — |
| 2 | 下界 `max(x, min)` | 逐元素 `max`/`nan_max` | `T.vmax(x32, min_val, buf1)`（`docs/Tilelang.language/比较操作/T.vmax.md`，src1 可标量；`customize_npuir.py` `npuir_max` B 支持标量） | ✅ | — |
| 3 | 上界 `min(·, max)` | 逐元素 `min`/`nan_min` | `T.vmin(x32, max_val, buf1)`（`docs/Tilelang.language/比较操作/T.vmin.md`，src2 可标量） | ✅ | — |
| 4 | 双侧 clamp | 逐元素 `min(max(...))` | `T.vclamp(x32, buf1, min_val, max_val)`（`docs/Tilelang.language/比较操作/T.vclamp.md`，前端展开为 npuir_max+npuir_min） | ✅ | — |
| 5 | NaN 检测 | 逐层 `isnan(a)||isnan(b)` | `T.vcmp(x32, x32, nan_mask, "ne")`（`docs/Tilelang.language/比较操作/T.vcmp.md`，`cmp_mod ∈ {eq,ne,lt,gt,ge,le}`，输出 bool，索引仅在片上使用） | ✅ | —（依赖 `"ne"` IEEE 无序语义，Stage 3 探针标定，见 §9.1；若证伪回退标量 `T.isnan` 兜底并改标「不可向量替代（API 缺失）」） |
| 6 | NaN 回填 select | 逐层 `Select(isnan, NaN, v)` | `T.vselect(nan_mask, x32, buf1, buf2)`（`docs/Tilelang.language/条件操作/T.vselect.md`） | ✅ | — |
| 7 | 降精度 cast（fp32→fp16/bf16） | 逐元素 `T.Cast(dtype, wide)` | `T.vcast(buf2, y_ub, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`） | ✅ | — |
| 8 | block 起始索引 `offset = block_id * block_size` | 标量乘加 | 无向量 API | ❌ | block 级任务映射索引，每 block 少量标量，非逐元素热点（可接受理由类别①） |
| 9 | 尾块大小 `tail_size = T.min(block_size, shape - offset)` | 标量 min | 无（`T.min` 标量形） | ❌ | 依赖动态 shape 的边界处理，每 block 一次，非逐元素热点（可接受理由类别④；mish.py 同款尾块） |
| 10 | NaN-bound 退化检测 `math.isnan(bound)` | trace-time Python 标量判定 | 无（编译期） | ❌ | host/trace 侧元数据计算，不在 kernel 内（可接受理由类别②） |
| 11 | `num_cores` / `tiles_per_core` 计算 | trace-time Python 标量 | 无（编译期） | ❌ | host/trace 侧 shape/分核元数据计算，不在 kernel 内（可接受理由类别②） |

**向量化结论**：逐元素计算（cast / max / min / clamp / NaN 检测 / NaN select / cast）已**全部向量化**（替代 API 均有 `docs/Tilelang.language/` 文档佐证：vcast/vmax/vmin/vclamp/vcmp/vselect）。保留 4 处标量：block 起始索引（block 级任务映射）、尾块大小（动态边界）、NaN-bound 检测与分核元数据（均 host/trace 侧，不在 kernel 内），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（输入 `[*S]` 连续 → 输出同 shape 同 layout），核内布局是设计变量。本算子展平为 1-D，非 batch 轴只有一个，属单轴算子；但为满足阻塞级要求，仍枚举候选并量化弃选。

**候选矩阵**（枚举有意义候选，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | I/O 原生 1-D 展平 `(N,)` | 展平轴（stride=1 连续） | 无（输入即连续，`flat` 为视图） | 基线：连续读写、无 repack、UB = O(block_size) | ✅ 采纳 |
| 2 | 保留 2-D `[M, K]` 视图，外层 M 串行、内层 K 向量化 | 尾轴 K（stride=1） | 无（reshape 视图） | 与候选 1 等价（1-D 展平即 `M=1` 的极限），但多一层 block 索引换算；K 与 1-D block_size 数学同构，无增益 | ❌（无结构差异，徒增索引；UB 预算相同） |
| 3 | 核内重排布局（如转置/交错） | 其他轴 | 核内 T.transpose 链（PL-1.1，~µs 级） | 逐元素算子无窗口/归约轴可重排——重排不改变逐元素访存模式，纯增 repack 代价（PL-1.3-host-permute 一路净亏） | ❌（逐元素无轴间复用，重排零收益） |

- **向量化轴** = 展平轴（stride=1 连续，无累加链、无跨步系数）。
- **整除性**：block_size=8192，fp16/bf16 向量宽 ×8、fp32 ×4 均整除（8192 = 1024×8 = 2048×4）✓；尾块 `tail_size` 动态，仅 copy 有效元素，垃圾 lane 不写回（与 mish.py 同款）。尾 lane 浪费率：0（整除档）；非整除档尾块 < 1 block（可忽略）。
- **累加链形态**：无累加（逐元素），无跨步系数阻碍。
- **repack 代价**：无（输入即 1-D 连续，无需转置/重排）。
- **UB 容量影响**：驻留缓冲 = O(block_size)（§4.5），布局候选间无差异。

**布局决策结论**：选定方案 #1「I/O 原生 1-D 连续布局 + 展平轴向量化」，无 repack、无重排。候选 #2/#3 弃选理由见矩阵（无结构差异 / 零收益纯增代价，且已按文档合法形态核对：逐元素算子无重排收益轴）。该结论同步落入 §3.3 伪代码与 §6 循环结构（向量维 = 展平轴，buffer 均为 1-D）。迁移对照：GPU 源码的隐式轴选择为「展平 1-D + thread/npt 两级」（§0.3），NPU Vector 轴与该 GPU thread 轴不对应，按本节独立评估后仍取展平轴（1-D 连续，与 GPU 展平语义一致），差异在并行组织（§0.6 R1）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（迁移默认）。

### 2.2 选型理由

纯 Vector 逐元素算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。Developer 模式由编译器自动管理同步与 buffer 分配，与 hardsigmoid/mish/lerp_tensor 同族 elementwise 先例一致。源算子 NPU wrapper 亦声明 `ascend_mode = "Developer"`（`examples/TileOPs/tileops/kernels/elementwise/clamp_scalar/clamp_scalar.py` L93）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix API 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（vcast/vmax/vmin/vclamp/vcmp/vselect），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 `set_flag/wait_flag`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x32 = cast(x)`（fp16/bf16 时） | 输入升 fp32（fp32 输入跳过），对齐 torch golden 的 fp32 opmath |
| 2 | `t = clamp(x32, min, max)` | 双侧 `vclamp`；仅 min→`vmax`；仅 max→`vmin`（absent 侧在 trace 时跳过） |
| 3 | `mask = isnan(x32)`（NPU：`vcmp(x32,x32,"ne")`） | 单点 NaN 掩码（IEEE 无序自比较） |
| 4 | `r = select(mask, x32, t)` | 单点 NaN 恢复（回选 x32，§1.6.1 项 1/3） |
| 5 | `y = cast(r)`（fp16/bf16 时） | 降回输出 dtype |
| 6 | NaN-bound 退化（trace-time） | 任一 present bound 为 NaN → 全 NaN 输出（§1.6.1 项 2） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 数据搬入 GM→UB | `T.copy(x[offset:offset+tail], x_ub[0:tail])` | 双显式 slice（TRAP-T-copy-region-semantics 推荐形态） | Developer |
| 1 | cast 升 fp32 | `T.vcast(x_ub, x32, round_mode="rint")` | fp16/bf16 → fp32 | Developer |
| 2a | 双侧 clamp | `T.vclamp(x32, buf1, min_val, max_val)` | 标量 min/max 广播 | Developer |
| 2b | 仅下界 | `T.vmax(x32, min_val, buf1)` | 标量 min（present(min) 且 not present(max)） | Developer |
| 2c | 仅上界 | `T.vmin(x32, max_val, buf1)` | 标量 max（present(max) 且 not present(min)） | Developer |
| 3 | NaN 掩码 | `T.vcmp(x32, x32, nan_mask, "ne")` | 自比较，nan_mask dtype=bool（`docs/Tilelang.language/比较操作/T.vcmp.md`） | Developer |
| 4 | NaN 恢复 | `T.vselect(nan_mask, x32, buf1, buf2)` | 真值回选 x32（非原位，规避 TRAP-vselect-inplace-carried-state） | Developer |
| 5 | cast 降 dtype | `T.vcast(buf2, y_ub, round_mode="rint")` | fp32 → fp16/bf16 | Developer |
| 6 | 数据搬出 UB→GM | `T.copy(y_ub[0:tail], y[offset:offset+tail])` | 双显式 slice | Developer |

> fp32 输入路径跳过步骤 1/5（无 vcast）。`min_val`/`max_val` 为构造期 Python 标量，trace-time `T.cast(min_val,"float32")` 上抛，与源 `op_func` 一致；absent 侧整段不生成。**cast 注意**：`T.vcast` 的 src 仅支持 buffer（不可标量，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.3），本设计所有 vcast 均为 buffer→buffer。

### 3.3 计算伪代码

> `T.Tensor` 的数据类型用**第二个位置参数**传入（不用 `dtype=` 关键字），避免 `name 'in_dtype' is not defined` 类 false alarm（design-template §3.3 约定，不违反迁移执行规则第 3 条）。

```python
import math
import functools
import tilelang
import tilelang.language as T

@functools.lru_cache(maxsize=32)
def clamp_scalar_fwd_kernel(N, dtype, output_dtype=None, min_val=None, max_val=None,
                            block_size=8192, num_cores=None):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    n_num = T.ceildiv(N, block_size)
    # 物理核数实查 ×2（纯 Vector），见 §5.5；num_cores 缺省或不合理时回退 n_num
    if num_cores is None or num_cores <= 0 or num_cores >= n_num:
        num_cores = n_num
    tiles_per_core = T.ceildiv(n_num, num_cores)            # 编译期常量，静态边界

    # trace-time：NaN-bound 退化判定（§1.6.1 项 2），finite bound 时该分支不触发
    nan_bound = (min_val is not None and math.isnan(min_val)) or \
                (max_val is not None and math.isnan(max_val))

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(_block_size, _num_cores):
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
            shape: T.int32,
        ):
            with T.Kernel(_num_cores, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((_block_size,), dtype)
                x32 = T.alloc_ub((_block_size,), compute_dtype)
                buf1 = T.alloc_ub((_block_size,), compute_dtype)
                buf2 = T.alloc_ub((_block_size,), compute_dtype)
                y_ub = T.alloc_ub((_block_size,), out_dtype)
                nan_mask = T.alloc_ub((_block_size,), "bool")

                for i in T.serial(tiles_per_core):
                    block_id = i * _num_cores + cid
                    if block_id < n_num:
                        offset = block_id * _block_size
                        tail_size = T.min(_block_size, shape - offset)
                        T.copy(x[offset:offset + tail_size], x_ub[0:tail_size])

                        if dtype != compute_dtype:
                            T.vcast(x_ub, x32, round_mode="rint")
                        else:
                            T.copy(x_ub, x32)

                        if nan_bound:
                            # NaN bound -> torch 全 NaN；仅用有限常数 0.0 构造 NaN
                            T.vmul(x32, 0.0, buf1)          # 有限->0；±inf->NaN；NaN->NaN
                            T.vdiv(buf1, 0.0, buf2)         # 0/0->NaN；NaN->NaN => 全 NaN
                        else:
                            if min_val is not None and max_val is not None:
                                T.vclamp(x32, buf1, T.cast(min_val, "float32"),
                                         T.cast(max_val, "float32"))
                            elif min_val is not None:
                                T.vmax(x32, T.cast(min_val, "float32"), buf1)
                            else:
                                T.vmin(x32, T.cast(max_val, "float32"), buf1)
                            T.vcmp(x32, x32, nan_mask, "ne")            # NaN 掩码
                            T.vselect(nan_mask, x32, buf1, buf2)        # 真值回选 x

                        if out_dtype != compute_dtype:
                            T.vcast(buf2, y_ub, round_mode="rint")
                        else:
                            T.copy(buf2, y_ub)
                        T.copy(y_ub[0:tail_size], y[offset:offset + tail_size])

        return main

    return kernel(block_size, num_cores)
```

> 注：`nan_mask` 为 `bool`（`vcmp` 输出，仅片上使用，不写回 GM——`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3）；`vselect` 写入独立 `buf2`（非原位 `buf1`），规避 TRAP-vselect-inplace-carried-state。fp32 路径可走更简 buffer 链（`x32` 复用 `x_ub`、`y_ub` 复用 `buf2`），驻留由 168KB 降至 ≤136KB（§4.5），Stage 3 实现时细化。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vclamp(src,dst,min,max)` | `docs/Tilelang.language/比较操作/T.vclamp.md`（标量/张量 min/max；前端展开 npuir_max+npuir_min） | ✅ 已验证（文档读） |
| `T.vmax(src1,src2,dst)` / `T.vmin(src1,src2,dst)` | `docs/Tilelang.language/比较操作/T.vmax.md` / `T.vmin.md`（src2 支持标量）；`customize_npuir.py` `npuir_max/min`（B 支持 scalar） | ✅ 已验证 |
| `T.vcmp(A,B,C,cmp_mod)` | `docs/Tilelang.language/比较操作/T.vcmp.md`（`cmp_mod ∈ {"eq","ne","lt","gt","ge","le"}`，C bool，仅片上） | ✅ 已验证 |
| `T.vselect(Cond,A,B,Out)` | `docs/Tilelang.language/条件操作/T.vselect.md`（4 参同 shape） | ✅ 已验证 |
| `T.vcast(src,dst,round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（f16/bf16→f32 支持 "rint"；src 仅 buffer）；mish.py `round_mode="rint"` | ✅ 已验证 |
| `T.vmul` / `T.vdiv` | `docs/Tilelang.language/数学操作/T.vmul.md` / `T.vdiv.md`（标量 B 支持） | ✅ 已验证（NaN-bound 分支用） |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md`；TRAP-T-copy-region-semantics | ✅ 已验证 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；mish.py | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | mish.py / hardsigmoid DESIGN.md | ✅ 已验证 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 `T.Kernel(ceildiv(N, block_size))`） | 一维 `T.Kernel(num_cores, is_npu=True)`，无三维 |
| GPU 专用 API | Yes（`T.Parallel`/`T.vectorized`/`T.unroll`/`tl.max_nan`/`tl.min_nan` CUDA 语义） | 全部改写为 NPU v-prefix API + UB 路径（§0.5/§0.6） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(N, block_size), threads=256)`（block+thread 两级） | 一维 `T.Kernel(num_cores, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(threads, npt)` 显式两级循环 | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| NaN min/max | `tl.max_nan/tl.min_nan`（CUDA intrinsic）+ 逐层 isnan/Select | 普通 `vclamp/vmax/vmin` + `vcmp(x,x,"ne")` + `vselect` 单点恢复 | §0.6 R2 |
| NaN bound | 逐层 nan_max/nan_min 的 isnan(bound) 分支自然传播 | trace-time 退化分支（全 NaN） | §0.6 R3 |
| NaN 值构造 | `T.cast(float("nan"),"float32")` 常数 | 回选输入 x 的 NaN 位模式（无 NaN 常数） | §1.6.1 项 3（规避 VP-2026-0142） |
| `threads×npt` | 1024(fp32)/2048(fp16/bf16) | `block_size=8192` 折叠（K11） | §5.2 |
| bound 传参 | 闭包常量 `T.cast(min_val,"float32")` | v-prefix 标量 operand（vclamp/vmax/vmin 的标量 min/max） | §0.1 / §3.2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/hardsigmoid/hardsigmoid_fwd_kernel/DESIGN.md` | 高度相似（同族 clamp/NaN 迁移） | `tcmp`/`vselect` NaN 单点恢复、v-prefix 链、fp32 中转、`T.alloc_ub`、persistent 分核、UB 预算表 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 高度相似（同族 unary 迁移） | Developer 模式、`T.alloc_ub`、`T.copy` 双显式 slice 尾块、v-prefix 链、fp32 中转、`T.Kernel(is_npu=True)` |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 相似（同族 elementwise） | fp32 中转模式（PL-1.17-subfp32-fp32-transit）、`T.min` 尾块 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `input` | `[*S]`（展平 1-D `N`） | `float16 \| bfloat16 \| float32` | 输入张量，host 展平视图 |
| `min` | 标量 | `Number` | 下界（可选；构造期参数，非 tensor） |
| `max` | 标量 | `Number` | 上界（可选；构造期参数，非 tensor） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | `[*S]`（与 input 相同） | 与 input 相同 dtype | 输出张量，host 回塑 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | input dtype | UB | 输入 tile（GM→UB） |
| `x32` | `(block_size,)` | float32 | UB | fp32 中转输入（vcast 目标 / NaN 回选源） |
| `buf1` | `(block_size,)` | float32 | UB | clamp 输出 |
| `buf2` | `(block_size,)` | float32 | UB | NaN 恢复输出 / 退化分支全 NaN |
| `y_ub` | `(block_size,)` | output dtype | UB | 输出 tile（UB→GM） |
| `nan_mask` | `(block_size,)` | bool | UB | NaN 掩码（vcmp "ne" 输出，仅片上） |

### 4.4 内存搬运路径

```
GM[x] --T.copy--> UB[x_ub] --vcast--> UB[x32] --clamp(vclamp/vmax/vmin)--> UB[buf1]
  --vcmp(x32,x32,"ne")+vselect(回选 x32)--> UB[buf2] --vcast--> UB[y_ub] --T.copy--> GM[y]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返；无 L1/inter-core 依赖。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `x_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| `x32` | (8192,) | fp32 | 32768 |
| `buf1` | (8192,) | fp32 | 32768 |
| `buf2` | (8192,) | fp32 | 32768 |
| `nan_mask` | (8192,) | bool | 8192 |
| `y_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| **总计（fp16/bf16 路径）** | | | **≈ 139264（136KB）** |
| **总计（fp32 路径，全 buffer 显式）** | | | **≈ 172032（168KB）** |
| **总计（fp32 路径，复用 `x32`←`x_ub` + `y_ub`←`buf2`）** | | | **≈ 106496（104KB）** |

对照目标平台 UB 容量 **192KB**（`CONST-capacity-910B2C`：UB=192KB/AIV；version stamp `tilelang 0.1.2+3a214cde` + CANN 8.5.0 / 2026-09-17 ssd 第三数据点）。按该条目 ×1.10–1.12 上偏系数（auto-multi-buffer=false 下 BishengIR 实际需求 ≈ 手工预算 ×1.10–1.12；逐 buffer 显式 alloc 结构实测 ratio ≈0.981，留 ~12% 余量即足，不必按 1.7x 悲观）：

- **fp16/bf16 路径**：136KB ×1.12 ≈ 152KB < 192KB ✓，余量 ~21%。
- **fp32 路径（全 buffer 显式）**：168KB ×1.12 ≈ 188KB < 192KB ✓，余量仅 ~2%（紧张但**不溢出**）。
- **fp32 路径（复用方案，推荐）**：fp32 输入即 fp32，`x32` 复用 `x_ub`（省 32KB → 136KB）；fp32 输出即 fp32，`buf2` 结果直接 `T.copy` 回 GM（`y_ub` 可省 → 104KB，×1.12 ≈ 116KB，余量 ~39%）。

结论：fp32 路径实际驻留 ≤136KB（复用 `x32`）乃至 ≤104KB（全复用），**无需单独调小 block_size**；即便按最保守的全 buffer 显式分配，168KB ×1.12 ≈ 188KB 仍在 192KB 之内。Stage 3 按 fp32 路径走复用链（`x32`←`x_ub`、`y_ub`←`buf2`）。

### 4.6 动态轴定义

无动态轴。`N` 为 JIT 编译期参数（`clamp_scalar_fwd_kernel(N, dtype, ...)` 每 shape 一特化），`shape`（`T.int32`）为运行时传入的实际元素数（用于尾块 `shape - offset`，与 mish.py 同款）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素运算（cast/clamp/compare/select），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_size = 8192        # 同族 hardsigmoid/mish 实测最优元素级块宽
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数
```

- **选择理由**：8192 为同族 elementwise 实测最优 block_size（hardsigmoid `block_size=8192`、mish perf_opt `BLOCK_SIZE=8192`）；32B 对齐（8192 × 2/4B 均为 32B 整数倍）；UB 预算 ~136–168KB < 192KB（§4.5，fp32 复用后 ≤136KB）；块宽足够使向量链发射开销摊薄（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。
- **与 NPU wrapper 脚手架差异**：已提供的 `examples/TileOPs/tileops/kernels/elementwise/clamp_scalar/clamp_scalar.py` 中 `_DEFAULT_BLOCK_SIZE` 为 `{float32:1024, float16/bf16:2048}`（GPU `threads×npt` 直接折叠值）。本设计据同族实测将 NPU `block_size` 定为 **8192**——**这是设计决策，Stage 3 实现时以本设计为准**；逻辑核数/分核据此计算（§5.5）。
- **块宽敏感性**：发射/流量张力见 §1.6.0「设计期估算」备注与 §6.3；Stage 4 可扫描 block_size（受 UB 192KB 约束）与 `T.Pipelined`。

### 5.3 约束分析

- **对齐约束**：block_size=8192，fp16/bf16 尾轴 8192×2B=16KB、fp32 8192×4B=32KB，均 32B 对齐 ✓。
- **UB 容量**：~136–168KB < 192KB（§4.5；fp32 全 buffer 168KB ×1.12 ≈ 188KB 余量 ~2%，复用后 136KB ×1.12 ≈ 152KB 余量 ~21%）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾块**：`tail_size = T.min(block_size, shape - offset)`，仅 `T.copy` 有效元素进/出；v-prefix 对全 buffer 计算，尾部垃圾元素不写回（与 mish.py 同款）。
- **极小 shape**（如 N=1024 < 8192）：`n_num=1`，`num_cores=1`，单核单 tile 处理，无 oversubscription。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **② 物理核数依据（实查）**：查询代码与实际返回值记录于此（禁止文档假设/经验值替代）：

```python
from tilelang.utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 算子核数翻倍：40
```

实查命令与输出（2026-10-13 本机执行，toolchain tilelang 0.1.2 dev-root + CANN 8.5.2）：`get_aicore_num()` 返回 **20**。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。（备注：`CONST-aicore-910B2C` 记录某 910B2C 设备为 24，与实查 20 不一致——设备档/型号差异，**一律以实查为准**，不以 24 假设。）

- **① 逻辑核数**：`num_logical_kernels = ceil(N / block_size)`，随 workload 变化：
  - smoke-1m（N=1048576）：128；16M（N=16777216）：2048；4M（N=4000000）：489；256M（N=268435456）：32768。

- **③ 规模判定与分核方案**：workload 跨越「无需适配」到「极大规模」多档。设计采用**统一 persistent 结构**一次覆盖全部档位（成熟 elementwise 形态，hardsigmoid/mish perf_opt + PL-1.6-copy-floor「grid-stride 聚集窗口实测优于每核连续分块」）：

- **分核方案**：`num_cores = min(ceil(N/block_size), 40)`，核内 `T.serial(tiles_per_core)` 串行处理多个逻辑块（`tiles_per_core = T.ceildiv(n_num, num_cores)`，编译期常量，**循环边界静态**）：
  - **小规模（逻辑核数 ≤ 40，如 N ≤ 327680）**：`num_cores = n_num`，每核 ≤1 tile，`tiles_per_core=1`，无串行循环、无负载不均——结论「**无需适配**」（逻辑核数 ≤ 物理核数）。
  - **中等 / 极大规模（逻辑核数 > 40，如 smoke-1m 128、16M 2048、256M 32768）**：固定启动内核数 = 40（= 物理向量核数），核内**串行**处理 `tiles_per_core = ceil(n_num/40)` 个 tile（smoke-1m：4；16M：52；256M：820），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均；循环边界 `tiles_per_core` 为编译期常量（N/block_size/40 均编译期已知），满足「核内串行边界静态」。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_cores, is_npu=True) as (cid, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(tiles_per_core)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（vcast/vclamp/vmax/vmin/vcmp/vselect） | 块内逐元素由 Vector 单元并行（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_cores, is_npu=True) as (cid, _):
    # UB buffer 分配 ...
    for i in T.serial(tiles_per_core):          # 核内串行 tile 循环（静态边界）
        block_id = i * num_cores + cid
        if block_id < n_num:                    # 尾 tile guard
            # 装载 -> v-prefix 计算链 -> 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对非退化路径采用 guard-free 2-stage `T.Pipelined(tiles_per_core, num_stages=2)` 双缓冲流水（mish perf_opt 形态），将向量链隐藏于 MTE2 窗口（PL-1.6-copy-floor：≥16M 档计算链被完全隐藏）。**这是消除 §1.6.0「发射不可重叠则主导」风险的直接手段**。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 shape：`tail_size = T.min(block_size, shape - offset)`；`T.copy` 仅装载/写回 `[0:tail_size]` 有效元素；v-prefix 对全 block_size buffer 计算，垃圾尾部不写回（与 mish.py 同款）。`block_id < n_num` guard 覆盖最后一波不完全 tile。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer）。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器自动管理 GM↔UB 搬运与向量计算间的依赖（`T.copy` 与后续 v-prefix op、以及各 v-op 之间的数据依赖由自动同步保证），无跨核协作（逐元素无跨元素依赖，各核独立）。

### 7.3 pass_configs 配置

Developer 模式默认配置，无需额外 `pass_configs`。环境变量 `TILELANG_ASCEND_MODE = "Developer"`（可选，默认即 Developer；wrapper 声明 `ascend_mode = "Developer"`）。NaN-bound 退化分支与 finite 主路径均为 trace-time 选定的单一程序，无运行时分支。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据（NaN → NaN、NaN bound → 全 NaN、fp32 opmath、输出同 dtype），移植源仓测试基准（`torch.clamp(x.float(), min, max).to(x.dtype)`），**不复刻 §0.6 的 NPU 算法**。

```python
import torch

def golden_clamp_scalar(x, min_val=None, max_val=None):
    """Clamp 参考实现：y = clamp(x, min, max)，fp32 opmath，NaN 传播。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_clamp_scalar.py 的
    ref_program = torch.clamp(x.float(), self.min_val, self.max_val).to(x.dtype)）。
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = torch.clamp(x_f32, min_val, max_val)   # NaN 输入/bound -> NaN（torch 语义）
    return y_f32.to(x_cpu.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |
| float32 | 1e-5 | 1e-5 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓测试 `_get_tolerances` 一致。kernel 与 golden 均为 fp32 中转，差 ≤1 ulp；fp32 中转链对齐 torch fp32 opmath，见 PL-1.17-subfp32-fp32-transit。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`N=1048576`，`float32` / `float16` / `bfloat16` 三 dtype，`randn` 输入，`min=-0.5, max=0.5`，vs golden 在对应容差下 `torch.testing.assert_close`。另覆盖三参数形态（both / min-only / max-only，`test_clamp_scalar_param_names`）。完整分层套件（L1 非整除/多 shape、L2 边界 size、Boundary 特殊值含 NaN/±inf）交由 `tilelang-op-develop` 规划，本设计不枚举。

**特别提醒（NaN 语义验证）**：Boundary 用例必须覆盖 NaN 输入 → NaN 输出（§0.1 语义层要求），验证 §0.6 R2 的 `vcmp("ne")`+`vselect` 单点恢复；并据此确认：① Ascend `vmax`/`vmin` 的实际 NaN 行为；② `npuir_cmp "ne"` 是否按 IEEE 无序比较使 NaN≠NaN→true（§9.1 风险点——若证伪则回退标量 `T.isnan` 兜底）。NaN bound（min=NaN / max=NaN）→ 全 NaN 也应作为 Boundary 用例验证 §0.6 R3。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| Ascend `vmax`/`vmin` NaN 语义未文档化 | Ascend C Max/Min 文档（CANN 9.0.0-beta1）「约束说明」仅载地址对齐/重叠，无特殊值处理说明；pattern-library `PL-1.11-causal-mask-scalartrap` P8 探针结论「vmax/vmin 钳位只治 ±Inf」指示 NPU vmax/vmin **不传播 NaN**。本设计用 `vcmp(x,x,"ne")`+`vselect` 单点恢复兜底。若 Stage 3 实测 vmax/vmin 本身传播 NaN，可移除恢复步（可选优化，不影响正确性）；反之必须保留 |
| `npuir_cmp "ne"` 的 NaN 无序比较语义未实测 | `vcmp(x,x,"ne")` 自比较检测 NaN 依赖 IEEE 无序比较（NaN≠NaN→true；±inf/有限→false）。Ascend 向量比较指令是否按 IEEE 无序实现**需 Stage 3 NaN Boundary 探针标定**（pattern-library `CONST-vcmp-ne-nan-unordered` 在 hardsigmoid 任务三 dtype 实测通过，本算子沿用其结论并待本任务重验）。若证伪，回退 §1.6.2 #5 方案 (c)：NaN 检测改标「不可向量替代（API 缺失）」（`T.isnan` 仅标量）+ 标量兜底，并修正 §6「无逐元素标量循环」声明 |
| VP-2026-0142（NaN 常数 cache 陷阱） | `T.cast(float("nan"),"float32")` NaN FloatImm 破坏 kernel cache JSON 元数据 → 每次 miss 重编译。本设计**回选输入 x 的 NaN 位模式**（§1.6.1 项 3），全程不使用 NaN 常数；NaN-bound 退化分支亦仅用有限常数 0.0。**不引入该陷阱** |
| UB 预算 | ~136–168KB < 192KB（§4.5）；fp32 全 buffer 168KB ×1.12 ≈ 188KB 余量 ~2%（复用 `x32`→`x_ub` 后 136KB ×1.12 ≈ 152KB 余量 ~21%，再复用 `y_ub`→`buf2` 后 104KB）；Stage 3 复用 buffer 进一步放宽 |
| 发射/流量张力（设计期估算） | 256M 档流量项 ~852µs 为下界；若发射成本不可跨 tile 重叠，32768 tile 的发射项可达 ~2.9ms > 流量项 → block_size 增大 / `T.Pipelined` 为 Stage 4 首要调优轴（§6.3）。属**设计期未实测假设**，Stage 4 实测标定 |
| 分核 | 物理核数实查 20（×2=40），与 `CONST-aicore-910B2C`（24）不一致——本环境设备返回 20，以实查为准 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_size 过大（>8192 × fp32 全 buffer 链） | 编译失败（`ub overflow`） | 维持 block_size=8192；fp32 路径复用 buffer |
| NaN 未传播 | 直接 `vclamp` 而 vmax/vmin 丢 NaN | 输出 clamp 值而非 NaN，违反语义 | §0.6 R2 的 `vcmp(x,x,"ne")`+`vselect` 单点恢复 |
| NaN bound 结果错误 | present bound=NaN 时走普通 clamp | 输出 clamp 值而非全 NaN | §0.6 R3 trace-time 退化分支（全 NaN） |
| 原位 vselect 丢状态 | `T.vselect(cond,A,buf,buf)` 在 T.serial 内原位 | loop-carried 状态下 tile-0 局部值（TRAP-vselect-inplace-carried-state） | 本设计写入独立 `buf2`（非原位）；tile 间本就无 carried 依赖 |
| 尾块越界 | `T.copy` 用 base+size 形态写尾块 | MTE 越界写坏相邻 tensor | 双显式 slice `T.copy(src[0:tail], dst[0:tail])`（TRAP-T-copy-region-semantics） |
| `dtype=` 关键字 false alarm | `T.Tensor((N,), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（design-template §3.3） |

### 9.3 特殊场景处理

- **非整除 shape**（N=4000000 等）：尾块 `tail_size` + `block_id < n_num` guard（§6.4）。
- **极小 shape**（N < block_size）：`n_num=1`，单核单 tile。
- **NaN / ±inf 输入**：NaN → NaN（§0.6 R2）；±inf → clamp 收敛到 min/max，无上溢。
- **NaN bound**：全 NaN（§0.6 R3）。
- **min > max**：与 torch 一致（`min(max(x,min),max)` 组合逻辑，结果 = max）；manifest 不出现。
- **混合精度**：fp16/bf16 走 fp32 中转（vcast），fp32 直算；输出 dtype 与输入一致。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/clamp_scalar/clamp_scalar_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，结果内嵌）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成 | §1.6.1 等价性验证（16 行全 EQUIV_PASS） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `clamp_scalar_fwd_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 测试 |
| `test_clamp_scalar.py` | ⬜ 待实现（Stage 3） | 精度测试 |

### 10.3 命名规范

- 项目目录名：`clamp_scalar`；算子目录名：`clamp_scalar_fwd_kernel`；接口函数：`clamp_scalar_fwd_kernel`（manifest 键 `ClampScalarFwdOp`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`clamp_scalar_fwd_kernel`）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）
