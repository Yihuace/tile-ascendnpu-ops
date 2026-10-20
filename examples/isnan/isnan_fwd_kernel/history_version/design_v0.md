# IsnanFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/comparison.py` 的 `IsnanFwdKernel(FloatPredicateKernel, UnaryPredicateFwdInterface)`，复用 `_base.py` 的 `FloatPredicateKernel` / `UnaryKernel`、`_builders.py` 的 `make_unary_regcopy`、`_op_body.py` 的 `store_unary_bool_as_int8`）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `isnan_fwd_kernel`，`target="npuir"`。
> 迁移工作区：`examples/isnan/isnan_fwd_kernel/`（project=`isnan`，op=`isnan_fwd_kernel`，scenario=migration，mode=harness）。
> 本设计按 skill `tilelang-op-design` 的 Phase M0（源算子三问解读）→ Phase R（算法调研）→ Phase M1（硬件耦合性判定与 NPU 重设计）→ 通用设计流程（Phase 1–4）产出；本文档为 `first_design` 基线（无 revision 段）。
> **谓词与输出头条结论**：本算子为 **float 逐元素谓词**（`bool = isnan(x)`），无浮点算术、无归约、无跨元素依赖。NaN 检测在 NPU 上用 **`vcmp(x, x, mask, "ne")` 自比较**（IEEE 无序：NaN≠NaN→true）在**原生 dtype**（fp16/bf16/fp32）上完成，**不使用**标量 `T.isnan`、**不做** fp32 宽化中转；bool 输出以 **int8 字节存储**（CG-2026-0019 先例：bool i1 不可 load/store GM）经 `vcast` 链落到 0/1 字节，host 侧 `view(torch.bool)` 回转。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：`torch.isnan(input)` 的逐元素 NaN 谓词，输出 bool：

$$
y_i = \operatorname{isnan}(x_i) =
\begin{cases}
\texttt{true}, & x_i \ \text{is NaN} \\
\texttt{false}, & \text{otherwise}
\end{cases}
$$

源实现（`IsnanFwdKernel.op_func`，`comparison.py` L186-191）：

```python
@staticmethod
def op_func(x):
    return T.isnan(T.cast(x, "float32"))
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | `y_i = isnan(x_i)`：NaN 元素为 True，其余（含 ±inf、±0、次正规、有限值）为 False |
| 规约语义 | **无规约**，纯逐元素，无跨元素依赖、无累加顺序问题 |
| dtype 语义 | 输入 `float16 \| bfloat16 \| float32`（`FloatPredicateKernel` 继承 `FloatUnaryKernel`，`SUPPORTED_DTYPES = FLOAT_DTYPES`）；**输出 `torch.bool`**（`OUTPUT_DTYPE = torch.bool`）。源码 `T.cast(x, "float32")` 是**谓词可用性适配**（把半精度宽化到 fp32 后再调 `T.isnan`），不是精度提升——NaN 性在宽化下保持不变；bool 结果经 `store_unary_bool_as_int8` 以 int8 字节 0/1 落盘 |
| 边界语义 | NaN → True（源 docstring「Element-wise isnan with torch-style bool output」）；±inf → `inf == inf` → False（不被误判为 NaN）；±0 / 次正规 / 有限值 → False。空维度按展平后处理；非整除 shape 由尾块处理 |
| host 侧语义 | `flat(x)` 展平（视图）→ kernel → `_restore_output_dtype(result).reshape(x.shape)`；`_bool_via_int8` 为真时 `_restore_output_dtype` = `result.view(torch.bool)`。均为 stride/shape/dtype 视图操作，**不改输入张量内容** |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.isnan`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input` | `[*S]`（任意，展平为 1-D `N=numel`） | `float16 \| bfloat16 \| float32` | 输入张量；kernel 收到 `flat(x)` 的 1-D 视图 |
| `output` | 与 input 相同 shape `[*S]` | `bool`（源码以 int8 字节存储落地） | 输出张量；`forward` 内 `_restore_output_dtype(result).reshape(x.shape)` 回塑（非转置布局） |

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host 展平：`flat(x)` → 1-D `(N,)` | `input [*S]` | `x_flat (N,)` | 前置（host 视图操作，不改数据） |
| 2 | 分块装载：`T.copy(x[bx*bs:(bx+1)*bs], x_reg)`（fragment） | `x_flat` | `x_reg (bs,)` fragment | 数据搬入 |
| 3 | 逐元素宽化：`T.cast(x, "float32")` | `x_reg` | `x_f32`（fp32） | 谓词可用性适配（非语义） |
| 4 | 逐元素谓词：`T.isnan(x_f32)` | `x_f32` | `pred`（bool） | `isnan(x_i)` |
| 5 | bool 字节落地：`if_then_else(pred, 1, 0)`（`store_unary_bool_as_int8`，`BOOL_STORAGE_DTYPE="int8"`） | `pred` | `y_int8` | bool 结果表示为 0/1 字节 |
| 6 | 分块写回：`T.copy(y_reg, y[bx*bs:(bx+1)*bs])` | `y_reg` | `y_flat`（int8） | 数据搬出 |
| 7 | host 回塑：`result.view(torch.bool).reshape(x.shape)` | `y_flat` | `output [*S]`（bool） | 后置（stride/shape/dtype 视图） |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x] --T.copy--> fragment/x_reg --T.cast(f32)--> x_f32 --T.isnan--> pred --if_then_else(1,0)--> y_int8 --T.copy--> GM[y]
```

`IsnanFwdKernel` 继承 `FloatPredicateKernel`（`DEFAULT_STRATEGY = "register_copy"`，`OUTPUT_DTYPE = torch.bool`）。`UnaryKernel` 经 `elementwise_output_plan(..., bool_storage=True)` 判定 `bool_via_int8`，`_get_effective_op_func` 用 `store_unary_bool_as_int8(op_func)` 包裹：`y = T.if_then_else(op_func(x), T.cast(1,"int8"), T.cast(0,"int8"))`；随后 `make_unary_regcopy` 走 fragment 全块路径（分块 `T.copy` 装载、`T.Parallel` 逐元素求解、整块写回）。

**循环与并行结构**：grid 一维 `T.Kernel(ceildiv(N, block_size), threads=128)`，block 内 `T.Parallel(threads, npt)` 两级元素并行；`block_size = threads × npt`。

**host 侧逻辑**：`forward` 中 `flat(x)`（展平，视图）→ `_compiled_fn(...)` → `_restore_output_dtype(result).reshape(x.shape)`（`view(bool)` + 回塑）。Op 层（`IsnanFwdOp._eager_forward`）为 `input.contiguous().reshape(-1) → kernel → reshape(orig_shape)`。全部为 stride/shape/dtype 视图操作，**不改输入张量内容**（满足 ascend-constraints 的 host 侧约束）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | register_copy（fragment 分块装载/写回） | 减少 GM 重复访问、放大访存粒度 | 分块 `T.copy` 到 fragment，逐元素计算后整块写回 | CUDA SMEM/寄存器 | 硬件强相关 |
| 2 | coalesced 向量化访问 | 尾轴连续读写打满带宽 | fragment 块 `T.copy` 连续搬运 | memory coalescing | 硬件强相关（NPU 有 MTE 等价） |
| 3 | `threads × npt` 两级并行执行模型 | 每线程处理 `npt` 元素放大吞吐 | CUDA thread/warp 组织 + `T.Parallel` | CUDA 线程模型 | 硬件强相关 |
| 4 | fp32 宽化后再谓词（`T.cast(x,"float32")` + `T.isnan`） | 让半精度输入能用上 `T.isnan` | 宽化到 fp32 再调 NaN 谓词 | CUDA `T.isnan` 的 dtype 适配 | 类型/API 适配（可移植，可替换） |
| 5 | bool 字节存储（`store_unary_bool_as_int8`，`BOOL_STORAGE_DTYPE = int8`） | bool 无 VMEM 型别，按字节 0/1 落盘 | `if_then_else(pred, 1, 0)` 写 int8，host `view(bool)` | 无（存储层） | 可移植 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。识别不出来 ≠ 不存在——上表为源码 `op_func` + `FloatPredicateKernel` / `store_unary_bool_as_int8` / `make_unary_regcopy` 的逐项核对结果。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `bool = isnan(x)`（NaN→True、±inf/有限→False） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| bool 结果按字节 0/1 存储 | 算法/存储 | 无（`BOOL_STORAGE_DTYPE` 是后端事实） | 有（NPU bool i1 同样不可 load/store GM） | **等价替换** | NPU 输出以 `int8` 字节存储承载（CG-2026-0019：uint8/bool 无运行时 out-dtype 映射、i1 不可 GM 存取）；host `view(torch.bool)` 回转（视图，不改内容）；判据见 `docs/Tilelang.language/比较操作/T.vcmp.md` §2.3 |
| `T.isnan` 谓词 + fp32 宽化（`T.cast(x,"float32")`） | 优化 | CUDA `T.isnan` 的 dtype 适配 | 无同款向量 `T.isnan`（`T.isnan` 为标量谓词） | **重新设计** | → §0.6 R2（`vcmp(x,x,"ne")` 原生 dtype 自比较 + 掩码→字节链） |
| register_copy fragment 分块装载/写回（SMEM/寄存器） | 优化 | SMEM/寄存器 | 有（UB） | **等价替换** | `T.alloc_ub` + `T.copy`（GM→UB→GM），ascend-constraints 已知限制 / decision-tree §3 |
| coalesced 连续访问 | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | `T.copy` 连续 slice 搬运（双显式 slice 形态，`docs/Tilelang.language/内存操作/T.copy.md`） |
| `threads × npt` 两级并行 + `T.Parallel` 标量循环 | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel） | **重新设计** | → §0.6 R1（block_size 折叠 + 一维 `T.Kernel` + UB 向量化） |
| swizzle / mma / warp shuffle | 优化 | SMEM bank / Tensor Core / warp | 不涉及（源未使用） | **舍弃（不适用）** | 源算子为纯逐元素谓词，未使用这些手段，无迁移对象（migration-analysis.md §5.3 同类项） |

> 每项处置均写明依据（migration-analysis.md §5.3 映射表 / `examples/` 佐证 / `docs/` 条目）。**无「舍弃理由待补」项**——唯一判「舍弃（不适用）」的 swizzle/mma/shuffle 本就不在源算法中。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（`threads × npt` → 一维 Kernel + UB 向量化）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=128)` + `T.Parallel(threads, npt)` 两级并行；`block_size = threads × npt`；fragment 装载/写回（`make_unary_regcopy`）。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；GPU 的 `threads × npt` 折叠为单一 `block_size`（ascend-constraints 已知限制「仅支持一维 Kernel」；GPU `threads=` kwarg 在 NPU 无效果，见 `TRAP-threads-kwarg-noop`）；元素级并行交给 Vector 单元（v-prefix 逐元素 API 对整块 UB buffer 计算，天然向量化，无需显式 `T.Parallel` 循环）；分块装载/写回用 `T.copy`（GM→UB→GM，32B 对齐连续 slice）。`block_size` 取值见 §5.2。
- **语义保持论证**：逐元素映射不变（每元素独立计算，无跨元素依赖、无规约），`threads/npt` 到 `block_size/核` 的任务映射变化不改变任何输出值——每个 `out[i]` 的谓词结果与源码逐步一致。边界语义：尾块（非整除）仅 copy 有效元素，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 fragment 尾块行为等价。

**重设计项 R2：NaN 谓词重设计（`T.isnan + fp32 宽化` → `vcmp(x,x,"ne")` 原生 dtype 自比较）**

- **源方案**：`T.isnan(T.cast(x, "float32"))`——把半精度输入宽化到 fp32，再调用 CUDA `T.isnan` 特殊函数得到 bool 谓词。
- **NPU 新算法**：不引入 fp32 中转、不调用标量 `T.isnan`；用**自比较** `T.vcmp(x_ub, x_ub, mask_ub, "ne")` 在**输入原生 dtype**（fp16/bf16/fp32）上直接得到 NaN 掩码（`mask_ub` dtype `bool`，片上使用）。依据：IEEE-754 下 NaN 是唯一不与自己相等的值（±inf 满足 `inf == inf`），故 `isnan(x) ⟺ (x != x)`；Ascend `npuir_cmp "ne"` 自比较为 IEEE **无序** lower（NaN≠NaN→true、±inf/有限→false），三 dtype 实测成立（pattern-library `CONST-vcmp-ne-nan-unordered`；`docs/Tilelang.language/比较操作/T.vcmp.md` §2.1 `"ne"` 合法、§2.2.1 fp16/fp32/bf16 均支持、§2.3 掩码输出 bool 且不可写回 GM）。
- **语义保持论证**：数学恒等 `isnan(x) ⟺ (x != x)`（§1.6.1 项 1 机器验证，fp16/bf16/fp32 全覆盖、0 违反）；宽化的消除等价（NaN 性在宽化下不变，§1.6.1 项 2 机器验证，0 违反）。边界语义核对：NaN → 掩码 true → 输出 1；±inf → `inf != inf` 为 false → 输出 0；±0 / 次正规 / 有限 → false → 输出 0，与源逐项一致。

**重设计项 R3：bool 输出表示重设计（`if_then_else(pred,1,0)` → 掩码→字节 cast 链）**

- **源方案**：`store_unary_bool_as_int8` 把谓词结果用 `T.if_then_else(pred, T.cast(1,"int8"), T.cast(0,"int8"))` 写成 int8 字节 0/1（fp32 谓词 + 逐元素选择）。
- **NPU 新算法**：谓词掩码（bool，片上）→ 0/1 字节由**两条 `T.vcast`** 完成：`T.vcast(mask_ub, tmp_f16_ub, "rint")`（bool i1 → fp16，得 1.0/0.0，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 支持 `i1 → f16`）+ `T.vcast(tmp_f16_ub, y_ub, "rint")`（fp16 → int8，同表支持 `f16 → i8`），写回 `int8` GM；host `view(torch.bool)`。之所以两步：vcast 表中**无** `i1 → i8` 直连，且 `T.vselect` 的 DataType 表不支持 int8/uint8 操作数（`docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1），故经 fp16 中转（0.0/1.0 精确可表示，无舍入）。
- **语义保持论证**：掩码 true ⟺ 字节 1 ⟺ `bool True`，掩码 false ⟺ 字节 0 ⟺ `bool False`；fp16 精确表示 0.0/1.0，`rint` 舍入恒等，字节位模式与源 `if_then_else(1,0)` 完全一致。int8 字节存储承载 bool 与源 `BOOL_STORAGE_DTYPE=int8` 同源（`view` 为元数据视图，不改内容）。

> **本算子三项重设计**：R1（并行结构）、R2（NaN 谓词）、R3（bool 输出表示）。源优化手段（§0.4）的意图承接：#1/#2 的「放大访存粒度、整块搬移」意图由 UB `T.copy` 分块搬移承接；#3 的「元素级并行放大吞吐」意图由 Vector 单元整块 v-prefix 计算承接；#4 的 fp32 宽化谓词由 R2 原生 dtype 自比较承接（**意图升级为等价化简，且消除宽化中转**，见 §1.6.1 项 1/2）；#5 的 bool int8 字节存储由 R3 等价承接。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/comparison.py`（`IsnanFwdKernel`）、`.../_base.py`（`FloatPredicateKernel` / `FloatUnaryKernel` / `UnaryKernel`）、`.../_builders.py`（`make_unary_regcopy`）、`.../_op_body.py`（`store_unary_bool_as_int8`）、`.../call_spec.py`（`UnaryPredicateFwdInterface`）。
- GPU manifest：`src/tileops/manifest/spec/elementwise_unary_math.yaml` 的 `IsnanFwdOp`（L424；`ref_api: torch.isnan`，workloads 16M/256M，`flops: prod(S)`）。
- NPU 侧工件：wrapper `examples/TileOPs/tileops/kernels/elementwise/isnan/isnan.py`、抽取 GPU 参考 `examples/TileOPs/tileops/kernels/elementwise/isnan/_isnan_fwd_kernels.py`、op `examples/TileOPs/tileops/ops/elementwise/isnan.py`、test `examples/TileOPs/tests/ops/test_isnan.py`、bench `examples/TileOPs/benchmarks/ops/bench_isnan.py`、manifest `examples/TileOPs/tileops/manifest/elementwise.yaml`。
- bool 输出先例：`examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel_DESIGN.md`（int8 字节存储 + host view，CG-2026-0019）。
- 源仓参考实现 / 测试基准：`torch.isnan`（test 的 `ref_program = torch.isnan(x.detach().cpu().float())`）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`torch.isnan`），**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

IsnanFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `isnan_fwd_kernel`。

### 1.2 功能描述

逐元素 NaN 谓词：`bool = isnan(x)`（NaN 元素为 True，其余为 False），用于 float16/bfloat16/float32 张量；输入展平为 1-D 后逐元素判定，输出 bool（以 int8 字节存储落地），输出与输入同 shape。

### 1.3 数学公式

$$
y_i = \operatorname{isnan}(x_i), \qquad
y_i = \begin{cases}
1, & x_i \ \text{is NaN} \\
0, & \text{otherwise}
\end{cases}
$$

（bool 语义：1 = True、0 = False；谓词不经浮点算术，无舍入。）

### 1.4 算法描述

单步逐元素谓词映射：展平为 1-D `N` 后，对每个元素独立判定是否为 NaN。NPU 实现路径 = 原生 dtype 上 `vcmp(x,x,"ne")` 自比较得到 NaN 掩码（bool），再经 `vcast` 链（bool→fp16→int8）落到 0/1 字节，写回 int8 GM；host `view(torch.bool)`。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**；谓词为精确布尔运算，**无 fp32 宽化中转**（这是本算子区别于同族浮点算术激活迁移的关键——NaN 是唯一与自身不等的值，检测不依赖精度）。本算法与 §1.6.0 调研选定算法一致。与源算法结构差异：(a) 并行组织 GPU `threads×npt` → NPU 一维 Kernel + `block_size` 折叠（§0.6 R1）；(b) NaN 谓词 `T.isnan+宽化` → `vcmp(x,x,"ne")` 原生 dtype（§0.6 R2）；(c) bool 结果 `if_then_else` → 掩码→字节 cast 链（§0.6 R3）。逐元素语义与源码逐步一致。

### 1.5 数据流图

```
输入 input [*S] --host 展平--> x_flat (N,) --T.copy--> UB[x_ub]
  --vcmp(x,x,"ne")--> UB[mask_ub](bool) --vcast(bool→f16)--> UB[tmp_f16]
  --vcast(f16→i8)--> UB[y_ub](int8) --T.copy--> y_flat (int8) --host view(bool)/回塑--> 输出 [*S]
```

### 1.6 算法调研与优化分析

> 设计第一优先级：先调研四问（§1.6.0），再数学等价优化公式（§1.6.1），再向量化替代分析（§1.6.2），最后向量化轴/布局决策（§1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。本节分析对象为 §1.6.0 选定、经 §0.6 R1–R3 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：NaN 谓词 `isnan` 的算法族（所有能算出相同 bool 结果的算法结构）。源算法/输入公式只是基线候选之一。本算子为**单步逐元素谓词**，按 `algorithm-research.md` §2 判为**轻量调研**（四问各一行结论 + 依据），但四问不可跳过。对照 `algorithm-candidates.md` 的 elementwise 命中行（R1：算符降代换 / 公共子表达式消除；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返——本算子无中间 GM 往返）。

**R1 等价化简公式候选**（基线候选在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源公式） | `T.isnan(T.cast(x, "float32"))`（宽化 + 特殊函数谓词） | — | — | ✅（基线） |
| 1 | 自比较 NaN 谓词 | `(x != x)`（`vcmp(x,x,"ne")`，原生 dtype） | 数学恒等（IEEE：NaN 唯一不等于自身；±inf==±inf）。§1.6.1 项 1 机器验证 | 少一次宽化 cast + 少一个 fp32 中转缓冲；无特殊函数 | ✅ |
| 2 | 位域判定（exp 全 1 且 mantissa≠0） | 整型位运算测试指数/尾数 | 数学恒等（IEEE 编码定义），但需 reinterpret + 位掩码 + 比较多步 | op 数增多，无收益 | ❌（无结构优势） |
| 3 | `x - x != 0` 变体 | `(x - x) != 0` | **不等价**：`inf - inf = NaN`，会把 ±inf 误判为 NaN | — | ❌（语义错误，剔除） |

**R2 在线算法**：**无「在线/离线」之分**（天然单遍）——NaN 谓词是逐元素映射，每个输出元素只依赖对应输入元素，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（`algorithm-research.md` §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导，不得只比 FLOPs）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线（宽化 + isnan） | N（1 谓词/elem，manifest 口径 `prod(S)`） | N·in_bytes + N·1（读 N + 写 N 字节，无中间 GM 往返） | 1 | O(block_size)（UB 驻留；宽化额外一份 fp32 缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1（自比较） | N | N·in_bytes + N·1 | 1 | O(block_size)（UB 驻留，**省去宽化缓冲**） | 完全并行 |
| 候选 2（位域） | ~3N（位与/比较多步） | N·in_bytes + N·1 | 1 | O(block_size) | 完全并行 |

**结论**：候选间访存 / 遍数 / 中间缓冲量级相同（带宽受限，访存主导），候选 1 在 R4 因**少一次宽化 cast、少一个 fp32 中转缓冲、无特殊函数**胜出，且数学恒等（§1.6.1），无并列争议；候选 2 无结构优势、候选 3 语义错误剔除。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰须附本地 docs / examples 目录佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线 | 逐元素谓词 → Vector；NPU 无向量 `T.isnan`，需自比较替代 | 工作集 = O(block_size) UB 驻留，≤64KB < 192KB（§4.5） | 尾轴连续，block_size 为 32B 整数倍且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓ | 循环边界 `ceildiv(N, block_size)` / `ceildiv(num_logical, num_kernels)` 均为编译期常量（N/block_size 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ❌ 淘汰（无向量 `T.isnan`；见 `docs/Tilelang.language/比较操作/T.vcmp.md`） |
| 候选 1（自比较 `vcmp "ne"`） | 逐元素比较 → Vector；`VCmpOp` 原生支持 fp16/fp32/bf16（`T.vcmp.md` §2.2.1） | 同上（且省宽化缓冲） | 同上 | 同上 | 同上 | ✅ **采纳（取代基线）** |
| 候选 2（位域） | 逐元素位运算 → Vector；reinterpret 需 vbitcast（仅 fp16/fp32，bf16 ×，`docs/Tilelang.language/数据类型转换操作/T.vbitcast.md` §2.2.1），bf16 路径受阻 | 同上 | 同上 | 同上 | 同上 | ❌ 淘汰（bf16 无 vbitcast；op 数更多，无收益） |

**定量 roofline（D-2）**：候选工作集 < UB、尾轴 32B 对齐、约束可静态化，主选（候选 1）纯 Vector 亲和，无候选因违反容量/带宽量级被淘汰。引用 `constants.md` 实测常数条目（`CONST-capacity-910B2C` UB 192KB、`CONST-vector-launch-overhead` 发射 ~0.5µs/op、`CONST-mte2-degradation` MTE2 带宽档、`CONST-vcmp-ne-nan-unordered` vcmp "ne" 语义）计算估算下界（见本节末「设计期估算」）。

**调研结论**：选定算法族 = **原生 dtype 自比较 NaN 谓词 + 掩码→字节落盘**——`vcmp(x,x,"ne")`（取代 `T.isnan(T.cast(x,"float32"))`）+ `vcast` 链（取代逐元素 `if_then_else`）。关键依据：R1 候选 1 数学恒等（§1.6.1 机器验证 0 违反）、候选 2 无结构优势且 bf16 受阻、候选 3 语义错误；R2 天然单遍；R3 访存/遍数/缓冲量级相同而候选 1 省宽化；R4 纯 Vector 亲和（`VCmpOp` 三 dtype 支持、UB 驻留、32B 对齐、静态边界）。与基线结构差异：**谓词由「宽化 + 特殊函数」降为「原生 dtype 自比较」**。「无更优替代（就复杂度量级而言）」调研范围：`algorithm-candidates.md` elementwise 命中行、恒等变形自查（R1 候选 2/3）、pattern-library `CONST-vcmp-ne-nan-unordered` / `CASE-elementwise-hardsigmoid-migration` 同族先例、源码 §0.3/§0.4。源优化手段（§0.4）中 #4「fp32 宽化谓词」被调研升级为原生 dtype 自比较（意图承接见 §0.6/§11.2），#5「bool 字节存储」等价承接，其余无被替换。

**设计期估算**（D-2，轻量调研豁免容量项，流量项必算）：

```
估算下界(manifest 代表 workload，流量项主导):
  16M f16/bf16 [4096,4096]: max(流量 ≈ 41.7µs, 发射 ≈ 1.5µs) ≈ 41.7µs
  16M f32 [4096,4096]:      max(流量 ≈ 69.4µs, 发射 ≈ 1.5µs) ≈ 69.4µs
  256M f16/bf16 [16384,16384]: max(流量 ≈ 1094µs, 发射 ≈ 1.5µs) ≈ 1.09ms
  流量项: N×(in_bytes + 1) ÷ 有效带宽；有效带宽 = CONST-mte2-degradation 档位 × 40 向量核
    N = 16,777,216：f16/bf16 3N ≈ 50.3MB / 1.208TB/s；f32 5N ≈ 83.9MB / 1.208TB/s
    N = 268,435,456：f16/bf16 3N ≈ 805MB / 0.736TB/s
  发射项: 计算链 ≈ 3 个向量 op（vcmp + 2×vcast）× 0.5µs/op ≈ 1.5µs（CONST-vector-launch-overhead）
  容量项: 轻量调研豁免（单步逐元素，无 fixed-loop 跨引擎往返；CONST-store-fixpipe-gm-only 不适用——无 Cube）
  未实测假设: CONST-mte2-degradation 曲线绑定 fp16 / 48 vector core / 4-set 轮换；本设计 40 向量核（实查 20×2）
    与 bf16/fp32 混合流量，档位按「总流量档」近似引用（元素字节已计入字节口径），标注为未实测外推；L2 驻留
    workload（smoke-1m 复用同输入）会高估时延（该条目 L2 口径注：同张量复用带宽偏高）
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（经 §0.6 R1–R3 重设计落地）。采纳项经 `verify_equiv.py` 机器验证（结果表见下）。谓词为精确布尔运算，**无浮点容差**（输出 bool，比较用精确相等）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | 自比较 NaN 谓词（消除特殊函数） | `pred = T.isnan(x)` | `pred = (x != x)`（`vcmp(x,x,"ne")`） | 数学恒等：IEEE-754 下 NaN 是唯一不与自己相等的值（±inf 满足 `inf == inf`、有限/±0/次正规均相等），故 `isnan(x) ⟺ (x != x)`。`verify_equiv.py` 项 1：fp32 4097+ 元素（随机 + NaN/±inf/±0/次正规/极值角点）mismatches=0，EQUIV_PASS | 消去特殊函数谓词调用；谓词由比较指令单点完成 |
| 2 | 原生 dtype 谓词（消除 fp32 宽化中转） | `pred = T.isnan(T.cast(x,"float32"))` | `pred = (x != x)`（在输入 dtype fp16/bf16/fp32 上直接比较） | 容差内（实为精确）等价：IEEE NaN 表示在宽化下不变（fp16/bf16 的 NaN 宽化到 fp32 仍为 NaN，非 NaN 仍非 NaN），故宽化不改变谓词结果。`verify_equiv.py` 项 2：fp16/bf16/fp32 各 4097+ 元素，mismatches=0（三 dtype 全 EQUIV_PASS） | 每元素省一次宽化 cast（向量 op）与一个 fp32 中转缓冲（block_size×4B） |

**机器验证结果表**（`verify_equiv.py`，torch CPU 精确布尔对照 + 随机/角点，已执行）：

| 采纳项 | 违反数（mismatches） | 结论 |
|--------|---------------------|------|
| 1. `isnan(x)` → `(x != x)`（fp32 自比较恒等） | 0 | EQUIV_PASS |
| 2. `isnan(cast(x,f32))` → `(x != x)`（fp16/bf16/fp32 原生 dtype） | 0 | EQUIV_PASS |

逐 dtype 结果：fp32 = 0 / float16 = 0 / bfloat16 = 0（全部 EQUIV_PASS）。

**优化结论**：采纳 2 项。优化后公式为（供 §3.1 公式拆解使用）：

$$
pred_i = (x_i \neq x_i), \qquad
y_i = \begin{cases}
1, & pred_i = \text{true} \\
0, & pred_i = \text{false}
\end{cases}
$$

（在输入原生 dtype 上比较，无宽化中转。无进一步优化空间：单次逐元素谓词映射，无公共子表达式（`x` 仅出现一次于比较）、无超越函数、无中间 GM 往返；掩码→字节的落盘是表示层必需步骤，非可省优化项。）

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | NaN 谓词 | `T.Parallel` 逐元素标量 `T.isnan`（宽化 + 特殊函数） | `T.vcmp(x_ub, x_ub, mask_ub, "ne")`（`docs/Tilelang.language/比较操作/T.vcmp.md` §2.1/§2.2.1/§2.4；pattern-library `CONST-vcmp-ne-nan-unordered` 三 dtype 实证；hardsigmoid 同款 vcmp 用法） | ✅ | — |
| 2 | 谓词掩码 → 0/1 字节 | 逐元素 `T.if_then_else(pred, 1, 0)`（int8） | `T.vcast(mask_ub, tmp_f16_ub, "rint")`（bool→f16）+ `T.vcast(tmp_f16_ub, y_ub, "rint")`（f16→i8）（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 支持 `i1→f16` 与 `f16→i8`） | ✅ | — |
| 3 | 数据搬入 GM→UB | 逐元素标量 load | `T.copy(x[offset:offset+tail], x_ub[0:tail])`（双显式 slice，`docs/Tilelang.language/内存操作/T.copy.md`） | ✅ | — |
| 4 | 数据搬出 UB→GM | 逐元素标量 store | `T.copy(y_ub[0:tail], y[offset:offset+tail])`（双显式 slice） | ✅ | — |
| 5 | block 起始索引 `offset = block_id * block_size` | 标量乘加 | 无向量 API | ❌ | block 级任务映射索引，每 block 少量标量，非逐元素热点（可接受理由类别①：block 级索引 / 任务映射计算） |
| 6 | 尾块大小 `tail_size = T.min(block_size, remaining)` | 标量 min | 无（`T.min` 标量形） | ❌ | 依赖动态 shape 的边界处理，每 block 一次，非逐元素热点（可接受理由类别④：动态边界；同族 mish / hardsigmoid / bitwise_not 同款尾块处理） |

**向量化结论**：逐元素计算（`vcmp` 谓词 + `vcast` 掩码落盘）与数据搬入搬出（`T.copy`）已**全部向量化**（替代 API 均有 `docs/Tilelang.language/比较操作/`、`/数据类型转换操作/`、`/内存操作/` 与 pattern-library 佐证）。保留 2 处标量：block 起始索引（block 级任务映射）与尾块大小（动态边界），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策

**单轴豁免说明**：本算子输入展平为 1-D `(N,)`，仅一个非 batch 维（展平轴），无 ≥2 个非 batch 维度可作向量化轴选择——逐元素谓词在**唯一的展平轴**上向量化（Vector 单元 128bit：fp16/bf16 ×8、fp32 ×4；掩码/字节路径按 1B 元素处理），不存在广播轴、窗口轴、归约轴或重排布局候选。按 SKILL.md Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」。

- **向量化轴** = 展平轴（stride=1 连续，无累加链、无跨步系数）。
- **整除性**：block_size=8192，fp16/bf16 向量宽 ×8、fp32 ×4 均整除（8192 = 1024×8 = 2048×4）✓；尾块 `tail_size` 动态，仅 copy 有效元素，垃圾 lane 不写回（与同族迁移同款）。
- **repack 代价**：无（输入即 1-D 连续，无需转置/重排；bool→f16→int8 为元素级 dtype 转换，非布局重排）。
- **UB 容量影响**：驻留缓冲 = O(block_size)，预算见 §4.5（峰值 ≤64KB < 192KB）。
- **迁移对照**：GPU 源算子的并行轴选择（`threads` 线程 × `npt` 每线程元素）是 CUDA 线程模型，与 NPU Vector 展平轴不对应——本设计独立按展平轴向量化（§0.6 R1）。

**布局决策结论**：选定「I/O 原生 1-D 连续布局 + 展平轴向量化」，无 repack、无重排候选（单轴豁免）。该结论同步落入 §3.3 伪代码与 §6 循环结构（向量维 = 展平轴，buffer 均为 1-D）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（迁移默认）。

### 2.2 选型理由

纯 Vector 逐元素谓词算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。算子体为 3 个向量 op（`vcmp` + 2×`vcast`），无需 Expert 的手动同步与多级 buffer 控制。与同族 elementwise（mish / hardsigmoid / bitwise_not）先例一致；NPU wrapper `IsnanFwdKernel` 声明 `ascend_mode = "Developer"`（`examples/TileOPs/tileops/kernels/elementwise/isnan/isnan.py`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix API 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（`T.vcmp` + `T.vcast`），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 set_flag/wait_flag） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_ub = load(x)` | GM→UB 搬入（原生 dtype） |
| 2 | `mask = (x_ub != x_ub)` | 自比较 NaN 谓词（原生 dtype，无宽化） |
| 3 | `t = cast_f16(mask)` | 掩码 → fp16（true→1.0、false→0.0） |
| 4 | `y = cast_i8(t)` | fp16 → int8 字节（1/0） |
| 5 | `store(y)` | UB→GM 搬出（int8 字节；host 侧 view bool） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 数据搬入 GM→UB | `T.copy(x[offset:offset+tail], x_ub[0:tail])` | 双显式 slice（`TRAP-T-copy-region-semantics` 推荐形态） | Developer |
| 1 | `mask = (x != x)` | `T.vcmp(x_ub, x_ub, mask_ub, "ne")` | A=B=`x_ub` 自比较；C=`mask_ub`（bool） | Developer |
| 2 | 掩码 → fp16 | `T.vcast(mask_ub, tmp_f16_ub, round_mode="rint")` | `i1 → f16`（true→1.0/false→0.0） | Developer |
| 3 | fp16 → int8 字节 | `T.vcast(tmp_f16_ub, y_ub, round_mode="rint")` | `f16 → i8`（1/0 精确） | Developer |
| 4 | 数据搬出 UB→GM | `T.copy(y_ub[0:tail], y[offset:offset+tail])` | 双显式 slice（int8 字节） | Developer |

> 无 `vadd/vmul/vclamp` 等算术 op（无浮点算术）；无 `vbrc`（无 NaN 常数广播——本算子输出的是「是否 NaN」的 bool，不是 NaN 值本身）；无标量 `T.isnan`（弃用，仅标量谓词）。

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型以**第二个位置参数**传入（不用 `dtype=` 关键字），避免产生 `name 'in_dtype' is not defined` 的 false alarm。输出以 `int8` 字节存储承载 bool（bool i1 不可 load/store GM，见 `docs/Tilelang.language/比较操作/T.vcmp.md` §2.3）；host 侧 `view(torch.bool)`。

```python
@tilelang.jit(out_idx=[1], target="npuir")
def isnan_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    out_dtype = output_dtype or "int8"          # bool byte storage (bool i1 cannot load/store GM)
    num_logical = T.ceildiv(N, block_size)      # compile-time logical tile count
    # Physical Vector cores (section 5.5: AI cores x 2 for a pure Vector op).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_logical, num_kernels)   # static bound

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # 1. UB buffers (v-prefix APIs require UB residency).  All buffers are
            #    allocated unconditionally (a conditional alloc is rejected by the
            #    TVM script parser).
            x_ub = T.alloc_ub((block_size,), dtype)
            mask_ub = T.alloc_ub((block_size,), "bool")     # vcmp output mask
            tmp_f16_ub = T.alloc_ub((block_size,), "float16")  # mask -> f16 (1.0/0.0)
            y_ub = T.alloc_ub((block_size,), out_dtype)     # int8 output byte tile

            # 2. Persistent in-core serial tile loop (static bound, grid-stride,
            #    if-guard masks out-of-range tiles).
            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # 3. Load GM -> UB (only valid elements copied in).
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 4. NaN predicate: self-compare x != x (IEEE unordered,
                    #    NaN != NaN -> true; +-inf/inf == inf -> false).  Native dtype.
                    T.vcmp(x_ub, x_ub, mask_ub, "ne")

                    # 5. Mask -> byte: bool -> f16 (1.0/0.0) -> int8 (1/0).
                    T.vcast(mask_ub, tmp_f16_ub, round_mode="rint")
                    T.vcast(tmp_f16_ub, y_ub, round_mode="rint")

                    # 6. Store UB -> GM (int8 bytes; host views bool).
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main
```

> 注：`T.Tensor((N,), dtype)` 用位置参数传 dtype（避免 false alarm，见本节开头约定）。bool 输出经 `int8` 字节存储：Op / Kernel 层在 host 侧 `result.view(torch.bool)` 回转（元数据视图，内容不变；见 §11.4）。输入 dtype fp16/bf16/fp32 均在原生 dtype 上比较，无宽化。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vcmp` | `docs/Tilelang.language/比较操作/T.vcmp.md`（`T.vcmp(A,B,C,cmp_mod)`；§2.1 `"ne"` 合法；§2.2.1 fp16/fp32/bf16 均 √；§2.3 掩码输出 bool、不可写回 GM）+ `tilelang/language/customize_npuir.py` `npuir_cmp`（源码断言 `cmp_mod ∈ {"eq","ne","lt","gt","ge","le"}`）+ pattern-library `CONST-vcmp-ne-nan-unordered`（三 dtype NaN 自比较实测） | ✅ 已验证（docs 读 + 源码读 + 实测常数条目） |
| `T.vcast`（bool→f16） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`i1 → f16`，rint） | ✅ 已验证（docs 读） |
| `T.vcast`（f16→i8） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16 → i8`，rint） | ✅ 已验证（docs 读） |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md` + 同族先例（`TRAP-T-copy-region-semantics`：src/dst 双显式 slice 为推荐通用形态） | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | `examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel.py` / `.../bitwise_not/.../bitwise_not_fwd_kernel.py` | ✅ 已验证 |
| `T.alloc_ub` | `examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel.py` | ✅ 已验证 |
| `T.ceildiv` / `T.min`（标量） | `docs/Tilelang.language/数学操作/T.ceildiv.md` + 同族尾块先例 | ✅ 已验证 |
| `T.isnan` | 仅标量谓词（`inspect.signature(T.isnan)` = `(x, span=None)`）；无向量掩码形式 | ❌ **弃用（不用于本设计）** |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 `T.Kernel(ceildiv(N, block_size))`） | 一维 `T.Kernel(num_kernels, is_npu=True)`，无三维 |
| GPU 专用 API | Yes（`T.isnan` 特殊函数 + `T.cast` 宽化；`T.Parallel(threads,npt)` CUDA 语义；bool i1 GM 存取） | 全部改写为 NPU v-prefix API + UB 路径（§0.5/§0.6 R1–R3）；bool 走 int8 字节存储（§3.3） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(N, block_size), threads=128)`（block+thread 两级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(threads, npt)` 显式元素循环 | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| NaN 谓词 | `T.isnan(T.cast(x,"float32"))`（宽化 + 特殊函数） | `T.vcmp(x,x,"ne")` 原生 dtype 自比较 | §0.6 R2 / §1.6.1 |
| bool 输出 | `if_then_else(pred,1,0)` int8（`store_unary_bool_as_int8`） | 掩码 → f16 → int8 `vcast` 链 | §0.6 R3 |
| bool 存储 | `BOOL_STORAGE_DTYPE = "int8"` + host `view(bool)` | `int8` 字节存储 + host `view(torch.bool)` | §0.5 等价替换 / §3.3 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md` | 高度相似（同族 float 逐元素迁移；vcmp 掩码先例） | 一维 Kernel + block_size 折叠、分核三要素、UB 预算口径、persistent 尾块、`vcmp(x,x,"ne")` 用法 |
| `examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel.py` | 高度相似（vcmp 掩码 + vselect 的 NPU 落地骨架） | `T.vcmp`/`T.alloc_ub(...,"bool")`/persistent 核内串行/`T.min` 尾块 |
| `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel_DESIGN.md` | 高度相似（bool 输出 int8 字节存储 + host view，CG-2026-0019） | 输出字节存储契约、host `view`、拒绝非目标 dtype、UB 预算 |
| `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py` | 相似（bool 输出的 NPU 实现骨架） | `int8` 字节存储、`T.Kernel(is_npu=True)`、`NPUUtils.get().get_aicore_num()` |
| `examples/TileOPs/tileops/kernels/elementwise/isnan/_isnan_fwd_kernels.py` | 相似（迁移对象抽取件） | 谓词语义与 bool-as-int8 落盘约定（CPU/GPU reference，需 NPU 重写） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `input` | `[*S]`（展平 1-D `N`） | `float16 \| bfloat16 \| float32` | 输入张量，host 展平视图 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | 与 input 相同 | `bool`（kernel 内以 `int8` 字节存储） | 输出张量，host `view(torch.bool)` 后回塑 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | 输入 dtype（fp16/bf16/fp32） | UB | 输入 tile（GM→UB） |
| `mask_ub` | `(block_size,)` | bool | UB | `vcmp(x,x,"ne")` 输出掩码（片上，不落 GM） |
| `tmp_f16_ub` | `(block_size,)` | float16 | UB | 掩码→字节中转（1.0/0.0） |
| `y_ub` | `(block_size,)` | int8 | UB | 输出字节 tile（UB→GM） |

### 4.4 内存搬运路径

```
GM[x] --T.copy--> UB[x_ub] --vcmp(x,x,"ne")--> UB[mask_ub](bool)
  --vcast(bool->f16)--> UB[tmp_f16_ub] --vcast(f16->i8)--> UB[y_ub] --T.copy--> GM[y]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `x_ub` | (8192,) | fp16/bf16 / fp32 | 16384 / 32768 |
| `mask_ub` | (8192,) | bool (1B) | 8192 |
| `tmp_f16_ub` | (8192,) | float16 | 16384 |
| `y_ub` | (8192,) | int8 (1B) | 8192 |
| **总计（fp16/bf16 路径）** | | | **49152（48KB）** |
| **总计（fp32 路径）** | | | **65536（64KB）** |

对照目标平台 UB 容量 **192KB**（`constants.md` `CONST-capacity-910B2C`：UB=192KB/AIV）：

- 显式 alloc 结构按该条目 ×1.10–1.12 上偏系数：峰值 64KB ×1.12 ≈ 72KB < 192KB ✓，余量 ~63%。
- 即便按 auto-multi-buffer 保守膨胀（`TRAP-UB-multibuffer-inflation` ~1.7x）：64KB ×1.7 ≈ 109KB < 192KB ✓。
- 全部 dtype 路径均不溢出，**无需进一步调小 block_size**。

### 4.6 动态轴定义

无动态轴。`N` 为 JIT 编译期参数（`isnan_fwd_kernel(N, dtype)` 每 shape 一特化），`shape`（`T.int32`）为运行时传入的实际元素数（用于尾块 `remaining = shape - offset`，与同族迁移同款）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素谓词（`vcmp`）与 dtype 转换（`vcast`），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_size = 8192                              # 同族 elementwise 实测最优
num_logical = T.ceildiv(N, block_size)         # 逻辑 tile 数
```

- **选择理由**：8192 为同族 elementwise（mish / hardsigmoid / hardswish / bitwise_not）实测最优 block_size；32B 对齐（8192 × 1/2/4B 均为 32B 整数倍）；UB 预算峰值 64KB < 192KB（§4.5）；宽块摊薄向量发射开销（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。

### 5.3 约束分析

- **对齐约束**：block_size=8192，fp16/bf16 尾轴 16KB、fp32 尾轴 32KB，均 32B 对齐且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓。
- **UB 容量**：峰值 64KB < 192KB（§4.5）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾块**：`tail_size = T.min(block_size, remaining)`，仅 `T.copy` 有效元素进/出；v-prefix 对全 buffer 计算，尾部垃圾元素不写回（与同族迁移同款）。
- **极小 shape**（如 N < block_size）：`num_logical=1`，`num_kernels=1`，单核单 tile 处理，无 oversubscription。
- **bool 字节视图**：输出在 Op / Kernel 层与 kernel 之间以 int8 视图传递，kernel 内不出现 i1 GM 访问。

### 5.5 分核策略（物理核数适配）

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（Cube/混合直接用返回值、纯 Vector 翻倍），禁止文档假设/经验值。

- **物理核数（实查）**：`NPUUtils.get().get_aicore_num()` → **20**（2026-10-14 于本任务环境实查）。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。查询代码与实际返回值记录于此：

```python
from tilelang.utils.npu_utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 翻倍：40
```

（注：`constants.md` `CONST-aicore-910B2C` 记录 910B2C 为 24 核，本环境设备实查返回 20——设备型号差异，一律以实查为准；同现象在本工作区 hardsigmoid / hardswish / bitwise_not 任务复现。）

- **逻辑核数**：`num_logical = ceil(N / block_size)`，随 workload（NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `IsnanFwdOp` 条目）变化：

| workload | shape | dtype | N = prod(S) | block_size | 逻辑核数 ceil(N/bs) | 每核串行 tile 数 ceil(num_logical/40) |
|----------|-------|-------|-------------|-----------|---------------------|--------------------------------------|
| 16M | [4096,4096] | float16 | 16,777,216 | 8192 | 2048 | 52 |
| 16M | [4096,4096] | bfloat16 | 16,777,216 | 8192 | 2048 | 52 |
| 16M | [4096,4096] | float32 | 16,777,216 | 8192 | 2048 | 52 |
| 256M | [16384,16384] | float16 | 268,435,456 | 8192 | 32768 | 820 |
| 256M | [16384,16384] | bfloat16 | 268,435,456 | 8192 | 32768 | 820 |

- **规模判定与分核方案**：workload 全部落在**极大规模**档（逻辑核数 2048 / 32768 ≫ 40）。设计采用**统一 persistent 结构**覆盖全部档位（成熟 elementwise 形态，`PL-1.6-copy-floor`「grid-stride 48 核聚集窗口实测优于每核连续分块 +3.2%」）：

  - **小规模（逻辑核数 ≤ 40，即 N ≤ 327680）**：`num_kernels = num_logical`，每核 ≤1 tile，`num_local_tasks=1`，无串行循环、无负载不均——结论「**无需适配**」（逻辑核数 ≤ 物理向量核数）。
  - **中等/极大规模（逻辑核数 > 40，如 16M 2048、256M 32768）**：固定启动内核数 = 40（= 物理向量核数），核内 `T.serial` **串行**处理 `num_local_tasks = ceil(num_logical/40)` 个 tile（16M：52；256M：820），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均；循环边界 `num_local_tasks` 为编译期常量（N / block_size / 40 均编译期已知），满足「**核内串行**边界静态」。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（`vcmp` / `vcast`） | 块内逐元素由 Vector 单元并行（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # UB buffer 分配 ...
    for i in T.serial(num_local_tasks):        # 核内串行 tile 循环（静态边界）
        block_id = i * num_kernels + cid
        if block_id < num_logical:             # 尾 tile guard
            # 装载 -> vcmp 谓词 -> vcast 掩码落盘 -> 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对计算链（`vcmp` + 2×`vcast`）采用 guard-free 2-stage `T.Pipelined(num_local_tasks, num_stages=2)` 双缓冲流水，将向量链隐藏于 MTE2 窗口（`PL-1.6-copy-floor`：≥16M 档计算链被完全隐藏，预计收益有限）。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 shape：`tail_size = T.min(block_size, remaining)`；`T.copy` 仅装载/写回 `[0:tail_size]` 有效元素；v-prefix 对全 block_size buffer 计算，垃圾尾部不写回（与同族迁移同款）。`block_id < num_logical` guard 覆盖最后一波不完全 tile。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer）。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器自动管理 GM↔UB 搬运与向量计算间的依赖（`T.copy` 与后续 `vcmp`/`vcast` 之间的数据依赖由自动同步保证），无跨核协作（逐元素无跨元素依赖，各核独立）。

### 7.3 pass_configs 配置

Developer 模式默认配置，无需额外 `pass_configs`。环境变量 `TILELANG_ASCEND_MODE = "Developer"`（可选，默认即 Developer；wrapper 声明 `ascend_mode = "Developer"`）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（`torch.isnan`，精确 bool），移植源仓参考实现，**不复刻 §0.6 的 NPU 算法**（不含 `vcmp` / `vcast` / 字节存储 / persistent 结构）。

```python
import torch

def golden_isnan(x):
    """NaN predicate reference: y = torch.isnan(x)  (bool output, exact).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_isnan.py: ref_program = torch.isnan(x.detach().cpu().float()),
    atol = rtol = 0).  Independent of the NPU algorithm (no vcmp self-compare /
    vcast byte chain / int8 byte storage / persistent structure).  Runs on CPU
    and does not require an NPU device.
    """
    return torch.isnan(x.detach().cpu().float())
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 / bfloat16 / float32 | 0.0 | 0.0 |

（与源仓测试 `_get_tolerances` 一致：`torch.isnan` 为精确谓词，输出 bool，比较用精确相等，max_abs_err 必须为 0。§1.6.1 机器验证 mismatches=0。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`N=1048576`，三 dtype（float16 / bfloat16 / float32）各一例，输入含随机值 + 注入 NaN/±inf（参照源仓 workload 的 special-value 抽样），vs golden（`torch.isnan`）以**精确相等**（atol=rtol=0，`torch.equal` 或 `torch.isnan(out) == golden` 全等）比对；另含拒绝非 float dtype 的负向用例（`IsnanFwdKernel(N_total=16, dtype=torch.int32)` 应抛 `ValueError`）。完整分层套件（L1 非整除 / 多 shape、L2 边界 size、Boundary 特殊值）交由 `tilelang-op-develop` 规划，本设计不枚举。

**特别说明（边界用例）**：Boundary 用例须覆盖 NaN（→True）、+inf / -inf（→False，验证不被误判）、±0（→False）、次正规（→False）与极值，全部要求精确相等；输出 bool 用例须经 int8 字节存储路径验证（`view(torch.int8)` / `view(torch.bool)` 往返）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| bool 不可直接 load/store GM | bool i1 无法 load/store GM（`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3），输出必须经 int8 字节存储；Op / Kernel 层 `view(torch.bool)` 为元数据视图，内容不变（§11.4） |
| 无向量 `T.isnan` | `T.isnan` 为标量谓词（无向量掩码形式），NaN 检测必须走 `vcmp(x,x,"ne")` 自比较（§0.6 R2） |
| vcmp 掩码不可落 GM | `vcmp` 输出的 bool 掩码仅片上可用，须经 `vcast` 转字节后写回（§0.6 R3） |
| vcast 无 `i1→i8` 直连 | 掩码→字节须 `i1→f16→i8` 两步（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1）；`vselect` 不支持 int8/uint8 操作数（`docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1），故不能直接 select 出 int8 |
| UB 预算 | 峰值 64KB（fp32）< 192KB；显式 alloc ×1.12 ≈ 72KB、保守 ×1.7 ≈ 109KB，均安全 |
| 分核 | 物理核数实查 20（纯 Vector ×2 = 40）；与 `constants.md` `CONST-aicore-910B2C`（24）不一致——以实查为准 |
| 无 fp32 宽化 | NaN 检测在原生 dtype 上完成（§1.6.1 项 2），**禁止**引入 fp32 中转（会徒增向量 op 与缓冲） |
| `vcmp "ne"` 无序语义 | 依赖 Ascend `npuir_cmp "ne"` 按 IEEE 无序实现（NaN≠NaN→true）；GPU 后端相反（有序，勿外推），`CONST-vcmp-ne-nan-unordered` 三 dtype 已实证 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bool 直接作为 GM buffer dtype | `T.Tensor((N,), "bool")` 或 bool 掩码直接写 GM | 编译失败 / 值错误（i1 不可 GM 存取） | bool 走 int8 字节存储 + host `view(bool)`（§3.3/§11.4） |
| 用 `T.isnan` 求值 | 试图对向量 buffer 调 `T.isnan` | 仅标量谓词，无向量形式，无法向量化 | 用 `T.vcmp(x_ub, x_ub, mask_ub, "ne")`（§0.6 R2） |
| 直接 vselect 输出 int8 | `T.vselect(mask, one_i8, zero_i8, out_i8)` | int8/uint8 操作数不受支持 | 经 f16 中转：`vcast(mask→f16)` + `vcast(f16→i8)`（§0.6 R3） |
| 误用 `x - x != 0` | 用减法变体判 NaN | ±inf 误判为 NaN（inf−inf=NaN） | 用自比较 `x != x`（§1.6.0 R1 候选 3 已剔除） |
| 尾块越界 | `T.copy` 用 base+size 形态写尾块 | MTE 越界写坏相邻 tensor | 双显式 slice `T.copy(src[0:tail], dst[0:tail])`（`TRAP-T-copy-region-semantics`） |
| `dtype=` 关键字 false alarm | `T.Tensor((N,), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（§3.3 约定） |

### 9.3 特殊场景处理

- **非整除 shape**（如 N 奇数）：尾块 `tail_size` + `block_id < num_logical` guard（§6.4）。
- **极小 shape**（N < block_size）：`num_logical=1`，单核单 tile。
- **±inf 输入**：自比较 `inf == inf` → 掩码 false → 输出 0（与 torch 一致）。
- **±0 / 次正规**：`x == x` → false → 输出 0。
- **float dtype 之外的输入**：Op / Kernel 层显式拒绝（`ValueError`，见 test 文件的负向用例）。
- **bool 输出**：int8 字节视图往返（§11.4）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/isnan/isnan_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，结果内嵌）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档 |
| `verify_equiv.py` | 已完成 | §1.6.1 等价性验证（2 项 EQUIV_PASS） |
| `isnan_fwd_kernel.py` | 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_isnan.py` | 待实现（Stage 3） | 精度测试 |

### 10.3 命名规范

- 项目目录名：`isnan`；算子目录名：`isnan_fwd_kernel`；接口函数：`isnan_fwd_kernel`（manifest 键 `IsnanFwdOp`）。

### 10.4 实现顺序

1. 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. Golden 函数（§8.1）
3. 算子实现（`isnan_fwd_kernel`）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）

---

## 11. 迁移决策与设计一致性核对

> 本节逐项核对 §0 迁移决策与 §1–§10 设计的一致性，并说明源优化手段的意图承接（防止「§0 说重设计、§3 又照抄源方案」的脱节）。

### 11.1 §0 → §1–§10 一致性

| §0 决策 | 后续落地 | 一致？ |
|---------|---------|--------|
| 语义层 `bool = isnan(x)`（NaN→True、±inf/有限→False） | §1.3/§1.4 公式与算法描述、§3.1 拆解、§8.1 golden | ✅ |
| §0.5 R1「并行结构重设计」 | §3.3 伪代码（一维 `T.Kernel(num_kernels, is_npu=True)` + block_size 折叠 + v-prefix）、§5.2/§5.5、§6.1 | ✅（未照抄 GPU `threads×npt`） |
| §0.5 R2「NaN 谓词重设计」 | §1.6.1 项 1/2（机器验证）+ §3.2 步骤 1 | ✅（未照抄 `T.isnan`+宽化） |
| §0.5 R3「bool 输出表示重设计」 | §3.2 步骤 2/3 + §3.3 掩码→字节链 + §11.4 视图契约 | ✅ |
| §0.5「fragment → UB」等价替换 | §4.3/§4.4（`T.alloc_ub` + `T.copy` GM↔UB） | ✅ |
| §0.5「coalesced → MTE 连续搬运」等价替换 | §3.2 步骤 0/4（双显式 slice `T.copy`） | ✅ |
| §0.5「bool int8 字节存储」等价替换 | §3.3 输出 int8 + §11.4 视图契约 | ✅ |
| §0.5「swizzle/mma/shuffle 舍弃（不适用）」 | §3.5.1（源未使用，无迁移对象） | ✅（不涉及） |

### 11.2 源优化手段（§0.4）意图承接

| §0.4 手段 | 承接结构 | 是否静默丢弃 |
|-----------|---------|-------------|
| #1 register_copy fragment 分块装载/写回 | UB `T.copy` 分块搬移（§4.4） | 否 |
| #2 coalesced 向量化访问 | UB 连续 slice `T.copy` + Vector 整块 v-prefix（§3.2/§6） | 否 |
| #3 `threads × npt` 两级并行 | §0.6 R1 重设计为一维 Kernel + block_size，元素级并行交给 Vector | 否（意图由 R1 承接） |
| #4 fp32 宽化谓词 `T.isnan(T.cast(x,"float32"))` | §0.6 R2 + §1.6.1 项 1/2 升级为原生 dtype 自比较（消宽化中转） | 否（升级为等价化简） |
| #5 bool 字节存储（`store_unary_bool_as_int8`） | §0.6 R3 掩码→字节 cast 链 + §11.4 视图契约 | 否 |

### 11.3 float dtype 契约

- **支持集**：`{float16, bfloat16, float32}`（`FloatPredicateKernel` 的 `FLOAT_DTYPES` 子集）；Op / Kernel 层显式拒绝 int / bool 输入（`ValueError`，非迁移范围）。
- **计算路径**：谓词在**原生 dtype** 上完成（`vcmp(x,x,"ne")`），**无 fp32 宽化中转**；谓词是精确布尔运算，输出 bool，**无浮点容差**（atol=rtol=0）。
- **依据**：`vcmp` 的 DataType 表覆盖 fp16/fp32/bf16（`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1）；IEEE `isnan(x) ⟺ (x != x)`，宽化不改变 NaN 性（§1.6.1 机器验证 0 违反）。

### 11.4 bool 字节存储契约

- kernel 以 `int8` 字节缓冲写 bool 结果（`T.Tensor((N,), "int8")`，`out_idx=[1]`）；Op / Kernel 层在离开 kernel 后 `result.view(torch.bool)`（stride/shape/dtype 元数据视图操作，**不改输入张量真实内容**，满足 ascend-constraints host 侧约束）。
- 依据：bool i1 无法 load/store GM（`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3）；且 CG-2026-0019 记载 npuir auto output dtype 映射缺 uint8/bool 键，字节域算子须用 `int8` 存储 + host view（bool 输出先例见 `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel_DESIGN.md`）。
- bool 结果表示：掩码 true → f16 1.0 → int8 1 → `True`；掩码 false → f16 0.0 → int8 0 → `False`（§0.6 R3）。
