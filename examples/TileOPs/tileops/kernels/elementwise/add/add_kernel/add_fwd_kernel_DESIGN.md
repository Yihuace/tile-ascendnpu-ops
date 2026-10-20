# AddFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/arithmetic.py` 的 `AddFwdKernel`，复用 `_base.py` 的 `AlphaScaledBinaryKernel`/`BinaryKernel` 及其 `register_copy` 策略，builder 为 `_builders.py` 的 `make_binary_register_copy`；broadcast 由 `_broadcast.py` 的 `coalesce_broadcast_dims`/`compute_broadcast_offsets`/`row_broadcast_split` 与 `_op_body.py` 驱动）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `add_fwd_kernel`，`target="npuir"`。
> 迁移工作区：`examples/add/add_fwd_kernel/`（project=`add`，op=`add_fwd_kernel`，scenario=migration，mode=harness）。
> 本设计按 skill `tilelang-op-design` 的三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出；programming_mode = `developer`，target = `npuir`。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 skill `tilelang-op-design` 的 references/migration-analysis.md）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：`torch.add(input, other, *, alpha=1)` 的逐元素二元加法，带 scalar 缩放与 numpy 风格广播：

$$
y = \operatorname{input} + \alpha \cdot \operatorname{other},\qquad
\operatorname{broadcast}(y.\text{shape}) = \operatorname{broadcast\_shapes}(A, B)
$$

源实现（`AddFwdKernel`，`arithmetic.py` L43-50；op body 由 `AlphaScaledBinaryKernel._alpha_op_func` 构造）：

```python
class AddFwdKernel(AlphaScaledBinaryKernel):
    @staticmethod
    def _combine(a, scaled_b):        # AddFwd 的 combine = 加法
        return a + scaled_b

def op_func(a, b):                    # _alpha_op_func 的浮点分支
    if alpha == 1:                    # 恒等乘子：跳过标量乘，保持 byte-identical
        return a + b
    scaled_b = T.cast(T.cast(alpha, "float32") * T.cast(b, "float32"), a.dtype)
    return a + scaled_b
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | 逐元素 `y = a + alpha * b`；`alpha` 为**构造期 scalar**（`int | float`，kw_only，默认 1），编入 op body（每个 alpha 值一个特化）；无跨元素依赖 |
| 规约语义 | **无规约**，纯逐元素，无累加顺序问题（manifest 的加法不引入规约语义） |
| dtype 语义 | NPU 契约 `T ∈ {float16, bfloat16, float32}`（GPU manifest `forall` 的 union 还含 `bool/uint8/int8/int16/int32/int64`；本次迁移按 GPU 实际浮点子集与 NPU 侧 `examples/TileOPs/tests/ops/test_add.py` 契约取浮点，见 §0.2 注）。输出 `same_as(input)`。**浮点分支把 `alpha * b` 的乘法放在 fp32 域、再 cast 回 `a.dtype`，最后在 `a.dtype` 内做加法**（`scaled_b = cast(cast(alpha,f32)*cast(b,f32), a.dtype); a + scaled_b`）。NPU 侧为对齐 torch golden（fp32 opmath），采用**完整 fp32 中转**（乘加均在 fp32，末尾一次 cast，§0.6 R2） |
| 边界语义（NaN/±inf） | 无专用 NaN 处理：加法与乘法的 IEEE 语义自然涌现。`x=NaN` 经 `a+alpha*b` 传播 NaN；`1*b=b` 对全部可表示值（含 NaN/±inf/−0.0/subnormal）精确成立；`inf+(-inf)=NaN`；`0*b` 保留（**不做 alpha==0 折叠**，否则 `b=±inf` 时 `0*inf=NaN` 与 `y=a` 语义相悖） |
| 空维度 / 非整除 | Op 层把两操作数广播为同一 shape 并展平为 1-D `N_total = prod(broadcast(A,B))`；任意 shape 按展平后处理，非整除由尾块处理（`N=0` 的退化由 Op 层 `N_total=1` 兜底，kernel 不单独处理） |
| **广播契约** | numpy 广播规则（`torch.broadcast_shapes`）。`cnn-feat-broadcast`（`B[256,1,1]` vs `A[16,256,56,56]`）与 `linear-bias`（`B[1,4096]` vs `A[1024,4096]`）是两个广播 workload；`hidden-state-prefill`/`llama-7b-ffn` 为 same-shape。**广播在 Op 层解析**（见 §0.3 host 逻辑、§0.5 判定、§0.6 R2），kernel 只见到两个 same-shape 连续 1-D 张量 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓测试基准 `torch.add(a.float(), b.float(), alpha).to(dtype)`），**不复刻 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input`（A） | `[*A]`（任意，如 `(2048,4096)` / `(16,256,56,56)`） | `float16 \| bfloat16 \| float32` | 第一操作数；kernel 收到广播后展平的 1-D `(N_total,)` 视图 |
| `other`（B） | `[*B]`（与 A 广播兼容，如 `(1,4096)` / `(256,1,1)`） | `same_as(input)` | 第二操作数（被 `alpha` 缩放）；kernel 收到广播后展平的 1-D `(N_total,)` 视图 |
| `alpha` | 标量（`int \| float`），非 tensor | — | `other` 的乘子（kw_only，默认 1）；构造期参数，编入 op body |
| `output` | `[*broadcast(A,B)]` | `same_as(input)` | 输出；无转置、与输入同 layout；Op 层 `view(out_shape)` 回塑 |

> **dtype 子集说明**：GPU `AddFwdKernel.SUPPORTED_DTYPES = BINARY_FULL_DTYPES`（含 bool/整数），但 NPU 侧 manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `AddFwdOp` 声明 `float16 | bfloat16 | float32`，且迁移测试 `test_add.py` 显式覆盖「非浮点 dtype 拒绝」。本设计跟随 GPU 实际浮点服务面 + NPU 移植测试的浮点契约（`_SUPPORTED_DTYPES = (float16, bfloat16, float32)`）。整数/bool 分支在源 builder 中保留（`_INT_DTYPES`）但不进入 NPU 服务面。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1（Op/host） | `out_shape = torch.broadcast_shapes(a_shape, b_shape)`；`expand(t, out_shape).contiguous().view(-1)` 展平两操作数 | `input [*A]`, `other [*B]` | `a_flat (N_total,)`, `b_flat (N_total,)` | 广播契约的解析（`BinaryElementwiseFwdInterface` / `AddFwdOp`） |
| 2（kernel） | 分块装载：`T.copy(a[bx·bs:(bx+1)·bs], a_reg)` + 同 b | `a_flat`, `b_flat` | `a_reg`, `b_reg`（fragment） | 数据搬入 |
| 3（kernel） | 逐元素 `merge = op_func(a,b)`：`alpha==1 → a+b`；否则 `scaled_b = cast(cast(alpha,f32)*cast(b,f32), dtype)`，`a + scaled_b` | `a_reg`, `b_reg`, `alpha` | `y_reg` | `a + alpha*b` |
| 4（kernel） | 分块写回：`T.copy(y_reg, y[bx·bs:(bx+1)·bs])` | `y_reg` | `y_flat` | 数据搬出 |
| 5（Op/host） | `result.view(out_shape)` | `y_flat` | `output [*broadcast(A,B)]` | 输出回塑 |

**GPU 双策略分派（源码级事实）**：`BinaryKernel` 依 `coalesce_broadcast_dims` + `is_contiguous_same_shape` 选策略——same-shape 连续走 `register_copy`（fragment 全块，`make_binary_register_copy`）；广播走 `explicit_parallel`（stride 版 `make_binary_explicit`，逐元素 `compute_broadcast_offsets` 解算 a_off/b_off），行广播另走 `_row_broadcast_prim`（`row_broadcast_split`）。`AlphaScaledBinaryKernel` 再叠加 `_alpha_op_func` 的 alpha 特化。

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[a], GM[b] --T.copy--> fragment/a_reg,b_reg(寄存器/SMEM) --op_func(alpha 特化)--> fragment/y_reg --T.copy--> GM[y]
```

same-shape 的 `make_binary_register_copy` 里 `contiguous = _narrows(dtype,out_dtype) and N%block_size==0`；add 输入输出同宽（`_narrows` 为 False）→ 一律走 fragment 全块路径（`T.Parallel(threads, npt)` + 整块 `T.copy`）。

**循环与并行结构**：grid 一维 `T.Kernel(T.ceildiv(N, block_size), threads=threads)`（`threads` 默认 256，`npt` 默认 8，`block_size = threads×npt`）；block 内 `threads × npt` 两级并行（`T.Parallel(threads, npt)`）。广播策略下二维 grid（`_row_broadcast_prim` 用 `T.Kernel(ceildiv(inner, block_cols), rows, threads=...)`）。

**host 侧逻辑**：`AddFwdOp._eager_forward`：`broadcast_shapes(input, other)` → `expand(out_shape).contiguous().view(-1)`（两个 1-D 视图）→ kernel dispatch → `result.view(out_shape)`；`BinaryKernel.forward`（GPU）内部以 `flat(a)=a.reshape(-1)` 加 stride 计划调用 kernel。全部为 shape/stride/广播物化，**不改动输入张量的真实数据值**（`expand` 为 stride-0 视图，`contiguous` 产出新张量）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | register_copy（fragment 分块装载/写回） | 减少 GM 重复访问、放大访存粒度 | 分块 `T.copy` 到 fragment，逐元素计算后整块写回 | CUDA SMEM/寄存器 | 硬件强相关 |
| 2 | coalesced 向量化访问 | 尾轴连续读写打满带宽 | fragment 块 `T.copy` 连续搬运（窄化分支用 `T.vectorized`） | memory coalescing | 硬件强相关（NPU 有 MTE 等价） |
| 3 | alpha 编入 op body + `alpha==1` 恒等折叠 | 每 alpha 一特化；默认 alpha 与无 alpha 前路径 byte-identical | 构造期 Python 分支决定 op body；`alpha==1` 时整段跳过标量乘 | 无（编译期特化） | 可移植 |
| 4 | fp32 域标量乘（`alpha*b`）+ 末尾 cast | 窄 dtype 下保住缩放项精度 | `T.cast(alpha,"float32")*T.cast(b,"float32")` 再 cast 回 | 无（纯精度策略） | 可移植 |
| 5 | `threads × npt` 两级并行执行模型 | 每线程处理 `npt` 元素放大吞吐 | CUDA thread/warp 组织 | CUDA 线程模型 | 硬件强相关 |
| 6 | 广播的 stride 化解算（`compute_broadcast_offsets`/`row_broadcast_split`） | same-shape 之外仍单遍、无物化 | 编译期 unrolled divmod 链求 a_off/b_off；行广播每块解算一次 | CUDA 逐线程 gather | 硬件强相关（见 §0.5） |
| 7 | `_param_key` lru_cache 特化 | 每 (alpha, dtype, shape) 组合独立编译缓存 | `@functools.lru_cache` + op body 名含参数 | 无（host 缓存机制） | 可移植 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。识别不出来 ≠ 不存在——上表为源码 `_alpha_op_func` + `BinaryKernel.__init__`/`_build_kernel` + `make_binary_register_copy`/`make_binary_explicit` 的逐项核对结果。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y = a + alpha*b`（广播、alpha 特化、NaN/±inf IEEE 传播、输出同 dtype） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| `alpha` 编入 op body + `alpha==1` 折叠（#3） | 算法 | 无 | — | **保留** | 编译期特化，NPU 侧同款（`alpha==1` 不生成 vmul；等价性经 verify_equiv.py 验证，§1.6.1 项 1） |
| fp32 域标量乘 + 末尾 cast（#4） | 算法 | 无 | — | **保留→强化为完整 fp32 中转** | 对齐 torch golden fp32 opmath（PL-1.17-subfp32-fp32-transit）；bf16 无原生 v-prefix 算术，中转是唯一路径（§0.6 R2） |
| `_param_key` lru_cache 特化（#7） | 优化 | 无 | 有 | **等价替换** | factory 层 `@functools.lru_cache(maxsize=32)`（NPU 侧同款；`examples/TileOPs/tileops/kernels/elementwise/add/_add_fwd_kernels.py`） |
| `threads × npt` 两级并行 + `T.Parallel(threads,npt)`（#5） | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel；`threads=` 在 npuir 无效果） | **重新设计** | → §0.6 R1（block_size 折叠 + 一维 `T.Kernel` + UB 向量化） |
| fragment 分块装载/写回（SMEM/寄存器，#1） | 优化 | SMEM/寄存器 | 有（UB） | **等价替换** | `T.alloc_ub` + `T.copy`（GM→UB→GM），ascend-constraints.md §1 / decision-tree.md §3 |
| coalesced 连续访问（#2） | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | `T.copy` 连续 slice 搬运（双显式 slice 形态，TRAP-T-copy-region-semantics） |
| 广播的 stride 化解算（#6，`compute_broadcast_offsets` 逐元素 divmod gather + `row_broadcast_split`） | 优化 | CUDA 逐线程 gather | 无（NPU 缺跨步 gather 向量指令） | **重新设计** | → §0.6 R2（广播改在 Op 层解析为 same-shape 展平，kernel 只见平铺输入） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目（SMEM→UB、warp→Vector、CUDA 线程模型→一维 Kernel）+ `examples/` 佐证（lerp_tensor / mish / hardsigmoid 同族迁移）+ `docs/` 条目（`docs/Tilelang.language/` v-prefix API）。无「舍弃」项（广播 stride 化解算被等价替换/重新设计而非舍弃——其「单遍、无物化」意图由 Op 层广播展平承接，见 §0.6 R2）。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（`threads × npt` → 一维 Kernel + UB 向量化）**

- **源方案**：`T.Kernel(T.ceildiv(N, block_size), threads=256)` + `T.Parallel(threads, npt)` 两级并行；`block_size = threads × npt`（默认 256×8=2048）；fragment 装载/写回。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；GPU 的 `threads × npt` 折叠为单一 `block_size`（ascend-constraints.md 已知限制「仅支持一维 Kernel」；VP-2026-0007 实测 `T.Kernel(..., threads=)` 在 npuir 无效果，KernelLaunch NPU 分支仅消费 grid_size）；元素级并行交给 Vector 单元（v-prefix 逐元素 API 对整块 UB buffer 计算，天然向量化）；分块装载/写回用 `T.copy`（GM→UB→GM，32B 对齐连续 slice）；大规模（逻辑核数 > 物理核数）用核内 `T.serial` persistent 串行（§5.5）。`block_size` 取值见 §5.2（8192，承接 mish/hardsigmoid/hardswish 同族元素级实测）。
- **语义保持论证**：逐元素映射不变（每元素独立计算，无跨元素依赖、无规约），`threads/npt` → `block_size/核` 的任务映射变化不改变任何输出值——每个 `y[i]` 的表达式与求值顺序与源码逐步一致（fp32 域 `alpha*b → +a`）。边界语义：尾块（非整除）仅 copy 有效元素，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 fragment 尾块行为等价。

**重设计项 R2：广播处理位置重设计（GPU kernel 内 stride 化解算 → NPU Op 层广播解析为 same-shape 展平）**

- **源方案**：GPU `BinaryKernel` 在 kernel 内用 `coalesce_broadcast_dims` 生成广播计划（`coalesced_shape`, `a_strides`, `b_strides`），`make_binary_explicit`/`_row_broadcast_prim` 逐元素或逐块 `compute_broadcast_offsets` 解算偏移，**在 kernel 内完成广播索引**，两操作数不物化。
- **NPU 新算法**：广播改在 **Op 层**解析——`out_shape = torch.broadcast_shapes(A, B)`，两操作数 `expand(out_shape).contiguous().view(-1)` 展平为 same-shape 连续 1-D，`add_fwd_kernel` 只见到两个平铺输入（`register_copy` 平铺形态）。**NPU 可行性证据**：① NPU 缺跨步 gather 向量指令（`.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md` 的 `TRAP-C10-strided-gather`「跨步 gather 向量指令缺失」），kernel 内逐元素 divmod 广播索引无法向量化；② 同族 NPU 先例已确立 Op 层预广播契约——lerp_tensor 迁移设计 `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel_DESIGN.md` §0.3 步 1「预广播（Op 层契约，wrapper 承接）」及 `examples/TileOPs/docs/gpu_to_npu_adaptation.md` §3.4 E15（broadcast→flatten→dispatch→reshape 流程设备无关，完全相同）；③ NPU 侧 scaffold `examples/TileOPs/tileops/ops/elementwise/add.py` 的 `_expand_flat` 即此形态。
- **代价与语义保持论证**：数学等价——广播的每一输出元素 `y[i]=a[a_off(i)] + alpha*b[b_off(i)]` 与展平后 `a_flat[i]+alpha*b_flat[i]` 逐位一致（`expand` 复制的是同一源元素）；dtype/边界（NaN/±inf）语义不变。**代价**：广播操作数被物化为 `N_total` 个元素，对 `cnn-feat-broadcast`/`linear-bias` 引入一次额外的设备端广播拷贝流量（写入 ~`N_total` 元素 + 读入原尺寸），设计期估算约 +21µs（§1.6.0），属已知代价；备选（kernel 内 `T.vbrc` 行广播，避免物化）列为 Stage 4 A/B 候选（§9.1）。

> §0.5 全部「重新设计」条目（R1/R2）已重设计；其余为保留/等价替换。**源优化手段（§0.4）意图承接**：#1 register_copy 与 #2 coalesced 的「放大访存粒度、整块搬移」意图由 UB `T.copy` 分块搬移承接；#5 `threads×npt` 的「元素级并行放大吞吐」意图由 Vector 单元整块 v-prefix 计算承接；#6 广播 stride 解的「单遍、无物化」意图由 Op 层广播展平承接（展平后仍单遍；物化代价见 R2）；#3 alpha 特化/折叠与 #4 fp32 精度为算法层直接保留（#4 强化为完整 fp32 中转）；#7 lru_cache 特化由 factory `@lru_cache` 等价替换承接。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/arithmetic.py`（`AddFwdKernel`）；`.../elementwise/_base.py`（`AlphaScaledBinaryKernel`/`BinaryKernel`/`_build_kernel`）、`.../elementwise/_builders.py`（`make_binary_register_copy`/`make_binary_explicit`/`_row_broadcast_prim`）、`.../elementwise/_broadcast.py`（`coalesce_broadcast_dims`/`compute_broadcast_offsets`/`row_broadcast_split`）、`.../elementwise/_op_body.py`；`/mnt/workspace/TileOPs-main/src/tileops/ops/elementwise/arithmetic.py`（`AddFwdOp`/`_AlphaScaledBinaryOp`）。
- 源 manifest 条目：`/mnt/workspace/TileOPs-main/src/tileops/manifest/spec/elementwise_binary.yaml`（`AddFwdOp`，~L98）。
- NPU 侧提取参考（合并 factory）：`examples/TileOPs/tileops/kernels/elementwise/add/_add_fwd_kernels.py`（factory `add_fwd_kernel(N, dtype, alpha=1, output_dtype=None, threads=256, num_per_thread=8)`）。
- NPU wrapper/op/test/bench：`examples/TileOPs/tileops/kernels/elementwise/add/add.py`、`examples/TileOPs/tileops/ops/elementwise/add.py`、`examples/TileOPs/tests/ops/test_add.py`、`examples/TileOPs/benchmarks/ops/bench_add.py`。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_add.py` 的 `ref_program = torch.add(a.float(), b.float(), alpha=self.alpha).to(a.dtype)`；tolerances 见同文件 `_get_tolerances`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现，**不复刻 NPU 算法**（保证验证独立性）。

---
## 1. 概述

### 1.1 算子名称

AddFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `add_fwd_kernel`。

### 1.2 功能描述

带 scalar 缩放与 numpy 广播的逐元素二元加法：`y = input + alpha * other`（`torch.add`），`alpha` 为构造期 `int | float`（默认 1）；广播在 Op 层解析为同形展平，kernel 只做平铺逐元素计算；中间 fp32，输出与输入同 shape/dtype。用于残差加、bias 加、FFN 偏置等线性代数前置/后置。

### 1.3 数学公式

$$
y_i = a_i + \alpha \cdot b_i,\qquad i = 1\ldots N,\quad N = \prod \operatorname{broadcast}(A, B)
$$

（`alpha` 为编译期常数；fp16/bf16 经 fp32 中转后按输出 dtype 舍入一次。）

### 1.4 算法描述

单步逐元素映射：Op 层把两操作数广播/展平为 1-D `N`，kernel 对每个元素独立计算 `a + alpha*b`（fp16/bf16 经 fp32 中转）。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**。本算法与 §1.6.0 调研选定算法一致（逐元素直接映射，无更优算法族）。与源算法结构差异：① 并行组织（GPU `threads×npt` → NPU 一维 Kernel + `block_size` 折叠，§0.6 R1）；② 广播处理位置（GPU kernel 内 stride 化解算 → NPU Op 层广播展平，§0.6 R2）；③ 精度路径（源「fp32 乘 + dtype 加」→ NPU「完整 fp32 乘加 + 一次 cast」，对齐 torch golden，§0.6 R2/§1.6.1）。计算表达式本身（`a + alpha*b`，fp32 域）与源语义一致。

### 1.5 数据流图

```
输入 input [*A] --Op 层 broadcast+expand+flatten--> a_flat (N,)
输入 other [*B] --Op 层 broadcast+expand+flatten--> b_flat (N,)
a_flat --T.copy--> UB[a_ub] --vcast--> UB[a32] ┐
b_flat --T.copy--> UB[b_ub] --vcast--> UB[b32] --vmul(alpha 标量)--> UB[t] --vadd(a32+t)--> UB[y32]
   --vcast--> UB[y_ub] --T.copy--> y_flat (N,) --Op 层 view--> 输出 [*broadcast(A,B)]
```

### 1.6 算法调研与优化分析

> 设计第一优先级：先调研四问（§1.6.0），再数学等价优化公式（§1.6.1），再向量化替代分析（§1.6.2），最后向量化轴/布局决策（§1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：`y = a + alpha*b`（带广播）的算法族——所有能算出相同结果的算法结构。源算法/输入公式只是基线候选之一。本算子为**单步逐元素**，按 algorithm-research.md §2 判为**轻量调研**（四问逐问回答并给依据）。对照 algorithm-candidates.md 的 elementwise 命中行（R1：公共子表达式消除 / 除法转乘倒数 / cast 链合并；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返——本算子无中间 GM 往返）与 pattern-library/INDEX.md 的 `PL-1.17-subfp32-fp32-transit`。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法） | `y = a + alpha*b`（标量乘始终生成；浮点经 fp32 乘再 cast 加） | — | — | ✅（基线） |
| 1 | `alpha==1` 恒等折叠（源已含） | `alpha==1 → y = a + b`（跳过 vmul） | 数学恒等：`1*b == b` 对全部可表示值（含 NaN/±inf/−0.0/subnormal）精确成立 | 少 1 次 vmul + 1 次 cast，省 1 个 UB buffer | ✅ |
| 2 | 完整 fp32 中转（乘加均在 fp32，末尾一次 cast） | `y = cast(fp32(a) + fp32(alpha)*fp32(b))` | 容差内等价：对齐 torch golden 的 fp32 opmath；源「fp32 乘→cast→dtype 加」与之差 ≤1 ulp | 免 bf16 无原生算术（唯一可行路径），且对齐 golden | ✅ |
| 3 | `alpha==0 → y = a` 折叠 | `y = a` | **不等价**：`b=±inf` 时源 `0*inf=NaN` → `a+NaN=NaN`，而 `y=a` 丢掉 NaN | 负收益（语义错误） | ❌（语义不成立，不纳入） |
| 4 | 交换律重排 `a + alpha*b → alpha*b + a` | 同值 | 数学恒等（加法交换） | 无（op 数、访存、精度同） | ❌（无结构差异） |
| 5 | 广播下推为 per-row 标量（kernel 内 stride 0 读） | `y[row,:] = a[row,:] + alpha*b_scalar(row)` | 数学恒等 | 免广播物化流量 | ✅（作为布局候选进 §1.6.3；可行性见 R4/§0.6 R2） |

**R2 在线算法**：**无「在线/离线」之分**（即天然单遍）——`a + alpha*b` 是逐元素映射，每个输出元素只依赖对应输入元素，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（algorithm-research.md §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导，不得只比 FLOPs）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线（`a + alpha*b`，fp32 乘加） | `(1 if alpha==1 else 2)·N`（manifest 口径；乘加=2） | `3·N·elem_bytes`（读 a + 读 b + 写 y，无中间 GM 往返） | 1 | `O(block_size)`（UB 驻留，无 GM 中间缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1（alpha==1 折叠） | `1·N`（alpha==1） | `3·N·elem_bytes` | 1 | `O(block_size)` | 完全并行 |
| 候选 2（完整 fp32 中转） | 同基线（实际 v-op 数 +2 vcast） | 同基线 | 1 | `O(block_size)` | 完全并行 |
| 候选 5（广播下推 per-row 标量） | 同基线 | 广播操作数**免物化**：kernel 读 `2·N + |B|`、写 `N`（比基线省一次广播操作数的读写往返） | 1 | `O(block_size)` | 完全并行；但需 kernel 内广播索引（R4 淘汰） |

**结论**：候选间 FLOPs / 遍数 / 中间缓冲四口径基本同（带宽受限，访存主导）；基线（含候选 1 折叠）即为最优访存下的最简 op 链；候选 5 访存更省但其 kernel 内广播索引在 NPU 不可行（R4）。无并列争议。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰须附 `docs/` 与 `examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线 | 逐元素 → Vector（vcast/vmul/vadd），**不落 Cube**，与元素级算力正确匹配 | 工作集 = `O(block_size)` UB 驻留，112–144KB < 192KB（§4.5） | 展平尾轴连续；`block_size=8192` 为 32B 整数倍且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓ | `ceildiv(N, block_size)` / `ceildiv(n_num, num_kernels)` 均编译期常量（N/block_size 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ✅ 采纳 |
| 候选 1（alpha==1 折叠） | 同上（v-op 更少） | 同上 | 同上 | 同上 | 同上 | ✅ 采纳（并入选定算法） |
| 候选 2（完整 fp32 中转） | 同上（+2 vcast，bf16 必经） | 同上 | 同上 | 同上 | 同上 | ✅ 采纳 |
| 候选 5（kernel 内广播 stride-0 读） | Vector，但逐元素 divmod 广播索引退化为**逐元素标量 gather** | 同上 | 广播轴整除性依赖具体 shape | 依赖动态 broadcast 计划 | 破坏向量化 | ❌ 淘汰：NPU 缺跨步 gather 向量指令（`TRAP-C10-strided-gather`，「跨步 gather 向量指令缺失」）；改走 Op 层展平（§0.6 R2） |

**定量 roofline（D-2）**：基线容量/带宽/发射均匹配 Vector 单元（工作集 < UB、尾轴 32B 对齐、约束可静态化），无候选因违反容量/带宽量级被淘汰。采用 pattern-library/constants.md 实测常数条目（版本戳 `tilelang 0.1.2+ed787bb`（2026-09-07 build）/ CANN 8.5.0 / Ascend910B2C）：

- 流量项：GM 读写总字节 ÷ 有效带宽（`CONST-mte2-degradation` 按总流量档位取值）；
- 发射项：向量 op 数 × ~0.5µs/op（`CONST-vector-launch-overhead`）；
- 容量项：轻量调研豁免（单步逐元素，无跨引擎中间矩阵往返，`CONST-store-fixpipe-gm-only` 不适用——无 Cube）。

```
估算下界(cnn-feat-broadcast fp16, N=12,845,056): max(流量 ≈63µs, 发射 ≈2.5µs) ≈ 63µs
  流量项: 3 × 12,845,056 × 2B = 77,070,336B ≈ 77MB；MTE2 有效带宽按 CONST-mte2-degradation
          12.8M 档 ~30.4 GB/s/core × 40 向量核 ≈ 1.216 TB/s → 77MB / 1.216TB/s ≈ 63µs
  发射项: 链长 ~5 v-op（vcast×2 + vmul + vadd + vcast）× 0.5µs ≈ 2.5µs（f16≈f32）
  容量项: 豁免（单步逐元素，无跨引擎往返）
估算下界(hidden-state-prefill fp32, N=8,388,608): max(流量 ≈81µs, 发射 ≈2.5µs) ≈ 81µs
  流量项: 3 × 8,388,608 × 4B = 100,663,296B ≈ 100.7MB ÷（8.4M 档 ~31.1 GB/s/core × 40 ≈ 1.244TB/s）≈ 81µs
附加代价: 广播 workload 的 Op 层展平物化——B[256,1,1]→[16,256,56,56] 写 ~N 个元素 ≈ +21µs（fp16），
          计入端到端但不入 kernel 下界（§0.6 R2 已知代价；备选 vbrc 行广播见 §9.1）
备注: 若发射成本不可跨 tile 重叠，大 N 档 tile 数（cnn-feat 1568 tile）的发射项可能上升——
      block_size 增大 / T.Pipelined 为 Stage 4 首要调优轴（§6.3 / §9.1）
```

**调研结论**：选定算法族 = **逐元素直接映射（单遍，`y = a + alpha*b`，fp32 中转，`alpha==1` 折叠，广播在 Op 层展平）**。关键依据：R1 无更优于基线的化简公式（候选 3 语义不成立、候选 4 无结构差异；候选 1/2 为源已含的恒等/精度策略）、R2 天然单遍、R3 四口径基线最优（带宽受限，访存主导）、R4 纯 Vector 亲和（UB 驻留、32B 对齐、静态边界），kernel 内广播候选因缺跨步 gather 淘汰。与基线结构差异：**计算表达式不变**，仅并行组织（§0.6 R1）、广播位置（§0.6 R2）与精度路径（完整 fp32 中转）不同。「无更优替代」调研范围：algorithm-candidates.md elementwise 命中行、恒等变形自查（R1 候选 1/3/4/5）、pattern-library 的 `PL-1.17-subfp32-fp32-transit` 与 `CONST-mte2-degradation` 同类实现、源码 §0.3/§0.4 的算法与优化手段。源优化手段（§0.4）在调研后被 §0.6 承接，无静默丢弃。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法；调研候选中初判采纳的项在此完成等价论证与收益量化。含采纳项，故生成并执行 `verify_equiv.py`（结果表见下）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | `alpha==1` 恒等乘子折叠 | `y = a + 1.0 * b`（标量乘始终生成） | `y = a + b`（alpha==1 时不生成 vmul） | **数学恒等**：IEEE 乘法 `1.0*b` 对全部可表示值精确等于 `b`（`1*inf=inf`、`1*NaN=NaN`、`1*(-0.0)=-0.0`、subnormal 不变），加法结果逐位相同 | 省 1 次 vmul + 1 次 cast；默认 alpha 路径 v-op 链由 5 降至 4；省 1 个 fp32 UB buffer |
| 2 | 完整 fp32 中转（乘加均在 fp32，末尾一次 cast） | 源式 `scaled_b = cast(cast(alpha,f32)*cast(b,f32), dtype); y = a + scaled_b`（fp32 乘→cast→dtype 加，两次舍入） | `y = cast(fp32(a) + fp32(alpha)*fp32(b))`（fp32 乘加→一次舍入） | 容差内等价：两式均为 `a + alpha*b` 的舍入实现，差 ≤1 ulp（fp16）/≤1 ulp（bf16）；新式对齐 torch golden 的 fp32 opmath（`torch.add(a.float(),b.float(),alpha).to(dtype)`），bf16 无原生 v-prefix 算术 → 中转是唯一可行路径（PL-1.17-subfp32-fp32-transit） | 免 bf16 编译失败；对齐 golden 使 L0 精度稳定（fp16/bf16 均 ≤1 ulp） |

**机器验证结果表**（`examples/add/add_fwd_kernel/verify_equiv.py`，torch CPU + fp64 参照 + 随机/角点值 + 逐 dtype 舍入，已执行；角点覆盖 ±inf/NaN/−0.0/subnormal/±max；shape 覆盖 same-shape（(4096,)）与广播（(64,128)×(1,128)、(4,8,4,4)×(8,1,1)、(3,1)×(1,4)、(1024,1024)×(1,1)）；alpha ∈ {1, 1.0, 0.5, 2.0, −1.0, 0.0, 3}）：

| 采纳项 | 覆盖范围 | max ulp 差（tolerance-unit） | 违反率 | 结论 |
|--------|---------|---------------------------|--------|------|
| 1. alpha==1 恒等折叠（`a+1*b` → `a+b`，折叠 vs 未折叠逐位对照） | alpha∈{1,1.0} × 3 dtype × 5 shape = 30 行 | 0.000（逐位相同） | 0 / 0（NaN 失配 0） | EQUIV_PASS |
| 2. 完整 fp32 中转（候选 vs torch golden / vs fp64 真值） | alpha∈{1,0.5,2.0,−1.0,0.0,3} × 3 dtype × 5 shape = 105 行 | 0.0128（< 1 = 容差带内；vs fp64 真值同为 0.0128） | 0 / 0（NaN 失配 0） | EQUIV_PASS |

> 对照项：**原生 dtype 乘加基线**（不做 fp32 中转）vs torch golden 在 `alpha=3` 时最大 1.093 tolerance-unit、累计 4 次违反（bf16 达 128 ulp、fp16 达 1024 ulp）——证明候选 2 的 fp32 中转必要性。全部 135 行 `EQUIV_PASS`，无未通过项，可采纳。脚本：`examples/add/add_fwd_kernel/verify_equiv.py`（Stage 2 维度 8 重跑复核）。

**优化结论**：采纳 2 项（1 个数学恒等项 + 1 个精度等价项）。优化后公式为：

$$
y_i = a_i + b_i \quad (\alpha = 1),\qquad
y_i = a_i + \alpha \cdot b_i \quad (\alpha \neq 1)
$$

（`alpha==1` 时折叠 v-mul；fp16/bf16 经 fp32 中转，末尾一次 cast。）

供 §3.1 公式拆解使用。**无其他可采纳优化项**：单次逐元素映射，无公共子表达式（`alpha*b` 只算一次）、无除法/超越函数、无 cast 链冗余、无中间 GM 往返；`alpha==0` 折叠因语义不成立（`0*inf=NaN`）被拒。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 升精度 cast（fp16/bf16→fp32） | 逐元素 `T.cast(b[i],"float32")` | `T.vcast(b_ub, b32, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1；mish/lerp 佐证） | ✅ | — |
| 2 | 标量缩放的乘法 `alpha*b` | 逐元素 `alpha*b[i]` | `T.vmul(b32, alpha_scalar, t)`（`docs/Tilelang.language/数学操作/T.vmul.md`，src2 支持标量） | ✅ | — |
| 3 | 逐元素加法 `a + alpha*b` | 逐元素 `a[i]+scaled[i]` | `T.vadd(a32, t, y32)`（`docs/Tilelang.language/数学操作/T.vadd.md`）或 `alpha==1` 时直接 `T.vadd(a32, b32, y32)` | ✅ | — |
| 4 | 降精度 cast（fp32→fp16/bf16） | 逐元素 `T.cast(y32, dtype)` | `T.vcast(y32, y_ub, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`） | ✅ | — |
| 5 | block 起始索引 `offset = block_id * block_size` | 标量乘加 | 无向量 API | ❌ | block 级任务映射索引，每 block 少量标量，非逐元素热点（可接受理由类别①） |
| 6 | 尾块大小 `tail_size = T.min(block_size, shape - offset)` | 标量 min | 无（`T.min` 标量形） | ❌ | 依赖动态 shape 的边界处理，每 block 一次，非逐元素热点（可接受理由类别④；mish 同款尾块） |
| 7 | `num_kernels`/`num_local_tasks`/广播 shape 计算 | trace-time Python 标量 | 无（编译期） | ❌ | host/trace 侧 shape/分核元数据计算，不在 kernel 内（可接受理由类别②） |
| 8 | Op 层广播（`broadcast_shapes` + `expand`） | host 张量操作 | 无（向量 API 不适用于 Op 层 shape 语义） | ❌ | host 侧 shape/stride 元数据与广播物化（广播语义解析），不在 kernel 计算环内（可接受理由类别②；§0.6 R2） |

**向量化结论**：核内逐元素计算（cast / 标量乘 / 加 / cast）已**全部向量化**（替代 API 均有 `docs/Tilelang.language/` 文档佐证：vcast/vmul/vadd）。保留 4 处标量：block 起始索引（block 级任务映射）、尾块大小（动态边界）、分核元数据与 Op 层广播（均 host/trace 侧，不在 kernel 计算环内），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（输入 `[*A]`/`[*B]` 广播进、输出 `[*broadcast(A,B)]`）。本算子在 Op 层把两操作数广播展平为 1-D same-shape 后，核内仅一个非 batch 轴（展平轴）；但广播的**处理轴 / 处理位置**是真实设计变量，故枚举候选并量化弃选。

**候选矩阵**（枚举有意义候选，弃选给量化理由，禁止「未考虑」）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | Op 层广播展平为 1-D same-shape（`expand→contiguous→view(-1)`），核内平铺逐元素 | 展平轴（stride=1 连续） | 广播操作数一次设备端物化（广播 workload：写 ~N 元素；same-shape：无） | 基线：连续读写、无核内 stride 索引、UB = O(block_size)；代价 = 广播物化 ≈+21µs（§1.6.0） | ✅ 采纳 |
| 2 | 保留 `[rows, inner]` 2-D 视图，外层 rows 串行、内层 inner 向量化 | inner（stride=1） | 无（Op 层已展平，仅 reshape 视图） | 与候选 1 数学同构（1-D 展平即 rows=1 极限）；多一层 block 索引换算，无增益；UB 预算相同 | ❌（无结构差异，徒增索引） |
| 3 | 核内广播布局（kernel 内按 broadcast 计划 stride 0 / 1 读，免物化） | inner（stride=1）+ 广播轴 stride 0 | 无（不物化） | 省广播物化流量，但需逐元素 divmod 广播索引 → NPU 缺跨步 gather 向量指令（`TRAP-C10-strided-gather`），退化为逐元素标量 gather | ❌（向量化被破坏，净亏；可行性证据见 §0.6 R2） |
| 4 | 核内行广播（`T.vbrc` 把行/列因子广播到 tile 后全形 `vadd`） | inner | 核内 `vbrc` 展开（无需物化） | 免物化且可向量化；但需按 workload 特化（行/列/标量因子形态不同）且 `vbrc` 同形 src/dst 触发空广播维 verify fail（`TRAP-vbrc-same-shape-empty-broadcast`，见 traps-compiler.md） | ❌（主选不采纳；列 Stage 4 A/B 备选，§9.1） |
| 5 | host permute 重排进出 | 其他轴 | 两次 host permute | 逐元素无窗口/归约轴可重排——重排不改变逐元素访存模式，纯增 repack 代价（PL-1.3-host-permute，百 µs 级通常净亏） | ❌（零收益纯增代价） |

- **向量化轴** = 展平轴（stride=1 连续，无累加链、无跨步系数）。
- **整除性**：`block_size=8192`，fp16/bf16 向量宽 ×8、fp32 ×4 均整除（8192 = 1024×8 = 2048×4）✓；尾块 `tail_size` 动态，仅 copy 有效元素，垃圾 lane 不写回（与 mish 同款）。尾 lane 浪费率：0（整除档）。
- **累加链形态**：无累加（逐元素），无跨步系数阻碍。
- **repack 代价**：候选 1 的 same-shape workload 无 repack；广播 workload 有一次 Op 层物化（≈+21µs，§1.6.0）。
- **UB 容量影响**：驻留缓冲 = O(block_size)（§4.5），布局候选间无差异。
- **迁移对照**：GPU 源码的并行轴选择（thread/npt）与广播 stride 计划是**输入而非结论**——GPU thread/warp 轴与 NPU Vector 展平轴不对应，且 kernel 内广播索引在 NPU 不可向量化（候选 3 淘汰）；本节独立评估后取「Op 层广播展平 + 展平轴向量化」。

**布局决策结论**：选定方案 #1「Op 层广播展平为 1-D same-shape + 展平轴向量化」，无核内 repack、无重新布局。候选 #2/#3/#4/#5 弃选理由见矩阵（无结构差异 / 缺跨步 gather / 同形 vbrc verify fail / 零收益纯增代价），且已按文档合法形态核对（`TRAP-C10-strided-gather`、`TRAP-vbrc-same-shape-empty-broadcast`、`PL-1.3-host-permute`）。该结论同步落入 §3.3 伪代码与 §6 循环结构（向量维 = 展平轴，buffer 均为 1-D）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（迁移默认）。

### 2.2 选型理由

纯 Vector 逐元素算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。Developer 模式由编译器自动管理同步与 buffer 分配，与 mish / hardsigmoid / hardswish 同族 elementwise 先例一致。源算子 NPU wrapper 亦声明 `ascend_mode = "Developer"`（`examples/TileOPs/tileops/kernels/elementwise/add/add.py` L113）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix API 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（vcast / vmul / vadd），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 `set_flag`/`wait_flag`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | Op 层：`a_flat, b_flat = flatten(broadcast(A,B))` | 广播解析（host，§0.6 R2） |
| 1 | `a32 = cast(a)`、`b32 = cast(b)`（fp16/bf16 时） | 升 fp32（fp32 输入跳过），对齐 torch golden 的 fp32 opmath |
| 2 | `alpha==1`：`y32 = a32 + b32` | 恒等折叠，跳过步骤 3（§1.6.1 项 1） |
| 3 | `alpha!=1`：`t = alpha * b32`；`y32 = a32 + t` | 标量缩放 + 加法（§1.6.1 项 2） |
| 4 | `y = cast(y32)`（fp16/bf16 时） | 降回输出 dtype（一次舍入） |
| 5 | Op 层：`output = y_flat.view(out_shape)` | 输出回塑（host） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 数据搬入 GM→UB | `T.copy(a[offset:offset+tail], a_ub[0:tail])` | 双显式 slice（TRAP-T-copy-region-semantics 推荐形态） | Developer |
| 1 | cast 升 fp32 | `T.vcast(a_ub, a32, round_mode="rint")` | fp16/bf16 → fp32 | Developer |
| 2a | `alpha==1` 加 | `T.vadd(a32, b32, y32)` | 同形 1-D tensor×tensor（折叠路径） | Developer |
| 3a | 标量乘 `alpha*b32` | `T.vmul(b32, alpha_scalar, t)` | `src2` 为 fp32 标量（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3 仅 src2 可标量） | Developer |
| 3b | 加 `a32 + t` | `T.vadd(a32, t, y32)` | 同形 1-D | Developer |
| 4 | cast 降 dtype | `T.vcast(y32, y_ub, round_mode="rint")` | fp32 → fp16/bf16 | Developer |
| 5 | 数据搬出 UB→GM | `T.copy(y_ub[0:tail], y[offset:offset+tail])` | 双显式 slice | Developer |

> fp32 输入路径跳过步骤 1/4（无 vcast，走同 dtype `T.copy`，且可直接 `T.vadd(a_ub, b_ub, y_ub)`）。`alpha` 为构造期 Python 标量，浮点路径 `T.cast(alpha, "float32")` 上抛 fp32 后作为 `T.vmul` 的标量 operand，与源 `_alpha_op_func` 一致。`alpha==1` 时整段跳过步骤 3（编译期分支）。

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型以**第二个位置参数**传入（不用 `dtype=` 关键字），避免产生 `name 'in_dtype' is not defined` 类 false alarm（design-template §3.3）。

```python
import functools
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

@functools.lru_cache(maxsize=32)
def add_fwd_kernel(N, dtype, alpha=1, output_dtype=None, block_size=8192, num_cores=None):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    n_num = T.ceildiv(N, block_size)                 # 逻辑 tile 数（编译期常量）
    # 物理核数实查 ×2（纯 Vector），见 §5.5
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores >= n_num:
        num_cores = min(n_num, vector_cores)
    num_local_tasks = T.ceildiv(n_num, num_cores)     # 编译期常量，静态边界

    alpha32 = T.cast(alpha, "float32")
    fold = (alpha == 1)                               # 恒等折叠（§1.6.1 项 1）

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(_num_cores, _block_size):
        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
            shape: T.int32,
        ):
            with T.Kernel(_num_cores, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((_block_size,), dtype)
                b_ub = T.alloc_ub((_block_size,), dtype)
                a32 = T.alloc_ub((_block_size,), compute_dtype)
                b32 = T.alloc_ub((_block_size,), compute_dtype)
                y_ub = T.alloc_ub((_block_size,), out_dtype)

                for i in T.serial(num_local_tasks):
                    block_id = i * _num_cores + cid
                    if block_id < n_num:
                        offset = block_id * _block_size
                        tail = T.min(_block_size, shape - offset)
                        T.copy(a[offset:offset + tail], a_ub[0:tail])
                        T.copy(b[offset:offset + tail], b_ub[0:tail])

                        if dtype != compute_dtype:
                            T.vcast(a_ub, a32, round_mode="rint")
                            T.vcast(b_ub, b32, round_mode="rint")
                        else:
                            T.copy(a_ub, a32)
                            T.copy(b_ub, b32)

                        if fold:
                            T.vadd(a32, b32, a32)          # alpha==1 折叠：y = a + b
                        else:
                            T.vmul(b32, alpha32, b32)       # t = alpha*b（原地）
                            T.vadd(a32, b32, a32)           # y = a + t（原地）

                        if out_dtype != compute_dtype:
                            T.vcast(a32, y_ub, round_mode="rint")
                        else:
                            T.copy(a32, y_ub)
                        T.copy(y_ub[0:tail], y[offset:offset + tail])

        return main

    return kernel(num_cores, block_size)
```

> 注：`a32`/`b32` 复用为计算缓冲（`T.vadd`/`T.vmul` 支持 `dst` 与 `src` 同 buffer 的原地操作，`docs/Tilelang.language/数学操作/T.vadd.md` §2.3 第 3 条 / `T.vmul.md` §2.3 第 3 条），驻留 ≤112KB（§4.5）。fp32 路径（`dtype==compute_dtype`）`a32`/`b32` 可与 `a_ub`/`b_ub` 合并，`y_ub` 可与结果 buffer 合并，驻留降至 ≤96KB。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vadd(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vadd.md`（src1 可标量；fp16/fp32 ✓、bf16 ×；dst 可原地） | ✅ 已验证（文档读） |
| `T.vmul(src1, src2, dst)` | `docs/Tilelang.language/数学操作/T.vmul.md`（仅 src2 可标量；fp16/fp32 ✓、bf16 ×；dst 可原地） | ✅ 已验证（文档读） |
| `T.vcast(src, dst, round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（f16→f32 仅 rint；f32→f16/bf16 含 rint；src 仅 buffer）；mish/lerp 佐证 | ✅ 已验证 |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md`；TRAP-T-copy-region-semantics | ✅ 已验证 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；mish / `examples/TileOPs/tileops/kernels/elementwise/add/add.py` | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | mish / hardsigmoid / hardswish 同族先例 | ✅ 已验证 |
| `T.ceildiv` / `T.min`（标量） | `docs/Tilelang.language/数学操作/T.ceildiv.md`；mish 尾块 `T.min` 先例 | ✅ 已验证 |
| `NPUUtils.get().get_aicore_num()` | `docs/开发指南.md` §3.3；本机实查（§5.5） | ✅ 已实查 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 grid；广播经 Op 层展平后仍一维） | 一维 `T.Kernel(num_cores, is_npu=True)`，无三维 |
| GPU 专用 API | Yes（`T.Parallel`/`T.vectorized`/`T.get_thread_binding` CUDA 语义；`threads=` 在 npuir 无效果） | 全部改写为 NPU v-prefix API + UB 路径（§0.5/§0.6 R1） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(ceildiv(N, block_size), threads=256)`（block+thread 两级） | 一维 `T.Kernel(num_cores, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(threads, npt)` 显式两级循环 | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| 广播处理 | kernel 内 stride 化解算（`compute_broadcast_offsets`） | Op 层 `broadcast_shapes` + `expand` 展平为 same-shape | §0.6 R2 |
| 精度路径 | `scaled_b=cast(fp32(alpha)*fp32(b),dtype); a+scaled_b`（dtype 加） | 完整 fp32 乘加 + 一次 cast（对齐 torch golden） | §0.6 R2 / §1.6.1 项 2 |
| `threads×npt` | 256×8 = 2048 | `block_size=8192` 折叠 | §5.2 |
| alpha 传参 | `_alpha_op_func` 闭包常量 + `_param_key` 特化 | factory 参数 + `@lru_cache`，`T.vmul` 标量 operand | §0.5 / §3.2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/add/add.py` | 最高（本算子 NPU scaffold） | `_DEFAULT_BLOCK_SIZE=8192`、`ascend_mode="Developer"`、`SUPPORTED_DTYPES` 浮点子集、flat same-shape 契约 |
| `examples/hardswish/hardswish_fwd_kernel/DESIGN.md` | 高度相似（同族二元/一元逐元素迁移） | 一维 Kernel + block_size 折叠、fp32 中转、分核三要素、UB 预算口径、persistent 模板 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 高度相似（同族 unary 迁移） | Developer 模式、`T.alloc_ub`、`T.copy` 双显式 slice 尾块、v-prefix 链 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 相似（同族二元 elementwise + fp32 中转） | Op 层预广播契约、fp32 中转模式（PL-1.17）、`T.min` 尾块 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `input`（a） | `[*A]`（Op 层展平 1-D `N`） | `float16 \| bfloat16 \| float32` | 第一操作数；kernel 收到广播后展平的 1-D 视图 |
| `other`（b） | `[*B]`（Op 层展平 1-D `N`） | `same_as(input)` | 第二操作数（被 alpha 缩放）；kernel 收到展平 1-D 视图 |
| `alpha` | 标量 | `int \| float` | 乘子（构造期参数，非 tensor） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output`（y） | `[*broadcast(A,B)]`（Op 层 `view` 回塑） | `same_as(input)` | 输出；无转置、与输入同 layout |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `a_ub` | `(block_size,)` | input dtype | UB | 输入 a tile（GM→UB） |
| `b_ub` | `(block_size,)` | input dtype | UB | 输入 b tile（GM→UB） |
| `a32` | `(block_size,)` | float32 | UB | a 的 fp32 中转 / 结果累存（原地复用） |
| `b32` | `(block_size,)` | float32 | UB | b 的 fp32 中转 / `alpha*b` 原地结果 |
| `y_ub` | `(block_size,)` | output dtype | UB | 输出 tile（UB→GM） |

### 4.4 内存搬运路径

```
GM[a] --T.copy--> UB[a_ub] --vcast--> UB[a32] ┐
GM[b] --T.copy--> UB[b_ub] --vcast--> UB[b32] --vmul(alpha 标量)--> UB[b32]
   --vadd(a32 + b32)--> UB[a32] --vcast--> UB[y_ub] --T.copy--> GM[y]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返；无 L1/inter-core 依赖。（`alpha==1` 时省 vmul；fp32 输入省两次 vcast。）

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `a_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| `b_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| `a32` | (8192,) | fp32 | 32768 |
| `b32` | (8192,) | fp32 | 32768 |
| `y_ub` | (8192,) | fp16/bf16 | 16384（fp32 时 32768） |
| **总计（fp16/bf16 路径）** | | | **114688（112KB）** |
| **总计（fp32 路径，全 buffer 显式）** | | | **131072（128KB）** |
| **总计（fp32 路径，复用 `a32`←`a_ub` + `b32`←`b_ub` + `y_ub`←结果）** | | | **98304（96KB）** |

对照目标平台 UB 容量 **192KB**（`CONST-capacity-910B2C`：UB=192KB/AIV；version stamp `tilelang 0.1.2+3a214cde` + CANN 8.5.0）。按该条目 ×1.10–1.12 上偏系数（`--enable-auto-multi-buffer=false` 显式 alloc 结构下 BishengIR 实际需求 ≈ 手工预算 ×1.10–1.12）：

- **fp16/bf16 路径**：112KB ×1.12 ≈ 125KB < 192KB ✓，余量 ~35%。
- **fp32 路径（全 buffer 显式）**：128KB ×1.12 ≈ 143KB < 192KB ✓，余量 ~25%。
- **fp32 路径（复用方案，推荐）**：fp32 输入即 fp32，`a32`/`b32` 复用 `a_ub`/`b_ub`，结果直接 `T.copy` 回 GM（`y_ub` 可省）→ 96KB，×1.12 ≈ 108KB，余量 ~44%。

结论：全部 dtype 路径均不溢出，**无需单独调小 block_size**。Stage 3 实现时按 fp32 路径走复用链（`a32`←`a_ub`、`b32`←`b_ub`、`y_ub`←结果）。

### 4.6 动态轴定义

无动态轴。`N` 为 JIT 编译期参数（`add_fwd_kernel(N, dtype, alpha, ...)` 每 (N, dtype, alpha) 一特化，`@lru_cache`），`shape`（`T.int32`）为运行时传入的实际元素数（用于尾块 `shape - offset`，与 mish 同款）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素运算（cast / 标量乘 / 加 / cast），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_size = 8192        # 同族 elementwise 实测最优块宽（mish/hardsigmoid/hardswish）
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数
```

- **选择理由**：8192 为同族元素级实测最优 `block_size`（`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/mish.py`、hardsigmoid/hardswish 迁移设计同取）；32B 对齐（8192 × 2/4B 均 32B 整数倍）；整除向量宽（fp16/bf16 ×8、fp32 ×4）；UB 预算 96–128KB < 192KB（§4.5）；块宽足够使向量链发射开销摊薄（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。
- **与 NPU wrapper 脚手架一致**：`examples/TileOPs/tileops/kernels/elementwise/add/add.py` 的 `_DEFAULT_BLOCK_SIZE` 三 dtype 均为 8192（明确「取 Stage 1 设计值」）。
- **块宽敏感性**：发射/流量张力见 §1.6.0「设计期估算」备注与 §6.3；Stage 4 可扫描 block_size（受 UB 192KB 约束）与 `T.Pipelined`。

### 5.3 约束分析

- **对齐约束**：`block_size=8192`，fp16/bf16 尾轴 8192×2B=16KB、fp32 8192×4B=32KB，均 32B 对齐 ✓；整除向量宽 ✓。
- **UB 容量**：96–128KB < 192KB（§4.5）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾块**：`tail = T.min(block_size, shape - offset)`，仅 `T.copy` 有效元素进/出；v-prefix 对全 buffer 计算，尾部垃圾元素不写回（与 mish 同款）。
- **极小 shape**（N < 8192）：`n_num=1`，`num_cores=1`，单核单 tile 处理，无 oversubscription。
- **广播 workload**：Op 层已展平为 same-shape 1-D；kernel 侧与非广播无差别（§0.6 R2）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（Cube/混合直接用返回值、纯 Vector 翻倍），禁止文档假设/经验值。

- **② 物理核数依据（实查）**：查询代码与实际返回值记录于此（禁止文档假设/经验值替代）：

```python
from tilelang.utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 算子核数翻倍：40
```

实查命令与输出（2026-10-13 本机执行，toolchain tilelang 0.1.2 dev-root + CANN 8.5.2）：`get_aicore_num()` 返回 **20**。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。（备注：`CONST-aicore-910B2C` 记录 910B2C 为 24，与实查 20 不一致——设备档/型号差异，**一律以实查为准**，不以 24 假设。）

- **① 逻辑核数**：`num_logical_kernels = ceil(N / block_size)`，随 workload（NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `AddFwdOp`）变化：

| workload | A shape | B shape | N = prod(broadcast) | 逻辑核数 ceil(N/8192) | 每核串行 tile 数 ceil(num_logical/40) |
|----------|---------|---------|---------------------|----------------------|--------------------------------------|
| hidden-state-prefill | [2048,4096] | [2048,4096] | 8,388,608 | 1024 | 26 |
| cnn-feat-broadcast | [16,256,56,56] | [256,1,1] | 12,845,056 | 1568 | 40 |
| llama-7b-ffn | [1024,11008] | [1024,11008] | 11,272,192 | 1376 | 35 |
| linear-bias | [1024,4096] | [1,4096] | 4,194,304 | 512 | 13 |

- **③ 规模判定与分核方案**：全部 workload 的逻辑核数（512–1568）**远超**物理向量核数 40，属**极大规模**。设计采用**统一 persistent 结构**（成熟 elementwise 形态，VP-2026-0003 纯 Vector 一维逐元素 persistent 模板：`num_kernels=min(ceil(N/block), aicore×2) + T.serial 静态边界 + 动态尾块零起点切片`；`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py` 实测 nc=40 全档贴 copy 地板）：

  - **分核方案**：`num_kernels = min(ceil(N/block_size), 40)`，核内 `T.serial(num_local_tasks)` 串行处理多个逻辑块（`num_local_tasks = T.ceildiv(n_num, num_kernels)`，编译期常量，**循环边界静态**）：
    - **小规模（逻辑核数 ≤ 40，即 N ≤ 327680）**：`num_kernels = n_num`，每核 ≤1 tile，`num_local_tasks=1`，无串行循环、无负载不均——结论「**无需适配**」（逻辑核数 ≤ 物理核数）。
    - **中 / 极大规模（逻辑核数 > 40，如四 workload 的 512–1568）**：固定启动内核数 = 40（= 物理向量核数），核内**串行**处理 `num_local_tasks` 个 tile（hidden 26；cnn-feat 40；llama 35；linear-bias 13），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均；循环边界 `num_local_tasks` 为编译期常量（N/block_size/40 均编译期已知），满足「**核内串行**边界静态」。

---
## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_cores, is_npu=True) as (cid, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（vcast / vmul / vadd） | 块内逐元素由 Vector 单元并行（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_cores, is_npu=True) as (cid, _):
    # UB buffer 分配 ...
    for i in T.serial(num_local_tasks):          # 核内串行 tile 循环（静态边界）
        block_id = i * num_cores + cid
        if block_id < n_num:                     # 尾 tile guard
            offset = block_id * block_size
            tail = T.min(block_size, shape - offset)
            # 装载 -> v-prefix 计算链（alpha==1 折叠 / alpha!=1 标量乘）-> 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对主路径采用 guard-free 2-stage `T.Pipelined(num_local_tasks, num_stages=2)` 双缓冲流水（mish perf_opt 形态），将向量链隐藏于 MTE2 窗口（PL-1.6-copy-floor：≥16M 档计算链被完全隐藏）。**这是消除 §1.6.0「发射不可跨 tile 重叠则可能上升」风险的直接手段**。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 shape：`tail = T.min(block_size, shape - offset)`；`T.copy` 仅装载/写回 `[0:tail]` 有效元素；v-prefix 对全 block_size buffer 计算，垃圾尾部不写回（与 mish 同款）。`block_id < n_num` guard 覆盖最后一波不完全 tile。manifest workload 的 N 均为 2 的幂或 8192 整除（cnn-feat 12,845,056 = 8192 × 1568 整除），尾块路径服务任意广播 shape 的动态 N。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer）。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器自动管理 GM↔UB 搬运与向量计算间的依赖（`T.copy` 与后续 v-prefix op、以及各 v-op 之间的数据依赖由自动同步保证），无跨核协作（逐元素无跨元素依赖，各核独立）。

### 7.3 pass_configs 配置

Developer 模式默认配置，无需额外 `pass_configs`。环境变量 `TILELANG_ASCEND_MODE = "Developer"`（可选，默认即 Developer；NPU wrapper 声明 `ascend_mode = "Developer"`）。`alpha==1` 折叠与非折叠均为 trace-time 选定的单一程序，无运行时分支。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：Golden 以 **§0.1 源算子语义**为唯一依据（广播、fp32 opmath、输出同 dtype），移植源仓测试基准（`examples/TileOPs/tests/ops/test_add.py` 的 `ref_program = torch.add(a.float(), b.float(), alpha).to(a.dtype)`），**不复刻 §0.6 的 NPU 算法**（不含 vmul/vadd fp32 链、persistent 结构、Op 层展平）。

```python
import torch

def golden_add_fwd(input, other, alpha=1):
    """Add 参考实现：y = input + alpha * other，numpy 广播，fp32 opmath。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_add.py 的
    ref_program = torch.add(a.float(), b.float(), alpha=self.alpha).to(a.dtype)）。
    独立于 NPU 算法：仅依赖 torch.add，广播由 torch 原生处理。
    """
    a = input.detach().cpu()
    b = other.detach().cpu()
    y = torch.add(a.float(), b.float(), alpha=alpha)   # fp32 计算 + 广播
    return y.to(a.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓测试 `_get_tolerances` 一致。kernel 用完整 fp32 中转链，对齐 torch fp32 opmath golden，见 PL-1.17-subfp32-fp32-transit；§1.6.1 机器验证候选 vs golden 最大 0.0128 tolerance-unit。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`N=1048576`（shape `[1024,1024]`，same-shape），`float32` / `float16` / `bfloat16` 三 dtype，`randn` 输入，vs golden 在对应容差下 `torch.testing.assert_close`；另覆盖 `alpha ∈ {0.5, 2.0}` 非默认路径与广播路径（如 `[64,128]×[1,128]`）。完整分层套件（L1 多 shape / 非整除、L2 边界 size、Boundary 特殊值含 ±inf/NaN/-0.0）交由 `tilelang-op-develop` 规划，本设计不枚举。

**边界用例要点**：加法无专用 NaN 链，NaN/±inf 由 `vadd`/`vmul` 的 IEEE 语义自然传播——`NaN → NaN`、`inf + (-inf) → NaN`；`alpha==0` 路径**不得**折叠为 `y=a`（`0*inf=NaN`，§1.6.0 R1 候选 3）；`alpha==1` 折叠对 `±inf/NaN/-0.0` 逐位保持（§1.6.1 项 1 机器验证）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| bf16 无原生 v-prefix 算术 | 官方 dtype 表对 `T.vadd`/`T.vmul` 的 bf16 列标 ×（`docs/Tilelang.language/数学操作/T.vadd.md` / `T.vmul.md`），故 bf16 必经 `T.vcast(bf16→fp32, rint)` 中转、算完再 `T.vcast(fp32→bf16)`；也顺带对齐 torch fp32 opmath golden（PL-1.17-subfp32-fp32-transit） |
| 广播 Op 层物化代价 | 广播 workload（cnn-feat / linear-bias）在 Op 层 `expand→contiguous` 物化广播操作数（写 ~N 元素），设计期估 ≈+21µs（fp16）；属已知代价（§0.6 R2）。**备选（Stage 4 A/B）**：kernel 内 `T.vbrc` 行/列广播（免物化），需按 workload 特化并规避 `TRAP-vbrc-same-shape-empty-broadcast`（同形 src/dst 空广播维 verify fail）；主选（Op 层展平）保持结构最简 |
| ascend-constraints §4（host 侧约束） | Op 层 `expand` 为 stride-0 视图（不改输入真实数据/物理排布）、`contiguous` 产出**新**张量（不改动输入张量）；符合「只做 stride/shape 元数据视图操作 + 不触发 padding」。同族 NPU 先例（lerp_tensor / add scaffold）已确立该 Op 层预广播契约（`examples/TileOPs/docs/gpu_to_npu_adaptation.md` §3.4 E15） |
| torch.add 与两轮 fp32 的舍入口径 | torch `torch.add(...,alpha)` 在部分幅值下采用单次舍入（fma/宽域），纯两轮 fp32 链（vmul→vadd）与 golden 最大差 0.0128 tolerance-unit（<1，容差带内，0 违反；§1.6.1）。L0 用 randn（O(1)）无影响；极端大幅值角点由容差覆盖 |
| 发射/流量张力（设计期估算） | 大 N 档流量项为下界；若发射成本不可跨 tile 重叠，1568 tile 的发射项可能上升 → block_size 增大 / `T.Pipelined` 为 Stage 4 首要调优轴（§6.3）。属**设计期未实测假设**，Stage 4 实测标定 |
| 分核 | 物理核数实查 20（×2=40），与 `CONST-aicore-910B2C`（24）不一致——本环境设备返回 20，以实查为准 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_size 过大（>8192 × fp32 全 buffer 链） | 编译失败（`ub overflow`） | 维持 block_size=8192；fp32 路径复用 buffer（§4.5） |
| bf16 原生 vadd/vmul 编译失败 | 直接对 bf16 buffer 调 `T.vadd`/`T.vmul` | 编译/运行失败 | 先 `T.vcast(bf16→fp32)` 中转（§3.2 / §9.1） |
| 原地 v-op 误判 | `T.vadd(a32, b32, a32)` 类 dst==src | 误以为非法 | `T.vadd`/`T.vmul` 文档明确支持 dst 与 src 同 buffer（§2.3） |
| 尾块越界 | `T.copy` 用 base+size 形态写尾块 | MTE 越界写坏相邻 tensor | 双显式 slice `T.copy(src[0:tail], dst[0:tail])`（TRAP-T-copy-region-semantics） |
| `dtype=` 关键字 false alarm | `T.Tensor((N,), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（design-template §3.3） |
| alpha==0 误折叠 | 把 `alpha==0` 折为 `y=a` | `b=±inf` 时丢 NaN，语义错误 | 仅折叠 `alpha==1`（§1.6.1 项 1；候选 3 被拒） |

### 9.3 特殊场景处理

- **非整除 shape**（任意广播 shape）：尾块 `tail` + `block_id < n_num` guard（§6.4）。
- **极小 shape**（N < block_size）：`n_num=1`，单核单 tile。
- **广播 shape**：Op 层展平，kernel 无感知（§0.6 R2）；输出 `view(out_shape)` 回塑。
- **NaN / ±inf 输入**：自然传播（§8.3）。
- **alpha 非默认**：构造期特化（`@lru_cache` + op body 分支），非默认 alpha 走同一 kernel 模板。
- **混合精度**：fp16/bf16 走 fp32 中转；fp32 直算；输出 dtype 与输入一致。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/add/add_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，结果内嵌）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成 | §1.6.1 等价性验证（135 行全 EQUIV_PASS） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `add_fwd_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_add.py` | ⬜ 待实现（Stage 3） | 精度测试 |

### 10.3 命名规范

- 项目目录名：`add`；算子目录名：`add_fwd_kernel`；接口函数：`add_fwd_kernel`（manifest 键 `AddFwdOp`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`add_fwd_kernel`）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）
