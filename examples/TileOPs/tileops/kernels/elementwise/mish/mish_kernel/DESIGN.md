# Mish 算子设计文档

## 0. 原始计算逻辑分析（迁移类任务）

### 0.1 功能概述

Mish 激活函数：`y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))`，参考 `torch.nn.functional.mish`。

### 0.2 输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| x | (N,) | float16 / bfloat16 / float32 | 1-D 展平向量（原始 shape 被 flatten） |
| y | (N,) | same_as(x) | 输出，shape/dtype 与输入一致 |

### 0.3 详细解读

GPU 源码路径：`examples/TileOPs/tileops/kernels/elementwise/mish/_mish_fwd_kernels.py`

GPU 实现要点：
- 工厂函数 `mish_fwd_kernel(N, dtype, output_dtype, threads=256, num_per_thread=8)` 闭包生成 kernel
- `@tilelang.jit(out_idx=[1])` 装饰内部函数 `kernel(threads_arg, npt_arg)`
- `@T.prim_func` 装饰 `main(x, y)`，x/y 均为 1-D `(N,)` 张量
- `block_size = threads * num_per_thread = 256 * 8 = 2048`
- GPU Kernel：`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg)`
- 数据流：`T.copy(GM → fragment)` → `T.Parallel(threads, npt)` 逐元素融合计算 → `T.copy(fragment → GM)`
- 计算公式（标量级融合）：`y[i] = x[i] * T.tanh(T.log(T.cast(1.0,'float32') + T.exp(x[i])))`
- 关键精度策略：`one = T.cast(1.0, 'float32')`，使 `1 + exp(x)` 在 float32 域计算（即使输入为 fp16/bf16 也自动提升），防止 exp 溢出并保持精度

### 0.4 标杆实现

```python
# PyTorch golden（与 torch.nn.functional.mish 等价）
def golden_mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))
    # 等价于: x * torch.tanh(torch.log(1 + torch.exp(x)))
```

---

## 1. 概述

### 1.1 算子名称

`mish_fwd_kernel`

### 1.2 功能描述

Mish 激活函数前向计算：对 1-D 展平输入向量逐元素应用 `y = x * tanh(log(1 + exp(x)))`，支持 float16 / bfloat16 / float32。

### 1.3 数学公式

$$
y_i = x_i \cdot \tanh\big(\ln\big(1 + e^{x_i}\big)\big), \quad i = 0, 1, \ldots, N-1
$$

### 1.4 算法描述

Mish 为多步逐元素运算，拆解为 5 个子步骤：

1. `exp(x)` — 自然指数
2. `1 + exp(x)` — 标量加法（softplus 的被减数部分）
3. `ln(1 + exp(x))` — 自然对数（softplus 结果）
4. `tanh(ln(1 + exp(x)))` — 双曲正切
5. `x * tanh(...)` — 逐元素乘法（最终 Mish 输出）

### 1.5 数据流图

```
GM[x] --T.copy--> UB[x_ub] --vcast(if fp16/bf16)--> UB[x_f32]
  UB[x_f32] --vexp--> UB[exp_ub]
  UB[exp_ub] --vadd(.,1.0)--> UB[sp_ub]
  UB[sp_ub] --vln--> UB[ln_ub]
  UB[ln_ub] --vtanh--> UB[tanh_ub]
  UB[x_f32] + UB[tanh_ub] --vmul--> UB[y_f32]
  UB[y_f32] --vcast(if fp16/bf16)--> UB[y_ub] --T.copy--> GM[y]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素运算），无 matmul | 不需要 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需要 Expert 级 buffer 控制 |
| 是否需要流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需要 CV 融合 |
| 超越函数 | exp/ln/tanh 有专用 v 前缀 API | 用 v-prefix API 链式调用 |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 多步运算（5步）→ 无需精细 buffer 控制 → Developer 模式`。

参考实现：`examples/elementwise/example_elementwise_exp2.py`、`examples/elementwise/example_elementwise_log2.py` 均为 Developer 模式 + `T.alloc_ub` + v-prefix API，无手动同步。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式 UB 分配（v-prefix API 要求 buffer 在 UB 上） |
| 计算方式 | v-prefix API 链式调用（vexp → vadd → vln → vtanh → vmul），非 T.Parallel 标量循环 |
| 同步 | Developer 自动同步（线性数据依赖，编译器自动插入 barrier） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `exp_x = exp(x)` | 自然指数 |
| 2 | `sp = 1 + exp_x` | 标量加法（softplus 被减数） |
| 3 | `ln_sp = ln(sp)` | 自然对数（softplus 结果） |
| 4 | `tanh_sp = tanh(ln_sp)` | 双曲正切 |
| 5 | `y = x * tanh_sp` | 逐元素乘法 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| Cast-in | `x_f32 = cast(x, fp32)` | `T.vcast(src, dst, round_mode="rint")` | src=x_ub(input dtype), dst=x_f32_ub(float32) | Developer |
| 1 | `exp_x = exp(x)` | `T.vexp(src, dst)` | src=x_f32_ub, dst=exp_ub (float32) | Developer |
| 2 | `sp = 1 + exp_x` | `T.vadd(src0, src1, dst)` | src0=exp_ub, src1=`T.cast(1.0,"float32")`(标量), dst=sp_ub | Developer |
| 3 | `ln_sp = ln(sp)` | `T.vln(src, dst)` | src=sp_ub, dst=ln_ub (float32) | Developer |
| 4 | `tanh_sp = tanh(ln_sp)` | `T.vtanh(src, dst)` | src=ln_ub, dst=tanh_ub (float32) | Developer |
| 5 | `y = x * tanh_sp` | `T.vmul(src0, src1, dst)` | src0=x_f32_ub, src1=tanh_ub, dst=y_f32_ub (float32) | Developer |
| Cast-out | `y = cast(y_f32, out_dtype)` | `T.vcast(src, dst, round_mode="rint")` | src=y_f32_ub(float32), dst=y_ub(output dtype) | Developer |

> **注意**：Cast-in/Cast-out 仅在 input dtype 为 float16/bfloat16 时执行；float32 输入直接计算，无需 cast。

### 3.3 计算伪代码

遵循迁移规则：`@T.prim_func` 函数参数名 `x, y` 及顺序不变，新增 `shape: T.int32` 用于动态 N 尾块处理。`T.Tensor` 的 dtype 改为第二位置参数（非关键字），遵循模板 §3.3 规则。

```python
@tilelang.jit(out_idx=[1], target="npuir")
def mish_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    out_dtype = output_dtype or dtype
    n_num = T.ceildiv(N, block_size)
    compute_dtype = "float32"  # 中间计算统一使用 float32

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # ---- 1. 分配 UB buffer ----
            # 输入/输出 dtype buffer（GM 交互）
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)
            # float32 计算 buffer（可复用）
            x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
            buf1_ub = T.alloc_ub((block_size,), compute_dtype)   # exp -> ln -> result
            buf2_ub = T.alloc_ub((block_size,), compute_dtype)   # softplus -> tanh

            # 标量常量（Kernel 内、Scope 外定义）
            one = T.cast(1.0, compute_dtype)

            # ---- 2. 尾块计算 ----
            offset = cid * block_size
            remaining = shape - offset
            tail_size = T.min(block_size, remaining)

            # ---- 3. 数据搬入 ----
            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

            # ---- 4. Cast 到 float32（fp16/bf16 时）----
            if dtype != compute_dtype:
                T.vcast(x_ub[0:tail_size], x_f32_ub[0:tail_size], round_mode="rint")
            else:
                T.copy(x_ub[0:tail_size], x_f32_ub[0:tail_size])

            # ---- 5. 计算: y = x * tanh(ln(1 + exp(x))) ----
            T.vexp(x_f32_ub[0:tail_size], buf1_ub[0:tail_size])          # buf1 = exp(x)
            T.vadd(buf1_ub[0:tail_size], one, buf2_ub[0:tail_size])      # buf2 = 1 + exp(x)
            T.vln(buf2_ub[0:tail_size], buf1_ub[0:tail_size])           # buf1 = ln(1+exp(x))
            T.vtanh(buf1_ub[0:tail_size], buf2_ub[0:tail_size])          # buf2 = tanh(ln(...))
            T.vmul(x_f32_ub[0:tail_size], buf2_ub[0:tail_size], buf1_ub[0:tail_size])  # buf1 = x * tanh(...)

            # ---- 6. Cast 回输出 dtype（fp16/bf16 时）----
            if out_dtype != compute_dtype:
                T.vcast(buf1_ub[0:tail_size], y_ub[0:tail_size], round_mode="rint")
            else:
                T.copy(buf1_ub[0:tail_size], y_ub[0:tail_size])

            # ---- 7. 数据搬出 ----
            T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main
```

> **伪代码说明**：`if dtype != compute_dtype` 为伪代码层面条件编译示意，实际实现中可通过 Python 层 `if dtype == "float32"` 分支控制（JIT 参数在编译期确定），不产生运行时分支。`T.vcast`/`T.vexp`/`T.vln`/`T.vtanh`/`T.vadd`/`T.vmul` 均支持 region slice `[0:tail_size]` 操作（参考 `vec_add_1d.py` 中 `A_VEC[0:tail_size]` 用法）。

### 3.4 API 可行性确认

| API | 来源 | 文档路径 | 验证状态 |
|-----|------|----------|----------|
| `T.vexp(src, dst)` | `tilelang/language/customize_npuir.py:295` `npuir_exp` | `docs/Tilelang.language/数学操作/T.vexp.md` | ✅ 已验证：fp16✓ fp32✓ bf16✗ |
| `T.vln(src, dst)` | `customize_npuir.py:334` `npuir_ln` | `docs/Tilelang.language/数学操作/T.vln.md` | ✅ 已验证：fp16✓ fp32✓ bf16✗ |
| `T.vtanh(src, dst)` | `customize_npuir.py:1634` `npuir_vtanh` | `docs/Tilelang.language/数学操作/T.vtanh.md` | ✅ 已验证：fp16✓ fp32✓ bf16✗ |
| `T.vadd(src0, src1, dst)` | `customize_npuir.py:116` `npuir_add` | `docs/Tilelang.language/数学操作/T.vadd.md` | ✅ 已验证：支持标量 src1，支持原地 dst=src0 |
| `T.vmul(src0, src1, dst)` | `customize_npuir.py:144` `npuir_mul` | `docs/Tilelang.language/数学操作/T.vmul.md` | ✅ 已验证：支持标量 src1，支持原地 dst=src0 |
| `T.vcast(src, dst, round_mode)` | `customize_npuir.py:750` `npuir_cast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | ✅ 已验证：bf16→f32✓ f32→bf16✓ f16→f32✓ f32→f16✓ |
| `T.alloc_ub(shape, dtype)` | examples 多处使用 | — | ✅ 已验证：exp2/log2/vec_add 示例均使用 |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 1-D 输入，`T.Kernel(1D, is_npu=True)`，不涉及 |
| 部分 GPU API 不可用 | Yes | GPU 使用 `T.exp`/`T.log`/`T.tanh` 标量融合 + `T.Parallel` + `threads=`；NPU 改为 v-prefix API 链式调用 + `is_npu=True` |
| GEMM 非整除 | No | 纯 Vector 算子，无 GEMM |
| L0C 容量上限 | No | 纯 Vector，不使用 L0C |

### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend npuir） | 转换方案 |
|--------|----------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1: 添加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(blocks, threads=threads_arg)` | `T.Kernel(blocks, is_npu=True)` | K2/K3: 移除 `threads=`，添加 `is_npu=True` |
| 并行模型 | `T.Parallel(threads, npt)` 标量循环 | v-prefix API 向量指令链 | K11: `threads * num_per_thread` 折叠为 `block_size` |
| 计算精度 | `one = T.cast(1.0,'float32')` 使标量加法提升到 fp32 | 全链路 fp32 中间计算 + vcast | fp16/bf16 先 vcast→fp32，计算后 vcast 回原 dtype |
| 超越函数 | `T.exp`/`T.log`/`T.tanh`（TIR 标量 op） | `T.vexp`/`T.vln`/`T.vtanh`（NPU 向量 op） | AGENTS.md 规则：v 前缀 API 优先 |
| Buffer 分配 | `T.alloc_fragment`（寄存器） | `T.alloc_ub`（UB） | NPU v-prefix API 要求 buffer 在 UB |
| 尾块处理 | 隐式（T.copy 切片可能越界，GPU 不检测） | 显式 `shape` 参数 + `tail_size` 动态裁剪 | 参考 `examples/elementwise/vec_add_1d.py` |
| prim_func 参数 | `main(x, y)` | `main(x, y, shape: T.int32)` | 新增 `shape` 用于动态 N 尾块，不改变 x/y 顺序与名称 |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似 | 1-D elementwise 模式、`T.alloc_ub` + v-prefix API、`shape` 动态参数 + `tail_size` 尾块处理、`T.Kernel(n_num, is_npu=True)` |
| `examples/elementwise/example_elementwise_exp2.py` | 高度相似 | `T.vexp2` unary v-prefix API 用法、`T.alloc_ub` + Tmp buffer 模式、persistent kernel（`T.serial` 多 tile 迭代） |
| `examples/elementwise/example_elementwise_log2.py` | 高度相似 | `T.vlog2` unary v-prefix API 用法、与 exp2 对称的结构 |
| `examples/elementwise/example_elementwise_add.py` | 中度相似 | Developer 模式 `@tilelang.jit(target="npuir")` 完整结构、`T.alloc_shared` + `T.Parallel` 标量计算模式（本项目改用 v-prefix API） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| x | (N,) | float16 / bfloat16 / float32 | 1-D 展平输入向量，N 为动态维度 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| y | (N,) | same_as(x) | 输出，shape/dtype 与输入一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| x_ub | (block_size,) | input dtype | UB | GM→UB 输入搬运（可复用为输出 cast 目标） |
| x_f32_ub | (block_size,) | float32 | UB | cast 后的 fp32 输入（参与最终 vmul） |
| buf1_ub | (block_size,) | float32 | UB | 中间结果：exp(x) → ln(1+exp(x)) → 最终 y_f32 |
| buf2_ub | (block_size,) | float32 | UB | 中间结果：1+exp(x) → tanh(ln(1+exp(x))) |
| y_ub | (block_size,) | output dtype | UB | cast 后的输出（GM→UB 输出搬运） |

> **buffer 复用策略**（伪代码中体现）：
> - step 1 输出 buf1=exp(x)，step 2 消费 buf1 后 buf1 空闲
> - step 3 输出 buf1=ln(1+exp(x))，step 4 消费 buf1 后 buf1 空闲
> - step 5 输出 buf1=y_f32（最终结果），buf2 消费后空闲
> - fp32 输入时 x_ub 和 y_ub 可合并（同 dtype），但为清晰性保持独立

### 4.4 内存搬运路径

```
纯 Vector 数据通路（无 Cube/L1/L0）:

fp16/bf16 路径:
  GM[x] --T.copy--> UB[x_ub(fp16/bf16)]
  UB[x_ub] --T.vcast--> UB[x_f32_ub(fp32)]
  UB[x_f32_ub] --T.vexp--> UB[buf1_ub(fp32)]
  UB[buf1_ub] --T.vadd(.,1.0)--> UB[buf2_ub(fp32)]
  UB[buf2_ub] --T.vln--> UB[buf1_ub(fp32)]
  UB[buf1_ub] --T.vtanh--> UB[buf2_ub(fp32)]
  UB[x_f32_ub] + UB[buf2_ub] --T.vmul--> UB[buf1_ub(fp32)]
  UB[buf1_ub] --T.vcast--> UB[y_ub(fp16/bf16)]
  UB[y_ub] --T.copy--> GM[y]

fp32 路径（无 cast）:
  GM[x] --T.copy--> UB[x_ub(fp32)] [= x_f32_ub]
  UB[x_ub] --T.vexp--> UB[buf1_ub]
  UB[buf1_ub] --T.vadd(.,1.0)--> UB[buf2_ub]
  UB[buf2_ub] --T.vln--> UB[buf1_ub]
  UB[buf1_ub] --T.vtanh--> UB[buf2_ub]
  UB[x_ub] + UB[buf2_ub] --T.vmul--> UB[buf1_ub]  [= y_ub]
  UB[buf1_ub] --T.copy--> GM[y]
```

### 4.5 UB 内存预算

**block_size = 2048**

| Buffer | Shape | dtype | 大小 (Bytes) | 说明 |
|--------|-------|-------|-------------|------|
| x_ub | (2048,) | fp16/bf16 (2B) | 4096 | 输入搬运 |
| x_f32_ub | (2048,) | fp32 (4B) | 8192 | cast 后输入 |
| buf1_ub | (2048,) | fp32 (4B) | 8192 | 中间结果 |
| buf2_ub | (2048,) | fp32 (4B) | 8192 | 中间结果 |
| y_ub | (2048,) | fp16/bf16 (2B) | 4096 | 输出搬运 |
| **总计 (fp16/bf16)** | | | **32768 (32KB)** | / 192KB (A2/A3 UB) ✓ 16.7% |
| **总计 (fp32)** | | | **32768 (32KB)** | 4 buffers × 2048 × 4B / 192KB ✓ |

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| N | `shape: T.int32`（prim_func 参数） | 1 ~ 数千万（manifest 最大 26,214,400） |

> N 同时作为编译期 `T.Tensor((N,), dtype)` 的 shape 和运行时 `shape` 参数。编译按特定 N 特化（lru_cache 模式），`shape` 参数用于尾块动态裁剪。manifest 所有 workload 的 N 均能被 block_size=2048 整除，但设计保留尾块处理以支持任意 N。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],           # y 是 prim_func 的第 2 个参数（index=1）
    target="npuir",        # NPU IR 后端
)
def mish_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅包含逐元素运算（exp/ln/tanh/add/mul），无 matmul，无归约，判定为纯 Vector。仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = 2048   # K11: GPU threads(256) * num_per_thread(8) 折叠为单一参数
                     # 选择理由:
                     #   1. 与 GPU 源码一致（256 * 8 = 2048）
                     #   2. UB 对齐: 2048 * 4B(fp32) = 8192B，是 32B 对齐的整数倍 ✓
                     #   3. UB 容量: 5 buffers × 8192B = 40KB < 192KB ✓
                     #   4. 足够大的 tile 减少 GM 访问次数

n_num = T.ceildiv(N, block_size)   # 总 block 数
```

### 5.3 约束分析

- **对齐约束**: UB 要求 32B 对齐。block_size=2048，fp32 元素 4B → 2048×4=8192B = 256×32B ✓；fp16 元素 2B → 2048×2=4096B = 128×32B ✓
- **UB 容量**: 总 buffer = 32KB（fp16/bf16）或 32KB（fp32）< 192KB（A2/A3 UB）✓ 占用仅 16.7%
- **L0/L1 容量**: 不适用（纯 Vector，不使用 L0/L1）
- **分形限制**: 不适用（无 Cube 计算）

### 5.4 尾块处理策略

**策略**: Kernel 内动态 block（参考 `examples/elementwise/vec_add_1d.py`）

```python
offset = cid * block_size
remaining = shape - offset
tail_size = T.min(block_size, remaining)

# 仅搬运有效元素
T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
# v-prefix API 对整个 buffer 计算（尾部垃圾元素不影响输出，仅 copy 出 tail_size 个有效元素）
# ...compute...
T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])
```

**非整除场景**:
- 当 `N % block_size != 0` 时，最后一个 block 的 `tail_size < block_size`
- v-prefix API 仍对整个 `block_size` buffer 计算（尾部为未初始化数据，NaN/Inf 传播不影响输出，因仅 copy 有效部分）
- 无需 host 侧 padding（遵守 Host 侧输入操作约束：不修改输入张量真实内容）

**manifest workload 整除验证**:
| Workload | N | N % 2048 | 尾块? |
|----------|---|----------|-------|
| smoke | 1,048,576 | 0 | No |
| yolo-p3 | 16×256×80×80 = 26,214,400 | 0 | No |
| yolo-p4 | 16×512×40×40 = 13,107,200 | 0 | No |
| fc-wide | 2048×4096 = 8,388,608 | 0 | No |

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级 | 1-D 并行 | `T.Kernel(n_num, is_npu=True)` | 每个 block 处理一个 block_size 分块 |
| 元素级 | 向量指令（无显式循环） | `T.vexp`/`T.vln`/`T.vtanh`/`T.vadd`/`T.vmul` | v-prefix API 对整个 buffer 一次完成，无需 T.Parallel |
| 尾块 | 标量条件 | `T.min(block_size, remaining)` | 动态裁剪有效元素数 |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(n_num, is_npu=True) as (cid, _):
    # block 内无显式循环——v-prefix API 对整 buffer 操作
    # 计算链: vexp → vadd → vln → vtanh → vmul
    # 每步都是一个完整的向量指令，编译器自动调度
```

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：
- 单 tile 计算（5 步 v-op）即可完成一个 block，无 K 维迭代
- UB 占用仅 16.7%，但单 tile 内数据依赖线性（step N 依赖 step N-1），无法并行化不同 step
- 若需进一步优化（性能调优阶段），可考虑 persistent kernel + double buffer（参考 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 迭代模式），但设计阶段保持简洁

### 6.4 尾块处理

见 §5.4。当 `N % block_size != 0` 时，最后 block 使用 `tail_size = T.min(block_size, N - cid * block_size)` 动态裁剪。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。数据依赖链为严格线性：

```
T.copy(GM→UB) → vcast → vexp → vadd → vln → vtanh → vmul → vcast → T.copy(UB→GM)
```

每一步都依赖前一步的输出，编译器自动在相邻 v-op 之间插入 `pipe_barrier` / MTE 等待。无需手动 `T.sync_block_set`/`T.sync_block_wait`。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| T.copy(GM→UB) 后 | 自动 | Developer 模式编译器插入 MTE2→V 等待 |
| 各 v-op 之间 | 自动 | 线性数据依赖，编译器插入 V→V barrier |
| T.copy(UB→GM) 前 | 自动 | Developer 模式编译器插入 V→MTE3 等待 |

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs
# os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # 可选，默认即为 Developer
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
import torch

def golden_mish(x):
    """Mish 参考实现，与 torch.nn.functional.mish 等价.

    对 fp16/bf16 输入，先上转 fp32 计算再转回，匹配 NPU kernel 精度策略。
    """
    x_f32 = x.to(torch.float32)
    # y = x * tanh(ln(1 + exp(x)))
    y_f32 = x_f32 * torch.tanh(torch.log(1 + torch.exp(x_f32)))
    return y_f32.to(x.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源: `test_mish.py` |
| float16 | 1e-3 | 1e-3 | 来源: `test_mish.py`；NPU kernel 在 fp32 域计算，误差主要来自 vcast 舍入 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源: `test_mish.py`；bf16 精度低，容忍度放宽 |

### 8.3 L0 门槛测试计划

> L0 为代表性门槛测试（验证核心路径正确性）。完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop` 阶段枚举。

| 测试名 | Shape | dtype | atol | rtol | 代表性说明 |
|--------|-------|-------|------|------|-----------|
| L0-smoke-fp16 | [1048576] | float16 | 1e-3 | 1e-3 | 最具代表性：1M 元素 + fp16，覆盖 smoke workload |
| L0-smoke-fp32 | [1048576] | float32 | 1e-5 | 1e-5 | fp32 路径验证（无 cast 分支） |
| L0-smoke-bf16 | [1048576] | bfloat16 | 1.6e-2 | 1.6e-2 | bf16 路径验证（含 cast 双向） |

**L0 测试要点**:
- 输入数据使用 `torch.randn` 生成，覆盖正负值（Mish 对负值有特殊行为：`mish(-1) ≈ -0.30`）
- 额外测试大正值（如 `x = torch.randn(...) * 5`）验证 exp 不溢出（fp32 中间计算保证）
- 输入需 `.npu()` 放置到 NPU 设备
- Golden 在 CPU 上用 fp32 计算，再 `.to(input.dtype)` 转回

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **bf16 不被 v-prefix 数学 API 支持** | vexp/vln/vtanh/vadd/vmul 均不支持 bf16 | 全链路 fp32 中间计算：`vcast(bf16→fp32)` → 计算 → `vcast(fp32→bf16)` |
| **fp16 exp 溢出风险** | `exp(x)` 对 x > 11 在 fp16 下溢出（65504） | fp16 也走 fp32 中间计算路径，与 GPU 源码 `one=T.cast(1.0,'float32')` 行为一致 |
| **v-prefix API 不支持 T.Parallel 内调用** | 不能像 GPU 那样在 T.Parallel 循环内融合标量 exp/log/tanh | 改为 v-prefix API 链式调用，每步一个向量指令 |
| **1-D buffer 与 v-add 推荐差异** | vadd 文档推荐 2D `[M,N]` buffer | `vec_add_1d.py` 已验证 1-D buffer 可用；1-D 是 Mish 的自然形态 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bf16 直接传入 vexp | 未做 vcast 上转 | 编译错误或运行时错误 | 检查 dtype，fp16/bf16 必须先 vcast 到 fp32 |
| exp 溢出 | fp16 直接计算 exp(x), x>11 | Inf 传播 | 确保 fp32 中间计算路径 |
| UB 未初始化尾部数据 | 尾块 tail_size < block_size | 无影响（仅 copy 有效部分），但 NaN 传播可能触发硬件异常 | v-prefix API 对全 buffer 计算，NaN/Inf 传播不影响输出 |
| shape 参数缺失 | prim_func 未声明 shape | 无法动态裁剪尾块 | 参考 vec_add_1d.py，shape 作为 T.int32 参数 |
| vcast round_mode 不支持 | 使用非 "rint" 模式做 f16→f32 | 编译错误 | f16→f32 和 bf16→f32 仅支持 "rint"；f32→f16/bf16 支持 "round"/"rint" 等 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail_size` 动态裁剪，无需 host padding |
| **极小 N（N < block_size）** | n_num=1，tail_size=N，正常工作 |
| **极大 N（N > 10M）** | n_num 很大（如 12800），NPU runtime 分配 block 到多核；若性能不足可在优化阶段改用 persistent kernel |
| **混合精度** | output_dtype != input_dtype 时，cast-out 使用 out_dtype（`out_dtype = output_dtype or dtype`） |
| **全零输入** | `mish(0) = 0 * tanh(ln(1+1)) = 0 * tanh(0.693) = 0`，无特殊处理 |
| **大负值输入** | `exp(x) → 0`, `ln(1+0) = 0`, `tanh(0) = 0`, `y = x * 0 = 0`，fp32 下无溢出 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/elementwise/mish/
├── mish.py             # 算子实现（kernel + golden + L0 测试）
├── DESIGN.md           # 本设计文档
└── README.md           # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `mish.py` | ⬜ 待实现 | 算子实现：`mish_fwd_kernel` kernel + `golden_mish` + L0 测试 |
| `test_mish.py` | ⬜ 待实现 | 完整分层测试（L1/L2/Boundary），放入 `testing/npuir/` |

### 10.3 命名规范

- 项目目录名: `elementwise`
- 算子目录名: `mish`
- 实现文件: `mish.py`（`@tilelang.jit` 装饰的函数名为 `mish_fwd_kernel`，保留 GPU 源码命名）
- 测试文件: `test_mish.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ⬜ Golden 函数（`golden_mish`，验证基准）
3. ⬜ 算子实现（`mish.py`：`mish_fwd_kernel` kernel）+ 与 Golden 函数精度比对
4. ⬜ L0 门槛测试（3 个代表性 case：fp16/fp32/bf16 × [1048576]）

### 10.5 迁移执行规则遵循确认

| 规则 | 遵循情况 |
|------|----------|
| 1. `@tilelang.jit` 装饰函数命名为 `mish_fwd_kernel` | ✅ 保留 GPU 源码工厂函数名作为 NPU jit 函数名 |
| 2. Kernel 函数参数语义保持（threads/npt 折叠为 block_size） | ✅ K11: `threads(256) * num_per_thread(8) = block_size(2048)` |
| 3. `@T.prim_func` 函数参数名 `x, y` 及顺序不变 | ✅ `main(x, y, shape)` — x/y 顺序与名称保持，新增 shape 用于 NPU 尾块 |
| 4. 从源码工程推断输入张量规格 | ✅ 已从 manifest 推断：1-D (N,) 展平，fp16/bf16/fp32 |
| K1-K4: `target="npuir"`, `is_npu=True`, 移除 `threads=` | ✅ |
| K11: threads * num_per_thread → block_size | ✅ |
