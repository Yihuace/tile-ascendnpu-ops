# _logsumexp_kernel_single 算子设计文档

## 0. 原始计算逻辑分析（迁移类任务）

### 0.1 功能概述

GPU 版 `_logsumexp_kernel_single` 实现单 tile logsumexp 前向核：对 `(M, N)` 输入张量沿最后一维（N 维）做 logsumexp 归约，输出 `(M,)` 向量。该变体假设 N 可整体放入 shared memory（single-tile path），大 N 的 tiled 变体不在本次迁移范围。

### 0.2 输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `x` | 输入 | `(M, N)` | float16 / bfloat16 / float32 | 2D 输入张量；M=非规约维乘积, N=规约维 |
| `y` | 输出 | `(M,)` | same_as(x) | 每行 logsumexp 结果 |

### 0.3 详细解读

GPU 源文件：`examples/TileOPs/tileops/kernels/reduction/_logsumexp_kernel_single.py`

GPU 核心逻辑：

1. `@tilelang.jit(out_idx=[1])` 装饰 `_func(block_m, threads)`，返回 `main` prim_func。
2. `N_padded = align_up(N, DEFAULT_ALIGNMENT=256)` — 因 CUDA shared memory T.copy 需要 256 元素对齐。
3. **两条路径**：
   - **padded 路径**（`_needs_pad=True`）：用 `T.Parallel` + `T.if_then_else` + `T.And` 做 masked element-wise load，越界列填 `-inf`，同时处理行尾（`M % block_m != 0`）。
   - **aligned 路径**（`_needs_pad=False`）：用 `T.copy` 快速搬运 `(block_m, N_padded)` 块。
4. **计算流程**（均在 fp32 fragment 上）：
   - `T.fill(row_max, -inf)` → `T.reduce_max(x_f32, row_max, dim=1, clear=False)`
   - `T.Parallel`: `x_f32[i,j] = T.exp(x_f32[i,j] - row_max[i])`
   - `T.reduce_sum(x_f32, row_sum, dim=1)`
   - `T.Parallel`: `out_local[i] = row_max[i] + T.log(row_sum[i])`
5. `T.copy(out_local, y[pid_m * block_m])` — 输出搬运。
6. `threads=threads` — GPU CUDA 线程参数。

### 0.4 标杆实现

- **PyTorch 参考**：`torch.logsumexp(x.float(), dim=-1).to(x.dtype)`
- **测试文件**：`examples/TileOPs/tests/ops/test_softmax.py`（LogSumExp 子集）
- **输出形状**：`(M,)`，无 keepdim。Op 层（`LogSumExpKernel` 类）负责把任意维输入 reshape 成 2D 并处理 keepdim。

---

## 1. 概述

### 1.1 算子名称

`_logsumexp_kernel_single`

### 1.2 功能描述

对 `(M, N)` 输入张量沿 N 维做数值稳定的 logsumexp 归约，输出 `(M,)` 向量。single-tile 路径假设 N 可整体放入 UB。

### 1.3 数学公式

$$
y[i] = \max_j(x[i,j]) + \ln\left(\sum_j \exp\left(x[i,j] - \max_j(x[i,j])\right)\right)
$$

参考 API：`torch.logsumexp`

### 1.4 算法描述

数值稳定的三步算法（per-row 独立）：

1. **reduce_max**：`m[i] = max_j(x[i,j])` — 每行取最大值
2. **shift + exp + reduce_sum**：`s[i] = sum_j(exp(x[i,j] - m[i]))` — 减去最大值后取指数再求和
3. **log + add**：`y[i] = m[i] + ln(s[i])` — 取对数后加回最大值

### 1.5 数据流图

```
GM[x] --T.copy--> UB[x_ub] --T.copy--> FRAG[x_local]
  --T.cast--> FRAG[x_f32]
  --T.reduce_max(dim=1)--> FRAG[row_max]
  --T.vsub(broadcast)--> FRAG[x_f32]  (x - max)
  --T.vexp(in-place)--> FRAG[x_f32]   (exp(x - max))
  --T.reduce_sum(dim=1)--> FRAG[row_sum]
  --T.vln(in-place)--> FRAG[row_sum]   (ln(sum))
  --T.vadd--> FRAG[row_sum]            (max + ln(sum))
  --T.Parallel+T.cast--> UB[out_ub]    (fp32 → original dtype)
  --T.copy--> GM[y]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（归约 + 逐元素运算），无 matmul | 不需要 Cube/L0 |
| 归约 | reduce_max + reduce_sum | Developer 模式下可用（决策树确认） |
| 内存层级 | 仅 GM ↔ UB，无 L1/L0 参与 | Developer 的 `alloc_shared`（映射 UB）即可 |
| 同步 | 单 block 内顺序执行，无核间协作 | Developer 自动同步即可 |
| 参考实现 | `test_logsumexp.py` dev 版用 `alloc_shared`；`layer_norm.py` 用 Developer 模式 + `alloc_fragment` | Developer 模式已验证可行 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射到 UB）+ `T.alloc_fragment`（计算 fragment） |
| 计算方式 | v 前缀 API（vsub, vexp, vln, vadd）+ reduce_max/reduce_sum + T.Parallel 标量运算 |
| 同步 | Developer 自动同步（无手动 sync_block_set/wait） |
| Kernel 启动 | `T.Kernel(grid, is_npu=True)` |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_f32 = cast_to_fp32(x)` | 输入提升到 fp32 以保证归约精度 |
| 2 | `m[i] = max_j(x_f32[i,j])` | 每行最大值 |
| 3 | `t[i,j] = x_f32[i,j] - m[i]` | 平移（数值稳定），broadcast 减法 |
| 4 | `t[i,j] = exp(t[i,j])` | 指数 |
| 5 | `s[i] = sum_j(t[i,j])` | 每行求和 |
| 6 | `s[i] = ln(s[i])` | 对数 |
| 7 | `y[i] = m[i] + s[i]` | 加回最大值 |
| 8 | `y = cast_to_dtype(y)` | 降回原始 dtype |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 文档/示例来源 |
|------|----------|-------------|------|------|--------------|
| Load | `x → UB` | `T.copy(x[...], x_ub)` | src=GM slice, dst=UB buffer | Developer | docs/Tilelang.language/内存操作/T.copy.md |
| Copy UB→FRAG | `x_ub → x_local` | `T.copy(x_ub, x_local)` | src=UB, dst=fragment | Developer | examples/norm/layer_norm.py L37-38 |
| Cast to fp32 | `x_local → x_f32` | `T.Parallel` + `T.cast(x_local[i,j], "float32")` | 标量 cast in Parallel | Developer | examples/norm/layer_norm.py L41-42 |
| reduce_max | `max_j(x_f32)` | `T.reduce_max(x_f32, row_max, dim=1)` | buf=fragment, out=fragment, dim=1 | Developer | docs/Tilelang.language/规约操作/T.reduce_max.md |
| Sub (broadcast) | `x_f32 - row_max` | `T.vsub(x_f32, row_max, x_f32)` | src0=(bm,N) fp32, src1=(bm,1) fp32, dst=(bm,N) fp32, 行广播 | Developer | docs/Tilelang.language/数学操作/T.vsub.md §2.2.2 |
| Exp | `exp(t)` | `T.vexp(x_f32, x_f32)` | src=dst=fragment, in-place | Developer | docs/Tilelang.language/数学操作/T.vexp.md |
| reduce_sum | `sum_j(t)` | `T.reduce_sum(x_f32, row_sum, dim=1)` | buf=fragment, out=fragment, dim=1 | Developer | docs/Tilelang.language/规约操作/T.reduce_sum.md |
| Log | `ln(s)` | `T.vln(row_sum, row_sum)` | src=dst=fragment, in-place | Developer | docs/Tilelang.language/数学操作/T.vLn.md（文件名为 doc bug，实际导出名 T.vln） |
| Add | `m + ln(s)` | `T.vadd(row_max, row_sum, row_sum)` | src0=src1=dst=fragment (bm,1), in-place | Developer | docs/Tilelang.language/数学操作/T.vadd.md |
| Cast to dtype | `fp32 → dtype` | `T.Parallel` + `T.cast(row_sum[i,0], dtype)` | 标量 cast in Parallel | Developer | examples/norm/layer_norm.py L64-69 |
| Store | `UB → GM` | `T.copy(out_ub[...], y[...])` | src=UB slice, dst=GM slice | Developer | docs/Tilelang.language/内存操作/T.copy.md |

### 3.3 计算伪代码

```python
# @tilelang.jit 装饰的函数：签名去掉 threads (K9)，添加 target="npuir"
@tilelang.jit(out_idx=[1], target="npuir")
def _func(block_m):
    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],    # 参数名/顺序不变 (迁移规则 3)
        y: T.Tensor[(M,), dtype],      # 参数名/顺序不变 (迁移规则 3)
    ):
        with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
            # --- Buffer 分配 ---
            x_ub = T.alloc_shared((block_m, N), dtype)           # UB: 输入搬运
            x_local = T.alloc_fragment((block_m, N), dtype)      # Fragment: 输入副本
            x_f32 = T.alloc_fragment((block_m, N), "float32")    # Fragment: fp32 工作区
            row_max = T.alloc_fragment((block_m, 1), "float32")  # Fragment: 行最大值
            row_sum = T.alloc_fragment((block_m, 1), "float32")  # Fragment: 行求和
            out_ub = T.alloc_shared((block_m,), dtype)           # UB: 输出搬运

            # --- 行尾处理 ---
            real_m = T.min(block_m, M - pid_m * block_m)

            # --- Step 1: 数据搬入 GM → UB → Fragment ---
            # 输入侧：src/dst 同时切片，shape 一致 (real_m, N)，消除越界读取
            T.copy(x[pid_m * block_m : pid_m * block_m + real_m, 0:N], x_ub[0:real_m, 0:N])
            T.copy(x_ub, x_local)

            # --- Step 2: Cast to fp32 ---
            for i, j in T.Parallel(block_m, N):
                x_f32[i, j] = T.cast(x_local[i, j], "float32")

            # --- Step 3: reduce_max ---
            T.reduce_max(x_f32, row_max, dim=1)

            # --- Step 4: x - max (broadcast) ---
            T.vsub(x_f32, row_max, x_f32)   # (bm,N) - (bm,1) → (bm,N)

            # --- Step 5: exp ---
            T.vexp(x_f32, x_f32)             # in-place

            # --- Step 6: reduce_sum ---
            T.reduce_sum(x_f32, row_sum, dim=1)

            # --- Step 7: ln(sum) ---
            T.vln(row_sum, row_sum)          # in-place

            # --- Step 8: max + ln(sum) ---
            T.vadd(row_max, row_sum, row_sum) # in-place

            # --- Step 9: Cast back to dtype ---
            for i in T.Parallel(block_m):
                out_ub[i] = T.cast(row_sum[i, 0], dtype)

            # --- Step 10: 数据搬出 UB → GM ---
            # 输出侧：显式截断 src 为 (real_m,)，与 dst 切片 shape 一致
            T.copy(out_ub[0:real_m], y[pid_m * block_m : pid_m * block_m + real_m])

    return main
```

### 3.4 API 可行性确认

| API | 文档路径 | 验证状态 | 备注 |
|-----|---------|----------|------|
| `T.alloc_shared` | docs/Tilelang.language/内存操作/T.alloc_shared.md | ✅ 已验证 | Developer 模式映射到 UB；支持 fp16/fp32（bf16 ×，需先 vcast 到 fp32） |
| `T.alloc_fragment` | docs/Tilelang.language/编译器提示操作/T.Parallel.md L62 | ✅ 已验证 | layer_norm.py / flash_attn_npuir_dev.py 均使用 |
| `T.copy` | docs/Tilelang.language/内存操作/T.copy.md | ✅ 已验证 | 支持 GM↔UB, UB↔Fragment；切片+T.min 尾块处理见 T.copy.md §2.4 示例 1 |
| `T.reduce_max` | docs/Tilelang.language/规约操作/T.reduce_max.md | ✅ 已验证 | fp16/fp32 ✓, bf16 ×；fragment 上可用（flash_attn L63） |
| `T.reduce_sum` | docs/Tilelang.language/规约操作/T.reduce_sum.md | ✅ 已验证 | fp16/fp32 ✓, bf16 ×；fragment 上可用（layer_norm L45） |
| `T.vsub` | docs/Tilelang.language/数学操作/T.vsub.md | ✅ 已验证 | fp16/fp32 ✓, bf16 ×；行广播 [M,N]-[M,1]→[M,N] ✓；fragment 上可用（flash_attn L65） |
| `T.vexp` | docs/Tilelang.language/数学操作/T.vexp.md | ✅ 已验证 | fp16/fp32 ✓, bf16 ×；fragment 上可用（flash_attn L69） |
| `T.vln` | docs/Tilelang.language/数学操作/T.vLn.md | ✅ 已验证 | fp16/fp32 ✓, bf16 ×；文档文件名为 T.vLn.md（doc bug），实际导出名为 T.vln；测试代码 test_logsumexp.py L57/L84 使用 T.vln |
| `T.vadd` | docs/Tilelang.language/数学操作/T.vadd.md | ✅ 已验证 | fp16/fp32 ✓, bf16 ×；fragment 上可用（flash_attn L72） |
| `T.cast` (scalar) | docs/Tilelang.language/编译器提示操作/T.Parallel.md L25 | ✅ 已验证 | T.Parallel 内标量 cast 可用（layer_norm L42, L66） |
| `T.min` | examples/flash_attention/flash_attn_npuir_dev.py L86 | ✅ 已验证 | 尾块行数计算 |
| `T.infinity` | docs/Tilelang.language/创建操作/T.infinity.md | ✅ 可用 | fp16/fp32 ✓（本设计未直接使用，保留为 fallback） |
| `T.Kernel(..., is_npu=True)` | docs/Tilelang.language/编译器提示操作/T.Kernel.md | ✅ 已验证 | NPU 一维 block；返回 (cid, _) |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 本算子仅一维 `T.Kernel(T.ceildiv(M, block_m), is_npu=True)`，不涉及 |
| 部分 GPU API 不可用 | Yes | 见 §3.5.2 差异表，所有 GPU 专用 API 已替换为 NPU 等价 |
| GEMM 要求 M,N 为 block 整数倍 | No | 无 GEMM；M 非整除通过 `T.min` + 切片处理 |
| L0C 容量上限 | No | 无 Cube 计算，不涉及 L0C |

### 3.5.2 参考实现差异说明（GPU → NPU）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| `threads` 参数 | `T.Kernel(grid, threads=threads)` | NPU 无 CUDA threads 概念 | **移除 threads** (K9)；`_func(block_m)` 仅保留 block_m；`T.Kernel(grid, is_npu=True)` |
| `target` | 无显式 target（隐式 CUDA） | `target="npuir"` | `@tilelang.jit(out_idx=[1], target="npuir")` |
| `T.alloc_shared` | CUDA shared memory | NPU UB（Unified Buffer） | 同名 API，Developer 模式自动映射到 UB |
| `T.alloc_fragment` | CUDA registers/local | NPU fragment（计算缓冲） | 同名 API，语义一致 |
| `T.fill(buf, val)` | GPU 初始化 fragment | NPU 用 `T.vbrc(val, buf)` | 本设计通过 `T.reduce_max(..., clear=True)` 默认清零，无需显式 fill |
| `T.exp(x)` (标量表达式) | `T.exp(x_f32[i,j] - row_max[i])` 复合表达式 | NPU v 前缀 API 是独立的 src→dst 操作 | 拆为 `T.vsub` + `T.vexp` 两步 |
| `T.log(x)` (标量表达式) | `T.log(row_sum[i])` | `T.vln(src, dst)` | 用 `T.vln(row_sum, row_sum)` in-place |
| `T.cast(x, dtype)` (标量) | T.Parallel 内标量 cast | 同名 API，T.Parallel 内可用 | 保持 `T.cast(x_local[i,j], "float32")`（layer_norm 验证） |
| `T.if_then_else` + `T.And` | masked load（padded 路径） | **不支持循环变量条件** | **取消 padded 路径**；NPU 无 256 对齐要求，用 `T.min` + 切片处理行尾 |
| `N_padded = align_up(N, 256)` | CUDA shared memory 对齐 | NPU UB 对齐为 32 Byte（编译器管理） | **取消 N_padded**，直接用 N；单路径设计 |
| `T.reduce_max(..., clear=False)` | 不清零，在 -inf fill 上累加 | NPU `clear` 参数可用 | 用 `clear=True`（默认），单次归约无需累加 |
| bf16 支持 | GPU 原生支持 bf16 计算 | **所有 v 前缀 API 和 reduce 不支持 bf16** | **bf16 → fp32 vcast → fp32 计算 → fp32 → bf16 vcast**；fp16 同理提升到 fp32 |
| `row_max` shape | `(block_m,)` 1D | NPU reduce 要求 dst rank 与 src 一致 | 用 `(block_m, 1)` 2D，T.Parallel 提取 `[i, 0]` |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `testing/npuir/softmax_ops/test_logsumexp.py` | **极高** | 完全相同的算法（reduce_max → vsub → vexp → reduce_sum → vln → vadd）；Developer 模式 `alloc_shared` 用法；v 前缀 API 调用顺序；`T.vln` 正确用法（L57/L84） |
| `testing/npuir/softmax_ops/test_softmax.py` | 高 | softmax 的 reduce_max → vsub → vexp → reduce → vdiv 模式；Developer 模式模板 |
| `testing/npuir/softmax_ops/test_log_softmax.py` | 高 | log_softmax 的 vsub + vexp + vln + vsub 模式；`T.vln` 别名用法 |
| `examples/norm/layer_norm.py` | 高 | block_m 分块 + `alloc_shared` + `alloc_fragment` + `T.Parallel` + `T.cast` fp32 提升 + `T.reduce_sum` fragment 模式 |
| `examples/flash_attention/flash_attn_npuir_dev.py` | 中高 | Developer 模式 `alloc_shared` + `alloc_fragment` 混用；`T.reduce_max`/`T.reduce_sum` on fragment；`T.vsub` broadcast；`T.vexp`/`T.vadd` in-place；`T.min` 尾块处理 |
| `examples/TileOPs/tileops/kernels/reduction/_logsumexp_kernel_single.py` | 源 | GPU 原始实现（迁移基准） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` | 上下文 | Op 层 wrapper（dispatcher + custom_op + Kernel 类）；K9 threads 移除注释 |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | 上下文 | `align_up` / `DEFAULT_ALIGNMENT` / `compute_tile_n` 定义 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 输入；M 动态, N 静态(per-kernel build) |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(M,)` | same_as(x) | 每行 logsumexp 结果；由 `out_idx=[1]` 自动分配 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_m, N)` | dtype | UB (alloc_shared) | GM→UB 输入搬运 |
| `x_local` | `(block_m, N)` | dtype | Fragment (alloc_fragment) | 输入副本，供 T.Parallel 读取 |
| `x_f32` | `(block_m, N)` | float32 | Fragment (alloc_fragment) | fp32 工作区，复用于 sub/exp |
| `row_max` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | 每行最大值 |
| `row_sum` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | 每行指数和 → ln → max+ln |
| `out_ub` | `(block_m,)` | dtype | UB (alloc_shared) | UB→GM 输出搬运 |

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L0 参与）：

GM[x] --T.copy--> UB[x_ub] --T.copy--> FRAG[x_local]
  --T.Parallel+T.cast--> FRAG[x_f32]
  --T.reduce_max--> FRAG[row_max]
  --T.vsub(broadcast)--> FRAG[x_f32]  (x - max)
  --T.vexp(in-place)--> FRAG[x_f32]   (exp(x - max))
  --T.reduce_sum--> FRAG[row_sum]
  --T.vln(in-place)--> FRAG[row_sum]  (ln(sum))
  --T.vadd--> FRAG[row_sum]           (max + ln(sum))
  --T.Parallel+T.cast--> UB[out_ub]   (fp32 → dtype)
  --T.copy--> GM[y]
```

**关键约束**：所有计算在 fragment (fp32) 上完成；GM 不能直达 fragment，必须经 UB 中转；UB 不能直达 L0（本算子无 L0）。

### 4.5 UB 内存预算

以 Ascend A2/A3 UB = 192KB (196608 bytes) 为基准。UB 上仅驻留 `x_ub` 和 `out_ub`（fragment 由编译器管理，生命周期可重叠）：

| Buffer | Shape | dtype | 大小公式 | N=4096 fp16 | N=4096 fp32 | N=32768 fp16 | N=16384 fp16 |
|--------|-------|-------|---------|-------------|-------------|--------------|--------------|
| `x_ub` | (bm, N) | dtype | bm×N×elem_bytes | bm×8KB | bm×16KB | bm×64KB | bm×32KB |
| `out_ub` | (bm,) | dtype | bm×elem_bytes | bm×2B | bm×4B | bm×2B | bm×2B |
| **合计** | | | | ≈bm×8KB | ≈bm×16KB | ≈bm×64KB | ≈bm×32KB |

**推荐 block_m（保守，确保 UB 不溢出）**：

| N 范围 | dtype | 推荐 block_m | UB 占用 | 说明 |
|--------|-------|-------------|---------|------|
| N ≤ 4096 | fp16/bf16 | 8-16 | 64-128KB | 富余充足 |
| N ≤ 4096 | fp32 | 4-8 | 64-128KB | fp32 占用翻倍 |
| 4096 < N ≤ 16384 | fp16/bf16 | 4 | 128KB | 留 64KB 余量 |
| 16384 < N ≤ 32768 | fp16/bf16 | 1-2 | 64-128KB | 临界，block_m=1 最安全 |
| N > 32768 | any | **不适用** | 超限 | 需 tiled 变体（不在本次范围） |

> **注意**：fragment buffer（x_local, x_f32, row_max, row_sum）是否占用 UB 取决于编译器实现。layer_norm.py 在 block_m=64, N_padded=4096 下正常工作（fragment 总量远超 192KB），说明编译器对 fragment 做了生命周期管理或映射到其他存储。如编译时遇到 UB 溢出，首要措施是减小 block_m。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 说明 |
|--------|----------|-----------|------|
| M | `T.ceildiv(M, block_m)` 在 T.Kernel grid 中 | 1 ~ 任意 | M 作为 prim_func 的编译时参数；不同 M 值需重新 JIT 编译 |
| N | 静态（per-kernel build） | 固定 | N 在 kernel build 时固定为常量 |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],           # y 是第 1 个输出（index 1），自动分配
    target="npuir",        # NPU 编译目标
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅包含 reduce + element-wise 运算（reduce_max, vsub, vexp, reduce_sum, vln, vadd），无 matmul，无 Cube 参与。决策树判定为"含归约 → Developer 模式 → T.alloc_shared → UB"。

### 5.2 Block 划分

```python
# 仅沿 M 维分块；N 维不分块（single-tile path）
block_m = config_block_m   # 由 op 层根据 N 和 dtype 选择（见 §4.5 推荐表）
grid = T.ceildiv(M, block_m)
# T.Kernel(grid, is_npu=True) as (pid_m, _)
```

**选择理由**：
- N 维不分块：single-tile 变体假设 N 整体放入 UB，一次 reduce_max 和 reduce_sum 即可完成。
- M 维分块：每个 block 处理 block_m 行，行间独立（reduce 沿 dim=1），无跨 block 依赖。
- block_m 取值由 UB 容量约束决定（见 §4.5）。

### 5.3 约束分析

- **对齐约束**: NPU UB 对齐 32 Byte。`T.alloc_shared` 由编译器管理起始地址对齐，无需手动 padding。N 无需 align_up 到 256（GPU 约束已消除）。
- **UB 容量**: `x_ub` (block_m × N × elem_bytes) + `out_ub` (block_m × elem_bytes) ≤ 192KB。推荐 block_m 见 §4.5。
- **L0 容量**: 不适用（无 Cube 计算）。
- **分形限制**: 不适用（无 GEMM）。
- **MAX_SINGLE_TILE_COLS (32512)**: GPU 的列数上限是 LLVM 向量化限制。NPU 无此限制。但 N > 32768 时 UB 容量不足，仍需 tiled 变体。

### 5.4 非整除处理策略

**M % block_m != 0（行尾处理）**：

```python
real_m = T.min(block_m, M - pid_m * block_m)
# 输入搬运：src/dst 同时切片，shape 一致 (real_m, N)，消除越界读取
T.copy(x[pid_m * block_m : pid_m * block_m + real_m, 0:N], x_ub[0:real_m, 0:N])
# 注：仅搬运 real_m 行有效数据到 x_ub[0:real_m]；x_ub 剩余行未初始化但不参与输出

# 输出搬运：显式截断 src 为 (real_m,)，与 dst 切片 shape 一致
T.copy(out_ub[0:real_m], y[pid_m * block_m : pid_m * block_m + real_m])
```

**策略说明**：
- 输入侧：用 `T.min` 计算有效行数 `real_m`，对 src 和 dst 同时切片 `x[pid_m*block_m : pid_m*block_m + real_m, 0:N]` → `x_ub[0:real_m, 0:N]`，使 src/dst shape 一致（均为 `(real_m, N)`），消除越界读取。此模式与 T.copy.md §2.4 示例 1 一致。`x_ub` 剩余行（real_m ~ block_m）未初始化，后续 `T.copy(x_ub, x_local)` 搬运完整 block_m 行会产生垃圾数据，但这些行的 reduce 结果在输出阶段被 `out_ub[0:real_m]` 截断丢弃。
- 输出侧：显式截断 src 为 `out_ub[0:real_m]`，使 src shape `(real_m,)` 与 dst 切片 shape `(real_m,)` 一致，符合 T.copy.md §2.2.2 "输入与输出 shape 要一致"要求。此模式与 `flash_attn_npuir_dev.py` L86-87 输出侧用 dst 切片限制写入范围的模式一致。
- **不使用 host 侧 padding**：遵守 ascend-constraints.md §4 "host 侧禁止改动输入 NPU 张量内的真实内容"。

**N 不对齐处理**：
- NPU 不要求 256 元素对齐（GPU 约束已消除）。
- 无需 `N_padded = align_up(N, 256)`，直接使用 N。
- 无需 padded/masked load 路径，无需 `T.if_then_else` / `T.And`。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向 | block 级并行 | `T.Kernel(T.ceildiv(M, block_m), is_npu=True)` | 每个 block 处理 block_m 行，行间无依赖 |
| 元素级 cast (输入) | 向量化并行 | `T.Parallel(block_m, N)` | 逐元素 fp32 cast，无数据依赖 |
| 元素级 cast (输出) | 向量化并行 | `T.Parallel(block_m)` | 逐元素 dtype cast，从 (bm,1) 提取到 (bm,) |
| reduce_max | 硬件归约 | `T.reduce_max(x_f32, row_max, dim=1)` | 硬件加速逐行归约 |
| vsub/vexp/vln/vadd | 向量指令 | `T.vsub / T.vexp / T.vln / T.vadd` | 硬件向量指令，全 buffer 操作 |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
    real_m = T.min(block_m, M - pid_m * block_m)

    # 顺序执行（单 block 内无循环）
    T.copy(x[pid_m * block_m : pid_m * block_m + real_m, 0:N], x_ub[0:real_m, 0:N])  # GM → UB（尾块切片）
    T.copy(x_ub, x_local)                       # UB → Fragment

    for i, j in T.Parallel(block_m, N):         # 并行 cast
        x_f32[i, j] = T.cast(x_local[i, j], "float32")

    T.reduce_max(x_f32, row_max, dim=1)         # 硬件归约
    T.vsub(x_f32, row_max, x_f32)               # 向量减（广播）
    T.vexp(x_f32, x_f32)                        # 向量 exp
    T.reduce_sum(x_f32, row_sum, dim=1)         # 硬件归约
    T.vln(row_sum, row_sum)                     # 向量 log
    T.vadd(row_max, row_sum, row_sum)           # 向量加

    for i in T.Parallel(block_m):               # 并行 cast + 提取
        out_ub[i] = T.cast(row_sum[i, 0], dtype)

    T.copy(out_ub[0:real_m], y[pid_m * block_m : pid_m * block_m + real_m])  # UB → GM（src 截断）
```

### 6.3 流水线优化

**不使用 T.Pipelined**。理由：
- 本算子是 single-tile 路径，每个 block 仅读取一次输入（无 K 维迭代）。
- 无 GM→UB 的多次搬运可重叠。
- 计算步骤之间存在严格数据依赖（reduce_max → sub → exp → reduce_sum → log → add），无法并行化。

### 6.4 尾块处理

见 §5.4。最后一个 block 用 `T.min(block_m, M - pid_m * block_m)` 计算有效行数 `real_m`，输入侧 src/dst 同时切片消除越界读取，输出侧显式截断 src 使 src/dst shape 一致。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步指令，无需手动 `T.sync_block_set` / `T.sync_block_wait`。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| T.copy(GM→UB) 后 | 自动 | 编译器在 MTE2→V 拓扑自动插入 pipe_barrier |
| T.Parallel cast 后 | 自动 | T.Parallel 隐式同步 |
| reduce_max 后 | 自动 | reduce 是同步操作 |
| vsub/vexp/vln/vadd 间 | 自动 | 向量指令顺序执行 |
| T.copy(UB→GM) 前 | 自动 | 编译器在 V→MTE3 拓扑自动插入 pipe_barrier |

### 7.3 pass_configs 配置

无特殊 pass_configs。Developer 模式使用默认编译流水线。

---

## 8. 验证方案

### 8.1 Golden 函数

```python
import torch

def golden_logsumexp_kernel_single(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现。

    输入: x of shape (M, N), dtype ∈ {float16, bfloat16, float32}
    输出: y of shape (M,), dtype = same_as(x)

    对应 torch.logsumexp(x, dim=-1)（无 keepdim）。
    """
    # 在 fp32 上计算以保证精度，最后转回原 dtype
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)
```

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_softmax.py` `_get_tolerances()`：

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 高精度 |
| float16 | 1e-3 | 1e-3 | fp16 容忍度 |
| bfloat16 | 1.6e-2 | 1.6e-2 | bf16 低精度（因 vcast fp32↔bf16 转换损失） |

### 8.3 L0 门槛测试计划

L0 测试聚焦"编译通过 + 基本精度正确"，使用代表性 shape：

| 测试编号 | Shape (M, N) | dtype | block_m | 验证目标 | 容忍度 |
|---------|-------------|-------|---------|---------|--------|
| L0-1 | (32, 256) | float16 | 8 | 基本功能：小 shape fp16 | atol=1e-3 |
| L0-2 | (32, 256) | float32 | 8 | fp32 无 cast 损失 | atol=1e-5 |
| L0-3 | (32, 256) | bfloat16 | 8 | bf16 → fp32 → bf16 路径 | atol=1.6e-2 |
| L0-4 | (33, 256) | float16 | 8 | **行尾非整除**（33 % 8 = 1） | atol=1e-3 |
| L0-5 | (32, 300) | float16 | 4 | **N 不对齐**（300 非 256 倍数） | atol=1e-3 |
| L0-6 | (4, 4096) | float16 | 4 | manifest 代表 shape（attn-weights-4k） | atol=1e-3 |
| L0-7 | (4, 4096) | bfloat16 | 4 | manifest bf16 shape | atol=1.6e-2 |
| L0-8 | (1024, 4096) | float16 | 8 | manifest M=1024 大 M shape | atol=1e-3 |
| L0-9 | (1, 256) | float16 | 1 | 极小 M（单行） | atol=1e-3 |
| L0-10 | (256, 32) | float16 | 4 | dim=0 归约（op 层 reshape 后 M=32, N=256） | atol=1e-3 |

**通过条件**：全部 10 项通过 atol/rtol 检查。

### 8.4 完整分层测试套件（交由 tilelang-op-develop 阶段）

- **L1（功能覆盖）**：1D 输入、多维 dim、keepdim=True/False、全部 dtype×shape 组合
- **L2（边界）**：M=1, N=1, M=block_m 恰好整除, M=block_m+1, N=1
- **Boundary**：极值输入（全 -inf, 全 +inf, 含 NaN）、随机大 shape

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **bf16 不被 v 前缀 API 支持** | vexp/vsub/vadd/vln/reduce_max/reduce_sum 均 × bf16 | 全程 fp32 计算：vcast bf16→fp32 → 计算 → vcast fp32→bf16 |
| **fp16 reduce 精度** | 大 N 时 fp16 reduce_max/reduce_sum 可能精度不足 | 全程 fp32 accumulation（即使输入 fp16 也先 cast 到 fp32） |
| **T.if_then_else 不支持循环变量条件** | GPU 的 masked load 路径无法移植 | 取消 padded 路径；NPU 无 256 对齐要求；用 T.min+切片处理行尾 |
| **UB 容量 192KB** | N > 32768 (fp16) 或 N > 16384 (fp32) 时 x_ub 放不下 | single-tile 仅适用 N ≤ 32768 (fp16, block_m=1)；更大 N 需 tiled 变体 |
| **fragment 内存管理不确定** | fragment 是否占用 UB 取决于编译器 | 如编译报 UB 溢出，优先减小 block_m；layer_norm 在 block_m=64, N=4096 下正常 |
| **reduce 输出 shape** | NPU reduce 要求 src/dst rank 一致 | 用 (block_m, 1) 而非 (block_m,)；T.Parallel 提取 [i,0] |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_m 过大或 N 过大 | 编译失败 / segfault | 减小 block_m；参考 §4.5 推荐表 |
| bf16 直接传入 v 前缀 API | 未做 vcast 到 fp32 | 编译错误（dtype 不支持） | 确保先 `T.cast(..., "float32")` 或 `T.vcast(..., round_mode="rint")` |
| T.Parallel 内使用 GM tensor | 直接在 T.Parallel 中读写 x/y | 编译错误 | 仅操作 UB/fragment buffer；GM 读写通过 T.copy |
| vsub broadcast shape 不匹配 | row_max 用 (block_m,) 而非 (block_m,1) | 编译错误 / 结果错误 | row_max/row_sum 用 (block_m, 1) 2D shape |
| 输出越界写入 | 未用 real_m 切片限制输出 | y 缓冲区溢出 | `T.copy(out_ub[0:real_m], y[pid_m*block_m : pid_m*block_m + real_m])` |
| `T.vln` 大小写误写为 `T.vLn` | 文档 T.vLn.md 用 `T.vLn`，代码导出为 `T.vln` | 编译错误（AttributeError） | 优先用 `T.vln`（代码导出标准名）；`T.vLn` 不存在，文档 T.vLn.md 中的 `T.vLn` 为文档错误 |
| 输入侧 T.copy 尾块越界读取 | 未对 src/dst 同时切片，直接 `T.copy(x[pid_m*block_m, 0], x_ub)` | 最后一个 block 读取越界行未定义数据 | 用 `T.min` + src/dst 同时切片：`T.copy(x[off:off+real_m, 0:N], x_ub[0:real_m, 0:N])` |
| 输出侧 T.copy src/dst shape 不匹配 | src `out_ub` (block_m,) 与 dst 切片 (real_m,) shape 不一致 | 编译错误或依赖未文档化截断 | 显式截断 src：`T.copy(out_ub[0:real_m], y[off:off+real_m])` |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| **N > 32768 (fp16)** | 超出 single-tile UB 容量。op 层应路由到 `_logsumexp_kernel_tiled`（不在本次迁移范围）。本 kernel 不应被调用。 |
| **N = 102400 (manifest lm-head-logits)** | 同上，需 tiled 变体。本 kernel 的适用边界为 N ≤ 32768 (fp16, block_m=1)。 |
| **M = 1** | 单行输入。block_m=1, grid=1。正常工作。 |
| **全 -inf 输入** | reduce_max = -inf, exp(-inf - (-inf)) = exp(NaN) = NaN。此为数学定义问题，与 GPU 行为一致。torch.logsumexp 对全 -inf 输入返回 -inf。如需匹配，op 层可做特殊处理。 |
| **dim=0 归约** | op 层将输入转置为 (N, M) 后调用本 kernel（M/N 互换），再转回。本 kernel 始终沿 dim=1 归约。 |
| **多维 dim=[0,2]** | op 层 reshape 为 2D：非规约维乘积 → M，规约维乘积 → N。例如 (4,128,4096,dim=[0,2]) → M=128, N=16384。 |

### 9.4 适用边界总结

| 条件 | 是否适用 single-tile | 说明 |
|------|---------------------|------|
| N ≤ 4096, 任意 dtype | ✅ 适用 | UB 充足，block_m 可取 8-16 |
| 4096 < N ≤ 16384, fp16/bf16 | ✅ 适用 | block_m ≤ 4 |
| 16384 < N ≤ 32768, fp16/bf16 | ⚠️ 临界 | block_m=1-2，需验证编译 |
| N = 32768, bf16 | ⚠️ 临界 | x_ub (1×32768×2=64KB) + x_f32 (1×32768×4=128KB) 可能超 UB |
| N > 32768, 任意 dtype | ❌ 不适用 | 需 tiled 变体 |
| N = 102400 | ❌ 不适用 | 需 tiled 变体 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/TileOPs/_logsumexp_kernel_single/
├── _logsumexp_kernel_single.py  # 算子实现 + golden + 简单测试
├── DESIGN.md                    # 本设计文档
└── .stage_state.json            # 编排层状态（不由本阶段维护）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `_logsumexp_kernel_single.py` | ⬜ 待实现 | 算子实现（kernel + golden + test） |

### 10.3 命名规范

- 项目目录名: `TileOPs`
- 算子目录名: `_logsumexp_kernel_single`
- 实现文件: `_logsumexp_kernel_single.py`
- Golden 函数: `golden_logsumexp_kernel_single`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）— 本文档
2. ⬜ Golden 函数 — `golden_logsumexp_kernel_single(x)` 基于 `torch.logsumexp`
3. ⬜ 算子实现 — `_logsumexp_kernel_single.py`：`@tilelang.jit(out_idx=[1], target="npuir") def _func(block_m)` + `@T.prim_func main(x, y)` + L0 测试
4. ⬜ 精度比对 — 与 Golden 函数对比，通过 L0 门槛测试全部 10 项

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 规则 1：算子名 = `_logsumexp_kernel_single` | ✅ | 不裁剪不变换 |
| 规则 2：`@tilelang.jit` 函数声明不变 | ✅ | `_func(block_m)` 保留（仅移除 K9 threads，添加 target="npuir"） |
| 规则 3：`@T.prim_func` 参数名/顺序不变 | ✅ | `main(x: T.Tensor[(M,N), dtype], y: T.Tensor[(M,), dtype])` 完全保持 |
| 规则 4：从源码推断输入规格 | ✅ | 从 GPU 源码 + manifest 推断，未询问用户 |
| K9：threads 移除 | ✅ | `_func(block_m)` 无 threads 参数 |
| Host 侧不改输入 | ✅ | 全部计算在 kernel 内；无 host 侧 padding |

---

## 11. 修订日志

### revision_index: 1（2026-08-13）

**修订来源**：Stage 2 REVIEW.md 检视不通过（1 阻塞 + 2 建议）。

**相对上一版（design_v0）的关键调整**：

| 序号 | 问题类型 | 问题描述 | 修复内容 | 影响章节 |
|------|----------|----------|----------|----------|
| 1 | 阻塞 | 伪代码使用不存在的 `T.vLn` API（实际导出名是 `T.vln`，全小写 l） | 全文所有 `T.vLn` → `T.vln`；§3.2/§3.4/§9.2 注明文档文件名 `T.vLn.md` 为 doc bug，实际导出名 `T.vln`；§9.2 错误表新增"大小写误写"条目 | §1.5 §2.3 §3.2 §3.3 §3.4 §3.5.2 §4.4 §6.1 §6.2 §7.2 §9.1 §9.2 |
| 2 | 建议 | 输入侧 `T.copy(x[pid_m*block_m, 0], x_ub)` 尾块越界读取 | 改为 src/dst 同时切片：`T.copy(x[off:off+real_m, 0:N], x_ub[0:real_m, 0:N])`，消除越界读取 | §3.3 §5.4 §6.2 |
| 3 | 建议 | 输出侧 `T.copy(out_ub, y[...])` src/dst shape 不匹配 | 显式截断 src：`T.copy(out_ub[0:real_m], y[...])`，使 src/dst shape 一致 | §3.3 §5.4 §6.2 |

**为何不会再犯同一错误**：

1. **API 名称验证固化**：本次修订通过 `tilelang/language/__init__.py` L104-105 直接验证导出名（`npuir_ln as vln`），并交叉验证 `testing/npuir/softmax_ops/test_logsumexp.py` L57/L84 的实际调用。§3.4 API 表已注明"文档文件名为 doc bug"，后续设计如再遇到文档与代码不一致，将以 `tilelang/language/__init__.py` 导出表为唯一真相源。
2. **T.copy shape 一致性固化**：§9.2 错误表新增"输入侧尾块越界读取"和"输出侧 src/dst shape 不匹配"两个条目，明确标注正确模式。后续伪代码中所有 `T.copy` 调用均需遵循 T.copy.md §2.2.2"输入与输出 shape 要一致"和 §2.4 示例 1 的 src/dst 同时切片模式。

**保持不变的设计决策**（已通过 Stage 2 检视）：
- 编程模式：Developer（自动同步，`alloc_shared` + `alloc_fragment`）
- 全程 fp32 计算（bf16/fp16 → cast fp32 → 计算 → cast 回原 dtype）
- 取消 padded 路径（NPU 无 256 对齐要求，无 `T.if_then_else` masked load）
- Tiling 策略：仅沿 M 维分块，N 维不分块（single-tile path）
- 循环结构：T.Kernel block 级并行 + T.Parallel 元素级 cast + v 前缀向量指令 + reduce 硬件归约
- 验证方案：Golden = `torch.logsumexp`，L0 门槛测试 10 项
- 技术约束：无三维 Kernel、无 GEMM、无 L0C、bf16 不被 v 前缀 API 支持
- 适用边界：N ≤ 32768 (fp16, block_m=1)
