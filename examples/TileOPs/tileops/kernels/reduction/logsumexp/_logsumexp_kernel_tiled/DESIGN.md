# _logsumexp_kernel_tiled 算子设计文档

## 0. 原始计算逻辑分析（迁移类任务）

### 0.1 功能概述

GPU 版 `_logsumexp_kernel_tiled` 实现多 tile logsumexp 前向核：对 `(M, N)` 输入张量沿最后一维（N 维）做 logsumexp 归约，输出 `(M,)` 向量。与 single-tile 变体不同，tiled 变体假设 N **不能**整体放入 UB（如 N=32768、N=102400），需沿 N 维分块，使用**在线 softmax 递推（online softmax recurrence）**跨 tile 单遍计算 running_max 和 rescaled running_sum，最终输出 `y[i] = running_max[i] + ln(running_sum[i])`。

### 0.2 输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `x` | 输入 | `(M, N)` | float16 / bfloat16 / float32 | 2D 输入张量；M=非规约维乘积, N=规约维（N 较大需分块） |
| `y` | 输出 | `(M,)` | same_as(x) | 每行 logsumexp 结果 |

### 0.3 详细解读

GPU 源文件：`examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`（函数 `_logsumexp_kernel_tiled`，第 81–169 行）

GPU 核心逻辑：

1. 工厂函数 `_logsumexp_kernel_tiled(M, N, dtype, tile_n)` 返回 `_func(block_m, threads)`，后者返回 `main` prim_func。
2. `N_padded = align_up(N, DEFAULT_ALIGNMENT=256)`，`num_tiles = ceildiv(N_padded, tile_n)`，`total_cols = num_tiles * tile_n`，`_needs_mask = total_cols > N`。
3. **在线递推主循环** `for t in T.Serial(num_tiles)`：
   - **masked load**：`_needs_mask=True` 时，非末 tile 用快速 `T.copy`，末 tile 用 `T.if_then_else` + `T.And` 填 `-inf`；`_needs_mask=False` 时全部用 `T.copy`。
   - `T.fill(tile_max, -inf)` → `T.reduce_max(tile_f32, tile_max, dim=1, clear=False)`
   - 保存 `prev_max = row_max`，更新 `row_max = T.max(row_max, tile_max)`（标量 `T.max` 在 `T.Parallel(block_m)` 中逐行计算）
   - `tile_f32 = T.exp(tile_f32 - row_max)`（用**新的** row_max 平移，标量 `T.exp` 在 `T.Parallel(block_m, tile_n)` 中逐元素计算）
   - `T.reduce_sum(tile_f32, tile_sum, dim=1)`
   - `row_sum = row_sum * T.exp(prev_max - row_max) + tile_sum`（标量运算在 `T.Parallel(block_m)` 中逐行计算）
4. **epilogue**：`out_local[i] = row_max[i] + T.log(row_sum[i])`（标量 `T.log`）
5. `T.copy(out_local, y[pid_m * block_m])` — 输出搬运。
6. `threads=threads` — GPU CUDA 线程参数。

### 0.4 标杆实现

- **PyTorch 参考**：`torch.logsumexp(x.float(), dim=-1).to(x.dtype)`
- **测试文件**：`examples/TileOPs/tests/ops/test_softmax.py`（LogSumExp 子集）
- **输出形状**：`(M,)`，无 keepdim。Op 层（`LogSumExpKernel` 类）负责把任意维输入 reshape 成 2D 并处理 keepdim。
- **manifest 工作负载**（`examples/TileOPs/tileops/manifest/reduction.yaml`）：
  - `{x_shape: [32, 32, 4096], dtypes: [float16, bfloat16], label: "attn-weights-4k"}` → 2D: M=1024, N=4096
  - `{x_shape: [32, 32, 32768], dtypes: [bfloat16], label: "attn-weights-32k"}` → 2D: M=1024, N=32768
  - `{x_shape: [4, 102400], dtypes: [float16, bfloat16], label: "lm-head-logits"}` → 2D: M=4, N=102400
  - `{x_shape: [4, 128, 4096], dtypes: [float16], dim: [0, 2], label: "3d-multidim-reduce"}` → 2D: M=128, N=16384

---

## 1. 概述

### 1.1 算子名称

`_logsumexp_kernel_tiled`

### 1.2 功能描述

对 `(M, N)` 输入张量沿 N 维做数值稳定的 logsumexp 归约，输出 `(M,)` 向量。tiled 路径针对大 N（超出单 tile UB 容量）场景，使用在线 softmax 递推跨 N-tile 单遍计算。

### 1.3 数学公式

$$
y[i] = \max_j(x[i,j]) + \ln\left(\sum_j \exp\left(x[i,j] - \max_j(x[i,j])\right)\right)
$$

**在线递推形式**（跨 tile 单遍）：

维护 running_max $m^{(t)}$ 和 rescaled running_sum $s^{(t)}$，每处理一个 tile $t$：

$$
\begin{aligned}
m_{\text{tile}}^{(t)}[i] &= \max_{j \in \text{tile}_t} x[i,j] \\
m^{(t)}[i] &= \max\left(m^{(t-1)}[i],\; m_{\text{tile}}^{(t)}[i]\right) \\
s^{(t)}[i] &= s^{(t-1)}[i] \cdot \exp\left(m^{(t-1)}[i] - m^{(t)}[i]\right) + \sum_{j \in \text{tile}_t} \exp\left(x[i,j] - m^{(t)}[i]\right)
\end{aligned}
$$

最终：$y[i] = m^{(T)}[i] + \ln(s^{(T)}[i])$

参考 API：`torch.logsumexp`

### 1.4 算法描述

数值稳定的在线递推算法（per-row 独立，跨 tile 单遍）：

1. **初始化**：`row_max = -inf`，`row_sum = 0`
2. **逐 tile 循环**（`T.serial(num_tiles)`）：
   - **加载 tile**：GM → UB → Fragment → cast fp32（尾 tile 列越界用 `T.if_then_else` 填 `-inf`）
   - **tile_max**：`tile_max = reduce_max(tile_f32, dim=1)`
   - **更新 running max**：`new_max = max(row_max, tile_max)`，`correction = exp(row_max - new_max)`
   - **shift + exp + sum**：`tile_f32 = exp(tile_f32 - new_max)`，`tile_sum = reduce_sum(tile_f32, dim=1)`
   - **rescale + accumulate**：`row_sum = row_sum * correction + tile_sum`，`row_max = new_max`
3. **epilogue**：`y = row_max + ln(row_sum)`

### 1.5 数据流图

```
GM[x] --T.copy(slice)--> UB[shared_buf] --T.copy--> FRAG[tile_local]
  --T.Parallel+T.if_then_else+T.cast--> FRAG[tile_f32]          (fp32, 尾列填 -inf)
  --T.reduce_max(dim=1)--> FRAG[tile_max]                       (bm,1)
  --T.vmax(row_max, tile_max)--> FRAG[new_max]                  (running max update)
  --T.vsub(row_max, new_max)--> FRAG[correction]                (rescale factor)
  --T.vexp--> FRAG[correction]
  --T.vsub(tile_f32, new_max)--> FRAG[tile_f32]                 (shift by new_max)
  --T.vexp(in-place)--> FRAG[tile_f32]
  --T.reduce_sum(dim=1)--> FRAG[tile_sum]                       (bm,1)
  --T.vmul(row_sum, correction)--> FRAG[row_sum]                (rescale old sum)
  --T.vadd(row_sum, tile_sum)--> FRAG[row_sum]                  (accumulate)
  --T.vbrc(0,tmp1)+T.vadd(tmp1,new_max)--> FRAG[row_max]        (copy new_max → row_max)
  [循环下一个 tile]
  [循环结束后]
  --T.vln(in-place)--> FRAG[row_sum]                             (ln(sum))
  --T.vadd(row_max, row_sum)--> FRAG[row_sum]                    (max + ln(sum))
  --T.Parallel+T.cast--> UB[out_ub]                              (fp32 → dtype)
  --T.copy(slice)--> GM[y]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（归约 + 逐元素 + 在线递推），无 matmul | 不需要 Cube/L0 |
| 归约 | reduce_max + reduce_sum（per-tile） | Developer 模式下可用 |
| 内存层级 | 仅 GM ↔ UB，无 L1/L0 参与 | Developer 的 `alloc_shared`（映射 UB）即可 |
| 在线递推 | 跨 tile 维护 running_max/sum，需向量 max/rescale | `T.vmax`/`T.vmul` 在 fragment 上可用（flash_attn 验证） |
| 同步 | 单 block 内顺序执行 + tile 循环，无核间协作 | Developer 自动同步即可 |
| 参考实现 | `flash_attn_npuir_dev.py` 用 Developer + `alloc_shared` + `alloc_fragment` + `T.vmax`/`T.vmul` 在线递推 | Developer 模式已验证可行 |
| single-tile 迁移 | 同一算法族，single-tile 已通过 Stage 2 检视 | 沿用 Developer 模式 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射到 UB）+ `T.alloc_fragment`（计算 fragment） |
| 计算方式 | v 前缀 API（vmax, vmul, vsub, vexp, vln, vadd, vbrc）+ reduce_max/reduce_sum + T.Parallel 标量运算（cast + if_then_else） |
| 同步 | Developer 自动同步（无手动 sync_block_set/wait） |
| Kernel 启动 | `T.Kernel(grid, is_npu=True)` |

---

## 3. API 映射设计

### 3.1 公式拆解（在线递推，per-tile）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `tile_f32 = cast_to_fp32(x_tile)` | 输入提升到 fp32；尾 tile 列越界填 -inf |
| 2 | `tile_max = max_j(tile_f32)` | 当前 tile 每行最大值 |
| 3 | `new_max = max(row_max, tile_max)` | 更新 running max |
| 4 | `correction = exp(row_max - new_max)` | 旧 sum 的 rescale 因子 |
| 5 | `tile_f32 = exp(tile_f32 - new_max)` | 用**新的** running max 平移后取指数 |
| 6 | `tile_sum = sum_j(tile_f32)` | 当前 tile 每行指数和 |
| 7 | `row_sum = row_sum * correction + tile_sum` | rescale 旧 sum 并累加 |
| 8 | `row_max = new_max` | 更新 row_max |
| **epilogue** | | |
| 9 | `row_sum = ln(row_sum)` | 对数 |
| 10 | `row_sum = row_max + row_sum` | 加回最大值 |
| 11 | `y = cast_to_dtype(row_sum)` | 降回原始 dtype |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 文档/示例来源 |
|------|----------|-------------|------|------|--------------|
| Load tile | `x → UB` | `T.copy(x[off_m:off_m+real_m, t*tile_n:t*tile_n+valid_n], shared_buf[0:real_m, 0:valid_n])` | src=GM slice, dst=UB slice, T.min 计算有效行列 | Developer | docs/Tilelang.language/内存操作/T.copy.md §2.4 示例 1/2 |
| Copy UB→FRAG | `shared_buf → tile_local` | `T.copy(shared_buf, tile_local)` | src=UB, dst=fragment | Developer | single-tile 实现 L85 |
| Cast+mask | `tile_local → tile_f32` | `T.Parallel(bm, tile_n)` + `T.if_then_else(t*tile_n+j < N, T.cast(...), -inf)` | 标量 cast + 条件掩码 in Parallel | Developer | testing/npuir/parallel_ops/test_loop_var_compute.py L37/L147；examples/deepseek_v4/inference/kernel.py L386 |
| reduce_max | `max_j(tile_f32)` | `T.reduce_max(tile_f32, tile_max, dim=1)` | buf=fragment, out=fragment(bm,1), dim=1, clear=True(默认) | Developer | docs/Tilelang.language/规约操作/T.reduce_max.md |
| Running max | `max(row_max, tile_max)` | `T.vmax(row_max, tile_max, new_max)` | src0=src1=dst=fragment(bm,1), fp32 | Developer | examples/flash_attention/flash_attn_npuir_dev.py L64；export: npuir_max as vmax |
| Correction sub | `row_max - new_max` | `T.vsub(row_max, new_max, correction)` | src0=(bm,1), src1=(bm,1), dst=(bm,1), fp32 | Developer | docs/Tilelang.language/数学操作/T.vsub.md；flash_attn L65 |
| Correction exp | `exp(correction)` | `T.vexp(correction, correction)` | src=dst=fragment, in-place | Developer | docs/Tilelang.language/数学操作/T.vexp.md；flash_attn L66 |
| Shift sub | `tile_f32 - new_max` | `T.vsub(tile_f32, new_max, tile_f32)` | src0=(bm,tn), src1=(bm,1), dst=(bm,tn), 行广播 | Developer | docs/Tilelang.language/数学操作/T.vsub.md §2.2.2；single-tile L95 |
| Exp | `exp(tile_f32)` | `T.vexp(tile_f32, tile_f32)` | src=dst=fragment, in-place | Developer | docs/Tilelang.language/数学操作/T.vexp.md |
| reduce_sum | `sum_j(tile_f32)` | `T.reduce_sum(tile_f32, tile_sum, dim=1)` | buf=fragment, out=fragment(bm,1), dim=1, clear=True(默认) | Developer | docs/Tilelang.language/规约操作/T.reduce_sum.md |
| Rescale mul | `row_sum * correction` | `T.vmul(row_sum, correction, row_sum)` | src0=src1=dst=fragment(bm,1), in-place | Developer | docs/Tilelang.language/数学操作/T.vmul.md；flash_attn L71 |
| Accumulate add | `row_sum + tile_sum` | `T.vadd(row_sum, tile_sum, row_sum)` | src0=src1=dst=fragment(bm,1), in-place | Developer | docs/Tilelang.language/数学操作/T.vadd.md；flash_attn L72 |
| Copy new_max→row_max | `row_max = new_max` | `T.vbrc(0, tmp1)` + `T.vadd(tmp1, new_max, row_max)` | 先零填 tmp1，再 vadd 复制 | Developer | flash_attn L76-77（fragment→fragment copy 不可用 T.copy） |
| Init row_max | `row_max = -inf` | `T.vbrc(-T.infinity("float32"), row_max)` | scalar → fragment(bm,1) | Developer | docs/Tilelang.language/shape操作/T.vbrc.md；flash_attn L52 |
| Init row_sum | `row_sum = 0` | `T.vbrc(0, row_sum)` | scalar → fragment(bm,1) | Developer | flash_attn L51 |
| Log | `ln(sum)` | `T.vln(row_sum, row_sum)` | src=dst=fragment(bm,1), in-place | Developer | docs/Tilelang.language/数学操作/T.vLn.md（doc bug，导出名 T.vln） |
| Add | `max + ln(sum)` | `T.vadd(row_max, row_sum, row_sum)` | src0=src1=dst=fragment(bm,1), in-place | Developer | docs/Tilelang.language/数学操作/T.vadd.md |
| Cast to dtype | `fp32 → dtype` | `T.Parallel(bm)` + `T.cast(row_sum[i,0], dtype)` | 标量 cast in Parallel | Developer | single-tile L110-111 |
| Store | `UB → GM` | `T.copy(out_ub[0:real_m], y[off_m:off_m+real_m])` | src=UB slice, dst=GM slice | Developer | docs/Tilelang.language/内存操作/T.copy.md；single-tile L115-118 |

### 3.3 计算伪代码

> **迁移规则 3 注意**：`@T.prim_func` 的 `T.Tensor` 参数将 dtype 作为第二位置参数（非 `dtype=` 关键字），遵循 design-template §3.3 指引，避免 false alarm。

```python
@tilelang.jit(out_idx=[1], target="npuir")       # K9: 移除 threads; 添加 target
def _func(block_m):
    @T.prim_func
    def main(
        x: T.Tensor[(M, N), dtype],               # 参数名/顺序不变 (迁移规则 3)
        y: T.Tensor[(M,), dtype],                 # 参数名/顺序不变 (迁移规则 3)
    ):
        with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):

            # --- Buffer 分配 ---
            # UB (alloc_shared) — GM 传输缓冲
            shared_buf = T.alloc_shared((block_m, tile_n), dtype)
            out_ub     = T.alloc_shared((block_m,), dtype)

            # Fragment (alloc_fragment) — 计算缓冲
            tile_local = T.alloc_fragment((block_m, tile_n), dtype)
            tile_f32   = T.alloc_fragment((block_m, tile_n), "float32")

            # 在线递推状态 (block_m, 1) fp32 — rank 与 reduce 输出一致
            row_max    = T.alloc_fragment((block_m, 1), "float32")  # running max
            row_sum    = T.alloc_fragment((block_m, 1), "float32")  # running sum
            tile_max   = T.alloc_fragment((block_m, 1), "float32")  # per-tile max
            tile_sum   = T.alloc_fragment((block_m, 1), "float32")  # per-tile sum
            new_max    = T.alloc_fragment((block_m, 1), "float32")  # max(row_max, tile_max)
            correction = T.alloc_fragment((block_m, 1), "float32")  # exp(old_max - new_max)
            tmp1       = T.alloc_fragment((block_m, 1), "float32")  # scratch (copy trick)

            # --- 标量常量 (T.Kernel 内、T.serial 外) ---
            value_min  = -T.infinity("float32")
            value_zero = 0

            # --- 行尾处理 ---
            real_m = T.min(block_m, M - pid_m * block_m)

            # --- 初始化递推状态 ---
            T.vbrc(value_min, row_max)          # row_max = -inf
            T.vbrc(value_zero, row_sum)         # row_sum = 0

            # --- 在线递推主循环 (T.serial, 非 T.Serial — 见 §3.5.2) ---
            num_tiles = T.ceildiv(N, tile_n)     # 编译时常量 (M,N,tile_n 均为工厂参数)
            for t in T.serial(num_tiles):

                # 1. 加载 tile: GM → UB (src/dst 同时切片, 消除越界读取)
                valid_n = T.min(tile_n, N - t * tile_n)
                T.copy(
                    x[pid_m * block_m : pid_m * block_m + real_m,
                      t * tile_n : t * tile_n + valid_n],
                    shared_buf[0:real_m, 0:valid_n],
                )
                T.copy(shared_buf, tile_local)   # UB → Fragment (完整 bm×tn, 尾区垃圾)

                # 2. Cast to fp32 + 尾列掩码 (-inf 填充)
                #    条件 t*tile_n+j < N 涉及 T.serial 循环变量 t — NPU 已验证可用
                for i, j in T.Parallel(block_m, tile_n):
                    tile_f32[i, j] = T.if_then_else(
                        t * tile_n + j < N,
                        T.cast(tile_local[i, j], "float32"),
                        value_min,
                    )

                # 3. Tile max
                T.reduce_max(tile_f32, tile_max, dim=1)     # clear=True (默认)

                # 4. New running max = max(row_max, tile_max)
                T.vmax(row_max, tile_max, new_max)

                # 5. Correction = exp(row_max - new_max)  [rescale factor for old sum]
                T.vsub(row_max, new_max, correction)        # correction = old_max - new_max (≤0)
                T.vexp(correction, correction)              # correction = exp(old_max - new_max)

                # 6. Shift tile by new_max, exp, sum
                T.vsub(tile_f32, new_max, tile_f32)         # (bm,tn) - (bm,1) → (bm,tn) 行广播
                T.vexp(tile_f32, tile_f32)                  # in-place
                T.reduce_sum(tile_f32, tile_sum, dim=1)     # clear=True (默认)

                # 7. Rescale old sum and accumulate
                T.vmul(row_sum, correction, row_sum)        # row_sum *= exp(old_max - new_max)
                T.vadd(row_sum, tile_sum, row_sum)          # row_sum += tile_sum

                # 8. Update row_max = new_max (fragment→fragment 不可用 T.copy, 用 vbrc+vadd)
                T.vbrc(value_zero, tmp1)                    # tmp1 = 0
                T.vadd(tmp1, new_max, row_max)              # row_max = 0 + new_max

            # --- Epilogue: y = max + ln(sum) ---
            T.vln(row_sum, row_sum)                         # ln(sum)
            T.vadd(row_max, row_sum, row_sum)               # max + ln(sum)

            # --- Cast back to dtype, extract (i,0) → (i,) ---
            for i in T.Parallel(block_m):
                out_ub[i] = T.cast(row_sum[i, 0], dtype)

            # --- Store: UB → GM (行尾截断) ---
            T.copy(out_ub[0:real_m], y[pid_m * block_m : pid_m * block_m + real_m])

    return main
```

### 3.4 API 可行性确认

> **验证方法**：通过 `tilelang/language/__init__.py` 导出表 + `python3 -c "hasattr(T, ...)"` 运行时验证 + docs 文档 + NPU 示例实际调用四重交叉确认。

| API | 导出验证 | 文档路径 | NPU 示例验证 | 备注 |
|-----|----------|---------|-------------|------|
| `T.alloc_shared` | ✅ `__init__.py` L41 | docs/Tilelang.language/内存操作/T.alloc_shared.md | test_logsumexp.py L73; single-tile L68 | Developer 模式映射 UB |
| `T.alloc_fragment` | ✅ `__init__.py` L42 | — (T.Parallel.md L62) | flash_attn_dev L34-45; single-tile L69-72 | fragment 计算缓冲 |
| `T.copy` | ✅ `__init__.py` L50 | docs/Tilelang.language/内存操作/T.copy.md | 所有 NPU 示例 | §2.4 T.min+切片模式 |
| `T.reduce_max` | ✅ `__init__.py` L53/L126 | docs/Tilelang.language/规约操作/T.reduce_max.md | flash_attn_dev L63; single-tile L92 | clear 参数支持, 默认 True |
| `T.reduce_sum` | ✅ `__init__.py` L55/L128 | docs/Tilelang.language/规约操作/T.reduce_sum.md | flash_attn_dev L70; single-tile L101 | clear 参数支持, 默认 True |
| `T.vmax` | ✅ `__init__.py` L77 (`npuir_max as vmax`) | **无独立 doc 页** | flash_attn_dev L64; test_max_pool2d L51; sparse_mla_fwd L127 | ⚠️ 无 doc 页, 但 3+ NPU 示例验证 |
| `T.vmul` | ✅ `__init__.py` L81 (`npuir_mul as vmul`) | docs/Tilelang.language/数学操作/T.vmul.md | flash_attn_dev L62/L71/L73; mish.py L115 | 行广播 [M,N]*[M,1]✓; in-place ✓ |
| `T.vsub` | ✅ `__init__.py` L75 (`npuir_sub as vsub`) | docs/Tilelang.language/数学操作/T.vsub.md | flash_attn_dev L65/L68; single-tile L95 | 行广播 [M,N]-[M,1]✓ |
| `T.vexp` | ✅ `__init__.py` L101 (`npuir_exp as vexp`) | docs/Tilelang.language/数学操作/T.vexp.md | flash_attn_dev L66/L69; single-tile L98 | in-place ✓ |
| `T.vln` | ✅ `__init__.py` L105 (`npuir_ln as vln`) | docs/Tilelang.language/数学操作/T.vLn.md (doc bug) | test_logsumexp.py L57/L84; single-tile L104 | 文档文件名 T.vLn.md 为 doc bug, 导出名 T.vln (全小写 l) |
| `T.vadd` | ✅ `__init__.py` L73 (`npuir_add as vadd`) | docs/Tilelang.language/数学操作/T.vadd.md | flash_attn_dev L72/L77; single-tile L107 | in-place ✓ |
| `T.vbrc` | ✅ `__init__.py` L117 (`npuir_brc as vbrc`) | docs/Tilelang.language/shape操作/T.vbrc.md | flash_attn_dev L50-53 | scalar→buffer 广播; 参数顺序 (value, buf) |
| `T.cast` (scalar) | ✅ builtin | T.Parallel.md L25 | layer_norm L42/L66; single-tile L89/L111 | T.Parallel 内标量 cast |
| `T.if_then_else` | ✅ builtin + `hasattr` True | — | test_loop_var_compute.py L37/L147; deepseek_v4 L386 | **条件可含循环变量 t** — NPU 已验证 |
| `T.infinity` | ✅ builtin + `hasattr` True | docs/Tilelang.language/创建操作/T.infinity.md | flash_attn_dev L49 | fp16/fp32 ✓ (bf16 ×, 本设计仅 fp32 用) |
| `T.min` | ✅ builtin + `hasattr` True | — (T.copy.md §2.4 示例) | T.copy.md 示例 L60-61; single-tile L76 | 标量 min |
| `T.serial` | ✅ builtin + `hasattr` True (`serial: True`) | T.infinity.md 示例 L53 | test_max_pool2d L47; T.infinity.md L53 | **小写 serial** — T.Serial (大写) 不存在! |
| `T.Parallel` | ✅ `__init__.py` L27 | — | 所有 NPU 示例 | 向量化并行 |
| `T.ceildiv` | ✅ builtin + `hasattr` True | — | 所有 NPU 示例 | 编译时整数除法 |
| `T.Kernel(..., is_npu=True)` | ✅ `__init__.py` L30 | — | 所有 NPU 示例 | NPU 一维 block |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 本算子仅一维 `T.Kernel(T.ceildiv(M, block_m), is_npu=True)`，不涉及 |
| 部分 GPU API 不可用 | Yes | 见 §3.5.2 差异表，所有 GPU 专用 API 已替换为 NPU 等价 |
| GEMM 要求 M,N 为 block 整数倍 | No | 无 GEMM；M 非整除通过 `T.min` + 切片处理；N 非整除通过 `T.if_then_else` 掩码处理 |
| L0C 容量上限 | No | 无 Cube 计算，不涉及 L0C |

### 3.5.2 参考实现差异说明（GPU → NPU）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 | 验证来源 |
|--------|----------------|-----------------|----------|----------|
| **`T.Serial` → `T.serial`** | `for t in T.Serial(num_tiles)` | **`T.Serial` 不存在**（`hasattr(T,'Serial')=False`） | 改为 `T.serial(num_tiles)`（小写 s） | `python3 -c hasattr` 验证；test_max_pool2d L47; T.infinity.md L53 |
| `threads` 参数 | `T.Kernel(grid, threads=threads)` | NPU 无 CUDA threads 概念 | **移除 threads** (K9); `T.Kernel(grid, is_npu=True)` | single-tile 迁移确认 |
| `target` | 无显式 target | `target="npuir"` | `@tilelang.jit(out_idx=[1], target="npuir")` | single-tile 迁移确认 |
| **`T.if_then_else` + `T.And` masked load** | 末 tile 用 `T.if_then_else(T.And(pid_m*bm+i<M, t*tn+j<N), cast(x), -inf)` 逐元素掩码加载 | **`T.if_then_else` 在 NPU 上可用**（含循环变量条件） | 保留 `T.if_then_else` 掩码方案；条件简化为 `t*tile_n+j < N`（行尾由输出截断处理，无需行条件） | **test_loop_var_compute.py L37/L147** (NPU Developer mode, `j < valid_block_n`); **deepseek_v4 L386** (`t*block+i < topk`, t 为 serial 循环变量) |
| `N_padded = align_up(N, 256)` | CUDA shared memory 256 对齐 | NPU UB 对齐 32 Byte（编译器管理） | **取消 N_padded**；直接用 N；`num_tiles = ceildiv(N, tile_n)` | single-tile 迁移确认 |
| `_needs_mask` 分支 | `_needs_mask=True` 时非末 tile 用 T.copy, 末 tile 用 if_then_else; `=False` 时全用 T.copy | NPU 不支持 T.If(t < num_tiles-1) 循环变量控制流分支 | **统一路径**：所有 tile 用 T.copy(slice) + T.if_then_else 掩码（非末 tile 条件恒真, 编译器优化） | 无需分支, 简化设计 |
| `T.fill(buf, -inf)` | GPU 初始化 fragment | NPU 用 `T.vbrc(val, buf)` | `T.vbrc(-T.infinity("float32"), row_max)` | flash_attn_dev L52; T.vbrc.md |
| `T.exp(x)` (标量) | `T.exp(tile_f32[i,j] - row_max[i])` 复合表达式 | NPU v 前缀 API 是独立 src→dst 操作 | 拆为 `T.vsub` + `T.vexp` 两步 | single-tile 迁移确认 |
| `T.max(a, b)` (标量, T.Parallel) | `row_max[i] = T.max(row_max[i], tile_max[i])` | NPU 有 `T.vmax` 向量 max | 用 `T.vmax(row_max, tile_max, new_max)` 向量操作 | flash_attn_dev L64 验证 |
| `T.log(x)` (标量) | `T.log(row_sum[i])` | `T.vln(src, dst)` | 用 `T.vln(row_sum, row_sum)` in-place | single-tile 迁移确认 |
| **递推标量运算** | `row_sum[i] = row_sum[i]*T.exp(prev_max[i]-row_max[i]) + tile_sum[i]` (T.Parallel 标量) | NPU v 前缀 API 分解 | 拆为 `vsub`→`vexp`→`vmul`→`vadd` 四步向量操作 | flash_attn_dev L65-72 验证 |
| **`prev_max = row_max` (fragment copy)** | `prev_max[i] = row_max[i]` (T.Parallel 标量赋值) | fragment→fragment 不可用 T.copy | 用 `T.vmax` 写入独立 `new_max`, 再 `vbrc(0,tmp1)+vadd(tmp1,new_max,row_max)` 更新 | flash_attn_dev L76-77 验证 |
| `T.reduce_max(..., clear=False)` | 不清零, 在 -inf fill 上累加 | NPU clear 参数可用 | 用 `clear=True`（默认）— 每 tile 独立归约, 无需累加 | T.reduce_max.md §2.1 |
| bf16 支持 | GPU 原生支持 bf16 计算 | **所有 v 前缀 API 和 reduce 不支持 bf16** | **bf16 → fp32 vcast → fp32 计算 → fp32 → bf16 vcast** | single-tile 迁移确认 |
| `row_max` shape | `(block_m,)` 1D | NPU reduce 要求 dst rank 与 src 一致 | 用 `(block_m, 1)` 2D | single-tile 迁移确认 |
| **尾 tile 处理** | `_needs_mask` 分支 + `T.If/Then/Else` 控制流 | NPU 不支持循环变量控制流分支 | **统一 T.copy(slice) + T.if_then_else 掩码**；`valid_n = T.min(tile_n, N - t*tile_n)` 切片加载 | T.copy.md §2.4; test_loop_var_compute.py |

### 3.5.3 ⭐ 关键技术发现：T.if_then_else 在 NPU 上支持循环变量条件

> **与 single-tile DESIGN 的差异**：single-tile DESIGN.md §3.5.2/§9.1 声称"T.if_then_else 不支持循环变量条件"，并取消了 padded 路径。**该结论已过时**。

**证据链**（四重验证）：

1. **`testing/npuir/parallel_ops/test_loop_var_compute.py`**（NPU Developer mode, target="npuir"）：
   - Case 1 (L37): `T.if_then_else(i * block_n + j < threshold, ...)` — i, j 为 T.Parallel 循环变量
   - Case 3 (L147): `T.if_then_else(j < valid_block_n, ...)` — j 为 T.Parallel 循环变量, valid_block_n 为编译时常量
   - Case 4 (L211): `T.if_then_else(j < valid_block_n, ...)` + `T.if_then_else(i <= j, ...)` — 双重条件
   - 全部通过 `torch.testing.assert_close` 精度验证

2. **`examples/deepseek_v4/inference/kernel.py` L386**：
   ```python
   idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
   ```
   条件 `t * block + i < topk` 中 **t 为 T.serial 循环变量** — 与本设计的 `t * tile_n + j < N` 完全同构。

3. **`tilelang/language/__init__.py`**：`T.if_then_else` 通过 `from .builtin import *` 导出, `hasattr(T, 'if_then_else') = True`。

4. **`T.And` 也可用**：`hasattr(T, 'And') = True`（本设计未使用, 因行尾由输出截断处理, 仅需列条件）。

**设计影响**：本设计保留 GPU 源码的 `T.if_then_else` 掩码方案（适配为统一路径），而非 single-tile 的"取消掩码路径"策略。这是 tiled 变体与 single-tile 的**关键设计差异**之一——tiled 变体必须处理尾 tile 列越界, 而 `T.if_then_else` 是已验证的最佳方案。

### 3.5.4 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/_logsumexp_kernel_single/DESIGN.md` | **极高（同族前序）** | API 映射表、技术约束检测、GPU→NPU 差异表、bf16 处理策略、reduce 输出 shape (block_m,1) 约定、T.copy 尾块切片模式、T.vln 导出名验证 |
| `examples/TileOPs/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | **极高（已验证实现）** | Developer 模式代码结构、`@tilelang.jit(out_idx=[1], target="npuir")` + `_func(block_m)` + `main(x, y)` 签名、v 前缀 API 调用顺序、T.copy 尾块切片、L0/L1/L2/Boundary 测试套件结构 |
| `examples/flash_attention/flash_attn_npuir_dev.py` | **极高（在线递推模板）** | **在线 softmax 递推的完整 NPU 实现**：`T.vmax(acc_m, local_max, new_max)` L64; `T.vsub`+`T.vexp` 计算 correction L65-66; `T.vmul(acc_l, correction, acc_l)` rescale L71; `T.vadd(acc_l, local_sum, acc_l)` accumulate L72; `T.vbrc(0,tmp1)+T.vadd(tmp1,new_max,acc_m)` copy L76-77; `(block_m, 1)` shaped fragment buffers; `T.Pipelined` 循环结构 |
| `testing/npuir/parallel_ops/test_loop_var_compute.py` | **极高（T.if_then_else 验证）** | **证明 T.if_then_else 在 NPU Developer mode 下支持循环变量条件**：Case 1/3/4 使用 `j < valid_block_n`、`i * block_n + j < threshold` 等条件, 全部通过精度验证 |
| `examples/deepseek_v4/inference/kernel.py` | 高 | L386: `T.if_then_else(t * block + i < topk, ...)` — t 为 serial 循环变量的条件, 与本设计 `t * tile_n + j < N` 同构 |
| `testing/npuir/softmax_ops/test_logsumexp.py` | 高 | logsumexp 的 reduce_max → vsub → vexp → reduce(sum) → vln → vadd 调用顺序; Developer (alloc_shared) 和 Expert (alloc_ub) 两种模式; T.vln 正确用法 (L57/L84) |
| `testing/npuir/pooling_ops/test_max_pool2d.py` | 中高 | `T.serial` (小写) 在 NPU Developer mode 中的用法 L47; `T.vmax` 向量 max 用法 L51; Python `if` 在 T.serial 内的用法 (本设计未采用) |
| `testing/npuir/softmax_ops/test_softmax.py` | 中 | softmax 的 reduce_max → vsub → vexp → reduce → vdiv 模式; Developer 模式模板 |
| `examples/norm/layer_norm.py` | 中 | block_m 分块 + alloc_shared + alloc_fragment + T.Parallel + T.cast fp32 提升 + T.reduce_sum fragment 模式 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py` | 源 | GPU 原始实现（迁移基准, L81-169） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/logsumexp.py` | 上下文 | Op 层 wrapper（dispatcher + Kernel 类）; `compute_tile_n` 调用; K9 threads 移除 |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | 上下文 | `align_up` / `DEFAULT_ALIGNMENT=256` / `MAX_SINGLE_TILE_COLS=32512` / `compute_tile_n` 定义 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 输入；M 动态, N 静态(per-kernel build)；N 较大需分块 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(M,)` | same_as(x) | 每行 logsumexp 结果；由 `out_idx=[1]` 自动分配 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 | 生命周期 |
|-----------|-------|-------|----------|------|----------|
| `shared_buf` | `(block_m, tile_n)` | dtype | UB (alloc_shared) | GM→UB 输入 tile 搬运 | 每 tile 覆写 |
| `out_ub` | `(block_m,)` | dtype | UB (alloc_shared) | UB→GM 输出搬运 | epilogue 后写入 |
| `tile_local` | `(block_m, tile_n)` | dtype | Fragment (alloc_fragment) | 输入副本, 供 T.Parallel 读取 | 每 tile 覆写 |
| `tile_f32` | `(block_m, tile_n)` | float32 | Fragment (alloc_fragment) | fp32 工作区, 复用于 shift+exp | 每 tile 覆写 |
| `row_max` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | running max, 跨 tile 累积 | 循环全程 |
| `row_sum` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | running sum → ln → max+ln | 循环全程 → epilogue |
| `tile_max` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | per-tile max | 每 tile 覆写 |
| `tile_sum` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | per-tile sum | 每 tile 覆写 |
| `new_max` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | max(row_max, tile_max) | 每 tile 覆写 |
| `correction` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | exp(old_max - new_max) rescale 因子 | 每 tile 覆写 |
| `tmp1` | `(block_m, 1)` | float32 | Fragment (alloc_fragment) | scratch (vbrc+vadd copy trick) | 每 tile 覆写 |

**与 single-tile 的缓冲区差异**：
- single-tile: `x_ub=(block_m, N)` + `x_local=(block_m, N)` + `x_f32=(block_m, N)` + 2 个 (block_m,1) = 5 个 buffer
- tiled: `shared_buf=(block_m, tile_n)` + `tile_local=(block_m, tile_n)` + `tile_f32=(block_m, tile_n)` + 7 个 (block_m,1) = 10 个 buffer
- tiled 的 (block_m, tile_n) buffer 用 tile_n ≪ N, UB 占用大幅降低; 7 个 (block_m,1) 递推 buffer 极小 (block_m×4B 每个)

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L0 参与）：

[tile 循环内] — 重复 num_tiles 次:
  GM[x tile] --T.copy(slice:real_m×valid_n)--> UB[shared_buf]
    --T.copy--> FRAG[tile_local]
    --T.Parallel+T.if_then_else+T.cast--> FRAG[tile_f32]      (fp32, 尾列 -inf)
    --T.reduce_max--> FRAG[tile_max]                           (bm,1)
    --T.vmax(row_max, tile_max)--> FRAG[new_max]               (bm,1)
    --T.vsub(row_max, new_max)--> FRAG[correction]             (bm,1)
    --T.vexp--> FRAG[correction]
    --T.vsub(tile_f32, new_max)--> FRAG[tile_f32]              (bm,tn, 行广播)
    --T.vexp--> FRAG[tile_f32]
    --T.reduce_sum--> FRAG[tile_sum]                           (bm,1)
    --T.vmul(row_sum, correction)--> FRAG[row_sum]             (rescale)
    --T.vadd(row_sum, tile_sum)--> FRAG[row_sum]               (accumulate)
    --T.vbrc(0,tmp1)+T.vadd(tmp1,new_max)--> FRAG[row_max]     (update max)

[循环结束后 — epilogue]:
  FRAG[row_sum] --T.vln--> FRAG[row_sum]                       (ln(sum))
  FRAG[row_max]+FRAG[row_sum] --T.vadd--> FRAG[row_sum]        (max+ln(sum))
  FRAG[row_sum] --T.Parallel+T.cast--> UB[out_ub]              (fp32→dtype)
  UB[out_ub] --T.copy(slice:real_m)--> GM[y]
```

**关键约束**：所有计算在 fragment (fp32) 上完成；GM 不能直达 fragment，必须经 UB 中转；fragment→fragment 不可用 T.copy（用 vbrc+vadd 替代）。

### 4.5 UB 内存预算

以 Ascend A2/A3 UB = 192KB (196608 bytes) 为基准。UB 上驻留 `shared_buf` 和 `out_ub`（fragment 由编译器管理，生命周期可重叠）：

| Buffer | Shape | dtype | 大小公式 | tile_n=2048 fp16 | tile_n=2048 fp32 | tile_n=4096 fp16 |
|--------|-------|-------|---------|-------------------|-------------------|-------------------|
| `shared_buf` | (bm, tile_n) | dtype | bm×tn×elem | bm×4KB | bm×8KB | bm×8KB |
| `out_ub` | (bm,) | dtype | bm×elem | bm×2B | bm×4B | bm×2B |
| **UB 合计** | | | | ≈bm×4KB | ≈bm×8KB | ≈bm×8KB |

**Fragment 总量（编译器管理，参考 single-tile 经验）**：

| Buffer | Shape | dtype | tile_n=2048 bm=4 | tile_n=4096 bm=4 |
|--------|-------|-------|-------------------|-------------------|
| tile_local | (4, 2048) | fp16 | 16KB | 32KB |
| tile_f32 | (4, 2048) | fp32 | 32KB | 64KB |
| 7×(4,1) fp32 | 7×16B | fp32 | 112B | 112B |
| **Fragment 合计** | | | ≈48KB | ≈96KB |

> **注意**：fragment 是否占用 UB 取决于编译器实现。single-tile 在 block_m=8, N=4096 (fragment 总量 ~96KB) 下正常工作。tiled 的 fragment 总量与 single-tile 中等配置相当, 因 tile_n ≪ N。

**推荐 block_m × tile_n 配置表**：

| N 范围 | dtype | 推荐 block_m | 推荐 tile_n | num_tiles | UB 占用 | 说明 |
|--------|-------|-------------|-------------|-----------|---------|------|
| N ≤ 4096 | fp16/bf16 | 8 | 2048 | ≤2 | 16KB | 小 N, 可选 single-tile 或 tiled |
| N ≤ 4096 | fp32 | 4 | 2048 | ≤2 | 32KB | fp32 占用翻倍 |
| 4096 < N ≤ 32768 | fp16/bf16 | 4 | 2048 | 3–16 | 16KB | attn-weights-32k 场景 |
| 32768 < N ≤ 102400 | fp16/bf16 | 4 | 2048 | 17–50 | 16KB | lm-head-logits 场景 |
| 32768 < N ≤ 102400 | fp32 | 2 | 2048 | 17–50 | 16KB | fp32 减半 block_m |
| 任意 N | 任意 | 1 | 4096 | N/4096 | 8–16KB | 最保守, 单行处理 |

**tile_n 取值策略**（由 op 层 `compute_tile_n()` 计算, 设计需约束）：

1. **UB 约束**：`block_m × tile_n × elem_bytes ≤ 192KB`（单个 shared_buf）
2. **整除优先**：若 N 有合适的因子 ≤ UB 上限, 选 `tile_n | N` 消除尾 tile（如 N=102400, tile_n=2048 → 50 tiles 无余数）
3. **非整除回退**：若 N 无合适因子, 选最大可行 tile_n, 尾 tile 用 `T.if_then_else` 掩码处理（本设计已验证可行）
4. **推荐值**：tile_n=2048（平衡 UB 占用与 tile 数量）; tile_n=4096（减少 tile 数, 需较大 UB）

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 说明 |
|--------|----------|-----------|------|
| M | `T.ceildiv(M, block_m)` 在 T.Kernel grid 中 | 1 ~ 任意 | M 作为 prim_func 的编译时参数；不同 M 值需重新 JIT 编译 |
| N | 静态（per-kernel build） | 固定 | N 在 kernel build 时固定为常量；tile_n 同为编译时常量; num_tiles = ceildiv(N, tile_n) 静态可计算 |

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

**判定依据**: 算子仅包含 reduce + element-wise + 在线递推运算（reduce_max, vmax, vsub, vexp, reduce_sum, vmul, vadd, vln, vbrc），无 matmul，无 Cube 参与。决策树判定为"含归约 → Developer 模式 → T.alloc_shared → UB"。

### 5.2 Block 划分

```python
# 沿 M 维分块（block 级并行）; 沿 N 维分 tile（T.serial 循环）
block_m = config_block_m    # 由 op 层根据 N, dtype, UB 预算选择（见 §4.5 推荐表）
tile_n  = config_tile_n     # 由 op 层 compute_tile_n() 计算（见 §4.5 取值策略）
grid = T.ceildiv(M, block_m)
num_tiles = T.ceildiv(N, tile_n)   # 编译时常量

# T.Kernel(grid, is_npu=True) as (pid_m, _)
# 循环: for t in T.serial(num_tiles)
```

**选择理由**：
- **M 维分块**：每个 block 处理 block_m 行，行间独立（reduce 沿 dim=1），无跨 block 依赖。block_m 取值由 UB 容量约束决定（见 §4.5）。
- **N 维分 tile**：tiled 变体的核心——N 太大不能整体放入 UB, 分成 num_tiles 个 tile 逐块处理。每 tile 大小 tile_n ≪ N, UB 占用从 `block_m × N` 降至 `block_m × tile_n`。
- **在线递推**：跨 tile 维护 running_max 和 rescaled running_sum, 单遍完成（无需两遍扫描 N）。

### 5.3 约束分析

- **对齐约束**: NPU UB 对齐 32 Byte。`T.alloc_shared` 由编译器管理起始地址对齐，无需手动 padding。N 无需 align_up 到 256（GPU 约束已消除）。
- **UB 容量**: `shared_buf` (block_m × tile_n × elem_bytes) + `out_ub` (block_m × elem_bytes) ≤ 192KB。推荐配置见 §4.5。
- **L0 容量**: 不适用（无 Cube 计算）。
- **MAX_SINGLE_TILE_COLS (32512)**: GPU 的列数上限是 LLVM 向量化限制。NPU 无此限制。tile_n 取 2048 或 4096, 远低于 32512。
- **静态循环边界**: `num_tiles = ceildiv(N, tile_n)` — M, N, tile_n 均为编译时工厂参数, num_tiles 静态可计算。满足 Ascend "只支持静态循环边界"约束。
- **递推数据依赖**: 每 tile 的 reduce_max/reduce_sum 依赖当前 tile 数据; row_max/row_sum 更新依赖前一 tile 状态。tile 间严格顺序依赖, 不可并行（但可用 T.Pipelined 重叠 load/compute, 见 §6.3）。

### 5.4 非整除处理策略

**M % block_m != 0（行尾处理）**：

```python
real_m = T.min(block_m, M - pid_m * block_m)
# 输入搬运：src/dst 同时切片，shape 一致 (real_m, valid_n)，消除越界读取
T.copy(x[pid_m*block_m : pid_m*block_m + real_m, t*tile_n : t*tile_n + valid_n],
      shared_buf[0:real_m, 0:valid_n])
# 输出搬运：显式截断 src 为 (real_m,)，与 dst 切片 shape 一致
T.copy(out_ub[0:real_m], y[pid_m*block_m : pid_m*block_m + real_m])
```

**策略说明**（沿用 single-tile 验证方案）：
- 输入侧：用 `T.min` 计算有效行数 `real_m`，对 src 和 dst 同时切片，使 src/dst shape 一致，消除越界读取。`shared_buf` 剩余行（real_m ~ block_m）未初始化, 后续 `T.copy(shared_buf, tile_local)` 搬运完整 block_m 行产生垃圾数据, 但这些行的 reduce 结果在输出阶段被 `out_ub[0:real_m]` 截断丢弃。
- 输出侧：显式截断 src 为 `out_ub[0:real_m]`，使 src/dst shape 一致。
- **不使用 host 侧 padding**：遵守 ascend-constraints.md §4 "host 侧禁止改动输入 NPU 张量内的真实内容"。

**N % tile_n != 0（尾 tile 列处理）— ⭐ tiled 变体核心难点**：

```python
valid_n = T.min(tile_n, N - t * tile_n)
# 1. 切片加载：仅搬运 valid_n 列有效数据，消除 GM 越界读取
T.copy(x[..., t*tile_n : t*tile_n + valid_n], shared_buf[..., 0:valid_n])
T.copy(shared_buf, tile_local)   # 完整 bm×tn, 列 [valid_n:tn] 为垃圾

# 2. 掩码 cast：尾列填 -inf（不影响 max; exp(-inf-max)=0 不影响 sum）
for i, j in T.Parallel(block_m, tile_n):
    tile_f32[i, j] = T.if_then_else(
        t * tile_n + j < N,           # 条件含 T.serial 循环变量 t — NPU 已验证可用
        T.cast(tile_local[i, j], "float32"),
        -T.infinity("float32"),
    )
```

**策略说明**：
- **方案选择**：采用 **T.copy 切片 + T.if_then_else 掩码**（方案 A 的增强版）。GPU 源码用 `T.If/Then/Else` 控制流分支区分末 tile, NPU 不支持循环变量控制流, 故改为**统一路径**——所有 tile 都走 T.copy(slice) + T.if_then_else。非末 tile 的条件 `t*tile_n+j < N` 恒真（编译器优化）, 末 tile 条件为 `j < valid_n`。
- **正确性保证**：
  - reduce_max：-inf 列被 max 忽略（除非全行 -inf, 见 §9.3）
  - reduce_sum：exp(-inf - max) = exp(-inf) = 0, -inf 列贡献 0
  - 垃圾行（i ≥ real_m）：其 per-row reduce 结果为垃圾, 但在输出阶段被 `out_ub[0:real_m]` 截断丢弃, 不影响有效行
- **T.if_then_else 可行性**：已通过 `test_loop_var_compute.py`（NPU Developer mode）和 `deepseek_v4/inference/kernel.py` L386 四重验证（见 §3.5.3）。
- **备选方案**（未采用）：若 tile_n | N（整除）, 则无尾 tile, T.if_then_else 条件恒真, 编译器优化为纯 cast。L0 测试优先使用整除 case 验证主路径, 再用非整除 case 验证掩码路径。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向 | block 级并行 | `T.Kernel(T.ceildiv(M, block_m), is_npu=True)` | 每个 block 处理 block_m 行, 行间无依赖 |
| **N-tile 方向** | **串行迭代** | **`T.serial(num_tiles)`** | 在线递推有跨 tile 数据依赖 (row_max/row_sum), 必须**串行**; ⚠️ 小写 serial (T.Serial 不存在) |
| 元素级 cast+mask | 向量化并行 | `T.Parallel(block_m, tile_n)` + `T.if_then_else` | 逐元素 fp32 cast + 尾列掩码, 无数据依赖 |
| 元素级 cast (输出) | 向量化并行 | `T.Parallel(block_m)` | 逐元素 dtype cast, 从 (bm,1) 提取到 (bm,) |
| reduce_max | 硬件归约 | `T.reduce_max(tile_f32, tile_max, dim=1)` | 硬件加速逐行归约 |
| vmax/vsub/vexp/vmul/vadd/vln/vbrc | 向量指令 | `T.vmax / T.vsub / T.vexp / T.vmul / T.vadd / T.vln / T.vbrc` | 硬件向量指令, 全 buffer 操作 |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
    real_m = T.min(block_m, M - pid_m * block_m)

    # 初始化递推状态
    T.vbrc(-T.infinity("float32"), row_max)     # row_max = -inf
    T.vbrc(0, row_sum)                           # row_sum = 0

    # === 在线递推主循环 ===
    for t in T.serial(num_tiles):                # ⚠️ T.serial (小写)
        # 1. 加载 tile (T.min + 切片)
        valid_n = T.min(tile_n, N - t * tile_n)
        T.copy(x[off_m:off_m+real_m, t*tile_n:t*tile_n+valid_n],
               shared_buf[0:real_m, 0:valid_n])
        T.copy(shared_buf, tile_local)

        # 2. Cast + 尾列掩码
        for i, j in T.Parallel(block_m, tile_n):
            tile_f32[i, j] = T.if_then_else(
                t * tile_n + j < N,
                T.cast(tile_local[i, j], "float32"),
                -T.infinity("float32"),
            )

        # 3-8. 在线递推更新 (全向量操作)
        T.reduce_max(tile_f32, tile_max, dim=1)
        T.vmax(row_max, tile_max, new_max)
        T.vsub(row_max, new_max, correction)
        T.vexp(correction, correction)
        T.vsub(tile_f32, new_max, tile_f32)       # 行广播
        T.vexp(tile_f32, tile_f32)
        T.reduce_sum(tile_f32, tile_sum, dim=1)
        T.vmul(row_sum, correction, row_sum)
        T.vadd(row_sum, tile_sum, row_sum)
        T.vbrc(0, tmp1)
        T.vadd(tmp1, new_max, row_max)

    # === Epilogue ===
    T.vln(row_sum, row_sum)
    T.vadd(row_max, row_sum, row_sum)

    # 输出 cast + 搬出
    for i in T.Parallel(block_m):
        out_ub[i] = T.cast(row_sum[i, 0], dtype)
    T.copy(out_ub[0:real_m], y[off_m:off_m+real_m])
```

### 6.3 流水线优化

**当前版本不使用 T.Pipelined**。理由：
- 在线递推有跨 tile 数据依赖（row_max/row_sum 累积）, tile 间计算不可并行。
- 但 tile 的 GM→UB load 与前一 tile 的 compute 理论上可重叠（load(t+1) ∥ compute(t)）。
- T.Pipelined 需双缓冲 shared_buf（2× UB 占用）, 对大 tile_n 可能超 UB。

**未来优化方向**（不在本次实现范围）：
- `T.Pipelined(num_tiles, num_stages=2)` 双缓冲 shared_buf, 重叠 load/compute。
- 需验证：双缓冲 UB 是否溢出（shared_buf 从 1 份 → 2 份）。
- 参考实现：`flash_attn_npuir_dev.py` L55 `T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2)` — 但 flash_attn 有 Cube/Vector 重叠, logsumexp 纯 Vector 的收益较小。

### 6.4 尾块处理

见 §5.4。包含两种尾块：
- **行尾**（M % block_m != 0）：`real_m = T.min(block_m, M - pid_m*block_m)`, 输入/输出 T.copy 切片。
- **列尾**（N % tile_n != 0, 尾 tile）：`valid_n = T.min(tile_n, N - t*tile_n)`, T.copy 切片 + T.if_then_else 掩码。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步指令，无需手动 `T.sync_block_set` / `T.sync_block_wait`。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| T.copy(GM→UB) 后 | 自动 | 编译器在 MTE2→V 拓扑自动插入 pipe_barrier |
| T.copy(UB→Fragment) 后 | 自动 | T.copy 隐式同步 |
| T.Parallel cast+mask 后 | 自动 | T.Parallel 隐式同步 |
| reduce_max/reduce_sum 后 | 自动 | reduce 是同步操作 |
| vmax/vsub/vexp/vmul/vadd/vln/vbrc 间 | 自动 | 向量指令顺序执行 |
| T.copy(UB→GM) 前 | 自动 | 编译器在 V→MTE3 拓扑自动插入 pipe_barrier |
| **T.serial 循环跨迭代** | 自动 | 递推数据依赖 (row_max/row_sum) 保证顺序; 编译器自动插入迭代间同步 |

### 7.3 pass_configs 配置

无特殊 pass_configs。Developer 模式使用默认编译流水线。

---

## 8. 验证方案

### 8.1 Golden 函数

```python
import torch

def golden_logsumexp_kernel_tiled(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现。

    输入: x of shape (M, N), dtype ∈ {float16, bfloat16, float32}
    输出: y of shape (M,), dtype = same_as(x)

    对应 torch.logsumexp(x, dim=-1)（无 keepdim）。
    在 fp32 上计算以保证精度，最后转回原 dtype。
    """
    return torch.logsumexp(x.float(), dim=-1).to(x.dtype)
```

> **Golden 一致性**：与 single-tile 的 `golden_logsumexp_kernel_single` 完全相同（同一数学运算, 仅 tile 策略不同）。tiled 与 single-tile 应对相同输入产出相同结果（在精度容忍度内）。

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_softmax.py` `_get_tolerances()`：

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 高精度 |
| float16 | 1e-3 | 1e-3 | fp16 容忍度 |
| bfloat16 | 1.6e-2 | 1.6e-2 | bf16 低精度（因 vcast fp32↔bf16 转换损失） |

### 8.3 L0 门槛测试计划

L0 测试聚焦"编译通过 + 基本精度正确"，**必须覆盖大 N 场景（tiled 变体的核心用途）**和**尾 tile 列掩码场景**：

| 测试编号 | Shape (M, N) | dtype | block_m | tile_n | num_tiles | 验证目标 | 容忍度 |
|---------|-------------|-------|---------|--------|-----------|---------|--------|
| L0-1 | (4, 32768) | float16 | 4 | 2048 | 16 | **manifest attn-weights-32k**（大 N, 整除） | atol=1e-3 |
| L0-2 | (4, 102400) | float16 | 4 | 2048 | 50 | **manifest lm-head-logits**（极大 N, 整除） | atol=1e-3 |
| L0-3 | (4, 102400) | bfloat16 | 4 | 2048 | 50 | manifest bf16 极大 N | atol=1.6e-2 |
| L0-4 | (32, 32, 4096)→(1024, 4096) | float16 | 8 | 2048 | 2 | **manifest attn-weights-4k**（多 tile, 小 num_tiles） | atol=1e-3 |
| L0-5 | (4, 5000) | float16 | 4 | 2048 | 3 | **尾 tile 列掩码**（5000%2048=904, 末 tile 904 列） | atol=1e-3 |
| L0-6 | (33, 4096) | float16 | 8 | 2048 | 2 | **行尾非整除**（33%8=1）+ tiled | atol=1e-3 |
| L0-7 | (4, 32768) | float32 | 4 | 2048 | 16 | fp32 大 N（无 cast 损失） | atol=1e-5 |
| L0-8 | (1, 102400) | float16 | 1 | 4096 | 25 | 极小 M（单行）+ 大 tile_n | atol=1e-3 |
| L0-9 | (128, 4096) | float16 | 8 | 1024 | 4 | 较大 M + 较小 tile_n（多 tile） | atol=1e-3 |
| L0-10 | (4, 300) | float16 | 4 | 128 | 3 | **小 N 非整除**（300%128=44, 末 tile 44 列）— 掩码路径验证 | atol=1e-3 |

**通过条件**：全部 10 项通过 atol/rtol 检查。

**测试覆盖矩阵**：

| 维度 | 覆盖情况 |
|------|----------|
| 大 N（tiled 核心用途） | L0-1 (32768), L0-2/L0-3 (102400), L0-7 (32768 fp32), L0-8 (102400 大 tile_n) |
| 尾 tile 列掩码 | L0-5 (5000%2048=904), L0-10 (300%128=44) |
| 行尾非整除 | L0-6 (33%8=1) |
| 整除（主路径） | L0-1, L0-2, L0-3, L0-4, L0-7, L0-8, L0-9 |
| 全 dtype | fp16 (L0-1/2/4/5/6/8/9/10), bf16 (L0-3), fp32 (L0-7) |
| manifest 工作负载 | attn-weights-4k (L0-4), attn-weights-32k (L0-1), lm-head-logits (L0-2/L0-3) |
| 多 tile_n 值 | 2048 (L0-1~7), 4096 (L0-8), 1024 (L0-9), 128 (L0-10) |
| 多 block_m 值 | 1 (L0-8), 4 (L0-1~3/5/7/10), 8 (L0-4/6/9) |

### 8.4 完整分层测试套件（交由 tilelang-op-develop 阶段）

- **L1（功能覆盖）**：全部 manifest 工作负载的 2D reshape shape；3d-multidim-reduce (4,128,4096,dim=[0,2])→M=128,N=16384；keepdim=True/False；1D 输入；多 tile_n × block_m 组合
- **L2（边界）**：M=1, N=1, M=block_m 恰好整除, M=block_m+1, N=tile_n 恰好 1 tile, N=tile_n+1 (2 tiles 末 tile 1 列)
- **Boundary**：极值输入（全 -inf, 全 +inf, 含 NaN, 大幅值 ×50）、随机大 shape (512, 65536)、与 single-tile 交叉验证（相同输入两 kernel 结果一致）

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 | 验证状态 |
|------|------|----------|----------|
| **`T.Serial` 不存在, 必须用 `T.serial`（小写）** | GPU 源码用 `T.Serial`, 直接迁移会 AttributeError | 改为 `T.serial(num_tiles)` | ✅ `hasattr` 验证 + test_max_pool2d L47 |
| **T.if_then_else 循环变量条件** | single-tile DESIGN 声称不支持 — **已过时** | 保留 T.if_then_else 掩码; 条件 `t*tile_n+j < N` | ✅ test_loop_var_compute.py + deepseek_v4 L386 四重验证 (§3.5.3) |
| **fragment→fragment 不可用 T.copy** | `prev_max = row_max` (fragment copy) 无法用 T.copy | 用 `T.vmax` 写入独立 new_max + `vbrc(0,tmp1)+vadd(tmp1,new_max,row_max)` 更新 | ✅ flash_attn_dev L76-77 验证 |
| **bf16 不被 v 前缀 API 支持** | vmax/vmul/vsub/vexp/vadd/vln/vbrc/reduce 均 × bf16 | 全程 fp32 计算：vcast bf16→fp32 → 计算 → vcast fp32→bf16 | ✅ single-tile 验证 |
| **fp16/fp32 reduce 精度** | 大 N (102400) 时 fp16 reduce 可能精度不足 | 全程 fp32 accumulation（即使输入 fp16 也先 cast 到 fp32） | ✅ single-tile 验证 |
| **UB 容量 192KB** | tile_n 过大或 block_m 过大时 shared_buf 溢出 | 参考 §4.5 推荐表; tile_n=2048, block_m=4 时 UB ~16KB | ✅ 预算计算 |
| **递推 buffer 数量多 (7 个 (bm,1))** | 比 single-tile 多 5 个 fragment | 每个 (block_m,1) fp32 = block_m×4B, 7 个共 block_m×28B (block_m=4 时 112B), 可忽略 | ✅ 预算计算 |
| **T.vmax 无独立 doc 页** | 无法从文档确认参数顺序和 shape 约束 | 以 `__init__.py` 导出 + flash_attn_dev L64 + test_max_pool L51 三重示例为准; 参数顺序 (src0, src1, dst) | ✅ 3+ NPU 示例验证 |
| **reduce 输出 shape** | NPU reduce 要求 src/dst rank 一致 | 用 (block_m, 1) 2D; T.Parallel 提取 [i,0] | ✅ single-tile 验证 |
| **T.vln 大小写** | 文档文件名 T.vLn.md, 导出名 T.vln | 优先用 `T.vln`（全小写 l） | ✅ __init__.py L105 + test_logsumexp L57 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **`T.Serial` 大写** | 从 GPU 源码直接复制循环 | AttributeError: 'Serial' | 改为 `T.serial`（小写 s） |
| **fragment→fragment T.copy** | `T.copy(new_max, row_max)` (两者均 fragment) | 编译错误或未文档化行为 | 用 `T.vbrc(0, tmp1) + T.vadd(tmp1, new_max, row_max)` |
| UB 溢出 | tile_n 或 block_m 过大 | 编译失败 / segfault | 减小 tile_n 或 block_m; 参考 §4.5 推荐表 |
| bf16 直接传入 v 前缀 API | 未做 vcast 到 fp32 | 编译错误（dtype 不支持） | 确保先 `T.cast(..., "float32")` |
| T.Parallel 内使用 GM tensor | 直接在 T.Parallel 中读写 x/y | 编译错误 | 仅操作 UB/fragment buffer；GM 读写通过 T.copy |
| vsub broadcast shape 不匹配 | row_max 用 (block_m,) 而非 (block_m,1) | 编译错误 / 结果错误 | row_max/row_sum 等 reduce 输出用 (block_m, 1) 2D shape |
| 输出越界写入 | 未用 real_m 切片限制输出 | y 缓冲区溢出 | `T.copy(out_ub[0:real_m], y[off:off+real_m])` |
| `T.vln` 大小写误写为 `T.vLn` | 文档 T.vLn.md 用 `T.vLn` | AttributeError | 用 `T.vln`（代码导出标准名） |
| 输入侧 T.copy 尾块越界读取 | 未对 src/dst 同时切片 | 最后一个 block 读取越界行 | `T.copy(x[off:off+real_m, t*tn:t*tn+valid_n], shared_buf[0:real_m, 0:valid_n])` |
| **尾 tile 列垃圾影响 reduce** | 仅 T.copy 切片未做 T.if_then_else 掩码 | 末 tile 列 [valid_n:tn] 垃圾数据污染 reduce_max/reduce_sum | T.if_then_else 掩码: `t*tile_n+j < N ? cast(x) : -inf` |
| **num_tiles 非静态** | N 或 tile_n 为运行时变量 | 编译错误（Ascend 要求静态循环边界） | N, tile_n 均为编译时工厂参数; num_tiles = ceildiv(N, tile_n) 静态可计算 |
| 在线递推顺序错误 | 先 shift 再 update max (用旧 max 平移) | 数值错误（exp 溢出） | 严格按 §3.1 步骤顺序: tile_max → new_max → correction → shift(new_max) → sum → rescale → accumulate |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| **N = 102400 (manifest lm-head-logits)** | num_tiles = 50 (tile_n=2048), 整除无尾 tile; T.serial(50) 50 次迭代; 每 tile 4×2048×2=16KB UB |
| **N = 32768 (manifest attn-weights-32k)** | num_tiles = 16 (tile_n=2048), 整除; 或 num_tiles = 8 (tile_n=4096) |
| **N = 4096 (manifest attn-weights-4k)** | num_tiles = 2 (tile_n=2048); 也可走 single-tile (N ≤ 32768), tiled 仍正确 |
| **N = 300 (非整除)** | tile_n=128, num_tiles=3, 末 tile 44 列; T.if_then_else 掩码 -inf 填充 |
| **M = 1** | 单行输入; block_m=1, grid=1; 正常工作 |
| **全 -inf 输入** | reduce_max = -inf; exp(-inf-(-inf))=exp(NaN)=NaN; 与 GPU/torch 行为一致 (torch.logsumexp 对全 -inf 返回 -inf); 如需匹配, op 层特殊处理 |
| **dim=0 归约** | op 层转置为 (N, M) 后调用本 kernel (M/N 互换); 本 kernel 始终沿 dim=1 归约 |
| **多维 dim=[0,2]** | op 层 reshape 为 2D: 非规约维乘积→M, 规约维乘积→N; 如 (4,128,4096,dim=[0,2])→M=128, N=16384 |
| **tile_n | N (整除)** | 无尾 tile; T.if_then_else 条件恒真; 编译器优化为纯 cast; 无额外开销 |
| **tile_n ∤ N (非整除)** | 末 tile 有 valid_n < tile_n 列; T.if_then_else 掩码 -inf; 正确性保证见 §5.4 |

### 9.4 适用边界总结

| 条件 | 是否适用 tiled | 说明 |
|------|----------------|------|
| N > 32768, 任意 dtype | ✅ 适用（核心场景） | single-tile UB 不足; tiled 降低 UB 至 block_m×tile_n |
| 4096 < N ≤ 32768, fp16/bf16 | ✅ 适用 | 也可走 single-tile (block_m=1-2); tiled 可用更大 block_m |
| N ≤ 4096, 任意 dtype | ⚠️ 可用但非最优 | single-tile 更高效 (无循环开销); tiled 仍正确 |
| N = 102400 | ✅ 适用 | manifest lm-head-logits; num_tiles=50 |
| N = 300 (非整除) | ✅ 适用 | T.if_then_else 掩码处理; L0-10 验证 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/TileOPs/_logsumexp_kernel_tiled/
├── _logsumexp_kernel_tiled.py  # 算子实现 + golden + 简单测试
├── DESIGN.md                    # 本设计文档
└── .stage_state.json            # 编排层状态（不由本阶段维护）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `_logsumexp_kernel_tiled.py` | ⬜ 待实现 | 算子实现（kernel + golden + L0 测试） |

### 10.3 命名规范

- 项目目录名: `TileOPs`
- 算子目录名: `_logsumexp_kernel_tiled`
- 实现文件: `_logsumexp_kernel_tiled.py`
- Golden 函数: `golden_logsumexp_kernel_tiled`
- 工厂函数: `_logsumexp_kernel_tiled(M, N, dtype, tile_n)` → `_func(block_m)` → `main`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）— 本文档
2. ⬜ Golden 函数 — `golden_logsumexp_kernel_tiled(x)` 基于 `torch.logsumexp`（与 single-tile golden 相同）
3. ⬜ 算子实现 — `_logsumexp_kernel_tiled.py`：`@tilelang.jit(out_idx=[1], target="npuir") def _func(block_m)` + `@T.prim_func main(x, y)` + T.serial 在线递推 + L0 测试
4. ⬜ 精度比对 — 与 Golden 函数对比，通过 L0 门槛测试全部 10 项

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 规则 1：算子名 = `_logsumexp_kernel_tiled` | ✅ | 不裁剪不变换 |
| 规则 2：`@tilelang.jit` 函数声明 | ✅ | `_func(block_m)` 保留 block_m；移除 K9 threads；添加 target="npuir" |
| 规则 3：`@T.prim_func` 参数名/顺序不变 | ✅ | `main(x: T.Tensor[(M,N), dtype], y: T.Tensor[(M,), dtype])` 完全保持（dtype 作为第二位置参数, 非 dtype= 关键字, 遵循 §3.3 指引） |
| 规则 4：从源码推断输入规格 | ✅ | 从 GPU 源码 + manifest 推断，未询问用户 |
| K9：threads 移除 | ✅ | `_func(block_m)` 无 threads 参数 |
| Host 侧不改输入 | ✅ | 全部计算在 kernel 内；无 host 侧 padding |

### 10.6 与 single-tile 的关键差异总结

| 维度 | single-tile | tiled（本次） |
|------|------------|--------------|
| N 范围 | N ≤ 32768（整体放入 UB） | N 任意大（如 102400），分 tile 处理 |
| 工厂参数 | `(M, N, dtype)` | `(M, N, dtype, tile_n)` — 多 tile_n 参数 |
| 算法 | 单遍 reduce_max → sub → exp → reduce_sum | **在线递推**：跨 tile 维护 running_max + rescaled running_sum |
| N 维循环 | 无（N 整体处理） | `T.serial(num_tiles)` 循环遍历 N-tile |
| 循环 API | 无 | **`T.serial`（小写）** — T.Serial 不存在 |
| 尾 tile 列处理 | 不涉及 | **T.if_then_else 掩码**（条件 `t*tile_n+j < N`, NPU 已验证可用） |
| UB 占用 | x_ub=(block_m, N) | shared_buf=(block_m, tile_n), tile_n ≪ N |
| 中间 buffer | row_max, row_sum (2 个) | row_max, row_sum, tile_max, tile_sum, new_max, correction, tmp1 (7 个) |
| 递推 max 更新 | 无 | `T.vmax(row_max, tile_max, new_max)` (flash_attn 模式) |
| 递推 sum rescale | 无 | `T.vmul(row_sum, correction, row_sum)` (flash_attn 模式) |
| fragment copy | 无 | `T.vbrc(0,tmp1)+T.vadd(tmp1,new_max,row_max)` (flash_attn 模式) |
| T.if_then_else | 取消（声称不支持） | **保留**（已验证支持循环变量条件） |
| 同类参考 | test_logsumexp.py, layer_norm.py | **flash_attn_npuir_dev.py**（在线递推模板）, **test_loop_var_compute.py**（T.if_then_else 验证） |

---

## 11. 修订日志

### revision_index: 0（首次设计, 2026-08-13）

**设计来源**：first_design 模式，迁移类任务（GPU `_logsumexp_kernel_tiled` → NPU target="npuir"）。

**关键设计决策**：

1. **编程模式**：Developer（沿用 single-tile，flash_attn_dev 验证）
2. **在线递推**：采用 flash_attn_npuir_dev.py 的 vmax+vmul+vadd 分解模式（7 个 (block_m,1) fragment buffer）
3. **尾 tile 列掩码**：T.if_then_else（条件 `t*tile_n+j < N`）— 通过 test_loop_var_compute.py + deepseek_v4 四重验证，**纠正 single-tile DESIGN 的过时结论**
4. **T.serial（小写）**：发现 T.Serial 不存在，必须用 T.serial — 运行时 hasattr 验证
5. **统一路径**：取消 GPU 的 _needs_mask 分支，所有 tile 统一走 T.copy(slice) + T.if_then_else（非末 tile 条件恒真，编译器优化）
6. **全程 fp32**：bf16/fp16 → cast fp32 → 计算 → cast 回原 dtype
7. **reduce 输出 (block_m, 1)**：沿用 single-tile 约定
