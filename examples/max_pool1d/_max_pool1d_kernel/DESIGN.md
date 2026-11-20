# MaxPool1dFwdOp（`_max_pool1d_kernel`）算子设计文档

> **迁移任务（TileOPs 同构仓迁移，harness 子模式）**：源算子 = `MaxPool1dFwdOp`
> （family=`pool`），本次重实现的 TileLang kernel 函数 = `_max_pool1d_kernel`
> （→ Kernel class `MaxPool1dKernel`，`ref_api` = `torch.nn.functional.max_pool1d(return_indices=False)`）。
> 本文按「源算子三问解读（§0）→ 算法调研（§1.6.0）→ 硬件耦合性判定与 NPU 重设计（§0.5/§0.6）
> → 通用设计流程（§1–§10）」组织。§1–§10 一律以 §0.6 重设计后的 NPU 算法为输入，不照抄源方案。
>
> - project = `max_pool1d`，op = `_max_pool1d_kernel`
> - 编程模式：**Developer**（迁移 prompt 指定；用户未另行指定）
> - 源算子路径（kernel）：`/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool1d.py`
>   （函数 `_max_pool1d_kernel`，第 15–68 行；同文件第 212–288 行的函数
>   `_max_pool1d_with_indices_kernel` 与第 291–348 行的 wrapped op / 类
>   `MaxPool1dWithIndicesKernel`（双输出 with_indices 变体，manifest `MaxPool1dIndicesFwdOp`）
>   **不在本次迁移范围**）
> - 源 helper：`/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`（`pool_output_dim`，第 80–97 行）
> - NPU 侧抽取参考件（Stage 0 产物，含逐字内联的 `pool_output_dim`）：
>   `examples/TileOPs/tileops/kernels/pool/max_pool1d/_max_pool1d_fwd_kernels.py`
> - NPU 侧集成包（host 层 / Kernel 类）：
>   `examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d.py`
> - 迁移元数据：`examples/TileOPs/tileops/kernels/pool/max_pool1d/.migration_meta.json`
> - 规格来源（workloads / roofline）：`examples/TileOPs/tileops/manifest/pool.yaml`（`MaxPool1dFwdOp`）
> - 精度用例：`examples/TileOPs/tests/ops/test_pool.py`（`MaxPool1dTest` 段，第 512–679 行）
> - 性能用例：`examples/TileOPs/benchmarks/ops/bench_pool.py`（函数 `test_max_pool1d_bench`）
> - 设备常数（本设计会话实查，2026-10-14）：`NPUUtils.get().get_aicore_num()` = **20**
>   （Ascend910B3，`npu-smi info` 显示 910B3）→ 纯 Vector 算子核数翻倍 = **40**；
>   toolchain stamp = `tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388 (commit 417dae9) + CANN ?`。
> - 同族前作（骨架复用对象）：`examples/max_pool2d/_max_pool2d_kernel/DESIGN.md`
>   （CASE-pool-maxpool2d-migration，窗口池化 2D 版，同源文件风格）、
>   `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md`
>   （CASE-pool-adaptivemaxpool2d-migration）、
>   `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`、
>   `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/DESIGN.md`。
> - 版本：**first_design v0**。
> - **hazard 前置声明（来自同族前作教训）**：本算子**必须**独立推导退化特化条件，不得照搬 adaptive
>   家族；且窗口池化（含 1D）的**整平面/整线归约充分条件必须显式要求 `dilation == 1`**——
>   窗口覆盖集合是 `{k_w·d_W} ∩ [0, L_in)`，只有 `d_W == 1` 时才等于整线。本设计以 `d_W == 1`
>   显式约束 + `verify_equiv.py` 的 **C4x 反例组**（2 配置 × 3 dtype）机器锁住。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 方法论：`tilelang-op-design` skill 的
> [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)。
> 逐行读源文件（`max_pool1d.py` 全文 348 行、`common.py` 全文 192 行、抽取件
> `_max_pool1d_fwd_kernels.py` 全文 100 行、集成包 `max_pool1d.py` 全文 231 行），**未臆测语义**。
> 本次迁移范围 = 单输出变体 `_max_pool1d_kernel`（manifest `MaxPool1dFwdOp`）；
> 同文件的 with_indices 变体不在范围内，本文不涉及。

### 0.1 源算子语义（做什么）

**数学语义**（`torch.nn.functional.max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode)`，
`return_indices=False`）：

$$
\text{out}[n,c,o_w]=\max_{0\le k_w<k_W}\
x_{\text{pad}}\big[n,c,\ o_w s_W-p_W+k_w d_W\big]
$$

其中 $x_{\text{pad}}$ 在 $[0,L_{in})$ 之外取 $-\infty$。
源实现以「越界 lane **跳过**」表达该 padding（`max_pool1d.py` 第 53 行
`if iw >= 0 and iw < l_in:`，初值 `max_val = T.cast(float("-inf"), accum_dtype)` 第 49 行）——
跳过 lane 与 padding $-\infty$ 在 $\max$ 下等价（$\max(-\infty,x)=x$），本设计已用
`verify_equiv.py` 的 **E1** 组（NPU 公式 vs `F.max_pool1d`，fp64 参照）机器复核为**逐位一致**。

关键语义要素：

| 要素 | 结论（源实现 + 实测复核） |
|------|--------------------------|
| **规约语义** | 规约轴 = 窗口的 $L$ 方向（`for kw in T.serial(kernel_w)`，第 51 行）；逐 $(n,c,o_w)$ **独立取最大值**，无跨通道 / 跨 batch 归约。比较顺序为 `k_w` 递增，但 $\max$ 可交换可结合，**取值集合**与顺序无关（本算子是纯选择，无累加，故无「累加顺序」敏感项）。 |
| **NaN 语义（max 特有）** | 源码**显式**追踪：`if T.isnan(val): has_nan = True`（第 55–56 行），末尾 `result = T.if_then_else(has_nan, cast(NaN), max_val)`（第 59–63 行）。即**窗口内任一元素为 NaN ⇒ 输出 NaN**。本设计 torch CPU 实测复核（2.9.0）：`max_pool1d([1,nan,3], 2)` → `nan` ✓。 |
| **±inf 语义** | IEEE 最大值语义：`{+inf,-inf}` 同窗 → `+inf`（**不是 NaN**）。故 NaN 探测**不可**用求和类判据（会把 `{+inf,-inf}` 误判为 NaN）。 |
| **dtype 语义** | 门禁 = `{float16, bfloat16, float32}`（源类构造第 147–150 行/集成包第 176–179 行；与 adaptive 家族不同，**含 fp32**）。kernel 内 `accum_dtype = "float"`（第 27 行）：逐元素 `T.cast(x[...], accum_dtype)` 参与比较，末尾 `T.cast(result, dtype)` 单次舍回。**对 max 而言 fp32 是「比较域」而非「累加域」**：-inf 初值、只做选择、输出即被选中元素在 fp16/bf16/fp32 中的原值，`dtype→fp32→dtype` 无损 ⇒ **out 逐位等于窗口内某输入元素（或 NaN）**。这正是精度用例 `atol=rtol=0` 的依据（`examples/TileOPs/tests/ops/test_pool.py` 第 568 行注释原文 "Max is exact (no accumulation): compare bit-for-bit like the GPU suite."）。 |
| **输出尺寸语义** | `out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)`（源第 28 行）。helper 含 `ceil_mode` 的两处修正（`effective_kernel = d*(k-1)+1`；ceil 分支向上取整；末尾「最后一窗起点落在 padding 内则回退一窗」）。**该公式不得在任何地方重算**（抽取件注释原文），本设计逐字引用抽取件的内联副本。 |
| **边界语义** | `padding` 约束 $0\le p_W\le \lfloor k_W/2\rfloor$、`s_W>0`、`d_W>0`（manifest `shape_rules`；op 层校验 "padding must be at most half"）⇒ 每个输出窗口**至少含一个有效输入元素**（无需空窗口兜底，但 `-inf` 初值语义天然给出 `-inf` 兜底）。`ceil_mode` 会令最后一个窗口部分落在 padding（回退规则见上）。 |
| **I/O 契约** | 输入 `x`：`(N, C, L_in)` **恰好 3D NCL**（op 层不接受 2D C×L，与 adaptive 家族不同——manifest 注释 "unlike the adaptive pool family the Op layer accepts exactly 3D NCL input"）；连续布局，尾轴 `L_in` 连续。输出 `out`：`(N, C, L_out)`，`same_as_input`，原生 NCL（**不是转置布局**），`out_idx=[1]`。 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（直接移植源仓测试基准
`F.max_pool1d`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（input） | `(N, C, L_in)`（**恰好 3D**，无 C×L 路径） | float16 \| bfloat16 \| float32 | NCL 连续；尾轴 `L_in` 连续 |
| `out`（output） | `(N, C, L_out)` | same_as_input | NCL 连续；`out_idx=[1]`；`L_out` 由 `pool_output_dim` 决定 |

**输出 shape（迁移必需字段）**：`(N, C, L_out)`，其中
`L_out = pool_output_dim(L_in, k_W, s_W, p_W, ceil_mode, d_W)`。
输出与输入同布局（NCL），**不是转置布局**；无 indices 输出。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（`max_pool1d.py` 第 15–68 行，逐行）：

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host，trace 期） | `out_l = pool_output_dim(...)`；`total_output = n*c_in*out_l`（第 28–29 行） | 编译期 shape 常量 | Python int | 输出长度 + `ceil_mode` 修正 |
| 2（grid） | `T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx`（第 38 行） | `block_m`、`threads` | 一维 block 网格 | 输出元素级并行划分 |
| 3（lane 扁平索引） | `out_idx = bx*block_m + i`；`ow = out_idx % out_l`；`channel_batch_idx = out_idx // out_l`；`c_idx = channel_batch_idx % c_in`；`batch = channel_batch_idx // c_in`；经 `if out_idx < total_output` 守卫（第 40–45 行） | `out_idx` | 标量 `(batch,c_idx,ow)` | 输出坐标 $(n,c,o_w)$ |
| 4（状态初值） | `max_val = T.cast(-inf, "float")`；`has_nan = False`（第 47–50 行） | — | 标量 fp32 + bool | 最大值初值 $-\infty$ + NaN 标记初值 |
| 5（窗口扫描） | `for kw in T.serial(kernel_w):` 内：`iw = ow*stride_w - pad_w + kw*dilation_w`；`if iw >= 0 and iw < l_in:` → `val = T.cast(x[batch,c_idx,iw], "float")`；`if T.isnan(val): has_nan = True`；`max_val = T.max(max_val, val)`（第 51–57 行） | `x` 的逐元素 GM 读 | 标量 fp32 + bool | $\max_{k_w}$（越界 lane 跳过 ≡ $-\infty$ padding）+ NaN 胜出标记 |
| 6（结果 + 写出） | `result = T.if_then_else(has_nan, T.cast(nan,"float"), max_val)`；`out[batch,c_idx,ow] = T.cast(result, dtype)`（第 59–64 行） | 标量 | 标量 + 一次 GM 写 | NaN 判决 + 单次舍回输出 dtype |

**数据流与内存访问模式（源硬件视角）**：

```
GM[x] --逐元素标量读 x[batch,c_idx,iw]--> register(max_val fp32, has_nan bool)
        --NaN 判决 + 单次 cast--> GM[out]
（无 SMEM/L1 中转；每个输出元素现场从 GM gather 其窗口的全部元素；相邻输出窗口重叠时重复读；
  源靠 GPU 的 L1/L2 缓存吸收这些重复访问）
```

**循环与并行结构**：
- grid = `ceildiv(total_output, block_m)` 个 block，每 block `threads` 个 CUDA 线程；
- `T.Parallel(block_m)`：每个线程（lane）负责**一个输出元素**，lane 内全为**标量**计算；
- lane 内循环：`T.serial(kernel_w)` **静态边界**（`k_W` 是 trace 期常量）+ 运行时 `if` 守卫跳过越界 lane；
- **lane 内无归约原语、无向量化**（每 lane 独立标量 `T.max` 链）；
- 无流水线（无 `T.Pipelined`）、无共享内存、无 SIMD。

**host 侧逻辑**（`max_pool1d.py` 第 15–68 行 + 集成包 `max_pool1d.py`）：
- `pool_output_dim` 计算输出长度（**含 `ceil_mode` 修正**，不得重算）；
- 参数规范化（`kernel_size`/`stride`/`padding`/`dilation` 的 int↔1-tuple 展开、`None` stride → kernel）
  与校验（`padding ≤ k//2`、`stride>0`、`dilation>0`、3D NCL、dtype 门禁）在 op 层完成；
- `functools.lru_cache(maxsize=32)` 缓存 kernel factory（按 shape/dtype）；op 层 `_kernel_cache`
  按 `(n,c,l_in,dtype,device)` 缓存 kernel 实例；
- 编译配置：`@tilelang.jit(out_idx=[1], compile_flags=["-O3","-DENABLE_BF16"])`（源第 31 行）；
- host 侧**无**输入张量内容改动（无 padding / 无 reshape 数据拷贝）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **一维 block 网格 + 输出元素扁平映射** | 把 `(N,C,L_out)` 三维问题压成一维可并行任务 | `T.ceildiv(total_output,block_m)` + `%`/`//` 反解 `(batch,c_idx,ow)` | GPU block 网格（非 hardware-specific） | **等价替换**（一维 `T.Kernel`） |
| 2 | **`T.Parallel(block_m)` 线程级 lane 并行** | 每线程处理一个输出元素、隐藏逐元素 gather 延迟 | CUDA thread/warp 两级并行模型 | **CUDA threads / warp**（NPU 无等价） | **重新设计** → §0.6 R1 |
| 3 | **静态边界窗口循环 + 运行时越界守卫** | `k_W` 为 trace 期常量，循环上界静态；守卫跳过 padding lane | `T.serial(kernel_w)` + `if iw >= 0 and iw < l_in` | 无（语言/编译器级约束） | **保留**（NPU `T.serial` 同样只接受静态边界） |
| 4 | **标量 max 比较链 + `-inf` 初值** | 单线程顺序比较、寄存器驻留；`-inf` 是 `max` 的单位元 | `max_val = T.max(max_val, val)` 于 `T.alloc_var` 标量 | **GPU 大寄存器堆 + 寄存器比较链** | **重新设计** → §0.6 R2 |
| 5 | **逐元素标量 NaN 追踪 + 结果覆盖** | 复刻 PyTorch「NaN 胜出」，不依赖 `T.max` 的 NaN 行为 | `has_nan \| T.isnan(val)`；末尾 `has_nan ? NaN : max_val` | 无（纯数值语义） | **保留**（语义层，需在 NPU 上向量化） |
| 6 | **fp32 比较域 + 单次输出舍入** | 与 PyTorch 比较语义一致；max 下无损 | `accum_dtype="float"`，逐元素 cast，末尾一次 cast 回 | 无（纯数值语义） | **保留**（语义层） |
| 7 | **逐元素 GM gather（无 tile 搬运 / 无 coalescing）** | 接口简单，按窗口坐标直接取值 | 直接索引 `x[batch,c_idx,iw]` | GPU 的 L1/L2 缓存吸收重复访问 | **重新设计** → §0.6 R3 |
| 8 | **`T.alloc_var` 标量寄存器状态** | lane 内状态（`max_val`/`has_nan`）寄存器驻留 | `T.alloc_var` + 顺序更新 | GPU 寄存器 + 单线程执行模型 | **重新设计** → §0.6 R1/R2 |
| 9 | **`out_idx < total_output` 尾任务守卫** | 处理 `total_output` 非 `block_m` 整数倍 | 运行时谓词跳过越界 lane | 无 | **保留**（模式可移植） |
| 10 | **`functools.lru_cache` factory 缓存 + op 层 kernel 实例缓存** | 避免重复 trace/编译 | host 侧缓存 | 无 | **保留**（host 侧） |
| 11 | **`-inf` 初值 + 越界跳过（≡ padding −∞）** | 让 padding 不参与 `max`，且无需掩码 | 初值 $-\infty$ + in-bounds 谓词 | 无（数值语义） | **保留**（语义层；已由 §1.6.1 E1 机器验证） |

> 识别不出来 ≠ 不存在。以上 11 项为源码中可见的全部优化/结构性选择；未识别的项会在迁移中被静默丢弃。
> 源码**未**使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / `T.Pipelined`，
> 故无「舍弃」类 GPU 专有手段。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（窗口取最大值；越界跳过 ≡ padding $-\infty$；`ceil_mode` 输出尺寸规则） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1）；已由 §1.6.1 E1 机器验证 |
| **NaN 胜出 + ±inf 最大值语义（#5）** | 语义 | 无 | — | **保留** | 数值语义；NPU 侧用 IEEE 正确的 `eq` 掩码乘性累积实现（§0.6 R2；`ne` 为 ordered-NE 不可用于 NaN 探测） |
| fp32 比较域 + 单次输出舍入（#6） | 语义 | 无 | — | **保留** | NPU Vector 支持 fp32 `T.vmax`/`T.vcast`（`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 fp32 ✓、`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 f32→f16/bf16 ✓） |
| I/O 契约（3D NCL→NCL、三 dtype、同 dtype 输出、无 indices） | 语义 | 无 | — | **保留** | layout 是契约 |
| padding 约束下窗口非空、`ceil_mode` 末窗回退 | 语义 | 无 | — | **保留** | 语义层 |
| 静态边界窗口循环 + 守卫（#3） | 算法 | 无（语言约束，非硬件） | 有（同款约束） | **保留** | NPU `T.serial` 同样只接受静态边界；本算子 `k_W` 是**编译期常量**，上界天然静态 |
| `-inf` 初值 + 越界跳过（#11） | 算法 | 无 | 有 | **保留** | `T.vbrc(-T.infinity("float32"), acc)` 整 buffer 广播；`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（scalar→任意 shape） |
| 尾任务守卫（#9） | 算法 | 无 | 有 | **保留** | `if uid < num_logical` 运行时谓词 |
| lru_cache / kernel 缓存（#10） | host | 无 | 有 | **保留** | host 缓存 |
| 一维 block 网格 + 输出扁平映射（#1） | 优化 | GPU block 网格 | 有（一维 `T.Kernel`） | **等价替换** | migration-analysis.md §5.3「block×thread 两级并行 → block 级任务划分 + 向量化元素并行」；`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3 |
| **`T.Parallel(block_m)` 线程级 lane 并行（#2）** | 优化 | CUDA threads / warp | **无** | **重新设计** → §0.6 R1 | NPU 无 CUDA thread 概念；每 lane 一个输出元素会退化为**标量**执行（pattern-library `PL-1.2-caxis-accum` 同族实测窗口累加 98.3% 标量，`layout.md` 诊断触发器） |
| **标量 max 比较链（#4）** | 优化 | GPU 寄存器堆 + 寄存器比较链 | **无直接等价** | **重新设计** → §0.6 R2 | migration-analysis.md §5.3「register 累加 → UB 中间缓冲 / Vector 归约」；NPU 侧改为 C 轴向量 `T.vmax` |
| **`T.alloc_var` 标量状态（#8）** | 优化 | GPU 单线程执行模型 | **无** | **重新设计** → §0.6 R1/R2 | 状态改为 UB 向量累加器（`acc` / `flag`），lane 映射到 C 轴 |
| **逐元素 GM gather（#7）** | 优化 | GPU L1/L2 缓存 | **无等价（且有害）** | **重新设计** → §0.6 R3 | NPU 逐元素 gather 无 coalescing、无 cache 吸收；改为 UB tile 搬运 + 向量化（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum`） |
| 退化/几何特化（源无；同族模式集） | 优化 | 无 | 有 | **重新设计** → §0.6 R4 | 源对所有网格一律走通用循环；NPU 侧补窗口数学**严格成立**的结构特化（§1.6.1 C4/C4b，C4 含 `d_W==1` 必要约束）——注意窗口池化的特化条件与 adaptive 家族**不同**（见 §0.6 R4） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目 / `docs/Tilelang.language/` 条目 /
> pattern-library 实测条目（见「依据」列）。
> **舍弃项：无。** 源码未使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / swizzle
> 等 GPU 专有手段，故无舍弃条目。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（每项含「源方案 → NPU 新算法 → 语义保持论证」）。
**重设计输入含 §1.6.0 算法调研结论**：调研选定「直接窗口取最大值 + 显式向量 NaN 追踪 + fp32 比较域 +
C 轴向量化 + 核内融合转置 + 输出块沿 L 分块」算法族（与源算法同族，但归约/追踪单元由标量改为 C 向量，
且 L 轴由「逐元素 gather」改为「连续 span 载入 + 输出块分块」），故 §0.6 的重设计直接承接调研结论；
源算法已识别优化手段（§0.4）的优化意图承接见每项「承接」行。

**重设计项 R1：并行结构重设计（线程级 lane 并行 → 一维 Kernel + `(n, o_blk, c_block)` 单元 + C 轴向量 lane）**

- **源方案**：`T.Kernel(ceildiv(total_output,block_m), threads=threads)` + `T.Parallel(block_m)`，
  每 CUDA 线程处理一个输出元素，lane 内 `T.alloc_var` 标量状态。
  性能意图：把 `N*C*L_out` 个输出元素摊到大量线程上并行，靠 warp/线程切换隐藏逐元素 GM gather 延迟。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`；**逻辑任务单元 = `(n, o_blk, c_block)`**，
  其中 `c_block` 是长度 `CH` 的**通道向量块**、`o_blk` 是长度 `BL` 的**输出位置块**；
  `num_logical = N * num_oblk * ceil(C/CH)`（一般网格）。单元内以 **C 轴为向量维**；
  `o_blk` 由单元 id 运行时解码；`k_w` 与块内输出位置 `owl` 为 `T.serial` 内层循环
  （`k_W`、`BL` 均为编译期常量），循环体为 C 向量 op。`num_logical > vector_cores` 时按
  persistent grid-stride 模式挂 `T.serial` 核内循环（静态界，见 §5.5）。
- **为何必须沿 L 分块（1D 相对 2D 的关键差异）**：2D 的 `W_in`（≤112）可整行驻留 UB，
  故 2D 单元只需按输出行分块；1D 的 `L_in` 可达 4096（`sincnet-speaker-local`），
  `(CH, L_in)` 整行驻留远超 UB ⇒ 单元必须同时沿输出轴（`o_blk`）分块，且载入区为
  一个**连续的输入 span**（`LSPAN` 长度）。
- **语义保持论证**：任务划分只是把互不相关的输出元素分给不同单元，**不改变任何窗口的取值集合与
  比较结果**（每个输出仍对其窗口取最大值，越界 lane 仍按 $-\infty$ 处理）；无跨单元归约，
  无边界语义变化。
  承接：源 #1 的一维扁平映射被保留为「一维 Kernel」，#2 的 lane 并行意图由「单元并行 + C 轴向量并行」承接。

**重设计项 R2：归约重设计（per-lane 标量比较链 + 标量 NaN 标记 → UB 内 C 轴向量取最大值 + 向量 NaN 掩码）**

- **源方案**：每输出元素在寄存器里做 `max_val = T.max(max_val, T.cast(x[...],"float"))` 标量比较链，
  并行 `has_nan = has_nan | T.isnan(val)` 标量 bool 累积，初值 $-\infty$/`False`。
- **NPU 新算法**：C 轴向量化（向量轴 = C，§1.6.3）：
  1. 累加器 `acc`（`BL × CH` fp32）经 `T.vbrc(-T.infinity("float32"), acc)` 置 $-\infty$（**单位元是 $-\infty$，非 0**）；
     NaN 追踪器 `flag`（`BL × CH` fp32）经 `T.vbrc(T.float32(1.0), flag)` 置 1。
  2. 逐元素 cast 改为 **tile 级向量 cast**：`T.vcast(row_dst, row_f32, round_mode="rint")` 一次把整块
     `(LSPAN, CH)` 提到 fp32（**fp32 输入时改用 `T.copy`**——同 dtype `vcast` 会退化为 round-to-integer
     误编译，见 §9.2）。
  3. 取最大值改为 **C 轴向量最大值**：`T.vmax(acc[owl:owl+1, :], row_f32[j:j+1, :], acc[owl:owl+1, :])`（原地），
     其中 `j = owl*sW + kw*dW - pW - (base - base_clamped)`（相对缓冲偏移，见 §1.6.1 C1）。
  4. **NaN 追踪向量化为「乘性 eq 掩码」**（max 特有，语义关键）：**每单元一次**（每载入块一行）
     `T.vcmp(row_f32, row_f32, eq_mask, "eq")`（$x{=}x$ 为 1、NaN 为 0）→
     `T.vcast(eq_mask, eq_f32, "rint")`；窗口内逐 (kw, owl)
     `T.vmul(flag[owl:owl+1, :], eq_f32[j:j+1, :], flag[owl:owl+1, :])`（`flag` 初值 1：0 吸收、1 中性）。
     **不依赖 `vmax` 的 NaN 行为**（该行为未文档化；`ne` 为 ordered-NE 亦不可用，见 §9.2）。
     掩码按**块级一次**生成（不是逐窗口元素重算），去掉 `BL·k_W` 倍的 `vcmp+vcast`（同族前作
     `PL-1.22-maxreduce-nan-tracking` ③' 的实测细化）。
  5. 输出：`cfinal = T.vcmp(flag, zero, "eq")`（`flag==0` ⟺ 窗口含 NaN）→
     `T.vselect(cfinal, nanbuf, acc, out32)`（`nanbuf = inf - inf` = NaN）→
     `T.vcast(out32, acc16, round_mode="rint")`（单次舍回 dtype；fp32 输入改用 `T.copy`）→
     `T.transpose(acc16, acc_t, [1,0])` → `T.copy` 写回。
- **语义保持论证**：
  - **数学等价**：窗口扫描不改变取最大值的元素集合；C 轴是**独立轴**（每通道独立取最大值，§0.1），
    把 `c` 放进向量 lane 只是 SIMD 数据并行，**不改变任何 `c` 的取值集合**。
  - **NaN/±inf 语义核对**（逐条对齐 §0.1）：`{+inf,-inf}` 无 NaN → `eq_mask` 恒 1 → `flag` 保持 1 →
    `cfinal` 恒假 → 输出 `acc`，而 `T.vmax(-inf,+inf)=+inf` ✓；窗口含 NaN → 对应 lane 的掩码为 0 →
    `flag` 变 0（吸收）→ `cfinal` 真 → 输出 NaN ✓。本设计**只用 IEEE 正确的 `eq` 比较**
    （`x!=x` 的 ordered-NE 陷阱见 §9.2），与求和类探测（会把 `{+inf,-inf}` 误判为 NaN）本质不同。
  - **dtype/舍入**：与源一致（fp32 比较域、末尾单次 `rint`）；`vcast` fp16/bf16→fp32 为无损上转，
    fp32→fp16/bf16 用 `rint`；对 max 结果无损（输出即输入元素）。
  - 承接：源 #4 的「寄存器比较链」意图由「UB fp32 向量最大值累加器」承接；#5 的「标量 bool NaN 追踪」
    由「C 向量 eq 掩码乘性累积」承接；#6 由「fp32 向量比较域 + 单次 `rint`」承接。

**重设计项 R3：数据搬运重设计（逐元素 GM gather → UB 连续 span 载入 + 核内融合转置）**

- **源方案**：直接在 GM 上按 `x[batch,c_idx,iw]` 逐元素取值。性能意图：接口简单；
  GPU 靠 L1/L2 缓存吸收重叠窗口的重复访问。
- **NPU 新算法**：按 C 轴向量化所需布局（C 最内连续）搬运，且沿 L 轴做**连续 span 载入 + 钳制**：
  1. 单元解码后求载入范围（**纯标量，每单元 2 次钳制**）：`base = ow0*sW - pW`、
     `span = (BL-1)*sW + (k_W-1)*d_W + 1`、`LSPAN = min(span, L_in)`、
     `s0 = max(0, min(base, L_in - LSPAN))`、`lo = max(0, base)`、`hi = min(L_in, base + span)`；
  2. GM→UB：按 `(real_c, hi-lo)` 片整片搬运 `T.copy(x[nb, cb*CH:cb*CH+real_c, s0:s0+LSPAN], row_src[0:real_c, 0:LSPAN])`
     —— **源起始 `s0` 为运行时值但恒在界内（`0 ≤ s0` 且 `s0+LSPAN ≤ L_in` 由钳制保证）、
     长度为静态 `LSPAN`**，避免动态偏移 subview（见 §9.2 陷阱）；
  3. UB 内**融合转置**：`T.transpose(row_src, row_dst, [1,0])` 把 `(CH, LSPAN)` 变成 `(LSPAN, CH)`（C 最内）；
     依据 `PL-1.1-transpose-chain`（核内二轴交换，实测 ~5.4µs / 4 次；`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 仅支持二轴交换）；
  4. 转置后 C 轴连续 → `T.vcast` 到 fp32 → 向量取最大值 / eq 掩码；
  5. 输出侧：`out32 (BL, CH)` → `vcast` → `acc16` → `T.transpose([1,0])` → `acc_t (CH, BL)` → `T.copy` 回 GM。
- **语义保持论证**：搬运/转置是**纯数据重排**（双射），不改变元素值与窗口取值集合；输出写回后与 NCL
  契约一致。无精度影响（不引入任何算术）。载入钳制只影响**载入的地址范围**，读出时仍按
  `0 ≤ iw < L_in` 逐 lane 守卫（§1.6.1 C1 的等价改写保证两者一致，`verify_equiv.py` C1 组机器验证）。
  承接：源 #7「靠缓存吸收重复访问」的意图由「UB 连续 span 常驻 + 同一输入元素被其归属的所有
  (kw, owl) 复用（每单元载入一次，`kW·BL` 次读取）」承接；整数级读放大由 §1.6.0 R3 口径量化
  （`sW < (k_W-1)·d_W+1` 时相邻输出块 halo 重叠；`BL` 越大放大越小）。

**重设计项 R4：结构特化（源无特化；窗口池化的特化条件与 adaptive 家族不同）**

> 该项属「算法替代」模式（migration-analysis.md §6.2），采纳项与 §1.6.0 调研结论一致、
> 并在 §1.6.1 完成等价性论证与机器验证。

- **源方案**：对所有输出长度一律走通用窗口循环（含 `L_out==1` 与 `L_out==L_in` 类形状）。
- **NPU 新算法 + 关键校正（本设计独立推导）**：adaptive 家族的两条退化特化（`out==1` → 整平面归约；
  `out==in` → 恒等拷贝）**不能照搬到窗口池化**——adaptive 的 bin 边界随 `out` 变化（`out==in` 时
  bin 恰为单位 bin），而**窗口池化的窗口形状由 `k/s/p/d` 固定**。逐条给出真正成立的充分条件：
  - **C4（整线归约）**：**充分条件**为 `L_out==1` 且 `p_W==0` 且 `d_W==1` 且
    `(k_W-1)·d_W+1 >= L_in`（此时窗口恰为 $[0,L_{in})$，输出 = 整线最大值）。**`d_W==1` 是必不可少的
    约束，省略即错**：窗口在 L 轴的**输入覆盖集合**是 $\{k_w d_W\}\cap[0,L_{in})$（等差子集），
    **仅当 $d_W=1$ 时才等于 $[0,L_{in})$**；$d_W>1$ 时未被覆盖的输入位置不参与窗口取最大值，
    而整线 `reduce_max` 会把它算进去 ⇒ 取值集合不同。**反例**：`L_in=5, k_W=2, s_W=1, p_W=0, d_W=4`
    时 `L_out=1`，且覆盖长度条件成立，但窗口仅覆盖位置 $\{0,4\}$；输入仅 `x[2]=1`、其余为 0 时
    `F.max_pool1d` = **0.0**，整线归约 = **1.0**（torch 数值复核 + `verify_equiv.py` C4x 组机器锁住）
    ⇒ **静默错值**，故必须显式排除。
    为降低实现与验证风险，本设计**不**利用 `L_in==1` 的特例放宽整线覆盖（数学上可写 `d_W==1 or L_in==1`），
    只用上述最简充分条件形态。上述四合一条件在 factory 层折叠为纯 Python bool（§3.3 `IS_GLOBAL`），
    **满足即走 C4，不满足一律走一般路径**（`d_W>1` 的 `out=1` 配置因此被正确挡在 C4 之外）。
    **注：manifest 的 `textcnn-global`（`[64,256,128]` k128s1p0，`L_out=1`）恰好触发 C4**——
    这是 1D 相对 2D 的一处差异（2D 的 manifest 三档均不触发任何特化）。
  - **C4b（恒等拷贝）**：`out==x` 当且仅当 `k_W==1 and d_W==1 and p_W==0 and s_W==1`（窗口恰为单元素）。
    **反例**：`L_in=5, k_W=3, s_W=1, p_W=1` 时 `L_out=5==L_in` 但窗口为 `{o_w-1,o_w,o_w+1}`（3 元素）
    ⇒ 恒等拷贝会给出错值。
- **语义保持论证**：见 §1.6.1（C4/C4b 在**上述充分条件成立时**与基线公式**逐位等价**，
  `verify_equiv.py` 全部 `EQUIV_PASS`）。C4 的 NaN 判决由独立 eq 掩码归约给出
  （`vcmp(eq)+vcast+reduce_min` 与 0 比较，**不用 `reduce_max`**——0/1 掩码上 `reduce_min==0` 才是
  「存在 NaN」，`reduce_max==0` 只在全 NaN 成立），**不依赖 `reduce_max` 的 NaN 行为**。
- 承接：源「通用窗口循环」在非退化网格上保持不变（不存在优化意图丢弃）；退化网格特化**不改变语义**，
  仅减少冗余循环与 op 数。**注意**：C4 被 `textcnn-global` 触发，C4b **不被当前 manifest workloads 触发**，
  属对 op 层一般参数空间的正确性/性能补全，容量受 §4.5 判据约束。

**重设计后的 NPU 算法一句话总结**：NCL 进 / NCL 出，核内按 `(n, o_blk, c_block)` 单元把 C 轴作为向量轴，
沿 L 轴做一次**钳制连续 span 载入**（`(CH, LSPAN)`），对 `k_W` 窗口做 fp32 向量取最大值（`T.vmax`），
并在**同一遍扫描内**用块级 eq 掩码乘性累积（`vcmp(eq)`+`vcast`+`vmul`）追踪 NaN，用核内二轴转置把 C
搬到最内连续轴；满足充分条件的退化长度（整线 / 单元素恒等）走结构特化路径。

### 0.7 标杆实现

- 源算子路径（获取 golden 语义依据）：`/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool1d.py`
  （函数 `_max_pool1d_kernel`）。
- 源仓参考实现 / 测试基准（NPU 侧移植后）：
  - `examples/TileOPs/tests/ops/test_pool.py`（`MaxPool1dTest.ref_program` = `F.max_pool1d(...)`，
    torch CPU 参考，`atol=rtol=0`；`dilation != 1` 时按测试注释在 CPU 上求 golden 再搬回设备）。
  - 源仓同文件（GPU）`tests/ops/test_pool.py` 的 `MaxPool1dTest`。
- **golden 以 §0.1 语义为唯一依据**，直接使用 `torch.nn.functional.max_pool1d`
  （优先移植源仓基准），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。
- 规格来源：`examples/TileOPs/tileops/manifest/pool.yaml` 的 `MaxPool1dFwdOp`
  （4 组 workloads；`roofline.flops = N*C*out_W*kW`、`bytes = (N*C*L_in + N*C*out_W)*elem_bytes`）。
- 基准来源：`examples/TileOPs/benchmarks/ops/bench_pool.py` 中的函数 `test_max_pool1d_bench`。

---

## 1. 概述

### 1.1 算子名称

`_max_pool1d_kernel`（迁移自 TileOPs `MaxPool1dFwdOp` / `MaxPool1dKernel`）。

### 1.2 功能描述

窗口式（非自适应）一维最大池化前向：对每个输出位置 $o_w$，在其
`kernel_size × stride × padding × dilation` 决定的窗口内取最大值；
窗口越界部分按 $-\infty$ 填充（源实现以「越界 lane 跳过」表达）；
窗口内任一元素为 NaN 时输出 NaN（NaN 胜出）。
输入 **恰好 3D NCL**，dtype 支持 `float16 / bfloat16 / float32`，输出同 dtype。

### 1.3 数学公式

$$
\text{out}[n,c,o_w]=\max_{0\le k_w<k_W}\
x_{\text{pad}}\big[n,c,\ o_w s_W-p_W+k_w d_W\big]
$$

$$
x_{\text{pad}}[n,c,i_w] = x[n,c,i_w]\ \ \text{if}\ \ 0\le i_w<L_{in};
\qquad x_{\text{pad}} = -\infty\ \ \text{otherwise}
$$

窗口含 NaN ⇒ 输出 NaN（NaN 胜出；$\{+\infty,-\infty\}$ 同窗取 $+\infty$）。
输出长度 $L_{out}=$ `pool_output_dim`$(L_{in},k_W,s_W,p_W,\text{ceil\_mode},d_W)$
（helper 逐字取自源仓 `common.py`，**不重算**）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，见 §0.6）** 单步 + 窗口归约类算子，纯 Vector：

1. host/trace 期（factory 内）：由 `(N,C,L_in,k_W,s_W,p_W,d_W,ceil_mode)` 算出输出长度 `L_out`
   （`pool_output_dim`）；判定退化特化路径（§1.6.1 C4/C4b 的**充分条件**）与 UB 容量档
   （通道向量块 `CH` 与**输出位置块 `BL`**，§4.5）；`num_logical / num_kernels / num_local` 在 factory
   层求为编译期常量。
2. 一般网格：逻辑任务单元 `(n, o_blk, c_block)`（`BL` 输出位置 × `CH` 通道）。单元内：
   - 计算载入范围 `base/span/LSPAN/s0/lo/hi`（纯标量，2 次钳制）；
   - `acc`（`BL × CH` fp32）经 `T.vbrc(-inf)` 置 $-\infty$；`flag`（`BL × CH` fp32）经 `T.vbrc(1.0)` 置 1；
   - `T.copy` 载入 `(real_c, LSPAN)` 连续 span → 核内转置 `(LSPAN, CH)` → `vcast` fp32（fp32 输入用 `T.copy`）；
   - 块级一次 `T.vcmp(row_f32, row_f32, eq_mask, "eq")` + `T.vcast` → `eq_f32`；
   - `for kw in T.serial(k_W): for owl in T.serial(BL):` 算 `iw = base + owl*sW + kw*dW`，
     若 `0<=iw<L_in`（**逐 lane 守卫**，相对缓冲下标 `j = iw - s0`）则 `T.vmax` 更新 `acc[owl]`，
     并 `T.vmul(flag[owl], eq_f32[j], flag[owl])`；
   - 末尾：`cfinal = (flag == 0)`（⟺ 窗口含 NaN）→ `out32 = vselect(cfinal, nanbuf, acc)`
     （`nanbuf = inf-inf` = NaN）→ `vcast` 回 dtype（fp32 用 `T.copy`）→ `transpose([1,0])` 回转 `(CH, BL)`
     → 写回 GM。
3. 退化长度（§1.6.1 C4/C4b 充分条件成立时）走结构特化路径（整线 `reduce_max` / 单元素恒等拷贝）。

**与源算法的结构差异**（来源标注）：

- 归约单元：源「每输出元素一个线程 + 标量比较链」→ NPU「每单元一个 C 向量块 + `T.vmax` 向量比较」（§0.6 R2）。
- NaN 追踪：源「标量 bool `has_nan \| isnan(val)`」→ NPU「C 向量 eq 掩码乘性累积」（§0.6 R2；语义不变）。
- 访存：源「逐元素 GM gather」→ NPU「UB 连续 span 载入 + 核内融合转置」（§0.6 R3）。
- 并行：源「block×thread 两级 + lane/元素」→ NPU「一维 Kernel + `(n,o_blk,c_block)` 单元 + C 轴向量 lane」（§0.6 R1）。
- **L 轴分块（1D 新增）**：源无分块（逐元素未载入）；NPU 因 `L_in` 可达 4096 必须沿输出轴做 `BL` 块
  - 连续 span 载入（§0.6 R1/R3）。
- 守卫：源「逐元素 in-bounds 谓词」→ NPU「载入范围钳制（每单元 2 次标量）+ 逐 lane 相对守卫」（§1.6.1 C1）。
- 结构特化：源无 → NPU 有（§0.6 R4 的**校正充分条件**，见 §1.6.1 C4/C4b）。
- **不变**：窗口取值集合、$-\infty$ padding 语义、NaN 胜出 + $\pm\infty$ 取 $+\infty$、`ceil_mode` 输出尺寸规则、
  fp32 比较域 + 单次输出舍入、NCL 契约。

### 1.5 数据流图

```
GM[x: (N,C,L_in) fp16/bf16/fp32]
   │  T.copy 连续 span (real_c, LSPAN)  [s0 钳制, 静态长度]
   ▼
UB[row_src: (CH, LSPAN) dtype] --T.transpose[1,0]--> UB[row_dst: (LSPAN, CH) dtype] --T.vcast(rint)--> UB[row_f32: (LSPAN, CH) f32]
   │  块级一次 NaN 探针：vcmp(row_f32,row_f32,"eq") + vcast --> UB[eq_mask/(eq_f32)]
   │  T.serial kw (静态界 k_W) × T.serial owl (静态界 BL) + 逐 lane 守卫
   ▼
   ├─ 值：   UB[acc  (BL,CH) f32 <- vmax(acc[owl], row_f32[j])，初值 -inf]
   └─ NaN：  UB[flag (BL,CH) f32 <- vmul(flag[owl], eq_f32[j])，初值 1]
   │
   │  cfinal = vcmp(flag, zero, "eq")            # flag==0 ⟺ 窗口含 NaN
   ├─> UB[out32 f32] = vselect(cfinal, nanbuf, acc)  --vcast(rint)--> UB[acc16] --transpose[1,0]--> UB[acc_t (CH,BL)] --T.copy--> GM[out]
（退化路径 C4：GM → UB 连续 (CH,L_in) → vcast f32 → reduce_max(dim=1→1)；
              eq 掩码 → vcast → reduce_min(dim=1→1) 判 has_nan；vselect 判决 → vcast → GM）
（退化路径 C4b：GM → UB 整线 (CH,L_in) → GM 纯拷贝）
（无 GM 中间往返、无 Cube / L1 / L0C 路径）
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**——对同一数学语义的算法族回答调研四问，选定算法族（1.6.0）；
> 再在保证数学等价的前提下优化公式（1.6.1）；再把保留下来的计算全部交给向量单元（1.6.2）；
> 最后回答「在哪个轴向量化」（1.6.3）。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = **同一数学语义的算法族**（所有能算出窗口最大值的结构）。
> 基线候选 = 输入公式 / 源算法（直接窗口取最大值，逐元素标量 + 标量 NaN 追踪）。
> 本算子属**窗口 / 归约类** → **完整调研**（R1 候选表 + R3 四口径表 + R4 逐候选评估）。
> 参考表命中行：`algorithm-candidates.md` 的 `ALG-pooling`（family=pooling）——
> `R1` 命中「avg pool = 常数权重卷积；sum pool × 常数 = avg pool」；`R2`「滑窗天然流式」；
> `R3`「窗口重叠数据的重复读」；`R4`「C 轴向量轴；跨步系数」；`known_impl` 指向 avg_pool2d 系列，
> `kb_links = PL-1.2-caxis-accum / PL-1.3-host-permute / PL-1.4-tiling-heuristic`。
> **命中行已全部评估**；其中 `R1` 命中项对 **max 不成立**（见 R1 候选 1/2）——这是本算子与同族
> avg 池化的关键差异。另命中 pattern-library 的 `PL-1.22-maxreduce-nan-tracking`
> （max 池化/归约的 NPU 向量实现形态，两任务独立实证；**本设计沿用其 ①–⑥ 全部结论**）。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算法**（直接窗口取最大值，逐元素标量 + 标量 NaN 追踪） | $\max_{k_w}x_{\text{pad}}$ | — | — | ✅（基线） |
| 0' | **基线向量化**（C 轴向量 `vmax` + 块级向量 eq 掩码 NaN 追踪 + L 轴 span 载入分块） | 同 #0 | **数学恒等**（C 轴独立，仅 SIMD 化；L 分块只改访存不改取值集合） | 标量链 → 向量链（`/CH` 次向量 op）；逐元素 gather → span 载入复用 | ✅（选定） |
| 1 | 常数权重卷积 / im2col+Cube GEMM（`ALG-pooling` 的 `R1` 命中行） | $\text{out}=W_0\otimes x$（0/1 权重） | **❌ 不等价**：$\max$ 非线性，**不存在权重矩阵使 $\max$ 等于线性加权和**；命中行仅对 avg/sum pool 成立；且 `L_out` 作 Cube N 轴常 < 分形下限 16 | — | ❌（数学不等价） |
| 2 | sum-pool × 常数（`ALG-pooling` 命中行） | $(\sum x)\cdot c$ | **❌ 不等价**（给出 avg/sum，不给 max） | — | ❌（数学不等价） |
| 3 | 前缀最大值 + 差分 | $P[b]-P[a]$，$P$ = 前缀最大值 | **❌ 不等价**：$\max$ **无逆元**，$\max P[b],P[a]$ 之差**不等于**区间最大值（例：$x=[5,1,3]$，区间 $[1,3)$ 的 max $=3$，而 $P[3]{-}P[1]=5{-}5=0$） | — | ❌（数学不等价） |
| 4 | 归约结构优化：值与 NaN 判决**同一遍扫描内融合**（源已同轮） | 单轮：`vmax` + eq 掩码 | 数学恒等（同轮与分轮结果相同） | 避免第二遍扫描 | ✅（并入 #0'） |
| 5 | **访存量削减：连续 span 载入 + 块内复用**（源为逐元素 GM gather，同一输入被 `k_W` 个窗口重复读） | 单元载入一次 `(CH, LSPAN)`，块内 `k_W·BL` 次读取全命中 UB | 数学恒等（读的是同一批元素；载入钳制只改地址范围，读出仍按 `0<=iw<L_in` 守卫） | GM 读次数由 `N·C·L_out·k_W` 降至 `N·C·(L_in + halo)` | ✅（并入 #0'） |
| 6 | 退化/几何结构特化（整线归约 / 单元素恒等） | `L_out==1`: $\max_{\text{all}}x$；恒等: `out=x` | 数学恒等（**仅在窗口数学上覆盖整线 / 恰为单元素时**；充分条件见 §1.6.1 C4/C4b，C4 含 `d_W==1`） | 整线：多次向量 `vmax` → 单次归约；恒等：全部归约 op → 纯拷贝 | ✅ |
| 7 | 可分离 1D→无（本算子已是 1 个空间维） | — | **不适用**（1D 无可分离结构；2D 的可分离候选对 1D 退化） | — | ❌（结构不适用） |

**R1 初判小结**：唯一数学等价的化简结构是「**退化长度结构特化**」（#6，**条件需按窗口语义重新推导**）、
「**NaN 判决融合**」（#4）与「**连续 span 载入的访存量削减**」（#5，1D 相对源的关键补强）；
与 avg 池化不同，max 的**全部代数化简路径（#1/#2/#3）因 $\max$ 非线性 / 无逆元而不成立**——
这是 max 归约相对 sum 归约的本质限制（数学事实，非 API 或性能问题）。

**R2 在线算法**：**有**（且基线本身已是单遍）。结构判据：算子可分解为 **running 最大值**
（窗口内最大值）+ **running NaN 标记**（窗口内是否已见 NaN），无跨元素依赖，故天然存在单遍流式变体——
每个输入元素读一次、参与其归属（窗口重叠时若干）输出的 `vmax` / NaN 判决。
**收益口径**：扫描遍数 **1**（无两遍式）、中间缓冲 `O(BL·CH + LSPAN·CH)`（远小于 `O(L_in·CH)` 的物化）、
UB 驻留可行（§4.5）。迁移任务佐证：源算法 §0.4 #4/#5 已是 running 比较与 running 标记
（`max_val = T.max(max_val,val)`、`if T.isnan(val): has_nan = True`），是「存在在线变体」的直接证据。
结论：**在线单遍变体 = 采用（= 基线结构）**，无第二遍扫描 / 无中间矩阵物化。

**R3 复杂度对比**（逐候选四口径；本算子是**带宽受限**算子，访存 / 遍数为主导项，不只看 FLOPs）：

口径：FLOPs = 元素访问次数（一次最大比较 = 1）；访存 Bytes = GM 读 + GM 写（**不含** UB 内搬运；
中间缓冲不落 GM）；扫描遍数 = 输入 GM 读取遍数；中间缓冲峰值 = 片上驻留量。

| 算法候选 | FLOPs（比较次数） | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|--------------------|---------|-------------|---------------------|
| 0 基线（直接标量） | `N·C·L_out·k_W` | `(N·C·L_in + N·C·L_out)·eb` | 1（窗口重叠时靠缓存吸收重复读） | `O(1)`（寄存器标量） | 元素级并行 `N·C·L_out`；**无跨核归约** |
| **0' 基线向量化（选定，C 轴 + span 载入分块）** | 同上（`/CH` 次向量 `vmax`） | 同上 + halo 读放大 `≈ LSPAN/(BL·sW)`（sincnet `k=s` = 1.009；ecg 重叠窗 BL=31 时 ≈ 1.11） | 1 | `acc+flag+row_f32+eq_*` = `O((BL + LSPAN)·CH)` | 单元并行 `N·num_oblk·ceil(C/CH)`；**无跨核归约** |
| 1 常数权重卷积 / GEMM | **不适用**（非线性，无法表达） | — | — | — | — |
| 2 sum-pool × 常数 | **不适用**（给出的不是 max） | — | — | — | — |
| 3 前缀最大值 + 差分 | **不适用**（max 无逆元） | — | — | — | — |
| 6 退化特化（整线归约） | **1 次向量归约**（vs `L_out` 次 × `k_W` 次 `vmax`） | 同上 | 1 | `(CH,L_in)` 块 + 掩码块（容量判据见 §4.5） | 同上 |

> 口径含：GM 输入读 + 输出写；不含：UB 内搬运 / 转置 / NaN 追踪 / 守卫（片上，计入 §1.6.0 估算行的发射项）。
> **基线在表内**（#0）；选定候选 #0'。候选 1/2/3 因**数学不等价**不参与复杂度对比（标注 N/A）；
> 候选 4/5 并入 #0'；候选 6 纳入（退化长度）；候选 7 结构不适用。
> **注（max 的额外发射代价）**：以 eq 掩码探测 NaN 需每元素额外 `vcmp + vcast + vmul`（本设计把
> `vcmp+vcast` 提到**块级一次**，仅余每元素 1 条 `vmul`）——avg 每元素只需 1 条 `vadd`，故发射量约为
> 同族 avg 的 2×；这是保持「NaN 胜出」语义的**必要代价**，非可省项。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线（标量） | **Vector 单元浪费**（逐元素标量比较，`PL-1.2` 同族实测窗口累加 98.3% 标量） | 寄存器 | — | 静态（`k_W` 编译期常量） | 无 | ❌ 计算单元失配（重设计为 #0'） |
| **0' 基线向量化（C 轴，选定）** | **Vector** ✓（`T.vmax` fp32 ✓ / `T.vcmp` fp32 ✓ / `T.vselect` fp32 ✓ / `T.vcast` ✓ / `T.vbrc` ✓ / `T.transpose` ✓） | 见 §4.5（代表档 ≈ 110 KiB 原始 / ×1.7 → ≈ 188 KiB，对照 192 KiB；`CONST-capacity-910B2C`） | **C 轴恒为 8 的整数倍**（C=8/80/128/256；fp16/bf16 向量宽 8、fp32 宽 4，8 两者皆是倍数）；`L_in`(32/128/2048/4096) 与 `LSPAN` 仅作转置后 row 长度，不作向量轴，不要求整除 | ✓（`k_W`/`BL`/`LSPAN` 编译期常量；动态量仅 `o_w`/`iw` 派生的标量守卫与 2 次钳制） | 无跨引擎流水；退化长度可 reduce 融合 | ✅ **选定** |
| 6 退化特化 | Vector reduce ✓ | `(CH,L_in)` 块 + 掩码块（容量判据 + 回退，§4.5） | ✓ | ✓ | 单归约 | ✅ 采纳（充分条件成立时） |
| 1 / 2 / 3 | **不适用**（max 非线性 / 无逆元，非 MAC 密集） | — | — | — | — | ❌ 数学不等价，无需亲和评估 |
| 7 可分离 | **不适用**（1D 无可分离结构） | — | — | — | — | ❌ 结构不适用 |

**R4 弃选理由的证据引用（negative-claim-evidence 口径）**：

- 候选 1/2/3：**数学事实**（$\max$ 非线性、无逆元），非 API / 性能断言——不作 API 检索需求
  （不存在可检索的「最大化简公式」）；候选 1 的额外「`L_out < 分形下限 16`」为分形事实
  （`docs/Tilelang.language/线性代数操作/T.gemm.md` 的分形约束 + migration-analysis.md §5.3
  「Tensor Core mma → Cube：分形限制 M≥16 / N≥16 / K≥32」）。
- 候选 0 的「标量占比」反例引用 pattern-library `layout.md` 的 `PL-1.2-caxis-accum` 与文档头
  「诊断触发器」（「窗口累加 98.3% 标量源于 `j·sW`（`sW≠1`）跨步系数阻碍向量化，原轴微调收益为 0，
  换轴后 5.6x」）。
- 候选 0'/6 的 dtype 支持矩阵来自 `docs/Tilelang.language/`（`T.vmax` / `T.vcmp` / `T.vselect` /
  `T.reduce_max` / `T.vcast` / `T.vbrc` / `T.transpose` 各 §2.2.1 表，见 §3.4）。
- **bf16 注意**：`T.vmax`/`T.reduce_max` 的 dtype 表**不含 bf16**
  （`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1、`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1），
  故 bf16 输入必须经 `T.vcast` 上转 fp32 后计算——与「fp32 比较域」语义一致（§0.6 R2），无额外代价。
- **`T.reduce_min` 无独立文档页**：其存在性以 `docs/Tilelang.language/规约操作/T.reduce.md` 条目
  及本仓多个已验证 kernel 的实测用法佐证（`examples/TileOPs/tileops/kernels/pool/max_pool2d/max_pool2d_kernel/_max_pool2d_kernel.py`、
  `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py` 等）；
  属「文档缺失但实测可用」，列入 §9.4 待确认项。

**设计期估算（D-2 roofline）** —— 协议见 `_shared/standards/hardware-cost-model.md` §2，
常数引用 `pattern-library/constants.md`（`CONST-mte2-degradation` HBM 3:1 混合读写地板 1.26 TB/s，
版本戳 `tilelang 0.1.2+15ad002b3d / 2026-09-21`；`CONST-vector-launch-overhead` 0.5µs/op，
版本戳 `tilelang 0.1.2+3a214cde / 2026-09-09`；`CONST-capacity-910B2C` 192 KB/AIV，
版本戳 `tilelang 0.1.2+1990aa9fe4 / 2026-09-17`）：

以 dispatch 代表 workload 计算（`CH`/`BL` 按 §4.5 容量档）：

| workload | 流量项（`bytes ÷ 有效带宽`） | 发射项（向量 op 数） | 容量项 | 估算下界 |
|----------|------------------------------|----------------------|--------|---------|
| `smoke`(1,8,32, k3s2p1) | 768 B ÷ 1.26 TB/s = **0.6 ns** | 1 单元 × 108 op = **108 op** | 0 | **launch 主导**（见下注） |
| `sincnet-speaker-local`(32,80,4096, k3s3p0) | 27 960 320 B ÷ 1.26 TB/s = **22.2 µs** | 3 392 单元 × 168 op = **0.57 M op**（CH=64） | 0 | **≈ 22.2 µs**（流量主导；发射项不确定） |
| `textcnn-global`(64,256,128, k128s1p0 → **C4**) | 4 227 072 B ÷ 1.26 TB/s = **3.35 µs** | 256 单元 × 13 op = **3 328 op**（CH=64） | 0 | **≈ 3.35 µs** |
| `ecg-cnn-dilated`(16,128,2048, k5s2p2 d2 ceil) | 12 578 816 B ÷ 1.26 TB/s = **9.98 µs** | 1 056 单元 × 322 op = **0.34 M op**（CH=64） | 0 | **≈ 9.98 µs** |

> **估算下界（dispatch 代表 = `sincnet-speaker-local`）: ≈ 22.2 µs**（流量 22.2 + 发射「未文档化假设」 + 容量 0；
> 引用 `CONST-mte2-degradation` / `CONST-capacity-910B2C`）。**端到端实测地板大概率由固定开销主导**
> （NPU kernel launch + Event 计时 ~50–60 µs 量级，见 `examples/TileOPs/docs/gpu_to_npu_adaptation.md` C12 行）。
> **发射项口径注（未文档化假设）**：`CONST-vector-launch-overhead`（~0.5µs/op）是 **bn=512 宽块大 op** 的
> 实测值，对本设计 ≤128 元素的窄向量 op（本算子 op 宽度 = `CH` = 8/64）**不能直接外推**
> （按 0.5µs/op 会得荒谬量级）。窄向量 op 的每-op 代价**未逐点实测**，标注「未文档化假设 + 估算依据」；
> **方向证据已有实测**：`layout.md` 的 `PL-1.4-tiling-heuristic` 同族池化实测
> 「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs」（更宽 C 向量更优），与 `CONST-vector-launch-overhead`
> 「宽块优于窄块 op 翻倍」方向一致。**本算子的 op 数（≈ `L_out·k_W·2/CH` 量级）与同族 2D 算子同阶**，
> 故发射项**可能成为 `sincnet` / `ecg` 的实际主导项** —— 列入 §9 风险点与 §1.6.3 裁决计划。
> **容量项为 0**：本设计无 `store_fixpipe` 类跨引擎中间矩阵往返（纯 Vector 单 kernel）。
> **设备差异注**：`CONST-mte2-degradation` / `CONST-vector-launch-overhead` / `CONST-capacity-910B2C`
> 均绑定 **Ascend910B2C**（48 Vector 核）；本设备为 **Ascend910B3**（实查 20 × 2 = 40）。带宽常数按
> 「同代同 HBM 的量级下界」引用，**跨型号复用属未实测假设**（§9 风险点）。**版本戳注**：本任务
> toolchain stamp 为 `417dae9`，与上述常数条目记录的版本不同 —— 跨 stamp 复用属未实测假设（§9.4）。

**调研结论**：选定算法族 = **「直接窗口取最大值 + fp32 比较域 + C 轴向量化 + 显式块级向量 eq 掩码
NaN 追踪 + 核内融合转置（NCL→C 最内）+ L 轴连续 span 载入分块 + 在线单遍」**（R3 表候选 #0'；
另采纳候选 #6 退化特化 → §1.6.1 **C4/C4b**，候选 #4 融合扫描、候选 #5 span 载入削减并入 #0'；
三项采纳与 §1.6.1 优化结论**逐项一致**）。
关键依据：R3 表显示 #0' 的 FLOPs（= `N·C·L_out·k_W`，与 manifest `roofline.flops` 逐字一致）与
访存量（= manifest `roofline.bytes` + halo 放大）均为各候选最小量级（候选 1/2/3 数学不等价）；
R4 表显示 #0' 唯一同时满足「Vector 计算单元匹配 + C 轴向量整除 + 全静态框界 + 无跨核归约 + 容量充裕」。
**与基线的结构差异一句话**：数学公式、取值集合与 NaN 判决完全不变，执行单元由「每输出一线程的标量
比较链 + 逐元素 GM gather」改为「`(n,o_blk,c_block)` 单元的 C 轴向量比较器 + 连续 span 载入 + 块级
向量 eq 掩码 NaN 追踪器」。
**调研范围声明**（确有两项结构特化被采纳，但**无更低复杂度算法族**）：
本调研覆盖 ① `algorithm-candidates.md` 的 `ALG-pooling` 命中行全部候选
（R1 常数权重卷积 / sum-pool×常数 / R2 滑窗流式 / R3 窗口重叠重复读 / R4 C 轴向量轴）——
并**逐项判定其对 max 的适用性**（R1 命中项对 max 因非线性不成立）；
② 恒等变形空间（前缀最大值差分、可分离 1D→无、归约融合、span 载入访存削减）；
③ pattern-library 同类实现（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum` / `PL-1.3-host-permute` /
`PL-1.4-tiling-heuristic` / `PL-1.22-maxreduce-nan-tracking`）；
④ 本仓 `examples/` 同类实现（§3.5.3）与**同族前作**（`examples/max_pool2d/_max_pool2d_kernel/DESIGN.md`
等，CASE-pool-maxpool2d-migration）。
**互联网检索未使用**（本算子算法族在本地参考表 + 源码 + pattern-library 已完整覆盖，属简单窗口 / 归约类，
无需联网补充）。
**下游约束**：§1.4 算法描述、§1.6.1–§1.6.3 的分析对象 = 选定算法 #0'；§0.5/§0.6 的处置与之**一致**
（§0.6 R2 = #0' 的归约向量化 + NaN 向量追踪、§0.6 R3 = C 轴布局 + span 载入、§0.6 R4 = 候选 #6）；
源算法 §0.4 已识别优化手段的意图承接逐项见 §0.6（**无静默丢弃**：11 项全部有处置，
#3/#4/#5/#6/#7/#11 的意图分别由 §0.6 R1/R2/R3 承接）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的算法 #0'（= 基线结构 + C 轴向量化 + 块级向量 eq 掩码 NaN 追踪 + L 轴 span 载入分块）。
> R1 表中初判可采纳的项在此完成四要素论证。

**语义等价性锚点 E1（基线公式的 NPU 表述）**：源以「越界 lane 跳过」表达 padding；
NPU 表述为「累加器初值 $-\infty$ + 仅 fold in-bounds lane」。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **E1** | 语义等价锚点：越界跳过 ≡ $-\infty$ padding + 累加器初值 $-\infty$ | `if in_bounds: max_val = max(max_val, val)`，`max_val` 初值 $-\infty$ | `acc ← -inf`；仅对 in-bounds lane 做 `acc ← max(acc, …)` | **数学恒等**：$\max(-\infty,x)=x$，跳过 lane 与 fold 一个 $-\infty$ 逐位等价；已由 `verify_equiv.py` E1 组（vs `F.max_pool1d`，fp64）机器验证为**逐位一致**（7 配置 × 3 dtype） | 无（表达式改写，非降本项）——作为后续优化项的语义锚 |

**采纳的数学等价优化项**：

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **C1** | 连续 span 载入的**钳制等价改写**（替代逐元素 GM gather 的地址计算） | 逐元素读 `x[batch,c,iw]`，判据 `0<=iw<L_in`（每输出元素 `k_W` 次） | 载入连续 span `[s0, s0+LSPAN)`（`s0=max(0,min(base, L_in-LSPAN))`，`LSPAN=min(span,L_in)`，`base=ow0*sW-pW`，`span=(BL-1)*sW+(k_W-1)*d_W+1`）；块内读 `j = iw - s0`，判据 `0<=iw<L_in` | **数学恒等**：对所有窗口内 `iw∈[base, base+span)`，`s0<=iw<s0+LSPAN ⟺ 0<=iw<L_in`（证：`s0=max(0,min(base,L_in-LSPAN))`，`iw<s0 ⇒ iw<0`；`iw≥s0+LSPAN ⇒ iw≥L_in`——因 `s0+LSPAN=min(base+LSPAN, L_in)≤L_in` 且 `iw<base+span`，`LSPAN=min(span,L_in)`）；且 `0≤s0`、`s0+LSPAN≤L_in` 保证载入恒在界内。已由 `verify_equiv.py` C1 组（3 种 `BL` × 7 配置 × 3 dtype）机器验证为**逐位一致**并逐 lane 断言下标落界 | GM 读次数由 `N·C·L_out·k_W`（逐元素）降至 `~N·C·L_in`（sincnet `k=s` = **1.009×**；ecg 重叠窗 BL=31 = **1.11×**）——读次数降 `k_W` 倍（sincnet 3×、ecg 5×、textcnn 128×） |
| **C4** | 退化长度结构特化：整线归约 | 通用窗口循环取最大值（窗口覆盖整线时仍逐窗口 fold） | 当 **`L_out==1` 且 `p_W==0` 且 `d_W==1` 且 `(k_W-1)·d_W+1 >= L_in`**（四条件）：按 `(CH, L_in)` 连续块搬运 + `T.vcast` fp32 + `T.reduce_max(dim=1)`；NaN 用 `vcmp(eq)`+`vcast`+`reduce_min` 掩码归约 + `vselect` 判决 | **数学恒等（条件成立时）**：`p_W=0` 且 `d_W=1` 时窗口 $=\{k_w\}$，覆盖长度条件保证 $\{0,\dots,k_W-1\}\supseteq[0,L_{in})$，故窗口**恰为** $[0,L_{in})$，`out[n,c,0] = max_{all i_w} x[n,c,:]`，取值集合完全相同 ⇒ **逐位等价**（max 只选择，无算术）。**`d_W==1` 项必不可少**（反例见 §0.6 R4；`verify_equiv.py` C4x 组 6 行机器锁住） | 每线 `L_out` 次 × `k_W` 次向量 `vmax` → **1 次向量归约** + 省去转置链。**注：manifest 的 `textcnn-global` 触发本项** |
| **C4b** | 退化长度结构特化：恒等拷贝 | 通用窗口循环（每窗口 max 单元素） | 当 **`k_W==1 and d_W==1 and p_W==0 and s_W==1`**：`out = x`（纯 UB 中转拷贝，同 dtype 不 cast） | **数学恒等（条件成立时）**：窗口恰为单元素 `{o_w}`，`max` 单元素 = 元素本身 ⇒ 逐元素恒等（NaN 也随元素原样搬出，语义一致） | 每线 `L_in` 次比较 + NaN 追踪（2 op/元素）→ **1 次拷贝**。**注：不被当前 manifest workloads 触发** |

> **无「倒数乘 / 归一化」优化空间**（显式结论）：与同族 avg 池化不同，max 池化**无除法 / 归一化**
> （结果即被选元素），故 avg 迁移的 C1（`sum/d → sum·(1/d)`）在本算子**不存在**——§1.6.1 无对应项，非遗漏。
> 亦**无高代价算子降代换**（无 exp / rsqrt 类）、**无多处出现的公共子表达式**（每元素仅参与一次比较；
> 唯一可 CSE 的是边界判据与 NaN 掩码——前者由 C1 的钳制载入消解、后者由 R2 的块级一次掩码消解，见上）。
> **1D 相对 2D 的守卫差异（显式结论）**：2D 有「H 判据每行一次外提」的 CSE（前作 C1）；1D **只有一条空间轴**，
> 逐 lane 判据 `0<=iw<L_in` 同时依赖 `(o_w, k_w)`，**不存在可外提的第二轴判据**——1D 的对应优化改由
> 「连续 span 的钳制载入（每单元 2 次标量钳制 + 静态长度载入）」承担（本表 C1）。这是 1D 与 2D
> 优化手段集合的本质差异，非遗漏。
> **候选 1/2/3 不采纳**：数学不等价（§1.6.0 R1）。

**等价性机器验证（D-1）**：采纳项与语义锚点全部由
`examples/max_pool1d/_max_pool1d_kernel/verify_equiv.py` 在 **fp64 参照**下对照验证（torch CPU）：

- **E1**：NPU 基线公式（`-inf` 初值 + 越界跳过）vs `F.max_pool1d`（判据 = 舍入到目标 dtype 后逐位一致）；
  含一条 **`ceil_mode` step-back** 用例（`(1,8,8)` k3s4p0 ceil → `out=2`，触发 helper 的末窗回退分支）
  与一条 `ecg` 长线实例（`(16,8,2048)` k5s2p2 d2 ceil）；
- **C1**：连续 span 钳制载入结构 vs 逐元素守卫基线（逐位一致；`BL ∈ {4,8,16}` 三档，逐 lane 断言
  `lo<=iw<hi ⟺ 0<=iw<L_in` 且缓冲下标 `iw-s0 ∈ [0,LSPAN)`）；
- **C4**：整线归约公式 vs 通用窗口基线（**在充分条件成立且输出非空**的 5 条配置上，逐位一致）；
- **C4x**（dilation 反例组）：`(1,8,5)` k2s1p0 d4 与 `(1,8,7)` k3s1p0 d3，均为 `out=1` 但 **`d>1`，
  断言「不满足 C4 条件、必须走一般路径」**，并与 `F.max_pool1d` 逐位对照；末尾另有 HAZARD 演示行
  （`L=5,k=2,s=1,p=0,d=4,x[2]=1` → 一般路径 = 0.0 == golden，整线 = 1.0 被 `d==1` 守卫排除）；
- **C4b**：恒等拷贝公式 vs 通用窗口基线（充分条件成立时逐位一致）；
- 用例：每 case-group × 6 种数据（`rand / nan / inf（含 ±inf） / subnormal / large / zeros`）× 3 dtype
  （fp16/bf16/fp32）；NaN-vs-NaN 计相等，NaN-vs-finite 恒判失败。
  **空真防护**：`mismatch()` 断言两侧 `numel() > 0` 且 shape 相同；C4 组 5 条配置全部 `out==1`（非空）。
- **运行结论**：`viol=0` 全部 **117** 个 case-group → `[EQUIV] ALL PASS — 117 case-groups, 0 failures`
  （分组计数：E1 21 / C1 63 / C4 15 / C4x 6 / C4b 12）。

内嵌结果表（**由脚本 stdout 逐行原样粘贴**；`worst|d|` = 缩放后最大绝对差在「不一致元素」上的上界；
`viol` = 不一致元素数）：

```
check dtype     shape            kW   kinds  worst|d|    viol   status
--------------------------------------------------------------------------------------------
E1    float16   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
E1    float16   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
E1    float32   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
E1    float16   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
E1    float16   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
E1    float16   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
E1    float16   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
E1    float32   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
E1    float16   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 32)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (2, 8, 64)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 4, 63)       3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 4, 97)       5    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (16, 8, 2048)    5    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 8)        3    6      0.0000e+00  0      EQUIV_PASS
C4    float16   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C4    float32   (1, 8, 128)      128  6      0.0000e+00  0      EQUIV_PASS
C4    float16   (1, 8, 32)       32   6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (1, 8, 32)       32   6      0.0000e+00  0      EQUIV_PASS
C4    float32   (1, 8, 32)       32   6      0.0000e+00  0      EQUIV_PASS
C4    float16   (2, 4, 17)       17   6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (2, 4, 17)       17   6      0.0000e+00  0      EQUIV_PASS
C4    float32   (2, 4, 17)       17   6      0.0000e+00  0      EQUIV_PASS
C4    float16   (1, 16, 64)      64   6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (1, 16, 64)      64   6      0.0000e+00  0      EQUIV_PASS
C4    float32   (1, 16, 64)      64   6      0.0000e+00  0      EQUIV_PASS
C4    float16   (3, 8, 33)       33   6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (3, 8, 33)       33   6      0.0000e+00  0      EQUIV_PASS
C4    float32   (3, 8, 33)       33   6      0.0000e+00  0      EQUIV_PASS
C4x   float16   (1, 8, 5)        2    6      0.0000e+00  0      EQUIV_PASS
C4x   bfloat16  (1, 8, 5)        2    6      0.0000e+00  0      EQUIV_PASS
C4x   float32   (1, 8, 5)        2    6      0.0000e+00  0      EQUIV_PASS
C4x   float16   (1, 8, 7)        3    6      0.0000e+00  0      EQUIV_PASS
C4x   bfloat16  (1, 8, 7)        3    6      0.0000e+00  0      EQUIV_PASS
C4x   float32   (1, 8, 7)        3    6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (1, 8, 32)       1    6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (1, 8, 32)       1    6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (1, 8, 32)       1    6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (2, 4, 17)       1    6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (2, 4, 17)       1    6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (2, 4, 17)       1    6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (1, 16, 64)      1    6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (1, 16, 64)      1    6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (1, 16, 64)      1    6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (3, 33, 25)      1    6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (3, 33, 25)      1    6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (3, 33, 25)      1    6      0.0000e+00  0      EQUIV_PASS
```

> 完整逐条输出（含每 case-group 的 6 种数据 kind）可复现：
> `/workspace/venv/bin/python examples/max_pool1d/_max_pool1d_kernel/verify_equiv.py`
> （脚本随算子目录留存，Stage 2 维度 8 会重跑复核。）

**优化结论**：采纳 **3 项**——**C1**（连续 span 钳制载入 + 访存量削减，逐位等价）、**C4**（整线归约特化，
**四条件含 `d_W==1`** 成立时逐位等价，**manifest `textcnn-global` 触发**）、**C4b**（恒等拷贝特化，
充分条件成立时逐位等价），并建立 **E1** 语义锚点；三者与锚点均已机器验证（117 case-group 全 `EQUIV_PASS`）。
候选 1/2/3 因**数学不等价**不采纳；候选 7（可分离）在 1D 结构不适用；
「倒数乘」优化在本算子**不存在**（max 无归一化）。**优化后公式（供 §3.1 公式拆解使用）**：

```
一般网格 (非退化):
  base = ow0*sW - pW ; span = (BL-1)*sW + (k_W-1)*d_W + 1      # 编译期常量
  LSPAN = min(span, L_in) ; s0 = max(0, min(base, L_in-LSPAN))  # 每单元 2 次标量钳制 (C1)
  acc[owl,c] ← -inf ; flag[owl,c] ← 1 ; load row[0:LSPAN, c] ← x[n,c,s0:s0+LSPAN]  # 连续 span 载入
  eq[c] ← (row[j,c] == row[j,c]) ; eq_f32 ← cast(eq)            # 块级一次 NaN 掩码
  for kw in [0,k_W):
    for owl in [0,BL):
      iw = base + owl*sW + kw*dW ; j = iw - s0
      if 0 <= iw < L_in:                                        # 逐 lane 守卫 (C1 等价改写)
        acc[owl]  ← max(acc[owl], row_f32[j])                   # C 向量 max
        flag[owl] ← flag[owl] * eq_f32[j]                       # eq 掩码乘性 NaN 追踪
  out[n,c,ow0+owl] = (flag[owl] == 0) ? NaN : acc[owl]          # NaN 胜出判决
退化长度 (充分条件成立时):
  C4  (L_out==1, p_W==0, d_W==1, (k_W-1)*d_W+1 >= L_in):        # d_W==1 必要
        out[n,c,0] = max_{all i_w} x[n,c,:]                      # 同 NaN 判决 (reduce_min==0)
  C4b (k_W==1, d_W==1, p_W==0, s_W==1):
        out[n,c,:] = x[n,c,:]
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案（§1.4 + §3.3 伪代码 + §6）中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
> 替代方案所用 API 均以 `docs/Tilelang.language/` 条目佐证（下表「佐证」列）。

| # | 计算点 | 原实现形态（源 GPU） | 向量替代方案 | 是否替代 | 不可替代理由 / 佐证 |
|---|--------|---------------------|-------------|---------|--------------------|
| 1 | 累加器初始化 `acc ← -inf` | 源 `max_val = T.cast(-inf)` 标量（每输出元素 1 次） | `T.vbrc(-T.infinity("float32"), acc)`（**整 buffer** 广播 $-\infty$） | ✅ | 佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（scalar→任意 shape 广播）、`docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（fp16/fp32 ✓）。**dst 必须是整 buffer**——region 形态会被静默丢弃（§9.2） |
| 2 | NaN 标记初始化 `flag ← 1` | 源 `has_nan = False` | `T.vbrc(T.float32(1.0), flag)`（整 buffer）；标量 operand 必须是 **typed PrimExpr** | ✅ | 同上；裸 Python 数命中前端 rank 断言（§9.2） |
| 3 | 窗口取最大值 `max_val = T.max(max_val, val)` | 逐元素标量比较（每输出元素 `k_W` 条） | `T.vmax(acc[owl:owl+1,:], row_f32[j:j+1,:], acc[owl:owl+1,:])`（C 轴向量，原地） | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1（fp32 ✓；**bf16 ✗**，须先上转）；`PL-1.2-caxis-accum`（本仓实测切片累加形态） |
| 4 | 逐元素 `T.cast(x[...],"float")` | 每元素一次标量 cast | `T.vcast(row_dst, row_f32, round_mode="rint")`（整 tile，形状一致、源为 UB）；**fp32 输入改用 `T.copy`** | ✅ | 佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint` ✓）、§2.2.2/§2.3（形状一致 + 源须 UB）；同 dtype `vcast` 误编译见 §9.2 |
| 5 | **NaN 追踪** `has_nan \| T.isnan(val)` | 每元素标量 `isnan` + 标量 bool 或 | **块级一次**：`T.vcmp(row_f32, row_f32, eq_mask, "eq")`（$x{=}x$ ⟺ 非 NaN，向量 bool）+ `T.vcast(eq_mask, eq_f32, "rint")`；窗口内逐 (kw,owl) `T.vmul(flag[owl:owl+1,:], eq_f32[j:j+1,:], flag[owl:owl+1,:])` | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1（fp32 ✓，输出 bool 片上）+ `docs/Tilelang.language/条件操作/T.vselect.md` + `docs/Tilelang.language/数学操作/T.vmul.md` §2.3（支持原地）。**必须用 `eq`**（`ne` 为 ordered-NE，NaN 恒假，不可作 NaN 探测）。块级一次掩码 = `PL-1.22` ③' |
| 6 | 结果 `if_then_else(has_nan, NaN, max_val)` | 每元素标量分支 | `T.vcmp(flag, zero, cfinal, "eq")` + `T.vselect(cfinal, nanbuf, acc, out32)` | ✅ | 佐证同 #5；`nanbuf = inf - inf`（`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1 fp32 ✓、`docs/Tilelang.language/创建操作/T.infinity.md`） |
| 7 | 最终输出 cast | 每元素一次标量 cast | `T.vcast(out32, acc16, round_mode="rint")`（整 tile）；**fp32 输入改用 `T.copy`** | ✅ | 同上 #4 |
| 8 | 布局重排（C→最内） | （源无；GPU 靠 coalesced 尾轴） | `T.transpose(src, dst, [1,0])` | ✅ | 佐证：`docs/Tilelang.language/创建操作/T.transpose.md` §2.3（二轴交换 ✓，fp16/bf16/fp32 ✓）；`PL-1.1-transpose-chain` |
| 9 | **载入范围钳制 `base/span/LSPAN/s0/lo/hi`** | （源无；源直接逐元素算地址） | 无（保持每单元 2 次标量 `min`/`max` 钳制） | ❌ | **理由（block 级索引 / 任务映射标量）**：`base` 仅依赖单元 id，是**每单元 2 次**的标量整数运算（非逐元素热点）；无「对标量循环变量做 min/max 钳制」的向量 API（v-prefix 面向 buffer 元素）。**1D 特有**（2D 的 W 维不需载入钳制） |
| 10 | **窗口 `k_w` 循环 `for kw in T.serial(k_W)`** | `T.serial` 静态界 + 逐 lane 守卫 | 无（保持 `T.serial` + `if 0<=iw<L_in`） | ❌ | **理由（tile 级顺序依赖）**：`kw` 是累加链的循环携带维（跨 `kw` 累加到同一 `acc`/`flag`）；`k_W` 编译期常量故上界静态。**循环体内已全部向量化**（#3/#5），故该循环是 §6 的 block/tile 级调度循环，非逐元素标量循环 |
| 11 | **输出位置 `owl` 循环 `for owl in T.serial(BL)`** | `T.serial` 静态界 + 逐 lane 守卫 | 无（保持 `T.serial` + 运行时 `acc`/`flag` 行索引 `owl`） | ❌ | **理由（tile 级顺序依赖 + 运行时行索引）**：`owl` 是 `acc`/`flag` 的运行时行索引（`BL` 编译期常量，`owl` 取值运行时）；循环体为 C 向量 op（切片 `[owl:owl+1,:]`），与 `PL-1.2-caxis-accum` 同形 |
| 12 | **逐 lane 越界守卫 `0<=iw<L_in`** | 逐元素 in-bounds 谓词 | 候选替代：把界外缓冲 lane 预填 `-inf` 后**无条件** `vmax`（免守卫） | ❌ | **理由（动态偏移写入陷阱 + 无收益）**：预填需对缓冲的**运行时偏移区间**写 `-inf`（`[0, lo-base)` / `[hi-lo, LSPAN)`），属 `TRAP-UB-dynsubview-dominance` 的静态多缓冲 IR 风险域；且它把所有 lane 都做一次 `vmax`（op 数不减）。本设计选择「载入钳制保证读出下标恒落界 + 逐 lane 标量守卫」，守卫是**标量谓词门控一条向量 op**（非逐元素标量计算），§6 不出现逐元素标量循环 |
| 13 | 单元任务解码 `(n, o_blk, c_block)` | — | 无 | ❌ | **理由（block 级索引 / 任务映射计算）**：每单元少量整数 `%`/`//`，非逐元素热点；无对应逐元素向量 API |
| 14 | 输出长度 / `CH` / `BL` / `LSPAN` / `num_logical` / `num_kernels` / `num_local` | — | 无 | ❌ | **理由（host/trace 侧元数据）**：factory 层 Python 整数算术（编译期常量），不在 kernel 运行时逐元素执行 |
| 15 | host 侧参数规范化 / kernel 缓存 | — | 无 | ❌ | **理由（host 侧元数据）**：只改 shape/stride 视图与缓存键，符合 `ascend-constraints.md` §4；不在 kernel 内 |
| 16 | `cfinal` 与 `zero`/`nanbuf` 的单元级构造 | — | `T.clear(zero)`、`T.vbrc(inf, infbuf)`+`T.vsub(infbuf,infbuf,nanbuf)`（整 buffer op，非逐元素循环） | ✅ | 佐证：`docs/Tilelang.language/内存操作/T.clear.md` §2.2（fp32 ✓）；`T.vbrc` §2.2.2；`T.vsub` §2.3（原地） |
| 17 | C4 整线归约 | 逐窗口 fold | `T.reduce_max`（值）+ `T.vcmp(eq)+T.vcast+T.reduce_min`（NaN 掩码） | ✅ | 佐证：`docs/Tilelang.language/规约操作/T.reduce_max.md`（fp16/fp32 ✓，**bf16 ✗**）；`T.reduce_min` 见 §1.6.0 R4 注 |

**向量化结论**：逐元素计算（累加器/标记初始化 / 取最大值 / NaN 探测 / NaN 判决 / cast / 布局重排 /
单元级常量构造 / C4 归约）
**已全部向量化**（含 §3.3 单元入口的 `acc`/`flag` 初始化 → **#1/#2**），替代方案 API 均有 `docs/` 条目佐证。
保留 **5 类** 循环 / 标量点，理由类别：**#10/#11** tile 级顺序依赖（循环携带累加维 + 运行时行索引，
循环体内已向量化）/ **#12** 逐 lane 标量谓词门控向量 op（动态偏移预填有陷阱风险且无收益）/
**#9** block 级索引 / 任务映射标量（1D 特有钳制）/ **#13** block 级任务解码 / **#14/#15** host 侧 trace 元数据。
无「以实现简单为由保留标量循环」的情形。**§6 循环结构中不出现逐元素标量循环**（与本结论一致）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（3D NCL 进 / NCL 出），但**核内布局与 lane 映射是自由变量**。
> 本节枚举「窗口 / 池化类」必选候选（reference：algorithm-research.md Phase 2 第 3 项
> 「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴」）。
> 迁移任务：源的并行轴选择（GPU 逐元素 lane 扁平映射）是**输入而非结论**，对照 §0.3 独立评估。

**轴质量评分维度**：

| 评分项 | 说明 | 候选 1（NCL / L 轴） | 候选 2（核内 NLC / C 轴） | 候选 3（host permute / C 轴） |
|--------|------|----------------------|---------------------------|------------------------------|
| 整除性 | 轴长 vs 向量宽度（fp16/bf16 ×8、fp32 ×4） | `L_in`=32/128/2048/4096（多为 2 幂）但**含非 8 倍数**（如 63/97 测试形）；且**窗口沿 L 的步进 `s_W`=2/3 ≠ 1** | **C 恒为 8 的整数倍**（8/80/128/256；8 同是 fp32 ×4 的倍数）✓ | 同候选 2 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | 视 `L_in` 而定（非 8 倍数时浪费）；**窗口累加的跨步系数 `k_w·s_W`（`s_W≠1`）使向量化退化为标量** | **0%** | 0% |
| 累加链形态 | 该轴作串行累加维（stride≠1）还是向量维（stride=1） | L 轴作归约维需**跨步窗口最大值**（`iw=o_w·s_W-p_W+k_w·d_W`，`s_W≠1`）；`PL-1.2` 同族实测该形态 98.3% 标量 | C 轴作向量维（转置后 stride=1 连续）；空间轴（`k_w`）作串行归约维——`max` 结构天然 | 同候选 2 |
| repack 代价 | 布局重排代价 | 无 repack，但需跨步窗口处理 | 核内融合转置链（`T.transpose` 二轴交换，`PL-1.1` 实测 4 次合计 ~5.4µs） | **host permute 106–147µs**（`PL-1.3`，已量化证伪） |
| UB 容量影响 | 布局对驻留缓冲的影响 | 小（row `(LSPAN)` 级） | 代表档 ≈ 110 KiB 原始（§4.5），×1.7 后 ≈ 188 KiB，对照 192 KiB | 同候选 2 |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|----------------|---------|
| 1 | I/O 原生 NCL | L（行内连续） | 无 | 基线：窗口步进 `s_W`=2/3≠1，跨步窗口最大值需逐元素或 gather（`T.reduce_max` 不支持 bf16）；`L_in` 非 8 倍数时尾 lane 浪费 | ❌ 量化理由：`PL-1.2`/`PL-1.4` 同族实测「`sW≠1` 跨步系数阻碍向量化 → 98.3% 标量，原轴微调收益 0，换轴后 5.6x」；且 bf16 须先上转 |
| 2 | **核内 NLC（融合转置）** | **C（恒为 8 的整数倍）** | 每次 span `T.transpose(row_src, row_dst, [1,0])`（`PL-1.1` 族，实测 ~µs 级） | 尾 lane 0%；转置代价 ~µs 级（远低于算子本体 3–22µs）；UB 增量见 §4.5 | ✅ **选定** |
| 3 | host permute NCL↔NLC 进出 | C | 两次 host permute | **106–147µs**（`PL-1.3`，实测）> 多数 workload 本体耗时；host 侧连续化会产生新张量（无收益） | ❌ 量化理由：repack 代价（106–147µs）> 算子本体流量（3–22µs）数倍，净亏 |
| 4 | Cube / implicit-GEMM（0/1 权重矩阵） | Cube N 轴（`L_out`） | 权重矩阵构造 | **数学不可行**（max 非线性，无 0/1 权重矩阵可表达）；且 `L_out` 常 < 16 违 Cube 分形下限 | ❌ 见 §1.6.0 R1/R4（数学不等价优先于亲和评估） |
| 5 | **L 轴分块（BL）+ 连续 span 载入**（与本候选矩阵的轴选择正交，是必需的容量维度） | C（同上） | 无（只分块不改布局） | 使 `(CH, LSPAN)` 可驻留 UB（`L_in` 达 4096 时整线不可驻留）；halo 读放大 `LSPAN/(BL·sW)` | ✅ **必须**（与候选 2 组合；非独立布局候选） |

**布局决策结论**：选定 **方案 #2 + #5：核内布局 = span 内 NLC（C 最内连续）、向量化轴 = C、
repack 路径 = 核内 `T.transpose([1,0])`（每次 span 一次）+ 输出侧 `T.transpose([1,0])` 回转；
L 轴按输出块 `BL` 分块、每块一次钳制连续 span 载入**。
该结论已同步至：§3.3 伪代码（`row_src/row_dst/row_f32/eq_*/acc/flag` 形状与内层向量维 = C、
`owl` 为 acc 行索引）、§4 内存规划（§4.3/§4.5 的 buffer 形状与预算）、§6 循环结构（累加内层向量维 = C）。
弃选方案量化理由见候选矩阵 #1/#3/#4。**迁移任务注**：源 GPU 的并行轴是「输出元素扁平 lane」
（§0.3），与 NPU Vector 轴不对应，此处已独立评估并替换为 C 轴（§0.6 R1）。

**D-3 实验裁决模式（主选 + 备选 + 裁决计划）**：主选与备选的分歧落在**窄向量 op 的每-op 发射代价**
的**外推区间**——`CONST-vector-launch-overhead` 只给了 bn=512 大 op 的 0.5µs/op 实测值，
本设计 op 宽度 = `CH`（8/64），未逐点实测；且本算子每元素比同族 avg 多 1 条向量 op（NaN 追踪）、
`L_out` 大（sincnet 1365）使 op 数放大，该外推更敏感。
**方向证据**：`layout.md` 的 `PL-1.4-tiling-heuristic` 直接实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs
——更宽 C 向量 + 更少 block 胜过更高空间并行度」，与 `CONST-vector-launch-overhead` 方向一致；
主选「容量允许的最大 CH（CH_MAX=64）」落在 `PL-1.4` 已实测范围（CH16/32/64）内，故主选以实测条目为主证据。

- **主选（进 Stage 3）**：`CH =` 容量允许的最大 8 的倍数（`≤ min(C,64)`，判据见 §4.5），
  `BL =` 在该 `CH` 下满足容量不等式且 `≤ min(L_out,32)` 的最大整数；单元 = `(n, o_blk, c_block)`；
  `num_logical > vector_cores` 时 persistent（见 §5.5）。
  代表值：`smoke` CH=8/BL=16 → 1 单元（无适配）；`sincnet` CH=64/BL=26 → 3 392 单元（persistent，40 核 × 85 任务）；
  `textcnn`（C4）CH=64 → 256 单元（×7）；`ecg` CH=64/BL=31 → 1 056 单元（×27）。
  **依据**：`PL-1.4` 实测 CH64 为该池化族最优档（21.6µs，优于 CH32/CH16）；本设计 C 轴向量化形态同族。
- **备选（≤1 个，结构完整，可直接实现）**：**主选 CH 的一半 + BL 尽量放大**
  （sincnet → CH=32/BL=53；ecg → CH=32/BL=63），即把 op 数放大 2×、核数放大 2×，换取更少 halo 重读。
  这是「发射代价主导 vs 吞吐主导」的另一端；buffer 形状与主选相同仅 CH 减半，分核三要素按新 `num_logical`
  重算（sincnet CH=32/BL=53 → num_cblk=3, num_oblk=26, num_logical=2 496，persistent 40 核 × 63 任务）。
  （备选存在的意义**仅限于**：验证更窄的 CH 是否使趋势翻转——`PL-1.4` 未覆盖 CH<16。）
- **裁决计划**：代表 shape = `sincnet-speaker-local`、`ecg-cnn-dilated`（各 1 组）；指标 = msprof op Task Duration；
  测量未知常数 = 窄向量 op 的每-op 代价（用 `perf_opt` 下两种 CH/BL 的实测反解）；
  判定阈值 = **备选 latency < 主选 × (1 − 0.10) 才翻转**（10% 稳健性门槛）；
  裁决结果回写路径 = Stage 4 实测数据经附录补记固化 §1.6.3 判定依据，若翻转则按 `perf-feedback.md`
  受控路由回写 DESIGN.md。
- **shape 分派评估**：本 kernel 由 `functools.lru_cache` 按 `(n,c,l_in,k,s,p,d,ceil,dtype)` 特化，
  **支持按 shape 分派**主选/备选——裁决计划须评估「大 `L_in`（UB 受限，如 sincnet）走主选；
  小 `L_in` 走更大 BL」的**按负载分派**（而非全局二选一）。
- **BL 说明**：`BL` 取容量允许的最大值（非 1）：1D 的 halo 重叠使 `BL=1` 产生 `(k_W-1)·d_W+1` 倍的
  span 重读（ecg 达 4.5× GM 读），大 `BL` 把重读摊薄到 `≈1.1×`（与 2D 的 BH=1 结论**方向相反**——
  2D 的 `W_in` 整行已驻留、BH>1 只增容量；1D 的 `L` 轴必须分块，halo 成本随 `BL` 下降）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（迁移 prompt 指定；`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`）。

### 2.2 选型理由

- 算子是**纯 Vector**（无 matmul / 无 Cube），仅需逐元素向量运算（`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`/
  `T.vcast`）、广播（`T.vbrc`）、清屏（`T.clear`）、归约（`T.reduce_max`/`T.reduce_min`）与布局重排
  （`T.transpose`）——正是 Developer 模式的目标域
  （`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`、
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/` 均为 Developer）。
- 无跨引擎流水、无手动 L1/L0 分级需求（`T.alloc_ub` 直映 UB）；编译器管理 UB 分配与自动同步（§7）。
- 迁移 prompt 明确要求 developer 模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 环境变量 | `TILELANG_ASCEND_MODE=Developer` |
| 内存分配 | `T.alloc_ub(...)`（GM→UB 路径） |
| 计算方式 | v-prefix 向量 API（`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`/`T.vcast`/`T.vbrc`/`T.clear`）+ `T.reduce_max`/`T.reduce_min` + `T.transpose` + `T.copy` |
| 同步 | 编译器自动同步（§7）；无手动 `T.set_flag`/`T.wait_flag` |
| Kernel 维度 | 一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `L_out ← pool_output_dim(L_in,k_W,s_W,p_W,ceil_mode,d_W)` | trace 期 host 侧；**不得重算**（逐字引用源 helper） |
| 2 | `base ← o_blk·BL·s_W - p_W`；`span ← (BL-1)·s_W+(k_W-1)·d_W+1`；`LSPAN ← min(span,L_in)`；`s0 ← max(0,min(base, L_in−LSPAN))` | 每单元 2 次标量钳制（C1） |
| 3 | `row_src[c, j] ← x[n, c0+c, s0+j]`，`j∈[0,LSPAN)` | 连续 span 载入 `(real_c, LSPAN)` |
| 4 | `row_dst[j, c] ← row_src[c, j]` | 核内二轴转置 → C 最内 |
| 5 | `row_f32[j, c] ← cast_f32(row_dst[j, c])` | 整 tile fp32 上转（`rint`） |
| 6 | `eq_f32[j, c] ← cast_f32(row_f32[j, c] == row_f32[j, c])` | 块级一次 NaN 掩码（`x=x` ⟺ 非 NaN） |
| 7 | `acc[owl, c] ← max(acc[owl, c], row_f32[j, c])`，`j = base+owl·s_W+k_w·d_W − s0` | C 轴向量取最大值（逐 lane 守卫后） |
| 8 | `flag[owl, c] ← flag[owl, c] · eq_f32[j, c]` | NaN 乘性累积 |
| 9 | `cfinal[owl,c] ← (flag == 0)`；`out32[owl,c] ← cfinal ? NaN : acc[owl,c]` | NaN 胜出判决 |
| 10 | `acc_t[c, owl] ← cast_dtype(out32[owl, c])` | 单次舍回 + 转置回转 |
| 11 | `out[n, c0+c, o_blk·BL+owl] ← acc_t[c, owl]` | span 写回 `(real_c, BL_real)` |
| C4 | `out[n,c,0] ← max_{i_w} x[n,c,i_w]`（NaN 掩码归约判决） | 充分条件成立时的整线归约（manifest `textcnn-global` 触发） |
| C4b | `out[n,c,i_w] ← x[n,c,i_w]` | 充分条件成立时的恒等拷贝 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | `acc ← -inf` | `T.vbrc(src, dst)` | `src = -T.infinity("float32")`（typed 标量），`dst = acc`（**整 buffer**） | Developer |
| 0b | `flag ← 1` | `T.vbrc(src, dst)` | `src = T.float32(1.0)`（typed 标量），`dst = flag`（**整 buffer**） | Developer |
| 0c | `zero ← 0` / `nanbuf ← NaN` | `T.clear(buf)` / `T.vbrc`+`T.vsub` | `T.clear(zero)`；`T.vbrc(T.infinity("float32"), infbuf)` + `T.vsub(infbuf, infbuf, nanbuf)`（`inf-inf = NaN`） | Developer |
| 3 | span GM→UB | `T.copy(src, dst)` | `src = x[n, c0:c0+real_c, s0:s0+LSPAN]`，`dst = row_src[0:real_c, 0:LSPAN]` | Developer |
| 4 | 二轴转置 | `T.transpose(src, dst, permutation)` | `permutation=[1,0]` | Developer |
| 5 | fp32 上转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"`（同形、源为 UB）；**fp32 输入改 `T.copy`** | Developer |
| 6 | NaN 探测（IEEE eq，块级一次） | `T.vcmp(A, B, C, cmp_mod)` + `T.vcast` | `A=B=row_f32`，`C=eq_mask`（bool 片上），`cmp_mod="eq"`；`T.vcast(eq_mask, eq_f32, "rint")`（i1→f32 ✓） | Developer |
| 7 | C 轴取最大值 | `T.vmax(src0, src1, dst)` | 切片 `(1,CH)`：`src0=dst=acc[owl:owl+1,:]`，`src1=row_f32[j:j+1,:]`（原地）；**fp32 ✓，bf16 须先上转** | Developer |
| 8 | NaN 累积 | `T.vmul(src1, src2, dst)` | `src1=dst=flag[owl:owl+1,:]`，`src2=eq_f32[j:j+1,:]`（原地） | Developer |
| 9a | has_nan 判决 | `T.vcmp` | `A=flag`，`B=zero`，`C=cfinal`，`cmp_mod="eq"` | Developer |
| 9b | `has_nan ? NaN : max` | `T.vselect(Cond, A, B, Out)` | `Cond=cfinal`，`A=nanbuf`，`B=acc`，`Out=out32` | Developer |
| 10a | dtype 回转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"`（fp32 输入改 `T.copy`） | Developer |
| 10b | 输出转置回转 | `T.transpose(src, dst, [1,0])` | `(BL,CH)→(CH,BL)` | Developer |
| 11 | span UB→GM | `T.copy(src, dst)` | `src = acc_t[0:real_c, 0:BL_real]`，`dst = out[n, c0:c0+real_c, ow0:ow0+BL_real]` | Developer |
| C4 | 整线取最大值 | `T.reduce_max(buffer, out, dim)` | `(CH,L_in)→(CH,1)`（`dim=1`，`clear=True`） | Developer |
| C4 | NaN 掩码归约 | `T.vcmp`+`T.vcast`+`T.reduce_min` | `mb=vcmp(eq)`；`mf=vcast(i1→f32,rint)`；`reduce_min(dim=1)` → `nmin ∈ {0,1}`（**0 ⟺ 存在 NaN**） | Developer |
| C4 | NaN 判决 | `T.vcmp`+`T.vselect` | `cond=vcmp(nmin,zero,"eq")`；`out32=vselect(cond,nanbuf,gmax)` | Developer |
| C4b | 恒等拷贝 | `T.copy` | `x[n, c0:c0+real_c, 0:L_in] → stage → out[n, c0:c0+real_c, 0:L_in]` | Developer |
| 尺寸 | `L_out` | Python `pool_output_dim`（host/trace） | 逐字复用源 helper | host |
| 分块 | `CH` / `BL` / `LSPAN` | factory 层容量判据（§4.5） | `CH = max 8 的倍数 ≤ min(C,64)`；`BL = max 整数 ≤ min(L_out,32) s.t. 预算 ≤ 192KB` | host |

### 3.3 计算伪代码

> 注（迁移模板要求）：`T.Tensor` 的数据类型以**第二个位置参数**传入，不用 `dtype=` 关键字。
> 注：`block_m` 为源接口兼容参数（K9 保留），NPU tiling 由 shape 派生，其值被校验后忽略。

```python
# 迁移决策后的 NPU 算法伪代码（Developer 模式；C 轴向量化 + span 载入 + eq 掩码 NaN 追踪）
@tilelang.jit(out_idx=[1], target="npuir")     # 注：本分支 jit 不接受 compile_flags（§9.2）
def _build(block_m=None):
    # 退化特化编译期谓词（纯名 Python bool，trace 时折叠；二者互斥且均须过 §4.5 容量判据）
    # 注意：IS_GLOBAL 的 d_W==1 是必要条件——窗口输入覆盖集合 = {k_w·d_W}，
    # d_W>1 时不等于整线（§0.6 R4 反例；缺此项会让 (1,8,5) k2s1p0 d4 这类
    # out=1 配置静默走整线归约而给出错值）。
    IS_IDENTITY = (k_W == 1 and d_W == 1 and p_W == 0 and s_W == 1)
    IS_GLOBAL = (L_out == 1 and p_W == 0 and d_W == 1
                 and (k_W - 1) * d_W + 1 >= L_in)
    @T.prim_func
    def main(
        x:   T.Tensor((N, C, L_in), dtype),        # dtype 位置参数（非 dtype=）
        out: T.Tensor((N, C, L_out), dtype),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # ---- C4b 恒等（k==1 ∧ d==1 ∧ p==0 ∧ s==1；单元 = 整张 (n, c_block) 线）----
            if IS_IDENTITY:
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb * CH)
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:L_in],
                               stage[0:real_c, 0:L_in])
                        T.copy(stage[0:real_c, 0:L_in],
                               out[nb, cb*CH:cb*CH+real_c, 0:L_in])
            # ---- C4 整线归约（四条件含 d_W==1；单元 = 整张 (n, c_block) 线）----
            elif IS_GLOBAL:
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb * CH)
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:L_in],
                               flat[0:real_c, 0:L_in])
                        if is_f32:  T.copy(flat, flat32)          # 同 dtype 不用 vcast（§9.2）
                        else:       T.vcast(flat, flat32, round_mode="rint")
                        T.reduce_max(flat32, gmax, dim=1, clear=True)   # (CH,L_in)->(CH,1)
                        T.vcmp(flat32, flat32, mb, "eq")                # 非 NaN 掩码(bool, 片上)
                        T.vcast(mb, mf32, round_mode="rint")            # i1 -> f32 {0,1}
                        T.reduce_min(mf32, nmin, dim=1, clear=True)     # 0 iff 线上有 NaN
                        T.clear(zero)                                   # (CH,1) f32 = 0
                        T.vcmp(nmin, zero, cond, "eq")                  # 0 即 has_nan
                        T.vbrc(T.infinity("float32"), infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)                  # inf - inf = NaN
                        T.vselect(cond, nanbuf, gmax, out32)            # has_nan ? NaN : max
                        if is_f32:  T.copy(out32, gout)
                        else:       T.vcast(out32, gout, round_mode="rint")
                        T.copy(gout[0:real_c, 0:1], out[nb, cb*CH:cb*CH+real_c, 0:1])
            # ---- 一般网格（单元 = (n, o_blk, c_block)）----
            else:
                acc   = T.alloc_ub((BL, CH), "float32")    # 最大值累加器（向量轴 = C）
                flag  = T.alloc_ub((BL, CH), "float32")    # NaN 乘性标记（初值 1）
                zeros = T.alloc_ub((BL, CH), "float32")    # 0 常量
                infbuf= T.alloc_ub((BL, CH), "float32")    # +inf 常量
                nanbuf= T.alloc_ub((BL, CH), "float32")    # NaN 常量
                out32 = T.alloc_ub((BL, CH), "float32")
                cfinal= T.alloc_ub((BL, CH), "bool")       # has_nan 判决条件（片上）
                row_src = T.alloc_ub((CH, LSPAN), dtype)
                row_dst = T.alloc_ub((LSPAN, CH), dtype)
                row_f32 = T.alloc_ub((LSPAN, CH), "float32")
                eq_mask = T.alloc_ub((LSPAN, CH), "bool")  # NaN 掩码（片上）
                eq_f32  = T.alloc_ub((LSPAN, CH), "float32")
                acc16 = T.alloc_ub((BL, CH), dtype)
                acc_t = T.alloc_ub((CH, BL), dtype)

                for t in T.serial(num_local):              # persistent（静态界）
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb  = uid // (num_oblk * num_cblk)
                        r   = uid %  (num_oblk * num_cblk)
                        oblk= r // num_cblk
                        cb  = r %  num_cblk
                        real_c = T.min(CH, C - cb * CH)
                        ow0 = oblk * BL
                        base = ow0 * s_W - p_W
                        s0 = T.max(0, T.min(base, L_in - LSPAN))   # 钳制载入起点
                        T.vbrc(-T.infinity("float32"), acc)        # acc <- -inf（整 buffer）
                        T.vbrc(T.float32(1.0), flag)               # flag <- 1（整 buffer）
                        T.clear(zeros)
                        T.vbrc(T.infinity("float32"), infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)             # NaN
                        T.copy(x[nb, cb*CH:cb*CH+real_c, s0:s0+LSPAN],
                               row_src[0:real_c, 0:LSPAN])         # 连续 span 载入（静态长度）
                        T.transpose(row_src, row_dst, permutation=[1, 0])
                        if is_f32:  T.copy(row_dst, row_f32)
                        else:       T.vcast(row_dst, row_f32, round_mode="rint")
                        T.vcmp(row_f32, row_f32, eq_mask, "eq")    # 块级一次 NaN 探针
                        T.vcast(eq_mask, eq_f32, round_mode="rint")
                        for kw in T.serial(k_W):                   # 静态界（k_W 编译期常量）
                            for owl in T.serial(BL):               # 静态界（BL 编译期常量）
                                iw = base + owl * s_W + kw * d_W
                                j  = iw - s0
                                if 0 <= iw and iw < L_in:          # 逐 lane 守卫
                                    T.vmax(acc[owl:owl+1, 0:CH],
                                           row_f32[j:j+1, 0:CH],
                                           acc[owl:owl+1, 0:CH])
                                    T.vmul(flag[owl:owl+1, 0:CH],
                                           eq_f32[j:j+1, 0:CH],
                                           flag[owl:owl+1, 0:CH])
                        T.vcmp(flag, zeros, cfinal, "eq")          # flag==0 <=> 含 NaN
                        T.vselect(cfinal, nanbuf, acc, out32)      # has_nan ? NaN : max
                        if is_f32:  T.copy(out32, acc16)
                        else:       T.vcast(out32, acc16, round_mode="rint")
                        BL_real = T.min(BL, L_out - ow0)           # 输出块尾
                        T.transpose(acc16, acc_t, permutation=[1, 0])
                        T.copy(acc_t[0:real_c, 0:BL_real],
                               out[nb, cb*CH:cb*CH+real_c, ow0:ow0+BL_real])
    return main
```

（伪代码为**决策示意**，Stage 3 可细化局部次序与 buffer 复用；但「向量轴 = C」「核内二轴转置」
「span 钳制载入」「块级 eq 掩码乘性 NaN 追踪 + `has_nan ? NaN : max`」
「C4/C4b 充分条件（C4 含 `d_W==1`）」是冻结决策。
**实现次序约束**：`T.vcmp(eq)`+`T.vcast` 必须在 `T.vmul(flag, …)` **之前**（掩码当轮生成当轮消费）；
`T.vcmp(flag, zeros)` 必须在**全部** `vmax`/`vmul` 之后；C4 分支的 NaN 判决必须用 `reduce_min`
（`reduce_max + ==0` 只在全 NaN 成立）。次序/算子错误会导致判据恒真/恒假，
`verify_equiv.py` 只验公式不验 op 次序，故 §9.2 列入实现级检查项。）

### 3.4 API 可行性确认

| API | 来源确认 | 是否验证 |
|-----|---------|---------|
| `T.Kernel(num, is_npu=True) as (cid, _)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3（NPU ≤2 维） | ✓ 现有 examples 广泛使用 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` §2.2.2（1D~2D 充分验证；本设计用 1D/2D + C4/C4b 的 2D） | ✓（同族前作同款） |
| `T.vbrc`（scalar→任意 shape 广播） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 | ✓ |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` §2.2（fp32 ✓，1~5 维） | ✓ |
| `T.copy`（切片 GM↔UB） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2 | ✓ |
| `T.transpose`（二轴交换，fp16/bf16/fp32/int32） | `docs/Tilelang.language/创建操作/T.transpose.md` §2.3 | ✓ |
| `T.vcast`（f16/bf16→f32 `rint`；f32→f16/bf16 `rint`；i1→f32 `rint`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1/§2.2.2/§2.3 | ✓（**同 dtype 禁用**，§9.2） |
| `T.vmax`（int16/int32/int64/fp16/fp32；**不含 bf16**；`dst` 可原地） | `docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 | ✓ |
| `T.vcmp`（fp16/fp32/bf16；输出 bool 片上；`eq` 模式 IEEE 正确） | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1/§2.3 | ✓ |
| `T.vselect`（fp16/fp32/bf16；bool 条件） | `docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1 | ✓ |
| `T.vmul`（fp16/fp32/int16/int32/int64；支持原地） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1/§2.3 | ✓ |
| `T.vsub`（fp32；标量/张量；原地） | `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1/§2.3 | ✓ |
| `T.infinity("float32")` | `docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（fp16/fp32 ✓；**bf16 ✗**） | ✓ |
| `T.reduce_max`（fp16/fp32 ✓；**不含 bf16**；`clear`） | `docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1/§2.1 | ✓ |
| `T.reduce_min`（掩码归约判决） | **无独立文档页**；见 `docs/Tilelang.language/规约操作/T.reduce.md` + 本仓已验证 kernel 实测用法（§1.6.0 R4 注、§9.4） | ✓（实测可用） |
| `T.min` / `T.max`（标量钳制） | `docs/Tilelang.language/比较操作/` 条目 | ✓（前端标量 min/max） |

> **dtype 约束提醒**：`T.vmax`/`T.reduce_max` 的 dtype 表**不含 bf16**，`T.vsub`/`T.vmul` 亦不含 bf16；
> `T.vcmp`/`T.vselect`/`T.transpose`/`T.vcast` 支持 bf16。故 **bf16 输入必须经 `T.vcast` 上转 fp32 后计算**
> ——与「fp32 比较域」语义一致（§0.6 R2），无额外代价（`nanbuf`/`zero`/`acc`/`flag` 均用 fp32）。
> **`bool`（i1）不可 load/store GM**：本设计的 `eq_mask`/`cfinal`/`mb` 均为**片上中间结果**，不写回 GM ✓。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（用一维 `T.Kernel(num_kernels)`；任务维度在 kernel 内解码） | 一维网格 |
| GPU 专用 API | 否（全部 v-prefix / `T.copy` / `T.transpose` / `T.reduce_*`，均为 NPU API） | — |
| GEMM 非整除 / L0C 溢出 | 否（无 Cube 计算，`L0C` 不适用） | — |
| 分核策略 | 涉及 | 见 §5.5（三要素齐全，persistent grid-stride） |
| host 侧禁止改输入内容 | 否（host 只做 shape/stride 元数据与缓存；无 padding / 无数据拷贝） | `ascend-constraints.md` §4 |
| 动态 `T.serial` 边界 | 否（`k_W`/`BL`/`LSPAN`/`L_out` 全为编译期常量，`T.serial` 上界天然静态） | 无需 adaptive 家族的 `max_bin_extent` 机制 |
| UB 容量 | 涉及（长 `L_in` 的 sincnet/ecg） | 容量驱动的 `CH`/`BL` 联合选择（§4.5 带字节数不等式） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 / 并行模型 | `T.Kernel(ceildiv(total_output,block_m), threads)` + `T.Parallel(block_m)`（线程 lane 级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1（`(n,o_blk,c_block)` 单元 + C 向量 lane） |
| 归约单元 | 每线程标量比较链（`T.alloc_var`） | UB fp32 向量比较器（C 轴） | §0.6 R2 |
| NaN 追踪 | 标量 bool `has_nan \| isnan(val)` | 块级 C 向量 eq 掩码乘性累积（`vcmp`+`vcast`+`vmul`） | §0.6 R2（语义不变） |
| 数据搬运 | 逐元素 GM gather | UB 连续 span 载入 + 核内二轴转置 | §0.6 R3 |
| L 轴分块 | 源无（逐元素未载入） | `BL` 输出块 + `LSPAN` 连续 span（`s0` 钳制） | §0.6 R1/R3（1D 必需，2D 的 `W_in` 整行可驻留） |
| 越界守卫 | 逐元素 in-bounds 谓词 | 载入钳制（每单元 2 次标量）+ 逐 lane 相对守卫 | §1.6.1 C1 |
| 结构特化 | 无 | C4/C4b（**校正充分条件**，容量守卫） | §0.6 R4 / §1.6.1 |
| `threads` 参数 | factory `(block_m, threads)`；`default_config={"block_m":256}` | **移除 `threads`**（NPU 无 CUDA 线程）；callable 只留 `block_m` | 脚手架已按 K9 移除（集成包 `max_pool1d.py` 第 68–95 行）；factory 签名改为 `_build(block_m)` |
| autotune | 源 `autotune_configs = [128,256,512]×[128,256,512]` | 移除（K8），启发式 config（容量驱动 CH/BL） | 集成包已移除 |
| 编译标记 | `compile_flags=["-O3","-DENABLE_BF16"]` | **本分支 jit 不接受 `compile_flags`**（§9.2），移除 | 同族前作 Stage 3 实测确认 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/max_pool2d/_max_pool2d_kernel/_max_pool2d_kernel.py` | **最高（同族窗口池化，CASE-pool-maxpool2d-migration，Stage 3 已过）** | Developer 一维 `T.Kernel(is_npu=True)`；`(n,o_h,c_block)` 单元解码；`T.vbrc`/`T.clear`/`T.vcast`/`T.vmax`/`T.vcmp`(eq)/`T.vmul`/`T.vselect`/`T.transpose` 用法；persistent 分核与 `real_c` 尾块；C4/C4b 特化与容量回退；fp32 同 dtype 用 `T.copy`；C4 NaN 用 `reduce_min` |
| `examples/max_pool2d/_max_pool2d_kernel/DESIGN.md` | **最高（同族设计文档，含 §1.6 全节与 69 组 D-1 验证）** | 窗口池化特化条件的正确推导（`d==1` 必要）、四口径复杂度表、D-3 裁决三件套、UB 逐 buffer 预算 |
| `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py` | 高（同族 max 变体实现，CASE-pool-adaptivemaxpool2d-migration） | max 的 fp32 比较域 + 显式 NaN 追踪；`T.vmax` + eq 掩码（`TRAP-vcmp-ne-ordered-nan` 的正确用法）；`reduce_min` 判决 |
| `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | 高（同族前作，CASE-pool-adaptiveavgpool2d-migration） | Developer 模式一维 `T.Kernel(is_npu=True)`；`CH=min(C,64)`；persistent 分核与 `real_c` 尾块；`T.vbrc`/`T.clear`/`T.transpose` 用法 |
| `examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d.py` | 高（本迁移的集成包 / host 层） | K5/K8/K9 适配形态；`block_m` 兼容参数；`pool_output_dim` 内联副本 |
| `examples/TileOPs/tests/ops/test_pool.py` | 高（规格/golden） | `MaxPool1dTest.ref_program = F.max_pool1d`，`atol=rtol=0`；`dilation != 1` 走 CPU golden |
| pattern-library `layout.md` 的 `PL-1.1`/`PL-1.2`/`PL-1.3`/`PL-1.4`/`PL-1.22` | 高（池化类实测模式） | 核内融合转置链、C 轴切片累加、host permute 证伪、tiling 启发式、max 归约 NaN 追踪形态 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N, C, L_in)` | float16 \| bfloat16 \| float32 | **恰好 3D NCL** 连续；尾轴 `L_in` 连续 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(N, C, L_out)` | same_as_input | NCL 连续；`out_idx=[1]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `row_src` | `(CH, LSPAN)` | dtype | UB | span GM→UB 暂存 |
| `row_dst` | `(LSPAN, CH)` | dtype | UB | 核内二轴转置输出（C 最内） |
| `row_f32` | `(LSPAN, CH)` | float32 | UB | fp32 上转 + 向量比较源 |
| `eq_mask` | `(LSPAN, CH)` | bool | UB | 块级 NaN 掩码（**片上**，不落 GM） |
| `eq_f32` | `(LSPAN, CH)` | float32 | UB | NaN 掩码（`{0,1}`，供 `vmul`） |
| `acc` | `(BL, CH)` | float32 | UB | C 轴向量最大值累加器（向量轴 = C，初值 $-\infty$） |
| `flag` | `(BL, CH)` | float32 | UB | C 轴 NaN 乘性标记（初值 1，0 吸收） |
| `zeros` / `infbuf` / `nanbuf` | `(BL, CH)` | float32 | UB | 0 常量 / +inf 常量 / NaN 常量（`inf-inf`） |
| `out32` | `(BL, CH)` | float32 | UB | NaN 判决结果（fp32） |
| `cfinal` | `(BL, CH)` | bool | UB | 窗口级 has_nan 判决条件（片上） |
| `acc16` / `acc_t` | `(BL, CH)` / `(CH, BL)` | dtype | UB | 舍回 + 转置回转 |
| `flat` / `flat32`（C4） | `(CH, L_in)` | dtype / float32 | UB | 整线归约退化路径 |
| `mb` / `mf32`（C4） | `(CH, L_in)` | bool / float32 | UB | C4 分支的 NaN 掩码（bool 片上）与 fp32 掩码 |
| `gmax` / `nmin` / `zero2` / `infbuf2` / `nanbuf2` / `out32b` / `cond` / `gout`（C4） | `(CH,1)` | float32 / bool / dtype | UB | 整线归约输出、NaN 掩码归约、判决与输出（**与一般路径的 `zeros`/`infbuf`/`nanbuf`/`out32` 同名不同 shape——分支互斥**） |
| `stage`（C4b） | `(CH, L_in)` | dtype | UB | 恒等退化路径中转 |

### 4.4 内存搬运路径

```
【一般网格】
GM[x] --T.copy span(real_c,LSPAN)--> UB[row_src] --T.transpose[1,0]--> UB[row_dst] --T.vcast(rint)--> UB[row_f32]
UB[row_f32] --T.vcmp(eq)+T.vcast--> UB[eq_mask/eq_f32]（块级一次）
UB[row_f32] --T.vmax(C 轴切片, kw serial × owl serial)--> UB[acc (BL,CH) f32]
UB[eq_f32]  --T.vmul 乘性累积--> UB[flag (BL,CH) f32]
UB[acc]/UB[flag] --T.vcmp(flag,zeros,"eq") + T.vselect(cfinal,nanbuf,acc)--> UB[out32]
                 --T.vcast(rint)--> UB[acc16] --T.transpose[1,0]--> UB[acc_t (CH,BL)] --T.copy span--> GM[out]
【C4 整线】GM[x] --T.copy 连续块(CH,L_in)--> UB[flat] --T.vcast--> UB[flat32] --T.reduce_max(dim=1)--> UB[gmax]
            UB[flat32] --T.vcmp(eq)+T.vcast+reduce_min--> UB[nmin] --T.vcmp(eq vs 0)--> cond --T.vselect(nanbuf,gmax)--> UB[out32] --T.vcast--> GM[out]
【C4b 恒等】GM[x] --T.copy 整线(CH,L_in)--> UB[stage] --T.copy--> GM[out]
（无 GM 中间往返、无 Cube / L1 / L0C 路径）
```

### 4.5 UB 内存预算

**一般路径逐 buffer 字节表（两个代表档；行字节 = 元素数 × 元素宽度）**：

| Buffer | shape（sincnet，CH=64，BL=26，LSPAN=78，fp16） | Bytes | shape（ecg，CH=64，BL=31，LSPAN=69，bf16） | Bytes |
|--------|------------------------------------------------|-------|-------------------------------------------|-------|
| `row_src` | `(64,78)` | 9 984 | `(64,69)` | 8 832 |
| `row_dst` | `(78,64)` | 9 984 | `(69,64)` | 8 832 |
| `row_f32` | `(78,64)` f32 | 19 968 | `(69,64)` f32 | 17 664 |
| `eq_mask` | `(78,64)` bool | 4 992 | `(69,64)` bool | 4 416 |
| `eq_f32` | `(78,64)` f32 | 19 968 | `(69,64)` f32 | 17 664 |
| `acc` | `(26,64)` f32 | 6 656 | `(31,64)` f32 | 7 936 |
| `flag` | `(26,64)` f32 | 6 656 | `(31,64)` f32 | 7 936 |
| `zeros` | `(26,64)` f32 | 6 656 | `(31,64)` f32 | 7 936 |
| `infbuf` | `(26,64)` f32 | 6 656 | `(31,64)` f32 | 7 936 |
| `nanbuf` | `(26,64)` f32 | 6 656 | `(31,64)` f32 | 7 936 |
| `out32` | `(26,64)` f32 | 6 656 | `(31,64)` f32 | 7 936 |
| `cfinal` | `(26,64)` bool | 1 664 | `(31,64)` bool | 1 984 |
| `acc16` | `(26,64)` fp16 | 3 328 | `(31,64)` bf16 | 3 968 |
| `acc_t` | `(64,26)` fp16 | 3 328 | `(64,31)` bf16 | 3 968 |
| **合计** | | **113 152 B ≈ 110.5 KiB** | | **114 944 B ≈ 112.3 KiB** |

**容量判据（带字节数的不等式，非代表 shape 外推）**：
`G(CH, BL, LSPAN, eb) × INFLATE ≤ 196 608 B (= 192 KiB)`，其中
`G = CH·LSPAN·(2·eb + 9) + BL·CH·(25 + 2·eb)`（逐 buffer 求和上式；`2·eb+9` = `row_src`+`row_dst`+`row_f32`+`eq_mask`+`eq_f32`；
`25+2·eb` = `acc`/`flag`/`zeros`/`infbuf`/`nanbuf`/`out32` 6×f32 + `cfinal` bool + `acc16`+`acc_t`），
`LSPAN = min((BL-1)·s_W + (k_W-1)·d_W + 1, L_in)`。
选型规则（**两维联合**）：`CH =` 满足不等式且 `≤ min(C,64)` 的**最大 8 的倍数**（`CH ≥ 8`；`C < 8` 时取 8 并以
`real_c` 收尾）；`BL =` 在该 `CH` 下满足不等式的**最大整数**，`BL ≤ min(L_out, 32)`；无解则 `CH` 折半重试。
`INFLATE` 双口径并列（`CONST-capacity-910B2C`）：**默认档（auto-multi-buffer 开）1.7**；关闭档 1.10–1.12。
设计按**默认档 1.7** 保守判据。**本小节字节口径统一为 KiB（1 KiB = 1024 B），并同时给出原始字节数以便复算**：

- `sincnet`（fp16）：`CH=64/BL=26` → `G = 113 152 B`，`×1.7 = 192 358 B = 187.8 KiB ≤ 192 KiB` ✓
  （`BL=27` → `LSPAN=81` → `G=117 504 B`，`×1.7 = 199 757 B ✗` ⇒ 容量判据选 BL=26）。
- `ecg`（bf16）：`CH=64/BL=31` → `G = 114 944 B`，`×1.7 = 195 405 B = 190.8 KiB ≤ 192 KiB` ✓
  （`BL=32` → `LSPAN=71` → `G=118 464 B`，`×1.7 = 201 389 B ✗` ⇒ 选 BL=31）。
- `smoke`（fp16，CH=8/BL=16/LSPAN=32）：`G = 7 040 B`，`×1.7 = 11 968 B ✓`；**fp32**（eb=4）：`G = 8 576 B`，
  `×1.7 = 14 579 B ✓`。
- 逐 buffer 求和、逐编译期分支复算（同族 Stage 2 教训：「代表 shape 结论 ≠ 整条特化路径结论」、
  漏计小 buffer 是低报高发区）——上表按 §3.3 的 `alloc` 清单一一对应。

**退化路径容量核算（独立于代表 shape；失败即回退一般路径，纯优化）**：依据
`_shared/standards/hardware-cost-model.md` §5 与同族三作教训（退化路径缓冲随 `L_in` 线性放大，
代表 shape 的容量结论**不覆盖**退化路径），逐条给不等式（KiB 口径同一般路径）：

- **C4**：`c4_bytes = CH·L_in·(eb + 9) + CH·(25 + eb) ≤ 196 608 / 1.7`，其中主项 = `flat`（`eb`）+
  `flat32`（4）+ `mb`（1）+ `mf32`（4）= `L_in` 方向每元素 `eb + 9`；小项 = `gmax`/`nmin`/`zero2`/`infbuf2`/
  `nanbuf2`/`out32b`（`(CH,1)` f32 ×6 → `24·CH`）、`cond`（`CH,1` bool → `CH`）、`gout`（`(CH,1)` dtype → `CH·eb`）。
  例：`textcnn-global` `(64,256,128)` k128s1p0、`CH=64`、`eb=2` → 主项 `64·128·11 = 90 112` +
  小项 `64·(25+2) = 1 728` = **91 840 B**，`×1.7 = 156 128 B = 152.5 KiB ✓`（**走 C4**）；
  `L_in=4096`（假想全局池化）`CH=64` → 主项 `64·4096·11 = 2 883 584 B` ≫ 上限 → **回退一般路径**。
- **C4b**：`c4b_bytes = CH·L_in·eb × 1.7 ≤ 196 608 B`。
  例：`(1,8,32)` k1s1p0 `CH=8` → `8·32·2 = 512 B` ✓；`(1,8,4096)` → `8·4096·2 = 65 536 B`，
  `×1.7 = 111 411 B = 108.8 KiB ✓`；`(1,8,16384)` → `8·16384·2 = 262 144 B`，`×1.7 = 445 645 B ✗ → 回退`。
- 回退后语义与精度**不变**（一般路径本就正确），仅放弃纯优化收益。

### 4.6 动态轴定义

**无动态轴**：`N, C, L_in, k_W, s_W, p_W, d_W, ceil_mode, L_out` 与派生的
`CH, BL, LSPAN, num_cblk, num_oblk, num_logical, num_kernels, num_local`
全部在 factory 构建期（trace 期）确定，kernel 由 `functools.lru_cache` 按该组合特化。
（`o_w`/`iw`/`j`/`s0`/`BASE` 判定在 kernel 内是运行时标量，但它们是**由编译期 shape 推出的运行时值**，
不是动态轴。）

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
def _max_pool1d_func(CH: int) -> ...      # 去掉源 GPU 的 threads 参数（K9）
```

> **注**：分支 jit **不接受 `compile_flags`**（源 GPU 的 `["-O3","-DENABLE_BF16"]` 须移除）——
> 同族前作 Stage 3 实测（`examples/max_pool2d/_max_pool2d_kernel/_max_pool2d_kernel.py` 第 52 行注释）；
> 列入 §9.2。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。

**判定依据**：算子仅含逐元素向量运算（`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`）、广播（`T.vbrc`）、
cast（`T.vcast`）、归约（`T.reduce_max`/`T.reduce_min`）与布局重排（`T.transpose`），无 `T.gemm` / matmul
→ 不需要 L1 + L0A/L0B/L0C（§1.6.0 R1/R4 已弃 Cube 路径——数学上不可表达）。
故核数按**纯 Vector 算子翻倍**规则取物理核数 ×2（§5.5）。

### 5.2 Block 划分

本算子的分块维度不是传统 GEMM 的 `(M,N)`，而是 **(通道块 CH) × (输出位置块 BL)** 的任务维度：

```python
CH = <满足 §4.5 容量判据的最大 8 的倍数，且 <= min(C, 64)；C < 8 时取 8>
BL = <在 CH 下满足 §4.5 不等式的最大整数，且 <= min(L_out, 32)>
LSPAN = min((BL - 1) * s_W + (k_W - 1) * d_W + 1, L_in)
num_cblk = ceil(C / CH)            # 通道块数
num_oblk = ceil(L_out / BL)        # 输出位置块数
# 逻辑单元数按编译期分支取（单元粒度是 num_logical 的唯一定义源，见 §5.5 ①）：
#   一般网格  单元 = (n, o_blk, c_block)  ->  num_logical = N * num_oblk * num_cblk
#   C4 整线   单元 = (n, c_block)          ->  num_logical = N * num_cblk（L_out==1 且 d==1，与上式同值）
#   C4b 恒等  单元 = (n, c_block)          ->  num_logical = N * num_cblk（整线；!= N*num_oblk*num_cblk）
# IS_GLOBAL / IS_IDENTITY 为 factory 编译期谓词（见 §3.3；IS_GLOBAL 含 d_W==1）：
#   d>1 的 out=1 配置（如 (1,8,5) k2s1p0 d4）IS_GLOBAL=False → 走一般网格分支
num_logical = N * num_cblk if (IS_GLOBAL or IS_IDENTITY) else N * num_oblk * num_cblk
```

- **`CH` 的选择理由**：① C 轴为向量轴，须为 8 的整数倍（fp16/bf16 向量宽 8、fp32 宽 4）；
  ② UB 预算上限（§4.5 判据；长 `L_in`/重叠窗时被容量压 BL）；
  ③ 与 §5.5 分核方案配合使 `num_logical` 落在物理核数附近/之上（persistent）。
- **`BL` 的选择理由**：① 1D 必需（`L_in` 达 4096，整线不可驻留）；② 越大 halo 重读越小
  （`LSPAN/(BL·s_W) → 1`，ecg BL=31 时 ≈1.11）；③ 受容量判据约束（§4.5），`BL ≤ min(L_out,32)`。
- **`num_logical` 必须与「一个逻辑单元处理多少数据」严格同粒度**：一般网格单元为 `(n, o_blk, c_block)`；
  退化路径（C4/C4b）单元为 `(n, c_block)` **整条线**。C4 因 `L_out==1` 两种口径同值；
  **唯一会分叉的是 C4b**（`out==in`，`num_oblk>1`）：通式给 `N*num_oblk*num_cblk`，整线单元只需
  `N*num_cblk`——若沿用通式会越界读写（同族前作 Stage 2 已实证）。本设计采用整线口径 `N*num_cblk`。

**代表性 shape 的分块结果**（`out` 由 `pool_output_dim` 给出）：

| workload | N | C | L_in | k/s/p/d | L_out | CH | BL | LSPAN | num_cblk | num_oblk | num_logical |
|----------|---|---|------|---------|-------|----|----|-------|----------|----------|-------------|
| `smoke-1x8x32-k3s2p1` | 1 | 8 | 32 | 3/2/1/1 | 16 | 8 | 16 | 32 | 1 | 1 | **1**（一般） |
| `sincnet-speaker-local` | 32 | 80 | 4096 | 3/3/0/1 | 1365 | 64 | 26 | 78 | 2 | 53 | **3 392**（一般，容量压 BL） |
| `textcnn-global` | 64 | 256 | 128 | 128/1/0/1 | 1 | 64 | — | — | 4 | — | **256**（**C4 整线**；`out=1 ∧ p=0 ∧ d=1 ∧ k>=in`） |
| `ecg-cnn-dilated` | 16 | 128 | 2048 | 5/2/2/2 ceil | 1023 | 64 | 31 | 69 | 2 | 33 | **1 056**（一般） |
| test `full-default-stride-dilation` | 1 | 4 | 63 | 3/3/1/2 | 21 | 8 | 21 | 59 | 1 | 1 | **1**（一般；C=4 → CH=8 收尾） |
| test `full-ceil-k5-s3-p2` | 1 | 4 | 97 | 5/3/2/1 ceil | 33 | 8 | 32 | 96 | 1 | 2 | **2**（一般） |
| C4 例 `(1,8,128)` k128s1p0 | 1 | 8 | 128 | 128/1/0/1 | 1 | 64 | — | — | 1 | — | **1**（C4；须 `out=1 ∧ p=0 ∧ d=1 ∧ k>=in`） |
| C4x 反例 `(1,8,5)` k2s1p0 d4 | 1 | 8 | 5 | 2/1/0/4 | 1 | 8 | 1 | 5 | 1 | 1 | **1**（**一般路径**，`d>1` 不走 C4） |
| C4b 例 `(1,8,32)` k1s1p0 d1 | 1 | 8 | 32 | 1/1/0/1 | 32 | 8 | 1 | 32 | 1 | 32 | **1**（C4b 整线） |

> **注意**：C4 被 manifest 的 `textcnn-global` 触发（1D 相对 2D 的差异）；C4b **不被当前 manifest workloads
> 触发**，属对 op 层一般参数空间的正确性/性能补全，容量受 §4.5 判据约束、失败即回退。
> C4x 反例行（`d>1`，`out=1`）**不走 C4**：因 `IS_GLOBAL` 的 `d_W==1` 条件不成立（§0.6 R4 反例；
> Stage 2 复核项），它按一般网格口径计 `num_logical = N × num_oblk × num_cblk = 1`。

### 5.3 约束分析

- **对齐约束**：向量轴 C 恒为 8 的整数倍（8 同是 fp32 向量宽 4 的倍数）；`CH` 取 8 的倍数 ✓。
  `L_in`/`LSPAN`（32/78/128/2048/4096）仅作转置后 row 长度，**不作向量轴**，不要求整除 ✓。
- **UB 容量**：代表档 ≈ 110.5 / 112.3 KiB（原始），默认档 ×1.7 → ≈ 187.8 / 190.8 KiB < 192 KiB ✓（§4.5；
  长 `L_in`/重叠窗被容量判据压 `BL`）。
- **L0 容量**：无 Cube 计算，不适用 ✓。
- **非整除**：本算子**不存在 GEMM 式整除问题**——span 长度用静态 `LSPAN`；任务维度用 persistent
  `T.serial` + `if uid < num_logical` 守卫（尾任务丢弃）；通道尾块用 `real_c = min(CH, C - cb*CH)`、
  输出块尾用 `BL_real = min(BL, L_out - ow0)` 动态 copy/写回范围，padded lane 永不写回 GM。
  窗口越界由 §1.6.1 C1 守卫处理（语义本来就要求）。

### 5.4 注意事项

- **尾单元**：`num_logical` 非 `num_kernels` 整数倍时（如 sincnet 3392 / 40 = 84.8），
  核内 `T.serial(num_local)` 的 `if uid < num_logical` 守卫丢弃越界任务。
  退化路径的 `num_logical` 按其整线单元粒度取（`N*num_cblk`）。
- **通道尾块**：`C % CH != 0`（如 C=80 → CH=64 + `real_c=16`；C=4 → CH=8 + `real_c=4`）——
  copy/写回用 `real_c` 动态范围；`row_src/row_dst/row_f32/eq_*` 的 padded lane 残留旧值，
  但 padded lane 的 `acc`/`flag` 永不写回 GM（各 lane 独立），无污染。
- **输出块尾**：`L_out % BL != 0`（如 sincnet 1365 = 26×52 + 13）——`BL_real` 动态写回范围。
- **载入钳制**：`s0 = max(0, min(base, L_in - LSPAN))` 保证 `0 ≤ s0` 且 `s0+LSPAN ≤ L_in`（见 §1.6.1 C1
  等价性证明）；`LSPAN ≤ L_in` 由定义保证。
- **窗口守卫**：`if 0 <= iw and iw < L_in`（逐 lane），`j = iw - s0` 恒落 `[0,LSPAN)`（C1 已机器断言）。
- **零输出保护**：`L_out > 0` 由 op 层保证（"output size must be greater than zero"）。
- **NaN 语义**：必须在所有输入（含 `±inf`/subnormal）下保持「NaN 胜出 + `±inf` 取 `+inf`」——
  由 `eq` 掩码 + `flag` 乘性累积保证（§0.6 R2），**不依赖 `T.vmax`/`T.reduce_max` 的 NaN 行为**。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1
> （依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**（**单元粒度 = `num_logical` 的唯一定义源**）：
  - **一般网格**（非退化）：单元 = `(n, o_blk, c_block)` → `num_logical = N × ceil(L_out/BL) × ceil(C/CH)`
    （纯 Vector 无空间 `block_N` 维度，故只有通道块维 × 输出位置块维）。
  - **退化路径 C4 整线**（`IS_GLOBAL`，充分条件 = `L_out==1 ∧ p_W==0 ∧ d_W==1 ∧ (k_W-1)·d_W+1>=L_in`）：
    单元 = 整张 `(n, c_block)` 线 → `num_logical = N × ceil(C/CH)`（因 `L_out==1`，与通式同值，无分叉）。
    `d_W>1` 的 `out=1` 配置 `IS_GLOBAL=False`，落入**一般网格**口径。
  - **退化路径 C4b 恒等**（`IS_IDENTITY`）：单元 = 整张 `(n, c_block)` 线 → `num_logical = N × ceil(C/CH)`，
    **显式不同于**通式 `N × ceil(L_out/BL) × ceil(C/CH)`（§5.2 说明）。
  - 代表值见 §5.2 表：smoke 1 / sincnet 3 392 / textcnn 256 / ecg 1 056。

- **② 物理核数依据（实查）**：
  本设计环境实际执行查询（项目 venv `/workspace/venv/bin/python`，2026-10-14，本设计会话）：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()
  # 实查返回值 = 20   (设备 Ascend910B3，npu-smi info 显示 910B3)
  ```
  **纯 Vector 算子核数翻倍**：`vector_cores = 20 × 2 = 40`。
  （**不采用** `CONST-aicore-910B2C` 的 24（翻倍 48）——本设备为 910B3，必须实查；
  核数常数绑定设备型号，跨型号复用会有 ~29% 级偏差。）

- **③ 规模判定与分核方案**（三选一，逐 workload 判定；`num_kernels = min(num_logical, 40)`，
  `num_local = ceil(num_logical / num_kernels)` 为 **Python 整数编译期常量**——静态界）：

  | workload | num_logical | 物理核 | 判定 | 方案 |
  |----------|-------------|--------|------|------|
  | `smoke` | 1 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = 1`，每核 1 个单元 |
  | 多数测试 shape | 1–2 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = num_logical`，每核 1 个单元 |
  | C4 / C4b 特化（各自充分条件成立） | 1–256 | 40 | 视 `N×ceil(C/CH)` | 若 ≤40 → 无需适配；否则 persistent |
  | `textcnn-global`（**C4**） | 256 | 40 | **极大规模**（256 = 40×6 + 16） | **核内串行（persistent）**：`num_kernels = 40`；`num_local = 7`（**静态值**）；`uid = t*num_kernels + cid`，`if uid < num_logical` 守卫 |
  | `ecg-cnn-dilated` | 1 056 | 40 | **极大规模**（1056 = 40×26 + 16） | 同款 persistent：`num_local = 27` |
  | `sincnet-speaker-local` | 3 392 | 40 | **极大规模**（3392 = 40×84 + 32） | 同款 persistent：`num_local = 85` |

  **通用规则（factory 级）**：`num_logical` 按分支粒度取（见 ①），`num_kernels = min(num_logical, vector_cores)`，
  `num_local = ceil(num_logical / num_kernels)`（Python 整数编译期常量）。当 `num_logical ≤ vector_cores`
  → `num_local = 1`，退化为「每核 1 单元 = 无需适配」；当 `num_logical > vector_cores` → 核内串行（persistent）。
  **核内串行循环边界 `num_local` 为静态值**——满足 `core-split-strategy.md` §1 要素③硬要求。
  > **不做空核的保证**：因 `num_kernels = min(num_logical, vector_cores)`，故 `num_kernels ≤ num_logical`，
  > 每核首个任务 `uid = cid < num_logical` 恒有效。
  > **分核增益预期**：三个 full workload 若按 GPU 式超发逻辑核（3392/256/1056），NPU 运行时会把超出物理核数的
  > 部分串行排入 40 核 → 引入核启动开销；persistent 化后固定 40 核 × 静态任务数（`num_local` ≤ 85），
  > 摊薄核启动开销。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block/tile 级调度循环。**不出现逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑单元（task） | block 级并行 | `T.Kernel(num_kernels)` | 每核处理 1 个或多个 `(n,o_blk,c_block)` 单元 |
| 累加器/标记初始化 | **整块填充/清除**（无循环） | `T.vbrc(-inf, acc)` + `T.vbrc(1.0, flag)` + `T.clear(zeros)` + `T.vbrc(inf, infbuf)` + `T.vsub` | 单元入口整 buffer 初始化（§1.6.2 #1/#2/#16） |
| 核内多任务 | 迭代（persistent） | `T.serial(num_local)`（静态界）+ `if uid<num_logical` | 极大规模时摊薄核启动开销（§5.5）；静态边界 |
| 窗口 `k_w` 方向 | 迭代（归约链） | `T.serial(k_W)`（**编译期常量上界**）+ `if 0<=iw<L_in` | tile 级顺序依赖（循环携带累加维）+ 逐 lane 守卫（§1.6.2 #10） |
| 输出位置 `owl` | 迭代（运行时行索引） | `T.serial(BL)`（**编译期常量上界**） | `acc`/`flag` 的运行时行索引；循环体为 C 向量 op（§1.6.2 #11） |
| 元素级（C 轴） | 向量化 | `T.vmax`/`T.vcmp`/`T.vcast`/`T.vmul`/`T.vselect`（向量轴 = C，§1.6.3 决策） | C 轴 lane 并行，无逐元素循环 |
| 载入钳制 | 标量（每单元 2 次） | `T.max`/`T.min`（block 级下标） | 非逐元素热点（§1.6.2 #9） |
| 退化路径 | 单次归约 / 拷贝 | `T.reduce_max`+`T.reduce_min`（C4）/ `T.copy`（C4b） | 充分条件成立时无窗口循环 |

### 6.2 循环伪代码

```python
# num_kernels / num_local 由 num_logical 导出；num_logical 按编译期分支的单元粒度取（§3.3/§5.5①）：
#   一般 = N*num_oblk*num_cblk；C4 整线 = N*num_cblk；C4b 恒等 = N*num_cblk
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 一维网格（<=2 维合法）
    for t in T.serial(num_local):                            # persistent（静态界）
        uid = t * num_kernels + cid
        if uid < num_logical:                                # 尾任务守卫（与 num_logical 同粒度）
            # 解码为 (n, o_blk, c_block)【一般】/ (n, c_block)【C4/C4b】后进入 §3.3 单元体
            ...
            base = ow0 * s_W - p_W
            s0 = T.max(0, T.min(base, L_in - LSPAN))         # 载入钳制（每单元 2 次标量）
            [T.copy span -> row_src; T.transpose; vcast -> row_f32; vcmp(eq)+vcast -> eq_f32]
            for kw in T.serial(k_W):                         # 编译期常量上界
                for owl in T.serial(BL):                     # 编译期常量上界
                    iw = base + owl * s_W + kw * d_W
                    j  = iw - s0
                    if 0 <= iw and iw < L_in:
                        T.vmax(...)                          # C 轴向量最大值
                        T.vmul(...)                          # C 轴向量 NaN 追踪（块级掩码）
            T.vcmp(flag, zeros, cfinal, "eq"); T.vselect(cfinal, nanbuf, acc, out32)
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- 单元内只有一条 `GM→UB→计算→UB→GM` 单阶段流，无多阶段 K 循环可流水；窗口循环次数少（`k_W` ∈ 1–5），
  双缓冲收益有限。
- Developer 模式下编译器可自动 multi-buffer；若 Stage 4 实测显示 MTE2 未与 Vector 重叠，
  可在 `t` 任务循环或 `kw` 循环上探索 `T.Pipelined(num_stages=2)`（列入 Stage 4 候选，不改主结构）。
- C4/C4b 退化路径为「1 次搬运 + 1 次归约 / 纯拷贝」，无流水空间。

### 6.4 尾块处理

- **任务尾块**：`num_logical` 非 `num_kernels` 整数倍 → `if uid < num_logical` 丢弃越界任务。
  退化路径 `num_logical` 按整线单元粒度取（§5.5 ①）。
- **通道尾块**：`real_c = min(CH, C - cb*CH)` 动态 copy/写回范围；padded lane 永不落 GM。
- **输出块尾**：`BL_real = min(BL, L_out - ow0)` 动态写回范围。
- **窗口尾块**：`iw` 守卫跳过界外位置（精确覆盖窗口，无掩码需求；与 adaptive 的「运行时 bin 边界差」不同，
  本算子窗口边界由编译期 `k/s/p/d` 与运行时 `o_w` 线性给出）。
- **无 GEMM 式整除要求**（§5.3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步（Developer 模式）**。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器在 `T.copy`（MTE2 搬运）与 v-prefix 向量 op（Vector 单元）之间
自动插入同步标志，`T.transpose`（VTransposeOp）与 `T.reduce_max`/`T.reduce_min`（VReduceOp）亦由编译器编排。
本算子无跨核通信、无 Cube/Vector 跨引擎依赖：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| （无） | — | Developer 自动同步；本算子单引擎（Vector）、单阶段，无 `sync_block_set/wait` 需求 |

### 7.3 pass_configs 配置

`@tilelang.jit(out_idx=[1], target="npuir")`；不设置自定义 `pass_configs`
（与 `lerp_tensor` / `logsumexp` / 同族池化先例一致）。
若 Stage 3/4 出现 UB 膨胀导致溢出，可评估 `--enable-auto-multi-buffer=false`
（`CONST-capacity-910B2C` 2026-09-16 反例数据点；关闭档膨胀 1.10–1.12）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现/测试基准），
> **不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F


def golden_max_pool1d(x, kernel_size, stride=None, padding=(0,),
                      dilation=(1,), ceil_mode=False):
    """PyTorch reference for max_pool1d, per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark (``MaxPool1dTest.ref_program``
    in examples/TileOPs/tests/ops/test_pool.py).  Exactly 3D NCL; independent of
    the NPU algorithm (no transpose / no C-vectorisation / no span-load /
    no C4-C4b specialisation).  Runs on CPU and does not require an NPU device.
    """
    return F.max_pool1d(
        x.detach().to(torch.float32),  # fp32 opmath domain for fp16/bf16 CPU ref
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    ).to(x.dtype)
```

（golden 以 fp32 中转后舍回输入 dtype，与 §0.1「fp32 比较域」一致；max 只选择不累加，故中转无损，
`atol=rtol=0` 成立。golden 语义已由 `verify_equiv.py` 的 E1 组对照本设计 NPU 公式复核。
**注**：测试 harness 在 `dilation != 1` 时因 torch_npu aclnn 参数支持面较窄而在 CPU 上求 golden 再搬回设备
（`test_pool.py` 第 525–541 行注释），与本 golden 的 CPU 实现一致。）

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 0 | 0 |
| bfloat16 | 0 | 0 |
| float32 | 0 | 0 |

（**逐位精确**：max 是被选中元素的原值，无累加、无舍入误差——依据 §0.1 与
`examples/TileOPs/tests/ops/test_pool.py` 第 568 行注释；源仓 GPU 套件同样用 `atol=0, rtol=0`。
模板通用档 1e-2/1e-4 在本算子不适用。）

### 8.3 L0 门槛测试计划（`tilelang-op-develop` 展开为 L1/L2/Boundary）

| # | 用例 | 覆盖点 |
|---|------|--------|
| L0-1 | `(1,8,32)` k3s2p1 fp32 | manifest smoke（fp32 优先测）；通用路径基线 |
| L0-2 | `(2,8,64)` k3s2p1 fp16 / bf16 | smoke 另两 dtype |
| L0-3 | `(32,80,4096)` k3s3p0 fp16 | manifest `sincnet-speaker-local`；**容量压 BL=26** + persistent 3 392 单元 |
| L0-4 | `(64,256,128)` k128s1p0 fp16 | manifest `textcnn-global`；**C4 整线归约特化**（`out=1`） |
| L0-5 | `(16,128,2048)` k5s2p2 d2 ceil bf16 | manifest `ecg-cnn-dilated`；非单位 dilation + ceil + 重叠窗（BL=31，halo ≈1.11×） |
| L0-6 | `(1,4,63)` k3 s=None p1 d2 fp16 | `stride=None`（= kernel）路径 + 非单位 dilation（**须走一般路径**） |
| L0-7 | `(1,4,97)` k5s3p2 ceil fp16/bf16 | 非重叠窗口 + ceil |
| L0-8 | **NaN 语义门禁**：窗口内置 NaN（首/中/尾多处，含边界与内部位置）→ 输出 NaN | §0.1 NaN 胜出；不得依赖 `vmax` NaN 行为（§9.2 实现级检查） |
| L0-9 | **±inf 语义门禁**：`{+inf,-inf}` 同窗 → `+inf`（非 NaN）；全 `-inf` 窗 → `-inf` | §0.1；`eq` 掩码不误伤 ±inf |
| L0-10 | **非连续输入** | op 层 `.contiguous()` 路径（test `full-noncontiguous-k3-fp16`） |
| L0-11 | **C4 特化 + 容量回退**：`(1,8,128)` k128s1p0 d1 → `out=1`（走 C4）；另加一个假想长线（`L_in` 使 `CH·L_in·11×1.7 > 192KB`，如 `(1,8,4096)` k4096s1p0）应回退一般路径 | §1.6.1 C4 充分条件 + 容量回退 |
| L0-11b | **C4 dilation 守卫**：`(1,8,5)` k2s1p0 **d4** 与 `(1,8,7)` k3s1p0 **d3**（均 `out=1` 但 `d>1`，**不得**走整线归约），另加 `(1,8,5)` d4 且输入仅 `x[2]=1` 的位精确对照 | §1.6.1 C4 的 `d_W==1` 必要约束；输出须与 golden 逐位一致（`verify_equiv.py` C4x 组 6 行 + HAZARD 演示为其机器对应） |
| L0-11c | **`ceil_mode` 回退**：`(1,8,8)` k3s4p0 ceil → `out=2`（触发 `pool_output_dim` 末窗回退分支） | helper 两处 ceil 修正逐字复制的正确性（`verify_equiv.py` E1 组已含该配置） |
| L0-12 | **C4b 特化 + 容量回退**：`(1,8,32)` k1s1p0 d1 → `out=x`，且 `(1,8,16384)` k1s1p0（应回退） | §1.6.1 C4b 充分条件 + 容量回退 |
| L0-13 | **通道尾块 / 输出块尾**：C=80（CH=64 + `real_c=16`）、C=4（CH=8 + `real_c=4`）；`L_out` 非 BL 整数倍（sincnet 1365） | `real_c` / `BL_real` 动态范围；padded lane 不落 GM |
| L0-14 | **参数/输入校验**：非 3D（4D NCHW）/ fp64 / `padding>k//2` / `stride=0` / `dilation=0` / 非正输出长度 | op 层拒绝路径（test 第 619–679 行） |

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 本算子为**纯 Vector**、单阶段、无跨核通信；无 Cube / L1 / L0C 路径。
- 循环上界（`k_W`/`BL`/`LSPAN`/`num_local`）全为编译期常量，**不需要** adaptive 家族的 `max_bin_extent` 机制。
- 退化路径（C4/C4b）的容量随 `L_in` 线性放大，须按 §4.5 不等式独立核算、失败回退（纯优化）。
- 编译期分支（`IS_IDENTITY`/`IS_GLOBAL`）用**纯名 Python bool**（trace 时折叠）；避免表达式条件
  （`not X` / `X and Y` 会生成 TIR runtime if）。
- `BL`/`LSPAN` 是 1D 新增的分块维度（2D 无）；`LSPAN` 由 `BL` 派生，二者必须与 §5.2 的一致口径同步。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **`T.vcmp(...,"ne")` 误用作 NaN 探测** | 想用 `v!=v` 判 NaN | `ne` 是 **ordered-NE**（NaN 恒假）→ 掩码恒假、静默失效；端到端 NaN 用例可能因 `vmax` 未文档化的 NaN 传播而**假阳性通过** | **必须用 `eq`**（`x==x` ⟺ 非 NaN）；本设计用块级 `eq` 掩码乘性累积 + `flag` 判据（§0.6 R2）。**实现级检查项**：断言 `flag` 初值 1、`vcmp(eq)→vcast→vmul` 次序、末尾 `vcmp(flag,zeros,"eq")` |
| **依赖 `T.vmax`/`T.reduce_max` 的 NaN 行为** | 直接 `vmax` 累加就当 NaN 会胜出 | 行为**未文档化**（工具链可能传播也可能忽略），跨版本不稳 | 显式 NaN 追踪（`flag`/掩码）；C4 用独立掩码归约判决，不依赖 `reduce_max` 的 NaN 行为 |
| **C4 的 NaN 判决误用 `reduce_max`** | 0/1 掩码上 `reduce_max + ==0` | `reduce_max==0` **只在全 NaN 线成立**（部分含 NaN 会漏判）；配注释「1.0 iff 无 NaN」则自相矛盾 | **必须用 `reduce_min + ==0`**（「存在 0」⟺ 存在 NaN）；同族前作 Stage 3 实测教训，本设计 §3.3 已冻结 |
| **`T.vcast` 同 dtype（fp32→fp32）** | fp32 输入时 `vcast(row_dst, row_f32, "rint")` | 退化为 round-to-integer → **数据错乱**；C4 路径实测返回错值 | 按 dtype 在 trace 期分派：fp32 走 `T.copy`，fp16/bf16 走 `T.vcast`（§3.2/§3.3 的 `is_f32` 分支） |
| **`@tilelang.jit` 传 `compile_flags`** | 照搬源 GPU 的 `["-O3","-DENABLE_BF16"]` | 本分支 jit 签名无该参数 → 类型错误 | 移除 `compile_flags`（同族前作 Stage 3 实测） |
| **`T.vbrc` destination 写成 buffer region** | `T.vbrc(-inf, acc[0:1, :])`（带运行时偏移的 UB region，且在 `T.serial` 内） | 写入被**静默丢弃**；int32 形态还会读未初始化 UB → 向量核 fault（core dump） | `T.vbrc` 的 dst **必须是整 buffer**（本设计 `acc`/`flag`/`infbuf` 均整 buffer 广播） |
| **`T.vbrc` 标量 operand 用裸 Python 数** | `T.vbrc(-inf, ...)` / `T.vbrc(1.0, ...)` | 命中前端 rank 断言 | 用 **typed PrimExpr**：`-T.infinity("float32")` / `T.float32(1.0)` |
| **`T.vbrc` 同形 src/dst** | src 与 dst 完全同形 | 空广播维数组 → MLIR verify 失败 | src 为**标量**（本设计即如此）；同形场景用 `T.copy` |
| **把 `for k in range(n)` 当 Python 编译期展开** | 在 `@T.prim_func` 内写 `for kw in range(k_W)` 期望展开为 op 序列 | `range` 被解析为 **serial TIR 循环**（loop var 是 `tir.Var`）；用该 var 索引 Python 列表 / `int()` 转换在 trace 期失败 | 本设计统一用 `T.serial(k_W)` / `T.serial(BL)` + 编译期常量上界 |
| **原位 `T.vselect` 在 `T.serial` 内丢 loop-carried 状态** | `T.vselect(cond, chunk, running, running)`（false 操作数同时是输出）跨多迭代 | 状态静默丢失 | 本设计**不用**原位 vselect 累积状态——NaN 累积走 **`T.vmul`**（文档明确支持原地，§3.2）；判决 `T.vselect(cfinal, nanbuf, acc, out32)` 的 `Out` 独立 |
| **`T.vcast` 形状不一致 / 源为 GM** | 直接 `T.vcast(GM_region, ub_f32)`，或 dst 形状不同（承担 reshape） | BiShengHIR 阶段失败 / 广播形状报错 | 先同 dtype `T.copy` GM→UB，再 `T.vcast` UB→UB（同形）；换布局用 `T.transpose` 而非 vcast |
| **`T.transpose` 多轴 / 先 reshape** | `permutation=[2,1,0]`（3-cycle）或 reshape→transpose | verify 失败 / **mis-compile 数据错乱** | 只做二轴交换（本设计用 `[1,0]`）；禁止 reshape 后再 transpose |
| **动态偏移 UB subview** | 在内层循环对 task 级 buffer 做运行时偏移切片**拷贝**（`T.copy(ub[s0:s0+t], ...)`） | auto-multi-buffer 非支配 IR（`operand #2 does not dominate this use`） | 本设计载入用**静态长度 + 钳制源起点**（`s0` 运行时但长度静态、目标从 0 起）；粒度读用 vmax/vmul 的**运行时切片**（与同族前作同形，已过） |
| **UB 溢出** | 长 `L_in` + 大 `BL`/`CH` | 编译期 UB overflow | 容量判据（§4.5）联合压 `BL`/`CH`（sincnet → BL=26）；必要时评估 `--enable-auto-multi-buffer=false` |
| **`real_c` / `BL_real` 尾块漏处理** | `C % CH != 0` / `L_out % BL != 0` | 越界读写 / 落盘错值 | copy/写回用动态 `real_c = min(CH, C-cb*CH)` 与 `BL_real = min(BL, L_out-ow0)`；padded lane 不落 GM |
| **op 次序错误（`vcmp(eq)` 与 `vmul`）** | 先 `vmul` 后 `vcmp` | 掩码与累积错位 → 判据恒真/恒假、静默退化 | 固定次序（§3.3 注释）；`verify_equiv.py` 只验公式不验 op 次序，须在 L0-8/L0-9 门禁实测 |
| **载入钳制 `s0` 越界** | `s0` 未按 `[0, L_in-LSPAN]` 钳制 | GM 越界读 | `s0 = max(0, min(base, L_in-LSPAN))`；`LSPAN ≤ L_in` 由定义保证（§1.6.1 C1 已机器断言下标落界） |

### 9.3 特殊场景处理

- **非整除分块**：无 GEMM 式要求；任务/通道/输出块/窗口四类尾块分别由 `uid` 守卫 / `real_c` / `BL_real` /
  `iw` 守卫处理（§6.4）。
- **极小 shape**：`C < 8`（如 C=4）→ `CH=8` + `real_c` 收尾；`L_out=1`（C4）或 `k=1`（C4b）走特化。
- **长 L**：`L_in` 达 4096 → `BL` 由容量压到 26；`L_in` 极大时若 `CH=8` 仍超容量则 `BL=1` 兜底。
- **混合精度**：三 dtype 均支持；bf16 经 `vcast` 上转 fp32 计算、末尾 `vcast` 舍回（与 fp32 比较域语义一致）。
- **`padding>0` + `ceil_mode`**：末窗部分落在 padding，由 `iw` 守卫按 $-\infty$ 处理，与 torch 一致。
- **`stride=None`**：op 层解析为 `s_W = k_W`（test `full-default-stride-dilation` 覆盖）。
- **NaN / ±inf**：见 §0.1 / L0-8 / L0-9；实现级次序检查见 §9.2。

### 9.4 待确认 / 风险登记（供 Stage 2 复核）

1. **发射项不确定性（最高优先）**：`sincnet` / `ecg` 的 op 数 ≈ 0.57 M / 0.34 M（窄向量 op，宽度 = `CH`），
   `CONST-vector-launch-overhead` 0.5µs/op **不适用于窄 op 外推**，发射项可能成为实际主导项
   （§1.6.0 估算行已标注「未文档化假设」）。→ §1.6.3 已给出主选/备选/裁决计划。
2. **设备差异**：roofline 引用的带宽/发射/容量常数均绑定 Ascend910B2C（48 向量核），本设备为 910B3（40）——
   跨型号复用属**未实测假设**，只作量级下界。
3. **toolchain stamp 差异**：本任务 stamp = `tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388 (commit 417dae9)`；
   同族前作实证 stamp 为 `0d3950aab...`/`db1783d9...` + CANN 9.0.0 + Ascend910B3。本设计沿用的
   实现级结论（fp32 同 dtype `vcast` 误编译、`reduce_min` 判 NaN、`compile_flags` 不受支持）**跨 stamp 复用属
   未实测假设**；采用**防御性设计**：① NaN 探测只用 `eq`（不依赖 `ne`）；② `vbrc` 只写整 buffer；
   ③ `T.serial` 用编译期常量上界（不依赖 `range` 展开）；④ `vcast` 保证同形且源为 UB；
   ⑤ fp32 路径用 `T.copy`。即设计本身对上述陷阱**结构免疫**——若某条已失效，不影响主结构（同族前作 §9.4 同款处置）。
4. **`T.reduce_min` 无独立文档页**：其存在性以 `docs/Tilelang.language/规约操作/T.reduce.md` 条目与
   同族已验证 kernel 的实测用法佐证（§1.6.0 R4 注）；属「文档缺失但实测可用」，列入本表待 Stage 3 复核。
5. **C4 被 manifest 触发、C4b 不被触发**：C4 的正确性由 `verify_equiv.py`（充分条件成立时逐位等价 +
   `d_W==1` 反例组机器锁住）与 L0-4/L0-11/L0-11b/L0-11c 覆盖；C4b 的正确性由 `verify_equiv.py` 与
   L0-12 覆盖，其收益不被当前 benchmark 反映。
6. **halo 重读放大**：`sW < (k_W-1)·d_W+1` 时相邻输出块的 span 重叠（ecg BL=31 时 ≈1.11× GM 读；
   BL=1 时达 4.5×）；已计入 §1.6.0 R3 口径与 §1.6.3 BL 说明，列为 Stage 4 可选探索项（不改主结构）。
7. **`BL`/`LSPAN` 联合容量判据的探针与否**：本设计按 `INFLATE=1.7` 默认档保守判据（§4.5），
   未关闭 auto-multi-buffer；若 Stage 3 遇 UB 膨胀误报，可评估关闭档 1.10–1.12 或进一步压 `BL`。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/max_pool1d/_max_pool1d_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1）
├── _max_pool1d_kernel.py   # 算子实现 + 分层测试（Stage 3 产出）
└── RETROSPECTIVE.md        # Stage 复盘（Stage 1 起追加）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成 | 等价性机器验证（**117** case-group 全 `EQUIV_PASS`；含 C4x dilation 反例组与 ceil step-back 用例） |
| `_max_pool1d_kernel.py` | 待实现 | 算子实现（Stage 3） |
| 分层测试（L0/L1/L2/Boundary） | 待实现 | 内嵌于实现文件或 `testing/`（Stage 3） |

### 10.3 命名规范

- 项目目录名: `max_pool1d`（snake_case）
- 算子目录名: `_max_pool1d_kernel`（源 factory 函数名）
- 实现文件: `_max_pool1d_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1，Stage 3）
3. ⬜ 算子实现（`_max_pool1d_kernel.py`）+ 与 Golden 精度比对（`atol=rtol=0`）
