# MaxPool2dFwdOp（`_max_pool2d_kernel`）算子设计文档

> **迁移任务（TileOPs 同构仓迁移，harness 子模式）**：源算子 = `MaxPool2dFwdOp`
> （family=`pool`），本次重实现的 TileLang kernel 函数 = `_max_pool2d_kernel`
> （→ Kernel class `MaxPool2dKernel`，`ref_api` = `torch.nn.functional.max_pool2d(return_indices=False)`）。
> 本文按「源算子三问解读（§0）→ 算法调研（§1.6.0）→ 硬件耦合性判定与 NPU 重设计（§0.5/§0.6）
> → 通用设计流程（§1–§10）」组织。§1–§10 一律以 §0.6 重设计后的 NPU 算法为输入，不照抄源方案。
>
> - project = `max_pool2d`，op = `_max_pool2d_kernel`
> - 编程模式：**Developer**（迁移 prompt 指定）
> - 源算子路径（kernel）：`/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool2d.py`
>   （函数 `_max_pool2d_kernel`，第 15–77 行；同文件第 258–344 行的函数
>   `_max_pool2d_with_indices_kernel` 与第 347–420 行的 wrapped op / 类
>   `MaxPool2dWithIndicesKernel`（双输出 with_indices 变体）**不在本次迁移范围**）
> - 源 helper：`/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`（`pool_output_dim`）
> - NPU 侧抽取参考件（Stage 0 产物，含逐字内联的 `pool_output_dim`）：
>   `examples/TileOPs/tileops/kernels/pool/max_pool2d/_max_pool2d_fwd_kernels.py`
> - 迁移元数据：`examples/TileOPs/tileops/kernels/pool/max_pool2d/.migration_meta.json`
> - 规格来源（workloads / roofline）：`examples/TileOPs/tileops/manifest/pool.yaml`（`MaxPool2dFwdOp`）
> - 精度用例：`examples/TileOPs/tests/ops/test_pool.py`（`MaxPool2dTest` 段，第 231–436 行）
> - 性能用例：`examples/TileOPs/benchmarks/ops/bench_pool.py`（`test_max_pool2d_bench`）
> - 设备常数（本设计会话实查，2026-10-14）：`NPUUtils.get().get_aicore_num()` = **20**
>   （Ascend910B3，`npu-smi info` 显示 `910B3`）→ 纯 Vector 算子核数翻倍 = **40**；
>   toolchain `tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c`（dev root build）
>   - `torch_npu 2.9.0.post8` + CANN 9.0.0。
> - 同族前作（骨架复用对象）：`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md`
>   （CASE-pool-adaptivemaxpool2d-migration）、
>   `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`、
>   `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/DESIGN.md`。
> - 版本：**revision v1**（`first_design v0` → Stage 2 检视「不通过」后修订；本次针对性闭合
>   REVIEW.md 问题 1（阻塞：C4 充分条件漏 `dilation==1`）+ 问题 2–5，详见 §1.6.1 与文末说明）。
> - **修订 v1 关键调整（相对 design_v0）**：① C4 整平面归约充分条件增补 `d_H==d_W==1` 必要约束
>   （§0.6 R4 / §1.6.1 / §3.3 `IS_GLOBAL` / §5.2 / §5.5 / §8.3 六处同步）；② `verify_equiv.py` 换入
>   合法 C4 用例、增 C4x dilation 反例组 + E1 ceil step-back 用例 + 空真断言（57 → 69 case-group）；
>   ③ §4.5 修正 fp32 CH=64 容量数字（190 272 B）并全节统一 KiB 口径；④ §4.5 C4 容量判据显式含小项；
>   ⑤ 4 处引用/命名细节修正（with_indices 行号、删 `reduce_max 要求静态形状`、`mf32p`→`mf32`、
>   §5.2 简写写全）。**为何不会再犯**：窗口池化特化的正确性判据必须核对窗口的**输入覆盖集合**
>   （$\{k_h d_H\}\times\{k_w d_W\}$）而非仅 `k/p` 与输出网格；本次以 `d==1` 显式约束 + 机器反例组
>   双重锁住。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 方法论：`tilelang-op-design` skill 的
> [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)。
> 逐行读源文件（`max_pool2d.py` 全文 420 行、抽取件
> `_max_pool2d_fwd_kernels.py` 全文 108 行、`pool_output_dim` 内联副本），**未臆测语义**。
> 本次迁移范围 = 单输出变体 `_max_pool2d_kernel`（manifest `MaxPool2dFwdOp`）；
> 同文件的 with_indices 变体不在范围内，本文不涉及。

### 0.1 源算子语义（做什么）

**数学语义**（`torch.nn.functional.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)`，
`return_indices=False`）：

$$
\text{out}[n,c,o_h,o_w]=\max_{0\le k_h<k_H}\ \max_{0\le k_w<k_W}\
x_{\text{pad}}\big[n,c,\ o_h s_H-p_H+k_h d_H,\ o_w s_W-p_W+k_w d_W\big]
$$

其中 $x_{\text{pad}}$ 在 $[0,H_{in})\times[0,W_{in})$ 之外取 $-\infty$。
源实现以「越界 lane **跳过**」表达该 padding（`max_pool2d.py` 第 63 行
`if ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:`，初值
`max_val = T.cast(float("-inf"), accum_dtype)` 第 57 行）——跳过 lane 与 padding $-\infty$
在 $\max$ 下等价（$\max(-\infty,x)=x$），本设计已用 `verify_equiv.py` 的 **E1** 组
（NPU 公式 vs `F.max_pool2d`，fp64 参照）机器复核为**逐位一致（0 mismatch）**。

关键语义要素：

| 要素 | 结论（源实现 + 实测复核） |
|------|--------------------------|
| **规约语义** | 规约轴 = 窗口的 $H\times W$ 两个方向（`for kh in T.serial(kernel_h): for kw in T.serial(kernel_w)`，第 59–60 行）；逐 $(n,c,o_h,o_w)$ **独立取最大值**，无跨通道 / 跨 batch 归约。比较顺序 `kh` 外层、`kw` 内层，但 $\max$ 可交换可结合，**取值集合**与顺序无关（本算子是纯选择，无累加，故无「累加顺序」敏感项）。 |
| **NaN 语义（max 特有）** | 源码**显式**追踪：`has_nan = has_nan \| T.isnan(val)`（第 65 行），末尾 `result = T.if_then_else(has_nan, cast(NaN), max_val)`（第 68–72 行）。即**窗口内任一元素为 NaN ⇒ 输出 NaN**。本设计 torch 实测复核（2.9.0 CPU）：`max_pool2d([1,nan;3,4], 2)` → `nan` ✓。 |
| **±inf 语义** | IEEE 最大值语义：`{+inf,-inf}` 同窗 → `+inf`（**不是 NaN**），实测复核 `max_pool2d([inf,-inf;1,2], 2)` → `inf` ✓。故 NaN 探测**不可**用求和类判据（会把 `{+inf,-inf}` 误判为 NaN）。 |
| **dtype 语义** | 门禁 = `{float16, bfloat16, float32}`（`_max_pool2d` 源类构造第 177–180 行；与 adaptive 家族不同，**含 fp32**）。kernel 内 `accum_dtype = "float"`（第 32 行）：逐元素 `T.cast(x[...], accum_dtype)` 参与比较，末尾 `T.cast(result, dtype)` 单次舍回。**对 max 而言 fp32 是「比较域」而非「累加域」**：-inf 初值、只做选择、输出即被选中元素在 fp16/bf16/fp32 中的原值，`dtype→fp32→dtype` 无损 ⇒ **out 逐位等于窗口内某输入元素（或 NaN）**。这正是精度用例 `atol=rtol=0` 的依据（`examples/TileOPs/tests/ops/test_pool.py` 第 323 行注释原文 "Max is exact (no accumulation): compare bit-for-bit like the GPU suite."）。 |
| **输出尺寸语义** | `out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)`、`out_w` 同理（源第 33–34 行）。helper 含 `ceil_mode` 的两处修正（`effective_kernel = d*(k-1)+1`；ceil 分支向上取整；末尾「最后一窗起点落在 padding 内则回退一窗」）。**该公式不得在任何地方重算**（抽取件注释原文），本设计逐字引用抽取件的内联副本。 |
| **边界语义** | `padding` 约束 $0\le p_H\le \lfloor k_H/2\rfloor$、$0\le p_W\le \lfloor k_W/2\rfloor$（manifest `shape_rules`；op 层校验 "padding must be at most half"）⇒ 每个输出窗口**至少含一个有效输入元素**（无需空窗口兜底，但 `-inf` 初值语义天然给出 `-inf` 兜底）。`ceil_mode` 会令最后一个窗口部分落在 padding（回退规则见上）。 |
| **I/O 契约** | 输入 `x`：`(N, C, H_in, W_in)` **恰好 4D NCHW**（op 层不接受 3D CHW，与 adaptive 家族不同——manifest 注释 "unlike the adaptive pool family the Op layer accepts exactly 4D NCHW input"）；连续布局，尾轴 `W_in` 连续。输出 `out`：`(N, C, out_H, out_W)`，`same_as_input`，原生 NCHW（**不是转置布局**），`out_idx=[1]`。 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（直接移植源仓测试基准
`F.max_pool2d`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（input） | `(N, C, H_in, W_in)`（**恰好 4D**，无 CHW 路径） | float16 \| bfloat16 \| float32 | NCHW 连续；尾轴 `W_in` 连续 |
| `out`（output） | `(N, C, out_H, out_W)` | same_as_input | NCHW 连续；`out_idx=[1]`；`out_H/out_W` 由 `pool_output_dim` 决定 |

**输出 shape（迁移必需字段）**：`(N, C, out_H, out_W)`，其中
`out_H = pool_output_dim(H_in, k_H, s_H, p_H, ceil_mode, d_H)`、`out_W` 同理。
输出与输入同布局（NCHW），**不是转置布局**；无 indices 输出。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（`max_pool2d.py` 第 15–77 行，逐行）：

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host，trace 期） | `out_h/out_w = pool_output_dim(...)`；`total_output = n*c_in*out_h*out_w`（第 33–35 行） | 编译期 shape 常量 | Python int | 输出网格规模 + `ceil_mode` 修正 |
| 2（grid） | `T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx`（第 44 行） | `block_m`、`threads` | 一维 block 网格 | 输出元素级并行划分 |
| 3（lane 扁平索引） | `out_idx = bx*block_m + i`；`ow = out_idx % out_w`；`oh = (out_idx//out_w) % out_h`；`c_idx = (out_idx//(out_w*out_h)) % c_in`；`batch = out_idx//(out_w*out_h*c_in)`；经 `if out_idx < total_output` 守卫（第 46–53 行） | `out_idx` | 标量 `(batch,c_idx,oh,ow)` | 输出坐标 $(n,c,o_h,o_w)$ |
| 4（状态初值） | `max_val = T.cast(-inf, "float")`；`has_nan = False`（第 55–58 行） | — | 标量 fp32 + bool | 最大值初值 $-\infty$ + NaN 标记初值 |
| 5（窗口扫描） | `for kh in T.serial(kernel_h): for kw in T.serial(kernel_w):` 内：`ih = oh*stride_h - pad_h + kh*dilation_h`；`iw = ow*stride_w - pad_w + kw*dilation_w`；`if 0<=ih<h_in and 0<=iw<w_in:` → `val = T.cast(x[batch,c_idx,ih,iw], "float")`；`has_nan = has_nan \| T.isnan(val)`；`max_val = T.max(max_val, val)`（第 59–66 行） | `x` 的逐元素 GM 读 | 标量 fp32 + bool | $\max_{k_h}\max_{k_w}$（越界 lane 跳过 ≡ $-\infty$ padding）+ NaN 胜出标记 |
| 6（结果 + 写出） | `result = T.if_then_else(has_nan, T.cast(nan,"float"), max_val)`；`out[batch,c_idx,oh,ow] = T.cast(result, dtype)`（第 68–73 行） | 标量 | 标量 + 一次 GM 写 | NaN 判决 + 单次舍回输出 dtype |

**数据流与内存访问模式（源硬件视角）**：

```
GM[x] --逐元素标量读 x[batch,c_idx,ih,iw]--> register(max_val fp32, has_nan bool)
        --NaN 判决 + 单次 cast--> GM[out]
（无 SMEM/L1 中转；每个输出元素现场从 GM gather 其窗口的全部元素；相邻输出窗口重叠时重复读；
  源靠 GPU 的 L1/L2 缓存吸收这些重复访问）
```

**循环与并行结构**：
- grid = `ceildiv(total_output, block_m)` 个 block，每 block `threads` 个 CUDA 线程；
- `T.Parallel(block_m)`：每个线程（lane）负责**一个输出元素**，lane 内全为**标量**计算；
- lane 内循环：`T.serial(kernel_h) × T.serial(kernel_w)` **静态边界**（`k_H`/`k_W` 是 trace 期常量）
  - 运行时 `if` 守卫跳过越界 lane；
- **lane 内无归约原语、无向量化**（每 lane 独立标量 `T.max` 链）；
- 无流水线（无 `T.Pipelined`）、无共享内存、无 SIMD。

**host 侧逻辑**（`max_pool2d.py` 第 15–77 行 + 源 op 层）：
- `pool_output_dim` 计算两轴输出长度（**含 `ceil_mode` 修正**，不得重算）；
- 参数规范化（`kernel_size`/`stride`/`padding`/`dilation` 的 int↔2-tuple 展开、`None` stride → kernel）
  与校验（`padding ≤ kernel//2`、`stride>0`、`dilation>0`、4D NCHW、dtype 门禁）在 op 层完成；
- `functools.lru_cache(maxsize=32)` 缓存 kernel factory（按 shape/dtype）；op 层 `_kernel_cache`
  按 `(n,c,h,w,dtype,device)` 缓存 kernel 实例；
- 编译配置：`@tilelang.jit(out_idx=[1], compile_flags=["-O3","-DENABLE_BF16"])`（第 37 行）；
- host 侧**无**输入张量内容改动（无 padding / 无 reshape 数据拷贝）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **一维 block 网格 + 输出元素扁平映射** | 把 `(N,C,o_H,o_W)` 四维问题压成一维可并行任务 | `T.ceildiv(total_output,block_m)` + `%`/`//` 反解 `(batch,c,oh,ow)` | GPU block 网格（非 hardware-specific） | **等价替换**（一维 `T.Kernel`） |
| 2 | **`T.Parallel(block_m)` 线程级 lane 并行** | 每线程处理一个输出元素、隐藏逐元素 gather 延迟 | CUDA thread/warp 两级并行模型 | **CUDA threads / warp**（NPU 无等价） | **重新设计** → §0.6 R1 |
| 3 | **静态边界窗口循环 + 运行时越界守卫** | `k_H`/`k_W` 为 trace 期常量，循环上界静态；守卫跳过 padding lane | `T.serial(kernel_h) × T.serial(kernel_w)` + `if 0<=ih<.. and 0<=iw<..` | 无（语言/编译器级约束） | **保留**（NPU `T.serial` 同样只接受静态边界） |
| 4 | **标量 max 比较链 + `-inf` 初值** | 单线程顺序比较、寄存器驻留；`-inf` 是 `max` 的单位元 | `max_val = T.max(max_val, val)` 于 `T.alloc_var` 标量 | **GPU 大寄存器堆 + 寄存器比较链** | **重新设计** → §0.6 R2 |
| 5 | **逐元素标量 NaN 追踪 + 结果覆盖** | 复刻 PyTorch「NaN 胜出」，不依赖 `T.max` 的 NaN 行为 | `has_nan = has_nan \| T.isnan(val)`；末尾 `has_nan ? NaN : max_val` | 无（纯数值语义） | **保留**（语义层，需在 NPU 上向量化） |
| 6 | **fp32 比较域 + 单次输出舍入** | 与 PyTorch 比较语义一致；max 下无损 | `accum_dtype="float"`，逐元素 cast，末尾一次 cast 回 | 无（纯数值语义） | **保留**（语义层） |
| 7 | **逐元素 GM gather（无 tile 搬运 / 无 coalescing）** | 接口简单，按窗口坐标直接取值 | 直接索引 `x[batch,c,ih,iw]` | GPU 的 L1/L2 缓存吸收重复访问 | **重新设计** → §0.6 R3 |
| 8 | **`T.alloc_var` 标量寄存器状态** | lane 内状态（`max_val`/`has_nan`）寄存器驻留 | `T.alloc_var` + 顺序更新 | GPU 寄存器 + 单线程执行模型 | **重新设计** → §0.6 R1/R2 |
| 9 | **`out_idx < total_output` 尾任务守卫** | 处理 `total_output` 非 `block_m` 整数倍 | 运行时谓词跳过越界 lane | 无 | **保留**（模式可移植） |
| 10 | **`functools.lru_cache` factory 缓存 + op 层 kernel 实例缓存** | 避免重复 trace/编译 | host 侧缓存 | 无 | **保留**（host 侧） |
| 11 | **`-inf` 初值 + 越界跳过（≡ padding −∞）** | 让 padding 不参与 `max`，且无需掩码 | 初值 $-\infty$ + in-bounds 谓词 | 无（数值语义） | **保留**（语义层；已由 §1.6.1 E1 机器验证） |

> 识别不出来 ≠ 不存在。以上 11 项为源码中可见的全部优化/结构性选择；未识别的项会在迁移中被静默丢弃。
> 源码**未**使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / `T.Pipelined`，故无「舍弃」类 GPU 专有手段。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（窗口取最大值；越界跳过 ≡ padding $-\infty$；`ceil_mode` 输出尺寸规则） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1）；已由 §1.6.1 E1 机器验证 |
| **NaN 胜出 + ±inf 最大值语义（#5）** | 语义 | 无 | — | **保留** | 数值语义；NPU 侧用 IEEE 正确的 `eq` 掩码乘性累积实现（§0.6 R2；`ne` 为 ordered-NE 不可用于 NaN 探测） |
| fp32 比较域 + 单次输出舍入（#6） | 语义 | 无 | — | **保留** | NPU Vector 支持 fp32 `T.vmax`/`T.vcast`（`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 fp32 ✓、`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 f32→f16/bf16 ✓） |
| I/O 契约（4D NCHW→NCHW、三 dtype、同 dtype 输出、无 indices） | 语义 | 无 | — | **保留** | layout 是契约 |
| padding 约束下窗口非空、`ceil_mode` 末窗回退 | 语义 | 无 | — | **保留** | 语义层 |
| 静态边界窗口循环 + 守卫（#3） | 算法 | 无（语言约束，非硬件） | 有（同款约束） | **保留** | NPU `T.serial` 同样只接受静态边界；本算子 `k_H`/`k_W` 是**编译期常量**，上界天然静态，比 adaptive 家族更简单 |
| `-inf` 初值 + 越界跳过（#11） | 算法 | 无 | 有 | **保留** | `T.vbrc(-T.infinity("float32"), acc)` 整 buffer 广播；`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（scalar→任意 shape） |
| 尾任务守卫（#9） | 算法 | 无 | 有 | **保留** | `if uid < num_logical` 运行时谓词 |
| lru_cache / kernel 缓存（#10） | host | 无 | 有 | **保留** | host 缓存 |
| 一维 block 网格 + 输出扁平映射（#1） | 优化 | GPU block 网格 | 有（一维 `T.Kernel`） | **等价替换** | migration-analysis.md §5.3「block×thread 两级并行 → block 级任务划分 + 向量化元素并行」；`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3 |
| **`T.Parallel(block_m)` 线程级 lane 并行（#2）** | 优化 | CUDA threads / warp | **无** | **重新设计** → §0.6 R1 | NPU 无 CUDA thread 概念；每 lane 一个输出元素会退化为**标量**执行（pattern-library `PL-1.2-caxis-accum` 同族实测窗口累加 98.3% 标量，`layout.md` 诊断触发器） |
| **标量 max 比较链（#4）** | 优化 | GPU 寄存器堆 + 寄存器比较链 | **无直接等价** | **重新设计** → §0.6 R2 | migration-analysis.md §5.3「register 累加 → UB 中间缓冲 / Vector 归约」；NPU 侧改为 C 轴向量 `T.vmax` |
| **`T.alloc_var` 标量状态（#8）** | 优化 | GPU 单线程执行模型 | **无** | **重新设计** → §0.6 R1/R2 | 状态改为 UB 向量累加器（`acc` / `flag`），lane 映射到 C 轴 |
| **逐元素 GM gather（#7）** | 优化 | GPU L1/L2 缓存 | **无等价（且有害）** | **重新设计** → §0.6 R3 | NPU 逐元素 gather 无 coalescing、无 cache 吸收；改为 UB tile 搬运 + 向量化（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum`） |
| 退化/几何特化（源无；同族模式集） | 优化 | 无 | 有 | **重新设计** → §0.6 R4 | 源对所有网格一律走通用双循环；NPU 侧补窗口数学**严格成立**的结构特化（§1.6.1 C4/C4b，C4 含 `d_H==d_W==1` 必要约束）——注意窗口池化的特化条件与 adaptive 家族**不同**（见 §0.6 R4） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目 / `docs/Tilelang.language/` 条目 / pattern-library 实测条目（见「依据」列）。
> **舍弃项：无。** 源码未使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / swizzle 等 GPU 专有手段，故无舍弃条目。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（每项含「源方案 → NPU 新算法 → 语义保持论证」）。
**重设计输入含 §1.6.0 算法调研结论**：调研选定「直接窗口取最大值 + 显式向量 NaN 追踪 + fp32 比较域 +
C 轴向量化 + 核内融合转置」算法族（与源算法同族，但归约/追踪单元由标量改为 C 向量），
故 §0.6 的重设计直接承接调研结论；源算法已识别优化手段（§0.4）的优化意图承接见每项「承接」行。

**重设计项 R1：并行结构重设计（线程级 lane 并行 → 一维 Kernel + C 轴向量任务划分）**

- **源方案**：`T.Kernel(ceildiv(total_output,block_m), threads=threads)` + `T.Parallel(block_m)`，
  每 CUDA 线程处理一个输出元素，lane 内 `T.alloc_var` 标量状态。
  性能意图：把 `N*C*out_H*out_W` 个输出元素摊到大量线程上并行，靠 warp/线程切换隐藏逐元素 GM gather 延迟。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`；**逻辑任务单元 = `(n, o_h, c_block)`**，
  其中 `c_block` 是长度 `CH` 的**通道向量块**；`num_logical = N * out_H * ceil(C/CH)`（一般网格）。
  单元内以 **C 轴为向量维**；`o_h` 由单元 id 运行时解码；`o_w` 与 `k_w` 为 `T.serial` 内层循环
  （`out_W`、`k_W` 均为编译期常量），循环体为 C 向量 op。`num_logical > vector_cores` 时按
  persistent grid-stride 模式挂 `T.serial` 核内循环（静态界，见 §5.5）。
- **语义保持论证**：任务划分只是把互不相关的输出元素分给不同单元，**不改变任何窗口的取值集合与比较结果**
  （每个输出仍对其窗口取最大值，越界 lane 仍按 $-\infty$ 处理）；无跨单元归约，无边界语义变化。
  承接：源 #1 的一维扁平映射被保留为「一维 Kernel」，#2 的 lane 并行意图由「单元并行 + C 轴向量并行」承接。

**重设计项 R2：归约重设计（per-lane 标量比较链 + 标量 NaN 标记 → UB 内 C 轴向量取最大值 + 向量 NaN 掩码）**

- **源方案**：每输出元素在寄存器里做 `max_val = T.max(max_val, T.cast(x[...],"float"))` 标量比较链，
  并行 `has_nan = has_nan | T.isnan(val)` 标量 bool 累积，初值 $-\infty$/`False`。
- **NPU 新算法**：C 轴向量化（向量轴 = C，§1.6.3）：
  1. 累加器 `acc`（`out_W × CH` fp32）经 `T.vbrc(-T.infinity("float32"), acc)` 置 $-\infty$（**单位元是 $-\infty$，非 0**）；
     NaN 追踪器 `flag`（`out_W × CH` fp32）经 `T.vbrc(T.float32(1.0), flag)` 置 1。
  2. 逐元素 cast 改为 **tile 级向量 cast**：`T.vcast(row_dst, row_f32, round_mode="rint")` 一次把整行 `(W_in, CH)` 提到 fp32。
  3. 取最大值改为 **C 轴向量最大值**：`T.vmax(acc[ow:ow+1, :], row_f32[iw:iw+1, :], acc[ow:ow+1, :])`（原地）。
  4. **NaN 追踪向量化为「乘性 eq 掩码」**（max 特有，语义关键）：行级
     `T.vcmp(row_f32, row_f32, cond, "eq")`（$x{=}x$ 为 1、NaN 为 0）→ `T.vcast(cond, mf32, "rint")`；
     逐元素 `T.vmul(flag[ow:ow+1, :], mf32, flag[ow:ow+1, :])`（`flag` 初值 1：0 吸收、1 中性）。
     **不依赖 `vmax` 的 NaN 行为**（该行为未文档化；`ne` 为 ordered-NE 亦不可用，见 §9.2）。
  5. 输出：`cfinal = T.vcmp(flag, zero, "eq")`（`flag==0` ⟺ 窗口含 NaN）→
     `T.vselect(cfinal, nanbuf, acc, out32)`（`nanbuf = inf - inf` = NaN）→
     `T.vcast(out32, acc16, round_mode="rint")`（单次舍回 dtype）→ `T.transpose(acc16, acc_t, [1,0])` → `T.copy` 写回。
- **语义保持论证**：
  - **数学等价**：窗口扫描不改变取最大值的元素集合；C 轴是**独立轴**（每通道独立取最大值，§0.1），
    把 `c` 放进向量 lane 只是 SIMD 数据并行，**不改变任何 `c` 的取值集合**。
  - **NaN/±inf 语义核对**（逐条对齐 §0.1 实测）：`{+inf,-inf}` 无 NaN → `cond` 恒真 → `mf32` 恒 1 →
    `flag` 保持 1 → `cfinal` 恒假 → 输出 `acc`，而 `T.vmax(-inf,+inf)=+inf` ✓；窗口含 NaN →
    对应 lane 的 `cond` 为 0 → `mf32=0` → `flag` 变 0（吸收）→ `cfinal` 真 → 输出 NaN ✓。
    本设计**只用 IEEE 正确的 `eq` 比较**（`x!=x` 的 ordered-NE 陷阱见 §9.2），与求和类探测（会把 `{+inf,-inf}`
    误判为 NaN）本质不同。
  - **dtype/舍入**：与源一致（fp32 比较域、末尾单次 `rint`）；`vcast` fp16/bf16→fp32 为无损上转，
    fp32→fp16/bf16 用 `rint`；对 max 结果无损（输出即输入元素）。
  - 承接：源 #4 的「寄存器比较链」意图由「UB fp32 向量最大值累加器」承接；#5 的「标量 bool NaN 追踪」
    由「C 向量 eq 掩码乘性累积」承接；#6 由「fp32 向量比较域 + 单次 `rint`」承接。

**重设计项 R3：数据搬运重设计（逐元素 GM gather → UB tile 搬运 + 核内融合转置）**

- **源方案**：直接在 GM 上按 `x[batch,c,ih,iw]` 逐元素取值。性能意图：接口简单；
  GPU 靠 L1/L2 缓存吸收重叠窗口的重复访问。
- **NPU 新算法**：按 C 轴向量化所需布局（C 最内连续）搬运：
  1. GM→UB：按 `(real_c, W_in)` 行片整片搬运 `T.copy(x[nb, cb*CH:cb*CH+real_c, ih, 0:W_in], row_src[0:real_c, 0:W_in])`；
  2. UB 内**融合转置链**：`T.transpose(row_src, row_dst, [1,0])` 把 `(CH, W_in)` 变成 `(W_in, CH)`（C 最内）；
     依据 `PL-1.1-transpose-chain`（核内二轴交换链，实测 ~5.4µs / 4 次；`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 仅支持二轴交换）；
  3. 转置后 C 轴连续 → `T.vcast` 到 fp32 → 向量取最大值 / eq 掩码；
  4. 输出侧：`out32 (out_W, CH)` → `vcast` → `acc16` → `T.transpose([1,0])` → `acc_t (CH, out_W)` → `T.copy` 回 GM。
- **语义保持论证**：搬运/转置是**纯数据重排**（双射），不改变元素值与窗口取值集合；输出写回后与 NCHW 契约一致。无精度影响（不引入任何算术）。
  承接：源 #7「靠缓存吸收重复访问」的意图由「UB 行片常驻 + 同一行被 `o_w` 方向全部输出列复用（`k_H` 行只搬一次/单元）」承接。
  **读放大说明**：`s_H < k_H` 时同一输入行参与多个输出行，产生 `k_H/s_H` 量级的行重复读
  （vgg/alexnet 的 `s_H=k_H` 无放大；resnet-stem `k3s2p1` ≈ 1.5×）——列入 §1.6.0 R3 口径与 §1.6.3 待裁决项。

**重设计项 R4：结构特化（源无特化；窗口池化的特化条件与 adaptive 家族不同）**

> 该项属「算法替代」模式（migration-analysis.md §6.2），采纳项与 §1.6.0 调研结论一致、并在 §1.6.1 完成等价性论证与机器验证。

- **源方案**：对所有输出网格一律走通用双循环（含 `out_H==out_W==1` 与 `out==in` 类形状）。
- **NPU 新算法 + 关键校正（本设计独立推导）**：adaptive 家族的两条退化特化（`out==1×1` → 整平面归约；
  `out==in` → 恒等拷贝）**不能照搬到窗口池化**——adaptive 的 bin 边界随 `out` 变化（`out==in` 时
  bin 恰为单位 bin），而**窗口池化的窗口形状由 `k/s/p/d` 固定**。逐条给出真正成立的充分条件：
  - **C4（整平面归约）**：**充分条件**为 `out_H==out_W==1` 且 `p_H==p_W==0` 且 `d_H==d_W==1` 且
    `(k_H-1)*d_H+1 >= H_in` 且 `(k_W-1)*d_W+1 >= W_in`（此时窗口恰为
    $[0,H_{in})\times[0,W_{in})$，输出 = 整平面最大值）。**`d_H==d_W==1` 是必不可少的约束，省略即错**
    （本设计 Stage 2 检视修正项，REVIEW 问题 1）：窗口在 H 轴的**输入覆盖集合**是
    $\{k_h d_H\}\cap[0,H_{in})$（步长 $d_H$），**仅当 $d_H=1$ 时才等于 $[0,H_{in})$**；
    $d_H>1$ 时未被覆盖的输入行不参与窗口取最大值，而整平面 `reduce_max` 会把它算进去 ⇒ 取值集合不同。
    **反例一（缺 dilation 约束）**：`H_in=W_in=5, k=(2,2), s=(1,1), p=(0,0), d=(4,4)`
    时 `out_H=out_W=1`，且 `k/p` 与覆盖长度条件全部成立，但窗口仅覆盖行 $\{0,4\}$ × 列 $\{0,4\}$；
    输入仅 `x[2,2]=1`、其余为 0 时 `F.max_pool2d` = **0.0**，整平面归约 = **1.0**
    （torch 数值复核 + `verify_equiv.py` C4x 组机器锁住）⇒ **静默错值**，故必须显式排除。
    **反例二（缺 coverage 约束）**：`H_in=3,k_H=3,s_H=4,p_H=1` 时 `out_H=1` 但窗口为
    `{-1,0,1}∩[0,3)={0,1}`，**不含** `i_h=2` ⇒ 整平面归约亦会给出错值。
    为降低实现与验证风险，本设计**不**利用 `H_in==1`（同理 `W_in==1`）的特例放宽整平面覆盖
    （数学上可写 `(d_H==1 or H_in==1)`），只用上述最简充分条件形态。
    上述五合一条件在 factory 层折叠为纯 Python bool（§3.3 `IS_GLOBAL`），
    **满足即走 C4，不满足一律走一般路径**（`d>1` 的 `out=1×1` 配置因此被正确挡在 C4 之外）。
  - **C4b（恒等拷贝）**：`out==x` 当且仅当 `k_H==k_W==1 and d_H==d_W==1 and p_H==p_W==0 and s_H==s_W==1`
    （窗口恰为单元素）。**反例**：`H_in=5,k_H=3,s_H=1,p_H=1` 时 `out_H=5==H_in` 但窗口为
    `{o_h-1,o_h,o_h+1}`（3 元素）⇒ 恒等拷贝会给出错值。
- **语义保持论证**：见 §1.6.1（C4/C4b 在**上述充分条件成立时**与基线公式**逐位等价**，
  `verify_equiv.py` 全部 `EQUIV_PASS`）。C4 的 NaN 判决由 `T.reduce_max` + 独立 eq 掩码归约给出，
  **不依赖 `reduce_max` 的 NaN 行为**（无论它传播还是忽略，掩码归约独立给出正确判决）。
- 承接：源「通用双循环」在非退化网格上保持不变（不存在优化意图丢弃）；退化网格特化**不改变语义**，
  仅减少冗余循环与 op 数。**注意**：两条特化均**不被当前 manifest workloads 触发**（见 §5.2 表），
  属对 op 层一般参数空间的正确性/性能补全，容量受 §4.5 判据约束。

**重设计后的 NPU 算法一句话总结**：NCHW 进 / NCHW 出，核内按 `(n,o_h,c_block)` 单元把 C 轴作为向量轴，
对 `k_H × k_W` 窗口做 fp32 向量取最大值（`T.vmax`），并在**同一遍扫描内**用 eq 掩码乘性累积
（`vcmp(eq)`+`vcast`+`vmul`）追踪 NaN，用核内二轴转置把 C 搬到最内连续轴；
满足充分条件的退化网格（整平面 / 单元素恒等）走结构特化路径。

### 0.7 标杆实现

- 源算子路径（获取 golden 语义依据）：`/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool2d.py`
  （函数 `_max_pool2d_kernel`）。
- 源仓参考实现 / 测试基准（NPU 侧移植后）：
  - `examples/TileOPs/tests/ops/test_pool.py`（`MaxPool2dTest.ref_program` = `F.max_pool2d(...)`，torch CPU 参考，`atol=rtol=0`）；
  - 源仓同文件（GPU）`tests/ops/test_pool.py` 的 `MaxPool2dTest`。
- **golden 以 §0.1 语义为唯一依据**，直接使用 `torch.nn.functional.max_pool2d`
  （优先移植源仓基准），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。
- 规格来源：`examples/TileOPs/tileops/manifest/pool.yaml` 的 `MaxPool2dFwdOp`
  （4 组 workloads；`roofline.flops = N*C*out_H*out_W*kH*kW`、`bytes = (N*C*H_in*W_in + N*C*out_H*out_W)*elem_bytes`）。
- 基准来源：`examples/TileOPs/benchmarks/ops/bench_pool.py` 中的函数 `test_max_pool2d_bench`。

---

## 1. 概述

### 1.1 算子名称

`_max_pool2d_kernel`（迁移自 TileOPs `MaxPool2dFwdOp` / `MaxPool2dKernel`）。

### 1.2 功能描述

窗口式（非自适应）二维最大池化前向：对每个输出位置 $(o_h,o_w)$，在其
`kernel_size × stride × padding × dilation` 决定的窗口内取最大值；
窗口越界部分按 $-\infty$ 填充（源实现以「越界 lane 跳过」表达）；
窗口内任一元素为 NaN 时输出 NaN（NaN 胜出）。
输入 **恰好 4D NCHW**，dtype 支持 `float16 / bfloat16 / float32`，输出同 dtype。

### 1.3 数学公式

$$
\text{out}[n,c,o_h,o_w]=\max_{0\le k_h<k_H}\ \max_{0\le k_w<k_W}\
x_{\text{pad}}\big[n,c,\ o_h s_H-p_H+k_h d_H,\ o_w s_W-p_W+k_w d_W\big]
$$

$$
x_{\text{pad}}[n,c,i_h,i_w] = x[n,c,i_h,i_w]\ \ \text{if}\ \ 0\le i_h<H_{in}\ \ \text{and}\ \ 0\le i_w<W_{in};
\qquad x_{\text{pad}} = -\infty\ \ \text{otherwise}
$$

窗口含 NaN ⇒ 输出 NaN（NaN 胜出；$\{+\infty,-\infty\}$ 同窗取 $+\infty$）。
输出尺寸 $out_H=$ `pool_output_dim`$(H_{in},k_H,s_H,p_H,\text{ceil\_mode},d_H)$、$out_W$ 同理
（helper 逐字取自源仓 `common.py`，**不重算**）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，见 §0.6）** 单步 + 窗口归约类算子，纯 Vector：

1. host/trace 期（factory 内）：由 `(N,C,H_in,W_in,k_H,k_W,s_H,s_W,p_H,p_W,d_H,d_W,ceil_mode)`
   算出输出两轴长度 `out_H/out_W`（`pool_output_dim`）；判定退化特化路径（§1.6.1 C4/C4b 的**充分条件**）
   与 UB 容量档（通道向量块 `CH`，§4.5）；`num_logical / num_kernels / num_local` 在 factory 层求为
   编译期常量。
2. 一般网格：逻辑任务单元 `(n, o_h, c_block)`（`CH` 通道向量块）。单元内：
   - `acc`（`out_W × CH` fp32）经 `T.vbrc(-inf)` 置 $-\infty$；`flag`（`out_W × CH` fp32）经 `T.vbrc(1.0)` 置 1（NaN 免标记）；
   - `for kh in T.serial(k_H)`：算 `ih = o_h*s_H - p_H + kh*d_H`，若 `0<=ih<H_in`（**每行一次守卫**，C1）
     则搬入输入行 `(CH, W_in)` → 核内转置 `(W_in, CH)` → `vcast` fp32；
   - `for ow in T.serial(out_W): for kw in T.serial(k_W)`：算 `iw = ow*s_W - p_W + kw*d_W`，
     若 `0<=iw<W_in` 则 `T.vmax` 更新 `acc[ow]`，并 `vcmp(eq)+vcast+vmul` 累积 `flag[ow]`；
   - 末尾：`cfinal = (flag == 0)`（⟺ 窗口含 NaN）→ `out32 = vselect(cfinal, nanbuf, acc)`（`nanbuf = inf-inf` = NaN）
     → `vcast` 回 dtype → `transpose([1,0])` 回转 `(CH, out_W)` → 写回 GM。
3. 退化网格（§1.6.1 C4/C4b 充分条件成立时）走结构特化路径（整平面 `reduce_max` / 单元素恒等拷贝）。

**与源算法的结构差异**（来源标注）：

- 归约单元：源「每输出元素一个线程 + 标量比较链」→ NPU「每单元一个 C 向量块 + `T.vmax` 向量比较」（§0.6 R2）。
- NaN 追踪：源「标量 bool `has_nan \| isnan(val)`」→ NPU「C 向量 eq 掩码乘性累积」（§0.6 R2；语义不变）。
- 访存：源「逐元素 GM gather」→ NPU「UB 行片搬运 + 核内融合转置」（§0.6 R3）。
- 并行：源「block×thread 两级 + lane/元素」→ NPU「一维 Kernel + `(n,o_h,c_block)` 单元 + C 轴向量 lane」（§0.6 R1）。
- 守卫：源「逐元素 in-bounds 谓词」→ NPU「H 方向提升为每行一次（C1）+ W 方向逐 lane」（§1.6.1 C1）。
- 结构特化：源无 → NPU 有（§0.6 R4 的**校正充分条件**，见 §1.6.1 C4/C4b）。
- **不变**：窗口取值集合、$-\infty$ padding 语义、NaN 胜出 + $\pm\infty$ 取 $+\infty$、`ceil_mode` 输出尺寸规则、
  fp32 比较域 + 单次输出舍入、NCHW 契约。

### 1.5 数据流图

```
GM[x: (N,C,H_in,W_in) fp16/bf16/fp32]
   │  T.copy 行片 (real_c, W_in)
   ▼
UB[row_src: (CH, W_in) dtype] --T.transpose[1,0]--> UB[row_dst: (W_in, CH) dtype] --T.vcast(rint)--> UB[row_f32: (W_in, CH) f32]
   │  T.serial kh (静态界 k_H) + 每行守卫 (0<=ih<H_in)
   ▼
   ├─ 值：   UB[acc  (out_W,CH) f32 <- vmax(acc, row_f32[iw]) ，初值 -inf]
   └─ NaN：  UB[flag (out_W,CH) f32 <- vmul(flag, vcast(vcmp(row_f32,row_f32,"eq"))) ，初值 1]
   │
   │  cfinal = vcmp(flag, zero, "eq")            # flag==0 ⟺ 窗口含 NaN
   ├─> UB[out32 f32] = vselect(cfinal, nanbuf, acc)  --vcast(rint)--> UB[acc16] --transpose[1,0]--> UB[acc_t (CH,out_W)] --T.copy--> GM[out]
（退化路径 C4：GM → UB 连续 (CH,H_in,W_in) → vcast f32 → reduce_max(dim=2→1)；
              eq 掩码 → vcast → reduce_max(dim=2→1) 判 has_nan；vselect 判决 → vcast → GM）
（退化路径 C4b：GM → UB 整平面 (CH,H_in,W_in) → GM 纯拷贝）
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
> **命中行已全部评估**；其中 `R1` 命中项对 **max 不成立**（见 R1 候选 1/4）——这是本算子与同族 avg 池化的关键差异。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算法**（直接窗口取最大值，逐元素标量 + 标量 NaN 追踪） | $\max_{k_h,k_w}x_{\text{pad}}$ | — | — | ✅（基线） |
| 0' | **基线向量化**（C 轴向量 `vmax` + 向量 eq 掩码 NaN 追踪） | 同 #0 | **数学恒等**（C 轴独立，仅 SIMD 化） | 标量链 → 向量链（`/CH` 次向量 op） | ✅（选定） |
| 1 | 常数权重卷积 / im2col+Cube GEMM（`ALG-pooling` 的 `R1` 命中行） | $\text{out}=W_0\otimes x$（0/1 权重） | **❌ 不等价**：$\max$ 非线性，**不存在权重矩阵使 $\max$ 等于线性加权和**；命中行仅对 avg/sum pool 成立 | — | ❌（数学不等价） |
| 2 | sum-pool × 常数（`ALG-pooling` 命中行） | $(\sum x)\cdot c$ | **❌ 不等价**（给出 avg/sum，不给 max） | — | ❌（数学不等价） |
| 3 | 可分离 2D（H/W 两轴分两遍取最大） | $\max_{i_h}(\max_{i_w}x)$ | 数学恒等（$\max$ 可交换结合） | 名义上可复用行部分最大值；见 R3 | ✅（量化后弃） |
| 4 | 前缀最大值 + 差分 | $P[b]-P[a]$，$P$ = 前缀最大值 | **❌ 不等价**：$\max$ **无逆元**，$\max P[b],P[a]$ 之差**不等于**区间最大值（例：$x=[5,1,3]$，区间 $[1,3)$ 的 max $=3$，而 $P[3]{-}P[1]=5{-}5=0$） | — | ❌（数学不等价） |
| 5 | 归约结构优化：值与 NaN 判决**同一遍扫描内融合**（源已同轮） | 单轮：`vmax` + eq 掩码 | 数学恒等（同轮与分轮结果相同） | 避免第二遍扫描 | ✅（并入 #0'） |
| 6 | 退化/几何结构特化（整平面 / 单元素恒等） | `out=1×1`: $\max_{\text{all}}x$；恒等: `out=x` | 数学恒等（**仅在窗口数学上覆盖整平面 / 恰为单元素时**；充分条件见 §1.6.1 C4/C4b） | 整平面：多次向量 `vmax` → 单次归约；恒等：全部归约 op → 纯拷贝 | ✅ |

**R1 初判小结**：唯一数学等价的化简结构是「**退化网格结构特化**」（#6，**条件需按窗口语义重新推导**）
与「**NaN 判决融合**」（#5，源已有）；与 avg 池化不同，max 的**全部代数化简路径（#1/#2/#4）因
$\max$ 非线性 / 无逆元而不成立**——这是 max 归约相对 sum 归约的本质限制（数学事实，非 API 或性能问题）。

**R2 在线算法**：**有**（且基线本身已是单遍）。结构判据：算子可分解为 **running 最大值**（窗口内最大值）
- **running NaN 标记**（窗口内是否已见 NaN），无跨元素依赖，故天然存在单遍流式变体——
每个输入元素读一次、参与其归属（窗口重叠时若干）输出的 `vmax` / NaN 判决。
**收益口径**：扫描遍数 **1**（无两遍式）、中间缓冲 `O(out_W·CH)`（远小于 `O(H·W)` 的物化）、UB 驻留完全可行（§4.5）。
迁移任务佐证：源算法 §0.4 #4/#5 已是 running 比较与 running 标记（`max_val = T.max(max_val,val)`、
`has_nan = has_nan | T.isnan(val)`），是「存在在线变体」的直接证据。
结论：**在线单遍变体 = 采用（= 基线结构）**，无第二遍扫描 / 无中间矩阵物化。

**R3 复杂度对比**（逐候选四口径；本算子是**带宽受限**算子，访存 / 遍数为主导项，不只看 FLOPs）：

口径：FLOPs = 元素访问次数（一次最大比较 = 1）；访存 Bytes = GM 读 + GM 写（**不含** UB 内搬运；
中间缓冲不落 GM）；扫描遍数 = 输入 GM 读取遍数；中间缓冲峰值 = 片上驻留量。

| 算法候选 | FLOPs（比较次数） | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|--------------------|---------|-------------|---------------------|
| 0 基线（直接标量） | `N·C·out_H·out_W·k_H·k_W` | `(N·C·H_in·W_in + N·C·out_H·out_W)·eb` | 1（窗口重叠时靠缓存吸收重复读） | `O(1)`（寄存器标量） | 元素级并行 `N·C·out_H·out_W`；**无跨核归约** |
| **0' 基线向量化（选定，C 轴）** | 同上（`/CH` 次向量 `vmax`） | 同上 + 行重复读放大 `≈ k_H/s_H`（vgg/alexnet = 1；resnet-stem ≈ 1.5） | 1 | `acc+flag+row_f32` 等 `O((out_W+W_in)·CH)` | 单元并行 `N·out_H·ceil(C/CH)`；**无跨核归约** |
| 1 常数权重卷积 / GEMM | **不适用**（非线性，无法表达） | — | — | — | — |
| 2 sum-pool × 常数 | **不适用**（给出的不是 max） | — | — | — | — |
| 3 可分离 2D | 每平面 `out_H·(W_in·k_H + out_W·k_W)`：resnet-stem `56·(336+168)=28224` vs 基线 `56·56·9=28224` → **比 1.0**；vgg `4704` vs `3136` → **比 1.50（更差）**；alexnet `6642` vs `6561` → 比 1.01 | 同上（+ 行部分最大值缓冲 `out_H·W_in·CH`） | 1 | + `O(out_H·W_in·CH)` | 同上，但多一中间缓冲 + 两遍串行 |
| 4 前缀最大值 + 差分 | **不适用**（max 无逆元） | — | — | — | — |
| 6 退化特化（整平面） | **1 次向量归约**（vs 多次 `vmax`） | 同上 | 1 | `(CH,H_in,W_in)` 块 + 掩码块（容量判据见 §4.5） | 同上 |

> 口径含：GM 输入读 + 输出写；不含：UB 内搬运 / 转置 / NaN 追踪 / 守卫（片上，计入 §1.6.0 估算行的发射项）。
> **基线在表内**（#0）；选定候选 #0'。候选 1/2/4 因**数学不等价**不参与复杂度对比（标注 N/A）；
> 候选 3 经量化在后（见 §1.6.1）；候选 6 纳入（退化网格）。
> **注（max 的额外发射代价）**：以 eq 掩码探测 NaN 需每元素额外 `vcmp + vcast + vmul` 三条向量 op
> （avg 只需 1 条 `vadd`），故发射量约为同族 avg 的 4×；这是保持「NaN 胜出」语义的**必要代价**，非可省项。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线（标量） | **Vector 单元浪费**（逐元素标量比较，`PL-1.2` 同族实测窗口累加 98.3% 标量） | 寄存器 | — | 静态（`k_H`/`k_W` 编译期常量） | 无 | ❌ 计算单元失配（重设计为 #0'） |
| **0' 基线向量化（C 轴，选定）** | **Vector** ✓（`T.vmax` fp32 ✓ / `T.vcmp` fp32 ✓ / `T.vselect` fp32 ✓ / `T.vcast` ✓ / `T.vbrc` ✓） | 见 §4.5（代表档 ≈ 72 KB，≪ UB 192 KB，`CONST-capacity-910B2C`） | **C 轴恒为 8 的整数倍**（C=8/64/128/2048；fp16/bf16 向量宽 8、fp32 宽 4，8 两者皆是倍数）；`W_in`=55/56/112 仅作转置后 row 长度，不作向量轴，不要求整除 | ✓（`k_H`/`k_W` 编译期常量；动态量仅 `o_h`/`kh`/`ow`/`kw` 派生的标量守卫） | 无跨引擎流水；退化网格可 reduce 融合 | ✅ **选定** |
| 3 可分离 2D | Vector ✓ | 需 + 行部分最大值缓冲（`out_H·W_in·CH`） | 同 #0' | ✓ | 两遍串行（破坏 R2 单遍） | ❌ **运算量更大**（vgg 比 1.50；均 ≥ 1.0）+ 额外缓冲 |
| 6 退化特化 | Vector reduce ✓ | `(CH,H_in,W_in)` 块 + 掩码块（容量判据 + 回退，§4.5） | ✓ | ✓ | 单归约 | ✅ 采纳（充分条件成立时） |
| 1 / 2 / 4 | **不适用**（max 非线性 / 无逆元，非 MAC 密集） | — | — | — | — | ❌ 数学不等价，无需亲和评估 |

**R4 弃选理由的证据引用（negative-claim-evidence 口径）**：

- 候选 1/2/4：**数学事实**（$\max$ 非线性、无逆元），非 API / 性能断言——不作 API 检索需求
  （不存在可检索的「最大化简公式」）。
- 候选 3：运算量为**我方独立复算**（每平面 `out_H·(W_in·k_H + out_W·k_W)` vs `out_H·out_W·k_H·k_W`，
  见 R3 表；`algorithm-candidates.md` 的 `ALG-pooling` 未列可分离候选，本项为恒等变形自查所得）。
- 候选 0 的「标量占比」反例引用 pattern-library `layout.md` 的 `PL-1.2-caxis-accum` 与文档头「诊断触发器」
  （「窗口累加 98.3% 标量源于 `j·sW`（`sW≠1`）跨步系数阻碍向量化，原轴微调收益为 0，换轴后 5.6x」）。
- 候选 0'/6 的 dtype 支持矩阵来自 `docs/Tilelang.language/`（`T.vmax` / `T.vcmp` / `T.vselect` /
  `T.reduce_max` / `T.vcast` / `T.vbrc` / `T.transpose` 各 §2.2.1 表，见 §3.4）。
- **bf16 注意**：`T.vmax`/`T.reduce_max` 的 dtype 表**不含 bf16**（`docs/Tilelang.language/比较操作/T.vmax.md`
  §2.2.1、`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1），故 bf16 输入必须经 `T.vcast`
  上转 fp32 后计算——与「fp32 比较域」语义一致（§0.6 R2），无额外代价。

**设计期估算（D-2 roofline）** —— 协议见 `_shared/standards/hardware-cost-model.md` §2，常数引用 `pattern-library/constants.md`
（`CONST-mte2-degradation` HBM 3:1 混合读写地板 1.26 TB/s；`CONST-vector-launch-overhead` 0.5µs/op；
`CONST-capacity-910B2C` 192 KB/AIV）：

以 dispatch 代表 workload 计算（`CH` 按 §4.5 容量档）：

| workload | 流量项（`bytes ÷ 有效带宽`） | 发射项（向量 op 数） | 容量项 | 估算下界 |
|----------|------------------------------|----------------------|--------|---------|
| `smoke`(1,8,16,16, k3s2p1) | 5 120 B ÷ 1.26 TB/s = **4.1 ns** | 8 单元 × 110 op = **880 op** | 0 | **launch 主导**（见下注） |
| `resnet-stem`(32,64,112,112, k3s2p1) | 64 225 280 B ÷ 1.26 TB/s = **51.0 µs** | 3 584 单元 × 686 op = **2.46 M op**（CH=32） | 0 | **≈ 51 µs**（流量主导；发射项不确定） |
| `vgg-block`(16,128,56,56, k2s2p0) | 16 056 320 B ÷ 1.26 TB/s = **12.7 µs** | 896 单元 × 235 op = **0.21 M op**（CH=64） | 0 | **≈ 12.7 µs** |
| `alexnet-ceil`(32,64,55,55, k3s2p0 ceil) | 15 376 384 B ÷ 1.26 TB/s = **12.2 µs** | 864 单元 × 338 op = **0.29 M op**（CH=64） | 0 | **≈ 12.2 µs** |

> **估算下界（dispatch 代表 = `resnet-stem`）: ≈ 51 µs**（流量 51.0 + 发射「未文档化假设」 + 容量 0；
> 引用 `CONST-mte2-degradation` / `CONST-capacity-910B2C`）。**端到端实测地板大概率由固定开销主导**
> （NPU kernel launch + Event 计时 ~50–60 µs 量级，见 `examples/TileOPs/docs/gpu_to_npu_adaptation.md` C12 行）。
> **发射项口径注（未文档化假设）**：`CONST-vector-launch-overhead`（~0.5µs/op）是 **bn=512 宽块大 op** 的
> 实测值，对本设计 ≤128 元素的窄向量 op（本算子 op 宽度 = `CH` = 32/64）**不能直接外推**
> （按 0.5µs/op 会得荒谬量级）。窄向量 op 的每-op 代价**未逐点实测**，标注「未文档化假设 + 估算依据」；
> **方向证据已有实测**：`layout.md` 的 `PL-1.4-tiling-heuristic` 同族池化实测
> 「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs」（更宽 C 向量更优），与 `CONST-vector-launch-overhead`
> 「宽块优于窄块 op 翻倍」方向一致。**本算子的 op 数（≈ 元素数/CH × 4）显著高于同族 adaptive 算子**
> （`out_W` 达 56），故发射项**可能成为 `resnet-stem` 的实际主导项** —— 列入 §9 风险点与 §1.6.3 裁决计划。
> **容量项为 0**：本设计无 `store_fixpipe` 类跨引擎中间矩阵往返（纯 Vector 单 kernel）。
> **设备差异注**：`CONST-mte2-degradation` / `CONST-vector-launch-overhead` / `CONST-capacity-910B2C`
> 均绑定 **Ascend910B2C**（48 Vector 核）；本设备为 **Ascend910B3**（实查 20 × 2 = 40）。带宽常数按
> 「同代同 HBM 的量级下界」引用，**跨型号复用属未实测假设**（§9 风险点）。

**调研结论**：选定算法族 = **「直接窗口取最大值 + fp32 比较域 + C 轴向量化 + 显式向量 eq 掩码 NaN 追踪
- 核内融合转置（NCHW→C 最内）+ 在线单遍」**（R3 表候选 #0'；另采纳候选 #6 退化特化 → §1.6.1 **C4/C4b**，
候选 #5 融合扫描并入 #0'；两项采纳与 §1.6.1 优化结论**逐项一致**）。
关键依据：R3 表显示 #0' 的 FLOPs（= `N·C·out_H·out_W·k_H·k_W`，与 manifest `roofline.flops` 逐字一致）与
访存量（= manifest `roofline.bytes`）均为各候选最小量级（候选 3 不降反升；候选 1/2/4 数学不等价）；
R4 表显示 #0' 唯一同时满足「Vector 计算单元匹配 + C 轴向量整除 + 全静态框界 + 无跨核归约 + 容量充裕」。
**与基线的结构差异一句话**：数学公式、取值集合与 NaN 判决完全不变，执行单元由「每输出一线程的标量比较链」
改为「`(n,o_h,c_block)` 单元的 C 轴向量比较器 + 向量 eq 掩码 NaN 追踪器」。
**调研范围声明**（确有两项结构特化被采纳，但**无更低复杂度算法族**）：
本调研覆盖 ① `algorithm-candidates.md` 的 `ALG-pooling` 命中行全部候选
（R1 常数权重卷积 / sum-pool×常数 / R2 滑窗流式 / R3 窗口重叠重复读 / R4 C 轴向量轴）——
并**逐项判定其对 max 的适用性**（R1 命中项对 max 因非线性不成立）；
② 恒等变形空间（可分离 2D、前缀最大值差分、归约融合）；
③ pattern-library 同类实现（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum` / `PL-1.3-host-permute` /
`PL-1.4-tiling-heuristic`，均在 `layout.md`）；
④ 本仓 `examples/` 同类实现（§3.5.3）与**同族前作**（`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md`
等，CASE-pool-adaptivemaxpool2d-migration）。
**互联网检索未使用**（本算子算法族在本地参考表 + 源码 + pattern-library 已完整覆盖，
属简单窗口 / 归约类，无需联网补充）。
**下游约束**：§1.4 算法描述、§1.6.1–§1.6.3 的分析对象 = 选定算法 #0'；§0.5/§0.6 的处置与之**一致**
（§0.6 R2 = #0' 的归约向量化 + NaN 向量追踪、§0.6 R3 = C 轴布局、§0.6 R4 = 候选 #6）；
源算法 §0.4 已识别优化手段的意图承接逐项见 §0.6（**无静默丢弃**：11 项全部有处置，
#3/#4/#5/#6/#7/#11 的意图分别由 §0.6 R1/R2/R3 承接）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的算法 #0'（= 基线结构 + C 轴向量化 + 向量 eq 掩码 NaN 追踪）。
> R1 表中初判可采纳的项在此完成四要素论证。

**语义等价性锚点 E1（基线公式的 NPU 表述）**：源以「越界 lane 跳过」表达 padding；
NPU 表述为「累加器初值 $-\infty$ + 仅 fold in-bounds lane」。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **E1** | 语义等价锚点：越界跳过 ≡ $-\infty$ padding + 累加器初值 $-\infty$ | `if in_bounds: max_val = max(max_val, val)`，`max_val` 初值 $-\infty$ | `acc ← -inf`；仅对 in-bounds lane 做 `acc ← max(acc, …)` | **数学恒等**：$\max(-\infty,x)=x$，跳过 lane 与 fold 一个 $-\infty$ 逐位等价；已由 `verify_equiv.py` E1 组（vs `F.max_pool2d`，fp64）机器验证为**逐位一致** | 无（表达式改写，非降本项）——作为后续优化项的语义锚 |

**采纳的数学等价优化项**：

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **C1** | 守卫外提 / 公共子表达式消除（H 方向） | 逐元素判据 `0<=ih<H_in and 0<=iw<W_in`（每输出元素 `k_H·k_W` 次，`ih` 只依赖 `(o_h,k_h)`） | H 判据 `0<=ih<H_in` **提升到每行一次**（`k_H` 次 / 单元）；W 判据 `0<=iw<W_in` 保留逐 lane（`iw` 依赖 `(o_w,k_w)`，不可再提） | **数学恒等**：`ih = o_h*s_H-p_H+k_h*d_H` 对 `k_w` 无依赖，是 `k_w` 循环不变量；条件合取可重排（逻辑等价）。已由 `verify_equiv.py` C1 组机器验证为**逐位一致** | H 判据求值次数由 `out_W·k_H·k_W` 降至 `k_H`（resnet-stem 每单元 504 → 3，−99.4%）；W 判据不变 |
| **C4** | 退化网格结构特化：整平面归约 | 通用双循环取最大值（窗口覆盖整平面时仍逐窗口 fold） | 当 **`out_H==out_W==1` 且 `p_H==p_W==0` 且 `d_H==d_W==1` 且 `(k_H-1)·d_H+1 >= H_in` 且 `(k_W-1)·d_W+1 >= W_in`**（五条件缺一不可）：按 `(CH, H_in, W_in)` 连续块搬运 + `T.vcast` fp32 + `T.reduce_max(dim=2→1)`；NaN 用 `vcmp(eq)`+`vcast`+`reduce_max` 掩码归约 + `vselect` 判决 | **数学恒等（条件成立时）**：`p_H=p_W=0` 且 `d_H=d_W=1` 时窗口 $=\{k_h\}\times\{k_w\}$，覆盖长度条件保证 $\{0,\dots,k_H-1\}\supseteq[0,H_{in})$、$\{0,\dots,k_W-1\}\supseteq[0,W_{in})$，故窗口**恰为** $[0,H_{in})\times[0,W_{in})$，`out[·,·,0,0] = max_{all} x[n,c,:,:]`，取值集合完全相同 ⇒ **逐位等价**（max 只选择，无算术）。**`d==1` 项由 Stage 2 检视补入**（反例见 §0.6 R4；`verify_equiv.py` C4x 组 6 行机器锁住） | 每平面多次向量 `vmax` → **1 次向量归约**（resnet 式全局配置下 `k_H·k_W` 次 → 1 次）+ 省去转置链。**注：不被当前 manifest workloads 触发** |
| **C4b** | 退化网格结构特化：恒等拷贝 | 通用双循环（每窗口 max 单元素） | 当 **`k_H==k_W==1 and d_H==d_W==1 and p_H==p_W==0 and s_H==s_W==1`**：`out = x`（纯 UB 中转拷贝，同 dtype 不 cast） | **数学恒等（条件成立时）**：窗口恰为单元素 `{o_h,o_w}`，`max` 单元素 = 元素本身 ⇒ 逐元素恒等（NaN 也随元素原样搬出，语义一致） | 每平面 `H_in·W_in` 次比较 + NaN 追踪（4 op/元素）→ **1 次拷贝**。**注：不被当前 manifest workloads 触发** |

> **无「倒数乘 / 归一化」优化空间**（显式结论）：与同族 avg 池化不同，max 池化**无除法 / 归一化**
> （结果即被选元素），故 avg 迁移的 C1（`sum/d → sum·(1/d)`）在本算子**不存在**——§1.6.1 无对应项，非遗漏。
> 亦**无高代价算子降代换**（无 exp / rsqrt 类）、**无多处出现的公共子表达式**（每元素仅参与一次比较；
> 唯一可 CSE 的是守卫判据，已由 C1 采纳并前移，见上）。
> **可分离 2D 不采纳（量化理由）**：R3 表显示其运算量对 manifest 三档配置**均 ≥ 基线**
> （resnet-stem 1.00、vgg 1.50、alexnet 1.01），且引入第二遍扫描（破坏 R2 单遍结论）与额外中间缓冲 ⇒ 弃。
> **候选 1/2/4 不采纳**：数学不等价（§1.6.0 R1）。

**等价性机器验证（D-1）**：采纳项与语义锚点全部由
`examples/max_pool2d/_max_pool2d_kernel/verify_equiv.py` 在 **fp64 参照**下对照验证（torch CPU）：

- **E1**：NPU 基线公式（`-inf` 初值 + 越界跳过）vs `F.max_pool2d`（判据 = 舍入到目标 dtype 后逐位一致）；
  含一条 **`ceil_mode` step-back** 用例（`(1,8,8,8)` k3s4p0 ceil → `out=2`，触发 helper 的末窗回退分支）；
- **C1**：守卫外提结构 vs 逐元素守卫基线（逐位一致）；
- **C4/C4b**：特化公式 vs 通用窗口基线（**在各自充分条件成立的配置上**，逐位一致）；
  C4 组 5 条配置**全部满足修正后的五条件（含 `d==1`）且输出非空**（原空真用例 `(1,8,5,6) k8s2` 已删除，见下注）；
- **C4x**（Stage 2 修订新增）：**dilation 反例组**——`(1,8,5,5)` k2s1p0 d4 与 `(1,8,7,7)` k3s1p0 d3，
  均为 `out=1×1` 但 **`d>1`，断言「不满足 C4 条件、必须走一般路径」**，并与 `F.max_pool2d` 逐位对照。
  这 6 行是本设计对 REVIEW 问题 1 的回归守卫：把 `d==1` 约束从 C4 充分条件中移除任一轴即会让
  该组（以及末尾的 HAZARD 演示行）失败；
- 用例：每 case-group × 6 种数据（`rand / nan / inf（含 ±inf） / subnormal / large / zeros`）× 3 dtype
  （fp16/bf16/fp32）；NaN-vs-NaN 计相等，NaN-vs-finite 恒判失败。
  **空真防护**：`mismatch()` 断言两侧 `numel() > 0` 且 shape 相同——原 `c4_cfg[4]`（`(1,8,5,6)` k8s2
  → `out=(0,0)`，比较 0 个元素）已替换为满足 C4 条件的 `(1,8,8,8)` k8s1p0（`out=1×1`）。
- **运行结论**：`viol=0` 全部 **69** 个 case-group → `[EQUIV] ALL PASS — 69 case-groups, 0 failures`
  （另打印 `[C4x-GUARD]` 反例演示：一般路径 = 0.0 == golden，整平面 = 1.0 被 `d==1` 守卫排除）。

内嵌结果表（**由脚本 stdout 逐行原样粘贴**；`worst|d|` = 缩放后最大绝对差在「不一致元素」上的上界；
`viol` = 不一致元素数）：

```
check dtype     shape            kH,kW   kinds  worst|d|    viol   status
--------------------------------------------------------------------------------------------
E1    float16   (1, 8, 16, 16)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 8, 16, 16)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 8, 16, 16)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    float16   (2, 4, 20, 20)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (2, 4, 20, 20)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    float32   (2, 4, 20, 20)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    float16   (1, 4, 23, 27)   (3, 5)  6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 4, 23, 27)   (3, 5)  6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 4, 23, 27)   (3, 5)  6      0.0000e+00  0      EQUIV_PASS
E1    float16   (2, 4, 21, 21)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (2, 4, 21, 21)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    float32   (2, 4, 21, 21)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    float16   (1, 8, 12, 14)   (2, 2)  6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 8, 12, 14)   (2, 2)  6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 8, 12, 14)   (2, 2)  6      0.0000e+00  0      EQUIV_PASS
E1    float16   (1, 8, 8, 8)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    bfloat16  (1, 8, 8, 8)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
E1    float32   (1, 8, 8, 8)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 16, 16)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 16, 16)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 16, 16)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (2, 4, 20, 20)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (2, 4, 20, 20)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (2, 4, 20, 20)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 4, 23, 27)   (3, 5)  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 4, 23, 27)   (3, 5)  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 4, 23, 27)   (3, 5)  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (2, 4, 21, 21)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (2, 4, 21, 21)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (2, 4, 21, 21)   (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 12, 14)   (2, 2)  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 12, 14)   (2, 2)  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 12, 14)   (2, 2)  6      0.0000e+00  0      EQUIV_PASS
C1    float16   (1, 8, 8, 8)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    bfloat16  (1, 8, 8, 8)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C1    float32   (1, 8, 8, 8)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C4    float16   (1, 8, 4, 4)     (4, 4)  6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (1, 8, 4, 4)     (4, 4)  6      0.0000e+00  0      EQUIV_PASS
C4    float32   (1, 8, 4, 4)     (4, 4)  6      0.0000e+00  0      EQUIV_PASS
C4    float16   (2, 4, 7, 9)     (7, 9)  6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (2, 4, 7, 9)     (7, 9)  6      0.0000e+00  0      EQUIV_PASS
C4    float32   (2, 4, 7, 9)     (7, 9)  6      0.0000e+00  0      EQUIV_PASS
C4    float16   (1, 8, 16, 16)   (16, 16) 6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (1, 8, 16, 16)   (16, 16) 6      0.0000e+00  0      EQUIV_PASS
C4    float32   (1, 8, 16, 16)   (16, 16) 6      0.0000e+00  0      EQUIV_PASS
C4    float16   (2, 16, 7, 7)    (7, 7)  6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (2, 16, 7, 7)    (7, 7)  6      0.0000e+00  0      EQUIV_PASS
C4    float32   (2, 16, 7, 7)    (7, 7)  6      0.0000e+00  0      EQUIV_PASS
C4    float16   (1, 8, 8, 8)     (8, 8)  6      0.0000e+00  0      EQUIV_PASS
C4    bfloat16  (1, 8, 8, 8)     (8, 8)  6      0.0000e+00  0      EQUIV_PASS
C4    float32   (1, 8, 8, 8)     (8, 8)  6      0.0000e+00  0      EQUIV_PASS
C4x   float16   (1, 8, 5, 5)     (2, 2)  6      0.0000e+00  0      EQUIV_PASS
C4x   bfloat16  (1, 8, 5, 5)     (2, 2)  6      0.0000e+00  0      EQUIV_PASS
C4x   float32   (1, 8, 5, 5)     (2, 2)  6      0.0000e+00  0      EQUIV_PASS
C4x   float16   (1, 8, 7, 7)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C4x   bfloat16  (1, 8, 7, 7)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C4x   float32   (1, 8, 7, 7)     (3, 3)  6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (1, 8, 4, 4)     (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (1, 8, 4, 4)     (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (1, 8, 4, 4)     (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (2, 4, 9, 11)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (2, 4, 9, 11)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (2, 4, 9, 11)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (1, 16, 5, 7)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (1, 16, 5, 7)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (1, 16, 5, 7)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   float16   (3, 33, 6, 8)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   bfloat16  (3, 33, 6, 8)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS
C4b   float32   (3, 33, 6, 8)    (1, 1)  6      0.0000e+00  0      EQUIV_PASS

[C4x-GUARD] dilation counterexample (H=W=5,k=2,s=1,p=0,d=4,x[2,2]=1):
[C4x-GUARD]   general path = 0.0  == golden 0.0;  whole-plane = 1.0 (excluded by the d==1 guard)
[C4x-GUARD]   OK — corrected C4 condition rejects d>1

criterion: exact — 0 element mismatches after rounding to the target
dtype (NaN-vs-NaN counts equal; NaN-vs-finite always fails).
[EQUIV] ALL PASS — 69 case-groups, 0 failures
```

> 完整逐条输出（含每 case-group 的 6 种数据 kind）可复现：
> `/workspace/venv/bin/python examples/max_pool2d/_max_pool2d_kernel/verify_equiv.py`
> （脚本随算子目录留存，Stage 2 维度 8 会重跑复核。**Stage 2 修订 v1**：C4 组空真用例 `(1,8,5,6) k8s2`
> 换为满足修正条件的 `(1,8,8,8) k8s1p0`，新增 C4x dilation 反例组（6 行）与 E1 的 ceil step-back
> 用例（3 行），`mismatch()` 加空真断言——case-group 总数 57 → **69**。）

**优化结论**：采纳 **3 项**——**C1**（守卫外提 / CSE，逐位等价）、**C4**（整平面归约特化，**五条件含
`d_H==d_W==1`** 成立时逐位等价）、
**C4b**（恒等拷贝特化，充分条件成立时逐位等价），并建立 **E1** 语义锚点；三者与锚点均已机器验证（69 case-group 全 `EQUIV_PASS`）。
候选「可分离 2D」经量化后不采纳（manifest 三档运算量均 ≥ 基线 + 破坏单遍）；候选 1/2/4 因**数学不等价**不采纳；
「倒数乘」优化在本算子**不存在**（max 无归一化）。**优化后公式（供 §3.1 公式拆解使用）**：

```
一般网格 (非退化):
  acc[c] ← -inf ; flag[c] ← 1                                  # 单位元 -inf；NaN 免标记
  for kh in [0,k_H) with 0 <= o_h*s_H-p_H+kh*d_H < H_in :       # H 守卫每行一次 (C1)
    row ← x[n, c, o_h*s_H-p_H+kh*d_H, 0:W_in]                   # 行片搬运
    for kw in [0,k_W), ow in [0,out_W) with 0 <= ow*s_W-p_W+kw*d_W < W_in :
      acc[ow]  ← max(acc[ow], row[ow*s_W-p_W+kw*d_W])           # C 向量 max
      flag[ow] ← flag[ow] * (row[..] == row[..] ? 1 : 0)        # eq 掩码乘性 NaN 追踪
  out[n,c,o_h,ow] = (flag[ow] == 0) ? NaN : acc[ow]             # NaN 胜出判决
退化网格 (充分条件成立时):
  C4  (out_H==out_W==1, p_H==p_W==0, d_H==1, d_W==1,           # d==1 必要（Stage 2 修正）
       (k_H-1)d_H+1>=H_in, (k_W-1)d_W+1>=W_in):
        out[n,c,0,0] = max_{all i_h,i_w} x[n,c,:,:]              # 同 NaN 判决
  C4b (k_H==k_W==1, d_H==d_W==1, p_H==p_W==0, s_H==s_W==1):
        out[n,c,:,:] = x[n,c,:,:]
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案（§1.4 + §3.3 伪代码 + §6）中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
> 替代方案所用 API 均以 `docs/Tilelang.language/` 条目佐证（下表「佐证」列）。

| # | 计算点 | 原实现形态（源 GPU） | 向量替代方案 | 是否替代 | 不可替代理由 / 佐证 |
|---|--------|---------------------|-------------|---------|--------------------|
| 1 | 累加器初始化 `acc ← -inf` | 源 `max_val = T.cast(-inf)` 标量（每输出元素 1 次） | `T.vbrc(-T.infinity("float32"), acc)`（**整 buffer** 广播 $-\infty$） | ✅ | 佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（scalar→任意 shape 广播）、`docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（fp16/fp32 ✓）。**dst 必须是整 buffer**——region 形态会被静默丢弃（§9.2） |
| 2 | NaN 标记初始化 `flag ← 1` | 源 `has_nan = False` | `T.vbrc(T.float32(1.0), flag)`（整 buffer）；标量 operand 必须是 **typed PrimExpr** | ✅ | 同上；裸 Python 数命中前端 rank 断言（§9.2） |
| 3 | 窗口取最大值 `max_val = T.max(max_val, val)` | 逐元素标量比较（每输出元素 `k_H·k_W` 条） | `T.vmax(acc[ow:ow+1,:], row_f32[iw:iw+1,:], acc[ow:ow+1,:])`（C 轴向量，原地） | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1（fp32 ✓；**bf16 ✗**，须先上转）；`PL-1.2-caxis-accum`（本仓实测切片累加形态） |
| 4 | 逐元素 `T.cast(x[...],"float")` | 每元素一次标量 cast | `T.vcast(row_dst, row_f32, round_mode="rint")`（整 tile，形状一致、源为 UB） | ✅ | 佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint` ✓）、§2.2.2/§2.3（形状一致 + 源须 UB） |
| 5 | **NaN 追踪** `has_nan \| T.isnan(val)` | 每元素标量 `isnan` + 标量 bool 或 | `T.vcmp(row_f32[iw:iw+1,:], row_f32[iw:iw+1,:], cond, "eq")`（$x{=}x$ ⟺ 非 NaN，向量 bool）+ `T.vcast(cond, mf32, "rint")` + `T.vmul(flag[ow:ow+1,:], mf32, flag[ow:ow+1,:])` | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1（fp32 ✓，输出 bool 片上）+ `docs/Tilelang.language/条件操作/T.vselect.md` + `docs/Tilelang.language/数学操作/T.vmul.md` §2.3（支持原地）。**必须用 `eq`（`ne` 为 ordered-NE，NaN 恒假，不可作 NaN 探测）** |
| 6 | 结果 `if_then_else(has_nan, NaN, max_val)` | 每元素标量分支 | `T.vcmp(flag, zero, cfinal, "eq")` + `T.vselect(cfinal, nanbuf, acc, out32)` | ✅ | 佐证同 #5；`nanbuf = inf - inf`（`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1 fp32 ✓、`docs/Tilelang.language/创建操作/T.infinity.md`） |
| 7 | 最终输出 cast | 每元素一次标量 cast | `T.vcast(out32, acc16, round_mode="rint")`（整 tile） | ✅ | 同上 #4 |
| 8 | 布局重排（C→最内） | （源无；GPU 靠 coalesced 尾轴） | `T.transpose(src, dst, [1,0])` | ✅ | 佐证：`docs/Tilelang.language/创建操作/T.transpose.md` §2.3（二轴交换 ✓，fp16/bf16/fp32 ✓）；`PL-1.1-transpose-chain` |
| 9 | **窗口 H 循环 `for kh in T.serial(k_H)`** | `T.serial` 静态界 + 每行守卫（C1 后） | 无（保持 `T.serial` + `if 0<=ih<H_in`） | ❌ | **理由（tile 级顺序依赖）**：`kh` 是累加链的循环携带维（跨 `kh` 累加到同一 `acc`/`flag`）；`k_H` 是编译期常量故上界静态。**循环体内已全部向量化**（#3/#4/#5），故该循环是 §6 的 block/tile 级调度循环，非逐元素标量循环 |
| 10 | **窗口 W 循环 `for kw in T.serial(k_W)` + 输出列 `for ow in T.serial(out_W)`** | `T.serial` 静态界 + 逐 lane 守卫 | 无（保持 `T.serial` + `if 0<=iw<W_in`） | ❌ | **理由（tile 级顺序依赖 + 运行时索引）**：`kw` 是累加链的循环携带维；`ow` 是 `acc`/`flag` 的运行时行索引（`out_W` 编译期常量，但 `ow` 取值运行时）。循环体为 C 向量 op（切片 `[ow:ow+1, :]`），与 `PL-1.2-caxis-accum` / 同族前作实现同形 |
| 11 | 单元任务解码 `(n, o_h, c_block)` | — | 无 | ❌ | **理由（block 级索引 / 任务映射计算）**：每单元少量整数 `%`/`//`，非逐元素热点；无对应逐元素向量 API |
| 12 | 输出尺寸 / `CH` / `num_logical` / `num_kernels` / `num_local` | — | 无 | ❌ | **理由（host/trace 侧元数据）**：factory 层 Python 整数算术（编译期常量），不在 kernel 运行时逐元素执行 |
| 13 | host 侧参数规范化 / kernel 缓存 | — | 无 | ❌ | **理由（host 侧元数据）**：只改 shape/stride 视图与缓存键，符合 `ascend-constraints.md` §4；不在 kernel 内 |
| 14 | `cfinal` 与 `zero`/`nanbuf` 的单元级构造 | — | `T.clear(zero)`、`T.vbrc(inf, nanbuf)`+`T.vsub(nanbuf,nanbuf,nanbuf)`（整 buffer op，非逐元素循环） | ✅ | 佐证：`docs/Tilelang.language/内存操作/T.clear.md` §2.2（fp32 ✓）；`T.vbrc` §2.2.2；`T.vsub` §2.3（原地） |

**向量化结论**：逐元素计算（累加器/标记初始化 / 取最大值 / NaN 探测 / NaN 判决 / cast / 布局重排 / 单元级常量构造）
**已全部向量化**（含 §3.3 单元入口的 `acc`/`flag` 初始化 → **#1/#2**），替代方案 API 均有 `docs/` 条目佐证。
保留 **4 类** 循环 / 标量点，理由类别：**#9/#10** tile 级顺序依赖（循环携带累加维 + 运行时行索引，
循环体内已向量化）/ **#11** block 级任务映射 / **#12/#13** host 侧 trace 元数据（不在 kernel 内）。
无「以实现简单为由保留标量循环」的情形。**§6 循环结构中不出现逐元素标量循环**（与本结论一致）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（4D NCHW 进 / NCHW 出），但**核内布局与 lane 映射是自由变量**。
> 本节枚举「窗口 / 池化类」必选候选（reference：algorithm-research.md Phase 2 第 3 项
> 「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴」）。
> 迁移任务：源的并行轴选择（GPU 逐元素 lane 扁平映射）是**输入而非结论**，对照 §0.3 独立评估。

**轴质量评分维度**：

| 评分项 | 说明 | 候选 1（NCHW / W 轴） | 候选 2（核内 NHWC / C 轴） | 候选 3（host permute / C 轴） |
|--------|------|----------------------|---------------------------|------------------------------|
| 整除性 | 轴长 vs 向量宽度（fp16/bf16 ×8、fp32 ×4） | `W_in`=55/56/112：56/112 ✓，55 ✗（尾 lane 浪费 1.8%）；且**窗口沿 W 的步进 `s_W`=2/3 ≠ 1** | **C 恒为 8 的整数倍**（8/64/128/2048；8 同是 fp32 ×4 的倍数）✓ | 同候选 2 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | W 轴 1.8%；**窗口累加的跨步系数 `k_w·s_W`（`s_W≠1`）使向量化退化为标量** | **0%** | 0% |
| 累加链形态 | 该轴作串行累加维（stride≠1）还是向量维（stride=1） | W 轴作归约维需**跨步窗口最大值**（`iw=o_w·s_W-p_W+k_w·d_W`，`s_W≠1`）；`PL-1.2` 同族实测该形态 98.3% 标量 | C 轴作向量维（转置后 stride=1 连续）；空间轴（`k_h`/`k_w`）作串行归约维——`max` 结构天然 | 同候选 2 |
| repack 代价 | 布局重排代价 | 无 repack，但需跨步窗口处理 | 核内融合转置链（`T.transpose` 二轴交换，`PL-1.1` 实测 4 次合计 ~5.4µs） | **host permute 106–147µs**（`PL-1.3`，已量化证伪） |
| UB 容量影响 | 布局对驻留缓冲的影响 | 小（row `(W_in)` 级） | 代表档 ≈ 72 KB（vgg/alexnet）/ ≈ 122 KB 膨胀后（§4.5），对照 192 KB | 同候选 2 |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|----------------|---------|
| 1 | I/O 原生 NCHW | W（行内连续） | 无 | 基线：窗口步进 `s_W`=2/3≠1，跨步窗口最大值需逐元素或 gather（`T.reduce_max` 不支持 bf16）；`W_in`=55 非对齐 | ❌ 量化理由：`PL-1.2`/`PL-1.4` 同族实测「`sW≠1` 跨步系数阻碍向量化 → 98.3% 标量，原轴微调收益 0，换轴后 5.6x」；且 bf16 须先上转 |
| 2 | **核内 NHWC（融合转置链）** | **C（恒为 8 的整数倍）** | 逐行二轴 `T.transpose(src,dst,[1,0])`（`PL-1.1` 族，实测 ~µs 级） | 尾 lane 0%；转置代价 ~µs 级（远低于算子本体 12–51µs）；UB 增量见 §4.5 | ✅ **选定** |
| 3 | host permute NCHW↔NHWC 进出 | C | 两次 host permute | **106–147µs**（`PL-1.3`，实测）> 多数 workload 本体耗时；host 侧连续化会产生新张量（无收益） | ❌ 量化理由：repack 代价（106–147µs）> 算子本体流量（12–51µs）数倍，净亏 |
| 4 | Cube / implicit-GEMM（0/1 权重矩阵） | Cube N 轴（`out_W`） | 权重矩阵构造 | **数学不可行**（max 非线性，无 0/1 权重矩阵可表达）；且 `out_W`=8 < 16 违 Cube 分形下限 | ❌ 见 §1.6.0 R1/R4（数学不等价优先于亲和评估） |

**布局决策结论**：选定 **方案 #2：核内布局 = 逐行 NHWC（C 最内连续）、向量化轴 = C、repack 路径 =
核内逐行二轴 `T.transpose`（`[1,0]`）+ 输出侧 `T.transpose([1,0])` 回转**。
该结论已同步至：§3.3 伪代码（`row_src/row_dst/row_f32/acc/flag` 形状与内层向量维 = C）、
§4 内存规划（§4.3/§4.5 的 buffer 形状与预算）、§6 循环结构（累加内层向量维 = C）。
弃选方案量化理由见候选矩阵 #1/#3/#4。**迁移任务注**：源 GPU 的并行轴是「输出元素扁平 lane」
（§0.3），与 NPU Vector 轴不对应，此处已独立评估并替换为 C 轴（§0.6 R1）。

**D-3 实验裁决模式（主选 + 备选 + 裁决计划）**：主选与备选的分歧落在**窄向量 op 的每-op 发射代价**
的**外推区间**——`CONST-vector-launch-overhead` 只给了 bn=512 大 op 的 0.5µs/op 实测值，
本设计 op 宽度 = `CH`（32/64），未逐点实测；且本算子每元素比同族 avg 多 3 条向量 op（NaN 追踪）、
`out_W` 达 56 使 op 数放大，该外推更敏感。
**方向证据**：`layout.md` 的 `PL-1.4-tiling-heuristic` 直接实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs
——更宽 C 向量 + 更少 block 胜过更高空间并行度」，与 `CONST-vector-launch-overhead` 方向一致；
主选「容量允许的最大 CH（代表档 = 64，受 UB 约束的 resnet-stem = 32）」恰落在 `PL-1.4` 已实测范围
（CH16/32/64）内，故主选以实测条目为主证据。

- **主选（进 Stage 3）**：`CH =` 容量允许的最大 8 的倍数（`≤ min(C,64)`，判据见 §4.5）；
  单元 = `(n, o_h, c_block)`；`num_logical > vector_cores` 时 persistent（见 §5.5）。
  代表值：`resnet-stem` CH=32 → 3 584 单元（persistent，40 核 × 90 任务）；`vgg-block` CH=64 → 896 单元（×23）；
  `alexnet-ceil` CH=64 → 864 单元（×22）；`smoke` CH=8 → 8 单元（无适配）。
  **依据**：`PL-1.4` 实测 CH64 为该池化族最优档（21.6µs，优于 CH32/CH16）；本设计 C 轴向量化形态同族。
- **备选（≤1 个，结构完整，可直接实现）**：**主选 CH 的一半**（resnet-stem → CH=16；vgg/alexnet → CH=32），
  即把 op 数放大 2×、核数放大 2×。这是「发射代价主导 vs 吞吐主导」的另一端；buffer 形状与主选相同仅 CH 减半
  （UB 预算同步缩小），分核三要素按新 `num_logical` 重算（resnet-stem CH=16 → 7 168 单元，persistent 40 核 × 180 任务）。
  （备选存在的意义**仅限于**：验证比主选更窄的 CH 是否使趋势翻转——`PL-1.4` 未覆盖 CH<16，该点属未实测外推。）
- **裁决计划**：代表 shape = `resnet-stem`、`vgg-block`（各 1 组）；指标 = msprof op Task Duration；
  测量未知常数 = 窄向量 op 的每-op 代价（用 `perf_opt` 下两种 CH 的实测反解）；
  判定阈值 = **备选 latency < 主选 × (1 − 0.10) 才翻转**（10% 稳健性门槛）；
  裁决结果回写路径 = Stage 4 实测数据经附录补记固化 §1.6.3 判定依据，若翻转则按 `perf-feedback.md` 受控路由回写 DESIGN.md。
- **shape 分派评估**：本 kernel 由 `functools.lru_cache` 按 `(n,c,h,w,k,s,p,d,ceil,dtype)` 特化，
  **支持按 shape 分派**主选/备选——裁决计划须评估「大 `W_in`（UB 受限，如 resnet-stem）走主选 CH=32；
  小 `W_in` 走更大 CH」的**按负载分派**（而非全局二选一）。
- **BH（每单元输出行数）说明**：主选 **BH=1**（单元 = 单输出行 `o_h`），依据 `PL-1.4` 的「BH=1 + 最宽 CH 最优」
  实测结论；`s_H < k_H` 时的行重复读放大（§0.6 R3，resnet-stem ≈ 1.5×）**已计入 §1.6.0 R3 口径**，
  列为 Stage 4 可选探索项（BH>1 复用输入行），不改主结构。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（迁移 prompt 指定；`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`）。

### 2.2 选型理由

- 算子是**纯 Vector**（无 matmul / 无 Cube），仅需逐元素向量运算（`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`/
  `T.vcast`）、广播（`T.vbrc`）、清屏（`T.clear`）与布局重排（`T.transpose`）——正是 Developer 模式的目标域
  （`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`、
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/` 均为 Developer）。
- 无跨引擎流水、无手动 L1/L0 分级需求（`T.alloc_ub` 直映 UB）；编译器管理 UB 分配与自动同步（§7）。
- 迁移 prompt 明确要求 developer 模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 环境变量 | `TILELANG_ASCEND_MODE=Developer` |
| 内存分配 | `T.alloc_ub(...)`（GM→UB 路径） |
| 计算方式 | v-prefix 向量 API（`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`/`T.vcast`/`T.vbrc`/`T.clear`）+ `T.transpose` + `T.copy` |
| 同步 | 编译器自动同步（§7）；无手动 `T.set_flag`/`T.wait_flag` |
| Kernel 维度 | 一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `out_H ← pool_output_dim(H_in,k_H,s_H,p_H,ceil_mode,d_H)`（`out_W` 同理） | trace 期 host 侧；**不得重算**（逐字引用源 helper） |
| 2 | `row_src[c, j] ← x[n, c0+c, o_h·s_H-p_H+k_h·d_H, j]` | 行片搬运 `(real_c, W_in)`（H 守卫后） |
| 3 | `row_dst[j, c] ← row_src[c, j]` | 核内二轴转置 → C 最内 |
| 4 | `row_f32[j, c] ← cast_f32(row_dst[j, c])` | 整 tile fp32 上转（`rint`） |
| 5 | `acc[o_w, c] ← max(acc[o_w, c], row_f32[o_w·s_W-p_W+k_w·d_W, c])` | C 轴向量取最大值（C1 守卫后） |
| 6 | `cond[o_w, c] ← (row_f32[iw, c] == row_f32[iw, c])`；`flag[o_w,c] ← flag[o_w,c] · cast_f32(cond)` | NaN 追踪（`x=x` + 乘性累积） |
| 7 | `cfinal[o_w,c] ← (flag == 0)`；`out32[o_w,c] ← cfinal ? NaN : acc[o_w,c]` | NaN 胜出判决 |
| 8 | `acc_t[c, o_w] ← cast_dtype(out32[o_w, c])` | 单次舍回 + 转置回转 |
| 9 | `out[n, c0+c, o_h, o_w] ← acc_t[c, o_w]` | 行片写回 `(real_c, out_W)` |
| C4 | `out[n,c,0,0] ← max_{i_h,i_w} x[n,c,i_h,i_w]`（NaN 掩码归约判决） | 充分条件成立时的整平面归约 |
| C4b | `out[n,c,i_h,i_w] ← x[n,c,i_h,i_w]` | 充分条件成立时的恒等拷贝 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | `acc ← -inf` | `T.vbrc(src, dst)` | `src = -T.infinity("float32")`（typed 标量），`dst = acc`（**整 buffer**） | Developer |
| 0b | `flag ← 1` | `T.vbrc(src, dst)` | `src = T.float32(1.0)`（typed 标量），`dst = flag`（整 buffer） | Developer |
| 0c | `zero ← 0` / `nanbuf ← NaN` | `T.clear(buf)` / `T.vbrc`+`T.vsub` | `T.clear(zero)`；`T.vbrc(T.infinity("float32"), nanbuf)` + `T.vsub(nanbuf, nanbuf, nanbuf)`（`inf-inf = NaN`） | Developer |
| 2 | 行片 GM→UB | `T.copy(src, dst)` | `src = x[n, c0:c0+real_c, ih, 0:W_in]`，`dst = row_src[0:real_c, 0:W_in]` | Developer |
| 3 | 二轴转置 | `T.transpose(src, dst, permutation)` | `permutation=[1,0]` | Developer |
| 4 | fp32 上转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"`（同形、源为 UB） | Developer |
| 5 | C 轴取最大值 | `T.vmax(src0, src1, dst)` | 切片 `(1,CH)`：`src0=acc[ow:ow+1,:]`，`src1=row_f32[iw:iw+1,:]`，`dst=acc[ow:ow+1,:]`（原地） | Developer |
| 6a | NaN 探测（IEEE eq） | `T.vcmp(A, B, C, cmp_mod)` | `A=B=row_f32[iw:iw+1,:]`，`C=cond[0:1,:]`（bool 片上），`cmp_mod="eq"` | Developer |
| 6b | eq 掩码转 f32 | `T.vcast(src, dst, round_mode)` | `src=cond`，`dst=mf32`，`round_mode="rint"`（i1→f32 ✓） | Developer |
| 6c | NaN 累积 | `T.vmul(src1, src2, dst)` | `src1=dst=flag[ow:ow+1,:]`，`src2=mf32`（原地） | Developer |
| 7a | has_nan 判决 | `T.vcmp` | `A=flag`，`B=zero`，`C=cfinal`，`cmp_mod="eq"` | Developer |
| 7b | `has_nan ? NaN : max` | `T.vselect(Cond, A, B, Out)` | `Cond=cfinal`，`A=nanbuf`，`B=acc`，`Out=out32` | Developer |
| 8a | dtype 回转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"` | Developer |
| 8b | 输出转置回转 | `T.transpose(src, dst, [1,0])` | `(out_W,CH)→(CH,out_W)` | Developer |
| 9 | 行片 UB→GM | `T.copy(src, dst)` | `src = acc_t[0:real_c, 0:out_W]`，`dst = out[n, c0:c0+real_c, o_h, 0:out_W]` | Developer |
| C4 | 全平面取最大值 | `T.reduce_max(buffer, out, dim)` | `(CH,H_in,W_in)→(CH,H_in,1)`（dim=2）→`(CH,1,1)`（dim=1），`clear=True` | Developer |
| C4 | NaN 掩码归约 | `T.vcmp`+`T.vcast`+`T.reduce_max` | `mb=vcmp(eq)`；`mf=vcast(i1→f32,rint)`；`reduce_max` → `nmax ∈ {0,1}` | Developer |
| C4 | NaN 判决 | `T.vcmp`+`T.vselect` | `cond=vcmp(nmax,zero,"gt")`；`out32=vselect(cond,nanbuf,gmax)` | Developer |
| C4b | 恒等拷贝 | `T.copy` | `x[n, c0:c0+real_c, 0:H_in, 0:W_in] → stage → out[n, c0:c0+real_c, 0:H_in, 0:W_in]` | Developer |
| 输出尺寸 | `out_H/out_W` | Python `pool_output_dim`（host/trace） | 逐字复用源 helper | host |
| 通道块 | `CH` | factory 层容量判据（§4.5） | `CH = max multiple-of-8 ≤ min(C,64)` s.t. 预算 ≤ 192KB | host |

### 3.3 计算伪代码

> 注（迁移模板要求）：`T.Tensor` 的数据类型以**第二个位置参数**传入，不用 `dtype=` 关键字。
> 注：`block_m` 为源接口兼容参数（K9 保留），NPU tiling 由 shape 派生，其值被校验后忽略。

```python
# 迁移决策后的 NPU 算法伪代码（Developer 模式；C 轴向量化 + eq 掩码 NaN 追踪 + 核内转置）
@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3", "-DENABLE_BF16"])
def _build(block_m=None):
    # 退化特化编译期谓词（纯名 Python bool，trace 时折叠；二者互斥且均须过 §4.5 容量判据）
    # 注意：IS_GLOBAL 的 d_H==d_W==1 是必要条件——窗口输入覆盖集合 = {k_h·d_H}×{k_w·d_W}，
    # d>1 时不等于整平面（§0.6 R4 反例一；Stage 2 修正）。缺此项会让 (1,8,5,5) k2s1p0 d4
    # 这类 out=1×1 配置静默走整平面归约而给出错值。
    IS_IDENTITY = (k_H == 1 and k_W == 1 and d_H == 1 and d_W == 1
                   and p_H == 0 and p_W == 0 and s_H == 1 and s_W == 1)
    IS_GLOBAL = (out_H == 1 and out_W == 1 and p_H == 0 and p_W == 0
                 and d_H == 1 and d_W == 1
                 and (k_H - 1) * d_H + 1 >= H_in
                 and (k_W - 1) * d_W + 1 >= W_in)
    @T.prim_func
    def main(
        x:   T.Tensor((N, C, H_in, W_in), dtype),   # dtype 位置参数（非 dtype=）
        out: T.Tensor((N, C, out_H, out_W), dtype),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # ---- C4b 恒等（k==1 ∧ d==1 ∧ p==0 ∧ s==1；单元 = 整张 (n, c_block) 平面）----
            if IS_IDENTITY:
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb * CH)
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in],
                               stage[0:real_c, 0:H_in, 0:W_in])
                        T.copy(stage[0:real_c, 0:H_in, 0:W_in],
                               out[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in])
            # ---- C4 整平面归约（五条件含 d==1；单元 = 整张 (n, c_block) 平面）----
            elif IS_GLOBAL:
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb * CH)
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in],
                               flat[0:real_c, 0:H_in, 0:W_in])
                        T.vcast(flat, flat32, round_mode="rint")
                        T.reduce_max(flat32, mid,   dim=2, clear=True)   # (CH,H_in,W_in)->(CH,H_in,1)
                        T.reduce_max(mid,    gmax,  dim=1, clear=True)   # ->(CH,1,1)
                        T.vcmp(flat32, flat32, mb, "eq")                 # 非 NaN 掩码(bool, 片上)
                        T.vcast(mb, mf32, round_mode="rint")             # i1 -> f32 {0,1}
                        T.reduce_max(mf32, midn, dim=2, clear=True)
                        T.reduce_max(midn, nmax, dim=1, clear=True)      # 1.0 iff 窗口无 NaN
                        T.clear(zero)                                    # (CH,1,1) f32 = 0
                        T.vcmp(nmax, zero, cond, "eq")                   # 0 即 has_nan
                        T.vbrc(T.infinity("float32"), nanbuf)
                        T.vsub(nanbuf, nanbuf, nanbuf)                   # inf - inf = NaN
                        T.vselect(cond, nanbuf, gmax, out32)             # has_nan ? NaN : max
                        T.vcast(out32, gout, round_mode="rint")
                        T.copy(gout[0:real_c, 0:1, 0:1], out[nb, cb*CH:cb*CH+real_c, 0:1, 0:1])
            # ---- 一般网格（单元 = (n, o_h, c_block)）----
            else:
                acc   = T.alloc_ub((out_W, CH), "float32")   # 最大值累加器（向量轴 = C）
                flag  = T.alloc_ub((out_W, CH), "float32")   # NaN 乘性标记（初值 1）
                mf32  = T.alloc_ub((1, CH), "float32")       # eq 掩码 {0,1}
                cond  = T.alloc_ub((1, CH), "bool")          # eq 掩码（片上）
                cfinal= T.alloc_ub((out_W, CH), "bool")      # has_nan 判决条件
                zero  = T.alloc_ub((out_W, CH), "float32")   # 0 常量
                nanbuf= T.alloc_ub((out_W, CH), "float32")   # NaN 常量
                out32 = T.alloc_ub((out_W, CH), "float32")
                acc16 = T.alloc_ub((out_W, CH), dtype)
                acc_t = T.alloc_ub((CH, out_W), dtype)
                row_src = T.alloc_ub((CH, W_in), dtype)
                row_dst = T.alloc_ub((W_in, CH), dtype)
                row_f32 = T.alloc_ub((W_in, CH), "float32")

                for t in T.serial(num_local):                # persistent（静态界）
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb  = uid // (out_H * num_cblk)
                        r   = uid %  (out_H * num_cblk)
                        o_h = r // num_cblk
                        cb  = r %  num_cblk
                        real_c = T.min(CH, C - cb * CH)
                        T.vbrc(-T.infinity("float32"), acc)      # acc <- -inf（整 buffer）
                        T.vbrc(T.float32(1.0), flag)              # flag <- 1（整 buffer）
                        T.clear(zero)
                        T.vbrc(T.infinity("float32"), nanbuf)
                        T.vsub(nanbuf, nanbuf, nanbuf)            # NaN
                        for kh in T.serial(k_H):                  # 静态界（k_H 编译期常量）
                            ih = o_h * s_H - p_H + kh * d_H
                            if 0 <= ih and ih < H_in:             # 每行一次守卫（C1）
                                T.copy(x[nb, cb*CH:cb*CH+real_c, ih, 0:W_in],
                                       row_src[0:real_c, 0:W_in])
                                T.transpose(row_src, row_dst, permutation=[1, 0])
                                T.vcast(row_dst, row_f32, round_mode="rint")
                                for kw in T.serial(k_W):          # 静态界
                                    for ow in T.serial(out_W):    # 静态界
                                        iw = ow * s_W - p_W + kw * d_W
                                        if 0 <= iw and iw < W_in:
                                            T.vmax(acc[ow:ow+1, 0:CH],
                                                   row_f32[iw:iw+1, 0:CH],
                                                   acc[ow:ow+1, 0:CH])
                                            T.vcmp(row_f32[iw:iw+1, 0:CH],
                                                   row_f32[iw:iw+1, 0:CH], cond, "eq")
                                            T.vcast(cond, mf32, round_mode="rint")
                                            T.vmul(flag[ow:ow+1, 0:CH], mf32,
                                                   flag[ow:ow+1, 0:CH])
                        T.vcmp(flag, zero, cfinal, "eq")          # flag==0 <=> 含 NaN
                        T.vselect(cfinal, nanbuf, acc, out32)     # has_nan ? NaN : max
                        T.vcast(out32, acc16, round_mode="rint")
                        T.transpose(acc16, acc_t, permutation=[1, 0])
                        T.copy(acc_t[0:real_c, 0:out_W],
                               out[nb, cb*CH:cb*CH+real_c, o_h, 0:out_W])
    return main
```

（伪代码为**决策示意**，Stage 3 可细化局部次序与 buffer 复用；但「向量轴 = C」「核内二轴转置」
「C1 守卫外提」「eq 掩码乘性 NaN 追踪 + `has_nan ? NaN : max`」「C4/C4b 充分条件（C4 含 `d_H==d_W==1`）」是冻结决策。
**实现次序约束**：`T.vcmp(eq)` 必须在 `T.vmul(flag, …)` **之前**（掩码当轮生成当轮消费）；
`T.vcmp(flag, zero)` 必须在**全部** `vmax`/`vmul` 之后——次序错误会导致判据恒真/恒假，
`verify_equiv.py` 只验公式不验 op 次序，故 §9.2 列入实现级检查项。）

### 3.4 API 可行性确认

| API | 来源确认 | 是否验证 |
|-----|---------|---------|
| `T.Kernel(num, is_npu=True) as (cid, _)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3（NPU ≤2 维） | ✓ 现有 examples 广泛使用 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` §2.2.2（1D~2D 充分验证；本设计用 1D/2D + C4/C4b 的 3D） | ✓（同族前作同款） |
| `T.vbrc`（scalar→任意 shape 广播） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 | ✓ |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` §2.2（fp32 ✓，1~5 维） | ✓ |
| `T.copy`（切片 GM↔UB） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2 | ✓ |
| `T.transpose`（二轴交换，fp16/bf16/fp32/int32） | `docs/Tilelang.language/创建操作/T.transpose.md` §2.3 | ✓ |
| `T.vcast`（f16/bf16→f32 `rint`；f32→f16/bf16 `rint`；i1→f32 `rint`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1/§2.2.2/§2.3 | ✓ |
| `T.vmax`（int16/int32/int64/fp16/fp32；**不含 bf16**；`dst` 可原地） | `docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 | ✓ |
| `T.vcmp`（fp16/fp32/bf16；输出 bool 片上；`eq` 模式 IEEE 正确） | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1/§2.3 | ✓ |
| `T.vselect`（fp16/fp32/bf16；bool 条件） | `docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1 | ✓ |
| `T.vmul`（fp16/fp32/int16/int32/int64；支持原地） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1/§2.3 | ✓ |
| `T.vsub`（fp32；标量/张量；原地） | `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1/§2.3 | ✓ |
| `T.infinity("float32")` | `docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（fp16/fp32 ✓；**bf16 ✗**） | ✓ |
| `T.reduce_max`（fp16/fp32 ✓；**不含 bf16**；`clear`） | `docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1/§2.1 | ✓ |

> **dtype 约束提醒**：`T.vmax`/`T.reduce_max` 的 dtype 表**不含 bf16**，`T.vsub`/`T.vmul` 亦不含 bf16；
> `T.vcmp`/`T.vselect`/`T.transpose`/`T.vcast` 支持 bf16。故 **bf16 输入必须经 `T.vcast` 上转 fp32 后计算**
> ——与「fp32 比较域」语义一致（§0.6 R2），无额外代价（`nanbuf`/`zero`/`acc`/`flag` 均用 fp32）。
> **`bool`（i1）不可 load/store GM**：本设计的 `cond`/`cfinal`/`mb` 均为**片上中间结果**，不写回 GM ✓。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（用一维 `T.Kernel(num_kernels)`；任务维度在 kernel 内解码） | 一维网格 |
| GPU 专用 API | 否（全部 v-prefix / `T.copy` / `T.transpose`，均为 NPU API） | — |
| GEMM 非整除 / L0C 溢出 | 否（无 Cube 计算，`L0C` 不适用） | — |
| 分核策略 | 涉及 | 见 §5.5（三要素齐全，persistent grid-stride） |
| host 侧禁止改输入内容 | 否（host 只做 shape/stride 元数据与缓存；无 padding / 无数据拷贝） | `ascend-constraints.md` §4 |
| 动态 `T.serial` 边界 | 否（`k_H`/`k_W`/`out_W` 全为编译期常量，`T.serial` 上界天然静态） | 比 adaptive 家族更简单；无需 `max_bin_extent` 机制 |
| UB 容量 | 涉及（大 `W_in` 的 `resnet-stem`） | 容量驱动的 `CH` 选择（§4.5 带字节数不等式） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 / 并行模型 | `T.Kernel(ceildiv(total_output,block_m), threads)` + `T.Parallel(block_m)`（线程 lane 级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1（`(n,o_h,c_block)` 单元 + C 向量 lane） |
| 归约单元 | 每线程标量比较链（`T.alloc_var`） | UB fp32 向量比较器（C 轴） | §0.6 R2 |
| NaN 追踪 | 标量 bool `has_nan \| isnan(val)` | C 向量 eq 掩码乘性累积（`vcmp`+`vcast`+`vmul`） | §0.6 R2（语义不变） |
| 数据搬运 | 逐元素 GM gather | UB 行片搬运 + 核内二轴转置 | §0.6 R3 |
| 越界守卫 | 逐元素 in-bounds 谓词 | H 方向每行一次（C1）+ W 方向逐 lane | §1.6.1 C1 |
| 结构特化 | 无 | C4/C4b（**校正充分条件**，容量守卫） | §0.6 R4 / §1.6.1 |
| `threads` 参数 | factory `(block_m, threads)`；`default_config={"block_m":256,"threads":256}` | **移除 `threads`**（NPU 无 CUDA 线程）；callable 只留 `block_m` | 脚手架已按 K9 移除；factory 签名改为 `_build(block_m)` |
| autotune | `autotune_configs = [128,256,512]×[128,256,512]` | 移除（K8），启发式 config（容量驱动 CH） | 脚手架已移除 |
| 编译标记 | `compile_flags=["-O3","-DENABLE_BF16"]` | 保留（设备无关） | — |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | **最高（同族前作，CASE-pool-adaptiveavgpool2d-migration）** | Developer 模式一维 `T.Kernel(is_npu=True)`；`CH=min(C,64)`；`(n,o_h,c_block)` 单元解码；`T.vbrc`/`T.clear`/`T.vcast`/`T.vadd`/`T.transpose` 用法；persistent 分核与 `real_c` 尾块；C4/C4b 特化 |
| `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md` | **最高（同族 max 变体设计，CASE-pool-adaptivemaxpool2d-migration）** | max 的 fp32 比较域 + 显式 NaN 追踪；`T.vmax` + eq 掩码（`TRAP-vcmp-ne-ordered-nan` 的正确用法）；C 轴向量化 + 核内转置的完整论证 |
| `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/DESIGN.md` | 高（同族第三作） | `T.vbrc` region 陷阱、`arange` in-chain 陷阱、退化路径逐 buffer 容量复算范式 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 高（结构模板） | persistent grid-stride `T.serial(num_local)` + `if block_id<num_logical` 守卫；`T.alloc_ub`；尾块 gate |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | 高（归约模板） | `T.reduce_max` 用法、fp32 中转、尾部切片 |
| `examples/TileOPs/tileops/kernels/pool/max_pool2d/_max_pool2d_fwd_kernels.py` | 高（本迁移的源函数抽取件） | GPU 源 kernel + 内联 `pool_output_dim`（迁移时逐字对照） |
| `examples/TileOPs/tests/ops/test_pool.py` / `examples/TileOPs/tileops/manifest/pool.yaml` | 高（规格/golden） | workload 与 `ref_program = F.max_pool2d`，`atol=rtol=0` |
| pattern-library `layout.md` 的 `PL-1.1`/`PL-1.2`/`PL-1.3`/`PL-1.4` | 高（池化类实测模式） | 核内融合转置链、C 轴切片累加、host permute 证伪、tiling 启发式 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N, C, H_in, W_in)` | float16 \| bfloat16 \| float32 | **恰好 4D NCHW** 连续；尾轴 `W_in` 连续 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(N, C, out_H, out_W)` | same_as_input | NCHW 连续；`out_idx=[1]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `row_src` | `(CH, W_in)` | dtype | UB | 行片 GM→UB 暂存 |
| `row_dst` | `(W_in, CH)` | dtype | UB | 核内二轴转置输出（C 最内） |
| `row_f32` | `(W_in, CH)` | float32 | UB | fp32 上转 + 向量比较源 |
| `acc` | `(out_W, CH)` | float32 | UB | C 轴向量最大值累加器（向量轴 = C，初值 $-\infty$） |
| `flag` | `(out_W, CH)` | float32 | UB | C 轴 NaN 乘性标记（初值 1，0 吸收） |
| `mf32` | `(1, CH)` | float32 | UB | 逐元素 eq 掩码（`{0,1}`） |
| `cond` | `(1, CH)` | bool | UB | 逐元素 eq 掩码（**片上**，不落 GM） |
| `cfinal` | `(out_W, CH)` | bool | UB | 窗口级 has_nan 判决条件（片上） |
| `zero` / `nanbuf` | `(out_W, CH)` | float32 | UB | 0 常量 / NaN 常量（`inf-inf`） |
| `out32` | `(out_W, CH)` | float32 | UB | NaN 判决结果（fp32） |
| `acc16` / `acc_t` | `(out_W, CH)` / `(CH, out_W)` | dtype | UB | 舍回 + 转置回转 |
| `flat` / `flat32`（C4） | `(CH, H_in, W_in)` | dtype / float32 | UB | 整平面归约退化路径 |
| `mid` / `gmax`（C4） | `(CH, H_in, 1)` / `(CH,1,1)` | float32 | UB | 全平面归约中间 / 输出 |
| `mb` / `mf32`（C4） | `(CH,H_in,W_in)` | bool / float32 | UB | C4 分支的 NaN 掩码（bool 片上）与 fp32 掩码；**与一般路径的 `mf32` 复用同名（分支互斥，shape 不同：`(CH,H_in,W_in)` vs `(1,CH)`）** |
| `midn` / `nmax` / `zero2` / `nanbuf2` / `out32b` / `gout`（C4） | `(CH,H_in,1)` / `(CH,1,1)` 等 | float32 / dtype | UB | NaN 掩码归约、判决、NaN 常量、输出 |
| `stage`（C4b） | `(CH, H_in, W_in)` | dtype | UB | 恒等退化路径中转（单元 = 整张平面，一次整平面搬运） |

### 4.4 内存搬运路径

```
【一般网格】
GM[x] --T.copy 行片(real_c,W_in)--> UB[row_src] --T.transpose[1,0]--> UB[row_dst] --T.vcast(rint)--> UB[row_f32]
UB[row_f32] --T.vmax(C 轴切片, kh serial × kw/ow serial)--> UB[acc (out_W,CH) f32]
UB[row_f32] --T.vcmp(eq)+T.vcast--> UB[cond/mf32] --T.vmul 乘性累积--> UB[flag (out_W,CH) f32]
UB[acc]/UB[flag] --T.vcmp(flag,zero,"eq") + T.vselect(cfinal,nanbuf,acc)--> UB[out32]
                 --T.vcast(rint)--> UB[acc16] --T.transpose[1,0]--> UB[acc_t (CH,out_W)] --T.copy 行片--> GM[out]
【C4 整平面】GM[x] --T.copy 连续块(CH,H_in,W_in)--> UB[flat] --T.vcast--> UB[flat32] --T.reduce_max(dim=2→1)--> UB[gmax]
             UB[flat32] --T.vcmp(eq)+T.vcast+reduce_max--> UB[nmax] --T.vcmp(eq vs 0)--> cond --T.vselect(nanbuf,gmax)--> UB[out32] --T.vcast--> GM[out]
【C4b 恒等】GM[x] --T.copy 整平面(CH,H_in,W_in)--> UB[stage] --T.copy--> GM[out]
（无 GM 中间往返、无 Cube / L1 / L0C 路径）
```

### 4.5 UB 内存预算

**一般路径逐 buffer 字节表（两个代表档；行字节 = 元素数 × 元素宽度）**：

| Buffer | shape（resnet-stem，CH=32，fp16） | Bytes | shape（vgg-block，CH=64，fp16） | Bytes |
|--------|-----------------------------------|-------|-------------------------------|-------|
| `row_src` | `(32,112)` | 7 168 | `(64,56)` | 7 168 |
| `row_dst` | `(112,32)` | 7 168 | `(56,64)` | 7 168 |
| `row_f32` | `(112,32)` f32 | 14 336 | `(56,64)` f32 | 14 336 |
| `acc` | `(56,32)` f32 | 7 168 | `(28,64)` f32 | 7 168 |
| `flag` | `(56,32)` f32 | 7 168 | `(28,64)` f32 | 7 168 |
| `mf32` | `(1,32)` f32 | 128 | `(1,64)` f32 | 256 |
| `cond` | `(1,32)` bool | 32 | `(1,64)` bool | 64 |
| `cfinal` | `(56,32)` bool | 1 792 | `(28,64)` bool | 1 792 |
| `zero` | `(56,32)` f32 | 7 168 | `(28,64)` f32 | 7 168 |
| `nanbuf` | `(56,32)` f32 | 7 168 | `(28,64)` f32 | 7 168 |
| `out32` | `(56,32)` f32 | 7 168 | `(28,64)` f32 | 7 168 |
| `acc16` | `(56,32)` fp16 | 3 584 | `(28,64)` fp16 | 3 584 |
| `acc_t` | `(32,56)` fp16 | 3 584 | `(64,28)` fp16 | 3 584 |
| **合计** | | **73 632 B ≈ 71.9 KiB** | | **73 792 B ≈ 72.1 KiB** |

**容量判据（带字节数的不等式，非代表 shape 外推）**：
`ub_bytes(CH, W_in, out_W, dtype) × INFLATE ≤ 196 608 B (= 192 KiB)`，
其中 `ub_bytes` 按上表逐 buffer 求和；`CH` 取满足该不等式且 `≤ min(C,64)` 的**最大 8 的倍数**
（`CH ≥ 8`；`C < 8` 时取 8 并以 `real_c` 收尾）。`INFLATE` 双口径并列（`CONST-capacity-910B2C`）：
**默认档（auto-multi-buffer 开）1.7**；关闭档 1.10–1.12。设计按**默认档 1.7** 保守判据。
**本小节字节口径统一为 KiB（1 KiB = 1024 B），并同时给出原始字节数以便复算**（Stage 2 修订 v1：
修正旧版在此混用 KiB / 十进制 KB 的不一致）：

- `resnet-stem`：`CH=64` → `ub_bytes = 147 264 B`，`×1.7 = 250 349 B = 244.5 KiB > 192 KiB` ✗（**否决**）；
  `CH=32` → `73 632 B`，`×1.7 = 125 174 B = 122.2 KiB ≤ 192 KiB` ✓ ⇒ **容量判据选 CH=32**。
- `vgg-block` / `alexnet-ceil`：`CH=64` → `73 792 B`（72.1 KiB）/ `71 680 B`（70.0 KiB），
  `×1.7 = 125 446 B`（122.5 KiB）/ `121 856 B`（119.0 KiB）✓。
- `smoke`：`CH=8`，合计 2 664 B（2.6 KiB）✓。
- fp32 输入时 `row_src/row_dst/acc16/acc_t` 宽度翻倍：`resnet-stem` `CH=32` → 95 136 B（92.9 KiB），
  `×1.7 = 161 731 B = 157.9 KiB ≤ 192 KiB` ✓；
  `CH=64` → 190 272 B（185.8 KiB），`×1.7 = 323 462 B = 315.9 KiB > 192 KiB` ✗ ⇒ 同样选 CH=32。
  （Stage 2 修订 v1：旧版此处自述「≈ 205 KB」为**不可机械复现**数字，按本表逐 buffer 口径复算实为
  190 272 B；结论不变——CH=64 仍应被否。）

**退化路径容量核算（独立于代表 shape；失败即回退一般路径，纯优化）**：依据
`_shared/standards/hardware-cost-model.md` §5 与同族三作教训（退化路径缓冲随 `H·W` 线性放大，
代表 shape 的容量结论**不覆盖**退化路径），逐条给不等式（KiB 口径同 §4.5 一般路径）：

- **C4**：判据**显式含主项与小项**（Stage 2 修订 v1：旧版声称「小项已计入」但算式不含，属自述与
  算式不一致）：
  `c4_bytes = CH·H_in·W_in·(elem_bytes + 9)  +  CH·(8·H_in + 20 + elem_bytes + 1)  ≤ 196 608 / 1.7`，
  其中主项 = `flat`（`elem_bytes`）+ `flat32`（4）+ `mb`（1）+ `mf32`（4）= `elem_bytes + 9`；
  小项 = `mid`/`midn` `(CH,H_in,1)` f32 ×2 → `8·CH·H_in`，`gmax`/`nmax`/`zero`/`nanbuf`/`out32`
  `(CH,1,1)` f32 ×5 → `20·CH`，`gout` `(CH,1,1)` dtype → `CH·elem_bytes`，`cond` `(CH,1,1)` bool → `CH`。
  例：`(1,8,4,4)` `CH=8` → 主项 `8·16·11 = 1 408` + 小项 `8·(8·4+20+2+1) = 440` = **1 848 B**，
  `×1.7 = 3 142 B = 3.1 KiB ✓`（小项占 24%，**并非可忽略**）；`(1,8,64,64)` → 主项 `8·4096·11 = 360 448` +
  小项 `8·(8·64+20+2+1) = 4 280` = 364 728 B，`×1.7 = 620 038 B = 605.5 KiB ✗ → 回退一般路径`。
- **C4b**：`c4b_bytes = CH·H_in·W_in·elem_bytes × 1.7 ≤ 196 608 B`。
  例：`(1,8,4,4)` `CH=8` → `8·16·2 = 256 B` ✓；`(1,8,64,64)` → `8·4096·2 = 65 536 B`，
  `×1.7 = 111 411 B = 108.8 KiB ✓`；`(1,8,112,112)` → `8·12544·2 = 200 704 B`，
  `×1.7 = 341 197 B = 333.2 KiB ✗ → 回退一般路径`。
- 回退后语义与精度**不变**（一般路径本就正确），仅放弃纯优化收益。

> **逐 buffer 求和、逐编译期分支复算**（同族 Stage 2 教训：「代表 shape 结论 ≠ 整条特化路径结论」、
> 漏计小 buffer 是低报高发区）——本表按 §3.3 的 `alloc` 清单一一对应，一般路径与两条退化路径分别复算。

### 4.6 动态轴定义

**无动态轴**：`N, C, H_in, W_in, k_H, k_W, s_H, s_W, p_H, p_W, d_H, d_W, ceil_mode, out_H, out_W`
全部在 factory 构建期（trace 期）确定，kernel 由 `functools.lru_cache` 按该组合特化。
（`o_h`/`ih`/`ow`/`iw`/`flag` 判定在 kernel 内是运行时标量 / 条件，但它们是**由编译期 shape 推出的运行时值**，
不是动态轴。）

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3", "-DENABLE_BF16"])
def _max_pool2d_func(CH: int) -> ...      # 去掉源 GPU 的 threads 参数（K9）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。

**判定依据**：算子仅含逐元素向量运算（`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`）、广播（`T.vbrc`）、
cast（`T.vcast`）、归约（`T.reduce_max`）与布局重排（`T.transpose`），无 `T.gemm` / matmul
→ 不需要 L1 + L0A/L0B/L0C（§1.6.0 R1/R4 已弃 Cube 路径——数学上不可表达）。
故核数按**纯 Vector 算子翻倍**规则取物理核数 ×2（§5.5）。

### 5.2 Block 划分

本算子的分块维度不是传统 GEMM 的 `(M,N)`，而是 **(通道块 CH) × (输出行 o_h)** 的任务维度：

```python
CH = <满足 §4.5 容量判据的最大 8 的倍数，且 <= min(C, 64)；C < 8 时取 8>
num_cblk = ceil(C / CH)            # 通道块数
# 逻辑单元数按编译期分支取（单元粒度是 num_logical 的唯一定义源，见 §5.5 ①）：
#   一般网格  单元 = (n, o_h, c_block)  ->  num_logical = N * out_H * num_cblk
#   C4 整平面 单元 = (n, c_block)        ->  num_logical = N * num_cblk（out_H==1 且 d==1，与上式同值）
#   C4b 恒等  单元 = (n, c_block)        ->  num_logical = N * num_cblk（整平面；!= N*out_H*num_cblk）
# IS_GLOBAL / IS_IDENTITY 为 factory 编译期谓词（见 §3.3；IS_GLOBAL 含 d_H==d_W==1）：
#   d>1 的 out=1×1 配置（如 (1,8,5,5) k2s1p0 d4）IS_GLOBAL=False → 走一般网格分支
num_logical = N * num_cblk if (IS_GLOBAL or IS_IDENTITY) else N * out_H * num_cblk
```

- **`CH` 的选择理由**：① C 轴为向量轴，须为 8 的整数倍（fp16/bf16 向量宽 8、fp32 宽 4）；
  ② UB 预算上限（§4.5 判据；`resnet-stem` 因 `W_in=112` 被容量判据压到 CH=32）；
  ③ 与 §5.5 分核方案配合使 `num_logical` 落在物理核数附近/之上（persistent）。
- **`num_logical` 必须与「一个逻辑单元处理多少数据」严格同粒度**：一般网格单元为 `(n, out_row, c_block)`；
  退化路径（C4/C4b）单元为 `(n, c_block)` **整张平面**。C4 因 `out_H==1` 两种口径同值；
  **唯一会分叉的是 C4b**（`out==in`）：通式给 `N*out_H*num_cblk`，整平面单元只需 `N*num_cblk`
  ——若沿用通式会越界读写（同族前作 Stage 2 已实证）。本设计采用整平面口径 `N*num_cblk`。

**代表性 shape 的分块结果**（`out` 由 `pool_output_dim` 给出）：

| workload | N | C | H_in×W_in | k/s/p/d | out | CH | num_cblk | num_logical |
|----------|---|---|-----------|---------|-----|----|----------|-------------|
| `smoke-1x8x16x16-k3s2p1` | 1 | 8 | 16×16 | 3/2/1/1 | 8×8 | 8 | 1 | **8**（一般） |
| `resnet-stem` | 32 | 64 | 112×112 | 3/2/1/1 | 56×56 | 32 | 2 | **3 584**（一般，容量压 CH） |
| `vgg-block` | 16 | 128 | 56×56 | 2/2/0/1 | 28×28 | 64 | 2 | **896**（一般） |
| `alexnet-ceil` | 32 | 64 | 55×55 | 3/2/0/1 ceil | 27×27 | 64 | 1 | **864**（一般） |
| test `full-default-stride-dilation` | 1 | 4 | 14×14 | k(3,3) s(3,3) p(1,1) d(2,1) | 4×5 | 8 | 1 | **4**（一般；C=4 → CH=8 收尾） |
| test `full-nonsquare-ceil` | 1 | 4 | 23×27 | 3,5/2,3/1,2/1 ceil | 12×10 | 8 | 1 | **12**（一般） |
| C4 例 `(1,8,16,16)` k16s1p0 **d1** | 1 | 8 | 16×16 | k(16,16) s(1,1) p(0,0) d(1,1) | 1×1 | 8 | 1 | **1**（C4 整平面；须 `out=1×1 ∧ p=0 ∧ d=1 ∧ 覆盖`) |
| C4x 反例 `(1,8,5,5)` k2s1p0 **d4** | 1 | 8 | 5×5 | k(2,2) s(1,1) p(0,0) d(4,4) | 1×1 | 8 | 1 | **1**（**一般路径**，`d>1` 不走 C4；见 §0.6 R4 反例一） |
| C4b 例 `(1,8,4,4)` k1s1p0 d1 | 1 | 8 | 4×4 | k(1,1) s(1,1) p(0,0) d(1,1) | 4×4 | 8 | 1 | **1**（C4b 整平面） |

> **注意**：C4/C4b 两行**不被当前 manifest workloads 触发**（manifest 三档 full 的 `k/s/p` 组合均非退化），
> 属对 op 层一般参数空间的正确性/性能补全（§0.6 R4），容量受 §4.5 判据约束、失败即回退。
> C4x 反例行（`d>1`，`out=1×1`）**不走 C4**：因 `IS_GLOBAL` 的 `d==1` 条件不成立（§0.6 R4 反例一，
> Stage 2 修正），它按一般网格口径计 `num_logical = N × out_H × num_cblk = 1`。

### 5.3 约束分析

- **对齐约束**：向量轴 C 恒为 8 的整数倍（8 同是 fp32 向量宽 4 的倍数）；`CH` 取 8 的倍数 ✓。
  `W_in`（55/56/112）仅作转置后 row 长度，**不作向量轴**，不要求整除 ✓。
- **UB 容量**：代表档 ≈ 72 KB（默认档 ×1.7 → ≈ 122 KB）< 192 KB ✓；`resnet-stem` 的 CH=64 被容量判据否决 → CH=32 ✓（§4.5）。
- **L0 容量**：无 Cube 计算，不适用 ✓。
- **非整除**：本算子**不存在 GEMM 式整除问题**——行长度用静态 `W_in`；任务维度用 persistent
  `T.serial` + `if uid < num_logical` 守卫（尾任务丢弃）；通道尾块用 `real_c = min(CH, C - cb*CH)`
  动态 copy 范围，padded lane 永不写回 GM。窗口越界由 §1.6.1 C1 守卫处理（语义本来就要求）。

### 5.4 注意事项

- **尾单元**：`num_logical` 非 `num_kernels` 整数倍时（如 resnet-stem 3584 / 40 = 89.6），
  核内 `T.serial(num_local)` 的 `if uid < num_logical` 守卫丢弃越界任务。
  退化路径的 `num_logical` 按其整平面单元粒度取（`N*num_cblk`）。
- **通道尾块**：`C % CH != 0`（如 C=96 → CH=64 + `real_c=32`；C=4 → CH=8 + `real_c=4`）——
  copy/写回用 `real_c` 动态范围；`row_src/row_dst/row_f32` 的 padded lane 残留旧值，
  但 padded lane 的 `acc`/`flag` 永不写回 GM（各 lane 独立），无污染。
- **H 越界守卫**：`if 0 <= ih and ih < H_in`（每行一次，C1）。窄窗口（如 `k1s1p0`）该守卫恒真。
- **W 越界守卫**：`if 0 <= iw and iw < W_in`（逐 lane）。`p_W=0` 且 `s_W≥k_W` 时多数 lane 恒真。
- **零输出保护**：`out_H/out_W > 0` 由 op 层保证（"output size must be greater than zero"）。
- **NaN 语义**：必须在所有输入（含 `±inf`/subnormal）下保持「NaN 胜出 + `±inf` 取 `+inf`」——
  由 `eq` 掩码 + `flag` 乘性累积保证（§0.6 R2），**不依赖 `T.vmax`/`T.reduce_max` 的 NaN 行为**。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**（**单元粒度 = `num_logical` 的唯一定义源**）：
  - **一般网格**（非退化）：单元 = `(n, o_h, c_block)` → `num_logical = N × out_H × ceil(C / CH)`
    （纯 Vector 无 `block_N` 维度，故只有通道块维 × 输出行维）。
  - **退化路径 C4 整平面**（`IS_GLOBAL`，充分条件 = `out_H==out_W==1 ∧ p_H==p_W==0 ∧ d_H==d_W==1 ∧
    (k_H-1)·d_H+1>=H_in ∧ (k_W-1)·d_W+1>=W_in`；`d==1` 项为 Stage 2 修正）：单元 = 整张
    `(n, c_block)` 平面 → `num_logical = N × ceil(C/CH)`（因 `out_H==1`，与通式同值，无分叉）。
    `d>1` 的 `out=1×1` 配置 `IS_GLOBAL=False`，落入**一般网格**口径（§0.6 R4 反例一）。
  - **退化路径 C4b 恒等**（`IS_IDENTITY`）：单元 = 整张 `(n, c_block)` 平面 → `num_logical = N × ceil(C/CH)`，
    **显式不同于**通式 `N × out_H × ceil(C/CH)`（§5.2 说明）。
  - 代表值见 §5.2 表：smoke 8 / resnet-stem 3 584 / vgg-block 896 / alexnet-ceil 864。

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
  | `smoke` | 8 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = 8`，每核 1 个单元 |
  | 多数测试 shape | 4–12 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = num_logical`，每核 1 个单元 |
  | C4 / C4b 特化（各自充分条件成立） | 1 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = 1`，1 核执行 1 个整平面单元 |
  | `vgg-block` | 896 | 40 | **极大规模**（逻辑核数 ≫ 物理核数：896 = 40×22 + 16，非整数倍） | **核内串行（persistent）**：固定 `num_kernels = 40`；`num_local = 23`（**静态值**）；`uid = t*num_kernels + cid`，`if uid < num_logical` 守卫 |
  | `alexnet-ceil` | 864 | 40 | **极大规模** | 同款 persistent：`num_local = 22`（864 = 40×21 + 24，负载差 1 任务） |
  | `resnet-stem` | 3 584 | 40 | **极大规模** | 同款 persistent：`num_local = 90`（3584 = 40×89 + 24） |

  **通用规则（factory 级）**：`num_logical` 按分支粒度取（见 ①），`num_kernels = min(num_logical, vector_cores)`，
  `num_local = ceil(num_logical / num_kernels)`（Python 整数编译期常量）。当 `num_logical ≤ vector_cores`
  → `num_local = 1`，退化为「每核 1 单元 = 无需适配」；当 `num_logical > vector_cores` → 核内串行（persistent）。
  **核内串行循环边界 `num_local` 为静态值**——满足 `core-split-strategy.md` §1 要素③硬要求。
  > **不做空核的保证**：因 `num_kernels = min(num_logical, vector_cores)`，故 `num_kernels ≤ num_logical`，
  > 每核首个任务 `uid = cid < num_logical` 恒有效。
  > **分核增益预期**：三个 full workload 若按 GPU 式超发逻辑核（3584/896/864），NPU 运行时会把超出物理核数的
  > 部分串行排入 40 核 → 引入核启动开销；persistent 化后固定 40 核 × 静态任务数（`num_local` ≤ 90），
  > 摊薄核启动开销。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block/tile 级调度循环。**不出现逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑单元（task） | block 级并行 | `T.Kernel(num_kernels)` | 每核处理 1 个或多个 `(n,o_h,c_block)` 单元 |
| 累加器/标记初始化 | **整块填充/清除**（无循环） | `T.vbrc(-inf, acc)` + `T.vbrc(1.0, flag)` + `T.clear(zero)` | 单元入口整 buffer 初始化（§1.6.2 #1/#2/#14） |
| 核内多任务 | 迭代（persistent） | `T.serial(num_local)`（静态界）+ `if uid<num_logical` | 极大规模时摊薄核启动开销（§5.5）；静态边界 |
| 窗口 `k_h` 方向 | 迭代（归约链） | `T.serial(k_H)`（**编译期常量上界**）+ `if 0<=ih<H_in` | tile 级顺序依赖（循环携带累加维）+ 每行守卫（§1.6.2 #9，C1） |
| 窗口 `k_w` 方向 | 迭代（归约链） | `T.serial(k_W)`（编译期常量上界）+ `if 0<=iw<W_in` | tile 级顺序依赖（§1.6.2 #10） |
| 输出列 `o_w` | 迭代（运行时行索引） | `T.serial(out_W)`（编译期常量上界） | `acc`/`flag` 的运行时行索引；循环体为 C 向量 op（§1.6.2 #10） |
| 元素级（C 轴） | 向量化 | `T.vmax`/`T.vcmp`/`T.vcast`/`T.vmul`/`T.vselect`（向量轴 = C，§1.6.3 决策） | C 轴 lane 并行，无逐元素循环 |
| 退化路径 | 单次归约 / 拷贝 | `T.reduce_max`（C4）/ `T.copy`（C4b） | 充分条件成立时无窗口循环 |

### 6.2 循环伪代码

```python
# num_kernels / num_local 由 num_logical 导出；num_logical 按编译期分支的单元粒度取（§3.3/§5.5①）：
#   一般 = N*out_H*num_cblk；C4 整平面 = N*num_cblk；C4b 恒等 = N*num_cblk
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 一维网格（<=2 维合法）
    for t in T.serial(num_local):                            # persistent（静态界）
        uid = t * num_kernels + cid
        if uid < num_logical:                                # 尾任务守卫（与 num_logical 同粒度）
            # 解码为 (n, o_h, c_block)【一般】/ (n, c_block)【C4/C4b】后进入 §3.3 单元体
            ...
            for kh in T.serial(k_H):                         # 编译期常量上界
                ih = o_h * s_H - p_H + kh * d_H
                if 0 <= ih and ih < H_in:                    # 每行守卫（C1）
                    [T.copy -> row_src; T.transpose; T.vcast -> row_f32]
                    for kw in T.serial(k_W):                 # 编译期常量上界
                        for ow in T.serial(out_W):           # 编译期常量上界
                            iw = ow * s_W - p_W + kw * d_W
                            if 0 <= iw and iw < W_in:
                                T.vmax(...)                  # C 轴向量最大值
                                T.vcmp(..., "eq"); T.vcast(...); T.vmul(...)  # C 轴向量 NaN 追踪
            T.vcmp(flag, zero, cfinal, "eq"); T.vselect(cfinal, nanbuf, acc, out32)
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- 单元内只有一条 `GM→UB→计算→UB→GM` 单阶段流，无多阶段 K 循环可流水；窗口循环次数少（`k_H` ∈ 1–3），
  双缓冲收益有限。
- Developer 模式下编译器可自动 multi-buffer；若 Stage 4 实测显示 MTE2 未与 Vector 重叠，
  可在 `t` 任务循环或 `k_h` 循环上探索 `T.Pipelined(num_stages=2)`（列入 Stage 4 候选，不改主结构）。
- C4/C4b 退化路径为「1 次搬运 + 1 次归约 / 纯拷贝」，无流水空间。

### 6.4 尾块处理

- **任务尾块**：`num_logical` 非 `num_kernels` 整数倍 → `if uid < num_logical` 丢弃越界任务。
  退化路径 `num_logical` 按整平面单元粒度取（§5.5 ①）。
- **通道尾块**：`real_c = min(CH, C - cb*CH)` 动态 copy/写回范围；padded lane 永不落 GM。
- **窗口尾块**：H 方向 `kh` 守卫跳过界外行；W 方向 `iw` 守卫跳过界外列（精确覆盖窗口，无掩码需求；
  与 adaptive 的「运行时 bin 边界差」不同，本算子窗口边界由编译期 `k/s/p/d` 与运行时 `o_h`/`o_w` 线性给出）。
- **无 GEMM 式整除要求**（§5.3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步（Developer 模式）**。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器在 `T.copy`（MTE2 搬运）与 v-prefix 向量 op（Vector 单元）之间
自动插入同步标志，`T.transpose`（VTransposeOp）与 `T.reduce_max`（VReduceOp）亦由编译器编排。
本算子无跨核通信、无 Cube/Vector 跨引擎依赖：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| （无） | — | Developer 自动同步；本算子单引擎（Vector）、单阶段，无 `sync_block_set/wait` 需求 |

### 7.3 pass_configs 配置

`@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3","-DENABLE_BF16"])`；
不设置自定义 `pass_configs`（与 `lerp_tensor` / `logsumexp` / 同族池化先例一致）。
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


def golden_max_pool2d(x, kernel_size, stride=None, padding=(0, 0),
                      dilation=(1, 1), ceil_mode=False):
    """PyTorch reference for max_pool2d, per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark (``MaxPool2dTest.ref_program``
    in examples/TileOPs/tests/ops/test_pool.py).  Exactly 4D NCHW; independent
    of the NPU algorithm (no transpose / no C-vectorisation / no C4-C4b
    specialisation).  Runs on CPU and does not require an NPU device.
    """
    return F.max_pool2d(
        x.detach().to(torch.float32),  # fp32 opmath domain for fp16/bf16 CPU ref
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    ).to(x.dtype)
```

（golden 以 fp32 中转后舍回输入 dtype，与 §0.1「fp32 比较域」一致；max 只选择不累加，故中转无损，
`atol=rtol=0` 成立。golden 语义已由 `verify_equiv.py` 的 E1 组对照本设计 NPU 公式复核。）

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 0 | 0 |
| bfloat16 | 0 | 0 |
| float32 | 0 | 0 |

（**逐位精确**：max 是被选中元素的原值，无累加、无舍入误差——依据 §0.1 实测与
`examples/TileOPs/tests/ops/test_pool.py` 第 323 行注释；源仓 GPU 套件同样用 `atol=0, rtol=0`。
模板通用档 1e-2/1e-4 在本算子不适用。）

### 8.3 L0 门槛测试计划（`tilelang-op-develop` 展开为 L1/L2/Boundary）

| # | 用例 | 覆盖点 |
|---|------|--------|
| L0-1 | `(1,8,16,16)` k3s2p1 fp16 | manifest smoke；通用路径基线 |
| L0-2 | `(32,64,112,112)` k3s2p1 fp16 | manifest `resnet-stem`；**容量压 CH=32** + persistent 3 584 单元 |
| L0-3 | `(16,128,56,56)` k2s2p0 fp16 | manifest `vgg-block`；非重叠窗口（`s=k`） |
| L0-4 | `(32,64,55,55)` k3s2p0 ceil fp16 | manifest `alexnet-ceil`；`ceil_mode` 末窗回退 |
| L0-5 | 同 L0-1/2 的 **fp32** 变体 | dtype 门禁含 fp32（与 adaptive 家族差异）；fp32 宽度预算 |
| L0-6 | `(2,64,55,57)` k3s2p0 bf16 | bf16 路径（**必须经 vcast 上转 fp32** 才能用 `vmax`） |
| L0-7 | `(1,4,23,27)` k(3,5)s(2,3)p(1,2) d1 ceil fp16 | 非方形 kernel/stride/padding + ceil |
| L0-8 | `(1,4,14,14)` k3s3p1 d(2,1) fp16 | 非单位 dilation；`stride=None`（= kernel）路径 |
| L0-9 | **NaN 语义门禁**：窗口内置 NaN（多处，含角点）→ 输出 NaN | §0.1 NaN 胜出；不得依赖 `vmax` NaN 行为（§9.2 实现级检查） |
| L0-10 | **±inf 语义门禁**：`{+inf,-inf}` 同窗 → `+inf`（非 NaN）；全 `-inf` 窗 → `-inf` | §0.1；`eq` 掩码不误伤 ±inf |
| L0-11 | **非连续输入** | op 层 `.contiguous()` 路径（test `full-noncontiguous-3x3-fp16`） |
| L0-12 | **C4 特化**：`(1,8,16,16)` k16s1p0 **d1** → `out=1×1`（满足修正后五条件、走 C4），且一个 `(1,8,64,64)` 大平面（应回退） | §1.6.1 C4 充分条件 + 容量回退 |
| L0-12b | **C4 dilation 守卫**：`(1,8,5,5)` k2s1p0 **d4** 与 `(1,8,7,7)` k3s1p0 **d3**（均 `out=1×1` 但 `d>1`，**不得**走整平面归约），另加 `(1,8,5,5)` d4 且输入仅 `x[2,2]=1`（中心元素）的位精确对照 | §1.6.1 C4 的 `d==1` 必要约束（REVIEW 问题 1）；输出须与 golden 逐位一致（`verify_equiv.py` C4x 组 6 行 + HAZARD 演示为其机器对应） |
| L0-12c | **`ceil_mode` 回退**：`(1,8,8,8)` k3s4p0 ceil → `out=2`（触发 `pool_output_dim` 末窗回退分支） | helper 两处 ceil 修正逐字复制的正确性（`verify_equiv.py` E1 组已含该配置） |
| L0-13 | **C4b 特化**：`(1,8,4,4)` k1s1p0 **d1** → `out=x`，且一个 `(1,8,112,112)` 大平面（应回退） | §1.6.1 C4b 充分条件 + 容量回退 |
| L0-14 | **通道尾块**：C=96（CH=64 + `real_c=32`）、C=4（CH=8 + `real_c=4`） | `real_c` 动态范围；padded lane 不落 GM |
| L0-15 | **参数/输入校验**：非 4D / fp64 / `padding>k//2` / 非正输出尺寸 | op 层拒绝路径（test 第 361–436 行） |

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 本算子为**纯 Vector**、单阶段、无跨核通信；无 Cube / L1 / L0C 路径。
- 窗口循环上界（`k_H`/`k_W`/`out_W`）全为编译期常量，**不需要** adaptive 家族的 `max_bin_extent` 机制。
- 退化路径（C4/C4b）的容量随 `H·W` 线性放大，须按 §4.5 不等式独立核算、失败回退（纯优化）。
- 编译期分支（`IS_IDENTITY`/`IS_GLOBAL`）用**纯名 Python bool**（trace 时折叠）；避免表达式条件
  （`not X` / `X and Y` 会生成 TIR runtime if）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **`T.vcmp(...,"ne")` 误用作 NaN 探测** | 想用 `v!=v` 判 NaN | `ne` 是 **ordered-NE**（NaN 恒假）→ 掩码恒假、静默失效；端到端 NaN 用例可能因 `vmax` 未文档化的 NaN 传播而**假阳性通过** | **必须用 `eq`**（`x==x` ⟺ 非 NaN）；本设计用 `eq` 掩码乘性累积 + `flag` 判据（§0.6 R2）。**实现级检查项**：断言 `flag` 初值 1、当轮 `vcmp(eq)`→`vcast`→`vmul` 次序、末尾 `vcmp(flag,zero,"eq")` |
| **依赖 `T.vmax`/`T.reduce_max` 的 NaN 行为** | 直接 `vmax` 累加就当 NaN 会胜出 | 行为**未文档化**（工具链可能传播也可能忽略），跨版本不稳 | 显式 NaN 追踪（`flag`/掩码）；C4 用独立掩码归约判决，不依赖 `reduce_max` 的 NaN 行为 |
| **`T.vbrc` destination 写成 buffer region** | `T.vbrc(-inf, acc[0:1, :])`（带运行时偏移的 UB region，且在 `T.serial` 内） | 写入被**静默丢弃**；int32 形态还会读未初始化 UB → 向量核 fault（core dump） | `T.vbrc` 的 dst **必须是整 buffer**（本设计 `acc`/`flag`/`nanbuf` 均整 buffer 广播）；需逐行填不同标量时用「标量→`(1,CH)` staging→`T.copy`」三件套 |
| **`T.vbrc` 标量 operand 用裸 Python 数** | `T.vbrc(-inf, ...)` / `T.vbrc(1.0, ...)` | 命中前端 rank 断言 | 用 **typed PrimExpr**：`-T.infinity("float32")` / `T.float32(1.0)` |
| **`T.vbrc` 同形 src/dst** | src 与 dst 完全同形 | 空广播维数组 → MLIR verify 失败 | src 为**标量**（本设计即如此）；同形场景用 `T.copy` |
| **把 `for k in range(n)` 当 Python 编译期展开** | 在 `@T.prim_func` 内写 `for kw in range(k_W)` 期望展开为 op 序列 | `range` 被解析为 **serial TIR 循环**（loop var 是 `tir.Var`）；用该 var 索引 Python 列表 / `int()` 转换在 trace 期失败 | 本设计统一用 `T.serial(k_W)` + 编译期常量上界（**不需要**展开为 op 序列），故不触发 |
| **原位 `T.vselect` 在 `T.serial` 内丢 loop-carried 状态** | `T.vselect(cond, chunk, running, running)`（false 操作数同时是输出）跨多迭代 | 状态静默丢失 | 本设计**不用**原位 vselect 累积状态——NaN 累积走 **`T.vmul`**（文档明确支持原地，§3.2 6c），判决 `T.vselect(cfinal, nanbuf, acc, out32)` 的 `Out` 独立 |
| **`T.vcast` 形状不一致 / 源为 GM** | 直接 `T.vcast(GM_region, ub_f32)`，或 dst 形状不同（承担 reshape） | BiShengHIR 阶段失败 / 广播形状报错 | 先同 dtype `T.copy` GM→UB，再 `T.vcast` UB→UB（同形）；换布局用 `T.transpose` 而非 vcast |
| **`T.transpose` 多轴 / 先 reshape** | `permutation=[1,2,0]`（3-cycle）或 reshape→transpose | verify 失败 / **mis-compile 数据错乱** | 只做二轴交换（本设计用 `[1,0]`）；禁止 reshape 后再 transpose（在自然形状 buffer 间转置） |
| **UB 溢出** | `W_in` 大（resnet-stem 112）+ CH=64 | 编译期 UB overflow | 容量判据（§4.5）压 CH（resnet-stem → 32）；必要时评估 `--enable-auto-multi-buffer=false` |
| **`real_c` 尾块漏处理** | `C % CH != 0` | 越界读写 / 落盘错值 | copy/写回用动态 `real_c = min(CH, C - cb*CH)`；padded lane 不落 GM |
| **op 次序错误（`vcmp(eq)` 与 `vmul`）** | 先 `vmul` 后 `vcmp` | 掩码与累积错位 → 判据恒真/恒假、静默退化 | 固定次序（§3.3 注释）；`verify_equiv.py` 只验公式不验 op 次序，须在 L0-9/L0-10 门禁实测 |

### 9.3 特殊场景处理

- **非整除分块**：无 GEMM 式要求；任务/通道/窗口三类尾块分别由 `uid` 守卫 / `real_c` / `ih,iw` 守卫处理（§6.4）。
- **极小 shape**：`C < 8`（如 C=4）→ `CH=8` + `real_c` 收尾；`out_H/out_W=1`（C4）或 `k=1`（C4b）走特化。
- **混合精度**：三 dtype 均支持；bf16 经 `vcast` 上转 fp32 计算、末尾 `vcast` 舍回（与 fp32 比较域语义一致）。
- **`padding>0` + `ceil_mode`**：末窗部分落在 padding，由 `ih,iw` 守卫按 $-\infty$ 处理，与 torch 一致。
- **NaN / ±inf**：见 §0.1 / L0-9 / L0-10；实现级次序检查见 §9.2。

### 9.4 待确认 / 风险登记（供 Stage 2 复核）

1. **发射项不确定性（最高优先）**：`resnet-stem` 的 op 数 ≈ 2.46 M（窄向量 op），
   `CONST-vector-launch-overhead` 0.5µs/op **不适用于窄 op 外推**，发射项可能成为实际主导项
   （§1.6.0 估算行已标注「未文档化假设」）。→ §1.6.3 已给出主选/备选/裁决计划。
2. **设备差异**：roofline 引用的带宽/发射/容量常数均绑定 Ascend910B2C（48 向量核），本设备为 910B3（40）——
   跨型号复用属**未实测假设**，只作量级下界。
3. **stale 条目重验**：本设计引用的 `TRAP-vcmp-ne-ordered-nan`、`TRAP-npuir-vbrc-region-dropped`、
   `TRAP-tir-range-loop-not-unrolled`、`TRAP-vcast-shape-and-gm-source`、`TRAP-UB-multibuffer-inflation`
   等条目已被 `kb_stale_check.py` 标记 `stale`（工具链已从条目记录的版本推进到 `0d3950aa`）。
   本设计**不依赖**这些陷阱是否仍存在：NaN 追踪改用 `eq`（不依赖 `ne`）、`vbrc` 只写整 buffer
   （不写 region）、`T.serial` 用编译期常量上界（不依赖 `range` 展开）、`vcast` 保证同形且源为 UB
   —— 即设计本身对这些陷阱是**结构免疫**的；若 Stage 2 复核发现某条已失效，不影响主结构。
4. **退化路径未被 manifest 触发**：C4/C4b 的正确性由 `verify_equiv.py`（充分条件成立时逐位等价，
   且 C4 的 `d==1` 必要约束由 **C4x 反例组 6 行**机器锁住）与
   L0-12/L0-12b/L0-12c/L0-13 覆盖；其收益不被当前 benchmark 反映。
5. **`s_H < k_H` 的行重复读放大**：`resnet-stem` ≈ 1.5× GM 读；已计入 §1.6.0/§1.6.3，列为 Stage 4
   BH>1 可选探索项（不改主结构）。
6. **退化路径 3D UB buffer**：`T.alloc_ub` 文档称 3D~5D「有待充分验证」；同族前作已实测通过 3D UB，
   本设计 3D buffer 与之同形；若 Stage 3 遇问题，回退为 2D `(CH, H_in*W_in)` + 视图（不改语义）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/max_pool2d/_max_pool2d_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1）
├── _max_pool2d_kernel.py   # 算子实现 + 分层测试（Stage 3 产出）
└── RETROSPECTIVE.md        # Stage 复盘（Stage 1 起追加）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成 | 等价性机器验证（**69** case-group 全 `EQUIV_PASS`；含 C4x dilation 反例组与 ceil step-back 用例） |
| `_max_pool2d_kernel.py` | 待实现 | 算子实现（Stage 3） |
| 分层测试（L0/L1/L2/Boundary） | 待实现 | 内嵌于实现文件或 `testing/`（Stage 3） |

### 10.3 命名规范

- 项目目录名: `max_pool2d`（snake_case）
- 算子目录名: `_max_pool2d_kernel`（源 factory 函数名）
- 实现文件: `_max_pool2d_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1，Stage 3）
3. ⬜ 算子实现（`_max_pool2d_kernel.py`）+ 与 Golden 精度比对（`atol=rtol=0`）

---

## 11. 性能目标（Stage 4 / optimize 场景，conductor 预检收集）

> 本节由 conductor 在 `scenario=optimize` 预检收集后**追加**（optimize.md §3；不覆盖既有内容）。

- 性能目标类型：`bandwidth_ratio`（用户口径：**`achieved_bw / 1.75 TB/s ≥ 70%`**，1.75 TB/s = 本族 fabric 聚合上限）
  - 覆盖集合：manifest `MaxPool2dFwdOp` 的 **3 个 full workload × 3 dtype（fp16/bf16/fp32）= 9 例**
    - `resnet-stem` `[32,64,112,112]` k3s2p1
    - `vgg-block` `[16,128,56,56]` k2s2p0
    - `alexnet-ceil` `[32,64,55,55]` k3s2p0 `ceil_mode=true`（几何特例路径）
  - 达标线由 optimizer 按 manifest `roofline.bytes = (in + out) * elem_bytes` 逐例换算为时延线（`µs = bytes / 0.7 / 1.75e12 * 1e6`）后逐例判定，不得跨例借用
  - `benchmark_source` 中 `smoke-1x8x16x16-k3s2p1`（label 含 `smoke`）→ `kind=smoke`，只做精度回归，不参与性能门禁
  - 迁移基线（上一工具链实测，须在本轮工具链重测后才作对账基准）：resnet-stem 1390.31 / 1391.37 / 2646.45 µs，vgg-block 113.64 / 116.36 / 190.88 µs，alexnet-ceil 201.76 / 202.08 / 365.59 µs（fp16 / bf16 / fp32；ledger §7，Ratio 12.11%–14.25% 为 **msprof compute-roofline** 口径，非本任务达标判据）
- benchmark 参数化入口：`benchmarks/ops/bench_pool.py::test_max_pool2d_bench`（manifest-driven，op `MaxPool2dFwdOp`，workloads 见 `tileops/manifest/pool.yaml`）
- 回归入口：`python perf_opt/_max_pool2d_kernel.py --level all`（**L0/L1 阻塞**；L2/Boundary 告警不阻塞）+ `pytest tests/ops/test_pool.py`（22/22 逐位 `atol=rtol=0`，不得回退）
- 噪声阈值：3%（默认，沿用同族 optimize 先例）；最大迭代数：10；`budget.max_stage4_experiments=40`
- 口径：`msprof op Task Duration(us)` 为唯一时延判据；msprof `Ratio` **如实记录、不作达标判据**（`CONST-msprof-ratio-compute-roofline`：Ratio 不与 latency 成反比）
- 关注 shape：`resnet-stem`（最大流量，1.39 ms 量级）与 `alexnet-ceil`（几何特例）优先；完整 workload 集合由 optimizer 展开 benchmark 实际参数化案例并分类，conductor 不预写 `tune` 集合

### 11.1 已知约束与结构红线（用户要求 + 同族前作结论，逐条须在 opt_log 留痕）

1. **姊妹结论只复用结构、不复用数值归因**：`AdaptiveMaxPool2dFwdOp` 的「扁平空间轴 + 宽归约 / NaN 平方追踪解锁 CH=64」成立语境是 **whole-plane（`out=1×1`）归约**；本算子的三档 workload 均为**窗口累加**（`k/s/p` 非退化），其 16–19 µs 地板带与 `CG-2026-0026/0027` 归因**不可平移**。窗口类先在小 workload 重验适用性再套用。
2. **`PL-1.13` 双 AIV subid 分片对纯 Vector kernel 是假收益**（adaptive_max_pool2d Stage 4 实测证伪：−20%~−41% 为假收益）；任何「白拿」类条目先小 A/B 再套用，A/B 结果记 `perf_records.jsonl`。
3. **几何特例最易回归**：改 UB 分档 / 空间轴扁平化 / `CH` 取档时，`ceil_mode`（alexnet-ceil 与 `full-nonsquare-ceil`）与 `dilation≠1`（`full-default-stride-dilation`、C4x 反例组）两条路径必须逐位复验，**禁 skip / xfail / 删除 / 放宽容差**换绿。
4. **D-3 实验裁决是必做项**（`core-split-strategy.md` §2.4 + 本文件 §1.6.3）：主选「容量允许的最大 `CH`」vs 备选「`CH` 取主选一半」，代表 shape = `resnet-stem` / `vgg-block`，指标 msprof Task Duration，**备选 latency < 主选 × 0.90 才翻转**；并评估「大 `W_in`（UB 受限）走主选 / 小 `W_in` 走更大 `CH`」的按负载分派。裁决须以本轮工具链实测反解未知常数（不得引用上一工具链条目数值）。
5. **已证伪路线勿重推**（同族 `max_pool1d` Stage 4，`TRAP-window-decimation-primitives`）：`T.gather`（≈1.8 ns/元素）、行距不匹配的 `extract_slice`（物化且标量化，Task ×77）、位置主序 + 3D 转置抽取（998.92 µs > baseline 302 µs）三条均实测否决；逃逸方向是**通道分块形状**与 **DMA 行长**（`CONST-dma-burst-rowstride-floor`：行长 128 B / 行距 8 KB ⇒ ≈0.38 TB/s）。
6. **`BH`（每单元输出行数）是可选探索项**（本文件 §1.6.3 末条）：`s_H < k_H` 时 `BH=1` 存在行重复读放大（resnet-stem ≈1.5×），`BH>1` 可复用输入行；不得改动的只有 factory 分派层级与既有三条结构路径的正确性。
7. **结构路径分派保持在 factory 层**：恒等 / C4 全局 / 一般网格三条路径均须保持正确（manifest 三档不触发退化路径，但测试覆盖 C4/C4b/C4x）。
8. `perf_opt/_max_pool2d_kernel.py` 须导出与 baseline 一致的工厂名 `_max_pool2d_kernel` 及 `pool_output_dim`（wrapper 切换块已按此写死双 import）；`build(block_m=None) -> main` 签名不变；tuned 默认参数以**模块级常量**暴露。
9. `tests/ops/test_pool.py`、`benchmarks/ops/bench_pool.py`、两侧 manifest、基准 kernel 与 wrapper **只读不改**；wrapper 切换由 conductor 在回归通过后机械翻转。

### 11.2 交付与验收

- 产物只写 `perf_opt/`（`_max_pool2d_kernel.py` + `opt_log.md` + `perf_records.jsonl` + `workload_inventory.json`，可选 `perf_feedback.md`）；
- 验收：`python perf_opt/_max_pool2d_kernel.py --level all` 的 **L0/L1 必须过（阻塞）**；
- `[DESIGN_LIMIT]` 仅在「步骤 0 copy-floor 标定 + 分核/DMA 粒度扫描 + D-3 裁决」均证 plateau 后产出，须含触发项、影响面与建议路由（`perf-feedback.md` schema）；不得为凑目标反复重试同一结构。
