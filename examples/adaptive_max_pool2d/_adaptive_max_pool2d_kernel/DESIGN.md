# AdaptiveMaxPool2dFwdOp（`_adaptive_max_pool2d_kernel`）算子设计文档

> **迁移任务（TileOPs 同构仓迁移，harness 子模式）**：源算子 = `AdaptiveMaxPool2dFwdOp`
> （family=`pool`），本次重实现的 TileLang kernel 函数 = `_adaptive_max_pool2d_kernel`。
> 本文按「源算子三问解读（§0）→ 算法调研（§1.6.0）→ 硬件耦合性判定与 NPU 重设计（§0.5/§0.6）
> → 通用设计流程（§1–§10）」组织。§1–§10 一律以 §0.6 重设计后的 NPU 算法为输入，不照抄源方案。
>
> - project = `adaptive_max_pool2d`，op = `_adaptive_max_pool2d_kernel`
> - 编程模式：**Developer**（迁移 prompt 指定）
> - 源算子路径：`/workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_max_pool2d.py`
>   （host 层：`/workspace/TileOPs-gpu/tileops/ops/pool/adaptive_max_pool2d.py`，源仓亦含于
>   `tileops/ops/pool.py` 共享池化模块）
> - 源 helper：`/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`
> - NPU 侧抽取参考文件：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/_adaptive_max_pool2d_fwd_kernels.py`
> - NPU 侧 wrapper / Op：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d.py`、`examples/TileOPs/tileops/ops/pool/adaptive_max_pool2d.py`
> - 迁移元数据：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/.migration_meta.json`
> - 设备常数（本设计会话实查）：`NPUUtils.get().get_aicore_num()` = **20**（Ascend910B3，2026-10-13），
>   toolchain `tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5` + `torch_npu 2.9.0.post8`。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 方法论：`tilelang-op-design` skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)。
> 逐行读源文件（`kernels/pool/adaptive_max_pool2d.py` 全文、`kernels/pool/common.py` 全文、
> `ops/pool/adaptive_max_pool2d.py` 的 `_AdaptivePool2dFwdOpBase`/`AdaptiveMaxPool2dFwdOp`/`_adaptive_pool2d_roofline`/`_normalize_output_size`），未臆测语义。
> 仅迁移**单输出**变体 `_adaptive_max_pool2d_kernel`；源文件中的 `_adaptive_max_pool2d_with_indices_kernel`（返回 indices）**不在迁移范围**（manifest `AdaptiveMaxPool2dFwdOp` = `return_indices=False`）。

### 0.1 源算子语义（做什么）

**数学语义**（`torch.nn.functional.adaptive_max_pool2d`，`return_indices=False`）：

对输出空间位置 $(o_h,o_w)$，输入空间轴被划分为半开区间（PyTorch adaptive-bin 划分，源实现由 `common.py:adaptive_bin` 给出）：

$$
\text{bin}_H(o_h)=\big[\lfloor o_h H_{in}/H_{out}\rfloor,\ \lceil (o_h+1)H_{in}/H_{out}\rceil\big),\qquad
\text{bin}_W(o_w)=\big[\lfloor o_w W_{in}/W_{out}\rfloor,\ \lceil (o_w+1)W_{in}/W_{out}\rceil\big)
$$

$$
\text{out}[n,c,o_h,o_w]=\max_{i_h\in\text{bin}_H(o_h)}\ \max_{i_w\in\text{bin}_W(o_w)}\ x[n,c,i_h,i_w]
$$

- **输出尺寸规则**（`_normalize_output_size` + `_resolve_out_dims`）：`output_size` 取
  `None | int | tuple/list[int|None, int|None]`；`None`（含 tuple 中的 `None`）→ 取对应输入轴尺寸。
  `out_H = H_in if oH is None else oH`，`out_W` 同理。标量 int 按 `(v,v)` 展开；`bool` 拒绝；`<=0` 拒绝。
- **规约语义**：规约轴 = H×W 两个空间轴（NCHW 第 2、3 轴），逐 `(n,c,o_h,o_w)` **独立取最大值**，
  **无跨通道/跨 batch 归约**。比较顺序在源实现中是「`kh` 外层、`kw` 内层」的逐元素串行 `T.max`，
  初值 `-inf`（`max_val = T.cast(float("-inf"), accum_dtype)`）。
- **NaN 语义（关键，max 特有）**：源码**显式**追踪 `has_nan = has_nan | T.isnan(val)`，
  最终 `result = T.if_then_else(has_nan, NaN, max_val)`（`adaptive_max_pool2d.py:52,62,65-69`）。
  即 **窗口内任一元素为 NaN → 输出 NaN**（"a NaN anywhere in the window wins"，源 wrapper docstring 原文）。
  **本设计已实测复核**（torch 2.9.0 CPU，2026-10-13）：
  - `F.adaptive_max_pool2d([1, nan; 3, 4], (1,1))` → `nan`（NaN 胜出）；
  - `F.adaptive_max_pool2d([inf, -inf; 1, 2], (1,1))` → `inf`（**±inf 同窗 → +inf，不是 NaN**）；
  - 说明「NaN 胜出」不能由求和类探测（`Σx != Σx`）替代——该式会把 `{+inf,-inf}` 误判为 NaN。**必须用 `v != v` 类探测**。
- **dtype 语义**：输入仅 **float16 / bfloat16**（`_validate_adaptive_pool_input_dtypes` 显式拒绝 fp32）；
  源 kernel 内 `accum_dtype = "float"`（`adaptive_max_pool2d.py:23`）——逐元素先
  `T.cast(x[...], "float")` 再参与 `T.max`，末尾 `T.cast(result, dtype)` 单次舍回。
  **对 max 而言该 fp32 中转是「比较域」而非「累加域」**：输出是**被选中的那个输入元素**，
  输入元素在 fp16/bf16 中可精确表示，`dtype→fp32→dtype` 无损，故 **out 逐位等于窗口内某输入元素**（或 NaN）。
  这正是精度用例 `atol=rtol=0`（依据：`examples/TileOPs/tests/ops/test_pool.py` 用例 `test_adaptive_max_pool2d` 第 130 行注释 "Max is exact (no accumulation)"）的依据。
- **边界语义**：
  - bin 永不为空（`common.py:100-108` 注释：即使 `size_out > size_in` 时 bin 也为 2 个元素）。
  - 相邻 bin 在非整除时**重叠一行/列**（`_adaptive_pool2d_roofline` 注释：`overlap by one row/col unless out | j*in`）。
  - `bin` 边界用**整数差** `(ih_end-ih_start)*(iw_end-iw_start)`，非比例近似。
  - `±inf` 按 IEEE 最大值语义（max 是选择，不产生新值）；NaN 按上文"胜出"规则。
  - 3D CHW 输入：`_eager_forward` 先 `unsqueeze(0)` 成 4D，计算后 `squeeze(0)` 回 3D（恒等 rank 保持）。
  - 无空维度处理（`out > 0` 由 op 层 `_normalize_output_size` 保证）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（优先移植源仓测试基准 `F.adaptive_max_pool2d`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（input） | `(N, C, H_in, W_in)`（op 层同时接受 3D `(C,H_in,W_in)`） | float16 \| bfloat16 | NCHW 连续布局，尾轴 `W_in` 连续 |
| `out`（output） | `(N, C, out_H, out_W)`；3D 输入时输出 3D `(C, out_H, out_W)` | same_as_input | NCHW 连续布局，`out_idx=[1]` |

**输出 shape（迁移必需字段）**：`(N, C, out_H, out_W)`，其中 `out_H = H_in if oH is None else oH`、`out_W` 同理。
输出与输入同布局（NCHW），**不是转置布局**；无 indices 输出。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（`kernels/pool/adaptive_max_pool2d.py:13-74`，逐行）：

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host，trace 期） | `total_output = n*c_in*out_h*out_w`；`max_kh = max_adaptive_bin_extent(h_in,out_h)`；`max_kw = max_adaptive_bin_extent(w_in,out_w)` | 编译期 shape 常量 | Python int | bin 循环的**静态上界**（TileLang 拒绝动态 `T.serial` 边界，源码注释 `adaptive_max_pool2d.py:53-54`） |
| 2（grid） | `T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx` | `block_m`、`threads` | 一维 block 网格 | 输出元素级并行划分 |
| 3（lane 索引解码） | `out_idx = bx*block_m + i`；`ow=out_idx%out_w`；`oh=(out_idx//out_w)%out_h`；`c=(out_idx//(out_w*out_h))%c_in`；`batch=out_idx//(out_w*out_h*c_in)`；经 `if out_idx < total_output` 守卫 | `out_idx` | 标量 `(batch,c,oh,ow)` | 输出坐标 $(n,c,o_h,o_w)$ |
| 4（bin 边界） | `ih_start,ih_end = adaptive_bin(oh,h_in,out_h)`；`iw_start,iw_end = adaptive_bin(ow,w_in,out_w)` | 标量 | 标量对 | $\text{bin}_H(o_h)$ / $\text{bin}_W(o_w)$ |
| 5（初值） | `max_val = cast(-inf, float)`；`has_nan = False` | — | 标量 fp32 / bool | 最大值初值 + NaN 标记初值 |
| 6（bin 取最大值 + NaN 追踪） | `for kh in T.serial(max_kh): for kw in T.serial(max_kw): if ih_start+kh<ih_end and iw_start+kw<iw_end: val=cast(x[batch,c,ih_start+kh,iw_start+kw],"float")；has_nan = has_nan \| T.isnan(val)；max_val = T.max(max_val, val)` | `x` 的逐元素 GM 读 | 标量 fp32 + bool | $\max_{i_h}\max_{i_w}$ + NaN 胜出规则 |
| 7（结果 + 写出） | `result = T.if_then_else(has_nan, cast(NaN,"float"), max_val)`；`out[batch,c,oh,ow] = T.cast(result, dtype)` | 标量 | 标量 | NaN 判决 + 单次舍回输出 dtype |

**数据流与内存访问模式（源硬件视角）**：

```
GM[x] --逐元素标量读 x[batch,c,ih,iw]--> register(max_val fp32, has_nan bool) --NaN 判决 + 单次 cast--> GM[out]
（无 SMEM/L1 中转；每个输出元素现场从 GM gather 其 bin 的全部元素；相邻输出元素重复读重叠行/列；
 GPU 侧靠 L1/L2 缓存吸收这些重复访问）
```

**循环与并行结构**：
- grid = `ceildiv(total_output, block_m)` 个 block，每 block `threads` 个 CUDA 线程；
- `T.Parallel(block_m)`：每个线程（lane）负责**一个输出元素**（lane 内是**标量**计算）；
- lane 内循环：`T.serial(max_kh) × T.serial(max_kw)` 嵌套，**静态边界** + 运行时 `if` 守卫跳过界外 lane；
- **lane 内无归约原语、无向量化**（每个 lane 独立标量 `T.max` 链）；
- 无流水线（无 `T.Pipelined`）、无共享内存、无 SIMD。

**host 侧逻辑**（`ops/pool/adaptive_max_pool2d.py`）：
- `_normalize_output_size`（`output_size` 规范成 2 元组）、`_resolve_out_dims`（`None` → 输入尺寸）；
- 3D→4D `unsqueeze(0)` / 输出 `squeeze(0)`（**只改 stride/view，不改数据**，符合 host 侧约束 `ascend-constraints.md` §4）；
- `x = x.contiguous()`（无实际拷贝时 no-op）；
- `_kernel_cache`（按 `(n,c,h,w,out_h,out_w,dtype,device)` 缓存 kernel 实例）；
- `functools.lru_cache(maxsize=32)` 缓存 kernel factory（按 shape/dtype）；
- `_adaptive_pool2d_roofline`（仅用于 manifest 性能规格，不参与计算）；
- 编译配置：`@tilelang.jit(out_idx=[1], compile_flags=["-O3","-DENABLE_BF16"])`（`adaptive_max_pool2d.py:28`）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **静态边界 bin 循环 + 运行时守卫** | 规避 TileLang「拒绝动态 `T.serial` 边界」限制，同时不虚增循环体 | `max_adaptive_bin_extent` 在 trace 期算出最宽 bin 作静态上界；`if ih_start+kh<ih_end and iw_start+kw<iw_end` 跳过界外 lane | 无（语言/编译器级约束） | **可移植**（NPU 同样只接受静态 `T.serial` 边界） |
| 2 | **fp32 比较域 + 单次输出舍入** | 与 PyTorch 比较语义一致 | `accum_dtype="float"`，逐元素 `T.cast(x,"float")` 比较，末尾一次 `T.cast(result,dtype)` | 无（纯数值语义，max 下无损） | **保留**（语义层） |
| 3 | **bin 大小整数边界取整** | 与 PyTorch adaptive 语义逐位一致 | bin 边界由整数 `adaptive_bin` 算出（非比例近似） | 无 | **保留**（语义层） |
| 4 | **显式 NaN 追踪 + 结果覆盖** | 复刻 PyTorch"NaN 胜出"语义，不依赖 `T.max` 的 NaN 行为 | `has_nan \| T.isnan(val)`，结果 `if_then_else(has_nan, NaN, max_val)` | 无（纯数值语义） | **保留**（语义层，但需在 NPU 上向量化实现） |
| 5 | **一维 block 网格 + 输出元素扁平化映射** | 把 `(N,C,oH,oW)` 四维问题压成一维可并行任务 | `T.ceildiv(total_output, block_m)` + `%`//`//` 解码 | GPU block 网格（非 hardware-specific） | **等价替换**（一维 Kernel） |
| 6 | **`T.Parallel(block_m)` 线程级 lane 并行** | 每个 CUDA 线程处理一个输出元素，隐藏 gather 延迟 | CUDA thread/warp 两级并行模型 | **CUDA threads / warp**（NPU 无等价） | **重新设计**（→§0.6 R1） |
| 7 | **lane 内标量串行 bin 取最大值** | 单线程顺序比较、寄存器驻留 | `max_val` 寄存器 + 逐元素 GM gather | **GPU 大寄存器堆 + 寄存器比较链** | **重新设计**（→§0.6 R2） |
| 8 | **逐元素 GM gather（无 tile 搬运 / 无 coalescing）** | 窗口不规则，源码直接按 bin 坐标逐元素取数 | 直接索引 `x[batch,c,ih,iw]` | GPU 的 L1/L2 缓存吸收重复访问 | **重新设计**（→§0.6 R3；NPU 侧应走 UB tile 搬运 + 向量化） |
| 9 | **`functools.lru_cache` factory 缓存 + `_kernel_cache` 实例缓存** | 避免重复 trace/编译 | host 侧缓存 | 无 | **保留**（host 侧） |
| 10 | **`adaptive_bin` / `max_adaptive_bin_extent` trace 期常数求值** | 把 bin 算术提前到 trace 期，运行期只剩边界差 | Python 整数运算（`adaptive_bin` 注释 `Evaluated while tracing`） | 无 | **保留**（host/trace 侧） |

> 识别不出来 ≠ 不存在。以上 10 项为源码中可见的全部优化/结构性选择（含 max 特有的 NaN 追踪 #4）；未识别的项会在迁移中被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（adaptive bin 取最大值、bin 整数边界、输出尺寸规则） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| **NaN 胜出 + ±inf 最大值语义（#4）** | 语义 | 无 | — | **保留** | 数值语义；NPU 侧用 `T.vcmp(...,"ne")`（`v!=v` 探测 NaN）+ `T.vselect` 向量化实现（语义必须保持，不得依赖 `T.vmax/reduce_max` 的 NaN 行为——该行为**未文档化**） |
| I/O 契约（NCHW→NCHW、fp16/bf16、同 dtype、无 indices） | 语义 | 无 | — | **保留** | 同上；layout 是契约 |
| fp32 比较域 + 单次输出舍入（#2） | 语义 | 无 | — | **保留** | 数值语义；NPU Vector 支持 fp32 `T.vmax/T.vcast`（`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 fp32 ✓、`数据类型转换操作/T.vcast.md` §2.2.1） |
| 非整除重叠 bin、空 bin 不存在 | 语义 | 无 | — | **保留** | 语义层 |
| 静态边界 bin 循环 + 守卫（#1） | 算法 | 无（语言约束，非硬件） | 有（同款约束） | **保留** | NPU `T.serial` 同样只接受静态边界；守卫生成同构 |
| `adaptive_bin` / `max_adaptive_bin_extent`（#10） | 算法/host | 无 | 有 | **保留** | trace 期 Python 整数算术，host 侧不触碰张量内容 |
| lru_cache / `_kernel_cache`（#9） | host | 无 | 有 | **保留** | host 缓存 |
| 一维 block 网格 + 输出扁平映射（#5） | 优化 | GPU block 网格 | 有（一维 `T.Kernel`，NPU 支持 ≤2 维） | **等价替换** | migration-analysis.md §5.3「block×thread 两级并行 → block 级任务划分 + 向量化元素并行」；`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3（NPU ≤2 维，1 维合法） |
| **`T.Parallel(block_m)` 线程级 lane 并行（#6）** | 优化 | CUDA threads / warp | **无** | **重新设计** → §0.6 R1 | NPU 无 CUDA thread 概念；每 lane 一个输出元素会退化为**标量**执行（pattern-library `layout.md` PL-1.2 实测窗口累加 98.3% 标量同源） |
| **lane 内标量串行 bin 取最大值（#7）** | 优化 | GPU 寄存器堆 + 寄存器比较链 | **无直接等价** | **重新设计** → §0.6 R2 | migration-analysis.md §5.3「warp shuffle 归约 → Vector 归约（重新设计）」「register 累加 → UB 中间缓冲」；NPU 侧改成 C 轴向量 `T.vmax` + 向量 `vcmp/vselect` NaN 追踪 |
| **逐元素 GM gather（#8）** | 优化 | GPU L1/L2 缓存 | **无等价（且有害）** | **重新设计** → §0.6 R3 | NPU 逐元素 gather 无 coalescing、无 cache 吸收；应改为 UB tile 搬运 + 向量化（PL-1.2/PL-1.4） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目 / `docs/Tilelang.language/` 条目 / pattern-library 实测条目（见上表「依据」列）。
> **舍弃项：无。** 源码未使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async 等 GPU 专有手段，故无舍弃条目。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（每项含「源方案 → NPU 新算法 → 语义保持论证」）。
**重设计输入含 §1.6.0 算法调研结论**：调研选定「直接自适应 bin 取最大值 + C 轴向量化 + 显式 NaN 向量追踪」算法族
（与源算法同族，但**归约单元由标量改为 C 向量、NaN 追踪由标量 bool 改为向量掩码**），故 §0.6 的重设计直接承接调研结论；
源算法已识别优化手段（§0.4）的优化意图承接见每项「承接」行。

**重设计项 R1：并行结构重设计（线程级 lane 并行 → 一维 Kernel + C 轴向量任务划分）**

- **源方案**：`T.Kernel(ceildiv(total_output,block_m), threads=threads)` + `T.Parallel(block_m)`，每 CUDA 线程处理一个输出元素。
  性能意图：把 `N*C*out_H*out_W` 个输出元素摊到大量线程上并行，靠 warp/线程切换隐藏逐元素 GM gather 延迟。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`；**逻辑任务单元 = `(n, o_h, c_block)`**，
  其中 `c_block` 是长度 `CH` 的**通道向量块**；`num_logical = N * out_H * ceil(C/CH)`（**一般网格**；
  退化路径 C4/C4b 的单元 = 整张 `(n, c_block)` 平面，`num_logical = N × ceil(C/CH)`，见 §5.5 ① / §1.6.1 C4/C4b）。
  单元内以 **C 轴为向量维**、`o_w` 为编译期展开的 Python 循环、`o_h` 由单元 id 运行时解码。
  `num_logical > vector_cores` 时按 persistent grid-stride 模式挂 `T.serial` 核内循环（承接源方案「把任务摊到并行单元」意图；见 §5.5）。
- **语义保持论证**：任务划分只是把互不相关的输出元素分给不同单元，**不改变任何窗口的取值集合与比较结果**
  （每个输出仍对其 bin 取最大值）；无跨单元归约，无边界语义变化。
  承接：源 #5 的一维扁平映射被保留为「一维 Kernel」，#6 的 lane 并行意图由「单元并行 + C 轴向量并行」承接。

**重设计项 R2：归约重设计（per-lane 标量 bin 取最大值 + 标量 NaN → UB 内 C 轴向量取最大值 + 向量 NaN 追踪）**

- **源方案**：每个输出元素在寄存器里做 `max_val = T.max(max_val, T.cast(x[...],"float"))` 的标量比较链，
  并行 `has_nan = has_nan | T.isnan(val)` 标量 bool 累积。
  性能意图：寄存器驻留、单线程顺序比较、避免跨线程归约。
- **NPU 新算法**：C 轴向量化（向量轴 = C，§1.6.3）：
  1. 逐元素 cast 改为 **tile 级向量 cast**：`T.vcast(row_dtype, row_f32, round_mode="rint")` 一次把整行 `(W_in, CH)` 提到 fp32。
  2. 取最大值改为 **C 轴向量最大值**：`T.vmax(acc[o_w], row_f32[iw0+kw], acc[o_w])`；
     `acc`（`out_W × CH` fp32）经 `T.vbrc(-T.infinity("float32"), acc)` 初始化为 `-inf`；
     `kh` 为 `T.serial(max_bin_h)` 静态边界 + `if kh < bin_h` 守卫（承接 #1）；
     `kw` 为 Python 编译期展开（`bin_w(o_w)` 在 trace 期是常数，见 §1.6.3）。
  3. **NaN 追踪向量化（max 特有，语义关键）**：`h_acc`（`out_W × CH` fp32，`T.clear` 置 0）与 `acc` 并行累积：
     对每个元素切片依次执行
     `T.vcmp(val_slice, val_slice, cond_bool, "ne")`（`v!=v` ⟺ NaN，**不依赖 vmax 的 NaN 行为**）与
     `T.vselect(cond_bool, val_slice, h_acc_slice, h_acc_slice)`（`h_acc` 在遇 NaN 时被写入 NaN，否则不变）；
     单元末尾 `T.vcmp(h_acc, h_acc, cfinal, "ne")` → `T.vselect(cfinal, h_acc, acc, out32)`，
     即 `has_nan ? NaN : max`（`h_acc` 本身即 NaN，无需构造 NaN 常量）。
  4. 输出：`T.vcast(out32, acc16, round_mode="rint")`（单次舍回 dtype）→ `T.transpose(acc16, acc_t, [1,0])` → `T.copy` 写回。
- **语义保持论证**：
  - **数学等价**：分块（`kh`/`kw` 循环）不改变取最大值的元素集合；C 轴是**独立轴**（每通道独立取最大值，§0.1），
    把 `c` 放进向量 lane 只是 SIMD 数据并行，**不改变任何 `c` 的取值集合**。
  - **NaN/±inf 语义核对**（逐条对齐 §0.1 实测）：
    - 窗口含 NaN → `cond` 在对应 lane 置真 → `h_acc` 变 NaN → `cfinal` 真 → 输出 NaN ✓（NaN 胜出）；
    - 窗口 `{+inf,-inf}` 无 NaN → `cond` 恒假 → `h_acc` 保持 0（非 NaN）→ `cfinal` 假 → 输出 `acc`；
      `T.vmax(-inf, +inf)=+inf`、`T.vmax(+inf,-inf)=+inf` → 输出 `+inf` ✓（与实测一致）；
    - 该设计**只用 `v != v` 判 NaN**，与求和类探测（会把 `{+inf,-inf}` 误判为 NaN）本质不同，故正确。
  - **dtype/舍入**：与源一致（fp32 比较域、末尾单次 `rint`）；`vcast` fp16/bf16→fp32 为无损上转，
    fp32→fp16/bf16 用 `rint`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1）；对 max 结果无损（输出即输入元素）。
  - **边界语义**：非整除重叠 bin（整数差，与源同式）、空 bin 不存在——全部不变。
  - 承接：源 #2（fp32 + 单次舍入）与 #4（NaN 追踪）分别由「fp32 向量比较域」与「向量 `vcmp`+`vselect` NaN 追踪」承接；
    #7 的「寄存器比较链」意图由「UB fp32 向量最大值累加器」承接。

**重设计项 R3：数据搬运重设计（逐元素 GM gather → UB tile 搬运 + 核内融合转置）**

- **源方案**：直接在 GM 上按 `x[batch,c,ih_start+kh,iw_start+kw]` 逐元素取值。性能意图：接口简单；
  GPU 靠 L1/L2 缓存吸收重叠行/列的重复访问。
- **NPU 新算法**：按 C 轴向量化所需布局（C 最内连续）搬运：
  1. GM→UB：按 `(real_c, W_in)` 行片整片搬运 `T.copy(x[n, c0:c0+real_c, ih, 0:W_in], row_src[0:real_c,0:W_in])`（同一 `c` 的整行连续）；
  2. UB 内**融合转置链**：`T.transpose(row_src, row_dst, [1,0])` 把 `(CH, W_in)` 变成 `(W_in, CH)`（C 最内）；
     依据 pattern-library `PL-1.1-transpose-chain`（核内二轴交换链，实测 ~µs 级；`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 仅支持二轴交换）；
  3. 转置后 C 轴连续 → `T.vcast` 到 fp32 → 向量取最大值；
  4. 输出侧：`out32 (out_W, CH)` 经 `T.vcast` + `T.transpose([1,0])` 变 `(CH, out_W)` 后 `T.copy` 回 `out[n, c0:c0+real_c, o_h, 0:out_W]`。
- **语义保持论证**：搬运/转置是**纯数据重排**（双射），不改变元素值、不改变取值集合；输出写回后与 NCHW 契约一致。无精度影响（不引入任何算术）。
  承接：源 #8「靠缓存吸收重复访问」的意图由「UB tile 常驻 + 重叠行只搬一次（单元内 `kh` 循环复用 row 片）」承接。

**重设计项 R4：算法替代/结构特化（退化网格）**

> 该项属「算法替代」模式（migration-analysis.md §6.2），采纳项与 §1.6.0 调研结论一致、并在 §1.6.1 完成等价性论证与机器验证。

- **源方案**：对所有输出网格一律走双重 bin 循环（含 `out_H=out_W=1` 全局池化、以及 `out==in` 恒等情形）。
- **NPU 新算法**（factory 级结构分发，形状静态可判定；仅在 UB 预算内启用）：
  - **C4（全局池化，`out_H==1 and out_W==1`）**：bin(0,0) = `[0,H_in)×[0,W_in)` 即整个平面 →
    按 `(CH, H_in, W_in)` 连续块搬运 + `T.vcast` fp32 + `T.reduce_max(buf, gmax, dim=2)` + `T.reduce_max(gmax, gout, dim=1)`
    （`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1 fp32 ✓，`clear=True`）；
    **NaN 需显式探测（不依赖 `reduce_max` 的 NaN 行为）**：`T.vcmp(flat_f32, flat_f32, mask_bool, "ne")` →
    `T.vcast(mask_bool, mask_f32, "rint")`（i1→f32 ✓）→ `reduce_max` 得 `nmax ∈ {0,1}` →
    `T.vcmp(nmax, zero, cond, "gt")` → `T.vselect(cond, nanbuf, gout, out32)`，其中 `nanbuf = inf - inf`
    （`T.vbrc(inf)` + `T.vsub(nanbuf,nanbuf,nanbuf)`）。
  - **C4b（恒等，`out_H==H_in and out_W==W_in`）**：所有 bin 大小 1 → `out = x`，走纯拷贝路径（UB 中转，同 dtype 不 cast）。
- **语义保持论证**：见 §1.6.1（C4/C4b 均与基线公式**逐位等价**，`verify_equiv.py` 132 用例全 `EQUIV_PASS`）。
  C4 的 NaN 路径用「掩码归约 + `v!=v` 探测」，对 `{±inf}`、全 NaN、部分 NaN 均与基线一致（`reduce_max` 的 NaN 行为无关紧要——无论它传播还是忽略，`cond` 都独立给出正确判决）。
- 承接：源方案「一般 bin 循环」在非退化网格上保持不变（不存在优化意图丢弃）；退化网格特化**不改变语义**，仅减少冗余循环。

**重设计后的 NPU 算法一句话总结**：NCHW 进 / NCHW 出，核内按 `(n,o_h,c_block)` 单元把 C 轴作为向量轴做自适应 bin 的 fp32 向量取最大值
（`T.vmax` + `T.vcmp/vselect` 显式 NaN 追踪），用核内二轴转置把 C 搬到最内连续轴；退化网格（全局池化 / 恒等）走结构特化路径。

### 0.7 标杆实现

- 源算子路径（获取 golden 语义依据）：`/workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_max_pool2d.py`（`AdaptiveMaxPool2dKernel`）。
- 源仓参考实现 / 测试基准：
  - `examples/TileOPs/tests/ops/test_pool.py`（NPU 侧移植后用例，`ref_program = F.adaptive_max_pool2d(input, size)`，torch CPU 参考；atol=rtol=0）；
  - 源仓同文件（GPU）`tests/ops/test_pool.py`。
- **golden 以 §0.1 语义为唯一依据**，直接使用 `torch.nn.functional.adaptive_max_pool2d`（优先移植源仓基准），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。
- 规格来源：`examples/TileOPs/tileops/manifest/pool.yaml` 的 `AdaptiveMaxPool2dFwdOp`（4 组 workloads；`ref_api` 已声明 `return_indices=False`）。
- 基准来源：`examples/TileOPs/benchmarks/ops/bench_pool.py` 中的函数 `test_adaptive_max_pool2d_bench`。

---

## 1. 概述

### 1.1 算子名称

`_adaptive_max_pool2d_kernel`（迁移自 TileOPs `AdaptiveMaxPool2dFwdOp` / `AdaptiveMaxPool2dKernel`）。

### 1.2 功能描述

自适应二维最大池化前向：把 NCHW 输入的空间轴按 PyTorch adaptive-bin 规则（`[floor(o*in/out), ceil((o+1)*in/out))`）划分，
每个输出元素为其 bin 内的最大值；窗口内任一元素为 NaN 时输出 NaN。

### 1.3 数学公式

$$
\text{out}[n,c,o_h,o_w]=\max_{i_h\in\text{bin}_H(o_h)}\ \max_{i_w\in\text{bin}_W(o_w)}\ x[n,c,i_h,i_w],
\qquad
\text{bin}_H(o_h)=\big[\lfloor o_h H_{in}/H_{out}\rfloor,\ \lceil (o_h+1)H_{in}/H_{out}\rceil\big]
$$

（`bin_W` 同理；`H_out = H_in if oH is None else oH`。NaN 胜出：$\exists$ NaN ∈ 窗口 ⇒ 输出 NaN。）

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，见 §0.6）** 单步 + 窗口归约类算子，纯 Vector：

1. host/trace 期（factory 内）：由 `(H_in,out_H)`/`(W_in,out_W)` 算出每个 `o_h`/`o_w` 的 bin 边界 `(start,end)`（`adaptive_bin`）
   与静态上界 `max_bin_h = max_adaptive_bin_extent(H_in,out_H)`、`max_bin_w`（编译期常数）。
2. 逻辑任务单元 `(n, o_h, c_block)`（`CH` 通道向量块）；单元内：
   - `acc`（`out_W × CH` fp32）经 `T.vbrc(-inf)` 置 `-inf`；`h_acc`（`out_W × CH` fp32）经 `T.clear` 置 0（NaN 追踪器）；
   - `for kh in T.serial(max_bin_h)`：若 `kh < bin_h(o_h)`（运行时守卫），搬入输入行 `(CH, W_in)` → 核内转置 `(W_in, CH)` → `vcast` fp32；
     随后对每个 `o_w`（编译期展开）把该 `o_w` 的 W-bin 片段逐切片 `vmax` 进 `acc[o_w]`，同时 `vcmp`+`vselect` 累积 `h_acc[o_w]`；
   - 末尾 `vcmp(h_acc,h_acc,"ne")` + `vselect` → `has_nan ? NaN : max` → `vcast` 回输出 dtype → 转置回 `(CH, out_W)` → 写回 GM。
3. 退化网格（`1×1` 全局池化 / `out==in` 恒等）走 §0.6 R4 的结构特化路径。

**与源算法的结构差异**（来源标注）：
- 归约单元：源「每输出元素一个线程 + 标量比较链」→ NPU「每单元一个 C 向量块 + `T.vmax` 向量比较」（§0.6 R2）。
- NaN 追踪：源「标量 bool `has_nan \| isnan(val)`」→ NPU「C 向量掩码 `vcmp`+`vselect` 累积」（§0.6 R2；语义不变）。
- 访存：源「逐元素 GM gather」→ NPU「UB tile 搬运 + 核内融合转置」（§0.6 R3）。
- 并行：源「block×thread 两级 + lane/元素」→ NPU「一维 Kernel + `(n,o_h,c_block)` 单元 + C 轴向量 lane」（§0.6 R1）。
- 退化网格特化：源无 → NPU 有（§0.6 R4，见 §1.6.1）。
- **不变**：bin 整数边界算术、`kh`→`kw` 的取值集合、NaN 胜出 + ±inf 最大值语义、fp32 比较域 + 单次输出舍入、静态边界 bin 循环 + 守卫、NCHW 契约。

### 1.5 数据流图

```
GM[x: (N,C,H_in,W_in) fp16/bf16]
   │  T.copy 行片 (CH, W_in)  [同一 c 的整行连续]
   ▼
UB[row_src: (CH, W_in) dtype] --T.transpose [1,0]--> UB[row_dst: (W_in, CH) dtype] --T.vcast(rint)--> UB[row_f32: (W_in, CH) f32]
   │                                                                                                            │  T.serial kh (静态界+守卫)
   │                                                                                                            ▼
   │                             UB[acc: (out_W, CH) f32 <- vmax 切片]  ∥  UB[h_acc: (out_W, CH) f32 <- vcmp/vselect NaN 追踪]
   │                                                                  │  vcmp(h_acc,h_acc,"ne") + vselect -> has_nan?NaN:max ; vcast(rint)
   │                                                                  ▼
   │                                                        UB[acc_t: (CH, out_W) dtype] --T.transpose [1,0]
   ▼
GM[out: (N,C,H_out,W_out) same_as_input]   ← T.copy 行片 (CH, out_W)
（退化路径 C4：GM → UB 连续 (CH,H*W) → vcast f32 → reduce_max(dim=2→1) + NaN 掩码归约 → vselect → vcast → GM）
（退化路径 C4b：GM → UB 整平面 (CH,H*W) → GM 纯拷贝）
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = **同一数学语义的算法族**（所有能算出 adaptive-bin 最大值的结构）：**基线 = 源算法**（直接自适应 bin 取最大值，逐元素标量）。
> 本算子属**窗口/归约类** → 完整调研（R1 候选表 + R3 四口径表 + R4 逐候选评估）。
> 参考表命中行：`algorithm-candidates.md` 的 `ALG-pooling`（family=pooling）——`R1: avg pool = 常数权重卷积；sum pool × 常数 = avg pool`；`R2: 滑窗天然流式`；`R3: 窗口重叠数据的重复读`；`R4: C 轴向量轴；跨步系数`；`known_impl` 指向 `avg_pool2d` 系列，`kb_links` = `PL-1.2-caxis-accum / PL-1.3-host-permute / PL-1.4-tiling-heuristic`。
> **命中行已全部评估**（见 R1/R3/R4）；**但 ALG-pooling 的 R1 命中项（常数权重卷积/GEMM）对 max 不成立**（见 R1 候选 1/6——max 是非线性算子），这是本算子与同族 avg 池化的关键差异。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算法**（直接自适应 bin 取最大值，逐元素标量 + 标量 NaN 追踪） | $\max_{kh}\max_{kw}x[ih_0+kh,iw_0+kw]$，NaN 胜出 | — | — | ✅（基线） |
| 0' | **基线向量化**（C 轴向量 `vmax` + 向量 NaN 追踪） | 同 #0（取值集合与判决完全一致），仅 SIMD 化 | **数学恒等**（C 轴是独立轴，仅数据并行） | 标量链 → 向量链（`/CH` 次向量 op） | ✅（选定） |
| 1 | 常数权重卷积 / im2col+Cube GEMM（ALG-pooling `R1` 命中行） | $\text{out}=W_0 \otimes x$（0/1 权重矩阵） | **❌ 不等价**：$\max$ 是非线性算子，**不存在权重矩阵 $W_0$ 使 $\max$ 等于线性加权和**；该命中行仅对 avg/sum pool 成立 | — | ❌（数学不等价） |
| 2 | 可分离 2D（H/W 两轴分两遍取最大值） | $\max_{ih}\big(\max_{iw}x\big)$ | 数学恒等（max 可交换/可结合） | 名义可复用行部分最大值；但见 R3 | ✅（量化后弃） |
| 3 | 前缀最大值 + 差分 | $P[b]-P[a]$，$P$ = 前缀最大值 | **❌ 不等价**：$\max$ **无逆元**（不同于求和），$\max P[b],P[a]$ 的差**不等于**区间最大值（例：$x=[5,1,3]$，区间 $[1,3)$ 的 max=3，而 $P[3]{-}P[1]=5{-}5=0$） | — | ❌（数学不等价） |
| 4 | 退化网格结构特化（`1×1` / 恒等） | `1×1`: $\max_{\text{all}}x$；恒等: `out=x` | 数学恒等（bin(0,0)=全平面；单位 bin 的最大值=元素本身） | 全局池化：$H{\cdot}W$ 次向量 `vmax` → 1 次向量归约 | ✅ |
| 5 | sum-pool × 常数（ALG-pooling 命中行） | $(\sum x)\cdot c$ | **❌ 不等价**（该式给 avg/sum，不给 max） | — | ❌（数学不等价） |
| 6 | 归约结构优化：`max` 值与 NaN 判决**在同一轮扫描内融合**（源已是同轮） | 单轮：`vmax` + `vcmp/vselect` | 数学恒等（同轮与分轮结果相同） | 避免第二遍扫描（源 #4 已是同轮，NPU 沿用） | ✅（并入 #0'） |

**R1 初判小结**：唯一数学等价的化简结构是「**退化网格特化**」（#4）与「**NaN 判决融合**」（#6，源已有）；
与 avg 池化不同，max 的**全部代数化简路径（#1/#3/#5）因 `max` 非线性/无逆元而不成立**——这是 max 归约相对 sum 归约的本质限制（非 API 或性能问题，是数学事实）。

**R2 在线算法**：**有**（且基线本身已是单遍）。结构判据：算子可分解为 **running 最大值**（窗口内的最大值）
- **running NaN 标记**（窗口内是否已见 NaN），无跨元素依赖的中间统计量，故天然存在单遍流式变体——
每个输入元素读一次、参与其唯一归属（非整除重叠时若干）输出 bin 的 `vmax`/NaN 判决，运行状态为 `O(out_W·CH)` 的累加器。
**收益口径**：扫描遍数 1（无两遍式）、中间缓冲 `O(out_W·CH)`（远小于 `O(H·W)` 的物化）、UB 驻留完全可行（见 §4.5 预算）。
迁移任务佐证：源算法 §0.4 #6/#7 已是 running 比较（`max_val = T.max(max_val, val)`），是「存在在线变体」的直接证据。
结论：**在线单遍变体 = 采用（= 基线结构），无第二遍扫描/无中间矩阵物化**。

**R3 复杂度对比**（逐候选四口径；本算子是**带宽受限**算子，访存/遍数为主导项，不只看 FLOPs）：

以 `spp-6x6`（`N=2,C=128,H=W=56,out=6×6`，`scan_H=scan_W=60`，fp16）为口径示例；口径：FLOPs 计「元素访问次数」（一次最大比较 = 1），
访存 Bytes = GM 读 + GM 写（**不含** UB 内搬运；中间缓冲不落 GM），扫描遍数 = 输入 GM 读取遍数，中间缓冲峰值 = 片上驻留量。

| 算法候选 | FLOPs（比较次数） | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|--------------------|---------|-------------|---------------------|
| 0 基线（直接标量） | `N·C·scan_H·scan_W` = 921 600 | `(N·C·H·W + N·C·out_H·out_W)·2` = 1.624 M | 1 | `O(1)`（寄存器标量） | 元素级并行 `N·C·out_H·out_W = 9216`；**无跨核归约** |
| 0' 基线向量化（**选定**，C 轴） | 921 600（`/CH` 次向量 `vmax`） | 同上 1.624 M（+ 重叠行 ~7% 重复读） | 1 | `acc+h_acc (out_W·CH)f32` + `row(W_in·CH)` ≈ 20–45 KB | 单元并行 `N·out_H·ceil(C/CH)`；**无跨核归约** |
| 1 常数权重卷积 / GEMM | **不适用**（非线性，无法表达） | — | — | — | — |
| 2 可分离 2D | `H_in·scan_W + scan_H·out_W` = 3360+360 = **3720/平面**（> 3600 = scan_H·scan_W） | 同上（+ 行部分最大值缓冲 `H_in·out_W·CH`） | 1 | + `R(H_in·out_W·CH)` | 同上，但多一中间缓冲 + 两遍串行 |
| 3 前缀最大值 + 差分 | **不适用**（max 无逆元） | — | — | — | — |
| 4 退化特化（`1×1`） | **1 次向量归约**（vs 基线 `H·W` 次向量 `vmax`） | 同上 | 1 | `(CH, H·W)` 块 + NaN 掩码块 | 同上 |

> 口径含：GM 输入读 + 输出写；不含：UB 内搬运/转置/NaN 追踪（片上，计入 §1.6.0 估算行的发射项）。候选 2 的额外中间缓冲已单列。
> **基线在表内**（#0）；选定候选 #0'（基线向量化，与调研结论一致）。候选 1/3 因**数学不等价**不参与复杂度对比（标注 N/A）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/`/`examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线（标量） | **Vector 单元浪费**（逐元素标量比较，PL-1.2 同族实测窗口累加 98.3% 标量） | 寄存器 | — | 静态（`max_bin_*` 界） | 无 | ❌ 计算单元失配（重设计为 #0'） |
| **0' 基线向量化（C 轴，选定）** | **Vector 逐元素/归约** ✓（`T.vmax` fp32 ✓ / `T.vcmp` fp32 ✓ / `T.vselect` fp32 ✓ / `T.reduce_max` fp32 ✓） | `acc+h_acc`+`row` ≈ 20–45 KB ≪ UB 192 KB（`CONST-capacity-910B2C`） | **C 轴恒为 8 的整数倍**（C=8/16/64/96/128/2048；fp16 向量宽 8）；`W_in`=55/56/57 仅作 row 长度，不要求整除 | ✓ 全部编译期静态（`adaptive_bin` 为 trace 期整数；动态量仅运行时标量 `o_h`/`kh` 守卫） | 无跨引擎流水；退化网格可 reduce 融合 | ✅ **选定** |
| 2 可分离 2D | Vector ✓ | 需 + 行部分最大值缓冲（`H_in·out_W·CH`，spp CH=64 时 ≈ 86 KB） | 同 #0' | ✓ | 两遍串行 | ❌ **运算量更大**（3720 > 3600/平面）+ 额外缓冲；详见 §1.6.1 |
| 4 退化特化 | Vector reduce ✓ | `(CH,H·W)` 块 + NaN 掩码块 ≈ 35 KB（resnet C4，`CH=min(C,64)=64`，见 §4.5） | ✓ | ✓ | 单归约 | ✅ 采纳（`1×1` / 恒等） |
| 1 常数权重卷积 / 6 GEMM | **不适用**（max 非线性，非 MAC 密集） | — | — | — | — | ❌ 数学不等价，无需亲和评估 |

**R4 弃选理由的证据引用（negative-claim-evidence 口径）**：
- 候选 1/3/5：**数学事实**（`max` 非线性、无逆元），非 API/性能断言——不作 API 检索需求（不存在可检索的"最大化简公式"）。
- 候选 2：运算量为**我方独立复算**（每平面 `H_in·scan_W + scan_H·out_W` vs `scan_H·scan_W`，见 R3 表）。
- 候选 0 的「标量占比」反例引用 pattern-library `layout.md` PL-1.2 诊断触发器（「窗口累加 98.3% 标量源于 `j·sW`（sW≠1）跨步系数阻碍向量化」）。
- 候选 0'/4 的 dtype 支持矩阵来自 `docs/Tilelang.language/`（`T.vmax`/`T.vcmp`/`T.vselect`/`T.reduce_max`/`T.vcast` 各 §2.2.1 表）。

**设计期估算（D-2 roofline）** —— 协议见 `_shared/standards/hardware-cost-model.md` §2，常数引用 `pattern-library/constants.md`：

| workload | 流量项（`bytes ÷ 有效带宽`） | 发射项（向量 op 数） | 容量项 | 估算下界 |
|----------|------------------------------|----------------------|--------|---------|
| `global-1x1`(8,2048,7,7) | 1.638 MB ÷ 1.26 TB/s = **1.30 µs**（HBM 3:1 混合地板，`CONST-mte2-degradation`）；L2 驻留口径（sets=1）÷ 2.07 TB/s ≈ 0.79 µs | C4 特化：256 单元 × ~13 op ≈ 3.3 K op | 0（无跨引擎往返） | **≈ 1.3 µs** |
| `spp-6x6`(2,128,56,56) | 1.624 MB ÷ 1.26 TB/s = **1.29 µs** | 14400 元素访问 × 3（vmax+vcmp+vselect）= **43.2 K 向量 op** | 0 | **≈ 1.3 µs** |
| `nondiv-7x7`(2,64,55,57) | 0.815 MB ÷ 1.26 TB/s = **0.65 µs** | 7686 × 3 = **23.1 K 向量 op** | 0 | **≈ 0.7 µs** |

> **估算下界: ≈ 1.3 µs（流量 1.29–1.30 + 发射「未文档化假设」 + 容量 0，引用 `CONST-mte2-degradation` / `CONST-capacity-910B2C`）；端到端实测地板大概率由固定开销主导（NPU kernel launch + Event 计时 ~50–60 µs，`examples/TileOPs/docs/gpu_to_npu_adaptation.md` C12 行）。**
> **发射项口径注**：`CONST-vector-launch-overhead`（`~0.5µs/op`，toolchain 2026-09-09 / 3a214cde）是 **bn=512 宽块大 op** 的实测值，
> 对本设计 ≤128 元素的窄向量 op **不能直接外推**（按 0.5µs/op 会得 ~21 ms 的荒谬值）——窄向量 op 代价**未逐点实测**，标注「未文档化假设」；
> **方向证据已有实测**：`layout.md` PL-1.4 同族池化实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs」（更宽 C 向量更优），与 `CONST-vector-launch-overhead`「宽块优于窄块 op 翻倍」方向一致。
> **容量项为 0**：本设计无 `store_fixpipe` 类跨引擎中间矩阵往返（纯 Vector 单 kernel）。
> **注（max 相对 avg 的额外发射代价）**：以 `v != v` 探测 NaN 需每元素额外 `vcmp + vselect` 两条向量 op（avg 只需 1 条 `vadd`），
> 故 spp 发射量约为同族 avg 的 3×；这是保持「NaN 胜出」语义的**必要代价**，非可省项（见 R2/R4 与 §1.6.2）。

**调研结论**：选定算法族 = **「直接自适应 bin 取最大值 + fp32 比较域 + C 轴向量化 + 显式向量 NaN 追踪 + 核内融合转置（NCHW→C 最内）」**
（R3 表候选 #0'；另采纳候选 #4 退化特化 → §1.6.1 **C4/C4b**，候选 #6 融合扫描并入 #0'；两项采纳与 §1.6.1 优化结论**逐项一致**）。
关键依据：R3 表显示 #0' 的 FLOPs（= `N·C·scan_H·scan_W`，与 manifest `roofline.flops` 逐字一致）与访存量（= manifest `roofline.bytes`）均为各候选最小量级，
候选 2 不降反升，候选 1/3/5 数学不等价；R4 表显示 #0' 唯一同时满足「Vector 计算单元匹配 + C 轴向量整除 + 全静态边界 + 无跨核归约 + 容量充裕」。
**与基线的结构差异一句话**：数学公式、取值集合与 NaN 判决完全不变，执行单元由「每输出一线程的标量比较链」改为「`(n,o_h,c_block)` 单元的 C 轴向量比较器 + 向量 NaN 追踪器」。
**调研范围声明**（确有一项结构特化被采纳，但**无更低复杂度算法族**）：
本调研覆盖 ① `algorithm-candidates.md` 的 `ALG-pooling` 命中行全部候选（R1 常数权重卷积 / sum-pool×常数 / R2 滑窗流式 / R3 窗口重叠重复读 / R4 C 轴向量轴）——
并**逐项判定其对 max 的适用性**（R1 命中项对 max 因非线性不成立）；② 恒等变形空间（可分离、前缀最大值差分、归约融合）；
③ pattern-library 同类实现（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum` / `PL-1.3-host-permute` / `PL-1.4-tiling-heuristic`，均在 `tilelang-op-optimize/references/pattern-library/layout.md`）；
④ 本仓 `examples/` 同类实现（§3.5.3）与**同族前作** `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/`（avg 池化迁移，CASE-pool-adaptiveavgpool2d-migration）。
**互联网检索未使用**（本算子算法族在本地参考表 + 源码 + pattern-library 已完整覆盖，属简单窗口/归约类，无需联网补充）。
**下游约束**：§1.4 算法描述、§1.6.1–§1.6.3 的分析对象 = 选定算法 #0'；§0.5/§0.6 的处置与之**一致**
（§0.6 R2 = #0' 的归约向量化 + NaN 向量追踪、§0.6 R3 = C 轴布局、§0.6 R4 = 候选 #4）；
源算法 §0.4 已识别优化手段的意图承接逐项见 §0.6（**无静默丢弃**：10 项全部有处置，#1–#4+#7+#8 的意图分别由 §0.6 R1/R2/R3 承接）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的算法 #0'（= 基线结构 + C 轴向量化 + 向量 NaN 追踪）。调研候选（R1 表）中初判可采纳的项在此完成四要素论证。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| C4 | **退化网格结构特化：全局池化（`out_H==1 and out_W==1`）** | 一般 bin 双循环取最大值（`bin(0,0)` 由通用公式给出 $\max_{kh<H}\max_{kw<W}x$） | $\max_{\text{all}} x$——按 `(CH, H*W)` 连续块搬运 + 单次 `T.reduce_max(dim=2→1)`；NaN 用 `vcmp(ne)`+`vcast`+`reduce_max` 掩码归约 + `vselect` 判决 | **数学恒等**：`out_H=out_W=1` 时 $\lfloor 0\cdot H/1\rfloor=0$、$\lceil 1\cdot H/1\rceil=H$，故 `bin(0,0) = [0,H)×[0,W)` = 整平面；取值集合完全相同 → **逐位等价**（max 选择元素，无算术） | `global-1x1`（`H=W=7`）：每单元 $H{\times}W=49$ 次向量 `vmax` ×3 op → **1 次向量归约 + 1 次掩码归约**（≈ 15×）；覆盖 4 组 manifest workload 中的最大 workload |
| C4b | **退化网格结构特化：恒等（`out_H==H_in and out_W==W_in`）** | 一般 bin 双循环（每个 bin 大小 1，max 即元素本身） | `out = x`（纯 UB 中转拷贝，同 dtype 不 cast） | **数学恒等**：`out_H=H_in` 时 `bin_H(o_h)=[o_h,o_h+1)`（$\lfloor o_h\rfloor=o_h$、$\lceil o_h+1\rceil=o_h+1$），单位 bin 的最大值 = 元素本身；逐元素恒等（NaN 也随元素原样搬出，语义一致） | 覆盖 op 层 `output_size=None` 的恒等用例与 `(None,…)`/`(…,None)` 半恒等轴；消除全部冗余归约 |

> **无「倒数乘 / 归一化」优化空间**（显式结论）：与同族 avg 池化不同，max 池化**无除法/归一化**（结果即被选元素），
> 故 avg 迁移的 C1（`sum/d → sum·(1/d)`）在本算子**不存在**——§1.6.1 无对应项，非遗漏。这是 `max` 归约相对 `sum` 归约的本质差异（数学事实，非设计选择）。
> 亦**无公共子表达式消除**（每元素仅参与一次比较）、**无高代价算子降代换**（无 exp/rsqrt 类）。

**等价性机器验证（D-1）**：两项采纳优化全部由 `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/verify_equiv.py` 在 **fp64 参照**下对照基线通用 bin 取最大值验证（torch CPU），共 **132 用例**：
- **C4/C4b**（逐位等价）：判据 = 舍入到目标 dtype 后与基线**逐位一致**（0 mismatch）。
- 用例集：C4 用 6 组 shape（含 4 组 manifest 相关 shape + `(1,1,1,1)`）× 6 种数据（rand / nan / inf（同时含 ±inf）/ subnormal / large / zeros）；C4b 用 5 组 shape × 6 种数据；两 dtype（fp16/bf16）。NaN-vs-NaN 计相等，NaN-vs-finite 恒判失败。
- `inf` 组的 `max_abs_diff=nan` 是**两式同为 ±inf 时 `inf-inf=nan` 的显示伪影**（脚本 `mismatch` 判据不受影响），`viol=0` → `EQUIV_PASS`。
- **运行结论**：`viol=0` 全部 132 用例 → `[EQUIV] ALL PASS — 132 cases, 0 failures`。结果表内嵌如下（逐条汇总；kinds = `rand,nan,inf,subnormal,large,zeros`，每行 6 用例）：

```
case-group (item/dtype/shape)                    | kinds | worst max_abs_diff | viol | status
-----------------------------------------------------------------------------------------------
C4 /float16 /(1,8,4,4)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /float16 /(8,2048,7,7)                        |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /float16 /(2,128,56,56)                       |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /float16 /(2,64,55,57)                        |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /float16 /(1,16,10,10)                        |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /float16 /(1,1,1,1)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /bfloat16/(1,8,4,4)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /bfloat16/(8,2048,7,7)                        |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /bfloat16/(2,128,56,56)                       |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /bfloat16/(2,64,55,57)                        |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /bfloat16/(1,16,10,10)                        |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4 /bfloat16/(1,1,1,1)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/float16 /(1,8,4,4)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/float16 /(1,8,9,11)                          |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/float16 /(1,16,5,7)                          |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/float16 /(2,4,9,11)                          |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/float16 /(1,1,1,1)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/bfloat16/(1,8,4,4)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/bfloat16/(1,8,9,11)                          |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/bfloat16/(1,16,5,7)                          |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/bfloat16/(2,4,9,11)                          |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS
C4b/bfloat16/(1,1,1,1)                           |   6   | 0.0e+00 (inf: nan*)|   0  | EQUIV_PASS

criterion: C4/C4b exact (0 mismatch after rounding to target dtype); 'viol' = element mismatches
(NaN-vs-finite always fails).  (* inf-kind max_abs_diff display artifact: inf-inf=nan; viol still 0.)

[EQUIV] ALL PASS — 132 cases, 0 failures
```

> 完整逐条输出（132 行）可复现：`/workspace/venv/bin/python verify_equiv.py`（脚本随算子目录留存，Stage 2 维度 8 会重跑复核）。

**优化结论**：采纳 **2 项**——**C4**（全局池化结构特化）、**C4b**（恒等结构特化），两者均为**逐位等价**且已机器验证（132 用例全 `EQUIV_PASS`）。
候选 C2（可分离）经量化后不采纳（每平面 3720 > 3600，运算量更大 + 额外缓冲）；候选 C1/C3/C5（常数权重卷积 / 前缀最大值差分 / sum-pool×常数）因**数学不等价**不采纳；
「倒数乘」优化在本算子**不存在**（max 无归一化）。**优化后公式（供 §3.1 公式拆解使用）**：

```
一般网格 (out_H×out_W，非退化):
  out[n,c,o_h,o_w] = max_{kh<bin_h(o_h)} max_{kw<bin_w(o_w)} x[n,c,ih0(o_h)+kh, iw0(o_w)+kw]
                     再按 NaN 胜出判决: has_nan ? NaN : max      # has_nan = OR(isnan(x))
退化网格:
  out_H==1 and out_W==1 :  out[n,c,0,0] = max_{all ih,iw} x      # NaN 胜出同一般网格
  out_H==H_in and out_W==W_in :  out[n,c,:,:] = x[n,c,:,:]
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案（§1.4 + §3.3 伪代码）中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
> 替代方案所用 API 均以 `docs/Tilelang.language/` 条目佐证（下表「佐证」列）。

| # | 计算点 | 原实现形态（源 GPU） | 向量替代方案 | 是否替代 | 不可替代理由 / 佐证 |
|---|--------|---------------------|-------------|---------|--------------------|
| 0 | **累加器初始化**（`acc ← -inf`；`h_acc ← 0`） | 源 `max_val = cast(-inf)` 标量（每输出元素 1 次） | `T.vbrc(-T.infinity("float32"), acc)`（整 buffer 广播 `-inf`）；`T.clear(h_acc)`（整 buffer 清零） | ✅ | 佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（scalar→任意 shape 广播）、`内存操作/T.clear.md` §2.2（fp32 ✓）、`创建操作/T.infinity.md` |
| 1 | bin 取最大值 `max_val = T.max(max_val, val)` | 逐元素标量比较（每输出元素一条） | `T.vmax(acc[o_w], row_f32[iw,:], acc[o_w])`（C 轴向量，原地） | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 fp32 ✓；PL-1.2-caxis-accum |
| 2 | 逐元素 `T.cast(x[...],"float")` | 每元素一次标量 cast | `T.vcast(row_dst, row_f32, round_mode="rint")`（整 tile） | ✅ | 佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint` ✓） |
| 3 | **NaN 追踪** `has_nan = has_nan \| T.isnan(val)` | 每元素标量 `isnan` + 标量 bool 或 | `T.vcmp(row_f32[iw,:], row_f32[iw,:], cond_bool, "ne")`（`v!=v` ⟺ NaN，向量 bool）+ `T.vselect(cond_bool, row_f32[iw,:], h_acc[o_w], h_acc[o_w])`（NaN 累积器） | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1（fp32 ✓，"ne" 模式，bool 片上）、`条件操作/T.vselect.md` §2.2.1（fp32 ✓）。**语义正确性**：`v!=v` 仅对 NaN 为真；不依赖 `T.vmax` 的 NaN 行为（未文档化） |
| 4 | 结果 `if_then_else(has_nan, NaN, max_val)` | 每元素标量分支 | `T.vcmp(h_acc, h_acc, cfinal, "ne")` + `T.vselect(cfinal, h_acc, acc, out32[o_w])`（`h_acc` 在遇 NaN 时即 NaN，无需构造 NaN 常量） | ✅ | 佐证：同 #3；`h_acc` 的 NaN 值与源 `cast(NaN)` 等价（输出位型一致） |
| 5 | 最终输出 cast | 每元素一次标量 cast | `T.vcast(out32, acc16, round_mode="rint")`（整 tile） | ✅ | 同上 #2 |
| 6 | 退化网格全平面取最大值 | （源无此路径） | `T.reduce_max(flat32, mid, dim=2)` + `T.reduce_max(mid, gmax, dim=1)`（C4） | ✅ | 佐证：`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1 fp32 ✓，`clear=True` 默认清空 |
| 7 | 退化网格 NaN 探测（C4） | （源无此路径） | `T.vcmp(flat32, flat32, mask_bool, "ne")` + `T.vcast(mask_bool, mask_f32, "rint")`（i1→f32 ✓）+ `reduce_max` 掩码归约 + `T.vcmp`+`T.vselect` | ✅ | 佐证：`T.vcast` §2.2.1 `i1→f32 rint ✓`；掩码为 {0,1} 无 NaN，归约行为无关 |
| 8 | 布局重排（C→最内） | （源无；GPU 靠 coalesced 尾轴） | `T.transpose(src, dst, [1,0])` | ✅ | 佐证：`docs/Tilelang.language/创建操作/T.transpose.md` §2.3（二轴交换 ✓，fp16/bf16/fp32 ✓）；PL-1.1-transpose-chain |
| 9 | **bin 循环 `for kh in T.serial(max_bin_h)`** | `T.serial` 静态界 + 守卫（源同款） | 无（保持 `T.serial` + `if kh<bin_h` 守卫） | ❌ | **理由（tile 级顺序依赖 + 动态边界）**：`kh` 是**累加链的循环携带维**（跨 `kh` 累加到同一 `acc`），且 bin 上界 `bin_h(o_h)` 依赖运行时解码的 `o_h`（**依赖动态 index 的边界**）；TileLang 只接受静态 `T.serial` 边界（源注释明证），无法向量化为单条 op。**但循环体内的比较/NaN 已向量化**（#1/#3），故该循环是 §6 的 block/tile 级调度循环，非逐元素标量循环。 |
| 10 | **`for kw in range(bin_w(o_w))`（Python 展开）** | 内层标量比较 | 展开为多条 `T.vmax`/`T.vcmp`/`T.vselect`（每条一次 C 向量 op） | ✅ | `o_w` 与 `bin_w(o_w)` 在 trace 期是编译期常数（`out_W` 静态、`adaptive_bin` trace 期求值），故用 Python `range` 展开为**向量 op 序列**，不是标量循环 |
| 11 | 单元任务解码 `(n, o_h, c_block)` | — | 无 | ❌ | **理由（block 级索引 / 任务映射计算）**：每单元少量整数 `%`//`//`，非逐元素热点；无对应逐元素向量 API |
| 12 | bin 边界算术 `adaptive_bin` / `bin_h/bin_w` | — | 无 | ❌ | **理由（host/trace 侧元数据）**：trace 期 Python 整数算术（编译期常数），不在 kernel 运行时逐元素执行 |
| 13 | host 侧 `unsqueeze/squeeze/contiguous` | — | 无 | ❌ | **理由（host 侧元数据）**：只改 shape/stride 视图，符合 `ascend-constraints.md` §4；不在 kernel 内 |

**向量化结论**：逐元素计算（累加器初始化 / 取最大值 / NaN 探测 / NaN 判决 / cast / 归约 / 布局重排）**已全部向量化**
（含 §3.3 单元入口的 `acc`/`h_acc` 初始化 → **#0**），替代方案 API 均有 `docs/` 条目佐证。
保留 **4 处** 循环 / 标量点，理由类别：**#9** tile 级顺序依赖 + 动态边界（循环体内已向量化）/ **#11** block 级任务映射 / **#12** host 侧 trace 元数据 / **#13** host 侧视图操作。
（**#10** 标 `✅`——它是「编译期展开为向量 op 序列」，非逐元素标量循环；与表中「是否替代 = ✅」一致。）
无「以实现简单为由保留标量循环」的情形。**§6 循环结构中不出现逐元素标量循环**（与本结论一致）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（NCHW 进 / NCHW 出），但**核内布局与 lane 映射是自由变量**。本节枚举「窗口/池化类」必选候选
> （reference：algorithm-research.md Phase 2 第 3 项「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴」）。
> 迁移任务：源的并行轴选择（GPU 逐元素 lane）是**输入而非结论**，对照 §0.3 独立评估。

**轴质量评分维度**：

| 评分项 | 说明 | 候选 1（NCHW / W 轴） | 候选 2（核内 NHWC / C 轴） | 候选 3（host permute / C 轴） |
|--------|------|----------------------|---------------------------|------------------------------|
| 整除性 | 轴长 vs 向量宽度（fp16 ×8） | `W_in`=55/56/57：56 ✓，55/57 ✗（尾 lane 浪费 1.8%/1.4%）；且**窗口宽 bin_w≈9–10 非 8 倍数** | **C 恒为 8 的整数倍**（8/16/64/96/128/2048）✓ | 同候选 2 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | W 轴 3.6%（55→56）；**段长 9–10 → ~12%** | **0%** | 0% |
| 累加链形态 | 该轴作串行累加维（stride≠1）还是向量维（stride=1） | W 轴作段内归约需**变长段最大值**（非单一 op）或前缀最大（C3 不可行） | C 轴作向量维（stride=1，转置后连续）；空间轴作串行归约维（`kh`/`kw`）——`max` 结构天然 | 同候选 2 |
| repack 代价 | 布局重排代价 | 无 repack，但需变长段处理 | 核内融合转置链（`T.transpose` 二轴交换，PL-1.1 实测 ~5.4µs / 4 次） | **host permute 106–147µs**（PL-1.3，已量化证伪） |
| UB 容量影响 | 布局对驻留缓冲的影响 | 小（row `(W_in)` 级） | `row(·,CH)` + `acc(out_W,CH)` + `h_acc(out_W,CH)` ≈ 20–45 KB（spp CH=64）/ ≈35 KB（resnet C4 CH=64），对照 §4.5 | 同候选 2 |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|----------------|---------|
| 1 | I/O 原生 NCHW | W（行内连续） | 无 | 基线：窗口段长 9–10 非 8 倍数、需变长段最大值（`T.reduce_max` 要求静态形状；变长段需逐元素或 gather）；`W_in`=55/57 非对齐 | ❌ 量化理由：段长 ~12% 尾 lane 浪费 + 无单一向量 op 表达变长段归约，且 `T.reduce_max` 不支持 bf16（`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1） |
| 2 | **核内 NHWC（融合转置链）** | **C（恒为 8 的整数倍）** | 逐行二轴 `T.transpose(src,dst,[1,0])`（PL-1.1 族，实测 ~µs 级） | 尾 lane 0%；转置代价 ~µs 级（远低于算子本体）；UB 增量 ~20–45 KB | ✅ **选定** |
| 3 | host permute NCHW↔NHWC 进出 | C | 两次 host permute | **106–147µs**（PL-1.3，实测）≥ 多数中等算子本体；host 侧连续化会产生新张量（无收益） | ❌ 量化理由：repack 代价 > 算子本体流量（估算 ~1.3µs 流量项）两个数量级，净亏 |
| 4 | Cube/implicit-GEMM（0/1 权重矩阵） | Cube N 轴（out_W） | 权重矩阵构造 | **数学不可行**（max 非线性，无 0/1 权重矩阵可表达）；且 `out_W=6/7 < 16` 违 Cube 分形下限 | ❌ 见 §1.6.0 R1/R4（数学不等价优先于此） |

**布局决策结论**：选定 **方案 #2：核内布局 = 逐行 NHWC（C 最内连续）、向量化轴 = C、repack 路径 = 核内逐行二轴 `T.transpose`（`[1,0]`）+ 输出侧 `T.transpose([1,0])` 回转**。
该结论已同步至：§3.3 伪代码（`row_src/row_dst/row_f32/acc/h_acc` 形状与内层向量维 = C）、§4 内存规划（§4.3/§4.5 的 buffer 形状与预算）、§6 循环结构（累加内层向量维 = C）。
弃选方案量化理由见候选矩阵 #1/#3/#4。

**D-3 实验裁决模式（主选 + 备选 + 裁决计划）**：主选与备选的分歧落在窄向量 op 的每-op 发射代价的**外推区间**——
`CONST-vector-launch-overhead` 只给了 bn=512 大 op 的 0.5µs/op 实测值，本设计 ≤128 元素的窄 op 未逐点实测；
且本算子每元素比同族 avg 多 2 条向量 op（NaN 追踪），op 数放大使该外推更敏感；
判定裕度（`CH=64` 少一半 op 但核数少一半）落在该外推的不确定区间内，故仍保留备选与裁决计划。
**方向证据**：pattern-library `layout.md` **PL-1.4** 直接实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs——更宽 C 向量 + 更少 block 胜过更高空间并行度」，
与 `CONST-vector-launch-overhead`「宽块 op 恒优于窄块 op 翻倍」方向一致；**主选 `CH=64` 恰为 PL-1.4 已实测范围（CH16/32/64）内的最优档**，故主选以实测条目为主证据，不属「未实证的纸面单选」。

- **主选（进 Stage 3）**：`CH = min(C, 64)`（8 的整数倍），逻辑单元 = `(n, o_h, c_block)`，`num_logical > vector_cores` 时 persistent（见 §5.5）。
  代表值：`spp` CH=64 → 24 单元；`nondiv` CH=64 → 14 单元；`global-1x1` CH=64 → 256 单元（persistent，40 核 × 7 任务）。
  **依据**：PL-1.4 实测 CH64 为该池化族最优档（21.6µs，优于 CH32/CH16）；与本设计 C 轴向量化形态同族。
- **备选（≤1 个，结构完整，可直接实现）**：**C4 特化 + `CH = min(C, 8)`（最小 CH）**，即把 op 数放大 8×、核数放大 8×（`spp` → 192 单元）。
  这是「发射代价主导 vs 吞吐主导」的另一端；buffer 形状与主选相同仅 CH 变小（UB 预算同步缩小），分核三要素按 `num_logical=192` 重算（persistent 40 核 × 5 任务）。
  （备选存在的意义**仅限于**：验证比 PL-1.4 已测范围（CH16/32/64）**更窄**的 CH=8 是否使趋势翻转——PL-1.4 未覆盖 CH<16，该点仍属未实测外推。）
- **裁决计划**：代表 shape = `spp-6x6`、`global-1x1`（各 1 组）；指标 = msprof op Task Duration；测量未知常数 = 窄向量 op 的每-op 代价（用 `perf_opt` 下两种 CH 的实测反解）；
  判定阈值 = **备选 latency < 主选 × (1 − 0.10) 才翻转**（10% 稳健性门槛）；裁决结果回写路径 = Stage 4 实测数据经附录补记固化 §1.6.3 判定依据，若翻转则按 `perf-feedback.md` 受控路由回写 DESIGN.md。
- **shape 分派评估**：本 kernel 由 `functools.lru_cache` 按 `(n,c,h,w,out_h,out_w,dtype)` 特化，**支持按 shape 分派**主选/备选——
  裁决计划须评估 `global-1x1` 等大 N 形状走大 CH、`spp/nondiv` 走小 CH 的**按负载分派**（而非全局二选一）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（迁移 prompt 指定；`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`）。

### 2.2 选型理由

- 算子是**纯 Vector**（无 matmul / 无 Cube），仅需逐元素向量运算 + 归约 + 布局重排 + 向量选择——正是 Developer 模式的目标域
  （`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`、`.../elementwise/lerp_tensor/...` 均为 Developer）。
- 无跨引擎流水、无手动 L1/L0 分级需求（`T.alloc_ub` 直映 UB）；编译器管理 UB 分配与自动同步（§7）。
- 迁移 prompt 明确要求 developer 模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 环境变量 | `TILELANG_ASCEND_MODE=Developer` |
| 内存分配 | `T.alloc_ub(...)`（GM→UB 路径） |
| 计算方式 | v-prefix 向量 API（`T.vmax/T.vcmp/T.vselect/T.vcast/T.vbrc/T.clear` + `T.reduce_max`）+ `T.transpose` + `T.copy` |
| 同步 | 编译器自动同步（§7）；无手动 `T.set_flag/T.wait_flag` |
| Kernel 维度 | 一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(ih0,ih1) ← bin_H(o_h)`；`(iw0,iw1) ← bin_W(o_w)` | 编译期（`o_w`）/ 运行时（`o_h`）bin 边界；`bin_h=ih1-ih0`、`bin_w=iw1-iw0` |
| 2 | `row_src[c, iw] ← x[n, c0+c, ih0+kh, iw]` | 行片搬运 `(real_c, W_in)` |
| 3 | `row_dst[iw, c] ← row_src[c, iw]` | 核内二轴转置 → C 最内 |
| 4 | `row_f32[iw, c] ← cast_f32(row_dst[iw, c])` | 整 tile fp32 上转（`rint`） |
| 5 | `acc[o_w, c] ← max(acc[o_w, c], row_f32[iw0(o_w)+kw, c])` | C 轴向量取最大值（`kh` serial + `kw` 展开） |
| 6 | `cond[o_w, c] ← (row_f32[iw0+kw, c] != row_f32[iw0+kw, c])`；`h_acc[o_w,c] ← cond ? row_f32 : h_acc` | NaN 追踪（`v!=v` + 选择累积） |
| 7 | `cfinal[o_w,c] ← (h_acc != h_acc)`；`out32[o_w,c] ← cfinal ? h_acc : acc` | NaN 胜出判决（`has_nan ? NaN : max`） |
| 8 | `acc_t[c, o_w] ← cast_dtype(out32[o_w, c])` | 单次舍回 + 转置回转 |
| 9 | `out[n, c0+c, o_h, o_w] ← acc_t[c, o_w]` | 行片写回 `(real_c, out_W)` |
| C4 | `out[n,c,0,0] ← max_{ih,iw} x`（NaN 掩码归约判决） | 退化：单次 `reduce_max` + 掩码归约 |
| C4b | `out[n,c,ih,iw] ← x[n,c,ih,iw]` | 退化：恒等拷贝 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | 累加器置 `-inf` | `T.vbrc(src, dst)` | `src = -T.infinity("float32")`（标量），`dst = acc` | Developer |
| 0b | NaN 追踪器清零 | `T.clear(buf)` | `buf = h_acc` | Developer |
| 2 | 行片 GM→UB | `T.copy(src, dst)` | `src = x[n, c0:c0+real_c, ih0+kh, 0:W_in]`，`dst = row_src[0:real_c, 0:W_in]` | Developer |
| 3 | 二轴转置 | `T.transpose(src, dst, permutation)` | `permutation=[1,0]`（`docs/Tilelang.language/创建操作/T.transpose.md`） | Developer |
| 4 | fp32 上转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"` | Developer |
| 5 | C 轴取最大值 | `T.vmax(src0, src1, dst)` | 切片 `(1,CH)`：`src0=acc[o_w:o_w+1,:]`，`src1=row_f32[iw:iw+1,:]`，`dst=acc[o_w:o_w+1,:]`（原地） | Developer |
| 6 | NaN 探测 | `T.vcmp(A, B, C, cmp_mod)` | `A=B=row_f32[iw:iw+1,:]`，`C=cond_bool[0:1,:]`（bool 片上），`cmp_mod="ne"` | Developer |
| 6b | NaN 累积 | `T.vselect(Cond, A, B, Out)` | `Cond=cond_bool[0:1,:]`，`A=row_f32[iw:iw+1,:]`（遇 NaN 时本身即 NaN），`B=Out=h_acc[o_w:o_w+1,:]` | Developer |
| 7a | NaN 判决条件 | `T.vcmp` | `A=B=h_acc`，`C=cfinal_bool`，`cmp_mod="ne"` | Developer |
| 7b | `has_nan?NaN:max` | `T.vselect` | `Cond=cfinal_bool`，`A=h_acc`（NaN 源），`B=acc`，`Out=out32` | Developer |
| 8a | dtype 回转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"` | Developer |
| 8b | 输出转置回转 | `T.transpose(src, dst, [1,0])` | `(out_W,CH)→(CH,out_W)` | Developer |
| 9 | 行片 UB→GM | `T.copy(src, dst)` | `src = acc_t[0:real_c, 0:out_W]`，`dst = out[n, c0:c0+real_c, o_h, 0:out_W]` | Developer |
| C4 | 全平面取最大值 | `T.reduce_max(buffer, out, dim)` | `(CH,H,W)→(CH,H,1)`（dim=2）→`(CH,1,1)`（dim=1），`clear=True` | Developer |
| C4 | NaN 掩码归约 | `T.vcmp`+`T.vcast`+`T.reduce_max` | `mask_bool=vcmp(ne)`；`mask_f32=vcast(i1→f32,"rint")`；`reduce_max` → `nmax∈{0,1}` | Developer |
| C4 | NaN 判决 | `T.vcmp`+`T.vselect` | `cond=vcmp(nmax,zero,"gt")`；`out32=vselect(cond, nanbuf, gmax)`，`nanbuf=inf-inf` | Developer |
| 边界差 | 整数 bin 边界 | Python `adaptive_bin`（host/trace） | 复用源 `common.py` 逻辑 | host |

### 3.3 计算伪代码

> 注（迁移模板要求）：`T.Tensor` 的数据类型以**第二个位置参数**传入，不用 `dtype=` 关键字。

```python
# 迁移决策后的 NPU 算法伪代码（Developer 模式；C 轴向量化 + 向量 NaN 追踪 + 核内转置）
@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3", "-DENABLE_BF16"])
def _build(block_m=None):          # block_m 为源接口兼容参数（K9 保留），NPU tiling 由 shape 派生，值被忽略
    @T.prim_func
    def main(
        x:   T.Tensor((N, C, H_in, W_in), dtype),       # dtype 位置参数（非 dtype=）
        out: T.Tensor((N, C, out_H, out_W), dtype),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # ---- C4b 恒等（out_H==H_in and out_W==W_in，编译期分支；单元 = 整张 (n, c_block) 平面）----
            if IS_IDENTITY:
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:                  # NUM_LOGICAL = N * num_cblk
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb*CH)      # 通道尾块
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in], stage[0:real_c, 0:H_in, 0:W_in])
                        T.copy(stage[0:real_c, 0:H_in, 0:W_in], out[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in])
            # ---- C4 全局池化（out_H==1 and out_W==1；单元 = 整张 (n, c_block) 平面）----
            elif IS_GLOBAL:
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb*CH)
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in], flat[0:real_c, 0:H_in, 0:W_in])
                        T.vcast(flat, flat32, round_mode="rint")
                        T.reduce_max(flat32, mid, dim=2, clear=True)     # (CH,H,W)->(CH,H,1)
                        T.reduce_max(mid,    gmax, dim=1, clear=True)    # (CH,H,1)->(CH,1,1)
                        T.vcmp(flat32, flat32, mb, "ne")                 # NaN 掩码(bool, 片上)
                        T.vcast(mb, mf32, round_mode="rint")             # i1 -> f32 {0,1}
                        T.reduce_max(mf32, midn, dim=2, clear=True)
                        T.reduce_max(midn, nmax, dim=1, clear=True)      # 1.0 iff 任一 NaN
                        T.clear(zero)                                    # (CH,1,1) f32 = 0
                        T.vcmp(nmax, zero, cond, "gt")                   # has_nan
                        T.vbrc(T.infinity("float32"), nanbuf)
                        T.vsub(nanbuf, nanbuf, nanbuf)                   # inf - inf = NaN (IEEE, 运行时)
                        T.vselect(cond, nanbuf, gmax, out32)             # has_nan ? NaN : max
                        T.vcast(out32, gout, round_mode="rint")
                        T.copy(gout[0:real_c, 0:1, 0:1], out[nb, cb*CH:cb*CH+real_c, 0:1, 0:1])
            # ---- 一般网格（单元 = (n, o_h, c_block)；NUM_LOGICAL = N*out_H*num_cblk）----
            else:
                acc   = T.alloc_ub((out_W, CH), "float32")   # 最大值累加器（向量轴 = C）
                h_acc = T.alloc_ub((out_W, CH), "float32")   # NaN 追踪器
                cond  = T.alloc_ub((1, CH), "bool")          # NaN 掩码（片上）
                row_src = T.alloc_ub((CH, W_in), dtype)
                row_dst = T.alloc_ub((W_in, CH), dtype)
                row_f32 = T.alloc_ub((W_in, CH), "float32")
                cfinal  = T.alloc_ub((out_W, CH), "bool")
                out32   = T.alloc_ub((out_W, CH), "float32")
                acc16   = T.alloc_ub((out_W, CH), dtype)
                acc_t   = T.alloc_ub((CH, out_W), dtype)

                for t in T.serial(num_local):                # persistent（静态界）
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:
                        nb   = uid // (out_H * num_cblk)
                        r    = uid %  (out_H * num_cblk)
                        o_h  = r // num_cblk
                        cb   = r %  num_cblk
                        real_c = T.min(CH, C - cb*CH)
                        ih0, ih1 = adaptive_bin(o_h, H_in, out_H)   # 运行时标量
                        bh = ih1 - ih0
                        T.vbrc(-T.infinity("float32"), acc)         # acc <- -inf
                        T.clear(h_acc)                              # h_acc <- 0
                        for kh in T.serial(MAX_BIN_H):              # 静态界 + 守卫
                            if kh < bh:
                                T.copy(x[nb, cb*CH:cb*CH+real_c, ih0+kh, 0:W_in],
                                       row_src[0:real_c, 0:W_in])
                                T.transpose(row_src, row_dst, permutation=[1, 0])
                                T.vcast(row_dst, row_f32, round_mode="rint")
                                for o_w in range(out_W):            # 编译期展开
                                    iw0, iw1 = adaptive_bin(o_w, W_in, out_W)  # 编译期常数
                                    for kw in range(iw1 - iw0):     # 编译期展开
                                        T.vmax(acc[o_w:o_w+1, 0:CH],
                                               row_f32[iw0+kw:iw0+kw+1, 0:CH],
                                               acc[o_w:o_w+1, 0:CH])          # 向量 max
                                        T.vcmp(row_f32[iw0+kw:iw0+kw+1, 0:CH],
                                               row_f32[iw0+kw:iw0+kw+1, 0:CH],
                                               cond, "ne")                    # v!=v -> NaN
                                        T.vselect(cond,
                                                  row_f32[iw0+kw:iw0+kw+1, 0:CH],
                                                  h_acc[o_w:o_w+1, 0:CH],
                                                  h_acc[o_w:o_w+1, 0:CH])     # NaN 累积
                        T.vcmp(h_acc, h_acc, cfinal, "ne")          # has_nan
                        T.vselect(cfinal, h_acc, acc, out32)        # has_nan ? NaN : max
                        T.vcast(out32, acc16, round_mode="rint")
                        T.transpose(acc16, acc_t, permutation=[1, 0])
                        T.copy(acc_t[0:real_c, 0:out_W],
                               out[nb, cb*CH:cb*CH+real_c, o_h, 0:out_W])
    return main
```
（伪代码为**决策示意**，Stage 3 可细化局部次序与 buffer 复用；但「向量轴 = C」「核内二轴转置」「静态 bin 界 + 守卫」「`v!=v` 向量 NaN 追踪 + `has_nan?NaN:max`」「C4/C4b 特化」是冻结决策。）

### 3.4 API 可行性确认

| API | 来源确认 | 是否验证 |
|-----|---------|---------|
| `T.Kernel(num, is_npu=True) as (cid, _)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3（NPU ≤2 维） | ✓ 现有 examples 广泛使用 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` §2.2.2（1D~2D 充分验证；本设计用 1D/2D/3D，3D 见下注） | ✓（`lerp_tensor`/`logsumexp` developer kernel 同款） |
| `T.clear`（buffer 清零，1~5 维） | `docs/Tilelang.language/内存操作/T.clear.md` §2.2（fp32 ✓） | ✓ |
| `T.copy`（切片 GM↔UB） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2 | ✓ |
| `T.transpose`（二轴交换，fp16/bf16/fp32） | `docs/Tilelang.language/创建操作/T.transpose.md` §2.3 | ✓（`argreduce` perf_opt 用 `[1,0]`） |
| `T.vcast`（f16/bf16→f32 `rint`；f32→f16/bf16 `rint`；i1→f32 `rint`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✓ |
| `T.vmax`（fp16/fp32；`dst` 可原地） | `docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 | ✓ |
| `T.vcmp`（fp16/fp32/bf16；"ne" 模式；输出 bool 片上） | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1/§2.3 | ✓ |
| `T.vselect`（fp16/fp32/bf16；bool 条件） | `docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1 | ✓ |
| `T.vbrc`（scalar→任意 shape 广播） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 | ✓ |
| `T.infinity("float32")` | `docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（fp16/fp32 ✓） | ✓ |
| `T.vsub`（fp32；标量/张量；原地） | `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1/§2.3 | ✓ |
| `T.reduce_max`（fp32 ✓，`clear`） | `docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1/§2.1 | ✓ |

> **dtype 约束提醒**：`T.vmax`/`T.reduce_max`/`T.vsub` 的 dtype 表**不含 bf16**（分别见 `docs/Tilelang.language/比较操作/T.vmax.md`、`docs/Tilelang.language/规约操作/T.reduce_max.md`、`docs/Tilelang.language/数学操作/T.vsub.md` 各 §2.2.1），
> `T.vcmp`/`T.vselect`/`T.transpose`/`T.vcast` 支持 bf16。故 **bf16 输入必须经 `T.vcast` 上转到 fp32 后计算**——
> 这与「fp32 比较域」语义一致（§0.6 R2），无需额外代价。`nanbuf` 用 fp32（`T.vsub` 支持 fp32）。
> **`bool`（i1）不可 load/store GM**（`T.vcmp`/`T.vbrc` §2.3）：本设计的 `cond`/`cfinal`/`mb` 均为**片上中间结果**，不写回 GM ✓。
> **注（3D UB buffer）**：`T.alloc_ub` 文档称 3D~5D「有待充分验证」；C4/C4b 路径用 3D buffer 的形态与同族前作
> `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`（C4 `flat/flat32/mid/gacc` 均为 3D）一致，
> 该前作 Stage 3 已实测通过——故本条属**已有同族实证**；若 Stage 3 遇 3D UB 问题，回退为 2D `(CH, H*W)` + reshape（Stage 3 细节）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel（`T.Kernel` 一维/最多二维 block） | 否（用一维 `T.Kernel(num_kernels)`；任务维度在 kernel 内解码） | 一维网格；`(n,o_h,c_block)` 由 `uid = t*num_kernels + cid` 解码 |
| GPU 专用 API | 否（全部 v-prefix / `T.copy` / `T.transpose`，均为 NPU API） | — |
| GEMM 非整除 / L0C 溢出 | 否（无 Cube 计算，`L0C` 不适用） | — |
| 分核策略 | 见 §5.5（三要素齐全） | persistent grid-stride |
| host 侧禁止改输入内容 | 否（host 只做 `unsqueeze/squeeze/contiguous` 视图操作） | `ascend-constraints.md` §4 |
| 动态 `T.serial` 边界 | 涉及（`kh` 上界 `bin_h(o_h)` 运行时依赖） | 静态上界 `MAX_BIN_H` + `if kh < bh` 守卫（源同款） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 / 并行模型 | `T.Kernel(ceildiv(total,block_m), threads)` + `T.Parallel(block_m)`（线程 lane 级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1（`(n,o_h,c_block)` 单元 + C 向量 lane） |
| 归约单元 | 每线程标量比较链 | UB fp32 向量比较器（C 轴） | §0.6 R2 |
| NaN 追踪 | 标量 bool `has_nan \| isnan(val)` | C 向量掩码 `vcmp`+`vselect` | §0.6 R2（语义不变） |
| 数据搬运 | 逐元素 GM gather | UB tile 搬运 + 核内二轴转置 | §0.6 R3 |
| `threads` 参数 | factory `(block_m, threads)`、`default_config={"block_m":256,"threads":256}` | **移除 `threads`**（NPU 无 CUDA 线程）；callable 只留 `block_m` | 脚手架已按 K9 移除；factory 签名改为 `_build(block_m)` |
| 退化网格 | 无特化 | C4/C4b 结构特化 | §0.6 R4 / §1.6.1 |
| autotune | `autotune_configs` | 移除（K8），启发式 config | 脚手架已移除 |
| 编译标记 | `compile_flags=["-O3","-DENABLE_BF16"]` | 保留（设备无关） | — |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | **最高（同族前作，CASE-pool-adaptiveavgpool2d-migration）** | Developer 模式一维 `T.Kernel(num_kernels,is_npu=True)`；`CH=min(C,64)`；`(n,o_h,c_block)` 单元解码；`T.vbrc`/`T.clear`/`T.vcast`/`T.vadd`/`T.transpose` 用法；C4/C4b 特化（`NUM_LOGICAL=N*num_cblk`）；persistent 分核与 `real_c` 尾块 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 高（结构模板） | persistent grid-stride `T.serial(num_local)` + `if block_id<num_logical` 守卫；`T.alloc_ub`；尾块 gate |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | 高（归约模板） | `T.reduce_max` 用法、fp32 中转、尾部 `T.min` 切片 |
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/perf_opt/_argreduce_kernel.py` | 中（转置 + `T.vcmp/vselect` 模板） | 核内二轴转置 `T.transpose(src,dst,[1,0])`（PL-1.1 族）+ `T.vcmp`/`T.vselect`/`T.vcast` 组合实测用法（argmax 的索引选择） |
| `examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | 高（同族 NPU 已落地产物） | C 轴向量化 + 核内转置 + C4/C4b + `NUM_LOGICAL` 粒度的完整 NPU 实现（max 版差异仅在归约算子与 NaN 追踪） |
| `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/_adaptive_max_pool2d_fwd_kernels.py` | 高（本迁移的源函数抽取件） | GPU 源 kernel + `adaptive_bin`/`max_adaptive_bin_extent` helper（迁移时逐字对照） |
| `examples/TileOPs/tests/ops/test_pool.py` / `examples/TileOPs/tileops/manifest/pool.yaml` | 高（规格/golden） | workload 与 `ref_program = F.adaptive_max_pool2d`，atol=rtol=0 |
| pattern-library `layout.md` PL-1.1/PL-1.2/PL-1.3/PL-1.4 | 高（池化类实测模式） | 核内融合转置链、C 轴切片累加、host permute 证伪、tiling 启发式 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N, C, H_in, W_in)` | float16 \| bfloat16 | NCHW 连续；3D CHW 由 op 层 `unsqueeze(0)` 适配 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(N, C, out_H, out_W)` | same_as_input | NCHW 连续；`out_idx=[1]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `row_src` | `(CH, W_in)` | dtype | UB | 行片 GM→UB 暂存 |
| `row_dst` | `(W_in, CH)` | dtype | UB | 核内二轴转置输出（C 最内） |
| `row_f32` | `(W_in, CH)` | float32 | UB | fp32 上转比较源 |
| `acc` | `(out_W, CH)` | float32 | UB | C 轴向量最大值累加器（向量轴 = C，初值 `-inf`） |
| `h_acc` | `(out_W, CH)` | float32 | UB | C 轴向量 NaN 追踪器（初值 0，遇 NaN 置 NaN） |
| `cond` | `(1, CH)` | bool | UB | 逐元素 NaN 掩码（**片上**，不落 GM） |
| `cfinal` | `(out_W, CH)` | bool | UB | 窗口级 has_nan 判决条件（片上） |
| `out32` | `(out_W, CH)` | float32 | UB | NaN 判决结果（fp32） |
| `acc16` / `acc_t` | `(out_W, CH)` / `(CH, out_W)` | dtype | UB | 舍回 + 转置回转 |
| `flat` / `flat32`（C4） | `(CH, H_in, W_in)` | dtype / float32 | UB | 全局池化退化路径 |
| `mid` / `gmax`（C4） | `(CH, H_in, 1)` / `(CH,1,1)` | float32 | UB | 全平面归约中间/输出 |
| `mb` / `mf32`（C4） | `(CH,H_in,W_in)` | bool / float32 | UB | NaN 掩码（bool 片上）与 fp32 掩码 |
| `midn` / `nmax` / `zero` / `nanbuf` / `out32` / `gout`（C4） | `(CH,H_in,1)` / `(CH,1,1)` 等 | float32 / dtype | UB | NaN 掩码归约、判决、NaN 常量、输出 |
| `stage`（C4b） | `(CH, H_in, W_in)` | dtype | UB | 恒等退化路径中转（单元 = 整张 `(n, c_block)` 平面，一次整平面 `T.copy`） |

### 4.4 内存搬运路径

```
【一般网格】
GM[x] --T.copy 行片(real_c,W_in)--> UB[row_src] --T.transpose[1,0]--> UB[row_dst] --T.vcast(rint)--> UB[row_f32]
UB[row_f32] --T.vmax(C 轴切片, kh serial × o_w/kw 展开)--> UB[acc (out_W,CH) f32]  ∥  UB[h_acc] --vcmp+vselect(NaN 追踪)
UB[acc]/UB[h_acc] --vcmp(h_acc,h_acc,"ne") + vselect --> UB[out32] --T.vcast(rint)--> UB[acc16] --transpose[1,0]--> UB[acc_t] --T.copy 行片--> GM[out]
【C4 全局池化】GM[x] --T.copy 连续块(CH,H,W)--> UB[flat] --vcast--> UB[flat32] --reduce_max(dim=2→1)--> UB[gmax]
               UB[flat32] --vcmp(ne)+vcast(i1→f32)+reduce_max--> UB[nmax] --vcmp(gt vs 0)--> cond --vselect(nanbuf,gmax)--> UB[out32] --vcast--> GM[out]
【C4b 恒等】GM[x] --T.copy 整平面(CH,H,W)--> UB[stage] --T.copy--> GM[out]（单元 = 整张 (n,c_block) 平面，NUM_LOGICAL=N*num_cblk）
（无 GM 中间往返、无 Cube/L1/L0C 路径）
```

### 4.5 UB 内存预算

| Buffer | Shape（代表 workload） | dtype | 大小 (Bytes) |
|--------|------------------------|-------|-------------|
| `row_src` / `row_dst` | `(64, 56)`（spp CH=64） | fp16 | 7 168 × 2 |
| `row_f32` | `(56, 64)` | fp32 | 14 336 |
| `acc` / `h_acc` / `out32` | `(6, 64)` 各 | fp32 | 1 536 × 3 |
| `cond` / `cfinal` | `(1,64)` / `(6,64)` | bool | 64 / 384 |
| `acc16` / `acc_t` | `(6,64)` / `(64,6)` | fp16 | 768 × 2 |
| **总计（spp CH=64）** | | | **≈ 34.9 KB** / 196 608 (192 KB, `CONST-capacity-910B2C` A2/A3) ✓ |
| C4（resnet CH=64）：`flat`+`flat32`+`mid`+`gmax`+`mb`+`mf32`+`midn`+`nmax`+`zero`+`nanbuf`+`out32`+`gout` | `(64,7,7)` / `(64,7,1)` / `(64,1,1)` 等 | — | ≈ 39.5 KB ✓ |
| C4b `stage`（identity CH≤64） | `(64, 49)` fp16（或 NCHW 下 `(64, H_in, W_in)`） | fp16 | ≤ 12.7 KB ✓ |

> **UB 膨胀口径两档**（`CONST-capacity-910B2C`，entry 见 `constants.md`）：
> - **默认档（auto-multi-buffer 开，本设计默认，§7.3）**：膨胀 **~1.7x** → spp ≈ 59 KB；resnet C4 ≈ 67 KB；C4b ≈ 22 KB。**均 ≪ 192 KB**。
> - **关闭档（`--enable-auto-multi-buffer=false`）**：膨胀 1.10–1.12 → spp ≈ 39 KB；resnet C4 ≈ 44 KB。同样充裕。
> **两档均在容量内**，故 §7.3「溢出时再评估关 multi-buffer」以本两档为可比基线。

### 4.6 动态轴定义

**无动态轴**：`N, C, H_in, W_in, out_H, out_W` 全部在 factory 构建期（trace 期）确定，kernel 由 `functools.lru_cache` 按 shape 特化。
（`o_h`/`bin_h` 在 kernel 内是运行时标量，但它们是**由编译期 shape 推出的运行时值**，不是动态轴。）

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3", "-DENABLE_BF16"])
def _adaptive_max_pool2d_func(CH: int) -> ...      # 去掉源 GPU 的 threads 参数（K9）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。

**判定依据**：算子仅含逐元素向量运算（`T.vmax`/`T.vcmp`/`T.vselect`）、归约（`T.reduce_max`）、cast（`T.vcast`）与布局重排（`T.transpose`），
无 `T.gemm` / matmul → 不需要 L1 + L0A/L0B/L0C（§1.6.0 R1/R4 已弃 Cube 路径——数学上不可表达）。
故核数按**纯 Vector 算子翻倍**规则取物理核数 ×2（§5.5）。

### 5.2 Block 划分

本算子的分块维度不是传统 GEMM 的 `(M,N)`，而是 **(通道块 CH) × (输出行 o_h)** 的任务维度：

```python
CH = min(C, 64)                    # 通道向量块（8 的整数倍）；主选值，D-3 备选见 §1.6.3
num_cblk     = ceil(C / CH)        # 通道块数（C=8/16/64/96/128/2048 → 1/1/1/2/2/32）
# 逻辑单元数按编译期分支取（**单元粒度是 num_logical 的唯一定义源**，见 §5.5 ①）：
#   一般网格  单元 = (n, o_h, c_block)  ->  NUM_LOGICAL = N * out_H * num_cblk
#   C4 全局   单元 = (n, c_block)        ->  NUM_LOGICAL = N * num_cblk（out_H==1，与通用式同值）
#   C4b 恒等  单元 = (n, c_block)        ->  NUM_LOGICAL = N * num_cblk（整平面；≠ N*out_H*num_cblk）
NUM_LOGICAL = N * num_cblk if (out_H == 1 and out_W == 1) or (out_H == H_in and out_W == W_in) \
              else N * out_H * num_cblk
```

- **为什么 `num_logical` 按分支取**：`num_logical` 必须与「一个逻辑单元处理多少数据」**严格同粒度**，否则任务解码越界。
  一般网格单元为 `(n, o_h, c_block)`，退化路径（C4/C4b）单元为 `(n, c_block)` **整张平面**，所需核数不同；
  C4 因 `out_H==1` 两种口径同值，**唯一会分叉的是 C4b**：`(1,8,9,11)` 恒等 case 通用式给 9，而整平面单元只需 1 ——
  若沿用通用式 9 配 §3.3 解码 `n = uid // num_cblk` 将产生 `n=0..8`（`N=1`）**越界读/写**。本版采用整平面单元口径 `NUM_LOGICAL = N*num_cblk`，
  与 §3.3 解码、§4.3 `stage=(CH, H_in, W_in)`、§5.5 ① / §6.2 全文唯一自洽。
- **CH = 64 的选择理由**：① C 轴为向量轴，CH 需为 8（fp16 向量宽）的整数倍——64 为 8 的整数倍；
  ② UB 预算充裕（spp ≈35 KB，§4.5）；③ 与 §5.5 分核方案配合使 `num_logical` 落在物理核数附近。
- **`MAX_BIN_H = max_adaptive_bin_extent(H_in, out_H)`、`MAX_BIN_W`**：bin 循环的静态上界（编译期常数）；`bin_h/bin_w` 为运行时边界差。

**代表性 shape 的分块结果**：

| workload | N | C | H_in×W_in | out | CH | num_cblk | NUM_LOGICAL（按单元粒度） |
|----------|---|---|-----------|-----|----|----------|-------------|
| smoke | 1 | 8 | 4×4 | 2×2 | 8 | 1 | 2（一般） |
| global-1x1 | 8 | 2048 | 7×7 | 1×1 | 64 | 32 | **256**（C4 特化，`N*num_cblk` = 通用式同值） |
| spp-6x6 | 2 | 128 | 56×56 | 6×6 | 64 | 2 | 24（一般） |
| nondiv-7x7 | 2 | 64 | 55×57 | 7×7 | 64 | 1 | 14（一般） |
| test `(1,96,28,30)`→7×5 | 1 | 96 | 28×30 | 7×5 | 64 | 2 | 14（一般） |
| test `(1,128,55,57)`→7×7 | 1 | 128 | 55×57 | 7×7 | 64 | 2 | 14（一般） |
| test `(1,16,11,13)`→`(5,None)` | 1 | 16 | 11×13 | 5×13 | 16 | 1 | 5（一般；H 非恒等，**非** C4b） |
| 3D `(16,11,13)`→(5,4)（op 层 unsqueeze 后 `(1,16,11,13)`） | 1 | 16 | 11×13 | 5×4 | 16 | 1 | 5（一般） |

### 5.3 约束分析

- **对齐约束**：向量轴 C 恒为 8 的整数倍（fp16/bf16 向量宽 8）；`CH=64` 为 8 倍数 ✓。`W_in`（55/56/57）仅作转置后的 row 长度，不作向量轴，不要求整除 ✓。
- **UB 容量**：spp CH=64 总 ≈35 KB（默认 auto-multi-buffer ×1.7 → ≈59 KB；关闭档 ×1.12 → 39 KB）< 192 KB ✓（§4.5）。resnet C4 CH=64 → ≈39.5 KB（×1.7 ≈ 67 KB）✓。
- **L0 容量**：无 Cube 计算，不适用 ✓。
- **非整除**：本算子**不存在 GEMM 式整除问题**——`total_output`、`W_in` 无需是 block 的整数倍：
  任务维度用 persistent `T.serial` + `if uid < NUM_LOGICAL` 守卫（尾块丢弃），row 长度用静态 `W_in`；
  通道尾块用 `real_c = min(CH, C - cb*CH)` 动态 copy 范围，padded lane 永不写回 GM。
  bin 非整除（重叠）由整数 `adaptive_bin` 精确处理（语义本来就要求）。

### 5.4 注意事项

- **尾单元**：`NUM_LOGICAL` 非 `num_kernels` 整数倍时（global-1x1: 256 / 40 = 6.4），核内 `T.serial(num_local)` 的 `if uid < NUM_LOGICAL` 守卫丢弃越界任务。
  **退化路径（C4/C4b）的 `NUM_LOGICAL` 按其整平面单元粒度取**（`N*num_cblk`）。
- **通道尾块**：`C % CH != 0`（如 C=96 → CH=64 + `real_c=32`）——copy/写回用 `real_c` 动态范围；`row_src/row_dst/row_f32` 的 padded lane 残留旧值，
  但 padded lane 的 `acc/h_acc` 永不写回 GM（各 lane 独立），无污染。
- **`kh` 越界守卫**：`MAX_BIN_H` 是 bin 的**最宽**上界；对较窄的 `o_h`，`if kh < bh` 跳过越界行（不读取 `>= ih1` 的输入行）——与源守卫同构（§0.4 #1）。
- **零输出保护**：`out_H/out_W > 0` 由 op 层保证（`_normalize_output_size` 拒绝 `<=0`）。
- **NaN 语义**：必须在所有输入（含 `±inf`/subnormal）下保持「NaN 胜出 + ±inf 取 +inf」——由 `v!=v` 探测 + `h_acc` 累积保证（§0.6 R2）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**（**单元粒度 = `num_logical` 的唯一定义源**）：
  - **一般网格**（非恒等、非全局）：单元 = `(n, o_h, c_block)` → `num_logical = N × out_H × ceil(C / CH)`（pure-Vector 无 `block_N` 维度，故只有通道块维 × 输出行维）。
  - **退化路径 C4 全局**（`out_H==out_W==1`）：单元 = 整张 `(n, c_block)` 平面 → `num_logical = N × ceil(C/CH)`。因 `out_H==1`，该值与通用式**同值**（global-1x1 `8 × 1 × 32 = 256`），无分叉。
  - **退化路径 C4b 恒等**（`out_H==H_in and out_W==W_in`）：单元 = 整张 `(n, c_block)` 平面 → `num_logical = N × ceil(C/CH)`，**显式不同于**通用式 `N × out_H × ceil(C/CH)`。
    依据：若沿用通用式而单元仍为整平面，§3.3 解码 `n = uid // num_cblk` 在 `N=1` 下会产生越界（见 §5.2）。
  代表值见 §5.2 表：smoke 2 / global-1x1 256（C4）/ spp 24 / nondiv 14。

- **② 物理核数依据（实查）**：
  本设计环境实际执行查询：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()
  # 实查返回值 = 20   (设备 Ascend910B3，查得于 2026-10-13，本设计会话)
  ```
  **纯 Vector 算子核数翻倍**：`vector_cores = 20 × 2 = 40`。
  （设备名核对：`npu-smi info` 显示 `910B3`；**不采用** pattern-library `CONST-aicore-910B2C`=24 的 910B2C 值——本设备为 910B3，必须实查。
  实查须在项目 venv（`/workspace/venv/bin/python`）下执行，系统 `python3` 缺 `torch` 会误报「环境无运行时」。）

- **③ 规模判定与分核方案**（三选一，逐 workload 判定）：

  | workload | NUM_LOGICAL | 物理核 | 判定 | 方案 |
  |----------|-------------|--------|------|------|
  | smoke | 2 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = min(2,40) = 2`，2 个单元各占 1 核 |
  | spp / nondiv / 多数测试 | 5–24 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = NUM_LOGICAL`，每核 1 个单元 |
  | `(1,8,9,11)` 恒等（**C4b**） | 1 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = 1`，1 核执行 1 个整平面单元（`N*num_cblk=1`）；**消除**了「通用式 9 + 整平面单元 → 越界」 |
  | global-1x1 | 256 | 40 | **极大规模**（无法通过调小 block 缩减核数到 ≤40，且核数非 40 的整数倍） | **核内串行（persistent）**：固定启动 `num_kernels = min(NUM_LOGICAL, vector_cores) = 40`；核内 `for t in T.serial(num_local)` 串行，`num_local = ceil(256/40) = 7`（**静态值**）；`uid = t*num_kernels + cid`，`if uid < NUM_LOGICAL` 守卫 |

  **通用规则（factory 级）**：`NUM_LOGICAL` 按分支粒度取（见 ①），`num_kernels = min(NUM_LOGICAL, vector_cores)`，
  `num_local = ceil(NUM_LOGICAL / num_kernels)`（**Python 整数编译期常量**，非 `T.ceildiv` 表达式）。
  当 `NUM_LOGICAL ≤ vector_cores` → `num_local = 1`，退化为「每核 1 单元 = 无需适配」；当 `NUM_LOGICAL > vector_cores` → 核内串行（persistent）。
  **核内串行循环边界 `num_local` 为静态值**（`NUM_LOGICAL`/`num_kernels` 均为编译期常数）——满足 `core-split-strategy.md` §1 要素③硬要求。
  > **不做空核的保证**：因 `num_kernels = min(NUM_LOGICAL, vector_cores)`，故 `num_kernels ≤ NUM_LOGICAL`，每个物理核的首个任务 `uid = cid < NUM_LOGICAL` 恒有效。
  > 分核增益预期：global-1x1 若按 GPU 式超发 256 核，NPU 运行时会将 216 核串行排入 40 核 → 引入核启动开销；persistent 化后固定 40 核、每核 7 个静态任务（256 = 40×6 + 16，负载差 1 个任务）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block/tile 级调度循环。**不出现逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑单元（task） | block 级并行 | `T.Kernel(num_kernels)` | 每核处理 1 个或多个 `(n,o_h,c_block)` 单元 |
| 累加器初始化 | **整块填充/清除**（无循环） | `T.vbrc(-inf, acc)` + `T.clear(h_acc)` | 单元入口整 buffer 初始化（§1.6.2 #0） |
| 核内多任务 | 迭代（persistent） | `T.serial(num_local)`（静态界）+ `if uid<NUM_LOGICAL` | 极大规模时摊薄核启动开销（§5.5）；静态边界 |
| bin 的 H 方向 | 迭代（归约链） | `T.serial(MAX_BIN_H)` + `if kh < bh` | tile 级顺序依赖 + 动态边界（§1.6.2 #9）；循环体内是向量 op |
| bin 的 W 方向 | **编译期展开** | Python `range(bin_w(o_w))` | `o_w`/`bin_w` 为 trace 期常数；展开为**向量 op 序列**（§1.6.2 #10） |
| 输出列 `o_w` | **编译期展开** | Python `range(out_W)` | `out_W` 静态且小（≤7）→ 展开避免运行时索引 |
| 元素级（C 轴） | 向量化 | `T.vmax` + `T.vcmp` + `T.vselect`（向量轴 = C，§1.6.3 决策） | C 轴 lane 并行，无逐元素循环 |
| 退化路径 | 单次归约 | `T.reduce_max`（C4） | 全局池化无 bin 循环 |

### 6.2 循环伪代码

```python
# num_kernels / num_local 由 NUM_LOGICAL 导出；NUM_LOGICAL 按编译期分支的单元粒度取（§3.3/§5.5①）：
#   一般 = N*out_H*num_cblk；C4 全局 = N*num_cblk；C4b 恒等 = N*num_cblk（整平面）
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 一维网格（≤2 维合法）
    for t in T.serial(num_local):                            # persistent（静态界）
        uid = t * num_kernels + cid
        if uid < NUM_LOGICAL:                                # 尾任务守卫（与 NUM_LOGICAL 同粒度）
            # 解码为 (n, o_h, c_block)【一般】/ (n, c_block)【C4/C4b】后进入 §3.3 的单元体
            ...
            for kh in T.serial(MAX_BIN_H):                   # 静态界
                if kh < bh:                                  # bin 边界守卫
                    [T.copy → row_src; T.transpose; T.vcast]
                    for o_w in range(out_W):                 # 编译期展开
                        for kw in range(bin_w(o_w)):         # 编译期展开
                            T.vmax(...)   # C 轴向量最大值
                            T.vcmp(..., "ne"); T.vselect(...)   # C 轴向量 NaN 追踪
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- 单元内只有一条 `GM→UB→计算→UB→GM` 单阶段流，无多阶段 K 循环可流水；`kh` 循环次数少（`MAX_BIN_H` 1–10），双缓冲收益有限。
- Developer 模式下编译器可自动 multi-buffer；若 Stage 4 实测显示 MTE2 未与 Vector 重叠，可在 `kh` 循环上探索 `T.Pipelined(num_stages=2)`（列入 Stage 4 候选，不改主结构）。
- C4/C4b 退化路径为「1 次搬运 + 1 次归约 / 纯拷贝」，无流水空间。

### 6.4 尾块处理

- **任务尾块**：`NUM_LOGICAL` 非 `num_kernels` 整数倍 → `if uid < NUM_LOGICAL` 丢弃越界任务。退化路径 `NUM_LOGICAL` 按整平面单元粒度取（§5.5 ①）。
- **通道尾块**：`real_c = min(CH, C - cb*CH)` 动态 copy/写回范围；padded lane 永不落 GM。
- **bin 尾块（H 方向）**：`kh >= bh` → 守卫跳过，不读入界外行。
- **bin 尾块（W 方向）**：`kw` 用编译期 `range(bin_w(o_w))`，**精确**覆盖 bin，无越界、无掩码需求。
- **无 GEMM 式整除要求**（§5.3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步（Developer 模式）**。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器在 `T.copy`（MTE2 搬运）与 v-prefix 向量 op（Vector 单元）之间自动插入同步标志，
`T.transpose`（VTransposeOp）与 `T.reduce_max`（VReduceOp）亦由编译器编排。本算子无跨核通信、无 Cube/Vector 跨引擎依赖：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| （无） | — | Developer 自动同步；本算子单引擎（Vector）、单阶段，无 `sync_block_set/wait` 需求 |

### 7.3 pass_configs 配置

`@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3","-DENABLE_BF16"])`；不设置自定义 `pass_configs`（与 `lerp_tensor` / `logsumexp` / 同族 avg 先例一致）。
若 Stage 3/4 出现 UB 膨胀导致溢出，可评估 `--enable-auto-multi-buffer=false`（`CONST-capacity-910B2C` 2026-09-16 反例数据点）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现/测试基准），**不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F


def golden_adaptive_max_pool2d(x: torch.Tensor, output_size):
    """PyTorch reference for adaptive_max_pool2d, per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark
    (see ``ref_program`` in examples/TileOPs/tests/ops/test_pool.py).  Accepts 3D CHW
    or 4D NCHW; ``None`` entries keep the input size; single output (no indices).
    Independent of the NPU algorithm (no transpose / no C-vectorisation / no
    C4-C4b specialisation).  Runs on CPU and does not require an NPU device.
    """
    size = (None, None) if output_size is None else output_size
    return F.adaptive_max_pool2d(x.detach().to(torch.float32), size).to(x.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 0 | 0 |
| bfloat16 | 0 | 0 |

> 依据：max 是**选择**而非累加，输出逐位等于窗口内某输入元素（§0.1），故精度用例 `atol=rtol=0`（`test_pool.py:130` "Max is exact"）；
> 与本设计的 fp32 比较域 + 单次 `rint` 舍回一致（fp16/bf16→fp32→dtype 无损）。**无 fp32 用例**（op 层 dtype gate 拒绝 fp32）。

### 8.3 L0 门槛测试计划（必须通过）

| # | 用例 | 目的 |
|---|------|------|
| L0-1 | `(1,8,4,4)` out `(2,2)` fp16 | manifest smoke（一般网格） |
| L0-2 | `(2,128,56,56)` out `(6,6)` fp16 | manifest spp（重叠 bin + 通道块 2） |
| L0-3 | `(8,2048,7,7)` out `(1,1)` fp16 | manifest global（**C4 特化 + persistent 256 单元**） |
| L0-4 | `(2,64,55,57)` out `(7,7)` bf16 | manifest nondiv（bf16 上转路径 + 非整除重叠） |
| L0-5 | `(1,8,9,11)` out None fp16 | **C4b 恒等**（`NUM_LOGICAL=1` 收敛路径） |
| L0-6 | `(1,16,11,13)` out `(5,None)` fp16 | 部分 None（H 非恒等 → 一般网格） |
| L0-7 | `(1,96,28,30)` out `(7,5)` bf16 | 通道尾块（C=96 → CH=64 + `real_c=32`） |
| L0-8 | 3D `(16,11,13)` out `(5,4)` fp16（op 层 unsqueeze） | CHW rank 保持 |
| L0-9 | NaN/±inf 值注入（`(1,16,11,13)`→`(5,4)`） | **NaN 胜出 + ±inf 取 +inf 语义**（`equal_nan=True`） |
| L0-10 | `block_m` guard（传 0/负） | factory 接口兼容性（K9 保留但校验） |

> 完整分层套件（L1 功能全覆 / L2 极端尺寸 / Boundary 特殊值）交由 `tilelang-op-develop`，不在此枚举。
> 精度对照：`torch.testing.assert_close(out, ref, atol=0, rtol=0, equal_nan=True)`（L0-9 需 `equal_nan=True`）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **NaN 语义必须由 `v != v` 探测**：不得依赖 `T.vmax` / `T.reduce_max` 的 NaN 行为（**未文档化**）；本设计用 `T.vcmp("ne")` + `T.vselect` 显式追踪。若 Stage 3 误用 vmax 的 NaN 传播实现，`{+inf,-inf}` 同窗或 all-NaN 窗可能出错（实测 PyTorch 语义：`{+inf,-inf}` → +inf，非 NaN）。
- **bf16 计算路径**：`T.vmax`/`T.reduce_max`/`T.vsub` 不支持 bf16 → bf16 必须先 `vcast` 到 fp32（与 fp32 比较域一致，无需额外代价）。
- **bool 不可落 GM**：`cond`/`cfinal`/`mb` 必须留片上（`T.vcmp` §2.3）。
- **3D UB buffer**：`T.alloc_ub` 文档称 3D~5D「有待充分验证」；C4/C4b 用 3D 形态与同族 avg 前作一致（已有实证）。若 Stage 3 遇阻，回退 2D `(CH, H*W)` + reshape。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|------|
| `acc` 未置 `-inf` | 用 `T.clear` 置 0 后直接 `vmax` | **全负窗口输出 0（错误）** | 必须 `T.vbrc(-T.infinity("float32"), acc)`（max 的单位元是 `-inf`，非 0） |
| NaN 追踪缺失/用求和探测 | 试图用 `Σx != Σx` 判 NaN | `{+inf,-inf}` 同窗误判 NaN | 用 `v != v`（`T.vcmp("ne")`） |
| `NUM_LOGICAL` 与单元粒度不一致（C4b） | 恒等路径沿用通用式 `N*out_H*num_cblk` | `n = uid//num_cblk` 越界读写 | C4b/C4 用 `N*num_cblk`（§5.5 ①） |
| `kh` 动态边界 | 直接 `T.serial(bin_h(o_h))` | TileLang 拒绝动态边界 | 静态 `MAX_BIN_H` + `if kh < bh` 守卫 |
| UB 溢出 | block/CH 过大 | 编译失败 | 减小 CH；或关 auto-multi-buffer（§7.3） |
| 通道尾块越界 | `C % CH != 0` 时按 CH copy | 越界读/写 | `real_c = min(CH, C - cb*CH)` |
| vcmp/vselect 形状不一致 | cond 用 `(CH,)` 而数据 `(1,CH)` | 编译/校验失败 | cond 与数据切片同形 `(1,CH)` |

### 9.3 特殊场景处理

- **非整除分块**：通道尾块 `real_c` 动态；任务尾块 `if uid < NUM_LOGICAL`；bin 重叠由整数 `adaptive_bin` 精确处理。
- **极小 shape**：`(1,1,1,1)`（global == identity，identity 优先 → 纯拷贝）；`C<8`（CH=max(8, round_up) 与 `real_c` 处理）。
- **扩大输出**（`out > in`，如 `(1,8,4,4)`→`(7,7)`）：bin 宽度 1–2，`MAX_BIN_H/W` 静态容纳；语义 = 元素复制/邻居最大值。
- **混合精度**：bf16 经 fp32 中转；输出单次 `rint` 舍回。
- **NaN/±inf**：所有路径（一般网格 / C4 / C4b）均保持语义；C4b 恒等路径天然保留 NaN（纯拷贝）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/
├── DESIGN.md                # 本设计文档
├── verify_equiv.py          # §1.6.1 采纳优化项的等价性机器验证（D-1）
├── RETROSPECTIVE.md         # Stage 1 复盘（Phase 8）
└── _adaptive_max_pool2d_kernel.py   # 算子实现（Stage 3 产出）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档 |
| `verify_equiv.py` | ✅ 已完成（132 用例全 PASS） | §1.6.1 等价性机器验证 |
| `_adaptive_max_pool2d_kernel.py` | ⬜ 待实现（Stage 3） | NPU kernel + golden + 分层测试 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘 |

### 10.3 命名规范

- 项目目录名：`adaptive_max_pool2d`（snake_case）
- 算子目录名：`_adaptive_max_pool2d_kernel`（snake_case，与迁移函数名一致）
- 实现文件：`_adaptive_max_pool2d_kernel.py`
- 输出契约与参数名保持 `_adaptive_max_pool2d_kernel(n, c_in, h_in, w_in, out_h, out_w, dtype)` 不变（`threads` 已按 K9 移除）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ 算子实现（`_adaptive_max_pool2d_kernel.py`）+ L0 精度比对
4. ⬜ 接入 NPU wrapper（`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/`），跑 `test_pool.py` / `bench_pool.py`

### 10.5 与 GPU 源接口的对照（迁移契约）

| 项 | GPU 源 | NPU 目标 |
|----|--------|---------|
| factory 签名 | `_adaptive_max_pool2d_kernel(n,c_in,h_in,w_in,out_h,out_w,dtype="float16")` | **不变** |
| 返回 callable | `_build(block_m, threads)` | `_build(block_m)`（K9：`threads` 移除，`block_m` 保留兼容/被忽略） |
| `@tilelang.jit` | `out_idx=[1], compile_flags=[...]` | `out_idx=[1], target="npuir", compile_flags=[...]` |
| 输出 | `out`（`out_idx=[1]`） | 不变 |
