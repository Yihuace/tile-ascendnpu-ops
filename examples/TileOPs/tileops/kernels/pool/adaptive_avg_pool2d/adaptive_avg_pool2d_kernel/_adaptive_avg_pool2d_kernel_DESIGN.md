# AdaptiveAvgPool2dFwdOp（`_adaptive_avg_pool2d_kernel`）算子设计文档

> **迁移任务（TileOPs 同构仓迁移，harness 子模式）**：源算子 = `AdaptiveAvgPool2dFwdOp`
> （family=`pool`），本次重实现的 TileLang kernel 函数 = `_adaptive_avg_pool2d_kernel`。
> 本文按「源算子三问解读（§0）→ 算法调研（§1.6.0）→ 硬件耦合性判定与 NPU 重设计（§0.5/§0.6）
> → 通用设计流程（§1–§10）」组织。§1–§10 一律以 §0.6 重设计后的 NPU 算法为输入，不照抄源方案。
>
> - project = `adaptive_avg_pool2d`，op = `_adaptive_avg_pool2d_kernel`
> - 编程模式：**Developer**（迁移 prompt 指定）
> - 源算子路径：`/workspace/TileOPs-gpu/tileops/ops/pool.py`
> - 源 kernel：`/workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_avg_pool2d.py`；
>   源 helper：`/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 方法论：`tilelang-op-design` skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)。
> 逐行读源文件（`pool.py` §`_AdaptivePool2dFwdOpBase`/`AdaptiveAvgPool2dFwdOp`/`_adaptive_pool2d_roofline`/`_normalize_output_size`、
> `kernels/pool/adaptive_avg_pool2d.py` 全文、`kernels/pool/common.py` 全文），未臆测语义。

### 0.1 源算子语义（做什么）

**数学语义**（`torch.nn.functional.adaptive_avg_pool2d` 语义）：

对输出空间位置 $(o_h,o_w)$，输入空间轴被划分为半开区间（PyTorch adaptive-bin 划分，源实现由 `common.py:adaptive_bin` 给出）：

$$
\text{bin}_H(o_h)=\big[\lfloor o_h H_{in}/H_{out}\rfloor,\ \lceil (o_h+1)H_{in}/H_{out}\rceil\big),\qquad
\text{bin}_W(o_w)=\big[\lfloor o_w W_{in}/W_{out}\rfloor,\ \lceil (o_w+1)W_{in}/W_{out}\rceil\big)
$$

$$
\text{out}[n,c,o_h,o_w]=\frac{1}{|\text{bin}_H(o_h)|\cdot|\text{bin}_W(o_w)|}\sum_{i_h\in\text{bin}_H(o_h)}\ \sum_{i_w\in\text{bin}_W(o_w)} x[n,c,i_h,i_w]
$$

- **输出尺寸规则**（`pool.py:_normalize_output_size` + `_resolve_out_dims`）：`output_size` 取 `None | int | tuple/list[int|None, int|None]`；
  `None`（含 tuple 中的 `None`）→ 取对应输入轴尺寸。`out_H = H_in if oH is None else oH`，`out_W` 同理。
  `output_size` 为标量 int 时按 `(v,v)` 展开；`bool` 被拒绝；`<=0` 被拒绝。
- **规约语义**：规约轴 = H×W 两个空间轴（NCHW 的第 2、3 轴），逐 `(n,c,o_h,o_w)` 独立求均值，**无跨通道/跨 batch 归约**。
  累加顺序在源实现中是「`kh` 外层、`kw` 内层」的逐元素串行累加（见 §0.3）。
- **dtype 语义**：输入仅 **float16 / bfloat16**（`pool.py:_validate_adaptive_pool_input_dtypes` 显式拒绝 fp32）；
  源 kernel 内 `accum_dtype = "float"`（`kernels/pool/adaptive_avg_pool2d.py:22`）——**中间累加在 fp32**，
  每个输入元素先 `T.cast(x[...], "float")` 再累加，最后 `T.cast(sum / bin_size, dtype)` 单次舍入回输出 dtype。输出 dtype = 输入 dtype。
- **边界语义**：
  - bin 永不为空（源码注释明确：即使 `size_out > size_in` 时 bin 也为 2 个元素，`common.py:100-108`）。
  - 相邻 bin 在非整除时**重叠一行/列**（`_adaptive_pool2d_roofline` 注释：`adjacent bins overlap by one row/col unless out | j*in`）。
  - `bin_size` 用**整数 bin 边界差** `(ih_end-ih_start)*(iw_end-iw_start)` 计算，不是比例近似。
  - nan/inf 按 IEEE 传播（累加 + 除法），无特殊分支。
  - 3D CHW 输入：`_eager_forward` 先 `unsqueeze(0)` 成 4D，计算后 `squeeze(0)` 回 3D（恒等 rank 保持）。
  - 无空维度处理（`out > 0` 由 op 层保证）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（优先移植源仓测试基准 `F.adaptive_avg_pool2d`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（input） | `(N, C, H_in, W_in)`（op 层同时接受 3D `(C,H_in,W_in)`） | float16 \| bfloat16 | NCHW 连续布局，尾轴 `W_in` 连续 |
| `out`（output） | `(N, C, out_H, out_W)`；3D 输入时输出 3D `(C, out_H, out_W)` | same_as_input | NCHW 连续布局，`out_idx=[1]` |

**输出 shape（迁移必需字段）**：`(N, C, out_H, out_W)`，其中 `out_H = H_in if oH is None else oH`、`out_W` 同理。
输出与输入同布局（NCHW），**不是转置布局**。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（`kernels/pool/adaptive_avg_pool2d.py`，逐行）：

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host，trace 期） | `total_output = n*c_in*out_h*out_w`；`max_kh = max_adaptive_bin_extent(h_in,out_h)`；`max_kw = max_adaptive_bin_extent(w_in,out_w)` | 编译期 shape 常量 | Python int | bin 循环的**静态上界**（TileLang 拒绝动态 `T.serial` 边界） |
| 2（grid） | `T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx` | `block_m`、`threads` | 一维 block 网格 | 输出元素级并行划分 |
| 3（lane 索引解码） | `out_idx = bx*block_m + i`；`ow=out_idx%out_w`；`oh=(out_idx//out_w)%out_h`；`c=(out_idx//(out_w*out_h))%c_in`；`batch=out_idx//(out_w*out_h*c_in)`；经 `if out_idx < total_output` 守卫 | `out_idx` | 标量 `(batch,c,oh,ow)` | 输出坐标 $(n,c,o_h,o_w)$ |
| 4（bin 边界） | `ih_start,ih_end = adaptive_bin(oh,h_in,out_h)`；`iw_start,iw_end = adaptive_bin(ow,w_in,out_w)` | 标量 | 标量对 | $\text{bin}_H(o_h)$ / $\text{bin}_W(o_w)$ |
| 5（bin 累加） | `sum_val = 0`；`for kh in T.serial(max_kh): for kw in T.serial(max_kw): if ih_start+kh<ih_end and iw_start+kw<iw_end: sum_val += T.cast(x[batch,c,ih_start+kh,iw_start+kw], "float")` | `x` 的逐元素 GM 读 | 标量 fp32 | 双重求和 $\sum_{i_h}\sum_{i_w}$ |
| 6（归一化 + 写出） | `bin_size=(ih_end-ih_start)*(iw_end-iw_start)`；`out[batch,c,oh,ow] = T.cast(sum_val / T.cast(bin_size,"float"), dtype)` | 标量 | 标量 | $\frac{1}{\lvert\text{bin}\rvert}\sum$ |

**数据流与内存访问模式（源硬件视角）**：

```
GM[x] --逐元素标量读 x[batch,c,ih,iw]--> register(sum_val, fp32) --bin_size 除法--> 单次 cast --> GM[out]
（无 SMEM/L1 中转；每个输出元素现场从 GM gather 其 bin 的全部元素；相邻输出元素重复读重叠行/列）
```

**循环与并行结构**：
- grid = `ceildiv(total_output, block_m)` 个 block，每 block `threads` 个 CUDA 线程；
- `T.Parallel(block_m)`：每个线程（lane）负责**一个输出元素**（lane 内是**标量**计算）；
- lane 内循环：`T.serial(max_kh) × T.serial(max_kw)` 嵌套，**静态边界** + 运行时 `if` 守卫跳过界外 lane（源码注释：`TileLang rejects dynamic T.serial bounds`）；
- 无流水线（无 `T.Pipelined`）、无共享内存、无归约原语。

**host 侧逻辑**（`pool.py`）：
- `_normalize_output_size`（把 `output_size` 规范成 2 元组）、`_resolve_out_dims`（`None` → 输入尺寸）；
- 3D→4D `unsqueeze(0)` / 输出 `squeeze(0)`（**只改 stride/view，不改数据**，符合 host 侧约束）；
- `x = x.contiguous()`（无实际拷贝时 no-op）；
- `_kernel_cache`（按 `(n,c,h,w,out_h,out_w,dtype,device)` 缓存 kernel 实例）；
- `functools.lru_cache(maxsize=32)` 缓存 kernel factory（按 shape/dtype）；
- `_adaptive_pool2d_roofline`（仅用于 manifest 性能规格，不参与计算）。
- 编译配置：`@tilelang.jit(out_idx=[1], compile_flags=["-O3","-DENABLE_BF16"])`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **静态边界 bin 循环 + 运行时守卫** | 规避 TileLang「拒绝动态 `T.serial` 边界」限制，同时不虚增循环体 | `max_adaptive_bin_extent` 在 trace 期算出最宽 bin 作为静态上界；`if ih_start+kh<ih_end and ...` 跳过界外 lane | 无（语言/编译器级约束） | **可移植**（NPU 同样只接受静态 `T.serial` 边界） |
| 2 | **fp32 累加 + 单次输出舍入** | 精度：避免逐元素 fp16 累加的舍入累积 | `accum_dtype="float"`，逐元素 `T.cast(x,"float")`，末尾一次 `T.cast(...,dtype)` | 无（纯数值语义） | **保留**（语义层） |
| 3 | **bin 大小整数边界取整除法** | 与 PyTorch adaptive 语义逐位一致 | `bin_size` 由整数 bin 边界差算出（非比例近似） | 无 | **保留**（语义层） |
| 4 | **一维 block 网格 + 输出元素扁平化映射** | 把 `(N,C,oH,oW)` 四维问题压成一维可并行任务 | `T.ceildiv(total_output, block_m)` + `%`//`//` 解码 | GPU block 网格（非 hardware-specific） | **等价替换**（一维 Kernel） |
| 5 | **`T.Parallel(block_m)` 线程级 lane 并行** | 每个 CUDA 线程处理一个输出元素，隐藏访存延迟 | CUDA thread/warp 两级并行模型 | **CUDA threads / warp**（NPU 无等价） | **重新设计**（→§0.6 R1） |
| 6 | **lane 内标量串行 bin 累加** | 单线程顺序累加、寄存器驻留 | `sum_val` 寄存器 + 逐元素 GM gather | **GPU 大寄存器堆 + 寄存器累加** | **重新设计**（→§0.6 R2） |
| 7 | **逐元素 GM gather（无 tile 搬运 / 无 coalescing）** | 窗口不规则，源码直接按 bin 坐标逐元素取数 | 直接索引 `x[batch,c,ih,iw]` | GPU 的 L1/L2 缓存吸收重复访问 | **重新设计**（→§0.6 R3；NPU 侧应走 UB tile 搬运 + 向量化） |
| 8 | **`functools.lru_cache` factory 缓存 + `_kernel_cache` 实例缓存** | 避免重复 trace/编译 | host 侧缓存 | 无 | **保留**（host 侧） |
| 9 | **`adaptive_bin` / `max_adaptive_bin_extent` trace 期常数求值** | 把 bin 算术提前到 trace 期，运行期只剩边界差 | Python 整数运算（`adaptive_bin` 注释：`Evaluated while tracing`） | 无 | **保留**（host/trace 侧） |

> 识别不出来 ≠ 不存在。以上 9 项为源码中可见的全部优化/结构性选择；未识别的项会在迁移中被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（adaptive bin 均值、bin 整数边界、输出尺寸规则） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| I/O 契约（NCHW→NCHW、fp16/bf16、同 dtype） | 语义 | 无 | — | **保留** | 同上；layout 是契约 |
| fp32 累加 + 单次输出舍入（#2） | 语义 | 无 | — | **保留** | 数值语义；NPU Vector 支持 fp32 `T.vadd`（`docs/Tilelang.language/数学操作/T.vadd.md` §2.2.1 fp32 ✓） |
| nan/inf 传播、非整除重叠 bin、空 bin 不存在 | 语义 | 无 | — | **保留** | 语义层 |
| 静态边界 bin 循环 + 守卫（#1） | 算法 | 无（语言约束，非硬件） | 有（同款约束） | **保留** | NPU `T.serial` 同样只接受静态边界（源码注释即为此约束而写）；守卫逻辑原样保留 |
| `adaptive_bin` / `max_adaptive_bin_extent`（#9） | 算法/host | 无 | 有 | **保留** | trace 期 Python 整数算术，host 侧不触碰张量内容 |
| lru_cache / `_kernel_cache`（#8） | host | 无 | 有 | **保留** | host 缓存 |
| 一维 block 网格 + 输出扁平映射（#4） | 优化 | GPU block 网格 | 有（一维 `T.Kernel`，NPU 支持 ≤2 维） | **等价替换** | migration-analysis.md §5.3「block×thread 两级并行 → block 级任务划分 + 向量化元素并行」；本项目 `T.Kernel` 一维为主（`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3：NPU 最多 2 维，1 维合法） |
| **`T.Parallel(block_m)` 线程级 lane 并行（#5）** | 优化 | CUDA threads / warp | **无** | **重新设计** → §0.6 R1 | NPU 无 CUDA thread 概念；每 lane 一个输出元素会退化为**标量**执行（与 PL-1.2 的 98.3% 标量反例同源） |
| **lane 内标量串行 bin 累加（#6）** | 优化 | GPU 寄存器堆 + 寄存器累加 | **无直接等价** | **重新设计** → §0.6 R2 | migration-analysis.md §5.3「warp shuffle 归约 → Vector 归约（重新设计）」「register 累加 → UB 中间缓冲」；NPU 侧改成 Vector 向量累加 |
| **逐元素 GM gather（#7）** | 优化 | GPU L1/L2 缓存 | **无等价（且有害）** | **重新设计** → §0.6 R3 | NPU 逐元素 gather 无 coalescing、无 cache 吸收；应改为 UB tile 搬运 + 向量化（pattern-library PL-1.2/PL-1.4） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目 / `docs/Tilelang.language/` 条目 / pattern-library 实测条目（见上表「依据」列）。
> **舍弃项：无。** 源码未使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async 等 GPU 专有手段，故无舍弃条目。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（每个重设计项含「源方案 → NPU 新算法 → 语义保持论证」）。
**重设计输入含 §1.6.0 算法调研结论**：调研选定「直接自适应 bin 累加 + C 轴向量化」算法族（与源算法同族但**累加单元由标量改为 C 向量**），
故 §0.6 的重设计可直接承接调研结论；源算法已识别优化手段（§0.4）的优化意图承接见每项「承接」行。

**重设计项 R1：并行结构重设计（线程级 lane 并行 → 一维 Kernel + C 轴向量任务划分）**

- **源方案**：`T.Kernel(ceildiv(total_output,block_m), threads=threads)` + `T.Parallel(block_m)`，每 CUDA 线程处理一个输出元素。
  性能意图：把 `N*C*out_H*out_W` 个输出元素摊到大量线程上并行，靠 warp/线程切换隐藏逐元素 GM gather 的延迟。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`；**逻辑任务单元 = `(n, o_h, c_block)`**，其中 `c_block` 是一个长度为 `CH` 的**通道向量块**；
  `num_logical = N * out_H * ceil(C/CH)`（**一般网格**；退化路径 C4/C4b 的单元 = 整张 `(n, c_block)` 平面，其 `num_logical = N × ceil(C/CH)`，见 §5.5 ① / §1.6.1 C4/C4b）。单元内以 **C 轴为向量维**、`o_w` 为编译期展开的 Python 循环、`o_h` 由单元 id 运行时解码。
  `num_logical > vector_cores` 时按 persistent grid-stride 模式挂 `T.serial` 核内循环（承接源方案「把任务摊到并行单元」的意图；见 §5.5）。
- **语义保持论证**：任务划分只是把互不相关的输出元素分给不同单元，**不改变任何元素的求和集合与求和顺序**（每个输出仍按 `kh` 外层、`kw` 内层累加其 bin）；无跨单元归约，无边界语义变化。
  承接：源 #4 的一维扁平映射被保留为「一维 Kernel」，#5 的 lane 并行意图由「单元并行 + C 轴向量并行」承接。

**重设计项 R2：归约重设计（per-lane 标量 bin 累加 → UB 内 Vector 向量累加）**

- **源方案**：每个输出元素在寄存器里做 `sum_val += T.cast(x[batch,c,ih,iw],"float")` 的标量串行累加。
  性能意图：寄存器驻留、单线程顺序累加、避免跨线程归约。
- **NPU 新算法**：
  1. 逐元素 cast 改为 **tile 级向量 cast**：`T.vcast(row_dtype, row_f32, round_mode="rint")` 一次把整行 `(W_in, CH)` 提到 fp32，
     而非源码的「每元素一次 `T.cast`」。
  2. bin 累加改为 **C 轴向量累加**：`acc[o_w, 0:CH] (fp32) += row_f32[iw0(o_w)+kw, 0:CH]`，用 `T.vadd`（fp32 ✓）。
     `kh` 为 `T.serial(max_bin_h)` 静态边界 + `if kh < bin_h` 守卫（承接 §0.4 #1 的「静态边界 + 守卫」手段）；
     `kw` 为 Python 编译期展开（`bin_w(o_w)` 在 trace 期是常数，见 §1.6.3）。
  3. 归一化：`acc[o_w] *= 1/bin_size`（`T.vmul` 标量重载，`docs/Tilelang.language/数学操作/T.vmul.md` §2.3「仅支持 src2 为标量」）；
     最后 `T.vcast(acc, out_ub, round_mode="rint")` 单次舍入回输出 dtype。
- **语义保持论证**：
  - **数学等价**：分块（`kh`/`kw` 循环）不改变求和集合；C 轴是**独立轴**（每个 `c` 独立求均值，见 §0.1 规约语义），
    把 `c` 放进向量 lane 只是 SIMD 数据并行，**不改变任何 `c` 的累加顺序**（每个 lane 仍是 `kh` 外、`kw` 内的顺序累加）。
  - **dtype/舍入**：与源一致（fp32 累加、末尾单次 `rint` 舍入）；`vcast` 的 fp16/bf16→fp32 为无损上转，fp32→fp16/bf16 用 `rint`（四舍五入到偶数），与源 `T.cast` 默认舍入路径一致（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1）。
  - **边界语义**：非整除重叠 bin（`bin` 边界用整数差，与源同式）、空 bin 不存在、nan/inf 按 IEEE 传播——全部不变。
  - **累加顺序差异**：源按 `for kh: for kw:` 顺序；NPU 侧相同（`kh` 外层 serial、`kw` 内层展开），**逐元素顺序一致**。
  - 承接：源 #2（fp32 累加 + 单次输出舍入）与 #6 的「寄存器累加」意图，分别由「UB fp32 向量累加器」与「tile 级 vcast + vadd」承接。

**重设计项 R3：数据搬运重设计（逐元素 GM gather → UB tile 搬运 + 核内融合转置）**

- **源方案**：直接在 GM 上按 `x[batch,c,ih_start+kh,iw_start+kw]` 逐元素取值。性能意图：接口简单；GPU 靠 L1/L2 缓存吸收重叠行/列的重复访问。
- **NPU 新算法**：按 C 轴向量化所需布局（C 最内连续）搬运：
  1. GM→UB：按 `(CH, W_in)` 行片整片搬运 `T.copy(x[n, c0:c0+CH, ih, 0:W_in], row_src)`（同一 `c` 的整行连续，DMA 段长 `W_in*2` 字节）；
  2. UB 内**融合转置链**：`T.transpose(row_src, row_dst, permutation=[1,0])` 把 `(CH, W_in)` 变成 `(W_in, CH)`（C 最内）；
     依据 pattern-library `PL-1.1-transpose-chain`（核内二轴交换链，实测 ~µs 级；`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 仅支持二轴交换）；
  3. 转置后 C 轴连续 → `T.vcast` 到 fp32 → 向量累加。
  4. 输出侧：`acc (out_W, CH)` 经 `T.transpose(acc, acc_t, [1,0])` 变 `(CH, out_W)` 后 `T.copy` 回 `out[n, c0:c0+CH, o_h, 0:out_W]`（同一 `c` 的整行连续）。
- **语义保持论证**：搬运/转置是**纯数据重排**，不改变元素值、不改变求和集合与顺序（转置是双射，累加仍在同一 `(kh,kw)` 顺序对上做）；输出写回后与 NCHW 契约一致。无精度影响（不引入任何算术）。
  承接：源 #7「靠缓存吸收重复访问」的意图由「UB tile 常驻 + 重叠行/列只搬一次（单元内 `kh` 循环复用 row 片）」承接。

**重设计项 R4：算法替代/结构特化（退化网格）**

> 该项属「算法替代」模式（migration-analysis.md §6.2），采纳项与 §1.6.0 调研结论一致、并在 §1.6.1 完成等价性论证与机器验证。

- **源方案**：对所有输出网格一律走双重 bin 循环（含 `out_H=out_W=1` 的全局池化、以及 `out==in` 的恒等情形）。
- **NPU 新算法**（factory 级结构分发，形状静态可判定）：
  - **C4（全局池化，`out_H == 1 and out_W == 1`）**：bin(0,0) = `[0,H_in)×[0,W_in)` 即整个平面 →
    按 `(CH, H_in*W_in)` 连续块搬运 + `T.vcast` 到 fp32 + `T.reduce_sum(buf, acc, dim=1)`（fp32 ✓，`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1）+ `T.vmul(acc, 1/(H_in*W_in))`（`1/(H·W)` 为编译期常数）。
  - **C4b（恒等，`out_H == H_in and out_W == W_in`）**：所有 bin 大小 1 → `out = x`，走纯拷贝路径（UB 中转，同 dtype 不 cast）。
- **语义保持论证**：见 §1.6.1（C4/C4b 均与基线公式**逐位等价**；`verify_equiv.py` 45 用例全 EQUIV_PASS，含 C1 容差内 ≤1 ulp）。
- 承接：源方案「一般 bin 循环」在非退化网格上保持不变（不存在优化意图丢弃）；退化网格上的特化**不改变语义**，仅减少冗余循环。

**重设计后的 NPU 算法一句话总结**：NCHW 进 / NCHW 出，核内按 `(n,o_h,c_block)` 单元把 C 轴作为向量轴做自适应 bin 的 fp32 向量累加，
用核内二轴转置把 C 搬到最内连续轴；退化网格（全局池化 / 恒等）走结构特化路径。

### 0.7 标杆实现

- 源算子路径（获取 golden 语义依据）：`/workspace/TileOPs-gpu/tileops/ops/pool.py`（`AdaptiveAvgPool2dFwdOp`）。
- 源仓参考实现 / 测试基准：
  - `examples/TileOPs/tests/ops/test_pool.py`（NPU 侧移植后的用例，`ref_program = F.adaptive_avg_pool2d(input, size)`，torch CPU/NPU 参考）；
  - 源仓同文件（GPU）`tests/ops/test_pool.py`。
- **golden 以 §0.1 语义为唯一依据**，直接使用 `torch.nn.functional.adaptive_avg_pool2d`（优先移植源仓基准），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。
- 规格来源：`examples/TileOPs/tileops/manifest/pool.yaml` 的 `AdaptiveAvgPool2dFwdOp`（4 组 workloads）。
---

## 1. 概述

### 1.1 算子名称

`_adaptive_avg_pool2d_kernel`（迁移自 TileOPs `AdaptiveAvgPool2dFwdOp` / `AdaptiveAvgPool2dKernel`）。

### 1.2 功能描述

自适应二维平均池化前向：把 NCHW 输入的空间轴按 PyTorch adaptive-bin 规则（`[floor(o*in/out), ceil((o+1)*in/out))`）划分，
每个输出元素为其 bin 内的算术平均值。

### 1.3 数学公式

$$
\text{out}[n,c,o_h,o_w]=\frac{1}{\lvert \text{bin}_H(o_h)\rvert\,\lvert \text{bin}_W(o_w)\rvert}\sum_{i_h\in\text{bin}_H(o_h)}\sum_{i_w\in\text{bin}_W(o_w)}x[n,c,i_h,i_w],
\qquad
\text{bin}_H(o_h)=\big[\lfloor o_h H_{in}/H_{out}\rfloor,\ \lceil (o_h+1)H_{in}/H_{out}\rceil\big)
$$

（`bin_W` 同理；`H_out = H_in if oH is None else oH`。）

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，见 §0.6）** 单步 + 窗口归约类算子，纯 Vector：

1. host/trace 期（factory 内）：由 `(H_in,out_H)`/`(W_in,out_W)` 算出每个 `o_h`/`o_w` 的 bin 边界 `(start,end)`（`adaptive_bin`）
   与静态上界 `max_bin_h = max_adaptive_bin_extent(H_in,out_H)`、`max_bin_w = max_adaptive_bin_extent(W_in,out_W)`；这些量都是编译期常数。
2. 逻辑任务单元 `(n, o_h, c_block)`（`CH` 通道向量块）；单元内：
   - `acc`（`out_W × CH` fp32）经 **`T.clear` 整块零初始化**（无逐元素标量循环，见 §1.6.2 #0）；
   - `for kh in T.serial(max_bin_h)`：若 `kh < bin_h(o_h)`（运行时守卫），搬入输入行 `(CH, W_in)` → 核内转置 `(W_in, CH)` → `vcast` 到 fp32；
     随后对每个 `o_w`（编译期展开）把该 `o_w` 的 W-bin 片段逐切片 `vadd` 进 `acc[o_w]`；
   - 归一化 `acc[o_w] *= 1/(bin_h*bin_w)` → `vcast` 回输出 dtype → 转置回 `(CH, out_W)` → 写回 GM。
3. 退化网格（`1×1` 全局池化 / `out==in` 恒等）走 §0.6 R4 的结构特化路径。

**与源算法的结构差异**（来源标注）：
- 累加单元：源「每输出元素一个线程 + 标量寄存器累加」→ NPU「每单元一个 C 向量块 + `T.vadd` 向量累加」（§0.6 R2）。
- 访存：源「逐元素 GM gather」→ NPU「UB tile 搬运 + 核内融合转置」（§0.6 R3）。
- 并行：源「block×thread 两级 + lane/元素」→ NPU「一维 Kernel + `(n,o_h,c_block)` 单元 + C 轴向量 lane」（§0.6 R1）。
- 退化网格特化：源无 → NPU 有（§0.6 R4，见 §1.6.1）。
- **不变**：bin 边界整数算术、`kh` 外 `kw` 内的累加顺序、fp32 累加 + 单次输出舍入、静态边界 bin 循环 + 守卫、NCHW 契约。

### 1.5 数据流图

```
GM[x: (N,C,H_in,W_in) fp16/bf16]
   │  T.copy 行片 (CH, W_in)  [同一 c 的整行连续]
   ▼
UB[row_src: (CH, W_in) dtype] --T.transpose [1,0]--> UB[row_dst: (W_in, CH) dtype] --T.vcast(rint)--> UB[row_f32: (W_in, CH) f32]
   │                                                                                                            │  T.serial kh (静态界+守卫)
   │                                                                                                            ▼
   │                                                        UB[acc: (out_W, CH) f32] <--T.vadd 切片累加（C 轴向量）
   │                                                                  │  T.vmul(acc, 1/bin_size) ; T.vcast(rint)
   │                                                                  ▼
   │                                                        UB[acc_t: (CH, out_W) dtype] --T.transpose [1,0]
   ▼
GM[out: (N,C,H_out,W_out) same_as_input]   ← T.copy 行片 (CH, out_W)
（退化路径 C4：GM → UB 连续 (CH, H*W) → vcast f32 → T.reduce_sum(dim=1) → vmul → vcast → GM）
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = **同一数学语义的算法族**（所有能算出 adaptive-bin 均值的结构）：**基线 = 源算法**（直接自适应 bin 累加，逐元素标量）。
> 本算子属**窗口/归约类** → 完整调研（R1 候选表 + R3 四口径表 + R4 逐候选评估）。
> 参考表命中行：`algorithm-candidates.md` 的 `ALG-pooling`（family=pooling）——`R1: avg pool = 常数权重卷积；sum pool × 常数 = avg pool`；`R2: 滑窗天然流式`；`R3: 窗口重叠数据的重复读`；`R4: C 轴向量轴；跨步系数`；`known_impl` 指向 `avg_pool2d` 系列，`kb_links` = `PL-1.2-caxis-accum / PL-1.3-host-permute / PL-1.4-tiling-heuristic`。命中行已全部评估（见 R1/R4）。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算法**（直接自适应 bin 累加，逐元素标量 + `sum/bin_size`） | $\frac{1}{d}\sum_{kh}\sum_{kw}x[ih_0+kh,iw_0+kw]$ | — | — | ✅（基线） |
| 1 | 倒数乘（`sum/d` → `sum·(1/d)`，PL-1.5「乘编译期常数倒数」族） | $\big(\sum x\big)\cdot\frac1d$ | 容差内等价（fp32 下 `1/d` 与除法的舍入差 ≤1 ulp，再单次舍入到 fp16/bf16） | 把每输出 1 次向量除换成 1 次标量倒数 + 1 次向量乘 | ✅ |
| 2 | 可分离 2D→1D（H/W 两轴分两遍） | $\sum_{ih\in B_H}\big(\sum_{iw\in B_W}x\big)$（行/列部分和复用） | 数学恒等（求和可交换） | 减少重叠行/列的重复读 | ✅ |
| 3 | 前缀和 + 差分（`T.cumsum` 后段差） | $\text{out}=\sum P[iw_1]-P[iw_0]$（P = W 轴前缀和） | 数学恒等（$P[b]-P[a]=\sum_{[a,b)}x$） | 用 $W_{in}$ 次前缀和替代 $scan_W$ 次累加 | ✅ |
| 4 | 退化网格结构特化（`1×1` / 恒等） | `1×1`: $\frac{1}{H W}\sum_{\text{all}}x$；恒等: `out=x` | 数学恒等（bin(0,0)=全平面；单位 bin 均值=元素本身） | 全局池化：$H{\cdot}W$ 次向量加 → 1 次向量归约 | ✅ |
| 5 | sum-pool × 常数（ALG-pooling 命中行） | 同 #1（`Σx` 后乘常数） | 与 #1 同 | 与 #1 同（本算子即 avg pool，无独立 sum-pool 语义） | ❌ 与 #1 合并 |
| 6 | im2col + 常数权重卷积 / Cube GEMM（ALG-pooling `R1` 命中行） | `W0 @ x`（$W_0$ 为 $out_W\times W_{in}$ 0/1 权重矩阵） | 数学恒等（自适应 bin 可表为 0/1 权重矩阵乘） | 用 Cube MAC 替代 Vector 累加 | ❌ 见 R4（弃选理由） |

**R2 在线算法**：**有**（且基线本身已是单遍）。结构判据：算子可分解为 **running 累加量**（bin 内的和），无跨元素依赖的中间统计量，
故天然存在单遍流式变体——每个输入元素读一次、累加进唯一归属的（若干）输出 bin，运行状态为 `O(out_W·CH)` 的累加器。
**收益口径**：扫描遍数 1（无两遍式）、中间缓冲 `O(out_W·CH)`（远小于 `O(H·W)` 的物化）、UB 驻留完全可行（见 §4.5 预算）。
迁移任务佐证：源算法 §0.4 #6 已是 running 累加（`sum_val += ...`），是「存在在线变体」的直接证据。
结论：**在线单遍变体 = 采用（= 基线结构），无第二遍扫描/无中间矩阵物化**。

**R3 复杂度对比**（逐候选四口径；本算子是**带宽受限**算子，访存/遍数为主导项，不只看 FLOPs）：

以 `spp-6x6`（`N=2,C=128,H=W=56,out=6×6`，`scan_H=scan_W=60`，fp16）为口径示例；口径：FLOPs 计「累加次数」（1 add/次），
访存 Bytes = GM 读 + GM 写（**不含** UB 内搬运；中间缓冲不落 GM），扫描遍数 = 输入 GM 读取遍数，中间缓冲峰值 = 片上驻留量。

| 算法候选 | FLOPs（累加次数） | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|--------------------|---------|-------------|---------------------|
| 0 基线（直接标量） | `N·C·scan_H·scan_W` = 921 600 | `(N·C·H·W + N·C·out_H·out_W)·2` = 1.624 M | 1 | `O(1)`（寄存器标量） | 元素级并行 `N·C·out_H·out_W = 9216`；**无跨核归约** |
| 0' 基线向量化（**选定**，C 轴） | 921 600（/CH 次向量加） | 同上 1.624 M（+ 重叠行 ~7% 重复读） | 1 | `acc(out_W·CH)f32` + `row(W_in·CH)` ≈ 15–30 KB | 单元并行 `N·out_H·ceil(C/CH)`；**无跨核归约** |
| 1 倒数乘 | 921 600（归一化 1 vec-div → 1 vec-mul） | 同上 | 1 | 同上 | 同上 |
| 2 可分离 2D | `H_in·scan_W + scan_H·out_W` = 3360+360 = **3720/平面**（> 3600 = scan_H·scan_W） | 同上（+ 行部分和缓冲 `H_in·out_W·CH`） | 1 | + `R(H_in·out_W·CH)` | 同上，但多一中间缓冲 |
| 3 前缀和 + 差分 | `H_in·out_H`(cumsum) + `N·C·out_H·out_W·2`(差) ≈ `N·C·out_H·(H_in/CH + 2·out_W)` | 同上 | 1 | + `P(W_in·CH)` | 同上 |
| 4 退化特化（`1×1`） | **1 次向量归约**（vs 基线 `H·W` 次向量加） | 同上 | 1 | `(CH, H·W)` 块 | 同上 |
| 6 im2col+Cube GEMM | `out_W·W_in·H_in·…` ≈ 同阶（MAC=2 FLOP 计则 2×） + 权重矩阵构造 | 同上 **+ 权重矩阵 GM 往返** | 1 | L0A/L0B/L0C 分形 + 权重矩阵 | 需分形对齐（`out_W=6 < 16` 不满足 M≥16）+ 权重构造 |

> 口径含：GM 输入读 + 输出写；不含：UB 内搬运/转置（片上，计入 §1.6.0 估算行的发射项）。候选 2/3/6 的额外中间缓冲已单列。
> **基线在表内**（#0）；选定候选 #0'（基线向量化，与调研结论一致）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/`/`examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线（标量） | **Vector 单元浪费**（逐元素标量累加，PL-1.2 实测窗口累加 98.3% 标量） | 寄存器 | — | 静态（`max_bin_*` 界） | 无 | ❌ 计算单元失配（重设计为 #0'） |
| **0' 基线向量化（C 轴，选定）** | **Vector 逐元素/归约** ✓（`T.vadd` fp32 ✓ / `T.reduce_sum` fp32 ✓） | `acc`+`row` ≈ 15–30 KB ≪ UB 192 KB（`CONST-capacity-910B2C`） | **C 轴恒为 8 的整数倍**（C=8/64/128/96/2048；fp16 向量宽 8）；`W_in`=55/56/57 仅作 row 长度，不要求整除 | ✓ 全部编译期静态（`adaptive_bin` 为 trace 期整数） | 无跨引擎流水；退化网格可 reduce 融合 | ✅ **选定** |
| 1 倒数乘 | 同 #0'（向量乘） | 同 #0' | 同 #0' | ✓ | 同 #0' | ✅ 采纳为微优化（并入 #0'；正式论证见 §1.6.1 **C1**，容差内等价，已机器验证） |
| 2 可分离 2D | Vector ✓ | 需 + 行部分和缓冲（`H_in·out_W·CH`，spp CH=64 时 ≈ 86 KB） | 同 #0' | ✓ | 两遍串行 | ❌ **运算量更大**（3720 > 3600/平面）+ 额外缓冲；详见 §1.6.1 |
| 3 前缀和 + 差分 | Vector ✓（但 `T.cumsum` 仅 fp16/fp32/int32/int64，bf16 ×，需先 vcast） | + `P(W_in·CH)` | ✓ | ✓ | 串行 | ❌ **精度风险**（`P[b]-P[a]` 相消误差）+ 段差为标量读、额外缓冲；见 §1.6.1 |
| 4 退化特化 | Vector reduce ✓ | `(CH,H·W)` 块 ≈ 19 KB（resnet C4，`CH=min(C,64)=64`，见 §4.5） | ✓ | ✓ | 单归约 | ✅ 采纳（`1×1` / 恒等） |
| 6 im2col+Cube GEMM | Cube MAC ✓ 但**主导计算量极小**（窗口和），投入 Cube 严重失配 | L0C 128 KB（`CONST-capacity-910B2C`） | ❌ **`out_W = 6/7 < 16` 不满足 Cube M≥16 分形**；`W_in=55/57` 亦非 16 倍数（`ascend-constraints.md` §1「GEMM 要求 M,N 为 block 整数倍」） | ✓ | 需权重矩阵构造 | ❌ 分形不满足 + 权重矩阵需构造（host 侧只允许视图操作，构造 0/1 权重属改内容；`ascend-constraints.md` §4）+ L0C 浪费 |

**R4 弃选理由的证据引用（negative-claim-evidence 口径）**：
- 候选 2/3：运算量为**我方独立复算**（每平面 `H_in·scan_W + scan_H·out_W` vs `scan_H·scan_W`，见 R3 表）；精度风险引用 `T.cumsum` 的语义（前缀和差分）为基础数学事实，非 API 断言。
- 候选 6：`out_W < 16` 违 Cube 分形下限，条款见 `.agents/skills/tilelang-op-design/references/ascend-constraints.md` §1（「GEMM 要求 M,N 为 block 整数倍」行）与 `docs/Tilelang.language/线性代数操作/T.gemm.md`；weight-matrix 构造属 host 侧改内容，条款见 `ascend-constraints.md` §4。
- 候选 0 的「标量占比」反例引用 pattern-library `layout.md` PL-1.2/诊断触发器（「窗口累加 98.3% 标量源于 `j·sW`（sW≠1）跨步系数阻碍向量化」）。
- 候选 1/3 的 dtype 支持矩阵来自 `docs/Tilelang.language/`（`T.vadd`/`T.vmul`/`T.reduce_sum`/`T.cumsum` 各 §2.2.1 表）。

**设计期估算（D-2 roofline）** —— 协议见 `_shared/standards/hardware-cost-model.md` §2，常数引用 `pattern-library/constants.md`：

| workload | 流量项（`bytes ÷ 有效带宽`） | 发射项（向量 op 数 × 代价） | 容量项 | 估算下界 |
|----------|------------------------------|------------------------------|--------|---------|
| `resnet-global`(8,2048,7,7) | 1.638 MB ÷ 1.26 TB/s = **1.30 µs**（HBM 3:1 混合地板，`CONST-mte2-degradation`，toolchain 2026-09-21 / 15ad002b3d）；L2 驻留口径（sets=1）÷ 2.07 TB/s = 0.79 µs | ≈ 256 单元 × ~4 op = 1024 op（C4 特化）；每 op 代价见下注 | 0（无跨引擎往返） | **≈ 1.3 µs** |
| `spp-6x6`(2,128,56,56) | 1.624 MB ÷ 1.26 TB/s = **1.29 µs** | 14400 `vadd` + ~720 cast/transpose ≈ 15 K op | 0 | **≈ 1.3 µs** |
| `nondiv-7x7`(2,64,55,57) | 0.815 MB ÷ 1.26 TB/s = **0.65 µs** | 7686 `vadd` + ~380 cast/transpose ≈ 8 K op | 0 | **≈ 0.7 µs** |

> **发射项口径注**：`CONST-vector-launch-overhead`（`~0.5µs/op`，toolchain 2026-09-09 / 3a214cde）是 **bn=512 宽块大 op** 的实测值，
> 对本设计 ≤128 元素的窄向量 op **不能直接外推**（按 0.5µs/op 会得 ~7 ms 的荒谬值）——窄向量 op 代价**未逐点实测**，标注「未文档化假设」；
> **方向证据已有实测**：`layout.md` PL-1.4 同族池化实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs」（更宽 C 向量更优），与 `CONST-vector-launch-overhead`「宽块优于窄块 op 翻倍」方向一致；
> 纯吞吐下界约 `op 数 × 向量宽 /(128 elem/cycle)`，按 40 核并行后每核约 10–40 µs 量级（估算依据：向量单元吞吐量级，未实测）。
> 端到端延迟另需加 NPU **kernel launch + Event 计时开销 ~50–60 µs**（`examples/TileOPs/docs/gpu_to_npu_adaptation.md` C12 行），
> 故实测地板大概率由固定开销主导，而非流量项。
> 容量项为 0：本设计无 `store_fixpipe` 类跨引擎中间矩阵往返（纯 Vector 单 kernel）。

**调研结论**：选定算法族 = **「直接自适应 bin 累加 + 中间累加 fp32 + C 轴向量化 + 核内融合转置（NCHW→C 最内）」**（R3 表候选 #0'；另采纳候选 #1 倒数乘 → §1.6.1 **C1**、候选 #4 退化特化 → §1.6.1 **C4/C4b**；三项采纳与 §1.6.1 优化结论**逐项一致**）。
关键依据：R3 表显示 #0' 的 FLOPs（= `N·C·scan_H·scan_W`，与 manifest `roofline.flops` 逐字一致）与访存量（= manifest `roofline.bytes`）均为各候选最小量级，
候选 2/3/6 不降反升或不可行；R4 表显示 #0' 唯一同时满足「Vector 计算单元匹配 + C 轴向量整除 + 全静态边界 + 无跨核归约 + 容量充裕」。
**与基线的结构差异一句话**：数学公式与求和顺序不变，累加单元由「每输出一线程的标量寄存器」改为「`(n,o_h,c_block)` 单元的 C 轴向量累加器（fp32 UB）」。
**调研范围声明**（「无更优替代」不成立——确有一项结构特化被采纳，但**无更低复杂度算法族**）：
本调研覆盖 ① `algorithm-candidates.md` 的 `ALG-pooling` 命中行全部候选（R1 常数权重卷积 / sum-pool×常数 / R2 滑窗流式 / R3 窗口重叠重复读 / R4 C 轴向量轴）；
② 恒等变形空间（倒数乘、可分离、前缀和差分）；③ pattern-library 同类实现（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum` / `PL-1.3-host-permute` / `PL-1.4-tiling-heuristic` / `PL-1.5-quickref`，均在 `tilelang-op-optimize/references/pattern-library/layout.md`）；
④ 本仓 `examples/` 同类实现（§3.5.3）。**互联网检索未使用**（本算子算法族在本地参考表 + 源码 + pattern-library 已完整覆盖，属简单窗口/归约类，无需联网补充）。
**下游约束**：§1.4 算法描述、§1.6.1–§1.6.3 的分析对象 = 选定算法 #0'；§0.5/§0.6 的处置与之**一致**（§0.6 R2 = #0' 的累加向量化、§0.6 R3 = C 轴布局、§0.6 R4 = 候选 #4）；
源算法 §0.4 已识别优化手段的意图承接逐项见 §0.6（**无静默丢弃**：9 项全部有处置，其中 #1+#2+#6+#7 的意图分别由 §0.6 R2/R3 承接）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的算法 #0'（= 基线结构 + C 轴向量化）。调研候选（R1 表）中初判可采纳的项在此完成四要素论证。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| C4 | **退化网格结构特化：全局池化（`out_H==1 and out_W==1`）** | 一般 bin 双循环求均值（`bin(0,0)` 由通用公式给出 $\sum_{kh<H}\sum_{kw<W}x /(H W)$） | $\Big(\sum_{\text{all}} x\Big)\cdot\frac{1}{H W}$——按 `(CH, H*W)` 连续块搬运 + 单次 `T.reduce_sum(dim=1)` + 编译期常数 `1/(H·W)` 乘 | **数学恒等**：`out_H=out_W=1` 时 $\lfloor 0\cdot H/1\rfloor=0$、$\lceil 1\cdot H/1\rceil=H$，故 `bin(0,0) = [0,H)×[0,W)` = 整平面；归一化因子 $H{\cdot}W$ 为编译期常数。**逐位等价**（同一元素集合、同一求和、同一除法） | `resnet-global`（`H=W=7`）：每单元 $H{\times}W=49$ 次向量加 → **1 次向量归约**（≈49×）；覆盖 4 组 manifest workload 中的 1 组（`resnet-global`，最大 workload） |
| C4b | **退化网格结构特化：恒等（`out_H==H_in and out_W==W_in`）** | 一般 bin 双循环（每个 bin 大小 1，`sum/1 = x`） | `out = x`（纯 UB 中转拷贝，同 dtype 不 cast） | **数学恒等**：`out_H=H_in` 时 `bin_H(o_h)=[o_h,o_h+1)`（$\lfloor o_h\rfloor=o_h$、$\lceil o_h+1\rceil=o_h+1$），单位 bin 均值 = 元素本身；逐元素恒等 | 覆盖测试用例 `full-all-none-identity-fp16`（9×11 恒等）与 `(None,…)`/`(…,None)` 的半恒等轴；消除全部冗余归约 |
| C1 | **倒数乘（`sum/d` → `sum·(1/d)`）——已采纳** | $\big(\sum x\big)\big/d$ | $\big(\sum x\big)\cdot\frac1d$，其中倒数 `inv = 1/d` 在 fp32 计算一次（标量），再与 fp32 累加器做向量乘 | **容差内等价（非逐位）**：fp32 下 `1/d` 相对除法的误差 ≤1 ulp（相对 ~6e-8），再单次 `rint` 舍入到 fp16/bf16（相对 ~1e-3 / ~4e-3）后，每个输出元素与基线差 **≤1 ulp**——已由 `verify_equiv.py` 的 C1 用例机器验证（45 用例全 `EQUIV_PASS`，见下），与 §8.2 精度标准（atol/rtol 1e-3 / 1.6e-2）相容 | 每输出列 1 次向量除 → 1 次标量倒数 + 1 次向量乘：spp 口径归一化向量 op 数 = `num_logical × out_W = 24 × 6 = 144`（`T.vdiv` → `T.vmul`），占「14400 累加 + 144 归一化」≈ **1.0%**。收益虽小但**零额外代价**（PL-1.5 已验证模式「乘编译期常数倒数 / divisor 恒定时 `T.vmul(acc, 1/d, acc)`」），且除法改乘降低向量管线压力。**采纳**（与 §0.6 R2 步骤 3、§3.3 伪代码实际取法一致） |
| C2 | 可分离 2D→1D——**未采纳** | $\sum_{i_h\in B_H}\sum_{i_w\in B_W}x$ | $\sum_{i_h\in B_H}\big(\sum_{i_w\in B_W}x\big)$ | 数学恒等（求和可交换） | **运算量为负收益**：spp 每平面 `H_in·scan_W + scan_H·out_W = 60·56 + 6·60 = 3720` > 基线 `scan_H·scan_W = 3600`；非整除（nondiv）差 3843 vs 3892，同样更差。且需额外行部分和缓冲。**不采纳** |
| C3 | 前缀和 + 差分——**未采纳** | $\sum_{[a,b)}x$ | $P[b]-P[a]$，$P$ = `T.cumsum` | 数学恒等（前缀和差 = 区间和） | 用 `cumsum` 替换累加并不减少 op 数（$W_{in}$ 前缀 + 段差 ≈ $scan_W$），且引入 **fp32 相消误差**（$P[b]-P[a]$ 对大 $P$、小区间时有效位丢失），并需额外 `P` 缓冲 + `T.cumsum` 不支持 bf16（`docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1）。**不采纳** |

**等价性机器验证（D-1）**：三项采纳优化全部由 `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/verify_equiv.py` 在 **fp64 参照**下对照基线通用 bin 均值验证（torch CPU），共 **45 用例**（4 组 manifest shape + 6 组 NPU 测试 shape + 退化/奇数/一般网格 shape + ±inf/NaN/subnormal/大值/全零角点）：
- **C4/C4b**（逐位等价，非仅容差内）：判据 = 舍入到目标 dtype 后与基线**逐位一致**（0 ulp）；
- **C1**（容差内等价）：判据 = 舍入到目标 dtype 后每个元素与基线差 **≤1 ulp**（`max_ulp ≤ 1`）；
  脚本同时报告逐位差个数（`bitdiff`，仅供参考）。`viol` 列计「超出该判据的元素数」（NaN/finite 不一致恒判失败）。结果表内嵌如下：

```
case                                             | kind | dtype    | max_abs_diff | viol | bitdiff | max_ulp | status
----------------------------------------------------------------------------------------------------------------------
C4/global/float16/(8, 2048, 7, 7)                | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/float16/(8, 2048, 7, 7)             | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/float16/(2, 128, 56, 56)               | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/float16/(2, 128, 56, 56)            | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/bfloat16/(2, 64, 55, 57)               | C4   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/bfloat16/(2, 64, 55, 57)            | C4b  | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/float16/(1, 8, 4, 4)                   | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/float16/(1, 8, 4, 4)                | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/float16/(1, 8, 9, 11)                  | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/float16/(1, 8, 9, 11)               | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/float16/(1, 16, 11, 13)                | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/float16/(1, 16, 11, 13)             | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/bfloat16/(1, 16, 10, 10)               | C4   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/bfloat16/(1, 16, 10, 10)            | C4b  | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/float16/(3, 5, 7, 3)                   | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/float16/(3, 5, 7, 3)                | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/global/bfloat16/(1, 1, 1, 1)                  | C4   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/identity/bfloat16/(1, 1, 1, 1)               | C4b  | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/reciprocal/float16/(2, 128, 56, 56)/out(6, 6) | C1   | float16  | 1.221e-04  |   0  |     15  |   1.00 | EQUIV_PASS
C1/reciprocal/bfloat16/(2, 64, 55, 57)/out(7, 7) | C1   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/reciprocal/float16/(1, 8, 4, 4)/out(2, 2)     | C1   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/reciprocal/float16/(3, 5, 7, 3)/out(2, 2)     | C1   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/reciprocal/bfloat16/(1, 16, 10, 10)/out(4, 4) | C1   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/special/subnormal/float16                     | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/special/subnormal/float16                    | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/special/subnormal/float16                     | C1   | float16  | 5.960e-08  |   0  |      1  |   1.00 | EQUIV_PASS
C4/special/large/float16                         | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/special/large/float16                         | C1   | float16  | 7.812e-03  |   0  |      1  |   1.00 | EQUIV_PASS
C4/special/inf-nan/float16                       | C4   | float16  | 0.000e+00  |   0  |      1  |   0.00 | EQUIV_PASS
C4b/special/inf-nan/float16                      | C4b  | float16  | nan  |   0  |      1  |    nan | EQUIV_PASS
C1/special/inf-nan/float16                       | C1   | float16  | 0.000e+00  |   0  |      2  |   0.00 | EQUIV_PASS
C4/special/zeros/float16                         | C4   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/special/zeros/float16                        | C4b  | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/special/zeros/float16                         | C1   | float16  | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/special/subnormal/bfloat16                    | C4   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/special/subnormal/bfloat16                   | C4b  | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/special/subnormal/bfloat16                    | C1   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/special/large/bfloat16                        | C4   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/special/large/bfloat16                        | C1   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4/special/inf-nan/bfloat16                      | C4   | bfloat16 | 0.000e+00  |   0  |      1  |   0.00 | EQUIV_PASS
C4b/special/inf-nan/bfloat16                     | C4b  | bfloat16 | nan  |   0  |      1  |    nan | EQUIV_PASS
C1/special/inf-nan/bfloat16                      | C1   | bfloat16 | 0.000e+00  |   0  |      2  |   0.00 | EQUIV_PASS
C4/special/zeros/bfloat16                        | C4   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C4b/special/zeros/bfloat16                       | C4b  | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS
C1/special/zeros/bfloat16                        | C1   | bfloat16 | 0.000e+00  |   0  |      0  |   0.00 | EQUIV_PASS

criterion: C4/C4b exact (0 ulp after rounding); C1 tolerance <= 1 ulp of target dtype; 'viol' counts admissible-tolerance violations (hard NaN/finite mismatches always fail); 'bitdiff' = bit-level differences (informational).

[EQUIV] ALL PASS — 45 cases, 0 failures
```

> `inf-nan` 用例的 `max_abs_diff=nan` 是 **NaN 对 NaN** 的预期结果（两式均产生 NaN），脚本已按 `both_nan` 归一化，`viol=0` → `EQUIV_PASS`。

**优化结论**：采纳 **3 项**——**C1**（倒数乘，容差内等价，机器验证 ≤1 ulp）、**C4**（全局池化结构特化）、**C4b**（恒等结构特化）；C4/C4b 为**逐位等价**。三项均已机器验证（§1.6.1 上表，45 用例全 `EQUIV_PASS`）；候选 C2/C3 经量化后不采纳（理由见上表）。
**优化后公式（供 §3.1 公式拆解使用）**：

```
一般网格 (out_H×out_W，非退化):
  out[n,c,o_h,o_w] = ( Σ_{kh<bin_h(o_h)} Σ_{kw<bin_w(o_w)} x[n,c,ih0(o_h)+kh, iw0(o_w)+kw] ) · (1/(bin_h(o_h)·bin_w(o_w)))   # C1：fp32 倒数 + 向量乘
退化网格:
  out_H==1 and out_W==1 :  out[n,c,0,0] = ( Σ_{all ih,iw} x ) · (1/(H_in·W_in))
  out_H==H_in and out_W==W_in :  out[n,c,:,:] = x[n,c,:,:]
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案（§1.4 + §3.3 伪代码）中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
> 替代方案所用 API 均以 `docs/Tilelang.language/` 条目佐证（下表「佐证」列）。

| # | 计算点 | 原实现形态（源 GPU） | 向量替代方案 | 是否替代 | 不可替代理由 / 佐证 |
|---|--------|---------------------|-------------|---------|--------------------|
| 0 | **累加器 `acc` 零初始化**（§3.3 单元入口） | 双重 Python 展开的逐元素标量存 `for o_w in range(out_W): for c in range(CH): acc[o_w,c] = 0.0`（spp 下 6×64 = 384 次/单元） | **`T.clear(acc)`**（整 buffer 清零；降级为 `hivm::VBrcOp`） | ✅ | 佐证：`docs/Tilelang.language/内存操作/T.clear.md`（1~5 维、fp16/fp32/bf16 ✓，`§2.4` 示例 `T.clear(A_VEC)`）；§9.2「累加器未清零」行原已给出 `T.clear` 方案 |
| 1 | lane 内 bin 累加 `sum_val += x[...]` | 逐元素标量累加（每输出元素一个） | `T.vadd(acc[o_w], row_f32[iw,:], acc[o_w])`（C 轴向量） | ✅ | 佐证：`docs/Tilelang.language/数学操作/T.vadd.md` §2.2.1 fp32 ✓；PL-1.2-caxis-accum（`layout.md`） |
| 2 | 逐元素 `T.cast(x[...],"float")` | 每元素一次标量 cast | `T.vcast(row_dst, row_f32, round_mode="rint")`（整 tile） | ✅ | 佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint` ✓） |
| 3 | 输出归一化 `sum / bin_size` | 每输出元素一次标量除 | `T.vmul(acc[o_w], inv, acc[o_w])`，`inv = 1/(bin_h·bin_w)` 为 fp32 标量（§1.6.1 **C1 已采纳**） | ✅ | 佐证：`docs/Tilelang.language/数学操作/T.vmul.md` §2.3「仅支持 src2 为标量」；PL-1.5「乘编译期常数倒数」 |
| 4 | 最终输出 cast | 每元素一次标量 cast | `T.vcast(acc, acc_t, round_mode="rint")`（整 tile） | ✅ | 同上 #2 |
| 5 | 退化网格全平面求和 | （源无此路径） | `T.reduce_sum(buf, acc, dim=1)` | ✅ | 佐证：`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1 fp32 ✓，`clear=True` 默认清空 |
| 6 | 布局重排（C→最内） | （源无；GPU 靠 coalesced 尾轴） | `T.transpose(src, dst, [1,0])` | ✅ | 佐证：`docs/Tilelang.language/创建操作/T.transpose.md` §2.3（二轴交换 ✓，fp16/fp32/bf16 ✓）；PL-1.1-transpose-chain |
| 7 | **bin 循环 `for kh in T.serial(max_bin_h)`** | `T.serial` 静态界 + 守卫（源同款） | 无（保持 `T.serial` + `if kh<bin_h` 守卫） | ❌ | **理由（tile 级顺序依赖 + 动态边界）**：`kh` 是**累加链的循环携带维**（跨 `kh` 累加到同一 `acc`），且 bin 上界 `bin_h(o_h)` 依赖运行时解码的 `o_h`（**依赖动态 index 的边界**）；TileLang 只接受静态 `T.serial` 边界（`kernels/pool/adaptive_avg_pool2d.py` 源注释明证），无法向量化为单条 op。**但循环体内的累加已向量化**（#1），故该循环是 §6 的 block/tile 级调度循环，非逐元素标量循环。 |
| 8 | **`for kw in range(bin_w(o_w))`（Python 展开）** | 内层标量加 | 展开为多条 `T.vadd`（每条一次 C 向量加） | ✅ | `o_w` 与 `bin_w(o_w)` 在 trace 期是编译期常数（`out_W` 静态、`adaptive_bin` trace 期求值），故用 Python `range` 展开为**向量 op 序列**，不是标量循环 |
| 9 | 单元任务解码 `(n, o_h, c_block)` | — | 无 | ❌ | **理由（block 级索引 / 任务映射计算）**：每单元少量整数 `%`//`//`，非逐元素热点；无对应逐元素向量 API |
| 10 | bin 边界算术 `adaptive_bin` / `bin_size` | — | 无 | ❌ | **理由（host/trace 侧元数据）**：trace 期 Python 整数算术（编译期常数），不在 kernel 运行时逐元素执行 |
| 11 | host 侧 `unsqueeze/squeeze/contiguous` | — | 无 | ❌ | **理由（host 侧元数据）**：只改 shape/stride 视图，符合 `ascend-constraints.md` §4；不在 kernel 内 |

**向量化结论**：逐元素计算（累加器初始化 / 累加 / cast / 归一化 / 归约 / 布局重排）**已全部向量化**（含 §3.3 单元入口的 `acc` 零初始化 → **#0 `T.clear`**），替代方案 API 均有 `docs/` 条目佐证。
保留 **4 处** 循环 / 标量点，理由类别：**#7** tile 级顺序依赖 + 动态边界（循环体内已向量化）/ **#9** block 级任务映射 / **#10** host 侧 trace 元数据 / **#11** host 侧视图操作。
（**#8** 标 `✅`——它是「编译期展开为向量 op 序列」，非逐元素标量循环，故不计入保留的逐元素标量点；与表中「是否替代 = ✅」一致。）
无「以实现简单为由保留标量循环」的情形。**§6 循环结构中不出现逐元素标量循环**（与本结论一致；§6.1 已为 #0 增列「累加器初始化｜整块清除｜`T.clear`」行）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（NCHW 进 / NCHW 出），但**核内布局与 lane 映射是自由变量**。本节枚举「窗口/池化类」必选候选
> （reference：algorithm-research.md Phase 2 第 3 项「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴」）。
> 迁移任务：源的并行轴选择（GPU 逐元素 lane）是**输入而非结论**，对照 §0.3 独立评估。

**轴质量评分维度**：

| 评分项 | 说明 | 候选 1（NCHW / W 轴） | 候选 2（核内 NHWC / C 轴） | 候选 3（host permute / C 轴） |
|--------|------|----------------------|---------------------------|------------------------------|
| 整除性 | 轴长 vs 向量宽度（fp16 ×8） | `W_in`=55/56/57：56 ✓，55/57 ✗（尾 lane 浪费 1.8%/1.4%）；且**窗口宽 bin_w≈9–10 非 8 倍数** | **C 恒为 8 的整数倍**（8/16/64/96/128/2048）✓ | 同候选 2 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | W 轴 3.6%（55→56）；**段长 9–10 → ~12%** | **0%** | 0% |
| 累加链形态 | 该轴作串行累加维（stride≠1）还是向量维（stride=1） | W 轴作段内累加需**变长段归约**（非单一 op）或前缀和（C3，已弃） | C 轴作向量维（stride=1，转置后连续）；空间轴作串行累加维（`kh`/`kw`）——Σ 结构天然 | 同候选 2 |
| repack 代价 | 布局重排代价 | 无 repack，但需变长段处理 | 核内融合转置链（`T.transpose` 二轴交换，PL-1.1 实测 ~5.4µs / 4 次） | **host permute 106–147µs**（PL-1.3，已量化证伪） |
| UB 容量影响 | 布局对驻留缓冲的影响 | 小（row `(W_in)` 级） | `row(·,CH)` + `acc(out_W,CH)` ≈ 15–30 KB（spp CH=64）/ ≈19 KB（resnet C4 CH=64），对照 §4.5 | 同候选 2 |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|----------------|---------|
| 1 | I/O 原生 NCHW | W（行内连续） | 无 | 基线：窗口段长 9–10 非 8 倍数、需变长段归约/前缀和（C3 弃）；`W_in`=55/57 非对齐 | ❌ 量化理由：段长 ~12% 尾 lane 浪费 + 无单一向量 op 可表达变长段归约（`T.reduce_sum` 要求静态形状；变长段需 gather/前缀和），且 `T.reduce_sum` 不支持 bf16（`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1） |
| 2 | **核内 NHWC（融合转置链）** | **C（恒为 8 的整数倍）** | 逐行二轴 `T.transpose(src,dst,[1,0])`（PL-1.1 族，实测 ~µs 级） | 尾 lane 0%；转置代价 ~µs 级（远低于算子本体）；UB 增量 ~15–30 KB | ✅ **选定** |
| 3 | host permute NCHW↔NHWC 进出 | C | 两次 host permute | **106–147µs**（PL-1.3，实测）≥ 多数中等算子本体；且 host 侧连续化会产生新张量（`ascend-constraints.md` §4 约束范围内但无收益） | ❌ 量化理由：repack 代价 > 算子本体流量（估算 ~1.3µs 流量项）两个数量级，净亏 |
| 4 | Cube/implicit-GEMM（0/1 权重矩阵） | Cube N 轴（out_W） | 权重矩阵构造 | `out_W=6/7 < 16` 违 Cube 分形下限；L0C 浪费；权重构造属 host 改内容 | ❌ 见 §1.6.0 R4（附 `ascend-constraints.md` §1/§4 条款） |

**布局决策结论**：选定 **方案 #2：核内布局 = 逐行 NHWC（C 最内连续）、向量化轴 = C、repack 路径 = 核内逐行二轴 `T.transpose`（`[1,0]`）+ 输出侧 `T.transpose([1,0])` 回转**。
该结论已同步至：§3.3 伪代码（`row_src/row_dst/row_f32/acc` 形状与内层向量维 = C）、§4 内存规划（§4.3/§4.5 的 buffer 形状与预算）、§6 循环结构（累加内层向量维 = C）。
弃选方案量化理由见候选矩阵 #1/#3/#4。

**D-3 实验裁决模式（主选 + 备选 + 裁决计划）**：主选与备选的分歧落在窄向量 op 的每-op 发射代价的**外推区间**——`CONST-vector-launch-overhead` 只给了 bn=512 大 op 的 0.5µs/op 实测值，本设计 ≤128 元素的窄 op 未逐点实测；判定裕度（`CH=64` 少一半 op 但核数少一半）落在该外推的不确定区间内，故仍保留备选与裁决计划。**但「更宽 C 向量更优」这一方向已有实测证据**：pattern-library `layout.md` **PL-1.4** 直接实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs——更宽 C 向量 + 更少 block 胜过更高空间并行度」，与 `CONST-vector-launch-overhead`「宽块 op 恒优于窄块 op 翻倍」方向一致；**主选 `CH=64` 恰为 PL-1.4 已实测范围（CH16/32/64）内的最优档**，故主选以实测条目为主证据，不属「未实证的纸面单选」：

- **主选（进 Stage 3）**：`CH = min(C, 64)`（8 的整数倍），逻辑单元 = `(n, o_h, c_block)`，`num_logical > vector_cores` 时 persistent（见 §5.5）。
  代表值：`spp` CH=64 → 24 单元；`nondiv` CH=64 → 14 单元；`resnet` CH=64 → 256 单元（persistent，40 核 × 7 任务）。
  **依据**：PL-1.4 实测 CH64 为该池化族最优档（21.6µs，优于 CH32/CH16）；与本设计 C 轴向量化形态同族。
- **备选（≤1 个，结构完整，可直接实现）**：**C4 特化 + `CH = min(C, 8)`（最小 CH）**，即把 op 数放大 8×、核数放大 8×（`spp` → 192 单元）。
  这是「发射代价主导 vs 吞吐主导」的另一端；buffer 形状与主选相同仅 CH 变小（UB 预算同步缩小），分核三要素按 `num_logical=192` 重算（persistent 40 核 × 5 任务）。
  （备选存在的意义**仅限于**：验证比 PL-1.4 已测范围（CH16/32/64）**更窄**的 CH=8 是否使趋势翻转——PL-1.4 未覆盖 CH<16，该点仍属未实测外推；若窄 op 发射代价极低（吞吐主导）则更小 CH 可能更快，但方向证据偏向主选。）
- **裁决计划**：代表 shape = `spp-6x6`、`nondiv-7x7`（各 1 组）；指标 = msprof op Task Duration；测量未知常数 = 窄向量 op 的每-op 代价（用 `perf_opt` 下两种 CH 的实测反解）；
  判定阈值 = **备选 latency < 主选 × (1 − 0.10) 才翻转**（10% 稳健性门槛，须显著优于 PL-1.4 已测的 CH64 优势）；裁决结果回写路径 = Stage 4 实测数据经附录补记固化 §1.6.3 判定依据，若翻转则按 `perf-feedback.md` 受控路由回写 DESIGN.md。
- **shape 分派评估**：本 kernel 由 `functools.lru_cache` 按 `(n,c,h,w,out_h,out_w,dtype)` 特化，**支持按 shape 分派**主选/备选——
  裁决计划须评估 `resnet-global` 等大 N 形状走大 CH、`spp/nondiv` 走小 CH 的**按负载分派**（而非全局二选一）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（迁移 prompt 指定；`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`）。

### 2.2 选型理由

- 算子是**纯 Vector**（无 matmul / 无 Cube），仅需逐元素向量运算 + 归约 + 布局重排——正是 Developer 模式的目标域（`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`、`.../elementwise/lerp_tensor/...` 均为 Developer）。
- 无跨引擎流水、无手动 L1/L0 分级需求（`T.alloc_ub` / `T.alloc_shared` 直映 UB）；编译器管理 UB 分配与自动同步（§7）。
- 迁移 prompt 明确要求 developer 模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 环境变量 | `TILELANG_ASCEND_MODE=Developer` |
| 内存分配 | `T.alloc_ub(...)`（GM→UB 路径），可选 `T.alloc_shared` 同义（Developer 下均映射 UB） |
| 计算方式 | v-prefix 向量 API（`T.vcast/T.vadd/T.vmul/T.reduce_sum`）+ `T.transpose` + `T.copy` |
| 同步 | 编译器自动同步（§7）；无手动 `T.set_flag/T.wait_flag` |
| Kernel 维度 | 一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(ih0,ih1) ← bin_H(o_h)`；`(iw0,iw1) ← bin_W(o_w)` | 编译期（`o_w`）/ 运行时（`o_h`）bin 边界；`bin_h=ih1-ih0`、`bin_w=iw1-iw0` |
| 2 | `row_src[c, iw] ← x[n, c0+c, ih0+kh, iw]` | 行片搬运 `(CH, W_in)` |
| 3 | `row_dst[iw, c] ← row_src[c, iw]` | 核内二轴转置 → C 最内 |
| 4 | `row_f32[iw, c] ← cast_f32(row_dst[iw, c])` | 整 tile fp32 上转（`rint`） |
| 5 | `acc[o_w, c] ← acc[o_w, c] + row_f32[iw0(o_w)+kw, c]` | C 轴向量累加（`kh` serial + `kw` 展开） |
| 6 | `acc[o_w, c] ← acc[o_w, c] · (1/(bin_h·bin_w))` | 归一化（§1.6.1 **C1 已采纳**：fp32 倒数 `inv` + 向量乘） |
| 7 | `acc_t[c, o_w] ← cast_dtype(acc[o_w, c])` | 转置回转 + 单次舍入 |
| 8 | `out[n, c0+c, o_h, o_w] ← acc_t[c, o_w]` | 行片写回 `(CH, out_W)` |
| C4 | `out[n,c,0,0] ← (Σ_{ih,iw} x[n,c,ih,iw]) · 1/(H_in·W_in)` | 退化：单次 `reduce_sum` |
| C4b | `out[n,c,ih,iw] ← x[n,c,ih,iw]` | 退化：恒等拷贝 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1b | 累加器零初始化 | `T.clear(acc)` | 整 buffer 清零（§1.6.2 #0） | Developer |
| 2 | 行片 GM→UB | `T.copy(src, dst)` | `src = x[n, c0:c0+CH, ih, 0:W_in]`，`dst = row_src[0:CH, 0:W_in]` | Developer |
| 3 | 二轴转置 | `T.transpose(src, dst, permutation)` | `permutation=[1,0]`（`docs/Tilelang.language/创建操作/T.transpose.md`） | Developer |
| 4 | fp32 上转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"` | Developer |
| 5 | C 轴累加 | `T.vadd(src0, src1, dst)` | `src0=acc[o_w:o_w+1,0:CH]`，`src1=row_f32[iw:iw+1,0:CH]`，`dst=acc[o_w:o_w+1,0:CH]`（原地） | Developer |
| 6 | 归一化 | `T.vmul(src1, src2, dst)` | `src2 = inv`（fp32 标量倒数，§1.6.1 C1；在 `T.serial` 循环内按 `o_w` 求值） | Developer |
| 7 | dtype 回转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"` | Developer |
| 8 | 行片 UB→GM | `T.copy(src, dst)` | `src = acc_t[0:CH, 0:out_W]`，`dst = out[n, c0:c0+CH, o_h, 0:out_W]` | Developer |
| C4 | 全平面归约 | `T.reduce_sum(buffer, out, dim)` | `dim=1`（`(CH, H*W) → (CH,1)`），`clear=True` | Developer |
| 边界差 | 整数 bin 边界 | Python `adaptive_bin`（host/trace） | 复用源 `common.py` 逻辑 | host |

### 3.3 计算伪代码

> 注（迁移模板要求）：`T.Tensor` 的数据类型以**第二个位置参数**传入，不用 `dtype=` 关键字。

```python
# 迁移决策后的 NPU 算法伪代码（Developer 模式；C 轴向量化 + 核内转置）
@tilelang.jit(out_idx=[1], target="npuir")
def kernel(block_m_unused_drop, CH):
    num_cblk = (C + CH - 1) // CH                            # 通道块数（编译期）
    # 逻辑单元数按编译期分支取——单元粒度决定 num_logical（见 §5.2 / §5.5 ①）：
    #   一般网格  单元 = (n, o_h, c_block)  -> NUM_LOGICAL = N*out_H*num_cblk
    #   C4 全局   单元 = (n, c_block)        -> NUM_LOGICAL = N*num_cblk（out_H==1，与通用式同值）
    #   C4b 恒等  单元 = (n, c_block)        -> NUM_LOGICAL = N*num_cblk（整平面；**不等于** N*out_H*num_cblk）
    NUM_LOGICAL = N * num_cblk if (IS_IDENTITY or IS_GLOBAL) else N * out_H * num_cblk
    num_kernels  = min(NUM_LOGICAL, VECTOR_CORES)            # 物理向量核（见 §5.5）
    num_local    = (NUM_LOGICAL + num_kernels - 1) // num_kernels   # 静态核内任务数（Python 整数，编译期常量）

    @T.prim_func
    def main(
        x:   T.Tensor((N, C, H_in, W_in), dtype),            # dtype 位置参数（非 dtype=）
        out: T.Tensor((N, C, out_H, out_W), dtype),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # ---- 退化网格结构特化（§0.6 R4 / §1.6.1 C4/C4b）----
            if IS_IDENTITY:                                   # out_H==H_in and out_W==W_in（编译期）
                for t in T.serial(num_local):                 # NUM_LOGICAL = N*num_cblk（整平面单元）
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:
                        n  = uid // num_cblk                     # 与 NUM_LOGICAL 同粒度：n∈[0,N)，无越界
                        cb = uid %  num_cblk
                        T.copy(x[n, cb*CH : cb*CH+CH, 0:H_in, 0:W_in], stage_ub)
                        T.copy(stage_ub, out[n, cb*CH : cb*CH+CH, 0:H_in, 0:W_in])
            elif IS_GLOBAL:                                   # out_H==1 and out_W==1
                for t in T.serial(num_local):                 # NUM_LOGICAL = N*num_cblk
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:
                        n  = uid // num_cblk
                        cb = uid %  num_cblk
                        T.copy(x[n, cb*CH:cb*CH+CH, 0:H_in*W_in], flat_ub)   # (CH, H*W)
                        T.vcast(flat_ub, flat_f32, round_mode="rint")
                        T.reduce_sum(flat_f32, gacc, dim=1)                  # (CH,1)
                        T.vmul(gacc, 1.0/(H_in*W_in), gacc)                  # 编译期常数倒数
                        T.vcast(gacc, gout_ub, round_mode="rint")
                        T.copy(gout_ub, out[n, cb*CH:cb*CH+CH, 0, 0])
            else:
                # ---- 一般网格 ----
                acc = T.alloc_ub((out_W, CH), "float32")      # 向量轴 = C（§1.6.3 决策）
                row_src = T.alloc_ub((CH, W_in), dtype)
                row_dst = T.alloc_ub((W_in, CH), dtype)
                row_f32 = T.alloc_ub((W_in, CH), "float32")
                acc_t   = T.alloc_ub((CH, out_W), dtype)

                for t in T.serial(num_local):                  # NUM_LOGICAL = N*out_H*num_cblk
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:
                        n   = uid // (out_H * num_cblk)
                        r   = uid %  (out_H * num_cblk)
                        o_h = r // num_cblk
                        cb  = r %  num_cblk
                        ih0, ih1 = adaptive_bin(o_h, H_in, out_H)   # 运行时标量
                        bh = ih1 - ih0
                        T.clear(acc)                                # 整块零初始化（§1.6.2 #0；非逐元素标量循环）
                        for kh in T.serial(MAX_BIN_H):              # 静态界 + 守卫（保留源手段）
                            if kh < bh:
                                T.copy(x[n, cb*CH:cb*CH+CH, ih0+kh, 0:W_in], row_src)
                                T.transpose(row_src, row_dst, permutation=[1, 0])
                                T.vcast(row_dst, row_f32, round_mode="rint")
                                for o_w in range(out_W):            # 编译期展开
                                    iw0, iw1 = adaptive_bin(o_w, W_in, out_W)  # 编译期常数
                                    for kw in range(iw1 - iw0):     # 编译期展开
                                        T.vadd(acc[o_w:o_w+1, 0:CH],
                                               row_f32[iw0+kw:iw0+kw+1, 0:CH],
                                               acc[o_w:o_w+1, 0:CH])   # C 轴向量累加
                        for o_w in range(out_W):                    # 归一化（编译期常数倒数）
                            iw0, iw1 = adaptive_bin(o_w, W_in, out_W)
                            inv = 1.0 / (bh * (iw1 - iw0))          # 运行时标量
                            T.vmul(acc[o_w:o_w+1, 0:CH],
                                   T.cast(inv, "float32"),
                                   acc[o_w:o_w+1, 0:CH])
                        T.vcast(acc, acc_t, round_mode="rint")      # (out_W,CH) -> (CH,out_W) 需先转置
                        # 注：实际实现次序为 acc -> acc16 中转 -> transpose -> 写回；细节交 Stage 3
                        T.copy(acc_t_sliced, out[n, cb*CH:cb*CH+CH, o_h, 0:out_W])
```
（伪代码为**决策示意**，Stage 3 可细化局部次序；但「向量轴 = C」「核内二轴转置」「静态 bin 界 + 守卫」「C4/C4b 特化」是冻结决策。）

### 3.4 API 可行性确认

| API | 来源确认 | 是否验证 |
|-----|---------|---------|
| `T.Kernel(num, is_npu=True) as (cid, _)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3（NPU ≤2 维） | ✓ 现有 examples 广泛使用 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；`lerp_tensor` developer kernel | ✓ |
| `T.clear`（buffer 全零初始化，1~5 维） | `docs/Tilelang.language/内存操作/T.clear.md` §2.2（fp16/fp32/bf16 ✓） | ✓（降级为 `hivm::VBrcOp`；§1.6.2 #0） |
| `T.copy`（切片 GM↔UB） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2 | ✓ |
| `T.transpose`（二轴交换，fp16/bf16/fp32） | `docs/Tilelang.language/创建操作/T.transpose.md` §2.3 | ✓（`argreduce` perf_opt 用 `[1,0]`） |
| `T.vcast`（f16/bf16→f32 `rint`；f32→f16/bf16 `rint`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✓ |
| `T.vadd`（fp16/fp32；`dst` 可原地） | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.1/§2.3 | ✓ |
| `T.vmul`（src2 标量） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.3 | ✓ |
| `T.reduce_sum`（fp16/fp32，`clear`） | `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1/§2.1 | ✓ |

> **dtype 约束提醒**：`T.vadd/T.vsub/T.vmul/T.reduce_sum` 的 dtype 表均**不含 bf16**（`docs/...` 各 §2.2.1）。故 **bf16 输入必须经 `T.vcast` 上转到 fp32 后计算**——这与「fp32 累加」语义一致（§0.6 R2），无需额外代价。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel（`T.Kernel` 一维/最多二维 block） | 否（用一维 `T.Kernel(num_kernels)`；任务维度在 kernel 内解码） | 一维网格；`(n,o_h,c_block)` 由 `uid = t*num_kernels + cid` 解码 |
| GPU 专用 API | 否（全部 v-prefix / `T.copy` / `T.transpose`，均为 NPU API） | — |
| GEMM 非整除 / L0C 溢出 | 否（无 Cube 计算，`L0C` 不适用） | — |
| 分核策略 | 见 §5.5（三要素齐全） | persistent grid-stride |
| host 侧禁止改输入内容 | 否（host 只做 `unsqueeze/squeeze/contiguous` 视图操作） | `ascend-constraints.md` §4 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 / 并行模型 | `T.Kernel(ceildiv(total,block_m), threads)` + `T.Parallel(block_m)`（线程 lane 级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1（`(n,o_h,c_block)` 单元 + C 向量 lane） |
| 累加单元 | 每线程标量寄存器累加 | UB fp32 向量累加器（C 轴） | §0.6 R2 |
| 数据搬运 | 逐元素 GM gather | UB tile 搬运 + 核内二轴转置 | §0.6 R3 |
| `threads` 参数 | factory `(block_m, threads)`、`default_config={"block_m":256,"threads":256}` | **移除 `threads`**（NPU 无 CUDA 线程） | 脚手架已按 K9 移除；factory 签名改为 `(CH)` |
| 退化网格 | 无特化 | C4/C4b 结构特化 | §0.6 R4 / §1.6.1 |
| autotune | `autotune_configs` | 移除（K8），启发式 config | 脚手架已移除 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 高（结构模板） | Developer 模式一维 `T.Kernel(num_kernels,is_npu=True)` + persistent grid-stride `T.serial(num_local)` + `if block_id<num_logical` 守卫；`T.alloc_ub`；v-prefix 链；尾块 `gate` |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | 高（归约模板） | `T.reduce_sum/reduce_max` 用法、`T.vsub(vadd/vexp)`、尾部 `T.min` 切片、fp32 中转 |
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/perf_opt/_argreduce_kernel.py` | 中（转置模板） | 核内二轴转置 `T.transpose(src,dst,[1,0])`（PL-1.1 族）实测用法 |
| `examples/elementwise/vec_add_2d_multi_buffer.py` | 中（拷贝/多 buffer） | GM↔UB `T.copy` 切片与 multi-buffer 语义 |
| `examples/TileOPs/tests/ops/test_pool.py` / `examples/TileOPs/tileops/manifest/pool.yaml` | 高（规格/golden） | workload 与 `ref_program = F.adaptive_avg_pool2d` |
| pattern-library `layout.md` PL-1.1/PL-1.2/PL-1.3/PL-1.4/PL-1.5 | 高（池化类实测模式） | C 轴切片累加、核内融合转置链、host permute 证伪、tiling 启发式 |

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
| `row_f32` | `(W_in, CH)` | float32 | UB | fp32 上转累加源 |
| `acc` | `(out_W, CH)` | float32 | UB | C 轴向量累加器（向量轴 = C） |
| `acc_t` | `(CH, out_W)` | dtype | UB | 输出转置回转（NCHW 写回） |
| `flat_ub`/`flat_f32`（C4） | `(CH, H_in*W_in)` | dtype / float32 | UB | 全局池化退化路径 |
| `gacc`（C4） | `(CH, 1)` | float32 | UB | 全平面归约输出 |
| `stage_ub`（C4b） | `(CH, H_in*W_in)` | dtype | UB | 恒等退化路径中转（**单元 = 整张 `(n, c_block)` 平面**，故一次整平面 `T.copy`；对应 `NUM_LOGICAL = N*num_cblk`，见 §5.2/§5.5①） |

### 4.4 内存搬运路径

```
【一般网格】
GM[x] --T.copy 行片(CH,W_in)--> UB[row_src] --T.transpose[1,0]--> UB[row_dst] --T.vcast(rint)--> UB[row_f32]
UB[row_f32] --T.vadd(C 轴切片累加, kh serial × o_w/kw 展开)--> UB[acc (out_W,CH) f32]
UB[acc] --T.vmul(1/bin_size) / T.vcast(rint)--> UB[acc_t (CH,out_W)] --T.copy 行片(CH,out_W)--> GM[out]
【C4 全局池化】GM[x] --T.copy 连续块(CH,H*W)--> UB[flat_ub] --vcast--> UB[flat_f32] --reduce_sum(dim=1)--> UB[gacc] --vmul--> vcast --> GM[out]
【C4b 恒等】GM[x] --T.copy 整平面(CH,H*W)--> UB[stage_ub] --T.copy--> GM[out]（单元 = 整张 (n,c_block) 平面，NUM_LOGICAL=N*num_cblk）
（无 GM 中间往返、无 Cube/L1/L0C 路径）
```

### 4.5 UB 内存预算

| Buffer | Shape（代表 workload） | dtype | 大小 (Bytes) |
|--------|------------------------|-------|-------------|
| `row_src` | `(64, 56)`（spp CH=64） | fp16 | 7 168 |
| `row_dst` | `(56, 64)` | fp16 | 7 168 |
| `row_f32` | `(56, 64)` | fp32 | 14 336 |
| `acc` | `(6, 64)` | fp32 | 1 536 |
| `acc_t` | `(64, 6)` | fp16 | 768 |
| **总计（spp CH=64）** | | | **40 976 B ≈ 41 KB** / 196 608 (192 KB, `CONST-capacity-910B2C` A2/A3) ✓ |
| C4 `flat_ub`+`flat_f32`+`gacc`（resnet CH=64，与 §5.2 `CH=min(C,64)` 一致） | `(64,49)` fp16 + `(64,49)` f32 + `(64,1)` f32 | — | 6 272 + 12 544 + 256 = **19 072 B ≈ 19 KB** ✓ |
| C4b `stage_ub`（identity CH≤64） | `(64, 99)` fp16 | — | 12.7 KB ✓ |

> **UB 膨胀口径两档**（`CONST-capacity-910B2C`，entry 见 `constants.md`；条目正文给出两档系数，`§7.3` 默认不关闭 multi-buffer）：
> - **默认档（auto-multi-buffer 开，本设计默认，§7.3）**：膨胀 **~1.7x** → spp 41 KB × 1.7 ≈ **70 KB**；resnet C4 19 KB × 1.7 ≈ **33 KB**；C4b 12.7 KB × 1.7 ≈ 22 KB。**均 ≪ 192 KB**。
> - **关闭档（`--enable-auto-multi-buffer=false`）**：膨胀 1.10–1.12（2026-09-15 重校 6797758）→ spp ≈ 46 KB；resnet C4 ≈ 21 KB。同样充裕。
> **两档均在容量内**，故 §7.3「溢出时再评估关 multi-buffer」以本两档为可比基线。

### 4.6 动态轴定义

**无动态轴**：`N, C, H_in, W_in, out_H, out_W` 全部在 factory 构建期（trace 期）确定，kernel 由 `functools.lru_cache` 按 shape 特化。
（`o_h`/`bin_h` 在 kernel 内是运行时标量，但它们是**由编译期 shape 推出的运行时值**，不是动态轴。）

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3", "-DENABLE_BF16"])
def _adaptive_avg_pool2d_func(CH: int) -> ...   # 去掉源 GPU 的 threads 参数（K9）
```
---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。

**判定依据**：算子仅含逐元素向量运算（`T.vadd`/`T.vmul`）、归约（`T.reduce_sum`）、cast（`T.vcast`）与布局重排（`T.transpose`），
无 `T.gemm` / matmul → 不需要 L1 + L0A/L0B/L0C（§0.6 R4 已弃 Cube 路径）。故核数按**纯 Vector 算子翻倍**规则取物理核数 ×2（§5.5）。

### 5.2 Block 划分

本算子的分块维度不是传统 GEMM 的 `(M,N)`，而是 **(通道块 CH) × (输出行 o_h)** 的任务维度：

```python
CH = min(C, 64)                    # 通道向量块（8 的整数倍）；主选值，D-3 备选见 §1.6.3
num_cblk     = ceil(C / CH)        # 通道块数（C=8/64/128/2048 → 1/1/2/32）
# 逻辑单元数按编译期分支取（**单元粒度是 num_logical 的唯一定义源**，见 §5.5 ①）：
#   一般网格  单元 = (n, o_h, c_block)  ->  NUM_LOGICAL = N * out_H * num_cblk
#   C4 全局   单元 = (n, c_block)        ->  NUM_LOGICAL = N * num_cblk（out_H==1，与通用式同值）
#   C4b 恒等  单元 = (n, c_block)        ->  NUM_LOGICAL = N * num_cblk（整平面；≠ N*out_H*num_cblk，见下表末行）
NUM_LOGICAL = N * num_cblk if (out_H == 1 and out_W == 1) or (out_H == H_in and out_W == W_in) \
              else N * out_H * num_cblk
```

- **为什么 `num_logical` 按分支取**：`num_logical` 必须与「一个逻辑单元处理多少数据」**严格同粒度**，否则任务解码越界。一般网格的单元为 `(n, o_h, c_block)`，退化路径（C4/C4b）的单元为 `(n, c_block)` **整张平面**，两者所需核数不同；C4 因 `out_H==1` 两种口径同值，**唯一会分叉的是 C4b（`out_H==H_in>1`）**：`(1,8,9,11)` 恒等 case 通用式给 9，而整平面单元只需 1 —— 若沿用通用式 9 配 §3.3 解码 `n = uid // num_cblk` 将产生 `n=0..8`（`N=1`）**越界读/写**（L0-5 直接覆盖该路径）。本版**采用整平面单元口径** `NUM_LOGICAL = N*num_cblk`，与 §3.3 解码、§4.3 `stage_ub=(CH, H_in*W_in)`、§5.5 ① / §6.2 全文唯一自洽。

- **CH = 64 的选择理由**：① C 轴为向量轴，CH 需为 8（fp16 向量宽）的整数倍——64 为 8 的整数倍；② UB 预算充裕（spp 41 KB，§4.5）；
  ③ 与 §5.5 的分核方案配合使 `num_logical` 落在物理核数附近（spp 24 / nondiv 14 / resnet 256）。
- **`max_bin_h = max_adaptive_bin_extent(H_in, out_H)`、`max_bin_w`**：bin 循环的静态上界（编译期常数）；`bin_h/bin_w` 为运行时边界差。

**代表性 shape 的分块结果**：

| workload | N | C | H_in×W_in | out | CH | num_cblk | NUM_LOGICAL（按单元粒度） |
|----------|---|---|-----------|-----|----|----------|-------------|
| smoke | 1 | 8 | 4×4 | 2×2 | 8 | 1 | 2（一般） |
| resnet-global | 8 | 2048 | 7×7 | 1×1 | 64 | 32 | **256**（C4 特化，`N*num_cblk` = 通用式同值） |
| spp-6x6 | 2 | 128 | 56×56 | 6×6 | 64 | 2 | 24（一般） |
| nondiv-7x7 | 2 | 64 | 55×57 | 7×7 | 64 | 1 | 14（一般） |
| test `(2,64,56,56)`→6×6 | 2 | 64 | 56×56 | 6×6 | 64 | 1 | 12（一般） |
| test `(1,96,28,30)`→7×5 | 1 | 96 | 28×30 | 7×5 | 64 | 2 | 14（一般） |
| test `(1,128,55,57)`→7×7 | 1 | 128 | 55×57 | 7×7 | 64 | 2 | 14（一般） |
| test `(1,16,11,13)`→`(None,5)` | 1 | 16 | 11×13 | 11×5 | 16 | 1 | 11（一般；H 恒等但 W 非恒等，**非** C4b） |
| test `(1,16,10,10)`→4 | 1 | 16 | 10×10 | 4×4 | 16 | 1 | 4（一般） |
| test `(1,8,9,11)`→None | 1 | 8 | 9×11 | 9×11 | 8 | 1 | **1**（C4b 恒等：单元 = 整平面 `(n,c_block)`，`N*num_cblk=1`，**非**通用式 9） |

> 末行是全文唯一 `NUM_LOGICAL` 两种口径分叉处：C4b 恒等路径的单元是**整张 `(n, c_block)` 平面**（§4.3 `stage_ub=(CH, H_in*W_in)` 一次整平面 `T.copy`；§3.3 两次整平面 `T.copy(..., 0:H_in,0:W_in)`），故按 `N*num_cblk` 计（=**1**），解码 `n = uid // num_cblk` 与之一致（`n∈[0,1)`）；此即消除「通用式 9 + 整平面单元 → `n=0..8` 越界」的收敛结果（§5.5 ① / §11.3）。

### 5.3 约束分析

- **对齐约束**：向量轴 C 恒为 8 的整数倍（fp16/bf16 向量宽 8）；`CH=64` 为 8 倍数 ✓。`W_in`（55/56/57）仅作转置后的 row 长度，不作向量轴，不要求整除 ✓。
- **UB 容量**：spp CH=64 总 41 KB（默认 auto-multi-buffer ×1.7 → ≈70 KB；关闭档 ×1.12 → 46 KB）< 192 KB ✓（§4.5）。resnet C4 CH=64 → 19 KB（×1.7 ≈ 33 KB）✓。
- **L0 容量**：无 Cube 计算，不适用 ✓。
- **非整除**：本算子**不存在 GEMM 式整除问题**——所有 shape（`total_output`、`W_in`）无需是 block 的整数倍：
  任务维度用 persistent `T.serial` + `if uid < NUM_LOGICAL` 守卫（尾块丢弃），row 长度用静态 `W_in`。bin 非整除（重叠）由整数 `adaptive_bin` 精确处理（语义本来就要求）。

### 5.4 注意事项

- **尾单元**：`NUM_LOGICAL` 非 `num_kernels` 整数倍时（resnet: 256 / 40 = 6.4），核内 `T.serial(num_local)` 的 `if uid < NUM_LOGICAL` 守卫丢弃越界任务（静态上界 `num_local = ceil(NUM_LOGICAL/num_kernels)`）——同 `lerp_tensor` 的 valid 模式。
  **退化路径（C4/C4b）的 `NUM_LOGICAL` 按其整平面单元粒度取**（`N*num_cblk`，见 §5.5 ①）：`(1,8,9,11)` 恒等 case `NUM_LOGICAL=1`，`num_kernels=1`、`num_local=1`，守卫恒真、无越界。
- **`kh` 越界守卫**：`max_bin_h` 是 bin 的**最宽**上界；对较窄的 `o_h`，`if kh < bh` 跳过越界行（不读取 `>= ih1` 的输入行）——与源算法守卫同构（§0.4 #1）。
- **零输出保护**：`out_H/out_W > 0` 由 op 层保证（`_normalize_output_size` 拒绝 `<=0`）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**（**单元粒度 = `num_logical` 的唯一定义源**）：
  - **一般网格**（`out_H>1` 或 `out_W>1` 且非恒等）：单元 = `(n, o_h, c_block)` → `num_logical = N × out_H × ceil(C / CH)`（pure-Vector 无 `block_N` 维度，故只有通道块维 × 输出行维）。
  - **退化路径 C4 全局**（`out_H==out_W==1`）：单元 = 整张 `(n, c_block)` 平面 → `num_logical = N × ceil(C/CH)`。因 `out_H==1`，该值与通用式**同值**（resnet `8 × 1 × 32 = 256`），无分叉。
  - **退化路径 C4b 恒等**（`out_H==H_in and out_W==W_in`，且 `H_in>1`）：单元 = 整张 `(n, c_block)` 平面（§4.3 `stage_ub=(CH, H_in*W_in)` 一次整平面 `T.copy`）→ `num_logical = N × ceil(C/CH)`，**显式不同于**通用式 `N × out_H × ceil(C/CH)`（`(1,8,9,11)` case = **1**，通用式为 9）。
    依据：若沿用通用式 9 而单元仍为整平面，§3.3 解码 `n = uid // num_cblk` 在 `N=1` 下会产生 `n=0..8` → **越界读/写**；故 C4b 路径**必须**取 `N × ceil(C/CH)`（本版采用的口径，见 §5.2 表末行）。
  代表值见 §5.2 表：smoke 2 / resnet 256（C4，同值）/ spp 24 / nondiv 14 / 恒等 `(1,8,9,11)` **1**。
  下文以 `NUM_LOGICAL` 记该「按分支粒度取值的逻辑核数」（§3.3/§5.2/§6.2 同）。

- **② 物理核数依据（实查）**：
  本设计环境实际执行查询：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()
  # 实查返回值 = 20   (设备 Ascend910B3，查得于 2026-10-13，本设计会话)
  ```
  **纯 Vector 算子核数翻倍**：`vector_cores = 20 × 2 = 40`。
  （设备名核对：`npu-smi info` 显示 `910B3`；不采用 pattern-library `CONST-aicore-910B2C`=24 的 910B2C 值——**本设备为 910B3，必须实查**，故重新查询得 20。
  `tilelang/utils/npu_utils.cpp:127` 确认底层为 `rtGetAiCoreCount`。）

- **③ 规模判定与分核方案**（三选一，逐 workload 判定）：

  | workload | NUM_LOGICAL | 物理核 | 判定 | 方案 |
  |----------|-------------|--------|------|------|
  | smoke | 2 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = min(2,40) = 2`，2 个单元各占 1 核；依据：`NUM_LOGICAL(2) ≤ vector_cores(40)`，无核超发、无负载不均 |
  | spp / nondiv / 多数测试 | 12–24 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = NUM_LOGICAL`，每核 1 个单元 |
  | `(1,8,9,11)` 恒等（**C4b**） | **1** | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = min(1,40) = 1`，1 核执行 1 个整平面单元（`NUM_LOGICAL=N*num_cblk=1`）；**消除**了「通用式 9 + 整平面单元 → `n=uid//num_cblk∈0..8` 越界」 |
  | resnet-global | 256 | 40 | **极大规模**（无法通过调小 block 缩减核数到 ≤40，且核数非 40 的整数倍） | **核内串行（persistent）**：固定启动 `num_kernels = min(NUM_LOGICAL, vector_cores) = 40`；核内 `for t in T.serial(num_local)` 串行处理多个逻辑单元，`num_local = ceil(256/40) = 7`（**静态值**）；`uid = t*num_kernels + cid`，`if uid < NUM_LOGICAL` 守卫 —— 摊薄核启动开销、消除 256-核超发 |

  **通用规则（factory 级）**：`NUM_LOGICAL` 按分支粒度取（见 ①），`num_kernels = min(NUM_LOGICAL, vector_cores)`，`num_local = ceil(NUM_LOGICAL / num_kernels)`（**Python 整数编译期常量**，非 `T.ceildiv` 表达式）。
  当 `NUM_LOGICAL ≤ vector_cores` → `num_local = 1`，退化为「每核 1 单元 = 无需适配」；当 `NUM_LOGICAL > vector_cores` → 核内串行（persistent）。
  **核内串行循环边界 `num_local` 为静态值**（`NUM_LOGICAL`/`num_kernels` 均为编译期常数）——满足 `core-split-strategy.md` §1 要素③硬要求。
  > **不做空核的保证**：因 `num_kernels = min(NUM_LOGICAL, vector_cores)`，故 `num_kernels ≤ NUM_LOGICAL`，每个物理核的首个任务 `uid = cid < NUM_LOGICAL` **恒有效**，不存在「空核执行越界任务」形态（对照 pattern-library 草稿 VP-2026-0058 记载的 `T.ceildiv` 负数域 empty-core 越界陷阱：本设计的 `num_local` 为 Python int 编译期常量、且核数不超过 `NUM_LOGICAL`，两条件均不触发该陷阱）。

  > 分核增益预期：resnet 若按 GPU 式超发 256 核，NPU 运行时会将 216 核串行排入 40 核 → 引入核启动开销；persistent 化后固定 40 核、每核 7 个静态任务，负载均衡（256 = 40×6 + 16，16 核 7 任务 / 24 核 6 任务，最大差 1 个任务）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block/tile 级调度循环。**不出现逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑单元（task） | block 级并行 | `T.Kernel(num_kernels)` | 每核处理 1 个或多个 `(n,o_h,c_block)` 单元 |
| 累加器初始化 | **整块清除**（无循环） | `T.clear(acc)` | 单元入口整 buffer 清零（§1.6.2 #0）；**非**逐元素标量循环 |
| 核内多任务 | 迭代（persistent） | `T.serial(num_local)`（静态界）+ `if uid<NUM_LOGICAL` | 极大规模时摊薄核启动开销（§5.5）；静态边界 |
| bin 的 H 方向 | 迭代（累加链） | `T.serial(MAX_BIN_H)` + `if kh < bh` | tile 级顺序依赖 + 动态边界（§1.6.2 #7）；循环体内是向量 op |
| bin 的 W 方向 | **编译期展开** | Python `range(bin_w(o_w))` | `o_w`/`bin_w` 为 trace 期常数；展开为**向量 op 序列**（§1.6.2 #8） |
| 输出列 `o_w` | **编译期展开** | Python `range(out_W)` | `out_W` 静态且小（≤7）→ 展开避免运行时索引 |
| 元素级（C 轴） | 向量化 | `T.vadd`（向量轴 = C，§1.6.3 决策） | C 轴 lane 并行，无逐元素循环 |
| 退化路径 | 单次归约 | `T.reduce_sum`（C4） | 全局池化无 bin 循环 |

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
                            T.vadd(...)                      # C 轴向量累加
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- 单元内只有一条 `GM→UB→计算→UB→GM` 单阶段流，无多阶段 K 循环可流水；`kh` 循环的次数少（`max_bin_h` 1–10），双缓冲收益有限。
- Developer 模式下编译器可自动 multi-buffer；若 Stage 4 实测显示 MTE2 未与 Vector 重叠，可在 `kh` 循环上探索 `T.Pipelined(num_stages=2)`（列入 Stage 4 候选，不改主结构）。
- C4 退化路径为「1 次搬运 + 1 次归约」，无流水空间。

### 6.4 尾块处理

- **任务尾块**：`NUM_LOGICAL` 非 `num_kernels` 整数倍 → `if uid < NUM_LOGICAL` 丢弃越界任务（不读不写）。退化路径（C4/C4b）`NUM_LOGICAL` 按整平面单元粒度取（`N*num_cblk`，§5.5 ①），与 `n = uid // num_cblk` 解码同粒度。
- **bin 尾块（H 方向）**：`kh >= bh` → 守卫跳过，不读入界外行。
- **bin 尾块（W 方向）**：`kw` 用编译期 `range(bin_w(o_w))`，**精确**覆盖 bin，无越界、无掩码需求。
- **无 GEMM 式整除要求**（§5.3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步（Developer 模式）**。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器在 `T.copy`（MTE2 搬运）与 v-prefix 向量 op（Vector 单元）之间自动插入同步标志，
`T.transpose`（VTransposeOp）与 `T.reduce_sum`（VReduceOp）亦由编译器编排。本算子无跨核通信、无 Cube/Vector 跨引擎依赖，故：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| （无） | — | Developer 自动同步；本算子单引擎（Vector）、单阶段，无 `sync_block_set/wait` 需求 |

### 7.3 pass_configs 配置

`@tilelang.jit(out_idx=[1], target="npuir", compile_flags=["-O3","-DENABLE_BF16"])`；不设置自定义 `pass_configs`（与 `lerp_tensor` / `logsumexp` 先例一致）。
若 Stage 3/4 出现 UB 膨胀导致溢出，可评估 `--enable-auto-multi-buffer=false`（`CONST-capacity-910B2C` 2026-09-16 反例数据点）。

---

## 8. 验证方案

### 8.1 Golden 函数

> **迁移任务**：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现 / 测试基准），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_adaptive_avg_pool2d(x: torch.Tensor, output_size) -> torch.Tensor:
    """PyTorch reference for adaptive_avg_pool2d, per §0.1 semantics.

    Directly uses the source-repo test benchmark
    (examples/TileOPs/tests/ops/test_pool.py: ref_program).
    Accepts 3D CHW or 4D NCHW; None entries keep the input size.
    Independent of the NPU algorithm (no transpose / no C-vectorization).
    """
    size = (None, None) if output_size is None else output_size
    return F.adaptive_avg_pool2d(x.detach().float(), size).to(x.dtype)
```

**golden 独立性核对**：该实现只用 `F.adaptive_avg_pool2d`（PyTorch 官方语义），不含 §0.6 的任何结构（无 C 轴向量化、无转置、无 C4/C4b 特化分支），
故精度对比是**独立验证**而非自证。

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（取自 `examples/TileOPs/tests/ops/test_pool.py`：`(1e-3,1e-3)` for fp16 / `(1.6e-2,1.6e-2)` for bf16。fp32 不适用——算子拒绝 fp32。）

### 8.3 L0 门槛测试计划（必须通过）

> 完整分层套件（L1/L2/Boundary）由 `tilelang-op-develop` 生成；本节只列 L0 门槛。

| 编号 | 用例 | 目的 |
|------|------|------|
| L0-1 | `(1,8,4,4,out=(2,2))` fp16（manifest smoke） | 基础正确性 + 单元小 shape |
| L0-2 | `(2,128,56,56,out=(6,6))` fp16（manifest spp） | 主 workload + 非整除 W-bin（重叠） |
| L0-3 | `(8,2048,7,7,out=(1,1))` fp16（manifest resnet） | **C4 特化**路径 + persistent（256 单元） |
| L0-4 | `(2,64,55,57,out=(7,7))` bf16（manifest nondiv） | bf16 + `vcast` 中转 + 双轴非整除 |
| L0-5 | `(1,8,9,11,out=None)` fp16（恒等） | **C4b 特化**路径（identity） |
| L0-6 | `(1,16,11,13,out=(None,5))` fp16 | 半 None 轴（H 恒等 + W 自适应） |
| L0-7 | `(1,16,10,10,out=4)` bf16 | 标量 `output_size` 展开 + bf16 |
| L0-8 | `(16,11,13)` 3D CHW → out `(5,4)` | 3D 输入 rank 保持（op 层 `unsqueeze/squeeze`） |
| L0-9 | `(2,64,56,56,out=(6,6))` fp16 | 与 L0-2 不同 N/C，验证 c_block 边界 |
| L0-10 | `(2,2048,7,7,out=(1,1))` fp16 | persistent 核数切换边界（num_logical ≫ 40） |

**L0 通过判据**：`torch.testing.assert_close(out, golden, atol, rtol)` 全通过；且 `op.kernel` 为 `AdaptiveAvgPool2dKernel`（与源测试一致）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **`T.transpose` 只支持二轴交换**（`docs/Tilelang.language/创建操作/T.transpose.md` §2.3）——本设计只做 `[1,0]` 二轴交换，满足。
   **硬风险**：`PL-1.1-transpose-chain` 明确「reshape→transpose 链会 mis-compile（数据错乱），必须始终在自然形状 buffer 间做 transpose」——
   实现时**禁止**在 transpose 前对 buffer 做 reshape，必须用自然形状 `(CH,W_in)`↔`(W_in,CH)`。
2. **不带浮点运算的算子（C4b 恒等路径）**：纯拷贝路径若 dtype 相同，`T.copy` 不插 cast；需确认 UB 中转拷贝在 npuir 下正确（Developer 支持 UB→UB）。
3. **窄向量 op 发射代价未逐点实测**（§1.6.0 发射项口径）——`CONST-vector-launch-overhead` 的 0.5µs/op 是 bn=512 大 op 实测值，本设计 ≤128 元素窄 op 属外推；**方向证据已有实测**（PL-1.4：CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs，「更宽 C 向量更优」），主选 CH=64 即该已测范围最优档；仅比 PL-1.4 更窄的 CH<16 属未实测外推，列 §1.6.3 D-3 裁决计划（备选 CH=8）。
4. **GM 行片 strided 搬运**：`T.copy(x[n, c0:c0+CH, ih, 0:W_in], row_src)` 是「4D tensor 的 2 维切片」（c 维 stride = `H_in*W_in`，尾轴连续长度 `W_in*2` 字节 = 110–114 B）。
   段长偏小（<128 B）→ 可能受 DMA 指令数限制。**若实测搬运动成为瓶颈**，Stage 4 备选：整片搬运 `(CH, bh*W_in)` 连续块（段长 `bh*W_in*2`，需 3-buffer 3D 转置，UB 压力大）——列入 Stage 4 A/B。
5. **`T.vadd`/`T.reduce_sum` 不支持 bf16**（§3.4）——bf16 必须 `vcast` 上转 fp32，本设计已内含，无额外风险。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| UB 溢出 | CH 过大（如 resnet 用 CH=1024 且 C4 路径 `(1024,49)` fp32） | 编译期 `ub overflow` 报错 | 按 §4.5 预算限 CH——统一取 `CH = min(C, 64)`（与 §5.2 一致；resnet C=2048 ⇒ CH=64，非 256）；factory 内加 host 侧 `CH` 上界守卫（`ValueError`） |
| 动态 `T.serial` 边界 | 用 `bin_h(o_h)` 作 `T.serial` 上界 | 编译期拒绝（源注释明证） | 用静态 `MAX_BIN_H` + `if kh<bh` 守卫（本设计） |
| `T.transpose` permutation >2 轴 | 误写 3-cycle | 编译期报错 | 只做 `[1,0]`；多次二轴交换用链 |
| reshape 后 transpose | 图省事 reshape `(CH*W_in,)` 再转置 | **数据错乱（静默错）** | 严格在自然形状 buffer 间转置（PL-1.1 硬约束） |
| bf16 直接 vadd | 忘记 `vcast` 上转 | 编译期 dtype 不支持 | 先 `T.vcast(src,dst,round_mode="rint")` |
| 累加器未清零 | `acc` 跨任务残留 | 结果偏大 | 每单元入口 `T.clear(acc)` 整块清零（§1.6.2 #0；**勿**用逐元素标量循环） |
| `uid` 越界读 | persistent 尾任务 | 越界/错误结果 | `if uid < NUM_LOGICAL` 守卫；**退化路径（C4/C4b）的 `NUM_LOGICAL` 必须与整平面单元同粒度**（`N*num_cblk`），否则解码 `n=uid//num_cblk` 会越界（§5.5 ①、§5.2 末行） |

### 9.3 特殊场景处理

- **非整除 bin（重叠行/列）**：由整数 `adaptive_bin` 精确处理（与 PyTorch 逐位一致），非近似——无需 padding。
- **极小 shape**：`(1,1,1,1)` 走 C4（`H_in=W_in=out=1`）；`out==in` 走 C4b；均已在 `verify_equiv.py` 覆盖。
- **极大 shape**：`num_logical ≫ vector_cores`（resnet 256）走 persistent（§5.5）。
- **混合精度**：bf16 输入 → fp32 累加 → bf16 输出（单次 `rint` 舍入），与源一致。
- **NaN/Inf**：按 IEEE 传播（`verify_equiv.py` 已覆盖 inf/nan 角点，两式一致）。
- **cv 融合**：不适用（纯 Vector）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/
├── DESIGN.md                    # 本设计文档
├── verify_equiv.py              # §1.6.1 数学等价优化的机器验证（D-1）
├── RETROSPECTIVE.md             # Stage 1 复盘（skill Phase 8）
└── （Stage 3 产出 `_adaptive_avg_pool2d_fwd_kernels.py` 等，不在本阶段范围）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档（§0 迁移分析 + §1.6 算法调研与优化） |
| `verify_equiv.py` | ✅ 已完成（45 用例全 EQUIV_PASS：C4/C4b 逐位等价 + C1 容差内 ≤1 ulp） | 数学等价优化机器验证脚本 |
| `_adaptive_avg_pool2d_fwd_kernels.py`（重写为 npuir） | ⬜ 待实现 | Stage 3 |
| `tests/ops/test_pool.py`（L0 通过） | ⬜ 待实现 | Stage 3 |

### 10.3 命名规范

- 项目目录名：`adaptive_avg_pool2d`（snake_case）
- 算子目录名：`_adaptive_avg_pool2d_kernel`
- 实现文件：`_adaptive_avg_pool2d_fwd_kernels.py`（harness 逐函数约定，函数名 `_adaptive_avg_pool2d_kernel` 不变）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（`F.adaptive_avg_pool2d`）
3. ⬜ kernel 重写（`_adaptive_avg_pool2d_kernel`，target=npuir）+ L0 精度比对

---

## 11. 修订记录（revision_index = 2）

> 本节为设计修订（Stage 2 检视路径 A）的可追溯记录。
> - **v1（revision_index = 1）**：上一版 = `history_version/design_v0.md`；检视报告 = `history_version/review_v0.md`（结论：不通过；维度 0/1–7 通过，维度 8 fail，2 阻塞 + 4 建议）。
> - **v2（revision_index = 2，本版）**：上一版 = `history_version/design_v1.md`；检视报告 = `REVIEW.md`（结论：不通过；**维度 0/8 均过，维度 3 fail**，1 阻塞 + 1 建议）。修复内容见 §11.3 / §11.4。

### 11.1 相对上一版的关键调整（v1）

| # | 检视问题 | 严重度 | 本版调整 |
|---|----------|--------|----------|
| 1 | §1.6.2 未盘点 §3.3 单元入口的 `acc` 零初始化逐元素标量循环 | 阻塞 | §1.6.2 增列 **#0**（`acc` 零初始化 → **`T.clear(acc)`**，佐证 `docs/Tilelang.language/内存操作/T.clear.md`）；§3.3 伪代码与 §1.4 步骤 2 改为 `T.clear` 形态；§6.1 增列「累加器初始化｜整块清除｜`T.clear`」行；§3.4/§9.2 同步 |
| 2 | 候选 #1「倒数乘」在 §1.6.0/§1.6.1/§0.6/§3.3 之间采纳结论互相矛盾 | 阻塞 | **统一为「采纳」**（与 §0.6 R2/§3.3 现状及 PL-1.5 已验证模式一致）：§1.6.1 C1 行改为已采纳并补齐四要素（容差内 ≤1 ulp）；优化结论改为「采纳 3 项（C1/C4/C4b）」；§3.1 步骤 6 去掉「或保留 ÷ d」歧义；`verify_equiv.py` 增 C1 用例与容差判据（45 用例全 PASS） |
| 3 | §4.5/§1.6.3 的 C4 行用 CH=256，与 §5.2 `CH=min(C,64)`/§5.5 不自洽 | 建议 | 统一 `CH = min(C,64)`：§4.5 C4 行改 `(64,49)` 口径 ≈19 KB；§1.6.3 UB 影响列改 ≈19 KB；§1.6.0 R4 #4 行、§5.3、§9.2 守卫同步 |
| 4 | §1.6.1 C1 收益分母用 24（应为 144） | 建议 | C1 行收益改为「归一化向量 op 数 = `num_logical × out_W = 144`，占比 ≈1.0%」 |
| 5 | §4.5 UB 膨胀系数 1.10–1.12 属 `auto-multi-buffer=false` 口径 | 建议 | §4.5 表下注改为两档并列：默认（multi-buffer 开）`~1.7x` → spp ≈70 KB / C4 ≈33 KB；关闭档 1.10–1.12 → spp ≈46 KB；两档均 < 192 KB |
| 6 | D-3 前提「窄 op 发射代价未实测」与 PL-1.4 实测张力 | 建议 | §1.6.3 D-3 段补入 PL-1.4 实测（CH64 21.6µs < …）为**主证据**，主选 CH=64 即已测最优档；备选 CH=8 的理由限定为「验证比 PL-1.4 已测范围更窄的 CH」；§9.1 #3 同步 |

**未被本轮改动的已通过部分**：§0（源算子三问解读 / 耦合性判定 / NPU 重设计，检视维度 0 全过）、§1.6.0 调研四问结论主体、§1.6.3 布局决策结论（方案 #2）、§2 编程模式、§4 内存层级、§5 Tiling 与分核三要素、§6 循环结构主体、§7 同步、§8 验证方案。

### 11.2 为何不会再犯同一错误

- **问题 1（盘点遗漏）**：本版把「§3.3 伪代码中的每一个循环 / 标量写点」作为 §1.6.2 盘点的**充分必要条件**逐行核对——新增 #0 后，§3.3 中不再存在任何逐元素标量写序列（`acc` 初始化、累加、归一化、cast 均为整块/向量 op；`kh` 为 tile 级守卫循环、`kw`/`o_w` 为编译期展开的向量 op 序列）。§6.1 与 §1.6.2 建立**一对一映射**（每个保留循环在 §6.1 有对应行），杜绝「§1.6.2 声明全覆盖、§6.1 却缺行」的再次发生。
- **问题 2（跨节结论矛盾）**：本版对同一候选建立**单一决定源**——采纳与否**只在 §1.6.1 判定**，§1.6.0 R4/调研结论、§0.6 R2、§3.1/§3.3 一律**引用**该判定（§1.6.0 行 321 与调研结论均显式标注「→ §1.6.1 C1」）。采纳项的等价性验证范围与「已采纳优化项」严格匹配：C1 属容差内等价，故 `verify_equiv.py` 显式增列 C1 用例与「≤1 ulp」判据（v0 仅覆盖 C4/C4b 逐位等价），并在 §1.6.1 内嵌结果表中逐行记录。

### 11.3 相对上一版的关键调整（v2，revision_index = 2）

> 上一版 = `history_version/design_v1.md`；检视报告 = `REVIEW.md`（**维度 0/8 均过，维度 3 fail**：1 阻塞 + 1 建议）。
> v1 已闭合的 v0 两项阻塞与四项建议（`T.clear` 盘点 / 候选 #1 倒数乘全文统一 + `verify_equiv.py` 45 用例 C1 ≤1 ulp / CH=64 口径统一 / C1 收益分母 144 / UB 膨胀双档 / D-3 PL-1.4 主证据）**全部保持闭合、未推翻**。

| # | 检视问题 | 严重度 | 本版调整（采用检视推荐方案 ①） |
|---|----------|--------|-------------------------------|
| 1 | §3.3 C4b 恒等路径的**核数 / 单元粒度 / 解码三要素不自洽**（`num_logical=9` 配整平面单元 + `n=uid//num_cblk` → `N=1` 下 `n=0..8` 越界） | 阻塞 | **收敛为「整平面单元口径」**：C4b 路径 `NUM_LOGICAL = N*num_cblk`（`(1,8,9,11)` = **1**，非通用式 9）。**同步修订**：§3.3 伪代码（新增按编译期分支取值的 `NUM_LOGICAL`，三处 `if uid < NUM_LOGICAL`、C4b 解码回归 `n = uid // num_cblk` 且注明与 `NUM_LOGICAL` 同粒度）；§5.2（分块代码 + 表头改 `NUM_LOGICAL`、`(1,8,9,11)` 行 9→**1** + 表下分叉说明）；§5.3/§5.4 守卫符号；§5.5 ①（退化路径单元粒度单列）+ ③（新增 C4b 行 + 通用规则改 `NUM_LOGICAL` + 空核保证）；§4.3/§4.4（`stage_ub` 整平面单元交叉标注）；§6.1/§6.2/§6.4 守卫符号与分支粒度注释；§9.2「`uid` 越界读」行补退化路径守卫；§0.6 R1 加一句退化路径粒度交叉引用 |
| 2 | §1.6.2 表内 #8 标 `✅`、结论段文字却写「#8 虽标 ❌」 | 建议 | §1.6.2 向量化结论段改为「**#8** 标 `✅`——编译期展开为向量 op 序列，非逐元素标量循环」，与表一致（保留计数 4 = #7/#9/#10/#11 不变） |

**未被本轮改动的已通过部分**（保持稳定）：§0.1–§0.5、§0.7、§1.6.0 调研四问结论、§1.6.1 三项采纳（C1/C4/C4b）与机器验证结果（`verify_equiv.py` **未改动**，45 用例仍全 `EQUIV_PASS`）、§1.6.2 其余各行与保留计数、§1.6.3 布局决策（方案 #2）与 D-3 三件套、§2 编程模式、§4.1/§4.2/§4.5/§4.6/§4.7、§5.1、§6.3、§7、§8、§9.1/§9.3、§10。

### 11.4 为何不会再犯同一错误（v2）

- **根因**：`num_logical`（核数）与「单元处理多少数据」（单元粒度）在 C4b 分支上被解耦——通用式 `N*out_H*num_cblk` 与整平面单元 `N*num_cblk` 在 `out_H>1` 时不等，而解码却按整平面粒度写，三者两两错配。
- **本版的机制性防线（不再依赖逐处记忆）**：
  1. **单一粒度原则**：明确规定「**单元粒度是 `NUM_LOGICAL` 的唯一定义源**」，并在 §3.3 伪代码顶层用编译期分支一次性定义 `NUM_LOGICAL`（一般 / C4 / C4b 三分支），下游 §5.2/§5.5/§6.2 一律**引用**该变量而非各自重算——消除「核数用通用式、解码用单元式」的可能。
  2. **解码与核数同源校验**：任何路径的解码公式都必须与其 `NUM_LOGICAL` 满足 `uid < NUM_LOGICAL ⇒ 解码出的 (n, …) 下标 < 对应维长`。C4b：`NUM_LOGICAL=N*num_cblk`、解码 `n=uid//num_cblk ∈ [0,N)` ✓（`(1,8,9,11)` `uid=0` → `n=0`）。
  3. **空核越界防线**：`num_kernels = min(NUM_LOGICAL, vector_cores)` 保证 `num_kernels ≤ NUM_LOGICAL`，每个核首个任务恒有效；`num_local` 为 Python int 编译期常量（非 `T.ceildiv` 表达式），不触发 pattern-library 草稿 VP-2026-0058 记载的 `T.ceildiv` 负数域 empty-core 越界陷阱。
  4. **几何特化路径全覆盖自检**：v2 复核了三条路径（一般 / C4 / C4b）的 (单元粒度, `NUM_LOGICAL`, 解码, `stage_ub`/buffer 形状) 四元组自洽性，并逐行核对 §5.2 分块表全部 10 行（含 C4b 分叉行）与 §5.5 ③ 判定表。
- **对已闭合结论的稳定性**：本次修订为 C4b 几何特化路径的 **§5/§3.3 局部修正**，不触及算法族（§1.6.0）、数学等价优化（§1.6.1）与等价性验证范围（`verify_equiv.py` 未改），故 v1 已闭合的全部阻塞/建议保持闭合。

---

## 12. 性能目标（Stage 4 / optimize 场景，conductor 预检收集）

> 本节由 conductor 在 `scenario=optimize` 预检收集后**追加**（optimize.md §3 / new-op.md §5；不覆盖既有内容）。

- 性能目标类型：`throughput`（roofline 利用率）
- 目标数值：报告 `Ratio`（`roofline_utilization_percent`，= Perf/Computility）**尽量接近 70%**
  - **优先 workload**：`resnet-global`（`[8,2048,7,7] out=(1,1)`，C4 全局池化路径；baseline 时延最大、天花板最高）
  - `spp-6x6` / `nondiv-7x7` 这类小时延 shape 受 ~4µs kernel 地板限制，到不了 70% 属预期，**不视为失败**（best-effort）
- baseline（Stage 5 report，msprof op Task Duration）：resnet-global 169.26µs / Ratio 2.30%；spp-6x6 16.60µs / 3.81%；nondiv-7x7 14.50µs / 2.35%；smoke 3.70µs
- benchmark 参数化入口：`benchmarks/ops/bench_pool.py::test_adaptive_avg_pool2d_bench`（manifest-driven，op `AdaptiveAvgPool2dFwdOp`）
- 回归入口：`python perf_opt/_adaptive_avg_pool2d_kernel.py --level all`（L0/L1 阻塞；L2/Boundary 告警不阻塞）
- 噪声阈值：3%（默认）；最大迭代数：10（默认）
- 口径：`msprof op Task Duration(us)` 为唯一时延口径
- 关注 shape：`resnet-global` 优先（只影响调优分析顺序，不缩减 workload 集合；完整集合由 optimizer 展开 benchmark 实际参数化案例并分类）
- 额外约束（用户要求）：
  1. 三条结构路径（恒等 / C4 全局 / 一般网格）的分派**保持在 factory 层**，三条均须保持正确，不得挪到 op 层或 wrapper。
  2. `perf_opt/_adaptive_avg_pool2d_kernel.py` 须同时导出 `_adaptive_avg_pool2d_kernel` / `adaptive_bin` / `max_adaptive_bin_extent`；factory 调用签名与 baseline 一致；tuned 默认参数以**模块级常量**暴露。
  3. `tests/ops/test_pool.py` 与 `benchmarks/ops/bench_pool.py` 由池化 6 个算子共用，**只读不改**。
  4. DESIGN.md §1.6.3 的 **D-3 实验裁决**（主选 `CH=64` vs 备选 `CH=8`，含 shape 分派评估）在 Stage 4 为**必做项**。
