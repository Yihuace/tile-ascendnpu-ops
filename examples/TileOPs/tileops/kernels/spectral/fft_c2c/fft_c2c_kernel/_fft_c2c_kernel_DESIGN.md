# _fft_c2c_kernel 算子设计文档

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `tilelang-op-design` skill 的 references/migration-analysis.md）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。
>
> 源算子：`/home/developer/TileOPs/tileops/kernels/fft.py`（工厂 `_fft_c2c_kernel` L16 起；内嵌 `_fft_lut_func` L75 起、`@T.prim_func _fft_lut_main` L404 起；封装类 `FFTC2CKernel` L476 起）。脚手架自包含副本：`examples/TileOPs/tileops/kernels/spectral/fft_c2c/_fft_c2c_kernels.py`。

### 0.1 源算子语义（做什么）

**数学语义**：沿最后一维的一维复到复离散傅里叶变换（C2C FFT），等价 `torch.fft.fft(x, dim=-1)`：

$$
X[k] = \sum_{n=0}^{N-1} x[n]\,\exp\!\left(-2\pi i\,\frac{k n}{N}\right),\qquad k=0,\dots,N-1
$$

`N` 为最后一维长度（**必须为 2 的幂**，源码 L60–L61 强制校验；NPU manifest `shape_rules` 同）。

**规约语义（累加顺序）**：DFT 是 `n=0..N-1` 的复数和。源实现用 FFT 蝶形网络重排累加分组（radix-2/4/8 分治），与朴素顺序求和的浮点舍入不同——两者数学等价、数值上均为合法的 DFT 求值。**语义层只要求 DFT 结果**（容差内），不要求位级一致的累加顺序。这是后续允许用其它算法族求解的前提。

**dtype 语义**：
- 输入 `complex64` → 实/虚分量 `float32`；`complex128` → 实/虚分量 `float64`（源码 L53–L58）。
- 中间计算精度 `accum_dtype = real_dtype`（L88）：`complex64` 全程 fp32 乘加（**无 fp32 提升累加**——源码对 complex64 不做双精度累加）；`complex128` 全程 fp64。
- 蝶形内 `cast` 到 `accum_dtype` 再 `cast` 回 `real_dtype`（L158–L169），对该两档 dtype 均为同型往返、无实际精度变化。
- 主机侧 LUT 用 fp64 建表再 cast 到 `real_dtype`（op 层 `_build_lut` L96–L109）。

**边界语义**：
- `N` 必须为正的 2 的幂（`n & (n-1) != 0 or n == 0` → `ValueError`）。
- `nan/inf` 经复数乘加自然传播（无特殊处理）；无 0-size 维度（op 层校验 `n>0`）。
- 输出与输入**同 shape、同 dtype**；沿最后一维变换；前导维（batch）被展平后并行处理再 reshape 回原形（op 层 `forward` L145–L167）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（`torch.fft.fft(x, dim=-1)`），**不复刻 §0.6 的 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `x`（op 契约） | 输入 | `[..., N]` | complex64 / complex128 | 复输入，`N` 为 2 的幂 |
| `output`（op 契约） | 输出 | `same_as_input` = `[..., N]` | `same_as_input` | 沿最后一维的 DFT；**输出 shape 与 dtype 同输入** |
| `x_real`/`x_imag`（kernel 实参） | 输入 | `(B, N)` | float32 / float64 | 复输入实/虚分量；`B = prod(前导维)`（无前导维时为 1） |
| `lut_real`/`lut_imag`（kernel 实参） | 输入 | `(N-1,)` | 同 `real_dtype` | 预计算 twiddle 因子（所有 stage 扁平拼装） |
| `y_real`/`y_imag`（kernel 返回） | 输出 | `(B, N)` | 同 `real_dtype` | 结果实/虚分量，op 层 `torch.complex` 合并 |

> 输出 shape 显式确认：源仓与 NPU manifest 均为 `same_as_input`。注意 **kernel 边界是实/虚双 fp 张量**，复数 dtype 仅存在于 op 契约层（源实现本身即如此拆分）。

### 0.3 实现算法解读（怎么算）

源实现为**两阶段、全 stage 覆盖**的 Cooley-Tukey 迭代 FFT（含 LUT twiddle 与 SMEM stage 融合）：

**计算步骤分解**（源码每个计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| P1 | 按位反转下标读取输入到 SMEM（`fused_bitrev_smem_stages`，L107–L134）：对每元素做 `log2n` 次标量移位/或运算求 `rev_idx`，`smem_r/i[i]=x_real/imag[bb, rev_idx]` | `x_real/imag` | `smem_r/i` | DFT 的输入重排（DIT 前置 bit-reversal） |
| P2 | SMEM 内蝶形 stage 0..`smem_stages-1`（L139–L171）：每 stage 逐元素求 `angle`、`tw_r=T.cos/sin(angle)`、`t = v·w`、`u±t` 写回 SMEM | `smem_r/i` | `smem_r/i` | radix-2 DIT 蝶形（小 stride 段） |
| P3 | SMEM 写回 GM（L174–L183） | `smem_r/i` | `y_real/imag` | 中间结果落盘 |
| P4 | LUT stage（`lut_butterfly_stage` L185–L231）：单 stage 一 kernel，`t = v·LUT[k]`、`u±t` | `y_real/imag`, `lut_real/imag` | `y_real/imag` | radix-2 蝶形（大 stride 段） |
| P5 | LUT radix-4 融合 stage（L239–L301）：两级蝶形合一 | 同上 | 同上 | 两 radix-2 级融合 |
| P6 | LUT radix-8 融合 stage（L303–L402）：三级蝶形合一 | 同上 | 同上 | 三 radix-2 级融合 |
| P7 | stage 调度（`_fft_lut_main` L404–L437）：P1+P2+P3 一次；P4/P5/P6 按 `lut_stage_count` 排布（`r8_count=(count//3)*3`，余 2 走 radix-4、余 1 走 radix-2） | — | — | 全 stage 覆盖 |

**数据流与内存访问模式**（源 GPU 视角）：

```
GM[x_real/imag] --(bit-reversed 逐元素 gather, P1)--> SMEM[smem_r/i (≤2·threads)]
   --(P2: log2(min(n,2·threads)) 个 radix-2 级, 全程 SMEM)--> SMEM
   --(P3)--> GM[y_real/imag]                          # 阶段一：1 次 GM 读 + 1 次 GM 写
   [阶段二：剩余 lut_stage_count 个 stage]
   GM[y] --(P4/P5/P6, 每 stage/融合组一 kernel, 读 LUT)--> GM[y]   # 每 stage 一读一写
```

- 阶段一（P1–P3）覆盖 stride `< 2·threads` 的小 stride 段，SMEM 内完成，GM 仅 1 读 1 写。
- 阶段二（P4–P6）覆盖大 stride 段，每 stage（或融合组）独立 kernel，**每 stage 一次 GM 往返**（`O(log n)` 次）。

**循环与并行结构**：grid = `(smem_blocks, batch_size)` 二维（L103/L207/L264/L330）；`threads=512`（默认 config，L526–L528）；块内 `T.Parallel(threads)` 元素级并行；SMEM stage 循环为 **Python 级 `for s in range(smem_stages)`**（编译期展开，L139）；LUT stage 循环同为 Python 级（L420）。`block_size=1024` 控制 LUT stage 每块工作量。

**host 侧逻辑**（op 层 `FFTC2COp.forward`，`examples/TileOPs/tileops/ops/spectral/fft_c2c.py` L123–L167）：
1. 校验 device/dtype/`N` 为 2 的幂；
2. `x.real.contiguous()` / `x.imag.contiguous()` 拆复数为实/虚；
3. 前导维展平为 `batch_size`（`x_real.reshape(batch_size, N)`）；
4. `_get_lut`：fp64 建 twiddle LUT（`angles` → `cos/sin` → cast `real_dtype`），带缓存；
5. kernel 派发；
6. 输出 `reshape` 回原 shape，`torch.complex(y_real, y_imag)` 合并。

**编译配置**：`TL_ENABLE_FAST_MATH: True`、`compile_flags=["-O3"]`（L68–L74）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | 预计算 twiddle LUT（`lut_real/imag`，size `N-1`） | 大 stride 段不再重算 `sin/cos` | host 建表，kernel 查表 | 无（纯算法/数据组织） | **可移植**（主机建常量表 → 常量输入张量） |
| O2 | SMEM stage 融合（阶段一，L91/L483–L487） | 前 `log2(min(N,2·threads))` 级在 SMEM 内一气呵成，GM 仅 1 读 1 写 | 分块驻留共享内存 | shared memory | **硬件强相关**（→ UB 等价替换） |
| O3 | bit-reversal 融合进 SMEM 装载（L107–L134） | 消去独立的 bit-reversal kernel | 装载时按 `rev_idx` 逐元素 gather | shared memory + 逐线程标量地址 | **硬件强相关**（NPU 无跨步 gather） |
| O4 | LUT stage radix-8/4 融合（L239–L402） | 减 kernel launch 数、减 GM 往返 | 一 kernel 处理 3/2 个连续 stage | 无（纯算法层融合） | **可移植**（但单趟设计下收益消失，见 §0.5） |
| O5 | per-stage 一 kernel（阶段二） | 大 stride 段用查表替代三角 | 多 kernel launch | — | 实现层（NPU 单趟设计下重新考虑） |
| O6 | 小 stride 段**即时计算**三角（P2，`T.cos/sin`） | 小 stride 段 twiddle 数少，免建表 | kernel 内 `T.cos/sin` | 标量三角/超越函数 | **硬件强相关**（NPU 无标量超越、vcos/vsin 角度域受限） |
| O7 | batch 映射到 grid y 轴（L103） | 多 FFT 并行 | 二维 grid | — | 实现层（NPU 一维 grid） |
| O8 | thread 级并行 + `block_size` ILP（L107/L208） | 元素级并行、LUT 段吞吐 | CUDA threads / ILP | CUDA thread 模型 | **硬件强相关**（NPU 无 threads 概念） |

> 识别不出来 ≠ 不存在。以上 8 项已逐项登记，§0.5 逐项处置，§0.6 承接意图，杜绝静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（DFT） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| complex64 dtype | 语义 | 无 | 无复数 dtype，但可实/虚分解 | **等价替换** | 拆为两份 fp32；kernel 全程 fp32，wrapper 侧 `x.real/x.imag` + `torch.complex` 合并（零拷贝视图）。**依据**：`src/target/codegen_npuir.cc::PrintType` L303–L418 无 complex 分支（LOG(FATAL)）；源实现本身即为实/虚双张量架构。详见 §0.6 R1 与 CG-2026-0019 |
| complex128 dtype | 语义 | 无 | **无**（fp64 向量元算缺失） | **重新设计（无可行路径）** | NPU 向量 dtype 表（T.vadd/T.vcos/T.vsin/T.deinterleave §2.2.1）均无 fp64 → complex128 不可实现；double-double(fp32×2) 软件模拟代价高且未验证 → **如实披露为能力缺口**（CG-2026-0019），manifest `fft-4k-c128-b64` 不支持 |
| O1 twiddle LUT | 算法/数据 | 无 | 有 | **保留** | host 建常量表作 kernel 输入张量（不修改输入内容，合规）；NPU 无标量三角 → LUT 更是**必需**（见 O6） |
| O2 SMEM stage 融合 | 优化 | shared memory | 有（UB） | **等价替换** | GM→UB 驻留整行；UB 192KB（CONST-capacity-910B2C，9362 一致）。本设计进一步做**单趟全 stage 融合**（整行常驻 UB），GM 往返从源 `O(log n)` 降到 **1 读 1 写** |
| O3 bit-reversal gather 装载 | 优化 | SMEM + 逐线程标量地址 | **无**（跨步 gather 向量指令缺失：TRAP-C10-strided-gather，`pattern-library/traps-compiler.md`） | **重新设计** | 改 **Stockham autosort**：全程无 bit-reversal 置换，每 stage 读写均为连续段（详见 §0.6 R3、§1.6.0） |
| O4 radix-8/4 融合 | 算法 | 无 | 有 | **保留（降优先级）** | 意图（减 GM 往返）在单趟 UB 设计下已由 O2 的整行常驻实现；radix 融合仅剩减 UB 内 ping-pong 趟数的小收益，列为 §1.6.3 可选 refinement |
| O5 per-stage 一 kernel | 实现 | 多 kernel launch | 有 | **舍弃** | 单趟 UB 设计无 per-stage kernel；舍弃理由：NPU 核启动开销 + 无必要 |
| O6 内核内即时 `T.cos/sin` | 优化 | 标量/向量三角 | 无（标量） | **重新设计** | kernel 内不做三角：twiddle 全部取自 LUT（O1）。**依据**：`T.vcos/T.vsin` 为张量级且角度域受限（T.vcos.md §2.3：`[-1,1]` 高精度、超 `[-2,2]` 急剧劣化），标量超越函数在 NPU 融合路径不可用（VP-2026-0140）；twiddle 角度覆盖 `[-2π,0]` 远超域 |
| O7 二维 grid | 实现 | grid y 轴 | 无一维以上 | **重新设计** | `T.Kernel` 仅一维（ascend-constraints.md §1）；展平为 `grid = B`（见 §0.6 R2、§5） |
| O8 threads / block_size | 实现 | CUDA thread 模型 | **无** | **舍弃** | `T.Kernel(..., threads=)` 在 npuir 无效果（TRAP-threads-kwarg-noop，`pattern-library/traps-compiler.md`）；块内元素并行交给 `T.Parallel` 向量化。**依据**：KernelLaunch NPU 分支仅消费 grid_size |
| swizzle 防 bank conflict | 优化 | SMEM bank | 无（NPU 无此问题） | **舍弃** | 不适用（migration-analysis.md §5.3） |

> 每项处置均有依据；舍弃项（O5/O8/swizzle）已写明理由。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（**判定输入含 §1.6.0 调研结论**——源算法只是基线候选之一）：

**重设计项 R1：复数 dtype → 实/虚分解（承接 CG-2026-0019 与源架构）**

- **源方案**：kernel 边界即为 `x_real/x_imag` 两个 `real_dtype` 张量，op 层拆分/合并；复数乘 `t = v·w` 展开为 4 实数乘 2 加。
- **NPU 新算法**：同源架构——kernel 全程 fp32（complex64 路径），实/虚独立 UB 缓冲，蝶形以 `vmul/vsub/vadd` 实数向量运算表达（`(a+bi)(c+di) = (ac-bd) + i(ad+bc)`）。complex128 无路径（见 §0.5/CG-2026-0019）。
- **语义保持论证**：复数乘展开为实数运算是**精确代数恒等式**（同数域，无精度变化）；complex64 路径全程 fp32 与源 `accum_dtype=real_dtype=float32` 完全一致，累加分组变化仅影响 fp32 舍入（容差 1e-4 内，§8.2）。边界：nan/inf 经实数乘加自然传播，与源一致。

**重设计项 R2：二维 grid + threads → 一维 grid**

- **源方案**：`T.Kernel(smem_blocks, B, threads=512)`，`threads` 控制 SMEM 覆盖深度。
- **NPU 新算法**：`T.Kernel(B, is_npu=True)` 一维；**整行常驻 UB**，一个 block 处理一个 FFT 行（不再有 SMEM 分块的 `smem_blocks` 维）；`threads` 语义整体舍弃，块内并行由 `T.Parallel` 承担。
- **语义保持论证**：block 到 FFT 行的映射是 1:1 任务划分，不改变计算语义；`threads` 仅影响源硬件调度（SMEM 覆盖深度），无数学语义。

**重设计项 R3：bit-reversal → Stockham autosort（关键重设计，承接 O3）**

- **源方案**：DIT 蝶形需 bit-reversed 输入；源用「装载时按 `rev_idx` 逐元素 gather」融合消去独立 kernel（O3）。
- **NPU 新算法**：改用 **Stockham autosort DIF radix-2**——**完全不做 bit-reversal 置换**，每 stage 以两次连续段读写 + ping-pong 实现，输出直接为自然序。第 h 级（`h = N/2, N/4, …, 1`，`n_groups = N/(2h)` 组）：
  ```
  for g in 0..n_groups-1, i in 0..h-1:
      lo = A[g*2h + i] ;  hi = A[g*2h + h + i] ;  w[i] = exp(-2πi·i/(2h))
      B[g*h + i]        = lo + hi
      B[g*h + i + N/2]  = (lo - hi) · w[i]
  ```
  读段 `A[g*2h..g*2h+h-1]`、`A[g*2h+h..g*2h+2h-1]` 与写段 `B[g*h..]`、`B[g*h+N/2..]` **全部为长度 h 的连续段**（无 gather/scatter）。
- **语义保持论证**：该迭代格式与 radix-2 DIF Cooley-Tukey 数学等价（**已数值验证**：n=2/4/8/16/64 与 `numpy.fft.fft`/`torch.fft.fft` 比对 max 误差 ≤ 8.8e-15（fp64）；见 §1.6.1 与 `verify_equiv.py` 的 full-FFT 行）。累加分组与源不同 → fp32 舍入差异在容差内。边界：`N` 为 2 的幂时 `n_groups`/`h` 均整除，无尾块。

**重设计项 R4：全 stage 单趟 UB 融合（承接 O2）**

- **源方案**：阶段一（小 stride）SMEM 融合 + 阶段二（大 stride）每 stage 独立 kernel（`O(log n)` 次 GM 往返）。
- **NPU 新算法**：`N·sizeof(fp32) = 16KB`（实/虚各）经 UB 可整行常驻 → **全部 `log2 N` 级在 UB 内一次完成**，GM 仅 1 读 1 写。UB 预算见 §4.5。
- **语义保持论证**：级间数据流与源逐级等价（同一蝶形网络），仅把 GM 中转变为 UB 内 ping-pong；数学语义不变，GM 流量从源 `O(log n)` 降至 2 趟（roofline 最优）。

**重设计项 R5：kernel 内三角 → LUT（承接 O6）**

- **源方案**：小 stride 段即时 `T.cos/sin`，大 stride 段查 LUT（O1/O6 混合）。
- **NPU 新算法**：**所有** stage 的 twiddle 取自 LUT（host fp64 建表 → fp32 常量张量），kernel 内零三角。LUT 布局沿用源（stage `h` 占 `LUT[h-1 : 2h-1]`，连续段）。
- **语义保持论证**：LUT 值为 `exp(-2πi·i/(2h))` 的 fp64 精确值 cast fp32，与源 LUT 同源；内核内三角路径在 NPU 不可用（角度域超界），LUT 是**唯一可行**等价路径。舍入差异：LUT fp32 vs fp64 精确三角 ≤ 0.475 ulp（§1.6.1 验证表）。

### 0.7 标杆实现

- **源算子路径**：`/home/developer/TileOPs/tileops/kernels/fft.py`（`_fft_c2c_kernel` + `FFTC2CKernel`）。
- **源仓参考实现/测试基准**：`examples/TileOPs/tests/ops/test_fft.py`（精度用例，tolerance：c64 `atol=rtol=1e-4`、c128 `1e-8`）；`examples/TileOPs/benchmarks/ops/bench_fft.py`（性能用例，基线 `torch.fft.fft`）。
- **golden**：以 §0.1 语义为依据的 `torch.fft.fft(x, dim=-1)`（§8.1），**不复刻 §0.6 的 NPU 算法**（验证独立性）。
- **manifest**：`examples/TileOPs/tileops/manifest/spectral.yaml` → `FFTC2COp`（workloads `[4096]` c64、`[64,4096]` c64/c128；roofline `flops=batch·5·n·log2 n`, `bytes=batch·2·n·elem_bytes`）。

---

## 1. 概述

### 1.1 算子名称

`_fft_c2c_kernel`（1D Complex-to-Complex FFT，等价 `torch.fft.fft`）。

### 1.2 功能描述

对输入张量沿最后一维做一维复到复离散傅里叶变换，前导维展平为 batch 并行处理；输出与输入同 shape、同 dtype。

### 1.3 数学公式

$$
X[k] = \sum_{n=0}^{N-1} x[n]\,\exp\!\left(-2\pi i\,\frac{k n}{N}\right),\qquad N = 2^{t},\ k=0,\dots,N-1
$$

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6；与源算法结构差异逐条标注来源）：

采用 **Stockham autosort DIF radix-2 单趟迭代 FFT**（§0.6 R3），整行常驻 UB（§0.6 R4），twiddle 全取 LUT（§0.6 R5），实/虚分解为 fp32（§0.6 R1）：

1. **搬入**：GM `(B,N)` → UB（实/虚各一份），1 次 GM 读；
2. **`t = log2 N` 级迭代**（`h = N/2, N/4, …, 1`），每级对全部 `n_groups = N/(2h)` 组做连续段蝶形，ping-pong 到另一 UB 缓冲；
3. **搬出**：UB → GM，1 次 GM 写。

**与源算法的结构差异**：
- 源用 DIT + bit-reversal gather 装载（O3）→ 本设计用 **Stockham autosort DIF**，无任何置换（§0.6 R3）；
- 源分两阶段、阶段二 per-stage kernel（`O(log n)` GM 往返）→ 本设计**单趟全 stage UB 融合**（§0.6 R4）；
- 源 grid 二维 + threads（O7/O8）→ 本设计**一维 grid = B**（§0.6 R2）；
- 源小 stride 段即时三角（O6）→ 本设计**全 LUT**（§0.6 R5）。
- 源 radix-8/4 融合（O4）意图（减 GM 往返）由单趟 UB 融合承接；radix 融合降为可选 refinement（§1.6.3）。

### 1.5 数据流图

```
GM[x_real/imag (B,N)] --T.copy--> UB[A_r/A_i (N)]           # 1 次 GM 读
   ┌──────────── 重复 t = log2(N) 级（h = N/2 → 1）────────────┐
   │  UB[A_r/A_i] --(连续段蝶形 + LUT twiddle, vadd/vsub/vmul)--> UB[B_r/B_i]
   │  ping-pong: A <-> B ；W[twiddle] (UB 常驻) 供 w[i] 查表       │
   └────────────────────────────────────────────────────────────┘
UB[out_r/out_i] --T.copy--> GM[y_real/imag (B,N)]            # 1 次 GM 写
   （op 层：torch.complex(y_real, y_imag) → [..., N] complex）
```

### 1.6 算法调研与优化分析 ⭐

> 先调研（1.6.0）→ 再数学等价优化（1.6.1）→ 再向量化替代（1.6.2）→ 再布局决策（1.6.3）。分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

**调研对象**：同一数学语义（`N` 点 DFT，`N` 为 2 的幂，复输入）的算法族。源算法（Cooley-Tukey DIT + bit-reversal + SMEM 融合 + LUT）只是**基线候选 #0**。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 对比 |
|---|------|------------|-----------|---------|--------------|
| 0 | **基线：源算法**（Cooley-Tukey DIT radix-2/4/8 + bit-reversal + SMEM/LUT） | `X=Σx[n]w^{kn}` 分治，DIT 需 bit-reversed 输入 | — | — | ✅（基线） |
| 1 | 朴素 DFT（直接按定义，或 DFT 矩阵乘法 `X = W·x`） | `X[k]=Σ_n x[n]w^{kn}`，`O(N²)` 复 MAC | 数学恒等 | 无（复杂度爆炸） | ✅（对照下界） |
| 2 | Cooley-Tukey radix-2 DIT（含 bit-reversal） | 分治 + 输入 bit-reversal | 恒等（容差内） | `O(N log N)`；需提取/散列重排 | ✅ |
| 3 | Cooley-Tukey radix-4 / radix-8 | 每级融合 2/3 个 radix-2 | 恒等（容差内） | 少级数/少复数乘（N/2→N/4 等）；结构更繁 | ✅ |
| 4 | **Stockham autosort**（DIF/DIT，ping-pong 自排序） | 每级连续段读 + 连续段写 + ping-pong，**无 bit-reversal** | 恒等（容差内，已数值验证 §1.6.1） | 消除重排（NPU 缺 gather）；访存模式规整 | ✅（**选定**） |
| 5 | LUT twiddle（twiddle 预计算查表） | twiddle 值 `w` 改为查常量表 | 恒等（LUT 精确值 cast，≤0.5 ulp） | 消三角计算（NPU 无标量三角） | ✅ |
| 6 | 混合基 / real-input 特化 | 非 2 幂基；实输入 half-size | — | 不适用（`N` 为 2 的幂；输入为复） | ❌（不适用） |
| 7 | Bluestein / Rader | 任意长度 → 卷积 | 恒等 | 不适用（`N` 为 2 的幂，无收益） | ❌（不适用） |
| 8 | 六步/四步分解（矩阵分解 + transpose，`N=N1·N2`） | 2D 视图 + 子 FFT + twiddle + 转置 | 恒等（容差内） | 暴露 `N1+N2` 路子 FFT 并行度；引入转置代价 | ✅（并行度备选，见 R4/§1.6.3） |
| 9 | DFT-as-matmul（Cube 路径） | `W[N,N]` 复矩阵 × `x` | 数学恒等 | Cube 算力密集，但 `W` 内存 `N²`（4096²·8B=128MB）不可行 + `O(N²)` FLOPs | ❌（内存/复杂度否决） |

**R2 在线算法**：**结论「无」**（针对本算子范围）。结构依据：FFT 是**固定的蝶形网络**，每个输出依赖全部输入，不存在可分解为 `running 统计量`（和/max/计数）的流式形式；单遍性已由算法本身满足（每级只读一遍数据，无跨样本的在线再缩放需求；对比 online softmax 的 running max/sum 无 FFT 对应物）。源算法亦未用任何 online 手段（§0.4 无相关条目），与结论一致。「流式 FFT / overlap-add」仅用于信号超长、无法整块驻留的场景——本算子 `N` 静态且整行（16KB/实虚）可驻留 UB，无分块必要，故在线变体无收益。

**R3 复杂度对比**（四口径；口径说明：FLOPs 计复数乘加按 4 实乘 2 实加、复加 2 实加折算为实数 FLOPs；访存 Bytes 指 GM 流量；扫描遍数指同一数据的 GM 重复读取次数；中间缓冲峰值指片上工作集）：

| 算法候选 | FLOPs（实数，含口径） | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|---------------------|-------------------|---------|-------------|---------------------|
| #0 基线源算法 | `~10·N·log2N`/行（radix-2 复蝶形 8 实 FLOPs × N/2 × log2N） | `B·(2 + O(log2N))·N·elem`（阶段二 per-stage GM 往返） | `1 + O(log2N)` | SMEM `2·min(N,2·threads)` | grid `smem_blocks×B`；跨核无（每 stage 独立 kernel） |
| #1 朴素 DFT | `~8·N²`/行（复 MAC = 8 实 FLOPs） | `B·2·N·elem` | 1 | `O(N)` | `N` 路（每输出独立）；但 FLOPs `N/log2N`× 更大 |
| #2 radix-2 DIT | `~10·N·log2N`/行 | 单趟 UB 时可 2 趟；否则含重排段 | 1–2 + 重排 | `O(N)`（整行） | `B`（行间）；bit-reversal 重排是 gather |
| #3 radix-4/8 | `~8·N·log2N`（乘数略降） | 同 #2 | 1–2 | `O(N)` | `B` |
| **#4 Stockham autosort（选定）** | `~10·N·log2N`/行 | **`B·2·N·elem`**（理想 2 趟，整行 UB） | **1（读）+1（写），无中间 GM 往返** | `O(N)`（`2·N` fp32 缓冲 ping-pong） | `B`；行内级间串行（蝶形依赖） |
| #5 LUT | 与 #2/#3/#4 同 + 查表访存（UB 内） | 同所依附算法 | 同 | `O(N)`（LUT `N` 复） | — |
| #8 四步分解 | `~10·N·log2N` + 转置开销 | `B·(2 + 转置)·N·elem` | 2–3（含转置） | `O(N)` 或分块 | `B·N1`（暴露子 FFT 并行度） |

> 基线在表内；访存/遍数为本算子主导项（算术强度≈3.75 FLOP/B < 脊点≈4，见 R4），故重点比较访存。多项式级 FLOPs 上 #0/#2/#4 同量级，差异在访存与重排。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水/融合 | 结论 |
|---------|-------------|---------|-----------|---------|-----------|------|
| #0 基线源 | Vector（蝶形逐元素）✓ | SMEM 分块 ✓ | ✓ | ✓ | 阶段二多 kernel | ⚠️ 结构可移植但 bit-reversal gather 在 NPU 不可用（TRAP-C10）、threads 无效 |
| #1 朴素 DFT | Vector（无 MAC 密集，`N²` 标量） | `O(N)` ✓ | ✓ | ✓ | 差（FLOPs 爆炸） | ❌ FLOPs `N/log2N`× 更大（N=4096 → 341×） |
| #2 radix-2 DIT | Vector ✓ | `O(N)` ✓ | ✓ | ✓ | ✓ | ❌ bit-reversal = gather（TRAP-C10 缺失）——除非整行重排成本可接受 |
| #3 radix-4/8 | Vector ✓ | `O(N)` ✓ | ✓ | ✓ | ✓ | ⚠️ 与 #4 组合为 refinement（少级数） |
| **#4 Stockham autosort** | **Vector ✓（全为连续段逐元素）** | **16KB/实虚（`N=4096` fp32）✓** | **h 为 2 的幂，段长 2^k ✓；fp32 向量宽 ×4** | **✓（`N` 静态）** | **✓（单趟 UB 融合）** | **✅ 选定**：无 gather、无重排、访存最优 |
| #5 LUT | ✓（查表 + 向量乘） | LUT `N` 复 ≈32KB | 段长 h 连续 ✓ | ✓ | ✓ | ✅ 采纳（与 #4 组合） |
| #8 四步分解 | Vector ✓ + 转置 | `O(N)` | 需转置对齐 | ✓ | 转置代价 | ⚠️ `B` 小（unbatched）时的并行度备选（§1.6.3 裁决） |
| #9 DFT-as-matmul（Cube） | Cube（MAC 密集）✓ | **`W` 需 `N²·8B`（128MB）❌** | 分形 ✓ | ✓ | — | ❌ `N²` 内存 + `O(N²)` FLOPs（内存/复杂度否决；`T.gemm` 的 L0C 128KB 与 L1 512KB 远小于 `W`） |

**定量 roofline（D-2）**：算术强度 `AI = 5·N·log2N / (2·N·elem_bytes) = 5·log2N/(2·8) = 3.75 FLOP/B`（c64）；9362 单 die 核→GM ≈1.06 TB/s（`CONST-ascend910-9362-profile`，2026-10-13 注记，主口径），vector fp32 峰值 ≈ 40 核×64 lane×1.65 GHz×2 ≈ 8.4 TFLOP/s → 脊点 ≈ 8 FLOP/B。`AI=3.75 < 8` → **内存受限**，访存/遍数为主导项（印证 R3 重访存口径）。

**调研结论**：**选定算法族 = Stockham autosort DIF radix-2（#4）+ LUT twiddle（#5），整行常驻 UB 单趟融合**，radix-4/8 融合（#3/O4）作为可选 refinement。关键依据：R3 表中 #4 的 GM 访存 `B·2·N·elem`（理想 2 趟）与「无中间 GM 往返」为所有候选最优；R4 中 #4 的计算全为连续段逐元素（Vector 亲和 ✓），且**消除 bit-reversal**（NPU 缺跨步 gather，TRAP-C10 佐证）——这是相对基线 #0 的决定性结构优势。与基线的结构差异一句话：**DIT+bit-reversal → DIF 自排序 + 整行 UB 单趟**。

调研范围说明（候选表来源）：对照 `algorithm-research.md` §5 与本 skill `references/algorithm-candidates.md`（kb_search `"fft complex"` 无命中，记录「已检索未命中」）、本仓 `examples/`（无 FFT 先例）、源算子族（Cooley-Tukey/radix 融合/LUT 的直接证据，§0.3/§0.4）、外部已知算法（Stockham autosort、四步/六步分解、DFT-as-matmul——按模型知识取算法思路，来源见下）。**互联网检索**：本次设计环境 `webfetch` 超时不可用（已尝试，见 §9.3），按模型知识 + 本地源继续，不阻塞；Stockham 索引格式在本地未命中文档，故**以自建数值验证替代文献引用**（§1.6.1 full-FFT 行，n=2..64 与 `numpy.fft.fft` 一致）。

**源算法优化手段的意图承接**（§0.4 → 本设计，杜绝静默丢弃）：
- O1（LUT）→ **保留**（#5，且因 NPU 无标量三角成为必需）；
- O2（SMEM 融合）→ **等价替换 + 强化**（整行 UB 单趟，GM 往返 `O(log n)` → 2）；
- O3（bit-reversal 融合装载）→ **重新设计为无重排**（Stockham autosort，消除 gather）；
- O4（radix-8/4 融合）→ **意图由单趟 UB 融合承接**；radix 融合降为可选 §1.6.3 refinement；
- O5（per-stage kernel）→ **舍弃**（单趟设计无 stage kernel）；
- O6（即时三角）→ **重新设计为全 LUT**（NPU 不可用标量三角）；
- O7（二维 grid）→ **重新设计为一维 grid=B**；
- O8（threads/block_size）→ **舍弃**（NPU 无 threads）。

**设计期估算**（dispatch 代表 workload `[64,4096]` c64，`B=64`，`N=4096`）：
`估算下界: ≈4.0 us（流量 3.96 + 发射 ~3.7 + 容量 0，引用 CONST-ascend910-9362-profile / CONST-vector-launch-overhead）`
- **流量项** = `B·2·N·8 B = 4.19 MB ÷ 1.06 TB/s ≈ 3.96 µs`（`CONST-ascend910-9362-profile` 核→GM 单 die 1.06 TB/s，2026-10-13；未重验注记适用）；
- **发射项** ≈ 3.7 µs：每行 `t·(N/2)·8` 实数 lane-op = `12·2048·8 = 196,608` lane-op ÷ 64 fp32 lane ≈ 3072 向量指令/行；按 1 instr/cycle@1.65 GHz ≈ 1.86 µs/行，64 行 / 40 核 × 1.6 波次 ≈ 3.7 µs —— **未文档化假设**：向量指令每周期 1 条、fp32 lane 宽 64（`CONST-vector-launch-overhead` 的 0.5µs/op 是 910B2C 大 tile 口径，不直接适用于本算子小向量，故改用指令数模型）；
- **容量项** = 0（无跨引擎往返；`CONST-store-fixpipe-gm-only` 不涉及）。
- 另：unbatched `[4096]` c64 —— 流量 0.062 µs、发射 ≈1.9 µs（单核、1/40 占用，见 §5.5 风险）。manifest 的 `[64,4096]` c128 因 fp64 向量元算缺失**不可实现**（CG-2026-0019）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（Stockham DIF radix-2 + LUT）。采纳项的等价论证与收益量化如下；**含采纳项，故产出并实跑 `verify_equiv.py`（结果表见下）**。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | 平凡 twiddle 特化 `w=1` | `t = (v_r+iv_i)·(1+0i)` → 4 mul + 2 add | `t = v`（**直接透传**） | 代数恒等 `v·1=v`；**机器验证 max 0 ulp**（等价） | 该 lane 省 4 mul+2 add（每个 `w=1` 蝶形）。DIF 各级 `i=0` 及 `h=1` 整级（`w≡1`）适用 |
| 2 | 平凡 twiddle 特化 `w=-1` | `t = v·(-1)` | `t = (-v_r, -v_i)`（**取反**） | 恒等 `v·(-1)=-v`；**机器验证 0 ulp** | 省 4 mul+2 add（`h=2` 级 `i` 非 0 的 `w=±i` 与 `w=-1` 混合形态） |
| 3 | 平凡 twiddle 特化 `w=±i` | `t = v·(0∓i)` | `w=-i: t=(v_i,-v_r)`；`w=+i: t=(-v_i,v_r)`（**交换+取反**） | 恒等 `v·(-i)=(v_i,-v_r)`、`v·i=(-v_i,v_r)`；**机器验证 0 ulp** | 省 4 mul+2 add。`h=2` 级 `w∈{1,-i}` 全覆盖 → **该级零实数乘** |
| 4 | twiddle LUT 降代换 | 内核内 `T.cos/sin`（NPU 不可用/域受限） | `w[i] = LUT[base+i]`（host fp64 建表 → fp32） | LUT = fp64 精确值 cast fp32；**机器验证 max 0.475 ulp**（全 stage，n∈{64,4096}） | 消除内核三角；访存仅 UB 内查表。**必需**（非纯收益） |
| 5 | 复数乘标准展开 | `(a+bi)(c+di)` | `(a·c−b·d) + i(a·d+b·c)`（4 mul+2 add） | 复数乘定义，精确恒等 | 基线形态 |
| 6 | **（评估后否决）Karatsuba 复数乘** | `4 mul + 2 add` | `3 mul + 5 add`（`k1=ac,k2=bd,k3=(a+b)(c+d)`；`re=k1−k2,im=k3−k1−k2`） | 代数恒等（浮点舍入略异） | **否决**：Ascend Vector 上 `vadd`/`vmul` 同为向量指令、吞吐相当，**op 数 8 > 6**，越多越亏（`docs/Tilelang.language/数学操作/T.vmul.md`/`T.vadd.md` 均为独立向量指令；`CONST-vector-launch-overhead` 按 op 数计时）——收益方向为负，故不采纳 |
| 7 | **（评估后否决）直接卷积/DCT 等其它变换** | — | — | — | 与 DFT 语义不等价（非恒等变换），非「等价优化」，不适用 |

**优化结论**：采纳 **4 项**（#1–#4，均为数学恒等或容差内等价，机器验证全 `EQUIV_PASS`）；#6 Karatsuba 因 Ascend 上 op 数增加而**否决**（等价论证成立但收益为负，故不采纳）；#5 为基线形态。优化后公式（供 §3.1）：**Stockham DIF 级 + 平凡 twiddle 分支特化（`w∈{1,-1,±i}` 免乘）+ 其余 twiddle 走 LUT + 标准复数乘 4mul2add**。无「不可优化」项——4 项采纳均有量化收益。

**D-1 等价性机器验证结果表**（`verify_equiv.py`，torch CPU fp64 参照，容差 4 ulp fp32；随机张量 + 角点值 `{0,±1,±0.5,±1e±30,±√2/2,±1e-8,±2,3.5}`；已实跑）：

| 采纳项 | max ulp 差 | 违反率 | 结论 |
|--------|-----------|--------|------|
| #1 `w=1`（skip mul） | 0.000 | 0.000000 | `EQUIV_PASS` |
| #2 `w=-1`（negate） | 0.000 | 0.000000 | `EQUIV_PASS` |
| #3 `w=-i`（swap+negate） | 0.000 | 0.000000 | `EQUIV_PASS` |
| #3 `w=+i`（swap+negate） | 0.000 | 0.000000 | `EQUIV_PASS` |
| #4 LUT(fp32) vs 内核 fp64 三角（全 stage，n∈{64,4096}） | 0.475 | 0.000000 | `EQUIV_PASS` |
| （整链完整性）Stockham radix-2 fp32 vs `torch.fft.fft`（n=4096） | abs err 6.83e-6 | — | `EQUIV_PASS`（< 1e-3） |

> 脚本：`examples/fft_c2c/_fft_c2c_kernel/verify_equiv.py`（本阶段实跑，退出码 0；Stage 2 检视重跑复核）。全部采纳项均 `EQUIV_PASS`，无未通过项。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（源/朴素） | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|----------------------|-------------|---------|--------------|
| 1 | bit-reversal 下标计算 | 源：每元素 `log2N` 次标量 `<<`/`&`/`>>` 循环（L115–L119） | **整段消除**：Stockham autosort 无 bit-reversal（§0.6 R3） | ✅ | — |
| 2 | 蝶形 twiddle 复数乘 | 标量 `t_r=v_r·tw_r−v_i·tw_i` 等（L163–L164） | `T.vmul`/`T.vsub`/`T.vadd` 向量链（实数）；平凡 twiddle 分支用 `T.vsub`/取反/`T.deinterleave` | ✅ | — |
| 3 | 蝶形加减 `u±t` | 标量 `u_r+t_r`/`u_r-t_r`（L166–L169） | `T.vadd(u,t)` / `T.vsub(u,t)` | ✅ | — |
| 4 | 内核内三角 `T.cos/sin(angle)` | 源：逐元素标量三角（L155–L156） | **整段消除**：twiddle 全取 LUT（§0.6 R5）；NPU 无标量三角、`vcos/vsin` 角度域超界（T.vcos.md §2.3） | ✅（改为查表向量乘） | — |
| 5 | 行搬入 / 搬出 | `T.copy` GM↔SMEM 分块 | `T.copy` GM↔UB（整行连续） | ✅ | — |
| 6 | stage 级调度循环 | 源：Python `for s in range(...)` 编译期展开（L139/L420） | Python `for s in range(t)` 编译期展开（`t=log2N` 为编译时常量，静态边界） | ❌（保留） | **tile 级顺序依赖**：级间 ping-pong 存在循环携带依赖（下一级读上一级输出），且级数为编译时常量 → 编译期展开、非逐元素标量循环；§6 不出现对应逐元素标量循环 |
| 7 | 蝶形下标/寻址（`g*2h+i`、`k`、LUT offset） | 标量索引算术 | 无独立逐元素循环：由 `T.Parallel(n_groups, h)` 的循环变量 + 仿射表达式承担，编译器生成向量地址 | ❌（保留） | **block/任务映射级索引计算**（仿射，非逐元素热点；不构成标量循环） |
| 8 | block → FFT 行映射（`batch row = cid`） | 源：grid y 轴映射 | `cid` 直接作为行号（每 block 少量标量） | ❌（保留） | **block 级索引 / 任务映射**（每 block 一个标量，非逐元素热点） |
| 9 | host 侧 LUT 建表（`angles`→`cos/sin`→cast） | 源 op 层 fp64 建表（L100–L108） | 保留 host 建表（**常量 materialization**，不改输入张量内容，合规） | ❌（host 元数据） | **host 侧常量表构建**（不在 kernel 内；输出为 LUT 常量输入，非输入张量内容改动，符合 ascend-constraints.md §4） |

**向量化结论**：逐元素计算（#1–#5）**已全部向量化**（#1/#4 为整段消除、#2/#3 用 v 前缀 API），替代方案 API 均有 docs 佐证（`T.vadd/vsub/vmul/vcos/T.deinterleave`，见 §3.2/§3.4）。保留 4 处非逐元素项（#6 编译期展开的级调度、#7 任务映射索引、#8 block 索引、#9 host 常量表），理由类别：**tile 级顺序依赖 / block 索引映射 / host 元数据**，逐项见上表；无一为「实现简单」类理由。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`[..., N]` 连续行），但**核内布局是自由变量**。本算子核内工作集 = 一行 `N` 个复数（实/虚各一段连续）。

**轴质量评分维度**（针对本算子）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | 段长 `h` 是否向量宽整数倍（fp32 ×4） | `h=2^k`：`h≥4` 时 ✓；`h=2,1` 尾 lane 浪费（见下） |
| 尾 lane 浪费率 | `(ceil(h/4)·4 − h)/h` 内层；`h=1` → 75–100% 浪费 | 仅末 2 级显著 |
| 累加链形态 | 蝶形内为元素级点运算（无串行累加维），连续段 → 向量友好 | ✓ |
| repack 代价 | 无重排（Stockham 连续段）+ ping-pong | 无 repack（0） |
| UB 容量影响 | 实/虚各 `N` fp32 + ping-pong = `4·N·4B` = 64KB（N=4096） | 见 §4.5 |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **核内 flat 行（实/虚各 `[N]`）+ ping-pong**（选定） | 每级连续段内层轴 `i∈[0,h)`（`h` 为编译时常量） | 无 | 基线；`h≥4` 全宽向量；末 2 级（`h=2,1`）效率降 | ✅ |
| 2 | 核内 flat + **radix-4/8 融合** | 融合后段长 `≥4h` | 无 | 增末级有效向量宽、减 ping-pong 趟数（`t→t/2` 或 `t/3`）；代价 = 更多寄存器/临时 | ✅（可选 refinement；与 #1 同布局） |
| 3 | 核内转置为 `[n_groups, h]`（`g` 连续）用于小 `h` 级 | 组轴 `g` | `T.transpose` 二轴交换链（实测 ~5.4µs，PL-1.1） | 使 `h=1` 级沿 `g` 向量化；代价 = 额外 UB 搬运 + 转置开销 | ❌（`h=1` 级可用 `T.deinterleave` 免转置，见下） |
| 4 | host permute 进出 | — | 两次 host 重排（~百 µs 级，PL-1.3） | repack 代价 ≫ 算子本体 | ❌（且 host 禁改输入内容；PL-1.3 实测净亏） |

> 迁移任务说明：源的并行轴选择（thread/warp、SMEM 分块）是**输入而非结论**（GPU thread 轴 ≠ NPU Vector 轴）。源用 `T.Parallel(threads)` 映射元素轴；本设计按 NPU Vector 语义独立评估为「连续段内层轴 `i`」（候选 #1），并对末级使用 `T.deinterleave`（源无此原语）。

**末 2 级（`h=2,1`）处理**：`h=1` 级的读 `A[2g]`、`A[2g+1]` 是**偶/奇解交织**——用 `T.deinterleave`（docs/`创建操作`…`shape操作`/`T.deinterleave.md`，fp16/fp32 ✓，`channel_nums=2`）把行拆成偶、奇两段后做 `vadd/vsub`，避免逐 lane 跨步访问。`h=2` 级读段长 2，可用 `T.deinterleave`（channel_nums=2）或直接 2 宽向量；且该级 twiddle ∈ `{1,-i}` → 平凡特化免乘（§1.6.1 #1/#3）。**此为设计期主选方案内的细化，非未实证常数裁决**（`T.deinterleave` 有文档佐证）。

**布局决策结论**：选定 **方案 #1（核内 flat 行 + ping-pong，向量轴 = 每级连续段内层轴 `i`）**，叠加 **方案 #2（radix 融合，可选）**；末 2 级用 `T.deinterleave` 处理。该结论同步落入 **§3.3 伪代码与 §6 循环结构**（蝶形内层向量维 = 连续段轴 `i`；buffer 形状 = `[N]` flat）。弃选 #3（转置，`T.deinterleave` 已免转置）、#4（host permute，代价 ≫ 收益）均有量化理由。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer。

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（蝶形 = 逐元素实数乘加 + 查表；无 matmul） | 不需 Cube/L0；Developer 的 `T.alloc_shared`（映射 UB）+ `T.alloc_fragment` 足够 |
| 归约 | 无归约（蝶形点运算），无 `T.reduce_*` 需求 | Developer 满足 |
| 数据搬运 | 仅 GM↔UB 两级（整行），无 L1/L0 | Developer `T.alloc_shared`/`T.copy` 满足 |
| 并行 | 行内 `T.Parallel` 向量化 + block 级 1 行/block | Developer `T.Parallel` 已验证 |
| 同步 | 单 kernel 内级间 ping-pong，自动同步即可 | Developer 自动同步 |
| 参考先例 | `flash_attn_npuir_dev.py`、`_logsumexp_kernel_tiled`（均 Developer + v 前缀 + alloc_shared/fragment） | Developer 已验证可行 |

用户/调度 prompt 指定编程模式 `developer`，与算子特征一致。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB 搬运缓冲，实/虚各 ping-pong 两份）；`T.alloc_fragment` 用于蝶形中间量（如需要）；`T.copy` 搬运 |
| 计算方式 | v 前缀 API（`T.vmul`/`T.vsub`/`T.vadd`/`T.deinterleave`）+ `T.Parallel` 元素级仿射索引 |
| 同步 | Developer 自动同步（无手动 `sync_block_set/wait`） |
| Kernel 启动 | `T.Kernel(B, is_npu=True)`（一维 grid；无 `threads`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 **优化后公式**（Stockham DIF 级 + 平凡 twiddle 特化 + 其余 LUT + 标准复数乘）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `lo = A[g·2h+i]`, `hi = A[g·2h+h+i]` | 读两个连续段（段长 `h`） |
| 2 | `w[i] = LUT[h-1+i]` | twiddle 查表（连续段）；`h=1`/`h=2` 走平凡特化（§1.6.1 #1–#3） |
| 3 | `dr = lo_r − hi_r`; `di = lo_i − hi_i` | 差值（`T.vsub`） |
| 4 | `B[g·h+i] = lo + hi` | 和（`T.vadd`） |
| 5 | `B[g·h+i+N/2] = (dr·w_r − di·w_i) + i(dr·w_i + di·w_r)` | 复数乘（`T.vmul`/`T.vsub`/`T.vadd`），写高位段 |
| 6 | 重复步骤 1–5 共 `t=log2N` 级，`h = N/2→1`，ping-pong | 全 stage 覆盖 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 行搬入 | `x[cid,:]→UB` | `T.copy(x_real[cid, 0:N], A_r)` | GM `(B,N)` slice → UB `(N,)` | Developer |
| 段读 | `A[g·2h+i]` | `T.Parallel(ng, h)` 内仿射索引读 | 循环变量 `g,i` 映射段内层轴 | Developer |
| twiddle 查表 | `LUT[h-1+i]` | `T.Parallel(ng, h)` 内读 `lut_real[lut_base+i]` | 连续段偏移 | Developer |
| 差值 | `lo−hi` | `T.vsub(hi, ...)` / 表达式 | 段向量 | Developer |
| 和 | `lo+hi` | `T.vadd(...)` | 段向量 | Developer |
| 复数乘 | `(dr+idi)(wr+iwi)` | `T.vmul` + `T.vsub` + `T.vadd` 链 | 段向量 | Developer |
| 平凡特化 `w=-i` | `(re,im)=(di,−dr)` | 交换 + `T.vsub` 取反（`h=2` 级） | — | Developer |
| 末级解交织 | 偶/奇分离 | `T.deinterleave(src, even, odd, channel_nums=2)` | `h=1` 级 `A[2g]`/`A[2g+1]` | Developer |
| 行搬出 | `UB→y[cid,:]` | `T.copy(A_r, y_real[cid, 0:N])` | UB → GM slice | Developer |
| block 网格 | 每 block 1 行 | `T.Kernel(B, is_npu=True) as (cid, _)` | 一维 grid | Developer |

### 3.3 计算伪代码

> 迁移规则 3：`T.Tensor` 的 dtype 作为第二位置参数传入（非 `dtype=` 关键字），避免 `name ... is not defined` false alarm。**无 `threads` 参数**（TRAP-threads-kwarg-noop）。

```python
@tilelang.jit(out_idx=[4, 5], target="npuir")   # y_real,y_imag 自动分配
def _fft_c2c_kernel(n, batch_size, dtype="float32"):
    t = int(math.log2(n))          # 级数（编译时常量）
    B = batch_size
    DT = dtype                     # float32（complex64 的实/虚分量）

    @T.prim_func
    def main(
        x_real: T.Tensor((B, n), DT),
        x_imag: T.Tensor((B, n), DT),
        lut_real: T.Tensor((n - 1,), DT),
        lut_imag: T.Tensor((n - 1,), DT),
        y_real: T.Tensor((B, n), DT),
        y_imag: T.Tensor((B, n), DT),
    ):
        with T.Kernel(B, is_npu=True) as (cid, _):
            # ping-pong 缓冲（实/虚各两份）
            A_r = T.alloc_shared((n,), DT); A_i = T.alloc_shared((n,), DT)
            Bc_r = T.alloc_shared((n,), DT); Bc_i = T.alloc_shared((n,), DT)

            T.copy(x_real[cid, 0:n], A_r)
            T.copy(x_imag[cid, 0:n], A_i)

            # ---- 编译期展开的 log2(n) 级 Stockham DIF ----
            for s in range(t):                 # Python range → 编译期展开（静态边界）
                h = 1 << (t - 1 - s)           # h = n/2, n/4, ..., 1
                ng = n // (2 * h)              # 组数
                lut_base = h - 1
                src_r, src_i = A_r, A_i
                dst_r, dst_i = Bc_r, Bc_i
                # 蝶形：内层轴 i（连续段）为向量化轴（§1.6.3）
                for g, i in T.Parallel(ng, h):
                    lo_r = src_r[g * 2 * h + i]; lo_i = src_i[g * 2 * h + i]
                    hi_r = src_r[g * 2 * h + h + i]; hi_i = src_i[g * 2 * h + h + i]
                    dst_r[g * h + i] = lo_r + hi_r
                    dst_i[g * h + i] = lo_i + hi_i
                    if h == 1:                 # w = 1（平凡特化，无乘）
                        dr = lo_r - hi_r; di = lo_i - hi_i
                    else:
                        wr = lut_real[lut_base + i]; wi = lut_imag[lut_base + i]
                        dr = lo_r - hi_r; di = lo_i - hi_i
                    dst_r[g * h + i + n // 2] = dr * wr - di * wi
                    dst_i[g * h + i + n // 2] = dr * wi + di * wr
                A_r, A_i, Bc_r, Bc_i = dst_r, dst_i, src_r, src_i   # 交换 ping-pong

            # 末次交换后结果在 A_r/A_i（Python 层可静态判定）
            T.copy(A_r, y_real[cid, 0:n])
            T.copy(A_i, y_imag[cid, 0:n])

    return main
```

> 说明：`h=2` 级可进一步特化（`w[0]=1, w[1]=-i` → `T.deinterleave` + 取反，免乘）；`h=1` 级用 `T.deinterleave` 替代等价的 `T.Parallel(ng,1)` 以取得偶/奇连续段。此二细化属 §1.6.3 主选方案内的可选 refinement（不改语义）。伪代码展示主选方案核心结构；`if h == 1:` 为 Python 编译期分支（非运行时控制流）。

### 3.4 API 可行性确认

> 验证方法：`tilelang/language/__init__.py` 导出表 + `python3 -c "hasattr(T, ...)"` 运行时验证（本阶段已执行，全部 `True`）+ docs + NPU 示例三重交叉。

| API | 导出验证 | 文档路径 | NPU 示例/测试佐证 | 备注 |
|-----|---------|---------|------------------|------|
| `T.Kernel(grid, is_npu=True)` | ✅ | — | 全部 NPU 示例 | 一维 block；**无 threads** |
| `T.alloc_shared` | ✅ | docs/Tilelang.language/内存操作/T.alloc_shared.md | logsumexp tiled | Developer 映射 UB |
| `T.copy` | ✅ | docs/Tilelang.language/内存操作/T.copy.md | 全部 NPU 示例 | 整行连续 slice |
| `T.Parallel` | ✅ | — | 全部 NPU 示例 | 元素级向量化 |
| `T.vadd` | ✅ | docs/Tilelang.language/数学操作/T.vadd.md | flash_attn_npuir_dev | fp16/fp32（**无 fp64**，CG-2026-0019） |
| `T.vsub` | ✅ | docs/Tilelang.language/数学操作/T.vsub.md | flash_attn_npuir_dev | 同上 |
| `T.vmul` | ✅ | docs/Tilelang.language/数学操作/T.vmul.md | flash_attn_npuir_dev | 同上 |
| `T.deinterleave` | ✅ | docs/Tilelang.language/shape操作/T.deinterleave.md | `testing/npuir/shape_manipulation_ops/test_slice_deinterleave_dev.py` | fp16/fp32/整型 ✓，fp64 ✗；`channel_nums=2` |
| `T.transpose`（仅列作弃选候选） | ✅ | docs/Tilelang.language/创建操作/T.transpose.md | `test_slice_transpose_dev.py`；PL-1.1（~5.4µs） | 未采纳（§1.6.3 #3） |
| `T.cast` | ✅ | builtin | layer_norm | 标量/元素 cast |
| Python `range`（级循环） | ✅ | — | — | 编译期展开，静态边界 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 涉及（源为二维 grid） | 展平为一维 `T.Kernel(B)`（§0.6 R2） |
| 部分 GPU API 不可用 | 涉及 | `T.cos/sin` 标量→ LUT；`threads` 移除；`T.Parallel` 替代 thread 级并行 |
| GEMM 要求 M,N 为 block 整数倍 | 不涉及 | 无 GEMM |
| L0C 容量上限 | 不涉及 | 无 Cube |
| 分核策略（物理核数适配） | 涉及 | 见 §5.5（物理 40 核实查） |
| 复数 dtype 缺失 | 涉及 | 实/虚分解 fp32（§0.6 R1）；complex128 不可实现（CG-2026-0019） |
| fp64 向量元算缺失 | 涉及 | complex128 路径不可实现（如实披露） |

#### 3.5.2 参考实现差异说明（GPU → NPU）

| 差异项 | 参考实现（GPU/源） | 本项目（Ascend） | 转换方案 |
|--------|-------------------|-----------------|----------|
| Kernel 维度 | `T.Kernel(smem_blocks, B, threads=512)` 二维 | 一维 | `T.Kernel(B, is_npu=True)`（§0.6 R2） |
| threads | `threads=512` 控 SMEM 覆盖 | 无 threads 概念 | 移除（TRAP-threads-kwarg-noop） |
| stage 结构 | DIT + bit-reversal gather 装载 | Stockham autosort DIF，无重排 | §0.6 R3 |
| stage 覆盖 | 阶段一 SMEM + 阶段二 per-stage kernel | 整行 UB 单趟 | §0.6 R4 |
| twiddle | 小 stride 即时 `T.cos/sin`，大 stride LUT | 全 LUT | §0.6 R5 |
| 复数 dtype | 直接 `complex64/128` | 实/虚 fp32 分解 | §0.6 R1 |
| stage 循环 | Python `for s in range(range)` 展开 | 同（编译期展开） | 保留 |
| radix 融合 | radix-8/4 LUT stage | 可选 refinement | §1.6.3 #2 |

#### 3.5.3 本项目同类实现参考

**必须列出**：本项目 `examples/` 与 `testing/npuir/` 中最相似实现。**本算子为 FFT（谱类），本仓无同族先例**（按 AGENTS.md「无近模板则显式声明后构建最小新 kernel」）。

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/flash_attention/flash_attn_npuir_dev.py` | 高（v 前缀向量链模板） | Developer 模式 `alloc_shared`/`alloc_fragment` + `T.vmul/vsub/vadd` 链、ping-pong 数据流、自动同步 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/DESIGN.md` | 高（迁移 DESIGN 体例 + 向量化模式） | 迁移 §0 体例、GPU→NPU 差异表、`T.Parallel`/v 前缀/UB 预算/T.copy 切片、`T.serial` 静态边界 |
| `examples/norm/layer_norm.py` | 中（分块 + fp32 提升 + 逐元素） | `alloc_shared`+`alloc_fragment`+`T.Parallel`+`T.cast` |
| `testing/npuir/shape_manipulation_ops/test_slice_deinterleave_dev.py` | 中（末级解交织） | `T.deinterleave` 用法（偶/奇分离，`channel_nums=2`） |
| `testing/npuir/parallel_ops/` | 中（`T.Parallel` 仿射索引/条件） | 元素级向量化与循环变量索引模式 |

> 无近模板 → 按 AGENTS.md 构建最小新 kernel；FFT 结构（蝶形 + LUT）为本仓首例，风险见 §9。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x_real`/`x_imag` | `(B, N)` | float32（对 complex64） | 复输入实/虚分量；`B` 动态（前导维乘积），`N` 静态（2 的幂） |
| `lut_real`/`lut_imag` | `(N-1,)` | float32 | twiddle 常量表（host 建，随 dtype 缓存） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y_real`/`y_imag` | `(B, N)` | float32 | 结果实/虚分量；op 层 `torch.complex` 合并为 `[..., N]` complex64 |

> 输出 shape/dtype 与输入同（§0.2）。c128 路径不可实现（CG-2026-0019）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 | 生命周期 |
|-----------|-------|-------|----------|------|----------|
| `A_r`/`A_i` | `(N,)` | float32 | UB（alloc_shared） | ping-pong 行缓冲 A | 整 kernel |
| `Bc_r`/`Bc_i` | `(N,)` | float32 | UB（alloc_shared） | ping-pong 行缓冲 B | 整 kernel |
| `lut`（常量） | `(N-1,)`×2 | float32 | GM→UB（按需载入） | twiddle 表（见 §4.5） | 可常驻或 per-stage 载入 |

> 输出不另设 UB 缓冲：最后一轮 ping-pong 的目标缓冲直接 `T.copy` 回 GM（就地）。

### 4.4 内存搬运路径

```
GM[x_real/imag (B,N)] --T.copy(整行)--> UB[A_r/A_i]
  [t = log2N 级: UB[A] --蝶形(vmul/vsub/vadd)+LUT查表--> UB[B]; ping-pong A<->B]
UB[out_r/out_i] --T.copy(整行)--> GM[y_real/imag (B,N)]
（op 层: torch.complex(y_real,y_imag) → [..., N] complex64）
```

**关键约束**：仅 GM↔UB 两级（无 L1/L0）；GM 流量 = 1 读 + 1 写（roofline 最优）；LUT 可常驻 UB（§4.5）或 per-stage 连续段载入（额外 ~0.5 趟 GM 读）。

### 4.5 UB 内存预算

以 9362/A2/A3 UB = 192KB（196608B；`CONST-capacity-910B2C` 容量值 9362 一致）为基准，`N=4096` fp32：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `A_r` | (4096,) | fp32 | 16384 |
| `A_i` | (4096,) | fp32 | 16384 |
| `Bc_r` | (4096,) | fp32 | 16384 |
| `Bc_i` | (4096,) | fp32 | 16384 |
| `lut_real` | (4095,) | fp32 | 16380 |
| `lut_imag` | (4095,) | fp32 | 16380 |
| **总计（LUT 常驻）** | | | **≈ 98300 / 196608 (51%)** |

> **膨胀余量**：显式 alloc 结构下 BishengIR 实际需求 ≈ 手工 × 1.10–1.12（`CONST-capacity-910B2C`，910B2C 实测；9362 未重验）→ ≈ 108–110KB < 192KB ✓。若走 auto-multi-buffer 的 ~1.7× 悲观口径则 ≈167KB 仍 < 192KB（本设计无 `T.Pipelined`，预期不触发 multi-buffer 膨胀）。**保守备选**：LUT 不常驻，改为 per-stage 连续段载入（峰值仅需 max stage LUT `N/2` 复 ≈16KB），代价 = 额外 ~0.5 趟 GM 读（§9 风险）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 说明 |
|--------|----------|-----------|------|
| `B`（batch） | `T.Kernel(B)` grid（编译时工厂参数 `batch_size`；不同 batch 触发重新 JIT） | ≥1 | 前导维乘积；unbatched 时 =1 |
| `N` | 静态（per-kernel build） | 2 的幂 | `t=log2N` 编译时常量 |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[4, 5],     # y_real, y_imag 自动分配
    target="npuir",
    # 注意：不使用 pass_configs TL_ENABLE_FAST_MATH（源 GPU 配置；NPU 路径按默认）
)
```

> `out_idx` 双输出在 NPU 的调用约定须按 TRAP-outidx-call-convention（`pattern-library/traps-runtime.md`）在 wrapper 调用点逐个核对签名；若双输出自动分配有异常，回退为**显式输出参数**形态（wrapper `torch.empty` 分配，见 CG-2026-0016 先例）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector。

**判定依据**: 算子仅含逐元素实数乘加与常量查表（蝶形），无 matmul、无归约；`T.alloc_shared`（UB）足够，不涉及 L1/L0。

### 5.2 Block 划分

```python
block = 1 行（一个完整 FFT 沿最后一维）   # 行内全 stage 串行；整行常驻 UB
grid  = B                                  # 每个 block 处理一行（一个独立 1D FFT）
```

**选择理由**：`N=4096` fp32 整行（实/虚各 16KB）可常驻 UB（§4.5）→ 单 block 完成全部 `log2N` 级，GM 仅 1 读 1 写。行间完全独立 → block 级并行。**不做行内分块**：DIF 首级 stride `N/2` 跨整行，分块会引入跨 block 依赖。

### 5.3 约束分析

- **对齐**：`N` 为 2 的幂，段长 `h=2^k` 为编译时整数，内层轴整除良好（`h≥4` 时 fp32 ×4 满宽）；UB 32B 对齐由编译器管理。
- **UB 容量**：≈96KB < 192KB ✓（§4.5）。
- **L0 容量**：不适用（无 Cube）。
- **静态循环边界**：级循环 `range(t=log2N)` 编译期展开（静态）；`ng=h` 均为编译时常量。
- **非整除**：`N` 静态且为 2 的幂 → 无尾块；`B` 任意由 grid 承接（不影响整除）。

### 5.4 注意事项

- **无 GEMM 非整除**问题（无 GEMM）。
- `N` 变化即触发重新 JIT（工厂参数）；支持 `N ∈ {64,128,…,4096}` 等 2 的幂。
- `B` 变化亦触发重新 JIT（源用 `functools.lru_cache` 工厂缓存，NPU 侧建议沿用）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。本算子为**纯 Vector** → 物理核数取 `get_aicore_num() * 2`。

- **物理核数（实查 + 记录）**：查询代码 `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()`，本机**实际返回值 = 20**（Ascend910_9362，与 `CONST-aicore-910_9362` 一致）→ **纯 Vector 物理核数 = 20 × 2 = 40**。（本阶段实跑已记录，见返回摘要。）
- **逻辑核数**：`num_logical_kernels = ceil(B/1) = B`（每 block 处理一行；无第二分块维）。
- **规模判定与分核方案**（四态）：
  - **B ≤ 40（含 unbatched `B=1`）**：逻辑核数 ≤ 物理核数 → **无需适配**。依据：每行独立、block 数不超物理核。**但 `B=1` 时核利用率仅 1/40 → 性能风险**（非适配问题），见 §9 与 §1.6.3 备选。
  - **B = 64（manifest `[64,4096]`）**：**无法对齐兜底**。64 = 2⁶ 与物理 40 = 2³·5 数学上无法整数倍对齐（standard §1 要素③）；取 `grid = B = 64`，运行时调度为 **1.6 波次**（先跑 40、再跑 24），相对理想 1 波次**临界路径 +60%、末波 16 核闲置**（利用率 64/80 = 80%）。替代 = 固定 `grid=40` persistent（24 核串行 2 行 / 16 核 1 行）——临界路径同为 2 波、无净收益且引入核内串行，故**选 `grid=B`**（更简、运行时调度）。
  - **极大规模**：本算子逻辑核数 = `B`（前导维乘积），实际 batch 有限（manifest ≤64），不涉及核内 `T.serial` persistent；如未来 `B≫40`，可固定 `grid=40` + 核内 `T.serial(T.ceildiv(B-kid,40))`（边界由静态 `B` 决定）。
  - **备选（提升 `B=1` 占用）**：四步分解（§1.6.0 #8）→ 子 FFT 并行度 `N1`（如 `N=4096=64×64` → 64 路由）；列为 Stage 4 裁决项（§9）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素蝶形已按 §1.6.2 全部向量化；下表含 block 级/级调度循环与元素级向量化。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| batch（行） | block 级并行 | `T.Kernel(B, is_npu=True)` | 每 block 一行，行间独立 |
| stage 级（`h=N/2→1`） | **编译期展开**（Python `range(t)`） | Python `for s in range(t)` | 级间 ping-pong 顺序依赖；级数 `t=log2N` 编译时常量（静态边界） |
| 蝶形元素 | 向量化并行 | `T.Parallel(ng, h)`（内层轴 `i` 向量化，§1.6.3） | 段内连续、仿射索引、无依赖 |
| 末级解交织（`h=1`） | 向量化 | `T.deinterleave(..., channel_nums=2)` | 偶/奇连续段分离 |

### 6.2 循环伪代码

```python
with T.Kernel(B, is_npu=True) as (cid, _):
    T.copy(x_real[cid, 0:N], A_r); T.copy(x_imag[cid, 0:N], A_i)
    for s in range(t):                      # 编译期展开
        h = 1 << (t-1-s); ng = N // (2*h)
        for g, i in T.Parallel(ng, h):      # 内层 i = 向量轴
            # 读 lo/hi（连续段），查 LUT，写 dst 的和段与差段（§3.3）
            ...
        A_r, A_i, Bc_r, Bc_i = Bc_r, Bc_i, A_r, A_i   # ping-pong（Python 层）
    T.copy(A_r, y_real[cid, 0:N]); T.copy(A_i, y_imag[cid, 0:N])
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：级间为 ping-pong 顺序依赖（下一级读上一级输出），无跨级并行；行搬入/搬出为单次整行连续搬运（占比小）。若需重叠，可对**行间**做多行/核或双缓冲，但受 UB 容量限制（§4.5），列为 Stage 4 可选。

### 6.4 尾块处理

- `N` 静态且为 2 的幂 → **无列尾块**。
- `B` 由 grid 直接承接，无「尾块」概念。
- 段长 `h=1` 的末级用 `T.deinterleave` 处理，避免逐 lane 跨步（§1.6.3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）。

### 7.2 同步点说明

Developer 模式下编译器自动插入同步，无手动 `T.sync_block_set/wait`。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy`(GM→UB) 后 | 自动 | MTE2→V 拓扑自动 barrier |
| 每个 stage 的 `T.Parallel` 蝶形后 | 自动 | 级间 ping-pong 需 stage 内写完成（下一级读）；编译器按 UB 读写依赖插 barrier |
| `T.copy`(UB→GM) 前 | 自动 | V→MTE3 拓扑自动 barrier |

### 7.3 pass_configs 配置

无特殊 `pass_configs`。Developer 模式默认编译流水线；源 GPU 的 `TL_ENABLE_FAST_MATH`/`-O3` 为 CUDA 侧配置，NPU 路径不套用。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源仓测试基准 `torch.fft.fft(x, dim=-1)`），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch

def golden_fft_c2c(x: torch.Tensor) -> torch.Tensor:
    """沿最后一维的一维 C2C FFT，等价 torch.fft.fft(x, dim=-1)。

    输入 x: [..., N] complex64/complex128，N 为 2 的幂；输出同 shape/dtype。
    对应源仓 tests/ops/test_fft.py::FFTTest.ref_program。
    """
    return torch.fft.fft(x, dim=-1)
```

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_fft.py`：

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| complex64 | 1e-4 | 1e-4 | fp32 分量；本设计全程 fp32，累加分组与源不同但容差内 |
| complex128 | 1e-8 | 1e-8 | **NPU 不支持**（CG-2026-0019）——L0 不覆盖 |

### 8.3 L0 门槛测试计划

| 编号 | Shape | dtype | 验证目标 | 容差 |
|------|-------|-------|----------|------|
| L0-1 | `[64]` | complex64 | manifest smoke（单行最小 N） | 1e-4 |
| L0-2 | `[4096]` | complex64 | manifest `fft-4k-c64-unbatched`（unbatched，单 block） | 1e-4 |
| L0-3 | `[64,4096]` | complex64 | manifest `fft-4k-c64-b64`（batched，grid=64） | 1e-4 |
| L0-4 | `[4,64]` | complex64 | 多前导维展平（batch 语义） | 1e-4 |
| L0-5 | `[1,4096]` | complex64 | batch=1 边界 | 1e-4 |
| L0-6 | `[2,4,4096]` | complex64 | 多前导维 + 大 N | 1e-4 |
| L0-7 | `[64,4096]` | complex128 | **预期不覆盖**（NPU 不支持；如 op 层拒绝则记 skipped） | — |

**通过条件**：L0-1~L0-6 全部通过 atol/rtol；L0-7 如实标记不支持。完整分层套件（L1 功能/L2 边界/Boundary）交由 `tilelang-op-develop`。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 | 状态 |
|------|------|----------|------|
| **complex128 不支持**（无复数 dtype + 无 fp64 向量元算） | manifest c128 workload 不可实现 | 如实披露（CG-2026-0019）；op 层对 c128 报错或标记不支持；complex64 走实/虚分解 | ✅ 已登记 |
| **unbatched `B=1` 低占用**（1/40 核） | 延迟受限、吞吐低 | 主选仍 `grid=B`；备选四步分解（§1.6.0 #8）→ 子 FFT 并行度 64，留 Stage 4 裁决 | 风险 |
| **UB 预算紧**（LUT 常驻 ≈96KB） | 若编译膨胀超预期 | 备选：LUT per-stage 载入（峰值 16KB），代价 ~0.5 趟 GM | 风险 |
| **末级小 `h` 向量效率**（`h=2,1`） | 尾 lane 浪费 | `T.deinterleave` + 平凡特化（§1.6.3）；radix 融合可选 | 已缓解 |
| **`out_idx` 双输出调用约定** | wrapper 调用点签名 | 按 TRAP-outidx-call-convention 核对；异常时回退显式输出（CG-2026-0016 先例） | 待实现验证 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 使用 `T.Kernel(..., threads=)` | 照抄源 | 参数无效（可能静默） | 移除 threads（TRAP-threads-kwarg-noop） |
| 内核内 `T.cos/sin` 标量三角 | 照抄源小 stride 段 | 不可用/段错误/域超界 | 全 LUT（VP-2026-0140、T.vcos.md §2.3） |
| bit-reversal gather 装载 | 照抄源 P1 | 跨步 gather 缺失 → 错误/慢 | Stockham 无重排（TRAP-C10） |
| 复数 dtype 直接建 `T.Tensor(..., "complex64")` | 直接照抄 op 契约 | codegen `Cannot convert type` | 实/虚 fp32 分解（CG-2026-0019） |
| `TL_ENABLE_FAST_MATH`/`-O3` 照抄 | 照抄源 jit | 非 NPU 配置 | 移除 |

### 9.3 特殊场景处理

- **非整除**：不适用（`N` 为 2 的幂静态）。
- **极小 shape**：`[64]` smoke 走同结构（`t=6`）。
- **`B` 动态**：触发重新 JIT；建议沿用源 `functools.lru_cache` 工厂缓存。
- **互联网检索降级**：本设计尝试 `webfetch` 获取 Stockham 文献（Wikipedia Stockham_FFT）**超时不可用**，按模型知识 + 本地源继续，并以**自建数值验证**（`verify_equiv.py` full-FFT 行，n=2..64 对照 `numpy.fft.fft`）替代文献引用——如实记录，不阻塞。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/fft_c2c/_fft_c2c_kernel/
├── DESIGN.md          # 本设计文档
├── verify_equiv.py    # D-1 等价性机器验证（已实跑，全 EQUIV_PASS）
└── RETROSPECTIVE.md   # 任务复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档（含 §0 迁移分析、§1.6 算法调研与优化） |
| `verify_equiv.py` | ✅ 已完成（实跑通过） | 数学等价优化验证 |
| `_fft_c2c_kernel.py` | ⬜ 待实现 | 算子实现（Stage 3） |
| `test__fft_c2c_kernel.py` | ⬜ 待实现 | 测试（Stage 3） |

### 10.3 命名规范

- 项目目录：`fft_c2c`；算子目录：`_fft_c2c_kernel`；实现文件：`_fft_c2c_kernel.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现 + L0 精度比对（§8.3）

---

## 附录 A：性能反馈（Stage 4 追加，2026-10-17）

> 本附录由 conductor 按 `[DESIGN_LIMIT]` 路由「附录补记」追加，**只增不改**既有章节。完整反馈见 `perf_opt/perf_feedback.md`。

**结论**：§1.6.2「逐元素计算（#1–#5）已全部向量化」的断言被实测**证伪**——Stage 4 实测该 kernel 的实际向量化率为 `aiv_vec_ratio = 1.2e-04`（`aiv_scalar_ratio = 0.9988`，即 **99.9% 标量执行**）。FFT 蝶形网络在本工具链上**未被向量化**；717.68µs（4k-unbatched）的根因是**标量吞吐墙**，而非带宽/算力墙。

**根因（编译器能力缺口）**：TileLang npuir 向量化 pass `NpuLoopVectorize` 的 `VectorizeSingleStatement`（`src/transform/npu_loop_vectorize.cc:1184`）要求循环体恰好一条 `BufferStore` 且 store 索引为裸 `Var`/常数；FFT 蝶形的跨段仿射索引（`g*h+i`）不满足该条件，而改写所需的 **UB buffer 别名 / view（`T.view` / `T.copy` / `Buffer(shape,dtype,src.data,scope=...)`）在本工具链上全线不可用**（编译期 SSA / `GetVarValue` 失败；`T.reshape` 实为 `npuir_reshape` 算子、非别名）→ 无法将蝶形拆成「单-store、纯 Var 索引」形式来解锁向量化。

**后果**：① 本轮 Stage 4 **无可采纳优化**（final 与 baseline 内容等价，时延在噪声内：717.68→719.53µs / 1498.32→1497.34µs）；② 结构性加速估计 **≥2x**（另一实测佐证：同一 kernel batch=64 时 23.4µs/行 vs unbatched 717.7µs/行 = **30.7× 核占用差**）；③ 已按 `AGENTS.md`「Compiler Capability Gap Rule」登记 `.agents/evolution/capability-gaps.md`。

**建议路由**：附录补记（本次已执行）。`scenario=optimize` 无 Stage 1，不适用「设计修订」；该地板为工具链能力缺口，非 tiling/参数可解。

