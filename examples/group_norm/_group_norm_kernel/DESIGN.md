# _group_norm_kernel 算子设计文档

> 迁移任务（harness 模式）：`GroupNormFwdOp` 主算子，kernel 函数 `_group_norm_kernel`。
> GPU 源算子（本函数迁移的精确源）：`/workspace/TileOPs-gpu/tileops/kernels/norm/group_norm.py` 第 45 至第 121 行（`@tilelang.jit(out_idx=[3])` 工厂 + `@T.prim_func main(x, weight, bias, y)`）。
> Stage 0 提取件（wrapper 的导入契约源）：`examples/TileOPs/tileops/kernels/norm/group_norm/_group_norm_fwd_kernels.py`；GPU op 层：`/workspace/TileOPs-gpu/tileops/ops/norm/group_norm.py` 的 `GroupNormFwdOp.forward`；NPU op 层：`examples/TileOPs/tileops/ops/norm/group_norm.py`。
> 工具链版本戳：tilelang 0.1.2+9307929320dc51c9d798d6a827cea8d4bceef3f8（dev root build，2026-10-15）+ CANN 9.0.0 + Ascend910B3（npu-smi 26.1.1）。
> 范围：仅迁移 affine 主算子 `GroupNormFwdOp`；同文件的 `_group_norm_no_affine_kernel` / `GroupNormNoAffineKernel` / `GroupNormNoAffineFwdOp` 不在范围（其语义与代码不得引入）。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

**数学语义**（GroupNorm 前向，affine 主路径）：输入 `x` 形状 `(N, C, *spatial)`，把它按 `N * num_groups` 行、每行 `D = (C / num_groups) * spatial_size` 元素重排为二维矩阵，然后**逐行**做归一化：

$$
\mu_{r} = \frac{1}{D}\sum_{j<D} x_{rj},\qquad
v_{r} = \frac{1}{D}\sum_{j<D}\left(x_{rj}-\mu_{r}\right)^2,\qquad
\text{rstd}_{r} = \frac{1}{\sqrt{v_{r}+\varepsilon}}
$$
$$
y_{rj} = \left(x_{rj}-\mu_{r}\right)\cdot \text{rstd}_{r}
$$

（kernel 层到此为止；op 层随后对归一化结果施加 per-channel affine：元素所在通道 `c` 的缩放 `w_c` 与平移 `b_c`，即 `y · w_c + b_c`。）

同时方差不做 Bessel 修正（ddof = 0，即除以真实 `D`，不是 `D-1`）；`eps` 加在方差上（manifest `params.eps` 默认 1e-5）。

**affine 分层（关键，不得折叠）**：源码把 affine 分成两层——

| 层 | weight/bias 形状 | 由谁施加 |
|----|------------------|----------|
| kernel 层（`main(x, weight, bias, y)`） | 行广播 `(D_padded,)` | kernel 内逐元素 `*weight[j] + bias[j]` |
| op 层（`GroupNormFwdOp.forward(x, weight, bias)`） | per-channel `(C,)` | op 层在 kernel 外做 `y * weight.reshape((1,C)+(1,)*len(spatial)) + bias.reshape(...)` |

`GroupNormFwdOp.forward` 调 kernel 时**只传 x**（`y_2d = kernel(x_2d)`，见 GPU op 层第 237 行）——此时 `GroupNormKernel.forward` 走 `weight is None` 分支，用 `_identity_affine_for(device)` 生成 `(D_padded,)` 的**单位权重 / 零偏置**。也就是说：**在本迁移的唯一使用路径（GroupNormFwdOp）上，kernel 层 affine 恒为单位元**，真正的 affine 由 op 层用 per-channel `(C,)` 施加。`GroupNormKernel` 的行广播 affine 能力是它与 InstanceNorm 共享的历史接口（源文件注释「shared by GroupNorm and InstanceNorm」），本任务的 manifest（`examples/TileOPs/tileops/manifest/norm.yaml`）只含 `AdaLayerNormFwdOp` 与 `GroupNormFwdOp` 两个条目，无 InstanceNorm 或无 affine 变体。

**规约语义**：规约轴 = 行内全部 `D` 个元素；**行间完全独立、无跨行依赖**（这是行块分核的结构基础）。累加在 **fp32** 域完成（源码 `x_f32` fragment 上 `T.reduce_sum`），采用**中心化两遍式**：先 `S1 = Σx` 求 mean，再对 `(x − mean)²` 求 `S2` 得方差（不是单遍矩式 `E[x²] − (E[x])²`）。

**dtype 语义**：fp32 中转链路——`x_local`（输入 dtype）逐元素 `T.cast(..., "float32")` 上cast，全程 fp32 参与统计与仿射，最后**单次舍回**输入 dtype（`x_local[i, j] = <fp32 表达式>` 隐式截断）。与 golden `F.group_norm(x.float(), ...).to(x.dtype)` 的「fp32 opmath + 单次舍回」一致。

**边界语义**：
- `M % block_m ≠ 0`（行尾）：kernel 内**没有**行边界谓词——`T.copy(x[pid_m*block_m, 0], shared_buf)` 是整块 `(block_m, D_padded)` 拷贝，越界由 **`block_m=1` 的结构性默认**规避（`M % 1 == 0`；非 affine 变体的 autotune 路径才另用 `_M_BLOCK_ALIGN = 16` 把 M 对齐）；
- `D` 非 256 对齐：`D_padded = align_up(D, 256)`，尾列零填充参与规约（零不贡献 `S1`），方差里显式减去填充贡献 `pad_count · mean²`，输出只取真实列（wrapper 侧 `y[:, :D]` 截断）；
- 输出 shape = 输入 shape（manifest shape_rules `output.shape == x.shape`），非 in-place；
- NaN/Inf：无特殊处理，按公式自然传播（全 inf 行 → mean = inf → 输出 NaN）；
- 约束：`x.ndim >= 2`；`C % num_groups == 0`；dtype ∈ {float32, float16, bfloat16}。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓测试基准 `F.group_norm(x.float(), ...)`），不复刻 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `x` | 输入 | `(M, D_padded)` | float16 / bfloat16 / float32 | kernel 级 2D 视图；`D_padded = align_up(D, 256)`，`M = N*num_groups` |
| `weight` | 输入 | `(D_padded,)` | same_as(x) | **行广播** affine 缩放；GroupNormFwdOp 路径恒为单位权重 |
| `bias` | 输入 | `(D_padded,)` | same_as(x) | **行广播** affine 平移；GroupNormFwdOp 路径恒为零偏置 |
| `y` | 输出 | `(M, D_padded)` | same_as(x) | `out_idx=[3]` 自动分配；wrapper 截断为 `(M, D)` |

输出 shape 从源码确认：`main` 的 `y: T.Tensor[(M, D_padded), dtype]` + `register_fake` 返回 `torch.empty((M, D_padded), ...)`（GPU 源第 139 至第 142 行）——**无转置、无布局变化**；op 层最终输出 `(N, C, *spatial)` 与输入同形（`source_output_shape` 已核对一致）。

### 0.3 实现算法解读（怎么算）

**源码结构**（GPU，工厂 + `lru_cache`）：`_group_norm_kernel(M, D, eps, dtype)` → `D_padded = _align_up(D, ALIGNMENT=256)`、`pad_count = D_padded − D` → `@tilelang.jit(out_idx=[3])` 内 `_func(block_m, threads)` → `@T.prim_func main(x, weight, bias, y)`。

**计算步骤分解**（源码每个计算语句均已归入下表，无遗漏；无臆造步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| L1 | 装载：`T.copy(x[pid_m*block_m, 0], shared_buf)`（整块 `(block_m, D_padded)`，SMEM 中转）→ `T.copy(shared_buf, x_local)` | GM x | `shared_buf`、`x_local` | 输入搬运 |
| L2 | fp32 上cast：`x_f32[i,j] = T.cast(x_local[i,j], "float32")`（`T.Parallel(block_m, D_padded)`） | dtype 副本 | `x_f32` | fp32 opmath（一次 cast、多遍复用） |
| S1 | 行和 `T.reduce_sum(x_f32, acc, dim=1)`；均值 `mean_val[i] = acc[i] / float(D)`（`T.Parallel(block_m)`） | `x_f32` | `acc`、`mean_val` | mean |
| S2 | 中心化平方（in-place 覆写）`x_f32[i,j] = (x_f32[i,j] − mean_val[i])²`；行和 `T.reduce_sum(x_f32, acc, dim=1)`；`rstd[i] = T.rsqrt((acc[i] − pad_count·mean_val[i]²)/float(D) + eps)` | `x_f32`、`mean_val` | `acc`、`rstd` | var（含填充校正）与 rstd |
| E1 | affine epilogue：`x_local[i,j] = (cast(x_local[i,j]) − mean_val[i]) · rstd[i] · cast(weight[j]) + cast(bias[j])`（`T.Parallel(block_m, D_padded)`，从**原始副本**重推 `(x − mean)`） | `x_local`、`mean_val`、`rstd`、`weight`、`bias` | `x_local`（dtype，隐式截断） | y 公式（kernel 层行广播 affine） |
| W1 | 写出：`T.copy(x_local, shared_buf)` → `T.copy(shared_buf, y[pid_m*block_m, 0])` | `x_local` | GM y | 输出搬运 |
| H1 | host 侧：`D_padded` 计算、`F.pad` 输入与 affine 至 `D_padded`、输出截断 `y[:, :D]`；op 层另有 reshape → kernel → reshape → **per-channel affine** | — | — | 分组重排与 affine 分层 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[x] --T.copy--> SMEM[shared_buf (bm,D_padded)] --T.copy--> FRAG[x_local (bm,D_padded) dtype]
  --T.Parallel cast--> FRAG[x_f32 fp32]
  --T.reduce_sum(dim=1)--> FRAG[acc (bm,)] --/D--> FRAG[mean_val]
  --(in-place 中心化平方)--> FRAG[x_f32] --T.reduce_sum(dim=1)--> FRAG[acc]
  --(acc - pad·mean²)/D + eps, T.rsqrt--> FRAG[rstd]
GM[weight]/GM[bias] --(epilogue 内逐元素 j 索引标量读)--> 复合表达式
FRAG[x_local] --epilogue（隐式截断）--> SMEM[shared_buf] --T.copy--> GM[y]
（SMEM 单缓冲在装载与写出间复用；affine 在 kernel 内、per-channel affine 在 op 层）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)` 一维 grid，每个 block 处理 `block_m` 行；block 内 `T.Parallel(block_m, D_padded)` 元素级并行（线程映射到行内元素，行内归约经线程级归约塌缩）；默认配置 `block_m=1`（`select_row_config` 固定值，理由见 §0.4 第 6 项）+ `threads=128`。

**host 侧逻辑**：`GroupNormKernel.__init__` 计算 `D_padded`、构建 `lru_cache` 工厂、缓存设备侧单位 affine；`forward` 做 `F.pad` / 单位 affine 兜底 / `_group_norm_wrapped` 调用 / `y[:, :D]` 截断；op 层 `GroupNormFwdOp.forward` 做 device/dtype/shape 校验、`contiguous()`、两次 `reshape`（均为视图）、kernel 调用、reshape 回原形、**kernel 外 per-channel affine**、`eval_roofline` 记账（`flops = 5*N*C*spatial`、`bytes = (2*N*C*spatial + 2*C)*elem_bytes`）。host 侧无输入张量内容改写（`F.pad` 只在 `D` 非 256 对齐时触发）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM 行块驻留（`alloc_shared` + 分块 `T.copy`） | 消 GM 重读：行内元素被 S1/S2/E1 三遍复用 | `(bm, D_padded)` 块经 shared memory 中转 | CUDA shared memory | 硬件强相关 |
| 2 | 256 元素对齐 padding（`D_padded`）+ 填充方差校正 | 满足 `T.copy` 的 SMEM 整块搬运对齐 | `align_up(D, 256)` 零填充；`(S2 − pad_count·mean²)/D` 恒等式 | CUDA SMEM 拷贝对齐约束（文件头注释明示） | 硬件强相关 |
| 3 | 中心化两遍方差（而非 `E[x²] − E[x]²`） | **数值稳定**：避免大均值小方差下的灾难性抵消 | 先 mean 再 `(x−mean)²` 求和 | 无（纯算法层） | 可移植 |
| 4 | fp32 全程计算 + 单次舍回 | 精度：fp32 opmath 对齐 torch golden | `x_f32` fragment 上算完再一次截断 | fragment 寄存器堆 | 硬件强相关（NPU 有 `vcast` 等价） |
| 5 | 一次 cast、三遍复用（`x_f32` 常驻） | 消重复 cast 与重复访存 | L2 结果跨 S1/S2/E1 复用 | 无（公共子表达式消除） | 可移植 |
| 6 | `block_m=1` 结构性默认 | 单行/CTA → 行内归约的跨线程 AllReduce 塌缩结构上不可能；同时天然规避行尾越界（整块拷贝无行谓词） | `select_row_config` 固定 `block_m=1` | CTA/warp 组织 | 硬件强相关 |
| 7 | `threads=128` | 实测 CUDA 最优线程数 | block 内线程级并行 | CUDA 线程模型 | 硬件强相关 |
| 8 | 单位 affine 兜底（`_identity_affine_for`） | 让「无 affine」调用仍走同一签名 | 缓存 `(D_padded,)` 的 ones/zeros | 无 | 可移植（接口契约） |
| 9 | SMEM 单缓冲在装载/写出间复用 | 省片上容量 | 同一 `shared_buf` 两次角色 | shared memory 容量 | 硬件强相关 |
| 10 | kernel 外 per-channel affine（op 层） | per-channel `(C,)` affine 与行广播布局不兼容时的正确施加方式 | op 层 reshape 后广播乘加 | 无 | 可移植（接口契约） |
| 11 | `lru_cache` 工厂 + per-shape JIT 特化 | 摊编译开销 | `functools.lru_cache(maxsize=32)` | 无 | 可移植（host 层） |
| 12 | `T.copy` 整块搬运（coalesced） | 访存带宽 | 块状 GM↔SMEM | memory coalescing | 硬件强相关 |

> 识别不出的优化会被静默丢弃。上表 12 项由源码逐语句反查得到（含文件头注释显式声明的 padding 机制与 affine 分层），§0.5 逐项给出四态处置。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据：`migration-analysis.md` §5 三层模型 + §5.3 能力映射表；`examples/` 与 `docs/` 佐证；pattern-library 实测条目。**判定输入含 §1.6.0 调研结论**（源算法与调研候选同台比较）。

| 条目（对应 §0.4 编号） | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义（分组重排、规约轴、ddof=0、eps 位置、输出同形） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| affine 分层（kernel 层单位元 + op 层 per-channel） | 语义 | 无 | — | **保留** | op 层 `GroupNormFwdOp.forward` 原样保留（`examples/TileOPs/tileops/ops/norm/group_norm.py`）；`(C,)` 权重绝不下传 kernel |
| 中心化两遍方差（#3） | 算法 | 无 | — | **保留** | 纯算法层；§1.6.0 调研用 D-1 机器证据否决单遍矩式（§1.6.1 采纳 O 项与否决项） |
| 行块驻留 SMEM → UB（#1、#9） | 优化 | shared memory | 有（UB 192KB/AIV） | **等价替换** | `T.alloc_shared`（Developer 映射 UB，`docs/Tilelang.language/内存操作/T.alloc_shared.md`）；预算按实测多缓冲平台值 20 B/元素重算（§4.5） |
| fp32 全程计算 + 单次舍回（#4、#5） | 优化 | fragment 寄存器 | 有（fp32 UB + `vcast`） | **等价替换** | `T.vcast(rint)` 升 fp32 → fp32 v-prefix 链 → `T.vcast(rint)` 单次舍回（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；bf16 无 Vector 算术故 fp32 中转是**必需**，见该文档 dtype 表） |
| `T.copy` 整块分级搬运（#12） | 优化 | coalescing | 有（GM↔UB） | **等价替换** | 双显式 slice 形态（`docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 1）；行块在无 padding 布局下是**整段连续 span**，单次 DMA 粒度 = `block_m × D × elem_bytes` |
| 256 元素对齐 padding + 填充方差校正（#2） | 优化 | CUDA SMEM 拷贝对齐 | NPU 无此约束 | **舍弃** | ① NPU `T.copy`/向量 op 接受任意行 extent（`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` 的 L0 用例 N = 13 / 514 / 1152 / 3000 / 4096 全部通过）；② padding 的成本在 NPU 侧更重：wrapper 的 `F.pad` 是一次**全量张量拷贝**（读 M·D + 写 M·D），对 `D` 非 256 对齐的 workload 相当于把输入流量翻倍；③ 同族前作的 durable 结论即「256-pad 与 pad 校正整体舍弃」（pattern-library `cases.md` 的 CASE-norm-adalayern-migration 条目）。优化意图（整块向量化搬运）由 UB 自然对齐 + 连续 span slice copy 承接 |
| `block_m` 结构性默认 1（#6 前半） | 优化 | CTA 行均匀性 + 整块拷贝无行谓词 | 担忧不存在（且 NPU 有更好形态） | **重新设计** | NPU 归约按行硬件降低（`T.reduce_sum dim=1`），无跨线程 AllReduce 塌缩问题；`block_m` 改由 UB 预算表驱动（§0.6 R2），行尾由 `T.min` + 双显式 slice + 垃圾行丢弃处理（§0.6 R3） |
| `threads=128` / block×thread 两级并行（#7） | 优化 | CUDA 线程模型 | 无 | **舍弃 + 重新设计** | `T.Kernel(threads=)` 在 npuir 无效果（pattern-library `traps-compiler.md` 的 TRAP-threads-kwarg-noop）；并行结构改为 1D persistent 分核（§0.6 R1），元素级并行交给 Vector 向量化（§1.6.2/§1.6.3） |
| SMEM 单缓冲复用（#9） | 优化 | SMEM 容量 | 有 | **等价替换** | UB 单 `stage` 缓冲双角色（装载中转 + 舍回中转），容量收益保留 |
| kernel 层行广播 affine（`weight[j]`/`bias[j]` 逐元素乘加，§0.1 表第 1 行） | 优化 | 无（纯 elementwise） | 有，但**在本迁移路径恒为单位元** | **等价替换** | kernel 层 affine 的**计算被保留在 op 层**（与 GPU 完全一致：op 层补 per-channel affine）；kernel 侧保留 `(D_padded,)` 形参但**不读**——本路径 wrapper 恒传单位权重/零偏置，去 `×1 + 0` 在 IEEE 下逐位恒等（§1.6.1 采纳项 O5 机器验证 max ulp = 0）。代价对照：若强行在 kernel 内实现行广播 affine，只有两条路——(a) 逐元素 `T.Parallel` 内 `weight[j]` 标量索引，正是 `examples/norm/group_norm.py` 第 48 至第 53 行的遗留形态，也是 §1.6.2 要求消除的标量循环；(b) `T.vbrc` 2-D 中转路径（src 需 rank 一致的 `(1, D)` 缓冲，见 `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 与示例 2），需额外 dtype 广播缓冲 + 每元素 +2 个向量 op，且该分支在本任务无任何用例覆盖（不可测路径） |
| `lru_cache` 工厂 + per-shape JIT 特化（#11） | host | 无 | — | **保留** | host 层硬件无关；Stage 0 提取件已保留工厂签名，Kernel 类实例缓存承接特化 |
| `out_idx=[3]` + 4 参签名 | 接口 | 无 | — | **保留** | 接口契约（§0.2）；NPU wrapper `_group_norm_wrapped` 已按 4 参调用移植（`examples/TileOPs/tileops/kernels/norm/group_norm/group_norm.py`） |
| `ALIGNMENT = 256` / `_align_up` 常量（后端常量） | host | CUDA SMEM 对齐 | 无意义 | **等价替换（后端常量适配）** | 迁移规则允许「与后端实现相关的参数」适配：NPU 无需行对齐 padding，故重导出 `ALIGNMENT = 1`（使 wrapper 的 `pad = D_padded − D` 恒为 0，`F.pad` 分支与 `y[:, :D]` 截断分支均不触发），保留 `_align_up` 通用实现。**该常量属后端实现细节，不改变 kernel 的对外参数列表**（`_group_norm_kernel(M, D, eps, dtype)`、`main(x, weight, bias, y)`、`out_idx=[3]` 全部不变） |

### 0.6 NPU 算法重设计

调研（Phase R，§1.6.0）选定算法族 = **「中心化两遍方差 + 整行（全 D）UB 驻留 + 精确 D + fp32 中转链」**；下列重设计项是它的 NPU 落地，源算法已识别优化手段的意图承接见本节末尾汇总。

**重设计项 R1：并行结构（grid 超发 → 1D persistent 分核）**

- **源方案**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)`——逻辑 block 数随 M 超发（image-g32 形态 `M=256`、`block_m=2` 即 128 个 block），依赖 CUDA 运行时调度；`threads` 折叠 CUDA 线程并行。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(num_logical, vector_cores)`，`vector_cores = NPUUtils.get().get_aicore_num() * 2`（纯 Vector 算子核数翻倍；本机实查 20 → 40，见 §5.5）；核内 `for i in T.serial(num_local_tasks)` grid-stride 映射 `block_id = i * num_kernels + cid`，`if block_id < num_logical` 守卫越界任务；所有量是 host 层 Python 编译期常量（`T.ceildiv`/整数运算在 trace 期折叠，pattern-library `layout.md` 的 PL-1.5 先例）。
- **语义保持论证**：行间完全独立（规约轴 = 行内 D，无跨行依赖），任务映射是行块集合上的双射；每个行块内的 mean/var/归一化公式与累加分组（按行一维归约）与源一致，累加顺序变化仅限同一行内的树形归约（fp32，容差内，§8.2 判定）。意图承接：消除超发 block 的串行调度开销（`core-split-strategy.md` §1 极大规模分支的 persistent 方案），并取得 grid-stride 聚集窗口的 MTE2 访问效益（pattern-library `constants.md` 的 CONST-mte2-degradation 条目记载 grid-stride 实测优于每核连续分块 +3.2%）。

**重设计项 R2：block_m 语义（结构性 1 → UB 预算表驱动）**

- **源方案**：`block_m = 1`（每 CTA 单行；既是行内归约均匀性的手段，也是规避整块拷贝行尾越界的唯一屏障）+ `threads = 128`。
- **NPU 新算法**：`_func(block_m=block_m_default)`，`block_m_default = min(8, max(1, _UB_MAX_ELEMS // D))`，`_UB_MAX_ELEMS = 9216`（= 192KB ÷ 实测多缓冲平台值 20 B/元素 = 9830，取 8 × 1152 留 ≥12KB 余量；同族前作 Stage 3 实测值，见 §4.5）；工厂层加 UB 预算 guard（超预算抛 `ValueError`）。
- **语义保持论证**：`block_m` 只改变每任务的行数，不改变行内计算与累加分组；`M % block_m ≠ 0` 的尾块由 R3 处理。行为等价性由 §8.3 的 L0 用例（含 `M % block_m = 1` 的尾块组）守护。

**重设计项 R3：边界处理（无行谓词的整块拷贝 + 256 零填充 → `T.min` + 双显式 slice + 垃圾行丢弃）**

- **源方案**：整块 `T.copy(x[pid_m*block_m, 0], shared_buf)`（区域由 dst 缓冲形状推定，见 pattern-library `traps-compiler.md` 的 TRAP-T-copy-region-semantics），行尾正确性完全依赖 `block_m = 1`；列方向靠 256 零填充 + 填充方差校正。
- **NPU 新算法**：`real_m = T.min(block_m, M - off_m)`；装载 `T.copy(x[off_m : off_m + real_m, 0:D], stage[0:real_m, 0:D])`（src/dst 双显式 slice，dst 零起点——规避同条目 TRAP-UB-dst-align 的非零起点问题）；所有 v-op/reduce 作用于完整 `(block_m, D)` UB 缓冲（尾行是未初始化垃圾，但**按行隔离**：`reduce dim=1` 逐行、v-op 逐元素或行广播，无跨行混合）；写出 `T.copy(stage[0:real_m, 0:D], y[off_m : off_m + real_m, 0:D])` 只写有效行；列方向**不做任何 padding**（精确 D）。
- **语义保持论证**：垃圾行的统计量无意义但不会流入有效行（行隔离 + 输出切片），与 `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` 的尾块模式一致（该实现 L0 含 M 尾块用例全部通过）；列方向精确 D 与源的 256 填充在实数域严格等价（填充列贡献 `mean²`，源显式减回），D-1 机器验证见 §1.6.1 采纳项 O4。

**重设计项 R4：dtype 路径（fragment 隐式 cast 链 → 显式 `vcast(rint)` fp32 中转链）**

- **源方案**：`T.Parallel` 内 `T.cast(..., "float32")` 上cast + epilogue 赋值隐式截断回 dtype。
- **NPU 新算法**：fp16/bf16 输入走 `T.vcast(stage, x_f32, round_mode="rint")`（f16→f32 / bf16→f32，无损）→ fp32 v-prefix 链 → `T.vcast(x_f32, stage, round_mode="rint")` **单次舍回**；fp32 输入无 cast（同 dtype 直拷直算）。跨 dtype 的 `T.copy` **不**用作 cast 融合（TRAP-C12-copy-dtype-cast）。
- **语义保持论证**：fp16/bf16 是 fp32 的子集，上cast 精确无损；fp32→目标 dtype 单次 RNE 舍回与 golden 的 `.to(x.dtype)` 舍入路径一致；bf16 无 Vector 算术（`docs/Tilelang.language/数学操作/T.vmul.md`、`T.vadd.md`、`T.vsub.md`、`docs/Tilelang.language/规约操作/T.reduce_sum.md` 的 dtype 表 bf16 列均为 ×），fp32 中转是**必需**而非可选。pattern-library `traps-compiler.md` 的 TRAP-fp16-opmath-golden 给出该通解与 ≤1 ulp fp16 的实证。

**重设计项 R5：rstd 指令路线（源 `T.rsqrt` → NPU `T.vsqrt` + `T.vdiv` 组合）**

- **源方案**：`rstd[i] = T.rsqrt(var + eps)`（CUDA 侧 `T.rsqrt`）。
- **NPU 新算法**：`T.vadd(var, eps, var)` → `T.vsqrt(var, s2)` → `T.vdiv(s2, var, rstd)`，即 `rstd = sqrt(v)/v`（实数域 ≡ `1/sqrt(v)`）。
- **语义保持论证**：实数恒等；fp32 实现在 var 扫描域上相对 fp64 的实测最大相对误差 6.1e-8（§1.6.1 采纳项 O2 的 D-1 证据），远小于 fp32 门禁 rtol 1e-5。**不采用** `T.vrsqrt`：该指令在 npuir 上是近似指令（实测最大相对误差 2.87e-3，pattern-library `traps-runtime.md` 的 TRAP-vrsqrt-plain-precision），在 fp32 1e-5 门禁下必然大面积违反——这是同族前作 Stage 3 实际踩过的坑，本设计在设计期前置规避。

**重设计项 R6：kernel 层 affine 的施加位置（重推 `(x−mean)` + `×weight[j] + bias[j]` → 单位仿射消除 + op 层 per-channel affine）**

- **源方案**：epilogue 从**原始 dtype 副本**重推 `(cast(x_local) − mean)`，再乘 `weight[j]`、加 `bias[j]`。
- **NPU 新算法**：统计相中心化后**保留** `d = x − mean`（fp32 驻留）供 epilogue 复用（CSE，采纳项 O3）；kernel 层 affine 恒定为单位元故整体省略（采纳项 O5，IEEE 逐位恒等）；per-channel affine 由 op 层在 kernel 外完成（与 GPU 完全一致）。
- **语义保持论证**：① `d` 是同一子表达式的值复用（上cast 无损、mean 相同 → `d` 与重推结果**逐位相同**，D-1 max ulp = 0）；② 单位仿射消除：`y·1.0 + 0.0` 在 IEEE 下逐位等于 `y`（乘 1 与加 +0.0 均不改变结果；`-0.0` 边界只影响符号位且与 golden 的 `F.group_norm` + `.to(dtype)` 舍回路径一致，D-1 max ulp = 0）；③ per-channel affine 的计算**没有丢失**，只是在 op 层施加（源 op 层同样如此）。

**重设计项 R7：后端对齐常量（`ALIGNMENT = 256` + host `F.pad` → 精确 D，无 padding）**

- **源方案**：`D_padded = align_up(D, 256)`，wrapper 对 x/weight/bias 做 `F.pad`，输出再 `y[:, :D]` 截断。
- **NPU 新算法**：重导出 `ALIGNMENT = 1`，kernel 工厂内仍按源结构算 `D_padded = _align_up(D, ALIGNMENT)`（等价于 `D`），prim_func 声明的行宽即精确 `D`；wrapper 的 `pad` 恒为 0，`F.pad` 与输出截断分支自然不触发。
- **语义保持论证**：行宽改变不改变行内语义（规约轴长度由 `D` 变为 `D_padded = D`，二者数值相同）；wrapper 契约（`D_padded` 属性、`register_fake` 返回 `(M, D_padded)`、输出截断）保持自洽；收益 = 去掉 host 侧全量 `F.pad` 拷贝（对非 256 对齐 workload 相当于输入流量翻倍）与填充列上的无效计算。

**舍弃项的意图承接汇总**（不得静默丢弃，migration-analysis §6.4）：

| 被替换/舍弃项 | 原意图 | 由谁承接 |
|--------------|--------|---------|
| 256 对齐 padding + 填充方差校正（§0.4 #2） | 向量化整块搬运 | 精确 D 下每行块是**连续 span**，单次 DMA 粒度 = `block_m·D·elem_bytes`（`constants.md` 的 CONST-copy-floor-dma-granularity：粒度越大拷贝地板越低）；UB 32B 对齐由编译器管理 |
| `block_m = 1`（§0.4 #6 前半） | 行内归约均匀 + 规避行尾越界 | `T.reduce_sum dim=1` 的硬件按行归约（无 AllReduce 塌缩）+ R3 的 `T.min` 尾块（UB 预算表改驱动 block_m） |
| `threads = 128` / 两级并行（§0.4 #7） | 块内并行度 | Vector lane 向量化（§1.6.2 全表）+ persistent 分核（R1） |
| kernel 内 `weight[j]`/`bias[j]` 标量索引乘加（§0.4 表中第 10 项对应的 kernel 侧实现） | 行广播 affine | op 层 per-channel affine（保留）+ 单位仿射消除（O5，逐位恒等） |
| SMEM 中转本身 | coalesced 带宽 | 保留为 UB 中转（`stage`）；无 padding 时行块为连续 span，coalescing 天然满足 |

### 0.7 标杆实现

- **源算子（迁移精确源）**：`/workspace/TileOPs-gpu/tileops/kernels/norm/group_norm.py` 第 45 至第 121 行；Stage 0 提取件 `examples/TileOPs/tileops/kernels/norm/group_norm/_group_norm_fwd_kernels.py`（含 `ALIGNMENT` / `_align_up` / `_group_norm_kernel`）。
- **NPU 侧待适配 wrapper（导入契约内不含修改）**：`examples/TileOPs/tileops/kernels/norm/group_norm/group_norm.py`（K5 至 K9 适配：`supported_archs=None`、`npub::group_norm_fwd`、无 autotune、移除 `threads`）；op 层 `examples/TileOPs/tileops/ops/norm/group_norm.py`。
- **参考实现 / 测试基准**：`examples/TileOPs/tests/ops/test_group_norm.py` 的 `GroupNormTest.ref_program`（`F.group_norm(x.float(), g, weight=weight.float(), bias=bias.float(), eps).to(x.dtype)`）——golden 以 §0.1 语义移植该实现，不复刻 NPU 算法。
- **manifest 规格**：`examples/TileOPs/tileops/manifest/norm.yaml` 的 `GroupNormFwdOp` 条目（workloads + shape_rules + roofline）；workload 生成器 `examples/TileOPs/tileops/workloads/norm.py`；性能基准 `examples/TileOPs/benchmarks/ops/bench_group_norm.py`。
- **同族前作（形态参照）**：`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` 与其设计文档 `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md`（row-reduction 迁移基准形态：persistent 分核 + 尾块三件套 + fp32 中转链 + UB 预算 guard；Stage 4 调优版在 `perf_opt` 子目录）。**注**：该前作的 Stage 1 任务工作区（含其 `RETROSPECTIVE.md`）未在本仓留存，本设计改以 pattern-library 的 durable 条目（`cases.md` 的 CASE-norm-adalayern-migration / CASE-norm-adalayern-stage4、`traps-runtime.md` 的 TRAP-vrsqrt-plain-precision、`traps-compiler.md` 的 TRAP-UB-multibuffer-inflation 第三证）作为同族教训来源。
- **同类 NPU 原型（形态对照，非迁移源）**：`examples/norm/group_norm.py`——同签名工厂与 `(M, N)` 语义的早期 NPU 原型，其 `T.Parallel` 标量循环与 `T.vrsqrt`、索引式 `T.copy` 形态正是本设计要替换的对象（§1.6.2/§1.6.3）。

---

## 1. 概述

### 1.1 算子名称

`_group_norm_kernel`（承载 `GroupNormFwdOp`，affine 主路径）

### 1.2 功能描述

对 `(M, D)` 二维视图的每一行做归一化（分组由 op 层的 reshape 完成：`(N, C, *spatial) → (N*num_groups, D)`，`D = (C/num_groups)·spatial_size`），得到 `y = (x − mean)·rstd`；per-channel affine 由 op 层在 kernel 外施加。NPU 侧实现与源算法的差异（见 §0.6/§1.6.0）：保留中心化两遍方差与 fp32 中转链，舍弃 256 对齐 padding 与 kernel 层单位仿射，并行组织改为 1D persistent 分核，rstd 改 `vsqrt + vdiv` 组合。

### 1.3 数学公式

$$
y_{rj} = \left(x_{rj}-\mu_{r}\right)\cdot\frac{1}{\sqrt{\frac{1}{D}\sum_{j'<D}\left(x_{rj'}-\mu_{r}\right)^2+\varepsilon}},
\qquad \mu_{r} = \frac{1}{D}\sum_{j<D} x_{rj}
$$

op 层随后施加 per-channel affine：$y_{nc\cdots} \leftarrow y_{nc\cdots}\cdot w_c + b_c$。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

与 §1.6.0 调研选定算法一致（中心化两遍 + 全 D 驻留 + 精确 D + fp32 中转），每任务处理 `block_m` 行、整行驻留 UB：

1. **装载**：`x[off_m : off_m+real_m, 0:D]` → UB `stage`（输入 dtype，双显式 slice）；
2. **上cast**（fp16/bf16）：`T.vcast(stage, x_f32, round_mode="rint")`；fp32 输入直拷入 `x_f32`；
3. **均值**：`T.reduce_sum(x_f32, s1, dim=1)` → `mean = s1 · fl(1/D)`（采纳项 O1）；
4. **中心化（保留 d）**：`T.vsub(x_f32, mean, x_f32)` → `d`（epilogue 复用，采纳项 O3）；
5. **方差**：`sq = d·d` → `T.reduce_sum(sq, s2, dim=1)` → `var = s2·fl(1/D)`（精确 D，无填充校正，采纳项 O4）→ `var += eps`；
6. **rstd**：`T.vsqrt(var, tmp)` + `T.vdiv(tmp, var, rstd)`（采纳项 O2）；
7. **归一化**：`T.vmul(x_f32, rstd, x_f32)`（行广播 `[block_m,D] × [block_m,1]`）；kernel 层单位仿射省略（采纳项 O5）；
8. **舍回 + 写出**：`T.vcast(x_f32, stage, rint)` → `T.copy(stage[0:real_m, 0:D], y[...])`（只写有效行）。

**与源算法的差异及来源**：无 256 填充与填充方差校正（§0.6 R7 / 采纳项 O4）；persistent 分核（§0.6 R1）；`block_m` 由 UB 预算驱动（§0.6 R2）；尾块 `T.min` + 双显式 slice（§0.6 R3）；显式 `vcast` fp32 中转（§0.6 R4）；rstd 组合式（§0.6 R5）；kernel 层单位仿射消除、per-channel affine 留在 op 层（§0.6 R6）。

### 1.5 数据流图

```
GM[x] --T.copy(slice)--> UB[stage(dtype)] --vcast(rint)--> UB[x_f32]
  --reduce_sum(dim=1)--> UB[s1 (bm,1)] --vmul(D_inv)--> UB[mean]
  --vsub(行广播)--> UB[x_f32 = d] --vmul--> UB[sq_f32 = d²]
  --reduce_sum(dim=1)--> UB[s2] --vmul(D_inv)--> UB[var] --vadd(eps)--> UB[var]
  --vsqrt--> UB[s2 = sqrt(var)] --vdiv(s2, var)--> UB[rstd]
  --vmul(行广播)--> UB[x_f32 = d·rstd]
  --vcast(rint)--> UB[stage(dtype)] --T.copy(slice)--> GM[y]
（fp32 dtype：全部 vcast 消失，GM↔UB 同 dtype 直拷；buffer 复用：stage 双角色（装载/舍回）、sq_f32 双角色（平方/开方暂存））
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**同一数学语义（per-group 行归一化：行均值 + 有偏方差 + 归一化）的算法族**。源算法（中心化两遍 + SMEM 驻留 + 256 padding + 行广播 affine）只是**基线候选之一**。
> 调研深度：**完整调研**（规约 / 统计类，按 `algorithm-research.md` §2 分级）。
> 信息源：`algorithm-candidates.md` 的 ALG-layernorm / ALG-rmsnorm 命中行（必查清单，逐行评估）；本仓 `examples/` 同类实现（`examples/norm/group_norm.py`、`examples/norm/layer_norm.py`、`examples/norm/instance_norm.py`、`examples/norm/ada_layer_norm_and_zero.py`）；pattern-library（`layout.md` PL-1.5、`traps-runtime.md` TRAP-vrsqrt-plain-precision、`traps-compiler.md` TRAP-UB-multibuffer-inflation、`constants.md` CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-copy-floor-dma-granularity / CONST-aicore-910B2C）；evolution 队列 VP-2026-0043/0044（row-reduction 迁移基准形态与矩式定量边界）；源码 §0.3/§0.4；结构判据（R2）。**未使用互联网检索**：本地源已覆盖四问所需的全部候选（含矩式、Welford、N-tile 流式与迁移基准形态），无未命中缺口。

> **证据路径（本文档引用的 pattern-library 条目与标准的完整路径，设计期已逐条打开核对；负向断言与弃选论证均指向这些条目）**：
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（CONST-mte2-degradation、CONST-vector-launch-overhead、CONST-copy-floor-dma-granularity、CONST-capacity-910B2C、CONST-aicore-910B2C、CONST-npuir-f32-division-not-ieee-rounded）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md`（TRAP-vrsqrt-plain-precision）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`（TRAP-UB-multibuffer-inflation、TRAP-T-copy-region-semantics、TRAP-UB-dst-align、TRAP-threads-kwarg-noop、TRAP-fp16-opmath-golden、TRAP-C10-strided-gather、TRAP-C12-copy-dtype-cast）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md`（PL-1.5-quickref、PL-1.1-transpose-chain、PL-1.3-host-permute）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/cases.md`（CASE-norm-adalayern-migration、CASE-norm-adalayern-stage4）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/repro/TRAP-vrsqrt-plain-precision.py`（上述陷阱的机器复现脚本）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/repro/CONST-npuir-f32-division-not-ieee-rounded.py`
> - `.agents/skills/_shared/standards/core-split-strategy.md`、`.agents/skills/_shared/standards/hardware-cost-model.md`、`.agents/skills/_shared/standards/negative-claim-evidence.md`
> - `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（ALG-layernorm / ALG-rmsnorm 条目）
> - `.agents/evolution/queue.md`（VP-2026-0043 矩式定量边界、VP-2026-0044 row-reduction 基准形态）

**R1 等价化简公式候选**：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | **基线**（GPU 源算法） | 中心化两遍 + `D_padded = align_up(D,256)` 零填充 + 填充方差校正 `(S2 − pad·m²)/D` + SMEM 驻留 + kernel 层行广播 affine（本路径恒为单位元） | — | — | ✅（基线） |
| 1 | 精确 D 两遍（去 padding） | 同基线的数学内核，行宽 = 精确 `D`，无填充校正 | **数学恒等**（填充列贡献 `m²`，源显式减回；实数域严格相等） | 去 host `F.pad` 全量拷贝（非 256 对齐 workload 相当于输入流量翻倍）+ 去填充列无效计算；小 D 大均值位形数值更稳（D-1 证据） | ✅（并入主选） |
| 2 | 单遍矩式 | `var = E[x²] − (E[x])²`（`examples/norm/group_norm.py` 的高性能原型即此法） | **否**——大均值 / 小方差行在 fp32 下灾难性抵消（D-1 机器证据：最大相对误差 6.0e3，容差违反率 100%） | 驻留域内仅省一次 `(bm,1)` 级 v-op 与一次中心化相的大 op，**GM 流量不变** | ❌（数值稳定性否决） |
| 3 | Welford 在线增量 | 逐 tile 增量 mean/M2 合并 | 数学等价（容差内） | 驻留域内无 GM 收益；合并链是逐 tile 标量序列依赖 | ❌（R2 详述） |
| 4 | N-tile（D 维分块）流式两遍 | `block_d < D` 分块 + `clear=False` 累加，输出遍重读 x | 等价 | 任意 `D` 可扩展（UB 无关） | ❌（域内 x 重读 1 遍 → 访存 +25%） |
| 5 | 除法转乘编译期倒数 | `S/D → S·fl(1/D)`（`D` 为编译期常量） | 容差内等价（fp32 ≤1 ulp；D-1 机器验证） | `(bm,1)` 上 vdiv → vmul（PL-1.5「乘编译期常数倒数」先例） | ✅（§1.6.1 采纳 O1） |
| 6 | rstd 组合式 | `1/sqrt(v)` → `sqrt(v)/v`（`T.vsqrt` + `T.vdiv`） | 恒等（实数域）；fp32 实测相对误差 6.1e-8 | 精度路线（替代 npuir 近似 `T.vrsqrt`，2.87e-3 → 6.1e-8） | ✅（§1.6.1 采纳 O2） |
| 7 | 中心化值复用（CSE） | 统计相保留 `d = x − mean`，epilogue 免重推 | 恒等（同值复用，逐位相同） | 每任务省一次 epilogue 重推（源为从 dtype 副本重 cast + 重减） | ✅（§1.6.1 采纳 O3） |
| 8 | 单位仿射消除 | kernel 层 `·weight[j] + bias[j]`（本路径 weight≡1、bias≡0）整体省略 | IEEE 逐位恒等（`×1.0`、`+0.0`） | 每元素 −1 vmul −1 vadd，并避免标量索引 / `vbrc` 中转链 | ✅（§1.6.1 采纳 O5） |

**R2 在线算法**：**有**在线（流式）变体——Welford 增量 mean/M2 与分块 running 统计（`algorithm-candidates.md` 的 ALG-layernorm 行明示「Welford 增量 mean/var；分块 running 统计」；本仓 `examples/norm/ada_layer_norm_and_zero.py` 的 per-tile `clear=False` 累加即是分块 running 统计的现存形态）。**但本 workload 域不采纳**，依据是**结构判据 + 驻留判据**（非"无依据的否定"）：

- 在线变体的收益口径是「扫描遍数（两遍 → 单遍）/ 中间缓冲 O(行) → O(tile)」。而本算子在域内的整行（`D` 最大 7200，manifest 四档 D ∈ {256, 4096, 6272, 7200}）**本就全量驻留 UB**：x 的 GM 扫描已经是 **1 遍**（读一次、写一次），中间缓冲已经是 O(行)——在线化**没有任何 GM 收益**；
- Welford 的合并链引入逐 tile 的**标量序列依赖**（每次合并是 `(count, mean, M2)` 三元组的串行更新），在 Vector 单元上是反亲和形态（同族前作的 durable 结论：驻留域内在线化无收益，见 `algorithm-candidates.md` 的 ALG-layernorm R2 与 evolution 队列 VP-2026-0044 的基准形态条目）；
- **驻留上限**（在线变体真正有意义的域边界）：多缓冲平台值 20 B/元素（§4.5）下整行驻留要求 `D ≤ 9216`（`block_m ≥ 1`）；`D > 9216` 时需 R3 候选 4 / 在线变体，属本任务域外（§9.3 备注，工厂 guard 拦截）。

**R3 复杂度对比**（口径：每行 `D` 元素、`eb` = 元素字节；「元素级 op」= 一次 `(block_m, D)` 规模的向量操作；FLOPs 口径对齐 manifest 的向量操作量纲（无超越函数的乘加折算）；**访存 Bytes = GM 流量**，含输入读 + 输出写 + 中间 GM 往返；「扫描遍数」= 同一张量的 GM 重复读取次数；**基线候选在表内**）：

| 算法候选 | 每元素级 op 数（统计相 + epilogue） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------------------------|---------------|---------|-------------|---------------------|
| 基线（GPU 源：256-pad + 填充校正 + kernel 层单位仿射 + host `F.pad`） | ≈3（cast/sub/sq）+1（mul ×2 见注）+2（pad 列多算）+2（单位仿射）≈ 8 | `2·M·D·eb` + **host `F.pad` 的 `2·M·D_pad·eb`（仅 pad≠0 时，等于输入流量翻倍）** | 1（kernel 内）＋1（host pad 遍） | `block_m × D_padded × (2+4+4) B`（SMEM/fragment，CUDA 侧） | 行独立；无跨核同步；但 grid 超发 → 串行调度开销 |
| **主选：精确 D 两遍驻留 + fp32 中转 + 单位仿射消除** | ≈1（cast）+1（sub）+1（sq）+1（mul·rstd）+1（舍回 cast）≈ 5（另加 2 次行归约与 4 个 `(bm,1)` 级小 op） | **`2·M·D·eb`（全候选最小）** | **1** | `block_m × D × 20 B`（多缓冲实测平台值，UB） | 行独立；persistent 分核无跨核归并；`num_local_tasks` 轮串行 |
| 单遍矩式驻留 | ≈5（省一次中心化相的大 op） | `2·M·D·eb` | 1 | 同上 | 同上（但数值否决） |
| Welford 在线驻留 | ≈5 + 逐 tile 合并链（每 tile 一次 reduce + 标量三元组更新） | `2·M·D·eb` | 1 | 同上 + running 统计量（`count/mean/M2`） | 同上；合并链串行 |
| D-tile 流式两遍 | ≈5 + 每 tile 归约开销 | `3·M·D·eb`（x 读 2 遍：统计遍 + 输出遍） | **2** | `block_m × block_d × 20 B`（更小，D 无上限） | 同上；仅在 `D > 9216` 时必要 |

> 注：主选与基线的差异不在 FLOPs 量级，而在 **host `F.pad` 的整遍流量（仅非 256 对齐时触发）+ 填充列的无效计算 + kernel 层单位仿射的每元素 2 个 op**；带宽受限部分（`2·M·D·eb`）二者相同。矩式/Welford/流式候选在驻留域内均**不能**降低 `2·M·D·eb` 这一主导项（流式反而 +50%），故主选由"流量 + 数值 + 亲和"三者共同决定，不存在纸面并列。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰均附证据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| 基线（256-pad + SMEM） | Vector ✓（但含 host `F.pad` 额外 kernel launch） | CUDA SMEM 口径不适用；NPU 需按 `(bm, D_padded)` 重算 | 256 对齐是 CUDA 约束，NPU 无对应要求 | 依赖 `block_m=1` 保行尾正确 | — | — | ❌ 直译不可用（需 §0.6 重设计） |
| **主选：精确 D 两遍驻留** | Vector ✓：`T.reduce_sum` + v-prefix 链（`docs/Tilelang.language/规约操作/T.reduce_sum.md`、`docs/Tilelang.language/数学操作/T.vmul.md`/`T.vsub.md`/`T.vadd.md`/`T.vdiv.md`/`T.vsqrt.md`），无 MAC 段 | ✓ `block_m·D·20 B ≤ 192KB`（§4.5 逐 workload 复算：最大 160KB） | ✓ 尾轴连续 stride = 1；行块是连续 span（无 padding）；fp16 向量宽 ×8 / fp32 ×4 的尾 lane 浪费见 §1.6.3 | ✓ `M/D/block_m` 全为工厂期常量；`T.serial` 边界为 Python 编译期常量 | 单任务内是顺序依赖链（reduce → center → reduce → rstd → epilogue）；装载/计算重叠交编译器 auto multi-buffer（不支持显式 `T.Pipelined`，理由见 §6.3） | 无跨核归并（行独立）→ 无 flag / workspace 成本 | ✅ **主选** |
| 单遍矩式 | Vector ✓ | 同上 | 同上 | 同上 | 同上 | 同上 | ❌ 数值否决（R1#2 的 D-1 证据：最大相对误差 6.0e3、违反率 100%） |
| Welford 在线 | 合并链为逐 tile 标量序列更新 → Vector 失分 | 同上 | 同上 | 同上 | 依赖链长，融合性差 | 同上 | ❌（R2） |
| D-tile 流式两遍 | Vector ✓ | 更松（D 无上限） | ✓（`block_d` 切片） | ✓ | ✓ | 同上 | ❌ 域内访存 +50%（R3）；仅 `D > 9216` 时备用（§9.3） |
| 转置布局（D,M） | 归约轴变跨步 → 与 TRAP-C10-strided-gather（跨步 gather 向量指令缺失）冲突 | — | — | — | — | — | ❌（§1.6.3 候选 4） |

**调研结论**：选定 **「中心化两遍方差 + 整行（全 D）UB 驻留 + 精确 D + fp32 中转链 + 单位仿射消除」算法族**（候选 0 的数学内核 + 候选 1/5/6/7/8 的等价化简与适配）。关键依据：R3 表中该族的 GM 流量 `2·M·D·eb` 为全候选最小（流式 +50%、基线含 host pad 整遍），R4 全项亲和（Vector 单元、容量 160KB ≤ 192KB、连续 span、静态边界、无跨核结构），且 R1 中候选 2 被 D-1 机器证据数值否决、候选 3/4 在驻留域内无收益。与基线的**结构差异一句话**：数学内核不变（中心化两遍），去掉 CUDA 对齐 padding 与 host `F.pad`、kernel 层单位仿射与 `threads`，并行组织从 grid 超发改为 persistent 分核，rstd 从近似指令改为组合式。源算法优化手段的意图承接见 §0.6 末尾汇总表（逐项落实，无静默丢弃）。调研范围：`algorithm-candidates.md` ALG-layernorm / ALG-rmsnorm 命中行、`examples/norm/` 四个同类实现、pattern-library 五条目、evolution 队列 VP-2026-0043/0044、源码 §0.3/§0.4、R2 结构判据。

**设计期估算（D-2 roofline）**：估算下界 = max(流量项, 发射项, 容量项)，按 dispatch 代表 workload = image-g32（`x_shape` (8,128,32,32)、num_groups 32、fp16 → `M = 8×32 = 256`、`D = (128/32)×1024 = 4096`、`block_m = 2`、`num_logical = 128`、`num_kernels = min(128, 40) = 40`、`num_local_tasks = 4`）：

- **流量项**：`2·M·D·eb = 2×256×4096×2 = 4.19 MB`（读 x + 写 y；kernel 不读 weight/bias）。按 CONST-mte2-degradation 的 1M 档 33.9 GB/s/核 × 40 核（本机实查纯 Vector 核数，§5.5）= 1.356 TB/s → **3.1 µs**；若按该条目的 L2 驻留口径 43.1 GB/s/核（bench harness 同输入逐 launch 复用）= 1.72 TB/s → **2.4 µs**。
- **发射项**：向量 op 数/任务 ≈ 12（fp16 路径：vcast、reduce_sum×2、vmul×3、vsub、vadd、vsqrt、vdiv、vcast 收尾）× `num_local_tasks = 4` × 0.5 µs/op（CONST-vector-launch-overhead）= **24 µs**（**该项主导**；fp32 路径无 2 次 vcast → 10 op/task）。
- **容量项**：0（纯 Vector，无跨引擎往返，无 CONST-store-fixpipe-gm-only 类约束）。
- **估算下界: ≈24 µs（流量 3.1 + 发射 24 + 容量 0，引用 CONST-mte2-degradation〔tilelang 0.1.2+ed787bb / 2026-09-07 实测，910B2C 48 核口径，本机 40 核为线性外推，属未实测假设〕、CONST-vector-launch-overhead〔0.1.2+3a214cde / 2026-09-09〕、CONST-capacity-910B2C〔UB 192KB / ub_cap 196608 B 本机实查一致〕、CONST-aicore-910B2C〔910B3 = 20 aicore / 40 aivector，本机实查一致〕）。**
- **模型保守性说明（诚实标注）**：0.5 µs/op 是"固定发射成本主导"的保守模型（其原始语境是小块 op）；按同族前作实测 baseline 折算，同规模 workload 的向量链实际暴露量约低 2 倍量级（ada_layer_norm 2048×4096 fp16 baseline 90.8 µs ÷ 其 19 op×22 task 的模型值 209 µs ≈ 0.43），故 24 µs 应视为**保守上界**而非常规意义的严格下界。Stage 4 Phase 1 按 `hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行，并据实测反解本档的真实 op 单价。
- **参数敏感方向（指导 Stage 4）**：发射项 ∝ `num_local_tasks`（每核串行轮数），故 **`block_m` 是本设计的第二杠杆**——auto multi-buffer 关闭口径下 UB 占用降至约 1.10 至 1.12 × 手工预算（11.2 B/元素），`D = 4096` 时 `block_m` 可由 2 提至 4 → `num_local_tasks` 4 → 2，发射项减半。该 A/B 计划见 §1.6.3（参数级，不触发结构级实验裁决三件套）。
- **其余 workload 的估算**（同法，流量项 / 发射项取大）：smoke-g8（16×256，bm=8，1 轮）≈ **6 µs**（发射主导）；wider-channel-g32（128×6272，bm=1，4 轮）≈ **24 µs**（流量 3.2MB ≈ 2.4 µs）；tail-spatial-g16（64×7200，bm=1，2 轮）≈ **12 µs**。
- **kernel 级 vs op 级口径提示**：本估算只覆盖 kernel；op 层的 per-channel affine（`y·w + b`）是**独立的一次逐元素 pass**（读 y + 读 w/b + 写 y，代表 workload 再 +4.19MB ≈ +3 µs）与一次额外 launch，这是 affine 分层接口的固有代价（GPU 侧同样如此），不在本 kernel 的设计范围内（§9.1 风险行）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 调研选定的算法族（中心化两遍 + 全 D 驻留 + 精确 D + fp32 中转 + 单位仿射消除）。逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）：

| # | 优化项 | 原式（源/基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|----------------|-----------|-----------|---------|
| O1 | 除法转乘编译期倒数 | `mean = S1 / D`；`var = S2 / D`（真除法） | `mean = S1 · fl(1/D)`；`var = S2 · fl(1/D)`（`D_inv` 为工厂期 Python 常量） | `D` 为编译期常量；`fl(1/D)` 相对误差 ≤ 2⁻²⁴（`D` 为 2 的幂时精确）。**D-1 机器验证**：31 案例（合计约 6.8 万元素）、最大 ulp 差 **1**、候选 vs 基线超容差元素数 **0**（逐案例均为 0） | `(block_m,1)` 规模上 2 次 `vdiv` → 2 次 `vmul`；buffer 极小故收益有限，但零风险且与 pattern-library `layout.md` 的 PL-1.5「乘编译期常数倒数」先例一致（同时规避 CONST-npuir-f32-division-not-ieee-rounded 记录的 fp32 除法非正确舍入特性） |
| O2 | rstd 组合式（精度路线） | `rstd = T.rsqrt(var+eps)`（npuir 上 lower 为**近似** Vector 指令） | `s = T.vsqrt(var)`；`rstd = T.vdiv(s, var)`，即 `sqrt(v)/v` | 实数域恒等：`sqrt(v)/v = v^0.5/v = v^-0.5 = 1/sqrt(v)`。**D-1 机器验证**：① 分段证据（var 扫描 [1e-30, 1e6]）组合式相对 fp64 最大相对误差 **6.10e-8**，与精确 `1/sqrt` 的 7.83e-8 同量级；而近似指令的实测上界 **2.87e-3**（>fp32 门禁 rtol 1e-5 的 287 倍）；② 全管线 31 案例：候选 11/31 满足准则 (a)（与"近似基线"在容差内），**20/31 由准则 (b) 承接**（候选相对 fp64 不比近似基线差），无案例两准则皆失败 → 判定采用准则 (b)（候选显著优于基线） | 精度收益（fp32 门禁从必然大面积违反 → 达标）；代价 = `(block_m,1)` 上 +1 个 `vsqrt`（`vdiv` 与源 `rsqrt` 同为 1 个 op） |
| O3 | 中心化值复用（公共子表达式消除） | epilogue 从原始 dtype 副本重推：`(cast(x_local) − mean)·rstd·w + b` | 统计相保留 `d = x_f32 − mean`（fp32 驻留），epilogue 直接用 `d` | 恒等——`d` 与重推结果**逐位相同**（fp16/bf16 → fp32 上cast 无损、`mean` 为同一 buffer 值）；D-1 31 案例 max ulp **0**、超容差元素数 **0** | 每任务省一次 `(block_m, D)` 级 epilogue 重推中的重复上cast与重减；并使 E1 段的表达式从"复合标量式"变为纯 v-op 链（可向量化，见 §1.6.2） |
| O4 | 精确 D 方差（去 256 对齐 padding 与填充校正） | `D_pad = align_up(D,256)`；`var = (Σ_{j<D_pad}(x̃_j − m)² − pad·m²)/D`（`x̃` 为零填充副本）+ host `F.pad` | `var = (Σ_{j<D}(x_j − m)²)·fl(1/D)`（精确 D，无填充，无 host pad） | **数学恒等**：填充列 `x̃_j = 0` 对中心化平方和的贡献是 `(0−m)² = m²`，故 `Σ_{j<D_pad}(x̃−m)² = Σ_{j<D}(x−m)² + pad·m²`，减回 `pad·m²` 后实数域严格相等。**D-1 机器验证**：31 案例中 **29/31 满足准则 (a)**（良态位形下两式超容差元素数 **0**、最大 ulp 差 **3**）；2/31（padding 抵消角点）由准则 (b) 承接且候选大幅更优——`D=2, pad=254, mean≈3e4` 时基线相对误差 9.95e-1 vs 精确式 9.39e-3；`D=196, pad=60` 时基线 4.57e2 vs 精确式 1.84e0（**基线自身的填充校正在 fp32 下发生灾难性抵消：`pad·m² ≈ 2.3e11` 与其 fp32 ulp ≈ 1.6e4 远大于真实的 `Σ(x−m)²`**） | 去 host `F.pad` 的整遍张量拷贝（manifest 四档中 `D` 非 256 对齐的两档——wider-channel-g32（`D = 6272`）与 tail-spatial-g16（`D = 7200`）——各省一整遍输入流量，另加输出写回的一遍）+ 去填充列无效计算（`pad/D` 比例的统计相开销）+ 免 `D_padded` 缓冲加宽；并消除基线在非对齐 D 下的数值脆弱性 |
| O5 | 单位仿射消除 | kernel 层 `y = ((x−m)·rstd)·weight[j] + bias[j]`（本迁移路径恒 `weight ≡ 1`、`bias ≡ 0`） | kernel 层 `y = (x−m)·rstd`（省略乘加），per-channel affine 仍由 op 层施加 | **IEEE 逐位恒等**：`a × 1.0 = a`、`a + 0.0 = a`（乘 1 与加正零均不改变数值；唯一差别是 `-0.0 + 0.0 → +0.0` 的符号位，而 golden 侧 `F.group_norm(...).to(dtype)` 在相同位置产生同样的正零，符号位差异不构成数值差异）。**D-1 机器验证**：31 案例 max ulp **0**、超容差元素数 **0**。同时 N 案例数据（含 ±inf/NaN、次正规、饱和）逐位一致 | 每元素 −1 `vmul` −1 `vadd`（代表 workload 约 −2/12 向量 op ≈ −17% 向量时延），并避免为行广播 affine 引入标量索引循环或 `vbrc` 中转缓冲（§0.5 已量化两条实现路线的代价） |

**D-1 等价性机器验证结果表**（脚本 `examples/group_norm/_group_norm_kernel/verify_equiv.py`，2026-10-15 执行，torch CPU 2.9.0+cpu、fp64 参照 + 角点值；判定双准则：**(a)** 候选式 vs 基线式在目标 dtype 容差内（超容差元素数 = 0），或 **(b)** 候选式对 fp64 精确数学的误差不劣于基线式（不超过目标 dtype 一个舍入单位）；每条采纳项以"仅回退该项"的方式隔离比较）：

| 采纳项 | 案例数 | 满足准则 (a) | 最大 ulp 差（候选 vs 基线，目标 dtype） | 候选 vs 基线超容差元素数 | 相对 fp64 的最大劣化 | 两准则皆失败 | 结论 |
|--------|-------|-------------|--------------------------------------|------------------------|---------------------|-------------|------|
| O1 除法→乘倒数 | 31 | 31/31 | 1 | 0 | +0.0e0 | 0 | **EQUIV_PASS** |
| O2 rstd 组合式 | 31 | 11/31（余 20 例由 (b) 承接） | 48008（近似基线的固有偏差） | 8174 | +0.0e0 | 0 | **EQUIV_PASS**（准则 (b)：候选相对 fp64 违反数 4884 vs 近似基线 35079） |
| O3 中心化值复用 | 31 | 31/31 | 0 | 0 | +0.0e0 | 0 | **EQUIV_PASS** |
| O4 精确 D 方差 | 31 | 29/31（2 例 padding 抵消角点由 (b) 承接） | 3220099892（角点处基线自身发散） | 588 | +3.3e-7（一例的 1 个 fp32 舍入单位级） | 0 | **EQUIV_PASS** |
| O5 单位仿射消除 | 31 | 31/31 | 0 | 0 | +0.0e0 | 0 | **EQUIV_PASS** |

形态覆盖：`randn` 于 `D ∈ {13, 64, 196, 256, 1152, 4096, 6272}` × 三 dtype（21 例）、大均值小方差（fp32 `3e4±0.5` / fp16 `6e4±16`）、小方差大均值 `1e3±1e-3`、常量行（`var = 0 → rstd = 1/sqrt(eps)`）、全零行、次正规（×1e-5）、饱和极值（×65504）、`±inf`/`NaN` 混合行、padding 抵消角点（`D = 2` 与 `D = 196` 配 `mean ≈ 3e4`）。残余的"vs fp64 违反数"由**两臂共享**的 fp32 行累加误差主导（候选与基线同值），非采纳项引入——脚本末尾已显式标注该口径。

**优化结论**：采纳 **O1 / O2 / O3 / O4 / O5** 五项（全部 EQUIV_PASS）。优化后公式（§3.1 的唯一输入；`r_D = fl(1/D)`，`fl32` 为 fp32 域运算）：

$$
\mu = S_1\cdot r_D,\qquad d = x - \mu,\qquad \sigma^{-1} = \frac{\sqrt{\left(\sum_j d_j^2\right)\cdot r_D + \varepsilon}}{\left(\sum_j d_j^2\right)\cdot r_D + \varepsilon},\qquad y = d\cdot\sigma^{-1}
$$

（`d` 与 `(Σd²)` 只算一次并被复用；kernel 层无 affine，per-channel affine 在 op 层。）

**否决项及依据**（负向断言举证，`negative-claim-evidence.md` §1；均附机器证据或 pattern-library 条目，非纸面论证）：

| 否决候选 | 否决理由 | 证据 |
|----------|----------|------|
| C1 单遍矩式 `var = E[x²] − (E[x])²` | 大均值 / 小方差位形下 fp32 灾难性抵消，精度门禁不可达 | **本任务 D-1 机器证据**（`verify_equiv.py` 的 rejection 段，31 案例）：`largemean-fp16` 矩式相对误差 **6.018e3**（两遍式 0.0）、超容差 **1536** 元素；`largemean-fp32` 矩式 2.467e2（两遍式 1.000e0）；`tinyvar-fp32` 矩式 1.042e0 且违反 768 元素；padding 抵消角点 `D=2` 矩式 `7.806e1` vs 两遍式 `2.110e-1`。交叉证据：`algorithm-candidates.md` 的 ALG-layernorm R1 行（矩式须评估 catastrophic cancellation）与 evolution 队列 VP-2026-0043 的定量边界（另一任务独立实测同向）；现存的矩式写法在 `examples/norm/group_norm.py` 第 43 至第 46 行（非迁移源，仅形态对照） |
| C2 原生 fp16/bf16 计算链（不经 fp32 中转） | bf16 在 npuir 上**无 Vector 算术**（v-mul/v-sub/v-add/reduce_sum/vdiv/vsqrt 的 bf16 列均为 ×）；fp16 原生链与 torch golden 的 fp32 opmath 舍入路径分歧 | `docs/Tilelang.language/数学操作/T.vmul.md`、`T.vadd.md`、`T.vsub.md`、`T.vdiv.md`、`T.vsqrt.md` 与 `docs/Tilelang.language/规约操作/T.reduce_sum.md` 的 dtype 表（bf16 列 ×）；pattern-library `traps-compiler.md` 的 TRAP-fp16-opmath-golden（fp32 中转链通解，≤1 ulp fp16） |
| C3 跨 dtype `T.copy` 充当 cast | 不减 vector pass 且舍入模式不可控 | pattern-library `traps-compiler.md` 的 TRAP-C12-copy-dtype-cast；`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条（Developer 模式自动插 `hivm::VCastOp`，RoundMode=RINT，不可通过 `T.copy` 定制） |
| C4 kernel 内实现行广播 affine（`weight[j]` 标量索引或 `vbrc` 中转） | 本迁移路径恒为单位元（收益为负），且两条实现路线的代价与可测性均差 | §0.5 对应行（路线 (a) 是 `examples/norm/group_norm.py` 第 48 至第 53 行的遗留标量循环形态；路线 (b) 需 rank 一致的 `(1, D)` 源缓冲与额外 dtype 广播缓冲，见 `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 与示例 2）；O5 已给出单位元下省略的逐位恒等机器证据 |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点范围：本设计实现方案（§3.3 伪代码）中的**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。API 佐证均指向 `docs/Tilelang.language/` 下的具体文档（已逐份打开核对 dtype 表与 shape 规则）。

| # | 计算点 | 源实现形态（GPU / 遗留原型） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|------------------------------|--------------------------|---------|--------------|
| 1 | x 上cast fp32 | `T.Parallel(block_m, D_padded)` 内逐元素 `T.cast`（源 L2） | `T.vcast(stage, x_f32, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`：f16→f32 rint ✓、bf16→f32 rint ✓） | ✅ | — |
| 2 | 行和 S1 / S2 | `T.reduce_sum(x_f32, acc, dim=1)`（源 S1/S2） | 同 API：`T.reduce_sum(x_f32, s1, dim=1)`（`docs/Tilelang.language/规约操作/T.reduce_sum.md`：`(M,N)→(M,1)`、fp32 ✓） | ✅ | — |
| 3 | mean / var 的逐行标量式 | `T.Parallel(block_m)` 循环 `mean_val[i] = acc[i] / float(D)`（源 S1/S2） | `T.vmul(s1, D_inv, mean)`、`T.vmul(s2, D_inv, var)`（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量广播 ✓） | ✅ | — |
| 4 | 中心化 + 平方 | `T.Parallel(block_m, D_padded)` 复合标量式 `(x−m)·(x−m)`（源 S2） | `T.vsub(x_f32, mean, x_f32)`（行广播 `[M,D]−[M,1]`，`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2 ✓、§2.3 in-place ✓）+ `T.vmul(x_f32, x_f32, sq_f32)` | ✅ | — |
| 5 | `var + eps` 与 rstd | `T.Parallel(block_m)` 标量式 `T.rsqrt(...)`（源 S2） | `T.vadd(var, eps, var)`（标量广播，`docs/Tilelang.language/数学操作/T.vadd.md`）+ `T.vsqrt(var, s2)`（`T.vsqrt.md` fp32 ✓）+ `T.vdiv(s2, var, rstd)`（`docs/Tilelang.language/数学操作/T.vdiv.md` §2.2.2 同 shape ✓、§2.3 in-place ✓） | ✅ | — |
| 6 | 归一化 + kernel 层 affine | `T.Parallel(block_m, D_padded)` 复合标量式含 `weight[j]` / `bias[j]` **标量索引读**（源 E1；遗留原型 `examples/norm/group_norm.py` 第 48 至第 53 行同形） | `T.vmul(x_f32, rstd, x_f32)`（行广播 `[M,D]×[M,1]` ✓）；单位仿射整体省略（O5）；per-channel affine 在 op 层由 torch 广播算子完成（`examples/TileOPs/tileops/ops/norm/group_norm.py`） | ✅ | — |
| 7 | 下cast + 写出 | 赋值隐式截断 + `T.copy` | `T.vcast(x_f32, stage, round_mode="rint")`（f32→f16 / f32→bf16 rint ✓）+ `T.copy` 双显式 slice（`docs/Tilelang.language/内存操作/T.copy.md` §2.4） | ✅ | — |
| 8 | 任务循环（persistent grid-stride） | 源为 grid 超发（每 block 一任务，无核内循环） | `for i in T.serial(num_local_tasks)` + `if block_id < num_logical` 守卫 | ❌ 保留 | **block 级任务映射计算**：每任务数个标量索引（`block_id` / `off_m`），非逐元素热点；该 `serial` 循环本身是 §0.6 R1 persistent 分核意图的载体，无逐元素等价的向量 API；同族前作同款结构（`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`） |
| 9 | 行尾 `real_m` | 源无行谓词（靠 `block_m = 1`） | `real_m = T.min(block_m, M − off_m)`（标量，1 次/任务，`docs/Tilelang.language/比较操作/T.min.md` 标量语义 ✓） | ❌ 保留 | **动态边界元数据**：`M % block_m ≠ 0` 时尾任务的行数；每任务 1 次标量 min 加两次 slice 界，深流水档被 MTE2 窗口隐藏（pattern-library `constants.md` 的 CONST-copy-floor-method：`mte2_ratio ≥ 0.95` 时标量/分支开销已被隐藏）；静态化需为尾任务复制整份 kernel 体，复杂度不值 |
| 10 | `D_inv` / `num_logical` / `num_kernels` / `num_local_tasks` / `D_padded` | host Python | host 层编译期常量（trace 期折叠为 `IntImm`，pattern-library `layout.md` 的 PL-1.5） | ❌（非 kernel 内） | **host 侧元数据计算**：不在 kernel 内，无向量化问题 |
| 11 | `T.copy` 的索引式区域（源 `T.copy(x[pid_m*block_m, 0], shared_buf)`） | 区域由 dst 形状推定的隐式形式 | 双显式 slice `T.copy(x[off:off+real_m, 0:D], stage[0:real_m, 0:D])` | ✅（形态替换） | —（规避 pattern-library `traps-compiler.md` 的 TRAP-T-copy-region-semantics；dst 零起点规避 TRAP-UB-dst-align） |

**向量化结论**：逐元素计算点（上cast / 中心化 / 平方 / 加减乘 / 开方除法 / 行归约 / 归一化 / 舍回 / 搬运）**已全部向量化**（替代 API 均在 `docs/Tilelang.language/` 有 dtype 与 shape 佐证，见上表）；保留的 3 类标量/循环为：**任务映射索引（#8）**、**动态边界 `T.min`（#9）**、**host 侧元数据（#10）**，逐项理由如上（均为 block 级 / 每任务 1 次 / kernel 外，无"实现简单"类理由）。§6 循环结构与本表一致：kernel 内**不存在**逐元素标量循环（源 L2/S2/E1 的四个 `T.Parallel` 元素级循环全部被 v-op 链替代）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

规约类算子必选候选：**水平归约（lane 映射到归约轴，用向量 reduce）vs 垂直扫描（lane 映射到独立行，串行步进归约维）**——两者可共用同一条张量轴但 lane 语义不同，故分别枚举；另外枚举 D 维分块流式与转置布局。两遍 vs 在线与 lane 映射的交互已在 §1.6.0 R2 覆盖（驻留域内在线化无收益）。

**轴质量评分**（主选轴 = D，最内维；昇腾 Vector 宽度：fp16/bf16 ×8、fp32 ×4）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | fp16：D = 4096（512×8 ✓）、6272（784×8 ✓）、7200（900×8 ✓）、1152（144×8 ✓）、256 / 64（✓）、196（24×8+4，尾 4）、13（1×8+5，尾 5）；fp32 ×4：196（49×4 ✓）、13（3×4+1） |
| 尾 lane 浪费率 | 域内最差 D = 13：fp16 (2×8−13)/13 = **23%**（仅在防御性微型用例出现）；D = 196：fp16 (25×8−196)/196 = **2.0%**；manifest 四档（256/4096/6272/7200）全部 **0%** |
| 累加链形态 | 归约维 = 尾轴连续、stride = 1；行广播 v-op 为文档形态（`[M,D]` 与 `[M,1]`，见 `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2）；**无跨步系数** |
| repack 代价 | **无**（I/O 原生行主序 = 核内布局，零重排） |
| UB 容量影响 | 全 D 驻留 → `block_m ≤ 9216 / D`（§4.5）；对价换来 x 的单遍 GM 扫描（全候选最小流量） |

**候选矩阵**（弃选项均给量化理由与证据）：

| # | 布局方案 | 向量化轴 | repack / 结构路径 | 预估收益 / 代价 | 采纳？ |
|---|---------|---------|------------------|-----------------|--------|
| 1 | I/O 原生 `(M, D)` 行主序 + **全 D 驻留 + 水平归约** | D（尾轴连续；`T.reduce_sum dim=1` + 行广播 v-op） | 无 | GM 流量 `2·M·D·eb`（全候选最小）、x 单遍扫描、行块为连续 span（DMA 粒度 = `block_m·D·eb`） | ✅ **主选** |
| 2 | 垂直扫描（lane → 独立行，串行步进 D） | 行 | 无 | 归约维访问 stride = D → 无向量宽度可利用；跨步 gather 向量指令在本工具链缺失（pattern-library `traps-compiler.md` 的 TRAP-C10-strided-gather，状态 verified）→ 退化标量 | ❌ |
| 3 | D-tile 流式（`block_d < D`，`T.serial` 分块 + `clear=False` 累加 + 输出遍重读 x） | D（分 tile） | 无 | UB 更松（D 无上限）但 **x GM 重读一遍 → 访存 `3·M·D·eb`（+50%）**；仅当 `D > 9216`（整行无法驻留）时必要 | ❌ 域内严格更差（R3 表）；留作超驻留 fallback（§9.3） |
| 4 | 转置布局 `(D, M)`（host permute 或核内 `T.transpose` 链） | M | host permute（pattern-library `layout.md` 的 PL-1.3 实测 106 至 147 µs，通常净亏）/ 核内 `T.transpose` 二轴交换链（PL-1.1 实测 µs 级） | 归约轴变为跨步 → repack 纯开销、零收益；核内转置链本身可行（非"GM 级重排"），但本算子没有需要它的广播轴不匹配问题 | ❌ |
| 5 | `(M, D)` 行主序 + **核内改行块形状**（如把两行折成一维 span）以放大 DMA 粒度 | D | 无 | 与主选等价（主选的行块在无 padding 下已是连续 span，DMA 粒度已由 `block_m·D·eb` 给出），无额外收益 | ❌（与主选重合，不单列） |

**主选的多缓冲 / block_m 参数级 A/B（非结构级，故不触发结构级实验裁决三件套）**：主选结构与 `block_m ∈ {1, 2, 4}` 均相容，唯一落入未实证区间的量是 **auto multi-buffer 的 UB 占用口径**对本 kernel 三缓冲形态的适用性（决定 `D = 4096` 时 `block_m = 2` 还是 4）：

- **主选（默认实现，进 Stage 3）**：编译器默认 auto multi-buffer 打开，UB 预算按**实测多缓冲平台值 20 B/元素**（pattern-library `traps-compiler.md` 的 TRAP-UB-multibuffer-inflation 第三证：fp16/bf16 中转循环 resident 10 B/元素与 14 B/元素两档的 multi-buffered 需求同为精确 20.01 B/元素；同族前作 Stage 3 编译溢出报文 `245760 B = 20·3·4096` 精确吻合本 kernel 的缓冲结构）→ `D = 4096` 时 `block_m = 2`、`num_local_tasks = 4`。
- **备选（参数级变体，Stage 4 可切换）**：`pass_configs` 关闭 auto multi-buffer（`npuir.enable_auto_multi_buffer=False`），UB 按约 1.10 至 1.12 × 手工预算（11.2 B/元素，pattern-library `constants.md` 的 CONST-capacity-910B2C 双口径；该条目的第六轮反例表明系数具结构依赖，显式 alloc 结构实测可低至 0.98×）→ `D = 4096` 时 `block_m` 可提至 4、`num_local_tasks` 4 → 2（发射项减半，见 §1.6.0 估算的敏感性方向），代价是失去编译器的自动装载/计算重叠（本 kernel 为发射主导而非流量主导，预期净收益为正，但需实测）。
- **裁决计划**：代表 shape = manifest 四档（smoke-g8 / image-g32 / wider-channel-g32 / tail-spatial-g16）；指标 = msprof op Task Duration（每档 ≥20 次采样、去首 5 次预热）＋该档 UB 编译需求（`ub overflow` 报文反推）；**判定阈值**：备选在该档 Task Duration < 主选 × 0.95（且精度门禁不回归）才翻转该档参数，否则维持主选；按 shape 分派（per-D 表）可达总体最优，故允许多档各自选择；裁决结果经附录补记回写本节，实测出的真实 UB 单价同步作为 pattern-library 候选常数。
- **不启动设计期探针（D-3）**：该决策只影响参数（`block_m` / 编译开关），不改结构；结构级结论（驻留式 vs 流式、水平 vs 垂直归约）的判定裕度 = 50% 访存差 + 亲和的绝对差，远超任何未实证常数的不确定区间，无需探针预算。

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：源 kernel 把 CTA 内 `threads = 128` 映射到行内元素（`block_m = 1`，行内归约经线程级归约塌缩），即 GPU **也**把最内维当作归约 / 向量轴——但那是 CUDA warp 布局（coalesced 访问 + warp shuffle）的产物，且它同时被 `block_m = 1` 绑定（§0.4 #6）。本节依据 NPU 的 Vector 宽度、`[M,D]↔[M,1]` 广播形态、UB 驻留容量与连续 span DMA 独立评估后**结论一致、理由不同**（NPU 侧：尾轴连续 stride = 1 + 行广播 v-op + 全 D 驻留单遍扫描 + 行块连续 span）。

**布局决策结论**：选定 **#1：核内布局 = I/O 原生 `(M, D)` 行主序、向量化轴 = D（尾轴）、水平归约（`T.reduce_sum dim=1`）、repack = 无**；该结论同步落入 §3.3 伪代码与 §6 循环结构（归约与行广播的内层向量维 = D；所有 v-op / reduce 的 buffer 形状 = `(block_m, D)`，与 §4.3 一致）；弃选候选的量化理由见上表（#2 依赖缺失 API 且有实证条目、#3 访存 +50%、#4 repack 零收益）。多缓冲口径相关的 `block_m` 取值属**参数级**，主选与备选差异与裁决计划已列明。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.reduce_sum` + v-prefix 逐元素链），无 matmul | 不涉及 Cube / L0A / L0B / L0C / L1 分形约束 |
| 归约 | `T.reduce_sum dim=1`（行归约） | Developer / Expert 均可用，无需手动控制流水 |
| 内存层级 | 仅 GM ↔ UB（经 `T.alloc_shared`，Developer 映射 UB） | 无跨引擎（Cube↔Vector）协作 |
| 同步 | 单 kernel 内顺序依赖链、无核间协作（行独立） | Developer 自动同步即可，无需 `sync_block_set/wait` |
| 先例 | 同族 row-reduction 迁移前作（`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`）为 Developer 模式一次通过；`examples/norm/group_norm.py`、`examples/norm/layer_norm.py` 亦为 Developer | Developer 已验证可行，无阻塞证据 |
| 用户指定 | op_requirements `programming_mode: developer` | 一致 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（映射 UB；v-op 操作数必须在 UB，见 `docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1） |
| 计算方式 | v-prefix API（`vcast` / `vmul` / `vsub` / `vadd` / `vsqrt` / `vdiv`）+ `T.reduce_sum` + 标量 `T.min` 尾块界 |
| 同步 | Developer 自动同步（无手动 `sync_block_set` / `sync_block_wait` / `pipe_barrier`） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维；三维 Kernel 不支持，见 §3.5.1） |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 的优化后公式（fp32 计算域；`r_D = fl(1/D)`）；下表步骤编号与 §3.3 伪代码一一对应：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_f32 = fp32(x)` | 上cast（fp32 输入为同 dtype 直拷） |
| 2 | `S1 = Σ_j x_f32[r, j]` | 行和 |
| 3 | `mean = S1 · r_D` | 均值（采纳项 O1） |
| 4 | `d = x_f32 − mean` | 中心化（保留 d，采纳项 O3） |
| 5 | `S2 = Σ_j d[r, j]²` | 中心化平方和 |
| 6 | `var = S2 · r_D + eps` | 有偏方差 + eps（精确 D，采纳项 O4） |
| 7 | `rstd = sqrt(var) / var` | `1/sqrt(var)` 的组合式实现（采纳项 O2） |
| 8 | `t = d · rstd` | 归一化（行广播） |
| 9 | `y = dtype_round(t)` | 单次舍回目标 dtype（kernel 层无 affine，采纳项 O5） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数与形态 | 模式 | 依据 |
|------|----------|-------------|-----------|------|------|
| 任务映射 | `block_id = i·num_kernels + cid` | `for i in T.serial(num_local_tasks)` + `if block_id < num_logical` | 静态边界（host 编译期常量折叠） | Developer | pattern-library `layout.md` 的 PL-1.5 折叠先例；同族前作同款 |
| 行尾 | `real_m = min(block_m, M − off_m)` | `T.min(block_m, M - off_m)` | 标量（1 次/任务） | Developer | `docs/Tilelang.language/比较操作/T.min.md`（标量语义 ✓） |
| 载入 x | GM → UB | `T.copy(x[off_m : off_m + real_m, 0:D], stage[0:real_m, 0:D])` | src / dst **双显式 slice**，dst 零起点 | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 1 与 §2.2.2 规则 2/4（尾块借用动态 extents） |
| 上cast | fp16/bf16 → fp32 | `T.vcast(stage, x_f32, round_mode="rint")` | `(bm,D) → (bm,D)` 同 shape | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（f16→f32 rint ✓、bf16→f32 rint ✓；shape 须一致） |
| 行和 | `S1`、`S2` | `T.reduce_sum(x_f32, s1, dim=1)`、`T.reduce_sum(sq_f32, s2, dim=1)` | `(bm,D) → (bm,1)`，`clear` 默认 True | Developer | `docs/Tilelang.language/规约操作/T.reduce_sum.md` |
| 均值 / 方差 | `S1·r_D`、`S2·r_D` | `T.vmul(s1, D_inv, mean)`、`T.vmul(s2, D_inv, var)` | `(bm,1) × 标量` | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量广播 |
| 中心化 | `d = x − mean` | `T.vsub(x_f32, mean, x_f32)` | `[bm,D] − [bm,1]` 行广播、in-place | Developer | `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2 行广播、§2.3 in-place |
| 平方 | `sq = d²` | `T.vmul(x_f32, x_f32, sq_f32)` | out-of-place（保留 d） | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` |
| 加 eps | `var + eps` | `T.vadd(var, eps, var)` | `(bm,1) + 标量` | Developer | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2 |
| rstd | `sqrt(v)/v` | `T.vsqrt(var, s2)` → `T.vdiv(s2, var, rstd)` | `(bm,1)` 同 shape 链；`s2` 复用为开方暂存 | Developer | `docs/Tilelang.language/数学操作/T.vsqrt.md`（fp32 ✓）、`T.vdiv.md` §2.2.2/§2.3 |
| 归一化 | `d · rstd` | `T.vmul(x_f32, rstd, x_f32)` | 行广播 in-place | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 |
| 舍回 | fp32 → dtype | `T.vcast(x_f32, stage, round_mode="rint")` | f32→f16 / f32→bf16 rint ✓；`stage` 复用 | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`；TRAP-fp16-opmath-golden 通解 |
| 写出 | UB → GM | `T.copy(stage[0:real_m, 0:D], y[off_m : off_m + real_m, 0:D])` | 双显式 slice，只写有效行 | Developer | `docs/Tilelang.language/内存操作/T.copy.md` |
| fp32 路径 | — | 同上但**无 vcast**：`T.copy(x[...], x_f32[0:real_m, 0:D])` 直拷 + `T.copy(x_f32[0:real_m, 0:D], y[...])` 直写 | 同 dtype copy | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.1（同 dtype） |
| kernel 层 affine | 单位元省略 | 不生成对应 op（`weight` / `bias` 形参保留但不读） | 见 §0.5 对应行与采纳项 O5 | Developer | §1.6.1 O5 的 D-1 逐位恒等证据 |

### 3.3 计算伪代码

> 接口保持：工厂签名 `_group_norm_kernel(M, D, eps, dtype)` 与 prim_func 参数名/顺序 `main(x, weight, bias, y)`、`out_idx=[3]` 全部与源一致；`_func(block_m)` 移除 `threads`（后端适配，pattern-library `traps-compiler.md` 的 TRAP-threads-kwarg-noop；NPU wrapper 已按 `(block_m)` 调用，见 `examples/TileOPs/tileops/kernels/norm/group_norm/group_norm.py`）。
> `ALIGNMENT` / `_align_up` 仍从本模块导出（wrapper 的导入契约），但 NPU 侧 `ALIGNMENT = 1`（无行对齐 padding 需求，§0.6 R7），`D_padded` 恒等于 `D`。
> `weight` / `bias` 形参保留以维持 wrapper 调用契约与 `out_idx=[3]` 的位置（同族前作以未使用的 `_dummy` 形参维持 `out_idx` 位置，证明未读形参可编译运行）；本路径 wrapper 恒传单位权重/零偏置，其乘法与加法在 IEEE 下逐位恒等（O5）。

```python
import tilelang
import tilelang.language as T
from tilelang.utils.npu_utils import NPUUtils

ALIGNMENT = 1  # NPU: no row-length alignment padding required (backend constant adaptation)

def _align_up(n, alignment):
    return ((n + alignment - 1) // alignment) * alignment

_UB_CAPACITY_BYTES = 196_608   # 192 KB per Vector core (queried ub_cap == 196608)
_UB_BYTES_PER_ELEM = 20        # measured auto-multi-buffer platform value (transit 10 B/elem resident)
_UB_MAX_ELEMS = 9_216          # 9830 (= 196608 // 20) rounded down to 8 x 1152, >= 12 KB margin
_BLOCK_M_CAP = 8

def _group_norm_kernel(M, D, eps, dtype):
    """Factory signature unchanged from the GPU source."""
    if M < 1 or D < 1:
        raise ValueError(f"M and D must be positive, got M={M}, D={D}")
    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {dtype!r}")

    D_padded = _align_up(D, ALIGNMENT)          # == D on NPU (ALIGNMENT = 1)
    pad_count = D_padded - D                    # == 0 (kept for structural parity)
    D_inv = 1.0 / float(D)                      # O1: compile-time reciprocal
    use_fp32_transit = dtype in ("float16", "bfloat16")
    block_m_default = min(_BLOCK_M_CAP, max(1, _UB_MAX_ELEMS // D))

    vector_cores = NPUUtils.get().get_aicore_num() * 2   # pure Vector: x2 (queried 20 -> 40)

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func(block_m=block_m_default):
        if block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")
        if block_m * D > _UB_MAX_ELEMS:
            raise ValueError(
                f"block_m={block_m} exceeds the UB budget for D={D} "
                f"(max {_UB_MAX_ELEMS} resident elements at {_UB_BYTES_PER_ELEM} B/elem)"
            )
        # host-side compile-time constants (fold to IntImm at trace time)
        num_logical = (M + block_m - 1) // block_m
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        if use_fp32_transit:

            @T.prim_func
            def main(
                x: T.Tensor[(M, D_padded), dtype],
                weight: T.Tensor[(D_padded,), dtype],   # interface only; unit affine in this path
                bias: T.Tensor[(D_padded,), dtype],     # interface only; unit affine in this path
                y: T.Tensor[(M, D_padded), dtype],      # out_idx=[3]
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    stage = T.alloc_shared((block_m, D_padded), dtype)       # x load / rounded result
                    x_f32 = T.alloc_shared((block_m, D_padded), "float32")   # x -> d -> d*rstd
                    sq_f32 = T.alloc_shared((block_m, D_padded), "float32")  # d*d -> sqrt(var) scratch
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            # 1. load + upcast
                            T.copy(x[off_m : off_m + real_m, 0:D], stage[0:real_m, 0:D])
                            T.vcast(stage, x_f32, round_mode="rint")

                            # 2. mean  (O1: multiply by fl(1/D))
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, D_inv, mean)

                            # 3. centred two-pass variance; d is kept (O3), exact D (O4)
                            T.vsub(x_f32, mean, x_f32)          # d = x - mean
                            T.vmul(x_f32, x_f32, sq_f32)        # d * d
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, D_inv, var)
                            T.vadd(var, eps, var)

                            # 4. rstd = sqrt(v)/v  (O2: vsqrt + vdiv, NOT the approximate vrsqrt)
                            T.vsqrt(var, s2)
                            T.vdiv(s2, var, rstd)

                            # 5. normalise (unit kernel affine -> weight/bias omitted, O5)
                            T.vmul(x_f32, rstd, x_f32)

                            # 6. round back once and store valid rows only
                            T.vcast(x_f32, stage, round_mode="rint")
                            T.copy(stage[0:real_m, 0:D], y[off_m : off_m + real_m, 0:D])

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(M, D_padded), dtype],
                weight: T.Tensor[(D_padded,), dtype],
                bias: T.Tensor[(D_padded,), dtype],
                y: T.Tensor[(M, D_padded), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_f32 = T.alloc_shared((block_m, D_padded), "float32")
                    sq_f32 = T.alloc_shared((block_m, D_padded), "float32")
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            T.copy(x[off_m : off_m + real_m, 0:D], x_f32[0:real_m, 0:D])
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, D_inv, mean)
                            T.vsub(x_f32, mean, x_f32)
                            T.vmul(x_f32, x_f32, sq_f32)
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, D_inv, var)
                            T.vadd(var, eps, var)
                            T.vsqrt(var, s2)
                            T.vdiv(s2, var, rstd)
                            T.vmul(x_f32, rstd, x_f32)
                            T.copy(x_f32[0:real_m, 0:D], y[off_m : off_m + real_m, 0:D])

        return main

    return _func
```

> dtype 分支置于工厂层（trace 体之外），规避 TVM script parser 的 if 块变量作用域限制（同族前作同款做法）。
> 尾块说明：v-op / reduce 作用于完整 `(block_m, D)` 缓冲，尾任务 `[real_m:]` 行为未初始化垃圾——按行隔离（`reduce dim=1` 逐行、v-op 逐元素或行广播），输出只写 `[0:real_m]`（§0.6 R3）。
> buffer 复用：`stage` 双角色（装载中转 / 舍回中转）；`sq_f32` 双角色（平方 / 开方暂存），与 §4.3 表一致。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `T.Kernel(一维, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` | ✅ 已核对 | NPU 目标；本设计一维，`(cid, _)` 解包 |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md` | ✅ 已核对 | Developer 映射 UB；v-op 操作数可用 |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1 / §2.2.2 / §2.4 | ✅ 已核对 | GM↔UB ✓；同 dtype ✓；尾块借用动态 extents（规则 2/4）；切片与 `size` 互斥（本设计不用 `size`） |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | ✅ 已核对 | f16↔f32 rint ✓、bf16→f32 ✓、f32→bf16 rint ✓；shape 须一致 |
| `T.reduce_sum` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` | ✅ 已核对 | fp32 ✓ bf16 ×；(M,N)→(M,1) ✓；`dim` 合法 |
| `T.vmul` / `T.vsub` / `T.vadd` / `T.vdiv` / `T.vsqrt` | `docs/Tilelang.language/数学操作/` 下同名文档 | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；行广播 `[M,D]↔[M,1]` ✓；标量 src ✓；in-place ✓；UB 操作数 ✓ |
| `T.min`（标量） | `docs/Tilelang.language/比较操作/T.min.md` | ✅ 已核对 | 标量语义，不支持 shape；本设计仅用于 `real_m` |
| `T.serial` + if 守卫 | 同族前作 `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`；`examples/norm/group_norm.py` 的高性能原型 | ✅ 佐证 | persistent grid-stride 同款；if 体内只写预分配 buffer |
| `@tilelang.jit(out_idx=[3], target="npuir")` | 源算子同款；NPU wrapper 已按 `out_idx=[3]` 移植 | ✅ 佐证 | `y` 在 index 3 自动分配返回 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 是否涉及 | 处理方案 |
|------|---------|----------|
| 不支持三维 Kernel | 否 | 一维 `T.Kernel(num_kernels, is_npu=True)` |
| GPU 专用 API（`threads=` / `cp.async` / `T.Parallel` 标量循环 / `alloc_fragment`） | 是 | 全部不迁移：`threads` 移除；无 cp.async 路径；`T.Parallel` 元素级循环改 v-op 链（§1.6.2）；计算缓冲改 `alloc_shared`（§2.3） |
| GEMM 分形 / 非整除 | 否 | 无 GEMM；行尾由 `T.min` + slice 处理；D 无 256 对齐要求（§0.6 R7） |
| L0C 容量（128KB） | 否 | 纯 Vector，无 L0 使用 |
| 物理核数适配 | 是 | §5.5 三要素（实查 20 → 纯 Vector 40） |
| bf16 无 Vector 算术 | 是 | fp32 中转链（§0.6 R4，必需） |
| UB 多缓冲膨胀 | 是 | §4.5 按实测平台值 20 B/元素 + 工厂 guard |
| host 侧不得改写输入张量内容 | 是 | 本设计 host 侧只做视图/元数据；`ALIGNMENT = 1` 使 wrapper 的 `F.pad` 分支不触发（§0.6 R7） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5 / §0.6）

| 差异项 | GPU 源 | 本项目（Ascend NPU） | 转换方案 |
|--------|--------|---------------------|----------|
| `threads` 参数 | `T.Kernel(grid, threads=threads)` | npuir 无效 | 移除；`_func(block_m)`（wrapper 已适配） |
| `T.alloc_shared` 语义 | CUDA shared memory | UB（192KB/AIV，本机实查 `ub_cap = 196608`） | `block_m` 预算表（§4.5）+ 工厂 guard |
| `T.alloc_fragment` 计算域 | CUDA 寄存器 | v-op 要求 UB 操作数 | 本设计全部 `alloc_shared` |
| `T.Parallel` 元素级循环 | 线程级并行 + 线程内归约塌缩 | v-prefix 向量 op + 硬件行归约 | §1.6.2 全表 |
| 整块 `T.copy`（区域由 dst 形状推定） | `T.copy(x[pid_m*block_m, 0], shared_buf)` | 双显式 slice（区域语义显式） | §0.6 R3；规避 TRAP-T-copy-region-semantics / TRAP-UB-dst-align |
| `T.rsqrt`（标量式） | fragment 上标量 | `T.vsqrt` + `T.vdiv` 组合（`(bm,1)` buffer） | §0.6 R5（规避 TRAP-vrsqrt-plain-precision） |
| `D_padded = align_up(D, 256)` + host `F.pad` | SMEM 拷贝对齐 | 无此约束（连续 span 直接搬运） | 精确 D；`ALIGNMENT = 1`（§0.6 R7） |
| `block_m = 1` 结构性默认 | 行归约均匀 + 规避行尾越界 | 按行硬件归约，无塌缩问题 | UB 预算表驱动（§0.6 R2）+ `T.min` 尾块（R3） |
| kernel 内 `weight[j]`/`bias[j]` 标量乘加 | 行广播 affine（本路径恒单位元） | 单位仿射省略；per-channel affine 在 op 层 | §0.6 R6 / 采纳项 O5 |
| 隐式 dtype 截断赋值 | `x_local[i,j] = <fp32 式>` | 显式 `T.vcast(rint)` + 同 dtype copy | TRAP-fp16-opmath-golden / TRAP-C12-copy-dtype-cast |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` | 极高（同族 row-reduction 已交付迁移） | persistent 分核模板、尾块 `T.min` + 双显式 slice + 垃圾行丢弃、fp32 中转链、UB 预算 guard 与实测 20 B/元素平台值、L0 用例分层设计 |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md` | 极高（同族设计文档） | §0 迁移分析结构、§1.6 调研与优化分析写法、§5.5 分核三要素实践 |
| `examples/norm/group_norm.py` | 高（同算子 NPU 早期原型，非迁移源） | 同签名工厂与 `(M, N)` 语义；其 `T.Parallel` 标量循环、`T.vrsqrt`、索引式 `T.copy` 是本设计要替换的形态（对照用） |
| `examples/norm/layer_norm.py` | 高（同族算子） | `reduce_sum` / 行广播形态、`T.rsqrt` 实调（本设计改组合式的依据之一） |
| `examples/norm/instance_norm.py` | 中（同族算子，含 kernel 层行广播 affine 用法） | 说明 kernel 层行广播 affine 的共享来源；本迁移范围外 |
| `examples/TileOPs/tileops/kernels/norm/group_norm/group_norm.py` | 上下文（本算子的 wrapper 契约） | `_group_norm_wrapped` 4 参调用、`register_fake`、`D_padded` 与 `F.pad` 逻辑（`ALIGNMENT = 1` 下不触发） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, D_padded)`（NPU 侧 `D_padded ≡ D`） | float16 / bfloat16 / float32 | `M = N·num_groups`、`D = (C/num_groups)·spatial_size`，均为工厂期常量（per-build JIT 特化） |
| `weight` | `(D_padded,)` | same_as(x) | kernel 层行广播 affine 形参；**本迁移路径恒为单位权重，不参与计算**（§0.5 / 采纳项 O5） |
| `bias` | `(D_padded,)` | same_as(x) | 同上，恒为零偏置 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(M, D_padded)` | same_as(x) | `out_idx=[3]` 自动分配；wrapper 侧 `y[:, :D]` 截断（`ALIGNMENT = 1` 时 `pad = 0`，截断分支不触发） |

### 4.3 中间缓冲区

**fp16 / bf16 路径**（fp32 中转）：

| Buffer 名 | Shape | dtype | 存储层级 | 作用（角色复用时序） |
|-----------|-------|-------|----------|---------------------|
| `stage` | `(block_m, D)` | 输入 dtype | UB（`alloc_shared`） | ① x 装载中转 → ② 结果舍回中转（双角色） |
| `x_f32` | `(block_m, D)` | float32 | UB | x fp32 → `d`（中心化后保留）→ `d·rstd`（三角色） |
| `sq_f32` | `(block_m, D)` | float32 | UB | ① `d²` → ② `sqrt(var)` 暂存（双角色） |
| `s1` / `mean` / `s2` / `var` / `rstd` | `(block_m, 1)` | float32 | UB | 行统计量（各 `4·block_m` B，相对可忽略） |

**fp32 路径**（无 cast、无 `stage`）：`x_f32` + `sq_f32` + 5 个 `(block_m, 1)`。

驻留元素数口径（决定 `block_m`）：fp16/bf16 = `10 B/元素`（2 + 4 + 4），fp32 = `8 B/元素`（4 + 4）。

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube / L0 / L1 参与）：

GM[x]        --T.copy(双显式 slice，连续 span)--> UB[stage(dtype)]
UB[stage]    --vcast(rint)--> UB[x_f32]        （fp32 输入为 GM -> x_f32 同 dtype 直拷）
（全部计算在 UB 内闭环：reduce -> center -> square -> reduce -> rstd -> normalise）
UB[x_f32]    --vcast(rint)--> UB[stage(dtype)] （fp32 输入直接 x_f32 -> GM）
UB[stage]    --T.copy(双显式 slice)--> GM[y]
```

关键约束：GM 不可直达 fragment；v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1）；本设计不经 fragment、无中间结果落回 GM、无跨引擎传输（容量项 = 0）。

### 4.5 UB 内存预算

**预算口径（双口径并列，遵循 `hardware-cost-model.md` §5 与 CONST-capacity-910B2C）**：

1. **默认档（auto multi-buffer 打开）**：按 pattern-library `traps-compiler.md` 的 TRAP-UB-multibuffer-inflation 第三证——fp16/bf16 中转循环在 `num_local_tasks ≥ 2` 时 multi-buffered UB 需求为**精确 20.01 B/元素**（与 resident 10 B/元素或 14 B/元素无关）；fp32 直连档同为 20.00 B/元素。同族前作 Stage 3 编译溢出报文实测 `245760 B = 20 × 3 × 4096`、`240000 B = 20 × 4 × 3000`，其缓冲结构与本 kernel 完全同形（`stage` + `x_f32` + `sq_f32` + 5 个 `(bm,1)`）。据此取 `_UB_MAX_ELEMS = 9216`（`196608 // 20 = 9830`，下取整到 8 × 1152，留 ≥ 12KB 余量）。
2. **显式关闭档（Stage 4 A/B 备选）**：`npuir.enable_auto_multi_buffer=False` 时手工预算 × 1.10 至 1.12（本机 `ub_cap = 196608` 实查；该系数具结构依赖，显式 alloc 结构实测可低至 0.98×）→ 11.2 B/元素（fp16）→ `D = 4096` 时 `block_m` 可达 4。

`block_m` 默认公式：`block_m = min(8, max(1, _UB_MAX_ELEMS // D))`（`_BLOCK_M_CAP = 8`，与同族前作一致）；工厂层 guard 拒绝超预算取值。

**逐 workload 复算（默认档 20 B/元素，预算 196608 B）**：

| workload（manifest label） | `x_shape` | num_groups | (M, D) 推导 | `block_m` | 驻留元素数 | 多缓冲 UB 需求 | ≤192KB |
|---------------------------|-----------|-----------|-------------|-----------|-----------|---------------|--------|
| smoke-g8 | (2, 32, 8, 8) | 8 | M = 2×8 = 16；D = (32/8)×64 = 256 | 8 | 2048 | 40 960 B（40 KB） | ✓ |
| image-g32 | (8, 128, 32, 32) | 32 | M = 8×32 = 256；D = (128/32)×1024 = 4096 | 2 | 8192 | 163 840 B（160 KB） | ✓ |
| wider-channel-g32 | (4, 256, 28, 28) | 32 | M = 4×32 = 128；D = (256/32)×784 = 6272 | 1 | 6272 | 125 440 B（122.5 KB） | ✓ |
| tail-spatial-g16 | (4, 128, 30, 30) | 16 | M = 4×16 = 64；D = (128/16)×900 = 7200 | 1 | 7200 | 144 000 B（140.6 KB） | ✓ |
| 测试用例最紧档 | (2, 32, 7, 7) | 8 | M = 16；D = 4×49 = 196 | 8 | 1568 | 31 360 B（30.6 KB） | ✓ |
| 域上界（超即拦截） | — | — | `D > 9216`（整行无法驻留） | — | — | — | ✗ 工厂 guard 报错（§9.3） |

**fp32 路径**：同表（20 B/元素平台值对两 dtype 一致），`D = 4096` 时 `block_m = 2`（`x_f32` + `sq_f32` = 8 B/元素 resident）。
**退化 / 极小档**：`D ≤ 1152` 时 `block_m` 被 `_BLOCK_M_CAP = 8` 截顶（而非 UB 容量），故极小 shape 的驻留远低于预算，无容量风险（`M < block_m` 时 `num_logical = 1`、`num_kernels = 1`，尾块 `real_m = M`）。

### 4.6 动态轴定义

无运行时动态轴：`M`、`D`、`eps`、`dtype` 均为工厂期常量（per-build JIT 特化，与源 `lru_cache` 工厂语义一致）；kernel 内唯一的运行时标量是尾块行数 `real_m = T.min(block_m, M − off_m)`。op 层把 `(N, C, *spatial)` reshape 为 `(M, D)` 是 host 元数据视图操作（连续张量，不复制数据，允许）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[3], target="npuir")   # y 在 index 3（源契约）
def _func(block_m=block_m_default): ...
```

无特殊 `pass_configs`（Developer 默认流水线）。Stage 4 的 A/B 备选会在工厂内附加 `pass_configs={"npuir.enable_auto_multi_buffer": False}`（§1.6.3 参数级裁决计划）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector（逐行统计 + 逐元素归一化），无 MAC 密集段 → 不涉及 Cube / L0A / L0B / L0C / L1 与分形约束。

### 5.2 Block 划分

```python
block_m = min(8, max(1, _UB_MAX_ELEMS // D))     # §4.5；仅沿 M 分块
block_d = D                                       # 全 D 驻留（§1.6.3 主选，D 维不分块）
num_logical      = ceil(M / block_m)              # 逻辑任务数（行块数）
num_kernels      = min(num_logical, vector_cores) # 实际启动内核数（≤ 物理核；vector_cores = 40）
num_local_tasks  = ceil(num_logical / num_kernels)# 核内串行任务数（Python 编译期常量，静态边界）
```

**选择理由**：D 维不分块——全 D 驻留使 x 只被 GM 扫描一遍（R3 表中流量最小；分块流式需重读一遍，+50%）；M 维分块——行间独立；`block_m` 由 UB 预算与实测多缓冲平台值决定，而非源的 CUDA 结构性 `block_m = 1`（§0.6 R2）。**workload 表（含 (M, D) 推导式与分核参数）见 §5.5**。

### 5.3 约束分析

- **对齐约束**：UB 对齐（32B）由编译器管理；`D` 无 256 元素对齐要求（§0.6 R7）；行块在精确 D 布局下是**连续 span**，单次 DMA 粒度 = `block_m · D · elem_bytes`（代表 workload：`2 × 4096 × 2 B = 16 KB`；`block_m` 提到 4 时 32 KB，对应 pattern-library `constants.md` 的 CONST-copy-floor-dma-granularity 中"粒度越大拷贝地板越低"的方向）。
- **UB 容量**：§4.5 逐 workload 复算，默认档最大 160 KB ≤ 192 KB ✓。
- **尾轴整除性**：§1.6.3 轴质量评分（manifest 四档 0% 尾 lane 浪费）。
- **L0 / L1 / 分形**：不适用（纯 Vector）。

### 5.4 注意事项

- **`M % block_m ≠ 0`（行尾）**：`real_m = T.min(block_m, M − off_m)` + src/dst 双显式 slice；v-op 全 buffer 操作、垃圾行按行隔离、输出只写有效行（§0.6 R3）。
- **wrapper 的默认 config**：NPU wrapper 的 `_select_row_config()` 返回 `{"block_m": 1}`（K8/K9 保留的 CUDA 时代结构性默认），即集成后的默认路径以 `block_m = 1` 调用本 kernel——**功能有效**（本设计对任意 `block_m ≥ 1` 成立），但每核串行轮数翻倍、发射项翻倍。这是 Stage 4 的**第一杠杆**：以 §4.5 的 `block_m` 表值或实测扫描值覆盖 config 选择（同族前作的 Stage 4 调优档案即把 wrapper 的 `select_row_config` 换成 per-shape 表）。本设计不修改 wrapper 文件（脚手架既有文件），仅在此登记该交互。
- **工厂层 UB guard**：`block_m * D > 9216` 抛 `ValueError`（§3.3）。
- **`eps` 与 `D_inv` 的 host 绑定**：`eps` 为工厂参数（float）；`D_inv = 1.0 / float(D)` 在工厂层求值一次，作为标量操作数传入（标量须在 `T.Kernel` 内、`T.Scope` 外使用，见 `docs/Tilelang.language/数学操作/T.vmul.md` §2.3）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**② 物理核数依据（实查记录，2026-10-15 本设计执行）**：

```python
# 在项目环境下（source /workspace/env.sh）执行
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()            # -> 20
from tilelang.utils.npu_arch import AscendArch
AscendArch().aivector_core_num             # -> 40   (纯 Vector 核数，直接实查)
AscendArch().ub_cap                        # -> 196608  (= 192 KB)
```

设备：**Ascend910B3**（`npu-smi 26.1.1` 显示 Name = 910B3，Chip 0）；**物理 AI Core = 20**，**纯 Vector 逻辑核数 = 40**（`aivector_core_num` 直查，与 `get_aicore_num() × 2` 一致）。该实查值与 pattern-library `constants.md` 的 CONST-aicore-910B2C 条目跨型号数据点（910B3：`get_aicore_num` = 20、`aivector_core_num` = 40、`ub_cap` = 196608 B）完全一致；**与 910B2C 的 24/48 不同，故不得跨型号复用核数值**（该条目明示 29% 偏差级）。

**① 逻辑核数计算**：`num_logical = ceil(M / block_m)`（仅 M 维分块；`block_d = D` 不分块，故逻辑网格是一维的）：

| workload | `x_shape` | (M, D) | `block_m` | `num_logical` | `num_kernels = min(num_logical, 40)` | `num_local_tasks` |
|----------|-----------|--------|-----------|---------------|--------------------------------------|-------------------|
| smoke-g8 | (2, 32, 8, 8) | (16, 256) | 8 | 2 | 2 | 1 |
| image-g32 | (8, 128, 32, 32) | (256, 4096) | 2 | 128 | 40 | 4 |
| wider-channel-g32 | (4, 256, 28, 28) | (128, 6272) | 1 | 128 | 40 | 4 |
| tail-spatial-g16 | (4, 128, 30, 30) | (64, 7200) | 1 | 64 | 40 | 2 |
| 最紧测试档 | (2, 32, 7, 7) | (16, 196) | 8 | 2 | 2 | 1 |

**③ 规模判定与分核方案**：统一采用 **persistent 核内串行**形态 `num_kernels = min(num_logical, 40)`，核内 `for i in T.serial(num_local_tasks)` + grid-stride 映射 `block_id = i · num_kernels + cid` + `if block_id < num_logical` 守卫：

- **逻辑核数 ≤ 物理核数（smoke-g8 与最紧测试档，2 ≤ 40）**：结论 **无需适配**——`min()` 自然退化 `num_kernels = num_logical`、每核 1 个任务，无串行轮次、无空核（因 `num_kernels ≤ num_logical`，每核首任务 `cid < num_logical` 恒有效）。
- **中等 / 极大规模（image-g32、wider-channel-g32、tail-spatial-g16）**：结论 = **核内串行 persistent 分核**——固定启动 40 个内核（物理核整数倍 = 1×），核内串行处理多个逻辑块：image-g32 `128 = 40×3 + 8`（8 核 4 任务、32 核 3 任务，负载不平衡 ≤ 1 任务）；wider-channel-g32 同形；tail-spatial-g16 `64 = 40 + 24`（24 核 2 任务、16 核 1 任务）。核内串行循环边界 `num_local_tasks` 是 host 层 Python 编译期常量（`T.ceildiv`/整数运算在 trace 期折叠），**边界静态**。
- 该形态消除逻辑核超发（如 image-g32 的 128 个逻辑核）带来的串行调度开销（`core-split-strategy.md` §1 要素③），并取得 grid-stride 聚集窗口的 MTE2 访问效益（pattern-library `constants.md` 的 CONST-mte2-degradation：grid-stride 实测优于每核连续分块 +3.2%）。
- **平衡性说明**：`block_m` 的取法已把负载不平衡限制在 ≤1 个任务；`block_m` 既是 UB 约束量也是负载粒度量，Stage 4 可按 §1.6.3 的参数级 A/B（含 auto multi-buffer 关闭档）联合扫描。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表只列 block / task 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|---------|-----|------|
| 任务维（M 行块） | 核级并行 + 核内串行 | `T.Kernel(num_kernels, is_npu=True)` + `T.serial(num_local_tasks)` | persistent 分核（§5.5）；行块间独立；静态边界 |
| 元素维（D） | 向量化（无循环） | v-op 全 buffer / `T.reduce_sum` | §1.6.2 #1 至 #7 |
| 尾块（行尾） | 标量 `min` + 双显式 slice | `T.min` + `T.copy` slice | §1.6.2 #9 / §0.6 R3 |

不使用 `T.Parallel`（Developer 下其元素级循环会引入 CUDA 时代的标量形态；本设计的元素级并行全部由 v-prefix 全 buffer 操作承担）。

### 6.2 循环伪代码

见 §3.3（结构：`with T.Kernel(num_kernels, is_npu=True) as (cid, _):` → UB buffer 分配（在 `T.serial` 之外，跨任务复用）→ `for i in T.serial(num_local_tasks):` → `if block_id < num_logical:` → 6 段计算体）。

### 6.3 流水线优化

**不使用显式 `T.Pipelined`**。理由：单任务内是顺序依赖链（`reduce` → `center` → `reduce` → `rstd` → `normalise`），没有可重叠的 K 维迭代；任务之间的装载 / 计算重叠交由编译器 auto multi-buffer 承担（同族前作实测：同形流量与链路下全部向量 pass 被 MTE2 窗口隐藏）。Stage 4 选项已在 §1.6.3 裁决计划中登记（auto multi-buffer 关闭档 + `block_m` 提档）；若 baseline 相对 §1.6.0 的估算下界偏差 > 2x，再评估显式双缓冲。

### 6.4 尾块处理

- `M % block_m ≠ 0`：最后一个行块 `real_m < block_m` —— 装载 / 写出双显式 slice 截断，v-op 对完整 buffer 计算、垃圾行按行隔离并被输出丢弃（§0.6 R3）；
- `num_logical % num_kernels ≠ 0`：`if block_id < num_logical` 守卫跳过越界任务（不产生空核，因为 `num_kernels ≤ num_logical`）；
- D 维无尾块（精确 D，无 padding）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步指令，无手动 `T.sync_block_set` / `T.sync_block_wait` / `T.pipe_barrier`：

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy`(GM→UB) 之后 | 自动 | 编译器在 MTE2→V 拓扑插屏障 |
| v-op / `reduce_sum` 之间 | 自动（顺序依赖） | 同引擎顺序执行 |
| `vcast` 之后 | 自动 | 同上 |
| `T.copy`(UB→GM) 之前 | 自动 | 编译器在 V→MTE3 拓扑插屏障 |
| `T.serial` 任务之间 | 自动 | 同核串行，无并发 |

**无核间协作**：行间独立、无跨核归并，故无需 `sync_block_set/wait`、无需 workspace、无 flag id 预算问题（CONST-flag-id-budget 不适用）。

### 7.3 pass_configs 配置

默认无特殊 `pass_configs`（Developer 默认编译流水线）。Stage 4 的 A/B 备选会在工厂内附加 `pass_configs={"npuir.enable_auto_multi_buffer": False}`，仅影响 UB 预算口径与装载重叠（§1.6.3）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 的**源算子语义**为唯一依据（移植源仓测试基准 `GroupNormTest.ref_program`，见 `examples/TileOPs/tests/ops/test_group_norm.py`），**不复刻 §0.6 的 NPU 算法**（无 fp32 中转链、无 persistent 结构、无 `vsqrt+vdiv` 组合）——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_group_norm(x, weight, bias, num_groups, eps=1e-5):
    """PyTorch reference (port of the source repo ref_program).

    x: (N, C, *spatial); weight/bias: per-channel (C,); same dtype.
    Semantics (section 0.1): fp32 opmath over the whole F.group_norm call,
    single round-back to the input dtype.
    """
    return F.group_norm(
        x.float(),
        num_groups,
        weight=weight.float(),
        bias=bias.float(),
        eps=eps,
    ).to(x.dtype)
```

kernel 级对照（分层测试用）可直接调用同一 golden：`golden_group_norm(x_3d, ...)`，其中 `x_3d` 由 `(M, D)` 视图 reshape 回 `(N, C, *spatial)`；kernel 级输出不含 affine，故对照时使用单位 `weight` / `bias`（等价于对归一化结果直接比对）。

### 8.2 精度标准

与源仓测试基准一致（`examples/TileOPs/tests/ops/test_group_norm.py` 的 `_get_tolerances`）：

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

fp32 门禁是**最紧档**，也是本设计采纳项 O2（rstd 组合式）的直接动因：`T.vrsqrt` 近似指令的 2.87e-3 相对误差在 1e-5 门禁下必然大面积违反（§1.6.1 证据），组合式实测 6.10e-8 留有两个数量级余量。fp16/bf16 走 fp32 中转链 + 单次 RNE 舍回，与 golden 的舍入路径一致（TRAP-fp16-opmath-golden 通解）。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确 + 边界语义」，完整 L1/L2/Boundary 分层套件交由 `tilelang-op-develop` 扩展。对照 = §8.1 golden（CPU）。

| # | 用例 | (M, D) 来源 | dtype | `block_m` | 验证目标 | 容忍度 |
|---|------|------------|-------|-----------|----------|--------|
| L0-1 | manifest smoke-g8 | (16, 256) | fp32 | 8 | fp32 直通路径（无 vcast），`M < block_m` 退化 | 1e-5 |
| L0-2 | manifest smoke-g8 | (16, 256) | fp16 | 8 | fp16 → fp32 → fp16 中转链 | 1e-3 |
| L0-3 | manifest smoke-g8 | (16, 256) | bf16 | 8 | bf16 → fp32 → bf16（bf16 无 Vector 算术） | 1.6e-2 |
| L0-4 | manifest image-g32 | (256, 4096) | fp16 | 2 | 代表 workload；`num_local_tasks = 4` 的 persistent 路径 | 1e-3 |
| L0-5 | manifest image-g32 | (256, 4096) | bf16 | 2 | 同上 bf16 | 1.6e-2 |
| L0-6 | manifest image-g32 | (256, 4096) | fp32 | 2 | 同上 fp32（最紧门禁） | 1e-5 |
| L0-7 | manifest wider-channel-g32 | (128, 6272) | fp16 | 1 | `D` 非 2 的幂且 `D > 4096`；`block_m = 1` 档 | 1e-3 |
| L0-8 | manifest tail-spatial-g16 | (64, 7200) | fp16 | 1 | `D = 7200`（`D_src = 900`，非 256 对齐 → 源需 pad，本设计精确 D）；`num_local_tasks = 2` | 1e-3 |
| L0-9 | 行尾 A | (17, 1152) | fp16 | 8 | `M % block_m = 1`（`17 = 8×2 + 1`）垃圾行丢弃 | 1e-3 |
| L0-10 | 行尾 B | (257, 4096) | fp16 | 2 | `M % block_m = 1` + 代表 D | 1e-3 |
| L0-11 | 行尾 + bf16 | (65, 7200) | bf16 | 1 | `M % 1 = 0` 边界（`block_m = 1` 无尾块）与 bf16 组合 | 1.6e-2 |
| L0-12 | 微小奇数 shape | (5, 13) | fp16 | 4 | `D = 13`（尾 axis 浪费 23%）、`M < block_m`、防御性 | 1e-3 |
| L0-13 | 非 2 的幂 cpg（测试用例形态） | (16, 196) | fp16 | 8 | `D = 4×49`（源 `x_shape` (2,32,7,7)、G=8） | 1e-3 |
| L0-14 | 3D spatial（测试用例形态） | (8, 256) | fp16 | 8 | `x_shape` (2,16,4,4,4)、G=4 → `D = 4×64`；op 层 reshape 路径 | 1e-3 |
| L0-15 | 1D spatial（测试用例形态） | (16, 64) | fp16 | 8 | `x_shape` (2,32,16)、G=8 → `D = 4×16` | 1e-3 |
| L0-16 | 非常量行（var = 0） | (16, 256) | fp32 | 8 | `rstd = 1/sqrt(eps)` 的数值路径（§1.6.1 角点） | 1e-5 |
| L0-17 | op 级端到端 | (2, 32, 8, 8) G=8 | fp16 | 8 | 经 `GroupNormFwdOp` 的完整路径（含 op 层 per-channel affine 与 reshape） | 1e-3 |

**通过条件**：全部用例的 `max_diff` 落在 `atol + rtol·|ref|` 内；对照 = §8.1 golden。**建议 L1 补充**（交 Stage 3 扩展）：`block_m ∈ {1, 2, 4, 8}` 扫描、`eps` 非默认值、UB guard 契约（`block_m` 超预算抛 `ValueError`）、`D = 9217` 域上界 guard；**Boundary 建议**：常量行 / 全零行 / 次正规 / `±inf`/`NaN` 混合行的传播一致性（与 §1.6.1 的 D-1 角点集对齐）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 / 风险 | 影响 | 处理方案 |
|------------|------|---------|
| **UB auto multi-buffer 膨胀口径** | `block_m` 表若失准则编译失败（`ub overflow`） | 默认按实测平台值 20 B/元素（同结构三缓冲实测）；工厂 guard；失败时降一档（§4.5 表内取小）；Stage 4 实测反解真实单价 |
| **发射项主导（roofline 估算 24 µs，模型保守）** | 若真实 op 单价高于 0.5 µs/op，代表 workload 会显著超出估算 | §1.6.0 已标注模型保守性；Stage 4 回填「设计估算 vs 实测」偏差行并反解单价；第一杠杆 = `block_m` 提档（§1.6.3 裁决计划） |
| **wrapper 默认 `block_m = 1`** | 集成后的默认路径每核串行轮数翻倍 → 发射项翻倍（功能不受影响） | §5.4 已登记；Stage 4 以 per-shape 表覆盖 config 选择（同族前作先例） |
| **kernel 层行广播 affine 未被读取** | 若有其它调用方（如未来 InstanceNorm 迁移）传非单位 `weight` / `bias`，本 kernel 不会施加 | 本迁移范围内唯一调用路径（`GroupNormFwdOp`）恒传单位元（wrapper `_identity_affine_for`），per-channel affine 由 op 层施加（与 GPU 完全一致）；风险登记，如后续迁移 InstanceNorm 或无 affine 变体须复评本项 |
| **op 层 per-channel affine 的额外一趟流量** | `y·w + b` 是独立逐元素 pass（代表 workload +4.19 MB ≈ +3 µs）与一次额外 launch | affine 分层接口的固有代价（GPU 侧相同），不在本 kernel 范围内；登记以便 Stage 4 的 op 级 roofline 对账 |
| **fp32 行累加顺序与 golden 不同** | torch `F.group_norm` 内部归约序与本 kernel 的行归约树序不同 | 容差判定（fp32 1e-5）；D-1 显示两式相对 fp64 的误差同量级（共享的 fp32 累加误差主导，残差非采纳项引入） |
| **`±inf` / `NaN` 行为** | 全 inf 行 → mean = inf → 输出 NaN；NaN 逐行传播 | 与公式语义一致（无特殊处理）；与源算子行为一致（§0.1）；Boundary 用例覆盖 |
| **roofline 常数版本戳** | 引用的 CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-copy-floor-dma-granularity 均绑定旧工具链（2026-09 之前的 build），本机工具链为 2026-10-15 dev root build | 估算只用于候选排序（各候选流量同为 `2·M·D·eb`，排序不依赖常数精确值）与 Stage 4 对账基线；结构性结论（驻留式 vs 流式的 50% 流量差、水平归约 vs 垂直扫描的亲和差）不依赖其精确值；引用时已逐条标注版本戳 |
| **设备型号绑定** | 分核核数（20 / 40）与 `ub_cap`（196608）绑定 Ascend910B3；910B2C 为 24 / 48 | §5.5 记录实查代码与返回值；工厂每次构建重查 `get_aicore_num()`，跨型号自动适配 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| UB 溢出 | `block_m` 超表值 / 实测膨胀 > 平台值 | 编译失败（`ub overflow`） | 降 `block_m`；§4.5 表内取小一档 |
| bf16 直传 v-op | 未做 fp32 中转 | 编译错误（dtype 不支持） | fp32 中转链（§0.6 R4） |
| **`T.vrsqrt` 直用** | 按源 L2 直译 `T.rsqrt` | fp32 门禁大面积违反（2.87e-3 常数相对误差；dtype 指纹：fp32 最差、bf16 被宽容差掩盖） | 用 `vsqrt + vdiv` 组合（采纳项 O2）；定位手法见 pattern-library `traps-runtime.md` 的 TRAP-vrsqrt-plain-precision |
| UB dst 非零起点切片 | 手写列偏移 slice | VEC 对齐错误（TRAP-UB-dst-align） | 全部 dst 零起点双显式 slice |
| `T.copy` 索引式区域 | 沿用源 `T.copy(x[base, 0], buf)` | 区域语义静默误读（TRAP-T-copy-region-semantics） | 双显式 slice |
| 跨 dtype `T.copy` 当 cast | GM fp16 → UB fp32 一步到位 | 隐藏 VCast 舍入不可控（TRAP-C12-copy-dtype-cast） | 显式 `vcast(rint)` + 同 dtype copy |
| `threads=` 参数残留 | 从源直译 | 无效果（TRAP-threads-kwarg-noop），误导调优 | `_func(block_m)` |
| `T.Parallel` 元素级循环残留 | 从源/早期原型直译 | 引入标量形态（本仓 argmax 案例实测标量化占壁钟 45%） | 全部改 v-op 全 buffer（§1.6.2） |
| `ALIGNMENT` 被误当"真实对齐需求" | 在 NPU 侧重新引入 padding | host `F.pad` 全量拷贝 + 填充列无效计算 | 保持 `ALIGNMENT = 1`（§0.6 R7），并核对 `pad == 0` |
| 标量操作数放在 `T.Kernel` 外 | `D_inv` / `eps` 直接写成 Python 全局 | 编译报错（标量须在 `T.Kernel` 内、`T.Scope` 外） | 工厂层求值、kernel 体内作为标量操作数使用（§3.3 形态） |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|---------|
| `D > 9216`（整行超出 UB 驻留预算，fp16/fp32 同界） | 超本任务域（manifest 最大 D = 7200）；工厂 guard 抛 `ValueError` 拦截；如需支持走 §1.6.0 R3 候选 4（D-tile 流式两遍，访存 +50%） |
| `M < block_m`（如 smoke-g8 的 M=16、极小 shape） | `num_logical = 1`、`num_kernels = 1`、`real_m = M`，尾块路径自然覆盖；`block_m` 的 UB 预算按 `block_m`（而非 `M`）算，仍在上限内 |
| `num_groups = 1` 或 `num_groups = C`（测试用例形态） | 只改变 Op 层的 `(M, D)` 推导（`cpg` 与 `spatial_size` 的组合），kernel 不可知（M/D 均为工厂期常量） |
| 3D / 1D spatial（测试用例形态） | Op 层 reshape 为 `(M, D)`（host 元数据视图，连续张量不复制） |
| 常量行（`var = 0`） | `rstd = 1/sqrt(eps)`（数学定义）；L0-16 覆盖；D-1 角点已对照 |
| 非连续输入 | Op 层 `x.contiguous()` 后 reshape（源同款）；kernel 只见连续 `(M, D)` |
| `eps` 为非默认值 | 工厂参数，`T.vadd(var, eps, var)` 的标量操作数；L1 补充用例覆盖 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/group_norm/_group_norm_kernel/
├── DESIGN.md            # 本设计文档
├── verify_equiv.py      # D-1 等价性机器验证脚本（已执行，5 项全 EQUIV_PASS）
├── _group_norm_kernel.py  # 算子实现（Stage 3 产出）
└── RETROSPECTIVE.md     # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档（含 §0 迁移分析、§1.6 调研与优化、§5 分核三要素） |
| `verify_equiv.py` | ✅ 已完成并执行 | 5 个采纳项 EQUIV_PASS（每项 31 案例）；矩式否决证据（rejection 段） |
| `_group_norm_kernel.py` | ⬜ 待实现（Stage 3） | 工厂 + kernel（+ golden 与 L0 套件） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |

### 10.3 命名规范

- 项目目录：`group_norm`；算子目录 / 实现文件：`_group_norm_kernel`（snake_case，与 Stage 0 提取件的函数名一致，不裁剪不变换）
- 工厂函数：`_group_norm_kernel(M, D, eps, dtype)`（签名与源一致）
- 导出：`ALIGNMENT`、`_align_up`、`_group_norm_kernel`（wrapper 导入契约的三个名字，Stage 5 集成时 `integrate_kernel.py` 依赖该集合）
- Golden 函数：`golden_group_norm(x, weight, bias, num_groups, eps)`

### 10.4 实现顺序

1. ✅ 设计文档（本文档）
2. ✅ 等价性机器验证（`verify_equiv.py`，D-1）
3. ⬜ Golden 函数（§8.1，验证基准）
4. ⬜ 算子实现 `_group_norm_kernel.py`（§3.3 伪代码 → 代码）+ L0 门槛测试（§8.3 全 17 项）
5. ⬜ 精度比对（vs golden，通过全部 L0）→ Stage 4 性能调优（§1.6.3 参数级 A/B）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| kernel 对外接口参数不变 | ✅ | 工厂 `(M, D, eps, dtype)` 原样；`main(x, weight, bias, y)` 参数名/顺序与 `out_idx=[3]` 不变 |
| 仅迁移 affine 主算子 | ✅ | `_group_norm_no_affine_kernel` / `GroupNormNoAffineKernel` / `GroupNormNoAffineFwdOp` 的语义与代码均未引入 |
| affine 分层不折叠 | ✅ | per-channel `(C,)` affine 留在 op 层；kernel 层单位元（§0.5 / O5） |
| 后端相关参数可适配 | ✅ | `threads` 移除、`ALIGNMENT` 256 → 1（均为后端实现常量，见 §0.5 对应行） |
| host 侧不改输入张量内容 | ✅ | 本设计 host 侧只做视图与元数据；`ALIGNMENT = 1` 消除 wrapper 的 `F.pad` |
| 从源码推断规格（不问用户） | ✅ | §0.1 / §0.2 由源码 + manifest 解读得出；输出 shape 已回填依据 |
| 不修改脚手架既有文件 | ✅ | 仅产出 `DESIGN.md` / `verify_equiv.py`（+ 本 skill 要求的 `RETROSPECTIVE.md`） |
