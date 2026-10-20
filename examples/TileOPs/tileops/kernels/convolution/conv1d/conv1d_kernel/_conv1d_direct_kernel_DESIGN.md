# _conv1d_direct_kernel 算子设计文档（Stage 1 — 迁移设计，Expert / 纯 Vector）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/convolution/conv1d.py` 的 `_conv1d_direct_kernel` builder，L316-402；`DepthwiseConv1dKernel` 类，L1074-1126）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），`target="npuir"`，编程模式 **Expert（单一 `T.Scope("Vector")`）**；**计算单元 = Vector（非 Cube）**，**不使用** L1 / L0A / L0B / L0C 与 `T.gemm`。
> 本设计按 skill `tilelang-op-design` 三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出，为本函数 5 个迁移函数中的第 5 个（函数 5，`_conv1d_direct_kernel`，服务 `DepthwiseConv1dKernel` 的 depthwise/direct 卷积）。
> **与同族 4 个 Cube 迁移函数（函数 1–4）的根本差异**：本算子是 **depthwise 直接法**——weight 形 `(c_out, 1, kernel_l)`，每个输出通道只有一个输入通道，逐元素 MAC `x[bz,oc,il]·weight[oc,0,kw]`，**输入通道方向上没有收缩（no tensor contraction）**，源码注释明言「with a single channel per group there is no contraction to tile, so each thread accumulates one output position over the kernel window instead of running an implicit GEMM」——因此**不能**（也不应）落到 Cube `T.gemm`，而应落 **Vector** 单元（逐元素 + 小 KW 规模归约）。同族函数 1–4 的 DESIGN.md（implicit GEMM / Cube）仅作为「哪些部分不该照搬」的对照，不作为本算子的结构基石。
> **结构对比对象**：`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_group_kernel_DESIGN.md`（函数 4，分组 implicit GEMM）说明「当 group 内只有 1 个输入通道时收缩轴退化为 1，direct 法另服务」，本设计即该 direct 法的 NPU 形态。
> 可行性探针（design probe，D-3）：`examples/conv1d/_conv1d_direct_kernel/history_version/design_probe_direct_vector.py`（Expert + 单一 `T.Scope("Vector")`：GM→UB、GM 偏移目的地的 padded workspace 构造、`vcast`/`vmul` 行广播/`reduce dims=[0]`/`vadd`/store 链路在 npuir 编译并运行成功，`max_abs_err ≈ 7.8e-4` vs `F.conv1d` golden）。同探针在 Developer 模式下编译失败（`multiple entry functions detected ('main_mix_aiv' / 'main_mix_aic')`），是本设计选 Expert 直证。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：**depthwise**（逐通道）一维卷积——输出通道 `c_o` 只与其**同名输入通道** `c_o`（`groups = c_in = c_out`，每组恰 1 输入 / 1 输出通道）沿核窗 `K_W` 做逐点乘加，**无跨输入通道求和**。对 NCL batch，逐 batch 计算：

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{k_w=0}^{K_W-1} W[c_o, 0, k_w]\cdot x[n,\, c_o,\, l\cdot s_W + k_w\cdot d_W - p_L], \qquad l \in [0, L_{out})
$$

其中 `L_out = (L_in + p_L + p_R - d_W·(K_W-1) - 1)//s_W + 1`（源码 `out_l` 定义，L331）；`s_W`=stride、`d_W`=dilation、`p_L`/`p_R`=pad_left/pad_right、`K_W`=kernel_l；`bias` 可选（`has_bias`）；weight 形 `(c_out, 1, K_W)`。**核心结构特征**：唯一规约轴是 `K_W`（小，代表 workload 为 3/31），输入通道轴**不参与收缩**（`x[bz, oc, il]` 直接以输出通道号 `oc` 索引输入通道）。

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | `out[n, c_o, :] = w_c_o ⊛ x[n, c_o, :]`——即**逐通道 1-D 互相关**（depthwise）；可写作 `out[b, c_o, ol] = Σ_{k_w} w[c_o,k_w]·xpad[b, c_o, ol·s_W + k_w·d_W]`（`xpad` 为 `x[n,c_o,:]` 按 `p_L` 左移置零后的补零行），等价于「每行（`b,c_o`）一条 1-D 相关，权重为该行 `K_W` 个标量」 |
| 规约语义 | 唯一规约 = 核窗 `K_W`（逐输出位置沿 `k_w` 标量乘加）；**规约/累加顺序** = 源码 `T.serial(kernel_l)` 外层 + fp32 fragment `out_local` 累加（`accum_dtype="float"`）；**无跨 `L` 的规约**（每个输出位置独立）、**无跨输入通道的规约**（depthwise 本质）、无 keepdim 语义 |
| dtype 语义 | 输入/权重/输出同 dtype（`float16 \| bfloat16`，`dtype=` JIT 参数）；**中间累加 fp32**（`accum_dtype = "float"`，L330）；**bias 在 fp32 累加域相加后统一 cast 到输出 dtype**（源码 `T.cast(out_local[i,j] + T.cast(bias[oc], accum_dtype), dtype)`，L372-375） |
| 边界语义 | **越界输入经掩码补 0**（`valid = (oc<c_out)&(ol<out_l)&(il>=0)&(il<l_in)`，L359）——`il = ol·s_W + k_w·d_W - p_L` 可 <0 或 ≥l_in（补零项对累加无贡献）；输出 M/N 尾块（`c_out % block_m`、`out_l % block_n`）由 `if oc<c_out and ol<out_l` 掩码处理（L370）；无 nan/inf 特殊语义（纯线性乘加，nan/inf 按 IEEE 自然传播）；空维度不产生（`L_out>0` 为 manifest shape rule）；**幂等**（无状态、无原子、无跨块依赖） |
| host 侧语义 | `Conv1dFwdOp.forward` 为 weight/input 做 `.contiguous()`（**metadata/物理连续化，不改数值内容**），depthwise dispatch 到 `DepthwiseConv1dKernel`（`groups = c_in = c_out`）；`DepthwiseConv1dKernel.forward` **直接 `launch(self, x, weight, bias=bias)`**（`GroupConv1dKernel.forward`，L1065-1071），**不做任何 `permute`/reshape/padding**；`_build_program` 仅调 `_conv1d_direct_kernel(...)`（L1095-1108）；weight 以原生 `(c_out,1,K_W)` 直接入 kernel |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓测试基准 `torch.nn.functional.conv1d(x, weight, bias, stride, padding, dilation, groups=c_in)`，weight 形 `(c_out,1,K_W)`），**不复刻 NPU 算法**。

### 0.2 源算子输入输出

源 builder `_conv1d_direct_kernel(n, c_in, l_in, c_out, kernel_l, stride_l, pad_left, pad_right, dilation_l, has_bias, dtype)` 的 prim_func 契约（bias 变体多一个 `bias` 入参）：

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL）；depthwise 下 `c_in == c_out == groups` |
| `weight` | `(c_out, 1, kernel_l)` | 同 `x` | depthwise 权重（**原生布局，无 host 重排**；GEMM 意义上无 A 矩阵） |
| `out` | `(n, c_out, out_l)` | 同 `x` | 输出（NCL；`L_out` 由 stride/dilation/padding 决定），`out_idx=[2]` |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

> Op 层输入 `weight` 为 `[C_out, C_in/groups, K_W] = [C_out, 1, K_W]`（`groups=C_in=C_out`）；builder 收到即此 shape。NPU 侧不改 op 层 I/O（`x`/`weight`/`out`/`bias` shape 与 dtype 与源一致）；`n / c_in / l_in / c_out / kernel_l / stride_l / pad_left / pad_right / dilation_l / has_bias` 均为 JIT 编译期参数。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | 三维 grid：`T.Kernel(ceildiv(out_l, block_n), ceildiv(c_out, block_m), n, threads=threads)` | — | block 编号 `(bx, by, bz)` | 任务划分（`bx`=L 方向、`by`=C_out 方向、`bz`=batch） |
| 2 | `T.clear(out_local)` | — | `out_local [block_m, block_n]` fp32 清零 | 累加器初始化 |
| 3 | 核窗循环 `T.serial(kernel_l)`：元素级并行 `T.Parallel(block_m, block_n)` 计算 `oc = by*block_m+i`、`ol = bx*block_n+j`、`il = ol*stride_l + kw*dilation_l - pad_left`、`valid = (oc<c_out)&(ol<out_l)&(il>=0)&(il<l_in)`，`out_local[i,j] += if_then_else(valid, cast(x[bz,oc,il],fp32)*cast(weight[oc,0,kw],fp32), 0)` | `x`, `weight` | `out_local`（fp32 累加） | `Σ_{k_w} W[c_o,0,k_w]·x[c_o, l·s_W+k_w·d_W-p_L]` + 越界零填充 |
| 4 | epilogue 元素级 `T.Parallel(block_m, block_n)`：`if oc<c_out and ol<out_l: out[bz,oc,ol] = cast(out_local[i,j] + (has_bias? cast(bias[oc],fp32):0), dtype)` | `out_local`, `bias` | `out` | bias 相加 + 输出 cast + M/N 尾块掩码 |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x], GM[weight] --逐元素读--> register fragment[out_local] (fp32, 每线程一个输出位置)
   --epilogue(+bias, cast, mask)--> GM[out]
  （无 SMEM staging、无 T.gemm、无 tensor core：每个输出位置只累加 K_W 次逐点乘加；
    x[bz,oc,il] 以输出通道号 oc 索引输入通道——depthwise 无通道收缩）
```

**循环与并行结构**：grid 三维 `(ceildiv(out_l, block_n), ceildiv(c_out, block_m), n)`；block 内 `threads`；核窗 `T.serial(kernel_l)` 外层串行（复用 `out_local` 累加器）；元素级 `T.Parallel(block_m, block_n)`（步骤 3 与 epilogue）。`DepthwiseConv1dKernel.default_config`：`block_m=1, block_n=128, block_k=1, num_stages=1, threads=128, enable_rasterization=True`（L1113-1122，`autotune_configs` 仅此一档）——`block_m=1` 即「每 block 一个输出通道、每线程一个输出位置」，与注释「each thread accumulates one output position over the kernel window」一致。

**host 侧逻辑**：`forward` 无预处理/后处理（`launch` 直调，L1065-1071）；`_build_program` 仅构造 builder（L1095-1108）；`_check_config` 空（L1110-1111，注释「One output position per thread takes any tile.」）。**无 host 端 padding / permute / im2col**。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **逐元素 MAC + register 累加**（`out_local` fragment fp32，`T.Parallel` 每线程一个输出位置） | 数据驻留寄存器、免中间写回；depthwise 无通道收缩故不做 tiling GEMM | 每线程沿 `k_w` 串行累加 1 个输出位置 | CUDA 寄存器堆 + 线程模型 | 硬件强相关（NPU 对应：UB fp32 累加 + Vector FMA/归约） |
| 2 | 核窗 `T.serial(kernel_l)` 外层循环 | 权重标量复用（每 tap 一个标量）、累加器驻留 | 串行遍历 `k_w`，`out_local` 原地累加 | 无（纯算法层循环结构） | 可移植（NPU 用 `T.serial(K_W)` 或一次多行装载 + 归约） |
| 3 | 块内 M/N tiling（`block_m × block_n`，默认 `1×128`） | 把 `c_out × out_l` 输出铺到 grid；`block_n` 放大每线程工作集 | 三维 grid + block 内 `T.Parallel` | CUDA 线程/block 模型 | 硬件强相关（NPU 一维 Kernel + persistent） |
| 4 | 边界掩码 `if_then_else((oc<c_out)&(ol<out_l)&(il>=0)&(il<l_in), ·, 0)` | 非整除尾块 + 补零边界的正确性 | 元素级条件零填充 | 无（纯算法/边界处理） | 可移植（重设计为 padded workspace + tail clamp） |
| 5 | swizzle 面板（`T.use_swizzle(CONV_SWIZZLE_PANEL)`） | 消 SMEM bank conflict | 索引重排 | SMEM bank 结构 | 硬件强相关（NPU 不适用；且本算子无 SMEM） |
| 6 | `threads` 执行参数（默认 128） | CUDA 线程组织 | CUDA 线程模型 | CUDA 线程模型 | 硬件强相关（NPU 一维 Kernel，无对应） |
| 7 | `num_stages`/`enable_rasterization` 调优参数 | CUDA 调度调优 | 分块/流水参数 | CUDA 调度 | 等价替换（NPU 映射为核内串行/persistent 调度，Stage 4 调优） |

> 识别不出来 ≠ 不存在。**源码本算子无**：SMEM tiling（无 `alloc_shared`）、Tensor Core mma（无 `T.gemm`）、异步搬运/多级流水（无 `T.Pipelined`）、warp shuffle、online 算法、persistent、手工 TMA、host 重排。**与函数 1–4（implicit GEMM/Cube）的差异清单**：无 SMEM tiling、无 tensor core、无 `T.Pipelined`、无 host permute——depthwise 直接法只保留「逐元素 MAC + 核窗串行累加」这一核心优化意图。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `out = Σ_{k_w} w·x_shift (+bias)`、fp32 累加、fp16/bf16 输出 | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| depthwise 结构（`c_in=c_out`，无输入通道收缩；`x[bz,oc,il]` 以 `oc` 索引输入通道） | 算法 | 无（纯数学结构） | — | **保留** | 关键判定：**收缩轴退化为 `K_W`（无通道收缩）→ 不构成 GEMM**，Compute 单元应落 **Vector**（逐元素 + 小规模归约），**不落 Cube** |
| 核窗 `T.serial(kernel_l)` + fp32 累加顺序 | 算法 | 无（纯数学分块） | — | **保留** | NPU 沿用「沿 `K_W` 的小规模规约 + fp32 累加」；实现形态改 Vector（见 R3） |
| 逐元素 MAC + register 累加（fragment `out_local`） | 优化 | 大寄存器堆 | 有（UB + Vector FMA/reduce） | **等价替换** | UB fp32 累加缓冲 + `T.vmul`/`T.vadd`/`T.reduce`（Element-wise 落 Vector 单元） |
| 三维 Kernel `T.Kernel(bx,by,bz)` + `threads` | 优化 | CUDA thread 模型 | 无（NPU 仅一维 Kernel + persistent） | **重新设计** | → §0.6 R1（一维 `T.Kernel` + `cid` 解码 + persistent 核内串行） |
| 逐元素 `T.Parallel` 掩码 MAC（`if_then_else` 条件写） | 优化 | 无（条件读/写） | 无（Expert Vector 无逐元素条件写；跨步 gather 亦缺） | **重新设计** | → §0.6 R3（per-kw 移位装载 + 权重行广播 `vmul` + `reduce`，补零由 padded workspace 前置） |
| 边界掩码 `(il>=0)&(il<l_in)`（元素级零填充） | 优化 | 无（纯算法/边界） | 部分（无逐元素条件写；UB 偏移目的地的 `T.copy` 静默错值） | **重新设计** | → §0.6 R2（零填充 GM workspace `xpad` + GM 偏移目的地构造 + tail clamp） |
| swizzle 面板（`T.use_swizzle`） | 优化 | SMEM bank 结构 | 无（NPU 无 bank conflict；本算子无 SMEM） | **舍弃** | 无共享内存，手动 swizzle 无意义 |
| `threads` 执行参数 | 优化 | CUDA 线程模型 | 无 | **舍弃** | npuir `T.Kernel(threads=)` 无效果（`TRAP-threads-kwarg-noop`） |
| host `.contiguous()` | 优化 | 无（物理连续化，仅改 stride/指针，不改数值） | 有（允许的只读视图/连续化） | **保留** | 仅 metadata/物理连续化，不违反 ascend-constraints.md §4（host 禁改输入内容）；NPU 侧不改 I/O |
| `num_stages`/`enable_rasterization` 调优参数 | 优化 | CUDA 调度 | 有（分块/调度参数） | **等价替换** | 映射为核内串行/persistent 调度与 block 取值（§5/§6，Stage 4 调优） |

> 每项处置依据：migration-analysis.md §5.3 映射表 + 本仓 Vector 先例（`examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel_DESIGN.md`、`examples/TileOPs/tileops/kernels/elementwise/add/add.py`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`）+ docs/Tilelang.language Vector 文档 + design probe 实测（`history_version/design_probe_direct_vector.py`）。舍弃项（swizzle / threads）理由已写明。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（三维 grid + threads → 一维 Kernel + persistent 核内串行）**

- **源方案**：`T.Kernel(ceildiv(out_l, block_n), ceildiv(c_out, block_m), n, threads=threads)` 三维 grid；`block_m=1` 时 `by` 即输出通道号；`T.Parallel(block_m, block_n)` 元素级并行。
- **NPU 新算法**：一维 `T.Kernel(num_cores, is_npu=True) as (kernel_id, _)`；`num_cores = NPUUtils.get().get_aicore_num() * 2 = 40`（**纯 Vector 算子翻倍**，§5.5）；把输出按**行**（row `r = bz·c_out + oc`，`R = n·c_out` 行）× **L 方向 tile**（`BN` 列）展平为逻辑任务 `num_logical = R · ceil(L_out / BN)`；核内 `T.serial(num_local_tasks)` 串行遍历（`num_local_tasks = T.ceildiv(num_logical, num_cores)`，编译期常量，边界静态），`task_id` 映射 `lid = task_id · num_cores + kernel_id`，`r = lid // GX_L`、`tile = lid % GX_L`（`GX_L = ceil(L_out/BN)`），再解码 `bz = r // c_out`、`oc = r % c_out`。所有逐元素计算交给 Vector 单元（`vmul`/`vadd`/`reduce`），不落 Cube。
- **语义保持论证**：任务映射（哪个核算哪个 `(row, tile)`）的变化不改变任何输出值——每 tile 的计算表达式与沿 `K_W` 的 fp32 累加顺序不变；整数任务到一维 persistent 的映射是双射（`lid ↔ (r, tile)`），尾 tile guard（`lid < num_logical`）仅跳过非法任务。row 之间无数据依赖（隐藏通道 / L 位置不相交），并行铺开与源一致。

**重设计项 R2：非整除 / 越界尾块重设计（元素级掩码 → 零填充 `xpad` workspace + 尾块 clamp）**

- **源方案**：`valid = (oc<c_out)&(ol<out_l)&(il>=0)&(il<l_in)` 逐元素 `if_then_else` 掩码（补零越界读 + M/N 尾块）。
- **NPU 新算法**：**输入侧越界**——Vector 前置段构造零填充 GM workspace `xpad[n, c_in, l_pad]`（wrapper 以 `torch.zeros` 分配，`l_pad = ceil16(p_L + l_in + p_R)`）：`xpad[bn, ci, p_L : p_L+l_in] = x[bn, ci, 0:l_in]`（**GM 偏移目的地**，见下），左右 pad 区与 `[l_in, l_pad)` 尾部保持 0。则任一 tap 的移位读 `xpad[bn, oc, ol·s_W + k_w·d_W : …]` **恒在界内**、无需逐元素掩码（补零自动完成）。**输出侧尾块**——`tn = T.min(BN, out_l - tile·BN)`，`T.copy` 只搬/写有效区（写 `out[bz, oc, tile·BN : tile·BN+tn]`）。**GM 偏移目的地**是唯一可用形态：UB 目标列偏移会静默错值（`examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py`），GM 偏移目的地正确——故固定「同形 UB 中转 + GM 偏移目的地」两步。
- **语义保持论证**：`xpad` 域内逐元素恒等、pad 区与尾部恒 0——与源掩码 `(il>=0)&(il<l_in)` 逐项一致（补 0 项对累加无贡献）；尾块 `tn` 裁剪与源 `ol<out_l` 掩码一致（`out_l` 公式不变）。数学等价（补 0 项不影响结果）。

**重设计项 R3：逐元素掩码 MAC → Vector 加权归约（per-kw 移位装载 + 权重行广播 + `reduce`）**

- **源方案**：`out_local[i,j] += cast(x[bz,oc,il],fp32) * cast(weight[oc,0,kw],fp32)`（逐元素标量 MAC，核窗 `T.serial` 外层）；fp32 fragment 累加。
- **NPU 新算法**（每逻辑任务 = 一行 `(bz,oc)` × 一段 L 输出 `[ol0, ol0+tn)`）：
  1. **per-kw 移位装载**：`xt = alloc_ub([K_W, BN])`；`for kw in T.serial(K_W): T.copy(xpad[bz, oc, ol0 + kw*d_W : ol0 + kw*d_W + tn], xt[kw:kw+1, 0:tn])`——每个 tap 一次**连续**（stride=1 时）移位装载（`ol·s_W + k_w·d_W` 对固定 tap 是连续段；`d_W` 只平移起点）。
  2. **权重行广播**：`wc = alloc_ub([K_W, 1])`；`T.copy(weight[oc, 0, 0:K_W], wc[0:K_W, 0:1])`（weight 行天然连续）；`T.vcast(wc, w32)`。
  3. **逐元素加权**：`T.vcast(xt, xt32)`（fp16/bf16→fp32）；`T.vmul(xt32, w32, prod)`——`[K_W, BN] · [K_W, 1]` **行广播**（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2），把该输出通道的 `K_W` 个权重广播到 L 轴。
  4. **核窗归约**：`T.reduce(prod, acc, dims=[0], reduce_mode="sum", clear=True)`——`[K_W, BN] → [1, BN]` fp32（`docs/Tilelang.language/规约操作/T.reduce.md`），**一次向量归约替代源核窗标量循环**。
- **语义保持论证**：`vmul` 行广播逐元素等价于 `xt[kw,j]·w[kw]`；`reduce(dims=[0],sum)` 对 `kw` 的求和与源 `Σ_{k_w}` 逐项对应；fp32 累加域一致（`accum_dtype="float"`）；`reduce` 的求和结合顺序可能与源 `T.serial(K_W)` 顺序不同，但 fp32 下仅为末位舍入差（≤ ulp 量级），落在 §8.2 容差内。边界（补零）由 R2 保证。

**重设计项 R4：stride>1 的移位装载重设计（跨步列采样 → polyphase 抽取 workspace）**

- **源方案**：`il = ol·s_W + k_w·d_W - p_L` 逐元素读——`s_W>1` 时对 `ol` 步长 `s_W`。
- **NPU 新算法**：`s_W>1` 时 GM→UB 连续装载要求最内维连续，跨步列采样不可直接 `T.copy`（否则静默平坦误读，CG-2026-0004 同族）。故加一段 Vector 前置：由 `xpad` 抽取物化 `xdec[n, c_in, s_W·Mdec]`，`xdec[bn, ci, r_kw·Mdec + m] = xpad[bn, ci, m·s_W + r_kw]`（`Mdec = l_pad // s_W`，`r_kw = (k_w·d_W) mod s_W`、`q_kw = (k_w·d_W - r_kw)//s_W`），用 `T.gather` + host 预建常量索引（`docs/Tilelang.language/索引与元素操作/T.gather.md`；跨步 gather 无向量指令，`TRAP-C10-strided-gather`）。则 tap `k_w` 的装载 `xt[kw, 0:tn] = xdec[bz, oc, r_kw·Mdec + ol0 + q_kw : … + tn]`（**连续**）。`s_W=1` 时退化为 `xpad` 直接移位装载（`r_kw=0`、`q_kw=k_w·d_W`；不物化 `xdec`）。机制与函数 2/4 的 polyphase 前置同构（`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py` L219-236）。
- **语义保持论证**：`(q_kw, r_kw)` 分解由 `k_w·d_W` 唯一确定（精确恒等，无近似）；`xdec[bn,ci,r·Mdec+m] = xpad[bn,ci,m·s_W+r]` 按定义逐元素恒等；tap `k_w` 读 `xdec` 的连续段与源 `x[.., ol·s_W + k_w·d_W - p_L]` 逐项一致。深度wise 代表 workload（`full-conformer-depthwise-k31-fp16`、smoke）均 `s_W=1`，`xdec` 路径服务一般 stride（保证算子完整性）。

**重设计项 R5：bias + 输出 cast epilogue（fragment 逐元素 → Vector）**

- **源方案**：epilogue `out[bz,oc,ol] = cast(out_local + cast(bias[oc],fp32), dtype)`（有 bias 时）。
- **NPU 新算法**：`acc[1, BN]`(fp32) 后，`has_bias` 时载入 `b11 = alloc_ub([1,1])`：`T.copy(bias[oc:oc+1], b11[0:1, 0:1])`，`T.vcast(b11, b11_32)`，`T.vadd(acc, b11_32, acc)`（`[1,N] + [1,1]` 列广播，`docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2）；随后 `T.vcast(acc, out_ub)`（fp32→dtype，一次舍入）并 `T.copy(out_ub[0:1, 0:tn], out[bz, oc:oc+1, ol0:ol0+tn])`。
- **语义保持论证**：bias 在 fp32 域相加后统一 cast（与源 `out_local + cast(bias,fp32)` 的加序与 dtype 一致，逐元素恒等）；cast 一次舍入，与源一致。

> §0.5 全部「重新设计」条目（R1–R5）已重设计。源优化手段意图承接：**逐元素 MAC + register 累加**（§0.4 #1）→ UB fp32 累加 + Vector FMA/reduce（R3）；**核窗 `T.serial` + 标量权重复用**（#2）→ per-kw 连续装载 + 权重行广播（R3）；**边界掩码**（#4）→ padded workspace + tail clamp（R2）；**三维 grid + threads**（#3/#6）→ 一维 persistent（R1）；swizzle（#5）为 NPU 无意义特性（舍弃）；调优参数（#7）→ 核内调度（§5/§6）。**不照搬同族函数 1–4 的 Cube/GEMM 结构**（depthwise 无通道收缩，非 GEMM）。

### 0.7 标杆实现

- 源算子路径（`source_op_path`）：`/mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py`（`_conv1d_direct_kernel` builder，L316-402；`DepthwiseConv1dKernel` 类，L1074-1126）。NPU 参考副本（源 builder 全文）：`examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py`（`_conv1d_direct_kernel`，L565-650）；NPU 分发/包装：`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d.py`。
- 源仓参考实现 / 测试基准：`torch.nn.functional.conv1d(x, weight, bias, stride, padding, dilation, groups=c_in)`（weight 形 `(c_out,1,K_W)`），见 `examples/TileOPs/tests/ops/test_conv1d.py`（`test_conv1d_depthwise_matches_torch`，L167-178）+ `examples/TileOPs/tileops/workloads/convolution.py`（`full-conformer-depthwise-k31-fp16`，L225-239；`Conv1dWorkload.ref_program` L293-308）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`F.conv1d`，`groups=c_in`），**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

`_conv1d_direct_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一，函数 5，服务 `DepthwiseConv1dKernel` 的 depthwise/direct 卷积）；NPU 接口函数同名，`target="npuir"`。

### 1.2 功能描述

**depthwise（逐通道）**一维卷积的 direct 实现：输出通道 `c_o` 只与其同名输入通道沿核窗 `K_W` 做逐点乘加（无输入通道收缩），`out[n,c_o,l] = Σ_{k_w} W[c_o,0,k_w]·x[n,c_o,l·s_W+k_w·d_W-p_L] (+ bias[c_o])`。用于 `groups = C_in = C_out` 的逐通道卷积（如 Conformer 深度可分离卷积 `full-conformer-depthwise-k31-fp16`：`C=64, K_W=31`）。**计算单元 = Vector**（逐元素 + 小规模 `K_W` 归约），不使用 Cube `T.gemm`（无 GEMM 结构）。

### 1.3 数学公式

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{k_w=0}^{K_W-1} W[c_o, 0, k_w]\, x\!\left[n,\, c_o,\, l\cdot s_W + k_w\cdot d_W - p_L\right], \qquad
L_{out} = \Big\lfloor\frac{L_{in}+p_L+p_R-d_W(K_W-1)-1}{s_W}\Big\rfloor + 1
$$

（`groups = C_{in} = C_{out}`，weight 形 `(C_{out}, 1, K_W)`；fp16/bf16 输入经 fp32 累加后按输出 dtype 舍入一次。）

### 1.4 算法描述

**逐行（逐通道）1-D 相关 + 核窗加权归约**：把输出按行 `r = bz·C_out + c_o`（共 `R = n·C_out` 行）× L 输出位置展平；对每行，取该输出通道的 `K_W` 个权重标量，构造 `[K_W, BN]` 的移位窗 tile（每 tap 一行，来自零填充 `xpad` 的连续移位段），权重行广播后逐元素相乘，沿 `K_W` 做一次向量归约得 `[1, BN]` fp32 累加，再 bias + cast。本算法与 §1.6.0 调研选定算法一致（**基线 = 逐行加权归约 + 核窗标量乘加；选定为 `[K_W,BN]` tile + 权重行广播 `vmul` + `reduce` 的结构重排**，数学等价、向量 op 数最少）。与源算法结构差异（均来自 §0.6 重设计）：并行组织（三维 grid+threads → 一维 persistent，R1）、非整除/越界（元素级掩码 → padded `xpad` + tail clamp，R2）、逐元素 MAC（→ per-kw 移位装载 + 权重行广播 + `reduce`，R3）、stride>1（跨步列采样 → polyphase `xdec`，R4）、bias epilogue（fragment 逐元素 → Vector `vadd`+`vcast`，R5）五处不同；计算表达式本体（`Σ_{k_w} w·x_shift`，fp32 累加）与源一致，**且不引入 GEMM/Cube 结构**。

### 1.5 数据流图

```
（stride=1；Vector 前置一次）
GM[x] --T.copy--> UB[ub2] --T.copy(GM 偏移目的地)--> GM[xpad]   (每 batch 一次；pad 区/尾部保持 0)

（主循环，逐任务 (行 r, L-tile)）
GM[xpad] --per-kw 连续移位 T.copy--> UB[xt[K_W, tn]] --vcast--> UB[xt32]
GM[weight] --T.copy(行, 连续)--> UB[wc[K_W,1]] --vcast--> UB[w32]
   --vmul(行广播 [K_W,tn]·[K_W,1])--> UB[prod] --reduce(dims=[0],sum)--> UB[acc[1, tn]] (fp32)
   --[has_bias] vadd(bias 列广播 [1,1])--> UB[acc] --vcast--> UB[out_ub] --T.copy--> GM[out]

（stride>1：多一段 Vector 前置物化 xdec，per-kw 装载改从 xdec 连续段）
GM[xpad] --T.gather(host 索引)--> GM[xdec] ; GM[xdec] --per-kw 连续段--> UB[xt] --> ...
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：调研四问 → 数学等价优化 → 向量化替代 → 向量化轴/布局。本节结论是 §3.1 公式拆解与 §6 循环结构的输入；迁移任务分析对象为 §0.6 重设计后的 NPU 算法（**逐行加权归约 / Vector**，非 Cube）。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**depthwise 1-D 卷积**的算法族。源算法（逐元素 MAC + 核窗串行累加）只是**基线候选之一**；候选须覆盖「把无通道收缩的逐通道相关视作向量可表达结构」的全部等价形态。本算子为**窗口类 + 规约类（小规约轴 `K_W`）**，按 algorithm-research.md §2 判为**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。信息源：algorithm-candidates.md（`ALG-conv` 命中行）、pattern-library constants.md、本仓 Vector 先例（add / mish / hardshrink 同类 elementwise）、同族函数 2/4 设计（对照，不照搬）、源算子 §0.3/§0.4、design probe `history_version/design_probe_direct_vector.py`。

**R1 等价化简公式候选**（基线必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线**（源算法 / 输入公式） | depthwise 逐元素 MAC：`out[r,ol] = Σ_{k_w} w[r,k_w]·xpad[r, ol·s_W + k_w·d_W]`（核窗 `T.serial` + fp32 累加） | — | — | ✅（基线） |
| 1 | **加权 tile + 行广播 + 归约**（选定） | `xt[K_W,BN]`（每 tap 一行移位段）·`w[K_W,1]`（行广播）→ `reduce(dims=[0],sum)` = `Σ_{k_w}` | **数学恒等**（逐元素乘 + 沿 `k_w` 求和重排；fp32 结合序差异 ≤ ulp） | 把 `K_W` 次标量 MAC 折叠为 1 次 `vmul` + 1 次 `reduce`（向量 op 数 O(K_W)→O(1)，除装载外） | ✅（选定） |
| 2 | **per-kw FMA 累加** | `for kw: acc += w[kw]·xt[kw]`（`vmul`+`vadd` 逐 tap） | **数学恒等** | 免 `reduce`，但向量 op 数 ∝ `K_W`（`K_W` 次 `vmul`+`vadd`）；`K_W=31` 时 op 数远多于候选 1 | ✅（R3 对比，弃） |
| 3 | **im2col + 显式 GEMM（Cube）** | 物化 `im2col`，把 depthwise 视作 `W_bd[C_out, C_in·K_W]`（块对角）dense GEMM | 数学等价（跨通道权重为 0） | **净亏**：FLOPs = `groups × 基线`（全 0 块白算）、权重物化 `O(C_out·C_in·K_W)` `groups` 倍；且无通道收缩时 GEMM 的 K 轴仅 `K_W`（极小）→ Cube 利用率极低 | ❌（FLOPs 与权重物化 groups 倍劣 + Cube 亲和性差） |
| 4 | **Winograd**（`ALG-conv` 命中行） | 小核 `F(m,r)` 变换域相关 | **容差内等价**（变换域舍入） | 仅对多抽头小幅降乘法、增加法；`K_W=3` 时理论降幅有限，`K_W=31` 时变换矩阵膨胀；NPU 无实测支撑、变换/逆变换向量 op 发射开销大 | ❌（NPU 无实测收益支撑 + 变换开销；见 R4） |
| 5 | **FFT 卷积** | 频域乘 + 逆变换 | 容差内等价 | 仅大核长序列有效；`K_W≤31`、`L_out` 小，FFT 开销 > 收益 | ❌（小算子不适用） |
| 6 | **split-K** | 分段 `K_W` + 二段归约 | 容差内等价 | 仅提升并行度；`K_W` 小、`L_out` 小，跨核归并代价 ≥ 收益 | ❌（归并代价 ≥ 收益） |
| 7 | **polyphase 抽取（stride>1 专用）** | `xpad[ci, ol·s_W+c] = xdec[ci, r_kw, ol+q_kw]`（`c=k_w·d_W`；`(q,r)` 唯一分解） | **精确恒等** | 使 stride>1 的移位装载回归连续 `T.copy`（否则跨步读不可向量化）；代价 = 一次 `xdec` 物化（≈ x 体量，仅 stride>1） | ✅（stride>1 采用） |

**R2 在线算法**：**结论「无独立在线变体」**——depthwise 是**窗口类逐行相关**：每个输出位置 `(r, ol)` 仅依赖该行窗口内 `K_W` 个输入，无跨位置 running 统计量、无重缩放、无跨 tile 归约；源算法 §0.3 亦为「核窗串行 + fp32 累加」的单遍结构。结构依据：输出无 cross-tile 依赖、无 running max/sum 语义（对比 online softmax 类）；滑窗本身即流式。候选 split-K（R1 候选 6）是并行化手段而非在线算法。

**R3 复杂度对比**（逐候选四口径；带宽受限算子上访存/遍数是主导项，不得只比 FLOPs；口径标明含什么不含什么。记 `R = n·C_out` 行）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **基线**（逐元素 MAC，核窗串行） | `2·n·C_out·L_out·K_W`（MAC=2；bias 加 `+C_out·L_out`，可忽略） | 输入读 `C_in·L_in·eb` + 权重读 `C_out·K_W·eb` + 输出写 `C_out·L_out·eb`；**无中间 GM 往返**（无 bias 时） | 1（每输入读一次；但逐元素读 x 无复用） | `O(BN)`（UB tile） | 高（tile 间独立；无跨核归并） |
| 候选 1（**选定**：加权 tile + 归约） | 同基线（**不增加 FLOPs**，只换执行结构） | 同基线 + `xpad` 前置写读 `≈ 2·C_in·l_pad·eb`（O(x)）；每 tap 的移位段从 L2 缓存读（`K_W·tn` × 任务数） | 1 | `O(K_W·BN)`（`xt`/`prod` UB tile）+ `O(BN)` | 同基线；向量 op 数 O(K_W)→O(1) |
| 候选 2（per-kw FMA） | 同基线 | 同基线 + `xpad` 前置 | 1 | `O(BN)` | 同基线；向量 op 数 ∝ `K_W` |
| 候选 3（im2col+dense GEMM） | `groups × 基线` | 同基线 + `im2col`/`W_bd` 物化 `O(C_out·C_in·K_W)` | 1 | GEMM tile | FLOPs/权重 groups 倍劣 |
| 候选 4（Winograd） | 乘法 `~2·n·C_out·L_out·K_W/2`，加法显著↑ | 变换/逆变换中间缓冲 | 1 | 变换域 tile | 变换链串行 |
| 候选 7（polyphase，仅 stride>1） | 同基线（**不增加 FLOPs**，只换采样） | 同基线 + `xdec` 物化 `≈ C_in·l_pad·eb`（仅 stride>1） | 1 | `O(s_W·Mdec)` tile | 同基线 |

**结论**：候选 1（加权 tile + 行广播 + 归约）四口径不劣于基线，且**向量 op 数最少**（除装载外固定 O(1)），为主导口径（本算子为**发射/延迟受限**的小算子，向量 op 数是主导项，见设计期估算）；候选 2 结构等价但向量 op 数 ∝ `K_W`（劣于候选 1）；候选 3 的 FLOPs/权重物化 groups 倍劣且 Cube 亲和性差；候选 4/5/6 为变换域/大核/归并手段，对本 shape 无正收益。**选定候选 1**；stride>1 叠加候选 7（polyphase）以**同 FLOPs** 换回连续 DMA。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰附本仓文档与 examples 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **基线 / 候选 1 / 候选 2** | 逐元素乘 + 小规模 `K_W` 归约 → **Vector**（`vmul`/`vadd`/`reduce`），**不落 Cube**，与元素级算力正确匹配 | `xt`/`prod`(K_W·BN·4B) + `acc`(BN·4B) + `xpad` 中转；`K_W=31,BN=128`：31·128·4≈15.9KB，合计 < 64KB < 192KB（§4.5） | 尾轴 `BN` 为 32B 整数倍且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓ | `ceil(L_out/BN)`、`ceil(num_logical/40)` 均编译期常量（N/BN/`K_W` 为 JIT 参数）✓ | 单步 `vmul`+`reduce` 可 `T.Pipelined`（Stage 4） | ✅ **采纳** |
| 候选 3（im2col+dense GEMM） | Cube 需通道收缩；depthwise 无收缩 → K 轴仅 `K_W`，Cube 利用率极低；且引入全 0 块 | L1/L0C（非片上必要） | 分形约束 | — | — | ❌（计算单元匹配度差 + FLOPs groups 倍劣） |
| 候选 4（Winograd） | 变换/逆变换 → Vector，乘法 → Cube 但布局复杂 | 变换域 tile 驻留 UB | 变换矩阵尺寸约束（`K_W` 大则膨胀） | — | 变换链引入额外 pass | ❌（NPU 无实测收益支撑；`K_W=3` 的 F(2,3) 理论乘法降幅 ~2.25×，而变换/逆变换向量 op 发射开销（`CONST-vector-launch-overhead` ~0.5µs/op）叠加后净亏——「未文档化假设 + 估算依据」） |
| 候选 5（FFT） | FFT → Vector | 频域缓冲 | — | — | — | ❌（`K_W≤31` 小核，FFT 开销 > 收益） |
| 候选 6（split-K） | 同基线 + 跨核归约（Vector） | 部分和 GM 往返 | — | 需动态 split 边界 | 二段归约需跨核同步 / 两 kernel | ❌（`K_W` 不大，归并代价 ≥ 收益） |
| 候选 7（polyphase） | 同基线（Vector）；物化 → Vector | `xdec` GM（非片上，≈ x 体量） | `T.gather` 连续段；`Mdec`/`s_W` 编译期常量 ✓ | `Mdec` 编译期常量 ✓ | 物化段与主循环可流水 | ✅ **采纳（仅 stride>1）** |

**调研结论**：选定算法族 = **逐行加权归约（Vector）**——输出按行 `r=(bz,c_o)` 展平，每行取 `K_W` 个权重标量，per-kw 从零填充 `xpad` 连续移位装载成 `[K_W, BN]` tile，权重以 `[K_W,1]` **行广播**（`vmul`），沿 `K_W` 一次 `reduce(sum)` 成 `[1,BN]` fp32 累加，bias 列广播相加后 cast。关键依据：R1 候选 1 与基线数学恒等且向量 op 数最少；R2 depthwise 天然单遍（无在线变体）；R3 候选 1 四口径不劣且发射项最优（本算子发射/延迟受限）；R4 计算单元正确落 **Vector（非 Cube）**、UB 容量达标、32B 对齐、静态边界。**与基线的结构差异**：无算法级差异（R1 确认为等价结构重排）——仅按 §0.6 重设计其并行组织（R1）、尾块/边界（R2）、MAC→Vector（R3）、stride>1（R4）、bias epilogue（R5）为 NPU 形态。**「无更优替代」调研范围**：algorithm-candidates.md（`ALG-conv` 命中行 im2col+GEMM / implicit GEMM / Winograd，经 `kb_search.py` 检索）、恒等变形自查（R1 候选 1–7 全部评估，含 depthwise 特有的的 block-diagonal dense GEMM 淘汰）、pattern-library（elementwise/constants/traps-compiler）、本仓 Vector 先例（add/mish/hardshrink）、同族函数 2/4 设计与实现（Cube 对照）、源算子 §0.3/§0.4、design probe 实测。源优化手段意图承接见 §0.6 末段。本结论是 §1.4 与 §1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2）**（完整调研，流量项 + 发射项 + 容量项均算；常数引 `tilelang-op-optimize/references/pattern-library/constants.md`，版本戳 `tilelang 0.1.2+3a214cde + CANN 8.5.0`）：

```
代表 workload: full-conformer-depthwise-k31-fp16 (N=1, C_in=C_out=64, L_in=128, K_W=31, stride=1, pad=15, dilation=1, groups=64, fp16, bias=True, L_out=128)
   行数 R = n·C_out = 64；BN=128 → ceil(L_out/BN)=1 → num_logical = 64；num_cores = 40（纯 Vector 翻倍）
估算下界: ≈ max(流量项, 容量项) 但实际由发射项主导 ≈ 数十 µs 量级（发射/延迟受限画像）
  流量项: GM = x 读 64·128·2=16KB + xpad 写 64·160·2≈20KB + xpad 主循环读（L2 吸收跨 tap 复用；compulsory≈20KB）
        + weight 读 64·31·2≈4KB + out 写 64·128·2=16KB ≈ 76KB；
        按 CONST-mte2-degradation（小 shape 高退化档，~30 GB/s/core）× 40 vector core ≈ 1.2 TB/s → 76KB/1.2TB/s ≈ 0.06µs（远非下界）
  发射项(主导): 每任务向量 op ≈ K_W 次移位 T.copy + ~7 (2×vcast + vmul + reduce + vadd + vcast + store)
        = 31 + 7 = 38；总 ≈ 64 任务 × 38 = 2432 + 前置（n·2≈2）→ 每核 ≈ 61 op
        × CONST-vector-launch-overhead ~0.5µs/op ≈ 30µs（估算下界；小 shape 并行度不足以隐藏延迟）
  容量项: xt/prod 15.9KB + acc 0.5KB + xpad 中转 16KB ≪ 192KB（不构成下界，引用 CONST-capacity-910B2C UB=192KB）
  注: 本算子代表 shape 属**发射/延迟受限**（非带宽受限）；候选 1 的 O(1) 向量归约相对候选 2（O(K_W) FMA）是降低发射项的关键
  参考: Stage 4 以 msprof Task Duration 对账（hardware-cost-model.md §3）；block 取值 / 2-D 行合并（§1.6.3 候选 2）为 Stage 4 首要调优轴
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（逐行加权归约 + 可选 bias；stride>1 叠加 polyphase）。逐项评估候选优化，无采纳项则写结论与依据。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | Winograd | 直接相关 | 变换域 `F(m,r)` 相关 | **容差内等价**（变换域舍入） | **净亏**：`K_W≤31` 时变换/逆变换向量 op 发射开销（`CONST-vector-launch-overhead`）叠加后净亏，NPU 无实测支撑；不采纳 |
| 2 | FFT 卷积 | 直接相关 | 频域乘 + 逆变换 | 容差内等价 | **净亏**：小核（`K_W≤31`）且 `L_out` 小，FFT 开销 > 收益；不采纳 |
| 3 | 除法/超越函数降代换、公共子表达式消除、稳定化变形 | — | — | **不适用**：本算子为纯乘加（沿 `K_W` 标量乘 + 求和），**无除法、无超越函数、无公共子表达式**（每项 `w·x` 仅用一次）、无溢出风险需稳定化 | 无 |
| 4 | 加法/乘法重排（结合律 / 交换律） | `Σ_{k_w} w_{k_w}·x_{k_w}` | 任意重排 | 数学等价（有限项求和可结合/交换；fp32 下舍入末位差 ≤ ulp） | **无收益**：不减少计算量/访存量；`reduce` 的结合序由实现决定，不引入额外收益；不采纳为「优化项」 |

**优化结论**：**无优化空间**。本算子为逐行「权重标量 × 移位输入 + 沿 `K_W` 求和」（外加 bias），是 depthwise 相关的最小计算形式——无除法的乘倒数、无超越函数降代换、无公共子表达式（每 `w·x` 项仅用一次）、无归约稳定化需求（纯线性乘加）。Winograd / FFT / split-K 等变换域或并行化候选经 §1.6.0 R1/R3/R4 评估均无正收益，不采纳；加法/乘法重排是 `reduce` 的实现属性、非公式级收益。**优化后公式 = 基线公式**，供 §3.1 公式拆解使用。**因无采纳的公式级优化项，按 skill Phase 2 第 5 步与 gate `S1-EQUIV-EXEC` 的豁免规则（「无优化空间」显式结论），不生成 `verify_equiv.py`。**

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 本算子**全部计算落 Vector 单元**（无 Cube）。§1.6.2 盘点实现方案中全部循环与标量计算点。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 核窗 `k_w` 标量 MAC 累加 | 源 `T.serial(kernel_l)` + 逐元素 `out_local += w·x` | **一次向量归约**：per-kw 连续装载成 `xt[K_W,BN]` → `vmul`（权重行广播）→ `T.reduce(dims=[0],sum)`（`docs/Tilelang.language/规约操作/T.reduce.md`） | ✅（Vector） | — |
| 2 | 权重标量乘 | 逐元素 `x·w[oc,0,kw]` | `T.vmul(xt32, w32, prod)`，`w32[K_W,1]` **行广播**到 `[K_W,BN]`（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2） | ✅ | — |
| 3 | 上升精度 cast（fp16/bf16→fp32） | 逐元素 `T.cast(·,fp32)` | `T.vcast(xt, xt32, round_mode="rint")`、`T.vcast(wc, w32, ...)`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`） | ✅ | — |
| 4 | bias 逐元素相加 | 源 `T.Parallel` 逐元素 `+bias[oc]` | `T.vadd(acc, b11_32, acc)`（`[1,BN]+[1,1]` **列广播**；`docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2） | ✅ | — |
| 5 | 输出 cast（fp32→fp16/bf16） | 源 `T.cast(out_local,dtype)` | `T.vcast(acc, out_ub, round_mode="rint")` | ✅ | — |
| 6 | per-kw 移位装载 | 源逐元素 `x[bz,oc,il]` | per-kw 连续 `T.copy`（`xpad` 移位段；stride>1 经 `xdec`）——DMA 搬运，非标量 | ✅ | — |
| 7 | 边界掩码 `(il>=0)&(il<l_in)` | 逐元素 `if_then_else` | 由 zero-filled `xpad` **隐式完成**（读恒在界内） | ✅ | — |
| 8 | 尾块尺寸 `tn = min(BN, out_l - ol0)` | 源 `if_then_else` 掩码 | 标量 `T.min` + `T.copy` 有效区 | ❌ | **依赖动态 shape 的边界处理**，每 tile 一次，非逐元素热点（理由类别④）；NPU 无「逐元素条件写向量」形态 |
| 9 | 任务解码 `r = lid // GX_L`、`oc = r % c_out` | 源三维 grid 隐式 | 标量整除/取模 | ❌ | **block 级任务映射计算**，每 tile 少量标量，非逐元素热点，无向量等价 API（理由类别①） |
| 10 | `lid < num_logical` 尾任务 guard | — | 标量比较 | ❌ | block 级索引守卫，每 tile 一次（理由类别①） |
| 11 | polyphase 抽取索引 `xdec[ci, r·Mdec+m] = xpad[ci, m·s_W+r]` | — | `T.gather` + host 常量索引（跨步 gather 无向量指令） | ❌ | **跨步读（步长 `s_W`）：`TRAP-C10-strided-gather` 记录 npuir 无跨步 gather 向量指令**，落 Vector 标量路径；仅 stride>1 触发（代表 workload 均 stride=1），每元素 ~1ns；属**动态边界 + API 缺失**（理由类别④/⑤） |
| 12 | `l_pad`/`num_logical`/`num_cores` 等 shape 计算 | trace-time Python 标量 | 无（编译期） | ❌ | host/trace 侧 shape/分核元数据计算，不在 kernel 计算环内（理由类别②） |

**向量化结论**：核内逐元素计算点（权重乘、核窗归约、升/降精度 cast、bias 相加、边界补零）已**全部向量化**（`vmul`/`reduce`/`vcast`/`vadd`，API 均有 `docs/Tilelang.language/` 文档与 design probe 佐证）；移位装载以 DMA `T.copy` 承载。保留 3 类标量点：polyphase 抽取索引（跨步 gather API 缺失，仅 stride>1）、任务解码、尾块尺寸/guard（均 block 级任务映射 / 动态边界，非逐元素热点），理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。**逐元素 + 小规约落 Vector 是正确单元选择（明示：不落 Cube）。**

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（NCL 进 / NCL 出，weight `(c_out,1,K_W)`）。核内布局与向量化轴是设计变量——1.6.2 回答「是否向量化」，本节回答「**在哪个轴向量化**」。本算子有 ≥2 个非 batch 维（`c_o`、`L_out`），须枚举候选。

**轴质量评分维度**（逐候选填写）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | 轴长是否为向量宽度整数倍（昇腾 Vector 128bit：fp16/bf16 ×8、fp32 ×4） | `L_out=128` ✓（128 = 16×8）；尾块 `tn` 动态裁剪 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | 代表 workload `L_out=128` 整除，尾 lane 浪费 0；一般 shape 由 `tn` 裁剪，垃圾 lane 不写回 |
| 累加链形态 | 该轴为串行累加维（stride≠1）还是向量维（stride=1） | 向量化轴 = `L_out`（stride=1 连续）；累加维 = `K_W`（由 `reduce` 一次归约）；**无跨步系数阻碍**（`d_W` 只平移起点） |
| repack 代价 | 改变布局需要的重排代价 | 选定方案：per-kw `T.copy` 直读 `xpad` 连续段、权重行连续读——**无需 repack**；`xpad` 前置一次性 O(x) |
| UB 容量影响 | 布局变化对驻留缓冲尺寸的影响 | 选定方案 `O(K_W·BN + BN)`；2-D 行合并候选（#2）同量级；对照 §4.5 |

**候选矩阵**（全部枚举，弃选必须给量化理由，禁止「未考虑」）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **逐行（`r=(bz,c_o)`）展平 + L-tile；核内 `[K_W,BN]` 移位 tile + 权重 `[K_W,1]` 行广播** | `L_out`（连续轴，stride=1） | 无（`xpad` 连续段直读 + 权重行连续读） | 基线：连续装载/写回、无核内 stride 索引、无 gather；向量 op 数 O(1)（除装载） | ✅ **采纳** |
| 2 | **2-D 行合并 tile `[BM, BN]`（BM 个输出通道 × BN 列）** | `L_out`（内层连续），外层 `c_o` | 需把 weight 按 `(kw, c_o)` 重排成 workspace `wt[K_W, c_out]` 以便取连续 `[BM,1]` 权重列（否则按 `(c_o, kw)` 存储的固定 tap 列步长 `K_W` → 跨步读） | 摊薄每 tile 固定开销、放大向量 op 宽度；代价 = 一次 weight 转置前置 + `[K_W,BM,BN]` 3-D tile 驻留（UB 压力↑） | ❌（代表 shape 小、`ceil(L_out/BN)=1`，行合并收益有限而 UB 压力与前置成本确定；列 Stage 4 A/B 备选，见下） |
| 3 | 全量 1-D 展平（把 `(r,ol)` 展成一维） | 展平轴 | 每元素权重不同 → 需按 `ol` 反解 `oc` 取权重（逐元素 gather/索引） | 破坏向量化（权重非按行常量） | ❌（权重逐元素 gather，向量化被破坏） |
| 4 | 核内 im2col 重排（物化 `[K_W, L_out]` 窗矩阵） | `L_out` | 物化 im2col 到 UB/GM | 与候选 1 的 per-kw 装载数学同构，但**额外物化** `[K_W,L_out]`（≈ 候选 1 的 `xt` 本就逐 tile 构造）→ 无收益、增访存 | ❌（同构且增物化） |
| 5 | host permute / pad 进出 | 其他轴 | 两次 host permute + padding | 违反 ascend-constraints.md §4（host 禁改输入内容、禁 padding）；百 µs 级净亏 | ❌（违规 + 净亏） |

> 迁移任务：GPU 源码的并行轴（`bx`=L、`by`=C_out、`bz`=batch）是**输入而非结论**；GPU 的 `threads × npt` 逐元素映射与 NPU Vector 轴不对应。本设计独立评估后取「逐行展平 + L 轴向量化」。

**布局决策结论**：选定方案 **#1**：核内布局 = **逐行展平 + L 轴（`out_l`，stride=1 连续）向量化**，每行 `K_W` 个权重以 `[K_W,1]` **行广播**（`vmul`）；移位窗 tile `xt[K_W, BN]`（行 = tap，列 = L）；核窗归约沿 `K_W`（`reduce dims=[0]`）。`xpad`（零填充 workspace，GM 偏移目的地构造）与 `xdec`（stride>1 polyphase，`T.gather`）为前置。该结论同步落入 §3.3 伪代码与 §6 循环结构（累加归约维 = `K_W`；向量维 = L 轴；buffer 形状为 `[K_W,BN]`/`[1,BN]`）。弃选方案（#2–#5）量化理由见候选矩阵；#2（2-D 行合并）列为 **Stage 4 A/B 备选**（§9.1）：当 `ceil(L_out/BN)>1` 或 `L_out` 大时，行合并可摊薄固定开销，届时按实测（判定阈值：备选 latency < 主选 × 0.9 才翻转）裁决。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Expert（单一 `T.Scope("Vector")`）**；**计算单元 = Vector，明确不落 Cube**。

### 2.2 选型理由

本算子是 **depthwise 逐通道相关**：weight 形 `(c_out,1,K_W)`，每输出通道只有 1 个输入通道，`x[bz,oc,il]·weight[oc,0,kw]` 逐点乘加，**输入通道方向无收缩**——不构成 GEMM/MAC 密集矩阵乘，**不应**（也无法有效）落 Cube `T.gemm`（把 depthwise 视作 block-diagonal dense GEMM 会使 FLOPs 与权重物化均 `groups` 倍劣，§1.6.0 R1 候选 3 已淘汰）。其计算是「逐元素乘 + 沿小轴 `K_W` 的规约」，正确落 **Vector** 单元（`vmul`/`vadd`/`reduce`）。

**为何 Expert 而非 Developer**：纯 Vector 逐元素算子通常用 Developer（如 `examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel_DESIGN.md`、mish/hardshard），但本算子需**零填充 `xpad` 前置**，其构造依赖「同形 UB 中转 + **GM 偏移目的地** `T.copy`」——该形态在 **Expert** 下经 repro（`examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py`）与 design probe 实测正确，且 design probe 证实**同一 kernel 在 Developer 下编译失败**（`multiple entry functions detected ('main_mix_aiv' / 'main_mix_aic')`）。此外，npnir scaffold `DepthwiseConv1dKernel.ascend_mode = "Expert"`（`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d.py` L351）与显式 `T.Scope` 控制一致。故选 **Expert + 单一 `T.Scope("Vector")`**（`T.Scope("Vector")` 单 Scope 形态由 repro L40-47 与 design probe 实证可用）。**不声明 `T.Scope("Cube")`、不使用 `T.alloc_L1`/`T.alloc_L0C`/`T.gemm`。**

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB：`ub2`（`xpad` 中转）、`xt`/`xt32`/`prod`（`[K_W,BN]`）、`wc`/`w32`（`[K_W,1]`）、`acc`/`out_ub`（`[1,BN]`）、`b11`（`[1,1]`）；`xpad`（及 stride>1 的 `xdec`）为 GM workspace，作为 prim_func 形参由 wrapper 分配（`torch.zeros`） |
| 计算方式 | v-prefix 整块逐元素 API：`T.vcast` / `T.vmul`（行广播）/ `T.vadd`（列广播）/ `T.reduce(dims=[0],sum)`；无 `T.gemm`、无 `T.Parallel` 元素循环 |
| 数据搬运 | `T.copy`（GM↔UB；GM 偏移目的地构造 `xpad`；`T.gather` 构造 `xdec`（仅 stride>1）） |
| 同步 | **手动同步**（Expert）：单 Vector Scope 内以 `with T.rs("PIPE_MTE2"/"PIPE_MTE3")` 标注搬运段（`T.Scope("Vector")` 单 Scope 无跨引擎握手；参照 repro / design probe 的实际形态） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式（= 基线，无采纳的公式级优化项）；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `xpad[bn, ci, p_L+i] = x[bn, ci, i]` | 零填充 workspace 前置（Vector，每 batch 一次；补 0 区隐式完成边界掩码） |
| 2 | `r = lid // GX_L`、`tile = lid % GX_L`、`bz = r // c_out`、`oc = r % c_out`、`ol0 = tile·BN`、`tn = min(BN, out_l - ol0)` | tile 任务解码（block 级标量） |
| 3 | `xt[kw, j] = xpad[bz, oc, ol0 + kw·d_W + j]` | per-kw 移位装载（`j∈[0,tn)`；stride=1 连续；stride>1 改从 `xdec[r_kw·Mdec + ol0 + q_kw]` 取） |
| 4 | `wc[kw, 0] = weight[oc, 0, kw]` | 权重行装载（连续） |
| 5 | `xt32 = fp32(xt)`、`w32 = fp32(wc)` | 升 fp32（fp32 输入跳过） |
| 6 | `prod[kw, j] = xt32[kw, j] · w32[kw, 0]` | 权重**行广播**逐元素乘（映射 `Σ_{k_w}` 的被加项） |
| 7 | `acc[0, j] = Σ_{k_w} prod[kw, j]` | 沿核窗 `K_W` 向量归约（替代源核窗标量循环） |
| 8 | `acc = acc + bias[oc]`（有 bias） | bias 列广播相加（fp32 域） |
| 9 | `out[bz, oc, ol0+j] = dtype(acc[0,j])` | 降 cast + 写回（尾块裁剪） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | task 循环与解码 | `T.Kernel(num_cores, is_npu=True) as (kernel_id, _)` + `T.serial(num_local_tasks)` + `T.min`/除法取模 | `num_cores = NPUUtils.get().get_aicore_num()*2` | Expert |
| 1 | `xpad` 构造 | `T.copy` + `T.copy`（GM 偏移目的地） | `T.copy(x[bn, 0:c_in, 0:l_in], ub2)`；`T.copy(ub2, xpad[bn, 0:c_in, p_L:p_L+l_in])` | Expert |
| 2 | per-kw 移位装载 | `T.copy`（连续段） | `T.copy(xpad[bz, oc, off:off+tn], xt[kw:kw+1, 0:tn])`，`off = ol0 + kw·d_W` | Expert |
| 3 | 权重行装载 | `T.copy`（连续） | `T.copy(weight[oc, 0, 0:K_W], wc[0:K_W, 0:1])` | Expert |
| 4 | 升精度 | `T.vcast` | `T.vcast(xt, xt32, round_mode="rint")`；`T.vcast(wc, w32, round_mode="rint")` | Expert |
| 5 | 加权（行广播） | `T.vmul` | `T.vmul(xt32, w32, prod)`（`[K_W,tn]·[K_W,1]→[K_W,tn]`） | Expert |
| 6 | 核窗归约 | `T.reduce` | `T.reduce(prod, acc, dims=[0], reduce_mode="sum", clear=True)`（`[K_W,tn]→[1,tn]`） | Expert |
| 7 | bias 相加（列广播） | `T.vadd` | `T.copy(bias[oc:oc+1], b11[0:1,0:1])`；`T.vcast(b11, b11_32)`；`T.vadd(acc, b11_32, acc)` | Expert |
| 8 | 降 cast + 写回 | `T.vcast` + `T.copy` | `T.vcast(acc, out_ub, round_mode="rint")`；`T.copy(out_ub[0:1, 0:tn], out[bz, oc:oc+1, ol0:ol0+tn])` | Expert |
| 9 | （仅 stride>1）polyphase | `T.gather` + `T.copy` | `T.copy(xpad[...], ubx)`；`T.copy(poly_idx[...], ubi)`；`T.gather(ubx, ubd, ubi, [cw, l_pad])`；`T.copy(ubd, xdec[...])` | Expert |
| 10 | 同步 | `with T.rs("PIPE_MTE2")` / `with T.rs("PIPE_MTE3")` | 搬运段包裹 | Expert |

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型以**第二个位置参数**传入（不用 `dtype=` 关键字），避免 `name 'in_dtype' is not defined` 类 false alarm（design-template §3.3）。本伪代码的「GM 偏移目的地 + `vcast`/`vmul` 行广播/`reduce dims=[0]`/`vadd` 列广播」链路已由 design probe 在 npuir 实测通过（`history_version/design_probe_direct_vector.py`）。

```python
# Expert 模式；ascend_mode = "Expert"（或 os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")）
import functools
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

@functools.lru_cache(maxsize=32)
def _conv1d_direct_kernel(n, c_in, l_in, c_out, kernel_l, stride_l,
                          pad_left, pad_right, dilation_l, has_bias, dtype="float16"):
    accum = "float32"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    block_N = 128                                   # L 方向 tile；承接源 default_config 的 block_n
    l_pad = ((pad_left + l_in + pad_right + 15) // 16) * 16   # 零填充 workspace 行宽（16 对齐）
    R = n * c_out                                   # 行数（= n·c_in，depthwise c_in==c_out）
    GX_L = T.ceildiv(out_l, block_N)                # L 方向 tile 数
    num_logical = R * GX_L                          # 逻辑任务数
    num_cores = NPUUtils.get().get_aicore_num() * 2 # 纯 Vector 翻倍（实查 20 → 40，§5.5）
    need_poly = stride_l != 1
    Mdec = (l_pad + stride_l - 1) // stride_l if need_poly else 0
    # 相位（编译期常量）：r_kw = (kw*d_W) mod s_W；q_kw = (kw*d_W - r_kw)//s_W
    r_kw = [(kw * dilation_l) % stride_l for kw in range(kernel_l)]
    q_kw = [(kw * dilation_l - r_kw[kw]) // stride_l for kw in range(kernel_l)]

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(_num_cores, _block_N):
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, 1, kernel_l), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            xpad: T.Tensor((n, c_in, l_pad), dtype),          # wrapper torch.zeros 分配
            xdec: T.Tensor((n, c_in, stride_l * Mdec), dtype), # 仅 stride>1 使用（stride=1 传最小占位）
            poly_idx: T.Tensor((c_in, l_pad), "int32"),        # 仅 stride>1 使用
        ):
            with T.Kernel(_num_cores, is_npu=True) as (kernel_id, _):
                num_local_tasks = T.ceildiv(num_logical, _num_cores)   # 编译期常量，边界静态
                ub2 = T.alloc_ub([c_in, l_in], dtype)

                # ---- 阶段 A：Vector 前置（每核冗余构造置零 workspace）----
                with T.Scope("Vector"):
                    with T.rs("PIPE_MTE2"):
                        for bn in T.serial(n):
                            T.copy(x[bn, 0:c_in, 0:l_in], ub2)
                            # GM 偏移目的地（唯一正确形态；UB 偏移目的地静默错值）
                            T.copy(ub2, xpad[bn, 0:c_in, pad_left:pad_left + l_in])
                    if need_poly:
                        # xdec[bn, ci, rho*Mdec + m] = xpad[bn, ci, m*s_W + rho]（T.gather + host 常量索引）
                        ubx = T.alloc_ub([c_in, l_pad], dtype)
                        ubi = T.alloc_ub([c_in, l_pad], "int32")
                        ubd = T.alloc_ub([c_in, l_pad], dtype)
                        for bn in T.serial(n):
                            T.copy(xpad[bn, 0:c_in, 0:l_pad], ubx)
                            T.copy(poly_idx[0:c_in, 0:l_pad], ubi)
                            T.gather(ubx, ubd, ubi, [c_in, l_pad])
                            T.copy(ubd, xdec[bn, 0:c_in, 0:stride_l * Mdec])

                # ---- 阶段 B：主循环（逐任务；全 Vector）----
                with T.Scope("Vector"):
                    xt = T.alloc_ub([kernel_l, _block_N], dtype)
                    xt32 = T.alloc_ub([kernel_l, _block_N], accum)
                    wc = T.alloc_ub([kernel_l, 1], dtype)
                    w32 = T.alloc_ub([kernel_l, 1], accum)
                    prod = T.alloc_ub([kernel_l, _block_N], accum)
                    acc = T.alloc_ub([1, _block_N], accum)
                    b11 = T.alloc_ub([1, 1], dtype)
                    b11_32 = T.alloc_ub([1, 1], accum)
                    out_ub = T.alloc_ub([1, _block_N], dtype)
                    for task_id in T.serial(num_local_tasks):
                        lid = task_id * _num_cores + kernel_id
                        if lid < num_logical:                       # 尾任务 guard
                            r = lid // GX_L
                            tile = lid % GX_L
                            bz = r // c_out
                            oc = r % c_out
                            ol0 = tile * _block_N
                            tn = T.min(_block_N, out_l - ol0)
                            with T.rs("PIPE_MTE2"):
                                for kw in T.serial(kernel_l):
                                    if need_poly:
                                        base = r_kw[kw] * Mdec + ol0 + q_kw[kw]
                                        T.copy(xdec[bz, oc, base:base + tn], xt[kw:kw+1, 0:tn])
                                    else:
                                        off = ol0 + kw * dilation_l
                                        T.copy(xpad[bz, oc, off:off + tn], xt[kw:kw+1, 0:tn])
                                T.copy(weight[oc, 0, 0:kernel_l], wc[0:kernel_l, 0:1])
                            T.vcast(xt, xt32, round_mode="rint")
                            T.vcast(wc, w32, round_mode="rint")
                            T.vmul(xt32, w32, prod)              # 权重行广播 [K_W,tn]·[K_W,1]
                            T.reduce(prod, acc, dims=[0], reduce_mode="sum", clear=True)
                            if has_bias:
                                T.copy(bias[oc:oc + 1], b11[0:1, 0:1])
                                T.vcast(b11, b11_32, round_mode="rint")
                                T.vadd(acc, b11_32, acc)         # bias 列广播 [1,tn]+[1,1]
                            T.vcast(acc, out_ub, round_mode="rint")
                            with T.rs("PIPE_MTE3"):
                                T.copy(out_ub[0:1, 0:tn], out[bz, oc:oc + 1, ol0:ol0 + tn])

        return main

    return kernel(num_cores, block_N)
```

> 注 1：`dtype` 经 `@functools.lru_cache` 特化；`xpad`/`xdec`/`poly_idx` 为 GM workspace/常量，作为 prim_func 形参由 wrapper 分配/传入（对齐 `_conv1d_kernel` 将 `w_flat`/`x_pad`/`xdec`/`poly_idx` 作为 `T.Tensor` 形参的形态）。stride=1 时 `xdec` 声明为最小占位张量（不改签名分支）。
> 注 2：`has_bias`/`need_poly` 为**编译期** Python 布尔（builder 内分支），非运行时条件。
> 注 3：fp32 输入路径跳过步骤 4 的 `vcast`（同 dtype 直用 `T.copy`），`xt32` 可与 `xt` 合并；bf16 无原生 v-prefix 算术（`vmul`/`vadd` 的 bf16 列不支持），故 **bf16 必经 fp32 中转**（`vcast` bf16↔fp32 支持）。
> 注 4：`xpad` 每次 launch 前由 wrapper 以 `torch.zeros` 置零（`torch.zeros` 是分配+初始化，非 host 改输入内容；符合 ascend-constraints.md §4）；kernel 内前置段只写域内区，pad 区/尾部恒 0。
> 注 5：**`dim` 语义单一来源**：向量维 = `L_out`（`xt`/`prod` 的列）；归约维 = `K_W`（`reduce dims=[0]`）；权重广播 = 行（`[K_W,1]→[K_W,BN]`）；bias 广播 = 列（`[1,1]→[1,BN]`）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.Scope("Vector")`（单 Scope，无 Cube） | `examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py` L40-47；design probe（`history_version/design_probe_direct_vector.py`） | ✅ 已验证（repro + probe 实测） |
| `T.copy`（GM↔UB；**GM 偏移目的地**） | `docs/Tilelang.language/内存操作/T.copy.md`；repro（GM 偏移正确、UB 偏移静默错值）；design probe | ✅ 已验证（repro + probe） |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；add/mish；design probe | ✅ 已验证 |
| `T.vcast(src, dst, round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（f16→f32 rint；f32→f16/bf16 rint；fp16/fp32/bf16 ✓） | ✅ 已验证（文档 + probe） |
| `T.vmul(src1, src2, dst)`（行广播 `[M,N]·[M,1]`） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2（行/列/标量广播；UB 上；dst 可原地）；design probe | ✅ 已验证（probe：`[3,8]·[3,1]`） |
| `T.vadd(src0, src1, dst)`（列广播 `[1,N]+[1,1]`） | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2；design probe（bias 路径） | ✅ 已验证 |
| `T.reduce(src, dst, dims=[0], reduce_mode="sum", clear=True)` | `docs/Tilelang.language/规约操作/T.reduce.md`（rank 一致；`(M,N)→(1,N)` 合法；`clear=True` 免预初始化；fp32 ✓） | ✅ 已验证（文档 + probe：`[3,8]→[1,8]`） |
| `T.gather(src, dst, indices, [m,n])`（仅 stride>1） | `docs/Tilelang.language/索引与元素操作/T.gather.md`；`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py` L219-236 | ✅ 已验证（同族函数 2 实现） |
| `T.rs("PIPE_MTE2"/"PIPE_MTE3")` | `examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py`；ssd 先例 | ✅ 已验证 |
| `T.serial` / `T.min` / `T.ceildiv` / `T.max` | `docs/Tilelang.language/数学操作/T.ceildiv.md`；add/mish 先例 | ✅ 已验证 |
| `NPUUtils.get().get_aicore_num()` | `tilelang.utils`；本机实查（§5.5） | ✅ 已实查 |
| `T.gemm` / `T.alloc_L1` / `T.alloc_L0C` | **本设计不使用**（depthwise 无通道收缩，非 GEMM） | ⚠️ 明确不采用（模式判定见 §2） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为三维 `T.Kernel(bx, by, bz)`） | §0.6 R1：一维 `T.Kernel(num_cores, is_npu=True)` + `lid` 解码 `(r, tile)` + persistent 核内串行 |
| GPU 专用 API | Yes（`threads`、`T.use_swizzle`、`T.alloc_fragment`、`T.Parallel` 元素级掩码 MAC、`compile_flags`） | 全部改写为 NPU Vector API（§0.5/§0.6） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube、无 L0C） | — |
| 分核策略 | **Yes** | §5.5 三要素（逻辑核数计算 / 物理核数 `get_aicore_num()` 实查（纯 Vector 翻倍）/ 规模判定与分核方案） |
| host 禁改输入内容 / 禁 padding | **Yes**（源无 host padding；NPU 侧 padding 须在 kernel 内） | §0.6 R2：padding 由 kernel 内 Vector 前置段构造 `xpad`（host 只 `torch.zeros` 分配 workspace） |
| TRAP-copy-ub-offset-stride（UB 偏移目的地静默错值） | **Yes**（`xpad` 需列偏移） | 用 **GM 偏移目的地**（同形 UB 中转 + `T.copy(ub, xpad[..., p_L:p_L+l_in])`）；repro 实证 |
| TRAP-C10-strided-gather（无跨步 gather 向量指令） | **Yes**（仅 stride>1） | §0.6 R4：stride>1 用 polyphase `xdec`（`T.gather` + host 常量索引）回归连续装载；stride=1 直读 `xpad` |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

> 迁移任务的完整差异分析见 §0.5/§0.6，此处只汇总影响 API 选型的关键技术差异。

| 差异项 | 参考实现（GPU，direct/depthwise） | 本项目（Ascend） | 转换方案 |
|--------|------------------|-----------------|----------|
| 计算单元 | 逐元素 CUDA 线程 + fragment（无 tensor core） | **Vector** 单元（`vmul`/`vadd`/`reduce`），无 Cube | §0.5 等价替换 / §1.6.0 选定 |
| Kernel 维度 | 三维 `T.Kernel(ceildiv(out_l,BN), ceildiv(c_out,BM), n, threads=128)` | 一维 `T.Kernel(num_cores, is_npu=True)` + `lid` 解码 | §0.6 R1 |
| 内存分配 | `T.alloc_fragment`（寄存器累加器） | `T.alloc_ub`（显式 UB） | §0.5 等价替换 |
| 核窗循环 | `T.serial(kernel_l)` + `T.Parallel` 逐元素 MAC | per-kw 移位装载 + 权重行广播 `vmul` + 一次 `reduce` | §0.6 R3 |
| 边界掩码 | 逐元素 `if_then_else((il>=0)&(il<l_in)&…)` | 零填充 `xpad`（隐式边界）+ 尾块 `T.min` 裁剪 | §0.6 R2 |
| 权重访问 | `weight[oc,0,kw]` 逐元素标量读 | 权重行连续 `T.copy` → `[K_W,1]` + `vmul` 行广播 | §0.6 R3 |
| 累加器初始化 | `T.clear(out_local)` | `T.reduce(..., clear=True)`（一次归约即初始化） | 语义等价 |
| stride>1 | 逐元素 `x[.., ol·s_W+…]`（跨步读） | polyphase `xdec` + 连续 `T.copy` | §0.6 R4 |
| 布局技巧 | `T.use_swizzle` + `enable_rasterization` | 无（无 SMEM） | §0.5 舍弃 swizzle |
| `threads` | `threads=128` | 无（一维 Kernel） | §0.5 舍弃（`TRAP-threads-kwarg-noop`） |
| 运行时 target | CUDA（`compile_flags`） | `target="npuir"` | §4.7 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/conv1d/_conv1d_direct_kernel/history_version/design_probe_direct_vector.py` | **最高（本算子直接探针）** | Expert + 单 `T.Scope("Vector")`；GM 偏移目的地构造 `xpad`；`vcast`/`vmul` 行广播/`reduce dims=[0]`/`vadd`/store 链路；golden = `F.conv1d(groups=C)` |
| `examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel_DESIGN.md` | 高（纯 Vector 逐元素迁移） | 一维 Kernel + `block_size` 折叠、fp32 中转、分核三要素（纯 Vector 翻倍）、UB 预算、persistent 模板 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 高（Vector 迁移先例） | `T.alloc_ub`、`T.copy` 显式 slice 尾块、v-prefix 链 |
| `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py` | 中（同族函数 2 实现的 Vector 前置段） | `x_pad`（GM 偏移目的地）+ `xdec`（`T.gather` + host `poly_idx`）前置形态；**Cube 部分不照搬** |
| `examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py` | 中（关键 trap 的绕法） | UB 偏移目的地错值、GM 偏移目的地正确——`xpad` 构造形态依据 |
---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL）；depthwise 下 `c_in == c_out` |
| `weight` | `(c_out, 1, kernel_l)` | 同 `x` | depthwise 权重（原生布局，无 host 重排；每输出通道 `K_W` 个标量） |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(n, c_out, out_l)` | 同 `x` | 输出（NCL），JIT `out_idx=[2]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `ub2` | `(c_in, l_in)` | input dtype | UB | `xpad` 构造中转（GM→UB，前置段） |
| `xt` | `(kernel_l, block_N)` | input dtype | UB | per-kw 移位窗 tile（行 = tap） |
| `xt32` | `(kernel_l, block_N)` | float32 | UB | `xt` 的 fp32（fp32 输入时与 `xt` 合并） |
| `wc` | `(kernel_l, 1)` | input dtype | UB | 本行权重列（行 = tap） |
| `w32` | `(kernel_l, 1)` | float32 | UB | 权重列 fp32（`vmul` 行广播源） |
| `prod` | `(kernel_l, block_N)` | float32 | UB | 加权结果（`xt32·w32` 行广播） |
| `acc` | `(1, block_N)` | float32 | UB | 核窗归约累加（`reduce dims=[0]`）→ bias 相加 |
| `b11` / `b11_32` | `(1, 1)` | dtype / float32 | UB | bias 标量（`vadd` 列广播源；仅 `has_bias`） |
| `out_ub` | `(1, block_N)` | output dtype | UB | 降 cast 输出 tile |
| `ubx` / `ubi` / `ubd` | `(c_in, l_pad)` | dtype / int32 / dtype | UB | polyphase 抽取（`T.gather`；仅 stride>1） |
| `xpad` | `(n, c_in, l_pad)` | input dtype | GM | 零填充输入 workspace（GM 偏移目的地构造；prim_func 形参） |
| `xdec` | `(n, c_in, stride_l·Mdec)` | input dtype | GM | polyphase 抽取 workspace（仅 stride>1；prim_func 形参） |
| `poly_idx` | `(c_in, l_pad)` | int32 | GM | polyphase 常量索引（仅 stride>1；wrapper 预建） |

### 4.4 内存搬运路径

```
前置（Vector，每核一次；冗余构造）:
GM[x] --T.copy--> UB[ub2] --T.copy(GM 偏移目的地)--> GM[xpad]
（stride>1）GM[xpad] --T.copy--> UB[ubx] --T.gather(host 索引)--> UB[ubd] --T.copy--> GM[xdec]

主循环（Vector，逐任务）:
GM[xpad]（或 GM[xdec]）--per-kw 连续 T.copy--> UB[xt] --vcast--> UB[xt32] ┐
GM[weight] --T.copy(行, 连续)--> UB[wc] --vcast--> UB[w32]              ┤
   --vmul(行广播)--> UB[prod] --reduce(dims=[0],sum)--> UB[acc] (fp32)  ┘
   --[has_bias] vadd(bias 列广播)--> UB[acc] --vcast--> UB[out_ub] --T.copy--> GM[out]
```

**规则**：纯 Vector——数据仅 GM→UB 读入、UB→GM 写出（`xpad`/`xdec` 为一次性前置 workspace，非每 tile 往返）；GM→UB 通用路径要求同 dtype（Expert `T.copy` 不跨 dtype），故升/降精度由显式 `T.vcast` 承担；`xpad` 构造用 **GM 偏移目的地**（UB 偏移目的地静默错值）；stride>1 的跨步抽取用 `T.gather`（无跨步向量指令）。

### 4.5 UB 内存预算

以代表 workload `full-conformer-depthwise-k31-fp16`（`c_in=c_out=64, l_in=128, K_W=31, block_N=128`，fp16）计：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `ub2` | (64, 128) | fp16 | 16384 |
| `xt` | (31, 128) | fp16 | 7936 |
| `xt32` | (31, 128) | fp32 | 15872 |
| `prod` | (31, 128) | fp32 | 15872 |
| `wc` | (31, 1) | fp16 | 62 |
| `w32` | (31, 1) | fp32 | 124 |
| `acc` | (1, 128) | fp32 | 512 |
| `b11` / `b11_32` | (1, 1) | fp16 / fp32 | 6 |
| `out_ub` | (1, 128) | fp16 | 256 |
| **总计（保守，`ub2` 与前缀段未复用）** | | | **≈ 57024（55.7KB）** |

对照目标平台 UB 容量 **192KB**（`CONST-capacity-910B2C`，`tilelang 0.1.2+3a214cde + CANN 8.5.0`），按该条目 ×1.10–1.12 上偏系数：55.7KB × 1.12 ≈ 62.4KB < 192KB ✓，余量 ~68%。**不溢出，无需单独调小 block_N**。Stage 3 可将 `xt32`/`prod`/`out_ub` 复用同一 UB 区，进一步降至 ~40KB。**UB 约束（一般）**：`(K_W·block_N·4)·2（xt32+prod）+ K_W·block_N·2（xt）≤ 192KB/1.12` → `K_W·block_N ≤ ~26K`（如 `block_N=128` 时 `K_W ≤ ~200`）；超限时对 `K_W` 分块归约（`T.serial` 分段 + `acc` 累加）。

**GM 工作区（不计入片上容量）**：`xpad` `n·c_in·l_pad·eb`（conformer：`1·64·160·2 ≈ 20KB`）；`xdec` 仅 stride>1；`poly_idx` 仅 stride>1（`c_in·l_pad·4`）。wrapper 侧按 JIT shape `torch.zeros`/`torch.empty` 分配。

### 4.6 动态轴定义

无动态轴。`n / c_in / l_in / c_out / kernel_l / stride_l / pad_left / pad_right / dilation_l / has_bias` 均为 JIT 编译期参数（每 shape 一特化，`@functools.lru_cache`）；`l_pad / Mdec / r_kw / q_kw / num_logical / num_cores / num_local_tasks` 均由编译期常量求值。所有循环边界（`num_local_tasks`、`kernel_l`、`n`）为编译期常量，满足「核内串行边界静态」。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
```

并以 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")`（或 wrapper `ascend_mode = "Expert"`）声明 Expert 模式（对齐 design probe / `DepthwiseConv1dKernel.ascend_mode="Expert"`）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**（逐元素乘 + 小规模 `K_W` 归约），**非 Cube、非混合**。

**判定依据**：depthwise 无输入通道收缩（`c_in=c_out`，每组 1 输入/1 输出通道），计算是「逐元素乘 + 沿 `K_W` 求和」，无矩阵乘；`T.gemm` 无适用结构。判定 = 逐元素/规约算子 → UB only。

### 5.2 Block 划分

```python
block_N = 128                       # L 方向 tile（承接源 DepthwiseConv1dKernel.default_config 的 block_n=128）
GX_L = ceildiv(out_l, block_N)      # L 方向 tile 数
R = n * c_out                       # 行数（= n·c_in）
num_logical = R * GX_L              # 逻辑任务数
```

- **选择理由**：`block_N=128` 承接源 `default_config`（L1114-1122）；代表 workload `L_out=128` → `GX_L=1`（每行一个完整 tile，无 tile 内尾块）；128 为 32B 整数倍且整除向量宽（fp16/bf16 ×8、fp32 ×4）；UB 预算 55.7KB < 192KB（§4.5）。行方向**不合并**（`block_M`≡1 行/任务）：depthwise 每行权重不同、无跨行复用，行合并仅摊薄固定开销且增 UB 压力——列为 Stage 4 A/B 备选（§1.6.3 候选 #2）。Stage 4 可扫描 `block_N`（64/128/256）。

### 5.3 约束分析

- **对齐约束**：`block_N=128`，fp16/bf16 尾轴 128·2=256B、fp32 256·4=1024B，均 32B 对齐 ✓；整除向量宽 ✓。
- **UB 容量**：55.7KB < 192KB（§4.5）✓。
- **L0/L1 容量**：无 Cube 计算，不适用。
- **规约对齐**：`reduce dims=[0]` 的 `src=[K_W, tn]`、`dst=[1, tn]`；尾部 `tn < block_N` 时按有效区（`xt` 仅写 `[0:tn]`，`reduce` 之 `size` 按 `[K_W, tn]`）。

### 5.4 注意事项

- **非整除尾块**：`tn = T.min(block_N, out_l - ol0)`；`T.copy` 仅搬/写有效区 `[0:tn]`，v-prefix 对有效区计算；`lid < num_logical` guard 覆盖最后一波不完全任务。
- **越界输入零填充**：由零填充 `xpad` 隐式完成（`xpad` 域外为 0），无逐元素掩码。
- **极小 shape**（`c_out < num_cores` 或 `L_out < block_N`）：`num_logical` 小 → `num_cores = min(num_logical, 40)`，无 oversubscription；`tn` 裁到 `out_l`。
- **`n`（batch）**：折进行号 `r = bz·c_out + oc`（`bz = r // c_out`）；各 batch 行独立，无跨 batch 依赖。
- **stride>1**：polyphase `xdec` 前置（§0.6 R4），per-kw 装载改从 `xdec` 连续段取；stride=1 直读 `xpad`（省一次物化）。
- **dilation>1**：移位装载起点平移 `kw·d_W`，不改变装载结构（stride=1 路径直接生效；stride>1 经 polyphase 相位）。
- **`has_bias=False`**：跳过 bias 载入与 `vadd`（无 `b11`/`b11_32`）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**：`num_logical_kernels = (n · c_out) × ceil(out_l / block_N) = R × GX_L`（本算子输出网格 `[n, c_out, out_l]`，行 = `n·c_out`、列 tile = `ceil(out_l/block_N)`）。以代表 workload 计：
  - `full-conformer-depthwise-k31-fp16`（N=1, C_in=C_out=64, L_in=128, K_W=31, stride=1, pad=15）：`out_l=128`，`GX_L=1`，`R=64`，`num_logical = 64 × 1 = 64`。
  - smoke `test_conv1d_depthwise_matches_torch`（N=1, C=32, L_in=128, K_W=3, stride=1, pad=1）：`out_l=128`，`GX_L=1`，`R=32`，`num_logical = 32`。

- **② 物理核数依据（实查）**：`NPUUtils.get().get_aicore_num()` **实查值 = 20**（2026-10-14 本机实查；设备 Ascend 910B3，与 `CONST-aicore-910B2C` 记录的 24 不同——**以实查为准**，禁止文档假设/经验值替代）。**本算子为纯 Vector，核数翻倍**（`core-split-strategy.md` §1 要素②；`ascend-constraints.md` §1）→ `num_cores = 20 × 2 = 40`。查询代码与实际返回值记录如下：

```python
from tilelang.utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
num_cores = physical_cores * 2                      # 纯 Vector 算子核数翻倍 = 40
```

- **③ 规模判定与分核方案**：采用**统一 persistent 结构**——`num_cores = min(num_logical, 40)`（编译期常量），核内 `T.serial(num_local_tasks)` 串行处理多个逻辑任务（`num_local_tasks = T.ceildiv(num_logical, num_cores)`，编译期常量，**边界静态**），尾任务 guard `lid < num_logical`：
  - **`full-conformer-depthwise-k31`（num_logical=64 > 40）**：属**中等规模**——`num_cores = 40`（= 物理向量核数整数倍 1×），`num_local_tasks = ceil(64/40) = 2`，每核 ≤2 任务（核内串行），内核总数 = 物理核数（40），无负载不均；启动 40 个内核而非 64 个，摊薄核启动开销。
  - **smoke（num_logical=32 ≤ 40）**：**逻辑核数 ≤ 物理核数**——`num_cores = min(32, 40) = 32`，`num_local_tasks = 1`，每核恰 1 任务，**结论「无需适配」**（直接映射，无串行超发、无负载不均）。
  - **极大规模（`num_logical ≫ 40`，如 batch 大 / `L_out` 长）**：同一结构覆盖——`num_cores=40`，每核 `num_local_tasks` 增加（核内串行），内核总数 = 物理核数（1×），循环边界静态。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环与核窗归约；逐元素标量循环仅 polyphase 抽取索引（§1.6.2 #11，跨步 gather API 缺失，仅 stride>1）。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级并行 | 核级并行 | `T.Kernel(num_cores, is_npu=True) as (kernel_id, _)` | 每核处理若干任务（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行，边界静态（§5.5） |
| 行/列 slab | 核内串行 | `T.serial`（前置 batch 循环） | 一次性 `xpad`/`xdec` 构造 |
| per-kw 移位装载 | tile 级串行 | `T.serial(kernel_l)`（每 tap 一次 `T.copy`） | 移位窗装载（tile 级，非逐元素） |
| 核窗归约 | **向量** | `T.reduce(dims=[0], reduce_mode="sum")` | 一次向量归约替代源核窗标量循环（§1.6.2 #1） |
| 逐元素乘 | **向量** | `T.vmul`（行广播） | 权重逐元素乘全部向量化（§1.6.2 #2） |
| polyphase 抽取 | Vector 搬运 | `T.gather`（仅 stride>1） | 跨步抽取（API 缺失，§1.6.2 #11） |

### 6.2 循环伪代码

```python
with T.Kernel(num_cores, is_npu=True) as (kernel_id, _):
    num_local_tasks = T.ceildiv(num_logical, num_cores)   # 编译期常量，静态边界
    with T.Scope("Vector"):
        # 阶段 A：前置（每核冗余构造 xpad；stride>1 加 xdec）
        for bn in T.serial(n):
            T.copy(x[bn, 0:c_in, 0:l_in], ub2); T.copy(ub2, xpad[bn, 0:c_in, p_L:p_L+l_in])
        # (need_poly) for bn: gather xpad -> xdec
        # 阶段 B：主循环
        for task_id in T.serial(num_local_tasks):
            lid = task_id * num_cores + kernel_id
            if lid < num_logical:
                r = lid // GX_L; tile = lid % GX_L; bz = r // c_out; oc = r % c_out
                ol0 = tile * block_N; tn = T.min(block_N, out_l - ol0)
                for kw in T.serial(kernel_l):                    # per-kw 移位装载
                    off = ol0 + kw * dilation_l                   # stride=1；stride>1 用 xdec 相位
                    T.copy(xpad[bz, oc, off:off+tn], xt[kw:kw+1, 0:tn])
                T.copy(weight[oc, 0, 0:kernel_l], wc[0:kernel_l, 0:1])
                T.vcast(xt, xt32); T.vcast(wc, w32)
                T.vmul(xt32, w32, prod)                          # 行广播
                T.reduce(prod, acc, dims=[0], reduce_mode="sum", clear=True)
                if has_bias: vadd(bias 列广播)
                T.vcast(acc, out_ub); T.copy(out_ub[0:1, 0:tn], out[bz, oc:oc+1, ol0:ol0+tn])
```

### 6.3 流水线优化

Stage 3 基线用 `T.serial`（正确性优先）；Stage 4 可选：① 对 per-kw 装载与首段计算启用 `T.Pipelined`/双缓冲（隐藏 GM→UB 于归约窗口）；② 行合并 2-D tile（§1.6.3 候选 #2）摊薄固定开销；③ 增大 `block_N` 减少任务数（受 UB 约束）。本算子代表 shape 为发射/延迟受限（§1.6.0 设计期估算），**减少向量 op 数**（已由候选 1 的 O(1) 归约实现）与 **增大每任务计算量**（行合并）为主要调优方向。

### 6.4 尾块处理

非整除 shape（`out_l % block_N`、`c_out`/`n` 任意）：`tn = T.min(block_N, out_l - ol0)`；`T.copy` 仅搬有效区 `[0:tn]`、`reduce` 按有效区、输出 `T.copy(out_ub[0:1, 0:tn], out[bz, oc:oc+1, ol0:ol0+tn])` 裁剪写出；`lid < num_logical` guard 覆盖最后一波不完全任务。输入越界（`il<0 || il>=l_in`）由 `xpad` 域外恒 0 隐式处理。所有尾块搬运走显式 slice。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**手动同步**（Expert）。单 `T.Scope("Vector")` 内以 `with T.rs("PIPE_MTE2")`（GM→UB 装载：`xpad`/`xdec`/移位窗/权重）、`with T.rs("PIPE_MTE3")`（UB→GM 写回：`out`）标注搬运段，供编译器编排 MTE2/Vector 依赖。**无跨引擎握手**（无 Cube、无 ws 中转、无 4-flag）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| 前置 `xpad`/`xdec` 构造搬运 | `with T.rs("PIPE_MTE2")` / `("PIPE_MTE3")` | 标注 GM→UB 读 / UB→GM 写段 |
| 前置完成 → 主循环读 `xpad`/`xdec` | 单 Scope 内顺序依赖（Vector 引擎内自动 handle；repro/design probe 实测无需显式 flag） | 同一 Vector Scope 内 MTE2 读写在编译器依赖编排下有序 |
| 主循环 per-kw 装载 / 权重装载 | `with T.rs("PIPE_MTE2")` | 标注 GM→UB 搬运段 |
| 输出写回 | `with T.rs("PIPE_MTE3")` | 标注 UB→GM 搬运段 |

> 参照 repro（`TRAP-copy-ub-offset-stride.py`）与 design probe：单 `T.Scope("Vector")` + `T.rs` 搬运段标注即可正确运行，无需 `sync_block_set/wait` 跨引擎握手（本算子无跨引擎数据流）。跨核：各核独立处理不相交任务，无跨核同步需求。

### 7.3 pass_configs 配置

Expert 模式默认；单 Vector Scope 无跨引擎 pass 边界需求。基线不引入额外 `pass_configs`；Stage 4 若启用自动多缓冲，须核对 `auto-multi-buffer` 对 UB 预算的膨胀（`CONST-capacity-910B2C`）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据（`out[n,c_o,l] = Σ_{k_w} W[c_o,0,k_w]·x[n,c_o,l·s_W+k_w·d_W-p_L] + bias[c_o]`，fp32 累加、fp16/bf16 输出），移植源仓参考实现（`torch.nn.functional.conv1d`，`groups=c_in`），**不复刻 §0.6 的 NPU 算法**（无 `xpad`/`xdec`/`reduce` 结构）——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_conv1d_depthwise(x, weight, bias=None, stride=1, padding=0, dilation=1, dtype="float16"):
    """depthwise Conv1d 参考实现（源仓测试基准同款：examples/TileOPs/tests/ops/test_conv1d.py）。

    以 §0.1 语义为依据：weight [C_out, 1, K_W] -> F.conv1d(groups=C_in=C_out)；
    在 fp32 域计算（对齐源 fp32 累加 + fp32 bias 相加），最后 cast 回输入 dtype。
    与 §0.6 的 xpad/移位装载/reduce 实现完全独立。
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    groups = x_cpu.shape[1]                                  # depthwise: groups = C_in = C_out
    out_f32 = F.conv1d(
        x_cpu.to(torch.float32), w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
    return out_f32.to(x_cpu.dtype)
```

> 说明：`F.conv1d(groups=C_in)` 与 builder 契约的 `weight [C_out,1,K_W]` 一致；golden 在 fp32 域计算后一次 cast——与源算子累加域一致（源 `accum_dtype="float"`）。等价于逐行相关 `out[r,:] = w_r ⊛ x_r`。

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓 `conv1d_tolerance` 容差口径一致；`K_W` 累加在 fp32，`reduce` 结合序与源 `T.serial` 顺序略有差异，落在容差内。design probe 实测 `max_abs_err ≈ 7.8e-4`（fp16，`K_W=3`，远小于 1e-2）。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）代表 workload：
- `full-conformer-depthwise-k31-fp16`（N=1, C_in=C_out=64, L_in=128, K_W=31, stride=1, pad=15, **has_bias=True**，`float16`）——覆盖大 `K_W` + bias 路径；
- smoke `test_conv1d_depthwise_matches_torch`（N=1, C=32, L_in=128, K_W=3, stride=1, pad=1, **has_bias=False**）——覆盖无 bias 路径。

`randn` 输入 + 随机 `weight`/`bias`，与 `golden_conv1d_depthwise`（§8.1）在 §8.2 容差下 `torch.testing.assert_close`；分别覆盖 `has_bias=False` 与 `has_bias=True` 两条路径。

**L0 必测点**：① 非整除尾块（`out_l` 非 `block_N` 整数倍，如 `L_out=100`）；② `has_bias` 两条路径一致；③ batch>1（`n` 折进行号）；④ **padding>0**（`xpad` 补零边界正确性）；⑤ dilation>1（移位起点平移）；⑥ stride>1（polyphase `xdec` 路径，如 `stride=2`）。完整分层套件（L1 非整除 / 多 shape / batch>1、L2 边界 size、Boundary 极小/极大 shape）交由 `tilelang-op-develop` 规划，本设计不枚举。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| Expert + 单 Vector Scope 依赖工具链版本 | 「`T.Scope("Vector")` + GM 偏移目的地 + `vmul` 行广播 + `reduce dims=[0]`」的全套结论绑定当前工具链（tilelang `0.1.2+3a214cde` + CANN 8.5.0，`PL-*`/`CONST-*` 版本戳）；工具链升级后须重验（design probe 已通过为当前证据） |
| Developer 模式不适用 | 同 kernel 在 Developer 下编译失败（`multiple entry functions detected ('main_mix_aiv' / 'main_mix_aic')`，design probe 实测）——故选 Expert；Stage 3 须保持 `ascend_mode="Expert"` |
| 物理核数 20（非 24） | 本机实查 `get_aicore_num()=20`（Ascend 910B3），与 `CONST-aicore-910B2C`（24）不一致——设计以实查为准（§5.5）；`num_cores` 由 `get_aicore_num()*2` 在 JIT 期求值，跨设备自适应 |
| bf16 无原生 v-prefix 算术 | `vmul`/`vadd` 的 bf16 列不支持（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1）——bf16 必经 fp32 中转（`vcast` bf16↔fp32 支持）；fp16/fp32 可直用 |
| UB 偏移目的地静默错值 | `T.copy` 的 UB 列偏移目的地会静默错值（`TRAP-copy-ub-offset-stride`）——`xpad` 构造固定用 **GM 偏移目的地**（同形 UB 中转两步） |
| 跨步 gather 缺失 | npuir 无跨步 gather 向量指令（`TRAP-C10-strided-gather`）——stride>1 的移位装载不能直接 `T.copy`，须 polyphase `xdec`（`T.gather` + host 常量索引） |
| 发射/延迟受限 | 代表 shape（`num_logical=64`）并行度不足（40 核 × ~2 任务），向量 op 发射开销（`CONST-vector-launch-overhead` ~0.5µs/op）为主导（§1.6.0 估算 ~30µs）——Stage 4 以行合并 2-D tile / 增大 `block_N` / `T.Pipelined` 调优；判定裕度依赖未实证常数（行合并收益），列为 §1.6.3 候选 #2 的 Stage 4 A/B 备选（阈值：备选 latency < 主选 × 0.9 才翻转） |
| `reduce dims=[0]` 效率 | `reduce` 沿外层维（stride = `block_N`）——文档支持 `(M,N)→(1,N)`，但外维归约效率未实测；若 Stage 3 出现瓶颈，备选：转 `[block_N,K_W]` 布局改内维归约（需重排）或改 per-kw FMA 累加（§1.6.0 R1 候选 2） |
| `xpad` 前置开销 | 每核冗余构造 `xpad`（O(x) 写；代表 shape ~20KB）+ `xdec`（仅 stride>1）。小 shape 下为一次性固定开销；大 `n·c_in·l_in` 时须评估其相对占比（Stage 4） |
| UB 容量 | `(K_W·block_N·4)·2 + K_W·block_N·2 ≤ ~171KB` → `K_W·block_N ≤ ~26K`；超限时对 `K_W` 分块归约（§4.5） |
| 极小 shape | `num_logical` 极小 → `num_cores=min(num_logical,40)`，`num_local_tasks=1`；`tn` 裁到 `out_l` |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| Developer 模式编译失败 | 用 Developer 编译本 kernel | `multiple entry functions detected ('main_mix_aiv'/'main_mix_aic')` 编译失败 | 固定 `ascend_mode="Expert"` + `T.Scope("Vector")`（§2.2） |
| UB 偏移目的地静默错值 | `T.copy(x, ub[0:R, PAD:PAD+N])` | `xpad` 错值（无编译错误） | GM 偏移目的地：`T.copy(ub, xpad[..., p_L:p_L+l_in])`（`TRAP-copy-ub-offset-stride`） |
| bf16 直接 `vmul`/`vadd` | 对 bf16 UB 直接调用 v-prefix 算术 | 编译失败（bf16 列不支持） | bf16 经 `vcast` 转 fp32 后运算，再 `vcast` 回（§9.1） |
| Expert `T.copy` 跨 dtype | GM↔UB 拷 fp32↔fp16 | 编译报 `does not support element type casting` | 显式 `T.vcast` 承担转换（§3.2） |
| `reduce` 形状非法 | `src`/`dst` rank 不一致或差异维 >1 | 编译/运行错误 | 保持 rank 一致：`src=[K_W,tn] → dst=[1,tn]`（`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2） |
| `vmul` 秩不匹配 | `[K_W,tn]·[K_W]`（rank 不一致） | 编译失败 | 权重存 `[K_W,1]` 用行广播（`vmul` 要求同 rank） |
| stride>1 直接 `T.copy` 跨步区域 | 对 `ol·s_W` 步长做 GM→UB 连续装载 | 静默平坦误读（整块错值） | polyphase `xdec` 后连续装载（§0.6 R4） |
| idle 核越界 | `num_logical < num_cores`（用 40 而非 min） | 越界任务 | `num_cores = min(num_logical, 40)` + `lid < num_logical` guard（§5.5） |
| 尾块越界写 | 用 base+size 形态写尾块 | 写坏相邻 GM | 双显式 slice `T.copy(src[0:tn], dst[0:tn])`（§6.4） |
| `dtype=` 关键字 false alarm | `T.Tensor(..., dtype=...)` | 编译报 `name 'in_dtype' is not defined`（编译执行正常） | 位置参数传 dtype（§3.3 约定） |

### 9.3 特殊场景处理

- **非整除 shape**（`out_l` 非 `block_N` 整数倍）：尾块 `tn` 裁剪 + `lid` guard（§6.4）。
- **极小 shape**（`num_logical ≤ 40`）：`num_cores=min(num_logical,40)`，无 oversubscription（§5.5）。
- **bf16**：全链 fp32 中转（`vcast` 升/降），`vmul`/`vadd`/`reduce` 均在 fp32 域（§9.1）。
- **`has_bias=False`**：无 `b11`/`vadd` 段（§5.4）。
- **stride=1**：直读 `xpad`（省 `xdec` 物化）；**stride>1**：polyphase `xdec`（§0.6 R4）。
- **batch>1**：`n` 折进行号 `r = bz·c_out + oc`（§5.4）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/conv1d/_conv1d_direct_kernel/
├── DESIGN.md                                        # 本设计文档
├── RETROSPECTIVE.md                                 # Stage 1 复盘（Phase 8）
└── history_version/
    └── design_probe_direct_vector.py                # 设计期探针（D-3，已实测）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `history_version/design_probe_direct_vector.py` | ✅ 已完成 | Expert + 单 Vector Scope 链路探针（已实测通过） |
| `verify_equiv.py` | —（不适用） | §1.6.1 结论为「无优化空间」，按 gate `S1-EQUIV-EXEC` 豁免规则不生成 |
| `_conv1d_direct_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_conv1d_direct.py` | ⬜ 待实现（Stage 3） | 精度测试（L0/L1/L2/Boundary） |

### 10.3 命名规范

- 项目目录名：`conv1d`；算子目录名：`_conv1d_direct_kernel`；接口函数：`_conv1d_direct_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一）；JIT `target="npuir"`，`out_idx=[2]`；`ascend_mode="Expert"`（单 `T.Scope("Vector")`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ Stage 1 复盘（RETROSPECTIVE.md）+ 设计期探针
2. ⬜ Golden 函数（§8.1，`F.conv1d(groups=C_in)`）
3. ⬜ stride=1 无 bias 实现（`xpad` 前置 + per-kw 移位装载 + `vmul` 行广播 + `reduce`）→ L0 精度比对
4. ⬜ bias 路径（`vadd` 列广播 epilogue）→ L0 精度比对
5. ⬜ 非整除尾块 / batch>1 / padding>0 / dilation>1 覆盖（L1/L2/Boundary）
6. ⬜ stride>1 polyphase（`xdec` + `T.gather`）路径（`stride=2` smoke）→ L0 精度比对
7. ⬜ Stage 4 调优（行合并 2-D tile A/B、`block_N` 扫描、`T.Pipelined`、`K_W` 分块上限核对）
