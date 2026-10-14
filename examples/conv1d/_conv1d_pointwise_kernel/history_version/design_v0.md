# _conv1d_pointwise_kernel 算子设计文档（Stage 1 — 迁移设计，Expert Cube）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/convolution/conv1d.py` 的 `_conv1d_pointwise_kernel` builder / `Conv1dPointwiseKernel` 类）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），`target="npuir"`，编程模式 **Expert**（`ascend_mode="Expert"`，`T.Scope("Cube")` + `T.alloc_L1` / `T.alloc_L0C` / `T.gemm`）。
> 本设计按 skill `tilelang-op-design` 三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出，为本函数 5 个迁移函数（`_conv1d_pointwise_kernel` 为函数 1）中的首个。
> 可行性探针已通过：`examples/conv1d/probe/conv1d_geomm_probe.py`（Cube implicit-GEMM conv 在 npuir 上编译并运行成功，`target="npuir"` + Expert；slice 形态 `T.copy` 处理跨步 im2col；避开 `T.load_nd2nz` 跨步误读，见 CG-2026-0004 / `TRAP-load-nd2nz-strided`）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：1x1（pointwise）一维卷积，等价于沿通道轴的矩阵乘（implicit GEMM）——对展平的 NCL batch，逐 batch 计算

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{c_i=0}^{C_{in}-1} W[c_o, c_i] \cdot x[n, c_i, l], \qquad l \in [0, L_{out})
$$

其中 `W = weight[:, :, 0]`（`kW=1`，`weight` 原始 shape `[C_out, C_in, 1]`），`L_out = L_in`（`kW=1, stride=1, pad=0, dilation=1`），`bias` 可选（`has_bias`）。

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | `out[n, c_o, l] = (W @ x[n])[c_o, l] (+ bias[c_o])`；`W` [C_out, C_in]，`x[n]` [C_in, L_in]，`out[n]` [C_out, L_out] |
| 规约语义 | 唯一规约轴 = 输入通道 `C_in`（GEMM 的 K 轴）；**规约/累加顺序** = 源码 `T.gemm` 的分块累加（fp32 累加器 `out_local`）；**无跨 L 的规约**（每个输出位置独立），无 keepdim 语义 |
| dtype 语义 | 输入/权重/输出同 dtype（`float16 \| bfloat16`，`dtype=` JIT 参数）；**中间累加 fp32**（`accum_dtype = "float"`，`T.alloc_fragment(out_local, accum_dtype)`）；**bias 在 fp32 累加域相加后统一 cast 到输出 dtype**（源码 `output_local + T.cast(bias[oc], accum_dtype)` 再 `T.cast(..., dtype)`） |
| 边界语义 | `L_out == L_in`（`kW=1, s=1, p=0, d=1`，无 padding 越界）；非整除：`c_out % block_m != 0`（M 尾块）与 `l_in % block_n != 0`（N 尾块）由源码 `if_then_else((oc<c_out)&(l_idx<l_in), ..., 0)` 掩码处理；无 nan/inf 特殊语义（纯线性累加，nan/inf 按 IEEE 自然传播）；空维度不产生（`L_out>0` 为 manifest shape rule） |
| host 侧语义 | `forward` 做 `weight[:, :, 0].contiguous()`（`[C_out, C_in, 1]` → `[C_out, C_in]` 视图+连续化），属实现手段而非算子契约；NPU 侧可等价替换 |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓参考实现 `torch.nn.functional.conv1d`，`kW=1`），**不复刻 NPU 算法**。

### 0.2 源算子输入输出

源 builder `_conv1d_pointwise_kernel(n, c_in, l_in, c_out, has_bias, dtype)` 的 prim_func 契约（bias 变体多一个 `bias` 入参）：

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL） |
| `weight` | `(c_out, c_in)` | 同 `x` | host 侧 `weight[:, :, 0].contiguous()` 得到的 2D 权重（GEMM A 操作数） |
| `out` | `(n, c_out, l_in)` | 同 `x` | 输出（NCL；`L_out = L_in`），`out_idx=[2]` |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

> Op 层输入 `weight` 为 `[C_out, C_in_g, kW] = [C_out, C_in, 1]`；builder 收到的是 host 归约掉 `kW` 轴的 2D 版。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host：`weight_2d = weight[:, :, 0].contiguous()` | `weight [C_out,C_in,1]` | `weight_2d [C_out,C_in]` | 前置（视图+连续化，`kW` 退化） |
| 2 | 三维 grid：`T.Kernel(ceildiv(l_in, block_n), ceildiv(c_out, block_m), n)` + `threads=128` | — | block 编号 `(bx, by, bz)` | 任务划分 |
| 3 | `T.clear(out_local)` | — | `out_local [block_m, block_n]` fp32 清零 | 累加器初始化 |
| 4 | K 循环 `T.Pipelined(ceildiv(c_in, block_k), num_stages)`：`T.copy(weight[by*block_m, k_iter*block_k], weight_shared)` | `weight_2d` | `weight_shared [block_m, block_k]` | A 操作数装载 |
| 5 | 同循环：整除时 `T.copy(x[bz, k_iter*block_k, bx*block_n], data_shared)`；非整除分支 `T.Parallel` 掩码 gather `if_then_else((ci<c_in)&(l_idx<l_in), x[bz,ci,l_idx], 0)` | `x` | `data_shared [block_k, block_n]` | B 操作数装载 + 尾部零填充 |
| 6 | 同循环：`T.gemm(weight_shared, data_shared, out_local)` | A/B tile | `out_local`（累加） | `sum_{c_i} W[c_o,c_i]·x[c_i,l]` |
| 7 | epilogue `T.Parallel(block_m, block_n)`：`out_shared[i,j] = if_then_else((oc<c_out)&(ol<l_in), cast(out_local[i,j] + cast(bias[oc],fp32), dtype), 0)`（无 bias 时省略加 bias） | `out_local`, `bias` | `out_shared [block_m, block_n]` dtype | bias 相加 + 输出 cast + M/N 尾块掩码 |
| 8 | `T.copy(out_shared, out[bz, by*block_m, bx*block_n])` | `out_shared` | `out` | 输出写回 |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x], GM[weight] --T.copy--> SMEM[weight_shared/data_shared] --T.gemm--> register fragment[out_local] (fp32)
   --epilogue(Parallel: +bias, cast, mask)--> SMEM[out_shared] --T.copy--> GM[out]
```

**循环与并行结构**：grid 三维 `(ceildiv(l_in, block_n), ceildiv(c_out, block_m), n)`；block 内 `threads=128`；K 维 `T.Pipelined(ceildiv(c_in, block_k), num_stages)` 分块累加；元素级 `T.Parallel(block_m, block_n)`（epilogue）。默认配置 `block_m=64, block_n=128, block_k=128`（`Conv1dPointwiseKernel.default_config`）。

**host 侧逻辑**：`forward` 中 `weight[:, :, 0].contiguous()`（仅视图+连续化，不改输入张量真实内容，满足 ascend-constraints.md §4）；`launch(self, x, weight_2d, bias=bias)`。无 im2col（`kW=1` 时退化为纯 GEMM，无需展开窗口）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（`weight_shared`/`data_shared`） | 减少 GM 重复访问、放大访存粒度 | K 分块驻留 shared memory | CUDA shared memory | 硬件强相关 |
| 2 | Tensor Core mma（`T.gemm`） | 矩阵算力（MAC 密集） | mma fragment + tensor core 指令 | tensor core（warp 级 mma） | 硬件强相关（NPU 有 Cube 等价） |
| 3 | 异步搬运 / 多级流水（`T.Pipelined(..., num_stages)`） | 访存与计算重叠 | 分块双缓冲流水 | cp.async / 异步拷贝引擎 | 硬件强相关（NPU 有 MTE 等价） |
| 4 | register 累加（`out_local` fragment fp32） | 数据驻留寄存器，消中间写回 | fragment 累加器 | 大寄存器堆 | 硬件强相关（NPU 有 L0C 等价） |
| 5 | 尾块掩码 gather（`if_then_else((ci<c_in)&(l_idx<l_in), x, 0)`） | 非整除 shape 的正确性 | 元素级条件零填充 | 无（纯算法/边界处理） | 可移植（重设计为分形 clamp + 裁剪） |
| 6 | coalesced 连续访问（尾轴连续 `T.copy`） | 访存带宽 | 连续 slice 读写 | memory coalescing | 硬件强相关（NPU 有 MTE 连续搬运） |
| 7 | swizzle 面板（`T.use_swizzle(CONV_SWIZZLE_PANEL)`） | 消 SMEM bank conflict | 索引重排 | SMEM bank 结构 | 硬件强相关（NPU 不适用） |
| 8 | `threads × npt` 元素级并行 | 每线程处理多元素放大吞吐 | CUDA thread 组织 | CUDA 线程模型 | 硬件强相关（NPU 一维 Kernel） |

> 识别不出来 ≠ 不存在。源码无 warp shuffle、无 online 算法（逐点 GEMM 无 cross-tile 在线统计量）、无 persistent、无手工 TMA。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `out = W@x (+bias)`、fp32 累加、fp16/bf16 输出 | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| GEMM 分块累加（K 轴分块，fp32 累加器） | 算法 | 无（纯数学分块） | — | **保留** | 纯算法层；NPU 沿用 K 分块累加（§6） |
| host `weight[:, :, 0].contiguous()` | 算法 | 无（视图+连续化） | — | **保留** | 视图/元数据操作，不改输入内容（ascend-constraints.md §4） |
| Tensor Core mma（`T.gemm`） | 优化 | tensor core | 有（Cube 单元 + L0A/L0B/L0C） | **等价替换** | `T.gemm(l1_a, l1_b, l0_c, initC=..., size=[M,K,N])`（migration-analysis.md §5.3；cube-skill `api-cube.md`） |
| SMEM tiling（`alloc_shared`） | 优化 | shared memory | 有（L1 512KB / UB 192KB） | **等价替换** | `T.alloc_L1`（Expert）+ slice 形态 `T.copy`（GM→L1 自动 ND2NZ） |
| register 累加（`out_local` fragment fp32） | 优化 | 大寄存器堆 | 有（L0C 128KB fp32 累加器） | **等价替换** | `T.alloc_L0C([block_m, block_n], "float32")` |
| 异步搬运 / `T.Pipelined(num_stages)` | 优化 | cp.async | 有（MTE 异步搬运） | **等价替换** | L1 分块 `T.copy` + 多缓冲（Stage 4 启用 `T.Pipelined` / ping-pong 槽） |
| coalesced 连续访问 | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | slice 形态 `T.copy` 连续搬运（`TRAP-T-copy-region-semantics`） |
| 三维 Kernel `T.Kernel(bx,by,bz)` + `threads=128` | 优化 | CUDA thread 模型 | 无（NPU 仅一维 Kernel + persistent） | **重新设计** | → §0.6 R1（一维 `T.Kernel` + cid 解码 + persistent 核内串行） |
| 元素级尾块掩码 gather（`if_then_else` + `T.Parallel`） | 优化 | 无（条件写） | 部分（无逐元素条件写向量形态） | **重新设计** | → §0.6 R2（分形 clamp `T.max(16, ceil16(tail))` + slice 尾块 `T.copy` + 裁剪 store） |
| epilogue bias 相加（`T.Parallel` 在 fragment 上） | 优化 | 无（纯数学） | 有（Vector 单元 vadd） | **重新设计** | → §0.6 R3（L0C→GM ws → Vector fp32 bias-add → cast → GM out，MixCV 双 Scope） |
| swizzle 面板（`T.use_swizzle`） | 优化 | SMEM bank 结构 | 无（NPU 无 bank conflict） | **舍弃** | NZ 分形布局由 `T.copy` ND2NZ 自动处理，手动 swizzle 无意义（migration-analysis.md §5.3） |
| `threads` 执行参数 | 优化 | CUDA 线程模型 | 无 | **舍弃** | npuir `T.Kernel(threads=)` 无效果（`TRAP-threads-kwarg-noop`） |
| `num_stages` / `enable_rasterization` 调优参数 | 优化 | CUDA 调度 | 有（分块/流水参数） | **等价替换** | 映射为 L1 分块/多缓冲参数（§5/§6，Stage 4 调优） |

> 每项处置依据：migration-analysis.md §5.3 映射表 + `examples/` Cube 先例（gqa/ssd）+ docs/Tilelang.language Cube 文档 + 探针实测。舍弃项（swizzle / threads）理由已写明。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（三维 grid + threads → 一维 Kernel + persistent 核内串行）**

- **源方案**：`T.Kernel(ceildiv(l_in, block_n), ceildiv(c_out, block_m), n, threads=128)` 三维 grid；元素级 `T.Parallel(block_m, block_n)`（128 线程）。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (kernel_id, subid)`；`num_kernels = 物理核数`（20，实查见 §5.5），核内 `T.serial` 串行遍历 `num_local_tasks`，`cid = task_id * num_kernels + kernel_id` 解码 `(bz, by, bx)`（`bz = cid // (GY*GX)`，`by = (cid // GX) % GY`，`bx = cid % GX`）；`cid` 以 `T.min(cid, num_logical-1)` 钳位以避免 idle 核越界（对齐 gqa 先例的 `T.ceildiv` 负数行为注释）。矩阵计算交给 Cube 分形（`T.gemm`），尾块裁剪见 R2。
- **语义保持论证**：任务映射（哪个 block 算哪个输出 tile）的变化不改变任何输出值——每 tile 的计算表达式与累加顺序（K 分块 fp32 累加）不变；整数 grid 到一维 persistent 的映射是双射（`cid ↔ (bz,by,bx)`），钳位仅让 idle 核重算最后一个合法 tile（同输入、同指令序、同输出，Bitwise 一致），与 gqa 先例一致。fp32 累加与源一致。

**重设计项 R2：非整除尾块重设计（元素级掩码 gather → 分形 clamp + slice 尾块 + 裁剪 store）**

- **源方案**：`T.Parallel(block_k, block_n)` 逐元素 `if_then_else((ci<c_in)&(l_idx<l_in), x[...], 0)` 掩码零填充；epilogue 同样逐元素掩码。
- **NPU 新算法**：K/N 维非整除时，`T.copy` 用**动态 extents 的 slice 形态**仅装载有效区（`weight[by*BM : by*BM+tm, k0:k0+tk] → l1_a[0:tm,0:tk]`；`x[bz, k0:k0+tk, bx*BN:bx*BN+tn] → l1_b[0:tk,0:tn]`，extents 借用 slice 侧、越界区读 0，见 `docs/Tilelang.language/内存操作/T.copy.md` 语义规则第 4/6 条）；`T.gemm` 的 `size` 用**分形 clamp** `tmc = T.max(16, ceil16(tm))`、`tkc = T.max(16, ceil16(tk))`、`tnc = T.max(16, ceil16(tn))`（分形 ≥16 约束）保证合法几何；输出用 `T.copy(l0_c[0:tl, 0:tn], out[..., by*BM:..+tl, bx*BN:..+tn])` **裁剪**写出有效区，裁剪掉的分形填充列不落 GM。
- **语义保持论证**：有效区元素的计算与源逐元素掩码完全一致（越界读补 0 → 对输出无贡献；填充列被裁剪不写回）；数学等价（补 0 项对累加无影响）。边界语义核对：`c_out`/`l_in` 非整除时输出 shape 与源一致（`L_out=L_in`），无越界写（裁剪 store + slice 双显式 extents）。注意 `TRAP-T-copy-region-semantics`（越界写坏相邻 GM / `[N,1]` 越界读）——所有尾块搬运必须走 slice 形态而非 base+size。

**重设计项 R3：bias epilogue 重设计（fragment 逐元素相加 → MixCV Vector 后处理）**

- **源方案**：epilogue 在 `out_local` fragment 上逐元素 `out_local[i,j] + cast(bias[oc], fp32)`，再 cast 到输出 dtype、写 `out_shared`。
- **NPU 新算法**：`has_bias=True` 时，Cube 段将 L0C 结果经 fixpipe（**不量化**，fp32→fp32）写入 per-(core,slot) GM workspace `ws`；`T.sync_block_set(slot)` 通知 Vector 段；Vector 段 `T.sync_block_wait(slot)` 后 `T.copy(ws→ub_f32)`、`T.vbrc(bias_row, bias_brc)`（`[BM,1]→[BM,BN]` 广播）+ `T.vadd(bias_brc, ub_f32, ub_f32)`（**fp32 域加 bias**，对齐源 `out_local + cast(bias, fp32)`）、`T.vcast(ub_f32, ub_f16)`、`T.copy(ub_f16 → out)`；`T.sync_block_set(slot_free)` 归还槽。`has_bias=False` 时**无 Vector 段**，L0C 直接 fixpipe（fp32→fp16，`CONST-store-fixpipe-gm-only` 认可的正确量化路径）写 `out`。
- **语义保持论证**：bias 在 fp32 域相加后统一 cast（与源 `out_local + cast(bias, fp32)` 的加序与 dtype 一致）；数学恒等 `W@x + bias ≡ W@x + bias`。跨引擎经 GM ws 中转不改变数值（fp32 直存，非量化）。边界：尾块裁剪同 R2；累加顺序不变。**跨引擎传输强制 GM 往返**（`CONST-store-fixpipe-gm-only`：跨引擎传输无法直写 L1/UB，只能 GM 往返）——这是 NPU 上将 bias 放到 Vector 的固有代价；探针与 gqa/ssd 先例均用 GM ws 做 Cube↔Vector 中继。

> §0.5 全部「重新设计」条目（R1/R2/R3）已重设计。源优化手段（§0.4 #1/#2/#4/#5/#6）意图承接：SMEM tiling→L1 分块 `T.copy`；Tensor core→Cube `T.gemm`+L0C；register 累加→L0C fp32 累加器；coalesced→MTE 连续 `T.copy`；尾块掩码→R2 分形 clamp+裁剪；`T.Pipelined`→L1 多缓冲（Stage 4）。舍弃项（#7 swizzle / threads）为 NPU 无意义特性。

### 0.7 标杆实现

- 源算子路径（`source_op_path`）：`/mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py`（`_conv1d_pointwise_kernel` builder，约 L543-638；`Conv1dPointwiseKernel` 类，L641-742）。NPU 参考副本：`examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py`（`_conv1d_pointwise_kernel`）；NPU 分发/包装：`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d.py`。
- 源仓参考实现 / 测试基准：`torch.nn.functional.conv1d(x, weight, bias)`（`kW=1`），见 `examples/TileOPs/tests/ops/test_conv1d.py`。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`F.conv1d`），**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

`_conv1d_pointwise_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一，函数 1）；NPU 接口函数同名，`target="npuir"`。

### 1.2 功能描述

1x1（pointwise）一维卷积：对展平 NCL batch 做沿通道轴的 implicit GEMM，`out[n,c_o,l] = (W @ x[n])[c_o,l] (+ bias[c_o])`，`L_out = L_in`。用于 KW==1 的 Conv1d 调用（如 1x1 通道投影）。

### 1.3 数学公式

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{c_i=0}^{C_{in}-1} W[c_o, c_i]\, x[n, c_i, l], \qquad W = \text{weight}[:, :, 0],\quad L_{out} = L_{in}
$$

### 1.4 算法描述

单步 implicit GEMM：将 `[C_out, C_in]` 权重作为 A、`[C_in, L_in]` 输入作为 B，沿 K=`C_in` 分块累加（fp32），输出 `[C_out, L_out]`（逐 batch），可选 bias epilogue。本算法与 §1.6.0 调研选定算法一致（**基线即选定算法**：plain GEMM + bias，无更优等价算法族）。与源算法结构差异：仅并行组织（三维 grid+threads → 一维 Kernel+persistent，§0.6 R1）、非整除尾块（掩码 gather → 分形 clamp+裁剪，§0.6 R2）、bias epilogue（fragment 逐元素 → MixCV Vector，§0.6 R3）三处不同；GEMM 计算本体（A=`W`、B=`x`、K=C_in、fp32 累加）与源完全一致。

### 1.5 数据流图

```
                    (has_bias=False, 纯 Cube)
GM[x] --slice T.copy--> L1[l1_b] --+
GM[weight] --slice T.copy--> L1[l1_a] --T.gemm--> L0C[l0_c] (fp32) --slice T.copy(fixpipe f32→f16)--> GM[out]

                    (has_bias=True, MixCV)
GM[x] --> L1[l1_b] --+
GM[weight] --> L1[l1_a] --T.gemm--> L0C[l0_c] (fp32) --T.copy(fixpipe f32→f32)--> GM[ws]
GM[ws] --T.copy--> UB[ub_f32] --vbrc(bias)+vadd(fp32)--T.vcast--> UB[ub_f16] --T.copy--> GM[out]
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：调研四问 → 数学等价优化 → 向量化替代 → 向量化轴/布局。本节结论是 §3.1 公式拆解与 §6 循环结构的输入；迁移任务分析对象为 §0.6 重设计后的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：pointwise conv1d（kW=1）的算法族。源算法 / 输入公式只是**基线候选之一**。本算子为**矩阵类（implicit GEMM）**，按 algorithm-research.md §2 判为**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。信息源：algorithm-candidates.md（`kb_search` 检索「conv1d / gemm / pointwise」）、pattern-library（layout.md / constants.md）、本仓 Cube 先例（gqa/ssd）、源算子 §0.3/§0.4、探针 `examples/conv1d/probe/conv1d_geomm_probe.py`。

**R1 等价化简公式候选**（基线必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线**（源算法 / 输入公式） | `out = W @ x + bias`（A=W[C_out,C_in] M×K，B=x[C_in,L] K×N，K 分块 fp32 累加） | — | — | ✅（基线） |
| 1 | bias 折叠（augmented-K） | `W_aug[C_out, C_in+1] = [W | bias]`，`x_aug[C_in+1, L] = [x ; 1]`，`out = W_aug @ x_aug` | 数学恒等（`bias[c_o]·1` 项） | 消 Vector epilogue + 消 GM ws 往返；但 K 变为 `C_in+1`（破坏 16 对齐，需 pad 到 `ceil16(C_in+1)`）+ 需物化 `x_aug`（O(C_in·L) 额外 GM 写） | ❌（K 对齐代价 + x_aug 物化访存 ≫ bias 收益） |
| 2 | 转置 GEMM | `out^T[L, C_out] = x^T[L,C_in] @ W^T[C_in,C_out]` | 数学等价（转置） | 无计算量收益；需转置输出布局（重排代价） | ❌（布局问题非公式化简，见 §1.6.3） |
| 3 | split-K 混合精度 | 分段 K + 原子/二段归约 | 容差内等价（fp32→fp16 分段归约舍入） | 仅提升并行度，不减少 FLOPs/访存 | ❌（本算子 K=C_in 不大，跨核归并代价 ≥ 收益；见 R3） |
| 4 | 权重常量化 / 量化 | — | 不适用（fp16/bf16 无量化契约） | — | ❌（改变 dtype 契约） |

**结论**：无更优于基线的等价化简公式。plain GEMM 的 FLOPs（`2·C_out·L·C_in`）已是该数学语义的下界；bias 折叠（候选 1）引入 K 对齐损失与 `x_aug` 物化访存，净亏；转置（候选 2）是布局选择而非计算化简。基线即选定。

**R2 在线算法**：**无独立「在线变体」结论（分块累加即为单遍流式）**。pointwise conv1d 有两个维度：沿 `L`（N 轴）各输出位置**相互独立**（无跨位置依赖，天然可并行、单遍）；沿 `C_in`（K 轴）的求和是**唯一规约**，其分块累加（running L0C 累加器）本身就是单遍流式结构——无「两遍扫描 + O(N) 中间缓冲」形态存在（源算法 §0.3 亦为单遍 K 分块累加）。结构依据：输出对每个 (c_o,l) 仅依赖对应 K 段输入，无 running max/sum 重缩放语义（对比 online softmax 类）。候选 split-K（R1 候选 3）是并行化手段而非在线算法。结论明确「无（单遍即最优）」。

**R3 复杂度对比**（逐候选四口径；带宽受限算子上访存/遍数是主导项，不得只比 FLOPs；口径标明含什么不含什么）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **基线** `W@x(+bias)`，K 分块 | `2·C_out·L·C_in`（MAC=2；bias 加为 `+C_out·L`，可忽略） | 输入读 `C_in·L·eb` + 权重读 `C_out·C_in·eb` + 输出写 `C_out·L·eb`；**无中间 GM 往返**（无 bias 时）；bias 时 `+ws fp32 往返 2·C_out·L·4B`（仅 bias 路径） | 1（每输入读一次；K 分块不重复读 x/W） | `O(block_m·block_n)`（L0C fp32 tile）+ `O(block_m·block_k + block_k·block_n)`（L1 tile） | 高（tile 间独立，logical = GY·GX·n 核；无跨核归并） |
| 候选 1（bias 折叠 augmented-K） | `2·C_out·L·(C_in+1)`（+K 填充后 `2·C_out·L·ceil16(C_in+1)`） | 输入读 `(C_in+1)·L·eb`（含物化 x_aug 写 `(C_in+1)·L·eb`）+ 权重读 + 输出写 | 1（但 x_aug 需额外写后读一遍） | 同基线 + x_aug O(C_in·L) GM | 同基线 |
| 候选 2（转置 GEMM） | `2·C_out·L·C_in`（同） | 同基线 + 输出转置重排 | 1 | 同基线 | 同基线 |
| 候选 3（split-K） | `2·C_out·L·C_in`（同） | 同基线 + 分段部分和 GM 往返 `O(split·C_out·L)` | 1（K 分块；但需二段归约读回部分和） | `O(split·block)` 部分和 | 并行度↑，**跨核归并代价**（部分和写回 + 二段读） |

**结论**：基线四口径全维度最优（无额外 GM 往返、单遍、O(tile) 驻留）；候选 1/2/3 均在某口径劣于基线（K 对齐/重排/归并代价），无并列争议。基线即选定。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰附本仓文档与 examples 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **基线** | MAC 密集 → **Cube**（`T.gemm` + L0A/L0B/L0C），正确落单元；epilogue（bias）→ Vector | L0C `block_m·block_n·4 = 64·128·4 = 32KB ≤ 128KB`；L1 `(64·128 + 128·128)·2 = 48KB/槽 ≤ 512KB`；UB（bias 路径）`≤ 80KB ≤ 192KB`（§4.5） | K/N 尾轴分形 ≥16（`ceil16` clamp）；32B 对齐（fp16 尾轴 128·2=256B）✓ | logical / persistent 边界均为编译期常量（N/block 为 JIT 参数）✓ | K 分块可 `T.Pipelined` 双缓冲；Cube 段与 bias Vector 段可 CV 融合 ✓ | ✅ **采纳** |
| 候选 1（bias 折叠） | 同基线（Cube） | 同基线（x_aug 额外 GM） | **K=`C_in+1` 非 16 对齐**（如 C_in=80 → 81），需 pad 到 96（+18% K） | 同基线 | 同基线 | ❌（K 对齐损失 + x_aug 物化访存 > bias Vector 收益） |
| 候选 2（转置 GEMM） | 同基线（Cube） | 同基线 | 输出转置需 repack（核内 `T.transpose` 或 host permute，百 µs 级净亏，见 layout.md PL-1.1/1.3） | 同基线 | 转置 epilogue 有 `TRAP-transpose-epilogue-poison`（毒化整 kernel 2.6x）风险 | ❌（重排代价；见 §1.6.3） |
| 候选 3（split-K） | 同基线 + 跨核归约（Vector） | 部分和 GM 往返 | — | 需动态 split 边界 | 二段归约需跨核同步 / 两 kernel | ❌（K=C_in 小，归并代价 ≥ 收益） |

**调研结论**：选定算法族 = **基线 plain implicit GEMM（A=`W[C_out,C_in]` M×K，B=`x[C_in,L]` K×N，K 分块 fp32 累加，可选 bias）**。关键依据：R1 无更优等价化简（基线已是最小 FLOPs 形式）；R2 单遍 K 分块累加即为最优流式（无两遍在线变体）；R3 基线四口径全维度最优（无额外 GM 往返、单遍、O(tile) 驻留）；R4 计算单元匹配（MAC→Cube）、片上容量达标（L0C 32KB / L1 48KB/槽）、分形对齐（ceil16 clamp）、静态边界、可流水/CV 融合。与基线（源算法）结构差异：**无算法级差异**——调研确认源算法即最优，仅需按 §0.6 重设计其并行组织（R1）、尾块处理（R2）、bias epilogue（R3）为 NPU 形态。「无更优替代」调研范围：algorithm-candidates.md（conv/gemm 族命中行，经 `kb_search.py` 检索）、恒等变形自查（R1 候选 1–4 全部评估）、pattern-library（layout.md PL-1.1/1.2/1.3、constants.md、cases.md 的 gqa/ssd Cube 迁移档案）、本仓 Cube 先例（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`）、源算子 §0.3/§0.4、探针实测。源优化手段意图承接见 §0.6 末段。本结论是 §1.4 与 §1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2）**（完整调研，流量项 + 发射项 + 容量项均算）：

```
代表 workload: whisper-1x1 (N=1, C_in=80, C_out=1280, L=3000, fp16)
估算下界: ≈ 14µs（流量 13.5µs + 发射 <2µs + 容量不约束，引用 CONST-mte2-degradation / CONST-L1-port-bw / CONST-vector-launch-overhead / CONST-capacity-910B2C）
  流量项: GM 总流量 = x 读 80·3000·2=0.48MB + W 读 1280·80·2=0.20MB + out 写 1280·3000·2=7.68MB ≈ 8.36MB；
         混合读写流量地板按 CONST-mte2-degradation（16M 档 30.2 / 64M 档 20.9 GB/s/core，8MB 档取 ~31 GB/s/core）× 20 核 ≈ 0.62 TB/s → 8.36MB / 0.62TB/s ≈ 13.5µs（输出写主导；输出占 92%）
  发射项(Cube 操作数流地板): L1→L0 操作数流 = Σ_tasks (block_m·K + K·block_n)·2B = 480·((64+128)·80)·2 ≈ 14.75MB
         ÷ CONST-L1-port-bw(≈148–154 GB/s/核) ÷ 20 核 ≈ 4.9µs
  容量项: L0C tile 32KB ≤ 128KB（无 store_fixpipe-GM 强制往返的超容问题）；不构成下界
  注: Cube MAC 吞吐未实测（constants.md 无 Cube FLOPs 常数）——标注「未实测假设」；0.61 GFLOP 按较大算力估计远小于流量项
  参考: 估算下界以流量项为主导（输出 7.68MB），符合带宽受限画像；Stage 4 以 msprof Task Duration 对账（hardware-cost-model.md §3）
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线 plain GEMM + optional bias）。逐项评估候选优化，无采纳项则写结论与依据。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | bias 折叠（augmented-K） | `W@x + bias` | `[W | bias] @ [x ; 1]` | 数学恒等（`bias[c_o]·1`）；但 K 破坏 16 对齐需 pad，且需物化 `x_aug` | **净亏**：K 81→96（+18% K 计算）+ `x_aug` 物化 O(C_in·L) 额外 GM 写读；不采纳 |
| 2 | 转置 GEMM | `out = W@x` | `out^T = x^T @ W^T` | 数学等价（转置） | **无收益**：FLOPs/访存同；引入输出重排（核内 `T.transpose` ~µs 或 host permute ~百µs 净亏，PL-1.1/1.3）；布局问题归 §1.6.3，不属公式化简；不采纳 |
| 3 | split-K | 单段 K 累加 | 分段部分和 + 二段归约 | 容差内等价（fp32 分段归约到 fp16 的舍入） | **净亏**：K=C_in 小，部分和 GM 往返 + 跨核归并代价 ≥ 收益；不采纳 |
| 4 | 除法/超越函数降代换、公共子表达式消除、稳定化变形 | — | — | **不适用**：本算子为纯 GEMM + 加法，无除法、无超越函数、无公共子表达式、无溢出风险需稳定化 | 无 |

**优化结论**：**无优化空间**。本算子为 plain implicit GEMM（`out = W@x (+bias)`）——无除法的乘倒数、无超越函数降代换、无公共子表达式（每项 M/K/N 元素仅用一次）、无归约稳定化需求（纯线性累加，无溢出/下溢风险）；三个候选（bias 折叠 / 转置 / split-K）经 §1.6.0 R1/R3/R4 评估均无正收益（K 对齐损失、重排代价、跨核归并代价），不采纳。优化后公式 = 基线公式（`out = W @ x + bias`），供 §3.1 公式拆解使用。**因无采纳优化项，按 skill Phase 2 第 5 步与 gate `S1-EQUIV-EXEC` 的豁免规则（「无优化空间」显式结论），不生成 `verify_equiv.py`。**

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 本算子主导计算为 Cube（矩阵单元），非逐元素向量——§1.6.2 盘点实现方案中全部循环与标量计算点，判定能否用向量 / Cube 单元等价替代。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | `W @ x` 矩阵乘累加 | 源 `T.gemm`（tensor core mma） | **Cube 单元** `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])` | ✅（Cube，非标量） | — |
| 2 | bias 逐元素相加 | 源 `T.Parallel` fragment 逐元素 `+bias[oc]` | `T.vbrc(bias_col, bias_brc)` + `T.vadd(bias_brc, ub_f32, ub_f32)`（Vector 段，`_ssd_chunk_scan_fwd_kernel.py` 同款 `T.vbrc`/`T.vadd` 用法佐证；`PL-1.15-broadcast-perf-tax`：列广播须走 `vbrc` 展开而非未文档化列广播） | ✅ | — |
| 3 | 输出 cast（fp32→fp16/bf16） | 源 `T.cast(out_local, dtype)`（epilogue） | 无 bias：fixpipe `T.copy(l0_c, out)` 的 fp32→fp16 量化（`store_fixpipe` 合法量化组合，`CONST-store-fixpipe-gm-only`：max_rel 4.7e-4 = 1 f16 ulp）；有 bias：`T.vcast(ub_f32, ub_f16)` | ✅ | — |
| 4 | K 维分块循环 | 源 `T.Pipelined(ceildiv(c_in, block_k))` | `T.serial(ceil(C_in/block_K))`（tile 级调度循环，非逐元素；Stage 4 可 `T.Pipelined` 多缓冲） | ✅（tile 级循环，非标量热点） | — |
| 5 | `cid → (bz, by, bx)` 解码 | 源三维 grid 隐式 | 标量 `cid // (GY*GX)`、`(cid // GX) % GY`、`cid % GX` | ❌ | **block 级任务映射计算**，每 tile 少量标量，非逐元素热点，无向量等价 API（可接受理由类别①） |
| 6 | 尾块尺寸 `tm/tk/tn/tmc/tkc/tnc` | 源 `if_then_else` 逐元素掩码 | 标量 `T.min` / `T.ceildiv` / `T.max(16, ...)` | ❌ | **依赖动态 shape 的边界处理**（尾块裁剪），每 tile 一次，非逐元素热点（可接受理由类别④）；NPU 无「逐元素条件写向量」形态，掩码 gather 以 slice 尾块 `T.copy` + 分形 clamp 替代 |
| 7 | `T.min(cid, num_logical-1)` 钳位 | — | 标量 `T.min` | ❌ | block 级索引守卫，每 tile 一次（类别①）；防 `T.ceildiv` 对 idle 核的越界（gqa 先例注释） |

**向量化结论**：逐元素计算点（bias 相加、输出 cast）已**全部向量化**（`vbrc`/`vadd`/`vcast`，API 均有 examples 佐证）；矩阵乘由 Cube 单元承接（`T.gemm`）；保留 3 处标量（cid 解码 / 尾块尺寸 / cid 钳位），均属 block 级任务映射或动态边界，非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。**矩阵乘不适用「逐元素向量化」——落 Cube 是正确单元选择（非保留标量循环）。**

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> Cube 类算子：fractal/NZ 布局与操作数轴选择是设计变量；epilogue（bias）属逐元素/广播类，其向量化轴独立评估。

**轴质量评分维度**（逐候选填写）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | N 轴（L）长是否为向量宽度整数倍 | L=3000 非 128 整数倍（尾块 56 / 128 ≈ 44% 尾 lane 浪费，仅末 tile 出现）；`ceil16` clamp 后分形对齐 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len`（vw=128 tile 粒度） | 末 tile `(24·128−3000)/3000 ≈ 2.4%`（整算子口径） |
| 累加链形态 | K 轴为串行累加维；N/M 为向量化/分形维 | K=C_in 串行累加（Cube 内部）；M/N 分形并行 |
| repack 代价 | 改变布局需要的重排代价 | 候选 1 **无 repack**（W=[M,K]、x=[K,N] 天然匹配）；候选 2 需 x 或输出转置（核内 `T.transpose` µs 级 / host permute 百 µs 级） |
| UB 容量影响 | 布局变化对驻留缓冲尺寸的影响 | 候选 1 无额外驻留；bias 路径 UB ≤80KB（§4.5） |

**候选矩阵**（全部枚举，弃选必须给量化理由，禁止「未考虑」）：

| # | 布局方案 | 向量化轴 / 操作数轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **I/O 原生**：A=`W[C_out,C_in]`（M×K，`a_transpose=False`）、B=`x[C_in,L]`（K×N，`b_transpose=False`）、dst=L0C`[C_out,L]` | M=C_out（分形行）、N=L（分形列） | **无**（W 天然 [M,K]、x 天然 [K,N]，NZ 由 `T.copy` ND2NZ 自动完成） | 基线；零 repack；探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 实证可编译可运行 | ✅ |
| 2 | 转置：A=`x^T`（需转置 x 到 [L,C_in]）或 B=`W^T` | N=C_out 方向 | x 转置（核内 `T.transpose` 或 host permute） | repack 代价 > 收益（PL-1.1 ~5.4µs/次、PL-1.3 host permute 百 µs 净亏）；`TRAP-transpose-epilogue-poison` 风险 | ❌（无收益、增代价） |
| 3 | host permute 进出（NCHW↔NHWC 式通道重排） | C 轴 | 两次 host permute | 百 µs 级 repack > 算子本体流量（whisper 本体 ~14µs），净亏（PL-1.3） | ❌（净亏） |
| 4 | b_transpose 变体（B 存 [L,C_in] 后 `b_transpose=True`） | — | x 需转置为 [L,C_in] | x 原为 [C_in,L]，转置代价同候选 2 | ❌（转置代价） |

> 迁移任务：GPU 源码的并行轴（`bx`=L、`by`=C_out、`bz`=batch）是**输入而非结论**——GPU 的 block 轴与 NPU Cube 分形轴不直接对应；本设计独立评估为「M=C_out、N=L」的分形映射（候选 1），与源码 A/B 角色一致（源码 `T.gemm(weight_shared[m,k], data_shared[k,n])` 即 M=block_m=C_out、N=block_n=L），NPU 侧天然承接。

**布局决策结论**：选定方案 **#1**：核内布局 = NZ 分形（`T.copy` GM→L1 自动 ND2NZ；L0C→GM 自动 fixpipe NZ2ND），操作数轴 = A=`W[C_out,C_in]`（M=C_out × K=C_in）、B=`x[C_in,L]`（K=C_in × N=L）、dst=L0C`[C_out,L]`，**无 repack**（天然匹配 `b_transpose=False`）。该结论同步落入 §3.3 伪代码与 §6 循环结构（L0C tile 形状 `[block_M, block_N]`，Cube 内层 `size=[M, K, N]` 与本节操作数轴一致）；弃选方案（#2/#3/#4）量化理由见候选矩阵。epilogue（bias）向量化轴 = N（L）连续轴（UB fp32 tile 行内连续，`vadd` 逐元素）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Expert（`ascend_mode="Expert"`）。

### 2.2 选型理由

本算子为 Cube（矩阵乘）主导算子，须显式使用 L1（Cube 操作数缓存）与 L0C（fp32 累加器）内存层级，并通过 `T.gemm` 走 Cube 单元。Expert 模式是 npuir 上显式 L1/L0C 与 `T.Scope("Cube")` 的唯一可行路径：Developer 模式的 `T.alloc_shared` / `T.alloc_fragment` 在 persistent + gemm 结构下存在运行时崩溃（`TRAP-DEVMODE-PERSIST-GEMM`：Developer+persistent+gemm 混排 unaligned UUB 崩溃；Expert 同结构正常），且 `PL-1.16-expert-dualscope-bypass` 记录 Developer 在标量化 / 条件构造 / persistent+gemm 下的结构级阻塞，绕法即 Expert 双 Scope。bias 路径需 Cube + Vector 双 Scope（MixCV），亦属 Expert（`TRAP-DEVMODE-PERSIST-GEMM` / ssd 迁移档案 CASE-ssd-chunkscan-migration 实证：MixCV 形态仅 Expert 支持）。探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 即以 `TILELANG_ASCEND_MODE=Expert` + `T.Scope("Cube")` 编译运行成功。**任务硬性要求编程模式 = Expert**。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_L1`（`l1_a`/`l1_b`）、`T.alloc_L0C`（`l0_c`，fp32 累加器）；bias 路径另 `T.alloc_ub`（`ub_f32`/`ub_f16`/`bias_brc`） |
| 计算方式 | Cube：`T.gemm(l1_a, l1_b, l0_c, initC=..., size=[M,K,N])`；Vector（bias）：`T.vbrc`+`T.vadd`+`T.vcast` |
| 数据搬运 | `T.copy` slice 形态（GM→L1 自动 ND2NZ；L0C→GM 自动 fixpipe NZ2ND + 可选 fp32→fp16 量化；GM↔UB 通用路径同 dtype） |
| 同步 | **手动同步**（Expert）：`with T.rs("PIPE_MTE2"/"PIPE_C"/"PIPE_FIX")` 包裹对应流水段；Cube↔Vector bias 握手用 `T.sync_block_set/wait` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式（= 基线，无采纳优化项）；每步映射 Cube / Vector API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(bz, by, bx) = decode(cid)` | tile 任务解码（block 级标量） |
| 2 | `l1_a = W[by·block_M : +tm, k0:k0+tk]` | A 操作数装载（GM→L1，ND2NZ），尾块 slice |
| 3 | `l1_b = x[bz, k0:k0+tk, bx·block_N : +tn]` | B 操作数装载（GM→L1，ND2NZ），尾块 slice |
| 4 | `l0_c = W_tile @ x_tile`（`initC=(k==0)`，fp32 累加） | Cube GEMM，K 分块累加 |
| 5 | `out_tile = cast_f16(l0_c)` | 无 bias：fixpipe L0C→GM（fp32→fp16 量化） |
| 6 | `out_tile = cast_f16(l0_c_f32 + bias[c_o])` | 有 bias：fixpipe→GM ws(fp32)，Vector `+bias`（fp32）再 cast，写 out |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | tile 解码 | `T.serial` / 标量算术 | `cid = T.min(task_id*num_kernels+kernel_id, num_logical-1)`；`bz/by/bx` 除法取模 | Expert |
| 1 | A 装载 | `T.copy(W[by*BM : by*BM+tm, k0:k0+tk], l1_a[0:tm, 0:tk])` | slice 形态（动态 extents 借用）；GM→L1 自动 ND2NZ | Expert |
| 2 | B 装载 | `T.copy(x[bz, k0:k0+tk, bx*BN : bx*BN+tn], l1_b[0:tk, 0:tn])` | slice 形态；绕过 `T.load_nd2nz` 跨步误读（`TRAP-load-nd2nz-strided` / CG-2026-0004） | Expert |
| 3 | GEMM | `T.gemm(l1_a, l1_b, l0_c, initC=(k==0), b_transpose=False, size=[tmc, tkc, tnc])` | `tmc=tkc=tnc=T.max(16,ceil16(...))` 分形 clamp | Expert |
| 4 | 无 bias 输出 | `T.copy(l0_c[0:tl, 0:tn], out[bz, by*BM : +tl, bx*BN : +tn])` | L0C→GM 自动 fixpipe（fp32→fp16 量化） | Expert |
| 5 | 有 bias 输出（Cube 段） | `T.copy(l0_c[0:tl, 0:tn], ws[core, slot, 0:tl, 0:tn])` + `T.sync_block_set(slot)` | fixpipe 不量化（fp32→fp32） | Expert |
| 6 | bias 相加（Vector 段） | `T.copy(ws[core,slot,0:tl,0:tn], ub_f32[0:tl,0:tn])`；`T.vbrc(bias_col, bias_brc)`；`T.vadd(bias_brc, ub_f32, ub_f32)`；`T.vcast(ub_f32, ub_f16, round_mode="rint")`；`T.copy(ub_f16[0:tl,0:tn], out[...])`；`T.sync_block_set(slot_free)` | `bias_col [block_M,1]` 先 `T.copy` 装入；`vbrc` 广播 `[block_M,1]→[block_M,block_N]` | Expert |
| 7 | 同步 | `with T.rs("PIPE_MTE2")` / `T.rs("PIPE_C")` / `T.rs("PIPE_FIX")` | 各流水段包裹 | Expert |

### 3.3 计算伪代码

```python
# Expert 模式；ascend_mode = "Expert"
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

@tilelang.jit(out_idx=[2], target="npuir")
def _conv1d_pointwise_kernel(n, c_in, l_in, c_out, has_bias, dtype="float16"):
    accum_dtype = "float32"
    out_l = l_in                      # kW=1, stride=1, pad=0, dilation=1
    block_M, block_N, block_K = 64, 128, 128   # 承接源 default_config
    GX = T.ceildiv(out_l, block_N)    # N 方向 tile 数（L）
    GY = T.ceildiv(c_out, block_M)    # M 方向 tile 数（C_out）
    num_logical = GY * GX * n
    num_kernels = NPUUtils.get().get_aicore_num()          # Cube 直取实查值 = 20（§5.5）

    def _body(x, weight, out, bias):
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
            num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_M, block_K], dtype)
                l1_b = T.alloc_L1([block_K, block_N], dtype)
                l0_c = T.alloc_L0C([block_M, block_N], accum_dtype)
                for task_id in T.serial(num_local_tasks):              # persistent 核内串行
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bz = cid // (GY * GX)
                    by = (cid // GX) % GY
                    bx = cid % GX
                    tm = T.min(block_M, c_out - by * block_M)
                    tn = T.min(block_N, out_l - bx * block_N)
                    tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                    tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                    for k in T.serial(T.ceildiv(c_in, block_K)):
                        tk = T.min(block_K, c_in - k * block_K)
                        tkc = T.max(16, T.ceildiv(tk, 16) * 16)
                        with T.rs("PIPE_MTE2"):
                            T.copy(weight[by * block_M : by * block_M + tm,
                                          k * block_K : k * block_K + tk],
                                   l1_a[0:tm, 0:tk])
                            T.copy(x[bz, k * block_K : k * block_K + tk,
                                     bx * block_N : bx * block_N + tn],
                                   l1_b[0:tk, 0:tn])
                        with T.rs("PIPE_C"):
                            T.gemm(l1_a, l1_b, l0_c, initC=(k == 0),
                                   b_transpose=False, size=[tmc, tkc, tnc])
                    if not has_bias:
                        with T.rs("PIPE_FIX"):
                            T.copy(l0_c[0:tm, 0:tn],
                                   out[bz, by * block_M : by * block_M + tm,
                                       bx * block_N : bx * block_N + tn])
                    else:
                        with T.rs("PIPE_FIX"):
                            T.copy(l0_c[0:tm, 0:tn],
                                   ws[kernel_id, 0, 0:tm, 0:tn])          # ws fp32 中继
                            T.sync_block_set(0)
            if has_bias:
                with T.Scope("Vector"):
                    ub_f32 = T.alloc_ub([block_M, block_N], accum_dtype)
                    ub_f16 = T.alloc_ub([block_M, block_N], dtype)
                    bias_col = T.alloc_ub([block_M, 1], accum_dtype)
                    bias_brc = T.alloc_ub([block_M, block_N], accum_dtype)
                    for task_id in T.serial(num_local_tasks):
                        cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                        bz = cid // (GY * GX); by = (cid // GX) % GY; bx = cid % GX
                        tm = T.min(block_M, c_out - by * block_M)
                        tn = T.min(block_N, out_l - bx * block_N)
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(0)
                            T.copy(ws[kernel_id, 0, 0:tm, 0:tn], ub_f32[0:tm, 0:tn])
                            T.copy(bias[by * block_M : by * block_M + tm],
                                   bias_col[0:tm, 0])                       # [bm] -> [bm,1]
                        T.vbrc(bias_col, bias_brc)
                        T.vadd(bias_brc, ub_f32, ub_f32)                    # fp32 域加 bias
                        T.vcast(ub_f32, ub_f16, round_mode="rint")
                        with T.rs("PIPE_MTE3"):
                            T.copy(ub_f16[0:tm, 0:tn],
                                   out[bz, by * block_M : by * block_M + tm,
                                       bx * block_N : bx * block_N + tn])
                            T.sync_block_set(1)                             # slot 归还

    if has_bias:
        @T.prim_func
        def _pointwise_bias_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, c_in), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            bias: T.Tensor((c_out,), dtype),
        ):
            _body(x, weight, out, bias)
        return _pointwise_bias_main

    @T.prim_func
    def _pointwise_main(
        x: T.Tensor((n, c_in, l_in), dtype),
        weight: T.Tensor((c_out, c_in), dtype),
        out: T.Tensor((n, c_out, out_l), dtype),
    ):
        _body(x, weight, out, None)
    return _pointwise_main
```

> 注 1：`T.Tensor((...), dtype)` 用**位置参数**传 dtype（避免 `dtype=` 关键字产生 `name 'in_dtype' is not defined` false alarm，design-template §3.3 约定）。
> 注 2：`T.gemm` 的 `size` 用分形 clamp 后的静态 `tmc/tkc/tnc`（≥16，ceil16）；`T.copy` 用动态 `tm/tk/tn` slice 有效区——两者配合完成尾块（§0.6 R2）。
> 注 3：无 bias 路径为**纯 Cube**（单 Scope），有 bias 路径为 **MixCV**（Cube + Vector 双 Scope，per-(kernel_id,slot) flag 握手）。`num_kernels = NPUUtils.get().get_aicore_num()` 在 JIT 期求值（编译期常量）；若 `num_logical < num_kernels`，`num_local_tasks` 对部分核为 0，`cid` 钳位防越界（gqa 先例）。
> 注 4：ws 形状 `[num_kernels, slots, block_M, block_N]` fp32（单槽 `slots=1` 基线；`PL-1.12` 深度 2 双槽为 Stage 4 选项）。该内存由 wrapper 按 JIT shape 分配并作为额外入参传入（对齐 ssd/gqa 的 ws 形态），或经 `T.alloc` 核内分配（Stage 3 细化）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.Scope("Cube")` / `T.Scope("Vector")` | `_gqa_prefill_fwd_kernel.py` L274/L388；`_ssd_chunk_scan_fwd_kernel.py` L234/L344；cube-skill `api-cube.md` | ✅ 已验证（两处 Cube 迁移先例） |
| `T.alloc_L1` | cube-skill `api-cube.md`；`docs/Tilelang.language/内存操作/T.alloc_L1.md`；gqa L278 | ✅ 已验证 |
| `T.alloc_L0C` | cube-skill `api-cube.md`；`docs/Tilelang.language/内存操作/T.alloc_L0C.md`（fp32 累加支持）；gqa L283 | ✅ 已验证 |
| `T.gemm(a, b, c, size, initC, a_transpose, b_transpose)` | `docs/Tilelang.language/线性代数操作/T.gemm.md`（`size=[a,b,c]`：src1 [a,b]/[b,a]，src2 [b,c]/[c,b]，dst [a,c]；fp16 输入 dst 须 fp32）；探针 L100；gqa L334；ssd L299 | ✅ 已验证（文档 + 探针 + 两先例） |
| `T.copy`（slice 形态 GM→L1 ND2NZ；L0C→GM fixpipe f32→f16） | `docs/Tilelang.language/内存操作/T.copy.md`（方向表 + 动态 extents 规则 + L0C→GM 量化组合）；探针 L88-108 | ✅ 已验证（探针实测） |
| `T.copy`（GM↔UB 通用路径，同 dtype） | 同上（Expert 方向表 GM↔UB √）；ssd L292 | ✅ 已验证 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；ssd L349；gqa L397 | ✅ 已验证 |
| `T.vbrc` / `T.vadd` / `T.vcast` | `_ssd_chunk_scan_fwd_kernel.py` L439（`T.vbrc`）、L443（`T.vadd`）、L480 注释（Expert `T.copy` 不能跨 dtype，须显式 cast）；gqa L421 | ✅ 已验证（两先例） |
| `T.rs("PIPE_MTE2"/"PIPE_C"/"PIPE_FIX"/"PIPE_MTE3")` | `_gqa_prefill_fwd_kernel.py` L307-386；`_ssd_chunk_scan_fwd_kernel.py` L253-341 | ✅ 已验证 |
| `T.sync_block_set` / `T.sync_block_wait` | gqa L308/L351；ssd L254/L267 | ✅ 已验证 |
| `T.serial`（核内串行）/ `T.ceildiv` / `T.min` / `T.max` | gqa L284；ssd L257 | ✅ 已验证 |
| `NPUUtils.get().get_aicore_num()` | `tilelang/utils/npu_utils.py` L112；本机实查 = 20（§5.5） | ✅ 已验证（实查） |
| `T.load_nd2nz` | `docs/Tilelang.language/内存操作/T.load_nd2nz.md`；**本设计避开**（`TRAP-load-nd2nz-strided` 跨步区域静默平坦误读；CG-2026-0004） | ⚠️ 存在但不用（改用 slice `T.copy`） |
| `T.store_fixpipe` | `docs/Tilelang.language/内存操作/T.store_fixpipe.md`；`T.copy` L0C→GM 等价自动转换，无需显式调用（cube-skill `api-cube.md`） | ✅ 等价（`T.copy` 替代） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为三维 `T.Kernel(bx, by, bz)`） | §0.6 R1：一维 `T.Kernel(num_kernels, is_npu=True)` + `cid` 解码 `(bz,by,bx)` + persistent 核内串行 |
| GPU 专用 API | Yes（`threads=128`、`T.use_swizzle`、`T.alloc_shared`/`T.alloc_fragment`、`T.Pipelined`/`T.Parallel` 元素级、`compile_flags`） | 全部改写为 NPU Expert Cube API（§0.5/§0.6） |
| GEMM 非整除 | **Yes**（`c_out % block_M`、`l_in % block_N`、`c_in % block_K` 均可能非整除） | §0.6 R2：分形 clamp `T.max(16, ceil16(tail))` + slice 尾块 `T.copy` + 裁剪 store |
| L0C 容量溢出 | 需核对 | `block_M·block_N·4 = 64·128·4 = 32KB ≤ 128KB`（`CONST-capacity-910B2C`）✓ |
| 分核策略 | **Yes** | §5.5 三要素（逻辑核数计算 / 物理核数 `get_aicore_num()` 实查 / 规模判定与分核方案） |
| `T.load_nd2nz` 跨步误读 | **Yes**（x 装载 `x[bz, ci0:ci0+BK, l0:l0+BN]` 行跨步 = l_in） | 改用 **slice 形态 `T.copy`**（文档化、探针实测的绕法；`TRAP-load-nd2nz-strided` / CG-2026-0004） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

> 迁移任务的完整差异分析见 §0.5/§0.6，此处只汇总影响 API 选型的关键技术差异。

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | 三维 `T.Kernel(ceildiv(l_in,BN), ceildiv(c_out,BM), n, threads=128)` | 一维 `T.Kernel(num_kernels, is_npu=True)` + cid 解码 | §0.6 R1 |
| GEMM API | `T.gemm(weight_shared, data_shared, out_local)`（fragment 累加，Developer） | `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])`（Expert，L1/L0C） | §0.6 R1（Cube 路径） |
| 内存分配 | `T.alloc_shared` + `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_L1` + `T.alloc_L0C` | §0.5 等价替换 |
| 累加器初始化 | `T.clear(out_local)` + 无 `initC` | `initC=(k==0)`（首 K 块清零，余块累加） | 语义等价（首块清零 ≡ `T.clear`） |
| 流水 | `T.Pipelined(ceil(c_in,block_k), num_stages)` | `T.serial`（Stage 4 可 `T.Pipelined` 多缓冲） | §0.5 等价替换 |
| 元素级并行 | `T.Parallel(block_m, block_n)` epilogue | Cube `T.gemm` + Vector `vbrc/vadd/vcast` | §0.6 R2/R3 |
| 非整除尾块 | 逐元素 `if_then_else` 掩码 gather | 分形 clamp + slice 尾块 + 裁剪 store | §0.6 R2 |
| 布局技巧 | `T.use_swizzle` + `enable_rasterization`（bank/rasterization） | 无（NZ 自动 + persistent 调度） | §0.5 舍弃 swizzle |
| `threads` | `threads=128` | 无（一维 Kernel） | §0.5 舍弃（`TRAP-threads-kwarg-noop`） |
| 运行时 target | CUDA（`compile_flags=["-O3","-DENABLE_BF16"]`） | `target="npuir"` | §4.7 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/conv1d/probe/conv1d_geomm_probe.py` | **高度相似（同算子族的可行性探针）** | `T.Scope("Cube")` + `T.alloc_L1/L0C` + `T.gemm(size=[...])`；slice 形态 `T.copy` 处理跨步 im2col；`target="npuir"` + Expert；golden = `F.conv1d` |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高（Cube Expert + MixCV + persistent） | 一维 `T.Kernel(NUM_KERNELS)` + `T.serial(num_local_tasks)` + `cid` 钳位；`T.Scope("Cube")`/`T.Scope("Vector")` 双 Scope；`T.rs("PIPE_*")` 手动同步；`T.sync_block_set/wait` 跨引擎握手；分形 clamp `T.max(16, T.ceildiv(...)*16)` |
| `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py` | 高（Cube Expert + Vector epilogue） | `T.gemm(..., initC=(k==0), size=[...])`；L0C→GM ws 中继；`T.vbrc`/`T.vadd` Vector 段；尾块 clamp；Expert `T.copy` 不跨 dtype（须显式 `T.vcast`） |
| `examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py` | 高（同源 builder 的 NPU 参考副本） | `_conv1d_pointwise_kernel` 源 builder 全文（param/tiling/data-flow 契约） |
| `.agents/skills/tilelang-cube-skill/references/api-cube.md` | —（API 指南） | Cube 内存/搬运/计算/出数 API 速查 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL） |
| `weight` | `(c_out, c_in)` | 同 `x` | host `weight[:, :, 0].contiguous()` 后的 2D 权重（GEMM A） |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(n, c_out, l_in)` | 同 `x` | 输出（NCL；`L_out = L_in`），JIT `out_idx=[2]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `l1_a` | `(block_M, block_K)` | input dtype | L1 | A 操作数（W tile），ND2NZ |
| `l1_b` | `(block_K, block_N)` | input dtype | L1 | B 操作数（x tile），ND2NZ |
| `l0_c` | `(block_M, block_N)` | float32 | L0C | fp32 累加器（`T.gemm` dst） |
| `ws` | `(num_kernels, slots, block_M, block_N)` | float32 | GM | Cube→Vector bias 中继（仅 bias 路径） |
| `ub_f32` | `(block_M, block_N)` | float32 | UB | bias 相加缓冲（仅 bias 路径） |
| `ub_f16` | `(block_M, block_N)` | input dtype | UB | cast 后输出缓冲（仅 bias 路径） |
| `bias_col` | `(block_M, 1)` | float32 | UB | 本 tile 的 bias 向量（仅 bias 路径） |
| `bias_brc` | `(block_M, block_N)` | float32 | UB | bias 广播缓冲（`vbrc`，仅 bias 路径） |

### 4.4 内存搬运路径

```
无 bias（纯 Cube）:
GM[weight] --slice T.copy(ND2NZ)--> L1[l1_a] --+
GM[x]      --slice T.copy(ND2NZ)--> L1[l1_b] --T.gemm(fp32)--> L0C[l0_c]
    --slice T.copy(fixpipe f32→f16)--> GM[out]

有 bias（MixCV）:
GM[weight] --> L1[l1_a] --+
GM[x]      --> L1[l1_b] --T.gemm(fp32)--> L0C[l0_c]
    --T.copy(fixpipe f32→f32)--> GM[ws]            (Cube 段)
GM[ws] --T.copy--> UB[ub_f32] --vbrc(bias_col)+vadd(fp32)--> UB[ub_f32]
    --vcast(rint)--> UB[ub_f16] --T.copy--> GM[out]  (Vector 段；bias_col 由 GM[bias] 装入)
```

**规则**：GM→L1 自动 ND2NZ（等价 `T.load_nd2nz`）；L0C→GM 自动 fixpipe（f32→f16 量化 / f32→f32 直存）；GM↔UB 通用路径要求同 dtype（Expert `T.copy` 不跨 dtype，`TRAP-C12-copy-dtype-cast`），故 bias 路径中 fp32→fp16 由显式 `T.vcast` 承担。

### 4.5 L0C / L1 / UB 内存预算

**L0C（128KB/核，`CONST-capacity-910B2C`）**：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `l0_c` | (64, 128) | fp32 | 64·128·4 = 32768 |
| **总计** | | | **32768 字节（32KB）；占 L0C 128KB 的 25%，余量 75%** |

**L1（512KB/核，`CONST-capacity-910B2C`）**：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `l1_a` | (64, 128) | fp16/bf16 | 16384 B |
| `l1_b` | (128, 128) | fp16/bf16 | 32768 B |
| **单槽小计** | | | **49152 B** |
| 双槽占用（ping-pong，Stage 4） | | | **98304 B** |

**UB（192KB/AIV，`CONST-capacity-910B2C`；仅 bias 路径）**：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `ub_f32` | (64, 128) | fp32 | 32768 B |
| `ub_f16` | (64, 128) | fp16/bf16 | 16384 B |
| `bias_col` | (64, 1) | fp32 | 256 B |
| `bias_brc` | (64, 128) | fp32 | 32768 B |
| **总计** | | | **≈ 82176 B（80KB，占 UB 192KB 的 ~42%，余量 ~58%；按 ×1.12 上偏系数 → ~90KB，余量 ~53%）** |

说明：UB 预算仅 bias 路径存在（无 bias 时纯 Cube，无 UB）。按 `CONST-capacity-910B2C`「显式 alloc 结构实测 ratio ≈0.98–1.12」口径，80KB 显式分配（逐 buffer 显式 `alloc_ub`）实际需求 ≈ 80–90KB，余量充分。若 Stage 4 需增大 `block_M`（如 128），bias 路径 UB 升至 ~160KB（×1.12 ≈ 179KB，仍 <192KB，余量紧张）——届时可将 `bias_brc` 与 `ub_f32` 复用以降至 ~128KB（§9 风险点）。

### 4.6 动态轴定义

无动态轴。`n / c_in / l_in / c_out / has_bias` 均为 JIT 编译期参数（每 shape 一特化）；尾块尺寸由编译期常量经 `T.min` 求值。所有循环边界（`num_local_tasks`、`ceil(c_in/block_K)`）为编译期常量，满足「核内串行边界静态」。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
```

并以 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")`（或 wrapper `ascend_mode = "Expert"`）声明 Expert 模式（对齐探针 `conv1d_geomm_probe.py` L41）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Cube（无 bias）/ 混合 Cube+Vector（有 bias，bias epilogue）。

**判定依据**：主体为矩阵乘 `W@x`（MAC 密集）→ Cube 单元；有 bias 时 epilogue 的逐元素加 bias 在 Vector 单元。判定依据 = `T.gemm` 主导 + 可选 Vector 后处理（§0.6 R3）。

### 5.2 Block 划分

```python
block_M, block_N, block_K = 64, 128, 128   # 承接源 default_config（Conv1dPointwiseKernel.default_config）
GX = ceildiv(out_l, block_N)               # N 方向（L）
GY = ceildiv(c_out, block_M)               # M 方向（C_out）
num_logical = GY * GX * n                  # 逻辑核数
```

- **选择理由**：`block_M=64, block_N=128, block_K=128` 为源算子 `default_config` 的取值（`weight_shared [block_m, block_k]` / `data_shared [block_k, block_n]`），迁移承接以保持数据流契约；`block_K=128` 使 K 分块数少（`c_in=80` 单块、`c_in=512` 四块）；L0C tile 32KB 留足余量；L1 单槽 48KB、双缓冲 96KB ≪ 512KB。Stage 4 可扫描更大 `block_M`（如 128，L0C 64KB）与 `block_N`。

### 5.3 约束分析

- **对齐约束**：K/N 尾轴分形 ≥16（`ceil16` clamp）；`block_N=128` fp16 尾轴 128·2=256B（32B 对齐 ✓）；`block_K=128` fp16 256B ✓。
- **L0C 容量**：`64·128·4 = 32KB ≤ 128KB` ✓（`CONST-capacity-910B2C`）。
- **L1 容量**：单槽 48KB、双缓冲 96KB ≤ 512KB ✓。
- **UB 容量**（bias 路径）：80KB ≤ 192KB ✓（§4.5）。
- **分形约束**：`T.gemm` M/N/K 分形 ≥16——`tmc/tkc/tnc = T.max(16, ceil16(tail))` 保证几何合法（§0.6 R2）。

### 5.4 注意事项

- **非整除尾块**：`c_out % block_M`、`l_in % block_N`、`c_in % block_K` 三处非整除均由 §0.6 R2 处理——`T.copy` slice 只搬有效区（越界补 0）、`T.gemm` `size` 用 `ceil16` clamp、输出用 slice `T.copy` 裁剪写出。禁止用 base+size 形态写尾块（`TRAP-T-copy-region-semantics`：越界写坏相邻 GM）。
- **极小 shape**（如 `c_out < block_M` 或 `l_in < block_N`）：`ceil16` clamp 到 16 保证分形合法，多余填充列被裁剪不写回。
- **batch n**：`num_logical = GY·GX·n`，n 折叠进一维 `cid`（`bz = cid // (GY·GX)`）。
- **K 大于 `c_in`**（`c_in < block_K`，如 80）：单 K 块，`tk = c_in`，`tkc = ceil16(80) = 80`。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：`num_logical_kernels = ceil(c_out / block_M) × ceil(out_l / block_N) × n = GY × GX × n`（本算子输出网格 `[n, c_out, l_in]`，M=C_out、N=L、batch=n）。以代表 workload 计：
  - whisper-1x1（N=1, C_in=80, C_out=1280, L=3000）：`GY=ceil(1280/64)=20`，`GX=ceil(3000/128)=24`，`num_logical = 20×24×1 = 480`。
  - encodec-deep-1x1（N=1, C_in=128, C_out=256, L=600）：`GY=4`，`GX=5`，`num_logical = 20`。
  - wide-1x1（N=1, C_in=512, C_out=1024, L=128）：`GY=16`，`GX=1`，`num_logical = 16`。

- **② 物理核数依据（实查）**：`NPUUtils.get().get_aicore_num()` **实查值 = 20**（2026-10-14 本机实查；设备 `npu-smi` 显示 Ascend 910B3，与 `CONST-aicore-910B2C` 记录的 24 不同——以实查为准，禁止文档假设/经验值替代）。**本算子为 Cube（含混合），直接使用返回值 = 20，不翻倍**（纯 Vector 才翻倍；`core-split-strategy.md` §1 要素②）。查询代码与实际返回值记录如下：

```python
from tilelang.utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
num_kernels = physical_cores                        # Cube/混合算子直取 = 20（不翻倍）
```

- **③ 规模判定与分核方案**：采用**统一 persistent 结构**一次覆盖全部规模档位，固定启动内核数 = 物理核数 20（`num_kernels = 20`），核内 `T.serial(num_local_tasks)` 串行处理多个逻辑块任务：

  ```
  num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)   # 编译期常量，边界静态
  cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)     # idle 核钳位
  ```

  - **whisper-1x1（num_logical=480）**：`480 = 20 × 24`，即**物理核数整数倍（24×）**，每核恰 24 个逻辑块，负载完全均匀、无空转；启动 20 个内核而非 480 个，摊薄核启动开销。
  - **encodec-deep-1x1（num_logical=20）**：`20 = 20 × 1`，**1×**，每核 1 块，无串行。
  - **wide-1x1（num_logical=16）**：`16 ≤ 20`（**逻辑核数 ≤ 物理核数**），`num_local_tasks` 对 `kernel_id < 16` 为 1、其余为 0——**无需适配**（结论：逻辑核数 ≤ 物理核数，直接映射，无串行超发）。
  - 规模判定：本算子逻辑核数随 shape 变化，横跨「无需适配」到「中等规模」档；统一 persistent 结构下，内核总数 = 物理核数（1×，整数倍），无负载不均，核内串行边界 `num_local_tasks` 为编译期常量（静态）✓。极大规模（`num_logical ≫ 20`，如 batch 大）同样由该结构覆盖（每核任务数增加，仍为「物理核数 × 1」）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 矩阵乘由 Cube 单元承接、bias 由 Vector 承接——本表循环仅 block 级 / tile 级调度循环与 K 分块累加；**无逐元素标量循环**（§1.6.2）。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_kernels, is_npu=True) as (kernel_id, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| K 方向 | K 分块累加 | `T.serial(ceil(c_in / block_K))` | K 维分块，`T.gemm(initC=(k==0))` 累加（tile 级循环，非逐元素） |
| 矩阵计算 | Cube | `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])` | MAC 密集 → Cube 分形（§1.6.3 操作数轴） |
| epilogue（bias） | Vector | `T.vbrc` + `T.vadd` + `T.vcast` | 逐元素加 bias 全部向量化（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
    with T.Scope("Cube"):
        for task_id in T.serial(num_local_tasks):        # 核内串行（静态边界）
            cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
            bz, by, bx = decode(cid)
            for k in T.serial(ceil(c_in / block_K)):     # K 分块累加
                with T.rs("PIPE_MTE2"): load l1_a, l1_b
                with T.rs("PIPE_C"):    T.gemm(..., initC=(k == 0))
            with T.rs("PIPE_FIX"):      store l0_c (→ out 或 ws)
    if has_bias:
        with T.Scope("Vector"):
            for task_id in T.serial(num_local_tasks):
                with T.rs("PIPE_MTE2"): sync_block_wait + load ws, bias_col
                vbrc + vadd + vcast
                with T.rs("PIPE_MTE3"): store ub_f16 → out + sync_block_set
```

### 6.3 流水线优化

Stage 3 基线用 `T.serial`（正确性优先）；Stage 4 可对 K 分块循环启用 `T.Pipelined(num_stages=2)` 或 L1 ping-pong 双缓冲（两套 `l1_a`/`l1_b` 槽），将 GM→L1 装载隐藏于 Cube 计算窗口（源 `T.Pipelined(..., num_stages)` 的 NPU 等价，`PL-1.12-task-pipeline-depth2` 见 Cube 深度 2 任务流水先例）。bias 路径的 Cube↔Vector 亦可用双槽（`PL-1.12` / ssd 深度 2）提升重叠。

### 6.4 尾块处理

非整除 shape（`c_out % block_M`、`l_in % block_N`、`c_in % block_K`）：`tm/tn/tk = T.min(block, remain)` 决定有效区；`T.copy` slice 只搬有效区（越界补 0）；`T.gemm` 的 `size` 用分形 clamp `tmc/tkc/tnc = T.max(16, ceil16(tail))`；输出用 slice `T.copy(l0_c[0:tm,0:tn], out[...])` 裁剪写出（填充列不落 GM）。所有尾块搬运走 slice 形态（`TRAP-T-copy-region-semantics` / `TRAP-load-nd2nz-strided`）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**手动同步**（Expert）。Cube 段各流水段用 `with T.rs("PIPE_...")` 显式标注；Cube↔Vector bias 交接用 `T.sync_block_set/wait`（per-(kernel_id, slot) flag）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| GM→L1 装载（`l1_a`/`l1_b`） | `with T.rs("PIPE_MTE2")` | 标注为 MTE2 搬运段，供编译器编排 MTE2/Cube 依赖（gqa/ssd 同款） |
| `T.gemm` | `with T.rs("PIPE_C")` | 标注 Cube 计算段；等待 L1 操作数就绪、保护 L1 复用 |
| L0C→GM / L0C→ws（出数） | `with T.rs("PIPE_FIX")` | 标注 fixpipe 段；保证 `l0_c` 写回后才复用/结束 |
| Cube 段 → Vector 段（bias 就绪） | `T.sync_block_set(0)` / `T.sync_block_wait(0)`（`with T.rs("PIPE_FIX")` / `with T.rs("PIPE_MTE2")`） | Cube 写完 ws 后置位，Vector 等位后读 ws（跨引擎生产者-消费者握手） |
| Vector 段（ws 读取/写回） | `with T.rs("PIPE_MTE2")` / `with T.rs("PIPE_MTE3")` | 标注 GM↔UB 搬运段 |
| Vector 段 → Cube 段（ws 槽归还） | `T.sync_block_set(1)` / `T.sync_block_wait(1)` | 防 Cube 下一轮覆盖 Vector 尚未读完的 ws（WAR 保护）；基线单槽，`PL-1.12`/ssd 用双槽深度 2 |

### 7.3 pass_configs 配置

Expert 模式默认；bias 路径需保证 Cube/Vector 双 Scope 的交叉流水按显式 flag 编排（`PL-1.16-expert-dualscope-bypass`：Expert 双 Scope + pass_configs 硬边界）。基线不引入额外 `pass_configs`；Stage 4 若启用自动多缓冲，须核对 `auto-multi-buffer` 对 L1/UB 预算的膨胀（`CONST-capacity-910B2C` / `TRAP-UB-multibuffer-inflation`）。

---
## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据（`out = W@x + bias`，`kW=1`，fp32 累加、fp16/bf16 输出），移植源仓参考实现（`torch.nn.functional.conv1d`），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_conv1d_pointwise(x, weight, bias=None, dtype="float16"):
    """Pointwise Conv1d 参考实现（kW=1, stride=1, pad=0, dilation=1）。

    以 §0.1 语义为依据：weight [C_out, C_in, 1] -> F.conv1d（核宽 1）；
    源仓测试基准同款（examples/TileOPs/tests/ops/test_conv1d.py 的 ref_program = F.conv1d(...)）。
    与 §0.6 的 Cube/persistent 实现完全独立。
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    # 通道轴 GEMM 语义（fp32 参照），再按目标 dtype 舍入
    out_f32 = F.conv1d(x_cpu.to(torch.float32), w_cpu.to(torch.float32),
                       None if b_cpu is None else b_cpu.to(torch.float32),
                       stride=1, padding=0, dilation=1)
    return out_f32.to(x_cpu.dtype)
```

> 说明：`F.conv1d`（`kW=1`）与 builder 契约的 `weight [C_out, C_in, 1]` 一致；等价于 `F.linear` 沿通道轴。golden 在 fp32 域计算（对齐源码 fp32 累加 + fp32 bias 相加），最后 cast 回输入 dtype——与源算子累加域一致。

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓 `convolution_verification` 容差口径一致；K=C_in 累加在 fp32，分形累加顺序与源分块顺序略有差异，落在容差内。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：代表 workload `whisper-1x1`（N=1, C_in=80, C_out=1280, L=3000，`float16`）与 `encodec-deep-1x1`（N=1, C_in=128, C_out=256, L=600，`float16`），`randn` 输入 + 随机 `weight`/`bias`，与 `golden_conv1d_pointwise`（§8.1）在 §8.2 容差下 `torch.testing.assert_close`；分别覆盖 `has_bias=False` 与 `has_bias=True` 两条路径。完整分层套件（L1 非整除 / 多 shape / batch>1、L2 边界 size、Boundary 极小/极大 shape 与 `C_in/C_out/L` 非 16 整除）交由 `tilelang-op-develop` 规划，本设计不枚举。

**L0 必测点**：① 非整除尾块（`C_out=1280` 不被 64 整除? 1280=20×64 整除；须另测如 `C_out=100`、`L=1000`、`C_in=50` 的非整除组合）；② `has_bias` 两条路径一致；③ batch>1（`n` 折叠进 `cid` 的正确性）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| Expert Cube 依赖工具链版本 | Cube Expert（`T.Scope("Cube")` + L1/L0C + `T.gemm`）的全部结论绑定当前工具链（tilelang `0.1.2+417dae99` + CANN，`PL-*`/`CONST-*` 条目版本戳）；工具链升级后须重验（探针已通过为当前证据） |
| 物理核数 20（非 24） | 本机实查 `get_aicore_num()=20`（设备 Ascend 910B3），与 `CONST-aicore-910B2C`（24）不一致——设计以实查为准（§5.5）；`num_kernels` 由 `get_aicore_num()` 在 JIT 期求值，跨设备自适应 |
| `T.gemm` 分形约束 | M/N/K 分形 ≥16——`tmc/tkc/tnc = T.max(16, ceil16(tail))` 保证；`c_in < 16` 的极小通道需补零到 16（K 填充，对输出无影响） |
| 跨引擎传输强制 GM 往返 | bias 路径 Cube→Vector 经 GM ws 中继（`CONST-store-fixpipe-gm-only`：S/P 类中间矩阵无法直写 L1/UB）——bias 路径比无 bias 路径多一次 fp32 输出写读（成本见 §1.6.0 设计期估算）；这是 NPU 上把 bias 放 Vector 的固有代价 |
| L0C/L1 容量 | L0C 32KB ≤ 128KB、L1 单槽 48KB ≤ 512KB（§4.5）——余量充分；若 Stage 4 增大 `block_M/block_N` 须复算（`block_M·block_N·4 ≤ 128KB` → `block_M·block_N ≤ 32768`） |
| UB 容量（bias 路径） | 80KB ≤ 192KB（§4.5）；若 Stage 4 增大 `block_M` 至 128 则 ~160KB（×1.12 ≈ 179KB，余量紧张）——须复用 `bias_brc`↔`ub_f32` 或减小 tile |
| `T.load_nd2nz` 跨步误读 | x 装载为跨步区域（行 stride = `l_in`）——`TRAP-load-nd2nz-strided` / CG-2026-0004：`load_nd2nz` / base+size `T.copy` 对跨步区域静默平坦误读；本设计一律用 slice 形态 `T.copy`（探针实测绕法） |
| Expert 通用路径不跨 dtype | `T.copy` 在 Expert 通用路径（GM↔UB）要求同 dtype（`TRAP-C12-copy-dtype-cast` / `T.copy` 文档 §2.3）——bias 路径 fp32→fp16 必须显式 `T.vcast`，不可依赖 `T.copy` 隐式转换 |
| 尾块越界写 | `TRAP-T-copy-region-semantics`（越界写坏相邻 GM / `[N,1]` 越界读）——所有尾块搬运走双显式 slice，禁止 base+size 形态 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C 溢出 | `block_M·block_N` 过大（>32768） | 编译失败（`cc overflow`） | 维持 `64×128`（32KB）或 `128×256` 上限（128KB） |
| L1 溢出 | `block_M/block_N/block_K` 过大或缓冲槽过多 | 编译失败（`cbuf overflow`） | 维持 `64/128/128`（单槽 48KB）；双缓冲 ≤96KB |
| 尾块 MTE 越界写 | 用 base+size 形态写尾块 | 写坏相邻 GM tensor | 双显式 slice `T.copy(src[0:tm], dst[0:tm])`（§6.4） |
| `T.gemm` 分形非法 | tail < 16 或非 `ceil16` | 编译/运行错误 | `T.max(16, ceil16(tail))` clamp（§0.6 R2） |
| `T.gemm` dst 误设 fp16 | `l0_c` 设 `float16` | 运行卡死（`T.gemm.md` §2.3：fp16 输入 dst 须 fp32） | `l0_c = T.alloc_L0C([...], "float32")` 固定 fp32 |
| bias 精度偏差 | bias 在 fp16 域相加（cast 后再加） | 与 golden（fp32 加 bias）偏差偏大 | bias 在 fp32 域相加（`vadd` fp32）后再 `vcast`（§0.6 R3） |
| Expert `T.copy` 跨 dtype | GM↔UB 拷贝 fp32↔fp16 | 编译报 `does not support element type casting` | 显式 `T.vcast` 承担转换（§9.1） |
| idle 核越界 | `num_logical < num_kernels` 时 `T.ceildiv` 对负数返回 1 | idle 核运行越界任务 | `cid = T.min(..., num_logical-1)` 钳位（§3.3 注 3，gqa 先例） |
| `dtype=` 关键字 false alarm | `T.Tensor(..., dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（§3.3 注 1） |

### 9.3 特殊场景处理

- **非整除 shape**（`C_in/C_out/L` 非 `block` 整数倍或非 16 整数倍）：分形 clamp + slice 尾块 + 裁剪 store（§6.4）。
- **极小 shape**（`C_in < 16`）：K 补零到 16（填充项对输出无影响）。（本算子 pointwise 的 `c_in` 通常 ≥16，如 64/80/128/512。）
- **batch>1**：`n` 折叠进一维 `cid`（`bz = cid // (GY·GX)`）；各 batch tile 独立，无跨 batch 依赖。
- **`has_bias=False`**：纯 Cube 单 Scope，无 UB、无 ws、无跨引擎握手——最小结构。
- **`has_bias=True`**：MixCV 双 Scope，fp32 ws 中继 + Vector bias epilogue（§0.6 R3）。
- **`c_in < block_K`**（如 80）：单 K 块，`tk=c_in`，`tkc=ceil16(c_in)`（80 已 16 对齐）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/conv1d/_conv1d_pointwise_kernel/
├── DESIGN.md               # 本设计文档
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `verify_equiv.py` | — （不适用） | §1.6.1 结论为「无优化空间」，按 gate `S1-EQUIV-EXEC` 豁免规则不生成 |
| `_conv1d_pointwise_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_conv1d_pointwise.py` | ⬜ 待实现（Stage 3） | 精度测试（L0/L1/L2/Boundary） |

### 10.3 命名规范

- 项目目录名：`conv1d`；算子目录名：`_conv1d_pointwise_kernel`；接口函数：`_conv1d_pointwise_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一）；JIT `target="npuir"`，`out_idx=[2]`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ Stage 1 复盘（RETROSPECTIVE.md）
2. ⬜ Golden 函数（§8.1，`F.conv1d` kW=1）
3. ⬜ 无 bias 纯 Cube 实现（最小结构）→ L0 精度比对
4. ⬜ bias 路径（MixCV 双 Scope + ws 中继）→ L0 精度比对
5. ⬜ 非整除尾块 / batch>1 覆盖（L1/L2/Boundary）
6. ⬜ Stage 4 调优（L1 双缓冲、`block_M` 扫描、bias ws 双槽）
