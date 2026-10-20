# _conv1d_group_kernel 算子设计文档（Stage 1 — 迁移设计，Expert Cube）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/convolution/conv1d.py` 的 `_conv1d_group_kernel` builder，L407-560；`GroupConv1dKernel` 类，L929-1051）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），`target="npuir"`，编程模式 **Expert**（`ascend_mode="Expert"`，`T.Scope("Cube")` + `T.alloc_L1` / `T.alloc_L0C` / `T.gemm` / `T.transpose`）。
> 本设计按 skill `tilelang-op-design` 三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出，为本函数 5 个迁移函数中的第 4 个（函数 4，`_conv1d_group_kernel`，服务 `groups>1` 的分组卷积）。
> **结构基石（复用函数 2 `_conv1d_kernel` 的一般迁移设计）**：本算子是**一般 conv1d 加上 group 轴**——复用其 Expert Cube implicit GEMM + `T.Scope("Cube")`/`T.Scope("Vector")` 双 Scope + 深度 2 双槽 4-flag MixCV bias 握手 + 一维 persistent 分核 + tap-major K 分块 + `w_flat`/`x_pad` 前置，**新增 group 轴 tiling/index**：grid z 折叠为 `n·groups`、`cid` 解码出 `(batch_id, group_id)`、A/B/bias/out 全部按 `group_id` 选片（`oc_base = group_id·C_out_g + by·block_M`，B 取该 group 的通道段）。已吸收函数 1/2 Stage 3 实测修正：跨引擎握手的「槽空闲」前导 `set(1)/set(3)` 置于**消费者（Vector）Scope**（同引擎自 set→自 wait 不 retire，见 `examples/conv1d/_conv1d_pointwise_kernel/repro/TRAP-sync-self-set-wait.py`）；bias 由输入 dtype→fp32 必须显式 `T.vcast`（Expert `T.copy` 不跨 dtype）。
> 与函数 2 的关键差异：**源码 group builder 无 host `weight.permute`**（`forward` = `launch(self, x, weight, bias=bias)`，直接传 `weight (C_out,C_in_g,K_W)`）；K 序仍为 `(k_w, c_i_g)`（`kw = k_idx // c_in_g`、`ci_g = k_idx % c_in_g`），故 A 在源码里是**逐元素 gather**（注释明言 "the weight is staged by gather because that order is not the one it is stored in"）——NPU 侧由 §0.6 R3 的 `w_flat` 前置一次性消除。
> 可行性探针：`examples/conv1d/probe/conv1d_geomm_probe.py`（Cube conv-as-GEMM 在 npuir 编译并运行成功；per-tap slice 形态 `T.copy` 承载跨步/重叠 im2col，`target="npuir"` + Expert；避开 `T.load_nd2nz` 跨步误读，见 CG-2026-0004）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：**分组**一维卷积（grouped Conv1d，`groups>1`）——对 NCL batch，逐 batch、逐 group 计算

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{c_i^g=0}^{C_{in}^g-1}\;\sum_{k_w=0}^{K_W-1} W[c_o, c_i^g, k_w]\cdot x[n,\; g(c_o)\cdot C_{in}^g + c_i^g,\; l\cdot s_W + k_w\cdot d_W - p_L], \qquad l \in [0, L_{out})
$$

其中 `g(c_o) = c_o // C_out^g` 为输出通道 `c_o` 所属 group，`C_in^g = C_in/groups`、`C_out^g = C_out/groups` 为单 group 的输入/输出通道数，`k_total = C_in^g·K_W`；`L_out = (L_in + p_L + p_R - d_W·(K_W-1) - 1)//s_W + 1`（源码 L424，`out_l` 定义）；`s_W`=stride、`d_W`=dilation、`p_L`/`p_R`=pad_left/pad_right、`K_W`=kernel_l；`bias` 可选（`has_bias`）。即：**每个 group 是一个独立的一般 conv1d（implicit GEMM，收缩轴 `C_in^g·K_W`），groups 个 group 沿输出通道切分**。

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | 逐 group：`out_g = reshape(W_g · im2col(x_g)) (+ bias_g)`，其中 `W_g[C_out^g, C_in^g·K_W]`、`im2col_g[C_in^g·K_W, L_out]`；`W_g[c_o^g, k_w·C_in^g + c_i^g] = W[group_id·C_out^g + c_o^g, c_i^g, k_w]`，`x_g = x[n, group_id·C_in^g : (group_id+1)·C_in^g, :]` |
| 规约语义 | 唯一规约轴 = 单 group 的 `K = C_in^g·K_W`（GEMM 的 K 轴，k 序 `(k_w, c_i^g)`）；**规约/累加顺序** = 源码 `T.gemm` 的 K 分块累加（fp32 累加器 `out_local`）；**无跨 L、无跨 group 的规约**（每个输出位置 `(n,c_o,l)` 独立，其 group 由 `c_o` 唯一确定） |
| dtype 语义 | 输入/权重/输出同 dtype（`float16 \| bfloat16`，`dtype=` JIT 参数）；**中间累加 fp32**（`accum_dtype = "float"`）；**bias 在 fp32 累加域相加后统一 cast 到输出 dtype**（源码 `T.cast(out_local[i,j] + T.cast(bias[oc], accum_dtype), dtype)`，L517） |
| 边界语义 | **越界输入经掩码补 0**（`data_shared[k,j] = if_then_else((k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in), x[batch_id, group_id·c_in_g+ci_g, il], 0)`，L502-506）——`il = ol·s_W + k_w·d_W - p_L` 可 <0 或 ≥l_in；A 同样掩码越界（`(oc_g<c_out_g)&(k_idx<k_total)`，L487-491）；输出非整除尾块（`c_out_g % block_m`、`out_l % block_n`）由 `if_then_else((oc_g<c_out_g)&(ol<out_l), …, 0)` 掩码处理（L510-525）；**group 溢出保护**：仅当 `c_out_g % block_m == 0` 时用 `T.copy(out_shared, out[batch_id, oc_base, bx*block_n])`（tile 恰止于本 group 最后一个通道、不会写进下一 group 的行），否则逐元素掩码写（L527-537）；无 nan/inf 特殊语义（纯线性累加，nan/inf 按 IEEE 自然传播）；空维度不产生（`L_out>0` 为 manifest shape rule） |
| host 侧语义 | `forward` 直接 `launch(self, x, weight, bias=bias)`（L1049-1051），**不做 `weight.permute`**（与函数 2 一般 kernel 的关键差异）——builder 收到原生 `weight (C_out, C_in_g, K_W)`；`_build_program` 仅校验 `c_in % groups == 0 and c_out % groups == 0`（L1011-1015）；`_check_config` 要求 `block_m`/`block_k` 为 16 的倍数（L1032-1038） |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓测试基准 `torch.nn.functional.conv1d(..., groups=groups)`），**不复刻 NPU 算法**。

### 0.2 源算子输入输出

源 builder `_conv1d_group_kernel(n, c_in, l_in, c_out, kernel_l, stride_l, pad_left, pad_right, dilation_l, has_bias, dtype, groups, c_in_g, c_out_g)` 的 prim_func 契约（bias 变体多一个 `bias` 入参；`c_in_g`/`c_out_g` 缺省时取 `c_in//groups`/`c_out//groups`）：

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL） |
| `weight` | `(c_out, c_in_g, kernel_l)` | 同 `x` | 分组权重（**原生布局，无 host 重排**；GEMM A 操作数按 group 取片） |
| `out` | `(n, c_out, out_l)` | 同 `x` | 输出（NCL；`L_out` 由 stride/dilation/padding 决定），`out_idx=[2]` |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

> Op 层输入 `weight` 为 `[C_out, C_in/groups, K_W]`（`groups` 由 `Conv1dFwdOp(groups=...)` 给定）；builder 收到的即此 shape。NPU 侧不改 op 层 I/O（`x`/`weight`/`out`/`bias` shape 与 dtype 与源一致）；`groups` 为 JIT 编译期参数。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host：`c_in_g = c_in//groups`、`c_out_g = c_out//groups`（缺省）；无 weight 重排；`launch` | `x`, `weight`, `bias` | — | 前置（分组尺寸推导） |
| 2 | 三维 grid：`T.Kernel(ceildiv(out_l, block_n), ceildiv(c_out_g, block_m), n*groups, threads=threads)` | — | block 编号 `(bx, by, bz)` | 任务划分（**group 折叠进 grid z**） |
| 3 | `batch_id = bz // groups`；`group_id = bz % groups`；`oc_base = group_id*c_out_g + by*block_m` | `bz` | `batch_id`, `group_id`, `oc_base` | **group 轴解码** |
| 4 | `T.clear(out_local)` | — | `out_local [block_m, block_n]` fp32 清零 | 累加器初始化 |
| 5 | K 循环 `T.Pipelined(ceildiv(k_total, block_k), num_stages)`：A gather `weight_shared[i,k] = if_then_else((oc_g<c_out_g)&(k_idx<k_total), weight[oc, ci_g, kw], 0)`（`oc = group_id*c_out_g + oc_g`、`kw = k_idx//c_in_g`、`ci_g = k_idx%c_in_g`） | `weight` | `weight_shared [block_m, block_k]` | A 操作数装载（按 group 取片 + K 序 reorder gather） |
| 6 | 同循环：B gather `data_shared[k,j] = if_then_else((k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in), x[batch_id, group_id*c_in_g + ci_g, il], 0)`（`il = ol*stride_l + kw*dilation_l - pad_left`） | `x` | `data_shared [block_k, block_n]` | B 操作数装载（**本 group 通道段的 im2col**）+ 越界零填充 |
| 7 | 同循环：`T.gemm(weight_shared, data_shared, out_local)` | A/B tile | `out_local`（累加） | `Σ_{k_w, c_i^g} W[c_o,c_i^g,k_w]·x[n, g·C_in^g+c_i^g, l·s_W+k_w·d_W-p_L]` |
| 8 | epilogue `T.Parallel(block_m, block_n)`：`out_shared[i,j] = if_then_else((oc_g<c_out_g)&(ol<out_l), cast(out_local[i,j]+cast(bias[oc],fp32), dtype), 0)`（无 bias 时省略加 bias；`oc = group_id*c_out_g + oc_g`） | `out_local`, `bias` | `out_shared [block_m, block_n]` dtype | bias 相加 + 输出 cast + M/N 尾块掩码 |
| 9 | 输出：`if c_out_g % block_m == 0: T.copy(out_shared, out[batch_id, oc_base, bx*block_n])` else 逐元素掩码写 `out[batch_id, oc, ol] = out_shared[i,j]`（仅 `oc_g<c_out_g and ol<out_l`） | `out_shared` | `out` | 输出写回（**group 溢出保护**） |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x], GM[weight] --gather(T.Parallel)--> SMEM[weight_shared/data_shared] --T.gemm--> register fragment[out_local] (fp32)
   --epilogue(Parallel: +bias, cast, mask)--> SMEM[out_shared] --T.copy 或掩码写--> GM[out]
  （A 逐元素 gather：oc=group_id*c_out_g+oc_g、ci_g=k_idx%c_in_g、kw=k_idx//c_in_g——K 序 (k_w,c_i_g) 非权重存储序；
    B 逐元素 gather：从本 group 的 x 通道段 [group_id*c_in_g, +c_in_g) 取 im2col，越界补 0）
```

**循环与并行结构**：grid 三维 `(ceildiv(out_l, block_n), ceildiv(c_out_g, block_m), n*groups)`；block 内 `threads`（默认 128）；K 维 `T.Pipelined(ceildiv(k_total, block_k), num_stages)` 分块累加；元素级 `T.Parallel(block_m, block_k)`（A gather）与 `T.Parallel(block_k, block_n)`（B gather）与 `T.Parallel(block_m, block_n)`（epilogue）。默认配置 `GroupConv1dKernel.default_config`：`block_m = min{c∈(16,32,64,128) : c ≥ c_out_g}`（否则 128）、`block_n=128`、`block_k=128`、`threads=128`、`enable_rasterization=True`（L1043-1055）。

**host 侧逻辑**：`forward` 无预处理/后处理（`launch` 直调）；`_build_program` 校验 group 整除性并缓存 builder（`functools.lru_cache(maxsize=64)`）。im2col 在 kernel 内隐式展开（K 轴 `(k_w, c_i^g)`），权重由 gather 按 K 序重排（未在 host 显式重排）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（`weight_shared`/`data_shared`） | 减少 GM 重复访问、放大访存粒度 | K 分块驻留 shared memory | CUDA shared memory | 硬件强相关 |
| 2 | Tensor Core mma（`T.gemm`） | 矩阵算力（MAC 密集） | mma fragment + tensor core 指令 | tensor core（warp 级 mma） | 硬件强相关（NPU 有 Cube 等价） |
| 3 | 异步搬运 / 多级流水（`T.Pipelined(..., num_stages)`） | 访存与计算重叠 | 分块双缓冲流水 | cp.async / 异步拷贝引擎 | 硬件强相关（NPU 有 MTE 等价，映射为 L1 多缓冲） |
| 4 | register 累加（`out_local` fragment fp32） | 数据驻留寄存器，消中间写回 | fragment 累加器 | 大寄存器堆 | 硬件强相关（NPU 有 L0C 等价） |
| 5 | **group 折叠进 grid z**（`bz = batch*n_groups...`，`bz//groups`/`bz%groups`） | 把 `n×groups` 个独立 GEMM 任务并行铺开 | 每个 (batch, group) 一个 block 集合 | 无（纯任务划分） | 可移植（NPU 一维 persistent + `cid` 解码） |
| 6 | **A 按 K 序 gather + 边界掩码**（`weight_shared[i,k]` 逐元素 `if_then_else`） | 使 K 序 `(k_w,c_i_g)` 匹配存储序 `(c_i_g,k_w)`；越界补 0 | 元素级条件 gather | 无（纯算法/边界处理） | **重新设计**（NPU Expert 无逐元素条件写：改 `w_flat` 前置 + slice，见 §0.6 R3） |
| 7 | B 逐元素掩码 gather（`if_then_else((k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in), x, 0)`） | 本 group 通道段 im2col + 越界零填充 | 元素级条件零填充 | 无（纯算法/边界处理） | **重新设计**（改 `x_gpad`/polyphase 前置 + in-bounds slice，见 §0.6 R4） |
| 8 | **输出 group 溢出保护**（`if c_out_g % block_m == 0: T.copy(...) else 掩码写`） | 尾块 tile 不写进下一个 group 的行 | 分支切换 `T.copy` / 掩码写 | 无（纯算法/边界处理） | 可移植（NPU slice extents 天然不越界，见 §0.6 R2） |
| 9 | coalesced 连续访问（输出 `T.copy` 尾轴连续） | 访存带宽 | 连续 slice 读写 | memory coalescing | 硬件强相关（NPU 有 MTE 连续搬运） |
| 10 | swizzle 面板（`T.use_swizzle(CONV_SWIZZLE_PANEL)`） | 消 SMEM bank conflict | 索引重排 | SMEM bank 结构 | 硬件强相关（NPU 不适用） |
| 11 | `threads` 执行参数（默认 128） | CUDA 线程组织 | CUDA 线程模型 | CUDA 线程模型 | 硬件强相关（NPU 一维 Kernel，无对应） |
| 12 | `num_stages` / `enable_rasterization` 调优参数 | CUDA 调度调优 | 分块/流水参数 | CUDA 调度 | 等价替换（NPU 映射为 L1 多缓冲参数） |

> 识别不出来 ≠ 不存在。源码无 warp shuffle、无 online 算法（逐点窗口 + K 分块累加）、无 persistent、无手工 TMA。**与函数 2 一般 kernel 的关键差异**：group builder **无 host `weight.permute`**（A 在 kernel 内按 K 序 gather，权重存储序与 GEMM K 序不一致）；group 折叠进 grid z（`n*groups`）；输出有 `c_out_g % block_m` 分支。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `out = im2col-GEMM (+bias)`、fp32 累加、fp16/bf16 输出 | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 分组语义（逐 group 独立 GEMM，无跨 group 耦合） | 语义 | 无 | — | **保留** | 语义层保留；group 轴映射为并行任务维度（§0.6 R1） |
| GEMM 分块累加（K 轴分块，fp32 累加器） | 算法 | 无（纯数学分块） | — | **保留** | 纯算法层；NPU 沿用 K 分块累加（§6） |
| K 轴序 `(k_w, c_i^g)`（im2col 隐式展开结构） | 算法 | 无（K 轴序可自由选择，A/B 一致即可） | — | **保留** | 选定 K 序 `(k_w, c_i^g)`——B 的 per-tap 连续 slice 依赖此序（§0.6 R4） |
| group 折叠进 grid z（`bz//groups`/`bz%groups`） | 算法 | 无（纯任务划分） | — | **保留 / 等价替换** | 一维 persistent 中把 `n·groups` 折进 `cid` 解码（§0.6 R1） |
| Tensor Core mma（`T.gemm`） | 优化 | tensor core | 有（Cube 单元 + L0A/L0B/L0C） | **等价替换** | `T.gemm(l1_a, l1_b, l0_c, initC=..., size=[M,K,N])` |
| SMEM tiling（`alloc_shared`） | 优化 | shared memory | 有（L1 512KB / UB 192KB） | **等价替换** | `T.alloc_L1`（Expert）+ slice 形态 `T.copy`（GM→L1 自动 ND2NZ） |
| register 累加（`out_local` fragment fp32） | 优化 | 大寄存器堆 | 有（L0C 128KB fp32 累加器） | **等价替换** | `T.alloc_L0C([block_m, block_n], "float32")` |
| 异步搬运 / `T.Pipelined(num_stages)` | 优化 | cp.async | 有（MTE 异步搬运） | **等价替换** | L1 分块 `T.copy` + 多缓冲（Stage 4 启用 `T.Pipelined` / ping-pong 槽） |
| coalesced 连续访问 | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | slice 形态 `T.copy` 连续搬运 |
| **A 按 K 序 gather + 边界掩码** | 优化 | 无（条件 gather） | 无（Expert Vector 无逐元素条件写/gather） | **重新设计** | → §0.6 R3（weight-layout 前置：`T.copy`+`T.transpose` 链产出 per-group `w_flat`，运行时按 slice 读） |
| **B 逐元素掩码 gather** | 优化 | 无（条件写） | 无（Expert Vector 无逐元素 `T.Parallel`） | **重新设计** | → §0.6 R4（`x_gpad` 连续零填充前置 + per-tap in-bounds slice；stride>1 用 polyphase 物化） |
| 三维 Kernel `T.Kernel(bx,by,bz)` + `threads` | 优化 | CUDA thread 模型 | 无（NPU 仅一维 Kernel + persistent） | **重新设计** | → §0.6 R1（一维 `T.Kernel` + `cid` 解码含 batch/group + persistent 核内串行） |
| 输出 group 溢出保护（`c_out_g % block_m` 分支） | 优化 | 无（纯算法/边界） | 有（slice extents 天然裁剪） | **等价替换** | → §0.6 R2（输出 `T.copy(l0_c[0:tm,0:tn], out[batch_id, oc_base:+tm, ...])`，`tm = min(block_M, c_out_g - by·block_M)` 恰止于 group 末行） |
| epilogue bias 相加（`T.Parallel` 在 fragment 上） | 优化 | 无（纯数学） | 有（Vector 单元 vadd） | **重新设计** | → §0.6 R5（L0C→GM ws → Vector fp32 bias-add（按 `oc_base` 取 group bias）→ cast → GM out，MixCV 双 Scope） |
| swizzle 面板（`T.use_swizzle`） | 优化 | SMEM bank 结构 | 无（NPU 无 bank conflict） | **舍弃** | NZ 分形布局由 `T.copy` ND2NZ 自动处理，手动 swizzle 无意义 |
| `threads` 执行参数 | 优化 | CUDA 线程模型 | 无 | **舍弃** | npuir `T.Kernel(threads=)` 无效果（`TRAP-threads-kwarg-noop`） |
| `num_stages` / `enable_rasterization` 调优参数 | 优化 | CUDA 调度 | 有（分块/流水参数） | **等价替换** | 映射为 L1 分块/多缓冲参数（§5/§6，Stage 4 调优） |

> 每项处置依据：migration-analysis.md §5.3 映射表 + `examples/` Cube 先例（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`）+ docs/Tilelang.language Cube 文档 + 探针实测 + 同族函数 2/3 的 Stage 3 实现。舍弃项（swizzle / threads）理由已写明。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（三维 grid + threads → 一维 Kernel + persistent 核内串行，含 group 轴解码）**

- **源方案**：`T.Kernel(ceildiv(out_l, block_n), ceildiv(c_out_g, block_m), n*groups, threads=threads)` 三维 grid；`bz` 经 `bz//groups`、`bz%groups` 解码 `(batch_id, group_id)`；`T.Parallel` 元素级并行。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (kernel_id, subid)`；`num_kernels = 物理核数`（20，实查见 §5.5），核内 `T.serial(num_local_tasks)` 串行遍历逻辑任务；`GY_g = ceildiv(c_out_g, block_M)`、`GX = ceildiv(out_l, block_N)`、`num_logical = GY_g·GX·n·groups`；`cid = T.min(task_id*num_kernels + kernel_id, num_logical-1)`（idle 核钳位），随后解码 `bz = cid // (GY_g·GX)`、`by = (cid // GX) % GY_g`、`bx = cid % GX`，再由 `bz` 解码 `batch_id = bz // groups`、`group_id = bz % groups`（保持源 `batch-major, group-minor` 迭代序）。矩阵计算交给 Cube 分形（`T.gemm`），尾块裁剪见 R2。
- **语义保持论证**：任务映射（哪个 block 算哪个输出 tile）的变化不改变任何输出值——每 tile 的计算表达式与累加顺序（K 分块 fp32 累加）不变；整数 grid 到一维 persistent 的映射是双射（`cid ↔ (bz,by,bx)`，`bz ↔ (batch_id,group_id)`），钳位仅让 idle 核重算最后一个合法 tile（同输入、同指令序、同输出，Bitwise 一致）。group 之间无数据依赖（输出通道不相交），并行铺开与源一致。fp32 累加与源一致。

**重设计项 R2：非整除/越界尾块重设计（元素级掩码 + group 溢出分支 → 分形 clamp + slice 尾块 + 越界零填充 + 裁剪 store）**

- **源方案**：A/B 装载与 epilogue 均逐元素 `if_then_else` 掩码（M/N/K 尾块 + 输入越界 `il<0 || il>=l_in` 补 0）；输出 `c_out_g % block_m` 分支（整除用 `T.copy`、否则逐元素掩码写以防写进下一 group）。
- **NPU 新算法**：M/N/K 三维尾块用**动态 extents 的 slice 形态** `T.copy` 只搬有效区——A `w_flat[group_id, kw, by·BM : by·BM+tm, kc·BK : kc·BK+kcl] → l1_a[0:tm, 0:kcl]`；B `x_gpad[batch_id, group_id, kc·BK : kc·BK+kcl, off : off+tn] → l1_b[0:kcl, 0:tn]`（越界区由 L1 预清零 + 只搬有效子区间实现零填充）；`T.gemm` 的 `size` 用**分形 clamp** `tmc = T.max(16, ceil16(tm))`、`tnc = T.max(16, ceil16(tn))`（分形 ≥16 约束），K extent = `kcl`（`c_in_g_pad` 为 16 倍数保证 `kcl` 是 16 倍数）；输出用 `T.copy(l0_c[0:tm, 0:tn], out[batch_id, oc_base : oc_base+tm, bx·BN : bx·BN+tn])` **裁剪**写出有效区，其中 `tm = T.min(block_M, c_out_g - by·block_M)`、`oc_base = group_id·c_out_g + by·block_M`——slice extent 恰止于本 group 末行，**天然不越界写到下一 group**（源 `c_out_g % block_m` 分支的 NPU 等价，无需运行时分支）。所有尾块搬运走 slice 形态（`TRAP-T-copy-region-semantics`：禁止 base+size 形态写尾块）。
- **语义保持论证**：有效区元素的计算与源逐元素掩码完全一致（越界读补 0 → 对输出无贡献；分形填充行/列被裁剪不写回）；数学等价（补 0 项对累加无影响）；输出 shape（`L_out` 与源公式一致）与非整除行为向源对齐；group 溢出保护由 slice extent 承载（写 `out[batch_id, oc_base, ...]` 的 `tm` 行，`oc_base+tm ≤ group_id·c_out_g + c_out_g ≤ c_out` 恒成立）。

**重设计项 R3：weight-layout 预处理内移（源 A 逐元素 K 序 gather → kernel 内 Vector per-group `T.transpose` 前置物化 `w_flat`）**

- **源方案**：A tile 由 `T.Parallel(block_m, block_k)` 逐元素 `weight_shared[i,k] = if_then_else((oc_g<c_out_g)&(k_idx<k_total), weight[group_id·c_out_g+oc_g, ci_g, kw], 0)`（`kw=k_idx//c_in_g`、`ci_g=k_idx%c_in_g`）gather 得到——因权重存储序 `(c_i^g, k_w)` 与 GEMM 的 K 序 `(k_w, c_i^g)` 不一致，且需按 group 取片。
- **NPU 新算法**：NPU Expert Vector 段无逐元素条件写（函数 2/3 Stage 3 实测），故把 K 序重排与 group 取片移入 **Vector 前置段**（每 launch 一次）：以 group 为外层、`co` 分块（`bm_w=16`）为内层，`T.copy(weight[g·c_out_g + co0 : +cw, 0:c_in_g, 0:K_W] → wsrc[cw, c_in_g, K_W])`（`cw = min(bm_w, c_out_g - co0)`），`T.transpose(wsrc, wt, permutation=[0,2,1])`（`[cw,c_in_g,K_W]→[cw,K_W,c_in_g]`）、`T.transpose(wt, wdst, permutation=[1,0,2])`（`→[K_W,cw,c_in_g]`，两次二轴交换链，`docs/Tilelang.language/创建操作/T.transpose.md` §2.3；fp16 ✓/bf16 ✓），再按 `kw` 分片 `T.copy(wdst[kw, 0:cw, 0:c_in_g] → w_flat[g, kw, co0:+cw, 0:c_in_g])`。`w_flat` 为 GM workspace，shape `(groups, K_W, c_out_g, c_in_g_pad)`，列 `[c_in_g, c_in_g_pad)` 补零（wrapper `torch.zeros` 分配，恒 0）。主 Cube 循环按 `(group_id, tap kw, co 块 by, 通道块 kc)` 取 A 片，L1 目标偏移 0（`l1_a[0:tm, 0:kcl]`）。
- **语义保持论证**：两次 `T.transpose` 是纯数据重排（`w_flat[g,k_w,c_o^g,c_i^g] = weight[g·C_out^g+c_o^g, c_i^g, k_w]`），与源 gather 逐元素恒等；仅改变 A 操作数的内存布局，不改变 GEMM 的 K 序语义与累加值（K 序仍 `(k_w,c_i^g)`）。边界：`co` 尾块 slice 裁剪（R2）；`c_in_g_pad` 补零列（`w_flat` 预零）与 B 的补零通道行（0）相乘恒为 `0 × 0 = 0`（不依赖未初始化位型）。group 取片由 `w_flat[group_id, ...]` 承载（每 group 一段独立转置，不跨 group 边界，故 `groups>1` 且 `c_out_g % bm_w ≠ 0` 时也正确）。

**重设计项 R4：group 通道段 im2col 重设计（逐元素掩码 gather → 连续 `x_gpad` 前置 + per-tap in-bounds slice；stride>1 polyphase）**

- **源方案**：B tile 由 `T.Parallel(block_k, block_n)` 逐元素 `x[batch_id, group_id·c_in_g + ci_g, il]`（`il = ol·s_W + k_w·d_W - p_L`）gather 得到，越界补 0。
- **NPU 新算法**：
  - **group 通道段零填充前置**：kernel 启动时 Vector 前置段把每个 group 的通道段连续拷入零填充 GM workspace `x_gpad[n, groups, c_in_g_pad, l_pad]`（`x_gpad[bn, g, ci, p_L + i] = x[bn, g·c_in_g + i, i]`，`ci ∈ [0, c_in_g)`、`i ∈ [0, l_in)`；`x_gpad` 由 wrapper 以 `torch.zeros` 分配，左右 L pad、补零通道行 `[c_in_g, c_in_g_pad)` 保持 0；`c_in_g_pad = max(16, ceil16(c_in_g))`、`l_pad = ceil((l_in + p_L + p_R)/16)·16`）。**注意 GM 侧偏移写入**：源连续拷入同形 UB，再写 `x_gpad[bn, g, 0:c_in_g, p_L : p_L+l_in]`（UB 目标列偏移会静默错值，函数 2 Stage 3 实测 `TRAP-copy-ub-offset-stride`）。
  - **stride_l == 1（任意 dilation / padding）**：对 tap `kw`，B 段由**一次 slice 形态 `T.copy`** 从 `x_gpad` 装载：`l1_b[0:kcl, 0:tn] = x_gpad[batch_id, group_id, kc·BK : kc·BK+kcl, off : off+tn]`，`off = bx·BN + kw·d_W`（**恒在界内**：`off+tn-1 ≤ (out_l-1) + (K_W-1)·d_W - 0 < l_pad`）。dilation 仅平移每 tap 起点，padding 只体现为 `x_gpad` 中的零区——均不破坏最内维连续。此形态由探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 实证（per-tap 连续矩形 slice）。
  - **stride_l > 1**：列采样 `x[..., il=ol·s_W+c]` 的最内维步长 = `s_W ≠ 1`，GM→L1 ND2NZ 要求区域最内维连续，故**不能**直接 slice（否则静默平坦误读，CG-2026-0004 同族）。改用 **polyphase 等价分解**：kernel 内 Vector 前置段对每个 `(bn, g)` 用 `T.gather` 物化 `xdec_g[n, groups, c_in_g_pad, l_pad]`，`xdec_g[bn,g,ci,rho·Mdec+m] = x_gpad[bn,g,ci,m·s_W+rho]`（host 预建常量索引，`Mdec = l_pad // s_W`；跨步 gather 无向量指令，`TRAP-C10-strided-gather`——与函数 2 同款绕法；Expert Vector 无逐元素 `T.Parallel`）。则对 tap `kw`，`ol·s_W + kw·d_W - p_L = (ol + q_kw)·s_W + r_kw`，`r_kw = (kw·d_W) mod s_W`、`q_kw = (kw·d_W - r_kw)//s_W`，故 B 段 `l1_b[0:kcl, 0:tn] = xdec_g[batch_id, group_id, kc·BK : kc·BK+kcl, r_kw·Mdec + bx·BN + q_kw : ... + tn]`（三维 slice，最内连续）。stride=1 可视作 `s_W=1` 的退化（`r_kw=0`、`q_kw=kw·d_W`），但为避免无谓的物化，stride=1 直接在 `x_gpad` 上 slice。
- **语义保持论证**：`x_gpad[bn,g,ci,p_L+i] = x[bn, g·c_in_g+ci, i]`（域内逐元素恒等）、越界与补零通道处为 0——与源逐元素掩码 `(k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in)` 逐项一致（补 0 项对累加无贡献）；group 取片由 `x_gpad[batch_id, group_id, ...]` 承载（不跨 group 通道边界）。polyphase 是**精确恒等分解**（`(q,r)` 由 `kw·d_W - p_L` 唯一确定，无近似）；stride=1 时 per-tap slice 与源快/慢路径完全等价（探针实证）。边界：尾块裁剪同 R2。

**重设计项 R5：bias epilogue 重设计（fragment 逐元素相加 → MixCV Vector 后处理，按 group 取 bias）**

- **源方案**：epilogue 在 `out_local` fragment 上逐元素 `out_local[i,j] + cast(bias[group_id*c_out_g + oc_g], fp32)`，再 cast 到输出 dtype、写 `out_shared`。
- **NPU 新算法**：`has_bias=True` 时，Cube 段将 L0C 结果经 fixpipe（**不量化**，fp32→fp32）写入 per-kernel_id、**深度 2 双槽**的 GM workspace `ws[kernel_id, slot, :, :]`（`slot = task_id % 2`）；Vector 段读 `ws` → **先用同 dtype `T.copy` 载入本 tile 的 bias 输入 dtype 向量，再显式 `T.vcast` 到 fp32**（Expert `T.copy` GM↔UB 不跨 dtype，`TRAP-C12-copy-dtype-cast`），`T.vbrc(bias_col, bias_brc)`（`[BM,1]→[BM,BN]` 广播）+ `T.vadd(bias_brc, ub_f32, ub_f32)`（**fp32 域加 bias**）、`T.vcast(ub_f32, ub_f16, round_mode="rint")`、`T.copy(ub_f16 → out)`。bias 向量按 group 取片：`T.copy(bias[oc_base : oc_base+tm], bias_16[0:tm, 0:1])`，`oc_base = group_id·c_out_g + by·block_M`。`has_bias=False` 时**无 Vector bias 段**，L0C 直接 fixpipe（fp32→fp16，`CONST-store-fixpipe-gm-only`）写 `out`。**跨引擎握手（4-flag 双向，对齐 `_ssd_chunk_scan_fwd_kernel.py` 的深度 2 双槽）**：flag `2*slot` = ws 就绪（Cube 写后 `set` / Vector 读前 `wait`）；flag `2*slot+1` = 槽空闲（Vector 消费后 `set` / Cube 写前 `wait`）。**「槽空闲」前导 `set(1)/set(3)` 由消费者（Vector）Scope 置位**（Cube 写前 `wait(2*slot+1)`；同引擎自 set→自 wait 不 retire，函数 1 Stage 3 实测 aicore timeout 507014）。
- **语义保持论证**：bias 在 fp32 域相加后统一 cast（与源 `out_local + cast(bias, fp32)` 的加序与 dtype 一致）；数学恒等 `im2col-GEMM + bias ≡ im2col-GEMM + bias`；bias 索引 `oc_base+i` 与源 `group_id·c_out_g + by·block_m + i` 逐项一致。跨引擎经 GM ws 中转不改变数值（fp32 直存，非量化）。握手正确性：① 读前就绪（Vector 收到 `set(2*slot)` 才读）；② 写前空闲（Cube 收到 `set(2*slot+1)` 才覆盖）；③ 前导两槽空闲由 Vector 置（首两任务无需等待）；④ 每个 flag id 严格 set/wait 交替，深度 2 使 Cube(t) 与 Vector(t-1) 并发且 slot 不冲突。跨引擎传输强制 GM 往返（`CONST-store-fixpipe-gm-only`）——NPU 上把 bias 放 Vector 的固有代价；gqa/ssd 先例均用 GM ws 中继。

> §0.5 全部「重新设计」条目（R1–R5）已重设计。源优化手段（§0.4 #1/#2/#3/#4/#8/#9）意图承接：SMEM tiling→L1 分块 `T.copy`；Tensor core→Cube `T.gemm`+L0C；`T.Pipelined`→L1 多缓冲（Stage 4）；register 累加→L0C fp32 累加器；coalesced→MTE 连续 `T.copy`；输出 group 溢出保护→slice extents 天然裁剪。**#5 group 折叠进 grid z→R1 `cid` 解码**；**#6 A 逐元素 K 序 gather→R3 `w_flat` 前置**；**#7 B 逐元素掩码 gather→R4 `x_gpad`/polyphase 前置**。#10 swizzle / #11 threads 为 NPU 无意义特性（舍弃）。**相对函数 2 一般设计，本算子新增 group 轴解码（R1）+ A 的 per-group 转置前置（R3 天然需要，因源码无 host 重排）+ B 的 per-group 通道段取片（R4）+ bias/out 的 group 偏移（R2/R5）**；其余（y_poly polyphase、尾块、握手）与函数 2 同构。

### 0.7 标杆实现

- 源算子路径（`source_op_path`）：`/mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py`（`_conv1d_group_kernel` builder，L407-560；`GroupConv1dKernel` 类，L929-1051）。NPU 参考副本：`examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py`（`_conv1d_group_kernel`，L429-562）；NPU 分发/包装：`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d.py`。同族已实现参考：`examples/conv1d/_conv1d_kernel/_conv1d_kernel.py`（函数 2 一般 kernel 的 Stage 3 实现，本设计的直接结构基石）、`examples/conv1d/_conv1d_kernel/DESIGN.md`（一般设计）。
- 源仓参考实现 / 测试基准：`torch.nn.functional.conv1d(x, weight, bias, stride, padding, dilation, groups=groups)`，见 `examples/TileOPs/tests/ops/test_conv1d.py`（`test_conv1d_grouped_matches_torch`，`Conv1dWorkload.ref_program`）+ `examples/TileOPs/tileops/workloads/convolution.py`（`full-groups2-k3-fp16` / `full-groups3-coutg24-fp16`）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`F.conv1d`, `groups=groups`），**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

`_conv1d_group_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一，函数 4，`GroupConv1dKernel`，服务 `groups>1` 的分组卷积）；NPU 接口函数同名，`target="npuir"`。

### 1.2 功能描述

**分组**一维卷积的 implicit-GEMM 实现：把**每个 group** 的权重 `[C_out^g, C_in^g, K_W]` 与该 group 的 `im2col(x_g)` 沿 `K = C_in^g·K_W` 收缩，`out[n,c_o,l] = Σ_{c_i^g,k_w} W[c_o,c_i^g,k_w]·x[n, g(c_o)·C_in^g + c_i^g, l·s_W + k_w·d_W - p_L] (+ bias[c_o])`，其中 groups 个 group 沿输出通道切分、彼此独立。相比函数 2 一般实现，本算子把 group 作为**额外并行维度**：grid z 折叠为 `n·groups`、A/B/bias/out 按 `group_id` 选片；K 序、尾块、polyphase、bias 握手与一般实现同构。

### 1.3 数学公式

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{c_i^g=0}^{C_{in}^g-1}\;\sum_{k_w=0}^{K_W-1} W[c_o, c_i^g, k_w]\, x\!\left[n,\; \left\lfloor c_o / C_{out}^g\right\rfloor C_{in}^g + c_i^g,\; l\cdot s_W + k_w\cdot d_W - p_L\right]
$$

$$
L_{out} = \Big\lfloor\frac{L_{in}+p_L+p_R-d_W(K_W-1)-1}{s_W}\Big\rfloor + 1,\qquad C_{in}^g = C_{in}/\text{groups},\quad C_{out}^g = C_{out}/\text{groups}
$$

### 1.4 算法描述

**逐 group 单步 implicit GEMM，group 作为并行维度**：把 group `g` 的权重 `W_g[C_out^g, C_in^g·K_W]`（K 序 `(k_w, c_i^g)`）作为 A、该 group 通道段的 im2col（`[C_in^g, L_out]` 逐 tap 平移窗）作为 B，沿 `K = C_in^g·K_W` 分块累加（fp32），输出 `[C_out^g, L_out]`（逐 batch、逐 group），可选 bias epilogue。本算法与 §1.6.0 调研选定算法一致（**基线 per-group implicit GEMM + group 轴并行即选定主算法**；stride>1 时叠加 polyphase 等价分解）。与源算法结构差异（均来自 §0.6 重设计）：并行组织（三维 grid+threads → 一维 Kernel+persistent 含 group 解码，R1）、非整除/越界尾块（掩码 gather → 分形 clamp+slice 尾块+零填充，R2）、A 的 K 序重排（kernel 内逐元素 gather → `w_flat` 前置 per-group `T.transpose`，R3）、group 通道段 im2col（逐元素 gather → `x_gpad` 连续前置 + per-tap slice / stride>1 polyphase，R4）、bias epilogue（fragment 逐元素 → MixCV Vector，按 group 取 bias，R5）五处不同；GEMM 计算本体（per-group K 序 `(k_w,c_i^g)`、fp32 累加）与源一致。

### 1.5 数据流图

```
（stride=1, 无 bias，纯 Cube）
GM[weight] --T.copy--> UB[wsrc] --T.transpose([0,2,1])--> UB[wt] --T.transpose([1,0,2])--> UB[wdst] --T.copy--> GM[w_flat]   (Vector 前置, 每 group/每 co 块一次)
GM[x] --T.copy--> UB[ub2] --T.copy(GM 侧偏移)--> GM[x_gpad]   (Vector 前置：每 group 通道段连续零填充)
GM[w_flat] --slice T.copy(ND2NZ)--> L1[l1_a] --+
GM[x_gpad]  --per-tap in-bounds slice T.copy(ND2NZ)--> L1[l1_b] --T.gemm(fp32)--> L0C[l0_c] --slice T.copy(fixpipe f32→f16)--> GM[out]

（stride=1, 有 bias，MixCV）
   ... 同上至 L0C[l0_c] --T.copy(fixpipe f32→f32)--> GM[ws_bias]
GM[ws_bias] --T.copy--> UB[ub_f32] --vcast(bias[oc_base:+tm])+vbrc+vadd(fp32)--T.vcast--> UB[ub_f16] --T.copy--> GM[out]

（stride>1：多一段 Vector 前置物化 xdec_g，B 改为从 xdec_g per-tap slice）
GM[x_gpad] --T.gather(host 索引)--> GM[xdec_g] ; GM[xdec_g] --per-tap slice--> L1[l1_b] --T.gemm--> ...
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：调研四问 → 数学等价优化 → 向量化替代 → 向量化轴/布局。本节结论是 §3.1 公式拆解与 §6 循环结构的输入；迁移任务分析对象为 §0.6 重设计后的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**分组** conv1d 的算法族。源算法 / 输入公式只是**基线候选之一**。本算子为**矩阵类（per-group implicit GEMM）+ 窗口类**，按 algorithm-research.md §2 判为**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。信息源：`algorithm-candidates.md`（`ALG-conv` 命中行）、pattern-library `constants.md`、本仓 Cube 先例（gqa/ssd）、同族函数 2 设计与实现（`examples/conv1d/_conv1d_kernel/DESIGN.md`、`examples/conv1d/_conv1d_kernel/_conv1d_kernel.py`）、源算子 §0.3/§0.4、探针 `examples/conv1d/probe/conv1d_geomm_probe.py`。

**R1 等价化简公式候选**（基线必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线**（源算法 / 输入公式） | 逐 group implicit GEMM：`out_g = W_g @ im2col(x_g) + bias_g`（K 序 `(k_w,c_i^g)`，K 分块 fp32 累加；groups 个 group 沿输出通道并行） | — | — | ✅（基线） |
| 1 | **im2col + 显式 GEMM**（`ALG-conv` 命中行） | 逐 group 先把 `im2col_g[C_in^g·K_W, L_out]` 物化到 GM，再做 plain GEMM | 数学等价（im2col 是恒等展开） | 无：物化 `im2col` 引入 `O(C_in^g·K_W·L_out)·groups` 额外 GM 写读，仅 stride>1 的必要绕路 | ✅（评估后弃） |
| 2 | **Winograd**（`ALG-conv` 命中行） | 多抽头小核的 F(m, r) 变换域卷积（逐 group） | **容差内等价**（变换域舍入） | 仅对多抽头（`K_W≥2`）小幅降乘法、增加法；NPU 无实测支撑且变换/逆变换开销大 | ❌（NPU 无实测收益支撑 + 变换开销，见 R4） |
| 3 | **FFT 卷积** | 频域乘 + 逆变换（逐 group） | 容差内等价 | 仅大核长序列有效（`K_W` 大）；本算子 `K_W≤10`，FFT 开销 > 收益 | ❌（小核不适用） |
| 4 | **split-K** | 分段 K + 二段归约 | 容差内等价（fp32 分段归约舍入） | 仅提升并行度，不减少 FLOPs/访存 | ❌（`K = C_in^g·K_W` 不大，group 已提供充足并行度，跨核归并代价 ≥ 收益） |
| 5 | **bias 折叠（augmented-K）** | `[W_g \| bias_g] @ [im2col_g ; 1]` | 数学恒等（`bias·1` 项） | 消 Vector epilogue；但 K 破坏 16 对齐 + 需物化 `x_aug` | ❌（K 对齐代价 + 物化访存 ≫ bias 收益） |
| 6 | **polyphase 分解（stride>1 专用）** | `x[ci, ol·s_W+c] = xdec_g[ci, r_kw, ol+q_kw]`，`c=kw·d_W-p_L`；把列采样移出 DMA 通路 | **精确恒等**（`c` 的 `(q,r)` 分解唯一） | 使 stride>1 的 B 装载回归 per-tap 连续 slice（否则 ND2NZ 非法）；代价 = 一次 `xdec_g` 物化（≈ x 体量，仅 stride>1） | ✅（stride>1 采用） |
| 7 | **block-diagonal dense GEMM** | 物化块对角权重 `W_bd[C_out, C_in, K_W]`（非本 group 块全 0），做**单个** dense GEMM（K=`C_in·K_W`） | 数学等价（跨 group 项权重为 0） | 无：FLOPs = **groups × 基线**（跨 group 的全 0 块白算），且物化 `W_bd`（`O(C_out·C_in·K_W)`，groups 倍权重） | ❌（FLOPs 与权重物化均 groups 倍劣） |
| 8 | **转置 GEMM** | `out_g^T = im2col_g^T @ W_g^T` | 数学等价（转置） | 无计算量收益；输出转置需重排 | ❌（布局问题非公式化简，见 §1.6.3） |
| 9 | **direct 逐输出串行 MAC**（同族 `_conv1d_direct_kernel` / depthwise 形态） | `out_local[i,j] += x·w` 逐 tap 标量（per group） | 数学等价 | 无 Cube 分形复用、无 K 分块累加 → 算力利用率极低 | ❌（Cube/GEMM 亲和性远劣于基线；depthwise 由 `DepthwiseConv1dKernel` 另服务） |

**R2 在线算法**：**结论「无独立在线变体」**——per-group implicit GEMM 天然单遍（每个输出位置 `(c_o,l)` 仅依赖对应 group 的 `K` 段输入与其窗口，无跨位置 running 统计量），源算法 §0.3 亦为单遍 K 分块累加，不存在「两遍扫描 + O(N) 中间缓冲」形态；滑动窗口本身即为流式结构，多个 group 相互独立（可并行、不可在线合并）。结构依据：输出无 cross-tile 依赖、无 running max/sum 重缩放语义（对比 online softmax 类）。候选 split-K（R1 候选 4）是并行化手段而非在线算法。

**R3 复杂度对比**（逐候选四口径；带宽受限算子上访存/遍数是主导项，不得只比 FLOPs；口径标明含什么不含什么。记 `C_out^g=C_out/groups`、`C_in^g=C_in/groups`，基线总 FLOPs = `Σ_g 2·C_out^g·L_out·C_in^g·K_W = 2·C_out·L_out·C_in^g·K_W`）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **基线**（per-group implicit GEMM，K 分块） | `2·C_out·L_out·C_in^g·K_W`（MAC=2；bias 加 `+C_out·L_out`，可忽略） | 输入读 `C_in·L_in·eb` + `x_gpad` 写 `groups·C_in^g_pad·l_pad·eb` + `x_gpad` 读（L2 吸收跨 by/tap 复用）≈ `groups·C_in^g_pad·l_pad·eb` + 权重读 `C_out·C_in^g·K_W·eb` + `w_flat` 写读 `2·groups·C_out^g·C_in^g_pad·K_W·eb` + 输出写 `C_out·L_out·eb`；**无 im2col 物化往返**（无 bias 时）；bias 时 `+ws fp32 往返 2·C_out·L_out·4B` | 1（每输入读一次；K 分块不重复读 x/W） | `O(block_M·block_N)`（L0C fp32 tile）+ `O(block_M·block_K + block_K·block_N)`（L1 tile） | 高（tile 间独立，logical = `GY_g·GX·n·groups`；无跨核归并） |
| 候选 1（im2col 显式） | 同基线 | 同基线 + `im2col` 物化 `2·C_in^g·K_W·L_out·eb`（逐 group 写+读） | 1（但 im2col 需物化后读一遍） | 同基线 + `O(C_in^g·K_W·L_out)` GM | 同基线 |
| 候选 7（block-diagonal dense） | **`groups × 基线`**（`2·C_out·L_out·C_in·K_W`） | 同基线 + `W_bd` 物化 `C_out·C_in·K_W·eb`（groups 倍权重） | 1 | 同基线 | 同基线，但 FLOPs groups 倍劣 |
| 候选 9（direct 逐 MAC） | 同基线（每输出 `2·C_in^g·K_W`，乘加无复用） | 输入重读 `C_in^g·L_out·K_W·eb`（每输出每 tap 重读 x；≫ 基线） | `K_W`（每 tap 扫一遍） | fragment `O(block_M·block_N)` | 同基线，但 Cube 亲和性极低 |
| 候选 4（split-K） | 同基线 | 同基线 + 分段部分和 GM 往返 `O(split·C_out·L_out)` | 1（K 分块；二段归约读回部分和） | `O(split·block)` 部分和 | 并行度↑，**跨核归并代价** |
| 候选 2（Winograd） | 乘法 `~C_out·L_out·C_in^g·K_W/2`，加法显著↑ | 变换/逆变换中间缓冲 | 1 | 变换域 tile | 变换链串行 |
| 候选 6（polyphase，仅 stride>1） | 同基线（**不增加 FLOPs**，只换采样） | 同基线 + `xdec_g` 物化 `≈ C_in·L_in·eb`（写+读，仅 stride>1） | 1（`xdec_g` 物化后单遍） | 同基线 + `O(C_in·L_g·s_W)` GM（≈ x 体量） | 同基线 |

**结论**：基线四口径最优（无额外 GM 往返（无 bias 时）、单遍、O(tile) 驻留；`x_gpad` 前置仅增加 `≈2·C_in·L_in·eb` 的 O(x) 流量，换来 0 掩码的连续 slice）；候选 7 的 FLOPs/权重物化均 groups 倍劣；候选 1/9 在某口径显著劣于基线（im2col 物化访存 / x 重读且无 Cube 复用）；候选 2/3/4 为变换域 / 大核 / 归并手段，对本 shape 无正收益。基线（per-group implicit GEMM + group 轴并行）即选定；stride>1 叠加候选 6（polyphase）以**同 FLOPs** 换回合法的连续 DMA。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰附本仓文档与 examples 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **基线** | MAC 密集 → **Cube**（`T.gemm` + L0A/L0B/L0C），正确落单元；weight-layout 前置 → Vector | L0C `64·128·4 = 32KB ≤ 128KB`；L1 `(64·128 + 128·128)·2 = 48KB/槽 ≤ 512KB`；UB 见 §4.5 | K/N 尾轴分形 ≥16（`ceil16` clamp）；32B 对齐（fp16 尾轴 128·2=256B）✓；`c_in_g_pad` 为 16 倍数 ⇒ `kcl` 16 倍数 | 全部循环边界（`num_local_tasks`、`n_kc`、`kernel_l`）为编译期常量（N/block/`groups` 为 JIT 参数）✓ | K 分块可 `T.Pipelined` 双缓冲；Cube 段与 weight/bias Vector 段可 CV 融合 ✓ | ✅ **采纳** |
| 候选 1（im2col 显式） | 同基线（Cube） | 同基线 + `im2col` GM（非片上，流量↑） | 同基线 | 同基线 | 同基线 | ❌（stride=1 时物化访存 ≫ 收益；stride>1 时由候选 6 更省） |
| 候选 6（polyphase） | 同基线（Cube）；物化 → Vector | `xdec_g` GM（非片上，≈ x 体量） | `T.gather` + 三维 slice，最内 `q` 连续；`Mdec`/`s_W` 编译期常量 ✓ | `Lg`/`s_W` 编译期常量 ✓ | 物化段与主循环可流水 | ✅ **采纳（仅 stride>1）** |
| 候选 7（block-diagonal dense） | 同基线（Cube），但 FLOPs groups 倍 → 算力浪费 | 同基线 + `W_bd`（groups 倍权重，落 L1 时须分块，流量↑） | 同基线 | 同基线 | 同基线 | ❌（FLOPs 与权重物化均 groups 倍劣于基线——计算单元匹配度差） |
| 候选 9（direct 逐 MAC） | 逐元素 MAC → Vector 标量（大段串行标量累加）→ **主导计算落错单元** | fragment 驻留 | `K_W` 循环静态 ✓ | — | 无 Cube 分块复用 | ❌（R4 检查清单第 1 条重度失分：MAC 密集段落 Vector 标量路径） |
| 候选 2（Winograd） | 变换/逆变换 → Vector，乘法 → Cube 但数据布局复杂 | 变换域 tile 驻留 UB | 变换矩阵尺寸约束（`K_W` 小则无收益） | — | 变换链引入额外 pass | ❌（NPU 无实测收益支撑；`K_W≤3` 的 F(m,r) 变换/逆变换向量 op 发射开销（`CONST-vector-launch-overhead` ~0.5µs/op）叠加后净亏——「未文档化假设 + 估算依据」） |
| 候选 3（FFT） | FFT → Vector | 频域缓冲 | — | — | — | ❌（`K_W≤10` 小核，FFT 开销 > 收益） |
| 候选 4（split-K） | 同基线 + 跨核归约（Vector） | 部分和 GM 往返 | — | 需动态 split 边界 | 二段归约需跨核同步 / 两 kernel | ❌（`K` 不大且 group 已提供并行度，归并代价 ≥ 收益） |

**调研结论**：选定算法族 = **基线 per-group implicit GEMM**（A = group `g` 的 `w_flat[g, K_W, C_out^g, C_in^g_pad]`（K 序 `(k_w,c_i^g)`）按 (tap, co 块) 取片，B = 该 group 通道段 im2col（per-tap 连续矩形）同序取片，沿 `K = C_in^g·K_W` 分块 fp32 累加，可选 bias；group 作为并行维度；stride>1 叠加 polyphase）。关键依据：R1 无更优等价化简（基线已是**跨 group 无冗余**的最小 FLOPs 形式；block-diagonal dense 为 groups 倍 FLOPs，Winograd/FFT 对小核无 NPU 实测收益，im2col/转置为布局或物化代价，direct 为逐元素标量）；R2 单遍窗口 K 分块累加即为最优流式（无两遍在线变体）；R3 基线（+stride>1 polyphase）四口径全维度最优；R4 计算单元匹配（MAC→Cube、布局重排/物化→Vector）、片上容量达标（L0C 32KB / L1 48KB/槽）、分形对齐（`ceil16` clamp + `c_in_g_pad` 16 倍数）、静态边界、可流水/CV 融合。与基线的结构差异：**无算法级差异**——调研确认源算法（per-group implicit GEMM）即最优；仅按 §0.6 重设计其并行组织（R1 含 group 解码）、尾块（R2）、weight-layout（R3）、group 通道段 im2col（R4）、bias epilogue（R5）为 NPU 形态。「无更优替代」调研范围：`algorithm-candidates.md`（`ALG-conv` 命中行 im2col+GEMM / implicit GEMM / Winograd，经 `kb_search.py` 检索）、恒等变形自查（R1 候选 1–9 全部评估，含分组特有的 block-diagonal dense GEMM）、pattern-library（`layout.md`、`constants.md`、`traps-runtime.md` 的 Cube 迁移档案）、本仓 Cube 先例（gqa/ssd）、同族函数 2/3 设计与实现、源算子 §0.3/§0.4、探针实测。源优化手段意图承接见 §0.6 末段。本结论是 §1.4 与 §1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2）**（完整调研，流量项 + 发射项 + 容量项均算）：

```
代表 workload: full-groups2-k3-fp16 (N=1, C_in=32, L_in=128, C_out=64, K_W=3, stride=1, pad=1, groups=2, fp16)
        → C_in^g=16, C_out^g=32, out_l=128, k_total=48
另一个代表: full-groups3-coutg24-fp16 (N=1, C_in=48, L_in=128, C_out=72, K_W=3, stride=1, pad=1, groups=3)
        → C_in^g=16, C_out^g=24, out_l=128, k_total=48
估算下界: ≈ 固定启动/发射地板（< 5µs 量级；小 shape 下三段均近零，引用 CONST-mte2-degradation /
  CONST-L1-port-bw / CONST-vector-launch-overhead / CONST-capacity-910B2C；常数版本戳见来源条目）
  流量项: GM = x 读 32·128·2=8KB + x_gpad 写 2·16·144·2≈9KB + x_gpad 读≈9KB + weight 读 64·16·3·2=6KB
        + w_flat 写读 2·2·16·3·2≈12KB + out 写 64·128·2=16KB ≈ 60KB；
        混合读写流量地板按 CONST-mte2-degradation（~31 GB/s/core 档）×20 核 ≈ 0.62TB/s → 60KB/0.62TB/s ≈ 0.1µs（远非下界）
  发射项: Cube 操作数流 = Σ_tasks (block_M·K + K·block_N)·2B；tasks=GY_g·GX·n·groups (groups2: GY_g=1,GX=1,groups=2 → 2)
        = 2·((64+128)·48)·2 ≈ 36.9KB ÷ CONST-L1-port-bw(≈148 GB/s/核) ÷ 20 核 ≈ 0.012µs；Vector 前置 op 数少，×0.5µs/op 亦小
  容量项: L0C tile 32KB ≤ 128KB；L1 单槽 48KB ≤ 512KB；UB 前置/bias 路径见 §4.5（不构成下界）
  注: 分组 workload 的 C_out^g/C_in^g/out_l 均远小于一般 whisper-conv1 量级，三段皆低于固定启动开销（kernel 启动 +
      前置段一次性发射，~ 数 µs 地板）——本算子代表 shape 属**启动/发射受限**画像，非带宽受限
  参考: 估算下界以固定启动地板为主导；Stage 4 以 msprof Task Duration 对账（hardware-cost-model.md §3）；
      大 shape（如 groups 小、C_out^g/out_l 大）则回归流量受限画像（同函数 2）
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线 per-group implicit GEMM + optional bias；stride>1 叠加 polyphase）。逐项评估候选优化，无采纳项则写结论与依据。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | block-diagonal dense GEMM | 逐 group GEMM（groups 次小 GEMM） | 单次 dense GEMM（块对角权重） | 数学等价（跨 group 项权重为 0） | **净亏**：FLOPs 与权重物化均 groups 倍（R3/R4）；不采纳 |
| 2 | Winograd | 直接卷积 | 变换域 F(m,r) 卷积 | 容差内等价（变换域舍入） | **净亏**：`K_W≤3` 小核理论乘法降幅有限，变换/逆变换向量 op 发射开销（`CONST-vector-launch-overhead`）叠加后净亏；不采纳 |
| 3 | bias 折叠（augmented-K） | `per-group im2col-GEMM + bias` | `[W_g \| bias_g] @ [im2col_g ; 1]` | 数学恒等（`bias·1` 项） | **净亏**：K 破坏 16 对齐需 pad，且需物化 `x_aug`（O(K·L_out) 额外 GM 写读）；不采纳 |
| 4 | 转置 GEMM | `out_g = W_g @ im2col_g` | `out_g^T = im2col_g^T @ W_g^T` | 数学等价（转置） | **无收益**：FLOPs/访存同；引入输出重排（PL-1.1 / PL-1.3）；布局问题归 §1.6.3；不采纳 |
| 5 | split-K | 单段 K 累加 | 分段部分和 + 二段归约 | 容差内等价（fp32 分段归约舍入） | **净亏**：K 不大且 group 已提供并行度，部分和 GM 往返 + 跨核归并代价 ≥ 收益；不采纳 |
| 6 | 除法/超越函数降代换、公共子表达式消除、稳定化变形 | — | — | **不适用**：本算子为纯 im2col-GEMM + 加法，无除法、无超越函数、无公共子表达式、无溢出风险需稳定化 | 无 |

**优化结论**：**无优化空间**。本算子为 per-group im2col implicit GEMM（`out_g = W_g @ im2col(x_g) (+bias)`）——无除法的乘倒数、无超越函数降代换、无公共子表达式（每项 M/K/N 元素仅用一次）、无归约稳定化需求（纯线性累加）；五个公式级候选（block-diagonal / Winograd / bias 折叠 / 转置 / split-K）经 §1.6.0 R1/R3/R4 评估均无正收益，不采纳。polyphase（§0.6 R4 / §1.6.0 R1 候选 6）是**精确恒等的结构分解**（算法路线，非公式化简），不改变 FLOPs，已在 §1.6.0 采纳为 stride>1 路线。优化后公式 = 基线公式，供 §3.1 公式拆解使用。**因无采纳的公式级优化项，按 skill Phase 2 第 5 步与 gate `S1-EQUIV-EXEC` 的豁免规则（「无优化空间」显式结论），不生成 `verify_equiv.py`。**

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 本算子主导计算为 Cube（矩阵单元）；非矩阵部分（weight-layout 重排、`x_gpad` 前置、stride>1 polyphase 物化、bias epilogue）在 Vector。§1.6.2 盘点实现方案中全部循环与标量计算点。**关键：Expert Vector 段无逐元素 `T.Parallel`**（函数 2/3 Stage 3 实测），故源码 A/B 的逐元素 gather 一律改为 DMA slice `T.copy`（不可向量化替代的标量点仅限索引/边界）。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | 逐 group `W_g @ im2col_g` 矩阵乘累加 | 源 `T.gemm`（tensor core mma） | **Cube 单元** `T.gemm(l1_a, l1_b, l0_c, initC, size=[tmc, kcl, tnc])` | ✅（Cube，非标量） | — |
| 2 | A 的 K 序重排 + group 取片 `[C_out,C_in^g,K_W]→[groups,K_W,C_out^g,C_in^g]` | 源逐元素 gather `weight[oc,ci_g,kw]` | kernel 内 `T.copy`(GM→UB) + 两次 `T.transpose`(UB, `[0,2,1]` 后 `[1,0,2]`) + `T.copy`(UB→GM)（`docs/Tilelang.language/创建操作/T.transpose.md`；PL-1.1 核内融合转置链） | ✅（UB 向量转置） | — |
| 3 | B 的 group 通道段 im2col（stride=1） | 源逐元素掩码 gather | **slice 形态 `T.copy`** per-tap（GM→L1 自动 ND2NZ，最内维连续；从 `x_gpad[batch_id, group_id, ...]` 取片；探针实测） | ✅（DMA 向量搬运） | — |
| 4 | `x_gpad` 零填充前置（每 group 通道段 + 左右 L pad） | 源逐元素掩码 gather（越界补 0） | `T.copy`(GM→UB) + `T.copy`(UB→GM，偏移在 GM 侧)；pad 区由 host `torch.zeros` 保证 | ✅（DMA 向量搬运） | — |
| 5 | stride>1 polyphase 物化 `xdec_g` | 源逐元素掩码 gather | Vector `T.gather`（host 常量索引）+ `T.copy`（函数 2 同款；`docs/Tilelang.language/索引与元素操作/T.gather.md`） | ✅（向量 gather 指令承载） | 跨步采样本无连续 DMA（`TRAP-C10-strided-gather`，仅 stride>1）；用 `T.gather` 一次物化 |
| 6 | bias 逐元素相加 | 源 `T.Parallel` fragment 逐元素 `+bias[oc]` | `T.vbrc(bias_col, bias_brc)` + `T.vadd(bias_brc, ub_f32, ub_f32)`（Vector 段；`_ssd_chunk_scan_fwd_kernel.py` 同款；`_conv1d_kernel.py` 同族） | ✅ | — |
| 7 | 输出 cast（fp32→fp16/bf16） | 源 `T.cast(out_local, dtype)`（epilogue） | 无 bias：fixpipe `T.copy(l0_c, out)`（`CONST-store-fixpipe-gm-only` 合法量化）；有 bias：`T.vcast(ub_f32, ub_f16, round_mode="rint")` | ✅ | — |
| 8 | K 维分块循环 | 源 `T.Pipelined(ceildiv(k_total, block_k))` | `T.serial(n_kc · kernel_l)`（tile 级调度；Stage 4 可 `T.Pipelined` 多缓冲） | ✅（tile 级循环，非逐元素热点） | — |
| 9 | `cid → (bz, by, bx) → (batch_id, group_id)` 解码 | 源三维 grid 隐式 | 标量 `cid // (GY_g·GX)`、`(cid // GX) % GY_g`、`cid % GX`、`bz // groups`、`bz % groups` | ❌ | **block 级任务映射计算**，每 tile 少量标量，非逐元素热点，无向量等价 API（理由类别①） |
| 10 | 尾块尺寸 `tm/tn/tmc/tnc/kcl/oc_base` | 源 `if_then_else` 逐元素掩码 + `c_out_g%block_m` 分支 | 标量 `T.min` / `T.ceildiv` / `T.max(16, …)` / 乘加 | ❌ | **依赖动态 shape 的边界处理**（尾块裁剪 / 越界 clamp / group 行基址），每 tile 一次，非逐元素热点（理由类别④）；NPU 无「逐元素条件写向量」形态 |
| 11 | `T.min(cid, num_logical-1)` 钳位 | — | 标量 `T.min` | ❌ | block 级索引守卫，每 tile 一次（类别①）；防 `T.ceildiv` 对 idle 核越界 |
| 12 | `off = bx·block_N + kw·d_W`（+ 相位 `r_kw·Mdec`）等地址算术 | 源 `il = ol*stride_l + kw*dilation_l - pad_left` 逐元素 | 标量算术（每 tap 一次） | ❌ | block/tile 级地址计算，非逐元素热点（类别①）；`j` 维由 `T.copy` 的 DMA 承载 |

**向量化结论**：逐元素计算点（A 重排、`x_gpad`/polyphase 构造、bias 相加、输出 cast）已**全部向量化**（`T.transpose` / `T.copy` / `T.gather` / `vbrc` / `vadd` / `vcast`，API 均有本仓 examples 与 docs 佐证）；矩阵乘由 Cube 单元承接（`T.gemm`）；group 通道段 im2col 搬运以 DMA slice `T.copy` 承载。保留 4 类标量点：cid/group 解码、尾块尺寸与 group 基址、cid 钳位、地址算术——均属 block/tile 级任务映射 / 动态边界，非逐元素热点，理由见上表。**与 §6 循环结构一致（§6 不出现逐元素标量循环，也不出现 Vector 逐元素 `T.Parallel`）。矩阵乘不适用「逐元素向量化」——落 Cube 是正确单元选择。**

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> Cube 类算子：fractal/NZ 布局与操作数轴选择是设计变量；weight-layout / im2col 轴序 / group 轴归属 / epilogue 归属独立评估。

**轴质量评分维度**（逐候选填写）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | N 轴（L_out）长是否为向量/分形宽度整数倍 | 代表 shape `L_out=128` 为 128 整数倍；一般 `L_out` 非 128 整数倍时 `ceil16` clamp 后分形对齐 |
| 尾 lane 浪费率 | `(ceil(len/16)·16 − len)/len`（16 为分形粒度） | `L_out=128` → 0%；非整除时 `<1%`（分形口径） |
| 累加链形态 | K 轴为串行累加维；N/M 为分形维 | K=`C_in^g_pad·K_W` 串行累加（tap-major 通道块，Cube 内部）；M/N 分形并行 |
| repack 代价 | 改变布局需要的重排代价 | 选定方案 K 序 `(k_w,c_i^g)`：A 需 weight per-group 重排（两次 `T.transpose` UB，一次性，非 per-tile）；B 无 repack（per-tap slice 直读 `x_gpad`） |
| UB 容量影响 | 布局变化对驻留缓冲尺寸的影响 | 选定方案无额外常驻；weight 前置 `3·bm_w·c_in_g·K_W·eb`、`x_gpad`/`xdec_g` 前置 `ch·l_in·eb`，bias 路径见 §4.5 |

**候选矩阵**（全部枚举，弃选必须给量化理由，禁止「未考虑」）：

| # | 布局方案 | 操作数轴 / K 序 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **K 序 `(k_w,c_i^g)` + group 外层**：A=`w_flat[groups, K_W, C_out^g, C_in^g_pad]`（按 `(group, kw, by, kc)` 取片）、B=`im2col`（per-tap 连续矩形，从 `x_gpad[batch_id, group_id, ...]` 取片）、dst=L0C`[C_out^g, L_out]` | M=C_out^g（分形行）、N=L_out（分形列）、K 序 `(k_w,c_i^g)`（`c_i^g` 最内） | A：weight `[C_out^g,C_in^g,K_W]→[K_W,C_out^g,C_in^g]` 的两次 `T.transpose`（UB 二轴交换链，按 group/co 块一次性 ~µs）；B：**无**（per-tap slice，最内 `L` 连续） | 基线；B 零 repack、探针实证；仅 A 一次性重排 | ✅ |
| 2 | **K 序 `(c_i^g,k_w)`**：A=weight 原生扁平 `[C_out^g, C_in^g·K_W]`（`k_w` 最内）；B=im2col `[k_total, L_out]` | M=C_out^g、N=L_out、K 序 `(c_i^g,k_w)` | A：**无**（原生扁平，`k_w` 最内连续）；B：im2col 需按 `(c_i^g,k_w)` 序物化（band/Toeplitz 非矩形 → 逐元素 gather，O(K·L_out)） | A 省一次转置，但 B 每次 per-tile 都要 gather（O(K·L_out) 标量，stride=1 也触发）→ 净亏 | ❌（B 物化访存 ≫ A 转置的一次性代价） |
| 3 | 转置 GEMM（A/B 角色互换，A=im2col^T） | M=L_out、N=C_out^g | A=im2col^T 需 gather（同 #2 问题）；输出转置 | 无收益、增代价 | ❌（重排代价；`TRAP-transpose-epilogue-poison` 风险） |
| 4 | host permute 进出（weight/x 通道重排，含 group 展平） | — | 两次 host permute（group 维并入） | 违反 ascend-constraints.md §4（host 禁改输入内容）；百 µs 级净亏 | ❌（违规 + 净亏） |
| 5 | B 轴选 `k_w` 内层 + `c_i^g` 外层（per-channel 分组装载） | — | 按 `c_i^g` 分组装载需 `c_in_g·K_W` 次小拷贝 | 指令数爆炸（`CONST-mte2-degradation` 指令受限；小块搬运指令受限） | ❌（per-tap 分组拷贝数最少） |
| 6 | **group 并入 M 轴**（把 groups 个 group 铺在 L0C 的 M 维，单 gemm 覆盖多 group） | M=`groups·C_out^g`=C_out | A 需块对角 `w_flat`（group 间全 0）→ FLOPs groups 倍（同 R1 候选 7） | 白算 groups 倍 MAC，权重物化 groups 倍 | ❌（等价于 block-diagonal dense，FLOPs/物化 groups 倍劣） |

> 迁移任务：GPU 源码的并行轴（`bx`=L、`by`=C_out^g、`bz`=batch×group）是**输入而非结论**；GPU 的隐式 K 序 `(k_w,c_i^g)`（`kw=k_idx//c_in_g`）是**算法输入**（B 的 per-tap 连续 slice 依赖此序），本设计承接为核内 A/B 轴序（候选 1）。GPU 的 group 折叠进 grid z（源码 `bz//groups`/`bz%groups`）承接为 NPU 的 `cid` 解码（候选 1 的并行维，不改变核内布局）。

**布局决策结论**：选定方案 **#1**：核内布局 = NZ 分形（`T.copy` GM→L1 自动 ND2NZ；L0C→GM 自动 fixpipe NZ2ND），操作数轴 = A=`w_flat[groups, K_W, C_out^g, C_in^g_pad]`（K 序 `(k_w,c_i^g)`，`c_i^g` 最内，按 `(group_id, kw, by, kc)` 取片）× B=`im2col`（K×N，最内 `L` 连续，per-tap 连续矩形，从 `x_gpad[batch_id, group_id, ...]` 取片）、dst=L0C`[C_out^g, L_out]`；**group 轴作为并行维（`cid` 解码），不进入核内 tile 布局**。A 的一次性 per-group weight 重排（`T.transpose` perm=[0,2,1] 链）与 B 的 per-tap slice 均落入 §3.3 伪代码与 §6 循环结构（K 循环 tap-major：外层 `(group, by, kc)` 解码、内层 `kw`，A 片 `w_flat[group_id, kw, by·BM:+tm, kc·BK:+kcl]`，`T.gemm` `size=[tmc, kcl, tnc]` 与本节操作数轴一致）；弃选方案（#2–#6）量化理由见候选矩阵。epilogue（bias）向量化轴 = N（L_out）连续轴（UB fp32 tile 行内连续，`vadd` 逐元素），bias 取片 = `bias[oc_base:+tm]`。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Expert（`ascend_mode="Expert"`）。

### 2.2 选型理由

本算子为 Cube（矩阵乘）主导算子，须显式使用 L1（Cube 操作数缓存）与 L0C（fp32 累加器）内存层级，并通过 `T.gemm` 走 Cube 单元。Expert 模式是 npuir 上显式 L1/L0C 与 `T.Scope("Cube")` 的唯一可行路径：Developer 模式的 `T.alloc_shared` / `T.alloc_fragment` 在 persistent + gemm 结构下存在运行时崩溃（`TRAP-DEVMODE-PERSIST-GEMM`），且 `PL-1.16-expert-dualscope-bypass` 记录 Developer 在标量化 / 条件构造 / persistent+gemm 下的结构级阻塞。本算子还需：weight-layout 前置（Vector `T.transpose`，per-group）、group 通道段 `x_gpad` 前置（Vector）、stride>1 polyphase 物化（Vector `T.gather`）、bias epilogue（Vector）——均需 Cube + Vector 双 Scope（MixCV），属 Expert。探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 即以 `TILELANG_ASCEND_MODE=Expert` + `T.Scope("Cube")` 编译运行成功；同族函数 2/3 的 Stage 3 实现（`examples/conv1d/_conv1d_kernel/_conv1d_kernel.py`）即 Expert MixCV。**任务硬性要求编程模式 = Expert**。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_L1`（`l1_a`/`l1_b`）、`T.alloc_L0C`（`l0_c`，fp32）；Vector 段 `T.alloc_ub`（weight 前置 `wsrc`/`wt`/`wdst`、`x_gpad` 前置 `ub2`、polyphase `ubx`/`ubi`/`ubd`、bias 路径 `ub_f32`/`ub_f16`/`bias_16`/`bias_col`/`bias_brc`） |
| 计算方式 | Cube：`T.gemm(l1_a, l1_b, l0_c, initC=..., size=[M,K,N])`；Vector：`T.copy`+`T.transpose`（weight 前置，per-group）、`T.copy`（`x_gpad` 前置）、`T.gather`（polyphase）、`T.vbrc`+`T.vadd`+`T.vcast`（bias） |
| 数据搬运 | `T.copy` slice 形态（GM→L1 自动 ND2NZ；L0C→GM 自动 fixpipe NZ2ND + 可选 fp32→fp16 量化；GM↔UB 通用路径同 dtype） |
| 同步 | **手动同步**（Expert）：`with T.rs("PIPE_MTE2"/"PIPE_C"/"PIPE_FIX"/"PIPE_MTE3")` 包裹对应流水段；跨引擎（Vector 前置→Cube、Cube↔Vector bias）用 `T.sync_block_set/wait` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式（= 基线，无采纳的公式级优化项）；每步映射 Cube / Vector API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `w_flat[g, k_w, c_o^g, c_i^g] = weight[g·C_out^g + c_o^g, c_i^g, k_w]` | weight-layout 前置（Vector，per-group 一次性；K 序 `(k_w,c_i^g)`） |
| 2 | `x_gpad[bn, g, c_i^g, p_L+i] = x[bn, g·C_in^g+c_i^g, i]`（域外/补零通道为 0） | group 通道段零填充前置（Vector，一次性；消除越界掩码） |
| 3 | `xdec_g[bn,g,c_i^g,rho·Mdec+m] = x_gpad[bn,g,c_i^g,m·s_W+rho]`（仅 stride>1） | polyphase 前置（Vector `T.gather`，一次性） |
| 4 | `(bz,by,bx) = decode(cid)`；`batch_id=bz//groups`、`group_id=bz%groups`；`oc_base=group_id·C_out^g+by·block_M` | tile 任务解码（block 级标量，含 group 轴） |
| 5 | `l1_a = w_flat[group_id, kw, by·block_M : +tm, kc·block_K : kc·block_K+kcl]` | A 操作数装载（GM→L1，ND2NZ；group 取片 + tap 取片），L1 目标偏移 0 |
| 6 | `l1_b[kw] = src[batch_id, group_id, kc·block_K : +kcl, off : off+tn]`（stride=1：`src=x_gpad`、`off=bx·block_N+kw·d_W`；stride>1：`src=xdec_g`、`off=r_kw·Mdec+bx·block_N+q_kw`） | B 操作数装载（本 group 通道段 per-tap 连续矩形，ND2NZ） |
| 7 | `l0_c = A_tile @ B_tile`（`initC=(kc+kw==0)`，K 分块 fp32 累加） | Cube GEMM（per group），K 分块累加 |
| 8 | `out_tile = cast_f16(l0_c)` | 无 bias：fixpipe L0C→GM（fp32→fp16 量化），写 `out[batch_id, oc_base:+tm, bx·block_N:+tn]` |
| 9 | `out_tile = cast_f16(l0_c_f32 + bias[oc_base:+tm])` | 有 bias：fixpipe→GM ws(fp32)，Vector `+bias`（按 group 取片，fp32）再 cast，写 out |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | weight-layout 前置 | `T.copy` + `T.transpose`×2 + `T.copy` | `T.copy(weight[g·C_out^g+co0 : +cw, 0:c_in_g, 0:K_W], wsrc)`；`T.transpose(wsrc, wt, permutation=[0,2,1])`；`T.transpose(wt, wdst, permutation=[1,0,2])`；`T.copy(wdst[kw], w_flat[g, kw, co0:+cw, 0:c_in_g])` | Expert |
| 1 | `x_gpad` 前置 | `T.copy`（GM→UB→GM，GM 侧偏移） | `T.copy(x[bn, g·c_in_g+cc0·ch_xp : +cr, 0:l_in], ub2[0:cr, 0:l_in])`；`T.copy(ub2[0:cr, 0:l_in], x_gpad[bn, g, cc0·ch_xp:+cr, p_L:p_L+l_in])`（UB 目标列偏移会静默错值，偏移放 GM 侧；函数 2 Stage 3 实测 `TRAP-copy-ub-offset-stride`） | Expert |
| 2 | polyphase 前置（仅 stride>1） | `T.copy` + `T.gather` | `T.copy(x_gpad[bn,g,cc0·cg:+cw, 0:l_pad], ubx)`；`T.copy(poly_idx[0:cw, 0:l_pad], ubi)`；`T.gather(ubx, ubd, ubi, [cw, l_pad])`；`T.copy(ubd, xdec_g[bn,g,cc0·cg:+cw, 0:l_pad])` | Expert |
| 3 | tile 解码 | `T.serial` / 标量算术 | `cid = T.min(task_id*num_kernels+kernel_id, num_logical-1)`；`bz/by/bx` 除法取模；`batch_id=bz//groups`、`group_id=bz%groups`、`oc_base=group_id*c_out_g+by*block_m` | Expert |
| 4 | A 装载 | `T.copy(w_flat[group_id, kw, by·BM:+tm, kc·BK:+kcl], l1_a[0:tm, 0:kcl])` | slice 形态；GM→L1 自动 ND2NZ；L1 目标偏移 0 | Expert |
| 5 | B 装载 | `T.copy(src[batch_id, group_id, kc·BK:+kcl, off:off+tn], l1_b[0:kcl, 0:tn])` | slice 形态；stride=1 用 `x_gpad`，stride>1 用 `xdec_g`；绕过 `T.load_nd2nz` 跨步误读（`TRAP-load-nd2nz-strided` / CG-2026-0004） | Expert |
| 6 | GEMM | `T.gemm(l1_a, l1_b, l0_c, initC=(kc+kw==0), b_transpose=False, size=[tmc, kcl, tnc])` | `tmc/tnc = T.max(16, ceil16(...))` 分形 clamp；K = `kcl`（16 倍数，`c_in_g_pad` 保证） | Expert |
| 7 | 无 bias 输出 | `T.copy(l0_c[0:tm, 0:tn], out[batch_id, oc_base:oc_base+tm, bx·BN:+tn])` | L0C→GM 自动 fixpipe（fp32→fp16 量化）；slice extent 恰止于 group 末行 | Expert |
| 8 | 有 bias 输出（Cube 段） | `T.sync_block_wait(2*slot+1)`（写前等槽空闲）；`T.copy(l0_c[0:tm,0:tn], ws[kernel_id, slot, 0:tm, 0:tn])`；`T.sync_block_set(2*slot)`（写后置就绪） | `slot = task_id % 2`；fixpipe 不量化（fp32→fp32） | Expert |
| 9 | bias 相加（Vector 段） | `T.sync_block_wait(2*slot)`（读前等就绪）；`T.copy(ws→ub_f32)`；`T.copy(bias[oc_base:+tm]→bias_16)`；`T.vcast(bias_16, bias_col, round_mode="rint")`；`T.vbrc(bias_col, bias_brc)`；`T.vadd(bias_brc, ub_f32, ub_f32)`；`T.vcast(ub_f32, ub_f16, round_mode="rint")`；`T.copy(ub_f16→out[batch_id, oc_base:+tm, bx·BN:+tn])`；`T.sync_block_set(2*slot+1)`（消费后归还） | bias 由输入 dtype 经显式 `T.vcast` 到 fp32（Expert `T.copy` 不跨 dtype）；bias 按 group 取片 | Expert |
| 10 | 同步 | `with T.rs("PIPE_MTE2")` / `T.rs("PIPE_C")` / `T.rs("PIPE_FIX")` / `T.rs("PIPE_MTE3")` | 各流水段包裹 | Expert |

### 3.3 计算伪代码

```python
# Expert 模式；ascend_mode = "Expert"（本算子 = 函数 2 一般 kernel + group 轴解码）
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")
import math
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

BM_W = 16          # weight-layout 前置 co 分块（UB 预算）
SLOTS = 2          # ws 深度 2 双槽
_UB_CAP = 32768


def _ceil16(v):
    return ((v + 15) // 16) * 16


def _pick_chunk(l_pad, elem_bytes, cap=_UB_CAP):
    c = 16
    while c > 1 and c * l_pad * elem_bytes > cap:
        c //= 2
    return c


@tilelang.jit(out_idx=[2], target="npuir")
def _conv1d_group_kernel(n, c_in, l_in, c_out, kernel_l, stride_l,
                         pad_left, pad_right, dilation_l, has_bias,
                         dtype="float16", groups=1, c_in_g=0, c_out_g=0):
    accum = "float32"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    c_in_g = c_in_g if c_in_g > 0 else c_in // groups
    c_out_g = c_out_g if c_out_g > 0 else c_out // groups
    k_total = c_in_g * kernel_l
    c_in_g_pad = max(16, _ceil16(c_in_g))
    lin = 16 * stride_l // math.gcd(16, stride_l) if stride_l != 1 else 16
    l_pad = ((l_in + pad_left + pad_right + lin - 1) // lin) * lin
    mdec = l_pad // stride_l if stride_l != 1 else 0
    ch_xp = _pick_chunk(l_pad, 2)
    cg = _pick_chunk(l_pad, 4)
    num_kernels = NPUUtils.get().get_aicore_num()          # 实查 = 20（§5.5）

    def _factory(block_m, block_n, block_k, num_stages):
        gx = (out_l + block_n - 1) // block_n
        gy = (c_out_g + block_m - 1) // block_m
        num_logical = gy * gx * n * groups                  # group 轴并入逻辑核数
        n_kc = (c_in_g_pad + block_k - 1) // block_k
        n_co = (c_out_g + BM_W - 1) // BM_W
        n_xp = (c_in_g_pad + ch_xp - 1) // ch_xp
        n_cg = (c_in_g_pad + cg - 1) // cg

        @T.macro
        def _prelude(x, weight, w_flat, x_gpad, xdec_g, poly_idx):
            with T.Scope("Vector"):
                wsrc = T.alloc_ub([BM_W, c_in_g, kernel_l], dtype)
                wt = T.alloc_ub([BM_W, kernel_l, c_in_g], dtype)
                wdst = T.alloc_ub([kernel_l, BM_W, c_in_g], dtype)
                ub2 = T.alloc_ub([ch_xp, l_in], dtype)
                # A 前置：每 group 的 weight -> w_flat[groups, K_W, C_out^g, C_in^g_pad]
                for g in T.serial(groups):
                    for co0 in T.serial(n_co):
                        cw = T.min(BM_W, c_out_g - co0 * BM_W)
                        T.copy(weight[g * c_out_g + co0 * BM_W:g * c_out_g + co0 * BM_W + cw,
                                      0:c_in_g, 0:kernel_l],
                               wsrc[0:cw, 0:c_in_g, 0:kernel_l])
                        T.transpose(wsrc, wt, permutation=[0, 2, 1])
                        T.transpose(wt, wdst, permutation=[1, 0, 2])
                        for kw in T.serial(kernel_l):
                            T.copy(wdst[kw, 0:cw, 0:c_in_g],
                                   w_flat[g, kw, co0 * BM_W:co0 * BM_W + cw, 0:c_in_g])
                # B 前置：每 group 通道段连续零填充 x_gpad[bn, g, c_i^g, p_L+i] = x[bn, g·c_in_g+i, i]
                for bn in T.serial(n):
                    for g in T.serial(groups):
                        for cc0 in T.serial(n_xp):
                            cr = T.min(ch_xp, T.max(0, c_in_g - cc0 * ch_xp))
                            if cr > 0:
                                T.copy(x[bn, g * c_in_g + cc0 * ch_xp:
                                         g * c_in_g + cc0 * ch_xp + cr, 0:l_in],
                                       ub2[0:cr, 0:l_in])
                                T.copy(ub2[0:cr, 0:l_in],
                                       x_gpad[bn, g, cc0 * ch_xp:cc0 * ch_xp + cr,
                                              pad_left:pad_left + l_in])
                if stride_l != 1:
                    ubx = T.alloc_ub([cg, l_pad], dtype)
                    ubi = T.alloc_ub([cg, l_pad], "int32")
                    ubd = T.alloc_ub([cg, l_pad], dtype)
                    for bn in T.serial(n):
                        for g in T.serial(groups):
                            for cc0 in T.serial(n_cg):
                                cw = T.min(cg, c_in_g_pad - cc0 * cg)
                                T.copy(x_gpad[bn, g, cc0 * cg:cc0 * cg + cw, 0:l_pad],
                                       ubx[0:cw, 0:l_pad])
                                T.copy(poly_idx[0:cw, 0:l_pad], ubi[0:cw, 0:l_pad])
                                T.gather(ubx, ubd, ubi, [cw, l_pad])
                                T.copy(ubd[0:cw, 0:l_pad],
                                       xdec_g[bn, g, cc0 * cg:cc0 * cg + cw, 0:l_pad])
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(4)                      # 前置就绪（Vector -> Cube）

        @T.macro
        def _cube(out, w_flat, x_gpad, xdec_g, ws, kernel_id, num_local_tasks):
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_m, block_k], dtype)
                l1_b = T.alloc_L1([block_k, block_n], dtype)
                l0_c = T.alloc_L0C([block_m, block_n], accum)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(4)                     # 等前置就绪
                for task_id in T.serial(num_local_tasks):    # persistent 核内串行
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bz = cid // (gy * gx)
                    by = (cid // gx) % gy
                    bx = cid % gx
                    batch_id = bz // groups                  # group 轴解码
                    group_id = bz % groups
                    oc_base = group_id * c_out_g + by * block_m
                    tm = T.min(block_m, c_out_g - by * block_m)
                    tn = T.min(block_n, out_l - bx * block_n)
                    tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                    tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                    for kc in T.serial(n_kc):
                        kcl = T.min(block_k, c_in_g_pad - kc * block_k)
                        for kw in T.serial(kernel_l):
                            with T.rs("PIPE_MTE2"):
                                # A：本 group + tap 的通道块（L1 目标偏移 0）
                                T.copy(w_flat[group_id, kw,
                                              by * block_m:by * block_m + tm,
                                              kc * block_k:kc * block_k + kcl],
                                       l1_a[0:tm, 0:kcl])
                                # B：本 group 通道段的 per-tap 连续矩形
                                if stride_l != 1:
                                    base = (((kw * dilation_l) % stride_l) * mdec
                                            + bx * block_n + (kw * dilation_l) // stride_l)
                                    T.copy(xdec_g[batch_id, group_id,
                                                  kc * block_k:kc * block_k + kcl,
                                                  base:base + tn],
                                           l1_b[0:kcl, 0:tn])
                                else:
                                    off = bx * block_n + kw * dilation_l
                                    T.copy(x_gpad[batch_id, group_id,
                                                  kc * block_k:kc * block_k + kcl,
                                                  off:off + tn],
                                           l1_b[0:kcl, 0:tn])
                            with T.rs("PIPE_C"):
                                T.gemm(l1_a, l1_b, l0_c, initC=(kc + kw == 0),
                                       b_transpose=False, size=[tmc, kcl, tnc])
                    if not has_bias:
                        with T.rs("PIPE_FIX"):
                            T.copy(l0_c[0:tm, 0:tn],
                                   out[batch_id, oc_base:oc_base + tm,
                                       bx * block_n:bx * block_n + tn])
                    else:
                        slot = task_id % SLOTS
                        with T.rs("PIPE_FIX"):
                            T.sync_block_wait(2 * slot + 1)      # 写前等该槽空闲（WAR）
                            T.copy(l0_c[0:tm, 0:tn],
                                   ws[kernel_id, slot, 0:tm, 0:tn])
                            T.sync_block_set(2 * slot)           # 写后置位：该槽 ws 就绪

        @T.macro
        def _bias_epilogue(out, bias, ws, kernel_id, num_local_tasks):
            with T.Scope("Vector"):
                ub_f32 = T.alloc_ub([block_m, block_n], accum)
                ub_f16 = T.alloc_ub([block_m, block_n], dtype)
                bias_16 = T.alloc_ub([block_m, 1], dtype)
                bias_col = T.alloc_ub([block_m, 1], accum)
                bias_brc = T.alloc_ub([block_m, block_n], accum)
                with T.rs("PIPE_MTE2"):                          # 前导：消费者（Vector）置两槽空闲
                    T.sync_block_set(1)                          # slot 0 free
                    T.sync_block_set(3)                          # slot 1 free
                for task_id in T.serial(num_local_tasks):
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bz = cid // (gy * gx); by = (cid // gx) % gy; bx = cid % gx
                    batch_id = bz // groups; group_id = bz % groups
                    oc_base = group_id * c_out_g + by * block_m
                    tm = T.min(block_m, c_out_g - by * block_m)
                    tn = T.min(block_n, out_l - bx * block_n)
                    slot = task_id % SLOTS
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(2 * slot)              # 读前等该槽就绪
                        T.copy(ws[kernel_id, slot, 0:tm, 0:tn], ub_f32[0:tm, 0:tn])
                        T.copy(bias[oc_base:oc_base + tm], bias_16[0:tm, 0:1])
                    T.vcast(bias_16, bias_col, round_mode="rint")   # dtype -> fp32
                    T.vbrc(bias_col, bias_brc)
                    T.vadd(bias_brc, ub_f32, ub_f32)             # fp32 域加 bias
                    T.vcast(ub_f32, ub_f16, round_mode="rint")
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub_f16[0:tm, 0:tn],
                               out[batch_id, oc_base:oc_base + tm,
                                   bx * block_n:bx * block_n + tn])
                        T.sync_block_set(2 * slot + 1)           # 消费后归还该槽（WAR 保护）

        if has_bias:
            @T.prim_func
            def _conv1d_group_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                bias: T.Tensor((c_out,), dtype),
                w_flat: T.Tensor((groups, kernel_l, c_out_g, c_in_g_pad), dtype),
                x_gpad: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
                xdec_g: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
                poly_idx: T.Tensor((cg, l_pad), "int32"),
                ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                    _prelude(x, weight, w_flat, x_gpad, xdec_g, poly_idx)
                    _cube(out, w_flat, x_gpad, xdec_g, ws, kernel_id, num_local_tasks)
                    _bias_epilogue(out, bias, ws, kernel_id, num_local_tasks)
            return _conv1d_group_bias_main

        @T.prim_func
        def _conv1d_group_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            w_flat: T.Tensor((groups, kernel_l, c_out_g, c_in_g_pad), dtype),
            x_gpad: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
            xdec_g: T.Tensor((n, groups, c_in_g_pad, l_pad), dtype),
            poly_idx: T.Tensor((cg, l_pad), "int32"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                _prelude(x, weight, w_flat, x_gpad, xdec_g, poly_idx)
                _cube(out, w_flat, x_gpad, xdec_g, None, kernel_id, num_local_tasks)
        return _conv1d_group_main

    return _factory
```

> 注 1：`T.Tensor((...), dtype)` 用**位置参数**传 dtype（避免 `dtype=` 关键字产生 `name 'in_dtype' is not defined` false alarm，design-template §3.3 约定）。
> 注 2：`w_flat` / `x_gpad` / `xdec_g` / `poly_idx` / `ws` 均为 GM workspace，作为 prim_func 形参由 wrapper 分配传入（对齐函数 2 Stage 3 实现与 `_ssd_chunk_scan_fwd_kernel.py` 将 ws 作为 `T.Tensor` 形参的形态）。`w_flat[groups, K_W, C_out^g, C_in^g_pad]` 的补零列 `[c_in_g, c_in_g_pad)` 与 `x_gpad` 的 pad 区/补零通道行需恒 0 → 由 wrapper 以 `torch.zeros` 分配；`xdec_g`/`ws` 可用 `torch.empty`。
> 注 3：**group 轴解码单一来源**（§3.1/§3.2/§3.3/§6.2/§7.2 一致）：`num_logical = gy·gx·n·groups`；`bz = cid//(gy·gx)`；`batch_id = bz//groups`、`group_id = bz%groups`；`oc_base = group_id·c_out_g + by·block_m`；A/B 按 `group_id` 取片；bias 按 `oc_base` 取片；out 按 `[batch_id, oc_base:+tm]` 写。group 从不进入核内 tile 布局（§1.6.3 候选 6 已弃选）。
> 注 4：**跨引擎握手单一来源**（§3.2/§3.3/§6.2/§7.2 一致）：前置就绪 flag `4`（Vector 写后 `set` / Cube 读前 `wait`）；bias flag `2*slot` = ws 就绪（Cube 写后 `set` / Vector 读前 `wait`）、`2*slot+1` = 槽空闲（Vector 消费后 `set` / Cube 写前 `wait`），**「槽空闲」前导 `set(1)/set(3)` 由消费者（Vector）Scope 置位**（吸收函数 1 Stage 3 修正：同引擎自 set→自 wait 不 retire，`TRAP-sync-self-set-wait`）；`ws` 按 `kernel_id` 索引，flag 为块内同步标志。
> 注 5：`num_kernels = NPUUtils.get().get_aicore_num()` 在 JIT 期求值（编译期常量）；`num_logical < num_kernels` 时 `cid` 钳位防 idle 核越界（gqa 先例）。**K 单一来源约束（防复发）**：`c_in_g_pad` 为 16 倍数、`block_k` 为 16 倍数 ⇒ `kcl = min(block_k, c_in_g_pad - kc·block_k)` 为 16 倍数；`T.gemm` 的 K（`size[1]`）= `kcl`，与本步 A/B 装入宽度逐字相等。
> 注 6：`stride_l != 1` 与 `stride_l == 1` 分支在 JIT 期（Python 编译期）求值——`stride_l` 为编译期常量，故只生成一条路径；`xdec_g`/`poly_idx` 形参在 stride=1 时仍声明（wrapper 传最小占位张量，避免分支改签名，函数 2 同款）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.Scope("Cube")` / `T.Scope("Vector")` | `_gqa_prefill_fwd_kernel.py`；`_ssd_chunk_scan_fwd_kernel.py`；函数 2 Stage 3 实现；cube-skill `references/api-cube.md` | ✅ 已验证（两处 Cube 迁移先例 + 同族实现） |
| `T.alloc_L1` / `T.alloc_L0C` | cube-skill `references/api-cube.md`；`docs/Tilelang.language/内存操作/T.alloc_L1.md`、`docs/Tilelang.language/内存操作/T.alloc_L0C.md` | ✅ 已验证 |
| `T.gemm(a, b, c, size, initC, a_transpose, b_transpose)` | `docs/Tilelang.language/线性代数操作/T.gemm.md`（`size=[a,b,c]`；fp16 输入 dst 须 fp32）；探针 L100；gqa/ssd；函数 2 实现 | ✅ 已验证（文档 + 探针 + 两先例 + 同族实现） |
| `T.copy`（slice 形态 GM→L1 ND2NZ；L0C→GM fixpipe f32→f16；多维 slice 单位维折叠） | `docs/Tilelang.language/内存操作/T.copy.md`（方向表 + 动态 extents 规则 + §2.2.2 单位维折叠）；探针 L88-108；函数 2 实现 | ✅ 已验证（探针 + 同族实现实测） |
| `T.copy`（GM↔UB 通用路径，同 dtype） | 同上（Expert 方向表 GM↔UB √） | ✅ 已验证 |
| `T.transpose(src, dst, permutation)`（UB 级，2 轴交换，可链式） | `docs/Tilelang.language/创建操作/T.transpose.md` §2.3（`permutation` 仅两轴交换，`>2` 轴分解为相邻轴交换链；fp16 ✓/bf16 ✓/fp32 ✓）；PL-1.1-transpose-chain；函数 2 实现 | ✅ 已验证（文档 + pattern-library + 同族实现） |
| `T.gather(src, dst, idx, shape)` | `docs/Tilelang.language/索引与元素操作/T.gather.md`；`_conv1d_kernel.py` stride>1 polyphase 用 | ✅ 已验证（同族实现 + 文档） |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；ssd；gqa | ✅ 已验证 |
| `T.vbrc` / `T.vadd` / `T.vcast` | `docs/Tilelang.language/shape操作/T.vbrc.md`、`docs/Tilelang.language/数学操作/T.vadd.md`、`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；`_ssd_chunk_scan_fwd_kernel.py`；函数 2 实现 | ✅ 已验证（两先例 + 同族实现） |
| `T.rs("PIPE_MTE2"/"PIPE_C"/"PIPE_FIX"/"PIPE_MTE3")` | `_gqa_prefill_fwd_kernel.py`；`_ssd_chunk_scan_fwd_kernel.py`；函数 2 实现 | ✅ 已验证 |
| `T.sync_block_set` / `T.sync_block_wait` | `docs/Tilelang.language/同步管道操作/T.sync_block_wait.md` / `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`；gqa/ssd；函数 2 实现 | ✅ 已验证 |
| `T.serial` / `T.ceildiv` / `T.min` / `T.max` / `T.if_then_else` | gqa；ssd；`docs/Tilelang.language/内存操作/T.copy.md` 示例；函数 2 实现 | ✅ 已验证 |
| `NPUUtils.get().get_aicore_num()` | `tilelang.utils`（`from tilelang.utils import NPUUtils`）；本机实查 = 20（§5.5） | ✅ 已验证（实查） |
| `T.load_nd2nz` | `docs/Tilelang.language/内存操作/T.load_nd2nz.md`；**本设计避开**（`TRAP-load-nd2nz-strided` 跨步区域静默平坦误读；CG-2026-0004） | ⚠️ 存在但不用（改用 slice `T.copy`） |
| `T.Parallel`（逐元素） | 函数 2 Stage 3 实测：Expert Vector 段**不可用**（GM 直写触发 NPU IR opt 崩溃 / UB 中转 output 未生成） | ⚠️ 存在但本算子**不用**（前沿装载全走 slice `T.copy`；无逐元素 Vector 写） |
| `T.store_fixpipe` | `docs/Tilelang.language/内存操作/T.store_fixpipe.md`；`T.copy` L0C→GM 等价自动转换 | ✅ 等价（`T.copy` 替代） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为三维 `T.Kernel(bx, by, bz)`，`bz` = batch×group） | §0.6 R1：一维 `T.Kernel(num_kernels, is_npu=True)` + `cid` 解码 `(bz,by,bx)` + group 轴解码（`bz//groups`/`bz%groups`）+ persistent 核内串行 |
| GPU 专用 API | Yes（`threads`、`T.use_swizzle`、`T.alloc_shared`/`T.alloc_fragment`、`T.Pipelined`/`T.Parallel` 元素级 gather、`compile_flags`） | 全部改写为 NPU Expert Cube API（§0.5/§0.6） |
| GEMM 非整除 | **Yes**（`c_out_g % block_M`、`out_l % block_N`、`c_in_g % block_K` 均可能非整除） | §0.6 R2：分形 clamp `T.max(16, ceil16(tail))` + slice 尾块 `T.copy` + 裁剪 store；`c_in_g_pad` 补零到 16 倍数使 `kcl` 16 倍数 |
| group 溢出（尾块写进下一 group） | **Yes**（源用 `c_out_g % block_m` 分支保护） | §0.6 R2：输出 slice extent `tm = min(block_M, c_out_g - by·block_M)` 恰止于 group 末行，`oc_base = group_id·c_out_g + by·block_M`；无需运行时分支 |
| L0C 容量溢出 | 需核对 | `block_M·block_N·4 = 64·128·4 = 32KB ≤ 128KB`（`CONST-capacity-910B2C`）✓ |
| 分核策略 | **Yes** | §5.5 三要素（逻辑核数计算 / 物理核数 `get_aicore_num()` 实查 / 规模判定与分核方案） |
| `T.load_nd2nz` 跨步误读 | **Yes**（B 装载 `x_gpad[batch_id, group_id, ci0:ci0+BK, off:off+BN]` 行跨步 = `l_pad`） | 改用 **slice 形态 `T.copy`**（文档化、探针实测的绕法；`TRAP-load-nd2nz-strided` / CG-2026-0004） |
| host 禁改输入内容 | **No**（本算子源 `forward` 无 host permute；未涉及） | 无需内移；A 的 K 序重排**本就在 kernel 内**（源为逐元素 gather，NPU 改为 R3 的 `T.transpose` 前置） |
| ND2NZ 要求区域最内维连续 | **Yes**（stride>1 的 im2col 列采样非连续） | §0.6 R4：stride=1 per-tap slice；stride>1 polyphase 物化后 slice |
| Expert Vector 段无逐元素 `T.Parallel` | **Yes**（源码 A/B 均逐元素 gather） | 全部改为 slice `T.copy`（A：R3 前置 + slice；B：R4 前置 + slice）；无逐元素 Vector 写 |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

> 迁移任务的完整差异分析见 §0.5/§0.6，此处只汇总影响 API 选型的关键技术差异。

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | 三维 `T.Kernel(ceildiv(out_l,BN), ceildiv(c_out_g,BM), n*groups, threads=128)` | 一维 `T.Kernel(num_kernels, is_npu=True)` + cid 解码（含 group 轴） | §0.6 R1 |
| GEMM API | `T.gemm(weight_shared, data_shared, out_local)`（fragment 累加，Developer） | `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])`（Expert，L1/L0C） | §0.6 R1（Cube 路径） |
| 内存分配 | `T.alloc_shared` + `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_L1` + `T.alloc_L0C` | §0.5 等价替换 |
| group 归属 | grid z = `n*groups`，`bz//groups`/`bz%groups` 解码 | 一维 `cid` 折叠 `n·groups`，同款解码 | §0.6 R1 |
| 权重布局 | **无 host 重排**；A 逐元素 gather（K 序 ≠ 存储序） | kernel 内 `T.copy`+`T.transpose`×2+`T.copy` → `w_flat[groups,K_W,C_out^g,C_in^g_pad]` | §0.6 R3 |
| im2col（stride=1） | 逐元素掩码 gather | `x_gpad` 连续前置 + per-tap in-bounds slice `T.copy` | §0.6 R4 |
| im2col（stride>1） | 逐元素 `T.Parallel` gather（`il=ol·s_W+kw·d_W-p_L`） | polyphase 物化（`T.gather`）+ per-tap slice | §0.6 R4 |
| 累加器初始化 | `T.clear(out_local)` + 无 `initC` | `initC=(kc+kw==0)`（首 K 块清零） | 语义等价 |
| 流水 | `T.Pipelined(ceil(k_total,block_k), num_stages)` | `T.serial`（Stage 4 可 `T.Pipelined` 多缓冲） | §0.5 等价替换 |
| 元素级并行 | `T.Parallel` A/B 装载 + epilogue | Cube `T.gemm` + Vector `transpose/copy/gather/vbrc/vadd/vcast` | §0.6 R2/R3/R4/R5 |
| 非整除/越界尾块 | 逐元素 `if_then_else` 掩码 + `c_out_g%block_m` 分支 | 分形 clamp + slice 尾块 + 越界零填充 + 裁剪 store | §0.6 R2/R4 |
| 布局技巧 | `T.use_swizzle` + `enable_rasterization` | 无（NZ 自动 + persistent 调度） | §0.5 舍弃 swizzle |
| `threads` | `threads=128` | 无（一维 Kernel） | §0.5 舍弃（`TRAP-threads-kwarg-noop`） |
| 运行时 target | CUDA（`compile_flags`） | `target="npuir"` | §4.7 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/conv1d/_conv1d_kernel/_conv1d_kernel.py` | **最高（同族函数 2 的 Stage 3 实现，本算子 = 其 + group 轴解码）** | Expert Cube + Vector 双 Scope；`T.alloc_L1/L0C` + `T.gemm(size=[...], initC=(kc+kw==0))`；weight-layout 两次 `T.transpose` 前置；`x_pad` 连续前置；stride>1 `T.gather` polyphase；K 分块 per-tap L1 偏移 0；尾块 clamp+裁剪；MixCV bias 深度 2 双槽 4-flag（消费者侧前导）；Expert `T.copy` 不跨 dtype |
| `examples/conv1d/_conv1d_kernel/DESIGN.md` | 高（同族函数 2 设计，本设计的直接来源） | §0.5/§0.6 迁移决策；§1.6 调研四问；§5.5 分核三要素；stride=1/stride>1 分支设计 |
| `examples/conv1d/_conv1d_unit_stride_kernel/DESIGN.md` | 高（同族函数 3 stride=1 特化） | tap-major 通道分块、`x_pad` 前置、L1 偏移 16 对齐、MixCV 握手 |
| `examples/conv1d/probe/conv1d_geomm_probe.py` | **高度相似（同算子族的可行性探针）** | `T.Scope("Cube")` + `T.alloc_L1/L0C` + `T.gemm(size=[...])`；per-tap slice 形态 `T.copy` 承载重叠/跨步 im2col；`target="npuir"` + Expert；golden = `F.conv1d` |
| `examples/conv1d/_conv1d_pointwise_kernel/DESIGN.md` | 高（K_W=1 的同族简化版） | Expert Cube 结构、MixCV bias 双槽 4-flag 握手（含 Stage 3 修正：前导置消费者侧、bias 显式 vcast） |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高（Cube Expert + MixCV + persistent） | 一维 `T.Kernel(NUM_KERNELS)` + `T.serial(num_local_tasks)` + `cid` 钳位；`T.Scope("Cube")`/`T.Scope("Vector")` 双 Scope；`T.rs("PIPE_*")` 手动同步；`T.sync_block_set/wait` 跨引擎握手；分形 clamp `T.max(16, T.ceildiv(...)*16)` |
| `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py` | 高（Cube Expert + Vector epilogue + 深度 2 双槽握手） | `T.gemm(..., initC=(k==0), size=[...])`；L0C→GM ws 中继；`T.vbrc`/`T.vadd`/`T.vcast` Vector 段；尾块 clamp；深度 2 双槽 4-flag（消费者侧前导） |
| `examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py` | 高（同源 builder 的 NPU 参考副本） | `_conv1d_group_kernel` 源 builder 全文（param/group 解码/数据流契约） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL） |
| `weight` | `(c_out, c_in_g, kernel_l)` | 同 `x` | 分组权重（Conv1d 原生布局；A 的 K 序重排在 kernel 内完成，见 §0.6 R3） |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(n, c_out, out_l)` | 同 `x` | 输出（NCL），JIT `out_idx=[2]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `l1_a` | `(block_M, block_K)` | input dtype | L1 | A 操作数（`w_flat[group_id, kw]` tile），ND2NZ |
| `l1_b` | `(block_K, block_N)` | input dtype | L1 | B 操作数（本 group 通道段 im2col tile），ND2NZ |
| `l0_c` | `(block_M, block_N)` | float32 | L0C | fp32 累加器（`T.gemm` dst） |
| `w_flat` | `(groups, kernel_l, c_out_g, c_in_g_pad)` | input dtype | GM | per-group weight-layout 前置产物（K 序 `(k_w,c_i^g)`；补零列恒 0；作为 prim_func 形参传入） |
| `x_gpad` | `(n, groups, c_in_g_pad, l_pad)` | input dtype | GM | per-group 通道段连续零填充（wrapper `torch.zeros` 分配；作为 prim_func 形参传入） |
| `xdec_g` | `(n, groups, c_in_g_pad, l_pad)` | input dtype | GM | polyphase 物化（仅 stride>1；作为 prim_func 形参传入） |
| `poly_idx` | `(cg, l_pad)` | int32 | GM | polyphase gather 常量索引（host 预建；仅 stride>1；作为 prim_func 形参传入） |
| `ws` | `(num_kernels, 2, block_M, block_N)` | float32 | GM | Cube→Vector bias 中继（深度 2 双槽；仅 bias 路径；作为 prim_func 形参传入） |
| `wsrc` | `(BM_W, c_in_g, kernel_l)` | input dtype | UB | weight 前置载入缓冲 |
| `wt` | `(BM_W, kernel_l, c_in_g)` | input dtype | UB | weight 前置第一次转置结果 |
| `wdst` | `(kernel_l, BM_W, c_in_g)` | input dtype | UB | weight 前置第二次转置结果 |
| `ub2` | `(ch_xp, l_in)` | input dtype | UB | `x_gpad` 前置搬运缓冲 |
| `ubx`/`ubi`/`ubd` | `(cg, l_pad)` | dtype/int32/dtype | UB | polyphase 物化缓冲（仅 stride>1） |
| `ub_f32` | `(block_M, block_N)` | float32 | UB | bias 相加缓冲（仅 bias 路径） |
| `ub_f16` | `(block_M, block_N)` | input dtype | UB | cast 后输出缓冲（仅 bias 路径） |
| `bias_16` | `(block_M, 1)` | input dtype | UB | 本 tile 的 bias 输入 dtype 向量（按 `oc_base` 取片；仅 bias 路径） |
| `bias_col` | `(block_M, 1)` | float32 | UB | bias 的 fp32 列向量（仅 bias 路径） |
| `bias_brc` | `(block_M, block_N)` | float32 | UB | bias 广播缓冲（`vbrc`，仅 bias 路径） |

### 4.4 内存搬运路径

```
前置（Vector，一次）:
GM[weight] --T.copy--> UB[wsrc] --T.transpose([0,2,1])--> UB[wt] --T.transpose([1,0,2])--> UB[wdst] --T.copy--> GM[w_flat]   (每 group/每 co 块)
GM[x] --T.copy--> UB[ub2] --T.copy(偏移在 GM 侧)--> GM[x_gpad]   (每 group 通道段；pad 区由 host 预零)
(stride>1) GM[x_gpad] --T.copy+UB-->  UB[ubx] --T.gather(poly_idx)--> UB[ubd] --T.copy--> GM[xdec_g]

主循环（Cube）:
GM[w_flat] --slice T.copy(ND2NZ)--> L1[l1_a] --+
GM[x_gpad] 或 GM[xdec_g] --per-tap slice T.copy(ND2NZ)--> L1[l1_b] --T.gemm(fp32)--> L0C[l0_c]
    --[无 bias] slice T.copy(fixpipe f32→f16)--> GM[out[batch_id, oc_base:+tm, ...]]
    --[有 bias] T.copy(fixpipe f32→f32)--> GM[ws]

bias epilogue（Vector）:
GM[ws] --T.copy--> UB[ub_f32] --vcast(bias[oc_base:+tm])+vbrc+vadd(fp32)--> UB[ub_f32]
    --vcast(rint)--> UB[ub_f16] --T.copy--> GM[out]  （bias_16 由 GM[bias[oc_base:+tm]] 载入）
```

**规则**：GM→L1 自动 ND2NZ（等价 `T.load_nd2nz`；要求区域最内维连续——stride=1 per-tap 矩形满足，stride>1 经 polyphase 后满足）；L0C→GM 自动 fixpipe（f32→f16 量化 / f32→f32 直存）；GM↔UB 通用路径要求同 dtype（Expert `T.copy` 不跨 dtype，`TRAP-C12-copy-dtype-cast`），故 bias 路径中 fp32→fp16 由显式 `T.vcast` 承担，输入 dtype→fp32 由 `T.copy`(同 dtype) + `T.vcast` 两步承担。bias 路径的 `GM[ws]` 为深度 2 双槽（`ws[kernel_id, slot]`，`slot = task_id % 2`）——详见 §7.2（4-flag 双向握手）。**`x_gpad` 前置的 GM 侧偏移**：`T.copy(ub2, x_gpad[bn, g, cc0:+cr, p_L:p_L+l_in])`（UB 目标列偏移会静默错值，函数 2 Stage 3 实测 `TRAP-copy-ub-offset-stride`）。

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

**UB（192KB/AIV，`CONST-capacity-910B2C`）**：

- **weight 前置段**：`wsrc`/`wt`/`wdst` 各 `BM_W·c_in_g·K_W·eb`（`BM_W=16`）。代表 shape（`c_in_g=16`,`K_W=3`,fp16）→ `16·16·3·2 = 1536 B` ×3 = **4.6KB**。**降档 guard**：若 `c_in_g·K_W` 极大（如 `c_in_g=256`,`K_W=10`），三项合计 `16·2560·2·3 ≈ 240KB > 192KB`——降 `BM_W`（16→8，减半）或分段。
- **`x_gpad` 前置段**：`ub2 = ch_xp·l_in·eb`，`ch_xp = _pick_chunk(l_pad, 2)` 限 `ch_xp·l_pad·eb ≤ 32KB`，故 `ub2 ≤ 32KB`。代表 shape（`l_pad=144`）→ `ch_xp=16` → `ub2 = 16·128·2 = 4KB`。
- **polyphase 段（仅 stride>1，与前置段不同时驻留、可复用）**：`ubx`+`ubd` 各 `cg·l_pad·eb` + `ubi` `cg·l_pad·4B`，各 ≤ 32KB → 合计 ≤ 96KB。
- **bias 段（与前置段不同时驻留、可复用）**：`ub_f32`(32KB) + `ub_f16`(16KB) + `bias_brc`(32KB) + `bias_16`(128B) + `bias_col`(256B) ≈ **80KB ≤ 192KB**。

说明：按 `CONST-capacity-910B2C`「显式 alloc 结构实测 ratio ≈0.98–1.12」口径，各段显式分配实际需求 ≈ 显式值 ×0.98–1.12 ≤ 192KB。代表 shape 的 `c_in_g` 通常远小于一般 conv 的 `c_in`，故 UB 预算比函数 2 更宽松；Stage 4 若启用 auto-multi-buffer，须按 `TRAP-UB-multibuffer-inflation` 复算。

**GM 工作区（不计入片上容量）**：`w_flat` `groups·K_W·c_out_g·c_in_g_pad·eb ≈ K_W·c_out·c_in_g_pad·eb`（groups2：3·64·16·2 = 6KB）；`x_gpad`/`xdec_g` `n·groups·c_in_g_pad·l_pad·eb`（groups2：1·2·16·144·2 = 9KB）；`poly_idx` `cg·l_pad·4B`；`ws` `num_kernels·2·block_M·block_N·4 = 20·2·64·128·4 = 1.31MB`。均占设备全局内存，不改变 §4.5 片上预算；wrapper 侧按 JIT shape 分配（`w_flat`/`x_gpad` 用 `torch.zeros`——补零列/补零通道/pad 区恒 0；`xdec_g`/`ws` 用 `torch.empty`）。

### 4.6 动态轴定义

无动态轴。`n / c_in / l_in / c_out / kernel_l / stride_l / pad_left / pad_right / dilation_l / has_bias / groups / c_in_g / c_out_g` 均为 JIT 编译期参数（每 shape 一特化）；`w_flat`/`x_gpad`/`xdec_g`/`ws` shape、`c_in_g_pad`/`l_pad`/`ch_xp`/`cg`、`n_kc`、`n_co`、`n_xp`、尾块尺寸与 `oc_base` 均由编译期常量经 `T.min` 求值。所有循环边界（`num_local_tasks`、`n_kc`、`kernel_l`、`groups`）为编译期常量，满足「核内串行边界静态」。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
```

并以 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")`（或 wrapper `ascend_mode = "Expert"`）声明 Expert 模式（对齐探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 与函数 2 实现）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：混合 Cube + Vector（Cube 主计算 + Vector weight-layout / `x_gpad` 前置 / stride>1 polyphase 物化 + Vector bias epilogue）。

**判定依据**：主体为逐 group 矩阵乘 `w_flat_g @ im2col_g`（MAC 密集）→ Cube 单元；weight-layout 重排、`x_gpad` 前置、polyphase 物化、bias epilogue 在 Vector。判定依据 = `T.gemm` 主导 + Vector 前置/后处理（§0.6 R3/R4/R5）。

### 5.2 Block 划分

```python
block_M, block_N, block_K = 64, 128, 128   # 承接源 default_config 量级
BM_W = 16                                   # weight 前置 co 分块（UB 预算）
c_in_g_pad = max(16, ceil16(c_in_g))        # 单 group 通道补零到 16 倍数（分形对齐 / kcl 16 倍数）
l_pad = ceil((l_in + pad_left + pad_right) / lin) * lin
GX = ceil(out_l / block_N)                  # N 方向（L_out）
GY_g = ceil(c_out_g / block_M)              # M 方向（单 group 的 C_out^g）
num_logical = GY_g * GX * n * groups        # 逻辑核数（含 group 轴）
```

- **选择理由**：`block_M=64, block_N=128, block_K=128` 承接源算子的 tile 量级（源 `GroupConv1dKernel.default_config` 取 `block_m = min{c∈(16,32,64,128) : c ≥ c_out_g}` 以贴合 `c_out_g`、`block_n=128`、`block_k=128`）；本设计基线统一取 64 以与函数 2 对齐、Stage 4 可按 `c_out_g` 扫描更小 `block_M`（16/32）。`block_K=128` 使通道块数少（`c_in_g_pad≤128` 时一块）；L0C tile 32KB 留足余量；L1 单槽 48KB、双缓冲 96KB ≪ 512KB。**group 轴不进入 tile 布局，仅作为并行维**（§1.6.3 候选 6 弃选）。

### 5.3 约束分析

- **对齐约束**：K/N 尾轴分形 ≥16（`ceil16` clamp）；`block_N=128` fp16 尾轴 128·2=256B（32B 对齐 ✓）；`block_K=128` fp16 256B ✓；`c_in_g_pad` 为 16 倍数 ⇒ `kcl` 为 16 倍数（无 K 尾块零填充缺口）。
- **L0C 容量**：`64·128·4 = 32KB ≤ 128KB` ✓（`CONST-capacity-910B2C`）。
- **L1 容量**：单槽 48KB、双缓冲 96KB ≤ 512KB ✓。
- **UB 容量**：代表 shape 的 weight 前置段 ≈ 4.6KB、bias 段 80KB、polyphase 段 ≤96KB（≤192KB）✓（§4.5）。
- **分形约束**：`T.gemm` M/N 分形 ≥16——`tmc/tnc = T.max(16, ceil16(tail))`；K = `kcl`（16 倍数）。
- **group 溢出约束**：输出 slice extent `tm` 相对 `c_out_g` 计算（非 `c_out`），`oc_base = group_id·c_out_g + by·block_M` ⇒ 永不写进下一 group。

### 5.4 注意事项

- **非整除尾块**：`c_out_g % block_M`、`out_l % block_N` 两处非整除由 §0.6 R2 处理——`T.copy` slice 只搬有效区、`T.gemm` `size=[tmc, kcl, tnc]`、输出用 slice `T.copy` 裁剪写出。禁止 base+size 形态写尾块（`TRAP-T-copy-region-semantics`）。
- **越界输入零填充**：由 `x_gpad` 连续前置承载（`x_gpad` host 预零 + 域内拷入），stride=1 时 per-tap slice 的 `off+tn-1 ≤ (out_l-1)+(K_W-1)·d_W < l_pad` 恒在界内，无需逐元素掩码；stride>1 经 polyphase 后同样恒在界内。
- **`c_in_g` 非 16 倍数**：`c_in_g_pad = max(16, ceil16(c_in_g))` 补零通道行（B=0）与 `w_flat` 补零列（A=0）相乘恒 0，对输出无贡献；`c_in_g` 可 `<16`（如每 group 极少通道），`c_in_g_pad=16` 保证分形与 `kcl` 合法性。
- **极小 shape**（如 `c_out_g < block_M` 或 `out_l < block_N`）：`ceil16` clamp 到 16 保证分形合法，多余填充行/列被裁剪不写回。
- **batch/group 折叠**：`num_logical = GY_g·GX·n·groups`，`n·groups` 折叠进一维 `cid`（`bz = cid // (GY_g·GX)`，再 `batch_id = bz//groups`、`group_id = bz%groups`）。
- **stride>1**：polyphase 物化 `xdec_g`（§0.6 R4），B 装载改从 `xdec_g` 取（相位 `r_kw`/`q_kw`）；stride=1 直接在 `x_gpad` 上 slice（省一次物化）。
- **`groups` 极大 / `c_in_g` 极小**：`DepthwiseConv1dKernel`（`c_in_g=c_out_g=1`）由 `_conv1d_direct_kernel` 另服务，不进本 kernel（`GroupConv1dKernel.refusal` 只要求 `groups>1`；`DepthwiseConv1dKernel` 优先级更高、`preferred_over` 指定）；本 kernel 服务 `c_in_g>1` 的分组调用。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：`num_logical_kernels = ceil(c_out_g / block_M) × ceil(out_l / block_N) × n × groups = GY_g × GX × n × groups`（本算子输出网格 `[n, groups, c_out_g, out_l]`，M=C_out^g、N=L_out、并行维=`n×groups`）。以代表 workload 计：
  - `full-groups2-k3-fp16`（N=1, C_in=32, L_in=128, C_out=64, K_W=3, stride=1, pad=1, groups=2）：`c_out_g=32`、`c_in_g=16`、`out_l=128`；`GY_g=ceil(32/64)=1`、`GX=ceil(128/128)=1`、`num_logical = 1×1×1×2 = 2`。
  - `full-groups3-coutg24-fp16`（N=1, C_in=48, L_in=128, C_out=72, K_W=3, groups=3）：`c_out_g=24`、`out_l=128`；`GY_g=1`、`GX=1`、`num_logical = 1×1×1×3 = 3`。
  - `test_conv1d_grouped_matches_torch`（N=1, C_in=32, C_out=64, K_W=3, groups=2, pad=1）：`num_logical = 1×1×1×2 = 2`。
  - 说明性大 shape（N=1, C_in=128, L_in=4096, C_out=1024, K_W=3, stride=1, pad=1, groups=2）：`c_out_g=512`、`out_l=4096`；`GY_g=ceil(512/64)=8`、`GX=ceil(4096/128)=32`、`num_logical = 8×32×1×2 = 512`。

- **② 物理核数依据（实查）**：`NPUUtils.get().get_aicore_num()` **实查值 = 20**（2026-10-14 本机实查；设备 Ascend 910B3，与 `CONST-aicore-910B2C` 记录的 24 不同——以实查为准，禁止文档假设/经验值替代）。**本算子为 Cube（含混合），直接使用返回值 = 20，不翻倍**（纯 Vector 才翻倍；`core-split-strategy.md` §1 要素②）。查询代码与实际返回值记录如下：

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

  - **`full-groups2-k3-fp16`（num_logical=2）/ `full-groups3-coutg24-fp16`（num_logical=3）/ grouped smoke（num_logical=2）**：均 `num_logical ≤ 物理核数 20`——**结论：逻辑核数 ≤ 物理核数，无需适配**（`num_local_tasks` 对 `kernel_id < num_logical` 为 1、其余为 0；直接映射，无串行超发，无负载不均）。
  - **说明性大 shape（num_logical=512）**：`512 = 20 × 25 + 12`——**极大规模档**：固定启动 20 内核（= 物理核数 ×1），每核 `T.serial` 处理 ≈26 个逻辑块，摊薄核启动开销（512 个内核的启动开销远大于 20 个）；核内串行边界 `num_local_tasks` 为编译期常量（静态）✓；`cid` 钳位保证 idle/多余任务重算最后一个合法 tile（bitwise 一致）。
  - 规模判定：本算子逻辑核数随 shape 横跨「无需适配」（代表 shape）到「极大规模」（大 shape）档；统一 persistent 结构下内核总数 = 物理核数（20），无核超发、无负载不均，核内串行边界静态。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 矩阵乘由 Cube 单元承接、weight-layout/`x_gpad`/polyphase/bias 由 Vector 承接——本表循环仅 block 级 / tile 级调度循环与 K 分块累加；逐元素计算点除 `x_gpad` 前置（DMA 向量搬运）外均已向量化，无逐元素标量循环（§1.6.2）。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_kernels, is_npu=True) as (kernel_id, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| group 轴 | 标量解码 | `batch_id = bz//groups`、`group_id = bz%groups`（每 tile 一次） | group 作为并行维折叠进 `cid`（非逐元素） |
| K 通道方向 | 通道分块累加 | `T.serial(n_kc)`（`kc`） | 通道块，`T.gemm` 累加（tile 级循环，非逐元素） |
| K tap 方向 | 核内串行 | `T.serial(kernel_l)`（`kw`，与 `kc` 两层嵌套） | per-tap slice 装载（tile 级，非逐元素） |
| weight 前置 | Vector 串行 + 搬运 | `T.serial(groups)` × `T.serial(n_co)` × `T.serial(kernel_l)` + `T.transpose` | 一次性 layout 重排（非每 tile） |
| `x_gpad` 前置 | Vector 串行 + 搬运 | `T.serial(n)` × `T.serial(groups)` × `T.serial(n_xp)` + `T.copy` | 一次性零填充（非每 tile） |
| polyphase 物化 | Vector 串行 + gather | `T.serial(n)` × `T.serial(groups)` × `T.serial(n_cg)` + `T.gather`（仅 stride>1） | 逐块 gather 物化（非逐元素标量） |
| 矩阵计算 | Cube | `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])` | MAC 密集 → Cube 分形（§1.6.3 操作数轴） |
| epilogue（bias） | Vector | `T.vbrc` + `T.vadd` + `T.vcast` | 逐元素加 bias 全部向量化（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
    with T.Scope("Vector"):                               # 阶段 A：一次性前置
        for g in serial(groups):                          #   weight per-group 转置 -> w_flat
            for co0 in serial(ceil(c_out_g / BM_W)):
                T.copy(weight -> wsrc); T.transpose(wsrc, wt, [0,2,1]); T.transpose(wt, wdst, [1,0,2])
                for kw in serial(kernel_l): T.copy(wdst[kw] -> w_flat[g, kw, ...])
        for bn in serial(n):                              #   x_gpad 零填充（每 group 通道段）
            for g in serial(groups):
                for cc0 in serial(n_xp):
                    T.copy(x -> ub2); T.copy(ub2 -> x_gpad[bn, g, ..., pad_left:+l_in])
        if stride_l != 1:                                 #   polyphase 物化
            for bn, g, cc0 in serial: T.copy + T.gather -> xdec_g
        with T.rs("PIPE_MTE3"): T.sync_block_set(4)       #   前置就绪
    with T.Scope("Cube"):
        with T.rs("PIPE_MTE2"): T.sync_block_wait(4)      # 等前置就绪
        for task_id in T.serial(num_local_tasks):         # 核内串行（静态边界）
            cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
            bz, by, bx = decode(cid)
            batch_id = bz // groups; group_id = bz % groups       # group 轴解码
            oc_base = group_id * c_out_g + by * block_m
            for kc in T.serial(n_kc):                     # 通道分块累加
                for kw in T.serial(kernel_l):             # per-tap
                    with T.rs("PIPE_MTE2"):
                        load l1_a from w_flat[group_id, kw, ...]
                        load l1_b from x_gpad/xdec_g[batch_id, group_id, ...]
                    with T.rs("PIPE_C"): T.gemm(..., initC=(kc + kw == 0))
            if not has_bias:
                with T.rs("PIPE_FIX"): store l0_c -> out[batch_id, oc_base:+tm, ...]
            else:
                slot = task_id % 2
                with T.rs("PIPE_FIX"):
                    T.sync_block_wait(2*slot + 1)          # 写前等该槽空闲（WAR）
                    store l0_c -> ws[kernel_id, slot]
                    T.sync_block_set(2*slot)               # 写后置位：ws 就绪
    if has_bias:
        with T.Scope("Vector"):
            with T.rs("PIPE_MTE2"): T.sync_block_set(1); T.sync_block_set(3)  # 前导（消费者侧）
            for task_id in T.serial(num_local_tasks):
                slot = task_id % 2
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(2*slot)              # 读前等 ws 就绪
                    load ws, bias[oc_base:+tm] -> ub_f32
                vcast(bias) + vbrc + vadd + vcast
                with T.rs("PIPE_MTE3"):
                    store ub_f16 -> out
                    T.sync_block_set(2*slot + 1)           # 消费后归还槽（WAR 保护）
```

### 6.3 流水线优化

Stage 3 基线用 `T.serial`（正确性优先）；Stage 4 可对 K 分块循环启用 `T.Pipelined(num_stages=2)` 或 L1 ping-pong 双缓冲（两套 `l1_a`/`l1_b` 槽），将 GM→L1 装载隐藏于 Cube 计算窗口（源 `T.Pipelined` 的 NPU 等价，`PL-1.12-task-pipeline-depth2`）。bias 路径的 Cube↔Vector 交接**已在基线采用深度 2 双槽**（`ws[kernel_id, task_id % 2, ...]` + 4-flag 双向握手，§7.2）——Cube(task t) 与 Vector(task t-1) 任务级重叠；Stage 4 可加深（`slots>2`）。weight-layout / `x_gpad` / polyphase 前置段为一次性，其耗时可与首个 tile 的 K 装载流水重叠（依赖 flag 4 的等待点尽量靠后）。

### 6.4 尾块处理

非整除 shape（`c_out_g % block_M`、`out_l % block_N`）：`tm/tn = T.min(block, remain)` 决定有效区；`T.copy` slice 只搬有效区（越界补 0）；`T.gemm` 的 `size=[tmc, kcl, tnc]` 用分形 clamp `tmc/tnc = T.max(16, ceil16(tail))`，K extent = `kcl`（`c_in_g_pad` 为 16 倍数 ⇒ `kcl` 16 倍数，无缺口）；输出用 slice `T.copy(l0_c[0:tm,0:tn], out[batch_id, oc_base:oc_base+tm, ...])` 裁剪写出（填充列不落 GM，且 `oc_base+tm ≤ c_out` ⇒ 不越 group 边界）。输入越界（`il<0 || il>=l_in`）由 `x_gpad` 前置 + per-tap slice 只搬有效子区间实现零填充。所有尾块搬运走 slice 形态（`TRAP-T-copy-region-semantics` / `TRAP-load-nd2nz-strided`）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**手动同步**（Expert）。Cube 段各流水段用 `with T.rs("PIPE_...")` 显式标注；跨引擎交接用 `T.sync_block_set/wait`——① Vector 前置→Cube 的前置就绪 flag（4）；② Cube↔Vector bias 交接的**深度 2 双槽 4-flag 双向握手**（`slot = task_id % 2`；`2*slot` = ws 就绪，`2*slot+1` = 槽空闲；**「槽空闲」前导由消费者（Vector）Scope 置位**；flag 为块内标志，每 block 独立）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| GM→L1 装载（`l1_a`/`l1_b`） | `with T.rs("PIPE_MTE2")` | 标注为 MTE2 搬运段，供编译器编排 MTE2/Cube 依赖（gqa/ssd 同款） |
| `T.gemm` | `with T.rs("PIPE_C")` | 标注 Cube 计算段；等待 L1 操作数就绪、保护 L1 复用 |
| L0C→GM（无 bias 出数） | `with T.rs("PIPE_FIX")` | 标注 fixpipe 段；保证 `l0_c` 写回后才复用/结束 |
| Vector 前置完成（weight-layout / `x_gpad` / polyphase） | `T.sync_block_set(4)`（`with T.rs("PIPE_MTE3")`，Vector 生产者） | 置位「前置就绪」，通知 Cube 可读 `w_flat`/`x_gpad`/`xdec_g`（跨引擎生产者→消费者） |
| Cube 首次读 `w_flat`/`x_gpad`/`xdec_g` 前 | `T.sync_block_wait(4)`（`with T.rs("PIPE_MTE2")`，Cube 消费者） | 等 Vector 前置写完才读（跨引擎就绪握手） |
| Cube 段前导（bias 路径） | `T.sync_block_set(1)` / `T.sync_block_set(3)`（`with T.rs("PIPE_MTE2")`，**Vector 消费者 Scope**） | 置两槽初始空闲；使 Cube 首两个任务（slot 0/1）的写前等待立即通过——**必须在消费者（Vector）引擎置位**（同引擎自 set→自 wait 不 retire；`TRAP-sync-self-set-wait` / 函数 1 Stage 3 修正） |
| Cube 写 `ws` **前**（WAR 保护） | `T.sync_block_wait(2*slot+1)`（`with T.rs("PIPE_FIX")`） | 等 Vector 归还该槽（上一轮用完）后才覆盖 `ws[kernel_id, slot]`——消除数据竞争的关键一环 |
| Cube 写 `ws` **后**（就绪通知） | `T.sync_block_set(2*slot)`（`with T.rs("PIPE_FIX")`） | 置位该槽 ws 就绪，通知 Vector 可读（跨引擎生产者→消费者握手） |
| Vector 读 `ws` **前**（就绪等待） | `T.sync_block_wait(2*slot)`（`with T.rs("PIPE_MTE2")`） | 等 Cube 本任务写完该槽后才读 `ws[kernel_id, slot]` |
| Vector `ws` 读取/写出搬运 | `with T.rs("PIPE_MTE2")` / `with T.rs("PIPE_MTE3")` | 标注 GM↔UB 搬运段（读 ws/bias、写 out） |
| Vector 消费后（槽归还） | `T.sync_block_set(2*slot+1)`（`with T.rs("PIPE_MTE3")`） | 通知 Cube 该槽已空闲，可被下一轮覆盖（WAR 保护的另一半） |

> 协议概要（§3.2/§3.3/§6.2/§7.2 单一来源）：前置就绪 flag `4`（Vector 写后 `set` / Cube 读前 `wait`）；`slot = task_id % 2`；bias 前导由 **Vector** `set(1); set(3)` 置两槽空闲；每任务 Cube `wait(2*slot+1)` → 写 ws → `set(2*slot)`；Vector `wait(2*slot)` → 读 ws + bias 后处理 → 写 out → `set(2*slot+1)`。每个 flag id 在生产/消费间严格 set/wait 交替，深度 2 使 Cube(T) 与 Vector(T-1) 并发而 slot 不冲突（依据 `_ssd_chunk_scan_fwd_kernel.py` 深度 2 双槽 4-flag 先例 + 函数 1/2 Stage 3 消费者侧前导修正）。group 轴只影响 `ws` 的 **内容索引**（out 写回位置），不影响握手拓扑。

### 7.3 pass_configs 配置

Expert 模式默认；bias 路径需保证 Cube/Vector 双 Scope 的交叉流水按显式 flag 编排（`PL-1.16-expert-dualscope-bypass`：Expert 双 Scope + pass_configs 硬边界）。基线不引入额外 `pass_configs`；Stage 4 若启用自动多缓冲，须核对 `auto-multi-buffer` 对 L1/UB 预算的膨胀（`CONST-capacity-910B2C` / `TRAP-UB-multibuffer-inflation`）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据（`out[n,c_o,l] = Σ_{c_i^g,k_w} W[c_o,c_i^g,k_w]·x[n, g(c_o)·C_in^g+c_i^g, l·s_W+k_w·d_W-p_L] + bias[c_o]`，fp32 累加、fp16/bf16 输出），移植源仓参考实现（`torch.nn.functional.conv1d` with `groups=groups`），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_conv1d_group(x, weight, bias=None, stride=1, padding=0, dilation=1,
                        groups=1, dtype="float16"):
    """分组 Conv1d 参考实现（源仓测试基准同款：examples/TileOPs/tests/ops/test_conv1d.py）。

    以 §0.1 语义为依据：weight [C_out, C_in/groups, K_W] -> F.conv1d(groups=groups)；
    在 fp32 域计算（对齐源码 fp32 累加 + fp32 bias 相加），最后 cast 回输入 dtype。
    与 §0.6 的 per-group Cube/persistent/polyphase 实现完全独立。
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    out_f32 = F.conv1d(
        x_cpu.to(torch.float32), w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
    return out_f32.to(x_cpu.dtype)
```

> 说明：`F.conv1d(groups=groups)` 与 builder 契约的 `weight [C_out, C_in_g, K_W]` 一致；golden 在 fp32 域计算，最后 cast 回输入 dtype——与源算子累加域一致（源 `accum_dtype="float"`）。

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓 `conv1d_tolerance` 口径一致；K=`C_in^g·K_W` 累加在 fp32，分形累加顺序与源分块顺序略有差异，落在容差内。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：代表 workload
- grouped smoke `test_conv1d_grouped_matches_torch`（N=1, C_in=32, L_in=128, C_out=64, C_in_g=16, C_out_g=32, K_W=3, **stride=1, pad=1**, groups=2，`float16`；覆盖非整除 M 尾块 `c_out_g=32 < block_M=64`）；
- `full-groups2-k3-fp16`（groups=2）与 `full-groups3-coutg24-fp16`（groups=3，`c_out_g=24` 非 block_M 整除）；
- **stride>1 + padding**（groups>1，如 groups=2, K_W=5, stride=2, pad=2，覆盖 polyphase 路径）；
- **dilation>1**（groups>1，覆盖 slice 起点平移）；
- batch>1（`n>1`，`n·groups` 折叠进 `cid`）。

`randn` 输入 + 随机 `weight`/`bias`，与 `golden_conv1d_group`（§8.1）在 §8.2 容差下 `torch.testing.assert_close`；分别覆盖 `has_bias=False` 与 `has_bias=True` 两条路径。完整分层套件（L1 非整除 / 多 shape / batch>1、L2 边界 size、Boundary 极小/极大 shape 与 `C_in_g/C_out_g/L` 非 16 整除）交由 `tilelang-op-develop` 规划，本设计不枚举。

**L0 必测点**：① 非整除尾块（`C_out_g` 非 `block_M` 整除、`L_out` 非 `block_N` 整除、`C_in_g` 非 16 整除）；② `has_bias` 两条路径一致；③ batch>1 且 groups>1（`bz = batch×group` 解码）；④ **stride>1 + padding**（polyphase 路径与源掩码一致性）；⑤ dilation>1（slice 起点平移）；⑥ **group 边界不溢出**（`c_out_g` 非 block_M 整除时输出 slice 恰止于 group 末行，末 group 之外通道不被覆盖）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| Expert Cube 依赖工具链版本 | Cube Expert（`T.Scope("Cube")` + L1/L0C + `T.gemm` + `T.transpose` + `T.gather`）的全部结论绑定当前工具链（tilelang `0.1.2+417dae99` + CANN，`PL-*`/`CONST-*` 条目版本戳）；工具链升级后须重验（探针已通过为当前证据） |
| 物理核数 20（非 24） | 本机实查 `get_aicore_num()=20`（设备 Ascend 910B3），与 `CONST-aicore-910B2C`（24）不一致——设计以实查为准（§5.5）；`num_kernels` 由 `get_aicore_num()` 在 JIT 期求值，跨设备自适应 |
| `T.gemm` 分形约束 | M/N 分形 ≥16——`tmc/tnc = T.max(16, ceil16(tail))` 保证；K 需 ≥16——`c_in_g_pad = max(16, ceil16(c_in_g))` 保证 `kcl` 16 倍数（`c_in_g` 极小时补零到 16，对输出无影响） |
| group 溢出保护 | 输出 slice extent `tm` 相对 `c_out_g`（非 `c_out`）、`oc_base = group_id·c_out_g + by·block_M`——`oc_base+tm ≤ c_out` 恒成立，等价源 `c_out_g % block_m == 0` 分支（§0.6 R2）。若误用 `c_out` 作 M 维会写进下一 group（Stage 3 实测风险点） |
| 跨引擎传输强制 GM 往返 | bias 路径 Cube→Vector 经 GM ws 中继；weight-layout / `x_gpad` / polyphase 亦经 GM workspace（`CONST-store-fixpipe-gm-only`）——这是 NPU 上把 layout/后处理放 Vector 的固有代价 |
| 跨引擎 ws 握手须双向完整 | 单槽握手若缺 Cube 侧写前等待会产生 WAR 数据竞争；本设计固定**深度 2 双槽 + 4-flag 双向握手**（§3.2/§3.3/§6.2/§7.2 一致）。**且「槽空闲」前导必须置于消费者（Vector）引擎**——同引擎自 set→自 wait 不 retire（`TRAP-sync-self-set-wait`，函数 1 Stage 3 实测 aicore timeout 507014） |
| ND2NZ 区域最内维须连续 | stride>1 的 im2col 列采样非连续，不能直接 GM→L1 slice——必须 polyphase 物化后 slice（§0.6 R4 / CG-2026-0004） |
| polyphase 物化经 `T.gather` | 跨步采样无连续 DMA；`xdec_g` 由 `T.gather` + host 常量索引物化（函数 2 同款）；仅 stride>1 触发，代表 shape 小（`xdec_g ≈ x` 体量）。若出现大 `c_in·l_in` 的 stride>1 workload，须评估该点（Stage 4） |
| weight-layout 前置开销 | `T.transpose` 一次性 `O(c_out·c_in_g·K_W)` 搬运（per-group）；以 `BM_W` 分块控制 UB（§4.5）。若 `c_in_g·K_W` 极大，须降 `BM_W` 或评估其对小 workload 的相对占比（Stage 4） |
| L0C/L1 容量 | L0C 32KB ≤ 128KB、L1 单槽 48KB ≤ 512KB（§4.5）——余量充分；若 Stage 4 增大 `block_M/block_N` 须复算（`block_M·block_N·4 ≤ 128KB`） |
| UB 容量 | 代表 shape 的 weight 前置段 ≈4.6KB、bias 段 80KB、polyphase 段 ≤96KB ≤ 192KB（§4.5）；大 `c_in_g·K_W` 需降 `BM_W` |
| `T.load_nd2nz` 跨步误读 | x 装载为跨步区域（行 stride = `l_pad`）——`TRAP-load-nd2nz-strided` / CG-2026-0004：`load_nd2nz` / base+size `T.copy` 对跨步区域静默平坦误读；本设计一律用 slice 形态 `T.copy`（探针实测绕法） |
| Expert 通用路径不跨 dtype | `T.copy` 在 Expert 通用路径（GM↔UB）要求同 dtype（`TRAP-C12-copy-dtype-cast` / `T.copy` 文档 §2.3）——bias 路径 fp32→fp16 必须显式 `T.vcast`，输入 dtype→fp32 亦须两步 |
| 尾块越界写 | `TRAP-T-copy-region-semantics`（越界写坏相邻 GM / `[N,1]` 越界读）——所有尾块搬运走双显式 slice，禁止 base+size 形态 |
| depthwise 不走本 kernel | `c_in_g=c_out_g=1`（depthwise）由 `DepthwiseConv1dKernel` / `_conv1d_direct_kernel` 另服务；本 kernel 服务 `groups>1` 且 `c_in_g>1` 的调用 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| group 溢出写 | M 尾块用 `c_out` 而非 `c_out_g`、`oc_base` 未加 `group_id·c_out_g` | 输出写进下一 group 的行（静默错值） | `tm = T.min(block_M, c_out_g - by*block_M)`、`oc_base = group_id*c_out_g + by*block_M`（§0.6 R2） |
| group/batch 解码错 | `bz` 未按 `batch_id=bz//groups`、`group_id=bz%groups` 解码（漏 group 轴） | 权重/x/out 取错 group | 严格按 §3.3 注 3 单一来源解码 |
| L0C 溢出 | `block_M·block_N` 过大（>32768） | 编译失败（`cc overflow`） | 维持 `64×128`（32KB）或 `128×256` 上限（128KB） |
| L1 溢出 | `block_M/block_N/block_K` 过大或缓冲槽过多 | 编译失败（`cbuf overflow`） | 维持 `64/128/128`（单槽 48KB）；双缓冲 ≤96KB |
| 尾块 MTE 越界写 | 用 base+size 形态写尾块 | 写坏相邻 GM tensor | 双显式 slice `T.copy(src[0:tm], dst[0:tm])`（§6.4） |
| `T.gemm` 分形非法 | tail < 16 或非 `ceil16` | 编译/运行错误 | `T.max(16, ceil16(tail))` clamp（§0.6 R2） |
| K 分形非法（`c_in_g` 小） | `c_in_g` 未补零到 16 | `kcl < 16` 分形非法 | `c_in_g_pad = max(16, ceil16(c_in_g))`（§0.6 R4） |
| `T.gemm` dst 误设 fp16 | `l0_c` 设 `float16` | 运行卡死（`T.gemm.md` §2.3） | `l0_c = T.alloc_L0C([...], "float32")` 固定 fp32 |
| bias 精度偏差 | bias 在 fp16 域相加（cast 后再加） | 与 golden（fp32 加 bias）偏差偏大 | bias 在 fp32 域相加（`vadd` fp32）后再 `vcast`（§0.6 R5） |
| bias 索引错 | Vector 段用 `by*block_m` 而非 `oc_base = group_id*c_out_g + by*block_m` | 取错 group 的 bias | bias 按 `oc_base` 取片（§0.6 R5） |
| Expert `T.copy` 跨 dtype | GM↔UB 拷贝 fp32↔fp16 | 编译报 `does not support element type casting` | 显式 `T.vcast` 承担转换（§9.1） |
| idle 核越界 | `num_logical < num_kernels` 时 `T.ceildiv` 对负数返回 1 | idle 核运行越界任务 | `cid = T.min(..., num_logical-1)` 钳位（§3.3 注 5，gqa 先例） |
| 跨引擎 ws 数据竞争 | 单槽 ws 且 Cube 写前无 `sync_block_wait`（WAR 未保护） | Vector 未读完即被覆盖 → 随机精度错误 | 深度 2 双槽 + 写前 `wait(2*slot+1)`（§3.3/§6.2/§7.2 一致） |
| 跨引擎握手死锁 | 前导 `set(1)/set(3)` 置于生产者（Cube）侧 | aicore timeout 507014（同引擎自 set→自 wait 不 retire） | 前导置于**消费者（Vector）**Scope（§7.2 / `TRAP-sync-self-set-wait`） |
| stride>1 ND2NZ 误读 | 直接对 strided 列采样做 GM→L1 slice | 静默平坦误读（整块错值） | polyphase 物化后 slice（§0.6 R4 / CG-2026-0004） |
| `x_gpad` UB 偏移错值 | `T.copy(ub2, x_gpad[..., p_L:p_L+l_in])` 误在 UB 侧偏移 | 静默错值 | 偏移放 GM 侧（`x_gpad[bn, g, ..., p_L:p_L+l_in]`；`TRAP-copy-ub-offset-stride`） |
| `T.transpose` 多于两轴 | `permutation` 非「交换两轴」 | 编译失败 | 仅用两轴交换，>2 轴分解为交换链（`T.transpose.md` §2.3） |
| `dtype=` 关键字 false alarm | `T.Tensor(..., dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（§3.3 注 1） |

### 9.3 特殊场景处理

- **非整除 shape**（`C_in_g/C_out_g/L_out` 非 block 整数倍或非 16 整数倍）：分形 clamp + slice 尾块 + 裁剪 store（§6.4）。
- **极小 shape**（`C_in_g < 16`）：K 补零到 16（填充项对输出无影响）。
- **group 尾块**：`c_out_g % block_M ≠ 0`——输出 slice `tm` 相对 `c_out_g` 计算（§0.6 R2），等价源 `c_out_g % block_m == 0` 分支。
- **batch>1 / groups>1**：`n·groups` 折叠进一维 `cid`（`bz = cid // (GY_g·GX)`，再解码 batch/group）；各 (batch, group) tile 独立，无跨 group 依赖。
- **`has_bias=False`**：无 Vector bias 段，无 ws、无 4-flag 握手；仍有 Vector weight-layout 前置段（stride>1 时另有 polyphase、恒有 `x_gpad` 前置段）。
- **`has_bias=True`**：MixCV，fp32 ws 中继（**深度 2 双槽 + 4-flag 双向握手**，§7.2，前导置消费者侧）+ Vector bias epilogue（按 `oc_base` 取 group bias，§0.6 R5）。
- **stride=1**：B 直接从 `x_gpad` per-tap slice（省 `xdec_g` 物化）。
- **stride>1**：先物化 `xdec_g`（§0.6 R4），B 从 `xdec_g` slice（相位）。
- **dilation>1**：slice 起点平移 `kw·d_W`，不改变装载结构（stride=1 直接生效；stride>1 经 polyphase 相位）。
- **`c_in_g < block_K` / `k_total < block_K`**：单 K 块，`kcl = c_in_g_pad`（≤block_K），`kc` 循环一次。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/conv1d/_conv1d_group_kernel/
├── DESIGN.md               # 本设计文档
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `verify_equiv.py` | —（不适用） | §1.6.1 结论为「无优化空间」，按 gate `S1-EQUIV-EXEC` 豁免规则不生成 |
| `_conv1d_group_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_conv1d_group.py` | ⬜ 待实现（Stage 3） | 精度测试（L0/L1/L2/Boundary） |

### 10.3 命名规范

- 项目目录名：`conv1d`；算子目录名：`_conv1d_group_kernel`；接口函数：`_conv1d_group_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一，函数 4）；JIT `target="npuir"`，`out_idx=[2]`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ Stage 1 复盘（RETROSPECTIVE.md）
2. ⬜ Golden 函数（§8.1，`F.conv1d(groups=groups)`）
3. ⬜ stride=1 无 bias 纯 Cube 实现（per-group weight-layout + `x_gpad` 前置 + per-tap slice + `T.gemm`，含 group 解码）→ L0 精度比对
4. ⬜ stride=1 bias 路径（MixCV 双 Scope + ws 中继 + 4-flag 握手，前导置 Vector，bias 按 `oc_base` 取片）→ L0 精度比对
5. ⬜ stride>1 polyphase 路径（groups>1 + stride>1，`xdec_g` 物化）→ L0 精度比对
6. ⬜ 非整除尾块 / group 尾块 / batch>1 / dilation>1 覆盖（L1/L2/Boundary）
7. ⬜ Stage 4 调优（L1 双缓冲、`block_M` 按 `c_out_g` 扫描、bias ws 双槽加深、weight 前置与首个 tile 重叠）
