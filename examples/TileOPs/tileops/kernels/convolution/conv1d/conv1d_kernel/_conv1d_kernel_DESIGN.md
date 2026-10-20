# _conv1d_kernel 算子设计文档（Stage 1 — 迁移设计，Expert Cube）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/convolution/conv1d.py` 的 `_conv1d_kernel` builder / `Conv1dKernel` 类）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），`target="npuir"`，编程模式 **Expert**（`ascend_mode="Expert"`，`T.Scope("Cube")` + `T.alloc_L1` / `T.alloc_L0C` / `T.gemm` / `T.transpose`）。
> 本设计按 skill `tilelang-op-design` 三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出，为本函数 5 个迁移函数（`_conv1d_kernel` 为函数 2）中的第 2 个。
> 结构基石（复用函数 1 `_conv1d_pointwise_kernel`）：Expert Cube implicit GEMM + `T.Scope("Cube")`/`T.Scope("Vector")` 双 Scope + 深度 2 双槽 4-flag MixCV bias 握手；**已吸收其 Stage 3 实测修正**：跨引擎握手的「槽空闲」前导 `set(1)/set(3)` 必须置于**消费者（Vector）Scope**（同引擎自 set→自 wait 不 retire，见 `examples/conv1d/_conv1d_pointwise_kernel/repro/TRAP-sync-self-set-wait.py`），bias 由输入 dtype→fp32 必须显式 `T.vcast`（Expert `T.copy` 不跨 dtype）。
> 可行性探针：`examples/conv1d/probe/conv1d_geomm_probe.py`（Cube conv-as-GEMM 在 npuir 编译并运行成功；per-tap slice 形态 `T.copy` 承载跨步/重叠 im2col，`target="npuir"` + Expert；避开 `T.load_nd2nz` 跨步误读，见 CG-2026-0004）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：一般一维卷积（dense，ungrouped）——对 NCL batch，逐 batch 计算

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{c_i=0}^{C_{in}-1}\sum_{k_w=0}^{K_W-1} W[c_o, c_i, k_w]\cdot x[n, c_i,\; l\cdot s_W + k_w\cdot d_W - p_L], \qquad l \in [0, L_{out})
$$

其中 `L_out = (L_in + p_L + p_R - d_W·(K_W-1) - 1) // s_W + 1`（源码 `out_l` 定义，L47），`s_W`=stride、`d_W`=dilation、`p_L`=pad_left、`p_R`=pad_right、`K_W`=kernel_l；`bias` 可选（`has_bias`）。`k_total = C_in·K_W`。

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | `out[n] = reshape(W_flat·im2col(x[n])) (+ bias)`，其中 `W_flat[C_out, C_in·K_W]`、`im2col[C_in·K_W, L_out]`；`W_flat[c_o, k_w·C_in + c_i] = W[c_o, c_i, k_w]` |
| 规约语义 | 唯一规约轴 = `K = C_in·K_W`（GEMM 的 K 轴，k 序 `(k_w, c_i)`）；**规约/累加顺序** = 源码 `T.gemm` 的 K 分块累加（fp32 累加器 `out_local`）；**无跨 L 的规约**（每个输出位置独立），无 keepdim 语义 |
| dtype 语义 | 输入/权重/输出同 dtype（`float16 \| bfloat16`，`dtype=` JIT 参数）；**中间累加 fp32**（`accum_dtype = "float"`）；**bias 在 fp32 累加域相加后统一 cast 到输出 dtype**（源码 `T.cast(out_local[i,j] + T.cast(bias[oc], accum_dtype), dtype)`，L113/L227） |
| 边界语义 | **越界输入经掩码补 0**（`in_bound = (k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in)`，L98/L212）——`il = ol·s_W + k_w·d_W - p_L` 可 <0 或 ≥l_in；输出非整除尾块（`c_out % block_m`、`out_l % block_n`）由 `if_then_else((oc<c_out)&(ol<out_l), …, 0)` 掩码处理（L111-L121）；无 nan/inf 特殊语义（纯线性累加，nan/inf 按 IEEE 自然传播）；空维度不产生（`L_out>0` 为 manifest shape rule） |
| host 侧语义 | `forward` 做 `weight.permute(0,2,1).contiguous().view(c_out, k_total)`（`[C_out,C_in,K_W] → [C_out, K_W, C_in]` 展平，L870），把隐式 GEMM 的 K 轴物化成 `(k_w, c_i)` 序；属实现手段而非算子契约；NPU 侧必须重设计（见 §0.5/§0.6 R3） |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓测试基准 `torch.nn.functional.conv1d`），**不复刻 NPU 算法**。

### 0.2 源算子输入输出

源 builder `_conv1d_kernel(n, c_in, l_in, c_out, kernel_l, stride_l, pad_left, pad_right, dilation_l, has_bias, dtype)` 的 prim_func 契约（bias 变体多一个 `bias` 入参）：

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL） |
| `weight_flat` | `(c_out, k_total)` | 同 `x` | host `weight.permute(0,2,1).contiguous().view(c_out, k_total)` 得到的 2D 权重（GEMM A 操作数）；`k_total = c_in·kernel_l` |
| `out` | `(n, c_out, out_l)` | 同 `x` | 输出（NCL；`L_out` 由 stride/dilation/padding 决定），`out_idx=[2]` |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

> Op 层输入 `weight` 为 `[C_out, C_in_g, K_W] = [C_out, C_in, K_W]`（groups=1）；builder 收到的是 host 归约/重排后的 2D 版。NPU 侧不改 op 层 I/O（`x`/`weight`/`out`/`bias` shape 与 dtype 与源一致）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | host：`weight_flat = weight.permute(0,2,1).contiguous().view(c_out, k_total)` | `weight [C_out,C_in,K_W]` | `weight_flat [C_out,k_total]`（k 序 `(k_w,c_i)`） | 前置（隐式 GEMM K 轴物化） |
| 2 | 三维 grid：`T.Kernel(ceildiv(out_l, block_n), ceildiv(c_out, block_m), n, threads=threads)` | — | block 编号 `(bx, by, bz)` | 任务划分 |
| 3 | 空间完整判定：`tile_ol_start/end`、`tile_input_start/end`、`tile_spatial_full` | — | 布尔 | 分支选择（快/慢路径） |
| 4 | `T.clear(out_local)` | — | `out_local [block_m, block_n]` fp32 清零 | 累加器初始化 |
| 5 | K 循环 `T.Pipelined(ceildiv(k_total, block_k), num_stages)`：`T.copy(weight_flat[by*block_m, k_iter*block_k], weight_shared)` | `weight_flat` | `weight_shared [block_m, block_k]` | A 操作数装载 |
| 6 | 同循环：B 装载——快路径 `T.copy(x[bz, k_iter*block_k, bx*block_n], data_shared)`（须 `tile_spatial_full`）或慢路径逐元素掩码 gather `kw=k_idx//c_in; ci=k_idx%c_in; il=ol*stride_l+kw*dilation_l-pad_left; data_shared[i,j]=if_then_else((k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in), x[bz,ci,il], 0)` | `x` | `data_shared [block_k, block_n]` | B 操作数装载（im2col 展开）+ 越界零填充 |
| 7 | 同循环：`T.gemm(weight_shared, data_shared, out_local)` | A/B tile | `out_local`（累加） | `sum_{k_w,c_i} W[c_o,c_i,k_w]·x[c_i, l·s_W+k_w·d_W-p_L]` |
| 8 | epilogue `T.Parallel(block_m, block_n)`：`out_shared[i,j] = if_then_else((oc<c_out)&(ol<out_l), cast(out_local[i,j]+cast(bias[oc],fp32), dtype), 0)`（无 bias 时省略加 bias） | `out_local`, `bias` | `out_shared [block_m, block_n]` dtype | bias 相加 + 输出 cast + M/N 尾块掩码 |
| 9 | `T.copy(out_shared, out[bz, by*block_m, bx*block_n])` | `out_shared` | `out` | 输出写回 |

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x], GM[weight_flat] --T.copy--> SMEM[weight_shared/data_shared] --T.gemm--> register fragment[out_local] (fp32)
   --epilogue(Parallel: +bias, cast, mask)--> SMEM[out_shared] --T.copy--> GM[out]
  （B 的 im2col 展开：data_shared[i,j] 由 k_idx→(kw,ci)、ol→il 的跨步/重叠 gather 得到，越界补 0）
```

**循环与并行结构**：grid 三维 `(ceildiv(out_l, block_n), ceildiv(c_out, block_m), n)`；block 内 `threads`（默认 128）；K 维 `T.Pipelined(ceildiv(k_total, block_k), num_stages)` 分块累加；元素级 `T.Parallel(block_k, block_n)`（B 装载）与 `T.Parallel(block_m, block_n)`（epilogue）。默认配置 `block_m=64, block_n=128, block_k=128`（`Conv1dKernel.default_config`，L823）。

**host 侧逻辑**：`forward` 中 `weight.permute(0,2,1).contiguous().view(c_out, k_total)`（真实内容重排——**非** metadata view）；`launch(self, x, weight_flat, bias=bias)`。im2col 在 kernel 内隐式展开（K 轴 `(k_w,c_i)`）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（`weight_shared`/`data_shared`） | 减少 GM 重复访问、放大访存粒度 | K 分块驻留 shared memory | CUDA shared memory | 硬件强相关 |
| 2 | Tensor Core mma（`T.gemm`） | 矩阵算力（MAC 密集） | mma fragment + tensor core 指令 | tensor core（warp 级 mma） | 硬件强相关（NPU 有 Cube 等价） |
| 3 | 异步搬运 / 多级流水（`T.Pipelined(..., num_stages)`） | 访存与计算重叠 | 分块双缓冲流水 | cp.async / 异步拷贝引擎 | 硬件强相关（NPU 有 MTE 等价） |
| 4 | register 累加（`out_local` fragment fp32） | 数据驻留寄存器，消中间写回 | fragment 累加器 | 大寄存器堆 | 硬件强相关（NPU 有 L0C 等价） |
| 5 | **implicit-GEMM K 轴物化**（host `permute(0,2,1).contiguous()`） | 使 K 轴序 `(k_w, c_i)` 与 B 的 per-tap 连续 slice 对齐 | 把权重重排成 `[C_out, K_W, C_in]` 扁平 | 无（host 内容重排） | **不可移植**（host 禁改输入内容，§0.5 判定重设计） |
| 6 | 尾块掩码 gather（`if_then_else((k_idx<k_total)&(ol<out_l)&(il>=0)&(il<l_in), x, 0)`） | 非整除/越界 shape 的正确性 | 元素级条件零填充 | 无（纯算法/边界处理） | 可移植（重设计为分形 clamp + slice 尾块 + 裁剪） |
| 7 | coalesced 连续访问（快路径 `T.copy` 尾轴连续） | 访存带宽 | 连续 slice 读写 | memory coalescing | 硬件强相关（NPU 有 MTE 连续搬运） |
| 8 | swizzle 面板（`T.use_swizzle(CONV_SWIZZLE_PANEL)`） | 消 SMEM bank conflict | 索引重排 | SMEM bank 结构 | 硬件强相关（NPU 不适用） |
| 9 | `threads × npt` 元素级并行 | 每线程处理多元素放大吞吐 | CUDA thread 组织 | CUDA 线程模型 | 硬件强相关（NPU 一维 Kernel） |
| 10 | 空间完整快路径（`tile_spatial_full` 时 `T.copy` 代 gather） | 快路径免逐元素掩码 | 判定窗口整体落在输入内 | 无（纯算法） | 可移植（NPU slice 直接表达） |

> 识别不出来 ≠ 不存在。源码无 warp shuffle、无 online 算法（逐点窗口 + K 分块累加）、无 persistent、无手工 TMA。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `out = im2col-GEMM (+bias)`、fp32 累加、fp16/bf16 输出 | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| GEMM 分块累加（K 轴分块，fp32 累加器） | 算法 | 无（纯数学分块） | — | **保留** | 纯算法层；NPU 沿用 K 分块累加（§6） |
| K 轴序 `(k_w, c_i)`（im2col 隐式展开结构） | 算法 | 无（K 轴序可自由选择，A/B 一致即可） | — | **保留** | 选定 K 序 `(k_w, c_i)`——B 的 per-tap 连续 slice 依赖此序（§0.6 R4） |
| Tensor Core mma（`T.gemm`） | 优化 | tensor core | 有（Cube 单元 + L0A/L0B/L0C） | **等价替换** | `T.gemm(l1_a, l1_b, l0_c, initC=..., size=[M,K,N])` |
| SMEM tiling（`alloc_shared`） | 优化 | shared memory | 有（L1 512KB / UB 192KB） | **等价替换** | `T.alloc_L1`（Expert）+ slice 形态 `T.copy`（GM→L1 自动 ND2NZ） |
| register 累加（`out_local` fragment fp32） | 优化 | 大寄存器堆 | 有（L0C 128KB fp32 累加器） | **等价替换** | `T.alloc_L0C([block_m, block_n], "float32")` |
| 异步搬运 / `T.Pipelined(num_stages)` | 优化 | cp.async | 有（MTE 异步搬运） | **等价替换** | L1 分块 `T.copy` + 多缓冲（Stage 4 启用 `T.Pipelined` / ping-pong 槽） |
| coalesced 连续访问 | 优化 | memory coalescing | 有（MTE 连续搬运） | **等价替换** | slice 形态 `T.copy` 连续搬运 |
| host `weight.permute(0,2,1).contiguous()` | 优化 | 无（host 内容重排） | 无（host 禁改输入内容） | **重新设计** | → §0.6 R3（kernel 内 Vector weight-layout 预处理：`T.transpose` UB 级 2 轴交换，产出 `w_flat[c_out, k_total]`） |
| 三维 Kernel `T.Kernel(bx,by,bz)` + `threads` | 优化 | CUDA thread 模型 | 无（NPU 仅一维 Kernel + persistent） | **重新设计** | → §0.6 R1（一维 `T.Kernel` + cid 解码 + persistent 核内串行） |
| B 装载的逐元素掩码 gather（`if_then_else` + `T.Parallel`） | 优化 | 无（条件写） | 部分（无逐元素条件写向量形态） | **重新设计** | → §0.6 R2/R4（slice 尾块 `T.copy` + 越界零填充 + 分形 clamp） |
| **跨步/重叠 im2col（stride>1 时的列采样）** | 优化 | memory coalescing（GPU 逐元素 gather） | 无（GM→L1 ND2NZ 要求区域最内维连续，stride>1 列采样非连续） | **重新设计** | → §0.6 R4（stride=1：per-tap 连续 slice；stride>1：polyphase 等价分解，把列采样权移出 DMA 通路） |
| epilogue bias 相加（`T.Parallel` 在 fragment 上） | 优化 | 无（纯数学） | 有（Vector 单元 vadd） | **重新设计** | → §0.6 R5（L0C→GM ws → Vector fp32 bias-add → cast → GM out，MixCV 双 Scope） |
| swizzle 面板（`T.use_swizzle`） | 优化 | SMEM bank 结构 | 无（NPU 无 bank conflict） | **舍弃** | NZ 分形布局由 `T.copy` ND2NZ 自动处理，手动 swizzle 无意义 |
| `threads` 执行参数 | 优化 | CUDA 线程模型 | 无 | **舍弃** | npuir `T.Kernel(threads=)` 无效果（`TRAP-threads-kwarg-noop`） |
| `num_stages` / `enable_rasterization` 调优参数 | 优化 | CUDA 调度 | 有（分块/流水参数） | **等价替换** | 映射为 L1 分块/多缓冲参数（§5/§6，Stage 4 调优） |

> 每项处置依据：migration-analysis.md §5.3 映射表 + `examples/` Cube 先例（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`）+ docs/Tilelang.language Cube 文档 + 探针实测。舍弃项（swizzle / threads）理由已写明。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（三维 grid + threads → 一维 Kernel + persistent 核内串行）**

- **源方案**：`T.Kernel(ceildiv(out_l, block_n), ceildiv(c_out, block_m), n, threads=threads)` 三维 grid；`T.Parallel` 元素级并行。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (kernel_id, subid)`；`num_kernels = 物理核数`（20，实查见 §5.5），核内 `T.serial` 串行遍历 `num_local_tasks`，`cid = task_id * num_kernels + kernel_id` 解码 `(bz, by, bx)`（`bz = cid // (GY*GX)`，`by = (cid // GX) % GY`，`bx = cid % GX`）；`cid` 以 `T.min(cid, num_logical-1)` 钳位以避免 idle 核越界（对齐 `_gqa_prefill_fwd_kernel.py` 的 `T.ceildiv` 负数行为注释）。矩阵计算交给 Cube 分形（`T.gemm`），尾块裁剪见 R2。
- **语义保持论证**：任务映射（哪个 block 算哪个输出 tile）的变化不改变任何输出值——每 tile 的计算表达式与累加顺序（K 分块 fp32 累加）不变；整数 grid 到一维 persistent 的映射是双射（`cid ↔ (bz,by,bx)`），钳位仅让 idle 核重算最后一个合法 tile（同输入、同指令序、同输出，Bitwise 一致）。fp32 累加与源一致。

**重设计项 R2：非整除/越界尾块重设计（元素级掩码 gather → 分形 clamp + slice 尾块 + 越界零填充 + 裁剪 store）**

- **源方案**：B 装载与 epilogue 均逐元素 `if_then_else` 掩码（M/N 尾块 + 输入越界 `il<0 || il>=l_in` 补 0）。
- **NPU 新算法**：K/N/M 三维尾块用**动态 extents 的 slice 形态** `T.copy` 只搬有效区——A `w_flat[by·BM : by·BM+tm, k0:k0+tk] → l1_a[0:tm,0:tk]`；B `src[bz, 0:tk, base:base+tn] → l1_b[0:tk,0:tn]`（越界区由 L1 预清零 + 只搬 `[max(0,base), min(l_in, base+tn))` 的有效子区间实现零填充）；`T.gemm` 的 `size` 用**分形 clamp** `tmc = T.max(16, ceil16(tm))`、`tkc = T.max(16, ceil16(tk))`、`tnc = T.max(16, ceil16(tn))`（分形 ≥16 约束）；输出用 `T.copy(l0_c[0:tl, 0:tn], out[..., +tl, +tn])` **裁剪**写出有效区（分形填充列不落 GM）。所有尾块搬运走 slice 形态（`TRAP-T-copy-region-semantics`：禁止 base+size 形态写尾块）。
- **语义保持论证**：有效区元素的计算与源逐元素掩码完全一致（越界读补 0 → 对输出无贡献；填充列被裁剪不写回）；数学等价（补 0 项对累加无影响）；输出 shape（`L_out` 与源公式一致）与非整除行为向源对齐。

**重设计项 R3：weight-layout 预处理内移（host `permute(0,2,1).contiguous()` → kernel 内 Vector `T.transpose` 预处理）**

- **源方案**：host `weight.permute(0,2,1).contiguous().view(c_out, k_total)` 把 `[C_out,C_in,K_W]` 重排为 `[C_out,K_W,C_in]` 扁平（K 序 `(k_w,c_i)`），使 B 的 per-tap slice（按 `c_i` 连续）与 A 的 K 行匹配。
- **NPU 新算法**：NPU host 侧**禁止改动输入张量真实内容**（ascend-constraints.md §4），故把该重排移入 kernel 的 **Vector 前置段**：按 `co` 分块（`bm_w`）`T.copy(weight[co0:co0+cw, 0:c_in, 0:K_W] → wsrc)`，`T.transpose(wsrc, wdst, permutation=[0,2,1])`（UB 级 2 轴交换，`docs/Tilelang.language/创建操作/T.transpose.md` §2.3；fp16 ✓），`T.copy(wdst → w_flat[co0:co0+cw, 0:K_W, 0:c_in])`。`w_flat` 为 GM workspace（`[c_out, K_W, c_in]`，扁平即 K 序 `(k_w,c_i)`），主 Cube 循环按 K 块 slice 读取（最内 `c_i` 连续）。该段每 launch 仅执行一次。
- **语义保持论证**：`T.transpose([0,2,1])` 是纯数据重排（`w_flat[c_o, k_w, c_i] = weight[c_o, c_i, k_w]`），与源 host 重排逐元素恒等；仅改变 A 操作数的内存布局，不改变 GEMM 的 K 序语义与累加值。边界：`co` 尾块 slice 裁剪（R2）。

**重设计项 R4：跨步/重叠/带 dilation im2col 重设计（逐元素跨步 gather → stride=1 per-tap 连续 slice；stride>1 polyphase 等价分解）**

- **源方案**：B tile 由 `T.Parallel(block_k, block_n)` 逐元素 `x[bz, ci, il]`（`il = ol·s_W + kw·d_W - p_L`）gather 得到；`tile_spatial_full` 时退化为一次 `T.copy`。
- **NPU 新算法**：
  - **stride_l == 1（任意 dilation / padding）**：每 tap `k_w` 的 B 段 `l1_b[kw·c_in : (kw+1)·c_in, 0:tn]` 由**一次 slice 形态 `T.copy`** 从 `x` 装载：源区间第二维为 `[l0 + kw·d_W - p_L, …]` 的连续段（最内维连续 ✓），越界（<0 或 ≥l_in）处只搬有效子区间、其余由 L1 预清零。dilation 仅平移每 tap 的起点，padding 只影响起点的负偏移——均不破坏最内连续。此路径由探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 实证（`xp[0:C_IN, bx*BLOCK_N+kw : bx*BLOCK_N+kw+BLOCK_N] → l1_b[kw*C_IN:(kw+1)*C_IN, 0:BLOCK_N]`）。
  - **stride_l > 1**：列采样 `x[ci, ol·s_W + c]` 的最内维步长 = `s_W ≠ 1`，GM→L1 ND2NZ 要求区域最内维连续，故**不能**直接 slice（否则静默平坦误读，CG-2026-0004 同族）。改用 **polyphase 等价分解**：kernel 内 Vector 前置段物化 `y_poly[n, c_in, s_W, Lg]`，`y_poly[bn, ci, r, q] = x[bn, ci, q·s_W + r - p_L]`（越界补 0），`Lg = ceildiv(l_in + p_L + (K_W-1)·d_W, s_W) + 2`；则对 tap `k_w`，`ol·s_W + kw·d_W - p_L = (ol + q_kw)·s_W + r_kw`，`r_kw = (kw·d_W) mod s_W`、`q_kw = (kw·d_W - r_kw) // s_W`，故 B 段 `l1_b[kw·c_in:(kw+1)·c_in, 0:tn] = y_poly[bz, 0:c_in, r_kw, (l0+q_kw):(l0+q_kw)+tn]`（`T.copy` 三/四维 slice，最内 `q` 连续，单位 `r` 维自动折叠——`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2）。stride=1 可视作 `s_W=1` 的退化（`r_kw=0`、`q_kw=kw·d_W`），但为避免无谓的 `y_poly` 物化，stride=1 直接在 `x` 上 slice。
- **语义保持论证**：polyphase 是**精确恒等分解**——`y_poly[bn,ci,r,q] = x[bn,ci,q·s_W+r-p_L]` 按定义精确，输出位置 `ol` 的求和项与源逐元素 gather 逐项对应（`r_kw`/`q_kw` 由 `kw·d_W - p_L` 唯一确定，无近似）；越界补 0 与源掩码 `il>=0 & il<l_in` 一致。stride=1 时 per-tap slice 与源快/慢路径完全等价（探针实证）。边界：尾块裁剪同 R2；`il` 越界零填充在两条路径上均成立。累加顺序不变（仍按 K 序 `(k_w,c_i)` 分块 fp32 累加）。

**重设计项 R5：bias epilogue 重设计（fragment 逐元素相加 → MixCV Vector 后处理）**

- **源方案**：epilogue 在 `out_local` fragment 上逐元素 `out_local[i,j] + cast(bias[oc], fp32)`，再 cast 到输出 dtype、写 `out_shared`。
- **NPU 新算法**：`has_bias=True` 时，Cube 段将 L0C 结果经 fixpipe（**不量化**，fp32→fp32）写入 per-kernel_id、**深度 2 双槽**的 GM workspace `ws[kernel_id, slot, :, :]`（`slot = task_id % 2`）；Vector 段读 `ws` → **先用同 dtype `T.copy` 载入输入 dtype 的 bias，再显式 `T.vcast` 到 fp32**（Expert `T.copy` GM↔UB 不跨 dtype，`TRAP-C12-copy-dtype-cast`；吸收函数 1 Stage 3 修正），`T.vbrc(bias_col, bias_brc)`（`[BM,1]→[BM,BN]` 广播）+ `T.vadd(bias_brc, ub_f32, ub_f32)`（**fp32 域加 bias**）、`T.vcast(ub_f32, ub_f16)`、`T.copy(ub_f16 → out)`。`has_bias=False` 时**无 Vector 段**，L0C 直接 fixpipe（fp32→fp16，`CONST-store-fixpipe-gm-only`）写 `out`。**跨引擎握手（4-flag 双向，对齐 `_ssd_chunk_scan_fwd_kernel.py` 的深度 2 双槽）**：flag `2*slot` = ws 就绪（Cube 写后 `set` / Vector 读前 `wait`）；flag `2*slot+1` = 槽空闲（Vector 消费后 `set` / Cube 写前 `wait`）。**「槽空闲」前导 `set(1)/set(3)` 由消费者（Vector）Scope 置位**（Cube 写前 `wait(2*slot+1)`；同引擎自 set→自 wait 不 retire，函数 1 Stage 3 实测 aicore timeout 507014）。
- **语义保持论证**：bias 在 fp32 域相加后统一 cast（与源 `out_local + cast(bias, fp32)` 的加序与 dtype 一致）；数学恒等 `im2col-GEMM + bias ≡ im2col-GEMM + bias`。跨引擎经 GM ws 中转不改变数值（fp32 直存，非量化）。握手正确性：① 读前就绪（Vector 收到 `set(2*slot)` 才读）；② 写前空闲（Cube 收到 `set(2*slot+1)` 才覆盖）；③ 前导两槽空闲由 Vector 置（首两任务无需等待）；④ 每个 flag id 严格 set/wait 交替，深度 2 使 Cube(t) 与 Vector(t-1) 并发且 slot 不冲突。跨引擎传输强制 GM 往返（`CONST-store-fixpipe-gm-only`）——NPU 上把 bias 放 Vector 的固有代价；gqa/ssd 先例均用 GM ws 中继。

> §0.5 全部「重新设计」条目（R1–R5）已重设计。源优化手段（§0.4 #1/#2/#4/#6/#7/#10）意图承接：SMEM tiling→L1 分块 `T.copy`；Tensor core→Cube `T.gemm`+L0C；register 累加→L0C fp32 累加器；coalesced→MTE 连续 `T.copy`；空间完整快路径→slice 直接表达；尾块掩码→R2 分形 clamp+越界零填充；`T.Pipelined`→L1 多缓冲（Stage 4）。#5（host 重排）→R3 内移。#8 swizzle / #9 threads 为 NPU 无意义特性（舍弃）。

### 0.7 标杆实现

- 源算子路径（`source_op_path`）：`/mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py`（`_conv1d_kernel` builder，L33-148；`Conv1dKernel` 类，L745-871）。NPU 参考副本：`examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py`（`_conv1d_kernel`）；NPU 分发/包装：`examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d.py`。
- 源仓参考实现 / 测试基准：`torch.nn.functional.conv1d(x, weight, bias, stride, padding, dilation)`，见 `examples/TileOPs/tests/ops/test_conv1d.py`（`ref_program`）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现（`F.conv1d`），**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

`_conv1d_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一，函数 2，`Conv1dKernel.general=True` 的一般回退实现）；NPU 接口函数同名，`target="npuir"`。

### 1.2 功能描述

一般（dense，ungrouped）一维卷积的 implicit-GEMM 实现：把权重 `[C_out, C_in, K_W]` 与 `im2col(x)` 沿 `K = C_in·K_W` 收缩，`out[n,c_o,l] = Σ_{c_i,k_w} W[c_o,c_i,k_w]·x[n,c_i, l·s_W + k_w·d_W - p_L] (+ bias[c_o])`。用于分组/pointwise/单步长专用 kernel 不覆盖的一般调用（任意 stride / dilation / padding，`K_W>1`）。

### 1.3 数学公式

$$
\text{out}[n, c_o, l] = \text{bias}[c_o] + \sum_{c_i=0}^{C_{in}-1}\sum_{k_w=0}^{K_W-1} W[c_o, c_i, k_w]\, x[n, c_i,\; l\cdot s_W + k_w\cdot d_W - p_L], \qquad L_{out} = \Big\lfloor\frac{L_{in}+p_L+p_R-d_W(K_W-1)-1}{s_W}\Big\rfloor + 1
$$

### 1.4 算法描述

单步 implicit GEMM：将 `[C_out, K_total]`（K 序 `(k_w,c_i)`）权重作为 A、`im2col`（`[C_in, L_out]` 逐 tap 平移窗）作为 B，沿 `K = C_in·K_W` 分块累加（fp32），输出 `[C_out, L_out]`（逐 batch），可选 bias epilogue。本算法与 §1.6.0 调研选定算法一致（**基线 implicit GEMM 即选定主算法**；stride>1 时叠加 polyphase 等价分解）。与源算法结构差异：仅并行组织（三维 grid+threads → 一维 Kernel+persistent，§0.6 R1）、非整除/越界尾块（掩码 gather → 分形 clamp+slice 尾块+零填充，§0.6 R2）、weight-layout（host 重排 → kernel 内 `T.transpose` 预处理，§0.6 R3）、跨步 im2col（逐元素 gather → stride=1 per-tap slice / stride>1 polyphase，§0.6 R4）、bias epilogue（fragment 逐元素 → MixCV Vector，§0.6 R5）五处不同；GEMM 计算本体（K 序 `(k_w,c_i)`、fp32 累加）与源一致。

### 1.5 数据流图

```
（stride=1, 无 bias，纯 Cube）
GM[weight] --T.copy--> UB[wsrc] --T.transpose--> UB[wdst] --T.copy--> GM[w_flat]   (Vector 前置, 一次)
GM[x] --per-tap slice T.copy(ND2NZ)--> L1[l1_b] --+
GM[w_flat] --slice T.copy(ND2NZ)--> L1[l1_a] --T.gemm(fp32)--> L0C[l0_c] --slice T.copy(fixpipe f32→f16)--> GM[out]

（stride=1, 有 bias，MixCV）
   ... 同上至 L0C[l0_c] --T.copy(fixpipe f32→f32)--> GM[ws_bias]
GM[ws_bias] --T.copy--> UB[ub_f32] --vcast(bias→fp32)+vbrc+vadd(fp32)--T.vcast--> UB[ub_f16] --T.copy--> GM[out]

（stride>1：多一段 Vector 前置物化 y_poly，B 改为从 y_poly per-tap slice）
GM[x] --Vector gather--> GM[y_poly] ; GM[y_poly] --per-tap slice--> L1[l1_b] --T.gemm--> ...
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：调研四问 → 数学等价优化 → 向量化替代 → 向量化轴/布局。本节结论是 §3.1 公式拆解与 §6 循环结构的输入；迁移任务分析对象为 §0.6 重设计后的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：一般 conv1d 的算法族。源算法 / 输入公式只是**基线候选之一**。本算子为**矩阵类（implicit GEMM）+ 窗口类**，按 algorithm-research.md §2 判为**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。信息源：algorithm-candidates.md（`ALG-conv` 命中行）、pattern-library constants.md、本仓 Cube 先例（gqa/ssd）、源算子 §0.3/§0.4、探针 `examples/conv1d/probe/conv1d_geomm_probe.py`。

**R1 等价化简公式候选**（基线必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线**（源算法 / 输入公式） | implicit GEMM：`out = W_flat @ im2col(x) + bias`（K 序 `(k_w,c_i)`，K 分块 fp32 累加；per-tap 平移窗 im2col） | — | — | ✅（基线） |
| 1 | **im2col + 显式 GEMM**（`ALG-conv` 命中行） | 先把 `im2col[C_in·K_W, L_out]` 物化到 GM，再做 plain GEMM | 数学等价（im2col 是恒等展开） | 无：物化 `im2col` 引入 `O(C_in·K_W·L_out)` 额外 GM 写读（≫ x 体量），仅 stride>1 的必要绕路 | ✅（stride>1 变体评估） |
| 2 | **Winograd**（`ALG-conv` 命中行） | 多抽头小核的 F(m, r) 变换域卷积 | **容差内等价**（变换域舍入） | 仅对多抽头（`K_W≥2`）小幅降乘法、增加法；NPU 无实测支撑且变换/逆变换开销大 | ❌（NPU 无实测收益支撑 + 变换开销，见 R4） |
| 3 | **FFT 卷积** | 频域乘 + 逆变换 | 容差内等价 | 仅大核长序列有效（`K_W` 大）；本算子 `K_W≤10`，FFT 开销 > 收益 | ❌（小核不适用） |
| 4 | **split-K** | 分段 K + 二段归约 | 容差内等价（fp32 分段归约舍入） | 仅提升并行度，不减少 FLOPs/访存 | ❌（K 不大，跨核归并代价 ≥ 收益） |
| 5 | **bias 折叠（augmented-K）** | `[W \| bias] @ [im2col ; 1]` | 数学恒等（`bias·1` 项） | 消 Vector epilogue；但 K 破坏 16 对齐 + 需物化 `x_aug` | ❌（K 对齐代价 + 物化访存 ≫ bias 收益） |
| 6 | **polyphase 分解（stride>1 专用）** | `x[ci, ol·s_W+c] = y_poly[ci, r_kw, ol+q_kw]`，`c=kw·d_W-p_L`；把列采样移出 DMA 通路 | **精确恒等**（`c` 的 `(q,r)` 分解唯一） | 使 stride>1 的 B 装载回归 per-tap 连续 slice（否则 ND2NZ 非法）；代价 = 一次 `y_poly` 物化（≈ x 体量，仅 stride>1） | ✅（stride>1 采用） |
| 7 | **转置 GEMM** | `out^T = im2col^T @ W_flat^T` | 数学等价（转置） | 无计算量收益；输出转置需重排 | ❌（布局问题非公式化简，见 §1.6.3） |

**R2 在线算法**：**结论「无独立在线变体」**——im2col 隐式展开天然单遍（每个输出位置 `(c_o,l)` 仅依赖对应的 `K` 段输入与其窗口，无跨位置 running 统计量），源算法 §0.3 亦为单遍 K 分块累加，不存在「两遍扫描 + O(N) 中间缓冲」形态；滑动窗口本身即为流式结构。结构依据：输出无 cross-tile 依赖、无 running max/sum 重缩放语义（对比 online softmax 类）。候选 split-K（R1 候选 4）是并行化手段而非在线算法。

**R3 复杂度对比**（逐候选四口径；带宽受限算子上访存/遍数是主导项，不得只比 FLOPs；口径标明含什么不含什么）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **基线**（implicit GEMM，K 分块） | `2·C_out·L_out·C_in·K_W`（MAC=2；bias 加 `+C_out·L_out`，可忽略） | 输入读 `C_in·L_in·eb` + 权重读 `C_out·C_in·K_W·eb` + 输出写 `C_out·L_out·eb`；**无中间 GM 往返**（无 bias 时）；bias 时 `+ws fp32 往返 2·C_out·L_out·4B` | 1（每输入读一次；K 分块不重复读 x/W） | `O(block_m·block_n)`（L0C fp32 tile）+ `O(block_m·block_k + block_k·block_n)`（L1 tile） | 高（tile 间独立，logical = GY·GX·n 核；无跨核归并） |
| 候选 1（im2col 显式） | 同基线 | 同基线 + `im2col` 物化 `2·C_in·K_W·L_out·eb`（写+读） + `y_poly`（若 stride>1） | 1（但 im2col 需物化后读一遍） | 同基线 + `O(C_in·K_W·L_out)` GM | 同基线 |
| 候选 6（polyphase，仅 stride>1） | 同基线（**不增加 FLOPs**，只换采样） | 同基线 + `y_poly` 物化 `≈ C_in·L_in·eb`（写+读，仅 stride>1） | 1（`y_poly` 物化后单遍） | 同基线 + `O(C_in·L_g·s_W)` GM（≈ x 体量） | 同基线 |
| 候选 4（split-K） | 同基线 | 同基线 + 分段部分和 GM 往返 `O(split·C_out·L_out)` | 1（K 分块；二段归约读回部分和） | `O(split·block)` 部分和 | 并行度↑，**跨核归并代价** |
| 候选 2（Winograd） | 乘法 `~C_out·L_out·C_in·K_W/2`，加法显著↑ | 变换/逆变换中间缓冲 | 1 | 变换域 tile | 变换链串行 |

**结论**：基线四口径最优（无额外 GM 往返、单遍、O(tile) 驻留）；stride>1 用候选 6（polyphase）以**同 FLOPs** 换回合法的连续 DMA（唯一满足 NPU ND2NZ 约束的等价变形）；候选 1/2/4 均在某口径劣于基线（物化访存 / 变换开销 / 归并代价）。基线（+stride>1 polyphase）即选定。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰附本仓文档与 examples 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **基线** | MAC 密集 → **Cube**（`T.gemm` + L0A/L0B/L0C），正确落单元；`T.transpose` 前置 → Vector | L0C `64·128·4 = 32KB ≤ 128KB`；L1 `(64·128 + 128·128)·2 = 48KB/槽 ≤ 512KB`；UB 见 §4.5 | K/N 尾轴分形 ≥16（`ceil16` clamp）；32B 对齐（fp16 尾轴 128·2=256B）✓ | logical / persistent 边界均为编译期常量（N/block 为 JIT 参数）✓ | K 分块可 `T.Pipelined` 双缓冲；Cube 段与 weight/bias Vector 段可 CV 融合 ✓ | ✅ **采纳** |
| 候选 1（im2col 显式） | 同基线（Cube） | 同基线 + `im2col` GM（非片上，流量↑） | 同基线 | 同基线 | 同基线 | ❌（stride=1 时物化访存 ≫ 收益；stride>1 时由候选 6 更省） |
| 候选 6（polyphase） | 同基线（Cube）；物化 → Vector | `y_poly` GM（非片上，≈ x 体量） | `T.copy` 三/四维 slice，最内 `q` 连续；单位维自动折叠（`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2）✓ | `Lg`/`s_W` 编译期常量 ✓ | 物化段与主循环可流水 | ✅ **采纳（仅 stride>1）** |
| 候选 2（Winograd） | 变换/逆变换 → Vector，乘法 → Cube 但数据布局复杂 | 变换域 tile 驻留 UB | 变换矩阵尺寸约束（`K_W` 小则无收益） | — | 变换链引入额外 pass | ❌（NPU 无实测收益支撑；小核变换开销 > 收益——「未文档化假设 + 估算依据」：`K_W≤3` 时 Winograd F(2,3) 理论乘法降幅仅 ~2.25×，而变换/逆变换的向量 op 发射开销（`CONST-vector-launch-overhead` ~0.5µs/op）叠加后净亏） |
| 候选 3（FFT） | FFT → Vector | 频域缓冲 | — | — | — | ❌（`K_W≤10` 小核，FFT 开销 > 收益） |
| 候选 4（split-K） | 同基线 + 跨核归约（Vector） | 部分和 GM 往返 | — | 需动态 split 边界 | 二段归约需跨核同步 / 两 kernel | ❌（K 不大，归并代价 ≥ 收益） |

**调研结论**：选定算法族 = **基线 implicit GEMM（A=`w_flat[C_out, C_in·K_W]`（K 序 `(k_w,c_i)`）M×K，B=`im2col[C_in·K_W, L_out]` K×N，K 分块 fp32 累加，可选 bias），stride>1 叠加 polyphase 等价分解**。关键依据：R1 无更优等价化简（基线已是最小 FLOPs 形式；Winograd/FFT 对小核无 NPU 实测收益，im2col/转置为布局或物化代价）；R2 单遍窗口 K 分块累加即为最优流式（无两遍在线变体）；R3 基线（+stride>1 polyphase）四口径全维度最优；R4 计算单元匹配（MAC→Cube、布局重排/物化→Vector）、片上容量达标（L0C 32KB / L1 48KB/槽）、分形对齐（ceil16 clamp）、静态边界、可流水/CV 融合。与基线的结构差异：**无算法级差异**——调研确认源算法即最优；仅按 §0.6 重设计其并行组织（R1）、尾块（R2）、weight-layout（R3）、跨步 im2col（R4）、bias epilogue（R5）为 NPU 形态。「无更优替代」调研范围：algorithm-candidates.md（`ALG-conv` 命中行 im2col+GEMM / implicit GEMM / Winograd，经 `kb_search.py` 检索）、恒等变形自查（R1 候选 1–7 全部评估）、pattern-library（layout.md、constants.md、traps-runtime.md 的 Cube 迁移档案）、本仓 Cube 先例（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`）、源算子 §0.3/§0.4、探针实测。源优化手段意图承接见 §0.6 末段。本结论是 §1.4 与 §1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2）**（完整调研，流量项 + 发射项 + 容量项均算）：

```
代表 workload: whisper-conv1 (N=1, C_in=80, L_in=3000, C_out=1280, K_W=3, stride=1, pad=0, fp16, L_out=2998)
估算下界: ≈ 18µs（流量 16µs + 发射 15µs，取主导；容量不约束，引用 CONST-mte2-degradation / CONST-L1-port-bw / CONST-vector-launch-overhead / CONST-capacity-910B2C）
  流量项(compulsory, L2 吸收跨 M-tile 复用): GM = x 读 80·3000·2=0.48MB + weight 读 1280·80·3·2=0.61MB
        + w_flat 写 1280·240·2=0.61MB + w_flat 读 0.61MB + out 写 1280·2998·2=7.68MB ≈ 10.0MB；
          混合读写流量地板按 CONST-mte2-degradation（~31 GB/s/core 档）× 20 核 ≈ 0.62 TB/s → 10.0MB/0.62TB/s ≈ 16µs（输出写主导，占 77%）
  发射项(Cube 操作数流地板): L1→L0 操作数流 = Σ_tasks (block_m·K + K·block_n)·2B = 480·((64+128)·240)·2 ≈ 44.2MB
        ÷ CONST-L1-port-bw(≈148–154 GB/s/核) ÷ 20 核 ≈ 15µs
  容量项: L0C tile 32KB ≤ 128KB；L1 单槽 48KB ≤ 512KB；UB 前置/bias 路径见 §4.5（不构成下界）
  注: Cube MAC 吞吐未实测（constants.md 无 Cube FLOPs 常数）——标注「未实测假设」；1.84 GFLOP 按较大算力估计远小于流量项
  参考: 估算下界以流量项/操作数流为主导，符合带宽受限画像；Stage 4 以 msprof Task Duration 对账（hardware-cost-model.md §3）
```

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（基线 implicit GEMM + optional bias；stride>1 叠加 polyphase）。逐项评估候选优化，无采纳项则写结论与依据。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | Winograd | 直接卷积 | 变换域 F(m,r) 卷积 | 容差内等价（变换域舍入） | **净亏**：`K_W≤3` 小核理论乘法降幅有限，变换/逆变换向量 op 发射开销（`CONST-vector-launch-overhead`）叠加后净亏，且 NPU 无实测支撑；不采纳 |
| 2 | bias 折叠（augmented-K） | `im2col-GEMM + bias` | `[W \| bias] @ [im2col ; 1]` | 数学恒等（`bias·1` 项） | **净亏**：K 破坏 16 对齐需 pad，且需物化 `x_aug`（O(K·L_out) 额外 GM 写读）；不采纳 |
| 3 | 转置 GEMM | `out = W_flat @ im2col` | `out^T = im2col^T @ W_flat^T` | 数学等价（转置） | **无收益**：FLOPs/访存同；引入输出重排（PL-1.1 / PL-1.3）；布局问题归 §1.6.3；不采纳 |
| 4 | split-K | 单段 K 累加 | 分段部分和 + 二段归约 | 容差内等价（fp32 分段归约舍入） | **净亏**：K 不大，部分和 GM 往返 + 跨核归并代价 ≥ 收益；不采纳 |
| 5 | 除法/超越函数降代换、公共子表达式消除、稳定化变形 | — | — | **不适用**：本算子为纯 im2col-GEMM + 加法，无除法、无超越函数、无公共子表达式、无溢出风险需稳定化 | 无 |

**优化结论**：**无优化空间**。本算子为 im2col implicit GEMM（`out = W_flat @ im2col(x) (+bias)`）——无除法的乘倒数、无超越函数降代换、无公共子表达式（每项 M/K/N 元素仅用一次）、无归约稳定化需求（纯线性累加）；四个公式级候选（Winograd / bias 折叠 / 转置 / split-K）经 §1.6.0 R1/R3/R4 评估均无正收益，不采纳。polyphase（§0.6 R4 / §1.6.0 R1 候选 6）是**精确恒等的结构分解**（算法路线，非公式化简），不改变 FLOPs，已在 §1.6.0 采纳为 stride>1 路线。优化后公式 = 基线公式，供 §3.1 公式拆解使用。**因无采纳的公式级优化项，按 skill Phase 2 第 5 步与 gate `S1-EQUIV-EXEC` 的豁免规则（「无优化空间」显式结论），不生成 `verify_equiv.py`。**

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 本算子主导计算为 Cube（矩阵单元）；非矩阵部分（weight-layout 重排、polyphase 物化、bias epilogue）在 Vector。§1.6.2 盘点实现方案中全部循环与标量计算点。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | `W_flat @ im2col` 矩阵乘累加 | 源 `T.gemm`（tensor core mma） | **Cube 单元** `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])` | ✅（Cube，非标量） | — |
| 2 | weight-layout 重排 `[C_out,C_in,K_W]→[C_out,K_W,C_in]` | 源 host `permute+contiguous` | kernel 内 `T.copy`(GM→UB) + `T.transpose`(UB, perm=[0,2,1]) + `T.copy`(UB→GM)（`docs/Tilelang.language/创建操作/T.transpose.md`；PL-1.1 核内融合转置链） | ✅（UB 向量转置） | — |
| 3 | im2col B 装载（stride=1） | 源逐元素 `T.Parallel` gather | **slice 形态 `T.copy`** per-tap（GM→L1 自动 ND2NZ，最内维连续；探针实证） | ✅（DMA 向量搬运） | — |
| 4 | im2col B 装载（stride>1） | 源逐元素 gather | Vector 前置物化 `y_poly`（`T.Parallel` 索引写）+ per-tap slice `T.copy` | ✅（DMA 搬运） | 物化点见 #7 |
| 5 | bias 逐元素相加 | 源 `T.Parallel` fragment 逐元素 `+bias[oc]` | `T.vbrc(bias_col, bias_brc)` + `T.vadd(bias_brc, ub_f32, ub_f32)`（Vector 段；`_ssd_chunk_scan_fwd_kernel.py` 同款 `T.vbrc`/`T.vadd`；`PL-1.15-broadcast-perf-tax`） | ✅ | — |
| 6 | 输出 cast（fp32→fp16/bf16） | 源 `T.cast(out_local, dtype)`（epilogue） | 无 bias：fixpipe `T.copy(l0_c, out)`（`CONST-store-fixpipe-gm-only` 合法量化）；有 bias：`T.vcast(ub_f32, ub_f16)` | ✅ | — |
| 7 | polyphase 物化索引 `y_poly[bn,ci,r,q] = x[bn,ci,q·s_W+r-p_L]` | — | `T.Parallel(Lg, s_W)` 多元素并行写入（带越界 `T.if_then_else` 零填充） | ❌（部分） | **跨步读（步长 `s_W`）：`TRAP-C10-strided-gather` 记录 npuir 无跨步 gather 向量指令**，逐元素索引写落在 Vector 标量路径；仅 stride>1 触发且代表 shape 小（`y_poly ≈ x` 体量），每元素 ~1ns；属**动态越界边界处理 + API 缺失**（理由类别④/⑤） |
| 8 | K 维分块循环 | 源 `T.Pipelined(ceildiv(k_total, block_k))` | `T.serial(ceil(k_total/block_K))`（tile 级调度；Stage 4 可 `T.Pipelined` 多缓冲） | ✅（tile 级循环，非逐元素热点） | — |
| 9 | `cid → (bz, by, bx)` 解码 | 源三维 grid 隐式 | 标量 `cid // (GY*GX)`、`(cid // GX) % GY`、`cid % GX` | ❌ | **block 级任务映射计算**，每 tile 少量标量，非逐元素热点，无向量等价 API（理由类别①） |
| 10 | 尾块尺寸 `tm/tk/tn/…c` | 源 `if_then_else` 逐元素掩码 | 标量 `T.min` / `T.ceildiv` / `T.max(16, …)` | ❌ | **依赖动态 shape 的边界处理**（尾块裁剪 / 越界 clamp），每 tile 一次，非逐元素热点（理由类别④）；NPU 无「逐元素条件写向量」形态 |
| 11 | `T.min(cid, num_logical-1)` 钳位 | — | 标量 `T.min` | ❌ | block 级索引守卫，每 tile 一次（类别①）；防 `T.ceildiv` 对 idle 核越界 |

**向量化结论**：逐元素计算点（weight 重排、bias 相加、输出 cast）已**全部向量化**（`T.transpose` / `vbrc` / `vadd` / `vcast`，API 均有本仓 examples 与 docs 佐证）；矩阵乘由 Cube 单元承接（`T.gemm`）；im2col 搬运以 DMA slice `T.copy` 承载。保留 4 类标量点：polyphase 物化索引（跨步 gather，API 缺失，仅 stride>1）、cid 解码、尾块尺寸、cid 钳位——均属 block 级任务映射 / 动态边界 / 跨步 gather 缺失，非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。**矩阵乘不适用「逐元素向量化」——落 Cube 是正确单元选择。**

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> Cube 类算子：fractal/NZ 布局与操作数轴选择是设计变量；weight-layout / im2col 轴序 / epilogue 归属独立评估。

**轴质量评分维度**（逐候选填写）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | N 轴（L_out）长是否为向量/分形宽度整数倍 | whisper-conv1 `L_out=2998` 非 128 整数倍（尾块 54 / 128）；`ceil16` clamp 后分形对齐 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len`（vw=128 tile 粒度） | 末 tile `(24·128−2998)/2998 ≈ 2.5%`（整算子口径） |
| 累加链形态 | K 轴为串行累加维；N/M 为分形维 | K=`C_in·K_W` 串行累加（Cube 内部）；M/N 分形并行 |
| repack 代价 | 改变布局需要的重排代价 | 选定方案 K 序 `(k_w,c_i)`：A 需 weight 重排（`T.transpose` UB，一次性，非 per-tile）；B 无 repack（per-tap slice 直读） |
| UB 容量影响 | 布局变化对驻留缓冲尺寸的影响 | 选定方案无额外常驻；weight 前置 `2·bm_w·c_in·K_W·2B`，bias 路径见 §4.5 |

**候选矩阵**（全部枚举，弃选必须给量化理由，禁止「未考虑」）：

| # | 布局方案 | 操作数轴 / K 序 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **K 序 `(k_w,c_i)`**：A=`w_flat[C_out, k_total]`（M×K）、B=`im2col[k_total, L_out]`（K×N）、dst=L0C`[C_out,L_out]` | M=C_out（分形行）、N=L_out（分形列）、K 序 `(k_w,c_i)`（`c_i` 最内） | A：weight `[C_out,C_in,K_W]→[C_out,K_W,C_in]` 的 `T.transpose`（UB 级 2 轴交换，一次性 ~µs）；B：**无**（per-tap slice，最内 `L` 连续） | 基线；B 零 repack、探针实证；仅 A 一次性重排 | ✅ |
| 2 | **K 序 `(c_i,k_w)`**：A=weight 原生扁平 `[C_out, C_in·K_W]`（`k_w` 最内）；B=im2col `[k_total, L_out]` | M=C_out、N=L_out、K 序 `(c_i,k_w)` | A：**无**（原生扁平，`k_w` 最内连续）；B：im2col 需按 `(c_i,k_w)` 序物化（band/Toeplitz 非矩形 → 逐元素 gather，O(K·L_out)） | A 省一次转置，但 B 每次 per-tile 都要 gather（O(K·L_out) 标量，stride=1 也触发）→ 净亏 | ❌（B 物化访存 ≫ A 转置的一次性代价） |
| 3 | 转置 GEMM（A/B 角色互换，A=im2col^T） | M=L_out、N=C_out | A=im2col^T 需 gather（同 #2 问题）；输出转置 | 无收益、增代价 | ❌（重排代价；`TRAP-transpose-epilogue-poison` 风险） |
| 4 | host permute 进出（weight/x 通道重排） | — | 两次 host permute | 违反 ascend-constraints.md §4（host 禁改输入内容）；百 µs 级净亏 | ❌（违规 + 净亏） |
| 5 | B 轴选 `k_w` 内层 + `c_i` 外层（per-channel 分组装载） | — | 按 `c_i` 分组装载需 `c_in·K_W` 次小拷贝 | 指令数爆炸（`CONST-mte2-degradation` 指令受限；小块搬运指令受限） | ❌（per-tap 分组拷贝数最少） |

> 迁移任务：GPU 源码的并行轴（`bx`=L、`by`=C_out、`bz`=batch）是**输入而非结论**；GPU 的隐式 K 序 `(k_w,c_i)`（`kw=k_idx//c_in`）是**算法输入**（B 的 per-tap 连续 slice 依赖此序），本设计承接为核内 A/B 轴序（候选 1）。

**布局决策结论**：选定方案 **#1**：核内布局 = NZ 分形（`T.copy` GM→L1 自动 ND2NZ；L0C→GM 自动 fixpipe NZ2ND），操作数轴 = A=`w_flat[C_out, C_in·K_W]`（K 序 `(k_w,c_i)`，`c_i` 最内）× B=`im2col[C_in·K_W, L_out]`（K×N，最内 `L` 连续）、dst=L0C`[C_out,L_out]`。A 的一次性 weight 重排（`T.transpose` perm=[0,2,1]）与 B 的 per-tap slice 均落入 §3.3 伪代码与 §6 循环结构（K 块按 `(k_w,c_i)` 序装载；L0C tile 形状 `[block_M, block_N]`，`T.gemm` 内层 `size=[M,K,N]` 与本节操作数轴一致）；弃选方案（#2–#5）量化理由见候选矩阵。epilogue（bias）向量化轴 = N（L_out）连续轴（UB fp32 tile 行内连续，`vadd` 逐元素）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Expert（`ascend_mode="Expert"`）。

### 2.2 选型理由

本算子为 Cube（矩阵乘）主导算子，须显式使用 L1（Cube 操作数缓存）与 L0C（fp32 累加器）内存层级，并通过 `T.gemm` 走 Cube 单元。Expert 模式是 npuir 上显式 L1/L0C 与 `T.Scope("Cube")` 的唯一可行路径：Developer 模式的 `T.alloc_shared` / `T.alloc_fragment` 在 persistent + gemm 结构下存在运行时崩溃（`TRAP-DEVMODE-PERSIST-GEMM`），且 `PL-1.16-expert-dualscope-bypass` 记录 Developer 在标量化 / 条件构造 / persistent+gemm 下的结构级阻塞。本算子还需：weight-layout 前置（Vector `T.transpose`）、stride>1 polyphase 物化（Vector）、bias epilogue（Vector）——均需 Cube + Vector 双 Scope（MixCV），属 Expert。探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 即以 `TILELANG_ASCEND_MODE=Expert` + `T.Scope("Cube")` 编译运行成功。**任务硬性要求编程模式 = Expert**。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_L1`（`l1_a`/`l1_b`）、`T.alloc_L0C`（`l0_c`，fp32）；Vector 段 `T.alloc_ub`（weight 前置 `wsrc`/`wdst`、bias 路径 `ub_f32`/`ub_f16`/`bias_16`/`bias_col`/`bias_brc`） |
| 计算方式 | Cube：`T.gemm(l1_a, l1_b, l0_c, initC=..., size=[M,K,N])`；Vector：`T.copy`+`T.transpose`（weight 前置）、`T.vbrc`+`T.vadd`+`T.vcast`（bias）、polyphase `T.Parallel` 索引写 |
| 数据搬运 | `T.copy` slice 形态（GM→L1 自动 ND2NZ；L0C→GM 自动 fixpipe NZ2ND + 可选 fp32→fp16 量化；GM↔UB 通用路径同 dtype） |
| 同步 | **手动同步**（Expert）：`with T.rs("PIPE_MTE2"/"PIPE_C"/"PIPE_FIX"/"PIPE_MTE3")` 包裹对应流水段；跨引擎（Vector 前置→Cube、Cube↔Vector bias）用 `T.sync_block_set/wait` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式（= 基线，无采纳的公式级优化项）；每步映射 Cube / Vector API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `w_flat[c_o, k_w, c_i] = weight[c_o, c_i, k_w]` | weight-layout 前置（Vector，一次性；K 序 `(k_w,c_i)`） |
| 2 | `y_poly[bn,c_i,r,q] = x[bn,c_i,q·s_W+r-p_L]`（仅 stride>1 物化） | polyphase 前置（Vector，一次性） |
| 3 | `(bz,by,bx) = decode(cid)` | tile 任务解码（block 级标量） |
| 4 | `l1_a = w_flat[by·block_M : +tm, k0:k0+tk]` | A 操作数装载（GM→L1，ND2NZ），尾块 slice |
| 5 | `l1_b[kw·c_in:(kw+1)·c_in, :] = src[bz, 0:c_in, off_kw : off_kw+tn]`（stride=1：`src=x`、`off_kw=l0+kw·d_W-p_L`；stride>1：`src=y_poly`、`off_kw=l0+q_kw[kw]`、相位 `r_kw[kw]`） | B 操作数装载（per-tap slice，ND2NZ），越界零填充 |
| 6 | `l0_c = w_tile @ x_tile`（`initC=(k==0)`，fp32 累加） | Cube GEMM，K 分块累加 |
| 7 | `out_tile = cast_f16(l0_c)` | 无 bias：fixpipe L0C→GM（fp32→fp16 量化） |
| 8 | `out_tile = cast_f16(l0_c_f32 + bias[c_o])` | 有 bias：fixpipe→GM ws(fp32)，Vector `+bias`（fp32）再 cast，写 out |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | weight-layout 前置 | `T.copy` + `T.transpose` + `T.copy` | `T.copy(weight[co0:+cw, 0:c_in, 0:K_W], wsrc)`；`T.transpose(wsrc, wdst, permutation=[0,2,1])`；`T.copy(wdst, w_flat[co0:+cw, 0:K_W, 0:c_in])` | Expert |
| 1 | polyphase 前置 | `T.Parallel` 索引写 | `T.Parallel(Lg, s_W)`：`p = q·s_W + r - p_L`；`y_poly[bn,ci,r,q] = T.if_then_else((p>=0)&(p<l_in), x[bn,ci,p], 0)`（逐元素集合 `ci`/`bn` 外层 `T.serial`） | Expert |
| 2 | tile 解码 | `T.serial` / 标量算术 | `cid = T.min(task_id*num_kernels+kernel_id, num_logical-1)`；`bz/by/bx` 除法取模 | Expert |
| 3 | A 装载 | `T.copy(w_flat[by*BM : +tm, k0:k0+tk], l1_a[0:tm, 0:tk])` | slice 形态；GM→L1 自动 ND2NZ | Expert |
| 4 | B 装载 | `T.copy(x[bz, 0:c_in, off : off+tn], l1_b[kw*c_in:(kw+1)*c_in, 0:tn])` | slice 形态；绕过 `T.load_nd2nz` 跨步误读（`TRAP-load-nd2nz-strided` / CG-2026-0004） | Expert |
| 5 | GEMM | `T.gemm(l1_a, l1_b, l0_c, initC=(k==0), b_transpose=False, size=[tmc, tkc, tnc])` | `tmc/tkc/tnc = T.max(16, ceil16(...))` 分形 clamp | Expert |
| 6 | 无 bias 输出 | `T.copy(l0_c[0:tl, 0:tn], out[bz, by*BM : +tl, bx*BN : +tn])` | L0C→GM 自动 fixpipe（fp32→fp16 量化） | Expert |
| 7 | 有 bias 输出（Cube 段） | `T.sync_block_wait(2*slot+1)`（写前等槽空闲）；`T.copy(l0_c[0:tl,0:tn], ws[kernel_id, slot, 0:tl, 0:tn])`；`T.sync_block_set(2*slot)`（写后置就绪） | `slot = task_id % 2`；fixpipe 不量化（fp32→fp32） | Expert |
| 8 | bias 相加（Vector 段） | `T.sync_block_wait(2*slot)`（读前等就绪）；`T.copy(ws→ub_f32)`；`T.copy(bias→bias_16)`；`T.vcast(bias_16, bias_col)`；`T.vbrc(bias_col, bias_brc)`；`T.vadd(bias_brc, ub_f32, ub_f32)`；`T.vcast(ub_f32, ub_f16)`；`T.copy(ub_f16→out)`；`T.sync_block_set(2*slot+1)`（消费后归还） | bias 由输入 dtype 经显式 `T.vcast` 到 fp32（Expert `T.copy` 不跨 dtype） | Expert |
| 9 | 同步 | `with T.rs("PIPE_MTE2")` / `T.rs("PIPE_C")` / `T.rs("PIPE_FIX")` / `T.rs("PIPE_MTE3")` | 各流水段包裹 | Expert |

### 3.3 计算伪代码

```python
# Expert 模式；ascend_mode = "Expert"
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

@tilelang.jit(out_idx=[2], target="npuir")
def _conv1d_kernel(n, c_in, l_in, c_out, kernel_l, stride_l,
                   pad_left, pad_right, dilation_l, has_bias, dtype="float16"):
    accum = "float32"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    k_total = c_in * kernel_l
    block_M, block_N, block_K = 64, 128, 128      # 承接源 default_config
    slots = 2                                      # ws 深度 2 双槽
    bm_w = 16                                      # weight-layout 前置 co 分块（UB 预算）
    GX = T.ceildiv(out_l, block_N)
    GY = T.ceildiv(c_out, block_M)
    num_logical = GY * GX * n
    num_kernels = NPUUtils.get().get_aicore_num()   # Cube 直取实查值 = 20（§5.5）
    need_poly = stride_l != 1
    Lg = T.ceildiv(l_in + pad_left + (kernel_l - 1) * dilation_l, stride_l) + 2
    # 相位（编译期常量）：r_kw = (kw*d_w) mod s_w；q_kw = (kw*d_w - r_kw)//s_w
    r_kw = [(kw * dilation_l) % stride_l for kw in range(kernel_l)]
    q_kw = [(kw * dilation_l - r_kw[kw]) // stride_l for kw in range(kernel_l)]

    def _body(x, weight, out, bias, w_flat, y_poly, ws):
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
            num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)

            # ---- 阶段 A：Vector 前置（weight-layout + 可选 polyphase），一次 ----
            with T.Scope("Vector"):
                for co0 in T.serial(T.ceildiv(c_out, bm_w)):
                    cw = T.min(bm_w, c_out - co0 * bm_w)
                    wsrc = T.alloc_ub([bm_w, c_in, kernel_l], dtype)
                    wdst = T.alloc_ub([bm_w, kernel_l, c_in], dtype)
                    T.copy(weight[co0 * bm_w : co0 * bm_w + cw, 0:c_in, 0:kernel_l],
                           wsrc[0:cw, 0:c_in, 0:kernel_l])
                    T.transpose(wsrc, wdst, permutation=[0, 2, 1])   # [C_in,K_W]->[K_W,C_in]
                    T.copy(wdst[0:cw, 0:kernel_l, 0:c_in],
                           w_flat[co0 * bm_w : co0 * bm_w + cw, 0:kernel_l, 0:c_in])
                if need_poly:
                    for bn in T.serial(n):
                        for ci in T.serial(c_in):
                            for q, r in T.Parallel(Lg, stride_l):
                                p = q * stride_l + r - pad_left
                                y_poly[bn, ci, r, q] = T.if_then_else(
                                    (p >= 0) & (p < l_in), x[bn, ci, p], T.cast(0.0, dtype))
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(4)                 # 4 = layout/polyphase ready (Vector -> Cube)

            # ---- 阶段 B：Cube 主循环 ----
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_M, block_K], dtype)
                l1_b = T.alloc_L1([block_K, block_N], dtype)
                l0_c = T.alloc_L0C([block_M, block_N], accum)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(4)                # 等 weight-layout/polyphase 就绪
                for task_id in T.serial(num_local_tasks):        # persistent 核内串行
                    cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                    bz = cid // (GY * GX)
                    by = (cid // GX) % GY
                    bx = cid % GX
                    tm = T.min(block_M, c_out - by * block_M)
                    tn = T.min(block_N, out_l - bx * block_N)
                    tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                    tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                    for k in T.serial(T.ceildiv(k_total, block_K)):
                        tk = T.min(block_K, k_total - k * block_K)
                        tkc = T.max(16, T.ceildiv(tk, 16) * 16)
                        with T.rs("PIPE_MTE2"):
                            T.copy(w_flat[by * block_M : by * block_M + tm,
                                          k * block_K : k * block_K + tk],
                                   l1_a[0:tm, 0:tk])
                            # per-tap B 装载：K 序 (k_w,c_i)，一个 tap 覆盖 c_in 行
                            for kw in T.serial(kernel_l):
                                row0 = kw * c_in - k * block_K
                                if (row0 > -c_in) and (row0 < tk):
                                    if need_poly:
                                        off = bx * block_N + q_kw[kw]
                                        T.copy(y_poly[bz, 0:c_in, r_kw[kw], off : off + tn],
                                               l1_b[T.max(0, row0) : T.max(0, row0) + rows, 0:tn])
                                    else:
                                        off = bx * block_N + kw * dilation_l - pad_left
                                        T.copy(x[bz, 0:c_in, off : off + tn],
                                               l1_b[T.max(0, row0) : T.max(0, row0) + rows, 0:tn])
                        with T.rs("PIPE_C"):
                            T.gemm(l1_a, l1_b, l0_c, initC=(k == 0),
                                   b_transpose=False, size=[tmc, tkc, tnc])
                    if not has_bias:
                        with T.rs("PIPE_FIX"):
                            T.copy(l0_c[0:tm, 0:tn],
                                   out[bz, by * block_M : by * block_M + tm,
                                       bx * block_N : bx * block_N + tn])
                    else:
                        slot = task_id % slots
                        with T.rs("PIPE_FIX"):
                            T.sync_block_wait(2 * slot + 1)      # 写前等该槽空闲（WAR）
                            T.copy(l0_c[0:tm, 0:tn],
                                   ws[kernel_id, slot, 0:tm, 0:tn])  # ws fp32 中继（不量化）
                            T.sync_block_set(2 * slot)           # 写后置位：该槽 ws 就绪

            # ---- 阶段 C：Vector bias epilogue（仅 has_bias） ----
            if has_bias:
                with T.Scope("Vector"):
                    ub_f32 = T.alloc_ub([block_M, block_N], accum)
                    ub_f16 = T.alloc_ub([block_M, block_N], dtype)
                    bias_16 = T.alloc_ub([block_M], dtype)
                    bias_col = T.alloc_ub([block_M, 1], accum)
                    bias_brc = T.alloc_ub([block_M, block_N], accum)
                    with T.rs("PIPE_MTE3"):                     # 前导：消费者（Vector）置两槽空闲
                        T.sync_block_set(1)                     # slot 0 free
                        T.sync_block_set(3)                     # slot 1 free
                    for task_id in T.serial(num_local_tasks):
                        cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                        bz = cid // (GY * GX); by = (cid // GX) % GY; bx = cid % GX
                        tm = T.min(block_M, c_out - by * block_M)
                        tn = T.min(block_N, out_l - bx * block_N)
                        slot = task_id % slots
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(2 * slot)             # 读前等该槽就绪
                            T.copy(ws[kernel_id, slot, 0:tm, 0:tn], ub_f32[0:tm, 0:tn])
                            T.copy(bias[by * block_M : by * block_M + tm], bias_16[0:tm])
                        T.vcast(bias_16, bias_col[0:tm, 0])         # dtype -> fp32（不跨 dtype T.copy）
                        T.vbrc(bias_col, bias_brc)
                        T.vadd(bias_brc, ub_f32, ub_f32)            # fp32 域加 bias
                        T.vcast(ub_f32, ub_f16, round_mode="rint")
                        with T.rs("PIPE_MTE3"):
                            T.copy(ub_f16[0:tm, 0:tn],
                                   out[bz, by * block_M : by * block_M + tm,
                                       bx * block_N : bx * block_N + tn])
                            T.sync_block_set(2 * slot + 1)          # 消费后归还该槽（WAR 保护）

    if has_bias:
        @T.prim_func
        def _conv1d_bias_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, c_in, kernel_l), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            bias: T.Tensor((c_out,), dtype),
            w_flat: T.Tensor((c_out, kernel_l, c_in), dtype),
            y_poly: T.Tensor((n, c_in, stride_l, Lg), dtype),
            ws: T.Tensor((num_kernels, slots, block_M, block_N), accum),
        ):
            _body(x, weight, out, bias, w_flat, y_poly, ws)
        return _conv1d_bias_main

    @T.prim_func
    def _conv1d_main(
        x: T.Tensor((n, c_in, l_in), dtype),
        weight: T.Tensor((c_out, c_in, kernel_l), dtype),
        out: T.Tensor((n, c_out, out_l), dtype),
        w_flat: T.Tensor((c_out, kernel_l, c_in), dtype),
        y_poly: T.Tensor((n, c_in, stride_l, Lg), dtype),
    ):
        _body(x, weight, out, None, w_flat, y_poly, None)
    return _conv1d_main
```

> 注 1：`T.Tensor((...), dtype)` 用**位置参数**传 dtype（避免 `dtype=` 关键字产生 `name 'in_dtype' is not defined` false alarm，design-template §3.3 约定）。
> 注 2：`w_flat` / `y_poly` / `ws` 均为 GM workspace，作为 prim_func 形参由 wrapper 分配传入（对齐 `_ssd_chunk_scan_fwd_kernel.py` 将 `ws_c`/`ws_lcb` 作为 `T.Tensor` 形参的形态）。`w_flat[c_out,K_W,c_in]` 扁平即 K 序 `(k_w,c_i)`；`y_poly` 仅 stride>1 使用（stride=1 时该形参仍声明，wrapper 传最小占位张量，避免分支改签名）。
> 注 3：K 序 `(k_w,c_i)` 使一个 K 块可跨多个 tap（`block_K=128`、`c_in=80` 时一块含 1.6 个 tap）；B 装载在 k 块内**逐 tap** slice（每 tap 覆盖 `c_in` 行，越界行由 L1 预清零 + `T.copy` 只搬有效子区间实现零填充）；`l1_b` 行偏移 `row0 = kw*c_in - k*block_K`，实现期按 §6.4 规则 clamp 到 `[0, block_K)`（`rows` = 有效行数）。
> 注 4：**跨引擎握手单一来源**（§3.2/§3.3/§6.2/§7.2 一致）：前置就绪 flag `4`（Vector 写后 `set` / Cube 读前 `wait`）；bias flag `2*slot` = ws 就绪（Cube 写后 `set` / Vector 读前 `wait`）、`2*slot+1` = 槽空闲（Vector 消费后 `set` / Cube 写前 `wait`），**「槽空闲」前导 `set(1)/set(3)` 由消费者（Vector）Scope 置位**（吸收函数 1 Stage 3 修正：同引擎自 set→自 wait 不 retire，`TRAP-sync-self-set-wait`）；`ws` 按 `kernel_id` 索引，flag 为块内同步标志。
> 注 5：`num_kernels = NPUUtils.get().get_aicore_num()` 在 JIT 期求值（编译期常量）；`num_logical < num_kernels` 时 `cid` 钳位防 idle 核越界（gqa 先例）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.Scope("Cube")` / `T.Scope("Vector")` | `_gqa_prefill_fwd_kernel.py`；`_ssd_chunk_scan_fwd_kernel.py`（L234/L344）；cube-skill `api-cube.md` | ✅ 已验证（两处 Cube 迁移先例） |
| `T.alloc_L1` / `T.alloc_L0C` | cube-skill `api-cube.md`；`docs/Tilelang.language/内存操作/T.alloc_L1.md`、`docs/Tilelang.language/内存操作/T.alloc_L0C.md` | ✅ 已验证 |
| `T.gemm(a, b, c, size, initC, a_transpose, b_transpose)` | `docs/Tilelang.language/线性代数操作/T.gemm.md`（`size=[a,b,c]`；fp16 输入 dst 须 fp32）；探针 L100；gqa/ssd | ✅ 已验证（文档 + 探针 + 两先例） |
| `T.copy`（slice 形态 GM→L1 ND2NZ；L0C→GM fixpipe f32→f16；多维 slice 单位维折叠） | `docs/Tilelang.language/内存操作/T.copy.md`（方向表 + 动态 extents 规则 + §2.2.2 单位维折叠）；探针 L88-108 | ✅ 已验证（探针实测） |
| `T.copy`（GM↔UB 通用路径，同 dtype） | 同上（Expert 方向表 GM↔UB √） | ✅ 已验证 |
| `T.transpose(src, dst, permutation)`（UB 级，2 轴交换） | `docs/Tilelang.language/创建操作/T.transpose.md` §2.3（`permutation` 仅两轴交换，fp16 ✓）；PL-1.1-transpose-chain | ✅ 已验证（文档 + pattern-library 实测） |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；ssd；gqa | ✅ 已验证 |
| `T.vbrc` / `T.vadd` / `T.vcast` | `_ssd_chunk_scan_fwd_kernel.py`（`T.vbrc`/`T.vadd`/`T.vcast`）；gqa | ✅ 已验证（两先例） |
| `T.rs("PIPE_MTE2"/"PIPE_C"/"PIPE_FIX"/"PIPE_MTE3")` | `_gqa_prefill_fwd_kernel.py`；`_ssd_chunk_scan_fwd_kernel.py`（L253-457） | ✅ 已验证 |
| `T.sync_block_set` / `T.sync_block_wait` | `docs/Tilelang.language/同步管道操作/T.sync_block_wait.md` / `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`；gqa/ssd | ✅ 已验证 |
| `T.serial` / `T.Parallel` / `T.ceildiv` / `T.min` / `T.max` / `T.if_then_else` | gqa；ssd；`docs/Tilelang.language/内存操作/T.copy.md` 示例 | ✅ 已验证 |
| `NPUUtils.get().get_aicore_num()` | `tilelang.utils`（`from tilelang.utils import NPUUtils`）；本机实查 = 20（§5.5） | ✅ 已验证（实查） |
| `T.load_nd2nz` | `docs/Tilelang.language/内存操作/T.load_nd2nz.md`；**本设计避开**（`TRAP-load-nd2nz-strided` 跨步区域静默平坦误读；CG-2026-0004） | ⚠️ 存在但不用（改用 slice `T.copy`） |
| `T.store_fixpipe` | `docs/Tilelang.language/内存操作/T.store_fixpipe.md`；`T.copy` L0C→GM 等价自动转换 | ✅ 等价（`T.copy` 替代） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为三维 `T.Kernel(bx, by, bz)`） | §0.6 R1：一维 `T.Kernel(num_kernels, is_npu=True)` + `cid` 解码 `(bz,by,bx)` + persistent 核内串行 |
| GPU 专用 API | Yes（`threads`、`T.use_swizzle`、`T.alloc_shared`/`T.alloc_fragment`、`T.Pipelined`/`T.Parallel` 元素级 gather、`compile_flags`） | 全部改写为 NPU Expert Cube API（§0.5/§0.6） |
| GEMM 非整除 | **Yes**（`c_out % block_M`、`out_l % block_N`、`k_total % block_K` 均可能非整除） | §0.6 R2：分形 clamp `T.max(16, ceil16(tail))` + slice 尾块 `T.copy` + 裁剪 store |
| L0C 容量溢出 | 需核对 | `block_M·block_N·4 = 64·128·4 = 32KB ≤ 128KB`（`CONST-capacity-910B2C`）✓ |
| 分核策略 | **Yes** | §5.5 三要素（逻辑核数计算 / 物理核数 `get_aicore_num()` 实查 / 规模判定与分核方案） |
| `T.load_nd2nz` 跨步误读 | **Yes**（x 装载 `x[bz, ci0:ci0+BK, l0:l0+BN]` 行跨步 = l_in） | 改用 **slice 形态 `T.copy`**（文档化、探针实测的绕法；`TRAP-load-nd2nz-strided` / CG-2026-0004） |
| host 禁改输入内容 | **Yes**（源 host `weight.permute(...).contiguous()`） | §0.6 R3：移入 kernel Vector 前置段（`T.transpose`） |
| ND2NZ 要求区域最内维连续 | **Yes**（stride>1 的 im2col 列采样非连续） | §0.6 R4：stride=1 per-tap slice；stride>1 polyphase 物化后 slice |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

> 迁移任务的完整差异分析见 §0.5/§0.6，此处只汇总影响 API 选型的关键技术差异。

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | 三维 `T.Kernel(ceildiv(out_l,BN), ceildiv(c_out,BM), n, threads=128)` | 一维 `T.Kernel(num_kernels, is_npu=True)` + cid 解码 | §0.6 R1 |
| GEMM API | `T.gemm(weight_shared, data_shared, out_local)`（fragment 累加，Developer） | `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])`（Expert，L1/L0C） | §0.6 R1（Cube 路径） |
| 内存分配 | `T.alloc_shared` + `T.alloc_fragment`（SMEM/寄存器） | `T.alloc_L1` + `T.alloc_L0C` | §0.5 等价替换 |
| 权重布局 | host `permute(0,2,1).contiguous().view(c_out, k_total)` | kernel 内 `T.copy`+`T.transpose`+`T.copy` → `w_flat` | §0.6 R3（host 预处理内移） |
| im2col（stride=1） | 快路径 `T.copy` / 慢路径逐元素 gather | per-tap slice `T.copy`（快慢路径统一为 slice） | §0.6 R4 |
| im2col（stride>1） | 逐元素 `T.Parallel` gather（`il=ol·s_W+kw·d_W-p_L`） | polyphase 物化 + per-tap slice | §0.6 R4 |
| 累加器初始化 | `T.clear(out_local)` + 无 `initC` | `initC=(k==0)`（首 K 块清零） | 语义等价 |
| 流水 | `T.Pipelined(ceil(k_total,block_k), num_stages)` | `T.serial`（Stage 4 可 `T.Pipelined` 多缓冲） | §0.5 等价替换 |
| 元素级并行 | `T.Parallel(block_k, block_n)` B 装载 + epilogue | Cube `T.gemm` + Vector `transpose/vbrc/vadd/vcast` | §0.6 R2/R3/R4/R5 |
| 非整除/越界尾块 | 逐元素 `if_then_else` 掩码 | 分形 clamp + slice 尾块 + 越界零填充 + 裁剪 store | §0.6 R2/R4 |
| 布局技巧 | `T.use_swizzle` + `enable_rasterization` | 无（NZ 自动 + persistent 调度） | §0.5 舍弃 swizzle |
| `threads` | `threads=128` | 无（一维 Kernel） | §0.5 舍弃（`TRAP-threads-kwarg-noop`） |
| 运行时 target | CUDA（`compile_flags`） | `target="npuir"` | §4.7 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/conv1d/probe/conv1d_geomm_probe.py` | **高度相似（同算子族的可行性探针）** | `T.Scope("Cube")` + `T.alloc_L1/L0C` + `T.gemm(size=[...])`；per-tap slice 形态 `T.copy` 承载重叠/跨步 im2col；`target="npuir"` + Expert；golden = `F.conv1d` |
| `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_pointwise_kernel_DESIGN.md` | 高（K_W=1 的同族简化版） | Expert Cube 结构、`T.gemm(initC=(k==0), size=[M,K,N])`、MixCV bias 双槽 4-flag 握手（含 Stage 3 修正：前导置消费者侧、bias 显式 vcast） |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高（Cube Expert + MixCV + persistent） | 一维 `T.Kernel(NUM_KERNELS)` + `T.serial(num_local_tasks)` + `cid` 钳位；`T.Scope("Cube")`/`T.Scope("Vector")` 双 Scope；`T.rs("PIPE_*")` 手动同步；`T.sync_block_set/wait` 跨引擎握手；分形 clamp `T.max(16, T.ceildiv(...)*16)` |
| `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py` | 高（Cube Expert + Vector epilogue + 深度 2 双槽握手） | `T.gemm(..., initC=(k==0), size=[...])`；L0C→GM ws 中继；`T.vbrc`/`T.vadd`/`T.vcast` Vector 段；尾块 clamp；Expert `T.copy` 不跨 dtype（须显式 `T.vcast`）；深度 2 双槽 4-flag（消费者侧前导） |
| `examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py` | 高（同源 builder 的 NPU 参考副本） | `_conv1d_kernel` 源 builder 全文（param/tiling/data-flow 契约） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(n, c_in, l_in)` | `float16 \| bfloat16` | 输入（NCL） |
| `weight` | `(c_out, c_in, kernel_l)` | 同 `x` | 权重（Conv1d 原生布局；weight-layout 在 kernel 内完成，见 §0.6 R3） |
| `bias`（可选） | `(c_out,)` | 同 `x` | 逐输出通道偏置（仅 `has_bias=True`） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(n, c_out, out_l)` | 同 `x` | 输出（NCL），JIT `out_idx=[2]` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `l1_a` | `(block_M, block_K)` | input dtype | L1 | A 操作数（w_flat tile），ND2NZ |
| `l1_b` | `(block_K, block_N)` | input dtype | L1 | B 操作数（im2col tile），ND2NZ |
| `l0_c` | `(block_M, block_N)` | float32 | L0C | fp32 累加器（`T.gemm` dst） |
| `w_flat` | `(c_out, kernel_l, c_in)` | input dtype | GM | weight-layout 前置产物（K 序 `(k_w,c_i)`；作为 prim_func 形参传入） |
| `y_poly` | `(n, c_in, stride_l, Lg)` | input dtype | GM | polyphase 物化（仅 stride>1；作为 prim_func 形参传入） |
| `ws` | `(num_kernels, 2, block_M, block_N)` | float32 | GM | Cube→Vector bias 中继（深度 2 双槽；仅 bias 路径；作为 prim_func 形参传入） |
| `wsrc` | `(bm_w, c_in, kernel_l)` | input dtype | UB | weight 前置载入缓冲 |
| `wdst` | `(bm_w, kernel_l, c_in)` | input dtype | UB | weight 前置转置结果 |
| `ub_f32` | `(block_M, block_N)` | float32 | UB | bias 相加缓冲（仅 bias 路径） |
| `ub_f16` | `(block_M, block_N)` | input dtype | UB | cast 后输出缓冲（仅 bias 路径） |
| `bias_16` | `(block_M,)` | input dtype | UB | 本 tile 的 bias 输入 dtype 向量（仅 bias 路径） |
| `bias_col` | `(block_M, 1)` | float32 | UB | bias 的 fp32 列向量（仅 bias 路径） |
| `bias_brc` | `(block_M, block_N)` | float32 | UB | bias 广播缓冲（`vbrc`，仅 bias 路径） |

### 4.4 内存搬运路径

```
前置（Vector，一次）:
GM[weight] --T.copy--> UB[wsrc] --T.transpose([0,2,1])--> UB[wdst] --T.copy--> GM[w_flat]
（stride>1）GM[x] --T.Parallel gather--> GM[y_poly]

主循环（Cube）:
GM[w_flat] --slice T.copy(ND2NZ)--> L1[l1_a] --+
GM[x] 或 GM[y_poly] --per-tap slice T.copy(ND2NZ)--> L1[l1_b] --T.gemm(fp32)--> L0C[l0_c]
    --[无 bias] slice T.copy(fixpipe f32→f16)--> GM[out]
    --[有 bias] T.copy(fixpipe f32→f32)--> GM[ws]

bias epilogue（Vector）:
GM[ws] --T.copy--> UB[ub_f32] --vcast(bias_16)+vbrc+vadd(fp32)--> UB[ub_f32]
    --vcast(rint)--> UB[ub_f16] --T.copy--> GM[out]  （bias_16 由 GM[bias] 载入）
```

**规则**：GM→L1 自动 ND2NZ（等价 `T.load_nd2nz`；要求区域最内维连续）；L0C→GM 自动 fixpipe（f32→f16 量化 / f32→f32 直存）；GM↔UB 通用路径要求同 dtype（Expert `T.copy` 不跨 dtype，`TRAP-C12-copy-dtype-cast`），故 bias 路径中 fp32→fp16 由显式 `T.vcast` 承担，输入 dtype→fp32 由 `T.copy`(同 dtype) + `T.vcast` 两步承担。bias 路径的 `GM[ws]` 为深度 2 双槽（`ws[kernel_id, slot]`，`slot = task_id % 2`）——详见 §7.2（4-flag 双向握手）。

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

- weight 前置段（`bm_w=16`）：`wsrc` + `wdst` 各 `16·c_in·K_W·eb`。以最大 `c_in·K_W` 计（encodec-deep：`c_in=128`,`K_W=10`，fp16）→ `16·1280·2 = 40960 B` ×2 = **80KB**（该段与 bias 段不同时驻留，可复用缓冲）。
- bias 段：`ub_f32`(32KB) + `ub_f16`(16KB) + `bias_brc`(32KB) + `bias_16`(64B) + `bias_col`(256B) ≈ **80KB**。
- polyphase 段仅做 DMA 索引写（无大驻留缓冲）。

说明：按 `CONST-capacity-910B2C`「显式 alloc 结构实测 ratio ≈0.98–1.12」口径，各段 80KB 显式分配实际需求 ≈ 80–90KB ≤ 192KB，余量充分。若 Stage 4 增大 `block_M` 至 128，bias 段升至 ~144KB（×1.12 ≈ 161KB，仍 <192KB）——届时复用 `bias_brc`↔`ub_f32` 可降至 ~112KB（§9 风险点）。

**GM 工作区（不计入片上容量）**：`w_flat` `c_out·K_W·c_in·eb`（whisper：1280·3·80·2 = 0.61MB）；`y_poly` `n·c_in·s_W·Lg·eb`（仅 stride>1）；`ws` `num_kernels·2·block_M·block_N·4 = 20·2·64·128·4 = 1.31MB`。均占设备全局内存，不改变 §4.5 片上预算；wrapper 侧按 JIT shape `torch.empty` 分配。

### 4.6 动态轴定义

无动态轴。`n / c_in / l_in / c_out / kernel_l / stride_l / pad_left / pad_right / dilation_l / has_bias` 均为 JIT 编译期参数（每 shape 一特化）；`w_flat`/`y_poly`/`ws` shape、`Lg`、`r_kw`/`q_kw` 相位、尾块尺寸均由编译期常量经 `T.min` 求值。所有循环边界（`num_local_tasks`、`ceil(k_total/block_K)`）为编译期常量，满足「核内串行边界静态」。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
```

并以 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")`（或 wrapper `ascend_mode = "Expert"`）声明 Expert 模式（对齐探针 `examples/conv1d/probe/conv1d_geomm_probe.py`）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：混合 Cube + Vector（Cube 主计算 + Vector 前置 weight-layout / polyphase + Vector bias epilogue）。

**判定依据**：主体为矩阵乘 `w_flat @ im2col`（MAC 密集）→ Cube 单元；weight-layout 重排、stride>1 polyphase 物化、bias epilogue 在 Vector。判定依据 = `T.gemm` 主导 + Vector 前置/后处理（§0.6 R3/R4/R5）。

### 5.2 Block 划分

```python
block_M, block_N, block_K = 64, 128, 128   # 承接源 default_config（Conv1dKernel.default_config）
bm_w = 16                                   # weight 前置 co 分块（UB 预算）
GX = ceildiv(out_l, block_N)                # N 方向（L_out）
GY = ceildiv(c_out, block_M)                # M 方向（C_out）
num_logical = GY * GX * n                   # 逻辑核数
```

- **选择理由**：`block_M=64, block_N=128, block_K=128` 为源算子 `default_config` 取值（`weight_shared [block_m, block_k]` / `data_shared [block_k, block_n]`），迁移承接以保持数据流契约；`block_K=128` 使 K 分块数少（`k_total=240` 两块、`k_total=1280`（encodec-deep，kW=10 时 `k_total=1280`）十块）；L0C tile 32KB 留足余量；L1 单槽 48KB、双缓冲 96KB ≪ 512KB。Stage 4 可扫描更大 `block_M`（如 128，L0C 64KB）与 `block_N`。

### 5.3 约束分析

- **对齐约束**：K/N 尾轴分形 ≥16（`ceil16` clamp）；`block_N=128` fp16 尾轴 128·2=256B（32B 对齐 ✓）；`block_K=128` fp16 256B ✓。
- **L0C 容量**：`64·128·4 = 32KB ≤ 128KB` ✓（`CONST-capacity-910B2C`）。
- **L1 容量**：单槽 48KB、双缓冲 96KB ≤ 512KB ✓。
- **UB 容量**：weight 前置段 80KB、bias 段 80KB（≤192KB）✓（§4.5）。
- **分形约束**：`T.gemm` M/N/K 分形 ≥16——`tmc/tkc/tnc = T.max(16, ceil16(tail))` 保证几何合法（§0.6 R2）。

### 5.4 注意事项

- **非整除尾块**：`c_out % block_M`、`out_l % block_N`、`k_total % block_K` 三处非整除均由 §0.6 R2 处理——`T.copy` slice 只搬有效区、`T.gemm` `size` 用 `ceil16` clamp、输出用 slice `T.copy` 裁剪写出。禁止 base+size 形态写尾块（`TRAP-T-copy-region-semantics`）。
- **越界输入零填充**：`il = ol·s_W + kw·d_W - p_L` 越界（<0 或 ≥l_in）时，per-tap slice 只搬 `x` 的有效子区间 `[max(0,off), min(l_in, off+tn))`，其余 L1 行由预清零保证为 0（等价源掩码）。
- **K 块内多 tap**：`block_K=128` 与 `c_in` 不必整除（如 `c_in=80`），一个 K 块跨多个 tap；B 装载在 K 块内**逐 tap** slice，行偏移 clamp（§3.3 注 3）。
- **极小 shape**（如 `c_out < block_M` 或 `out_l < block_N`）：`ceil16` clamp 到 16 保证分形合法，多余填充列被裁剪不写回。
- **batch n**：`num_logical = GY·GX·n`，n 折叠进一维 `cid`（`bz = cid // (GY·GX)`）。
- **stride>1**：polyphase 物化 `y_poly`（§0.6 R4），B 装载改从 `y_poly` 取；stride=1 直接在 `x` 上 slice（省一次物化）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：`num_logical_kernels = ceil(c_out / block_M) × ceil(out_l / block_N) × n = GY × GX × n`（本算子输出网格 `[n, c_out, out_l]`，M=C_out、N=L_out、batch=n）。以代表 workload 计：
  - whisper-conv1（N=1, C_in=80, L_in=3000, C_out=1280, K_W=3, stride=1, pad=0）：`out_l=2998`，`GY=ceil(1280/64)=20`，`GX=ceil(2998/128)=24`，`num_logical = 20×24×1 = 480`。
  - encodec-deep（N=1, C_in=128, L_in=600, C_out=256, K_W=10, stride=1）：`out_l = 600−9=591`，`GY=4`，`GX=ceil(591/128)=5`，`num_logical = 20`。
  - wav2vec2-layer1（N=1, C_in=1, L_in=16000, C_out=512, K_W=10, stride=5）：`out_l = (16000−9−1)//5+1 = 3199`，`GY=8`，`GX=25`，`num_logical = 200`。
  - smoke `test_conv1d_no_bias_matches_torch`（N=1, C_in=32, L_in=256, C_out=64, K_W=5, stride=2, pad=2）：`out_l = (256+4−4−1)//2+1 = 128`，`GY=1`，`GX=1`，`num_logical = 1`。

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

  - **whisper-conv1（num_logical=480）**：`480 = 20 × 24`，即**物理核数整数倍（24×）**，每核恰 24 个逻辑块，负载完全均匀、无空转；启动 20 个内核而非 480 个，摊薄核启动开销。
  - **encodec-deep（num_logical=20）**：`20 = 20 × 1`，**1×**，每核 1 块，无串行。
  - **wav2vec2-layer1（num_logical=200）**：`200 = 20 × 10`，**10×**，每核 10 块，均匀。
  - **smoke stride2（num_logical=1）**：`1 ≤ 20`（**逻辑核数 ≤ 物理核数**），`num_local_tasks` 对 `kernel_id=0` 为 1、其余为 0——**无需适配**（结论：逻辑核数 ≤ 物理核数，直接映射，无串行超发）。
  - 规模判定：本算子逻辑核数随 shape 变化，横跨「无需适配」到「中等规模」档；统一 persistent 结构下，内核总数 = 物理核数（1×，整数倍），无负载不均，核内串行边界 `num_local_tasks` 为编译期常量（静态）✓。极大规模（`num_logical ≫ 20`，如 batch 大 / 长序列）同样由该结构覆盖（每核任务数增加，仍为「物理核数 × 1」）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 矩阵乘由 Cube 单元承接、weight-layout/polyphase/bias 由 Vector 承接——本表循环仅 block 级 / tile 级调度循环与 K 分块累加；逐元素计算点除 polyphase 物化索引（§1.6.2 #7，跨步 gather API 缺失）外均已向量化，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级并行 | 核级并行 | `T.Kernel(num_kernels, is_npu=True) as (kernel_id, _)` | 每核处理若干 tile（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(num_local_tasks)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| K 方向 | K 分块累加 | `T.serial(ceil(k_total / block_K))` | K 维分块，`T.gemm(initC=(k==0))` 累加（tile 级循环，非逐元素） |
| K 块内 tap | 核内串行 | `T.serial(kernel_l)`（k 块内逐 tap slice B） | per-tap slice 装载（tile 级，非逐元素） |
| weight 前置 | Vector 串行 + 搬运 | `T.serial(ceil(c_out/bm_w))` + `T.transpose` | 一次性 layout 重排（非每 tile） |
| polyphase 物化 | Vector 多元素并行 | `T.Parallel(Lg, stride_l)`（仅 stride>1） | 逐元素索引写（跨步 gather API 缺失，§1.6.2 #7） |
| 矩阵计算 | Cube | `T.gemm(l1_a, l1_b, l0_c, initC, size=[M,K,N])` | MAC 密集 → Cube 分形（§1.6.3 操作数轴） |
| epilogue（bias） | Vector | `T.vbrc` + `T.vadd` + `T.vcast` | 逐元素加 bias 全部向量化（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
    with T.Scope("Vector"):                               # 阶段 A：一次性前置
        for co0 in T.serial(ceil(c_out / bm_w)):
            T.copy(weight -> wsrc); T.transpose(wsrc, wdst, [0,2,1]); T.copy(wdst -> w_flat)
        if need_poly:
            for bn, ci in serial: T.Parallel(Lg, stride_l) -> y_poly gather
        with T.rs("PIPE_MTE3"): T.sync_block_set(4)        # layout/polyphase ready
    with T.Scope("Cube"):
        with T.rs("PIPE_MTE2"): T.sync_block_wait(4)       # 等前置就绪
        for task_id in T.serial(num_local_tasks):          # 核内串行（静态边界）
            cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
            bz, by, bx = decode(cid)
            for k in T.serial(ceil(k_total / block_K)):     # K 分块累加
                with T.rs("PIPE_MTE2"):
                    load l1_a from w_flat
                    for kw in T.serial(kernel_l): load per-tap l1_b (x 或 y_poly)
                with T.rs("PIPE_C"): T.gemm(..., initC=(k == 0))
            if not has_bias:
                with T.rs("PIPE_FIX"): store l0_c -> out
            else:
                slot = task_id % 2
                with T.rs("PIPE_FIX"):
                    T.sync_block_wait(2*slot + 1)          # 写前等该槽空闲（WAR）
                    store l0_c -> ws[kernel_id, slot]
                    T.sync_block_set(2*slot)               # 写后置位：ws 就绪
    if has_bias:
        with T.Scope("Vector"):
            with T.rs("PIPE_MTE3"): T.sync_block_set(1); T.sync_block_set(3)  # 前导（消费者侧）
            for task_id in T.serial(num_local_tasks):
                slot = task_id % 2
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(2*slot)              # 读前等 ws 就绪
                    load ws, bias(16) -> ub_f32
                vcast(bias) + vbrc + vadd + vcast
                with T.rs("PIPE_MTE3"):
                    store ub_f16 -> out
                    T.sync_block_set(2*slot + 1)           # 消费后归还槽（WAR 保护）
```

### 6.3 流水线优化

Stage 3 基线用 `T.serial`（正确性优先）；Stage 4 可对 K 分块循环启用 `T.Pipelined(num_stages=2)` 或 L1 ping-pong 双缓冲（两套 `l1_a`/`l1_b` 槽），将 GM→L1 装载隐藏于 Cube 计算窗口（源 `T.Pipelined` 的 NPU 等价，`PL-1.12-task-pipeline-depth2`）。bias 路径的 Cube↔Vector 交接**已在基线采用深度 2 双槽**（`ws[kernel_id, task_id % 2, ...]` + 4-flag 双向握手，§7.2）——Cube(task t) 与 Vector(task t-1) 任务级重叠；Stage 4 可加深（`slots>2`）。weight-layout / polyphase 前置段为一次性，其耗时可与首个 tile 的 K 装载流水重叠（依赖 flag 4 的等待点尽量靠后）。

### 6.4 尾块处理

非整除 shape（`c_out % block_M`、`out_l % block_N`、`k_total % block_K`）：`tm/tn/tk = T.min(block, remain)` 决定有效区；`T.copy` slice 只搬有效区（越界补 0）；`T.gemm` 的 `size` 用分形 clamp `tmc/tkc/tnc = T.max(16, ceil16(tail))`；输出用 slice `T.copy(l0_c[0:tm,0:tn], out[...])` 裁剪写出（填充列不落 GM）。输入越界（`il<0 || il>=l_in`）由 per-tap slice 只搬有效子区间 + L1 预清零实现零填充。K 块内 tap 行偏移 clamp 到 `[0, block_K)`。所有尾块搬运走 slice 形态（`TRAP-T-copy-region-semantics` / `TRAP-load-nd2nz-strided`）。

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
| Vector 前置完成（weight-layout / polyphase） | `T.sync_block_set(4)`（`with T.rs("PIPE_MTE3")`，Vector 生产者） | 置位「layout/polyphase 就绪」，通知 Cube 可读 `w_flat`/`y_poly`（跨引擎生产者→消费者） |
| Cube 首次读 `w_flat`/`y_poly` 前 | `T.sync_block_wait(4)`（`with T.rs("PIPE_MTE2")`，Cube 消费者） | 等 Vector 前置写完才读（跨引擎就绪握手） |
| Cube 段前导（bias 路径） | `T.sync_block_set(1)` / `T.sync_block_set(3)`（`with T.rs("PIPE_MTE3")`，**Vector 消费者 Scope**） | 置两槽初始空闲；使 Cube 首两个任务（slot 0/1）的写前等待立即通过——**必须在消费者（Vector）引擎置位**（同引擎自 set→自 wait 不 retire；`TRAP-sync-self-set-wait` / 函数 1 Stage 3 修正） |
| Cube 写 `ws` **前**（WAR 保护） | `T.sync_block_wait(2*slot+1)`（`with T.rs("PIPE_FIX")`） | 等 Vector 归还该槽（上一轮用完）后才覆盖 `ws[kernel_id, slot]`——消除数据竞争的关键一环 |
| Cube 写 `ws` **后**（就绪通知） | `T.sync_block_set(2*slot)`（`with T.rs("PIPE_FIX")`） | 置位该槽 ws 就绪，通知 Vector 可读（跨引擎生产者→消费者握手） |
| Vector 读 `ws` **前**（就绪等待） | `T.sync_block_wait(2*slot)`（`with T.rs("PIPE_MTE2")`） | 等 Cube 本任务写完该槽后才读 `ws[kernel_id, slot]` |
| Vector `ws` 读取/写出搬运 | `with T.rs("PIPE_MTE2")` / `with T.rs("PIPE_MTE3")` | 标注 GM↔UB 搬运段（读 ws/bias、写 out） |
| Vector 消费后（槽归还） | `T.sync_block_set(2*slot+1)`（`with T.rs("PIPE_MTE3")`） | 通知 Cube 该槽已空闲，可被下一轮覆盖（WAR 保护的另一半） |

> 协议概要（§3.2/§3.3/§6.2/§7.2 单一来源）：前置就绪 flag `4`（Vector 写后 `set` / Cube 读前 `wait`）；`slot = task_id % 2`；bias 前导由 **Vector** `set(1); set(3)` 置两槽空闲；每任务 Cube `wait(2*slot+1)` → 写 ws → `set(2*slot)`；Vector `wait(2*slot)` → 读 ws + bias 后处理 → 写 out → `set(2*slot+1)`。每个 flag id 在生产/消费间严格 set/wait 交替，深度 2 使 Cube(T) 与 Vector(T-1) 并发而 slot 不冲突（依据 `_ssd_chunk_scan_fwd_kernel.py` 深度 2 双槽 4-flag 先例 + 函数 1 Stage 3 消费者侧前导修正）。

### 7.3 pass_configs 配置

Expert 模式默认；bias 路径需保证 Cube/Vector 双 Scope 的交叉流水按显式 flag 编排（`PL-1.16-expert-dualscope-bypass`：Expert 双 Scope + pass_configs 硬边界）。基线不引入额外 `pass_configs`；Stage 4 若启用自动多缓冲，须核对 `auto-multi-buffer` 对 L1/UB 预算的膨胀（`CONST-capacity-910B2C` / `TRAP-UB-multibuffer-inflation`）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据（`out[n,c_o,l] = Σ_{c_i,k_w} W[c_o,c_i,k_w]·x[n,c_i,l·s_W+k_w·d_W-p_L] + bias[c_o]`，fp32 累加、fp16/bf16 输出），移植源仓参考实现（`torch.nn.functional.conv1d`），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, dtype="float16"):
    """一般 Conv1d 参考实现（源仓测试基准同款：examples/TileOPs/tests/ops/test_conv1d.py）。

    以 §0.1 语义为依据：weight [C_out, C_in_g, K_W] -> F.conv1d（groups=1）；
    在 fp32 域计算（对齐源码 fp32 累加 + fp32 bias 相加），最后 cast 回输入 dtype。
    与 §0.6 的 Cube/persistent/polyphase 实现完全独立。
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    out_f32 = F.conv1d(
        x_cpu.to(torch.float32), w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=stride, padding=padding, dilation=dilation, groups=1,
    )
    return out_f32.to(x_cpu.dtype)
```

> 说明：`F.conv1d` 与 builder 契约的 `weight [C_out, C_in, K_W]` 一致；golden 在 fp32 域计算，最后 cast 回输入 dtype——与源算子累加域一致（源 `accum_dtype="float"`）。

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓 `convolution_verification` 容差口径一致；K=`C_in·K_W` 累加在 fp32，分形累加顺序与源分块顺序略有差异，落在容差内。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：代表 workload
- `whisper-conv1`（N=1, C_in=80, L_in=3000, C_out=1280, K_W=3, stride=1, pad=0，`float16`/`bfloat16`）；
- `wav2vec2-layer1`（N=1, C_in=1, L_in=16000, C_out=512, K_W=10, stride=5，覆盖 **stride>1 polyphase 路径**）；
- smoke `test_conv1d_no_bias_matches_torch` / `test_conv1d_bias_matches_torch`（N=1, C_in=32, L_in=256, C_out=64, K_W=5, **stride=2, pad=2**，覆盖 stride>1 + padding）。

`randn` 输入 + 随机 `weight`/`bias`，与 `golden_conv1d`（§8.1）在 §8.2 容差下 `torch.testing.assert_close`；分别覆盖 `has_bias=False` 与 `has_bias=True` 两条路径。完整分层套件（L1 非整除 / 多 shape / batch>1、L2 边界 size、Boundary 极小/极大 shape 与 `C_in/C_out/L` 非 16 整除）交由 `tilelang-op-develop` 规划，本设计不枚举。

**L0 必测点**：① 非整除尾块（`C_out=100`、`L=1000`、`C_in=50`、`k_total` 非 `block_K` 整除组合）；② `has_bias` 两条路径一致；③ batch>1（`n` 折叠进 `cid`）；④ **stride>1 + padding**（polyphase 路径与源掩码一致性）；⑤ dilation>1（slice 起点平移）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| Expert Cube 依赖工具链版本 | Cube Expert（`T.Scope("Cube")` + L1/L0C + `T.gemm` + `T.transpose`）的全部结论绑定当前工具链（tilelang `0.1.2+417dae99` + CANN，`PL-*`/`CONST-*` 条目版本戳）；工具链升级后须重验（探针已通过为当前证据） |
| 物理核数 20（非 24） | 本机实查 `get_aicore_num()=20`（设备 Ascend 910B3），与 `CONST-aicore-910B2C`（24）不一致——设计以实查为准（§5.5）；`num_kernels` 由 `get_aicore_num()` 在 JIT 期求值，跨设备自适应 |
| `T.gemm` 分形约束 | M/N/K 分形 ≥16——`tmc/tkc/tnc = T.max(16, ceil16(tail))` 保证；`c_in` 极小需补零到 16（K 填充，对输出无影响） |
| 跨引擎传输强制 GM 往返 | bias 路径 Cube→Vector 经 GM ws 中继；weight-layout / polyphase 亦经 GM workspace（`CONST-store-fixpipe-gm-only`）——这是 NPU 上把 layout/后处理放 Vector 的固有代价（见 §1.6.0 设计期估算） |
| 跨引擎 ws 握手须双向完整 | 单槽握手若缺 Cube 侧写前等待会产生 WAR 数据竞争；本设计固定**深度 2 双槽 + 4-flag 双向握手**（§3.2/§3.3/§6.2/§7.2 一致）。**且「槽空闲」前导必须置于消费者（Vector）引擎**——同引擎自 set→自 wait 不 retire（`TRAP-sync-self-set-wait`，函数 1 Stage 3 实测 aicore timeout 507014） |
| ND2NZ 区域最内维须连续 | stride>1 的 im2col 列采样非连续，不能直接 GM→L1 slice——必须 polyphase 物化后 slice（§0.6 R4 / CG-2026-0004） |
| polyphase 物化索引为标量路径 | 跨步 gather 无向量指令（`TRAP-C10-strided-gather`），`y_poly` 物化为逐元素（~1ns/elem）；仅 stride>1 触发，代表 shape 小（`y_poly ≈ x` 体量）。若出现大 `c_in·l_in` 的 stride>1 workload，须评估该点（Stage 4） |
| weight-layout 前置开销 | `T.transpose` 一次性 `O(c_out·c_in·K_W)` 搬运；以 `bm_w` 分块控制 UB（§4.5）。若 `c_out·c_in·K_W` 极大，须评估其对小 workload（smoke）的相对占比（Stage 4） |
| L0C/L1 容量 | L0C 32KB ≤ 128KB、L1 单槽 48KB ≤ 512KB（§4.5）——余量充分；若 Stage 4 增大 `block_M/block_N` 须复算（`block_M·block_N·4 ≤ 128KB` → `block_M·block_N ≤ 32768`） |
| UB 容量 | weight 前置段 80KB、bias 段 80KB ≤ 192KB（§4.5）；若 Stage 4 增大 `block_M` 至 128 则 bias 段 ~144KB（×1.12 ≈ 161KB）——须复用 `bias_brc`↔`ub_f32` 或减小 tile |
| `T.load_nd2nz` 跨步误读 | x 装载为跨步区域（行 stride = `l_in`）——`TRAP-load-nd2nz-strided` / CG-2026-0004：`load_nd2nz` / base+size `T.copy` 对跨步区域静默平坦误读；本设计一律用 slice 形态 `T.copy`（探针实测绕法） |
| Expert 通用路径不跨 dtype | `T.copy` 在 Expert 通用路径（GM↔UB）要求同 dtype（`TRAP-C12-copy-dtype-cast` / `T.copy` 文档 §2.3）——bias 路径 fp32→fp16 必须显式 `T.vcast`，输入 dtype→fp32 亦须两步 |
| 尾块越界写 | `TRAP-T-copy-region-semantics`（越界写坏相邻 GM / `[N,1]` 越界读）——所有尾块搬运走双显式 slice，禁止 base+size 形态 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C 溢出 | `block_M·block_N` 过大（>32768） | 编译失败（`cc overflow`） | 维持 `64×128`（32KB）或 `128×256` 上限（128KB） |
| L1 溢出 | `block_M/block_N/block_K` 过大或缓冲槽过多 | 编译失败（`cbuf overflow`） | 维持 `64/128/128`（单槽 48KB）；双缓冲 ≤96KB |
| 尾块 MTE 越界写 | 用 base+size 形态写尾块 | 写坏相邻 GM tensor | 双显式 slice `T.copy(src[0:tm], dst[0:tm])`（§6.4） |
| `T.gemm` 分形非法 | tail < 16 或非 `ceil16` | 编译/运行错误 | `T.max(16, ceil16(tail))` clamp（§0.6 R2） |
| `T.gemm` dst 误设 fp16 | `l0_c` 设 `float16` | 运行卡死（`T.gemm.md` §2.3） | `l0_c = T.alloc_L0C([...], "float32")` 固定 fp32 |
| bias 精度偏差 | bias 在 fp16 域相加（cast 后再加） | 与 golden（fp32 加 bias）偏差偏大 | bias 在 fp32 域相加（`vadd` fp32）后再 `vcast`（§0.6 R5） |
| Expert `T.copy` 跨 dtype | GM↔UB 拷贝 fp32↔fp16 | 编译报 `does not support element type casting` | 显式 `T.vcast` 承担转换（§9.1） |
| idle 核越界 | `num_logical < num_kernels` 时 `T.ceildiv` 对负数返回 1 | idle 核运行越界任务 | `cid = T.min(..., num_logical-1)` 钳位（§3.3 注 5，gqa 先例） |
| 跨引擎 ws 数据竞争 | 单槽 ws 且 Cube 写前无 `sync_block_wait`（WAR 未保护） | Vector 未读完即被覆盖 → 随机精度错误 | 深度 2 双槽 + 写前 `wait(2*slot+1)`（§3.3/§6.2/§7.2 一致） |
| 跨引擎握手死锁 | 前导 `set(1)/set(3)` 置于生产者（Cube）侧 | aicore timeout 507014（同引擎自 set→自 wait 不 retire） | 前导置于**消费者（Vector）**Scope（§7.2 / `TRAP-sync-self-set-wait`） |
| stride>1 ND2NZ 误读 | 直接对 strided 列采样做 GM→L1 slice | 静默平坦误读（整块错值） | polyphase 物化后 slice（§0.6 R4 / CG-2026-0004） |
| `T.transpose` 多于两轴 | `permutation` 非「交换两轴」 | 编译失败 | 仅用两轴交换 `[0,2,1]`（`T.transpose.md` §2.3） |
| `dtype=` 关键字 false alarm | `T.Tensor(..., dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（§3.3 注 1） |

### 9.3 特殊场景处理

- **非整除 shape**（`C_in/C_out/L_out` 非 block 整数倍或非 16 整数倍）：分形 clamp + slice 尾块 + 裁剪 store（§6.4）。
- **极小 shape**（`C_in < 16`）：K 补零到 16（填充项对输出无影响）。
- **batch>1**：`n` 折叠进一维 `cid`（`bz = cid // (GY·GX)`）；各 batch tile 独立，无跨 batch 依赖。
- **`has_bias=False`**：无 Vector bias 段，无 ws、无 4-flag 握手；仍有 Vector weight-layout 前置段（stride>1 时另有 polyphase 段）。
- **`has_bias=True`**：MixCV，fp32 ws 中继（**深度 2 双槽 + 4-flag 双向握手**，§7.2，前导置消费者侧）+ Vector bias epilogue（§0.6 R5）。
- **stride=1**：B 直接从 `x` per-tap slice（省 `y_poly` 物化）。
- **stride>1**：先物化 `y_poly`（§0.6 R4），B 从 `y_poly` slice；smoke `stride=2`、`wav2vec2-layer1`（stride=5）覆盖。
- **dilation>1**：slice 起点平移 `kw·d_W`，不改变装载结构（stride=1 路径直接生效；stride>1 路径经 polyphase 相位）。
- **`c_in < block_K` / `k_total < block_K`**：单 K 块，`tk=k_total`，`tkc=ceil16(k_total)`。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/conv1d/_conv1d_kernel/
├── DESIGN.md               # 本设计文档
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `verify_equiv.py` | —（不适用） | §1.6.1 结论为「无优化空间」，按 gate `S1-EQUIV-EXEC` 豁免规则不生成 |
| `_conv1d_kernel.py` | ⬜ 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_conv1d.py` | ⬜ 待实现（Stage 3） | 精度测试（L0/L1/L2/Boundary） |

### 10.3 命名规范

- 项目目录名：`conv1d`；算子目录名：`_conv1d_kernel`；接口函数：`_conv1d_kernel`（`Conv1dFwdOp` 的 5 个迁移 builder 之一）；JIT `target="npuir"`，`out_idx=[2]`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ Stage 1 复盘（RETROSPECTIVE.md）
2. ⬜ Golden 函数（§8.1，`F.conv1d`）
3. ⬜ stride=1 无 bias 纯 Cube 实现（weight-layout + per-tap slice + `T.gemm`）→ L0 精度比对
4. ⬜ stride=1 bias 路径（MixCV 双 Scope + ws 中继 + 4-flag 握手，前导置 Vector）→ L0 精度比对
5. ⬜ stride>1 polyphase 路径（smoke `stride=2,pad=2`）→ L0 精度比对
6. ⬜ 非整除尾块 / batch>1 / dilation>1 覆盖（L1/L2/Boundary）
7. ⬜ Stage 4 调优（L1 双缓冲、`block_M` 扫描、bias ws 双槽加深、weight 前置与首个 tile 重叠）
