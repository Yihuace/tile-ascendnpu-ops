# _fused_add_layer_norm_kernel 算子设计文档

> 迁移任务（harness 模式）：FusedAddLayerNormFwdOp（融合 residual add + LayerNorm，双输出）。
> 源算子（GPU 原始源码）：`TileOPs-main/src/tileops/kernels/norm/fused_add_norm.py`（`FusedAddLayerNormKernel`，L48-141）。
> Stage 0 提取件（本函数迁移的精确源）：`examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/_fused_add_layer_norm_fwd_kernels.py`。
> NPU wrapper/op/test/bench：`examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm.py` / `examples/TileOPs/tileops/ops/norm/fused_add_layer_norm.py` / `examples/TileOPs/tests/ops/test_fused_add_layer_norm.py` / `examples/TileOPs/benchmarks/ops/bench_fused_add_layer_norm.py`。
> 工具链版本戳：tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（2026-10-10 build）+ CANN 8.5.2 + Ascend910B3（`NPUUtils.get().get_aicore_num()` 实查 = 20）+ torch 2.7.1+cpu + torch_npu 2.7.1.post4。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

**数学语义**（FusedAddLayerNorm 前向，双输出）：对 `(M, N)` 输入 `x` 与同形 `residual`，先逐元素相加得 `s = x + residual`，再对 `s` 的每一行（最后一维 N）做 layer_norm 并施加向量仿射 `weight`/`bias`：

$$
s_{ij} = x_{ij} + \text{residual}_{ij},\quad
\mu_i = \frac{1}{N}\sum_{j<N} s_{ij},\quad
\text{var}_i = \frac{1}{N}\sum_{j<N}(s_{ij}-\mu_i)^2,\quad
\text{rstd}_i = \frac{1}{\sqrt{\text{var}_i + \varepsilon}}
$$
$$
y_{ij} = (s_{ij}-\mu_i)\cdot\text{rstd}_i\cdot w_j + b_j,\qquad \text{residual\_out}_{ij} = s_{ij}
$$

`weight`/`bias` 是长度 N 的仿射向量（layer_norm 的 scale/shift），不是逐 token 张量。方差为有偏方差（ddof=0）；`eps` 默认 1e-5，加在方差上（**不是**加在标准差上、**不是**加在均方上）。

**双输出契约（本算子与单输出 norm 的核心差异）**：输出有两个——`output`（= `y`，归一化结果）与 `residual_out`（预归一化残差和 `s`）。`residual_out` 的存在使下游 residual 连接可复用该和而不重算（源文件 docstring L6-9）。两输出同 shape、同 dtype 于输入。

**舍入语义（从源码确认，迁移必须保持）**：源码把 fp32 的 `x+residual` 写回**输入 dtype 的 fragment**（`x_local[i,j] = add_f32[i,j]`，隐式截断），`residual_out` 即该舍入后的和；`y` 的中心化也从该舍入后的副本重推（`T.cast(x_local[i,j],"float32") - mean_val[i]`）。即 **residual_out = round_dtype(x + residual)，y = LayerNorm(residual_out)**（「先舍回、再归一化」）。仓库参考实现 `tests/ops/test_fused_add_layer_norm.py:ref_program` 采用完全相同的路径（`added = (x.float()+residual.float()).to(x.dtype)` 后再 `F.layer_norm(added.float(), ...)`）。

**规约语义**：规约轴 = 最后一维（dim=-1），行间完全独立、无跨行依赖；源码在 **fp32** 域做 S1（mean）与 S2（中心化平方和，**中心化两遍式**）。**一处细微差异**：源码的 S1/S2 在**未舍回的** fp32 和 `add_f32` 上计算，而中心化/输出用的是**舍回后的** `x_local`；golden 则在舍回值上算 S1/S2。二者在 fp16/bf16 上相差 ≤ 1 dtype ulp（仅统计量），本设计选择 golden 对齐的「舍回后统计」（§1.6.1、verify_equiv 判别段 B 机器验证在容差内，0 违反）。

**dtype 语义**：中间计算全程 fp32——x/residual 逐元素 cast 到 fp32 相加，weight/bias cast 到 fp32 参与仿射，两输出单次舍回输入 dtype。即「fp32 opmath + 单次舍回」，与 golden `x.float()+residual.float() → .to(dtype) → F.layer_norm(...) → .to(dtype)` 一致。

**边界语义**：
- `M % block_m ≠ 0`：源码 `row_guard` 分支用 `pid_m*block_m + i < M` 谓词掩码保证越界行不读不写；
- N 非 256 对齐：源码 `N_padded = align_up(N, 256)`，越界列零填充参与第一遍 S1（零贡献），中心化遍用 `if_then_else(j<N, (·)², 0)` 显式屏蔽 pad 列；
- NaN/Inf：自然传播（无特殊处理）——全 inf 行 → mean=inf → 输出 NaN；
- eps=1e-5 默认（manifest params）；M≥1、N≥1。

**I/O 契约**：输入 `x`/`residual` 同形 `(M, N)`、`weight`/`bias` `(N,)`、dtype ∈ {float32, float16, bfloat16}、四者同 dtype；输出 `output`/`residual_out` 与 `x` 同 shape 同 dtype（manifest shape_rules：`output.shape == x.shape`、`residual_out.shape == x.shape`），非 in-place。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `x` | 输入 | `(M, N)` | float16 / bfloat16 / float32 | M=非规约维乘积（Op 层把任意前导维 reshape 成 2D），N=hidden dim |
| `residual` | 输入 | `(M, N)` | same_as(x) | 残差，与 x 同形 |
| `weight` | 输入 | `(N,)` | same_as(x) | 仿射缩放向量 |
| `bias` | 输入 | `(N,)` | same_as(x) | 仿射平移向量 |
| `output` | 输出 | `(M, N)` | same_as(x) | 归一化结果 y（`out_idx=[4]`） |
| `residual_out` | 输出 | `(M, N)` | same_as(x) | 预归一化残差和 s（`out_idx=[5]`） |

输出 shape 从源码确认（回填标注依据：源 `main` 签名 L58-65 两输出均 `T.Tensor[(M, N_padded), dtype]`；host 侧 `outputs = [out[:, :self.N] ...]` 裁掉 pad——**无转置、无布局变化**）。

### 0.3 实现算法解读（怎么算）

**源码结构**（GPU，工厂模式）：`_fused_add_layer_norm_kernel(M, N, eps, dtype)` → `N_padded = align_up(N, ALIGNMENT=256)` → `@tilelang.jit(out_idx=[4, 5])` 内 `_func(block_m, threads)` → `@T.prim_func main(x, residual, weight, bias, y, residual_out)`。GPU 默认配置 `select_row_config()` = `{block_m: 1, threads: 128}`。

**计算步骤分解**（源码每个计算语句均已归入；行块 = block_m 行）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| L1 | 装载 x：`row_guard` 分支 `T.Parallel` + `T.if_then_else(pid_m*block_m+i<M, x[...], 0)`；否则整块 `T.Parallel` 直读 | GM x | x_local (bm,N_padded, dtype) | 输入搬运 |
| L2 | 装载 residual：同上（谓词/直读两分支） | GM residual | r_local (bm,N_padded, dtype) | 输入搬运 |
| A1 | fp32 相加：`add_f32[i,j] = cast(x_local)+cast(r_local)` | x_local, r_local | add_f32 (bm,N_padded) fp32 | s = x + residual |
| A2 | 舍回副本：`x_local[i,j] = add_f32[i,j]`（隐式截断回 dtype） | add_f32 | x_local (dtype) | residual_out；且为 y 中心化副本 |
| S1 | 行和：`T.reduce_sum(add_f32, acc, dim=1)`；均值 `mean_val[i] = acc[i] / float(N)`（除以真实 N；pad 列为 0 不贡献） | add_f32 | acc、mean_val | mean |
| S2 | 中心化平方（in-place）：`add_f32[i,j] = if(j<N, (add_f32[i,j]−mean)², 0.0)`；`T.reduce_sum`；`rstd[i] = T.rsqrt(acc[i]/float(N)+eps)` | add_f32、mean_val | acc、rstd | var（pad 列显式置零） |
| E1 | 输出 y：`y[...] = cast((cast(x_local,f32) − mean)·rstd·cast(weight[j],f32) + cast(bias[j],f32), dtype)`（谓词限定写出） | x_local、mean、rstd、weight、bias | y | y 公式 |
| E2 | 写出 residual_out：`residual_out[...] = x_local[i,j]`（谓词限定写出） | x_local | residual_out | residual_out = s |

**数据流与内存访问模式**（源硬件视角）：

```
GM[x] --T.Parallel(谓词/直读)--> FRAG[x_local](dtype, bm,N_padded)
GM[residual] --T.Parallel--> FRAG[r_local](dtype)
  --(逐元素 cast+add)--> FRAG[add_f32](fp32)
  --(舍回副本)--> FRAG[x_local](dtype)  [residual_out 源 + y 中心化副本]
  --T.reduce_sum(dim=1)--> FRAG[acc] --> FRAG[mean_val]
  --(in-place 中心化平方, pad 置零)--> add_f32 --T.reduce_sum--> acc --> FRAG[rstd](T.rsqrt)
FRAG[x_local] --epilogue 标量复合式(重 cast 重减 + weight/bias 逐列)+隐式截断--> GM[y]
FRAG[x_local] --T.Parallel--> GM[residual_out]
```

**循环与并行结构**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)` 一维 grid（每 block 处理 block_m 行）；block 内 `T.Parallel(bm, N_padded)` 元素级并行（线程映射到行内元素，行内归约经线程级归约塌缩）；GPU 默认 `block_m=1`（`select_row_config`：单行/CTA 规避布局推断塌缩）+ `threads=128`。`row_guard = M % block_m != 0` 仅在 M 非整除时启用谓词分支。

**host 侧逻辑**：`FusedAddLayerNormKernel.__init__` 计算 `N_padded`；`forward` 把 x/residual reshape `(-1, N)`、weight/bias reshape `(N,)`，调用 `_fused_add_layer_norm_kernel(M, N, eps, dtype)(block_m, threads)(...)`；`pad = N_padded - N`，`if pad:` 用 `F.pad` 补零、输出再 `out[:, :N]` 裁剪；`lru_cache(maxsize=32)` 工厂按 (M,N,eps,dtype) 特化。Op 层 `FusedAddLayerNormFwdOp.forward` 做 `contiguous().reshape(-1, N)`、按 `(M,N,dtype,device)` 缓存 kernel、两输出 reshape 回原 shape，`eval_roofline` 返回 `(8*M*N, (4*M*N+2*N)*elem_bytes)`。**无数据内容预处理**（纯 shape 元数据视图 + padding）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 行块 register fragment 驻留（x_local/r_local/add_f32） | 消 GM 重读（x/residual 跨多遍复用） | (bm,N_padded) 块驻留寄存器 | 大寄存器堆 | 硬件强相关 |
| 2 | 融合 residual add 进首遍 | 消一次独立 add 的 GM 往返 | 首遍读 x/res 后直接 fp32 相加 | 无（纯算法层融合） | 可移植 |
| 3 | 双输出复用预归一化和 | 下游 residual 免重算 | `residual_out = x_local`（舍回副本） | 无（接口契约） | 可移植 |
| 4 | 256 元素对齐 padding（N_padded） | 16 字节访问整除、整块向量化搬运 | `align_up(N, 256)` 零填充 | CUDA SMEM/访问对齐约束 | 硬件强相关 |
| 5 | 中心化两遍方差 + pad 列显式置零 | 数值稳定（vs 矩式 E[x²]−E[x]²） | `(x−mean)²` 求和，pad 列 `if j<N else 0` | 无（纯算法层） | 可移植 |
| 6 | fp32 fragment 全程计算 | 精度（fp32 opmath + 单次舍回） | cast 一次、算完截断 | fragment 寄存器 | 硬件强相关 |
| 7 | 融合 epilogue（y 单表达式） | 消中间写回 | `T.Parallel` 复合表达式 + cast 折叠 | 无（模式可移植） | 模式可移植 |
| 8 | 谓词掩码边界 load/store（row_guard） | 越界安全（M 尾行） | `if_then_else` + `T.And` 谓词 | 无（模式可移植，NPU 有更优 slice 形态） | 模式可移植 |
| 9 | block_m=1 + threads=128 配置 | 行内归约均匀性 + 实测最优线程数 | 单行/CTA 结构性规避 AllReduce 塌缩 | CTA/warp 组织 | 硬件强相关 |
| 10 | lru_cache 工厂 + per-shape JIT 特化 | 摊编译开销 | `functools.lru_cache(maxsize=32)` | 无（host 层） | 可移植 |
| 11 | T.Parallel 直读（无 shared 中转） | coalesced 带宽、省一次中转 | 两操作数各读一次、沿线程已持行归约 | memory coalescing | 硬件强相关 |
| 12 | out_idx=[4,5] 双输出自动分配 | 接口契约（两输出） | jit out_idx 列表 | 无 | 可移植（接口契约） |

> 识别不出的优化会被静默丢弃。Stage 0 观察项（threads 移除、lru_cache 丢失、K5-K11 类适配）已并入上表与 §0.5 处置。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据：`migration-analysis.md` §5 三层模型 + §5.3 映射表；`examples/` 佐证；`docs/` 条目；pattern-library 实测条目。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（公式/双输出/规约轴/ddof/eps 位置/舍回语义） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| 融合 residual add（#2） | 算法 | 无 | — | **保留** | 纯算法层融合；首遍读 x/res 后 fp32 相加 |
| 中心化两遍方差 + pad 列置零（#5） | 算法 | 无 | — | **保留**（pad 机制等价替换见下行） | 纯算法层（`migration-analysis` §5.3「分块累积 → 保留」同类）；§1.6.0 调研确认稳定性优于矩式（D-1 机器证据） |
| 行块 fragment 驻留（#1） | 优化 | register/共享内存 | 有（UB 192KB/AIV） | **等价替换** | `T.alloc_shared`（Developer 映射 UB，`docs/Tilelang.language/内存操作/T.alloc_shared.md`）；预算按 192KB/1.7 膨胀裕度重算（§4.5，TRAP-UB-multibuffer-inflation，`constants.md`） |
| fp32 全程计算（#6） | 优化 | fragment 寄存器 | 有（fp32 UB + vcast） | **等价替换** | `T.vcast(rint)` 升 fp32 → fp32 v-prefix 链 → `T.vcast(rint)` 单次舍回（TRAP-fp16-opmath-golden 对齐通解） |
| 融合 epilogue（#7） | 优化 | 无 | 有 | **等价替换** | v-op 链（vmul + vbrc 广播 + vmul + vadd），意图（消中间写回）保留 |
| 256 对齐 padding + pad 列置零（#4/#5 部分） | 优化 | SMEM/访问对齐约束 | 无此约束 | **舍弃** | NPU UB 对齐由编译器管理（32B，`docs/Tilelang.language/内存操作/T.alloc_shared.md`）；slice 精确 N 直接消零填充（logsumexp 设计同款结论）；pad 列置零随 padding 一并消失，恒等式见 §1.6.1 |
| threads 参数 / block×thread 两级并行（#9 部分） | 优化 | CUDA 线程 | 无 | **舍弃 + 重新设计** | TRAP-threads-kwarg-noop（`T.Kernel(threads=)` 在 npuir 无效果）；并行结构 → §0.6 重设计项 R1 |
| block_m=1 结构性默认（#9 部分） | 优化 | CTA 行均匀性 | 担忧不存在 | **重新设计** | NPU reduce 按行硬件归约（`T.reduce_sum` dim=1，`docs/Tilelang.language/规约操作/T.reduce_sum.md`），无跨线程 AllReduce 塌缩；bm 改由 UB 预算表驱动（§0.6 R2） |
| 谓词掩码边界（if_then_else + T.And，`row_guard`） | 优化 | 无 | 有更优形态 | **重新设计** | `T.min` + src/dst 双显式 slice + 垃圾行丢弃（TRAP-T-copy-region-semantics；logsumexp/ada 尾块模式）→ §0.6 R3 |
| 双输出（#12） | 接口 | 无 | — | **保留** | 接口契约（§0.2）；`@tilelang.jit(out_idx=[4,5])` 自动分配两输出（NPU wrapper 已按此移植） |
| lru_cache 工厂 + JIT 特化（#10） | host | 无 | — | **保留** | host 层硬件无关；NPU wrapper 用 Kernel 类实例缓存承接 |
| `T.ptx_cp_async` / `threads=` / `T.sync_threads` / `@T.macro` | 优化 | CUDA 专属 | 无 | **舍弃** | 源码本函数未用 cp.async（仅 RMSNorm 变体用）；threads/sync_threads 后端不适用；`@T.macro` 直接内联展开（与本函数无关，记录以免误抄） |
| 一维 Kernel（GPU `T.Kernel(ceildiv(M,bm), threads=)`） | 优化 | CUDA grid/CTA | 本项目仅支持一维 `T.Kernel` | **重新设计** | 一维 `T.Kernel(num_kernels, is_npu=True)` persistent 分核（§0.6 R1 / §5.5） |

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（grid 超发 → 一维 persistent 分核）**

- **源方案**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)`——逻辑 block 数随 M 超发（M=2048, bm=1 → 2048 个 block），依赖 CUDA 运行时调度；threads 折叠 CUDA 线程并行。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`，`num_kernels = min(ceil(M/block_m), 40)`（40 = 20 AICore × 2，纯 Vector 算子翻倍，§5.5 实查）；核内 `for i in T.serial(num_local_tasks)` grid-stride 任务映射 `block_id = i*num_kernels + cid`，`if block_id < num_logical` 守卫尾任务；所有量为 host 层 Python 编译期常量（PL-1.5 折叠先例）。意图承接：消除超发 block 的串行调度开销（core-split-strategy.md §1 极大规模分支）+ grid-stride 聚集窗口访问对 MTE2 更优（PL-1.6 实测 +3.2%）。
- **语义保持论证**：行间独立（规约轴=行内 N），任务映射是行块集合的双射，每行公式与累加顺序不变；尾任务守卫保证不越界。参照 ada 迁移（`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md` §0.6 R1）与 lerp_tensor 持久化形态（`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py`）。

**重设计项 R2: block_m 语义（结构性 1 → UB 预算表驱动）**

- **源方案**：`block_m=1`（每 CTA 单行，规避跨线程 AllReduce 塌缩）+ threads=128。
- **NPU 新算法**：`_func(block_m=默认表值)`，默认 `bm = min(8, max(1, (UB_SAFE_BYTES − 4N) // (bpe·N)))`，其中 fp16/bf16 `bpe=10`（stage 2 + x_f32 4 + sq_f32 4）、fp32 `bpe=8`（x_f32 4 + sq_f32 4），另加单个 (1,N) fp32 `w1` 缓冲（4N 字节，见 §4.5）；`UB_SAFE_BYTES = 192KB/1.7 ≈ 113KB`（TRAP-UB-multibuffer-inflation 裕度）；工厂层加 UB 预算 guard（lerp E1 模式）。
- **语义保持论证**：bm 只改变每任务行数，不改变行内计算；M % bm ≠ 0 的尾块由 R3 处理。NPU wrapper 默认 `block_m` 由工厂默认值治理（GPU `threads` 为 CUDA 执行模型产物，已移除）。

**重设计项 R3: 边界处理（谓词掩码 + 零填充 → T.min + 双显式 slice + 垃圾行丢弃）**

- **源方案**：`row_guard` 分支 `T.if_then_else(T.And(...), x[...], 0)` 掩码装载 + pad 列置零 + 谓词限定写出；`N_padded` 零填充。
- **NPU 新算法**：统一单路径（无 needs_pad/row_guard 分支）：`real_m = T.min(block_m, M - off_m)`（`docs/Tilelang.language/比较操作/T.min.md` 标量 min）；装载 `T.copy(x[off:off+real_m, 0:N], stage[0:real_m, 0:N])`（src/dst 双显式 slice，dst 零起点——TRAP-UB-dst-align）；v-op 全 buffer 操作（含未初始化垃圾行，规约/广播按行隔离）；写出 `T.copy(stage[0:real_m, 0:N], residual_out[off:off+real_m, 0:N])` 与 `T.copy(stage[0:real_m, 0:N], output[off:off+real_m, 0:N])` 只写有效行。
- **语义保持论证**：垃圾行（real_m..bm）的统计量是垃圾但**按行隔离**（reduce dim=1 逐行、v-op 逐元素/行广播，无跨行混合），输出切片只取有效行（logsumexp 设计 §5.4、ada 设计 §0.6 R3 同款）；N 不做任何 padding，方差直接对真实 N 归约（GPU pad 校正随 padding 一并消失，§1.6.1）。

**重设计项 R4: dtype 路径（fragment 隐式 cast 链 → 显式 vcast(rint) fp32 中转链）**

- **源方案**：`T.Parallel` 内 `T.cast(..., "float32")` 上cast + epilogue 赋值隐式截断回 dtype；`x_local[i,j]=add_f32[i,j]` 隐式舍回。
- **NPU 新算法**：fp16/bf16 输入：`T.vcast(stage, x_f32, round_mode="rint")` 上cast（f16→f32 / bf16→f32 rint，无损）→ fp32 v-prefix 链 → 舍回副本 `T.vcast(x_f32 → stage, rint)` 得 residual_out → `T.vcast(stage → x_f32, rint)` 重入 fp32 做统计 → epilogue `T.vcast(x_f32 → stage, rint)` 单次舍回得 y；fp32 输入：无 cast（同 dtype 直拷直算）。
- **语义保持论证**：fp16/bf16→fp32 精确无损（fp16/bf16 是 fp32 子集，rint 无操作用）；fp32→目标 dtype 单次 RNE 舍回 = golden 的 `.to(x.dtype)` 舍入路径；TRAP-fp16-opmath-golden 实证该「fp32 中转链」与 torch CPU golden（fp32 opmath + 单次舍回）差 ≤1 ulp fp16；**bf16 无 Vector 算术**（vadd/vmul/vsub/reduce_sum 的 bf16 列均为 ×，见 `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 / `T.vadd.md` / `T.vsub.md` / `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1）——fp32 中转是**必需**而非可选。跨 dtype `T.copy` 的隐藏 VCast 不用作 cast 融合（TRAP-C12）。

**重设计项 R5: 仿射逐列向量（weight/bias 逐元素索引 → vbrc 列广播）**

- **源方案**：epilogue 逐元素 `weight[j]`/`bias[j]`（寄存器/线程沿行已持的列索引直接取数）。
- **NPU 新算法**：weight/bias 是 (N,) 逐列因子，而 NPU v-op 的自动广播仅文档化行广播 `[M,N]↔[M,1]` 与标量（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 / `T.vadd.md` §2.2.2），**不含** `[M,N]×[1,N]` 列因子。故把 weight/bias 先装入 (1,N) fp32 缓冲 `w1`，再用 `T.vbrc` 显式展开到 (bm,N) 后参与运算（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2：`src (1,N) → dst (bm,N)` 合法；bf16 亦支持）。`bm≥2` 时展开到复用的 `sq_f32`；`bm==1` 时 `w1 (1,N)` 与 `x_f32 (1,N)` 同形，直接 `T.vmul` 无需 vbrc（规避 TRAP-vbrc-same-shape-empty-broadcast）。
- **语义保持论证**：`vbrc` 把逐列因子复制到每一行 = 行广播的列因子形态；每行乘/加同一 `w_j`/`b_j`，与源逐元素索引等价（PL-1.15-broadcast-perf-tax：未文档化列广播可编译但有 +12~16% 税，故走 vbrc 显式展开）。

**舍弃项的意图承接汇总**（不得静默丢弃，migration-analysis §6.4）：
- padding（#4）意图 = 向量化整块搬运 → UB 自然 32B 管理 + slice copy 承接；
- 谓词掩码（#8）意图 = 越界安全 → `T.min` + 双显式 slice + 垃圾行丢弃承接；
- threads（#9）意图 = 块内并行度 → Vector lane 向量化（§1.6.2/§1.6.3）承接。

### 0.7 标杆实现

- 源算子：`examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/_fused_add_layer_norm_fwd_kernels.py`（提取件，迁移精确源）；GPU 原始：`TileOPs-main/src/tileops/kernels/norm/fused_add_norm.py`。
- 参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_fused_add_layer_norm.py` `FusedAddLayerNormTest.ref_program`（`added = (x.float()+residual.float()).to(x.dtype)` → `F.layer_norm(added.float(), (n,), weight.float(), bias.float(), eps)` → `.to(x.dtype)`，返回 `(y, added)`）——golden 以 §0.1 语义为依据移植此实现，不复刻 NPU 算法。
- manifest：`examples/TileOPs/tileops/manifest/norm.yaml` `FusedAddLayerNormFwdOp` 条目（workloads + roofline 公式 flops=8*M*N / bytes=(4*M*N + 2*N)*elem_bytes）。
- 性能基准：`examples/TileOPs/benchmarks/ops/bench_fused_add_layer_norm.py`（manifest 驱动，4 输入多输入形态）。

---

## 1. 概述

### 1.1 算子名称

`_fused_add_layer_norm_kernel`（FusedAddLayerNormFwdOp）

### 1.2 功能描述

对 `(M, N)` 张量按最后一维做 LayerNorm，同时返回预归一化残差和 `s = x + residual`（NPU 重设计后算法，与源算法的差异见 §0.6/§1.6.0：中心化两遍方差保留、pad 机制舍弃、persistent 分核、fp32 中转链、vbrc 列广播）。

### 1.3 数学公式

$$
s_{ij}=x_{ij}+\text{residual}_{ij},\qquad
\mu_i=\frac{1}{N}\sum_{j} s_{ij},\qquad
\text{rstd}_i=\frac{1}{\sqrt{\tfrac{1}{N}\sum_{j}(s_{ij}-\mu_i)^2+\varepsilon}}
$$
$$
y_{ij}=(s_{ij}-\mu_i)\cdot\text{rstd}_i\cdot w_j + b_j,\qquad \text{residual\_out}_{ij}=s_{ij}
$$

`residual_out` 为输入 dtype 舍回后的和；`y` 在 fp32 域按上式算得后单次舍回。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

与 §1.6.0 调研选定算法一致（中心化两遍驻留式，golden 对齐「舍回后统计」），每任务处理 block_m 行、整行驻留 UB：

1. **装载+相加**：x、residual 的 (real_m, N) 块 GM→UB（dtype）→ `vcast(rint)` 升 fp32 → `vadd` 得 `s_f32`（fp32 输入同 dtype 直拷、直加）；
2. **舍回副本（residual_out）**：`vcast(s_f32 → stage, rint)` 得 `round_dtype(s)` → `T.copy` 写 GM `residual_out`；再 `vcast(stage → x_f32, rint)` 把舍回值重入 fp32（与 golden 一致，见 §1.6.1 判别段 B）；
3. **均值**：`reduce_sum(x_f32, dim=1)` → S1；`mean = S1 · fl(1/N)`（O1 倒数乘）；
4. **中心化（保留）**：`x_f32 := x_f32 − mean`（行广播，得 d，epilogue 复用——O3）；
5. **方差**：`sq := d·d`；`reduce_sum` → S2；`var = S2·fl(1/N)`；`var += eps`；`rstd = sqrt(var)/var`（O2，vsqrt+vdiv 组合，绕开近似 vrsqrt）；
6. **归一化**：`x_f32 := d · rstd`（行广播）；
7. **仿射**：weight 块 (N,)→UB→`vcast`→fp32 `w1` →（`bm≥2` 时 `vbrc` 展开到 `sq_f32`）→ `x_f32 ·= w`；bias 同款 → `x_f32 += b`（R5）；
8. **舍回+写出**：`vcast(rint)` 回 dtype → GM `output`（双显式 slice，只写有效行）。

与源算法差异及来源：无 pad 机制（§0.6 R3 / §1.6.1）；persistent 分核（§0.6 R1）；bm 预算表驱动（§0.6 R2）；fp32 中转链（§0.6 R4）；vbrc 列广播（§0.6 R5，源码本为逐元素取列）；倒数乘与中心化复用（§1.6.1 O1/O3）；rstd 走 sqrt/div 组合（O2）。

### 1.5 数据流图

```
GM[x] --T.copy(slice)--> UB[stage(dtype)] --vcast(rint)--> UB[x_f32]
GM[residual] --T.copy(slice)--> UB[stage(dtype)] --vcast(rint)--> UB[sq_f32] --vadd--> UB[x_f32 = s]
UB[x_f32] --vcast(rint)--> UB[stage] --T.copy(slice)--> GM[residual_out]
UB[stage] --vcast(rint)--> UB[x_f32 = s32]
  --reduce_sum(dim=1)--> UB[s1] --vmul(N_inv)--> UB[mean]
  --vsub(行广播)--> UB[x_f32 = d] --vmul--> UB[sq_f32 = d²]
  --reduce_sum(dim=1)--> UB[s2] --vmul(N_inv)--> UB[var] --vadd(eps)--> UB[var]
  --vsqrt--> UB[s2'] --vdiv--> UB[rstd]
  --vmul(行广播, x_f32:=d·rstd)--> UB[x_f32]
GM[weight] --copy--> UB[stage(1,N)] --vcast--> UB[w1] --vbrc--> UB[sq_f32] --vmul--> UB[x_f32]
GM[bias]   --copy--> UB[stage(1,N)] --vcast--> UB[w1] --vbrc--> UB[sq_f32] --vadd--> UB[x_f32]
  --vcast(rint)--> UB[stage(dtype)] --T.copy(slice)--> GM[output]
（fp32 dtype：全部 vcast 消失，GM↔UB 同 dtype 直拷；buffer 复用：stage 角色 x/residual 装载→residual_out→y；sq_f32 角色 residual32→d²→weight 展开→bias 展开；bm==1 时无 vbrc，w1 直接参与 vmul/vadd）
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（per-row fused-add LayerNorm + 双输出）的算法族。源算法（中心化两遍 + SMEM/fragment 驻留 + 256-pad + 逐元素列索引）只是基线候选之一。调研深度：**完整调研**（规约/统计/多步类）。信息源：`algorithm-candidates.md` `ALG-layernorm`/`ALG-rmsnorm`/`ALG-reduction` 命中行 + 本仓 examples（ada 迁移设计 `.../ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md`、logsumexp 设计 `.../reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md`、lerp_tensor 实现 `.../elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py`）+ pattern-library（PL-1.5/PL-1.6/PL-1.10/PL-1.15、TRAP-fp16-opmath-golden/TRAP-vrsqrt-plain-precision/TRAP-C12）+ 源码 §0.3/§0.4 + 结构判据。无互联网检索（本地源全覆盖四问所需候选）。**子形态核对**：本算子与 ada 的差异是「双输出 + 逐列仿射 + 输入为和」；与 RMSNorm 的差异是「有 mean 减法、有 bias」。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | 基线（GPU 源算法） | 中心化两遍 + 256-pad + 谓词掩码 + SMEM/fragment 驻留 + 逐元素列索引 + 双输出 | — | — | ✅（基线） |
| 1 | 精确 N 两遍（去 pad / 去谓词） | 同基线数学，(bm, N) 精确缓冲 + slice 尾块 | 数学恒等（去零填充；pad 列零贡献） | 去 pad 记账 + 免 N_padded 缓冲加宽；小 N 大均值行数值更稳 | ✅（并入主选） |
| 2 | 单遍矩式 | var = E[s²] − (E[s])² | **否**——大均值小方差行 fp32 catastrophic cancellation（D-1 机器证据：fp32 违反率 0.13–0.21） | 驻留域内仅省统计相约 2N 向量操作（GM 流量不变） | ❌（数值稳定性否决） |
| 3 | Welford 在线 | 逐 tile 增量 mean/M2 合并 | 数学等价（容差内） | 驻留域内无 GM 收益；合并链逐 tile 标量序列依赖 | ❌（R2 详述） |
| 4 | N-tile 流式两遍 | block_n<N 分块 + `clear=False` 累加 + 输出遍重读 s | 等价 | 任意 N 可扩展（UB 无关） | ❌（域内 s 重读 1 遍，+25% GM 流量） |
| 5 | rsqrt 组合替代 1/sqrt（O2） | rstd = sqrt(v)/v（vsqrt+vdiv） | 数学恒等（v>0）；绕开近似 vrsqrt | 精度：近似指令 2.9e-3 → 组合 1.07e-7（TRAP-vrsqrt-plain-precision） | ✅（§1.6.1 采纳） |
| 6 | 除法转乘倒数（O1） | S/N → S·fl(1/N)（N 编译期常量） | 容差内（fp64 ≤1 ulp；fp16/bf16 舍回后逐位一致——D-1） | (bm,1) 上 vdiv→vmul | ✅（§1.6.1 采纳） |
| 7 | 中心化值复用（O3，CSE） | 统计相得 d 后保留，epilogue 复用 | 恒等（同值复用） | 每任务 −1 次 (bm,N) vsub | ✅（§1.6.1 采纳） |
| 8 | 舍回后统计（对齐 golden） | 统计量在 round_dtype(s) 上算（golden 路径） | 与源「未舍回统计」相差 ≤1 dtype ulp（判别段 B 机器验证 0 违反） | 双输出天然自洽（residual_out 与 y 用同一舍回值） | ✅（设计选择，§1.6.1） |
| 9 | 列因子广播替代 | vbrc 展开 (1,N)→(bm,N)（R5） | 恒等（复制列因子） | 替代未文档化列广播 +12~16% 税（PL-1.15） | ✅（§1.6.2/§3.2） |

**R2 在线算法**：**有**在线变体——Welford 增量统计与分块 running 统计（`algorithm-candidates.md` `ALG-layernorm` 行：「Welford 增量 mean/var；分块 running 统计」）。**但不采纳**，结构依据：在线变体的收益口径是「扫描遍数（两遍→单遍）/ 中间缓冲 O(N)→O(tile)」，而本算子工作负载域（manifest N ∈ {4096, 8192, 5120}）内整行可驻留 UB（fp16/bf16 bm·N·10B ≤ 113KB：N=4096@bm=2 → 80KB、N=8192@bm=1 → 80KB；fp32 bm·N·8B：N=5120@bm=2 → 80KB）——驻留式 s 的 GM 扫描已是 1 遍、中间缓冲已是 O(行)，在线化无任何 GM 收益；Welford 的逐 tile 合并链（每 tile 一次 reduce + 标量合并序列）反引入 Vector 不亲和的标量依赖链。仅当 N 超出驻留上限（fp16 约 N>~11000，§4.5）时流式结构才有意义（超出本任务域，§9.3 备注）。

**R3 复杂度对比**（每行 N 元素、eb=元素字节；「驻留」= 整行驻留 UB；FLOPs 口径=统计相+epilogue 的向量操作量纲（含 add 融合），无超越函数；GM Bytes 口径=输入读+输出写+中间 GM 往返（本算子无中间往返项；weight/bias 仅 2N）；扫描遍数=同一张量（x/residual/s）的 GM 重复读取次数）：

| 算法候选 | FLOPs（每行） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线两遍驻留（=主选） | ≈8N（add 1N + 统计 3N + 仿射 2N + 输出 2N；与 manifest roofline flops=8MN 一致） | **4MN·eb**（读 x/residual + 写 y/residual_out；weight/bias 仅 2N） | **1** | bm×N×10B（fp16）+4N（w1，UB） | 行独立；无跨核同步 |
| 矩式单遍驻留 | ≈6N 统计+仿射 | 4MN·eb | 1 | 同上 | 同上 |
| Welford 在线驻留 | ≈7N + 逐 tile 合并链 | 4MN·eb | 1 | 同上 + running 统计量 | 同上 |
| N-tile 流式两遍 | ≈8N + 每 tile 归约开销 | **5MN·eb**（s 读 2 遍：统计遍 + 输出遍） | **2** | 更小（N 无上限） | 同上 |

prefill 2048×4096 fp16 参照：驻留式 67.1MB vs 流式 83.9MB——**+25% 流量**（带宽受限算子上是主导项；PL-1.6：该流量规模下计算链已被 MTE2 完全隐藏，流量差即时间差）。

**R4 硬件亲和性评估**（检查清单逐项；负向淘汰证据：docs 路径+条款 / pattern-library 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 两遍驻留（主选） | Vector：reduce_sum + v-prefix 逐元素（`T.reduce_sum.md` / `T.vmul.md`），无 MAC 段 ✓ | bm·N×10B + 4N ≤ 113KB（192KB/1.7 裕度，`constants.md` CONST-capacity-910B2C + TRAP-UB-multibuffer-inflation）✓ | N 尾轴连续 stride=1；fp16/bf16 向量宽 ×8：4096/8192 整除 ✓，5120 fp32 ×4 整除 ✓ | M/N/bm 编译期常量；serial 边界静态（PL-1.5 折叠）✓ | 顺序依赖链（add→reduce→center→reduce→epilogue）；装载与计算 overlap 交由 auto multi-buffer（PL-1.6 实测隐藏）✓ | ✅ 主选 |
| 矩式驻留 | 同上 ✓ | 同上 | 同上 | 同上 | 同上 | ❌ 数值否决（R1#2，D-1 证据） |
| Welford 在线 | 合并链逐 tile 标量序列依赖，Vector 失分 | 同上 | 同上 | 同上 | 依赖链长，融合性差 | ❌ |
| N-tile 流式 | Vector ✓ | 更松（N 无上限） | 同上 | block_n 切片 ✓ | ✓ | ❌ 域内流量 +25%（R3）；仅 N>驻留上限时备用 |

**调研结论**：选定**「中心化两遍方差 + 整行 UB 驻留 + 精确 N + 舍回后统计」算法族**（候选 #0+#1+#8 融合，即源算法的数学内核 + NPU 去 pad 实现 + golden 对齐统计域），并采纳 rsqrt 组合、倒数乘、CSE、vbrc 列广播（#5–#9）。关键依据：R3 表中驻留式 GM 流量 4MN·eb 为全候选最小（流式 +25%）；R4 全项亲和；矩式/Welford 在驻留域内无流量收益且各有数值/向量化否决项。与基线的结构差异一句话：数学内核不变，去掉 GPU 对齐 padding 与谓词掩码、并行组织从 grid 超发改为 persistent 分核、dtype 路径改为显式 fp32 中转链、列因子改 vbrc 展开、rstd 走 vsqrt+vdiv。源算法优化手段意图承接见 §0.6 末尾汇总。调研范围：`algorithm-candidates.md` ALG-layernorm/ALG-rmsnorm/ALG-reduction 命中行、ada/logsumexp 迁移设计、lerp_tensor 实现、pattern-library PL-1.5/PL-1.6/PL-1.10/PL-1.15 + TRAP-*、源码 §0.3/§0.4、结构判据（R2）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；代表 workload = llama-8b-prefill 2048×4096 fp16）**：

- **估算下界: ≈32–53 µs**（流量项 67.1MB ÷ 1.26–2.07 TB/s——3:1 R/W 混合地板 1.26TB/s〔CONST-mte2-degradation〕与 sets=1 L2 驻留口径 ~2.07TB/s 并列给出区间；发射项：~20 op/task × 深流水被 MTE2 隐藏〔CONST-copy-floor-method：≥16M 元素档向量 pass 全隐藏〕；容量项：0——纯 Vector 无跨引擎往返）。
- 分 workload 估算（流量项 / 发射项取大者）：smoke-llama 64×4096（4.2MB）≈8–15µs（发射主导）；llama-8b-prefill 2048×4096 fp16（67.1MB）≈32–53µs；llama-70b-prefill 2048×8192 fp16（134MB，bm=1）≈65–110µs（bm=1 时 weight/bias 逐 task 重读，见下）；llama-decode 1×4096 bf16（32.8KB）≈10µs（单 task，~20 op × ~0.5µs〔CONST-vector-launch-overhead〕+ 搬运延迟）。
- 常数引用：CONST-mte2-degradation（tilelang 0.1.2+ed787bb，2026-09-07）/ CONST-vector-launch-overhead（0.1.2+3a214cde，2026-09-09）/ CONST-copy-floor-method（0.1.2+ed787bb）/ CONST-capacity-910B2C（3a214cde / 2026-08-24）。**⚠️ stale 标注**：上述条目均被 kb_stale_check 标记 stale（当前工具链 417dae99，2026-10-10 > 各条目版本戳）——数值仅用于候选排序与 Stage 4 baseline 对账（`hardware-cost-model.md` §3 要求 Stage 4 回填「设计估算 vs 实测」偏差行）；结构性结论（流式 +25% 流量、驻留可行）不依赖其精确值。
- **weight/bias 重读口径（未文档化假设 + 估算依据）**：bm=1 时每 task 重读 weight/bias（2N 元素），最坏情形使读侧从 2MN 升到 4MN；但 weight/bias 仅 16KB/个（N=8192 fp16）远小于 L2，按 L2 驻留计不入 HBM 流量地板（`constants.md` CONST-mte2-degradation「L2 驻留口径」）——设计期按 4MN·eb HBM 地板估算，Stage 4 实测校准。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法（中心化两遍驻留式，舍回后统计）。逐项四要素：

| # | 优化项 | 原式（源/GPU 语义） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 除法转乘倒数（PL-1.5「乘编译期常数倒数」先例） | `mean = S1 / N`；`var = S2 / N`（真除法） | `mean = S1 · fl(1/N)`；`var = S2 · fl(1/N)`（N_inv 工厂期 Python 常量） | N 为编译期常量；fl(1/N) 相对误差 ≤2⁻²⁴（N=2^k 时精确）；D-1 机器验证 24 案例全过：fp16/bf16 舍回后**逐位一致**（max ulp=0），fp32 ≤1 ulp（容差 1e-5 内） | (bm,1) buffer 上 2 次 vdiv → 2 次 vmul（op 型替换；buffer 极小收益有限，但零风险且与 PL-1.5 模式一致） |
| O2 | rsqrt 组合替代 1/sqrt | `rstd = 1/sqrt(var+eps)`（源 `T.rsqrt`→NPU `T.vrsqrt` 近似指令 2.9e-3） | `rstd = sqrt(v)/v`，`v = var+eps`（`T.vsqrt` + `T.vdiv`） | 实数恒等 `sqrt(v)/v = 1/sqrt(v)`（v>0，因 eps=1e-5）；TRAP-vrsqrt-plain-precision 实证组合式 fp32 1.07e-7 vs 近似指令 2.9e-3 | 1 次近似向量指令 → 2 次全精度向量指令（op 数 +1，换 fp32 1e-5 门禁达标） |
| O3 | 中心化值复用（公共子表达式消除） | epilogue 从原始 x 副本重推 `(s−mean)`（重减） | 统计相中心化后**保留** d：`x_f32 := x_f32−mean`（驻留），epilogue `y = ((d·rstd)·w) + b` | 恒等——d 是同一子表达式的值复用（D-1：0 ulp） | 每任务 −1 次 (bm,N) vsub；浅流水档约 −1µs/task；深流水档被 MTE2 隐藏（PL-1.6） |
| O4 | 舍回后统计（对齐 golden，设计选择） | 源：S1/S2 在未舍回 fp32 和 `add_f32` 上算，中心化用舍回副本 | 统计量在 `round_dtype(s)`（= residual_out）重入 fp32 后算 | 与源相差 ≤1 dtype ulp（仅统计量）；与 golden 的 `F.layer_norm(added.float(), ...)` 路径**逐项一致** | 双输出自洽（residual_out 与 y 用同一舍回值）；判别段 B 机器验证 0 违反 |
| — | vbrc 列广播（R5） | 逐元素 `weight[j]` | `vbrc((1,N)→(bm,N))` 后 vmul/vadd | 复制列因子 = 逐元素取列（D-1 不涉浮点语义变化，正确性由 golden 覆盖） | 替代未文档化列广播 +12~16% 税（PL-1.15） |

**D-1 等价性机器验证结果表**（脚本 `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/verify_equiv.py` 已执行，2026-10-14，torch CPU、fp64 参照 + 角点值；判定双准则：(a) 候选式 vs 基线式在 dtype 容差内，或 (b) 违反率 = 0）：

| 验证项 | 案例数 | fp16/bf16 舍回后 max ulp | fp32 表现 | 违反率 | 结论 |
|--------|--------|---------|---------|--------|------|
| O1+O2+O3+O4 联合（候选式 vs 基线式） | 24（3 dtype × 8 形态：N∈{1,2,514,1152,3000,4096,5120,8192}） | **0（逐位一致）** | randn/大均值行 max_ulp ≤1（容差 1e-5 内） | **0**（三 dtype 全 0） | **EQUIV_PASS** |
| 判别段 A：矩式（拒绝证据，非采纳项） | 8 形态 × 3 dtype | — | — | fp32 max 0.2100、fp16 max 0.1554（bf16 宽容差掩蔽 0.0005） | **验证失败（FAIL）→ 拒绝采纳**（机器证据支撑 R1#2 否决） |
| 判别段 B：round-first 统计 vs 源 unrounded 统计 | 24 | 0 | ≤1 | **0** | **DISCRIM_OK**（源统计在容差内，设计取 golden 对齐） |

形态覆盖：randn 行、大均值小方差行（3e3±0.5，矩式消灾难压力源）、常量行（var=0 → rstd=sqrt(eps)/eps）、全零行、次正规点位、±inf 混合行、NaN 点位、极值 weight/bias（±65504 / ±3e38）、N=1/2、多 shape。

**优化结论**：采纳 O1/O2/O3/O4（+ R5 列广播替代）。优化后公式（§3.1 唯一输入）：

$$
\mu = S_1\cdot r_N,\quad d = s-\mu,\quad \text{rstd} = \frac{\sqrt{\,S_2\cdot r_N + \varepsilon\,}}{S_2\cdot r_N + \varepsilon},\quad y = ((d\cdot\text{rstd})\cdot w) + b,\qquad r_N = \underline{\text{fl}(1/N)}
$$

其中 `s` 为舍回后重入 fp32 的和，`S1=Σs`、`S2=Σd²`（`residual_out = round_dtype(s)`）。

**否决项及依据**（负向断言举证，`negative-claim-evidence.md` §1）：
- 矩式 var=E[s²]−E[s]²：D-1 机器证据（上表判别段 A），非纸面论证；
- fp16 原生域计算链（不经 fp32 中转）：TRAP-fp16-opmath-golden（torch CPU golden fp32 opmath + 单次舍回；fp16 原生三步链差 2–3 ulp）——fp32 中转为精度必需；
- 跨 dtype `T.copy` 作 cast 融合：TRAP-C12（`docs/Tilelang.language/内存操作/T.copy.md` §3——跨 dtype copy 不减少 vector pass 且舍入不可控）——上cast 数值无损故安全，但按陷阱纪律统一走显式 `vcast(rint)`；
- 未文档化列广播直接在 `[M,N]×[1,N]` 上用 vmul/vadd：PL-1.15-broadcast-perf-tax（可编译可算对但有 +12~16% 性能税）——改走 vbrc 展开。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（GPU 源） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|---------------------|--------------------------|---------|--------------|
| 1 | x/residual 装载 | `T.Parallel(bm,Np)` 谓词掩码 / 直读 | `T.copy` 双显式 slice（尾块 `T.min`） | ✅ | — |
| 2 | 上cast fp32（x/res） | `T.Parallel` 逐元素 `T.cast` | `T.vcast(stage, x_f32, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`：f16→f32 rint ✓ bf16→f32 rint ✓） | ✅ | — |
| 3 | 残差相加 | `T.Parallel` 逐元素 `cast+cast+add` | `T.vadd(x_f32, sq_f32, x_f32)`（`docs/Tilelang.language/数学操作/T.vadd.md` 同 shape ✓） | ✅ | — |
| 4 | 行和 S1/S2 | `T.reduce_sum(add_f32, acc, dim=1)` | 同 API（`docs/Tilelang.language/规约操作/T.reduce_sum.md`：(M,N)→(M,1)，fp32 ✓） | ✅ | — |
| 5 | mean/var 标量行计算 | `T.Parallel(bm)` 循环 `acc[i]/N` | `T.vmul(s1, N_inv, mean)`（(bm,1) 单向量 op；`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量广播 ✓） | ✅ | — |
| 6 | 中心化+平方 | `T.Parallel(bm,Np)` 标量复合式 | `T.vsub(x_f32, mean, x_f32)`（行广播 [M,N]−[M,1]，`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2）+ `T.vmul(x_f32, x_f32, sq_f32)` | ✅ | — |
| 7 | var+eps / rstd | 标量式 `T.rsqrt(...)` | `T.vadd(var, eps, var)`（标量 ✓）+ `T.vsqrt(var, s2)` + `T.vdiv(s2, var, rstd)`（`docs/Tilelang.language/数学操作/T.vsqrt.md` / `T.vdiv.md`，O2） | ✅ | — |
| 8 | epilogue 仿射 | `T.Parallel(bm,Np)` 复合标量表达式（逐元素 `weight[j]`/`bias[j]`） | `T.vmul(x_f32, rstd, x_f32)`（行广播）+ w1/bias→`vcast`+`vbrc`(bm≥2)/直接(bm==1)+`vmul`/`vadd`（R5） | ✅ | — |
| 9 | 舍回副本 + 下cast | 赋值隐式截断 | `T.vcast(x_f32 → stage, rint)`（f32→f16 / f32→bf16 rint ✓）；双显式 slice copy 写出 | ✅ | — |
| 10 | 任务循环（persistent） | grid 超发（每 block 一任务，无核内循环） | `for i in T.serial(num_local_tasks)` + `if block_id < num_logical` 守卫 | ❌ 保留 | **block 级任务映射计算**：每 task 数个标量索引（block_id/off_m），非逐元素热点；serial 循环本身是 persistent 分核意图（§0.6 R1）的载体，无逐元素等价 API（lerp/ada 同款结构） |
| 11 | 行尾 real_m | 谓词掩码（逐元素 `T.And`） | `real_m = T.min(block_m, M − off_m)`（标量，1 次/task，`docs/Tilelang.language/比较操作/T.min.md`） | ❌ 保留 | **动态边界元数据**：M % bm ≠ 0 的尾任务行数；per-task 1 次标量 min + slice 界，深流水档被 MTE2 隐藏（CONST-copy-floor-method）；静态化需为尾任务复制 kernel 体（复杂度不值） |
| 12 | bm==1 分支（是否 vbrc） | —（源无此分支） | host 层编译期 `if block_m == 1` 常量分支 | ❌（非 kernel 内） | **host 侧元数据/编译期分支**：规避 TRAP-vbrc-same-shape-empty-broadcast，非 kernel 内逐元素计算 |

**向量化结论**：逐元素计算（cast/加/减/乘/vsqrt/vdiv/rsqrt/规约/vbrc 广播/仿射）**已全部向量化**（替代 API 均有 docs+examples 佐证，上表）；保留 3 类标量/循环——任务映射索引（#10）、动态边界 min（#11）、host 编译期分支（#12），逐项理由如上，无「实现简单」类理由。§6 循环结构与本表一致（kernel 内无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

规约类算子必选候选：**水平归约（lane→归约轴）vs 垂直扫描（lane→独立行）**；另评估 N-tile 流式与转置布局；两遍 vs online 与 lane 映射的交互已在 §1.6.0 R2 覆盖（驻留域内在线化无收益）。**双输出 + 列因子（weight/bias）不改变归约轴选择**：两输出共享同一 (bm,N) 布局与同一 N 归约轴。

**轴质量评分**（主选轴 N；昇腾 Vector 128bit：fp16/bf16 ×8、fp32 ×4）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | fp16/bf16：N=4096（512×8 ✓）、8192（1024×8 ✓）；fp32 ×4：N=5120（1280×4 ✓） |
| 尾 lane 浪费率 | 全 manifest 形状 0%（4096/8192 为 8 的倍数，5120 为 4 的倍数） |
| 累加链形态 | 归约维 = 尾轴连续 stride=1（v-op 全 buffer 行广播 [M,N]↔[M,1] 为文档形态，`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2）；无跨步系数 |
| repack 代价 | **无**（I/O 原生行主序 = 核内布局）；列因子 weight/bias 走核内 `vbrc`（UB 级，非 GM 重排） |
| UB 容量影响 | 全 N 驻留 → bm 受 10B·bm·N + 4N ≤ 113KB 约束（§4.5 表）；对价 = x/residual 单遍 GM 扫描 |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳？ |
|---|---------|---------|------------|--------------|--------|
| 1 | I/O 原生 (M,N) 行主序 + **全 N 驻留** | N（尾轴连续；`reduce_sum dim=1` 水平归约 + 行广播 v-op + vbrc 列广播） | 无 | GM 流量 4MN·eb（全候选最小）；x/residual 扫描 1 遍；UB 预算驱动 bm（§4.5） | ✅ **主选** |
| 2 | N-tile 流式（block_n<N，`T.serial` 分块 + `clear=False` 累加；输出遍重读 s） | N（分 tile） | 无 | UB 更松但 **s GM 重读 1 遍 → 5MN·eb（+25% 流量，prefill +16.8MB ≈ +13µs）**；仅 N>驻留上限时必要 | ❌ 域内 strictly worse（R3 表）；留作 N 上界 fallback（§9.3） |
| 3 | 垂直扫描（lane→独立行，串行步进 N） | 行 | 无 | 归约维访问 stride=M（跨步）→ 无向量宽度；且跨步 gather 向量指令缺失（TRAP-C10-strided-gather，`traps-compiler.md`） | ❌ |
| 4 | 转置布局 (N,M)（host permute 或核内转置链） | M | host permute ~百 µs（PL-1.3 净亏）/ 核内 T.transpose 二轴交换链 ~µs（PL-1.1） | 归约轴变跨步——repack 纯开销零收益 | ❌ |
| 5 | 列因子 host 预展开（weight/bias 预先广播成 (M,N)） | N | host 侧生成 (M,N) 张量（额外 GM 往返，M 级） | 省核内 vbrc 但引入 2MN 额外 GM 读写 → 净亏；且 host 侧改动张量违反 ascend-constraints §4 | ❌ |

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：GPU 将 CTA 内 threads 映射到行内元素（block_m=1 + threads=128，线程级跨 N 归约）——即 GPU 也选了 N 为归约/向量轴，但那是 CUDA warp 组织的结果；本节依据 NPU Vector 宽度/广播形态/UB 容量独立评估后**结论一致、理由不同**（NPU：尾轴连续 stride=1 + [M,N]↔[M,1] 广播 + 全 N 驻留单遍扫描）。列因子（weight/bias）在 GPU 上是线程沿行已持列索引的寄存器取数，在 NPU v-op 上无对应——本设计独立选择核内 `vbrc` 展开（§0.6 R5）。

**实验裁决模式评估**：主选 #1 vs #2 的判定裕度 = 25% GM 流量差，远超未实证常数的不确定区间 → 无需三件套；唯一落入未实证区间的量是 **auto-multi-buffer 膨胀系数（~1.7x，TRAP-UB-multibuffer-inflation）对本 kernel 4-buffer 形态的适用性**（决定 N=8192 fp16 时 bm 是否需回退）——这是**参数级**（block_m 取值）而非结构级决策：主选结构对 bm∈{1,2} 均成立，设计期取保守值（bp 表，§4.5），bm 留 Stage 4 实测扫描（pattern-library INDEX.md §3「本库模式是起点不是终点」；编译期 UB 溢出报错即天然 guard）。**不启动设计期探针（D-3）**：结构决策不依赖该常数。

**布局决策结论**：选定 #1：核内布局 = I/O 原生 (M,N) 行主序、向量化轴 = N（尾轴）、repack = 无（列因子走 UB 级 vbrc）；该结论同步落入 §3.3 伪代码与 §6 循环结构（归约/广播的内层向量维 = N；buffer 形状 = (block_m, N)）；弃选方案的量化理由见候选矩阵。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（reduce_sum + v-prefix 逐元素 + vbrc 广播），无 matmul | 无 Cube/L0/L1 参与 |
| 归约 | `T.reduce_sum` dim=1 | Developer/Expert 均可用（decision-tree.md「T.reduce_sum/max/min 在两种模式下都可使用」），无需手动控制 |
| 内存层级 | 仅 GM↔UB | `T.alloc_shared`（Developer 映射 UB，`docs/Tilelang.language/内存操作/T.alloc_shared.md`） |
| 同步 | 单 kernel 内顺序依赖链、无核间协作 | Developer 自动同步 |
| 先例 | ada 迁移（同族 row-reduction，Developer persistent）已交付；logsumexp（同族 row-reduction，Developer）在库；lerp_tensor（同款多输入多输出流量，Developer persistent）已调优 | Developer 已验证可行 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB）——v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1） |
| 计算方式 | v-prefix API（vcast/vmul/vsub/vadd/vsqrt/vdiv/vbrc）+ `T.reduce_sum` + `T.min` 标量边界 |
| 同步 | Developer 自动同步（无手动 sync_block_set/wait） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维） |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 优化后公式（fp32 计算域；`r_N = fl(1/N)`；`s` 为舍回后重入 fp32 的和）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x32 = fp32(x)`；`r32 = fp32(residual)`；`s = x32 + r32` | 上cast + 融合残差加 |
| 2 | `s_dtype = dtype_round(s)` | 舍回（= residual_out，§8.1 语义） |
| 3 | `s32 = fp32(s_dtype)` | 重入 fp32（O4，golden 对齐） |
| 4 | `S1 = Σ_j s32[i,j]` | 行和 |
| 5 | `mean = S1 · r_N` | 均值（O1） |
| 6 | `d = s32 − mean` | 中心化（保留，O3） |
| 7 | `S2 = Σ_j d[i,j]²` | 中心化平方和 |
| 8 | `v = S2 · r_N + eps`；`rstd = sqrt(v)/v` | （O1 + O2） |
| 9 | `t = d · rstd` | 归一化 |
| 10 | `t = t · fp32(weight)`；`t = t + fp32(bias)` | 仿射（列因子，R5） |
| 11 | `y = dtype_round(t)` | 单次舍回（rint）→ output |
| 12 | `residual_out = s_dtype` | 第二输出（步骤 2 产物） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 依据 |
|------|----------|-------------|------|------|------|
| 任务映射 | `block_id = i·K + cid` | `T.serial(num_local_tasks)` + `if block_id < num_logical` | 静态边界（host 折叠） | Developer | ada R1 / lerp 同款 |
| 行尾 | `real_m = min(bm, M−off)` | `T.min(block_m, M - off_m)` | 标量 | Developer | `docs/Tilelang.language/比较操作/T.min.md`；logsumexp 设计 §5.4 |
| 载入 x/residual | GM→UB | `T.copy(x[off:off+real_m, 0:N], stage[0:real_m, 0:N])` | src/dst 双显式 slice，dst 零起点 | Developer | `docs/Tilelang.language/内存操作/T.copy.md` + TRAP-T-copy-region-semantics + TRAP-UB-dst-align |
| 上cast | fp16/bf16→fp32 | `T.vcast(stage, x_f32, round_mode="rint")` | (bm,N)→(bm,N) 同 shape | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（f16→f32 / bf16→f32 rint ✓） |
| 残差加 | `s = x + residual` | `T.vadd(x_f32, sq_f32, x_f32)` | 同 shape (bm,N) | Developer | `docs/Tilelang.language/数学操作/T.vadd.md` |
| 舍回（residual_out） | `s_dtype = round(s)` | `T.vcast(x_f32, stage, round_mode="rint")` + `T.copy(stage[0:real_m,0:N], residual_out[off:off+real_m,0:N])` | f32→f16 / f32→bf16 rint ✓ | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`；TRAP-fp16-opmath-golden 通解 |
| 重入 fp32 | `s32 = fp32(s_dtype)` | `T.vcast(stage, x_f32, round_mode="rint")` | f16→f32 / bf16→f32 rint ✓ | Developer | 同上（O4） |
| 行和 | S1, S2 | `T.reduce_sum(x_f32, s1, dim=1)` | (bm,N)→(bm,1)，clear=True（默认） | Developer | `docs/Tilelang.language/规约操作/T.reduce_sum.md` |
| 均值 / 方差 | `S·r_N` | `T.vmul(s1, N_inv, mean)`；`T.vmul(s2, N_inv, var)` | (bm,1)×标量 | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量广播 |
| 中心化 | `d = s − mean` | `T.vsub(x_f32, mean, x_f32)` | [bm,N]−[bm,1] 行广播，in-place | Developer | `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2 |
| 平方 | `sq = d²` | `T.vmul(x_f32, x_f32, sq_f32)` | out-of-place（保 d） | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` |
| var+eps | `v = S2·r_N + eps` | `T.vadd(var, eps, var)` | 标量 src1 | Developer | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2 |
| rstd | `sqrt(v)/v` | `T.vsqrt(var, s2)` + `T.vdiv(s2, var, rstd)` | (bm,1) 同 shape | Developer | `docs/Tilelang.language/数学操作/T.vsqrt.md` / `T.vdiv.md`（O2） |
| 归一化 | `d·rstd` | `T.vmul(x_f32, rstd, x_f32)` | 行广播 in-place | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 |
| 列因子装载 | `w_j`/`b_j` | `T.copy(weight[0:N], stage[0:1, 0:N])` + `T.vcast(stage[0:1,0:N], w1, rint)` | (1,N) fp32 | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` |
| 列广播 | `[1,N]→[bm,N]` | `T.vbrc(w1, sq_f32)` | bm≥2 时；bm==1 直接同形 vmul/vadd | Developer | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（src (1,N) → dst (bm,N) ✓，bf16 亦支持） |
| 仿射乘/加 | `t·w + b` | `T.vmul(x_f32, sq_f32, x_f32)`；`T.vadd(x_f32, sq_f32, x_f32)`（bm≥2）；bm==1 用 `T.vmul(x_f32, w1, x_f32)`/`T.vadd(x_f32, w1, x_f32)` | 同形 (bm,N) | Developer | 同上 |
| 舍回 + 写出 | fp32→dtype | `T.vcast(x_f32, stage, round_mode="rint")` + `T.copy(stage[0:real_m,0:N], output[off:off+real_m,0:N])` | 双显式 slice | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` / `docs/Tilelang.language/内存操作/T.copy.md` |
| fp32 路径 | — | 同上但**无 vcast/stage**：`T.copy(x[...], x_f32[...])`、residual 拷入 `sq_f32`、`T.copy(x_f32[...], residual_out[...])` 直写、weight/bias 同 dtype 拷入 `w1` | 同 dtype | Developer | `docs/Tilelang.language/内存操作/T.copy.md` |

### 3.3 计算伪代码

> 按模板 §3.3 注记：`T.Tensor` 的 dtype 以第二个位置参数（下标形式）传入，不用 `dtype=` 关键字（避免 `name 'dtype' is not defined` 类 false alarm）。工厂接口与源一致 `(M, N, eps, dtype)`；`_func(block_m)` 移除 threads（TRAP-threads-kwarg-noop；NPU wrapper `build_kernel(*identity)` 已按 `(M,N,eps,dtype)` 调用）。**双输出** `out_idx=[4, 5]`。任务级 bm==1/.bm≥2 与 dtype 两分支均在**工厂层**解析（trace 体之外，规避 TRAP-C11 parser 作用域）。

```python
def _fused_add_layer_norm_kernel(M, N, eps, dtype):
    N_inv = 1.0 / float(N)                              # O1: 编译期常数（Kernel 内、Scope 外的标量操作数）
    use_fp32_transit = dtype in ("float16", "bfloat16")  # bf16 无 Vector 算术 -> 必需中转
    bpe = 10 if use_fp32_transit else 8                 # fp16/bf16: stage 2 + x_f32 4 + sq_f32 4
    UB_SAFE_BYTES = 115_000                             # 192KB / 1.7（TRAP-UB-multibuffer-inflation）
    block_m_default = min(8, max(1, (UB_SAFE_BYTES - 4 * N) // (bpe * N)))
    vector_cores = NPUUtils.get().get_aicore_num() * 2  # 实查：20 -> 40（纯 Vector 翻倍，§5.5）

    @tilelang.jit(out_idx=[4, 5], target="npuir")
    def _func(block_m=block_m_default):
        if block_m * bpe * N + 4 * N > UB_SAFE_BYTES:
            raise ValueError(f"block_m={block_m} exceeds UB budget for N={N}, dtype={dtype}")
        num_logical = (M + block_m - 1) // block_m            # host 编译期常量（PL-1.5 折叠）
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        if use_fp32_transit:
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                residual: T.Tensor[(M, N), dtype],
                weight: T.Tensor[(N,), dtype],
                bias: T.Tensor[(N,), dtype],
                output: T.Tensor[(M, N), dtype],         # out_idx=4
                residual_out: T.Tensor[(M, N), dtype],   # out_idx=5
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    stage = T.alloc_shared((block_m, N), dtype)        # x/residual 中转 -> residual_out -> y
                    x_f32 = T.alloc_shared((block_m, N), "float32")    # s -> s32 -> d -> t
                    sq_f32 = T.alloc_shared((block_m, N), "float32")   # r32 -> d^2 -> weight brc -> bias brc
                    w1 = T.alloc_shared((1, N), "float32")             # weight/bias fp32（复用）
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")       # 兼作 sqrt(var) 暂存
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            # --- 1. 装载 + 上cast + 残差加 ---
                            T.copy(x[off_m:off_m + real_m, 0:N], stage[0:real_m, 0:N])
                            T.vcast(stage, x_f32, round_mode="rint")
                            T.copy(residual[off_m:off_m + real_m, 0:N], stage[0:real_m, 0:N])
                            T.vcast(stage, sq_f32, round_mode="rint")
                            T.vadd(x_f32, sq_f32, x_f32)                 # s = x + residual

                            # --- 2. 舍回（residual_out）+ 重入 fp32（O4） ---
                            T.vcast(x_f32, stage, round_mode="rint")     # round_dtype(s)
                            T.copy(stage[0:real_m, 0:N], residual_out[off_m:off_m + real_m, 0:N])
                            T.vcast(stage, x_f32, round_mode="rint")     # s32

                            # --- 3. 均值（O1） ---
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, N_inv, mean)

                            # --- 4. 中心化两遍方差（d 保留，O3） ---
                            T.vsub(x_f32, mean, x_f32)                   # d = s - mean
                            T.vmul(x_f32, x_f32, sq_f32)                 # sq = d^2
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, N_inv, var)
                            T.vadd(var, eps, var)
                            T.vsqrt(var, s2)                             # s2 := sqrt(v)
                            T.vdiv(s2, var, rstd)                        # rstd = sqrt(v)/v（O2）

                            # --- 5. 归一化 + 仿射（列因子，R5） ---
                            T.vmul(x_f32, rstd, x_f32)                   # d * rstd
                            T.copy(weight[0:N], stage[0:1, 0:N])
                            T.vcast(stage[0:1, 0:N], w1, round_mode="rint")
                            if block_m == 1:                             # 编译期分支：同形直接算，规避空广播
                                T.vmul(x_f32, w1, x_f32)
                            else:
                                T.vbrc(w1, sq_f32)
                                T.vmul(x_f32, sq_f32, x_f32)
                            T.copy(bias[0:N], stage[0:1, 0:N])
                            T.vcast(stage[0:1, 0:N], w1, round_mode="rint")
                            if block_m == 1:
                                T.vadd(x_f32, w1, x_f32)
                            else:
                                T.vbrc(w1, sq_f32)
                                T.vadd(x_f32, sq_f32, x_f32)

                            # --- 6. 舍回 + 写出（只写有效行） ---
                            T.vcast(x_f32, stage, round_mode="rint")
                            T.copy(stage[0:real_m, 0:N], output[off_m:off_m + real_m, 0:N])
        else:
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                residual: T.Tensor[(M, N), dtype],
                weight: T.Tensor[(N,), dtype],
                bias: T.Tensor[(N,), dtype],
                output: T.Tensor[(M, N), dtype],
                residual_out: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_f32 = T.alloc_shared((block_m, N), "float32")    # s -> d -> t
                    sq_f32 = T.alloc_shared((block_m, N), "float32")   # r32 -> d^2 -> weight brc -> bias brc
                    w1 = T.alloc_shared((1, N), "float32")
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
                            T.copy(x[off_m:off_m + real_m, 0:N], x_f32[0:real_m, 0:N])
                            T.copy(residual[off_m:off_m + real_m, 0:N], sq_f32[0:real_m, 0:N])
                            T.vadd(x_f32, sq_f32, x_f32)                 # s
                            T.copy(x_f32[0:real_m, 0:N], residual_out[off_m:off_m + real_m, 0:N])
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, N_inv, mean)
                            T.vsub(x_f32, mean, x_f32)
                            T.vmul(x_f32, x_f32, sq_f32)
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, N_inv, var)
                            T.vadd(var, eps, var)
                            T.vsqrt(var, s2)
                            T.vdiv(s2, var, rstd)
                            T.vmul(x_f32, rstd, x_f32)
                            T.copy(weight[0:N], w1[0:1, 0:N])            # 同 dtype 直拷
                            if block_m == 1:
                                T.vmul(x_f32, w1, x_f32)
                            else:
                                T.vbrc(w1, sq_f32)
                                T.vmul(x_f32, sq_f32, x_f32)
                            T.copy(bias[0:N], w1[0:1, 0:N])
                            if block_m == 1:
                                T.vadd(x_f32, w1, x_f32)
                            else:
                                T.vbrc(w1, sq_f32)
                                T.vadd(x_f32, sq_f32, x_f32)
                            T.copy(x_f32[0:real_m, 0:N], output[off_m:off_m + real_m, 0:N])
        return main

    return _func
```

> 尾块说明：v-op 作用于完整 (block_m, N) buffer，尾任务 [real_m:] 行为未初始化垃圾——按行隔离（reduce dim=1 / 行广播），写出只取有效行（logsumexp 设计 §5.4 + ada 设计 §0.6 R3 同款）。
> bm==1 分支（`if block_m == 1:`）为**工厂层 Python 编译期分支**：`block_m` 是 trace 期 Python 常量，两分支在 trace 时二选一；实现时须确保该 if 位于 factory 层（trace 体之外或 @T.macro 层）以规避 TRAP-C11 parser 作用域（与 dtype 分支同处置）。
> 接口保持：`main(x, residual, weight, bias, output, residual_out)` 参数名/顺序与源一致（源名 y/residual_out，本设计取 output/residual_out 对齐 manifest signature 语义）；`out_idx=[4, 5]`。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `T.Kernel(一维, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` | ✅ 已核对 | NPU ≤2D block；本设计 1D；`(cid, _)` 解包 |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md` | ✅ 已核对 | Developer 映射 UB；1D~2D ✓；v-op 操作数可用 |
| `T.copy`（slice 形态） | `docs/Tilelang.language/内存操作/T.copy.md` | ✅ 已核对 | GM↔UB ✓；尾块借 slice extents；src/dst 同 shape；dst 零起点 |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | ✅ 已核对 | f16→f32 rint ✓；bf16→f32 rint ✓；f32→f16 rint ✓；f32→bf16 rint ✓；shape 须一致 |
| `T.reduce_sum` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` | ✅ 已核对 | fp32 ✓ bf16 ×；(M,N)→(M,1)；clear 默认 True |
| `T.vmul` / `T.vsub` / `T.vadd` | `docs/Tilelang.language/数学操作/T.vmul.md` / `T.vsub.md` / `T.vadd.md` | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；行广播 [M,N]↔[M,1] ✓；标量 src ✓；in-place ✓；UB 操作数 |
| `T.vsqrt` / `T.vdiv` | `docs/Tilelang.language/数学操作/T.vsqrt.md` / `T.vdiv.md` | ✅ 已核对 | fp16/fp32 ✓（bf16 中转后为 fp32）；组合 rstd（O2，TRAP-vrsqrt-plain-precision） |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` | ✅ 已核对 | src (1,N)→dst (bm,N) 合法；bf16 支持；**同形 src/dst 触发空广播失败**（TRAP-vbrc-same-shape-empty-broadcast）→ bm==1 分支规避 |
| `T.min` | `docs/Tilelang.language/比较操作/T.min.md` | ✅ 已核对 | 标量 min（fp16/fp32/int ✓），尾块行数 |
| `T.serial` + if 守卫 | ada 设计 §3.3 / lerp_tensor 实现 | ✅ 佐证 | persistent grid-stride 同款（if 体内只写预分配 buffer，规避 C11 作用域） |
| `@tilelang.jit(out_idx=[4, 5], target="npuir")` | 源 `out_idx=[4,5]`；NPU logsumexp/lerp out_idx 单输出先例 | ✅ 佐证 | 双输出自动分配（本算子接口契约） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(num_kernels, is_npu=True)` |
| GPU 专用 API | Yes | `threads=`/`T.sync_threads`/`T.if_then_else`+`T.And` 谓词 不迁移（§0.5 舍弃/替换清单） |
| GEMM 分形/非整除 | No | 无 GEMM；M 非整除经 T.min+slice；N 无 256 对齐要求 |
| L0C 容量 | No | 纯 Vector，无 L0 |
| 物理核数适配 | Yes | §5.5 三要素（实查 20→40） |
| bf16 无 Vector 算术 | Yes | §0.6 R4 fp32 中转链（必需） |
| 列因子广播无文档化直连 | Yes | §0.6 R5 vbrc 展开（PL-1.15） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5/§0.6）

| 差异项 | GPU 源 | 本项目（Ascend） | 转换方案 |
|--------|--------|-----------------|----------|
| threads 参数 | `T.Kernel(grid, threads=)` | npuir 无效（TRAP-threads-kwarg-noop） | 移除；`_func(block_m)` |
| `T.alloc_fragment` 计算域 | CUDA 寄存器 | v-op 要求 UB 操作数 | 本设计全部 `alloc_shared` |
| `T.Parallel` 标量循环 | 元素级并行 | v-prefix 向量 op | §1.6.2 全表 |
| `T.if_then_else`+`T.And` 谓词 | masked load + row_guard | `T.min`+双显式 slice+垃圾行丢弃 | §0.6 R3 |
| `T.rsqrt`（标量式） | fragment 标量 | `T.vrsqrt` 近似 → `T.vsqrt`+`T.vdiv` 组合 | §1.6.1 O2 |
| N_padded=align_up(N,256) | SMEM 对齐 | 无此约束（编译器 32B 管理） | 精确 N + slice |
| 逐元素 `weight[j]`/`bias[j]` | 寄存器列索引 | 无逐元素取列 | `T.vbrc` 列广播（§0.6 R5） |
| 隐式 dtype 截断赋值 | `x_local[i,j]=add_f32[i,j]` | 显式 `T.vcast(rint)` + 同 dtype copy | TRAP-C12 / TRAP-fp16-opmath-golden |
| 双输出 out_idx=[4,5] | 同源 | 同源保留 | `@tilelang.jit(out_idx=[4,5], target="npuir")` |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md` | 极高（同族 row-reduction 迁移设计，已交付） | Developer persistent 行归约结构、尾块 T.min+slice+垃圾行丢弃、fp32 计算域、bf16 fp32 中转、vsqrt+vdiv rstd 绕法、UB 预算表、分核实查 |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` | 极高（同族 NPU 实现） | `T.vsqrt`+`T.vdiv` rstd 组合（L313-318）、`_get_factory` 缓存、L0 测试骨架 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` | 高（同族 row-reduction 设计） | 尾块 `T.min` + src/dst 双显式 slice + 垃圾行丢弃、bf16 fp32 中转、`(bm,1)` reduce 输出形态 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py` | 高（同款多输入多输出流量，已调优） | persistent min(logical, cores) 分核 + grid-stride serial、vcast(rint) fp32 中转链、UB 预算 guard |
| `examples/flash_attention/flash_attn_npuir.py` | 中（NPU 参考实现集首条） | T.min/T.vbrc 广播/持久化结构形态佐证 |
| `examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm.py` | 上下文 | NPU wrapper 移植形态（`build_kernel` 调用约定、K5-K11 适配） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 输入；M、N 为工厂期常量（per-build JIT 特化） |
| `residual` | `(M, N)` | same_as(x) | 残差 |
| `weight` | `(N,)` | same_as(x) | 仿射缩放 |
| `bias` | `(N,)` | same_as(x) | 仿射平移 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | `(M, N)` | same_as(x) | `out_idx=[4]` 自动分配（归一化结果 y） |
| `residual_out` | `(M, N)` | same_as(x) | `out_idx=[5]` 自动分配（预归一化残差和 s） |

### 4.3 中间缓冲区

**fp16/bf16 路径**（fp32 中转）：

| Buffer 名 | Shape | dtype | 存储层级 | 用途（角色复用时序） |
|-----------|-------|-------|----------|---------------------|
| `stage` | `(block_m, N)` | dtype | UB (alloc_shared) | ① x 中转 → ② residual 中转 → ③ residual_out（舍回）→ ④ weight/bias 装载首行 → ⑤ y 舍回中转 |
| `x_f32` | `(block_m, N)` | float32 | UB | ① x32 → ② s（和）→ ③ s32（重入）→ ④ d（中心化，保留）→ ⑤ t（归一化·仿射结果） |
| `sq_f32` | `(block_m, N)` | float32 | UB | ① residual32 → ② d² → ③ weight 展开 (bm,N) → ④ bias 展开 (bm,N) |
| `w1` | `(1, N)` | float32 | UB | weight/bias 的 fp32 副本（复用） |
| `s1`/`mean`/`s2`/`var`/`rstd` | `(block_m, 1)` | float32 | UB | 行统计量（s2 兼作 sqrt(var) 暂存；各 4·bm B，可忽略） |

**fp32 路径**（无 cast）：`x_f32` + `sq_f32`（residual32 → d² → weight/bias 展开）+ `w1` + 5 个 (bm,1)；无 stage。

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L0/L1 参与）：

GM[x]        --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[x_f32]
GM[residual] --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[sq_f32] --vadd--> UB[x_f32 = s]
UB[x_f32] --vcast(rint)--> UB[stage] --T.copy(双slice)--> GM[residual_out]
UB[stage] --vcast(rint)--> UB[x_f32 = s32]
GM[weight] --T.copy--> UB[stage(1,N)] --vcast--> UB[w1] --vbrc--> UB[sq_f32] --vmul--> UB[x_f32]
GM[bias]   --T.copy--> UB[stage(1,N)] --vcast--> UB[w1] --vbrc--> UB[sq_f32] --vadd--> UB[x_f32]
UB[x_f32] --vcast(rint)--> UB[stage] --T.copy(双slice)--> GM[output]
全部计算（reduce/v-op/vbrc）在 UB 内闭环，无中间 GM 往返、无跨引擎传输。
```

关键约束：GM 不可直达 fragment/L0——本设计不经 fragment，全部经 UB；v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1）。

### 4.5 UB 内存预算

预算目标：总驻留 ≤ 192KB ÷ 1.7（auto-multi-buffer 膨胀裕度，TRAP-UB-multibuffer-inflation + CONST-capacity-910B2C）≈ **113KB**（`_UB_SAFE_BYTES = 115_000`）。总驻留 = `bpe·bm·N + 4N`（4N = `w1`）。

**fp16/bf16（10 B/元素/行 = stage 2 + x_f32 4 + sq_f32 4；另 +4N）**：

| N | block_m 默认 | 驻留 (B) | ×1.7 膨胀后 | ≤192KB？ |
|---|-------------|----------|------------|----------|
| 4096 | 2 | 98,304（96KB） | 167KB | ✓ |
| 8192 | 1 | 114,688（112KB） | 195KB | ✓（临界，99.2%） |

**fp32（8 B/元素/行 = x_f32 4 + sq_f32 4；另 +4N）**：

| N | block_m 默认 | 驻留 (B) | ×1.7 | ≤192KB？ |
|---|-------------|----------|------|----------|
| 4096 | 3 | 114,688（112KB） | 195KB | ✓（临界） |
| 5120 | 2 | 102,400（100KB） | 174KB | ✓ |
| 8192 | 1 | 98,304（96KB） | 163KB | ✓ |

默认公式：`block_m = min(8, max(1, (UB_SAFE_BYTES − 4N) // (bpe·N)))`；工厂层 guard 拒绝超预算 block_m（lerp E1 模式）。Stage 4 扫描空间：N=4096 fp16 bm∈{1,2}、N=8192 fp16 bm=1（临界，若实测膨胀 >1.7 触编译溢出则回退 N-tile 或减 w1 计划）；编译期 UB 溢出报错为天然 guard。**注**：`constants.md` CONST-capacity-910B2C 明确 auto-multi-buffer=false 的显式 alloc 结构实测 ≈ 手工核算（×1.10–1.12，甚至 <1），故上表 ×1.7 为保守上界。

### 4.6 动态轴定义

无运行时动态轴。M、N 均为工厂期常量（per-build JIT 特化，源 lru_cache 工厂同款语义）；尾块 real_m 是 kernel 内 `T.min` 标量。Op 层（`FusedAddLayerNormFwdOp`）把任意前导维 reshape 成 2D——host 元数据视图操作。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[4, 5], target="npuir")   # 双输出在 index 4/5（源契约），NPU 编译目标
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：仅 reduce_sum + v-prefix 逐元素运算 + vbrc 广播，无 matmul（decision-tree.md「含归约 → Developer → alloc_shared → UB」分支）。

### 5.2 Block 划分

```python
block_m = min(8, max(1, (UB_SAFE_BYTES - 4 * N) // (bpe * N)))  # §4.5 表；仅沿 M 分块
block_n = N                                                      # 全 N 驻留（§1.6.3 主选，不分块）
num_logical = ceil(M / block_m)                                  # 逻辑任务数（行块数）
num_kernels = min(num_logical, 40)                               # 实际启动内核数（≤ 物理核）
num_local_tasks = ceil(num_logical / num_kernels)                # 核内串行任务数（静态）
```

**选择理由**：N 维不分块——全 N 驻留使 x/residual 单遍 GM 扫描（R3：流量 4MN·eb 全候选最小）；M 维分块——行间独立；block_m 由 UB 预算决定（§4.5），非 GPU 的结构性 block_m=1（§0.6 R2）。

### 5.3 约束分析

- **对齐约束**：UB 32B 对齐由编译器管理（`docs/Tilelang.language/内存操作/T.alloc_shared.md`）；N 无 256 对齐要求（GPU 约束已消除）；尾轴向量宽整除性见 §1.6.3（manifest 形状全 0 浪费）。
- **UB 容量**：§4.5 表，全负载域 ≤ 114,688B（×1.7 ≤ 195KB）✓。
- **L0/L1/分形**：不适用（纯 Vector）。

### 5.4 注意事项

- **M % block_m ≠ 0（行尾）**：`real_m = T.min(block_m, M − off_m)`，src/dst 双显式 slice 装载/写出；v-op 全 buffer 操作，垃圾行按行隔离、输出截断丢弃（§0.6 R3）。
- **N 非对齐**：无 padded 路径——精确 N 缓冲 + slice（GPU 的 needs_pad 双路径在 NPU 统一为单路径）。
- **wrapper 默认 block_m**：NPU wrapper 由工厂默认治理；集成期/Stage 4 以 §4.5 表值或扫描值覆盖。
- **block_m 上限 guard**：工厂层 ValueError（§3.3）。
- **双输出写出**：`residual_out` 在流水线中段写出（步骤 2），`output` 在末尾写出（步骤 6）；两者共用 `stage`（时序不重叠——residual_out 写完后 stage 才被 weight/bias 装载复用）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数**：**20**（AI Core，Ascend910B3）。实查记录（2026-10-14，本设计执行）：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  n = NPUUtils.get().get_aicore_num()   # -> 20
  vector_cores = n * 2                  # 纯 Vector 算子翻倍 -> 40
  ```
  与 `constants.md` CONST-aicore-910B2C 的 2026-10-12 hardsigmoid 实测注（本机 `get_aicore_num()` 返回 20，设备档 910B3）一致；禁止按 24 假设。
- **逻辑核数**：`num_logical = ceil(M/block_m)`（仅 M 维分块，block_N=N 不分块）：

  | workload | shape | dtype | block_m | num_logical |
  |----------|-------|-------|---------|-------------|
  | smoke-llama | 64×4096 | fp32/fp16/bf16 | 2 / 3 | 32 / 22 |
  | llama-8b-prefill | 2048×4096 | fp16/bf16 | 2 | 1024 |
  | llama-8b-decode | 1×4096 | bf16 | 2 | 1 |
  | llama-70b-prefill | 2048×8192 | fp16/bf16 | 1 | 2048 |
  | llama-70b-decode | 1×8192 | bf16 | 1 | 1 |
  | llama-13b-prefill | 2048×5120 | fp32 | 2 | 1024 |

- **规模判定与分核方案**：persistent 统一式 `num_kernels = min(num_logical, 40)`，核内 `T.serial(num_local_tasks)` grid-stride（`block_id = i·num_kernels + cid`），`if block_id < num_logical` 守卫；**循环边界静态**（num_local_tasks 为 host 层 Python 编译期常量，PL-1.5 折叠先例）：
  - smoke（22 ≤ 40）与 decode（1 ≤ 40）：**无需适配**——min() 自然退化为 num_kernels=num_logical、每核 1 task，无串行轮次；
  - llama-8b-prefill / 13b-prefill（1024 > 40）：**极大规模分支**——固定启动 40 内核、核内 serial 26 task（1024 = 40×25 + 24：24 核 26、16 核 25，不平衡 ≤1 task ≈ 3.8%）；
  - llama-70b-prefill（2048 > 40）：同上——40 内核 × 52 task（2048 = 40×51 + 8：8 核 52、32 核 51，不平衡 ≤1 task ≈ 1.9%）。
  该形态消除逻辑核超发的串行调度开销（core-split-strategy §1），并获 grid-stride 聚集窗口的 MTE2 访问效益（PL-1.6 实测 +3.2%）；ada（同族）与 lerp（同流量形态）为已调优先例。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 block/task 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务维（M 行块） | 核级并行 + 核内串行 | `T.Kernel(num_kernels)` + `T.serial(num_local_tasks)` | persistent 分核（§5.5）；行块间独立 |
| 元素维（N） | 向量化（无循环） | v-op / reduce_sum / vbrc 全 buffer 操作 | §1.6.2 #1-#9 |
| 尾块 | 标量 min + slice | `T.min` + 双显式 slice | §1.6.2 #11 |

### 6.2 循环伪代码

见 §3.3（结构：`with T.Kernel(num_kernels, is_npu=True) as (cid, _):` → buffer 分配（serial 外，跨 task 复用）→ `for i in T.serial(num_local_tasks):` → `if block_id < num_logical:` → 6 段计算体）。

### 6.3 流水线优化

**不使用显式 `T.Pipelined`**。理由：单任务内为顺序依赖链（装载→加→舍回→reduce→center→reduce→归一化→仿射），无 K 维迭代可重叠；任务间的装载/计算 overlap 交由编译器 auto multi-buffer——PL-1.6 实测（同款多输入多输出流量）：≥16M 元素档全部向量 pass 隐藏于 MTE2 窗口内（delta ≤1.1µs）。Stage 4 选项：显式双缓冲（x/residual 预取下一 task）与 `T.Pipelined` 包裹 serial 循环——若 baseline 偏离 §1.6.0 估算下界 >2x 再评估。

### 6.4 尾块处理

M % block_m ≠ 0：最后一个行块的 real_m < block_m——装载/写出双显式 slice 截断；v-op 全 buffer（垃圾行按行隔离、输出丢弃）。num_logical % num_kernels ≠ 0：if 守卫跳过越界任务。N 无尾块（精确 N）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步指令，无手动 `T.sync_block_set/wait`：

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| T.copy(GM→UB) 后 | 自动 | 编译器在 MTE2→V 拓扑自动插 pipe_barrier |
| v-op / reduce / vbrc 之间 | 自动（顺序执行） | 同引擎顺序依赖 |
| vcast 后 | 自动 | 同上 |
| T.copy(UB→GM) 前 | 自动 | 编译器在 V→MTE3 拓扑自动插 pipe_barrier |
| serial 任务间 | 自动 | 同核串行，无并发 |

无核间协作（行独立，无跨核归并）——无需 sync_block_set/wait、无需 workspace。

### 7.3 pass_configs 配置

无特殊 pass_configs。Developer 模式默认编译流水线（ada/logsumexp/lerp 同款）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（移植源仓测试基准 `FusedAddLayerNormTest.ref_program`），**不复刻 §0.6 的 NPU 算法**（无 vcast / fp32 中转 / persistent 结构 / vbrc）——保证验证独立性。

```python
import torch
import torch.nn.functional as F

def golden_fused_add_layer_norm(x, residual, weight, bias, eps=1e-5):
    """PyTorch 参考实现（源仓 tests/ops/test_fused_add_layer_norm.py:ref_program 移植）。

    输入: x/residual (M, N) 同形同 dtype；weight/bias (N,)；eps 默认 1e-5
    输出: (y, residual_out) —— y 同形同 dtype；residual_out = round_dtype(x + residual)
    语义（§0.1）: 先 fp32 相加、舍回输入 dtype 得 residual_out，再对该值做 LayerNorm
                  （fp32 opmath + 单次舍回 .to(x.dtype)）
    """
    N = x.shape[-1]
    added = (x.float() + residual.float()).to(x.dtype)              # residual_out（单次舍回）
    y = F.layer_norm(added.float(), (N,), weight.float(), bias.float(), eps).to(x.dtype)
    return y, added
```

**独立性说明**：golden 直接调用 `torch.nn.functional.layer_norm`（独立的 PyTorch 实现），不含 NPU 设计中的持久化分核、vbrc 列广播、fp32 中转链、vsqrt+vdiv 组合等任何算法特征；NPU 算法若与 golden 数值一致即证明迁移正确。

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_fused_add_layer_norm.py` `_get_tolerances()`：

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

fp32 中转链对齐 golden opmath（TRAP-fp16-opmath-golden：≤1 ulp fp16；D-1 verify_equiv 佐证：fp16/bf16 逐位一致）；rstd 走 vsqrt+vdiv 组合避开近似 vrsqrt（TRAP-vrsqrt-plain-precision，fp32 1e-5 门禁必需）。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」（完整 L1/L2/Boundary 分层套件交由 tilelang-op-develop）：

| # | Shape (M, N) | dtype | block_m | 验证目标 | 容忍度 |
|---|-------------|-------|---------|---------|--------|
| L0-1 | (64, 4096) | fp32 | 3 | 对齐 shape fp32 直通路径（无 vcast） | 1e-5 |
| L0-2 | (64, 4096) | fp16 | 2 | 对齐 shape fp32 中转链 | 1e-3 |
| L0-3 | (64, 4096) | bf16 | 2 | bf16 → fp32 → bf16 路径 | 1.6e-2 |
| L0-4 | (1025, 4096) | fp16 | 2 | **M 尾块**（1025 % 2 = 1，垃圾行丢弃） | 1e-3 |
| L0-5 | (1025, 4096) | bf16 | 2 | M 尾块 + bf16 | 1.6e-2 |
| L0-6 | (2048, 4096) | fp16 | 2 | manifest llama-8b-prefill | 1e-3 |
| L0-7 | (1, 4096) | bf16 | 2 | manifest llama-8b-decode（单行） | 1.6e-2 |
| L0-8 | (2048, 8192) | fp16 | 1 | manifest llama-70b-prefill（bm=1 路径，临界 UB） | 1e-3 |
| L0-9 | (1, 8192) | bf16 | 1 | manifest llama-70b-decode | 1.6e-2 |
| L0-10 | (2048, 5120) | fp32 | 2 | manifest llama-13b-prefill（非 2 的幂 N） | 1e-5 |
| L0-11 | (2, 512, 4096) 3D | fp16 | 2 | Op 层 reshape 路径（host 元数据视图） | 1e-3 |
| L0-12 | (16, 1152) | fp16 | 8 | 自然非对齐 N（防御性） | 1e-3 |
| L0-13 | (1, 2) | fp32 | 1 | 微型 shape（N=2，防御性） | 1e-5 |

**通过条件**：全部项通过 atol/rtol（两输出各自比对）；对照 = §8.1 golden（CPU）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **UB auto-multi-buffer 膨胀 ~1.7x**（TRAP-UB-multibuffer-inflation） | bm 表按 1.7 裕度预算；N=8192 fp16（112KB，临界 99.2%）若实际膨胀 >1.7 则编译失败 | 工厂 guard + 失败时降 bm 不可能（已是 1）→ 回退：去掉独立 w1（reload 复用 sq_f32 计划）或 N-tile 流式变体（§1.6.3 #2）；`constants.md` 注显式 alloc 结构实测 ≈ 手工核算（×1.10–1.12），实际余量更宽 |
| **roofline 常数 stale** | §1.6.0 估算引用的 CONST-* 均被 kb_stale_check 标记 | 估算仅作排序与对账基线；Stage 4 Phase 1 按硬件成本模型 §3 回填偏差行 |
| **bf16 无 Vector 算术**（vadd/vmul/vsub/reduce/vbrc 之外中 bf16 ×） | bf16 直算编译失败 | fp32 中转链（§0.6 R4，必需） |
| **`T.vrsqrt` 近似（2.9e-3）** | fp32 1e-5 门禁大面积违反 | vsqrt+vdiv 组合（§1.6.1 O2，TRAP-vrsqrt-plain-precision） |
| **未文档化列广播税** | 若直接用 `[M,N]×[1,N]` vmul/vadd 则 +12~16%（PL-1.15） | vbrc 显式展开（§0.6 R5） |
| **TRAP-vbrc-same-shape-empty-broadcast** | bm==1 时 vbrc 同形触发 MLIR verify fail | 工厂层 bm==1 分支直接同形 vmul/vadd（§3.3） |
| **未初始化垃圾行流经 v-op/reduce** | 尾任务垃圾行统计量无意义 | 按行隔离 + 输出截断（logsumexp/ada 已验模式）；L0-4/L0-5 用例捕获 |
| **N > 驻留上限**（fp16 约 N>~11000） | UB 超预算 | 超出本任务域（manifest max N=8192）；工厂 guard 报错拦截；后续走 N-tile 流式（§1.6.3 #2） |
| **weight/bias 逐 task 重读（bm=1）** | 最坏读侧 2MN→4MN | 按 L2 驻留计不入 HBM 地板（§1.6.0 口径）；Stage 4 可 eval 一次性预载（UB 换流量） |
| **fp32 求和序 vs golden** | torch F.layer_norm 内部序与 NPU reduce 树序不同 | 容差判定（fp32 1e-5）；verify_equiv 显示两式 fp64err 同量级 |
| **inf/NaN 行为** | 全 inf 行 → NaN 输出（公式自然传播） | 与公式语义一致；测试用例均为有限随机输入（源仓同款）；verify_equiv 已对照两式传播一致性 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_m 超表 / 膨胀 >1.7 | 编译失败 `ub overflow` | 降 bm / 去 w1 / N-tile；§4.5 表 |
| bf16 直传 v-op | 未做 vcast fp32 | 编译错误（dtype 不支持） | fp32 中转链 |
| UB dst 非零起点切片 | 手写列偏移 slice | VEC 对齐错误（TRAP-UB-dst-align） | 全部 0 起点双显式 slice |
| T.copy base+size 跨步形态 | 图省事用索引+size | 区域语义静默误读（TRAP-T-copy-region-semantics） | 双显式 slice 形态 |
| 跨 dtype T.copy 当 cast | GM fp16 → UB fp32 一步 | 隐藏 VCast 舍入不可控（TRAP-C12） | 显式 vcast(rint) + 同 dtype copy |
| vbrc 同形 src/dst | bm==1 仍调 vbrc | MLIR verify fail（空广播维） | bm==1 分支直接同形 vmul/vadd |
| `T.vrsqrt` 误写 `T.rsqrt`（调用量） | 文档文件名 T.rsqrt.md 误导 | AttributeError | 代码导出名 `T.vrsqrt`（本设计不用，走 vsqrt+vdiv） |
| 列广播直接 vmul | 想省 vbrc 用 [M,N]×[1,N] | 可编译但有 +12~16% 税（PL-1.15） | vbrc 显式展开 |
| 忽略第二输出 | 只写 output，漏 residual_out | 精度比对 residual_out 全错 | 步骤 2 显式写出 residual_out（§3.3） |
| if 体内新定义变量跨块引用 | TVM parser 作用域（TRAP-C11） | Undefined variable | if 体内只写预分配 buffer；dtype/bm 分支置于工厂层 |
| 双输出 out_idx 顺序反 | out_idx=[5,4] | wrapper 拿到 (residual_out, y) 错位 | out_idx=[4,5]（y 在前，residual_out 在后，对齐源） |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| N > 11000（fp16/bf16） | 超出驻留域：工厂 guard ValueError 拦截；后续如需支持走 N-tile 流式变体（§1.6.3 #2：T.serial 分块 + clear=False 累加 + 输出遍重读，流量 5MN·eb）——不在本任务范围（manifest max N=8192） |
| M = 1（decode） | num_logical=1 → 单内核单任务；bm 仍按预算（real_m=1）；延迟 bound ~10µs（§1.6.0 估算） |
| 3D 输入 (B, S, H) | Op 层 reshape (B·S, H) 后调 kernel（host 元数据视图）；L0-11 覆盖 |
| 常量行（var=0） | rstd = sqrt(eps)/eps（数学定义）；verify_equiv 角点已对照 |
| 全 inf / NaN 行 | 自然传播（与源/golden 一致）；verify_equiv 已对照两式传播一致性 |
| wrapper 默认 block_m | 功能 valid；性能优化留给集成期 config 覆盖 / Stage 4 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/
├── DESIGN.md                          # 本设计文档
├── verify_equiv.py                    # D-1 等价性机器验证脚本（已执行，EQUIV_PASS）
├── _fused_add_layer_norm_kernel.py    # 算子实现（Stage 3 产出）
└── RETROSPECTIVE.md                   # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | O1+O2+O3+O4 EQUIV_PASS（24 案例）；矩式拒绝证据（判别段 A FAIL） |
| `_fused_add_layer_norm_kernel.py` | ⬜ 待实现（Stage 3） | 工厂 + kernel + golden + L0 测试 |

### 10.3 命名规范

- 项目目录名：`fused_add_layer_norm`；算子目录名 / 实现文件：`_fused_add_layer_norm_kernel`（snake_case，不裁剪不变换）
- Golden 函数：`golden_fused_add_layer_norm(x, residual, weight, bias, eps)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）——本文档
2. ✅ 等价性机器验证（verify_equiv.py，D-1）
3. ⬜ Golden 函数（§8.1，验证基准）
4. ⬜ 算子实现 `_fused_add_layer_norm_kernel.py`（§3.3 伪代码 → 代码）+ L0 门槛测试（§8.3 全 13 项）
5. ⬜ 精度比对（vs golden，两输出各自通过全部 L0）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 接口参数不变 | ✅ | 工厂签名 `(M, N, eps, dtype)` 与源一致 |
| `@T.prim_func` 参数名/顺序不变 | ✅ | `main(x, residual, weight, bias, output, residual_out)` + `out_idx=[4,5]`（源契约，源名 y/residual_out） |
| threads 移除（K9） | ✅ | `_func(block_m)`（TRAP-threads-kwarg-noop） |
| host 侧不改输入 | ✅ | 全部计算在 kernel 内；无 host padding/permute |
| 从源码推断规格（不问用户） | ✅ | §0.1/§0.2 从源码 + manifest 解读；输出 shape 已回填标注依据 |
| 双输出契约 | ✅ | §0.1/§3.3/§4.2：residual_out = round_dtype(x+residual)；output = 归一化 y |
| 迁移纪律（不照抄 GPU 三维 Kernel/warp shuffle） | ✅ | §0.5/§0.6 四态判定 + NPU 重设计（persistent、T.min+slice、vbrc） |

---

## 11. 修订日志

**revision_index: 0（first_design）**

首次设计（migration-harness，`_fused_add_layer_norm_kernel`，2026-10-14）。无上一版；无历史否决路径。

**关键设计决策摘要**：

| # | 决策 | 依据 |
|---|------|------|
| 1 | 中心化两遍方差（非矩式） | §1.6.0 R1#2 / §1.6.1 判别段 A：矩式 fp32 违反率 0.13–0.21（D-1 机器证据） |
| 2 | 全 N 驻留 + M 行块 persistent 分核（num_kernels=min(logical,40)） | §1.6.0 R3（流量最小）/ §1.6.3 主选 #1 / §5.5（实查 20→40） |
| 3 | 尾块 T.min + src/dst 双显式 slice + 垃圾行丢弃 | §0.6 R3（TRAP-T-copy-region-semantics） |
| 4 | bf16/fp16 → fp32 中转链（显式 vcast(rint)） | §0.6 R4（TRAP-fp16-opmath-golden / TRAP-C12） |
| 5 | rstd = sqrt(v)/v（vsqrt+vdiv），避开近似 vrsqrt | §1.6.1 O2（TRAP-vrsqrt-plain-precision） |
| 6 | 舍回后统计（对齐 golden），residual_out 与 y 共用舍回值 | §1.6.1 O4 / 判别段 B |
| 7 | 列因子走 vbrc 核内展开（bm==1 直接同形） | §0.6 R5 / PL-1.15 / TRAP-vbrc-same-shape-empty-broadcast |
| 8 | 双输出 out_idx=[4,5]，residual_out 中段写出、output 末段写出 | §0.1 双输出契约 |
