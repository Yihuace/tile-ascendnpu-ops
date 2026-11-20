# `_flashattn_bwd_preprocess_kernel` 算子设计文档

> **算子**：`_flashattn_bwd_preprocess_kernel`（承载 `FlashAttnBwdPreprocessKernel`，MultiHeadAttentionBwdOp backward 三件套之一）
> **项目**：`multi_head_attention`（NPU 侧追加进既有 `multi_head_attention` 包，不改动 fwd 产物）
> **迁移来源**：`/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`（第 26 至 53 行定义，第 56 至 96 行为 Kernel 类）
> **目标后端**：`target="npuir"`（Ascend910B3，物理 AI Core 20 / 纯 Vector 核 40 / UB 192KB，均为本机实查）
> **编程模式**：Developer（`op_requirements.programming_mode`）
> **冻结版本**：本文件为 Stage 1 交付物；`verify_equiv.py` 已执行（§1.6.1 结果表）

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

**数学语义**：`FlashAttention` 反向传播的预处理器，逐行点积（row-wise dot product）：

$$
\delta[b,h,s] \;=\; \sum_{d=0}^{D-1} \mathrm{fp32}\big(o[b,s,h,d]\big)\cdot \mathrm{fp32}\big(do[b,s,h,d]\big)
$$

即 `delta = rowsum(o ∘ do)`——FlashAttention 论文中的逐行统计量 $D_i = \sum_j (o_{ij}\circ do_{ij})$，作为主 kernel 中
`dst_cast = P * (dV - delta) * scale` 的减项输入。

| 语义要素 | 解读（源码证据） | 是否必须保持 |
|----------|------------------|--------------|
| 数学映射 | 沿 head-dim `D` 的逐行点积，逐 `(b,h,s)` 行独立；无跨行耦合 | ✅ 必须 |
| I/O 契约 | `o` / `do`：`(batch, seq_len, heads, dim)`，BSHD，head-dim 最内连续；`delta`：`(batch, heads, seq_len)` **fp32**。输出 layout 与输入不同（`H`/`S` 轴互换），是本迁移的核心结构约束 | ✅ 必须 |
| dtype 语义 | 输入 `fp16`/`bf16`；**乘积与累加在 fp32**（源码 `acc = T.alloc_fragment([blk, blk], accum_dtype="float")` 且 `acc[i,j] += o_frag[i,j]*do_frag[i,j]`）；输出 fp32，**无最终舍回**（与 postprocess kernel 不同） | ✅ 必须 |
| 规约语义 | 规约轴 = `dim`（最内）；**累加顺序**：先按 `blk=256` 的 D 分块把乘积累加进 `(blk,blk)` fp32 tile，循环结束后对 D 轴做一次 tile 归约（`T.reduce_sum(acc, delta_frag, 1)`）；无 keepdim 语义（输出降到 1 维） | 容差内保持（见 §1.6.1 P2） |
| 边界语义 | 无 mask、无 causal、无 eps、无归一化；`NaN`/`Inf` 按 IEEE 自然传播（乘积与加法）；`D = 0` → 空和为 `+0.0`；`S` 非 `blk` 整数倍时由 `T.ceildiv` 多头/超发 block 覆盖，源实现未做尾块裁剪（>0 的越界读由 `ceildiv` 产生的空 block 承担，见 §0.3 注） | ✅ 必须 |
| 空/极值 | 全零输入 → `delta = +0.0`；饱和输入（fp16 $\pm65504$）乘积可达 $4.29\times 10^9$（fp32 可容纳，不进 inf）；`±inf` 行 → `±inf` 或 `NaN`（`inf*0`） | ✅ 必须 |
| host 侧语义 | **无 host 预处理**：`forward(o, do)` 直接调用 kernel，`out_idx=[2]` 由框架分配输出。op 层只做 `do = do.contiguous()`（`tileops/ops/attention/gqa.py` 第 1379 行）——属元数据/连续性视图，非算子契约 | ✅ 保留 |

> **语义解读完成标准核对**：§8.1 的 golden 函数完全按本表实现（`(o.float() * do.float()).sum(-1).transpose(1, 2)`），
> 不引用任何 NPU 侧算法结构，验证独立性成立。
> **歧义项**：源码注释把 `do` 写作「gradient of output **reciprocal**」（第 36 行）——该措辞不准确（它是 `output` 的梯度，
> 与 reciprocal 无关）；语义以主 kernel 的使用点为准（`qkt*(dst - delta)`），§0.3 已交叉核对，**不属待确认项**。

### 0.2 源算子输入输出规格

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `o` | 输入 | `(batch, seq_len, heads, dim)` | float16 / bfloat16 | attention 前向输出，BSHD |
| `do` | 输入 | `(batch, seq_len, heads, dim)` | float16 / bfloat16 | `o` 的梯度，同 shape 同 dtype |
| `delta` | 输出 | `(batch, heads, seq_len)` | **float32** | `out_idx=[2]` 自动分配；下游主 kernel 以 `delta[bz, bx, s-slice]`（S 连续）读取 |

工厂签名（源）：`_flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype) -> Callable`，
外层**直接**由 `@tilelang.jit(out_idx=[2])` 装饰（注意：本工厂**没有** `pass_configs`，也**没有** `compile_flags`——
源文件中 `compile_flags=["-O3","-DENABLE_BF16"]` 只出现在 `_mha_bwd_kernel` / `_gqa_bwd_kernel` /
`_*_wgmma_pipelined_kernel` 等**其它**工厂上，第 188 至 193、339 至 344、528 至 532、681 至 685 行）。
无 `functools.lru_cache`（其它工厂有）。`blk = 256` 为**工厂内固定常量**，不暴露为参数；Kernel 类的 `default_config` 为空、
`autotune_configs` 未定义（第 56 至 96 行的文档字符串明确「launch geometry is fixed」）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源码第 33 至 51 行，逐语句归位，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|------|------|------|------|----------|
| S0 | 三维 grid 任务划分 `T.Kernel(heads, ceildiv(seq_len, blk), batch) as (bx, by, bz)`，`bx`=head、`by`=seq 块、`bz`=batch | — | block 索引 | 并行结构（非数学语义） |
| S1 | `T.clear(acc)`（`(256,256)` fp32 fragment） | — | `acc = 0` | 累加初值（`+0.0`） |
| S2 | D 轴分块循环 `for k in range(ceildiv(dim, blk))` | — | — | 支持 `dim > 256` |
| S3 | `T.copy(o[bz, by*blk:(by+1)*blk, bx, k*blk:(k+1)*blk], o_frag)`（`(256,256)` dtype fragment） | `o` | `o_frag` | 读 `o` 的一个 `(seq 块, dim 块)` 二维跨步子区域（行 stride = `heads*dim`） |
| S4 | 同 S3 拷 `do` → `do_frag` | `do` | `do_frag` | 读 `do` |
| S5 | `for i, j in T.Parallel(blk, blk): acc[i,j] += o_frag[i,j]*do_frag[i,j]` | `o_frag`,`do_frag` | `acc` | **乘积 + fp32 累加**（FMA，fp32 域） |
| S6 | `T.reduce_sum(acc, delta_frag, 1)`（`(256,256) → (256,)`） | `acc` | `delta_frag` | **沿 D 轴的行归约** |
| S7 | `T.copy(delta_frag, delta[bz, bx, by*blk:(by+1)*blk])` | `delta_frag` | `delta` | 写出 S 连续段（输出 H-major/S-minor） |

**数据流图（源硬件视角，CUDA）**：

```
GM[o (B,S,H,D)] --T.copy(2D 跨步子块, 行 stride=H·D)--> SMEM/fragment[o_frag (256,256) dtype]
GM[do]          --T.copy(同上)------------------------> fragment[do_frag (256,256) dtype]
   o_frag × do_frag --T.Parallel(i,j), fp32 FMA--> fragment[acc (256,256) fp32]   (K 循环复用)
   acc --T.reduce_sum(dim=1,=warp shuffle 树归约)--> fragment[delta_frag (256, fp32)]
   delta_frag --T.copy--> GM[delta (B,H,S) fp32]（S 连续）
```

**循环与并行结构**：grid = `(heads, ceildiv(S,256), batch)` 三维（每 block 一个 `(head, seq 块, batch)` 任务）；block 内
`threads` 默认（128）；`T.Parallel(256,256)` 把 65536 个乘积映射到线程；行归约由 fragment 的 `T.reduce_sum` 完成
（CUDA 上是 warp/CTA 内树形归约）。**无跨 block 协作、无原子加、无 workspace**（行完全独立）。

**host 侧逻辑**：无（无 pad / reshape / cast / permute）。仅 op 层 `do.contiguous()`。

> **尾块注（源码行为核对）**：`by` 覆盖 `ceildiv(S,256)` 个块，当 `S % 256 ≠ 0` 时最后一个 block 的
> `o[bz, by*blk:(by+1)*blk, bx, ...]` 区域**跨越 `S` 边界**——源实现未加谓词，依赖 `T.copy` 的区域语义（越界部分被裁）
> 与 `delta_frag` 写出时同样裁掉越界行。迁移侧必须**显式**处理该尾块（§0.6 R5），不得依赖隐式裁剪。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 代码特征 | 目的 | 依赖的源硬件特性 | 迁移初判 |
|---|----------|----------|------|------------------|----------|
| 1 | **SMEM/fragment tiling + 复用** | `T.alloc_fragment([blk,blk])` 三块 tile，K 循环内复用 | 每元素只从 GM 读一次（本算子是纯流式归约，无重读） | CUDA fragment/SMEM | 硬件强相关（NPU 等价 = UB） |
| 2 | **fp32 寄存器累加** | `acc` 为 fp32 fragment，FMA 累加 | 乘积/累加不在低精度域反复舍入 | 大寄存器堆 | 硬件强相关（NPU = UB fp32 缓冲 + v-op） |
| 3 | **D 轴分块（`blk=256`）** | `for k in range(ceildiv(dim, blk))` | 使任意 `dim` 都能在固定缓冲下处理 | 无（算法层） | 可移植（本域 `dim ≤ 128` 时退化为单次，§1.6.0 R2） |
| 4 | **block 级 grid 并行** | 三维 grid = (heads, S/256, batch) | 行间独立 → 完全并行 | 无（算法层） | 意图保留，形态重设计（NPU 一维 `T.Kernel` + persistent，§0.6 R1） |
| 5 | **fragment 归约（warp shuffle 树）** | `T.reduce_sum(acc, delta_frag, 1)` | 块内跨线程行归约 | warp shuffle | **重新设计**（NPU = `T.reduce_sum` 硬件归约，且归约轴/声明显式影响向量化，§0.6 R3） |
| 6 | **跨步 2D 区域拷贝** | `T.copy(o[bz, s-slice, bx, d-slice], o_frag)` | 单条 DMA 描述符覆盖 `blk×blk` 跨步子块 | memory coalescing | **保留**（npuir 同形态已验证，§0.6 R8） |
| 7 | **`out_idx=[2]` 输出自动分配** | 装饰器参数 | host 侧零开销 | 框架 | 保留（接口契约） |
| 8 | **threads 级元素并行** | `T.Parallel(blk, blk)` | 元素级并行 | CUDA 线程 | **等价替换**（NPU = v-prefix 向量 op，§0.6 R6） |

> **未识别 ≠ 不存在**：本算子源码极短，逐语句已全部归类（S0 至 S7 + 上表 8 项）。与同文件其它工厂相比，本工厂**没有**
> `pass_configs`（无 `TL_ENABLE_FAST_MATH`）、**没有** `compile_flags`、**没有** `T.Pipelined`、
> **没有** `T.gemm`/wgmma、**没有** swizzle/`annotate_layout`/`atomic_add` —— 这些均**不属于**本函数的优化手段，
> 迁移时不得从本文件其它 kernel 误搬（尤其不得搬 `make_dq_layout` 的 Ampere 8×8 fragment 重排，见 §3.5.2）。

### 0.5 硬件耦合性分析与 NPU 适配决策

三层判定模型（`migration-analysis.md` §5.1）+ 四态处置（§5.2）。逐项依据见「依据」列（映射表条目 / `examples/` / `docs/`）。

| 项 | 层 | 源方案 | 处置 | 依据 |
|----|----|--------|------|------|
| 数学映射 / I/O 契约 / fp32 累加 / 边界语义 | 语义层 | §0.1 | **保留** | `migration-analysis.md` §5.1（语义层无条件保留） |
| `delta` 的 S 连续 layout（H-major） | 语义层 | `(B,H,S)` | **保留** | 下游主 kernel 按 `delta[bz, bx, s-slice]` 读，layout 是契约 |
| 三维 grid `T.Kernel(heads, ceildiv(S,blk), batch)` | 优化层 | 三维 block | **重新设计** | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1（`is_npu` 最多 2 维 block）+ 映射表「三维 Kernel → 一维展开」 |
| `T.alloc_fragment` 计算缓冲 | 优化层 | CUDA 寄存器/fragment | **等价替换** | 映射表「register 累加 → UB 中间缓冲」；`docs/Tilelang.language/数学操作/T.vmul.md` §2.3（v-op 操作数必须在 UB） |
| `blk = 256` 的 `(256,256)` 方形 tile | 优化层 | 256×256 fragment 对（各 128KB@fp16）+ 256KB fp32 acc | **重新设计** | CUDA fragment 无容量上限概念；NPU UB=192KB（`CONST-capacity-910B2C`），tile 必须按 UB 预算反推（§4.5） |
| `T.Parallel(blk,blk)` 逐元素 FMA 累加 | 优化层 | 线程级并行 + `acc[i,j] +=` | **等价替换** | NPU 等价 = `T.vmul`（`docs/Tilelang.language/数学操作/T.vmul.md`，支持 dst 与 src 同 buffer 的原位操作）；标量形态在本仓有标量化实测教训（§1.6.2） |
| `T.reduce_sum(acc, delta_frag, 1)`（二维 tile、归约最内轴） | 优化层 | warp shuffle 树归约 | **重新设计** | 映射表「warp shuffle 归约 → Vector 归约，需重排归约策略」；本仓实测：二维 `(C,plane)` 的 `reduce(dim=最内轴)` **被标量化**（`CONST-dual-output-whole-plane-reduce-ub` 的「归约轴形态」段），合法快形态 = 三维扁平声明 `(C,1,plane)` + `dim=2`（`PL-1.23-flat-spatial-axis-reduce`） |
| D 轴分块循环 `for k in range(ceildiv(dim, blk))` | 算法层 | 分块累加 | **保留（域内退化为单次）** | 纯数学分块累加、与硬件无关；本域 `dim ≤ 128` 时 UB 可整行驻留 → 单次归约（§1.6.0 R2/R3），分块形态作为超预算 fallback 保留（§9.3） |
| 跨步 2D 区域 `T.copy`（行 stride = `H·D`） | 优化层 | `o[bz, s-slice, bx, d-slice]` | **保留** | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 1/2（静态 1 维折叠、切片 extents）；本仓 npuir 同形态既有实现：`examples/deepseek_v32/sparse_mla_bwd.py`（同数学语义）、`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`（大 shape 已验证 attention 装载形态，§3.5.3） |
| `threads=128` | — | 本工厂**未使用** | **不适用** | 本工厂 `T.Kernel` 无 `threads` 参数（§0.2） |
| `pass_configs` / `compile_flags` | — | 本工厂**未使用** | **不适用** | §0.2；且 `tilelang.jit` 在 npuir 无 `compile_flags` 形参（预注入知识 `VP-2026-0158`，故本函数天然免疫） |
| `make_dq_layout`（Ampere 8×8 fragment 重排） | 优化层 | 本文件**其它** kernel 的 `annotate_layout` | **舍弃（不属于本函数）** | §0.2 范围界定；映射表「swizzle → NZ 分形布局，手动 swizzle 无意义」 |
| `supported_archs=[80,89,90]` | 宿主层 | CUDA SM 列表 | **舍弃** | wrapper 已按 K5 改写 `supported_archs = None`（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention.py`） |

**判定统计**：保留 **4** 项（语义层 2 + 算法层 1 + 优化层 1）、等价替换 **2** 项、重新设计 **3** 项、舍弃 **2** 项、
不适用 **2** 项（共 13 项）。**舍弃项均有理由**（见上表）；**无静默丢弃**（§0.6 末尾逐项承接核对）。

### 0.6 NPU 算法重设计

对 §0.5 判为「重新设计」的 3 项（**R1** 并行结构 / **R2** tile 形态 / **R3** 归约形态）逐项给出「源方案 → NPU 新算法 → 语义保持论证」；
另给出 NPU 适配必需的 4 项配套设计（**R4** fp32 中转链 / **R5** 尾块 / **R6** 元素级向量化 / **R7** 乘积原位复用）。
重设计直接承接 §1.6.0 调研结论（「单遍单块全 D 驻留 + 逐行独立一维 persistent 分核 + 三维扁平声明归约」）。

**R1（三维 Kernel → 一维）**

- 源方案：`T.Kernel(heads, ceildiv(S,blk), batch)`，每 block 一个 `(h, s-block, b)` 任务。
- NPU 新算法：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(num_logical, 40)`；
  逻辑任务 `uid` 用整除/取模分解回 `(b, h, s-block)`，核内 `for i in T.serial(num_local_tasks)` + `if uid < num_logical`
  做 grid-stride persistent 处理（§5.5）。
- 语义保持论证：仅改变**任务到核的映射**，每个任务内部的 `(b,h,s-block)` 计算完全不变；行间独立 → 任务顺序无关；
  纯索引算术（`uid → (b,h,s)`）为双射，无重复/遗漏（`num_kernels ≥ 1` 时 `uid = i·num_kernels + cid` 覆盖
  `[0, num_logical)` 全部取值，越界由守卫跳过）。边界语义不变（尾块见 R5）。

**R2（tile 形态 → UB 预算驱动）**

- 源方案：`(256,256)` 的 `o_frag`/`do_frag`（各 128KB@fp16）+ `(256,256)` fp32 `acc`（256KB）+ `(256,)` `delta_frag`。
- NPU 新算法：整行驻留形态 `(blk_s, 1, D)`，`blk_s = 元素预算 ÷ D`（§4.5 逐 buffer 复算）；三个 buffer
  `stage`（输入 dtype，2B/元素，`o`/`do` 顺序复用）+ `acc32`（fp32，4B/元素，`o` 的 fp32 载体 → 原位乘积，O1）
  - `aux32`（fp32，4B/元素，`do` 的 fp32 载体），声明量 10 B/元素（§4.3）。
- 语义保持论证：tile 只切片 **seq 轴**（行）与 **head-dim 轴**位置，不改变任何元素的计算与其分组：每个
  `delta[b,h,s]` 仍是同一个 `D` 元素点积；行间独立 ⇒ 行块切分不影响结果。`D` 轴**不切分**（整行驻留）
  ⇒ 归约集合与源完全一致（累加顺序差异见 §1.6.1 P2）。fp32 累加域不变。

**R3（归约形态 → 三维扁平声明 + 最内轴归约）**

- 源方案：二维 `(blk, blk)` tile 上 `T.reduce_sum(acc, delta_frag, 1)`（归约轴 = 最内轴），CUDA warp shuffle 树。
- NPU 新算法：UB tile 声明为 **三维** `(blk_s, 1, D)`，归约 `T.reduce_sum(prod3d, delta_ub, dim=2)`
  （`(blk_s,1,D) → (blk_s,1,1)`），再以 `T.copy(delta_ub[0:real_s, 0, 0], delta[bz, bx, s0:s0+real_s])` 写出。
- 语义保持论证：三维声明与二维是**同一块连续内存的形状重解释**（中间维恒为 1，不引入任何填充/重排）
  ⇒ 元素与内存序不变，`dim=2` 归约的数学集合与源的 `dim=1` 完全相同（同一条最内连续轴、同一个 `D`）。
  改的只是**声明形态**以命中 npuir 的向量化归约 lowering 判据（二维最内轴归约被标量化，见 `CONST-dual-output-whole-plane-reduce-ub`
  与 `TRAP-inner-axis-reduce-scalarize`；三维扁平形态为已验证快路径，`PL-1.23-flat-spatial-axis-reduce` 实测 166.26 → 16.68 µs）。
  输出写出的元素集合不变（S 连续段，尾块由 `real_s` 裁剪）。

**R4（`do` 的 fp32 中转链；bf16 无向量算术）**

- 源方案：`acc[i,j] += o_frag[i,j]*do_frag[i,j]`，dtype 输入经隐式提升在 fp32 域做 FMA。
- NPU 新算法：**显式**两段 `T.vcast(..., round_mode="rint")` 把 `o`/`do` 上抛 fp32，再 `T.vmul` 在 fp32 域求积。
- 语义保持论证：`fp16 → fp32` 与 `bf16 → fp32` 上抛**逐位精确**（尾数 11/8 位 ⊂ fp32 24 位），且两个上抛值的
  fp32 乘积**精确**（22/16 ≤ 24 位）——**机器证据**：§1.6.1 `verify_equiv.py` 的 P1（32 案例、最大 ulp 差 **0**、
  超容差 **0**）。故该链与源「fp32 FMA」给出逐位相同的乘积。必要性佐证：`docs/Tilelang.language/数学操作/T.vmul.md`
  §2.2.1 与 `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1 的 dtype 表 **bf16 列均为 ×**（无 bf16 向量算术），
  故 bf16 路径**必须**经 fp32 中转；`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条（Developer 跨 dtype copy
  自动插 `hivm::VCastOp`，RoundMode=RINT 不可定制）与 `TRAP-C12-copy-dtype-cast` 要求用**显式 `T.vcast` + 同 dtype copy**，
  故不采用「跨 dtype `T.copy` 当 cast」。
- **与 R2 的协同**：`stage` 一块缓冲顺序承载 `o` 与 `do`（第二次装载覆盖第一次），把声明量压到 10 B/元素。

**R5（尾块：非整除 `S`）**

- 源方案：依赖 `ceildiv` 超发 + `T.copy` 区域裁剪（§0.3 注，隐式）。
- NPU 新算法：`real_s = T.min(blk_s, S - s0)`（标量，1 次/任务），装载/写出用**双显式 slice**，v-op 与 `reduce` 作用于
  完整 `(blk_s,1,D)` buffer（垃圾行按行隔离，`dim=2` 归约逐行独立），**只写 `[0:real_s]`**。
- 语义保持论证：输出元素集合与源一致（`[s0, min(s0+blk_s, S))`）；垃圾行永不落 GM；垃圾输入（未初始化 UB）
  可能为 NaN/Inf，但逐行独立 ⇒ 不污染有效行。与同族前作同款形态（`examples/group_norm/_group_norm_kernel/DESIGN.md` §0.6 R3）。

**R6（元素级并行 → v-prefix 向量链）**

- 源方案：`T.Parallel(blk,blk)` 线程级 FMA。
- NPU 新算法：`T.vmul`（全 buffer，原位）替代逐元素循环；标量循环的盘点与理由见 §1.6.2。
- 语义保持论证：逐元素乘法的元素映射不变（无跨元素依赖、无广播），`T.vmul` 的 dtype/shape 规则（`[M,N]*[M,N]→[M,N]`、
  原位 `dst` 合法）与 §3.3 伪代码一致。

**R7（乘积缓冲复用，`§1.6.1` O1）**

- 源方案：`o_frag`/`do_frag`/`acc` 三块，乘积写入 `acc`。
- NPU 新算法：乘积**原位**写回 `acc32`（`T.vmul(acc32, aux32, acc32)`）。
- 语义保持论证：同一 fp32 乘法、同一目标元素、无别名冲突（src2 与 dst 不同 buffer）⇒ **逐位相同**；
  机器证据：§1.6.1 O1（32 案例、最大 ulp 差 0、逐位相等）。收益：少一块 4 B/元素的 UB 缓冲。

**源算法优化手段承接核对（§0.4 逐项，无静默丢弃）**：

| §0.4 项 | 意图 | 承接方式（→ 本设计） |
|---------|------|----------------------|
| #1 SMEM/fragment tiling 复用 | 每元素只读一次 GM | 承接：UB `stage` 双角色顺序复用 + 声明量压到 10 B/元素（§4.3），GM 扫描仍为 **1 遍**（§1.6.0 R3） |
| #2 fp32 寄存器累加 | 低精度不反复舍入 | 承接：fp32 UB 载体 + `T.vmul`/`T.reduce_sum` fp32 域（R4/R6） |
| #3 D 轴分块（blk=256） | 支持任意 `dim` | 承接：本域 `dim ≤ 128` 时退化为单次（归约集合不变）；超 UB 预算时以分块累加 fallback 保留（§9.3） |
| #4 block 级 grid 并行 | 行间完全并行 | 承接：一维 persistent 分核（R1/§5.5），行独立性质不变 |
| #5 fragment 归约 | 块内高效行归约 | 承接并**重设计**：三维扁平声明 + `dim=2` 硬件归约（R3），动机与意图（最快的行归约）一致 |
| #6 跨步 2D 区域拷贝 | 单描述符覆盖跨步子块 | **原样保留**（R8/§0.5） |
| #7 `out_idx=[2]` | host 零开销 | 原样保留（§3.3 接口） |
| #8 threads 级元素并行 | 元素级并行 | 等价替换为 v-prefix 向量链（R6） |

### 0.7 标杆实现（迁移参考）

| 参考 | 相似度 | 关键参考点 |
|------|--------|-----------|
| `examples/deepseek_v32/sparse_mla_bwd.py` | **同数学语义的既有 npuir 实现（最高）** | `preprocess()`：三维跨步 `T.copy(o[bz, s-slice, bx, d-slice], o_tmp)` + `T.vcast` + 累积 + `T.reduce_sum(acc, delta, dim=1)` + `T.copy(delta[:,0], Delta[bz, s-slice, bx])`；证实本设计的装载/写出形态在 `target="npuir"` 上可编译可跑 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention.py` | 极高（**wrapper 契约**） | `FlashAttnBwdPreprocessKernel` 的构造与调用（`_flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype_str)`、`forward(o, do)` 返回 `delta`）；K5/K8 已改写 `supported_archs=None`、去掉 `tune` |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/_multi_head_attention_bwd_kernels.py` | 极高（**迁移源工件**） | Stage 0 提取的 GPU 源码副本（第 60 至 86 行为本函数）；迁移对象与 §0.1 语义的对照基准 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高（**同族大 shape 已验证**） | attention 族三维跨步装载 `T.copy(q[bz, k_idx*bn:(k_idx+1)*bn, bx, :], q_frag)` 在绩效达标路径上被使用（证明该装载形态可达高带宽）；`NUM_KERNELS = 24` 为 910B2C 实查值（本机 910B3 = 20，§5.5） |
| `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | 高（**形态已验证**） | 三维 UB tile + `T.vcast` + `T.reduce_sum(..., dim=2)`（`(ch,h,w)→(ch,h,1)`）+ 一维 persistent 分核 + `T.min` 尾块 + 工厂 UB guard 的完整可运行范式 |
| `examples/group_norm/_group_norm_kernel/DESIGN.md` | 高（**同族行归约设计文档**） | §0 迁移分析结构、§1.6 调研/优化写法、§1.6.3 参数级 A/B 写法、§5.5 分核三要素实践、§8 L0 计划 |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` | 中高（同族已交付迁移） | persistent 分核模板、UB 预算 guard、fp32 中转链、L0 套件分层 |
| `/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py` | — （**源算子**） | 第 26 至 53 行（kernel）与第 56 至 96 行（Kernel 类）；第 1377 至 1381 行的 op 层调用链（`tileops/ops/attention/gqa.py`）证实 `delta` 的 S 连续读法 |
---

## 1. 概述

### 1.1 算子名称

`_flashattn_bwd_preprocess_kernel`（承载 `FlashAttnBwdPreprocessKernel`：MultiHeadAttentionBwdOp 的 backward 预处理 kernel，产出 `delta`）

### 1.2 功能描述

对 BSHD 布局的 `o`（attention 前向输出）与 `do`（其梯度）沿 **head-dim 轴 `D`** 做逐行点积，得到 `(B,H,S)` fp32 的
逐行统计量 `delta`，供主 kernel `_gqa_bwd_kernel` 使用。**NPU 侧实现与源算法的差异**（§0.6 / §1.6.0）：数学内核不变
（逐行点积 + fp32 累加 + 单遍扫描），并行组织从三维 grid 改为**一维 persistent 分核**，UB tile 改为
**UB 预算驱动的整行驻留形态 `(blk_s, 1, D)`**，行归约从二维最内轴归约改为**三维扁平声明的 `dim=2` 归约**，
元素级 `T.Parallel` 累加改为 `T.vmul` 向量链，乘积缓冲原位复用（O1）。

### 1.3 数学公式

$$
\delta[b,h,s] \;=\; \sum_{d=0}^{D-1} \mathrm{fp32}\big(o[b,s,h,d]\big)\cdot \mathrm{fp32}\big(do[b,s,h,d]\big),
\qquad
\delta \in \mathbb{R}^{B\times H\times S}\ (\text{fp32})
$$

`o, do ∈ {fp16, bf16}^{B×S×H×D}`；无缩放、无归一化、无 mask；fp32 累加域（与源一致）。公式与 §0.1 语义逐项对应。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

与 §1.6.0 调研选定算法一致（**单遍单块全 D 驻留 + 逐行独立一维 persistent 分核 + 三维扁平声明归约**）。
每个逻辑任务处理 `(b, h, s0:s0+blk_s)`：

1. **装载 `o`**：`T.copy(o[bz, s0:s0+real_s, bx, 0:D], stage[0:real_s, 0:1, 0:D])`（三维跨步区域，行 stride = `H·D`）；
2. **上抛 fp32**：`T.vcast(stage, acc32, round_mode="rint")`（`stage` 承载 `o`）；
3. **装载 `do`**：同形 `T.copy` **复用 `stage`**（覆盖上一步的 `o`，把声明量压到 10 B/元素，§0.6 R4/R7）；
4. **上抛 fp32**：`T.vcast(stage, aux32, round_mode="rint")`；
5. **乘积（O1 原位）**：`T.vmul(acc32, aux32, acc32)` → `acc32 = o·do`（fp32，逐位等于源 FMA 的乘积，§1.6.1 P1）；
6. **行归约**：`T.reduce_sum(acc32, delta_ub, dim=2)`，`(blk_s,1,D) → (blk_s,1,1)`（三维扁平形态，§0.6 R3）；
7. **写出**：`T.copy(delta_ub[0:real_s, 0, 0], delta[bz, bx, s0:s0+real_s])`（S 连续段，尾块裁剪）。

标量/循环盘点（逐项理由见 §1.6.2）：无逐元素标量循环；保留的标量仅为 `real_s`（动态边界，1 次/任务）与 block 级索引算术。

**与源算法的差异及来源**：一维 persistent 分核（§0.6 R1）；tile 形态 UB 预算驱动（R2）；三维扁平声明归约（R3）；
fp32 中转链（R4）；显式尾块（R5）；v-prefix 向量链（R6）；乘积原位复用（R7）；跨步区域拷贝与 `out_idx` 保留（R8/§0.5）。

### 1.5 数据流图

```
GM[o (B,S,H,D) dtype]  --T.copy(三维跨步切片, 行 stride=H·D)--> UB[stage (blk_s,1,D) dtype]
UB[stage]              --T.vcast(rint, 逐位精确)-------------> UB[acc32 (blk_s,1,D) fp32]
GM[do]                 --T.copy(同形, 复用 stage)-------------> UB[stage]（覆盖 o）
UB[stage]              --T.vcast(rint)-----------------------> UB[aux32 (blk_s,1,D) fp32]
UB[acc32] × UB[aux32]  --T.vmul(原位 dst=acc32, O1)----------> UB[acc32 = o·do]
UB[acc32]              --T.reduce_sum(dim=2, 硬件行归约)------> UB[delta_ub (blk_s,1,1) fp32]
UB[delta_ub][0:real_s] --T.copy(1-D 连续段)-------------------> GM[delta (B,H,S) fp32]

数据层级：仅 GM ↔ UB（无 L1 / L0 / 跨引擎传输；容量项 = 0）；无中间结果落回 GM；无核间协作。
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：同一数学语义（**逐行点积 / row-wise dot product over a contiguous reduce axis with a transposed output
> layout**）的算法族——源算法（三维 grid + 二维 tile 最内轴归约 + fragment FMA）只是**基线候选之一**。
> **调研深度**：**完整调研**（规约类，按 `references/algorithm-research.md` §2 分级：R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。
> **信息源**：`algorithm-candidates.md` 的 `ALG-attention` 命中行（必查清单）与 `ALG-layernorm`/`ALG-rmsnorm` 的 row-reduction 条目（同族规约）；
> 本仓 `examples/` 同类实现（`examples/deepseek_v32/sparse_mla_bwd.py`、`examples/deepseek_v4/example_sparse_attn_bwd_kernel.py`、
> `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`、`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`）；
> pattern-library（`layout.md` 的 PL-1.2 / PL-1.23 / PL-1.5，`traps-compiler.md` 的 TRAP-inner-axis-reduce-scalarize /
> TRAP-transpose-ub-gm-roundtrip / TRAP-C10-strided-gather / TRAP-copy-layout-propagation-dma-amplification、
> `traps-runtime.md`、`attention.md` 的 PL-1.9 / PL-1.12、`elementwise.md` 的 PL-1.6 / PL-1.19 / PL-1.20、
> `constants.md` 的 CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-msprof-task-floor-910B3 /
> CONST-capacity-910B2C / CONST-aicore-910B2C / CONST-reduce-ub-per-element / CONST-dual-output-whole-plane-reduce-ub）；
> 源码 §0.3 / §0.4；结构判据（R2）。
> **未使用互联网检索**：本地源（参考表命中行 + 本仓两个同语义 npuir 实现 + pattern-library 归约/布局实测条目 + 源码）
> 已覆盖四问所需的全部候选，无未命中缺口；本算子为单步归约、候选空间封闭，无联网必要。

> **证据路径（负向断言与弃选论证所指向的条目，设计期已逐条打开核对）**：
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（CONST-mte2-degradation、CONST-vector-launch-overhead、CONST-msprof-task-floor-910B3、CONST-capacity-910B2C、CONST-aicore-910B2C、CONST-reduce-ub-per-element、CONST-dual-output-whole-plane-reduce-ub）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md`（PL-1.1-transpose-chain、PL-1.2-caxis-accum、PL-1.5-quickref、PL-1.21-vector-persistent-split、PL-1.23-flat-spatial-axis-reduce）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`（TRAP-inner-axis-reduce-scalarize、TRAP-transpose-ub-gm-roundtrip、TRAP-C10-strided-gather、TRAP-copy-layout-propagation-dma-amplification、TRAP-UB-multibuffer-inflation、TRAP-T-copy-region-semantics、TRAP-UB-dst-align、TRAP-C12-copy-dtype-cast、TRAP-npuir-reshape-dst-buffer-shadow）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/attention.md`（PL-1.9-blockwidth、PL-1.12-task-pipeline-depth2）
> - `.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md`（PL-1.6-copy-floor、PL-1.10-loads-first-decoupling、PL-1.19-scatter-col-write-amplification、PL-1.20-nsplit-perrow-partial）
> - `.agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md`（BP_pipeline_overlap）、`.agents/evolution/capability-gaps.md`（CG-2026-0027 `T.Pipelined` 于单元循环无增益）
> - `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`、`.agents/skills/tilelang-op-design/references/algorithm-research.md`、`.agents/skills/tilelang-op-design/references/migration-analysis.md`
> - `.agents/skills/_shared/standards/core-split-strategy.md`、`.agents/skills/_shared/standards/hardware-cost-model.md`、`.agents/skills/_shared/standards/negative-claim-evidence.md`
> - `docs/Tilelang.language/规约操作/T.reduce_sum.md`、`docs/Tilelang.language/数学操作/T.vmul.md`、`docs/Tilelang.language/数据类型转换操作/T.vcast.md`、`docs/Tilelang.language/内存操作/T.copy.md`、`docs/Tilelang.language/内存操作/T.alloc_shared.md`、`docs/Tilelang.language/编译器提示操作/T.Kernel.md`、`docs/Tilelang.language/创建操作/T.transpose.md`

**R1 等价化简公式候选**：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | **基线**（GPU 源算法） | 三维 grid `(H, ⌈S/256⌉, B)`；`(256,256)` tile 的 `o`/`do` fragment + `(256,256)` fp32 `acc`；D 分块循环内 `acc += o·do`；末尾 `T.reduce_sum(acc, delta, 1)`（二维、归约最内轴）；S 连续写出 | — | — | ✅（基线） |
| 1 | **单遍全 D 驻留 + 三维扁平声明单次归约**（本设计主选） | `(blk_s, 1, D)` UB tile 整行驻留；`vcast`×2 + `vmul`（原位）+ `T.reduce_sum(dim=2)` 单次 | **数学恒等**（同一元素集合、同一 `D` 轴、同一 fp32 乘积；仅 tile 形状与声明秩不同，§0.6 R2/R3） | 无 GM 收益（本就单遍）；UB 声明量 10 B/元素（可控且可预算）；归约命中向量化 lowering；乘积缓冲 −4 B/元素 | ✅（主选） |
| 2 | D 轴分块流式累加（`clear=False` 跨分块累加） | tile 改为 `(blk_s, 1, d_c)`，`for kc in T.serial(D/d_c)` 内 `reduce_sum(..., clear=False)` | 容差内等价（分块部分和 + 结合序变化；§1.6.1 P2 机器验证 0 判否） | 驻留缓冲 O(D)→O(d_c)；**但 GM 扫描仍为 1 遍**（无重读）、`vcast`/`vmul`/`reduce` 的 pass 数按元素总量守恒（§1.6.0 R3 注），且单行 DMA burst 由 `D·eb` 缩到 `d_c·eb`（更碎） | ❌（R3/R4：域内严格不优，留作超驻留 fallback，§9.3） |
| 3 | 转置布局 `(D, blk_s)` + 按行 `reduce(dim=1)` | 归约轴变为中轴、最内轴 = `blk_s` 宽 | 数学恒等 | 理论上换取「归约轴非最内 + 最内轴宽」的 lowering 判据（`TRAP-inner-axis-reduce-scalarize` 的判据），但需 UB 内重排 | ❌（R4：repack 纯开销且高陷阱暴露——`TRAP-transpose-ub-gm-roundtrip`、`TRAP-copy-layout-propagation-dma-amplification`；候选 1 已用免转置路径命中同一判据） |
| 4 | 垂直扫描（lane → 独立输出行，`T.serial` 步进 D） | `for d in T.serial(D): acc[0:blk_s] += prod[.,d]` | 数学恒等（累加序变化） | 无需硬件行归约 op | ❌（R3/R4：向量 pass 数 ≈ `D`× 主选（D=128 时每 chunk 128 个 pass ≈ 32 µs/chunk），比单次 reduce 高两个量级） |
| 5 | Cube/GEMM 形式 `delta = diag(O·DOᵀ)` | 用 `T.gemm` 算 `(blk_s, blk_s)` 全积再取对角 | 数学恒等（对角元素） | 无 | ❌（R3/R4：FLOPs 放大 `blk_s` 倍（64×），且需 L1/L0A/L0B/L0C + NZ 布局 + 对角提取；纯 Vector 单元本就匹配该算子，用 Cube 是错配） |
| 6 | 两阶段跨核归约（D 分块分给不同核 + workspace 归并） | kernel1 写 `(B,H,S,nb)` 部分和，kernel2 归并 | 数学恒等（容差内） | 提高 D 维并行度 | ❌（R3：新增 `2·B·H·S·nb·4` 字节的 GM 往返 + 第二次 launch；本算子 `D ≤ 128` 本就整行驻留，D 维并行度无需求） |
| 7 | 输入 dtype 乘积（fp16 域 `o·do` 后再上抛） | `T.vmul` fp16 → fp32 累加 | **否**——fp16 乘积需 22 位尾数 > 11 位，舍入不可忽略；饱和输入下 fp16 乘积**溢出为 inf** | 省 2 个 `vcast` + 一半 fp32 缓冲 | ❌（数值否决，§1.6.1 R1 机器证据：19.7% 元素超容差、最大绝对偏差 inf；且 bf16 无向量算术，`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 的 bf16 列为 ×） |

> 注：本算子无超越函数、无除法、无归一化、无公共子表达式（每个输出元素是独立的一次点积）——R1 的「恒等变形空间」
> 在**数学式子层面**是空的（无化简对象）；R1 的候选全部来自 **结构重排 / 数据布局 / 计算单元归属** 三个方向（上表 2 至 7），
> 这是本算子族（纯流式逐行点积）的固有形态：**唯一的优化面是访存与归约 lowering 的形态**，而非代数化简。

**R2 在线算法**：**有**在线（流式）变体——分块 running 累加（`reduce_sum(..., clear=False)` 或 tile 内累加）；
且**源算法本身已使用它**（`for k in range(ceildiv(dim, blk))` 内的 `acc += o·do` 即是分块 running 累加，`ALG-attention`
行亦明示「分块累加」为该族可移植手段）。**但本域不采纳**，依据是结构判据 + 容量判据（非无依据的否定）：

- **在线变体的收益口径在本算子失效**：其口径是「扫描遍数（两遍 → 单遍）/ 中间缓冲 `O(N)` → `O(tile)`」。
  而本算子的规约轴元素**只被消费一次、无任何重读**——基线实现已经是**单遍扫描**（`R3` 表：所有候选 scan passes = 1），
  在线化**不产生任何 GM 收益**；中间缓冲方面，本域 `D ≤ 128` 远小于整行驻留预算（§4.5），故 O(D) 缓冲本就不是约束。
- **无重缩放需求**：本算子无 `max`/`sum` 归一化（对比 online softmax 需要 running max + rescale），
  「在线」退化为一维 running sum，**无任何数值稳定化动机**。
- **结构判据**：把分块累加真的搬进 kernel 只会把单行 DMA burst 从 `D·eb` 缩到 `d_c·eb`
  （D=64 时 128B → 更碎），pass 数按元素总量守恒 ⇒ 域内**严格不优于**主选（候选 2 的否决理由）。
- **域边界**：本 fallback 真正有意义的域是 `D > 整行驻留预算`（本设计预算下 `D > 4096`，远出 manifest 域
  `D ∈ {64,128}`），届时按候选 2 处理（§9.3 登记）。

**R3 复杂度对比**（四口径；口径定义：**FLOPs** = 乘 1 次/元素 + 加 ≈1 次/元素（fp32，无超越函数折算）；
**访存 Bytes = GM 流量**（输入读 + 输出写 + 中间 GM 往返，不含 UB 内流量）；**扫描遍数** = 同一输入元素的 GM 重复读取次数；
**中间缓冲峰值** = UB/片上驻留；附可并行度与跨核代价。基线候选在表内；数字按 manifest 四档给出区间）：

| 算法候选 | FLOPs（`≈2·B·S·H·D`） | GM 访存 Bytes（`2·B·S·H·D·eb + 4·B·H·S`） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|------------------------------------------|---------|-------------|--------------------|
| 0 基线（GPU 源：三维 grid + 256² tile） | 1.34e8（full-fp16） | **270.5 MB**（full-fp16） | **1** | CUDA 侧 2×(256·256·2B)+256·256·4B = **512 KB**（fragment/SMEM，CUDA 无 UB 上限概念） | 逻辑核 `B·H·⌈S/256⌉`（full-fp16 = 2048）→ CUDA 侧无物理核数约束问题；无跨核代价 |
| **1 主选：全 D 驻留 + 三维扁平单次归约** | 同上 | **270.5 MB（= 基线，全候选最小）** | **1** | 声明 `blk_s·D·10 B`（= 4096·10 = 40 KB）+ 归约链 lowering 单价（≤32 B/元素，§4.5） | 逻辑核 `B·H·⌈S/blk_s⌉`（full-fp16 = 16384）→ persistent 40 核（§5.5）；无跨核代价 |
| 2 D 分块流式累加 | 同上 | 270.5 MB（**无重读，与主选相同**） | 1 | `blk_s·d_c·10 B`（更小，D 无上限） | 同上；单行 burst 更碎（`d_c·eb`）；pass 数守恒 ⇒ 域内不优 |
| 3 转置布局 `(D, blk_s)` | 同上 | 270.5 MB（GM 流量不变） | 1 | 主选 + 转置中转缓冲（可能触发 UB→GM→UB 往返，`TRAP-transpose-ub-gm-roundtrip`） | 同上；repack 为纯开销 |
| 4 垂直扫描 | 同上 | 270.5 MB | 1 | 主选（无乘积 tile 之外的缓冲） | 同上；**向量 op 数 ≈ `D`×主选**（D=128 → 每 chunk 128 pass） |
| 5 Cube/GEMM diag | **`≈2·B·S·H·D·blk_s`（放大 blk_s 倍）** | 270.5 MB（+ L1/L0 内部流量，跨引擎） | 1 | L0A/L0B/L0C 驻留（L0C 128KB 上限，`CONST-capacity-910B2C`）+ 对角提取 | Cube 与 Vector 协同（CV 融合）代价；FLOPs 错配 |
| 6 两阶段跨核归约 | 同上 | **270.5 MB + `2·B·H·S·nb·4`（部分和 GM 往返）** | **2**（部分和写 + 读） | 更小（D 可任意分块） | 需第二次 launch 或核间 flag 同步（`CONST-flag-id-budget`）；D 维并行度无需求 |
| 7 输入 dtype 乘积 | 同上（但 fp16 域） | 270.5 MB | 1 | 5 B/元素级（更省） | 同上（数值否决见 §1.6.1 R1） |

> **口径注**：① 除候选 6（两阶段，多一份部分和 GM 往返、扫描 2 遍）外，候选 0–5/7 的 GM 流量与扫描遍数**相同**
> （本算子是纯单遍流式归约，这些候选都不改变 I/O 下界）——故本算子的排序**主要由 R4（归约 lowering 形态、
> 计算单元匹配、UB 容量、burst 粒度）决定**，不是纸面并列：候选 4/5/6 有量级级或结构性劣势，候选 2/3 是「无收益 + 有代价」。
> ② 每元素级表达式中的 `eb` = 输入元素字节（2），常数 4 = fp32 `delta` 的字节数。
> ③ pass 数守恒的算术：一次 tile 级向量 pass 的成本按 pass 计（`CONST-vector-launch-overhead`），
> 而 pass 数 ≈ `总元素 / 单 tile 元素` × 每 tile pass 数；分块（候选 2）同时缩小 tile 元素数与 tile 内 pass 数，
> 比值为常数 ⇒ **分块不减少 pass 总数**（只改变 burst 粒度与驻留量）。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构；
负向淘汰均附证据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| 0 基线（GPU 源） | CUDA fragment/warp（NPU 无对应） | CUDA 口径不适用；NPU 需按 `(blk_s,1,D)` 重算 | 无 256 对齐要求（源无 pad） | 源 `ceildiv` 超发 + 隐式裁剪 | fragment 归约 | grid 超发 | ❌ 直译不可用（§0.6 R1/R2/R3 重设计） |
| **1 主选：全 D 驻留 + 三维扁平归约** | ✅ Vector：`T.vcast`/`T.vmul`/`T.reduce_sum`（`docs/Tilelang.language/数学操作/T.vmul.md`、`docs/Tilelang.language/规约操作/T.reduce_sum.md`） | ✅ 声明 40 KB（4096 元素 × 10 B）+ 归约链单价上限 32 B/元素（≤128 KB）⇒ 合计 ≤168 KB ≤ 192 KB（§4.5 双口径） | ✅ 尾轴 `D ∈ {64,128}`：fp16 向量宽 ×8 整除（尾 lane 浪费 0%）；`blk_s ∈ {32,64}` 为 8 的整数倍 | ✅ `B/S/H/D` 全为工厂期常量；`T.serial` 边界 `num_local_tasks` 为 host 折叠常量 | ✅ 单任务内为依赖链（cast→mul→reduce），跨任务重叠交 auto multi-buffer；不用 `T.Pipelined`（`CG-2026-0027`：单元循环 `T.Pipelined` 无增益） | ✅ 无跨核归并（行独立）⇒ 无 flag / workspace 成本 | ✅ **主选** |
| 2 D 分块流式 | ✅ Vector | ✅ 更松 | ✅ | ✅ | ✅ | 同上 | ❌ 域内无收益 + burst 更碎（R3/R2） |
| 3 转置布局 | ✅ Vector | ⚠ 中转缓冲 | ⚠ `(D, blk_s)` 两边均需 ≥ 向量宽且为宽整数倍（PL-1.1 收窄后判据） | ✅ | ⚠ 转置与归约链串行化 | 同上 | ❌ repack 纯开销（`TRAP-transpose-ub-gm-roundtrip` 实测 UB 内转置可降级为 UB→GM→UB 往返 5.6→138.2 µs，代价强依赖 shape/dtype） |
| 4 垂直扫描 | ⚠ 归约维度按 stride 访问；`TRAP-C10-strided-gather`（跨步 gather 向量指令缺失） | ✅ | ⚠ 归约轴跨步访问 | ✅ | ⚠ pass 数 ×D | 同上 | ❌ 量级劣势（R3） |
| 5 Cube/GEMM diag | ❌ 纯 Vector 算子用 Cube（FLOPs 放大 `blk_s` 倍） | ❌ L0C 128 KB 约束 + 对角提取 | ✅ 分形 M/N≥16、K≥32 可满足（D=128） | ✅ | ⚠ CV 融合与跨引擎同步 | ❌ 需 CV 协同 | ❌ 错配（R3） |
| 6 两阶段跨核 | ✅ | ✅ | ✅ | ⚠ 需要 flag/workspace | ⚠ 两 phase | ❌ 跨核归并 + 第二 kernel | ❌ 域内无需求（R3） |
| 7 输入 dtype 乘积 | ⚠ fp16 `vmul` 存在但精度不可达；bf16 **无向量算术** | ✅ | ✅ | ✅ | ✅ | 同上 | ❌ 数值否决 + bf16 不可用 |

**调研结论**：选定 **候选 1「单遍全 D 驻留 + 逐行独立一维 persistent 分核 + 三维扁平声明单次归约 + fp32 中转链 + 乘积原位复用」**
（= 候选 0 的数学内核 + 候选 1 的结构重排 + §1.6.1 O1）。关键依据：**R3** 表中该候选的 GM 流量与扫描遍数均为全候选最小
（270.5 MB / 1 遍；候选 6 更多、候选 5 的 FLOPs 放大 `blk_s` 倍），**R4** 全项亲和（Vector 单元、声明 40 KB + 归约链 ≤131 KB
≤ 192 KB、尾轴 8 整除、静态边界、无跨核结构），且 R1 中候选 4/5/6 的优势面（更多并行度/更少 UB）在本域**无需求**、
候选 2/3 存在「无收益 + 有代价」的结构劣势、候选 7 被机器证据数值否决。
**与基线的结构差异一句话**：数学内核（逐行点积 + fp32 累加 + 单遍）完全不变，改的是**并行结构**（三维 grid → 一维 persistent）、
**tile 形态**（256² fragment → UB 预算驱动的 `(blk_s,1,D)` 整行驻留）、**归约声明形态**（二维最内轴 → 三维扁平 `dim=2`）
与**元素级实现**（thread 级 `T.Parallel` FMA → `T.vmul` 向量链）。
**调研范围**（供「无更优替代」类断言核对）：`algorithm-candidates.md` 的 `ALG-attention` 命中行
- `ALG-layernorm`/`ALG-rmsnorm` 的 row-reduction 条目；本仓三个同语义 npuir 实现（`examples/deepseek_v32/sparse_mla_bwd.py`、
`examples/deepseek_v4/example_sparse_attn_bwd_kernel.py`、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`）
- 两个同族已交付迁移（`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`、
`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`）；
pattern-library 六个条目文件；源码 §0.3/§0.4；R2 结构判据（单遍性 / 无重缩放）。
**源算法优化手段承接**见 §0.6 末尾核对表（8 项逐项落实，**无静默丢弃**）。

**设计期估算（D-2 roofline；协议 `_shared/standards/hardware-cost-model.md` §2；常数带版本戳）**：
估算下界 = max(流量项, 发射项, 容量项)，按 manifest 四档（§5.5 给出 `blk_s` / 逻辑核数 / 核内轮数）：

| workload（manifest） | 配置 | 流量项（GM 字节 ÷ 有效带宽） | 发射项（向量 pass 数 × 单价） | 容量项 | 估算下界 |
|----------------------|------|------------------------------|-------------------------------|--------|----------|
| `smoke-bwd-fp16`（1,1024,8,64） | D=64, blk_s=64, 128 逻辑核 → 40 核，4 轮 | 2.13 MB ÷ (33.9 GB/s × 40)〔`CONST-mte2-degradation`〕= **1.6 µs** | 4 轮 × 4 pass × 0.25–0.5 µs = **4–8 µs** | 0 | **≈4–8 µs**（并受 ≈2.4 µs 固定地板约束〔`CONST-msprof-task-floor-910B3`〕） |
| `smoke-bwd-bf16`（1,1024,8,64） | 同上 | 1.6 µs | 4–8 µs | 0 | **≈4–8 µs** |
| `full-bwd-fp16-large`（16,2048,16,128） | D=128, blk_s=32, 16384 逻辑核 → 40 核，**410 轮** | 270.5 MB ÷ (19.6 GB/s × 40 = 782 GB/s) = **346 µs** | 410 × 4 × 0.25–0.5 = **410–820 µs**（串行上界；同族实测校准 ×0.44 ≈ **180–360 µs**，见下保守性说明） | 0 | **≈346–820 µs（流量项为可信地板 346 µs；发射项按上界给出）** |
| `full-bwd-bf16`（4,4096,16,128） | D=128, blk_s=32, 8192 逻辑核 → 40 核，205 轮 | 135.3 MB ÷ (20.6 GB/s × 40 = 824 GB/s) = **164 µs** | 205 × 4 × 0.25–0.5 = **205–410 µs**（校准后 ≈90–180 µs） | 0 | **≈164–410 µs** |

> **估算下界（汇总）**：`max(流量项, 发射项, 容量项)`。代表 workload `full-bwd-fp16-large` 的**流量项 346 µs 为可信地板**
> （口径明确：268.4 MB 输入读 + 2.1 MB 输出写，单遍无中间 GM 往返），发射项 410–820 µs 为**串行上界模型**
> （`CONST-vector-launch-overhead` 的 ~0.5 µs/op 语境是小块 op；`CONST-msprof-task-floor-910B3` 在本设备同代口径下
> 实测逐元素 pass 单价 **0.23–0.3 µs/pass**，故按双档给出）。**容量项 = 0**（纯 Vector 单引擎、无 L0/L1、
> 无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only` 不适用）。
> **模型保守性说明（诚实标注）**：发射项按「pass 数 × 单价」线性外推，未计入 auto multi-buffer 对
> MTE2 窗口的重叠隐藏；同族实测校准锚点：argmin `(2048,4096)` fp16（1024 block、~10 op/block、48 核）实测 47 µs
> vs 串行模型 106 µs（比值 0.44，见 `examples/argmin/_argreduce_kernel/DESIGN.md` 的 §1.6.0 估算表）。
> 按该比值折算，本算子 full-fp16 档的发射项落入 **180–360 µs**，与流量项 346 µs 同量级 ⇒
> **本算子落在「流量与发射双敏感」区**，这直接决定 Stage 4 的第一杠杆（见下「参数敏感方向」）。
> **参数敏感方向（指导 Stage 4）**：发射项 ∝ 每核 pass 数 ∝ `总元素 ÷ 单 tile 元素 ÷ 40`，
> 故**唯一杠杆是放大单 tile 元素数（UB 预算）**：`blk_s` 由 32 提到 64（元素预算 4096 → 8192）可把 full-fp16 的
> 核内轮数 410 → 205、发射项减半。该调整只依赖未实测常数「归约链 lowering 的 UB 单价」（§4.5 双口径区间
> 8–32 B/元素），属**参数级** A/B（不触发结构级实验裁决三件套，同族先例见 `examples/group_norm/_group_norm_kernel/DESIGN.md` §1.6.3）。
> **常数适用性声明**：`CONST-mte2-degradation` / `CONST-vector-launch-overhead` 绑定 **910B2C + CANN 8.5.0**
> （本设备为 **910B3 + CANN 9.0.0**）⇒ 按「未实测假设：沿用同代 910B 曲线量级」引用，且 910B2C 的 48 vector core 口径
> 按 40 核做**线性外推**（属未实测假设）；`CONST-msprof-task-floor-910B3` 与 `CONST-aicore-910B2C` 的 910B3 数据点
> 为**本工具链同设备**实测（可信档）。核数（20 / 40）与 UB 容量（196608 B）**已本机实查复核**（§5.5），无假设成分。
> Stage 4 须按 `hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行。
#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 调研选定的算法（单遍全 D 驻留 + 逐行独立 persistent 分核 + 三维扁平声明归约 + fp32 中转链）。
> 逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）。
> **R1 已结论**：本算子在**代数式子层面无化简对象**（单次逐行点积：无除法 / 无超越函数 / 无归一化 / 无公共子表达式 /
> 无矩式可降代换——每个输出元素是独立的一次 `D` 项点积），故 §1.6.1 的采纳项只能来自**计算域与缓冲结构**的等价重排。
> 该结论的依据：`algorithm-candidates.md` 的 `ALG-attention` 命中行（其 R1 列只列「scale 融入 Q/K」与「softmax 分母倒数乘法化」，
> 二者均属 softmax/scale 环节，**本算子不含 softmax 与 scale**）+ §1.6.0 R1 的恒等变形空间枚举（8 个候选，无代数化简类）。

| # | 优化项 | 原式（源 / 基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|------------------|-----------|-----------|---------|
| **O1** | **乘积缓冲原位复用** | 源：乘积写入独立于操作数的 `acc` tile（`acc[i,j] += o_frag[i,j]*do_frag[i,j]`，`acc` 与 `o_frag`/`do_frag` 各占一块）；迁移直译形态需三块全尺寸缓冲（`stage` 2B + `acc32` 4B + `prod32` 4B = **14 B/元素**） | `prod = vmul(acc32, aux32)` 写回 `acc32` 自身（`T.vmul(acc32, aux32, acc32)`，**10 B/元素**） | 逐元素同一 fp32 乘法、同一操作数、同一目标位置；`docs/Tilelang.language/数学操作/T.vmul.md` §2.3 第 3 条明确「`dst` 可与 `src1` 或 `src2` 为同一 buffer（支持原地操作）」，且本式 `src2 = aux32 ≠ dst = acc32`，无读写别名冲突。**D-1 机器验证**：32 案例（含 ±inf/NaN / 次正规 / 饱和 / 全零 / 消去位形、`D ∈ {1,…,4096}`、fp16 与 bf16）**逐位相等**、最大 ulp 差 **0**、超容差元素 **0** | UB 声明量 **14 → 10 B/元素（−29%）**；在 4096 元素的单元预算下释放 16 KB，正是 §4.5 中把预算上探到 8192 元素档（发射项减半，§1.6.0 参数敏感方向）的前提 |

**设计所依赖的数值等价性主张（同样以机器验证固化，供 Stage 2 复核）**：

| # | 主张 | 内容 | 机器验证结论 |
|---|------|------|--------------|
| **P1** | fp16/bf16 上抛 + fp32 乘积**逐位精确** | `fp32(a)·fp32(b) ≡ a·b`（fp64 精确乘积）当 `a,b` 为 fp16/bf16：fp16 11+11 = 22 位 ≤ fp32 的 24 位；bf16 8+8 = 16 位 ≤ 24 位 | 32 案例（含角点）最大 ulp 差 **0**、超容差 **0** ⇒ 本设计的 `vcast`+`vmul` 链与源「fp32 FMA」的乘积**逐位相同**；这同时证明 §0.6 R4 的中转链**不引入**任何精度损失 |
| **P2** | 归约**结合序**变化在容差内 | 源序 = 按 256 分块累积乘积 tile 后做 tile 归约；本设计序 = 对同一乘积平面做一次 `dim=2` 硬件归约（实现上为分层/树形归约） | 32 案例、三种可能的分层归约形态（全 pairwise 树 / 8-lane 块树 / 64-lane 块树）逐一对照源序：判否案例 **0**（准则 (a) 在常规案例全通过；唯一超容差案例 = 人为构造的「全 `±65504` 饱和」角点，2 个元素，由准则 (b) 承接——该角点源序自身相对 fp64 的绝对偏差已达 4.096e+03，候选 6.144e+03，同量级）；常规案例最大 ulp 差 **1280**。**边界注（诚实标注）**：若硬件归约实现为**全串行**累加（非树形），最大 ulp 差升至 3.1e9、超容差 3 例——故本设计要求 `T.reduce_sum` 的 lowering 为分层归约（已验证形态），Stage 4 若发现精度异常（尤其 `±inf`/消去位形）应优先按 §9.2 用 `clear=False` 分块部分和固化结合序 |
| **R1** | **否决候选证据**：input-dtype（fp16）乘积 | 候选式 `fp32(fp16(o·do))` vs 基线 `fp32(o)·fp32(do)` | 16 个 fp16 案例、323 776 个元素中 **63 671 个超容差（19.7%）**，最大绝对偏差 **inf**（饱和输入下 fp16 乘积上溢为 inf）⇒ **数值否决，禁止采纳**（§1.6.0 R1 候选 7） |

**D-1 等价性机器验证结果表**（脚本 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/verify_equiv.py`，2026-10-16 设计期执行，
torch CPU 2.9.0+cpu、fp64 参照 + 角点值 + 32 案例；判定双准则：**(a)** 候选式 vs 基线式在目标 dtype 容差内
（`atol = rtol = 1e-4`，超容差元素数 = 0）；**(b)** 候选式对 fp64 精确数学的偏差不超过基线式偏差 + 一个容差预算）：

| 采纳 / 主张项 | 案例数 | 最大 ulp 差 | 超容差元素数（准则 a） | 准则 (b) 判否案例数 | 结论 |
|--------------|-------|------------|----------------------|-------------------|------|
| **O1 乘积缓冲原位复用** | 32 | **0** | **0** | 0 | **EQUIV_PASS**（逐位相等） |
| P1 上抛 + 乘积精确性 | 32 | **0** | **0** | 0 | **EQUIV_PASS** |
| P2 归约结合序（设计序 vs 源序） | 32 | 1280（常规案例） | 2（全在「全 ±65504 饱和」角点） | **0** | **EQUIV_PASS**（角点由准则 (b) 承接，见上表注） |
| R1 否决候选（fp16 乘积） | 16（fp16） | — | 63 671 / 323 776 元素 | — | **判否（未采纳）** |

形态覆盖：fp16 与 bf16 两 dtype × `D ∈ {1, 7, 63, 64, 128, 255, 256, 1024, 4096}` × 行数 16 至 256；
角点值覆盖 `±inf`、`NaN`、fp16 次正规（×1e-5）、fp16 饱和（±65504）、bf16 极值（±3e38）、全零、混合符号消去位形。
脚本已随算子目录留存，Stage 2 可直接重跑复核（`python verify_equiv.py`，无需 NPU）。

**优化结论**：采纳 **O1（乘积缓冲原位复用，逐位恒等，EQUIV_PASS）**；同时以机器验证固化 **P1/P2** 两项数值等价性主张
（中转链逐位精确、归约结合序在容差内），并以 **R1** 的机器证据否决「input-dtype 乘积」候选。
**优化后公式（§3.1 的唯一输入；`fp32` 为 fp32 域运算，`d` 为 head-dim 下标）**：

$$
\delta[b,h,s] \;=\; \mathrm{reduce}_{d}\Big(\mathrm{fp32}\big(o[b,s,h,d]\big)\;\odot\;\mathrm{fp32}\big(do[b,s,h,d]\big)\Big),
\qquad \text{(O1 adopted: product buffer reuses the left-operand carrier)}
$$

**否决项及依据**（负向断言举证，`negative-claim-evidence.md` §1；不满足条件者一律不采纳）：

| 否决候选 | 否决理由 | 证据 |
|----------|----------|------|
| C1 input-dtype（fp16）域求积 | 乘积需 22 位尾数、fp16 只有 11 位；饱和输入上溢 inf；bf16 更差且根本无向量算术 | **本任务 D-1 机器证据**（63 671/323 776 元素超容差、最大绝对偏差 inf）；`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 的 bf16 列为 × |
| C2 跨 dtype `T.copy` 当 cast（省 `stage` 缓冲与 1 个 pass） | ① 不减向量 pass（编译器仍插 `hivm::VCastOp`，只是从显式变隐式）；② 舍入模式不可控（RINT 不可定制），一旦需要 `floor`/`trunc` 语义即静默错值；③ 本仓既有纪律与本设计同族前作均要求显式 `vcast` + 同 dtype copy | `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条（Developer 自动插 `hivm::VCastOp`，RoundMode=RINT，不可通过 `T.copy` 定制）；`.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md` 的 TRAP-C12-copy-dtype-cast；同族先例 `examples/group_norm/_group_norm_kernel/DESIGN.md` §1.6.1 的 C3 行 |
| C3 bf16 直算（不经 fp32） | bf16 无 Vector 算术（`vmul`/`vcast`/`reduce_sum` 的 bf16 列均为 ×），bf16 路径**必须** fp32 中转 | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1、`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1、`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1 的 dtype 表 |
| C4 Cube/GEMM（`diag(O·DOᵀ)`）形式做「乘积+归约」 | FLOPs 放大 `blk_s` 倍（full-fp16 档 64×），并引入 L1/L0A/L0B/L0C + NZ 布局 + 对角提取 + CV 协同；纯 Vector 单元本就匹配本算子 | §1.6.0 R3/R4 候选 5 行；`CONST-capacity-910B2C`（L0C 128KB）；`docs/Tilelang.language/线性代数操作/` 的 `T.gemm` 分形约束 |
| C5 归约轴转置（换取「归约轴非最内」的 lowering 判据） | repack 纯开销：核内二轴转置在本工具链可被降级为 UB→GM→UB 往返；且转置同时把 GM→UB 的 DMA 退化为 16B/元素分散读 | `TRAP-transpose-ub-gm-roundtrip`（5.6 → 138.2 µs）、`TRAP-copy-layout-propagation-dma-amplification`（36.75 → 336 KB/block）；候选 1 已用免转置路径命中同一判据 |
| C6 D 轴分块以降低驻留 | 无 GM 收益（本算子单遍、无重读），pass 总数守恒，单行 DMA burst 变碎 | §1.6.0 R2/R3 候选 2 行（含 pass 数守恒算术） |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点范围：本设计实现方案（§3.3 伪代码）中的**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
API 佐证均指向 `docs/Tilelang.language/` 下的具体文档（已逐份打开核对 dtype 表与 shape 规则）。

| # | 计算点 | 源实现形态（GPU） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|-------------------|--------------------------|---------|--------------|
| 1 | `o` / `do` 装载 | `T.copy(o[bz, s-slice, bx, d-slice], o_frag)` | `T.copy(o[bz, s0:s0+real_s, bx, 0:dim], stage[0:real_s, 0:1, 0:dim])`（双显式切片，**双显式**规避 `TRAP-T-copy-region-semantics`；dst 零起点规避 `TRAP-UB-dst-align`） | ✅（形态替换） | —（MTE 搬运本身非标量计算） |
| 2 | fp16/bf16 → fp32 上抛 | `acc[i,j] += o_frag[i,j]*do_frag[i,j]` 内隐式提升（CUDA FMA） | `T.vcast(stage, acc32, round_mode="rint")` ×2（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1：f16→f32 rint ✓、bf16→f32 rint ✓；§2.2.2 shape 须一致） | ✅ | — |
| 3 | **逐元素乘积 + 累加**（B1） | `for i, j in T.Parallel(blk, blk): acc[i,j] += o_frag[i,j]*do_frag[i,j]`（thread 级并行 FMA） | `T.vmul(acc32, aux32, acc32)`（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 同 shape `[M,N]*[M,N]`、§2.3 第 3 条原位 ✓、§2.3 第 1 条操作数在 UB ✓） | ✅ | — |
| 4 | **沿 D 轴的行归约**（B2） | `T.reduce_sum(acc, delta_frag, 1)`（二维 tile、归约最内轴；CUDA = warp shuffle 树） | `T.reduce_sum(acc32, dlta, dim=2)`，tile 声明为**三维** `(blk_s,1,D)`（`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.2：src/dst 同 rank、dst 在归约维尺寸为 1；fp32 ✓、bf16 ×）。**该形态重排是必需的而非可选的**：二维「归约轴 = 最内轴」形态在本工具链**被标量化**（`TRAP-inner-axis-reduce-scalarize` 实测单 op 使 kernel 5.6 → 155.1 µs；`CONST-dual-output-whole-plane-reduce-ub` 的「归约轴形态」段实测二维 `(C,plane)` 的 `reduce(dim=1)` 标量化 `scalar = 98.1 µs`）；三维扁平形态是已验证快路径（`PL-1.23-flat-spatial-axis-reduce` 166.26 → 16.68 µs） | ✅（声明形态重排，见 §0.6 R3） | — |
| 5 | `delta` 写出 | `T.copy(delta_frag, delta[bz, bx, s-slice])` | `T.copy(dlta[0:real_s, 0, 0], delta[bz, bx, s0:s0+real_s])`（1-D 连续段；本仓同形态 npuir 先例 `examples/deepseek_v32/sparse_mla_bwd.py` 的 `T.copy(delta[:, 0], Delta[bz, s-slice, bx])`） | ✅（形态替换） | — |
| 6 | D 轴分块循环 | `for k in range(T.ceildiv(dim, blk))`（串行 D 循环） | **域内消除**：`D ≤ 4096`（manifest 域 `D ∈ {64,128}`）整行驻留 ⇒ 单次归约，无循环 | ✅（消除） | —（仅当 `D >` 整行驻留预算时保留为 §9.3 fallback 的 `T.serial` 分块循环，理由同 §1.6.0 R2） |
| 7 | 尾块行数 `real_s` | 源无行谓词（靠 `ceildiv` 超发 + 隐式裁剪） | `real_s = T.min(blk_s, seq_len - s0)`（标量，1 次/任务，`docs/Tilelang.language/比较操作/T.min.md` 标量语义 ✓） | ❌ 保留 | **依赖动态边界的边界处理**：`seq_len % blk_s ≠ 0` 时尾任务的有效行数；每任务 1 次标量 `min` + 两次切片界，深流水档被 MTE2 窗口隐藏（`CONST-copy-floor-method`：`mte2_ratio ≥ 0.95` 时标量/分支开销已被隐藏）；静态化需为尾块复制整份 kernel 体，复杂度不值（同族前作同款判定） |
| 8 | 任务映射索引（`uid → (bz,bx,by)` 的整除/取模与 grid-stride） | 源为三维 grid（无核内映射） | `for i in T.serial(num_local_tasks)` + `if uid < num_logical` + 整除/取模分解 | ❌ 保留 | **block 级索引 / 任务映射计算**：每任务数个标量索引（`uid`/`s0`），非逐元素热点；该 `serial` 循环是 §0.6 R1 persistent 分核的载体，无逐元素等价的向量 API；同族前作同款结构（`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`、`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`） |
| 9 | `blk_s` / `num_logical` / `num_kernels` / `num_local_tasks` / `num_d_chunks` | host Python（源为 `blk=256` 常量） | host 层编译期常量（trace 期折叠为 `IntImm`，`PL-1.5-quickref` 的 host 编译期常量折叠） | ❌（非 kernel 内） | **host 侧 shape 元数据计算**：不在 kernel 内，无向量化问题 |
| 10 | 标量广播点（bias / per-row scale / per-row divisor） | 源**不存在**（无仿射、无归一化） | 不适用 | —（无此计算点） | — |

**向量化结论**：逐元素与归约计算点（上抛 ×2、乘积、行归约、装载/写出）**已全部向量化**，替代 API 均在
`docs/Tilelang.language/` 有 dtype 与 shape 佐证（上表）；**源的 `T.Parallel` 逐元素 FMA 循环（B1）被 `T.vmul` 全 buffer 替代**，
**源的二维最内轴 `reduce_sum`（B2）被三维扁平声明的 `dim=2` 归约替代**。保留的 3 类标量/循环为：
**动态边界 `T.min`（#7）**、**block 级任务映射（#8）**、**host 侧元数据（#9）**——逐项理由如上
（均为「每任务 1 次」/「block 级」/「kernel 外」，无「实现简单」类理由）。§6 循环结构与本表一致：
kernel 内**不存在**逐元素标量循环（源 S5 的 `T.Parallel(blk,blk)` 已消除），也不存在 D 维串行循环（域内整行驻留）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

规约类算子的必选候选对：**水平归约（lane 映射到归约轴 D，用向量 reduce）vs 垂直扫描（lane 映射到独立输出行，串行步进 D）**——
两者共用同一条张量轴但 lane 语义不同，故分别枚举；另外枚举「装载粒度/布局重排」方向的候选（把 `H` 折进单元 vs 逐 head 单元、
整行驻留 vs D 分块）。I/O layout（BSHD 输入 / `(B,H,S)` 输出）是**契约**，不在候选之列。

**轴质量评分**（主选：向量化轴 = 归约轴 `D`（尾轴连续）；昇腾 Vector 宽度 fp16/bf16 ×8、fp32 ×4）：

| 评分项 | 判定 |
|--------|------|
| 对齐 / 整除性 | manifest `D ∈ {64, 128}`：fp16/bf16 下 `64 = 8×8` ✓、`128 = 8×16` ✓；工厂对任意 `D ≥ 1` 成立（`reduce dim=2` 按整段 `D` 归约，非整除只影响尾 lane 利用率） |
| 尾 lane 浪费率 | manifest 两档 **0%**；`D = 7` 时 (8−7)/7 = 14%（仅防御性微型用例）；`D = 63` 时 1.6% |
| 累加链形态 | 归约维 = 最内连续轴、stride = 1；无跨步系数、无广播维（无行广播 v-op，因无 scale/bias） |
| repack 代价 | **无**（I/O 原生 BSHD 序 = 核内布局；声明秩由 2 升 3 只是加了一条恒为 1 的中间轴，不移动任何元素） |
| UB 容量影响 | 单行驻留 ⇒ 单元元素数 = `blk_s·D ≤ 4096`（**基数是 D 而非 `blk_s`**：`D` 越大 `blk_s` 越小，§4.5） |
| 装载粒度 | 每行 burst = `D·eb`（128 B @D=64 / 256 B @D=128），行间 stride = `H·D·eb`；与 attention 族主 kernel 的装载形态一致（§3.5.3） |

**候选矩阵**（弃选项均给量化理由与证据；`blk_s` 按 §4.5 的 4096 元素预算取值）：

| # | 布局方案 | 向量化轴 / lane 语义 | repack / 结构路径 | 预估收益 / 代价 | 采纳？ |
|---|---------|---------------------|------------------|-----------------|--------|
| 1 | **逐 `(b,h)` 单元的 I/O 原生装载 + `(blk_s,1,D)` 三维声明 + 水平归约** | lane → 归约轴 D（`reduce dim=2`） | 无 | 读 burst `D·eb`（128/256 B）× `blk_s` 行；写 1-D 连续段 `blk_s·4 B`；GM 流量 = 下界；声明 10 B/元素 | ✅ **主选** |
| 2 | 垂直扫描（lane → 输出行 `blk_s`，`T.serial` 步进 D） | lane → 输出轴 | 无 | 向量 pass 数 ≈ `D`× 主选（D=128 时 ≥128 pass/chunk）；无 T.reduce 但换不来任何流量收益 | ❌（量级劣势；R3 候选 4） |
| 3 | 按 `(b, s-block)` 单元装载**全 `H`**（连续大 burst）+ `reduce dim=2` + H 次逐 head 写出 | lane → 归约轴 D | 无 repack，但写出改为 `H` 次跨步 UB 切片读 | 读 burst 变 **连续** `blk_s·H·D·eb`（8 KB 级）；代价：单元元素数 = `blk_s·H·D` ⇒ `blk_s = 4096/(H·D)` 退化（D=64/H=8 → `blk_s=8`，写出 32 B/head；D=128/H=16 → `blk_s=2`，写出 8 B/head，**退化到不可用**）；每单元多 `H` 个跨步写出 pass（`TRAP-C10-strided-gather` 形态） | ❌ **主选弃选**（D=128 档结构性退化）；保留为 §1.6.4 备选（仅 `H·D ≤ 512` 的 shape 可行） |
| 4 | 归约轴转置（`(D, blk_s)`）+ 按行 `reduce(dim=1)` | lane → `blk_s` | 核内二轴转置链 | 命中「归约轴非最内」判据，但 repack 为纯开销且高陷阱暴露 | ❌（`TRAP-transpose-ub-gm-roundtrip` / `TRAP-copy-layout-propagation-dma-amplification` / `PL-1.1` 的收窄判据） |
| 5 | D 分块流式（`(blk_s,1,d_c)` + `clear=False`） | lane → `d_c` | 无 | 驻留更小但 burst 变碎、pass 守恒、无流量收益 | ❌（域内，§9.3 fallback） |
| 6 | 二维声明 `(blk_s, D)` + `reduce(dim=1)` | lane → 归约轴 D | 无 | 与候选 1 元素布局完全相同，但归约最内轴形态**被标量化** | ❌（**证据**：`CONST-dual-output-whole-plane-reduce-ub` 的「归约轴形态」段实测二维 `(C,plane)` 的 `reduce(dim=1)` 标量化 `scalar = 98.1 µs`、Task 237 µs、`UB_to_GM` 96 KB/blk 转置往返；`TRAP-inner-axis-reduce-scalarize` 单 op 5.6 → 155.1 µs） |
| 7 | Cube/CV 融合（`diag(O·DOᵀ)`） | Cube MAC | 需 L1/L0A/L0B/L0C + NZ | FLOPs ×`blk_s`；CV 协同 | ❌（§1.6.0 R3 候选 5 / §1.6.1 C4） |

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：源 kernel 把 `T.Parallel(blk, blk)` 映射到 `(seq 行, dim)` 二维线程网格，
归约沿 `dim` 轴做 warp/CTA 树归约——即 GPU **也**把最内连续轴 `D` 当作归约轴，但那是 CUDA coalescing + warp shuffle 的产物，
且被 `blk=256` 的方形 tile 绑定（§0.4 #1/#5）。本节依据 NPU 的向量宽度、`reduce` 的 lowering 判据（三维扁平形态）、
UB 容量（`blk_s·D ≤ 4096`）与跨步 row-burst 装载形态**独立评估**后结论一致（归约轴 = `D`）**但理由不同**
（NPU 侧：尾轴连续 stride = 1 + 三维扁平声明命中向量化归约 + 单遍全 D 驻留 + 行独立无跨核归并）；
**且 lane 语义不同**——源在 256×256 线程块内并行、本设计在 40 个核内按行块切分（§5.5）。

**布局决策结论**：选定 **候选 1——核内布局 = I/O 原生 BSHD（逐 `(b,h)` 单元的跨步行块装载）+ UB tile 三维声明 `(blk_s,1,D)`
- 向量化轴 = 归约轴 `D`（尾轴）+ 水平归约（`reduce dim=2`）+ repack = 无**。
该结论同步落入 §3.3 伪代码、§4.3 缓冲表与 §6 循环结构（所有 v-op / reduce 的 buffer 形状 = `(blk_s,1,D)`，
归约 `dim=2`，向量维 = `D`）；弃选候选的量化理由见上表（#3 在代表 shape 下结构性退化、#4 repack 零收益、
#5 域内无收益、#6 有实测标量化证据、#7 FLOPs 错配）。

**多缓冲 / 单元元素预算的参数级 A/B（非结构级，故不触发结构级实验裁决三件套）**：

- **主选（默认实现，进 Stage 3）**：单元元素预算 `_UB_ELEMS_BUDGET = 4096`（保守档，§4.5 双口径取小），
  工厂默认 `raw = max(1, 4096 // D)`、`blk_s = raw`（`raw < 8`）或 `(raw // 8) * 8`（`raw ≥ 8`）
  → `D = 64` 时 `blk_s = 64`、`D = 128` 时 `blk_s = 32`、`D = 4096` 时 `blk_s = 1`（§4.5/§5.2/§5.5 表）。
- **参数级备选（Stage 4 可切换，不改结构）**：`pass_configs={"npuir.enable_auto_multi_buffer": False}` 关闭自动多缓冲
  （UB 口径从「声明 ×1.7」降到 ≈「声明 ×1.10–1.12」，`CONST-capacity-910B2C` 双口径）并把预算上探到
  `_UB_ELEMS_TUNED = 8192` → `blk_s` 翻倍、核内轮数减半（full-fp16 档 410 → 205，发射项减半，§1.6.0 参数敏感方向）；
  代价是失去编译器的自动装载/计算重叠（本算子落在流量/发射双敏感区，净收益需实测）。
  **可行条件**：该档要求归约链 lowering 单价 ≤ `196608 / 8192 = 24` B/元素（即落在 pattern-library 已测的 8–32 区间**下半段**），
  由 L0-10 的编译阶梯直接筛出（§4.5）。
- **裁决计划**：代表 shape = manifest 四档；指标 = msprof op `Task Duration`（每档 ≥20 次采样、去首 5 次预热）
  ＋该档的 UB 编译需求（`ub overflow` 报文反推，反解真实归约链单价）；**判定阈值**：备选在该档
  `Task Duration < 主选 × 0.95`（且 L0 精度门禁不回归）才翻转该档；**按 shape 分派**（per-D 表）允许不同档各自选择；
  裁决结果经附录补记回写本节，反解出的归约链 UB 单价同步作为 pattern-library 候选常数。
- **不启动设计期探针（D-3）**：该决策只影响**参数**（元素预算 / 编译开关），不改结构；结构级结论
  （水平归约 vs 垂直扫描、三维扁平 vs 二维、逐 head 单元 vs 全 H 单元）的判定裕度是**量级级**或**结构性退化**
  （候选 2 约 `D`× pass、候选 3 在 D=128 时 `blk_s=2`、候选 6 有实测 40× 级差距），不落在任何未实证常数的
  不确定区间内，故无需探针预算。

#### 1.6.4 备选方案（结构级，供 Stage 4 A/B；主选胜出则仅存档）

> 依据 Phase 2 第 3 项：备选 = §1.6.3 候选 3（全 `H` 连续装载单元），**仅对 `H·D ≤ 512` 的 shape 结构可行**
> （如 `smoke-bwd-*` 的 `H=8, D=64`）；对 `D=128, H=16` 档 `blk_s` 退化为 2，**不纳入 A/B 清单**（结构性退化）。
> 该备选与主选的取舍依赖一个**未实测常数**，故按实验裁决模式给出完整三件套。

| 维度 | 主选（候选 1） | 备选 B（候选 3） |
|------|---------------|-----------------|
| 逻辑任务 | `(b, h, s-block)` ⇒ `num_logical = B·H·⌈S/blk_s⌉` | `(b, s-block)`（全 `H` 一单元）⇒ `num_logical = B·⌈S/blk_s⌉` |
| `blk_s`（4096 元素预算） | `4096 // D`（D=64 → 64；D=128 → 32） | `4096 // (H·D)`（H=8,D=64 → 8；H=16,D=128 → 2 ❌退化） |
| UB buffer | `stage (blk_s,1,D)` dtype、`acc32`/`aux32 (blk_s,1,D)` fp32、`dlta (blk_s,1,1)` fp32 | `stage (blk_s,H,D)` dtype、`acc32`/`aux32 (blk_s,H,D)` fp32、`dlta (blk_s,H,1)` fp32 |
| 装载 | 逐 `(b,h)` 跨步行块：burst `D·eb` × `blk_s` 行（stride `H·D·eb`） | 连续整块 `T.copy(o[bz, s0:s0+real_s, 0:H, 0:D], stage[...])`：单 burst `blk_s·H·D·eb`（8 KB 级） |
| 归约 | `reduce_sum(dim=2)` → `(blk_s,1,1)` | `reduce_sum(dim=2)` → `(blk_s,H,1)` |
| 写出 | 1 次 1-D 连续段（`blk_s·4 B`） | `H` 次跨步切片：`for h in T.serial(H): T.copy(dlta[0:real_s, h, 0], delta[bz, h, s0:s0+real_s])`（各 `blk_s·4 B`，8–32 B/head） |
| dtype 路径 | fp32 中转（同主选） | fp32 中转（同主选） |
| 分核三要素重算（smoke-bwd-*） | `num_logical = 128` → 40 核 / 4 轮 | `num_logical = 1·⌈1024/8⌉ = 128` → 40 核 / 4 轮 |
| 优势 | 写出粒度为 256 B 级连续段；无跨步 UB 读；结构最简、有同语义 npuir 先例 | 装载为单条连续大 burst（DMA 描述符数大幅下降） |
| 代价 | 装载为跨步行 burst（128–256 B） | 写出退化为 `H` 次跨步 UB 切片读（`TRAP-C10-strided-gather` 形态）；每单元多 `H` 个 MTE3 pass；`blk_s` 小 ⇒ 任务粒度碎 |

**实验裁决计划**：代表 shape = `smoke-bwd-fp16`（1,1024,8,64）与 `smoke-bwd-bf16`（同形，备选 B 仅覆盖该档）；
指标 = msprof op `Task Duration`（每档 ≥20 次采样、去首 5 次预热）＋ `aiv_mte2_time`/`aiv_vec_time` 分解
（用于**反解未实测常数「跨步 row-burst 的 MTE2 有效带宽折损率」**：主选 `mte2` 时间 ÷ 该档数据量 = 折损后带宽，
与连续装载档对比即得折损率）；**判定阈值**：备选在该档 `Task Duration < 主选 × 0.95` 且 L0 精度不回归 ⇒ 翻转该档
（按 shape 分派，允许仅 smoke 档走备选）；裁决结果回写本节并作为常数候选（含工具链版本戳）提交 pattern-library。
**不启动设计期探针**：该常数（跨步 burst 折损率）可由 Stage 4 的一次 msprof 直接反解，且主选有同族大 shape
已验证装载形态（§3.5.3）作为先验，判定裕度不落在「必须设计期探针」的区间。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.vcast`/`T.vmul`/`T.reduce_sum`），无 matmul | 不涉及 Cube / L0A / L0B / L0C / L1 分形约束 |
| 归约 | `T.reduce_sum(dim=2)`（硬件行归约） | Developer / Expert 均可用，无需手动流水控制 |
| 内存层级 | 仅 GM ↔ UB（`T.alloc_shared` 在 Developer 映射 UB） | 无 L1 / 无跨引擎协作，无 `Scope("Cube")` |
| 同步 | 单 kernel 内顺序依赖链、行独立、无核间协作 | Developer 自动同步即可，无需 `sync_block_set`/`sync_block_wait`/`pipe_barrier` |
| 流水 | 单任务内无 K 维可重叠迭代（`CG-2026-0027`：单元循环 `T.Pipelined` 实测无增益） | 无可用的显式流水目标 |
| 先例 | 同语义 npuir 实现（`examples/deepseek_v32/sparse_mla_bwd.py`、`examples/deepseek_v4/example_sparse_attn_bwd_kernel.py`）与同族已交付迁移（`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`、`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`）均为 Developer 形态 | Developer 已验证可行，无阻塞证据 |
| 用户指定 | `op_requirements.programming_mode = developer` | 一致 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（Developer 映射 UB；v-op 操作数必须在 UB，见 `docs/Tilelang.language/数学操作/T.vmul.md` §2.3 第 1 条） |
| 计算方式 | v-prefix API（`vcast`/`vmul`）+ `T.reduce_sum(dim=2)` + 标量 `T.min` 尾块界 |
| 同步 | Developer 自动同步（无任何手动同步原语） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维；NPU 最多 2 维 block，见 `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1） |
| 输出分配 | `@tilelang.jit(out_idx=[2], target="npuir")`（保留源契约：`delta` 在 index 2） |
| 环境变量 | `ASCEND_MODE = "Developer"`（`TILELANG_ASCEND_MODE`，与同族实现一致） |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 的优化后公式（fp32 计算域，O1 已采纳）。步骤编号与 §3.3 伪代码一一对应：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `u = i·num_kernels + cid`；`(bz,bx,by) = unflatten(u)`；`s0 = by·blk_s`；`real_s = min(blk_s, S−s0)` | 任务映射与尾块界（标量，#8/#7） |
| 2 | `stage ⟵ o[bz, s0:s0+real_s, bx, 0:D]` | 装载 `o` 的跨步行块 |
| 3 | `acc32 = fp32(stage)` | `o` 上抛（逐位精确，P1） |
| 4 | `stage ⟵ do[bz, s0:s0+real_s, bx, 0:D]` | 装载 `do`（复用缓冲） |
| 5 | `aux32 = fp32(stage)` | `do` 上抛（P1） |
| 6 | `acc32 = acc32 ⊙ aux32`（**原位**） | 乘积（O1，逐位等于源 FMA 乘积） |
| 7 | `dlta[r,0,0] = Σ_{d<D} acc32[r,0,d]`，`r ∈ [0, blk_s)` | 沿 D 轴行归约（三维声明，`dim=2`） |
| 8 | `delta[bz,bx,s0:s0+real_s] ⟵ dlta[0:real_s,0,0]` | 写出 S 连续段（尾块裁剪） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数与形态 | 模式 | 依据 |
|------|----------|-------------|-----------|------|------|
| 工厂 | — | `@tilelang.jit(out_idx=[2], target="npuir")` | 工厂签名 `(batch, heads, seq_len, dim, dtype, blk_elems=4096)`；`out_idx=[2]` 保留源契约（`delta` 自动分配） | Developer | 源同款装饰器（§0.2）；npuir 形态见 `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` |
| 核数实查 | — | `NPUUtils.get().get_aicore_num() * 2` | 本机实查 20 → 40（纯 Vector 算子翻倍） | Developer | `.agents/skills/_shared/standards/core-split-strategy.md` §1 要素② |
| 任务映射 | `u = i·num_kernels + cid` | `for i in T.serial(num_local_tasks)` + `if u < num_logical` | 静态边界（host 编译期常量折叠） | Developer | `PL-1.5-quickref` 的 host 编译期常量折叠；同族前作同款 |
| 尾块 | `real_s = min(blk_s, S−s0)` | `T.min(blk_s, seq_len - s0)` | 标量（1 次/任务） | Developer | `docs/Tilelang.language/比较操作/T.min.md`（标量语义 ✓） |
| 装载 `o`/`do` | GM → UB | `T.copy(o[bz, s0:s0+real_s, bx, 0:dim], stage[0:real_s, 0:1, 0:dim])`（`do` 同形） | src/dst **双显式 slice**、dst 零起点、同一 dtype | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2 规则 1/2/4（尾块借用动态 extents）+ §2.4；规避 `TRAP-T-copy-region-semantics` / `TRAP-UB-dst-align` |
| 上抛 ×2 | fp16/bf16 → fp32 | `T.vcast(stage, acc32, round_mode="rint")`（`aux32` 同） | `(blk_s,1,D) → (blk_s,1,D)` 同 shape | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16→f32 rint ✓、bf16→f32 rint ✓）+ §2.2.2（shape 须一致）；三维 vcast 实测佐证：`PL-1.23-flat-spatial-axis-reduce` 的 `(ch,h_in,w_in)` 三维 `T.vcast` |
| 乘积（原位） | `acc32 ⊙ aux32` | `T.vmul(acc32, aux32, acc32)` | 同 shape `[M,N]×[M,N]`；`dst = src1`（原位合法） | Developer | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 同 shape + §2.3 第 1/3 条（操作数在 UB、原位 ✓）；采纳项 O1 |
| 行归约 | `Σ_d` | `T.reduce_sum(acc32, dlta, dim=2)` | `(blk_s,1,D) → (blk_s,1,1)`；`clear=True`（默认） | Developer | `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.2（同 rank、归约维尺寸 1）+ §2.2.1（fp32 ✓、bf16 ×） |
| 写出 | UB → GM | `T.copy(dlta[0:real_s, 0, 0], delta[bz, bx, s0:s0+real_s])` | 1-D 连续段；两侧 extents 由 `real_s` 对齐 | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 1（静态 1 维折叠）；本仓 npuir 同形态先例见 §3.5.3 |
| 编译配置 | — | 无 `pass_configs`、无 `compile_flags` | 源本函数亦无（§0.2）；npuir `tilelang.jit` 无 `compile_flags` 形参 | Developer | 预注入知识 `VP-2026-0158`（本函数天然免疫该差异） |

### 3.3 计算伪代码

> **接口保持**：工厂签名与源一致（`(batch, heads, seq_len, dim, dtype)` 位置参数全部保留，`dtype` 为字符串）；
> 仅新增**带默认值的可选** `blk_elems`（Stage 4 参数级 A/B 用；wrapper 按位置调用不受影响）。
> `main(o, do, delta)` 的参数名/顺序与源一致；`out_idx=[2]` 不变；**无 `threads`、无 `compile_flags`、无 `pass_configs`**。
> `blk_s` 由 `dim` 与元素预算在**工厂层**推出（host 编译期常量，trace 期折叠），不进入运行时。

```python
import tilelang
import tilelang.language as T
from tilelang.utils.npu_utils import NPUUtils

# ---- 设计常量（§4.5 / §1.6.3） ----
_UB_CAPACITY_BYTES = 196_608      # 192 KB per Vector core（本机实查 ub_cap == 196608）
_UB_ELEMS_BUDGET = 4_096          # 保守单元元素预算（§4.5 双口径取小；Stage 4 A/B 档 = 8192）
_VECTOR_WIDTH = 8                 # fp16/bf16 向量宽（尾轴对齐粒度）
_SUPPORTED_DTYPES = ("float16", "bfloat16")


@tilelang.jit(out_idx=[2], target="npuir")          # 源契约：delta 在 index 2
def _flashattn_bwd_preprocess_kernel(batch: int, heads: int, seq_len: int, dim: int,
                                     dtype: str, blk_elems: int = _UB_ELEMS_BUDGET):
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if min(batch, heads, seq_len, dim) < 1:
        raise ValueError("batch/heads/seq_len/dim must be positive")

    # ---- host 侧编译期常量（trace 期折叠为 IntImm）----
    raw = max(1, blk_elems // dim)                        # 单行驻留的元素预算
    blk_s = raw if raw < 8 else (raw // 8) * 8            # ≥8 时对齐到向量宽；<8 时保容量优先（§5.2/§5.4）
    nblk = (seq_len + blk_s - 1) // blk_s
    num_logical = batch * heads * nblk
    vector_cores = NPUUtils.get().get_aicore_num() * 2    # queried 20 -> 40（§5.5）
    num_kernels = min(num_logical, vector_cores)          # 纯 Vector：核数翻倍
    num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

    @T.prim_func
    def flash_bwd_prep(
        o: T.Tensor((batch, seq_len, heads, dim), dtype),      # type: ignore
        do: T.Tensor((batch, seq_len, heads, dim), dtype),     # d(out)，同形同 dtype
        delta: T.Tensor((batch, heads, seq_len), "float32"),   # out_idx=[2]
    ) -> None:
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            stage = T.alloc_shared((blk_s, 1, dim), dtype)       # o / do 装载中转（双角色）
            acc32 = T.alloc_shared((blk_s, 1, dim), "float32")   # o 的 fp32 载体 → 乘积（原位, O1）
            aux32 = T.alloc_shared((blk_s, 1, dim), "float32")   # do 的 fp32 载体
            dlta = T.alloc_shared((blk_s, 1, 1), "float32")      # 行归约结果

            for i in T.serial(num_local_tasks):
                uid = i * num_kernels + cid
                if uid < num_logical:
                    bz = uid // (heads * nblk)
                    rem = uid % (heads * nblk)
                    bx = rem // nblk
                    by = rem % nblk
                    s0 = by * blk_s
                    real_s = T.min(blk_s, seq_len - s0)

                    T.copy(o[bz, s0:s0 + real_s, bx, 0:dim], stage[0:real_s, 0:1, 0:dim])
                    T.vcast(stage, acc32, round_mode="rint")          # 逐位精确（P1）
                    T.copy(do[bz, s0:s0 + real_s, bx, 0:dim], stage[0:real_s, 0:1, 0:dim])
                    T.vcast(stage, aux32, round_mode="rint")          # 复用 stage
                    T.vmul(acc32, aux32, acc32)                       # 乘积原位（O1）
                    T.reduce_sum(acc32, dlta, dim=2)                  # (blk_s,1,D) -> (blk_s,1,1)
                    T.copy(dlta[0:real_s, 0, 0], delta[bz, bx, s0:s0 + real_s])

    return flash_bwd_prep
```

> 尾块说明：装载用 `real_s` 显式裁剪（垃圾行不入 UB 有效区）；v-op / `reduce` 作用于完整 `(blk_s,1,D)` buffer，
> 但 `dim=2` 归约**逐行独立**，垃圾行不污染有效行；写出只覆盖 `[0:real_s]` ⇒ 与源输出元素集合一致（§0.6 R5）。
> buffer 复用：`stage` 双角色（`o` 装载 → `do` 装载）与 `acc32` 双角色（`o` 的 fp32 载体 → 乘积载体，O1），与 §4.3 表一致。
> 若编译报 `ub overflow`：按 §9.3 的降级阶梯把 `blk_elems` 降到 2048（`blk_s` 减半），功能不变。
> 若三维 UB buffer 声明不可用（`docs/Tilelang.language/内存操作/T.alloc_shared.md` §2.2.2 标注「3D~5D 有待充分验证」）：
> 按 §9.2 的形态回退（二维声明 + `dim=1`，接受归约 lowering 的标量化代价，或改用 §1.6.4 备选 B 的三维写法变体），
> 该风险与实测反证见 §9.1。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `@tilelang.jit(out_idx=..., target="npuir")` | `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | ✅ 佐证（已交付迁移同款装饰器 + `out_idx`） | 本函数**无** `compile_flags`/`pass_configs`，与源一致 |
| `T.Kernel(一维, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1 | ✅ 已核对 | `is_npu` 最多 2 维 block；本设计一维，`(cid, _)` 解包 |
| `T.serial` + `if` 守卫（static bound） | 同族前作与同语义 npuir 实现（§3.5.3） | ✅ 佐证 | persistent grid-stride；`if` 体内只写预分配 buffer |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md` §2.1/§2.2.2 | ⚠ 已核对（含限制） | Developer 映射 UB；文档 §2.2.2 标注 3D~5D「有待充分验证」，但三维 UB buffer 在已交付迁移（`PL-1.23-flat-spatial-axis-reduce`、`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`）中实测可用 ⇒ 风险登记见 §9.1 |
| `T.copy`（双显式 slice，含尾块动态 extent） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2/§2.4 | ✅ 已核对 | 规则 1（静态 1 维折叠）、规则 2/4（尾块借用 extents）；切片与 `size` 互斥（本设计不用 `size`）；同 dtype |
| `T.vcast`（f16/bf16 → f32，rint） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1/§2.2.2 | ✅ 已核对 | f16→f32 rint ✓、bf16→f32 rint ✓；shape 须一致；src 须为 UB（本设计 src 为 UB `stage`） |
| `T.vmul`（同 shape，原位 dst=src1） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1/§2.2.2/§2.3 | ✅ 已核对 | fp16/fp32 ✓、bf16 ×（本设计在 fp32 域）；同 shape；原位合法；操作数在 UB |
| `T.reduce_sum`（`(M,1,N) → (M,1,1)`，`dim=2`，fp32） | `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1/§2.2.2 | ✅ 已核对 | fp32 ✓、bf16 ×；src/dst 同 rank、归约维尺寸 1；转 `hivm::VReduceOp` |
| `T.min`（标量） | `docs/Tilelang.language/比较操作/T.min.md` | ✅ 已核对 | 仅用于 `real_s` |
| `NPUUtils.get().get_aicore_num()` | `.agents/skills/_shared/standards/core-split-strategy.md` §1 要素② | ✅ 已实查 | 本机返回 20（§5.5 记录查询代码与返回值） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 是否涉及 | 处理方案 |
|------|---------|---------|
| 不支持三维 Kernel | **是** | 源为三维 grid ⇒ 一维 `T.Kernel(num_kernels, is_npu=True)` + 整除/取模任务分解（§0.6 R1） |
| GPU 专用 API（`threads=` / `T.Parallel` 元素循环 / `alloc_fragment` / wgmma / swizzle） | **是** | `threads` 本工厂未用；`T.Parallel` 元素循环 → `T.vmul`（§1.6.2 #3）；计算缓冲 `alloc_fragment` → `alloc_shared`（§0.5）；wgmma/swizzle/`make_dq_layout` 属本文件**其它** kernel，不迁移（§0.5 舍弃行） |
| GEMM 分形 / 非整除 | 否（无 GEMM） | 无 matmul；`D` 无分形要求；`S` 非整除由 `real_s` + 双显式 slice 处理（§0.6 R5） |
| L0C 容量（128KB） | 否 | 纯 Vector，不用 L0 |
| 物理核数适配 | **是** | §5.5 三要素（实查 20 → 纯 Vector 40，persistent 核内串行） |
| bf16 无 Vector 算术 | **是** | fp32 中转链（§0.6 R4，必需；dtype 表佐证） |
| UB 多缓冲膨胀 / 归约链 UB 单价 | **是** | §4.5 双口径逐 buffer 复算 + 工厂 guard + 降级阶梯（§9.3） |
| 三维 UB buffer 的文档状态 | **是** | 文档标注「待充分验证」但同族已交付迁移实测可用；风险与回退形态见 §9.1/§9.2 |
| host 侧不得改写输入张量内容 | 否 | host 侧只做工厂期常量推导；无 pad/cast/permute |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5 / §0.6）

| 差异项 | GPU 源 | 本项目（Ascend NPU） | 转换方案 |
|--------|--------|---------------------|----------|
| Kernel 维度 | `T.Kernel(heads, ⌈S/blk⌉, batch)`（三维） | 一维，最多 2 维 block | 一维 + persistent 任务分解（§0.6 R1） |
| `T.alloc_fragment` 计算域 | CUDA 寄存器 / fragment | v-op 要求 UB 操作数 | 全部 `T.alloc_shared`（映射 UB） |
| tile 形态 | `(256,256)` 方形（`blk=256` 固定） | UB 预算驱动的整行驻留 `(blk_s,1,D)` | §4.5 逐 buffer 复算 + 工厂 guard |
| `T.Parallel(blk,blk)` 元素级累加 | 线程级并行 FMA | v-prefix 向量 op | `T.vmul`（原位，O1） |
| `T.reduce_sum(acc, delta, 1)`（二维、最内轴） | warp shuffle 树 | 二维修约最内轴被标量化 | 三维扁平声明 + `dim=2`（§0.6 R3，附实测证据） |
| D 轴分块循环 | `for k in range(ceildiv(dim,blk))` | 域内整行驻留 ⇒ 单次归约 | 循环消除（§1.6.2 #6），超预算时保留为 fallback §9.3 |
| 隐式 dtype 提升 | fragment 上 `o*do` 隐式转 fp32 | 显式 v-op 域 | 显式 `T.vcast(rint)` + fp32 `T.vmul`（§0.6 R4 / TRAP-C12-copy-dtype-cast） |
| 尾块处理 | `ceildiv` 超发 + 隐式区域裁剪 | 显式动态边界 | `T.min` + 双显式 slice（§0.6 R5） |
| `out_idx=[2]` | 源契约 | 保留 | `@tilelang.jit(out_idx=[2], target="npuir")` |
| `supported_archs=[80,89,90]` | CUDA SM 列表 | 无意义 | wrapper 已按 K5 置 `None`（本 kernel 不改 wrapper） |
| `compile_flags` / `pass_configs` | 本函数**均无** | npuir 无 `compile_flags` 形参 | 无需处理（本函数天然免疫，`VP-2026-0158`） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/deepseek_v32/sparse_mla_bwd.py` | **极高（同数学语义的 npuir 实现）** | `preprocess()`：三维跨步 `T.copy(o[bz, s-slice, bx, d-slice], o_tmp)`、`T.vcast`、累积、`T.reduce_sum(acc, delta, dim=1)`、`T.copy(delta[:,0], Delta[bz, s-slice, bx])`——证实本设计全部装载/写出/归约形态在 `target="npuir"` 上可编译可跑；其 `T.Parallel` 累加形态正是本设计 §1.6.2 要替换的形态 |
| `examples/deepseek_v4/example_sparse_attn_bwd_kernel.py` | 极高（同数学语义，bf16） | 同款 preprocess 的 `alloc_fragment` 变体；示范 bf16 输入路径与 `out_idx=[-1]` 用法（本设计改用显式 `vcast` + `alloc_shared`） |
| `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` | 高（同族混合 kernel） | attention bwd 族中 `delta` 的生产/消费链路（`T.vmul`/`T.vsub` 链）与切片写出形态 |
| `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | 高（**三维 UB + persistent 已交付迁移**） | 三维 `T.alloc_ub((ch,h_in,w_in))` + 三维 `T.vcast` + `T.reduce_sum(..., dim=2)`（`(ch,h,w)→(ch,h,1)`）+ 一维 persistent 分核 + `T.min` 尾块 + 工厂 UB guard 的完整可运行范式 |
| `examples/group_norm/_group_norm_kernel/DESIGN.md` | 高（**同族行归约设计文档**） | §0 迁移分析结构、§1.6 调研/优化写法、§1.6.3 参数级 A/B 写法、§5.5 分核三要素、§8.1 golden 独立性要求 |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` | 中高（同族已交付迁移） | persistent 分核模板、UB 预算 guard、fp32 中转链、L0 分层套件 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention.py` | 极高（**wrapper 契约**） | `FlashAttnBwdPreprocessKernel` 的构造（5 位置参数）与 `forward(o, do) -> delta`；K5/K8 已改写 `supported_archs=None`、去 `tune` |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/_multi_head_attention_bwd_kernels.py` | 极高（**Stage 0 提取的迁移源工件**） | 第 60 至 86 行为本函数（与 GPU 源逐字一致），作为 §0.1 语义对照基准 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高（**同族大 shape 已验证**） | attention 族三维跨步装载（`T.copy(q[bz, k_idx*bn:(k_idx+1)*bn, bx, :], q_frag)`）在绩效达标路径上被使用；`NUM_KERNELS = 24` 为 910B2C 实查值（本机 910B3 = 20） |
| `examples/TileOPs/tests/ops/test_multi_head_attention.py` | 高（**Stage 5 验收口径**） | `MhaBwdFixture` 的 manifest 四档 shape（本设计 §5.5/§8.3 直接采用）；op 级容差 `atol=5e-3, rtol=1e-5` |
| `/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py` | —（**源算子**） | 第 26 至 53 行（kernel）、第 56 至 96 行（Kernel 类）；`tileops/ops/attention/gqa.py` 第 1377 至 1381 行证实 `delta` 的 S 连续读法 |
---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `o` | `(batch, seq_len, heads, dim)` | float16 / bfloat16 | BSHD；head-dim 最内连续（stride 1），seq 轴 stride = `heads·dim`，均为工厂期常量 |
| `do` | `(batch, seq_len, heads, dim)` | 同 `o` | `o` 的梯度；op 层调用前已 `contiguous()`（`tileops/ops/attention/gqa.py` 第 1379 行，host 元数据/连续性操作，允许） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `delta` | `(batch, heads, seq_len)` | **float32** | `out_idx=[2]` 自动分配；seq 轴最内连续（stride 1），head 轴 stride = `seq_len`；与源输出 shape/dtype 完全一致（`source_output_shape = (B,H,S)`） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 作用（角色复用时序） | 声明字节/元素 |
|-----------|-------|-------|----------|---------------------|--------------|
| `stage` | `(blk_s, 1, dim)` | 输入 dtype | UB（`alloc_shared`） | ① `o` 装载 → ② `do` 装载（**双角色**，§0.6 R4） | 2 |
| `acc32` | `(blk_s, 1, dim)` | float32 | UB | ① `o` 的 fp32 载体 → ② 乘积载体（**双角色**，O1 原位） | 4 |
| `aux32` | `(blk_s, 1, dim)` | float32 | UB | `do` 的 fp32 载体 | 4 |
| `dlta` | `(blk_s, 1, 1)` | float32 | UB | 行归约结果（`4·blk_s` B，相对可忽略 ≈ 0.06 B/元素 @D=64） | ≈0 |

**驻留元素数口径**：`E = blk_s·D`（单元元素数）；**声明量 = 10 B/元素**（2 + 4 + 4）。无 `(blk_s, dim)` 乘积专用缓冲（O1 已消除）。

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube / L0 / L1 参与）：

GM[o (B,S,H,D) dtype]  --T.copy(双显式 slice；逐 (b,h) 跨步行块，行 burst = D·eb)--> UB[stage (blk_s,1,D)]
UB[stage]              --T.vcast(rint：fp16/bf16 → fp32，逐位精确)--------------------> UB[acc32]
GM[do]                 --T.copy(同形；复用 stage)--------------------------------------> UB[stage]
UB[stage]              --T.vcast(rint)-------------------------------------------------> UB[aux32]
（全部计算在 UB 内闭环：乘积 → 行归约；无中间结果落回 GM、无跨引擎传输 ⇒ 容量项 = 0）
UB[acc32] × UB[aux32]  --T.vmul(原位 dst=acc32，O1)------------------------------------> UB[acc32 = o·do]
UB[acc32]              --T.reduce_sum(dim=2)-------------------------------------------> UB[dlta (blk_s,1,1)]
UB[dlta][0:real_s]     --T.copy(1-D 连续段，burst = blk_s·4 B)--------------------------> GM[delta (B,H,S) fp32]
```

关键约束：GM 不可直达 fragment；v-op / reduce 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3 第 1 条）；
本设计不经 fragment、不经 L1/L0、无跨引擎往返（⇒ §1.6.0 roofline 的容量项 = 0）。

### 4.5 UB 内存预算

**预算口径（双口径并列，遵循 `hardware-cost-model.md` §5 与 `CONST-capacity-910B2C`）**：

1. **口径 A（声明量 × auto multi-buffer 膨胀）**：`CONST-capacity-910B2C` 的默认档膨胀 **~1.7x**
   （BishengIR `ub overflow, requires N bits while 1572864 bits available` 反推；另一档 `--enable-auto-multi-buffer=false`
   为 ×1.10–1.12）⇒ 17 B/元素 ⇒ `E ≤ 196608 / 17 = 11 565`。
2. **口径 B（声明量 + 归约链 lowering 单价）**：`CONST-reduce-ub-per-element` 记录 **BishengHIR 归约 lowering 的 UB 单价
   按输入 tile 元素数线性记账**（`hivm.hir.vreduce` lowering 物化多份中间张量；该记账**与 auto-multi-buffer 开关无关**，
   实测 open/close 数字完全相同）。其三维整平面（`(CH,1,H*W)` + 原位 `vmul` + `reduce_sum`，与本设计同为「扁平空间轴」形态）
   单价为 **≤32 B/元素**。（该条目的适用域限定为三维整平面形态；同族另一条 2-D 行归约实测单价低至 8 B/元素，
   故 32 是**保守上界**。）⇒ 10 + 32 = 42 B/元素 ⇒ `E ≤ 196608 / 42 = 4 681`。
3. **取小**：`_UB_ELEMS_BUDGET = 4 096`（2 的幂，留余量）——覆盖口径 A 与口径 B 的上界；
   单元取值为 `raw = max(1, blk_elems // D)`、`blk_s = raw`（`raw < 8`）或 `(raw // 8) · 8`（`raw ≥ 8`，对齐向量宽）。

**逐 workload 复算（`blk_elems = 4096`、`blk_s` 按上式）**：

| workload（manifest） | (B,S,H,D) | dtype | `blk_s` | `E = blk_s·D` | 声明量（10 B/元素） | 口径 A（×1.7 = 17 B/元素） | 口径 B（42 B/元素） | ≤192KB |
|----------------------|-----------|-------|---------|---------------|---------------------|---------------------------|---------------------|--------|
| `smoke-bwd-fp16` | (1,1024,8,64) | fp16 | 64 | 4 096 | 40 960 B（40 KB） | 69 632 B（68 KB） | 172 032 B（168 KB） | ✓ |
| `smoke-bwd-bf16` | (1,1024,8,64) | bf16 | 64 | 4 096 | 40 KB | 68 KB | 168 KB | ✓ |
| `full-bwd-fp16-large` | (16,2048,16,128) | fp16 | 32 | 4 096 | 40 KB | 68 KB | 168 KB | ✓ |
| `full-bwd-bf16` | (4,4096,16,128) | bf16 | 32 | 4 096 | 40 KB | 68 KB | 168 KB | ✓ |
| 测试用极端小档（`D = 1`） | (1,8,1,1) | fp16 | 4096 | 4 096 | 40 KB | 68 KB | 168 KB | ✓ |
| **域上界（超即拦截）** | `D > 4096`（`raw = blk_elems//D < 1`）：单行驻留本身即超预算 ⇒ 工厂 guard 拒绝，走 §9.3 的 D 分块 fallback | — | — | — | — | — | — | ⚠ 见 §9.3 |

**降级阶梯（工厂 guard + Stage 4 反解）**：

| 档 | 元素预算 | 可行条件 | 发射项（full-fp16 档核内轮数） |
|----|---------|----------|------------------------------|
| 上探档 | 8 192 | 归约链 lowering 单价 ≤ 24 B/元素（`196608/8192 = 24`；落在 [8, 32] 区间内，需 Stage 4 反解；或 auto multi-buffer 关闭档） | 205（减半） |
| **默认档** | **4 096** | 覆盖单价上界 32 B/元素（口径 B） | **410** |
| 兜底档 | 2 048 | 任何单价 ≤ 88 B/元素 | 820 |

> **UB guard**：工厂层校验 `blk_s·D ≤ blk_elems`；编译期若报 `ub overflow`（`blk_s·D` 超出真实单价允许值），
> 按阶梯降档（无需改代码，只改 `blk_elems` 实参/默认值），功能与精度完全不变（§9.3）。
> **诚实标注**：口径 B 的单价取值区间（8–32 B/元素）来自 pattern-library 两个不同形态的实测条目，
> **本设计形态（3-D 扁平 + 原位 vmul + 单次 reduce_sum）的精确单价未实测** ⇒ 按上界 32 处理并给出阶梯；
> Stage 4 首次编译/首轮 msprof 即可反解（`ub overflow` 报文或 Task Duration），并按 `hardware-cost-model.md` §3 回写。

### 4.6 动态轴定义

无运行时动态 shape：`batch`/`heads`/`seq_len`/`dim`/`dtype` 均为工厂期常量（per-build JIT 特化，与源工厂语义一致）。
kernel 内唯一的运行时标量是尾块行数 `real_s = T.min(blk_s, seq_len − s0)`。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")   # delta 在 index 2（源契约）
def _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype,
                                     blk_elems=_UB_ELEMS_BUDGET): ...
```

无 `pass_configs`（无 `TL_ENABLE_FAST_MATH` 需求：无超越函数）、无 `compile_flags`（源本函数亦无；且 npuir 的 `tilelang.jit`
无该形参）。Stage 4 的 A/B 备选会在工厂内附加 `pass_configs={"npuir.enable_auto_multi_buffer": False}`（§1.6.3 参数级计划）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector（逐元素乘积 + 逐行归约），无 MAC 密集段 ⇒ 不涉及 Cube / L0A / L0B / L0C / L1 与分形约束。

### 5.2 Block 划分

```python
blk_elems   = _UB_ELEMS_BUDGET                      # 4096（§4.5；Stage 4 A/B 档 8192）
raw         = max(1, blk_elems // dim)              # 单行驻留的元素预算
blk_s       = raw if raw < 8 else (raw // 8) * 8    # ≥8 时对齐到 fp16/bf16 向量宽；<8 时容量优先
blk_d       = dim                                   # 归约轴不分块（整行驻留，§1.6.3 主选）
num_logical     = batch * heads * ceil(seq_len / blk_s)   # 逻辑任务数（一维）
num_kernels     = min(num_logical, vector_cores)          # 实际启动内核数（vector_cores = 40）
num_local_tasks = ceil(num_logical / num_kernels)         # 核内串行任务数（host 编译期常量，静态边界）
```

**选择理由**：① 归约轴 `D` 不分块——整行驻留使输入只被 GM 扫描一遍（`R3` 表中流量最小）且单行 burst 最大；
② `blk_s` 由 UB 元素预算与 `D` 反推（不是源的固定 256——256×256 方形 tile 在 NPU UB 上放不下，§0.5）；
③ 「逐 `(b,h)` 单元」而非「逐 `(b, s-block)` 单元」——后者单元元素数为 `blk_s·H·D`，在同预算下 `blk_s` 退化
（D=128/H=16 时 `blk_s=2`，§1.6.3 候选 3 / §1.6.4 备选 B）。

### 5.3 约束分析

- **对齐约束**：UB 对齐（32B）由编译器管理；尾轴 `D` 在 manifest 域 `∈ {64,128}` 上均为 fp16/bf16 向量宽（×8）的整数倍
  ⇒ 尾 lane 浪费 0%；`blk_s ∈ {32, 64}` 亦为 8 的整数倍。输出 1-D 段长 `blk_s·4 B ∈ {128, 256} B`（32B 对齐 ✓）。
- **UB 容量**：§4.5 逐 workload 复算，双口径下最大 168 KB ≤ 192 KB ✓。
- **尾轴整除性**：见 §1.6.3 轴质量评分（manifest 两档 0% 浪费）。
- **L0 / L1 / 分形**：不适用（纯 Vector）。
- **burst 粒度**：装载行 burst = `D·eb`（128 B @D=64 / 256 B @D=128），行间 stride = `H·D·eb`；
  与 attention 族主 kernel 的装载形态一致（§3.5.3），折损率由 Stage 4 实测（§1.6.4 裁决计划）。

### 5.4 注意事项

- **`seq_len % blk_s ≠ 0`（尾块）**：`real_s = T.min(blk_s, seq_len − s0)` + src/dst 双显式 slice；v-op / `reduce` 全 buffer 操作、
  垃圾行按行隔离（`dim=2` 逐行独立）、写出只覆盖 `[0:real_s]`（§0.6 R5）。
- **工厂 UB guard**：`blk_s·D > blk_elems` 抛 `ValueError`；编译期 `ub overflow` 按 §4.5 阶梯降档。
- **wrapper 交互**：NPU wrapper 的 `FlashAttnBwdPreprocessKernel` **无** `default_config` / `autotune_configs`（源即如此），
  调用形态为 `_flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype_str)` 后 `forward(o, do)`；
  本设计**保持 5 个位置参数不变**（`blk_elems` 为带默认值的可选参数），故集成后无需改 wrapper。
  该设计把源「固定 `blk=256`」的**不可调**变成**工厂内 shape 驱动**（`blk_s = f(dim)`），是纯内部改进，不改变接口契约。
- **`D` 极小或极大**：`raw = max(1, blk_elems // dim)`；`raw < 8` 时 `blk_s = raw`（**容量契约优先于向量宽对齐**，
  如 `D = 4096` ⇒ `blk_s = 1`），单元元素数恒 `≤ blk_elems`；`D = 1` ⇒ `blk_s` 截顶到 4096。功能正确性由 L0-8/L0-9 覆盖。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**② 物理核数依据（实查记录，2026-10-16 本设计执行）**：

```python
# 在项目环境下执行（/workspace/venv/bin/python）
from tilelang.utils.npu_utils import NPUUtils
from tilelang.utils.npu_arch import AscendArch
NPUUtils.get().get_aicore_num()      # -> 20      （物理 AI Core）
AscendArch().aivector_core_num       # -> 40      （纯 Vector 核数，直接实查）
AscendArch().ub_cap                  # -> 196608  （= 192 KB，实查）
# torch.npu.get_device_name(0)       # -> "Ascend910B3"
```

设备：**Ascend910B3**；**物理 AI Core = 20**、**纯 Vector 逻辑核数 = 40**（`get_aicore_num() × 2`，与
`aivector_core_num` 直查一致），与 pattern-library `constants.md` 的 `CONST-aicore-910B2C` 条目跨型号数据点
（910B3：20 / 40 / 196608 B）一致；**与 910B2C 的 24 / 48 不同，不得跨型号复用核数值**。
工厂每次构建重查 `get_aicore_num()`（跨型号自动适配）。

**① 逻辑核数计算**：`num_logical = batch × heads × ⌈seq_len / blk_s⌉`（仅 seq 轴分块，逻辑网格一维）：

| workload（manifest） | (B,S,H,D) | `blk_s` | `num_logical` | `num_kernels = min(逻辑, 40)` | `num_local_tasks` | 负载分布 |
|----------------------|-----------|---------|---------------|-------------------------------|-------------------|----------|
| `smoke-bwd-fp16` | (1,1024,8,64) | 64 | 1×8×16 = **128** | 40 | **4** | 128 = 40×3+8 ⇒ 8 核 4 任务、32 核 3 任务（不平衡 ≤1 任务） |
| `smoke-bwd-bf16` | (1,1024,8,64) | 64 | **128** | 40 | **4** | 同上 |
| `full-bwd-fp16-large` | (16,2048,16,128) | 32 | 16×16×64 = **16 384** | 40 | **410** | 16 384 = 40×409+24 ⇒ 24 核 410 任务、16 核 409 任务 |
| `full-bwd-bf16` | (4,4096,16,128) | 32 | 4×16×128 = **8 192** | 40 | **205** | 8 192 = 40×204+32 ⇒ 32 核 205 任务、8 核 204 任务 |

**③ 规模判定与分核方案**：四档逻辑核数均 **> 物理核数 40** ⇒ 判定为**极大规模**，
统一采用 **persistent 核内串行**形态：`num_kernels = min(num_logical, 40)`（= 物理核数整数倍 1×），
核内 `for i in T.serial(num_local_tasks)` + grid-stride 映射 `uid = i·num_kernels + cid` + `if uid < num_logical` 守卫：

- **核内串行循环边界静态**：`num_local_tasks` 为 host 层 Python 编译期常量（`T.ceildiv`/整数运算在 trace 期折叠为 `IntImm`），
  满足 `core-split-strategy.md` §1 要素③ 的静态边界要求。
- **负载均衡**：不平衡 ≤ 1 个任务（上表第 7 列）；任务的元素数相同（`blk_s·D` 恒定，除每 (b,h) 最后一个尾块），
  故最大核时延 ≈ `ceil(num_logical/40) × 单任务时延`。
- **收益**：消除逻辑核超发（full-fp16 档 16 384 个逻辑核若直接下发会被串行调度）的核启动开销，
  并取得 grid-stride 聚集窗口的 MTE2 访问效益（`CONST-mte2-degradation` 记录 grid-stride 优于每核连续分块 +3.2%）。
- **无核间归并**：行独立 ⇒ 无 `sync_block_set/wait`、无 workspace、无 flag id 预算问题
  （`CONST-flag-id-budget` 不适用），跨核代价 = 0（已计入 §1.6.0 R3 的「跨核结构」列）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表只列 block / task 级调度与「被消除的源循环」。

| 维度 | 循环类型 | API | 理由 |
|------|---------|-----|------|
| 任务维（逻辑任务 = `(b,h,s-block)`） | 核级并行 + 核内串行 | `T.Kernel(num_kernels, is_npu=True)` + `T.serial(num_local_tasks)` | persistent 分核（§5.5）；任务间完全独立；边界静态 |
| 单元内元素维（`blk_s × D`） | **向量化（无循环）** | `T.vcast` / `T.vmul` / `T.reduce_sum` 全 buffer 操作 | §1.6.2 #2/#3/#4 全部替代 |
| 归约轴 `D` | **无循环**（单次 `reduce dim=2`） | `T.reduce_sum(acc32, dlta, dim=2)` | 整行驻留（域内 `D ≤ 4096`）；源 `for k in range(ceildiv(dim, blk))` 被消除（§1.6.2 #6） |
| 尾块（`seq_len % blk_s ≠ 0`） | 标量 `min` + 双显式 slice | `T.min` + `T.copy` slice | §1.6.2 #7 / §0.6 R5 |

**不使用 `T.Parallel`**（Developer 下其元素级循环引入标量化形态；本设计的元素级并行全部由 v-prefix 全 buffer 操作承担，
同族实测教训见 §1.6.2 #3 与 `TRAP-parallel2d-bm-ge2-scalarize`）。
**不使用 `T.Pipelined`**（见 §6.3）。
**不使用 `T.Persistent`**（本分支 Expert 专用模板；Developer 下的 persistent 语义由 `T.serial` + grid-stride 表达，
与已交付迁移一致）。

### 6.2 循环伪代码

见 §3.3（结构：`with T.Kernel(num_kernels, is_npu=True) as (cid, _):` → UB buffer 分配（在 `T.serial` **之外**，跨任务复用）
→ `for i in T.serial(num_local_tasks):` → `if uid < num_logical:` → 计算体（§3.1 步骤 2 至 8））。

### 6.3 流水线优化

**不使用显式 `T.Pipelined`**。理由：① 单任务内是顺序依赖链（loader → cast → mul → reduce → store），
**没有可重叠的 K 维迭代**（`D` 不分块，源的分块循环已被消除）；
② 本分支实测：单元循环上的 `T.Pipelined(2/3)` **无增益**（`CG-2026-0027`「于单元循环实测无增益（未能形成归约 consumer 与
DMA 的双缓冲重叠）」；`BP_pipeline_overlap` 亦记录该结构无收益）；
③ 任务之间的装载/计算重叠交由编译器 auto multi-buffer 承担（同族实测：同形流量与链路下全部向量 pass 被 MTE2 窗口隐藏，
见 `CONST-copy-floor-method`）。
Stage 4 选项已在 §1.6.3 参数级 A/B 中登记（auto multi-buffer 关闭档 + 元素预算提档）；若 baseline 相对 §1.6.0 的估算
下界偏差 > 2×，再评估显式双缓冲与 D 分块流式（§9.3 fallback）。

### 6.4 尾块处理

- `seq_len % blk_s ≠ 0`：最后一个 s-block `real_s < blk_s` —— 装载/写出双显式 slice 截断，v-op 对完整 buffer 计算、
  垃圾行按行隔离并被输出丢弃（§0.6 R5）；
- `num_logical % num_kernels ≠ 0`：`if uid < num_logical` 守卫跳过越界任务（无空核，因为 `num_kernels ≤ num_logical`）；
- `D` 维无尾块（不分块，整段归约）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步（Developer 模式）**——无任何手动同步原语。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy`(GM→UB) 之后 | 自动 | 编译器在 MTE2→V 拓扑插入屏障 |
| `T.vcast` → `T.vmul` → `T.reduce_sum` 之间 | 自动（顺序依赖） | 同引擎顺序执行 |
| `T.copy`(UB→GM) 之前 | 自动 | 编译器在 V→MTE3 拓扑插入屏障 |
| `T.serial` 任务之间 | 自动 | 同核串行、无并发；buffer 跨任务复用由编译器自动插入 WAR/RAW 屏障 |

**无核间协作**：行（任务）完全独立、无跨核归并 ⇒ 无需 `T.sync_block_set` / `T.sync_block_wait` / `T.pipe_barrier`、
无需 workspace、无 flag id 预算问题（`CONST-flag-id-budget` 不适用）。
**无 `pass_configs`**：Developer 默认流水线；不加 `TL_ENABLE_FAST_MATH`（本算子无超越函数，加了无收益且改变 fp 语义的
风险不可接受——`delta` 是下游梯度的减项，语义应与源一致）。
---

## 8. 验证方案

### 8.1 Golden 函数

> **迁移任务：golden 以 §0.1 的源算子语义为唯一依据，不复刻 §0.6 的 NPU 算法**（无 persistent 结构、无三维扁平声明、
> 无 fp32 显式中转、无原位乘积）——保证验证独立性。
> **来源说明**：GPU 仓没有针对本函数的独立单测（`delta` 是 op 内部产物，`tests/ops/attention/test_mha.py` 只用 SDPA 反向
> 间接校验），故 golden 依据 = ① 源 kernel 的算术（§0.3 的 S2 至 S7 逐句）+ ② 主 kernel 的使用点（
> `tileops/ops/attention/gqa.py` 与 `_gqa_bwd_kernel` 中 `qkt*(dst − delta)`）+ ③ FlashAttention 论文的 $D_i=\sum_j(o_{ij}\circ do_{ij})$
> 定义。三者一致，无歧义。

```python
import torch

def golden_flashattn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """Row-wise `delta = rowsum(o * do)` reference (DESIGN section 0.1).

    o, do : (B, S, H, D) fp16/bf16 (BSHD), device-independent (CPU is fine)
    return: (B, H, S) fp32

    Semantics: products and accumulation in fp32 (the source accumulates into a
    fp32 accumulator); upcast fp16/bf16 -> fp32 is exact; no normalisation, no
    dtype round-back (delta stays fp32).
    """
    o32 = o.detach().float()
    do32 = do.detach().float()
    return (o32 * do32).sum(dim=-1).transpose(1, 2).contiguous()   # (B,S,H) -> (B,H,S)
```

**独立性核对**：golden 不含 `blk_s`/持久化/`vcast`/`vmul`/`reduce` 任何结构信息；`sum(-1)` 用 torch 的归约实现，
与 kernel 的归约树序**不同**（这正是 §1.6.1 P2 机器验证覆盖的容差来源）。

### 8.2 精度标准

| 层级 | dtype | atol | rtol | 依据 |
|------|-------|------|------|------|
| **本 kernel（`delta`）** | **float32** | **1e-4** | **1e-4** | `delta` 是 fp32 中间量：乘积**逐位精确**（§1.6.1 P1），唯一误差源是 `D` 项 fp32 求和的结合序（相对误差 ~`D·eps` = 128×6e-8 ≈ 7.7e-6 < 1e-4）；与 §1.6.1 P2 的机器验证口径一致（该脚本用同一组 `atol=rtol=1e-4`） |
| 下游 op 级（`MultiHeadAttentionBwdOp`） | fp16 / bf16 | 5e-3 | 1e-5 | `examples/TileOPs/tests/ops/test_multi_head_attention.py` 的 `test_mha_bwd`（Stage 5 验收口径） |
| 输入 dtype 无关性 | — | — | — | 上抛 fp16/bf16 → fp32 逐位精确（P1），故同一 fp32 门禁对两种输入 dtype 都成立 |

> `delta` 的相对误差在消去位形（`o·do` 混合符号、和远小于项）下会放大——这是任何 fp32 求和的固有性质，
> 源实现同样如此（§1.6.1 P2 的角点行：源序相对 fp64 的偏差与候选同量级）。故 L0 以**常规分布数据 + 显式角点**双轨覆盖。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确 + 边界语义 + 分核/UB 契约」；完整分层套件（L1/L2/Boundary）交由 `tilelang-op-develop` 扩展。
对照 = §8.1 golden（CPU）；输入在 CPU 生成后 `.npu()`（设备无关生成，避免用 NPU 数据当 golden 输入）。

| # | 用例 | (B,S,H,D) / dtype | `blk_s` | 验证目标 | 容忍度 |
|---|------|-------------------|---------|----------|--------|
| L0-1 | manifest `smoke-bwd-fp16` | (1,1024,8,64) fp16 | 64 | 代表小档；4 轮 persistent；fp16 中转链 | 1e-4 |
| L0-2 | manifest `smoke-bwd-bf16` | (1,1024,8,64) bf16 | 64 | bf16 路径（**必须** fp32 中转，§0.6 R4） | 1e-4 |
| L0-3 | manifest `full-bwd-fp16-large` | (16,2048,16,128) fp16 | 32 | 代表大档；`num_local_tasks = 410` 的持久路径；`blk_s=32` | 1e-4 |
| L0-4 | manifest `full-bwd-bf16` | (4,4096,16,128) bf16 | 32 | 同上 bf16；`num_local_tasks = 205` | 1e-4 |
| L0-5 | 尾块 A（`S % blk_s = 1`） | (1,1025,8,64) fp16 | 64 | `1025 = 64×16 + 1` ⇒ `real_s=1`，垃圾行隔离 + 写出裁剪 | 1e-4 |
| L0-6 | 尾块 B + bf16 | (2,1000,3,128) bf16 | 32 | `1000 = 32×31 + 8`；`num_logical = 2×3×32 = 192`（非 40 整数倍）⇒ 守卫路径 | 1e-4 |
| L0-7 | `D` 非向量宽整数倍 | (1,64,4,63) fp16 | 65 → 取 64 | `D=63`（尾 lane 浪费 1.6%）；`blk_s = 4096//63 → 64` | 1e-4 |
| L0-8 | `D = 1`（极小） | (1,64,4,1) fp16 | 4096 | 退化边界：`blk_s` 截顶；`reduce dim=2` 单元素 | 1e-4 |
| L0-9 | `D = 4096`（单行驻留边界） | (1,512,2,4096) fp16 | 1 | `blk_s = 1`（预算只够单行）；验证 `E = blk_s·D ≤ 4096` 契约不破 | 1e-4 |
| L0-10 | UB 预算阶梯 | (1,1024,8,64) fp16，`blk_elems ∈ {8192, 4096, 2048}` | 128 / 64 / 32 | 三档均可编译（记录哪档 `ub overflow`）→ 反解归约链 UB 单价（§4.5） | 1e-4 |
| L0-11 | 全零输入 | (1,128,4,64) fp16，`o = 0` | 64 | `delta` 逐位 `+0.0` | 精确 |
| L0-12 | 常量输入（`o = do = 1`） | (1,128,4,64) fp16 | 64 | `delta ≡ D`（=64，fp32 精确） | 精确 |
| L0-13 | `±inf` / `NaN` 传播 | (1,16,2,128) fp16，构造 inf 行 / NaN 行 | 32 | 与 golden `equal_nan=True` 一致；无越界污染（行隔离） | 精确（NaN 位置一致） |
| L0-14 | 接口契约 | 工厂 5 位置参数调用 + `out_idx=[2]` 自动分配 | — | 签名与源一致；`forward(o, do)` 返回 `(B,H,S)` fp32 | — |

**通过条件**：全部用例 `max_diff` 落在 `atol + rtol·|ref|` 内（L0-11/12/13 为精确档）；
**建议 L1 补充**（交 Stage 3 扩展）：`blk_elems` 扫描（§1.6.3 参数级 A/B 与 §1.6.4 备选 B 的 A/B 基线采集）、
`H·D ≤ 512` 档的备选 B 对照（§1.6.4 裁决计划）、`blk_s < 8` 的退化档、工厂 guard 契约（非法 dtype / 非正 shape 抛 `ValueError`）；
**Boundary 建议**：fp16 饱和（±65504）乘积上溢边界、次正规输入、混合符号消去位形（与 §1.6.1 的 D-1 角点集对齐）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 / 风险 | 影响 | 处理方案 |
|------------|------|---------|
| **归约链 lowering 的 UB 单价未实测**（区间 8–32 B/元素，`CONST-reduce-ub-per-element`） | `blk_s` 取大 ⇒ 编译期 `ub overflow`（编译失败，非静默错值） | 默认档按**保守上界** 32 B/元素取 `E = 4096`（§4.5）；工厂 guard + 降级阶梯（8192/4096/2048）；L0-10 直接测出真实档位并反解单价；Stage 4 回写 constants.md |
| **三维 UB buffer 的文档状态**（`docs/Tilelang.language/内存操作/T.alloc_shared.md` §2.2.2：「Ascend A2/A3：1D~2D；3D~5D 有待充分验证」） | 若三维声明在目标工具链不可用，本设计的**核心布局决策**（三维扁平归约）失效 | **实测反证**：三维 UB tile + `dim=2` 归约在已交付迁移中实测可用（`PL-1.23-flat-spatial-axis-reduce`：910B3 + CANN 9.0.0 实测 166.26 → 16.68 µs；`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` 为可运行实现）。**回退形态**（§9.2 第 1 行）：二维声明 + `dim=1`（接受标量化代价，实测单 op 5.6 → 155.1 µs 的教训）或改用 §1.6.4 备选 B 的等价三维写法变体；回退触发时**必须**在 Stage 3 记录并回写本节 |
| **跨步行块装载的 MTE2 折损率未实测**（`D·eb` = 128/256 B burst + `H·D·eb` 步长） | 流量项可能高于 §1.6.0 的 346 µs（乐观） | 有同族强先验：attention 族主 kernel 用同款装载且在绩效达标路径上（§3.5.3）；§1.6.4 裁决计划用 msprof 反解折损率；若折损显著且 `H·D ≤ 512`，切换备选 B |
| **发射项与流量项同量级**（full-fp16 档：流量 346 µs vs 发射上界 410–820 µs） | 若发射单价偏高，代表 shape 会显著超出流量下界 | §1.6.0 已给出双档与同族校准（×0.44）；**第一杠杆 = 元素预算提档**（§1.6.3 参数级 A/B，410 → 205 轮）；Stage 4 回填「设计估算 vs 实测」偏差行 |
| **归约结合序精度**（P2 的诚实边界） | 若 `T.reduce_sum` 的 lowering 为**全串行**累加（而非树形），消去/饱和位形下超容差元素会增多（机器验证：3 例 / 最大 ulp 3.1e9） | 常规数据下无影响（P2：判否案例 0）；Stage 3 的 L0-13（±inf/饱和）若出现精度偏差 ⇒ 改用 `clear=False` 的 D 分块部分和**固化**结合序（§9.3 fallback），并把该决定回写本节 |
| **`D` 极大（> 4096）时的单行驻留不足** | `blk_s` 被压到 1 以下（公式 `max(1, ...)` 兜底），单元元素数超预算或粒度退化 | 超本任务域（manifest `D ∈ {64,128}`）；工厂 guard 明确拒绝并给出 fallback 路径（§9.3） |
| **`blk_s < 8` 时的向量宽不对齐**（如 `D = 4096` ⇒ `blk_s = 1`） | 向量化利用率下降（不改变正确性） | 域外退化档；L0-9 覆盖正确性；容量契约（`E ≤ 4096`）优先于对齐 |
| **尾块垃圾行的 NaN/Inf** | 理论上有越界污染风险 | 归约 `dim=2` **逐行独立** ⇒ 垃圾行不污染有效行；写出只覆盖 `[0:real_s]`；L0-5/L0-6 覆盖 |
| **roofline 常数版本戳** | `CONST-mte2-degradation` / `CONST-vector-launch-overhead` 绑定 910B2C + CANN 8.5.0（本机 910B3 + CANN 9.0.0），且 48 核 → 40 核为线性外推 | 估算只用于候选排序与 Stage 4 对账基线；结构性结论（水平 vs 垂直、三维 vs 二维、单元粒度）不依赖常数精确值；引用处已逐条标注版本戳与「未实测假设」 |
| **设备型号绑定** | 核数（20 / 40）与 `ub_cap`（196608）绑定 Ascend910B3 | §5.5 记录实查代码与返回值；工厂每次构建重查 `get_aicore_num()`，跨型号自动适配 |
| **接口契约（不改 wrapper）** | 若为调优而修改工厂签名（如把 `threads` 加回），会破坏 `FlashAttnBwdPreprocessKernel` 的调用 | 本设计只**新增带默认值**的可选参数；5 个位置参数与 `out_idx=[2]` 严格保持（§3.3 接口保持声明） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| **二维声明 + `reduce(dim=1)`** | 照源直译为 `(blk_s, D)` 二维 tile 并归约最内轴 | 归约 lowering 被标量化（实测单 op 5.6 → 155.1 µs）；并可能触发 DMA 布局传播退化 | 三维扁平声明 `(blk_s,1,D)` + `dim=2`（§0.6 R3 / §1.6.3 候选 6 的证据行）；若三维不可用则用 §1.6.4 备选 B 的等价三维变体并记录 |
| bf16 直传 v-op | 未做 fp32 中转 | 编译错误（dtype 不支持） | fp32 中转链（§0.6 R4）；dtype 表佐证 |
| 跨 dtype `T.copy` 当 cast | 省掉 `stage` 与一次 `vcast` | 舍入模式不可控（RINT 硬编码）、不减 pass | 显式 `T.vcast(rint)` + 同 dtype `T.copy`（`TRAP-C12-copy-dtype-cast` + `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条） |
| `T.Parallel` 元素级循环残留 | 照源直译 `T.Parallel(blk_s, blk_s)` 写 `acc += o*do` | 引入标量化形态（本仓实测标量化可占壁钟 45%；`TRAP-parallel2d-bm-ge2-scalarize`） | 改 `T.vmul` 全 buffer 原位（§1.6.2 #3） |
| UB dst 非零起点切片 | 手写列/行偏移 slice | VEC 对齐错误（`TRAP-UB-dst-align`） | 所有 UB dst 用零起点双显式 slice（§3.3 形态） |
| `T.copy` 索引式区域 | 沿用源 `T.copy(o[bz, by*blk, bx, k*blk], buf)` 的 `size` 推定形式 | 区域语义静默误读（`TRAP-T-copy-region-semantics`） | 双显式 slice + 显式 `real_s` extents |
| `T.reshape` 重解释形状 | 想把二维 tile 变三维声明 | npuir 分支 `T.reshape` 被 `npuir_reshape(src,dst)` 覆盖，通用形语义漂移（`TRAP-npuir-reshape-dst-buffer-shadow`） | **直接按三维声明分配**（本设计不调用 `T.reshape`） |
| 标量操作数放在 `T.Kernel` 外 | `blk_s` 等直接用 Python 全局 | 编译报错（标量须在 `T.Kernel` 内、`T.Scope` 外） | 工厂层求值、kernel 体内作为标量/常量使用（§3.3 形态） |
| `threads=` 参数残留 | 从同文件其它 kernel 误搬 | 无效果且误导调优（`TRAP-threads-kwarg-noop`） | 本函数本无 `threads`；不要加 |
| 误搬 `make_dq_layout` / swizzle | 从同文件其它 kernel 复制 | NPU 无 SMEM bank conflict，手动 swizzle 无意义（§0.5 舍弃行） | 不迁移 |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|---------|
| `blk_s·D > blk_elems`（单元超预算） | 工厂 guard 抛 `ValueError`；按 §4.5 阶梯降档（8192 → 4096 → 2048） |
| 编译期 `ub overflow` | 同上降档；若 2048 档仍溢出，则说明本工具链的归约链单价远高于 32 B/元素 ⇒ 追加一档 1024 并回写 constants.md 候选 |
| `D > 4096`（单行驻留不足） | 超本任务域（manifest 最大 128）。fallback = §1.6.0 R2 的 D 分块流式（`(blk_s,1,d_c)` + `T.clear(dlta)` + `T.reduce_sum(..., clear=False)` 跨分块累加，pass 数守恒、无 GM 重读；单行 burst 变碎为代价）。该路径**本设计不实现**，仅登记（若 Stage 3 遇到须回 Stage 1 补设计） |
| 归约精度异常（消去/饱和位形） | 改 D 分块部分和（`clear=False`）**固化**结合序，降低对 lowering 树的依赖；该选项与 §9.3 上一行的结构相同（共享实现路径） |
| `seq_len < blk_s` | `real_s = seq_len`，`num_logical = B·H`（每 (b,h) 一个任务），尾块路径自然覆盖 |
| `batch = heads = 1` | `num_logical = ⌈seq_len/blk_s⌉`；若 ≤ 40 则 `num_kernels = num_logical`（无核内串行轮次、无空核），属 §5.5 要素③ 的「无需适配」分支 |
| 非连续输入 | op 层已对 `do` 做 `contiguous()`（源同款）；kernel 只见连续 BSHD（本设计 host 侧不做任何张量内容改写） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/
├── DESIGN.md                       # 本设计文档（含 §0 迁移分析、§1.6 调研与优化、§5.5 分核三要素）
├── verify_equiv.py                 # D-1 等价性机器验证脚本（已执行：O1/P1/P2 全 EQUIV_PASS）
├── _flashattn_bwd_preprocess_kernel.py   # 算子实现（Stage 3 产出）
└── RETROSPECTIVE.md                # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | 32 案例 ×（O1 / P1 / P2）+ R1 否决证据；`SUMMARY: ALL_EQUIV_PASS` |
| `_flashattn_bwd_preprocess_kernel.py` | ⬜ 待实现（Stage 3） | 工厂 + kernel（+ §8.1 golden 与 §8.3 的 L0 套件） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |

### 10.3 命名规范

- 项目目录：`multi_head_attention`；算子目录 / 实现文件：`_flashattn_bwd_preprocess_kernel`（与 Stage 0 提取件的函数名一致，不裁剪不变换）
- 工厂函数：`_flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype, blk_elems=4096)`（前 5 个参数与源签名逐位一致）
- Golden 函数：`golden_flashattn_bwd_preprocess(o, do)`
- 常量：`_UB_CAPACITY_BYTES` / `_UB_ELEMS_BUDGET` / `_VECTOR_WIDTH`

### 10.4 实现顺序

1. ✅ 设计文档（本文档）
2. ✅ 等价性机器验证（`verify_equiv.py`，D-1）
3. ⬜ Golden 函数（§8.1）+ 算子实现（§3.3 伪代码 → 代码）
4. ⬜ L0 门槛测试（§8.3 全 14 项）→ 精度比对 → Stage 4（§1.6.3 参数级 A/B + §1.6.4 备选 B 裁决）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| kernel 对外接口参数不变 | ✅ | 工厂 5 个位置参数与源一致（仅新增带默认值的 `blk_elems`）；`main(o, do, delta)` 参数名/顺序与 `out_idx=[2]` 不变 |
| 只迁移本函数 | ✅ | `_gqa_bwd_kernel` / `_flashattn_bwd_postprocess_kernel` / wgmma 变体均未引入；`make_dq_layout` 未迁移 |
| 不改动 fwd 产物 | ✅ | 本任务只产出 `DESIGN.md`（+ `verify_equiv.py`）；未触碰 `multi_head_attention_kernel` 下任何文件 |
| 后端相关参数可适配 | ✅ | `threads`（本函数无）、`supported_archs`（wrapper 已改）、`compile_flags`（本函数无）均不引入；`blk=256` 固定几何改为 shape 驱动的 `blk_s`（工厂内部量） |
| host 侧不改输入张量内容 | ✅ | host 侧只做编译期常量推导；`do.contiguous()` 属 op 层既有行为（源同款） |
| 从源码推断规格（不问用户） | ✅ | §0.1 / §0.2 全部由源码 + Kernel 类 + op 层调用链解读得出；`delta` 输出 shape 已回填依据 |
| 不修改脚手架既有文件 | ✅ | 仅产出 `DESIGN.md` / `verify_equiv.py`（+ skill 要求的 `RETROSPECTIVE.md`） |
| 负向断言举证 | ✅ | §1.6.0/§1.6.1/§1.6.3 的每条弃选均附 `docs/`、`examples/`、pattern-library 条目路径与具体条款，或显式标注「未实测假设 + 估算依据」 |

---

## 11. 关键结论与 Stage 4 交接

1. **算法族选择**：单遍全 D 驻留 + 逐行独立一维 persistent 分核 + 三维扁平声明单次归约 + fp32 中转链 + 乘积原位复用
   （§1.6.0 调研结论；GM 流量与扫描遍数均为全候选最小）。
2. **数学等价优化**：采纳 **O1**（乘积原位复用，逐位恒等，机器验证 EQUIV_PASS）；以 **P1/P2** 机器验证固化
   「上抛+乘积逐位精确」与「归约结合序容差内」；以 **R1** 机器证据否决 input-dtype 乘积（§1.6.1）。
3. **向量化**：全部逐元素/归约计算点已向量化；仅保留动态边界、block 级任务映射、host 元数据三类标量（§1.6.2）。
4. **布局**：I/O 原生 BSHD + `(blk_s,1,D)` 三维声明 + 向量化轴 = 归约轴 `D` + 水平归约；备选 B（全 `H` 连续装载单元）
   仅对 `H·D ≤ 512` 可行（§1.6.3 / §1.6.4）。
5. **分核**：实查 20 AI Core / 40 纯 Vector 核；四档 manifest 逻辑核数 128–16 384 均 > 40 ⇒ persistent 核内串行
   （静态边界，负载不平衡 ≤1 任务）（§5.5）。
6. **Stage 4 第一优先级**（按影响排序）：① 反解归约链 UB 单价并尽量提档元素预算（410 → 205 轮，发射项减半）；
   ② 实测跨步 row-burst 的 MTE2 折损率（必要时切换 §1.6.4 备选 B）；③ 回填「设计估算 vs 实测」偏差行
   （§1.6.0 估算：full-fp16 档 ≈346–820 µs，其中流量地板 346 µs）。
