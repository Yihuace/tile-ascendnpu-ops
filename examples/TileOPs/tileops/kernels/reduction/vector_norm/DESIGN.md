# DESIGN.md — `VectorNormKernel`(TileLang-NPUIR)· 覆盖 `L1NormFwdOp` / `L2NormFwdOp` / `InfNormFwdOp`

- **project** = `vector_norm`,**kernel** = `VectorNormKernel`,**family** = **reduction**(本批第一个归约族算子)
- **覆盖的 manifest Op**:`L1NormFwdOp`(ord=1)、`L2NormFwdOp`(ord=2)、`InfNormFwdOp`(ord=∞)—— 三者**共用一个 kernel 类**,由 Python 层 `op_kind` 在 trace 期分支
- **迁移类型**:migration(harness 子模式)· **编程模式**:Developer
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/reduction.py`
- **`source_output_shape`**:`(M, N) → (M,)`(沿 `dim=-1` 归约;`M` 为未归约维之积,`N` 为归约维之积)
- **golden 独立性声明**:§9.1 的 golden 依据 §0.1 语义实现(`torch.linalg.vector_norm`),**未复刻 NPU 算法**
- **与同批逐元素族的根本差异**:本算子是**归约**,存在**归约轴**与**归约顺序**问题;而逐元素族(6 个)是 `(N,)→(N,)` 的逐点映射。故 §0.1 的语义分析必须包含**归约顺序/精度**维度。

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | 沿 `dim=-1` 逐行计算范数:$L_1 = \sum_j \lvert x_{ij}\rvert$;$L_2 = \sqrt{\sum_j x_{ij}^2}$;$L_\infty = \max_j \lvert x_{ij}\rvert$ | `VectorNormKernel` 的 `op_kind` 三分支;**逐条独立读源码确认**(见 §0.3) |
| I/O 契约 | $(M, N) \to (M,)$;输出 dtype 同输入;非 in-place | manifest `shape_rules`;`_ReduceOpBase` |
| **归约语义** | 归约轴 = 最后一维(`dim=-1`);**归约顺序会影响浮点结果**(求和结合律不成立) ⇒ 必须与源算子在**同一精度域**上累加 | §0.3 的源码(显式升 fp32 后 `reduce_sum`) |
| 精度策略 | **显式升 fp32 再累加**(`x_f32[i,j] = T.cast(x_local[i,j], "float32")`),累加与开方均在 fp32,**最后才落回 `dtype`** | 源码 §0.3 的第 2 步 |
| **$L_2$ 的开方** | `T.vsqrt(row_acc, row_acc)` 在 fp32 累加**之后**执行,结果再落 `dtype` | 源码 §0.3 的 l2 分支 |
| **$L_\infty$ 的 NaN 语义** | **`T.reduce_max` 不传播 NaN**(源码显式注明),故 NaN 行由 **Op 层 `_post_kernel` 打补丁** | 源码 §0.3 的 inf 分支注释 |
| dtype 语义 | `_ReduceOpBase` 的 dtype 处理;NPU 侧 kernel 按 `dtype` 泛化 | `ops/reduction/vector_norm.py:17/67/117` |
| host 侧语义 | `dim` 归一化 + reshape 到 `(M, N)`;`_post_kernel` 对 InfNorm 打 NaN 补丁 | `_ReduceOpBase` |

**完成标准核对**:已可写出与源算子数值一致的 golden —— `torch.linalg.vector_norm(x.float(), ord=..., dim=-1).to(dtype)`(§9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | `(M, N)`,`dtype ∈ {float16, bfloat16, float32}`;`M` = 未归约维之积,`N` = 归约维之积 |
| 输出 | `(M,)`,同 dtype |
| 归约轴 | **最后一维**(`dim=-1`) |
| 动态轴 | **无**;`M`/`N` 构造期确定 |
| 输出 shape 确认 | **归约掉 `dim=-1`**(已由源码与 manifest `shape_rules` 确认) |

### §0.3 第二问:实现算法(How)

**入口签名(两个函数,对应抽取结果)**:

1. `_vector_norm_kernel(M, N, op_kind, dtype)` —— **单 tile 版**:要求 $N$ 能整体放进 UB
2. `_vector_norm_kernel_tiled(M, N, op_kind, dtype, tile_n)` —— **分块版**:沿 $N$ 方向切 `tile_n`,用于 $N$ 过大

**Kernel 结构(单 tile 版,T.Kernel 一维)**:

```
T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _)
```

- **并行维 = $M$**(每核负责 `block_m` 行);**归约维 = $N$**(在一核内完成)
- 尾块行数 `real_m = T.min(block_m, M - pid_m * block_m)`

**五步数据流**:

| 步 | 操作 | API |
|---|---|---|
| 1 | GM → UB → fragment(尾块源/目的**同步切片**到 `(real_m, N)`) | `T.copy` |
| 2 | 升 fp32 累加域 | `T.Parallel` + `T.cast` |
| 3 | 逐行变换 + 归约(按 `op_kind` 三分支) | `T.vabs`/`T.vmul`/`T.reduce_sum`/`T.reduce_max`/`T.vsqrt` |
| 4 | 落回原 dtype | `T.Parallel` + `T.cast` |
| 5 | UB → GM(尾块目的切片到 `real_m`) | `T.copy` |

**归约分支(Python 层 `op_kind` 在 trace 期解析,故只发射一个分支)**:

| `op_kind` | 算子序列 |
|---|---|
| `l1` | $\lvert x\rvert$ → `reduce_sum(dim=1)` |
| `l2` | $x^2$ → `reduce_sum(dim=1)` → `sqrt` |
| `inf` | $\lvert x\rvert$ → `reduce_max(dim=1)` |

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| G1 | **register_copy 策略** | 合并访存 | 整块 `T.copy` 进 fragment → 计算 → 整块写回 | 大寄存器堆 |
| G2 | **沿 $M$ 分块并行** | 摊薄调度开销 | `block_m` 行/核;`ceildiv(M, block_m)` 个核 | 多核并行 |
| G3 | **归约维整体驻留** | 避免二次读取 | $N$ 全部进 UB/fragment,一次遍历完成归约 | UB 容量 |
| G4 | **分块变体(`tile_n`)** | 支持大 $N$ | 沿 $N$ 切块,块间累加(在线式) | UB 容量受限时的退路 |
| G5 | autotune | 找最优执行参数 | `autotune_configs` | — |

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| 三种范数的数学语义、`(M,N)→(M,)` 契约、**fp32 累加域** | 语义 | **保留** | 语义层不可妥协;累加精度直接决定结果 |
| G1 fragment 中转 | 优化 | **等价替换** | 映射表「register 累加 → L0C/UB 中间缓冲」→ UB staging + fragment |
| G2 沿 $M$ 并行 | 优化 | **重新设计** | 源的 block/thread 两级 → 一维 `T.Kernel` 按 $M$ 分块 |
| G3 归约维驻留 | 优化 | **等价替换** | → UB/fragment 驻留整行 |
| G4 分块变体 | 优化 | **等价替换** | → `tile_n` 分块 + 块间累加(本仓保留两个函数即为此) |
| G5 autotune | 优化 | **舍弃** | K8:NPU 无 autotune |
| **归约算子映射**($\sum$/$\max$) | 优化 | **等价替换** | → `T.reduce_sum` / `T.reduce_max`(fragment 上的行归约) |

### §0.6 NPU 算法重设计

#### 重设计项 1 —— 并行结构(G2)

- **源方案**:CUDA grid × block × thread;归约在 block 内由多线程协同完成。
- **NPU 新算法**:一维 `T.Kernel(T.ceildiv(M, block_m), is_npu=True)`,**每核独立完成 `block_m` 行的整条归约**(归约维不跨核)。
- **语义保持论证**:归约仅在**单核内**完成 ⇒ 不引入跨核的部分和与二次归约,**归约顺序与源侧的"一行一核内累加"一致**(同样升 fp32 累加)。这是本设计选择"归约维不切分到核间"的**语义动机**,不只是性能取舍。

> ⚠️ **这是本算子最重要的语义决策**:若把归约维切分到多核再二次合并,会改变加法结合顺序 ⇒ 结果与源算子产生**浮点级差异**。设计**明确不这么做**(除 `tile_n` 分块变体在**核内**逐块累加,顺序仍是确定的串行)。

#### 重设计项 2 —— 归约原语映射

| 源 | NPU | 备注 |
|---|---|---|
| `sum` 沿最后一维 | `T.reduce_sum(x_f32, row_acc, dim=1)` | 作用在 **fragment** 上 |
| `max` 沿最后一维 | `T.reduce_max(x_f32, row_acc, dim=1)` | 同上 |
| `sqrt` | `T.vsqrt(row_acc, row_acc)` | fp32 域,累加之后 |

**语义保持论证**:累加/取极值都在 fp32 域、在单核内、按行完成,与源方案的精度路径一致。

#### 重设计项 3 —— **$L_\infty$ 的 NaN 语义缺失(本设计的关键发现)**

**源码显式注明**:

> `# NOTE: T.reduce_max does not propagate NaN; NaN rows are patched at the Op layer (InfNormFwdOp._post_kernel), exactly as in the GPU implementation.`

⇒ **`T.reduce_max` 不传播 NaN**。若输入某行含 NaN,归约结果**不是 NaN**(取决于实现的比较语义),与 $L_\infty$ 的数学期望(`max` 遇 NaN 应为 NaN)不符。

**处置(沿用源仓口径)**:GPU 侧同样如此处理 —— 在 **Op 层 `_post_kernel`** 对 InfNorm 打 NaN 补丁。本迁移**保持同一分工**:
- **kernel 层**只做 `reduce_max`,**不**承担 NaN 语义;
- **Op 层**补丁保持不变(不在本 Agent 的可改范围内,且源侧即如此)。

> ⚠️ **须在 §9.2 的 L1 显式验证**:含 NaN 的行在 **Op 层补丁之后**结果正确。若 kernel 层单独测(不经 Op 层),InfNorm 的 NaN 行**预期不通过**——这是**已知且与源一致的行为**,不是缺陷。

#### 重设计项 4 —— 尾块处理

`real_m = T.min(block_m, M - pid_m * block_m)`;源/目的 `T.copy` **同步切片**到 `(real_m, N)` / `real_m`,避免越界读与形状不一致。
**语义保持论证**:切片只改变参与归约的行数,不改变任何一行的归约路径。

#### 重设计项 5 —— 缓冲分配 API 的**已实测裁决**(原待核实项)

既有 NPU 实现使用 **`T.alloc_shared((block_m, N), dtype)`** 做 GM 暂存与输出暂存,其注释称 *"Developer mode: alloc_shared -> UB"*,且该实现**测试通过**。这与逐元素族的教训(实测 `T.alloc_shared` 在 npuir 上降到 **L1**,导致 store 编译失败)**存在张力**。

> **Stage 3 实测裁决(已解决)**
>
> 本设计初版把该差异登记为待核实项。Stage 3 的实测结论是:
>
> **`T.alloc_shared` 的落地层级取决于 `TILELANG_ASCEND_MODE` 是否以「环境变量」形式生效。**
>
> - **环境变量未生效时**:`T.alloc_shared` 落到 **`#hivm.address_space<cbuf>`(即 L1)**,且 kernel 被降到 **Cube 路径**(IR 中 `hivm.module_core_type<AIC>` / `mix_mode = "aic"`),`cbuf → fragment` 的 staging copy 编不过:
>   ```
>   error: 'hivm.hir.copy' op Unsupported copy from cbuf to zero!
>   ```
> - **环境变量生效时**(`TILELANG_ASCEND_MODE=Developer`):**同一份源码编译并运行成功**。
>
> **关键陷阱**:既有实现用 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` 设置它;而本批逐元素族的交付件写的是**普通 python 变量** `ASCEND_MODE = "Developer"` —— **后者不设置环境变量**。逐元素族(纯 UB、无 fragment 归约)在该差异下仍能通过,但**归约族会直接编译失败**。
>
> ⇒ **结论**:本设计**保留 `T.alloc_shared` 的既有形态**(它在 Developer 环境变量生效时是正确的),但**交付件必须显式导出该环境变量**;`T.alloc_shared` 与 `T.alloc_ub` 的差异在本算子上**被环境模式主导**,不能只按 API 名判断层级。

### §0.7 迁移决策摘要

- **三问结论**:语义 = 三范数沿 `dim=-1`、`(M,N)→(M,)`、**fp32 累加域**、$L_\infty$ 的 NaN 由 Op 层补丁;算法 = $M$ 分块一维核 + 归约维单核完成 + 单 tile/分块两变体;优化手段 = 5 项。
- **耦合性判定统计**:**保留 1**、**等价替换 4**、**重新设计 1**、**舍弃 1**。
- **重设计项清单**:① 并行结构(**归约维不跨核** —— 语义动机);② 归约原语映射;③ **$L_\infty$ 的 NaN 语义分工**;④ 尾块切片;⑤ **缓冲 API 待核实**。
- **源优化意图承接**:G1 → UB staging + fragment;G2 → `block_m` 一维划分;G3 → 整行驻留;G4 → `tile_n` 分块变体(两函数均保留);G5 → 舍弃。**无静默丢弃。**

---

## §1 概述

### §1.1 算子概述

向量范数(沿最后一维):$L_1$ / $L_2$ / $L_\infty$。用于归一化、相似度、梯度裁剪等。
**迁移后 NPU 侧算法**:一维核按 $M$ 分块,**每核独立完成 `block_m` 行的整条归约**(归约维不跨核)+ fp32 累加域。

### §1.2 计算语义

- 输入:$x \in \mathbb{R}^{M\times N}$
- 输出:$y_i = \lVert x_{i,:}\rVert_p$,$p\in\{1,2,\infty\}$,形状 `(M,)`
- **归约轴 = 最后一维**;累加在 fp32 域

### §1.3 算子特征分析

| 维度 | 判定 |
|---|---|
| 计算类型 | **纯 Vector + 归约**(无 Cube);UB + fragment |
| 复杂度级别 | **单步归约**(每行一次遍历:搬运 → 变换 → 行归约 → 落回) |
| 动态 shape | **无** |
| host 预处理 | `dim` 归一化 + reshape 到 `(M, N)`;InfNorm 的 `_post_kernel` NaN 补丁 |

### §1.4 算法描述(依据 Phase R 调研选定算法)

对 $M$ 行按 `block_m` 分块,每核:

1. `T.copy` 把 `block_m × N`(尾块 `real_m × N`)从 GM 搬入 UB,再进 fragment;
2. 升 fp32 到累加域;
3. **按 `op_kind` 逐行归约**(见 §0.3 三分支表);
4. 落回原 dtype(每行一个标量);
5. `T.copy` 写回 `(M,)`(尾块切片)。

⇒ **归约在单核内按行完成**,与源侧同一精度路径。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R —— 归约族需完整调研)

> 深度分级:本算子含**归约结构**(存在"切分归约维再合并"这一可选结构),故按**完整调研**处理。

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **无可更省的等价形式** | 候选:① **直接求和/取极值(采纳)** —— $L_1$/$L_\infty$ 各 1 条归约、$L_2$ 加 1 条 `vmul` 与 1 条 `vsqrt`,已是 roofline 下界;② 用 `logsumexp` 之类统一三种范数 —— 会引入 `exp/log`,精度与代价都更差,且 $L_\infty$ 无法如此表达;③ 用 `x·x` 的 `dot` 指令替代 `vmul+reduce_sum` —— 本仓 `T.dot` 面向矩阵乘语义,不适用逐行自点积。⇒ 基线胜出。 |
| **R2 在线算法** | **有,且本仓已实现为第二个函数** | 结构依据:归约语义允许**分块累加**(沿 $N$ 切 `tile_n`,块间累加部分和),这正是 `_vector_norm_kernel_tiled` 的用途。**故本算子的 R2 结论与逐元素族(无在线变体)不同**:分块版是**必要的存在**,用于 $N$ 过大时 UB 放不下整行。 |
| **R3 复杂度四口径** | 基线:**FLOPs = 2·M·N**($L_2$ 为 $2MN+M$)、**访存 = (M·N + M)·elem_bytes**(读 1 + 写 M)、**扫描遍数 = 1**(单 tile 版)/ $N/\text{tile\_n}$ 遍(分块版)、**中间缓冲 = 0**(UB/fragment 驻留,不落 GM) | 与 manifest roofline 逐字一致(`flops: 2*M*N`,`bytes: (M*N + M)*elem_bytes`) |
| **R4 硬件亲和** | 基线亲和:纯 Vector + fragment 归约;`T.reduce_sum`/`T.reduce_max`/`T.vsqrt` 均为 npui 原语;归约维驻留受 **UB 容量**约束(⇒ 必须有 `tile_n` 退路);$M$ 方向的分块粒度 `block_m` 需与 UB 预算匹配 | 逐候选评估;③ 因语义不符被拒 |

**调研结论**:选定 = **$M$ 分块 + 归约维单核完成 + 单 tile/分块两变体**。

**设计期 roofline 估算下界(D-2)**

```
流量项 = (M·N + M)·elem_bytes / HBM带宽
发射项 = M · N · (每元素 op 数) / 向量吞吐      ← 归约的每元素 op 数很小(1~2)
容量项 = block_m × N × elem_bytes × (缓冲数) / UB容量   ← 归约族特有:容量项可能成为**约束**(不是下界主导,而是可行性约束)
估算下界 = max(流量项, 发射项, 容量项)
```

**设计期结论**:本算子的关键风险**不在带宽**而在 **UB 容量可行性**(§4.3 详算);若 $\text{block\_m}\times N$ 超出 UB,必须走 `tile_n` 分块变体。故 §5.3 的分核判定必须同时给**容量约束下的分核方案**。

#### §1.6.1 数学等价优化(公式级)

**结论:公式层无优化空间(显式结论 + 逐类依据)**

| 优化类别 | 结论 | 依据 |
|---|---|---|
| 恒等变形降代价 | 不适用(已逐候选否决) | R1 的三候选均不更省或语义不符 |
| 稳定化变形 | **部分适用:累加域提升即是稳定化** | 源与 NPU 侧**均已**显式升 fp32 累加(§0.3 第 2 步)—— 该稳定化**已由源算子承担**,不需另加 |
| **归约顺序优化** | **不适用,且刻意不引入** | 归约顺序影响浮点结果;本设计**明确不在核间切分归约维**(§0.6 重设计项 1),正是为了不改结合顺序。这是"**刻意不优化**"的语义决策 |
| 公共子表达式消除 | 已应用 | `x_f32` 复用为变换与归约的输入;$L_2$ 的 `vmul(x,x,x)` **原位** |
| 访存量削减 | 已最优 | 读 1 遍 + 写 M = roofline 下界 |
| **归约结构优化** | **可选项已存在但仅按需启用** | `tile_n` 分块版是"归约结构"的另一形态;**仅在 $N$ 超出 UB 容量时启用**(不是性能优化,而是**可行性退路**) |

> 因**无公式层采纳项**,且归约结构的分块属**可行性退路**而非等价优化,`verify_equiv.py` **豁免**,不生成。

#### §1.6.2 向量化替代分析(循环 / 标量消除)

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | 升 fp32 逐元素 cast | 逐元素循环 | **已向量化替代** | `for i, j in T.Parallel(block_m, N): x_f32[i,j] = T.cast(...)` —— `T.Parallel` 展开为向量 lane |
| V2 | 逐元素变换($\lvert x\rvert$ / $x^2$) | 逐元素循环 | **已向量化替代** | `T.vabs` / `T.vmul` |
| V3 | **沿 $N$ 的行归约** | 归约循环 | **已向量化替代** | `T.reduce_sum` / `T.reduce_max`(`dim=1`,fragment 上) |
| V4 | 落回 dtype 的逐元素 cast | 逐元素循环 | **已向量化替代** | `T.Parallel` + `T.cast` |
| V5 | `real_m = T.min(block_m, M - pid_m*block_m)` | block 级边界 | **保留(不可替代)** | block 级索引 / 任务映射计算 |
| V6 | `T.ceildiv(M, block_m)` | host 元数据 | **保留(不可替代)** | host 侧 shape 元数据 |

**结论**:逐元素与归约热点(V1–V4)全部向量化;**无逐元素标量循环残留**。V5/V6 为 block/host 级标量。

#### §1.6.3 向量化轴与数据布局决策

**归约算子,轴的选择是本设计的核心决策之一**:

- **候选轴枚举**:**2 个** —— ① 沿 $M$(未归约维)分块;② 沿 $N$(归约维)分块。
- **评分**:

| 方案 | 归约顺序影响 | 容量 | 判据 |
|---|---|---|---|
| ① **沿 $M$ 分块(主选)** | **无影响**(归约维在核内完整完成) | 受 $N$ 约束 | **语义最优** + 满足容量(manifest 的 N 在 UB 内) |
| ② 沿 $N$ 分块再合并 | **改变加法结合顺序** ⇒ 与源产生浮点差异 | 容量友好 | **仅作退路**(`tile_n` 变体),且**只在核内串行累加**(不跨核),故顺序仍确定 |

- **向量宽度对齐**:归约轴 $N$ 的向量化由 `T.reduce_*` 承担;`block_m × N` 的 UB 占用须 32B 对齐(§4.3 详算)。
- **决策落地**:主选 = 沿 $M$ 分块 + 归约维驻留,是 §3.3 与 §5 的唯一依据;`tile_n` 变体仅在容量不足时启用。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无**(纯 Vector + fragment 归约) |
| 是否需要手工同步 | **否** —— 块内为"搬入 → cast → 变换 → 归约 → cast → 搬出"的线性依赖,自动同步可正确处理 |
| 是否需要控制 L1/L0 层级 | **否** |
| 是否需要跨核通信 | **否**(**归约维不跨核**,块间完全独立 —— 这是语义要求,见 §0.6 重设计项 1) |
| 与既有实现一致性 | 本仓 `kernels/reduction/{vector_norm,logsumexp}` 同为 Developer |

---

## §3 API 映射设计

### §3.1 公式拆解

| 步 | 计算 | 对应公式 |
|---|---|---|
| 1 | 取 `block_m` 行($M$ 方向分块) | 输出分块 |
| 2 | 升 fp32 | 累加域 |
| 3 | 逐行归约(按 `op_kind`) | $L_1/L_2/L_\infty$ |
| 4 | 落回 dtype + 写回 | 输出 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| grid 划分(沿 $M$) | `T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _)` | 既有实现 `_vector_norm_kernel.py:72` |
| **GM 暂存 / 输出暂存** | `T.alloc_shared((block_m, N), dtype)` / `T.alloc_shared((block_m,), dtype)` | 既有实现 `:74`/`:78`;**落地层级见 §0.6 重设计项 5(待核实项)** |
| **归约中间量** | `T.alloc_fragment((block_m, N), dtype)` / `(block_m, N), "float32"`;行累加器 `T.alloc_fragment((block_m, 1), "float32")` | 既有实现 `:75-77`;**`T.reduce_*` 要求操作数为 fragment** |
| GM → UB → fragment | `T.copy(x[pid_m*block_m : pid_m*block_m+real_m, 0:N], x_ub[0:real_m, 0:N])`;`T.copy(x_ub, x_local)` | 既有实现 `:86-90`;尾块源/目的**同步切片** |
| **升 fp32** | `for i, j in T.Parallel(block_m, N): x_f32[i, j] = T.cast(x_local[i, j], "float32")` | 既有实现 `:93-94`;**逐元素 cast 由 `T.Parallel` 向量化** |
| 逐元素变换 | `T.vabs(x_f32, x_f32)`($L_1$/$L_\infty$)、`T.vmul(x_f32, x_f32, x_f32)`($L_2$,**原位**) | 既有实现 `:101`/`:105`/`:113` |
| **行归约** | **`T.reduce_sum(x_f32, row_acc, dim=1)`** / **`T.reduce_max(x_f32, row_acc, dim=1)`** | 既有实现 `:102`/`:106`/`:114` |
| 开方 | `T.vsqrt(row_acc, row_acc)`($L_2$,fp32 域) | 既有实现 `:107` |
| 落回 dtype | `for i in T.Parallel(block_m): out_ub[i] = T.cast(row_acc[i, 0], dtype)` | 既有实现 `:117-118` |
| UB → GM | `T.copy(out_ub[0:real_m], y[pid_m*block_m : pid_m*block_m+real_m])` | 既有实现 `:121-124`;尾块目的切片 |
| 尾块 | `real_m = T.min(block_m, M - pid_m * block_m)` | 既有实现 `:81` |

> ⚠️ **三条须注意的点**:
> 1. **`T.reduce_*` 的操作数必须是 fragment**(不是 UB)—— 故中间量用 `T.alloc_fragment`。
> 2. **`T.reduce_max` 不传播 NaN** —— 源码显式注明,NaN 语义由 **Op 层**补丁承担(§0.6 重设计项 3)。
> 3. **`op_kind` 在 trace 期解析**(Python 层 `if/elif/else`),故 kernel 内**只发射一个分支**,无运行期分支开销。

### §3.3 伪代码

```python
def _vector_norm_kernel(M, N, op_kind, dtype):          # 单 tile 版:N 整体驻留
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], y: T.Tensor[(M,), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                x_ub    = T.alloc_shared((block_m, N), dtype)      # GM staging
                x_local = T.alloc_fragment((block_m, N), dtype)
                x_f32   = T.alloc_fragment((block_m, N), "float32")
                row_acc = T.alloc_fragment((block_m, 1), "float32")
                out_ub  = T.alloc_shared((block_m,), dtype)

                real_m = T.min(block_m, M - pid_m * block_m)       # V5

                T.copy(x[pid_m * block_m : pid_m * block_m + real_m, 0:N],
                       x_ub[0:real_m, 0:N])
                T.copy(x_ub, x_local)

                for i, j in T.Parallel(block_m, N):                # V1
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                if op_kind == "l1":                                # trace 期解析
                    T.vabs(x_f32, x_f32)                           # V2
                    T.reduce_sum(x_f32, row_acc, dim=1)            # V3
                elif op_kind == "l2":
                    T.vmul(x_f32, x_f32, x_f32)                    # V2
                    T.reduce_sum(x_f32, row_acc, dim=1)            # V3
                    T.vsqrt(row_acc, row_acc)
                else:  # "inf"
                    T.vabs(x_f32, x_f32)                           # V2
                    T.reduce_max(x_f32, row_acc, dim=1)            # V3
                    # NOTE: reduce_max does not propagate NaN -- the Op layer
                    # patches NaN rows (InfNormFwdOp._post_kernel), as on GPU.

                for i in T.Parallel(block_m):                      # V4
                    out_ub[i] = T.cast(row_acc[i, 0], dtype)

                T.copy(out_ub[0:real_m],
                       y[pid_m * block_m : pid_m * block_m + real_m])
        return main
    return _func
```

**分块版 `_vector_norm_kernel_tiled(M, N, op_kind, dtype, tile_n)`**:同一结构,但沿 $N$ 切 `tile_n`,**在核内串行累加各块的部分和**(不跨核)。**仅在 $\text{block\_m}=1$ 也放不下整行时启用。**

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| **既有 vector_norm NPU 实现(含 capacity 公式与三分支)** | `examples/TileOPs/tileops/kernels/reduction/vector_norm/_vector_norm_kernel/_vector_norm_kernel.py` |
| wrapper 的 capacity 检查与 `_ub_bytes_for` | `examples/TileOPs/tileops/kernels/reduction/vector_norm/vector_norm.py:152/161` |
| 同族归约实现(尾块 `real_m` 形态) | `examples/TileOPs/tileops/kernels/reduction/logsumexp/…` |
| 已走完整流程的逐元素族(对照:无归约) | `examples/{abs,ceil,floor,exp,cos}/*_fwd_kernel/` |
| `T.reduce_sum`/`T.reduce_max`/`T.vsqrt` 定义 | `tilelang/language/customize_npuir.py` |

---

## §4 数据规格与内存规划

### §4.1 数据规格

| 张量 | shape | dtype | 所在 | 分配 API |
|---|---|---|---|---|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | GM | `T.Tensor` 入参 |
| `y` | `(M,)` | 同 `x` | GM | `T.Tensor` 出参(`out_idx=[1]`) |
| `x_ub` | `(block_m, N)` | 同 `x` | UB(staging) | `T.alloc_shared`(**待核实**,§0.6 重设计项 5) |
| `x_local` | `(block_m, N)` | 同 `x` | fragment | `T.alloc_fragment` |
| `x_f32` | `(block_m, N)` | float32 | fragment | `T.alloc_fragment`(累加域) |
| `row_acc` | `(block_m, 1)` | float32 | fragment | `T.alloc_fragment`(行归约结果) |
| `out_ub` | `(block_m,)` | 同 `y` | UB(staging) | `T.alloc_shared`(同待核实项) |

> **层级 ↔ API 对照**:GM = `T.Tensor`;UB = **`T.alloc_ub`**(**本算子的既有实现用 `T.alloc_shared`,见待核实项**);fragment = `T.alloc_fragment`(**`T.reduce_*` 的操作数必须是 fragment**);L1 = `T.alloc_shared`/`T.alloc_L1`。
> ⚠️ **本算子是"归约 + fragment"结构**,与逐元素族的"纯 UB"结构不同:中间量必须在 fragment 上才能被 `T.reduce_*` 消费。

### §4.2 搬运路径

```
GM(x) --T.copy--> UB(x_ub) --T.copy--> fragment(x_local) --cast--> fragment(x_f32)
   --vabs/vmul--> --reduce_*--> fragment(row_acc) --cast--> UB(out_ub) --T.copy--> GM(y)
```

### §4.3 UB 容量预算(本算子可行性判定的核心)

**capacity 公式(来自 wrapper 的 docstring,原文)**:

```
block_m * N * (2*elem_bytes + 4)   bytes
```

—— 即**两块 dtype tile(`x_ub` + `x_local`)+ 一块 fp32 shadow(`x_f32`)**。

**最坏 workload 的容量核算**:`long-row-*` = `(M, N) = (64, 8192)`,`dtype = bfloat16`(elem_bytes = 2)

| buffer | 尺寸表达式 | bf16 峰值(block_m=1) |
|---|---|---|
| x_ub | block_m × N × elem_bytes | 16 KB |
| x_local | block_m × N × elem_bytes | 16 KB |
| x_f32 | block_m × N × 4 | 32 KB |

**合计**:64 KB。

> `row_acc`(`block_m×1×4` = 4 B)与 `out_ub`(`block_m×elem_bytes` = 2 B)为**每核标量级**,不进上表(block_m=1 时分别为 4 B / 2 B)。

UB 容量上限 192 KB,自动多缓冲膨胀系数 1.7 → 有效预算约 113 KB(来源:`pattern-library/constants.md` 的 `CONST-capacity-910B2C`)。

**结论**:最坏 workload(`long-row-*`,N=8192 / bf16)在 `block_m=1` 下占 64 KB,**在有效预算内**。

> **逐 workload 的可行性与 `block_m` 上限见 §5.3 ③** —— 该分析属**分核策略**(`block_m` 的选择依据),故置于该节,不在本节重复列表。

### §4.4 中间缓冲峰值

**0** —— 无跨核部分和、无 workspace。**这是 §0.6 重设计项 1(归约维不跨核)带来的直接好处**:不存在"核间合并缓冲区"。

---

## §5 Tiling 策略

### §5.1 Block 划分与 Tile Shape

| 项 | 值 |
|---|---|
| 划分维度 | **未归约维 $M$**(沿 $M$ 分块) |
| Tile Shape | `block_m` 行 × $N$ 列($N$ 为整行) |
| Block 数 | `T.ceildiv(M, block_m)` |
| 每 block | `block_m` 行(尾块 `real_m` 行),**归约维 $N$ 不切分** |

### §5.2 非整除处理策略

**Kernel 内运行期切片**:`real_m = T.min(block_m, M - pid_m * block_m)`。源/目的 `T.copy` **同步切片**到 `(real_m, N)` / `real_m`,避免越界读与形状不一致。

### §5.3 分核策略(三要素)

**① 逻辑核数计算**

```
逻辑核数 = T.ceildiv(M, block_m)
```

| workload | M | block_m 建议 | 逻辑核数 |
|---|---|---|---|
| smoke-small (32, 512) | 32 | 1 | 32 |
| hidden-state (2048, 4096) | 2048 | 3(容量上限) | 683 |
| long-row (64, 8192) | 64 | **1**(容量上限) | 64 |
| 3d-multidim (4,128,512) dim=[0,2] | 512 | 1 | 512 |

**② 物理核数依据(实查,非假设)**

```
torch.npu.get_device_properties(0) → device: Ascend910B2C
    cube_core_num   = 24
    vector_core_num = 48      ← 本算子为纯 Vector,取此值
```

> **查询记录**:经 `torch_npu` 的 `get_device_properties(0)` 实查,`vector_core_num = 48`、`cube_core_num = 24`、设备名 `Ascend910B2C`。未使用假设值。

**③ 规模判定与分核方案(含**容量约束**——归约族特有)**

| workload | 逻辑核数 | 物理核数 | 倍数 | 容量是否允许放大 `block_m` | 判定 | 方案 |
|---|---|---|---|---|---|---|
| smoke-small | 32 | 48 | **0.67×**(核数少于物理核) | 允许(`block_m` ≤ 18) | **小规模** | 可增大 `block_m` 以复用核 |
| hidden-state | 683 | 48 | 14.2× | 允许(≤3) | 中等规模 | **无需适配** |
| long-row | 64 | 48 | 1.33× | **不允许**(容量上限 =1) | **中等规模但容量受限** | **无需核内 `T.serial`**;`block_m` 锁死为 1 |
| 3d-multidim | 512 | 48 | 10.7× | 允许 | 中等规模 | 无需适配 |

**结论:三选一命中「无需适配依据」**,但**两点须显式说明**(归约族特有):
1. `smoke-small` 的逻辑核数(32)**少于**物理核数(48)—— 属"核数不足"而非"核数过剩",**不需要** `T.serial` 压缩核数;可通过增大 `block_m` 提升每核工作量,但该 workload 本身极小(6 KB),不是性能关键。
2. `long-row` 的 `block_m` 被 **UB 容量锁死为 1**(64 KB/行)—— 这是**容量约束而非分核策略选择**;此时 $M$ 方向仍有 64 个逻辑核,负载均衡可接受。

**`block_m` 的容量上限(逐 workload,依据 §4.3 的 `block_m×N×(2·elem_bytes+4)` 公式与 113 KB 有效预算)**:

| workload | (M, N) | dtype | `block_m=1` 的 UB 占用 | `block_m` 上限 |
|---|---|---|---|---|
| smoke-small | (32, 512) | fp32 | 512×12 = 6 KB | 18 |
| hidden-state-* | (2048, 4096) | fp16/bf16 | 4096×8 = 32 KB | 3 |
| long-row-* | (64, 8192) | bf16 | 8192×8 = 64 KB | **1** |
| 3d-multidim | (4,128,512) dim=[0,2] → (512, 128) | fp16 | 128×8 = 1 KB | 100+(受 $M$ 限制) |

⇒ **全部 workload 在 `block_m=1` 下均可容纳**(最紧的 `long-row-*` 占 64 KB ≤ 113 KB)⇒ **单 tile 版足够,`tile_n` 分块版在 manifest 范围内不需要**;但 `block_m` 必须**按 $(N, \text{dtype})$ 选择**(不能固定)—— wrapper 的 `_ub_bytes_for(block_m)` 正是为此。**当 $N$ 大到 `block_m=1` 也放不下时**才启用 `tile_n` 变体(该分支在 manifest 内**不会被触发**,但必须保留:源侧即有两个函数)。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| `T.Parallel` | **使用 × 2** | ① 升 fp32 的逐元素 cast ② 落回 dtype 的逐元素 cast |
| **`T.reduce_sum` / `T.reduce_max`** | **使用 × 1** | 沿 `dim=1` 的行归约(fragment 上)—— **归约族的核心结构** |
| `T.serial` | **不使用** | 单 tile 版每核仅 1 行块;分块版的 `tile_n` 循环由 `tile_n` 参数化的**块内串行**表达(不在 manifest 范围内触发) |
| `T.Pipelined` | **不使用** | 无多阶段可重叠 |
| `T.Persistent` | **不使用** | 无核常驻需求 |

> 一致性核对:§1.6.2 判定 V1–V4 已向量化(其中 V3 的行归约由 `T.reduce_*` 承载);V5/V6 为 block/host 级标量。

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入 |
| 手工同步点 | **无** |
| 依赖性质 | 线性:`GM→UB`(copy)→`UB→fragment`(copy)→`fragment` 内 cast/变换/归约→`fragment→UB`(cast)→`UB→GM`(copy) |
| **跨核同步** | **不需要** —— **归约维不跨核**(§0.6 重设计项 1),无部分和交换、无核间合并 |

---

## §8 CV 融合设计

**n/a** —— 纯 Vector + fragment 归约,不涉及 Cube 计算。

---

## §9 验证方案

### §9.1 Golden 函数草案(独立于 NPU 算法)

```python
_ORD = {"l1": 1, "l2": 2, "inf": float("inf")}

def golden_vector_norm(x: torch.Tensor, op_kind: str) -> torch.Tensor:
    """PyTorch 参考实现 —— 按 §0.1 语义。

    fp32 域归约后回落原 dtype,与源仓参考实现口径一致。只依据语义实现,
    不复刻 NPU 侧的 M 分块 / fragment 形态 —— 保证验证独立性。
    """
    x_cpu = x.detach().cpu()
    return torch.linalg.vector_norm(
        x_cpu.to(torch.float32), ord=_ORD[op_kind], dim=-1
    ).to(x_cpu.dtype)
```

### §9.2 L0 门槛测试计划

| 用例 | (M, N) | dtype | 容差(atol/rtol) |
|---|---|---|---|
| smoke-small | (32, 512) | float32 / float16 / bfloat16 | 逐 dtype 沿用源仓 |
| hidden-state | (2048, 4096) | float16 / bfloat16 | 同上 |
| long-row | (64, 8192) | bfloat16 | 同上 |

容差逐字沿用源仓 `_get_tolerances`。

**L1(阻塞层)必须包含的用例 —— 逐条对应 §0.1 声明为"必须保持"的语义**:

| 类别 | 用例 | 为什么放在**阻塞层** |
|---|---|---|
| **三个 `op_kind` 各自覆盖** | `l1` / `l2` / `inf` × 三个 dtype | **一个 kernel 三个语义分支**,任一分支错都不能被其它分支掩盖 |
| manifest 的 4 个 workload | `(32,512)` / `(2048,4096)` / `(64,8192)` / 3D `(4,128,512) dim=[0,2]` | 覆盖大 $N$(64 KB/行,容量最紧)与多维 `dim` |
| **非整除 $M$** | `M % block_m != 0`(如 `block_m=3` 时 M=2048+1) | **尾块 `real_m` 路径的唯一覆盖** |
| $N$ 很小 | `N = 1`(单元素行)、`N = 2` | 归约退化为恒等/双元素,边界对齐 |
| $M$ 很小 | `M = 1` | 单行输入 |
| **$L_\infty$ 的 NaN 语义**(经 Op 层) | 含 NaN 的行,`op_kind="inf"` | §0.1 与 §0.6 重设计项 3 已声明:`reduce_max` **不传播 NaN**,由 Op 层补丁承担 —— **必须验证补丁链路生效** |
| **$L_1$/$L_2$ 的 NaN 传播** | 含 NaN 的行,`op_kind in {l1, l2}` | 与 $L_\infty$ **不同**:求和/平方和会自动传播 NaN,**必须分别验证**,不能假设三个分支一致 |
| $L_2$ 的开方域 | 全零行($\sqrt{0}$)、含极大值的行(平方可能溢出 fp32) | `vmul` + `sqrt` 的边界 |
| **3D `dim=[0,2]`** | `(4,128,512)` with `dim=[0,2]` | 非最后维的归约经 host 侧 reshape 实现;**验证 reshape 语义** |

> ⚠️ 边界语义与三分支既为"必须保持",就应落在 **L1(阻塞层)**。

### §9.3 与既有实现的关系(需后续 Stage 裁决)

> 本算子已有手工 NPU 实现(`_vector_norm_kernel/_vector_norm_kernel.py`,7496 B),其结构与本设计**一致**($M$ 分块 + fragment 归约 + `real_m` 尾块 + 三分支),且其 docstring 已记录 capacity 公式。
>
> 差异仅在:① 本设计把 §9.2 的**三分支分别覆盖**与 **NaN 分支差异**明确列为 L1 阻塞用例;② 本设计把 **`alloc_shared` 的落地层级**登记为待核实项(§0.6 重设计项 5 / §10 R5),而既有实现以注释断言其"映射到 UB"。
>
> **本设计不修改既有文件**;是否以本设计为准由 Stage 2 裁决。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| **R1** | **`T.reduce_max` 不传播 NaN** ⇒ $L_\infty$ 的 NaN 语义**不在 kernel 内** | **高** | §0.6 重设计项 3 已声明;**分工沿用源仓**(kernel 只做 `reduce_max`,Op 层 `_post_kernel` 补丁);§9.2 要求**经 Op 层的完整链路**验证,并**分别**验证 $L_1$/$L_2$(它们会自动传播 NaN)—— **不得假设三分支行为一致** |
| **R2** | **归约顺序影响浮点结果** | **高** | §0.6 重设计项 1 已明确**不在核间切分归约维**,保证结合顺序与源一致;§1.6.1 把"归约顺序优化"列为**刻意不引入**项 |
| **R3** | **UB 容量约束可能锁死 `block_m=1`** | 中 | §4.3 逐 workload 核算,`long-row` 锁死为 1(64 KB/行);wrapper 已有 `_ub_bytes_for` 与超限报错路径;§5.3 ③ 已把"容量是否允许放大 `block_m`"列入分核判定 |
| R4 | **$N$ 超容量时需 `tile_n` 分块版** | 低(manifest 内不触发) | 源侧即有两个函数,本设计保留;**但 manifest 范围内不触发** ⇒ 该分支的实测覆盖有限,**须在复盘中如实标注** |
| **R5** | **`T.alloc_shared` 的落地层级待核实**:既有实现用它做 staging 且通过,与逐元素族"`alloc_shared`→L1 导致 store 失败"的教训存在张力 | 中 | §0.6 重设计项 5 已登记;**Stage 3 须实测确认给出裁决**(不得两边都写、含糊了事) |
| R6 | 非整除 $M$ 的尾块路径 | 中 | §9.2 已列 L1;源/目的 `T.copy` **必须同步切片**(否则越界或形状不一致) |
| R7 | 3D `dim=[0,2]` 的 reshape 语义 | 中 | 由 host 侧 `_ReduceOpBase` 承担;§9.2 已列 L1 验证 reshape 正确性 |
| R8 | 技术约束检测结论 | — | 三维 Kernel ❌;GPU 专用 API ❌;GEMM 非整除 ❌;L0C 溢出 ❌ → **无技术约束冲突** |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/vector_norm/vector_norm_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/vector_norm/vector_norm_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**(§1.6.1 公式层无采纳优化项;分块变体属**可行性退路**而非等价优化) |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/reduction/vector_norm/…` | ✅ Stage 0 已产出 |
| 5 | npuir kernel(2 函数)+ golden + 分层测试 | `examples/vector_norm/vector_norm_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |
