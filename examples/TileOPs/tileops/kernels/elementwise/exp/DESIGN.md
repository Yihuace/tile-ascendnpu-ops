# DESIGN.md — `ExpFwdOp`(TileLang-NPUIR)

- **project** = `exp`,**op_name** = `ExpFwdOp`,**family** = `elementwise`
- **迁移类型**:migration(harness 子模式)· **编程模式**:Developer
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`
- **`source_output_shape`**:与输入同形;kernel 层 1-D flat `(N,)`
- **golden 独立性声明**:§9.1 的 golden 依据 §0.1 语义实现,**未复刻 NPU 算法**
- **与同批 `ceil`/`floor` 的关键差异**:那两个算子因 **npui 无对应一元指令** 而被迫走 `vcast` round-trip + 饱和旁路 + NaN 传播(共 +5 op);**本算子 npui 有 `npuir_exp`**,是**直接映射**,无需任何补偿路径。
- **另一处差异**:`ExpFwdOp` 继承 **`UnaryOp`**(而非 `ceil`/`floor`/`abs` 的 `_IntIdentityUnaryOp`),**没有整型短路分支** —— 非 float dtype 由基类的 dtype 校验拒绝(见 `test_exp_rejects_non_float_dtype`)。

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | $y_i = e^{x_i}$,逐元素一元映射,无跨元素依赖 | `ExpFwdKernel.op_func(x) = T.exp(T.cast(x, "float32"))`(`kernels/elementwise.py:1982-1987`) |
| I/O 契约 | 输出 shape **与输入逐维相同**;dtype 相同(`same_as(input)`);非 in-place | manifest `shape_rules`;`_base.py` 用 `empty_like` |
| 规约语义 | **无规约** | 计算图无 reduce |
| dtype 语义 | kernel 家族 `FloatUnaryKernel` **float-only**;`op_func` 内**显式升 fp32 再算** | `:1982-1987` |
| **边界语义** | $+\infty \to +\infty$、$-\infty \to 0^+$、`NaN → NaN`;上溢(指数超过 dtype 范围)→ `+inf` | `T.exp`/`torch.exp` 算术语义 |
| **真值域差异(本算子特有)** | 计算在 **fp32 域**完成,**结果再落回 `out_dtype`**。故对 fp16/bf16 输出:$x$ 使 $e^x$ 超出该 dtype 范围时,结果为 `+inf`,**而 fp32 参照仍是有限值** —— 这是**源算子的既有语义**,须保持一致 | `op_func` 的 `T.cast(x, "float32")` + 结果隐式落回 `out_dtype` |
| dtype 支撑 | `ExpFwdOp(UnaryOp)` —— **无整型短路**;非 float dtype 由基类校验拒绝 | `ops/elementwise/math_unary.py:31`;`tests/ops/test_exp.py::test_exp_rejects_non_float_dtype` |
| host 侧语义 | 无预处理;仅 flatten(`N_total`)与 reshape 回原形 | `UnaryOp.__init__` / `_eager_forward` |

**完成标准核对**:已可写出与源算子数值一致的 golden —— `torch.exp(x.float()).to(dtype)`(§9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | 任意 rank 的 `float16 / bfloat16 / float32`,kernel 层 flatten 为 `(N_total,)` |
| 输出 | 与输入同 shape、同 dtype |
| 动态轴 | **无**;`N_total` 构造期确定 |

### §0.3 第二问:实现算法(How)

1. **入口签名**:`exp_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)`。
2. **Kernel 结构**:`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx`,`block_size = threads × num_per_thread = 2048`;一维 grid,每 block 处理 `block_size` 个连续元素。
3. **计算原语**:三策略(`direct` / `explicit_parallel` / `register_copy`(**默认**));逐元素体为 `y[idx] = T.exp(T.cast(x[idx], "float32"))` —— **先升 fp32 再算**,结果隐式落回 `out_dtype`。
4. **内存分配**:`register_copy` 用 `T.alloc_fragment((block_size,), dtype)` ×2;**无 SMEM**、无跨 block 共享。
5. **循环与调度**:`T.Parallel(threads, npt)`;无 `T.serial` / `T.Pipelined`。
6. **host 侧逻辑**:`UnaryOp.__init__` 构造 kernel;`_eager_forward` flatten → kernel → reshape。
7. **编译配置**:无 `pass_configs`;策略与 `threads/npt` 由 `default_config` 决定。

**数据流(GPU 视角)**:`GM → fragment(整块) → 逐元素 exp(fp32 域) → fragment → GM`。

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| G1 | **register_copy 策略**(默认) | 合并访存、提高有效带宽 | 整块 `T.copy` 进 fragment → 计算 → 整块写回 | 大寄存器堆 + coalesced 访存 |
| G2 | **`threads × npt` 二维并行** | 摊薄调度开销 | `T.Parallel(256, 8)`,每 block 2048 元素 | CUDA block/thread 两级并行 |
| G3 | **按 dtype 调 `npt`** | fp16/bf16 用 `npt=8` 达 128-bit 宽访存 | `_strategy_npt(strategy, dtype)` | LDG.128 |
| G4 | 三策略编译期可选 | 按 shape/dtype 选最优 | `direct` / `explicit_parallel` / `register_copy` | — |
| G5 | autotune(`threads`×`npt` 搜索) | 找最优执行参数 | `autotune_configs` / `autotune()` | — |
| **G6** | **fp32 域计算** | 精度对齐(低精度 dtype 上直接算 exp 会有更大的相对误差) | `T.exp(T.cast(x, "float32"))`;结果回落 `out_dtype` | CUDA `expf` |

> 本算子未使用:SMEM tiling、swizzle、warp shuffle、`T.gemm`、异步流水。

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| $e^x$ 数学语义、I/O 契约、边界/dtype 语义 | 语义 | **保留** | 语义层不可妥协 |
| G1 fragment 中转 | 优化 | **等价替换** | 映射表「register 累加 → L0C/UB 中间缓冲」→ UB tile + `T.copy` |
| G2 `threads × npt` 两级并行 | 优化 | **重新设计** | 「block × thread 两级并行 → 一维 block + 向量化元素并行」;本项目 `T.Kernel` 仅接受一维 block 数 |
| `T.Parallel(threads, npt)` | 优化 | **等价替换** | → 向量算子(`T.vexp`)覆盖整 tile |
| G3 按 dtype 调 `npt` | 优化 | **等价替换** | → 向量宽度对齐(fp16/bf16 ×8、fp32 ×4),由 `block_size` 承载 |
| G4 三策略编译期可选 | 优化 | **重新设计** | npuir Developer 下 UB 驻留是唯一自然形态,三策略收敛为单一实现 |
| G5 autotune | 优化 | **舍弃** | K8:NPU 无 autotune。理由:NPU 侧编译开销高且无 autotune 框架 |
| **G6 fp32 域计算** | 优化 | **等价替换** | npui **有 `npuir_exp`**,且支持 fp32 —— 直接在 fp32 UB 上 `T.vexp`,与源"升 fp32 再算"**逐位对应**;**无需重设计** |
| SMEM tiling / swizzle / warp shuffle | — | **n/a** | 源码未使用 |

### §0.6 NPU 算法重设计

#### 重设计项 1 —— 并行结构(G2)

- **源方案**:CUDA 三级并行(grid/block/thread)。
- **NPU 新算法**:一维 `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)`,每核处理连续 `block_size` 个元素;块内整块搬入 UB 后由**向量指令**处理;尾块用运行期 `tail_size = T.min(block_size, shape - offset)` 切片。
- **语义保持论证**:逐元素映射,**无规约、无重排、无累加顺序** → 并行度变化不改变任何元素的取值路径。

#### 重设计项 2 —— 策略收敛(G4)

- **源方案**:三套 kernel 体按 config 编译期选择。
- **NPU 新算法**:收敛为单一实现 = 源 `register_copy` 的等价替换形态。**G4 的性能意图承接**:`block_size` 仍是可配置量,三策略差异退化为"UB tile 大小"一个自由变量。
- **语义保持论证**:同重设计项 1。

#### 重设计项 3 —— **`exp` 的 API 映射:G6 无需重设计**(本算子与 `ceil`/`floor` 的**关键差异**)

`ceil`/`floor` 的核心难点是 **npui 缺一元指令**,被迫用 `vcast` round-trip 替代,并因此引入值域与 NaN 两类分歧(+5 op 补偿)。**本算子不存在该问题**:

| 项 | 实测结论 | 依据 |
|---|---|---|
| npui 是否有一元 exp | ✅ **有** —— `npuir_exp` 存在 | 读 `customize_npuir.py`(53 个 `npuir_*` 定义中含 `exp`/`exp2`) |
| 是否支持 fp32 | ✅ 支持 | 同上 |
| 与源 `T.exp(T.cast(x, "float32"))` 的对应 | **逐位对应**:升 fp32 → `vexp` → 回落 | 源 `op_func` 本身即"升 fp32 再算" |

⇒ **G6 判为「等价替换」而非「重新设计」**:NPU 侧只需 `widen → vexp → narrow`,与源算子在同一域上计算,**不引入任何新的值域/边界分歧**。

**语义保持论证**:四类输入**继承自 `vexp` 的指令语义**,与源 `T.exp` 的对应关系为:

| 输入类 | 期望(`torch.exp` 口径) | 承接方 |
|---|---|---|
| `NaN` | `NaN` | `vexp` 的指令语义(**Stage 3 的 L1 必须实测**,见 §9.2) |
| `+inf` | `+inf` | 同上 |
| `-inf` | `+0.0` | 同上 |
| 上溢($x$ 大) | `+inf`(在**计算域 fp32** 判定) | 同上 |
| 下溢($x$ 很负) | `+0.0` | 同上 |
| fp16/bf16 结果超范围 | `+inf`(**源算子既有语义**,§0.1) | `widen → vexp(fp32) → narrow` 的两步落差 |

> ⚠️ **本设计的唯一残余未验证点**:`vexp` 对 `NaN`/`±inf` 的**指令级传播行为**未在本项目实测过。
> `ceil`/`floor` 的教训(它们的 `vcast` 对 NaN **不传播**,实测得 `0`)说明**不能假设**指令会正确传播特殊值。
> 故 §9.2 把 NaN/±inf/上溢/下溢**全部列入 L1 阻塞层**;若 Stage 3 实测发现 `vexp` 不传播,则按 `[DESIGN_ERROR]` 返回并补 C2 式旁路(与 ceil/floor 同法)。

#### 重设计项 4 —— 尾块处理

运行期切片 `tail_size = T.min(block_size, shape - offset)`,覆盖 `N % block_size != 0`;**语义保持论证**:切片只改变每块元素个数,不改变映射与取值。

### §0.7 迁移决策摘要

- **三问结论**:语义 = 逐元素 $e^x$、shape/dtype 保持、无规约、**fp32 域计算**;算法 = 一维 grid + `threads×npt` 块内映射 + 三策略;优化手段 = 6 项(G1–G6)。
- **耦合性判定统计**:**保留 1**、**等价替换 4**(fragment 中转 / `T.Parallel` 形态 / npt 宽度对齐 / **G6 fp32 域计算**)、**重新设计 2**(两级并行 → 一维;三策略 → 单实现)、**舍弃 1**(autotune)、**n/a 1**。
- **重设计项清单**:① 并行结构;② 策略收敛。**(无 G6 类重设计 —— 与 ceil/floor 的关键差异)**
- **源优化意图承接**:G1 → UB tile 中转;G2 → `block_size` 一维划分;G3 → 向量宽度对齐;G4 → `block_size` 可配;G5 → **舍弃**;G6 → **等价替换**(`widen → vexp → narrow`)。**无静默丢弃。**

---

## §1 概述

### §1.1 算子概述

`ExpFwdOp`(exp)—— 逐元素自然指数 $y = e^x$。用于激活(SiLU/GELU 族)、softmax 前置、概率化等场景。
**迁移后 NPU 侧算法**:一维 grid + UB tile 中转 + **单条 `T.vexp`**(fp32 域)。

### §1.2 计算语义

- 输入:$x \in \mathbb{R}^{N}$,dtype ∈ {float16, bfloat16, float32}
- 输出:$y_i = e^{x_i}$,同 shape 同 dtype
- 无规约、无跨元素依赖

### §1.3 算子特征分析

| 维度 | 判定 |
|---|---|
| 计算类型 | **纯 Vector** → 仅需 UB,无 Cube / 无 L1/L0 |
| 复杂度级别 | **单步**(单次搬运 + 一条向量算子,无循环、无中间落 GM) |
| 动态 shape | **无** |
| host 预处理 | **无** |

### §1.4 算法描述(依据 Phase R 调研选定算法)

对 $N$ 个元素按 `block_size` 切块,每块:

1. `T.copy` 把该块从 GM 搬入 UB `x_ub`;
2. 需要时先升到 fp32(`x_ub → f_ub`;fp32 直通);
3. `T.vexp(f_ub, e_ub)` —— **在 fp32 域**求 $e^x$;
4. 结果回落 `out_dtype`(fp32 直通);
5. `T.copy` 写回 GM。

⇒ 与源算子**同域计算**(源即"升 fp32 再算"),**无补偿路径**。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R —— 轻量调研)

> 深度分级:单步纯逐元素 → **轻量调研**(四问各一行结论 + 依据)。
> 调研范围:同一数学语义($y=e^x$)的算法族 + 本仓 elementwise 族既有实现 + `references/algorithm-candidates.md` 的 R1/R2 参考表。

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **无可更省的等价形式**(基线即最优) | 候选:① `exp2(x · log2(e))` —— **2 op**,比基线(1 op)更贵,且多一次乘法引入额外舍入;② 多项式/分段逼近 —— 精度不满足 fp32 的 1e-5 容差,且 op 数远超 1;③ 位运算拼指数 —— 精度更差。**基线 `vexp` 为单条指令**,候选均不更省或不等价 → 基线胜出。 |
| **R2 在线算法** | **无** | 结构依据:输出每个元素只依赖对应输入元素,不存在 running 统计量或跨块依赖 → 不存在流式/单遍变体;本算法本身就是单遍。 |
| **R3 复杂度四口径** | 基线:**FLOPs = N**(1 op/elem,指数计 1)、**访存 = 2·N·elem_bytes**(读 1 遍 + 写 1 遍)、**扫描遍数 = 1**、**中间缓冲 = 0**(UB tile 驻留,不落 GM) | 与 manifest roofline 逐字一致(`flops: N`,`bytes: 2*N*elem_bytes`)—— **已在 roofline 下界**。 |
| **R4 硬件亲和** | 基线亲和:纯 Vector、只需 UB(峰值见 §4.3)、无 Cube/分形约束;`vexp` 是 npui 原生一元算子;向量宽度 fp16/bf16 ×8、fp32 ×4,`block_size=2048` 对三者均整除;边界静态(尾块用运行期 `tail_size`)。 | 基线**亲和**;无被拒候选(候选均在 R1 阶段已排除) |

**调研结论**:选定算法族 = **基线(UB tile 中转 + 单条 `T.vexp`)**,无更优替代。

**设计期 roofline 估算下界(D-2)**

```
流量项 = 2·N·elem_bytes / HBM带宽
发射项 = N / 向量吞吐
容量项 = 0(单块 UB 驻留)
估算下界 = max(流量项, 发射项, 容量项) = 流量项
```

确定性结论:纯带宽受限,故不启用设计期探针(D-3 不适用)。
**实测校验(来自同批 `abs`/`ceil`/`floor`,本算子结构与 `abs` 最接近 —— 单条向量算子)**:实测带宽 ≈ **0.193 TB/s**(fp16/bf16)= 峰值 1.8 TB/s 的 **10.7%**,即**实际并非带宽受限而是每 tile 开销主导**。该 roofline 结论在实测下不成立,已记入 §10 R8。

#### §1.6.1 数学等价优化(公式级)

**结论:公式层无优化空间(显式结论 + 逐类依据)**

| 优化类别 | 结论 | 依据 |
|---|---|---|
| 恒等变形降代价 | **不适用(且已逐候选否决)** | `exp2`+乘法的替代式需 2 op 且引入额外舍入;R1 已给出否决依据 |
| 稳定化变形 | 不适用 | **无规约** → 不存在"先减最大值"的稳定化需求(那是 softmax 一类归约算子的手段);本算子是逐元素映射,上溢即 `+inf`,与源一致 |
| 公共子表达式消除 | 不适用 | 每元素仅参与一次计算 |
| 算子降代换 | 不适用 | `vexp` 已是 npui 原生一元算子,无更廉替代 |
| 访存量削减 | 已最优 | 读 1 遍 + 写 1 遍 = roofline 下界;无中间落 GM、无重复读 |
| 归约结构优化 | 不适用 | 无规约 |

> 因**无采纳项**,按 skill Phase 2 第 5 步规则(「显式『无优化空间』结论除外」)`verify_equiv.py` **豁免**,不生成。
> 与 `ceil`/`floor` 不同:本算子**没有**因 API 缺失而引入的重设计层分歧,故 §1.6.1 的豁免**不存在掩盖风险**。

#### §1.6.2 向量化替代分析(循环 / 标量消除)

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | 逐元素 $e^x$ | 逐元素循环 | **已向量化替代** | `T.vexp(f_ub, e_ub)`(`customize_npuir.py` 的 `npuir_exp(A,B)`;**源在前**) |
| V2 | `offset = cid * block_size` | block 级索引 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算**,每 block 1 个标量 |
| V3 | `tail_size = T.min(block_size, shape - offset)` | block 级边界 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算**;非动态 shape |
| V4 | `T.ceildiv(N, block_size)`(grid 维) | host 元数据 | **保留(不可替代)** | 理由类别:**host 侧 shape 元数据计算** |

**结论**:逐元素热点(V1)已完全向量化;保留的 3 处均为 block 级/host 级标量,**无逐元素标量循环残留**。

#### §1.6.3 向量化轴与数据布局决策

**单轴算子**:I/O 契约已把数据定为 1-D flat `(N,)`,核内布局无自由度。

- **候选轴枚举**:**仅 1 个** —— flat N 的连续轴。按 skill 豁免条款不展开多候选评分。
- **向量宽度对齐**:`block_size = 2048`;fp16/bf16 ×8 → 整除 ✓;fp32 ×4 → 整除 ✓;尾 lane 浪费率 0。
- **32B 对齐**:2048×2B=4KB(fp16)、2048×4B=8KB(fp32),均为 32B 整数倍 ✓。
- **决策落地**:主选布局 = 连续 1-D + `block_size=2048`,是 §3.3 伪代码与 §4 缓冲形状的唯一依据。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无**(纯 Vector) |
| 是否需要手工同步 | **否** —— 块内为"搬入 → 向量算子 → 搬出"的线性依赖,自动同步可正确处理 |
| 是否需要控制 L1/L0 层级 | **否**(只用 UB) |
| 是否需要跨核通信 | **否**(块间完全独立) |
| 与既有实现一致性 | 本仓 `kernels/elementwise/{abs,ceil,floor,exp}` 同为 Developer |

---

## §3 API 映射设计

### §3.1 公式拆解

依据 §1.6.1(公式层无优化项):

| 步 | 计算 | 对应公式 |
|---|---|---|
| 1 | 取块:读入 `x[offset : offset+tail_size]` | 输入分块 |
| 2 | $y_j = e^{x_j}$(fp32 域) | §1.4 |
| 3 | 写回 `y[offset : offset+tail_size]` | 输出分块 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| grid 划分 | `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)` | 本仓同族实现 |
| **UB 缓冲分配** | **`T.alloc_ub((block_size,), dtype)`** | 本仓同族实现 |
| GM → UB | `T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])` | 同上 |
| **升位(fp16/bf16)** | `T.vcast(x_ub, f_ub, round_mode="rint")`;fp32 直通(`T.copy`) | `customize_npuir.py:723-759` |
| **$e^x$** | **`T.vexp(f_ub, e_ub)`** — 语义 `e_ub = exp(f_ub)`,**源在前** | `customize_npuir.py` 的 `npuir_exp(A,B)`;⚠️ 写反不报错、只出错值 |
| **降回输出 dtype** | `T.vcast(e_ub, y_ub, round_mode="rint")`;fp32 直通(`T.copy`) | 同上 |
| UB → GM | `T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])` | 本仓同族实现 |
| 尾块 | `T.min(block_size, shape - offset)` | 本仓 `logsumexp` 的 `real_m` 同形 |

> ⚠️ **API 陷阱(实测)**:
> 1. **npui 向量算子是源在前、目标在后**(`T.vexp(A,B)` 即 `B = exp(A)`)—— 与 ascendc-pto 线的 `T.tile.*(dst,src)` **相反**;写反不报错、只出错值。
> 2. **Vector 缓冲必须 `T.alloc_ub`**;`T.alloc_shared` 降到 L1 → `UB→GM` store 编译失败(报错文本不含 buffer 名)。
> 3. `round_mode` 只在**真发生 dtype 变化**的 `vcast` 上生效(本算子的升/降位均真变 dtype,不受影响)。

### §3.3 伪代码

```python
def exp_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype),
                 y: T.Tensor((N,), out_dtype),
                 shape: T.int32):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                f_ub = T.alloc_ub((block_size,), compute_dtype)
                e_ub = T.alloc_ub((block_size,), compute_dtype)
                y_ub = T.alloc_ub((block_size,), out_dtype)

                offset = cid * block_size                      # V2 保留(block 级)
                tail_size = T.min(block_size, shape - offset)  # V3 保留(block 级)

                T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                # 1) widen to fp32 (fp32 goes straight through)
                if dtype == compute_dtype:
                    T.copy(x_ub, f_ub)
                else:
                    T.vcast(x_ub, f_ub, round_mode="rint")

                # 2) exp in the fp32 domain -- source-first: e_ub = exp(f_ub)
                T.vexp(f_ub, e_ub)

                # 3) narrow to the output dtype
                if out_dtype == compute_dtype:
                    T.copy(e_ub, y_ub)
                else:
                    T.vcast(e_ub, y_ub, round_mode="rint")

                T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])
        return main
    return _func
```

> 无补偿路径(与 `ceil`/`floor` 的关键差异):npui 有 `npuir_exp`,且与源"升 fp32 再算"逐位对应。

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| 一元逐元素 + `T.alloc_ub` + 分层套件(**结构最接近本算子**) | `examples/TileOPs/tileops/kernels/elementwise/abs/abs_kernel/abs.py` |
| 已走完整流程的同批算子(含 harshest 的 C2 方案,供对照) | `examples/ceil/ceil_fwd_kernel/ceil_fwd_kernel.py`、`examples/floor/floor_fwd_kernel/floor_fwd_kernel.py` |
| 既有 exp NPU 实现(只读参考) | `examples/TileOPs/tileops/kernels/elementwise/exp/exp_kernel/exp.py` |
| `npuir_exp` 与 `vcast` 定义 | `tilelang/language/customize_npuir.py` |
| 尾块切片既有形态 | `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/…` |

---

## §4 数据规格与内存规划

### §4.1 数据规格

| 张量 | shape | dtype | 所在 | 分配 API |
|---|---|---|---|---|
| `x` | `(N,)` | float16 / bfloat16 / float32 | GM | `T.Tensor` 入参 |
| `y` | `(N,)` | 同 `x` | GM | `T.Tensor` 出参(`out_idx=[1]`) |
| `shape` | 标量 | int32 | 入参 | `T.int32` 运行期标量 |
| `x_ub` | `(block_size,)` | 同 `x` | UB | `T.alloc_ub` |
| `f_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(升位后的 x;fp32 直通时为其副本) |
| `e_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(fp32 域的 $e^x$) |
| `y_ub` | `(block_size,)` | 同 `y` | UB | `T.alloc_ub`(降位后的输出) |

> **内存层级 ↔ 分配 API 固定对照**:GM = `T.Tensor`;UB = **`T.alloc_ub`**;L1 = `T.alloc_shared` / `T.alloc_L1`;L0C = `T.alloc_L0C`。
> **本算子只用 UB**。误写 `T.alloc_shared` 会降到 L1,使 `UB→GM` store 编译失败。

### §4.2 搬运路径

```
GM(x) --T.copy--> UB(x_ub) --vcast--> UB(f_ub) --vexp--> UB(e_ub) --vcast--> UB(y_ub) --T.copy--> GM(y)
```

无 L1 / L0 参与;无 GM 中间结果落盘。

### §4.3 UB 容量预算

| buffer | 尺寸表达式 | fp32 峰值 |
|---|---|---|
| x_ub | block_size × elem_bytes | 8 KB |
| f_ub | block_size × 4 | 8 KB |
| e_ub | block_size × 4 | 8 KB |
| y_ub | block_size × elem_bytes | 8 KB |

**合计**:32 KB(fp32、`block_size = 2048`)。

UB 容量上限 192 KB,自动多缓冲膨胀系数 1.7 → 有效预算约 113 KB(来源:`pattern-library/constants.md` 的 `CONST-capacity-910B2C`)。峰值占用低于有效预算。

**结论**:无容量压力,不需要为容量缩小 `block_size`。

### §4.4 中间缓冲峰值

**0** —— 无跨块中间量,无 workspace,无跨核共享。

---

## §5 Tiling 策略

### §5.1 Block 划分与 Tile Shape

| 项 | 值 |
|---|---|
| 划分维度 | 唯一一维轴 `N` |
| Tile Shape | `block_size = 2048` |
| Block 数 | `T.ceildiv(N, block_size)` |
| 每 block | 连续 `block_size` 个元素(尾块为 `tail_size`) |

### §5.2 非整除处理策略

**Kernel 内运行期切片**:`tail_size = T.min(block_size, shape - offset)`。源/目的切片共用 `tail_size`;UB 中超出 `tail_size` 的尾 lane 不参与写出,故无需 `T.clear`。

### §5.3 分核策略(三要素)

**① 逻辑核数计算**

```
逻辑核数 = T.ceildiv(N, block_size)
```

| workload | N | 逻辑核数 |
|---|---|---|
| smoke-1m | 1 048 576 | 512 |
| elementwise-16m | 16 777 216 | 8192 |
| elementwise-64m | 67 108 864 | 32 768 |
| elementwise-256m | 268 435 456 | 131 072 |

**② 物理核数依据(实查,非假设)**

```
torch.npu.get_device_properties(0) → device: Ascend910B2C
    cube_core_num   = 24
    vector_core_num = 48      ← 本算子为纯 Vector,取此值
```

> **查询记录**:经 `torch_npu` 的 `get_device_properties(0)` 实查,`vector_core_num = 48`、`cube_core_num = 24`、设备名 `Ascend910B2C`。未使用假设值。

**③ 规模判定与分核方案**

| workload | 逻辑核数 | 物理核数 | 倍数 | 判定 | 方案 |
|---|---|---|---|---|---|
| smoke-1m | 512 | 48 | 10.7× | 中等规模 | **无需适配** |
| 16m | 8192 | 48 | 170.7× | 中等规模 | 无需适配 |
| 64m | 32 768 | 48 | 682.7× | 中等规模 | 无需适配 |
| 256m | 131 072 | 48 | 2730.7× | 极大规模 | 无需核内 `T.serial` |

**结论:三选一命中「无需适配依据」**。依据:一维 `T.Kernel(grid)` 由运行时调度器分发到 48 个物理核,**块间完全独立**;块粒度 2048 元素 ≈ 4KB 足够小、负载均衡良好;边界虽为运行期 `tail_size`,但它由 `cid` 静态推导,不构成动态 shape。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| `T.Parallel` | **不使用** | §1.6.2 判定逐元素热点由 `T.vexp` 覆盖,无需展开 |
| `T.serial` | **不使用** | 每核仅 1 块,无核内任务循环 |
| `T.Pipelined` | **不使用** | 单步算子,无多阶段可重叠 |
| `T.Persistent` | **不使用** | 无核常驻需求 |

**总结**:kernel 内**无显式循环** —— 数据并行由 `T.Kernel` 的 grid 维承担,块内计算由单条 `T.vexp` 承担。

> 一致性核对:§1.6.2 判定 V1 已向量化,故 §6 无对应逐元素标量循环;V2–V4 三处 block/host 级标量在 §3.3 保留且均非循环体。

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入(基于数据依赖分析) |
| 手工同步点 | **无** |
| 依赖性质 | 线性:`GM→UB`(copy) → `UB→UB`(vcast/vexp) → `UB→GM`(copy);同块内无跨 lane 通信、无跨核依赖 |
| 跨核同步 | 不需要(块间独立) |

**正确性依据**:Developer 模式下 `T.alloc_ub` 的 UB 生命周期与访问顺序由编译器自动管理;本算子依赖线性、无环、各 buffer 独立无 aliasing(`x_ub`/`f_ub`/`e_ub`/`y_ub` 互不重叠)。

---

## §8 CV 融合设计

**n/a** —— 纯 Vector,不涉及 Cube 计算,无 `Scope("Cube")`/`Scope("Vector")` 协同、无 `T.sync_block_set/wait`、无 `PIPE_FIX`。

---

## §9 验证方案

### §9.1 Golden 函数草案(独立于 NPU 算法)

```python
def golden_exp(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现 —— 按 §0.1 语义。

    fp32 域求指数后回落原 dtype,与源仓参考实现口径一致
    (源 op_func 即 T.exp(T.cast(x, "float32")))。只依据语义实现,
    不复刻 NPU 侧实现 —— 保证验证独立性。
    """
    x_cpu = x.detach().cpu()
    return torch.exp(x_cpu.to(torch.float32)).to(x_cpu.dtype)
```

### §9.2 L0 门槛测试计划

| 用例 | N | dtype | 容差(atol/rtol) |
|---|---|---|---|
| smoke | 1 048 576 | float16 | 1e-3 / 1e-3 |
| smoke | 1 048 576 | bfloat16 | 1.6e-2 / 1.6e-2 |
| smoke | 1 048 576 | float32 | 1e-5 / 1e-5 |

容差逐字沿用源仓 `_get_tolerances`。

**L1(阻塞层)必须包含的用例 —— 逐条对应 §0.1 声明为"必须保持"的边界语义**:

| 类别 | 用例 | 为什么放在**阻塞层** |
|---|---|---|
| 非整除 shape | `N % 2048 != 0`(如 N=1 / 2 / 10 / 2049 / 100003) | 尾块路径的唯一覆盖 |
| 常规值 | `randn × 3`(覆盖正负两侧) | 基础正确性 |
| **`NaN` 传播** | `NaN` | §0.1 必须保持;**`vexp` 对特殊值的指令级行为未在本项目实测过** —— `ceil`/`floor` 的教训(`vcast` 对 NaN 得 `0`)说明不能假设 |
| **`+inf` / `-inf`** | `+inf → +inf`、`-inf → +0.0` | 同上 |
| **上溢(fp32 域)** | `x = 100.0`(fp32 下 $e^{100}$ 溢出)→ `+inf` | 源算子既有语义 |
| **下溢** | `x = -100.0` → `+0.0` | 同上 |
| **fp16/bf16 结果超范围** | fp16 下 `x = 20.0`($e^{20} \approx 4.85\times10^8$ > fp16 max)→ `+inf`;**而 fp32 参照仍是有限值** | §0.1 的"计算域 fp32、结果落 out_dtype"两步落差 —— **这是本算子最易被误解的一条语义** |
| $x \approx 0$ | `±0.0`、`±1e-8` → $\approx 1.0$ | 近零侧的稳定性 |

> ⚠️ 边界语义既为"必须保持",就应落在 **L1(阻塞层)**,不宜放到 L2/Boundary(告警层)。

### §9.3 与既有实现的关系(需后续 Stage 裁决)

> 本算子已有手工 NPU 实现且测试通过(`exp_kernel/exp.py`),其形态与本设计**一致**(`widen → vexp → narrow`,并已注明 `T.vexp(A,B)` 源在前)。
> **差异仅在**:本设计把 §9.2 的边界用例(NaN / ±inf / 上溢 / 下溢 / fp16 超范围)明确列为 **L1 阻塞层**;既有实现的测试(`tests/ops/test_exp.py`,18 项)以 shape/dtype 为主。
> **本设计不修改既有文件**;是否以本设计为准由 Stage 2 裁决。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| **R1** | **`vexp` 对 `NaN`/`±inf` 的指令级传播行为未在本项目实测过** | **高** | §0.6 已显式声明该未验证点;§9.2 把 NaN/±inf/上溢/下溢/超范围**全部列入 L1 阻塞层**;**若 Stage 3 实测发现 `vexp` 不传播**,按 `[DESIGN_ERROR]` 返回并补 C2 式旁路(与 `ceil`/`floor` 同法,代价 +2~4 op) |
| R2 | **fp16/bf16 结果超范围 → `+inf`,而 fp32 参照有限** | 中 | 这是**源算子既有语义**(§0.1);golden 用 `torch.exp(x.float()).to(dtype)` 与此一致;§9.2 已列 L1 用例 |
| R3 | `T.vexp` 的参数顺序(源在前) | 高 | §3.2 已注明依据;写反不报错、只出错值 |
| R4 | 尾块非整除路径的实测覆盖 | 中 | manifest 的 4 个 workload 的 N 均被 2048 整除 → **非整除路径未被 manifest 覆盖**;Stage 3 的 L1 必须补 |
| R5 | 整型 dtype | 低 | `ExpFwdOp(UnaryOp)` **无整型短路**,非 float dtype 由基类校验拒绝(源侧即如此,`test_exp_rejects_non_float_dtype`);NPU 侧一致 |
| R6 | 技术约束检测结论 | — | 三维 Kernel ❌ 未使用;GPU 专用 API ❌;GEMM 非整除 ❌ 不适用;L0C 溢出 ❌ 不适用 → **无技术约束冲突** |
| **R7** | **§1.6.0 的"纯带宽受限"结论被同族实测否证**:`abs`/`ceil`/`floor` 实测带宽 ≈ **0.193 TB/s = 峰值 1.8 TB/s 的 10.7%**,说明实际由**每 tile 开销**主导而非带宽 | 中 | 本算子结构与 `abs` 最接近(单条向量算子),预期同量级;Stage 4(调优)不在 harness 迁移范围内,故该性能缺口**已知且未处理**,须如实记录而非掩盖 |
| R8 | **无重设计层分歧** | — | 与 `ceil`/`floor` 不同:npui 有 `npuir_exp`,不存在因 API 缺失而引入的值域/NaN 分歧,故 §1.6.1 的豁免无掩盖风险 |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/exp/exp_fwd_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/exp/exp_fwd_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**(§1.6.1 公式层无采纳优化项;本算子**无**重设计层分歧,不存在掩盖风险) |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/…` | ✅ Stage 0 已产出 |
| 5 | npuir kernel + golden + 分层测试 | `examples/exp/exp_fwd_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |

