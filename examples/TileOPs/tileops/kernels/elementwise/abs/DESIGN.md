# DESIGN.md — `AbsFwdOp`(TileLang-NPUIR)

- **project** = `abs`,**op_name** = `AbsFwdOp`,**family** = `elementwise`
- **迁移类型**:migration(harness 子模式)
- **编程模式**:Developer
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`(GPU TileOPs)
- **`source_output_shape`**:与输入同形(逐元素);kernel 层为 1-D flat `(N,)`
- **golden 独立性声明**:§9.1 的 golden 依据 §0.1 语义实现(PyTorch 参考),**未复刻 NPU 算法**

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | $y_i = |x_i|$,逐元素一元映射,无跨元素依赖 | `AbsFwdKernel.op_func(x) = T.abs(x)`(`kernels/elementwise.py:2014-2019`) |
| I/O 契约 | 输出 shape **与输入逐维相同**;dtype 相同(`same_as(input)`);非 in-place(另有 in-place 变体) | manifest `shape_rules: output.shape == input.shape`;`_base.py:151` `torch.empty_like(x, dtype=out_dtype)` |
| 规约语义 | **无规约**(无归约轴、无 keepdim、无累加顺序问题) | 计算图无 reduce |
| dtype 语义 | kernel 家族 `FloatUnaryKernel` 的 `SUPPORTED_DTYPES = _FLOAT_DTYPES`,**float-only**;中间无精度提升(单算子直接在输入 dtype 上算) | `kernels/elementwise.py:1297` |
| 边界语义 | `+inf → +inf`、`-inf → +inf`、`NaN → NaN`、`-0.0 → +0.0`;均为 `T.abs` 的算术语义 | `T.abs` 语义 |
| 整型语义 | manifest 声明 `int8/int16/int32/int64/uint8`,但 `AbsFwdOp` 继承 **`_IntIdentityUnaryOp`**:整型走 `_int_handler`(默认 `input.clone()`)→ **在 op 层短路,不构造 kernel** | `ops/elementwise/math_unary.py:59`;`ops/elementwise/_base.py:1146-1190` |
| host 侧语义 | 无预处理;仅 flatten(`N_total`)与 reshape 回原形 | `UnaryOp.__init__(N_total, dtype)`;`_eager_forward` reshape |

**完成标准核对**:已可写出与源算子数值一致的 golden —— `torch.abs(x.float()).to(dtype)`(见 §9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | 任意 rank 的 `float16 / bfloat16 / float32` 张量,kernel 层 flatten 为 `(N_total,)` |
| 输出 | 与输入同 shape、同 dtype |
| 动态轴 | **无**;`N_total` 在构造期确定(`UnaryOp(N_total=...)`),kernel 按 `N` 静态实例化 |
| in-place | 非默认;另有 `_register_unary_inplace_custom_op` 注册的 in-place 变体(NPU 侧不在本次范围) |

### §0.3 第二问:实现算法(How)

源码结构化阅读结果:

1. **入口签名**:`abs_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)`,返回 `kernel(threads_arg, npt_arg)`(regcopy/explicit)或 `kernel(threads_arg)`(direct)。
2. **Kernel 结构**:`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx`,其中 `block_size = threads × num_per_thread`。**一维 grid,每 block 处理 `block_size` 个连续元素**。
3. **计算原语**:三种策略的 `@T.prim_func` 体——
   | 策略 | 体 |
   |---|---|
   | `direct`(1 元素/线程) | `for i in T.Parallel(threads_arg): y[bx*t+i] = op_func(x[bx*t+i])` |
   | `explicit_parallel` | `for i,j in T.Parallel(threads_arg, npt_arg): idx=(bx*t+i)*npt+j; y[idx]=op_func(x[idx])` |
   | `register_copy`(**默认**) | `x_reg=alloc_fragment(block_size)`;`T.copy(x[bx*bs:(bx+1)*bs], x_reg)`;`for i,j in T.Parallel(t,npt): y_reg[i*npt+j]=op_func(x_reg[i*npt+j])`;`T.copy(y_reg, y[...])` |
4. **内存分配**:`T.alloc_fragment((block_size,), dtype)` ×2 —— 仅寄存器/fragment 级,**无 SMEM**、无跨 block 共享。
5. **循环与调度**:`T.Parallel(threads, npt)` 二维线程映射;无 `T.serial` / `T.Pipelined` / unroll 指令。
6. **host 侧逻辑**:`UnaryOp.__init__` 构造 kernel;`_eager_forward` flatten → kernel → reshape 回原形。`init_config(config, tune)` 缓存编译后的 callable。
7. **编译配置**:无 `pass_configs`;策略与 `threads/npt` 由 `default_config` 决定(`strategy` 来自 `DEFAULT_STRATEGY` 或 config)。

**完成标准核对**:每个计算语句均归入上表策略体之一;无臆造步骤;数据流 `GM → fragment → compute → fragment → GM` 覆盖全部 buffer。

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| G1 | **register_copy 策略**(默认) | 合并访存、提高有效带宽 | 整块 `T.copy` 进 fragment → 计算 → 整块写回,替代逐元素 GM 读写 | 大寄存器堆 + coalesced 访存 |
| G2 | **`threads × npt` 二维并行** | 每 block 摊薄调度开销 | `T.Parallel(threads, npt)`;`block_size = 256×8 = 2048` | CUDA block/thread 两级并行 |
| G3 | **按 dtype 调 `npt`**(128-bit 宽访存) | fp16/bf16 用 `npt=8` 使单线程访存达 LDG.128 | `_strategy_npt(strategy, dtype)` | LDG.128 |
| G4 | **三策略编译期可选** | 按 shape/dtype 选最优结构 | `direct` / `explicit_parallel` / `register_copy` | — |
| G5 | autotune(`threads`×`npt` 搜索) | 找最优执行参数 | `autotune_configs` + `autotune()` | — |

> abs **未使用**:SMEM tiling、swizzle、warp shuffle、`T.gemm`、异步流水。该算子是最简单的一元逐元素结构。

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| `y=|x|` 数学语义、I/O 契约、边界/dtype 语义 | 语义 | **保留** | 语义层不可妥协(migration-analysis §5.1) |
| G1 fragment 中转 | 优化 | **等价替换** | 映射表「register 累加 → L0C/UB 中间缓冲」→ UB tile + `T.copy` |
| G2 `threads × npt` 两级并行 | 优化 | **重新设计** | 映射表「block × thread 两级并行 → 一维 block + 向量化元素并行」;本项目 `T.Kernel` 仅接受一维 block 数 |
| `T.Parallel(threads, npt)` | 优化 | **等价替换** | 同上;→ 向量算子(`T.vabs`)覆盖整 tile |
| G3 按 dtype 调 `npt` | 优化 | **等价替换** | NPU 侧对应"向量宽度对齐"(fp16/bf16 ×8、fp32 ×4);由 `block_size` 承载 |
| G4 三策略编译期可选 | 优化 | **重新设计** | npuir Developer 模式下 UB 驻留是唯一自然形态,三策略收敛为一实现;性能意图(单策略内的访存合并)由 G1 的等价替换承接 |
| G5 autotune | 优化 | **舍弃** | K8:NPU 无 autotune;`init_config(config)` 不含 `tune`。**理由**:本项目 NPU 侧 kernel 编译开销高、且无 autotune 框架,启发式配置是既定约定 |
| SMEM tiling / swizzle / warp shuffle | — | **n/a** | abs 源码未使用 |

### §0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的两项(G2 并行结构、G4 策略收敛):

**重设计项 1 —— 并行结构(G2)**

- **源方案**:CUDA 三级并行(`grid=ceildiv(N, threads*npt)`、`block=threads`、每线程 `npt` 个连续元素)。
- **NPU 新算法**:一维任务划分 —— `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)`,每个 block(cid)负责连续 `block_size` 个元素;块内**不做线程映射**,而是把整块搬入 UB 后由**向量指令**一次性处理;尾块用运行期切片 `tail_size = T.min(block_size, N - offset)` 处理。
- **语义保持论证**:逐元素映射,**无规约、无重排、无累加顺序** —— 并行度变化不改变任何元素的取值路径,数学恒等。边界语义(NaN/±inf/±0)由 `T.vabs` 指令语义保证,与源 `T.abs` 一致。

**重设计项 2 —— 策略收敛(G4)**

- **源方案**:三套 kernel 体(`direct` / `explicit_parallel` / `register_copy`)按 config 编译期选择。
- **NPU 新算法**:收敛为**单一实现** = 源 `register_copy` 的等价替换形态(UB tile 中转 + 向量算子)。**源 G4 的性能意图(按场景选结构)由什么承接**:NPU 侧 `block_size` 仍是可配置量,通过它调节单块粒度;三策略差异在 npuir 上退化为"UB tile 大小"这一个自由变量,故不需要三套 kernel 体。
- **语义保持论证**:同重设计项 1(逐元素、无重排)。

**重设计项 3 —— 尾块处理(承接源边界的非必需部分)**

- 源 `register_copy` 用 `T.copy(x[bx*bs:(bx+1)*bs], x_reg)` 隐含要求 `N` 被 `block_size` 整除;NPU 侧显式引入 `tail_size` 运行期切片,覆盖 `N % block_size != 0`(本算子 workload 中 `1048576 % 2048 == 0`,但 `4096×4096`/`8192×8192`/`16384×16384` 同样整除 —— 仍按非整除路径设计以求通用)。
- **语义保持论证**:切片仅改变每块处理的元素个数,不改变元素的映射与取值。

### §0.7 迁移决策摘要

- **三问结论**:语义 = 逐元素 `|x|`、shape/dtype 保持、无规约;算法 = 一维 grid + `threads×npt` 块内映射 + 三策略;优化手段 = 5 项(G1–G5)。
- **耦合性判定统计**:**保留 1** 项(语义层整体)、**等价替换 3** 项(fragment 中转 / `T.Parallel` 形态 / npt 宽度对齐)、**重新设计 2** 项(两级并行 → 一维、三策略 → 单实现)、**舍弃 1** 项(autotune)、**n/a 1** 项(SMEM/swizzle,源未用)。
- **重设计项清单**:① 并行结构重设计(一维 Kernel + 向量块内处理 + 尾块切片);② 策略收敛(承接 G4 性能意图为 `block_size` 自由变量)。
- **源优化意图承接**:G1 → UB tile 中转;G2 → `block_size` 一维划分;G3 → 向量宽度对齐;G4 → `block_size` 可配;G5 → **舍弃**(见 §0.5 理由)。**无静默丢弃**。

---

## §1 概述

### §1.1 算子概述

`AbsFwdOp`(abs)—— 逐元素绝对值 $y = |x|$。用于激活前置、幅度统计、量化前处理等场景。
**迁移后 NPU 侧算法**:一维 grid + UB tile 中转 + `T.vabs` 向量算子(与源结构不同之处见 §0.6)。

### §1.2 计算语义

- 输入:$x \in \mathbb{R}^{N}$,dtype ∈ {float16, bfloat16, float32}
- 输出:$y_i = |x_i|$,同 shape 同 dtype
- 无规约、无跨元素依赖、无中间标量状态

### §1.3 算子特征分析

| 维度 | 判定 |
|---|---|
| 计算类型 | **纯 Vector**(逐元素) → 仅需 UB,**无 Cube / 无需 L1/L0** |
| 复杂度级别 | **单步**(单次搬运 + 单条向量算子,无循环、无中间缓冲) |
| 动态 shape | **无**(`N` 为构造期常量) |
| host 预处理 | **无** |

### §1.4 算法描述(依据 Phase R 调研选定算法)

对 $N$ 个元素,按 `block_size` 切分为 $\lceil N / block\_size \rceil$ 个块,每块:

1. `T.copy` 把该块从 GM 搬入 UB 缓冲 `x_ub`;
2. `T.vabs(x_ub, y_ub)` 一次性算出整块的 $|x|$;
3. `T.copy` 把 `y_ub` 写回 GM。

无逐元素标量循环(§1.6.2 已确认该点可完全向量化)。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R — 轻量调研)

> 深度分级依据:单步逐元素类 → 按 skill 规则执行**轻量调研**(四问各一行结论 + 依据)。
> 调研范围:同一数学语义($y=|x|$)的算法族 + 本仓 elementwise 族既有实现(`kernels/elementwise/lerp_tensor.py`、`mish.py`)+ 本 skill `references/algorithm-candidates.md` 的 R1/R2 参考表。

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **无可更省的等价形式**(基线即最优) | 候选:① `sign(x)·x`(2 op,更贵);② `x·(x≥0 ? 1 : −1)`(含比较与选择,更贵);③ 位运算清符号位(`bitcast→and 0x7FFFFFFF→bitcast`,需 3 条指令且依赖 `vbitcast` 的原地 API —— 本仓零使用先例,**未探明**)。基线 `vabs` 为单条指令,候选均不更省 → 基线胜出。 |
| **R2 在线算法** | **无** | 结构依据:输出每个元素只依赖对应输入元素,不存在 running 统计量或跨块依赖,故**不存在**流式/单遍变体可谈 —— 本算法本身就是单遍。 |
| **R3 复杂度四口径** | 基线:**FLOPs = N**(1 op/elem)、**访存 = 2·N·elem_bytes**(读 1 遍 + 写 1 遍)、**扫描遍数 = 1**、**中间缓冲 = 0**(UB tile 驻留,不落 GM) | 与 manifest roofline 逐字一致(`flops: N`,`bytes: 2*N*elem_bytes`)—— **已在 roofline 下界,无下降空间**。 |
| **R4 硬件亲和** | 逐候选:基线 `vabs` → 纯 Vector、只需 UB(192KB 绰绰有余)、无 Cube、无分形约束;向量宽度 fp16/bf16 ×8、fp32 ×4,`block_size=2048` 对三者均整除(32B 对齐满足);边界静态(`N` 构造期常量,尾块用运行期 `tail_size`)。候选 ③(位运算)依赖未探明 API → 不亲和不采纳。 | 基线**亲和**。 |

**调研结论**:选定算法族 = **基线(UB tile 中转 + 单条向量 `vabs`)**,无更优替代。调研范围已写明(R1 三候选 + R2 结构判据 + R3 与 roofline 对齐 + R4 逐候选)。

**设计期 roofline 估算下界(D-2)**:

```
流量项 = 2·N·elem_bytes / HBM带宽
发射项 = N / 向量吞吐
容量项 = 0(单块 UB 驻留,无容量约束)
估算下界 = max(流量项, 发射项, 容量项) = 流量项
```

确定性结论:**本算子纯带宽受限**,发射项与容量项均不占优,无需引入常数标定;故不启用设计期探针(D-3 不适用)。

#### §1.6.1 数学等价优化(公式级)

**结论:无优化空间(显式结论 + 依据)**

| 优化类别 | 结论 | 依据 |
|---|---|---|
| 恒等变形降代价 | 不适用 | 无除法、无 sqrt、无 log/exp —— `|x|` 为单条指令,无分解余地 |
| 稳定化变形 | 不适用 | 无规约、无 exp,不存在上溢/下溢路径 |
| 公共子表达式消除 | 不适用 | 每个元素只参与一次计算,无重复子式 |
| 算子降代换 | 不适用 | 已是最低代价算子(`vabs`);降代换目标不存在 |
| 访存量削减 | 已最优 | 读 1 遍 + 写 1 遍 = roofline 下界;无中间落 GM、无重复读、无多余 cast |
| 归约结构优化 | 不适用 | 无规约 |

> 因**无采纳项**,按 skill Phase 2 第 5 步规则(「显式『无优化空间』结论除外」)`verify_equiv.py` **豁免**,不生成。

#### §1.6.2 向量化替代分析(循环 / 标量消除)

盘点实现方案中的**全部**计算点:

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | $y_i = |x_i|$ 逐元素映射 | 逐元素循环 | **已向量化替代** | `T.vabs(x_ub, y_ub)`(v-prefix 向量算子,`tilelang/language/customize_npuir.py: npuir_abs` 佐证;仓内调用 `examples/deepseek_v4/example_act_quant_kernel.py:61`) |
| V2 | `offset = cid * block_size` | block 级索引 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算** —— 每 block 仅 1 个标量,非逐元素热点 |
| V3 | `tail_size = T.min(block_size, N - offset)` | block 级边界 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算**;**非**动态 shape(尾块与 `cid` 绑定,`N` 为构造期常量),每 block 1 个标量 |
| V4 | `T.ceildiv(N, block_size)`(grid 维) | host 元数据 | **保留(不可替代)** | 理由类别:**host 侧 shape 元数据计算**,不在 kernel 内 |

**结论**:逐元素热点(V1)已完全向量化;保留的 3 处(V2–V4)均为 block 级/host 级标量,**无逐元素标量循环残留**。构成 §6 循环结构的约束:§6 中不得出现对应 V1 的逐元素标量循环。

#### §1.6.3 向量化轴与数据布局决策

本算子是**单轴算子**:I/O 契约已把数据定为 1-D flat `(N,)`,核内布局无自由度。

- **候选轴枚举**:仅 1 个候选 —— flat N 的连续轴。按 skill 豁免条款(「纯 GEMV / 单轴算子可写明『仅一个候选轴』豁免」),不展开多候选评分。
- **向量宽度对齐**:`block_size = 2048`;fp16/bf16 向量宽 ×8 → 2048 % 8 = 0 ✓;fp32 ×4 → 2048 % 4 = 0 ✓;**尾 lane 浪费率 0**。
- **32B 对齐**:`block_size=2048` 的元素字节数 = 2048×2B(fp16)= 4KB、2048×4B(fp32)= 8KB,均为 32B 的整数倍 ✓。
- **决策落地**:主选布局(连续 1-D、`block_size=2048`)= §3.3 伪代码与 §4 缓冲形状的唯一依据;§6 无独立的"内层向量维"(整 tile 由单条向量算子处理)。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无**(纯 Vector) |
| 是否需要手工同步 | **否** —— 块内为"搬入 → 向量算子 → 搬出"的线性数据依赖,自动同步可正确处理 |
| 是否需要控制 L1/L0 层级 | **否**(只用 UB) |
| 是否需要跨核通信 | **否**(块间完全独立) |
| 与既有实现一致性 | 本仓 `kernels/elementwise/{mish,lerp_tensor}` 同为 Developer 模式 |

**结论**:Developer 模式即可完整表达该算法,无需 Expert 的手工同步/层级控制能力。Expert 会引入无收益的复杂度。

---

## §3 API 映射设计

### §3.1 公式拆解

依据 §1.6.1(无优化项,即原式):

| 步 | 计算 | 对应公式 |
|---|---|---|
| 1 | 取块:读入 `x[offset : offset+tail_size]` | 输入分块 |
| 2 | $y_j = |x_j|$, $j \in [0, tail\_size)$ | $y=|x|$ |
| 3 | 写回 `y[offset : offset+tail_size]` | 输出分块 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| grid 划分 | `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)` | 本仓 `kernels/reduction/logsumexp/...`、`kernels/norm/ada_layer_norm/...` 同形用法 |
| UB 缓冲分配 | **`T.alloc_ub((block_size,), dtype)`** | 本仓 `kernels/elementwise/mish/...`、`kernels/elementwise/abs/abs_kernel/abs.py` 的既有形态 |
| GM → UB | `T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])` | 同上 |
| **$\|x\|$** | **`T.vabs(x_ub, y_ub)`**(语义 `y_ub = \|x_ub\|`,**源在前、目标在后**) | `tilelang/language/customize_npuir.py: npuir_abs(A,B)`;仓内调用 `examples/deepseek_v4/example_act_quant_kernel.py:61`、`examples/deepseek_v4/engram/engram_gate_kernel_fwd.py:178` |
| UB → GM | `T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])` | 本仓同族实现 |
| 尾块计算 | `T.min(block_size, N - offset)` | 本仓 `logsumexp` 的 `real_m = T.min(block_m, M - pid_m*block_m)` 同形 |

> ⚠️ **API 陷阱已规避**:npui 一元算子是"源在前、目标在后"(`T.vabs(A, B)` 即 `B = |A|`),与 ascendc-pto 线的 `T.tile.abs(dst, src)` **相反**。此处依据源码与仓内真实调用核对,未凭记忆。

### §3.3 伪代码

```python
def abs_fwd_kernel(N, dtype, block_size=2048):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)

                offset = cid * block_size                      # V2 保留(block 级)
                tail_size = T.min(block_size, N - offset)      # V3 保留(block 级)

                T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
                T.vabs(x_ub, y_ub)                             # V1 已向量化
                T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])
        return main
    return _func
```

> 伪代码中的 buffer 形状(`(block_size,)`)与向量维均与 §1.6.3 选定布局一致;无逐元素标量循环,符合 §1.6.2 结论。

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| **一元逐元素 + `T.Kernel(is_npu=True)` + UB 中转**(最接近的同族实现) | `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` |
| 一元逐元素的 Op/Kernel/wrapper 三层结构 | `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor.py` |
| 尾块切片 `T.min(block, total - offset)` 的既有形态 | `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` |
| 一维 `T.Kernel` + 持久化的「每核一块」写法 | `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/_ada_layer_norm_kernel.py` |
| **`T.vabs(A, B)` 参数顺序的真实调用佐证** | `examples/deepseek_v4/example_act_quant_kernel.py:61`、`examples/deepseek_v4/engram/engram_gate_kernel_fwd.py:178` |
| npuir 一元算子定义(`npuir_abs`) | `tilelang/language/customize_npuir.py`(`npuir_abs(A, B)` → `B = |A|`) |


---

## §4 数据规格与内存规划

### §4.1 数据规格

| 张量 | shape | dtype | 所在 | **分配 API** |
|---|---|---|---|---|
| `x` | `(N,)`(kernel 层;逻辑 rank 任意) | float16 / bfloat16 / float32 | GM | `T.Tensor` 入参 |
| `y` | `(N,)` | 同 `x` | GM | `T.Tensor` 出参(`out_idx=[1]`) |
| `x_ub` / `y_ub` | `(block_size,)` | 同 `x` | UB | **`T.alloc_ub`** |

> **内存层级 ↔ 分配 API 的固定对应(逐条取值,不得自行填写)**
>
> | 层级 | 分配 API |
> |---|---|
> | GM | `T.Tensor`(kernel 入/出参) |
> | **UB** | **`T.alloc_ub`** |
> | L1 | `T.alloc_shared` / `T.alloc_L1` |
> | L0C | `T.alloc_L0C` |
>
> **本算子只用 UB**(§1.3 判定"纯 Vector")。**误写成 `T.alloc_shared` 会降到 L1**,使 Vector 侧的 `UB→GM` store 在编译期失败:
> `loc("kernel.npuir"): error: 'hivm.hir.store' op only support store ub to gm currently!`
> 该报错文本**不含** buffer 名或 API 名,定位只能靠本对照表。实证见
> `repro/TRAP-alloc-shared-vs-ub.py`(A: `alloc_shared` 报该错;B: `alloc_ub` 位精确)。

### §4.2 搬运路径(GM → UB → GM)

```
GM(x) --T.copy--> UB(x_ub) --T.vabs--> UB(y_ub) --T.copy--> GM(y)
```

无 L1 / L0 参与(纯 Vector,§1.3)。无 GM 中间结果落盘。

### §4.3 UB 容量预算

每个 block 需要两个 UB 缓冲(独立分配,不共用):

| buffer | 尺寸表达式 | fp32 峰值 |
|---|---|---|
| x_ub | block_size × elem_bytes | 8 KB |
| y_ub | block_size × elem_bytes | 8 KB |

**合计**:16 KB(fp32、`block_size = 2048` 时;fp16/bf16 各减半为 8 KB)。

UB 容量上限 192 KB,自动多缓冲膨胀系数 1.7 → 有效预算约 113 KB
(来源:`pattern-library/constants.md` 的 `CONST-capacity-910B2C`)。本算子峰值占用远低于有效预算。

**结论**:`block_size = 2048` 的 UB 占用无容量压力;故不需要为容量而缩小 `block_size`(§1.6.3 的 `block_size` 取值不受 UB 约束)。

### §4.4 中间缓冲峰值

**0** —— 无跨块中间量,无 workspace,无跨核共享(§1.6.0 R3 的"中间缓冲 = 0"在此落实)。

---

## §5 Tiling 策略

### §5.1 Block 划分与 Tile Shape

| 项 | 值 |
|---|---|
| 划分维度 | 唯一的一维轴 `N` |
| Tile Shape | `block_size = 2048`(元素数) |
| Block 数 | `T.ceildiv(N, block_size)` |
| 每 block 处理 | 连续 `block_size` 个元素(尾块为 `tail_size`) |

### §5.2 非整除处理策略

**Kernel 内运行期切片**(非 host padding+crop):

```
tail_size = T.min(block_size, N - offset)      # 尾块自动收缩
```

- 源/目的切片共用 `tail_size`(`T.copy` 的形状一致性要求);
- UB 中超出 `tail_size` 的尾 lane **不参与计算也不写出**(`T.vabs` 作用于整个 tile,但只有 `[0, tail_size)` 被搬出),故无需 `T.clear`;
- 依据:本仓 `logsumexp` 的 `real_m` 切片是同一模式的既有实现。

### §5.3 分核策略(三要素)

**① 逻辑核数计算**

```
逻辑核数 = T.ceildiv(N, block_size)
```

| workload | N | block_size | 逻辑核数 |
|---|---|---|---|
| smoke-1m | 1 048 576 | 2048 | 512 |
| elementwise-16m | 4096×4096 = 16 777 216 | 2048 | 8192 |
| elementwise-64m | 8192×8192 = 67 108 864 | 2048 | 32 768 |
| elementwise-256m | 16384×16384 = 268 435 456 | 2048 | 131 072 |

**② 物理核数依据(实查,非假设)**

```
torch.npu.get_device_properties(0) → device: Ascend910B2C
    cube_core_num   = 24
    vector_core_num = 48      ← 本算子为纯 Vector,取此值
```

> **查询记录**:经 `torch_npu` 的 `get_device_properties(0)` 实查,`vector_core_num = 48`、`cube_core_num = 24`、设备名 `Ascend910B2C`。未使用任何假设值。
> (旁证:本仓 `kernels/norm/ada_layer_norm` 的实现注释记为 `get_aicore_num() * 2`,与 48 一致。)

**③ 规模判定与分核方案**

| workload | 逻辑核数 | 物理核数 | 倍数 | 判定 | 方案 |
|---|---|---|---|---|---|
| smoke-1m | 512 | 48 | 10.7× | 中等规模 | **无需特殊适配** |
| 16m | 8192 | 48 | 170.7× | 中等规模 | 无需特殊适配 |
| 64m | 32 768 | 48 | 682.7× | 中等规模 | 无需特殊适配 |
| 256m | 131 072 | 48 | 2730.7× | **极大规模** | 无需核内 `T.serial`(见下) |

**结论:三选一命中「无需适配依据」**。依据:一维 `T.Kernel(grid)` 的 grid 维数由运行时调度器分发到 48 个物理核,**块之间完全独立**(无跨块归约、无共享状态、无顺序依赖),故:

- 不需要"对齐物理核整数倍"(块数与核数不成整数倍不产生正确性或负载问题,尾部空闲核只是利用率问题,且块粒度 2048 元素 ≈ 4KB 足够小,负载均衡良好);
- 不需要"核内 `T.serial` 串行处理"循环来压缩核数 —— 该方案用于**动态边界**或**核内任务映射需静态化**的场景,本算子无此需求(边界虽为运行期 `tail_size`,但它由 `cid` 静态推导,不构成动态 shape)。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| `T.Parallel` | **不使用** | §1.6.2 已判定逐元素热点由向量算子 `T.vabs` 覆盖,无需 `T.Parallel` 展开 |
| `T.serial` | **不使用** | 每核只处理 1 个块,无核内任务循环(§5.3 判定无需) |
| `T.Pipelined` | **不使用** | 单步算子,无"搬运-计算"多阶段可重叠(UB 中转本身即一次搬入一次搬出,无 stage 可分) |
| `T.Persistent` | **不使用** | 无核常驻需求 |

**循环结构总结**:kernel 内**无显式循环** —— 唯一的数据并行由 `T.Kernel` 的 grid 维承担,块内计算由单条向量算子承担。

> 一致性核对:§1.6.2 判定 V1 已向量化,故 §6 中未出现对应的逐元素标量循环;V2–V4 三处 block/host 级标量在 §3.3 伪代码中保留且均非循环体。

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入(基于数据依赖分析) |
| 手工同步点 | **无** |
| 数据依赖性质 | 线性:`GM→UB`(copy) → `UB→UB`(vabs) → `UB→GM`(copy);同块内无跨 lane 通信、无跨核依赖 |
| 跨核同步 | 不需要(块间独立,§5.3) |

**正确性依据**:Developer 模式下 `T.alloc_ub` 的 UB 生命周期与访问顺序由编译器自动管理;本算子的依赖是线性的、无环、无 aliasing 冲突(`x_ub` 与 `y_ub` 为独立缓冲),自动同步可正确覆盖。

---

## §8 CV 融合设计

**n/a** —— 本算子为**纯 Vector**,不涉及 Cube 计算,无 `Scope("Cube")` / `Scope("Vector")` 协同、无 `T.sync_block_set/wait`、无 `PIPE_FIX`。§1.3 判定为"纯 Vector"与此一致。

---

## §9 验证方案

### §9.1 Golden 函数草案(依据 §0.1 语义,独立于 NPU 算法)

```python
def golden_abs(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现 —— 按 §0.1 语义。

    fp32 opmath 后回落原 dtype,与源仓测试基准
    (tests/ops/test_unary_math.py 的 test_abs: _randn + torch.abs) 一致。
    注意:本函数只依据语义,不复刻 NPU 算法 —— 保证验证独立性。
    """
    return torch.abs(x.float()).to(x.dtype)
```

**边界语义覆盖**(§0.1 表逐项):`+inf → +inf`、`-inf → +inf`、`NaN → NaN`、`-0.0 → +0.0`;`randn` 输入天然覆盖正负两侧(`-x` 与 `+x` 落在同一输出),另可加显式角点用例。

### §9.2 L0 门槛测试计划

| 用例 | N | dtype | 容差(atol / rtol) |
|---|---|---|---|
| smoke | 1 048 576 | float16 | 1e-3 / 1e-3 |
| smoke | 1 048 576 | bfloat16 | 1.6e-2 / 1.6e-2 |
| smoke | 1 048 576 | float32 | 1e-5 / 1e-5 |

容差来源:源仓 `tests/ops/test_unary_math.py: _get_tolerances`(逐字沿用)。

**L0 通过判据**:三个 dtype 的 smoke 用例全部通过 → 进入 L1/L2/Boundary 分层套件(由 Stage 3 `tilelang-op-develop` 承载)。

### §9.3 与既有实现的关系(需后续 Stage 裁决)

> 本算子在本仓**已有一份手工实现**且当前 Tier 2 通过,位于
> `examples/TileOPs/tileops/kernels/elementwise/abs/abs_kernel/abs.py`。
> 本设计与该实现**无冲突**:两者都是"一维 grid + UB tile + `T.vabs`(源在前) + 尾块切片"的结构,
> 且都遵循 K3/K11(去 `threads`、`threads×npt → block_size`)。差异仅在表述粒度:
> 本设计显式给出了分核三要素的实查记录(§5.3)与算法调研四问(§1.6.0),既有实现的注释中未展开。
> **本设计不修改该文件**;是否以本设计为准由 Stage 2 检视裁决。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| R1 | **`T.vabs` 参数顺序陷阱**(源在前、目标在后) | 高 | §3.2 已注明依据(源码 + 仓内调用);Stage 3 实现时须复核,写反会得到未定义结果而非报错 |
| R2 | 尾块非整除路径的实测覆盖 | 中 | 本算子 4 个 workload 的 N 均被 2048 整除,**非整除路径未被 workload 覆盖**;Stage 3 的 L1 套件须补 `N % 2048 != 0` 用例(既有实现的分层自查已含 N=1/2/10) |
| R3 | 整型 dtype 未覆盖 | 中 | manifest 声明 int8/int16/int32/int64/uint8,但源侧在 **op 层短路**(`_IntIdentityUnaryOp`);NPU 侧现为 float-only。**列为后续**,不在本次范围(与源侧 kernel 家族 `FloatUnaryKernel` 的 float-only 性质一致) |
| R4 | `inplace` 变体未迁移 | 低 | 源侧另有 `_register_unary_inplace_custom_op`;NPU 侧 op 层支持 `inplace=True`(拷贝回),未用到独立 custom_op。列为后续 |
| R5 | 技术约束检测结论 | — | 三维 Kernel ❌ 未使用;GEMM 非整除 ❌ 不适用;L0C 溢出 ❌ 不适用(无 Cube);GPU 专用 API ❌ 未使用 → **无技术约束冲突** |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/abs/abs_fwd_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/abs/abs_fwd_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**(§1.6.1 无采纳优化项) |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/...` | ✅ Stage 0 已产出 |
| 5 | npuir kernel + golden + 分层测试 | `examples/abs/abs_fwd_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |

**本次 Stage 1 仅产出第 1、2 项**(约束:不修改任何既有代码文件)。

