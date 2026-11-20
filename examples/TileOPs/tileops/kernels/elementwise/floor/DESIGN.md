# DESIGN.md — `FloorFwdOp`(TileLang-NPUIR)

- **project** = `floor`,**op_name** = `FloorFwdOp`,**family** = `elementwise`
- **迁移类型**:migration(harness 子模式)· **编程模式**:Developer
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`
- **`source_output_shape`**:与输入同形;kernel 层 1-D flat `(N,)`
- **golden 独立性声明**:§9.1 的 golden 依据 §0.1 语义实现,**未复刻 NPU 算法**
- **与 `ceil` 的关系**:本算子与同批迁的 `ceil` **结构同构**(都是一元取整、都因指令缺失而走 cast round-trip);
  本设计沿用 `ceil` 已验证的 **C2 饱和旁路 + NaN 传播(V2)** 方案,仅把 `round_mode` 由 `"ceil"` 换为 `"floor"`。
  详细推导与实测证据见 `examples/ceil/ceil_fwd_kernel/`(含 `repro/TRAP-vcmp-nan-non-ieee.py`)。

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | $y_i = \lfloor x_i \rfloor$(向 $-\infty$ 取整),逐元素一元映射,无跨元素依赖 | `FloorFwdKernel.op_func(x) = T.floor(T.cast(x, "float32"))`(`kernels/elementwise.py:2069-2078`) |
| I/O 契约 | 输出 shape **与输入逐维相同**;dtype 相同(`same_as(input)`);非 in-place | manifest `shape_rules`;`_base.py` 用 `empty_like` |
| 规约语义 | **无规约** | 计算图无 reduce |
| dtype 语义 | kernel 家族 `FloatUnaryKernel` **float-only**;`op_func` 内**显式升 fp32 再算**(GPU 侧原因:`hfloor` 不提供 `cutlass::half_t` 重载) | `:2069-2078` docstring 原文 |
| 边界语义 | $+\infty \to +\infty$、$-\infty \to -\infty$、`NaN → NaN`、$\pm 0 \to \pm 0$;均属 `T.floor` 算术语义 | `T.floor` 语义 |
| **整型语义** | manifest 声明整型 dtype,但 `FloorFwdOp` 继承 **`_IntIdentityUnaryOp`**:整型走 `_int_handler`(恒等克隆)→ **在 op 层短路,不构造 kernel** | `ops/elementwise/math_unary.py:145`;`_base.py:1146-1190` |
| host 侧语义 | 无预处理;仅 flatten(`N_total`)与 reshape 回原形 | `UnaryOp.__init__` / `_eager_forward` |

**完成标准核对**:已可写出与源算子数值一致的 golden —— `torch.floor(x.float()).to(dtype)`(§9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | 任意 rank 的 `float16 / bfloat16 / float32`,kernel 层 flatten 为 `(N_total,)` |
| 输出 | 与输入同 shape、同 dtype |
| 动态轴 | **无**;`N_total` 构造期确定 |

### §0.3 第二问:实现算法(How)

1. **入口签名**:`floor_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)`。
2. **Kernel 结构**:`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx`,`block_size = threads × num_per_thread = 2048`;一维 grid,每 block 处理 `block_size` 个连续元素。
3. **计算原语**:三策略(`direct` / `explicit_parallel` / `register_copy`(**默认**));逐元素体为 `y[idx] = T.floor(T.cast(x[idx], "float32"))` —— **先升 fp32 再取整**,结果再隐式落回 `out_dtype`。
4. **内存分配**:`register_copy` 用 `T.alloc_fragment((block_size,), dtype)` ×2;**无 SMEM**、无跨 block 共享。
5. **循环与调度**:`T.Parallel(threads, npt)`;无 `T.serial` / `T.Pipelined`。
6. **host 侧逻辑**:`UnaryOp.__init__` 构造 kernel;`_eager_forward` flatten → kernel → reshape。
7. **编译配置**:无 `pass_configs`;策略与 `threads/npt` 由 `default_config` 决定。

**数据流(GPU 视角)**:`GM → fragment(整块) → 逐元素 floor(fp32 域) → fragment → GM`。全部 buffer 有来源,无臆造步骤。

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| G1 | **register_copy 策略**(默认) | 合并访存、提高有效带宽 | 整块 `T.copy` 进 fragment → 计算 → 整块写回 | 大寄存器堆 + coalesced 访存 |
| G2 | **`threads × npt` 二维并行** | 摊薄调度开销 | `T.Parallel(256, 8)`,每 block 2048 元素 | CUDA block/thread 两级并行 |
| G3 | **按 dtype 调 `npt`** | fp16/bf16 用 `npt=8` 达 128-bit 宽访存 | `_strategy_npt(strategy, dtype)` | LDG.128 |
| G4 | 三策略编译期可选 | 按 shape/dtype 选最优 | `direct` / `explicit_parallel` / `register_copy` | — |
| G5 | autotune(`threads`×`npt` 搜索) | 找最优执行参数 | `autotune_configs` / `autotune()` | — |
| **G6** | **fp32 提升后取整** | 规避 `hfloor` 无 half 重载 | `T.floor(T.cast(x, "float32"))` —— **API 可用性驱动的写法,不是性能优化** | CUDA `floorf` |

> 本算子未使用:SMEM tiling、swizzle、warp shuffle、`T.gemm`、异步流水。

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| $\lfloor x \rfloor$ 数学语义、I/O 契约、边界/dtype 语义 | 语义 | **保留** | 语义层不可妥协 |
| G1 fragment 中转 | 优化 | **等价替换** | 映射表「register 累加 → L0C/UB 中间缓冲」→ UB tile + `T.copy` |
| G2 `threads × npt` 两级并行 | 优化 | **重新设计** | 「block × thread 两级并行 → 一维 block + 向量化元素并行」;本项目 `T.Kernel` 仅接受一维 block 数 |
| `T.Parallel(threads, npt)` | 优化 | **等价替换** | → 向量算子序列覆盖整 tile |
| G3 按 dtype 调 `npt` | 优化 | **等价替换** | → 向量宽度对齐(fp16/bf16 ×8、fp32 ×4),由 `block_size` 承载 |
| G4 三策略编译期可选 | 优化 | **重新设计** | npuir Developer 下 UB 驻留是唯一自然形态,三策略收敛为单一实现 |
| G5 autotune | 优化 | **舍弃** | K8:NPU 无 autotune。理由:NPU 侧编译开销高且无 autotune 框架 |
| **G6 fp32 提升后取整** | 优化 | **重新设计**(见 §0.6) | npuir **无一元 `floor`** 指令 → 必须换实现路径;该重设计引入范围限制与 NaN 分歧 |
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

#### 重设计项 3 —— **`floor` 的 API 替代路径 ⭐ 本设计的核心**

**源方案**:`T.floor(T.cast(x, "float32"))` —— CUDA 的 `floorf` 在**整个 fp32 值域**上定义。

**NPU 约束(已实测)**:本 npuir build **没有一元 `floor`**。核对 `tilelang/language/customize_npuir.py`:53 个 `npuir_*` 定义中 `ceil`/`floor`/`round`/`trunc`/`fix`/`nearbyint` **全部不存在**;但 **`npuir_cast` 支持 round mode**:`valid_round_mode = {"round","rint","floor","ceil","trunc","odd"}`。

**候选枚举与对比**

| 候选 | 形式 | 等价域 | 代价 | 判定 |
|---|---|---|---|---|
| C0 基线(理想) | `T.floor(x)` | 全 fp32 值域 | 1 op | **不可用**(指令缺失) |
| **C1 int32 round-trip** | `vcast(x→int32, round_mode="floor")` → `vcast(→fp32)` | $\lvert x\rvert < 2^{31}$(**非全值域**) | 2–4 op | 初版采纳,**已被路径 A 修订降为回退方案** |
| **C2 饱和旁路 + NaN 传播(采纳)** | $m_{ge}=\lvert x\rvert\ge 2^{31}$;$m_{ord}=$ 有序数;`y = vselect(m_ord, vselect(m_ge, x, C1(x)), x)` | **全 fp32 值域** | +5 op | **采纳** |
| C3 指数缩放 round-trip | 按 $\lvert x\rvert$ 的指数缩放后再取整 | 可扩到更大值域,但仍有上界且实现复杂 | +ops | **舍弃**:收益不确定、复杂度高 |

**C1 的等价性与边界**

- **等价域**:$\lvert x\rvert < 2^{31}$ 时与 `T.floor(fp32)` 逐位等价。
- **不等价域**:$\lvert x\rvert \ge 2^{31}$ 时 fp32→int32 **溢出**,取值未定义;而源算子在该区间有确定值($\lfloor x\rfloor = x$,因为 $\lvert x\rvert \ge 2^{23}$ 的 fp32 已无小数部分)。
- **另一类分歧**:`NaN` 经 `vcast(NaN→int32)` 得未定义值(实测得 `0`),而源算子 `floor(NaN) = NaN`。
- **⇒ 两类真实语义分歧**(值域 + NaN),均非"精度差异"。
- **承接来源(非静默)**:既有手写实现的 docstring 已显式记录范围限制(原文:"**Range limitation**: the int32 intermediate caps the representable magnitude at `|x| < 2**31` — same caveat as `ceil.py`"),并在测试契约中限定;但该实现**未处理 NaN**。

**采纳 C2 的依据(逐条,沿用 ceil 的实测结论)**

1. **裁决前提已实测成立**(本任务实测,同 ceil):UB 内 int32 溢出**不触发异常**,产出确定的回绕/饱和值(2^31)。依据:`examples/ceil/ceil_fwd_kernel/history_version/design_probe_int32_overflow.py`(`3.0e9 → 2147483648.0`,无异常)。
2. **NaN 判据已实测定案**(同 ceil):本硬件 `vcmp(x,x,"ne")` 与 `vcmp(x,c_nan,"ne")` **恒 false**;`vcmp(x,x,"eq")` 对 **NaN 为 false**、对有序数(含 ±inf)为 true。依据:`examples/ceil/ceil_fwd_kernel/repro/TRAP-vcmp-nan-non-ieee.py`(四断言)。
3. **选择子操作数顺序是硬约束**(同 ceil,已实测):`vselect(m_ord, tmp, x)` —— 有序数取取整结果、NaN 取 x。写反会让**所有正常值**绕过取整(实测 `0.5 → 0.5`)。
4. **代价可接受**:+5 op;UB 由 7 缓冲增至 10 缓冲 / 68 KB ≤ 113 KB 有效预算。
5. **`floor` 与 `ceil` 在阈值论证上完全同构**:两者的旁路条件都是 $\lvert x\rvert \ge 2^{31}$,且该区 fp32 已为整数,故 `floor(x) == x` 与 `ceil(x) == x` **同时成立** —— 同一旁路对两者都正确。

**等价性论证(全 fp32 值域)**:

| 输入类 | `m_ord` | `m_ge` | 取哪条 | 结果 | 与源算子 |
|---|---|---|---|---|---|
| `NaN` | false | (不参与) | `f_ub` | `NaN` | ✅ |
| `±inf` | true | true | `f_ub` | `±inf` | ✅ |
| $\lvert x\rvert \ge 2^{31}$(有限) | true | true | `f_ub` = x(该区 fp32 已为整数) | $x=\lfloor x\rfloor$ | ✅ |
| 其余 | true | false | `t_ub` = C1 结果 | $\lfloor x\rfloor$ | ✅ |

⇒ 四类输入逐类覆盖,全 fp32 值域与源算子等价。

#### 重设计项 4 —— 尾块处理

运行期切片 `tail_size = T.min(block_size, shape - offset)`,覆盖 `N % block_size != 0`;**语义保持论证**:切片只改变每块元素个数,不改变映射与取值。

### §0.7 迁移决策摘要

- **三问结论**:语义 = 逐元素 $\lfloor x\rfloor$、shape/dtype 保持、无规约、fp32 域计算;算法 = 一维 grid + `threads×npt` 块内映射 + 三策略;优化手段 = 6 项(G1–G6)。
- **耦合性判定统计**:**保留 1**、**等价替换 3**、**重新设计 3**、**舍弃 1**、**n/a 1**。
- **重设计项清单**:① 并行结构;② 策略收敛;③ **`floor` 的 API 替代路径(C2,含值域与 NaN 两类分歧的覆盖)**。
- **源优化意图承接**:G1 → UB tile 中转;G2 → `block_size` 一维划分;G3 → 向量宽度对齐;G4 → `block_size` 可配;G5 → **舍弃**;G6 → 由 **C2** 承接(取得全 fp32 值域等价)。**无静默丢弃。**

---

## §1 概述

### §1.1 算子概述

`FloorFwdOp`(floor)—— 逐元素向下取整 $y = \lfloor x\rfloor$。
**迁移后 NPU 侧算法**:一维 grid + UB tile 中转 + **int32 round-trip 取整 + 饱和旁路 + NaN 传播**。

### §1.2 计算语义

- 输入:$x \in \mathbb{R}^{N}$,dtype ∈ {float16, bfloat16, float32}
- 输出:$y_i = \lfloor x_i \rfloor$,同 shape 同 dtype
- 无规约、无跨元素依赖

### §1.3 算子特征分析

| 维度 | 判定 |
|---|---|
| 计算类型 | **纯 Vector** → 仅需 UB,无 Cube / 无 L1/L0 |
| 复杂度级别 | **单步**(单次搬运 + 少量向量算子,无循环、无中间落 GM) |
| 动态 shape | **无** |
| host 预处理 | **无** |

### §1.4 算法描述(依据 Phase R 调研选定算法)

对 $N$ 个元素按 `block_size` 切块,每块:

1. `T.copy` 把该块从 GM 搬入 UB `x_ub`;
2. 需要时先升到 fp32(`x_ub→f_ub`;fp32 直通);
3. **饱和旁路 + NaN 传播(C2 完整版)**:
   - `abs_ub = T.vabs(f_ub)` —— 取幅度;
   - `m_ge = T.vcmp(abs_ub, c_ub, "ge")` —— 标记 $|x| \ge 2^{31}$;
   - `m_ord = T.vcmp(f_ub, f_ub, "eq")` —— 标记**有序数**(true = 含 ±inf 的有序值,**false = NaN**);
   - `i_ub = T.vcast(f_ub, round_mode="floor")` → `t_ub = vcast(i_ub, round_mode="rint")` —— C1 取整;
   - `tmp_ub = T.vselect(m_ge, f_ub, t_ub)` —— $|x| \ge 2^{31}$ 时取 $x$(该带内 fp32 已无小数部分,$\lfloor x\rfloor = x$);
   - `abs_ub = T.vselect(m_ord, tmp_ub, f_ub)` —— **有序数取 `tmp_ub`;NaN 取 $x$ 本身**。⚠️ 操作数顺序是硬约束;
4. `T.copy` 把结果写回 GM。

⇒ 全 fp32 值域与源算子等价:**四类输入(NaN / ±inf / $\lvert x\rvert \ge 2^{31}$ / 其余)逐类覆盖**。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R —— 轻量调研)

> 深度分级:单步纯逐元素 → **轻量调研**(四问各一行结论 + 依据)。
> 调研范围:同一数学语义($y=\lfloor x\rfloor$)的算法族 + 本仓 elementwise 族既有实现(含同批迁的 `ceil`)+ `references/algorithm-candidates.md` 的 R1/R2 参考表。

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **无可更省的等价形式**(基线即最优) | 候选:① `-ceil(-x)` —— npuir **无 ceil**,不可用;② `trunc(x) - (x < trunc(x) ? 1 : 0)` —— npuir **无 trunc**,且需比较+选择,更贵;③ `round(x - 0.5)` —— **语义不等价**(`round` 为 banker's rounding;负半轴与 ties 反例:`floor(0.5)=0` vs `round(0.0)=0` 巧合,但 `floor(-0.5)=-1` 而 `round(1.0)=1`,取 `x=-0.75`:`floor=-1`,`round(-1.25)=-1` 巧合 —— 严格反例在 ties 上);④ 位运算/指数缩放见 §0.6 C3(舍弃)。基线 `vcast` round-trip 为 2–4 条指令,候选均不更省或不等价 → 基线胜出。 |
| **R2 在线算法** | **无** | 结构依据:输出每个元素只依赖对应输入元素,不存在 running 统计量或跨块依赖 → 不存在流式/单遍变体;本算法本身就是单遍。 |
| **R3 复杂度四口径** | 基线:**FLOPs = N**(1 op/elem)、**访存 = 2·N·elem_bytes**(读 1 遍 + 写 1 遍)、**扫描遍数 = 1**、**中间缓冲 = 0**(UB tile 驻留,不落 GM) | 与 manifest roofline 逐字一致(`flops: N`,`bytes: 2*N*elem_bytes`)—— **已在 roofline 下界**。C2 的额外 `vabs`/`vcmp`/`vselect` 属发射项,不改变流量项。 |
| **R4 硬件亲和** | 基线亲和:纯 Vector、只需 UB、无 Cube/分形约束;向量宽度 fp16/bf16 ×8、fp32 ×4,`block_size=2048` 对三者均整除;边界静态(尾块用运行期 `tail_size`)。**C2 的掩码路径**需 `vcmp`(uint8)+ `vselect`,属已验证原语;**C3 舍弃**。 | 逐候选评估见上 |

**调研结论**:选定算法族 = **int32 round-trip + C2 饱和旁路 + NaN 传播**,无更优替代。

**设计期 roofline 估算下界(D-2)**

```
流量项 = 2·N·elem_bytes / HBM带宽
发射项 = N·(每元素指令数) / 向量吞吐
容量项 = 0(单块 UB 驻留)
估算下界 = max(流量项, 发射项, 容量项) = 流量项
```

确定性结论:纯带宽受限;C2 增加的发射项不改变流量项主导地位,故不启用设计期探针(D-3 不适用)。
**实测校验(来自同批 `ceil`,本算子结构相同)**:实测带宽 ≈ **0.193 TB/s**(fp16/bf16)= 峰值 1.8 TB/s 的 **10.7%**,即**实际并非带宽受限而是每 tile 开销主导** —— 该 roofline 结论在实测下不成立,已记入 §10 R12。

#### §1.6.1 数学等价优化(公式级)

**结论:公式层无优化空间(显式结论 + 逐类依据)**

| 优化类别 | 结论 | 依据 |
|---|---|---|
| 恒等变形降代价 | 不适用 | 无除法/sqrt/log/exp;取整为单条指令语义,无分解余地 |
| 稳定化变形 | 不适用 | 无规约、无 exp,不存在上溢/下溢路径 |
| 公共子表达式消除 | 不适用 | 每元素仅参与一次计算 |
| 算子降代换 | **已在 §0.6 完成**(以 `vcast` round-trip 替代缺失的 `floor`),属**重设计层**而非公式层优化 | §0.6 重设计项 3 |
| 访存量削减 | 已最优 | 读 1 遍 + 写 1 遍 = roofline 下界;无中间落 GM、无重复读 |
| 归约结构优化 | 不适用 | 无规约 |

> 因**无采纳项**,按 skill Phase 2 第 5 步规则(「显式『无优化空间』结论除外」)`verify_equiv.py` **豁免**,不生成。
> ⚠️ **注意**:§0.6 C2 的**值域与 NaN 分歧**不属于"公式优化",不得用本节的豁免条款掩盖 —— 它们已在 §0.6 正面声明并在 §10 列为风险项。

#### §1.6.2 向量化替代分析(循环 / 标量消除)

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | 逐元素取整 | 逐元素循环 | **已向量化替代** | `T.vcast`(round_mode="floor"/"rint")—— `customize_npuir.py:723` 的 `npuir_cast` |
| V2 | `offset = cid * block_size` | block 级索引 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算**,每 block 1 个标量 |
| V3 | `tail_size = T.min(block_size, shape - offset)` | block 级边界 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算**;非动态 shape |
| V4 | `T.ceildiv(N, block_size)`(grid 维) | host 元数据 | **保留(不可替代)** | 理由类别:**host 侧 shape 元数据计算** |

**结论**:逐元素热点(V1)已完全向量化;保留的 3 处均为 block 级/host 级标量,**无逐元素标量循环残留**。

#### §1.6.3 向量化轴与数据布局决策

**单轴算子**:I/O 契约已把数据定为 1-D flat `(N,)`,核内布局无自由度。

- **候选轴枚举**:**仅 1 个** —— flat N 的连续轴。按 skill 豁免条款(「纯 GEMV / 单轴算子可写明『仅一个候选轴』豁免」)不展开多候选评分。
- **向量宽度对齐**:`block_size = 2048`;fp16/bf16 ×8 → 2048%8=0 ✓;fp32 ×4 → 2048%4=0 ✓;尾 lane 浪费率 0。
- **32B 对齐**:2048×2B=4KB(fp16)、2048×4B=8KB(fp32),均为 32B 整数倍 ✓。
- **决策落地**:主选布局 = 连续 1-D + `block_size=2048`,是 §3.3 伪代码与 §4 缓冲形状的唯一依据。

#### §1.6.4 前置实测与裁决记录(沿用 ceil,已生效)

本算子的 C2 方案依赖两个**已在同批 `ceil` 任务实测成立**的前提,本设计直接沿用其结论并保留复测入口:

| 前提 | 结论 | 实测来源 |
|---|---|---|
| UB 内 int32 溢出**不 fault**(回绕/饱和) | ✅ 成立(`3.0e9 → 2147483648.0`) | `examples/ceil/ceil_fwd_kernel/history_version/design_probe_int32_overflow.py` |
| NaN 判据 `vcmp(x,x,"eq")` 的 true 集 = 有序数(NaN 为 false) | ✅ 成立;`ne` 族恒 false 不可用 | `examples/ceil/ceil_fwd_kernel/repro/TRAP-vcmp-nan-non-ieee.py` |
| 选择子顺序 `vselect(m_ord, ceil, x)` 正确、写反破坏全部正常值 | ✅ 成立(V1 FAIL / V2 PASS) | `examples/ceil/ceil_fwd_kernel/history_version/design_probe_final_sequence.py` |

**回退条件**:若后续工具链变更导致溢出 lane 转为 fault,则改用 C1 + **契约层的域限制裁决**(不得在 DESIGN.md 内自行接受分歧);复测入口同上。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无**(纯 Vector) |
| 是否需要手工同步 | **否** —— 块内为"搬入 → 向量算子序列 → 搬出"的线性依赖,自动同步可正确处理 |
| 是否需要控制 L1/L0 层级 | **否**(只用 UB) |
| 是否需要跨核通信 | **否**(块间完全独立) |
| 与既有实现一致性 | 本仓 `kernels/elementwise/{abs,ceil,floor}` 同为 Developer |

---

## §3 API 映射设计

### §3.1 公式拆解

依据 §1.6.1(公式层无优化项):

| 步 | 计算 | 对应公式 |
|---|---|---|
| 1 | 取块:读入 `x[offset : offset+tail_size]` | 输入分块 |
| 2 | $y_j = \lfloor x_j \rfloor$ | §0.6 C2 路径 |
| 3 | 写回 `y[offset : offset+tail_size]` | 输出分块 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| grid 划分 | `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)` | 本仓 `kernels/elementwise/{abs,ceil}` 同形 |
| **UB 缓冲分配** | **`T.alloc_ub((block_size,), dtype)`** | 本仓同族实现 |
| GM → UB | `T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])` | 同上 |
| **升位(fp16/bf16)** | `T.vcast(x_ub, f_ub, round_mode="rint")`;fp32 直通(`T.copy`) | `customize_npuir.py:723-759` |
| **取整(fp32 直连)** | **`T.vcast(f_ub, i_ub, round_mode="floor")`** | 同上;⚠️ `round_mode` 只在**真发生 dtype 变化**的 `vcast` 上生效 |
| **取整后回落** | `T.vcast(i_ub, t_ub, round_mode="rint")` | 同上 |
| **取幅度(C2)** | `T.vabs(f_ub, abs_ub)` | `customize_npuir.py` 的 `npuir_abs(A,B)` |
| **阈值比较(C2)** | `T.vcmp(abs_ub, c_ub, m_ge, "ge")` — **两操作数都须为缓冲** | `customize_npuir.py` 的 `npuir_cmp` |
| **有序数判据(C2)** | **`T.vcmp(f_ub, f_ub, m_ord, "eq")`** — true 集 = **有序数**(含 ±inf);**对 NaN 为 false** | 实测见 §1.6.4;⚠️ `ne` 族在本硬件恒 false,不可用 |
| **旁路选择(C2)** | `T.vselect(m_ge, f_ub, t_ub, tmp_ub)`;**`T.vselect(m_ord, tmp_ub, f_ub, out_ub)`** — `Out = Cond ? A : B`;**顺序是硬约束** | `customize_npuir.py` 的 `npuir_select` |
| **常量缓冲构造(C2)** | `for j in T.Parallel(block_size): c_ub[j] = 2**31` | `vcmp` 不接受标量操作数 |
| UB → GM | `T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])` | 本仓同族实现 |
| 尾块 | `T.min(block_size, shape - offset)` | 本仓 `logsumexp` 的 `real_m` 同形 |

> ⚠️ **API 陷阱(已实测,必须遵守)**:
> 1. `round_mode` **只在真正发生 dtype 变化的 `vcast` 上生效**;fp32 必须直接 `vcast` 到 int32,不得插入冗余的同 dtype `vcast`(会**静默退化为截断**)。
> 2. 选择子操作数顺序:**`vselect(m_ord, tmp_ub, f_ub)`** —— `m_ord` 为真表示**有序数**,此时必须取**取整结果**;写反会让**所有正常值**绕过取整(实测 `0.5 → 0.5`)。
> 3. 通道顺序:npui 向量算子是**源在前、目标在后**。

### §3.3 伪代码

```python
def floor_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    out_dtype = output_dtype or dtype
    compute_dtype, idx_dtype, MAG = "float32", "int32", 2 ** 31

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype),
                 y: T.Tensor((N,), out_dtype),
                 shape: T.int32):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                x_ub    = T.alloc_ub((block_size,), dtype)
                f_ub    = T.alloc_ub((block_size,), compute_dtype)
                abs_ub  = T.alloc_ub((block_size,), compute_dtype)
                c_ub    = T.alloc_ub((block_size,), compute_dtype)
                t_ub    = T.alloc_ub((block_size,), compute_dtype)
                i_ub    = T.alloc_ub((block_size,), idx_dtype)
                tmp_ub  = T.alloc_ub((block_size,), compute_dtype)
                m_ge    = T.alloc_ub((block_size,), "uint8")
                m_ord   = T.alloc_ub((block_size,), "uint8")
                y_ub    = T.alloc_ub((block_size,), out_dtype)

                offset = cid * block_size                      # V2 保留(block 级)
                tail_size = T.min(block_size, shape - offset)  # V3 保留(block 级)
                for j in T.Parallel(block_size):
                    c_ub[j] = MAG

                T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                # 1) widen to fp32 (fp32 goes straight through)
                if dtype == compute_dtype:
                    T.copy(x_ub, f_ub)
                else:
                    T.vcast(x_ub, f_ub, round_mode="rint")

                # 2) C2 saturation bypass + NaN propagation
                T.vabs(f_ub, abs_ub)                        # |x|
                T.vcmp(abs_ub, c_ub, m_ge, "ge")            # |x| >= 2**31
                # `eq` is true for ORDERED values (incl. +-inf), false for NaN.
                # Do NOT invert the select operands below.
                T.vcmp(f_ub, f_ub, m_ord, "eq")
                T.vcast(f_ub, i_ub, round_mode="floor")     # C1 floor (junk where overflow/NaN)
                T.vcast(i_ub, t_ub, round_mode="rint")
                T.vselect(m_ge, f_ub, t_ub, tmp_ub)         # |x| >= 2**31 -> x itself
                T.vselect(m_ord, tmp_ub, f_ub, abs_ub)      # ordered -> tmp ; NaN -> x

                # 3) narrow to the output dtype
                if out_dtype == compute_dtype:
                    T.copy(abs_ub, y_ub)
                else:
                    T.vcast(abs_ub, y_ub, round_mode="rint")

                T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])
        return main
    return _func
```

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| **同批已走完整流程的同构算子(最直接参考)** | `examples/ceil/ceil_fwd_kernel/ceil_fwd_kernel.py` + `DESIGN.md` + `REVIEW.md` |
| 一元逐元素 + `T.alloc_ub` + 分层套件 | `examples/TileOPs/tileops/kernels/elementwise/abs/abs_kernel/abs.py` |
| 既有 floor NPU 实现(走 C1,**非**本设计形态,只读) | `examples/TileOPs/tileops/kernels/elementwise/floor/floor_kernel/floor.py` |
| `vcmp`/`vselect`/`vcast` 定义与 round_mode 合法集 | `tilelang/language/customize_npuir.py:723-759` |
| NaN 判据与选择子顺序的实测 repro | `examples/ceil/ceil_fwd_kernel/repro/TRAP-vcmp-nan-non-ieee.py` |

---

## §4 数据规格与内存规划

### §4.1 数据规格

| 张量 | shape | dtype | 所在 | 分配 API |
|---|---|---|---|---|
| `x` | `(N,)` | float16 / bfloat16 / float32 | GM | `T.Tensor` 入参 |
| `y` | `(N,)` | 同 `x` | GM | `T.Tensor` 出参(`out_idx=[1]`) |
| `shape` | 标量 | int32 | 入参 | `T.int32` 运行期标量 |
| `x_ub` | `(block_size,)` | 同 `x` | UB | `T.alloc_ub` |
| `f_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(升位后的 x) |
| `abs_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(先承 `\|x\|`,后承最终结果) |
| `c_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(**常量缓冲,承载 $2^{31}$**) |
| `t_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(承 C1 取整结果) |
| `i_ub` | `(block_size,)` | int32 | UB | `T.alloc_ub` |
| `tmp_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(承饱和旁路 `vselect` 的中间结果) |
| `m_ge` | `(block_size,)` | **uint8** | UB | `T.alloc_ub`(`vcmp` 输出必须是 uint8) |
| `m_ord` | `(block_size,)` | **uint8** | UB | `T.alloc_ub`(**有序数判据**;对 NaN 为 false) |
| `y_ub` | `(block_size,)` | 同 `y` | UB | `T.alloc_ub` |

> **内存层级 ↔ 分配 API 固定对照**:GM = `T.Tensor`;UB = **`T.alloc_ub`**;L1 = `T.alloc_shared` / `T.alloc_L1`;L0C = `T.alloc_L0C`。
> **本算子只用 UB**。误写 `T.alloc_shared` 会降到 L1,使 `UB→GM` store 编译失败:`'hivm.hir.store' op only support store ub to gm currently!`。

### §4.2 搬运路径

```
GM(x) --T.copy--> UB(x_ub) --vcast/vabs/vcmp/vselect 序列--> UB(y_ub) --T.copy--> GM(y)
```

无 L1 / L0 参与;无 GM 中间结果落盘。

### §4.3 UB 容量预算

| buffer | 尺寸表达式 | fp32 峰值 |
|---|---|---|
| x_ub | block_size × elem_bytes | 8 KB |
| f_ub | block_size × 4 | 8 KB |
| abs_ub | block_size × 4 | 8 KB |
| c_ub | block_size × 4 | 8 KB |
| t_ub | block_size × 4 | 8 KB |
| i_ub | block_size × 4 | 8 KB |
| tmp_ub | block_size × 4 | 8 KB |
| m_ge | block_size × 1 | 2 KB |
| m_ord | block_size × 1 | 2 KB |
| y_ub | block_size × elem_bytes | 8 KB |

**合计**:68 KB(fp32、`block_size = 2048`)。

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

**结论:三选一命中「无需适配依据」**。依据:一维 `T.Kernel(grid)` 由运行时调度器分发到 48 个物理核,**块间完全独立**(无跨块归约、无共享状态、无顺序依赖);块粒度 2048 元素 ≈ 4KB 足够小、负载均衡良好;边界虽为运行期 `tail_size`,但它由 `cid` 静态推导,不构成动态 shape。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| `T.Parallel` | **仅用于常量缓冲初始化**(`for j in T.Parallel(block_size): c_ub[j] = MAG`) | §1.6.2 判定逐元素热点由向量算子覆盖,无需展开 |
| `T.serial` | **不使用** | 每核仅 1 块,无核内任务循环 |
| `T.Pipelined` | **不使用** | 单步算子,无多阶段可重叠 |
| `T.Persistent` | **不使用** | 无核常驻需求 |

> 一致性核对:§1.6.2 判定 V1 已向量化,故 §6 无对应逐元素标量循环;V2–V4 三处 block/host 级标量在 §3.3 保留且均非循环体。

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入(基于数据依赖分析) |
| 手工同步点 | **无** |
| 依赖性质 | 线性:`GM→UB`(copy) → `UB→UB`(向量算子序列) → `UB→GM`(copy);同块内无跨 lane 通信、无跨核依赖 |
| 跨核同步 | 不需要(块间独立) |

**正确性依据**:Developer 模式下 `T.alloc_ub` 的 UB 生命周期与访问顺序由编译器自动管理;本算子依赖线性、无环、各 buffer 独立无 aliasing。

---

## §8 CV 融合设计

**n/a** —— 纯 Vector,不涉及 Cube 计算,无 `Scope("Cube")`/`Scope("Vector")` 协同、无 `T.sync_block_set/wait`、无 `PIPE_FIX`。

---

## §9 验证方案

### §9.1 Golden 函数草案(独立于 NPU 算法)

```python
def golden_floor(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现 —— 按 §0.1 语义。

    fp32 域取整后回落原 dtype,与源仓参考实现口径一致
    (源 op_func 即 T.floor(T.cast(x, "float32")))。只依据语义实现,
    不复刻 NPU 侧的 int32 round-trip —— 保证验证独立性。
    """
    x_cpu = x.detach().cpu()
    return torch.floor(x_cpu.to(torch.float32)).to(x_cpu.dtype)
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
| **`floor` 的 ties 与跨符号** | `0.5 / -0.5 / 1.0 / -1.0 / 2.5 / -2.5 / ±0.0` | ties 是取整语义的核心;`floor(-0.5) = -1` 与 `ceil(-0.5) = 0` 的**负半轴差异**须显式验证 |
| **NaN 传播** | `NaN` | §0.1 声明必须保持;C2 的 `m_ord` 一路专为它而设 |
| **±inf 传播** | `+inf / -inf` | C1 单跑下会变成 `±2.147e9`,C2 修好;**必须验证** |
| **超 int32 范围的有限值** | `\|x\| \in [2^{31}, 2^{32}]` | 值域分歧的原始触发点;C2 的 `m_ge` 一路专为它而设 |
| $2^{23}$ 附近 | `\|x\|` 略大于 / 略小于 $2^{23}$ | fp32 由"有小数部分"转为"已是整数"的语义分界参考点 |

> ⚠️ 边界语义既为"必须保持",就应落在 **L1(阻塞层)**,不宜放到 L2/Boundary(告警层)。

### §9.3 与既有实现的关系(需后续 Stage 裁决)

> 本算子已有手工 NPU 实现且测试通过(`floor_kernel/floor.py`)。该实现走 **C1 路径**(`vcast(x→int32, round_mode="floor")`),其 docstring 已显式记录范围限制(原文:"**Range limitation**: the int32 intermediate caps the representable magnitude at `|x| < 2**31` —— same caveat as `ceil.py`"),但**未处理 NaN**,故在下列三类边界上不正确(实测同 ceil):

| 项 | 既有实现(实测同 ceil 的 C1 路径) | 本设计(C2 完整版) |
|---|---|---|
| $\lvert x\rvert \ge 2^{31}$ | ❌ 变 `±2.147e9` | ✅ 旁路取 $x$ |
| `±inf` | ❌ 变 `±2.147e9` | ✅ 旁路取 $x$ |
| **`NaN`** | ❌ **变 `0`** | ✅ `m_ord` 路取 $x$ |

> ⚠️ **既有实现的三项边界缺陷是已交付代码里的真实缺陷**(其测试 `tests/ops/test_floor.py` 完全不含 NaN/±inf/超范围用例,故未暴露)。**本设计不修改该文件**;该项已同步登记为 §10 R10,须在后续 Stage 或独立修复中处理。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| **R1** | **值域分歧已在设计中被 C2 覆盖**,但**既有实现仍带该缺陷** | 中 | §0.6 与 §9.3 已声明;C2 的等价性有同批 ceil 的实测支撑(§1.6.4) |
| **R2** | **`round_mode` 只在真发生 dtype 变化的 `vcast` 上生效** | 高 | §3.2 已注明;fp32 必须直接 `vcast` 到 int32,不得先经冗余 fp32→fp32 `vcast`(会静默退化为截断) |
| **R3** | **C2 依赖「UB 内 int32 溢出只回绕、不触发异常」** | **已实测消解**(同批 ceil) | 实测来源 §1.6.4;工具链变更后可用同一探针复测 |
| R4 | `T.vcast`/`T.vabs` 的参数顺序(源在前) | 高 | §3.2 已注明依据;写反不报错、只出错值 |
| R5 | 尾块非整除路径的实测覆盖 | 中 | manifest 的 4 个 workload 的 N 均被 2048 整除 → **非整除路径未被 manifest 覆盖**;Stage 3 的 L1 必须补 |
| R6 | 整型 dtype 未覆盖 | 中 | manifest 声明整型,但源侧在 **op 层短路**(`_IntIdentityUnaryOp`);NPU 侧为 float-only。**列为后续** |
| **R7** | **`NaN` 传播依赖本硬件的非 IEEE `vcmp` 行为**(`eq` 对 NaN 为 false;`ne` 族恒 false) | 中 | 判据有实测依据(§1.6.4);**Stage 3 须把 NaN 用例放进 L1(阻塞层)** |
| R8 | 技术约束检测结论 | — | 三维 Kernel ❌ 未使用;GPU 专用 API ❌;GEMM 非整除 ❌ 不适用;L0C 溢出 ❌ 不适用 → **无技术约束冲突** |
| **R9** | **选择子操作数顺序是硬约束** | 高 | `vselect(m_ord, tmp_ub, f_ub)`;写反会让**所有正常值**绕过取整(实测 `0.5→0.5`)。§3.3 已给行内注释,Stage 3 须照抄 |
| **R10** | **既有已交付实现存在三项边界缺陷(同 ceil)**:`\|x\|\ge 2^{31}` / `±inf` / `NaN`;其测试无此类用例故未暴露。**这是集成侧代码里的真实缺陷,不是设计问题。** | **高** | ① 本设计(C2 完整版)已覆盖;② **本设计不修改既有文件**,故集成侧缺陷**仍然存在**,须由后续 Stage 3 按本设计重写 kernel、或独立修复项处理;③ 已在 §9.3 与本节交叉引用 |
| **R11** | **C2 的缓冲复用**:`abs_ub` 先承 `\|x\|` 后承最终结果 | 中 | 两次使用**串行且活跃区间不重叠**(`vabs` 的读出被 `vcmp` 消费完之后 `vselect` 才写入);Stage 3 须复核,冲突时改独立缓冲(68 KB → 76 KB,仍 ≤ 113 KB) |
| **R12** | **§1.6.0 的"纯带宽受限"结论被实测否证**:同批 ceil 实测带宽 ≈ 0.193 TB/s = 峰值 1.8 TB/s 的 **10.7%**,说明实际由**每 tile 开销**主导而非带宽 | 中 | 本算子结构相同,预期同量级;Stage 4(调优)不在 harness 迁移范围内,故该性能缺口**已知且未处理**,须如实记录而非掩盖 |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/floor/floor_fwd_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/floor/floor_fwd_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**(§1.6.1 公式层无采纳优化项;C2 的等价性由**同批 ceil 的实测**承载,见 §1.6.4) |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/…` | ✅ Stage 0 已产出 |
| 5 | npuir kernel + golden + 分层测试 | `examples/floor/floor_fwd_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |

