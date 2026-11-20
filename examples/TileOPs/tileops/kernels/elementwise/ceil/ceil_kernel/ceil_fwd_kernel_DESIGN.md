# DESIGN.md — `CeilFwdOp`(TileLang-NPUIR)

- **project** = `ceil`,**op_name** = `CeilFwdOp`,**family** = `elementwise`
- **迁移类型**:migration(harness 子模式)· **编程模式**:Developer
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`
- **`source_output_shape`**:与输入同形;kernel 层 1-D flat `(N,)`
- **golden 独立性声明**:§9.1 的 golden 依据 §0.1 语义实现(源仓参考实现口径),**未复刻 NPU 算法**
- **本设计与 abs 的主要差异**:NPU 侧不存在一元 `ceil` 指令,必须经 int32 round-trip 表达 —— 该路径**引入了源算子没有的范围限制**(见 §0.6、§10),是本设计的核心取舍。

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | $y_i = \lceil x_i \rceil$(向 $+\infty$ 取整),逐元素一元映射,无跨元素依赖 | `CeilFwdKernel.op_func(x) = T.ceil(T.cast(x, "float32"))`(`kernels/elementwise.py:2081-2090`) |
| I/O 契约 | 输出 shape **与输入逐维相同**;dtype 相同(`same_as(input)`);非 in-place | manifest `shape_rules`;`_base.py` 的 `register_fake` 用 `empty_like` |
| 规约语义 | **无规约**(无归约轴、无累加顺序问题) | 计算图无 reduce |
| dtype 语义 | kernel 家族 `FloatUnaryKernel` **float-only**;`op_func` 内**显式升 fp32 再算**(GPU 侧原因:`hceil` 不提供 `cutlass::half_t` 重载) | `:2081-2090` 的 docstring 原文 |
| 边界语义 | $+\infty \to +\infty$、$-\infty \to -\infty$、`NaN → NaN`、$\pm 0 \to \pm 0$;均属 `T.ceil` 算术语义 | `T.ceil` 语义 |
| **整型语义** | manifest 声明 `int8/int16/int32/int64/uint8`,但 `CeilFwdOp` 继承 **`_IntIdentityUnaryOp`**:整型走 `_int_handler`(恒等克隆)→ **在 op 层短路,不构造 kernel** | `ops/elementwise/math_unary.py:152`;`_base.py:1146-1190` |
| host 侧语义 | 无预处理;仅 flatten(`N_total`)与 reshape 回原形 | `UnaryOp.__init__` / `_eager_forward` |

**完成标准核对**:已可写出与源算子数值一致的 golden —— `torch.ceil(x.float()).to(dtype)`(§9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | 任意 rank 的 `float16 / bfloat16 / float32`,kernel 层 flatten 为 `(N_total,)` |
| 输出 | 与输入同 shape、同 dtype |
| 动态轴 | **无**;`N_total` 构造期确定 |
| 输出 shape 确认 | **与输入同形**(逐元素),已从 `op_func` 的逐元素性与 `empty_like` 确认 |

### §0.3 第二问:实现算法(How)

1. **入口签名**:`ceil_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)` → `kernel(threads_arg, npt_arg)` 或 `kernel(threads_arg)`。
2. **Kernel 结构**:`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx`,`block_size = threads × num_per_thread = 2048`。一维 grid,每 block 处理 `block_size` 个连续元素。
3. **计算原语**:三策略(`direct` / `explicit_parallel` / `register_copy`(**默认**));逐元素体为 `y[idx] = T.ceil(T.cast(x[idx], "float32"))` —— **先升 fp32 再取整**,结果再隐式落回 `out_dtype`。
4. **内存分配**:`register_copy` 用 `T.alloc_fragment((block_size,), dtype)` ×2(fragment 中转),**无 SMEM**、无跨 block 共享。
5. **循环与调度**:`T.Parallel(threads, npt)`;无 `T.serial` / `T.Pipelined`。
6. **host 侧逻辑**:`UnaryOp.__init__` 构造 kernel;`_eager_forward` flatten → kernel → reshape。
7. **编译配置**:无 `pass_configs`;策略与 `threads/npt` 由 `default_config` 决定。

**数据流(GPU 视角)**:`GM → fragment(整块) → 逐元素 ceil(fp32 域) → fragment → GM`。全部 buffer 有来源,无臆造步骤。

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| G1 | **register_copy 策略**(默认) | 合并访存、提高有效带宽 | 整块 `T.copy` 进 fragment → 计算 → 整块写回 | 大寄存器堆 + coalesced 访存 |
| G2 | **`threads × npt` 二维并行** | 摊薄调度开销 | `T.Parallel(256, 8)`,每 block 2048 元素 | CUDA block/thread 两级并行 |
| G3 | **按 dtype 调 `npt`** | fp16/bf16 用 `npt=8` 达 128-bit 宽访存 | `_strategy_npt(strategy, dtype)` | LDG.128 |
| G4 | 三策略编译期可选 | 按 shape/dtype 选最优 | `direct` / `explicit_parallel` / `register_copy` | — |
| G5 | autotune(`threads`×`npt` 搜索) | 找最优执行参数 | `autotune_configs` / `autotune()` | — |
| **G6** | **fp32 提升后取整** | 规避 `hceil` 无 half 重载 | `T.ceil(T.cast(x, "float32"))` —— **这是 API 可用性驱动的写法,不是性能优化** | CUDA `ceilf` |

> 本算子未使用:SMEM tiling、swizzle、warp shuffle、`T.gemm`、异步流水。

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| $\lceil x \rceil$ 数学语义、I/O 契约、边界/dtype 语义 | 语义 | **保留** | 语义层不可妥协(`migration-analysis.md` §5.1) |
| G1 fragment 中转 | 优化 | **等价替换** | §5.3 映射表「register 累加 → L0C/UB 中间缓冲」→ UB tile + `T.copy` |
| G2 `threads × npt` 两级并行 | 优化 | **重新设计** | §5.3「block × thread 两级并行 → 一维 block + 向量化元素并行」;本项目 `T.Kernel` 仅接受一维 block 数 |
| `T.Parallel(threads, npt)` | 优化 | **等价替换** | → 向量算子序列覆盖整 tile |
| G3 按 dtype 调 `npt` | 优化 | **等价替换** | → 向量宽度对齐(fp16/bf16 ×8、fp32 ×4),由 `block_size` 承载 |
| G4 三策略编译期可选 | 优化 | **重新设计** | npuir Developer 下 UB 驻留是唯一自然形态,三策略收敛为单一实现;性能意图由 G1 的等价替换承接 |
| G5 autotune | 优化 | **舍弃** | K8:NPU 无 autotune;`init_config(config)` 不含 `tune`。**理由**:本项目 NPU 侧编译开销高且无 autotune 框架,启发式配置是既定约定 |
| **G6 fp32 提升后取整** | 优化 | **重新设计**(见 §0.6) | npuir 无一元 `ceil` 指令 → 必须换实现路径;**该重设计引入范围限制,是本设计的核心取舍** |
| SMEM tiling / swizzle / warp shuffle | — | **n/a** | 源码未使用 |

### §0.6 NPU 算法重设计

#### 重设计项 1 —— 并行结构(G2)

- **源方案**:CUDA 三级并行(grid/block/thread)。
- **NPU 新算法**:一维 `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)`,每核处理连续 `block_size` 个元素;块内不做线程映射,整块搬入 UB 后由**向量指令**处理;尾块用运行期 `tail_size = T.min(block_size, N - offset)` 切片。
- **语义保持论证**:逐元素映射,**无规约、无重排、无累加顺序** → 并行度变化不改变任何元素的取值路径,数学恒等。

#### 重设计项 2 —— 策略收敛(G4)

- **源方案**:三套 kernel 体按 config 编译期选择。
- **NPU 新算法**:收敛为单一实现 = 源 `register_copy` 的等价替换形态(UB tile 中转 + 向量算子)。**G4 的性能意图承接**:NPU 侧 `block_size` 仍是可配置量,三策略差异退化为"UB tile 大小"这一个自由变量。
- **语义保持论证**:同重设计项 1。

#### 重设计项 3 —— **`ceil` 的 API 替代路径(G6)⭐ 本设计的核心取舍**

**源方案**:`T.ceil(T.cast(x, "float32"))` —— CUDA 的 `ceilf` 在**整个 fp32 值域**上定义。

**NPU 约束(已实测)**:本 npuir build **没有一元 `ceil`**。已亲自核对的指令集:
`tilelang/language/customize_npuir.py` 的一元算子集合为
`abs/add/and/arange/atomic_add/bitcast/brc/cast/clamp/clear/cmp/concat/cumsum/deinterleave/div/dot/exp/exp2/fill/flip/gather/interleave/ln/load_nd2nz/log2/max/min/mul/not/or/pad/pow/print/rec/reduce/relu/reshape/rsqrt/select/shl/shr/sigmoid/sqrt/store_*/sub/transpose/vcos/verf/vsin/vtanh/xor` —— **无 `ceil`/`floor`/`round`/`trunc`**。
但 **`npuir_cast` 支持 round mode**:`valid_round_mode = {"round","rint","floor","ceil","trunc","odd"}`(`customize_npuir.py:740`)。

**候选枚举与对比**

| 候选 | 形式 | 等价域 | 代价 | 判定 |
|---|---|---|---|---|
| **C0 基线(理想)** | `T.ceil(x)` | 全 fp32 值域 | 1 op | **不可用**(指令缺失) |
| **C1 int32 round-trip** | `vcast(x→int32, round_mode="ceil")` → `vcast(→fp32)` | $\lvert x\rvert < 2^{31}$(**非全值域**) | 2 op(+低位 dtype 的升降位步) | 初版采纳,**已被路径 A 修订降为回退方案** |
| **C2 饱和旁路(全值域精确)** | $m = \lvert x\rvert \ge 2^{31}$;`y = vselect(m, x, C1(x))` | **全 fp32 值域** | **+3 op**(`vabs` + `vcmp` + `vselect`)+ 3 buffer | **采纳(路径 A 修订后)** —— 见下 |
| C3 指数缩放 round-trip | 按 $\lvert x\rvert$ 的指数缩放后再取整 | 可扩到更大值域,但仍有上界且实现复杂度高 | +ops(需指数提取/还原) | **舍弃**:收益不确定且显著增加复杂度 |

**C1 的等价性与边界(必须明示,不得含糊)**

- **等价域**:$\lvert x\rvert < 2^{31}$ 时,int32 round-trip 与 `T.ceil(fp32)` **逐位等价**(fp32 在该区间可精确转 int32 并取整回写)。
- **不等价域**:$\lvert x\rvert \ge 2^{31}$ 时 fp32→int32 **溢出**,取值由硬件行为决定、**未定义**;而源算子在此区间仍有确定值($\lceil x\rceil = x$,因为 $\lvert x\rvert \ge 2^{23}$ 的 fp32 已无小数部分)。
- **⇒ 这是一条真实的语义分歧,不是"精度差异"**。源算子无此限制,$\lvert x\rvert \ge 2^{31}$ 的输入在本设计下**结果不确定**。
- **承接来源(非静默)**:该限制在**既有手写实现的模块 docstring 中已显式记录**(`examples/TileOPs/tileops/kernels/elementwise/ceil/ceil_kernel/ceil.py:25-29`),并在**测试契约**中限定:`tests/ops/test_ceil.py:8` 原文「kept inside the representable magnitude range (randn-scaled)」,且**全文件无任何超出 int32 范围的断言**(已逐行核对)。故该分歧是**被文档化、被测试域限定**的,而非遗漏。

**修订记录(路径 A:采纳 C2)**

> **触发**:Stage 2 检视判 **不通过**(阻塞项:NPU 重设计的语义保持论证是**限定域**的,而全值域等价的 C2 已被设计且可行;「测试契约已限定域」不构成放弃语义层保全的理由)。
> **修订前置实测(已执行,见 `history_version/design_probe_int32_overflow.py`)**:
>
> ```
> [in-range]  x=1000.5    y[0]=1001.0        expect 1001.0  OK
> [overflow]  x=3.000e+09 y[0]=2147483648.0  no fault(回绕/饱和到 2^31,未触发异常)
> VERDICT: overflowing lanes do NOT fault -> C2 (saturation bypass) is implementable
> ```
>
> **实测结论**:UB 内 int32 溢出**不触发异常**,而是产出确定的回绕/饱和值(2^31)。⇒ **C2 的裁决前提成立**,采纳 C2 可**完全消除**值域分歧。

**C2 的算子序列(比初版估计更省:仅 +3 op)**

```
f_ub   = widen(x_ub)                     # 仅低位 dtype 需要;fp32 直接复用
abs_ub = vabs(f_ub)                      # ① 幅度
mask   = vcmp(abs_ub, c_ub, "ge")         # ② uint8 掩码(|x| >= 2^31)
ceil_f = C1(f_ub)                        # ③ C1 取整(溢出处产出 2^31,随后被丢弃)
y_ub   = vselect(mask, f_ub, ceil_f)     # ④ |x| >= 2^31 时取 x(该带内 fp32 已无小数部分,ceil(x)==x)
```

- **`y = vselect(m_ge, x, ceil)`** 与 `|= vabs`、`= vcmp` 共 **3 个额外向量算子 + 3 个额外 UB buffer**(`abs_ub` / `m_ge` / 常量缓冲 `c_ub`),而非初版估计的 +4~5 op;NaN 那一路再加 **1 个 `vcmp` + 1 个选择子操作数反转**(掩码 `m_ord`),合计相对纯 C1 **+5 op**。
- ⚠️ **`vcmp` 的两个操作数都必须是缓冲**(本仓实测:传 Python 标量会报 `'int' object has no attribute 'buffer'`)→ 常量 $2^{31}$ 必须物化为 `c_ub` 缓冲。
- ⚠️ `vselect(Cond, A, B, Out)` 语义为 `Out = Cond ? A : B`,掩码为 `vcmp` 产出的 uint8 —— **可直接复用同一掩码,无需 ones/zeros 与第二个掩码**(初版估计多算了这一步)。

**C1 作为回退方案的保留条件**:若后续工具链变更导致"溢出 lane 会 fault"(可用同一探针复测),则改用 C1 + **契约层的域限制裁决**(不得在 DESIGN.md 内自行接受分歧)。

> **对本修订的声明**:采纳 C2 后,本实现在**整个 fp32 值域上与源算子 `T.ceil` 等价**;`|x| >= 2^31` 不再是分歧带而是由旁路精确覆盖。原风险项(§10 R1)相应降级。

#### 重设计项 3 续 —— NaN 传播(Stage 2 round 2 发现,第三类分歧)

> **触发**:Stage 2 **round 2** 判 **不通过**。检视方独立推演并实跑取证,发现 C1 与 C2 **均把 NaN 变为 0**,而源算子与 golden(`torch.ceil`)传播 NaN。检视方初稿提出的"翻转掩码极性即可修复"**被其实测推翻**,故要求先探明本硬件上真正可用的 NaN 判据。

**实测一:各变体对 NaN / ±inf / 超范围值的实际输出**(`history_version/design_probe_nan_polarity.py`)

| 输入 | C1 单跑 | C2(ge 极性) | lt 极性 | 源算子 |
|---|---|---|---|---|
| `NaN` | `0` ❌ | `0` ❌ | `0` ❌ | `NaN` |
| `+inf` | `2.147e9` ❌ | `+inf` ✅ | `+inf` ✅ | `+inf` |
| `-inf` | `-2.147e9` ❌ | `-inf` ✅ | `-inf` ✅ | `-inf` |
| `3e9` | `2.147e9` ❌ | `3e9` ✅ | `3e9` ✅ | `3e9` |

⇒ C2 修好了值域与 ±inf;但 **NaN 在所有极性与变体下均为 `0`**。极性之所以无效:本环境 `vcmp` 对 NaN **非 IEEE**(IEEE 下 `NaN >= c` 与 `NaN < c` 同为 false),`lt` 表现为 `ge` 的取反,故 NaN 只是被从一条 C1 分支挪到另一条。

**实测二:NaN 判据的可用性**(`history_version/design_probe_nan_detect.py`,输入集 `[NaN, +inf, -inf, +0.0, -0.5, +0.5, 3e9]`)

| 判据 | 在输入集上的 true 集 | 可用性 |
|---|---|---|
| A `vcmp(x, x, "ne")` | `{}`(恒 false) | ❌ 不可用(IEEE 应为 NaN→true) |
| **B `vcmp(x, x, "eq")`** | `{所有有序数}`(**对 NaN 为 false**) | ✅ **可用** —— 其**补集**恰为 `{NaN}` |
| C `¬(x≥0) ∧ ¬(x<0)` | `{NaN}` | ✅ 亦可,但需 2 `vcmp` + 2 `vselect` |
| D `vcmp(x, c_nan, "ne")` | `{}`(恒 false) | ❌ 不可用 |

> ⚠️ **勘误(round 3 修订)**:本条早先曾写成「判据 B 的 true 集恰为 `{NaN}`」—— **该表述是错的**,源自查时读错了探针的打印:`detect` 探针的 `eq` 分支输出的是 `out = eq ? 0 : 1`(即 **`not eq`**),因此它打印的 `[1,0,…]` 含义是「**`eq` 对 NaN 为 false**」而非「true 集为 `{NaN}`」。
> **该错误被 ED-C 转正后的 repro 用一条断言抓出**(`repro/TRAP-vcmp-nan-non-ieee.py` 的断言 B 失败),并经端到端探针 `history_version/design_probe_final_sequence.py` 定案。**这正是"转正 + 断言"相对"只看探针打印"的价值。**

**采纳判据 B,但语义与选择子顺序必须按下述写法**(硬约束,写反会破坏**所有正常值**):

```
m_ord = vcmp(f_ub, f_ub, "eq")     # true = 有序数(含 ±inf);false = NaN
```

**实测三:端到端序列判定**(`history_version/design_probe_final_sequence.py`,输入 7 例)

| 序列 | NaN | +inf | -inf | 3e9 | 0.5 | -0.5 | 1.0 | 结论 |
|---|---|---|---|---|---|---|---|---|
| V1 `vselect(m_ord, f_ub, sat)`(早先写法) | ❌ `0.0` | ✅ | ✅ | ✅ | ❌ `0.5` | ❌ `-0.5` | ✅ | **FAIL** —— 对有序数取了原始 `x`,**整算子失效** |
| **V2 `vselect(m_ord, sat, f_ub)`(采纳)** | ✅ `NaN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **PASS** |

⇒ **采纳 V2**:`m_ord` 为真(有序数)时取**饱和旁路结果** `sat`;为假(NaN)时取 `x` 本身,从而恢复 NaN 传播。

**修订后的完整序列(在 C2 之上再加 NaN 一路,共 +5 op)**

```
f_ub   = widen(x_ub)                        # 仅低位 dtype;fp32 直通
abs_ub = vabs(f_ub)                         # |x|
m_ge   = vcmp(abs_ub, c_ub, "ge")           # |x| >= 2^31
m_ord  = vcmp(f_ub, f_ub, "eq")             # true = 有序数(含 ±inf),false = NaN
i_ub   = vcast(f_ub, round_mode="ceil")     # C1 取整(溢出/NaN lane 无效,随后被丢弃)
t_ub   = vcast(i_ub, → fp32)                # ceil 结果
sat_ub = vselect(m_ge,  f_ub, t_ub)         # |x| >= 2^31 -> 取 x(该区 fp32 已为整数)
abs_ub = vselect(m_ord, sat_ub, f_ub)       # 有序数 -> sat;NaN -> x 本身(NaN 传播)
y_ub   = narrow(abs_ub)
```

**代价**:相对纯 C1 共 **+5 op**(1 `vabs` + 2 `vcmp` + 2 `vselect`);UB 由 7 缓冲增至 **10 缓冲 / 68 KB**(明细见 §4.3),仍 ≤ 113 KB 有效预算。

**等价性论证(全 fp32 值域)**:

| 输入类 | `m_ord` | `m_ge` | 取哪条 | 结果 | 与源算子 |
|---|---|---|---|---|---|
| `NaN` | false | (不参与) | `f_ub` | `NaN` | ✅ 一致 |
| `±inf` | true | true | `f_ub` | `±inf` | ✅ 一致 |
| `\|x\| >= 2^31`(有限) | true | true | `f_ub` = x(该区 fp32 已为整数) | `x` = `ceil(x)` | ✅ 一致 |
| `\|x\| < 2^31` | true | false | `t_ub` = C1 结果 | `ceil(x)` | ✅ 一致 |

⇒ 四类输入**逐类覆盖**,全 fp32 值域与源算子等价。

#### 重设计项 4 —— 尾块处理

运行期切片 `tail_size = T.min(block_size, N - offset)`,覆盖 `N % block_size != 0`;**语义保持论证**:切片只改变每块元素个数,不改变映射与取值。

### §0.7 迁移决策摘要

- **三问结论**:语义 = 逐元素 $\lceil x\rceil$、shape/dtype 保持、无规约、fp32 域计算;算法 = 一维 grid + `threads×npt` 块内映射 + 三策略;优化手段 = 6 项(G1–G6)。
- **耦合性判定统计**:**保留 1**(语义层整体)、**等价替换 3**(fragment 中转 / `T.Parallel` 形态 / npt 宽度对齐)、**重新设计 3**(两级并行 → 一维;三策略 → 单实现;**G6 `ceil` 实现路径**)、**舍弃 1**(autotune)、**n/a 1**(SMEM/swizzle)。
- **重设计项清单**:① 并行结构(一维 Kernel + 向量块内处理 + 尾块切片);② 策略收敛(`block_size` 承接 G4);③ **`ceil` 的 API 替代路径 —— 采纳 C2(饱和旁路),取得全 fp32 值域等价**(§0.6 重设计项 3)。
- **源优化意图承接**:G1 → UB tile 中转;G2 → `block_size` 一维划分;G3 → 向量宽度对齐;G4 → `block_size` 可配;G5 → **舍弃**(理由见 §0.5);G6 → 由 **C2 饱和旁路**承接(取得**全 fp32 值域等价**;触发、实测与采纳过程见 §0.6 重设计项 3)。**无静默丢弃。**

---

## §1 概述

### §1.1 算子概述

`CeilFwdOp`(ceil)—— 逐元素向上取整 $y = \lceil x\rceil$。用于整数化前处理、量化相关算子。
**迁移后 NPU 侧算法**:一维 grid + UB tile 中转 + **int32 round-trip 取整**(与源结构不同之处见 §0.6)。

### §1.2 计算语义

- 输入:$x \in \mathbb{R}^{N}$,dtype ∈ {float16, bfloat16, float32}
- 输出:$y_i = \lceil x_i \rceil$,同 shape 同 dtype
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
2. 需要时先升到 fp32(`x_ub→f_ub`);
3. **饱和旁路 + NaN 传播(C2 完整版)**:
   - `abs_ub = T.vabs(f_ub)` —— 取幅度;
   - `m_ge = T.vcmp(abs_ub, c_ub, "ge")` —— 标记 $|x| \ge 2^{31}$(uint8 掩码);
   - `m_ord = T.vcmp(f_ub, f_ub, "eq")` —— 标记**有序数**:true = 含 ±inf 在内的有序值,**false = NaN**;
   - `i_ub = T.vcast(f_ub, round_mode="ceil")` 后回落到 `t_ub` —— C1 取整(溢出/NaN lane 的取值随后被丢弃);
   - `sat_ub = T.vselect(m_ge, f_ub, t_ub)` —— $|x| \ge 2^{31}$ 时取 $x$(该带内 fp32 已无小数部分,$\lceil x\rceil = x$);
   - `abs_ub = T.vselect(m_ord, sat_ub, f_ub)` —— **有序数取 `sat_ub`;NaN 取 $x$ 本身**,恢复 NaN 传播。⚠️ 操作数顺序是硬约束:写反会让**所有正常值**都绕过取整(实测 V1 在 `0.5` 上得 `0.5`);
4. `T.copy` 把结果写回 GM。

⇒ 该序列在**整个 fp32 值域**上与源算子 $T.ceil$ 等价:**四类输入(NaN / ±inf / $\lvert x\rvert \ge 2^{31}$ / 其余)逐类覆盖**,无残留分歧。

无逐元素标量循环(§1.6.2 已确认该点被向量算子覆盖)。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R —— 轻量调研)

> 深度分级:单步纯逐元素 → 按 skill 规则执行**轻量调研**(四问各一行结论 + 依据)。
> 调研范围:同一数学语义($y=\lceil x\rceil$)的算法族 + 本仓 elementwise 族既有实现 + `references/algorithm-candidates.md` 的 R1/R2 参考表。

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **无可更省的等价形式**(基线即最优) | 候选:① `-floor(-x)` —— npuir **无 floor**,不可用(同 §0.6 指令集核对);② `trunc(x) + (x > trunc(x) ? 1 : 0)` —— npuir **无 trunc**,且需比较+选择,更贵;③ `round(x + 0.5)` —— **语义不等价**(`round` 为 banker's rounding,`ceil(0.5)=1` 而 `round(1.0)=1` 巧合,`ceil(-0.5)=0` 而 `round(0.0)=0` 巧合,但 `ceil(2.5)=3` 而 `round(3.0)=3` 巧合、`ceil(0.25)=1` 而 `round(0.75)=1` 巧合 —— 组合反例:`ceil(-2.5) = -2` 而 `round(-2.0) = -2` 巧合;取 `x=-0.25`:`ceil=-0`? no —— 严格说 `round(x+0.5)` 在负半轴与 ties 上均不等价,故弃)。④ 位运算/指数缩放见 §0.6 C3(舍弃)。基线 `vcast` round-trip 为 2–4 条指令,候选均不更省或不等价 → 基线胜出。 |
| **R2 在线算法** | **无** | 结构依据:输出每个元素只依赖对应输入元素,不存在 running 统计量或跨块依赖 → **不存在**流式/单遍变体可谈;本算法本身就是单遍。 |
| **R3 复杂度四口径** | 基线:**FLOPs = N**(1 op/elem,取整计 1)、**访存 = 2·N·elem_bytes**(读 1 遍 + 写 1 遍)、**扫描遍数 = 1**、**中间缓冲 = 0**(UB tile 驻留,不落 GM) | 与 manifest roofline 逐字一致(`flops: N`,`bytes: 2*N*elem_bytes`)—— **已在 roofline 下界**。注意:C1 的额外 `vcast` 是**发射项**开销,不改变流量项。 |
| **R4 硬件亲和** | 基线亲和:纯 Vector、只需 UB(峰值见 §4.3)、无 Cube/分形约束;向量宽度 fp16/bf16 ×8、fp32 ×4,`block_size=2048` 对三者均整除;边界静态(尾块用运行期 `tail_size`)。**C2 的额外掩码路径**:需 `vcmp`(uint8 掩码)+ `vselect`,属已验证原语;**C3 舍弃**(收益不确定、复杂度高)。 | 逐候选评估见上 |

**调研结论**:选定算法族 = **int32 round-trip(§0.6 C1)**,无更优替代。调研范围已写明(R1 四候选 + R2 结构判据 + R3 与 roofline 对齐 + R4 逐候选)。

**设计期 roofline 估算下界(D-2)**

```
流量项 = 2·N·elem_bytes / HBM带宽
发射项 = N·(每元素指令数) / 向量吞吐     ← C1 的每元素指令数略高于 1
容量项 = 0(单块 UB 驻留)
估算下界 = max(流量项, 发射项, 容量项) = 流量项
```

确定性结论:纯带宽受限;C1 相对 C0 增加的发射项不改变流量项主导地位,故不启用设计期探针(D-3 不适用)。

#### §1.6.1 数学等价优化(公式级)

**结论:公式层无优化空间(显式结论 + 逐类依据)**

| 优化类别 | 结论 | 依据 |
|---|---|---|
| 恒等变形降代价 | 不适用 | 无除法/sqrt/log/exp;取整为单条指令语义,无分解余地 |
| 稳定化变形 | 不适用 | 无规约、无 exp,不存在上溢/下溢路径 |
| 公共子表达式消除 | 不适用 | 每元素仅参与一次计算 |
| 算子降代换 | **已在 §0.6 完成**(以 `vcast` round-trip 替代缺失的 `ceil`),属**重设计层**而非公式层优化 | §0.6 重设计项 3 |
| 访存量削减 | 已最优 | 读 1 遍 + 写 1 遍 = roofline 下界;无中间落 GM、无重复读 |
| 归约结构优化 | 不适用 | 无规约 |

> 因**无采纳项**,按 skill Phase 2 第 5 步规则(「显式『无优化空间』结论除外」)`verify_equiv.py` **豁免**,不生成。
> ⚠️ **注意**:§0.6 C1 的**范围分歧**不属于"公式优化",不得用本节的豁免条款掩盖 —— 它已在 §0.6 正面声明,并在 §10 列为风险项。

#### §1.6.2 向量化替代分析(循环 / 标量消除)

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | 逐元素取整 | 逐元素循环 | **已向量化替代** | `T.vcast`(round_mode="ceil"/"rint")—— `customize_npuir.py:723` 的 `npuir_cast` |
| V2 | `offset = cid * block_size` | block 级索引 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算**,每 block 1 个标量 |
| V3 | `tail_size = T.min(block_size, N - offset)` | block 级边界 | **保留(不可替代)** | 理由类别:**block 级索引 / 任务映射计算**;非动态 shape(`shape` 为传入的运行期标量,按 cid 静态推导) |
| V4 | `T.ceildiv(N, block_size)`(grid 维) | host 元数据 | **保留(不可替代)** | 理由类别:**host 侧 shape 元数据计算**,不在 kernel 内 |

**结论**:逐元素热点(V1)已完全向量化;保留的 3 处均为 block 级/host 级标量,**无逐元素标量循环残留**。

#### §1.6.3 向量化轴与数据布局决策

**单轴算子**:I/O 契约已把数据定为 1-D flat `(N,)`,核内布局无自由度。

- **候选轴枚举**:**仅 1 个** —— flat N 的连续轴。按 skill 豁免条款(「纯 GEMV / 单轴算子可写明『仅一个候选轴』豁免」)不展开多候选评分。
- **向量宽度对齐**:`block_size = 2048`;fp16/bf16 ×8 → 2048%8=0 ✓;fp32 ×4 → 2048%4=0 ✓;**尾 lane 浪费率 0**。
- **32B 对齐**:2048×2B=4KB(fp16)、2048×4B=8KB(fp32),均为 32B 整数倍 ✓。
- **决策落地**:主选布局 = 连续 1-D + `block_size=2048`,是 §3.3 伪代码与 §4 缓冲形状的唯一依据。

#### §1.6.4 备选方案与实验裁决计划(C2:全值域精确路径)

> 按 skill「实验裁决模式」要求,当采纳项与备选项的判定依赖**未在本项目实测的常数/前提**时,须给出备选方案的结构设计与裁决计划。本处依赖的前提 = **「UB 内 int32 溢出只回绕、不触发异常」**(未实测)。

**备选方案 C2(结构完整,达到 Stage 4 可直接实现的深度)**

| 项 | 设计 |
|---|---|
| buffer | `x_ub(dtype)` / `f_ub(fp32)` / `abs_ub(fp32)` / `c_ub(fp32)` / `t_ub(fp32)` / `i_ub(int32)` / `tmp_ub(fp32)` / `m_ge(uint8)` / `m_ord(uint8)` / `y_ub(out_dtype)` —— **10 个 `T.alloc_ub`**,均 `(block_size,)` |
| UB 预算 | fp32 峰值 = **68 KB**(见 §4.3);≤ 113 KB 有效预算 |
| 循环结构 | 无循环;UB 内向量算子序列:① `vcast(x→f)`(低位 dtype;fp32 直通)② `vabs(f→abs)` ③ `vcmp(abs, c_ub, m_ge, "ge")` ④ **`vcmp(f, f, m_ord, "eq")`**(有序数判据;对 NaN 为 false)⑤ C1 取整(`vcast→int32` 再回落 `t_ub`)⑥ `vselect(m_ge, f, t, tmp)`(饱和旁路)⑦ **`vselect(m_ord, tmp, f, out)`**(有序数→取整结果;NaN→x 本身)⑧ `vcast(out→y)`(低位 dtype) |
| 分核三要素 | 同主选(① `ceildiv(N,2048)` ② 物理核实查 48 ③ 判定同 §5.3) |
| 代价 | 相对纯 C1:**+5 op/元素**(1 `vabs` + 2 `vcmp` + 2 `vselect`);UB 由 7 缓冲增至 **10 缓冲 / 68 KB**(≤ 113 KB 有效预算) |

**裁决计划**

- **代表 shape**:`[1048576]`(smoke)+ 含超范围值的构造用例(`|x| ∈ [2^31, 2^32]`,当前测试套件**不含**此类)
- **测量指标**:① 数值正确性——超范围输入下 `y == x` 是否成立;② 是否触发运行时异常(UB 溢出安全性)
- **判定阈值**:若超范围输入下 C1 结果非确定值或 C2 能稳定给出 `y == x`,且 C2 的 latency 增幅 `< 15%`,则翻转至 C2
- **回写路径**:裁决结果写回本设计 §0.6 与 §1.6.4,并修订 §10 的范围风险项
- **shape 特化**:本算子工厂带 `lru_cache`,可按"输入是否可能超范围"分派,但 kernel 内无法预知输入值 → 分派不可行,只能全局二选一

**裁决结果(已执行)**

该前提**已实测成立** —— 见 `history_version/design_probe_int32_overflow.py` 的实跑输出:

```
[in-range]  x=1000.5    y[0]=1001.0        expect 1001.0  OK
[overflow]  x=3.000e+09 y[0]=2147483648.0  no fault(回绕/饱和到 2^31,未触发异常)
VERDICT: overflowing lanes do NOT fault -> C2 (saturation bypass) is implementable
```

⇒ **C2 已升为主选**(§0.6 重设计项 3),C1 降为**回退方案**(仅在工具链变更导致溢出 lane 转为 fault 时启用,并须配合契约层的域限制裁决)。本小节保留作为该裁决的完整记录与复测入口。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无**(纯 Vector) |
| 是否需要手工同步 | **否** —— 块内为"搬入 → 向量算子序列 → 搬出"的线性依赖,自动同步可正确处理 |
| 是否需要控制 L1/L0 层级 | **否**(只用 UB) |
| 是否需要跨核通信 | **否**(块间完全独立) |
| 与既有实现一致性 | 本仓 `kernels/elementwise/{abs,mish,lerp_tensor}` 同为 Developer |

**结论**:Developer 可完整表达该算法;Expert 会引入无收益的复杂度。

---

## §3 API 映射设计

### §3.1 公式拆解

依据 §1.6.1(公式层无优化项):

| 步 | 计算 | 对应公式 |
|---|---|---|
| 1 | 取块:读入 `x[offset : offset+tail_size]` | 输入分块 |
| 2 | $y_j = \lceil x_j \rceil$,$j \in [0, tail\_size)$ | §0.6 C1 路径 |
| 3 | 写回 `y[offset : offset+tail_size]` | 输出分块 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| grid 划分 | `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)` | 本仓 `kernels/elementwise/abs/abs_kernel/abs.py` 同形 |
| **UB 缓冲分配** | **`T.alloc_ub((block_size,), dtype)`** | 本仓 `kernels/elementwise/{abs,ceil}/…` 既有形态 |
| GM → UB | `T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])` | 同上 |
| **取整(fp32 直连)** | **`T.vcast(x_ub, i_ub, round_mode="ceil")`** | `customize_npuir.py:723-759`(`npuir_cast`,round_mode 合法集含 `"ceil"`) |
| **升位(fp16/bf16)** | `T.vcast(x_ub, f_ub, round_mode="rint")` | 同上 |
| **取整后回落** | `T.vcast(i_ub, f_ub, round_mode="rint")`;`T.vcast(f_ub, y_ub, round_mode="rint")` | 同上 |
| UB → GM | `T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])` | 本仓同族实现 |
| 尾块 | `T.min(block_size, shape - offset)` | 本仓 `logsumexp` 的 `real_m` 同形 |
| **取幅度(C2)** | `T.vabs(f_ub, abs_ub)` | `customize_npuir.py` 的 `npuir_abs(A,B)`;仓内调用 `examples/deepseek_v4/example_act_quant_kernel.py:61` |
| **阈值比较(C2)** | `T.vcmp(abs_ub, c_ub, m_ge, "ge")` — **两操作数都须为缓冲** | `customize_npuir.py` 的 `npuir_cmp`;`c_ub` 为常量缓冲 |
| **有序数判据(C2)** | **`T.vcmp(f_ub, f_ub, m_ord, "eq")`** — true 集 = **有序数**(含 ±inf);**对 NaN 为 false**。其补集才是 `{NaN}` | 实测:`history_version/design_probe_nan_detect.py`;⚠️ `vcmp(x,x,"ne")` 与 `vcmp(x,c_nan,"ne")` 在本硬件**恒 false**,不可用 |
| **旁路选择(C2)** | `T.vselect(m_ge, f_ub, t_ub, tmp_ub)`;**`T.vselect(m_ord, tmp_ub, f_ub, out_ub)`** — `Out = Cond ? A : B`;**操作数顺序是硬约束** | `customize_npuir.py` 的 `npuir_select`;顺序错误的后果见 `history_version/design_probe_final_sequence.py` 的 V1(在 `0.5` 上得 `0.5`) |
| **常量缓冲构造(C2)** | `for j in T.Parallel(block_size): c_ub[j] = 2**31` | `vcmp` 不接受标量操作数,常量必须物化 |

> ⚠️ **API 陷阱(已实测,必须遵守)**:`round_mode` **只在真正发生 dtype 变化的 `vcast` 上生效**。若先把 fp32 经一次冗余的 fp32→fp32 `vcast`,后续 fp32→int32 会**静默退化为截断**(`0.5 → 0` 而非 `1`)。故 **fp32 必须直接 `vcast` 到 int32**,只有低精度 dtype 才需要先升位到 fp32。依据:`examples/TileOPs/tileops/kernels/elementwise/ceil/ceil_kernel/ceil.py:91-96` 的实测注记。
> ⚠️ **通道顺序**:npui 向量算子是**源在前、目标在后**(`T.vcast(A, B)` 即 `B = cast(A)`)。

### §3.3 伪代码

```python
def ceil_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    idx_dtype = "int32"
    MAG = 2 ** 31          # saturation threshold (C2)

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

                # constant buffer: vcmp needs BOTH operands to be buffers
                for j in T.Parallel(block_size):
                    c_ub[j] = MAG

                T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                # 1) widen to fp32
                if dtype == compute_dtype:
                    T.copy(x_ub, f_ub)
                else:
                    T.vcast(x_ub, f_ub, round_mode="rint")

                # 2) C2 saturation bypass + NaN propagation
                T.vabs(f_ub, abs_ub)                        # |x|
                T.vcmp(abs_ub, c_ub, m_ge, "ge")            # |x| >= 2**31
                # this build's `eq` is true for ORDERED values (incl. +-inf) and
                # FALSE for NaN -- do NOT invert the select operands: writing
                # vselect(m_ord, f, sat) makes every normal value bypass the
                # ceil (measured: 0.5 -> 0.5).  See design_probe_final_sequence.py
                T.vcmp(f_ub, f_ub, m_ord, "eq")             # true iff ordered (not NaN)
                T.vcast(f_ub, i_ub, round_mode="ceil")      # C1 ceil; overflow/NaN lanes are junk
                T.vcast(i_ub, t_ub, round_mode="rint")      # t = ceil_fp32
                # |x| >= 2**31 -> take x itself (already integral, ceil(x) == x)
                T.vselect(m_ge, f_ub, t_ub, tmp_ub)
                # ordered -> the ceil result; NaN -> x itself (restores propagation)
                T.vselect(m_ord, tmp_ub, f_ub, abs_ub)

                # 3) narrow to the output dtype
                if out_dtype == compute_dtype:
                    T.copy(abs_ub, y_ub)
                else:
                    T.vcast(abs_ub, y_ub, round_mode="rint")

                T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])
        return main
    return _func
```

> **别名说明**:`abs_ub` 在第 1 步承 `|x|`、第 2 步末尾承 `vselect` 的结果 —— 两次使用**串行且活跃区间不重叠**(`vabs` 的读出完成后 `vcmp` 才消费,而 `vselect` 写作时 `vabs` 的结果已被 `vcmp` 读完)。`t_ub` 只被写一次(`vcast` 回落)后即被 `vselect` 读走。**Stage 3 须复核该复用无 aliasing 冲突**,冲突时改为独立缓冲(UB 预算仍充裕,见 §4.3)。

> buffer 形状与向量维与 §1.6.3 主选布局一致;无逐元素标量循环,符合 §1.6.2 结论。

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| 一元逐元素 + `T.alloc_ub` + 分层套件(最接近) | `examples/TileOPs/tileops/kernels/elementwise/abs/abs_kernel/abs.py` |
| **同算子的既有 NPU 实现(含 int32 round-trip 与陷阱注记)** | `examples/TileOPs/tileops/kernels/elementwise/ceil/ceil_kernel/ceil.py` |
| `T.cast`/`vcast` 的 round_mode 合法集与语义 | `tilelang/language/customize_npuir.py:723-759`(`npuir_cast`) |
| 尾块切片 `T.min(block, total - offset)` 既有形态 | `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/…` |

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
| `abs_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(先承 `\|x\|`,后承 `vselect` 结果 —— 活跃区间不重叠,见 §3.3) |
| `c_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(**常量缓冲,承载 $2^{31}$** —— `vcmp` 不接受标量操作数) |
| `t_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(承 C1 取整结果) |
| `i_ub` | `(block_size,)` | int32 | UB | `T.alloc_ub` |
| `tmp_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(承饱和旁路 `vselect` 的中间结果) |
| `m_ge` | `(block_size,)` | **uint8** | UB | `T.alloc_ub`(`vcmp` 输出必须是 uint8;标记 $\lvert x\rvert \ge 2^{31}$) |
| `m_ord` | `(block_size,)` | **uint8** | UB | `T.alloc_ub`(**有序数判据**:true = 含 ±inf 的有序值;**对 NaN 为 false**;其补集为 `{NaN}`) |
| `y_ub` | `(block_size,)` | 同 `y` | UB | `T.alloc_ub` |

> **内存层级 ↔ 分配 API 固定对照(逐条取值,不得自行填写)**:GM = `T.Tensor`;UB = **`T.alloc_ub`**;L1 = `T.alloc_shared` / `T.alloc_L1`;L0C = `T.alloc_L0C`。
> **本算子只用 UB**(§1.3 判定"纯 Vector")。误写成 `T.alloc_shared` 会降到 L1,使 Vector 侧 `UB→GM` store 编译失败:`error: 'hivm.hir.store' op only support store ub to gm currently!`(该文本不含 buffer 名与 API 名)。

### §4.2 搬运路径

```
GM(x) --T.copy--> UB(x_ub) --vcast×2~4--> UB(y_ub) --T.copy--> GM(y)
```

无 L1 / L0 参与;无 GM 中间结果落盘。fp32 与 fp16/bf16 两条分支使用**互斥的** UB 子集。

### §4.3 UB 容量预算

每核按需分配下列 UB 缓冲(两条 dtype 分支使用互斥子集):

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

**结论**:`block_size = 2048` 在采纳 C2 完整版(含 NaN 传播)后仍无容量压力(68 KB ≤ 113 KB),不需要为容量缩小它,也不需要为 §3.3 的缓冲复用做妥协。若 Stage 3 发现 `abs_ub` 的复用存在 aliasing 冲突而改为独立缓冲,峰值仍仅 76 KB,仍 ≤ 113 KB。

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

**Kernel 内运行期切片**(非 host padding+crop):`tail_size = T.min(block_size, shape - offset)`。
源/目的切片共用 `tail_size`;UB 中超出 `tail_size` 的尾 lane 不参与写出,故无需 `T.clear`。
依据:本仓 `logsumexp` 的 `real_m` 为同一模式的既有实现。

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

> **查询记录**:经 `torch_npu` 的 `get_device_properties(0)` 实查,`vector_core_num = 48`、`cube_core_num = 24`、设备名 `Ascend910B2C`。未使用假设值。(旁证:本仓 `kernels/norm/ada_layer_norm` 实现注释记为 `get_aicore_num() * 2`,与 48 一致。)

**③ 规模判定与分核方案**

| workload | 逻辑核数 | 物理核数 | 倍数 | 判定 | 方案 |
|---|---|---|---|---|---|
| smoke-1m | 512 | 48 | 10.7× | 中等规模 | **无需适配** |
| 16m | 8192 | 48 | 170.7× | 中等规模 | 无需适配 |
| 64m | 32 768 | 48 | 682.7× | 中等规模 | 无需适配 |
| 256m | 131 072 | 48 | 2730.7× | 极大规模 | 无需核内 `T.serial` |

**结论:三选一命中「无需适配依据」**。依据:一维 `T.Kernel(grid)` 的 grid 由运行时调度器分发到 48 个物理核,**块间完全独立**(无跨块归约、无共享状态、无顺序依赖);块数不整除核数不产生正确性问题,且块粒度 2048 元素 ≈ 4KB 足够小、负载均衡良好;边界虽为运行期 `tail_size`,但它由 `cid` 静态推导,不构成动态 shape,故不需要核内 `T.serial` 压缩核数。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| `T.Parallel` | **不使用** | §1.6.2 判定逐元素热点由 `vcast` 覆盖,无需展开 |
| `T.serial` | **不使用** | 每核仅 1 块,无核内任务循环(§5.3 判定无需) |
| `T.Pipelined` | **不使用** | 单步算子,无多阶段可重叠 |
| `T.Persistent` | **不使用** | 无核常驻需求 |

**总结**:kernel 内**无显式循环** —— 数据并行由 `T.Kernel` 的 grid 维承担,块内计算由 `vcast` 序列承担。

> 一致性核对:§1.6.2 判定 V1 已向量化,故 §6 无对应逐元素标量循环;V2–V4 三处 block/host 级标量在 §3.3 保留且均非循环体。

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入(基于数据依赖分析) |
| 手工同步点 | **无** |
| 依赖性质 | 线性:`GM→UB`(copy) → `UB→UB`(vcast 链) → `UB→GM`(copy);同块内无跨 lane 通信、无跨核依赖 |
| 跨核同步 | 不需要(块间独立) |

**正确性依据**:Developer 模式下 `T.alloc_ub` 的 UB 生命周期与访问顺序由编译器自动管理;本算子依赖线性、无环、`x_ub`/`f_ub`/`i_ub`/`y_ub` 为独立缓冲(注:`f_ub` 在非 fp32 分支被复用两次 —— 先 fp32 承接入参、后承接 int32 回写;**该复用是既有实现的既有形态**,本设计沿用并在 Stage 3 复核其无 aliasing 冲突)。

---

## §8 CV 融合设计

**n/a** —— 纯 Vector,不涉及 Cube 计算,无 `Scope("Cube")`/`Scope("Vector")` 协同、无 `T.sync_block_set/wait`、无 `PIPE_FIX`。

---

## §9 验证方案

### §9.1 Golden 函数草案(独立于 NPU 算法)

```python
def golden_ceil(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现 —— 按 §0.1 语义。

    fp32 域取整后回落原 dtype,与源仓参考实现口径一致
    (源 op_func 即 T.ceil(T.cast(x, "float32")))。只依据语义实现,
    不复刻 NPU 侧的 int32 round-trip —— 保证验证独立性。
    """
    x_cpu = x.detach().cpu()
    return torch.ceil(x_cpu.to(torch.float32)).to(x_cpu.dtype)
```

**边界语义覆盖**(§0.1 逐项):`+inf → +inf`、`-inf → -inf`、`NaN → NaN`、`±0 → ±0`;有符号幅度(0.5/-0.5、1.0/-1.0、2.5/-2.5 等)。

### §9.2 L0 门槛测试计划

| 用例 | N | dtype | 容差(atol/rtol) |
|---|---|---|---|
| smoke | 1 048 576 | float16 | 1e-3 / 1e-3 |
| smoke | 1 048 576 | bfloat16 | 1.6e-2 / 1.6e-2 |
| smoke | 1 048 576 | float32 | 1e-5 / 1e-5 |

容差逐字沿用源仓 `_get_tolerances`。

**L0 通过判据**:三个 dtype 的 smoke 全过 → 进入 L1/L2/Boundary 分层套件(由 Stage 3 承载)。

**L1(阻塞层)必须包含的用例 —— 逐条对应 §0.1 声明为"必须保持"的边界语义**:

| 类别 | 用例 | 为什么放在**阻塞层** |
|---|---|---|
| 非整除 shape | `N % 2048 != 0`(如 N=1 / 2 / 10 / 2049 / 100003) | 尾块路径的唯一覆盖 |
| `ceil` 的 ties 与跨符号 | `0.5 / -0.5 / 1.0 / -1.0 / 2.5 / -2.5 / ±0.0` | ties 是取整语义的核心 |
| **NaN 传播** | `NaN` | §0.1 声明必须保持;**且本设计的 C2 完整版专为它加了 `m_ord` 一路** —— 不测等于该路无覆盖 |
| **±inf 传播** | `+inf / -inf` | 同上:`±inf` 在 C1 单跑下会变成 `±2.147e9`(实测),C2 修好;**必须验证** |
| **超 int32 范围的有限值** | `\|x\| \in [2^{31}, 2^{32}]` | 值域分歧的原始触发点;C2 的 `m_ge` 一路专为它而设 |
| $2^{23}$ 附近 | `\|x\|` 略大于 / 略小于 $2^{23}$ | 该处 fp32 由"有小数部分"转为"已是整数",是旁路阈值的语义分界参考点 |

> ⚠️ **边界语义既是"必须保持",就应落在 L1(阻塞层)**,不宜放到 L2/Boundary(告警层)—— 告警层失败不阻断,无法保证语义保全。

### §9.3 与既有实现的关系(需后续 Stage 裁决)

> 本算子已有手工 NPU 实现且 8 passed(`ceil_kernel/ceil.py`)。本设计与其**算法同源**(int32 round-trip、`alloc_ub`、尾块切片、两条 dtype 分支),但在**边界语义**上有**明确超出**:
>
> | 项 | 既有实现 | 本设计(C2 完整版) |
> |---|---|---|
> | $\lvert x\rvert \ge 2^{31}$ | ❌ 变 `±2.147e9`(实测) | ✅ 旁路取 $x$ |
> | `±inf` | ❌ 变 `±2.147e9`(实测) | ✅ 旁路取 $x$ |
> | **`NaN`** | ❌ **变 `0`(实测)** | ✅ `m_ord` 路取 $x$ |
>
> ⚠️ **既有实现的三项边界缺陷是已交付代码里的真实缺陷**(其测试 `tests/ops/test_ceil.py` 完全不含 NaN/±inf/超范围用例,故未暴露)。**本设计不修改该文件**;该项已同步登记为 §10 R10,须在后续 Stage 或独立修复中处理,**不得只在设计里补而放任集成侧继续带缺陷**。
>
> 是否以本设计为准由 Stage 2 裁决。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| **R1** | **值域分歧 —— 已由 C2 消除**。初版采纳的 C1(int32 round-trip)在 $\lvert x\rvert \ge 2^{31}$ 处未定义,而源算子在该区间有确定值($\lceil x\rceil = x$);**路径 A 修订采纳 C2 后,该分歧不再存在**,本实现在整个 fp32 值域上与源算子等价。 | ~~高~~ → **已闭环** | ① §0.6 重设计项 3 记录了触发、实测与采纳过程;② 实测依据:`history_version/design_probe_int32_overflow.py`(`3.0e9 → 2147483648.0`,溢出不 fault);③ **残留条件**:若后续工具链变更导致溢出 lane 转为 fault,须改用 C1 + 契约层的域限制裁决(可用同一探针复测)。 |
| R2 | **`round_mode` 只在真发生 dtype 变化的 `vcast` 上生效** | 高 | §3.2 已注明;fp32 必须直接 `vcast` 到 int32,不得先经冗余 fp32→fp32 `vcast`(会静默退化为截断,`0.5→0`)。Stage 3 实现时须复核。 |
| R3 | **C2 依赖「UB 内 int32 溢出只回绕、不触发异常」这一前提** | **已实测消解** | 该前提**已实测成立**(`history_version/design_probe_int32_overflow.py`:`x=3.0e9` 经 C1 链路输出 `2147483648.0`,无异常)。工具链变更后可用同一探针复测。 |
| **R9** | **C2 的缓冲复用别名风险**:`abs_ub` 先承 `\|x\|`、后承 `vselect` 结果;`t_ub` 单写单读 | 中 | §3.3 已给出「活跃区间不重叠」的论证;**Stage 3 须复核**;若冲突,改为独立缓冲(UB 预算 68 KB → 76 KB,仍 ≤ 113 KB,见 §4.3) |
| **R10** | **既有已交付实现存在三项边界缺陷(实测)**:`\|x\| \ge 2^{31}` → `±2.147e9`、`±inf` → `±2.147e9`、**`NaN` → `0`**;其测试无 NaN/±inf/超范围用例故未暴露。**这是集成侧代码里的真实缺陷,不是设计问题。** | **高** | ① 本设计(C2 完整版)已覆盖这三项(§0.6、§9.3);② **但本设计不修改既有文件**,故集成侧缺陷**仍然存在**,须由后续 Stage 3 按本设计重写 kernel、或独立修复项处理;③ 已在 §9.3 与本节交叉引用,**不得只在设计里补而放任集成侧带缺陷** |
| **R11** | **有序数判据依赖本硬件的非 IEEE 行为**:`vcmp(x,x,"eq")` 对 NaN 返回 **false**、对有序数返回 true,而 IEEE 下 `eq` 也是"除 NaN 外为真"—— 表面上一致,但本硬件**同时**让 `ne`/`c_nan` 比较**恒 false**(IEEE 应为 NaN→true),故**不能靠"取反"构造 NaN 判据**,只能靠**反转选择子操作数**(V2)。该行为是**实测结论**,非标准保证 | 中 | 判据有实测依据(`history_version/design_probe_final_sequence.py` 的 V1/V2 对照);**Stage 3 须把 NaN 用例放进 L1(阻塞层)**(§9.2),使该行为变更能被测试捕获;kernel 内保留注释指向探针 |
| R4 | `T.vcast` 的参数顺序(源在前) | 高 | §3.2 已注明依据;写反不报错、只出错值。 |
| R5 | 尾块非整除路径的实测覆盖 | 中 | manifest 的 4 个 workload 的 N 均被 2048 整除 → **非整除路径未被 manifest 覆盖**;Stage 3 的 L1 套件必须补 `N % 2048 != 0`(既有实现已含 N=100/2049/3000)。 |
| R6 | 整型 dtype 未覆盖 | 中 | manifest 声明 int8/int16/int32/int64/uint8,但源侧在 **op 层短路**(`_IntIdentityUnaryOp`);NPU 侧为 float-only,与源 kernel 家族 `FloatUnaryKernel` 的 float-only 性质一致。**列为后续**。 |
| R7 | `f_ub` 在非 fp32 分支被复用(先承接入参、后承接 int32 回写) | 中 | 既有实现的既有形态;Stage 3 须复核复用不产生 aliasing 冲突(既有实现已 8 passed,风险较低)。 |
| R8 | 技术约束检测结论 | — | 三维 Kernel ❌ 未使用;GPU 专用 API ❌ 未使用;GEMM 非整除 ❌ 不适用;L0C 溢出 ❌ 不适用(无 Cube) → **无技术约束冲突**。 |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/ceil/ceil_fwd_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/ceil/ceil_fwd_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**(§1.6.1 公式层无采纳优化项;C2 是 §0.6 的**重设计层 API 替代**,其等价性由**实测**(溢出行为探针)与全值域论证承载,不属公式等价性验证) |
| 3b | **设计探针(C2 裁决前置)** | `examples/ceil/ceil_fwd_kernel/history_version/design_probe_int32_overflow.py` | ✅ 已产出并**实跑**:`3.0e9 → 2147483648.0`,溢出不 fault |
| 3c | **设计探针(NaN 极性取证,round 2 前置)** | `examples/ceil/ceil_fwd_kernel/history_version/design_probe_nan_polarity.py` | ✅ 已产出并**实跑**:NaN 在 C1/ge/lt 三变体下均输出 `0`;据此**推翻**"翻转极性可修复 NaN"的推断 |
| 3d | **设计探针(NaN 判据选取)** | `examples/ceil/ceil_fwd_kernel/history_version/design_probe_nan_detect.py` | ✅ 已产出并**实跑**:判据 B `vcmp(x,x,"eq")` 对 **NaN 为 false**、对有序数为 true(判据 A/D 恒 false,C 亦可) |
| 3e | **设计探针(端到端序列定案)** | `examples/ceil/ceil_fwd_kernel/history_version/design_probe_final_sequence.py` | ✅ 已产出并**实跑**:V1(选择子写反)在 `0.5` 上得 `0.5`、NaN 得 `0.0`;**V2 在全部 7 例(含 NaN/±inf/3e9/0.5/-0.5/1.0)通过** —— 本设计据此定为 V2 |
| 3f | **repro(ED-C 转正)** | `examples/ceil/ceil_fwd_kernel/repro/TRAP-vcmp-nan-non-ieee.py` | ✅ 已产出并**实跑**:A/D 恒 false、B 对 NaN 为 false —— **正是该 repro 的断言抓出了设计中"读错探针打印"的错误** |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/…` | ✅ Stage 0 已产出 |
| 5 | npuir kernel + golden + 分层测试 | `examples/ceil/ceil_fwd_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |

