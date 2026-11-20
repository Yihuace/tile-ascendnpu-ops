# DESIGN.md — `AdaLayerNormKernel`(TileLang-NPUIR)· 覆盖 `AdaLayerNormFwdOp` 与 `AdaLayerNormZeroFwdOp`

- **project** = `ada_layer_norm`,**kernel** = `AdaLayerNormKernel`,**family** = **norm**
- **覆盖的 manifest Op**:**两个** —— `AdaLayerNormFwdOp`(非门控)与 `AdaLayerNormZeroFwdOp`(**门控 / AdaLN-Zero**);两者 **Phase 0 已核实映射到同一 `kernel_map`**
- **迁移类型**:migration(harness 子模式)· **编程模式**:Developer(**以环境变量形式生效**)
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/norm/ada_layer_norm.py`(`AdaLayerNormKernel` @305)
- **`source_output_shape`**:`y: (M, N)`(**单输出**)
- **本算子的四处特殊性**(本批至此最复杂的一个):
  1. **一个 kernel 覆盖两个语义变体**,由 Python 层 `has_gate` 切换;而**签名对两者相同**(`out_idx=[4]` 恒定)—— 非门控时第 4 槽填占位张量且内核从不读它
  2. **两层归一化**(LayerNorm:减均值后再除标准差),外加 `scale`/`shift` 仿射,门控变体再加一路 `gate` 逐元素乘
  3. **持久核(persistent core)+ 核内串行任务循环** —— 与同批其余算子的"一核一块"形态**不同**
  4. `vrsqrt` 精度不足(实测 ~2.7e-3),**越 fp32 `1e-5` 与 fp16 `1e-3` 双重门** ⇒ 必须 `vsqrt + vdiv`

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

**数学定义(逐字取自既有实现 docstring)**:

```
非门控(AdaLayerNorm):
    mean_i = mean_j(x_ij)
    var_i  = mean_j((x_ij - mean_i)^2)
    rstd_i = rsqrt(var_i + eps)
    xn_ij  = (x_ij - mean_i) * rstd_i
    y_ij   = scale_ij * xn_ij + shift_ij

门控(AdaLayerNormZero,has_gate=True):
    y_ij   = gate_ij * (scale_ij * xn_ij + shift_ij)      # 仅多一路逐元素乘
```

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | **两层归一化**(先减均值、再除标准差)后做仿射 `scale*xn + shift`;门控变体再乘 `gate` | 既有实现 docstring 原文给出 `rstd_i = rsqrt(sum_j(x_ij - mean_i)^2 / N + eps)` |
| I/O 契约 | 输入:`x (M,N)`、`scale (M,N)`、`shift (M,N)`、`gate (M,N)`(**非门控时为占位**);输出:`y (M,N)` | 既有实现签名 `main(x, scale, shift, gate, y)` |
| **`out_idx=[4]` 对两变体恒定** | `gate` 占据签名索引 3,**使输出始终落在索引 4** ⇒ 门控与非门控**共用同一 `out_idx`**(源契约) | 既有实现注释原文:"`out_idx=[4]` therefore stays identical for both variants (source contract)" |
| **归约语义** | 需**两次**行归约:先 $\text{mean}(x)$,再 $\text{mean}((x-\text{mean})^2)$ | §0.3 |
| 精度策略 | 半精度输入走 **fp32 transit**(`use_fp32_transit = dtype in ("float16","bfloat16")`) | 既有实现注释:"bf16 has no Vector arithmetic; fp16 transit matches the golden's fp32 opmath rounding path" |
| **`eps` 语义** | `rstd = rsqrt(var + eps)`,加在**方差之后、开方之前**;默认 `1e-5`(golden 亦为 `1e-5`) | 既有 goldder `def golden_ada_layer_norm(x, scale, shift, eps=1e-5)` |
| **`N_inv` 折叠** | `N_inv = 1.0/N` 为**编译期常量**,除以 N 改为乘 `fl(1/N)` | 既有实现注释:"O1 … compile-time reciprocal, used as the scalar src2 of `T.vmul`" |
| host 侧语义 | reshape 到 `(M, N)`;`M = product(x.shape[:-1])` | manifest `vars` |

**完成标准核对**:已可写出与源算子数值一致的 golden(§9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | `x`/`scale`/`shift`:必填,均 `(M, N)`;`gate`:`(M, N)`,**非门控时传占位** |
| 输出 | `y`:`(M, N)`,**单输出**,`out_idx=[4]` |
| 归约轴 | **最后一维**(两次:均值、方差) |
| 动态轴 | **无** |
| dtype | fp16 / bf16(**fp32 transit**);有独立 dtype 支撑集合 `_SUPPORTED_DTYPES` |

### §0.3 第二问:实现算法(How)

**入口签名**:`_ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False)`
- `use_cp_async` **保留但被忽略**(CUDA `cp.async` 在 NPU 上不存在)

**Kernel 结构:持久核 + 核内串行任务循环**(与同批其余算子不同):

```
num_logical    = ceildiv(M, block_m)                  # 逻辑任务数
num_kernels    = min(num_logical, vector_cores)       # 物理核数(实查 ×2)
num_local_tasks= ceildiv(num_logical, num_kernels)    # 每核任务数(核内串行)
vector_cores   = NPUUtils.get().get_aicore_num() * 2  # 24 × 2 = 48,每次工厂调用重查
```

**host 侧 UB 容量护栏**(`ValueError` 而非裸 `assert`,以便 `python -O` 下仍生效):

```
if block_m * N > _UB_MAX_ELEMS: raise ValueError(...)
```

**数据流(每任务)**:

```
读 x / scale / shift(+ gate,门控时)
  → fp32 transit(半精度)
  → 行归约 mean
  → 减均值、平方、行归约 var
  → rstd = vsqrt + vdiv(不用 vrsqrt,见重设计项 3)
  → xn = (x - mean) * rstd
  → y = scale*xn + shift        (非门控)
    y = gate*(scale*xn + shift) (门控)
  → 写 y
```

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 依据 |
|---|---|---|---|
| **O1** | **编译期倒数折叠** `N_inv = 1/N` | 除 → 乘 | 既有实现注释 |
| **O2** | **`use_fp32_transit`** | 半精度统一升 fp32 算 | 既有实现注释 |
| **O3** | **持久核 + 核内串行任务** | 摊薄核启动/调度开销 | 既有实现 §5.5 |
| G1 | 算子融合(归一化 + 仿射 + 门控一次完成) | 省 GM 往返 | roofline 按 4~5 路计 |
| G2 | `block_m` 由 UB 预算反推 | 容量可行 | `_default_block_m(N)` + 主机护栏 |

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| 两层归一化 + 仿射 + 门控的数学语义、`out_idx=[4]` 双变体契约 | 语义 | **保留** | 语义层不可妥协 |
| **O1 倒数折叠** | 优化 | **等价替换** | 编译期常量乘,数学等价(见 §1.6.1) |
| **O2 fp32 transit** | 优化 | **等价替换** | → fp32 UB 缓冲 |
| **O3 持久核 + 核内串行** | 优化 | **重新设计** | 源的 block/thread 两级 → **持久核 + `T.serial` 核内任务循环**(本批首次采用该形态) |
| **`1/sqrt` 实现路径** | 优化 | **重新设计** | `vrsqrt` 精度不足 ⇒ `vsqrt + vdiv`(§0.6 重设计项 3) |
| 主机侧 UB 护栏 | — | **等价替换** | NPU 无 CUDA 侧预算概念,改为 host `ValueError` |
| autotune | 优化 | **舍弃** | K8:NPU 无 autotune |

### §0.6 NPU 算法重设计

#### 重设计项 1 —— 并行结构:**持久核 + 核内串行任务**(O3)

- **源方案**:CUDA grid × block × thread,每个 block 处理一个分块。
- **NPU 新算法**:**物理核数固定为 $48$**(实查 ×2),逻辑任务按 `num_local_tasks` **在核内串行**执行。
- **语义保持论证**:任务之间**完全独立**(行归约、无跨任务依赖),故"核内串行"与"核间并行"只改变**执行次序**,**不改变任何元素的取值路径**,也不改变任何一行内部的归约顺序 ⇒ 数学恒等。

#### 重设计项 2 —— 双变体共用同一 kernel(`has_gate`)

- **机制**:Python 层 `has_gate` 在 **trace 期**决定是否发射"乘 `gate`"那一步;签名恒定(`gate` 槽位非门控时填占位且**从不被读**)。
- **语义保持论证**:门控变体相对非门控**仅多一路逐元素乘**:$y = g\cdot(\text{scale}\cdot xn + \text{shift})$。共用主体**不引入任何近似**;两变体的差异是**精确的代数新增项**。
- **设计意义**:该结构使 **13 个算子里的第 10、11 项合并为一个迁移单元**(Phase 0 已核实同 `kernel_map`)。

#### 重设计项 3 —— **`1/sqrt` 实现路径:`vsqrt + vdiv`(与 `fused_add_rms_norm` 同源)**

- **NPU 约束(实测,既有实现 docstring 原文)**:`T.vrsqrt` 在本工具链上约 **2.7e-3 相对误差**;该误差传播到 $y$ 约为 $2.7\times10^{-3}\cdot|\text{d}\cdot\text{rstd}\cdot\text{scale}|$,**同时违反 fp32 `1e-5` 与 fp16 `1e-3` 两道门**(实测约 **87% / 6%** 的元素越界)。
- **采纳**:`rstd = 1/sqrt(v) == sqrt(v)/v` → `T.vsqrt` + `T.vdiv`(作用在 `(block_m, 1)` 缓冲上)。
- **等价性论证**:对 $v>0$ 恒等;$v = \text{var}+\epsilon > 0$ 由 `eps` 保证。
- **证据**:既有实现附 **`repro/TRAP-vrsqrt-plain-precision.py`**(自包含、可复跑)—— 本设计**沿用该 repro 作为依据**,不再另造。

#### 重设计项 4 —— 主机侧 UB 容量护栏

- **源方案**:无(由 CUDA 侧资源模型隐式约束)。
- **NPU 新算法**:host 侧用 `block_m * N > _UB_MAX_ELEMS` 判定并 `raise ValueError`(带完整的预算说明文本)。
- **为什么用 `ValueError` 而非 `assert`**:`assert` 在 `python -O` 下被剥离,护栏会失效。**该取舍正确,本设计沿用。**

### §0.7 迁移决策摘要

- **三问结论**:语义 = **两层归一化** + 仿射 (+ 门控)、**单输出 `out_idx=[4]` 且双变体恒定**、fp32 transit、`eps` 在方差后;算法 = **持久核 + 核内串行任务** + 主机容量护栏 + 五项优化手段。
- **耦合性判定统计**:**保留 1**、**等价替换 3**、**重新设计 3**(持久核 / `1/sqrt` 路径 / 主机护栏)、**舍弃 1**(autotune)。
- **重设计项清单**:① 持久核 + 核内串行;② 双变体共用(`has_gate`);③ **`1/sqrt` 改 `vsqrt+vdiv`**;④ 主机侧 UB 护栏。
- **源优化意图承接**:O1 → 编译期倒数折叠;O2 → fp32 transit;O3 → 持久核;G1 → 单 kernel 融合;G2 → `block_m` 由预算反推。**无静默丢弃。**

---

## §1 概述

### §1.1 算子概述

`AdaLayerNormKernel` —— 自适应 LayerNorm(归一化 + 仿射 `scale`/`shift`),并有**门控变体**(AdaLN-Zero,$y = g\cdot(\text{scale}\cdot xn+\text{shift})$)。用于 DiT/AdaLN 系扩散模型的条件注入。
**迁移后 NPU 侧算法**:**持久核 + 核内串行任务**;半精度走 fp32 transit;`1/sqrt` 用 `vsqrt+vdiv`。

### §1.2 计算语义

- 输入:$x,\text{scale},\text{shift},g \in \mathbb{R}^{M\times N}$,$\epsilon$;输出 $y\in\mathbb{R}^{M\times N}$
- 非门控:$y = \text{scale}\cdot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\text{shift}$;门控:上式再乘 $g$

### §1.3 算子特征分析

| 维度 | 判定 |
|---|---|
| 计算类型 | **纯 Vector + 两次行归约**(无 Cube) |
| 复杂度级别 | **单步融合**(读 3~4 路 → 两次归约 → 仿射 → 写 1 路) |
| 动态 shape | **无** |
| **输出数** | 1(但**两变体共用同一 `out_idx`**) |
| **调度形态** | **持久核 + 核内串行任务**(本批唯一) |

### §1.4 算法描述

沿 $M$ 生成逻辑任务;物理核固定 48;每核按 `num_local_tasks` **串行**处理任务;每任务在 fp32 域完成 §0.3 的序列,写回 `y`。归约维不跨核。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R —— 完整调研)

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **一处已采纳的等价改写(倒数折叠);一处必需的精度改写** | 候选:① **基线**;② **`N_inv=1/N` 编译期折叠(已采纳, O1)** —— 除以 $N$ 改为乘 $\text{fl}(1/N)$,与实数域等价、与源实现一致;③ **`1/sqrt(v) → sqrt(v)/v`(必需)** —— 因 `vrsqrt` 精度不足;④ 把 `scale`/`shift` 的仿射与归一化拆成两 kernel —— **反而不省**(多一路 GM 往返) |
| **R2 在线算法** | **有,且已在既有实现中:核内串行任务** | 结构依据:任务之间独立 ⇒ 允许"少量核 + 核内多次串行"的持久化形态。**与 `vector_norm` 的"分块累加"不同**:这里的串行是**任务级**的(每任务各自完整),**不改变**任何一行内部的归约顺序 |
| **R3 复杂度四口径** | **FLOPs** = $5MN$(非门控)/ $6MN$(门控);**访存** = $4MN\cdot\text{elem}$(非门控)/ $5MN\cdot\text{elem}$(门控);**扫描遍数 = 1**; 中间缓冲 **0**(UB 驻留) | 与 manifest roofline **逐字一致**(`5*M*N`/`4*M*N*elem_bytes` 与 `6*M*N`/`5*M*N*elem_bytes`) |
| **R4 硬件亲和** | 持久核与 48 物理核匹配;**半精度 bf16 无 Vector 算术** ⇒ 必须 fp32 transit;`vrsqrt` 精度不足须避开;`block_m` 受 UB 预算约束(有 host 护栏) | 逐候选评估;④ 因访存上升被拒 |

**调研结论**:选定 = **持久核 + 核内串行 + fp32 transit + `vsqrt/vdiv` + 倒数折叠**。

**设计期 roofline 估算下界(D-2)**

```
流量项 = 4~5·M·N·elem_bytes / HBM带宽
发射项 = 5~6·M·N / 向量吞吐
容量项 = block_m·N·(多缓冲 B/元素) / UB容量      ← 有 host 护栏
估算下界 = max(流量项, 发射项, 容量项)
```

#### §1.6.1 数学等价优化(公式级)

| 优化类别 | 结论 | 依据 |
|---|---|---|
| **恒等变形降代价** | **一条已采纳:`N_inv = 1/N` 编译期折叠**(O1);**一条必需:`1/sqrt(v) → sqrt(v)/v`** | 两者均数学等价(后者对 $v>0$,由 `eps` 保证) |
| 稳定化变形 | 不适用 | 无 `exp`;`eps` 保证方差非零 |
| 公共子表达式消除 | 已应用 | `N_inv` 复用;`x-mean` 结果复用于平方与 `xn` |
| 算子降代换 | 不适用 | 无更低代换空间 |
| 访存量削减 | **已最优(融合)** | 单 kernel 完成归一化+仿射(+门控);拆开会使访存从 4~5 路升至 7~8 路 |
| **归约结构优化** | 不适用 | 归约维整体驻留;**核内串行是任务级、不是归约级** ⇒ 不改变归约顺序 |

> 因**无性能类未决采纳项**(O1 已由源实现给出、`1/sqrt` 改写是精度驱动),`verify_equiv.py` **豁免**;既有实现已附 `repro/TRAP-vrsqrt-plain-precision.py` 作为精度依据。

#### §1.6.2 向量化替代分析

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | 归约 mean / var | 归约 | **已向量化替代** | `T.reduce(..., reduce_mode="sum")` |
| V2 | 减均值 / 平方 / 乘 `rstd` / 乘 `scale` / 加 `shift` / 乘 `gate` | 逐元素 | **已向量化替代** | `T.vsub`/`T.vmul`/`T.vadd` |
| V3 | `rstd` 的倒数(标量) | **每行标量** | **保留(不可替代)** | 理由类别:**每行 1 个标量的 `vsqrt`+`vdiv`**(唯一需解释的标量点;`1/sqrt` 的等价改写) |
| V4 | `N_inv`、`num_kernels`、`num_local_tasks` | **host/编译期常量** | **保留(不可替代)** | 理由类别:host 元数据 / 编译期折叠(PL-1.5) |
| V5 | 核内任务循环 | 任务级循环 | **保留(不可替代)** | 理由类别:**持久核的任务映射**,非逐元素;其存在是为摊薄核启动开销(§0.6 重设计项 1) |

**结论**:逐元素与归约热点(V1/V2)全部向量化;V3 为每行标量,V4 为编译期常量,V5 为任务级映射 —— **无逐元素标量残留**。

#### §1.6.3 向量化轴与数据布局决策

- **候选轴**:**2 个** —— ① 沿 $M$(未归约维)分任务;② 沿 $N$(归约维)切分再合并。
- **评分**:① **主选**(归约维在核内完整完成 ⇒ 不改结合顺序;任务级串行不影响行内语义);② 不采纳(会改变加法结合顺序,且容量无压力时不值得引入语义风险)。
- **`block_m` 的选择**:由 UB 预算反推($\text{block\_m}\cdot N \le \text{UB 元素上限}$),`_default_block_m(N)` 给默认值,host 侧另设护栏。
- **决策落地**:主选 = 沿 $M$ 分任务 + 核内串行 + 归约维驻留,是 §3.3 与 §5 的唯一依据。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无** |
| 是否需要手工同步 | **否** —— 任务之间独立,核内串行为线性序列 |
| 是否需要控制 L1/L0 层级 | **否** |
| 是否需要跨核通信 | **否**(归约维不跨核;任务级串行不引入跨核依赖) |
| 与既有实现一致性 | 本仓 `kernels/norm/ada_layer_norm` 为 Developer |

> ⚠️ **落地形式是硬约束**:`TILELANG_ASCEND_MODE` **必须以环境变量形式生效**。既有实现注释写的是 *"Developer mode: alloc_shared maps to UB, compiler auto-sync"*,但本批已在 `vector_norm` 上**实测**(**E1**):缺该环境变量时,含归约的核会被降到 Cube 路径并使 `alloc_shared` 落 `cbuf`(L1),`cbuf → fragment` copy 编不过。**交付件须显式 `os.environ.setdefault`。**

---

## §3 API 映射设计

### §3.1 公式拆解

| 步 | 计算 | 对应式 |
|---|---|---|
| 1 | 读 `x`/`scale`/`shift`(+`gate`) | 输入 |
| 2 | 升 fp32(半精度) | fp32 transit |
| 3 | $\text{mean} = \sum x \cdot N_{inv}$ | 第 1 次行归约 |
| 4 | $\text{var} = \sum (x-\text{mean})^2 \cdot N_{inv}$ | 第 2 次行归约 |
| 5 | $\text{rstd} = 1/\sqrt{\text{var}+\epsilon}$ | `vsqrt`+`vdiv` |
| 6 | $xn = (x-\text{mean})\cdot\text{rstd}$ | 逐元素 |
| 7 | $y = \text{scale}\cdot xn+\text{shift}$(+门控乘 `gate`) | 逐元素 |
| 8 | 降回 dtype、写 `y` | 输出 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| **持久核** | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`(`num_kernels = min(ceildiv(M,block_m), 48)`) | 既有实现 §5.5 |
| **核内串行任务循环** | **`T.serial(num_local_tasks)`** | 既有实现 §5.5;**本批首次使用该形态** |
| UB 缓冲 | `T.alloc_shared((block_m, N), dtype)` / fp32 版 | 既有实现;**须配 Developer 环境变量(E1)** |
| 升 fp32 | `T.vcast(..., round_mode="rint")`(仅半精度路径;fp32 用 `T.copy`) | 既有实现 O2 |
| **行归约(两次)** | `T.reduce(..., reduce_mode="sum")` | 既有实现 |
| **倒数折叠** | `T.vmul(a, N_inv, b)`(`N_inv` 为编译期常量) | 既有实现 O1 |
| **`1/sqrt` 组合** | **`T.vsqrt(rstd, tmp)` + `T.vdiv(tmp, rstd, rstd)`** | 既有实现;**依据 `repro/TRAP-vrsqrt-plain-precision.py`** |
| 逐元素变换 | `T.vsub` / `T.vmul` / `T.vadd` | 既有实现 |
| **门控(仅 `has_gate=True`)** | 额外一路 `T.vmul(y_partial, gate, y)` | 既有实现 |
| host 容量护栏 | `if block_m * N > _UB_MAX_ELEMS: raise ValueError(...)` | 既有实现;**`ValueError` 而非 `assert`(防 `-O` 剥离)** |
| **`out_idx`** | **`out_idx=[4]`(两变体恒定)** | 既有实现注释 |

### §3.3 伪代码

```python
def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False):
    N_inv = 1.0 / float(N)                                  # O1 编译期折叠
    use_fp32_transit = dtype in ("float16", "bfloat16")     # O2
    block_m_default = _default_block_m(N)
    vector_cores = NPUUtils.get().get_aicore_num() * 2      # 实查 24 × 2 = 48

    @tilelang.jit(out_idx=[4], target="npuir")              # ★ 两变体恒定
    def _func(block_m=block_m_default):
        if block_m < 1:
            raise ValueError(...)
        if block_m * N > _UB_MAX_ELEMS:                     # host 容量护栏
            raise ValueError(f"block_m={block_m} exceeds the UB budget for N={N} ...")

        num_logical     = (M + block_m - 1) // block_m
        num_kernels     = min(num_logical, vector_cores)    # 持久核
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype],       # 0
                 scale: T.Tensor[(M, N), dtype],   # 1
                 shift: T.Tensor[(M, N), dtype],   # 2
                 gate: T.Tensor[(M, N), dtype],    # 3  (非门控时传占位,从不读)
                 y: T.Tensor[(M, N), dtype]):      # 4  ★ out_idx=[4]
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # ... UB 缓冲(block_m, N)dtype / fp32
                for t in T.serial(num_local_tasks):         # ★ 核内串行任务
                    task = cid * num_local_tasks + t
                    if task < num_logical:
                        row0 = task * block_m
                        # 读 x/scale/shift(+gate) → fp32 transit
                        # mean = reduce_sum(x) * N_inv
                        # var  = reduce_sum((x-mean)^2) * N_inv
                        # rstd = vsqrt(var+eps); rstd = vdiv(sqrt, var+eps)
                        # xn   = (x - mean) * rstd
                        # y    = scale*xn + shift
                        # if has_gate: y = gate * y      ← Python 层 trace 期决定
                        # 写 y
        return main
    return _func
```

> `has_gate` 与 `use_fp32_transit` 均为 **Python 层 trace 期**的值 ⇒ **只发射对应分支**,无运行期分支开销。

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| **既有 ada_layer_norm NPU 实现(基线版)** | `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`(981 行) |
| **既有 `perf_opt` 变体** | 同目录 `perf_opt/_ada_layer_norm_kernel.py` |
| **`vrsqrt` 精度证据 repro** | 同目录 `repro/TRAP-vrsqrt-plain-precision.py` |
| 同族(同 `vsqrt+vdiv` 约定) | `examples/TileOPs/tileops/kernels/norm/fused_add_rms_norm/…` |
| 已走完整流程的同批算子 | `examples/{abs,ceil,floor,exp,cos,vector_norm,fused_add_rms_norm}/…` |

---

## §4 数据规格与内存规划

### §4.1 数据规格

| 张量 | shape | dtype | 所在 | 分配 API |
|---|---|---|---|---|
| `x` / `scale` / `shift` | `(M, N)` | fp16 / bf16 | GM | `T.Tensor` 入参(索引 0/1/2) |
| `gate` | `(M, N)` | 同上 | GM | 入参(索引 3;**非门控时为占位,内核从不读**) |
| `y` | `(M, N)` | 同上 | GM | 出参(`out_idx=[4]`) |
| `x_ub` / `scale_ub` / `shift_ub` / `y_ub` | `(block_m, N)` | 同输入 | UB | `T.alloc_shared`(**须配 E1 环境变量**) |
| `x_f32` / `sq_f32` / `mean_ub` / `rstd` 等 | `(block_m, N)` 或 `(block_m, 1)` | float32 | UB | fp32 transit / 归约结果 |
| `gate_ub`(仅门控) | `(block_m, N)` | 同输入 | UB | `T.alloc_shared` |

> **层级 ↔ API 对照**:GM = `T.Tensor`;UB = `T.alloc_ub` **或** `alloc_shared`+Developer 环境变量(既有实现用后者,见 E1);`T.reduce_*` 的操作数须为 fragment(本算子既有实现用 UB 形态的 `T.reduce`,以此为准并留待 Stage 3 实测)。

### §4.2 搬运路径

```
GM(x, scale, shift[, gate]) --T.copy--> UB --vcast--> UB(fp32)
   --reduce(mean)--> --reduce(var)--> --vsqrt+vdiv--> rstd
   --vsub/vmul/vadd--> UB(y_fp32) --vcast--> UB(y_dtype) --T.copy--> GM(y)
```

### §4.3 UB 容量预算(有主机护栏)

**预算模型(既有实现的常量口径)**:多缓冲下按 **20 B/元素**计,UB 容量 `_UB_CAPACITY_BYTES = 196 608`(192 KB),故**驻留元素上限**

```
_UB_MAX_ELEMS = 9 216        # 既有实现直接给出该常量
_default_block_m(N) = min(_BLOCK_M_CAP, max(1, _UB_MAX_ELEMS // N))
host 护栏:  if block_m * N > _UB_MAX_ELEMS: raise ValueError(...)
```

**逐 buffer 核算**:**最坏活跃 workload** = `llama-3.1-8b-prefill` = `(2048, 4096)`,`N = 4096` ⇒ `block_m = min(cap, 9216 // 4096) = 2`;dtype = fp16(elem_bytes = 2)。

| buffer | 尺寸表达式 | fp16 峰值(block_m=2, N=4096) |
|---|---|---|
| x_ub | block_m × N × elem_bytes | 16 KB |
| scale_ub | block_m × N × elem_bytes | 16 KB |
| shift_ub | block_m × N × elem_bytes | 16 KB |
| y_ub | block_m × N × elem_bytes | 16 KB |
| gate_ub | block_m × N × elem_bytes | 16 KB |
| x_f32 | block_m × N × 4 | 32 KB |
| sq_f32 | block_m × N × 4 | 32 KB |
| tmp_f32 | block_m × N × 4 | 32 KB |

**合计**:176 KB。

> `rstd` 等行标量为 `block_m × 4` = 8 B,不进上表。
> **交叉校验**:`block_m × N × 20 B = 2 × 4096 × 20 = 163 840 B = 160 KB`,与上表 176 KB 同量级(差值来自 fp32 中间量的具体条数);两者**均在 192 KB 之下**。

UB 容量上限 192 KB,自动多缓冲膨胀系数 1.7 → 有效预算约 113 KB(来源:`pattern-library/constants.md` 的 `CONST-capacity-910B2C`)。

**结论与护栏**:
- **`block_m` 由容量反推**(`_default_block_m(N)`),**160~176 KB 未超 192 KB**;
- **本算子是本批唯一把容量约束写成"主机侧显式护栏(`ValueError`)"的算子** —— 比"设计时核算 + 靠 Stage 3 实测"更硬:护栏在**运行期**按 `block_m * N > _UB_MAX_ELEMS` 拦截,且用 `ValueError` 而非 `assert`(防 `python -O` 剥离);
- §9.2 因此把**护栏行为本身**列为阻塞用例(须断言它真的抛 `ValueError`)。

### §4.4 中间缓冲峰值

**0** —— 无跨核部分和、无 workspace。

---

## §5 Tiling 策略

### §5.1 Block 划分与 Tile Shape

| 项 | 值 |
|---|---|
| 划分维度 | **未归约维 $M$**(划分为逻辑任务) |
| Tile Shape | `(block_m, N)` |
| 物理核数 | **48**(实查 ×2,**持久核**) |
| 核内任务数 | `num_local_tasks = ceildiv(num_logical, num_kernels)`(**串行**) |

### §5.2 非整除处理策略

1. **$M$ 不整除 `block_m`**:任务数取上取整,尾任务行数不足 ⇒ 内核按实际行数切分(须在 Stage 3 明确边界处理)。
2. **逻辑任务数 $>$ 物理核数**:核内串行(§0.6 重设计项 1)。
3. **`task < num_logical` 守卫**:防止核内最后一批空任务越界。

### §5.3 分核策略(三要素)

**① 逻辑核数计算**

```
num_logical      = ceildiv(M, block_m)
num_kernels      = min(num_logical, vector_cores)      # ← 物理核封顶
num_local_tasks  = ceildiv(num_logical, num_kernels)
```

**② 物理核数依据(实查,非假设)**

```
torch.npu.get_device_properties(0) → device: Ascend910B2C
    cube_core_num   = 24
    vector_core_num = 48
```

> **既有实现的口径**:`vector_cores = NPUUtils.get().get_aicore_num() * 2` —— 即**按 AI core 数 ×2 推算纯 Vector 的物理核数**(实测 24 × 2 = 48),且**每次工厂调用重查**以便在其他设备上自适应。该口径与 `torch.npu.get_device_properties` 的 `vector_core_num = 48` **一致**。

**③ 规模判定与分核方案**

| 情形 | 判定 | 方案 |
|---|---|---|
| 逻辑任务 ≤ 48 | 核数足够 | 一核一任务,`num_local_tasks = 1` |
| 逻辑任务 > 48 | 核数不足 | **持久核 + 核内串行**(`num_local_tasks > 1`)—— 选中「核内串行」分支 |

**结论:命中「持久核 + 核内串行」分支**(本批唯一)。依据:任务之间独立,串行只改执行次序、不改任何行内的归约顺序 ⇒ 语义安全;且摊薄核启动/调度开销(优化目标 O3)。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| **`T.serial`** | **使用 × 1** | **核内任务串行循环**(`for t in T.serial(num_local_tasks)`)—— 本批首次 |
| `T.Parallel` | 可能用于逐元素 | 逐元素亦可能全由 v-ops 覆盖 |
| `T.reduce` | **使用 × 2** | 均值 + 方差两次行归约 |
| `T.Pipelined` | **不使用** | 无多阶段可重叠 |
| `T.Persistent` | **不使用**(但形态上等价于持久核) | 持久化由"核数封顶 + 核内串行"实现,未用 `T.Persistent` 原语 |

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入 |
| 手工同步点 | **无** |
| 依赖性质 | 每任务内部线性;**任务之间独立** |
| 跨核同步 | **不需要** |

> ⚠️ 本算子**含核内串行循环**,依赖链比同批其他算子更长;**Stage 3 须确认自动同步在 `T.serial` 体内仍正确**(既有实现已通过测试,以此为准)。

---

## §8 CV 融合设计

**n/a** —— 纯 Vector + 行归约。

---

## §9 验证方案

### §9.1 Golden 函数草案(独立于 NPU 算法)

```python
def golden_ada_layer_norm(x, scale, shift, gate=None, eps=1e-5):
    """按 §0.1 语义;门控时返回 gate * (scale*Norm(x) + shift)。"""
    xn = F.layer_norm(x.float(), (N,), weight=None, bias=None, eps=eps)
    y = scale.float() * xn + shift.float()
    if gate is not None:
        y = gate.float() * y
    return y.to(x.dtype)
```

> 既有实现即用 `F.layer_norm`(见其 `golden_ada_layer_norm`),**与 §0.1 语义一致**;`eps` 默认 `1e-5`。

### §9.2 L0 门槛测试计划

| 用例 | 说明 |
|---|---|
| `(M, N)` 小规模 × {fp16, bf16} | smoke;两变体各跑 |
| 主 workload × {fp16, bf16} | manifest 的 3 条各自 |

**L1(阻塞层)必须包含的用例**:

| 类别 | 用例 | 为什么放在**阻塞层** |
|---|---|---|
| **两个变体各自覆盖** | `has_gate=False`(`AdaLayerNormFwdOp`)与 `has_gate=True`(`AdaLayerNormZeroFwdOp`)**分别**跑 | 两 Op 共用一个 kernel;**只测一个会漏掉另一条 trace 分支** |
| **门控语义** | 独立断言 $y = g\cdot(\text{scale}\cdot xn+\text{shift})$(不依赖非门控结果) | 门控是**精确的代数新增项**,须独立验证 |
| **`rstd` 精度** | 与 fp32 golden 比对,容差按 dtype | **`vrsqrt` 被拒的直接理由**(实测 87%/6% 元素越界);须验证 `vsqrt+vdiv` 满足 fp32 `1e-5` 与 fp16 `1e-3` 双门 |
| **`eps` 生效** | 常数列(方差为 0)⇒ `rstd = 1/sqrt(eps)`,断言输出有限 | `eps` 的唯一必要场景(否则除零) |
| **`N_inv` 折叠等价** | 与"归约后除以 N"对比(数值上应在容差内一致) | O1 是等价改写,须有覆盖 |
| **非整除 $M$** | `M % block_m != 0` | 尾任务路径 |
| **逻辑任务 > 物理核**(触发核内串行) | 大 $M$ 使 `num_local_tasks > 1` | **持久核分支的唯一覆盖** |
| 极小 $M$ | `M = 1`(单行 ⇒ `num_kernels = 1`) | 单核路径 |
| **容量护栏** | 构造 `block_m * N > _UB_MAX_ELEMS` 的调用,**断言抛 `ValueError`** | 护栏是**行为契约**(须验证它真的拦得住) |
| `scale`/`shift` 边界 | `scale ≡ 0`(输出只剩 `shift`)、`shift ≡ 0`(输出只剩 `scale*xn`) | 仿射路径的边界 |
| `NaN` 传播 | 含 NaN 的行 | 经 `vsub`/`vmul`/`reduce` 应传播 |

### §9.3 与既有实现的关系

> 本算子已有手工 NPU 实现(**基线版 + `perf_opt` 变体**),其 docstring 已记录 `vrsqrt` 精度不足(附 `repro/TRAP-vrsqrt-plain-precision.py`)、`N_inv` 折叠(O1)、fp32 transit(O2)、持久核(O3)与 host 护栏。
>
> 差异仅在:本设计把 §9.2 的**两变体分别覆盖**、**门控独立断言**、**容量护栏行为**、**核内串行分支**明确列为 L1 阻塞用例。
>
> ⚠️ **本算子集成侧已存在 `perf_opt/` 变体** ⇒ wrapper 的双 import 切换块**需保留两条路径**(与 `abs` 的做法一致:baseline 激活 + perf_opt 注释占位)。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| **R1** | **`T.vrsqrt` 精度不足**(~2.7e-3,越 fp32 `1e-5` 与 fp16 `1e-3` 双门;实测 87%/6% 元素越界) | **高** | §0.6 重设计项 3;依据既有 `repro/TRAP-vrsqrt-plain-precision.py`;§9.2 把 `rstd` 精度列 L1 |
| **R2** | **两变体共用一个 kernel,只测一个会漏另一条 trace 分支** | **高** | §9.2 要求两变体**分别**跑阻塞用例;门控另做独立断言 |
| **R3** | **核内串行循环(`T.serial`)的自动同步正确性** | **高** | §7 已标注;既有实现已通过测试,Stage 3 须在**触发核内串行**的规模上实测(§9.2 专门列该用例) |
| **R4** | **`alloc_shared` 需要 Developer 环境变量生效**(**E1**) | **高** | §2 硬约束;交付件须 `os.environ.setdefault` |
| **R5** | **容量护栏的"行为契约"** | 中 | §9.2 要求**断言抛 `ValueError`**(而非只核实现有);用 `ValueError` 而非 `assert` 以防 `-O` 剥离 —— 该取舍已由既有实现给出 |
| **R6** | **非整除 $M$ 的尾任务路径** | 中 | §5.2 已列;§9.2 覆盖 |
| **R7** | **`perf_opt` 变体的存在** | 中 | 集成时 wrapper 的双 import 块须保留两条路径(baseline 激活 + perf_opt 注释占位) |
| **R8** | 技术约束检测结论 | — | 三维 Kernel ❌;GPU 专用 API ❌(`use_cp_async` 保留但忽略);GEMM 非整除 ❌;L0C 溢出 ❌ → **无技术约束冲突** |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/ada_layer_norm/ada_layer_norm_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/ada_layer_norm/ada_layer_norm_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**:O1(倒数折叠)由源实现给出;`1/sqrt` 改写是精度驱动,其依据是既有 `repro/TRAP-vrsqrt-plain-precision.py` |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/norm/ada_layer_norm/…` | ✅ Stage 0 已产出 |
| 5 | npuir kernel(两变体)+ golden + 分层测试 | `examples/ada_layer_norm/ada_layer_norm_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |
