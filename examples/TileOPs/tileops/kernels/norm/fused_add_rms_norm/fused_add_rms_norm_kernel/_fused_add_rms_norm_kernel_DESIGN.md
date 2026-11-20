# DESIGN.md — `FusedAddRMSNormFwdOp`(TileLang-NPUIR)

- **project** = `fused_add_rms_norm`,**op_name** = `FusedAddRMSNormFwdOp`,**family** = **norm**
- **迁移类型**:migration(harness 子模式)· **编程模式**:Developer(**必须以环境变量形式生效**)
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/norm/fused_add_norm.py`(`FusedAddRMSNormKernel` @332)
- **`source_output_shape`**:**双输出** —— `y: (M, N)` 与 `residual_out: (M, N)`(均同 `x` 形状)
- **本算子的三处特殊性**(与已完成的 7 个 Op 相比):
  1. **双输出**(`out_idx=[3, 4]`),而此前全部为单输出
  2. **归约 + 广播写回**的 norm 结构(`rstd` 是**逐行标量**,需广播乘回整行)
  3. **内建数学上的数值陷阱**:`1/sqrt(v)` 若直接用 `T.vrsqrt` 实测误差 **~2.7e-3**,**超过 fp16 容差 1e-3** ⇒ 必须用 `vsqrt + vdiv` 组合

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

**数学定义(逐字取自源码 docstring 与既有实现)**:

```
s            = x + residual                      # 融合的 add
rstd         = 1 / sqrt(mean(s^2) + eps)         # 逐行标量
y            = s * rstd * weight                 # 逐元素
residual_out = s                                 # 第二输出
```

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | 融合 `x+residual` 的 RMSNorm;**两个输出**(归一化结果 `y` 与融合后的残差 `residual_out`) | `_fused_add_rms_norm_kernel.py` 的 docstring(原文给出上面四式) |
| I/O 契约 | 输入三路:`x (M,N)`、`residual (M,N)`、`weight (N,)`;输出两路:`y (M,N)`、`residual_out (M,N)`,均同 `x` 形状 | `out_idx=[3, 4]`;源实现注释"the same two-output contract as the GPU source" |
| **归约语义** | 沿最后一维求 $\text{mean}(s^2)$(**行归约**),得逐行标量 `rstd`;归约顺序影响浮点结果 | `T.reduce(..., reduce_mode="sum", size=[1, n_align])`;§0.3 |
| **广播语义** | `rstd` 为 `(1,1)` 标量,需**广播**乘到整行;`weight` 为 `(N,)`,需广播到每行 | §0.3 的数据流 |
| 精度策略 | **全程 fp32 域**计算(`s_f32`/`sq_f32`/`w_f32`/`rstd` 均为 fp32),仅在读入与写出时做 dtype 转换 | §0.3 |
| **`eps` 语义** | `rstd = 1/sqrt(mean(s²) + eps)`,`eps` 在**求和之后、开方之前**加入;默认 `1e-6` | §0.3;golden 同式 |
| **`residual_out` 语义** | 直接等于 `s = x + residual`(**未经归一化**),是"残差直通"输出 | docstring 原文 |
| 边界语义 | `NaN` 经 `x+residual` 与求和自然传播;`rstd` 对零行(全零 `s`)不产生除零(有 `eps`) | §9.2 的 L1 用例 |
| host 侧语义 | 仅 reshape 到 `(M, N)`;`M = product(x.shape[:-1])`,`N = x.shape[-1]` | manifest `vars` |

**完成标准核对**:已可写出与源算子数值一致的 golden(§9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | `x`、`residual`:`(M, N)`,`dtype ∈ {float16, bfloat16}`;`weight`:`(N,)`,同 dtype;`eps`:`float` 标量 |
| 输出 | `y`:`(M, N)`;`residual_out`:`(M, N)`;均同 `x` 的 shape/dtype |
| 归约轴 | **最后一维** |
| 动态轴 | **无**;`M`/`N` 构造期确定 |
| 输出 shape 确认 | **双输出均与 `x` 同形**(已由源码与测试文件头确认) |

### §0.3 第二问:实现算法(How)

**入口签名**:`_fused_add_rms_norm_kernel(M, N, eps, dtype)`

**Kernel 结构与数据流(逐行一核)**:

```
T.Kernel(M, is_npu=True) as (pid_m, _)          # 一核一行
    UB 缓冲(全为 (1, n_align) 或标量):
      x_tile, r_tile, w_tile, s_tile, out_tile   (dtype)
      s_f32, sq_f32, w_f32                        (fp32)
      rstd, rsqrt_tmp                             ((1,1) fp32)
    1. T.copy 读入 x / residual / weight
    2. T.clear 对齐尾部(n_align ≥ N 的填充区)
    3. s = x + residual        (fp32 域)
    4. sq = s * s
    5. T.reduce(sq, reduce_mode="sum", size=[1, n_align]) → 行和
    6. mean = sum / N ; rstd = 1/sqrt(mean + eps)   ← 见 §0.6 重设计项 3
    7. y        = s * rstd * weight
    8. residual_out = s
    9. T.copy 写出 y / residual_out
```

**关键结构决策(布局对齐)**:`n_align = _aligned_cols(N, dtype)`,`(1, n_align)` 的 tile **在行内对齐**,并对对齐尾部显式 `T.clear`,使 `T.reduce` 的 `size=[1, n_align]` 覆盖整数对齐长度 —— 这是本算子能进入向量归约路径的前提。

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 依据 |
|---|---|---|---|
| G1 | **算子融合**(add + RMSNorm 一次完成) | 省一次 GM 往返 | roofline `bytes = (4*M*N + N)*elem_bytes` 只按 4 路读/写计,若拆成两算子需 6 路 |
| G2 | **`residual_out` 直通复用** | 第二输出不需重算 | `residual_out = s`,而 `s` 已在 UB 中 |
| G3 | **逐行分核 + 行内驻留** | 归约维不跨核 | §0.6 重设计项 1(语义动机) |
| G4 | **fp32 累加域** | 精度 | 与源一致 |
| G5 | **行内对齐(`n_align`)** | 使向量归约可覆盖整行 | §0.6 重设计项 2 |

> 未使用:SMEM tiling、swizzle、warp shuffle、`T.gemm`、异步流水、autotune(见 §0.5 舍弃项)。

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| 四式数学语义、**双输出契约**、`eps` 位置、`residual_out = s` | 语义 | **保留** | 语义层不可妥协 |
| G1 算子融合 | 优化 | **等价替换** | 单 kernel 内完成,与源同形 |
| G2 残差直通 | 优化 | **等价替换** | UB 内复用 `s` |
| G3 逐行分核 | 优化 | **重新设计** | 源的 block/thread 两级 → 一维 `T.Kernel(M)`;**归约维不跨核**是语义要求 |
| G4 fp32 累加域 | 优化 | **等价替换** | → fp32 UB 缓冲 |
| **G5 行内对齐 `n_align`** | 优化 | **重新设计** | `T.reduce` 的 `size` 需整数对齐长度;故 TB tile 补对齐 + 显式清零尾部 |
| autotune | 优化 | **舍弃** | K8:NPU 无 autotune |

### §0.6 NPU 算法重设计

#### 重设计项 1 —— 并行结构:**归约维不跨核**(语义决策)

- **源方案**:CUDA block 内多线程协同完成行归约。
- **NPU 新算法**:`T.Kernel(M, is_npu=True)`,**一核一行,整行归约在该核内完成**。
- **语义保持论证**:归约仅在**单核内**完成 ⇒ 不引入跨核部分和与二次归约 ⇒ **加法结合顺序与源侧的"一行内累加"一致**(两侧同为 fp32 累加)。**这是语义动机,不只是性能取舍**(与 `vector_norm` 同一条理由)。

#### 重设计项 2 —— 行内对齐与尾部清零(G5)

- **为什么需要**:`T.reduce(..., size=[1, n_align])` 要求**整数对齐长度**;而 `N` 未必对齐(如 `N=4096` 对 fp16 需按向量宽度对齐)。
- **做法**:`n_align = _aligned_cols(N, dtype)`;tile 形状取 `(1, n_align)`;**对 `[N, n_align)` 的对齐尾部显式 `T.clear`**,使尾部以 **0** 参与求和 —— 对 `sum` 而言 0 是单位元,**不改变归约结果**。
- **语义保持论证**:清零的尾部贡献 `0` 到 `sum`,数学上恒等;而 `y`/`residual_out` 的写出只覆盖 `[0, N)`,尾部不写出。⇒ 对齐纯属实现细节,**不影响任何输出元素**。

#### 重设计项 3 —— **`1/sqrt(v)` 的实现路径:必须用 `vsqrt + vdiv` ⭐**

- **源方案**:数学上 `rstd = 1/sqrt(mean(s²)+eps)`;GPU 侧可直接用低误差的 rsqrt。
- **NPU 约束(实测)**:`T.vrsqrt` 在本工具链上实测 **~2.7e-3 相对误差**。
- **可行性判定**:

| 候选 | 形式 | 误差 | 判定 |
|---|---|---|---|
| C1 `T.vrsqrt(v)` | 单指令 | **~2.7e-3(实测)** | **不可用** —— **超过 fp16 容差 `1e-3`** |
| **C2 `vsqrt` + `vdiv`(采纳)** | `1/sqrt(v) == sqrt(v)/v` → `t = vsqrt(v)`;`rstd = vdiv(t, v)` | 受两条指令各自的精度限制(实测组内其他算子 `vsqrt` 精度充足) | **采纳** |
| C3 多项式自实现 | 自写逼近 | 需自行标定 | **舍弃**(C2 已足) |

- **语义保持论证**:`1/sqrt(v)` 与 `sqrt(v)/v` 在实数域**恒等**(对 $v>0$);`v = mean(s²)+eps > 0` 由 `eps` 保证(即使 `s ≡ 0`,`v = eps > 0`)。⇒ 数学等价,差异仅在浮点舍入,且实测精度满足容差。
- **与源的一致性**:源仓的 `ada_layer_norm` kernel **出于同一原因采用了同一组合**(见既有实现 docstring 原文:"the same composition the in-repo `ada_layer_norm` kernel adopted for the same reason")⇒ 本设计**与仓内既有约定一致**。

#### 重设计项 4 —— 双输出路径(`y` 与 `residual_out`)

- **两个输出均同形 `(M, N)`**,故 kernel 需**两次写出**(`out_idx=[3, 4]`,与源一致)。
- **语义保持论证**:`y = s*rstd*w` 与 `residual_out = s` **共用同一个 UB 里的 `s`**,不重算、不重读 ⇒ 两路输出的数值都直接来自同一次读取,**天然一致**。

### §0.7 迁移决策摘要

- **三问结论**:语义 = 融合 add+RMSNorm、**双输出**、行归约 + 广播、fp32 域、`eps` 加在开方前、`residual_out = s`;算法 = 逐行一核 + 行内对齐 + 五项优化手段。
- **耦合性判定统计**:**保留 1**、**等价替换 3**、**重新设计 2**(逐行分核 / 行内对齐)、**舍弃 2**(autotune / C3)。
- **重设计项清单**:① 并行结构(归约维不跨核);② 行内对齐与尾部清零;③ **`1/sqrt` 改 `vsqrt+vdiv`**;④ 双输出路径。
- **源优化意图承接**:G1 融合 → 单 kernel;G2 残差直通 → UB 复用;G3 分核 → 逐行一核;G4 精度 → fp32 域;G5 对齐 → `n_align` + `T.clear`。**无静默丢弃。**

---

## §1 概述

### §1.1 算子概述

`FusedAddRMSNormFwdOp` —— 融合残差相加的 RMSNorm:一次完成 `x+residual`、行 RMS 归一化、权重缩放,**并额外输出融合后的残差**。用于 Transformer 的 pre-norm 结构(残差直通给下一层)。
**迁移后 NPU 侧算法**:逐行一核 + 行内对齐 `(1, n_align)` tile + fp32 域 + `vsqrt`/`vdiv` 组合。

### §1.2 计算语义

- 输入:$x, \text{residual} \in \mathbb{R}^{M\times N}$,$w \in \mathbb{R}^{N}$,$\epsilon$
- $s = x + \text{residual}$;$\text{rstd} = 1/\sqrt{\frac{1}{N}\sum_j s_j^2 + \epsilon}$
- 输出:$y = s\cdot\text{rstd}\cdot w$,**`residual_out` = s**

### §1.3 算子特征分析

| 维度 | 判定 |
|---|---|
| 计算类型 | **纯 Vector + 行归约**(无 Cube) |
| 复杂度级别 | **单步融合**(一次读入 → 行归约 → 广播 → 两次写出) |
| 动态 shape | **无** |
| host 预处理 | reshape 到 `(M, N)` |
| **输出数** | **2**(本批唯一的多输出算子) |

### §1.4 算法描述

对 $M$ 行**逐行一核**(`T.Kernel(M)`);每核读入整行三路输入,在 fp32 域完成 §0.3 的九步,写出两路输出。归约维不跨核。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R —— 完整调研)

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **无可更省的等价形式;但 `1/sqrt` 有等价改写且是必需的** | 候选:① **基线四式(采纳)**;② 把 `x+residual` 与 norm 拆成两算子 —— **反而不省**(多一次 GM 往返,roofline 从 4 路读/写涨到 6 路);③ `rstd` 用 `vrsqrt` 单指令 —— **精度不足(2.7e-3 > fp16 容差 1e-3)**,被 §0.6 C1 否决;④ `sqrt(v)/v` 组合(**采纳**,与实数域恒等)。 |
| **R2 在线算法** | **无(本算子)** | 结构依据:归约维**必须整体驻留**(一核一行),无分块累加的必要 —— 因为 manifest 的最大 $N$ 为 **16384**,`(1, n_align)` 的 fp32 缓冲仍远小于 UB 预算(§4.3 详算)。⇒ 与 `vector_norm` 不同,本算子**不需要**分块变体。 |
| **R3 复杂度四口径** | 基线:**FLOPs = 5·M·N**(add 1 + square 1 + sum 1 + mul-by-rstd 1 + mul-by-weight 1)、**访存 = (4·M·N + N)·elem_bytes**(读 x/residual/weight 3 路 + 写 y/residual_out 2 路,weight 只读 1 行)、**扫描遍数 = 1**、**中间缓冲 = 0** | 与 manifest roofline **逐字一致**(`flops: 5*M*N`,`bytes: (4*M*N+N)*elem_bytes`) |
| **R4 硬件亲和** | 基线亲和:纯 Vector;`T.reduce(..., reduce_mode="sum")` 为本仓既有归约形态;`vadd`/`vmul`/`vsqrt`/`vdiv` 均为 npui 原语;**`vrsqrt` 精度不足**须避开;行内对齐是进入向量归约路径的前提 | 逐候选评估;C1 因精度被拒 |

**调研结论**:选定 = **基线四式 + `vsqrt/vdiv` 组合 + 逐行一核**。

**设计期 roofline 估算下界(D-2)**

```
流量项 = (4·M·N + N)·elem_bytes / HBM带宽
发射项 = 5·M·N / 向量吞吐
容量项 = (1, n_align) × 多块 fp32 缓冲 / UB容量     ← 见 §4.3
估算下界 = max(流量项, 发射项, 容量项)
```

**设计期结论**:本算子**读 4 路写 2 路**(流量项显著高于逐元素族的 2 路),故**流量项是主导项**(与逐元素族的"实测非带宽受限"不同,本算子数据量更大、算术更少)。这一点应在 Stage 3 用实测校验,并记入 §10 R6。

#### §1.6.1 数学等价优化(公式级)

**结论:公式层**无更省形式**,但含一条**必需的等价改写**($1/\sqrt{v} \to \sqrt{v}/v$)**

| 优化类别 | 结论 | 依据 |
|---|---|---|
| 恒等变形降代价 | **一条必需改写**:`1/sqrt(v)` → `sqrt(v)/v`(因 `vrsqrt` 精度不足)—— **这是精度驱动的等价改写**,非性能优化 | §0.6 重设计项 3 |
| 稳定化变形 | 不适用 | 无 `exp`;`eps` 已保证分母非零 |
| 公共子表达式消除 | 已应用 | `s` 被三处复用(add 结果 / 平方和 / 两路输出);`sq_f32` 原位复用 |
| 算子降代换 | 不适用 | 无更低代换空间 |
| 访存量削减 | **已最优且已融合** | 读 3 路写 2 路 = 融合后的下界;若拆算子会增至 6 路 |
| 归约结构优化 | 不适用 | 归约维整体驻留(无分块需求,见 §1.6.0 R2) |

> 因**无性能类采纳项**(唯一改写是精度驱动的),`verify_equiv.py` **豁免**。
> ⚠️ 与 `ceil`/`floor` 同理:该豁免**不掩盖** §0.6 重设计项的等价性论证责任。

#### §1.6.2 向量化替代分析

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | `s = x + residual` | 逐元素 | **已向量化替代** | `T.vadd` |
| V2 | `sq = s*s` | 逐元素 | **已向量化替代** | `T.vmul` |
| V3 | **行求和** | 归约 | **已向量化替代** | `T.reduce(..., reduce_mode="sum", size=[1, n_align])` |
| V4 | `mean = sum/N`、`v = mean+eps` | **标量** | **保留(不可替代)** | 理由类别:**每核 1 个标量的常数折叠/配置计算**,在 `(1,1)` 缓冲上完成 |
| V5 | `t = vsqrt(v)`;`rstd = vdiv(t, v)` | **标量** | **保留(不可替代)** | 同上;**唯一需要解释的标量点**:`1/sqrt` 的等价改写(§0.6 重设计项 3) |
| V6 | `y = s*rstd*w` | 逐元素 | **已向量化替代** | `T.vmul`×(与广播的 `rstd`/`w` 相乘) |
| V7 | `residual_out = s` | 逐元素 | **已向量化替代** | 直接 `T.copy`(残差直通,无计算) |
| V8 | 对齐尾部清零 | 逐元素 | **已向量化替代** | `T.clear` |
| V9 | `real_m`/grid 计算 | block/host 级 | **保留(不可替代)** | block 级索引 / host 元数据 |

**结论**:逐元素热点(V1/V2/V6/V7/V8)与行归约(V3)全部向量化;保留的 V4/V5/V9 为**每核标量级**(每核 1 个,不是逐元素),**无逐元素标量残留**。

#### §1.6.3 向量化轴与数据布局决策

**归约算子,轴选择**:

- **候选轴枚举**:**2 个** —— ① 沿 $M$(未归约维)分核;② 沿 $N$(归约维)分核再合并。
- **评分**:

| 方案 | 归约顺序影响 | 容量 | 判据 |
|---|---|---|---|
| ① **沿 $M$ 分核(主选)** | **无影响**(归约维在核内完整完成) | `(1, n_align)` × 多块 fp32,最大 `N=16384` ⇒ 见 §4.3 | **语义最优** + 容量满足 |
| ② 沿 $N$ 分块再合并 | **改变加法结合顺序** | 容量更省 | **不采纳**:本算子 $N$ 最大 16384,容量无压力 ⇒ 无理由引入语义风险 |

- **行内对齐**:`n_align = _aligned_cols(N, dtype)`,使 `T.reduce` 的 `size=[1, n_align]` 覆盖整数对齐长度;尾部 `T.clear`。
- **决策落地**:主选 = 沿 $M$ 分核 + 行内对齐,是 §3.3 与 §5 的唯一依据。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无**(纯 Vector + 行归约) |
| 是否需要手工同步 | **否** —— 块内为线性序列 |
| 是否需要控制 L1/L0 层级 | **否**(只用 UB) |
| 是否需要跨核通信 | **否**(**归约维不跨核**) |
| 与既有实现一致性 | 本仓 `kernels/norm/{fused_add_rms_norm,ada_layer_norm}` 同为 Developer |

> ⚠️ **落地形式是硬约束**:`TILELANG_ASCEND_MODE` **必须以环境变量形式生效**(`os.environ.setdefault`)。普通 python 变量**不设置环境变量**;在含归约的算子上,缺它会被降到 Cube 路径并使 `alloc_shared` 落 `cbuf`(L1),导致 `cbuf → fragment` copy 编不过。本算子按 `vector_norm` 实测的结论(**E1**)要求交付件显式导出该环境变量。

---

## §3 API 映射设计

### §3.1 公式拆解

| 步 | 计算 | 对应式 |
|---|---|---|
| 1 | $s = x + \text{residual}$ | 融合 add |
| 2 | $\text{sum} = \sum_j s_j^2$;$\text{mean} = \text{sum}/N$ | 行归约 |
| 3 | $\text{rstd} = 1/\sqrt{\text{mean}+\epsilon}$ | 归一化因子 |
| 4 | $y = s\cdot\text{rstd}\cdot w$;`residual_out` $= s$ | 双输出 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| grid 划分(逐行一核) | `T.Kernel(M, is_npu=True) as (pid_m, _)` | 既有实现 `:106` |
| **UB 缓冲** | **`T.alloc_ub((1, n_align), dtype)`** / `(1, n_align), "float32"` / `(1, 1), "float32"` | 既有实现 `:107-116` |
| GM → UB | `T.copy(x[pid_m:pid_m+1, 0:N], x_tile[0:1, 0:N])`(三路同法) | 既有实现 `:126-128` |
| **对齐尾部清零** | `T.clear(x_tile)` / `T.clear(r_tile)` / `T.clear(w_tile)`(`tail_cols > 0` 时) | 既有实现 `:118-124` |
| 升 fp32 | `T.vcast(x_tile, s_f32)` | 既有实现 `:131-132` |
| 加 / 乘 / 开方 / 除 | `T.vadd` / `T.vmul` / `T.vsqrt` / `T.vdiv` | 既有实现 `:133/138/156/157` |
| **行归约** | **`T.reduce(sq_f32, rstd, dims=1, reduce_mode="sum", size=[1, n_align])`** | 既有实现 `:140-146` |
| 加 eps | `T.vadd(rstd, eps, rstd)` | 既有实现 `:147` |
| **`1/sqrt` 组合** | **`T.vsqrt(rstd, rsqrt_tmp)` + `T.vdiv(rsqrt_tmp, rstd, rstd)`** | 既有实现 `:156-157`;**见 §0.6 重设计项 3** |
| **广播** | 以标量元素索引 **`rstd[0, 0]`** 参与 `T.vmul` | 既有实现 `:160`;既有注释:"the only broadcast the npuir v-ops accept here" |
| 降回 dtype | `T.vcast(..., round_mode="rint")` | 既有实现 `:134/161/165` |
| **双输出** | `out_idx=[3, 4]`;`y` 与 `residual_out` 各一次 `T.copy` | 既有实现 `:94/135/166` |

> ⚠️ **四条硬约束(既有实现注记 + 本批实测)**:
> 1. **`T.vrsqrt` 精度不足**(~2.7e-3 > fp16 容差 1e-3)⇒ 必须用 `vsqrt + vdiv`
> 2. **`T.vdiv` 要求两个操作数形状相同** ⇒ 开方结果须放进**独立的 `(1,1)` 缓冲**(`rsqrt_tmp`),不能塞进行宽缓冲
> 3. **广播只能以标量元素索引形式**(`rstd[0, 0]`)参与 `T.vmul` —— 这是 npuir v-ops 在此处接受的唯一广播形式
> 4. **`avg_factor = 1/N` 在归约前乘入**(`sq_f32 *= avg_factor`),而非归约后再除 —— 与既有实现及 golden 一致

### §3.3 伪代码

```python
_ALIGN_ELEMS = {"float16": 16, "bfloat16": 16, "float32": 8}

def _fused_add_rms_norm_kernel(M, N, eps, dtype):
    n_align = _align_up(N, _ALIGN_ELEMS[dtype])     # 32B 对齐(按元素计)
    tail_cols = n_align - N
    avg_factor = 1.0 / float(N)

    @tilelang.jit(out_idx=[3, 4], target="npuir")   # ★ 双输出
    def _func(block_m=1):
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype),
                 residual: T.Tensor((M, N), dtype),
                 weight: T.Tensor((N,), dtype),
                 y: T.Tensor((M, N), dtype),
                 residual_out: T.Tensor((M, N), dtype)):
            with T.Kernel(M, is_npu=True) as (pid_m, _):
                x_tile   = T.alloc_ub((1, n_align), dtype)
                r_tile   = T.alloc_ub((1, n_align), dtype)
                w_tile   = T.alloc_ub((1, n_align), dtype)
                s_tile   = T.alloc_ub((1, n_align), dtype)
                out_tile = T.alloc_ub((1, n_align), dtype)
                s_f32    = T.alloc_ub((1, n_align), "float32")
                sq_f32   = T.alloc_ub((1, n_align), "float32")
                w_f32    = T.alloc_ub((1, n_align), "float32")
                rstd      = T.alloc_ub((1, 1), "float32")
                rsqrt_tmp = T.alloc_ub((1, 1), "float32")

                if tail_cols > 0:                 # 对齐尾部清零(0 是 sum 的单位元)
                    T.clear(x_tile); T.clear(r_tile); T.clear(w_tile)

                T.copy(x[pid_m:pid_m+1, 0:N], x_tile[0:1, 0:N])
                T.copy(residual[pid_m:pid_m+1, 0:N], r_tile[0:1, 0:N])
                T.copy(weight[0:N], w_tile[0, 0:N])

                # s = x + residual (fp32域),并直通写出 residual_out
                T.vcast(x_tile, s_f32); T.vcast(r_tile, sq_f32)
                T.vadd(s_f32, sq_f32, s_f32)
                T.vcast(s_f32, s_tile, round_mode="rint")
                T.copy(s_tile[0:1, 0:N], residual_out[pid_m:pid_m+1, 0:N])

                # rstd = 1/sqrt(mean(s^2) + eps)
                T.vmul(s_f32, s_f32, sq_f32)          # s^2
                T.vmul(sq_f32, avg_factor, sq_f32)    # ★ 1/N 在归约前乘入
                T.reduce(sq_f32, rstd, dims=1, reduce_mode="sum", size=[1, n_align])
                T.vadd(rstd, eps, rstd)
                T.vsqrt(rstd, rsqrt_tmp)              # ★ 不用 vrsqrt(精度不足)
                T.vdiv(rsqrt_tmp, rstd, rstd)         # 1/sqrt(v) == sqrt(v)/v

                # y = s * rstd * weight   (广播靠 rstd[0, 0])
                T.vmul(s_f32, rstd[0, 0], s_f32)
                T.vcast(s_f32, out_tile, round_mode="rint")
                T.vcast(out_tile, s_f32)              # 回 fp32 继续算
                T.vcast(w_tile, w_f32)
                T.vmul(s_f32, w_f32, s_f32)
                T.vcast(s_f32, out_tile, round_mode="rint")
                T.copy(out_tile[0:1, 0:N], y[pid_m:pid_m+1, 0:N])
        return main
    return _func
```

> 注:既有实现为 `dtype != "float32"` 与 `float32` 两条分支(`use_transit`);伪代码展示的是 fp16/bf16 路径(manifest 的活跃路径)。

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| **既有 fused_add_rms_norm NPU 实现(含 `vsqt/vdiv` 偏差说明)** | `examples/TileOPs/tileops/kernels/norm/fused_add_rms_norm/_fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py` |
| **仓内同算子的 NPU 参考(`single_n.py`,本实现的骨架来源)** | `examples/norm/fused_add_rms_norm/kernels/single_n.py` |
| 同族(同 `vsqrt+vdiv` 组合的既有约定) | 本仓 `kernels/norm/ada_layer_norm/…` |
| 已走完整流程的同批算子 | `examples/{abs,ceil,floor,exp,cos,vector_norm}/…` |
| `T.reduce` 的 `reduce_mode`/`size` 用法 | `tilelang/language/customize_npuir.py` |

---

## §4 数据规格与内存规划

### §4.1 数据规格

| 张量 | shape | dtype | 所在 | 分配 API |
|---|---|---|---|---|
| `x` | `(M, N)` | fp16 / bf16(f32 另有分支) | GM | `T.Tensor` 入参 |
| `residual` | `(M, N)` | 同 `x` | GM | `T.Tensor` 入参 |
| `weight` | `(N,)` | 同 `x` | GM | `T.Tensor` 入参 |
| `y` | `(M, N)` | 同 `x` | GM | 出参(`out_idx` **第 1 个**) |
| `residual_out` | `(M, N)` | 同 `x` | GM | 出参(`out_idx` **第 2 个**) |
| `x_tile` | `(1, n_align)` | 同 `x` | UB | `T.alloc_ub` |
| `r_tile` | `(1, n_align)` | 同 `x` | UB | `T.alloc_ub` |
| `w_tile` | `(1, n_align)` | 同 `x` | UB | `T.alloc_ub` |
| `s_tile` | `(1, n_align)` | 同 `x` | UB | `T.alloc_ub`(承 `s` 的 dtype 副本,写出 `residual_out`) |
| `out_tile` | `(1, n_align)` | 同 `x` | UB | `T.alloc_ub`(承 `y`) |
| `s_f32` | `(1, n_align)` | float32 | UB | `T.alloc_ub`(主累加/计算域) |
| `sq_f32` | `(1, n_align)` | float32 | UB | `T.alloc_ub`(平方 / 中转) |
| `w_f32` | `(1, n_align)` | float32 | UB | `T.alloc_ub`(weight 的 fp32 副本) |
| `rstd` | `(1, 1)` | float32 | UB | `T.alloc_ub`(逐行标量) |
| `rsqrt_tmp` | `(1, 1)` | float32 | UB | `T.alloc_ub`(**`vdiv` 需等形操作数**,故独立) |

> **层级 ↔ API 对照**:GM = `T.Tensor`;**UB = `T.alloc_ub`**(本算子既有实现即用 `T.alloc_ub`);L1 = `T.alloc_shared`。
> ⚠️ **本算子不用 `alloc_shared`** —— 与 `vector_norm` 的 `alloc_shared` 形态不同,故**不触发 E3 的层级歧义**;但仍需 Developer 环境变量生效(§2 硬约束)。

### §4.2 搬运路径

```
GM(x, residual, weight) --T.copy--> UB(dtype tiles) --vcast--> UB(fp32 域)
   --vadd/vmul/reduce/vsqrt/vdiv--> rstd(标量)
   --广播(vmul 载 rstd[0,0])--> UB(out_tile) --T.copy--> GM(y)
                               UB(s_tile)   --T.copy--> GM(residual_out)
```

### §4.3 UB 容量预算(**本算子最紧的一项,须如实核算**)

**最坏活跃 workload**:`llama-3.1-70b-prefill` = `(2048, 8192)` / **fp16**,`n_align = align_up(8192, 16) = 8192`。

| buffer | 尺寸表达式 | fp16 峰值 |
|---|---|---|
| x_tile | n_align × elem_bytes | 16 KB |
| r_tile | n_align × elem_bytes | 16 KB |
| w_tile | n_align × elem_bytes | 16 KB |
| s_tile | n_align × elem_bytes | 16 KB |
| out_tile | n_align × elem_bytes | 16 KB |
| s_f32 | n_align × 4 | 32 KB |
| sq_f32 | n_align × 4 | 32 KB |
| w_f32 | n_align × 4 | 32 KB |

**合计**:176 KB。

> `rstd` 与 `rsqrt_tmp` 均为 `(1,1)` fp32 = 4 B,不进上表。

UB 容量上限 192 KB,自动多缓冲膨胀系数 1.7 → 有效预算约 113 KB(来源:`pattern-library/constants.md` 的 `CONST-capacity-910B2C`)。

**⚠️ 如实标注:176 KB 落在两个阈值之间**

| 判据 | 阈值 | 176 KB 的结果 |
|---|---|---|
| 原始容量上限 | 192 KB | ✅ **未超** |
| 膨胀折算后的有效预算 | 113 KB | ❌ **超出** |

**如何理解这个落差(不掩盖、也不误判)**:
1. **膨胀系数(`~1.7`)针对的是自动多缓冲**(为软件流水复制缓冲)。本 kernel **不含** `T.Pipelined`、无多阶段重叠 ⇒ 该系数**是否适用于此存疑**;113 KB 是**指导值**而非硬约束。
2. 上表列的是**声明并集**(两个 dtype 分支、以及各阶段的缓冲),**不是峰值活跃集**。按 §3.3 的序列逐一分析活跃区间,峰值约为 6~7 块。
3. **既有实现已在 `(2048, 8192)/fp16` 上编译并测试通过**(28 用例族内的该 workload 在集成侧通过)⇒ 实际分配**在硬件上限内**。

**处置**:本设计**不通过删减表项来规避门禁**;若机械门禁因此报 `fail`,**如实记录并在 §10 R1 中给出处置**(属"声明并集超指导值、原始上限未超"的已知情况,须由 Stage 2 裁决是否需要收紧缓冲)。

### §4.4 中间缓冲峰值

**0** —— 无跨核部分和、无 workspace(归约维不跨核)。

---

## §5 Tiling 策略

### §5.1 Block 划分与 Tile Shape

| 项 | 值 |
|---|---|
| 划分维度 | **未归约维 $M$** |
| Tile Shape | `(1, n_align)`(整行) |
| Block 数 | **`M`**(一核一行) |
| `block_m` | 固定为 **1**(保留参数仅为与 GPU 工厂签名兼容) |

### §5.2 非整除与对齐处理策略

两件事**分开**:
1. **行数不整除**:本算子逐行一核,**无行维分块**,故无 `real_m` 尾块问题。
2. **行内对齐**:`n_align = align_up(N, _ALIGN_ELEMS[dtype])`;**对齐尾部显式 `T.clear`**;源/目的 `T.copy` 只覆盖 `[0, N)`。

### §5.3 分核策略(三要素)

**① 逻辑核数计算**:`逻辑核数 = M`(一核一行)。

| workload | (M, N) | dtype | 逻辑核数 |
|---|---|---|---|
| smoke-small | (64, 512) | fp32/fp16/bf16 | 64 |
| 8b-prefill | (2048, 4096) | fp16/bf16 | 2048 |
| 8b-decode | (1, 4096) | bf16 | **1** |
| 70b-prefill | (2048, 8192) | fp16/bf16 | 2048 |

**② 物理核数依据(实查,非假设)**

```
torch.npu.get_device_properties(0) → device: Ascend910B2C
    cube_core_num   = 24
    vector_core_num = 48      ← 本算子为纯 Vector,取此值
```

**③ 规模判定与分核方案**

| workload | 逻辑核数 | 物理核数 | 倍数 | 判定 | 方案 |
|---|---|---|---|---|---|
| smoke-small | 64 | 48 | 1.33× | 中等规模 | 无需适配 |
| 8b-prefill | 2048 | 48 | 42.7× | 中等规模 | 无需适配 |
| **8b-decode** | **1** | 48 | **0.02×** | **极小规模(decode 单行)** | **无需适配**:仅 1 行 ⇒ 只用 1 个核,是**算子语义决定的**(decode 阶段只有 1 个 token);核间无并行度可用,也**不应**为此拆分归约维(会引入结合顺序差异) |
| 70b-prefill | 2048 | 48 | 42.7× | 中等规模 | 无需适配 |

**结论:三选一命中「无需适配依据」**。特别说明:`8b-decode` 的 `M=1` 使**全算子只用 1 个核** —— 这是 **decode 场景的固有形态**(单 token),**不构成缺陷**,也不该通过拆分归约维来"提升并行度"(那会改变归约顺序)。该形态已由 manifest 显式覆盖(专门的 `-decode` workload)。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| `T.Parallel` | **不使用** | 逐元素变换由 `vadd`/`vmul`/`vcast` 覆盖 |
| **`T.reduce`** | **使用 × 1** | 行求和(`reduce_mode="sum"`,`size=[1, n_align]`) |
| `T.serial` | **不使用** | 一核一行,核内无任务循环 |
| `T.Pipelined` | **不使用** | 无多阶段可重叠(也正是 §4.3 中"膨胀系数是否适用"存疑的原因) |
| `T.Persistent` | **不使用** | 无核常驻需求 |

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入 |
| 手工同步点 | **无** |
| 依赖性质 | 线性:三路读入 → fp32 计算 → 标量归约 → 广播回写 → 两路写出 |
| 跨核同步 | **不需要**(归约维不跨核) |

---

## §8 CV 融合设计

**n/a** —— 纯 Vector + 行归约,不涉及 Cube 计算。

---

## §9 验证方案

### §9.1 Golden 函数草案(独立于 NPU 算法)

```python
def golden_fused_add_rms_norm(x, residual, weight, eps=1e-6):
    """PyTorch CPU 参考实现 —— 按 §0.1 的四式,返回 (y, residual_out)。"""
    s_f32 = (x.to(torch.float32) + residual.to(torch.float32))
    rstd = torch.rsqrt(s_f32.square().mean(dim=-1, keepdim=True) + eps)
    y = (s_f32 * rstd * weight.to(torch.float32)).to(x.dtype)
    return y, s_f32.to(x.dtype)
```

> 与既有测试文件头的 `_reference` **同式**(四式逐条一致);它**不**复刻 NPU 侧的 `vsqrt+vdiv` 组合与对齐细节,保持验证独立性。

### §9.2 L0 门槛测试计划

| 用例 | (M, N) | dtype | 说明 |
|---|---|---|---|
| smoke-small | (64, 512) | fp32 / fp16 / bf16 | 三个 dtype 全跑 |
| 8b-prefill | (2048, 4096) | fp16 / bf16 | 主 workload |
| **70b-prefill** | **(2048, 8192)** | fp16 / bf16 | **容量最紧**(176 KB 声明并集) |
| **8b-decode** | **(1, 4096)** | bf16 | **M=1**,单核路径 |

容差与既有测试一致(fp32 `1e-5` / fp16 `1e-3` / bf16 `1.6e-2`)。

**L1(阻塞层)必须包含的用例**:

| 类别 | 用例 | 为什么放在**阻塞层** |
|---|---|---|
| **双输出各自验证** | **`y` 与 `residual_out` 都要逐元素断言** | 本算子是**双输出**;只验 `y` 会漏掉"残差直通错" |
| manifest 全部活跃 workload | 上表四档(含 `M=1` 与 `N=8192`) | 覆盖容量最紧与最小规模 |
| **非 16 对齐的 N** | `N = 4095 / 8191 / 100`(**故意不满足 16 元素对齐**) | **对齐尾部清零路径的唯一覆盖** —— 若尾部未清零,`sum` 会被污染 |
| $N$ 极小 | `N = 1 / 2 / 16` | 对齐与归约的边界 |
| `M` 极小 | `M = 1`(已在 decode workload 覆盖) | 单核路径 |
| **`eps` 生效** | 全零行(`s ≡ 0`)⇒ `rstd = 1/sqrt(eps)`;断言输出为**有限值**(无除零) | `eps` 的**唯一必要场景** |
| **`rstd` 精度** | 与 fp32 golden 比对,容差按 dtype | **`vrsqrt` 被拒的直接理由**;须验证 `vsqrt+vdiv` 组合确实满足 fp16 容差 |
| `NaN` 传播 | 含 NaN 的 `x`/`residual` | 经 `vadd`/`vmul`/`reduce` 应自然传播(与 `reduce_max` 情形不同) |
| **`residual_out == x + residual`** | 独立断言(不依赖 `y`) | 第二输出的**契约级**验证 |
| 权重边界 | `weight ≡ 0`(输出应全 0)、`weight ≡ 1`(退化为纯 RMSNorm) | 广播路径的边界 |

### §9.3 与既有实现的关系

> 本算子已有手工 NPU 实现(`_fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py`),其结构与本设计**一致**;其 docstring 已记录 `vrsqrt` 精度不足的偏差与其来源(`ada_layer_norm` 的同一约定)。
>
> 差异仅在:本设计把 §9.2 的 **`y`/`residual_out` 双输出分别断言**、**非 16 对齐 N**、**`eps` 生效(全零行)**、**`rstd` 精度**明确列为 L1 阻塞用例。
> **本设计不修改既有文件**;是否以本设计为准由 Stage 2 裁决。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| **R1** | **UB 声明并集 176 KB 超"膨胀折算"的 113 KB 指导值**(原始 192 KB 上限未超) | **高** | §4.3 已如实核算与解释(膨胀系数针对多缓冲,本 kernel 无 `T.Pipelined`;声明并集≠峰值活跃集;既有实现已在最紧 workload 上通过)。**Stage 3 须在 `(2048,8192)/fp16` 上实测确认可编译可运行**;若失败则须收紧缓冲(可复用 `sq_f32`/`w_f32` 的活跃区间) |
| **R2** | **`T.vrsqrt` 精度不足**(~2.7e-3 > fp16 容差 1e-3) | **高** | §0.6 重设计项 3 已给出等价改写 `sqrt(v)/v`;§9.2 把 `rstd` 精度列入 L1 |
| **R3** | **`T.vdiv` 要求等形操作数** ⇒ 开方结果须独立 `(1,1)` 缓冲 | 高 | §3.2 陷阱 2;既有实现已如此安置 `rsqrt_tmp` |
| **R4** | **广播只能以标量索引 `rstd[0, 0]` 形式**参与 `T.vmul` | 高 | §3.2 陷阱 3(既有注释原文:"the only broadcast the npuir v-ops accept here") |
| **R5** | **对齐尾部必须清零**,否则污染 `sum` | 高 | §0.6 重设计项 2;§9.2 用**非 16 对齐的 N**(4095/8191/100)做阻塞覆盖 |
| **R6** | **§1.6.0 的"流量项主导"结论未经实测** | 中 | 本算子读 3 路写 2 路(逐元素族为 1 读 1 写),流量项显著更高;但 harness 迁移**跳过 Stage 4**,故**实测瓶颈未测** —— **如实记录,不用推测代替实测** |
| **R7** | **`M=1`(decode)只用 1 个核** | 低 | §5.3 ③ 已说明这是 decode 场景固有形态,**不构成缺陷**;不得为提升并行度而拆分归约维(会改变结合顺序) |
| **R8** | **`avg_factor` 在归约前乘入** | 中 | §3.2 陷阱 4;须与 golden(`mean` 在归约后取)在数值上一致 —— 两者数学等价,但浮点舍入点不同;§9.2 的精度用例覆盖 |
| R9 | 技术约束检测结论 | — | 三维 Kernel ❌;GPU 专用 API ❌;GEMM 非整除 ❌;L0C 溢出 ❌ → **无技术约束冲突** |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/fused_add_rms_norm/fused_add_rms_norm_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/fused_add_rms_norm/fused_add_rms_norm_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**(§1.6.1 无性能类采纳项;唯一改写是精度驱动的 `1/sqrt` 等价式) |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/norm/fused_add_rms_norm/…` | ✅ Stage 0 已产出(manifest 实际为 `norm.yaml`,bench 为 `bench_fused_add_rms_norm.py`) |
| 5 | npuir kernel + golden + 分层测试 | `examples/fused_add_rms_norm/fused_add_rms_norm_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |
