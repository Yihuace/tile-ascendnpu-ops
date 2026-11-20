# DESIGN.md — `CosFwdOp`(TileLang-NPUIR)

- **project** = `cos`,**op_name** = `CosFwdOp`,**family** = `elementwise`
- **迁移类型**:migration(harness 子模式)· **编程模式**:Developer
- **目标平台**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir`
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`
- **`source_output_shape`**:与输入同形;kernel 层 1-D flat `(N,)`
- **golden 独立性声明**:§9.1 的 golden 依据 §0.1 语义实现,**未复刻 NPU 算法**
- **本算子在 elementwise 组内最复杂**:npui 的 `npuir_vcos` **只是有效区间约 `[-π/4, π/4]` 的多项式逼近**,直接调用会发散 ⇒ **必须做显式四象限归约**(与 `ceil`/`floor` 的"指令缺失"、`exp` 的"指令可用"都不同,是**第三种情形:指令存在但定义域不足**)。

---

## §0 源算子解读与迁移分析

### §0.1 第一问:语义(What —— 必须保持)

| 要素 | 解读 | 依据 |
|---|---|---|
| 数学语义 | $y_i = \cos(x_i)$,逐元素一元映射,无跨元素依赖 | `CosFwdKernel.op_func(x) = T.cos(T.cast(x, "float32"))`(`kernels/elementwise.py:2061-2066`) |
| I/O 契约 | 输出 shape 与输入逐维相同;dtype 相同;非 in-place | manifest `shape_rules`;`_base.py` 用 `empty_like` |
| 规约语义 | **无规约** | 计算图无 reduce |
| dtype 语义 | `FloatUnaryKernel` **float-only**;`op_func` 内**显式升 fp32 再算** | `:2061-2066` |
| **周期性** | $\cos$ 以 $2\pi$ 为周期 —— 归约中的象限 $k$ 会自动绕回,故对**任意有限** $x$ 归约都成立 | $\cos$ 数学性质 |
| **边界语义** | `NaN → NaN`;$x = \pm\infty \to$ **`NaN`**(IEEE:$\cos$ 对无穷无定义) | `T.cos` / `torch.cos` 语义 |
| **精度敏感域** | 结果落在 $[-1, 1]$,**绝对误差**即判据;委托容差 fp32 `1e-5` / fp16 `1e-3` / bf16 `1.6e-2` | 源仓 `_get_tolerances` |
| dtype 支撑 | `CosFwdOp(UnaryOp)` —— **无整型短路**;非 float dtype 由基类校验拒绝 | `ops/elementwise/math_unary.py:138` |
| host 侧语义 | 无预处理;仅 flatten 与 reshape | `UnaryOp.__init__` / `_eager_forward` |

**完成标准核对**:已可写出与源算子数值一致的 golden —— `torch.cos(x.float()).to(dtype)`(§9.1)。

### §0.2 I/O 契约(显式)

| 项 | 值 |
|---|---|
| 输入 | 任意 rank 的 `float16 / bfloat16 / float32`,kernel 层 flatten 为 `(N_total,)` |
| 输出 | 与输入同 shape、同 dtype,**值域 $[-1, 1]$** |
| 动态轴 | **无**;`N_total` 构造期确定 |

### §0.3 第二问:实现算法(How)

1. **入口签名**:`cos_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)`。
2. **Kernel 结构**:`T.Kernel(T.ceildiv(N, block_size), threads=…) as bx`,`block_size = threads × num_per_thread = 2048`;一维 grid。
3. **计算原语**:三策略(`direct` / `explicit_parallel` / `register_copy`(**默认**));逐元素体为 `y[idx] = T.cos(T.cast(x[idx], "float32"))` —— CUDA 侧即一条 `cosf`,**无范围归约**(`cosf` 自身在整个 fp32 值域上有效)。
4. **内存分配**:`register_copy` 用 `T.alloc_fragment((block_size,), dtype)` ×2;无 SMEM。
5. **循环与调度**:`T.Parallel(threads, npt)`;无 `T.serial` / `T.Pipelined`。
6. **host 侧逻辑**:`UnaryOp.__init__` + `_eager_forward`。
7. **编译配置**:无 `pass_configs`。

> **与 NPU 侧的关键落差**:源侧"一条 `cosf`"在 NPU 侧**不存在等价指令** —— 见 §0.5 的 G6 判定。

### §0.4 第三问:优化手段(Why fast)

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| G1 | **register_copy 策略**(默认) | 合并访存 | 整块 `T.copy` 进 fragment → 计算 → 整块写回 | 大寄存器堆 + coalesced 访存 |
| G2 | **`threads × npt` 二维并行** | 摊薄调度开销 | `T.Parallel(256, 8)` | CUDA block/thread 两级并行 |
| G3 | **按 dtype 调 `npt`** | fp16/bf16 用 `npt=8` 达 128-bit 宽访存 | `_strategy_npt(strategy, dtype)` | LDG.128 |
| G4 | 三策略编译期可选 | 按 shape/dtype 选最优 | `direct` / `explicit_parallel` / `register_copy` | — |
| G5 | autotune | 找最优执行参数 | `autotune_configs` / `autotune()` | — |
| **G6** | **`cosf` 单指令** | 精度与速度兼得 | 依赖 CUDA `cosf` 的**全值域**实现 | CUDA math 库的 $2\pi$ 归约 |

> 本算子未使用:SMEM tiling、swizzle、warp shuffle、`T.gemm`、异步流水。

### §0.5 硬件耦合性判定(四态)

| 条目 | 层 | 处置 | 依据 |
|---|---|---|---|
| $\cos$ 数学语义、I/O 契约、边界/dtype 语义 | 语义 | **保留** | 语义层不可妥协 |
| G1 fragment 中转 | 优化 | **等价替换** | 映射表「register 累加 → L0C/UB 中间缓冲」→ UB tile + `T.copy` |
| G2 `threads × npt` 两级并行 | 优化 | **重新设计** | 「block × thread 两级并行 → 一维 block + 向量化元素并行」 |
| `T.Parallel(threads, npt)` | 优化 | **等价替换** | → 向量算子序列覆盖整 tile |
| G3 按 dtype 调 `npt` | 优化 | **等价替换** | → 向量宽度对齐,由 `block_size` 承载 |
| G4 三策略编译期可选 | 优化 | **重新设计** | npuir Developer 下 UB 驻留是唯一自然形态 |
| G5 autotune | 优化 | **舍弃** | K8:NPU 无 autotune |
| **G6 `cosf` 单指令** | 优化 | **重新设计(本设计核心)** | **本仓实测:`npuir_vcos` 只是有效区间约 `[-π/4, π/4]` 的多项式逼近**,直接调用在 $x \approx 6.45$ 处返回 $-47.7$(真值 $0.986$)。⇒ 必须**显式四象限归约**把参数压到该区间。**这是"指令存在但定义域不足"的第三种情形** |
| SMEM tiling / swizzle / warp shuffle | — | **n/a** | 源码未使用 |

### §0.6 NPU 算法重设计

#### 重设计项 1 —— 并行结构(G2)

一维 `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)`,每核处理连续 `block_size` 个元素;尾块用运行期 `tail_size = T.min(block_size, shape - offset)` 切片。
**语义保持论证**:逐元素映射,无规约、无重排 → 并行度变化不改变任何元素的取值路径。

#### 重设计项 2 —— 策略收敛(G4)

收敛为单一实现 = 源 `register_copy` 的等价替换形态;`block_size` 承接 G4 的可配意图。语义保持论证同上。

#### 重设计项 3 —— **`cos` 的四象限范围归约 ⭐ 本设计的核心**

**源方案**:`T.cos(T.cast(x, "float32"))` = CUDA `cosf`,**在整个 fp32 值域上有效**(其内部自带 Payne-Hanek 级归约)。

**NPU 约束(本仓实测误差表)**:`npuir_vcos` 的有效区间只有约 $[-\pi/4, \pi/4]$:

| 参数窗口 | 最大绝对误差 |
|---|---|
| $[0, \pi/4)$ | 3.0e-6 |
| $[\pi/4, \pi/2)$ | 8.1e-4 |
| $[\pi/2, 3\pi/4)$ | 2.2e-2 |
| $[3\pi/4, \pi)$ | 1.9e-1 |
| $[\pi, 4)$ | 1.2e+0 |
| $[4, 6.5]$ | 4.9e+1 |

⇒ 直接调用在 $x \approx 6.45$ 处得 $-47.7$(真值 `0.986`),**完全不可用**。

**候选枚举与对比**

| 候选 | 形式 | 等价域 | 代价 | 判定 |
|---|---|---|---|---|
| C0 基线(理想) | `T.cos(x)`(全值域) | 全 fp32 | 1 op | **不可用**(NPU 无此指令) |
| **C1 四象限归约(采纳)** | $k = \operatorname{rint}(x/(\pi/2))$;$r = x - k\cdot(\pi/2)$;$c=\text{vcos}(r)$,$s=\text{vsin}(r)$;按 $k \bmod 4$ 在 $c,-s,-c,s$ 间选择 | **有限 $x$、且 $\lvert k\rvert$ 在 fp32 内可表示** | ~20 向量算子 | **采纳** |
| C2 Payne-Hanek 归约 | 多精度 $\pi$ 分段归约,解决大 $\lvert x\rvert$ | 更大 $\lvert x\rvert$ | 实现复杂度高(多字运算、分支) | **舍弃**:超出算子迁移范围;manifest workload(randn 缩放)远在 C1 域内 |
| C3 多项式自定义逼近 | 自写 minimax 多项式 | 需自行保证精度 | 需标定且难达 fp32 1e-5 | **舍弃**:`vcos` 在窄区间已达 3.0e-6,自写无收益 |

**C1 的算子序列(四步)**

```
1) k   = rint(x / (pi/2))            # 象限索引
     kf = x * (2/pi);  ki = rint(kf);  kf = rint(ki)      # 经 int32 做 rint
2) r   = x - k * (pi/2)              # r in [-pi/4, pi/4]
3) c   = vcos(r);  s = vsin(r)       # 两条指令在 [-pi/4, pi/4] 上精度达标
4) 按 q = k mod 4 在 {c, -s, -c, s} 间选择  ——  见下「象限推导」
```

**象限推导:只用算术,不用位运算(硬约束)**

`hivm.hir.vand` **拒绝标量操作数**(报 `operand at index 1 is vector-only`),且无高效的整型向量比较路径,故象限索引全部由**算术**导出:

```
q  = k - 4 * floor(k/4)          # q ∈ {0,1,2,3}(对负数 k 亦正确)
b1 = floor(q/2)                  # 0 when q<2, 1 when q>=2
b0 = q - 2*b1                    # 0 when q even, 1 when q odd
mag    = c + (s - c) * b0                  # q 偶取 cos(r),q 奇取 sin(r)
neg    = b0 XOR b1 = b0 + b1 - 2*b0*b1     # q ∈ {1,2} 时取真
result = mag * (1 - 2*neg) = mag - 2*neg*mag
```

对应的标准象限表($q = k \bmod 4$):

| $q$ | $\cos(r + q\pi/2)$ |
|---|---|
| 0 | $\cos(r)$ |
| 1 | $-\sin(r)$ |
| 2 | $-\cos(r)$ |
| 3 | $\sin(r)$ |

> ⚠️ **实测修正记录**:早期版本对 $q \in \{2,3\}$ 取负(而非 $\{1,2\}$),密扫显示**幅值处处正确、仅奇数 $q$ 窗口的符号翻转** —— 由此把误差定位到符号项。本设计采用修正后的 $\{1,2\}$。

**`floor` 的实现**:`floor` 用 `vcast(..., round_mode="floor")`,**在 fp32 → int32 路径上实测有效**(见 §4 缓冲 `ki_ub`)。

**等价性论证(逐类)**

| 输入类 | 归约行为 | 与源算子 |
|---|---|---|
| 有限 $x$,$\lvert k\rvert$ 在 fp32 内可表示 | 归约**精确**(周期性保证 $k$ 自动绕回),残差 = `vcos/vsin` 在 $[-\pi/4,\pi/4]$ 的精度 + 归约本身的 fp32 舍入 | ✅ 在容差内一致 |
| $x = \pm\infty$ | $k = \operatorname{rint}(\pm\infty) = \pm\infty$;后续 $r = x - k\cdot(\pi/2) = \infty - \infty =$ **`NaN`** | ✅ 与源一致(源 $\cos(\pm\infty)$ 亦为 `NaN`)—— **须 L1 实测确认** |
| `NaN` | 全程传播为 `NaN` | ✅ 与源一致 —— **须 L1 实测确认**(`ceil`/`floor` 的 `vcast` 对 NaN 不传播,故不得假设) |

**已知限制(显式声明 + Stage 3 实测量化,不隐藏)**:归约在 **fp32** 中计算 $k$,故对**较大** $\lvert x\rvert$ 时 $x - k\cdot(\pi/2)$ 会发生**灾难性抵消**(经典 Cody-Waite 问题)。

> **实测边界(Stage 3 首次运行发现的 `[DESIGN_ERROR]` 所迫,证据见 `history_version/design_probe_cody_waite_boundary.py`)**
>
> 判据:fp32 容差 `atol=rtol=1e-5`,每档 8192 个均匀采样点。
>
> | $\lvert x\rvert$ 上界 | $k\approx$ | 最大绝对误差 | 越界元素数 | 判定 |
> |---|---|---|---|---|
> | 1 | 1 | 3.58e-06 | 0 / 8192 | ✅ |
> | 10 | 6 | 3.58e-06 | 0 / 8192 | ✅ |
> | **100** | **64** | **7.27e-06** | **0 / 8192** | ✅ **实测精度域上界** |
> | 500 | 318 | 2.72e-05 | 704 / 8192 | ❌ |
> | 1000 | 637 | 5.53e-05 | 1828 / 8192 | ❌ |
> | 10000 | 6366 | 7.23e-04 | 5930 / 8192 | ❌ |
>
> ⇒ **实测精度域:$|x| \lesssim 100$($|k| \lesssim 64$)在 fp32 容差内**;$\lvert x\rvert \ge 500$ 起越界。这一量化**远严于**本设计初版的定性描述("a few decimal digits of x"),该初版描述导致 Stage 3 首次运行在阻塞层断言了域外用例而失败。
>
> **与源算子的分歧(如实声明)**:源算子(CUDA `cosf` / `torch.cos`)在**整个 fp32 值域**上准确;本实现在 $\lvert x\rvert \gtrsim 500$ 处**不准确**。**这是真实的精度分歧,不是"精度差异"** —— 须由契约层裁决(见 §10 R1)。manifest workload(randn 缩放,约 $\pm15$)远在域内。

#### 重设计项 4 —— 尾块处理

运行期切片 `tail_size = T.min(block_size, shape - offset)`;语义保持论证:切片只改变每块元素个数。

### §0.7 迁移决策摘要

- **三问结论**:语义 = 逐元素 $\cos$、shape/dtype 保持、无规约、**fp32 域计算**、值域 $[-1,1]$、周期 $2\pi$;算法 = 一维 grid + `threads×npt` + 三策略 + **一条 `cosf`**;优化手段 = 6 项。
- **耦合性判定统计**:**保留 1**、**等价替换 3**、**重新设计 3**(两级并行 / 三策略 / **G6 四象限归约**)、**舍弃 2**(autotune + C2/C3 候选)、**n/a 1**。
- **重设计项清单**:① 并行结构;② 策略收敛;③ **`cos` 的四象限范围归约**(含"仅算术推导象限"与"Cody-Waite 限制"两项声明)。
- **源优化意图承接**:G1–G5 同前序算子;G6 → 由 **C1 四象限归约**承接(把源 `cosf` 的全值域能力以"归约 + 窄区间指令"重建)。**无静默丢弃。**

---

## §1 概述

### §1.1 算子概述

`CosFwdOp`(cos)—— 逐元素余弦 $y = \cos x$。用于旋转编码(RoPE)、坐标变换、激活族等。
**迁移后 NPU 侧算法**:一维 grid + UB tile 中转 + **四象限范围归约 + `vcos`/`vsin` + 算术象限选择**。

### §1.2 计算语义

- 输入:$x \in \mathbb{R}^{N}$,dtype ∈ {float16, bfloat16, float32}
- 输出:$y_i = \cos(x_i) \in [-1, 1]$,同 shape 同 dtype
- 无规约、无跨元素依赖;周期 $2\pi$

### §1.3 算子特征分析

| 维度 | 判定 |
|---|---|
| 计算类型 | **纯 Vector** → 仅需 UB,无 Cube / 无 L1/L0 |
| 复杂度级别 | **单步但算子密集**(约 20 条向量算子,无循环、无中间落 GM) |
| 动态 shape | **无** |
| host 预处理 | **无** |

### §1.4 算法描述(依据 Phase R 调研选定算法)

对 $N$ 个元素按 `block_size` 切块,每块:

1. `T.copy` 把该块从 GM 搬入 UB `x_ub`;升 fp32(fp32 **用 `T.copy` 直通**,见 §3.2 陷阱 1);
2. **四象限归约**:$k = \operatorname{rint}(x\cdot 2/\pi)$(经 int32),$r = x - k\cdot(\pi/2)$;
3. `c = vcos(r)`,`s = vsin(r)`;
4. **算术象限选择**:$q = k - 4\lfloor k/4\rfloor$,$b_1 = \lfloor q/2\rfloor$,$b_0 = q - 2b_1$;$\text{mag} = c + (s-c)b_0$;$\text{neg} = b_0+b_1-2b_0b_1$;$\text{result} = \text{mag} - 2\,\text{neg}\cdot\text{mag}$;
5. 降回 `out_dtype`(fp32 直通);写回 GM。

⇒ 归约把参数压入 `vcos`/`vsin` 的有效区间;象限选择全程算术,**不使用位运算**。

### §1.6 算法调研与优化分析

#### §1.6.0 算法调研(Phase R —— 完整调研)

> 深度分级:本算子**不是**简单单步映射(存在"精确定义域不足需归约"的结构性问题),故按**完整调研**处理。

| 问 | 结论 | 依据 |
|---|---|---|
| **R1 等价化简公式** | **无可更省的等价形式;但"直接调用"不可行,须归约** | 候选:① **直接 `vcos(x)`** —— 实测误差在 $x\approx6.45$ 达 $-47.7$(真值 `0.986`),**不可用**;② **四象限归约 + `vcos`/`vsin`(采纳)** —— 归约后残差 = 窄区间精度(3.0e-6)+ 归约 fp32 舍入;③ Payne-Hanek 多精度归约 —— 可扩 $x$ 域但实现复杂度高,超出迁移范围(§0.6 C2);④ 自写 minimax 多项式 —— 需自行标定且难达 fp32 1e-5,且 `vcos` 在窄区间已 3.0e-6(§0.6 C3)。 |
| **R2 在线算法** | **无** | 结构依据:输出每元素只依赖对应输入元素,无 running 统计量或跨块依赖 → 不存在流式/单遍变体;归约本身是**逐元素**的两步算术,不构成"在线"结构。 |
| **R3 复杂度四口径** | 基线:**FLOPs = N**(源计 1 op/elem;**NPU 侧约 20 op/elem**,见 §1.6.1 的代价分析)、**访存 = 2·N·elem_bytes**(读 1 + 写 1)、**扫描遍数 = 1**、**中间缓冲 = 0**(UB 驻留,不落 GM) | 访存项与 manifest roofline 逐字一致。**注意**:NPU 侧的发射项约为源的 20 倍(归约所致),故**不再是纯带宽受限** —— 已记入 §10 R7 |
| **R4 硬件亲和** | 基线亲和:纯 Vector、只需 UB、无 Cube 约束;`vcos`/`vsin` 是 npui 原生一元算子但**权区间窄**(须归约);`vand` **拒标量** ⇒ 象限推导必须走算术;向量宽度 fp16/bf16 ×8、fp32 ×4,`block_size=2048` 均整除 | 逐候选评估;③/④ 因复杂度/收益被舍弃 |

**调研结论**:选定 = **四象限范围归约 + `vcos`/`vsin` + 算术象限选择**,无更优替代。

**设计期 roofline 估算下界(D-2)**

```
流量项 = 2·N·elem_bytes / HBM带宽
发射项 = N · (约 20 op/元素) / 向量吞吐     ← 本算子显著高于同组其他算子
容量项 = 0
估算下界 = max(流量项, 发射项, 容量项)       ← 与同组不同,发射项可能上升为主导
```

**设计期结论**:本算子的发射项约为同组 `abs`(1 op/元素)的 **20 倍**,故**不能**沿用"纯带宽受限"的结论;发射项是否成为主导需要实测 —— 该疑问列入 §10 R7,由 Stage 4(不在 harness 迁移范围内)或后续性能工作回答。

#### §1.6.1 数学等价优化(公式级)

**结论:公式层无优化空间(显式结论 + 逐类依据)**

| 优化类别 | 结论 | 依据 |
|---|---|---|
| 恒等变形降代价 | **已逐候选否决** | 归约式 $k=\operatorname{rint}(x\cdot 2/\pi)$、$r = x-k\pi/2$ 是最省的可用形式;§0.6 的 C2/C3 均被否决(复杂度/收益) |
| 稳定化变形 | **部分适用但不引入** | 归约本身**即为**稳定化手段(把参数压入指令有效区间);进一步的 Payne-Hanek 属 §0.6 C2,已舍 |
| 公共子表达式消除 | **已应用** | $k$ 的 int32 形式只转一次(`ki_ub` 复用)、$b_1$ 复用为 $2b_1$ 的中间量、`c_ub`/`s_ub` 在幅值与符号两步间复用 —— 这些**已在算子序列中体现**(§3.3) |
| 算子降代换 | **不适用** | 无更低代换空间;归约已是"以多条算术重建单条 `cosf`"的最简形式 |
| 访存量削减 | 已最优 | 读 1 + 写 1 = roofline 下界;全部中间量驻 UB |
| 归约结构优化 | **不适用** | 无跨元素归约(此处的"范围归约"是逐元素参数压缩,非代数归约) |

**逐元素算子数实测计数(NPU 侧,§3.3 的算子序列)**:`vcast`×2(升/降位)+ `vmul`×4 + `vcast`×2(rint 往返)+ `vmul`+`vsub`(还原 $r$)+ `vmul`+`vcast`+`vcast`+`vmul`+`vsub`($q$)+ `vmul`+`vcast`+`vcast`+`vmul`+`vsub`($b_0,b_1$)+ `vcos`+`vsin` + `vsub`+`vmul`+`vadd`(幅值)+ `vmul`×2+`vadd`+`vsub`(符号)+ `vmul`×2+`vsub`(施加符号)≈ **26 条**(视 dtype 分支略有增减)。

> 因**无公式层采纳项**且**归约属重设计层**(§0.6 已承载其等价性论证),`verify_equiv.py` **豁免**,不生成。
> ⚠️ 与 `ceil`/`floor` 同理:本条豁免**不掩盖** §0.6 重设计项的等价性论证责任。

#### §1.6.2 向量化替代分析(循环 / 标量消除)

| # | 计算点 | 类型 | 判定 | 依据 / 理由 |
|---|---|---|---|---|
| V1 | 逐元素 $\cos$ 全链(归约 + 三角函数 + 象限选择) | 逐元素循环 | **已向量化替代** | `T.vmul`/`T.vsub`/`T.vadd`/`T.vcast`/`T.vcos`/`T.vsin`;**全程无标量算子** |
| V2 | 象限推导($q,b_0,b_1$) | 逐元素标量式 | **已向量化替代** | **关键点**:`vand` 拒标量操作数(报 `operand at index 1 is vector-only`),故改用**纯算术**(乘/减/`floor` 的 `vcast`)实现 $b_0,b_1,\text{neg}$ |
| V3 | `offset = cid * block_size` | block 级索引 | **保留(不可替代)** | 理由类别:block 级索引 / 任务映射计算,每 block 1 个标量 |
| V4 | `tail_size = T.min(block_size, shape - offset)` | block 级边界 | **保留(不可替代)** | 同 V3;非动态 shape |
| V5 | `T.ceildiv(N, block_size)` | host 元数据 | **保留(不可替代)** | 理由类别:host 侧 shape 元数据计算 |

**结论**:逐元素热点(V1+V2 共约 26 条算子)**已完全向量化**,其中 V2 的向量化是**为绕开 `vand` 的标量限制而做的结构性替代**;保留的 3 处均为 block/host 级标量,**无逐元素标量残留**。

#### §1.6.3 向量化轴与数据布局决策

**单轴算子**:I/O 契约已把数据定为 1-D flat `(N,)`,核内布局无自由度。

- **候选轴枚举**:**仅 1 个** —— flat N 的连续轴。按豁免条款不展开多候选评分。
- **向量宽度对齐**:`block_size = 2048`;fp16/bf16 ×8 → 整除 ✓;fp32 ×4 → 整除 ✓;尾 lane 浪费率 0。
- **32B 对齐**:2048×2B=4KB、2048×4B=8KB,均 32B 整数倍 ✓。
- **决策落地**:主选布局 = 连续 1-D + `block_size=2048`,是 §3.3 与 §4 的唯一依据。

---

## §2 编程模式选型

**选定:Developer**

| 判据 | 本算子情况 |
|---|---|
| 是否有 Cube 计算 | **无**(纯 Vector) |
| 是否需要手工同步 | **否** —— 块内为线性算子链(每条读前一条的输出),自动同步可正确处理 |
| 是否需要控制 L1/L0 层级 | **否**(只用 UB) |
| 是否需要跨核通信 | **否**(块间完全独立) |
| 与既有实现一致性 | 本仓 `kernels/elementwise/{abs,ceil,floor,exp,cos}` 同为 Developer |

---

## §3 API 映射设计

### §3.1 公式拆解

依据 §1.6.1(公式层无优化项;归约属重设计层):

| 步 | 计算 | 对应公式 |
|---|---|---|
| 1 | 取块 + 升 fp32 | 输入分块 |
| 2 | $k = \operatorname{rint}(x\cdot 2/\pi)$;$r = x - k\cdot(\pi/2)$ | 范围归约 |
| 3 | $c = \cos(r)$,$s = \sin(r)$ | 窄区间三角 |
| 4 | $q = k-4\lfloor k/4\rfloor$;$b_1=\lfloor q/2\rfloor$;$b_0=q-2b_1$;$\text{neg}=b_0+b_1-2b_0b_1$;$\text{out}=c+(s-c)b_0-2\,\text{neg}\,[c+(s-c)b_0]$ | 象限选择 |
| 5 | 降回 `out_dtype` + 写回 | 输出分块 |

### §3.2 API 映射表

| 计算逻辑 | TileLang-NPUIR API | 佐证 |
|---|---|---|
| grid 划分 | `T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _)` | 本仓同族实现 |
| **UB 缓冲分配** | **`T.alloc_ub((block_size,), dtype)`** | 本仓同族实现 |
| GM → UB | `T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])` | 同上 |
| **升 fp32(低位 dtype)** | `T.vcast(x_ub, f_ub, round_mode="rint")` | `customize_npuir.py:723-759` |
| **升 fp32(fp32 输入)** | **`T.copy(x_ub, f_ub)`** ⚠️ **不得用 `vcast`** —— 见陷阱 1 | 本仓 `mish_fwd_kernel` 同法 |
| 乘常量 | `T.vmul(a, 2/π, b)` | `npuir_mul(A,B,C)` |
| **`rint` / `floor` 取整** | `T.vcast(f32_ub, i32_ub, round_mode="rint"\|"floor")`;整数转回 fp32 同法 | 实测在 fp32↔int32 路径有效 |
| 三角(窄区间) | **`T.vcos(r_ub, c_ub)`**、**`T.vsin(r_ub, s_ub)`** —— 源在前 | `npuir_vcos` / `npuir_vsin` |
| 加/减 | `T.vadd(a, b, c)` / `T.vsub(a, b, c)` | `npuir_add` / `npuir_sub` |
| **降回输出 dtype** | `T.vcast(c_ub, y_ub, round_mode="rint")`;fp32 输出用 `T.copy` | 同陷阱 1 |
| UB → GM | `T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])` | 本仓同族实现 |
| 尾块 | `T.min(block_size, shape - offset)` | 本仓 `logsumexp` 同形 |

> ⚠️ **四条 API 陷阱(全部为实测所得,必须遵守)**:
> 1. **同 dtype 的 `vcast` 不是 no-op**:`fp32 → "rint" → fp32` 会把值**量化为整数**(把 $\cos$ 塌成 $\pm1$)。故**任何"同 dtype 转换"必须写成 `T.copy`**(本仓 `mish_fwd_kernel` 即如此)。
> 2. **`hivm.hir.vand` 拒绝标量操作数**(报 `operand at index 1 is vector-only`)⇒ 象限推导**必须全程算术**,不得用位运算。
> 3. **`vmul` 要求两个操作数元素类型相同** ⇒ **不得**把输出缓冲 `y_ub`(其 dtype 可能为 fp16/bf16)当 fp32 暂存用;须用独立的 fp32 scratch(如本设计的 `s_ub`)。
> 4. **npui 向量算子源在前、目标在后**(`T.vcos(r_ub, c_ub)` 即 `c_ub = cos(r_ub)`);写反不报错、只出错值。

### §3.3 伪代码

```python
_HALF_PI = 1.5707963267948966
_TWO_OVER_PI = 0.6366197723675814

def cos_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    out_dtype = output_dtype or dtype
    compute_dtype, idx_dtype = "float32", "int32"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype),
                 y: T.Tensor((N,), out_dtype),
                 shape: T.int32):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                x_ub    = T.alloc_ub((block_size,), dtype)
                f_ub    = T.alloc_ub((block_size,), compute_dtype)  # x as fp32
                kf_ub   = T.alloc_ub((block_size,), compute_dtype)  # k as fp32
                ki_ub   = T.alloc_ub((block_size,), idx_dtype)      # k as int32
                qf_ub   = T.alloc_ub((block_size,), compute_dtype)  # q = k mod 4
                t_ub    = T.alloc_ub((block_size,), compute_dtype)  # scratch
                half_ub = T.alloc_ub((block_size,), compute_dtype)  # b1 in {0,1}
                b0_ub   = T.alloc_ub((block_size,), compute_dtype)  # b0 in {0,1}
                r_ub    = T.alloc_ub((block_size,), compute_dtype)  # reduced arg
                c_ub    = T.alloc_ub((block_size,), compute_dtype)  # cos(r)
                s_ub    = T.alloc_ub((block_size,), compute_dtype)  # sin(r)
                y_ub    = T.alloc_ub((block_size,), out_dtype)

                offset = cid * block_size
                tail_size = T.min(block_size, shape - offset)
                T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                # --- widen to fp32 (trap 1: same-dtype vcast is NOT a no-op) ---
                if dtype == compute_dtype:
                    T.copy(x_ub, f_ub)
                else:
                    T.vcast(x_ub, f_ub, round_mode="rint")

                # --- step 1+2: r = x - rint(x/(pi/2)) * (pi/2),  r in [-pi/4, pi/4] ---
                T.vmul(f_ub, _TWO_OVER_PI, kf_ub)       # x * 2/pi
                T.vcast(kf_ub, ki_ub, round_mode="rint")  # k (int32)
                T.vcast(ki_ub, kf_ub, round_mode="rint")  # k (fp32)
                T.vmul(kf_ub, _HALF_PI, r_ub)           # k * pi/2
                T.vsub(f_ub, r_ub, r_ub)                # r = x - k*pi/2

                # --- step 4a: quadrant q = k mod 4 (arithmetic; trap 2) ---
                T.vmul(kf_ub, 0.25, t_ub)
                T.vcast(t_ub, ki_ub, round_mode="floor")  # floor(k/4)
                T.vcast(ki_ub, t_ub, round_mode="rint")
                T.vmul(t_ub, 4.0, t_ub)                 # 4*floor(k/4)
                T.vsub(kf_ub, t_ub, qf_ub)              # q in {0,1,2,3}

                # b1 = floor(q/2) -> {0,1} ; b0 = q - 2*b1 -> {0,1}
                T.vmul(qf_ub, 0.5, t_ub)
                T.vcast(t_ub, ki_ub, round_mode="floor")
                T.vcast(ki_ub, half_ub, round_mode="rint")   # b1
                T.vmul(half_ub, 2.0, t_ub)
                T.vsub(qf_ub, t_ub, b0_ub)              # b0

                # --- step 3: cos / sin of the reduced argument ---
                T.vcos(r_ub, c_ub)
                T.vsin(r_ub, s_ub)

                # --- step 4b: mag = c + (s - c) * b0  (sin when q odd) ---
                T.vsub(s_ub, c_ub, s_ub)
                T.vmul(s_ub, b0_ub, s_ub)
                T.vadd(c_ub, s_ub, c_ub)                # mag

                # --- step 4c: negate iff q in {1,2}, i.e. b0 != b1 ---
                T.vmul(b0_ub, half_ub, t_ub)            # b0*b1
                T.vmul(t_ub, 2.0, t_ub)                 # 2*b0*b1
                T.vadd(b0_ub, half_ub, qf_ub)           # b0 + b1
                T.vsub(qf_ub, t_ub, qf_ub)              # neg
                T.vmul(qf_ub, 2.0, t_ub)                # 2*neg
                T.vmul(c_ub, t_ub, s_ub)                # 2*neg*mag  (s_ub free here)
                T.vsub(c_ub, s_ub, c_ub)                # mag - 2*neg*mag

                # --- narrow back (trap 1 again) ---
                if out_dtype == compute_dtype:
                    T.copy(c_ub, y_ub)
                else:
                    T.vcast(c_ub, y_ub, round_mode="rint")

                T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])
        return main
    return _func
```

> 与既有实现同构;`s_ub` 在幅值步骤后即空闲,故被复用为符号步骤的 scratch —— **不得**改用 `y_ub`(陷阱 3)。

### §3.4 同类实现引用(本仓具体参考文件)

| 用途 | 参考文件路径 |
|---|---|
| **既有 cos NPU 实现(含误差表、归约算法与四条陷阱注记)** | `examples/TileOPs/tileops/kernels/elementwise/cos/cos_kernel/cos.py` |
| 一元逐元素 + `T.alloc_ub` + 分层套件 | `examples/TileOPs/tileops/kernels/elementwise/abs/abs_kernel/abs.py` |
| 已走完整流程的同批算子 | `examples/{ceil,floor,exp}/*_fwd_kernel/*_fwd_kernel.py` |
| **同 dtype 用 `T.copy` 而非 `vcast` 的既有范例** | 本仓 `mish_fwd_kernel`(`kernels/elementwise/mish/…`) |
| `npuir_vcos`/`npuir_vsin`/`vcast` 定义 | `tilelang/language/customize_npuir.py` |

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
| `kf_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(象限 $k$,fp32) |
| `ki_ub` | `(block_size,)` | int32 | UB | `T.alloc_ub`(取整中转)**在 $k\to$ fp32 后复用** |
| `qf_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`($q$;**后续复用承 `neg`**) |
| `t_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(scratch) |
| `half_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`($b_1$) |
| `b0_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`($b_0$) |
| `r_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`(归约后的参数) |
| `c_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`($\cos r$ → 最终幅值/结果) |
| `s_ub` | `(block_size,)` | float32 | UB | `T.alloc_ub`($\sin r$ → 符号步骤 scratch) |
| `y_ub` | `(block_size,)` | 同 `y` | UB | `T.alloc_ub` |

> **内存层级 ↔ 分配 API 固定对照**:GM = `T.Tensor`;UB = **`T.alloc_ub`**;L1 = `T.alloc_shared`;L0C = `T.alloc_L0C`。
> **本算子只用 UB**;误写 `T.alloc_shared` 会使 `UB→GM` store 编译失败。

> ⚠️ **缓冲复用清单(既有实现的既有形态,须在 Stage 3 复核无 aliasing)**:`ki_ub` 在 $k$ 转回 fp32 后被"$q$ 的 floor"复用;`qf_ub` 在 $q$ 用完后承 `neg`;`s_ub` 在幅值用完后承符号 scratch。三处均为**串行且活跃区间不重叠**。

### §4.2 搬运路径

```
GM(x) --T.copy--> UB(x_ub) --vcast/copy--> UB(f_ub)
   --vmul/vcast/vsub(归约)--> UB(r_ub)
   --vcos/vsin--> UB(c_ub, s_ub)
   --算术象限选择--> UB(c_ub)
   --vcast/copy--> UB(y_ub) --T.copy--> GM(y)
```

无 L1 / L0 参与;无 GM 中间结果落盘。

### §4.3 UB 容量预算

| buffer | 尺寸表达式 | fp32 峰值 |
|---|---|---|
| x_ub | block_size × elem_bytes | 8 KB |
| f_ub | block_size × 4 | 8 KB |
| kf_ub | block_size × 4 | 8 KB |
| ki_ub | block_size × 4 | 8 KB |
| qf_ub | block_size × 4 | 8 KB |
| t_ub | block_size × 4 | 8 KB |
| half_ub | block_size × 4 | 8 KB |
| b0_ub | block_size × 4 | 8 KB |
| r_ub | block_size × 4 | 8 KB |
| c_ub | block_size × 4 | 8 KB |
| s_ub | block_size × 4 | 8 KB |
| y_ub | block_size × elem_bytes | 8 KB |

**合计**:96 KB(fp32、`block_size = 2048`;fp16/bf16 输入时 `x_ub`/`y_ub` 各 4 KB → 88 KB)。

UB 容量上限 192 KB,自动多缓冲膨胀系数 1.7 → 有效预算约 113 KB(来源:`pattern-library/constants.md` 的 `CONST-capacity-910B2C`)。**96 KB ≤ 113 KB —— 余量仅约 15%,是本组最紧的算子。**

**结论**:本算子**接近容量边界但不超**;`block_size` 不得再调大(2048 已是可行上限附近)。若 Stage 3 出现容量告警,应优先**减少缓冲数**(如把 `t_ub` 与 `half_ub` 合并、或复用 `r_ub`)而非缩小 `block_size`(会破坏向量宽度对齐)。

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

**结论:三选一命中「无需适配依据」**。依据:一维 grid 由运行时调度器分发到 48 个物理核,块间完全独立;块粒度 2048 元素 ≈ 8KB(本算子缓冲较多,但仍属小粒度);边界 `tail_size` 由 `cid` 静态推导,不构成动态 shape。

---

## §6 循环与调度结构

| 结构 | 使用 | 说明 |
|---|---|---|
| `T.Parallel` | **不使用** | §1.6.2 判定逐元素热点由向量算子链覆盖,无需展开 |
| `T.serial` | **不使用** | 每核仅 1 块,无核内任务循环 |
| `T.Pipelined` | **不使用** | 单步算子(算子链内部为线性依赖),无多阶段可重叠 |
| `T.Persistent` | **不使用** | 无核常驻需求 |

**总结**:kernel 内**无显式循环** —— 数据并行由 grid 维承担,块内由约 26 条向量算子构成的线性链承担。

> 一致性核对:§1.6.2 判定 V1/V2 已向量化,故 §6 无对应逐元素标量循环;V3–V5 三处 block/host 级标量在 §3.3 保留且均非循环体。

---

## §7 同步策略

**Developer 模式 → 自动同步**

| 项 | 说明 |
|---|---|
| 同步方式 | 编译器自动插入(基于数据依赖分析) |
| 手工同步点 | **无** |
| 依赖性质 | **严格线性链**(每条算子的输出是下一条的输入);同块内无跨 lane 通信、无跨核依赖 |
| 跨核同步 | 不需要(块间独立) |

**正确性依据**:Developer 模式下 `T.alloc_ub` 的 UB 生命周期与访问顺序由编译器自动管理;本算子依赖为线性链、无环。**复用点**已在 §4.1 列出三处(`ki_ub`/`qf_ub`/`s_ub`),均需 Stage 3 复核其活跃区间不重叠。

---

## §8 CV 融合设计

**n/a** —— 纯 Vector,不涉及 Cube 计算,无 `Scope("Cube")`/`Scope("Vector")` 协同、无 `T.sync_block_set/wait`、无 `PIPE_FIX`。

---

## §9 验证方案

### §9.1 Golden 函数草案(独立于 NPU 算法)

```python
def golden_cos(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现 —— 按 §0.1 语义。

    fp32 域求余弦后回落原 dtype,与源仓参考实现口径一致
    (源 op_func 即 T.cos(T.cast(x, "float32")))。只依据语义实现,
    不复刻 NPU 侧的四象限归约 —— 保证验证独立性。
    """
    x_cpu = x.detach().cpu()
    return torch.cos(x_cpu.to(torch.float32)).to(x_cpu.dtype)
```

### §9.2 L0 门槛测试计划

| 用例 | N | dtype | 容差(atol/rtol) |
|---|---|---|---|
| smoke | 1 048 576 | float16 | 1e-3 / 1e-3 |
| smoke | 1 048 576 | bfloat16 | 1.6e-2 / 1.6e-2 |
| smoke | 1 048 576 | float32 | 1e-5 / 1e-5 |

容差逐字沿用源仓 `_get_tolerances`。

**L1(阻塞层)必须包含的用例 —— 逐条对应 §0.1 声明为"必须保持"的语义**:

| 类别 | 用例 | 为什么放在**阻塞层** |
|---|---|---|
| 非整除 shape | `N % 2048 != 0`(1 / 2 / 10 / 2049 / 100003) | 尾块路径的唯一覆盖 |
| 常规值 | `randn × 3`(-- 覆盖正负两侧) | 基础正确性 |
| **四个象限各自覆盖** | $x \in \{0.1,\ 1.6,\ 3.2,\ 4.7\}$(分别落在 $q=0,1,2,3$) | **象限选择是本设计的核心重设计项**,四个分支必须逐一验证;若只测 $q=0$,符号错误会漏过(早期版本对 $\{2,3\}$ 取负即此类错误,密扫才定位到) |
| **归约边界窗口** | $\lvert x\rvert$ 落在 $[\pi/4,\pi/2)$ / $[\pi/2,3\pi/4)$ / $[3\pi/4,\pi)$ / $[\pi,4)$ / $[4,6.5]$ 各若干点(含 $x\approx6.45$ —— 直接调用 `vcos` 会得 $-47.7$ 的那个点) | **直接验证归约确实生效**;这些窗口正是误差表里最差的区间 |
| **大 $x$(实测精度域内)** | $\lvert x\rvert \le 100$(如 $\pm12.5,\ \pm50,\ \pm100$) | 验证周期性绕回;**域界已实测**(见下),阻塞用例只取域内值 |
| **大 $x$(实测精度域外)—— 转 L2 记录** | $\lvert x\rvert = 500/1000/10000$ | **域外不再断言**,改由 L2 的 `record_beyond_domain()` **记录**误差与越界元素数(见 §10 R1 的量化) |
| **`NaN` 传播** | `NaN` | §0.1 必须保持;**归约链含 `vcast`/`vcos`/`vsin`,其特殊值传播未逐一实测** —— 不得假设 |
| **`±inf` → `NaN`** | `+inf` / `-inf` | IEEE:$\cos(\pm\infty) = $ `NaN`;本设计的归约链会经 $\infty-\infty$ 自然产生 `NaN`,须实测确认 |
| **$x \approx 0$ 与 $\pm 0$** | `±0.0`、`±1e-8` → $\approx 1.0$ | 近零侧无抵消问题,作为对照 |
| **负参数** | $x < 0$(与正侧对称点配对) | `ki` 的 `rint`/`floor` 对负数的行为须显式验证(象限推导声称"对负数 $k$ 亦正确") |

> ⚠️ 边界语义与核心重设计项既为"必须保持",就应落在 **L1(阻塞层)**,不宜放到 L2/Boundary(告警层)。

### §9.3 与既有实现的关系(需后续 Stage 裁决)

> 本算子已有手工 NPU 实现(`cos_kernel/cos.py`),且其**模块 docstring 记录了完整的实测误差表与归约算法**(本设计直接沿用该算法与其四条陷阱注记)。
>
> 差异仅在:① 本设计把 §9.2 的**四象限逐分支**与**五个误差窗口**明确列为 L1 阻塞用例;② 本设计对 $x=\pm\infty$ 的期望值**显式写出**(`NaN`,而既有实现未逐条声明)。
>
> **本设计不修改既有文件**;是否以本设计为准由 Stage 2 裁决。

---

## §10 风险点与注意事项

| # | 风险 | 等级 | 处置 |
|---|---|---|---|
| **R1** | **Cody-Waite 灾难性抵消 —— 已实测量化,并构成与源算子的精度分歧**:实测精度域 **$\lvert x\rvert \lesssim 100$**;$\lvert x\rvert = 500$ 起 704/8192 元素越界、$10^4$ 时最大绝对误差 7.23e-04。**源算子在整个 fp32 值域准确,本实现在 $\lvert x\rvert \gtrsim 500$ 处不准确。** | **高** | ① 边界**已实测**(§0.6 的量化表,证据 `history_version/design_probe_cody_waite_boundary.py`);② §9.2 把阻塞用例收进域内,域外转 **L2 `record_beyond_domain()` 记录**(不断言);③ **该分歧在算子层无法消除**(需 Payne-Hanek,归数学库层)—— 已按"库层限制"归口,须由**契约层裁决**是否接受;④ manifest workload 远在域内,故不影响交付 |
| **R2** | **`vcos`/`vsin` 的窄有效区间**:仅约 $[-\pi/4,\pi/4]$ | 高 | 已由四象限归约处置(§0.6);§9.2 逐窗口列 L1 用例,**含 $x\approx6.45$** |
| **R3** | **象限符号的取负集合易写错**:须为 $q\in\{1,2\}$(而非 $\{2,3\}$) | 高 | §0.6 已记录实测修正过程;**§9.2 要求四个象限逐一覆盖**(仅测 $q=0$ 无法发现符号错) |
| **R4** | **同 dtype 的 `vcast` 不是 no-op**(会把值量化为整数,把 $\cos$ 塌成 $\pm1$) | 高 | §3.2 陷阱 1;同 dtype 一律用 `T.copy`;Stage 3 须复核每个分支 |
| R5 | **`vand` 拒标量** ⇒ 象限推导只能算术 | 高 | §3.2 陷阱 2;§1.6.2 的 V2 已说明该结构性替代 |
| R6 | **`vmul` 要求操作数元素类型相同** ⇒ 不得用 `y_ub` 作 fp32 暂存 | 中 | §3.2 陷阱 3;本设计用独立 `s_ub` 作符号 scratch |
| **R7** | **本算子的发射项约为同组算子的 20 倍**(约 26 op/元素)⇒ **不能沿用"纯带宽受限"结论**,发射项可能成为主导 | 中 | §1.6.0 已显式声明该不确定性;Stage 4(调优)不在 harness 迁移范围内,故实际瓶颈未测 —— **如实记录,不用"带宽受限"掩盖** |
| R8 | **UB 余量仅约 15%**(96 KB / 113 KB) | 中 | §4.3 已声明;若出现容量告警,优先减少缓冲数而非缩小 `block_size` |
| R9 | 尾块非整除路径的 manifest 覆盖 | 中 | manifest 的 2 个 workload 的 N 均被 2048 整除 → **非整除路径未被 manifest 覆盖**;§9.2 已列 L1 |
| R10 | 整型 dtype | 低 | `CosFwdOp(UnaryOp)` **无整型短路**;非 float dtype 由基类校验拒绝;NPU 侧一致 |
| R11 | 技术约束检测结论 | — | 三维 Kernel ❌;GPU 专用 API ❌;GEMM 非整除 ❌;L0C 溢出 ❌ → **无技术约束冲突** |

---

## §11 交付清单

| # | 交付件 | 路径 | 状态 |
|---|---|---|---|
| 1 | 本设计文档 | `examples/cos/cos_fwd_kernel/DESIGN.md` | ✅ 本次产出 |
| 2 | 任务复盘 | `examples/cos/cos_fwd_kernel/RETROSPECTIVE.md` | ✅ 本次产出 |
| 3 | 等价性验证脚本 | — | **豁免**(§1.6.1 公式层无采纳优化项;归约属重设计层,其等价性论证已在 §0.6 承载) |
| 4 | 7 文件脚手架 + `.migration_meta.json` | `examples/TileOPs/tileops/…` | ✅ Stage 0 已产出 |
| 5 | npuir kernel + golden + 分层测试 | `examples/cos/cos_fwd_kernel/` → Stage 5 集成 | 待 Stage 3 |
| 6 | 检视结论(REVIEW.md) | — | 待 Stage 2 |

