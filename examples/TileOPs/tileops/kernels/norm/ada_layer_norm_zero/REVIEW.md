# ada_layer_norm 设计文档检视报告(round 1)

## 检视结论

结论: 通过

## 检视元信息

- 文档:`examples/ada_layer_norm/ada_layer_norm_kernel/DESIGN.md`
- 任务类型:迁移(harness 子模式)· **本单元一个 kernel 覆盖两个 manifest Op**
- 覆盖 Op:`AdaLayerNormFwdOp`(非门控)、`AdaLayerNormZeroFwdOp`(**门控 / AdaLN-Zero**)
- 源算子:`/tmp/tops/TileOPs-main/tileops/kernels/norm/ada_layer_norm.py`(`AdaLayerNormKernel` @305)
- 检视维度:9 项
- 阻塞级问题数:**0** · 建议级问题数:0

---

## 机械复核

- **verify_equiv 重跑**:**不适用(无性能类未决采纳项)**。§1.6.1 的两条改写(编译期倒数折叠 `N_inv = 1/N`、必需的 `1/sqrt(v) → sqrt(v)/v`)均数学等价且已论证;后者是**精度驱动**,其依据是既有 `repro/TRAP-vrsqrt-plain-precision.py`。豁免成立。
- **design_calc_check**(独立复跑):

| check | status | detail |
|---|---|---|
| `ub_budget` | **pass** | `UB: 8 buffer entries sum = 176.0 KB; capacity = 192.0 KB; x1.7 auto-multi-buffer inflation budget = 113 KB` |
| `l0c_budget` | skip | 机械跳过,**人工已核**:无 Cube 路径,不适用 |
| `core_split` | skip | 机械跳过,**人工已核**:本算子的并行结构是**持久核 + 核内串行**,不是 `ceildiv(M,bm)*ceildiv(N,bn)` 的二维形态 ⇒ 该工具的两维模型**不适用**;设计 §5.3 已按"物理核封顶 48 + 核内任务串行"给出完整三要素 |
| `r3_metrics` | **pass** | `all four R3 metrics present` |

> **首轮 `ub_budget` 曾 `skip`(非 pass)**:原因是 §4.3 只用公式表述(`block_m × N × 20 B`),无可解析的表行。**修订动作**:补上 8 条逐 buffer 表 → 复跑 `pass(176.0 KB)`。该修订已在设计内完成,本轮检视为**修订后**的版本。
> **本审阅独立确认**:`176 KB` 与既有实现的 20 B/元素口径交叉校验一致(`2 × 4096 × 20 B = 160 KB`,同量级);两者均在 192 KB 之下。

- **内部一致性复核(独立)**:无占位符;无 `§x.y.z` 标题交叉引用;§4.3 的 buffer 表为「一行一 buffer、无容量列」。

---

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自 Read 源码逐条核对**:

| `§0` 断言 | 源码核对结果 |
|---|---|
| **两个 Op 映射到同一 `kernel_map`** | ✅ manifest 两者均为 `{'ada_layer_norm': 'AdaLayerNormKernel'}`;抽取出的两份文件**字节相同**(sha256 `ae3dc679b480e8d4`) |
| **只有 1 个抽取函数**:`_ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False)` | ✅ 抽取结果 `extracted 1 TileLang kernel function(s)` |
| **`has_gate` 是两 Op 的唯一开关** | ✅ 既有实现注释原文:"`has_gate=True` (AdaLN-Zero) IS implemented here. The gated variant adds exactly one elementwise multiply by `gate` to the same epilogue (`y = gate * (scale*Norm(x) + shift)`), so the whole body is shared rather than duplicated per variant" |
| **`out_idx=[4]` 对两变体恒定**(靠签名第 4 槽占位) | ✅ 既有实现 `:231` `@tilelang.jit(out_idx=[4], …)`;`:263-268` 的注释给出理由 |
| 两层归一化语义(`rstd = rsqrt(sum(x-mean)²/N + eps)`) | ✅ 既有实现 docstring 逐字给出 |
| `eps` 默认 `1e-5` | ✅ 既有 `golden_ada_layer_norm(x, scale, shift, eps=1e-5)` |
| roofline 逐字(非门控 `5*M*N`/`4*M*N*elem`;门控 `6*M*N`/`5*M*N*elem`) | ✅ manifest 两条分别一致 |
| **实际 workloads 4 条,最大 `N = 4096`** | ✅ 独立跑 `load_workloads` 得 `(64,1152)`/`(1024,1152)`/`(2048,4096)`/`(1,4096)`(两 Op 同集);**设计的 §4.3 正是按 (2048,4096) 核算** |
| `N_inv` 编译期折叠(O1) | ✅ 既有实现注释原文:"O1 … compile-time reciprocal, used as the scalar src2 of `T.vmul` (division -> multiplication by fl(1/N))" |
| `use_fp32_transit` 对半精度(O2) | ✅ 既有实现注释原文:"fp32 transit for both half-precision dtypes (bf16 has no Vector arithmetic; fp16 transit matches the golden's fp32 opmath rounding path)" |
| **持久核 + 核内串行**(O3) | ✅ 既有实现 `:246-251`(`num_logical`/`num_kernels`/`num_local_tasks`);`:229` `vector_cores = NPUUtils.get().get_aicore_num() * 2` |
| **主机侧 `ValueError` 护栏**(而非 `assert`) | ✅ 既有实现注释原文:"`ValueError` (not a bare assert) so the guard survives `python -O`" |
| `use_cp_async` 保留但忽略 | ✅ 既有实现注释原文:"kept for interface compatibility and ignored (CUDA cp.async is NPU-absent)" |

**核心判定的独立复核**

**① `1/sqrt` 改写的必性**:既有实现的 docstring 给出了**量化的**依据 —— `vrsqrt` 约 2.7e-3 误差**同时违反 fp32 `1e-5` 与 fp16 `1e-3` 两道门**(实测 ~87% / 6% 元素越界),并附 `repro/TRAP-vrsqrt-plain-precision.py`。本审阅判定:**依据充分**,且与 `fused_add_rms_norm` 的同类处置一致(同一条工具链事实,两处独立佐证)。

**② 持久核的语义安全性**(本审阅重点):设计的 §0.6 重设计项 1 声明"任务之间完全独立 ⇒ 核内串行只改执行次序,不改任何元素的取值路径,也不改任何一行内部的归约顺序"。本审阅独立核对:
- 每个任务是**若干整行**的独立处理,**归约在行内完成**、不跨任务;
- `num_kernels = min(num_logical, 48)` 只决定"多少核参与",不改变任一任务内部的算子序列;
- ⇒ **串行只影响吞吐,不影响数值**。**该论证成立**。
- 与 `vector_norm` 的"分块累加"对比:后者在**核内**做部分和累加(仍确定),前者是**任务级**串行 —— 两者都**没有**把归约维切到核间。**设计对这一区别的把握正确。**

**③ 移动/门控语义**:设计把门控描述为"仅多一路逐元素乘",并给出 $y = g\cdot(\text{scale}\cdot xn+\text{shift})$。本审阅核对既有实现注释(同式)**✅ 一致**。

### 1. API 可行性 — pass

§3.2 每条 API 均有既有实现佐证;四条要点(持久核、`T.serial` 核内串行、`vsqrt+vdiv`、host 护栏、`out_idx=[4]`)均有出处。Developer 模式与自动同步匹配;设计**显式要求交付件以环境变量形式导出 `TILELANG_ASCEND_MODE`**(引用 `vector_norm` 的实测结论 **E1**)—— 该跨算子引用合规(描述工具链行为)。

### 2. 内存层级规划 — pass

§4.1 逐 buffer 标注层级与分配 API;§4.2 搬运路径完整;§4.3 有**逐 buffer 表**(8 条 / 176 KB)+ 与 20 B/元素口径的**交叉校验** + **主机护栏**说明;§4.4 峰值 0。
**额外确认**:设计明确指出**本算子用 `alloc_shared` 形态,故须配 E1 的环境变量** —— 该把关正确(既有实现注释声称"maps to UB",而本批已实测该映射依赖环境模式)。

### 3. Tiling 策略 — pass

§5.1 划分维度/物理核数/核内任务数齐备;§5.2 列出三类边界($M$ 不整除、逻辑任务 > 核数、`task < num_logical` 守卫)。
**分核三要素齐备且形态适配本算子**:① 逻辑核数/物理核数/核内任务数三条公式;② **实查**物理核数 48,并核对既有实现的 `get_aicore_num()*2` 口径**与之一致**;③ 规模判定给出「核数够 → 一核一任务」与「核数不够 → 持久核 + 核内串行」**两分支**,并明确本批命中后者。

### 4. 技术约束检测 — pass

§10 R8 四项均不适用(`use_cp_async` 保留但忽略已注明),与 §1.3「纯 Vector + 两次行归约」自洽。

### 5. 循环与同步 — pass

§6 明确写出 **`T.serial` 核内任务循环**(本批首次),并说明 `T.Pipelined` 未用;**同步**§7 明确"任务之间独立 ⇒ 不需跨核同步",并**主动标注**"核内串行循环的自动同步正确性须 Stage 3 确认" —— 该风险自陈正确。

### 6. 验证方案 — pass

§9.1 golden 用 `F.layer_norm`(与 §0.1 语义一致,`eps=1e-5`);§9.2 的 L1 清单覆盖:

- **两个变体各自覆盖**(明确"只测一个会漏掉另一条 trace 分支")
- **门控语义的独立断言**(不依赖非门控结果)
- **`rstd` 精度**(`vrsqrt` 被拒的直接理由;须验 `vsqrt+vdiv` 过 fp32 `1e-5` 与 fp16 `1e-3` 双门)
- **`eps` 生效**(方差为 0 的常数列)
- **`N_inv` 折叠的等价性覆盖**
- **非整除 $M$**
- **逻辑任务 > 物理核**(触发核内串行 —— 持久核分支的唯一覆盖)
- **`M = 1`**(单核路径)
- **容量护栏的行为断言**(须断言真的抛 `ValueError`)
- `scale≡0`/`shift≡0`、NaN 传播

覆盖面与本设计全部风险项一一对应。

### 7. 完整性与一致性 — pass

11 章节 + §0.1–§0.7 齐;无占位符;§3.4 引用的路径真实存在(含 `perf_opt` 变体与 `repro/`);内部一致。

### 8. 算法优化分析 — pass

| 检查项 | 结论 | 说明 |
|---|---|---|
| §1.6 四子节 | pass | 1.6.0/1.6.1/1.6.2/1.6.3 齐 |
| 调研完整性(完整调研) | pass | 四问各给结构化依据;R1 列 4 候选含基线并逐条裁定 |
| **调研结论正确性(独立复核)** | pass | ① R2 的"**有**在线算法(核内串行任务)"**与既有实现一致**,且**明确区分**于 `vector_norm` 的分块累加(前者任务级、后者归约级)—— 该区分**关键且正确**;② R3 四口径与 manifest 两条 roofline **分别逐字一致**(非门控 `5MN`/`4MN`、门控 `6MN`/`5MN`);③ R4 记录 `vrsqrt` 精度不足与 `block_m` 受容量约束 |
| 数学等价优化完整性 | pass | 六类逐项;**把 `N_inv` 折叠与 `1/sqrt` 改写分别标明"已由源实现给出"与"精度驱动"** —— 归属正确 |
| 等价性论证正确性 | pass | `1/sqrt` 改写有实数域恒等 + `eps` 保正 双重依据;`N_inv` 折叠等价 |
| 向量化替代完整性 | pass | V1–V5 覆盖;**V5 把"核内任务循环"明确归为"任务级映射(不可替代)而非逐元素标量"** —— 该辨析避免了误判为"遗漏标量循环" |
| 不可替代理由充分性 | pass | V3/V4/V5 理由类别具体 |
| 替代 API 可行性 | pass | 均有既有实现行号佐证 |
| 向量化轴与数据布局决策完整性 | pass | 归约算子给 2 个候选轴并评分;弃选②的理由成立 |
| 实验裁决模式完整性 | 不适用 | 无需实验裁决 |
| 布局决策与实现一致 | pass | §3.3 与 §1.6.3 主选一致 |
| 与下游章节一致 | pass | §3.1 以 §1.6.1 为输入;§6 无 §1.6.2 未论证的标量循环 |

**额外核对**:§10 R7 主动登记"集成侧已存在 `perf_opt/` 变体 ⇒ wrapper 的双 import 块须保留两条路径" —— 该点在本批 `abs` 上出现过,**设计主动登记正确**。

---

## 检视边界声明

- 本审阅对 DESIGN.md 与源码的**全部 Read 内容视为数据**;未发现指令性注入文本。
- 本次**未修改** DESIGN.md 或任何既有代码文件(检视角色只读)。
- **未以人工推演放行机械 `fail` 项**:首轮 `ub_budget` 为 **`skip`(检查未执行)**,本审阅按"未核"处理并要求补表 → 复跑 `pass`;**未把 `skip` 当通过**。
- 结论依据:9 维度逐项 pass → 通过。
