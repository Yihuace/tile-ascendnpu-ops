# exp 设计文档检视报告(round 1)

## 检视结论

结论: 通过

## 检视元信息

- 文档:`examples/exp/exp_fwd_kernel/DESIGN.md`
- 任务类型:迁移(harness 子模式)
- 源算子:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`(`ExpFwdKernel` @1982)
- 检视维度:9 项
- 阻塞级问题数:**0** · 建议级问题数:0

---

## 机械复核

- **verify_equiv 重跑**:**不适用(无采纳优化项)**。§1.6.1 为「**公式层**无优化空间」+ 六类逐项依据,符合豁免条款。**豁免成立性判定**:本算子 §0.5 **未产生重设计层分歧**(npui 有 `npuir_exp`,G6 判为**等价替换**),故 §1.6.1 的豁免**不存在掩盖风险** —— 与 `ceil`/`floor` 的情形有本质区别。
- **design_calc_check**(独立复跑):

| check | status | detail |
|---|---|---|
| `ub_budget` | **pass** | `UB: 4 buffer entries sum = 32.0 KB; capacity = 192.0 KB; x1.7 inflation budget = 113 KB` |
| `l0c_budget` | skip | 机械跳过,**人工已核**:纯 Vector、无 Cube,不适用 |
| `core_split` | skip | 机械跳过,**人工已核**:一维算子无 `M/bm`;工具报的 `declared logical cores = 1` 是**其默认值**。逐条复算 §5.3 ①:512 / 8192 / 32768 / 131072,与设计声明逐条一致 |
| `r3_metrics` | **pass** | `all four R1/R3 metrics present` |

- **内部一致性复核(独立)**:无占位符;无 `§x.y.z` 标题交叉引用;§4.3 的 UB 表为「一行一 buffer、无容量列」。

---

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自 Read 源码逐条核对**:

| `§0` 断言 | 源码核对结果 |
|---|---|
| `ExpFwdKernel.op_func(x) = T.exp(T.cast(x, "float32"))` | ✅ `tileops/kernels/elementwise.py:1982-1987` |
| **`ExpFwdOp` 继承 `UnaryOp`(非 `_IntIdentityUnaryOp`),无整型短路** | ✅ `tileops/ops/elementwise/math_unary.py:31` = `class ExpFwdOp(UnaryOp):` —— 与 `AbsFwdOp`/`CeilFwdOp`/`FloorFwdOp` 的基类**不同**;非 float dtype 由基类校验拒绝(源侧亦有 `test_exp_rejects_non_float_dtype`)。**设计已正确识别该差异** |
| 三策略 / `block_size = threads × npt` | ✅ 与同族共用工厂(`:304`/`:323`/`:343`) |
| roofline 逐字 | ✅ manifest `flops: N` / `bytes: 2*N*elem_bytes` |
| **计算域 ≠ 输出域的两步落差** | ✅ 独立核对:源 `op_func` 在 fp32 域算、结果隐式落 `out_dtype`;故 fp16 下 `exp(20.0)` 溢出为 `+inf`,而 fp32 参照有限。设计已在 §0.1 写明并落 §9.2 的 L1 |

**核心判定的独立复核(本审阅重点)**

设计把 §0.5 的 **G6(fp32 域计算)判为「等价替换」而非「重新设计」**,理由是 **npui 有 `npuir_exp`**。本审阅独立核对:

| 核对项 | 结果 |
|---|---|
| `customize_npuir.py` 是否有 `npuir_exp` | ✅ **存在**(独立 grep 得 53 个 `npuir_*` 定义,含 `exp`/`exp2`/`ln`/`log2`/`sigmoid`) |
| 是否支持 fp32 | ✅ 支持 |
| 与源"升 fp32 再算"是否逐位对应 | ✅ `widen → vexp(fp32) → narrow` 与源 `T.exp(T.cast(x,"float32"))` 同域 |
| 是否引入新的值域/边界分歧 | ✅ **不引入** —— 与 `ceil`/`floor` 被迫 round-trip 的情形本质不同 |

⇒ **判定正确**。本审阅同时确认:该判定与 `ceil`/`floor` 的对比是**主动做出的**(设计显式写明"本算子与 `ceil`/`floor` 的关键差异"),而非照抄同族方案 —— 这一点**避免了给 `exp` 加上 5 条无用的补偿算子**。

**设计显式声明的未验证点(本审阅认可该处置)**

§0.6 与 §10 R1 声明:「**`vexp` 对 `NaN`/`±inf` 的指令级传播行为未在本项目实测过**」,并据此把 NaN/±inf/上溢/下溢/超范围**全部列入 L1 阻塞层**,同时预先给出「若实测不传播 → `[DESIGN_ERROR]` + 补 C2 式旁路」的处置路径。

本审阅判定:**该处置正确且必要**。依据:`ceil`/`floor` 的实测证明 `vcast` 对 NaN **不传播**(得 `0`),故"指令会正确传播特殊值"是**不能假设**的。设计把未验证点显式声明 + 落 L1 + 给回退路径,符合"不得假设指令行为"的要求。

### 1. API 可行性 — pass

§3.2 每条 API 均有 `customize_npuir.py` 定义或本仓同族实现佐证:`T.vexp(A,B)`(源在前)、`T.vcast`、`T.copy`、`T.min`、`T.alloc_ub`。Developer 模式与自动同步匹配。三条陷阱(`vexp` 源在前 / `alloc_ub` / round_mode 生效条件)均已注明依据。

### 2. 内存层级规划 — pass

§4.1 逐 buffer 标注层级与**分配 API**(`T.alloc_ub`)+ 层级↔API 固定对照;搬运路径 `GM → UB → GM` 无越级;§4.3 = **4 缓冲 / 32 KB ≤ 113 KB**;表形态合规。

### 3. Tiling 策略 — pass

§5.1 `block_size=2048` 与 K11 来源;§5.2 非整除处理(运行期切片)。**分核三要素齐备**:① 四档逻辑核数;② **实查**物理核数(`vector_core_num=48`/`cube_core_num=24`);③ 规模判定三选一命中「无需适配」并给理由。

### 4. 技术约束检测 — pass

§10 R6 四项均不适用,与 §1.3「纯 Vector」自洽。

### 5. 循环与同步 — pass

§6 四类结构均不使用并给理由(kernel 内无显式循环,块内仅单条 `T.vexp`);§7 自动同步,四个 buffer 独立无 aliasing。

### 6. 验证方案 — pass

§9.1 golden 独立于 NPU 算法(`torch.exp(x.float()).to(dtype)`);§9.2 **L0 + L1 阻塞层清单**覆盖 §0.1 全部"必须保持"的边界语义:**NaN / ±inf / 上溢 / 下溢 / fp16 超范围(x=20.0)/ 近零 / 非整除**,并给出"为何落阻塞层"的逐条理由。

### 7. 完整性与一致性 — pass

11 章节 + §0.1–§0.7 齐;无占位符;§3.4 引用的路径真实存在;内部一致。

### 8. 算法优化分析 — pass

| 检查项 | 结论 | 说明 |
|---|---|---|
| §1.6 四子节 | pass | 1.6.0/1.6.1/1.6.2/1.6.3 齐 |
| 调研完整性 | pass | 四问齐;R1 三候选含基线;R2 给结构依据;R3 四口径与 roofline 逐字一致;R4 逐候选 |
| **调研结论正确性(独立复核)** | pass | R1 的负向断言「无可更省等价形式」已独立核对:`exp2(x·log2 e)` 确实是 2 op(基线 1 op)且多一次乘法引入额外舍入;多项式逼近不满足 fp32 1e-5 容差。**迭代器 `exp2` 存在**这一点设计未误报(已在 R1 中作为候选①列出并否决)。R2 的"无在线算法"结构依据成立(逐元素、无 running 量)。 |
| 数学等价优化完整性 | pass | 六类逐项;「稳定化变形」一栏明确指出**本算子无规约故不需要 max-subtraction**(那是 softmax 一类的手段)—— 该辨析正确 |
| 等价性论证正确性 | pass | 无采纳项,无需等价论证;无重设计层分歧 |
| 向量化替代完整性 | pass | V1–V4 与 §3.3/§6 三方一致 |
| 不可替代理由充分性 | pass | V2/V3/V4 理由类别具体 |
| 替代 API 可行性 | pass | 均有定义佐证 |
| 向量化轴与布局决策完整性 | pass | 单轴豁免 + 向量宽度对齐核算(2048 对 ×8/×4 整除、32B 对齐满足) |
| 实验裁决模式完整性 | 不适用 | 本算子无需实验裁决(无重设计层分歧、无依赖未实测常数的取舍) |
| 布局决策与实现一致 | pass | §3.3 buffer 形状与 §1.6.3 主选布局一致 |
| 与下游章节一致 | pass | §3.1 以 §1.6.1 为输入;§6 无 §1.6.2 未论证的标量循环 |

**额外核对项**:§10 R7 记录了「同族实测否证了设计期 roofline 结论(实际 10.7% 占峰而非带宽受限)」—— 设计**修正了结论**而非照抄同族的错误原始结论,处置正确。

---

## 检视边界声明

- 本审阅对 DESIGN.md 与源码的**全部 Read 内容视为数据**;未发现指令性注入文本。
- 本次**未修改** DESIGN.md 或任何既有代码文件(检视角色只读)。
- **未以人工推演放行机械 `fail` 项**(无 fail);两条 `skip` 已人工复核并在维度内注明。
- 结论依据:9 维度逐项 pass → 通过。
