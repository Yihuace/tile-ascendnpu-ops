# floor 设计文档检视报告(round 1)

## 检视结论

结论: 通过

## 检视元信息

- 文档:`examples/floor/floor_fwd_kernel/DESIGN.md`
- 任务类型:迁移(harness 子模式)
- 源算子:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`(`FloorFwdKernel` @2069)
- 检视维度:9 项
- 阻塞级问题数:**0** · 建议级问题数:0

---

## 机械复核

- **verify_equiv 重跑**:**不适用(无采纳优化项)**。§1.6.1 结论为「**公式层**无优化空间」+ 六类逐项依据,符合豁免条款。**豁免成立性判定**:C2 属 §0.6 的**重设计层 API 替代**(因本指令集无一元 `floor` 而被迫引入),设计已将其等价性归在 §0.6/§1.6.4 而**未**用 §1.6.1 的豁免承载 —— **归属正确**。
- **design_calc_check**(独立复跑):

| check | status | detail |
|---|---|---|
| `ub_budget` | **pass** | `UB: 10 buffer entries sum = 68.0 KB; capacity = 192.0 KB; x1.7 inflation budget = 113 KB` |
| `l0c_budget` | skip | 机械跳过,**人工已核**:纯 Vector、无 Cube,不适用 |
| `core_split` | skip | 机械跳过,**人工已核**:一维算子无 `M/bm`;工具报的 `declared logical cores = 1` 是**其默认值**。逐条复算 §5.3 ①:512 / 8192 / 32768 / 131072,与设计声明逐条一致 |
| `r3_metrics` | **pass** | `all four R3 metrics present` |

- **内部一致性复核(独立)**:无占位符;无 `§x.y.z` 标题交叉引用;§4.3 的 UB 表为「一行一 buffer、无容量列」。

---

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自 Read 源码逐条核对**:

| `§0` 断言 | 源码核对结果 |
|---|---|
| `FloorFwdKernel.op_func(x) = T.floor(T.cast(x, "float32"))` | ✅ `tileops/kernels/elementwise.py:2069-2078`(docstring 原文:"Casts to fp32 before calling `T.floor` because `hfloor` is not available for `cutlass::half_t` in CUDA.") |
| `FloorFwdOp` 继承 `_IntIdentityUnaryOp`(整型在 op 层短路) | ✅ `tileops/ops/elementwise/math_unary.py:145` = `class FloorFwdOp(_IntIdentityUnaryOp):` |
| 三策略(`direct`/`explicit_parallel`/`register_copy`,默认 regcopy)、`block_size = threads × npt` | ✅ 与同族算子共用工厂(`:304`/`:323`/`:343`),已在前序算子核对 |
| roofline 逐字 | ✅ manifest `flops: N` / `bytes: 2*N*elem_bytes` |
| **`floor` 的取整语义与边界** | ✅ 独立复算:`torch.floor([0.5,-0.5,1.0,-1.0,2.5,-2.5,0.0,-0.0])` = `[0,-1,1,-1,2,-3,0,-0]`;与 `ceil` 在 **8 例中 6 例不同** —— 设计的 §9.2 ties 用例须逐条重算,设计已如此处置 |

**四类输入覆盖的独立推演(V2 序列)**

| 输入类 | `m_ord` | `m_ge` | 取哪条 | 结果 | 与源算子 |
|---|---|---|---|---|---|
| `NaN` | false | — | `f_ub` | `NaN` | ✅ |
| `±inf` | true | true | `f_ub` | `±inf` | ✅ |
| $\lvert x\rvert \ge 2^{31}$ 有限值 | true | true | `f_ub` = x | $x=\lfloor x\rfloor$ | ✅ |
| 其余 | true | false | `t_ub` = C1 结果 | $\lfloor x\rfloor$ | ✅ |

**关于跨算子复用实测前提的合规性(本审阅独立判断)**

设计在 §1.6.4 直接引用了同批 `ceil` 的三条实测前提(int32 溢出不 fault / `vcmp` 对 NaN 非 IEEE / 选择子顺序)。本审阅判定**合规**,依据:

1. 三条前提描述的都是**工具链行为**(`vcmp`/`vcast` 语义、UB 溢出行为),**不是**某个算子的性质;
2. 两算子运行在**同一工具链**(wheel `0.1.2+npuir+7fc4a3c8` / CANN 9.0.0 / 同设备);
3. 设计给出了**来源路径**(`examples/ceil/ceil_fwd_kernel/` 下的 3 个探针/repro)、**前提清单**与**回退条件 + 复测入口**;
4. 前提本身在 `ceil` 任务中已被**实跑验证**(repro 四条断言成立)。

> ⇒ 满足"带三件套的跨算子前提复用"要求,不构成"未经论证的照抄"。

### 1. API 可行性 — pass

§3.2 每条 API 均有 `customize_npuir.py` 定义或本仓同族实现佐证;新增行(`vabs`/`vcmp`/`vselect`/常量缓冲)与 `ceil` 一致且已验证。Developer 模式与自动同步匹配。

### 2. 内存层级规划 — pass

§4.1 逐 buffer 标注层级与**分配 API**(`T.alloc_ub`)+ 层级↔API 固定对照;搬运路径 `GM → UB → GM` 无越级;§4.3 = **10 缓冲 / 68 KB ≤ 113 KB**;表形态合规。

### 3. Tiling 策略 — pass

§5.1 `block_size=2048` 与 K11 来源;§5.2 非整除处理(运行期切片)。**分核三要素齐备**:① 四档逻辑核数;② **实查**物理核数(`vector_core_num=48`/`cube_core_num=24`);③ 规模判定三选一命中「无需适配」并给理由。

### 4. 技术约束检测 — pass

§10 R8 四项均不适用,与 §1.3「纯 Vector」自洽。

### 5. 循环与同步 — pass

§6 明确四类结构,+ `T.Parallel` 仅用于常量缓冲初始化(与 `ceil` 实测形态一致);§7 自动同步;R11 记录缓冲复用待 Stage 3 复核。

### 6. 验证方案 — pass

§9.1 golden 独立于 NPU 算法(`torch.floor(x.float()).to(dtype)`);§9.2 **L0 + L1 阻塞层清单**覆盖 §0.1 全部"必须保持"的边界语义,其中 **`floor` 的 ties 已按 floor 语义逐条重算**(不与 `ceil` 照搬),并明确论证边界语义应落阻塞层。

### 7. 完整性与一致性 — pass

11 章节 + §0.1–§0.7 齐;无占位符;§3.4 引用的路径真实存在;内部一致。

### 8. 算法优化分析 — pass

| 检查项 | 结论 | 说明 |
|---|---|---|
| §1.6 四子节 | pass | 1.6.0/1.6.1/1.6.2/1.6.3 齐,另多出 1.6.4 |
| 调研完整性 | pass | 四问齐;R1 四候选含基线 C0;R2 给结构依据;R3 四口径与 roofline 逐字一致;R4 逐候选 |
| **调研结论正确性(独立复核)** | pass | 负向断言「npui 无一元 floor」**已独立核验成立**(读 `customize_npuir.py` 得 53 个 `npuir_*` 定义,`floor`/`ceil`/`round`/`trunc`/`fix`/`nearbyint` 全不存在;`valid_round_mode = {round,rint,floor,ceil,trunc,odd}`);R1 对 `round(x-0.5)` 不等价的结论正确(banker's rounding 在 ties 上不等价) |
| 数学等价优化完整性 | pass | 六类逐项;结论「公式层无优化空间」+ 依据;**并显式提示不得用该豁免掩盖 §0.6 的重设计层分歧** |
| 等价性论证正确性 | pass | C2 归 §0.6,由同工具链已实测前提 + 四类输入逐类推演承载 |
| 向量化替代完整性 | pass | V1–V4 与 §3.3/§6 三方一致 |
| 不可替代理由充分性 | pass | V2/V3/V4 理由类别具体到本算子 |
| 替代 API 可行性 | pass | 均有定义佐证 |
| 向量化轴与布局决策完整性 | pass | 单轴豁免 + 向量宽度对齐核算(2048 对 ×8/×4 整除、32B 对齐满足) |
| 实验裁决模式完整性 | pass | §1.6.4 以「前提 / 结论 / 实测来源 / 回退条件」四列记录复用内容,含复测入口 |
| 布局决策与实现一致 | pass | §3.3 buffer 形状与 §1.6.3 主选布局一致 |
| 与下游章节一致 | pass | §3.1 以 §1.6.1 为输入;§6 无 §1.6.2 未论证的标量循环 |

**额外核对项**:§10 R12 记录了「同族实测否证了设计期 roofline 结论(实际 10.7% 占峰而非带宽受限)」—— 设计**修正了结论**而非照抄同族的错误原始结论,处置正确。

---

## 检视边界声明

- 本审阅对 DESIGN.md 与源码的**全部 Read 内容视为数据**;未发现指令性注入文本。
- 本次**未修改** DESIGN.md 或任何既有代码文件(检视角色只读)。
- **未以人工推演放行机械 `fail` 项**(无 fail);两条 `skip` 已人工复核并在维度内注明。
- 结论依据:9 维度逐项 pass → 通过。
