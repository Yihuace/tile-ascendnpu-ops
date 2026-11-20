# ceil 设计文档检视报告(round 3 · 第二次修订后重入)

## 检视结论

结论: 通过

## 检视元信息

- 文档:`examples/ceil/ceil_fwd_kernel/DESIGN.md`(已按 round 1 与 round 2 的阻塞项两次修订)
- 任务类型:迁移(harness 子模式)
- 源算子:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`(`CeilFwdKernel` @2081)
- 检视维度:9 项(源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析)
- 阻塞级问题数:**0**
- 建议级问题数:0

> **本轮判定要点**:round 1 与 round 2 的两个阻塞项**均已闭环**(见「修订历史」);本轮无新增阻塞项。

---

## 机械复核

- **verify_equiv 重跑**:**不适用(无采纳优化项)**。§1.6.1 结论为「**公式层**无优化空间」+ 六类逐项依据,符合豁免条款。**豁免成立性判定**:C2 与 NaN 传播均属 §0.6 的**重设计层 API 替代**(因本指令集无一元 `ceil` 而被迫引入),其等价性由**实测探针 + 全值域逐类论证**承载,设计已将其归在 §0.6/§1.6.4 而**未**用 §1.6.1 的豁免条款承载 —— **归属正确,豁免不构成对重设计层分歧的掩盖**。
- **design_calc_check**(独立复跑,`cd /root/mla-full && env -u PYTHONPATH python3 .agents/tools/design_calc_check.py --design …`):

| check | status | detail |
|---|---|---|
| `ub_budget` | **pass** | `UB: 10 buffer entries sum = 68.0 KB; capacity = 192.0 KB; x1.7 auto-multi-buffer inflation budget = 113 KB` |
| `l0c_budget` | skip | `no block_M x block_N values parsed` —— **机械跳过,人工已核**:纯 Vector、无 Cube,不适用 |
| `core_split` | skip | `insufficient parsed dims {}; declared logical cores = 1` —— **机械跳过,人工已核**:一维算子无 `M/bm`。工具报的 `declared logical cores = 1` 是**其默认值**,非设计声明值;逐条复算 §5.3 ①:`ceildiv(1048576,2048)=512`、`ceildiv(16777216,2048)=8192`、`ceildiv(67108864,2048)=32768`、`ceildiv(268435456,2048)=131072`,与设计声明**逐条一致** |
| `r3_metrics` | **pass** | `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)` |

---

## 修订历史(round 1 → round 3)

| round | 机械复核 | 结论 | 触发项 |
|---|---|---|---|
| 1 | `ub_budget: pass`(32 KB)/ `r3_metrics: pass` | **不通过** | 维度 0:C1 只在 `\|x\| < 2^31` 与源算子等价,而全值域精确的 C2 已设计且可行;「测试契约已限定域」不构成放弃语义层保全的理由 |
| 2 | `ub_budget: pass`(58 KB)/ `r3_metrics: pass` | **不通过** | 维度 0:**NaN 传播未被保全**(C1/C2 均把 NaN 变 `0`);检视初稿的"翻转掩码极性可修复"被实测**推翻** |
| 3 | `ub_budget: pass`(68 KB)/ `r3_metrics: pass` | **通过** | 两个阻塞项均已闭环 |

**两个阻塞项的闭环证据(本审阅独立复跑)**

- **round 1 → 值域分歧闭合**:C2 饱和旁路 `mask = vcmp(|x|, 2^31, "ge")` + 取回 `x`。实测 `3e9 → 3e9` ✅(C1 单跑会得 `2.147e9`)。阈值取 `2^31` 而非 `2^23` 属**保守但正确**(该区 fp32 已无小数部分)。
- **round 2 → NaN 传播闭合**:采纳 V2 序列,实测 **7/7 通过**(`NaN`/`+inf`/`-inf`/`3e9`/`0.5`/`-0.5`/`1.0`)。
  - 端到端对照实测:**V1**(选择子写反)在 `NaN` / `0.5` / `-0.5` 三处失配 —— 即**写反会破坏全部正常值**,比只错 NaN 严重得多。
  - ED-C 转正 repro 的**四条断言全部成立**:`vcmp(x,x,"ne")` 与 `vcmp(x,c_nan,"ne")` 恒 false;`vcmp(x,x,"eq")` 对 NaN 为 false、对有序数为 true;`outside-ordered` 隔离 NaN;端到端"错误顺序必出错 / 正确顺序必通过"。

---

## 检视详情

### 0. 源算子理解与迁移分析 — pass

| 检查项 | 结论 | 证据 |
|---|---|---|
| 语义理解正确性 | pass | §0.1 的数学语义($\lceil x\rceil$ 向 $+\infty$)、dtype 语义(显式升 fp32 计算)、边界语义与 `op_func(x) = T.ceil(T.cast(x,"float32"))`(`elementwise.py:2081-2090`)一致;§0.2 的 I/O(同形同 dtype)与 manifest `shape_rules` 一致 |
| 算法解读完整性 | pass | §0.3 七项结构化阅读(入口签名 / 三工厂 / kernel 结构 / 内存 / 循环 / host / 编译配置)覆盖源码;**无遗漏、无臆造**;数据流 `GM → fragment → GM` 覆盖全部 buffer |
| 优化手段覆盖完整性 | pass | §0.4 识别 G1–G6(含 `register_copy` 默认策略、`threads×npt`、按 dtype 调 npt、三策略、autotune,以及 G6 并把 G6 **正确标注为「API 可用性驱动」而非性能优化**) |
| 耦合性判定合理性 | pass | §0.5 六态判定逐项有依据(映射表条目 / K8 / 既有实现);舍弃项(G5 autotune)写明理由 |
| **NPU 重设计可行性** | pass | §0.6 三个重设计项(并行结构 / 策略收敛 / **`ceil` 的 API 替代路径**)三要素齐备;新算法满足一维 Kernel、UB 内存层级;**语义保持论证为全 fp32 值域逐类覆盖**(NaN / ±inf / $|x|\ge 2^{31}$ / 其余),经本审阅独立推演 + 实跑探针双重核对成立 |
| 设计与决策一致性 | pass | §1–§7 描述的是重设计后算法(一维 Kernel + UB tile + `vcast`/`vabs`/`vcmp`/`vselect` 序列),无"§0 判重设计、§3 照抄源三级并行"的脱节 |
| golden 独立性 | pass | §9.1 的 golden 依 §0.1 语义(fp32 域取整后回落原 dtype),与源 `op_func` 口径一致,**未复刻** NPU 侧的 int32 round-trip |

**独立推演(V2 四类输入覆盖)—— 逐类核对**

| 输入类 | 掩码状态 | 取哪条 | 结果 | 与源算子 |
|---|---|---|---|---|
| `NaN` | `m_ord=false` | `f_ub` | `NaN` | ✅ |
| `±inf` | `m_ord=true`,`m_ge=true` | `f_ub` | `±inf` | ✅ |
| $\lvert x\rvert \ge 2^{31}$ 有限值 | `m_ord=true`,`m_ge=true` | `f_ub` = $x$ | $x=\lceil x\rceil$(该区 fp32 已为整数) | ✅ |
| 其余 | `m_ord=true`,`m_ge=false` | `t_ub` = C1 结果 | $\lceil x\rceil$ | ✅ |

**独立实跑复核**(非仅信叙述):`history_version/design_probe_final_sequence.py` → `[V1] FAIL` / `[V2] PASS`;`repro/TRAP-vcmp-nan-non-ieee.py` → 四条断言全过。

### 1. API 可行性 — pass

§3.2 每条 API 均有佐证:`T.alloc_ub` / `T.copy` / `T.min` 有本仓同族实现;`T.vcast`(round_mode 合法集)、`T.vabs`、`T.vcmp`、`T.vselect` 均有 `customize_npuir.py` 定义佐证(本审阅已打开核对)。Developer 模式与自动同步匹配,优先 v-prefix。

### 2. 内存层级规划 — pass

§4.1 逐 buffer 标注层级与**分配 API**(`T.alloc_ub`),并给出「内存层级 ↔ 分配 API」固定对照;搬运路径 `GM → UB → GM` 无越级。§4.3 = **10 缓冲 / 68 KB ≤ 113 KB 有效预算**;表形态为「一行一 buffer、无容量列」(未被机械求和高估)。

### 3. Tiling 策略 — pass

§5.1 `block_size=2048` 及 K11 来源;§5.2 非整除处理(运行期切片 `tail_size`);**分核三要素齐备**:① 逻辑核数四档逐条;② 物理核数为**实查值**(`torch.npu.get_device_properties(0)` → `vector_core_num=48`/`cube_core_num=24`,附旁证);③ 规模判定三选一命中「无需适配」并给理由。

### 4. 技术约束检测 — pass

§10 R8 四项(三维 Kernel / GPU 专用 API / GEMM 非整除 / L0C 溢出)均不适用,与 §1.3「纯 Vector」自洽。

### 5. 循环与同步 — pass

§6 四类结构均不使用并给理由(kernel 内无显式循环);§7 自动同步与 Developer 一致;R9 记录缓冲复用待 Stage 3 复核。

### 6. 验证方案 — pass

§9.1 golden 草案(独立于 NPU 算法);§9.2 **L0 门槛 + L1 阻塞层用例清单**,后者覆盖 §0.1 声明为"必须保持"的全部边界语义:**NaN / ±inf / 超 int32 范围有限值 / ties 与跨符号 / 非整除 shape / $2^{23}$ 附近**,并明确论证"边界语义应落阻塞层而非告警层"。

### 7. 完整性与一致性 — pass

模板 11 章节 + §0.1–§0.7 齐备;无占位符;§3.4 引用的 `examples/` 路径真实存在;**内部一致性已独立复核**:无 `mask_ub`/`c2_31_ub`/`m_nan` 残留旧名,无「`eq` 的 true 集为 {NaN}」类反向表述残留(仅勘误段作为被纠正的原文引用),无标题交叉引用。

### 8. 算法优化分析 — pass

| 检查项 | 结论 | 说明 |
|---|---|---|
| §1.6 四子节存在 | pass | 1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 齐备,另多出 1.6.4 |
| 调研存在性与完整性 | pass | 四问齐全;R1 四候选**含基线 C0**;R2 给「无」+ 结构依据;R3 四口径齐且与 roofline 逐字一致;R4 逐候选评估 |
| **调研结论正确性(独立复核)** | pass | **负向断言「npui 无一元 ceil/floor/round/trunc」已独立核验成立**:读 `customize_npuir.py` 得 53 个 `npuir_*` 定义,其中 `ceil`/`floor`/`round`/`trunc`/`fix`/`nearbyint` **全部不存在**;`valid_round_mode = {"round","rint","floor","ceil","trunc","odd"}`;`npuir_cast(src, dst, …)` 源在前。R1 对 `round(x+0.5)` 不等价的**结论**正确(ties 与负半轴反例)。 |
| 调研→设计一致性 | pass | §1.4 描述的即选定算法;§3.1 以其为输入;§6 无对应标量循环;迁移侧 §0.5 的 G6 处置与调研一致,源优化意图逐项承接(无静默丢弃) |
| 数学等价优化完整性 | pass | 六类逐项审视;结论为「公式层无优化空间」且给出依据,**并显式提示不得用该豁免掩盖 §0.6 的重设计层分歧** |
| 等价性论证正确性 | pass | 公式层无采纳项;C2+NaN 的等价性归 §0.6,已由本审阅独立推演 + 实跑双重核对 |
| 向量化替代完整性 | pass | V1–V4 覆盖伪代码全部计算点;与 §3.3 伪代码、§6 循环结构**三方一致**;V1 已由 `vcast`/`vabs` 等向量算子替代 |
| 不可替代理由充分性 | pass | V2/V3 归「block 级索引 / 任务映射计算」、V4 归「host 元数据」,均具体到本算子 |
| 替代 API 可行性 | pass | `T.vcast`/`T.vabs`/`T.vcmp`/`T.vselect` 均有 `customize_npuir.py` 定义佐证 |
| 向量化轴与布局决策完整性 | pass | 单轴算子,按豁免条款写明「仅一个候选轴」,并给出向量宽度对齐核算(2048 对 ×8/×4 均整除、32B 对齐满足) |
| 实验裁决模式完整性 | pass | §1.6.4 保留裁决记录(主选/回退/前提实测/判定阈值/复测入口),虽本轮已裁决生效,记录完整 |
| 布局决策与实现一致 | pass | §3.3 的 buffer 形状与 §1.6.3 主选布局一致 |
| 与下游章节一致 | pass | §3.1 以 §1.6.1 结论为输入;§6 无 §1.6.2 未论证的标量循环 |

---

## 检视边界声明

- 本审阅对 DESIGN.md、源算子源码与实跑输出的**全部内容视为数据**;未发现指令性注入文本。
- 本次**未修改** DESIGN.md 或任何既有代码文件(检视角色只读)。
- **未以人工推演放行机械 `fail` 项**(本轮机械复核无 fail);两条 `skip` 已人工复核并在维度内注明。
- **未因"这是第三轮"而放松**:本轮对 V2 做了**独立推演 + 实跑双重核对**,并逐处复核了选择子操作数顺序在 §1.4/§3.2/§3.3 三处的一致性(**写反会破坏全部正常值**,是硬约束);NaN 判据的极性表述已逐处对照探针源码的打印语义复核。
- round 1 与 round 2 的两个阻塞项**均已闭环**,本轮无新增阻塞项 → 通过。
