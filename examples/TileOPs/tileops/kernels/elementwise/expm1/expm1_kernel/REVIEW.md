# expm1_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/expm1/expm1_fwd_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness）
- 源算子: /home/developer/TileOPs/tileops/kernels/elementwise.py（`Expm1FwdKernel` L2142–L2147）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py` → **ALL 6 ITEMS EQUIV_PASS**（viol 全 0）。执行输出与 §1.6.1 内嵌结果表**逐格一致**：#6 max_abs=1.245e-07、#1 fp32 max_abs=3.815e-06（max_ulp 5.0000）、#2 fp16=0.000e+00、#3 bf16=0.000e+00、#4 fp16 中转=0.000e+00、#5 bf16 中转=0.000e+00；小 x 相对精度诊断表（1e-7→naive 1.921e-01 / piece 5.000e-08 等 5 行）亦逐值一致。脚本实际执行成功，非脚本缺失/非不适用。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/expm1/expm1_fwd_kernel/DESIGN.md`）JSON checks 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan；advisory checker; verify manually"）→ **人工复核**：§4.5 fp32 路径 `x_ub/y_ub/xf/xa/g/e/thr` 7×8 KB + `cond` 2 KB = **58 KB** < 192 KB ✓；fp16/bf16 路径 4+4+8×5+2 = 50 KB ✓；×1.7 最坏膨胀 fp32 ≈ 99 KB < 192 KB ✓。算术自洽。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API"）→ 人工确认纯 Vector，无 L0C，不适用。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = 1"）→ **人工复核**：§5.5 三要素齐全，物理核数**实查** `NPUUtils.get().get_aicore_num()=20` / `get_aivector_core_num()=40`（纯 Vector ×2），逻辑核数 `ceildiv(N,2048)` = 512~131,072，取核内 `T.serial` 网格跨步 persistent 形态，残留不均 +1.6%(1M)/+0.24%(16M) 已显式记录。算术自洽。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**亲自 Read 源码核对（源真相 = `elementwise.py`）**：

- **语义理解（§0.1）**：源码 `Expm1FwdKernel.op_func(x)`（L2145–L2147）= `T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")`，与 §0.1/§0.7 引用的语句**逐字一致**。§0.1 准确指出 `T.exp` 输入恒为 fp32（强制中转）、`y_reg` dtype=输入 dtype、末端单次舍回；`FloatUnaryKernel.SUPPORTED_DTYPES = _FLOAT_DTYPES`（源码 L1297–L1300）实为 `(fp16, bf16, fp32)`（L154–L158），与 §0.1 dtype 语义一致（非 §0.2 表的三种 dtype 臆造）。§0.1 对「源实现 vs 契约差距」的判断（源 `exp−1` 为 `torch.expm1` 劣化直译、小 |x| 抵消）经源码与 manifest `ref_api: torch.expm1`（`elementwise.yaml` L179–L180）+ 测试 golden `torch.expm1`（`test_unary_math.py` L115）核对成立；manifest `shape_rules: output.shape == input.shape`（L190–L191）与 §0.2 输出 shape 声明一致。规约/边界语义（无规约；0→0/±∞/NaN/88.72 溢出）与 golden 行为一致。
- **算法解读完整性（§0.3）**：核对源码 `_make_unary_regcopy`（L343–L363）：`block_size = threads*num_per_thread`（L346）、`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg)`（L353）、`T.copy` 载入（L356）→ `T.Parallel(threads_arg, npt_arg)` 标量映射（L357–L358）→ `T.copy` 写回（L359）。§0.3 步骤 1–3 表与之一一对应，无遗漏、无臆造；host 侧 flatten（`N_total=numel`）与 §0.3/§0.2 一致。
- **优化手段覆盖完整性（§0.4）**：6 项（register_copy / block×thread 并行 / fp32 中转 / strategy-aware npt / 无 autotune / 单超越指令）均可在源码定位：`DEFAULT_STRATEGY="register_copy"`（L700）、`default_config` 的 `threads=256`+`_strategy_npt`（L815–L820）、`_strategy_npt`（L181–L195，fp32→4、register_copy fp16/bf16→8）与 §0.3「fp32 npt=4→1024 / fp16/bf16 npt=8→2048」**逐值一致**；source 无 autotune 默认调用（`UnaryKernel.autotune` 为可选路径，L845–L866）。层级标注（实现层/语义层/算法层）合理。
- **耦合性判定合理性（§0.5）**：逐条四态处置（保留/等价替换/重新设计/舍弃）+ 依据。`register_copy` 判「重新设计」（NPU 无 fragment/threads 模型）→ R1；block×thread 两级并行判「重新设计」→ R1；fp32 中转判「等价替换」→ R3；`exp−1` 计算核判「重新设计（算法替代）」→ R2。处置与 `ascend-constraints.md`（一维 Kernel、UB 层级、无 threads=）不矛盾，无「warp shuffle 判保留」式误判。
- **NPU 重设计可行性（§0.6）**：R1（thread 并行→一维 Kernel+网格跨步）/R2（分段 expm1）/R3（标量 cast→vcast 中转）/R4（阈值选路）均有「源方案→NPU 新算法→语义保持论证」三要素，且新算法满足 NPU 约束（一维 Kernel、UB 驻留、32B 对齐、容量；§5.5 分核）。R2 语义保持论证含误差界、阈值连续性、边界核对，经本 Agent **独立复算**（见维度 8）。性能意图逐项承接（§0.6 承接说明），未静默丢弃。
- **设计与决策一致性（§1–§7 ↔ §0.5/§0.6）**：§1.4 描述分段链（与 R2 一致）、§3.3 伪代码为 R1–R4 落地、§6 循环为核内 `T.serial`（与 R1 一致）、§7 自动同步（Developer）。无「§0 判重设计、§3 照抄源方案」脱节。
- **golden 独立性（§8.1）**：golden = `torch.expm1(x.to(fp32)).to(dtype)`，以 §0.1 契约语义（`torch.expm1`）为依据，**未复用** Taylor 系数/分段结构，不复刻 §0.6 NPU 算法，验证独立性成立。

### 1. API 可行性 — pass

§3.2/§3.3 全部 API 均在本地 docs/examples 有佐证并经目标设备探针实测（§3.4）：`T.vexp`（`docs/.../数学操作/T.vexp.md`，dtype 表仅 fp16/fp32，与 §1.6.0 bf16 结论一致；探针 P1 rel 1.182e-7）、`T.vsub`（`T.vsub.md` §2.3 标量 src1 + 原地 dst；探针 P2 max_abs 0.0）、`T.vabs`（`T.vabs.md` `hivm::VAbsOp`）、`T.vmul`/`T.vadd`（`T.vmul.md` §2.3 标量广播仅 src2，与设计 `T.vmul(xc, _C5, g)` 用法一致；dtype 仅 fp16/fp32）、`T.vcmp`+`T.vselect`（`T.vcmp.md` §2.3 输出 bool 不可写回 GM、`T.vselect.md`）、`T.vcast`（`T.vcast.md` §2.2.1：f16→f32 / bf16→f32 **仅 rint**，f32→f16/bf16 含 rint，与 R3/R9 一致）、`T.copy`/`T.alloc_ub`/`T.fill`/`T.Kernel(is_npu=True)`（examples 先例）。与 Developer 模式匹配（自动同步、无手动 set_flag）。v-prefix 优先原则遵循（全链 v 前缀）。无「凭记忆猜 API」。

### 2. 内存层级规划 — pass

§4.4 路径完整：GM→UB→GM（纯 Vector，无越级；无 L1/L0 需求）。§4.5 UB 预算逐 buffer 标注层级（UB）+ dtype：fp32 路径 58 KB、fp16/bf16 50 KB，×1.7（`TRAP-UB-multibuffer-inflation`）最坏 ≈99 KB < 192 KB/AIV ✓（机检 `ub_budget` skip，人工复算见「机械复核」）。buffer 复用（`xa`→小分支、`xc` 计算域别名/链尾原地）已量化记录，无悬空/重复 buffer。L0C 不适用。

### 3. Tiling 策略 — pass

§5.2 `block_size=2048` 含具体值与理由（K11 折叠 256×8；32B 对齐；UB 容量；发射项权衡），尾轴对齐满足（fp32 512 向量槽/fp16 256 槽，尾 lane 浪费 0%）。§5.4 非整除处理：kernel 内动态 `tail = T.min(block_size, shape-off)`（不改 host 张量），manifest workload 整除、测试 4,000,000 有余 1216 尾块，策略覆盖。§5.5 分核三要素齐全：① 逻辑核数 `ceildiv(N,2048)`；② 物理核数**实查**（`NPUUtils.get().get_aicore_num()=20`、`get_aivector_core_num()=40`，纯 Vector ×2，附代码与返回值）；③ 极大规模判定与网格跨步方案（`T.serial` 静态上界、`task=i*core_num+cid` 守卫、残留不均 +1.6%/+0.24% 显式记录），核数与 block 取值自洽（`core-split-strategy.md` §1 三要素达标；机检 `core_split` skip → 人工复核通过）。**非阻塞观察**：§5.2 声明「统一取 2048」，而集成 wrapper `_DEFAULT_BLOCK_SIZE`（`expm1.py` L71–L74）fp32 默认仍为 1024——kernel 以参数化 `block_size` 支持二者，Stage 3 集成时按 wrapper 传值即可，不影响设计成立。

### 4. 技术约束检测 — pass

一维 `T.Kernel(core_num, is_npu=True)`，无三维 Kernel。非 GEMM（无 padding/crop 需求，尾块由 `tail` 处理）。无 Cube → 无 L0C 溢出。§3.5.1 逐项核对本项目已知限制（bf16 v-prefix 数学 API 缺失→fp32 中转；无原生 `vexpm1`→自建；条件 `T.alloc` 非法→顶层无条件分配；bool 不可写回 GM→片上配 vselect；标量操作数在 `T.Kernel` 内 `T.Scope` 外）。§9.4 已登记能力缺口（无原生 `vexpm1`，Frontend API 层），符合 Capability Gap Rule。

### 5. 循环与同步 — pass

§6.1 明确 `T.Kernel`（核级 block 并行）+ `T.serial`（tile 级网格跨步）+ 元素级向量化（无 `T.Parallel` 标量循环）。§6.3 说明不用 `T.Pipelined` 的理由（单 tile 线性依赖、跨 tile 由 auto-multi-buffer 处理）。§6.4 尾块逻辑与 §5.4 一致。§7 同步为 Developer 自动同步（无跨引擎/跨核协作，无手动 set_flag/wait_flag），与模式选型一致。

### 6. 验证方案 — pass

§8.1 含 PyTorch golden 草案（`torch.expm1`，保持 fp32 中转，独立于 NPU 算法）。§8.2 按 dtype 给 atol/rtol：fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与 `test_unary_math.py` `_get_tolerances`（L79–L84）**逐值一致**。§8.3 L0 计划 7 例含具体 shape/dtype/门禁：randn 全域 3 dtype、角点 `repeat([0.0,88.8,-88.8,1e-7])`（与 `test_expm1_edge` L149–L153 输入一致）、小 x 敏感段相对误差专项（L0-5）、阈值边界（L0-6）、非整除尾块（L0-7）。误差预算核对与实测一致。

### 7. 完整性与一致性 — pass

模板 11 章节（§0–§10）齐全，迁移类 §0 含 0.1–0.7 全小节。全文无模板占位符、无待办标记、无未填内容。§3.5.3 引用的 examples 路径经 Glob 全部真实存在：`examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`、`example_elementwise_exp2.py`、`examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py`、`examples/erf/erf_fwd_kernel/DESIGN.md`、`examples/TileOPs/tileops/kernels/elementwise/expm1/`（`_expm1_fwd_kernels.py`+`expm1.py`）。API 映射（§3.2）、内存规划（§4）、Tiling（§5）三处相互一致，无矛盾。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：含 1.6.0/1.6.1/1.6.2/1.6.3 四子节，单轴算子 1.6.3 有豁免说明，四问逐问有结论。

**算法调研完整性**：R1 候选表**含基线候选**（候选 0 = 源 `exp−1`）并纳入 R3 对比；R2「无在线化收益」有**结构判据**（无 running 统计量、逐元素天然单遍）且与 `algorithm-candidates.md` `ALG-elementwise` R2「逐元素映射天然单遍」一致；R3 四口径（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）齐全、基线在表、口径写清（机检 `r3_metrics` pass）；R4 逐候选六维亲和评估。调研范围（①–⑤）明确。

**调研结论正确性（独立复核）**：
- **R1 候选覆盖**：对照 `algorithm-candidates.md` `ALG-elementwise`（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」）——CSE（§1.6.1 #5）、cast 链合并（§1.6.1 #4）已覆盖；「除法乘倒数」对本算子不适用（公式无除法）。参考表**不含 expm1 专门逼近条目**（已核实），逼近族（Cephes/Taylor/libm 区间约减）由外部知识补充并标注为 algorithm-research.md §6 第 5 层「只取算法思路」，API 存在性另由本地 docs + 设备探针佐证——合规。
- **负向断言逐条复核**：①「无原生 `vexpm1`」——`grep expm1 docs/Tilelang.language/` 零命中、`docs/.../数学操作/` 枚举无 `T.vexpm1`、`tilelang/language/customize_npuir.py` 无 `npuir_expm1`（仅 L295 `npuir_exp`），**成立**；②「无 `vsinh`」——`数学操作/` 目录无 `T.vsinh.md`，**成立**；③「`T.vexp2` 存在」——`docs/.../数学操作/T.vexp2.md` 存在，候选 7 引用属实。无「与参考表/源码证据矛盾」的漏候选。
- **复杂度复算**：候选 2 = 16 op（abs1+cmp1+Horner10+mul1+exp1+sub1+select1，fp32）/18（+2 vcast）——逐 op 点数**自洽**；全部候选访存同为理论下界 `2·N·elem_bytes`（输入读+输出写，**无中间 GM 往返**，无漏计）；扫描遍数 1；中间缓冲为 UB 单 tile 驻留（非 O(N)）。算术正确。
- **调研→设计一致性**：§1.6.0 选定候选 2 → §1.4 描述分段链、§3.1 公式以优化后分段式为输入、§6 无逐元素标量循环；§0.5/§0.6 处置与选族一致，源优化意图（§0.4）逐项承接说明齐备，无静默丢弃。

**数学等价优化完整性（§1.6.1）**：5 项采纳项各有「原式→优化后公式→等价性论证→收益量化」四要素。**独立推演**：#1 小 x Taylor 截断界复算——`|R| ≤ |x|⁷/5040·1/(1-|x|/8)`，在 |x|=T 处 = 1.1917e-7×1.0453 ≈ **1.245e-7**，与 verify_equiv #6 实测 max_abs **1.245e-07 完全吻合**；相对误差 `≤ T⁶/5040·e^T = 3.438e-7×1.4142 ≈ 4.86e-7` ✓（设计 4.86e-7）。#2 Horner 恒等（多项式结合律重排）成立，15→11 op ✓。#3 vselect 选路（条件函数逐元素选取）恒等 ✓。#4 fp32 中转（f16/bf16↔f32 往返无损）✓。#5 CSE/原地复用不改变数值 ✓。**注**：§0.6 R2 大 x 分支分别列出减 1 舍入 ≤2.0e-7 与 vexp 放大误差 ≤4.0e-7，其保守和约 6.0e-7，而小结写「综合 ≲5e-7」——量级陈述略保守偏差，但两分支均 ≪ rtol 1e-5（≥16× 余量），不影响结论正确性。

**等价性机器验证**：verify_equiv.py 重跑 6/6 EQUIV_PASS，与 §1.6.1 内嵌表一致（判据 = Stage 3 门禁同口径；分段式非位级恒等，故以门禁违规率 0 为判据、max_ulp 不作判据，理由充分）。

**向量化替代完整性**：§1.6.2 覆盖 12 个计算点，与 §3.3 伪代码、§6 循环结构交叉核对——kernel 内**无逐元素标量循环**；#1–#8、#12 全部向量化（API 有 docs 佐证 + 设备探针实测），#9–#11（核内调度循环/block 索引/动态边界）保留理由具体到本算子（tile 级调度、每 tile 少量标量、动态 shape 守卫），非「实现简单」式泛谈。

**替代 API 可行性**：全部替代 API 在 docs/探针有佐证（见维度 1），v-prefix 优先。

**向量化轴与布局决策**：§1.6.3 以「1-D `[N]` 单轴算子，唯一候选向量化轴即 1D 元素轴」豁免，并给出轴质量量化（尾 lane 浪费 0%、无累加链、无 repack），迁移注对照源码隐式轴选择（GPU thread×npt 为输入非结论）。弃选候选「无」符合单轴豁免（skill Phase 2 第 3 项）。无「把 I/O 契约当内部布局依据」问题。

**布局决策与实现一致**：§1.6.3 选定的一维连续布局与 §3.3 伪代码、§4 内存规划（buffer 形状 `(block_size,)`、UB 预算）、§6 循环结构三方一致。

**与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致。
