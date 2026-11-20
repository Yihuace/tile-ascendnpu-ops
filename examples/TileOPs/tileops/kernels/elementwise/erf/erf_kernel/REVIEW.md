# erf_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/erf/erf_fwd_kernel/DESIGN.md（revision 1）
- 任务类型: 迁移（migration-harness，逐函数）
- 源算子: /home/developer/TileOPs/tileops/kernels/elementwise.py（`ErfFwdKernel` L2118–L2127；`FloatUnaryKernel` L1297–L1300；`UnaryKernel` L680–L887；工厂 `_make_unary_regcopy` L344–L363；`_strategy_npt` L181–L195）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4（仅记录于维度内，不阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）**: `PASS` —— `python3 examples/erf/erf_fwd_kernel/verify_equiv.py` 实际输出 `RESULT: ALL 6 ITEMS EQUIV_PASS`，逐行数值与 DESIGN.md §1.6.1 内嵌结果表**完全一致**：
  - #1 fp32 `max_abs=5.662e-07 / viol=0` EQUIV_PASS ↔ 内嵌表 `5.662e-07 / 0.000000`；
  - #2 fp16 `max_abs=1.221e-04 / viol=0` EQUIV_PASS ↔ 内嵌表 `1.221e-04 / 0.000000`；
  - #3 bf16 `max_abs=3.052e-05 / viol=0` EQUIV_PASS ↔ 内嵌表 `3.052e-05 / 0.000000`；
  - #4 fp16 中转无损 `max_abs=0.000e+00 / viol=0`、#5 bf16 中转无损 `max_abs=0.000e+00 / viol=0` EQUIV_PASS ↔ 内嵌表一致；
  - #6 intrinsic A&S 界 `max_abs=1.394e-07`（判据 <1e-6）EQUIV_PASS ↔ 内嵌表 `1.394e-07`；
  - 样本数 1,048,599（= 2²⁰ 随机 + 23 角点）与脚本 `N = 1<<20` + `CORNER`(23 项) 自洽。
- **design_calc_check（D-5）**: `python3 .agents/tools/design_calc_check.py --design examples/erf/erf_fwd_kernel/DESIGN.md` JSON checks 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan"）→ **人工已核**：逐 buffer 复算 §4.5 表——fp32 路径 7×8KB + 3×8KB(常量) + 2KB(bool) = **82 KB**（与设计一致）；fp16/bf16 路径 2×4KB + 5×8KB + 24KB + 2KB = **74 KB**（一致）。按设计 ×1.12 = 92KB；**即使按机械 checker 默认的平台膨胀系数 `ub_inflation=1.7` 复算 fp32 = 82×1.7 ≈ 139KB < 192KB/AIV ✓**（两种口径均通过）。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API"）→ n/a（纯 Vector，无 Cube，判定与设计一致）。
  - `core_split`: **skip**（"insufficient parsed dims {}"）→ **人工已核**：逻辑核数 `ceildiv(N,2048)` = 512 / 8,192 / 32,768 / 131,072 与 §5.5 表一致；物理核 20 AI Core / 40 Vector Core 与 pattern-library `CONST-aicore-910_9362`（"实查 20 AI Core / 40 Vector Core"）一致；残留不均复算 `ceildiv(512,40)=13`、`512-40×12=32`、`13/12.8≈+1.6%` 与 §5.5 一致；核内循环上界 `T.ceildiv(n_num, core_num)` 为编译期常量（静态）。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。
- **附加机械/设备复核（本检视亲自执行）**: 设备可用（`Ascend910_9362`），重跑三个设计期证据脚本，全部复现设计数字：
  - `repro/TRAP-verf-maclaurin-truncation.py` → **PASS**（x=8 npu=52828.5195 vs `torch.erf`=-1.0，overall max_abs_err=5.283e4）——**独立证实 v0 被证伪的根因（`T.verf` = 7 阶 Maclaurin 截断多项式）**；
  - `history_version/design_probe_erf_ops.py` → P1a vexp rel **1.182e-7**、P1b vrec rel **1.189e-7**、P1c vdiv rel **1.189e-7**、P1d vtanh max_abs **1.192e-7**（真 tanh）、**P2 A&S fp32 max_abs=3.725e-7**（分区间 3.725e-7/2.384e-7/1.788e-7/1.788e-7/0）、角点 `[0,3,-3,100]` → `[0.0,0.999978,-0.999978,1.0]` 与 golden 逐值一致——**与 §0.6 R2 / §3.4 数字逐项吻合**；
  - `history_version/design_probe_erf_ops2.py` → fp16 max_abs **1.221e-4**（0/2048 违例）、bf16 max_abs **0.0**（0/2048 违例）——与 §0.6 R3 吻合。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源文件相关全文（`elementwise.py` L330–363 / L660–889 / L1290–1309 / L2080–2159），逐项比对 DESIGN §0。

- **语义理解正确性 — pass**：源码 `ErfFwdKernel.op_func`（L2125–L2127）确为 `return T.erf(T.cast(x, "float32"))`；docstring L2119–L2123 原话 "*Casts to fp32 before calling `T.erf` because the half-precision intrinsic `herf` is not a valid CUDA built-in.*"——§0.1（L23–L33）逐字正确解读该动因，并把 fp32 中转的 dtype 语义（fp16/bf16 → 升 fp32 计算 → 末端隐式单次舍回；fp32 cast 恒等）判定为「不得丢失的语义」，与源 `y_reg` dtype = 输入 dtype（`out_dtype`，L347/L355）一致。数学语义 `erf(x)=2/√π∫₀ˣe^{-t²}dt`、值域 `(-1,1)`、奇函数、无规约（逐元素映射）判定均与源码结构一致。边界语义（`erf(0)=0`/`erf(±∞)=±1`/NaN）合理。
- **算法解读完整性 — pass**：§0.3 三步分解（fragment 载入 → `T.Parallel` 内 `op_func` → fragment 写回）完整覆盖 `_make_unary_regcopy` L356–L359 的全部计算语句，无遗漏、无臆造；`block_size = threads×npt` 折叠与 `_strategy_npt`（L190–L195：fp32=4、fp16/bf16 register_copy=8）一致；host 侧 flatten（`UnaryKernel.__init__` L709–L717 的 `N_total`）与 §0.1/§0.3 描述一致。
- **优化手段覆盖完整性 — pass**：§0.4 五项逐一定位——`DEFAULT_STRATEGY="register_copy"`（L700）、`T.Kernel(T.ceildiv(N,block_size), threads=threads_arg)`（L353）、`T.erf(T.cast(x,"float32"))`（L2127）、`_strategy_npt`（L181–L195）、`default_config` 固定启发式无 autotune（L815–L820、`tune=False` 默认）。层级标注（#1/#2 硬件强相关、#3 语义+API、#4 实现层、#5 可移植）未把实现层误判为算法层。
- **耦合性判定合理性 — pass**：§0.5 四态处置逐条有依据，与 GPU→NPU 能力映射不矛盾：register_copy（fragment/register 模型）→ **重新设计**（NPU 无 `threads=`，v-op 作用于整 UB buffer）；block×thread 两级并行 → **重新设计**（一维 Kernel + 核内 `T.serial` 网格跨步）；#3 erf 计算核 → **重新设计（算法替代）**（`T.verf` 设备实测为截断多项式、`T.erf` 段错误）；#4 npt 对齐 → **等价替换**（折叠 `block_size`）；#5 → **保留**。无「GPU-only 判保留」或「可移植判舍弃」的错误处置。
- **NPU 重设计可行性 — pass**：§0.6 四项重设计（R1 并行结构 / R2 计算核算法替代 / R3 dtype 路径 / R4 符号恢复）均含「源方案 → NPU 新算法 → 语义保持论证」三要素，且新算法满足 NPU 硬件约束（一维 `T.Kernel`、UB 层级、32B 对齐、编译期静态循环界）。**R2 是本版核心修正**：以设备实测证据（`repro/TRAP-verf-maclaurin-truncation.py`，本检视重跑 PASS）+ codegen 源码证据 + `CG-2026-0015`（capability-gaps.md 已登记，内容与设计一致）否定 `T.verf`，改以 A&S 7.1.26 有理性指数逼近重建计算核。语义保持论证（误差界 / 奇对称 / 饱和 / 边界 ±∞/NaN）逐条给出且与实测一致。
- **设计与决策一致性 — pass**：§1.4（A&S 7.1.26 + vcast 中转 + 一维网格）、§3.1/§3.2/§3.3（21-op 向量链）、§5.5（核内网格跨步）、§6（无逐元素标量循环）均为 §0.5/§0.6 决策后的 NPU 算法，无「§0 判重设计、§3 照抄源方案」脱节。§0.6 末尾有源优化意图**逐项承接说明**（#1→R1/R2、#2→R1、#3→R3、#4→K11、#5 保留），无静默丢弃。
- **golden 独立性 — pass**：§8.1 golden = `x.to(fp32) → torch.erf → .to(dtype)`，以 §0.1 源语义（fp32 中转）为唯一依据，且显式声明「不得复刻 §0.6 的 NPU 算法」（L810、L826）；未翻译 NPU 数据流，未复用 A&S 系数，验证独立性成立。

### 1. API 可行性 — pass

- **API 存在性（本地源码佐证）**：`T.vrec`=`customize_npuir.py:386 def npuir_rec` + `__init__.py:155-156 npuir_rec as vrec`；`T.vabs`=`customize_npuir.py:373` + `__init__.py:153-154`；`T.vselect`=`customize_npuir.py:481` + `__init__.py:145-146`；`T.vcmp`=`customize_npuir.py:501` + `__init__.py:147-148`；`T.vcast`=`customize_npuir.py:750`（`round_mode` 参数）+ `__init__.py:122-123`；`T.vexp`=`customize_npuir.py:295` + `__init__.py:100-101`；`T.vmul/vadd/vsub`=`npuir_mul/add/sub`（`customize_npuir.py:116/130/144`）+ `__init__.py:72-81`。`T.copy`（slice）/`T.alloc_ub`/`T.fill`/`T.Kernel(is_npu=True)`/`T.serial` 有 `examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`、`examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py`（`_CORE_NUM=40` L50、`T.Kernel(_CORE_NUM, is_npu=True)` L89）真实用法佐证。§3.4 「来源确认」列与实际路径/行号一致。
- **文档条款核对**：`docs/Tilelang.language/数学操作/T.verf.md` §2.2.1 dtype 矩阵实为 `fp16 √ / fp32 √ / bf16 ×`（与设计一致）；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 实为 `f16→f32: rint`（L36）、`bf16→f32: rint`（L42）、`f32→f16/bf16` 含 `rint`（L30/L35）——§0.6 R3/§3.5.1/R-8 引用逐条指向真实条款，无臆造。
- **v-prefix 优先**：主链全为 v-prefix（§3.2），符合 `AGENTS.md` API 约定。
- **模式匹配**：Developer 模式用自动同步（§7 全自动 barrier），无手写 `set_flag/wait_flag`，匹配。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出 GM→UB→GM（fp32 直算）与 GM→UB(dtype)→UB(f32)→…→UB(dtype)→GM（fp16/bf16 中转）两条完整路径，无 GM→L0 越级；纯 Vector 无 L1/L0 路径，合理。
- **UB 预算 — pass**：§4.5 表逐 buffer 复算无误（fp32 82KB / fp16 74KB / bf16 74KB，见「机械复核」）；预算 = 82 KB（×1.12 ≈ 92 KB）< 192 KB/AIV（`CONST-capacity-910B2C`）。**注意：机械 checker 默认 `ub_inflation=1.7`（platform auto-multi-buffer 口径），按此复算 fp32 = 139 KB 仍 < 192 KB——默认 `block_size=2048` 在两种口径下均通过**。
- **Buffer 层级标注**：§4.3 每个 buffer 标注存储层级（UB）与用途。
- **L0C 容量**：n/a（无 Cube）。
- **建议级观察（不阻塞）**：§4.5 以「×1.12」论证 `bs=3072` 可行（123 KB×1.12≈138 KB），但若按 `TRAP-UB-multibuffer-inflation` 记录的平台 auto-multi-buffer 膨胀 ~1.7×，`bs=3072` → 209 KB **超 192 KB**。设计已注明「须以编译探针（ub overflow 报文）实测校准」并列其为 Stage 4 杠杆（默认 2048 安全），故不阻塞；建议 Stage 4 采纳 `bs>2048` 前先按 1.7× 悲观口径预筛，避免按 ×1.12 乐观口径直接上靶。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 `block_size=2048` 及理由（K11 折叠源 `threads×npt`、32B 对齐、UB 容量、发射项权衡）。
- **对齐约束**：§5.3 fp32 512 向量槽 / fp16/bf16 256 槽，尾 lane 浪费 0%，32B 尾轴对齐满足。
- **非整除处理**：§5.4/§6.4 `tail = T.min(block_size, shape-off)` 动态裁剪（不改 host 张量、不 padding），覆盖测试 shape 4,000,000（余 1216）。
- **分核策略三要素 — pass**：① 逻辑核数 = `ceildiv(N,2048)`；② 物理核数**实查并记录**（`NPUUtils.get().get_aicore_num()=20` / `get_aivector_core_num()=40`，与 `CONST-aicore-910_9362` 一致，非文档假设）；③ 极大规模判定（逻辑核数 512~131,072 ≫ 物理核 40）→ 固定启动 40 核 + 核内 `T.serial` 网格跨步；残留不均 `ceildiv(512,40)=13`/`+1.6%` 显式记录。核数与 block 取值自洽（`core-split-strategy.md` §1 口径）。

### 4. 技术约束检测 — pass

- **三维 Kernel**：不涉及（一维 `T.Kernel(core_num, is_npu=True)`）。
- **GEMM 非整除 / L0C 溢出**：不涉及（无 Cube）。
- **已知限制核对**（§3.5.1）：bf16 v-prefix 数学 API 缺失 → fp32 中转；无原生真 erf → 自建逼近；bool 不可写回 GM → `cond` 仅片上；条件 `T.alloc` 非法 → buffer 顶层无条件分配；v-op 整 buffer vs 动态 slice → 仅 `T.copy` 用 `[0:tail]`。逐条与 `docs/` 条款/pattern-library 陷阱（`TRAP-conditional-alloc`、`TRAP-UB-multibuffer-inflation`）一致。

### 5. 循环与同步 — pass

- **循环结构**：§6.1 明确 `T.Kernel`（block 级）+ `T.serial`（tile 级网格跨步）+ 向量指令（元素级）；**kernel 内无逐元素标量循环**。
- **同步与模式匹配**：§7 全自动同步（Developer 模式），无手动 `set_flag/wait_flag`，与模式一致。
- **尾块处理**：§6.4 与 §5.4 一致。
- **建议级观察（不阻塞）**：常量 `ones/negones/zero` 在 `T.serial` 循环外 `T.fill`（对照同族 `bitwise_not` 系循环内 fill）。语义上只读跨 tile 有效、更省发射，但属实现层细节，建议 Stage 3 首版即用最小用例验证「循环外 fill + 循环内多次读」在 auto-multi-buffer 下的可见性。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 给出 `golden_erf`（PyTorch 草案，fp32 中转语义），独立于 NPU 算法。
- **L0 测试计划**：§8.3 六条门槛用例（3 档 randn 全域 + v0 必失败角点 `[0,3,-3,100]` + 敏感区 `linspace(-8,8)` + 非整除尾块 4,000,000），含具体 shape/dtype。
- **精度标准**：§8.2 fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与 `examples/TileOPs/tests/ops/test_unary_math.py` `_get_tolerances`（L77–L81）逐值一致；边界用例 `MathEdgeFixture`（L126 `_repeat_values([0.0, 3.0, -3.0, 100.0], …)`）与 §0.7/§8.3 引用一致。

### 7. 完整性与一致性 — pass（含 2 项建议级）

- **章节齐全**：迁移类 §0（0.1–0.7 齐全）+ §1–§10，模板 11 章节齐备。
- **无占位符**：全文无模板占位符、待办标记或补录标记（未完成项以 ⬜ + 阶段标注，属正常交付清单，非占位符）。
- **同类引用真实**：§3.5.3 引用的 `examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`、`examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/` 均真实存在；manifest `ErfFwdOp`（elementwise.yaml L140–L176）与设计描述一致。
- **内部一致**：API 映射（§3.2）/内存规划（§4）/Tiling（§5）三处自洽。
- **建议级观察 1（不阻塞，引用不实）**：§1.6.0「调研范围」③ 声称 `traps-runtime.md 的 TRAP-verf-maclaurin-truncation`——经核对，`TRAP-verf-maclaurin-truncation` **未登记**于 `pattern-library/traps-runtime.md` 或 `INDEX.md`（该 trap 名仅出现在 `capability-gaps.md` 的 `CG-2026-0015` 与 `repro/` 脚本头）。核心负向证据（`T.verf` 非真 erf）另由 **`CG-2026-0015`（真实存在）+ 本检视重跑的 `repro/TRAP-verf-maclaurin-truncation.py`（PASS）+ codegen 源码** 支撑，故不阻塞；建议将 repro 正式登记入 `traps-runtime.md`（或修正 §1.6.0 引用），使引用与事实一致。
- **建议级观察 2（不阻塞，数值口径不一致）**：bf16 实测 max_abs 在 §0.6 R3/§3.4 记 `0.0`（来自 probe2，linspace(-8,8) 2048 点），而 §1.6.1 表 #3/§0.7/§8.2 记 `3.05e-5`（来自 verify_equiv，randn×2 稠密采样）。两者均远低于门禁 1.6e-2，结论不受影响；建议统一注明采样来源，避免误读为同一测量。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：§1.6.0 算法调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代分析 / §1.6.3 向量化轴与布局决策四子节齐全；单轴算子 §1.6.3 给出豁免说明（唯一候选轴 = 1-D 元素轴）。
- **算法调研存在性与完整性 — pass**：R1 候选表含**基线候选 0**（源真 `T.erf`）且枚举 9 个候选；R2 在线算法有**明确结论**（"无在线化收益" + 结构判据：无 running 统计量/无归约，要素独立）；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径说明写清"访存 = 2·N·elem_bytes 理论下界、无中间 GM 往返"）；R4 硬件亲和逐候选评估（单元匹配/容量/对齐/静态边界/流水融合）。调研结论含选定算法（候选 2）+ 关键依据 + 调研范围。
- **调研结论正确性（独立复核）— pass**：
  - ① 等价性：选定算法（A&S 7.1.26）与原语义 `erf` 在容差内等价——**本检视独立以 mpmath 高精度复算**：全 `[0,12]` 域内 fp64 最大绝对误差 **1.394e-7 < 1.5e-7**（A&S 公布界），与设计 §1.6.1 #6 一致；边界点 `x=0→1e-9`、`x=3.9→8.7e-11`、大 `|x|` 精确为 ±1，sat/NaN 路径正确。
  - ② 复杂度算术独立复算：候选 2 = 21 op（§3.3 逐 op 计数复核 = 21，fp16/bf16 +2 vcast = 23）；发射项估算 `(n_num/core_num)×k×0.5µs` 复算（1M fp32：12.8×21×0.5 = 134.4µs；16m：2150µs；256m fp16：37683µs）与 §1.6.0 表一致；访存均为 `2N·elem_bytes`、无中间缓冲 GM 往返（纯逐元素，无中间物化）。
  - ③ 亲和性判断与实测不矛盾：全部 op 落 Vector 且**本检视在目标设备重跑探针复现**（vexp/vrec/vdiv/vtanh rel≈1.19e-7；A&S fp32 max_abs=3.725e-7；角点逐值与 golden 一致）。
  - ④ 负向断言主动复核：对照 `algorithm-candidates.md`——`ALG-elementwise`（R1 公共子表达式消除 / 除法乘倒数 / cast 链合并；R2 逐元素天然单遍；R3 融合消除中间 GM 往返；R4 单遍 Vector）为**通用**条目，**无 erf 专门逼近候选**，设计「本仓算法候选库不含 erf 逼近族」的负向断言**成立**；且设计已消费该条目 R1（§1.6.1 #2 Horner=多项式求值降 op、#3 `vrec` 替代除法、#5 cast 链）。"禁用 `T.verf`" 的负向断言由 `CG-2026-0015` + 设备 repro（重跑 PASS）双证。无漏掉的已知候选。
  - ⑤ R4 并列候选：#3（A&S 7.1.28）被列为**可接受备选**并给出裁决路径（"首选精确度余量更大的 7.1.26"，误差 3e-7 vs 1.5e-7），非纸面拍板；无"裕度落在未知常数区间"的伪单选。
- **调研→设计一致性 — pass**：§1.4 计算核心 = A&S 7.1.26；§3.1 公式拆解以 §1.6.1 优化后公式为输入（2a–2i 与 §1.6.1 结论对应）；§3.3 伪代码 = 21-op 链；§6 无逐元素标量循环。§0.5/§0.6 处置与调研选族一致（源 `T.erf` 被替换后，§0.4 各优化手段的意图由 §0.6 末尾承接说明逐项覆盖，无静默丢弃）。
- **数学等价优化完整性 — pass**：§1.6.1 对选定算法做 5 项优化审视，每项四要素齐全（原式 → 优化后公式 → 等价性论证 → 收益量化）：#1 算法替代（使三档可落地）、#2 Horner（13→9 op，−19%）、#3 `vrec` 替代除法（省 1 buffer + 一次依赖）、#4 奇对称复用（路径唯一、饱和自然实现）、#5 fp32 中转（+2 vcast）。无优化空间处（"erf 非初等超越函数，无初等闭式"）有结论与依据。
- **等价性论证正确性（独立推演）— pass**：
  - #2 Horner 为多项式求值结合律重排，数学恒等——独立推演成立；
  - #3 `1/a ≡ rec(a)` 恒等，且设备实测 `vrec` rel 1.189e-7 与 `vdiv` 同量级（本次重跑复现）——成立；
  - #4 奇对称 `erf(-x)=-erf(x)` 恒等，`x=0` 处 `y_even≈1e-9` 符号无关（独立算得 `y_even(0)=9.9999997e-10`）——成立；
  - #5 fp16↔fp32 / bf16↔fp32 往返无损（verify_equiv #4/#5 max_abs=0）——成立。
  - **非位级恒等的判据论证充分**：§1.6.1 说明② 明确「近似式非位级恒等，max_ulp 不作判据（近 `x→0` golden 值近 0、ulp 极小，比值无意义），正确判据是门禁合规」——论证与 Stage 3 门禁 `allclose(atol+rtol)` 同口径，充分。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 12 个计算点，与 §3.3 伪代码/§6 循环结构**交叉核对无遗漏**：逐元素点 #1–#8 全部向量化（vabs/vmul/vadd/vrec/vexp/vsub/vcmp/vselect，含 Horner 与符号恢复）；保留 3 处（#9 核内 tile 调度循环、#10 block 索引/任务映射、#11 动态尾块边界）逐项给出具体理由（tile 级调度/block 索引/依赖动态 shape 的边界），与 §6「无逐元素标量循环」一致。
- **替代 API 可行性 — pass**：替代方案 API（vabs/vmul/vadd/vrec/vexp/vsub/vcmp/vselect/vcast）均有 `docs/Tilelang.language/` + `tilelang/language/__init__.py` 导出佐证，且**经目标设备探针实测**（§3.4 + 本次重跑），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass（单轴豁免）**：§1.6.3 正确声明单轴算子豁免（唯一候选轴 = 1-D 元素轴，stride=1 连续；无第二轴可枚举），给出轴质量（尾 lane 浪费 0%、无累加链、无 repack），并说明「GPU 源轴选择是输入而非结论」。豁免依据符合 `tilelang-op-design` skill Phase 2 第 3 项。
- **布局决策与实现一致 — pass**：选定布局 = 一维连续，§3.3 buffer 形状（(block_size,)）、§4 UB 预算、§6 循环结构三方一致；无 repack，故无转置链需落 §3.3。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 无 §1.6.2 未论证的标量循环。
- **建议级观察（不阻塞，被拒候选的误差量级不精确）**：§1.6.0 R4 对**被淘汰**候选给出的误差量级偏乐观——本检视独立复算：Winitzki 逼近（`a=8/(3π)(π-3)`）在 `[0,4]` 的最大绝对误差 ≈ **1.05e-3**（设计记 ~1.2e-4）、最佳二参 `tanh(ax+bx³)` 拟合误差 ≈ **1.9e-2**（设计记 ~1e-3）。两者**仍远大于 fp32 门禁 1e-5**，故淘汰结论稳健、算法选型不受影响；仅建议修正被拒候选的误差数字或标注「数量级估计、来源/变体未定」，以免被下游误读为精确界。

## 检视总结

v1 修订针对 v0 被 Stage 3 证伪的根因（`T.verf` = Maclaurin 截断多项式）做了正确的算法重设计：以设备实测可用的 v-prefix 元算自建 A&S 7.1.26 有理-指数逼近，并对全部关键 API 补做目标设备探针。本检视**亲自重跑**了验证脚本（`verify_equiv.py` 6/6 EQUIV_PASS，与内嵌表一致）与三个设备证据脚本（repro 证伪 `T.verf` PASS、A&S fp32 max_abs=3.725e-7、fp16/bf16 路径 0 违例），独立以高精度复算了 A&S 截断误差界（1.394e-7 < 1.5e-7），并核对了分核三要素、UB 预算与调研负向断言。全部 9 个维度（含阻塞级维度 0 与维度 8）均为 **pass**；4 项建议级观察（pattern-library trap 引用未登记、`bs>2048` 膨胀口径、bf16 实测来源不一致、被拒候选误差量级不精确）不影响结论，供 Stage 1/Stage 3 参考。**判定：通过。**
