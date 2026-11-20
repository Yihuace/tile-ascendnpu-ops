# isfinite_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/isfinite/isfinite_fwd_kernel/DESIGN.md
- 任务类型: 迁移（harness）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/comparison.py（`IsfiniteFwdKernel`），复用 `_base.py` / `_builders.py` / `_op_body.py` / `ops/elementwise/comparison.py`
- 检视维度: 9 项（0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 1

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/isfinite/isfinite_fwd_kernel/verify_equiv.py` → 执行成功（EXIT=0）。采纳项 1（`isfinite(x)` → `(x>-inf)∧(x<+inf)`，fp16/bf16/fp32）mismatches 全 0 / EQUIV_PASS；采纳项 2（`isfinite(cast(x,f32))` → 原生 dtype 双侧有序式，fp16/bf16/fp32）mismatches 全 0 / EQUIV_PASS；判别段 A（`|x|<+inf`）三 dtype 0（DISCRIM_OK）；判别段 B（弃选平方式）fp16 3 / bf16 3 / fp32 5（>0，DISCRIM_OK）。**执行输出与 §1.6.1 内嵌结果表逐行一致，全部 EQUIV_PASS + 判别成立。**
- **design_calc_check**: 注：工具实际仅接受 `--design <path>`（`design_calc_check.py:280`），不接受任务书写的 `--dir`（`--dir` 报 error 退出 2）；故按 skill Phase 1 第 7 步以 `--design examples/isfinite/isfinite_fwd_kernel/DESIGN.md` 执行并通过（EXIT=0）。JSON checks 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan"）→ 人工复核已核：§4.5 显式逐 buffer 表，fp16/bf16 峰值 52KB、fp32 峰值 76KB，均 < 192KB（`CONST-capacity-910B2C`），`×1.7` 悲观膨胀 129KB 亦安全。**人工核算与设计一致。**
  - `l0c_budget`: **skip**（"no affirmative Cube-path API"）→ 本算子纯 Vector，无 Cube，n/a 成立（人工已核：§5.1 无 `T.gemm`/`load_nd2nz`/`store_fixpipe`）。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = 40"）→ 人工复核已核：§5.5 三要素齐全（实查物理核 20、纯 Vector ×2=40；逻辑核 4096/65536；核内串行 `ceil(num_logical/40)` = 103/1639），算术独立复算一致。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）→ §1.6.0 R3 四口径表齐全。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

亲自 Read 源算子全文核对（`comparison.py`、`_base.py`、`_builders.py`、`_op_body.py`、`ops/elementwise/comparison.py`、`ops/elementwise/_base.py`）：

- **语义理解正确性（§0.1/§0.2）**：源码 `IsfiniteFwdKernel.op_func` = `T.isfinite(T.cast(x, "float32"))`（`comparison.py:202-207`），与 §0.1 引用完全一致；`IsfiniteFwdKernel(FloatPredicateKernel, UnaryPredicateFwdInterface)`（`comparison.py:202`），`FloatPredicateKernel.SUPPORTED_DTYPES = FLOAT_DTYPES = (float16, bfloat16, float32)`、`OUTPUT_DTYPE = torch.bool`、`DEFAULT_STRATEGY = "register_copy"`（`_base.py:503,509,510`）——§0.1「输入三 float dtype / 输出 bool」结论与源码一致。§0.2 I/O（`input [*S]` → 展平 1-D、`output` 同 shape bool）与 `UnaryOp._call_spec`（`ops/elementwise/_base.py:74-81`）+ `UnaryKernel.forward`（`_base.py:253-256`：`flat(x)`→kernel→`_restore_output_dtype(...).reshape(x.shape)`）一致，**非转置、无跨元素依赖**判断正确。
- **算法解读完整性（§0.3）**：§0.3 步骤 1–7 覆盖源码全部计算语句——host 展平、分块装载、`T.cast` 宽化、`T.isfinite`、`if_then_else` bool 落地、分块写回、host 回塑，与 `make_unary_regcopy`（`_builders.py:85-121`）+ `store_unary_bool_as_int8`（`_op_body.py:50-72`，`BOOL_STORAGE_DTYPE="int8"` 见 `_dtype.py:9`）+ `UnaryKernel._get_effective_op_func`（`_base.py:207-212`）逐项对应，无遗漏、无臆造。数据流图覆盖 x→x_reg→x_f32→pred→y_int8。
- **优化手段覆盖（§0.4）**：识别 register_copy fragment（#1）、coalesced（#2）、`threads×npt`（#3）、fp32 宽化谓词（#4）、bool int8 字节存储（#5），与源码结构一致；未误把实现层优化当算法层（明确标注硬件强相关）。源未用 swizzle/mma/warp shuffle，§0.4 注记「识别不出来 ≠ 不存在」并给出逐项核对依据，正确。
- **耦合性判定（§0.5）**：逐条四态处置（语义/算法保留；bool 字节存储、fragment→UB、coalesced→MTE 等价替换；`T.isfinite`+宽化、`threads×npt` 重新设计；swizzle/mma/shuffle 舍弃不适用），依据指向 `T.vcmp.md` §2.3（bool i1 不可 GM 存取）、CG-2026-0019、ascend-constraints。抽检：`T.isfinite` 无向量形式、bool i1 不可 GM 存取的处置与文档/`_op_body.py` 事实一致。
- **NPU 重设计可行性（§0.6）**：R1 一维 Kernel + block_size 折叠（满足「仅一维 Kernel」）；R2 `(x>-inf)∧(x<+inf)` 免 vabs 双侧有序式（满足全 dtype 原生统一）；R3 双侧掩码→f16 `vmul`→int8 字节链（满足 bool 不可 GM 存取）。三项均有「源方案 → NPU 新算法 → 语义保持论证」三要素，边界语义（±inf/NaN→False、±0/次正规/极值→True）逐项论证。
- **设计与决策一致性（§11.1/§11.2）**：§1–§10 描述的是重设计后算法，无「§0 判重设计、§3 照抄源方案」脱节；源优化手段 #1–#5 意图逐项承接（#4 升级为等价化简，非静默丢弃）。
- **golden 独立性（§8.1）**：golden = `torch.isfinite(x.detach().cpu().float())`（§8.1），与源仓 `test_isfinite.py` 参考一致，**不复刻** NPU 的 `vcmp`/`vmul`/`vcast`/字节存储/persistent 结构，独立性成立。

> 观察（建议级，不阻塞）：§0.3 第 72 行将源 `make_unary_regcopy` 一律描述为「fragment 全块路径」，但源码对 bool 输出且 `N % block_size == 0` 时走 `contiguous` 局部运行路径（`_builders.py:96,101-110`）；此为源实现路径细化，算法层语义（分块装载→逐元素映射→整块写回）不变，不影响迁移正确性。

### 1. API 可行性 — pass

§3.2/§3.4 每条 API 均有佐证，且已亲自打开文档核对：

- `T.vcmp`：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.1（`eq/ne/lt/gt/ge/le`）、§2.2.1（fp16/fp32/bf16 均 √）、§2.3（输出 bool、不可写回 GM）——与 §3.2/§3.4 一致；§2.1 无 `unordered` 模态，佐证 CG-2026-0024。
- `T.vcast`：`.../数据类型转换操作/T.vcast.md` §2.2.1 含 `i1→f16（rint）`、`f16→i8（rint）`，**无 `i1→i8` 直连**——与设计「经 fp16 中转」一致。
- `T.vmul`：`.../数学操作/T.vmul.md` §2.2.1（fp16 √）、§2.3（dst 可与 src1/src2 同 buffer）——佐证 fp16 域原地合并。
- `T.vabs` 弃用：`.../数学操作/T.vabs.md` §2.2.1 **bf16 ×**（fp16 √/fp32 √）——佐证免 vabs 双侧式动机。
- `T.vand`/`T.vnot`/`T.vselect` 弃用：`.../逻辑操作/T.vand.md` §2.2.1 bool ×、`.../逻辑操作/T.vnot.md` §2.2.1 bool/int1 ×、`.../条件操作/T.vselect.md` §2.2.1 bool 操作数 ×——与设计「bool 域合并不走这些 API」一致。
- `T.infinity` fp16/fp32：`.../创建操作/T.infinity.md` §2.2.1（fp16 √ / fp32 √ / **bf16 ×**）。
- `T.Kernel(一维, is_npu=True)` / `T.alloc_ub` / 双显式 slice `T.copy`：同族先例 `examples/hardsigmoid/hardsigmoid_fwd_kernel/hardsigmoid_fwd_kernel.py`（实存）。
- v-prefix 优先：设计主体用 `T.vcmp`/`T.vcast`/`T.vmul`，`T.fill` 有 `T.vbrc` 等价替代说明，符合基线。Developer 模式自动同步，无手动 set_flag/wait_flag。**pass**。

### 2. 内存层级规划 — pass

- 路径完整（§4.4）：纯 Vector，仅 GM↔UB 两级，无越级 GM→L0 直连。
- UB 预算（§4.5）：显式逐 buffer 表，fp16/bf16 峰值 53248B（52KB）、fp32 峰值 77824B（76KB），均 < 192KB（`CONST-capacity-910B2C`，已核对 constants.md 条目）。独立复算：fp16 路径 `5×8192 + 3×4096 = 53248`；fp32 路径 `3×16384 + 2×8192 + 3×4096 = 77824`，与设计一致。
- 存储层级标注：§4.3 每 buffer 标注 UB。
- 机械 `ub_budget` skip → 人工已核（见机械复核段）。
- 无 Cube，L0C 不适用。**pass**。

### 3. Tiling 策略 — pass

- Block 划分明确（§5.2）：`block_size=4096`，理由为容纳两个 ±inf 常量 tile 下 UB 安全（若 8192，fp32 峰值 ~156KB，×1.7 → 265KB 超限）。
- 对齐约束：4096 为 32B 整数倍且整除向量宽（512×8 / 1024×4），§5.3 论证成立。
- 非整除处理（§6.4/§5.4）：`tail_size = T.min(block_size, remaining)` + `block_id < num_logical` guard。
- 分核策略三要素（§5.5）：① 逻辑核数 `ceil(N/block_size)`；② 物理核数**实查** `NPUUtils.get().get_aicore_num()` → 20（纯 Vector ×2 = 40），记录查询代码与返回值，且与 `CONST-aicore-910B2C`（24）差异注明「以实查为准」；③ 规模判定与方案（小规模 `num_kernels=num_logical`；极大规模固定 40 核核内串行）。独立复算：16M→`ceil(4096/40)=103`；256M→`ceil(65536/40)=1639`，与 §5.5 表一致。**pass**。

### 4. 技术约束检测 — pass

- 三维 Kernel：源即一维，NPU 一维 `T.Kernel(num_kernels, is_npu=True)`，无三维。
- GEMM 非整除 / L0C 溢出：无 Cube，不适用。
- §3.5.1 逐条列出已知限制（`T.infinity` bf16、`T.vabs` bf16、条件性 `T.alloc_ub` CG-2026-0020、bool i1 GM）并给出处理方案，覆盖完整。**pass**。

### 5. 循环与同步 — pass

- 循环结构（§6.1）：block 级核并行 `T.Kernel`、tile 级 `T.serial(num_local_tasks)`（静态边界）、元素级 v-prefix 整块向量化，选择明确；§6 无逐元素标量循环，与 §1.6.2 结论一致。
- 同步（§7）：Developer 自动同步，无手动 flag，与模式匹配。
- 尾块（§6.4）：guard + `T.min` 尾块逻辑，垃圾尾部不写回。**pass**。

### 6. 验证方案 — pass

- Golden（§8.1）：`torch.isfinite(x.detach().cpu().float())` PyTorch 参考草案，含独立性说明。
- 精度标准（§8.2）：三 float dtype `atol=rtol=0`（谓词精确判等），与源仓 `_get_tolerances` 一致。
- L0 计划（§8.3）：`N=1048576`、三 dtype、注入 NaN/±inf/±0/次正规/极值、精确相等比对，另含拒绝非 float dtype 负向用例与关键判别探针。**pass**。

### 7. 完整性与一致性 — pass

- 章节齐全：迁移类含 §0（0.1–0.7 完整）+ §1–§11，模板章节齐全。
- 无占位符：全文无 `{placeholder}`/`TODO`/`待补充`。
- 同类引用真实（Glob 实存）：`examples/isinf/isinf_fwd_kernel/DESIGN.md`、`examples/isnan/isnan_fwd_kernel/DESIGN.md`、`examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py`、`examples/hardsigmoid/hardsigmoid_fwd_kernel/DESIGN.md`、`examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md`、`examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/elementwise/isfinite/_isfinite_fwd_kernels.py` 均存在。
- 内部一致：§3.1/§3.2/§4.4/§4.5 对 buffer 与工具链、dtype 描述相互一致。**pass**。

### 8. 算法优化分析 — pass

§1.6 四子节齐全（1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 布局决策）。

- **§1.6.0 调研存在性与完整性**：四问逐问有结论——R1 候选表含**基线候选 0**（源 `T.isfinite(T.cast(x,f32))`）并评估候选 1–5（含命中行「算符降代换/公共子表达式消除」）；R2 明确「无在线/离线之分（逐元素天然单遍）」并给结构依据；R3 复杂度四口径表齐全（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值），基线在表；R4 逐候选评估（单元匹配/容量/对齐/静态边界/流水/跨核），引用 `constants.md` 实测常数。轻量调研（`algorithm-research.md` §2 单步逐元素）深度匹配。
- **调研结论独立复核**：
  - *等价性推演*：`isfinite(x) ⟺ (|x|<+inf) ⟺ (x>-inf)∧(x<+inf)` 逐情形推演成立——有限值（±0/次正规/极值）两侧同真；`+inf`（`+inf<+inf` 假）、`-inf`（`-inf>-inf` 假）、NaN（有序 gt/lt 对 NaN 恒假）→ 三者均 False。与 `verify_equiv.py` 重跑一致（fp16/bf16/fp32 各 4115 元素 mismatches=0）。
  - *复杂度复算*：候选间访存/遍数/缓冲量级相同（`N·in_bytes+N·1`、单遍、O(block_size)），候选 2 省宽化缓冲；独立复算与 §1.6.0 R3 表一致（无中间 GM 往返，未漏计）。
  - *负向断言复核*：候选 1（`|x|<+inf`，vabs）淘汰理由 `T.vabs` bf16 × 经 `T.vabs.md` §2.2.1 亲自核对属实；候选 5（位域）`vbitcast` bf16 × 经 `T.vbitcast.md` §2.2.1 核对属实；候选 3 更长、候选 4 语义错误（判别段 B mismatches>0）均有证据。对照 `algorithm-candidates.md` elementwise 命中行（`ALG-elementwise`）未发现被漏掉的已知候选；「无更优替代」写明调研范围。
  - *亲和性淘汰依据*：容量/带宽淘汰引用 `constants.md` 条目；`vcmp` 语义不引用已被推翻的 `CONST-vcmp-ne-nan-unordered`（`INDEX.md` 标 `overturned` 2026-10-14），改引本工具链实测 CG-2026-0024——**此项复核后确认设计判断正确**。
- **弃选论证前提核对（逐条负向论断 API 文档核对）**：亲自打开 `T.vabs.md`（bf16 ×）、`T.vbitcast.md`（bf16 ×）、`T.vand.md`（bool ×）、`T.vnot.md`（bool/int1 ×）、`T.vselect.md`（bool 操作数 ×）、`T.vcmp.md`（无 unordered 模态、`ne` 为有序 → 弃用 `vcmp "ne"`）、`T.vcast.md`（无 `i1→i8` 直连）——**每条弃选/否决前提均与文档核对一致，无「不支持/代价高/无先例」类无据断言**。设计不做 repack（单轴），无重排代价前提可错。
- **调研→设计一致性**：§1.4 算法描述、§3.1 拆解（步骤 2–7）、§6 循环结构均以 §1.6.0 选定算法为对象，与 §1.6.1 优化后公式一致；迁移任务 §0.5/§0.6 处置与调研选族一致，源优化手段 #4 意图承接（升级为等价化简）有逐项说明，无静默丢弃。
- **数学等价优化完整性**：§1.6.1 两项采纳均含四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」；等价性论证经独立推演成立；收益量化（省特殊函数调用 / 省宽化 cast + fp32 中转缓冲）明确；无进一步优化空间有结论（无公共子表达式、无超越函数、无中间 GM 往返）。
- **等价性机器验证**：`verify_equiv.py` 重跑结果与 §1.6.1 内嵌表**逐行一致**（采纳项 1/2 全 EQUIV_PASS、判别段 A 成立、判别段 B 失配），机械复核通过。
- **向量化替代完整性**：§1.6.2 覆盖 §3.3 伪代码与 §6 循环的全部计算点（两个谓词、掩码合并、字节落盘、搬入/搬出、常量预置 = 已替代；block 起始索引 / 尾块大小 = 保留）；与 §6 交叉核对无遗漏——§6 不出现逐元素标量循环。
- **不可替代理由充分性**：保留的 2 处（block 索引＝任务映射、尾块＝动态边界）理由具体到本算子且有同族先例，非泛泛而谈。
- **替代 API 可行性**：替代 API 均有 `docs/Tilelang.language/` 佐证，优先 v-prefix，与维度 1 口径一致。
- **向量化轴与布局决策完整性**：§1.6.3 给出单轴豁免论证（展平 1-D 仅一个非 batch 维，无广播/窗口/归约/重排候选轴），符合 SKILL.md Phase 2「单轴算子豁免」；迁移对照说明 GPU 轴选择是 CUDA 模型输入而非 NPU 结论。
- **布局决策与实现一致**：§1.6.3 结论（I/O 原生 1-D 连续 + 展平轴向量化、无 repack）与 §3.3 伪代码、§4.3 buffer 形状、§6 循环三方一致。

**pass**（本维度无阻塞项，机械 `r3_metrics` pass、等价性重跑一致、弃选前提逐条经文档核对）。
