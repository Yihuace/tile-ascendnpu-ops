# bitwise_not_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness）
- 源算子: `/home/developer/TileOPs/tileops/kernels/elementwise.py`（`BitwiseNotFwdKernel` L2273–L2289；`UnaryKernel` L680–L887；`_make_unary_direct` L304–L320）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（均记录于各维度说明，不阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `cd examples/bitwise_not/bitwise_not_fwd_kernel && python3 verify_equiv.py` → `RESULT: ALL 6 ITEMS EQUIV_PASS`（int8/int16/int32/int64 `~x vs x^-1`、uint8 `~x vs x^255`、bool `(x==0) vs (x^1)` 各 1048586/1048583/1048580 samples，max_ulp=0，viol_rate=0.000000）。**执行结果与 §1.6.1 内嵌结果表逐项一致（样本数、verdict 全同）**。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md`
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan"）→ 人工复核：§4.5 表 int64 最大 32KB×1.7=54KB < 192KB（A2/A3），各 dtype 逐项复算成立 → **pass**
  - `l0c_budget`: **skip**（"no affirmative Cube-path API"）→ 纯 Vector 无 L0C，n/a → **pass**
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = 1"，解析器未能识别 1-D 网格）→ 人工复核 §5.5 三要素成立 → **pass**
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）
- **kb_stale_check**（辅助核对）: `CONST-capacity-910B2C` / `CONST-vector-launch-overhead` / `CONST-mte2-degradation` / `PL-1.6-copy-floor` / `PL-1.10` / `PL-1.14` / `PL-1.17` 均报 stale（与设计 §1.6.0/§9 R-7 的 stale 声明一致）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read 源算子 `/home/developer/TileOPs/tileops/kernels/elementwise.py` 全文相关段落，与 §0.1–§0.7 逐项核对。

| 检查项 | 结论 | 源码证据 ↔ DESIGN.md §号 |
|--------|------|--------------------------|
| 语义理解正确性 | pass | 源码 `op_func` L2283–L2289 三分支：`x == T.cast(0,"bool")` / `T.bitwise_xor(x, T.cast(255,"uint8"))` / `T.bitwise_xor(x, T.cast(-1, x.dtype))`，与 §0.1 表格三行逐字一致；`OUTPUT_DTYPE` 未覆盖（父类 L701 `None`）→ `output_dtype = dtype`，§0.1「同 dtype」正确；无规约（`_make_unary_direct` L313–L316 逐元素 `y[idx]=op_func(x[idx])`），§0.1「无规约语义」正确；§0.1 边界 `~INT_MIN=INT_MAX`/`~UINT8_MAX=0` 与补码定义一致 |
| I/O 一致性 | pass | 源码 L312 `main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype))`，输出 `out_dtype = output_dtype or dtype`（L307）↔ §0.2 `x/y` shape `(N,)`、`same_as_input`；manifest L112–113 `output.shape == input.shape` 已核 |
| 算法解读完整性 | pass | `_make_unary_direct` L313–L316 仅两条计算语句（`idx = bx*threads_arg+i`；`y[idx]=op_func(x[idx])`），§0.3 表两行完整覆盖，无遗漏/臆造；host 侧 flatten（`N_total`）已列出（L716）；`bool` 强制 `direct` 与源码 L742–L751 一致 |
| 优化手段覆盖完整性 | pass | §0.4 五项逐项对应源码：#1 `direct` 策略（L280 `DEFAULT_STRATEGY="direct"`）、#3 掩码 XOR 注释（L2276–L2277 "T.bitwise_not fails on vectorized int4 CUDA types"）、#4 bool 标量路径（L736–L751 注释 + 强制 `direct`）、#5 无 autotune（L304 `lru_cache` 工厂无搜索）。层级标注（实现层/硬件强相关）合理，未把实现层优化误判为算法层 |
| 耦合性判定合理性 | pass | §0.5 计 保留 4（语义/算法/#5/host）/ 等价替换 1（#2 register→UB）/ 重新设计 3（#1 并行模型、#3 补码指令、#4 bool 路径）/ 舍弃 0。逐项与 `migration-analysis.md §5.3` 能力映射（GPU thread 模型→一维 Kernel、register→UB）及 NPU 约束（bool i1 不能 load/store GM）不矛盾；无「warp shuffle 判保留」类错误 |
| NPU 重设计可行性 | pass | §0.6 三项均含「源方案 → NPU 新算法 → 语义保持论证」：#1 一维 `T.Kernel(core_num, is_npu=True)`+`T.serial` 网格跨步（语义保持：逐元素独立、无累加顺序变化，静态上界+`task<n_num` 守卫）；#2 原生 `T.vnot`（补码恒等式）；#3 bool 存储语义 `fill(1)+vxor`（{0,1} 同构）。均满足 NPU 硬件约束（一维 Kernel / UB / 32B 对齐），§0.6 R1 附模式先例 `vec_add_2d.py` L28–L42（已核） |
| 设计与决策一致性 | pass | §1.4/§3.3/§6 均为 §0.5/§0.6 决策后的 NPU 算法（vnot/vxor + 一维网格跨步），无「§0 判重设计、§3 照抄源方案」；§3.5.2 转换方案表逐项对应 §0.5/§0.6 |
| golden 独立性 | pass | §8.1 golden=`torch.bitwise_not(x)`（§0.1 语义唯一依据），未复刻 §0.6 NPU 数据流；与源测试 `examples/TileOPs/tests/ops/test_bitwise.py` L30–L31 `ref_program = torch.bitwise_not(x)` + `exact_compare` 同口径 |

> 建议级提示（不影响结论）：§0.3 明确 source `direct` 运行时 `block_size=256`（不乘 npt），而 §5.2 以「K11 折叠 `256×8=2048`」作为 block_size 依据。二者口径不同但均可追溯：K11 折叠为迁移约定（wrapper `examples/TileOPs/.../bitwise_not/bitwise_not.py` L47–L51 明载 `threads*npt=2048`，且 `_strategy_npt` L181–L195 对非 fp32 返回 8）。建议在 §5.2 一句话区分「运行时 source tile=256」与「K11 折叠约定=2048」，避免读者误读。

### 1. API 可行性 — pass

- **API 存在性**（逐条查证）：`T.vnot` ← `tilelang/language/__init__.py:158 npuir_not as vnot` + `docs/Tilelang.language/逻辑操作/T.vnot.md`（§2.2.1 支持 int8/16/32/64/uint8，§2.4 示例）；`T.vxor` ← `__init__.py:89 npuir_xor as vxor` + `T.vxor.md`（§2.4 示例，转 `hivm::VXorOp`）；`T.fill` ← `tilelang/language/customize_npuir.py:702 def npuir_fill` + `__init__.py:119 npuir_fill as fill`；`T.copy`/`T.alloc_ub`/`T.Kernel(is_npu=True)`/`T.serial` ← `examples/elementwise/vec_add_1d.py` L27–L38、`vec_add_2d.py` L28–L42（均已核）。**全部有 sources/examples 佐证**。
- **API 与模式匹配**：Developer 模式，§7 用编译器自动同步（无手动 `sync_block_set/wait`），匹配。
- **v-prefix 优先**：`T.vnot`/`T.vxor` 均为 v-prefix，符合 AGENTS.md。
- 建议级提示：§3.4/§9 R-3 称「`T.vxor(x,1,y)` 不可为标量」，但 `T.vxor.md` §2.1 明载 src1 可为「Python 标量 int / float：会被提升为与 C 相同 dtype 的常量并广播」——**文档与探针证据存在张力**（详见维度 8）。设计采用「tensor src1」（`fill`+`vxor`），无论标量命题真假均可实现，故不阻塞；建议 Stage 1 核对并在 §3.4 收敛表述。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 路径 GM→UB→GM（纯 Vector，无 L1/L0，无越级）；§4.3 各 buffer 标注存储层级（UB），无 GM→L0 直连。
- **UB 预算**（design_calc_check `ub_budget` skip → 人工复核）：§4.5 `block_size=2048`，int64 = (16+16)KB×1.7 = 54KB、int32 27KB、int16 14KB、int8 7KB、bool（3×1B）10KB，**均 < 192KB**（A2/A3）。逐项算术独立复算成立。
- **L0C 容量**：不适用（无 Cube 路径，`l0c_budget` skip 合理）。
- 无阻塞问题。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 `block_size=2048`，四条理由（迁移可追溯 / 32B 对齐 / UB 容量 / 发射项权衡）具体。
- **对齐约束**：§5.3 int32 4B → 512 向量槽、尾 lane 浪费 0%；int64/int16/int8/uint8 均 32B 尾轴对齐。
- **非整除处理**：§5.4 `tail = T.min(block_size, shape-off)` 动态裁剪，未做 host padding（符合 `ascend-constraints.md §4`）。
- **分核策略三要素**（design_calc_check `core_split` skip → 人工按 `core-split-strategy.md §1` 复核）：
  - ① 逻辑核数 `ceildiv(N,2048)`：512 / 8192 / 32768 / 131072（复算正确）；
  - ② 物理核数**实查**：`NPUUtils.get().get_aicore_num()=20`、`get_aivector_core_num()=40`（纯 Vector 翻倍），记录了查询代码与返回值；与 `constants.md CONST-aicore-910_9362`（20 AI Core / 40 Vector Core）一致，未使用 910B2C 的 24 核假设值；
  - ③ 规模判定「极大规模」：取固定启动 40 核 + 核内 `T.serial` 网格跨步，循环上界 `ceildiv(n_num,40)` 为静态值；残留不均显式记录（1M：512=40×12+32 → 前 32 核 13 次、后 8 核 12 次，13/12.8≈+1.6%，**算术复算成立**；大 N <0.05%）。
  - 物理核数含非 2 因子 5，数学上无法用 2^n 的 block_size 对齐 40 的整数倍，判定依据成立。
- 无阻塞问题。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel`，§3.5.1）。GEMM 非整除：不涉及。L0C 溢出：不涉及。
- 已识别 `GPU 专用 API`（`T.bitwise_xor`/`T.Parallel`/`threads=`）并给出替换（§0.5/§0.6/§3.5.2）。
- 无阻塞问题。

### 5. 循环与同步 — pass（建议级）

- §6.1 明确核级 `T.Kernel` / tile 级 `T.serial` / 元素级向量化三层；无逐元素标量循环。
- §7 同步策略（Developer 自动同步 + 线性依赖链）与模式一致；无跨引擎/跨核协作，无手动 flag。
- §6.3 `T.Pipelined` 不使用的理由（无 K 维迭代、跨 tile 重叠交 auto-multi-buffer）合理；§6.4 尾块见 §5.4。
- 建议级提示：§6.3 引用的 `PL-1.6-copy-floor` 已 stale，设计已声明仅作背景并交 Stage 4 实测（§9 R-7），表述自洽。

### 6. 验证方案 — pass

- §8.1 含 PyTorch 参考实现 `golden_bitwise_not = torch.bitwise_not`（§0.1 语义独立）。
- §8.2 精度标准：整数/布尔位运算 **exact（逐位相等）**、明确不用 atol/rtol——对本算子（整数域位运算无舍入）是正确口径，与源 `exact_compare`（`torch.equal`）一致。
- §8.3 L0 门槛测试计划含具体 shape/dtype/判定：int32/int64 smoke、uint8 全范围（含 out_idx dtype 注意）、bool、非整除尾块（4,000,000，已核实不整除 2048：4000000 % 2048 = 1216）。
- 无阻塞问题。

### 7. 完整性与一致性 — pass（建议级）

- 章节齐全：11 章模板 + §0（0.1–0.7 小节齐全）。
- 无占位符：全文无未填写占位标记（「⬜ 待实现」为交付清单的进度状态标记，非占位符）。
- 同类引用真实：§3.5.3 所列 `examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`、`example_elementwise_exp2.py`、`docs/.../T.vnot.md`、`examples/TileOPs/.../mish_kernel/DESIGN.md` 均存在（已 Glob）；§3.4 的 `tilelang/language/__init__.py:158/89/119`、`customize_npuir.py:702` 行号准确。
- 内部一致：§3.2 API 映射 / §4 内存规划 / §5 Tiling 相互一致（block_size=2048、UB buffer 数与 §4.5/§4.3 一致）。
- 建议级提示：§5.2 的 block_size 推导表述见维度 0 提示；建议一并收敛。

### 8. 算法优化分析 — pass（建议级）

**§1.6 章节存在性**：含 §1.6.0 调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代 / §1.6.3 轴与布局决策，四子节齐全。

**算法调研存在性与完整性（四问 + 基线）**：
- R1 候选表**含基线候选**（候选 0 = 源「掩码 XOR」直译），并评估原生 vnot、bool 存储改写、代数变形（判无适用）；✅
- R2「无在线化收益」有明确结论 + 结构判据（无 running 统计量）；
- R3 四口径齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）；`design_calc_check r3_metrics = pass`；
- R4 逐候选评估（单元匹配 / 容量 / 对齐 / 静态边界 / 流水 / 跨核）。

**调研结论正确性（独立复核）**：
- **等价性独立推演**：`~x = x ^ (-1)`（有符号补码逐位恒等，int8/16/32/64 全域成立）；`~x = x ^ 255`（uint8 8 位全 1）；bool `x ^ 1 ≡ (x==0)`（x∈{0,1} 时 0↔1 同构）。三项均位级恒等、无舍入 → 成立。
- **复杂度算术独立复算**：三候选访存同为理论下界 `2·N·elem_bytes`（一读一写、单遍、无 O(N) 中间缓冲，**无漏计的中间缓冲 GM 往返**）；候选 1 少一掩码 buffer、少一条指令；bool 路径 3×1B buffer = 6KB @bs2048。设计数字与复算一致。
- **负向断言复核**：对照 `tilelang-op-design/references/algorithm-candidates.md` —— `ALG-elementwise`（family elementwise）R2「逐元素映射天然单遍」支持「无在线变体」结论；R1 所列 CSE / 除法乘倒数 / cast 链合并对「单目逐元素位反」确不适用，`ALG-elementwise` 无更优化简候选；源算法（§0.3）本身即单遍直写，无 online 手段 → 无矛盾。对照 `pattern-library/elementwise.md`，设计亦标注 `PL-1.6/1.10/1.14/1.17` stale（经 `kb_stale_check` 证实）且不据此决策。
- **并列候选裁决**：三候选访存同为下界，设计以「指令数/UB buffer 数」一级裁决（候选 1 胜），差异明确非纸面裕度拍板。
- **独立复核发现（建议级）**：R4 淘汰基线「掩码 XOR」的负向前提——「`hivm.hir.vxor` 不接受标量 src1」——与所引 `T.vxor.md` §2.1「src1 可为 Python 标量（提升并广播）」**存在张力**；且探针 `design_probe_vnot_int.py` 的 `kern_vxor1` 实测传的是 `T.cast(1, dtype)`（PrimExpr）而非 Python 标量。该负向前提用于论证基线的额外 `fill` 开销。**不阻塞**：采纳项 `T.vnot`（原生 `VNotOp` 单指令、doc §2.2.1 支持、少一 buffer）独立成立，bool 路径采用 tensor src1 的 `fill+vxor` 亦成立；但建议 Stage 1 核对并在 §3.4/§9 R-3/§1.6.0 R4 收敛该表述（若 Python 标量确实可用，bool 分支可省 `ones_ub` 与一条 fill，属 Stage 4 优化空间）。

**调研→设计一致性**：§1.4 算法描述、§3.1 公式拆解（显式声明输入为 §1.6.1 优化后公式 `y=T.vnot(x)` / bool `y=T.vxor(x,1)`）、§6 循环结构均以 §1.6.0 选定算法（vnot + bool 存储语义）为准；§0.5/§0.6 处置与调研选族一致，源优化意图承接逐项说明（§0.6 承接说明段），无静默丢弃。

**数学等价优化完整性**：§1.6.1 三项采纳项均含四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）：`xor(-1)→vnot`、`xor(255)→vnot`、`x==0→x^1`。系统审视了算子降代换与域内改写；对单目位反无更多恒等变形空间已给结论。

**等价性论证正确性 + 机器验证**：逐项独立推演成立；`verify_equiv.py` 重跑 6 项全 `EQUIV_PASS`，**与 §1.6.1 内嵌表逐项一致**；无 fp16/bf16 舍入问题（整数/布尔域），与 §8.2 exact 标准一致。

**向量化替代覆盖完整性**：§1.6.2 表列 5 个计算点，与 §3.3 伪代码、§6 交叉核对：逐元素计算点 #1/#2 已向量化（`T.vnot` / `T.fill`+`T.vxor`，docs + 探针双佐证）；保留 #3 核内调度循环 / #4 block 索引 / #5 动态边界，理由为「tile 级调度循环 + block 索引 + 动态边界」，**具体到本算子且成立**（kernel 内无逐元素标量循环）。无遗漏点。

**替代 API 佐证**：`T.vnot`（`T.vnot.md` §2.4）、`T.vxor`（`T.vxor.md` §2.4）、`T.fill`（`customize_npuir.py:702`）均在 docs/源码有佐证，且为 v-prefix。

**向量化轴与布局决策完整性**：§1.6.3 按「单轴算子豁免」处理——1-D `[N]` 仅一个候选轴，给出豁免依据（无第二轴可枚举、stride=1、无累加链、无需 repack、I/O 布局=核内布局），符合 skill Phase 2 第 3 项对纯单轴算子的豁免规定；并注明 GPU 轴选择是输入而非结论。**布局决策与 §3.3/§4/§6 一致**（UB 一维 buffer、`T.vnot`/`T.vxor` 整 buffer、无 repack）。

**与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入；§6 无 §1.6.2 未论证的逐元素标量循环。

**D-2 roofline（估算）**：§1.6.0 估算明确标注 `CONST-mte2-degradation` 绑定 910B2C 且 `PL-1.6-copy-floor` stale、「带宽数字待重验、仅作量级估算」；设备以 910_9362 实查 40 Vector 核；容量项 0（纯 Vector）。对 stale 常数的处置合规，无阻塞。

---

## 汇总

- **阻塞级问题**: 0
- **建议级问题**: 3（① §3.4/§9 R-3/§1.6.0 R4 的 `vxor` 标量 src1 负向前提与 `T.vxor.md` §2.1 存在张力，需收敛表述；② §1.6.0 调研范围建议显式引用 `algorithm-candidates.md` 的 `ALG-elementwise` 命中行；③ §5.2 block_size 推导需区分「source 运行时 tile=256」与「K11 折叠约定=2048」）
- 以上建议级问题不影响 Stage 3 实现可行性与正确性，**结论为通过**。
