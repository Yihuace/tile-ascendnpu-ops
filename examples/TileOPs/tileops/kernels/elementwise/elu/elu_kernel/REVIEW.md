# _make_elu_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/elu/_make_elu_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness）
- 源算子: /home/developer/TileOPs/tileops/kernels/elementwise.py（工厂 `_make_elu_kernel` L2497–L2546；Kernel class `EluFwdKernel(ParametricUnaryKernel)` L2548–L2560）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**。`python3 examples/elu/_make_elu_kernel/verify_equiv.py` 实际输出 8/8 `EQUIV_PASS`，与 §1.6.1 内嵌结果表**逐项一致**（#1–#4/#7/#8 max_abs 0.000e+00、viol 0，#5 max_abs 2.980e-08、#6 max_abs 1.192e-07，违反率全 0）；内置非门禁诊断段（expm1-based vs golden 小 x 相对差 1.611e-01 @x=-1e-7）与设计表一致。等价性机器验证有效且与自述一致。
- **design_calc_check** (`python3 .agents/tools/design_calc_check.py --design examples/elu/_make_elu_kernel/DESIGN.md`):
  - `ub_budget`: **skip**（解析器未取到分级 buffer 表）→ 人工复核：§4.5 fp32 合计 8+8+8+8+8+2=42 KB、fp16/bf16 34 KB，×1.7 最坏膨胀 ≈ 71/58 KB，均 < 192 KB 上限——**人工已核，通过**。
  - `l0c_budget`: **pass**（"block 40x102 x fp32 accum = 15.9 KB vs L0C 128 KB"）；本算子纯 Vector 无 L0C，不适用。
  - `core_split`: **skip**（"insufficient parsed dims; declared logical cores = 1"）→ 人工复核：§5.5 三要素齐全（逻辑核数 512–16,384 / 物理核数实查 40 / 极大规模网格跨步），与本节 §3 结论一致——**人工已核，通过**。
  - `r3_metrics`: **pass**（四口径 FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值齐全）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass
**检视方式：亲自 Read 源码 `/home/developer/TileOPs/tileops/kernels/elementwise.py`（L2497–L2560）与 DESIGN.md §0 逐项核对，并在目标设备实跑复核关键前提。**

- **语义理解正确性 — pass（核心前提已设备双证）**：§0.1 数学语义 `y = x if x>0 else alpha*(exp(x)-1)` 与源码不符排查项——源码非 fp8 分支计算体 `T.if_then_else(v32 > zero, val, T.Cast(val.dtype, a * (T.exp(v32) - one)))`（源码 L2538–L2539），设计口径一致。**本检视亲自复核了全案最关键前提「golden `F.elu` 负分支用 `exp(x)−1` 而非 `expm1(x)`」**：本会话在**目标设备 `Ascend910_9362` 实跑** `F.elu([-1e-7,-1e-6,-1e-3,0.0,1e-7,-1.0,-88.8])`，输出 `[-1.1920928955078125e-07, -1.0132789611816406e-06, -0.0009995102882385254, 0.0, 1.0000000116860974e-07, -0.6321205496788025, -1.0]`，与 `torch.where(x>0, x, exp(x)-1)` **逐值一致**（fp32 域）；CPU fp32 复核同结论（`F.elu(-1e-7)==exp(-1e-7)-1==-1.1920929e-7`，与 `expm1(-1e-7)=-9.9999994e-8` 相差 16.1%）。→ **§0.6 R2「计算核直译源公式（保留）」的判定成立**：源公式与契约 golden 天然同式，不存在 expm1 任务那种「源实现精度不足需算法替代」的差距；因此「不替代、直译」是正确处置，且明确拒绝 expm1/Taylor 消抵消方案有实测依据。dtype 语义（源强制 fp32 中转 `v32 = T.cast(val,'float32')`、`a = T.cast(alpha,'float32')`，L2535–2537）与 §0.1 一致；`alpha` 为 compile-time 特化（`EluFwdKernel._builder_args → (alpha,)`，源码 L2559）与 §0.1 一致。
- **算法解读完整性 — pass**：§0.3 步骤表 5 步（fragment load L2531 → cast L2537 → `a*(exp(v32)-1)` L2538–2539 → `if_then_else` 含隐式舍回 → fragment store L2541）覆盖源码非 fp8 计算路径全部计算语句，无遗漏、无臆造；数据流图覆盖 `x_reg`/`y_reg` 两 buffer；host 侧 flatten/`inplace copy_` 已列出（§0.1/§0.3）。
- **优化手段覆盖完整性 — pass**：§0.4 识别的 6 项（`register_copy` L2529–2531/L2541、block×thread 两级并行 L2528/L2532、fp32 中转 L2535–2537、strategy-aware `npt`、无 autotune、单条 `T.exp` 超越指令）与源码逐项对应；源码无非 fp8 分支的其他隐藏优化（无 `T.Pipelined`、无 SMEM、无 warp 原语），未见遗漏。未把实现层优化误判为算法层。
- **耦合性判定合理性 — pass（抽检）**：§0.5 表 9 行，处置计数 = **保留 5 / 等价替换 2 / 重新设计 2 / 舍弃 0**，与设计自述一致。抽检：`#1 register_copy`（GPU register/fragment 模型，NPU 无一维 `threads=`）判「重新设计」、`#2 block×thread`（GPU 线程模型）判「重新设计」——处置与 GPU→NPU 能力映射一致；`#3 fp32 中转` 判「等价替换」（`T.vcast` 有 `rint` 支持，doc §2.2.1）合理；`#4 strategy npt 对齐` 判「等价替换」（折叠为 `block_size`）合理。无「warp shuffle 类判保留」等矛盾，无舍弃项。
- **NPU 重设计可行性 — pass**：§0.6 R1–R4 逐项有「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步（非三维 Kernel、无 `threads=`，满足 NPU 一维约束；语义保持论证：元素独立、无累加顺序变化、`tail` 守卫覆盖非整除）。R2 `vexp→vsub→vmul→vcmp/vselect` 向量链，边界（x=0/±∞/NaN/88.8）逐项论证与 golden 一致。R3 `vcast` fp32 中转（bf16 为 API 强制，doc 佐证 `vexp/vsub/vmul` 不含 bf16）。R4 无分支 `vcmp+vselect`（bool 仅片上）。新算法满足 NPU 内存层级/对齐/容量约束。
- **设计与决策一致性 — pass**：§1.4 算法描述、§3.1 公式拆解（输入为 §1.6.1 优化后公式）、§3.3 伪代码（一维核 + 网格跨步 + 向量链）、§4 内存规划、§5 分核折叠、§6 循环结构描述的都是 §0.5/§0.6 决策后的 NPU 算法；未出现「§0 判重设计、§3 照抄源方案」的脱节。§0.5/§0.6 处置与 §1.6.0 调研选族一致（选定「基线直译 + alpha 折叠」）；源 §0.4 优化意图在第 199 行逐项承接（#1→R2/R4、#2→R1、#3→R3、#4→K11+§5.3、#5 保留、#6→1×vexp+4–6 op），无静默丢弃。
- **golden 独立性 — pass**：§8.1 golden `golden_elu = F.elu(x.to(float32), alpha).to(dtype)`，以 §0.1 契约语义为唯一依据，**不复刻 §0.6 的向量链结构**（直接调库），验证独立性成立。
- **备注（建议级，见维度 7）**：§0.7/§9.1 R-14/§10 声称 NPU 测试文件含 `test_elu_alpha`/`EluAlphaFixture`，实测该用例不存在，不影响维度 0 的语义/算法/耦合性结论。

### 1. API 可行性 — pass
§3.2 每条 API 均在 `examples/` 或 `docs/Tilelang.language/` 找到佐证，本检视逐项核对：`T.vexp`/`T.vsub`/`T.vmul`/`T.vcmp`/`T.vselect`/`T.vcast` 在 `examples/erf/erf_fwd_kernel/erf_fwd_kernel.py`（L161/165/182/184/185/190）与 `examples/bitwise_not/...` 实有使用；`T.alloc_ub`/`T.copy`/`T.fill` 见 `examples/elementwise/vec_add_1d.py`（L28/34/37）与 `bitwise_not_fwd_kernel.py`（L104）；`T.Kernel(n, is_npu=True)`+`T.serial` 见 `bitwise_not_fwd_kernel.py` L89–L108、`vec_add_2d.py`。docs 侧：`T.vexp.md`/`T.vsub.md`/`T.vmul.md`/`T.vcmp.md`/`T.vselect.md`/`T.vcast.md` 均存在且规格与设计一致（vexp/vsub/vmul 仅 fp16/fp32、vcmp/vselect 含 bf16、vcast `f16→f32`/`f32→bf16`/`bf16→f32` 均支持 `rint`）。Developer 模式（自动同步）+ v-prefix 优先（无 `npuir_xxx`）符合 AGENTS.md 约定。

### 2. 内存层级规划 — pass
§4.4 给出完整 `GM[x]→UB→GM[y]`（fp32）/`GM→UB→vcast→UB(fp32)→链→vcast→UB→GM`（fp16/bf16）路径，无 GM→L0 直连等越级；纯 Vector 无 L1/L0A/L0B/L0C（正确）。§4.3 每个 buffer 标注 UB 层级。§4.5 UB 预算 fp32 42 KB / fp16·bf16 34 KB（×1.7 膨胀 ≈71/58 KB）< A2/A3 192 KB——本检视手工复算一致（`ub_budget` 机械 skip 已人工复核）。bf16 经 fp32 中转由 doc dtype 矩阵确证。

### 3. Tiling 策略 — pass
§5.2 给出 `block_size` 具体值（fp32=1024 / fp16·bf16=2048，源 `threads×npt` K11 折叠）及理由；§5.3 对齐（`block_size×elem_bytes` 为 32B 整数倍、尾 lane 浪费 0%）成立。**分核三要素齐全**：① 逻辑核数 `ceil(N/block_size)`（512–16,384）；② 物理核数**实查**——本检视在目标设备实跑 `NPUUtils.get().get_aicore_num()=20`、`get_aivector_core_num()=40`（与 §5.5 及 `CONST-aicore-910_9362` 一致）；③ 极大规模（逻辑核 ≫ 40，且 40 含非 2 因子 5 无法由 2 的幂 block_size 整除对齐）→ 固定 40 核 + `T.serial` 网格跨步，循环上界静态，残留负载不均已显式记录（与 `core-split-strategy.md` §1 要素③一致）。模式先例 `vec_add_2d.py`/`bitwise_not_fwd_kernel.py` 真实存在。非整除由 §5.4 `tail = T.min(block_size, shape-off)` 处理。
> 备注（不影响结论）：§5.5 smoke-1m fp32 残留不均记为 `+2.3%`，按设计自述的 ceil/floor 口径复算为 `26/25−1 = 4.0%`（mlp-hidden `103/102−1=0.98%`、16m `410/409−1=0.24%` 两例自洽）；该数字仅为负载不均标注，分核方案本身正确，建议 Stage 4 以实测为准。

### 4. 技术约束检测 — pass
无三维 Kernel（一维 `T.Kernel(40)`）；非 GEMM；无 L0C 溢出；bf16 v-prefix 数学 API 缺失已由 fp32 中转处理；`vcmp` bool 仅片上（`T.vcmp.md` §2.3 i1 不可 load/store GM 已核对）；编译期分支 `if need_cast:/need_mul:` 在 `T.Kernel` 顶层无条件分配 buffer（规避 `TRAP-conditional-alloc`）；`alpha`/常数以 Python 编译期常量传入。

### 5. 循环与同步 — pass
§6.1 明确循环层级：核级 `T.Kernel(core_num, is_npu=True)`（block 级并行）、tile 级 `T.serial(ceildiv(n_num, core_num))`（网格跨步）、元素级向量化（4–7 op，无 `T.Parallel` 标量循环）；与 §1.6.2 向量化结论一致。§6.3 未用 `T.Pipelined`（单 tile 内线性依赖，跨 tile 由 auto-multi-buffer 处理），理由成立。§7 同步为 Developer 自动同步（无手动 set/wait，无 CV 融合），与模式匹配。§5.4/§6.4 尾块处理明确。

### 6. 验证方案 — pass
§8.1 含 PyTorch golden 草案（`F.elu(x.float(), alpha).to(dtype)`，独立于 NPU 向量链）；§8.2 按 dtype 给 atol/rtol（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与 NPU `examples/TileOPs/tests/ops/test_activation.py::_get_tolerances` 一致，本检视核对）；§8.3 L0 计划覆盖 randn 全域（fp32/fp16/bf16）、角点（±1e-7/±88.8/0/±inf）、负分支小 x 敏感段、分支点、非整除尾块、非默认 alpha，规则 shape 与 dtype 明确，判定门槛明确。

### 7. 完整性与一致性 — warn
模板章节齐全（§0.1–§0.7 + §1–§10，含 §1.6.0–§1.6.3），无占位符残留；示例引用路径经 Glob 核对**真实存在**（`vec_add_1d.py`/`vec_add_2d.py`/`bitwise_not_fwd_kernel.py`/`expm1 DESIGN.md`/`erf_fwd_kernel.py`/NPU `elu.py` 等）；API 映射/内存规划/Tiling 三处描述相互一致。**唯一建议级问题**：§0.7、§9.1 R-14、§10 引用的 `test_activation.py::EluAlphaFixture` / `test_elu_alpha`（称其 golden 硬编码 `F.elu(x,1.0)` 与 `op(alpha=0.5/2.0)` 不一致）在 NPU 测试文件中**不存在**——该文件开头明确写「The GPU `test_elu` exercises the default `alpha=1.0` only, so no non-default `alpha` case is ported here」，全仓 grep 无 `test_elu_alpha`/`EluAlphaFixture`。此为本设计中唯一可核实的引用不实项，不改变 kernel 设计与精度结论（alpha 通用性由 `verify_equiv` #5/#6 与 L0-8 覆盖），不影响通过。

### 8. 算法优化分析 — pass
- **§1.6 章节存在性 — pass**：§1.6.0 算法调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代分析 / §1.6.3 向量化轴与布局决策四子节齐全。
- **算法调研存在性与完整性 — pass**：R1 候选表**含基线**（#0 源算法直译，且参考表命中行 `ALG-elementwise` 逐项评估：CSE→§1.6.1 #3、除法乘倒数 N/A、cast 链合并→§1.6.1 #2）；R2 明确结论「无在线化收益」+ 结构依据（无跨元素依赖/无 running 统计量），与 `ALG-elementwise`「逐元素映射天然单遍」一致；R3 复杂度表**四口径齐全且基线在表**；R4 逐候选亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）。**调研范围写明**（参考表 / examples 全系 / pattern-library / 外部已知算法 / docs 全目录枚举），非空白。
- **调研结论正确性（独立复核）— pass**：① 等价性——选定「基线直译 + alpha 折叠」与原语义等价，本检视设备实跑确认负分支 `exp−1` 与 golden 逐值一致；② 复杂度——四口径复算：访存 `2·N·elem_bytes`（输入读+输出写，无中间 GM 往返）成立、扫描 1 遍、中间缓冲 O(tile) 非 O(N)，op 数 4–7 与 §3.3 一致（`r3_metrics` 机械 pass）；③ 亲和性——与 `CONST-elementwise-op-precision-910_9362`（vexp rel 1.182e-7、vsub 精确）不矛盾；④ **负向断言主动复核**：本检视枚举 `docs/Tilelang.language/数学操作/` 全目录（`T.vexp/T.vexp2/T.vrelu/...`，**无 `T.velu`**）+ `customize_npuir.py`（有 `npuir_exp` L295，**无 `npuir_elu`**）+ grep 全 `tilelang/language/`，证实「无原生 velu」；`T.vrelu` 语义为 `max(x,0)` 非 ELU；「无在线变体」对照 `ALG-elementwise` R2 成立；对照候选库无 ELU 专门条目，设计未漏已命中的已知候选；⑤ 无并列候选（#2/#3 等价但 op 更多，被基线支配），无需裁决路径。
- **调研→设计一致性 — pass**：§1.4（直译 + fp32 中转 + 无分支选路 + 一维网格跨步）、§3.1（以 §1.6.1 优化后公式为输入）、§6（无逐元素标量循环）均与 §1.6.0 选定算法同口径；迁移 §0.5/§0.6 处置与调研选族一致，源 §0.4 优化意图逐项承接（无静默丢弃）。
- **数学等价优化完整性 — pass**：§1.6.1 三项采纳项均具四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」：#1 编译期 `alpha==1.0` 折叠（数学恒等 ×1.0，省 1 op/tile）、#2 fp16/bf16 fp32 中转（≡源 fp32 opmath 语义）、#3 CSE/原地复用（`xc` 链尾原地、`e` 原地、零 buffer 循环外一次填充，省 1 buffer）。
- **等价性论证独立推演 — pass**：本检视独立推演并机器复跑：#1 ×1.0 恒等元成立；#2 `fp16↔fp32`/`bf16↔fp32` 往返无损（`rint` 单次舍回）、fp32 域算后单次舍回 ≡ 源语义成立；#3 中间量生命期结束后复用不改数值、`vcmp` 先于 `vcmp` 后对 `xc` 的覆写（无 WAR 冒险）成立。`verify_equiv.py` 重跑 8/8 `EQUIV_PASS` 且与 §1.6.1 内嵌表一致。被拒绝的 expm1/Taylor 项有非等价诊断（x=-1e-7 差 16.1%）支撑，判定为「非等价、禁止采纳」正确。
- **向量化替代覆盖完整性 — pass**：§1.6.2 覆盖 §3.3 伪代码 / §6 全部计算点（9 项），逐元素计算点 #1–#6 全部向量化（`vcmp/vselect/vexp/vsub/vmul/vcast/fill`，均有 docs 佐证 + 设备探针），保留 #7 核内调度循环 / #8 block 索引 / #9 动态尾块边界三项，与 §3.3/§6 交叉核对一致——kernel 内**无逐元素标量循环**。
- **不可替代理由充分性 — pass**：保留 3 项理由具体到本算子（tile 级任务调度循环、block 级少量标量索引、依赖动态 shape 的尾块边界），非「实现简单/照源码写」式空话；并给出先例 `vec_add_2d.py` L32 / `PL-1.22-persistent-1d-grid-stride`。
- **替代 API 佐证 — pass**：替代方案 API 均有 `docs/Tilelang.language/` 与 `examples/` 佐证（与维度 1 同口径），且优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 对 1-D 单轴算子按 skill 豁免条款记录「唯一候选轴 = 1D 元素轴」的豁免依据（轴质量：32B 对齐、尾 lane 浪费 0%、无累加链、无 repack；I/O 布局即核内布局，一维连续无需 `T.transpose`/host permute），并正确指出「GPU 轴选择是输入而非结论」。单轴豁免成立，无第二轴可枚举，无弃选项需量化理由。
- **布局决策与实现一致 / 与下游章节一致 — pass**：§1.6.3 选定轴与 §3.3 伪代码（向量链作用于 UB 一维 buffer）、§4（buffer 形状 `(block_size,)` 与 UB 预算）、§6（无逐元素标量循环）三方一致；§3.1 以 §1.6.1 优化后公式为输入；§6 无 §1.6.2 未论证的标量循环。
