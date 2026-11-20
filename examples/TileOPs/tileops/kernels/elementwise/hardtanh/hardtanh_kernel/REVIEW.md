# _make_hardtanh_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/hardtanh/_make_hardtanh_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness）
- 源算子: /home/developer/TileOPs/tileops/kernels/elementwise.py（`_make_hardtanh_kernel` L2563–L2608、`HardtanhFwdKernel` L2611–L2624）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 1

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/hardtanh/_make_hardtanh_kernel/verify_equiv.py` → `RESULT: ALL 20 ITEMS EQUIV_PASS`（20 行门禁项全 `max_abs=0.000e+00`、`vratio=0.0000`、`viol=0`；另有 6 行非门禁 `[DIAG]` native-dtype 项亦全 PASS）。逐行比对与 DESIGN.md §1.6.1 内嵌结果表**完全一致**（20/20，含 3 dtype × {fp32 中转 vs GOLDEN 三边界组、fold 三形态、中转无损}）。D-1 无失效。
- **design_calc_check**（`.agents/tools/design_calc_check.py --design …`）JSON checks 摘要：
  - `ub_budget`: **skip**（未解析到 per-level buffer 表）→ 人工复核：DESIGN.md §4.5 fp32=16 KB / fp16·bf16=24 KB，×1.7 最坏膨胀 = 28 KB / 41 KB < 192 KB（A2/A3）✓；
  - `l0c_budget`: **pass**（40×25 fp32 accum = 3.9 KB vs 128 KB；本算子无 Cube，实为 N/A）✓；
  - `core_split`: **skip**（`insufficient parsed dims {}`）→ 人工复核：§5.5 逻辑核数 512~16,384 ≫ 物理 40，核内 `T.serial` 网格跨步、循环上界 `T.ceildiv(n_num, core_num)` 静态 ✓；
  - `r3_metrics`: **pass**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值 四口径齐全）✓。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源算子全文（`/home/developer/TileOPs/tileops/kernels/elementwise.py` L2548–L2625）+ 抽取脚手架（`examples/TileOPs/tileops/kernels/elementwise/hardtanh/_hardtanh_fwd_kernels.py` 由 wrapper 引入）+ wrapper（`hardtanh.py`）+ manifest + 测试基准，与 §0 逐项核对。

- **语义理解正确性 — pass**：§0.1 载 `y = min(max(x, min_val), max_val)`，与源码 L2599–L2603 `T.min(T.max(v32, lo), hi)` 一致；计算域判定「源非 fp8 分支本身即 fp32 中转」经源码 L2600–L2603 逐语句核对成立（`lo/hi = T.cast(min_val/max_val,"float32")`、`v32 = T.cast(val,"float32")`、末端 `T.Cast(val.dtype, …)`）；wrapper docstring L32–L37 明载「fp32 transit … is semantically required and must be preserved」，§0.1 引用准确。契约 golden `F.hardtanh(x.float(), lo, hi).to(dtype)` 与 `test_activation.py` L216–L217 一致；容差 `_get_tolerances`（L39–L45：fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）与 §8.2 一致。输出 shape `same_as_input` 与 manifest L324 `output.shape == input.shape` 一致；`inplace`（manifest L321 默认 false）host 侧处理正确。
- **算法解读完整性 — pass**：§0.3 步骤 1–4（fragment load → cast↑ → `min(max)`+cast↓ → store）逐条覆盖源码 L2595–L2604 全部计算语句，无遗漏、无臆造；数据流图覆盖 `x_reg`/`y_reg` 两 buffer；`block_size=threads×npt` 且 fp32 npt=4 / fp16·bf16 npt=8 经 `_strategy_npt`（源码 L181–L188）+ `ParametricUnaryKernel.default_config`（L815–L819）+ wrapper `_DEFAULT_BLOCK_SIZE`（L73–L77）三处佐证成立；host 侧逻辑（flatten / `is_fp8` 分派）已列出。
- **优化手段覆盖完整性 — pass**：§0.4 的 7 项（register_copy / block×thread / fp32 域 / npt 对齐 / 无 autotune / `min(max)` 2 op / `lru_cache`）与源码逐一对应：`T.alloc_fragment`+`T.copy`+`T.Parallel`（L2595–L2604）、`threads=256`（L2594）、fp32 域（L2600–L2603）、`_strategy_npt`、工厂无 autotune、`min(max)`（L2603）、`@functools.lru_cache(maxsize=32)`（L2563）。未发现被漏识别的源码优化；层级标注（语义/优化/算法）合理。
- **耦合性判定合理性 — pass**：§0.5 每条有四态处置 + 依据。抽检：fp8 分支判「舍弃」有依据（manifest L315 输入 dtype 仅 fp16/bf16/fp32 + wrapper `SUPPORTED_DTYPES` L160）；register_copy / block×thread 判「重新设计」与「NPU 无一维 `threads=`、v-op 作用于整 UB buffer」事实一致；fp32 域判「保留」；`min(max)` 判「等价替换（并优化）」。无「warp shuffle 判保留」类矛盾。
- **NPU 重设计可行性 — pass**：§0.6 R1（一维 `T.Kernel(40, is_npu=True)` + 网格跨步）与 R2（`vclamp`/`vmin`/`vmax`/`copy`/`fill` 向量链 + fp32 中转）均含「源方案 → NPU 新算法 → 语义保持论证」三要素；一维 Kernel、UB 层级、尾块 `tail`、分核方案满足 NPU 约束；R2 语义保持论证含默认逐位等价、折叠恒等、域一致、边界四款，并经 `verify_equiv.py` 逐位验证。
- **设计与决策一致性 — pass**：§1.4（`vclamp` + 折叠）/§3.1–3.2（API 映射）/§4（UB 布局）/§5（分核）/§6（循环）/§7（同步）均描述 §0.5/§0.6 决策后的 NPU 算法；无「§0 判重设计、§3 照抄源方案」脱节。
- **golden 独立性 — pass**：§8.1 golden 直接调库 `F.hardtanh(x.float(), lo, hi).to(dtype)`，未复刻 §0.6 向量链，验证独立性成立；`verify_equiv.py` 同样以库实现为基线。

### 1. API 可行性 — pass

§3.2/§3.4 每条 API 均实查佐证：`T.vclamp`（`docs/Tilelang.language/比较操作/T.vclamp.md` 存在，§2.1 标量/张量边界、§3 前端展开 `npuir_max`+`npuir_min`）、`T.vmin`（`T.vmin.md` §2.2.2 第 3 项明示 src2 可用标量）、`T.vmax`（`T.vmax.md` §2.1 参数表 src1 仅 `tensor`——设计据此**规避标量边界**、改用预填常量 buffer，判据来自文档差异，非臆测）、`T.vcast`（`T.vcast.md` §2.2.1 `f16→f32`/`bf16→f32`/`f32→f16`/`f32→bf16` 均含 `rint`）、`T.copy`/`T.alloc_ub`/`T.fill`（`tilelang/language/allocate.py::alloc_ub` L108、`fill.py::fill` L10 已核，示例 `examples/elementwise/vec_add_1d.py`）、`T.Kernel(n, is_npu=True)`+`T.serial`（`vec_add_2d.py`/`PL-1.22`）。全部 v-prefix API，符合 AGENTS.md。注意项（建议级、非阻塞）：§3.3 用 `vclamp(xc, xc, …)`/`vmin(xc, …, xc)` 的 `src==dst` 原地别名，文档示例均为 src/dst 分离，设计引 `PL-1.23` 佐证；Stage 3 编译探针须实测该原地形态（若报错，退回独立 dst buffer 即可，不影响设计正确性）。

### 2. 内存层级规划 — pass

§4.4 完整 GM→UB→GM 纯 Vector 路径，无 GM→L0 直连等越级路径；§4.3 各 buffer 标注 UB 层级。§4.5 UB 预算：fp32 16 KB / fp16·bf16 24 KB（含未用路径仍分配的 `xf`/`lo_ub`），×1.7 最坏膨胀 28/41 KB < 192 KB，机械 ub_budget 虽 skip 但人工复算成立（见「机械复核」）。无 L0/L1/L0C（无 Cube）。`T.alloc_ub` 全程顶层无条件分配，规避 `TRAP-conditional-alloc`。

### 3. Tiling 策略 — warn

- **Block 划分**：§5.2 明确 `block_size = {fp32:1024, fp16/bf16:2048}`（K11 折叠）及四条理由 ✓。
- **对齐约束**：§5.3 fp32 1024×4B / fp16·bf16 2048×2B = 4096B，32B 整数倍、尾 lane 浪费 0% ✓。
- **非整除处理**：§5.4/§6.4 `tail = T.min(block_size, shape-off)` 动态裁剪，覆盖 manifest `[16,256,56,56]`（12,845,056 % 2048 = 512）与测试 4,000,000（余 1216）✓。
- **分核策略三要素（`core-split-strategy.md` §1/§2.2） — 齐全且自洽**：① 逻辑核数 `ceil(N/block_size)`，逐 workload 列举（512~16,384）✓；② 物理核数**实查记录** `NPUUtils.get().get_aicore_num()→20`、`get_aivector_core_num()→40`（纯 Vector 翻倍），查询代码与返回值均记录，且与 `CONST-aicore-910_9362` 一致（`tilelang/utils/npu_utils.py` L112/L117 方法存在）✓；③ 极大规模判定 + 固定启动 40 核 + 核内 `T.serial` 网格跨步、循环上界静态 ✓；残留负载不均显式记录 ✓。
- **warn（建议级，机械复核外的独立复算发现）**：§5.5 ③ 的残留负载不均算式中，**`bounded-hidden fp16`（n_num=4096）临界值写为 +0.20%，独立复算应为 +0.59%**——4096 = 40×102+16，16 核做 103 次、24 核做 102 次，`103/102.4 = 1.00586`。其余四个 workload 的数值经复算均正确（1M：+1.56% / +1.56%；conv-feat：+0.13%；16m：+0.10% / +0.10%），且该处数值破坏了随规模单调下降的序列，判定为笔误。该错误不影响分核决策与正确性（仅一处性能附录数字），故不阻塞；建议 Stage 1 在 revision/Stage 4 顺手更正为 +0.59%。

### 4. 技术约束检测 — pass

§3.5.1 逐条对照已知限制：无三维 Kernel（一维）；GPU 专用 API（`threads=`/`T.Parallel`）已换 v-prefix；非 GEMM 故无 GEMM 非整除/L0C 溢出；物理核数限制由核内串行处置；bf16 v-prefix 缺失由 fp32 中转处置；条件 alloc 由顶层无条件分配规避；`vmax` 标量不确定性由常量 buffer 规避；v-op 动态 slice 由整 buffer 规避。GEMM 非整除/L0C 已在维度 2/3 处理，无遗漏。

### 5. 循环与同步 — pass

§6.1 明确三级结构：核级 `T.Kernel`、tile 级 `T.serial`、元素级向量化（无 `T.Parallel` 标量循环），与 §1.6.2 结论一致；§6.3 不用 `T.Pipelined`（单数据通路、无 K 维）理由成立，跨 tile 重叠交 Developer auto-multi-buffer；§6.4 尾块逻辑与 §5.4 一致。§7 同步为 Developer 自动同步，与 §2 模式一致，无跨引擎/跨核协作故无手动 set/wait。§7.4 CV 融合 n/a 判定正确（无 `T.gemm`，非 MixCV）。

### 6. 验证方案 — pass

§8.1 含 golden（库实现，独立于 NPU 结构）；§8.2 按 dtype 给 atol/rtol（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与源 `_get_tolerances` 一致）；§8.3 L0 计划覆盖 randn 全域三 dtype、角点（`test_hardtanh_edge` 同款 `repeat([-88.8,-1.0,-0.5,0,0.5,1.0,88.8])`）、上下界邻域、极值 ±65504/±inf、非整除尾块 4,000,000、manifest 多 shape、退化边界折叠（L0-9/L0-10），判定准则与 Stage 3 门禁同口径（`assert_close`，`equal_nan=True`）。

### 7. 完整性与一致性 — pass

章节齐全（§0 含 0.1–0.7，§1–§10）；无占位符（grep 无 `TODO`/`待补充`/`{placeholder}`）；§3.5.3 及 §0.7 引用的 `examples/` 路径经 Glob 逐条核对**全部真实存在**（leaky_relu / elu / bitwise_not / elementwise / test_clamp_dev.py / hardtanh.py / _hardtanh_fwd_kernels.py / test_activation.py / elementwise.yaml）；API 映射、内存规划、Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：含 §1.6.0 调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与布局决策四子节。
- **算法调研存在性与完整性（独立复核）— pass**：R1 候选表含**基线（源 `min(max)`）**在内共 8 行且逐候选评估；R2 在线算法有明确「无在线化收益」结论及结构判据（无 cross-element 依赖）；R3 覆盖四口径（FLOPs v-op 数 / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线在表；R4 逐候选六维亲和评估。设计难度匹配（轻量逐元素但达完整深度）。
- **调研结论正确性（独立复核）— pass**：
  - ① **负向断言「bf16 无 `vclamp`/`vmin`/`vmax`」逐格回 docs 核对成立**：`T.vclamp.md` §2.2.1 bf16 列 `×`、`T.vmin.md` §2.2.1 bf16 `×`、`T.vmax.md` §2.2.1 bf16 `×`（fp16/fp32 均 `√`）；`T.vcast.md` §2.2.1 `f16→f32`/`bf16→f32`/`f32→f16`/`f32→bf16` 均含 `rint`。设计与文档无矛盾。
  - ② **负向断言「无在线变体」对照候选库成立**：`algorithm-candidates.md` `ALG-elementwise` R2 记「逐元素映射天然单遍」，与设计结论一致；无已知在线候选被漏。
  - ③ **复杂度算术独立复算**：全部候选访存 = `2·N·elem_bytes`（1 读 1 写、无中间 GM 往返），FLOPs（v-op/元素）基线 2、折叠 1/0、#4/#5 ≥4、#6 3，扫描遍数 1、中间缓冲 O(block_size)，逐项复算与设计一致；§1.6.0 roofline 表 6 行（流量/发射/估算下界）逐格复算无误（如 smoke-1m fp32：8.4 MB/1.10 TB/s≈7.6µs、ceil(1024/40)×2×0.5µs=26µs）。
  - ④ **亲和性淘汰依据核对**：R4 容量/带宽引用 `constants.md` 条目齐全；`CONST-ascend910-9362-profile`（verified：核→GM≈1.06 TB/s、HBM≈1.64 TB/s、BitwiseNotFwdOp 1.06–1.10 TB/s）、`CONST-vector-launch-overhead`（910B2C、stale）、`CONST-capacity-910B2C`（192KB）均实存且设计标注了设备/stale 适用性。
  - ⑤ **未实证常数与诚实声明**：发射项 0.5µs/op 常数绑 910B2C 且 stale，设计显式标注「未文档化假设 + 估算依据」并附 `bitwise_not` 实测反证（短链实际流量主导），未做过度断言。
  - **候选覆盖复核**：`ALG-elementwise` R1（CSE/除法乘倒数/cast 链合并）已由 §1.6.1 #5（常量提升）/E1（cast 链）/调研范围说明消化；无漏候选。`PL-1.28`（标量参数恒等折叠，支持候选 #2）、`PL-1.17`（sub-fp32 fp32 中转，支持计算域）、`PL-1.22`（一维网格跨步）、`PL-1.23`（v-op 原地复用）均在 pattern-library INDEX.md 实存。
- **调研→设计一致性 — pass**：§1.6.0 选定「`vclamp` + 边界折叠 + fp32 中转」实际驱动 §1.4 算法描述、§3.1 公式拆解（以优化后公式为输入）、§6 循环结构；迁移侧 §0.5/§0.6 处置与调研选族一致，源 §0.4 七项优化意图逐项承接（无静默丢弃）。
- **数学等价优化完整性 — pass**：§1.6.1 采纳 2 项（#1 编译期边界折叠、#2 fp32 中转），每项四要素齐全（原式 → 优化后 → 等价论证 → 收益）；无优化空间结论有依据（默认 `(-1,1)` 2 op 已为最短闭式）。
- **等价性论证独立推演 — pass**：逐项独立推演成立——`max(x,-inf)=x`、`min(x,+inf)=x`（含 ±inf、NaN 经 torch 语义一致）、两侧 inf 恒等、`clamp(x,c,c)=c`；fp32 中转与契约 golden 同域；`verify_equiv.py` 重跑 20/20 `EQUIV_PASS`、`max_abs=0.0`，与 §1.6.1 内嵌表逐行一致。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 8 个计算点，与 §3.3 伪代码、§6 循环结构交叉核对一致（kernel 内无逐元素标量循环）；逐元素计算点 #1–#5 已全部向量化，保留的 #6 核内调度循环/#7 block 索引/#8 动态边界逐项给出具体理由（tile 级调度、block 索引、动态 shape 边界），非空泛。
- **替代 API 佐证 — pass**：`vclamp`/`vmin`/`vmax`/`vcast` 均有 docs + `testing/npuir/arith_ops/test_clamp_dev.py` 实跑先例（该文件实存）。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按 skill Phase 2 第 3 项对单轴算子记录豁免依据（唯一 1D 元素轴、stride=1、无 repack、无 `T.transpose`），轴质量量化（尾 lane 浪费 0%、32B 对齐），并明确「GPU 轴是输入非结论」；布局决策与 §3.3/§4/§6 一致。
- **实验裁决模式 — 不适用（pass）**：算法选择（`vclamp` vs 基线 `min(max)`）不依赖未实证常数（两者同 op 数、同访存下界），判定裕度不落在未知量区间，无需主选+备选+裁决三件套。

> 说明：本次结论为「通过」，全维度无阻塞级 fail；维度 3 记录的 1 项建议级发现（残留负载不均 `+0.20%` 应为 `+0.59%`）不阻塞，供 Stage 1 revision / Stage 4 顺手修正。
