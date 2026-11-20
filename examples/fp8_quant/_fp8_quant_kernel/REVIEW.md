# _fp8_quant_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/fp8_quant/_fp8_quant_kernel/DESIGN.md
- 任务类型: 迁移（scenario=migration, migration_mode=harness）
- 源算子: /workspace/TileOPs-gpu/tileops/kernels/fp8_quant.py
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3

> 本轮为**修订版（v1）复检**。上一轮（v0）结论为「不通过」，唯一阻塞项 = 软件 e4m3 编码符号位（`sb=128·min(max(−v·1e30,0),1)`）在 `0<|v|<1e-30` 负值区间产生错误字节。本检视**亲自重跑机器验证 + 独立复推等价性 + 独立复算复杂度/UB/分核**，确认该缺陷已真正修复。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS（5/5 EQUIV_PASS，与 §1.6.1 内嵌表逐行一致）**。
  重跑命令 `/workspace/venv/bin/python examples/fp8_quant/_fp8_quant_kernel/verify_equiv.py`（torch 2.9.0+cpu, seed=20261014）：

  | item | verdict | detail（实跑输出） | 与 §1.6.1 内嵌表 |
  |------|---------|-------------------|------------------|
  | E1 div -> reciprocal-mul | EQUIV_PASS | fp8 byte mismatch=0/394496, max fp32 ulp=2 (info only) | 一致 |
  | E2 absmax reduction rewrite | EQUIV_PASS | bit-exact=True | 一致 |
  | E3 clamp necessity | EQUIV_PASS | overshoot elems=570 (max 1.0 ulp above 448), observable in fp8=0/394496 | 一致 |
  | E4 software e4m3 encoder | EQUIV_PASS | contract mismatch=7/397568 (all exact -0.0, accepted) ; unexpected mismatch=0 ; nan-lane mismatch=16 (out-of-contract, excluded) ; cos(finite)=1.0000200272 | 一致 |
  | E5 sign-form guard (v0 broken / v1 fixed) | EQUIV_PASS | v0 unexpected mismatch=12/13 ; v1 unexpected mismatch=0/1 | 一致 |

  **空真守卫核对**：逐组核验——E1/E4 的比较对象为 `(257,D)×4 幅值档 + corners(259,128) + random_tiny_neg(1024,128) + nonfinite(24,128)` 全量数组（E1 nt=394496、E4 nt=397568），**无 0 元素比较**；E5 的 `x` 为 `TINY ± + ±0 + ±1e-30` 共 **28** 元素，非空。符号式在 A 组核对中两侧分支均被触发（true/false 均非 0）。**脚本自身未发现空真 PASS 行**。

- **design_calc_check**: `/workspace/venv/bin/python3 .agents/tools/design_calc_check.py --design examples/fp8_quant/_fp8_quant_kernel/DESIGN.md`
  ```json
  {"checks": [
    {"check":"ub_budget","status":"skip","detail":"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
    {"check":"l0c_budget","status":"skip","detail":"no block_M x block_N values parsed"},
    {"check":"core_split","status":"skip","detail":"insufficient parsed dims {}; declared logical cores = 256; recognised workload rows = 6"},
    {"check":"r3_metrics","status":"pass","detail":"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]}
  ```
  - `r3_metrics` = **pass**（四口径齐全，维度 8 采纳）。
  - `ub_budget` = skip → **人工复核（已核，正确）**：§4.5 表 14 行按所列 shape/dtype 逐行复算 = `8192(x_in) + 8×16384(x32,av,lv,t1,t2,step,sbv,ei) + 4096(u8) + 4096(cond) + 384(amax/amax_c/inv) + 128(scale) = 147968B`（fp16 输入）；fp32 输入最坏 = 156160B。`×1.12` → **165724B / 174899B < 196608B**，余量 **≈30.2KB / ≈21.2KB**。**上轮问题 2 的手算错误已修正且修正值经本检视复算一致**（维度 2 记 pass）。
  - `l0c_budget` = skip → 人工复核：**不适用**（纯 Vector，无 Cube/L0C）。
  - `core_split` = skip → **人工复算（已核，正确）**：§5.5 表 6 行 `n_tiles = ceil(R/block_m)` 逐行复算全对（256/64→4、8192/64→128、8192/64→128、4096/32→128、16384/128→128、4096/64→64）；`num_blocks=min(40,n_tiles)`、`iters=ceil(n_tiles/num_blocks)` 静态；三要素齐全（维度 3 记 pass）。checker 的 `declared logical cores = 256` 为其启发式抓取（误计 `BLOCK_ELEMS=4096` 等常量），非设计错误。
  - **`skip ≠ pass`**：本检视对全部 skip 项完成人工复算（见上）。

- **物理核数实查复核（维度 3 要素②）**：本检视亲自重跑
  `NPUUtils.get().get_aicore_num()` → **20**、`get_aivector_core_num()` → **40**、`get_arch()` → **Ascend910B3**
  与 §5.5/§11 记录**完全一致**（纯 Vector 核数翻倍 20×2=40 成立）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

- **语义理解正确性 — pass**：§0.1 数学语义（逐组 `amax=max_d|x|` → `amax_c=max(amax,1e-4)` → `scale=amax_c/448` → `out=cast_fp8(clamp(x/scale,±448))`）、规约语义（规约轴 = 末维 D，`max` 与顺序无关、无跨元素累加）、dtype 语义（中间 fp32、输出 `fp8_e4m3fn`）、边界语义（`clamp(min=1e-4)` 防零除）与源码 `fp8_quant.py:39-57` 逐条一致。§0.2 I/O `[B,S_kv,G,D]/fp16|bf16|fp32 → scale fp32 [B,S_kv,G] + out fp8 [B,S_kv,G,D]` 与源码签名（第 26–30 行）、`register_fake`（第 72–78 行）一致，**非 in-place**（`mutates_args=()` 第 64 行）核对正确。
- **算法解读完整性 — pass**：§0.3 的 S1–S6 覆盖源码全部计算语句——S1↔第 39–40 行装载、S2↔第 43 行 `reduce_absmax`、S3↔第 44–46 行 clamp+mul、S4↔第 49–51 行量化+隐式 fp8 窄化、S5↔第 54–55 行 scale 写回、S6↔第 56–57 行 out 写回；无遗漏、无臆造。host 侧（wrapper/register_fake/Kernel 类/`lru_cache`/autotune，第 64–136 行）已列出。
- **优化手段覆盖完整性 — pass**：O1–O12 覆盖源码可追溯的并行组织（三维 grid）、fragment 驻留、融合 absmax、双写回、`out_idx` 多输出、`lru_cache`、autotune，逐项标硬件依赖，未混淆算法层/实现层。
- **耦合性判定合理性 — pass**：§0.5 每条目四态处置 + 依据；O1（三维 grid→一维）与 O12（autotune/threads）判「重新设计」合 `ascend-constraints.md` §1；O8（fp8 窄化）判「重新设计」有 §0.5.1 源码/实测双重实证（见下）；O2（fragment→UB）判「等价替换」合 `constants.md`。
- **NPU 重设计可行性 — pass（上轮阻塞项已修复，本检视独立复核成立）**：R1（三维展平为一维行维）、R2（线程→Vector）、R4（除→乘倒数）、R5（静态配置）重设计与语义保持论证成立；**R3（软件 e4m3 编码）的语义保持论证现已成立**——符号位改为**与幅值无关**的比较式 `sb=128·[v<|v|]`（⟺`[v<0]`），本检视独立推演 + 独立实算确认全幅值域正确（证据见维度 8）。
  - §0.5.1 的能力缺口复核有源码证据（`codegen_npuir_api.cc:493` / `codegen_npuir_api_a5.cc:764` 无 float8 分支、末尾 `LOG(FATAL)`，全仓 grep 零命中）与 runtime 实测（`.view(dtype)` 可用、`.to(fp8)`=561103、fp8 `zeros_`=161002），负向断言限制在已举证的范围内，符合根 AGENTS.md「负向断言须有具体证据」。
- **§1–§7 与迁移决策一致性 — pass**：§1.4/§1.5/§3.1–§3.5/§4/§5/§6/§7 均描述 §0.5/§0.6 决策后的 NPU 算法（一维 kernel、uint8 载体、无线程轴、静态配置、`cond` 片上 bool），无「§0 判重设计、§3 照抄源方案」脱节。
- **golden 独立性 — pass**：§8.1 golden 直接移植源仓 `tests/ops/test_fp8_quant.py:13-19` 的 `ref_program`（torch 原生 `abs().amax(-1,keepdim).clamp(min=1e-4)/448` → `clamp(x/scale,±448)` → `.to(float8_e4m3fn)`，本检视亲自 Read 核对），**不使用**软件编码器、不使用乘倒数、不使用 uint8 载体——与 §0.6 的 NPU 算法解耦，验证独立性成立。
- **host/op 层引用路径真实性 — pass（附澄清建议）**：§0.1 与 §0.6-R3 引用 `tileops/ops/quantization/fp8_quant.py`；该路径在 NPU 侧脚手架 `examples/TileOPs/` 下**真实存在**（125 行，`forward` 位于 103–125 行，与 §0.1 所注「Op 层第 104–125 行」吻合），其语义（ndim/dtype 校验 + passthrough、无预处理/后处理）与设计描述一致；源仓对应文件为 `tileops/ops/fp8_quant.py`（66 行，验证/passthrough 在 47–66 行，语义相同）。**语义解读正确、所引文件与行号均可解析**，故不构成「路径不存在」的维度 0 fail；但源仓 manifest（`tileops/manifest/quantization.yaml` 的 `source.op: tileops/ops/fp8_quant.py`）用的是扁平路径，设计把脚手架的 `quantization/` 分组路径置于「（源码）」列，易让读者误以为源仓路径。已列非阻塞改进建议（建议 2）。
- **多输出/变体函数的迁移范围 — pass**：源算子仅一个 forward（无 `_with_indices` 类变体），§0.2/§4.2 明确双输出（scale + output），无范围蔓延。

### 1. API 可行性 — pass

- §3.2/§3.4 列出的每条 API 均有佐证：`T.Kernel(n, is_npu=True)`/`T.alloc_ub`/`T.copy`/`T.Parallel` 见 `examples/elementwise/vec_add_2d.py`（本检视 Read 核对）；`T.vabs`/`T.vclamp`/`T.vlog2`/`T.vexp2`/`T.vcast(floor/rint)` 均已在 `history_version/design_probe_tileops.py` 实编 COMPILE_OK，文档存在（本检视逐路径 `ls` 核对全部存在）。
- **本轮新增符号链 API 复核（亲自打开文档核对）**：
  - `T.vcmp(A,B,C,cmp_mod)`：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.2 要求 A/B 同 shape（设计用 `q` 与 `|q|` 同 shape ✓）、§2.3 明确「bool 输出不可直接写回 GM（i1 无 load/store），需配合片上使用」——设计正是如此（`cond` 仅片上）；doc §2.2.1 fp32 列 `√`。
  - `T.vcast(cond, sbv, "rint")`（i1→f32）：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 规格表**第 56 行 `i1 | f32 | rint` 确证存在**。
  - `T.alloc_ub((block_m,D), "bool")`：本仓 **大量先例**（`examples/TileOPs/.../max_pool2d`、`adaptive_max_pool2d`、`_gqa_prefill_fwd_kernel` 等均 `T.alloc_ub(...,"bool")`），可行。
  - 官方测试先例 `testing/npuir/bf16_support_ops/test_vselect_bf16.py`（`T.vcmp(local_a, 0.5, bool_mask, "gt")` + `vselect`）。
- v-prefix 优先：计算路径全用 `T.vabs`/`T.vlog2`/`T.vexp2`/`T.vcast`/`T.vclamp`/`T.vmul`/`T.vcmp`/`T.reduce`，符合根 AGENTS.md。
- 唯一的不可用 API（`e4m3_float8`）已由 §0.5.1 实测披露并绕开；`T.vcmp` fp32 向量化未单独实测已由 §9.1-R11 显式披露（标注「未实测/未文档化」）并给出 `T.vselect` / 自归一化比值式两级备选——属已披露风险，非缺陷。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**：§4.4 给出 `GM → UB → GM` 完整路径，无 GM→L0 越级；纯 Vector 无 L1/L0 路径，标注正确。
- **UB 预算 — pass（上轮问题 2 已修复，本检视复算一致）**：§4.5 表 14 行逐行复算 = **147968B**（fp16 输入）/ **156160B**（fp32 输入最坏）；`×1.12` → **165724B / 174899B < 196608B（192KB）**，余量 **≈30.2KB / ≈21.2KB**。**上轮 131456B 的算术错误已修正为 143872B 的推导并加上 v1 新增 `cond`（4096B）= 147968B，修正值与表内 14 行一致**（本检视独立相加复核）。§5.3 与 §9.1-R5 引用同一数字，已同步一致。降级路径（`BLOCK_ELEMS` 4096→2048）保留。
- **L0C 容量 — n/a**：无 Cube。
- **Buffer 层级标注 — pass**：§4.3 每个 buffer 标注 UB 与 dtype（含新增 `cond` bool，仅片上、标注正确）。

### 3. Tiling 策略 — pass

- **Block 划分 — pass**：§5.2 给出 `BLOCK_ELEMS=4096`、`block_m=BLOCK_ELEMS//D`（D=32/64/128 → 128/64/32），理由为 UB 预算。
- **对齐约束 — pass**：尾轴 D∈{32,64,128}，满足 32B 对齐与 fp32 lane×4 / fp16 lane×8 整除。
- **非整除处理 — pass**：§5.4 行维尾块用 kernel 内行裁剪（`rows=min(block_m,R-row0)`），不用 host padding（合 `ascend-constraints.md` §4）。
- **分核策略三要素 — pass**：
  - ① 逻辑核数：§5.5 表 6 行逐行复算正确（4/128/128/128/128/64）；
  - ② 物理核数：`NPUUtils.get().get_aicore_num()` 实查 **20**，纯 Vector 翻倍 **40**，本检视**独立重跑同值**（含 `get_aivector_core_num()=40`、`get_arch()=Ascend910B3`），非文档假设；
  - ③ 规模判定与方案：smoke（n_tiles=4 ≤ 40）判「无需适配」；其余（128 > 40，`block_m` 增至 205 才使 n_tiles ≤ 40 而 `block_m·D·4` 会撑爆 UB）判「极大规模」→ `num_blocks=min(40,n_tiles)` 固定启动 + 核内 `T.serial(iters)`，`iters` 静态，尾任务 `if tile<n_tiles` 守卫。负载不均（128/40 → 8 核 4 任务、32 核 3 任务，不均率 0.8）已量化披露并给出 `num_blocks=32` 备选。核数与 block 取值自洽。

### 4. 技术约束检测 — pass

三维 Kernel（命中 → §0.6-R1 展平为一维）、GPU 专用 API（命中 → §0.5 处置）、GEMM 非整除（不涉及）、L0C 溢出（不涉及）、分核策略（命中 → §5.5 三要素）五项强制检测均已有处置且处置合理。

### 5. 循环与同步 — pass

- §6.1/§6.2 明确 `T.Kernel`（block 级）+ `T.serial(iters)`（核内任务，静态边界）+ `T.Parallel`/v-prefix（元素级）+ `T.reduce`（D 归约）+ `T.Parallel(block_m)`（行级），与 `examples/elementwise/vec_add_2d.py` 的 `T.serial`+`if block_id<n` 形态一致。
- §6.3 主动不使用 `T.Pipelined`（规避 CG-2026-0001/0002），§7 自动同步（Developer）与模式匹配，无手动 set/wait。
- §6.4 说明尾块行裁剪逻辑，D 轴无尾块。v1 新增的 `cond` 比较链为顺序 v-op，不改变循环结构，§6 与 §1.6.2 仍一致。

### 6. 验证方案 — pass

- §8.1 含可执行 golden（移植源仓 `ref_program`，含 CPU-routed fp8 cast 的 golden-availability 说明；本检视已读源测试确认其来源）与移植源仓的 cosine 比较器（阈值 0.99，与 `test_fp8_quant.py:34-40` 一致）。
- §8.3 含 L0 门槛测试计划：3 种输入 dtype × smoke shape + 全零/±448 边界/±inf/NaN 角点 + 多 kv_group；§8.2 给出 fp8（cosine≥0.99）、scale（atol/rtol=1e-4）等判据。
- §8.2 的「附加强判据（fp8 字节级，仅允许精确 -0.0）」在本版编码下**已成立**（E4：契约域 7/397568 全为精确 -0.0、非预期不一致 0），L0-4 角点用例正是能覆盖该判据的位置。NaN lane 的排除已在 §8.3 L0-4 与 §9.1-R6 显式披露（非阻塞建议 1）。

### 7. 完整性与一致性 — pass

11 章节齐全，迁移类 §0 且 0.1–0.7 小节完整（本检视逐节核对存在）；无占位符残留——本检视三类占位符标记（模板变量花括号、待办标记、未完成填充语句）`grep` 复核均未命中，且 `gate_lint.py` 的 `TEMPLATE_BRACE_RE`（`\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}`）不再命中（v1 已将含中文的 ASCII 花括号集合记号改为全角 ｛｝，残留的 `{0,1}`/`{32,64}` 为无中文的集合记法，不触发该规则）——**上轮「机械修复」性修订正确且完整**；§3.5.3 引用的 examples/ 路径与 docs/ 路径逐条 `ls` 核对**全部存在**；API 映射、内存规划、Tiling 三处描述相互一致（§4.5 小计与 §5.3/§9.1-R5 已同步，上轮内部不一致已消除）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在 — pass**：含 §1.6.0 调研、§1.6.1 数学等价优化、§1.6.2 向量化替代、§1.6.3 向量化轴与布局决策四子节。
- **调研存在性与完整性 — pass**：R1 候选表含基线候选 0（源算法）并枚举 8 个候选；R2 在线算法给出明确结论「无更优在线变体」+ 结构依据（D 为组内短轴 ≤128 全驻留、组间独立、单遍）；R3 复杂度四口径（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）齐备、有基线、口径说明清晰；R4 逐候选六项亲和评估。调研范围段声明了查过的 examples/testing/参考表/pattern-library/源算子/模型知识。
- **调研结论正确性（独立复核）— pass**：
  - **复杂度独立复算**：基线 FLOPs `6·524288+8192 = 3,153,920 ≈ 3.15M` ✓；访存 `524288·2 + 524288·1 + 8192·4 = 1,605,632B ≈ 1.60MB` ✓；候选 4 `12·524288+8192 ≈ 6.30M` ✓；扫描遍数 1（x 全驻留 UB）✓；**中间缓冲无 GM 往返漏计**（全部驻留 UB）✓。
  - **负向断言复核**：对照 `algorithm-candidates.md` ALG-reduction 命中行（R2 列「分块归约+合并；Welford」）——设计已在 R1 候选 7 显式枚举 online D 分块并在 R3/R4 以「D≤128 全驻留、分块反增遍数/破坏向量并行」否决，**未漏候选**，结构依据成立 ✓。R1 候选 6（per-tensor/channel）判「非等价（改变 amax 分组）」正确 ✓。R4 淘汰依据与 `constants.md`/`ascend-constraints.md` 无矛盾 ✓。
  - R3 无并列候选（候选 4 为 codegen 缺口下唯一保契约路径），无需裁决路径 ✓。
- **调研→设计一致性 — pass**：§1.4 算法族 = §1.6.0 选定基线族（无算法族替换）；§3.1 以 §1.6.1 优化后公式为输入（§3.1 步骤 11 即 `[q<abs(q)]` 比较式）；§6 循环结构与 §1.6.2 结论一致；§1.6.0 末段逐项映射源优化意图承接（O2→UB、O3→尾轴搬运、O4→`T.reduce`、O5/O9→向量化、O8→R3 编码、O11→`lru_cache`、O12→R5），无静默丢弃 ✓。
- **数学等价优化完整性 — pass**：§1.6.1 四项均含「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素。
- **等价性论证正确性（独立推演）— pass（上轮阻塞项已修复）**：
  - 采纳项 3（软件 e4m3 编码骨架 `bits=8(ec+6)+rint(a·2^(3-ec))`）：本检视独立推演字段进位（`rint` 达 16 时经 `(ec+7)·8+mantissa` 字段连续性进位）、次正规（`ec` 下钳 −6 后 `rint(a·2^9)` = `rint(a/2^-9)`，步长与 e4m3 次正规一致）、饱和（`a≥448 → ec=8 → 126`）三处边界，均正确。
  - 采纳项 4（符号位）**本检视独立推演**：因 `a=|v|`，`v<|v| ⟺ v<0`（`v>0`→假；`v<0`→真；`v=±0`→假；`NaN`→假），该等价**不依赖任何有限乘数，对任意可表示幅值成立**，故对 `0<|v|<1e-30`、次正规、`±0` 全域成立。**独立实算**（`/tmp/opencode/independent_fp8_check.py`，非设计脚本）：在 **2,400,849** 个 lane（含 fp32 全指数扫描 `2^-149..2^129`、每档 ±1.5×、`±0/±inf/NaN`）上 `[v<|v|]` 与 `[v<0]` 分歧 **0**，且两侧分支均被触发（true=1,151,315 / false=1,249,534），**非空真**。
  - **编码器端到端独立复核**：本检视对 v1 编码器在**独立生成**的 ~3.1M lane（2,000,000 全幅值均匀 + 全指数对数档 + 2×500,000 落在原失效区 `±1e-31/±1e-40` + 每 power-of-two 的 tie/边界 + 特殊值）上与 torch `to(float8_e4m3fn)` 逐字节比对：**非预期不一致 = 0**，唯一差异为精确 `-0.0`（已接受）与 NaN lane（契约域外）。**上轮指出的 `(0,1e-30)` 负值覆盖缺口已被脚本角点集闭合，本检视在缺口区域亦 0 不一致 → 不存在「覆盖缺口下的假 PASS」**。
  - **E5 回归守卫非空真且证伪力成立**：独立实算 v0 有限乘数式在微小负值角点 **12/13 字节非 `-0.0` 错**（如 `-9.9e-31 → 0x7E=+448`、`-1e-35 → 0x00`），v1 比较式 **0** 错——与设计 §1.6.1 E5 明细逐字一致。
  - **附：`T.vlog2` 近似精度未构成风险（本检视独立验证）**：`T.vlog2` 文档明确实现为 `ln`+`mul(1/ln2)`（`docs/.../数学操作/T.vlog2.md` §1/§3），非精确整数 log2。本检视实算：对 log2 结果加 ≤1% 相对扰动后再 `floor`（模拟近似误差导致的 ±1 指数误差），编码器对近 power-of-two 全角点集**仍 0 字节错**——`(ec+7)·8+mantissa` 的字段连续性在 ±1 指数误差处自校正（验证 903 个 power-of-two 邻域值 + 随机集）。故该近似不破坏字节等价，无需新增风险条目。
- **向量化替代完整性 — pass**：§1.6.2 盘点 14 个计算点（与 §3.3 伪代码、§6 循环结构逐点交叉核对，无遗漏），11 个逐元素/归约点已向量化（含 v1 的 `#10` 符号位比较链，替代 API 有 docs/testing/examples 佐证），保留 3 处块级/host 标量点（block 索引 #12、块级控制流 #13、host 元数据 #14）；§6 无 §1.6.2 未论证的逐元素标量循环 ✓。
- **不可替代理由充分性 — pass**：3 处保留分别归因 block 索引 / 块级边界控制流 / host 元数据，属具体到本算子的理由，非遗「照源码写」。
- **替代 API 可行性 — pass**：替代 API 均在 docs/examples 有佐证（`vcmp`/`vcast(i1→f32,rint)`/`alloc_ub bool` 均已亲自打开文档/先例核对），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 有 ≥2 非 batch 轴，候选矩阵枚举 5 案并量化评分（整除性/尾 lane 浪费/累加链形态/repack 代价/UB 影响），弃选理由量化（#2 `T.transpose` 核内 UB 级已核 `docs/.../T.transpose.md`、#4 host permute 百µs 级）；选 #1（D-last，向量轴=D）。迁移说明正确处理「源 GPU 轴是输入非结论」。
- **布局决策与 §3.3/§4/§6 一致性 — pass**：累加/归约内层向量维 = D，buffer 形状 `(block_m,D)`，重排路径（wrapper `.view`）在 §3.3/§4.4 有对应实现。
- **实验裁决模式 — pass/n-a**：§1.6.3 判定裕度为 0（#1 支配其余），唯一未实证常数（向量 lane 吞吐）不影响轴选择，故无需三件套。
- **与下游章节一致 — pass**：§3.1 用优化后公式（步骤 11 = 比较式符号位）、§6 与 §1.6.2 一致。

## 非阻塞改进建议

> 结论为通过；以下为**非阻塞 warn**，不影响进入 Stage 3，供 Stage 1/3 择机消化。

### 建议 1: NaN 在软件编码下为 `0x00`（torch 为 `0x7F`），「对外契约完全保持」措辞应显式限定契约域
- 严重程度: 建议
- 位置: DESIGN.md §0.6-R3（第 1 条「唯一差异是精确 -0.0」与第 3 条边界核对）、§1.6.1、§8.2、§9.1-R6
- 说明:
  §0.6-R3 第 1 条称「与 torch 逐字节一致，**唯一差异**是精确 `-0.0`」，但同节第 3 条与 §9.1-R6/§9.3 又披露 NaN 路径给出 `0x00`（torch `0x7F`）——两处措辞存在轻微张力（「唯一差异」未涵盖 NaN）。本检视确认：NaN 差异**确已显式披露且排除于契约域**（源侧 cosine 比较器 `test_fp8_quant.py:34-40` 无法处理 NaN，源工作负载与测试均无 NaN，属合理的「未文档化假设：索引张量输入内不应出现 NaN」），故非阻塞；但「契约域外」的窄化应显式化，避免下游误读绝对一致性。
- 修改建议:
  1. 将 §0.6-R3 第 1 条与 §8.2 的强判据表述改为「**契约域（有限输入含 ±0/次正规；±inf 经 clamp 后有限）内**与 torch 逐字节一致，唯一差异为精确 `-0.0`；NaN 属契约域外（见 §9.1-R6）」——即在「逐字节一致」前补「契约域内」限定，与 §9.1-R6 口径一致。
  2. 可选增强（非必须）：若欲覆盖 NaN，注意 `T.vcmp(...,"ne")` 在本工具链可能不遵循 IEEE（pattern-library `traps-compiler.md` 记录的 `TRAP-vcmp-ne-ordered-nan`），故「NaN 探测」路径本身有不确定性；在 §9.1-R6 补一句「NaN lane 的 kernel 侧 `T.cast(NaN,"uint8")` 行为为 torch 推定值（`0x00`），NPU 实际行为未实测，L0-4 仅作记录不断言」，保持证据诚实。

### 建议 2: §0.1 host 侧语义引用的 Op 层路径应标明归属根（源仓 vs NPU 脚手架）
- 严重程度: 建议
- 位置: DESIGN.md §0.1（host 侧语义行）、§0.6-R3 备选 ②
- 说明:
  §0.1「依据（源码）」列引用 `tileops/ops/quantization/fp8_quant.py` 与「Op 层第 104–125 行」。该路径+行号精确对应 **NPU 脚手架副本** `examples/TileOPs/tileops/ops/quantization/fp8_quant.py`（125 行，`forward` 在 103–125）；而**源仓**对应文件为 `tileops/ops/fp8_quant.py`（66 行，本检视 Read 核对；manifest `tileops/manifest/quantization.yaml` 的 `source.op` 亦为扁平路径）。语义描述（ndim/dtype 校验 + passthrough）对两者**均正确**，且所引文件真实存在，故非阻塞；但把脚手架分组路径写在「源码」列易误导读者（源仓 ops 无 `quantization/` 子目录）。
- 修改建议:
  将 §0.1 该单元格改为显式区分，例如：`Op 层（源仓 tileops/ops/fp8_quant.py:47-66；NPU 脚手架 examples/TileOPs/tileops/ops/quantization/fp8_quant.py:103-125）只做 ndim/dtype 校验与 passthrough，无预处理/后处理`；§0.6-R3 备选 ② 中的路径同样补根路径。

### 建议 3: E5 明细的「tot」分母语义与 E4 的口径不一致（脚本可读性）
- 严重程度: 建议
- 位置: `verify_equiv.py` `e5()`（第 257–263 行）
- 说明:
  E4 的明细 `contract mismatch=7/397568` 中分母 `nt` 为**元素总数**；而 E5 的 `v0 unexpected mismatch=12/13` 中分母 `tot_v0` 实为**总不一致字节数**（`_classify` 第 4 返回值 = `m.sum()`），并非元素总数（E5 的 `x` 共 28 元素）。同一「`/`」记法在两处含义不同，易被误读为「12/13 个元素错」。判定逻辑本身正确（`other_v0>0 and other_v1==0`），仅标签语义需澄清。
- 修改建议:
  将 E5 明细改为显式标注，如 `v0 unexpected mismatch=12 (mismatched bytes 13) ; v1 unexpected mismatch=0 (mismatched bytes 1)`，或把 `_classify` 的第 4 返回值重命名为 `n_mismatch` 并在两处统一注明「分母 = 元素总数 / 不一致字节数」。
