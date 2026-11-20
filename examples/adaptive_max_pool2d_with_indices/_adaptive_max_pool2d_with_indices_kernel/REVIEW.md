# `_adaptive_max_pool2d_with_indices_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/DESIGN.md`（revision v1，`revision_index = 1`；对照版 `history_version/design_v0.md`）
- 任务类型: 迁移（TileOPs 同构仓迁移，harness 子模式，**双输出**变体 values + int64 indices）
- 源算子:
  - `/workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_max_pool2d.py`（目标函数 `_adaptive_max_pool2d_with_indices_kernel` 第 119–206 行；同文件 13–74 行单输出变体为对照；全文 253 行已逐行核对）
  - `/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`（`adaptive_bin` / `max_adaptive_bin_extent` / `AdaptivePool2dKernelBase`；全文 192 行已核对）
  - `/workspace/TileOPs-gpu/tileops/ops/pool.py`（共享 host/op 模块：`_AdaptivePool2dFwdOpBase` / `AdaptiveMaxPool2dIndicesFwdOp` / `_adaptive_pool2d_roofline` / `_normalize_output_size`，第 918–1173 行已核对）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 上一轮结论: **不通过**（2 阻塞 + 5 建议）；本轮为**修订后重检**
- 阻塞级问题数: **0**
- 建议级问题数: 3

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS`
  - 命令：`/workspace/venv/bin/python examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/verify_equiv.py`
  - 实测输出与 §1.6.1 内嵌结果表**逐字一致**：`N1/N2/N3/N3b = 340 checks 各`、**`N3-ord = 342 checks`（本轮新增守卫确实存在）**、`C4/C4b = 120 checks 各`；
    `worst ulp-ish diff = 0.0e+00`、`miss = 0`、`viol_rate = 0.0%`；**7 项全 `EQUIV_PASS`**，末行 `[EQUIV] ALL PASS`，
    `total (shape, out, kind) cases swept: 340 x 2 dtypes`。脚本存在、执行成功、与内嵌表一致——无脚本缺失 / 无执行失败 / 无不一致。
  - **`N3-ord` 独立核查（本轮重点，亲自读脚本 + 独立复算）**：`_running_first_idx(x, oh, ow, value_first=False)` 建模正确次序
    （`gt = v > acc` **先于** `acc = max(acc, v)`）、`value_first=True` 建模 v0 错误次序（先更新 `acc` 再比较）；
    `_order_counterexample_ok()` 硬编码反证用例存在；主循环的双向断言 `order_ok`（正确次序复现基线）与
    `guard_ok = (~detectable | (badi != bidx)).all()`（值优先次序在一切可检出元素上必被检出）齐备。
    **本 Reviewer 独立复算反证用例**：`arange(12).reshape(1,1,3,4)→(2,2)` 的 `bin(1,1)` = 行 1..2 × 列 2..3（扁平 6/7/10/11），
    最大值在扁平 **11**；`seed f0 = 1*4+2 = 6` ⇒ 正确次序 **11**、错误次序退化 **6** ⇒ 反证用例真实可检出 ✓。
    **独立复算守卫非空性**：general 矩阵（fp64）× 10 种数据上，守卫 `detectable` 掩码 `(oki != seed) & ~isnan(bval)` 命中 **9769** 个元素
    （≫ 0，非空成立；脚本自身已断言其可检出性）。见维度 8 的次要标注。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design <path>` JSON checks 摘要：
  - `ub_budget` = **skip**（`no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)`）
    → **人工复核结论：通过**。本 Reviewer 按 §3.3 buffer 清单 + §4.3 逐项独立复算（见维度 2）：一般网格 spp-6x6/CH=64 = **66 688 B ≈ 65.1 KB**
    （与 §4.5 一致，×1.7 ≈ 111 KB < 192 KB ✓）；**退化路径两行精确复现**：C4 `CH_deg=32`/plane 49 = **68 200 B ≈ 66.6 KB**（×1.7 = 115 940 B ≈ 113.2 KB ✓）、
    C4b `CH_deg=8`/plane 99 = **15 048 B ≈ 14.7 KB**（×1.7 = 25 582 B ≈ 25.0 KB ✓）——**无漏 buffer**。
  - `core_split` = **skip**（`insufficient parsed dims {}; declared logical cores = None`）→ **人工复核结论：通过**。
    本 Reviewer 按 §3.3/§5.2/§5.5 独立复算：一般网格 `NUM_LOGICAL = N*out_H*ceil(C/CH)`（spp 24、nondiv 14、`(1,96,28,30)` 14、`(1,16,11,13)` 5、smoke 2）；
    C4 全局 `NUM_LOGICAL = N*ceil(C/CH_deg)` —— `global-1x1` = `8*ceil(2048/32)` = `8*64` = **512**，`num_kernels = min(512,40) = 40`、
    `num_local = ceil(512/40) = **13**`（静态值）、`512 = 40*12 + 32`（负载差 1 任务）✓；C4b 恒等 `(1,8,9,11)`（C=8 ⇒ `CH_deg=8`）= 1 单元 ✓。
    单元粒度与 §3.3 解码、§4.3 buffer、§5.5 ① 全文唯一自洽 ✓。
  - `l0c_budget` = **pass**（`block 56x56 x fp32 accum = 12.2 KB vs L0C 128 KB`）—— 该输出为**误解析**（本算子纯 Vector、无 `T.gemm`/Cube/L0C 路径）；
    **人工复核结论：不适用**（§5.1/§7.4 已声明无 Cube，`L0C` 无意义）。
  - `r3_metrics` = **pass**（`all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)`）✓。
- **附加机械核对（本 Reviewer 主动执行）**：
  - **物理核数独立实查复现**：`/workspace/venv/bin/python -c "...NPUUtils.get().get_aicore_num()"` → **20**（Ascend910B3）⇒ 纯 Vector 翻倍 **40**，
    与 §5.5 ② 记录**完全一致** ✓（`CONST-aicore-910B2C` 的 910B2C 值 24 未误用）。
  - **源语义独立 torch 实测**（torch 2.9.0 CPU，本会话）：`arange(12) 3×4→2×2` 值 `[5,7,9,11]`/索引 `[5,7,9,11]`（`f=ih*W_in+iw`，W_in=4）；
    全 0 窗口 → idx **0**（tie first-hit）；两个最大值（扁平 0 与 2）→ idx **0**；两个 NaN（扁平 0 与 2）→ idx **2**（**last-NaN**）；
    `{+inf,-inf}` → 值 `inf` / idx **0**（**非 NaN**）；全 `-inf` → 值 `-inf` / idx **0**；`inf`+`nan` → 值 `nan` / idx **1**；
    `2×2→3×3` 扩张 bin → 索引 `[[0,1,1],[2,3,3],[2,3,3]]`；bf16 NaN 窗口 → 值 `nan` / idx 0；3D CHW `(2,3,4)→(2,2)` → 两路输出 shape `(2,2,2)`、索引 int64。
    **§0.1 实测表全部独立证实** ✓。
  - **API 佐证亲自回查**：`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 dtype 表 Ascend 行含 **int32 √ / int64 √**（设计「int32 转置无本仓先例但 docs 支持 + 预置 fp32 回退」的处置有据）；
    `docs/Tilelang.language/逻辑操作/T.vand.md` / `T.vor.md` **存在**（int32 支持）——R1 #8 复合键弃选所引位运算 API 真实存在，**该弃选不是 API 缺口而是收益/代价裁决**（见维度 8）。
  - **引用路径逐条实查**：§3.5.3 的 14 条 `examples/` 引用**全部存在**；设计声明「源仓无独立 `ops/pool/` 目录」经
    `ls /workspace/TileOPs-gpu/tileops/ops/` 证实（有 `pool.py`、**无** `pool/` 目录）✓。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源算子三文件全文（目标函数 `adaptive_max_pool2d.py` 第 119–206 行、`common.py` 全文、`ops/pool.py` 第 918–1173 行），
与 §0.1–§0.7 逐项核对，并对 §0.1 每条语义断言做独立 torch 实测（见「机械复核」末项）。

- **语义理解正确性 — pass（源码证据逐条落实）**：
  - 扁平索引 `f = ih*w_in + iw`、int64 落盘 ↔ 源码第 179 行 `flat_idx = T.cast(ih * w_in + iw, idx_dtype)`、第 200–202 行
    `indices[...] = T.cast(T.if_then_else(has_nan, nan_idx, max_idx), "int64")`（§0.1 #1，DESIGN 第 122–123 行）；实测 `arange(12)→(2,2)` = `[5,7,9,11]` ✓
  - tie → **first-hit** ↔ 第 190 行 `elif val > max_val:`（严格 `>`，相等不覆写）（§0.1 #2）；实测两个最大值 → idx 0 ✓
  - 窗口含 NaN → 值 NaN + **last NaN visited** ↔ 第 181–185 行 `if is_nan: nan_idx = flat_idx; has_nan = True`，源码注释
    「PyTorch records the last NaN visited in a pooling window」；末尾第 194–202 行判决（§0.1 #3）；实测两个 NaN（扁平 0 与 2）→ idx **2** ✓
  - 访问序行主序 ↔ 第 173–174 行 `for kh in T.serial(max_kh): for kw in T.serial(max_kw):`（§0.1 #4）✓
  - `{+inf,-inf}` → 值 `+inf` **非 NaN**（§0.1 #5，实测 ✓）；全 `-inf` → 值 `-inf`、索引首个元素（§0.1 实测 ✓）
  - bin 永不为空 / 整数差边界 ↔ `common.py` 第 100–118 行（docstring「Bins are never empty, including when `size_out > size_in`」）（§0.1 #6）✓
  - dtype 语义 ↔ 第 129 行 `accum_dtype = "float"`、第 178 行 `T.cast(x[...], accum_dtype)`、第 199 行单次舍回；
    输入仅 fp16/bf16 ↔ `ops/pool.py` 第 951–956 行 `_validate_adaptive_pool_input_dtypes`（§0.1 dtype 段）✓
  - 索引 dtype ↔ 第 139 行 `idx_dtype = "int32" if h_in * w_in < 2**31 else "int64"`（§0.1 已声明本设计不支持 `>= 2**31` 分支，§9.1）✓
  - I/O 契约（双 NCHW 同形、`indices` int64、3D→4D `unsqueeze(0)` → **双路各自 `squeeze(0)`**）↔ `ops/pool.py` 第 1041–1089 行
    `if self._returns_indices: return out.squeeze(0), indices.squeeze(0)`（§0.2 / §0.1 边界语义）✓
- **算法解读完整性 — pass**：§0.3 步骤表 1–7 覆盖第 119–206 行**全部**计算语句（`total_output`/`max_kh`/`max_kw`/`idx_dtype`、
  一维网格、`out_idx` 解码、`adaptive_bin`、5 个状态初值、三路更新分支 `if is_nan / elif first_valid / elif val > max_val`、NaN 判决、双写出），
  无遗漏亦无臆造；host 侧逻辑（`_normalize_output_size` / `_resolve_out_dims` / `unsqueeze`/`squeeze`/`contiguous` / `_kernel_cache` /
  `functools.lru_cache(maxsize=32)` / dtype gate / `_adaptive_pool2d_roofline`（含 `out_elems*8`）/ `compile_flags`）逐条对应 `ops/pool.py` 与第 134 行 ✓；
  数据流图覆盖全部 buffer ✓。
- **优化手段覆盖完整性 — pass**：§0.4 列出 14 项；核对源码确认**无** warp shuffle / SMEM swizzle / mma / `T.Pipelined` / cp.async，
  故「舍弃项：无」（§0.5 末）成立。双输出变体相对单输出变体的新增项 #5 first-hit（第 190 行）、#6 last-NaN（第 181–185 行）、
  #7 int32 索引（第 139 行）、#8 双输出融合（第 179/200–202 行）识别准确；**上一轮建议 4（#4 机制引用范围）已修正**——
  §0.4 #4「机制」列现引用**本次目标函数**第 180–185 行，并注明单输出变体形态（第 62 行）不在迁移范围（DESIGN 第 238 行）✓。
- **耦合性判定合理性 — pass**：§0.5 每项四态处置（保留/等价替换/重新设计/舍弃）+ 依据（migration-analysis.md §5.3 / `docs/Tilelang.language/` / pattern-library）；
  三条「重新设计」与 GPU→NPU 映射一致（#10 CUDA thread lane → 一维 Kernel + C 向量；#11 寄存器标量状态机 → UB 向量累加器；#12 逐元素 GM gather → UB tile + 核内转置）；
  语义层项无条件「保留」有依据；「舍弃项：无」与源码事实一致 ✓。
- **NPU 重设计可行性 — pass**：§0.6 R1–R5 均含「源方案 → NPU 新算法 → 语义保持论证」，满足 NPU 约束：一维 `T.Kernel` ✓、无跨核归约 ✓、
  静态 `T.serial` 界 + 守卫 ✓、C 轴为 8 的整数倍对齐（代表 shape）✓、UB 预算两档均 < 192 KB ✓（含修正后的退化路径，见维度 2）。
  **上一轮阻塞级 1 已修复**：R3 的 first-hit 等价性（破纪录集合取 max）与**实现次序约束**（`gt` 与**更新前**的 `acc` 比较）在 §0.6 R3、§1.6.1 N3、
  §3.1 次序注、§3.3/§6.2 注释、§1.6.2 #1/#4、§9.2 六处**一致声明**；承接行 `gt`+`vmax`（**上一轮建议 5 的 `vmin` 笔误已修正**，DESIGN 第 366–367 行）✓。
- **golden 独立性 — pass**：§8.1 直接 `F.adaptive_max_pool2d(x.to(torch.float32), size, return_indices=True)`，显式声明「不复刻 §0.6 的 NPU 算法」；
  不含 C 轴向量化 / 转置 / 索引累加器 / C4-C4b 结构 ✓。fp16/bf16 → fp32 为**无损上转**，输出为「选择」故 fp32 域结论与源 dtype 逐位一致，`atol=rtol=0` 依据成立 ✓。
- **host/op 层引用路径真实性 — pass**：设计声明「源仓无独立 `ops/pool/` 目录文件」（第 15–16 行）——本 Reviewer 实查 `ls /workspace/TileOPs-gpu/tileops/ops/`
  确认存在 `pool.py` 且 **`ops/pool` 目录不存在** ✓；NPU 侧 Stage 0 产物路径（wrapper / Op / 抽取件 / `.migration_meta.json`）**逐条存在** ✓。
- **多输出/变体函数的迁移范围 — pass**：`.migration_meta.json` 的 `extracted_functions = ["_adaptive_max_pool2d_with_indices_kernel"]`、
  `op_name = AdaptiveMaxPool2dIndicesFwdOp`，与设计声明「本次范围 = 双输出变体；单输出变体（第 13–74 行）由同族前作迁移」**一致**，无范围蔓延 ✓；
  manifest 双输出契约（扁平 int64 索引 / tie first hit / last NaN visited）与 §0.1 逐字一致 ✓。
- **设计与决策一致性 — pass（上一轮唯一扣分点已修复）**：§1.4/§3.1/§3.2/§3.3/§4/§5/§6 整体描述的是 §0.5/§0.6 决策后的 NPU 算法
  （C 轴向量轴、核内二轴转置、三路累加器、C4/C4b、`NUM_LOGICAL` 单元粒度、退化路径 `CH_deg`）；
  **上一轮 §3.3/§6.2 的 first-hit 次序偏差已修正**——`T.vcmp(row_f32, acc, gt_b, "gt")` 现**先于** `T.vmax(acc, row_f32, acc)`（§3.3 第 1112–1119 行、§6.2 第 1527–1530 行），
  与 §1.4 第 3 步、§0.6 R3、§1.6.1 N3 一致，文档内部矛盾消除 ✓。

### 1. API 可行性 — pass

- **API 存在性 — pass**：§3.2/§3.4 的每条 API 均在本仓 `docs/Tilelang.language/` 有**实读佐证**（本轮回查 dtype 表）：
  `T.transpose` Ascend 行含 **int32 √ / int64 √**（`创建操作/T.transpose.md` §2.2.1）、`T.vcast` `i32→i64 rint` / `f16·bf16→f32 rint` √、
  `T.vcmp` int32/fp16/fp32/bf16 √、`T.vselect` int32/int64/fp16/fp32/bf16 √、`T.vmax` int32/fp16/fp32 √（**无 bf16**）、
  `T.vmul` int32/fp32 √、`T.vbrc` int32 等 √、`T.arange`（1-D）√、`T.reduce`/`T.reduce_max` int32/fp16/fp32 √、`T.infinity` √、
  `T.vand`/`T.vor` **存在**（`逻辑操作/`）。设计**主动声明**「int32 `T.transpose` 本仓无 examples 先例并预置 fp32 承载回退（+ `h_in*w_in ≤ 2**24` 守卫）」
  （§9.1）——属**已识别风险的合理处置**，非隐性能力缺口 ✓。
- **API 与模式匹配 — pass**：Developer 模式下全程自动同步（§7.1/§7.2 无 `T.set_flag`/`T.wait_flag`），内存用 `T.alloc_ub`，计算用 v-prefix ✓。
- **v-prefix 优先 — pass**：全部为 `T.vmax/T.vcmp/T.vselect/T.vmul/T.vcast/T.vbrc` + `T.reduce_max/T.reduce_min` + `T.arange/T.reshape/T.transpose/T.copy`，无 `npuir_xxx` 旧式 API ✓。

### 2. 内存层级规划 — pass（上一轮阻塞级 2 已修复）

- **搬运路径完整 — pass**：§4.4 给出 `GM[x] → UB[row_src] →(transpose) UB[row_dst] →(vcast) UB[row_f32] → 向量计算 → UB[acc_t/idx64] →(T.copy) GM[out]/GM[indices]`，
  无越级路径、无 GM→L0 直连；C4/C4b 各自路径完整 ✓。
- **UB 预算 — pass（逐路径独立复算，含退化路径）**：
  - **一般网格**：按 §3.3 一般分支 alloc 清单逐 buffer 独立复算 spp-6x6/CH=64/W_in=56/out_W=6/fp16 =
    `row_src/row_dst` 7 168×2 + `row_f32/eq_f32` 14 336×2 + `eq_mask` 3 584 + `acc/out32/zeros/infbuf/nanbuf` 1 536×5
    - `max_idx/nan_idx/idx32/idx_t` 1 536×4 + `idx64` 3 072 + `cfinal` 384 + `cand/nanc/fvec/neg1` 256×4 + `zero_e` 256 + `acc16/acc_t` 768×2
    = **66 688 B ≈ 65.1 KB**（= §4.5），默认档 ×1.7 ≈ 111 KB、关闭档 ×1.12 ≈ 72 KB，**两档均 < 192 KB** ✓。
  - **退化路径 C4（`global-1x1`，plane 7×7=49，`CH_deg=min(2048,32)=32`）**：按 §3.3 C4 分支**实际 alloc 清单**逐 buffer 求和——
    12 个平面缓冲（`flat`2+`flat2`4+`mb`1+`mf32`4+`k_j`4+`cand`4+`candn`4+`neg1w`4+`bigk`4+`eqk_b`1+`gbrc`4+`k1w`4 = 40 B/elem）`32·49·40 = 62 720`
    - 归约/判决小缓冲 `5 088`（`mid/midn/midk/midn2` `(CH,H,1)` 4×224×4 = 3 584；`gmax/nmin/firstk/nank/zeros/idx32/out32/infbuf/nanbuf` 9×32×4 = 1 152；`cond` 32；`idx64` 256；`gout` 64）
    - `k1d`+`k1r` `392` = **68 200 B ≈ 66.6 KB**（= §4.5），×1.7 = 115 940 B ≈ **113.2 KB < 192 KB ✓**。**无漏 buffer**。
  - **退化路径 C4b（恒等 `(1,8,9,11)`，plane 99，`CH_deg=min(8,32)=8`）**：`stage`(2)+`idxb4`(4)+`idx64`(8) 平面缓冲 `8·99·18 = 14 256`
    - `idxbl`(4) `792` + `idxp4`(4) `396` + `idx1d`(4) `396` = **15 048 B ≈ 14.7 KB**（= §4.5），×1.7 = 25 582 B ≈ **25.0 KB < 192 KB ✓**。
    `CH_deg=32` 一般上界 = `32·99·18 + 792 + 396 + 396 = 57 816 B ≈ 56.5 KB`（×1.7 ≈ 96 KB ✓），与 §4.5/D-3 注一致。**上一轮漏计的 `idx1d/idxp4/idxb4/idxbl` 已计入** ✓。
  - **容量判据为带字节数不等式**（§4.5）：`degenerate_bytes(CH_deg, H_in, W_in, dtype) × inflate ≤ 192 KB`，与 §3.3 alloc 清单一一对应，`CH_deg=8` 仍超则**回退一般路径**（`(1,8,32,32)→1×1` 实算 ≈332 KB ⇒ 回退，与 `VP-2026-0146` 同向）✓。
  - 次要标注（非阻塞）：§4.5 的**说明性**举例 `(1,8,32,32)→1×1` 记 ≈340 268 B，本 Reviewer 复算 = **340 344 B**（差 76 B，0.02%）；
    两行**决策相关**数字（68 200 / 15 048）与设计**逐字节一致**，结论（回退）不因此改变。见「非阻塞改进建议」。
- L0C 容量 — **不适用**（纯 Vector，无 Cube/L0C 路径，§5.1/§7.4 已声明）。
- Buffer 层级标注 — pass：§4.3 逐 buffer 标注 UB，无 L1/L0A/L0B/L0C 使用 ✓。

### 3. Tiling 策略 — pass

- Block 划分明确 — pass：§5.2 给出 `CH = min(C, 64)`（一般）与 `CH_deg = min(C, 32)`（退化，容量驱动）及全部代表 shape 的 `num_cblk`/`NUM_LOGICAL` 表，理由齐备；
  `NUM_LOGICAL` 单元粒度按编译期分支取（一般 `N*out_H*num_cblk`、C4/C4b `N*num_cblk`）——本 Reviewer 复核：分叉必要性成立（C4 因 `out_H==1` 两种口径同值；
  **唯一分叉是 C4b**——若沿用通用式 9 而单元为整平面，`n = uid // num_cblk` 在 `N=1` 下越界），与 `ALG-pooling` 命中行第 ④ 条实测教训一致 ✓。
- 对齐约束 — pass：向量轴 C 为代表 shape 的 8 的整数倍（8/16/64/96/128/2048）；`W_in`（55/56/57）仅作转置后 row 长度、不作向量轴 ✓。
- 非整除处理 — pass：任务尾块 `if uid < NUM_LOGICAL`；通道尾块 `real_c = min(CH, C - cb*CH)`（**值与索引两路写回都用**，padded lane 不落 GM）；bin 重叠由整数 `adaptive_bin` 精确处理 ✓。
- 分核策略三要素 — pass：① 逻辑核数 = §5.2 表（本 Reviewer 逐 shape 复算一致）；② 物理核数**实查**并**独立复现**：`NPUUtils.get().get_aicore_num()` → **20**（910B3），纯 Vector 翻倍 **40** ✓；
  ③ 规模判定逐 workload 给出（≤ 物理核 → 无需适配；`global-1x1` 512 > 40 → persistent 40 核 × **13** 静态任务，`512 = 40×12 + 32`），
  `num_local` 为编译期常量 ✓、无空核保证 ✓、负载差 1 任务 ✓。核数与 block 取值自洽（`CH_deg=32` 与 512 单元、`num_local` 与分核表一致）✓。

### 4. 技术约束检测 — pass

三维 Kernel 不涉及（一维 `T.Kernel(num_kernels, is_npu=True)`，任务维度在核内解码）✓；GEMM 非整除 / L0C 溢出不适用 ✓；
动态 `T.serial` 边界：`MAX_BIN_H/MAX_BIN_W` 静态 + 守卫 ✓；Python `range` 不展开（`TRAP-tir-range-loop-not-unrolled`）→ 统一 `T.serial` ✓；
`target="npuir"` 无 `compile_flags` 形参 → 弃用源配置 ✓；int32 索引位宽守卫（§9.1）✓；`vcmp("ne")` ordered-NE 陷阱 → 改 `eq` 取反 ✓；
in-place `T.vselect` loop-carried 陷阱 → 只用独立 scratch + 原地 `vmax`/`vmul` ✓（`TRAP-vselect-inplace-carried-state` 绕法）。

### 5. 循环与同步 — pass

§6.1 明确每类循环的 API 与理由（`T.Kernel` 任务并行 / `T.serial(num_local)` persistent / `kh`、`o_w`、`kw` 调度循环 / C 轴向量维 / 退化归约）；
§6.3 说明不使用 `T.Pipelined` 的理由（单阶段、`kh` 次数少、Stage 4 候选）；§6.4 尾块四类齐备；
§7 为 Developer 自动同步、无手动同步点，与单引擎单阶段事实一致；`T.arange` 1-D 形态 + `T.reshape`（元数据级）规避跨流水可见性风险（§7.2 注，含 `T.pipe_barrier` 局部加固预案）✓；
**上一轮 §6.1 `o_w` 行与 §1.6.2 枚举未对齐的问题已消除**（§1.6.2 新增 #18 覆盖 `for ow in T.serial(out_W)`）✓。

### 6. 验证方案 — pass

- Golden 函数 — pass：§8.1 含可运行 PyTorch 参考实现草案，以 §0.1 语义为唯一依据、独立于 §0.6（见维度 0）✓。
- L0 测试计划 — pass：12 条用例覆盖一般网格 / bf16+通道尾块 / 非整除重叠 / 部分 `None` / C4 persistent 512 单元 / C4b 恒等 / 扩张 bin / 3D CHW rank 保持 /
  **fp16 语义门禁（L0-9）** / **bf16 语义门禁（L0-9b，本轮新增）** / factory guard / indices 形状 dtype 断言；shape 与 dtype 具体，
  并与源 `test_pool.py` 的 4 组 parametrize（`smoke-spp-6x6-fp16` `(2,64,56,56)→(6,6)`、`smoke-asymmetric-bf16` `(1,96,28,30)→(7,5)`、
  `full-nondiv-fp16` `(1,128,55,57)→(7,7)`、`full-partial-none-fp16` `(1,16,11,13)→(5,None)`）及 manifest 3 组 workload 对应（label 逐字核对一致）✓。
- 精度标准 — pass：值 `atol=rtol=0, equal_nan=True`（max 是选择非累加，依据源测试注释「Max is exact (no accumulation) and the index comparison is exact too」
  与 `test.check(op, ..., atol=0, rtol=0)`），索引 `atol=rtol=0` 精确比对（`VP-2026-0131`）✓。
- **上一轮建议 7（L0 语义门禁未覆盖 bf16 路径）已修复**：§8.3 新增 **L0-9b**，与 L0-9 同注入集合（入窗 NaN / 两个 NaN / 全并列 / `{+inf,-inf}` / 全 `-inf`），
  断言值 `equal_nan=True` 全量比对 + 索引精确比对 + `cfinal` 机制观测 ✓。
  附加的「机制级证据要求」（不得仅以「值输出 NaN」为通过依据）与「first-hit 实现级检查（`arange(12)→(2,2)` 的 `bin(1,1)` 须为 11）」是**加分项** ✓。

### 7. 完整性与一致性 — warn

- 章节齐全 — pass：§0（0.1–0.7 全）→ §1–§10 模板章节齐全；
  且 **§1.6 含 1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四子节** ✓。
- 无占位符 — pass：本 Reviewer 对全文做字面量扫描（模板占位符形式、常见未完成标记、空章节留白），**均未命中** ✓。
- 同类引用真实 — pass：§3.5.3 的 14 条 `examples/` 引用**逐条实查存在**（同族前作 max/avg DESIGN.md 与实现、argmax `_argreduce_kernel.py`、
  lerp_tensor、logsumexp、Stage 0 抽取件/wrapper/Op、test/bench/manifest、`.migration_meta.json`）✓。
- **内部一致 — warn（唯一扣分点，建议级）**：**上一轮三处内部矛盾均已修复**——① §3.3/§6.2 的 first-hit 次序与 §1.4/§0.6/§1.6.1 现已一致；
  ② §0.6 R3 承接行的 `vmin` 笔误改为 `vmax`；③ §1.6.2 的 #16 拆为 #16（trace 期静态界）+ #17（运行时标量 bin 边界），与 §3.1/§3.3 一致。
  残留一处**极轻微**不自洽：§9.3「极小 shape」行写「`C<8`（`CH=8` + `real_c<8` 处理）」，而 §5.2 定义为 `CH = min(C, 64)`（`C<8` 时 `CH=C<8`）；
  两者对 `C<8` 的 `CH` 取值表述不一致（且 §5.2 注释「8 的整数倍，且 ≥ 8」在 `C<8` 时不成立）。manifest 与 test 的 shape 全部 `C ≥ 8`，**不影响任何实际路径**，故为建议级。
- 其余（§3.2 API 映射 ↔ §4 内存规划 ↔ §5 Tiling ↔ §6 循环 ↔ §1.6.2 枚举）三处描述相互自洽 ✓。

### 8. 算法优化分析 — pass（上一轮阻塞级 1 与本维度两处建议已修复）

- **§1.6 章节存在 — pass**：1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四子节齐全 ✓。
- **算法调研存在性与完整性 — pass**：R1 候选表含**基线候选 #0**（源标量状态机）与 #0'（基线向量化，**选定**），并逐项评估 `ALG-pooling` 命中行全部条目
  （常数权重卷积 / sum-pool×常数 / 滑窗流式 / 窗口重叠重复读 / C 轴向量轴 / 几何特化模式集）——本 Reviewer 实读 `algorithm-candidates.md` 的 `ALG-pooling` 行确认文本与 §1.6.0 引用**逐字相符**；
  R2 有明确结论「**有**在线变体，且基线本身已是单遍」并附结构判据 + **源码证据**（§0.4 #8 值/索引同轮、#11 running 比较状态机）；
  R3 覆盖**四口径**（FLOPs / GM 访存 Bytes / 扫描遍数 / 缓冲峰值）且**基线在表**、口径写清；R4 逐候选六维亲和评估；
  调研范围声明（含「互联网检索未使用」的理由）+ 下游约束声明齐备 ✓。`ALG-reduction` / argmax 族索引输出候选已纳入评估并部分采纳（C4）✓。
- **调研结论正确性（独立复核） — pass**：
  - **等价性独立推演成立**：本 Reviewer 独立验证 §0.1 全部语义（torch 实测，见机械复核），并独立推演 N1（`eq` 为 `isnan` 的 IEEE 补集，乘性 0 吸收 ≡ 存在 NaN）、
    N2（访问序扁平**严格递增** ⇒ 末次覆写 = NaN 位置集合扁平最大值）、N3（严格 `>` 记录按扁平严格递增出现 ⇒ 末条记录 = 破纪录集合扁平最大值 =
    首次达到窗口最终最大值的元素；全 `-inf` 窗口由初值 `f0` 覆盖且任何记录下标 ≥ `f0` 故初值不被污染；`{+inf,-inf}` 两序均得 `+inf` 位置；NaN 窗口下 `max_idx` 无关性）——
    **推演与 §1.6.1 一致，无矛盾**；C4（`out=1×1` ⇒ bin = 整平面，`min{k:v_k=max}` = first-hit、`max{k:isnan}` = last-NaN、eq 掩码 `reduce_min` 独立判 has_nan，
    与 `T.reduce_max` 的 NaN 约定无关）与 C4b（单位 bin ⇒ 恒等 + 常量索引模式）亦成立 ✓。容差口径正确（输出为**选择**，无算术，`0 ULP` 判据与 §8.2 `atol=rtol=0` 一致）✓。
  - **复杂度算术独立复算一致**：spp-6x6 → `scan_H = scan_W = 60`（j=1,2,4,5 处 `(j*56)%6≠0`）✓、FLOPs `2*128*60*60 = 921 600` = manifest `roofline.flops` ✓；
    Bytes `(2*128*56*56 + 2*128*6*6)*2 + 2*128*6*6*8 = 1 697 792` = manifest `roofline.bytes` = 源 `_adaptive_pool2d_roofline` 同式 ✓；
    `global-1x1` 1 769 472 ✓、`nondiv-7x7` 865 280 ✓（`scan_H=61/scan_W=63`）；候选 2 运算量 `H_in·scan_W + scan_H·out_W = 3360+360 = 3720 > 3600` ✓ —— **中间缓冲未漏计 GM 往返**（本设计无 GM 中间往返，容量项 0 已显式声明）。
  - **负向断言主动复核 — pass**：候选 1/3/5（常数权重卷积 / 前缀最大值差分 / sum-pool×常数）属**数学事实类**（`max` 非线性、无逆元），独立推演认同（`x=[5,1,3]`、区间 `[1,3)`：`P[3]-P[1]=0 ≠ max=3`）✓；
    候选 7（两遍式）访存翻倍 ✓；候选 2（可分离 2D）弃选依据成立（运算量更大 3720 > 3600 + 额外缓冲 `R(H_in·out_W·CH)` = `56*6*64*4 = 86 KB` 叠加一般路径 65 KB 后 ×1.7 超 192 KB，属**容量级决定性理由**，非纸面拍板）✓；
    候选 9 容量否决（`scan_H·scan_W` 窗 × CH=64 ≈ 256 KB > 192 KB）✓；候选 0 的「标量占比 98.3%」与 `PL-1.2-caxis-accum` 实读一致 ✓。
    **R1 #8「复合键编码」— 上一轮建议 4 已修复且本轮重点复核成立**：v0 的「信息论不可行」前提（值序需全 32 位）已被替换为 **「工程性弃选（非信息论）」**——
    新前提独立核实**正确**：fp16→fp32 低 **13** 位尾数恒零、bf16→fp32 低 **16** 位恒零 ⇒ 平面索引 `< 2^13`（fp16）/`< 2^16`（bf16）**可**塞入低位，
    本算子 manifest 与测试 shape 的平面量级 49–3136 **全落在可塞区间内**（< 8192）；新理由三条（① 收益有限——每元素仅省 3 条 op，本算子带宽受限、发射非主导；
    ② 代价与风险——`T.vand`/`T.vor` 位运算构包（**本 Reviewer 确认该两 API 真实存在**，故属代价而非能力缺口）+ 反转并列方向 + NaN/±inf 位型共存证明 + 分段守卫；
    ③ 与 argmax 族已验证形态不一致）**成立且自洽**，「不采纳」结论可辩护，R3 标 N/A 合理 ✓。
  - **R3 并列候选裁决路径 — pass**：候选 2 有容量级决定性依据；布局侧分歧（`CH=64` vs `CH=8`）已移交 §1.6.3 D-3 三件套裁决 ✓。
  - **互联网来源**：声明未使用（附范围理由）✓，无需来源记录。
  - 次要标注（非阻塞）：§1.6.1「守卫非空性证据」记 9779 个可检出元素，本 Reviewer 按守卫**实际掩码** `(oki != seed) & ~isnan(bval)` 复算 = **9769**（差 10；脚本本身**不打印**该计数）。
    守卫非空性（≫0）与 `N3-ord` 全 PASS 均成立，不影响结论。见「非阻塞改进建议」。
- **调研→设计一致性 — pass**：选定算法 #0'（单遍在线状态机 + fp32 比较域 + C 轴向量化 + 显式向量 NaN 追踪 + 向量 first-hit/last-NaN + 核内融合转置）
  实际驱动 §1.4 算法描述（第 1–3 步与 §0.6 R1–R5 逐条对应）、§3.1 公式拆解（以 §1.6.1 优化后公式为输入）、§6 循环结构（`kh`/`kw`/`ow` 调度 + C 轴向量维）✓；
  迁移任务 §0.5/§0.6 处置与调研选族一致（R2 = 值/NaN 向量化、R3 = 索引向量化、R4 = C 轴布局、R5 = 候选 #4 + #9），
  源算法 §0.4 的 14 项优化手段**逐项有承接说明、无静默丢弃** ✓。
- **数学等价优化完整性 — pass**：§1.6.1 采纳 5 项（N1/N2/N3/C4/C4b）+ 稳健性对照 N3b + **实现次序守卫 N3-ord**，每项四要素齐备
  （原式 → 优化后公式 → 等价性论证 → 收益量化，如 C4「1 次归约 + 1 次掩码归约 ≈ 15×」）；并显式给出三类**无优化空间**结论（无除法/归一化、无公共子表达式冗余、无高代价算子降代换）✓。
- **等价性论证正确性 — pass**：5 项独立推演均成立（见上）；容差口径正确（选择型算子，0 ULP）✓。
  **实现次序（上一轮阻塞级 1）独立复核 — pass**：§3.3 一般分支的 op 次序为 `vcmp(gt)`（与**更新前** `acc` 比较）→ `vselect` → `vmax(max_idx)` → **再** `vmax(acc)` → `vmul(h_flag)` → `vcmp(isnan)` → `vselect` → `vmax(nan_idx)`；
  §6.2 同形。本 Reviewer 独立推演：若先更新 `acc` 则判据退化为 `v > max(a,v)` **恒假** ⇒ `cand ≡ −1` ⇒ `max_idx ≡ f0`（反例 `arange(12)→(2,2)` 的 `bin(1,1)` 得 6 而非 11）；
  **当前次序正确**，且 `verify_equiv.py` 的 `N3-ord` 以双向断言（正确次序复现基线 + 值优先次序在可检出元素上必被检出）+ 硬编码反证用例给出机器守卫，
  **其建模（`_running_first_idx` 的 `value_first` 分支）与 §3.3 op 次序一致**（本 Reviewer 逐行比对脚本与伪代码）✓。该修复同时消除了 §1.4↔§3.3 的文档内部矛盾。
- **向量化替代完整性 — pass（上一轮建议 3 已修复）**：§1.6.2 现含 **20 行（#0–#19）**，覆盖初始化 / 取最大值 / cast / NaN 掩码 / first-hit / last-NaN / 扁平下标广播 /
  NaN 判决 / 输出 cast / 退化归约与 arange 常量模式 / 布局重排 / **`kh`(#13) / `kw`(#14) / 任务解码(#15) / 静态界(#16) / 运行时 bin 算术(#17) / `o_w` 调度循环(#18) / host 侧(#19)**；
  与 §3.3 伪代码、§6.1 循环结构**交叉核对无遗漏**（`T.serial(num_local)` 由 #15 覆盖、`kh`/`kw`/`ow` 分别由 #13/#14/#18 覆盖），
  「保留 7 处」枚举与 §3.3/§6 的循环集合一一对应 ✓；替代方案**全部**有 `docs/` 佐证 ✓。
  注：N1 的 `h_flag` 累加器为独立机制保留但不在输出判决关键路径（见「非阻塞改进建议」）。
- **不可替代理由充分性 — pass**：保留项理由具体到本算子（tile 级循环携带依赖 + 依赖运行时 `o_h`/`ow` 的动态 bin 边界 / block 级任务映射 / host 元数据 /
  输出列 tile 级调度），无「实现简单」式理由；无「有向量 API 可替代却未替代」的情形 ✓。
- **替代 API 可行性 — pass**：与维度 1 同口径，逐 API 实读 docs 佐证，v-prefix 优先 ✓。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按窗口/池化类候选清单枚举（`I/O 原生布局 + 最内连续轴`（#1）、`核内重排布局 + 高整除性轴`（#2）、
  host permute（#3）、Cube/implicit-GEMM（#4）、窗口 tile 驻留 + reduce argmin（#5）），逐候选有量化评分（整除性 / 尾 lane 浪费 / 累加链形态 / repack 代价 / UB 影响）；
  **弃选均含量化理由**（#3 host permute `106–147 µs`（`PL-1.3`）、#5 容量 256 KB > 192 KB、#1 段长 9–10 非 8 倍数 + `T.reduce_max` 不支持 bf16、
  #4 数学不等价且 `out_W=6/7 < 16` 违 Cube 分形下限）；**未把 I/O 契约（NCHW）当核内布局依据**（显式区分契约 vs 设计变量）；
  迁移任务已对照源码隐式轴选择（§1.6.3 声明「源的并行轴选择是输入而非结论」）✓；**弃选前提自查**：本 Reviewer 打开 `T.transpose.md` 核对（核内二轴交换、dtype 含 int32/int64），
  与设计陈述一致，无「核内重排 = GM 级 transpose」类前提错误 ✓；布局族引用 `PL-1.1/1.2/1.3/1.4` 均为 fresh ✓。
- **实验裁决模式完整性 — pass**：判定裕度落在「窄向量 op 每-op 代价」未实测外推区间，设计给出三件套——主选 `CH=64`（依据 fresh 的 `PL-1.4` 实测
  「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs」，本 Reviewer 实读 `layout.md` 一致）、备选 `CH=min(C,8)`（含结构/分核三要素按 192 单元重算，达可实现深度）、
  裁决计划（代表 shape `spp-6x6`/`global-1x1`、指标 msprof Task Duration、未知常数反解、阈值「备选 < 主选 ×0.9 才翻转」、回写路径 + shape 分派评估）✓；
  并明确退化路径 `CH_deg=32` 由**容量判据**决定、独立于该性能 A/B ✓。
- **布局决策与实现一致 — pass**：选定「核内逐行 NHWC（C 最内）+ 向量轴 C + 核内二轴 transpose」已同步至 §3.3（buffer 形状与内层向量维 = C）、
  §4.3/§4.5（按重排布局计算形状与预算）、§6.1（累加内层向量维 = C）三方一致 ✓。
- **与下游章节一致 — pass（上一轮阻塞级 1 已修复）**：§3.1 以 §1.6.1 优化后公式为输入 ✓；§6 循环结构与 §1.6.2 结论一致（无 §1.6.2 未论证的逐元素标量循环）✓；
  **§3.3/§6.2 的实现次序现与 §1.6.1 N3 的等价性前提一致**（`gt` 与更新前累加器比较），并被 `N3-ord` 机器化，verification–implementation gap 已闭合 ✓。

## 非阻塞改进建议

> 以下均为**建议级**（不影响 `结论: 通过`，不阻塞 Stage 3；供 Stage 3 落地或后续修订参考）。

1. **【§7 warn】`C<8` 时 `CH` 取值表述统一**：§5.2 定义 `CH = min(C, 64)`（`C<8` ⇒ `CH=C<8`）而 §9.3 写「`C<8`（`CH=8` + `real_c<8` 处理）」。
   建议二选一并全文统一：① 若实际取 `CH = max(8, min(C, 64))`（向量宽 8、`real_c<8` 补零），则修正 §5.2 公式与注释「8 的整数倍，且 ≥ 8」（`CH_deg` 同理）；
   ② 若保持 `min(C,64)`，则把 §9.3 改为「`C<8` ⇒ `CH=C`，向量 lane 数 < 8，`real_c=C`」。manifest/test 全部 shape `C ≥ 8`，仅为文档自洽。
2. **【§1.6.2 / §1.6.1】`h_flag` 累加器为冗余机制**：最终值与索引的 NaN 判决统一取 `cfinal = (nan_idx >= 0)`（§5.4），`h_flag`（`(out_W,CH)` fp32 + 每元素 1 条 `vmul`）不在输出关键路径上。
   设计已声明其为 N1 的独立机制校核；建议在 §1.6.1 N1 的「收益量化」中注明「该 op 不承载输出判决（判决由 N2 的 `nan_idx>=0` 承担）」，或在 Stage 3 确认机制校核可由测试承担后删去该 buffer/op（省 ≈1.5 KB UB 与每元素 1 op）。
3. **【§1.6.1 / §4.5】两处说明性数字的精确化**：① §1.6.1「守卫非空性证据」记 **9779** 个可检出元素，而 `verify_equiv.py` 守卫实际掩码 `(oki != seed) & ~isnan(bval)` 复算为 **9769**（脚本不打印该计数）；
   ② §4.5 说明性举例 `(1,8,32,32)→1×1` 记 `≈340 268 B`，本 Reviewer 按同一清单复算为 **340 344 B**。
   两处均为**说明性**数字（非决策依据，两行决策相关数字 68 200 / 15 048 已逐字节一致），建议更正或将守卫非空计数加入脚本输出以便机械复现。
