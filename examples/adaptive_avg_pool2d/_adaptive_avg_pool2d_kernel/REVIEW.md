# _adaptive_avg_pool2d_kernel 设计文档检视报告（revision_index = 2）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md（修订版 v2，含 §11 修订记录）
- 任务类型: 迁移（TileOPs harness 逐函数，project=`adaptive_avg_pool2d`，op=`_adaptive_avg_pool2d_kernel`）
- 源算子: `/workspace/TileOPs-gpu/tileops/ops/pool.py`（`AdaptiveAvgPool2dFwdOp` / `_AdaptivePool2dFwdOpBase`）
  源 kernel: `/workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_avg_pool2d.py`；源 helper: `/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`
- 上一版: `history_version/design_v1.md`（检视 `history_version/review_v1.md`：**维度 0/8 均过，维度 3 fail**，1 阻塞 + 1 建议）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS`（脚本存在；命令 `/workspace/venv/bin/python examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/verify_equiv.py`，**exit=0**）。
  逐例输出与 §1.6.1 内嵌结果表**逐行一致**（45 行，case 名 / kind / dtype / max_abs_diff / viol / bitdiff / max_ulp / status 全同），末行 `[EQUIV] ALL PASS — 45 cases, 0 failures` 与内嵌表末行**逐字一致**，无 `EQUIV_FAIL`。
  判据分档 `_ULP_TOL={C4:0.0, C4b:0.0, C1:1.0}` 与「已采纳 3 项（C1 容差内 / C4·C4b 逐位）」严格匹配；C1 spp fp16 实测 `max_abs_diff=1.221e-04, bitdiff=15, max_ulp=1.00, viol=0`，inf-nan 用例 `both_nan` 归一化后 `viol=0`。
  （v1 检视遗留的「下一版 design_v1.md」= 本版 v2 的输入版本；本轮**未改动 `verify_equiv.py`**，45/45 稳定复现。）
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md`（**exit=0**）JSON checks 摘要：
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan"。→ **人工复核（机械跳过）**：§4.5 逐项复算 `row_src 64×56×2=7168` + `row_dst 56×64×2=7168` + `row_f32 56×64×4=14336` + `acc 6×64×4=1536` + `acc_t 64×6×2=768` = **40976 B**，与文档声明 40 976 B 一致；C4 行（`CH=min(C,64)=64`）`64×49×2=6272` + `64×49×4=12544` + `64×1×4=256` = **19072 B ≈ 19 KB**，与文档一致；C4b `64×99×2=12672 B ≈ 12.7 KB` ✓。默认档膨胀 ×1.7 → spp（默认档 ≈70 KB）/ C4 ≈33 KB；关闭档 ×1.12 → spp ≈46 KB / C4 ≈21 KB，**两档均 < 192 KB（`CONST-capacity-910B2C`）**，容量结论成立。
  - `l0c_budget`: **skip** — "no block_M x block_N values parsed"。→ **人工复核**：纯 Vector 设计，无 Cube 路径，**L0C 不适用（n/a 成立）**。
  - `core_split`: **skip** — "insufficient parsed dims {}; declared logical cores = None"（解析器按 `M/bm` 命名匹配，本设计任务维为 `(n,out_H,c_block)`，无法机械解析）。→ **人工复核（机械跳过）**：§5.5 ① `NUM_LOGICAL` 按**编译期分支的单元粒度**取（一般 `N*out_H*num_cblk`；C4/C4b 整平面单元 `N*num_cblk`），我逐行复算 §5.2 表全部 10 行：smoke 2 / resnet 256 / spp 24 / nondiv 14 / `(2,64,56,56)→6×6` 12 / `(1,96,28,30)→7×5` 14 / `(1,128,55,57)→7×7` 14 / `(1,16,11,13)→(None,5)` 11 / `(1,16,10,10)→4` 4 / **`(1,8,9,11)→None` 1（C4b 整平面单元，非通用式 9）** —— **与表逐行一致**；② 物理核数**本检视会话独立实查** `NPUUtils.get().get_aicore_num()` = **20**（`910B3`），纯 Vector ×2 = 40，与 §5.5 记录一致（未误用 `CONST-aicore-910B2C`=24）；③ 规模判定与方案：`NUM_LOGICAL ≤ 40` 给「无需适配」，resnet（256）给 persistent、`num_local=ceil(256/40)=7`（Python int 静态值）——算术自洽。**三要素齐全、数值正确，v1 阻塞（C4b 路径三要素不自洽）已收敛**（见维度 3）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read `pool.py`（1199 行）、`kernels/pool/adaptive_avg_pool2d.py`（110 行）、`kernels/pool/common.py`（192 行）全文，与 §0.1–§0.7 逐项核对。§11.3 声明 §0 本轮未改动，复核确认修订未破坏 §0。

- **语义理解正确性（§0.1）— pass**：
  - bin 划分公式 ↔ 源码 `common.py:108` `adaptive_bin` 返回 `((o*size_in)//size_out, ((o+1)*size_in+size_out-1)//size_out)`；后者 `((o+1)·in+out−1)//out == ceil((o+1)·in/out)`，**逐字一致**。
  - 输出尺寸规则 ↔ `pool.py:918-948` `_normalize_output_size`（`None→(None,None)`、`bool` 拒绝、`<=0` 拒绝）+ `:1012-1019` `_resolve_out_dims`（`None→输入尺寸`）——§0.1 陈述全部对上。
  - dtype 语义 ↔ `pool.py:951-956` `_validate_adaptive_pool_input_dtypes` 仅 `{float16, bfloat16}`（**显式拒绝 fp32，§0.1 正确**）；`kernels/pool/adaptive_avg_pool2d.py:22` `accum_dtype="float"`；`:55-58` 每元素 `T.cast(x[...], accum_dtype)`；`:61-64` 末尾单次 `T.cast(sum_val/bin_size, dtype)` —— §0.1「fp32 累加 + 单次输出舍入」**源码级确认**。
  - 累加顺序 ↔ `:52-53` `for kh: for kw:`；§0.1「`kh` 外层、`kw` 内层」**正确**。
  - 边界语义 ↔ `common.py:105-108` docstring「Bins are never empty…」；`pool.py:968-971` 重叠注释 + `scan_h/scan_w`；`adaptive_avg_pool2d.py:60` 整数差 `bin_size` —— 三点**全部有源码出处**。
  - 3D CHW ↔ `pool.py:1042-1044` `unsqueeze(0)` / `:1086-1089` `squeeze(0)`（仅 view）**正确**。
- **I/O 与输出 shape（§0.2）— pass**：`x:(N,C,H_in,W_in)` fp16/bf16、`out:(N,C,out_H,out_W)` same_as_input、`out_idx=[1]`（`adaptive_avg_pool2d.py:27`）——**与源码签名一致**；输出同布局（NCHW），非转置布局。
- **算法解读完整性（§0.3）— pass**：六步分解逐条落到源码行（步骤 1↔`:23-25`；2↔`:34-35`；3↔`:36-43`；4↔`:45-46`；5↔`:48-58`；6↔`:60-64`）；数据流图「无 SMEM/L1 中转、逐元素 GM gather」与源码（全文无 `alloc_shared`/`T.copy`/归约原语）一致；host 侧逻辑全部真实存在（`pool.py:1057,1059-1071`、`kernels:12,27,97`）。**无臆造、无遗漏计算语句**。
- **优化手段覆盖完整性（§0.4）— pass**：9 项逐项有源码依据（#1↔`:50-54`；#2↔`:22,55-58,61-63`；#3↔`:60`；#4↔`:34-36`；#5↔`:35`；#6↔`:48` `T.alloc_var`；#7↔`:55-57`；#8↔`kernels:12`+`pool.py:1059-1071`；#9↔`common.py:105`+`max_adaptive_bin_extent`）。反向核对：源文件确无 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async —— §0.5「**舍弃项：无**」**成立**；9 项无一项被静默丢弃。
- **耦合性判定合理性（§0.5）— pass**：抽检 #4（→等价替换）、#5（thread lane→重新设计）、#6（寄存器标量累加→重新设计）、#7（GM gather→重新设计）、#1/#2/#8/#9（保留）——与 `migration-analysis.md` §5.3 映射表及 §5.4 判定规则 1 不矛盾；每项附依据，#1「语言级约束（非硬件）」与源码注释（`:50-51`）自洽。
- **NPU 重设计可行性（§0.6）— pass**：R1–R4 均含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1「任务划分不改求和集合与顺序」成立；R2「C 轴为独立规约轴 ⇒ 置于 SIMD lane 不改任一 `c` 的累加顺序」与 §0.1 规约语义（无跨通道归约）吻合；R3「转置为双射、纯重排无算术」成立；R4 引用 `migration-analysis.md` §6.2「算法替代」行。新算法满足 NPU 约束（一维 `T.Kernel`、`T.transpose` 仅二轴、C 轴整除、UB 充裕）。§0.6 R1 已加退化路径粒度交叉引用（`num_logical = N × ceil(C/CH)`，见 §5.5 ① / §1.6.1 C4/C4b）。
- **设计与决策一致性（§1–§7 ↔ §0.5/§0.6）— pass**：§1.4/§2.3/§3.2/§3.3/§4.4/§5.2/§5.5/§6.1 一致描述「一维 Kernel + `(n,o_h,c_block)` 单元 + C 轴向量累加 + 核内二轴转置 + C4/C4b 退化特化」；§3.5.2 差异表逐项标注 §0.6 R1/R2/R3/R4。**v1 阻塞所指的 §3.3 C4b 解码已收敛**，不再构成跨节脱节。
- **golden 独立性（§8.1）— pass**：`golden_adaptive_avg_pool2d` 仅调用 `F.adaptive_avg_pool2d`，与源仓基准 `examples/TileOPs/tests/ops/test_pool.py:62-65` 同源；**不含** §0.6 的 C 轴向量化/转置/C4-C4b 分支 ⇒ 独立验证而非自证。

> 结论：源算子语义、算法步骤、优化手段识别、耦合性判定、重设计可行性与 golden 独立性**均经源码级核对通过**，未发现语义理解偏差。

### 1. API 可行性 — pass

- §3.2/§3.4 列出的 9 个 API **全部有本地佐证**，本会话逐条打开文档核对签名与 dtype/限制：
  - `T.Kernel`（`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3，NPU ≤2 维，设计用一维合法）；
  - `T.alloc_ub`（`docs/.../内存操作/T.alloc_ub.md`）；
  - `T.clear`：`docs/Tilelang.language/内存操作/T.clear.md` §2.2.1 dtype 表 **fp16/fp32/bf16 全 √**、§2.2.2「支持 1~5 维 tensor」、§2.4 示例 `T.clear(A_VEC)`、转 `hivm::VBrcOp`——与设计「整 buffer 清零」一致 ✓；
  - `T.copy`（§2.1/§2.2.2，支持切片）；
  - `T.transpose`（`创建操作/T.transpose.md` §2.3：底层 `hivm.hir.vtranspose` **只支持转置两个轴**，dtype 含 fp16/fp32/bf16；§3 示例含 `permutation=[1,0]`）——设计只做 `[1,0]` 二轴交换 ✓；
  - `T.vcast`（`数据类型转换操作/T.vcast.md` §2.2.1：`f16→f32 rint` / `bf16→f32 rint` / `f32→f16,bf16 rint` **全在表内**）✓；
  - `T.vadd`（§2.2.1：fp16/fp32 ✓、**bf16 ×**；§2.3 `dst` 可原地）✓；
  - `T.vmul`（§2.2.1：fp16/fp32 ✓、bf16 ×；§2.3「**仅支持 `src2` 为标量**」）✓；
  - `T.reduce_sum`（§2.2.1：fp16/fp32 ✓、**bf16 ×**，`clear` 默认 True）✓。
- **模式匹配**：Developer 模式 + 编译器自动同步，§7.2 无手动 `set_flag/wait_flag` —— 与 §2.1 一致。
- **v-prefix 优先**：全部计算用 `T.vcast/T.vadd/T.vmul/T.reduce_sum` + `T.clear`/`T.transpose`/`T.copy`，无 `npuir_xxx` 旧式 API ✓。
- §3.4 dtype 提醒「`T.vadd/T.vsub/T.vmul/T.reduce_sum` 不含 bf16，故 bf16 须经 `T.vcast` 上转」与各文档 §2.2.1 表**逐条吻合** ✓。

### 2. 内存层级规划 — pass

- §4.4 搬运路径完整：`GM[x] → UB[row_src] → UB[row_dst] → UB[row_f32] → UB[acc] → UB[acc_t] → GM[out]`；C4/C4b 路径亦 GM↔UB 两级；**无 GM→L0 直连等越级路径**，无 Cube/L1/L0C 路径（纯 Vector，「L0C 不适用」成立）。
- UB 预算 §4.5 复算 = **40976 B（spp CH=64）/ 19072 B（C4 resnet CH=64）**，均 ≪ 192 KB；`design_calc_check.py` 的 `ub_budget` 为 skip，已人工逐项复算（见「机械复核」段），**加和与声明一致、容量结论成立**；两档膨胀（×1.7 / ×1.12）亦在容量内。
- 每个 buffer 均标注存储层级（§4.3 表全部 UB；`flat_ub/flat_f32/gacc/stage_ub` 亦为 UB）✓。
- L0C：无 Cube 计算，`l0c_budget` skip → n/a ✓。

### 3. Tiling 策略 — pass

> **v1 阻塞的复核对象**：§3.3 C4b 恒等路径的「核数 / 单元粒度 / 解码」三要素不自洽。**本轮独立复核确认已真正修复**。

- **Block 划分明确（§5.2）— pass**：`CH=min(C,64)`（8 的整数倍）、`num_cblk=ceil(C/CH)`；`NUM_LOGICAL` 按**编译期分支的单元粒度**取值（一般 `N*out_H*num_cblk`；C4 全局 / C4b 恒等整平面单元 `N*num_cblk`），附 10 组代表性 shape 分块表。
- **对齐约束 — pass**：向量轴 C 恒为 8 的整数倍（8/16/64/96/128/2048）；`W_in`（55/56/57）仅作转置后 row 长度，不作向量轴，不要求整除 ✓。
- **非整除处理 — pass**：任务尾块用 persistent `T.serial` + `if uid < NUM_LOGICAL` 守卫（静态上界 `num_local`）；bin 非整除（重叠）由整数 `adaptive_bin` 精确处理，无 GEMM 式整除要求 ✓。
- **分核三要素齐全且数值正确（§5.5）— pass**：
  - **① 逻辑核数计算**：单元粒度 = `NUM_LOGICAL` 的**唯一定义源**，§3.3 顶层用编译期分支一次性定义（一般 / C4 / C4b 三分支），§5.2/§5.5/§6.2 一律引用。逐行复算 §5.2 表 10 行全部通过（含 `(1,8,9,11)→None` = **1**，非通用式 9）。
  - **② 物理核数依据（实查）**：**本检视会话独立实查** `NPUUtils.get().get_aicore_num()` = **20**（`910B3`），纯 Vector ×2 = 40，与 §5.5 记录一致；未误用 pattern-library 的 `CONST-aicore-910B2C`=24。
  - **③ 规模判定与分核方案**：`NUM_LOGICAL ≤ 40` → 「无需适配」（smoke/多数测试）；resnet（256）→ **persistent**（固定 40 核、`num_local=7` Python int 静态值、`uid=t*num_kernels+cid`、守卫 `if uid<NUM_LOGICAL`），负载均衡（16 核 7 任务 / 24 核 6 任务，最大差 1）。**核内串行边界为静态值**（满足 `core-split-strategy.md` §1 要素③硬要求）。
- **v1 阻塞 — C4b 单元粒度/核数/解码自洽性 — 已修复（独立复核）**：
  - **§3.3**：`NUM_LOGICAL = N*num_cblk if (IS_IDENTITY or IS_GLOBAL) else N*out_H*num_cblk`；C4b 分支解码 `n = uid // num_cblk`、`cb = uid % num_cblk`，且注释「与 NUM_LOGICAL 同粒度：n∈[0,N)，无越界」。
  - **越界风险消解推演**：对 `(1,8,9,11)` 恒等 case，`N=1`、`CH=8`、`num_cblk=ceil(8/8)=1` → `NUM_LOGICAL = N*num_cblk = 1`；`num_kernels = min(1,40) = 1`；`num_local = ceil(1/1) = 1`。唯一任务 `t=0, cid=0 → uid=0 < 1` → `n = 0//1 = 0 ∈ [0,1)`、`cb = 0`，**无 `n=0..8` 越界**（v1 阻塞形态消除）。而通用式 `N*out_H*num_cblk` 会给出 9，若与整平面单元 + `n=uid//num_cblk` 混用即越界——本版已排除该混用。
  - **全文唯一自洽（逐处核对）**：§3.3（行 581-586 定义 + 596-602 解码）/ §4.3（`stage_ub=(CH,H_in*W_in)` 整平面单元，行 733）/ §4.4（行 743）/ §5.2（代码 798-799 + 表末行 821 + 分叉说明 823）/ §5.3（行 831 守卫符号）/ §5.4（行 836 退化路径粒度）/ §5.5（① 行 847 + ③ 表 869 + 通用规则 872-875）/ §6.1（行 891）/ §6.2（行 901-906）/ §6.4（行 926）/ §9.2（行 1033）**均写 `NUM_LOGICAL` 且取整平面单元口径**，无「核数用通用式、解码用单元式」残留。
  - **一般网格不受影响**：§5.2 表前 9 行（非 C4b）仍按 `N*out_H*num_cblk`，与 §3.3 一般分支解码 `n=uid//(out_H*num_cblk)` 同粒度；`(1,16,11,13)→(None,5)`（H 恒等但 W 非恒等）正确判为**一般网格**（11），未误入 C4b。
  - **判定**：满足维度 3「核数与 block 取值自洽」（`core-split-strategy.md` §1 要素①/核数自洽），**pass**。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel(num_kernels, is_npu=True)`，§3.5.1 已声明，`T.Kernel.md` §2.3 确认 NPU ≤2 维、1 维合法）。
- GEMM 非整除 / L0C 溢出：不涉及（无 Cube 计算，§3.5.1 标注「L0C 不适用」正确）。
- GPU 专用 API：无（全部 v-prefix / `T.copy` / `T.transpose` / `T.clear`）；`threads` 参数已按脚手架 K9 移除（§3.5.2/§4.7 一致）。
- host 侧约束：§3.5.1 声明 host 只做 `unsqueeze/squeeze/contiguous` 视图操作，与 `ascend-constraints.md` §4「host 侧禁止改动输入张量内容」不冲突 ✓。

### 5. 循环与同步 — pass

- §6.1 逐维给出循环类型与 API：task 级 `T.Kernel` / 累加器初始化「整块清除」`T.clear` / 核内多任务 `T.serial(num_local)`（静态界+守卫）/ bin H `T.serial(MAX_BIN_H)`（静态界+`if kh<bh`）/ bin W Python `range` 展开 / `o_w` Python 展开 / 元素级 C 轴 `T.vadd` 向量化 / 退化路径单次 `T.reduce_sum`。**§6.1 与 §1.6.2 一对一映射，无「§1.6.2 声明全覆盖、§6.1 缺行」问题**。
- 同步策略与模式匹配：Developer 自动同步，§7.2 明确「无手动同步点」，与「单引擎（Vector）、单阶段、无跨核通信」相符 ✓。
- 尾块处理 §6.4：任务尾块（守卫丢弃）、H 方向 bin 尾块（`kh>=bh` 跳过不读）、W 方向（编译期 `range(bin_w(o_w))` 精确覆盖、无掩码）三点齐备；退化路径 C4/C4b 的守卫符号已同步为「与 `N*num_cblk` 同粒度」✓；明确「无 GEMM 式整除要求」。
- §6.3 不流水化的理由（单阶段流、`kh` 次数 1–10）合理，且给 Stage 4 留 `T.Pipelined` 探索项。

### 6. 验证方案 — pass

- §8.1 含 PyTorch 参考实现草案，与源仓基准（`examples/TileOPs/tests/ops/test_pool.py:62-65` `ref_program = F.adaptive_avg_pool2d`）同源；明确「不复刻 §0.6 NPU 算法」并给独立性核对 ✓。
- §8.2 精度标准 `float16 (1e-3,1e-3)` / `bfloat16 (1.6e-2,1.6e-2)` —— **与 `test_pool.py:79` 逐字一致**（fp32 不适用，因算子拒绝 fp32）✓。
- §8.3 L0 门槛计划 10 例，覆盖 4 组 manifest workload（已核对 `pool.yaml:41,47,48,49`：smoke `[1,8,4,4]→(2,2)` fp16、`resnet-global [8,2048,7,7]→(1,1)` fp16、`spp-6x6 [2,128,56,56]→(6,6)` fp16、`nondiv-7x7 [2,64,55,57]→(7,7)` bf16 全部命中）+ 6 组 `test_pool.py` PARAMS 用例（`:37-55` 逐条对照命中）+ C4/C4b 特化 + 3D CHW rank 保持；L0-5 `(1,8,9,11,out=None)` 直接覆盖本轮修复的 C4b 路径。
- L0 通过判据含 `op.kernel` 类型断言（`AdaptiveAvgPool2dKernel`），与 `test_pool.py:81` 一致 ✓。

### 7. 完整性与一致性 — pass

- **章节齐全**：§0（0.1–0.7 全）+ §1–§11（含 §11 修订记录），模板齐备 ✓。
- **无占位符**：全文逐词扫描未发现模板花括号占位符与任何未完成标记 ✓。
- **同类引用真实**：§3.5.3 引用的 6 条 `examples/` 路径**逐条实测存在**（`_make_lerp_tensor_kernel.py`、`_logsumexp_kernel_single.py`、`perf_opt/_argreduce_kernel.py`、`vec_add_2d_multi_buffer.py`、`test_pool.py`、`pool.yaml`）；§1.6.0/§1.6.3 引用的 pattern-library `PL-1.1/1.2/1.3/1.4/1.5` 在 `layout.md` 中**全部存在**（PL-1.1「~5.4µs / 4 次转置」、PL-1.2「C 轴切片累加」、PL-1.3「106–147µs」、PL-1.4「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs」、PL-1.5）；`CONST-capacity-910B2C`/`CONST-vector-launch-overhead`/`CONST-mte2-degradation` 均在 `constants.md` ✓。
- **内部一致 — pass**：§1.6.2 表内 #8 标 `✅` 与结论段「**#8** 标 `✅`——编译期展开为向量 op 序列」**一致**（v1 建议已修复）；保留计数 4（#7/#9/#10/#11）与表内标记自洽。API 映射 / 内存规划 / Tiling / 循环四处的 C4/C4b 口径与 §5.2/§5.5 一致，无矛盾。

### 8. 算法优化分析 — pass

> 检视方式：§1.6.0 调研结论**独立复核**（对照 `algorithm-candidates.md` ALG-pooling 命中行 + 源码证据 + pattern-library 实测条目 + 复杂度独立复算）；§1.6.1 等价性**独立推演** + `verify_equiv.py` **重跑比对**；§1.6.2 覆盖与 §3.3/§6 **交叉核对**。**未凭设计自述**。

**8.1 §1.6 章节存在性 — pass**：§1.6.0/1.6.1/1.6.2/1.6.3 四子节齐备。

**8.2 算法调研存在性与完整性（§1.6.0）— pass**
- 四问齐全：R1 候选表**含基线**（#0 源算法，表内标注「基线」）且参考表 ALG-pooling 命中行两条（常数权重卷积 / sum-pool×常数）均已评估并标注纳入/合并；R2 在线算法有**明确结论**（「有，且基线本身已是单遍」——结构判据 = running 累加量 + `O(out_W·CH)` 状态，并以源码 `sum_val += ...`（§0.4 #6）为「存在在线变体」的直接证据，**与源码一致**）；R3 覆盖**四口径**（FLOPs / GM 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表，口径在表下注明「不含 UB 内搬运、中间缓冲不落 GM」）；R4 逐候选六维评估（单元匹配/容量/对齐静态边界/流水/结论）+ 负向淘汰证据引用。
- **候选库命中行覆盖核对**：`algorithm-candidates.md:53` `ALG-pooling` 的 R1「avg pool = 常数权重卷积；sum pool × 常数」→ 设计 R1 表 #5/#6 **均已评估**；R2「滑窗天然流式」→ §1.6.0 R2 ✓；R3「窗口重叠数据的重复读」→ R3 表口径 ✓；R4「C 轴向量轴；跨步系数」→ 候选 #0' 与 §1.6.3 ✓；`kb_links` PL-1.2/1.3/1.4 全部被引用并消费 ✓。**无漏候选**。
- **复杂度独立复算**：spp（`N=2,C=128,H=W=56,out=6×6`）——`scan_H` 依 `pool.py:970` 复算 = `56 + #{j∈1..5: (j·56)%6≠0}` = 56+4 = **60** ✓；FLOPs = `2·128·60·60 = 921600` ✓ 与 manifest `roofline.flops` 一致；访存 = `(2·128·3136 + 2·128·36)·2 = 1624064 B ≈ 1.624 M` ✓。候选 2 可分离 vs 基线（`56·60 + 6·60 = 3720 > 3600`）✓；nondiv 差 3843 vs 3892 ✓ **独立复算与设计数字一致**；**中间缓冲未漏计 GM 往返**（候选 2/3 额外缓冲单列，候选 6 权重矩阵 GM 往返计入）。
- **负向断言逐条复核**：
  - 「无更低复杂度算法族」——成立（下界 = 读满输入 `N·C·H·W`，基线仅多约 4% 重叠读；设计已在「调研范围声明」写明四项范围）。
  - 候选 6（Cube/implicit GEMM）淘汰依据「`out_W=6/7 < 16` 违 Cube 分形下限 + 权重构造属 host 改内容」——已核 `ascend-constraints.md` §1 行 23「GEMM 要求 M,N 为 block 整数倍（`M < block_M` 时零 block 启动 → 输出全零或除零编译崩溃）」与 §4 行 60-62「host 侧禁止改动输入 NPU 张量内容」，条款真实且指向正确 ✓。
  - 候选 3（前缀和差分）「`T.cumsum` 不支持 bf16」——已开 `docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1：dtype 表 `fp16/fp32/int32/int64 √，bf16 ×`，**证据成立** ✓。
  - §1.6.3 候选 1/§1.6.1「`T.reduce_sum` 不支持 bf16」——`docs/.../规约操作/T.reduce_sum.md` §2.2.1 `bf16 ×`，**成立** ✓。
  - §1.6.3 候选 3「host permute 106–147µs」——`layout.md` PL-1.3 实测条目一致 ✓。
  - 互联网来源：设计显式声明「未使用」并给理由（本地参考表 + 源码 + pattern-library 已覆盖），符合要求 ✓。
- **调研→设计一致性 — pass**：§1.6.0 R4 表行「1 倒数乘」=「✅ 采纳为微优化（并入 #0'；正式论证见 §1.6.1 C1）」、调研结论段=「另**采纳**候选 #1 倒数乘 → §1.6.1 C1…三项采纳与 §1.6.1 优化结论**逐项一致**」、§1.6.1 C1 行=「**已采纳**」+ 优化结论=「**采纳 3 项（C1/C4/C4b）**」、§0.6 R2 步骤 3 与 §3.3 归一化实际用倒数乘、§3.1 步骤 6 无「或保留 ÷ d」歧义——**全文对候选 #1 已收敛为唯一结论「采纳」**（v1 已闭合、本轮保持）。迁移承接无静默丢弃（§0.4 九项处置逐项可见，§0.6 逐项标注承接）。

**8.3 数学等价优化完整性（§1.6.1）— pass**
- 四要素核对：**C4**（原式 → `(Σx)·1/(H·W)` → 逐位等价论证 → 收益 `resnet` 49 次向量加 → 1 次归约 ≈49×）；**C4b**（单位 bin 均值 = 元素本身 → 收益消除全部冗余归约）；**C1**（`sum/d → sum·(1/d)` → 容差内等价论证 → 收益 144 次归一化向量 op、占比 ≈1.0%）——三项四要素齐全；C2/C3 未采纳均给量化理由（豁免成立）。
- **等价性独立推演**：
  - C4：`out_H=out_W=1 ⇒ bin_H(0)=[⌊0⌋, ⌈H/1⌉)=[0,H)`、`bin_W(0)=[0,W)`，归一化因子 `H·W` 为编译期常数 ⇒ 与基线**同一元素集合、同一除法**，逐位等价 ✓。
  - C4b：`out_H=H_in ⇒ bin_H(o_h)=[⌊o_h·H/H⌋, ⌈(o_h+1)·H/H⌉)=[o_h,o_h+1)`（单位 bin），`bin_size=1 ⇒ out=x` ✓。
  - C1：`inv=fl32(1/d)` 相对误差 ≤2^-24≈6e-8（**远低于 fp16 ulp 2^-10≈1e-3、bf16 2^-8**），乘后相对差 ≈1.2e-7；舍入到 fp16/bf16 后每个元素差 ≤1 ulp——**推演成立**，且与 §8.2 atol/rtol 1e-3/1.6e-2 相容 ✓。
- **机器验证比对**：`verify_equiv.py` 重跑 **exit=0、45/45 `EQUIV_PASS`**，与 §1.6.1 内嵌表**逐行一致**；C4/C4b 判据 0 ulp（逐位）、C1 判据 ≤1 ulp（脚本 `_ULP_TOL` 分档），**D-1 验证范围与「已采纳 3 项」严格匹配**。
- **§3.1 以优化后公式为输入**：§3.1 表步骤 6 采用 C1 倒数乘式、C4/C4b 行采用 §1.6.1 优化后公式 ✓。

**8.4 向量化替代完整性（§1.6.2）— pass**
- **已覆盖点核对（与 §3.3 伪代码逐行交叉）**：#0 累加器零初始化 ↔ 伪代码 `T.clear(acc)` ✓；#1 累加 ↔ `T.vadd` ✓；#2 逐元素 cast ↔ `T.vcast` ✓；#3 归一化 ↔ `T.vmul` ✓；#4 输出 cast ↔ `T.vcast` ✓；#5 退化归约 ↔ `T.reduce_sum` ✓；#6 布局重排 ↔ `T.transpose` ✓；#7 `kh` 循环 ↔ §6.1 保留 ✓；#8 `kw` 编译期展开 ✓；#9 任务解码 ✓；#10 bin 边界（host/trace）✓；#11 host 视图 ✓。
- §1.6.2 保留计数「4 处（#7/#9/#10/#11）」与表内标记一致（#8 标 ✅、不计入保留），**盘点自洽**（v1 建议 #8 表/文矛盾**已修复**）。
- 替代 API 全部有 `docs/` 条目佐证；不可替代项（#7/#9/#10/#11）理由具体到本算子（tile 级顺序依赖 + 动态边界 / block 级索引 / host 元数据 / host 视图），非泛泛而谈 ✓。

**8.5 向量化轴与布局决策完整性（§1.6.3）— pass**
- 类别必选候选齐全（窗口/池化类）：「I/O 原生布局 + 最内连续轴」（#1，W 轴）与「核内重排布局 + 高整除性轴」（#2，C 轴）两类均已枚举，另加 host permute（#3）与 Cube/implicit-GEMM（#4）；迁移任务已显式声明「源的并行轴选择是输入而非结论」✓。
- 逐候选量化评分（整除性/尾 lane 浪费/累加链形态/repack 代价/UB 影响），弃选均有量化理由；**未把 I/O 契约当内部布局依据** ✓。
- **弃选论证前提亲自核对**：候选 #2 repack 前提「核内 `T.transpose` 二轴交换 ~µs 级」与 `T.transpose.md` §2.3（核内 UB 级、仅二轴）+ PL-1.1（~5.4µs）一致；#3 host permute 引用 PL-1.3 实测；#4 分形前提与 `ascend-constraints.md` §1 一致 ✓。
- **D-3 实验裁决三件套完整**：主选 `CH=min(C,64)` + 备选 `CH=min(C,8)`（buffer 形状/UB 预算/分核三要素按 `num_logical=192`、persistent 40 核×5 任务重算，达可直接实现深度）+ 裁决计划（代表 shape、msprof 指标、反解未知常数、阈值「备选 < 主选×(1−0.10)」、回写路径）+ 按 shape 分派评估；D-3 段已把 PL-1.4 实测（CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs）列为主证据，主选即已测最优档，§9.1 #3 同步 ✓。

**8.6 与下游章节一致性（§3.1/§6）— pass**：§3.1 以 §1.6.1 优化后公式（含 C1 倒数乘）为输入；§6 循环结构中不出现 §1.6.2 未论证的逐元素标量循环（`T.clear` 整块清零、`kh` 为图块级守卫循环、`kw`/`o_w` 为编译期展开的向量 op 序列）✓。

## 检视附注（非阻塞，不影响结论）

- **本轮修订闭合确认（对照 `history_version/review_v1.md`）**：
  - **v1 阻塞（C4b 恒等路径核数/单元粒度/解码不自洽，越界风险）**：**已真正修复**——收敛为整平面单元口径 `NUM_LOGICAL = N*num_cblk`（`(1,8,9,11)` = 1），§3.3/§4.3/§4.4/§5.2/§5.3/§5.4/§5.5/§6.1/§6.2/§6.4/§9.2 全文一致，越界风险经独立推演消除（见维度 3）。
  - **v1 建议（§1.6.2 #8 表/文符号矛盾）**：**已修复**——结论段改为「#8 标 ✅」，与表一致（见维度 7）。
  - **v0 的 2 阻塞 + 4 建议回归**：`T.clear` 盘点（§1.6.2 #0 / §3.3 / §6.1 / §1.4 / §3.4 / §9.2）、候选 #1 倒数乘全文收敛为「采纳」+ `verify_equiv.py` C1 ≤1 ulp 分档（45/45 PASS）、CH=64 口径统一、C1 收益分母 144、UB 膨胀双档、D-3 补 PL-1.4 主证据——**六项均保持闭合、未被本轮修订推翻**。
- **Stage 3 实施提醒（伪代码为决策示意，§3.3 已声明「局部次序交 Stage 3」）**：§3.3 的 C4/C4b 退化路径 `T.copy` 切片形态与声明 buffer 形状存在 rank 差异（C4 `x[n, cb*CH:cb*CH+CH, 0:H_in*W_in]` 对 4D 输入的第三索引切片 > `H_in`、目标 `flat_ub=(CH,H_in*W_in)`；C4b `x[n, c0:c1, 0:H_in, 0:W_in]` 为 3D 而 `stage_ub=(CH,H_in*W_in)` 为 2D），Stage 3 需按实际 UB buffer 形状调整切片/归约维（如按 `(CH,H_in,W_in)` 装载后跨维归约，或显式处理展平）。该点为**伪代码示意层面的实施细节**，不影响 §1.6.1 的 C4/C4b 数学等价结论（已由 `verify_equiv.py` 独立验证）、不改变分核/内存/轴决策，**不构成阻塞**。
