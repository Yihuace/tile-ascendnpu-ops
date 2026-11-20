# _group_norm_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/group_norm/_group_norm_kernel/DESIGN.md`
- 任务类型: 迁移（harness 子模式；NPU 脚手架与集成文件位于 `examples/TileOPs/`）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/norm/group_norm.py` 第 45–121 行（`_group_norm_kernel`）+ `/workspace/TileOPs-gpu/tileops/ops/norm/group_norm.py` 的 `GroupNormFwdOp.forward`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 7

## 机械复核（D-1/D-5）

**前置门禁**：`python3 .agents/tools/statectl.py gate 1 --dir examples/group_norm/_group_norm_kernel`
→ `{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 1, "warnings": []}`（零失败、零警告；含 `S1-REF-PATHS` 引用路径存在性核对 → 设计引用的 `docs/`/`examples/` 路径逐一存在）

**D-1 重跑等价性验证**：`verify_equiv.py` 重跑结论 **PASS**（与 §1.6.1 内嵌结果表逐格一致，全部 `EQUIV_PASS`）。
说明：系统 `python3` 缺 `torch`（`ModuleNotFoundError`），改按设计文档记录的运行环境用 `/workspace/venv/bin/python3`（torch 2.9.0+cpu）执行，退出码 0。

| 采纳项 | cases | 准则 (a) | max ulp(cand vs base) | 超容差元素数 | 相对 fp64 最大劣化 | 两准则皆失败 | 判定 | 与 §1.6.1 表一致 |
|---|---|---|---|---|---|---|---|---|
| O1 除法→乘倒数 | 31 | 31/31 | 1 | 0 | +0.0e+00 | 0 | EQUIV_PASS | ✅ 一致 |
| O2 rstd 组合式 | 31 | 11/31（20 例由 (b) 承接） | 48008 | 8174 | +0.0e+00 | 0 | EQUIV_PASS | ✅ 一致 |
| O3 中心化值复用 | 31 | 31/31 | 0 | 0 | +0.0e+00 | 0 | EQUIV_PASS | ✅ 一致 |
| O4 精确 D 方差 | 31 | 29/31（2 例角点由 (b) 承接） | 3220099892 | 588 | +3.3e-07 | 0 | EQUIV_PASS | ✅ 一致 |
| O5 单位仿射消除 | 31 | 31/31 | 0 | 0 | +0.0e+00 | 0 | EQUIV_PASS | ✅ 一致 |

证据块复核（与 §1.6.1 正文数字一致）：O2 分段证据 raw `T.vrsqrt` 2.87e-3 / 组合式 6.10e-8 / 精确 `1/sqrt` 7.83e-8；O4 角点 `padcancel-d2-fp32` 基线 9.952e-1 vs 精确式 9.389e-3、`padcancel-d196-fp32` 基线 4.572e+2 vs 精确式 1.841e+0；否决项 C1 矩式 worst 6.018e+3（largemean-fp16，违反 1536 元素），与 §1.6.1 否决表数字一致。

**空真守卫核对（逐组）**：31 组 case 的比较元素数全部 > 0（最小 `padcancel-d2-fp32` = 4×2 = 8 元素），**无零元素空真行**；O2/O3/O5 三组在全部 31 行上两臂实际不同（O2 的近似扰动、O3 的重推 vs 复用、O5 的乘 1 加 0），非空真。**覆盖缺口（非空真但不可判别）**：O1 组中 `D ∈ {64, 256, 4096}`（2 的幂，`fl(1/D)` 精确）与 O4 组中 `D ∈ {256, 4096}`（pad = 0，两臂管线恒等）的行两臂逐位相同，对该采纳项无判别力（O1 不可判别 9/31、O4 不可判别 6/31）；有效判别样本仍充足（O1 侧非 2 的幂 D 共 12 行 + 196 角点，O4 侧非对齐 D 共 15 行 + 2 个 padcancel 角点），且 O2/O4 另有独立的分段扫描 / 角点证据块支撑结论，故不构成阻塞（列入「非阻塞改进建议」#4）。

**D-5 算术复算**：`python3 .agents/tools/design_calc_check.py --design examples/group_norm/_group_norm_kernel/DESIGN.md`

```json
{"capacity_constants": {"ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0,
                        "source": "pattern-library/constants.md CONST-capacity-910B2C"},
 "checks": [
   {"check": "ub_budget",   "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
   {"check": "l0c_budget",  "status": "skip", "detail": "no affirmative Cube-path API ... in this design"},
   {"check": "core_split",  "status": "skip", "detail": "insufficient parsed dims {'bm': 1}; declared logical cores = None"},
   {"check": "r3_metrics",  "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
 ]}
```

- `r3_metrics` **pass**（§1.6.0 R3 表四口径齐全：元素级 op 数 / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线候选在表）。
- `l0c_budget` **skip** → 人工核：纯 Vector 设计，全篇无 `T.gemm` / `load_nd2nz` / `store_fixpipe` / `Scope("Cube")`，不涉及 L0/L1/分形（§2.1/§5.1）→ 不适用，判定无风险。
- `ub_budget` **skip** → **人工已核**：按 §4.5 口径独立复算——`block_m = min(8, max(1, 9216//D))`（D=256→8、4096→2、6272→1、7200→1、196→8），驻留元素数 × 20 B/元素 = 40 960 / 163 840 / 125 440 / 144 000 / 31 360 B，全部 ≤ 196 608 B（最大 160 KB）；`_UB_MAX_ELEMS = 9216`（= `196608 // 20 = 9830` 下取整到 8×1152，余量 ≥12 KB）与 `_UB_CAPACITY_BYTES = 196_608`（= 本机 `ub_cap` 实查值）自洽。**fp32 路径口径差异见「非阻塞改进建议」#1**（不影响结论：同一缓冲结构的同族前作 fp32 溢出报文 `245760 B = 20 × 3 × 4096` 精确反证 20 B/元素对该结构成立）。
- `core_split` **skip**（解析维度不足）→ **人工已核**：§5.5 三要素齐全——① `num_logical = ceil(M/block_m)` 逐 workload 表；② 物理核实查记录（`NPUUtils.get().get_aicore_num() → 20`、`AscendArch().aivector_core_num → 40`、`ub_cap → 196608`，设备 Ascend910B3 / npu-smi 26.1.1），与 CONST-aicore-910B2C 的 910B3 数据点（20/40/196608）一致，且明确禁止跨型号复用 910B2C 的 24/48；③ 规模判定（≤物理核 → 无需适配；中/大规模 → 固定 40 核 persistent 核内串行 + grid-stride + `if block_id < num_logical` 守卫），核内串行边界 `num_local_tasks` 为 host 编译期常量（静态）。核数/block 取值逐 workload 自洽（§5.5 表）。

**独立推演（不依赖上述脚本）**：§1.6.1 五项优化的等价性逐项复核见维度 8；§1.6.0 R3 复杂度逐候选复算见维度 8。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

（已亲自 Read `/workspace/TileOPs-gpu/tileops/kernels/norm/group_norm.py` 全文与 `/workspace/TileOPs-gpu/tileops/ops/norm/group_norm.py` 全文，逐项与 §0 核对）

| 检查项 | 结论 | 源码证据 ↔ DESIGN 章节 |
|---|---|---|
| 语义理解正确性 | pass | 源码 L58 `D_padded=_align_up(D,256)` / L90 `acc[i]/float(D)`（ddof=0）/ L101 `(acc − pad_count·mean²)/float(D) + eps`（eps 加在方差上）/ L95-96 中心化两遍（非矩式）/ L84-85 上 cast fp32 + L108 隐式截断（fp32 opmath 单次舍回）↔ §0.1 全部一致；约束 `x.ndim>=2`/`C%num_groups==0`/dtype 三值 ↔ op 层 L100-114 一致 |
| I/O 与输出 shape | pass | 源码 L64-69 `main(x, weight, bias, y)`、L61 `out_idx=[3]`、L66-68 `weight/bias: (D_padded,)`、L139-142 `register_fake` 返回 `torch.empty((M, D_padded))`（无转置）↔ §0.2 表与「无转置、无布局变化」结论一致；op 层 L227-244 输出 reshape 回 `orig_shape` 且 `y[:, :D]` 截断 ↔ §0.2/§4.2 一致 |
| affine 分层（关键） | pass | **kernel 层**：L67-68 `(D_padded,)` 行广播 + L107-113 kernel 内 `weight[j]/bias[j]`；**op 层**：L237 `y_2d = kernel(x_2d)`（只传 x，故走 L232-233 `_identity_affine_for` 单位元分支）、L242-243 `weight.reshape((1,C)+…)` per-channel affine 在 kernel 外。↔ §0.1 表、§0.5 末行、§0.6 R6、§1.6.1 O5 完全相符。且与 NPU 脚手架 wrapper（`examples/TileOPs/tileops/kernels/norm/group_norm/group_norm.py`：4 参 custom_op、`_identity_affine_for`、per-channel affine 在 op 层 `examples/TileOPs/tileops/ops/norm/group_norm.py`）一致，**未折叠** |
| 算法解读完整性 | pass | §0.3 表 L1/L2/S1/S2/E1/W1/H1 逐语句对得上：L80-81 ↔ L1；L84-85 ↔ L2；L88-90 ↔ S1；L95-104 ↔ S2；L107-113 ↔ E1；L116-117 ↔ W1；host 侧 `F.pad`/单位 affine/`y[:,:D]`/op 层 reshape+affine ↔ H1。**无遗漏步骤、无臆造步骤**；数据流图含全部 buffer（`shared_buf`/`x_local`/`x_f32`/`acc`/`mean_val`/`rstd`） |
| 优化手段覆盖完整性 | pass | §0.4 12 项均可由源码逐语句反查（SMEM 驻留 L72/L80-81；256 padding 及填充校正 L10-12 文件头注释 + L58/L101；中心化两遍 L95-104；fp32 全程 L74/L85；单次 cast 复用 L83-85 注释；`block_m=1` 结构性默认 L186 `select_row_config`；`threads=128`；单位 affine 兜底 L192-203；SMEM 单缓冲复用 L72/L80/L116；op 层 kernel 外 affine L242-243；`lru_cache` L45；`T.copy` 整块搬运 L80/L117）；无「实现层/算法层」误判 |
| 耦合性判定合理性 | pass | §0.5 每项有四态处置 + 依据：保留（语义/affine 分层/两遍方差/`lru_cache`/`out_idx`）、等价替换（SMEM→UB、fp32 中转→`vcast`、整块 `T.copy`、单缓冲复用、`ALIGNMENT` 后端常量）、舍弃（256 padding / `threads`）、重新设计（`block_m` 结构性 1、`threads`+两级并行）。与 `core-split-strategy.md`、`ascend-constraints`（经由 pattern-library traps 条目）无矛盾；舍弃项均写明理由且意图由 §0.6 末表逐项承接（无静默丢弃） |
| NPU 重设计可行性 | pass | §0.6 R1–R7 均为「源方案 → NPU 新算法 → 语义保持论证」三要素：R1 一维 persistent（满足「不支持三维 Kernel」）；R2 `block_m` UB 驱动；R3 `T.min` + 双显式 slice + 垃圾行丢弃（dst 零起点）；R4 显式 `vcast` fp32 中转（bf16 无 Vector 算术，必需）；R5 `vsqrt+vdiv` 规避近似 `vrsqrt`；R6 单位仿射消除 + op 层 affine；R7 `ALIGNMENT=1` 精确 D。重设计均满足 NPU 硬件约束（一维 Kernel / UB 容量 / 无 256 对齐要求 / 静态边界） |
| 设计与决策一致性 | pass | §1.4/§3.1/§3.3/§4/§5/§6/§7 描述的是 §0.5/§0.6 决策后的算法：§1.4 列 8 步与 §3.3 伪代码一致；§3.3 接口（工厂 `_group_norm_kernel(M,D,eps,dtype)`、`main(x,weight,bias,y)`、`out_idx=[3]`）与源一致；无「§0 判重设计、§3 照抄源方案」脱节（§0.6 R3 的 `T.min`+双 slice 已落入 §3.3/§6.4；R7 的 `ALIGNMENT=1` 已落入 §3.3/§4.5） |
| golden 独立性 | pass | §8.1 golden = 移植源仓 `GroupNormTest.ref_program`（`F.group_norm(x.float(), g, weight.float(), bias.float(), eps).to(x.dtype)`，已在 `examples/TileOPs/tests/ops/test_group_norm.py` 实读核对逐字一致），显式声明**不复刻** §0.6 的 NPU 算法（无 fp32 中转链/persistent/vsqrt+vdiv）；kernel 级对照用单位 affine。golden 依据 §0.1 语义，独立于 §0.6 |
| host/op 层引用路径真实性 | pass | 逐条核对：源算子 L45-121 ✓、op 层 `GroupNormFwdOp.forward` ✓（L154-244）、L237 的「只传 x」✓、L139-142 `register_fake` ✓；`examples/TileOPs/…` 全部路径 `ls` 存在（见维度 7）；`gate 1` 的 `S1-REF-PATHS` 亦通过 |
| 多输出/变体函数的迁移范围 | pass | 源文件含 no-affine 变体（L251-418：`_group_norm_no_affine_kernel` 的 `main(x,y)` + `out_idx=[1]`、`GroupNormNoAffineKernel`、op 层 `GroupNormNoAffineFwdOp`）。设计**显式排除**该变体（文档头 L7、§0.5、§10.5、§0.7）且全文未出现 no-affine 语义/代码/`2` 参签名；与 manifest（`examples/TileOPs/tileops/manifest/norm.yaml` 的 `GroupNormFwdOp` 条目、注释声明 no-affine 未迁移）及脚手架（NPU wrapper/op 层均无 no-affine）一致，**无范围蔓延**。主算子三输入（x/weight/bias）与 manifest `signature.inputs` 一致 |

**§1–§7 与迁移决策一致性**：§1.2/§1.4 的差异清单、§3.5.2 差异表、§0.6 末「意图承接汇总表」三处互相一致且逐项可追（256 padding→连续 span DMA；`block_m=1`→按行归约 + `T.min`；`threads`→Vector 向量化 + persistent；kernel 内 `weight[j]/bias[j]`→op 层 affine + 单位仿射消除；SMEM→UB 中转）。

### 1. API 可行性 — pass

- §3.2 逐条 API 在 `docs/Tilelang.language/` 与同族已交付实现中有佐证，均已亲自打开核对：
  - `T.Kernel(N, is_npu=True)` 一维 + `(cid, _)` 解包、`T.serial` + `if` 守卫、尾块 `T.min` + 双显式 slice `T.copy`、`T.vcast(rint)`、`T.reduce_sum(dim=1)`、`T.vmul/vsub/vadd/vsqrt/vdiv`、`alloc_shared` 缓冲布局——与 `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` L263-345（已交付、已 Stage 4 调优的同族 row-reduction 实现）**逐语句同形**（含 `T.vadd(var, eps, var)`、`T.vsqrt(var, s2)`、`T.vdiv(s2, var, rstd)`、`T.vsub(x_f32, mean, x_f32)`）。
  - `T.vcast` dtype 表：f16→f32 rint ✓ / bf16→f32 rint ✓ / f32→f16,bf16 rint ✓（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` L29-42）。
  - `T.reduce_sum`：bf16 列 ×、fp32/fp16 √，`(M,N)→(M,1)` 同 rank 约束 ✓。
  - `T.vmul/vadd/vsub/vdiv/vsqrt`：bf16 列 ×（C2 的依据成立）、fp16/fp32 √；行广播 `[M,N]↔[M,1]` 支持；标量仅可作 src2；dst 可 in-place；**操作数须在 UB**（`T.vmul.md` §2.2.1/§2.2.2/§2.3）——设计的全部 v-op 均为 UB `alloc_shared` 且 in-place 写入合法。
  - `T.min`：标量语义、不支持 shape ✓（仅用于 `real_m`）。
  - `T.copy`：§2.1 同 dtype、§2.2.2 规则、§2.4 双显式 slice；`切片` 与 `size` 互斥（设计不用 `size`）✓。
- **API 与模式匹配**：Developer 模式，自动同步，无手动 `sync_block_set/wait`/`pipe_barrier`（§7.2）✓；`T.Parallel` 元素级循环已按 §1.6.2 全部改为 v-op 全 buffer ✓。
- **v-prefix 优先**：全部新 API 为 v 前缀（`vcast/vmul/vsub/vadd/vsqrt/vdiv` + `reduce_sum`），未使用 `npuir_*` ✓。

### 2. 内存层级规划 — warn

- **搬运路径完整**：§4.4 给出 GM→UB→（计算闭环）→UB→GM，纯 Vector 无 L1/L0 参与；无 GM→L0 越级路径；无 GM→GM 拷贝 ✓。
- **Buffer 层级标注**：§4.3 全部 buffer 标注 UB，含角色复用时序（`stage` 双角色、`x_f32` 三角色、`sq_f32` 双角色）✓，与 §3.3 一致。
- **UB 预算**：见「机械复核」`ub_budget` 人工复算，manifest 四档 + 最紧测试档全部 ≤ 196 608 B（最大 160 KB）✓；`block_m` 公式、`_UB_MAX_ELEMS`、工厂 guard 三者自洽 ✓。
- **L0C 容量**：不适用（无 Cube）✓。
- **warn 理由**：§4.5 把 fp32 直连档的平台值写成「同为 20.00 B/元素」并归到 TRAP-UB-multibuffer-inflation 第三证，而该条目正文写的是「**fp32 直连无论 8 B 还是 16 B resident，同为精确 26.00 B/elem**」（`traps-compiler.md` L98 与 ada `perf_opt/opt_log.md` L234 同口径）。**但**同族前作源码注释内的原始溢出报文（`_ada_layer_norm_kernel.py` L138-139：「`245760 B = 20*3*4096`（**fp32** bm=3 N=4096）、`240000 B = 20*4*3000`（**fp32**，exact match）」）与本 kernel 的缓冲结构（`stage`+`x_f32`+`sq_f32`+5×(bm,1)）完全同形，对 8 B/elem resident 的 fp32 结构**精确反证** 20 B/元素；即设计采用的数值有其引用的原始实测支撑，条目的 26 应属 v2（16 B/elem resident）结构或蒸馏偏差。**独立复算确认风险边界**：若按 26 B/元素，fp32 `D = 4096, block_m = 2`（设计 L0-6 用例）需 212 992 B > 196 608 B；manifest 的 fp32 workload 仅 smoke-g8（D=256）→ 53 KB 安全，集成路径 wrapper 默认 `block_m = 1` → 106 KB 安全。故按建议级处置（见「非阻塞改进建议」#1），不阻塞。

### 3. Tiling 策略 — pass

- **Block 划分**：§5.2 给出 `block_m = min(8, max(1, 9216//D))`、`block_d = D`（全 D 驻留）及理由（D 不分块 → x 单遍扫描；M 分块 → 行独立）；§4.5 逐 workload 表给出具体取值与来源 ✓。
- **对齐约束**：尾轴连续 stride = 1；D 无 256 对齐要求（§0.6 R7）且由 pattern-library/前作 L0（N=13/514/1152/3000/4096 全过）佐证；UB 32B 对齐交编译器，多缓冲 UB 需求按元素计（与行宽对齐无关）✓；行块是连续 span，DMA 粒度 = `block_m·D·elem_bytes`（16 KB）并引用 CONST-copy-floor-dma-granularity 的方向性结论 ✓。
- **非整除处理**：纯 Vector 无 GEMM；行尾用 `T.min` + 双显式 slice + 垃圾行丢弃（§0.6 R3 / §5.4 / §6.4），垃圾行按行隔离的论证成立（`reduce dim=1` 逐行、v-op 逐元素或行广播，无跨行混合），输出只写有效行 ✓。
- **分核策略（三要素，对照 `core-split-strategy.md` §1 与 §2.2）**：
  - ① 逻辑核数：`num_logical = ceil(M/block_m)`（一维网格，`block_d = D`）✓ 逐 workload 表列全。
  - ② 物理核数依据：**实查记录**（查询代码 + 返回值 + 设备名 + 日期）：`NPUUtils.get().get_aicore_num() → 20`、`AscendArch().aivector_core_num → 40`、`ub_cap → 196608`；纯 Vector 核数翻倍（20→40）与直查一致；并明确禁止跨型号复用 910B2C 的 24/48（与 CONST-aicore-910B2C 的 910B3 数据点一致）✓。
  - ③ 规模判定与分核方案：逻辑 ≤ 物理（2 ≤ 40）→「无需适配」+ 依据；中/大规模 → 固定 40 核 persistent 核内串行 + grid-stride + 守卫，**核内串行边界为 host 编译期常量（静态）**（`num_local_tasks = ceil(num_logical/num_kernels)`，trace 期折叠，PL-1.5 先例）；负载不平衡 ≤1 任务并已量化（image-g32 `128 = 40×3+8`、tail-spatial-g16 `64 = 40+24`）✓。
  - 核数与 block 取值自洽（§5.5 表逐 workload 给出 num_logical/num_kernels/num_local_tasks）✓。
- 机械 `core_split` 为 skip（解析维度不足），已按上表人工核验（见「机械复核」）。

### 4. 技术约束检测 — pass

| 约束 | 处理 | 判定 |
|---|---|---|
| 不支持三维 Kernel | 一维 `T.Kernel(num_kernels, is_npu=True)`，无需 block_metadata | pass |
| GEMM 非整除 / 分形 | 不适用（无 GEMM） | pass |
| L0C 溢出 | 不适用（纯 Vector） | pass |
| bf16 无 Vector 算术 | fp32 中转链（§0.6 R4，文档 dtype 表已核） | pass |
| UB 多缓冲膨胀 | §4.5 平台值 + 工厂 guard + §9.1 风险登记 + §1.6.3 参数级 A/B 降档路径 | pass（warn 见维度 2） |
| `threads=` 残留 | 已移除（TRAP-threads-kwarg-noop，已核） | pass |
| host 侧不改写输入张量内容 | `ALIGNMENT = 1` 使 wrapper 的 `F.pad` 分支不触发；host 只做视图/元数据 | pass |

### 5. 循环与同步 — pass

- §6.1 明确循环类型：任务维 `T.Kernel` + `T.serial`（persistent）、元素维向量化（无循环）、尾块标量 `T.min` + slice；未使用 `T.Parallel` 且给出理由 ✓。
- §6.3 不使用显式 `T.Pipelined` 的理由具体（单任务内顺序依赖链、无 K 维可重叠；重叠交编译器 auto multi-buffer），并有 Stage 4 回退条件（相对估算下界偏差 >2x 再评估显式双缓冲）✓。
- §6.4 尾块三类情形（`M % block_m`、`num_logical % num_kernels`、D 维无尾块）齐备 ✓。
- §7.2 同步点逐位置说明（MTE2→V、V 内部、V→MTE3），与 Developer 自动同步一致；无核间协作（行独立）→ 无需 flag/workspace ✓，与 §2.2/§2.3 一致。

### 6. 验证方案 — pass

- **Golden**：§8.1 含完整 PyTorch 参考实现草案，与 `examples/TileOPs/tests/ops/test_group_norm.py` 的 `ref_program` 逐字一致；签名 `(x, weight, bias, num_groups, eps)` 覆盖 3 张量输入 ✓；显式声明不复刻 NPU 算法（独立性）✓。
- **精度标准**：§8.2 按 dtype 给出 atol/rtol（fp32 1e-5/1e-5、fp16 1e-3/1e-3、bf16 1.6e-2/1.6e-2），与源仓 `_get_tolerances` 一致 ✓；并给出 fp32 最紧档与 O2 采纳项的因果关系（`vrsqrt` 2.87e-3 必违反 → 组合式 6.10e-8 留两数量级余量）✓。
- **L0 测试计划**：§8.3 共 17 例，覆盖三 dtype、block 整除与非整除（`D=13/196`、`M % block_m = 1` 的 (17,1152)/(257,4096)、`M < block_m`）、manifest 四档代表 shape、1D/3D spatial、非常量行、op 级端到端（含 per-channel affine 与 reshape）✓；通过条件（`atol + rtol·|ref|`）明确 ✓。
- **多输入组织**：验证侧 golden 与 L0-17（op 级端到端）按 3 输入组织；性能侧 §0.7 指向 `benchmarks/ops/bench_group_norm.py`（已实读：明确「`workloads_to_params` 不可用（单输入契约）」并走自定义 `_build_params` 读 `x_shape`+`num_groups`），manifest roofline `bytes = (2·N·C·spatial + 2·C)·elem_bytes` 已含 weight/bias 读；§1.6.0 另明确 kernel 级 vs op 级口径（kernel 不读 weight/bias，op 层 affine 是独立 pass +4.19 MB ≈ +3 µs）。**设计全篇不依赖单输入 `workloads_to_params` 契约** ✓。

### 7. 完整性与一致性 — pass

- 章节齐全：§0（0.1–0.7 全）/§1（含 1.6.0–1.6.3）/§2–§10 齐备 ✓。
- 无占位符类残留：全文未出现模板占位符与"待…"式空标记；§10.2 的「⬜ Stage 3 待实现」属交付清单的进度状态标记，非占位符 ✓。
- 同类引用真实：§3.5.3 及全文引用的 `examples/` 路径逐一 `ls` 存在（`examples/norm/{group_norm,layer_norm,instance_norm,ada_layer_norm_and_zero}.py`、`examples/TileOPs/…/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` 及其 `_DESIGN.md`、`tests/ops/test_group_norm.py`、`benchmarks/ops/bench_group_norm.py`、`manifest/norm.yaml`、`tileops/{ops,kernels}/norm/group_norm/…`）；`gate 1` 的 `S1-REF-PATHS` 通过 ✓。（**内容级**引用偏差见维度 8 warn 与非阻塞建议 #2/#3。）
- 内部一致：API 映射（§3.2）/内存规划（§4.3-4.5）/Tiling（§5.2/§5.5）/循环（§6.1）/§1.6.3 布局结论五处相互一致（buffer 形状恒为 `(block_m, D)`；`block_m` 表两处同值；`ALIGNMENT=1` 三处呼应）✓。
- 集成契约自洽（额外核对）：设计导出 `ALIGNMENT`/`_align_up`/`_group_norm_kernel` 与脚手架 wrapper 的导入名集合逐字一致；`integrate_kernel.py` 会把 wrapper 的 `from ._group_norm_fwd_kernels import …` 重写为 `from .group_norm_kernel import …`（baseline 块），故 `ALIGNMENT = 1` 在集成后生效、wrapper 的 `F.pad` 分支不触发——设计 §0.6 R7/§10.3 的这条链路可行 ✓。

### 8. 算法优化分析 — warn

**（1）§1.6 章节存在性**：四子节 1.6.0/1.6.1/1.6.2/1.6.3 齐全且各自有实质分析 ✓。

**（2）算法调研存在性与完整性（§1.6.0）**：调研四问齐全——R1 候选表 8 行**含基线候选 0**（源算法）且注明 ALG-layernorm/ALG-rmsnorm 命中行；R2 在线算法给出**明确结论**（"有" Welford/分块 running，附本仓 `examples/norm/ada_layer_norm_and_zero.py` 的 `clear=False` 现存形态——已实读核对 L65/75/173/183）并给域内不采纳的结构判据；R3 复杂度表四口径（元素级 op 数 / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）+ 基线在表 + 口径写明；R4 逐候选六维亲和评估；调研结论含选定算法与关键依据；「无更优替代」写明调研范围；深度与规约类算子匹配 ✓。**独立复算 R3 算术**：主选 `2·M·D·eb`（读 x + 写 y）与基线带宽项相同 ✓；基线另含 host `F.pad` 的 `2·M·D_pad·eb`（仅 pad≠0）✓（6272/7200 两档确实非 256 对齐）；D-tile 流式 x 读两遍 → `3·M·D·eb`（+50%）✓；Welford/矩式在驻留域内流量同为 `2·M·D·eb`（无 GM 收益）✓；**中间缓冲均由 UB 承载、无 GM 往返漏计** ✓。D-2 roofline 复算：`2×256×4096×2 = 4.19 MB`；流量项 4.19/1.356 = 3.1 µs ✓（33.9×40）；L2 口径 4.19/1.724 = 2.4 µs ✓；发射项 12 op × 4 task × 0.5 µs = 24 µs ✓；估算下界 ≈24 µs ✓，并诚实标注 0.5 µs/op 模型保守（ada 实测折算比 0.43）。

**（3）调研结论正确性（独立复核）**：
- 等价性：O1–O5 逐项独立推演成立（见下）；R1 候选 1（精确 D）的恒等式 `Σ_{j<D_pad}(x̃−m)² = Σ_{j<D}(x−m)² + pad·m²` 推演成立 ✓。
- 负向断言主动复核：C1 矩式的灾难性抵消否决——**机器证据可复现**（`verify_equiv.py` rejection 段：largemean-fp16 相对误差 6.018e+3 / 违反 1536 元素；padcancel-d2 矩式 7.806e1 vs 两遍式 2.110e-1），交叉证据 `algorithm-candidates.md` ALG-layernorm R1（矩式须评估 cancellation）与 queue VP-2026-0043 已核对存在 → 否决成立（**但一条辅助引用错误，见 (4)**）。C2 bf16 无 Vector 算术——已打开 `T.vmul/vadd/vsub/vdiv/vsqrt/reduce_sum` 文档 dtype 表核对 bf16 列均为 ×，成立 ✓。C3 跨 dtype `T.copy` 当 cast——`T.copy.md` §2.3 第 4 条（Developer 自动插 `hivm::VCastOp`，RoundMode=RINT 不可定制）已核对，成立（措辞可更精确，见建议 #7）✓。C4 kernel 内行广播 affine——`T.vbrc.md` §2.2.2（rank 一致 + src 维为 1）与 `T.vmul.md` §2.2.2（仅列 `[M,N]×[M,1]` 行广播，未列 `[1,N]×[M,N]`）已核对，两条实现路线的代价论证有据；`examples/norm/group_norm.py` L48-53 的标量索引形态**逐行核对属实** ✓。R2「在线变体存在但驻留域内无收益」为有结构依据的结论（非"无依据否定"），且给出驻留上限域界（D ≤ 9216 > manifest 最大 7200）✓。
- 与参考表/源码证据对照：ALG-layernorm 的 R2 明示「Welford 增量；分块 running」、R4 明示 row-reduction 迁移基准形态（全 N 驻留 + M 行块分核 + 尾块三件）与「N>11520 才需 N-tile 流式」——设计的 R2/R3/§5.5 与之同向 ✓，无「已知候选被漏掉」。
- 并列候选裁决：主选与基线/矩式/Welford/流式之间流量/数值/亲和的判定裕度大（最小 50% 访存差或数值 5 个数量级），不存在纸面并列 ✓。

**（4）调研→设计一致性**：§1.4（中心化两遍 + 全 D 驻留 + 精确 D + fp32 中转 + 单位仿射消除）与 §1.6.0 选定算法族一致；§3.1 以 §1.6.1 优化后公式（`r_D`、`sqrt(v)/v`、`d` 复用、kernel 无 affine）为唯一输入 ✓；§6 无逐元素标量循环、`T.Pipelined` 未用（与 1.6.2/1.6.3 一致）✓；迁移任务的 §0.5/§0.6 处置（R1-R7）与调研选族一致，§0.4 优化手段意图**逐项承接**（§0.6 末汇总表，无静默丢弃）✓。

**（5）数学等价优化完整性 + 等价性独立推演**（§1.6.1）：

| 项 | 原式 → 优化后 | 独立推演结论 | 四要素 |
|---|---|---|---|
| O1 | `S/D` → `S·fl(1/D)` | 成立：`D` 编译期常量；`fl(1/D)` 相对误差 ≤2⁻²⁴，`D` 为 2 的幂时精确；实测 max ulp 1、超容差 0（正确：整机重跑一致） | 齐 |
| O2 | `1/sqrt(v)` → `sqrt(v)/v` | 成立：实数域 `v^0.5/v = v^-0.5`；在设计的调用点 `v = var+eps ≥ eps > 0` 严格为正（不会出现 `0/0`），误差 6.10e-8 ≪ fp32 门禁 1e-5；替代近似指令 2.87e-3 的动机成立 | 齐 |
| O3 | epilogue 重推 `(cast(x)−m)` → 复用统计相 `d` | 成立：同一 fp32 值复用（上 cast 无损、mean 同一 buffer），逐位相同（实测 max ulp 0） | 齐 |
| O4 | 256-pad + 填充校正 → 精确 D | 成立：填充列贡献恰为 `pad·m²`，减回后实数域严格相等；**且方向更优**（基线在非对齐 D 下自身发生灾难性抵消，实测 4.572e2 vs 1.841e0） | 齐 |
| O5 | `(d·rstd)·1 + 0` → `d·rstd` | 成立：`a×1.0 = a`、`a+0.0 = a`（仅 `-0.0→+0.0` 符号位，`allclose` 无关，且 golden 同步产生 +0.0）；代表性前提已核（wrapper + op 层在该迁移路径恒传单位权重/零偏置，`GPU op 层 L237` 与 NPU wrapper `_identity_affine_for` 双证） | 齐 |

机器验证与内嵌表一致且全 EQUIV_PASS（见「机械复核」）；空真覆盖缺口已单列（建议 #4）。

**（6）向量化替代完整性**：§1.6.2 盘点 11 个计算点，与 §3.3 伪代码、§6 循环结构交叉核对——**无遗漏**：源 L2/S1/S2/E1 的四个 `T.Parallel` 元素级循环全部映射到 v-op 链（#1/#3/#4/#5/#6/#7），行归约 → `T.reduce_sum`（#2），搬运 → 双显式 slice（#11）；保留的 3 类（#8 任务映射索引、#9 动态尾块 `T.min`、#10 host 元数据）逐项给出**具体到本算子**的理由（block 级任务映射 / 每任务 1 次标量 + MTE2 窗口隐藏 / kernel 外），无"实现简单"类理由 ✓；替代 API 佐证齐全（`docs/Tilelang.language/` 具体文档 + 前作实调）✓。§6 复核确认 kernel 内不存在被 §1.6.2 判定"已向量化"的逐元素标量循环 ✓。

**（7）向量化轴与布局决策完整性（§1.6.3）**：规约类必选候选齐全——**水平归约 vs 垂直扫描两种 lane 映射分别枚举**（#1/#2，并说明与"选哪条轴"正交）+ D 维分块流式（#3）+ 转置布局（#4）+ 行块形状变体（#5）；轴质量评分逐项量化（整除性/尾 lane 浪费率 0%–23%/累加链形态/repack 代价 0/UB 影响）；弃选均有量化理由（#2 归约维 stride=D 无向量宽度可利 + TRAP-C10 跨步 gather 缺失；#3 访存 +50%；#4 repack 纯开销 + PL-1.3 host permute 106–147 µs）；**弃选论证前提已亲自核对 API 文档**：`T.transpose` 只支持二轴交换、且设计正确写成"核内 UB 级可行（非 GM 级重排）"（与 PL-1.1 的适用域一致），未把 I/O 契约（NCHW 行主序）当作内部布局依据（设计了核内重排候选并量化弃选）；**GPU 源码隐式轴选择已独立复核**（源 `threads=128` 映射行内元素 = 与 NPU 同轴但理由不同）✓；布局决策与 §3.3 buffer 形状（`(block_m, D)`）、§4.5 UB 预算、§6 循环结构三方一致 ✓。
**实验裁决模式**：判定裕度依赖的唯一未实证量为 auto multi-buffer 对本 kernel 三缓冲形态的 UB 占用口径 → 主选（默认开启，20 B/元素，D=4096 时 bm=2）+ 备选（关闭 auto multi-buffer，11.2 B/元素，bm 可至 4、`num_local_tasks` 4→2，含 UB/循环/发射项后果）+ 裁决计划（代表 shape = manifest 四档、指标 = msprof Task Duration ≥20 次采样去预热 + UB 报文反推、**阈值 = 备选 < 主选 × 0.95 且精度不回归**、允许 per-D 分派、结果回写本节）三件套齐全；并说明该决策为参数级、故不启动设计期探针（D-3 理由为结构级裕度 50% 访存差 ≫ 常数不确定区间）✓。

**warn 理由（证据精度问题，均不影响实质结论）**：
- (a) §1.6.0 R1#2 与 §1.6.1 C1 把 `examples/norm/group_norm.py` 当作"单遍矩式原型"引用（"高性能原型即此法"/"第 43 至第 46 行"），**实读证伪**：该文件 L43-46 是两个 reduce + `rstd = acc/N + eps; T.vrsqrt` 的两遍式，`_group_norm_kernel_high_perf`（L63-124）亦为两遍 N-tiled 流式；全 `examples/norm/` 无矩式实现（grep 无命中）。C1 否决本身由任务自有机器证据 + ALG-layernorm R1 + VP-2026-0043 支撑，故不阻塞（建议 #2）。
- (b) §9.2 把 `T.Parallel` 元素级循环的代价记为"本仓 argmax 案例实测标量化占壁钟 45%"：45% 实为 attention `vcmp int16` 全形态标量化（`attention.md` L123 / `traps-compiler.md` L177）的数字，argmax 案例是 `T.if_then_else` 融合形态；且 TRAP-parallel2d-bm-ge2-scalarize 的 2026-10-15 补充已收窄触发条件（普通逐元素 2D `T.Parallel` 仍正常向量化）。设计的替代决策本身正确（源的 L2/E1 含 `mean_val[i]`/`weight[j]` 标量索引形态，属该条目的触发形态），仅归因与图号需订正（建议 #3）。
- (c) §1.6.3 候选 4 引用 PL-1.1 的"实测 µs 级"未标注该条目已 `status: stale`（2026-10-14 收窄：仅"两边 ≥ 向量宽且为向量宽整数倍"的 2D 形态仍走 UB 内）；本算子的弃选理由（归约轴变跨步、零收益）不依赖该代价数字，故不阻塞（建议 #5）。
- (d) §4.5 fp32 平台值归因（见维度 2 warn；建议 #1）。

## 非阻塞改进建议

（结论为通过；以下为建议级，供 Stage 1 视情况吸收，不阻塞进入 Stage 3）

1. **§4.5：fp32 直连档平台值的归因与余量**。把 "fp32 直连档同为 20.00 B/元素" 改为直接引用原始实测报文（同族前作 `_ada_layer_norm_kernel.py` L138-139：`245760 B = 20*3*4096`（fp32, bm=3, N=4096）、`240000 B = 20*4*3000`（fp32, exact match）），并注明 TRAP-UB-multibuffer-inflation 条目正文对 fp32 记的是 26.00 B/elem（疑为 16 B/elem resident 的 v2 结构）。同时给 fp32 路径留余量：把工厂 guard 的 `_UB_MAX_ELEMS` 在 fp32 档下调（如 `196608 // 26 = 7561`）或对 `dtype == "float32" and block_m * D > 7561` 时降档，并核对 L0-6（(256,4096) fp32, block_m=2）——按 26 B/元素需 212 992 B > 192 KB，按 20 B/元素为 163 840 B。manifest 四档无此风险（fp32 仅 D=256），集成路径 wrapper 默认 `block_m=1` 亦安全。
2. **§1.6.0 R1#2 / §1.6.1 C1：订正矩式原型的引用**。`examples/norm/group_norm.py` 无矩式实现（L43-46 为两遍式 rstd，L63-124 为两遍 N-tiled 流式）；删除或改引真实来源（任务自有 `verify_equiv.py` rejection 段 + `algorithm-candidates.md` ALG-layernorm R1 + queue VP-2026-0043），保留机器证据为 C1 的主证据。
3. **§9.2 / §1.6.2：收窄 `T.Parallel` 的代价论断**。按 TRAP-parallel2d-bm-ge2-scalarize（含 2026-10-15 适用域收窄）改写触发条件（`T.if_then_else` 融合 / `(bm,1)` 直接索引等形态；普通逐元素 2D loop 正常向量化），并将 45% 数字归回 attention `vcmp int16` 案例；说明本设计替代源 L2/E1 的实际理由（E1 含 `mean_val[i]`/`weight[j]` 标量索引）。
4. **`verify_equiv.py`：标注不可判别行**。O1 组 `D ∈ {64,256,4096}`（`fl(1/D)` 精确）与 O4 组 `D ∈ {256,4096}`（pad = 0）两臂逐位相同、无判别力（O1 9/31、O4 6/31）；建议在脚本输出与 §1.6.1 表加一列/一句标注"non-discriminating (pad = 0 / power-of-two D)"，并在报告口径中把有效判别样本数写明（O1 ≥13、O4 ≥17，含全部角点）。结论不受影响（有效样本已给出超容差 0 与 max ulp 1/3 的证据）。
5. **§1.6.3 候选 4：补 PL-1.1 的 stale/适用域标注**（2026-10-14 重验：仅"两边 ≥ 向量宽且为向量宽整数倍"的 2D 形态仍 UB 内执行；池化族 3D 二轴交换降级为 UB→GM→UB）。本算法的弃选理由（归约轴变跨步）不依赖该代价，仅需与条目现状一致。
6. **§0.1：`_M_BLOCK_ALIGN` 归属措辞**。源中该常量由 `GroupNormNoAffineKernel.__init__`（L375）无条件使用（非"autotune 路径"），且其注释（L35-37）解释的是"行数对齐到候选 block_m 的公共倍数以保 `T.copy` 不越 M 界"；建议照实描述（不影响迁移范围结论）。
7. **§1.6.1 O1 的理由引用**：`CONST-npuir-f32-division-not-ieee-rounded` 的绕法一节明确"(M,1)/(M,1) 列形态实测逐位精确"且"勿换 reciprocal-mul"（针对字节级 tie 敏感的量化比对）。O1 本身已被机器验证（max ulp 1、超容差 0）且属容差内等价，建议在该处补一句区分（本算子的除法落在 `(bm,1)` 列形态；O2 的 `T.vdiv` 亦为该形态），避免与条目绕法口径表面冲突。
