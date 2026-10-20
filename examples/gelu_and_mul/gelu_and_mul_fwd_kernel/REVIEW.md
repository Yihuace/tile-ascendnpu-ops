结论: 通过

# gelu_and_mul_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

（本版为 v1 被 Stage 3 `[DESIGN_ERROR]` 拒收后的**重检（v2）**。上一版唯一阻塞根因——GELU 的 erf 建立在 NPU-native `T.verf` 上，而本工具链把它 lower 为**截断的 4 项 Maclaurin 级数**（仅 `|x|≲1` 有效）——**已彻底修复**：v2 删除 `T.verf`/`T.erf`，改用**显式、全域有效的 fp32 clamped deg-20 多项式**（`A=4`、`c0=1/A=0.25`）+ 参数饱和 + `vcmp/vselect` NaN 单点恢复，全 dtype 共用。本检视**亲自读源码复核工具链 lowering**（`codegen_npuir_api.cc` `VerfCodegen`，确认 v1 根因成立）、**独立重算多项式全域误差**（fp64 域 2.64e-8 / fp32 模拟 3.06e-7，与设计 `RANGE` 段 2.98e-7 同量级）、**独立重跑** `verify_equiv.py`（42 行全 `EQUIV_PASS`）与 `design_calc_check.py`。9 维度全部 pass（warn 不阻塞），0 阻塞级问题。给出 2 项**建议级**：§4.5 UB 总计行算术低报、erf 多项式「minimax」命名与 §4.7 最小二乘构造口径不符。）

## 检视元信息

- 文档: examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel_DESIGN.md
- 任务类型: 迁移（GPU TileOPs → TileLang-NPUIR，harness），mode=revision（v1 → v2；v1 冻结于 `history_version/design_v1.md`，v0 于 `history_version/design_v0.md`）
- 源算子: `/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`GeluAndMulFwdKernel` L386-402，`activation_func` L395-402）；`.../elementwise/_base.py`（`FusedGatedKernel` L419-497，`_get_effective_op_func` L458-467）；`.../elementwise/_builders.py`（`make_fused_gated_explicit` L686-708）；`.../elementwise/_erf.py`（`erf` L30-69）；`.../elementwise/_nan.py`（`keep_nan` L76-84）；`/mnt/workspace/TileOPs-main/src/tileops/ops/elementwise/activations.py`（`GeluAndMulFwdOp` L273-281）+ `.../ops/elementwise/_base.py`（`FusedGatedOp.forward` L153-157）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: EQUIV_PASS**（`python3 examples/gelu_and_mul/gelu_and_mul_fwd_kernel/verify_equiv.py` → `total rows: 42   EQUIV_PASS: 42   not-pass: 0`，`OVERALL: EQUIV_PASS`，exit 0）。与 §1.6.1 内嵌表**逐格一致**：item1 `cand vs 基线` fp32 ≤0.0931 / fp16 ≤0.0673 / bf16 ≤0.3109；item2 fp32 ≤0.0931 / fp16 ≤0.1024 / bf16 ≤0.4310；item1 `cand vs fp64` fp32 ≤0.0637 / fp16 ≤0.0309 / bf16 ≤0.0075；item2 `cand vs fp64` fp32 ≤0.0696 / fp16 ≤0.0309 / bf16 ≤0.0075；两 item 违反率 0、NaN 失配 0。**新增 `RANGE` 段**：全域 `|x|≤8` fp32 vs fp64 `max abs err = 2.980e-07`（< fp32 atol 1e-5）、`erf(+inf)=+1 / erf(-inf)=-1 / erf(±4)=±1` 精确饱和、`erf(NaN)=NaN`、`erf(0)=0`、`erf(-0.0)` signbit 保持——**全 OK**。**`BOUNDARY` 段**：三 dtype 下 `source=[nan, inf] golden=[nan, inf]`（`gelu(-inf)=NaN`、`gelu(+inf)=+inf`）——全 OK。**对照段 `counter fp32-gpu-poly`**（被拒绝的 GPU deg10 基线控制组，非采纳项）：`max_ulp=5.6170`、逐 shape 违反 `16×128=35 / 8×256=45 / 1×128=5 / 4×64=9 / 32×128=109 / 64×512=719 / 3×257=19`（累计 941），与 §1.6.1 L310 所列数字**完全相同**。
- **design_calc_check**（注：任务书给的调用形参 `--dir` 与工具签名不符，工具 `--help` 明确要求 `--design <DESIGN.md 路径>`；已用正确形参重跑）：
  ```json
  {
    "design": "examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel_DESIGN.md",
    "capacity_constants": {
      "ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0,
      "source": "pattern-library/constants.md CONST-capacity-910B2C"
    },
    "checks": [
      {"check": "ub_budget",   "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
      {"check": "l0c_budget",  "status": "skip", "detail": "no affirmative Cube-path API (T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)) in this design"},
      {"check": "core_split",  "status": "skip", "detail": "insufficient parsed dims {'M': 1, 'N': 14336, 'bn': 2048}; declared logical cores = 2048"},
      {"check": "r3_metrics",  "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
    ]
  }
  ```
  - `ub_budget` **skip → 人工复核**：逐行复算 §4.5 得 fp16/bf16 全显式 **71680B（70KB）**、fp32 全显式 **83968B（82KB）**、fp32 复用 **59392B（58KB）**，与设计所列 61440/81920/57344 分别低报 10240/2048/2048B（见维度 2 warn）；**修正后仍 ≪ 192KB（最大 82KB×1.12≈92KB），不溢出**，维度 2 阻塞判据满足。
  - `l0c_budget` **skip 合理**：纯 Vector，§3.2 无任何 Cube API。
  - `core_split` **skip → 人工复核**：§5.5 三要素齐全，逻辑核数 `M×ceil(N/block_N)` 与 `tiles_per_core=ceil(logical/40)` 算术自洽（见维度 3）。
  - `r3_metrics` **pass**。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 核对方式：**亲自完整 Read 源算子全链路源码**（`activations.py` L386-402、`_base.py` L419-497、`_builders.py` L686-708、`_erf.py` L30-69、`_nan.py` L76-84、`ops/elementwise/_base.py` L126-157、`ops/elementwise/activations.py` L273-281），与 §0.1–§0.7 逐项比对；并 Glob 核对全部同类引用路径真实存在。

| 检查项 | 结论 | 证据（源码位置 ↔ DESIGN 章节） |
|--------|------|--------------------------------|
| 语义理解正确性（**门禁要素 (a)**） | pass | 数学语义 `y=gelu(gate)*value`、门控切分 `gate=x[:,:N]`/`value=x[:,N:]`、单 `[M,2N]` 张量进/`[M,N]` 出、同 dtype、无规约——与源码 `_builders.py` L695-704（`x[by,col]`、`x[by,N+col]`、`y[by,col]`）、`_base.py` L458-467（`fused_op(gate,value)=act(gate)*value`）、`ops/elementwise/_base.py` L156（`n=x.shape[1]//2`）逐条一致（§0.1/§0.2）。erf 计算与 `_erf.py` L66-68 分派（fp32→`T.erf`、其余→clamp+deg10 多项式+`keep_nan`）一致。边界语义 `gelu(-inf)=NaN`（源式 `x*half*(one+erf_val)` 在 `x=-inf` 处 `-inf*0`，IEEE）已由 `verify_equiv.py` `BOUNDARY` 段三 dtype 机检（见机械复核）。 |
| 算法解读完整性 | pass | §0.3 步骤表覆盖源码全部计算语句（校验/`contiguous`、gate/value 列偏移取数、`gate_f32`、erf、窄 dtype 乘加、`*value`、返回），无遗漏/无臆造；数据流图覆盖全部 buffer；host 侧仅 `contiguous+校验+调 callable`（`ops/elementwise/_base.py` L153-157）一致。 |
| 优化手段覆盖完整性 | pass | §0.4 九项（`explicit_parallel` 直读、`threads×npt` 两级并行、同张量列偏移、clamps+降次多项式、fp32 参数域、`keep_nan`、`INV_SQRT2` 常数、`out_dtype` 分派、`@lru_cache`）逐项与源码对应并标硬件依赖；「无 swizzle/mma/warp shuffle/online」经源码核对属实。 |
| 耦合性判定合理性（**门禁要素 (c)**） | pass | §0.5 每条目四态处置+依据。**erf 求值耦合性判定已按 v1 教训重写**：源 `_erf.py` 的「clamps+降次多项式 / 精确 `T.erf`」判为**硬件强相关且 NPU 无可用等价物**（`fminf/fmaxf` 丢 NaN、`T.erf` 经 `math::ErfOp→hivm::VErfOp`），处置 = **重新设计**，理由直指工具链缺口——与 §9.1 源码证据一致，不再把 `T.verf` 误判为「可整体替代」。 |
| NPU 重设计可行性（**门禁要素 (b)**） | pass | §0.6 R1（二维 grid+`T.Parallel` → 一维 `T.Kernel`+`block_N` 折叠+UB 向量化）与 R2（erf 求值 → 显式 clamped deg-20 多项式）均含「源方案 → NPU 新算法 → 语义保持论证」三要素；R2 新算法满足 NPU 约束（纯算术 Vector、UB 驻留、32B 对齐、静态边界），**并显式覆盖全部实轴**（`|z|≤4` 多项式逼近、`|z|>4` clamp+clip 精确饱和到 ±1）。语义保持论证含容差内等价、精度路径澄清、边界核对（NaN/±inf）、验证方式四项，且由 §1.6.1 机器验证支撑。 |
| 设计与决策一致性 | pass | §1.4/§1.5/§3.1/§3.2/§4.4/§5/§6 描述的是 §0.5/§0.6 决策后的 NPU 算法（一维 Kernel、UB v-prefix 链、显式多项式 erf、fp32 单 cast）；§3.3 伪代码**无 `T.verf`/`T.erf` 残留**（仅 §0.0/§3.4/§9 以「已证伪、禁用」语境出现）；§6 无 §0.5 已判重设计的 `threads/npt` 残留。 |
| golden 独立性 | pass | §8.1 golden（`x[...,:half].float()`、`F.gelu(gate.float())*value.float()`.to(dtype)）与源仓测试基准 `test_gelu_and_mul.py::ref_program`（L63-69）逐字一致，仅依赖 torch 原生 `F.gelu`（erf 精确形式）+乘法，**未复刻** NPU 的多项式/分派/persistent 结构。 |

> **一处需说明的既存物（非设计缺陷）**：工作区内的 `gelu_and_mul_fwd_kernel.py` 仍是 v1 遗留、第 176 行仍调用 `T.verf`。DESIGN §10.2/§10.4 已明确标注该文件「**Stage 3 重做**（v1 版因 `T.verf` 缺陷须重写）」。本检视对象为 DESIGN.md，Stage 3 必须从 §3.3 的显式多项式重新实现——此处仅作交接提醒，不构成设计阻塞。

### 1. API 可行性 — pass

- §3.2/§3.4 列出的每条 v-prefix API 均在 `docs/Tilelang.language/` 有真实文档：`T.vmul`/`T.vadd`/`T.vsub`（数学操作/）、`T.vclamp`/`T.vcmp`（比较操作/）、`T.vselect`（条件操作/）、`T.vcast`（数据类型转换操作/）、`T.copy`/`T.alloc_ub`（内存操作/）——已逐一 Glob 确认存在。
- **调用签名逐一核对**：`T.vcmp(A,B,C,cmp_mod)`（文档确认 A/B/C/mode 次序）与 §3.2「`T.vcmp(g32,g32,nan_mask,"ne")`」吻合；`T.vselect(Cond,A,B,Out)` 与「`T.vselect(nan_mask,g32,e32,e32)`」吻合（mask 真取 `g32`=NaN，否则取 `e32`）。文档载明 `vcmp` 输出 bool **不可写回 GM**，设计 §4.3 明确 `nan_mask` 片内使用、不写回——一致。
- **dtype 约束处理正确**：`T.vmul`/`T.vsub`/`T.vclamp` 文档 dtype 表对 bf16 标 `×`，设计 §3.3 对 fp16/bf16 先 `T.vcast` 升 fp32、全链在 fp32、末尾单 `T.vcast` 降回——bf16 无原生 v-prefix 算术的风险已被覆盖。
- `T.vbrc` 常数填块对齐 hardsigmoid 先例；`T.Kernel(一维, is_npu=True)`、`T.ceildiv`/`T.min` 均有同族先例。**无 `T.verf`/`T.erf`**。

### 2. 内存层级规划 — warn（非阻塞）

- 搬运路径完整：§4.4 给出 `GM→UB`（`T.copy` 两段连续切片）→ UB 内 v-prefix 链 → `UB→GM` 的纯 Vector 两级路径，**无 GM→L0 越级、无中间 GM 往返、无 Cube/L0C**（正确）。
- 每个 buffer 均标注层级（§4.3：全部 UB；`nan_mask` 标注片内 bool）。
- **UB 预算（阻塞判据）满足**：逐行独立复算 §4.5 得 fp16/bf16 全显式 **71680B（70KB）**、fp32 全显式 **83968B（82KB）**、fp32 复用 **59392B（58KB）**，×1.10–1.12 上偏后最大 ≈92KB，**均 < 192KB**（`CONST-capacity-910B2C`），无溢出；`design_calc_check` 的 `ub_budget` 为 skip（解析器未识别表结构），已人工复核替代。
- **建议级（warn）**：§4.5 三个「总计」行与同表逐行列值不自洽——fp16 行少计 `e32`(8192B)+`nan_mask`(2048B)=10240B（61440 vs 71680）、fp32 全显式行少计 `nan_mask`(2048B)（81920 vs 83968）、fp32 复用行少计 `nan_mask`(2048B)（57344 vs 59392）。该低报**不影响结论**（修正后仍 ≪192KB），但 §4.5 表内 `nan_mask` 行、`e32` 行与总计行应重算对齐（建议 Stage 1 顺手修正）。

### 3. Tiling 策略 — pass

- §5.2 给出 `block_N=2048` 及理由（取自 GPU fp16/bf16 折叠值 `threads×npt=256×8`、32B 对齐、整除向量宽、整除 manifest `N`），§5.3 约束（对齐/UB/L0）逐条成立。
- **分核策略三要素齐全**（§5.5）：① 逻辑核数 `M×ceil(N/block_N)`；② 物理核数 `NPUUtils.get().get_aicore_num()` **实查=20**，纯 Vector ×2=40（记录查询代码与返回值，且指出与 `CONST-aicore-910B2C` 记 24 不一致、以实查为准——与 pattern-library 该条目的 hardsigmoid 实测注记一致）；③ 规模判定与方案（`≤40` 单 tile 无需适配 / `>40` persistent 串行 `tiles_per_core=ceil(logical/40)`，循环边界静态）。
- **算术独立复算**（`core_split` 机械 skip → 人工）：smoke `16×1=16`→num_cores16/tiles1 ✓；prefill `2048×7=14336`→tiles `ceil(14336/40)=359` ✓；decode `1×7=7`→num_cores7/tiles1 ✓；qwen3 `32×6=192`→tiles `ceil(192/40)=5` ✓——与 §5.5 表逐行吻合，且与 manifest `elementwise.yaml` L420-428 的 workload 一致。

### 4. 技术约束检测 — pass

一维 `T.Kernel`（无三维 Kernel 约束冲突，§3.5.1 已列）；无 GEMM（非整除 padding/crop 不适用）；无 Cube/L0C（L0C 溢出不适用）；`T.copy` 尾列用双显式切片 `out_ub[0:tail]→y[m,n0:n0+tail]`（§6.4，规避 base+size 越界写）。

### 5. 循环与同步 — pass

§6.1 明确 `T.Kernel`（核级）/`T.serial(tiles_per_core)`（核内 persistent，静态边界）/v-prefix 整块（元素级向量化）三层；§6.4 尾块 `tail=T.min(block_N,N−n0)` + `task` guard；§7 自动同步（Developer 模式），无手动 `set_flag/wait_flag`，与 §2 模式选型一致。`T.serial` 顺序 + Stage 4 可选 `T.Pipelined` 双缓冲的说明合理。

### 6. 验证方案 — pass

§8.1 golden 为 PyTorch 参考草案（独立于 NPU 算法，见维度 0）；§8.2 精度标准 fp32 1e-5/fp16 1e-2/bf16 1.6e-2，与源仓 `_get_tolerances` 一致；§8.3 给出 L0 门槛（manifest 主 workload `2048×14336` fp16/bf16、decode `1×14336`、smoke `16×128` 三 dtype，`assert_close`）。**特别提醒合理**：Boundary 须覆盖大 `|gate|` 饱和（`gelu(-inf)=NaN` 不得断言 0）、`|gate|>1` 全域精度（v1 盲区，目标 <1e-6；实测 2.98e-7）、NaN→NaN。fp32 1e-5 容差对多项式误差 3e-7 的裕度经 `verify_equiv.py` 角点/随机 0 违反佐证。

### 7. 完整性与一致性 — pass

迁移类 §0 且 0.1–0.7 小节齐全；§1–§10 模板章节齐全；**无 `TODO`/`待补充`/`{placeholder}`**；§3.5.3 引用的 5 个同类实现路径与全部 `docs/` 路径经 Glob 核实**真实存在**；API 映射（§3.2）、内存规划（§4.3-4.5）、Tiling（§5.2）三处 buffer/形状/分核描述相互一致，无矛盾。

### 8. 算法优化分析 — pass

> 核对方式：以 §1.6 为对象逐项**独立推演/复核**，并交叉核对 §3.1/§6 双向一致；**调研结论与弃选论证前提均亲自查证**（读 `algorithm-candidates.md`、`algorithm-research.md`、`pattern-library/`、工具链 lowering 源码、API 文档）。

- **§1.6 章节存在性**：§1.6.0 调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代 / §1.6.3 向量化轴与布局决策四子节齐全。
- **算法调研存在性与完整性**：§1.6.0 四问齐全——R1 候选表**含基线候选**（候选 0=源算法）并逐候选初判；R2 在线算法给出**明确结论「无（单遍即最优）」**且有结构依据（逐元素无跨元素依赖，与 `ALG-elementwise` 命中行「R2：逐元素映射天然单遍」一致）；R3 复杂度**四口径齐全**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表）；R4 硬件亲和逐候选评估（单元/容量/对齐/静态边界/流水/跨核）。「无更优替代」写明调研范围。
- **调研结论正确性（独立复核）**：
  - **等价性推演**：选定算法 `y=g·0.5·(1+erf_poly(g/√2))·v`（fp32 域求值+末尾单 cast）与源语义 `y=gelu(gate)*value` 数学等价，差异仅在 erf 求值精化与舍入路径——独立成立。
  - **复杂度算术独立复算**：`flops=6·M·N`、`bytes=3·M·N·elem`（读 x 的 2MN + 写 y 的 MN）、遍数 1、中间缓冲 O(block_N)——与 manifest `elementwise.yaml` L430-434 的 roofline 逐字吻合；**无中间缓冲 GM 往返漏计**（纯逐元素无中间张量）。
  - **负向断言逐条复核**：①「无在线变体」——对照 `algorithm-candidates.md` `ALG-elementwise`（R2 天然单遍）成立；②「无更少 op 的代数形式」——候选 3（因子重排）/5（合并）确无 op 削减，成立；③ 候选 4 tanh 近似被拒——**核对源仓**：`GeluTanhAndMulFwdKernel` 是**另一个** manifest 算子（`GeluAndMulFwdOp` 契约明确要求精确 erf 形式），淘汰正确；④ 候选 8 `T.verf` 被拒——**亲自读工具链源码证实**（见下）。
  - **亲和性淘汰佐证**：容量/带宽淘汰引用 `pattern-library/constants.md`（`CONST-capacity-910B2C` 192KB、`CONST-mte2-degradation`、`CONST-vector-launch-overhead` ~0.5µs/op）；D-2 roofline 常数（176M 档 19.5 GB/s/core×40≈780GB/s→226µs；发射 ~55 op×0.5µs≈27.5µs）算术自洽。
  - **并列候选裁决**：无并列争议（候选 1 是全 dtype 唯一正确解），§5.4 另列 Stage 4 备选（窄 dtype deg10、exp 型 minimax）作为出口。
- **弃选论证前提核对（强制，含 `docs/` 亲自查证）**：§1.6.3 候选矩阵 5 项。候选 2（host 切分）前提 =「host 不得改输入真实内容」——**打开 `ascend-constraints.md` §4** 核对属实（host 禁止改动输入张量真实内容/触发 padding，仅允许视图）；候选 3（核内 `T.transpose`）前提 =「核内 UB 级二轴交换、~µs 级零收益」——**打开 `docs/Tilelang.language/创建操作/T.transpose.md`** 核对（确认底层 `hivm.hir.vtranspose` 仅支持两轴交换、核内 UB 操作）并与 `PL-1.1-transpose-chain` 实测（~5.4µs 可忽略，但本算子纯逐元素无归约轴可借）对照，**设计论断与文档/实测一致**；候选 4/5 前提（UB 压力 / 数学同构）合理。
- **数学等价优化完整性 + 等价性论证独立推演**：§1.6.1 采纳 2 项，均具「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素。**item1（erf 求值升级 deg10→deg20）**：独立推演——多项式在 `|z|≤4` 一致逼近 erf、`|z|>4` 经 `t=±4, P(0)=c0=1/A=0.25` 精确饱和到 ±1，容差内等价成立；**item2（全 fp32 链+末尾单 cast）**：独立推演——两式均实现同一数学映射、仅舍入路径不同（差 ≤1 ulp），新式对齐 torch fp32 opmath golden，等价成立。两项均由 `verify_equiv.py` 机器验证（**重跑 42 行全 `EQUIV_PASS`**，与 §1.6.1 内嵌表逐格一致）。
- **deg-20 多项式独立 sanity check（额外焦点）**：本检视另写独立脚本重算——`c0=0.25=1/A` 精确、`t·P(0)=1` 精确（t=4）；**fp64 域 `|x|≤8` max abs err 2.64e-8，fp32 模拟 max abs err 3.06e-7**（与 `RANGE` 段 2.98e-7 同量级，差异仅网格）；系数幅度 ≤1.97（power 基在 `w∈[0,1]` 条件良好）；`P(w)` 对 `erf(A√(1−w))/(A√(1−w))` 拟合残差 ~8e-9——**多项式声称的域/精度可信且已被 `RANGE` 机器验证**。**前提核对**：`T.verf` 确被证伪——亲自读 `codegen_npuir_api.cc` L2415-2434 `VerfCodegen`，确认把 `tl.npuir_verf` lower 为 `(2/√π)(x−x³/3+x⁵/10−x⁷/42)`（截断 Maclaurin）；并核对 `HFusionToHIVM.cpp` L249（`math::ErfOp→hivm::VErfOp`）与本工具链致命中转，**设计对 v1 根因与耦合性缺口的引用正确**。全篇无对任何失效 API 的残留依赖。
  - **建议级（warn）**：§0.0/§0.6/§1.6.0 通篇称该多项式为「clamped **minimax** 多项式」，而 §4.7 自述构造方式为「在 `w∈[0,1]` 稠密网格上对 `erf(4√(1−w))/(4√(1−w))` 做**最小二乘**（power 基）」——最小二乘 ≠ minimax。数值精度已机检达标，命名与构造口径不一致属文档瑕疵，建议统一表述（如「clamped least-squares/minimax 多项式」或将构造改为真正 minimax 拟合），不影响任何判定。
- **向量化替代覆盖完整性（与 §3.3/§6 交叉核对）**：§1.6.2 覆盖全部核内逐元素计算点（2 次升 cast、erf 参数乘、参数 clamp、`w` 链、deg20 Horner、`t·P`、clip、`vcmp+vselect` NaN 恢复、`+1`、`×0.5`、`×gate`、`×value`、降 cast）——全部映射到 `docs/` 有据的 v-prefix API；保留 3 处标量（任务行列解算、尾列大小、分核/Op 元数据）均为非逐元素热点且有具体理由，与 §3.3 伪代码、§6 循环结构交叉核对一致（§6 无逐元素标量循环，§1.6.2 判为向量化的点不再出现标量形态）。
- **向量化轴与布局决策完整性**：§1.6.3 候选矩阵枚举有意义候选（含基线行优先方案）并逐项量化评分/弃选理由，无「未考虑」式论证；**迁移对照正确**（GPU thread/warp 轴与 NPU Vector 列轴不对应，source 轴选择作输入非结论）；布局决策与 §3.3（1-D 列块 buffer）、§4（UB 预算）、§6（向量维=行内列轴）一致。
- **与下游一致**：§3.1 公式拆解以 §1.6.1 优化后公式为输入（`erf_poly`）✓；§6 循环结构与 §1.6.2 结论一致 ✓；§0.5/§0.6 处置与 §1.6.0 选定算法一致、源优化手段（#4/#6/#8 等）意图承接有逐项说明、无静默丢弃 ✓。

---

**门禁自检**：结论字面量存在且与详情一致（全维度 pass/warn，无 fail）✓；迁移 9 维度齐全（含维度 0 源码证据、维度 8 独立推演/复核证据）✓；机械复核段含 `verify_equiv` 重跑与 `design_calc_check` JSON 摘要及 skip 项人工结论 ✓；无占位符 ✓；通过结论下未输出问题清单章节 ✓。
