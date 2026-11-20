# _make_leaky_relu_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness）
- 源算子: /home/developer/TileOPs/tileops/kernels/elementwise.py（工厂 `_make_leaky_relu_kernel` L2427–L2478；`LeakyReluFwdKernel` L2481–L2493）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: PASS。`python3 examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py` 实跑输出 `RESULT: ALL 20 ITEMS EQUIV_PASS`（exit 0），逐行与 DESIGN.md §1.6.1 内嵌结果表一致（samples 1,048,611；#1 fp32-transit vs SRC input-dtype fp16 `max_abs=4.883e-04 / vratio 0.2597`、bf16 `3.906e-03 / 0.1297`；#1/#2 vs GOLDEN 三种 dtype 均 `max_abs=0.000e+00`；回退 where s=0.5/2.0 精确）。非门禁诊断行亦一致：`max-form slope=2.0 max_abs=6.550e+04 viol=1048591`、`slope=100.0 max_abs=6.485e+06`。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md` 实跑 JSON：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"）——机械跳过，**人工已核**：fp32 bs=1024 合计 `4+4+4+4+4+1 = 21 KB`、fp16/bf16 bs=2048 合计 `4+4+8+8+8+2 = 34 KB`，×1.7 最坏膨胀 = 36 / 58 KB < 192 KB ✓（§4.5 表与 §4.3 buffer 表口径自洽）。
  - `l0c_budget`: pass（`block 40x25 x fp32 accum = 3.9 KB`；实为纯 Vector，无 L0C，不构成约束）。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = 1"）——机械跳过，**人工已核**：§5.5 三要素齐全——逻辑核数 `ceil(N/bs)` = 512~16384、物理核数实查（`get_aicore_num()=20`、`get_aivector_core_num()=40`，见 `CONST-aicore-910_9362`）、极大规模判定 + `T.serial(ceildiv(n_num,core_num))` 静态上界网格跨步 + `if task < n_num` 守卫；残留不均显式记录（smoke-1m +1.56%）。
  - `r3_metrics`: pass（"all four R3 metrics present"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：**亲自 Read 源算子 `elementwise.py` L2426–L2493 全文**，与 DESIGN.md §0.1–§0.7 逐项核对；并复核 `verify_equiv.py` 与设备探针 `design_probe_leaky_relu.py` 的实测数值。

- **语义理解正确性 — pass**。§0.1 数学语义 `y = x if x>0 else s·x` 与源 `T.if_then_else(val > zero, val, slope * val)`（源码 L2473）一致；dtype 语义核对正确：源**非 fp8 分支 `slope = T.cast(negative_slope, val.dtype)`（L2472）、`zero = T.cast(0, val.dtype)`（L2471）→ 全程在输入 dtype 域**，§0.1 表述准确。fp8 分支 `accum = _fp8_accum_dtype_str()`（L2441）实为 `"float16"`（源 L210–L212），§0.5 描述一致。边界语义 `x=0` 两支同值、`NaN→NaN`、`-inf→s·(-inf)=-inf` 与 torch 行为一致。**核心分歧点（计算域）核对成立**：契约 golden `F.leaky_relu(x.float(), 0.01).to(dtype)`（`examples/TileOPs/tests/ops/test_activation.py` L152–L153，与 GPU 源 `test_activation.py` L222–L228 同式）确在 **fp32 域**计算，而源非 fp8 分支在输入 dtype 域计算——§0.1 指出该差异并量化，属实。
- **算法解读完整性 — pass**。§0.3 步骤表（L110–L117）覆盖源非 fp8 分支全部计算语句：`alloc_fragment`（L2466–L2467）、`T.copy` 载入（L2468）、`T.Parallel` 内 `val/zero/slope` + `if_then_else`（L2469–L2473）、`T.copy` 写回（L2474），无遗漏/无臆造；数据流图覆盖 x_reg/y_reg 两 buffer；host 侧逻辑（`N_total` flatten、`inplace` copy_、`lru_cache` 特化）已列出且与 §0.1 一致。
- **优化手段覆盖完整性 — pass**。§0.4 列 7 项（#1 `register_copy` fragment 链、#2 block×thread 两级并行、#3 输入 dtype 域算术、#4 strategy-aware `npt`、#5 无 autotune、#6 3-op 计算核、#7 `lru_cache` 特化），与源码逐项对得上（#4 依据 `ParametricUnaryKernel.default_config` 源 L2404–L2411：fp32 npt=4、非 fp32 npt=8、`_DEFAULT_THREADS=256` → block_size 1024/2048）；未把实现层优化误判为算法层。
- **耦合性判定合理性 — pass**。§0.5 四态处置齐备且计数与调度说明一致：**保留 5**（语义 / 算法结构 / #5 / #7 / host）、**等价替换 3**（#3→fp32 中转、#6→vmul+vmax、#4→block_size 折叠）、**重新设计 2**（#1→UB+v-prefix、#2→一维核+网格跨步）、**舍弃 1**（fp8 分支，依据 manifest `SUPPORTED_DTYPES` 仅 fp16/bf16/fp32）。处置与 NPU 能力映射不矛盾：`threads=`/fragment 无 NPU 等价判「重新设计」正确。
- **NPU 重设计可行性 — pass**。§0.6 R1/R2/R3 每项均含「源方案 → NPU 新算法 → 语义保持论证」。R2 的 `max(x,s·x)` 语义保持论证经**独立数学推演核对**：`0<s≤1` 且 `x>0` 时 `s·x≤x`，round-to-nearest 下 `round(s·x)≤x`（`v≥x` 时最近可表示数不低于 `x`）→ `max=x`；`x≤0` 时 `s·x≥x` → `round(s·x)≥x` → `max=round(s·x)`，与 golden 负分支同一次 fp32 乘——**推演成立**。R3 计算域论证由设备探针复证（见维度 8）。R1 满足 NPU 一维 Kernel/UB 层级约束。
- **设计与决策一致性 — pass**。§1.4/§3.1/§3.3/§4/§6/§8 一致描述 §0.5/§0.6 决策后的算法（fp32 中转 + max/where 双形态 + 一维核网格跨步），无「§0 判重设计、§3 照抄源方案」的脱节。
- **golden 独立性 — pass**。§8.1 golden 以 §0.1 源算子语义为唯一依据（`F.leaky_relu(x.float(), negative_slope).to(dtype)`，直接调用库实现），**未复刻 §0.6 的 vmul+vmax 向量链**，验证独立性成立。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 API 均有 `docs/Tilelang.language/` 佐证 + 目标设备实跑：
- `T.vmul(src1, src2/dst, dst)`——`数学操作/T.vmul.md` §2.1/§2.3（scalar src2 广播、dst 可与 src 同 buffer、标量操作数须在 `T.Kernel` 内 `T.Scope` 外、操作数须在 UB）；§2.2.1 dtype fp16 ✓/fp32 ✓/**bf16 ✗**——佐证 §0.6 R3 的 bf16 强制中转。
- `T.vmax(src0,src1,dst)`——`比较操作/T.vmax.md` §2.1（两入参均 `tensor`，**无 scalar 形态**）§2.2.1（fp16/fp32 ✓、bf16 ✗）——佐证 R-11「vmax 无标量广播」。
- `T.vcmp(A,B,C,cmp_mod)`——`比较操作/T.vcmp.md` §2.1（`"gt"` 合法）§2.3（**输出 bool 不可写回 GM**）——佐证 R-10。
- `T.vselect(Cond,A,B,Out)`——`条件操作/T.vselect.md` §2.1（fp16/fp32/bf16 ✓）。
- `T.vcast(src,dst,round_mode)`——`数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32`/`bf16→f32` 仅 `rint`；`f32→f16/bf16` 支持 `rint`）——§3.3 统一 `round_mode="rint"` 合规。
- `T.copy`(slice) / `T.alloc_ub` / `T.fill` / `T.Kernel(n,is_npu=True)`+`T.serial`——先例 `examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`、`examples/bitwise_not/.../bitwise_not_fwd_kernel.py`（均实存，Glob 核对通过）。
- **v-prefix 优先**：全用 v-prefix，无 `npuir_xxx`。**模式匹配**：Developer 模式自动同步，无手动 set/wait_flag。
- **负向断言核对**：`ls docs/Tilelang.language/数学操作/` 实枚举（vabs/vadd/vcos/vexp/vexp2/vln/vlog2/vmul/vpow/vrelu/vsin/vsqrt/vsub/vtanh/verf/vrsqrt/vsigmoid/vdiv）**确无 `vleaky_relu`**——§3.5.1「无原生 vleaky_relu 元算」成立。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出 GM→UB→GM 双路径（fp32 零中转 / fp16·bf16 经 `vcast` 中转），纯 Vector 无越级路径。
- **UB 预算（机械 skip → 人工复核）**：§4.3 + §4.5 逐 buffer 标注层级（全部 UB），合计 fp32 21 KB、fp16/bf16 34 KB，×1.7 最坏膨胀 = 36/58 KB < 192 KB ✓。`cond`（bool）与 vcmp doc §2.3 同为片上使用，从不写回 GM。
- **L0C 容量**：n/a（无 Cube）。
- **Buffer 层级标注**：§4.3 六 buffer 均标 UB。

### 3. Tiling 策略 — pass

- **Block 划分**：§5.2 `block_size` = K11 折叠值 fp32 1024 / fp16·bf16 2048，与 wrapper `_DEFAULT_BLOCK_SIZE`（实读 `leaky_relu.py` L86–L90）一致，理由（可追溯/对齐/容量）具体。
- **对齐约束**：block_size×elem_bytes = 4096 B，32 B 整数倍；尾 lane 浪费 0%（§5.3）。
- **非整除处理**：§5.4 用 K 内动态尾块 `tail = T.min(block_size, shape-off)`，不改动 host 张量、不做 padding（合规 `ascend-constraints.md` §4）；manifest workload 均整除，测试 4,000,000 % 2048 = 1216 触发尾块，已覆盖。
- **分核策略三要素（机械 skip → 人工复核）**：① 逻辑核数 `ceil(N/bs)` 计算并分 workload 列表（512~16384）；② 物理核数**实查**（`NPUUtils.get().get_aicore_num()`=20、`get_aivector_core_num()`=40，且 `pattern-library/constants.md` 的 `CONST-aicore-910_9362` 独立记录「实查 20 AI Core / 40 Vector Core」，2026-10-08 dev root build 实查，与本设计一致）；③ 极大规模判定 + 核内 `T.serial` 网格跨步（循环上界静态）、残留不均显式量化（+1.56%/+0.10%）。核数与 block 取值自洽（40 固定启动 + 网格跨步）。

### 4. 技术约束检测 — pass

- 无三维 Kernel（一维 `T.Kernel(40, is_npu=True)`）。
- 无 GEMM/非整除 GEMM、无 L0C 溢出。
- bf16 v-prefix 数学 API 缺失已识别并给出 fp32 中转处置（§3.5.1）；`vmax` 无标量广播、`vcmp` bool 不可写回 GM、条件 `T.alloc` 非法、v-op 须整 buffer，均逐条给出规避方案（§3.5.1 / §9.1）。

### 5. 循环与同步 — pass

- **循环结构**：§6.1 明确核级 `T.Kernel(core_num,is_npu=True)`、tile 级 `T.serial(ceildiv(n_num,core_num))`（网格跨步）、元素级向量化（无 `T.Parallel` 标量循环）；`T.Persistent` 用 `T.serial` 网格跨步等价实现并说明。
- **同步匹配**：§7 Developer 自动同步，数据依赖线性（copy-in → [vcast↑] → 链 → [vcast↓] → copy-out），无手动 set/wait_flag，无 CV 融合（§7.4 判定非 MixCV，符合 AGENTS.md MixCV 触发规则）。
- **尾块处理**：§6.4 引用 §5.4 `tail` 动态裁剪。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 `golden_leaky_relu` = `F.leaky_relu(x_f32, s).to(x.dtype)`，与契约（NPU `test_activation.py` L152–L153）逐字一致，独立于 NPU 向量链。
- **L0 测试计划**：§8.3 含 10 项（各 dtype randn、角点、负分支敏感段、±inf/NaN、x=0、非整除尾块、非默认 slope 回退），规则 shape/dtype/判定齐全。
- **精度标准**：§8.2 fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与 `_get_tolerances`（tests L37–L43）一致；误差预算以设备实测给出（fp32 0.0、fp16 randn 3.05e-5、bf16 2.44e-4），余量 ≥30×/≥60×。

### 7. 完整性与一致性 — pass

- **章节齐全**：§0（0.1–0.7 全）+ §1–§10。
- **无占位符**：全文无模板占位符号、无待办标记、无未完成内容标记（实检确认）。
- **同类引用真实**：§3.5.3 引用的 `examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`、`bitwise_not_fwd_kernel.py`、`elu/_make_elu_kernel/DESIGN.md`、`expm1`/`erf` DESIGN、`leaky_relu/` 目录均实存（Glob 核对）。
- **内部一致**：API 映射、内存规划、Tiling 三处一致（fp32 无中转/ fp16·bf16 经 xf 中转、全 40 核网格跨步）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在**：含 1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 布局决策四子节。
- **算法调研存在性与完整性（独立复核）**：R1 候选表 **含基线**（源 where 形式），覆盖 `max` 形式、relu/min-max、`x·select(1,s)`、slope 恒等/relu、fp32 中转、超越函数逼近 6 候选；R2 **明确结论「无在线化收益」+ 结构判据**（无 running 统计量/无跨元素依赖），与 `algorithm-candidates.md` 的 `ALG-elementwise` R2「逐元素映射天然单遍」一致；R3 四口径（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）齐全，**基线在表**且在设计中一致；R4 逐候选评估（单元匹配/容量/对齐/静态边界/流水/跨核）。调研深度与算子复杂度匹配（含多等价形态 + 计算域分歧，达完整调研深度合理）。
- **调研结论正确性（独立复核）**：
  - 等价性：选定「max 形式 + fp32 中转」经独立数学推演（见维度 0 R2）与 `verify_equiv.py` 重跑（20/20 EQUIV_PASS，与内嵌表一致）双证。
  - 复杂度算术独立复算：基线 where 3 op、max 2 op、relu/min-max 4 op 均正确；访存恒为 **2·N·elem_bytes**（与 manifest roofline `bytes=2*N*elem_bytes` 同口径，**无中间 GM 往返**）；扫描遍数 1；缓冲峰值 O(block_size)——**无漏计中间缓冲 GM 往返**；`r3_metrics` 机械 pass。
  - 负向断言逐条复核：①「无在线变体」→ 逐元素无跨元素依赖，与 `ALG-elementwise` R2 一致；②「无原生 vleaky_relu」→ `docs/数学操作/` 实枚举确认；③「无更优替代」→ §1.6.0 调研范围列 algorithm-candidates/`ALG-elementwise`（含 `TRAP-fp16-opmath-golden`「sub-fp32 逐元素算子 fp32 中转链对齐 torch golden ≤1 ulp」——**该实测直接支持本设计 fp32 中转决策**，实读 traps-runtime.md L46 条目确认）、examples 全系、pattern-library 命中项。**未发现漏掉的已知候选**。
  - 亲和性淘汰依据：bf16 淘汰引用 doc §2.2.1（实核）；容量淘汰引用 `constants.md` 条目（`CONST-capacity-910B2C` 实存，并标注 stale + 设备差异）。
  - 并列候选裁决：默认 `s=0.01` 命中 max 路径；`s≤0/s>1` 有 where 回退，无纸面拍板未实证常数的单选项。
  - `TRAP-fp16-opmath-golden` 双触发条件（bf16 无 v-prefix 数学 API / fp16 需对齐 torch CPU golden）与 §0.6 R3 论证一致。
- **调研→设计一致性**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均以 §1.6.0 选定算法（max + fp32 中转 + 编译期特化）为输入；迁移任务源优化意图承接逐项说明（§0.6 末尾承接表），无静默丢弃。
- **数学等价优化完整性**：§1.6.1 三个采纳项（#1 fp32 中转、#2 max 形式、#3 CSE/原地复用）均有「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；E1/E2 拒绝项有理由记录。
- **等价性论证正确性**：逐项独立推演成立；`max` 形式 guard `0<s≤1` 经非门禁诊断（slope=2.0/100.0 max_abs 6.55e4/6.485e6，与 DESIGN 表一致）证必要；容差内等价与 §8.2 精度标准一致。
- **向量化替代完整性**：§1.6.2 覆盖 8 个计算点（#1 二路选择、#2 乘 slope、#3 取大、#4 fp32 中转、#5 零常量、#6 核内调度循环、#7 block 索引、#8 动态尾块）；逐元素点 #1–#5 **已全部向量化**；保留 #6–#8 逐项给出具体理由（tile 级调度循环 / block 索引 / 动态边界）。与 §3.3 伪代码、§6 循环结构**交叉核对一致**：kernel 内无逐元素标量循环。
- **不可替代理由充分性**：保留 3 项理由具体到本算子（含先例 `PL-1.22`），非泛泛「实现简单」。
- **替代 API 佐证**：`vmax/vcmp/vselect/vmul/vcast/fill` 均有 docs 佐证 + 设备探针实跑，优先 v-prefix。
- **向量化轴与布局决策完整性**：§1.6.3 单轴算子豁免**成立**（1-D `[N]` 展平后唯一候选向量化轴即元素轴，无第二轴可枚举）；记录轴质量（元素 4B/2B、尾 lane 浪费 0%、无累加链、无 repack）、迁移注（GPU 轴是输入非结论）、决策落地（§3.3/§6 一致）。
- **布局决策与实现一致**：选定一维连续布局与 §3.3（整 buffer v-op）、§4（buffer 形状 `(block_size,)`）、§6 三方一致，无 repack。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入（表头明确），§6 无 §1.6.2 未论证的标量循环。

**非阻塞观察（不影响结论）**：① §0/§1.6.1 用词「违反比」实际为 `max(|cand-base| / (atol+rtol·|base|))`（容差消耗比，非元素违反率）；DESIGN 已同时标注「viol=0 / 仍 PASS / 余量小」，实质无误，仅命名略松。② §1.6.0 R3 口径注「fp16/bf16 各候选均 +2 vcast」中基线（输入 dtype 域）行实为 3 op 不含 cast，注文略宽；表内数字自洽，不影响结论。二者属表述层面，均未构成阻塞或需修订项。

## 机械复核证据摘要

| 检查 | 结果 | 证据 |
|------|------|------|
| verify_equiv 重跑 vs §1.6.1 内嵌表 | **一致，全 EQUIV_PASS** | 实跑 20 行逐项匹配（含 fp16 4.883e-4/0.2597、bf16 3.906e-3/0.1297、GOLDEN 全 0.0、DIAG 6.550e4/6.485e6） |
| 设备探针重跑（Ascend910_9362） | **ALL PASS** | fp32dom where/max：fp32 0.0、fp16 3.052e-5、bf16 2.441e-4；角点/小 x/尾块 0.0；inputdom fp16 角点 4.883e-4（0.2597）；slope 0.5/1.0/0.001 0.0 |
| design_calc_check.ub_budget | skip→人工 pass | 21/34 KB ×1.7 = 36/58 KB < 192 KB |
| design_calc_check.l0c_budget | pass | 3.9 KB vs 128 KB（n/a 纯 Vector） |
| design_calc_check.core_split | skip→人工 pass | 逻辑 512~16384 / 物理 40 / 网格跨步静态上界 |
| design_calc_check.r3_metrics | pass | 四口径齐全 |
