# `_argreduce_kernel`（承载 `ArgminFwdOp`）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/argmin/_argreduce_kernel/DESIGN.md`（**v1**，Stage 1 `revision` 后重检；上版备份 `history_version/design_v0.md`）
- 任务类型: 迁移（migration-harness；`scenario=migration`）
- 源算子: 源仓 `kernels` 下 `reduction` 目录中的 `argreduce.py`（`_argreduce_kernel` 工厂 + `ArgreduceKernel`；GPU 侧 `ArgminFwdOp`/`ArgmaxFwdOp` 共用）
- 检视轮次: **第 2 轮**（本报告覆盖第 1 轮 `REVIEW.md`；第 1 轮结论=通过、0 阻塞 + 5 条建议级）
- 检视维度: 9 项（维度 0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2（均为**表述精度**类残留，见「非阻塞改进建议」；上轮 5 条建议已**逐条落实**并复跑验证）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS（ALL EQUIV_PASS，controls_ok=True）**
  - 执行：`/workspace/venv/bin/python examples/argmin/_argreduce_kernel/verify_equiv.py`，**exit 0**（默认 `python3` 无 torch，改用本机 venv 解释器 torch **2.9.0+cpu**——检视环境选择，非脚本缺陷）。
  - 执行输出与 §1.6.1 内嵌结果表**逐行一致**：`E1 mask vs serial-scan baseline: rows=264 violations=0`、`E2 direct-min vs -max(-x) transform baseline: 264/0`、`E3 online vs direct argmin: 90/0`、`E4 native-dtype pipeline vs fp32 up-cast pipeline: 264/0`、`E5 padded(+inf) vs real width: 264/0`；`TOTAL equivalence groups=5 violations=0 rows=1146 -> ALL EQUIV_PASS | controls_ok=True`。
  - **判别力对照（v1 新增，本轮亲验）**：`CTL last-occurrence: rows=72 violations=72 -> CONTROL_OK`、`CTL small-sentinel: 72/57 -> CONTROL_OK`、`CTL lossy coarsened domain: 3/2 -> CONTROL_OK`——三组故意错误候选**均被报出差异**，脚本判别力由机器断言（若为 0 违反则报 `CONTROL_BROKEN` 并失败）。
  - **空真守卫核对（逐组）**：5 个门禁组比较元素数 = **264/264/90/264/264**，3 个对照组 = **72/72/3**，**全部 > 0**，无 0 元素空真 PASS；case 前置条件均成立（门禁组 NaN-free、E3 `tile_n` 静态、E5 `n_padded ≥ W`、`cand_padded_argmin` 对不足宽度 raise）。
  - **上轮「E2 死赋值 / E4 精确单射恒真」两类退化对照已消除**：E2 基线现为真实 `-max(-x)` 变换域（`verify_equiv.py:106–122`，`m = -neg.max(...)`，无覆盖语句），与直接域 `cand_direct_min_argmin` 为**两条不同实现路径**；E4 改为**原生 dtype 流水**（`cand_native_argmin:147–151`）vs **fp32 上抛流水**（`cand_upcast_argmin:154–158`）的真实跨域对照；两者由 `INFO NaN` 行（npu-candidate=哨兵 `2^30` vs source-fmax 基线非 NaN 极值/回退 0 vs torch 首 NaN 索引，**三方各异**）佐证**结构上可分**。
  - `INFO NaN boundary` 4 行为**文档化非等价域**记录（§0.1 边界表 / §9.2-R1），非门禁项。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/argmin/_argreduce_kernel/DESIGN.md`（**exit 0**）——JSON `checks` 摘要：

  | check | status | detail | 人工复核结论 |
  |-------|--------|--------|-------------|
  | `ub_budget` | **skip** | no per-level buffer tables parsed from the memory plan (advisory checker; verify manually) | **机械跳过，人工已核**：按 §4.5 B/elem 表（fp16 8/10、bf16 10/14、fp32 12/16）逐 workload 复算——smoke 16/20/24 KB、non-aligned 32/40 KB、hidden-state fp16 bm=2 = 64 KB、hidden bf16 bm=1 = 56 KB、fp32 bm=1 = 64 KB，均 ≤ 64 KB 手工预算、×2 双缓冲 ≤ 128 KB ≤ 192 KB − 8 KB reserve；B/elem 与同族已验证 Stage-4 kernel 的 `_B_elem`（`argmax_kernel/perf_opt/_argreduce_kernel.py:134–141`）**逐值一致** |
  | `l0c_budget` | **skip** | no affirmative Cube-path API (T.gemm/load_2nz/store_fixpipe/Scope(Cube)) in this design | **机械跳过，人工已核**：纯 Vector（§5.1），无 Cube → L0C 容量约束不适用（§3.5.1、§5.3） |
  | `core_split` | **skip** | insufficient parsed dims {}; declared logical cores = None | **机械跳过，人工已核**：§5.5 逐负载复算 `num_logical = ceil(M/block_m)` = smoke 4、non-aligned 32、hidden fp16 1024、hidden bf16 2048、lm-head 4、dim=0 64；`num_kernels = min(logical, 40)`；核内 `T.serial` 静态上界 26/52/2。发射门控 80 = 2×40 与本机实查 `aivector_core_num=40` 自洽 |
  | `r3_metrics` | **pass** | all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak) | **机械通过**（§1.6.0-R3 四口径齐全） |

  JSON 头部：`capacity_constants = {ub_kb: 192.0, ub_inflation: 1.7, l0c_kb: 128.0, source: pattern-library/constants.md CONST-capacity-910B2C}`。
- **设备实查复核（本机独立复验，非采信设计自述）**：`NPUUtils.get().get_aicore_num()` = **20**；`NPUUtils.get().get_aivector_core_num()` = **40**；`AscendArch().aivector_core_num` = **40**；`AscendArch().ub_cap` = **196608**、`chip_name` = `Ascend910B`。与 DESIGN §0 实查表及 §5.5 **逐值一致**（运行期查询，非静态回退值）。

## 上轮 5 条建议落实核对（第 2 轮专项）

| # | 上轮建议 | v1 落实载体 | 亲验结论 |
|---|---------|------------|---------|
| 1 | `verify_equiv.py` E2/E4 无判别力 → 补强 | 脚本重写（`verify_equiv.py:26–41` 修订说明、`106–122` E2 真变换域基线、`147–158` E4 双 dtype 流水、`203–236` 3 组错误候选）；§1.6.1 v1 注 + 结果表重跑回写 | **落实**：E2/E4 现为真实对照路径；3 组 CONTROL 全 `CONTROL_OK`（亲跑复核，见机械复核段） |
| 2 | §4.5 记账叙述不准确（`idx_*` / 8 KB reserve） | §4.5 `940–947` 改为「探针定标的**代理预算** + PlanMemory 活跃期复用/别名吸收」，显式声明**非可编译证明**、跨设备（910B3）须 Stage 3 L0-4 复标定 | **落实**：与 §4.3 清单、§9.2-R4 一致。注：`§0.6-R4` 汇总式仍写作通用 `B(dtype)`（见非阻塞建议 1） |
| 3 | §1.6.2 结论段计数「4 处」应为「3 处」 | §1.6.2 结论段 `553–555` 改为「保留 **3 处**（#8 / #10 / #11）」，并明确「#9 属**已替代**侧，不计入保留集」 | **落实**：结论段与表体（#9 标 ✅、列入「#1–#7、#9」）**逐项对齐**，无二义 |
| 4 | §5.2 基础阶梯 `p=1` 误用 `B_multi` → 应用 `B_bm1` | §5.2 基础阶梯 `1000–1002` 与 §3.3 工厂式 `760–761` 均改为 `p · N_padded · (B_bm1 if p == 1 else B_multi) <= UB_MANUAL_BUDGET`；§5.2 补「预算过滤器」注 `1046–1050`（`N_padded ∈ (4096,5461] × fp32` 类形状正确回落 tiled） | **落实**：与同族 Stage-4 verified kernel `_select_config`（`perf_opt/_argreduce_kernel.py:203–207`，标 REVIEW N1 fix）**形态一致**；manifest 负载不受影响（fp32@4096=64 KB 恰在上限、bf16@4096=56 KB） |
| 5 | §1.6.0-R3 的 C1 应记 ~4–5 pass | §1.6.0-R3 C1 行 `444` 改为「≈ **4–5·N_padded 向量 op**（含 `vbrc` 广播与候选 `reduce_min`）」 | **落实**：与 §1.6.2-#6 / §1.6.3-#5 的链式候选（`reduce_min + vbrc + vcmp + vselect + reduce_min` = 5 op）一一对应 |

> 结论：上轮 5 条建议**逐条落实且经本机重跑/复算复核成立**。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源 kernel（源仓 `kernels` 下 `reduction` 目录的 `argreduce.py` 全 264 行）、源 op 层（源仓 `ops` 下 `reduction` 目录的 `argreduce.py`、`reduce.py`）、源 `_primitives.py`，并实读 NPU 侧 op/base/wrapper/manifest；设备查询在**本机实时执行**复验；逐条对 DESIGN §0.1–§0.7。

| 检查项 | 结论 | 源码证据 ↔ DESIGN 章节 |
|--------|------|----------------------|
| 语义理解正确性 | **pass** | 源 `main` 签名 `x: T.Tensor[(M, N_padded), dtype]`、`out: T.Tensor[(M,), "int64"]`（源第 57–60 行）与 §0.2 I/O 表一致；first-occurrence 由 `T.fill(out_idx, 0)`（源第 95 行）＋ `for j in T.Serial(N): if x_f32[i,j]==row_extreme[i]: out_idx[i]=...; T.loop_break()`（源第 96–100 行）实现，与 §0.1 表述一致；`T.cast(shared_buf[i,j], "float32")`（源第 71–72 行）→ §0.1「先 cast fp32」一致。**NaN 行为**：argmin 走 `reduce_max(neg_x)` 的 **fmax NaN-忽略**（源第 85–89 行）→ 全 NaN 行 `row_extreme` 经 `T.fill(-inf)`＋fmax 保持 `-inf`、第 91–92 行取负后为 `+inf` → 扫描 `NaN == +inf` 永不成立 → 输出回退值 0；§0.1 边界表与 `verify_equiv.py` INFO 行（row1 all-NaN → source-fmax-baseline=0）**双向印证** |
| 算法解读完整性 | **pass** | §0.3 步骤 1–5 覆盖源全部计算语句：第 68 行 `T.copy`（步骤 1）、第 71–72 行 cast（步骤 2）、第 79–80 行 argmax 分支（3a）、第 85–92 行 argmin `-max(-x)` 三分支（3b）、第 95 行 fill（3c）、第 96–100 行扫描（步骤 4）、第 103 行 `T.copy`（步骤 5）——**无遗漏、无臆造**；host 侧（§0.3 末）与源 op 层 `_prepare_input`＋第 34 行 `@functools.lru_cache(maxsize=32)` 一致 |
| 优化手段覆盖完整性 | **pass** | §0.4 八项（SMEM tiling / 两步法 / fp32 cast / `-max(-x)` / host pad 单位元 / 48 KB 启发式 / `lru_cache` / 重 pad 布局约束）逐项可在源码定位（第 62、68、74–100、71–72、85–92、52、182–216、34、208–211 行）；**正确判定源无** warp shuffle / mma / swizzle / 异步流水（源码确为朴素两步法）；**源缺陷识别正确**：第 68 / 103 行 `T.copy` 只给 base 坐标、由 dst 推形状，**无 `M % block_m` 守卫**（`(129,512)` 且 bm=8 → `129 % 8 = 1` 即越界）——§0.4 注＋§0.5 对应行＋§0.6-R5 修正三件套 |
| 耦合性判定合理性 | **pass** | §0.5 十二条均含四态处置（保留/等价替换/重新设计/舍弃）＋依据：`-max(-x)`→等价替换直接 `reduce_min`、串行扫描→重新设计掩码索引、`threads`/两级并行→重新设计 persistent、fp32 cast→fp16 舍弃/bf16 保留（API 需求）、host pad 单位元→保留语义/舍弃 256 对齐、源尾块无守卫→重新设计；与 `migration-analysis.md §5.3`（shared memory→L1/UB）、`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.1（bf16 ×）、§3.5.1 陷阱表**不矛盾**；`_pad_value` 方向性（argmin=**+inf**）与源/NPU op 层实读一致。**无「warp shuffle 判保留」「舍弃项无理由」类失误** |
| NPU 重设计可行性 | **pass** | §0.6 R1–R5 均含「源方案 → NPU 新算法 → 语义保持论证」三要素：R1 掩码索引取 min（E1 机器验证）、R2 一维 `T.Kernel`＋`T.serial` 静态边界＋动态 guard、R3 pad 单位元契约（E5 机器验证）、R4 UB 预算公式（§5.2 实例化）、R5 尾块 `real_m = T.min(bm, M−start)`＋双显式 slice（VP-2026-0044 先例）。新算法满足 NPU 硬件约束（一维 Kernel、UB 层级、32 B 尾轴对齐、容量） |
| 设计与决策一致性 | **pass** | §1.4（直接 `reduce_min`、显式向量链、persistent 40 核）、§3.1/§3.2、§4.4、§5.2、§6 全部为 §0.5/§0.6 决策后的 NPU 算法——源 `-max(-x)`、串行扫描、`threads`、256 对齐**均未出现在实现章节**；§1.6.0 选定 C1+C2 **实际驱动** §1.4/§3.1/§6；无「§0 判重设计、§3 照抄源方案」脱节 |
| golden 独立性 | **pass** | §8.1 golden = `torch.argmin(x, dim=-1)`，显式声明「**不复刻** §0.6 NPU 算法（无掩码候选 / `reduce_min` / 在线递推结构）」——以 §0.1 语义为唯一依据，验证独立 |
| host/op 层引用路径真实性 | **pass** | §0.2/§0.7 引用的源仓与仓内路径**逐条 `ls` 存在**：源 op 层两文件、源 `_primitives.py`；NPU op 层 `argmin.py`、`reduce.py`、`argmin` 子目录的 wrapper、manifest、test/bench、`_argmin_fwd_kernels.py` 全部命中——无「引用不存在的单算子文件」 |
| 多输出/变体迁移范围 | **pass** | 源 `argreduce.py` 共享 `ArgminFwdOp`/`ArgmaxFwdOp`。§0.2 显式声明「本迁移单元范围 = argmin；argmax 为独立迁移单元，工厂接口保留双 kind（§8.2 L0-10）」，与 manifest `ArgminFwdOp` 段一致；NPU op 层亦只移植 `ArgminFwdOp`——**无范围蔓延** |

**集成前置核对（VP-2026-0134 / VP-2026-0128，亲读目标工件）**：§0.2/§5.6 三项声明**与源/NPU op 层实际契约一致**：

1. `ArgminFwdOp` **未覆写** `_kernel_handles_padding` → 继承 `_ReduceOpBase._kernel_handles_padding = False`（NPU `reduce.py` **第 91 行**，与引用一致）；
2. `_pad_value()` = `float("inf")`（NPU op 层 `argmin.py` **第 76–78 行**，与引用一致）——方向性正确（反向会让 pad 列胜出、静默错值）；
3. `_kernel_handles_padding=False` 时 op 层执行 `F.pad(x, ..., value=pv)` 到 `align_up(N,256)` 后才调 kernel（NPU `reduce.py` 第 465 / 492 行）→ kernel 必须声明 `N_padded`，与 §0.2「一致性规则」一致；NPU wrapper 以**原始 N** 调工厂（kernel 内派生 `N_padded`），契约自洽。

**残留引用精度（不可阻塞）**：§0.2 第 2 条引「源仓 `reduce.py` 第 460–472 行」，实际 `F.pad` 位于 460–466（multi-dim）/ 487–492（single-dim）——范围端点略宽，语义无误（见非阻塞建议 2）。

### 1. API 可行性 — pass

§3.2/§3.4/§3.5.1 所列 API **均有本仓佐证且优先 v-prefix**：`T.reduce_min(dim=1)`（`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.1 + `T.reduce_max.md` + `testing/npuir/reduction_ops/test_reduce.py`）、`T.vbrc`（`shape操作/T.vbrc.md` §2.1 允许 `scalar|tensor`、§2.2.2 张量广播）、`T.vcmp`（`比较操作/T.vcmp.md`）、`T.vselect`（`条件操作/T.vselect.md`）、`T.vcast`（`数据类型转换操作/T.vcast.md` §2.2.1：f32→i64 `rint`、f16/bf16→f32 `rint`）、`T.arange`+`T.reshape`（`创建操作/T.arange.md`、`shape操作/T.reshape.md`）、`T.Kernel(一维, is_npu=True)`+`T.serial`+动态 guard（`T.vcast.md` §2.4 + logsumexp 家族）。所有形态与**已硬件验证**的同族 Stage-4 kernel `perf_opt/_argreduce_kernel.py` 同构。Developer 模式 = 自动同步，§7 无手动 `set_flag/wait_flag`，与 §2.1 一致。v-prefix 优先符合 `AGENTS.md`。

### 2. 内存层级规划 — pass

搬运路径 GM→UB→fragment→GM 完整（§4.4），无 GM→L0 越级；纯 Vector，无 L1/L0C 需求；Buffer 层级逐项标注（§4.3）。UB 预算：机械 checker `ub_budget` **skip**，人工逐配置复算通过（见机械复核段）；B/elem 表与同族 Stage-4 verified kernel `_B_elem` **逐值一致**，其探针锚点（P3a：fp16 bm=4@4096 编译拒绝、bm=2 fragment 形态 PASS）为 64 KB 手工预算提供实证。§4.5 v1 记账口径（代理预算 + PlanMemory 别名吸收）**已修正**且显式声明非可编译证明——上轮建议 2 已消解。

### 3. Tiling 策略 — pass

§5.5 分核三要素齐全：① 逻辑核数逐负载给出；② 物理核数**本机实查**（`get_aicore_num()`=20 / `aivector_core_num`=40 / `ub_cap`=196608，与 §0 表逐值一致，非假设值）；③ 规模判定 + persistent 方案表。核数与 block 取值自洽，**发射门控 80 = 2×40** 与本机核数同步。逐负载独立复算：smoke bm=8→4 核；non-aligned bm=8（基础阶梯封顶，虽预算容许 p=16）→32 核；hidden fp16 bm=2→1024 逻辑/40 物理/核内 26；hidden bf16 bm=1→2048/40/52；lm-head bm=1→4；dim=0 bm=32→64/40/2。`block_m` 阶梯（基础 `{1,2,4,8}` + 扩展 `{16..2048}` 发射门控）与预算过滤器（**v1 已修正 p=1 用 `B_bm1`**）逻辑一致；非整除（M）由 §5.4 三件套处理；非整除（N）由 pad 契约（E5 已验证）处理；`assert N_padded <= 2**24` 与 E1/E3 精确域前提对应。

### 4. 技术约束检测 — pass

§3.5.1 逐条覆盖：无三维 Kernel、纯 Vector 核数翻倍、bf16 reduce 不支持→预转 fp32（`T.reduce.md` §2.2.1 实读 bf16=×）、reduce 混合 dtype 静默错值→同 dtype、`(·,1)` 不变操作数 bug→`vbrc` 同形化、shared 多消费者竞态→fragment + 64 KB、multitile + bm>1 SIGSEGV→tiled 强制 bm=1、`vcast` f32→f32 破坏→同 dtype `T.copy`、2D Parallel 融合标量化→显式向量链、原位 `vselect` 丢 loop-carried 状态→无别名更新。均与 `docs/` 或同族实测对应，无 L0C 溢出 / GEMM 非整除未处理项（不适用）。

### 5. 循环与同步 — pass

§6 明确循环类型：persistent 任务循环 `T.serial`（静态上界 + grid-stride + guard）、tiled tile 循环 `T.serial(num_full−1)`（静态；tile-0 剥离初始化）、元素级向量化（`vbrc`/`vcmp`/`vselect` + `reduce_min`）、索引构建与出栈向量化、标量 `T.min`/guard；无 `T.Pipelined`（Developer auto-multi-buffer 自动施加，§6.3）。§7 同步 = Developer 自动（单 kernel、无核间/跨引擎；C6 备选两段 kernel 由 wrapper 顺序发射保证依赖），与 §2.1 模式一致。尾块处理 §6.4 → §5.4（M：`real_m` + 双 slice；N：pad 全宽；tiled：静态尾 tile）。

### 6. 验证方案 — pass

§8.1 golden = `torch.argmin`（PyTorch 参考实现草案存在，且独立于 NPU 算法）；§8.2 L0 门槛计划 L0-1..L0-11 覆盖 manifest 全负载 × dtype、非对齐宽度判别（L0-6/L0-11，针对 VP-2026-0128 步长错位）、M 尾块源缺陷复验（L0-7）、M=1/tiled 分界（L0-8）、角点 tie（L0-9）、双 kind 契约（L0-10）。**精度标准**：索引输出采用**精确比对**（`torch.equal`＋`out.dtype==int64`）而非 atol/rtol——理由（索引语义、容差会掩盖 tie-break/哨兵类 bug）成立，属对通用 dtype 容差标准的**正确特化**。L2 记录不阻断项（NaN/混合零/M=0/超界断言）。

### 7. 完整性与一致性 — pass

章节齐全（§0 含 0.1–0.7 + §1.1–§1.6.4 + §2–§10 + 修订说明）；全文**无模板占位符**（已扫描，0 命中）；§3.5.3 引用的 examples 路径**全部存在**（逐条核对）；API 映射 / 内存规划 / Tiling 三处相互自洽。**上轮 2 处内部不一致（§1.6.2 计数、§5.2 p=1）均已修正**（v1 建议 3/4），本轮未发现新的内部矛盾（仅余 §0.6-R4 汇总式表述精度，见非阻塞建议 1）。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：含 1.6.0（调研）/1.6.1（等价优化）/1.6.2（向量化替代）/1.6.3（轴与布局决策），另附 1.6.4（C6 实验裁决三件套）。

**① §1.6.0 调研完整性（四问齐全，基线在表）**：R1 候选表 6 项（含**基线** #0 GPU 两步法）；R2 结论明确「**有**在线变体」（结构判据：argmin 可分解为可增量合并的 running `(min,idx)`，与 online softmax `(max,sum)` 同构）；R3 四口径表（FLOPs / 访存量 Bytes / GM 遍数 / 中间缓冲峰值，基线在表、口径写清，含「tiled 两遍式」对照行）；R4 逐候选亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）。「无更优替代」未作空白断言，且写明**调研范围**（`algorithm-candidates.md` `ALG-reduction` R1–R4 命中行、`kb_search` 5 条、`examples/` logsumexp/argmax/ssd_chunk_scan、pattern-library、恒等变形空间 C3/C4/C5 全枚举、结构判据）。

**② 调研结论正确性（独立复核，复算/核对）**：

- **候选覆盖复核**（对照设计 skill `references/algorithm-candidates.md` `ALG-reduction` 命中行 `81–87`：R1 max+sum 合并扫描、R2 分块归约+合并/Welford、R3 扫描遍数/跨核归并代价、R4 水平 vs 垂直 lane）：逐项被 §1.6.0（C1/C2/C6）与 §1.6.3 轴矩阵承接；**Welford 属方差族、对 argmin 不适用**（未强套，正确）；**未见遗漏的已知候选**。
- **负向断言逐条复核**：无「无化简公式/无在线变体/无更低复杂度算法」类空白断言；C3（int64 打包）「净亏」与 C4（全排序 `O(N log N) ≫ O(N)`）的淘汰依据为结构性/复杂度性——**独立复算成立**（键构造 ~5 全宽 pass + reduce + 解码 ≈ 7 遍 vs C1 ~4–5 遍）；C5 作为 E2 对照纳入。R4 容量/带宽淘汰逐条引 `pattern-library/constants.md`（`CONST-capacity-910B2C`、`CONST-vector-launch-overhead` 等，均经 `INDEX.md` 核对为 **verified** 条目）并标版本戳。
- **复杂度四口径独立复算**：基线 `N_padded·elem+8 / 1 遍`；**C1 `≈4–5·N_padded`**（`reduce_min + vbrc + vcmp + vselect + reduce_min` = 5 op，与 v1 建议 5 修正后一致，本维复核确认）；C2 online `1 遍 GM / O(tile) 缓冲`（每 tile `tn+tn+tn+O(1)`）；tiled 两遍式 `2N_padded·elem+8 / 2 遍`——增量与遍数逻辑一致。**未发现中间缓冲漏计 GM 往返**（驻留路径无 GM 中间往返；tiled 路径 workspace=0；C6 备选明确计入 `(M,nchunk)×2` 往返 ≈3.8 KB ≪ 800 KB 读流量）。
- **并列候选裁决路径**：C1 vs C2 由 **N 驻留阈值**（结构性）分派而非纸面拍板；C6 交由 **§1.6.4 Stage 4 A/B**（主选 + 完整备选结构 + 判定阈值 `C6 < 0.8× 主选` + 回写路径），未纸面单选。
- **R4 跨设备常数额外核对**：绑定 910B2C / CANN 8.5.0 者统一标「未实测假设：沿用同代 910B 曲线量级，Stage 4 校准」；核数/UB 容量为**本机实查**（20/40/196608）——符合 `AGENTS.md` 负向/未文档化假设规则。

**③ 调研→设计一致性**：§1.4/§3.1/§6 实现的正是 §1.6.0 选定 C1+C2（resident/tiled 双路径、显式向量链、persistent 40）；§0.5/§0.6 处置与选族一致；**源算法优化意图逐项承接**（§1.6.0 结论段：两步结构保留、fp32 cast 由 E4 论证舍弃、`-max(-x)` 由 E2 替换、单位元 pad 由 R3 保留 / 256 对齐舍弃、SMEM→UB、#6/#8→R4、#7 保留）——**无静默丢弃**。

**④ §1.6.1 四要素与等价性独立推演（逐条，独立于文档自述）**：

- **E1** `first = reduce_min_j(ite(x_j=m, j, BIG))`：m 由实际数据归约得出（有限/±inf 时至少一列命中）；`BIG=2^30`（fp32 精确 2 的幂）> 任何合法 j（`j < N_padded ≤ 2^24`，fp32 精确整数域）⟹ `reduce_min` 取最小命中 j = 首现。**成立**（264 行含 tie/全 ±inf/亚正规/混合零/多 N 机器验证 0 违反）。
- **E2** `argmin = first-match(min(x))` 替换 `-max(-x)`：取负严格单调且保等 ⟹ 索引恒等（不经值域算术）。**成立**——本轮 E2 组已为**真变换域 vs 直接域**对照，机器验证 264/0；INFO NaN 行证明两路径结构可分。
- **E3** online `(r_m,r_f)` tile-0 直写 + **严格小于**更新：归纳成立——处理 tiles `0..t` 后 `(r_m,r_f)`=(已见列 min, 其首现)；`c_m<r_m` 时新极值必来自本 tile 且更早列不可能命中（≥r_m>c_m）；`c_m=r_m` 保留更早索引（升序 tile）；tile-0 直写消除哨兵初始化陷阱（全 `+inf` 行 `c_m=+inf=r_m` 不触发严格小于则首现丢失）。**成立**（90 行含跨 tile tie/尾 tile/全 `+inf`）。
- **E4** fp16 免 fp32 上抛：min 为精确选择运算，fp16↔fp32 单射保序保等 ⟹ 首现索引恒等；bf16 保留 cast 属 **API 需求**（`T.reduce.md` §2.2.1 bf16 ×，非精度需求）。**成立**——本轮 E4 组为原生 dtype 流水 vs fp32 上抛流水的真实跨域对照，机器验证 264/0；`CONTROL lossy coarsened` 证明非单射路径可被报出。
- **E5** `+inf` pad 索引中性：pad 值 = argmin 单位元；pad 列命中（`=+inf`）⟹ 全行皆 `+inf` ⟹ 实列 j=0 同样命中且索引更小 ⟹ 全域计算取到的仍是实列首现。**成立**（264 行含全 `+inf` 行）。
- **收益量化**：E1 标量 N/2 次不可向量化 op → 2 向量 pass；E2 省 2 全宽 pass；E3 GM 2 遍→1 遍（−800 KB@lm-head）+ 缓冲 O(N)→O(tile)；E4 省 1 全宽 pass + 2 B/elem；E5 免 kernel 侧掩码/尾 tile 特化（代价 pad 108.5 KB 非带宽主导）。四要素齐全。

**⑤ §1.6.2 向量化替代完整性（与 §3.3/§6 交叉核对）**：盘点 §1.4 全部循环/标量点 11 项，**逐项与 §3.3 伪代码、§6 循环结构对照无遗漏**：已替代 8 项（#1 扫描→向量链、#2 极值→`reduce_min`、#3 取负/还原→消除、#4 cast→消除/`vcast`、#5 索引→arange+reshape+vbrc、#6 广播→`vbrc`、#7 在线更新→`vcmp/vselect` 无别名、#9 出栈→`reshape`+`vcast(rint)`），**不可替代 3 项**（#8 tile 基址全局化=tile 级顺序依赖载体、#10 `real_m`/guard=动态边界、#11 host 配置=元数据）——保留理由具体到本算子（非「实现简单/照源码」）。**替代 API 佐证**齐全并优先 v-prefix。结论段计数「保留 3 处」与表体**逐项对齐**（v1 建议 3 已消解）。

**⑥ §1.6.3 向量化轴与布局决策完整性**：I/O layout 视为契约、核内布局为设计变量（开篇申明）；按**规约类必选候选**枚举——水平归约（lane→N，reduce `dim=1`）vs 垂直扫描（lane→M）**两种 lane 映射分别枚举**，含两遍/online（C2）交互；候选矩阵 5 行含 GPU 源轴对照（「源码 thread/warp 轴是输入非结论」）。逐候选量化（整除性/累加链/repack/UB 影响）。**弃选理由量化且经核对**：#2 发射爆炸（`N_padded=4096 → ms 量级/行`）；#3 核内转置「纯开销零收益」（主选轴即 I/O 原生连续轴、无 repack 需求；代价引 `PL-1.1-transpose-chain` ~5.4 µs，经 `pattern-library/INDEX.md:33` 核对为 verified）——且设计**主动声明**「`T.transpose.md` 的 dtype 表在本算子无需核对——该路径已被量化收益否决而非能力否决」，有效回避同族误判；#4 交 §1.6.4 A/B；#5 以 `TRAP-parallel2d-bm-ge2-scalarize`/CG-2026-0015 实测（`bm≥2` 标量化 ratio 0.84–0.95）裁决。**布局决策与 §3.3/§4/§6 三方一致**（buffer 形状、累加维、重排路径）。

**⑦ 实验裁决模式完整性**：主选（§1.4 双路径）不依赖未实证常数即可成形；C6 备选提供**完整三件套**（① 工厂期静态分派谓词 `M≤16 ∧ ceil(M/bm)<40 ∧ N_padded≥32768`；② 两段 kernel 完整结构含 buffer 形状/UB 预算/tie-break/分核三要素（partial 逻辑 80 > 物理 40 → persistent）；③ 判定阈值 `C6 < 0.8×` + 代表 shape + 回写路径 + 按 shape 分派可行性），达可实现深度。

**⑧ 与下游一致**：§3.1 以 §1.6.1 优化后公式为输入；§6 无 §1.6.2 已判定向量化点的对应标量循环；§1.6.3 布局决策同步落入 §3.3/§6。**编译期特化谓词**：本算子无窗口/恒等轴类特化谓词（不适用），无「充分条件漏参数」风险面。

## 非阻塞改进建议（结论为通过，不构成阻塞）

### 建议 1（建议级 · 表述精度，可选）: §0.6-R4 汇总式仍写作通用 `B(dtype)`

- 位置: DESIGN.md §0.6-R4（约第 283–284 行）
- 问题描述: §0.6-R4 的汇总式为 `block_m = max{p ∈ {1,2,4,8} : p·N_padded·B(dtype) ≤ 64 KB}`，使用通用 `B(dtype)`，未写出 v1 建议 4 在 §5.2/§3.3 已落实的 `p=1 → B_bm1` 分档（§0.6-R4 自称「沿用同族已验证模型，§5.2」，读者可回溯正确式）。**非阻塞**：规范式在 §5.2/§3.3 均正确，且 Stage 3 以 §5.2 为单一事实源；manifest 负载不受影响。
- 修改建议: 将 §0.6-R4 汇总式同步为 `p · N_padded · (B_bm1 if p == 1 else B_multi) ≤ 64 KB`，与 §5.2/§3.3 完全同形（与上轮建议 4 同口径），使全文对该表达式只有一个写法。

### 建议 2（建议级 · 引用精度，可选）: §0.2 引用行号范围略宽

- 位置: DESIGN.md §0.2 第 2 条
- 问题描述: 引「源仓 `reduce.py` 第 460–472 行」；实际 `F.pad` 位于 460–466（multi-dim）与 487–492（single-dim）。范围端点略宽，语义无误（上轮已记，本版未改）。
- 修改建议: 收窄为「第 460–466 行（multi-dim）/ 487–492 行（single-dim）」，避免后续 Stage 复读时误以为 467–486 段亦参与 pad。

## 检视范围与方法说明

- 亲自 Read：DESIGN.md 全文（v1，含修订说明）；源 kernel / 源 op 层 / `_primitives.py`；`verify_equiv.py`（v1 全文）；NPU 侧 op/base/wrapper/manifest；`tilelang-op-design` skill `references/algorithm-candidates.md`；pattern-library `INDEX.md`/`constants.md`/`traps-compiler.md` 相关条目；相关 `docs/Tilelang.language/` API 文档。
- 机械复核：重跑 `verify_equiv.py`（`/workspace/venv/bin/python`，exit 0，ALL EQUIV_PASS + 3 controls OK）与 `design_calc_check.py`（exit 0，r3_metrics pass、3 项 skip 已人工复核）。
- 设备实查：本机实时运行 `NPUUtils`/`AscendArch` 复验（20/40/196608/Ascend910B）。
- 上轮 5 条建议：逐条核对落实载体并复跑/复算确认。
