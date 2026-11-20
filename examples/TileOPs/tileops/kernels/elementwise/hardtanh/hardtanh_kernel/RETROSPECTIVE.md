## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 迁移任务的「计算域」判定不能照搬同族先例模板：`leaky_relu` 的核心决策是「输入 dtype 域 → fp32 中转」，但 `hardtanh` 源码**本身即 fp32 中转**，若套用同族先例模板会误判为「需采纳 fp32 中转优化项」。需先做 §0.1「源与契约 golden 是否同域」核对再决定处置。 | `examples/hardtanh/_make_hardtanh_kernel/DESIGN.md#0.1`（域对照表）对照 `examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md#0.6 R3` | 在 `references/migration-analysis.md` §3「优化手段解读」补一条提醒：迁移设计先核对「源计算域 vs 契约 golden 域」，同域则该项为『保留』而非『优化采纳』，避免把保留项误报为优化项 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| clamp/投影类逐元素算子：fp16 原生 dtype clamp 与 fp32-中转 golden **逐位等价**（round 与 clamp 可交换——clamp 输出恒为 {x, lo, hi} 之一，且 round(lo) 与 lo 之间不存在格点），故 fp16 中转的 2 次 vcast 在数值上非必需；其保留理由应为「契约/一致性/ROI≈0」而非「正确性」 | P | `examples/hardtanh/_make_hardtanh_kernel/verify_equiv.py` 诊断行（native-dtype vs GOLDEN 三 dtype max_abs=0.0）；`DESIGN.md#1.6.1` E1 行 | `python3 examples/hardtanh/_make_hardtanh_kernel/verify_equiv.py`（看 `[DIAG]` 行） | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-13（CPU-only 数值；设备语义待重验） | pattern-library/elementwise.md（补 PL-1.17 的第四触发/边界：**投影类**算子不需要 fp32 中转即可对齐 golden） |
| （负面发现）`vclamp` 是 NPU 原生融合元算（前端展开 `npuir_max`+`npuir_min`），但 **dtype 矩阵不含 bf16**（`docs/Tilelang.language/比较操作/T.vclamp.md` §2.2.1 同 `T.vmin`/`T.vmax`）——bf16 逐元素 clamp 必须 fp32 中转；且 `T.vmin` 文档明示 src2 可标量而 `T.vmax.md` 仅示例张量-张量，故 `vmax(x, scalar)` 形态应规避（用预填常量 buffer 替代）。 | P | `DESIGN.md#1.6.0`（dtype 矩阵注）+ `#3.2` 注；`docs/Tilelang.language/比较操作/T.vclamp.md` §2.2.1、`T.vmax.md` §2.4、`T.vmin.md` §2.1 | `grep -n "bf16" docs/Tilelang.language/比较操作/T.vclamp.md` | 同上 | pattern-library/traps-compiler.md（TRAP-expert-v-operands 补充：vclamp/vmax 标量操作数差异） |

### Transferable Lessons（可迁移教训）

- 迁移设计第一步永远是「源计算域 vs 契约 golden 域」核对：**同域 = 保留，异域 = 等价替换（需机器验证）**；同族先例的「优化项」不可照搬。
- `T.vclamp(src, dst, lo, hi)` 是 NPU 原生 clamp，前端即 `npuir_max`+`npuir_min`，**bf16 不支持**；逐元素 clamp 的 bf16 路径必须 `T.vcast` 升 fp32 再算、末端 `rint` 舍回。
- `T.vmin` 文档允许标量第二操作数，`T.vmax` 文档仅示例张量-张量——单侧 clamp 若用 `vmax(x, scalar)` 存在操作数规则风险，改用预填常量 buffer 作第二操作数更稳。
- D-1 `verify_equiv.py` 的「非门禁诊断」区可用来量化**被拒候选**的等价性：若被拒候选在数值上亦逐位等价，则弃选理由必须写成 ROI/契约/API 可行性（不得写成"不等价"），否则与证据矛盾。


## Stage 2 (review) Retrospective — first_review — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 对本类迁移逐元素设计文档的**机械解析失效**：§4.5 的「Buffer 名/Shape/dtype/存储层级」四列表 与 §5.5 的逻辑核数表均未被解析，`ub_budget` 与 `core_split` 双双 skip，机械复核退化为纯人工复算（本类算子格式稳定、可覆盖）。 | `.agents/tools/design_calc_check.py` 输出 `{"check":"ub_budget","status":"skip","detail":"no per-level buffer tables parsed…"}`、`{"check":"core_split","status":"skip","detail":"insufficient parsed dims {}"}`；对照 `DESIGN.md#4.5`、`#5.5`（表格结构完整可解析） | 扩展 checker：解析「中间缓冲区」四列表累加各层级字节，并对 §5.5 的 `ceildiv(N, block_size)` 复算逻辑核数 | R |
| Review-flow | 分核策略标准（`core-split-strategy.md` §1 要素③）要求「显式记录残留负载不均与临界路径惩罚」，但该**残留百分比的算术正确性无门禁亦无机械复核位**——本次独立复算发现 §5.5 一处笔误。人工复算是唯一拦截手段。 | `DESIGN.md#5.5（3）`（bounded-hidden fp16 写 `+0.20%`）vs 独立复算：n_num=4096=40×102+16 → ceildiv=103、avg=102.4 → 103/102.4=1.00586→**+0.59%**（同表其余 5 项复算均正确） | 在 `core-split-strategy.md` §2.2 检视要点补一句「残留百分比须可独立复算」；或在 `design_calc_check.py` 增 residue 复算项 | P |
| Info-source | 本任务维度 0/8 的关键判据（bf16 dtype 矩阵、vmin/vmax 标量差异）全部可从本仓 `docs/` 静态取证，无需设备；但 skill 未把「dtype 矩阵逐格回读」写成显式步骤，检视者若不细读易以设计自述替代。 | `docs/Tilelang.language/比较操作/T.vclamp.md`/`T.vmin.md`/`T.vmax.md` §2.2.1（bf16 列 ×）、`T.vcast.md` §2.2.1（rint 约束） | `tilelang-design-review` skill 维度 8 负向断言复核补显式一步：API 能力/ dtype 断言 → 打开被引 doc 的 §2.2.1 逐格取值 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| （负面发现，文档非对称）`T.vmax.md` §2.1 参数表 `src1` 类型仅 `tensor`，而 `T.vmin.md` §2.1 `src2` 标注 `tensor or scalar`——单侧 clamp 折叠时 `vmax(x, scalar)` 有操作数规则风险、`vmin(x, scalar)` 文档许可，**二者不可对称设计**（本设计因此对 `hi=+inf` 折叠用预填常量 buffer 规避） | P | `docs/Tilelang.language/比较操作/T.vmax.md` §2.1 vs `T.vmin.md` §2.1；`examples/hardtanh/_make_hardtanh_kernel/DESIGN.md#3.2` 注 | `grep -n "scalar" docs/Tilelang.language/比较操作/T.vmin.md docs/Tilelang.language/比较操作/T.vmax.md` | docs only（无设备实测，待 Stage 3 编译探针） | pattern-library/traps-compiler.md（TRAP-expert-v-operands 补 vmin/vmax 标量操作数差异条目） |
| 残留负载不均百分比的通用复算口径（网格跨步 persistent 形态）：`n_num = P·q + r` → `r` 个核跑 `q+1`、`P−r` 个跑 `q`，临界 = `ceildiv(n_num,P)/(n_num/P) − 1`——可用于检视核对该列数字，避免笔误 | P | `examples/hardtanh/_make_hardtanh_kernel/DESIGN.md#5.5（3）`（同表 6 项，1 项笔误 0.20% vs 复算 0.59%） | 手工复算（本检视）；可脚本化 | 不适用（纯算术） | `_shared/standards/core-split-strategy.md` §1 要素③（补复算口径与示例） |

### Transferable Lessons（可迁移教训）

- 检视核内持久化（grid-stride）设计的残留负载不均时，**逐项独立复算** `ceildiv(n_num, P) / (n_num / P) − 1`：本仓设计此列数字易有笔误（本次 §5.5 6 项中 1 项偏差），且笔误会破坏随规模单调性，可作快速目检线索。
- sub-fp32 逐元素算子检视的两个必查文档点：① `vclamp`/`vmin`/`vmax` 的 dtype 矩阵（bf16 列为 `×`，故 bf16 必经 `vcast` fp32 中转）；② `vcast` 的 `round_mode` 约束（`f16→f32`/`bf16→f32`/`f32→f16`/`f32→bf16` 仅 `rint`）。逐格回读被引 doc §2.2.1，不采信设计自述。
- 单侧 clamp 的标量操作数能力**不对称**：`vmin` 文档许可 scalar src2、`vmax` 文档仅张量-张量；折叠实现须分别处置（`vmin(x,scalar)` vs `vmax(x, 常量buffer)`）。
- 当迁移源码本身已用 fp32 中转（如 hardtanh）且 wrapper 契约要求保留时，检视应识别为「保留项」而非「优化采纳项」，避免与同族 `leaky_relu`（源为输入 dtype 域）混淆。


## Stage 3 (develop) Retrospective — first_impl — 2026-10-13

> attempt 链：`first_impl` 单次通过（retry_count=0）。L0=13 / L1=22 / L2=6 / Boundary=9 全过，全层级 `max_diff=0.000000e+00`（逐位一致）。DESIGN.md §3.3 伪代码**逐行直译**即编译通过，未触发任何 retry / precision_fix。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| API-doc gap | `T.vclamp.md` §2.4 两个示例均为 **src/dst 分离** buffer，未给出 `vclamp(src, src, ...)` 原地形态；Stage 2 因此将该原地写法列为「建议级待编译探针」项。实测原地形态编译运行均通过且逐位一致，文档缺口使检视无法静态消解该不确定。 | `docs/Tilelang.language/比较操作/T.vclamp.md` §2.4（仅 `src_ub, dst_ub` 分离示例）；`REVIEW.md` 维度 1 注；`repro/PL-1.23-vclamp-inplace-alias.py` 实跑 PASS | `T.vclamp.md` §2.4 增补一个原地示例（`T.vclamp(x_ub, x_ub, lo, hi)`），或在 §2.3 特殊限制说明标注「src/dst 允许同 buffer（前端 `max` 先读后 `min` 覆写）」 | P |
| Test-harness | `@tilelang.jit(out_idx=[1])` 且 prim_func 无运行时 `shape` 参数的 kernel，调用时只接受显式输入个数（无 `shape` 则传 1 个实参）；写最小 repro 时按任务 kernel 的 `(x, N)` 双参习惯传参报 `Expected 1 inputs, got 2`。 | 本任务 `repro/PL-1.23-vclamp-inplace-alias.py` 首跑 stderr：`jit_npu.py:1281 ValueError: Expected 1 inputs, got 2` | 在 repro/模板说明补一句：无动态 `shape` 参数的 out_idx kernel 按「输入张量个数」传参，勿套用带尾块守卫 kernel 的 `(x, N)` 形态 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.vclamp(src, src, lo, hi)` 整 buffer 原地别名（scalar 边界）设备可用且逐位一致：前端 `npuir_max(src,lo,dst)` 先读、`npuir_min(dst,hi,dst)` 后覆写，故 `src is dst` 安全——可直接作计算域 buffer 的链尾承载形态，省一个 fp32 结果 buffer。`T.vmin(x, scalar, x)` 原地形态同样设备通过。 | P | `repro/PL-1.23-vclamp-inplace-alias.py` 实跑 PASS（fp32 角点/±inf/NaN 与 `torch.clamp` 逐位一致）；本任务 kernel L0–Boundary 全 `max_diff=0.0`（`vclamp(xc,xc,...)` / `vmin(xc,scalar,xc)` / `vmax(xc,lo_ub,xc)` 三形态均复现） | `examples/hardtanh/_make_hardtanh_kernel/repro/PL-1.23-vclamp-inplace-alias.py` | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-13（设备实测） | elementwise.md **PL-1.23-vop-chain-inplace-alias**（补 `vclamp`/`vmin` 原地元算证，回填 repro-missing） |
| 设备 `VMaxOp`/`VMinOp`（即 `vclamp`/`vmin`/`vmax`）实施 **NaN 传播**语义（非 IEEE `maxNum`）：`vclamp(nan, lo, hi) = nan`，与 torch `clamp`/`hardtanh` 契约一致（`equal_nan=True` 比对通过）。DESIGN §9.1 R-4「设备 NaN 语义未验证」由此**消解**为已实证。 | D | 本任务 `Boundary-nan` 用例（`repeat([nan,0,1,-1])` fp32）`max_diff=0.0` + `assert_close(equal_nan=True)` PASS；`repro/PL-1.23-vclamp-inplace-alias.py` 断言 [3] 实跑 PASS | `examples/hardtanh/_make_hardtanh_kernel/repro/PL-1.23-vclamp-inplace-alias.py`（断言 [3]） | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-13 | pattern-library/traps-runtime.md 或 constants.md（补 VMax/VMin NaN 传播语义条目，绑定 910_9362 版本戳） |
| 编译期边界折叠 4 形态（`vclamp` 默认 2 op / `vmin` 1 op / `vmax` 1 op / `copy` 0 op / `fill` 常量）设备端全部逐位一致（fp32/fp16/bf16 三 dtype 各 4 形态 L1 全 `max_diff=0.0`）；`hi=+inf` 折叠用预填 fp32 常量 buffer `lo_ub` 作 `vmax` 第二操作数、`vmin` 用 scalar src2——两条工作路径均设备通过，未见操作数规则拒绝。 | P | 本任务 L1 `fold-lo-inf`/`fold-hi-inf`/`fold-both-inf`/`fold-const` 四项 × 三 dtype 全 PASS `max_diff=0.0` | 无（任务内 L1 覆盖；并入上条 repro 的 local fold 亦可） | elementwise.md **PL-1.28-const-scalar-param-fold**（补 clamp 类恒等/退化边界折叠的设备实证） |

### Transferable Lessons（可迁移教训）

- 逐元素 clamp 类迁移：**默认 `(-1,1)` 两侧有限不触发折叠**（2 op = 源最短闭式，诚实记录）；折叠只面向 `±inf` / `lo==hi` 退化参数域，别把「默认无收益」误写成优化收益。
- `T.vclamp` / `T.vmin` 支持 scalar 边界（含原地 `src is dst`），但 `T.vmax` 文档仅张量-张量：`hi=+inf` 单侧折叠应预填 fp32 常量 buffer（循环外 `T.fill` 一次、只读跨 tile），别赌 `vmax(x, scalar)`。
- bf16 必经 fp32 中转（`vclamp`/`vmin`/`vmax` dtype 矩阵无 bf16）；`vcast` 升降统一 `round_mode="rint"`；本任务 fp16·bf16 全层级 `max_diff=0.0` 佐证 fp32 中转与契约 golden 逐位一致。
- 设备 `VMaxOp`/`VMinOp` 传播 NaN（非 `maxNum`）——clamp 类算子的 NaN 边界无需在 kernel 内特判，`equal_nan=True` 门禁即可。
- 若 DESIGN 伪代码已按元算 dtype 矩阵与操作数规则避坑（本任务设计期即规避 `vmax` scalar、`vcast` 非 rint、条件 `alloc`），Stage 3 常可**零 retry 直译通过**；盲写易在 bf16/标量操作数/条件 alloc 三处踩坑。
