## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目未覆盖 clamp 家族特有的 `lo>hi` 顺序敏感性；R1 候选需靠恒等变形自查补足 | `examples/clamp/_make_clamp_tensor_kernel/DESIGN.md#1.6.0` R1 候选 1/3/4（均因 `lo>hi` 非等价被证伪） | 在 algorithm-candidates.md `ALG-elementwise` 条目补一行「min/max 顺序敏感类（clamp/clip）：`max(min(x,hi),lo)` 与 relu 分解在 `lo>hi` 时非等价」 | D |
| Design-flow | 同族异 op 的先例（hardtanh）计算方法为「编译期边界常量折叠 + 运行时 `shape: T.int32`」，直接模板化会与本算子（张量边界、N 编译期特化）冲突；skill 未显式提示「边界常量 vs 张量边界」是折叠可用性的分水岭 | `examples/hardtanh/_make_hardtanh_kernel/DESIGN.md#1.6.1` vs 本 DESIGN#1.6.1 E1 | 在 skill Phase 2 或 algorithm-research.md 增加一句：clamp 家族须先判定边界是「编译期常量」还是「运行时张量」——后者不存在常量折叠空间 | P |
| Constraint-doc | ascend-constraints.md §4「host 侧仅允许视图操作」与 harness Op 层既有的 `expand().contiguous()` 广播物化存在张力，标准未说明「既有冻结契约 vs 新设计」的边界归属 | 本 DESIGN#9.1 R-3；`examples/TileOPs/tileops/ops/elementwise/clamp.py` L115–L119 | 在 ascend-constraints.md §4 补一句：harness 既有 Op 层预处理契约（O6 preserved）默认继承，不因 kernel 重设计强制改写；如需合规须显式列为跨契约变更 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| tensor-bound `T.vclamp`/`T.vmin`/`T.vmax` 设备原生传播 NaN（非 IEEE maxNum）且 `lo>hi→hi`，故 CUDA 源码的显式 isnan 守卫在 NPU 逐位冗余 | D | `examples/clamp/_make_clamp_tensor_kernel/history_version/design_probe_clamp_nan.py`（PROBE PASS，断言 [1]–[4b]）；对照 `verify_equiv.py` 非门禁诊断「maxNum vs 守卫式」2 点违反 | `python3 examples/clamp/_make_clamp_tensor_kernel/history_version/design_probe_clamp_nan.py` | tilelang 0.1.2+c3a94f1d561cd941ca19577d5ef74bd635bd292a / CANN 9.0.0 / Ascend910_9362 / torch_npu 2.7.1.post4 / 2026-10-13 | pattern-library/elementwise.md（扩 PL-1.23 第 5 证）+ constants.md 候选 |
| clamp 家族 `lo>hi` 顺序敏感性：`min(max(x,lo),hi)` 与 `max(min(x,hi),lo)` / relu 分解 / median 形式非等价（6.3 万点违反，max_abs≈5.0） | D | `examples/clamp/_make_clamp_tensor_kernel/verify_equiv.py` 非门禁诊断（fp32/fp16） | `python3 examples/clamp/_make_clamp_tensor_kernel/verify_equiv.py` | 同上（CPU fp64 torch 参照，无设备依赖） | pattern-library/traps-*.md 或 algorithm-candidates.md ALG-elementwise |
| 同族迁移先例承接的第三类前提：**边界形态**（编译期常量 vs 运行时张量）决定「编译期常量折叠」类优化是否可用 | P | `examples/hardtanh/_make_hardtanh_kernel/DESIGN.md#1.6.1`（常量折叠 20/20 PASS） vs 本 DESIGN#1.6.1 E1（张量边界 → 不可折叠） | none（设计期结构判据，条件 = 对同族两算子分别核对边界形态） | 同上 | pattern-library/elementwise.md（PL-1.27 前提清单补第 ③ 项） |

### Transferable Lessons（可迁移教训）

- TileOPs harness 的 elementwise 家族 Op 层**已把广播物化并展平为 `(N,)`**（`_expand_flat = expand().contiguous().view(-1)`），kernel 侧一律按扁平一维设计，不要在核内重做广播（需改冻结契约且 vclamp 张量边界要求同 shape）。
- 迁移 clamp/clip 类算子前先核对三件事：① 边界是**编译期常量还是运行时张量**（决定能否常量折叠）；② `min`/`max` **顺序**是否可交换（`lo>hi` 语义使顺序敏感，属正确性问题）；③ 设备 max/min 是否传播 NaN（决定源显式守卫是否为冗余）。三者都用机器验证（verify_equiv.py / 设备探针）留痕，不要凭先例推断。
- 设备端「v-prefix 原语是否传播 NaN」是可 10 分钟内探针测定的**可迁移设备语义**（张量边界 `vclamp`/`vmin`/`vmax` 一次测清）；设计期测定后，CUDA 源里的 `isnan`/`if_then_else` 类守卫可整体消去（8 op → 2 op）。
- 同 elementwise family 内各算子的 kernel **输出约定**（`out_idx` / 是否带运行时 `shape: T.int32`）逐算子不同：若 N 是编译期常量则优先沿用「无运行时标量 + 静态尾块」（`tail = T.min(block_size, N − off)`），使入参个数与冻结 wrapper 调用点天然一致，免改 wrapper。

## Stage 2 (design-review) Retrospective — review-v1 — 2026-10-13

> 复盘对象 = 本次**检视流程**；设计问题已随 REVIEW.md 流转给 Stage 1，不重复抄入。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mech-check | D-1/D-5 机械复核只覆盖 `verify_equiv.py` 与 `design_calc_check.py`，**不含**设计期设备探针（D-3）；当采纳优化项的前提是「设备语义断言」（本任务 = NPU 原生 NaN 传播）时，reviewer 必须自行定位并重跑探针，否则等价性机器验证只证了「前提下的冗余」而未验前提本身 | 本次需手动 `python3 history_version/design_probe_clamp_nan.py` 才拿到 PROBE PASS；`verify_equiv.py` #1 用 fp64 `torch.minimum/maximum`（propagating）作参照，前提未在 D-1 内证 | tilelang-design-review skill Phase 1 第 7 步补一句：若 §1.6.1 采纳项依赖设备/硬件语义断言，须定位并重跑设计期探针脚本，把结果计入「机械复核」段 | P |
| Mech-check | `design_calc_check.py` 对纯 Vector 一维持久化算子的 `ub_budget` / `core_split` 均返回 `skip`（无法解析），实际依赖 reviewer 手工复算，机检对这类算子几乎空转 | 本次实测 `ub_budget`/`l0c_budget`/`core_split` 三 skip、仅 `r3_metrics` pass；UB 与分核均由人工复算（§4.5 / §5.5） | 在 design_calc_check.py 增加「纯 Vector 一维 persistent」解析分支（从 §4 表格 + §5.5 三要素抽取），或在校验清单中把该形态明确标注为「人工复核为主」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.alloc_ub` 在 Developer 形态下可用（`T.alloc_ub.md` §2.1 文本标注其为 `[Expert mode]`、`alloc_shared` 为 `[Developer mode]`，二者标签与实测不符） | D | `examples/hardtanh/_make_hardtanh_kernel/_make_hardtanh_kernel.py` L144–L147（Developer + `alloc_ub`，L0–Boundary 全绿）、`examples/elementwise/vec_add_1d.py` L28–L30、`T.vmin.md` §2.4 / `T.vcast.md` §2.4 示例均 Developer + `alloc_ub` | `python3 examples/hardtanh/_make_hardtanh_kernel/_make_hardtanh_kernel.py --level all` | tilelang 0.1.2+c3a94f1d / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | docs/Tilelang.language/内存操作/T.alloc_ub.md §2.1（更正模式标签为「Developer/Expert 均可用」） |

### Transferable Lessons（检视流程可迁移）

- 迁移任务的维度 0 与维度 8 存在**证据链合流**：维度 0 的「耦合性判定」（源守卫是否硬件特有）与维度 8 的「采纳项等价性」共享同一前提（设备 max/min 是否传播 NaN）。检视时对这类「跨维度共同前提」应一次查证、两处引用，避免只看设计自述的「已核对」。
- 否决类候选（本任务 R4-B/C 核内广播）的检视重点不是「能否实现」而是「**权衡量化是否落到本算子**」：设计以「冻结契约 + manifest 零广播」两点否决，而非断言不可实现——此时 reviewer 应确认契约确为冻结（读 wrapper/Op 模块头 O6 字段）与 manifest 确无广播 workload，两点都成立方可 pass。
- 当设计把某代价论断显式标注为「未文档化假设」时（本任务 R4-C 的 stride-0 代价），检视只需确认该标注存在且替代通路有已验证先例，不要求设计给出文档级证明——但要回文档核对「是否真的没写」（本次核对 `docs/Tilelang.language/` 全目录无 `T.isnan`、`T.copy.md` 无 stride-0/gather，标注成立）。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-13

> attempt 链：`first_impl` 单次通过（retry_count=0）。L0=40 / L1=26 / L2=4 / Boundary=9 全过，全层级、全 dtype `max_diff=0.000000e+00`（逐位一致）。DESIGN.md §3.3 伪代码逐行直译即编译通过，未触发任何 retry / precision_fix。三条设计决策（R2 fp32 中转、R3 persistent 分核、R5 无 NaN 守卫）在设备端一次性得到全量正向证据。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Impl-rule | DESIGN 伪代码用 `num_kernels = min(T.ceildiv(N, block_size), vector_cores)`；`T.ceildiv` 在 trace 期可能返回 TIR 表达式，`min()` 作用于其上有比较语义风险（REVIEW 维度 3 已标 warn）。skill 未把「persistent 核数/循环上界常量须用 Python 整数算术」写成实现规则。本任务改用 `(N + bs - 1)//bs` + `min(nl, cores)` + `(nl + nk - 1)//nk`（N/bs 均为编译期 Python int）→ 静态边界零风险通过。 | `DESIGN.md#3.3`（`min(T.ceildiv(...))`）；`REVIEW.md` 维度 3 warn 行；本任务实现 L0–Boundary 全过（5 个核数切换边界用例覆盖） | 在 `tilelang-op-develop` skill Phase 2 补一条实现规则：N/block_size 为编译期常量时，persistent 核数、`num_local_tasks`、`tail` 上界常量一律用 Python 整数算术，勿把 `min()`/整除作用于 TIR 表达式 | R |
| Test-harness | 分层测试的 max_diff 统计聚合若用 `0.0` 初始化并配严格 `>` 更新，会把**全 bit-exact 的正确结果**丢掉（摘要打印 `n/a`），与「全过但无数据」在观感上同型，易误判为测试未生效。本任务首跑即踩：三 dtype 全 0.0 → 摘要 `n/a`。改用 `None` 初始化（`best is None or max_diff > best`）后正常显示 `0.000e+00`。 | 本任务首跑 `--level all` 摘要行 `Summary max_diff per dtype: n/a`（实际 L0 40/40 PASS、max_diff 全 0）；修正后 `bfloat16=0.000e+00, float16=0.000e+00, float32=0.000e+00` | 在 skill 的模板/说明补一句：统计聚合以 `None` 初始化（或首值直接写入），避免严格 `>` + 0.0 种子吞掉全零（bit-exact）结果 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 设备 `VMaxOp`/`VMinOp` 的 **tensor-bound** 形态（`T.vclamp(x, y, lo_ub, hi_ub)`，3 个同 shape UB 操作数）同样实施 NaN 传播 + `lo>hi→hi`，且 fp16/bf16 经 `vcast(rint)→vclamp(fp32)→vcast(rint)` 中转链后与 `torch.clamp(x.float(),lo.float(),hi.float()).to(dtype)` **逐位一致**（atol=rtol=0，equal_nan=True）——CUDA 源的显式 isnan 守卫在张量边界 clamp 上亦逐位冗余（8 op → 2 op）；既证补全 PL-1.23 第 4 证（仅覆盖标量边界）的路由缺口 | D | `examples/clamp/_make_clamp_tensor_kernel/repro/TRAP-vmax-vmin-nan-propagation.py`（实跑 REPRO PASS，fp32/fp16/bf16）；任务内 L0-3/L0-9/L1-bf16-nan/Boundary-nan 全 `max_diff=0.0` | `examples/clamp/_make_clamp_tensor_kernel/repro/TRAP-vmax-vmin-nan-propagation.py` | tilelang 0.1.2+c3a94f1d561cd941ca19577d5ef74bd635bd292a / CANN 9.0.0 / Ascend910_9362 / torch_npu 2.7.1.post4 / 2026-10-13（设备实测） | traps-runtime.md **TRAP-vmax-vmin-nan-propagation**（补 tensor-bound 形态 + repro 回填）；elementwise.md **PL-1.23**（第 5 证） |
| 「无条件全量 alloc（禁止条件 alloc）+ 逐 buffer 显式 `T.alloc_ub`」的 8-buffer 结构在 UB 预算**上限值**上设备编译运行通过：fp16/bf16 `block_size=8192`（=192KB 预算上限，4×fp16+4×fp32）、fp32 `block_size=6144`（上限）与默认 4096/2048 全部 L1 PASS——逐 buffer 显式 alloc 形态**未出现** auto-multi-buffer 1.7x 膨胀导致的 UB 溢出（DESIGN §9.1 R-6 的担忧未兑现），无需预先降 block_size | D | 本任务 L1 `bs=8192`（fp16/bf16）、`bs=6144`（fp32）用例 PASS `max_diff=0.0`；UB 预算表 DESIGN#4.5（fp16 bs4096 = 96KB 手工） | none（现象在 L1 用例内覆盖；如需最小化可裁 8-alloc 骨架） | 同上 | elementwise.md / traps-compiler.md **TRAP-UB-multibuffer-inflation**（补「逐 buffer 显式 alloc 形态在 9362 上未触发 1.7x 溢出」的正向证据） |
| tensor-bound 3 输入 clamp 的 **fp16/fp32 保留源 fp32 中转** 即与契约 golden（`torch.clamp(x.float(),...).to(dtype)`）**逐位一致**（三 dtype × {both/min-only/max-only} × {整除/尾块/广播/0-dim/NaN/lo>hi} 全 `max_diff=0.0`），无需任何精度修复——「投影类算子 fp32 中转 = 契约对齐」在 tensor-bound 变体上复现 | D | 本任务全量 `--level all`：L0=40 / L1=26 / L2=4 / Boundary=9，三 dtype 摘要 `max_diff=0.000e+00` | `python _make_clamp_tensor_kernel.py --level all` | 同上 | elementwise.md **PL-1.17-subfp32-fp32-transit**（补「tensor-bound 投影类 clamp」第四触发/边界） |

### Transferable Lessons（可迁移教训）

- 迁移「三输入张量边界 clamp」类算子：设备 tensor-bound `vclamp`/`vmin`/`vmax` 传播 NaN（repro 已证），CUDA 源的 `isnan`+`if_then_else` 守卫在 NPU 上逐位冗余，可直接删；`lo>hi→hi` 由 `min(max(...))` 顺序天然给出，**切勿交换 min/max 顺序**。
- bf16 必走 `vcast(rint)` fp32 中转（`vclamp`/`vmin`/`vmax` dtype 矩阵无 bf16）；fp16/fp32 保留源 fp32 中转即与契约 golden 逐位一致（本任务全层级 `max_diff=0.0`，无精度重试）。`round_mode` 只能 `"rint"`。
- 「无条件全量 alloc（条件 alloc 非法）+ 逐 buffer 显式 alloc」结构：不要在实现前就按 auto-multi-buffer 1.7x 膨胀提前降 block_size——本任务在预算上限（fp16/bf16 bs8192、fp32 bs6144）实测编译运行通过；先按手工预算取默认值实测，溢出再降。
- persistent 核数/循环上界常量一律用 Python 整数算术（N、block_size 编译期已知）：`nl=(N+bs-1)//bs`、`nk=min(nl, phys_cores)`、`nlt=(nl+nk-1)//nk`，避免 `min()` 作用于 TIR 表达式。
- `out_idx` 与主函数入参个数逐分支对齐：both → `main(x,lo,hi,y)`/`out_idx=[3]`；单边界 → `main(x,bound,y)`/`out_idx=[2]`；**不引入运行时 `shape` 标量**（N 编译期特化 + `tail=T.min(bs, N-off)` 静态尾块），使入参个数与冻结 wrapper 调用点（`_compiled_fn(x,lo,hi)` / `(x,bound)`）天然一致，免改 wrapper。

