# RETROSPECTIVE — prefill_blocksolve_A_bthd（GDN prefill 函数 2/6）

## Stage 1 (design) Retrospective — first_design — 2026-10-17

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 迁移任务的关键"能不能这么算"判断（Cube 链式回喂、`initC=False` 落盘后累加）在 skill 文档里只有"读 EXPERT_FINDINGS"的间接指引，实际必须自建探针才能定案；探针耗时 ≈ 全 Stage 的 40% | DESIGN.md §3.4/§9；`history_version/design_probe_doubling_cube.py` | 在 `tilelang-op-design/references/migration-analysis.md` §6.2 增加一句："算法候选的**可行性**由 Cube 侧 op 代价与回喂通路决定时，先跑 D-3 探针再排候选" | R |
| Design-flow | `verify_equiv.py` 的"违反率"判据口径未在 skill 中给公式，首版用 fp32 绝对容差导致 3 项假 `EQUIV_FAIL`（源式的 fp16 门表舍入与 fp16 输出量化被误判） | `verify_equiv.py` 头部注释 + DESIGN.md §1.6.1 判据行 | 在 `tilelang-op-design/SKILL.md` Phase 2 第 5 步补充："违反率判据 = `|cand-ref| <= rtol*|ref| + 2*eps_out*max|ref|`（rtol 取输出 dtype 口径）" | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Expert 下 fp32(HF32) gemm 64×64×64 可用且 err ≈ 3e-8；`initC=False` 可累加进"已 store 过"的 L0C（err 2.9e-8） | D | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_accum_form.py`（V1/V2/V3） | `python3 design_probe_accum_form.py` | tilelang 0.1.2 dev root + CANN 9.0.0 / 910B3 | pattern-library/constants.md |
| 64×64 fp32 的 gemm/store/load 平均 **0.25 µs/op**（6 op 一级 = 1.52 µs，`reps` 差分法） | D | `history_version/design_probe_doubling_cube.py` PROBE-TIMING | `python3 design_probe_doubling_cube.py` | 同上 | pattern-library/constants.md |
| 回喂读回 **`out_idx` 张量**在 CANN 9.0 上不是"数值错"而是 **bishengir SIGSEGV**（比既有 K2 结论更硬） | D | `history_version/design_probe_doubling_cube.py` 首版（读 `out_t1` 即崩）；`EXPERT_FINDINGS.md` §12.1 K2 | 同上（首版形态） | 同上 | pattern-library/traps-compiler.md（追加到 CONST-store-fixpipe-gm-only 相关条目） |
| L0C(fp32) → GM(**fp16**) → L1 回喂在 64×64、`initC=False` 形态下数值错（err 1.875），与 probe 13 的"可用"结论**不矛盾但形态敏感** | D | `design_probe_accum_form.py` V4 | 同上 | 同上 | pattern-library/traps-runtime.md |
| 三角逆的"依赖性"而非 "FLOPs" 才是 NPU 选型主导项：同一语义下 25 gemm(47 µs) vs 11 gemm(11 µs)，MAC 反而多 8× | P | DESIGN.md §1.6.0 R3/R4 | `verify_equiv.py`（仅数值面） | 同上 | algorithm-candidates.md（三角求解族） |

### Transferable Lessons（可迁移教训）

- **迁移"小矩阵求逆"类算子时，先算 op 数再算 FLOPs**：NPU 上 16³ gemm 与 64³ gemm 都是一次发射，
  MAC 差 64×；源实现"少算乘加"的分块策略在 GPU 上赢、在 NPU 上输。判断依据用"依赖级 gemm 数 ×
  实测 op 代价"，不要用 MAC 数排序候选。
- **log-doubling（`A = Π(I+N^{2^i})`）是三角逆在"无片上回喂"硬件上的正确形态**：11 个 gemm、
  无行串行链、天然 fp32 累加；`initC=False` 让"加 T_j"这一步零成本。凡"下三角/幂零"结构都可套用。
- **`verify_equiv.py` 必须先把容差口径对齐"输出 dtype"**，否则源式自身的舍入（fp16 门表、fp16 中间块）
  会把"更准的候选"判成不等价——首版就因此得出 3 个假 FAIL。
- **探针要"分层二分"**：一次写完整链（Gram+掩码+5 级链）失败时无法定位；本次先二分出
  "读 `out_idx` 张量 = 崩"再拼装，总耗时反而更低。

---

## Stage 1 (design) Retrospective — revision 1 — 2026-10-17

> 触发：Stage 2 检视不通过（`REVIEW.md`：4 阻塞 + 9 建议）。本节的"哪类设计判断被推翻"比数值更重要。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow / D-1 | `verify_equiv.py` 只从"给定 `S`"出发验算，**看不见 `S` 本身的构造**——于是"广播方向写反"和"负号丢失"两个阻塞级缺陷在 D-1 全绿的情况下漏到 Stage 2 | DESIGN.md v0 §3.1/§3.3 vs REVIEW 问题 1/2；本版 `verify_equiv.py` 的 A6 与两条反例对照 | `tilelang-op-design/SKILL.md` Phase 2 第 5 步补一条："D-1 的用例必须覆盖**从原始输入到采纳公式输入**的完整构造链（伪代码逐句展开），并对每个已发现的符号/方向类缺陷留一条**反例对照**（必须 FAIL）" | R |
| API-mapping | 迁移设计的"输出落盘"只查了 `store_fixpipe` 的 dtype/配对表，**没查 rank 对齐规则**（`size` 与 4D dst 的交互），导致 64 行 token 被写成 (head,64) 平面 | v0 §3.2 S7 / §1.6.3 L4；`T.store_fixpipe.md` §2.1 + 前端 `copy.py` `buffer_load_to_tile_region`；本版 `design_probe_store_layout.py` | `references/ascend-constraints.md` 增"输出落盘"检查项：dst 带 batch/head 轴时，必须用真实 shape 复核 region（`size` 只对齐最后维），或直接跑一个 8 KB 最小落盘探针 | R |
| Doc-authority | 三处"文档签名 ≠ 实现"造成无效尝试：`T.vcast(src,dst,"rint")`（第 3 位置参是 `size`）、`T.arange` 掩码链静默全 0、`T.transpose` 的 `size` 参数会 `IndexError` | 本版 `design_probe_vcast_shapes.py` / `design_probe_mask.py`；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §1 vs `customize_npuir.py:750` | 在 `references/info-sources.md` 的优先级表加一行："**函数签名类事实以 `tilelang/language/*.py` 的实现为准**，文档签名只作索引；设计期用最小微探针确认（≤1 分钟级）" | R |
| Probe-budget | 本次为闭环 3 个真机问题用了 4 个探针（超出 D-3 的"≤2 探针"预算），但每个都 ≤2 min 且都是"照 REVIEW 要求不得留给 Stage 3"的项 | 本文件 §10 探针清单；`REVIEW.md` 问题 3 的"设计不应把问题留给 Stage 3" | `SKILL.md` D-3 预算改为"**≤2 个探索性探针** + 若干 ≤2 min 的**形态微探针**（只验 API 签名/region 语义/单 op 时延），后者不计入预算" | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`arange`+`reshape`+`vbrc`+`vcmp` 构掩码在本工具链静默产出全 0**（单 arange / 双 arange 两种写法皆然，无编译错、无告警）；替代 = host 常量 0/1 tile + `vmul`（精确）或 `vmin`+算术 | D | `history_version/design_probe_mask.py`（MA/MC 全 0，MD/MB 精确）；本版 DESIGN §1.6.2 V9 / §9 R15 | `bash _probe_env.sh python3 design_probe_mask.py` | tilelang 0.1.2 dev root + CANN 9.0.0 / 910B3 / 2026-10-17 | pattern-library/traps-compiler.md（新 TRAP） |
| **`T.vcast` 的第 3 个位置参数是 `size`**（`round_mode` 只能关键字传）；位置写法报 `RegionOp: load->indices.size() == ndim ... ndim = 4`（`"rint"` 被当 size → 4 维） | D | `history_version/design_probe_vcast_shapes.py`（M1/M2 位置写法 FAIL，关键字 OK）；`tilelang/language/customize_npuir.py:750` | 同上 | 同上 | docs `T.vcast.md` §1（补签名说明）+ pattern-library/traps-compiler.md |
| **同一 L1 buffer 可同时作 gemm 的 A 与 B（`b_transpose=False`）**：fp32 rel 5.5e-9、fp16 rel 2.8e-5（2 级 doubling 链） | D | `history_version/design_probe_gate_sign.py` PROBE-CHAIN；v0 的"不能同 buffer"断言由此推翻 | `bash _probe_env.sh python3 design_probe_gate_sign.py` | 同上 | pattern-library/constants.md 或 layout.md（gemm 形态速查） |
| **strided region 落盘（64×128 B 段、行距 512 B）在 UB→GM 上正确且只慢 2.29×**（0.194 vs 0.084 µs/8 KB）；相邻 (token,head) 槽零污染 | D | `history_version/design_probe_store_layout.py` | 同上 | 同上 | pattern-library/elementwise.md（与 PL-1.14/PL-1.19 互链：本形态是"128 B 段、行距 512 B"的**可接受**特例，占比放大只在 `<128 B` 行时发生） |
| **Vector 门/符号全链路的设备侧对拍方法**：`e` vs `exp(g_i-g_j)` 与 vs `exp(g_j-g_i)` 双对照 + `N` vs `±β·e·G·mask` 双对照，能把"方向/符号"类缺陷变成可判定的数值差异（1.4e-6 vs 1.07e14；1.1e-7 vs 2.5e-1） | P | `design_probe_gate_sign.py` PROBE-GATE；本版 DESIGN §3.4 / §1.6.1 A6 | 同上 | 同上 | `tilelang-op-design/references/algorithm-research.md`（D-1 用例设计） |
| v0 的 `design_probe_accum_form.py` **V4 分支从未实现**（if-chain 无 `elif mode=="V4"`），"fp16 scratch err=1.875"是未初始化 `out_idx` 的垃圾值 ⇒ 该结论作废 | D（负面） | 代码复核 + 本版 PROBE-CHAIN 的真实 fp16 数据（2 级 rel 2.8e-5） | `sed -n 35,56p history_version/design_probe_accum_form.py` | 同上 | pattern-library/traps-runtime.md（"探针空分支"反面案例；并入 Stage 3 的探针纪律） |

### Transferable Lessons（可迁移教训）

- **"散文等价"与"可执行等价"是两回事**：`A = Π(I+N^{2^i})` 的数学论证可以全对，而把它落到 `vbrc/vsub`
  上仍可能算成 `exp(g_j-g_i)`。凡"符号/方向/广播轴"进入实现的环节，D-1 必须从**原始输入**开始逐句展开，
  并留反例对照证明判据有牙齿。
- **"API 存在"≠"API 语义与文档一致"**：本任务 3 个（`vcast` 位置参数、`arange` 掩码、`store_fixpipe` region）
  都是"签名/语义陷阱"，共同点是**不报错或报错信息指向错误位置**。设计期用 ≤2 min 的形态微探针换掉
  Stage 3 的整轮调试，性价比极高。
- **输出布局是多函数流水线里最容易被低估的接口**：`A[token, head, 64]` 固定 head 后 64 行**天然跨步**，
  任何"直接落盘"的写法都要先问"这个 region 到底是几维"。本任务为此付出的代价 = 1 个 8 KB 中转 +
  1 次 strided region 写（+0.3 µs/任务），远比 head-major 改契约（+74 µs、+64 MB）便宜。
- **探针的每个分支都要证明"它真的跑了"**：v0 的 V4 空分支给出了一个被后续两轮设计引用的"实测结论"。
  纪律：probe 输出里带分支标签 + 断言（`assert mode in {...}`），且"结论数字"必须能追到一次真实执行。
- **计数类断言（gemm 数 / 任务数 / op 数）必须带推导式与源码行号**：v0 的 15/25/11/6144 四个数字都错，
  而它们直接决定了 roofline 与"选 C1 而非 B0"的量化依据。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-17

> 产物：`prefill_blocksolve_A_bthd.py`（kernel + golden + L0/L1/L2/boundary + L0-h dump 通道），
> 模式 Expert（`os.environ[...] = "Expert"` 赋值 + `scoped_ascend_mode`）。
> 结论：算法与数值**已验证**（6 个 golden 用例 + 2 个 dump 通道真机通过，max_abs_err 2.8e-5…2.3e-4，
> 0 NaN，padding 行精确 `e_i`，`N` 严格上三角恒 0，门/符号 `n_err 6.6e-9`），
> 但 **L0-e / L0-i / L1-a / L1-b / Boundary / L2 的编译被工具链缺陷阻断**（见下）。
> 出口：`[DESIGN_ERROR]`（根因在工具链层，非算法/设计层；证据见 CG-2026-0003 第三证）。

### Skill Flow Issues

| 问题 | 现象 | 影响 | 建议 |
|---|---|---|---|
| skill 未要求"编译可行性分档" | Stage 3 的 L0 门槛是"L0-a~i 全绿"，但未提示"同一 kernel 的**部分形状可编译、部分必崩**"这种工具链形态；一轮 `--level L0` 会在首个崩溃用例处中止，看不到全貌 | 首轮误判为"实现有 bug"，多轮无效重编译 | L0/L1 执行前先用 try/except 逐用例跑一遍**landscape**（本次 17 个用例一次跑完，5 分钟定位到"6 过 11 崩"），再把结果与 L0 计划对齐 |
| 设计 §8.1 的 golden 片段对 `valid<64` 不可执行 | `A[0, base:base+valid, hid, :] = solve(I+S[:valid,:valid])` 把 `[valid,valid]` 赋给 `[valid,64]`，torch 直接 RuntimeError | 若照抄会让 golden 在**第一个尾 chunk 用例**就崩，且错误信息与算子无关 | 设计期 golden 片段应做一次"至少跑通一个尾 chunk"的可执行性验证；实现按语义补 `[..., 0:valid]`（列 ≥ valid 恒 0，因为 `N` 的该列也为 0） |
| `_blocked_inverse_fp64` 不能只喂 `S[:valid,:valid]` | 该实现按 16 分块，`valid=36` 只覆盖 32 行 → 交叉校验差 1.0 | 会误报"golden 之间不一致" | 源算法本来就按整 64 行求解（越界行 β=0 ⇒ S 的该行该列均为 0 ⇒ padding 块是单位阵），交叉校验必须用整 64×64 |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **CG-2026-0003 第三证：ConvertHFusionToHIVM 的 `getResult(0)` 越界不是"有 mmadL1 必崩"，而是按 IR 规模/形态分档**——同一份 Expert MixCV 源码（13 mmadL1 + 3 fixpipe）在 `S=64,H=1` / `S=100,H=1` / `S=64,H=16,dk=128` 编译并通过精度（2.8e-5…2.3e-4），在 `S=64,H=2` / `S=1600,H=1` / `S=1024,H=4` / `S=3000,H=16` / `S=4096,H=16` / `S=56,65,H=2` 必崩（`err code: -11`）。IR 级二分：删 1 个 `nd2nz`（R3-a 的 `kz→l1k`）→ 通过；删 6 级链（-20 op）→ 通过；给 Vector 段**加** 2 个 UB buffer → 原本通过的形状转为必崩 | D（负面，含"通过形状亦可能被静默误编译"） | 本文件 `--level L1` 首例即崩；逐用例 landscape 见 RETROSPECTIVE 上文；IR 存证 `/tmp/opencode/fail_shim.npuir`（全量）+ `--mlir-print-ir-before-all` 定位到崩溃 pass = `ConvertHFusionToHIVM` | `repro/TRAP-hfusion-mmadl1-segv.py`（good+blocked 双建，末行 verdict） | tilelang 0.1.2 dev root + CANN 9.0.0（bishengir 1.1.0/428ab8fdab46）+ 910B3 / 2026-10-17 | `.agents/evolution/capability-gaps.md` CG-2026-0003（已 +1 occurrence，tasks/last_seen 已更新） |
| **非方形 L1 tile（`[64,128]` fp16）同时作 `b_transpose=True` gemm 的 A 与 B 会 SIGSEGV；方形 `[64,64]` 共享同一 buffer 正常**。2-op 微探针里两种形态都通过 ⇒ 也是"交互触发"（不是单一 op 非法） | D | 逐形状实现：dk=128 首版直接崩；改为 A/B 各一块 L1（同一 GM 区复制两次，字节相同）后 L0-b/L0-g/L2-32768 均可编译；`/tmp/opencode/probe_gram_128.py`（4/4 通过，说明单独测不出来） | `python3 /tmp/opencode/probe_gram_128.py` | 同上 | pattern-library/traps-compiler.md（新 TRAP：共享 L1 操作数仅对**方形** tile 安全）+ DESIGN §3.4 的"同一 L1 可同作 A/B"需加**方形前提** |
| **`T.copy` 的动态 extent（`x[b:b+vv]`）本身不是崩溃源**：Cube（GM→L1 动态 extent）与 Vector（`g/β` 列切片 + `(64,1)→(1,64)` 转置落位 + 动态 extent）**单独**都能编译运行（4/4 通过），只有与"同一 L1 的第二次 MTE2 写 + 12 gemm 链"同时出现才崩 | P | `/tmp/opencode/probe_dyn_extent.py`（cube-static/cube-dyn/vec-static/vec-dyn 全 BUILD+RUN OK） | 同上 | 同上 | traps-compiler.md（"动态 extent 无罪，别把它当作崩溃的第一嫌疑人"） |
| **尾 chunk 语义在真机可判定的三段断言**：`torch.isnan(out).sum()==0` + `A_pad[total_tokens:]` 每行 `== e_i`（严格 `== 0` 误差）+ `A_pad` 预填 sentinel（1024.0）后"0 个 sentinel 残留"⇒ 把 R3-a/R3-b 与"落盘 exactly-once"同时变成硬断言；L0-d（S=100）实测 28 行 padding 全 `e_i`、0 sentinel | P | 本文件 `_run_case` / `_check_padding`；L0-d/L0-g/boundary 用例 | `python3 prefill_blocksolve_A_bthd.py --level L0` | 同上 | tilelang-op-develop 的测试模板（尾块三断言 + sentinel 覆盖法） |

### Transferable Lessons（可迁移教训）

- **凡 Cube（`T.gemm`）算子，先把"编译可行性"当成一个独立变量**：同一份源码按形状分档可能
  "6 个能编、11 个必崩"（CG-2026-0003）。Stage 3 第一步应是**逐用例 landscape**（try/except 全跑），
  不要用 `--level L0` 的首个崩溃当"实现有 bug"的证据；对 Cube 算子，L0 计划的形状选择还要额外
  覆盖最小/最大 shape，别把"能编译"当成必然。
- **CUDA 经验里的"共享操作数 buffer"在 NPU 上只在方形 tile 上安全**：`[64,64]` 同 buffer 作 A/B
  实测 rel 5.5e-9，`[64,128]` 直接 SIGSEGV（且 2-op 微探针测不出来）。凡 `dim_k`/K 维非方的 L1
  操作数，默认**A/B 各一块**，用"同一 GM 区复制两次"保语义（+1 次 MTE2）。
- **动态 extent 不是崩溃的头号嫌疑人**：它单独可编译可运行；Cube 侧崩溃时的第一嫌疑人应是
  "同一 L1 buffer 的写入次数 + gemm 链长度"这类**规模/形态**因素。
- **golden 的交叉校验必须换算法不换语义**：本任务用"源分块前代（fp64 全 64 行）"对拍
  `torch.linalg.solve(I+S)`，实测 3.5e-18，一举抓住两处实现坑（只解 `valid` 子块、只写 `valid` 列）。
  凡尾块存在，交叉校验必须包含一个 `valid<64` 的用例。
- **落盘覆盖可以用 sentinel 而不是 `torch.empty` 来证明**：预填一个在 fp16/bf16 都精确可表示的
  哨兵（1024.0），跑完断言"0 残留" ⇒ 一次断言同时覆盖"每个任务都写"与"没写到别的 head/行"，
  比事后看 NaN 更早、更直接。
