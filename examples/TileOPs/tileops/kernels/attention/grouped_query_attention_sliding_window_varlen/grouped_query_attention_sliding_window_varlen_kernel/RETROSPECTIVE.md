# `_gqa_sw_fwd_varlen_kernel` RETROSPECTIVE

## Stage 1 (design) Retrospective — first_design（增量迁移，定长版为骨架）— 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 迁移任务的「同族已迁移姊妹 DESIGN」是最高置信母本；本次定长版 `_gqa_sw_fwd_kernel/DESIGN.md` + `CASE-gqa-sliding-window-migration` 直接给出两相位结构、flag 协议、sync shim、尾带机制与全部陷阱，使 varlen 增量设计只需聚焦 8 条新轴。skill 未显式把「同族姊妹 DESIGN/RETROSPECTIVE」列为迁移首选入口（仅列 examples/ 源码） | `DESIGN.md#0.6`（复用点表）+ `DESIGN.md#0.7`；`pattern-library/cases.md` CASE-gqa-sliding-window-migration | `tilelang-op-design/SKILL.md` Phase M0 §3.1 阅读协议 + info-sources.md 增加：「迁移任务检索同类实现 = 同族**已迁移**算子的 DESIGN.md/RETROSPECTIVE（非仅源码 examples）」 | R |
| Constraint-doc | 运行期 GM 标量读 `cu[i]`（int32）经 slice 起点/extent/循环界/输出地址——本仓 `examples/` 无先例、docs 未文档化，只能设备探针实测 | `history_version/design_probe_varlen_structure.py` variant VARLEN（PASS maxdiff=0.0） | 建议 pattern-library 增「迁移 harness：运行期 cu_seqlens 标量读可用于 NPU 寻址」模式条目 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Developer auto CV-split 不可承载 persistent+gemm+v-prefix——**在本结构上复现**（variant A PASS / B BLOCKED `map::at` / C PASS） | D | `history_version/design_probe_varlen_structure.py`；`DESIGN.md#2.2` | `source env_npuir.sh && python design_probe_varlen_structure.py` | tilelang 0.1.2+76e45509964c313ceca43930cceff43c7820551c + Ascend910B3 + CANN 9.0.0 | `.agents/evolution/capability-gaps.md` CG-2026-0010（occurrences+1）+ `traps-compiler.md` |
| **运行期 per-sample `cu_seqlens` 标量 GM 读可用于 NPU 寻址**（slice 起点/extent/循环界/输出地址，含 ghost 块 real_m=0）——设备 bit-exact | D | `history_version/design_probe_varlen_structure.py` variant VARLEN（maxdiff=0.0） | 同上 | 同上 | `traps-runtime.md` / `cases.md`（varlen attention 迁移模式） |
| **「copy int32 数组→UB 再标量索引」路径不可用**：`InternalError: Check failed: cannot find variable cub`（1-D UB int32 buffer + 标量索引读取在 Scope 内被拒） | D | `history_version/design_probe_varlen_structure.py` variant VARLEN_COPY（BLOCKED） | 同上 | 同上 | `traps-compiler.md`（新条目候选；与 CG-2026-0005 前端 API 缺口相关） |
| **varlen per-row 列 OOB 惩罚阈值必含行号**：`j ≥ tn_real ⟺ diff ≥ (tn_real−1)−i`（绝对列条件不可写成 `diff ≥ tn_real`；首版探针因省略行号项 4/7 配置不 bit-exact，修复后 7/7） | D | `history_version/design_probe_varlen_mask.py`（7/7 bit-exact）；`DESIGN.md#0.6 V2` | `python design_probe_varlen_mask.py` | 同上 | `attention.md`（varlen sliding-window 掩码模式）+ `traps-runtime.md` TRAP-npuir-stale-ub-tail-columns-nan 关联 |
| **offset 平移掩码的运行期标量须走 `kid·0.0 + (int_expr)` 顶层 Add 形态**，且 `vsub` 标量恒作 src2（`K_lo−diff` 以 `−(diff−K_lo)` 实现）；裸 `tir.Cast` / 标量作 src1 触发 `NpuirOperand::FromExpr cannot handle tir.Cast` | D | `history_version/design_probe_varlen_mask.py`；`DESIGN.md#9.1` 第 17 条 | 同上 | 同上 | `traps-compiler.md` TRAP-expert-v-operands（本次为第 4 证） |

### Transferable Lessons（可迁移教训）

- **迁移任务先精读「同族已迁移姊妹」的 DESIGN.md + RETROSPECTIVE + cases.md 案例档案，再动手设计**：母本已封装的 Expert 两相位骨架（双 Scope/ws 三族/scales 延迟回放/transpose-free lse/sync shim/尾带机制）可逐语句复用，设计精力应集中在本算子的**语义增量轴**；从零重做会重复母本已排除的坑。
- **运行期 per-sample 长度（cu_seqlens）在 NPU 上用核内 GM 标量读承载是可行的**（slice 起点/extent/循环界/输出地址全可用，device bit-exact）；但**「copy 到 UB 再标量索引」不可用**——两者都要设备探针实测确认，勿凭 GPU 直觉假设。
- **varlen 尾带机制必须常开**（运行期 `kv_len % bn ≠ 0` 无法 trace-time 门控）：N 载体每核零初始化 + 全宽列 OOB 惩罚 + `l1_b` zbuf 零初始化三件套缺一不可（加法掩码清不掉 NaN，`0×NaN=NaN`）。
- **绝对列条件（列 OOB）在 diff 空间必须含行号**：`j ≥ T ⟺ diff ≥ T − i`；写成 `diff ≥ T` 是静默错值——凡「掩码条件从绝对下标改写为块内 diff 空间」都须逐项端点代入校验。
- **flag 预算守卫在运行期 loop_count 下改为解析上界**（`ceildiv(bm+wl+wr, bn)+2`），不可沿用编译期逐块扫描（长度运行期未知）。
- **结构红利数字必须「表达式 + 数值」且区分口径**：块数削减（−16.7%）≠ FLOPs 削减（−24.9%）——两者分别复算，勿混用（同族姊妹 Stage 2 曾因块数算术不可复算被判不通过）。
- **运行期标量进 v-op 阈值**：用 `kernel_id·0.0 + (int_expr)` 顶层 Add 形态（裸 `tir.Cast` 被拒）；`vsub` 标量恒作 src2，`scalar − buffer` 以 `−(buffer − scalar)` 实现。

## Stage 2 (review) Retrospective — first review — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py`（D-5）对本设计的 UB/L0C/core_split 三项**全部 skip**，等价于机械下界失效、须全量人工复算：(a) `ub_budget` 因 §4.5 标题为 `### 4.5 UB 内存预算` 但表体无 `**UB**` 级标记、level 正则未命中；(b) `l0c_budget` 因块尺寸以 `l0_c [64,256]` 形态给出、`block_M x block_N` 正则未命中；(c) `core_split` 因设计用 `ceildiv(max_seqlen_q,bm)` 而非 `M`/`N` 口径、且 §5.5 逻辑核数以表列出无单值 | `REVIEW.md#机械复核`（三项 skip + 人工复核结论）；`examples/…/_gqa_sw_fwd_varlen_kernel/DESIGN.md#4.5/#5.2/#5.5` | `.agents/tools/design_calc_check.py`：UB 表 level 识别增补「标题含 UB/L1/L0C 且表体行含 `ub_`/`l0_c` buffer 名」路径；core_split 支持 `ceildiv(max_seqlen_q|S,bm)×H×B` 形态或改为读取 §5.5 表 | R |
| Review-flow | 维度 8「复杂度算术复算」判据未区分**权威单元格**与**非权威单元格（比对行/中间派生量）**：本设计的选定候选/headline 数字（E=15360、FLOPs=103.2G、HBM=167.8MB、fabric=4.5GB、D-2）全部可复算且正确，但 §1.6.0 R3 的两处非权威单元格（#2 per-core ws「1.05MB」vs 行内公式 0.47MB / §4.3 9.8MB；#0/#1 基线行 fabric 合计 4.7GB vs 分量 4.53GB）自相矛盾——「差异不可解释即 fail」的字面口径会导致对非权威单元格的过重判罚 | `REVIEW.md#8`（判定 warn 非 fail 的推理）；DESIGN.md §1.6.0 R3 表 | `tilelang-design-review/SKILL.md` 维度 8「调研结论正确性」行加注：复杂度复算差异的严重度按「是否影响选定算法/下游章节/结论」分级——权威单元格（选定候选 headline + §4/§5 落地量）不一致 → fail；非权威比对/派生单元格自相矛盾且不改变结论 → warn + 订正建议 | R |
| Review-flow | 检查项缺口：**结构守卫/预算公式与其落地表值分歧**（如 §5.2 `L_max_bound = ceildiv(bm+eff_left+eff_right,bn_eff)+2` 公式值 7 vs 同节表值 6 vs §4.3 pad_band 依据的表值 6）不在现有维度表任何一行；两者都是合法守卫（`+1≤16`）故非阻塞，但属可机械核对的「公式↔表值」一致性缺口 | `REVIEW.md#8`；DESIGN.md §5.2(L812 公式 vs L823–826 表) | `tilelang-design-review/SKILL.md` 维度 8 或维度 3 增「派生守卫量（flag 预算/band 上界/pad_band）公式与表值一致性」检查项 | R |
| Review-flow | 检查项缺口：维度 2（UB 预算）未含 `TRAP-npuir-per-copy-ub-staging-sum` 口径（每 `T.copy(GM→UB)` 单独分配 staging、按 copy site 求和、不做 liveness 合并）——本设计 UB `原位复用` 到 ~154–186KB 余量薄，varlen 新增 UB-dest copy（S/O）此口径下可能触边；设计 §4.5/§9.1 未引用该陷阱 | `REVIEW.md#2`（建议）；`.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md` TRAP-npuir-per-copy-ub-staging-sum（910B3/CANN9.0.0，同工具链） | `tilelang-design-review/SKILL.md` 维度 2 通过标准补「若含多处 GM→UB `T.copy`，UB 预算须按 Σ(copy site × tile bytes) staging 口径核算（引 TRAP-npuir-per-copy-ub-staging-sum）」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `design_calc_check.py` 对「标题式 UB 表 + `l0_c [M,N]` 块尺寸 + `ceildiv(max_seqlen_q,bm)×H×B` 逻辑核数」三种常见表达全 skip → 机械下界虚化（本任务 UB/L0C/core_split 三项全 skip，人工补核） | D | `REVIEW.md#机械复核`（3 项 skip 的 detail 原文）；`design_calc_check.py` L86–160/L163–205/L208–252 | `python3 .agents/tools/design_calc_check.py --design examples/grouped_query_attention_sliding_window_varlen/_gqa_sw_fwd_varlen_kernel/DESIGN.md`（ub_budget/l0c_budget/core_split = skip） | python3（advisory，无设备依赖）；样例设计 toolchain tilelang 0.1.2+76e4550 / Ascend910B3 / CANN 9.0.0 | `.agents/tools/design_calc_check.py`（三处正则/口径） |
| varlen ragged attention 的「运行期 `cu_seqlens` 核内标量 GM 读」+「offset 右对齐双侧算术惩罚 + per-row 列 OOB」在 **910B3/CANN9.0.0/本 build** 上经**独立复跑**逐项存活（verify_equiv 11/11、P-V1 A/C/V1 PASS·B/V2 BLOCKED、P-V2 7/7 bit-exact）——Stage 2 机械复核可作同族回归基准 | D | `REVIEW.md#机械复核`；`history_version/design_probe_varlen_structure.py`、`design_probe_varlen_mask.py`、`verify_equiv.py` | `source env_npuir.sh && python verify_equiv.py`；`python design_probe_varlen_structure.py`；`python design_probe_varlen_mask.py` | tilelang 0.1.2+76e45509964c313ceca43930cceff43c7820551c / Ascend910B3（20 AIC/40 AIV）/ CANN 9.0.0 / 2026-10-16 | `pattern-library/cases.md` CASE-gqa-sliding-window-migration（补 varlen 独立复跑证据行） |

### Transferable Lessons（可迁移教训）

- **Stage 2 机械下界（design_calc_check）不可假定有效**：本任务 UB/L0C/core_split 三项**全 skip**，检视者必须逐项人工复算并显式标注「机械跳过，人工已核」——浅层依赖 skip 会把「未核对」误报为「通过」。同类设计（标题式内存表 / `l0_c [M,N]` 块尺寸 / `ceildiv(max_seqlen_q,bm)` 分核）尤须留意。
- **复杂度算术复算须分级**：headline/权威单元格（选定候选的 E/FLOPs/Bytes/ws 落地量、§4/§5 引用值）与§0 自洽 → 决定性；比对行/中间派生单元格（基线行、per-core 中间量）自相矛盾 → 只作 warn + 订正建议。本任务前者全对（E=15360/FLOPs=103.2G/HBM=167.8MB 逐式复算通过），后者两处小矛盾不改变结论。
- **派生守卫/预算量（flag 预算 `L_max_bound`、`pad_band`、band 上界）要同时核「公式 ↔ 表值 ↔ 下游引用」三者一致**：本设计 §5.2 公式(+2) 与同节表值、§4.3 pad_band 依据的表值差 1——两者都是合法守卫故非阻塞，但同型分歧若方向相反（公式 < 真实 loop_count 上界）会直接导致 flag id 溢出，属须机械核对项。
- **UB「原位复用」预算要叠加 per-copy-site staging 口径**：`T.copy(GM→UB)` 的 lowering staging 按 copy site 求和、不做 liveness 合并；显式 alloc 结构（auto-multi-buffer=false）虽不按 1.7× 膨胀，但多处 UB-dest copy + 余量薄（本设计 ~6KB）仍需按 `Σ(copy site × tile)` 复核，勿只核「显式 buffer 之和」。
- **增量迁移检视的高置信入口 = 姊妹已通过 DESIGN/REVIEW 的已闭合结论**：本任务骨架（定长版）已过 Stage 2 + 集成，varlen 只动 8 条新轴——检视应把精力压在新增轴（ragged 寻址/offset 平移掩码/常开尾带/运行期 flag 守卫）并**独立复跑**其设备探针与等价性脚本，而非重审已闭合的母本结论。

## Stage 3 (develop) Retrospective — retry_impl（attempt 2，上一 attempt 空返回零工件）— 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Context-budget | 上一 attempt（`retry_impl` attempt 1）在 **143KB DESIGN.md 整读**后**空返回、零工件**异常终止（无半成品可续）。`tilelang-op-develop` Phase 1 仍写「Read DESIGN.md 全文」——对超大设计文档（>1200 行/150KB）会触发上下文超载空返回，且该失败与代码错误同型、stderr 无法区分 | 本 attempt 按分级读取（先读定长版母本源码，再定点读 DESIGN §0.5/§0.6/§3/§4/§5/§6/§7/§8/§9）+ **先落盘再测**，一次通过 L0→L1→L2→Boundary | `tilelang-op-develop/SKILL.md` Phase 1 + conductor retry 提示词：DESIGN.md 超阈值时强制「目录+grep 定位章节 → 分段 Read」；Phase 2 明确「先写 `{op}.py` 到磁盘再编译迭代」 | R |
| Info-source | 同族**已迁移**定长版 `_gqa_sw_fwd_kernel.py`（1061 行，22/22 通过）是最高价值骨架：两相位结构/ws 三族/scales 延迟回放/双 AIV 半分/sync shim/编译重试**逐语句可复用**，本次只改写 V1–V8 八处 | 本任务 `_gqa_sw_fwd_varlen_kernel.py` 直接以其为模板，L0–Boundary 全绿 | 同上条（info-sources：迁移任务首选同族已迁移源码文件而非仅 DESIGN） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **本容器 `bishengir-compile` 的 SIGSEGV 为进程状态相关的概率性崩溃**：同一 .npuir 在长寿命进程内可**连续 6 次** SIGSEGV 而重试用尽，在全新进程内第 2 次即成功；24 次重试 + 0.3s 间隔后 L0–Boundary 全绿（51/51）。定长版仅 6 次重试——**长 sweep 任务应把重试上限提到 ≥20** | D | `/tmp/varlen_all.log`（`--level all` 首次运行：wl=128 case 连续 6× `err code: -11` → `RuntimeError: NPU IR opt failed`）；`/tmp/varlen_all2.log`、`/tmp/varlen_all3.log`（24 重试后 L0 4 + L1 26 + L2 12 + Boundary 9 全 PASS，exit 0） | 无独立 repro（概率性、依赖进程累积状态；既有 `TRAP-bishengir-sync-instr-mode-attr` / CG-2026-0023 已覆盖该工具链缺陷） | tilelang 0.1.2+2562795b / Ascend910B3（20 AIC/40 AIV）/ CANN 9.0.0 / 2026-10-16 | `traps-compiler.md`（CG-2026-0023 关联：重试上限 ≥20 + sleep）+ `tilelang-op-develop` 编译重试模板 |
| **varlen 8 轴（V1–V8）在 Expert 两相位骨架上端到端可用**：ragged 运行期 `cu_seqlens` 标量读 + offset 平移双侧掩码 + per-row 列 OOB + 常开 ragged 尾带 + packed NHD 寻址 + `(H,total_q)` lse 视图，经 L0 4 / L1 26 / L2 12 / Boundary 9 全绿（bf16 tier-1 flips 由 tier-2/tier-3 分类吸收，无真缺陷） | D | `_gqa_sw_fwd_varlen_kernel.py`（`--level all` exit 0，51/51 out=PASS）；P-V1/P-V2 设计探针 | `examples/grouped_query_attention_sliding_window_varlen/_gqa_sw_fwd_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py` | 同工具链戳 | `cases.md`（CASE-gqa-sliding-window-migration 补 varlen 实现归档） |
| **设计 §5.2 的 flag 预算上界对 `wl<0` 情形不完备**：`eff_left = max_seqlen_q if wl<0` 的推导仅在 `kv_len ≲ q_len+bm` 时成立；KV-cache（`kv_len ≫ max_seqlen_q`，如 q=128/kv=2048）下 `loop_count` 可达 `ceildiv(kv_len,bn)` ≫ 解析式 → flag id 溢出、静默错值。**修正**：`wl<0` 时用 `L_max_bound = ceildiv(total_k, bn_eff)+1`（`total_k` 工厂期已知）；manifest 域 `wl=-1` 用例 `kv_len≤512` 恰好未触发，属**潜伏缺陷** | D | 本实现 `_band_upper_bound()`：`if window_size_left < 0: return ceildiv(total_k, bn_eff)+1`；manifest KV-cache `wl=-1` 用例（L1/L2 kvcache-causal）全绿 | `_gqa_sw_fwd_varlen_kernel.py#_band_upper_bound` | 同工具链戳 | `attention.md`（varlen flag 预算守卫：`wl<0` 分支须含 kv 长度项） |
| **负 offset（`sk<sq`，q 比 kv 长）产生全掩蔽行**：算术惩罚式对全掩蔽行给 `S''=S−1e38`、`m=−1e38` → `P=exp(0)=1`（而非源 `exp(−inf)=0`），lse 退化。**域约束**：`offset = kv_len−q_len ≥ 0` 是算术惩罚式成立的前提（每行≥1 有效 key），与定长版姊妹同；manifest/测试/benchmark 全域 `sk≥sq`，故不触发——Boundary 用例须保持 `sk≥sq` | D | `_run_case(1,[16],[13],...,causal)` → `out max_diff=8.3e-1, lse diff=inf`；改为 `[13]/[13]`、`[13]/[29]` 后 PASS（`/tmp/varlen_all3.log`） | 无（域外用例） | 同工具链戳 | `attention.md`（迁移 harness 域约束：offset≥0；全掩蔽行不受算术惩罚式支持） |
| **`max_seqlen_q` 延迟编译折叠**：接口须保持 `(bm,bn,ns)(q,k,v,cu_q,cu_k,max_seqlen_q)` 逐字形态（wrapper 零改动），但 `num_q_blocks=ceildiv(max_seqlen_q,bm)` 是 trace-time 常量 → 将 JIT 编译**延迟到首次 `wrapped` 调用**并按 `max_seqlen_q` lru_cache；且该 cache 必须**位于 `(bm,bn,ns)` 层之下**，否则每次 forward 重新编译 | P | `_gqa_sw_fwd_varlen_kernel.py`：`_gqa_sw_fwd_varlen_func(bm,bn,ns)` → 内层 `@lru_cache _compile(max_seqlen_q)` → `wrapped`；`_contract_checks` 验证 lru_cache 身份 | `_gqa_sw_fwd_varlen_kernel.py` | 同工具链戳 | `attention.md`（host 侧编译期折叠 + 双层 lru_cache 形态） |
| **ghost 块 0-extent copy 的地址须钳位**：`real_m = clamp(min(bm,q_len−bx·bm),0,·)` 已被 P-V1 证 0-extent 可承载；但 slice 起点 `q_start+bx·bm` 在 ghost 块可越出 `total_q`——用 `q_off = q_start + min(bx·bm, q_len)` 钳到样本尾，保证 0-extent 访存地址在界内 | P | 混合长度用例（L0-4 `[128,256,384]`、B-ghost-msq196）全绿 | `_gqa_sw_fwd_varlen_kernel.py#q_off` | 同工具链戳 | `traps-runtime.md`（varlen ghost 块寻址钳位） |

### Transferable Lessons（可迁移教训）

- **超大设计文档（>1200 行/150KB）禁止整读**：先读同族已迁移源码母本（最高信息密度），再按章节 grep+分段 Read DESIGN（§0.5/§0.6 决策 + §3/§4/§5/§6/§7 落地 + §8/§9 验证/风险）；**先把 `{op}.py` 落盘再编译迭代**——上一 attempt 整读 143KB 后空返回零工件，本次分级读取 + 先落盘一次通过。
- **本容器 `bishengir-compile` 的 SIGSEGV 是进程状态相关的概率性崩溃**：同一 kernel 在长寿命进程内可连续多次崩溃、在全新进程内即成功。**编译重试上限应 ≥20（本次用 24）+ 0.3s 间隔**；6 次会在长 sweep 中偶发耗尽。
- **flag 预算上界（运行期 loop_count）对 `wl<0` 必须含 kv 长度项**：`eff_left=max_seqlen_q` 的推导只在 `kv_len ≈ q_len` 域成立；KV-cache 且无左窗时 `loop_count ≈ ceildiv(kv_len,bn)`，须用 `ceildiv(total_k,bn)+1`。**凡「运行期 loop_count 的解析上界」都要对每个 wl 分支代端点核验**，勿只核 manifest headline 域。
- **算术惩罚式掩码的域前提是「每行 ≥1 有效 key」**（`offset = kv_len−q_len ≥ 0`）：负 offset 的全掩蔽行会被算成 `P=exp(0)=1`。迁移/测试用例须保持 `sk≥sq`；否则需回退到 `-inf` 预填/`vselect` 路径。
- **接口含 host 侧编译期折叠量（如 `max_seqlen_q`）时用「延迟编译 + 该量 lru_cache」**：既保留源调用形态（wrapper 零改动），又把该量变成 trace-time 常量；cache 必须放在最内层（按折叠量 key），否则每次 forward 重编译。
- **varlen ghost 块的 0-长度 slice copy 可行（P-V1），但起点须钳位**（`q_start + min(bx·bm, q_len)`），否则越界地址 + 0 extent 的组合未获证同 P-V1 安全。

