## Stage 1 (design) Retrospective — first_design — 2026-10-16

> 算子：`grouped_query_attention_sliding_window / _gqa_sw_fwd_kernel`（GPU→NPU 迁移，harness）。产物：`DESIGN.md` + `verify_equiv.py`（11/11 EQUIV_PASS）+ 两个设计期探针（P1 编程模式 / P2 带状掩码）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 首选定范式的「直接母本」是先例库中**同型已迁移算子**（MHA prefill Expert 两相位），其 DESIGN 快照（151KB）与 perf_opt 档案是低风险高置信复用的关键——但 tilelang-op-design 的 Phase 3「同类实现检索」入口（Glob/kb_search）未显式提示「同族已迁移算子的 DESIGN.md 优先于源码先例」 | `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel_DESIGN.md`（直接复用 §0.5–§0.6 全套结构） | 在 `tilelang-op-design/references/info-sources.md` 或 SKILL Phase 3 补一条：迁移任务检索同类实现时，须优先查找**同族已迁移算子的 DESIGN.md/RETROSPECTIVE**（不只源码 examples） | C |
| Info-source | 设计期探针（D-3）在「编程模式可行性」这类**非性能常数**问题上同样决定性强，但 hardware-cost-model.md §1 只列了「容量/带宽/常量」三类触发位置，未含「编程模式/codegen 行为」 | `history_version/design_probe_devmode_mix.py`（Developer BLOCKED / Expert PASS，直接决定 §2） | 在 `_shared/standards/hardware-cost-model.md` §1 或 `tilelang-op-design/references/algorithm-research.md` §6 补：涉及编程模式可行性 / codegen 行为的结论必须设备探针实测，不得纸面推断 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Developer auto CV-split 对 persistent+gemm+v-prefix 的**编译期**触发层（BishengIR `std::out_of_range`/`map::at`），补充 CG-2026-0010 的运行期崩溃层 | D | `history_version/design_probe_devmode_mix.py`（变体 A 无向量 PASS / B 加 `T.vmul` BLOCKED / C Expert PASS）；`DESIGN.md` §2.2 | `python history_version/design_probe_devmode_mix.py`（子进程内跑 Developer/Expert 变体） | tilelang `0.1.2+cf6f999389f4f3481da5778aa06d73478684ce2f` + CANN 9.0.0 + Ascend910B3 | `.agents/evolution/capability-gaps.md`（CG-2026-0010 occurrences+1 + 编译期注记） |
| `vcmp` int16 标量化 warning 在 Ascend910B3 / CANN 9.0.0 / 新 tilelang build 上**复现**（跨工具链/SoC 存活），印证 PL-1.11 算术惩罚掩码的选型前提 | D | `history_version/design_probe_band_mask.py`（编译输出 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency`，同链 bit-exact 通过）；`DESIGN.md` §0.6 E-mask / §1.6.3 | `python history_version/design_probe_band_mask.py` | 同上 | `pattern-library/attention.md` PL-1.11（补 910B3/CANN9.0.0 重验注记） |
| 设计期探针脚本模板：需写入**真实 .py 文件**再子进程运行——`python -c` 内联源码使 TVM parser 的 `inspect.getsource` 报 `OSError: could not get source code`，且设备错误会杀主进程 | P | `history_version/design_probe_devmode_mix.py` 首版（`-c` 内联全失败）→ 改临时文件后成功 | `DESIGN.md` §2.2 / `design_probe_devmode_mix.py` 文件头 | 同上 | `pattern-library/repro/README.md`（探针模板纪律） |
| 隔离 harness 中「**仅 Vector Scope** 的 `T.Kernel(N, is_npu=True)`」运行时 fault（retCode=0x26），补上双 Scope（哪怕 Cube 侧空 gemm）后正常——Expert 形态似要求双 Scope 并存 | P | `/tmp/p2c.py`（Vector-only 运行时崩）vs `history_version/design_probe_band_mask.py`（双 Scope 通过） | 构造 Vector-only 最小 kernel 复现 | 同上 | `pattern-library/traps-runtime.md`（Expert MIX 须双 Scope） |

### Transferable Lessons（可迁移教训）

- **同型已迁移算子的 DESIGN.md 是第一复用资产**：本任务直接承接 MHA prefill Expert 两相位（结构、flag 协议、ws 三族、transpose-free lse、尾块机制），新增仅「带状 KV 范围 + 双侧算术惩罚掩码 + 两段式带状分界」三处；先精读先例 DESIGN 再动手，避免从源码重推整条链路。
- **滑动窗口的 KV 循环是 `[k_start, k_end)` 而非 `[0, NK)`**：flag id 用**带内块局部下标 l**（两 pass 与两 Vector pass 迭代同一区间）；flag 预算守卫从「S_kv/bn」变为「(bm+wl+wr)/bn」，`L_max+1 ≤ 16`。被裁剪的全掩码块 = online 递推恒等元（α=1、Δℓ=0、Δo=0），裁剪是**恒等精确**而非近似。
- **双侧带状掩码用「算术惩罚式」而非 `vcmp`/`vselect`**：`pen = max(clamp(diff−K_up,0,1), clamp(K_lo−diff,0,1))`，`S += pen·(−1e38)`——全速 ALU 路径、值无关、NaN 免疫（有限大数哨兵）；`vcmp` int16 在多工具链上标量化（PL-1.11 + 本任务复现）。仅 ragged 尾带域需回退 `vselect`（加法掩码无法清 NaN）。
- **mask 分界的两种形态别混**：MHA 的单侧因果只有一条 K_A 下界；滑动窗口**两侧各一条**（`L_lo_full` 左窗、`L_hi_full` 因果），用两条顺序 `if` 而非复合条件（规避 parser 对复合条件的风险）。
- **设计期探针写文件别用 `-c`**：TVM parser 需真实源文件（`inspect`）；设备错误会杀进程——探针须写 `.py` 文件 + 子进程隔离。
- **Explicit 双 Scope 是 MIX 承载的前提**：Developer auto CV-split 对 persistent+gemm+向量混排不可用（本容器编译期崩），Variant A（无向量）OK / B（加向量）BLOCKED 的对照是干净的判据。

## Stage 2 (design review) Retrospective — 2026-10-16

> 对象：`_gqa_sw_fwd_kernel` 的 Stage 2 设计检视流程自身（结论 = 不通过；2 阻塞 + 3 建议）。设计层问题已随 `REVIEW.md` 流转给 Stage 1，不在本节重复。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | `tilelang-design-review` 的机械复核（D-1/D-5）只规定重跑 `verify_equiv.py` + `design_calc_check.py`，**未规定复跑 DESIGN.md 引用的设计期设备探针（P1/P2）**。本轮 P2 探针（被 §0.6/§1.6.3 当作设备证据引用）实际不可运行，只有人工复跑才发现——若照 skill 默认流程会漏检 | `history_version/design_probe_band_mask.py` 复跑 trace 报 `'int' object has no attribute 'buffer'`；DESIGN.md §0.6/§1.6.3/§10.2 引其结果为「7 组 bit-exact + vcmp 标量化 warning」 | 在 SKILL Phase 1 第 7 步（机械复核）补：**复跑 DESIGN.md 引用的全部设计期探针脚本，结果与设计自述一致方通过**；探针不可运行/与自述不符 → 维度 8 fail（证据链） | R |
| Dimension-coverage | 维度 8「调研结论正确性」的复杂度复算条目枚举四口径（FLOPs/访存/遍数/缓冲），**未显式覆盖 R3 叙述中派生的辅助量（块数/指令数等）**——本轮的错误恰好落在「执行块数 3584（−61%）」这一派生量上（FLOPs 四口径全对，故按现表易判 pass） | DESIGN.md §1.6.0 R3 表 #0/#1/#4、§1.6.1 M4；独立复算 Σloop_count=7680（非 3584） | 在维度 8「复杂度算术独立复算」条目补：**R3 表内出现的全部数值（含派生块数/流量括号乘积）须逐项复算，并核对表内数字间的自洽（如块数×块字节=所写 GB）** | R |
| Dimension-coverage | 维度 8「替代 API 可行性」只判「API 是否存在」，**未覆盖「API 调用形态是否合法」**（标量操作数位置/类型约束）——本轮 P2 探针正是被 `T.vcmp` 不接受裸标量所卡 | `tilelang/language/customize_npuir.py` L59–74/L501–520（`npnir_cmp` 走 `_to_region`，裸 int 无 `.buffer`）；`AscendBinaryOp` 仅第二操作数可为标量 | 在维度 1/8 的 API 可行性条目补：抽查关键 API 的**操作数形态约束**（标量位置、dtype、是否需 let-bound）并与设计用例核对 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 设计期设备探针脚本**必须可复跑**：裸 Python 标量传 `T.vcmp` 等 npuir 比较算子会 trace 失败（`'int' object has no attribute 'buffer'`）；阈值须 let-bound 或 kernel 派生 PrimExpr | P | 复跑 `history_version/design_probe_band_mask.py` 失败；改写阈值为 `sub*0+const` 后同链编译运行正常 | `python history_version/design_probe_band_mask.py`（失败）vs 本 Agent /tmp `probe_mask3.py`（成功） | tilelang `0.1.2+cf6f999…` / CANN 9.0.0 / Ascend910B3 | `tilelang-op-design` references（探针模板纪律）；`_shared/standards/negative-claim-evidence.md` |
| 双侧带状**算术惩罚掩码链**（`vsub/vmax/vmin/vmax` + rint，阈值 PrimExpr）在 Ascend910B3 / CANN 9.0.0 上 7 配置 **bit-exact 复现**——PL-1.11 的选型前提跨 SoC/工具链存活（910B2C→910B3） | D | 本 Agent 独立设备探针（7 组 (s_lo,n_lo,row0,wl)，含 wl=0 与 subid 分半域）= 全部 bit-exact；vcmp/vand/vselect 备选链亦正常编译运行 | 本 Agent 检视期 /tmp `probe_mask3.py`（复现脚本），结果表见 `REVIEW.md` 机械复核段 | tilelang `0.1.2+cf6f999…` + CANN 9.0.0 + Ascend910B3 | `pattern-library/attention.md` PL-1.11（补 910B3/CANN 9.0.0 重验注记） |

### Transferable Lessons（可迁移教训）

- **「证据产物」本身要复跑，不只看设计正文**：设计把探针结论写得很确定（「7 组 bit-exact + warning」），但交付的脚本连编译都过不了、且测的是被否决的形态——检视必须亲手执行被引用的产物，不能只审文字。
- **四口径全对不等于复杂度无误**：派生量（执行块数、流量括号乘积）与四口径同等重要，且要自校验（块数×块字节 = 所写 GB）。
- **API「存在」不等于「该形态可用」**：npnir 二元算子的标量位置约束、vcmp 不接受裸标量等形态限制，须在 API 可行性维度抽查。
- **主选/备选链的设备证据要分开核**：本设计引一条探针同时「验证主选 + 复现备选 warning」，实际产物只含备选链——主选真实形态须另行设备复验（本轮已补，结论成立）。

## Stage 1 (design) Retrospective — revision 1 — 2026-10-16

> 对象：`_gqa_sw_fwd_kernel` 的设计修订（mode=revision, revision_index=1，输入 = Stage 2 `REVIEW.md` 的 2 阻塞 + 3 建议）。产物：订正后的 `DESIGN.md`（覆盖）+ 修复并重跑的 P2 探针 + `verify_equiv.py` 重跑（11/11）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Arithmetic-discipline | 设计正文的**派生算术**（执行块数、fabric GB）在同一表内自相矛盾却通过了 first_design 自检——根因是「结构红利数字」以叙述式给出（无表达式、无逐项推导），生成期无法机械复核。修订把全部此类数字改为「表达式 + 数值」并加逐 bx 复算表 | `DESIGN.md`（v0）§1.6.0 R3：「3584 块（−61%）」与同表「0.46GB」互斥；独立复算 Σloop_count=7680（−16.7%） | 在 `tilelang-op-design` SKILL 的「设计期 roofline（D-2）」/ §1.6.0 模板补：**复杂度/流量表的每个数值须自带表达式推导，且同表数值间须可复算对账**（块数×块字节 = 所写 GB） | R |
| Probe-minimal-kernel | 探针在隔离 harness 里**无输入张量的纯输出 kernel**（`out_idx=[-1]` 且 prim_func 只含输出参数）运行时 abort（aivec `error code 0x800000`）；加任一输入张量即正常——该陷阱未见于现有 traps | `design_probe_band_mask.py` v1→v2 对照；`/tmp/opencode/p2_bisect.py`（plain FAIL / inp PASS）；对照 `examples/elementwise/vec_add_2d.py`（有输入，PASS） | 在 `pattern-library/traps-runtime.md` 或 `tilelang-op-design` references 补：**探针/最小 kernel 须至少含一个输入张量**（纯输出 kernel 运行时崩溃） | P |
| Probe-warning-capture | 编译期 warning（vcmp 标量化）只在**未命中 jit 缓存**时复现——探针重复运行会静默丢失该证据（首跑 True、再跑 False）。修复 = 脚本内置 `tilelang.disable_cache()` 强制复编译 | `design_probe_band_mask.py`（fix 前二次运行 `vcmp_warn=False`；加 disable_cache 后确定性 True + 捕获 warning 文本） | 在探针模板纪律补：**依赖编译期 warning/日志的证据须在脚本内强制复编译**（否则缓存使结论不可复现） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 滑动窗口**执行块数**口径（`Σ loop_count`）与**内段全有效块**口径（`Σ max(loop_count−2,0)`）必须分开表述——后者误当执行块数会与全 causal 相减出 −61% 的假红利（真值 −16.7%） | D | `DESIGN.md`（v1）§1.6.0 R3 复算表；Σloop_count=120/hb ×64=7680；Σmax(lc−2,0)=60×64=3840 | `python /tmp/opencode/recompute.py`（逐 bx loop_count 复算） | tilelang `0.1.2+cf6f999…` / CANN 9.0.0 / Ascend910B3 | `pattern-library/attention.md`（滑动窗口块数口径）；`tilelang-op-design/references/algorithm-research.md` §5 |
| `T.vcmp` 阈值须 **let-bound / kernel 派生 PrimExpr**（`ku = kid*0 + const`）；裸 Python int 触发 trace 期 `'int' object has no attribute 'buffer'`（`npnir_cmp`→`_to_region` 走 `data.buffer`） | P | 修复后的 `history_version/design_probe_band_mask.py`（`ku = kid*0 + KU`，编译运行均正常，ALT 链 7/7 bit-exact） | `python history_version/design_probe_band_mask.py` | 同上 | `pattern-library/traps-compiler.md` TRAP-expert-v-operands（补「裸标量 int」触发形态） |
| 双侧带状**算术惩罚掩码链**（`vsub/vmax/vmin/vmax` + `vmul(pen,−1e38)` + `vadd`）在 Ascend910B3 / CANN 9.0.0 上 **7/7 配置 bit-exact（有效掩码 + S 哨兵逐位）且零标量化 warning**——PL-1.11 主选前提跨 SoC 存活 | D | `history_version/design_probe_band_mask.py` 输出（MAIN 7/7 bit-exact, warning=False；ALT vcmp 7/7, warning=True） | `python history_version/design_probe_band_mask.py` | 同上 | `pattern-library/attention.md` PL-1.11（910B3/CANN 9.0.0 重验注记） |
| PL-1.11「`0×NaN=NaN`」在滑动窗口尾带域的落点：`gemm2` K 维 fractal band 的 `P带(0)×V残留(NaN)` 须 `l1_b`(V) `zbuf` 零初始化根治——「V残留任意值×0=0」假设对 NaN 位不成立 | D | `DESIGN.md` §6.4/§0.6 E7（本次补入）；PL-1.11 NaN 免疫边界条 | `pattern-library/attention.md` PL-1.11（原文条） | 同上 | `pattern-library/attention.md` PL-1.11（尾带落点补充） |

### Transferable Lessons（可迁移教训）

- **结构红利数字必须「表达式 + 数值」并自校验**：执行块数、fabric GB 等派生量若以叙述式给出，极易内部矛盾（本任务 v0 的 3584 vs 0.46GB）。修订把每个数字改成可复算式，并加逐 bx 复算表——这是设计自检能机械复算的前提。
- **区分「执行块数」与「内段全有效块」**：带状裁剪的执行块数 = Σloop_count；内段（零掩码）块 = Σmax(loop_count−2,0)。两者不可混用，更不可拿内段块与全 causal 总数相减。
- **探针要能确定性复现被引证据**：编译期 warning 依赖未命中缓存；纯输出 kernel 在本设备 abort。探针须内置 `disable_cache()` 强制复编译、至少含一个输入张量，且覆盖**被采纳的形态**（主选链本身，而非仅备选链）。
- **修订不破坏已验证结论**：块数/fabric 属「结构红利量级」表述，订正后 FLOPs（51.6/68.8 GFLOP，−25%）、算法选型、等价论证、Expert 模式（P1）均不变——修订只改算术与证据表述，不回归已验证的算法层结论。

## Stage 2 (design review) Retrospective — re-review (revision_index=1) — 2026-10-16

> 对象：`_gqa_sw_fwd_kernel` 的 Stage 2 **修订后重新检视**流程自身（结论 = 通过；0 阻塞 + 1 建议）。设计层问题已随 `REVIEW.md` 流转给 Stage 1，不在本节重复。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | 上轮 v0 复盘已提出「机械复核须复跑设计期探针」，但 `tilelang-design-review` 的 Phase 1 第 7 步仍只规定「重跑 `verify_equiv.py` + `design_calc_check.py`」——本轮回检时 P1/P2 探针复跑仍靠 conductor 显式指定；skill 自身未把它列为强制步骤，照默认流程会漏检「探针不可运行/测的不是被采纳形态」 | `DESIGN.md` §2.2/§0.6 E-mask 引 P1/P2 结论；本轮实跑 `design_probe_devmode_mix.py`（A PASS/B BLOCKED/C PASS）、`design_probe_band_mask.py`（MAIN 7/7 零 warning） | 将「复跑 DESIGN.md 引用的全部设计期探针，结果与自述一致方通过」写入 review SKILL Phase 1 第 7 步（机械复核）作为强制项 | R |
| Dimension-coverage | 维度 8「复杂度算术独立复算」条目聚焦 §1.6.0 R3 表，**未显式覆盖 §11 性能预期表中同 formula 派生的 FLOPs 列**——本轮 smoke 行 FLOPs 与 manifest 口径差 ~2×，仅在主动扩算全部 §11 行时才发现 | `DESIGN.md` §11 smoke 行 `0.27 GFLOP` vs 独立复算 0.54 GFLOP（`4·B·H·total_attended·D`，`total_attended=Σ_q(q+1)=131328`） | 在维度 8「复杂度算术独立复算」条目把覆盖范围从「§1.6.0 R3 表」扩到「全文所有同源派生量（§11/§5.4 等）」，并核对跨表口径一致 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 修正确认：v0 的 2 阻塞在 v1 逐项闭合——R3 执行块数经逐 bx 复算 = 7680（−16.7%）且 fabric 逐项对账自洽；P2 探针修复后可运行且覆盖主选形态（MAIN 7/7 bit-exact 零 warning） | D | `REVIEW.md`「上轮问题闭合核对」表 + 机械复核段；`verify_equiv.py` 11/11；P1/P2 复跑输出 | `python history_version/design_probe_band_mask.py`；`python history_version/design_probe_devmode_mix.py`；逐 bx `loop_count` 复算 | tilelang `0.1.2+cf6f999389f4f3481da5778aa06d73478684ce2f` / CANN 9.0.0 / Ascend910B3 | `pattern-library/attention.md`（滑动窗口块数口径重验注记） |
| §11 类「best_effort 性能预期表」的 FLOPs 列也会滑出 manifest 口径（本任务 smoke 行 0.27 vs 0.54 GFLOP）——跨表同 formula 派生量须一并复算，否则检视易漏 | P | `DESIGN.md` §11 smoke 行；独立复算 `4·B·H·Σ(q+1)·D`=0.538 GFLOP（其余 4 workload 行 6.45/51.6 均正确） | 按 manifest roofline 公式逐 workload 复算 §11 FLOPs 列 | 同上 | `tilelang-design-review` SKILL 维度 8（复杂度复算范围）；`tilelang-op-design` §11 模板 |

### Transferable Lessons（可迁移教训）

- **「修订后重检」须做一次全维度独立重算，而非只核对上轮问题是否改对**：本轮除确认 2 阻塞闭合外，对全部 9 维度重新独立复算（源码逐句核对、R3 逐 bx、fabric 逐项、FLOPs、探针实跑），才捕获到 §11 smoke 的残留数字滑出——只核对修订清单会漏检。
- **派生量复算要覆盖「所有出现该量的表」**：同一 FLOPs/Bytes formula 常在 §1.6.0 与 §11（及 §5.4）重复出现；只复算主表会漏掉别处的同源笔误（本任务 smoke 行）。
- **负向断言的证据要打开到条款级亲验**：本轮逐一亲验 `T.set_flag.md §1`/`T.sync_block_set.md §1`（核内范围）、`T.reduce.md §2.3`（clear=False 静默错误）、`T.store_fixpipe.md §2.1`（src L0C/dst GM）、`T.transpose.md`+traps-compiler 毒化条——断言均成立。
- **证据产物修复后要实跑确认覆盖了被采纳形态**：上轮 P2 只含被否决链；本轮实跑确认修复版 MAIN 链 7/7 bit-exact 且零 warning，方判证据链闭合。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-16

> 对象：`_gqa_sw_fwd_kernel` 的 Stage 3 实现（mode=first_impl, attempt_index=1）。终态 `[PRECISION_PASS]`：L0 4 例 + 契约、L1 8 例、L2 12 例、Boundary 8 例全通过（`--level all` 两次 exit 0）。本节覆盖 attempt 链中的全部中间失败与实测发现。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Toolchain-probe | `tilelang-op-develop` Phase 1 要求读 pattern-library 的 traps，但**未要求「首次使用 `T.sync_block_set`/`T.sync_block_wait` 前先在目标容器做最小编译探针」**——本容器（CANN 9.0.0 + 该 tilelang build）的 `sync_block_set` codegen 属性名与安装版 BishengIR 解析器不一致，任何含该 API 的 kernel 直接编译失败；MHA 母本此类结构在另一容器（CANN 8.5）验证过，照抄即踩 | `repro/TRAP-bishengir-sync-instr-mode-attr.py`（同一 .npuir 未改写即被 bishengir 拒；改写一个属性名即通过）；`src/target/codegen_npuir.cc` 与 bishengir 诊断字符串 `'sync_instr_mode'` 不一致 | 在 develop SKILL Phase 1 补一条：「跨容器复用某一 Expert 结构前，先对其中**每个跨引擎同步/搬运原语**做最小编译探针」；并把新 trap 登记进 pattern-library/traps-compiler.md | R |
| Template-gap | `templates/op_template.py` 与 SKILL Phase 2 未提示 **TVMScript「trace-time `if` 会 scope 变量」** 的坑：把 trace-time 选择写成 `if py_bool: v = ...` / `elif` 链，后续使用处报 `Undefined variable`；母本用三元表达式规避但未在 skill 里点明 | 首次 L0 编译报 `Undefined variable: k_end`（本文件 `_gqa_sw_fwd_kernel.py` 早期版本）；改三元后通过 | 在 develop SKILL Phase 2 加一条已知 parser 约束（与既有「D4 vbrc 标量须 let-bound」并列）：trace-time 分支选择一律用三元/字典分派，不得用赋值型 `if` | P |
| Compiler-flakiness | `tilelang-op-develop` 未提示 **bishengir-compile 在本容器会非确定性 SIGSEGV**（同 shape 单独编译通过、放在编译序列中崩溃），且 `@tilelang.jit` 无内置重试——照默认写法一次崩溃即 RUNTIME_FAIL 假阴性 | `--level all` 首次 L1/L2 均在特定编译序列崩溃；加 6 次重试后两次全绿；`dbg5/dbg8` 对照（单独编译 vs 序列编译） | 在 develop SKILL Phase 4 或 error-fixer 里加：「本容器 bishengir-compile 有非确定性 SIGSEGV；编译失败先区分 `SIGSEGV`（重试）与 lowering 诊断（不重试），并给出重试包装片段」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`T.sync_block_set` 在本容器不可编译**：codegen 发 `syn_instr_mode`，CANN 9.0.0 bishengir 解析器要 `sync_instr_mode`（其诊断字符串即 `'sync_instr_mode'`），原样报 `Dialect \`' not found for custom op 'syn_instr_mode'`；`sync_block_wait` 无此属性不受影响 | D | `repro/TRAP-bishengir-sync-instr-mode-attr.py`（未改写 IR 被拒 + 单属性名改写后编译通过）；`_gqa_sw_fwd_kernel.py` 顶部 `_install_sync_block_attr_shim()`（L0/L1 通过） | `python repro/TRAP-bishengir-sync-instr-mode-attr.py` | tilelang `0.1.2+cf6f999…` / CANN 9.0.0 / Ascend910B3 | `pattern-library/traps-compiler.md`（新增 TRAP-bishengir-sync-instr-mode-attr）；`tilelang-npuir-overview`（版本戳注记） |
| **`T.floordiv` 负分子被 lower 成截断 `arith.divsi`**：形如 `n >= T.floordiv(X,bn) - k_start + 1`（X 符号未知）在生成 IR 里成 `divsi(X,bn)`，`divsi(-255,256)=0≠floor=-1`——q 前导块的分支静默不进入，causal 掩码整块丢失（out ~92% 错、lse 偏 ~9）。修法 = 用代数等价的**无除法直接整数比较**（`n > s_lo+row0-(bn-1)` / `n < s_lo+row0+rm-1-wl`） | D | `repro/TRAP-npuir-negative-floordiv-divsi.py`（buggy 形出现 `arith.divsi`、direct 形无除法）；`_gqa_sw_fwd_kernel.py` 的 `apply_upper/left` 直接比较（L0-1 由 46% 错→全过） | `python repro/TRAP-npuir-negative-floordiv-divsi.py` | tilelang `0.1.2+cf6f999…` / CANN 9.0.0 / Ascend910B3 | `pattern-library/traps-compiler.md`（新增 TRAP-npuir-negative-floordiv-divsi）；`docs/Tilelang.language/数学操作/T.ceildiv.md`（增「负分子」告警） |
| **部分宽度 `T.copy` 后做全宽 `T.reduce` 会吃进 stale 尾列**：ragged KV 块只写 `[0,tnc)`（tnc<bn_eff），全宽行 max 读到未初始化尾列 → NaN（首用）或上块残留（静默错）；且 **`-1e38` 算术掩码无法清 NaN**（`NaN+finite=NaN`），故必须 (a) 全宽列 OOB 掩码 + (b) N 载体每核一次零初始化 | D | `repro/TRAP-npuir-stale-ub-tail-columns-nan.py`（A NaN尾无掩码→NaN；B NaN尾+算术掩码→仍 NaN；C 零初始化+全宽掩码→正确）；`_gqa_sw_fwd_kernel.py` 的 `has_tail` 分支（L2/Boundary S=100/200 由 NaN→全过且确定性） | `python repro/TRAP-npuir-stale-ub-tail-columns-nan.py` | tilelang `0.1.2+cf6f999…` / CANN 9.0.0 / Ascend910B3 | `pattern-library/traps-runtime.md` / `attention.md` PL-1.11（尾列落点补充）；`tilelang-vector-skill`（部分宽度 copy 纪律） |
| **`vsub`/`vmax`/`vmin` 的标量操作数不能是裸 `tir.Cast`**：`T.cast(x, dtype)` 作为标量 operand 触发 `AttributeError: <class 'tvm.tir.expr.Cast'> has no attribute buffer`（`_to_region` 走 `data.buffer`）；须写成「kernel 派生 float 表达式」`kid*0.0 + (int_expr)`（顶层节点为 Add 即可） | D | `_gqa_sw_fwd_kernel.py` has_band OOB 首次 trace 报错；改 `tnr = kid*0.0 + (tn_real-1)` 后通过 | `repro/TRAP-npuir-negative-floordiv-divsi.py`（同类标量形态） | 同上 | `pattern-library/traps-compiler.md` TRAP-expert-v-operands（补「裸 tir.Cast」形态） |
| **bishengir-compile 非确定性 SIGSEGV**：同一 shape 单独编译通过，处于编译序列中则 `died with <Signals.SIGSEGV: 11>`（栈帧重复暗示栈溢出/递归）；`ulimit -s unlimited` 不解决；`@tilelang.jit` 无重试 | D | `dbg5`（fp16-long→bf16-long 序列单独编译 OK）vs `run_L1`（同序列崩溃）；加 6 次重试后 L1/L2/Boundary 全绿 | `python _gqa_sw_fwd_kernel.py --level all` | 同上 | `tilelang-error-fixer`（新增「编译期 SIGSEGV 重试」条目） |
| **trace-time 选择必须用三元/字典分派**：TVMScript 把 `if <python bool>:` 当 TIR If，赋值变量被 scope 在分支内，后续 `Undefined variable` | P | 首次 L0 `Undefined variable: k_end`（早期 `if causal_t: k_end=...` / `elif`）；改嵌套三元后通过；`repro` 脚本同坑（改为三元） | 同 trace 报错信息 | 同上 | `tilelang-op-develop` SKILL Phase 2 / op_template.py |

### Transferable Lessons（可迁移教训）

- **跨容器复用 Expert 结构前，先对「跨引擎同步原语」做最小编译探针**：母本（MHA）在 CANN 8.5/910B2C 验证过的两相位结构，在 CANN 9.0.0/910B3 上**第一步 `T.sync_block_set` 就编译失败**（codegen 属性名与解析器不一致）。同型的还有 `T.floordiv` 负分子 lower 差异——**工具链版本戳不同，先探针后照抄**。
- **`T.floordiv`/`T.ceildiv` 一旦分子可能为负，就改用代数等价的直接整数比较**：生成 IR 可能是截断 `arith.divsi`，且只在运行时才显形（编译通过、结果错）。块分类谓词（"块是否全有效"）写成 `n > s_lo+row0-(bn-1)` 这类无除法比较，既避坑又免一次除法；并在 dump 的 `.npuir` 里确认没有意外 `divsi`。
- **部分宽度 `T.copy` 进更宽 UB buffer 后，任何全宽归约/逐元素链都必须显式覆盖未写列**：要么全宽掩码 + 载体预零初始化（保证有限），要么别用全宽归约。**`-1e38` 算术掩码对 NaN 无效**（`NaN+finite=NaN`）——"掩码能清一切"是错的，stale 列的 NaN 必须在源头消除（零初始化/zbuf）。
- **v-prefix 标量 operand 只接受「kernel 派生 float PrimExpr」**：裸 Python 常量、裸 `tir.Cast` 都会被 `_to_region` 拒；统一写 `kid*0.0 + (expr)` 的形态，一次到位。
- **编译成功 ≠ 编译稳定**：本容器 bishengir-compile 会非确定性 SIGSEGV（序列相关）；把「区分 SIGSEGV 与 lowering 诊断 + 有界重试」写进编译包装，可避免把工具链抖动误判为算子失败。
- **多形状回归必须包含「非整除尾块」用例**：manifest 全域整除（零尾块），只有 L2/Boundary 的 S=100/200 触发 stale 尾列 NaN——若只跑 manifest 会漏掉整类缺陷；L2/Boundary 的告警层不应被当作"可选"而长期忽略。

