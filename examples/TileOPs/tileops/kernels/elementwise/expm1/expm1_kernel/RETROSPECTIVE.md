# RETROSPECTIVE — expm1_fwd_kernel（迁移 harness）

## Stage 1 (design) Retrospective — first_design — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 任务预注入的 pattern-library 常数条目（`CONST-mte2-degradation` / `CONST-vector-launch-overhead` / `CONST-capacity-910B2C` / `PL-1.17-subfp32-fp32-transit`）经 `kb_stale_check.py` 全部标记 stale（当前工具链 `tilelang 0.1.2+90a93c89` vs 条目戳 0.1.2+ed787bb/3a214cde），设计期 roofline 只能作「待重验」量级估算 | `examples/expm1/expm1_fwd_kernel/DESIGN.md#1.6.0`（设计期估算段）；`python3 .agents/tools/kb_stale_check.py`（stale entries: 137） | 在 conductor 调度 prompt 的「知识预注入」段标注条目 stale 状态（当前仅在外部摘录里人工标注），或让 kb_search 输出 stale 标记 | R |
| Info-source | 算法候选库 `algorithm-candidates.md` 的 `ALG-elementwise` 只覆盖「CSE/倒数/Horner」类通用化简，未含 expm1/超越函数逼近候选族——设计者须自行从模型知识补充（Cephes/Taylor/libm 区间约减） | `examples/expm1/expm1_fwd_kernel/DESIGN.md#1.6.0`（调研范围 ①）；`.agents/skills/tilelang-op-design/references/algorithm-candidates.md` `ALG-elementwise` | 待 erf / expm1 两个自建逼近先例合入后，在 algorithm-candidates.md 增「超越函数逼近族」条目（Taylor/minimax/Cephes 有理/libm 约减 + 误差界） | C |
| Design-flow | 迁移任务的「源算子契约语义 ≠ 源实现数值输出」情形缺少显式检查项：源 `op_func = exp(x)-1` 是契约 `torch.expm1` 的**劣化直译**（小 x 相对误差 19%），golden 必须取契约语义而非源直译数值 | `examples/expm1/expm1_fwd_kernel/DESIGN.md#0.1`（⚠️ 源实现与契约的差距） | 在 tilelang-op-design skill 的 Phase M0 第一问（语义）增加检查项：「源实现是否与 ref_api 契约等价（数值直译 vs 特化实现）」，golden 以 ref_api 契约为准 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| expm1 小 \|x\| 灾难性抵消的设备实证 + 分段 Taylor 修复：源直译 `vexp(x)-1` 在 x=1e-7 相对误差 19.2%（绝对误差仍 <atol 1e-5，纯绝对门禁测不出）；分段 `\|x\|<0.5ln2` Taylor6（x·Σxⁿ/(n+1)!）+ 大 \|x\| `vexp-1` 后小 x rel ≤1.2e-7、fp16/bf16 bit-exact | D | `examples/expm1/expm1_fwd_kernel/DESIGN.md#0.6`/`#1.6.1`；`verify_equiv.py`（6/6 EQUIV_PASS）；`history_version/design_probe_expm1_ops.py` P0/P3/P4 | `python examples/expm1/expm1_fwd_kernel/history_version/design_probe_expm1_ops.py`；`python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py` | tilelang 0.1.2+90a93c89 / Ascend910_9362 / CANN（本会话实跑 2026-10-12） | pattern-library/elementwise.md |
| `T.vexp` fp32 设备语义为真 exp（rel 1.182e-7，近 0 与 ±10 全域一致）、`T.vsub(x,1)` 精确——补充 VP-2026-0141 设备元算精度矩阵的逐区间证据 | D | `examples/expm1/expm1_fwd_kernel/history_version/design_probe_expm1_ops.py`（P1/P2）；DESIGN.md#3.4 | `python examples/expm1/expm1_fwd_kernel/history_version/design_probe_expm1_ops.py` | tilelang 0.1.2+90a93c89 / Ascend910_9362 / 2026-10-12 | pattern-library/traps-runtime.md（与 VP-2026-0141 合并） |
| 「绝对门禁掩盖 expm1 相对精度缺陷」模式：带 atol 的 `torch.testing.assert_close` 下，`exp(x)-1` 全域通过门禁（worst ratio 0.012）却在小 x 违反 ref_api 语义；判据应含小 x 相对误差专项用例（L0-5 linspace(-1e-3,1e-3)） | P | `examples/expm1/expm1_fwd_kernel/DESIGN.md#1.6.1`（小 x 相对精度诊断表）/`#8.3`（L0-5）；`verify_equiv.py` small-x diagnostic | `python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py` | tilelang 0.1.2+90a93c89 / 2026-10-12 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- 迁移带 `expm1`/`log1p`/`log`/`sqrt` 类**条件性小参数特化**的 libm 函数时：源 GPU `op_func` 常是 `exp(x)-1` 式直译，绝对门禁（atol=1e-5）会掩盖其小 `|x|` 相对精度缺陷（19%）；golden 必须取 ref_api（`torch.expm1`）契约，且 L0 须含小 `|x|` 相对误差专项用例，不能只看 `assert_close` 是否通过。
- npuir 无原生 `vexpm1` 元算（`docs/Tilelang.language/数学操作/` 全目录 + `customize_npuir.py` registry 均无）；`expm1` 须自建分段链（小 x Taylor + 大 x `vexp-1`），阈值 `0.5·ln2` 使 `exp≥√2`、抵消放大因子 ≤3.414。同族 `erf`/`expm1` 的「无元算→自建逼近」可复用同一候选调研 + 设备探针 + `verify_equiv.py` 范式。
- 分段（含 `vcmp`+`vselect`）的向量链中，被 `vcmp` 用完的中间 buffer（如 `xa=|x|`）可在条件判定后**原地复用**为分支结果，无需独立结果 buffer；`vselect` 的两分支均计算（SIMD 屏蔽），op 数须按「两分支之和」估，不能按「只走一支」估。
- `vexp` 在目标设备溢出（如 fp32 `x>88.7`）返回 `inf`，与 `torch.expm1` 的 `inf` 一致，无需额外饱和/裁剪分支——大 `|x|` 边界可直接依赖 `vexp` 的 inf/0 语义。

## Stage 2 (review) Retrospective — design_review — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | skill Phase 1 第 7 步的机械复核只覆盖 `verify_equiv.py`（CPU）+ `design_calc_check.py`，**不覆盖设备探针**；本次 §3.4 的关键 API 语义/精度主张（P0 抵消 19.2%、P1 vexp rel 1.182e-7、P2 vsub 精确、P3/P4 端到端）来自 `history_version/design_probe_expm1_ops.py`，review 上下文无 NPU 无法独立重跑，只能采信设计自述——缺少「探针脚本可重跑性/结果文件留痕」的机械核验位 | `examples/expm1/expm1_fwd_kernel/DESIGN.md#3.4`；`history_version/design_probe_expm1_ops.py`；skill Phase 1 第 7 步 | 在 tilelang-design-review skill Phase 1 第 7 步增加「设备探针结果留痕核对」：要求 §3.4 引用的探针脚本落一份结果输出文件（如 `history_version/probe_result.json`），review 核对其存在性与数值一致性（无法重跑时至少核对留痕） | R |
| Review-flow | 维度 8 现有检查项要求「等价性论证正确性」，但对**分段/逼近类**算子缺少「误差界独立复算」的显式子项（Taylor 截断尾界、阈值处两分支连续性、大 |x| 抵消放大因子）；本次靠本 Agent 手工推演 `|R|≤|x|⁷/5040·1/(1-|x|/8)` 才完成复核，检查项未点名 | `examples/expm1/expm1_fwd_kernel/DESIGN.md#0.6` R2 / `#1.6.1`；本 Agent 独立复算（得 1.245e-7 与 verify_equiv #6 实测吻合） | 在 tilelang-design-review skill 维度 8「等价性论证正确性」行补子项：分段/逼近类须独立复算各分支误差上界 + 阈值连续性，且与机器验证的 intrinsic 项交叉核对 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 分段/逼近类（expm1）设计检视的误差界独立复算范式：Taylor 截断尾界 `|R|≤|x|⁷/5040·1/(1-|x|/8)`（T 处 1.245e-7）与大 x 抵消放大因子 `√2/(√2-1)=3.414`，两分支相对误差各 ≤4.86e-7 / ≤4.0e-7 均 ≪ rtol 1e-5 | P | `examples/expm1/expm1_fwd_kernel/DESIGN.md#0.6`/`#1.6.1`；`verify_equiv.py` #6 max_abs=1.245e-07 | `python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 / Ascend910_9362 / 2026-10-12 | tilelang-design-review skill（维度 8） |

### Transferable Lessons（可迁移教训）

- 检视「分段逼近 + 阈值选路」的迁移设计时，**必须独立复算误差界**（小 x 截断尾界、大 x 抵消放大因子、阈值处两分支连续性），不能只读设计的自述界；并把它与 `verify_equiv.py` 的 intrinsic 项交叉核对——本次独立复算的 1.245e-7 与机器验证 #6 实测逐值吻合，是验证论证正确的有效手段。
- 设计的「相对误差 ≲X、≥N× 余量」类小结陈述要**独立核对算术**：本次 §0.6 R2 分别列出大 x 减 1 舍入 ≤2.0e-7 与 vexp 放大误差 ≤4.0e-7（保守和 ≈6.0e-7），小结却写「综合 ≲5e-7」——量级陈述略保守偏差，不影响结论（仍 ≪ 门禁），但揭示此类小结需复算。
- 迁移检视当源实现是契约的**劣化直译**（如 `exp(x)-1` 之于 `torch.expm1`）时，golden 取契约语义（`torch.expm1`）而非源直译数值是正确判据；检视须确认 golden 独立于 NPU 重设计算法（不复用其系数/结构），否则精度对比沦为自证。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 迁移参照实现 `erf_fwd_kernel.py`（及可能的 op_template 系）的 `_max_diff` 直接 `(y-g).abs().max()`，golden 含 `inf`/`nan` 时 `inf-inf=nan` 使 max_diff 打印为 `nan`（`assert_close(equal_nan=True)` 仍能判过，故不阻塞，但诊断数值失真）；本次 expm1 因 L0 强制含 inf/NaN 角点，须自行加 `isfinite(golden)` 掩码 | `examples/erf/erf_fwd_kernel/erf_fwd_kernel.py#_max_diff`（无 finite 掩码）；`examples/expm1/expm1_fwd_kernel/expm1_fwd_kernel.py#_max_diff`（加掩码） | 在 op_template / 迁移参照实现的 `_max_diff`（及相对误差 helper）统一加 `isfinite(golden)`（相对误差再加 `g != 0`）掩码，避免 inf/nan 用例诊断数值为 nan | P |
| Develop-flow | DESIGN.md §8.3 措辞「L0-1..L0-7 全过即进入 Stage 3」把 Stage 3 的**退出**门禁写成**进入**条件（时序倒置），实现者易误读为「设计期已跑过 L0」；实际 L0 是 Stage 3 内必过的退出判据，需在 Stage 3 重新执行 | `examples/expm1/expm1_fwd_kernel/DESIGN.md#8.3`（「门槛标准：L0-1..L0-7 全过即进入 Stage 3」） | 在 tilelang-op-design skill 的 §8.3 模板把该句改为「L0 为 Stage 3 内的门槛用例集，须在 Stage 3 落地并通过」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| expm1 分段链（16-op fp32：vabs+vcmp+Horner10+vmul+vexp+vsub+vselect）在 40 核 persistent 网格跨步 + 动态尾块形态下的端到端设备实测：fp32 全域 max_rel=5.13e-7（randn 16m / linspace ±9 / 阈值带 / 268m manifest，均 ≪rtol 1e-5）；小 x 敏感段 (linspace(-1e-3,1e-3)) max_rel=1.19e-7、角点 x=1e-7 rel=7.1e-8；fp16/bf16 绝大多数用例 bit-exact（max_diff=0）；`vexp` 溢出 88.8→inf、-inf→-1、nan→nan 均逐值一致 | D | `examples/expm1/expm1_fwd_kernel/expm1_fwd_kernel.py`；本会话 `--level all` 实测输出 | `cd examples/expm1/expm1_fwd_kernel && python expm1_fwd_kernel.py --level all` | tilelang 0.1.2+90a93c89 / Ascend910_9362 / 2026-10-12 | pattern-library/elementwise.md（与 VP-2026-0141 / erf 分段逼近先例合并） |
| 无原生 `vexpm1` 时「整 buffer v-op 16-op 链 + 循环外一次性 `T.fill` 阈值 buffer + 仅 `T.copy` 用动态 tail」的 Developer 模式形态可直接落地：40 核 `T.serial(ceildiv(n_num,40))` 网格跨步、编译期 `if need_cast` 分支、`xa` 原地复用、`xc` 链尾原地承载结果，全程无编译错误、无需手动 sync（Developer 自动同步） | P | `examples/expm1/expm1_fwd_kernel/expm1_fwd_kernel.py`（kernel 工厂）；DESIGN.md#3.3/#5.5 | `cd examples/expm1/expm1_fwd_kernel && python expm1_fwd_kernel.py --level L0` | tilelang 0.1.2+90a93c89 / Ascend910_9362 / 2026-10-12 | pattern-library/elementwise.md / traps-compiler.md |
| fp16/bf16「fp32 域算 + 单次 rint 舍回」并非恒 bit-exact：大 N（16m / 67m）下 fp16 出现 1 ULP 舍入差（max_diff=4.77e-7，rel=6.91e-4），位于 fp16 半位取舍边界——距 rtol 1e-3 仅 ~1.45× 余量；为源 fp32 中转语义的固有结果，非实现缺陷 | D | `examples/expm1/expm1_fwd_kernel/expm1_fwd_kernel.py`；本会话 L1-16m fp16 / Boundary-manifest fp16 实测行 | `cd examples/expm1/expm1_fwd_kernel && python expm1_fwd_kernel.py --level all`（grep L1-16m dtype=float16） | tilelang 0.1.2+90a93c89 / Ascend910_9362 / 2026-10-12 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- 逐元素 float 迁移中，若 golden 的取值域含 `±inf`/`nan`（exp/expm1/log 类），边界用例的 `_max_diff` 必须用 `isfinite(golden)` 掩码、相对误差 helper 必须再用 `golden != 0` 掩码，否则 `inf-inf=nan` 会让诊断数值变成 nan（`assert_close(equal_nan=True)` 仍能判过，问题被掩盖）。
- 逐元素分段/逼近链的常量 buffer（阈值、系数）应在 `T.serial` tile 循环**外**一次性 `T.fill`；所有 v-prefix 元算只作用于**整 UB buffer**，动态 tail 切片只允许出现在 `T.copy` 中——该形态从 `erf`/`bitwise_not` 迁移到 16-op 链（含 `vcmp`+`vselect` 分段）零编译问题，可放心复用。
- fp16/bf16 走「升 fp32 计算 → 单次 `vcast(rint)` 舍回」时不要假设对 golden bit-exact：在半位取舍边界会出现 1 ULP 差（fp16 相对 ~4.9e-4），须保留 rtol 余量（1e-3 时余量约 1.45×）；这是源 fp32 中转语义的固有行为，不是缺陷。
- 分段链中被 `vcmp` 用完的中间 buffer（如 `|x|`）可在判定后原地复用为分支结果，`xc` 可在链尾原地承载 `vselect` 输出；这种复用不改变数值，但要求严格的顺序（复用写必须在所有读取完成之后），否则产生结果错误。
