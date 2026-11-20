# RETROSPECTIVE — erf_fwd_kernel

## Stage 1 (design) Retrospective — first_design — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 调度 prompt 的「知识预注入」把 `PL-1.17-subfp32-fp32-transit` 等四条列为 stale/待重验，但 `pattern-library/elementwise.md` 与 `INDEX.md` 中这些条目 front-matter 仍写 `status: verified`——两处状态口径不一致，设计期需自行判断能否引用 | `examples/erf/erf_fwd_kernel/DESIGN.md` §1.6.0 调研范围③；`.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md` PL-1.17（status: verified）vs 调度 prompt 重验清单 | 明确 stale 条目的单一事实源（INDEX/主题文件的 status 与调度 prompt 预注入应一致，或标注「预注入口径覆盖文件 status」） | R |
| Constraint-doc | `T.verf.md` §2.2.1 只给 fp16/fp32，未说明 bf16 的推荐替代路径；bf16 中转结论需自行由 `T.vcast.md` dtype 矩阵拼接 | `docs/Tilelang.language/数学操作/T.verf.md` §2.2.1 | 在 bf16 不支持的 v-prefix 数学 API 文档中加注「bf16 请走 vcast 升 fp32 计算 + 舍回」的一句指引 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| erf unary 迁移：`T.verf` fp32 直算 + fp16/bf16 fp32 中转链（bf16 为 API 强制中转，fp16/bf16 与 golden bit-exact） | C | `examples/erf/erf_fwd_kernel/DESIGN.md` §0.6 R3 / §1.6.1 | `python3 examples/erf/erf_fwd_kernel/verify_equiv.py` | torch 2.7.1+cpu（设计期 CPU 验证）；NPU 部分 910_9362 实查 | pattern-library/elementwise.md（erf 族首例，供同类超越函数迁移参考） |
| 设计期 D-2 估算：超越函数链（k=5 v-op/tile）发射项主导，可能显著高于流量项——与同族 integer（k=1）逐元素画像不同 | D | `examples/erf/erf_fwd_kernel/DESIGN.md` §1.6.0 估算表（smoke-1m fp32 flow 6.2µs vs em 19.2µs；16m fp16 em 512µs） | none（估算，非实测） | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | pattern-library/constants.md（待 Stage 4 实测校准后决定是否合入） |

### Transferable Lessons（可迁移教训）

- **bf16 的 v-prefix 数学 API 缺口（vadd/vmul/vdiv/verf 均不含 bf16）决定 bf16 逐元素算子必须走 fp32 中转链**：`vcast(bf16→f32, rint)` → fp32 域 v-op 链 → `vcast(f32→bf16, rint)`。`T.vcast.md` §2.2.1 是 dtype 矩阵的权威来源（bf16↔f32 仅 `rint`）。
- **fp16 逐元素算子即使 v-prefix API 原生支持 fp16，也宜保留 fp32 中转以对齐 torch CPU golden**：torch 对 fp16 经 fp32 opmath 计算，原生 fp16 域逐步舍入会与 golden 差 2–3 ulp（`TRAP-fp16-opmath-golden` 机理）；设计期用 `verify_equiv.py`（fp64 参照 + 角点）可机器拦截该分歧，避免 Stage 3 才暴露。
- **v-prefix op 必须作用于整 UB buffer**：动态 slice（如 `x_ub[0:tail]`）只可包在 `T.copy` 里，直接传给 `T.verf`/`T.vcast` 会触发编译/运行错误（前一任务 bitwise_not R-6 实证）。
- **编译期 `if` 内禁止 `T.alloc_ub`**（`TRAP-conditional-alloc`）：dtype 分派分支的 buffer 一律在 `with T.Kernel` 顶层无条件分配，分支内只放 v-op 调用，否则 trace 报 `Undefined variable`。
- **分核策略对纯 Vector 一维算子**：物理核 = `NPUUtils.get().get_aivector_core_num()` 实查（910_9362 = 20 aicore ×2 = 40）；逻辑核数为 2 的幂时与含非 2 因子的 40 无法整除对齐，按标准取「核内 grid-stride serial persistent」并显式记录残留不均。

## Stage 2 (review) Retrospective — first_review — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 在**一维逐元素类**设计上同时 skip `ub_budget` 与 `core_split`（无 per-level buffer 表、解析 dims 为空），使维度 2/3 的全部数值复核退化为纯人工；机械拦截对该算子族近乎失效 | `examples/erf/erf_fwd_kernel/REVIEW.md` 机械复核段（`ub_budget: skip` / `core_split: skip`）；`.agents/tools/design_calc_check.py` 输出 | 为一维逐元素族扩展解析（UB 表按 buffer 名/字节列即可提取；core_split 从 `ceildiv(...)` 文本提取）或在 skill 维度表明示该族的**强制人工核对清单** | R |
| Info-source | 设计自述「`PL-1.17` / `PL-1.6` 等四条按调度 prompt 标为 stale」与 pattern-library 主题文件 front-matter（`status: verified`）不一致，检视无法从单一事实源判定这些条目是否可作决策前提，只能采信设计的保守处置（不以其为决策依据） | `examples/erf/erf_fwd_kernel/DESIGN.md` §1.6.0 调研范围③；`.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md` PL-1.6/PL-1.17（均 `status: verified`, `repro: repro-missing`） | 统一 stale 事实源（文件 status 或预注入清单择一为准），供 Stage 2 判「负向论断是否与实测矛盾」时有确定依据 | R |
| Constraint-doc | 维度 2 的 UB 口径判定标准（skill 只写「总和不超过 192KB」）落后于 `TRAP-UB-multibuffer-inflation` 已更新的「按结构类平台值（transit≈20 / fp32≈26 B/elem）」口径，检视时需自行补做第二套口径复算 | `.agents/skills/tilelang-design-review/SKILL.md` 维度 2；`.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md` L98 | 在维度 2 通过标准中并列平台 per-elem 口径（或注明「resident×系数口径已废弃」） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 等价性机械复核的 **CPU 代理盲区**：`verify_equiv.py` 以 CPU fp64/fp32 参照验证的是「dtype 域等价」，无法覆盖设备端 v-prefix op（`T.verf`）的实现精度；维度 8 应增设判定项「设备 op 精度须由 Stage 3 设备 L0 兜底，不得据 CPU 代理宣告设备 bit-exact」 | P | `examples/erf/erf_fwd_kernel/verify_equiv.py` L87–93（candidate = `torch.erf(x.to(float64))`，纯 CPU）；`examples/erf/erf_fwd_kernel/DESIGN.md` §1.6.1 表 / §3.4（「NPU 设备实测留 Stage 3」） | `python3 examples/erf/erf_fwd_kernel/verify_equiv.py` | torch 2.7.1+cpu；NPU 侧未实测（标注设计期） | tilelang-design-review skill（维度 8 等价性论证正确性行） |
| 维度 8 复杂度表复算应增列「中间缓冲峰值」单元与下游 §4.5 的**交叉一致性**检查：本次发现 R3 表 fp32 峰值（24KB）与 R4/§4.5（16KB）不一致（疑似错抄 fp16 行），非阻塞但表内自洽是复算可信前提 | P | `examples/erf/erf_fwd_kernel/DESIGN.md` §1.6.0 R3 表「候选 1 fp32 … 24KB@bs2048」vs 同节 R4 表「fp32: x+y=16KB」/ §4.5 | none（人工比对） | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 | tilelang-design-review skill（维度 8 复杂度复算行） |

### Transferable Lessons（可迁移教训）

- **`design_calc_check.py` 对一维逐元素算子会同时 skip `ub_budget` 与 `core_split`**：检视该族时必须人工复算——UB 按平台 per-elem 口径（fp16/bf16 中转 transit≈20 B/elem、fp32≈26 B/elem，见 `TRAP-UB-multibuffer-inflation`）与「resident×系数」两套口径各算一遍；core_split 需手工核对 `ceildiv(N, block_size)`、物理核实查值、grid-stride 残留不均百分比。
- **区分「CPU 参照域验证的 dtype 域等价」与「设备 v-prefix op 实现精度」**：`verify_equiv.py` 全 EQUIV_PASS 只证明算法域等价（fp32 域算 erf + 单次舍回 ≡ torch golden），不证明 NPU `T.verf` 的实测精度；该结论须交 Stage 3 设备 L0 门禁（erf 族 fp16 容差 1e-3 ≈ 1 个 fp16 ulp，余量狭窄，需重点观测）。
- **负向断言复核须落到具体参考表命中行**：本算子命中 `ALG-elementwise`（CSE/除法乘倒数/cast 链合并 + 逐元素天然单遍），逐项对照源码证据（单超越函数无 CSE、无用户级除法、cast 链即中转链）后，确认「无化简/无在线变体」成立。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-12

**结果**：`[DESIGN_ERROR]`。产物 `erf_fwd_kernel.py` 已按 DESIGN.md §3.3 逐句落地（一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` grid-stride + 整 buffer `T.verf`/`T.vcast`，R-1/R-2/R-3 三项重设计端到端编译并运行成功），**但首个 L0 用例即灾难性精度失败**：fp32 `randn([1048576])` `max_diff=1528.097`（59.9% 元素超差，门禁 atol=1e-5）。

**根因（设计层，实现层不可修复）**：非 A5 设备上 npuir 的 `T.verf` **不是真正的误差函数**，而是 7 阶 Maclaurin 截断多项式
`erf(x) ≈ (2/√π)(x − x³/3 + x⁵/10 − x⁷/42)`。设备实测：`|x|≤0.5` 误差 ~9e-6（小值域可用）；`x=1` 误差 4.3e-3；`x=2` 误差 1.56；`x=3` 误差 37.9；`x=8` 输出 52828.5（应为 1.0，与截断多项式逐值吻合到 1e-3 相对误差）。`randn` 输入 `|x|` 典型达 ~5.5，故 fp32（atol 1e-5）、fp16（1e-3）、bf16（1.6e-2）三档全部超差，且 DESIGN §8.3 自定的 L0-4 角点用例 `[0,3,-3,100]` 亦必失败（erf(3) 被算为 37.9）。

**代码级证据**：`src/target/codegen_npuir_api.cc:2419–2524`、`src/target/codegen_npuir_dev.cc:3227–3272`、`src/target/codegen_npuir_api_a5.cc` 的 `VerfCodegen` 均按上述截断多项式展开（comment 明写 "erf(x) ≈ (2/√π) * (x - x³/3 + x⁵/10 - x⁷/42)"，body 用 `arith::DivFOp` + `hivm::VMulOp/VAddOp` 展开）；仅 `src/target/codegen_npuir_dev_a5.cc:3995–4003` 用 `mlir::math::ErfOp`（真 erf），但该路径面向 **A5 代设备**，本设备 `Ascend910_9362` 不适用。`docs/Tilelang.language/数学操作/T.verf.md` §3 只记「转译为 Constant/DivF/VMul/VAdd」，未披露是截断多项式，亦未标注精度适用范围。

**为何是设计层而非实现层**：DESIGN §1.6.0 把候选 1（原生 `T.verf` + fp32 中转）定为唯一采纳项，其等价性依据 `verify_equiv.py` 是 **CPU fp64/fp32 参照域**（候选式 `erf_fp64(x)`），只证明「fp32 域算 erf + 单次舍回」的算法域等价，**从未测过 NPU `T.verf` 的设备实现**；DESIGN §3.4/R-5 与 REVIEW 维度 8 备注② 均已显式把该验证留待 Stage 3。Stage 3 设备实测证伪了设计对该 API 的核心假设。实现层**无**可修复手段：① 设备上无第二个真 erf API（`T.erf` 标量循环在本 NPU 上编译期段错误，实测 EXIT=139，且设计已判 NPU 无标量融合路径）；② 自建 erf（如 `T.vexp` 是准确的，rel err ~1.2e-7，可搭有理/多项式逼近）属**新算法**，DESIGN §1.6.0 候选 5 明确以「非精确等价」否决，且需重新做精度域设计与等价性论证——超出「不自行改设计」的边界。

**建议设计修订方向（供 Stage 1 参考）**：取消对 `T.verf` 的依赖，改用 `T.vexp` 系原生向量 op 自建满足 fp32 atol/rtol=1e-5 的 erf 逼近（例如 Abramowitz-Stegun 7.1.26 类有理逼近，或 `erf(x)=sign(x)·(1−erfc(|x|))` 的 exp 形式），并按 `algorithm-candidates.md` 的精度域等价流程（fp64 参照 + 角点 + **设备实测**）重新验证；或经用户批准推动 `npuir_verf` 补齐（见能力缺口登记）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| API-doc | `docs/.../T.verf.md` 把 `T.verf` 描述为「误差函数」，仅列 dtype/shape 支持与 §3 转译目标（Constant/DivF/VMul/VAdd），**未披露 non-A5 codegen 实为 7 阶 Maclaurin 截断**，也无精度适用区间/误差界；设计期据文档字面即认定其为真 erf，导致这一定量级最大误差 5e4 的语义缺口直到 Stage 3 才暴露 | `docs/Tilelang.language/数学操作/T.verf.md` §1/§3；`src/target/codegen_npuir_api.cc:2419–2424`（comment 明写截断多项式）；实测 `repro/TRAP-verf-maclaurin-truncation.py` | 在 `T.verf.md` 增「实现说明」行：non-A5 路径为 `(2/√π)(x−x³/3+x⁵/10−x⁷/42)` 截断，标注精度有效区间（约 \|x\|≤0.5）与 A5 路径差异；或直接标注「不建议用于 \|x\|>1 的输入」 | D |
| Develop-flow | skill/op_template 未要求「采纳某 v-prefix API 前先做设备 micro-probe 量化其实现精度」——本任务把 API 精度实测推迟到整 kernel L0，直到 5 个用例的第 1 个才炸；若在 Phase 2 先跑 20 行 grid 探针即可一次定位根因，省去整 kernel 编译与误判 | `erf_fwd_kernel.py` L0 首次运行 stderr（`max_diff=1528`）；根因定位探针 `/tmp/opencode/erf_probe/p2.py`（后转正为 `repro/TRAP-verf-maclaurin-truncation.py`） | 在 `tilelang-op-develop` Phase 2 增加「新采纳的元算/超越函数 v-prefix API 先用小 grid 探针与 `torch` 参照比对（含 \|x\| 全域），再进整 kernel」；超越函数类算子尤其适用 | P |
| Info-source | 设计/检视阶段已两度预警「CPU 代理无法覆盖设备 op 精度、须 Stage 3 兜底」（REVIEW 维度 8 备注②、Stage 2 Transferable Lessons），但该预警未转化为 Stage 3 的**前置检查项**，Stage 3 仍按整 kernel 直接跑 L0——预警与执行之间缺一个可操作的门禁动作 | `REVIEW.md` §8 备注②；本文件 Stage 2 Transferable Lessons 第 2 条；Stage 3 L0 首次运行 | 在 `tilelang-op-develop` 的 Phase 1/2 之间加一条：凡设计文档显式标注「设备实测留 Stage 3」的 API，Stage 3 必须先补该实测并回填，再进入整 kernel 实现 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir `T.verf`（non-A5 codegen：api / dev / api_a5 三路径）**不是误差函数**，而是 7 阶 Maclaurin 截断 `(2/√π)(x−x³/3+x⁵/10−x⁷/42)`：\|x\|≤0.5 误差 ~9e-6、x=1 误差 4.3e-3、x=2 误差 1.56、x=8 输出 52828.5（真值 1.0）；`randn` 输入的 erf 逐元素算子全部超 fp32/fp16/bf16 门禁。仅 DEV_A5 路径用 `mlir::math::ErfOp`（真 erf） | D | `src/target/codegen_npuir_api.cc:2419–2524` / `codegen_npuir_dev.cc:3227–3272`（comment 明写截断式）；设备实测 `repro/TRAP-verf-maclaurin-truncation.py`（x=8 NPU=52828.5195 vs maclaurin7=52828.5203，rel<1e-3） | `examples/erf/erf_fwd_kernel/repro/TRAP-verf-maclaurin-truncation.py`（2026-10-12 实跑 PASS，3 条断言成立） | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | pattern-library/traps-runtime.md（新条 TRAP-verf-maclaurin-truncation） |
| `T.erf`（TIR 标量 erf，GPU 源码原形态）在 NPU `target="npuir"` 下**编译期段错误**（process killed, EXIT=139），无 Python 异常/诊断；即 NPU 侧的 `T.erf` 标量融合路径不可用（与 DESIGN §1.6.0 R4「NPU 无 T.Parallel 标量融合路径」定性一致，此为进程级实证） | D | `/tmp/opencode/erf_probe/p3a.py`（`y_ub[i] = T.erf(x_ub[i])` in `T.serial`）实跑：`start T.erf scalar-loop` → `compiling...` → Segmentation fault (EXIT=139)，无 stderr 诊断 | none（崩溃形态难以自包含断言；需人工在 NPU 复现 `p3a.py` 形态） | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | pattern-library/traps-compiler.md（`T.erf` 标量循环 segfault；标注 provenance=erf 任务 Stage 3 探针 p3a.py） |

### Transferable Lessons（可迁移教训）

- **采纳任何 npuir 超越函数 v-prefix API（verf / vtanh / vsin / vexp …）前，必须先在设备上用 `torch` 参照做全域（含大 `|x|`）micro-probe 量化其实现精度**；`docs/Tilelang.language/数学操作/*.md` 只写 dtype/shape 与「转译目标 op」，**不保证是真函数**——`T.verf` 即 non-A5 路径的 7 阶 Maclaurin 截断（仅 `|x|≲0.5` 可用），`vtanh` 同样是 `x−x³/3+2x⁵/15−17x⁷/315` 截断（源 `codegen_npuir_api.cc:2521+`），二者都不可用于 `randn` 级输入。设计期 `verify_equiv.py` 的 CPU fp64 参照**只验证算法域等价、不验证设备实现**——迁移 harness 的「bit-exact」结论必须来自设备实测。

## Stage 1 (design) Retrospective — revision_1（路径 B：实施期设计错误）— 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | skill 的 Phase 2 等价性机器验证（D-1）只要求「CPU fp64 参照」，不要求「采纳的新 API 必须在目标设备先做语义/精度 micro-probe」——本任务 v0 因只走 CPU 验证而把设备实现假设当真，Stage 3 才炸（损失整轮）。修订 1 自行把设备探针作为 §3.4 的强制列 | `examples/erf/erf_fwd_kernel/history_version/design_probe_erf_ops.py` / `design_probe_erf_ops2.py`；`.agents/skills/tilelang-op-design/SKILL.md` Phase 2 第 5 步（只提 CPU fp64） | 在 Phase 2/R4 增一条：凡**首次采纳**的 v-prefix 元算（尤其超越函数）须附目标设备 micro-probe（语义 + dtype 矩阵 + 全域精度），作为采纳前置；或把「设计期设备探针」升级为 R4 必做项 | R |
| Info-source | 调度预注入把 `PL-1.17-subfp32-fp32-transit` 标 stale，但主题文件 front-matter 仍 `status: verified`（与 v0 复盘同问题，未消解） | `pattern-library/elementwise.md` PL-1.17（status: verified）vs 调度 prompt 重验清单 | 同 v0 复盘 Skill Flow Issue（统一 stale 事实源） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **无原生 erf 元算下的自建逼近方案（A&S 7.1.26）**：设备实测可用元算 `vabs/vmul/vadd/vrec/vexp/vcmp/vselect` 组成 21-op fp32 向量链，全域（±8）`max_abs=3.725e-7`（fp32 atol 1e-5 的 27× 余量）；fp16 路径 `1.221e-4` / bf16 `0.0`（0 违例）；角点 `[0,3,-3,100]→[0,1,−1,1]` 与 golden 一致 | D | `examples/erf/erf_fwd_kernel/history_version/design_probe_erf_ops.py` + `design_probe_erf_ops2.py`（2026-10-12 实跑）；`DESIGN.md` §0.6 R2 / §1.6.1 | `python examples/erf/erf_fwd_kernel/history_version/design_probe_erf_ops.py`（打印 P1a–P2 各项）；`python .../design_probe_erf_ops2.py` | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | tilelang-op-design/references/algorithm-candidates.md（erf/超越函数逼近族首例）|
| **`T.vtanh` 在 Developer 模式是真 tanh**（`max_abs=1.192e-7` vs `torch.tanh`，±10）——证伪 Stage 3 复盘「vtanh 亦为截断多项式（`x−x³/3+…`）」的**未验证断言**；截断式仅在 Expert 模式（`docs/.../T.vtanh.md` §3）。教训：截断嫌疑须按模式分派并设备实测，不得跨模式外推 | D（证伪更正） | `history_version/design_probe_erf_ops.py` P1d 输出；`docs/Tilelang.language/数学操作/T.vtanh.md` §3 | `python examples/erf/erf_fwd_kernel/history_version/design_probe_erf_ops.py` | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | pattern-library/traps-runtime.md（`TRAP-verf-maclaurin-truncation` 条目需补「vtanh Developer 模式不适用」的反例边界） |
| **设备精度矩阵**：`T.vexp` rel 1.182e-7、`T.vrec` rel 1.189e-7、`T.vdiv` rel 1.189e-7、`T.vtanh` 1.192e-7（均 fp32） | D | 同上探针 P1a–P1d | 同探针 | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | pattern-library/constants.md（设备元算精度候选）|
| 设计期 D-2 估算修正：无原生 erf 元算使 op 数 3→21，发射项增至 flow 的 ~20×（smoke-1m fp32 estimate 134µs vs flow 6.2µs）——**「无原生元算」的固有性能代价须在设计期显式量化** | P | `DESIGN.md` §1.6.0「设计期估算」表 | none（估算） | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | pattern-library/constants.md（Stage 4 实测校准候选）|

### Transferable Lessons（可迁移教训）

- **设计期机器验证（D-1）的 CPU fp64 参照存在设备代理盲区**：它只能证「算法域等价」，不能证「设备元算实现精度」。凡采纳**新**的 npuir 元算（尤其超越函数）作为核心计算路径，必须在设计期先做**目标设备 micro-probe**（语义 + dtype 矩阵 + 全域精度，附脚本落 `history_version/design_probe_*.py`），再进公式优化 / tiling。erf v0 的 2485s 级返工即此盲区代价。
- **「无更优替代」的调研必须是「同语义算法族的完整枚举」**：v0 把输入算法（`T.verf`）当唯一候选；修订 1 从零枚举 9 个候选用 R3/R4 逐项裁决，才找到可行解（A&S 7.1.26）。调研四问的价值在输入算法被证伪时才显现——故必须前置执行，不可因算子「简单」而省略。
- **负向断言（含能力缺口）必须落证据 + 登记**：`T.verf` 截断、`T.erf` segfault、`T.vtanh` Developer 非截断，三类结论均由设备实测 + codegen 源码路径共同支撑，并登记 `CG-2026-0015`；未验证的截断嫌疑（vtanh）不得当结论复用。
- **逼近式替代元算时，「误差界 → 门禁余量」必须在设计期给出数字**：A&S 7.1.26 的 1.5e-7 理论界 + 3.7e-7 设备实测 vs atol 1e-5 = 27× 余量，并显式覆盖 `randn` 大 `|x|` 尾段与 `[0,3,-3,100]` 角点，才能宣称「容差内等价」。
- **op 数是逼近链的第一性能驱动**：NPU 逐元素发射 ~0.5µs/op（`CONST-vector-launch-overhead`），21-op 链的发射项远超流量项；无原生元算时，block_size（摊薄）与 Horner/倒数/CSE 等公式级降 op 是主要杠杆，且须诚实登记为风险与 Stage 4 对账项。
- **`T.erf` 标量循环在 NPU 编译期段错误（EXIT=139，无诊断）**：迁移 GPU 源码里的 `T.erf(T.cast(x,'f32'))` 形态时，不要尝试用 `for i in T.serial: buf[i] = T.erf(buf[i])` 承载，进程直接崩溃且无 Python 异常可捕获；NPU 侧超越函数只能用 v-prefix 整 buffer op（若该 op 精确）。
- **遇到「设计所采纳的核心 API 语义不可用」时，正确出口是 `[DESIGN_ERROR]` 而非自行换算法**：即使已找到可用的替代（本任务 `T.vexp` 实测 rel err 1.2e-7，可搭 erf 逼近），自建算法属设计层决策（需重新做精度域设计与等价性论证），应带完整证据回设计修订循环，同时把缺失能力登记到 `.agents/evolution/capability-gaps.md`、把可复现的最小反例转正为 `repro/`。
- **设计文档中显式标注「设备实测留 Stage X」的假设是高风险项**：Stage 3 应在写整 kernel 前优先补该实测（本任务若先跑探针，可在数分钟内判定 `T.verf` 不可用，而非编译 5 个 kernel 后才在第 1 个 L0 用例失败）；后续迁移函数（同批 harness 其他 unary/binary elementwise）应先扫一遍设计里的「留待实测」条款。

## Stage 1 (design) Retrospective — revision_2（机械门禁修正）— 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Gate-mechanics | `gate_lint.py` 的 `S1-EQUIV-EXEC` 对 §1.6.1 区域做**纯字面量**扫描：为检测真实的失败项，规则 `if "EQUIV_FAIL" in r161` 会把**说明性文字**里出现的同名字面量也判为失败。修订 1 的 §1.6.1 说明第 ④ 条写「…超差计数全 0，无 `EQUIV_FAIL`」——语义上正确，却触发门禁（内容零错误、机械误报） | `.agents/tools/gate_lint.py:327`（`if "EQUIV_FAIL" in r161`）；`examples/erf/erf_fwd_kernel/history_version/design_v1.md:327` | 门禁规则应改为**结构化判定**：只扫描结果表的「结论」列，或要求失败项须由脚本产出的 verdict 单元格承载；不应把自由说明文字纳入字面量匹配。或在 design 模板中明确禁止在 §1.6.1 正文里书写失败项字面量 | R |

### Transferable Lessons（可迁移教训）

- **门禁字面量规则会与「诚实说明」冲突**：为保证论证透明而在正文写「无 X 项」时，若 X 恰好是门禁的否定字面量，就会误报。设计文档在 §1.6.1 说明区应**避免直接书写门禁否定字面量**，改用同义表述（「全部通过」「超差计数均为 0」）。本次修正即仅改此一处转述，技术结论（候选式/基线式/6 行结果表/`verify_equiv.py`）全部保持不变。
- **机械修正应保持技术内容冻结**：本次为 revision_2，重跑 `verify_equiv.py` 仍为 ALL 6 ITEMS EQUIV_PASS，`design_lint` 返回无 failure / 无 warning。证明修正未弱化任何论证。

## Stage 2 (review) Retrospective — revision_1_review — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 对该一维逐元素设计仍同时 skip `ub_budget` 与 `core_split`（"no per-level buffer tables parsed" / "insufficient parsed dims {}"），维度 2/3 的机械拦截完全失效，退化为纯人工（v0/first_review 已提同问题，revision_1 未消解） | 本检视输出 `design_calc_check` JSON：`ub_budget: skip`、`core_split: skip` | 为该族扩展解析（UB 表按 buffer 名/字节列提取；core_split 从 `ceildiv(...)`/核数常量文本提取），或在 skill 维度表列出该族**强制人工核对清单** | R |
| Info-source | `design_calc_check.py` 的 `capacity_constants.ub_inflation` 默认 **1.7**（910B2C auto-multi-buffer 口径），而 DESIGN §4.5 用 **1.12**（显式 alloc 结构口径）——同一 UB 预算存在两套并存系数，检视须自行补算第二口径方能判「负向/容量论断是否成立」；本次按 1.7 复算 fp32=139KB 仍 <192KB 才判 pass | `.agents/tools/design_calc_check.py` 输出 `capacity_constants.ub_inflation=1.7`；`DESIGN.md` §4.5（×1.12）；`pattern-library/constants.md` CONST-capacity-910B2C（两口径均有记载） | 在 tool 输出中并列两口径（或注明「本结构适用哪一口径」），使 Stage 2 无需人工推断 | R |
| Citation-check | Stage 2 skill 无「DESIGN 引用的 pattern-library 条目须真实存在」的检查项。本次发现 §1.6.0 调研范围③ 宣称 `traps-runtime.md 的 TRAP-verf-maclaurin-truncation`，该条实际**未登记**（仅存于 `capability-gaps.md` 的 CG-2026-0015 与 repro 脚本头）——引用不实但非阻塞（核心证据另有 CG-2026-0015 + repro 复核） | `grep TRAP-verf pattern-library/` 无命中；`traps-runtime.md` 条目清单无 verf 条；`DESIGN.md` §1.6.0 调研范围③ | 在维度 7/8 增设「pattern-library/reference 条目 ID 存在性」核对（含 repro 文件头声明的 Entry 是否已登记） | R |
| Device-recheck | Stage 2 的机械复核只要求 CPU 脚本（`verify_equiv.py` + `design_calc_check.py`），未要求（或提示）在设备可用时重跑设计期设备探针；而 v0 的失败恰是「CPU 参照域无法覆盖设备实现」。本次设备可用，检视方自行重跑 `repro/TRAP-verf-maclaurin-truncation.py` 与两个 `design_probe_*.py` 复现了全部设备数字（3.725e-7 等），显著提升了维度 8 复核强度 | 本检视实际重跑三脚本全部复现；`.agents/skills/tilelang-design-review/SKILL.md` Phase 1 第 7 步仅列 CPU 两脚本 | 在 Phase 1 第 7 步增「设备可用时，重跑设计期设备探针并比对 §3.4/§0.6 数字」为可选/推荐步骤 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **迁移检视的「设备探针可复现性」判据**：维度 8 的设备精度断言应要求设计期探针脚本**可被 Stage 2 独立重跑并复现同一数量级**；仅「脚本存在 + 自述结果」不足以形成证据闭环（v0 教训的反向应用）——本次重跑 A&S 链 fp32 复现 `max_abs=3.725e-7`（逐分区间 3.725/2.384/1.788/1.788/0 e-7）、角点逐值与 golden 一致，方确认 §0.6/§3.4 数字可信 | P | `history_version/design_probe_erf_ops.py`（重跑输出）；`DESIGN.md` §0.6 R2 / §3.4 | `python examples/erf/erf_fwd_kernel/history_version/design_probe_erf_ops.py` / `_ops2.py` / `repro/TRAP-verf-maclaurin-truncation.py` | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | tilelang-design-review skill（维度 8「替代 API 佐证」/「调研结论正确性」行） |
| 被拒候选的误差量级须核对或标注「量级估计」：本次独立复算发现 §1.6.0 R4 对被淘汰候选的误差量级偏乐观（Winitzki 实测 ≈1.05e-3 vs 设计记 ~1.2e-4；最佳二参 tanh ≈1.9e-2 vs 记 ~1e-3），结论（不达标）稳健但数字不精确 | P | `DESIGN.md` §1.6.0 R4 表；独立 mpmath/numpy/scipy 复算 | none（人工复算） | torch/numpy/scipy（CPU） | tilelang-design-review skill（维度 8 调研结论正确性行：被拒候选误差同样纳入复算） |

### Transferable Lessons（可迁移教训）

- **设备可用时，Stage 2 应亲自重跑设计期设备探针**：skill 只强制 CPU 两脚本，但迁移检视的价值恰在「设备实现 vs 文档/CPU 代理」的落差。重跑成本（本次约 3–4 分钟）远低于因采信未复现数字而放行错误设计的代价。
- **被拒候选的误差量级也要独立复算**：淘汰结论由「远大于门禁」支撑，故量级偏乐观（1e-3 写成 1e-4）通常不翻转结论；但若某被拒候选的真实误差落在门禁附近，量级错误就可能误淘汰可行解——须逐候选复算或标注「量级估计」。
- **UB 容量口径存在双系数，检视须按最悲观口径复算一遍**：`design_calc_check.py` 默认 1.7（platform auto-multi-buffer）与设计常用 1.12（显式 alloc）并存；只要按 1.7 复算仍 <192KB，容量结论才稳健。`bs` 放大档（>2048）尤其要用 1.7 预筛。
- **引用不实 ≠ 证据缺失，但须记录**：设计引用了一个未登记的 pattern-library 条目名（TRAP-verf），属 provenance 缺口；只要核心负向断言另有真实证据（CG-2026-0015 + repro 重跑），判 warn 并建议登记/改引，不因此阻塞。
- **`verify_equiv.py` 重跑须逐行与内嵌表比对，而非只看 `RESULT` 行**：本次 6 行 max_abs 全部一致（5.662e-7 / 1.221e-4 / 3.052e-5 / 0 / 0 / 1.394e-7），确认内嵌表非手抄失真。

## Stage 3 (develop) Retrospective — first_impl (revision_1) — 2026-10-12

**结果**：`[PRECISION_PASS]`。本 attempt 为设计修订（v1）后的首次实现，按 `DESIGN.md` §3.3 的 A&S 7.1.26 21-op v-prefix 向量链逐句落地（一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` grid-stride + 整 buffer `v-ops` + fp32 中转 + 动态尾块 `tail`）。**首次 L0 即全过，实现层零重试**（`stage_retry_count[3]=0`）。

分层结果（设备实测，逐用例 max_diff）：

| 层 | 结果 | 关键数值 |
|----|------|---------|
| L0（6 用例：randn 全域 ×3 dtype + 角点 `[0,3,-3,100]` + `linspace(-8,8)` + 非整除尾块） | pass 6/6 | fp32 5.81–5.89e-7；fp16 1.2207e-4；bf16 7.63e-6；**角点 max_diff=0.0** |
| L1（8 用例：16m ×3 dtype + 尾块 ×3 dtype + exact/partial tile） | pass 8/8 | fp32 6.63e-7；fp16 1.2207e-4；bf16 7.63e-6 |
| L2（6 用例：1/2/10/2049/4095/8191） | pass 6/6（不阻塞） | 均 ≤6.1e-5 |
| Boundary（special/inf-nan/manifest 16m–64m） | pass（不阻塞） | inf→±1、NaN→NaN、±3.9→±1 逐值一致 |

**接口/边界实证**：`x=+inf→1.0`、`x=−inf→−1.0`、`x=NaN→NaN`、`x=±3.9→±1.0`、`x=0→0`（fp32/fp16/bf16 三档逐值一致，与 `torch.erf` fp32 语义一致）；大 `|x|` 饱和由 `e^{−x²}` 下溢自然实现，无需 clamp。v0 的必失败角点 `[0,3,-3,100]` 本版 **max_diff=0.0**（`erf(3)=0.999978`，不再是截断多项式的 37.9）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | DESIGN.md §10.1/§10.2/§10.3 规定交付「`erf_fwd_kernel.py`（kernel+golden）+ 独立 `test_erf.py`（分层测试）」两份文件，而 `tilelang-op-develop` 的 `{op}.py` 结构规范要求 kernel/golden/`run_L0..run_boundary`/`--level` main **同文件内嵌**。两处交付物口径不一致，实施者需自行二选一（本任务按 skill/task 要求内嵌于 `erf_fwd_kernel.py`，未产出 `test_erf.py`） | `examples/erf/erf_fwd_kernel/DESIGN.md` §10.2/§10.3；`.agents/skills/tilelang-op-develop/SKILL.md` §4（`{op}.py` 结构规范含分层测试与 main） | 统一 design 模板与 develop skill 的测试文件口径（design §10 改为「分层测试内嵌 `{op}.py`」或 develop 明确支持外置 `test_{op}.py`） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **无原生 erf 元算的自建逼近链通过整 kernel 设备验收**：A&S 7.1.26（21 op：`vabs/vmul/vadd/vrec/vexp/vsub/vcmp/vselect`）在完整 grid-stride kernel（persistent `T.serial` + 动态尾块 + 16m/64m 大 N）上复现设计期探针精度——fp32 max_abs 6.63e-7（门禁 1e-5，15× 余量）、fp16 1.2207e-4（<1 fp16 ulp@1.0）、bf16 7.63e-6，L0/L1 全过 | D | `examples/erf/erf_fwd_kernel/erf_fwd_kernel.py`（`--level all` 输出）；对照设计期 `history_version/design_probe_erf_ops*.py` | `cd examples/erf/erf_fwd_kernel && python erf_fwd_kernel.py --level all` | tilelang 0.1.2+6bffe5ceab95e3faddbacdf4cbc6adaae0568db6 / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | pattern-library/elementwise.md（erf 族首例：无原生元算时的逼近链落地模板）；CG-2026-0015 闭环 |
| **op 链内 UB buffer 原地复用别名纪律（elementwise 长链适用）**：设备验证可安全成立的三条——① `T.vrec(t, t)` 原地倒数；② Horner 结束后把 `t` 复用为 sign（`T.vselect(cond, ones, negones, t)`）；③ 计算域 buffer（fp32 路径 = `x_ub`，fp16/bf16 路径 = `xf`）在链尾经 `T.vsub(...)`/`T.vmul(...)` 原地承载结果并直接搬出，无需独立 fp32 结果 buffer。前提：读取源数据的 `vabs/vcmp` 必须发生在覆盖该 buffer 之前（`vcmp` 取 `x≥0` 掩码先行捕获） | P | `examples/erf/erf_fwd_kernel/erf_fwd_kernel.py` L~150–195（chain + reuse）；`DESIGN.md` §3.3 ⑤；`history_version/design_probe_erf_ops2.py` 同款复用 | `cd examples/erf/erf_fwd_kernel && python erf_fwd_kernel.py --level L0`（角点 max_diff=0.0 证明 `cond` 未被覆盖） | tilelang 0.1.2+6bffe5ceab95e3faddbacdf4cbc6adaae0568db6 / Ascend910_9362 / CANN 9.0.0 / 2026-10-12 | pattern-library/elementwise.md（v-op 原地复用与别名生命周期） |

### Transferable Lessons（可迁移教训）

- **同一批 harness 的后续 unary/binary elementwise 函数实施时**：若设计文档已把「新采纳 v-prefix API 的目标设备 micro-probe」落到 `history_version/design_probe_*.py`，Stage 3 只需在整 kernel 规模（多核 grid-stride + 动态尾块 + 大 N）复跑验证，不必重复设计期小 grid 探针——erf 本次因此首次 L0 即全过、零重试。反之，若设计未附设备探针，Stage 3 必须先补探针（v0 的 `T.verf` 教训）。
- **长 v-op 链的 UB 复用前提是「读取源数据的前置 op 先执行、之后才允许覆写」**：`vabs`/`vcmp` 等从输入 buffer 取值必须在 `vsub`/`vmul` 原地写回同一 buffer 之前完成；含符号算子的 elementwise 须先 `vcmp` 捕获 `x≥0` 掩码再复用该 buffer，否则符号判定读到被覆盖后的值。此形态经设备验证（角点 `[0,3,-3,100]` max_diff=0.0）。
- **erf 类无原生元算的逐元素算子，最终边界行为由逼近式的自然饱和给出**：`|x|≳6` 时 `exp(−x²)` 在 fp32 下下溢为 0 → 输出 = `sign(x)`，与真值 fp32 舍入（`|x|≳3.9` 即 ±1）一致，无需显式 clamp；`±inf`/`NaN` 沿 `vrec(∞)=0`、`vcmp(NaN≥0)=false`、`vexp(−∞)=0` 的 IEEE 传播得到 `±1`/`NaN`（设备实测逐值一致）。
- **一维逐元素动态尾块在本 kernel 形态下由 `tail = T.min(block_size, shape-off)` + `T.copy` 用 slice、v-ops 用整 buffer 达成**（`4_000_000 % 2048 = 1216` 尾块 fp16 max_diff=1.22e-4 通过）；沿用 `PL-1.22`/R-7 纪律，不把动态 slice 传给 v-op。
