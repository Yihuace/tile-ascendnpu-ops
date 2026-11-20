## Stage 1 (design) Retrospective — first_design (migration-harness: dropout/_make_dropout_kernel) — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | `tilelang-op-design/references/algorithm-candidates.md` 无「随机/PRNG 算子族」条目，dropout 调研只能靠模型知识 + 编译器源码手工枚举 RNG 候选（counter-based hash / LCG / xorshift / host 物化 / `fill_rng_2d`） | DESIGN.md §1.6.0 R1（子层 A 候选表）；kb_search "dropout RNG random mask" 无直接命中 | algorithm-candidates.md 新增 `ALG-rng-counter-based`（含 known_impl / kb_links：本设计 `design_probe_dropout_rng.py`） | C |
| Info-source | pattern-library / constraints 未记载「npuir 无 RNG 原语」「`T.vshr` int32 为算术右移且退化为标量指令」两条硬事实，须本任务从零复核（dir(T)/docs/源码/设备探针） | DESIGN.md §1.6.0 RNG 复核表 + §3.4；CG-2026-0021 | `tilelang-op-design/references/ascend-constraints.md` §1 增两条（RNG 原语缺失 / `vshr` 算术+标量）并指向 CG-2026-0021 | R |
| Design-flow | 迁移「随机算子」时 §0.1 语义解读的「contract golden」不能是解析函数——skill 的 golden 独立性准则（不得复刻 NPU 算法）对随机算子须扩展为「分布契约 + 重放契约」，脚本需性质化 | DESIGN.md §0.1 / §8.1 / verify_equiv.py（性质化） | migration-analysis.md §2.2 或 design skill 增「随机/非确定算子 golden 形态」说明 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir 无 RNG 前端原语：`T.rng_init`/`T.rng_rand_float` 不存在（`dir(T)` 无 rng 属性、docs 全 16 子目录无条目、customize_npuir.py 无算子）；BishengIR 有 counter-based LCG `linalg.fill_rng_2d`（core_named_ops.py L1814）但前端不可达 → 随机算子须核内自实现 PRNG | D | DESIGN.md §1.6.0 RNG 复核表 / §0.6 R2；.agents/evolution/capability-gaps.md CG-2026-0021 | `python -c "import tilelang.language as T; print([a for a in dir(T) if 'rng' in a.lower() or 'rand' in a.lower()])"` → `[]` | tilelang 0.1.2+fa52567a / Ascend910_9362 / CANN 9.0.0 / 2026-10-14 | pattern-library traps-compiler.md + capability-gaps.md CG-2026-0021 |
| `T.vshr`（int32）是**算术右移**且编译期退化为**标量指令**（告警 "will execute by scalar instruction with low efficiency"），且移位量须为 tensor（`npuir_shr` 不接受 Python 标量）——hash/PRNG 应避免 `vshr`，改用「左移 + 回绕乘」构造 avalanche | D | DESIGN.md §3.4 / §0.6 R2 / §1.6.1 #3；design_probe_dropout_rng.py S1 | `python design_probe_dropout_rng.py`（vshr 段：`-1>>1 = -1` 证算术；编译告警证标量退化） | tilelang 0.1.2+fa52567a / Ascend910_9362 / CANN 9.0.0 / 2026-10-14 | pattern-library traps-compiler.md（与 VP-2026-0130 并列） |
| `T.vmul`（int32）**回绕 mod 2^32**（可用于 hash 乘法）；但 int32 常量若以 `0x9E3779B1`（>2^31）写入 `vmul` 会编译报 `Literal value exceeds maximum of int32`——须用有符号形式（`-1640531527`） | D | DESIGN.md §3.4 / §9.2；design_probe_dropout_rng.py S2 | `python design_probe_dropout_rng.py`（S2：`2147483647*3→2147483645` 证回绕；无符号常量报错见探针迭代留痕） | tilelang 0.1.2+fa52567a / Ascend910_9362 / CANN 9.0.0 / 2026-10-14 | pattern-library elementwise.md / traps-compiler.md |
| torch `native_dropout` 是**乘性掩码**语义（`x*(m*scale)`；实测 p=1 对 `±inf` 返回 NaN、保留 `-0.0`）——故算术 0/1 掩码乘 `(x*scale)*m` **既免 910_9362 上 6.5x 的 `vselect`，又比源 `if_then_else`（对丢弃 `±inf` 返回精确 0）更贴合契约 golden** | D | DESIGN.md §0.6 R4 / §1.6.1 #1；verify_equiv.py（诊断行：源选择式 2–3/6 偏离，算术乘 0/6） | `python verify_equiv.py`（ALL EQUIV_PASS；诊断段） | tilelang(n/a, CPU) + torch CPU / 2026-10-14 | pattern-library PL-1.29 关联 / algorithm-candidates.md |
| 1-D `(block_size,)` 的 `T.arange(lane,[1],0)` ramp 设备实测逐位正确（未触发 `VP-2026-0171` 的 `(1,N)` 单行静默错尾元素）——**对 VP-2026-0171 的反例数据点**（该 bug 限于 rank-2 单行 tile，非 rank-1 buffer） | D | DESIGN.md §3.4 / §1.6.2 #1；design_probe_dropout_rng.py（hash 与 numpy 逐位一致即证 ramp 正确） | `python design_probe_dropout_rng.py`（`[M1/M2] bit-exact vs numpy=True`） | tilelang 0.1.2+fa52567a / Ascend910_9362 / CANN 9.0.0 / 2026-10-14 | pattern-library elementwise.md（更新 VP-2026-0171 适用范围） |
| 随机算子的 D-1 机器等价验证范式：分布等价（top-11 χ² + lag-1 χ²）+ 确定性重放 + 阈值概率精确性（≤2^-24）+ 掩码算术对 golden 的逐值一致（含 ±inf/NaN 角点）——可复用为任何非确定算子的等价判据 | P | DESIGN.md §1.6.1（结果表）/ verify_equiv.py | `python verify_equiv.py`（FAIL items=0） | torch CPU（无设备）/ 2026-10-14 | algorithm-research.md §3 R2 / quality-checklist.md |

### Transferable Lessons（可迁移教训）

- **npuir（target="npuir"）没有任何 RNG 前端原语**：`T.rng_init`/`T.rng_rand_float` 不存在，BishengIR 的 counter-based `linalg.fill_rng_2d` 未暴露给前端。迁移任何随机算子（dropout / 采样 / 随机增强）必须在 kernel 内用 int32 向量元算自实现 counter-based hash（`f(seed, 元素索引)` 无状态纯函数，天然可重放且与线程布局无关）；host 侧物化 mask 被 `ascend-constraints.md` §4 禁止（核心计算不得移出 kernel）。
- **`T.vshr`（int32）是算术右移 + 标量退化，禁用**：构造 hash 时不要用 `h ^= h >> k`（右移），改用 `h ^= h << k`（左移，标量移位可用）+ 回绕乘（`T.vmul` int32 回绕）来补 avalanche；需要「右移」语义时用**阈值放大**的代数等价变形（如 `⌊hm/256⌋ ≥ T ⟺ hm ≥ 256T`）绕开。
- **int32 hash 常量必须用有符号形式**：`T.vmul(h, 0x9E3779B1, h)` 编译报 `Literal value exceeds maximum of int32`，写 `-1640531527`（同 bit pattern）；`T.vxor` 的 src1 必须是 tensor（标量报 "cannot be a scalar"），用常量 buffer（`T.fill`）承载；`T.vmax` 无标量 src1（下界裁剪用零 buffer），`T.vmin` 允许标量 src2。
- **torch dropout 是乘性掩码**（`x*(m*scale)`，丢弃 `±inf` → NaN、保留 `-0.0`）：用「算术 0/1 掩码乘」而非 `if_then_else`/`vselect`——既免 `vselect` 的高代价，又贴合契约 golden；迁移随机算子时不要照抄源选择式的「精确 0」语义，先实测 torch 基线语义。
- **随机/非确定算子的 golden 不能是解析函数**：验证须性质化（统计 3σ + 确定性重放 + 缩放 + 边界），并用 fp64 参照单独验证「掩码算术」对 torch 语义的逐值一致（`verify_equiv.py`），不得以「与 GPU mask 逐位相同」为判据（cuRAND 流跨设备不可复现）。
- **1-D `(N,)` 的 `T.arange` 可用**（rank-1 buffer 不触发 `(1,N)` 单行陷阱 `VP-2026-0171`）；若遇异常，退路 = 2-D `(2, bs/2)` ramp + `T.reshape`。

## Stage 2 (design-review) Retrospective — first_review (migration-harness: dropout/_make_dropout_kernel) — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | `design_calc_check.py` 对含「多 buffer（12 个）逐级别表 + 分核 §5.5 三要素」的纯 Vector 设计**全部 skip**（`ub_budget`/`l0c_budget`/`core_split` 均 `no/insufficient parsed … dims {}`），机械位几乎未提供拦截力，UB/分核全靠人工复算 | 本次 `design_calc_check` JSON：3 skip + 仅 `r3_metrics` pass；DESIGN.md §4.5（12 buffer 表）/§5.5 | 工具解析器扩展：识别「单位 KB 的逐 buffer 表」与「`num_logical_kernels = ceil(N/block_size)` + 逻辑核数表」，或文档化 skip 触发条件供检视者判定 | R |
| Check-gap | 检视维度表**无「随机/非确定算子统计验证的 seed 覆盖」检查项**——本任务 verify_equiv 的 χ²/lag-1 门禁只用单一 seed，检视者须自行跨 seed 复算才发现投影位 χ² 对 seed 敏感（虽经掩码级自相关复核判定无实质相关） | REVIEW.md §机械复核 / 备注 2；本 Agent 跨 9 seed 独立扫描 | 本 skill §3 维度 8 或 design skill 增「随机算子等价验证须覆盖 ≥2 个不同 seed（含生产/测试用 seed）」检查项 | R |
| Evidence-rule | 「负向断言」规则只要求「引用 doc 路径 + 限制条款 或 标注未文档化假设」，但**设备探针类负向结论**（如 `T.vshr` 算术右移+标量退化）无「须附探针可复现命令/输出摘要」要求，检视者无法区分探针实测与自述 | DESIGN.md §3.4/§R-2（`vshr` 结论）；`capability-gaps.md` CG-2026-0021 | evidence 规则增「设备探针负面结论须附探针脚本路径 + 关键输出行」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 随机算子设计检视的独立复核范式：① 亲自跨 seed 复算 χ²/lag-1（暴露单 seed 门禁的 seed 敏感性）；② 对**最终二值掩码**做 lag-1 自相关 / 2×2 配对 χ²，判定投影位统计波动是否转化为掩码层相关——本例 9 seed 掩码自相关仅 1e-4~5e-4、配对 χ²≤1.2，证明设计采纳的 hash 在算子契约层无实质相关 | P | REVIEW.md §8「调研结论正确性（独立复核）」；本 Agent 独立脚本（跨 seed χ² + 掩码级自相关） | 本次检视临时脚本逻辑（可固化为 `verify_review_stochastic.py`：跨 seed 均匀性 + 掩码级独立性） | / 2026-10-14 | tilelang-design-review skill（§3 维度 8 检查项）+ algorithm-research.md |
| 迁移「随机算子」设计检视要点：源 RNG 原语缺失时，等价判据须为「分布等价 + 确定性重放」（非与 GPU 掩码逐位一致）——检视时须独立核对设计是否显式声明该口径，且 verify_equiv 的候选/基线选取须为「采纳式 vs 契约 golden」而非「采纳式 vs 源式」；同时独立核对算术掩码乘在 `±inf/NaN/-0.0` 角点与 torch 乘性语义逐值一致 | P | REVIEW.md §0「语义理解正确性」/§8「等价性独立推演」；DESIGN.md §0.1/§1.6.1/verify_equiv 诊断段 | `python verify_equiv.py`（诊断段：源选择式 2–3/6 偏离，算术乘 0/6） | torch CPU / 2026-10-14 | migration-analysis.md §2.2 + algorithms-research |

### Transferable Lessons（可迁移教训）

- **随机算子检视不能只看设计自述的「χ² 达标」**：单 seed 门禁会掩盖 seed 敏感性。检视者应跨 seed 独立复算，并**下探到算子实际使用的二值掩码层**做独立性复核——投影位统计的波动可能不转化为掩码层相关（本例即如此）。
- **纯 Vector 迁移设计的机械复算位常失效**：`design_calc_check.py` 对 12-buffer 表与 `ceil(N/block_size)` 分核形态解析失败 → UB/分核必须人工逐 buffer、逐 workload 复算，并**独立重跑 `NPUUtils.get().get_aicore_num()`/`get_aivector_core_num()`**（本例实得 20/40，验证设计 §5.5 ②）。
- **迁移随机算子的等价口径**：源 RNG 原语不可用时，「分布 + 重放 + 缩放 + 边界」性质化验证是正确判据（不得要求跨设备掩码逐位一致）；检视时须核 verify_equiv 的基线选「契约 golden」而非「源式」——源式在 `±inf` 角点反而偏离 torch 乘性契约，是采纳算术掩码乘的必要性证据。
- **负向 API 断言的旁证方式**：检视「某 API 不存在」时，除设计所引 doc 路径外，应亲跑 `dir(T)` 过滤 + 全目录 grep + 全仓 grep，并区分「编译器有 named op 但前端无映射」（如 `linalg.fill_rng_2d`，CG-2026-0021）与「完全不存在」。

## Stage 3 (develop) Retrospective — first_impl (migration-harness: dropout/_make_dropout_kernel) — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Test-design | `tilelang-op-develop` §4 模板与 DESIGN.md §8.1 的 `check_dropout_contract` 均以「count zeros / N」作为 drop-rate 度量，但**未写明该度量只在常量非零输入（ones）上无偏**；Stage 3 直接用 randn 输入复用该检查会产出假失败（L1-2 bf16 replay rate=0.498056，3σ=1.5e-3，而设备掩码实为正确） | `_make_dropout_kernel.py` §check_dropout_contract；本次 L1-2 首轮失败与 `run_L1` 修复 | 开发 skill §4 或 design §8.1 为随机算子标注「rate 检查须用 ones 输入；随机输入用结构 + 重放判据」 | R |
| API-trap-doc | DESIGN.md §3.3 的「无运行时 `shape` 参数 + 编译期 N 算 `tail`」+「`h`/`t` 双 buffer 原地复用（idx→hash→mask 单缓冲链）」形态未在 pattern-library elementwise.md 记为已验证形态，Stage 3 须自行确认可用 | `_make_dropout_kernel.py` `main(x,y)`（无 shape）+ 设备实跑与 numpy int32 模型逐位一致 | pattern-library elementwise.md 增「编译期静态 N + 尾块；v-op 原地复用读改写同 buffer」已验形态 | P |
| Test-robustness | golden 结构检查若写成「`y ∈ {0, 常量 scale}`」则仅对 ones 输入成立；对 randn 输入会把每个保留元素 `x_i*scale` 误判越界 | 本次首轮 L0-4 结构失败 → 改为元素级 `y ∈ {0, x*scale}` | 开发 skill 模板的精度比较函数默认用「逐元素期望（x 参与）」，避免常量期望的隐式 ones 假设 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 随机算子的 rate/零计数契约度量**仅在常量非零输入（ones）上无偏**：fp16 randn(1M) 含 ~7e2 精确 0、bf16 ~6e3（fp32 ~0）；`(y==0).mean()` 高估、`(y==0)&(x!=0)` 低估，均无法从随机输入恢复真实 drop rate。验证随机算子时 rate 用 ones，随机输入改走结构（`y∈{0,x·scale}`）+ 精确重放 | P | `examples/dropout/_make_dropout_kernel/_make_dropout_kernel.py`（`check_dropout_contract` / `_run_case` auto rate）；首次 L1-2 失败记录 | `examples/dropout/_make_dropout_kernel/repro/CONST-dropout-rate-metric-random-input.py`（torch CPU，断言通过） | torch CPU / 2026-10-14 | pattern-library elementwise.md（随机算子测试模式）+ tilelang-op-develop §4 |
| DESIGN §3.3 的核内 counter-hash dropout 形态在 910_9362 上**逐位可用**：一维 40 核 + `T.serial` 网格跨步、**无运行时 shape 参数**（`tail = T.min(bs, N-off)`，N 编译期常量）、`h`（idx→hash）与 `t`（hash→mask）**单缓冲原地复用**、常量 buffer 循环外 `T.fill`、1-D `T.arange(lane,[1],0)`；ones 输入的零计数与 numpy int32 模型**完全相等**（N=1M p=0.5 seed=777 → 500963） | D | `examples/dropout/_make_dropout_kernel/_make_dropout_kernel.py`（kernel 实跑）；本次设备实测 | `python examples/dropout/_make_dropout_kernel/_make_dropout_kernel.py --level all`（L0 22 + L1 12 全过） | tilelang 0.1.2+fa52567a / Ascend910_9362 / CANN 9.0.0 / 2026-10-14 | pattern-library elementwise.md（persistent 1-D grid-stride + 原地复用） |

### Transferable Lessons（可迁移教训）

- **随机算子的 drop-rate 契约只在常量非零输入上能量准**：低精度随机输入自带精确 0（fp16 ~7e2/1M、bf16 ~6e3/1M），任何基于「零计数」的 rate 度量都会偏离；rate 门禁用 ones 输入，随机输入用**元素级结构检查**（每个输出 ∈ `{0, x·scale}`）+ **精确重放**（同 seed `torch.equal`）。
- **精度比较函数不要隐含「输入为常量」假设**：结构/缩放检查应写成逐元素 `|y - x*scale| <= atol + rtol*|x*scale|`，而非与常量 `scale` 比较；后者只对 ones 成立，换成 randn 会假失败。
- **npuir 上「编译期静态 N + 无 shape 参数」的尾块是可用的**：`main(x, y)` 内 `tail = T.min(block_size, N - off)`（N 为 Python 常量）与带运行时 `shape` 的写法等价；配合核内 `T.serial` 网格跨步（循环上界静态）构成完整 persistent 形态，先例 `bitwise_not_fwd_kernel.py`。
- **v-op 支持同 buffer 读改写（原地复用）**：`T.vadd(lane, base, h)` → `T.vxor(h, seedb, h)` → … 单缓冲链在设备上与 numpy int32 模型逐位一致；`h`/`t` 两 buffer 即可承载 idx→hash→mask 全链，按 DESIGN 预算省 UB（本项目 12-buffer 布局 ×1.7 仍 < 192KB）。


