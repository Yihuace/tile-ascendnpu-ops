# RETROSPECTIVE — examples/elu/_make_elu_kernel/

> 追加式日志：每个 Stage 执行追加一个章节，不覆盖历史章节。

## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | `tilelang-op-design` SKILL.md §4 Phase 4 的章节模板把 §8 列为「CV 融合设计」、§9 列为「验证方案」，但同仓机械门禁 `gate_lint.py` 的 `DESIGN_REQUIRED_SECTIONS` 只要求 §8，并在 §8 region 内检索 `Golden`/`L0`（即要求 §8 = 验证方案）。按模板编号写（CV=§8、验证=§9）会被 `S1-GOLDEN-PLAN`/`S1-L0-PLAN` 判 fail——本任务首跑即撞上（3 fail，2 个源于此）。 | 本目录 `DESIGN.md`（修订前 §8=CV/§9=验证）；`.agents/tools/gate_lint.py` L156 `DESIGN_REQUIRED_SECTIONS`、L382–L388 `S1-GOLDEN-PLAN`/`S1-L0-PLAN`（读 `region_all(sections, headings, "8")`） | 统一两处：把 design-template/§4 的验证方案改为 §8、CV 融合并入 §7 子节（与 expm1/erf 先例一致），或放宽 gate 使其接受验证方案位于 §9 | R |
| Info-source | 同族 expm1 任务的「小 \|x\| 分段 Taylor 消抵消」是**算子特定**结论，不是可无条件复用的 elementwise 配方；本任务若照搬会引入 16.1% 偏差（见 Value Point）。设计流程缺少「同族方案适用性前提核对」的显式检查项。 | 本目录 `DESIGN.md` §0.6 R2 拒绝项 / §1.6.1 诊断表；`examples/expm1/expm1_fwd_kernel/DESIGN.md` §0.6 R2 | 在 `references/algorithm-research.md` §4 迁移段加一句：同族先例的方案仅在「契约语义（golden）相同」时可直接承接，否则须先核对 golden 的数值实现 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| device F.elu 负分支用 exp(x)-1 而非 expm1(x)（CPU+设备双证）——决定 ELU 迁移用直译而非 Taylor 消抵消 | D | 本目录 `history_version/design_probe_elu.py` 第 1 项（设备 `F.elu(-1e-7)=-1.1920929e-7` == `exp-1`，≠ `expm1=-9.99999e-8`） | `python3 history_version/design_probe_elu.py`（需 NPU） | tilelang 0.1.2 dev root build + Ascend910_9362 + CANN 9.0.0 / 2026-10-13 | pattern-library/elementwise.md（新条目：F.elu/ELU golden 语义） |
| ELU 向量链设备端到端精度：fp32 max_abs 5.96e-8 / fp16 4.88e-4 / bf16 3.9e-3 / 角点 0.0 / 尾块 7.6e-6 / alpha=0.5 2.98e-8 | D | 本目录 `history_version/design_probe_elu.py`（smoke/corner/small-x/linspace/tail/alpha 全 PASS） | `python3 history_version/design_probe_elu.py` | tilelang 0.1.2 dev root build + Ascend910_9362 + CANN 9.0.0 / 2026-10-13 | pattern-library/constants.md（ELU 链精度数据点，补 CONST-elementwise-op-precision-910_9362） |
| 对 golden 用 exp-1 的算子，禁止套用 expm1/Taylor 消抵消（else 引入小 x 相对偏差，且是非等价优化） | C | 本目录 `verify_equiv.py` 非门禁诊断（x=-1e-7 相对差 1.611e-01）+ `DESIGN.md` §0.6 R2 拒绝项 | `python3 verify_equiv.py`（CPU） | 不依赖工具链（torch CPU） | pattern-library/cases.md（CASE-elementwise-elu-migration） |
| 编译期 `alpha==1.0` 折叠省去 vmul（×1.0 恒等）——elementwise 标量参数特化的通用 op 数削减形态 | P | 本目录 `DESIGN.md` §1.6.1 #1 + `verify_equiv.py` #4（EQUIV_PASS max_abs 0.0） | `python3 verify_equiv.py` | 不依赖工具链 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- 迁移 elementwise 激活/数学算子前，**必须先在 CPU 和设备上核对 golden 的数值实现**（例如 `F.elu` 负分支究竟是 `exp-1` 还是 `expm1`，`F.softplus` 的阈值分支等）；同族先例的算法修正方案（如 expm1 的 Taylor 消抵消）只有在**契约 golden 与先例同式**时才可承接，否则会引入非等价偏差。
- 当源公式本身与 golden 逐值一致且 NPU 有对应原生元算（如 `T.vexp`）时，**不要为了"更精确"引入更复杂的近似链**——那会偏离契约语义；正确做法是直译 + 仅做数学恒等的 op 削减（如编译期 ×1.0 折叠）。
- v-prefix 比较 API（`T.vcmp`）要求两个操作数**同 shape 的 tensor**（不支持标量直接比较），需显式 `T.fill` 一个同 shape 常量 buffer（循环外填充，不产生 per-tile op）。
- DESIGN.md 机械门禁要求 **§8 = 验证方案**（含 Golden + L0）——写文档时按 gate schema 而非 skill 模板的表面编号组织章节（CV 融合并入 §7）。
- 交付清单里**不要出现尚未创建的下游产物路径**（如 `{op}.py`）——`S1-REF-PATHS` 会逐一核对存在性并判 fail；用「算子目录下 `_make_elu_kernel.py`」等非仓库相对路径表述。

## Stage 2 (review) Retrospective — first_review — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 7「同类引用真实」只核对**引用文件路径是否存在**，不核对被引用文件内**具体条目/用例名**是否存在。本任务 DESIGN.md §0.7/§9.1 R-14/§10 声称 NPU 测试文件含 `test_elu_alpha`/`EluAlphaFixture`，文件路径真实但该用例不存在（NPU 测试文件开头明写 no non-default alpha case ported），检视时需额外 grep 用例名才能发现。 | 本目录 `DESIGN.md` L204/L901/L945；`examples/TileOPs/tests/ops/test_activation.py` L7–L8（"no non-default `alpha` case is ported here"）与 grep 无 `test_elu_alpha`/`EluAlphaFixture` 结果 | 在 `tilelang-design-review` SKILL.md 维度 7「同类引用真实」检查项补一句：被引用文件中被点名的具体测试用例/函数名须存在（`grep` 用例名），不能仅核对文件路径 | R |
| Review-flow | `design_calc_check.py` 对纯 Vector 算子的 `ub_budget` 与 `core_split` 均返回 `skip`（"no per-level buffer tables parsed" / "insufficient parsed dims"），机械拦截位在本类算子上失效，只能人工复核——检视维度 2/3 的机械兜底对 elementwise 类不可用。 | 本会话 `python3 .agents/tools/design_calc_check.py --design examples/elu/_make_elu_kernel/DESIGN.md` 输出：`ub_budget skip`、`core_split skip`、`r3_metrics pass`、`l0c_budget pass` | 在 `design_calc_check.py` 增加对一维 `(block_size,)` buffer 表的解析（`alloc_ub` 行），使纯 Vector 算子 UB 预算与核数算术也走机械计算 | R |
| Info-source | 检视侧复核「无原生 velu」负向断言时，`docs/Tilelang.language/数学操作/` 目录枚举 + `customize_npuir.py` grep 有效，但 review skill 未把「负向 API 断言须枚举全目录」列为维度 8 的显式动作（依赖检视者自觉）。 | 本会话 `ls docs/Tilelang.language/数学操作/`（无 `T.velu.md`，有 `T.vrelu.md`）+ grep `tilelang/language/` 无 `npuir_elu` | 在 review skill 维度 8「调研结论正确性」④ 负向断言复核条目中，显式写出「负向 API 断言须枚举 `docs/Tilelang.language/` 全子目录 + grep `customize_npuir.py`」的核对动作 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 设备侧独立复核 golden 数值实现（F.elu 负分支 == exp-1）作为维度 0 迁移前提的门禁动作——非仅依赖 DESIGN 自述 | D | 本会话设备实跑 `F.elu([-1e-7,-1e-6,-1e-3,…]) == torch.where(x>0,x,exp(x)-1)` 逐值一致；CPU fp32 同结论 | `python3 -c "import torch,torch_npu,torch.nn.functional as F; x=torch.tensor([-1e-7,-1e-6,-1e-3],dtype=torch.float32); print(F.elu(x.npu(),1.0).cpu().tolist(), (torch.exp(x)-1).tolist())"` | tilelang 0.1.2 dev root build + Ascend910_9362 + CANN 9.0.0 / 2026-10-13 | pattern-library/elementwise.md（ELU golden 语义 + 复核动作） |
| 迁移设计文档中被点名的测试用例名须真实存在（文件路径存在 ≠ 用例存在） | C | 本目录 `DESIGN.md` §9.1 R-14 引用不存在的 `EluAlphaFixture`/`test_elu_alpha`；`examples/TileOPs/tests/ops/test_activation.py` L7–L8 | `grep -n "test_elu_alpha\|EluAlphaFixture" examples/TileOPs/tests/ops/test_activation.py`（无命中） | 不依赖工具链 | tilelang-design-review SKILL.md 维度 7 |

### Transferable Lessons（可迁移教训）

- 迁移检视维度 0 的「golden 数值实现」前提**必须在目标设备实跑复核**，不得只信 DESIGN 自述：同名算子（如 `F.elu`）的 CPU/设备实现可能用 `exp-1` 而非 `expm1`，这直接决定迁移该直译还是该做算法替代——前提错则整个迁移方案失效。
- 引用校验要「文件路径 + 具体条目名」两级核对：被引用文件真实存在不代表其中被点名的用例/函数也真实存在（本任务 `test_elu_alpha` 即反例）。
- 纯 Vector 算子的 `design_calc_check.py` 的 `ub_budget`/`core_split` 会 skip，检视者须手工复算 UB 预算与分核算术，并标注「机械跳过，人工已核」。
- 负向 API 断言（"无原生 X"）复核须枚举 API 文档全目录 + grep 语言层导出（`customize_npuir.py` / `tilelang/language/`），不能凭记忆或单一目录判断。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | `tilelang-op-develop` skill §4 结构规范只要求含 `run_L0..run_boundary` + `--level`，未规定每个用例须打印「PASS/WARN + max_diff」，导致各算子报告口径不一（本实现自发遵循 erf 先例打印，但这只是惯例非规范）。 | 本目录 `_make_elu_kernel.py` `_run_tensor`（打印 `[tag] PASS: ... max_diff=...`）与 `examples/erf/erf_fwd_kernel/erf_fwd_kernel.py` L249 同款；skill §4 与 `templates/op_template.py` 均未提该要求 | 在 `tilelang-op-develop` §4 结构规范补一句：分层测试每用例须打印 `PASS/WARN + max_diff`，供 conductor/evolver 直接抽样 | R |
| Info-source | DESIGN.md §8.3 的 L0-1/L0-5/L0-6 实为同一 `(N=1<<20, float32, alpha=1.0)` 编译配置（仅输入分布不同），JIT 按 N 特化 → 三者共享一份编译产物；设计表把「输入分布」当「编译配置」列出，未说明编译去重，实现者易高估编译规模与耗时。 | `DESIGN.md` §8.3 L0-1/L0-5/L0-6 均为 N=1<<20 fp32；本目录 `_make_elu_kernel.py` L0 段实跑仅 10 次 `AscendNPU IR compile success`（12 个用例） | `tilelang-op-design` §8.3 注明「同一 (N,dtype,alpha) 的用例共享编译产物」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ELU 直译向量链（vcmp/vexp/vsub/vselect，alpha=1 折叠 vmul）NPU 端到端实测：fp32 max_diff 5.96e-8 / fp16 7.63e-6 / bf16 0.0；角点(-88.8/±1e-7/0/88.8)/±inf/NaN/非整除尾块全 PASS | D | 本目录 `_make_elu_kernel.py` run `--level all` 全层输出（L0/L1/L2/Boundary 全 PASS，无 WARN） | `cd examples/elu/_make_elu_kernel && python _make_elu_kernel.py --level all`（需 NPU） | tilelang 0.1.2+6d36281e (dev root build) + Ascend910_9362 + torch_npu 2.7.1.post4；CANN 版本本会话不可查（未记录） / 2026-10-13 | pattern-library/constants.md（ELU 链精度数据点） |
| K11 折叠 block_size（fp32=1024 / fp16·bf16=2048）不改变链精度：fp32 1M max_diff 与设计期 bs=2048 探针同为 5.960464e-08；bf16 全域 max_diff 恒为 0（输入/输出 bf16 量化支配链内 fp32 误差） | D | 本目录 `_make_elu_kernel.py` L0-smoke fp32 `max_diff=5.960464e-08`（bs=1024）；`history_version/design_probe_elu.py` 同值（bs=2048）；bf16 各用例 max_diff=0.000000e+00 | `cd examples/elu/_make_elu_kernel && python _make_elu_kernel.py --level L0` | tilelang 0.1.2+6d36281e (dev root build) + Ascend910_9362 / 2026-10-13 | pattern-library/elementwise.md |
| `T.vselect(cond, xc, e, xc)` 以计算域 buffer 同时作 dest 与 srcA 的原地别名，在 npuir Developer 模式合法且数值正确；链内 `e` 原地 i-i 复用（vexp→vsub→vmul）同样成立——ELU/erf 两族二次实证 | P | 本目录 `_make_elu_kernel.py` L166–L176（`T.vexp(xc,e)`/`T.vsub(e,1,e)`/`T.vselect(cond,xc,e,xc)`）；`examples/erf/erf_fwd_kernel/erf_fwd_kernel.py` L184–L186 同形态 | `cd examples/elu/_make_elu_kernel && python _make_elu_kernel.py --level L0` | tilelang 0.1.2+6d36281e (dev root build) + Ascend910_9362 / 2026-10-13 | pattern-library/layout.md 或 traps-compiler.md（PL-1.23 二次证据） |

### Transferable Lessons（可迁移教训）

- 迁移 elementwise 算子时，若 DESIGN 已给出「计算核直译 + 仅数学恒等削减」的方案，Stage 3 应**逐 op 照伪代码落地**，不做任何"顺手更精确"的替换（如把 `vexp(x)-1` 换成 expm1/Taylor 消抵消）——本算子逐 op 直译一次通过且与 golden 逐位一致；任何算法"改良"都会引入非等价偏差。
- 一维 Vector 算子用「`T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `tail = T.min(block_size, shape-off)` 尾块守卫」可在任意 N（含 N=1 / 2047 / 400 万非整除）下一次编译通过；**v-op 只作用于整 UB buffer、动态 slice 只进 `T.copy`** 是硬约束，务必遵守。
- 同一 `(N, dtype, alpha)` 的多个测试用例共享同一份 JIT 编译产物——设计 L0 用例表若按输入分布而非编译配置拆分，实现侧实际编译次数会低于表观用例数（本算子 12 个 L0 用例仅 10 次编译），排期时按编译配置数而非用例数估算。
- bf16 路径的端到端 max_diff 常为 0：输入/输出往返 bf16 的量化步长（~1e-2 相对）远大于链内 fp32 误差（~1e-7），故不能以 `bf16 max_diff==0` 反推"链更准"；精度结论须主要看 fp32/fp16 用例。
- 若 DESIGN 阶段已跑通目标设备端到端探针（本算子 `history_version/design_probe_elu.py` 几乎就是最终 kernel），Stage 3 可直接以其为雏形、仅补齐分层测试，显著省返工——设计期设备探针的投入回报很高。
