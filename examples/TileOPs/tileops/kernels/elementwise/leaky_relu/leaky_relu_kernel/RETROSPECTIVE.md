## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 迁移脚手架 wrapper 的算术域注释（`leaky_relu.py` docstring：输入 dtype 域“semantically required，must be preserved”）与本任务契约 golden（`test_activation.py::LeakyReluTest.ref_program = F.leaky_relu(x.float(), 0.01)`，fp32 域）**冲突**；skill 未提示“脚手架注释只是适配者解释、契约 golden 才是语义权威”的优先级规则 | DESIGN.md §0.1 / §0.6 R3 / §3.3 接口适配说明；`examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py` L10-L12,L34-L38 | 在 `tilelang-op-design` references/migration-analysis.md §2.1 增一条：迁移脚手架/coverage 注释属适配者解释，语义权威以源仓测试 golden 为准，冲突时以实测裁决 | R |
| Design-flow | 单步逐元素类的调研深度边界模糊：`algorithm-research.md` §2 判“轻量调研（四问各一行）”，但本算子因存在多等价形态 + 计算域分歧，实测需要 R1 候选表 + R3 四口径 + R4 逐候选才够；skill 未给“轻量算子何时升级为完整调研”的判据 | DESIGN.md §1.6.0 开头段（自述“属轻量调研范畴，但达到完整调研深度”） | 在 `algorithm-research.md` §2 增判据：轻量算子若存在 ≥2 个等价/特化形态或计算域/精度分歧，R1/R3/R4 须升级为完整表 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| LeakyReLU 计算域实测：fp32 中转 vs 输入 dtype 域对 `F.leaky_relu(x.float())` golden 的偏差（fp16 角点违反比 0.2597 / bf16 0.1297；fp32 中转逐位一致） | D | DESIGN.md §0.6 R3 / §1.6.1 #1；`examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py` | `python3 examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py`；设备端 `python3 .../history_version/design_probe_leaky_relu.py` | tilelang 0.1.2+e6aa5a80 / Ascend910_9362 / CANN 9.0.0 | pattern-library/constants.md（新增 CONST-leaky-relu-compute-domain-910_9362）或 algorithm-candidates.md ALG-elementwise |
| `max(x, s*x)` 等价形态：`0<s<=1` 时与 `where(x>0,x,s*x)` 在 fp32 域**逐位一致**（2 op 替代 3 op）；`s>1` 时非等价（诊断 max_abs 6.55e4）——编译期 guard 必要性实证 | D | DESIGN.md §1.6.1 #2 + 非门禁诊断表；`verify_equiv.py` #2 行 + DIAG 行 | `python3 .../verify_equiv.py`（#2 行 PASS；DIAG slope=2.0 EQUIV_FAIL） | tilelang 0.1.2+e6aa5a80 / Ascend910_9362 / CANN 9.0.0 | algorithm-candidates.md ALG-elementwise（R1 候选行） |
| 设备元算 dtype 补证：`T.vmax` / `T.vrelu` 同样**不含 bf16**（与 `T.vmul/vsub` 一致），故任何含 max/mul 的逐元素链对 bf16 均须 fp32 中转；`vmax` 无标量广播（须先 `vmul` 生成 tensor） | D | DESIGN.md §1.6.0 R4 dtype 矩阵段 / §3.5.1；`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1；设备探针 fp32dom-max 实跑 | `python3 .../history_version/design_probe_leaky_relu.py` | tilelang 0.1.2+e6aa5a80 / Ascend910_9362 / CANN 9.0.0 | pattern-library/traps-*.md（dtype 矩阵补证） |
| 迁移脚手架注释与契约 golden 冲突的裁决范式：以源仓测试 `ref_program` 为语义权威，用 verify_equiv/diagnostic 实测裁决，接口契约（签名/`main(x,y,shape)`/`block_size`）照旧遵守 | P | DESIGN.md §0.1 关键核对表 / §0.6 R3 / §3.3 接口适配说明 / §9.1 R-15 | 见 `examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md` | n/a | tilelang-op-design references/migration-analysis.md §2.1 |

### Transferable Lessons（可迁移教训）

- 迁移含标量参数的逐元素算子时，先核对**源算术域 vs 契约 golden 域**：源若在输入 dtype 域做 `cast(scalar, dtype)*x`，而测试 golden 用 `F.xxx(x.float(), scalar)`，半精度下该标量（如 0.01）不可精确表示 → 输入 dtype 域会偏离 golden。裁决：以 fp32 中转域实现（bf16 通常本就强制中转），并用 verify_equiv 量化两种域的门禁违反比。
- `max(x, s*x)` 是 LeakyReLU 的 2-op 等价形态，但**仅当 `0<s<=1`**：`s>1` 时正分支被错误放大（非等价）。任何此类“用 max/min 折叠分支”的等价形态都必须带编译期 guard，并产出非门禁 diagnostic 证明 guard 外的非等价。
- `T.vmax` / `T.vrelu` 与 `T.vmul`/`T.vsub` 一样**不支持 bf16**（docs §2.2.1 已核）；`T.vmax` 亦**无标量广播**（须先 `vmul` 生成同 shape tensor 再 tensor-tensor vmax）。设计逐元素链时 bf16 一律规划 fp32 中转。
- 迁移 wrapper/脚手架里的注释与 docstring 是**适配者解释而非语义契约**；与源仓测试 golden 冲突时，以 golden 为语义权威，用设备探针 + 等价性脚本实测裁决，同时完整遵守其接口契约（签名、`block_size`、`main(x,y,shape)`）。

## Stage 2 (design review) Retrospective — review — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | `design_calc_check.py` 对本设计的 `ub_budget` 与 `core_split` 两项均返回 `skip`（"no per-level buffer tables parsed" / "insufficient parsed dims {}; declared logical cores = 1"），而 DESIGN §4.3/§4.5 有显式逐 buffer 表、§5.5 有分核三要素——检查器对这两个正是其立项目标（VP-2026-0009 / 分核）的检查项**解析失效**，检视被迫全部转人工复核，机械门禁形同虚设 | `python3 .agents/tools/design_calc_check.py --design examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md` → `ub_budget: skip`, `core_split: skip` | 在 `tilelang-design-review` Phase 1 第 7 步与 `design_calc_check` 增强解析器：识别 §4.3/§4.5 类 buffer 表与 §5.5 逻辑/物理核数字；解析失败时输出可操作的「为何 skip」而非仅 "insufficient dims" | R |
| Evidence-vocab | verify_equiv/设备探针输出的 `viol_ratio`（= `max(|cand-base|/(atol+rtol|base|))`，容差消耗比）被 DESIGN 成文为「违反比」，与「元素违反率」易混；skill 无术语约定，检视需额外人工确认其非违反率 | DESIGN.md §0/§1.6.1「违反比 0.2597」；`verify_equiv.py` L98 `ratio` 定义；实跑两者 viol=0 | 在 `tilelang-op-design` 的 verify_equiv 模板/`design_calc_check` 输出中统一术语为「容差消耗比 / tolerance-margin ratio」，避免与 violation rate 混淆 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移逐元素算子的「计算域」检视法：契约 golden 域（`F.xxx(x.float(), s).to(dtype)`，fp32）与源 GPU 域（`cast(s,dtype)`，输入 dtype）分歧时，检视须独立 `python -c` 复算 `cast(s,dtype)` 表示误差并重跑 verify_equiv + 设备探针，确认设计所选域对 golden 的一致性 | P | DESIGN.md §0.6 R3 / §1.6.1 #1；`verify_equiv.py` 重跑 20/20 EQUIV_PASS；设备探针 `ALL PASS` | `python3 examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py`；`python3 .../history_version/design_probe_leaky_relu.py` | tilelang 0.1.2+e6aa5a80 / Ascend910_9362 / CANN 9.0.0 | tilelang-design-review（维度 0/8 计算域核对子项） |

### Transferable Lessons（可迁移教训）

- 迁移逐元素算子检视的第一风险是**计算域**：源 GPU 常在输入 dtype 域算 `cast(scalar,dtype)*x`，而契约测试 golden 多为 `F.xxx(x.float(), s)`（fp32 opmath）。检视必须亲自读源、读测试 golden，并用 `verify_equiv`/探针复核两域相对 golden 的实际偏差，不得因「源直译」而默认保留源域。
- `max(x, s*x)` 类「用取大折叠分支」的等价形态，检视须**自证 guard**：`0<s≤1` 时与 where 逐位一致，`s>1` 非等价（诊断 max_abs 6.55e4）——不重推等价条件不得给 pass。
- 机械门禁（`design_calc_check` 的 ub/core_split）对本格式设计返回 `skip` 时，**必须人工补做**并写明结论，不能以 skip 当 pass；返回中应显式记录 skip→人工复核。
- 本设计设备探针可直接重跑（Ascend910_9362 在线），检视应优先重跑设备探针而非仅信设计自述——本任务重跑结果与 DESIGN §3.4/§8.2 全部数字一致。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | Stage 3 的精度门禁直接复用了 ELU 模板的「golden 从喂给 kernel 的同一张量推导」约定，实测与设计期探针的报告**不一致**（设计报 fp16 randn 3.05e-5 / bf16 2.44e-4，Stage 3 实测逐位 0.0）；skill 未要求 Stage 3 **核对设计期误差预算数字的可复现性**，相位差一个「先升 fp32 再 cast」的输入就全量过门禁而不会暴露探针 harness 缺陷 | DESIGN.md §3.4/§8.2（fp16 randn 3.05e-5 / bf16 2.44e-4）；`history_version/design_probe_leaky_relu.py` `_run`（golden 取未舍入的 fp32 `x_cpu.float()`，kernel 喂 `x_cpu.to(dt)`）；Stage 3 `python _make_leaky_relu_kernel.py --level all` 全 41 用例 `max_diff=0.0` | 在 `tilelang-op-develop` Phase 1/4 增一条：Stage 3 测试 harness 必须以喂给 kernel 的同一张量推导 golden；若实测精度**显著优于**设计期误差预算，先怀疑设计探针 harness 输入/ golden 不同源而非「实现变好了」，并把差异记入 RETROSPECTIVE | R |
| Design-flow | DESIGN §5.5 分核表把 `[16,512,32,32]` 的 N 记为 16,777,216，实际 numel = 8,388,608（§1.6.0 roofline 表同错）；`design_calc_check` 对分核项 skip，未捕获 | DESIGN.md §5.5 行 `gan-feat-deep [16,512,32,32]`、§1.6.0 D-2 roofline 表；Stage 3 实测 `[L0-manifest(16, 512, 32, 32)] shape=(8388608,)` | 在 design skill 的表格自查中加入「4-D workload 的 N 必须等于 product(shape)」机械校验 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| LeakyReLU fp32-中转链（`vmul`+`vmax`，2 op，`0<s≤1`）对契约 golden `F.leaky_relu(x.float(), s).to(dtype)` 在 Ascend910_9362 上**逐位一致**：41 用例全 `max_diff=0.0`（fp32/fp16/bf16 各 1M randn、4 seed `torch.equal=True`、角点、小 x 敏感段、±inf/NaN、4e6 非整除尾块、8M/16M/67M 4-D manifest）；slope=2.0/100.0/0.0 回退路径同样 0.0 | `_make_leaky_relu_kernel.py` run（`--level all`，全 PASS）；`repro/TRAP-golden-input-cast-mismatch.py` 同源 golden `torch.equal=True` | `python examples/leaky_relu/_make_leaky_relu_kernel/_make_leaky_relu_kernel.py --level all` | tilelang 0.1.2+e6aa5a80 / Ascend910_9362 / CANN 9.0.0 | pattern-library/constants.md（CONST-leaky-relu-fp32-transit-bit-exact-910_9362）或 traps-runtime.md TRAP-fp16-opmath-golden 补证 |
| **负面/更正**：DESIGN 报的 fp16 randn 3.052e-5 / bf16 2.441e-4 非 kernel 误差，而是**设计探针 harness 的输入/golden 不同源伪误差**——golden 从「未舍入 fp32 randn」算、kernel 喂「half-rounded 张量」，二者差恰为一次输入 cast（CPU 上逐 seed 复现精确同值 3.052e-05 / 2.441e-04）；同源 harness 下 kernel 逐位一致。裁决规则：逐元素精度 harness 的 golden 必须从喂给 kernel 的**同一张量**推导 | `history_version/design_probe_leaky_relu.py` `_run` L151-157 + `_golden` L111-112；DESIGN.md §8.2；CPU 复现脚本 | `python examples/leaky_relu/_make_leaky_relu_kernel/repro/TRAP-golden-input-cast-mismatch.py`（91 行，实跑 PASS） | tilelang 0.1.2+e6aa5a80 / Ascend910_9362 / CANN 9.0.0 | pattern-library/traps-runtime.md（新增 TRAP-golden-input-cast-mismatch） |
| 编译期 guard 设备实证：`max(x,s*x)` 严格守卫 `0<s≤1` 在设备上成立——guard 内（s=0.5）与 guard 外回退（s=2.0 / 100.0）分别与 golden 逐位一致（`max_diff=0.0`）；`s=0.0` fp16 回退路径亦 0.0（覆盖 `-inf` 角点 `0·(-inf)=NaN`） | `_make_leaky_relu_kernel.py` run：`L0-slope2.0` / `L1-slope0.5` / `L1-slope100.0` / `L1-slope0.0` 均 `max_diff=0.000000e+00` | `python .../_make_leaky_relu_kernel.py --level all` | tilelang 0.1.2+e6aa5a80 / Ascend910_9362 / CANN 9.0.0 | algorithm-candidates.md ALG-elementwise（R1 候选行设备补证） |

### Transferable Lessons（可迁移教训）

- **逐元素精度 harness 的 golden 必须从喂给 kernel 的同一张量推导**。若 golden 从 pre-cast fp32 源张量算、而 kernel 收到的是 cast 后（fp16/bf16）张量，测出的 fp16 `3.05e-5` / bf16 `2.44e-4`（randn）是**一次输入 cast 差**，不是 kernel 误差——同源 harness 下该 kernel 逐位一致。迁移任务 Stage 3 若实测远优于 DESIGN 的误差预算，先按此复查设计探针 harness，而不是当作「实现更好」。
- 逐元素 parametric activation（`y = x if x>s0 else s*x` 族）的 NPU 迁移可直接照抄 ELU 模板结构一次通过：`T.alloc_ub` 顶层**无条件**分配全部 buffer（含回退路径专用 `z`/`cond`）、`T.Kernel(40, is_npu=True)` + `T.serial` grid-stride（静态上界 + `if task < n_num` 守卫 + `tail = T.min(bs, shape-off)` 只包在 `T.copy` 里）、v-op 作用于整 buffer、零常量循环外 `T.fill`、`vcast` 统一 `round_mode="rint"`。
- 含编译期标量参数的逐元素算子：把标量做成 Python 编译期常量并据其取值特化向量链形态（如 `0<s≤1` 用 2-op `vmul+vmax`，否则 3-op `vcmp+vmul+vselect`），可在不增运行时分支的前提下减 1 op；特化的等价域必须由编译期 guard 严格限定，guard 外必须回退到与契约等价的形态。
