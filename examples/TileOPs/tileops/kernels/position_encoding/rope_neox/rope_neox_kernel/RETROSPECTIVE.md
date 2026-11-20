# RETROSPECTIVE — examples/rope_neox/_make_rope_neox_1d/

## Stage 1 (design) Retrospective — first_design — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | `kb_stale_check` 将本任务消费的全部预注入 pattern-library 条目（含 `status: verified` 的 `ALG-elementwise`/`PL-1.22`/`PL-1.17`/`CONST-ascend910-9362-profile`）一律标 `[stale]`（当前 toolchain 与条目版本戳不同）——设计需逐条标注「背景/待重验」，增加人工核对成本 | `examples/rope_neox/_make_rope_neox_1d/DESIGN.md#1.6.0` 调研范围段 + §9 R-12 | 在 `pattern-library/INDEX.md` 为 `stale` 增加「可作背景、不作决策依据」的显式语义与引用模板，减少每次设计重复判断 | R |
| Constraint-doc | v-prefix 元算 dtype 矩阵在 `docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.1 标 bf16 ✗，但 `testing/npuir/bf16_support_ops/test_vv_bf16.py` 用 bf16 `T.vmul/T.vadd`（Developer 2D）——文档与测试相左，设计须标注「待确认」 | `examples/rope_neox/_make_rope_neox_1d/DESIGN.md#9.1` R-3；`docs/Tilelang.language/数学操作/T.vmul.md#2.2.1`；`testing/npuir/bf16_support_ops/test_vv_bf16.py` | 澄清 bf16 v-op 支持状态（文档更正或测试标注 xfail），避免迁移设计反复核对 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| none | - | - | - | - | - |

### Transferable Lessons（可迁移教训）

- **RoPE（neox/rotate_half）迁移到 NPU 的推荐计算核**：源「逐元素 flat 索引 + `col % half` + `if_then_else` 配对」含逐元素跨步 gather（NPU 无廉价 gather 向量指令）；改为**半段（half-split）结构** `a=x[:,:half]`、`b=x[:,half:]` → `y1=a*c−b*s`、`y2=b*c+a*s`（4 `vmul`+1 `vsub`+1 `vadd`，零 gather），且 cos/sin 按行块单次装载、两半复用（cos/sin GM 流量 2×→1×）。该结构与源公式**逐位等价**（`(−b)*s ≡ −(b*s)`、`t1−t2 ≡ t1+(−t2)` IEEE 精确）。
- **RoPE 契约 golden 是 fp32 opmath**（源仓 `ref_rope_neox`：`x.float()*cos_full.float() + rot.float()*sin_full.float()` 后单次舍回）——与源 GPU kernel（输入 dtype 域）**异域**，NPU 必须 fp32 中转（`vcast(rint)` 升 → 链 → `vcast(rint)` 舍回），否则 fp16/bf16 超容差（fp16 `max_abs=7.81e-3 > 1e-3`）。同族先例的「输入 dtype 域直译」**不可照搬**。
- **v-op 只作用于整 buffer，动态/静态切片只出现在 `T.copy`**（`TRAP-expert-v-operands`）；`(br, half)` 2D UB tile + `T.copy(x[r0:r0+tr, 0:half], a_ub[0:tr, 0:half])` 是尾块安全形态；`block_rows = min(block_size//head_dim, seq_len)` 防 `block_m > M` 假性 UB overflow。
- **物理 Vector 核数须实查**：`NPUUtils.get().get_aicore_num()` = 20、`get_aivector_core_num()` = 40（纯 Vector 翻倍），核数取值禁止用文档假设值；`seq_len=2ⁿ` 的算子无法与 40 = 2³×5 对齐，残留负载不均须显式记录（网格跨步 persistent）。
- **`gate_lint` 的 `S1-EQUIV-EXEC` 是字面量规则**：§1.6.1 正文出现 `EQUIV_FAIL` 字样（哪怕是「EQUIV_FAIL = 0」说明文字）即误判失败——撰写时须回避该字面量（同 `CASE-elementwise-erf-migration` / VP-2026-0143 记录）。

## Stage 2 (review) Retrospective — first_review — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `verify_equiv.py` 的等价性检查可能退化为**自比较**（`candidate_no_cse` 直接 `return candidate(...)`），机器结果仍 30/30 EQUIV_PASS，机械复核「与内嵌表一致」无法识别——维度 8 的「等价性机器验证有效」判定缺乏对**脚本候选/基线是否真实现分歧**的检查项 | `examples/rope_neox/_make_rope_neox_1d/verify_equiv.py#L69-L70`；`REVIEW.md#机械复核`/`#8` | 在本 skill 维度 8 增补检视要点：「逐采纳项核对 `verify_equiv.py` 中候选式与基线式是否为**不同实现**；凡 `return <other_fn>(...)` 式转发/恒等对照的项，须由 reviewer 独立推演并注明『机器验证为该式自比较，等价性以人工推演为准』」 | P |
| Info-source | `design_calc_check.py` 的 `core_split` 对纯 Vector 算子/二维网格形态返回 `skip`（`insufficient parsed dims`），分核三要素退回人工；本次人工可核（§5.5 齐全），但机械覆盖率不足以拦截分核类错误 | `design_calc_check.py` 输出 `core_split: skip (insufficient parsed dims {'M': 256})`；`REVIEW.md#机械复核` | 为 `design_calc_check.py` 增加「按 `seq_len/block_rows` 表述的逻辑核数」解析分支，或在其 skip 详情中显式提示「须人工核对 `core-split-strategy.md` §1 三要素」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| none | - | - | - | - | - |

### Transferable Lessons（可迁移教训）

- **迁移检视中「wrapper 契约对齐」要落到具体参数流**：DESIGN §3.3 声明与 wrapper 完全对齐，但工厂把 `block_rows` 绑在**工厂体参数**（默认 2048），而 wrapper 只把 dtype 相关 `block_size`（fp32=1024）传给**内层 callable**——fp32 路径的依赖默认值静默改变 tiling，文档内部（§3.3 vs §5.2/§4.5/§5.5）出现不一致。检视迁移设计时应核对「工厂签名 / 返回 callable 签名 / 参数在何层被消费」三者与 wrapper 的 `_build_kernel`+`forward` 调用链是否闭合，而不只看签名是否匹配。
- **验证方案的等价性脚本要防「自证」**：`verify_equiv.py` 中凡出现 `return <other_fn>(...)` 的转发式函数（如 `candidate_no_cse`），该对照行即为恒等的自比较，机器 PASS 不能作为该优化项的证据；必须由 reviewer 独立推演其等价性（本例 #3 cos/sin CSE 平凡等价，推演成立）。
- **迁移任务维度 0 的证据链**：源仓语义须回源到「源 GPU kernel + 源仓测试 golden（`ref_rope_neox`）+ Op 层 dtype 生成」三处——本例源 kernel 在输入 dtype 域、golden 在 fp32 域，二者**异域**是迁移的最关键判据（决定 fp32 中转为正确性要求）；仅读源 kernel 会漏判该域差。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 迁移设计的工厂伪代码把 tiling 参数 `block_rows` 绑在**工厂体参数**（默认 `block_size=2048`），而 wrapper 只把 dtype 相关 `block_size` 传给**内层编译 callable**——fp32 路径的声明 tiling（1024）被静默替换为默认值（2048）；REVIEW 已标建议级，Stage 3 通过把 `block_rows`/`n_num` 移入内层 `kernel(blk)` 修复 | `_make_rope_neox_1d.py#_make_rope_neox_1d`（内层 `block_rows = max(1, min(int(blk)//head_dim, seq_len))`）；`REVIEW.md#7` 建议级问题 1 | 在 design skill 的迁移「接口适配」检查中增补：凡工厂返回 `kernel(block_size)` 形态（K11），**全部 tiling 派生必须在被传入参数的那一层计算**；DESIGN 伪代码禁止用工厂默认值派生 `block_rows` | R |
| Info-source | DESIGN §9.1 R-3 要求「Stage 3 设备探针核对 bf16 v-op 文档/示例分歧」——探针结果（bf16 原生 v-op 设备可用）与 `docs/.../T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.1 的 bf16 ✗ 相左；文档未随设备能力更新，迁移设计每遇 bf16 都要重新核对 | `repro/TRAP-bf16-vop-doc-matrix-divergence.py`（ALL ASSERTIONS PASS）；`docs/Tilelang.language/数学操作/T.vmul.md#2.2.1` | 更新三份 v-op 文档 §2.2.1 的 bf16 列（改为 √ 并注明设备/版本），或加「文档矩阵为保守声明，以设备实测为准」脚注 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| bf16 v-prefix 数学 op（`T.vmul`/`vadd`/`vsub`）文档矩阵标 bf16 ✗，但 Ascend910_9362 上原生 bf16 形态编译+运行+与 torch 逐位一致（推翻设计期"按 doc 无 bf16"的默认假设） | D | `_make_rope_neox_1d.py`（本函数不依赖该路径）；`DESIGN.md#9.1` R-3；探针实跑 5/5 PASS | `repro/TRAP-bf16-vop-doc-matrix-divergence.py`（`python repro/…py` 直接运行，断言 [1][2][3]） | tilelang 0.1.2+a2357253662babf317ada194080078f928d155b3 / Ascend910_9362 / CANN 9.0.0 / torch 2.7.1 + torch_npu 2.7.1.post4 / Developer | `traps-compiler.md`（或 `constants.md` 的 dtype 矩阵条目） |
| 半段（half-split）RoPE 向量链 + fp32 中转在 NPU 设备上与契约 golden（fp32 opmath + 单次舍回）**逐位一致**（fp16/bf16/fp32 × 全 manifest shape + 行尾非整除 + ±inf/NaN 角点，全部 `max_diff=0.0`）——实证 verify_equiv 的 0-ulp 预言与 fp32 中转为正确性必需 | D | `_make_rope_neox_1d.py#run_L0/run_L1`（9+17+6+9 = 41 例 PASS，max_diff=0.0）；`verify_equiv.py`（30/30） | `python examples/rope_neox/_make_rope_neox_1d/_make_rope_neox_1d.py --level all`（NPU） | 同上 | `elementwise.md` / 新增 position-encoding 模式条目 |
| 「一维 persistent 核 + 核内 `T.serial` 网格跨步 + 2D UB tile + 动态行尾 `tr=T.min(...)` + 整 buffer v-op」在 **2D 行块 tile**（非 flat 1D）形态下的尾块安全模板（`block_rows=max(1,min(blk//head_dim,seq_len))` 防假性 UB overflow；行切片只出现于 `T.copy`） | P | `_make_rope_neox_1d.py#_make_rope_neox_1d`；L0-tail(2050)/L0-tiny(1)/L1-nonpow2(1000) PASS；先例 `examples/posembedding/rope_llama31/kernels/table_low_precision.py` | `python examples/rope_neox/_make_rope_neox_1d/_make_rope_neox_1d.py --level L0` | 同上 | `pattern-library/elementwise.md`（PL-1.22 的 2D-tile 尾块补充） |

### Transferable Lessons（可迁移教训）

- **K11 工厂返回 `kernel(block_size)` 时，所有 tiling 必须在被传入参数的这一层（jit 内层）计算**：`block_rows = max(1, min(int(blk)//head_dim, seq_len))`、`n_num`、`iters` 都要用 `blk` 派生；若写在工厂体外层用其**默认值**，wrapper 的 dtype 相关 `block_size`（fp32=1024）会被静默忽略，声明 tiling 与实际不符（数值仍可能正确，但性能/预算声明失真）。
- **NPU 上 bf16 原生 v-prefix 数学 op 是可用的**：`T.vmul`/`T.vadd`/`T.vsub` 的 bf16 在 Ascend910_9362 上编译、运行且与 torch 逐位一致，`docs/.../{T.vmul,T.vadd,T.vsub}.md` §2.2.1 的 bf16 ✗ 是**保守/过期声明**。因此「是否走 fp32 中转」应由**精度契约**决定（本例契约 golden 是 fp32 opmath，fp16 输入域 `max_abs=7.81e-3` 超容差），不能仅凭文档矩阵判定 API 缺失。
- **RoPE half-split 的尾块安全形态**：2D tile `(block_rows, half)`，`T.copy(x[r0:r0+tr, 0:half], a_ub[0:tr, 0:half])` 动态行切片只出现在 `T.copy`；v-op 一律作用于整 buffer；`tr = T.min(block_rows, seq_len - r0)`。`head_dim` 拆半的两半都取自同一 tile，故行块边界 = 自然的分核边界，无需跨 tile 拼接。
- **cos/sin 单次装载 + 两半复用**：`c_ub/s_ub` 各按行块装载一次，`y1`/`y2` 共用，cos/sin GM 流量减半；golden 用 `cat([cos,cos],-1)` 的全宽展开式实现，二者逐位一致（复用不改变数值），故不必为 CSE 单独写对照实现（`verify_equiv.py` 的自比较项不影响结论）。
