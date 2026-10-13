# RETROSPECTIVE.md — hardswish_fwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 同族迁移设计存在「照抄 Hardsigmoid 的 NaN 单点恢复链」的仿写风险：Hardsigmoid 因输出无 `x *` 因子、普通 min/max 丢 NaN 而必须用专用 NaN 链，Hardswish 输出含 `x *` 因子、NaN 由乘法天然传播，源码 `op_func` 刻意用普通 `T.min/T.max`。若设计期不逐 op 读源码而按同族先例仿写，会给 Hardswish 强加 2–3 个冗余向量 op | DESIGN.md §0.1（源码依据：「Hardsigmoid 无 `x*` 因子…hardswish 含 `x*` 因子…源码刻意选用普通 `T.min/T.max`」）、§0.5（NaN 免处理 clamp 判「保留」）、§0.6（无 NaN 重设计项）、§11.1（与 Hardsigmoid R2 明确区分）；GPU `activations.py` 两者 `op_func` 注释差异 | tilelang-op-design skill 的 `references/migration-analysis.md` §4 增补：迁移「带 NaN 语义的 elementwise 激活族」时，须逐 op 判定「输出是否含 `x*` 因子」——含则该因子天然传播 NaN，不可照搬同族其它算子的专用 NaN 链 | P |
| Info-source | GPU `make_unary_regcopy` 的 `contiguous`（窄化逐线程连续 run）分支对 Hardswish **不触发**（输出 dtype 与输入相同 → `_narrows` 为假），源码实际恒走 fragment 全块路径；若只读 builder 函数名/注释易误判 GPU 用了「逐线程 vectorized run」并据此设计 | DESIGN.md §0.3（`_narrows(dtype,out_dtype)` 为假 → fragment 路径）；GPU `_builders.py` `make_unary_regcopy` 的 `contiguous = _narrows(...) and N % block_size == 0` | `references/migration-analysis.md` §3 源码结构化阅读协议增补：读 GPU builder 时须先判定 `if` 分支的 dtype/shape 谓词在本算子实际输入下是否成立，勿把未触发的分支当作实际算法 | P |
| Info-source | 物理核数 `NPUUtils.get().get_aicore_num()` 本环境实查返回 20（纯 Vector ×2 = 40），与 `constants.md` `CONST-aicore-910B2C` 记录的 24 不一致；同一环境 hardsigmoid 任务已记录同现象，说明非偶发 | DESIGN.md §5.5（实查 20 ×2 = 40；`CONST-aicore-910B2C` 记 24） | `constants.md` `CONST-aicore-910B2C` 拆分设备档位或增补「本工作区设备实查 20」的稳定注释（hardsigmoid + hardswish 两任务复现） | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| hardswish 型（输出含 `x * clamp(...)` 因子）逐元素激活迁移**无需** NaN 专用链：普通 `vclamp`（vmax/vmin 丢 NaN）+ 末尾 `x*clamped` 乘法即天然传播 NaN/±inf，与 `F.hardswish` 一致——与 hardsigmoid 型（无 `x*` 因子、须专用 NaN 链）的关键区分 | C | DESIGN.md §0.1/§0.5/§0.6/§11.1；GPU `activations.py`（`HardswishFwdKernel` 用 `T.min/T.max`，`HardsigmoidFwdKernel` 用 `nan_min/nan_max`） | none（设计判定；行为由 §8.1 golden `F.hardswish` 覆盖） | tilelang 0.1.2 dev-root（/home/developer/tilelang-mlir-ascend/build）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（`CASE-elementwise-hardsigmoid-migration` 的 known_impl 增量） |
| 除法转乘倒数在 hardswish 上机器验证 EQUIV_PASS：`x*clamp(x+3,0,6)/6` → `x*clamp(x+3,0,6)*(1/6)`，fp64 参照 max 1 ulp / 0 违反，fp32/fp16/bf16 三舍入路径 0 违反 | P | DESIGN.md §1.6.1（结果表）；`verify_equiv.py` | `python examples/hardswish/hardswish_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（同族乘倒数先例） |

### Transferable Lessons（可迁移教训）

- 迁移 hardsigmoid/hardswish/hardtanh/relu6 族的**下一个函数**前，先判定「输出表达式是否含 `x *` 因子」：含（如 hardswish）则普通 `vclamp` + 乘法传播即可，**不要**照抄 hardsigmoid 的 `vcmp(x,x,"ne")`+`vbrc`+`vselect` NaN 链（会多 2–3 个冗余向量 op）；不含（如 hardsigmoid/hardtanh 的纯 clamp）才需要专用 NaN 恢复。判据来自 GPU 源码两者 `op_func` 分别用 `T.min/T.max` 与 `nan_min/nan_max`。
- 读 GPU `make_unary_regcopy` / `make_*` builder 时，必须核对分支谓词（`_narrows`、`N % block_size == 0`、`stage` 等）在本算子实际 dtype/shape 下是否成立；未触发的分支不代表源算法的实际路径（hardswish 输出同 dtype → `contiguous` 分支恒不触发，恒走 fragment 全块路径）。
- 物理核数一律以本环境 `NPUUtils.get().get_aicore_num()` 实查为准（本工作区返回 20，非 `CONST-aicore-910B2C` 的 24），纯 Vector 算子再 ×2；设计 §5.5 必须记录查询代码与实际返回值，勿照抄知识库数值。
- `T.copy` 尾块用「双显式 slice」形态（`src[0:tail]`/`dst[0:tail]`）而非 base+size 标量基址形态；`T.Tensor((N,), dtype)` 用位置参数传 dtype 避免 `name 'in_dtype' is not defined` false alarm。

## Stage 3 (develop) Retrospective — first_impl（migration-harness, Developer）— 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Test-authoring | Boundary NaN 探针沿用了 hardsigmoid/clamp 的「输入 NaN 掩码 == 输出 NaN 掩码」断言，对 hardswish 型算子会**假失败**：hardswish 除 `NaN→NaN` 外还有 `-inf→NaN`（`-inf * 0`），故输出 NaN 数 = 输入 NaN 数 + 输入 `-inf` 数。首轮 `--level all` 报 `FAIL NaN mask mismatch: input 512 vs output 1024`，而 `assert_close(equal_nan=True)` 已通过——即内核精度正确、仅断言写错 | `hardswish_fwd_kernel.py` run_boundary NaN 探针首轮输出（input 512 / output 1024）；DESIGN.md §0.1（`-inf→NaN`）；修正为与 golden NaN 掩码比较后全 PASS | tilelang-op-develop skill 的测试模板/`tilelang-op-optimize` traps：elementwise 边界探针断言 NaN 掩码时，须与 **golden 的 NaN 掩码**比较（`torch.equal(isnan(y), isnan(golden))`）+ 「输入每一个 NaN 位置输出亦为 NaN」，不得直接与输入 NaN 掩码比较（对 `-inf→NaN`/`inf-inf` 型算子会误报） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| hardswish 型（输出含 `x *` 因子）NPU 实现**实测确认无需 NaN 专用链**：普通 `vclamp`（vmax/vmin 丢 NaN）+ 末尾 `x*clamped` 乘法在真实 NPU 上得到 `NaN→NaN / +inf→+inf / -inf→NaN`，与 `F.hardswish` golden（`equal_nan=True`）在 fp32/fp16/bf16 三 dtype 全部吻合（`nan_in=512, nan_out=1024, golden=1024`，1024 = 512 NaN + 512 个 `-inf`） | D | `hardswish_fwd_kernel.py` `--level all` Boundary 输出；DESIGN.md §0.1/§0.6；GPU `activations.py`（`HardswishFwdKernel` 用 `T.min/T.max`） | none（行为由嵌入 Boundary 探针覆盖；无需单独 repro 文件） | tilelang 0.1.2 dev-root（/home/developer/tilelang-mlir-ascend/build）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（`CASE-elementwise-hardsigmoid-migration` known_impl 增量：hardswish 变体无 NaN 链，实测 PASS） |
| 同族 elementwise 迁移 kernel 结构可直接复用（hardsigmoid→hardswish）：仅删去 NaN 恢复 op（`vbrc`+`vcmp`+`vselect` + `nan_mask` buffer），并把计算链从 `vadd→vclamp→vmul` 改为 `vadd→vclamp→vmul(x*clamped)→vmul(×1/6)`；persistent 分核/尾块/fp32 中转 `T.copy` 形态逐字可复用，首轮即三层全 PASS | P | `hardswish_fwd_kernel.py` 与 `hardsigmoid_fwd_kernel.py` diff；本轮 L0/L1/L2/Boundary 全 PASS | none | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- 逐元素激活迁移的 Boundary NaN 探针断言须以 **golden NaN 掩码**为基准（`torch.equal(isnan(y), isnan(golden))`）而非输入 NaN 掩码：凡输出表达式含 `x *` 因子者，`-inf` 也会经 `-inf * 0` 变成输出 NaN，使得输出 NaN 数多于输入 NaN 数（hardswish 实测 input 512 → output 1024）。
- 迁移同族 elementwise（hardsigmoid/hardswish/hardtanh/relu6）时，最省事且正确的路径是克隆已通过的同族 `{op}.py` 骨架（persistent 分核 + `T.serial` + 尾块双显式 slice `T.copy` + fp32 `vcast` 中转 + `_run_case`/`_max_diff_metric`/`main`），只替换计算链与 golden：hardswish 相对 hardsigmoid 只需**删** NaN 链（`vbrc/vcmp/vselect` + `nan_mask`）并在 clamp 后**加** `T.vmul(x_f32_ub, buf2_ub, buf1_ub)`。
- `T.vclamp(src, dst, min, max)` 在真实 NPU 上对 fp32 标量边界（`T.cast(0.0/6.0,"float32")`）工作正常；`T.vadd(x_f32, three, buf1)` 标量在 src1、`T.vmul(buf1, one_sixth, buf2)` 标量在 src2 的实参位置与 docs 一致，实测无坑。
