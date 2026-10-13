# RETROSPECTIVE.md — bitwise_not_fwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 整数 / 布尔逐元素迁移族（bitwise 类）与浮点逐元素族的设计套路**正交**：浮点族（hardsigmoid/hardswish）核心是 fp32 中转 + NaN 链；整数族核心是「字节存储 + 原生向量位运算」，二者不可互相仿写。若照抄浮点族模板会给整型算子强加无意义的 `vcast`/fp32 buffer/浮点容差 | DESIGN.md §0.1（无 fp32 中转、bit-exact）、§1.6.1（整数 dtype 契约）、§3.2（无 vcast/vbrc/vcmp/vselect）、§8.2（atol=rtol=0） | tilelang-op-design skill 的 `references/migration-analysis.md` / `algorithm-research.md` 增补：设计 elementwise 迁移前须先按 dtype 域分族（浮点族 → fp32 中转 + NaN 链；整数/布尔族 → 原生位运算 + 字节存储），族间模板不可混用 | P |
| Info-source | NPU bool（i1）不可 load/store GM，须以字节存储传递；GPU 源亦用 `BOOL_STORAGE_DTYPE = int8` 字节存储，但若只读 GPU `op_func`（`x == 0`）易误以为可直接对 bool 张量调 kernel | DESIGN.md §0.5（bool 字节存储等价替换）、§3.3（uint8 字节存储）、§11.4；`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3 / `T.vxor.md` §2.3 | `references/migration-analysis.md` §3 阅读协议增补：含 bool 输入的迁移须核对「bool 能否 load/store GM」——不能则须在 host/Op 层做 uint8 字节视图往返，kernel 内不得出现 i1 GM 访问 | P |
| Tooling | `constants.md` `CONST-aicore-910B2C` 记 24 核，本环境 `NPUUtils.get().get_aicore_num()` 实查返回 20（hardsigmoid/hardswish/bitwise_not 三任务复现）；纯 Vector ×2 = 40 | DESIGN.md §5.5（实查 20 ×2 = 40） | `constants.md` `CONST-aicore-910B2C` 增补「本工作区设备实查 20（三任务复现）」稳定注释或拆分设备档位 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 整型 bitwise-NOT 迁移的等价化简（bit-exact 机器验证）：整型 `xor(x, all-ones)` → 原生 `T.vnot(x)`（uint8/int8/int16/int32/int64）；bool 字节域 `(x == 0)` → `(x ^ 1)`（uint8 vxor）——两式均 bit-exact、mismatches=0，且各减一次操作数/掩码物化 | P | DESIGN.md §1.6.1（优化结论 + 结果表）；`verify_equiv.py` | `python examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root（/home/developer/tilelang-mlir-ascend/build）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（整数 elementwise 迁移先例，与浮点族 fp32 中转先例并列） |
| 整数 / 布尔 elementwise 迁移**无需** fp32 中转与浮点容差：计算路径 = 直接在整型 dtype 上 `vnot`（`VNotOp` DataType 表覆盖 uint8/int8/int16/int32/int64）；bool 走 uint8 字节域 `vxor`——与浮点族（hardsigmoid/hardswish 的 fp32 中转 + NaN 链）是关键分族依据 | C | DESIGN.md §0.1/§1.6.1/§11.3；`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1 | none（设计判定；行为由 §8.1 golden `torch.bitwise_not` 精确比对覆盖） | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- 迁移 elementwise 前先按 **dtype 域分族**：浮点族（fp16/bf16/fp32）走 fp32 中转 + NaN 链；整数/布尔族走**原生位运算 + 字节存储**，**无 fp32 中转、无浮点容差**（atol=rtol=0）。两类模板不可互相仿写。
- bool（i1）在 NPU 上不可 load/store GM（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3），必须经 `uint8` 字节存储；Op 层 `view(uint8)` / `view(bool)` 是元数据视图（内容不变）。bool 的逐元素逻辑 NOT 用字节域 `x ^ 1`（单 `vxor`），**不可**对它用 `vnot`（会得 255/254 而非 0/1），也不可与 uint8 的 `vnot`（按位取反）混用。
- 设计期 roofline 的 MTE2 带宽曲线（`CONST-mte2-degradation`）绑定 fp16 / 48 vector core；本设计为整型 / 40 向量核，档位按「总流量档」近似引用属未实测外推，须在估算行显式标注；L2 驻留 workload（smoke-1m 复用同输入）会高估时延。
- `T.copy` 尾块用「双显式 slice」形态（`src[0:tail]`/`dst[0:tail]`）；`T.Tensor((N,), storage_dtype)` 用位置参数传 dtype 避免 `name 'in_dtype' is not defined` false alarm；int64 用较小 block_size（4096）约束 UB 峰值。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | DESIGN §3.3 伪代码用 `T.vxor(x_ub, 1, y_ub)` / `T.cast(1,"uint8")` 标量操作数；npuir 代码生成把标量降为 `i8` 标量操作数，`hivm.hir.vxor` 校验拒绝（要求两个 source 均为 shaped tensor）。REVIEW §1 已标「建议级口径不一致」，实测实为**阻塞级 lowering 失败** | `bitwise_not_fwd_kernel.py`（bool 路径 `T.fill`+`T.vxor` 注释）；编译报错 "operand #1 must be variadic of shaped of any type values, but got 'i8'"；`repro/TRAP-npuir-vxor-scalar-operand.py` | `docs/Tilelang.language/逻辑操作/T.vxor.md` §2.1 修文案：Python 标量 src1 在当前 codegen 不被 promote/broadcast（vxor 需 shaped 操作数）；示例改 `T.fill` 物化常量 tile | P |
| Develop-flow | DESIGN §3.3/§0.5 指定 bool 用 `uint8` 字节存储；但 npuir 运行时 `JitKernel_NPU._tvm_dtype_to_torch` 缺 `uint8` 映射（fallback float32），声明 uint8 的 `out_idx` auto output 被错配为 float32 → 结果损坏。设计中 uint8 dtype 的「native vnot」路径在本工具链不可直接落地 | `tilelang/jit/jit_npu.py` L1696-1708（mapping 无 uint8）；DESIGN §3.3 `storage_dtype="uint8"`；`repro/TRAP-npuir-uint8-output-dtype-map.py` | pattern-library/traps-runtime.md 新增 TRAP：npuir auto output dtype 映射缺 uint8 → 字节域算子用 int8 存储 + host view | D |
| Develop-flow | npuir 核内「条件性 `T.alloc_ub`」（`if is_bool: c_ub = T.alloc_ub(...)`）触发 TVM script parser 报 `Undefined variable: c_ub`；分支专用 buffer 须无条件分配 | `bitwise_not_fwd_kernel.py`（c_ub 无条件 alloc 注释）；首轮探针报错 `error: Undefined variable: c_ub` | 开发指南/陷阱库补记：npuir 条件分配 buffer 不被 parser 支持，分支 buffer 一律无条件分配 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir `T.vxor` 标量 src1 不可 lowering（要求 shaped 操作数）：`T.vxor(x_ub, 1, y_ub)` → MLIR 校验 `operand #1 ... but got 'i8'`；绕法 = `T.fill(c_ub, T.cast(1,"int8"))` 物化常量 tile 后 `T.vxor(x_ub, c_ub, y_ub)`，对 bool 字节 {0,1} 与 `torch.bitwise_not` bit-exact | D | `bitwise_not_fwd_kernel.py`（bool 路径 `T.fill`+`T.vxor`）；DESIGN §3.3 原式 | `python examples/bitwise_not/bitwise_not_fwd_kernel/repro/TRAP-npuir-vxor-scalar-operand.py` | tilelang 0.1.2 dev-root（/home/developer/tilelang-mlir-ascend/build）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/traps-runtime.md |
| npuir auto output dtype 映射缺 `uint8`：`JitKernel_NPU._tvm_dtype_to_torch` fallback `torch.float32` → 声明 uint8 的 `out_idx` 输出被错配、数值损坏；uint8 逐元素算子须改用 **int8 字节存储** + host `view`（bit-identical，vnot/vxor 上等价） | D | `tilelang/jit/jit_npu.py` L1696-1708；`bitwise_not_fwd_kernel.py` `_to_storage`/`_from_storage` | `python examples/bitwise_not/bitwise_not_fwd_kernel/repro/TRAP-npuir-uint8-output-dtype-map.py` | 同上 | pattern-library/traps-runtime.md |
| int8 字节存储承载 bool/uint8（与 GPU 源 `BOOL_STORAGE_DTYPE=int8` 同源）：bool 走字节域 `x ^ 1`、uint8 走 `vnot`，二者在 int8 存储上均 bit-exact；bool 常量操作数经 `T.fill` 物化 | C | `bitwise_not_fwd_kernel.py`（storage_dtype 选择 + compute 分支） | `examples/bitwise_not/bitwise_not_fwd_kernel/repro/TRAP-npuir-uint8-output-dtype-map.py` | 同上 | pattern-library/elementwise.md（整数/布尔族字节存储先例） |

### Transferable Lessons（可迁移教训）

- npuir 上 `T.vxor`/`T.vand`/`T.vor` 等按位运算的第二个操作数**必须是 shaped tensor**；标量 src1（Python int 或 `T.cast` 标量）会生成 `i8` 标量操作数被 MLIR 校验拒绝（`docs/.../T.vxor.md` §2.1 的「标量广播」文案与实现不符）。含常量的按位运算先用 `T.fill(c_ub, T.cast(v, dtype))` 物化常量 tile 再运算。
- 声明为 `uint8` 的 npuir tensor 经 `@tilelang.jit(out_idx=...)` 自动分配的输出会被运行时错配为 `torch.float32`（`_tvm_dtype_to_torch` 无 uint8 键），结果损坏；字节域算子统一用 **int8 字节存储**（bool/uint8 在 host `view(torch.int8)` 进、`view(bool/uint8)` 出；int8 在 vnot/vxor 上与 uint8 bit-identical）。
- npuir 核内不允许「条件性 `T.alloc_ub`」（`if <compile-time bool>: buf = T.alloc_ub(...)` 触发 TVM parser `Undefined variable`）；分支所需的 buffer 必须无条件分配，未用到的分支仅占少量 UB。
- 整数/布尔逐元素算子的分核持久化结构（`T.Kernel(num_kernels, is_npu=True)` + `T.serial(num_local_tasks)` + `T.min` 尾块 guard）与浮点族完全同构，可直接复用；区别只在 compute 是 `vnot`/`vxor` 而非 fp32 中转链。
