# RETROSPECTIVE — AddFwdOp（`add_fwd_kernel`，migration-harness）

## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | gate `S1-EQUIV-EXEC` 以裸子串判定 `EQUIV_FAIL`，否定式表述（如「无 `EQUIV_FAIL` 项」）会被误判为「存在未通过项」而 fail——设计者在 §1.6.1 表达「全部通过」时天然想写这个词 | DESIGN.md §1.6.1（初稿含「无 `EQUIV_FAIL` 项」，gate 报 S1-EQUIV-EXEC）；`.agents/tools/gate_lint.py` L320-328 | gate_lint 的 EQUIV_FAIL 判定改为「存在 `EQUIV_FAIL` 作为结论值而非否定语境」，或在 design 模板注明「禁止在 §1.6.1 写出 `EQUIV_FAIL` 字面量（含否定）」 | R |
| Design-flow | gate `S1-REF-PATHS` 的 `REF_PATH_RE` 会把英文斜杠列举（如 prose 里的 `docs/examples 佐证`）误识别为仓库路径 `docs/examples` 并报「路径不存在」——真实意图是「docs/ 或 examples/」 | DESIGN.md §1.6.0 R4（初稿「附 docs/examples 佐证」，gate 报「引用的仓库路径不存在：docs/examples」）；`.agents/tools/gate_lint.py` L43-45,L65-85 | REF_PATH_RE 增加「相邻段须为已知目录/文件名」约束，或在 design 自检清单注明「prose 中勿写 `docs/examples` 连写，写 `docs/` 与 `examples/`」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| NPU elementwise binary 的广播契约：GPU kernel 内 stride 计划（`compute_broadcast_offsets`）→ NPU Op 层 `torch.broadcast_shapes`+`expand`+`flatten`（kernel 只见 same-shape 平铺），依据 = NPU 缺跨步 gather 向量指令 + lerp_tensor E15 先例 | P | examples/TileOPs/tileops/ops/elementwise/add.py（`_expand_flat`）；examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel_DESIGN.md#0.3（预广播 Op 层契约）；examples/TileOPs/docs/gpu_to_npu_adaptation.md §3.4 E15；pattern-library/traps-compiler.md `TRAP-C10-strided-gather` | none | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C（实查 20 AI Core） | pattern-library/cases.md 或 elementwise.md |
| `torch.add(scalar alpha)` 与两轮 fp32 链（vmul→vadd）**非逐位一致**——torch 侧对部分幅值用单次舍入（fma/宽域），两轮 fp32 与其差最大 0.0128 tolerance-unit（<1，0 违反）；`alpha` 缩放类迁移的 golden 对照须用容差、不可假设位相等 | C | examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel_DESIGN.md §1.6.1（EQUIV 表 + 对照项）；examples/add/add_fwd_kernel/verify_equiv.py | `python3 examples/add/add_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C / 2026-10-13 | pattern-library/traps-runtime.md（数值口径类） |

### Transferable Lessons（可迁移教训）

- 迁移带 numpy 广播的二元逐元素算子到 npuir：**广播在 Op 层解析**（`broadcast_shapes` → `expand` → `flatten` 为 same-shape 1-D），kernel 只写平铺逐元素链——npuir 缺跨步 gather 向量指令，把广播索引放进 kernel 会退化为逐元素标量 gather。代价是广播操作数一次设备端物化流量。
- bf16 在所有 v-prefix 算术 API（vadd/vmul/vclamp/…）的官方 dtype 表均为 ×，**任何 fp16/bf16 逐元素迁移都应走 fp32 中转**（`T.vcast(rint)` 双端），这同时对齐 torch fp32 opmath golden。
- **只折叠 `alpha==1`，绝不折叠 `alpha==0`**：`0*b` 在 `b=±inf` 时产生 NaN，折叠为 `y=a` 会改变 IEEE 语义；`1*b==b` 对全部可表示值逐位成立，才是安全折叠。
- `T.vadd`/`T.vmul` 支持 `dst` 与 `src` 同 buffer 的原地操作，可用同一 UB buffer 承载中转/结果以压 UB 预算。
- DESIGN 散文里避免两类 gate 误伤：① 勿连写英文目录斜杠列举（`docs/examples`）——会被路径存在性检查当作路径；② 勿在 §1.6.1 写出 `EQUIV_FAIL` 字面量（含「无 `EQUIV_FAIL`」这类否定），改写「全部通过 / 无未通过项」。

## Stage 3 (implement) Retrospective — first_impl — 2026-10-13

编程模式：Developer；产物 `examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py`（`ASCEND_MODE = "Developer"`）；`python add_fwd_kernel.py --level all` exit 0，L0/L1 全过（L2/Boundary 亦全过，无 warn）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Implement-flow | harness prompt 把 `add_fwd_kernel` 的 JIT 装饰器写成 `out_idx=[1]`，但二元加法（两个输入 a、b + 输出 y）正确索引是 `out_idx=[2]`，且 NPU wrapper `AddFwdKernel.forward` 以 `add_fwd_kernel(...)(a, b, N_total)` 三参调用（`out_idx=[1]` 下非输出参只有 2 个，调用必然失配）；frozen DESIGN §3.3/§4.7 亦写 `out_idx=[2]`。`out_idx=[1]` 疑为从 clamp（一元）harness 模板复制 | add_fwd_kernel.py `@tilelang.jit(out_idx=[2], ...)`；examples/TileOPs/tileops/kernels/elementwise/add/add.py L156-161；DESIGN.md §4.7 | harness 生成 prompt 时按算子输入元数推导 `out_idx`（n 输入 → `[n]`），勿套用一元模板；或在 prompt 中显式标注「以 DESIGN 为准」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir 二元逐元素 `alpha` 特化落地形态：`alpha` 作为 factory Python 标量，`alpha==1` 走 trace-time 折叠（仅 `vadd`）、否则 `vmul(b32, float(alpha), b32)` + `vadd`（原地链）；`alpha` 用原生 Python float（二元 op 前端只接受裸标量，不接受 cast 表达式）；fp16/bf16 恒经 fp32 中转。实测 f16/bf16 全部 dtype 与 alpha∈{1,2.0,−1.5,0,−1,0.5,3} max_diff=0（bit-exact），f32 仅 9.5e-7（<1e-5 tol） | D | add_fwd_kernel.py（kernel + L0/L1/Boundary 输出）；DESIGN.md §3.2/§1.6.1 | `python add_fwd_kernel.py --level all` | tilelang 0.1.2+417dae99 dev-root + CANN 8.5.0 + Ascend910B3（实查 20 AI Core → 40） | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- npuir 二元 elementwise 迁移的 factory 直接用 `@tilelang.jit(out_idx=[2], target="npuir")` 装饰（本算子输入数=2 → 输出索引=2），`num_kernels=min(ceil(N/block_size), aicore×2)` 持久化分核——与一元（clamp，`out_idx=[1]`）唯一差别就是输出索引；照抄一元模板会错。
- `alpha` 标量传 `T.vmul` 时必须用原生 Python `float`（如 `float(alpha)`），**不可**传 `T.cast(alpha,"float32")` 表达式——NPU 二元 op 前端只接受裸标量 operand（clamp 已记录同型约束）。`alpha==1` 折叠与 `alpha==0` 不折叠分别用实测确认：inf/NaN 用例下 alpha==1 逐位成立、alpha==0 保留 `0*inf=NaN`（nan_out>0）。
- 广播测试的独立性技巧：golden 在**原始 shape** 上用 `torch.add` 原生广播；kernel 侧在测试 harness 里手工 `expand(out_shape).contiguous().view(-1)` 复现 Op 层契约，再 `view(out_shape)` 回塑——两层路径解耦，可验证 Op 层展平与 kernel 平铺链的端到端等价。
