# GeluTanhAndMulFwdKernel — RETROSPECTIVE

## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source / Constraint-doc | `docs/Tilelang.language/数学操作/T.vtanh.md` §3 claims the Developer-mode lowering is `tanh(x)=(exp(2x)-1)/(exp(2x)+1)` (a faithful exp identity); on this toolchain `tl.npuir_vtanh` is mode-dispatched and NEITHER path is faithful over the operator's range: default/Expert = truncated Maclaurin (|x|≲0.5), Developer = exp identity but `+|x|>~44`/`+inf` overflow to NaN. Doc-vs-toolchain semantic gap of the same class as `T.verf` (CG-2026-0021). | `repro/TRAP-vtanh-truncated-taylor.py` (default + Developer assertions); `history_version/design_probe_vtanh_truthfulness.py`; codegen `tilelang-mlir-ascend/src/target/codegen_npuir_api.cc` / `codegen_npuir_dev.cc` | `docs/Tilelang.language/数学操作/T.vtanh.md`: document the actual active lowering(s), the valid domain of each, and the `TILELANG_ASCEND_MODE` dispatch | C |
| Info-source / Info-source | The toolchain kernel compile cache does not appear to key on `TILELANG_ASCEND_MODE` — an early default-mode compile of `T.vtanh` (truncated) was silently reused for a later Developer-mode run (and vice-versa), producing contradictory probe results until distinct block sizes isolated the cache entries. Mode-dependent codegen must be probed with cache-busting. | `history_version/design_probe_vtanh_truthfulness.py` docstring; `repro/TRAP-vtanh-truncated-taylor.py` (`blk = 512 if default else 544`) | tilelang docs / debug guide: document the on-disk jit cache key and whether it includes `TILELANG_ASCEND_MODE` | P |
| Info-source | Negative-claim discipline worked as intended: the pre-injected `CG-2026-0021` (T.verf truncation) prompted an *analogous* hardware check for `T.vtanh` before design — the check found the same trap. | DESIGN.md §0.6 R2 / §9.1 | keep pre-injecting same-toolchain transcendental traps for activation-family migrations | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir `T.vtanh` is mode-dispatched and neither path is a faithful tanh: default/Expert = truncated Maclaurin `x − x³/3 + 2x⁵/15 − 17x⁷/315` (valid only `\|x\|≲0.5`); Developer = exp identity `(e^{2x}−1)/(e^{2x}+1)` (finite `\|x\|<~44`, else overflow → NaN; `+inf→NaN`) | D | `examples/gelu_tanh_and_mul/gelu_tanh_and_mul_fwd_kernel/repro/TRAP-vtanh-truncated-taylor.py`; `history_version/design_probe_vtanh_truthfulness.py`; DESIGN.md §0.6 R2 / §9.1 | `python repro/TRAP-vtanh-truncated-taylor.py [default\|Developer]` (NPU) | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3 | `.agents/evolution/capability-gaps.md` (extend CG-2026-0021 family) + pattern-library traps |
| `T.vcast(fp32→fp32)` corrupts data on this toolchain (output thresholded); same-dtype fp32 paths must use `T.copy` | D | `history_version/design_probe_sigmoid_gelu_mechanism.py` (broken fp32 sigmoid before the fix: g=1.5→2.0, g=0.001→0.0; correct after `T.copy`); DESIGN.md §9.2 | `python history_version/design_probe_sigmoid_gelu_mechanism.py` (NPU; compare vcast vs copy on fp32) | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3 | pattern-library traps + `.agents/evolution/capability-gaps.md` |
| GELU-tanh as a single well-conditioned sigmoid via `1+tanh(u)=2σ(2u)`: `gelu_tanh(g)=g·σ(2u)`, σ explicit fp32 `vexp/vdiv`; robust saturation (no `∞/∞`), NaN/-0.0 preserved, 2 fewer ops than unfold | P | `examples/gelu_tanh_and_mul/gelu_tanh_and_mul_fwd_kernel/verify_equiv.py`; `history_version/design_probe_sigmoid_gelu_mechanism.py`; DESIGN.md §0.6 R2 / §1.6.1 | `python verify_equiv.py` (CPU) ; `python history_version/design_probe_sigmoid_gelu_mechanism.py` (NPU) | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3 | pattern-library elementwise + algorithm-candidates |

### Transferable Lessons（可迁移教训）

- **原生超越函数 API 在本工具链一律需做域外数值实测，不能只看文档/签名**：`T.verf`（CG-2026-0021）与 `T.vtanh`（本任务）都被文档描述为忠实函数，实测均非全范围忠实——`T.verf` 是低阶截断级数（`\|x\|≲1`），`T.vtanh` 按 `TILELANG_ASCEND_MODE` 分派：默认/Expert 为截断级数、Developer 为 exp 恒等式（有限 `\|x\|` 忠实但 `+inf`/溢出失败）；`T.tanh` 只是 TIR 标量表达式，无向量形态。激活族迁移在定 API 前先跑「域外 + `±inf`/NaN」小探针，且**按模式分派时必须 cache-bust**（见下）。
- **tanh 类激活优先用恒等式把 tanh 消去**：`1+tanh(u)=2σ(2u)`，于是 tanh-GELU 变成单次 sigmoid `g·σ(2u)`——既绕开不可用的原生 tanh，又更省 op 且避开近 0 相消；用 `1/(1+e^{−2u})`（而非 `(e^{2u}−1)/(e^{2u}+1)`）才能在大参数处自饱和而不产生 `∞/∞=NaN`。
- **fp32→fp32 禁用 `T.vcast`**：同 dtype 走 `T.copy`；`T.vcast` 只用于 fp16/bf16↔fp32。fp32 输入/输出路径照搬窄 dtype 的 vcast 会静默产出错误数值。
- **`vdiv` 只允许第二操作数为标量**：被除数须是 UB 张量，用 `T.vbrc` 填一个 ones buffer（本任务 sigmoid 的 `1/den` 即此形态）。
