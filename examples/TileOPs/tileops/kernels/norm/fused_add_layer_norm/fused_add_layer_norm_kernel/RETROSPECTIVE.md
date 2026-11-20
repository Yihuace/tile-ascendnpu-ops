# RETROSPECTIVE.md — _fused_add_layer_norm_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-14

### 结论

`FusedAddLayerNormFwdOp`（`_fused_add_layer_norm_kernel`，融合 residual add + LayerNorm，**双输出** `y` / `residual_out`）迁移设计完成，复用 norm 族 row-reduction 模板「全 N 驻留 UB + M 行块 persistent 分核（num_kernels=min(logical,40)，实查 20 AICore ×2）+ 尾块 `T.min` + src/dst 双显式 slice + 垃圾行丢弃」。相对 ada 单输出迁移的三个新增设计点：(1) **双输出契约**——`residual_out = round_dtype(x+residual)`、`out_idx=[4,5]`，两输出共用 `stage`（时序不重叠）；(2) **列因子（weight/bias (N,)）无文档化直连广播**——改用 `T.vbrc` 核内展开（`bm==1` 同形直接 vmul/vadd，规避空广播陷阱）；(3) **舍回后统计**对齐 golden（`added=(x.float()+r.float()).to(dtype)` 后再 LN）。等价性由 `verify_equiv.py` 机器验证（O1/O2/O3/O4 联合 24 案例全 EQUIV_PASS + 矩式拒绝判别段 + round-first 判别段）。gate 1 一次通过。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | **列因子广播（weight/bias (N,)）是 v-op 文档盲区**：`vmul/vadd/vsub` 文档化的自动广播仅「相同 shape / 行广播 [M,N]↔[M,1] / 标量」，**不含** `[M,N]×[1,N]` 列因子；`PL-1.15-broadcast-perf-tax` 只述「未文档化列广播可编译但有 +12~16% 税」，未给「标准配方」。本设计要求每个逐列因子算子自行判定 `vbrc` 展开与 bm==1 同形分支 | DESIGN.md §0.6 R5、§3.2、§9.1；`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 / `T.vadd.md` §2.2.2；`pattern-library/layout.md` PL-1.15 | 在 `docs/Tilelang.language/数学操作/T.vmul.md`（或 `shape操作/T.vbrc.md`）补「列因子广播标准配方」：`vbrc (1,N)->(bm,N)` + bm==1 退化为同形直接算（+ 指向 PL-1.15 的税与陷阱） | P |
| Info-source | **bf16 无 Vector 算术不覆盖 `vbrc` 判据的读法**：本算子 bf16 走 fp32 中转链后列因子也在 fp32 域 vbrc，规避了 bf16；但设计须逐一核对每个新引入 API 的 dtype 表（`vbrc` 虽支持 bf16，本设计仍在中转链后使用），易漏 | DESIGN.md §0.6 R4/R5、§3.4；`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.1、`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 | 在 `docs/Tilelang.language/` 的 dtype 表索引或 `tilelang-cube/vector-skill` 增补「fp32 中转链内所有 v-op 均在 fp32 域，dtype 表按中转后 dtype 核对」一句话 | C |
| Design-flow | **双输出 + 中段写出的 buffer 复用时序**须在设计中显式给出，否则 Stage 3 易漏写第二输出或过早复用 `stage` | DESIGN.md §0.1/§3.3/§4.3/§5.4；`tests/ops/test_fused_add_layer_norm.py:test_fused_add_layer_norm_kernel_returns_two_outputs` | 在 `tilelang-op-design` 模板 §4.3/§5.4 增补「多输出 op：逐输出标注写出时点与共享 buffer 的生命周期不重叠论证」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **列因子 (N,) 的 vbrc 核内展开配方（含 bm==1 同形退化）**：v-op 文档无 `[M,N]×[1,N]` 直连广播；标准做法 = 把 weight/bias 装入 (1,N) fp32 缓冲，`bm≥2` 用 `T.vbrc (1,N)->(bm,N)` 展开到复用缓冲再 vmul/vadd，`bm==1` 直接同形 vmul/vadd（规避 TRAP-vbrc-same-shape-empty-broadcast 的空广播维 MLIR verify fail）。替代未文档化列广播的 +12~16% 税（PL-1.15） | P | DESIGN.md §0.6 R5、§3.2、§9.2；`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 合法例（src (1,N,K)→dst (M,N,K)）；TRAP-vbrc-same-shape-empty-broadcast | none（机制判定；正确性由 §8.1 golden 两输出比对覆盖） | tilelang 0.1.2 dev-root（417dae99）+ Ascend910B3（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/layout.md（PL-1.15 补列因子标准配方） |
| **fused-add LayerNorm 的「舍回后统计」语义锚点**：源 kernel 的 S1/S2 在未舍回 fp32 和上算、中心化用舍回副本，而 golden(`ref_program`) 全用舍回值；二者在 fp16/bf16 相差 ≤1 dtype ulp（仅统计量）。迁移应显式选择 golden 对齐的「舍回后统计」，使双输出自洽（residual_out 与 y 用同一舍回值），并以 fp64 判别段机器验证两口径在容差内 | D | DESIGN.md §0.1/§1.6.1 O4、verify_equiv 判别段 B（24 案例 max_ulp≤1、违反率 0）；`tests/ops/test_fused_add_layer_norm.py:ref_program`；源 `_fused_add_layer_norm_fwd_kernels.py` L96-104 | `python examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root（417dae99）+ Ascend910B3 + CANN 8.5.2 | pattern-library/cases.md（fused-add LayerNorm 迁移档案）/ algorithm-candidates ALG-layernorm |

### Transferable Lessons（可迁移教训）

- **逐列因子（(N,) 广播到 (M,N)）在 npuir v-op 上无文档化直连广播**：不要写 `vmul([M,N],[1,N])`（PL-1.15：未文档化列广播可编译可算对但 +12~16% 税）；标准配方 = 装入 (1,N) 缓冲 → `T.vbrc` 展开到 (M,N) → vmul/vadd；**bm==1 时不可调 vbrc**（同形 src/dst 触发空广播维 MLIR verify fail，TRAP-vbrc-same-shape-empty-broadcast），直接同形 vmul/vadd，且该分支须放在工厂层（编译期）以规避 TRAP-C11 parser 作用域。
- **多输出 op 的两个输出各有写出时点**：第二输出（如 residual_out）常在计算中段就绪，第一输出在末尾；两者可共享同一 dtype 中转 buffer，但必须论证生命周期不重叠（本算子：residual_out 写出后 stage 才被 weight/bias 装载复用）。`out_idx` 顺序必须与 wrapper 解包顺序一致（[4,5] → (y, residual_out)）。
- **含 residual 和的 norm，其「统计域」语义有两种合法口径**（源：未舍回和上统计 / golden：舍回值上统计），fp16/bf16 差 ≤1 ulp；迁移应显式选择并与 golden 对齐，用 fp64 判别段机器验证——不要默认照抄源 kernel 或 golden 任一侧而不加标注。
- **vrsqrt 近似（2.9e-3）在含 rsqrt 且容差 <1e-2 的算子（LayerNorm 族）上不可用**：改 `T.vsqrt` + `T.vdiv` 组合（sqrt(v)/v）达 fp32 1.07e-7（TRAP-vrsqrt-plain-precision，norm 族通法；本算子 fp32 门禁 1e-5 必需）。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-14

### 结论

`_fused_add_layer_norm_kernel` 实现完成（`ASCEND_MODE = "Developer"`，`@tilelang.jit(out_idx=[4,5], target="npuir")`），严格按 DESIGN.md §3.3 伪代码落盘：persistent 分核 `num_kernels=min(num_logical, vector_cores)`（实查 20 AI Core → 40）、尾块 `T.min` + src/dst 双显式 slice + 垃圾行丢弃、fp16/bf16 fp32 中转链（`T.vcast(rint)`）、`rstd = sqrt(v)/v`（`T.vsqrt`+`T.vdiv`）、列因子 `T.vbrc`（bm≥2）/同形直接算（bm==1）。真实 NPU 验证：`--level all` 全绿——L0（13/13）、L1（全过）、L2（6/6）、Boundary（6/6）。判定 `[PRECISION_PASS]`。

max_diff（vs golden，两输出各自比对）：fp32 **2.861e-06**（<1e-5 门禁）；fp16 **3.906e-03**（≤1 ulp @ |y|≈4，<1e-3+rtol·|ref|）；bf16 **1.562e-02**（L0）/ 3.125e-02（L1，rtol 覆盖）。Boundary `extreme-affine`（±65504 weight/bias）max_abs=1.0 但两侧同为饱和/NaN，`equal_nan=True` 判等通过（非缺陷，见下 O2 角点注）。

### 调试记录（attempt 1，1 次修复，≤5 上限）

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | fp16/bf16 路径编译失败：`error: 'hivm.hir.vcast' op expects 'transpose' to be a permutation, found 0`（fp32 路径同批全过） | 列因子装载写成 DESIGN §3.3 的 `T.copy(weight[0:N], stage[0:1, 0:N])` + `T.vcast(stage[0:1, 0:N], w1)`——`block_m≥2` 时 `stage` 为 `(block_m,N)`，其**子行 slice** `stage[0:1,0:N]` 被 lowering 成对整个 `(2,N)` 缓冲的 `hivm.hir.vcast ... -> tensor<1xN>`（带非法 `transpose=0`），即 slice-of-buffer 退化 | 用**专用 (1,N) 中转缓冲 `wstage`**（非 `stage` 的子行）：`T.copy(weight[0:N], wstage[0:1,0:N])` + `T.vcast(wstage, w1)`，weight/bias 各自完整 buffer，编译通过 |

`T.copy(weight[0:N], wstage[0:1,0:N])`（rank-1 src → (1,N) dst）本身工作正常（fp32 路径同型 `T.copy(weight[0:N], w1[0:1,0:N])` 一次通过）；失败点在**对 `(block_m,N)` 缓冲做 `[0:1,:]` 子行切片的 vcast/T.copy**——与 REVIEW 维度 1 的 weight→(1,N) 建议级风险同源，实测结论：单列因子应给独立 (1,N) buffer，不要在复用大缓冲上切首行。

### 精度要点

- fp16 max 3.906e-03 = fp16 在 |y|≈4 处的 1 ulp：源/golden/kernel 均 fp32 opmath + 单次舍回，残差来自规约树序（torch layer_norm 内部序 vs NPU `reduce_sum` 树序）对 mean/var 的 ≤1 ulp 抖动，非算法缺陷；`verify_equiv` 判别段 B 已证两统计口径在容差内。
- **O2 v=+inf 角点（REVIEW 维度 8 建议级）实测无害**：`rstd=sqrt(v)/v` 在 `v=+inf`（仅当 `d²` fp32 上溢，|d|≳1.8e19）退化为 NaN，而源 `rsqrt(inf)=0`；本任务 workload 与全部随机/角点输入（含 Boundary-large ×100、extreme-affine ±65504）均不可达该阈值，Boundary 全过。若未来域含 fp32 超大量级需在文档标注该边界（代码未偏离 DESIGN §1.6.1 采纳式）。

### 与 DESIGN 的偏差登记

| 项 | DESIGN 值 | 实现值 | 依据 |
|----|-----------|--------|------|
| UB 预算常数 | 192KB/1.7≈113KB（`bpe·bm·N+4N`） | 实测 `20 B/elem`（ada 同工具链实证），`_UB_MAX_ELEMS=9216` | 与 ada_layer_norm attempt-1 实测一致（fp32 bm=3 N=4096 = 245760B 溢出）；DESIGN §9.1 contingency 授权降档 |
| L0-1 fp32 N=4096 | bm=3 | bm=2 | 同上（3·4096=12288>9216） |
| L0-10 fp32 N=5120 | bm=2 | bm=1 | 同上（2·5120=10240>9216） |
| 列因子装载 buffer | `stage[0:1,0:N]` | 专用 `wstage (1,N)` | 见调试记录 #1（slice-of-buffer 的 vcast 退化） |

### Transferable Lessons（写给本迁移任务后续函数）

- **不要在复用的 `(block_m,N)` 大缓冲上取 `[0:1, 0:N]` 子行喂 `T.vcast`/`T.copy`**：`block_m≥2` 时该 slice 会被 lowering 成对整个缓冲的转置式 vcast，BiShengHIR 报 `'hivm.hir.vcast' op expects 'transpose' to be a permutation, found 0`；给单行因子/单行临时量分配独立的 `(1,N)` buffer。fp32 路径 `T.copy(weight[0:N], w1[0:1,0:N])`（rank-1→(1,N) 直拷）不受影响。
- **UB 预算必须按实测的 auto-multi-buffer 系数（本工具链 20 B/elem，persistent serial ≥2 次时）而非 DESIGN 的 1.7x 名义裕度**：norm 族 `bpe·bm·N`（fp16 10 / fp32 8）会被 multi-buffer 放大约 2.0x/2.5x；容差 = 名义 §4.5 表值可能超限，实现期按 9216 元素上限降 bm。
- **`T.vbrc` 的 bm==1 退化分支 + dtype 分支必须放工厂层（trace 体外的 Python `if`）**：本算子 `use_fp32_transit` 与 `if block_m == 1` 均是 trace 期常量分支，规避 TVM parser C11 作用域问题，编译一次成功。
- **双输出中段写出（residual_out）在 fp32 路径下也能安全复用**：fp32 直拷路径 `T.copy(x_f32[0:real_m,0:N], residual_out[...])` 后，`x_f32` 继续承载 d/t，无生命周期冲突。
