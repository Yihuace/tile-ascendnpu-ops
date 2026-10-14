# RETROSPECTIVE.md — _batch_norm_bwd_split_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-14

### 结论

`BatchNormBwdOp` 的拆分核 `_batch_norm_bwd_split_kernel` 迁移设计完成（Expert / Vector）。相对同族宽核（one core per channel）的**核心新增设计点 = 跨段归约的 NPU 落地方案**：源三段（stats 分块部分和 → finalize 合并 → apply）保持为三个 npuir kernel，**合并采用「GM 部分和 workspace `[C,splits]` + 第二阶段 finalize kernel」，明确不使用 atomics**（源注释要求 `grad_bias/grad_weight` 不依赖合并顺序；atomics 累加顺序不定破坏该确定性，且 NPU 单 grid 无全局 barrier，跨段交接只能靠 workspace + 顺序 kernel 或 atomics，取前者）。段切分沿 **S 轴仿射子段**（`c = t // splits`、`sp = t % splits`，cap `splits ≤ S`），段内 lane 向量化累加 + 两步 `T.reduce`（`clear=False` + `size=[N, r2]` 截尾）。等价性由 `verify_equiv.py` 机器验证（O1 倒数乘 + O2 分布式 epilogue，21 案例 max ulp 0 全 EQUIV_PASS；raw-moment 判别段 A fp32 违反率 0.1553 → 拒采）。**本函数设计探针 `history_version/design_probe_split_phase.py` 真机 PASS**（验证 floordiv/mod 段索引用于 `T.copy` 中间轴切片、部分和 workspace 标量 slice 写、finalize 合并、逐通道 apply 三段结构，`max_abs_err grad_x=9.5e-07`）。gate 1 一次通过（无 failure、无 warning）。实查 `NPUUtils.get().get_aicore_num() = 20`（×2 = 40）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **Python 内建 `min()` 置于 `@T.prim_func` trace 体内触发 trace 期报错**：`K = min(ntasks, VEC)`（ntasks/VEC 均为 Python int）在 prim_func 体内求值时后端报 `Cannot use and / or / not operator to Expr, hint: use tvm.tir.all / tvm.tir.any instead`。`min/max` 内建对 trace 期表达式不适用；编译期常量（`K/nl/chunk/n_tile/bS/splits`）必须在**工厂层**（`@tilelang.jit` 装饰函数体内、`@T.prim_func` 之外）算好并以闭包常量注入 | 本函数设计探针首轮 trace 报错指向 `design_probe_split_phase.py` 的 `K = min(ntasks, VEC)`；上移到工厂层后三段全编译成功 | 在 `tilelang-op-design` 模板 §3.3 注 / 自检清单补一条：「`@T.prim_func` trace 体内禁用 Python 内建 `min/max/and/or`；编译期常量在工厂层预算」；或 `tilelang-op-develop` 模板同步 | R |
| Info-source | **跨段（跨 block）归约在 NPU 上的落地方案无独立文档先例**：`migration-analysis.md` §6.2 仅一句「跨 block：workspace + 核间同步，或拆两阶段 kernel」，未给「无 atomics、确定性合并」的显式配方与 API 形态（部分和标量 slice 写、`(1,splits)` 行 reduce 合并） | DESIGN.md §0.6 R3/§3.3/§7.2；探针 `design_probe_split_phase.py`（workspace 写 + finalize 合并真机 PASS） | 在 pattern-library/layout.md 或 `migration-analysis.md` §6.2 增补「跨 block 归约 NPU 配方」：GM workspace `[K, splits]` + 第二阶段 kernel（固定序合并，避 atomics）；标注 `T.atomic_add` 可用但破坏顺序确定性 | C |
| Info-source | **floordiv/mod 通道-段索引用于 `T.copy` 中间轴 strided 切片**：文档/探针未覆盖「`c = t // splits` 作为 `T.copy(grad_out[0:N, c, s0:s0+real], ...)` 的中间轴索引」组合，设计者易因「非仿射」先验否决段切分 | DESIGN.md §1.6.3 候选 #2（flat 轴被拒）/ #1（S 轴采用）；设计探针真机验证 floordiv/mod 切片可用 | 在 `docs/Tilelang.language/内存操作/T.copy.md` 或 pattern-library 记录「运行时标量（含 floordiv/mod 结果）可作 `T.copy` 任意轴起始索引；extent 可运行时（配合 `T.min`）」 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **拆分核（跨 block 通道归约）NPU 配方 = GM 部分和 workspace `[C,splits]` + 第二阶段 finalize kernel（固定序合并，无 atomics）**：`stats` 逐 (通道,段) 写 `partial_sum[c, sp]`（无写冲突）；`finalize` 载 `(1,splits)` 行 `T.reduce(dims=[1])` 合并；比 atomics 更契合源「合并顺序无关」契约且 NPU 单 grid 无全局 barrier | D | DESIGN.md §0.6 R3/§3.3/§7.2；探针 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py` 真机 PASS（grad_x 9.5e-07） | `python examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py` | tilelang 0.1.2+417dae99 + Ascend910B3（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/layout.md（跨 block 归约）/ migration-analysis.md §6.2 |
| **floordiv/mod 段索引（`c = t // splits`、`sp = t % splits`）可作 `T.copy` 中间轴 strided 切片，1D persistent grid-stride 下可用**：段切分沿 S 轴仿射子段（`[0:N, s0:s0+real)`），避免 flat 段（`l//S`/`l%S` 逐元素索引）破坏向量化 | D | DESIGN.md §1.6.3 候选 #1；探针 `design_probe_split_phase.py`（`c = t // SPLITS` 真机编译运行） | `python examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py` | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | docs/Tilelang.language/内存操作/T.copy.md / pattern-library |
| **`@T.prim_func` trace 体内禁用 Python 内建 `min/max/and/or`；编译期常量在工厂层预算**：`K = min(ntasks, VEC)` 在 prim_func 体内报 `Cannot use and/or/not operator to Expr`；移到 `@tilelang.jit` 工厂层后全通过 | R | 本函数设计探针首轮 trace 报错（`design_probe_split_phase.py`）；DESIGN.md §9.2 常见错误行 | `python examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py`（首轮 fail、上移工厂层后 PASS） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | tilelang-op-design 模板 §3.3 / tilelang-op-develop 模板 |

### Transferable Lessons（写给本迁移任务后续函数）

- **跨 block（跨段）归约的 NPU 配方 = GM 部分和 workspace + 第二阶段 kernel，不用 atomics**：每个 (通道,段) 写自己那格 `partial_sum[c, sp]`（无写冲突），第二阶段 kernel 载 `(1,splits)` 行做 `T.reduce(dims=[1])` 固定序合并。NPU 单 grid 无全局 barrier，跨段交接只能靠「workspace + 顺序 kernel 下发」或 atomics；需要确定性（如源注释「不得依赖合并顺序」）时选前者。`T.atomic_add` fp32 可用但累加顺序不定，破坏确定性。
- **段切分沿「规则轴的仿射子段」而非扁平区间**：通道逐段区域取 `[0:N, s0:s0+real)`（沿 S 轴），`c = t//splits`、`sp = t%splits` 的 floordiv/mod 结果可作 `T.copy` 的中间轴起始索引（真机验证）；避免 flat 段（元素索引 `l//S`/`l%S` 非仿射）破坏向量化。`splits` cap 到被切轴长度以保证每段非空。
- **`@T.prim_func` trace 体内不要用 Python 内建 `min/max/and/or`**（会对 TIR 表达式报 `Cannot use and/or/not operator to Expr`）：所有编译期常量（分核数、任务数、子块数、bS、chunk、splits）在**工厂层**（`@tilelang.jit` 装饰体、`@T.prim_func` 之外）算好，trace 体内只用常量或 `T.min`。
- **尾段/尾子块边界 = I/O 双 slice 截断 + 归约 `size=[...]` 截断（成对）**：段 extent `real=T.min(chunk, S-s0)`、子块 extent `r2=T.min(bS, s0+real-ss)`；装载/写出用双显式 slice，归约传 `size=[N, r2]`；缺一即把未初始化/残留列计入部分和（同族宽核 v0 缺陷根因）。
- **三段共用同一 `splits`**：工厂内 `splits`（取 wrapper `_split_seed` 语义）必须与 host 预分配的 workspace `[C, splits]` 形状一致，否则错位；三段 `@T.prim_func` 参数名/顺序逐字保持源，`apply` 保持 `out_idx=[-1]`。
