# Stage 1 复盘 — `_conv1d_pointwise_kernel`（Conv1dFwdOp 函数 1/5）

## Stage 1 (design) Retrospective — first_design — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 设计期实查物理核数时，skill 文档给的导入路径 `from tilelang.utils.npu import NPUUtils` 在本工具链不存在；实际可用路径为 `from tilelang.utils import NPUUtils`（`tilelang/utils/__init__.py` 再导出 `npu_utils.NPUUtils`） | 本任务 `python3 -c "from tilelang.utils.npu import NPUUtils"` → `ModuleNotFoundError`；`from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → `20` | 在 `.agents/skills/_shared/standards/core-split-strategy.md` §1 要素② 与 skill Phase 1 第 5 项补一句可用导入路径（`from tilelang.utils import NPUUtils`） | D |
| Design-flow | 门禁 `S1-REF-PATHS` 对正文中的**省略号伪路径**（`examples/TileOPs/.../_xxx.py`）与**裸目录前缀**（`docs/TileLang`、`docs/examples`）机械判失败——设计书写「先例路径」时习惯用省略号缩写，易踩 | 本任务首次 gate 1 失败 3 条 `S1-REF-PATHS`（省略号路径 + `docs/TileLang`/`docs/examples` 裸前缀） | design 模板/quality-checklist 增一条：引用 `examples/`/`docs/`/`.agents/` 路径必须写完整且真实存在，禁止 `...` 缩写；负向佐证写「本仓文档与 examples 佐证」而非 `docs/examples` 连写 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Cube Expert 迁移的 pointwise/GEMM 设计骨架**：无 bias=纯 Cube（`T.Scope("Cube")` + `T.alloc_L1/L0C` + `T.gemm(initC=(k==0), size=[M,K,N])` + L0C→GM fixpipe f32→f16）；有 bias=MixCV（Cube 经 GM fp32 ws 中继 → Vector `vbrc`+`vadd`(fp32)+`vcast`），因 `CONST-store-fixpipe-gm-only` 跨引擎强制 GM 往返 | P | `examples/conv1d/_conv1d_pointwise_kernel/DESIGN.md` §0.6 R1/R3、§3.3、§4.4 | 见 `examples/conv1d/probe/conv1d_geomm_probe.py`（Cube GEMM conv 探针）；参考 `_gqa_prefill_fwd_kernel.py` / `_ssd_chunk_scan_fwd_kernel.py` | tilelang 0.1.2+417dae99 + Ascend910B3（`get_aicore_num`=20）+ CANN（npu-smi 25.5.0） | pattern-library / cases.md |
| 一维 Kernel 分核：Cube 直取 `get_aicore_num()`（不翻倍），统一 persistent `num_kernels=物理核数` 覆盖「无需适配 / 1× 整数倍 / 大规模」全档；`cid = T.min(task_id*num_kernels+kernel_id, num_logical-1)` 钳位防 idle 核越界 | P | DESIGN.md §5.5、§3.3 注 3 | `from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → 20 | 同上 | `.agents/skills/_shared/standards/core-split-strategy.md` / pattern-library |
| pointwise conv1d（kW=1）无数学等价优化空间：plain GEMM `out=W@x(+bias)` 已是 FLOPs 下界；bias 折叠（augmented-K）因 K=`C_in+1` 破坏 16 对齐 + `x_aug` 物化访存而净亏；转置 GEMM 是布局选择非公式化简 | P | DESIGN.md §1.6.0 R1、§1.6.1 | none（纸面论证 + R3 四口径对比） | 同上 | pattern-library / algorithm-candidates.md |

### Transferable Lessons（可迁移教训）

- 本仓 NPU 物理核数**必须实查**且导入路径为 `from tilelang.utils import NPUUtils`（`from tilelang.utils.npu import ...` 会 `ModuleNotFoundError`）；本机设备为 Ascend 910B3，`get_aicore_num()=20`（≠ pattern-library `CONST-aicore-910B2C` 记录的 24），一律以实查为准。
- Cube 迁移一律避开 `T.load_nd2nz` / base+size 形态 `T.copy` 处理**跨步区域**（`TRAP-load-nd2nz-strided` / CG-2026-0004 静默平坦误读）；改用 slice 形态 `T.copy`（GM→L1 自动 ND2NZ），探针已实证。
- Cube 的 K 维尾块须用**分形 clamp** `T.max(16, T.ceildiv(tail,16)*16)` 传 `T.gemm` 的 `size`，而 `T.copy` 用动态 slice extents 搬有效区、输出取 slice 裁剪——两者配合规避分形非法与越界写（`TRAP-T-copy-region-semantics`）。
- Expert 模式 `T.copy` 在 GM↔UB 通用路径**不跨 dtype**（`TRAP-C12-copy-dtype-cast`）：fp32→fp16 必须显式 `T.vcast`，不能靠 `T.copy` 隐式转换。
- 无 bias 时 pointwise 是**纯 Cube 单 Scope**，bias 才引入 Vector 段——设计上应把「无 bias 最小结构」作为基线先实现/验证，再叠加 MixCV epilogue。
- 写 DESIGN.md 引用仓库路径时**禁止省略号缩写**，且避免 `docs/xxx` 后紧跟中文顿号产生伪路径 token（门禁 `S1-REF-PATHS` 机械校验每条引用路径真实存在）。

## Stage 1 (design) Retrospective — revision v1（MixCV ws 握手协议修正）— 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 设计模板/自检清单未把「跨引擎（Cube↔Vector）生产者-消费者握手」列为**必须双向完整**的强制项，导致 v0 的 §7.2（散文写了双向 set/wait）与 §3.3/§6.2（伪代码只落生产端 set + 消费端 wait，漏生产端写前 wait）各写各的、自相矛盾 | v0 DESIGN.md §3.3/§6.2 vs §7.2（REVIEW 问题 1 阻塞）；v1 改为深度 2 双槽 4-flag 双向握手 | design-template / quality-checklist 增一条硬性项：凡出现跨引擎中转缓冲（GM ws 等），必须用**同一张 flag 语义表**定义握手，并在 §3.3/§6.2/§7 三节逐字段一致（flag 编号 / set-wait 方向 / 放置位置），禁止各节独立描述 | R |
| Design-flow | 首版设计习惯用省略号缩写引用先例文件，踩 `S1-REF-PATHS`（v0 已记录）；本版复核确认修复方式（写完整真实路径 + 用简称不带仓库前缀）有效 | v1 gate 1 `S1-REF-PATHS` 0 失败 | 与 v0 条目一致，无需新增 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **跨引擎 GM ws 中继必须用深度 2 双槽 4-flag 双向握手**（`2*slot`=就绪、`2*slot+1`=空闲；生产者在写前 `wait(slot_free)`、写后 `set(ready)`；消费者读前 `wait(ready)`、消费后 `set(slot_free)`；前导生产者置两槽空闲）——单槽必须在生产端写前补 `wait`，否则 WAR 数据竞争（随机精度错误，非编译错误） | P | DESIGN.md v1 §0.6 R3 / §3.2 / §3.3 / §6.2 / §7.2；对齐 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`（深度 2、前导 set(1)/set(3)、Cube `wait(2*slot)`/`set(2*slot+1)`、Vector `wait(2*slot+1)`/`set(2*slot)`；其注释「placing the wait BEFORE any ws[slot] write prevents the off-by-one WAR race」） | `T.sync_block_wait.md` §2.4 规范示例（Cube 在 store_fixpipe 前 `sync_block_wait(1)`）；ssd 内核 L253-255/266-267/341-342/398-399/456-457 | tilelang 0.1.2+417dae99 + Ascend910B3（`get_aicore_num`=20）+ CANN（npu-smi 25.5.0） | pattern-library / cases.md |
| **跨引擎中转缓冲须显式列入 prim_func 签名**（非仅注 4 一句「wrapper 分配」）——对齐 ssd 将 `ws_c`/`ws_lcb` 作为 `T.Tensor` 形参；伪代码引用未声明符号会让 Stage 3 直译即遇未定义符号 | P | DESIGN.md v1 §3.3（`ws: T.Tensor((num_kernels, slots, block_M, block_N), "float32")` 形参）/ §4.3 / §4.5 | `_ssd_chunk_scan_fwd_kernel.py` L227-228（ws 形参） | 同上 | pattern-library / cases.md |

### Transferable Lessons（可迁移教训）

- **跨引擎握手自查三件套**：写 NPU 双 Scope（Cube↔Vector）设计时，对每个中转缓冲逐一核对——① 生产端**写前**是否 `wait(槽空闲)`（WAR）；② 生产端**写后**是否 `set(就绪)`；③ 消费端**读前**是否 `wait(就绪)`、**消费后**是否 `set(槽空闲)`；并保证 §3.3（伪代码）/§6（循环）/§7（同步表）三处 flag 编号与放置位置完全一致。v0 缺 ①，是典型「握手只画了一半」。
- **深度 2 双槽**是 Cube↔Vector ws 中继的默认正确形态（`PL-1.12`）：既消除 WAR 竞争，又让 Cube(task t) 与 Vector(task t-1) 任务级重叠；前导必须由生产端置两槽空闲，否则首两个任务死等。
- **跨引擎缓冲声明**：凡跨引擎（经 GM）中转的 workspace 必须在 prim_func 签名（或显式 `alloc`）中声明清楚，不能只在注释里提一句——否则下游 Stage 3 直译即缺符号。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | Stage 2 对 ws 握手的机械复核只验证了 **flag 编号 / set-wait 方向 / 放置位置（写前读前）**，未验证**前导置位落在哪个引擎**；把 DESIGN §3.3/§7.2 的「生产端（Cube）置两槽空闲」判为正确。实际该前导是**同引擎自 set→自 wait**（Cube set，Cube wait），`sync_block_set/wait` 是 AIC↔AIV 事件、不回流到 setter，故首两个任务死等 → 设备 aicore timeout。 | DESIGN.md v1 §3.3 注 3 / §6.2；REVIEW.md 维度 5（独立时间线复推判为 race-free）；Stage 3 实测 `acl error 507014 / aicore execution times out`（`_bias_main_mix_aic.so`） | `tilelang-design-review` 检查项「跨引擎握手」补一条：前导/初始置位必须与等待方**不同引擎**（setter != waiter），显式排除同引擎自 set→自 wait；对齐 ssd 先例（前导在消费者侧） | R |
| Develop-flow | DESIGN §3.3 把输入 dtype 的 `bias`（fp16/bf16）用 `T.copy` 直接装入 fp32 的 `bias_col`，与自身 §4.4/§9.1「Expert `T.copy` 不跨 dtype」矛盾；Stage 3 直译即编译失败。 | DESIGN.md §3.3 步骤 6 vs §4.4；实测编译错误 `codegen_npuir_api.cc:1421 Check failed: (src->dtype == dst->dtype) ... T.copy does not support element type casting` | design-template §3.3 伪代码自检：凡 GM 输入为输出 dtype、片上中间为 fp32 的搬入，必须写「dtype 缓冲 + 显式 `T.vcast`」两步，不得一步 `T.copy` 跨 dtype | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **跨引擎 flag 握手的前导必须置于消费者引擎**：`sync_block_set`/`sync_block_wait` 是 AIC↔AIV 事件，同引擎自 set→自 wait 不 retire（设备 aicore timeout 507014，非编译错误）。当生产者（Cube）在写前 `wait(slot_free)` 时，"空闲"初值必须由**消费者（Vector）**引擎 `set`（对齐 ssd 的消费者侧前导）。 | D | DESIGN.md v1 §3.3/§6.2/§7.2（生产端前导，实测死锁）；修复后 `_conv1d_pointwise_kernel.py` Vector Scope 前置 `sync_block_set(1)/set(3)` | `examples/conv1d/_conv1d_pointwise_kernel/repro/TRAP-sync-self-set-wait.py`（消费者侧前导 bit-consistent 通过；生产侧变体见头部注记） | tilelang 0.1.2+417dae99 + Ascend910B3（`get_aicore_num`=20）+ CANN（npu-smi 25.5.0） | pattern-library / traps-runtime.md |
| **MixCV bias 从输入 dtype 到 fp32 域必须显式两跳**（`f16/bf16` 缓冲 `T.copy` 载入 → `T.vcast` 到 fp32 → `vbrc`/`vadd`）：Expert `T.copy` 不跨 dtype，`bias[..] -> bias_col(fp32)` 一步会触发 `T.copy does not support element type casting`。 | D | Stage 3 编译错误 `codegen_npuir_api.cc:1421`；修复后 L0/L1 全过（fp16 atol 1e-2） | `examples/conv1d/_conv1d_pointwise_kernel/_conv1d_pointwise_kernel.py` Vector Scope（`bias_16` → `vcast` → `bias_col`） | 同上 | pattern-library / traps-compiler.md（补 TRAP-C12 的 conv 迁移例） |

### Transferable Lessons（可迁移教训）

- **跨引擎（Cube↔Vector）flag 握手的「前导/初始置位」必须放在与等待方不同的引擎上**：`sync_block_set`/`sync_block_wait` 是 AIC↔AIV 事件，同引擎自 set→自 wait 不 retire（设备侧表现为 aicore timeout 507014，而非编译错误）。生产者在写前 `wait(free)` 时，free 的初值由**消费者**引擎置位（对齐 `_ssd_chunk_scan_fwd_kernel.py` 的消费者侧前导）。
- **MixCV 的输入-dtype → fp32 中间缓冲一律两跳**：先 `T.copy` 到同 dtype UB，再显式 `T.vcast` 到 fp32；Expert 模式 `T.copy` GM↔UB 不跨 dtype（`TRAP-C12`）。bias 的 fp32 域相加（`vbrc`+`vadd`）后统一 `vcast` 回输出 dtype。
- **尾块三分工**：`T.copy` 用动态 slice extents 只搬有效区（越界补 0）、`T.gemm` 的 `size` 用分形 clamp `T.max(16, ceil16(tail))`、输出用 slice 裁剪——非整除 `C_out/L/C_in` 均由该组合覆盖（本函数 L0/L1/L2/Boundary 含 `c_in=17/20/50/80/96`、`c_out=33/100/333`、`l=33/777/1000/3000` 均通过）。
- **`out_idx=[2]` 允许输出参数不在末尾**：bias 变体签名 `(x, weight, out, bias, ws)` 中 `out_idx=[2]` 有效，调用按非输出参数顺序 `kernel(x, weight, bias, ws)` 传入即可。
- **Mix（Cube+Vector）内核死锁的快速判据**：`aicore execution times out`（acl 507014）且栈指向编译产物里的 `*_mix_aic.so` 时，应优先怀疑跨引擎握手（同引擎自 set→自 wait / 前导放错引擎），而非精度问题。
