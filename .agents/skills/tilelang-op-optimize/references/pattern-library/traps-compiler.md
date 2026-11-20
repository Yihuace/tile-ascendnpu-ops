# 已知编译器/解析器陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。运行时/数值/语义类陷阱见 [traps-runtime.md](traps-runtime.md)。
>
> **证伪协议（强制，canonical 见 INDEX.md §2，两者同步演进）**：
>
> 1. **否定任何 API/模式前，必须用文档合法形态测试**——先查 `docs/Tilelang.language/` 确认 API 的合法参数/形式，穷举代表性写法后再下结论。实测教训：曾以非法 3-cycle permutation（`[1,2,0]`）与 Parallel 循环累加形式测出"编译失败"，误判"transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益。
> 2. **一切编译器约束结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
> 3. 证伪更正时须在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

---
id: TRAP-C1-old-compiler-transpose
kind: trap
family: [general]
apis: [T.transpose]
dtype: []
device: 910B2C
status: overturned
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08-27 前旧 build
repro: repro-missing
---

### C1 旧编译器 transpose/C-slice 形态不可用

**已失效**（2026-08-27 tilelang 重编译后推翻；合法形态见 [layout.md](layout.md) PL-1.1/PL-1.2）。

---
id: TRAP-C10-strided-gather
kind: trap
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C10 跨步 gather 向量指令缺失

非 C 轴连续的跨步访问小心——跨步 gather 向量指令缺失（截至 2026-08-28 build 有效）。

---
id: TRAP-C11-parser-vartable
kind: trap
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C11 parser var-table 作用域问题

有效（截至 2026-08-28 build）。

---
id: TRAP-C12-copy-dtype-cast
kind: trap
family: [elementwise]
apis: [T.copy]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: 截至 2026-08-28 build；lerp 证伪 2026-09-07
repro: repro-missing
---

### C12 `T.copy` 静默跨 dtype 转换

有效（截至 2026-08-28 build）；Developer GM→UB lowering 为隐藏 hivm::VCastOp（`docs/Tilelang.language/内存操作/T.copy.md` §3），跨 dtype copy 不减少 vector pass 且舍入不可控——勿用作 cast 融合（lerp_tensor 2026-09-07 文档证伪）。

---
id: TRAP-UB-multibuffer-inflation
kind: trap
family: [elementwise, general, reduction]
apis: [T.Pipelined, T.serial]
dtype: []
device: 910B2C
status: verified
origin_task: mish-optimize-2026-08（首证）/ lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z（第二证）/ ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（第三证：结构依赖精测；2026-09-10 蒸馏溯源归位——原回写误标 145312Z，实际 optimize task_id 以 stage_state.json 为准）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z（第四证首源：基础分配差发现）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第四证重校：×1.10–1.12 4 点标定 + UB diet 绕法）
toolchain: 2026-08-24 mish 首证；2026-09-07 lerp_tensor 第二次独立证据；2026-09-10 ada_layer_norm 精测（tilelang 0.1.2+a83118285a + CANN 8.5.0）；2026-09-15 基础分配差（首源 tilelang 0.1.2+28783f45 / 重校 0.1.2+6797758，均 + CANN 8.5.0）
repro: repro-missing
---

### auto-multi-buffer UB 膨胀：footprint 与 staging 数量解耦（~20/26 B/elem 平台）+ num_local/静态 extents 依赖

大 block_size 探索须按膨胀后预算评估；削 buffer 不必然解锁（lerp fp32 3-buffer 12288/16384 仍 blocked）。有效（2026-08-24 mish 首证；2026-09-07 lerp_tensor 第二次独立证据：`ub overflow, requires 1966080 bits while 1572864 bits available`）。

**第三证精测（ada_layer_norm，2026-09-10，编译溢出报文逐配置读数）**——三条可操作规律：

1. **膨胀后 footprint 是"结构类平台值"，不随 staging 数量线性增长**：persistent serial（num_local_tasks≥2）激活时，fp16/bf16 中转循环（vcast 链）无论 resident 10 B/elem（3 缓冲）还是 14 B/elem（5 缓冲），multi-buffered 需求同为**精确 20.01 B/elem**；fp32 直连无论 8 B 还是 16 B resident，同为**精确 26.00 B/elem**。⇒ UB 预算 guard 从"resident × 膨胀系数"改为"按结构类查平台值"（transit≈20/21、fp32≈26/27 B/elem）。
2. **需求随 num_local_tasks 变化**：同配置（bm=8×N=1152 transit）nl=3 可编译（20.01 B/elem=184.3KB）、nl=4 溢出（22.00 B/elem=202,784B）——分核参数（num_kernels）会改变 UB 可行域，调 num_kernels 必须连带复验编译。
3. **静态 slice extents 也改变需求**：`M % block_m == 0` 特化（real_m 常量化、静态 extents）使 transit 需求 20.01→22.00 B/elem——静态化优化在 UB 贴限时可能直接翻车。

**第四证/姊妹现象（2026-09-15 两任务实证，VP-2026-0059 两证合入）——基础分配差（multi-buffer 关闭仍在）**：`--enable-auto-multi-buffer=false` 下 BishengIR per-AIV UB 实际需求 ≈ 真实手工预算 × **1.10–1.12**（首证 28783f45：178.2KB 预算实测 ~204.1KB +15%；第二证 6797758 4 点标定：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%）——手工逐 buffer 预算表**不能当可编译证明**，设计期按系数放大且勿漏计小 buffer（第五轮实证：DESIGN §4.5 漏计 ub_cond2 8.2KB）；宽 config 上靶前以编译探针校准（`ub overflow` 报文 requires bits 即精确实测需求）。系数已录入 constants.md CONST-capacity-910B2C；与 CG-2026-0008（能力缺口本体，recurring）互链。绕法实证：UB diet 削减手工预算 ≥12.4KB（N/D staging 合并 −24.6KB + rowmat 删除 −16.3KB）解锁 (64,256) 宽块。**反例边界（2026-09-16 第六轮 r9，a13585dc）**：显式 alloc 结构（全 UB buffer 首维 = half）实测 0.981×manual（actual 低于手工）——系数结构依赖、非单向上偏，该结构 ~12% 余量即足（constants.md CONST-capacity-910B2C / PL-1.12 r9 update）。

机制推测（未实证）：pass 按 buffer 角色/循环形态决定复制哪些实例，非全量 ×2。溯源：`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md`（Iteration 2/3 探针记录 + probe_ub_v2.py）；第四证溯源：首证 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md` Round 2 + 第二证 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/opt_log.md` Phase 1 编译探针（probe_ub.py，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z）。

---
id: TRAP-vbrc-scalar-shared
kind: trap
family: [expert]
apis: [T.vbrc, T.if_then_else]
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### vbrc 标量→shared 不可用；serial 变量条件 segfault

vbrc 标量→shared 不可用；serial 变量条件 if_then_else select segfault（有效，截至 2026-08-28 build）。

---
id: TRAP-threads-kwarg-noop
kind: trap
family: [general, migration]
apis: [T.Kernel]
dtype: []
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: tilelang-mlir-dev dev root build 2026-09-07 源码通读
repro: repro/TRAP-threads-kwarg-noop.py
---

### `T.Kernel(..., threads=)` 在 npuir 无效果

`src/ir.cc` L175–203 KernelLaunch NPU 分支仅消费 grid_size、block_size 不生成任何 threadIdx 绑定（Python 签名仍接受并透传该 kwarg）；GPU 源码迁移遇 `threads=` 直接移除即可，threads/npt 折叠为单一 block_size 由 wrapper 参数传入（复现：`python repro/TRAP-threads-kwarg-noop.py`——机械核对 src/ir.cc NPU 分支无 threadIdx 绑定）。有效（tilelang-mlir-dev dev root build 2026-09-07 源码通读；origin provenance：lerp_tensor 任务 REVIEW.md 附#28——任务工作区，未上库，允许失效）。

---
id: TRAP-tvm-parser-rules
kind: trap
family: [expert]
apis: [T.prim_func, T.rs, T.alloc]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C；developer 谱系 v3_deadbuf 跨工具链二次一致（tilelang 67db6f3 + CANN 26.0.rc1）
repro: repro/TRAP-tvm-parser-rules.py
---

### TVM script 解析器四条硬规则

违反报 Undefined variable / trace error：`if` 语句一律生成 TIR runtime if（不折叠、赋值作用域困在 if 体内）→ trace-time 选择用标量三元式或 Python 层预计算；三元式**两臂都会求值**（buffer 切片臂不能用哨兵 tensor → 无条件全尺寸 alloc）；条件 `T.alloc` 非法（哨兵只可用于 shape 三元式且该 buffer 不得在运行 if 体内引用）；`with T.rs()` 块内局部量块外不可见（标量常量提到 Scope 顶层）。条件 alloc 规则与 developer 谱系 v3_deadbuf 教训跨工具链二次一致。复现：`python /tmp/opencode/probe_forms.py on`（四形态逐一验证，session-local；镜像于 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md` D4）。

**〔第六轮补充实证，2026-09-16，tilelang a13585dc + CANN 8.5.0，origin_task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T041000Z〕**精确折叠边界修正（上文「一律 runtime if」过粗）：**纯名 Python bool 的无-else `if X:` 在 trace 时折叠**（第五/六轮 `if band_free:` / `if use_pipe2:` 均产出单 trace，实证两次）；**else 分支的存在破坏折叠或作用域**——`if use_pipe2: slot = task_id % 2 else: slot = 0` 报 Undefined variable（即使条件是纯名常量），绕法 = factory 层 Python int 常量 + 单算术表达式 `slot = (task_id % 2) * pipe2_on`（r7f Vec slot 报错→算术消解通过；r8g 生成脚本同型失败三证）。runtime 条件 `if task_id + 1 < N:` 内的 T.serial/rs/变量定义合法（r7f Cube 段编译通过）。

---
id: TRAP-expert-v-operands
kind: trap
family: [expert, attention]
apis: [T.vcmp, T.vbrc, T.vtanh]
dtype: [fp32, int16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro-missing
---

### Expert v 算子操作数规则

`T.vcmp` 标量操作数拒绝 `tir.Cast`（codegen `NpuirOperand::FromExpr cannot handle the expr with type of "tir.Cast"`）；**〔2026-09-15 第五轮实测补充，工具链 6797758〕vcmp 在 int16 上对所有操作数形态（运行时标量 PrimExpr / 文档合法的同 shape int16 张量）一律降级为标量指令执行**（编译警告 `will execute by scalar instruction with low efficiency`，[32,256] 实测 **236µs/链**）——在 causal mask 链上占整 kernel 壁钟 ~45%（aiv_scalar 94–97% 的主成分，与 flag 自旋同形，须查编译警告区分）。**绕法 = 算术惩罚掩码**：`scr = clamp(tpl - thr, 0, 1)`（vmax/vmin 标量广播，文档合法）→ `S += scr·(-1e38)`（f32 大数吸收，masked 位精确变 -1e38，与 vselect 假支逐位一致；NaN 无法被算术掩埋——stale 列/band 垃圾须 vselect 选择语义或源头零初始化）。标量 PrimExpr B 操作数合法有先例（`examples/deepseek_v4/example_sparse_attn_kernel_highperf.py` L353）；`T.vbrc` 标量源须 let 绑定局部量（字面量触发 rank 断言）；`T.vtanh` 需 fp32 同 dtype 操作数（int16 scratch 不可复用，MLIR verify 报错）。**v-op 切片仅接受静态边界**（运行时 Var 触发 parser `int(Var)` 拒绝，P10 探针）；**脚本解析器只折叠纯名常量 if**（`not X`/`X and Y` 表达式生成 TIR runtime if，须预组合纯名）；**条件 T.alloc 永不折叠**（含常量条件，v3_deadbuf 的强化实证——用宽度切换 `w = X if cond else 16` 替代）。证据：debug_log.md D4/D5 + 本轮任务工作区 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/`（溯源：probe_vcmp_forms.py P6 / probe_forms.py P1-P5 / probe_rt_slice2.py P10 / opt_log.md Round 2）；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915（第五轮 Stage 4，工具链 6797758）。**〔2026-11-01 第三现，5b062018 + CANN 9.0.0 / 910B3〕** int32/bool 二维列掩码（decode `[16,256]`，非 causal band）同样降级：`aiv_scalar_time` 34.85 vs `aiv_vec_time` 0.92 µs（37×）。三现的共同条件是**int32 比较操作数**——GQA-bwd 把 int32 换 fp32 即消除告警（3.47×），decode 任务改用算术掩码并复用既有 int32 buffer（零新增 UB，7.6×）。细节与复现见 attention.md PL-1.11 第三现段。

**〔第六轮补充实证，2026-09-16，tilelang a13585dc，origin_task 同上〕**v-op 操作数拒绝**带 runtime 首维索引的多维 buffer 切片**：`T.vlog2(ub_ell_snap[st, 0:half, 0:1], ...)`（snap 为 [2,half,1]，st 是 runtime Var）在 codegen `getBroadcastDim` 报 `buffer_shape0.size() == buffer_shape1.size()` 检查失败（3 维 vs 2 维不折叠）；`T.vmul(ub_m_snap[st, ...], ...)` 同型。绕法：**先 `T.copy` 拷回单槽 buffer 再喂 v-op**（plain `T.copy` 的源/目标接受 runtime 索引切片——`ws_s[kernel_id, slot, ...]` 先例，r8g epi 修复实证）。与既有「v-op 切片仅静态边界」（P10）同族：静态边界 + runtime 首维索引亦不可用。
---
id: TRAP-transpose-epilogue-poison
kind: trap
family: [attention, expert, general]
apis: [T.transpose]
dtype: [fp16, f16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 + Ascend910B2C，2026-09-09
repro: repro-missing
---

### 活跃源 `T.transpose([N,1]→[1,N])` epilogue 毒化整 kernel（2.6x）

epilogue 位置的 transpose 若其源为活跃生产者（被前序 v-op 写过且结果被消费），整个 kernel 管道重叠崩塌（实测 fa4096 98→255µs 级）；单 op 无罪（vlog2/vmul/vadd/死源 transpose 各自 97µs 级）、4 个平凡 vadd 无罪——**组合阈值效应，pass 层根因未定位（未文档化假设，依据 = 14 点 morph 实测矩阵）**。绕法：目标 GM 张量增维视图（如 [B,H,S,1]，与 [B,H,S] 同内存）+ UB→GM 自然 2D 区域拷贝（`ub[0:real_m,0:1] → dst[·,·,bx:bx+real_m,0]`）+ host 侧 reshape 还原契约形状——零 transpose、零拷贝。**全谱系存量税**：v9 级 per-block kernel 同样中招（226.72→181.46µs，-20%，perf-only 探针）。证据：`perf_opt/opt_log.md` round 11 morph 矩阵（14 点二分）+ perf_feedback.md 修正附录 + `profiles/round11/`、`profiles/final3/` raw；复现：morph 阶梯脚本形态与 14 点实测矩阵镜像于 opt_log round 11 Diagnosis（scratch `/tmp/opencode/ref_morph.py`，session-local），终值对账 `msprof op --kernel-name=_gqa_prefill_fwd_main_mix_aic --launch-count=20 --warm-up=5`（median of 20）跑 bench `--use-default-config`。

---
id: TRAP-UB-dynsubview-dominance
kind: trap
family: [compiler, ub]
apis: [T.copy, T.alloc_ub]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### task 级 UB 行 buffer + 嵌套循环动态偏移 subview → auto-multi-buffer 非支配 IR

- **形态**：UB buffer（如 [1,Q] 因子行）在 task 级装载、在内层循环中以动态偏移切片消费（`T.copy(ub_row[0:1, s0:s0+ts], blk[0:1, 0:ts])`，s0 为循环变量派生）——Q≥128（切片非全宽）时 BiSheng `--enable-auto-multi-buffer=true` 产出非支配 IR：`error: operand #2 does not dominate this use`（位于 task 级 GM→UB copy 行）；Q=64（单切片=全宽）不触发。
- **绕法**：①退回 per-block GM 直载（基线形态，多 ~12 次 tiny GM 读/任务）；②切片偏移静态化（trace 分派）；③尝试关 auto-multi-buffer（未在本任务验证——会全局改变 L1/UB 多缓冲行为，风险自担）。
- 关联：PL-1.9-hardlimits「v-op codegen 拒绝多维 UB 切片操作数」——本条是 copy 路径 + multi-buffer pass 的同族表现。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/logs/round1/v3_hoist_L0.log`（报错全文）；npuir 现场见 opt_log R1 v3 行分析。

---
id: TRAP-parallel2d-bm-ge2-scalarize
kind: trap
family: [reduction, elementwise]
apis: [T.Parallel, T.if_then_else]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_parallel2d_bm2_scalarize.py
---

### 2D T.Parallel(bm,N) 融合 if_then_else 在 bm≥2 整体标量化（与操作数形态无关）

`for i,j in T.Parallel(bm,N): cand[i,j]=T.if_then_else(x[i,j]==ext[i,j], T.cast(j,"float32"), BIG)` 在 bm≥2 时 scalar ratio 0.84–0.95（(2,4096) 222µs）；bm=1 同形态向量化正常（tiled (1,5120) 25.9µs）。**操作数变体对照证伪旧归因**：(bm,1) 直接索引 m[i,0]（0.95）、ext_brc 物化 + x_work staging（0.948）——两者同样标量化，触发器是循环形态本身。绕法：bm≥2 用显式向量 op 链（vbrc/vcmp/vselect，r1a 形态 47µs）或 per-row (1,W) 展开循环（PL-1.20）。证伪更正留痕：argmax opt_log Iteration 4（session-1 嫌疑 (a)/(b) 均否）。

---
id: TRAP-vbrc-same-shape-empty-broadcast
kind: trap
family: [general]
apis: [T.vbrc]
dtype: []
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_vbrc_same_shape.py
---

### vbrc 同形 src/dst → 空广播维数组 MLIR verify fail

`T.vbrc(idx_row (1,N), idx_j (1,N))`（src 与 dst 完全同形，无尺寸差维）→ `'hivm.hir.vbrc' op have empty broadcast dims array` 编译期 verify 失败。绕法：同形场景直接用源缓冲（或 T.copy）；vbrc 仅在存在尺寸 1 的差维时使用。

---
id: TRAP-vselect-inplace-carried-state
kind: trap
family: [reduction, general]
apis: [T.vselect, T.copy, T.serial]
dtype: [fp16]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/TRAP-vselect-inplace-carried-state.py
---

### 原位 T.vselect(cond, A, B, B) 在多迭代 T.serial 循环丢失 loop-carried 状态

在线递推（running (max, idx) 类）跨 tile 级联时，`T.vselect(cond, chunk, running, running)`（false 操作数同时是输出）在多迭代 `T.serial` 循环中静默丢失状态——最终输出停留在 tile-0 局部值。触发边界：单迭代（num_full==2）不受影响，num_full ≥ 3（≥2 次更新迭代）才暴露。实测（argmax (4,102400) tn=5120、num_full=20，fp16）：inplace 得 `[4779,3248,2345,1833]`（全为 tile-0 局部索引，真值 `[31919,3248,44964,44710]`）；同 kernel 无别名形态 bit-exact 与 torch.argmax 一致。绕法 = select 到独立 scratch + `T.copy` 回写（多一对 (bm,1) 微缓冲的可忽略代价）。诊断指纹：输出全落 tile-0 局部索引 ⟹ loop-carried 状态未携带。

---
id: TRAP-reduce-mixed-dtype-silent-corrupt
kind: trap
family: [reduction, elementwise]
apis: [T.reduce_max, T.reduce_min, T.reduce_sum]
dtype: [fp16, bf16, fp32]
device: 910B2C / 910B3
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z / TileOPs-_engram_decode_kernel-20261030T121905Z（第二证：reduce_sum）
toolchain: 首证 tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C；第二证 tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3 / 2026-10-30
repro: repro-missing
---

### reduce 的 src/dst dtype 不一致时静默产出错误值（reduce_max/min 哨兵极值、reduce_sum 量级漂移）

`T.reduce_max` / `T.reduce_min` / `T.reduce_sum` 在 dst dtype ≠ src dtype 时不做隐式 cast，也不报错——实测两种情况：

1. **极值归约（首证，argmax 任务）**：fp16 src → fp32 dst 实测 extreme 全落 `2^30` 哨兵级垃圾值（`[0.75,117,13,−152]` vs 真值 `[3.09,2.81,3.98,2.89]`），索引主路径同时正确——易误判为搬运/索引链缺陷。`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.4 的 `dtype≠accum_dtype` 示例与实现矛盾（文档示例不可信）。
2. **求和归约（第二证，engram_decode 任务 round 2）**：fp16 暂存 + fp16→fp32 `T.reduce_sum` 在完整 kernel 内编译期报 `memref.expand_shape ... found 1, expected 2`；同形态最小探针**编译通过但静默算错**（fp16→fp32 的 abs-sum 3.73e6 vs fp16→fp16 的 886，差 3 个数量级）；fp16→fp16 正确但精度仅 ~1%。因此「半精度暂存 + 混合 dtype 归约」这条向量加速路线在本工具链不可用（dtype 表也无 bf16——bf16 源先 `T.vcast(..., round_mode="rint")` 升 fp32 再 reduce）。

**纪律**：reduce 链一律 src/dst 同 dtype（需要 fp32 精度时先 vcast 到 fp32 再 reduce，reduce 后再 cast 回）；**诊断指纹**——极值列全落哨兵/异常量级 + 索引主路径正确 ⟹ 混合 dtype reduce，而非搬运/索引链缺陷；求和场景表现为量级级漂移（+10^n）而非符号翻转。

---
id: TRAP-parallel-scalar-broadcast-segv
kind: trap
family: [gemv, reduction, elementwise]
apis: [T.Parallel, T.alloc_shared]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: engram_decode_mcore-engram_decode_mcore-20261031T012849Z
toolchain: tilelang 0.1.2+3ed7c91420e925ff3ec388fb981dd6ebf45ff4bf + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-10-31；TILELANG_ASCEND_MODE=Developer
repro: repro/TRAP-parallel-scalar-broadcast-segv.py
---

### `T.Parallel` 体内内联读 `(1,1)` UB 标量元素 → codegen 静默 SIGSEGV

**现象**（910B3 / 0.1.2+3ed7c914 / Developer）：把 `(1,1)` UB 标量的元素直接写进 `T.Parallel` 体内的逐元素表达式

```python
for j in T.Parallel(D_s):
    vn[j] = cv[0, 0] * v_c[j] * wv[j]        # cv 为 (1,1) shared/UB buffer
```

编译期 codegen 进程直接 SIGSEGV（子进程 rc = −11），**无 traceback、无 IR 诊断文本**，Python 侧只看到进程被杀。

**为什么容易误判为等价**：同一个 `buf[0,0]` 在**标量语句**里长期合法且在库基线中在用——本算子族基线的 `sh[0,0] = sqrt_tmp[0,0] / sh[0,0]` 就是此形态；「标量语句合法」不能外推到「`T.Parallel` 体内合法」。

**绕法（无实测代价）**：循环外先把标量绑成 Python 名（`cval = cv[0, 0]`），体内用 `cval`；改后同 kernel 与 golden 逐位一致（`max_err = 0.0`）。同类改写适用于 `(1,1)` 上算出的倒数 / `vsqrt` / `vsigmoid` 结果（本算子 apply 段两处内联读——`cv[0,0]` 与 `al[0,0]`——均为该形态）。

**诊断指纹**：编译进程 SIGSEGV（−11）+ 日志停在 codegen + Python 无异常栈 ⟹ 先查 `T.Parallel` 体内是否内联读了标量 buffer 元素。与 `TRAP-vbrc-scalar-shared`（vbrc 标量→shared 不可用 / serial 变量条件 select segfault）同属「标量进向量域」家族，但触发动作不同：本条是**读**、且发生在 `T.Parallel` 体内。

复现：`python repro/TRAP-parallel-scalar-broadcast-segv.py`（BAD 形态子进程 rc=−11 断言；GOOD 形态 bit-exact）。

---
id: TRAP-primfunc-static-branch-and-loop
kind: trap
family: [general, reduction, elementwise]
apis: [T.prim_func, T.serial, T.Parallel]
dtype: [fp32]
device: 910B3
status: verified
origin_task: engram_decode_mcore-engram_decode_mcore-20261031T012849Z
toolchain: tilelang 0.1.2+3ed7c91420e925ff3ec388fb981dd6ebf45ff4bf + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-10-31；TILELANG_ASCEND_MODE=Developer
repro: repro/TRAP-primfunc-static-branch-and-loop.py
---

### `T.prim_func` 体内 `for p in range(k)` 不按 Python 展开：符号下标 + 链式比较 / Python 元组迭代均被 parser 拒

**现象**（910B3 / 0.1.2+3ed7c914 / Developer）：dilated-causal 卷积的 tap 循环按设计写成「编译期 Python 分支 + Python range 展开」，在 prim_func 体内有两种失败形态：

1. `for p in range(n): idx = L - (w-1-p)*dil; if 0 <= idx < L:` → `Literal value N exceeds maximum of bool`。根因是 **`range(k)` 在 prim_func 体内不是 trace 期展开**，`p` 是符号 Var，`idx` 因此是符号表达式，链式比较被降成 bool 转换后越界。
2. `for _p, idx in taps:`（`taps` 是 factory 里算好的 Python 元组）→ `Expect the for loop to be one of the following: range, T.serial, T.grid, T.parallel, ...`——parser 只接受这几种循环形态，Python 可迭代对象不在此列。

**绕法**：把**守卫语义**在 factory 层折成编译期常量，体内只留静态 trip count + 恒合法的 affine 行号。本例：合法 tap 是后缀区间 `p ∈ [p_start, w-2]`，`p_start = max(0, w-1-L//dil)`、`n = (w-1) - p_start`；体内 `for t in T.serial(n): p = t + p_start; x[L-(w-1-p)*dil, :]`——无符号 `if`、无元组迭代，且把 `L < dil*(w-1)` 的守卫（丢弃越界 tap）吸收进 `p_start`。

**对设计措辞的更正**：设计文档写「tap 循环在 codegen 期展开 / 编译期 Python 分支」在本工具链不成立——`range(n)` 展开出来的是常数 trip count 的**符号循环**，不是 Python 迭代；能在 factory 层算出的只有守卫**语义**（`p_start` / `n` / 行号），不是循环体本身。设计与检视引用该措辞时按本条改写。

**与既有条目的关系**：`TRAP-tvm-parser-rules` 记四条硬规则（if / 三元式 / 条件 alloc / `T.rs` 作用域），本条补「循环形态与静态性」；此外仓内文档曾引用条目 id `TRAP-C13-parser-no-python-unroll`（见 capability-gaps.md CG-2026-0018 的登记簿一致性注），该 id 一直无落盘条目——本条即该死链所指现象的实测落盘。

复现：`python repro/TRAP-primfunc-static-branch-and-loop.py`（两个 BAD 子进程断言诊断文本；GOOD 三档 bit-exact，含 `L < dil(w-1)` 守卫档）。

---
id: TRAP-npuir-vector-composite-rhs
kind: trap
family: [general, elementwise, expert, spectral]
apis: [T.Parallel, T.vsub, T.vmul]
dtype: [fp32]
device: Ascend910_9362
status: verified
origin_task: fft_c2c_vec-fft_c2c_vec-20261031T103133Z
toolchain: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362（Expert 模式）/ 2026-10-31
repro: repro/TRAP-npuir-vector-composite-rhs.py
---

### `T.Parallel` 体复合 RHS 静默写 0（编译成功、结果全零）

**实测事实**：`for i in T.Parallel(h): dst[0,0,i] = (a[i]-b[i])*wr[i] - (c[i]-d[i])*wi[i]`（单个 RHS 含 7 个运算）编译输出 `AscendNPU IR compile success`，运行后 `dst` 逐元素为 **0.0**；同一算式拆成 8 条「每句 1 个 op」、每条各用内核自己声明的 3-D UB 中转后 fp32 精确（err 3e-8）。三个独立探针同向复现（复合版 `dr = 0.0`、单 op 版 3e-8；同 workload 的 MSE 2.77 vs 0）。

**机制（推断，未定位到具体 pass 行）**：RHS 被 `NpuLoopVectorize` 内 `DecomposeExpression(tmp_bufs)`（`src/transform/npu_loop_vectorize.cc`，`VectorizeSingleStatement` 的 tmp_bufs 路径）拆出临时量，临时量被 UB 规划落在活跃缓冲之上，乘链读到零。可用的合法形态 = **每个 RHS 恰好一个 op** + 每个中转是内核显式 `T.alloc_ub` 的 buffer（不依赖派生临时量）。

**诊断指纹**：编译无警告、无报错，但整块输出恒 0（区别于 `TRAP-reduce-mixed-dtype-silent-corrupt` 的哨兵值/量级漂移，也区别于 `TRAP-shared-row-broadcast-affine-index` 的错址——后者错值无固定模式）⟹ 先按本条目把 RHS 拆成单 op 复核。

**代价（实测）**：拆单 op 后 `fft_c2c_vec` 复数乘按 `tr/ti -> pv/qv -> dif` 展开，每 butterfly 的语句数由基线 4 条膨胀到 10 条、另加 4 次中转 UB 往返；在低 lane 级上该膨胀直接吃掉向量化收益（净时延见 queue VP-2026-0164）。

复现：`python repro/TRAP-npuir-vector-composite-rhs.py`（复合分支打印 2.77 量级误差、单 op 分支断言 1e-5 内）。

---
id: TRAP-npuir-vector-lut-alignment
kind: trap
family: [general, spectral, elementwise]
apis: [T.Parallel, T.vmul]
dtype: [fp32]
device: Ascend910_9362
status: verified
origin_task: fft_c2c_vec-fft_c2c_vec-20261031T103133Z
toolchain: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362（Expert 模式）/ 2026-10-31
repro: repro/TRAP-npuir-vector-lut-alignment.py
---

### 向量操作数基址非 32 B 对齐 / 越界读 → 运行期 Vector core abort

**实测事实**：Stockham twiddle 的基线布局 `LUT[h-1+i]` 使每级段起点为 `h-1`（`h ≥ 2` 时恒为奇数 → 4 B 粒度）。以该段作 `T.Parallel` store 的向量操作数（`dst[0,0,i] = t[0,0,i] * lut[0,0,h-1+i]`）时编译通过，运行期 abort：`The write address of the MTE instruction is out of range. CCU instruction address check error. ... [vector core exception]`。同一 kernel 改到 `base % 8 == 0` 的对齐段后 fp32 精确。

**触发边界（实测）**：隔离最小例只复现「对齐时正确」，未复现 abort ⟹ 触发条件含**分配尺寸非 32 B 倍数导致的越界读**（不只基址奇偶），故缓解形态（每级段 32 B 对齐 + 表尾补齐至 32 B 倍数）对该类广播操作数无条件生效——不能靠「基址恰好对齐」规避。

**工程形态与代价**：对齐后 LUT 物理尺寸 > 逻辑 `(n-1)`（本例 `npad ≥ n-1`、段首填 pad），pad 元素只在表尾段被读到且不参与运算；这是 fft_c2c_vec 实现相对 DESIGN §1.2/§3.3 flat `(n-1,)` 布局的已披露偏差。

**诊断注意**：该 abort 杀进程，进程内 try/except 捕获不到——探针用子进程跑并用返回码/输出文本断言（repro 的 `--unsafe` 分支即此形态）。同族参照 `TRAP-UB-dst-align`（dst 非零切片 + 32 B 倍宽度）：那条是写侧对齐，本条是**读侧操作数**段基址与分配尾部越界。

复现：`python repro/TRAP-npuir-vector-lut-alignment.py`（对齐分支断言精确）；`python repro/TRAP-npuir-vector-lut-alignment.py --unsafe`（子进程断言 abort）。

---
id: TRAP-l1-4th-live-buffer-copy-segv
kind: trap
family: [expert, mixcv, attention, cube, double-buffer]
apis: [T.alloc_L1, T.copy, T.clear, T.gemm]
dtype: [fp16, bf16]
device: Ascend910B3
status: verified
origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z（Stage 4 Round 10 / M1；2026-11-04 蒸馏 D2 归位，原误标 …T190000Z）；Round 11 编译探针复核 TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z
toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3（Expert 双 Scope）/ 2026-11-04（Round 11 同戳）
repro: repro-missing
---

### 第 4 块「活」L1 缓冲的首条 GM→L1 `T.copy` → bishengir-compile SIGSEGV；同一缓冲 `T.clear` 合法

**实测事实（探针阶梯，同 session 同一工具链，compile-only gate）**：在一个已用 3 块活 L1（Q 16 KB + P 32 KB + K/V 64 KB = 112 KB）的 Expert 双 Scope attention kernel 上：

| 改动 | 活 L1 | 结果 |
|---|---|---|
| 参照（当前最优） | 3 / 112 KB | **26–42 s 编译通过，0 失败** |
| +第 4 块 64 KB + `T.copy(zbuf, l1_b2)` | 4 / 176 KB | 400 s，5 次 SIGSEGV，0 成功 |
| +第 4 块 4 KB + `T.copy(zbuf[0:16, 0:dim], l1_pad)` | 4 / 116 KB | 72 次 SIGSEGV，0 成功 |
| +第 4 块 32 KB，**从不使用** | 4 / 144 KB | **26 s 通过**（死分配被消除） |
| +第 4 块 64 KB + **`T.clear(l1_b2)`** | 4 / 176 KB | **26 s 通过** |
| 上条 + **一条**真实 `T.copy(v…, l1_b2[…])` 放进原块循环 | 4 / 176 KB | 600 s，28 次 SIGSEGV |
| 独立最小 kernel：4 块活 L1 + 2 gemm + 4 条 GM→L1 copy | 4 | **37 s 通过** |

**判据（三条互证）**：① 与容量无关——116 KB 失败、176 KB 的 `T.clear` 形态通过，且本前端对 cbuf 超量本就无容量诊断（见 `CONST-capacity`「L1 不可测」段）；② 与「4 块 L1」通例无关——独立最小 kernel 同形态通过 ⟹ 与本 kernel 的 IR 体量/依赖图交互；③ 触发点 = **「目的地址是第 4 块活 L1 缓冲」的第一条 GM→L1 copy**（A1/A3/OL 的 SIGSEGV 次数 5 / 72 / 28，0 次成功——原 opt_log 记「0/38」与本条探针表逐项和不符（5+72+28=105），此处以可核对的逐项值为准），而 `T.clear`（非 copy 路径）与「未使用的第 4 块」都不触发。失败形态是 `err code: -11`（SIGSEGV，落在 ConvertHFusionToHIVM/ConvertHIVMToStandard），与 bm=80 UB 溢出的 SIGSEGV 指纹相同但成因不同（那条有 `ub overflow` 依据，本条无任何诊断输出）。

**同族第二形态（body 复制的编译不收敛）**：改用 3 块活 L1 绕开第 4 块后，把 Cube 内层块循环按**奇偶成对展开**（稳定对 + 剥离末对 + 奇尾，或单 pair body + 钳位前瞻）以静态化双缓冲的缓冲选择，`bishengir-compile` **不收敛**：25 次记录中 1 次 SIGSEGV + 其余 >15 min @100% CPU（基线 26–42 s）。⇒ 在本 kernel 上「成对展开的 Cube 循环体」本身即是编译封堵点。

**Round 11 收敛：封堵限特定结构，不是双缓冲意图本身（同一算子，编译探针 dim=128）**：`nqb=2`（相邻两 q-block 共享 K 装载）的另一种结构在**不增加活 L1 块数、不做 trace-time 成对展开**时编译通过——`l1_q` 取 `[2*bm, dim]` 装两个 q-block 的 Q（packed-NHD 行连续，一次 GM→L1 copy 填满）；pass-1 每 band step 只有**一个 body**（一次 K load + 一次 `M=2*bm` 融合 gemm + 一次 S store，两 q-block 的 S tile 是 L0C 相邻行）；pass-2 只有**一个 body**，放在 `for slot in T.serial(2)` 内用**运行期 `slot`** 选 P/O 行窗口。实测 dim=128 冷编译+首启 **8.2 s**（msprof run 日志 bench 首调用 9.2 s、wall 31.3 s 含采集），同基线 26–42 s，**0 SIGSEGV**；L0 的 dim=64 冷编译 ~29 s。因此本条封堵边界读作：①「目的地址是**第 4 块活 L1 缓冲**的首条 GM→L1 `T.copy`」；②「Cube 内层块循环的 **trace-time 成对展开**」——不是「任何 L1 双缓冲意图」或「任何循环体规模翻倍」。同轮该 nqb=2 形态只共享 K、V 仍逐 qb 载入且未带 depth-2 流水，8b-long fp16 单 run（1 launch）**2282.1 µs vs 基线 2003.35 µs（+13.9%）** ⇒ 可编译不等于有收益；其 dim=64 形状另触发**运行期** L0A 越界（`CG-2026-0030`，与本节编译封堵点不同）。

**工程含义（何时先做编译探针）**：手写 L1 双缓冲（`PL-1.18` 的 L1 端口互拖是性能层否决，本条是**可行层**否决）在 Expert 双 Scope + 大 IR kernel 上要先过「第 4 块活 L1 的 copy 探针」与「展体探针」两道编译门，再谈收益。任何「用第二块操作数缓冲做 push/pull 重叠」的方案在此 gate 前不该进性能实验。**零初始化的等价写法**：本 kernel 用「zero 参数 + `T.copy(zbuf, buf)`」做 NaN 防护的零初始化时，`T.clear(buf)` 走不同编译路径（C1 通过 vs A1/A3/OL 的 copy 路径失败）——同一 init 语义下 `T.clear` 是该形态的可编译写法，`T.copy` 不是。

**repro 缺口（Tier 1，repro-missing 形态）**：触发依赖该 kernel 的 IR 体量——独立最小 kernel 同形态不复现（上表 4L1），故本轮只能给出「delta 骨架 + 探针阶梯证据」，最小化载体尚未得到（`probes/_probe_onelive_c.py` 是最小 delta 的完整形态：当前最优 + 1 块 L1 + 1 条 `T.copy` + `T.clear`；任务工作区路径，provenance 允许失效）。

**代价/证据**：A1/A3/OL 三形态合计 105 次 SIGSEGV、0 次成功+ 2 种 3 缓冲展体形态各 >15 min 不收敛；唯一通过编译的零新增缓冲形态（把 GM→L1 的独立装载提到跨引擎 `sync_block_wait` 之前）实测 +1.4%（两个 long 同向），说明该 kernel 的重叠缺口不在「MTE2 空转」而在 L1 端口读写互拖。
