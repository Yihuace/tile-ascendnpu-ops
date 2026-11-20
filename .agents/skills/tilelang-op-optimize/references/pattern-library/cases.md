# 案例索引与参考实现集（同类问题的正/反例参考）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）。
>
> **用途**：按"哪类问题该参考哪个算子目录"组织 `examples/` 的检索入口。**目标未上库的条目标注「任务工作区，未上库」并给出 durable 载体**（该经验的持久形态 = pattern-library 自包含条目 / repro）——工作区目录随时可能被清理，条目知识不随之失效（ED-A）。Stage 1 设计（强制步骤 0.5）与 optimize skill Phase 0 均会读取本文件——命中时优先精读对应算子目录，优先于盲目 Glob。**维护**：由 `tilelang-skill-evolver` 在任务终态追加（C 类，Tier 0）；条目 = 路径 + 一句话 + 适用触发条件，不复制内容；路径失效或长期零引用 → deprecate。
>
> **反例档案**（BLOCKED 终态任务的根因链 + 能力缺口互链）见本文件末节——capability-gaps（`.agents/evolution/capability-gaps.md`）的 open 条目实质是"该形态当前做不了"的反例，以指针行互链（只加指针不复制内容，边界按该簿 §与 pattern-library 的边界）。

---
id: CASE-pool-maxpool3d-tileops
kind: case
family: [pool]
mode: [developer]
dtype: []
status: verified
origin_task: max_pool3d-harness-migration-2026-08
toolchain: 2026-08
repro: none
---

### `examples/TileOPs/tileops/kernels/pool/max_pool3d/max_pool3d_kernel/`（任务工作区，未上库）

**durable 载体**：本文件无自包含池化条目时见 layout.md PL-1.1/1.2/1.4（模式与实测数字自包含）；集成包结构形态见已上库的 mish/logsumexp 同构目录。

TileOPs 集成包形态：kernel + `{func}_DESIGN.md` 快照 + 聚合 `__init__.py` + integration_log。适用触发条件：migration-harness 集成结构参考；Stage 5 集成问题排查。

---
id: CASE-elementwise-mish-opt
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, fp32]
status: verified
origin_task: mish-optimize-2026-08
toolchain: 2026-08（详见其 opt_log 版本戳）
repro: none
---

### `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/`

elementwise cast 类完整调优档案（persistent 甜点、multi-buffer 教训、Expert 流水平台约束、run 双态协议）。适用触发条件：elementwise 调优前必读先例；<3% 结论多 run 协议。

---
id: CASE-reduction-logsumexp
kind: case
family: [reduction]
mode: [developer]
dtype: []
status: verified
origin_task: logsumexp-migration-2026-08
toolchain: 2026-08
repro: none
---

### `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`

规约类（logsumexp）独立算子目录：DESIGN.md + kernel + 分层测试。适用触发条件：规约类（水平归约/online 单遍）设计与 L0 测试计划参考。

---
id: CASE-elementwise-lerp-opt
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
repro: none
---

### `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/`

elementwise 3 输入 persistent 调优：copy-floor 探针标定法 + MTE2 带宽退化曲线 + 交错 A/B 采纳复核 + grid-stride vs 连续分块反例。适用触发条件：判定"已贴内存地板"时的方法学参考；多输入 elementwise 调优。

---
id: CASE-elementwise-lerp-precision
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, bf16]
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: tilelang-mlir-dev dev root build（2026-09-07）；golden torch 2.9.0+cpu
repro: none
---

### `examples/lerp_tensor/_make_lerp_tensor_kernel/`（任务工作区，未上库）

**durable 载体**：traps-runtime.md TRAP-fp16-opmath-golden（现象/对齐通解/数字自包含）。

fp16 精度失败定征与修复完整档案：attempt-1 原生域失败基线（`history_version/`）+ R2 fp32 中转修复（≤1 ulp，全量 62 PASS）+ kernel 头部 Implementation Notes 等价性论证。适用触发条件：fp16/bf16 逐元素迁移精度失败（golden opmath 域分歧）定征与修复参考；vcast 中转路径 UB 混合字节预算复算案例。

---
id: CASE-pool-maxpool3d-standalone
kind: case
family: [pool]
mode: [developer]
dtype: []
status: verified
origin_task: maxpool3d-2026-08
toolchain: 2026-08
repro: none
---

### `examples/maxpool3d/_max_pool3d_kernel/`（任务工作区，未上库）

**durable 载体**：layout.md PL-1.1/1.2/1.4（池化类模式与实测数字自包含）。

standalone 形态的池化算子目录（DESIGN.md + kernel）。适用触发条件：窗口/池化类设计参考（分核策略、边界处理）。

---
id: CASE-attention-gqa-expert-full
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: none
---

### `examples/multi_head_attention/_gqa_prefill_fwd_kernel/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.8（bf16 直连）、traps-compiler/runtime.md attention 族陷阱条目（正文自包含）。

Expert 模式 GQA/MHA prefill attention 完整档案：v0→v1→v2 设计修订链（K_A ceildiv→floordiv 差 1 反例 + 检视建议自我纠正全程 + per-lane 阈值符号错误）+ 3 轮 REVIEW + debug_log D1–D6（load_nd2nz 误读 / T.copy 语义 / 解析器规则 / tier-3 精度分类）+ bf16 直连 + E7 守卫不变式。适用触发条件：attention 族 Expert 迁移/重设计；跨引擎 flag 流水协议设计；bf16 Cube 直连决策；分界公式检视反例。〔2026-09-15 注：该目录 DESIGN.md/REVIEW.md 已被两相位 v3 重设计覆盖（见 CASE-attention-twophase-causal-regen）；v0→v2 修订链与 E1-E7 单遍设计存档于 history_version/（design_v0/v1/v2.md），debug_log.md（含 D1–D6）仍在本目录——引用本条修订链/检视反例时以 history_version/ 为准。origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z〕

---
id: CASE-attention-expert-stage4
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, f16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z + 20260909T033622Z + 20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08～09（三轮）
repro: none
---

### `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.9-blockwidth/hardlimits/twophase（三轮结论自包含）+ `repro/PATT-twophase-restructure.py` + constants.md 硬件常数表。

attention 族 expert Stage 4 完整调优档案：块宽摊减调优序（bn 64→512）、cbuf=512KB 实测、wide 单槽 L1 模式、clean trace 特化、S-f16 传输裁决、G=2 备选 A 裁决、H=1 任务平衡公式、工厂内 tuned 分派表（S4-5 模式）；第二轮硬上限测绘（L0C 128KB / L1 端口 ~148–154GB/s/核 / 向量发射 ~0.5µs/op / fabric ≥1.73TB/s）+ f16 softmax 链（终值 17.55/27.24/69.13/226.72µs）+ [DESIGN_LIMIT] 五段式反馈档案（perf_feedback.md，硬目标 2.09x/2.27x 缺口归因）；**第三轮两相位重构达标**（参考实现 flash_attn_npuir.py 结构迁移 + transpose-free lse，终值 12.94/18.11/30.88/98.05µs 四硬目标全达成、fa1024/2048/4096 优于参考、三轮累计 vs Stage-3 基线 -77.2%/-82.2%/-91.8%/-91.1%）+ [DESIGN_LIMIT] 负结论被同 API 结构推翻的完整闭环档案（perf_feedback.md 修正附录：局部地板 vs 绝对地板界定 + 两处根因 + morph 阶梯 14 点矩阵）。适用触发条件：attention/persistent 双 Scope 类调优前必读；跨引擎流水结构选型（块宽 vs 流水深度）；备选 A/B 裁决执行范例；设计层天花板判定与 [DESIGN_LIMIT] 归因/证据组织参考；**结构不可达类负结论的修正闭环参考**。

---
id: CASE-attention-developer-stage4
kind: case
family: [attention]
mode: [developer]
dtype: [fp16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
repro: none
---

### `examples/multi_head_attention/_prev_task_20260907_developer_optimize/perf_opt/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.7（expert vs developer 同门对照结论自包含）。

developer 谱系（简单 tiling）同 4 case 调优档案：S4-1 mask 消除 + S4-2/3/4 死缓冲/in-place/退化 trace-time 特化 + S4-5 工厂内 shape 分派（终值 16.1/33.6/105.2/300.3µs）。适用触发条件：与 expert 谱系（attention.md PL-1.9 系列）做同门调优杠杆对照（mask 消除/config 扫描 vs 块宽摊减/L1 上限突破）；编程模式/谱系选型参考。

---
id: CASE-deepseek-v4-highperf
kind: case
family: [attention, expert]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: 2026-09-07
repro: none
---

### `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py`

Expert 模式 attention/跨引擎结构先例：双 Scope + GM workspace 多槽 + per-slot flag + staggered stream + 运行时 if + int16 索引矩阵 vcmp mask 链（L345–354）+ PIPE_V 内 T.arange strides（L611）。适用触发条件：Expert 模式 kernel 结构设计/先例检索（Stage 1 设计起步）；vcmp 标量规则与 flag 协议先例核对。

---
id: CASE-ref-flash-attn-npuir
kind: case
family: [attention]
mode: [developer]
dtype: [fp16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z（第三轮实证）
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-09
repro: none
---

### 参考实现集：`examples/flash_attention/flash_attn_npuir.py`

**参考实现集首条**（T-1 参照锚定）。仓库内同 API 可达结构的参照锚点：两相位（pass-1 全 S / pass-2 全 O_partial）+ 非持久 grid + 单 L0C 顺序复用 + Q hoist + transpose-free lse——第三轮 [DESIGN_LIMIT] 负结论正是被该结构推翻（四硬目标全达成 12.94/18.11/30.88/98.05µs）。适用触发条件：attention 族调优 plateau / 拟产出 `[DESIGN_LIMIT]` 时的**强制参照对照**（perf-feedback.md §1 参照锚定门槛）；两相位结构迁移的母本。

---
id: CASE-norm-adalayern-migration
kind: case
family: [norm]
mode: [developer]
dtype: [fp32, fp16, bf16]
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
repro: none
---

### `examples/ada_layer_norm/_ada_layer_norm_kernel/`（任务工作区，未上库）

**durable 载体**：vrsqrt 陷阱与绕法自包含于 traps-runtime.md TRAP-vrsqrt-plain-precision（含知识域 repro）；迁移决策与双判据验证见该目录 DESIGN.md/verify_equiv.py（溯源，允许失效）。

norm 族首个迁移档案（Developer 模式单函数，零设计修订零重试一次过）：vrsqrt 近似指令精度链定位（TRAP-vrsqrt-plain-precision，分层探针单次运行定位）+ GPU 256-pad 方差校正整体舍弃（恒等式机器验证 0 违反，精确 N 式在小 N 大均值时数值更优）+ verify_equiv 45 案例等价性验证设计（矩式方差 rejection evidence）。适用触发条件：layernorm/rmsnorm/AdaLN-Zero gated 族迁移设计参考（wrapper 已保留 has_gate 分支，gated 变体复用同一集成路径）；Stage 3 精度失败含 rsqrt 段时优先对照。

---
id: CASE-norm-adalayern-stage4
kind: case
family: [norm, reduction]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
repro: none
---

### `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/`（任务工作区，未上库）

**durable 载体**：staging 解耦模式自包含于 elementwise.md PL-1.10 + repro/PL-1.10-loads-first-decoupling.py（知识域）；调优数字溯源 opt_log.md（允许失效）。

norm/row-reduction 族完整调优档案（6 轮，全组几何平均 2.14x）：wrapper 部署态 bm=1 → per-shape 调优表（第一杠杆）+ per-input staging 解耦（PL-1.10-loads-first-decoupling）+ bm 任务几何精搜（bm=7 非单调局部最优）+ 调优表条目 U 曲线闭合验证 + A/B 交错协议两次防误判 + 保护性泛化验证（「UB 85% 松弛律」单点否定）。适用触发条件：norm/规约族 Stage 4 调优；多输入 staging 复用链串行化诊断（pipe 忙时和 ≈ wall 信号）；调优表条目鲁棒性验证方法参考。

---
id: CASE-attention-twophase-causal-regen
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z（Stage 1→3 重生成）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4 调优）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4 调优）
toolchain: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967（Stage 1→3）+ tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD，第五轮）+ tilelang 0.1.2+a13585dc（第六轮），均 + CANN 8.5.0 / Ascend910B2C
repro: none
---

### `examples/multi_head_attention/_gqa_prefill_fwd_kernel/`（任务工作区，未上库）

**durable 载体**：两相位 causal 适配五机制待 queue VP-2026-0057（1/2）；config 依赖性数据已合入 attention.md §1.9-twophase causal 域 bullet（VP-2026-0063 两证，2026-09-15）；第五轮调优知识自包含于 attention.md PL-1.11 + traps-compiler.md TRAP-expert-v-operands/TRAP-UB 第四证 + constants.md CONST-capacity UB 系数；**第六轮调优知识自包含于 attention.md PL-1.12 + repro/PL-1.12-bn-clamp-bm-guard.py**（屏障审计 / 深度 2 / f32 载体 / Ratio 口径 / r9 分派守卫）。

两相位 S/P 物化 v3 重生成完整档案（manifest causal 多头域 Stage 1→3 重跑，用户判定上一版为 case_fa 特化后从设计层重新生成）：DESIGN.md v3（persistent 24 核双 Scope + per-n-block 同 id 三次握手 + E6 flag 预算-bn 耦合守卫）+ verify_equiv.py（13/13 EQUIV_PASS，双跑一致——等价性机器验证脚本结构参照：基线式忠实到常数与运算序 / fp64 参照 / 违反率双口径 / 探针行分离，torch CPU 无 NPU 依赖）+ REVIEW.md 9 维度通过 + Stage 3 `--level all` 29/29 + RETROSPECTIVE.md 2026-09-15 章节（gate 误报修复姿势 / T.ceildiv 空核钳制 / per-core ws 槽 row0 索引 / 装载不满宽掩码门控口径 / dump IR 新 cache 目录）。**第五轮 Stage 4 调优（2026-09-15，4 轮，14.2x）**：判别实验链教科书档案（因子分离 → bn/bm 扫描证伪 → TILELANG_DUMP_IR 编译警告定位 → 微探针定案）锁定 vcmp int16 标量化根因（占壁钟 45%）→ 算术惩罚掩码 + UB diet 宽块解锁 + zbuf l1_b 零初始化；探针族 probe_vcmp_forms/penalty_chain/nan_clamp/rt_slice/l2flake 等（微探针方法论参照）；perf_records.jsonl 36 行 + profiles/{phase1,round2,round3,final}/。**第六轮 Stage 4 调优（2026-09-16，roofline Ratio 硬目标 20%→35% 口径）**：TASKDONE 屏障冗余审计删除（全域 −7~−21%，纯程序顺序论证零实验成本）+ Cube 深度 2 任务流水（flag slot 双槽 2×nk≤15）+ per-shape bm 分派 + fp16 域 f32 ws_s 载体（8b-long 1191.63→989.26µs = −16.8%，vs 第五轮启动基线累计 17.1x；long 4 配置 Ratio 22.64–23.63%——20% 口径达标、35% 口径差 11.4–12.4pt）；r9 precision_fix 分派守卫收窄（bn 钳位域 × bm80 联合 UB 核算判据，S_kv≥3841 域修复，TileOPs pytest 6/6）+ 双镜像同步先例。适用触发条件：causal 多头域两相位/persistent 结构设计参考；causal mask 链标量化排查与判别实验设计参照；verify_equiv 脚本结构参照；per-n-block flag 协议与尾块门控设计；task 级流水与同步屏障审计参照；派发守卫组合的联合 UB 核算参照；msprof roofline Ratio 口径目标调优参照；前谱系（Sep-07 E1-E7 单遍）见 history_version/ 与 CASE-attention-gqa-expert-full，调优轮次史见 CASE-attention-expert-stage4。

---
id: CASE-attention-mha-config-unvalidated
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z
toolchain: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
repro: none
---

### 反例：设计默认 config 路径未经目标域编译验证即出厂（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md` Round 2；任务工作区，未上库——溯源允许失效，根因链要点已自包含于本条）

**durable 载体**：UB 预算偏差实证已合入 traps-compiler.md TRAP-UB-multibuffer-inflation 第四证（VP-2026-0059 两证，2026-09-15，含 ×1.10–1.12 系数 + UB diet 绕法）；能力缺口本体见 capability-gaps CG-2026-0008（recurring）。

反例档案：设计默认 config（E6 替换路径 bn_eff=256）在 dim=128 causal 域的首次编译发生在 Stage 5 bench 且直接 UB 硬溢出（8/10 失败）——根因三层：Stage 3 L1 门禁变体集不含 wrapper-default 派发路径（VP-2026-0065）+ DESIGN §4.5 手工预算低估实际分配 ~26KB（VP-2026-0059 / CG-2026-0008）+ pytest 域全 non-causal 掩盖（causal dim=128 traced 变体从未被 pytest 编译）。修复 = wrapper default_config num_stages 1→2 路由到门禁验证过的逐字路径（VP-2026-0074 config-契约范式），1 attempt 闭环、bench 10/10 复核。适用触发条件：带 config 替换语义 kernel 的集成期 bench 编译失败排查；「pytest 全绿 ≠ manifest 域全绿」的域覆盖核对；宽 config UB 预算校准。

---
id: CASE-ssd-chunkscan-migration
kind: case
family: [mamba, ssd, mixcv]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z / ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z（4515de8 重跑）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z（二轮调优 Stage 4 追记）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z（重建会话：断裂修复 + 三胜出 repro 骨架重推导 + 增量调优 +9.7%）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（npu-smi 26.0.rc1）/ 2026-09-17；重跑 tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20；二轮调优 tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21；重建会话 tilelang 0.1.2+96f287eeaa（与 15ad002 delta 仅 .agents 文档、tilelang 源码零改动）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
repro: none
---

### `examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/` + `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/`（任务工作区 + 集成包）

**durable 载体**：Stage 4 调优知识自包含于 attention.md PL-1.12 update（消费侧前导 set）/ PL-1.13 + repro/PL-1.13-aiv-dup-subid-split.py（双 AIV 分片）/ elementwise.md PL-1.14（ws 块连续）/ layout.md PL-1.15（列广播性能税）/ traps-compiler.md TRAP-UB-dynsubview-dominance / traps-runtime.md TRAP-L1-band-dst-tail-overrun / TRAP-DEVMODE-PERSIST-GEMM + repro（Developer+persistent 崩溃）/ constants.md CONST-mte2 指令维度口径；PL-1.16（Expert 双 Scope 绕法第二证）/ PL-1.17（fp32 中转第二证）/ elementwise 均以本任务为第二证合入。

**〔2026-09-21 4515de8 重跑任务 Stage 4 追记，tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / 910B2C〕**：二轮调优 7 轮 10 分支 plateau，几何 1.078×（w2 223.53→204.33 / w3 638.85→601.61 / w4 3902.26→3616.80µs，msprof op median-of-20 + ab_test）——三项新胜出（prev_states lt 循环冗余重读削除〔旧段数记账漏算 ×4〕/ L0C acc 乒乓配对循环 / vbrc hoist 干净形态）+ 五项新否决（L1 双缓冲二次实证 / 运行时 if 进 Cube 热循环调度毒 w4 +15.65% / x 流头局部最优 / 发射序变体 / acc 深度 4）。知识自包含于 attention.md PL-1.18 update + repro/PL-1.18-floor2-wins.py + constants.md CONST-mte2 段数更正注。stop_reason=plateau；[DESIGN_LIMIT] 双门槛核对立（实测 204µs 落在 DESIGN 估算区间 170–280µs 内，无 >2× 实证替代结构）。

**〔2026-09-21 重建会话追记（task 20260921T120526Z，perf_opt 被外部删除后的断裂修复 + 增量调优），tilelang 0.1.2+96f287eeaa / CANN 8.5.0 / 910B2C〕**：上一调优会话 DONE 后 `perf_opt/` 被外部删除（wrapper 指向 perf_opt 致算子断裂，14 实验分支 + 64 行 records 永久丢失——未提交产物不可恢复实证，queue VP-2026-0117）；本会话 mode=full 重建三步：① Stage 3 基准拷贝 drop-in 修复断裂（factory 签名一致，L0 首编即过 = PL-1.12/1.13/1.14/1.16/TRAP-L1-band/TRAP-DEVMODE 结构在 96f287eeaa〔与 15ad002 tilelang 源码零改动〕存活重验，probe-w2 锚点 217.58 vs 旧 217.65 −0.03% 无漂移）；② 从 `repro/PL-1.18-floor2-wins.py` 骨架重推导三胜出（prevhoist/l0c2x/vbrchoist——ED-A/ED-B 自包含性经工件全失场景检验）；③ 11 个 benchmark workload 全量调优（H=24/80 双族，分支全量测量制 + 双独立测量合并检查）：**model-scale 7 workload 几何 +9.7%**（全 11 +6.4%；serving-130m +16.1% / longctx-130m +15.2%；w2 probe 217.58→200.18µs −8.0%，复现并略超旧二轮 −8.6%）；跨引擎累积序调制实证（vbrchoist 单点 H=80 回退在 prevhoist 合入后消失——queue VP-2026-0115/0112）；probe_bn64.py 补配置路径覆盖缺口（queue VP-2026-0065 第三现）；stop_reason=blocked（候选穷尽：H-fold / 任务级行装载 / bl=128 邻域 / per-lt 细粒度 flag / vbrchoist-lite 五新否决 + 旧五否决 docs-only delta carry-over 论证）。适用触发条件补充：perf_opt 断裂/重建场景（repro 骨架重推导路径）；MixCV 调优的累积序检验、全量测量制与双独立测量合并门参照。

mamba/SSD 族首个 MixCV Expert 迁移完整档案（设计修订 1 轮——列因子广播错向 / FLOPs 2× 高估 / PL-1.11 误判 stale 三阻塞；Stage 3 Developer→Expert 模式切换〔模式级不兼容实证〕+ L0–Boundary 全过；Stage 4 六轮 2.91× 几何平均〔608.15→217.65µs@w2：深度 2 任务流水 −39.6% → Cube band 组装 → block_n=128 → AIV subid 蛇形分片 −34.4%；stop_reason=blocked——UB 容量/编译器 dominance/API/性能税清单〕；Stage 5 首次集成即全过 smoke 2/2 + full 4/4 + bench 11/11）。TileOPs bench_mamba 11 dispatch 基线（wrapper 默认 config 口径）：Perf 稳定 32–34 TOps/s 平台、Ratio 7.5–14.2%、小 dispatch（<40µs）启动开销主导——**口径注记**：bench 经 wrapper `default_config` 显式传参（block_n=64/num_stages=3），优先于 kernel 内嵌 TUNED_DEFAULT_CONFIG(block_n=128/num_stages=2)，与 Stage 4 自建 workload 数值不可同口径对比（VP-2026-0093）。适用触发条件：mamba/SSD/chunk-scan 族迁移设计；MixCV persistent 因子链（Vector 产因子 → ws 中继 → Cube 消费）结构参考；AIV 分片与任务级流水参照；band 分域掩码（band-free 惩罚 / band-carrying vselect）落地参照。

**〔2026-09-20 4515de8 重跑档案（task 20260920T122332Z，迁移重做——升级工具链上的先例知识重验）〕**：设计以「结构性结论继承（有 HEAD 生产代码佐证）/ 硬边界标待重验」两分法消费 stale 预注入条目（VP-2026-0101），3 轮检视收敛：v0 协议/形状缺陷（尾置 wait WAR 竞争〔VP-2026-0096〕+ P_tiles≥2 域形状矛盾）→ v1 写实占位伪代码自创 **L1→L1 cbuf→cbuf 拷贝硬编译失败**（CG-2026-0002 occurrences 2 / queue VP-2026-0095）→ v2 清零；Stage 3 直接复刻 HEAD 旧 perf_opt 终版 + 三处差异适配（pass_configs 按规格扩为双关闭〔PL-1.16 4515de8 重验〕、golden 内联仓内参考 torch 计算〔薄包装被 S3-GOLDEN-TORCH 拒绝，VP-2026-0107〕、测试套件按新 L0 计划扩 w2 用例）——**首编即过、零调试往返**（L0–Boundary 全绿，max_diff ≤1.1e-3 bf16 / 1.4e-4 fp16；Boundary B-n16 实测 gemm K=16 静默有效〔VP-2026-0098〕）；R3 表 ws 中继流量曾按 per-l-tile 口径低估 2.5×（87→216MB@w2，per-task 全维乘积口径修正——复杂度表复算教训）；Stage 5 集成 attempts=0 首过（工厂闭包自分配 ws + 输入 cast 的零胶水形态〔VP-2026-0099〕；report 11 条 "missing tileops candidate record" 告警≠失败〔VP-2026-0100〕；11/11 bench msprof 全有效，ratio 0.13–14.00%）。适用触发条件补充：迁移重做任务（旧工件被 Stage 0 删除、仅存于 git HEAD——检视须 `git show HEAD:<path>` 核对，VP-2026-0102）的设计/检视/实现全链参照。

---
id: CASE-CG-INDEX
kind: case
family: [general]
mode: [developer, expert]
dtype: []
status: verified
origin_task: conductor-improvement-2026-09-10
toolchain: 登记于 2026-09-10；各条目版本戳见 .agents/evolution/capability-gaps.md
repro: none
---

### 反例档案（能力缺口互链）

capability-gaps 登记簿（`.agents/evolution/capability-gaps.md`）open 条目指针——"该形态当前做不了"的反例，调优/设计遇同族负向结论时先查：

| gap_id | 层 | 一句话触发条件 |
|--------|-----|---------------|
| CG-2026-0001 | codegen (Developer mode) | Developer 形态 codegen 能力缺口 |
| CG-2026-0002 | BishengIR（cbuf→cbuf copy） | cbuf→cbuf 搬运不支持（vbrc 清零 lowering + Expert T.copy L1→L1 双形态；recurring） |
| CG-2026-0003 | TileLangIR pass (Expert fixpipe lowering) | Expert fixpipe lowering 与 BishengIR 版本配对缺口 |
| CG-2026-0004 | tladapter/codegen (load_nd2nz 区域描述符下推) | load_nd2nz 跨步区域静默误读（绕法见 traps-runtime.md TRAP-load-nd2nz-strided） |
| CG-2026-0005 | Frontend API / codegen | 前端 API 能力缺口 |
| CG-2026-0006 | TileLangIR pass / codegen（向量算子融合与 f16 打包发射） | 向量算子融合缺口（f16≈f32 发射速率的机制根源，见 attention.md PL-1.9-hardlimits） |
| CG-2026-0007 | runtime / codegen（跨引擎同步原语粒度） | 跨引擎同步原语粒度缺口 |
| CG-2026-0008 | BishengIR（UB 基础分配可见性） | UB 手工预算 vs 实际分配差（宽 config 不可编译形态） |
| CG-2026-0009 | runtime / codegen（MTE2/MTE3 引擎边界同 buffer WAR） | MTE2/MTE3 同 buffer WAR 无顺序原语（Vec 深度 2 blocked） |
| CG-2026-0010 | codegen（Developer mode persistent+gemm） | Developer 模式 persistent 分核 + gemm 混排运行时崩溃 |
| CG-2026-0011 | Frontend API（gemm dst L0C region 写 / acc 相加原语） | gemm dst 不支持 L0C region 写、两 L0C acc 无相加原语 |
| CG-2026-0012 | BishengIR（auto-multi-buffer 动态 subview 支配性） | 动态偏移 UB subview 进嵌套循环产出非支配 IR |
| CG-2026-0018 | Frontend API / runtime（跨 AI Core 归约与完成栅栏） | 无跨 AI Core barrier/归约原语 → grid=batch 小 batch decode 无法 split-N/split-K（Ratio 被 batch/40 钉死；sync_block 仅同核 / workspace 无前端入口 / atomic_add 无跨 block 栅栏） |

> BLOCKED 终态任务的反例根因链条目同入本节（由 evolver 追加，标注「反例」）。

---
id: CASE-argmax-argreduce-stage4
kind: case
family: [reduction]
apis: []
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro-missing
---

### argmax/_argreduce_kernel C6 N-split 单 launch 集成案例

适用：小 M、大 N 的 ArgMax/ArgMin 规约，单 kernel 的 bm=1 路径无法充分利用
V-core。该案例采用 PL-1.20 的 Expert N-split 结构：各 V-core 执行 per-row
`(1, tn)` partial 并写入二维 workspace；所有 producer 经 `T.block_barrier`
汇合后，lane 0 在同一个 `@T.prim_func` 内完成 dim-0 规约和首现索引 tie-break。
编译时仅该 N-split fused 分支使用 `TILELANG_ASCEND_MODE=expert`；其他路径保持
各自原有模式。性能测量以完整 fused kernel 的 `msprof op` Task Duration 为准。
关键代码证据与最小正确性复现见 PL-1.20 的
`repro/repro_nsplit_pattern.py`。

---
id: CASE-argmax-argreduce-integration
kind: case
family: [reduction]
apis: []
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2+ubuntu.22.4.npuir / Ascend910B2C / CANN 8.5.0 / git 013dbbf5
repro: repro-missing
---

### TileOPs reduce 族集成契约参考（argmax 首集成的胶水修复档案）

`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/`（任务级溯源，允许失效——集成包为任务交付件，未随知识库演进存活；integration_log.md + wrapper `argmax.py` + history_version/argmax_s5_attempt1.py 修复前备份）。触发条件：任一 reduce 族算子接入 TileOPs（ArgminFwdOp 等共享 ArgreduceKernel 的迁移单元前置阅读）。要点：① Op 层 `_ReduceOpBase._prepare_input` 在 `_kernel_handles_padding=False` 时 pad 到 align_up(N,256)——kernel jit 声明宽度须取 N_padded（raw-N 契约与 Op 层 F.pad 冲突的症状：行 0 正确、行 ≥1 全错且不报错，初次 report 17 failed 与 N_padded≠N 用例 100% 重合）；② wrapper `default_config` 转调 kernel 模块 `_select_config`（GPU 时代 smem 启发式选出的 block_m 在新 kernel 上直接 `ub overflow, requires 2097408 bits while 1572864 bits available`）；③ argmax 语义 -inf pad 与首现索引兼容（全 -inf 行 kernel 与 torch 同返 0），argmin 共享 kernel 时 `_pad_value` 须为 +inf；④ 修复后 42/42 + benchmark 5 case 全有效（run.json 20260928_030100_974855）。

---
id: CASE-engram-decode-stage4
kind: case
family: [reduction, gemv, elementwise]
apis: [T.reduce_sum, T.copy, T.Kernel]
dtype: [fp16, bf16]
device: 910B3
status: verified
origin_task: TileOPs-_engram_decode_kernel-20261030T121905Z
toolchain: 首证 tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0；round 3 复验 tilelang 0.1.2+01805d7a + CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
repro: repro/CONST-ub-budget-vs-pattern-cap.py
---

### grid=batch 小 batch decode 类算子的 Stage 4 Ratio 天花板档案（反证链 + 三胜出）

`examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/`（任务交付件路径，provenance 允许失效）。**触发条件**：grid = batch（每 batch 一 block）、batch ≤ 8、单 block 工作量固定、优化目标是 roofline `Ratio` 而非绝对时延——此时 Ratio 由「核数比例 × 单核效率」与「launch 地板」两条约束决定，而非实现质量。

- **两轮结果**：round 1 用「非累加乘积 tile + `T.reduce_sum(dim=0, clear=False)`（`hivm.hir.vreduce`）」替掉 `T.Parallel` 内元素级 loop-carried 累加（默认 lower 为全标量 `scf.for`），三档 workload 提速 191–302×（迁移期基线 5988.77 / 23668.15 / 5988.63µs → 30.18 / 78.40 / 31.31µs）；round 2 把 `new_conv_state` shift 的逐行 2×小 DMA 换成整块 2-D 两次 `T.copy`，再降 5.86% / 6.14% / 8.75%（28.411 / 73.591 / 28.571µs，精度逐位一致）。
- **Ratio 口径对照**：本管线（`--aic-metrics=...,Roofline` + `--kernel-name=main`）复现官方 report 的 Ratio 偏差 <2%（官方 benchmark.md：29.9406µs→1.0229% / 73.6615µs→3.8706% / 28.6206µs→5.1963%，tune 平均 3.36%；本地 final 平均 3.3850%——官方含 wrapper padding/host 路径）。**round 3 同向复现**：官方 27.9806 / 73.6215 / 26.8005 µs（1.0945% / 3.8727% / 5.4232%，tune 平均 3.4635%）vs 本地 msprof `--launch-count=20` 26.651 / 73.541 / 26.791 µs（平均 3.4827%）——b1 差 +5.0%（1.33µs），b4 +0.11%、b8 +0.04%；两个任务同向 ⟹ 小 workload 的官方 Ratio 系统性偏高约 5%（归因未定，见 queue VP-2026-0153）。采数规则见合入后的 profile-collection.md roofline Ratio 段（原 VP-2026-0081）。
- **60% 目标裁决（不可达）**：b1 的 roofline ideal 仅 0.306µs，小于空 kernel 地板 1.89µs ⇒ b1 Ratio ≤16.2% 与实现无关（0.306/1.89）；核数比例把 b4/b8 钉在 10%/20%。三档上界：① 结构冻结·完美 pipe 重叠 ≈4.76%、② 结构冻结·绝对上界 ≈11.3%、③ 假设跨核归约免费的重设计 ≈39–41%（能力缺口 CG-2026-0018）。当前 3.3850% = ① 档的 71%。
- **三胜出机制**：① `vreduce` 替代逐元素标量累加（单核向量/标量分离）；② 逐行小 DMA → 整块 2-D copy（每 DMA 指令 ≈0.16µs 固定开销，行数 11–19 时收益 5.9–8.8%）；③ BK=16 保住 `--enable-auto-multi-buffer` 重叠（slab 放大暂存块 → UB 翻倍 → 多缓冲被挤掉，78.4→107.2µs）。**round 3 修正**：破坏多缓冲的不是「放大 tile」而是「引入第二段暂存 / 显式分块复用」，编译期 tile 常量本身放到预算上限仍有 +6% 收益（见 elementwise.md `PL-1.21-overlap-saturation-signal`）。
- **否决清单（可直接复用）**：BK=32（d=512 UB overflow，`requires 1606656 bits while 1572864 bits available`）；fp16 暂存 + fp16→fp32 混合 reduce（编译 `memref.expand_shape ... found 1, expected 2`，最小探针静默算错——见 traps-compiler.md TRAP-reduce-mixed-dtype-silent-corrupt）；`e_local[k0*BK+kk]` affine 索引行广播（编译通过、结果错 `y_maxdiff=1.49`）；跨核 split-N / split-K（CG-2026-0018）。
- **round 3（2026-10-30，tilelang 0.1.2+01805d7a / CANN 9.0.0）**：只采纳一个编译期 tiling 改动 —— `BK = TUNED_STAGE_ELEMS // d_padded`（d=256→32、d=512→16），b1 28.361→**26.651 µs**（1.146%）、b8 28.611→**26.791 µs**（5.425%）、b4 73.591→73.541 µs（不变），tune 平均 3.3832%→**3.4827%**。
  - **逐项目标未达成且已证明不可达**：目标是 `max(pipe)`（b1 19.66 / b4 54.78 / b8 19.74 µs = 平均 4.76%），它要求 scalar 完全被隐藏；本轮对 `profiles/**/prof.json` 的 `duration/vec/scalar` 逐档反解出 `duration ≈ vec + a·scalar`：**pipeline 未被破坏的分支 a∈[0.43,0.68]**（b4 基线 54.78+0.47×40.24=73.7 vs 实测 73.59；b1 16.19+0.68×15.49=26.65 vs 实测 26.65），**流水被拆的分支 a≥0.9**（`r3a_eub` b4 1.03、`r3f_bk32inplace` b4 1.28、`r3n_bk64inplace` 1.21、`r3t_slab2` b4 0.97，同档 `mte2_wait` 0.44–0.83）。opt_log §2.2 原记「12 个分支一致 a∈[0.47,0.68]」为过窄表述——含退化分支的完整分布是 0.43–1.28（蒸馏时逐档机械复算，见 queue VP-2026-0150）。可达包络 ≈3.78%（按 a=0.5 估计，量级成立、逐档不精确）。
  - **round 3 新增否决清单（可直接复用）**：`e_n` 改从常驻 `e_local` 做 UB→UB 切片（b4 73.6→102.2，拆掉了 auto multi-buffer）；`T.unroll(BK)` 逐行 FMA 累积替代 mul+vreduce（vec ×4，写带宽 214 GB/s）；K/V 各自暂存（mte2 51→29 但 duration +3.8%）；原地乘积（d=256 有效 27.26，d=512 崩到 95.7）；`T.Pipelined(num_stages=2)`（= 基线，编译器已等效）；reduce 操作数行 padding（b1 48.37 / b4 120.45，vec 翻倍；条目待 Tier 1 二次证据，见 queue VP-2026-0149）；W 暂存改 f16 的混合精度乘（编译通过、结果逐位一致，但 **vec/duration 一位小数不变** ⟹ 否定「UB 带宽是 vec 瓶颈」）；2×8192 元素 slab + 列分块 reduce（b4 514 µs，scalar 422 µs）。
  - **BK 上限的真实来源**：不是 pattern 级元素上限，而是 GEMV 暂存与尾部工作集共享 192 KiB 预算且 Python 侧腾不动（见 constants.md `CONST-ub-budget-vs-pattern-cap`，含最小复现）。
- **审计入口**：`perf_opt/opt_log.md`（round 3 全文：P1–P4 现象、12 分支对比表、可达包络推导）+ `perf_opt/history/opt_log_round2.md`（H1 反解与三档上界）+ `perf_opt/history/opt_log_round1.md` + `perf_opt/perf_records.jsonl` 35 行（baseline 3 + candidate 29 + final 3，逐项可对账）+ `perf_opt/history/perf_records_round{1,2}.jsonl` + `perf_opt/probe/h1_probe.py` + `perf_opt/run_exp.py`（本轮批处理 runner）。

---
id: CASE-engram-decode-mcore-ratio-ceiling
kind: case
family: [gemv, reduction, elementwise]
apis: [T.Kernel, T.reduce_sum, T.copy]
dtype: [fp16, bf16]
device: 910B3
status: verified
origin_task: engram_decode_mcore-engram_decode_mcore-20261031T012849Z
toolchain: tilelang 0.1.2+3ed7c91420e925ff3c388fb981dd6ebf45ff4bf + CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-31
repro: repro-missing
---

### 「多核重做」把 grid=batch 的小 batch decode 从 3.47% 推到 9.95%（2.87×）——同时证明 40% 不可达（路线无关上界 ≤22.2%）

> 端到端形态的复现依赖任务工作区的两段采集脚本（`bench_stages.py` + `run_bench.sh`，每段单独一次 msprof 进程），故本条登记为 `repro-missing`；与可复现性相关的**常数**（FLOOR、每核向量吞吐、段宽-带宽）与**判据**已内嵌本文与 constants.md，可脱离过程文件复算。

`examples/engram_decode_mcore/engram_decode_mcore/`（任务工作区，provenance 允许失效）。**触发条件**：`grid = batch`（batch ≤ 8）的 GEMV + 多段 elementwise 融合算子，上游把平均 roofline Ratio 目标定在结构性天花板之上（本例 40%，当前 3.47%），需要回答「换并行轴能拿多少 / 目标是否物理可达」。与 `CASE-engram-decode-stage4` 是同一算子的两个阶段：stage4 档案是**单核内**把 12 个分支全部试完并把 Ratio 停在 3.48%，本档案是**跨核重做**（上一版 DESIGN §5.5 显式放弃的维度，对应 CG-2026-0018）。两档案合起来才构成「该算子族 Ratio 天花板」的完整证据链。

**交付形态**：两阶段 `stats → apply`（两次 kernel 发射即跨核同步点，**无 atomics、无自旋**）；split-d 列分块 `T.Kernel(B·ns)`；GM workspace 两个（`v_ws (B,d_padded)` fp32、`p_ws (B,ns,8)` fp32，免初始化，9KB@b4）；`ns = max{2^k : B·2^k ≤ 40 ∧ d/2^k ≥ 32}` → **b1 8 / b4 8 / b8 4**（逻辑核数 8 / 32 / 32，占用 20% / 80% / 80%）。A 段每块做列块 GEMV + 4 个局部部分和；B 段不再物化 `h_n/k_n`（0 个 d 长 pass、0 次 reduce）。

**实测（op 级 = A+B，两次独立采集 `bench_out/`、`bench_out2/`；分子取 op 级 `eval_roofline()` ideal）**：

| workload | A (µs) | B (µs) | op = A+B | Ratio(op) | 基线 Ratio |
|---|---:|---:|---:|---:|---:|
| b1 (dmem512 d256) | 6.82 / 6.70 | 3.32 / 3.35 | **10.14 / 10.05** | 3.01% / 3.04% | 1.140% |
| b4 (dmem1024 d512) | 13.00 / 13.22 | 4.77 / 4.73 | **17.77 / 17.95** | 16.04% / 15.88% | 3.877% |
| b8 (dmem512 d256 bf16) | 8.50 / 8.72 | 4.78 / 4.76 | **13.28 / 13.48** | 10.94% / 10.78% | 5.399% |

3 档均值 **10.00% / 9.90%（≈9.95%）**，基线同口径 3.472% → **2.87×**；设计期估算 11.1% ⇒ 实测为其 **0.90×**。**只报 A 段会虚报**（b4 只算 A 段是 21.9%，真实 op 级 15.9%）；两段 kernel 的 Ratio 分子仍取 **op 级** ideal，分母必须 A+B 求和。

**路线无关上界（本档案的核心结论，可复算）**：任何实现都必须付 (a) 一次 kernel 发射 ≥ `FLOOR(n)`、(b) 全部向量工作 ≥ `总向量元素数 / (n × 每核实测吞吐)`（`4·B·d_mem·d`，mul+reduce 两遍）；带宽与末阶段（apply）**全按免费**，并对块数 `n ≤ 40` **取最优**：

```
下界(n) = FLOOR(n) + 4·B·d_mem·d / (n · throughput)
  FLOOR: 1.570(1) / 1.600(8) / 2.520(32) / 2.850(40) µs   （本戳实测）
  throughput: 22163 / 34652 / 34062 elem-op/µs（b1/b4/b8 实测）
⇒ b1 3.08µs(9.92%) / b4 8.90(32.04%) / b8 5.93(24.51%)，均值 ≤ 22.2%
```

对照 40% 门槛（0.7635 / 7.128 / 3.634 µs）：分别低 4.0× / 1.25× / 1.63×；均值论证——b1 上限 9.92% 则 b4+b8 需 ≥110.1%（各 ≥55% ⇒ b4 ≤5.18 < 8.90、b8 ≤2.64 < 5.93）。**鲁棒性**：即使假设 `mul+reduce` 融合成单遍（工具链不支持，融合形态实测 vec ×4），均值上限也只到 31.5%，b4 仍需 ≤5.33 < 5.88、b8 ≤2.72 < 4.39。**b1 的独立硬顶**：`ideal_op = 0.3054µs`，单 block 空 kernel 地板 1.570µs ⇒ ≤19.45%，40% 门槛（0.7635）比该地板还低 2.06×。

**能力缺口无关性**：CG-2026-0018（跨核归约/栅栏缺失）补齐最多抹掉 B 段（3.32/4.77/4.78µs），即回到该路线的 A 段天花板——split-d **15.7%**（实测）/ split-K **16.2%**（同公式模型），仍远低于 40%。故 CG-2026-0018 的 `blocked_algo` 中「若跨核归约免费 → 均值 ≈39–41%」应订正为 **≤22.2%**（本档案给出可复算推导）。

**偏差归因（设计 vs 实测）**：A 段比 P2 探针 +4% / +10% / +10%（多出的 4 个统计量 / 2 次额外 reduce / `h`·`w_h` 列读 / 8 元组打包）；B 段比模型 +8% / +18% / +19%（模型对 14 条 DMA 的 0.5× 暴露系数偏乐观）。B 段由**发射**主导：vec 仅 0.37/0.37/0.34µs，**scalar 0.85/1.92/1.37µs**，即两段方案的 B 段 3.3–4.8µs 里大部分是发射地板。

**实现期抓到的四个陷阱（均已有知识域 repro）**：`T.Parallel` 体内内联读 `(1,1)` UB 标量 → codegen SIGSEGV（[traps-compiler.md TRAP-parallel-scalar-broadcast-segv](traps-compiler.md)）；`prim_func` 体内 `range()` 不按 Python 展开、符号链式比较被拒（[TRAP-primfunc-static-branch-and-loop](traps-compiler.md)）；`ε` 放在 `α²` 折叠之前 → 门控饱和区 `y_maxdiff=7.7`（[traps-runtime.md TRAP-eps-before-alpha2-fold](traps-runtime.md)）；`(ns,8)` 上 `dim=0` reduce 数值精确且成本 ≈0（[constants.md CONST-reduce-dims0-ns8](constants.md)）。**分核常数**：跨步列块读 26–30 GB/s 且列切片 <128B 时 payload 放大 2×（段宽决定带宽，与连续读 68 GB/s 不是同一口径）。

**可复用判据**：① Ratio 类硬目标先算**路线无关**上界（`FLOOR(n) + 总向量工作/n`，带宽与末阶段免费，n 也纳入最小化），再决定是否投入结构改造；② 单路线实测只能称「该路线的观测值」；③ 两段/多段 kernel 必须按 A+B 求和报 op 级 Ratio，且每段单独进程采集（两段同名 `main` 时 `--kernel-name` 会一起命中，`_parse_msprof_roofline` 静默只取一段）。

---
id: CASE-fft-c2c-vec-design-blocked
kind: case
family: [spectral, elementwise, expert]
apis: [T.Parallel, T.vadd, T.vsub, T.vmul, T.deinterleave]
dtype: [fp32]
device: Ascend910_9362
status: verified
origin_task: fft_c2c_vec-fft_c2c_vec-20261031T103133Z
toolchain: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362（Expert 模式）/ 2026-10-31
repro: repro-missing
---

### 反例档案：蝶形**能**向量化（向量率 ×800），但净收益为零、n≥2048 不可交付 ⇒ Stage 3 `[DESIGN_ERROR]`

`examples/fft_c2c_vec/fft_c2c_vec/`（任务工作区，provenance 允许失效）。终态 **FAILED / BLOCKED_DESIGN**（stage_plan [1,2,3]，Stage 2 检视通过后 Stage 3 判设计错误）。**触发条件**：`T.Parallel` 逐元素主体在 npuir 上被判「无法向量化」而准备放弃；或设计假设「级缓冲 liveness 复用 → UB 峰值 = 两级」。

**① 机制成立（推翻上一轮口径）**：形态 =「每级 4 个独立的 `T.Parallel(h_s)`、每体恰 1 条 `BufferStore`、级间仿射 3 下标读 `prev[g//ng_prev, g%ng_prev, i]` / `[..., h+i]`」。实测 IR（n=8）：`vadd 10 / vsub 14 / vmul 8 / vdeinterleave 4`、`arith.mulf = 0`、`scf.for = 1`；`aiv_vec_ratio` 由基线 **1.2e-4 → 0.093–0.105（≈800×）**；n=16…1024 全绿（max_diff ≤ 2e-5）。
> **对上轮 `perf_feedback.md` 的订正**：基线结论「npuir 无法向量化蝶形 / 向量化完全未发生」口径过宽——真正的入口门槛是**循环体恰 1 条 BufferStore**（`npu_loop_vectorize.cc` 的 `VectorizeSingleStatement` 走嵌套循环递归分支时不报错、静默标量化），索引形态本身不是门槛（原判断只核到 store 下标规则 `L1184` 与 load 规则 `L251`，漏了入口条件）。上轮 `r1_serialvec`（`T.serial`+`T.vectorized` 4 单 store 循环）失败与之相容：`T.vectorized` 不进 `current_loop_vars_`。

**② 但不产生收益**：向量化把每 butterfly 的语句数从 4 膨胀到 **10**（`TRAP-npuir-vector-composite-rhs` 迫使每句 1 op + 4 个显式 3-D 中转）；后段级 `h=2` 只有 2-lane，平均 lane 利用率 ≈5.5/64。实测（`msprof op` Task Duration，中位数）：

| workload | 向量化实现 | 基线（标量） | 比 |
|---|---:|---:|---:|
| n=64, B=1 | 12.82 µs | 6.42 µs | **2.0× 慢** |
| n=1024, B=1 | 172.4 µs | 152.1 µs | 1.13× 慢 |
| n=1024, B=40 | 175.0 µs | 180.9 µs | 1.03× 快 |

**③ n≥2048 不可交付（容量）**：设计 §4.5 的「liveness 复用 → 128 KB」被推翻——UB 规划只回收级内短命中转（11 份 → 1 份），**跨级存活的 11 对 `(2,ng,h)` 一份都不合并**；n=2048 需 221 KB、n=4096 需 **464 KB > 192 KB**（编译期 `ub overflow`）。降级链 radix-4（288 KB）/ radix-8（224 KB）/ flat ping-pong 全部不成立（见 capability-gaps CG-2026-0023）。n≤1024 可编译。

**④ 60% 不可达**：三档 Ratio 上界 smoke ≈0.039% / unbatched 0.55–1.25% / b64 8.8–17.7%，均值上界 ≈6–21%（`ideal` 地板换算 60% ⇒ 2.0 µs，远低于任何实现的地板）。

**实现期两条陷阱（均已转正为知识域 repro）**：多 op RHS 静默写 0（[traps-compiler.md TRAP-npuir-vector-composite-rhs](traps-compiler.md)）；twiddle 段起点奇数 → 向量操作数非 32 B 对齐 → 运行期 abort（[TRAP-npuir-vector-lut-alignment](traps-compiler.md)，对应能力缺口 CG-2026-0024）。

**交付物**：`DESIGN.md`（v2，§0.1 单 store 体规则 / §1.6.3 两段式仿射读）+ `REVIEW.md`（首检不通过 2 阻塞 → 复检通过）+ `history_version/design_v1_review_failed.md` + `history_version/design_probe_01_vecform.py` + `fft_c2c_vec.py`（可编译、n≤1024 数值正确）+ `repro/`（2 个）+ `RETROSPECTIVE.md`。**可复用判据**：① 「能向量化」不等于「值得向量化」——先按「膨胀后语句数 × 平均 lane」估时延再决策；② 每个新语句形态先跑 ≤20 行数值探针（编译成功 ≠ 算对）；③ 跨级/跨块的每级独立缓冲按「跨级存活总和」预算，不假设 ping-pong 复用。

---
id: CASE-attention-gqa-bwd-stage4
kind: case
family: [attention, mixcv]
apis: [T.gemm, T.atomic_add, T.copy, store_fixpipe]
dtype: [fp16, bf16]
device: Ascend910B3
status: verified
origin_task: TileOPs-_gqa_bwd_kernel-20261031T051147Z（第一轮 2 项采纳 + 40% 不可达判定）；TileOPs-_gqa_bwd_kernel-20261107T122917Z（Round 21 3-pass BN=128 capacity-partitioned widening 采纳）
toolchain: tilelang 0.1.2+9738284cd6 / CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV，rated 1800 MHz，msprof 25.5.0）/ 2026-10-31 · tilelang 0.1.2+9cfae94c + CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ msprof 25.5.0 / 2026-11-07（Round 21）
repro: repro-missing
---

### GQA backward（BSHD，Developer 单 program cube→vector→cube 链）Stage 4 档案：访存形态两项消除 −65.9%，剩 70% 为每迭代发射地板

`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/`（任务工作区，provenance 允许失效）。终态 DONE（`perf_feedback.action=archive`），stage_plan=[4]，单 attempt 18970 s、零重试。**触发条件**：BSHD attention backward 的 `dq` 跨 kv-block 归约、或 profile 显示 `aiv_mte3` 独大；或 `BN` 曾被「静态 UB 加和超 192 KB」判为不可行；或单 program 内顺序执行 cube→vector→cube 链而 cube 占比 <10%。

**结果**：8 档 tune workload 几何均值 **22966.3 → 8001.4 µs（−65.9%）**；官方口径 Ratio **1.4610% → 4.2375%（2.90×）**（同批官方 report 复测 4.1389–4.3569%，均值 4.2451%）。逐档 final 1995.8 / 1993.7 / 14018.2 / 14026.8 / 1989.1 / 1995.3 / 13998.2 / 13994.1 µs（全部来自同一 artifact sha `815ad7f9…`，final 记录 sha 一致）。精度：L0 8/8 + L1 12/12 + L2 9 + Boundary 8 全 PASS（groups 1/2/4/8 × n_kv 1/4 × D 64/128 × causal/非 causal）。

**本轮采纳 2 项（Stage 4 内可解）**：
1. **`dq` 逐行 1-D 原子 → rank-2 跨步区域原子**：`dq` 以 `(B·S, H·D)` 视图声明 + 工厂内薄闭包（`dq.view(B*S, H*D)`），对外 9 张量签名不变、wrapper 零改动。实测每 block：`aiv_mte3` 220.3→**4.8** µs、`aiv_mte2` 141.1→**18.3**、`aic_scalar_wait_id2` 409.1→**23.6**、`Task Duration` 5865.6→1995.8 µs。机制：GM 通道原被原子流量挤占，逐行原子消失后装载通道同步恢复。单价口径 224 ns/行 vs 32 ns/行（约 7×）。条目化见 queue VP-2026-0168（D，repro-missing）。
2. **`BN` 32→64（`dim=128` 档）**：baseline 为 `BN = 64 if dim<=64 else 32`，改为常量 64 后 −1.4%/−4.3%（短/长档）。**推翻设计期「静态 UB 加和 240 KB > 192 KB 不可行」**——分配器按生命周期复用，实际 workspace 155 648 B/block，编译通过。条目化见 queue VP-2026-0169（D，repro-missing）。

**本轮否决（含跨轮复现）**：`num_stages=2` 第三次实测 <3%（参数级无收益）；宽原子暂存 `[BN,groups,dim]` fp32（64–128 KB）三种写法**编译失败**（`hivm.hir.copy: Unsupported copy from cc to cbuf` / `hivm.hir.load: src and dst shape should be the same`）。

**剩余天花板（设计层，非 tiling 可解）**：修完后 `aiv_scalar` 106.1 / `aiv_time` 150.3 = **70%**，`aiv_vec` 18.2、`mte2` 18.3、`mte3` 4.8、cube 实算 8.6（**cube 占比 5.5%**，baseline 2.3%）。即瓶颈自「访存形态」转为**每迭代指令发射 + Developer 双 AIV 冗余执行**（`vector0`/`vector1` 的 mte2/vec 计数一致 ⇒ 两份相同工作，见 [PL-1.13](attention.md)）。40% 目标需 cube 占比 **≥53%**（= 实测 fp16 峰值 270.1 TFLOPS 的 44% 持续算力）；`BM` 上探被 L0C 128 KB 双封顶（`BM=128` 需 `dk_f`/`dv_f` 各 64 KB）。

**同算子参照上界（路线对照，非本档案实测）**：`examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §10.4 的 **3-pass 单引擎相位拆分**（Cube-only `_gqa_bwd_sdw` → Vector-only `_gqa_bwd_pds` → Mix `_gqa_bwd_mul`；GM workspace + `by0` 标量 chunk 入参使 dk/dv 保持单写者）在 910B2 / 24 核同工具链族实测 1016.8 / 7518.6 µs ⇒ Ratio ≈**8.2% / 8.1%**，仍只有 40% 目标的 1/4.9。该路线的三项结构代价（kernel 数量改变 op 契约与官方单 kernel-name 采数口径 / 需跨 pass GM workspace 与标量入参 / 纯 Vector pass 每 block 仅 1 AIV 且需 `T.pipe_barrier("PIPE_ALL")` 关 WAR 竞态）见该任务 `perf_feedback.md`。

**能力缺口裁决**：本条不增量 CG-2026-0018——本任务的跨核交接路径（多 kernel 发射 + 外部 workspace 张量）**可用**（参照实现已落地），阻塞项是 op 契约 + 官方 Ratio 单 op 采数口径，属流程/口径而非前端能力缺失；Developer 双 AIV 冗余由 PL-1.13 已覆盖。

**可复用判据**：① Ratio 类硬目标先算「达成该 Ratio 所需的引擎占比」再决定是否投结构改造（本条：40% ⇒ cube ≥53%，现 5.5%）；② BSHD `dq` 类跨核归约的原子形态决定 GM 通道是否被挤占（行粒度形态的代价会以 `mte2`「装载变慢」假象出现，见 [CONST-gm-to-ub-bw-dilution](constants.md)）；③ 容量类否决（静态加和 vs 分配器实际占用）在写结论前需编译探针，口径须标注；④ 参照实现的上界只能称「该器件、该口径下的观测值」。

**⑪ Round 21（2026-11-07）：3-pass BN=128 capacity-partitioned widening 采纳——几何 −25.2%，上文「每迭代发射地板」被证明可摊薄**。上文把剩余 70% 归为每迭代发射地板并判「tiling 不可解」；Round 21 直接攻该地板的**迭代数**：`sdw`/`mul`/workspace slot 用 BN=128（BM 保持 64），纯 Vector 的 `pds` 保持 64 行子块并在 slot 内编译期展开 `n_sub = block_n//64`（`q0 = i·128 + s·64`），逐 score 元素的算术/tile 形状/元素顺序不变。**8 档 final（msprof op，3-pass 中位之和）**：8b-short fp16 **945.36**（−21.46%）/ bf16 945.54（−22.14%）、8b-long fp16 **5411.35**（−28.74%）/ bf16 5420.17（−29.14%）、70b-short fp16 925.86（−21.77%）/ bf16 934.78（−21.42%）、70b-long fp16 5459.31（−28.29%）/ bf16 5482.13（−28.19%）；几何 3021.51→**2259.39 µs（−25.2%）**，无一回退。探索轮 3-run 交错 A/B 3/3 同向（8b-short −22.14% / 8b-long −28.88%）。**逐 pass 归因**（元素总量 / workspace 字节 / dq 原子字节均不变）：`mul` −42.9%（577.43→329.91）/ −48.2%（4003.30→2074.14），`sdw` −10.3%（260.07→233.40）/ −17.5%（1763.84→1455.47），`pds` +0.9% / +2.2%（分区预期，迭代数不变）——迭代数减半的固定成本摊薄在 mul 里最厚（每迭代 3 GEMM + 区域原子 + 握手链）。**容量裁决两则**：① `mul` 的 L0C = `2·BM·D + BN·D = 32768` 个 fp32（100.0%、零余量）被 `bishengir-compile` 接受（workspace `65536 bytes/block`）⇒ L0C 判据是「≤ 32768」，等号可用，设计期「留 25% 余量」的先验被证伪；② 同 BN=128 的窄口在纯 Vector pass 的 UB（`pds` 报 `ub overflow requires 1871872 bits while 1572864 bits available` = 228.5 KB vs 192 KB），`mul` 的 UB 静态 240 KB（125%）反而编译通过（live-range 复用，见 [CONST-ub-alloc-live-range](constants.md)）。**可行性前置**：BN 必须整除 `seq_len`（`_split_geom` 的 gcd 回退），8 个 tune workload 的 `seq_len ∈ {512,2048}` ⇒ BN=96 在两处都退成 gcd=32，BN=128 是唯一可评估加宽点。**精度**：转正后 `--level all` EXIT=0 **37 PASS** 无 WARN/FAIL（原记录同时给分层 L0 8/8、L1 12/12、L2 12/12、Boundary 8/8；总 PASS 数与四层之和不等，两处原样并列不拼接），8b-short fp16 `max_diff 6.305695e-03` 与 BN=64 交付件 6.31e-03 同量级；best sha `26edd4c1` → `0ff4a4f3`。**采纳成本**：探索轮 5400 s 预算在 A/B 后仅剩 ~300 s，采纳路径另起一次 dispatch（`--level all` 2×511–514 s + 8 档 final ≈27 min，总 ≈76 min）；每 msprof run 因 `cache.clear_cache()` 重编译 ≈230 s。官方口径 Ratio 本轮未重测（如实登记）。条目化见 queue VP-2026-0221（P，分区加宽）、VP-2026-0222（P，整除性筛选）、VP-2026-0220（D，L0C 等号边界）。

---
id: CASE-rope-neox-1d-fixed-floor
kind: case
family: [elementwise, position_encoding]
apis: [T.copy, T.vmul, T.vsub, T.vadd, T.cast]
dtype: [bf16, fp16, fp32]
device: Ascend910B3
status: verified
origin_task: TileOPs-_make_rope_neox_1d-20261031T184833Z / TileOPs-_make_rope_neox_1d-20261104T005217Z + …005218Z（跨 op 采纳同步）/ TileOPs-_make_rope_neox_1d-20261104T065720Z（Round 4 受控实验）
toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-11-01；TILELANG_ASCEND_MODE=Developer；2026-11-04 Round 4 同设备同工具链（tilelang dev root build / CANN 9.0.0 / torch_npu 2.7.1）
repro: repro-missing
---

### 1d RoPE（half-split，纯 Vector）Stage 4 档案：时延由「launch/ramp + 每核骨架」固定项主导，不是带宽受限

`examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/`（任务工作区，provenance 允许失效）。终态 DONE，`plateau / no_gain`（官方 tune 均值 27.374% → 27.351%，持平）。**触发条件**：小 shape 元素算子（`ideal_us < 4 µs`、单核工作量固定、40 AIV 全开）时延明显高于「流量 ÷ 带宽 + 1.9 µs 发射地板」估算；或 `cos/sin` 表与数据同量级；或在 UB 预算紧的逐元素算子上考虑 `T.Pipelined` 双缓冲。

**核心数据（8k-d128-bf16，`msprof op` Task Duration 总 6.35 µs，五探针加法闭合）**：launch/ramp **1.74**（`Task Duration − aiv_time`，5 探针 1.73–1.83）+ 每核骨架 **1.78**（40 核唤醒 + 7 个 DMA/UB 描述符 + 循环/guard，10 KB `tiny_ub` 探针）+ x/y 数据 4.19 MB **0.48**（L2 8.7 TB/s）+ cos/sin 表 2.10 MB **1.44**（1.45 TB/s）+ rope 6-op 链 **0.91**。**真实固定地板 3.52 µs**（非空 kernel 的 1.9–2.85 µs）；`x`/`y` 从来不是瓶颈。常数与复现条件条目化见 queue VP-2026-0173（D，repro-missing，target constants.md）。

**本轮采纳（唯一结构收益）**：fp32 中转 buffer 7→6——先算 `a*s`/`b*s` 使 fp32 输入 buffer 直接充当输出（去掉 `y1f`/`y2f`，数学与舍入同序）→ 每行 UB 2816→2560 B → 8k tile 52→69 行 → 波次 4→3 → 6.35→6.07 µs（交错 A/B **−3.45%**，3/3 对同向）。另三档 tiling/op 序列逐字未变（2k-d64 交错 A/B 判 `tie`，无回退）。

**跨 op 采纳同步复证（2026-11-04，task `TileOPs-_make_rope_neox_1d-20261104T005217Z` / `…005218Z`；同一 sha `c6098298` 的 adoption sync，无新调优轮次）**：r3 二进制逐字节采纳进 `rope_longrope` / `rope_llama31` 两个同级 op 目录（`perf_opt/` 侧，与规范位 `cmp` 一致；旧 r2_contig `796840bc` 归档 `perf_opt/history/`），两目录各自在**本目录内**重采 5-run 交错 A/B（`msprof op` Task Duration，`perf_opt/_profile_adopt.py` / `_ab_adopt.py`；成本约 12 / 20 个 run）：LongRope 8k-d128-bf16 **6.29→6.15（+2.2%，5/5 pair 不慢于 r2）**、2k-d128-fp16 3.87→3.91（−1.0%，tie）；Llama31 8k **6.32→6.11（+3.3%，5/5 快）**、2k 3.96→3.96（tie）；规范位同口径 +3.5%。三个目录的 8k 5-run 中位落在 **6.11–6.15 µs（0.6% 内）**，而官方单采样在 6.08–6.40 间（±2.6%）——同 sha + 同 shape 换目录时可复用实现，数值仍以本目录重采为准（`artifact_path` 指本目录而 `duration_us` 借自别目录，会让「记录可对账」退化为「靠标注可信」）。2k 档三目录一律 tie，与「2k 由 3.52 µs 固定地板主导」一致：固定项占比越高，结构改动的相对收益越被地板吸收。官方三档 Ratio 均值：LongRope 36.02%→36.34%，Llama31 35.49% / 35.91%（旧 r2 36.00%）——该指标对 1d 单档改动的敏感度、以及 8k 分母随 kernel 版本漂移 1.8% 的事实见 queue VP-2026-0194（D，repro-missing）；采纳验收以本目录交错 A/B 时延为准。另：两目录开工时 `statectl gate 4` 均为失败（`S4-PERF-RECORDS-COVERAGE`，6 / 10 处——上一轮采纳把规范位 `perf_records.jsonl` 拷回后 `workload_inventory.json` 的 `kernel_id` 仍指 baseline 路径、records 用 `perf_opt/` 路径且混入 2d 行），本次改为 inventory/records 同源重写后 gate 4 通过，流程侧提案见 queue VP-2026-0195/0196/0197。

**Round 4 受控实验（2026-11-04，task `TileOPs-_make_rope_neox_1d-20261104T065720Z`；单 attempt 6456 s、零重试）**：在 r3 current best（sha `c6098298`）之上验证 4k/8k 残余 3–5% 空间，三个候选全部否决 → `no_gain`，r3 逐字节保留，官方三档均值 27.35% 不变。① `r4_prefetch`（零额外 UB 的下一 tile 预取，复用 up-cast 后已死的 `x_stage`/`c_ub`/`s_ub`）：5-run 交错 A/B 4k 4.73→4.86（**+2.75%，更慢**）、8k 6.28→6.24（−0.64%，tie）、2k 3.63→3.57（tie）；同批 raw profile 显示收益来源不成立——`mte2_wait` 0.754→0.638、`scalar_vector_stall` 1.214→0.013，但 `vec_time` 2.466→2.703、新增 `scalar_mte2_stall` 0.821，`aiv_time` 4.351→4.557（更差）：等待只是在 pipe 间搬家。② `r4_fuseup`（跨 dtype `T.copy` 融合 split+up-cast，每 tile 少 2 条 vector 指令，L0 9/9 PASS）：8k 6.41→**7.29（−13.7%）**、4k 4.89→5.10（−4.3%），`vec_time` 2.466→2.667、`mte2_wait` 0.754→0.850、`vec_total_cflt` 0.128→0.199——反向优化。③ `r4_vcastslice`（down-cast 直写 `o_stage[0:rows, 0:half]` 静态切片）：编译期 `src/target/codegen_npuir_dev.cc:175` shape check 失败，v-op 目标必须是整 buffer（静态切片同样非法）。**残余空间量化边界**：4k = 1.7 launch/ramp + 3.0 aiv、8k = 1.7 + 4.6 aiv；aiv 内 ~1.8–2.0 µs 为每核骨架，每 iteration 仅 0.56（4k）/0.73（8k）µs 可动；波次 8k 119/120、4k 79/80、2k 40/40 已整数最优，再降一档需 tile ≥103 行 ≈263.7 KB > 176 KB UB。

**本轮否决（已实证，非假设）**：`T.Pipelined(num_stages=2)` **−6.8%**（多缓冲把 tile 压到 26 行、波次 3→8，描述符翻倍）；`tile=60` **−7.4%**（137 tiles = 3.4 波，未整波）。条目化见 queue VP-2026-0174 / VP-2026-0175。

**60% 不可达（官方三档精确算术）**：`neox-1d-2k-d64`(7.73%) / `neox-1d-4k-d128`(25.28%) / `neox-2d-b2-s2k-h32`(49.04%) 各自 `ideal/地板` 上界 15.5% / 63.1% / 100% ⇒ 均值上限 **(15.5+63.1+100)/3 = 59.5% < 60%**（差 0.5 点）。该上界已取最乐观口径（用 1.9 µs 空 kernel 地板而非实测 3.52 µs on-shape 地板）；换 3.52 µs 后三档上界降至 8.3% / 34.0% / 86.3%（均值 ≤42.9%），而 2k-d64 现测 7.73% 已占 8.3% 上界的 93%。

**可复用判据**：① plateau 判定前同 buffer/tile 的最小工作量探针是必要证据，流量/时延反算会把固定项误读为「带宽没打满」（上一轮即据此误判已贴 roofline）；② 小 shape 的残余收益只来自波次几何（`n_num` 贴 `cores × iters` 整波），不来自带宽或 op 数削减；③ `T.Pipelined` 在 UB 预算紧的逐元素算子族上收益方向为负（多缓冲以 tile 减半为代价）；④ `aiv_time ≫ max(pipe)` 不构成重叠缺口证据——本轮 4k 的 `aiv_time` 3.03 vs `vec` 1.50 中，缺口主体是 1.8–2.0 µs/核骨架，可动量 = `(aiv_time − 骨架)/iters`（4k/8k 仅 0.56/0.73 µs）；零额外 UB 的轮转预取在该判据下无收益（见 Round 4 段）。

---
id: CASE-dsa-decode-gather-floor
kind: case
family: [attention, sparse-mla, gather]
apis: [T.copy, T.Parallel, T.Pipelined, T.gemm]
dtype: [fp16, bf16]
device: Ascend910B3
status: verified
origin_task: TileOPs-_sparse_mla_kernel-20261101T014649Z
toolchain: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0（msprof 25.5.0）/ torch_npu 2.7.1.post4 / Ascend910B3（20 AI Core）/ 2026-11-01；TILELANG_ASCEND_MODE=Developer MixCV
repro: repro-missing
---

### 稀疏 MLA decode（逐行动态索引 gather 主导）Stage 4 档案：gather 成本 ∝ 条数 ⇒ 40% 结构性不可达，天花板 7.0–7.5%

`examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/`（任务工作区，provenance 允许失效）。终态 DONE，`perf_feedback.action=archive`（`[DESIGN_LIMIT]`），单 Stage 4 + attempt 1 runtime fail 4869 s → attempt 2 complete 420 s。**触发条件**：attention/sparse-MLA/paged-KV/MoE 路由类算子以「运行时索引 → 每行一条 `T.copy`」从 GM 取数，且 profile 显示 `aic_cube_ratio` ≈1e-4、`aic_scalar_wait` 占满、`mte2` 近零；或 roofline 把 gather 建模为「字节数 ÷ 带宽」。

**核心归因（三 workload 一致）**：逐行 1152 B copy 成本 **1.341 / 1.349 / 1.351 µs/行**；改变目的域（shared→UB，`p_ubrow`）**−0.74%**、改 `T.Parallel(Dqk)` 逐元素众集（`p_ubpar`）**+0.04%**，而 64 条 → 1 条连续 copy（数学无效的降级探针）**18.05×**（smoke 9045.26→501.15 µs）⇒ 成本 ∝ 条数，与目的域/向量级形态/`block_N`/`num_stages` 无关。DESIGN §1.6.0 R4 的 4.0–9.4 ms 估算 vs 实测 566.6 ms，偏差 60–142×。条目见 [CONST-dyn-row-gather-opcount-dest-independent](constants.md)。

**量化上界**：零-gather 地板 26525.50 / 6828.78 / 1793.82 µs ⇒ Ratio 上界 **7.52% / 7.31% / 6.96%**；40% 对应 4987 µs，比地板仍慢 5.32×/5.47×/5.75×。gather 免费时最优 ≈7.0–7.5%（相对 0.35% 约 20–21×）。

**三条封路（已实测）**：`BH=64`（L0C 144 KB > 128 KB、UB 264448 B > 196608 B，编译失败）；更大 `block_N`（行地址随机不可连续）；k 路拆分（`hivm.hir.load` 的 `pad_mode` 仅左右 padding 且前端未暴露，见 capability-gaps `CG-2026-0025`）。结构与参照实现的移植阻塞在 Developer 非持久 grid 内无法插相位（CG-2026-0010 / TRAP-DEVMODE-PERSIST-GEMM）。

**可复用判据**：① 目标可达性先用「删掉瓶颈操作」的降级探针测地板，再决定是否投入参数/结构搜索（本例 <2 分钟采集即裁决 40% 不可达）；② 稀疏索引 gather 的优化维度只有「条数」（行数/批共享/共享前缀），不是字节数或目的域；③ 该地板与 launch 地板互补——前者答「瓶颈免费能到多少」，后者答「空 kernel 地板多大」。

---
id: CASE-fp8-lightning-indexer-stage4
kind: case
family: [attention, attention_indexing]
apis: [T.gemm, T.copy, T.cast, T.reduce_sum, T.vsel, T.vadd]
dtype: [fp16, bf16, fp32]
device: Ascend910B3
status: verified
origin_task: TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z
toolchain: tilelang 0.1.2+63c8a2e6 / CANN 9.0.0 / torch_npu 2.7.1.post4 / Ascend910B3（1800 MHz）/ 2026-11-01；Developer MixCV
repro: repro-missing
---

### FP8 Lightning Indexer（fp8 无链路 → bf16 路径）Stage 4 档案：口径误用纠正 + epilogue fp16 1.92×

`examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/`（任务工作区，provenance 允许失效）。终态 DONE（`design_limit_signal: none`），单 Stage 4 attempt 3478 s、零重试；官方 report `examples/TileOPs/reports/tileops/20261101_120609_257701_FP8LightningIndexerOp/`。**触发条件**：官方 KPI 是 `Ratio(%)`（roofline 型），且 profile 显示 cube 占比 <10%、AIV scalar+vec 占 >50%；或上一轮把「写出口径」当成算法下界（空间被高估一个数量级）。

**口径拆解（本档案的核心）**：153 µs = `S·Skv·4B = 268.4 MB ÷ 1.75 TB/s` 是**纯 Logits 写出口径**（零计算零输入），任何真实实现都到不了；官方 `_parse_roofline` 的 951 µs 用**硬件计数 op**：cube 2.7488e11 + vec 9.566e9 = 2.8444e11 ÷ 299.09 TOPS。⇒ 真瓶颈是**计数向量 op**：fp32 下 vec 单独需 2280 µs > cube 理想 932 µs，即 fp32 epilogue 的结构上限 **41.7%**；fp16 使计数不变而 vec 峰值翻倍 ⇒ 上限 ~83%。条目见 queue VP-2026-0179。

**采纳（1.92×）**：epilogue 全链 fp16（L0C→UB 转换拷贝 + relu/加权/头归约全 fp16，输出仍 fp32）+ 由 `acc_ub` 减半释放的 UB 余量把 `bN 64→128`：s4k 5449.5→2835.7 µs、官方 Ratio 8.76%→16.83/16.45%（s8k/s4k），`corr_diff` ≤1.7e-07。条目见 queue VP-2026-0180（PL-1.22）。

**否决（已实测）**：`bQ 8→16` +68%（换轴应抬 bN 而非 bQ）；`bN=128/st3` 平区（<5%）；`bN=256` 编译失败（CG-2026-0025）；树形 `vadd` 归约不可表达（`T.vadd` 不接受 Var 偏移 region，queue VP-2026-0182）。

**40% 未达但非设计层天花板**：40% 需 1.74 µs/tile，串行 pipe 理想 1.10 µs/tile（Ratio 63%），实测 4.01 µs/tile；剩余 2.9 µs/tile 是标量地址/同步/未隐藏 DMA（需「一次交接多 kv tile」或持久化 grid）。按工作计数下界（1502 < 2378 µs）不能断言结构不可达，故不产出 `[DESIGN_LIMIT]`——与 `CASE-engram-decode-mcore-ratio-ceiling` 的「不可达」结论形成对照。

**可复用判据**：① Ratio 型 KPI 先拆口径再排候选（写出口径 ≠ 下界）；② 比值型指标不奖励减少冗余工作（窗口外 tile 占 96%，skip 对 Ratio 中性）；③ epilogue dtype 是双刃杠杆：既降时间又抬上限。

---
id: CASE-mha-decode-paged-stage4
kind: case
family: [attention, decode, mixcv]
apis: [T.vcmp, T.vselect, T.vsub, T.vclamp, T.vcast, T.vmul, T.vbrc, T.arange]
dtype: [fp16]
device: Ascend910B3
status: verified
origin_task: TileOPs-_mha_decode_paged_kernel-20261101
toolchain: tilelang 0.1.2+5b062018 / CANN 9.0.0 / torch_npu 2.7.1.post4 / Ascend910B3（1800 MHz）/ 2026-11-01；Developer MixCV
repro: repro-missing
---

### paged decode MHA Stage 4 档案：采数伪影纠正 + int32 列掩码标量化（7.6×）

`examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/`（任务工作区，provenance 允许失效）。终态 DONE、`design_limit_signal: none`；官方 report `examples/TileOPs/reports/tileops/20261101_184433_143505_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/`。**触发条件**：官方 Ratio 仅 0.0x% 且 profile 显示 `aiv_scalar` 占单 block 八成以上；或沿用旧 report 的极低 Ratio 作基线。

**① 采数伪影（先纠尺子再调优）**：旧 report `20261015_074433_562940_…` 的 `captured_op_count=8–9`、`kernel_name_requested=None`，解析器按时延匹配错选 `Cast_…/BroadcastTo_…/ZerosLike_…` 框架小算子 ⇒ 1.82–2.52 µs / 0.012–0.017%。带 `--kernel-name=split_partial` 校准后 `captured_op_count=1`，真值 12.56–27.04 µs / 1.20–5.51%（四档均值 0.31%→2.38%）。**判据**：任何官方 Ratio 低于同族经验一个数量级时，先查 `captured_op_count` 与 `op_names` 再谈优化空间。

**② 根因与修复**：`hivm.hir.vcmp` 在 `[16,256]` int32/bool 列掩码上降级为逐元素标量（`aiv_scalar` 34.85 vs `aiv_vec` 0.92 µs），编译警告 `will execute by scalar instruction` 为唯一指纹；改算术掩码（复用既有 int32 buffer，零新增 UB）后 150.27→19.86 µs。终版四 workload 293.17→27.12 / 150.27→18.17 / 77.92→12.59 / 158.29→22.20 µs（几何 7.6×），整算子（split+combine）15.9–30.1 µs。

**③ 波次平衡**：Phase-A grid = `batch*heads*ns_eff`，取 `ns_eff = min(num_split, nkv, max(1, 20//(batch*heads)))`——16 块（1 波）对 batch*heads=16 最优，8 头取 ns=2 得 16 块；`ns=1` 在 8 头只剩 8 块反而回退（+39%）。区间 12.59–27.12 µs。

**④ 否决（已实测）**：runtime scalar `if` 包裹掩码链（3 变体全部 L1 失败且 max_diff 逐位相同 2.719e-01 ⇒ 条件体未生效）；page-table 标量载入探针仅 0.86 µs；`num_stages=3` +2.4%（<3% 噪声阈）；全 fp32 掩码中间量（+48 KB UB 风险 + causal parser 缺陷）。

**⑤ 剩余空间（未实证）**：K/V 为 `[N_kv,H,D]`，单 head 切片每行仅 128 B ⇒ 每页 512 条 128 B 段；「按 KV 段切、块内循环 head」可把段数 ÷8，估计再降 2–3×，但改动 grid 语义与 8 份在线 softmax 状态（UB 8×`[16,D]`），预算内未实现。**可复用判据**：口径（captured_op_count）→ 标量化警告 → 波次几何 → 段宽，四步按序排查，前两步零实验成本。

---
id: CASE-dropout-stage4-ratio-opcount
kind: case
family: [regularization, dropout, elementwise]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: TileOPs-_make_dropout_kernel-20261101T145102Z（R6）/ TileOPs-_make_dropout_kernel-20261102T065942Z（R9，第 9 轮）
toolchain: tilelang 0.1.2+d58388995fc943c5d79035e2a312116ea6d086cd / CANN 9.0.0 / torch_npu 2.7.1 / Ascend910B3 / 2026-11-01；R9 段 tilelang dev build `/mnt/workspace/tilelang-ascend-npu-ops/build` + CANN 9.0.0 / Ascend910B3 / 2026-11-02
repro: repro-missing
---

### dropout Stage 4 档案：调优成果未上线（wrapper 默认参数未同步）+ Ratio 与 op 数同源

`examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/`（任务工作区，provenance 允许失效）。终态 DONE。**触发条件**：Stage 4 已把 tile/`block_size` 类默认参数调大而官方 Ratio 长期不动；或同一算子的「真实时延明显下降但 Ratio 持平」。

**① 调优成果未上线（本档案最高价值，流程缺陷）**：R1–R5 把 `block_size` 1024/2048 → 8192 并已落到 `perf_opt/{op}.py::_DEFAULT_BLOCK_SIZE`，但 wrapper `dropout.py::_DEFAULT_BLOCK_SIZE` 仍是 1024/2048 ⇒ 官方 report 长期停在优化前口径：`20261101_221636_325181` 三档 44.07/49.89/29.69%（均值 **41.21%**），其 `config` 列明写 `block_size 2048/1024`；同步 wrapper 后 `20261101_223520_519180` 同一 kernel 立刻 56.02/62.10/58.39%（均值 **58.84%**）。**判据**：perf_opt 切换/收敛时必须机械比对 `perf_opt/{func}.py` 与 wrapper 的默认参数常量（`ast` 解析模块级 dict 字面量，逐 key 比对，差异 → 阻断切换）；不靠人眼核对注释。

**② 增益来源（R6 单点升级）**：p=0.5 时掩码阈值退化为 hash 符号位，`vclamp(h,0,1)` 取代 `vabs`+`vsub`+`vclamp`（−2 个 int32 向量 op）⇒ 三 workload 一致 −20.7%~−21.4%（27.898→22.116 / 73.929→58.118 / 28.735→22.716 µs），op 数 fp16 13→11。

**③ Ratio 与 op 数同源（口径）**：时延 −21% 而 ideal 由 ≈15.70 → ≈11.96 µs，官方 Ratio 58.84%→**58.27%**（−0.57 点）。这是 `Ratio = ideal/duration` 的分子分母同源性，**不是回退**；11-op 版本真实快 21%，不采用 13-op 版本去刷 Ratio。对照原始未上线 8192 时的 41.21% → 58.27% 仍是真实提升。

**④ 已量化后否决**：波数守卫收益上界 = 尾波损失，W1/W3 仅 1.54%（512 tile/13 波）、W2 为 0%；3-op counter hash 统计不合格（lag1 自相关 −0.389 / +0.723，block-χ² 87 / 2578 @ dof 4095）；`vbitcast` 融合「掩码 cast + scale 乘」被后端拒绝（`vmul requires the same element type`）；bs>8192 `ub overflow`。

**⑤ 剩余空间**：有效 R+W 0.72–1.48 TB/s vs 同机逐元素天花板 2.49–2.67 TB/s（1.7–3.5×），差距全部落在 11 op 中 7 个 hash+掩码 op 上，即「前端无 RNG 原语」（CG-2026-0026，任务工件旧引 CG-2026-0021）决定，非 tiling 可解。

**⑥ 验收口径切换：声明式 roofline 取代计数 roofline（R9，用户决定）**。主口径改为 `DropoutOp.eval_roofline()`（`examples/TileOPs/tileops/ops/regularization/dropout.py` L142–166：`flops = N`、`bytes = 2·N·elem_bytes`、`ideal = max(bytes/1.75 TB/s, flops/peak)`）——**分子与实现无关**，只由 N/dtype 决定（本轮三档 `ideal_declared` = 9.587 / 23.9675 / 19.174 µs）。副口径（A）= msprof 计数口径 `perf_tops/computility`，其分子是工具从发射指令反推的 ops（随实现变）。同一 final（官方 report `20261102_154948_981690_DropoutOp`）两口径并列：**B 47.84/45.07/93.08（均值 62.00%）** vs **A 55.15/64.57/58.07（均值 59.26%）**；R8 对照 report `20261101_235107_195598`：B 43.66/41.25/85.36（56.76%）、A 54.47/63.02/57.31（58.27%）。**两口径方向可相反**：R6 减 2 个 int32 op 使 A −0.57 点（分子与分母同降）；R9 减 1 个 full-tile op 使 A +0.99 点（counted ops 仅 −4.5%，AI 5.5552→5.3052，而 duration −8.3~8.7%）⇒ 计数口径对小 kernel 系统性偏低，且与真实时延不同向；报告须声明主口径并并列 A/B。注意分母协议：官方 report 单采样（20.04/53.18/20.60 µs）与 `perf_records`（`--launch-count=20` 中位，19.907/53.212/20.536 µs）差 ≤0.7%，切换口径时须固定测量协议。

**⑦ R9 bit-exact 融合（vbrc full-tile 账单）**：`T.fill(h, off+seed)` + `T.vadd(lane, h, h)` 合并为「**1 元素 UB 标量 buffer** `sbase = T.alloc_var("int32")` + `sbase[0] = off+seed` + 广播 `T.vadd(lane, sbase, h)`」。字面 `T.vadd(lane, T.cast(off+seed,"int32"), h)` 被前端拒绝（`AscendBinaryOp.buildTirCall` 只接受 int/float 字面量与 `tir.expr.Var`，其余走 `_to_region` 报 `Add has no attribute buffer`），`with T.let(s, off+seed)` 绑定的 `tir.Var` 使前端段错误（rc=139）。NPUIR 每 device module census：`vbrc` **2→1**（被删的是写满 tile 的 32 KiB int32 广播），`sbase[0]` 存储不产生 `hivm.hir.v*`。实测：官方时延 21.96→20.04 / 58.10→53.18 / 22.46→20.60 µs（−8.74/−8.47/−8.28%；med-20 为 21.896→19.907 / 58.151→53.212 / 22.491→20.536，−9.08/−8.49/−8.69%），`aiv_vec_time` 19.054→17.111 / 54.468→49.665 / 19.603→17.662 µs（−8.8~−10.2%），B 均值 56.76→62.00%。`bit_exact=True`：5 shape × {ones, randn} 全部 `torch.equal`（`perf_opt/probes/round9_bitexact.py` / `logs/round9/bitexact_a.log`），掩码逐位相同，ones rate 0.49987–0.49993 不变，p=0.75 与尾块 shape 亦逐位相同。分支 B（int32→fp16 单跳 `vcast`）被方言拒绝 → **CG-2026-0027**，故逻辑 op 11→10（物理 11，`vclamp`→`vmin`+`vmax`），未达 9。

---
id: CASE-attention-gqa-sw-varlen-stage4
kind: case
family: [attention, gqa, sliding-window, varlen, expert, mixcv]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z（Stage 4 Round 10 / M1）；Round 11 nqb=2 编译探针 TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z；Round 12 三路可行性判定 TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z；Round 13 G=2 head-group batching 采纳并转正 TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z；官方 report 逐 case 隔离补跑 TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z；同一算子早期轮次见 opt_log §5 rounds 1–5（同一 task_id 谱系）
toolchain: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04（Round 11 / Round 12）· 2026-11-05（Round 13）· 2026-11-08（官方 report 补跑；report 记录 pip `tilelang 0.1.1.10+ubuntu.20.4.cann900`、repo commit 9cfae94c、msprof 25.5.0）
repro: repro-missing
---

### GQA-SlidingWindow-Varlen Stage 4 档案：op 数瘦身 1.83× 后，片内重叠被编译器封堵

`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/`（任务工作区，provenance 允许失效）。**触发条件**：Expert 两相位 persistent MIX kernel（Cube `T.gemm` ×2 + Vector softmax/掩码）在「op 数 / 载体 dtype」类优化收束后，评估手写 L1 双缓冲或更深任务流水以换取重叠；或同类算子出现 `mte2` 单管道独大（≈65% 墙钟）而四管道和/墙钟 ≈1.5× 的画像。

**① 采纳的三项（1.83× 几何）**：T1 跨任务深度 2 流水（TASKDONE 屏障移除 + `l1_q` 独立；flag 判据 `2·L_max ≤ 15`）；T2 列 OOB 惩罚链运行期守卫（−36.2%，见 VP-2026-0203）；T3 fp16 路径 f32 S 载体（−3.1%，AB 裁决）。8 个 tune workload 全部 −42.7%~−47.1%，几何 **1.83×**（8b/70b × short/long × fp16/bf16，msprof op Task Duration median-of-20）。

**② 四管道账（8b-long fp16，round 9 final，20 launch）**：`mte2 0.650 / mte1 0.307 / cube 0.243 / fixpipe 0.315`（比值对墙钟；70b-long 同构 0.648/0.303/0.241/0.312）——四者之和 **1.51–1.54 × 墙钟**，即结构本身已有 ~50% 重叠；mte2 单管道理论上界 **1.52×**。该比值是「串行」与「已重叠但某管道独大」的区分器（与 BP_pipeline_overlap、PL-1.21 的已重叠判据互补；数据条目见 queue VP-2026-0201）。

**③ 片内重叠为何不可达（round 10）**：4 条手写双缓冲分支 M1a/M1b/M1d 全部 `bishengir-compile` 失败（SIGSEGV 或 >15 min 不收敛，基线 26–42 s）；唯一可编译的零新增缓冲形态 M1c（把与跨引擎 flag 无关的 V 装载提到 `sync_block_wait` 之前）两个 long **同向 +1.4% 回退**——mte2 忙比 0.634→0.618、四管道和/墙钟 1.507→1.474（重叠更差）而墙钟反升 ⇒ 缺口在 L1 端口读写互拖（MTE2 写 vs MTE1 操作数读），不是 MTE2 空转。编译封堵点与探针阶梯见 `traps-compiler.md#TRAP-l1-4th-live-buffer-copy-segv`、能力缺口 `CG-2026-0029`。

**④ 其余否决（同谱系）**：折叠式掩码 +0.1%（判据 = 无条件执行的 tile op 数，非「哪些块需要掩码」）；O 载体 f32 +0.08%；α 递推瘦身 +0.01%；bm=32/bn=512 +79%；bm=80 UB 溢出（204.8 KB > 192 KB/AIV）；bn=128 +26%（`L_max=11 ⇒ 2·L_max=22 > 15`，pipe2 回落单槽）；vselect 备选编译 8 min 56 s 未终止；f16 softmax 链被 codegen 拒绝（`T.vsub(f16 tile, f32 scalar, f16 tile)`）。`(bm,bn)=(64,256)` 同时是 UB 与 flag 预算的上界交点。

**⑤ 剩余空间**：Cube 侧 `aic_scalar_ratio 0.95`（flag 等待 ~1073 µs）与 mte2 独大并存；理论 1.52× 被 ③ 的编译器封堵点吃掉了可实现的第二条操作数缓冲。nqb=2 同属「循环体翻倍」，round 10 曾预期撞同一编译墙；round 11 的编译探针证伪了该外推（见 ⑥）。

**⑥ nqb=2 编译探针（round 11：CONVERGED，但不可晋升）**：验证「相邻两 q-block 共享 K/V 装载」能否在 ≤3 块活 L1 + 非成对展开 body 下表达。**实测形态**：`num_logical = ceildiv(ceildiv(max_seqlen_q, bm), 2) * H * B`；活 L1 恰 3 块（`l1_q [2*bm, dim]` 32 KB @dim128 + `l1_a [bm, 256]` 32 KB + `l1_b [bn_eff, dim]` 64 KB；原独立 O 缓冲去掉，pass-2 只保留一个 q-block in flight）；pass-1 单 body（一次 K load + 一次 `M=2*bm` 融合 gemm + 一次 S store，两 q-block 的 S tile 是 L0C 相邻行）；pass-2 单 body 在 `for slot in T.serial(2)` 内用运行期 `slot` 取 qb0/qb1 的 P 行窗口与 O 行窗口；flag id = `s*L_max + l`（`2*L_max ≤ 15`）。**编译结果**：dim=128 冷编译+首启 **8.2 s**（基线 26–42 s；msprof run 日志 bench 首调用 9.2 s、wall 31.3 s 含采集），0 次 SIGSEGV；L0 的 dim=64 冷编译 ~29 s。**性能不成立**：仅共享 K、V 仍逐 qb 载入，且未携带 T1 depth-2 流水 ⇒ 8b-long fp16 单 run（msprof Task Duration，1 launch）**2282.1 µs vs 基线 2003.35 µs（median-of-20）= +13.9%**；单 run 且缺 V 共享/流水，不构成候选收益结论。**新 device fault**：dim=64 且样本 `kv_len % bn_eff(256) == 0` 时 `aicore exception: The operation address of L0A exceeds the maximum range of L0A / mte load3d out of range`（t256/t512/c3/c4 FAULT；k=128/384 PASS；dim=128 真实 workload 正常）——编译通过掩盖运行期故障，L0 门禁当前不通过（`CG-2026-0030`）。**正收益路径**：V 共享要求 Cube pass-2 变 band-outer（每 band step 一次 V load 供两个 q-block），Vector 也须按 pair 拆 pass-1/pass-2 ⇒ 两个 q-block 的在线 softmax 状态同时驻留 UB（UB 翻倍）——下一道门是 UB 预算，不是编译器。**（R12 谓词修正，见 ⑦）** 上述 `kv_len % bn_eff(256) == 0` 只是相关性：真正触发条件是「**动态 `T.serial(loop_count)` 的最后一轮 band tile 满宽**（tnc == bn_eff == 256，dim=64）」——`q=128/k=384`（l=0 处有满 256 块但非末轮）PASS，`q=384/k=384`（末轮满宽）FAULT；R11 记的「k=384 PASS」反例在 R12 等价驱动下按 `q=384/k=384` 运行为 FAULT、按 `q=128/k=384` 才 PASS（两轮 shape 表是否逐字相同未确证）。绕法 = `dim==64 → bn_eff=min(bn_eff,128)`，L0 4/4 + contract PASS，dim=128 的 IR 不变（8b-long 单 run 2278.72 vs 2282.1 µs，−0.15%）。条目见 `traps-runtime.md#TRAP-l0a-load3d-out-of-range`（queue VP-2026-0211）、缺口 `CG-2026-0030`。

**⑦ V 共享的 UB 预算核算（round 12：容量门通过，否决来自流水）**：以 round-11 nqb2 探针 lowered `kernel.mlir`（Vector scope 与冻结 kernel 字节一致）逐行求 `memref.alloc … address_space<ub>` 之和，实测 **157,952 B = 154.25 KB** 恰等于 user `T.alloc_ub` 逐 buffer 之和（**plan total 另含 38,656 B 编译器 staging**）。〔2026-11-05 R13 更正〕该等式只说明**本结构无 live-range 复用**，不能推广为「分配器不做生命周期复用」——R13 的 4 缓冲探针实测分配器按 live range 复用（disjoint 变体第 4 块 alias 到 slot 0、concurrent 256 KB 报 `ub overflow`）；条目与 repro 见 `constants.md#CONST-ub-alloc-live-range`（原 queue VP-2026-0214 的「静态加和」泛化已按此否决）。逐 buffer（dim=128，per-AIV 行 32）：`ub_f16_N [32,256]f16 16.00` / `ub_f32_N`=S `32.00` / `ub_f16_D [32,128]f16 8.00` / `ub_f32_D 16.00` / `acc_o 16.00` / 9×`[32,1]` 标量 1.125 / `scales [224,1]` 0.875 / `ub_diff [32,256]f32 32.00` / `ub_pen 32.00` / `ub_rowi`+`ub_thr` 0.25。三个投影：**A** 两个 q-block 常驻状态翻倍 + 逐 band 共享 scratch（S/diff/pen/D 先 qb0 再 qb1）= **171.50 KB（余 20.50，10.7%）**；**B** = A + 两份 S tile 常驻 = **203.50 KB（超 11.50）**；**C** = 每个 `[half,*]` 缓冲翻倍 = 282.25 KB（超 90.25）。⇒ 共享 scratch 形态**拟合**；坚持两份 S 常驻时的最小收缩 = S 载体落 f16（−32 KB → 171.5 KB，同族前序实测 +3.1%）；bn 256→128 省 48 KB 但 +26% 且杀 pipe2，bm 64→32 +85%，`acc_o` 跨整 band 活跃不可时分。**Cube 侧不是编译器封堵而是流水**：V 共享可压到 **3 块活 L1**（Q/P/V：V 装载 hoist 出 `for slot in T.serial(2)` 的 pass-2 循环 + qb0/qb1 复用同一 P 缓冲、两次 gemm 间加 WAR 同步），4 块形态（Q,P0,P1,V）才撞 `CG-2026-0029`；代价 = band-outer pass-2 把两次 PV gemm 串行化、pair-split 下 Vector 的 softmax/PV 重叠亦丢失 ⇒ 「UB 门」通过后下一道门是流水，不是容量。

**⑧ 双 kernel 拆分（round 12 实测 3.93×，否决）**：K1 = QK^T + online softmax → `ws_p/ws_scale/ws_ell/lse`，K2 = P@V + 延迟 rescale + 归一。**L0 PASS 4/4 + contract**（fp16 + bf16）、K1+K2 编译 12.5 s、无 SIGSEGV ⇒ 正确性与可编译性都不构成障碍。launch 边界强迫整个 P 落 HBM：`ws_p [4096,64,1792] f16 = 896 MiB`（该 kernel 单 launch 内 P ring 仅 **9.2 MiB**）+ `ws_s 8.75` + `ws_o 2.19` + `ws_scale 7.0` + `ws_ell 1.0` = **914.9 MiB**。实测（`msprof op` Task Duration，各 3 次）：**K1 5533.86 µs**（aiv vec 2701、aiv mte3 2114、aic mte2 679、cube 238、`aic_wait_id_sum` 4985）、**K2 2336.00 µs**（aic mte2 1475、cube 245、fixpipe 305）⇒ 合计 **7869.86 µs = 3.93× 基线 2003.35 µs**。机制两条：① P 写落到 896 MiB 非 L2 驻留区 ⇒ mte3 256→2114 µs、vec 1189→2701 µs（mte3 反压）；② 单槽 TASKDONE 握手吃掉基线的跨相位重叠。**跨族对照**：DSA 谱系的两阶段拆分成败点相同而结果相反——那边 P/partial 是 `(B,ns,8)` 级（9 KB@b4），拆分只付两次发射（`CASE-dsa-decode-gather-floor`、`CASE-engram-decode-mcore-ratio-ceiling`）；判据 = 先算 P 全量物化 `B·H·Skv·D·elem` 与 L2/基线 ring 之比。host 分块保 P 在 L2 的变体未实现，地板外推 = K1 vec 1.19 ms + K2 mte2 ~0.5 ms + 0.6 ms 发射 ≈ 2.3–3.2 ms（1.15–1.6×）且仍丢 QK/PV 重叠（未实证假设）。条目见 queue VP-2026-0213。

**⑨ 40% 终局判定（round 12，否决）**：目标 1202 µs（1.67×）。当前结构字节账下**单管道下界 = max(mte2 1291, mte1 609, cube 483, fixpipe 625) = 1291 µs = 1.55×**——上界已短于目标，任何重叠改造都不可能达标；要抬高上界须削减 mte2 字节（K+V 共享）并同时藏住它，而三路各有实测否决：K-only 共享 +13.9%（丢 depth-2）、V 共享通过容量门但需 band-outer pass-2 + pair-split（⑦）、双 kernel 3.93×（⑧）。绑定约束全部在工具链层：第 4 块活 L1（`CG-2026-0029`）、dim=64 末轮满宽 L0A 越界（`CG-2026-0030`，绕法 bn=128 仅限 dim=64）、launch 拆分的 P 物化成本。**判据（事实）**：Ratio 硬目标与 `max(单管道)` 直接比对即可终判（本例 1.55× < 1.67×），先例检索 + morph 对照只在上界达标后才需要；该上界只对实测过的结构族有效（减少搬运/换载体可抬高它）——「在本结构字节账下」是该结论的适用域，跨结构族不成立。

**⑩ head-group batching G=2（round 13，采纳；本算子首个结构级正收益）**：`heads/heads_kv = 4`（8b 32/8、70b 64/8）⇒ **两个 query head 在块内共享同一个 KV head 的 K/V**。`head_group = 2` 仅当 `groups >= 2 and groups % 2 == 0 and heads % 2 == 0`（trace-time），否则退回单 head 体（`H/Hkv = 8/8` 的 L1 例覆盖回退路径）。Cube 每 band step **一次融合 gemm**（`l1_q[G*bm, dim]`、head `gh` 打包在行 `[gh*bm, gh*bm+tmc)`，`size=[bm+tmc, dim, tnc]`、`l0_c[2*bm,256] f32 = 128 KB` 整块 L0C），每 band step 各一次 K（pass-1）/V（pass-2）`GM→L1`；Vector 在 band 循环**内**用运行期 `for gh in T.serial(head_group)`（trace-time 展开会再加一份 38.7 KB staging 并溢出），flag 预算 `2·L_max = 14 ≤ 15`。**UB 账本**：G=1 current best user 157,952 + staging 38,656 = 196,608 B（零余量）；未瘦身 G=2 = 175,616 + 38,656 = 214,272 B（超 17,664 B）；`ub_f16_D/f32_D` 别名到 pass-1 的 `ub_f16_N/f32_N`（`[half,dim]` 前缀、两段生命周期互斥）**−24,576 B** ⇒ 151,040 + 38,656 = **189,696 B（185.2 KB，余 6,912 B / 3.5%）**，编译通过、L0 4/4 + contract PASS（fp16 1.953e-3 / bf16 1.562e-2 tier2 / lse 1.9e-6，与 G=1 逐位相同）。分配器语义与 staging 税见 `constants.md#CONST-ub-alloc-live-range`。**性能（msprof Task Duration，交错 A/B 按臂中位）**：8b-long fp16 2017.22→**1767.48 µs（1.141×）**、70b-long fp16 3991.16→**3487.04（1.145×）**、8b-short fp16 344.56→**312.60（1.102×）**、bf16 1.125–1.144×。**归因为纯 GM→L1 带宽**：`mte2 1306.03→758.27 µs（−42%）`、`GM_to_L1 126,064→77,280 KB（−38.7%）`；融合 `M=bm+tmc` gemm 使 L0B 操作数装载减半（`mte1_ratio 0.307→0.196`）、cube +5.5%、fixpipe −9%、scalar −11%；四管道和/墙钟 1.535→1.176。**官方 benchmark 8 tune 用例** 14.32/14.20/25.02/24.35/15.78/15.53/25.45/24.77 → 均值 **19.92%**（旧 17.53%），10/10 PASS（1023.91 s）。**B（causal full-band proof）不建分支**：pass-1 每 band step 已有 3 个运行期 `scf.if`（因果上界 / 左窗 / ragged 列 OOB）；8b-long 每 task/half 3.75 个 band tile 中上界链执行 1.0 次、左窗 0.5 次、列 OOB 0 次 ⇒ **1.5/3.75 = 40%**，且这 40% 全是跨边界、必须掩码的 tile，跳过量为 0。**C（V 转置）跳过**：R12 实测 long 净 −0.28 µs（`V[total_k,Hkv,dim]` 打包形态下 `[D,L]` 操作数是跨步 GM 读），预期 ≤5% 且符号为负。**G=4 当前被容量封堵**：`acc_o[half,4*dim] f32 = 64 KB（+32 KB）` ⇒ user ≈186 KB + staging 38.7 KB = 224.7 KB > 192 KB；且 `l0_c[4*bm,256] = 256 KB > L0C 128 KB`（须拆两个 `M=2bm` gemm）。解锁次序（参数/shape 级，非设计级）：① `acc_o` 落 f16（需精度 A/B，省 32 KB）；② `ub_diff`/`ub_pen` 别名（每 band step 重算 `j-i` 基，约 +3 tile-op/step）。**报告口径问题（round 13）**：官方 `tileops.reporting.cli run` 的 correctness 阶段因 G=2 按 shape 逐个重编译，40 min 只跑到 8/26（全 PASS）、未生成 `run.json`；性能数据改用官方 benchmark harness 单跑（10/10 PASS），全量正确性来自 embedded `--level all` 51/51（L0 4/4 + contract / L1 26/26 / L2 12/12 / Boundary 9/9）。

**⑪ 官方 report 逐 case 隔离补跑（round 13-report，2026-11-08）——⑩ 的「跑不完」被同一内核以进程隔离形态跑完**。task `TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z`（无 kernel 改动，sha 保持 `6befe384…`）用 `perf_opt/logs/round13_report/report_isolated_varlen.py`（一 pytest id 一进程；`Popen(start_new_session=True)` + 超时 `os.killpg(SIGKILL)`；只对 timeout/signal/compiler-crash 三类瞬时失败重试；`--resume` 断点续跑；逐 case junit）跑完官方 CLI 等价流程：**correctness 26/26**（平均 max abs err 1.42e-04；前 7 例缓存命中 skip）、**benchmark 10/10**、`run.json` 生成、`status=passed`，run dir `examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/`〔工作区运行产物，provenance 允许失效〕。**8 档 Ratio**：8b-short 14.05 / 14.33（fp16/bf16）、8b-long 24.73 / 24.58、70b-short 15.50 / 15.54、70b-long 25.15 / 25.05；时延 314.83 / 308.91 / 1774.74 / 1787.26 / 570.73 / 569.89 / 3490.69 / 3508.15 µs。与 ⑩ 的差异只在执行形态（隔离 + 重试 + resume），故 ⑩ 记的「correctness 40 min 只到 8/26」是未隔离批量执行的产物，不是该 report 流程的固有上限。**同批硬故障两则**：① case 8（KV-cache + 非 causal + 双侧窗口，q=[64,128]、k=[256,512]、wl=64、wr=64、fp16）在批量跑中卡 26 min 并污染后续 case，**单独进程 53.8 s PASS**（`perf_opt/logs/round13_report/case8_standalone.log`）⇒ 单次卡死不构成「确定性挂死」证据；② `subprocess.run(timeout=)` 只杀直接子进程，`bishengir-compile` 孤儿占住设备使后续 case 串行化——修复形态即上述 `start_new_session` + `killpg`。条目化见 queue VP-2026-0227（P，隔离执行形态）、VP-2026-0226（D，卡死可判定性）、VP-2026-0208（R，runner 卫生追加第 ④ 则）。
