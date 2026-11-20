# 进化提案队列（Evolution Queue）

> TileLang-Op-Conductor 自进化机制的提案队列。由 `tilelang-skill-evolver` 维护（唯一写入者）。
>
> - **用途**：暂存未达合入阈值的 P 类（模式方法）提案与待人工审批的 R 类（流程规则）diff 提案；D/C 类（实测数据/案例索引）不进本队列，由 evolver 直接合入 pattern-library。
> - **字段规范**：见 `.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。
> - **状态**：`pending`（待确认/待审批）→ `verified`（Tier 1 证据达阈值待合入）→ `merged`；或 `rejected` / `expired`（90 天未决）/ `conflict`（与现有条目矛盾且无法自动裁决，留人工）。
> - **审阅方式**：Tier 2（R 类）提案经用户批准后由 conductor 调度 evolver（`mode=apply`）执行写入；Tier 1 达阈值后由 evolver 在下次蒸馏时合入。

## 字段速查

```yaml
proposal_id: VP-{YYYY}-{NNNN}     # 唯一 ID，由 evolver 递增分配
type: D / P / R / C               # D 实测数据 / P 模式方法 / R 流程规则 / C 案例索引
title: <一句话标题>
evidence:                         # provenance（ED-A 语义，2026-09-10 起）：来源 task_id + 出处描述——允许失效，不做路径存在性核验
  - <task_id + 出处（如 该任务 opt_log round 3 / 路径#定位——路径允许失效）>
repro: <知识域自包含脚本路径（pattern-library/repro/<id>.py，须存在且可执行）或 repro-missing 或 none>
toolchain_stamp: <tilelang build/commit + 设备 + CANN 版本，或 none>
target_doc: <目标文件仓库相对路径（pattern-library 相关指向 pattern-library/ 主题文件）>
delta: <五种合法动作之一：add / update / consolidate / negate / deprecate + 条目正文>
status: pending / verified / merged / rejected / expired / conflict
confirmations: {n}/{2}            # Tier 1 合入阈值：2 次独立证据（不同任务）；E-3 快速通道：另一上下文 repro_runner/ab_test 复现成功计 2/2
created_by: <task_id + 日期>
decided_by: evolver / human / -   # 裁决者
```

> **历史条目路径映射（2026-09-10 pattern-library 拆分）**：本文件中既有条目的 `target_doc: .../references/pattern-library.md` 与 delta 中的「§1/§2/§4」章节引用，合入时按 pattern-library/INDEX.md 的历史引用对照映射到主题文件（§1→layout/elementwise/attention.md、§2→traps-compiler/traps-runtime.md、§4→cases.md、§1.6 常数→constants.md）——历史提案文本不改写，合入锚点按映射解析。
>
> 完整字段规范：`.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。

---

## Pending

### Tier 1（P 类，2 次独立证据后合入）

## VP-2026-0003

- type: P
- title: 纯 Vector 一维逐元素 persistent 模板：num_kernels=min(ceil(N/block), aicore×2) + T.serial 静态边界 grid-stride + 动态尾块零起点切片
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/DESIGN.md#5.5（分核三要素）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 提案 + Transferable Lessons）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md（persistent nc=48 全档贴 copy 地板；grid-stride 实测最优访问模式）
  - examples/elementwise/vec_add_1d.py L31–L38（尾块处理官方先例）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py --level all
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24（纯 Vector 翻倍 48 vector cores）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「纯 Vector 一维逐元素 persistent 模板」（核数实查/纯 Vector 翻倍规则已有 ascend-constraints.md「物理核数限制」行，仅交叉引用不重复）：
    模板：num_kernels = min(ceil(N/block_size), aicore×2)；核内 T.serial(ceil(num_logical/num_kernels)) 静态边界（N 工厂特化保证编译期常量，可直接用 T.ceildiv+min 求值，见 §1.5 host 层常量折叠行）+ block_id = i×num_kernels + cid + if block_id < num_logical 屏蔽越界。
    尾块：tail = T.min(block_size, N−t0) + UB 零起点切片 T.copy(x[t0:t0+tail], x_ub[0:tail])——v-prefix 整 buffer 操作对 UB 陈旧段一并计算、只拷出 [0:tail]，正确性无碍。
    提取期注意：TileOPs wrapper 的 E3/E4 适配注释（如 threads= 取代）是提取期笼统预估，核数/grid 取值以 DESIGN.md 分核三要素为准，冲突时设计文档优先。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0004

- type: P
- title: BP_grid_stride_vs_contig_chunk：带宽饱和区连续分块映射回退 +3.2%（256M），grid-stride 聚集窗口对 HBM 混合流更优
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-3（v3_op2：256M fp16 1765.01 vs 1710.07，+3.2%；1M/16M tie）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/round3/v3op2_256m_fp16_bs4096
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：msprof op --kernel-name=main --launch-count=15 ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v3_op2.py --dtype float16 --N 268435456 --block-size 4096（对照 grid-stride 基线）
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_grid_stride_vs_contig_chunk：
    触发信号：带宽饱和区（mte2_ratio ≥0.95）考虑把 grid-stride 任务映射改为每核连续分块以改善「DRAM 局部性」。
    反证/判定：lerp_tensor 256M 实测连续分块 +3.2% 回退、小 N tie——grid-stride 的 N 核聚集移动窗口（fewer open pages / 行缓冲命中）对 HBM 混合读写流更优，「连续流局部性」直觉在混合流下反向；数据侧见 pattern-library §1.6 访问模式条。
    动作：饱和区维持 grid-stride；连续分块仅在纯读/写单一流向或 L2 友好 workload 复测后考虑。
    验证：Task Duration 非回退 + mte2_ratio 不恶化。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0005

- type: P
- title: 标量/分支削减优化在 MTE2 饱和区无效的判定式：mte2_ratio ≥0.95 时标量已被搬运隐藏，先查饱和再立项
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 静态尾块全档 tie/-0.6%；guard-free 机制证据 defer <0.3%）
  - 同文件 Baseline 表（≥16M mte2_ratio 0.95–0.99；1M 档 0.70–0.84 流水浅区）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：msprof op ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v2_op2.py --dtype float16 --N 1048576 --block-size 4096
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    update BP_ub_traffic_floor 触发信号，追加：「per-tile 标量/分支开销（动态尾块 T.min、if-guard、地址计算）立项削减前，先查 mte2_ratio——≥0.95（搬运饱和）时标量已被隐藏，实测静态尾块/去 guard 全档 tie/-0.6%（lerp_tensor 2026-09-07）；仅 mte2_ratio <0.9 的流水浅区（小 N 档 0.70–0.84）可能有收益」。数据侧见 pattern-library §1.6 标量削减判定式条。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0006
- type: P
- title: 精度失败定征顺序：dtype 隔离 → 违反率与 shape/block/核数无关性 → golden opmath 域小实验（避免盲目排查分核/内存层级）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Transferable Lessons（定征顺序实证有效）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py（定征输入：fp16 ~0.125% 违反、bf16 逐位一致、fp32 全过、违反率与 shape 无关）
- repro: 对 attempt-1 基线分 dtype 跑 --level L0，观察违反的 dtype 隔离性与违反率 shape 无关性
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24；golden torch 2.9.0+cpu
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新文件首条目（合入时创建 references/precision-patterns.md）「精度失败定征顺序」：
    ① dtype 隔离：哪个 dtype 挂 / 哪些全过（全 dtype 挂 → 结构/同步缺陷；单 dtype 挂 → 舍入路径统计性质）。
    ② 无关性检查：违反率与 shape/block_size/核数无关 → 纯舍入路径性质，排除 tiling/分核/同步排查方向。
    ③ golden opmath 域小实验：数行脚本核实 golden（torch CPU op）在目标 dtype 的实际计算域（fp16 常见 fp32 opmath + 单次舍回），再决定 kernel 原生域 vs fp32 中转（模式见 pattern-library §2 opmath 行 + VP-2026-0002）。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0007
- type: D
- title: T.Kernel(..., threads=) 在 npuir 无效果（KernelLaunch NPU 分支仅消费 grid_size）——设计侧约束表路由
- evidence:
  - src/ir.cc L175–203（NPU 分支无 block_size→threadIdx 消费路径）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 附#28
  - tilelang/language/kernel.py L218–223（Python 签名仍接受并透传）
- repro: grep -n -A 30 "is_npu_kernel_frame" src/ir.cc
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07，源码通读）
- target_doc: .agents/skills/tilelang-op-design/references/ascend-constraints.md
- delta: |
    add §1「本项目已知限制」表新行：
    | GPU `threads=` kwarg 无效果 | `T.Kernel(..., threads=)` 的 block_size 在 npuir KernelLaunch 不生成 threadIdx 绑定（Python 签名仍透传） | GPU 源码迁移残留 `threads=` 无声无效，threads/npt 语义需显式落到 block_size | 迁移时直接移除 `threads=`，折叠为单一 block_size 由 wrapper 参数传入（实证见 pattern-library §2 threads= 行；lerp_tensor 2026-09-07） |
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: 三件套齐备；因 target_doc 为 op-design references（Tier 1 写域）而入队——设计期（GPU→NPU 迁移）是该事实的主要消费场景，事实本体已 Tier 0 合入 pattern-library §2。

## VP-2026-0015
- type: P
- title: Expert 跨引擎 workspace 的跨任务槽位复用握手：per-task 槽位重启 + FLAG_TASKDONE 边界（避免 causal 变长 NK 的全局块前缀索引）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（flag 协议安全性传递性论证）+ #§7.2（四族 flag + FLAG_V 写侧 staging 补全）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：三族 + 单槽 + 跨任务五条论证独立推演全部成立）
  - 〔2026-09-16 第六轮限定，task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z〕TASKDONE 屏障在两相位 v3 结构经 WAR 全链审计判定**冗余**并删除（全域 −7~−21%）：ws_s/ws_p/ws_o 的跨 task WAR 由「Cube 单指令流程序顺序 + P-ready 消费链 + AIV 串行」闭合（attention.md PL-1.12）——握手模式是保守安全默认而非必要条件，**本条合入时须附该限定**（结构特定的程序序闭合可免除屏障；另见 VP-2026-0080 屏障冗余审计 BP 提案，两案互链）
- repro: 任一 per-task 变长内循环（causal NK 变长）的 Expert staggered 流设计
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（Expert 流水模式条目〔VP-2026-0013〕的握手子模式，合入时紧随其后）：
    模式：per-task 槽位重启（slot = i % slots 任务内重计数、跨任务从 0 重启）+ FLAG_TASKDONE 跨任务握手（Vector epilogue 完成后 set、Cube 下一任务首个 S 存储前 wait）——跨任务槽位复用安全，避免 causal 变长 NK 下的全局块前缀索引（数据依赖、不可静态化）。
    配套：flag 协议安全性按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手族单列；写侧 staging（FLAG_V 族，highperf V0S→C1L 同构）补全读侧无序窗口。
    实测：L2-gap100 四例 + kv32-guard（M7 单块退化共存位形）全过（DESIGN §8.2 / debug_log.md Final results）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0016

- type: P
- title: slots≥2 分块装载的尾块守卫从不变式推导（nk_total ≤ slots）而非 shape 枚举——枚举法漏浅块数域；判据同源反推测试域
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E7 + #§8.2（缺口域 4 例：S_kv=100 ∈ (bn,2bn)、ns=2，条件与测试同源）+ #§7.2（Vector 零填充预备段）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 3：v0 守卫仅覆盖 nk_total==1，slots=2/nk_total=2 时部分块为槽位首次使用——L2 套件 kv32/tail520 均不在缺口域，缺口靠检视推导而非测试暴露）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md（Final results：L2-gap100 四例 causal/non-causal × fp16/bf16 全过）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python examples/multi_head_attention/_gqa_prefill_fwd_kernel/_gqa_prefill_fwd_kernel.py --level all（L2 段 gap 四例即判据同源测试域）
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「多槽分块装载的尾块守卫不变式」：
    判据推导：部分块 p=nk_total−1 的残留安全性 ⟺「该槽位 p%slots 此前已被整块装载」⟺ nk_total > slots；唯一破坏点 = 核启动后槽位首用即部分块（残留为未初始化 L1/GM〔torch.empty〕，0×NaN=NaN 污染有效行）——正确守卫条件 = nk_total ≤ slots（含跨任务归纳：槽位尾行状态恒 ∈ {零, 有限真实数据}）。
    枚举法反例：从 shape 枚举（seq_len_kv < bn ⟺ nk_total==1）出发漏浅块数域（nk_total=2、slots=2）且测试域与条件不同源；判据同源反推测试域 = 按判据取边界值 S_kv ∈ (bn,2bn) + 必须触发 slots=2。
    配套：Vector 零填充预备段 + FLAG_V 握手（无它则 staging 写与 Cube 读无序）；守卫面随路线结构性简化（删载体族后 GM 未初始化域缺口自动消失——路线裁决的「正确性面」第四维）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0017
- type: P
- title: bf16 输出精度门禁设计须评估 golden 自身 P 量化噪声的放大位形——tier-2 绝对下限 max(5e-3, 2·flip_rel·vmax) + tier-3 窄域分类机制
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md#D6（L1-70b-long-bf16：1/16777216 元素超 tier-2 1.1%；fp64 逐元素验证 |kernel−true|=4.3e-4 vs |golden−true|=5.8e-3——golden 是偏离方；P 1-ulp 翻转经 causal 早行 p≈0.33×|v|≈2.06 放大 → 5.4e-3 > tier-2）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md（Stage 3 Precision-gate 行）
  - 交叉引用：queue VP-2026-0006（精度失败定征顺序——同文件合入时互链）、pattern-library §2 torch golden opmath 行（golden 侧域差异同类现象）
- repro: bf16 randn 域 attention：causal 早行（有效位少 × P 量化翻转 × 大 |v|）位形对照 tier-2 门限；元素级 fp64 对照定位偏离侧
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C；golden = torch SDPA NPU
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「bf16 输出门禁的 golden 噪声放大位形与 tier-3 分类」（合入时与 VP-2026-0006 定征顺序条目同文件、互相交叉引用）：
    机理：golden 自身 P 量化到输入 dtype 的 1-ulp 翻转，在 causal 早行（有效位置少、p 大、|v| 大）放大为输出偏差 5.4e-3 > tier-2（5e-3）——kernel 比 golden 更接近 fp64 真值（4.3e-4 vs 5.8e-3）仍可能超差；该机理是源算法 P-quantization-to-input-dtype 的固有性质，非 kernel 缺陷。
    门禁设计：tier-2 绝对下限应取 max(5e-3, 2·flip_rel·vmax)（bf16 randn 域 ≈ 4e-2）；未重校门禁前的过渡机制 = tier-3 窄域分类：发生率 ≤0.01% + 放大包络（2·flip_rel·|v|max）+ lse 门不受影响 + 逐例上报——真实缺陷（如丢 mask：79.8% mismatch）仍为响亮失败。
    定位手法：元素级 fp64 对照判定偏离侧（golden vs kernel 谁离真值远）再定机制，避免把 golden 噪声当 kernel 缺陷排查（亦见 pattern-library §2 零输入探针行的 fp64 对照法）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0018
- type: P
- title: AIV/lane 半分类设计的参数化标量阈值按最劣 lane 全局坐标独立重推 + 双 lane 端点数值例（单 AIV 思维自检必然逃逸）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1：v0 L220/L536 K_blk = base − row0，正确为 base + row0；smoke bx=4/块 4/vid=1 行 288 数值例复现——正确 j≤32 vs 错误 j≤−32）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3（v1 修复：行映射式 q_pos = s_lo + row0 + i 先行 + 阈值 PrimExpr 标量）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → DESIGN.md 修订链
- repro: 任一含 vid/row0 半分 + 块级 mask/阈值构造的 Expert 设计：对每个 per-lane 标量公式代入 vid=0 / vid=1 两个端点数值例
- toolchain_stamp: 设计/检视期证据（无运行时依赖）；审查环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（并入 VP-2026-0014 向量 mask 构造链小节的「分界/阈值推导模板」子条，合入时紧随其后）：
    修复模板：① 先写行映射式（lane 的全局坐标，如 q_pos = s_lo + row0 + i）再代源条件推导阈值（K_blk = base + row0）；② 阈值走 PrimExpr 标量而非动态 offset（T.arange.md offset 参数文档类型为 int，运行时 PrimExpr 支持无文档）；③ 共用分界（K_A 类）对非推导 lane 标注保守性（仅性能不损正确性——取整方向反例见 §4 expert attention 案例行：ceildiv 版对 vid=0 是激进上界、floordiv 版精确）。
    逃逸机理：以 vid=0（row0=0）代入的阈值在 vid=1 恰好相差 2·row0 且符号方向隐蔽（该 +row0 写成 −row0 时 vid=0 完全正确、vid=1 全错）——单 AIV 思维的自检必然逃逸，双端点代入是最低成本拦截。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0034
- type: P
- title: BP 跨引擎 flag 串行链：dual-Scope persistent 结构下块宽摊减优先于流水深度（bn↑ 同削 flag 往返/向量 op/gemm 碎片，ns/slots 在宽块上失效）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-1（busy 核分解：aic_scalar 41%/aiv_scalar 33% = flag 等待自旋，aic_cube 仅 9% 近峰值——ns=1→2 在 bn=64 上零变化）+ #Iteration-3（bn=512 单槽 -15~-16% vs G2@bn=256；bn=512 双槽 L1 溢出 invalid_compile）+ #Final-Summary（跨轮 busy 核画像主线：flag 等待 25–47% 恒存，块宽摊减是唯一有效方向）
  - pattern-library §1.9（数据侧：块宽摊减律 + cbuf=512KB 上限公式 + wide 单槽模式 + H=1 平衡公式；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10（2026-09-09 同谱系第二轮续调补充〔task 20260909T033622Z，按 VP-2026-0035 同 op 谱系先例不计独立确认〕：v10_dp bn=256 深流水全 case 回退 +11~12%——向量发射开销定律〔每 op ~0.5µs 固定成本〕给出块宽摊减律的机制解释；另实测第三硬上限 L0C=128KB）
  - 〔2026-09-17 适用边界数据点（不同结构类，不计确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕mamba 族 MixCV baseline 的串行形态为 **task 级 ping-pong 零重叠**（Vector(T) 产完 → Cube(T) 消费，非块内 flag 自旋）——该位形下**深度 2 任务流水是第一杠杆**（v1_pipe −39.6%，attention.md PL-1.12 update 消费侧前导 set 形态），块宽摊减居次（v6_bn128 −10.5%）：BP_cross_engine_serial_chain 的「块宽优先」结论适用于块内 flag 自旋类；task 级零重叠类先解除流水串行
- repro: 同结构 kernel（Expert dual-Scope persistent + per-slot flag）扫 bn 64→256→512 × ns∈{1,2}，fa2048/fa4096 non-causal fp16 对照 msprof op Task Duration
- toolchain_stamp: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_cross_engine_serial_chain（目录行置于 BP_pipeline_overlap 之后，正文紧随 BP_pipeline_overlap 节，交叉引用之）：
    触发信号：Expert dual-Scope persistent kernel（Cube/Vector + GM workspace 多槽 + per-slot flag 跨引擎流水）busy 核 aic_scalar 或 aiv_scalar 占比 30%+ 且归因为 flag 等待自旋；cube/mte 已近峰值而总时延远超计算分量；加深流水（slots/ns↑）零变化或被抵消。
    常见反证/不确定点：非跨引擎结构的 MTE/compute 串行归 BP_pipeline_overlap；bn 增大触发 L1(cbuf)=512KB / UB=192KB 上限、bm 增大触发 L0C(cc)=128KB（bm≤51@bn=512，2026-09-09 第二轮实测；上限公式见 pattern-library §1.9）——上限不否定方向，转 wide 单槽（L1 单槽 + GM ws/flag 双槽）继续换块宽。
    推荐动作：先块宽摊减（bn↑ 同时削减向量 op 数/块、flag 往返数/列、gemm 碎片化）再评估流水深度——宽块上 ns=1≈ns=2（双槽收益被单槽 L1 v-load↔gemm2 WAR 串行抵消）；配 bm 任务平衡（wall-rows = ceil(ceildiv(S,bm)/24)·bm）。机制基础（2026-09-09 第二轮实测）：向量算子按发射开销计费——每 op ~0.5µs 固定成本、f16≈f32 逐元素速率，同元素总量下窄块 op 翻倍使 vec-active 翻倍（v10_dp bn=256 实测 +12% 回退）。
    验证指标：Task Duration 下降 + busy 核 scalar（flag 自旋）占比下降 + causal/full 等非目标 dispatch 非回退（full 变体逐字节保留）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0038
- type: P
- title: 跨引擎/在线递推 kernel 精度失败的误差结构定征：全块均匀且可复现 → 排除竞态、指向跨迭代状态生命周期（rescale 因子覆盖）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10-OP2（v10_dp 首版精度 FAIL：全块均匀误差 → alpha/ellcur 跨 defer-2 生命周期被 V1(j) 覆盖；3 路 ping-pong〔平铺 buffer + 运行时三分支〕修复后 L0 全过）
  - 同文件#Skill-Retrospective-第二轮（「先看误差结构再定位」纪律：无它会把递推因子覆盖误判为竞态而放弃深流水方向）
  - pattern-library §1.9 sync_block_wait 行（修复形态交叉引用：v-op codegen 拒绝多维 UB 切片操作数，平铺多 buffer + 运行时三分支为合法形态）
- repro: 任一含 online 递推状态（running max/sum + rescale 因子）跨流水 deferral 生命周期的 Expert kernel：首版精度 FAIL 时先记录误差的空间分布（全块均匀 vs 部分核/非确定）再定排查方向
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「误差结构定征（均匀 vs 部分核）」（合入时与 VP-2026-0006 定征顺序、VP-2026-0017 golden 噪声放大位形同文件并互相交叉引用）：
    判据：精度 FAIL 的误差空间分布是竞态/同步缺陷与系统性状态缺陷的判别器——全块均匀且可复现（各块/各核一致）→ 排除竞态假设，指向跨迭代状态覆盖（online 递推的 rescale 因子〔running max/sum 的 α、ell〕在流水 deferral 生命周期内被后继迭代写入）；部分核/非确定分布 → 竞态/同步方向（flag 协议、握手窗口）。
    修复形态：递推状态跨 deferral 需按流水深度 ping-pong 多路；v-op codegen 拒绝多维 UB 切片操作数——平铺多 buffer + 运行时索引三分支是合法形态（pattern-library §1.9）。
    定位顺序与 §2 零输入探针行的元素级 fp64 对照法互补：先判分布（均匀/部分）定缺陷类别，再下钻元素级对照定偏离侧。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0043
- type: D
- title: layernorm 矩式方差（var=E[x²]−E[x]²）fp32 计算域失败定量：mean≈3000/σ≈0.5 时 fp64err 1.4e3（两遍式 6.3e-4，差 6 个量级）、违反率 25%，fp16 粗栅格 83%——「须评估 catastrophic cancellation」从定性升定量
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（moment_form + run_rejection_evidence 段——torch CPU 设计期脚本，无 NPU 依赖）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Value Point Proposals 首行）
- repro: repro-missing（知识域最小 repro 待同族任务回填；矩式 vs 两遍式差分最小化仅需数行 torch CPU 脚本）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（输出「C2 rejection evidence」段）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-layernorm R1 行（R1 已有指向本条的 pending 指针——VP-2026-0044 三件套合入时预写；本条闭环时把定量句写入 R1 并把指针转为条目引用）：
    R1 单遍式矩方差的 catastrophic cancellation 定量边界（ada_layer_norm 2026-09-10 verify_equiv 实测，torch CPU）：mean≈3000/σ≈0.5（pad 校正小 N 大均值位形）时矩式 fp64err 1.4e3 vs 两遍式 6.3e-4（差 6 个量级）、容差违反率 25%；fp16 粗栅格（6e4 行）违反率 83%——两遍式为默认候选。配套：GPU 源的 256-pad 方差校正在 NPU 精确 N 式下可整体舍弃且数值更优（恒等式机器验证 0 违反，pad 列贡献 m² 显式减去）。
- status: pending
- confirmations: 1/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: 因 target 为 Tier 1 写域（algorithm-candidates.md）+ 缺知识域 repro 而入队（VP-2026-0007 先例形态）；R1 行的 pending 指针已随 VP-2026-0044 合入预写。

## VP-2026-0052
- type: P
- title: 分层中间量导出定位法（系统性 vs 随机误差判别）：out_idx 多输出落 GM 对照 fp64 + violation 率 dtype 指纹（bf16 全过/fp16 ~6%/fp32 ~87% ⇒ 常数相对误差 ~1e-3 = 近似指令特征）单次运行定位到具体指令
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Value Point Proposals 第三行：探针 stage A/B 对照——S1/mean/d 精确 1.1e-7 而 rstd 偏 1.566e-3，误差独占定位在 vrsqrt 指令）
  - 同文件 Stage 3 首跑三 dtype violation 率分布（12 失败用例的 violations/N 统计）
  - pattern-library TRAP-vrsqrt-plain-precision（2026-09-10 Tier 0 合入；「定位手法」段即本方法的首次实证）
- repro: 复现条件——任一含统计链中间量（mean/var/rstd 等）的精度失败定位：中间量经 out_idx=[1,2] 多输出落 GM 对照 fp64 精确值逐级 dump；无固定脚本（方法形态）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md（合入时与 VP-2026-0006 定征顺序、VP-2026-0017 golden 噪声、VP-2026-0038 误差结构定征同文件互链）
- delta: |
    add 新条目「中间量分层导出定位」（合入时置于 VP-2026-0006 定征顺序条目之后）：
    手法：kernel 统计链中间量（S1/mean/d/S2/var/rstd）经 out_idx 多输出落 GM，对照 fp64 精确值逐级比较——单次运行把误差定位到具体指令段（ada 实证：mean/d 精确 1.1e-7 而 rstd 偏 1.57e-3 ⇒ 独占于 vrsqrt）。
    指纹判据（先于探针预判根因类别）：「violation 率 vs 容差」的 dtype 分布——bf16 全过 / fp16 ~6% / fp32 ~87% ⇒ 常数相对误差 ~1e-3 量级 = 近似指令特征（非数据依赖、非同步 bug）；与 §2 零输入探针行的元素级 fp64 对照法互补（先判分布定类别，再下钻元素级定偏离侧）。
- status: pending
- confirmations: 1/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: 原提案 target 为 tilelang-debug-helper SKILL.md（Tier 2 写域）；知识本体为精度定位手法（数据/模式类），改路由至 error-fixer precision-patterns.md（P 类 Tier 1，与 VP-2026-0006/0017/0038 同域互链）——保守序内降档处理，如需保持原 target 请在审批时说明。

## VP-2026-0057
- type: P
- title: 两相位结构的 causal 多头域适配机制集（persistent 变长任务循环 / 尾块钳位 + 装载不满宽掩码门控 / K_A 对角链承接 / flag 预算-bn 耦合守卫 / per-n-block flag 跨任务复用条件）——fa 域母本的 causal 化五机制
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md v3 §0.6 E1/E2/E3/E6/E7 + §5.2/§7.2（机制设计）+ §6.4（flag 协议传递性论证）
  - Stage 3 实证：同目录 RETROSPECTIVE.md Stage 3（2026-09-15）章节——`--level all` 29/29 全绿（L0 4 + L1 8 + L2 10 + Boundary 7；E7 缺口域四例 + M7 单块退化 + GQA 位形覆盖）
  - 母本对照：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py `_builder_2phase`（fa 域 B=1/H=1/整除/非 persistent 特化——五机制均为其不具备的 causal 多头增量）
  - 〔2026-09-16 第六轮机制集存活验证，task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z——同 op 谱系不计独立确认（VP-2026-0035 先例）〕跨工具链迁移 6797758→a13585dc 后 stale 重验（probe_ub.py 直跑 4 点复校，config 空间封闭性结论全部维持）+ 第六轮全部门禁绿；机制⑤ 预言的双槽 id-offset 变体（flag id = i + slot×nk_total，2×nk_total ≤ 15）由 r7f 实现并实测（短域 −7.4%）；机制④ E6 钳位域在 S_kv=4096 首次触发并经 r9 守卫收窄修复（联合 UB 核算判据）。**合入注意**：pipe2 slot-offset 与 E6 钳位分析已写入 attention.md PL-1.12，本条合入时以五机制清单为主体、机制③⑤细节指向 PL-1.12 防重复
- repro: repro-missing（机制效应只在完整 kernel 规模显现；ED-B delta 骨架待同族任务回填）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目 PL-1.9-twophase-causal（置于 PL-1.9-twophase 之后；attention.md 已近 16KB 预算，合入时触发 consolidate）：
    **两相位结构 causal 多头域适配五机制**（fa 域母本 → causal 多头域的必要增量，Stage 3 29/29 实证）：
    ① per-task 变长 NK 的 persistent 任务循环 + 轮转均衡（母本为非持久定长）；② 尾块分形下限钳位（M/N≥16、K≥32 的 tmc/tnc 公式）+ **尾带 OOB 掩码门控按「装载不满宽」口径**——Vector 装载宽是 tn_real 而非分形宽，统一门控 `has_kv_tail = (S_kv % bn_eff != 0)` + 运行时 `i >= K_A`（non-causal 时 K_A=kv_full；按「分形带存在」判定会漏 rem≥32 且 16|rem 的不满宽残留）；③ causal 对角 mask 链平移进 pass-1（K_A floordiv 两段式 + per-AIV K_blk = base + row0 阈值）；④ flag 预算-bn 耦合守卫：per-n-block 下标 flag 方案要求 NK_max + TASKDONE ≤ 16 → 工厂期强制 `bn_eff = max(bn_caller, ceil16(ceildiv(S_kv, 15)))`——守卫恒生效、设计默认仅替换 wrapper 默认值（config 契约分层：正确性守卫不可被用户 config 绕过）；⑤ per-n-block 同 id 三次握手的跨任务复用安全条件（单槽）：Cube 到达任务 t+1 的 set_S(i) 前程序序必经 pass-2(t) 的 wait_P(i)（同 id 前一生命周期消费点）——id 事件在任务边界前排空；双槽变体需 id-offset（2·NK_max+2 ≤ 16 才合规）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0058
- type: D
- title: T.ceildiv 负数域 lower 静默错误（divsi 截断 + 后续 pass 复活恒正改写）——persistent 空核越界任务 MTE 崩溃；绕法 = cid = T.min(task_id·N+kernel_id, num_logical-1) 幂等钳制
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Skill Flow Issues 首行 + Value Point Proposals 首行
  - 复现链：one-task (1,1,1,64,64) 崩溃（num_logical < 24 时空核执行越界任务 cid ≥ num_logical → s_lo ≥ S_q → MTE write-out-of-range）+ dump_ir2.log L130-134/L371-374（多 pass 快照对照：lower 成 `divsi(x,N)+1`，x∈(-N,0] 时返回 1 而非 0；恒正参数的 `floordiv(n-k+N-1,N)` 改写被后续 pass 重新规范化回错误形态——纯表达式改写不可靠）
  - 幂等性验证：two-batch (2,1,1,64,64) 钳制下 PASS（幽灵任务重算最后一个合法任务，同输入同指令序位级一致；各核独立 ws 槽 + 核内 flag 通道保证安全；饱和域 num_logical ≥ 24 零开销）
- repro: repro-missing（NPU 编译探针 + dump IR 对照待同族 persistent 任务回填；任务内复现条件 = 任一 num_logical < 24 的 persistent case 如 (1,1,1,64,64)）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-ceildiv-negative-lower：
    **`T.ceildiv` 对非正参数的 lower 是静默错误**：`T.ceildiv(n-k, N)` lower 成截断除法 `divsi(x,N)+1`，x∈(-N,0] 时返回 1 而非 0——persistent kernel 空核（kernel_id ≥ num_logical，仅 num_logical < 物理核数 24 时存在）执行越界任务直接 MTE write-out-of-range 崩溃；更隐蔽的是后续 BishengIR pass 会把恒正参数的 `floordiv(n-k+N-1, N)` 改写重新规范化回 divsi+1 错误形态（纯表达式改写不可靠）。**绕法 = cid 钳制幂等方案**：`cid = T.min(task_id*N + kernel_id, num_logical-1)` 把幽灵任务钉到最后一个合法任务——幂等安全（同输入同指令序位级一致重算 + 各核独立 ws 槽）、饱和域零开销、比 if 守卫（parser var-table 风险）与表达式改写（pass 复活）都稳。官方 persistent 模板（docs/开发指南.md §3.3）在 num_logical ≥ 核数域从未触发该分支。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0060
- type: D
- title: lse [B,H,S,1] 视图直写形态实测可用——`T.copy(ub[0:rm,0:1], lse[bz,by,s:s+rm,0])`（2D src → 1D dst 切片）bit-exact；[N,1] 源 MTE DDR fault 限定于特定 size= 组合而非该直写形态的普遍性质
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Skill Flow Issues 末行（probe_lse_store.py 三形态 A/B/C 全过 bit-exact + 全量 29/29 lse 门通过；E6 transpose-free 方案落地无障碍）
  - 被澄清条目：pattern-library traps-runtime.md TRAP-T-copy-region-semantics 规则③（"[N,1] UB 源 + size=[1,N] 按步长 N 越界读——lse 需先 T.transpose 到 [1,N]"，源自 Sep-07 任务 E1-E7 impl note 2）；traps-compiler.md TRAP-transpose-epilogue-poison 的绕法（[B,H,S,1] 视图 + 2D 区域拷贝）即本形态，第三轮已生产使用
- repro: repro-missing（NPU 探针；任务内复现条件 = probe_lse_store.py 三形态对照——session-local provenance，关键发现镜像于 RETROSPECTIVE Stage 3）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    update 条目 TRAP-T-copy-region-semantics 规则③（边界澄清，原规则保留）：
    ③ [N,1] UB 源 + **size=[1,N]** 组合按步长越界读（MTE DDR fault）——fault 限定于该 size= 组合；**2D src → 1D dst 切片直写 `T.copy(ub[0:rm, 0:1], gm[b, h, s:s+rm, 0])` 实测 bit-exact 可用**（2026-09-15 探针三形态全过 + 29 用例 lse 门；Sep-07 记载的「[N,1] UB 源 MTE DDR fault」不构成对该直写形态的否定）——行向量写 GM 尾维连续区首选 [B,H,S,1] 增维视图 + 2D 区域直写（transpose-free lse，与毒化条目绕法一致），T.transpose 到 [1,N] 仅在必须 base+size 形态时需要。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0061
- type: P
- title: 错误指纹速查——lse 恒等于 log2(S_kv) ⇒ S 通道读到未初始化零页（伴生指纹：前几行 NaN = 越界更远脏页），秒级定位上游 workspace 生产者/槽位错误
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 2 行 + Skill Flow Issues 第 2 行（two-task (1,1,1,128,128)：lse=7.0=log2(128) 恒值直接锁定 ws_s 通道；bx=1 任务全错〔lse 恒值 + 4 行 NaN〕而 bx=0 完美 = per-core 槽 row0 索引 bug——全局/本地行基巧合掩盖）
- repro: 复现条件——two-task (1,1,1,128,128) 类位形（S 通道读到 torch.empty 零页时指纹必现）；无固定脚本（指纹判据形态）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 诊断指纹条目（置于 TRAP-zero-input-crash 之后，同「部分行数据错」诊断族）：
    **lse 恒值 log2(S_kv) 指纹**：lse = log2(ell) + m·LOG2E——输入全零分数（S 通道读到未初始化零页 workspace）时 ell=S_kv、m=0 → lse 精确恒值 log2(S_kv)；「前几行 NaN = 越界更远读到脏页」为伴生指纹。判据指向「上游生产者未写 / 读错槽位」（workspace 槽位索引、flag 时序），先查 Cube 是否写/Vector 是否读对槽位再查计算链，无需逐指令分析（2026-09-15 per-core ws 槽全局行基误用案例：bx=0 任务「全局=本地」巧合掩盖、bx≥1 全错，指纹秒级锁定）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0062
- type: P
- title: MTE 越界类三层崩溃定位法（崩溃边界矩阵 → 错误值指纹 → TILELANG_DUMP_IR + 新 TILELANG_CACHE_DIR）——tilelang 缓存命中跳过编译也跳过 IR 打印，dump 必须换 cache 目录
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 3 行 + Transferable Lessons（one-task 崩溃 40 分钟定位：矩阵 5 分钟锁定「nl<24 崩 / bx≥1 错 / NK 无关」→ IR divsi 对照 → pass 复活确认；probe_nd2nz_offset.py 排除法）
  - 工具行为实证：TILELANG_CACHE_DIR 命中缓存时跳过编译也跳过 IR 打印——TILELANG_DUMP_IR=TRUE 对已缓存 kernel 无输出
- repro: 复现条件——任一 MTE 越界崩溃的 persistent kernel（任务数 × NK × bx 三维隔离矩阵 + dump IR 对照）；cache 行为 = 对已缓存 kernel 设 TILELANG_DUMP_IR=TRUE 观察无输出、换新 TILELANG_CACHE_DIR 后有输出
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-debug-helper/references/mlir-dump-guide.md
- delta: |
    add 增补两节（置于环境变量段之后）：
    **① dump IR 必须配新 TILELANG_CACHE_DIR**：tilelang 缓存命中时跳过编译也跳过 IR 打印——`TILELANG_DUMP_IR=TRUE` 对已缓存 kernel 无输出，须指向新 cache 目录强制重编译。
    **② 崩溃边界矩阵法（MTE 越界类三层定位）**：一层 崩溃边界矩阵（任务数 × NK × bx 三维隔离小用例各跑一次，快速锁定崩溃维度——如「nl<24 崩 / bx≥1 错 / NK 无关」）→ 二层 错误值指纹（恒值/NaN/部分行分布——如 lse 恒 log2(S_kv) ⇒ S 通道零页，见 traps-runtime 指纹条目）→ 三层 TILELANG_DUMP_IR（新 cache 目录）看 TIR/MLIR 双层指令形态（lower 结果 vs pass 后形态对照，确认被哪层改写）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0080
- type: P
- title: BP_sync_barrier_redundancy：persistent/两相位结构的同步屏障冗余审计前置（列出每个 barrier 保护对象集，逐对象找更细粒度已有信号覆盖）——TASKDONE 全 drain 屏障删除实测全域 −7~−21%，纯程序顺序论证零实验成本
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4）：opt_log Phase 1 诊断归因（单槽 ws + TASKDONE 全 drain 屏障使每 task 边界完全串行；WAR 全链审计结论：屏障在数据竞争角度冗余——ws_s/ws_p/ws_o 的所有跨 task WAR 由「Cube 单指令流程序顺序 + P-ready 消费链 + AIV 串行」闭合）+ Round 6 r6a（删除后全域 −7~−21%：8b-long 1189.18→1021.58 / 8b-short 397.79→315.58 / smoke 49.97→46.26µs；全引擎利用率等比提升 cube 20.4→23.7% / vec 49.9→57.4% / mte2 52.1→62.1% 而 GM 流量不变——节约纯等待）
  - 知识域锚点：attention.md PL-1.12「TASKDONE 屏障冗余审计」段（方法 + 逐对象论证 + 数字自包含）
  - 交叉引用：queue VP-2026-0015（FLAG_TASKDONE 跨任务握手模式——保守安全默认，本条是其「可审计免除」限定，两案合入时互链）；VP-2026-0034（BP_cross_engine_serial_chain——块宽/深度之外的第三杠杆）
- repro: repro-missing（delta 最小形态 = 删除 task 边界的 TASKDONE set/wait 同步对〔2 个 sync op〕，效应依赖完整 kernel 的跨 task 指令流衔接；机制归因与逐对象 WAR 论证自包含于 PL-1.12——知识域骨架待同族 persistent 任务回填）
- toolchain_stamp: tilelang 0.1.2+a13585dc / CANN 8.5.0 / Ascend910B2C / 2026-09-16
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_sync_barrier_redundancy（目录行 + 正文，追加于 BP_run_state_bimodality 之后）：
    触发信号：persistent / 两相位（跨引擎 task 循环 + GM workspace）结构存在 task 边界全 drain 屏障（TASKDONE 类：下一 task 的生产等待上一 task 全部消费完成）；时长因子分解显示 per-task 固定成本 F 主导（引擎利用率低而 GM 流量已定、busy 核 scalar 占比高但非 flag 自旋类）。
    审计方法（零实验成本）：列出每个 barrier 保护的对象集（ws buffer × 读写者对），逐对象找更细粒度的已有信号覆盖——同引擎单指令流程序顺序（天然保序）、消费链 flag（P-ready/S-ready 类既有握手）、AIV 串行流。全部对象被覆盖 ⇒ 屏障在数据竞争角度冗余、可删。
    实测（attention 第六轮 r6a，a13585dc）：TASKDONE 删除全域 −7~−21%，引擎利用率等比提升而 GM 流量不变（归因成立的判据：节约的是纯等待）。
    边界：删除前提是 WAR 链逐对象闭合论证（数据竞争角度），非盲目删同步；跨引擎同 buffer WAR 无同步原语的形态不可删（CG-2026-0009 / PL-1.12「Vec 侧同构改造 blocked」教训——引擎独占 buffer 是流水深度 >1 的隐含前提）。
    验证：全域 dispatch 非回退（--level all 全绿）+ 利用率等比提升 + GM 流量不变。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0083
- type: D
- title: SSD/衰减类因子的 cast 侧选择——因子乘在「值随因子缩小的操作数」一侧再 cast 噪声随真值缩小；移到对侧（K 侧因子移动）fp16 绝对噪声放大 38–135×
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py（OPT-3 否决行 + 回归 case——K 侧因子化 cast(x·exp·dt) + cb 直读 gemm 的 max 2.9e-3 > atol 1e-3，噪声放大 38–135×；cb 侧因子化 cast(cb·exp·exp) 噪声随 exp 缩小）+ DESIGN.md §1.6.1 否决表
  - 〔2026-09-20 跨工具链强化证据（同 op 谱系重跑，不计独立确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z，tilelang 0.1.2+4515de8 / torch CPU〕verify_equiv.py OPT-X 回归行：fp16 下 exp(anchor−dA_s) 正大指数 → inf → cast inf → ×exp(负) = **直接 NaN**（此前记录为 38–135× 噪声放大），bf16 1e26 量级（3.4e+22/2.9e+24 实测行）；OPT-Y（对角 L-side 因子化）NaN×2 同步机器复证——否决项保留 FAIL 回归记录的设计使跨任务复证零成本（RE-1 建议）
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现命令见 evidence provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py（含否决项回归 case）
- toolchain_stamp: torch CPU（2.x，脚本自带）；设计工具链 tilelang 0.1.2+1990aa9fe4 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（SSD/mamba 族新增条目）
- delta: |
    add SSD/mamba 族条目注意事项：衰减/门控类因子（exp(−Δ)、softmax 权重）做 dtype cast 时，因子乘在**值随因子缩小的操作数**上再 cast，绝对量化噪声随真值同步缩小；把因子移到对侧（值不缩小的操作数）会放大绝对噪声并可能超出绝对 atol 门禁——纸面「乘法交换律恒等」≠「容差内等价」，须机器验证（verify_equiv 首轮否决即此形态，返工集中在叙述层）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0084
- type: P
- title: L-side anchor 因子化（exp_l·exp_as）存在 fp32 溢出边界——对角块内 s ≥ l0 ⟹ exp_as ≥ 1，块内衰减 > ~88 溢出污染下三角；直接差分 + 掩码前置是无条件安全 fallback
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：DESIGN.md §1.6.1 否决行 #5 / §1.6.3 候选 #9；源码 ssd_chunk_scan.py L350–371 微块 M32 锚点选择注释（anchor 机制正是为此）
- repro: repro-missing（分析型——边界可由 verify_equiv.py deep_decay 案例扩展复现）
- toolchain_stamp: 分析基于 fp32 exp 上限 ~e^88.7（无工具链依赖）；核对环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（SSD 族注意事项，与 VP-2026-0083 同条目或紧邻）
- delta: |
    add SSD 族数值安全注意：L-side anchor 因子化的安全性依赖「两个因子指数均 ≤ 0」的单调论证；对角块（因子定义域重叠，s ≥ l0 ⟹ exp_as ≥ 1）内该论证失效——块内衰减 > ~88（fp32 exp 上限 e^88.7）溢出污染下三角（真实分布 24σ 罕见但非零）。直接差分形式下三角指数恒 ≤ 0、上三角被掩码，无条件安全；源实现的微块 M32 锚点选择即该边界的工程规避。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0085
- type: D
- title: 累加 gemm 的 A 操作数必须覆盖完整 K 维（N 维）——n-loop 复用单块 [bl,bn] 在 N > bn 时静默算错（max_diff 0.40 符号翻转），小 N 用例全绿是假象
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：Stage 3 修复链——history 路径 c_scaled = C·exp(dA_l) 按 [bl,bn] 单块 + n-loop 复用，N=128/bn=64 时 n-block 1 读到第 0 块（静默算错，仅 L1-256 暴露）；修复 = c_scaled 全维 [bl,N] + Cube n-loop 逐块 T.copy（修复前 L1-256-full max_diff 3.9e-1 → 修复后 1.4e-4）
- repro: repro-missing（NPU kernel 级；任务内复现 = python3 _ssd_chunk_scan_fwd_kernel.py --level all 观察 L1-256-full FAIL→PASS——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 语义陷阱条目（置于 TRAP-L1-band-dst-tail-overrun 之后）：**累加 gemm 的 A 操作数 K 维覆盖**——history/state 类 Σ_n A[:,n]·B[:,n] 的 n-loop 每块读的是**不同的 A 列段**，只算一块 [bl,bn] 复用会静默算错（N > bn 时才暴露；N_tiles=1 的 L0 用例全绿是假象——验证用例须含 N 维 > block_n 的 case）。修复形态：A 侧因子按 [bl,N] 全维物化，Cube n-loop 逐块 copy 消费。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0086
- type: D
- title: Expert 模式 T.copy 通用路径不支持跨 dtype（编译报错 "does not support element type casting"）——与 TRAP-C12 的 Developer 静默 VCast 互补；三条落地口径
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：Stage 3 三处跨 dtype copy 报错修复链（prev_states(fp32)→state_t(dtype)、cb/C/dt(dtype)→f32 因子）——显式 dtype 中转 buffer + T.vcast(..., round_mode="rint")（f16/bf16→f32 仅 rint）+ gemm B 的 fp32→dtype cast 放 host wrapped() 内 .to(dtype)（对齐源码 T.cast 语义，golden 仍 fp32）
  - docs/Tilelang.language/内存操作/T.copy.md §2.3 条 2（通用路径 dtype 一致要求）
  - 被补充条目：traps-compiler.md TRAP-C12-copy-dtype-cast（Developer GM→UB 隐藏 VCastOp 静默转换——同一 API 两模式行为相反）
- repro: repro-missing（NPU 编译探针；任务内复现 = grep -n "vcast\|\.to(torch_dtype)" 于任务 kernel——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update 条目 TRAP-C12-copy-dtype-cast（追加 Expert 模式互补段）：Developer GM→UB copy 静默插 VCastOp（原文保留）；**Expert 通用路径要求 src/dst dtype 一致**——跨 dtype copy 直接编译报错（"T.copy does not support element type casting"），须显式中转：① 同 dtype copy + T.vcast(..., round_mode="rint")（f16/bf16→f32 仅 rint）；② gemm B 操作数的 fp32→dtype cast 可在 host .to(dtype) 完成；③ 5D 输入 Vector scope GM→UB 动态 extent copy 触发 codegen "cannot find variable" 时关 TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION（PL-1.16 硬边界）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0087
- type: D
- title: 向量二元 op 广播形态速查——vmul/vsub/vadd 文档化仅四形（无 [M,N]*[1,N]）、M==N 时方向写反静默算错、vmin 族内原生支持一般列广播（规则不族内统一）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：Stage 2 REVIEW.md 问题 1（v0 列因子 [bs,1] 错向——bl==bs=64 时编译不报错静默算错，本任务设计修订 3 阻塞之首）+ re_review Value Point Proposals（T.vmin.md §2.2.2 明示 src1 (1,N,K)/src2 (M,N,K)→(M,N,K) 一般列广播合法，vs T.vmul/T.vsub/T.vadd §2.2.2 仅四形：[M,N]*[M,N] / [M,N]*[M,1] 行因子 / [1,N]*[1,1] 退化 / [M,N]*scalar——无 [M,N]*[1,N]）
  - docs/Tilelang.language/数学操作/T.vmul.md#§2.2.2、T.vsub.md、T.vadd.md；比较操作/T.vmin.md#§2.2.2；shape操作/T.vbrc.md#§2.2.2（列因子合法路径：vbrc (1,N,K)→(M,N,K) 展开后全形运算）
  - 数值演示（torch）：cb*f[64:128].unsqueeze(-1) vs unsqueeze(0) 两形不等、差值 O(数据)（repro 命令可复现）
  - 反例先例：GQA _gqa_prefill_fwd_kernel.py L401/L405/L645（真行因子 [half,1]）与 L430（j 轴依赖改 T.arange 全形物化）；全仓 9 处 T.vmul 无列因子广播用例
  - 关联已合入：layout.md PL-1.15-broadcast-perf-tax（性能税半边——本条补形态合法性半边，两半互补成完整速查）
- repro: repro-missing（文档口径核对 + torch 数值演示；知识域最小脚本待同族任务回填）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / Ascend910B2C / npu-smi 26.0.rc1（文档口径，未上机）；数值演示 torch 2.7.1+cpu / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    add「向量二元 op 广播形态速查」条目（与 PL-1.15 互补，合入时互链）：**行因子 [n,1] 沿尾轴广播（随 i 变化）；随列 j 变化的因子必须 [1,n]——而 [M,N]*[1,N] 不在 vmul/vsub/vadd 文档化 Shape 清单内**（仅四形），列因子须 T.vbrc 展开为 [M,N] 全形再运算（vbrc.md §2.2.2 合法形 src (1,N,K)→dst (M,N,K)）；**M==N（如 bl==bs）时方向写反不报 shape 错、静默算错**。族内不统一：T.vmin §2.2.2 原生支持 (1,N,K)→(M,N,K) 一般列广播——逐 op 读 §2.2.2 是唯一可靠做法，禁止按族类推。配套已核实（供 SSD/causal 族直接引用）：T.vmin/T.vadd 均导出（tilelang/language/__init__.py L73/L79）、fp32 √ / bf16 ×、标量形有生产先例（engram_fwd.py L76）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0088
- type: P
- title: BP_aiv_duplication_check——MixCV（Mix Block Dim = 2× Block Dim）画像两 AIV 子块指标相同 ⟹ Vector 程序重复执行 ⟹ subid 分片为第一候选（本轮 −34% 单点）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：opt_log §9 流程观察 1（发现花了 4 轮——前 3 轮把 vector0/vector1 相同指标读作"分摊或镜像"，R4 才用「若分摊则 ratio 应减半」反证确认）+ R4/R5（v8_aivsplit w2 −34.4%；数据侧已回写 attention.md PL-1.13）
- repro: repro/PL-1.13-aiv-dup-subid-split.py（知识域，2026-09-17 转正——分片骨架 + 机制归因 + 判据）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_aiv_duplication_check（目录行置于 BP_compute_granularity 之后）：触发信号：MixCV kernel（Mix Block Dim = 2× Block Dim）的 msprof PipeUtilization 中两 AIV 子块 vec/mte ratio **完全相同**而非各半（相同时每 AIV 承担 100% 向量工作——AIV 产能 2× 浪费）。反证：ratio 各半 ⟹ 已分摊（GQA v11 类先例已分片）。动作：subid 边界表达式分片（蛇形均衡式见 PL-1.13；禁 if 守卫——TRAP-tvm-parser-rules）为第一候选；分片维度须与 ws 写区域正交。验证：per-AIV vec/mte2/mte3 时间减半 + Task Duration 下降（ssd 实测 −28~−36%）。机制与完整形态：attention.md PL-1.13（互链，不复制）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0095
- type: D
- title: L1→L1（cbuf→cbuf）`T.copy` 硬编译失败（Expert 最小探针实证）——L1 域内重排/组装须单 buffer GM→L1 列偏移直载（dst 侧尾块裁剪联防）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 复检轮：REVIEW.md（复检轮）机械复核段编译探针表 + 附录（探针 1/2 关键代码与报错全文——第三轮 REVIEW 覆盖后残留描述见该任务 RETROSPECTIVE.md Stage 2 revision 章节）；DESIGN.md v1→v2 §1.4 band 组装修订（L347–353 两跳 → 单跳）
  - docs/Tilelang.language/内存操作/T.copy.md §2.3 搬运方向表无 L1→L1 方向（GM→UB/UB→GM/UB→UB/GM→L1/L0C→GM）；全仓 examples 零先例
  - 〔provenance，允许失效〕探针重建最小形态：`T.Kernel(1, is_npu=True)` + `T.Scope("Cube")` + 两个 `T.alloc_L1([64,64],"float16")` + `T.copy(l1_a, l1_b)` → 预期 bishengir-compile err code 1 报 'Unsupported copy from cbuf to cbuf!'；探针 2（GM→L1 列偏移直载 + 单条 band gemm）预期编译通过
  - 关联：capability-gaps CG-2026-0002（cbuf→cbuf 不支持，2026-09-20 occurrences 2 升级 recurring——vbrc lowering + Expert T.copy 双形态）；traps-runtime.md TRAP-L1-band-dst-tail-overrun（dst 侧 ts 裁剪联防）；traps-compiler.md TRAP-C12（dtype 面，互补）
- repro: repro-missing（知识域最小 repro 待同族任务回填——探针原件 transient 丢失；重建最小形态与预期报错见 evidence provenance 项）
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 新条目 TRAP-L1-L1-copy-unsupported（置于 TRAP-C12 之后）：**`T.copy` L1→L1（cbuf→cbuf）方向硬编译失败**：`'hivm.hir.copy' op Unsupported copy from cbuf to cbuf!`（bishengir-compile err code 1）——docs 方向表无此方向、全仓 examples 零先例、最小 Expert 探针实证（三证合一，2026-09-20，4515de8）。**绕法**：L1 域内「重排/组装」意图落为单 buffer GM→L1 列偏移 dst 直载（`T.copy(ws[s_blk,0:bl,0:ts], l1_band[0:bl,s0:s0+ts])`，src/dst 双侧尾块裁剪 ts=T.min(bs,Q−s0)——与 TRAP-L1-band-dst-tail-overrun 联防）+ 单条 band gemm `size=[tmc,l0+tl,tnp]`（探针 2 编译通过），或经 GM 中转；设计期引入新搬运方向先过三关（方向表在列 + examples 先例 + 最小编译探针）。与 TRAP-C12 构成 T.copy 方向/dtype 双约束；缺口本体 CG-2026-0002（recurring）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 设计期「写实占位伪代码」自创两跳结构引入未证实原语、复检探针证伪（修订写实对照纪律提案见 VP-2026-0104）；报错文本与 CG-2026-0002 首证（Developer vbrc lowering，2026-09-04）同源。

## VP-2026-0096
- type: D
- title: 深度 2 任务流水 slot-free wait 的正确位置 = Vector 任务头（先于任何 ws[slot] 写）——尾置形态在「prologue 前导 set + 事件计数」协议下产生 off-by-one WAR 竞争
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 revision：REVIEW.md（首轮）问题 1 令牌推演（CONS0 set 序 {prologue, C 完成任务 0, 2, …} × 尾置 wait 序 {T=0 尾, 2 尾, …} ⟹ T=2 写 ws[0] 只受 prologue 放行、不受 C 消费任务 0 约束）；DESIGN.md v1 §7.2 行 #4 头置形态
  - 对齐基准：`git show 4515de8:examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py` L356–362（旧 verified 终版注释 "from task 2 on it waits for Cube's consumption of task (T-2)"）
  - 4515de8 复刻头置形态首编即过 + L0–Boundary 全绿（同任务 Stage 3）
- repro: repro-missing（令牌推演为纸面证明——set 序×wait 序×受保护对象写序位置逐对配对，无需运行即可复核；verified 形态经 git show 对照）
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update 条目 PL-1.12-task-pipeline-depth2 的「消费侧前导 set」update bullet，追加：**wait 置位位置是深度 N 流水的正确性字段（非实现细节）**——slot-free wait 必须 Vector 任务头前置（先于任何 ws[slot] 写）；尾置形态在「prologue 前导 set + 事件计数」协议下产生 off-by-one WAR 竞争（写只被上上次同槽令牌放行，稳态下快引擎覆写慢引擎正在读的槽位）；前置形态下任务 W 写同槽受 C 完成任务 W−2 约束（V 领先 ≤1 任务）。后续算子复刻该结构时 wait 位置须与条目形态逐行比对（2026-09-20 重跑任务 v0 曾以尾置形态逃逸设计自检、检视期令牌推演才检出）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 检视侧令牌推演检查项已由 VP-2026-0029 承载（本任务为其第二任务独立证据）；本条为数据侧位置规则条目化。

## VP-2026-0097
- type: D
- title: band 域 stale 列掩码双判据必要性机器证明——stale 列 NaN 可跨下三角（i ≥ j ≥ ts），加法掩码污染下三角、单一因果 vselect 亦泄漏，`(j≤i)&(j<ts)` 组合才干净
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1：verify_equiv.py stale-NaN 专项案（输出 add_polluted=True / causal_leak=True / full_clean=True）+ DESIGN.md §1.6.1 OPT-B2 行（band-carrying 域形态）
- repro: repro-missing（verify_equiv.py 为任务过程文件；stale-NaN 案三形态对照可由数行 torch 脚本复现——下次同族任务顺手转正）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update 条目 PL-1.11-causal-mask-scalartrap 的「NaN 免疫边界」bullet：band-carrying 域的 vselect 守卫须**双判据** `(j≤i)&(j<ts)`（列 OOB 守卫与因果守卫缺一不可）——机器证明（2026-09-20 verify_equiv stale-NaN 案）：stale 列 NaN 可跨越下三角（i ≥ j ≥ ts 位），① 加法掩码 NaN+x=NaN 污染下三角；② 单一因果 vselect（j≤i）不足——stale 列与下三角交集的 NaN 被选路保留；③ 组合判据干净。与既有「band-carrying 保留 vselect 选择语义」合并为完整判据。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 与 TRAP-L1-band-dst-tail-overrun（dst 侧裁剪）互为 Vector 侧掩码对偶——合入时互链。

## VP-2026-0098
- type: D
- title: Cube gemm 分形 K 维下限「K ≥ 32」为保守口径——K=16（size=[tmc,16,tnp]）在 4515de8 编译且数值正确（max_diff 2.75e-5，静默无 stale 带污染）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 3：Boundary B-n16 用例 PASS（history gemm N=16 作 K 维）；DESIGN.md §5.3 注意 3 / §9.1（原断言「N=32 贴分形下限」为安全最小，未覆盖 N=16）
  - 被细化口径：.agents/skills/tilelang-op-design/references/decision-tree.md L74–75（L0A: M ≥ 16, K ≥ 32 / L0B: K ≥ 32, N ≥ 16）
- repro: repro-missing（端到端 kernel 级——B-n16 用例；K<16 与 M/N 下界未测，结论边界开放）
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/references/decision-tree.md
- delta: |
    update 分形限制条目（L74–75 邻域）追加实测注记：K=16 在 4515de8 静默有效（编译 + 数值正确，2026-09-20 ssd 重跑任务 B-n16 实证）——「K ≥ 32」按保守口径使用（设计仍可取 32 为安全最小），但不再作为硬下限否定 K∈[16,32) 形态；K<16 未测。合入时在 queue VP-2026-0057 机制②（尾块钳位 K≥32 公式）加同口径注。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 单点边界实测，泛化边界（更小 K / M/N）开放——条目按「保守口径 + 实测下界」双标注，防过度泛化。

## VP-2026-0099
- type: P
- title: Expert MixCV kernel 工厂闭包自包含约定——返回闭包内自分配 workspace + 输入 dtype cast，对 wrapper 暴露与 GPU 源一致的张量签名 → 集成零胶水（attempts=0 首过）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 5：examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/_ssd_chunk_scan_fwd_kernel.py L461–484（`_inner`/`wrapped` 内部分配 ws_c/ws_lcb + prev_states fp32→dtype cast）；integration_log.md（attempts=0——首次 import 冒烟 + 4/4 正确性 + 11/11 benchmark 一次通过）
- repro: 复现条件——任一 Expert kernel 工厂把 workspace 分配与输入 cast 封装进返回闭包（对外签名与 GPU 源一致）后走 integrate_kernel.py 全流程
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir（4515de8 谱系）; Ascend910B2C; CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md（Expert 族结构约定 bullet，合入时与预算 consolidate 协调）
- delta: |
    add bullet（PL-1.16 结构形态段邻位）：**工厂闭包自包含约定**：kernel 工厂返回的闭包内部完成 workspace 分配（ws_c/ws_lcb 等）与输入 dtype cast（如 prev_states fp32→dtype），对 wrapper/测试/benchmark 暴露与 GPU 源算子一致的张量签名——wrapper 零胶水改动即可切换集成（2026-09-20 ssd 重跑任务 Stage 5 attempts=0 实证：smoke + 4/4 正确性 + 11/11 bench 一次通过）；workspace 需求不构成改 wrapper 的理由（与 PL-1.16 wrapper 契约兼容条呼应）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 与 VP-2026-0093（config 口径衔接）/ VP-2026-0074（config-契约范式）构成 Stage 5 集成约定三件——合入时互链。

## VP-2026-0100
- type: D
- title: TileOPs report 新集成算子首跑的 N 条 "missing tileops candidate record" 告警 ≠ benchmark 失败（N=workload 数）——有效性看 run.json summary 字段，不数 warnings
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 5：examples/TileOPs/reports/tileops/20260920_144100_174551_SSDChunkScanFwdOp/run.json（summary：benchmark_passed=true, msprof_case_count=11, profiler_fallback_count=0, invalid_profile_count=0；warnings：11 条 candidate record 告警）+ report.md#Warnings；integration_log.md 备注
  - 〔provenance，允许失效〕判定命令：TileOPs 根目录以 reporting CLI 对任一新集成 Op 跑 msprof 采数后查 run.json 的 warnings 与 summary 两段
- repro: repro-missing（需 TileOPs + 新集成 Op 环境；判定口径自包含于 delta——run.json summary 三字段，判定命令见 evidence provenance 项）
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir; Ascend910B2C; CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 小条目（TRAP-BENCH-CONFIG-CALIBRATION 之后，采数判读族）：新集成算子首跑 report 的 "missing tileops candidate record" 告警（每 workload 一条）是候选基线记录缺失的提示性告警，不影响 status=passed 与 msprof 数值有效性（fallback=0 / invalid_profile=0）——判定 benchmark 有效性看 run.json summary 字段（benchmark_passed / msprof_case_count / invalid_profile_count），不数 warnings（2026-09-20 ssd 重跑任务 11 条告警 + 11/11 全 msprof 实证）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 消费场景在 Stage 5 integrator 侧——若审批倾向改 integrator known-fixes（Tier 2 域）而非 traps-runtime 条目，可在审批时改路由（事实本体不变）。

## VP-2026-0110
- type: P
- title: BP-engine-criticality-drift——persistent 双引擎 kernel 的引擎临界性随对侧优化漂移，「引擎 X 非关键」结论每轮 winner 合并后重算（任务周期 vs 引擎 busy 逐 workload 核算）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log.md#round-4 现象节（逐引擎任务周期核算：w2 任务周期 6.58µs vs Cube mte2 busy 4.45µs/任务 ⟹ w2 已 AIV-bound——旧档 PL-1.18 的「AIV vec 62% 非关键」在 prevhoist+l0c2x 两轮 Cube 优化后不再成立，vbrchoist 因此获益 −3.5%）
  - 方法有效性的自然实验：同文件 R4（vbrchoist 获益分布 w2 −3.45% / w4 −3.30% 响应，w3 −0.27% 不响应——AIV-bound/AIV 共临界/Cube-bound 三态与逐 workload 约束判定逐一互证）
  - 〔2026-09-21 重建会话方向翻转级证据（同 op 谱系不计确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z〕vbrchoist 单点在无 prevhoist 的 base 上 H=80 族回退 −4.2~−6.7%（双引擎 mte2 +22~28%）、叠加 prevhoist 后全转正 +2.2~+10.2%——不只收益幅度，效应方向本身随对侧累积序翻转（本条「临界性漂移」的更强形态，opt_log R1 机制分析 + R2 复测）；姊妹 BP 提案 VP-2026-0115（累积序检验步骤），两案合入时互链
  - 数据侧落点：attention.md PL-1.18 update③ 获益分布注（2026-09-21 已回写）；流程侧姊妹提案 VP-2026-0112（iteration-diagnosis.md 已知迭代结论补注——方法/流程两 Tier 拆分，合入时互链）
- repro: none（方法类：判据与核算式自包含于 delta；效应载体为 attention.md PL-1.18 + repro/PL-1.18-floor2-wins.py 骨架）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_engine_criticality_drift（目录行置于 BP_run_state_bimodality 之后）：
    触发信号：Expert/persistent 双引擎（Cube+Vector）kernel 的 profile 或前轮结论含「引擎 X 非关键路径 / 非瓶颈」定性；或本轮 winner 改变了某一引擎的负载/串行结构（字节削减、链解串行、op 削减类合并）。
    核算方法（每轮 winner 合并后执行）：逐 workload 比较「任务周期（Task Duration ÷ 每核任务数）」与「各引擎 per-task busy 时间」——引擎 busy 之和逼近或超过任务周期 ⟺ 该 workload 转为该引擎 bound；临界性是随结构状态漂移的动态量，不是一次性判定（ssd 实证：AIV「非关键」结论在两轮 Cube 优化后失效，w2 转 AIV-bound 使 Vector 侧 hoist 重新有收益）。
    推荐动作：临界性反转的 workload 上重启「此前因非关键被否决」的对侧候选（旧否决注须核对其归因是否含形态税等可修正因素——ssd v11 vbrc hoist 失败实为 vsub dst=src2 alias 税，干净形态胜出）；Cube-bound workload 对 Vector 侧优化不响应是判定正确的佐证。
    验证指标：候选获益分布与逐 workload 临界性判定一致（响应/不响应互证）；任务周期与引擎 busy 核算在合并后重新闭合。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0115
- type: P
- title: BP_mixcv_cumulative_order——MixCV 跨引擎候选单点验证的累积序调制：与先验矛盾的 workload 族分裂（方向翻转）先检验累积序假设（合入对侧已验证胜出后复测），再谈 per-shape 条件路径
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z（重建会话 Stage 4）：opt_log.md#Round-1-机制分析 + #Round-2-结论与机制（round1_vbrchoist 在无 prevhoist 的 base 上 H=80 族〔2.7B，G=1 下 cb 读放大 3.3×于 H=24〕回退 −4.2~−6.7%〔双引擎 mte2 传输时间 +22~28%，L2 局部性劣化〕；同一形态叠加 prevhoist 之上 model-scale 全转正 +2.2~+10.2%——回退消失；判定「Round 1 的 H 族分裂是单点叠加顺序的伪象，非 vbrchoist 形态本身的缺陷」；本会话曾据 H 分裂考虑 trace-time H 条件分支，累积序复测后证伪该需求——避免了无谓的核内分派复杂度）
  - 数据侧落点：attention.md PL-1.18 update②（2026-09-21 optimizer 已回写、evolver 复核对账相符——机制归因自包含：AIV 侧候选的单点效应受 Cube 侧 base 状态调制，prevhoist 削减 Cube mte2 压力后 L2/总线交互改变）
  - 姊妹条目：queue VP-2026-0110（BP_engine_criticality_drift——临界性随对侧优化漂移；本条是其「方向翻转级」补全：不只收益幅度，效应方向本身可随累积序翻转）+ VP-2026-0112（流程侧每轮重算规则——三案合入时互链）
- repro: repro/PL-1.18-floor2-wins.py（知识域结构骨架——三胜出与累积序效应载体；效应数据见该任务 perf_records round1/round2〔provenance，允许失效〕）
- toolchain_stamp: tilelang 0.1.2+96f287eeaa（与 15ad002 delta 仅 .agents 文档、tilelang 源码零改动）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_mixcv_cumulative_order（目录行与正文置于 BP_engine_criticality_drift 之后——若 VP-2026-0110 尚未合入则置于 BP_run_state_bimodality 之后，两案互链）：
    触发信号：MixCV / 跨引擎（Cube+Vector）kernel 的单侧（Vector 或 Cube）候选单点验证出现与先验证据矛盾的 workload 族分裂——尤其方向翻转（某 workload 族稳定回退而先验记录该形态有效，或反之）。
    检验步骤（先于 per-shape 条件路径）：① 刻画分裂族的结构特征（H/G/读放大比等维度）；② 把候选叠加到「已合入对侧已验证胜出」的 current best 上复测（复现既有累积序）；③ 分裂消失 ⟹ 单点结论依赖 base 状态（叠加顺序伪象）；分裂仍在 ⟹ 才评估核内 trace-time 条件分支。
    机制：单侧候选的单点效应受对侧 base 状态调制（对侧字节削减/链解串行改变 L2/总线交互与双引擎 mte2 传输速度）——「单点分支从 current best 派生」的纪律须以已合入的累积序为基。
    实测（ssd 重建会话，96f287e）：vbrchoist 单点 H=80 回退 −4.2~−6.7%（双引擎 mte2 +22~28%）→ prevhoist 之上 +2.2~+10.2%（回退消失）；机制归因与画像 diff 见 attention.md PL-1.18 update②（互链，不复制）。
    验证指标：族分裂消失 + 全域（model-scale）非回退；与 BP_engine_criticality_drift 的逐 workload 临界性核算结论一致（响应/不响应互证）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0120
- type: D
- title: bm=1 时 T.Parallel 融合循环条件引用 (·,1) 不变操作数被误 lower 为 arith.fptoui(f16→i1)——MLIR verify fail；先 T.vbrc 同形化修复；bm≥2 正常
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：P2 首版编译失败 IR（`arith.fptoui` f16→i1 广播，/tmp 探针日志）；触发条件 `for i,j in T.Parallel(1,N): ite(x[i,j]==m[i,0],...)`（m 形状 (1,1)——条件中的 (·,1) 不变操作数）；vbrc 同形化（先 T.vbrc(m, ext) 展开）后 PASS；DESIGN.md §9.1-C-2
- repro: repro-missing（知识域最小 repro 待同族任务回填；条件形态自包含于 delta）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-parallel-cond-unit-operand-fptoui：bm=1 的 T.Parallel 融合循环条件引用形状 (·,1)/(1,1) 的不变操作数（如 m[i,0] 与标量/布尔比较组合）时被误 lower 为 `arith.fptoui(f16→i1)` 广播——MLIR verify 编译期失败；bm≥2 同形态正常。绕法 = 先 T.vbrc 把 (1,1) 缓冲同形化展开到 (1,N) 再做元素对元素比较。与 TRAP-vbrc-same-shape-empty-broadcast（同形 vbrc 空广播维）构成 bm=1 形态族的两面：前者是「该 vbrc 而未 vbrc」，后者是「不该 vbrc 而 vbrc」。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: -

## VP-2026-0121
- type: D
- title: shared 多消费者缓冲在 auto-multi-buffer 预算超限时静默数据竞争（非确定错误值跨 run 变化）+ 未用 T.alloc_fragment 不被 DCE——fragment 计算形态 + ≤64KB/block 手工预算双纪律
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1/2：design_probe_argmax.py p3c/p3d 首版（fp16/bf16 bm=2，shared 缓冲喂 ≥2 个算子 + 含未用 ext_brc）非确定 BIG 行 15–25/2048、3 run 变化（竞态指纹）；p3a/p3d/p3e fragment 版 3/3 稳定 PASS；DESIGN.md §4.5/§9.1-C-3/C-4；Stage 2 实机重放 p3a bm=2@hidden-state 3/3 稳定复证
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现 = python3 history_version/design_probe_argmax.py --case p3d〔bm=2〕多 run 观察非确定 mismatch——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-shared-multiconsumer-silent-race：alloc_shared 缓冲被 ≥2 个算子消费（多消费者）且 auto-multi-buffer 双缓冲使预算超限时发生**静默数据竞争**——错误值非确定（同行跨 run 变化，15–25/2048 级），无报错无警告，是「最难查的一类」；同场实证：未用的 T.alloc_fragment 缓冲不被 DCE（计入 UB 规划挤占双缓冲预算）。规避双纪律：① 计算缓冲一律 alloc_fragment、alloc_shared 仅作 GM staging（bm/dtype 分派用独立 prim_func，不写「无条件 alloc + 条件使用」）；② 手工 UB 预算 ≤64KB/block（×2 双缓冲 = 128KB ≤ 184KB 可用）。诊断指纹：错误行数跨 run 变化 ⟹ 竞态（vs 全哨兵 = 匹配链断裂、固定行错 = 步长/索引）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0122（CONST-capacity 预算数据点）联动：本条是「超限后果」，该条是「超限阈值实测」。

## VP-2026-0122
- type: D
- title: CONST-capacity-910B2C 追加 argmax 结构膨胀数据点——宽 N 1.60×/2.0×、窄内维 N<8 ~4×（(2048,4) bm=2048 手工 64KB→requires 256KB 溢出，bm=1024 通过）、bm=32@N_padded=256 ~2× 编译实证
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z：DESIGN.md §4.5 预算表（fp16 bm=4@4096 手工 160KB → requires 2097408 bits=256.03KB=1.60×；128KB→256KB=2.0×——宽 N 定标）；Stage 3 首跑 (4096,4)/(524288,4) fp16 bm=2048 手工 64KB → requires 2097408 bits（~4×，窄 N）编译溢出、降档 bm=1024 后 3/3 PASS；Stage 5 集成 run.json：bm=32@N_padded=256 手动预算 80KB、×2 约 160KB < 192KB 编译通过（latency 5145µs，profiler_valid=true）
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现 = _select_config 阶梯复算 + 窄 N bm 阶梯 cap 逻辑——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2（dev root build 2026-09-24 + 集成 013dbbf5）+ CANN 8.5.0 + Ascend910B2C / 2026-09-24~28
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    update 条目 CONST-capacity-910B2C（auto-multi-buffer 膨胀系数段）追加数据点：① 宽 N（N≥256）argmax 结构实测 1.60×（fp16 bm=4@4096：160KB→256.03KB）与 2.0×（128KB→256KB），与既有 ×1.7 量级互证；② **窄内维 N<8 膨胀 ~4×**（(2048,4) fp16 bm=2048：手工 64KB → requires 256KB）——窄 N 的扩展阶梯（小 N 大 M）预算解除不能沿用宽 N 的 64KB/block 口径（Stage 3 实测 bm 2048→1024 才过）；③ bm=32@N_padded=256 手动 80KB ×2 ≈160KB < 192KB 编译通过（集成环境实证）。交叉注记 TRAP-UB-multibuffer-inflation（膨胀条目）与 VP-2026-0121（超限静默竞态——预算红线不仅是编译溢出，超限先于溢出即竞态）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 数据点为多 attempt 汇总（设计探针 + Stage 3 首跑 + Stage 5 bench），同一任务内互证但非跨任务独立确认。

## VP-2026-0123
- type: D
- title: TRAP-vbrc-same-shape-empty-broadcast 追加窄 N 垃圾索引形态——bm=1 vbrc (1,1)→(1,N) 在 N<8 产出垃圾/哨兵索引（N=2..6 全错、N=6→2^30），N=1 编译失败（同形空广播），N=8/bm≥2 正常
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 3：调试 (1,N) bm=1——N=8 PASS、N=2..6 FAIL（N=6 → 1073741824=2^30 全哨兵）、N=1 compile fail（empty broadcast dims，与 TRAP-vbrc-same-shape-empty-broadcast 同族）；bm=8 时 N=3/5 奇数窄列亦失败、N=4/8 通过；契约内最小 N=4 已由 L0-6 覆盖故未阻塞
- repro: repro-missing（调试脚本在 /tmp 探针（可转正）；条件形态自包含于 delta）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update 条目 TRAP-vbrc-same-shape-empty-broadcast（追加窄 N 形态段，原文保留）：除同形空广播维编译失败（N=1 形态）外，**vbrc (1,1)→(1,N) 在 N<8 时产出垃圾索引**（cand 全落 BIG 哨兵/错误 int64；N=6 实测全 2^30）——窄 N 向量化对 vbrc 广播与 candidate 循环的边界在 N=8；bm≥2 直接索引 + N=4 正常。窄 N（N<8）负载的候选索引构造避开 (1,1)→(1,N) vbrc 形态。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: -

## VP-2026-0124
- type: D
- title: T.Parallel 内标量直接写 GM 输出张量元素（无 UB staging）产出错误值——同行 int64 主路径经 alloc_shared + T.copy 出栈正确（单次观察，待二次隔离证据）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2：design_probe_argmax_dbg.py kern_same 的 `dbg[pid*bm+i,0] = T.cast(ext16[i,0],"float32")`（L69-70）重放输出 extreme=[-0.845,0.452,-1.135,-0.852] vs torch=[3.36,3.08,3.10,2.67] 而同行 ok=True（经 out_ub+T.copy 的索引主路径正确）；对照 kern_mixed 的 T.copy(ext32, dbg[...]) 导出路径（该变体错值来自混合 dtype reduce 非导出路径）
- repro: repro-missing（隔离实验未做：把 dbg 改成 UB staging + T.copy 后是否恢复——单次观察，合入前需第二次隔离证据；任务内复现命令见 evidence provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：cd examples/argmax/_argreduce_kernel/history_version && python3 design_probe_argmax_dbg.py（看 [C-same] 的 extreme 行 vs ok 行）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-parallel-scalar-direct-gm-write（待二次隔离证据后合入）：T.Parallel 循环体内标量表达式直接写 GM 输出张量元素（无 UB staging）实测产出错误值；同一 kernel 内经 alloc_shared + T.copy 出栈的主路径正确。隔离归因待补：UB staging + T.copy 化后是否恢复（当前不排除与调试 buffer 布局的其他交互）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 提案源头（Stage 2 复盘）自标「单次观察，合入前需第二次隔离证据」——本条即按该口径入队等待。

## VP-2026-0125
- type: D
- title: argreduce 边界语义三基准分野——torch-NPU 设备 argmax/argmin 对 ±0.0 用可区分全序（vs torch-CPU IEEE 相等首现）；NPU reduce 传播 NaN → 掩码匹配式输出哨兵 2^30（vs torch 首 NaN 索引）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：探针实测（DESIGN.md §9.2-R1/R2）——`[0,-0,0]`→npu argmax=0/argmin=1、`[-0,0,-0]`→argmax=1/argmin=0，CPU 全 0（三 dtype 一致复现）；NaN 行 4 位形：NPU reduce 传播 NaN 使掩码匹配全失败 → 输出哨兵 2^30，torch 返首 NaN 索引；±inf 精确
- repro: repro-missing（知识域最小 repro 待同族任务回填；±0.0 复现 = python3 -c "import torch,torch_npu; ..." 对照 CPU/NPU——provenance 允许失效）
- toolchain_stamp: torch 2.x npu + CPU 对照 + tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-argreduce-edge-semantics（golden 基准选择类）：① torch-NPU 设备 argmax/argmin 对 ±0.0 用**可区分全序**（argmax 偏好 +0.0、argmin 偏好 −0.0），与 torch-CPU（IEEE 相等取首现）不一致——跨设备 golden 对齐时 ±0.0 混合行是已知分歧点；② NPU 硬件 reduce 传播 NaN，掩码匹配式 argmax（vmax/vcmp 链）在 NaN 输入下全匹配失败输出哨兵 2^30，torch 返首 NaN 索引——含 NaN 的负载须在 L2/边界用例记录行为差（记录不断言崩溃），golden 选择与容差设计据此分野（torch-CPU golden / torch-NPU 设备实现 / 硬件 reduce 语义三方各不相同，设计期逐个实测记录比统一假设更诚实）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: -

## VP-2026-0126
- type: D
- title: multitile bm>1 组合触发编译器 SIGSEGV（触发矩阵：bm=1 任意 tile ✓ / bm>1 单 tile ✓ / bm>1 多 tile ✗）——前次任务 repro 被清理后靠 pyc 考古恢复；CG-2026-0014 互链
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：capability-gaps CG-2026-0014（本任务 §9.4-R9 补录——前次任务已登记但条目随工件清理丢失，源 repro .py 仅存 repro/__pycache__ pyc）；DESIGN.md §9.1-C-5 触发矩阵引用（本任务引用结论未复现崩溃本身）；pyc 反序列化 docstring 考古恢复路径见 Stage 1 复盘 Info-source 行
- repro: repro-missing（源 repro 已清理仅存 pyc——重建触发矩阵三形态最小 kernel 待同族任务回填：bm=1 多 tile / bm>1 单 tile / bm>1 多 tile）
- toolchain_stamp: 前次任务 2026-09-23（tilelang 0.1.2 dev build + CANN 8.5.0 + Ascend910B2C）；本任务引用环境 2026-09-24 同源
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    不执行原提案。编译器缺陷、触发矩阵与 bm=1 正确性绕法已由 capability-gaps
    CG-2026-0014 承载，且该登记簿现已纳入 kb_search，无需重复写入 pattern-library。
- status: rejected
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: human 2026-10-10
- decided_note: 重复登记；原性能绕法会增加 device kernel launch 数量，不满足源/目标 launch 一一对应约束。

## VP-2026-0127
- type: D
- title: T.vcast f32→f32 非恒等（数据破坏 ~0.5 diff）——同 dtype 缓冲复制用 T.copy；前次任务 pyc 考古恢复
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：repro/__pycache__ pyc docstring 考古（前次任务 2026-09-23 实证）；Stage 1 Transferable Lessons（「禁 T.vcast f32→f32（非恒等、数据破坏 ~0.5 diff）；同 dtype 缓冲复制用 T.copy」）
- repro: repro-missing（知识域最小 repro 待同族任务回填——f32 buffer 经 T.vcast f32→f32 后与 T.copy 对照 diff）
- toolchain_stamp: 前次任务 2026-09-23（tilelang 0.1.2 dev build + CANN 8.5.0 + Ascend910B2C）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-vcast-f32-f32-not-identity：T.vcast 在 src/dst 同为 f32 时非恒等——实测数据破坏（~0.5 量级 diff）；同 dtype 缓冲复制用 T.copy。与 TRAP-C12-copy-dtype-cast（T.copy 跨 dtype 静默转换）构成镜像对：「copy 换 cast 会静默变值」vs「cast 当 copy 用会静默坏值」。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 前次任务实证（经 pyc 考古恢复），本任务 Stage 1 消费未复现——证据链含考古 provenance。

## VP-2026-0128
- type: P
- title: TileOPs reduce 族 Op 层 pad 契约与 host pad 消除——kernel 声明宽度须等于 Op 层 padded 宽度（错位症状：行 0 对、行≥1 全错不报错）；高杠杆项 = `_kernel_handles_padding=True` + kernel 原始 N 契约（小 N 负载最高 64× 流量改善）；argmin 共享 kernel 时 `_pad_value` 须 +inf
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 5：integration_log.md attempt-1（初次 report 17 failed 与 N_padded≠N 用例 100% 重合；最小复现：声明 (8,300) 喂 (8,300) 全对、Op 层 pad 到 (8,512) 后行 ≥1 全错——步长错位）；修复 = forward/__init__ 传 N_padded，42/42 通过
  - host pad 消除证据：tileops/ops/reduction/reduce.py 模块注释（"until their kernels are converted"——仓库既定方向）+ DESIGN.md §0.6-R3/§5.5 + 前次任务 perf_opt/logs（3d 负载 ~195–204µs 主因 host pad，262KB/4KB 流量比）；Stage 5 集成 bench 对照：padded 路径 bm=32@N_padded=256 = 5145µs vs standalone 原始 N narrow 路径 12.5µs
  - argmin 方向性：argmax.py wrapper __init__ 注释 + integration_log 遗留说明（-inf pad 永不赢 argmax、全 -inf 行 kernel 与 torch 同返 0；argmin 必须 +inf）
- repro: repro-missing（集成契约需 TileOPs 环境；错位最小复现 = 构造 (8,300) fp16 输入，声明 (8,300) + Op 层 pad (8,512) 喂入 → 行≥1 错位——provenance 允许失效；知识域锚点 = cases.md CASE-argmax-argreduce-integration 已 Tier 0 合入）
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir / Ascend910B2C / CANN 8.5.0 / git 013dbbf5 / 2026-09-28
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-reduction 行 R4 追加注记：**Op 层 pad 契约**——TileOPs reduce 族 Op 层默认把输入 pad 到 align_up(N,256)（`_ReduceOpBase._prepare_input`，`_kernel_handles_padding=False` 时）；kernel jit 声明宽度须取 N_padded（raw-N 契约与 F.pad 冲突症状：行 0 正确、行 ≥1 全错且不报错）；**迁移高杠杆项** = 翻转 `_kernel_handles_padding=True` + kernel 接原始 N（规约单位元 padding 语义上不改变结果），小 N/非对齐 N 的 dim=0/中间维负载最高 64× 流量改善（harness reduce.py 注释明示为仓库既定方向）；op_kind 涉及 min/max 的共享 kernel 须核对 `_pad_value` 方向性（argmax −inf 安全 / argmin 必须 +inf）。kb_links 补 CASE-argmax-argreduce-integration。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0134（design SKILL 集成前置核对）为同一发现的知识侧/流程侧两 Tier 拆分。

## VP-2026-0129
- type: P
- title: reduction 族小 N 大 M 负载设计规则集——workload (M,N) 按 Op 层 reshape 规则推导（dim=0 → M=prod(其余)、N=该维长度）；block_m 由发射项决定 + 门控扩展阶梯 {16..2048}（窄 N 膨胀 4× 换算 + cap bm≤M）；tile_n 整除优先 + 裕度后置校正
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2/revision/3/5：REVIEW.md 问题 1（v0 (512,4) 手抄 vs manifest+Op 层真值 (524288,4) 差 1024×——错误根因是抄错后无推导式可自检）；DESIGN.md v1 §5.2/§5.5（门控代码：基础阶梯 {1,2,4,8}〔GPU 48KB SMEM 承袭〕下 block 数 > 48×2 时启用 {16..2048} 档、UB 预算过滤器封顶——3d (524288,4) fp16：bm 8→2048、65536→256 block、发射项 4.8ms→18.7µs）；Stage 3 修正（bm=2048 窄 N 4× 膨胀溢出 → 实取 1024；(1,3)/(1,5) bm=8 垃圾行 → cap min(block_m,M)）；Stage 5 bench（bm=32@N_padded=256 编译通过）；tile_n 规则（DESIGN v1 §5.2：候选集 = N 因子 ∩ 256 倍数 ∩ ≤cap ∩ slab·elem·tn ≤ 0.9×UB_MANUAL_BUDGET 取最大，空集回退 cap+静态尾 tile——复现 lm-head fp16 5120 / bf16 4096）
- repro: 纯工厂期算术复算（host 侧、设备无关）：65536/(4×8)=2048；ceildiv(524288,2048)=256；256×7op÷48×0.5µs≈18.7µs；反事实 ceildiv(524288,8)→≈4.8ms；factors(102400)≤6400 且 5×2×tn≤58982 → 5120
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C（发射常数 CONST-vector-launch-overhead）/ 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-reduction 行 R4 追加小 N 大 M 负载设计规则集：① **(M,N) 推导**——manifest 负载一律按 Op 层 reshape 规则重算（dim=0/中间维 → M=除规约维外全维乘积、N=该维长度；movedim(dim,−1) 后 reshape），不按 x_shape 字面读（v0 差 1024× 实证）；② **bm 由发射项决定**——小 N 大 M 负载 UB 预算允许的 bm 远大于 GPU 承袭阶梯 {1,2,4,8}（block 数爆到数万、每 block 几个向量 op、0.5µs/op 发射主导：65536 block ≈4.8ms vs bm=2048 256 block ≈18.7µs）；门控扩展阶梯 {16..2048}（blocks > aicore×2 时启用）+ UB 预算过滤器封顶；**窄内维 N<8 的预算解除按 ~4× 膨胀换算**（宽 N 2× 口径在 (2048,4) 实测溢出，降档 1024 才过）且 **cap block_m = min(block_m, M)**（bm>M 尾行是 UB 未初始化垃圾、窄 N 下被向量化器污染有效行）；③ **tile_n 整除优先 + 裕度后置**——候选集 = N 的因子 ∩ 256 倍数 ∩ ≤ tile_n_cap ∩ slab·elem·tn ≤ 0.9×手工 UB 预算，取最大（零尾 tile + ≥10% 裕度）；空集回退 cap + 静态尾 tile（不依赖 compute_tile_n 的除数接受判据——其「除数优先」仅在不增加 tile 数时生效）。kb_links 补 CONST-vector-launch-overhead / VP-2026-0122（窄 N 膨胀）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0044（row-reduction 基准形态，已合入）互补：0044 覆盖 N 可驻留的常规形态，本条覆盖小 N 大 M 与 N 超驻留两端的阶梯/分派规则。

## VP-2026-0130
- type: P
- title: 调优迭代效率两手法——① 归因「二选一」嫌疑列表当轮用变体对照消歧（不留到后续轮——argmax 标量化归因跨轮证伪实证）② 高成本负载先微基准定标（copy-only 行宽-效率曲线 W≥32 渐近）再上真 shape 采数
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 4：opt_log Iteration 4（session-1 对 baseline 标量化留两个嫌疑〔(bm,1) 列广播操作数 / varange 生成失败〕未闭合，round 4 r4a 操作数变体对照一次证伪两者——真触发器是 2D T.Parallel(bm,N) 融合循环本身；证伪更正留痕 canonical §2-3）；Iteration 2（probe_gran copy-only 微基准定标行宽曲线：W=4→9.74µs / W=8→6.28 / W=16→5.56 / W=32→4.52µs，W≥32 达渐近线——r2a 窄行宽视图设计直接建立在该曲线上，避免真 shape 1632µs 负载多轮高成本采数）；SESSION_TIMING_ANALYSIS §3（诊断消歧并行化建议 + 3d baseline 采数最贵）
- repro: none（方法类：判据与实例自包含于 delta；效应载体 = traps-compiler.md TRAP-parallel2d-bm-ge2-scalarize 证伪更正段）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24~28
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add「迭代效率两手法」小节：① **嫌疑列表当轮消歧**——Phase 1 诊断产出「二选一」式归因嫌疑时，当轮即对每个嫌疑派一个操作数/形态变体实验分支做对照（各 ~1 分支成本），不跨轮次延迟：argmax 实证（session-1 两嫌疑留到 round 4 才由 r4a 变体对照双双证伪——若 round 1 当轮消歧，链式形态可提前 3 轮锁定；证伪更正按 canonical §2-3 留痕：误判根因 + 合法形态 + 新数据）；② **高成本负载微基准定标先行**——baseline 采数最贵的 workload（多 wave/读膨胀/诊断链长）先用更小的快速回归 shape 或 copy-only 微基准定标关键常数（行宽-效率曲线、膨胀系数），再上真 shape——argmax 3d 负载的窄行宽视图设计直接建立在 probe_gran W 曲线（W≥32 渐近 4.52µs/4MB）上，避免 1632µs 级负载的多轮盲扫。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: ①的实证即 TRAP-parallel2d-bm-ge2-scalarize 的证伪更正链（Stage 4 已回写知识侧），本条是其流程侧（当轮消歧纪律）的条目化。

## VP-2026-0131
- type: P
- title: 索引输出 kernel 的失配行特征三指纹——全哨兵（2^30）= 匹配链断裂 / 跨 run 行数变化 = 竞态 / 全落 tile-0 局部索引 = loop-carried 状态未携带；int64 索引 torch.equal 精确比对是免费 oracle
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1/3：p3c/p3d shared 多消费者竞态（BIG 行 15–25/2048 跨 run 变化）；Stage 3 首跑 vselect 状态丢失（got=[4779,3248,2345,1833] 全落 tile-0 局部索引）；N=6 vbrc 窄 N（cand 全 2^30 哨兵）；Stage 3 复盘 Transferable Lessons（「失配行特征比数值回归更好定位根因」）
- repro: none（指纹判据自包含于 delta；三个指纹的载体实例分别见 VP-2026-0121/TRAP-vselect-inplace-carried-state/VP-2026-0123）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 条目「索引输出 kernel 失配行特征三指纹」（置于定征顺序条目之后，互链）：argmax/argmin 类输出 int64 索引的 kernel，`torch.equal` 精确比对是免费 oracle（任何 tie-break/哨兵/状态携带 bug 立即暴露）；失配行的**空间与时间特征**先于逐元素排查定根因类别——① 输出全落哨兵值（2^30）⟹ 掩码匹配链断裂（NaN 传播 / vbrc 窄 N 垃圾）；② 错误行数跨 run 变化 ⟹ 竞态（shared 多消费者超预算）；③ 输出全落 tile-0/首块局部索引 ⟹ loop-carried 状态未携带（原位 vselect 类）；固定行错位（行 0 对行≥1 错）⟹ 步长/声明宽度错位。与 VP-2026-0006（定征顺序）、VP-2026-0038（误差结构定征）同族互补：本条是索引输出域的空间指纹版。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0061（lse 恒值指纹）同「错误指纹速查」族——合入时互链。

## VP-2026-0142
- type: D
- title: Ascend910B3 单 AIV 核 kernel launch 地板 = 1.89µs（1 block / 512B copy）；40 block 空载 3.36µs——roofline ideal < 2µs 的 workload 其 Ratio 上限 <16%，与实现无关
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z 的 perf_opt/opt_log.md §2.2（H1 反解实测常数表）/§4.3（60% 目标裁决）；profiles/probe/{floor_nblk1_nch1,floor_nblk40_nch1}（provenance，允许失效）
  - 同任务 §4.2 应用：b1（decode d_mem=512 d=256）roofline ideal 0.306µs → 0.306/1.89 = 16.2% 硬上限；实测 1.0753% 说明该 workload 的 Ratio 与代码质量近乎无关
  - 设备实查：NPUUtils.get().get_aicore_num()==20 / AIV 40 核 / get_arch()=="Ascend910B3"
  - **第二证（不同任务，2026-10-31，task engram_decode_mcore-engram_decode_mcore-20261031T012849Z）**：独立探针 `history_version/design_probe_floor.py`（`--nblk n`，同口径 msprof op `--launch-count=10 --warm-up=3`）在本戳重测 **FLOOR(1)=1.570 / FLOOR(8)=1.600 / FLOOR(32)=2.520 / FLOOR(40)=2.850 µs**（aiv_time 0.80–1.08µs，其余为发射/收尾）——与首证 76e4550 上的 1.89 / 3.36µs 相比系统性偏低 ≈17–19%（**工具链漂移，两戳各自成立、引用须带戳**；本任务的设计期上界推导一律用本戳值）。同任务用途：b1 的 40% 门槛 0.7635µs 比 FLOOR(1)=1.570 低 2.06×，单条即封死 40%（DESIGN §5.6.2 论证 1）。
  - **第三数据点（不同任务，2026-11-01，task TileOPs-_make_rope_neox_1d-20261031T184833Z，**不计确认**——本条已 2/2，且本条探针是空 kernel，本数据点是同形态骨架，两者口径不同）**：40 AIV 满开、7 个 DMA/UB 描述符 + 循环 guard 的「同形态骨架」探针（每核 1 行 10 KB）Task Duration **3.52 µs**（`aiv_time` 1.78，差值 1.74 为 launch/ramp）。与第二戳 FLOOR(40)=2.850 µs 相差 +0.67 µs（≈+24%），差值即描述符与循环 guard 的结构项——**空 kernel 地板用作 Ratio 上界时对多描述符 kernel 偏乐观**（同一任务按 1.897 µs 算出的上界 15.5/63.1/100 ⇒ 均值 59.5%；按 3.52 µs 为 8.3/34.0/86.3 ⇒ 均值 ≤42.9%）。常数条目化见 VP-2026-0173。
- repro: repro-missing（复现条件：单 block kernel 做一次 512B GM→UB→GM 连续 T.copy，`msprof op --launch-count=20 --warm-up=5` 取 Task Duration 中位数；**探针自校正纪律**：探针必须先用同形态连续搬运校正——本轮首版写成 stride-32 gather store 得到 33µs 荒谬值〔MTE3 1GB/s〕，改为连续 T.copy 后同为 1.89µs）
- toolchain_stamp: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 条目 CONST-launch-floor-910B3（置于 CONST-vector-launch-overhead 之后；与旧条目的区别：旧条目是「向量算子发射 ~0.5µs/op」，本条是 kernel 级 launch/teardown 地板——两者量级与适用面不同，不合并）：

    ---
    id: CONST-launch-floor-910B3
    kind: constant
    family: [general, reduction, gemv]
    apis: [T.Kernel, T.copy]
    dtype: []
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T121905Z
    toolchain: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
    source: PL-1.6-copy-floor 同族方法 / 本任务 h1_probe.py
    repro: repro-missing
    ---

    ### kernel launch 地板 ≈1.89µs（1 block，512B GM→UB→GM；Ascend910B3）

    1 block、仅 512B 搬入搬出的空 kernel 实测 Task Duration **1.89µs**（`aiv_time` 1.36µs）；同工作量铺满 40 block 时 **3.36µs**。用法：把 roofline ideal 时间除以 launch 地板即得该 workload 的 **Ratio 硬上限**——`ideal / 1.89µs` 与代码好坏无关。实测例：decode b1（理想 0.306µs）上限 16.2%，而其核数比例上限更低（1/40 核）。判据：`grid < 物理核数` 且 `Task Duration < 50µs` 时，先测地板再谈天花板，否则会把「不可能」误判为「还没优化够」。**探针自校正**：同用途探针须用与目标同形态的连续 `T.copy` 编写——本轮 stride-32 gather store 形态给出 33µs（MTE3 ~1GB/s）的荒谬值，改连续搬运后复现 1.89µs。
- status: pending（2/2 已达成但 **repro-missing 保持 pending**——与 VP-2026-0143/0153 同口径，待同族任务补知识域 repro 后端升 Tier 0 合入 constants.md；另注：数值随工具链漂移 1.89→1.570µs，合入时按「双戳并列」写）
- confirmations: 2/2
- confirmed_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31（第二证：独立探针重测本戳地板 1.570/1.600/2.520/2.850µs，见 evidence 末行）
- created_by: task TileOPs-_engram_decode_kernel-20261030T121905Z 2026-10-30
- decided_by: -
- decided_note: -

## VP-2026-0143
- type: D
- title: Ascend910B3 单 AIV 核 MTE2 常数：流式 GM 读 68 GB/s（fp16→fp32 转换 62 GB/s，转换不额外收费）、每次 DMA 指令固定开销 ≈0.16µs——tile < ~16KB 时搬运时间由指令数而非带宽决定
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z 的 perf_opt/opt_log.md §2.2（三点拟合：512KB 分 64/32/16 次 = 13.97/9.38/6.65µs）；profiles/probe/{conv_c4096_x64,conv_c8192_x32,conv_c16384_x16,read_nblk1_nch8,convert_nblk1_nch8}（provenance，允许失效）
  - 同任务 §4.1 用途：DESIGN D-2 估算失准 2.4–2.9× 的三个主因之一即「单核带宽 ≠ 芯片峰值」；§2.5 R2-H1 用途：把 11–19 行逐行 2 次小 DMA 合并为整块 2-D copy，三档 workload −5.86%/−6.14%/−8.75%
  - 与既有 CONST-mte2-degradation（910B2C，48 核聚合口径 18–43 GB/s/core）**口径不同**：本条是 910B3 单核单 launch 的流式口径（68 GB/s），不构成矛盾，合入时须并列注明
  - **第二证（第二任务）** task TileOPs-_engram_decode_kernel-20261030T145026Z（round 3，tilelang 换代为 +01805d7a）：同族 kernel 的 `aiv_scalar_time` 与每 block 的 W/e 拷贝**条数**逐档反解——b1 d=256 BK=16（96 条，scalar 19.61 µs → 0.204 µs/条）、b4 d=512 BK=16（192 条 → 40.17 µs → 0.209 µs/条）、b1 BK=32（48 条 → 15.49 µs → 0.323 µs/条）；数据来自 perf_opt/profiles/{baseline3,final3,round3}/*/prof.json（provenance，允许失效）
  - **第二证的关键修正（固定底噪）**：BK 16→32 把 b1 的条数减半（96→48）但 scalar 只降 19.61→15.49（−21%），**不满足无截距线性**——两点拟合（96→19.61、192→40.17）得斜率 0.214、截距 −0.94 µs，对 48 条预测 9.3 µs 而实测 15.49 µs ⟹ b1 存在 ≈6 µs 的固定 scalar 底噪（循环控制 + 向量/条件指令发射），«每指令 0.2 µs × 条数»只能当**上界**估算，不能当预测式（BK=64 时 scalar 仍 13.6 µs，印证底噪存在）
  - **第三数据点（2026-10-31，task engram_decode_mcore-engram_decode_mcore-20261031T012849Z，工具链 3ed7c914）：段宽决定带宽，64B 段触发 2× payload 放大**——同一 910B3 单核，split-d 的**跨步列块** GM→UB 读 per-core 有效带宽实测 **26–30 GB/s**（b1 129.1KB/4.317µs = 29.9；b4 258.1/8.563 = 30.1；b8 129.1/4.961 = 26.0），与连续 32KB 流式的 68 GB/s 相差 2.3×；且列切片 <128B（`D_s=32` fp16 = 64B）时 payload **放大 2×**（每块读 129.1KB 而有效仅 64.5KB），`D_s≥64`（128B）时无放大（1.0×）。**引用纪律**：910B3 单核 MTE2 存在两个不可混用的口径——连续整行流式（68 GB/s）与跨步列块（26–30 GB/s，且随段宽劣化）；设计期流量项按被访问形态选口径（DESIGN §1.6.0 R4 的 split-d/split-K 两栏即分别使用两者）。该条同时是 `CONST-gm-to-ub-bw-dilution`「段宽 ≥ 数百字节时性能正常」的下界补充（64B 段低于该阈值）。
- repro: repro-missing（复现条件：单 block 连续 32KB T.copy 循环 —— 改 chunk 数 16/32/64、总字节固定 512KB，用 msprof `mte2_active_bw` 与 Task Duration 三点拟合直线，斜率即 per-DMA 固定开销）
- toolchain_stamp: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    update 条目 CONST-mte2-degradation（原位追加 910B3 单核段 + front-matter device/origin_task/toolchain 扩写；不新开条目——同主题「MTE2 有效带宽 / 搬运代价」已有载体）

    new 段（追加于该条目 2026-09-21 段数更正段之后）：

    **〔2026-10-30 第二设备族数据点：Ascend910B3 单核 + 指令维度绝对开销（task TileOPs-_engram_decode_kernel-20261030T121905Z）〕** 910B3（20 AI Core / 40 AIV）单核单 launch 流式 GM 读实测 **68 GB/s**（fp16→fp32 转换形态 62 GB/s——**dtype 转换不额外收费**，只随字节计）；**每次 DMA 指令固定开销 ≈0.16µs**（512KB 用 64/32/16 次搬运 = 13.97/9.38/6.65µs 三点拟合）——由此得判据：**tile 小于 ~16KB 时，搬运时间由指令数而非带宽决定**，合并小 copy 优先于提高带宽。口径注：与本条上方 910B2C 聚合曲线（18–43 GB/s/core，48 核 grid-stride/4-set 轮换）不是同一口径（单核 vs 聚合、单 launch vs 大流量），引用时须注明设备与并发度。应用例：11–19 行 state-shift 的逐行 2 次小 DMA（每次 ~0.16µs 固定开销）合并为整块 2-D copy，三档 workload 提速 5.86% / 6.14% / 8.75%。

    **〔2026-10-30 第二次实测 · 工具链 +01805d7a：每指令开销上修到 0.20–0.21 µs，且存在 ≈6 µs 固定底噪（task TileOPs-_engram_decode_kernel-20261030T145026Z round 3）〕** 同口径三点：96 条 → 19.61 µs（0.204 µs/条）、192 条 → 40.17 µs（0.209 µs/条）、48 条 → 15.49 µs。前两档支持「每 DMA 指令 ≈0.2 µs」，第三档不满足无截距线性（48×0.21=10.1 ≪ 15.49）⟹ **该 kernel 的 scalar 至少含 ≈6 µs 与条数无关的固定底噪**（循环控制/向量指令发射），故「合并 DMA 的收益上限 = Δ条数 × 0.2 µs」只在条数差占主导时成立。版本对照：76e4550 上 0.16 µs、01805d7a 上 0.20–0.21 µs（+25%，同一物理常数的工具链漂移，引用时须带戳）。
    应用推论：BK 16→32（条数减半）实测只换来 scalar −21%、duration −5.7%（`PL-1.23-stage-tile-max-legal` 候选，见 queue VP-2026-0152）；把 BK 再推到 64 时 scalar 继续降到 13.6 µs 但 pipeline 崩（30.87 µs）⟹ 底噪存在时「条数 → 0」不是可行目标。
- status: pending（**2/2 已达成但仍 pending**：repro-missing + 数值随工具链漂移 0.16→0.20–0.21 µs，待同族任务补 repro 并确认哪个戳下取值后合入）
- confirmations: 2/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T121905Z 2026-10-30
- confirmed_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31（round 3 第二证 + 固定底噪修正）
- decided_by: -
- decided_note: -

## VP-2026-0144
- type: D
- title: shared/UB buffer 的「含串行循环变量的 affine 组合索引」做行广播——编译通过但静默取错地址（结果错 y_maxdiff=1.49）；合法形态 = 每轮先 copy 成 (BK,) 小 buffer 再用纯循环变量索引
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z 的 perf_opt/opt_log.md §2.4 R2-B/§2.5（`prod[kk,j] = w[kk,j] * e_local[k0*BK+kk]` 编译 pass、L0 fail `y_maxdiff=1.49`，被 `bench_driver.py --check` 单例 golden 拦下）；`perf_opt/_engram_decode_kernel_r2b_elocal.py`（失败留档，provenance 允许失效）
  - 对照合法形态：每轮 `T.copy(e_src[k0*BK:(k0+1)*BK], e_n)` 后用 `e_n[kk]`（纯循环变量索引），同 kernel 其余逻辑不变即 L0 通过
  - 语义关联但不重复：queue VP-2026-0087/V-0088 族记的是**算子 Shape 形态**（vmul/vsub/vadd 无 [M,N]*[1,N]、vmin 支持列广播）——本条是**索引表达式**被错降级，机制不同
- repro: repro-missing（复现条件：UB 上 (1,N) buffer 用 `buf[k0*BK+kk]`（k0/kk 为 T.serial 循环变量）作二元 op 的一侧操作数，与「先 copy 出 (BK,) 再 `buf2[kk]`」两形态对拍同一 golden）
- toolchain_stamp: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-shared-row-broadcast-affine-index（置于 TRAP-reduce-mixed-dtype-silent-corrupt 之后；互链 PL-1.15-broadcast-perf-tax 与 VP-2026-0087 广播形态族）：

    ---
    id: TRAP-shared-row-broadcast-affine-index
    kind: trap
    family: [gemv, reduction, elementwise]
    apis: [T.copy, T.vmul]
    dtype: [fp16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T121905Z
    toolchain: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
    repro: repro-missing
    ---

    ### 共享 buffer 的行广播索引含串行循环变量 affine 组合 → 编译通过、静默取错地址

    shared/UB 上 `(1,N)` 的行向量按 K 分块被消费时，若直接以 `e_local[k0*BK+kk]`（`k0`/`kk` 为 `T.serial` 循环变量）作二元 op 的一侧操作数，kernel **编译无警告、无报错，但取到错误元素**——实测 GEMV 输出 `y_maxdiff=1.49`（该 kernel 的 L0 门禁把结果错拦在最便宜的阶段）。合法形态：每轮先 `T.copy(e_src[k0*BK:(k0+1)*BK], e_n)` 拷成 `(BK,)` 小 buffer，再用**纯循环变量** `e_n[kk]` 参与运算（多一次 (BK,) 小 copy，代价可忽略）。判据：广播操作数的索引表达式须是「平行维自身」的纯函数（`e_n[kk]`），含循环变量 affine 组合（`k0*BK+kk`）时按本条处理。诊断指纹：编译全绿 + 单元 golden 量级级错值、无固定行/列模式 ⟹ 索引降级而非 Shape 或 dtype 问题（区分见 TRAP-reduce-mixed-dtype-silent-corrupt）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T121905Z 2026-10-30
- decided_by: -
- decided_note: -

## VP-2026-0146
- type: P
- title: 逐行小 DMA → 整块 2-D 区域 copy 合并——state-shift/滑窗类尾段的三档稳定收益 5.9–8.8%（机制 = 每 DMA 指令 ≈0.16µs 固定开销）
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z 的 perf_opt/opt_log.md §2.4 R2-H1/§2.6（`new_conv_state` shift 从「每行 2 次小 copy」改为 `(max_conv_len-1, d_padded)` 两次整块 2-D copy：28.411/73.591/28.571µs，−5.86%/−6.14%/−8.75%，L0/L1 全过且 state-shift 逐位一致 max_err=0.0e+00）；perf_records.jsonl baseline vs final 行可对账（provenance，允许失效）
  - 边际递减证据：同类延伸（同时合并 conv_w 3 行小 copy）打平 <0.2%——收益集中在「行数 × 每次 2 条」这类指令数主导的位置，不与搬运字节数成正比
  - 机制归因：constants.md CONST-mte2-degradation 910B3 段（每 DMA 指令固定开销 ≈0.16µs）；行数 11/19/17 时 2×行 次小 DMA ≈ 3.5–6µs，正是收益量级
- repro: repro-missing（慢→快关键更改：慢形态 = `for r in T.serial(L-1): T.copy(src[r,:], dst[r+1,:])`（逐行 2 次 copy）；快形态 = `T.copy(src[0:L-1, :], dst[1:L, :])` 一次 2-D 区域 copy。效应在完整 kernel 规模显现，小规模对照只验证正确性与方向）
- toolchain_stamp: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add 条目 PL-1.22-row-dma-batch-merge（置于 PL-1.21-overlap-saturation-signal 之后）：

    ---
    id: PL-1.22-row-dma-batch-merge
    kind: pattern
    family: [recurrent, elementwise]
    apis: [T.copy, T.serial]
    dtype: [fp16, bf16]
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T121905Z
    toolchain: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
    repro: repro-missing
    ---

    ### 逐行小 DMA → 整块 2-D 区域 copy（state-shift / 滑窗搬移类）

    **慢形态**：`for r in T.serial(L-1): T.copy(src[r,:], dst[r+1,:])`——每行 1–2 条 DMA 指令，L=11–19 行即 22–38 条指令；每次 DMA 的固定开销 ≈0.16µs（constants.md CONST-mte2-degradation 910B3 段），指令账 ≈3.5–6µs，占该 kernel 10–20%。
    **快形态**：`T.copy(src[0:L-1, :], dst[1:L, :])` 一次 2-D 区域 copy（区域语义按 TRAP-T-copy-region-semantics 校验边界）。
    **实测**（910B3，recurrent decode 三档 workload）：28.411 / 73.591 / 28.571µs vs 30.181 / 78.402 / 31.311µs（−5.86% / −6.14% / −8.75%），L0 与 L1 全过、state-shift 逐位一致；同类延伸（把 conv_w 的 3 行小 copy 一并合并）打平 <0.2% ⟹ 收益由**行数**驱动而非总字节数。
    **判据**：滑窗/状态搬移段的总时间 ≈ 行数 × 每行 DMA 条数 × 0.16µs 时优先合并为整块；行数 ≤3 时不必改（噪声内）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T121905Z 2026-10-30
- decided_by: -
- decided_note: -

## VP-2026-0149
- type: D
- title: 用行 padding 修 UB bank-group 冲突会让 vec 时间翻倍——非 2 的幂/非 128B 对齐行宽把连续向量段切成碎段（b1 17.57→31.81 µs）
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T145026Z 的 perf_opt/opt_log.md §2.3 R3-M / §2.5（`r3m_pad8`：reduce 操作数行宽 `d_padded`→`d_padded+8` 个 fp32）+ perf_records.jsonl candidate `r3m_pad8` 三行（48.371 / 120.452 / 49.271 µs）（provenance，允许失效）
  - 同任务 profiles/round3/r3m_pad8_b{1,4,8}/prof.json 的 pipe 三元组：vec 17.57→31.81（b1）/ 54.78→87.45（b4）、scalar 19.61→34.34 / 40.24→68.67，与 baseline3 同 workload 逐档对照
  - 触发前提实测：`aiv_vec_bankgroup_cflt_ratio` 0.145–0.209（b4 最高），行步长 `d_padded*4` 为 2 的幂
  - 同族否决（同任务）：FMA 链（`r3c_fma`，vec ×4）与列分块 reduce（`r3t_slab2`，b4 514 µs）都更差 ⟹ 冲突只能从改变归约形态消除，而该形态改法在本工具链均负向
- repro: repro-missing（复现条件：同一 kernel 两分支仅差 reduce 操作数 buffer 的第二维宽度（`d_padded` vs `d_padded+8` fp32），`msprof op --kernel-name=main --launch-count=20` 对照 `aiv_vec_time` 与 Task Duration；断言只锁方向——padding 分支 `aiv_vec_time ≥ 1.5×` 基线）
- toolchain_stamp: tilelang 0.1.2+01805d7a07ca111e43868c07b496bffb30813ba3 / CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-ub-non-pow2-row-stride（置于 TRAP-reduce-mixed-dtype-silent-corrupt 之后；与 TRAP-UB-dst-align（dst 非零切片 + 32B 倍宽度）不同——本条是**操作数行宽**非对齐，症状是 vec 时间翻倍而非对齐错误码）：

    ---
    id: TRAP-ub-non-pow2-row-stride
    kind: trap
    family: [reduction, elementwise]
    apis: [T.reduce_sum, T.Parallel]
    dtype: [fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T145026Z（round 3）
    toolchain: tilelang 0.1.2+01805d7a / CANN 9.0.0 / Ascend910B3
    repro: repro-missing
    ---

    ### 用 padding 行宽「打散 UB bank group 冲突」会让向量时间翻倍

    现象：`(BK, d_padded)` 乘积 tile 上 `dim=0` 的 `hivm.vreduce` 有稳定的 bank-group 冲突（`aiv_vec_bankgroup_cflt_ratio` 0.145–0.209，b4 最高）；行步长 `d_padded*4` 是 2 的幂，跨行读取集中在少数 bank group。

    尝试：把 reduce 操作数行宽改为 `d_padded + 8` 个 fp32（+32 B），期望打散 bank group。
    **实测：整体劣化 60%+**（b1 28.36→48.37 µs，vec_time 17.57→31.81；b4 73.59→120.45，vec 54.78→87.45）——非 2 的幂/非 128 B 对齐的行宽会让原本连续的向量段被切成不对齐的碎段，代价远大于冲突收益。

    **判据**：Ascend 上 stride padding 不是修 bank 冲突的可用手段（那是 GPU shared-memory 的习惯做法）；本设备的向量指令要求行宽对齐连续，冲突只能从「改变归约形态」消除（本算子族实测 FMA 链与列分块 reduce 都更差，见 CASE-engram-decode-stage4）。诊断指纹：packing 后 `vec_time` 大幅上升而 `aiv_vec_total_cflt_ratio` 反而下降 ⟹ 典型「对齐被打碎」。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31（evolver 审计：optimizer 任务内回写 `traps-compiler.md` 的该条目无 repro、绕过 Tier 1 治理，经审计回退为提案）
- decided_by: -
- decided_note: -

## VP-2026-0150
- type: P
- title: 报性能上界用「可达包络」而非 max(pipe)：`duration ≈ vec + a·scalar`，pipeline 完好时 a∈[0.43,0.68]、流水被拆时 a≥0.9
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T145026Z 的 perf_opt/profiles/**/prof.json 逐档反解（evolver 蒸馏时机械复算 64 个 profile，含 round2/round3/baseline3/final3/probe）：engram_decode 主 kernel 的 44 档分布在 a∈[0.43,1.28]；pipeline 完好档 a∈[0.43,0.68]（baseline3 b4 0.47 / b1 0.55、final3 b1 0.68、r3b_bk32_n256 b1 0.68），流水被拆档 a≥0.9（r3a_eub b4 1.03、r3f_bk32inplace b4 1.28、r3n_bk64inplace 1.21、r3t_slab2 b4 0.97），同档 `aiv_mte2_wait_ratio` 0.44–0.83
  - 同任务 opt_log.md §2.2 P1/§4.2/§4.3（12 分支对比表 + 可达包络表：b1 16.19+0.5×15.49=23.94 µs→1.276%；tune 平均包络 3.784%）——**§2.2 原记「12 个分支一致 a∈[0.47,0.68]」与 §4.2「没有任何分支把 a 压到 0.6 以下」两句互斥且与逐档数据不符**（0.47<0.6 大量存在），蒸馏时据 prof.json 重新分段定界
  - 官方对照：本轮 final 达 tune 平均 3.4827%（本地口径）/ 3.4635%（官方口径），为可达包络 3.784% 的 92%
  - **数据点（2026-10-31，task engram_decode_mcore-engram_decode_mcore-20261031T012849Z，**不计独立确认**——同设备同工具链、不同 kernel 结构〔两段 split-d〕，只能确认加性形式、无法复现分段判据〔该任务 prof.json 无 `aiv_mte2_wait_ratio`，只有利用率〕）**：12 个段-档三元组（dur/vec/scalar）给出 `dur ≈ FLOOR(n) + vec + a·scalar`，**FLOOR 项必须显式保留**——否则短段（B 段 3.3–4.8µs）的 a 会被发射地板吸收而虚高。实算 b1/b4/b8：A 段 a = 1.19/0.48/0.56、B 段 a = 1.60/0.98/1.40（FLOOR 取 1.60〔n=8〕/2.52〔n=32〕）；即 b4/b8 的 A 段落在「pipeline 完好」带内，b1 的 A 段（mte2 利用率 0.73）与全部 B 段（发射主导）落在 a≥0.9 带。**结论**：加性包络形式跨结构存活且需补 FLOOR 项；分段判据（a<0.75 vs >0.85）在本任务无法验证（缺 wait_ratio），提案合入时建议把判据的输入条件写清（须 `aiv_mte2_wait_ratio` 可用）。
- repro: repro-missing（复现条件：`perf_opt/parse_prof.py` 对 `profiles/*/prof.json` 的 `duration_us` / `aiv_vec_time(us)` / `aiv_scalar_time(us)` 逐档算 a=(duration−vec)/scalar，按 `aiv_mte2_wait_ratio` 分段；断言只锁分段关系——a(pipeline 完好) < 0.75 且 a(流水被拆) > 0.85）
- toolchain_stamp: tilelang 0.1.2+01805d7a07ca111e43868c07b496bffb30813ba3（部分档位为 +76e4550，逐档见 prof.json 所属轮次）/ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    add 条目 PL-scalar-vec-half-overlap（置于 layout.md 末尾，PL-1.17 之后）：

    ---
    id: PL-scalar-vec-half-overlap
    kind: pattern
    family: [gemv, reduction]
    apis: [T.copy, T.reduce_sum, T.Parallel]
    dtype: [fp16, bf16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T145026Z（round 3）
    toolchain: tilelang 0.1.2+01805d7a / CANN 9.0.0 / Ascend910B3
    repro: repro-missing
    ---

    ### 报性能上界用「可达包络」：`duration ≈ vec + a·scalar`，a 分段由 pipeline 完好度决定

    **优化见解**：给 Vector 类 kernel 报性能上界时，`max(vec_time, mte2_time, scalar_time)` 这个常用式子在本设备**系统性偏乐观**。逐档反解（64 个 profile，含 4 轮）：

    ```
    Task Duration ≈ vec_time + a * scalar_time
      pipeline 完好（aiv_mte2_wait_ratio < 0.05）: a ∈ [0.43, 0.68]
      流水被拆（mte2_wait 0.44–0.83）          : a ≥ 0.9（scalar 完全暴露）
    ```

    （b4 基线 54.78 + 0.47×40.24 = 73.7 vs 实测 73.59；b1 final 16.19 + 0.68×15.49 = 26.65 vs 实测 26.65）

    机制：AIV 的 scalar 发射期（DMA 指令发射 + 循环控制 + 向量指令发射）无法与 vec 执行完全重叠；`aiv_vec_wait_ratio` / `aiv_mte2_wait_ratio` 可低到 0.02，说明「等数据」不是瓶颈，**发射期才是**。因此：
    - 降低 mte2 而不同时降 scalar 的改动，duration 基本不动（4 次独立复现：mte2 −45%，duration ±3%）。
    - 让 duration → `max(pipe)` 需要 scalar → 0（DMA 条数与指令数压到 0），实测不可达；结构冻结下的可达包络是 `vec + ~0.5·scalar`。
    - 报上界时同时给「理想重叠上界（max(pipe)）」与「可达包络（vec+0.5·scalar）」，后者用于判定目标是否现实。

    **不确定点**：a 随循环体形态漂移（0.43–1.28），且「a≥0.9」同时是流水失效的诊断信号；该式用于估算与上界判定，不用于预测单点。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31（evolver 审计：optimizer 任务内以 `status: verified` 直落 layout.md，P 类首证未走 Tier 1；经审计回退为提案并修正 a 区间）
- decided_by: -
- decided_note: -

## VP-2026-0151
- type: P
- title: 编译期容量类失败的归因法——变体不变性检验：报错量不随实现变体变化 ⟹ 共享预算/编译器固定常量；随变体变化 ⟹ 该资源总量
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T145026Z 的 perf_opt/opt_log.md §2.2 P4 / §2.3（R3-B/R3-K/R3-R：三种省 UB 实现——原结构 / 死 buffer alias 回收 / f16 暂存——报错字节数**完全相同** `requires 1606656 bits while 1572864 bits available`）+ §6.1 第 2 条
  - 同任务二分证据：删掉 GEMV 之后的尾部整段后 BK=32/d=512 **可编译**（`_engram_decode_kernel_r3u_gemvonly.py`）；裸 mul+reduce 最小探针 (32,512)=16384 元素**可编译**（知识域 `repro/CONST-ub-budget-vs-pattern-cap.py`）
  - 反例成本：第 2 轮把同一失败记为「UB 总量不足」，第 3 轮为此消耗 2 个 alias 回收实验分支（R3-K/R3-R）且报错量零变化
- repro: repro/CONST-ub-budget-vs-pattern-cap.py（可运行部分：16384 元素裸 mul+reduce 可编译；归因两分法的人工判读步骤见该文件头部）
- toolchain_stamp: tilelang 0.1.2+01805d7a07ca111e43868c07b496bffb30813ba3 / CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add 小节「编译期容量类失败：变体不变性检验」（置于该文件「诊断步骤」族内，与既有 UB 预算判据小节并列）：

    ### 编译期容量类失败：变体不变性检验

    `ub overflow`（`requires N bits while M bits available`）一类编译期容量失败，在**完整 kernel** 上有两种常见归因：① 该 pattern/段自身超限；② 与 kernel 其他工作集共享预算。
    判别动作（两步，均只需编译不需运行）：
    1. **二分**：删掉与被怀疑段无关的区域后重编——不通过 ⟹ 该段自身超限；通过 ⟹ 共享预算交互。
    2. **变体不变性**：用 2–3 个省资源实现变体（死 buffer alias 回收 / 半精度暂存 / 原地操作）重编，比较报错字节数——**一字不变 ⟹ 可回收量 ≈0**（归因到共享预算或 compiler-fixed 常量），**随变体线性变化 ⟹ 归因到该资源总量**。
    3. **最小 pattern 反证**：把被怀疑的 pattern 单独抽成最小 kernel 编译——若最小形态可编译，则「pattern 级元素上限」这一归因被证伪。
    实测例（engram_decode round 3）：三步依次否证「8192 元素 pattern cap」、确认「共享 192 KiB 预算」、确定可用 tile = `8192 // d_padded`（见 constants.md `CONST-ub-budget-vs-pattern-cap`）。
    反例成本：未做变体不变性检验时，同一失败会被记为「总量不足」，从而反复投入无效的 buffer 回收分支（本算子第 2→3 轮）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0152
- type: P
- title: GEMV/归约类 staging tile 取 UB 预算允许的最大值（编译期常量 BK = TUNED_STAGE_ELEMS // d_padded + 整除回退）——d=256 时 DMA 条数减半，b1 −5.7%/b8 −6.4%
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T145026Z 的 perf_opt/opt_log.md §2.3 R3-B′/§2.6/§3（`r3b_bk32_n256` 26.751/26.771 vs base 28.361/28.611；合并后可编译 26.651/26.791）；perf_records.jsonl candidate + final 行逐项可对账（provenance，允许失效）
  - 同任务 profiles/round3/r3b_bk32_n256_b1/prof.json 对照 baseline3：scalar 19.61→15.51、mte2 18.95→14.36、vec 17.57→16.19、duration 28.36→26.75；**b4（d=512）tile 上限仍是 16，无收益**（73.59→73.53，噪声）
  - 参数面已穷举（§3 Autotune Log）：BK @ d=256 ∈ {16,32,64}→可行点 32（64 pipeline 崩 30.87）；BK @ d=512 ∈ {16,32}→只有 16（4 种实现均 UB overflow）；tile 形态三种只有整块可复用
- repro: repro-missing（慢→快关键更改：慢形态 `BK=16` 常量 vs 快形态 `BK = TUNED_STAGE_ELEMS // d_padded`（8192//d_padded）并保留 `d_mem % BK == 0` 的整除回退；效应须在完整 kernel 上以 `msprof op` 的 `aiv_scalar_time` + Task Duration 对照，断言锁方向（快形态 scalar_time ≤ 0.85× 慢形态））
- toolchain_stamp: tilelang 0.1.2+01805d7a07ca111e43868c07b496bffb30813ba3 / CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add 条目 PL-1.23-stage-tile-max-legal（置于 PL-1.22-row-dma-batch-merge 之后）：

    ---
    id: PL-1.23-stage-tile-max-legal
    kind: pattern
    family: [gemv, reduction]
    apis: [T.copy, T.reduce_sum, T.serial]
    dtype: [fp16, bf16]
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T145026Z（round 3）
    toolchain: tilelang 0.1.2+01805d7a / CANN 9.0.0 / Ascend910B3
    repro: repro-missing
    ---

    ### staging tile 取「UB 预算允许的最大值」，写成编译期常量而非 shape 分派

    **优化见解**：GEMV/归约类小 tile 的 `aiv_scalar_time` 与 DMA/Copy 语句**条数**近似成正比（单核 910B3 每指令 0.16 µs @76e4550 / 0.20–0.21 µs @01805d7a，见 CONST-mte2-degradation 910B3 段）。tile 元素数固定时，条数 ∝ 1/BK ⟹ 在不触发 UB overflow 的前提下取最大 BK 是最省 scalar 的单参数改动；写成 `BK = TUNED_STAGE_ELEMS // d_padded` 即把「元素预算」与「d 的形状」解耦，d 变化时自动落到预算上限（d=256→32、d=512→16）。

    **关键代码更改（慢→快）**：
    ```
    # 慢：固定 BK=16（d=256 时 96 条 W/e 拷贝，scalar 19.61 µs）
    BK = 16
    # 快：预算驱动的编译期常量（d=256 得 BK=32，48 条，scalar 15.51 µs，duration −5.7%）
    TUNED_STAGE_ELEMS = 8192
    BK = TUNED_STAGE_ELEMS // d_padded
    while d_mem % BK:            # 整除回退，保持在工厂内做（不进入 @T.prim_func）
        BK //= 2
    ```
    **实测**（910B3 / 01805d7a，engram_decode）：b1 28.361→26.751 µs（−5.7%）、b8 28.611→26.771（−6.4%）、b4 73.591→73.541（d=512 上限仍是 16，无收益）。**边界**：把 BK 再推到 64 时 scalar 确实降到 13.6 µs 但 pipeline 崩（30.87 µs，busy 0.47）⟹ 上限由 `CONST-ub-budget-vs-pattern-cap` 的共享预算而非「越大越好」决定；判据：改后 `aiv_mte2_wait_ratio` 仍 <0.05 即安全。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0153
- type: D
- title: 官方 TileOPs bench 的 Ratio 与 Stage 4 本地 msprof op 口径在小 workload 上系统性差 ~5%（b1 +5.0%/+5.4%，两任务同向；b4/b8 ≤0.2%）——跨口径比较改进幅度须同口径
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T145026Z：官方 `examples/TileOPs/reports/tileops/20261031_004711_153786_EngramDecodeOp/benchmark.md` 27.9806 / 73.6215 / 26.8005 µs（1.0945% / 3.8727% / 5.4232%，avg 3.4635%）vs 同任务 opt_log Final 表（本地 `--launch-count=20`）26.651 / 73.541 / 26.791 µs（avg 3.4827%）→ b1 +5.0%（+1.33 µs）、b4 +0.11%、b8 +0.04%
  - task TileOPs-_engram_decode_kernel-20261030T121905Z：官方 `20261030_220319_894369_EngramDecodeOp/benchmark.md` 29.9406 / 73.6615 / 28.6206 µs vs 本地 28.411 / 73.591 / 28.571 µs → b1 +5.4%（+1.53 µs）、b4 +0.10%、b8 +0.17%（两任务同向）
  - 两任务的官方 msprof 元数据均为 `sample_count: 1`、`kernel_name_resolved: main`；差异集中在最小的 b1（launch 占比最高档），b4（83% 时长）几乎无差 ⟹ 指向「小 kernel 的官方采样含额外固定开销」，但**归因未定**（未做同 kernel 双口径交错实测）
- repro: repro-missing（复现条件：同一最终 kernel 先后跑官方 `pytest benchmarks/ops/bench_*` 与 `msprof op --kernel-name=main --launch-count=20`，逐 workload 对照两个口径的 µs 与 Ratio；断言只锁方向——`official_us > local_us` 且相对差随 workload 时长单调递减）
- toolchain_stamp: 官方侧 `git_commit 01805d7a07ca111e43868c07b496bffb30813ba3` / tilelang 0.1.1.10+ubuntu.20.4.cann900（report 环境元数据）/ CANN 9.0.0 / Ascend910B3 / 2026-10-31；本地侧 tilelang 0.1.2+01805d7a / CANN 9.0.0
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-official-vs-local-ratio-gap（置于 TRAP-BENCH-CONFIG-CALIBRATION 之后，二者互链：前者是「配置口径」，本条是「采样口径」）：

    ---
    id: TRAP-official-vs-local-ratio-gap
    kind: trap
    family: [general, gemv, reduction]
    apis: [T.Kernel]
    dtype: [fp16, bf16]
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T121905Z / TileOPs-_engram_decode_kernel-20261030T145026Z
    toolchain: 官方侧 git 01805d7a + CANN 9.0.0 / 910B3；本地侧 tilelang 0.1.2+01805d7a / CANN 9.0.0
    repro: repro-missing
    ---

    ### 官方 bench Ratio 与本地 msprof op 口径在小 workload 上差 ~5%

    同一最终 kernel、同一设备：官方 `benchmark.md`（harness 内 msprof，`sample_count: 1`）与 Stage 4 本地 `msprof op --kernel-name=main --launch-count=20` 的 Task Duration 并不相同。

    | workload | 官方 µs / Ratio | 本地 µs / Ratio | 差 |
    |---|---:|---:|---:|
    | decode b1（d_mem512 d256） | 27.9806 / 1.0945% | 26.651 / 1.1460% | +5.0%（+1.33 µs） |
    | decode b4（d_mem1024 d512） | 73.6215 / 3.8727% | 73.541 / 3.8770% | +0.11% |
    | decode b8（d_mem512 d256 bf16） | 26.8005 / 5.4232% | 26.791 / 5.4250% | +0.04% |
    | （上一任务同向）b1 | 29.9406 / 1.0229% | 28.411 / 1.0773% | +5.4%（+1.53 µs） |

    **判据**：tune 平均 Ratio 两口径分别 3.4635%（官方）与 3.4827%（本地），量级一致；但**逐档**比较时小 workload（launch/teardown 占比高）的官方口径系统性偏高约 5%，大 workload <0.2%。因此「改进幅度」只在同一口径内比较；把本地 µs 的改进比例直接当官方 Ratio 的改进会低估约 1 个百分点（b1）。差异归因未定（候选：官方单次采样 + harness 侧 host 路径），引用时按「已实测的口径差」使用，不据此推断机制。
- status: pending
- confirmations: 2/2（两任务独立复现；D 类仍缺 repro，未升 Tier 0）
- created_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0156
- type: P
- title: Ratio 类硬目标的「路线无关上界」——`min over n≤40 [ FLOOR(n) + 总向量元素数/(n × 每核实测吞吐) ]`（带宽与末阶段全免费），单路线实测不得冒充全局界
- evidence:
  - task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 的 DESIGN.md §5.6.2 论证 4（公式 + n=8/16/32/40 四档表 + 40% 门槛比较 + 融合假设鲁棒性）；§0.4 第 1 条记录了该公式的来历——v1 把单路线（split-d，且其带宽被自身实测证明劣化 2.3×）A 段实测 16.28% 称为「任何实现的严格下界」，被 Stage 2 判为阻塞项
  - REVIEW.md「阻塞 1 独立复算」（检视方按同一公式独立算出 b1 3.078 / b4 8.902 / b8 5.929 µs → 9.92% / 32.04% / 24.51%，均值 22.16%）：公式可被第三方在不知道实现细节的情况下复算
  - 常数来源：空 kernel 发射地板 FLOOR(1/8/32/40)=1.570/1.600/2.520/2.850µs（P1 探针）+ 每核向量吞吐 22163/34652/34062 elem-op/µs（P2 探针反解）；两者均为本戳实测（provenance：任务工作区 history_version/design_probe_{floor,split_gemv}.py 与 probe_out/*/prof.json，允许失效）
  - 反证价值：同一公式还证伪了知识库 CG-2026-0018 的「补齐跨核归约 ≈39–41%」外推（偏高 1.8×，见该条目 2026-10-31 订正）；即该判据同时是「目标可行性」与「知识库外推可信度」的检查工具
  - **第三任务数据点（不计确认——形态为「参照实现上界」而非路线无关公式）**：task TileOPs-_gqa_bwd_kernel-20261031T051147Z 对 40% 硬目标先做两层上界再投预算：①目标换算（实测 fp16 峰值 270.1 TFLOPS ⇒ 理论 Ratio 上限 90.2%；40% = 持续 119.7 TFLOPS = 实测峰值的 44%，本仓 attention 族最好水平 70 TFLOPS / 23.6%）；②同算子最优参照上界（3-pass 单引擎相位拆分 1016.8 / 7518.6 µs ⇒ ≈8.2%）。结论「40% 不可达」与后续 2.90× 的正向收益并存——上界判据不阻止在可行方向取收益，只决定是否投入结构重做（该任务据此把预算限定在 Stage 4 内可解的两项）
- repro: repro-missing（复现条件：① 空 kernel 探针测 n=1/8/32/40 的 FLOOR；② 单段向量探针反解每核吞吐〔每块元素数 ÷ vec 时间〕；③ 按 `下界(n)=FLOOR(n)+总元素数/(n·吞吐)` 对 n 取最优，与 40%/目标门槛比较。断言只锁量级关系：上界 < 目标门槛，且任一单路线实测 A 段天花板 ≥ 该上界）
- toolchain_stamp: tilelang 0.1.2+3ed7c91420e925ff3ec388fb981dd6ebf45ff4bf + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-10-31
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    add 小节「Ratio 类硬目标的路线无关上界（R4 可行性判据）」：

    ### Ratio 类硬目标的路线无关上界

    **判据**：Ratio 目标（`Ratio = ideal / Duration`）的可达性用下式先判，再决定是否投入结构改造：

    ```
    下界(n) = FLOOR(n) + (总向量元素数 / (n × 每核实测吞吐))
    Ratio 上限 = ideal / min over n≤40 [ 下界(n) ]
    ```

    三项假设必须显式写出：① **带宽项与末阶段（apply/elementwise 尾段）全按免费**（乐观）；② 总向量元素数按本工具链的实际 pass 数计（`mul+reduce` 不融合 ⇒ 两遍；融合假设需另算）；③ n 取到物理核数为上限、且对每档取**最好**的实测吞吐（乐观）。得到的是**乐观**下界，实际只会更慢。

    **纪律（本判据的来历）**：单路线实测（哪怕它已是所有路线里最好的一条）只能称「该路线的观测值」，不能称「任何实现的严格界」——否则下一轮「换个分块轴」即可推翻。必须把**并行度 n 也纳入最小化**，否则界会被「再加核」一句反驳（本例检视方给的同形式值固定了各路线实测块数，得 20.0%；把 n 也最小化后为 22.2%）。

    **实测例（engram_decode 族，b1/b4/b8）**：总向量元素数 4·B·d_mem·d = 524288 / 8388608 / 4194304；吞吐 22163 / 34652 / 34062 elem-op/µs；FLOOR 1.570/1.600/2.520/2.850 µs ⇒ 下界 3.08 / 8.90 / 5.93 µs ⇒ 上限 **9.92% / 32.04% / 24.51%，均值 ≤22.2%**。对 40% 门槛（0.7635/7.128/3.634 µs）分别低 4.0× / 1.25× / 1.63×；即使假设融合成单遍，均值上限也只到 31.5%（b4 仍需 ≤5.33 < 5.88）。**结论**：该 workload 族 40% 不可达，与能力缺口是否补齐无关（补齐至多抹掉末阶段）。
- status: pending
- confirmations: 1/2
- created_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0157
- type: P
- title: 多段（多 launch）kernel 的 op 级 Ratio 记账：A+B 求和、每段独立进程采集、B 段发射地板是纯税（占 op 级 28–33%）
- evidence:
  - task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 两次独立采集（bench_out/、bench_out2/，`msprof op --kernel-name=main --launch-count=10 --warm-up=3`，每段单独进程）：A+B = 10.14/10.05、17.77/17.95、13.28/13.48 µs；若只报 A 段，b4 会显示 21.9% 而真实 op 级 15.9%（虚报 6pt）
  - 同任务 B 段 pipe 分解：vec 仅 0.37/0.37/0.34 µs，**scalar 0.85/1.92/1.37 µs**，mte2 1.14/1.02/0.99 µs ⇒ B 段 3.3–4.8 µs 大部分是发射地板（两段方案的固定税），且模型（FLOOR + 0.5×14×0.21=1.47）对 B 段偏乐观 8–19%
  - 仓内同类事故（借用案例，非本任务实测）：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/integration_log.md`——三段 prim_func 命名为 `stats/finalize/apply` 使官方 `--kernel-name` 过滤失配，msprof 回退无过滤并捕获全部 op，该记录被判 invalid（任务工作区，provenance 允许失效）
  - 采集链路依据：`benchmark_base.py::_parse_msprof_roofline` 在多 op 命中时取组内首元素（本任务检视方亲阅 L405-450）；两段同名 `main` 时单进程捕获会静默只取一段且不报警
  - **第三任务数据点（不计确认——为口径冲突的定性识别，未实测失真数值）**：task TileOPs-_gqa_bwd_kernel-20261031T051147Z 的 `perf_feedback.md`「不可移植论证 1」：同算子参照的 3-pass 形态把一次 op 调用拆成 3 个 pass × `n_chunks` 次发射（8b-long 16 chunk、workspace 预算 256 MB），而 TileOPs 官方 Ratio 按单 op 名捕获（`--kernel-name=_gqa_bwd_main_mix_aic`），「一次调用 = 多 kernel 时延之和」与单 kernel 捕获口径不一致 ⇒ 该结构在官方口径下无法记账。本条据此把「多 pass 化的记账前提」列为结构决策的准入项：先确认采集口径能表达 Σ段，再评估拆分收益
- repro: repro-missing（复现条件：两段 `prim_func` 同名 `main`，先单进程 `msprof op --kernel-name=main` 捕获（得到的是单段或混淆结果），再每段各起一个进程单独采集，比较 `Duration(A)+Duration(B)` 与单次捕获值。断言只锁方向：单次捕获值 < A+B，且据此算出的 Ratio 高于真实 op 级）
- toolchain_stamp: tilelang 0.1.2+3ed7c91420e925ff3c388fb981dd6ebf45ff4bf + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add 小节「多段（多 launch）kernel 的 Ratio 记账」（置于「诊断上下文模板」之前，与 Ratio 类硬目标判据相邻）：

    ### 多段（多 launch）kernel 的 Ratio 记账

    一个 op 拆成两次（或多次）kernel 发射时，`Ratio` 的口径必须是**全段求和**：

    1. **分母 = Σ Duration(段)**，不是任何单段的 Duration；**分子 = op 级 `eval_roofline()` ideal**（不是被捕获单条目的 ideal）。分母分段、分子整 op，才能与官方 report 对齐。
    2. **每段单独进程采集**：两段同名 `main` 时，`--kernel-name=main` 会一起命中（或命中其中一段），`_parse_msprof_roofline` 在多 op 命中时取组内首元素且不报警——单进程采集只能得到其中一段，Ratio 会虚高数个百分点。异名（`stats/finalize/apply`）则会让官方过滤完全失配、msprof 回退无过滤并捕获全部 op，官方记录判 invalid。两种命名都要求**一进程一段**。
    3. **分段上界预检**：额外的一段 = 多付一次发射地板。实测某两段方案 B 段 3.3–4.8 µs 中 vec 仅 0.34–0.37 µs、scalar 0.85–1.92 µs——这段税占 op 级的 28–33%，且设计期模型（`0.5×` 暴露系数）对它普遍偏乐观 8–19%。判据：分段收益（多核并行 / 消除 d 长 pass）必须大于这段地板税，否则整段并入更划算；报告必须给出「A/B 分段表 + 段数」而非单一 Ratio。
- status: pending
- confirmations: 1/2
- created_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0220
- type: D
- title: L0C 累加器容量判据是「≤ 32768 个 fp32」，等号可用（`2·BM·D + BN·D = 32768` 的 pass 编译通过）；3-pass 结构加宽 tile 的容量窄口实测落在纯 Vector pass 的 UB（228.5 KB vs 192 KB），不在 Cube 侧 L0C
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261107T122917Z 的 perf_opt/opt_log.md §21.0/§21.1/§21.6：BN=128 时 pass3 `_gqa_bwd_mul` 的 L0C = `2·BM·D + BN·D = 2·64·128 + 128·128 = 32768` 个 fp32（128 KB 的 100.0%、零余量），`bishengir-compile` rc=0（COMPILE_OK 43.8 s），`_gqa_bwd_mul_infer_workspace_shape_function() = 65536 bytes/block`（`logs/round21/probe_bn128_mul.log`）
  - 同一 BN=128 下 `sdw`（L0C 16384 = 50%、UB 160 KB = 83.3%）与 `mul` 均编译通过，唯 pass2 `_gqa_bwd_pds` 报 `ub overflow, requires 1871872 bits while 1572864 bits available`（1871872/8 = 233,984 B = 228.5 KB vs 196,608 B = 192 KB）
  - 同轮把 `pds` 改为「保持 64 行子块 + slot 内编译期展开 `n_sub = block_n//64`」后 rc=0（UB 回落到 140.5 KB static），L0 8/8 PASS，转正后 `--level all` EXIT=0 37 PASS 无 WARN/FAIL，8 档 final 无一回退（几何 −25.2%）
  - 与既有条目的关系：`CONST-capacity-910B2C` 已给 L0C = 128 KiB（`cc overflow` 反推），但未写等号边界；`VP-2026-0169` / `CONST-ub-alloc-live-range` 讲 UB 侧「静态加和 ≠ 分配器占用」。本条增量 = L0C 侧判据取 `≤`（100% 可用）+ 加宽的容量窄口定位在纯 Vector pass 的 UB
  - 口径核对：本条与 `CONST-capacity-910B2C` 的 L0C 段同一层级、同一取证方式（overflow 报文反推 + 编译探针），工具链戳一致（9cfae94c）；与 910B2C 历史条目无矛盾（设备/版本戳不同，各自成立）
- repro: repro-missing（复现条件 = 同一多 pass kernel 只把 pass3 的 BN 设到 `2·BM·D + BN·D = 32768` 个 fp32（D=128、BM=64 → BN=128）并单 pass 独立进程编译；断言只锁方向：pass3 rc=0 且 workspace 报文字节 < 128 KB，纯 Vector pass 报 `ub overflow` 且 `requires` bits > 1572864）
- toolchain_stamp: tilelang 0.1.2+9cfae94c + CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ msprof 25.5.0 / 2026-11-07
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    动作: update（`CONST-capacity-910B2C` 的 L0C 口径段末尾追加等号边界与窄口定位；不新开条目，避免同主题双条目）
    定位锚: "**L1 (cbuf) = 512KB/核**（`cbuf overflow, requires N bits while 4194304 bits available` 反推，2026-09-08）。双槽 k/v/p 的 bn 上限公式：`2·slots·bn·dim·2 + slots·bm·bn·2 + bm·dim·2 ≤ 512KB`（bm=44 时 bn≤427；wide 单槽模式 bn=512 可行，k+v+p+q=312KB）。"
    new 文本（锚句之后追加）:
    **〔2026-11-07 GQA-Bwd Round 21：L0C 判据取 `≤`，等号可用〕** 数据点：3-pass bwd 的 `mul` pass 在 BN=128、BM=64、D=128 时 L0C 占用 `2·BM·D + BN·D = 32768` 个 fp32（128 KB 的 100.0%、零余量），`bishengir-compile` rc=0（workspace `65536 bytes/block`）⇒ L0C 硬判据是「≤ 32768 个 fp32」，设计期按「留 25% 余量」做的加宽否决在本例被证伪。同结构下真正的窄口是**纯 Vector pass 的 UB**：同一 BN=128 的 `pds` 报 `ub overflow, requires 1871872 bits while 1572864 bits available`（228.5 KB vs 192 KB），而 L0C 75% 的 `sdw` 与 L0C 100% 的 `mul` 均编译通过 ⇒ 两个容量门互相独立、不可用其一推断另一（L0C 满配不影响 UB 判据，反之亦然）。判据：多 pass 结构里按 pass 独立编译，失败报文中的 `requires N bits` 即该 pass 的精确需求（秒级可获得），据此定位窄口所在 pass 与层级。
- status: pending
- confirmations: -/-
- created_by: task GQA-three-ops-20261108（bwd: TileOPs-_gqa_bwd_kernel-20261107T122917Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1（同 VP-2026-0168/0169/0142 先例：任务内 `perf_opt/_r21_compile_probe.py` 依赖候选 impl 与设备、非 ED-B 自包含形态，ED-C 机械转正不成立）。

## VP-2026-0221
- type: P
- title: capacity-partitioned widening——多 pass 图里只加宽容量可行的 pass，容量受限的 pass 保持原子块宽度并在 slot 内编译期展开子块（`n_sub = block_n//64`）；3-pass GQA bwd 实测 8 档 −21%~−29%（几何 −25.2%），收益全部来自迭代数减半
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261107T122917Z 的 perf_opt/opt_log.md §21.2/§21.3/§21.5：`sdw`/`mul`/workspace slot 用 BN=128；`pds` 保持 64 行子块（`q0 = i·128 + s·64`，`n_sub = block_n//64`），逐 score 元素的算术、tile 形状、元素顺序不变 ⇒ 数值与原形态同量级（8b-short fp16 `max_diff 6.305695e-03` vs BN=64 交付件 6.31e-03）
  - 逐 pass 归因（同一对 run 的 per-pass 中位之和）：`mul` −42.9%（short 577.43→329.91）/ −48.2%（long 4003.30→2074.14）；`sdw` −10.3%（260.07→233.40）/ −17.5%（1763.84→1455.47）；`pds` +0.9%（370.41→373.81）/ +2.2%（1830.42→1870.22，分区预期——它仍按 64 行子块跑，迭代数不变）
  - 元素总量 / workspace 字节 / dq 原子字节均不变（`ntile×BN×BM` 恒定）⇒ 收益只来自迭代数减半的每迭代固定成本；`mul` 每迭代 3 GEMM + 区域原子 + 握手链，故摊薄幅度最大
  - 8 档 final（`msprof op` Task Duration，3-pass 中位之和）：945.36 / 945.54 / 5411.35 / 5420.17 / 925.86 / 934.78 / 5459.31 / 5482.13 µs，几何 3021.51→**2259.39（−25.2%）**，无一回退；探索轮 3-run 交错 A/B 3/3 同向（−22.14% / −28.88%）；转正后 `--level all` 37 PASS
  - 与既有条目的关系：`PL-1.9-blockwidth` 给「块宽摊减律」（同 pass 内每块固定成本 ÷ 元素数），本条给多 pass 图上的**分区**形态——整图统一加宽会在容量受限 pass 的 UB 门失败（VP-2026-0220 数据点），分区是拿到同一量级收益的可编译形态；容量口径见 `CONST-ub-alloc-live-range`
- repro: repro-missing（效应只在完整多 pass kernel 规模显现，非 ED-B 可运行最小形态；复现条件 = 同一 kernel 两分支 A 各 pass 同宽、B 容量可行 pass 加宽 + 受限 pass 保持子块宽并 trace-time 展开 `n_sub`；断言只锁方向：B 的加宽 pass 时长 ≤0.7×A、受限 pass 在 ±3% 内、两分支精度同量级）
- toolchain_stamp: tilelang 0.1.2+9cfae94c + CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ msprof 25.5.0 / 2026-11-07
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目 `PL-1.25-capacity-partitioned-widening`（置于 `PL-1.24` 之后；attention 族多 pass 结构）：

    ---
    id: PL-1.25-capacity-partitioned-widening
    kind: pattern
    family: [attention, mixcv, cube]
    apis: [T.gemm, T.copy, T.atomic_add]
    dtype: [fp16, bf16]
    device: 910B3
    status: verified
    origin_task: TileOPs-_gqa_bwd_kernel-20261107T122917Z
    toolchain: tilelang 0.1.2+9cfae94c + CANN 9.0.0 / Ascend910B3 / 2026-11-07
    repro: repro-missing
    ---

    ### 1.25 capacity-partitioned widening（分 pass 容量分区加宽）

    **适用形态**：一个 op 由多个 pass（Cube-only / Vector-only / Mix）串成，墙钟由每迭代固定项（跨引擎握手、barrier、描述符链、原子发射）主导，设备容量却按 pass 分别受限。
    **做法**：只把容量可行的 pass 的块宽加倍（本例 `sdw`/`mul` 与 workspace slot 用 BN=128），容量受限的 pass 保持原子块宽（`pds` 仍 64 行）并在 slot 内 trace-time 展开子块。整图统一加宽会在受限 pass 的 UB 门失败（本例 `ub overflow 1871872 / 1572864 bits`），分区形态拿到同一量级收益且逐 score 元素表达式不变。
    **最小 delta 骨架**（ED-B，py_compile 级）：
    ```python
    TUNED_BN = 128                      # sdw / mul / ws slot
    # pds：保持 64 行子块，slot 内 trace-time 展开（不是 T.serial）
    n_sub = block_n // 64
    for s in range(n_sub):
        q0 = i * block_n + s * 64
        ...
    ```
    **优化见解（机制归因）**：加宽把 q 方向迭代数减半而元素总量、workspace 字节、原子字节全不变，因此收益 = 迭代数 × 每迭代固定成本；实测逐 pass 摊薄幅度与该 pass 每迭代的握手链厚度单调（`mul` −42.9%/−48.2% > `sdw` −10.3%/−17.5% > `pds` +0.9%/+2.2%，`pds` 迭代数未变故在噪声内）。该机制与 `PL-1.9-blockwidth` 的「块宽摊减律」同源，区别在于分区：容量门是 per-pass 的，故加宽点也按 pass 选。
    **实测**（8 档，msprof op Task Duration）：945.36 / 945.54 / 5411.35 / 5420.17 / 925.86 / 934.78 / 5459.31 / 5482.13 µs，几何 3021.51→2259.39（−25.2%），无一回退；精度 `--level all` 37 PASS，`max_diff` 与同形 BN=64 交付件同量级。
    **前置筛选**：块宽候选集受整除性与 per-pass 容量双重约束，见 queue VP-2026-0222。
- status: pending
- confirmations: 1/2
- created_by: task GQA-three-ops-20261108（bwd: TileOPs-_gqa_bwd_kernel-20261107T122917Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失（效应只在完整 3-pass kernel 规模显现，非 ED-B 可运行最小形态），按 merge-policy §1 入队待同族任务补 repro 或给出第二任务证据后合入。

## VP-2026-0222
- type: P
- title: tile 宽度的可评估集合 = `seq_len` 的因子 ∩ 容量可行集——工厂的 `TUNED_BN if S % TUNED_BN == 0 else gcd(S, TUNED_BN)` 回退会把「中间宽度」静默变成收窄（S∈{512,2048} 上 BN=96 退成 32），加宽实验前先做整除性筛选
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261107T122917Z 的 perf_opt/opt_log.md §21.0 附注：`_split_geom` 的 `block_n = TUNED_BN if seq_len % TUNED_BN == 0 else gcd(seq_len, TUNED_BN)`；8 个 tune workload 的 `seq_len ∈ {512,2048}`（2 的幂）⇒ BN=96 在两者上都回退成 gcd=32，BN=128 是唯一可评估的加宽点（512%128=0、2048%128=0）
  - 反证代价：若按「优先 BN=96 折中」的直觉派实验，得到的「BN=96 更快/无收益」实为 BN=32 的数据（收窄 50%），方向与标签相反且不可从结果反推
  - 同轮 tail/gcd 形状复现回退：S=96/160/224→`block_n=32`、320→64、384→128（L2 12/12 覆盖），说明回退是工厂层常态而非异常
  - 与既有条目的关系：`PL-1.4-tiling-heuristic` 的「回退分发」是 shape 兼容分支（C%16≠0 走非重排路径），不涉及宽度取 gcd；本条是实验设计前的可评估集合筛选
- repro: repro-missing（复现条件 = 任一带 `gcd` 回退的工厂，打印不同 `TUNED_BN` 下各 workload 的实际 `block_n`；断言方向：非因子宽度的 `block_n < TUNED_BN`）
- toolchain_stamp: tilelang 0.1.2+9cfae94c + CANN 9.0.0 / Ascend910B3 / 2026-11-07
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add 小节「块宽候选的可评估集合（整除性筛选先于容量预算）」（置于「诊断上下文模板」之前）：

    ### 块宽候选的可评估集合（整除性筛选先于容量预算）

    工厂按「整除则用 TUNED_X，否则 gcd 回退」分块时（如 `block_n = TUNED_BN if seq_len % TUNED_BN == 0 else gcd(seq_len, TUNED_BN)`），块宽的实际取值是 `gcd(seq_len, TUNED_X)`：**可评估的加宽集合 = seq_len 的因子 ∩ 容量可行集**，中间宽度会静默变成收窄。

    1. **先打印再派实验**：对每个 tune workload 的 `seq_len` 逐候选宽度算出实际 `block_n`，只保留 `block_n > 当前值` 的候选。实测例：`seq_len ∈ {512,2048}` 时 BN=96 → gcd=32（收窄 50%），BN=128 才等于候选值；tail 形状 S=96/160/224 → 32、320 → 64、384 → 128。
    2. **再按容量预算裁决**：整除性通过后，逐 pass 核对 L0C / UB（用编译探针的 `requires N bits` 报文取精确值），容量门是 per-pass 的（见 VP-2026-0221 的分区加宽）。
    3. **报告口径**：候选宽度必须写成「TUNED 值 → 实测 block_n」，不得只写 TUNED 值——两者在非因子形状上不同，只写前者会把收窄实验记成加宽实验。
- status: pending
- confirmations: 1/2
- created_by: task GQA-three-ops-20261108（bwd: TileOPs-_gqa_bwd_kernel-20261107T122917Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1。

## VP-2026-0223
- type: D
- title: msprof 采集/解析口径三则——`--aic-metrics=Default` 本就落盘 `PipeUtilization` 等 CSV（「无 pipe 列」是解析器只 walk `OpBasicInfo` 的 bug）、`aic_scalar_wait_time`/`aiv_scalar_wait_time` 聚合列不可当等待时间、pipe 表须按 `sub_block_id` 分组取 median
- evidence:
  - task TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z 的 perf_opt/opt_log.md §20.1/§20.6（推翻该算子 R19 的「msprof 盲区」结论）：R19 自己产出的 raw 目录 `profiles/round19/base_r19_8b-long_float16/OPPROF_*/_gqa_sw_fwd_main_mix_aic/6/` 里已有完整 `PipeUtilization_*.csv`（60 行 = 20 block × 3 sub_block：`cube0`/`vector0`/`vector1`）与 `ArithmeticUtilization` / `Memory` / `MemoryL0` / `MemoryUB` / `ResourceConflictRatio`；旧 `_parse_op_basic()` 只 walk `OpBasicInfo*.csv`，而 `aic_*/aiv_*` 只存在于 `PipeUtilization*.csv` ⇒ `g("aic_mte2_ratio")` 恒为 `None`
  - 显式 `--aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline` 复采与 R18b 记录差 ≤0.7%（317.33 vs 317.84 / 963.26 vs 962.62 / 310.40 vs 312.50），pipe 值与 R19 `Default` 目录重解析逐项一致（`aic_cube_ratio` 0.2361 vs 0.2367）⇒ 指标列表不改变时延口径
  - 聚合列陷阱：cube0 的 `aic_scalar_wait_time` 恒 `0.000000`，真实值只在 `aic_scalar_wait_id0..9_time`（逐 id 求和 744.3 µs ≈ `aic_scalar_time` 746.1 µs）；`aiv_scalar_wait_time` ≈ `aiv_vec_time`（short 159.21 vs 159.35、long 652.04 vs 637.71）⇒ 它是「scalar 等向量」= 向量忙，不计 idle/气泡
  - 分组口径：`PipeUtilization` 是每 block × sub_block 行，取 median 前须按 `sub_block_id` 分组（`aic_*` ← `cube0`、`aiv_*` ← `vector0`，`vector1` 用 `aiv_vec_ratio_v1` 做一致性核对）
  - 附带工具能力：`bench.py parse --cfg` 可对既有 raw 目录重解析（不重新 profiling）——R19 两条重解析得 965.39/965.13 µs，与 R19 `Task Duration` 一致
- repro: repro-missing（复现条件 = 同一 raw 目录用 `--aic-metrics=Default` 与显式列表各采一次，比对 `PipeUtilization*.csv` 是否存在、聚合 wait 列与 `*_id*` 列之和是否相等、`sub_block_id` 分组 median 与全表 median 的差）
- toolchain_stamp: tilelang 0.1.2+9cfae94c + CANN 9.0.0（msprof 25.5.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-11-07
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 条目 `CONST-msprof-column-caliber`（置于 `CONST-gm-to-ub-bw-dilution` 之后，同属采集口径类）：

    ---
    id: CONST-msprof-column-caliber
    kind: constant
    family: [general, profiling]
    apis: []
    dtype: []
    device: 910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z（推翻同算子 R19 的「msprof 盲区」结论）
    toolchain: tilelang 0.1.2+9cfae94c + CANN 9.0.0 / msprof 25.5.0 / Ascend910B3 / 2026-11-07
    repro: repro-missing
    ---

    ### msprof `op` 的列口径三则（CANN 9.0.0 / msprof 25.5.0）

    1. **`--aic-metrics=Default` 本就落盘 pipe 表**：`PipeUtilization_*.csv`（60 行 = 20 block × 3 sub_block：`cube0`/`vector0`/`vector1`）与 `ArithmeticUtilization` / `Memory*` / `ResourceConflictRatio` 都会产出。「msprof 无 pipe 列」的实测成因是解析器只 walk `OpBasicInfo*.csv`（`aic_*/aiv_*` 只存在于 `PipeUtilization*.csv`）——该情形在 raw 目录 `ls` 一次即可判定，与工具链限制无关。显式列表 `BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline` 与 `Default` 的时延口径差 ≤0.7%。
    2. **聚合 wait 列不是等待时间**：`aic_scalar_wait_time` 恒 `0.000000`，真实值在 `aic_scalar_wait_id0..9_time`（逐 id 求和 744.3 µs ≈ `aic_scalar_time` 746.1 µs）；`aiv_scalar_wait_time` ≈ `aiv_vec_time`（159.21 vs 159.35 / 652.04 vs 637.71），语义是「scalar 等向量」= **向量忙**，不可当 idle/气泡计。
    3. **按 `sub_block_id` 分组取 median**：表是每 block × sub_block 行，`aic_*` 取 `cube0`、`aiv_*` 取 `vector0`，`vector1` 单独核对（Mix kernel 两 AIV 子块相同还是各半是 PL-1.13 的判据）。
    4. 既有 raw 目录可离线重解析（`bench.py parse --cfg`），不必重采；R19 两条重解析 965.39/965.13 µs 与原 `Task Duration` 一致。
- status: pending
- confirmations: -/-
- created_by: task GQA-three-ops-20261108（SW fwd: TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1（采集类 repro 需 msprof 实跑与 raw 目录，非纯脚本可断言）。

## VP-2026-0224
- type: P
- title: 任务级墙钟两点拟合 `墙钟/核 = f·N_task + u·N_step` 把「每任务固定项」与「每 band-step 真实工作」分离——GQA-SW fwd 实测 f=8.117 µs/任务、u=2.852 µs/band-step，short 比 long 慢 1.65× 全部由 f÷平均 band 长度（1.50 vs 3.75）解释
- evidence:
  - task TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z 的 perf_opt/opt_log.md §20.2/§20.4 + `logs/round20_diag/model_fit.json`：8b-short `25.6·8.117 + 38.4·2.852 = 317.3`（实测 317.33）、8b-long `51.2·8.117 + 192·2.852 = 963.1`（实测 963.26）、独立校验点 70b-short 预测 317.3 vs 实测 310.40（−2.2%）
  - 每 band-step 的真实工作 short/long 相同：cube 1.204 vs 1.186、mte1 0.925 vs 0.908、fixpipe 0.924 vs 0.954（差 ≤2%）；mte2 2.082 vs 1.796 的 +16% 来自每任务 Q hoist 32 KB 摊到更少步数
  - 分解：short 每 band-step 墙钟 8.26 µs = 任务开销 5.41 + 真实工作 2.85；long 5.02 = 2.16 + 2.85 ⇒ short 的 65% 墙钟是每任务固定项（8.12/12.40），long 43%；官方 Ratio 14.06% vs 22.74%（1.62×）与 1.65× 一一对应
  - 四选项逐条裁决（每项带 pipe 证据）：(a) 任务级填充/排空成立（cube flag 等待占墙钟 91%/76%，且 `l=0` 的等待是 `l≥1` 的 4–5.6 倍）；(b) Q-hoist/epilogue 只 ≤2.4 µs/任务；(c)「G=2 fused gemm 在短 band 下利用率低」否决（每 band-step cube 时间 +1.5%，噪声内）；(d) 口径问题否决（显式 metrics 复采差 ≤0.7%）
  - 未定项如实标注：f 中 ≈2.4 µs 是向量侧 init/epilogue 工作、≈5.7 µs 是依赖气泡，精确分界需 `n_slots=3` 实测；结构杠杆（跨任务流水深度）的收益上界 = 5.7/12.4 ≈ 24%（short），现实预期 −5~12% short / 0~5% long；flag 预算判据 `3·L_max ≤ 15`（short 6、long 15，恰好可容，见 `CONST-flag-id-budget`）
  - 与既有条目的关系：`PL-1.9-blockwidth` 讲同一 pass 内每块固定成本 ÷ 元素数；本条把「每任务」与「每步」两级固定项分开报价，用于解释同结构不同 workload 的 Ratio 差并给结构杠杆定上界
- repro: repro-missing（复现条件 = 同一内核取 3 个以上平均 band 长度不同的 workload，各采 median-of-20 Task Duration，解两点线性方程后在第 3 个形状上校验；断言方向：拟合残差 <5%，且每步分项 cube/mte1/fixpipe 在长短档差 <3%）
- toolchain_stamp: tilelang 0.1.2+9cfae94c + CANN 9.0.0（msprof 25.5.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-11-07
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add 小节「任务级固定项 vs 每步工作（两点拟合）」（置于「诊断上下文模板」之前）：

    ### 任务级固定项 vs 每步工作（两点拟合）

    同结构 kernel 在不同 workload 上的 Ratio 差，先分清它来自「每任务固定项」还是「每步工作」：

    1. **拟合式**：`墙钟/核 = f·N_task + u·N_step`。取两个平均步数（`N_step/N_task`）差异大的 workload 解出 `f`、`u`，用第三个形状校验（本例 f=8.117 µs/任务、u=2.852 µs/band-step；校验点 70b-short 残差 −2.2%）。
    2. **判读**：把每步墙钟拆成「任务开销 / L + 真实工作」——short（L=1.50）8.26 = 5.41 + 2.85，long（L=3.75）5.02 = 2.16 + 2.85，两者慢 1.65× 全部由 `f÷L` 解释，且每步分项（cube/mte1/fixpipe）长短档差 ≤2% ⇒ 瓶颈是任务边界而非每步算术。
    3. **定位任务头气泡**：按 flag id 分组看 `aic_scalar_wait_id*`——`l=0` 的等待显著大于 `l≥1`（本例 4–5.6 倍）即任务头串行往返，不是负载不均（同轮每核时长极差 6.8%）。
    4. **杠杆上界**：可动的只有 f 中的依赖气泡部分（f − 向量侧 init/epilogue 工作量）；上界 = 气泡/每任务墙钟（本例 ≈24% short），并受 flag 预算 `n_slots·L_max ≤ 15` 约束。上界只是上界：同结构的每步 ping-pong 不会被跨任务深度消掉。
- status: pending
- confirmations: 1/2
- created_by: task GQA-three-ops-20261108（SW fwd: TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；与 VP-2026-0178（采集预算）/ VP-2026-0113（<5% A/B 协议）互补不重复。

## VP-2026-0226
- type: D
- title: 单次卡死不等于确定性挂死——同一 case 单独进程 53.8 s PASS；`bishengir-compile` 呈三态（成功/卡死/崩溃），且 `subprocess.run(timeout=)` 留下的编译孤儿会污染后续 case，使「批量卡死」被误判为该 shape 的确定性失败
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z（官方 report 补跑）的 `perf_opt/logs/round13_report/`：case 8（KV-cache + 非 causal + 双侧窗口，`q=[64,128]`、`k=[256,512]`、`wl=64`、`wr=64`、fp16）在批量跑中卡 26 min 并污染后续 case；同 case 单独进程 **53.8 s PASS**（`case8_standalone.log`）
  - 同批第二证（不同算子 task TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z §20.4）：未改 kernel 的新 shape 探针 `probe-mid` 使 `bishengir-compile` 进入重试循环 >900 s（观测到 msprof 树 16:58 龄 + 单个 `bishengir-compile` 子进程 1:42 龄），超时 kill 后仍留 4 个孤儿进程（`ps -eo pid,etime,cmd`）
  - 机制：`subprocess.run(timeout=)` 只杀直接子进程，`msprof op → msopprof → python launch → bishengir-compile` 四层树的下层成为孤儿并占住设备，下一 case 串行化/超时 ⇒ 后续失败是被污染，不是该 case 的性质
  - 判据：把一次卡死定性为「确定性不可编译」需两个前提——① 独立进程、`npu-smi` 确认设备空闲时复现；② 无残留 `bishengir-compile` 子进程。缺任一则结论只写「未判定」
  - 与既有条目的关系：VP-2026-0212（探针三态：compiles / compiles-but-device-faults / not-compiles）管编译后的运行期分档；VP-2026-0208（runner 卫生）管进程卫生；本条补「卡死的可判定性前提」
- repro: repro-missing（复现条件 = 同一 case 先批量跑出现卡死，再在确认设备空闲后独立进程单跑；断言方向：独立进程 PASS 且批量卡死前存在孤儿 `bishengir-compile` 进程）
- toolchain_stamp: tilelang 0.1.2+9cfae94c + CANN 9.0.0（msprof 25.5.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-11-07～08
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 `TRAP-bishengir-tri-state-hang`（置于 `TRAP-l1-4th-live-buffer-copy-segv` 之后）：

    ---
    id: TRAP-bishengir-tri-state-hang
    kind: trap
    family: [general, compiler]
    apis: []
    dtype: []
    device: 910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z（+ 第二证 TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z）
    toolchain: tilelang 0.1.2+9cfae94c + CANN 9.0.0 / msprof 25.5.0 / Ascend910B3 / 2026-11-07
    repro: repro-missing
    ---

    ### `bishengir-compile` 三态失败 + 孤儿进程污染

    **现象**：同一 shape 在一次批量执行中卡 >26 min，单独进程跑 53.8 s PASS；另一轮未改 kernel 的新 shape 探针使编译进入重试循环 >900 s，kill 后仍留 4 个 `bishengir-compile` 孤儿。编译期的失败形态是**三态**（成功 / 卡死 / 崩溃），不是二值。
    **机制**：`subprocess.run(timeout=)` 只杀直接子进程，`msprof op → msopprof → python launch → bishengir-compile` 四层树的下层变孤儿并占住设备；后续 case 串行化或超时，看起来像「批量卡死」。
    **判据**：把一次卡死定性为「确定性不可编译」需同时满足——① 独立进程 + `npu-smi` 确认空闲时复现；② `ps` 无残留 `bishengir-compile`。否则结论写「未判定」。清理用 `Popen(start_new_session=True)` + 超时 `os.killpg(os.getpgid(pid), SIGKILL)`，逐 case 重试只覆盖 timeout/signal/compiler-crash 三类。
    **代价**：26 min 卡死 + 后续 case 被污染 ≈ 半轮预算；单次卡死若被当结论会误杀整条路径。
- status: pending
- confirmations: -/-
- created_by: task GQA-three-ops-20261108（varlen: TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1（复现需真实设备上的挂死现场，不适合作知识域常驻脚本）。

## VP-2026-0227
- type: P
- title: 官方 report 的 per-case 进程隔离执行形态（一 pytest id 一进程 + 只对瞬时失败重试 + `--resume` 断点续跑 + 组杀）使按 shape 重编译的 kernel 也能完整产出 `run.json`——GQA-SW-Varlen R13 内核 26/26 correctness + 10/10 benchmark、status=passed
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z（无 kernel 改动，sha 保持 `6befe384…`）：`perf_opt/logs/round13_report/report_isolated_varlen.py`（`Popen(start_new_session=True)` + 超时 `os.killpg(SIGKILL)` + 逐 case 重试 + `--resume` + 逐 case junit）执行官方 CLI 等价流程，`report_isolated_resume.log` 末行 `[isolated-report] status=passed`，run dir `examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/`
  - 产物：correctness **26/26**（前 7 例缓存命中 skip、其余 rc=0；平均 max abs err 1.42e-04）、benchmark **10/10** rc=0、`run.json` 生成；8 档 Ratio 14.05/14.33（8b-short fp16/bf16）、24.73/24.58（8b-long）、15.50/15.54（70b-short）、25.15/25.05（70b-long），时延 314.83/308.91/1774.74/1787.26/570.73/569.89/3490.69/3508.15 µs
  - 反例对照：同算子 R13 用官方 CLI 一次跑到底时，correctness 因 G=2 按 shape 重编译 40 min 只完成 8/26 且无 `run.json`（queue VP-2026-0219 首证，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z）⇒ 隔离执行把「跑不完」变成「可跑完」，差异只在执行形态
  - 重试边界：只对 timeout/signal/compiler-crash 三类瞬时失败重试，PASS/FAIL 不重试（避免把确定性失败刷成通过）；`--resume` 以逐 case junit 判已完成项
  - 与既有条目的关系：VP-2026-0209（闭包取 `_compile` 的替代 driver）解决「采集窗口被 host 算子污染」；本条走官方 harness，解决「完整性与跨 case 污染」；VP-2026-0219 记时间预算与本条同一问题域、本条约是其后续解
- repro: repro-missing（脚本留在任务 `perf_opt/logs/round13_report/`，未进知识域；复现条件 = 任一按 shape 重编译的 kernel 用逐 case 独立进程跑官方 report，断言方向：`run.json` 生成 + 执行的正确性用例数等于收集数）
- toolchain_stamp: tilelang 0.1.2+9cfae94c（report 记录 pip `0.1.1.10+ubuntu.20.4.cann900`、repo commit 9cfae94c）+ CANN 9.0.0 / Ascend910B3 / 2026-11-08
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    add 小节「官方 report 的逐 case 隔离执行形态」（置于「采集有效性判定」段之后）：

    ### 官方 report 的逐 case 隔离执行形态

    按 shape 重编译的 kernel（varlen / per-shape 分派）在官方 `tileops.reporting.cli run` 中会有两类问题：用例耗时随重编译线性增长而跑不完；一个 case 挂死后污染后续 case。实测可交付形态是**逐 case 进程隔离**：

    1. 每个 pytest id 起独立进程（`Popen(..., start_new_session=True)`），超时 `os.killpg(os.getpgid(pid), SIGKILL)` —— 只杀直接子进程会留下 `bishengir-compile` 孤儿并占设备；
    2. 重试只覆盖 timeout / signal / compiler-crash 三类瞬时失败，PASS/FAIL 不重试；
    3. 用逐 case junit + `--resume` 断点续跑，中断后从第一个未完成 id 继续；
    4. 判定以 `run.json` 是否生成 + 正确性用例数是否等于收集数为准（`status=passed`）。

    实测效果（GQA-SW-Varlen R13 内核，sha `6befe384`）：26/26 correctness + 10/10 benchmark + `run.json`，8 档 Ratio 14.05–25.15%；同内核未隔离批量跑时 40 min 只到 8/26 且无 `run.json`。
- status: pending
- confirmations: 1/2
- created_by: task GQA-three-ops-20261108（varlen: TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；与 VP-2026-0219（同问题域，R 类提案）互为「问题记录 / 解法形态」，合入时互链。

### Tier 2（R 类，结构化 diff 提案，待人工批准后 mode=apply 执行）

## VP-2026-0008
- type: R
- title: op-design Phase M0 第一问增加 golden 计算域核实检查项（fp16 golden 常见 fp32 opmath，勿假设与源算子舍入路径一致）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Skill Flow Issues（设计/检视期隐含假设未被拦截，L0 实测才暴露）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/.task_timeline.jsonl（Stage 3 attempt-1 precision fail，2485s）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py
- repro: torch.lerp CPU fp16（fp32 opmath+单次舍回）vs 逐步 fp16 舍入三步链：N=2^24 差 2–3 ulp、违反率 ~0.125%（pattern-library §2 opmath 行）
- toolchain_stamp: torch 2.9.0+cpu；tilelang-mlir-dev dev root build（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（扩展一行）
    定位锚: "2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。**golden 计算域核实**：golden 若为 `torch.*` CPU 实现，须用数行最小实验核实其在目标 dtype 下的实际 opmath 域——fp16 常见 fp32 opmath + 单次舍回（torch.lerp 实证：NPU 原生域逐步舍入差 2–3 ulp、容差违反率 ~0.125%，pattern-library §2）；「忠实源算子原生 dtype 舍入」与「过 golden 容差」冲突时以 golden 为准（源仓测试 ref 即 golden），kernel 侧用 fp32 中转对齐。
    动机: lerp_tensor Stage 3 attempt-1 precision fail（2485s）源于设计期「golden 舍入 = 源算子原生舍入」隐含假设；REVIEW 维度 8 预警了 v-prefix 侧舍入语义缺口但未覆盖 golden 侧 opmath 差异——设计期一道 3 行实验即可消除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0009
- type: R
- title: design-review 维度 2 UB 预算检查项补多 dtype 混合字节复算口径（dtype 中转路径禁沿用单一 dtype N-buffer 口径）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 维度 2 勘误（DESIGN.md §4.5 注与表自相矛盾：bf16 误写 fp16 同款 24576，正确 196608/24=8192）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 2 Value Point Proposals
- repro: 含 vcast 升精度中转路径的 block_size 上界 = UB容量 / Σ(bytes_i × count_i)（bf16 中转 24B/elem→8192；fp16 中转 20B/elem→9830 保守 8192）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；A2/A3 UB 192KB
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（改一行）
    定位锚: "| UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      | UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB）；含 dtype 中转（vcast 升精度）路径须按 Σ(elem_bytes × buffer_count) 混合字节宽复算，不得沿用单一 dtype 的 N-buffer 口径（lerp_tensor 实证：bf16 中转 24B/elem 上界 8192，误沿 fp16 口径 24576 需 480KB UB） |
    动机: lerp_tensor DESIGN.md §4.5 出现口径混用且注表自相矛盾，检视维度 2 原口径只查总和未查混合字节宽——勘误后才发现；若未拦截，Stage 3 assert 与 Stage 4 block_size 扫描范围都会错。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0011
- type: R
- title: conductor optimize 场景预检 baseline 字段补测量口径标注：Stage 5 events（host 侧 wall）数据禁作 headroom 依据
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md（Stage 5 bench events 口径 1M fp16 182.83us「慢于 torch」vs torch msprof 15.22us）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/.stage_state.json user_requirement（任务框定被 events 基线误导：「中小 N 慢于 torch，派发开销主导」）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md Baseline P1（msprof kernel-only 1M 7.30us 快于 torch 2.0–2.3x，全档快于 torch 1.65–2.3x）
- repro: 同 workload 双口径对照：events（host wall，含派发）vs msprof op Task Duration（kernel-only）
- toolchain_stamp: torch 2.9.0+cpu / torch_npu 2.9.0.post2；msprof cann-8.5.0；tilelang 0.1.2+ed787bb（2026-09-07 build）
- target_doc: .opencode/agents/conductor-scenarios/optimize.md
- delta: |
    动作: update（扩展一行）
    定位锚: "- 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）。**baseline 口径标注**：来自 Stage 5 bench（`prof_mode=events`，host 侧 wall、含端到端派发开销）的数值仅趋势参考、禁止作为 headroom 依据——kernel 时延唯一口径为 `msprof op Task Duration`（pattern-library §2 已弃用 event 口径；两口径方向可相反：lerp_tensor 2026-09-07 events 判 1M 慢于 torch、msprof kernel-only 判快 2.0–2.3x）；透传给 optimizer 的 events 基线须附此标注；
    动机: lerp_tensor optimize 任务的 user_requirement 即被 Stage 5 events 基线误导框定（「中小 N 慢于 torch」实为端到端派发开销失真）；靠 optimize skill 强制 msprof 唯一口径才免于方向错误——调度层透传时即标注可根除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0012

- type: R
- title: integrator 第一步后补 lint/format 自检步骤（TileOPs rule set B 对集成包生效，B023 类闭包绑定违反须在 pytest 前修复并留档）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md History mod 1（B023 "Function definition does not bind loop variable" + format.sh ruff-format auto-join；幂等重跑 integrate_kernel.py 会覆盖修复）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_s5_attempt1.py（pre-fix 备份）
  - 同款再现（2026-09-10 蒸馏追加，不同任务）：examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md Issues 第 1 条 + op 级 RETROSPECTIVE.md Transferable 末条（ALIGNMENT/_align_up 胶水补丁被幂等重跑整文件覆盖——「重跑前先备份 history_version/，重跑后按 Debug history 复放补丁」）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：cd examples/TileOPs && bash format.sh --files tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py（pre-fix 报 B023）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；TileOPs pyproject rule set B；torch 2.9.0+cpu / torch_npu 2.9.0.post2
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（第一步 bullet 列表末尾追加一条）
    定位锚: "- 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。
  - 脚本完成后、第二步 pytest 前，对集成的 kernel 文件执行 `bash format.sh --files <integrated files>`（TileOPs rule set B 对集成包生效，conductor 产物目录不受其约束）并修复 lint 违反（如 B023 循环变量闭包绑定用 default-arg 绑定：`def _to_npu(t, _dt=torch_dtype): ...`）；format/lint 修改记入 `integration_log.md` History（mod 条目）并在 `history_version/` 留 pre-fix 备份——幂等重跑 integrate_kernel.py 会整文件覆盖这些修复，重跑前须先备份；若修改触及 pytest 不覆盖的内嵌 L0 测试代码，须单独复跑内嵌 L0。
    动机: lerp_tensor 集成后 format.sh 才暴露 1 处 B023（内嵌测试闭包），且 format 自动改动使集成副本与 conductor 产物出现双向分歧；pytest 不覆盖内嵌 L0 代码，须单独复跑。前置自检把该类修改收敛为标准流程步骤。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0019
- type: R
- title: gate 1 S1-PLACEHOLDER 模板变量正则对中文数学集合记法误判——含分隔符的 {零, 有限真实数据} 类花括号集合被判「疑似模板变量未替换」造成 Stage 1 无效重试
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 两次 runtime fail：2026-09-07T12:44:47Z〔2960s〕与 13:47:06Z〔1518s〕，stage_retry_count 1→2）
  - .agents/tools/gate_lint.py#L36（TEMPLATE_BRACE_RE 定义）+ #L208-215（S1-PLACEHOLDER 判定）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`TEMPLATE_BRACE_RE.finditer("槽位尾行状态恒 ∈ {零, 有限真实数据}")` 命中 `{零, 有限真实数据}` → fail；误判文本出现在被门禁拒绝的中间版本上（归档 design_v0/v1/DESIGN.md 已无中文花括号——修订后通过）；发生账户来自 conductor 终态钩子输入（复盘工件未覆盖该两次 fail 的 gate 侧原因，同时构成复盘缺口）
  - 〔2026-09-15 复发，task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z〕新形态两处误报（Stage 1 gate 修复重试的 2×S1-PLACEHOLDER）：`e^{块内分数 − m_cur}`（CJK 数学上标，无列表分隔符）与 `_{\text{Cube 部和，f16 物化回传}}`（LaTeX \text{中文} 下标）——**前者不受本提案正则排除**（负向前瞻仅排除 [,、;；∈≤≥]，无分隔符的 `^{中文 − var}` 形态仍命中），审批时建议扩大排除集（brace 组前缀为 `^`/`_` 或含 `\text{` 时跳过）；writer 侧排版规则另立 VP-2026-0066 互补（改写消除模式命中：括号形式 / 注解移出公式，语义零变化）；纯 ASCII 花括号（`e^{m_prev − m_cur}`）不触发
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}', '恒 ∈ {零, 有限真实数据}')])"
- toolchain_stamp: gate_lint.py 现行版本（statectl gate 1 机械核对层）；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换一行正则定义）
    定位锚: 'TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")'
    old 文本: |
      TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    new 文本: |
      # CJK-brace groups that look like unfilled template slots. Set/math
      # notation with CJK members and list separators (e.g. "{零, 有限真实数据}")
      # is legitimate prose, not a template variable: skip such groups.
      TEMPLATE_BRACE_RE = re.compile(r"\{(?![^{}\n]*[,、;；∈≤≥])[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    动机: 负向前瞻排除含列表分隔符/数学关系符的花括号组——已验证 `{零, 有限真实数据}`/`{保留、舍弃}` 不再命中，`{算子名称}`/`{如: xxx}` 类真模板变量仍命中；本任务 Stage 1 两次 runtime 重试中 gate 误判贡献了无效重试（设计文档的集合记法是合法内容）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0020
- type: R
- title: gate 1 S1-REF-PATHS 路径存在性核对未剥离 `#`/`::` 锚后缀与省略号——`DESIGN.md::修订记录`/`.py#L345` 类合法引用被判「引用的仓库路径不存在」
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（同 VP-2026-0019 的两次 Stage 1 runtime fail）
  - .agents/tools/gate_lint.py#L64-84（_ref_path_failures：REF_PATH_RE 的排除字符类不含 ASCII `:`/`#`，token 连同锚后缀一起做 os.path.exists 判定）+ #L42-44（REF_PATH_RE 定义）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`examples/.../DESIGN.md::修订记录` 与 `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345` 两个 token 均含锚后缀导致 exists 判 false；发生账户同 VP-2026-0019（conductor 终态钩子输入 + 复盘缺口）
  - **复发（2026-10-31，task engram_decode_mcore-engram_decode_mcore-20261031T012849Z）**：Stage 1 attempt 1（`2212s`，verdict/reason=`runtime`）由 `path.py::func` 形式的路径字面量触发 S1-REF-PATHS 误判；retry 后 154s 通过——与 status 侧「提案 pending 未 apply 期间复发」的既有形态一致（本提案自 2026-09-07 起已积压 4 起实证）。**复盘缺口披露**：该次 fail 的 gate 侧原因未进 `RETROSPECTIVE.md`（同 ⑧⑩⑮ 形态），证据来自 conductor 终态钩子输入 + timeline 的 fail/complete 时间序（`01:28:54 start → 02:05:46 fail 2212s → 02:08:21 complete 154s`）。**writer 侧规避（本次交付的实际做法）**：路径字面量一律不带 `::`/`#` 锚后缀，锚信息另起括号说明——语义零损失。
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'(?:docs|examples|testing|\.agents)/[^\s\`\x27\"<>()\[\]（）【】：，。；、！？]+', '见 examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md::修订记录')])"
- toolchain_stamp: gate_lint.py 现行版本；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换 token 清理两行）
    定位锚: |
      (于 _ref_path_failures 函数体内)
          token = m.group(0).rstrip(".,;:!?、。】")
    old 文本: |
          token = m.group(0).rstrip(".,;:!?、。】")
    new 文本: |
          # Strip anchor suffixes (path#heading / path::section) before the
          # existence check; elided paths (examples/foo/...) are not concrete.
          token = re.split(r"#|::", m.group(0), maxsplit=1)[0].rstrip(".,;:!?、。】")
          if token.endswith("...") or "/..." in token:
              continue
    动机: 设计文档以 `path#heading`/`path::section` 引用仓库工件是合法引证形态（本任务 DESIGN/REVIEW 多处使用）；现行正则把锚后缀并入 token 判不存在——已验证修复后 `DESIGN.md::修订记录` → `DESIGN.md`、`.py#L345` → `.py` 正确判定，尾部省略号路径 rstrip 后落到父目录存在性判定。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0023
- type: R
- title: DESIGN 伪代码单一权威源规则——伪代码与实现注/设计步骤矛盾时必须先消歧再冻结（单边权威声明不够）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（VP 行 4：v0 §3.3 伪代码 α 行 vsub(ub_m, ub_mcur) 与实现注①「快照式 m_prev」矛盾——修复 = 伪代码改写为注的形态而非加「以注为准」标记）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Skill Flow Issues Design-internal 行：§3.3 伪代码 mask→softcap 与 E3/§3.1 softcap→mask 矛盾——照抄伪代码会破坏 P(OOB)=0 不变式，实现被迫自行消歧）
- repro: 复现条件——含伪代码 + 实现注双源且顺序/操作数不一致的设计文档（本任务 v0 两处实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表后、Phase 5 前插入附注）
    定位锚: |
      11. 交付清单

  ```
  ### Phase 5：质量自检
  ```
  old 文本: |
      11. 交付清单

  ```
### Phase 5：质量自检
```
  new 文本: |
      11. 交付清单

      > **伪代码单一权威源**：§3 伪代码即权威实现形态，实现注只解释不纠偏——两者矛盾时修正伪代码本身（改写为注的形态），而非加「以注为准」标记；伪代码与其他章节（设计步骤/论证）多源矛盾时必须先消歧再冻结——单边权威声明不够（2026-09-07 attention expert 任务实证：mask/softcap 顺序矛盾若照抄伪代码会破坏 P(OOB)=0 不变式，softcap(−1e38)≈−softcap 非精确 0）。

```
### Phase 5：质量自检
```
  动机: 同一设计文档两处伪代码-注/步骤矛盾（α 行操作数顺序、mask/softcap 顺序），一处靠检视拦截、一处漏到 Stage 3 由实现者自行消歧——冻结前的消歧义务是缺口。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0024
- type: R
- title: tilelang-op-design Phase 6 修订纪律三条——多路线裁决三步法+正确性面第四维 / 公式修复「公式-论证-自述」三同步 / 概念删除与计数预算标注 grep 清零判据
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → design_v1.md → DESIGN.md（v0→v1→v2 修订链）+ DESIGN.md 修订记录（v1/v2 两节「为何不会再犯」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（路线裁决三步法 + 第四维 + grep 清零 + 三同步各 VP 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（三同步模板 + 计数/预算标注同步义务：「8 参」「6 输入」需带边界 pattern 防子串误命中）
  - 〔2026-09-20 第三任务证据 + 建议并入第 4 条，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕designer 修订头「问题编号 → 调整 → 为何不会再犯」三列表（DESIGN.md v1/v2 两轮均合格）使复检可逐项对账——第三轮复检据此对六步修复逐项核实；建议审批时把「修订头三列表」并入本提案 Phase 6 修订纪律（新增第 4 条：修订版头部必须带该表，防复检对账退化）
- repro: 复现条件——任一 Stage 2 不通过后的针对性修订（本任务 2 轮修订、牵连面 10+ 章节实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境同上
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 6 小节扩展）
    定位锚: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  old 文本: |
      ### Phase 6：针对性修订

  ```
仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  new 文本: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

  检视不通过后的针对性修订（revision mode）附加纪律（2026-09-07 attention expert 任务 v0→v1→v2 修订链实证）：

1. **多路线修复必须写明路线裁决**：检视给出多条修复路线时，修订版写明裁决依据与回退设计，不默认选改动面小的路线。裁决步骤：① 证据分级（文档规格 > 前序设计断言 > 形态缺口——缺口降级为「未实证假设」而非「不支持」）；② 结构收益量化（如直连使 dtype 同构、删 flag/ws 族）；③ 回退完整性（被否决路线降级为 R-x 回退而非删除）；④ 正确性面作为独立第四维（flag 族数、未初始化域、握手竞态数随路线的削减）。
2. **公式修复的「公式-论证-自述」三同步**：修正分界/阈值/取整公式（ceildiv↔floordiv 类）时同步恢复依赖该公式的整条下游论证链（保守性论证 / 收益前提 / 量化自述占比）——只替换公式 token 而不同步论证会保留旧矛盾或制造新矛盾；修复前执行代数重推 + 整除/不整除双数值例。
3. **概念删除与计数/预算标注的完成判据 = token grep 清零**：删除某概念（API/buffer/flag/参数）或切换预算基准后，以旧 token（含「N 参」「N 输入」「NKB」类标注）全文 grep 清零为完成判据（子串误命中用带边界符号的精确 pattern；合法语境白名单 = 回退方案/修订记录/历史版引用）。
```
  动机: 本任务两轮修订中：路线裁决无模板（自行三步法完成）、v1 删参未同步计数（第 2 轮问题 2/3）、K_A 修复若无三同步会留下 §5.4 占比与 M5 前提的内部矛盾——三条纪律均已被 3 轮检视逐项验证有效。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0025
- type: R
- title: 弃选论证证据规则补第 6 条「前序实证的证据边界」——归档工件「N/M 实证通过」只证明实测路径可行，不得外延为替代路径不可行
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（问题 2：上一轮 23/23 载体链实证被用作「直连 bf16 不可行」的证据；负向断言「T.gemm bf16 ×」两轮原样继承且引用同一文档——两轮均未打开文档核对表格）
  - docs/Tilelang.language/线性代数操作/T.gemm.md#§2.2.1/§2.3（实际 bf16 √ + 实测可用——断言被推翻的对照证据）
  - examples/multi_head_attention/_prev_task_20260904_developer/_gqa_prefill_fwd_kernel/debug_log.md（前序实证的原始出处——provenance 交叉核对）
- repro: pytest testing/npuir/bf16_support_ops/test_gemm_bf16.py（被误读为「bf16 不可行」的反例证据链复核）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07 两轮设计 + 本轮复核）
- target_doc: .agents/skills/_shared/standards/negative-claim-evidence.md
- delta: |
    动作: update（§2 规则要点追加第 6 条）
    定位锚: "，与实测矛盾即 fail。"
    old 文本: |
      ，与实测矛盾即 fail。
    new 文本: |
      ，与实测矛盾即 fail。
      6. **前序实证的证据边界**：归档工件/前序任务的「N/M 实证通过」只证明其实测过的实现路径可行，不得外延为替代路径不可行——引用前序结论做负向断言时降级为「该路径已验证」并为替代路径另行举证（文档表格/测试/先例）；从前序任务继承的负向断言同样必须亲自打开所引文档复核到具体表格行/限制条款（「引用了正确路径、转述了相反内容」= 两轮设计同款失效形态，2026-09-07 attention expert 任务 bf16 断言实证：T.gemm.md §2.2.1 实为 bf16 √，前序断言驱动了整条载体链重设计后被推翻）。
    动机: bf16 负向断言跨两轮任务存活并驱动 E5 全链设计（载体链 +32KB UB、2 个 flag 族、Vector 载体负载），本轮文档+测试亲核推翻后直连形态 bf16/fp16 延迟差 ≤0.2%（pattern-library §1.8）——继承断言不亲核的代价是整条设计路线。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0026
- type: R
- title: design-review 维度 8 补「公式与参数化标量验证」强制块——⟺ 分界声明代数重推+双数值例 / 量化自述对账 / per-lane 双端点代入 / 继承负向断言亲核
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 2 轮问题 1：K_A ceildiv 假等价在 v0/v1 两轮存活——`块 n 全有效 ⟺ n < ceildiv(a, bn)` 的 +1/−1 补偿直觉使公式「看起来对」；8 个 bx 数值枚举 design−precise=1、免 mask 域违例 2016 位置/任务）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（VP 行：自述即测试向量——§5.4「每任务 1–2 块走 mask」与公式联立 NK−K_A∈[1,2] 直接暴露公式错误）+ #Stage-2-r1（per-lane 双端点行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md#L221（同款公式两轮存活证据）
- repro: 复现条件——任一含「块级分界 + 条件跳过」的 DESIGN 检视（causal K_A / sliding window / 守卫域判定）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07，3 轮检视）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 引言 blockquote 追加一段）
    定位锚: "不得只复算数字而继承设计的前提。"
    old 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。
    new 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。**公式与参数化标量验证（强制）**：① 一切「⟺」分界/阈值/取整类声明（causal 分界、sliding window 边界、守卫域判定）执行代数重推 + 整除/不整除双数值例（a=65/bn=64/n=1 类反例）——ceildiv↔floordiv 差 1 的补偿直觉使公式「看起来对」，文字论证无法发现（K_A 反例两轮存活）；② 设计文档内的量化自述（块数/占比/频次）是与公式对账的不变式，自述-公式联立矛盾即线索；③ AIV/lane 半分类的参数化标量阈值代入两个 lane 端点数值例（vid=0/vid=1——单 AIV 思维推导在另一 lane 符号方向隐蔽出错）；④ 从前序任务/归档工件继承的负向断言必须亲自打开所引文档复核到表格单元格——前序实证只对其实测过的路径有效。
    动机: 本任务 3 轮检视中 K_A 假等价公式在第 1 轮维度 0/8 下 pass（文字论证自洽）、per-lane 符号错误靠 smoke 数值例才发现——四项核对均为「文字论证全对、公式错」形态的最低成本拦截器。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0027
- type: R
- title: 检视报告修改建议中的替代公式/参数写入前必须先经独立数值验证——检视建议是 Stage 1 revision 的逐字输入，建议错误的修复成本高于漏检
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1 修改建议③把 ceildiv 版 K_A 误背书为「保守下界」；第 2 轮问题 1 代数重推 + 反例 c=64/bn=64/n=1 + 8 个 bx 数值枚举确证为假等价——ceildiv 对 vid=0 是激进上界）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（v1 曾把建议③逐字固化进「K_A 保守性说明（防 Stage 3 误改）」——错误由「待核公式」升级为「已论证事实」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（Skill Flow Issues 首行 + Transferable「检视者对自己前轮输出的建议负有验证义务」）
- repro: 复现条件——检视报告给出具体替代公式/取整方向且被修订版逐字采纳（本任务 v1 实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 4 追加一段）
    定位锚: "按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。"
    old 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。
    new 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。

  ```
  修改建议中给出的具体替代公式/参数/取整方向，写入前先经独立数值验证（代数重推或数值例）——检视建议是 Stage 1 revision 的逐字输入，建议自身携带错误会被修订版固化升级为「已论证事实」（2026-09-07 attention expert 任务实证：第 1 轮建议③把 ceildiv 版 K_A 误背书为「保守下界」，v1 逐字固化，第 2 轮代数重推 + 8 个 bx 数值枚举才确证为假等价）；检视者对自己前轮输出的建议负有与设计同等的验证义务——重审时必须重推建议本身的正确性，「建议被执行了」与「建议的目标达成了」是两回事。
```
  动机: 建议③误背书使错误公式获得检视权威加持、跨 v0/v1 两版存活且被「防误改」说明保护——该形态比漏检更难被发现，写入前验证是唯一低成本拦截位。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0028
- type: R
- title: design-review 修订版重审工作流——复用白名单/公式重推黑名单 + diff 锁定深查范围 + 新引入 API 亲验 + 建议原文逐句验收清账 + 三态判定
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮：v1→v2 diff 仅 8 处与修订记录声明一致；「复用 R2」标注 + K_A 全链重推策略；问题 4 三态判定「核心诉求成立 + 残留建议级」实践）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2/r3（复用白名单/重推黑名单行、diff 全量比对法、不变量对账清单、API 亲验行、原建议文本即 requirements 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v1.md（diff 比对基准）
  - 〔2026-09-17 第二证（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕RETROSPECTIVE.md re_review 章节——section-diff 复用形态独立复现：v0↔v1 按章节字节 diff 得 10 节 SAME / 9 节 CHG，SAME 节沿用首轮结论（同时证明修订无附带损伤）、CHG 节全量复核；**「修订残留只能靠全文 grep 抓、changelog 不可信作覆盖面依据」再证实**：v1 changelog 自称清理 6 处残留，复检仍独立发现 7 处未同步（§5.3/§9.1/§9.2 旧 UB 数字与 v1 直接矛盾等）——补充机械步骤见 VP-2026-0092（grep 关键词扫描 + 验证脚本 md5 比对）
  - 〔2026-09-20 第三任务独立复现，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕复检三段式：§0.1–§0.4 与 v0 逐字一致（diff 确认）→ 首轮源码行号级核对结论安全沿用；§0.5–§0.7 与 v1 逐字一致 → 上轮修复保留；变更半径审计（v1→v2 diff 28删/40增全落修复域，无越界改动/决策翻案）+ 决定性探针复跑（band 直载形态编译复证）——section-diff 复用 + diff 锁定深查范围第三任务实证；建议合入时把「变更半径审计 / 决定性探针复跑」两步骤并入本提案 new 文本的重审工作流
- repro: 复现条件——Stage 2 第 N 轮（N≥2）修订版重审（本任务 R2/R3 两轮实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 2 追加重审规则）
    定位锚: "按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。"
    old 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。
    new 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。

  ```
  修订版重审（第 N 轮，N ≥ 2）附加规则（2026-09-07 attention expert 任务 3 轮检视实证）：
- **复用白名单 / 重推黑名单**：前轮已验证的事实核对类结论（文档原文/源码行号/路径存在性/数字复算）可复用（标注「复用 R{N-1}」）；一切分界/阈值/取整公式与保守/激进的定性判断无论修订是否触碰均须重推——修订未触碰的段落不等于正确的段落（K_A 公式 v0/v1 两轮存活，唯一新阻塞恰在「复用区」）。
- **diff 锁定深查范围**：对「其余内容逐字保持上一版」类声明，以 `diff design_v{N-1}.md DESIGN.md` 全量比对验证——既验证声明属实（无未声明改动的隐性风险面），又自动锁定本轮深查范围（diff 命中处 = 必查）。
- **修订版新引入的 API 一律亲验存在性**（伪代码/映射表中首次出现者）——设计自述的「实查/亲验」记录不构成佐证，重跑同款 3 行验证才是检视证据。
- **修复验证对照建议原文逐句清账**：原建议文本是修复验收的 requirements 文档（每个验收子句独立判定达成/残留）；多目标建议部分达成时判「核心诉求成立 + 残留建议级」三态（记维度 warn + 修改方向），不整体放行也不整体阻塞。
```
  动机: 本任务 R2 漏检根因 = 复用策略未区分结论类型（事实核对 vs 推演判断）；R3 以 diff 锁定 + 公式全量重推 + 三态判定完成收口——规则均经实战验证。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0029
- type: R
- title: design-review 维度 5 新增「flag 协议安全性」检查行——per-slot flag 逐族传递性推演法（写 i+slots 前读 i 已完成）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（三族 + 单槽 + 跨任务五条传递性论证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：本轮对五条论证独立推演全部成立）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（VP 行 5：推演法可直接复用）
  - 〔2026-09-20 第二任务独立证据，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕令牌推演法双用实证：① v0 检视据此检出尾置 slot-free wait 的 off-by-one WAR 竞争（T=2 写 ws[0] 只受 prologue 前导令牌放行——set 序 × wait 序逐对配对 + 写序位置核对，REVIEW.md 问题 1）；② v1→v2 修复正确性据此纸面复核（无需运行）。wait 任务头前置结论与 GQA/旧 ssd verified 形态一致——位置规则的数据条目化提案见 VP-2026-0096（合入时互链）
- repro: 任一 per-slot flag + 多槽 workspace 的 Expert 设计检视
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 5 表追加一行）
    定位锚: "| 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |"
    old 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
    new 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
      | flag 协议安全性 | per-slot flag + 多槽 workspace 设计按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手（TASKDONE 类）单列核对（2026-09-07 attention expert 任务 §6.3 五条论证推演法） |
    动机: Expert 跨引擎流水的正确性核心在 flag 协议而非循环结构——维度 5 原三行（循环/同步模式/尾块）不覆盖 flag 族传递性，本轮靠检视者自发推演完成，方法已验证可复用。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0030
- type: R
- title: design-review 维度 7 新增「统计行对账」检查行——四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮维度 7 warn：统计行分项归类与表体处置列标签 2 处错位——表体字面分布 6/7/6/4 vs 统计 5/8/5/5，跨 v0/v1/v2 三版存活）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r3（Skill Flow Issues 首行：前三轮检视只数总数 23=23，未做分项-标签逐行比对）
- repro: 复现条件——含四态/多态分类统计行的设计文档（本任务 §0.5 处置表三版存活实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 7 表追加一行）
    定位锚: "| 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |"
    old 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
    new 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
      | 统计行对账 | 四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）——总数对齐但分项归类错位（6/7/6/4 vs 5/8/5/5）在 v0/v1/v2 三版存活，仅靠总数对账无法发现（2026-09-07 attention expert 任务实证） |
    动机: 总数对账只能发现双计类错误；分项归类与标签的错位需逐行比对——低成本高收益动作，防止错误统计被后续章节继承。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0031
- type: R
- title: conductor 修订调度 prompt 附加仲裁规则——检视建议跨轮冲突时显式标注「以最新轮为准 + 禁止恢复项」；design_error_summary 位置冲突以 REVIEW.md 为准
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（「以第 2 轮报告的 floordiv 公式为准，严禁按第一轮检视建议③的字面恢复」——本轮 conductor 调度输入显式指令防止了修订者回退到已推翻建议）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（Skill Flow Issues 两行：仲裁输入行 + design_error_summary 位置引用笔误行——「M5 收益前提（§1.6.2 #5）」实位于 §1.6.1）
- repro: 复现条件——Stage 2 多轮检视且后轮推翻前轮建议（本任务 v0→v1→v2 实证）
- toolchain_stamp: 流程规则（conductor 调度层）；证据环境 2026-09-07
- target_doc: .opencode/agents/tilelang-op-conductor.md
- delta: |
    动作: update（设计修订循环步骤 3 追加仲裁规则）
    定位锚: "3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。"
    old 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。
    new 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。修订调度 prompt 附加仲裁规则：① 前轮检视建议被后续轮次推翻时，显式标注「以最新轮为准 + 禁止恢复项」（如「严禁按第 1 轮建议③字面恢复」）——无此指令时修订者可能因「上一轮建议已被执行过」而复用错误公式（2026-09-07 attention expert 任务实证：误背书的 ceildiv 建议曾逐字固化进 v1）；② `design_error_summary` 的位置引用与 REVIEW.md 位置列表冲突时以 REVIEW.md 为准（或摘要只引问题编号不重复位置）——摘要位置笔误会误导修订定位。
    动机: 检视建议跨轮冲突时修订者面对两个「检视权威」；显式仲裁输入是被实证有效的防回退机制（本任务 v2 修订记录明示该指令起了作用）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0032

- type: R
- title: integrate_kernel.py 补 stale perf_opt lineage 告警——换代重集成不清理上一任务 perf_opt/，占位符会静默解析到旧谱系调优 kernel（工厂签名兼容、测试不拦截）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md（§Bench observations 末条 + Files 清单：perf_opt/ 保留的 developer 谱系产物〔mtime 09-07 04:51，43KB〕早于 expert 基线 kernel〔09-07 17:22，52KB〕）
  - examples/multi_head_attention/RETROSPECTIVE.md#Stage-5（Skill Flow Issues 行 + Transferable 首条）
  - 〔2026-09-15 预警场景实际发生，task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z〕换代重集成（单遍 expert→两相位 v3）再次命中且以另一形态：integrate_kernel.py 幂等检查误判 already integrated（正则匹配到 r1-r3 调优轮注释掉的 baseline 行），wrapper 未重写、Sep-07 谱系 perf_opt import 静默保持激活（`w._gqa_prefill_fwd_kernel.__module__` 修复前解析到 perf_opt 旧谱系 kernel）；证据：integration_log.md Round 2 Integration steps Step 1（`wrapper_rewritten: false`）+ Step 2 人工胶水修复记录 + op 级 RETROSPECTIVE.md Stage 5 round-2 Skill Flow Issues 首行。互补缺陷另立 VP-2026-0064（幂等检查剥离注释行——修「不重写」侧；本条修「重写后不告警」侧），同 VP-2026-0055/0056 集群建议同批审批
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：换代重集成后 `ls -la tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/` 对比基线源 mtime——本任务实测旧谱系早 12.5 小时
- toolchain_stamp: tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0；TileOPs pyproject 环境
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（main 复制循环内追加谱系检查）
    定位锚: |
      (main() 的 sources 复制循环内)
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    old 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    new 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
          perf_dst = target_dir / "perf_opt" / src.name
          if perf_dst.exists() and perf_dst.stat().st_mtime < src.stat().st_mtime:
              print(
                  f"[warn] stale perf_opt lineage: {perf_dst} predates the "
                  f"baseline source {src}; it likely derives from a previous "
                  f"task lineage -- verify lineage (mtime / origin task) "
                  f"before activating the perf_opt import"
              )
    动机: 换代重集成（developer→expert）整文件覆盖基线但保留旧 perf_opt/；wrapper 的 perf_opt import 占位符切换后会解析到旧谱系 kernel——工厂签名兼容、smoke/full/bench 均不拦截，只有谱系核对能发现（本任务实测两谱系相差 12.5 小时与 9KB/640↔949 行）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0033
- type: R
- title: op_template.py 的 _run_case 加长签名传参规范注释——位序敏感 case 参数（≥4 个或多个同类整型）用 kwargs 或 case 元组，防 heads↔seq 写反
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Test-harness 行：6 参位序 (batch, heads, heads_kv, Sq, Skv, dim) 在 13 处手写调用中写反 2 类〔heads↔seq〕，1 次 SDPA GQA 崩溃 + 1 次 ValueError；全 MHA 位形静默通过、GQA 位形才炸）
  - /tmp/opencode/full_run2.log#L48（RuntimeError: The size of tensor a (1024) must match the size of tensor b (16)——位序写反的崩溃实录，session-local）
- repro: 复现条件——长位序 case 签名手写调用 ≥10 处（本任务 13 处写反 2 类实证）
- toolchain_stamp: 测试工程规则（无运行时依赖）；环境 torch_npu 2.x + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-develop/templates/op_template.py
- delta: |
    动作: update（_run_case 定义行前追加规范注释）
    定位锚: "def _run_case(M, N, dtype, tag):"
    old 文本: |
      def _run_case(M, N, dtype, tag):
    new 文本: |
      # Case-signature rule: when a kernel's case parameters grow beyond ~4
      # positional args, or contain several same-type integers (shape / dims /
      # head counts), pass them as kwargs (or wrap a case tuple) at every call
      # site -- transposed positional args (heads<->seq) pass silently on
      # all-MHA shapes and only crash on GQA/dispatch variants (2026-09-07
      # attention expert task: 2 of 13 hand-written calls transposed).
      def _run_case(M, N, dtype, tag):
    动机: 位序写反在 GQA 位形（heads≠heads_kv）才暴露，全 MHA 测试全绿掩盖错误——模板级规范使新算子测试套件从第一处调用就免于此类浪费。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0036
- type: R
- title: optimize SKILL.md Phase 2 步骤 7 补 blocked/invalid 分支归宿——perf_records.jsonl 仅收录完成验证与测量的分支（duration_us 恒为实测数、不得 null），结构性不可行证据归 opt_log invalid 行
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-3-注记（gate 4 窄修复记录：4 个 blocked-compile 分支〔bn=512 双槽 L1 溢出〕曾以 duration_us: null 写入 perf_records.jsonl，按契约移除、证据归 opt_log invalid_compile 行）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/.task_timeline.jsonl（Stage 4 attempt 1 fail 7546s〔gate S4-PERF-RECORDS-SCHEMA〕→ attempt 2 窄修复 complete 314s）
  - .agents/skills/_shared/standards/signal-registry.md#§5（字段契约：duration_us 类型 number；「每轮每个实验分支验证后追加一行」）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#终局验证-注记（2026-09-09 第二轮同主题实证〔task 20260909T033622Z，并入不新开 id〕：final causal 首行 3779.0 为手抄转录笔误，append-only 纪律下以更正行 3773.39 + opt_log 注记收尾——手抄数字是 null 之外第二类结构化记录失真源）
- repro: 复现条件——调优轮出现 blocked-compile/invalid 分支时的记录行为（本任务 Round 3：bn=512 双槽 4 分支全部编译失败无测量值）
- toolchain_stamp: gate_lint.py 现行 S4-PERF-RECORDS-SCHEMA 规则；发生环境 tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（步骤 7 扩展）
    定位锚: "7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。"
    old 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。
    new 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。**blocked-compile / invalid 分支不写入 perf_records.jsonl**——`duration_us` 恒为实测数值、不得为 null（signal-registry §5 契约，gate 4 `S4-PERF-RECORDS-SCHEMA` 强校验）；其结构性不可行证据（报错文本、L1/UB 溢出公式）记入 opt_log 当轮实验分支表的 invalid/blocked 行，证据不丢失。**行内数值一律从 msprof 提取输出直接复制（或管道）生成，禁止手抄**（2026-09-09 第二轮实证：final causal 首行 3779.0 转录笔误，靠 append-only 更正行 + opt_log 注记收尾）。
    动机: 契约的「验证后追加」语义未在写入指令处显式表达负面清单，optimizer 把 blocked 分支也「忠实记录」进结构化文件——2026-09-08 expert Stage 4 实证：4 个 bn=512 双槽 L1 溢出分支误记 null 致 gate 失败一次 attempt（7546s），窄修复又耗 314s；显式归宿规则可省整次重试。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0037
- type: R
- title: profile-collection.md msprof 输出目录权限规则从叶子目录扩展到父目录——group-writable 父目录下仅 chmod 700 输出目录采集仍被拒（8/8 baseline 首采失败实证）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective（BP-R1 proposal：本轮 8 个 baseline 采集全部因 group-writable 父目录失败一次后才修正，前置检查可省一轮）
  - .agents/skills/tilelang-op-optimize/references/profile-collection.md#L96（现行规则仅 chmod 输出目录自身，未覆盖父目录）
- repro: 复现条件——在 group-writable 父目录（如默认 umask 002 的共享 checkout）下新建 msprof 输出目录、仅对叶子目录 chmod 700 后执行 msprof op 采集
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（权限规则行扩展）
    定位锚: "新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。"
    old 文本: |
      新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。
    new 文本: |
      新建 `msprof op` 输出目录后，先确保该目录**及其父目录**（至少直接父目录）group/other 不可写，例如 `chmod 700 {output_dir}` 并同步检查父目录；msprof 的权限校验覆盖父目录——仅叶子目录 700 而父目录 group-writable 时采集仍被拒（2026-09-08 expert Stage 4 实证：8/8 baseline 首采失败，父目录收紧后一次通过）。
    动机: 现行规则只覆盖输出目录自身；本任务 8 个 baseline 采集全部因 group-writable **父目录**失败一次后才修正——预检规则覆盖父目录可直接省一轮重采。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: BP-R1 整合去重：optimizer 自提的「profile-collection.md 补权限预检」经查现行文件已有叶子目录规则（67db6f3 期引入），本任务证据表明其不充分——按 update 形态提案而非 add。

## VP-2026-0039
- type: R
- title: profile-collection.md 补「探针 kernel 测硬件上限」标准形态——复刻目标数据流多模式对照 + 编译报错反推容量 + 真实 kernel 反推双证
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-8（probe_kvbw 四模式〔single/shared/distinct/pp〕同为 ~110.8µs kernel-only → L1 端口 r+w 148–154GB/s/核；bm=64@bn=512 `cc overflow` 报错反推 L0C=128KB；fabric ≥1.73TB/s 聚合）
  - 同文件#Skill-Retrospective-第二轮 有效手法①（硬上限测绘先行——一轮内把设计空间不确定性收敛为可算术的复合地板，直接决定 DESIGN_LIMIT 判定的证据质量）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md（三硬上限构成 [DESIGN_LIMIT] 2.09x/2.27x 缺口归因的实测基础）
  - pattern-library §1.9（数据侧落点：L0C/L1 端口/发射定律/fabric 条目，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z）
- repro: 复刻目标 kernel 数据流（同结构装载/gemm 模式）的最小探针 + 编译期容量压限（`overflow, requires N bits while M bits available` 报错文本）+ 真实 kernel msprof 计数器反推（流量÷时间）三方对照
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（文件末尾追加一节）
    定位锚: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
  ```
  ```
  old 文本: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
```
  ```
    new 文本: |
      返回：

  ```
```text
  PERF_DATA_COLLECTED
```

---

## 6. 探针 kernel 测硬件上限（DESIGN_LIMIT 证据 / 设计空间测绘）

  适用：调优目标涉及硬天花板判定（设计层归因、[DESIGN_LIMIT] 反馈、结构方向投入前的可行性测绘）时，在迭代候选之外用探针 kernel 把硬件/编译器上限测成可算术的数字。标准形态（2026-09-09 attention expert 第二轮实证：一轮测绘直接支撑 DESIGN_LIMIT 判定）：

  1. **复刻目标数据流的最小探针**：与目标 kernel 同结构的装载/gemm/访问模式，多模式对照（single / shared / distinct / ping-pong 双槽）区分 fabric、端口、容量因子——四模式同耗时 ⇒ 端口读写串行（双槽不能解除），非 fabric 争用。
  2. **编译失败报错反推容量上限**：刻意压限触发 BishengIR `overflow, requires N bits while M bits available` 报错（L0C cc / L1 cbuf / UB），从报错文本反推硬容量与 bm/bn 联合约束式。
  3. **真实 kernel 计数器反推双证**：探针值须与真实 kernel 的 msprof 计数器反推值互证（如 L1 r+w 流量 ÷ (mte2+mte1) 时间 = 端口带宽）；单源探针结论降级为待证。
  4. **探针纪律**：跨引擎探针必须带 Cube→Vector flag 同步（缺同步的数值崩坏是探针自身竞态而非 API 缺陷，pattern-library §1.9）；探针 kernel 保留 ≥1 个输入 tensor（§2 零输入行）。
  5. **归宿**：探针结论回写 pattern-library §1（数据侧）；触及编译器能力缺口时登记 `.agents/evolution/capability-gaps.md`；探针 kernel 脚本与日志入 `perf_opt/logs/`（jsonl 不收录非本算子分支的探针行，证据存日志与 opt_log 当轮表）。
```
  动机: DESIGN_LIMIT 判定与结构方向裁决的证据质量取决于硬上限数字的可信度；本轮 round 8 以该形态一轮完成 L0C/L1 端口/fabric 三项测绘 + 复合地板算术，直接封死无效调参方向（bm 上调/深流水/两遍 softmax）并支撑 2.09x/2.27x 缺口归因——无此形态需多轮试错反推。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0040
- type: R
- title: iteration-diagnosis.md「已知迭代结论」补续调任务前置基线复核——上轮交付驱动完整性（手改漂移检测）+ 基线重采对账（偏差 ≤ 噪声阈值）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#第二轮-前置修复（上轮终版测试驱动被用户手改：run_fa_tuned 仅余 fa4096、main 绕过 --level 分派；恢复后 `--level all` 全过〔logs/round8/driver_restore_level_all.log〕——手改漂移重新验证 ✓）
  - 同文件#Round-8-Diagnosis-1（基线复核：4 case 重采 18.19/29.69/77.62/263.93µs，与上轮终值记录偏差 ≤0.4%——谱系/环境一致性验证后才开调）
- repro: 复现条件——任一以「上轮终值为基线」的续调 optimize 任务（用户在手改过的 perf_opt 交付物上发起第二轮）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「已知迭代结论」小节扩展）
    定位锚: |
      ### 已知迭代结论

  ```
每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
  ```
  ```
  old 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
```
  ```
  new 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
本轮 profile 相比上一轮 profile 哪些指标发生变化
```

**续调任务（基线 = 上轮终值）的额外前置**（2026-09-09 attention expert 第二轮实证）：开调前完成两项复核——① 上轮交付测试驱动完整性：确认 run/case 分派逻辑与上轮终版一致（用户可能手改交付物——实证：run_fa_tuned 仅余 1 case、main 绕过 --level 分派；恢复后 `--level all` 全过再开调）；② 基线重采对账：重采上轮终值 workload 与上轮记录对账，偏差 ≤ 噪声阈值（实证 4 case ≤0.4%）才算谱系/环境一致，超阈值先查谱系与环境再开调。
```
  动机: 续调任务若直接信任被手改的测试驱动或未复核的基线，后续所有测量与归因都建立在失真地基上；本轮前置修复 + 重采对账各耗数分钟，拦截的是整轮方向的偏移风险。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0041
- type: R
- title: iteration-diagnosis.md 补「参考实现对照（morph ladder）」标准定位法——结构移植后性能不达预期时在参考实现上逐级叠加契约增量二分定位（op 级无罪 ≠ 组合无罪）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective-第三轮（有效手法① morph 阶梯决定性 + skill 流程问题①：形态学排查消耗 4 个 no_gain 分支后才启动，应更早切换；流程问题②：第二轮 [DESIGN_LIMIT] 地板归因未做同 contract 最小增量对照——有则 -20% transpose 税第二轮即被发现）
  - 同文件#Round-11-Diagnosis（morph 14 点矩阵：MORPH=0 参考原版 95.8 → MORPH=1 4D BSHD 输入 96.1 无罪 → MORPH=2c +lse 254.5 爆炸 → write-only 97.2 无罪 → 单 op 全无罪（vlog2/vmul+vadd/死源 transpose/4 个平凡 vadd）→ 任意含「活跃源 transpose」组合全部 253–255——唯一毒化点定位）
  - pattern-library §2 活跃源 transpose 毒化行（组合阈值效应数据侧，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（负结论推翻闭环的同款方法论缺口）
- repro: 复现条件——任一「结构移植/重写后性能不达预期（>1.3x）且仓库内存在同族参考实现」的调优轮（本轮实证：v11 初版 250.68 vs 参考 95.8–99µs，逐字转录参考 body 仍 255.53——上下文级差异形态学排查完全无效）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 代码结构观察」小节末条之后、「### 已知迭代结论」标题之前插入新小节）
    定位锚: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 已知迭代结论
  ```
    old 文本: （即上述定位锚原文）
  new 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 参考实现对照（morph ladder）

  仓库内存在同族参考实现（`examples/` 同 API 结构先例）且本方 kernel 结构移植/重写后性能显著不达预期（>1.3x）时，在逐项排查 op 形态 / pass_configs / 装载写法（含逐字转录参考 body）均无差异后，**优先切换 morph 阶梯定位，不在形态学排查上继续消耗实验分支**（2026-09-09 attention expert 第三轮实证：4 个形态学分支 no_gain 后 morph 阶梯一轮定位毒化点；且 [DESIGN_LIMIT] 地板归因若做此对照，-20% transpose 毒化税在第二轮即被发现）：

1. 在参考实现上逐级叠加本方契约增量（输入布局 → 输出契约 → 逐 op），每级一个实测点，二分定位差异源——**单 op / 单开关的无罪结论不可外推为组合无罪**（编译级阈值效应：活跃源 transpose 单 op 无罪、死源无罪、4 个平凡 vadd 无罪，任意含活跃源 transpose 的组合全部 2.6x 劣化，pattern-library §2）。
2. 对照点分级：契约级（输入/输出布局）先于 op 级；write-only 探针（只写不算）区分「计算成本」与「数据流形态触发」。
3. 纪律：perf-only 探针（故意破坏输出只测性能）在记录中显式标注；morph/探针脚本与实测点矩阵记入 opt_log 当轮（perf_records.jsonl 不收录非本算子 dispatch 的探针行）；scratch 探针结论须镜像回 opt_log 才可作为证据引用。

### 已知迭代结论
```
  动机: 「逐字转录参考仍慢 2.6x」类上下文级差异只有 morph 阶梯能定位（形态学排查对该类完全无效）；组合阈值效应使 op 级排除法结构性漏检——两条均指向该方法应为标准动作而非末位手段。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0045
- type: R
- title: op-design Phase 2 第 5 步等价性验证补双判据——基线式自身数值退化（pad 校正消灾难/大中间量结合序）时「候选 vs 基线」单判据假性 FAIL；补「候选对 fp64 误差 ≤ 基线误差（候选不得更差）」判据
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（compare() 双准则实现 + 首跑输出 randn-N2 fp32 案例 crit=b：基线 fp64err 1.58e-4 vs 候选 1.9e-7——候选更接近精确数学却被判 FAIL，首跑 3/8 违反）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Skill Flow Issues 首行 + Value Point Proposals 第二行：补判据后 45 案例全过且不掩盖真实回归）
- repro: 复现条件——任一基线式含 pad 校正/大中间量结合序的等价性验证（双判据形态自包含于 delta）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（输出「cases passing via criterion (b)」行）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（第 5 步「对照对象」bullet 扩展）
    定位锚: "   - **对照对象**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **对照对象与判定双判据**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）。**基线式自身数值退化时（GPU pad 校正的 catastrophic cancellation、大中间量乘法结合序）单判据「候选 vs 基线」会产生假性 FAIL**——补充判据 (b)：候选对 fp64 精确数学的误差 ≤ 基线误差（候选不得更差）；判定 = (a) 候选 vs 基线容差内，或 (b) 成立；(b) 路径通过须在结果表标注角点与双侧 fp64err（ada_layer_norm 实证 2026-09-10：randn-N2 fp32 基线 fp64err 1.58e-4 vs 候选 1.9e-7，补判据后 45 案例全过且不掩盖真实回归）；
    动机: D-1 验证的对照口径只写了「候选式 vs 基线式（同 dtype 舍入路径）」，基线退化场景把更优的候选判 FAIL——首跑假性 FAIL 3/8 案例浪费一轮排查；双判据以「候选不得更差」守住回归底线。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0046
- type: R
- title: op-design Phase 4 内存规划章节补 auto-multi-buffer 膨胀预算标定规则——按结构类平台值（TRAP-UB 第三证）而非「resident × 系数」，无同构先例时显式标 estimate + Stage 3 compile-probe 验证步骤
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Skill Flow Issues 首行：DESIGN §4.5 按「驻留 × 1.7」估膨胀，§8.3 L0-1 据此选 fp32 bm=3 首跑即编译失败 requires 245,760B > 192KB，一轮编译探针才修正为 20 B/elem 律并降档 bm 3→2）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md（Iteration 2/3 探针：transit 20.01 / fp32 26.00 / nl=4 22.00 B/elem——平台值精测）
  - pattern-library traps-compiler.md TRAP-UB-multibuffer-inflation（第三证，2026-09-10 已合入）
- repro: compile-probe 复现条件：persistent serial（num_local_tasks≥2）+ 20×bm×N > 196608 → bishengir 报文 `ub overflow, requires <bits> while 1572864 bits available`（报文 requires bits 即精确实测 UB 需求）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表第 4 项扩展）
    定位锚: "4. 数据规格与内存规划"
    old 文本: |
      4. 数据规格与内存规划
    new 文本: |
      4. 数据规格与内存规划（UB 膨胀预算按结构类平台值标定：persistent serial auto-multi-buffer 下 transit ≈20 B/elem、fp32 ≈26 B/elem、nl≥4 或静态 extents +2（pattern-library TRAP-UB-multibuffer-inflation 第三证）——不以「resident × 系数」估算；无同构先例的结构显式标 estimate 并写明 Stage 3 compile-probe 验证步骤：`ub overflow` 报文 requires bits 即该配置精确实测 UB 需求，读数反推系数后修正工厂 guard 与默认 bm 表）
    动机: ada_layer_norm DESIGN 按 ×1.7 估算（当时无同构先例），Stage 3 首跑编译失败一轮（245,760B > 192KB）才被 compile-probe 修正为 20 B/elem 律——设计期标定规则 + 探针验证步骤可把该修复前移或至少把验证步骤预置于设计中。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0047
- type: R
- title: design_calc_check.py 补两类表格形态解析——「按 N 分行的驻留预算表」（§4.5）与「workload 表格式分核表」（§5.5）当前均 skip，VP-2026-0009 混合字节口径类错误的机械拦截位在这些形态上失效
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 首行：ub_budget=skip、core_split=skip 的机械复核 JSON + REVIEW.md 人工逐行复算补位记录）
  - .agents/tools/design_calc_check.py#L100-103（check_ub_budget 行过滤 `^[a-zA-Z_]|缓冲` 把 N 值数字行当 header 跳过）+ #L156-158（skip detail "no per-level buffer tables parsed"）+ #L231-233（core_split skip "insufficient parsed dims"）
  - 交叉引用：queue VP-2026-0009（UB 预算混合字节口径——本条是其机械拦截位的形态扩展）
  - 〔2026-09-17 第二证（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕design_calc_check 对该 DESIGN.md **两轮（v0/v1）均 skip 3/4**（ub_budget / l0c_budget / core_split），且 v1 §4.5 已含逐行 Bytes 列与「稳态峰值合计」行、§5.2 有 bl/bs/bp/bn 代码块、§5.5 有 w1–w4 逻辑核数列表仍全部无法解析——与 ada_layer_norm（N 分行驻留表形态）构成**两类表格形态的独立第二证**；REVIEW 只能人工复算（UB 115.25KB 逐项加和、L0C 16KB、逻辑核数 8/768/2560/16384）。机械拦截位（VP-2026-0009 类）连续两任务不生效，建议审批时把本条的 §5.5 workload 分核表解析与 ssd 的 w1–w4 列表形态并入 new 文本 2 的识别范围
  - 〔2026-09-20 第三现 + 新子缺陷，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕4 项检查 3 skip（ub_budget/l0c_budget/core_split——该任务 §4.3 多张 buffer 表 + §5.2 block 常量代码块 + §5.5 分核三要素列表均无法解析，仅 r3_metrics 生效，全靠人工复算兜底）；**新子缺陷**：l0c_budget 解析取 "block 64x128"（bn=128 被当 block_N），与真实 L0C acc [bl,bp]=[64,64] 不符（结论 pass 侥幸同向）——GEMM 类多 block 参数文档的维度取用需语义识别而非首个数值对，建议并入 new 文本 2 识别范围（优先匹配 alloc_L0C/L0C 容量句上下文中的 [bl,bp] 组合）
  - 〔2026-09-28 第四现（不同任务，reduction 族）+ 两类新表形态，task argmax-_argreduce_kernel-20260924T031005Z〕design_calc_check 对该 DESIGN v0/v1 两轮检视 + v1 复审（同一任务三轮）ub_budget/core_split 全 skip，仅 r3_metrics 生效；人工复算 ~20 组数字（含 12 组分核算术）是发现膨胀系数数据点混用与 C1 缓冲区间漏 fp32 bm=1 的唯一途径。**两类新表形态**（建议并入识别范围）：① §4.5「B/elem 表 + 逐配置预算表」（列名含 手工预算/×2 双缓冲/结论）；② §5.5「逐 workload 分核表」（列名含 num_row_blocks/num_kernels/核内任务数）。REVIEW.md v0/v1 机械复核表（skip 三行 + 人工复算结论）。**合入优先级建议上调**：四任务（ada/ssd×2/argmax）连续命中，reduction 族最严重的阻塞项（workload (M,N) 差 1024×）正落在 skip 的 core_split 域——与 VP-2026-0136（--manifest 真值推导模式）互补：本条修「表解析」，该条补「真值来源」
- repro: 复现条件——DESIGN.md 采用「按 N 分行的驻留预算表」（§4.5）或「workload 表格式分核表」（§5.5）形态时跑 design_calc_check 两个检查项（当前均 skip）
  - 〔provenance，允许失效〕任务内复现命令：python3 .agents/tools/design_calc_check.py check --design examples/ada_layer_norm/_ada_layer_norm_kernel/DESIGN.md
- toolchain_stamp: design_calc_check.py 现行版本；失效环境 tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/tools/design_calc_check.py
- delta: |
    动作: update（两处解析扩展）
    定位锚 1: |
      (check_ub_budget 函数体内)
          if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
              continue  # header rows
    old 文本 1: （即上述定位锚 1 原文）
    new 文本 1: |
          if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
              # Row-N residence tables (N / block_m / 驻留字节 columns):
              # numeric first cell (N) + a per-elem-bytes or 驻留 column ->
              # parse as per-N footprint rows (bytes_per_elem * N) for the
              # UB budget check, so mixed-dtype byte widths are still caught.
              if not (cells and re.match(r"^\d", cells[0]) and any(
                  ("B/elem" in c or "字节" in c or "驻留" in c) for c in cells
              )):
                  continue  # header rows
    定位锚 2: |
      (check_core_split 函数体内，skip 分支)
              "detail": f"insufficient parsed dims {nums}; declared logical "
    old 文本 2: （即上述定位锚 2 原文所在 skip 分支）
    new 文本 2: |
      同分支前增加 workload 表格式识别：首列为 workload 名、含 "MxN"/shape 与 bm 列的表格行，从 (M, bm) 重算 num_logical = ceildiv(M, bm) 与表中声明值对照（无法解析全部维度时保留 skip，但 detail 注明已识别的 workload 行数）
    动机: ada_layer_norm 的 §4.5/§5.5 采用「N 分行驻留表 + workload 分核表」形态，两项机械检查双双 skip，混合字节口径错误只能靠人工逐行复算拦截——正是 VP-2026-0009 定义的机械拦截位，扩展解析使该防线对常见表格形态生效。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0048
- type: R
- title: core-split-strategy.md §2.2 reviewer 侧物理核数实查的替代验证条款——reviewer 无 NPU 设备连接时以「实查记录存在 + 常数表交叉 + 同款先例」三证放行，设备变更场景标注复核责任在 Stage 3/4
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 第三行：reviewer 无法独立重放 NPUUtils 实查〔需设备连接〕，本次以设计实查记录 §5.5 + CONST-aicore-910B2C + lerp 同款查询三方交叉佐证放行）
  - .agents/skills/_shared/standards/core-split-strategy.md §1 ②（实查记录要求，无 reviewer 侧独立验证手段条款）
  - 〔2026-09-28「有设备时直接重放」对照数据点，task argmax-_argreduce_kernel-20260924T031005Z〕reviewer 有设备场景：`NPUUtils.get().get_aicore_num()` = 24 于 v0 检视（05:15）与 v1 复审（06:34）两次独立重放一致；另五轮 Stage 1 探针实机重放（P1/P2/P2b/p3a×3/dbg，~12min）把 C-1~C-4 与冻结配置 bm=2 全部独立定证——「有设备优先重放实查、无设备走三证放行」的完整双分支口径（REVIEW.md v0 机械复核段 + Stage 2 复盘 VP 行，此前 VP 记录的是无设备分支）
- repro: 只读核对（无运行时依赖）——reviewer 环境无 NPU 设备连接时核数实查不可重放的位形
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/_shared/standards/core-split-strategy.md
- delta: |
    动作: update（§2.2 透传文本扩展）
    定位锚: "> 维度 3（Tiling 策略）须核对分核策略三要素齐全、物理核数为 NPUUtils.get().get_aicore_num() 实查值（纯 Vector 算子翻倍）、核数与 block 取值自洽、核内串行边界静态。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      > 维度 3（Tiling 策略）须核对分核策略三要素齐全、物理核数为 NPUUtils.get().get_aicore_num() 实查值（纯 Vector 算子翻倍）、核数与 block 取值自洽、核内串行边界静态。reviewer 环境无 NPU 设备连接、实查不可独立重放时，以三证放行：设计实查记录存在（查询代码 + 返回值）+ constants.md 物理核数条目交叉（CONST-aicore-910B2C）+ 同款先例（examples/ 同族查询记录）；目标设备非 910B2C 时三证链失效，核数复核责任显式移交 Stage 3/4（首跑核数断言或 profile Block Dim 核对）。
    动机: reviewer 侧无设备是常态位形（检视可在无 NPU 环境执行），现行条款只有 designer 侧实查义务、无 reviewer 侧替代验证手段——三证放行把该场景的核对义务显式化，避免形同虚设或过度阻塞。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0049
- type: R
- title: design-review Phase 1 第 7 步 verify_equiv 重跑核对补口径——「与内嵌表一致」须逐格对上确界（汇总格 = 逐案例输出的上确界），聚合口径不当的汇总值不构成不一致但须记建议级微瑕
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Value Point Proposals 首行：bf16 extreme-mod 案例逐输出 max_ulp=1 而 DESIGN §1.6.1 内嵌汇总格写 0——violation 计数与 EQUIV_PASS 判定不受影响，但「与内嵌表一致」的核对结论未逐格对上确界）
  - .agents/skills/tilelang-design-review/SKILL.md#L139（现行口径「执行结果须与 §1.6.1 内嵌结果表一致且全部 EQUIV_PASS」——「一致」未定义粒度）
  - 〔2026-09-28 第二形态（不同任务）：行数粒度对账，task argmax-_argreduce_kernel-20260924T031005Z〕重跑 verify_equiv 的 per-item 行数（E3=33）与 DESIGN 内嵌表（40 行）不符而总数与结论一致——逐行核对发现行数差异记为建议级（不因总数相符跳过）；「一致」口径除上确界外还须覆盖**行数粒度**（内嵌表行数 vs 脚本输出行数），出处：该任务 Stage 2 复盘 Transferable Lessons 末条 + REVIEW.md 等价性段
- repro: 复现条件——任一含内嵌汇总格的 verify_equiv 结果表重跑核对（汇总格 vs 逐案例输出上确界）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（bf16 段 extreme-mod 行 vs 内嵌表）
- toolchain_stamp: torch CPU + tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步首条 bullet 扩展）
    定位锚: "   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）。「一致」的核对口径：汇总格（如「逐位一致 max ulp=0」）须等于逐案例输出的**上确界**（ada_layer_norm 实证：bf16 extreme-mod 逐案例 max_ulp=1 而汇总格写 0——violation 计数与判定不受影响，记建议级微瑕并要求修正汇总格，不判 fail）；
    动机: 「与内嵌表一致」未定义粒度时，聚合口径差异（min vs sup）与真实漂移不可区分——逐格对上确界是零成本口径收紧。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0050
- type: R
- title: design-review 维度 1「API 存在性」补 docs 无文档 API 的 examples 佐证豁免口径——T.min 类 API 在 docs/Tilelang.language/ 全子目录无独立文档（含未映射目录已枚举），3 处 examples 实调构成合法佐证
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 第二行：`ls docs/Tilelang.language/` 无 T.min.md；examples/flash_attention/flash_attn_npuir_dev.py L86 / lerp / logsumexp 三处实调）
  - docs/Tilelang.language/ 全子目录枚举（含创建操作/索引与元素操作/条件操作/排序操作/逻辑操作/原子操作/编译器提示操作——负向断言举证标准要求 docs 路径，但该 API 连正向使用也无文档可引）
- repro: 复现条件——语言文档全部子目录枚举无该 API 独立条目，且仓库示例代码实调检索得 ≥2 处跨算子目录的使用
  - 〔provenance，允许失效〕任务内核对命令：ls docs/Tilelang.language/数学操作/ | grep -i min（无 T.min 条目）；grep -rn "T\.min(" examples/ --include="*.py" | head（多处实调）
- toolchain_stamp: 仓库 docs 现状 2026-09-10；与 tilelang 版本无关
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 1 表「API 存在性」行扩展）
    定位锚: "| API 存在性 | DESIGN.md §3.2 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | 设计用了 `T.flash_attention` 但项目无此 API |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      | API 存在性 | DESIGN.md §3.2 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | 设计用了 `T.flash_attention` 但项目无此 API |
      | docs 缺文档 API 的佐证口径 | API 在 docs/Tilelang.language/ 全子目录（含未映射目录）无独立文档时（实证：T.min，2026-09-10 枚举），≥2 处 `examples/` 实调（跨算子目录）构成存在性佐证，检视记录注明「docs 无条目 + N 处实调」——负向断言举证标准不因文档缺口而弃用，但正向存在性不要求 docs 条目 | 仅有 docs 条目而无任何实调（文档与实现漂移风险） |
    动机: 负向断言举证标准（须引 docs 路径）遇到「API 无文档但实际存在」时举证责任无法满足——T.min 实证靠 3 处 examples 佐证放行；口径明示化避免同类场景每次重新论证。配套（超出本提案写域，需人工/contributor 处理）：docs/Tilelang.language/数学操作/ 补 T.min 条目（标量 min，含尾块 real_mode 用法）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0051
- type: R
- title: develop SKILL.md Phase 4 结果收集补失败分类要求——compile/runtime-error 与 precision-fail 分类上报（混排同列会使 conductor 的 precision_fix 路由把编译错误判为精度错）
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Skill Flow Issues 第二行：首跑 12 个 L0 失败混有两类根因〔1 个编译期 UB 溢出 RUNTIME-ERROR + 11 个 vrsqrt 精度 FAIL〕，汇总输出未分类）
  - 同任务首跑输出（`[collected] L0-1 RUNTIME-ERROR` 与 `[L0-5] FAIL: max_diff=...` 混排同列）
  - .agents/skills/tilelang-op-develop/SKILL.md#L48（现行「3. 收集结果：max_diff、失败用例 shape、层级」——无失败分类要求）
- repro: 复现条件——任一 L0 首跑同时含编译失败与精度失败的混合位形（本任务实证）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 4 收集步骤扩展）
    定位锚: "3. 收集结果：max_diff、失败用例 shape、层级。"
    old 文本: |
      3. 收集结果：max_diff、失败用例 shape、层级。
    new 文本: |
      3. 收集结果：max_diff、失败用例 shape、层级；失败须按 **compile/runtime-error vs precision-fail 分类**上报（分类计数 + 各类代表案例），不得混排同列——直接影响四出口判定与 conductor 路由精度（ada_layer_norm 实证 2026-09-10：首跑 12 失败 = 1 编译期 UB 溢出 + 11 精度 FAIL，未分类时 precision_fix 路由会把编译错误误判为精度错）。
    动机: 四出口判定与 conductor 路由按失败类型分流；混合失败位形下无分类的汇总使路由依据失真。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0053
- type: R
- title: optimize SKILL.md Phase 2 第 6 步「每个实验分支跑 L0 精度回归」补分支类型分层——参数分支（同 kernel 不同 config）可用 bench 内嵌 golden check 等价替代，结构分支必须全量套件
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md#Skill-Retrospective（BP proposal：参数扫描分支〔同 kernel 不同 bm〕的精度门禁 = bench `--check` 精确 (M,N,dtype,bm) golden 比对；结构分支 = 全量 `--level all`——该分层未显式定义，「每个实验分支跑 L0」对参数分支语义模糊，内嵌 L0 表不覆盖被扫的 bm）
  - 同文件 Iteration 1/2/3（参数分支 18+16+8 行记录全走 --check、结构分支走 --level all 的实际执行形态）
  - 〔2026-09-17 证据扩展（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕**结构分支 L0 回归须含 bf16 × 非整除 shape 组合**：v5_xband 引入的 L1 band 尾块越界（`s0:s0+bs` 未按 `ts=min(bs,Q−s0)` 裁剪）在 fp16 Q=96 靠 L1 布局运气通过 L0、潜伏 4 轮，仅 Boundary 层 bf16×Q=96 拦截（traps-runtime.md TRAP-L1-band-dst-tail-overrun；opt_log §6.1/§9.2）——「结构分支必须全量 --level all」若不含 bf16×非整除组合仍是假门禁
- repro: 复现条件——任一含参数扫描分支（同 kernel 不同 bm/config）的调优轮（本任务 round 1/3 实证形态）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 2 第 6 步扩展）
    定位锚: "6. 每个实验分支跑 L0 精度回归。"
    old 文本: |
      6. 每个实验分支跑 L0 精度回归。
    new 文本: |
      6. 每个实验分支跑 L0 精度回归，按分支类型分层：**参数分支**（同 kernel 仅改 config，如 bm/num_kernels 扫描）可用 bench 内嵌 golden check（精确 (shape, dtype, config) 比对，如 bench.py `--check`）等价替代完整 L0 套件——内嵌 L0 表不覆盖被扫的 config 值；**结构分支**（kernel 代码改动）必须全量 `--level all`（ada_layer_norm 2026-09-10 实证形态：round 1 参数扫描 18 行全 --check、round 2 结构改动全量 + 分支 `--level all`）。
    动机: 「每个实验分支跑 L0」对参数分支语义模糊（L0 表按基准 config 参数化，不覆盖扫描值）——显式分层消除歧义，避免参数分支误跑不覆盖的全量（浪费）或结构分支误用单点 check（漏检）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0054
- type: R
- title: ab_test.py 输出目录创建后内置 chmod 700——msprof 权限校验拒绝 group-writable 目录，工具自建目录不 chmod 时采集必失败（需 umask 077 前置调用才可避开）
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md#Skill-Retrospective-工具/环境问题（ab_test.py 建目录未 chmod 700，msprof 拒写 "writable by any other users or group users"——需 umask 077 前置调用）
  - .agents/tools/ab_test.py#L50-51（os.makedirs 两处，无 chmod）
  - 交叉引用：queue VP-2026-0037（msprof 父目录权限规则——profile-collection.md 规则侧，本条为工具侧互补修复）
- repro: 复现条件——默认 umask 002 的共享 checkout 下以默认权限建 msprof 输出目录后执行采集（本任务实证形态；VP-2026-0037 的 8/8 首采失败为同机制父目录侧实证）
  - 〔2026-09-17 第三现（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕opt_log §9.4——ab_test.py 的 msprof 输出目录 group-writable 拒采再现（R5 蛇形裁决 pair 采集），本任务以预创建 pair 目录规避；两提案（0037 规则侧 / 0054 工具侧）均 pending 未 apply 期间的第三次发生
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-10
- target_doc: .agents/tools/ab_test.py
- delta: |
    动作: update（两处 makedirs 后追加 chmod）
    定位锚: |
      (run_pair 函数体内)
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
    old 文本: |
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
    new 文本: |
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
          # msprof refuses to write into group/world-writable dirs (same
          # check as VP-2026-0037's parent-dir rule); self-created dirs
          # must be tightened explicitly regardless of umask.
          os.chmod(out_dir, 0o700)
    动机: msprof 权限校验覆盖输出目录自身；工具自建目录依赖调用方 umask 077 是隐式契约——内置 chmod 消除对调用环境的依赖（规则侧 VP-2026-0037 已覆盖「须 700」，本条使工具自身满足该规则）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0055
- type: R
- title: integrate_kernel.py 产物侧校验两缺陷——① wrapper 契约名在产物中无顶层定义时 gen_init_py 静默降级 [warn]（__init__.py 仍声称 __all__ 含该名，smoke 才 ImportError 且无修复指引；幂等重跑覆盖集成侧补丁）；② gen_kernel_source_block 的 stem fallback 把非函数名（常量/helper）当模块路径生成无意义 perf_opt 占位行
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md（Debug history attempt-1：首跑 `[warn] ALIGNMENT: not found in integrated modules` ×2 → `[error] import smoke test FAILED`；Issues 第 1/2 条）+ ada_layer_norm.py L61/L63（`# from .ada_layer_norm_kernel.perf_opt.ALIGNMENT import ALIGNMENT` 类占位行）
  - examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#L158-162（defining_module==None → [warn] 注释行）+ #L211（`stem = defining_module(name, integrated_files) or name`）
  - 交叉引用：queue VP-2026-0032（同文件 stale lineage 告警——缺陷集群）、VP-2026-0012（幂等重跑覆盖集成侧修复——本任务 Stage 5 同款再现，证据链互链）
- repro: 重跑 python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/norm/ada_layer_norm/.migration_meta.json（覆盖集成副本后 smoke 必现 ImportError——provenance，允许失效）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（两处）
    定位锚 1: |
      (gen_init_py 函数体内)
              stem = defining_module(name, integrated_files)
    （其后 [warn] 分支，L158-162）
    old 文本 1: |
              stem = defining_module(name, integrated_files)
    new 文本 1: |
              stem = defining_module(name, integrated_files)
    （[warn] 分支升级为可操作 [error]：parse_wrapper_imports 得到的 wrapper 契约名在产物顶层定义缺失时——提示两条修复路径：集成包内逐字 re-export GPU extracted 的同名定义（Stage 5 胶水先例，见 ada_layer_norm integration_log attempt-1）/ 调整 scaffolder-integrator 契约透传 helper 名；并保留已修补的非产物段不再整文件覆盖〔幂等重跑保护〕）
    定位锚 2: |
      (gen_kernel_source_block 函数体内)
              stem = defining_module(name, integrated_files) or name
    old 文本 2: |
              stem = defining_module(name, integrated_files) or name
    new 文本 2: |
              stem = defining_module(name, integrated_files)
              # Non-function names (constants/helpers) have no per-func
              # perf_opt module: fall back to the single integrated module
              # stem and import by name, or emit no placeholder line --
              # never use the bare name as a module path.
    动机: 契约名缺失从「静默 warn + smoke 期 ImportError 无指引」升级为前置可操作 error（修复手法已在 ada 案例验证）；占位行 stem fallback 对常量/helper 生成语义错误的模块路径（注释态无害但误导后续 wrapper 切换）。两缺陷与 VP-2026-0032/0012 同属 integrate_kernel.py 幂等与校验缺口集群，建议同批审批。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0056
- type: R
- title: integrator 已知修复目录「import / 路径错误」行补 extracted-import 契约缺口形态——wrapper 从 GPU extracted 模块导入的全部名字（含常量/helper）构成集成 re-export 契约；NPU 重设计丢弃对应机制时最小修复 = 集成副本逐字 re-export 同名定义
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md（Debug history attempt-1 完整修复链：ALIGNMENT/_align_up 逐字 re-export + __init__ 修正 re-export；安全判别——名字只影响被 NPU 工厂接受但忽略的 flag 或无消费者的 config 选择器路径）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm.py L13-21（逐字源：GPU extracted _ada_layer_norm_fwd_kernels.py）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/RETROSPECTIVE.md（Stage 5 Transferable Lessons：wrapper 政策函数与 pytest 政策测试仍校验 GPU 语义，属 wrapper 胶水契约——集成期不顺手 NPU 化）
- repro: 复现条件——任何 wrapper extracted-import 含非 kernel 名字（常量/helper）且 NPU 产物不定义它的迁移（重跑 integrate_kernel.py 覆盖集成副本 → smoke 必现 ImportError）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（已知修复目录表「import / 路径错误」行扩展）
    定位锚: "| import / 路径错误 | 检查聚合 `__init__.py` 与相对导入 |"
    old 文本: |
      | import / 路径错误 | 检查聚合 `__init__.py` 与相对导入 |
    new 文本: |
      | import / 路径错误 | 检查聚合 `__init__.py` 与相对导入。**extracted-import 契约缺口**（ImportError 指向常量/helper 名而非 kernel 名时）：wrapper 从 GPU extracted 模块导入的全部名字构成集成包 re-export 契约；NPU 重设计丢弃对应机制（padding/cp.async 类）致产物不顶层定义这些名字时，最小修复 = 集成副本逐字 re-export GPU extracted 的同名定义（语义零漂移，wrapper 政策函数与 pytest 政策测试继续校验 GPU 语义——集成期不顺手 NPU 化）；安全判别：名字只影响被 NPU 工厂接受但忽略的 flag 或无消费者的 config 选择器路径。修复后幂等重跑会覆盖补丁（同 VP-2026-0012/0055 caveat） |
    动机: 该形态在本任务以 1 attempt 闭环（修复手法验证有效），但检索无门——已知修复目录只有泛化的「检查 __init__」；契约语义（名字清单 = 契约）与最小修复形态入目录后，同类迁移任务可一次定位。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0064
- type: R
- title: integrate_kernel.py 幂等检查只认未注释的激活 import——rewrite_wrapper_import 的 "already integrated" 探测正则匹配到被注释的 baseline 行（调优轮 perf_opt 激活态遗留），wrapper 不重写、旧谱系 perf_opt kernel 静默保持生效
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 Integration steps Step 1（`[wrapper] ... already integrated`〔wrapper_rewritten: false〕时 wrapper L65 baseline 被注释、L67 perf_opt import 生效；`[smoke]` 仅 import 冒烟不区分激活源）+ Step 2 人工胶水修复（`w._gqa_prefill_fwd_kernel.__module__` 修复前解析到 perf_opt 旧谱系）
  - examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#L224-226（text 全文 re.search，不剥离注释行）
  - 交叉引用：queue VP-2026-0032（stale perf_opt lineage 告警——预警场景本轮实际发生的另一形态，修「重写后不告警」侧）、VP-2026-0055/0056（同文件幂等与校验缺口集群，建议同批审批）
- repro: 复现条件——wrapper 处于 perf_opt 激活态（调优轮遗留：baseline import 被注释）时重跑 integrate_kernel.py → 输出 already integrated 且 wrapper_rewritten=false，旧谱系 perf_opt import 保持激活
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15；脚本现行版本
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（幂等检查剥离注释行后匹配）
    定位锚: |
      (rewrite_wrapper_import 函数体内)
          text = wrapper_path.read_text(encoding="utf-8")
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", text):
              return False
    old 文本: |
          text = wrapper_path.read_text(encoding="utf-8")
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", text):
              return False
    new 文本: |
          text = wrapper_path.read_text(encoding="utf-8")
          # Idempotency must consider only ACTIVE (uncommented) imports: a
          # commented-out baseline line (left by a perf_opt-activation tuning
          # round) must not satisfy this probe -- otherwise the wrapper is
          # never rewritten and the stale perf_opt import stays active
          # (factory-compatible signature; import smoke does not detect it).
          active_lines = [
              line for line in text.splitlines()
              if not line.lstrip().startswith("#")
          ]
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", "\n".join(active_lines)):
              return False
    动机: 换代重集成遇 perf_opt 激活态 wrapper 时被注释的 baseline 行骗过幂等检查——旧谱系调优 kernel 静默保持生效且 smoke 不拦截（2026-09-15 实际发生，靠人工 __module__ 核对才发现）；与 VP-2026-0032 成对互补。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0065
- type: R
- title: develop SKILL.md Phase 4 L1 门禁补「wrapper-default 派发路径」——带 config 替换语义的 kernel，出厂 dispatch 的 traced 变体必须 ∈ 被门禁编译验证过的变体集
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/RETROSPECTIVE.md Stage 5 round-2 Skill Flow Issues 第 2 行（Stage 3 L1 门禁用逐字 config `fn(64,64,ns=2)` → bn_eff∈{64,144} 验证 manifest 域，wrapper 实际派发走 E6 替换路径 (64,64,1)→(64,256)；L0-3/L0-4 仅 dim=64 smoke 验证替换路径）
  - integration_log.md Round 2 §Design-layer finding（E6 替换路径在 dim=128 causal 域的首次编译发生在 Stage 5 bench 且直接 UB 溢出 8/10；pytest 4 参数全 causal=False 三重掩盖）
  - 〔2026-09-16 第二证（不同任务），task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z r9 precision_fix〕同类失效再发生：r7e per-shape bm=80 分派 × E6 bn 钳位（S_kv≥3841 → bn_eff=288）的组合变体从未被任何门禁编译（standalone 29 例 S_kv≤2048 / manifest S≤2048 / 此前 pytest 域未覆盖该 S），首次编译发生在 TileOPs pytest full-fwd-bf16 且直接 UB 溢出 13472B（bishengir-compile exit 1）——「出厂 dispatch 的 traced 变体 ∉ 门禁验证集」的同一失效类，且两个守卫各自安全、组合超限（联合核算判据 + repro：attention.md PL-1.12 r9 update / repro/PL-1.12-bn-clamp-bm-guard.py）。Tier 2 人工门不因证据数豁免，供审批参考；与 VP-2026-0074（修复侧 config-契约范式）互补
  - 〔2026-09-21 第三现（不同任务、kernel 内配置分支亚型），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 重建会话〕tuned bn=128 经 min(bn,N) 钳位使 N_TILES 恒为 1——kernel 自带 L0/L1 门禁（TUNED config 口径）**永不触达 N_TILES_MULTI 代码路径**，而 tileops wrapper 的 GPU 启发式默认 bn=min(64,N)（H=80 时 N_tiles=2）会走该路径；结构改动涉及配置分支时须补 bn<N 探针（本会话 `perf_opt/probe_bn64.py` 4 case：双 dtype × 尾块 × 双 pp 全过）。**建议 apply 时把 new 文本的「门禁变体集」口径扩为「默认/tuned 双配置 × 关键配置分支形态」的笛卡尔积**（含 tilelang-op-optimize SKILL.md Phase 2 分支验证清单同句）；数据侧已回写 attention.md PL-1.18 update③（BP_config_path_coverage）
- repro: 复现条件——任一 kernel 工厂带 config 替换/分派语义的迁移任务（wrapper default_config 派发的 traced 变体未入 L1 变体集时，出厂首编后移到 Stage 5 bench 暴露）
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 4 第 2 步扩展）
    定位锚: "2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。"
    old 文本: |
      2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。
    new 文本: |
      2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。**门禁变体集覆盖 wrapper-default 派发路径**：kernel 工厂带 config 替换/分派语义时，wrapper `default_config` 实际派发的 traced 变体（含替换路径与惰性旋钮组合）必须被 L1 在目标域（dtype/dim/causal 维度）编译验证——至少一组与 wrapper `default_config` 全同的调用 + 替换路径的目标域案例（2026-09-15 反例：E6 替换路径 bn_eff=256 在 dim=128 causal 域首次编译发生在 Stage 5 bench 且 UB 硬溢出 8/10，L1 逐字 config + L0 dim=64 smoke + pytest 全 non-causal 三重掩盖）。
    动机: 出厂 dispatch 变体不在门禁变体集内时，唯一编译门控后移到 Stage 5 bench——集成期编译失败按功能缺陷查根因的成本远高于 L1 期发现。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0066
- type: R
- title: design SKILL.md Phase 4 补 CJK 排版规则——数学记号中 CJK 只出现在圆括号或正文、不进 {...}（含 LaTeX \text{中文} 下标与 e^{中文 − var} 上标），规避 S1-PLACEHOLDER 误报；命中合法记号时改写消除模式命中而非申诉豁免
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 gate-修复重试章节 Skill Flow Issues 首行（DESIGN.md L41/L446 两处误报 `e^{块内分数 − m_cur}`、`_{\text{Cube 部和，f16 物化回传}}`；修复 = 括号形式改写 + 注解移出公式，`grep -P '\{[^}]*[\x{4e00}-\x{9fff}][^}]*\}'` 复核清零，语义零变化）
  - 交叉引用：queue VP-2026-0019（gate_lint 正则侧修复——本条为 writer 侧互补；2026-09-15 新形态 `e^{中文 − var}` 无列表分隔符，VP-2026-0019 现提案正则不覆盖，见其证据链追加）
- repro: 复现条件——DESIGN.md 数学记号含 CJK-in-braces 形态（`grep -P '\{[^}]*[\x{4e00}-\x{9fff}][^}]*\}' DESIGN.md` 有命中）时跑 gate 1
- toolchain_stamp: gate_lint.py 现行版本；误报发生于 2026-09-15 任务
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 首行后追加一条 blockquote）
    定位锚: "基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节："
    old 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：
    new 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

      > **CJK 排版规则（S1-PLACEHOLDER 误报规避）**：DESIGN.md 数学记号中的 CJK 只出现在圆括号或正文，不进 `{...}`（含 LaTeX `\text{中文}` 下标与 `e^{中文 − var}` 形态上标）——gate 1 的 CJK-in-braces 模式会命中此类记号判「疑似模板变量未替换」。命中合法记号时的修复姿势 = 改写消除模式命中（括号形式 `e^(...)` / 注解移出公式为正文括注），语义零变化、门禁可机械复核，不申诉豁免（2026-09-15 实证：两处误报改写后 `grep -P` 复核清零；纯 ASCII 花括号不触发）。
    动机: gate 1 误报消耗整段修复重试（本任务 Stage 1 重试的 2/4 项）；writer 侧预防与 gate 侧正则修复（VP-2026-0019）双管齐下，正则修复未落地前排版规则是唯一防线。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0067
- type: R
- title: design SKILL.md「超长文档分段落盘」从条件建议升级为阈值无条件触发——VP-2026-0022 于 2026-09-11 apply 后 2026-09-15 仍复现首写截断（「预计超过」的自行判断是失效点），heredoc 分段追加为可靠替代
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节 Skill Flow Issues 首行（1116 行 / ~95KB DESIGN.md 首次 write 即截断——上版建议已合入仍复现且更早失败；bash heredoc 6 段追加成功）
  - .agents/skills/tilelang-op-design/SKILL.md Phase 4 blockquote（VP-2026-0022 2026-09-11 apply 文本——条件式「预计超过 ~60KB」）
  - 旁证（同任务 Stage 3 侧 sibling）：.task_timeline.jsonl Stage 3 attempt 1 fail 809s（空返回——单次 LLM 响应生成整个大文件被 provider 截断；VP-2026-0021 同样 2026-09-11 apply 后仍首发失败，conductor 增量写入流程指令后才成功）——分段纪律的条件式表述在两个 agent 上同日失效
- repro: 复现条件——单次 Write 输出 >60KB 的 DESIGN.md（2026-09-15 本任务首设计 attempt，规则已合入状态下复现）
- toolchain_stamp: 会话层行为（无运行时依赖）；复现环境 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（替换 Phase 4「超长文档分段落盘」blockquote 措辞）
    定位锚: "> **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。"
    old 文本: |
      > **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。
    new 文本: |
      > **超长文档分段落盘（>60KB 一律执行）**：DESIGN.md 预计超过 ~60KB / ~1200 行时一律分段落盘（bash heredoc 分段追加为可靠替代——单次 Write 截断后 heredoc 不受工具 JSON 体积限制），不以单次整文件 Write 交付，不以「预计不会超」的自行判断跳过——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 首证：1249 行首写即截断；**2026-09-15 复发**：本规则 2026-09-11 合入后 designer 仍先单次 Write 且首写即截断〔~95KB〕，heredoc 6 段恢复——条件式「预计超过」判断是失效点）。
    动机: 条件式建议在两个 agent（design/develop）上同日失效（Stage 3 侧 VP-2026-0021 同款复发）——阈值触发改为无条件 + 明示 heredoc 替代，把「是否分段」从 agent 自由裁量变为机械规则。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0068
- type: R
- title: design SKILL.md Phase 2 等价性机器验证补「判定口径」bullet 三条——ULP 阈值按恒等式域界第一性推导（勿拍小常数）/ 逐项隔离显式关闭其他量化开关 / 探针行与判定行分离
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 gate-修复重试章节 Value Point Proposals 后三行（M1 首跑 15 ulp 对拍定 8/16 阈值假 FAIL → 域界推导 |t|≤87 → 上界 ~120 ulp，阈值 128 后 viol 双口径 0；M14 沿用默认管线被 M12 的 ~2.6e-3 相对误差污染 viol 87% 假 FAIL → 隔离后 0；基线式失败演示只进 info 不参与判定）+ verify_equiv.py（13/13 EQUIV_PASS，双跑 diff 一致）
- repro: 复现条件——任一含 softcap/深尾域或多项量化开关的 verify_equiv 编写（任务内复现命令见 evidence provenance 项，torch CPU）
- toolchain_stamp: torch 2.7.1+cpu / 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 第 5 项「对照对象」bullet 后追加一条）
    定位锚: "   - **对照对象**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）；"
    old 文本: |
      （即上述定位锚原文，保持不变）
    new 文本: |
      （定位锚原文保留）其后追加：
         - **判定口径**：① ULP 阈值按恒等式结构第一性推导，不拍小常数——exp 类恒等式的 ulp 差与指数入参域界成正比（|δt| ≤ c·u·|t|max → P 域 ulp ≤ c·|t|max·ln2；softcap 域 |t|≤87 → 上界 ~120 ulp，实测 15——拍 8/16 在深尾假 FAIL）；② 逐项隔离对照显式关闭其他量化/物化开关（被验证项之外全 off，否则他项误差污染本项参照门——M14 实测 viol 87% 假 FAIL → 隔离后 0）；③ 基线式失败模式演示（溢出/NaN/发散）只进 info 不进 PASS/FAIL（探针行与判定行分离）（2026-09-15 attention 两相位 verify_equiv 实证，13/13）。
    动机: 首跑两类假 FAIL（阈值拍脑袋 + 隔离缺失）都发生在「脚本已产出、口径不当」阶段——判定口径入 skill 可在首次编写时避开，省一轮 gate 往返（本任务 gate 重试 4 项中 2 项属 EQUIV-EXEC 类）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0069
- type: R
- title: algorithm-research.md §6 信息源层「本仓 examples 同类实现」补全目录清单完成标准——结构先例检索以 grep -rl 族关键词显式列举命中清单并逐一归类为完成，不以命中第一个目录为完成
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节（Research-flow 行：本轮以 `grep -rl "softmax\|attention" examples/` 显式列举 12+ 文件并逐一归类〔结构原型 / tuned 母本 / 机制先例 / 证据档案〕才闭环；仅按 AGENTS.md 关键词路由会漏 deepseek/mixcv/torch_tl_ops 侧先例）+ Transferable 首条
  - 前序根因（provenance）：examples/TileOPs/.../perf_opt/perf_feedback.md 修正附录（上版漏检两遍式先例 → 三轮调优 + [DESIGN_LIMIT] 才纠正）
- repro: 复现条件——任一迁移/重设计任务的结构先例检索（对照「已调研」断言 vs grep -rl 清单）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    动作: update（§6 第 3 层扩展）
    定位锚: "3. 本仓 `examples/` 同类实现所用算法；"
    old 文本: |
      3. 本仓 `examples/` 同类实现所用算法；
    new 文本: |
      3. 本仓 `examples/` 同类实现所用算法——**完成标准 = 全目录清单**：结构先例检索以 `grep -rl "<族关键词>" examples/` 显式列举命中文件并逐一归类（结构原型 / tuned 母本 / 机制先例 / 证据档案），不以命中第一个目录（如关键词路由的 flash_attention/）为完成（2026-09-15 实证：上版漏检 deepseek/mixcv 侧两遍式先例致 [DESIGN_LIMIT] 才纠正，本轮清单法 12+ 文件闭环；检索动作可复现、清单可检视，优于「已调研」断言）；
    动机: 调研漏检的代价是整条结构选型错误（上版三轮调优 + DESIGN_LIMIT 才暴露）；清单化完成标准使检视可机械化对账。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0070
- type: R
- title: algorithm-research.md §3 R4「流水可融合性」补行为维度注记——flag wait 是否阻塞发射流属结构属性须引实测档案佐证；候选间流量近似（<10%）不构成结构裁决依据，裁决权移交行为维度
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节（Info-source 行 + Transferable 第 2 条：R3 流量口径下单遍链与两相位仅差 ~10%〔2.51 vs 2.78GB〕，实测性能差 2.31×，决定项是 R4 的发射流行为〔wait 与依赖对齐方式〕——R3/R4 分工「流量筛选 vs 行为定序」应显式提示）
- repro: 复现条件——任一候选间 fabric 流量差 <10% 的结构裁决（纸面流量对比无法区分的位形）
- toolchain_stamp: 流程规则；证据环境 2026-09-15（2.31× 锚点 = fa 域 perf_records 实测，tilelang 0.1.2+3a214cde 2026-09-09）
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    动作: update（§3 R4 检查清单第 5 项扩展）
    定位锚: "  5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行；"
    old 文本: |
      5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行；
    new 文本: |
      5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行——flag wait 是否阻塞发射流属**结构属性**（非 tiling 可解），结论须引用 examples/ 实测档案（msprof pipe 利用率 / aic_scalar 占比）而非纸面推断；R3 类流量近似（候选间差 <10%）不构成结构裁决依据，此时裁决权移交本维度行为侧（2026-09-15 实证：单遍链 vs 两相位 fabric 流量差 ~10% 而实测差 2.31×，决定项是发射流行为）；
    动机: 上版误选单遍结构的根因之一即「流量相近时按流量拍板」——R3（流量筛选）与 R4（行为定序）分工显式化后，流量平局候选自动进入行为维度举证。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0071
- type: R
- title: T.sync_block_set.md 补「同一 event_id 多次 set 的聚合语义」条款——2-AIV 同 id 双生产（各 set 一次、Cube 单次 wait）的放行条件（双 AIV 均完成 vs 任一完成）无文档规定，设计只能以先例背书
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 2（2026-09-15）Skill Flow Issues 首行（T.sync_block_set.md §1 仅「同一 block 中的其他执行单元」一句、无多生产者语义条款；reviewer 只能以双生产先例〔E1-E7 29/29 + v11nt fa-tuned〕背书；设计侧列为 R-1 风险项 + 拆双 id 回退）
  - docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1（现行文本无该条款）
  - 〔2026-09-17 生产先例追加（不同任务不同族），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕mamba 族 MixCV persistent（AIV subid 蛇形分片后**双 AIV 仍执行相同 set/wait 序列**、同 flag id）全量门禁 + Stage 4 六轮 + TileOPs 集成 11/11 bench 通过——「one-set-multi-wait」形态第二个跨族生产实证（GQA dual-producer 之外）
- repro: 只读核对（无运行时依赖）——docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1 无聚合语义条款
- toolchain_stamp: 仓库 docs 现状 2026-09-15；与 tilelang 版本无关（文档条款缺失）
- target_doc: docs/Tilelang.language/同步管道操作/T.sync_block_set.md
- delta: |
    动作: update（§1 简介段后追加语义说明段）
    定位锚: "简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。"
    old 文本: |
      简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。
    new 文本: |
      简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。

      **同一 event_id 的多次 set 语义**：同一 block 内两个执行单元（如 2-AIV 半分类）对同一 event_id 各 set 一次、消费侧单次 wait 的聚合/覆盖语义（双生产者均完成才放行，还是任一完成即放行）本文档未规定——依赖该形态的设计应以仓内可运行先例佐证（examples/flash_attention/flash_attn_npuir.py 两相位、examples/multi_head_attention/ 双生产形态），并在设计文档中单列风险项与拆双 id 回退方案。
    动机: 2-AIV 半分类是 attention 族 Expert 设计的常见形态，聚合语义无文档时 reviewer 无法独立核验 flag 协议安全性（只能先例背书）——条款化（哪怕先标注「未规定 + 推荐单生产者」）使该风险项可在 docs 层闭环。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0072
- type: R
- title: T.gemm.md §1 签名模式标注与 Expert 生产用法口径不一致——L1/L0C 形态在 Scope("Cube") 内生产使用 T.gemm 名称（flash_attn_npuir / fp8_lighting_indexer 先例），与 [Expert mode] T.npuir_dot 标注的关系未澄清（两任务连续检视提出）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 2（2026-09-15）Skill Flow Issues 第 2 行（T.gemm.md L8-9 两行签名标注 vs examples/flash_attention/flash_attn_npuir.py L102-109 Scope("Cube") 内 `T.gemm(l1, l1, l0c, initC, b_transpose, size)` 生产使用——reviewer 依先例判定可行但文档层无法自洽闭环）
  - 前序独立证据：同文件 Sep-07 Stage 2 round 2 Skill Flow Issues 第 3 行（examples/deepseek_v32/fp8_lighting_indexer.py L72-79 同款形态落差——两轮检视连续提出）
- repro: 只读核对（无运行时依赖）——T.gemm.md §1 两行签名标注与仓内 Expert 生产先例的形态落差（先例路径见 evidence）
- toolchain_stamp: 仓库 docs 与 examples 现状 2026-09-15；与工具链版本无关（文档-先例落差）
- target_doc: docs/Tilelang.language/线性代数操作/T.gemm.md
- delta: |
    动作: update（§1 签名区追加关系说明行）
    定位锚: "T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]"
    old 文本: |
      T.gemm(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Developer mode]
      T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]
    new 文本: |
      T.gemm(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Developer mode]
      T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]

      > 注：Expert 模式 `Scope("Cube")` 内以 L1/L0C 张量为操作数的形态（如 `T.gemm(l1, l1, l0c, initC, b_transpose, size)`）在仓内先例（examples/flash_attention/flash_attn_npuir.py、examples/deepseek_v32/fp8_lighting_indexer.py）中以 `T.gemm` 名称生产使用；上两行签名标注按 fragment（Developer）形态理解，L1/L0C 形态下 `T.gemm` 与 `T.npuir_dot` 的别名/等价关系以先例为准。
    动机: 同一形态落差在两轮独立检视中被连续提出（reviewer 每次都须以先例对抗文档标注）——一句关系说明消除文档-先例张力，使维度 1 API 存在性核对可从 docs 层闭环。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0073
- type: R
- title: develop SKILL.md Phase 2 补「母本移植逐项甄别」检查项——结构母本只提供 API 形态与程序序，布局假设/整除域特化/非 persistent 假设须按目标任务重推导；移植后首个回归须含 S_q > bm 的多头用例
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 4 行 + Transferable 第 2 条（v11_2phase _builder_2phase 为 fa 域〔B=1/H=1/整除/非 persistent〕特化实验形态：全局行 ws 布局/无 mask 链/无 TASKDONE/无钳位在 persistent+多头+causal+尾块域全部需适配——ws 槽全局行基误用即直接诱因之一：bx=0 任务「全局=本地」巧合掩盖、bx≥1 全错〔lse 恒 log2(S_kv) 指纹〕，row0 索引修复后 29/29）
- repro: 复现条件——任一从实验文件/先例 kernel 移植结构的开发任务（对照母本假设清单 vs 目标任务域）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 清单追加第 4 项）
    定位锚: "3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。"
    old 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
    new 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
      4. **母本移植逐项甄别**：从实验文件/先例 kernel 移植结构时，「结构母本」只提供 API 形态与程序序——布局假设（全局行 vs per-core 本地行）、整除域特化（尾块/掩码/TASKDONE 缺失）、非 persistent 假设须按目标任务域重推导并显式列差异清单（2026-09-15 实证：v11_2phase 全局行 ws 布局直接移植致 per-core 槽 row0 索引 bug——bx=0 任务巧合掩盖、bx≥1 全错，lse 恒 log2(S_kv) 指纹锁定；移植后首个回归须含 S_q > bm 的多头用例）。
    动机: 母本的隐藏域假设（整除、全局行基、无尾块）只在目标域位形暴露——差异清单 + 定向回归用例把该类 bug 从「bx≥1 用例碰运气发现」变为「首个回归必现」。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0074
- type: R
- title: integrator 已知修复目录「config 参数（block/tile）不适配 NPU」行补 config-契约集成范式——kernel 带 config 替换语义时 wrapper default_config 是出厂契约决策：惰性旋钮路由到门禁验证过的逐字路径（1 行胶水修复恢复整个域可编译）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §调试历史 attempt 1（分类命中该已知修复行 → 根因链 5 项 → 最小复现双路径 A fail / B ok → 修复 = wrapper default_config num_stages 1→2：(64,64,2) 在 E6 替换集外 → 逐字路径 bn_eff∈{64,144} = Stage 3 L1 门禁验证过的 traced 变体；num_stages 为结构惰性旋钮〔单槽两相位 stage 不变，DESIGN R-7〕语义零影响 → smoke 4/4 + full 6/6 + bench 10/10 复核）
  - .opencode/agents/tilelang-op-integrator.md L102（现行已知修复行——未展开带替换语义 kernel 的契约层选法）
- repro: 复现条件——任一 kernel 工厂带 config 替换/分派语义的集成期编译失败（wrapper 还原 (64,64,1) 时 `pytest "benchmarks/ops/bench_multi_head_attention.py::test_mha_fwd_bench[llama-3.1-8b-short-float16]" -v --tb=short` 复现 UB 溢出）
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（已知修复目录表「config 参数」行扩展）
    定位锚: "| config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件） |"
    old 文本: |
      | config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件） |
    new 文本: |
      | config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件）。**config-契约范式（kernel 带 config 替换语义时）**：wrapper `default_config` 是出厂契约决策而非 GPU 遗产直搬——替换集外的**逐字路径 + 惰性旋钮**（如 num_stages，结构 stage 不变的设计保留项）是集成层对齐「Stage 3 门禁验证过的 traced 变体」的合法工具：改惰性旋钮选中逐字路径即路由回门禁变体集（2026-09-15 实证：num_stages 1→2 使 (64,64,2) 落在 E6 替换集外 → bn_eff∈{64,144} 逐字路径，1 行胶水修复恢复 manifest 域可编译，pytest/bench 全绿复核）；修复后以工厂解析检查（`w.{func}.__module__`）确认激活源 |
    动机: 该修复形态已验证（1 attempt 闭环）但检索无门——现行行只说「调整 default_config」，未展开替换语义 kernel 的契约层选法（何时改旋钮、为何逐字路径合法）；范式入目录后同类集成失败可一次定位。Stage 3 门禁对 wrapper-default 派发路径的覆盖缺口另立 VP-2026-0065（防患侧），本条为修复侧。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0075
- type: R
- title: optimize SKILL.md 证伪协议第 4 条「标量占比触发换轴」前置编译警告排查——aiv_scalar 高占比与 flag 自旋同形，`will execute by scalar instruction` 警告是两者的区分器（vcmp int16 标量化占壁钟 45% 实证）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4）：examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/opt_log.md#Phase-1（baseline aiv_scalar 94.5–96.8% 全 24 核均匀、两引擎 >90% 时间等待、aic_cube 0.7–1.4%——非引擎吞吐非带宽瓶颈）+ #判别实验（TILELANG_DUMP_IR + stderr 定位 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low efficiency`，仅 causal trace 有、每变体 2 处）+ #Round-2（根因定案：vcmp int16 全形态标量化 236µs/链 @[32,256]，对角块聚合 ≈ 壁钟 45%；修复后 14.2x）
  - pattern-library/traps-compiler.md TRAP-expert-v-operands 第五轮补充段（「与 flag 自旋同形，须查编译警告区分」）+ attention.md PL-1.11（数据面档案）
  - 误诊路径对照：bottleneck-patterns BP_cross_engine_serial_chain（VP-2026-0034）触发信号即「busy 核 scalar 占比 30%+ 且归因为 flag 等待自旋」——两种根因（标量化陷阱 / flag 自旋）在 aiv_scalar 指标上同形，归因步骤缺区分器是误诊源头
- repro: `TILELANG_DUMP_IR=TRUE`（配新 `TILELANG_CACHE_DIR`，见 VP-2026-0062——缓存命中跳过编译也跳过 IR 打印）重编译目标 kernel，stderr grep `will execute by scalar instruction`；命中则热点段含被降级的 v-op（对照 msprof aiv_scalar 占比与「链宽 × 实测单链耗时」估算贡献占比）
- toolchain_stamp: tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（证伪协议第 4 条原位扩展）
    定位锚: "4. **诊断信号强制触发换轴分析**：msprof 显示热点段标量执行占比 > 50% 时，强制评估向量化轴/布局重排候选（pattern-library/layout.md），不得只在原轴上微调参数。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. **诊断信号强制触发换轴分析（先排查标量化，再评估换轴）**：msprof 显示热点段标量执行占比 > 50% 时，第一步先用 `TILELANG_DUMP_IR=TRUE`（配新 `TILELANG_CACHE_DIR`）重编译并 grep stderr 的 `will execute by scalar instruction` 警告——命中说明热点 v-op（如 vcmp int16）被降级为标量指令执行（traps-compiler.md TRAP-expert-v-operands：与 flag 等待自旋在 profile 上同形，第五轮实证占壁钟 ~45%），按对应 TRAP 条目绕法处理；未命中且归因为等待/自旋时再强制评估向量化轴/布局重排候选（pattern-library/layout.md），不得只在原轴上微调参数。
    动机: 第五轮 baseline aiv_scalar 94–97% 与良性 flag 自旋同形，若直接走「换轴」路径会漏掉 vcmp int16 全形态标量化这一占壁钟 45% 的根因（本轮靠 IR dump 警告 + 微探针才定案，修复即 14.2x）；编译警告 grep 是零成本区分器，前置可避免一类高代价误诊。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0076
- type: R
- title: iteration-diagnosis.md 注意事项补微探针输出分片规范——`T.Kernel` 多 AIV 并发写同一 GM 输出缓冲存在竞态，探针结论（「链坏」类）可能是探针自身 artifact 而非被测机制缺陷
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4）：examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/probe_penalty_chain.py（P9 修正前后对照）+ opt_log.md#Round-2-实现要点（调试链：P9 探针因 T.Kernel(1) 双 AIV 写同一 GM 输出而竞态，产出误导性「链坏」结论，浪费一轮调试）
  - 同族先例（不同形态）：pattern-library/attention.md PL-1.9-blockwidth「探针教训」——缺 Cube→Vector flag 同步的数值崩坏是探针自身竞态而非 API 缺陷（跨引擎形态）；本条补多 AIV 写竞态形态，两形态同属「探针异常先怀疑探针自身」
- repro: 微探针 kernel 以 `T.Kernel` 启动 ≥2 AIV 且输出为单一 GM 张量时，对照按 subid 分片（`out[subid, ...]` 各写各区）或单 AIV 版本的输出稳定性（P9 实证：分片前结果不可复现地「坏」、分片后稳定）
- toolchain_stamp: tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「## 注意事项」列表末尾追加一条）
    定位锚: "- 如果所有候选优化点都不够直接，先补充 profile 或检查 profile 口径，不要盲目扩大搜索空间。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 如果所有候选优化点都不够直接，先补充 profile 或检查 profile 口径，不要盲目扩大搜索空间。
  - 微探针（判别实验 kernel）的输出缓冲按 AIV 分片（`out[subid, ...]` 各写各区）或显式单 AIV——`T.Kernel` 多 AIV 并发写同一 GM 输出区域存在竞态，会产出误导性的「机制坏」结论（2026-09-15 实证：双 AIV 竞态使向量链探针误报「链坏」，浪费一轮调试；同族形态见 pattern-library PL-1.9 探针教训——探针数值异常先怀疑探针自身同步/写冲突，再怀疑被测机制）。
    动机: 微探针是第五轮定案 vcmp 标量化的关键手段，但其自身的多 AIV 写竞态曾误导一轮调试——判别实验的可信性以探针正确性为前提，分片输出是零成本规范。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0077
- type: R
- title: optimize SKILL.md Phase 3 TileOPs 集成算子条目补「perf_opt 双镜像同步」规则——conductor 工作区与 tileops 集成目录为独立 inode，单侧修复时 pytest 门禁测旧拷贝（错误位级相同无法区分源与镜像）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：DESIGN.md §11.2 集成拓扑注意（「TileOPs wrapper import 的是 tileops/kernels/.../perf_opt/ 镜像拷贝（独立 inode），perf_opt 侧任何修改须双镜像同步，否则 pytest 门禁测的是旧拷贝」——conductor 收束补记）
  - 同任务 opt_log Round 10（r9 precision_fix：修复 conductor 工作区 perf_opt 后手动同步 TileOPs 集成镜像 `tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py`，wrapper 的 baseline/perf_opt 切换块不动；同步后 TileOPs pytest 6/6 passed）
  - 双 inode 实核（2026-09-16 蒸馏会话）：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` 与 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` 两份 81708B 拷贝并存（mtime 06:22 / 06:26）
- repro: 复现条件——conductor 工作区与 TileOPs 集成目录存在两份独立 perf_opt 拷贝的 optimize/precision_fix 任务，单侧修改工作区 kernel 后直接跑 TileOPs pytest（门禁测的是未同步的集成镜像，错误位级相同无法区分源与镜像）
- toolchain_stamp: tilelang 0.1.2+a13585dc + CANN 8.5.0 + Ascend910B2C / 2026-09-16；TileOPs pyproject 环境
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 3 条扩展）
    定位锚: "3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。**双镜像同步（conductor 工作区 + TileOPs 集成目录）**：任务工作区（`examples/{project}/{op}/perf_opt/`）与 TileOPs 集成目录（`tileops/.../perf_opt/`）是独立 inode 的两份拷贝——凡修改任务工作区 perf_opt kernel（含 precision_fix 会话），须同步拷贝到 TileOPs 集成镜像（wrapper 的 baseline/perf_opt 切换块不动）后再跑 TileOPs pytest/回归门禁；单侧修复时门禁测的是旧拷贝，且位级相同的错误使门禁无法区分源与镜像（2026-09-16 r9 precision_fix 实证：同步后 pytest 6/6）。
    动机: TileOPs wrapper import 的是集成目录镜像而非任务工作区文件；r9 precision_fix 若只修工作区，pytest 门禁会在旧拷贝上复现位级相同的 UB 溢出——既浪费修复验证又可能误判修复无效。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0078
- type: R
- title: statectl.py 状态迁移命令加并发互斥锁（flock）——双 conductor 会话并行驱动同一任务目录时 load→modify→write 交错，产出零产物 runtime fail 与交叉状态写入（4 次实证）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：.task_timeline.jsonl（6 次 fail 中 4 次 runtime 归因双会话竞争：03:38–06:15 期间另一窗口与本会话交错驱动——attempt 1「03:38 并行调度 #1 已 runtime 失败（819s 零产物）」记录于 .stage_state.json user_requirement；attempt 2 期间出现外来 set/snapshot 写入〔03:58:15〕；stage_retry_count 被推至 6 污染 E-2 指标）
  - .agents/tools/statectl.py 现行版本（Ctx.load→write 为无锁 read-modify-write；cmd_start 无活跃 attempt 互斥；atomic_write_json 只保证单次写原子性不保证序列化）
- repro: 复现条件——两个进程同时对同一 op 目录执行 statectl start/set/fail（本任务 2026-09-16 03:38–06:15 实录：interleaved set/snapshot + 819s 零产物 runtime fail ×1 + 竞争性 fail ×3）
- toolchain_stamp: statectl.py 现行版本；发生环境 2026-09-16（与 tilelang 工具链无关）
- target_doc: .agents/tools/statectl.py
- delta: |
    动作: update（main() 分发加互斥锁 + 新增 acquire_op_lock 辅助函数）
    定位锚: |
      def main(argv=None) -> int:
          parser = build_parser()
          args = parser.parse_args(argv)
          if getattr(args, "func", None) is None:
              parser.error("missing command")
          return args.func(args)
    old 文本: （即上述定位锚原文）
    new 文本: |
      MUTATING_ACTIONS = {"init", "start", "fail", "complete", "set", "snapshot", "repair"}


      def acquire_op_lock(op_dir: str):
          """Exclusive lock for state-mutating commands.

          Two conductor sessions driving the same op dir interleave
          load->modify->write cycles: silent state corruption, phantom
          runtime fails and cross-written snapshots (2026-09-16 gqa
          task: 4 competing runtime fails, first one 819s zero-artifact).
          flock converts that into a loud immediate E-BUSY.
          """
          import fcntl

          lock_path = os.path.join(op_dir, ".stage_state.json.lock")
          fh = open(lock_path, "w", encoding="utf-8")
          try:
              fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
          except OSError:
              try:
                  holder = open(lock_path, encoding="utf-8").read().strip() or "?"
              except OSError:
                  holder = "?"
              fh.close()
              emit(
                  {
                      "action": "busy",
                      "errors": [
                          err(
                              "E-BUSY",
                              f"另一 statectl 会话持有 {lock_path}（holder pid {holder}）"
                              "——同一任务目录禁止并行驱动；确认无活跃会话后可删除锁文件重试",
                          )
                      ],
                  },
                  1,
              )
              return None
          fh.write(f"{os.getpid()}\n")
          fh.flush()
          return fh


      def main(argv=None) -> int:
          parser = build_parser()
          args = parser.parse_args(argv)
          if getattr(args, "func", None) is None:
              parser.error("missing command")
          lock_fh = None
          if args.command in MUTATING_ACTIONS:
              lock_fh = acquire_op_lock(os.path.abspath(args.dir or os.getcwd()))
              if lock_fh is None:
                  return 1
          try:
              return args.func(args)
          finally:
              if lock_fh is not None:
                  lock_fh.close()  # releases flock
      （注：只读命令 show/verify/gate/timeline-summary 不加锁；migration 子树若同构需要可另行评估，本提案只覆盖主状态命令）
    动机: 并行双会话的每一次交错都产生真实成本（819s 零产物 attempt + 交叉写入使时间线不可信 + stage_retry_count 污染 E-2 北极星指标）；flock 是 OS 级机械守卫，比「conductor 自觉不并行」可靠——同任务目录的第二驱动者立即收到 E-BUSY 而非静默竞争。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0079
- type: R
- title: statectl.py cmd_fail 补「未闭合 start 前置校验」——重复 fail 记账（无中间 start 的第二个 fail）虚增 stage_retry_count，污染重试指标与 E-2 抽取
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：.task_timeline.jsonl L14–15（06:15:32 fail 971s / 06:15:39 fail 978s——第二个 fail 前无任何 start，duration 均回溯到同一 start#5，stage_retry_count 5→6 虚增；conductor 归因记账侧重复调用）
  - .agents/tools/statectl.py 现行版本（cmd_fail 非 BLOCKED 路径无「本 stage 存在未闭合 attempt」校验；stage_attempt_duration 只回溯最近 start，第二个 fail 复用同一 start 的 duration）
- repro: 复现条件——对同一 stage 连续调用两次 statectl fail（无中间 start）：第二条 fail 照常追加时间线并递增 stage_retry_count（本任务 2026-09-16 06:15 实录，7 秒间隔双记账）
- toolchain_stamp: statectl.py 现行版本；发生环境 2026-09-16
- target_doc: .agents/tools/statectl.py
- delta: |
    动作: update（cmd_fail 非 BLOCKED 路径头部插入校验块）
    定位锚: |
      （cmd_fail 函数体内，E-TERMINAL 守卫块之后、design_revision 分支之前）
          if state["phase"] in TERMINAL_PHASES:
              errors.append(err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 fail"))
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)
    old 文本: （即上述定位锚原文）
    new 文本: |
          if state["phase"] in TERMINAL_PHASES:
              errors.append(err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 fail"))
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)

          # A fail must close an OPEN attempt: the most recent start/fail/
          # complete event for this stage has to be a start. Duplicate fail
          # invocations (2026-09-16: two fails 7s apart, no intervening
          # start) otherwise inflate stage_retry_count and pollute the
          # E-2 retry metrics.
          last_stage_action = None
          for ev in read_timeline_events(ctx.op_dir):
              if ev.get("stage") == n and ev.get("action") in (
                  "start", "fail", "complete"
              ):
                  last_stage_action = ev.get("action")
          if last_stage_action != "start":
              errors.append(
                  err(
                      "E-NO-OPEN-ATTEMPT",
                      f"Stage {n} 无未闭合的 start（最近事件为 {last_stage_action}）"
                      "——重复 fail 记账拒绝，不递增 stage_retry_count",
                  )
              )
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)
    动机: fail 是「闭合一个 attempt」的语义事件；无 start 的第二个 fail 只能来自记账侧重复调用——照单全收会虚增重试计数（本任务 6→实为 5），并使 duration 回溯到已消费的 start 产出误导性 duration_s。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0082
- type: R
- title: optimize SKILL.md T-4 实验批处理 runner 补适用边界——结构重构类分支（指令流重排/流水深度/同步协议）默认手工 diff 串行，生成脚本化仅用于参数扫描类
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：opt_log Skill Retrospective（第六轮）第 6 条（T-4 runner 未搭建的决策依据：分支数 8、依赖交互调试多，手工串行可控性更高；r8g 结构分支的生成脚本引入 3 次脚本 bug，叠加 2 次编译错 + 1 次精度失败 + 1 次 flaky 超时，消耗 ~40% 轮次时间）
  - 〔2026-09-21 第二任务证据（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z〕opt_log Skill Retrospective Skill Flow Issues 第 4 行：str.replace 型代码生成分支时同构 body（even/odd/tail）文本相同导致多点命中——round3 cband 生成脚本 + round7 xmid 两度返工共 3 次工具事故均为此类（单点实验纪律本身无缺口）；作者建议的缓解工具（run_experiments 模板提供 branch-from-base 结构化 diff 生成）作为本提案 apply 时的可选配套一并裁决
- repro: 复现条件——以脚本生成结构重构类分支（指令流重排/多槽 buffer/flag 协议变更）且需多轮交互调试的调优轮（本任务 r8g 实证形态）
- toolchain_stamp: 会话层工作流（无运行时依赖）；发生环境 tilelang 0.1.2+a13585dc / 2026-09-16
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（T-4 段末尾追加一句）
    定位锚: "脚本模板与适配说明见 [autotune.md](references/autotune.md)「实验批处理 runner」节。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      脚本模板与适配说明见 [autotune.md](references/autotune.md)「实验批处理 runner」节。**适用边界**：runner 适用于参数扫描/配置类分支（同 kernel 改 config）；**结构重构类分支（指令流重排 / 流水深度 / 同步协议 / 多槽 buffer 变更）默认手工 diff 串行**——此类分支依赖完整 kernel 上下文且需交互调试，脚本生成的变体一旦携带 bug，其调试成本高于手工串行（2026-09-16 attention 第六轮 r8g 实证：生成脚本引入 3 次脚本 bug，叠加编译/精度/flaky 共消耗 ~40% 轮次时间）。
    动机: T-4 的收益模型（交互轮次下降 60%+）建立在「分支可批量执行」的前提上；结构分支不满足该前提，脚本化反而放大失败面——边界标注使 runner 用在参数类刀刃上。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0089
- type: R
- title: pattern-library 条目时效性判定口径——以条目自身 `status:` + front-matter 重验行为准，kb_stale_check 全局 stale_count 可因工具链戳批量失配虚高（stale_count=83 误伤 verified 条目致掩码主备倒置）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 2 Skill Flow Issues 第 3 行 + Value Point Proposals——设计把 PL-1.11（front-matter `status: verified` + L117 载 2026-09-16 a13585dc 重验维持行）记作「stale 线索」并据此把实测已根治的病态形态（int16 vcmp 标量化占壁钟 45%）选为主选，Stage 2 独立复核才发现；根因 = kb_stale_check 全局 stale_count=83（工具链 0.1.2+1990aa9fe4 与条目戳批量失配）被当作单条目判定依据
  - Stage 1 first_design 同源问题：全局 stale 使设计期 roofline 只能按"估算口径"引用关键常数（CONST-capacity / CONST-vector-launch-overhead / CONST-mte2），削弱 D-2 定量性
  - pattern-library/attention.md#L110-L131（PL-1.11 status: verified + 重验行）vs DESIGN.md §1.6.3 候选 #3「stale 线索」定性
  - 〔2026-09-20 正确消费实例，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕重跑任务设计以条目自身 status + 重验行消费 PL-1.11（DESIGN §1.6.1 OPT-B 主选依据引「PL-1.11 verified，2026-09-16 a13585dc 重验维持」），未被全局 stale_count 误导；其流程化产物（stale 条目两分法消费指引）另立 VP-2026-0101（判后如何消费 vs 本条如何判 stale，两案互补互链）
  - 〔2026-09-21 第三现（规模峰值），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 重建会话〕stale_count=125 几乎全库假阳性——根因为 repo HEAD（96f287e，evolution 文档 commit）被当作 tilelang 工具链戳：`git diff --stat 15ad002..96f287e` 仅 7 个 .agents/ 文件、tilelang 源码零改动；任务内以 docs-only delta 论证 + probe 锚点复现（217.58 vs 217.65，−0.03%）自行消解，未造成错误决策，但 Phase 0 被迫逐条 git diff 自证。**工具侧根因修复提案见 VP-2026-0116**（stamp diff 过滤非源码路径——本条与其构成消费侧/工具侧互补，三案〔含 0101〕建议同批审批）
- repro: python3 .agents/tools/kb_stale_check.py（观察全局 stale_count 与单条 status 的口径差）；grep -n -A3 "^id: PL-1.11" .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md（读 status 与重验行）
- toolchain_stamp: kb_stale_check.py 现行版本；发生环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 第 5 条「弃选论证查证」末尾追加一句）
    定位锚: "代价类论断未引用 pattern-library 已有实测数据的，要求补证（该量已有实测则不再是"未实证假设"）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      代价类论断未引用 pattern-library 已有实测数据的，要求补证（该量已有实测则不再是"未实证假设"）。**条目时效性判定口径**：pattern-library 条目是否 stale 以**条目自身 front-matter `status:` 字段 + 重验行**为准；`kb_stale_check` 的全局 stale_count 可能因工具链版本戳批量失配而虚高（2026-09-17 ssd 实证：stale_count=83 误导设计把 `status: verified` 且已跨工具链重验维持的 PL-1.11 降级为「stale 线索」，致掩码主备倒置、实测已根治形态被重选为主选）——不得据全局计数把 verified 条目降级，工具链戳失配只触发「待重验」标注而非结论失效。
    动机: 同一口径建议同步写入 tilelang-op-design 的弃选论证证据规则（Phase 2/Phase 3 引用 pattern-library 处）——本提案主锚 design-review，op-design 侧同步属同语义扩散，审批时可一并裁决。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0090
- type: R
- title: 机器验证脚本的「模型形态」须与设计规格 API 级操作数同形核对——EQUIV_PASS 只覆盖数学、不覆盖 API 形态偏差（[1,bs] 脚本正确 + [bs,1] 规格错误仍全绿放行）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 2 Skill Flow Issues 首行 + Value Point Proposals——verify_equiv.py 重跑全 EQUIV_PASS 且与 §1.6.1 内嵌表逐项相符（skill Phase 1 第 7 步判定条件全满足），但脚本用 unsqueeze(0)（[1,bs] 列因子）建模、DESIGN.md §3.2/§3.3 却写 [bs,1]——数学被验证正确、API 形态错误，「假安心」直到 Stage 2 形态对读才发现
- repro: 复现条件——verify_equiv 类脚本与 DESIGN §3.2/§3.3 的形态对读（任务内对读命令见 evidence provenance 项）
  - 〔provenance，允许失效〕任务内对读命令：grep -n "unsqueeze" examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py 与 grep -n "0:1\]" examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/DESIGN.md 对读
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「重跑等价性验证」bullet 扩展）
    定位锚: "   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）。**模型形态同形核对**：抽查验证脚本中各因子/操作数的 shape 形态（unsqueeze 方向、[n,1] vs [1,n]）与 §3.2/§3.3 的 API 级操作数是否同形——形态不一致时机器验证不覆盖该偏差（2026-09-17 ssd 实证：脚本 [1,bs] 正确 + 规格写 [bs,1]，EQUIV_PASS 全绿仍放行至 Stage 2），须在维度 8 单列；
    动机: 配套（同语义扩散，审批时可一并裁决）：tilelang-op-design SKILL.md Phase 2 机器验证产出规范补一句「verify_equiv 脚本注释须标注每个因子的 shape 形态（如 `# [1,bs]`），供检视侧与 §3.2/§3.3 交叉核对」——形态标注是同形核对的可机械化前提。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0091
- type: R
- title: 编程模式选型硬约束——persistent 分核（核内 T.serial 多任务 + gemm）⟹ Expert 模式；design-review 维度 4 增「模式与分核策略一致性」必查行
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 3 Skill Flow Issues 首行——DESIGN §2.1/§0.7 声明 Developer 模式（user_requirement 亦指定）与自身 §0.6 R1 persistent 24 核分核策略不兼容，Stage 2 检视未拦截；Developer 首实现运行时崩溃（"unaligned UUB addresses"），Expert 同结构正常（repro/TRAP-DEVMODE-PERSIST-GEMM.py 双模式对照 + CG-2026-0010）
  - 判定依据：全仓 24 处 is_npu=True 中 Developer 仅用于非 persistent 网格（flash_attn_npuir_dev.py、fp8_lighting_indexer.py），persistent 混合算子（GQA、sparse_mla_fwd_exp.py）全部 Expert
  - 〔2026-09-20 知识生效先例，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕重跑任务设计期直接按 Expert 双 Scope 展开（算子结构落在 CG-2026-0010 崩溃域即判定），未再为 Developer 形态设计 fallback——省一版无效设计（该任务 Stage 1 Transferable Lessons）；本条 pending 期间的注入生效证据（kb_search 预注入命中 CG-2026-0010/PL-1.16）
- repro: python3 .agents/skills/tilelang-op-optimize/references/pattern-library/repro/TRAP-DEVMODE-PERSIST-GEMM.py（Expert 数值断言 + Developer 崩溃双模式对照）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表第 2 项扩展）
    定位锚: "2. 编程模式选型"
    old 文本: |
      2. 编程模式选型
    new 文本: |
      2. 编程模式选型（**persistent 硬约束**：分核策略含核内 `T.serial` 多任务 + `T.gemm`（persistent 混合算子）⟹ **Expert 模式**（显式 `T.Scope("Cube")`/`("Vector")` + alloc_L1/L0C/alloc_ub + sync_block_set/wait）——Developer 模式对该结构类运行时崩溃（"unaligned UUB addresses"，pattern-library TRAP-DEVMODE-PERSIST-GEMM / CG-2026-0010，2026-09-17 ssd 实证）；Developer 仅适用于非 persistent 简单网格；user_requirement 指定 Developer 而结构落入该类时，在设计文档记录实测仲裁依据后切换并披露）
    动机: 模式选型错误不在 Stage 2 拦截（检视只看文档不编译），以 Stage 3 运行时崩溃形式暴露成本最高（本任务白耗一次 attempt + 模式重写）。配套（同语义扩散）：tilelang-design-review SKILL.md 维度 4 表追加一行 `| 模式-分核一致性 | §2.1 编程模式与 §0.6/§5 分核策略一致（persistent + gemm ⟹ Expert，TRAP-DEVMODE-PERSIST-GEMM） |`（锚：维度 4 表末行 "| L0C 溢出 | 已在内存层级维度处理 |" 之后）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0092
- type: R
- title: 修订复检两机械步骤——v0 错误值/表述提炼成 grep 关键词全文扫描（区分历史说明 vs 未同步残留）+ 验证脚本 md5 比对（防「改脚本过关」）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md re_review 章节——v1 changelog 自称清理 6 处残留，复检独立发现 **7 处未同步**（§5.3/§9.1/§9.2 旧 UB 数字 115/129/196KB 与 v1 的 99/111/168KB 直接矛盾、§6.4/§3.5.1 的 `tk ≥ 32` 旧钳位、3 处 verified 条目仍标 stale）；grep 关键词组：`115KB\|129KB\|196KB\|贴限`、`tk ≥ 32\|K ≥ 32`、`stale 线索\|stale 条目`；另复检额外做了 verify_equiv.py md5 比对（dd0f0a8f… 与首轮一致）+ 形态同形核对确认分叉已闭合——均为 skill 未要求的自发动作
  - 交叉引用：queue VP-2026-0028（修订版重审工作流——diff 锁定深查范围；本条补其未覆盖的两个机械步骤）、VP-2026-0090（形态同形核对）
- repro: 复现条件——任一 Stage 2 不通过后修订版的复检（本任务 7 处残留实证）
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「重跑等价性验证」bullet 后追加同层 bullet）
    定位锚: "   - **算术复算**：`python3 .agents/tools/design_calc_check.py --design <path>`——逐项消费其 JSON 输出："
    old 文本: （即上述定位锚原文所在 bullet 起始行）
    new 文本: |
       - **算术复算**：`python3 .agents/tools/design_calc_check.py --design <path>`——逐项消费其 JSON 输出：
       - **修订复检（revision re-review）两机械步骤**（2026-09-17 ssd 实证：changelog 自称清理 6 处、复检 grep 仍抓出 7 处未同步，其中 2 处与修订版主体直接矛盾）：① 对每个已修复的阻塞/建议项，把 v0 的错误值与错误表述提炼成 grep 关键词在修订版全文扫描，命中处逐一判定「历史说明（changelog/风险条目/源码描述节，合法）」vs「未同步残留（现行规格节 §3–§6/§9 结论句，记建议级）」——不得以 changelog 修复清单作覆盖面依据；② 复检时记录并比对机器验证脚本（verify_equiv.py 等）与首轮的 md5——脚本未变而结论仍 PASS 才是有效证据，脚本有变更须判定变更性质（新增 case 合法、放宽阈值或改变建模形态需重新独立推演）。
    动机: 修订残留与「改脚本过关」是复检的两个结构性盲区——摘要/风险/循环章节最易漏改（本任务 7 处分布实证），md5 是零成本防篡改核。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0093
- type: R
- title: Stage 5 集成两防线——① wrapper `default_config` 与 kernel 内嵌 tuned 默认的口径衔接（翻转切换块激活 perf_opt 时同步核对，二者取一或标注口径差异）② 集成前工厂签名 diff 核对（零成本防线）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：op 级 RETROSPECTIVE.md Stage 5——wrapper 脚手架 `default_config`（baseline 启发式硬编码 block_n=64/num_stages=3）在 perf_opt 激活后仍显式传参，覆盖调优版 kernel 内嵌 `TUNED_DEFAULT_CONFIG(block_n=128/num_stages=2)`（custom_op 显式传参优先级更高），Stage 4 tuned config 集成态不生效；integration_log.md「Bench 观察项 #2」（bench 输出 config 与 opt_log §final 不一致）+ 工厂签名核对段（`diff <(sed -n '/^def .../,/^):/p' ...)` 输出 SIGNATURE_IDENTICAL——签名逐参一致使 wrapper 胶水零改动完成切换）
  - 〔2026-09-28 第二证（不同任务，reduction 族，wrapper 侧 GPU 启发式亚型），task argmax-_argreduce_kernel-20260924T031005Z〕Stage 5 集成：wrapper `default_config` 沿用 GPU 时代 `SHARED_MEMORY_BUDGET_BYTES=49152` smem 启发式选 bm=4，kernel 探针校准阶梯 `_select_config` 给 bm=2（含 min(block_m,M) 钳制与窄 N 预算减半，外部启发式均不掌握）——bm=4×4096 直接 `ub overflow, requires 2097408 bits while 1572864 bits available`；修复 = `default_config` 转调 `_select_config(self.M, self.N_padded, self.dtype_str)`，同一 report 42/42 通过（修复前后 run：20260928_024504 vs 20260928_030100）。诊断指纹：`ub overflow, requires N bits` 报文先查「谁传的 block_m、是否经 kernel 校准」。出处：examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/integration_log.md#attempt-1 + op 级 RETROSPECTIVE.md Stage 5 VP 行；工具侧联动见 VP-2026-0140（integrate_kernel.py 自动搬运 `_select_config` 成对 import——本次手工补行，翻转切换块时易漏翻）
- repro: 复现条件——任一 perf_opt 激活态集成（wrapper default_config 显式传参 vs kernel TUNED_DEFAULT_CONFIG 内嵌默认的优先级差）
- toolchain_stamp: tilelang npuir dev build（1990aa9fe4 谱系）+ CANN 8.5.0 / 2026-09-17
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（第一步 bullet 列表末尾追加两条）
    定位锚: "- 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。"
    old 文本: （即上述定位锚原文）
    new 文本: |
   - 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。
   - **工厂签名 diff 核对（集成前零成本防线）**：对 baseline 与 perf_opt 的同函数工厂签名段做 sed+diff 比对（`diff <(sed -n '/^def {func}/,/^):/p' <baseline> <perf_opt>)`）——签名逐参一致时 wrapper 胶水零改动即可切换，不一致时先对齐签名再翻转（2026-09-17 ssd 实证：SIGNATURE_IDENTICAL）。
   - **config 口径衔接（翻转切换块激活 perf_opt 时）**：核对 wrapper `default_config` 与调优版 kernel 内嵌 tuned 默认（TUNED_DEFAULT_CONFIG 类模块级常量）——custom_op 显式传参优先级高于内嵌默认，wrapper 旧 config 会静默覆盖 tuned config 使 Stage 4 调优集成态不生效；二者取一（改 wrapper 引用 tuned 常量，或保持 wrapper 默认并在 integration_log 标注口径差异与不可同口径对比的 bench 边界，2026-09-17 ssd 形态）。
    动机: Stage 4 调优结论与 Stage 5 bench 数值的可比性依赖 config 口径一致；本任务 bench 11 dispatch 全部跑在 wrapper 默认 config 上（与 tuned 不同口径），不核对则调优收益在集成态静默丢失且无告警。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0094
- type: R
- title: 关键决策引用须抄录一手来源限定条款原文（形态/数字/状态三类均适用）+ §3/§4/§6 三方对读细化为可 grep 的机械对读点
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 1 revision 章节——v0 三个阻塞错误共同根因 = 引用证据未打开一手来源核对：① 列因子形态凭直觉写 [bs,1]（未对照 T.vmul.md §2.2.2 四形清单）；② intra FLOPs 手算 6.46G 未与 manifest 权威公式 9.7G→6.44G 交叉核对（2× 高估直送审）；③ PL-1.11 只看 kb_stale_check 全局计数即降级（未读条目自身 status + 重验行）——形态/数字/状态三类各踩一个；另 v0 §3.3 伪代码与 §6.1 循环表、§4.5 预算表与 §3.3 buffer 使用两处自相矛盾（成文未做三方对读终检，质量清单 #21 执行流于形式）
- repro: 复现条件——任一引用 docs/pattern-library/manifest 三类信息做关键决策的设计期（本任务 3 阻塞 + 2 内部矛盾实证）
- toolchain_stamp: 设计层规则（无运行时依赖）；证据环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 3 步骤 0.5 段落之后追加一条机械规则）
    定位锚: "> 迁移任务的额外信息源：Phase M0/R/M1 的源算子解读、算法调研结论与迁移决策（优先级介于 `examples/` 同类实现与外部参考实现之间——迁移决策界定"算法该怎么设计"，`examples/` 界定"API 怎么用"）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      > 迁移任务的额外信息源：Phase M0/R/M1 的源算子解读、算法调研结论与迁移决策（优先级介于 `examples/` 同类实现与外部参考实现之间——迁移决策界定"算法该怎么设计"，`examples/` 界定"API 怎么用"）。

      > **引用抄录规则**：关键决策引用的每个 docs / pattern-library / manifest 来源，引用时须抄录其限定条款原文——三类信息都是被引用对象：**形态类**（shape 支持清单，如 T.vmul.md §2.2.2 四形）、**数字类**（实测倍数/复杂度，须与 manifest 权威公式双源交叉）、**状态类**（条目 front-matter status + 重验戳）。「负向断言须引用条款」已有，正向选型引用同样适用（2026-09-17 ssd 实证：v0 三个阻塞错误恰好各踩一类——形态凭直觉、数字单源手算 2× 高估、状态只看全局计数）。
    动机: 配套（同语义扩散，审批时可一并裁决）：Phase 2 产出规范「§1.6.3 布局决策与 §3.3/§4/§6 三方一致」细化为三个可 grep 的机械对读点——buffer 名集合（§3.3 vs §4.5 vs §6.1）、循环 API 形态（§3.3 vs §6.1）、因子 shape 形态（§3.2/§3.3 vs §1.6.3）——对读点可机械化后才不流于形式（本任务 v0 两处自相矛盾均属三方对读缺失）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0101
- type: R
- title: 知识预注入 stale 条目消费两分法——结构性结论（有 HEAD 生产代码佐证）可继承为主选、硬边界结论（编译器/容量/支配性）必列 Stage 3 重验清单
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 first_design Skill Flow Issues 首行 + DESIGN.md §0.6 引言 / §9.2 R-1（两分法消费全记录：PL-1.16/1.12/1.13/1.14 结构性继承 + CG-2026-0010/0011/0012、UB 容量标待重验）
  - 有效性验证：3 轮检视均判定 stale 引用处置恰当；Stage 3 重验四项全过（pass_configs 双关闭 / UB 记账 / 旧 verified 结构首编即过 / golden）；Developer fallback 设计直接跳过（VP-2026-0091 知识生效先例）
  - 交叉引用：queue VP-2026-0089（stale 判定口径——如何判 stale；本条为判后如何消费，互补互链）
- repro: 复现条件——工具链升级后重跑已知先例的迁移/重做任务（本次 4515de8 vs 旧任务 1990aa9fe4：E-1 预注入条目全部标 stale）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（步骤 0.5 段落末尾追加）
    定位锚: "该库是任务实测的积累（如核内融合转置链实测代价、C 轴切片累加已验证形态、跨步系数阻碍向量化案例），其优先级高于 docs 规格与 examples 先例（见 info-sources.md 优先级表）；§1.6.3 的候选枚举与代价模型必须先对照该库，库内已有的实测数字不得当作"未实证常数"重新假设。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + "**stale 条目两分法消费**（工具链升级后重跑先例任务时；2026-09-20 ssd 重跑实证——3 轮检视收敛 + Stage 3 重验四项全过）：结构性结论（双 Scope/流水深度/分片/布局类，有 HEAD 生产代码或先例 kernel 佐证）可继承为主选结构；硬边界结论（编译器行为/容量上限/pass 支配性类，CG-2026-00xx 与 TRAP-*/CONST-* 条目）必须列入 Stage 3 首日重验清单——避免全盘照抄（旧结论失效风险）与全盘重验（丢先例红利）两个极端。条目是否 stale 以条目自身 front-matter status + 重验行为准（VP-2026-0089），全局 stale_count 不作单条目判定依据。"
    动机: 重跑任务的预注入条目在工具链升级后全部标 stale——无消费指引时要么盲信旧结论、要么整体降级重推；两分法使本次重跑零无效重设计（Developer fallback 直接跳过）且硬边界全部实测重验。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0102
- type: R
- title: 迁移重做任务检视前置一步——先 `git status --short -- examples/` 识别 Stage 0 脚手架删除的旧工件，先例引用一律经 `git show HEAD:<path>` 核对提交版本（仅查工作树会误判「引用不存在」或漏核关键先例）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 first review Process 行：Stage 0 脚手架删除旧任务集成工件（`ssd_chunk_scan_kernel/` 整目录 D 状态），而设计大量引用旧集成版/perf_opt 终版作生产先例——host-cast 先例、wait 置位 verified 形态均只存于 git；REVIEW.md 元信息「工件注入披露」+ 问题 1/7 的 git 证据链
- repro: 复现条件——任一迁移重做任务（Stage 0 脚手架重建清理旧任务集成产物）的 Stage 2 检视
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 步骤 2 扩展）
    定位锚: "2. **迁移任务**（`DESIGN.md` 含 §0）：Read `source_op_path` 指向的源算子代码全文；§0 与源码不一致的项在维度 0 中逐条记录证据（源码行/语句 + DESIGN.md 章节）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      2. **迁移任务**（`DESIGN.md` 含 §0）：Read `source_op_path` 指向的源算子代码全文；§0 与源码不一致的项在维度 0 中逐条记录证据（源码行/语句 + DESIGN.md 章节）。**迁移重做任务附加**：先 `git status --short -- examples/` 识别 Stage 0 脚手架删除的旧任务工件（整目录 D 状态常见）——设计引用的旧集成版/perf_opt 终版等先例只存于 git，一律以 `git show HEAD:<path>` 核对提交版本；仅查工作树会误判「引用不存在」（维度 7 假 fail）或漏核关键先例（2026-09-20 ssd 重跑任务：host-cast 先例与 wait 置位 verified 形态均仅存于 git）。
    动机: 重做任务的先例引用合法性无法在工作树核验——不识别删除态会把合法引用误判为断链，或让设计的关键先例逃过核对。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0103
- type: R
- title: design SKILL Phase 5 伪代码自检两条——① Expert 双域伪代码逐 flag 令牌推演（set 序×wait 序×写序位置配对，并与 verified 实现行级比对）② 设计自定 L0 边界用例的伪代码形状走查（非整除/P_tiles≥2/退化域逐一过 copy/gemm src/dst 形状与循环变量绑定）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 revision 两轮 Skill Flow Issues：v0 尾置 wait 在 §0.6 R7 论证文字正确的情况下被写出（论证与伪代码脱节——WAR 竞争逃逸 Phase 5 自检、Stage 2 令牌推演才检出，REVIEW.md 问题 1）；v0 x 装载（0:P 全宽、pp 循环外）与 Vector 域 pp 未绑定只在 P=64（当前 workload 全集）侥幸成立——P=128 是设计自定 L0-2 门禁用例却未过伪代码走查（REVIEW.md 问题 2/3）
- repro: 复现条件——任一 Expert 双域流水设计的 Phase 5 自检（本任务两类缺陷均逃逸自检、检视期拦截）
- toolchain_stamp: 设计层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 5 段末追加两条自检）
    定位锚: "按照 [references/quality-checklist.md](references/quality-checklist.md) 中的自检清单逐项检查，确保文档质量。**所有任务必须通过算法级检查项（算法调研完整（§1.6.0 调研四问齐全、结论有依据）/ 数学等价优化分析完整 / 向量化替代分析完整）**；**迁移任务必须额外通过清单中的迁移专属检查项**（源算子解读完整性 / 耦合性判定有依据 / 重设计有语义保持论证 / §1–§7 与迁移决策一致）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + 两条新增自检：
      > **Expert 双域伪代码令牌推演**：对伪代码中每个 `sync_block_set/wait` 做令牌推演（同 id 的 set 序与 wait 序按事件计数逐对配对 + 受保护对象的写发生在 wait 之后的哪个位置），并与引用的 verified 实现行级比对——论证文字正确而伪代码尾置 wait 的脱节形态无法靠通读发现（2026-09-20 ssd 重跑任务 v0 实证：WAR 竞争逃逸 Phase 5、Stage 2 才检出）。
      > **伪代码形状走查**：对设计自定的 L0 边界用例（非整除 / P_tiles≥2 / 单 tile 退化域）逐一在伪代码上走一遍 copy/gemm 的 src/dst 形状与循环变量绑定——「当前 manifest workload 全整除/全 P=64」的侥幸域会掩盖形状矛盾（同任务 v0：P=128 是自定门禁用例却未走查，形状矛盾逃逸到检视期）。
    动机: 两类缺陷（协议脱节/形状矛盾）都属「通读自检必然通过、机械推演才暴露」形态；走查域取设计自定的 L0 用例使自检与测试计划同源。与 VP-2026-0029（检视侧维度 5 行）构成设计自检↔检视双防线。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0104
- type: R
- title: 修订「写实占位伪代码」对照纪律——写实依据 = 占位符对应 verified 实现的真实形态（逐行比对搬运方向/buffer 数）；写实引入新搬运原语前置三关检查（docs 方向表在列 + examples 先例 + 最小编译探针）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 revision_index=2 Skill Flow Issues：v1 修问题 5① 时把 band 组装从占位式（`T.gemm(l1_lcb_band...)`）写实为自创 `T.copy(l1_lcb, l1_band[...])` L1→L1 两跳——方向表无此方向、全仓零先例、检视探针实测硬失败（'Unsupported copy from cbuf to cbuf!'）；旧 verified 实为单 buffer GM→L1 列偏移直载；DESIGN.md v1→v2 §1.4 对比（L347–353 两跳 → 单跳）
  - 错误谱系：v0 协议/形状缺陷 → v1 写实自创原语 → v2 清零（「写实化」是修订链最后一个高危环节）
- repro: 复现条件——任一修订轮的「把占位伪代码写实/具体化」动作（本任务该环节引入唯一新阻塞）
- toolchain_stamp: 设计层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 6 段末追加）
    定位锚: |
      ### Phase 6：针对性修订

      仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
    old 文本: （即上述定位锚原文）
    new 文本: |
      ### Phase 6：针对性修订

      仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

      > **「写实占位伪代码」对照纪律**：修订把占位伪代码写实/具体化时，若该占位符对应某个 verified 实现（HEAD 旧终版/先例 kernel），写实的唯一依据是该实现的真实形态——逐行比对搬运方向与 buffer 数，不在「让它更具体」的名义下发明 verified 实现里不存在的结构（两跳/中间 buffer）；写实引入**新搬运原语**时前置三关检查：docs 方向表在列 + 全仓 examples 先例 + 最小编译探针（~1–2 分钟）——任一关不过即回退到 verified 形态（2026-09-20 ssd 重跑任务实证：写实自创 L1→L1 拷贝，探针硬失败，一轮修订白费）。
    动机: 修订链的「写实化」环节是新增缺陷高发区；三关检查使原语合法性在设计期闭合，不留给检视兜底（同任务检视侧的编译探针手段见 VP-2026-0105）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0105
- type: R
- title: design-review Phase 1 第 4 步补「编译探针」为可选决定性手段——API 可行性争议时编写最小 kernel 仅编译不运行（~1–2 分钟/个），同时探针「设计形态」（证伪）与「建议修复形态」（证可行），使阻塞问题的修改建议自带可行性证明
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 复检轮：问题 1 处置全程（探针 1 证伪 L1→L1 两跳、探针 2 证单跳直载可行——修复建议自带编译级证明，Stage 3 零试错）；REVIEW.md（复检轮）机械复核段「编译探针」表；第三轮探针 2 复跑再证（"COMPILED OK"）
- repro: 复现条件——设计形态无文档/先例佐证的 API 可行性争议（本任务 L1→L1 copy：方向表无此方向 + 全仓零先例）
- toolchain_stamp: 检视层规则；证据环境 tilelang 0.1.2+4515de8 / CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 4 步扩展）
    定位锚: "4. 必要时 Grep `tilelang/language/` 确认 API 是否有导出佐证（仅静态文本核对，不执行）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. 必要时 Grep `tilelang/language/` 确认 API 是否有导出佐证（仅静态文本核对，不执行）。**编译探针（可选决定性手段）**：设计形态/修复建议无文档与先例佐证、或负向断言需实证时，编写最小 kernel（仅含争议原语，Expert/Developer 按设计模式）**仅编译不运行**（~1–2 分钟/个）——同时探针「设计形态」（证伪）与「建议修复形态」（证可行），使阻塞问题的修改建议自带可行性证明（2026-09-20 ssd 重跑任务实证：探针 1 证伪 L1→L1 拷贝 + 探针 2 证单跳直载可行，修订一轮收敛、Stage 3 零试错；探针代码与报错全文内嵌 REVIEW.md 附录）。
    动机: 「检视只看文档不编译」使 API 可行性争议只能先例背书或留待 Stage 3 暴露——编译探针把该类争议在检视期闭合。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0106
- type: R
- title: design-review Phase 1 补充核对三则——① 同结构多表述位一致性（§1.4/§3.2/§3.3/§6.2/§4.4 平行表述位作为一个核对单元，占位符/写实化位置是重点风险区）② 复杂度表 ws/中继流量按声明形状全维乘积复算（per-task 口径）③ 数字联动核对含标定域外的外推/假设句
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z：① Stage 2 复检轮 Review-method 行（v0 检视只标 §1.4 占位式命名，漏 §3.2 步骤 7 口径矛盾——v1 写实后才暴露为字面结果错误）② Stage 2 first review 问题 4（R3 表 ws 流量 per-l-tile 口径低估 2.5×：87 vs 216MB@w2，独立复算拦截）③ Stage 2 第三轮维度 2 观察项（§4.5「Q=512 → 144KB」外推行漏改〔应为 160KB〕，两轮均未联动——非当前 workload、结论方向不变）
- repro: 复现条件——任一多章节 DESIGN 检视（三则均本任务三轮实证：字面结果错误 / 2.5× 流量低估 / 外推数字漏改）
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步之后追加第 8 步）
    定位锚: "   - 两项机械结果摘要（含 skip 项）附入 REVIEW.md「机械复核」段——与维度 8 的「独立推演」互补而非替代。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + 新步骤：
      8. **补充核对三则**（2026-09-20 ssd 重跑任务三轮检视实证）：
         - **同结构多表述位一致性**：同一结构（协议/搬运/循环）在设计文档的全部平行表述位（§1.4 伪代码 / §3.2 API 映射 / §3.3 / §6.2 循环表 / §4.4 路径图 / §4.5 预算表）作为一个核对单元整体检视——占位符与「写实化」位置是重点风险区（本任务 §3.2 步骤 7 的口径矛盾在 v0 只被读作命名占位，v1 写实后成为字面结果错误）；
         - **复杂度表流量复算口径**：ws/中继类流量按声明的 ws 形状全维乘积（per-task 口径）复算，不得采信 per-l-tile/per-AIV 局部口径（本任务 R3 表曾低估 2.5×：87 vs 216MB@w2——流量项为地板时会误导 Stage 4 目标）；
         - **外推/假设句数字联动**：数字联动核对须含标定域外的插值/外推句（grep 容量单位与「→」外推句式——本任务「Q=512 → 144KB」两轮漏改，应为 160KB）。
    动机: 三则都是「逐维度检视通过、整体对读才暴露」的缺口——第一则曾放过一个字面结果错误到复检轮，后两则靠人工复算拦截。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0107
- type: R
- title: develop SKILL 两条——① Phase 3 golden 落盘形态 = 函数体内联 torch 计算（薄包装/别名导入被 S3-GOLDEN-TORCH AST 检查拒绝）② Phase 4 首跑建议 --level L0 单层（逐 shape 可见编译结果，定位后切 all）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 3：① gate 3 报 S3-GOLDEN-TORCH（薄包装 `return ssd_chunk_scan_fwd_ref(...)` 函数体无 torch.* 调用——attempt 1 白耗 742s）；内联仓内参考的 torch 计算（纯 fp32 einsum 双路径 + tril 掩码 materialize 照抄）后 pass: true；② L0 各用例 shape 不同触发多次编译、逐一可见 "AscendNPU IR compile success"（本任务 L0 首跑全绿省去分层二分）
  - 〔2026-09-28 新亚型（不同任务，reduction 族），task argmax-_argreduce_kernel-20260924T031005Z Stage 3〕**张量方法形态**：golden 用张量方法 `x.argmax(dim=...)` 而非模块函数 `torch.argmax(x, dim=...)`——函数体同样无 `torch.*` 前缀调用，S3-GOLDEN-TORCH AST 检查判 fail（attempt 1 runtime fail 2731s → 修复 attempt 2 complete 70s，修法 = torch.* 命名空间直调）。建议 apply 时在 new 文本 1 的第 3 条补一句：「张量方法调用（`x.argmax`/`x.max` 等）同样不含 torch.* 前缀，须改写为 `torch.argmax(x, ...)` 模块函数直调（2026-09-28 argmax 实证：2731s 白耗）」；出处：examples/argmax/SESSION_TIMING_ANALYSIS.md §3 第 3 条 + .task_timeline.jsonl Stage 3 fail 事件
- repro: 复现条件——① 迁移任务复用仓内参考函数作 golden（`import ... as` 别名与薄包装委托两种形态均被拒）；② 任一多 shape L0 套件首跑
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20；gate_lint S3-GOLDEN-TORCH 现行版本
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（两处）
    定位锚 1: "2. Golden 必须在 CPU 上可独立运行（不依赖 torch_npu）。"
    old 文本 1: （即上述定位锚 1 原文）
    new 文本 1: |
      2. Golden 必须在 CPU 上可独立运行（不依赖 torch_npu）。
      3. **golden 函数体必须内联 torch 计算**（迁移任务复用仓内参考函数时的正确落盘）：把参考函数的 torch 计算逐段抄进 `def golden_...` 函数体（函数体直接含 `torch.einsum/tril/exp` 等调用）——`import ... as` 别名或 `return ref(...)` 薄包装的函数体无 torch.* 调用，会被 gate `S3-GOLDEN-TORCH` 的 AST 检查判 fail（2026-09-20 ssd 重跑任务实证：薄包装被拒白耗一次 attempt 742s；内联后语义与参考逐位一致，且无需 sys.path 指向参考目录）。
    定位锚 2: "1. 跑 L0：`python {op}.py --level L0`。"
    old 文本 2: （即上述定位锚 2 原文）
    new 文本 2: |
      1. 跑 L0：`python {op}.py --level L0`。（Expert 双域 kernel 首跑建议保持单层 L0——各用例 shape 不同会触发多次编译，逐一可见编译结果，定位到具体 shape 的编译失败后再切 `--level all`。）
    动机: golden 形态是 gate 交互问题（AST 检查只认函数体内的 torch.* 调用，委托形态语义等价也拒）；L0 分层是零成本定位习惯。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0108
- type: R
- title: develop SKILL Phase 2 补「迁移重做任务实现基线」——先 `git show HEAD` 取旧 verified 终版（perf_opt/）作结构模板逐行复刻，只做规格差异适配（pass_configs 按新规格 / golden 内联 / 测试套件按新 L0 计划）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 3：实现直接复刻 `git show 4515de8:examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py` 完整结构形态 + 三处差异适配——首次编译即通过、零调试往返（L0–Boundary 全绿，比从伪代码重推省 ~90% 时间）；RETROSPECTIVE.md Stage 3 Value Point Proposals 第 3 行 + Transferable 首条（后续 mamba 族未迁移成员直接以该文件为结构模板）
- repro: 复现条件——迁移重做任务（同 op 旧 verified 终版存在于 git HEAD、工作树副本被 Stage 0 删除）的 Stage 3 实现
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 清单追加第 4 项）
    定位锚: "3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
      4. **迁移重做任务**（同 op 旧 verified 终版在 git HEAD、Stage 0 已删工作树副本）：实现基线 = `git show HEAD:<旧 perf_opt 终版路径>` 的完整结构逐行复刻（每个搬运方向/flag 置位/slice 裁剪已在旧工具链实证），只做规格差异适配——pass_configs 按新 DESIGN 规格、golden 内联仓内参考、测试套件按新 L0 计划扩充；从伪代码重推是下策（2026-09-20 ssd 重跑任务实证：复刻 + 三处适配首编即过零调试往返，省 ~90% 时间）。
    动机: 重做任务的最大红利是旧 verified 实现（结构层每个 API 形态已验证）——不取用即放弃；差异适配清单使复刻不退化为盲抄（规格演进点显式处理）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0109
- type: R
- title: integrate_kernel.py rewrite_wrapper_import 同步清理被替换 import 正上方的「Part A 提取说明」陈旧注释——避免描述已死 import 的注释误导后续维护者
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 5：examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan.py L56–L58（"# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py / # (pattern B) from {gpu_repo_root}/... / # ---" 注释块保留在 baseline/perf_opt 选择块之上，描述的是已被替换的 GPU 提取 import——且 {gpu_repo_root} 占位符未填充）；op 级 RETROSPECTIVE.md Stage 5 Skill Flow Issues 首行
- repro: 复现条件——任一 wrapper 带脚手架提取说明注释的集成（rewrite_wrapper_import 替换 import 后注释残留）
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir / 2026-09-20；脚本现行版本
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（rewrite_wrapper_import 拼接处追加注释清理）
    定位锚: |
      (rewrite_wrapper_import 函数体内)
          block = gen_kernel_source_block(names, integrated_files, op_slug)
          text = text[: m.start()] + block + text[m.end() :]
          wrapper_path.write_text(text, encoding="utf-8")
    old 文本: |
          block = gen_kernel_source_block(names, integrated_files, op_slug)
          text = text[: m.start()] + block + text[m.end() :]
          wrapper_path.write_text(text, encoding="utf-8")
    new 文本: |
          block = gen_kernel_source_block(names, integrated_files, op_slug)
          # Drop the stale "Part A: ... extracted ..." comment block sitting
          # directly above the replaced import: it documents the dead
          # GPU-extraction import (with unfilled {gpu_repo_root} placeholders)
          # and misleads maintainers after the swap (2026-09-20 ssd case).
          pre = text[: m.start()].rstrip("\n")
          part_a = re.search(r"\n(?:#[^\n]*\n)+$", pre)
          if part_a and "extracted" in part_a.group(0):
              pre = pre[: part_a.start()].rstrip("\n") + "\n\n"
          text = pre + block + text[m.end() :]
          wrapper_path.write_text(text, encoding="utf-8")
    动机: 脚手架生成的提取说明注释在 import 被替换后成为死文档；重跑/换代集成时该注释与新的 baseline/perf_opt 切换块并排出现，误导 wrapper 维护者（该文件是 wrapper 胶水的唯一权威说明位）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0111
- type: R
- title: iteration-diagnosis.md Step 2「代码结构观察」补机械互证检查——继承的段数/流量记账须做「引擎指令数 ÷ 任务数」与 kernel 拷贝语句清单逐项对照（指令数是装载冗余的直接探测器，比散文记账可靠）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log.md §1 末节（Phase 1 代码结构新发现：prev_states 切片对 lt 无依赖却在 lt 循环体内 ⟹ Q=256 时 4× 重读 = 48KB/任务冗余 = Cube mte2 字节 20%；实测 19.0 指令/任务 = 1(x)+Σ_lt[1(ws_c)+1(prev)+(lt+1)(band)] 与代码结构精确互证）
  - 误判代价实证：旧任务（1990aa9fe4 谱系）两套段数拆分共同漏算 ×4（「1216 段不可减」的地板定位错误），本轮据此再挖出几何 1.078× 中的第一项（prevhoist −2.0~−3.4%）；constants.md CONST-mte2 段数更正注 + attention.md PL-1.18 update①（2026-09-21 已回写）
  - .task_timeline.jsonl（Stage 4 无 fail 于该项——发现靠 optimizer 自发代码结构核对，非流程强制；Skill Retrospective Skill Flow Issues 首行自评「旧档数字未被要求做代码结构级互证」）
- repro: python3 .agents/tools/kb_search.py "mte2 段数 指令数 互证"（数据侧落点：constants.md CONST-mte2 / attention.md PL-1.18）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 代码结构观察」bullet 列表末条之后追加一条）
    定位锚: "- 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。"
    old 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。
    new 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。
      - 装载冗余互证（含继承数字）：对每个搬运引擎把「指令数（如 aic_mte2_instructions）÷ 每核任务数」与 kernel 内拷贝语句清单逐项对照——每条 T.copy 对应一条指令，对不上即有漏算/冗余；逐维核对循环体内张量索引的循环不变性（索引不含某循环变量 ⟺ 该装载可提升出该循环）。从旧档/前轮继承的段数/流量记账在用于地板定位前必须过此互证（2026-09-21 ssd 实证：两套旧拆分共同漏算 prev ×4 重读，致「1216 段不可减」错误地板定位跨任务存活，指令数互证一轮发现 20% 可削字节）。
    动机: 散文式段数记账不可机械复核，错误记账一旦进入「继承裁决」链就跨任务存活并封死候选方向；指令数是 msprof 直接给出的机械计数，与代码结构一一对应——把互证设为 Step 2 标准动作可在首轮 profile 即发现记账盲区，而不是依赖 optimizer 自发核对。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0112
- type: R
- title: iteration-diagnosis.md「已知迭代结论」节补注——引擎临界性是动态量，「引擎 X 非关键」类结论每轮 winner 合并后须重算（AIV「非关键」在两轮 Cube 优化后失效、w2 转 AIV-bound）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log.md R2→R4 约束迁移链（prevhoist+l0c2x 后 w2 任务周期 6.58µs vs Cube mte2 busy 4.45µs/任务 ⟹ AIV-bound；vbrchoist 据此立项获益 −3.45%）+ Skill Retrospective Skill Flow Issues 第 2 行（「AIV『非关键路径』结论会随 Cube 侧优化失效」）
  - 知识侧姊妹条目：queue VP-2026-0110（BP_engine_criticality_drift，Tier 1——方法本体与判据；本条为流程侧每轮强制重算规则，两案合入时互链）
  - 〔2026-09-21 重建会话同谱系追加（不计确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z〕方向翻转级证据：vbrchoist 单点 H=80 族回退（−4.2~−6.7%）在 prevhoist 合入后消失（+2.2~+10.2%）——单点结论依赖 base 状态的更强形态（数据侧 attention.md PL-1.18 update②）；**建议 apply 时在同节邻位增补「累积序检验」一句**：与先验证据矛盾的 workload 族分裂先合入对侧已验证胜出复测、再谈 per-shape 条件路径（姊妹 BP 提案 VP-2026-0115，合入时互链）
- repro: none（流程规则；核算方法与实证数据自包含于 VP-2026-0110 delta 与 attention.md PL-1.18 update③）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 已知迭代结论」text block 之后追加一段）
    定位锚: |
      本轮 profile 相比上一轮 profile 哪些指标发生变化
      ```
    old 文本: |
      本轮 profile 相比上一轮 profile 哪些指标发生变化
      ```
    new 文本: |
      本轮 profile 相比上一轮 profile 哪些指标发生变化
      ```

      **引擎临界性是动态量（2026-09-21 ssd 实证）**：双引擎（Cube+Vector）kernel 的「引擎 X 非关键路径」结论只对当时的结构状态有效——每轮 winner 合并后须重算逐 workload 的「任务周期（Task Duration ÷ 每核任务数）vs 各引擎 per-task busy」核算，临界性反转时重启此前因「非关键」被否决的对侧候选（ssd：AIV「非关键」在两轮 Cube 优化后失效，w2 转 AIV-bound，Vector 侧 hoist 重新获益 −3.5%）。重算方法与判据见 bottleneck-patterns.md BP_engine_criticality_drift。
    动机: 旧结论按字面继承会把「当前非关键」固化为「永远非关键」，直接封死对侧候选池；重算成本是一次除法对照，收益是整轮优化方向（本任务 vbrchoist 即靠该核算立项）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0113
- type: R
- title: ab_test 交错协议判读规则——3-pair 符号检验统计功效不足（3/3 同向仍 p=0.25 判 tie），采纳判据 = 配对方向一致 + 机制佐证（资源占用严格更优），临界情形 --pairs 5 复测
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log R1 ab_test（ab_round1_w4：3/3 pair 一致 −2.85/−2.27/−3.02%，合并 −2.84%，verdict 仍 tie——p=0.25 为 3-pair 符号检验固有功效上限）+ R4（ab_round4_w2：3/3 一致 −2.97% 同判 tie）+ R2（ab_round2_w3：B_better 3/3 −3.0~−4.0% 合并 −3.97%——幅度够大时协议能判）
  - 本任务正确采纳路径：靠 iteration-diagnosis.md Step 6 既有「资源占用决胜」规则 + 机制佐证（mte2 指令 19→16/任务、Cube mte2 字节 −20%、busy 3289→2913µs −11.4%）采纳 prevhoist——判据形态有效但未在 T-2 协议条成文，本轮为自发正确应用
  - 工具侧查重确认：`.agents/tools/ab_test.py` 已支持 `--pairs`（L108，默认 3）——缺口在判读协议文档，非工具
  - **第三数据点（不同任务，2026-11-01，task TileOPs-_make_rope_neox_1d-20261031T184833Z，**不计确认**——Tier 2 无计数，仅补证据面）**：同一 artifact 单 run 两次测 3.58 / 3.71 µs（+3.6%），交错 A/B（3 pair）判 `tie`（3.56 vs 3.59，pair 差 +0.09/+0.03/−0.01 不同向）——印证 3-pair 判 tie 不等于「有回退」；同任务 8k 档 3/3 同向 −0.25/−0.14/−0.20（合并 −3.45%）判 `B_better`。**该任务的经验补充**：T-2 交错裁决在**每个** <5% 差异上都应执行，不能只在「打算采纳」时执行（否则单 run 噪声会以「回退」形态写进 Final 表：本轮 2k-d64 的 Final 表 +3.9% 即属噪声带，非真实回退）。
  - **第四数据点（2026-11-04，task TileOPs-_make_rope_neox_1d-20261104T065720Z，Tier 2 无计数）**：`r4_prefetch` 单 run 粗筛 4k 4.89→4.76（+2.7%）、8k 6.41→6.30（+1.7%）「看似增益」；5-run 交错 A/B 后 4k **4.73→4.86（+2.75%，更慢）**、8k 6.28→6.24（−0.64%，tie）、2k 3.63→3.57（tie）。同一 artifact、同一 session 内单 run 方向与 A/B 相反 ⇒ 再次印证「任何 <5% 单 run 差异必须先 A/B，且只在 A/B 后再作采纳/否决依据」。
- repro: python3 .agents/tools/ab_test.py --a <baseline> --b <candidate> --pairs 5 ...（工具现成；复现条件 = 任一 <5% 增益的 3-pair 交错协议输出 tie 而 pair 方向全一致）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（Step 6 交错 A/B 协议 bullet 末尾追加判读规则）
    定位锚: "工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。"
    old 文本: |
      工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。
    new 文本: |
      工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。**判读规则（2026-09-21 ssd 实证）**：3-pair 符号检验的统计功效上限 p=0.25——**3/3 配对同向仍判 tie 是协议固有，不是「无差异」证据**（两例：3/3 一致 −2.84% / −2.97% 均判 tie）；此形态的采纳判据 = ① 配对方向全一致（≥3/3 同向）**且** ② 机制佐证独立成立（资源占用严格更优：指令数/忙比/字节数等机制性削减指标，与 Task Duration 无关的计数）——两条件同时满足可采纳（本任务 prevhoist：pair 全同向 + mte2 指令 19→16/任务 + busy −11.4%）；仅方向一致而无机制佐证时用 `--pairs 5` 复测提高功效，不得据 3-pair tie 丢弃。
    动机: tie 判定若无配套采纳规则，3/3 同向的真增益会被当噪声丢弃（或反之被主观采纳）——机制佐证把「统计上未决」转化为「机制上可判」，且佐证指标（指令数/字节）不受 run 双态噪声影响。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0114
- type: R
- title: signal-registry.md §5 S4 产物格式契约显式化——perf_records final 的 artifact_path 以 perf_opt/ 为基准（裸文件名），Final Performance Test Data 表为 ##/### 标题 + 恰 5 列纯数字（gate 4 机械解析形态）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：.task_timeline.jsonl（Stage 4 attempt 1 fail 7681s，verdict=runtime，2026-09-21T02:43:32Z——gate 4 S4-PERF-RECORDS-RECON 对账失败；attempt 2 窄修复 complete 170s）——失败细节来自 conductor 终态钩子输入，opt_log §7 复盘未覆盖该次 fail 的 gate 侧原因（复盘缺口形态，同 stats ⑧⑩ 先例）
  - 两处产物格式错误（attempt 1）：① perf_records.jsonl final 记录 artifact_path 用仓库根相对路径（examples/TileOPs/... 前缀），gate 以 perf_records.jsonl 所在目录（perf_opt/）为基准拼接成双重前缀 → abspath 不匹配 → 「final 未测最终 perf_opt 文件」；② opt_log Final Performance Test Data 用加粗列表项而非 ##/### 标题 + 6 列且数值加粗——gate_lint.py L908 要求 `^#{2,3} Final Performance Test Data\s*$` 标题、L919 要求恰 5 列、L925 要求 float() 可解析（加粗 → ValueError / 6 列 → 行被跳过 → 「Final 表缺少 {key}」）
  - 修复形态（attempt 2，170s）：final 行 artifact_path = 裸文件名 `_ssd_chunk_scan_fwd_kernel.py`（perf_records.jsonl 末 3 行）+ opt_log §6 `### Final Performance Test Data` + 5 列纯数字表
  - **第四亚型（2026-10-30 engram_decode round 3，task TileOPs-_engram_decode_kernel-20261030T145026Z）**：`artifact_path` / 标题 / 列数 / 纯数字均已合规，仍被 `S4-PERF-RECORDS-RECON` 判失败——**Final 表首列 `kernel_id` 写成了缩写** `…::_engram_decode_kernel::main`，而 gate 以 `(inventory.kernel_id, workload_id)` 为 key 逐字比对（`gate_lint.py` L840 `key = (item["kernel_id"], item["workload_id"])` vs L1071 `claims.get(key)`），缩写形态使 claims key 不匹配 → 报「Final 表缺少 {key}」；改为 workload_inventory.json 的完整字面量 `examples/TileOPs/tileops/kernels/.../_engram_decode_kernel.py::_engram_decode_kernel::main` 后一次通过（attempt 2 极短）。同轮另有一处记录重建（docstring 修改导致 final 哈希失配 → 把旧记录集原样移入 history/ 后重采 3 条 final），非 gate 失败原因
  - .agents/tools/gate_lint.py L882-884（相对路径 join dirname(records_path)）/ L908-927（标题正则 + 5 列 + float 解析）；现行 signal-registry.md §5 已列 5 列字段名但未写标题层级/纯数字/路径基准三要素
- repro: python3 .agents/tools/gate_lint.py（现行 S4-PERF-RECORDS-RECON 规则；负例 = final 行 artifact_path 带仓库根前缀 + Final 表加粗 6 列形态，正例 = 本任务 attempt 2 修复后产物）
- toolchain_stamp: gate_lint.py 现行版本（statectl gate 4 机械核对层）；发生环境 tilelang 0.1.2+15ad002b3d / CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/_shared/standards/signal-registry.md
- delta: |
    动作: update（§5 字段契约段扩展一句）
    定位锚: "`phase` 为 `baseline` / `candidate` / `merged` / `final`。`artifact_path` 指向实际测量的候选 kernel 文件，`artifact_sha256` 是采集时该文件的 SHA256；final 记录的路径和哈希须与最终 `perf_opt/{op}.py` 一致。"
    old 文本: |
      `phase` 为 `baseline` / `candidate` / `merged` / `final`。`artifact_path` 指向实际测量的候选 kernel 文件，`artifact_sha256` 是采集时该文件的 SHA256；final 记录的路径和哈希须与最终 `perf_opt/{op}.py` 一致。
    new 文本: |
      `phase` 为 `baseline` / `candidate` / `merged` / `final`。`artifact_path` 指向实际测量的候选 kernel 文件，`artifact_sha256` 是采集时该文件的 SHA256；final 记录的路径和哈希须与最终 `perf_opt/{op}.py` 一致。**`artifact_path` 相对基准 = `perf_records.jsonl` 所在目录（`perf_opt/`）——final 记录写裸文件名 `{op}.py`，仓库根相对路径会被 gate 以 `perf_opt/` 为基准拼接成双重前缀而判「final 未测最终 perf_opt 文件」（2026-09-21 ssd 二轮调优实证：Stage 4 attempt 1 整轮 gate 对账失败重试）**。Final Performance Test Data 表的机械解析形态：`##`/`###` 标题行 + 恰 5 列 `| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |`，时延列为纯数字（不加粗、不用列表项替代表格行——6 列或加粗数值均使 gate 4 `S4-PERF-RECORDS-RECON` 解析失败或跳行）。**`kernel_id` 列须与 `workload_inventory.json` 的 `kernel_id` 字符串逐字相同（含仓库相对全路径与 `::kernel::main` 后缀）**——缩写（如 `…::_engram_decode_kernel::main`）会使 gate 的 `(kernel_id, workload_id)` claims key 不匹配并报「Final 表缺少 {key}」（2026-10-30 engram_decode round 3 实证，本条第四亚型）。
    动机: 契约的字段名已成文但「路径基准 / 标题层级 / 纯数字」三个 gate 解析要素未写——首轮按直觉写仓库根相对路径与富文本表格会整轮 gate 失败（本任务 7681s attempt 的收尾对账失败 + 170s 窄修复 + 一次 stage_retry；与 VP-2026-0036 的 null/手抄亚型同属 S4-PERF-RECORDS 产物契约违例家族，第三亚型）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0116
- type: R
- title: kb_stale_check.py 的 stamp diff 过滤非源码路径——repo HEAD（含 .agents/docs 知识库 commit）被当作 tilelang 工具链戳产生全库假阳性（stale_count=125，实际 tilelang 源码零改动）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z：opt_log.md §0 E-5 段（启动时 kb_stale_check 报 stale_count=125 几乎全库，本 kernel 直接相关条目全部标「待重验」——五项结构否决的 carry-over 论证本可直接成立，被迫逐条 git diff 自证）+ Skill Retrospective Skill Flow Issues 第 2 行
  - 机械复核（2026-09-21 蒸馏会话）：`git diff --stat 15ad002..96f287e` = 7 个文件全部位于 .agents/（queue/stats/INDEX/attention/cases/constants/repro——evolution commit），tilelang/ 源码零改动；kb_stale_check 现跑 stale_count=124——假阳性复现
  - 根因：.agents/tools/kb_stale_check.py detect_current_stamp() 以 `git rev-parse --short=10 HEAD`（repo HEAD）为 tilelang_commit——知识库文档 commit 也推进 HEAD，使全部旧工具链戳条目批量失配
  - 交叉引用：queue VP-2026-0089（消费侧口径——全局 stale_count 不作单条目判定依据，已三现）/ VP-2026-0101（消费侧两分法）——本条为工具侧根因修复，三案互补建议同批审批
- repro: python3 .agents/tools/kb_stale_check.py（观察 stale_count）；git diff --stat 15ad002..96f287e（docs-only delta 复现命令——两相对照即假阳性证据）
- toolchain_stamp: kb_stale_check.py 现行版本；发生环境 tilelang 0.1.2+96f287eeaa / CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/tools/kb_stale_check.py
- delta: |
    动作: update（detect_current_stamp 的 HEAD 查询替换为源码树过滤查询）
    定位锚: |
      (detect_current_stamp 函数体内)
          try:
              head = subprocess.run(
                  ["git", "rev-parse", "--short=10", "HEAD"],
                  cwd=REPO_ROOT,
                  capture_output=True,
                  text=True,
                  timeout=10,
              )
              if head.returncode == 0:
                  stamp["tilelang_commit"] = head.stdout.strip()
          except Exception:  # noqa: BLE001 -- advisory tool
              pass
    old 文本: （即上述定位锚原文）
    new 文本: |
          try:
              # The stamp must track the tilelang SOURCE tree, not repo HEAD:
              # knowledge-base/doc-only commits (.agents/, docs/, examples/)
              # also advance HEAD and would flip the stamp, producing
              # whole-library false positives (2026-09-21 ssd rebuild task:
              # stale_count=125 while `git diff --stat 15ad002..96f287e`
              # showed 7 .agents files and zero tilelang source changes).
              head = subprocess.run(
                  [
                      "git", "log", "-1", "--format=%h",
                      "--",
                      "tilelang/", "src/", "tilelangir/", "3rdparty/",
                      "pyproject.toml", "CMakeLists.txt",
                  ],
                  cwd=REPO_ROOT,
                  capture_output=True,
                  text=True,
                  timeout=10,
              )
              if head.returncode == 0 and head.stdout.strip():
                  stamp["tilelang_commit"] = head.stdout.strip()
          except Exception:  # noqa: BLE001 -- advisory tool
              pass
    动机: stale 检测的语义是「tilelang 行为可能变了」；文档 commit 不改变编译器行为，却使每次蒸馏后的全库条目批量假阳性——optimizer Phase 0 被迫逐条 git diff 自证（本任务实证），或更糟：照单全收「待重验」清单浪费调优预算。源码树过滤（tilelang/ src/ tilelangir/ 3rdparty/ + 构建文件——路径集 apply 时可复核增删）使 stamp 只随真实工具链变更推进。
- evidence（第三现，2026-10-30）：engram_decode 蒸馏提交后 `kb_stale_check` stale_count 132→142——本轮新增/入队 7 条条目戳 `76e4550`（任务实际调优 build），而 evolution commit 自身把 HEAD 推进到新 commit（7ffd1b17），使「刚写入的条目」立即被判 stale（+10）。即 stale 判定的比较基准是「当前 HEAD」而非「条目写入时的 HEAD」，任何进化提交都会自我判 stale——与本条首现（源码零改动仍 stale）同根：以 commit 相等性代替「源码树是否变化」的判定。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0117
- type: R
- title: conductor 终态钩子前置「任务产物保全」步骤——DONE 前对算子目录未提交产物做零工作树影响的 git 保全（临时索引 write-tree + update-ref），防 perf_opt 外部删除致算子断裂与调优档案永久丢失
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z：opt_log.md 任务背景（上一调优会话〔task ...20260921T003531Z〕DONE 后 perf_opt/ 目录被外部删除——wrapper 指向 perf_opt 导致算子断裂；14 个实验分支 + 64 行 perf_records 永久丢失）+ Skill Retrospective Skill Flow Issues 第 3 行 + 重建全程（本会话 mode=full 重建耗 10413s，靠 repro/PL-1.18-floor2-wins.py 骨架才恢复三胜出结构——实验数据与 opt_log 不可恢复）
  - 既有机制缺口：statectl `complete` 的「工件 SHA256 快照」只记哈希不保内容（.stage_state.json artifact_hashes——目录删除后无法恢复）；perf_opt/ 未提交文件在 git 工作区无任何恢复途径
  - 数据侧互补：attention.md PL-1.18 update①（repro 骨架自包含性经工件全失场景检验——ED-A/ED-B 的正反面实证：结构可由骨架重推导，过程数据全失）
- repro: 复现条件——任一 DONE 后算子目录 perf_opt/ 被外部删除（wrapper 断裂 + 档案丢失；恢复路径仅 repro 骨架重推导，实验记录不可恢复）
- toolchain_stamp: 会话层工作流（无运行时依赖）；发生环境 2026-09-21（git 仓可用）
- target_doc: .opencode/agents/tilelang-op-conductor.md
- delta: |
    动作: update（「### 1. 任务终态蒸馏钩子（必执行）」步骤列表前插入第 0 步）
    定位锚: |
      `phase` 进入 `DONE` 或 `FAILED` 后（最终报告输出后、同一会话内）：

      1. 判断是否存在**可蒸馏信号**（任一为真即有）：
    old 文本: （即上述定位锚原文——intro 行 + 步骤 1 起始行）
    new 文本: |
      `phase` 进入 `DONE` 或 `FAILED` 后（最终报告输出后、同一会话内）：

      0. **任务产物保全（零工作树影响，best-effort）**：对任务算子目录的未提交产物（`perf_opt/` 全部文件、`history_version/`、`integration_log.md` 等）执行 git 保全——`idx=$(mktemp); GIT_INDEX_FILE=$idx git read-tree HEAD; GIT_INDEX_FILE=$idx git add -- <算子目录>; tree=$(GIT_INDEX_FILE=$idx git write-tree); commit=$(git commit-tree $tree -p HEAD -m "task artifacts: {task_id}"); git update-ref refs/task-artifacts/{task_id} $commit; rm -f $idx`（分支头与工作树均不动；恢复 = `git checkout refs/task-artifacts/{task_id} -- <算子目录>`）。失败不阻塞、不重试，记录到最终报告。动机：上一会话 DONE 后 perf_opt/ 被外部删除致 wrapper 断裂 + 14 实验分支/64 行 records 永久丢失（2026-09-21 ssd 重建任务实证——重建耗整个会话且实验数据不可恢复；statectl SHA256 快照只记哈希不保内容）。
      1. 判断是否存在**可蒸馏信号**（任一为真即有）：
    动机: 调优产物是后续蒸馏/续调/断裂修复的唯一载体；「wrapper 指向 perf_opt + 目录可被外部删除 + 内容无 git 保全」三条件叠加使每个 DONE 任务暴露于不可恢复损失——零工作树影响的 ref 保全把恢复成本从「整个重建会话」降到一条 checkout 命令。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: -
- decided_note: 备选实现（apply 时可裁决）：① statectl `complete` 内置同构保全（工具侧，需改 .agents/tools/statectl.py——机械动作更可靠）；② optimize SKILL.md Phase 4 交付清单加「提示用户 commit perf_opt/」提醒（最轻量）。三形态取一即可。

## VP-2026-0118
- type: R
- title: profile-collection.md 补长采集后台运行形态——`cd X && nohup ... &` 的 `&` 作用于整链，调用方 shell 工具超时 SIGKILL 连带杀死采集进程组；须 setsid 脱离进程组启动
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z：opt_log.md §2 注（perf_records.jsonl 前三行为启动事故——首次后台 baseline 采集被 shell 超时连带杀死前已落 3 行，重启后重测同三案例产生同分布重复记录，append-only 契约下须逐行注记对账）+ Skill Retrospective Skill Flow Issues 第 5 行 + perf_opt/logs/phase1_baseline_all.log（首次启动事故日志）
- repro: 复现条件——以 `cd <dir> && nohup <采集命令> > log 2>&1 &` 形态从带超时的 shell 工具启动分钟级采集，超时触发后观察采集进程被连带 SIGKILL（对照 setsid 形态存活）
- toolchain_stamp: 会话层工作流（无运行时依赖）；发生环境 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（§3 串行运行 msprof op 的注意事项后追加一段）
    定位锚: "注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。"
    old 文本: |
      注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。
    new 文本: |
      注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。

      长采集（全 workload baseline 扫描等分钟级任务）以后台方式运行时，用 `setsid` 脱离当前 shell 的进程组启动（`setsid nohup bash -c '<采集命令>' > {log} 2>&1 &`）——`cd X && nohup ... &` 形态中 `&` 作用于整条链，调用方 shell 工具超时的 SIGKILL 会连带杀死采集进程组，产出半途记录（2026-09-21 ssd 重建任务实证：首次 baseline 采集即被连带杀死、重启后 3 行同分布重复记录，append-only 契约下须逐行注记对账）。
    动机: 采集被中途杀死产生两类成本：半途记录污染 append-only 对账（须注记区分）+ 整轮重采；setsid 是零成本进程组隔离。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0132
- type: R
- title: op-design SKILL D-3 探针纪律三条——① 预算口径补「同一探针文件的失败-修复迭代计入同一探针，总墙钟超限须在 DESIGN 附录如实记录」② 探针须覆盖冻结目标配置的规模挡位（多迭代 serial 循环 ≥3 次更新；小规模替代形状显式声明未覆盖维度）③ 探针执行 stdout tee 落盘 probe_logs + DESIGN 附录 A 增「日志路径」列
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1/2/3：① D-3 预算「≤2 个、每探针 ≤10min」按文件数合规但 8 轮 case 迭代总墙钟 ~25min（design_probe_argmax.py --case 逐轮演进，DESIGN 附录 A）；② tiled 在线递推探针只验了 num_full==2（P2 (3,700) tn=256 / P2b (96,8192) tn=4096），冻结配置 tn=5120→num_full=20 首跑才首次执行——原位 vselect 状态丢失 bug 恰在多迭代暴露（Stage 3 首跑 (4,102400) 输出全落 tile-0 局部索引）；③ Stage 1 探针无 stdout 落盘，Stage 2 交叉验证 C-1~C-4/C-8 只能重新在设备上跑（P1/P2/P2b/p3a×3/dbg 五轮 ~12min 墙钟）；无设备 reviewer 只能纸面接受声明（本轮探针窗口 2026-09-24 03:50–04:15 零 log 文件）
- repro: 复现条件——任一设计期探针产出被 Stage 2 检视消费的场景（探针清单含多迭代循环结构 / 探针结论被交叉验证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（D-3 段扩展，三处）
    定位锚: "严格预算：**≤2 个探针、每探针 ≤10min**，只测未知常数（最小张量/最小 kernel），不实现完整算子；探针脚本落 `examples/{project}/{op}/history_version/design_probe_*.py`（供 gate 与检视核对）"
    old 文本: （即上述定位锚原文）
    new 文本: |
      严格预算：**≤2 个探针、每探针 ≤10min**（同一探针文件的失败-修复迭代与逐 case 轮次计入同一探针预算；总墙钟超限时在 DESIGN 附录如实记录而非拆分文件规避），只测未知常数（最小张量/最小 kernel），不实现完整算子；探针脚本落 `examples/{project}/{op}/history_version/design_probe_*.py`（供 gate 与检视核对）。**规模挡位覆盖**：对方块循环本身有依赖的候选结构（loop-carried 状态 / 级联更新），探针须用**冻结目标配置的 tile 数/迭代数挡位**跑一遍（多迭代 serial 循环 ≥3 次更新）——小规模替代形状须显式声明其未覆盖的规模维度（argmax 实证：num_full=2 探针全过而 num_full=20 首跑暴露状态丢失）。**探针日志落盘**：探针执行 `tee` 到 `history_version/probe_logs/<case>.log`（含时间戳与工具链戳），DESIGN 附录 A 表增「日志路径」列——Stage 2 检视优先读日志比对，无日志且设备可用时重放、设备不可用时标注不可核验（argmax 实证：无落盘使 reviewer 五轮重放 ~12min / 纸面接受两分支都发生）；
    动机: 三条缺口同源于「探针产出的可消费性」：预算口径可被文件数合规绕过、规模挡位缺口使 bug 逃逸到 Stage 3（2731s attempt）、日志缺失使检视要么重放要么盲信。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: ② 的消费侧（design-review SKILL「有 probe_logs 则先读日志比对」）随本条 new 文本一并生效，review SKILL 侧不再单独立案。

## VP-2026-0133
- type: R
- title: op-design SKILL Phase 3 补「引用仓内既有 Python 原语必须 Read 实现核对参数单位与接受判据」——与 DSL API 查文档同纪律；附 compute_tile_n 参数口径实例（budget=字节 / num_buffers=同形缓冲个数 / elem_bytes=单缓冲元素字节 / 除数优先仅在不增加 tile 数时生效）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1 revision：v0 引用 `tileops/kernels/reduction/_primitives.py` 的 compute_tile_n 未读实现、凭参数名猜语义（budget 折元素数 + 再传 elem_bytes/num_buffers 二次折算）——fp16 lm-head 得 tile_n=512 而非 5120（tile 数 ×10），Stage 2 检视读 L110-196 实现才推翻（`per_buffer = block_m*elem_bytes; max_cols = budget // (num_buffers*per_buffer)`；`if div_tiles <= max_tiles: return best_dividing`）；复算：compute_tile_n(1,2,102400,256,6553,5)=512 vs budget=65536→6400；bf16 elem_bytes=2,num_buffers=7,budget=65536→4608+尾 tile（非零尾 4096）
- repro: 复现条件——任一设计引用仓内 host 侧工具函数（_primitives.py 等）套用公式的场景；复算命令自包含于证据（纯 host 算术，与设备无关）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 dev root build 2026-09-24 / 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 3 信息收集章节补一条）
    定位锚: "### Phase 3：信息收集"
    old 文本: （即上述定位锚原文行——在该节 checklist 末尾追加）
    new 文本: |
      ### Phase 3：信息收集
      （节内追加一条）**引用仓内既有 Python 原语（`_primitives.py` 等 host 侧工具函数）时必须 Read 其实现核对参数单位与接受判据**，与 DSL API 查文档同纪律——凭参数名猜语义是 1024× 级错误的来源之一（argmax 实证 2026-09-24：compute_tile_n 的 budget 为**字节**、num_buffers 为**同形缓冲个数**、elem_bytes 为单缓冲元素字节，预算先折元素数再传参会二次折算使 tile_n 512≠5120〔tile 数 ×10〕；其「除数优先」仅在**不增加 tile 数**时生效——bf16 返回 4608+尾 tile 而非零尾 4096，想要零尾 tile 须 wrapper 侧自有规则）。
    动机: 「引 API 查文档」纪律只覆盖了 DSL API，漏了仓内 host 侧工具函数——v0 凭参数名猜语义导致 tile 数 ×10 的设计错误存活到 Stage 2 才被源码对账推翻。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 知识侧（wrapper tile_n 整除优先规则）已随 VP-2026-0129 入队 algorithm-candidates；本条为流程侧（Read 实现纪律 + 口径注记）。

## VP-2026-0134
- type: R
- title: op-design SKILL DESIGN 产出要求补「集成前置核对」——reduce 族（及一切 Op 层有输入预处理的族）设计阶段必须确认目标 Op 层 `_kernel_handles_padding`/`_pad_value` 行为，kernel 声明宽度与 Op 实际传入宽度一致
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 5：集成前置核对只查 manifest test/bench 与 meta 一致性，未查 kernel 数据入口契约与 TileOPs Op 层 pad 行为一致性——DESIGN 冻结的 raw-N 契约（R3）与 `_ReduceOpBase._prepare_input` 的 F.pad(N_padded, -inf) 冲突直到 Stage 5 report 才暴露（初次 report 17 failed 全落在 N_padded≠N 用例，修复耗一整轮 integrator attempt）；tileops/ops/reduction/reduce.py#_prepare_input + integration_log.md#attempt-1
- repro: 复现条件——任一 reduce 族（或 Op 层含 pad/reshape 预处理）算子的 harness 迁移设计
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ubuntu.22.4.npuir / git 013dbbf5 / 2026-09-28
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（DESIGN 产出 checklist / 交付清单段补一条）
    定位锚: design SKILL.md「11. 交付清单」（Phase 4 checklist 段）
    old 文本: （定位锚节内追加一条，apply 时按当时文件内容锁定精确锚）
    new 文本: |
      - **集成前置契约核对（迁移任务）**：设计阶段确认目标 TileOPs Op 层的输入预处理行为——reduce 族读 `tileops/ops/reduction/reduce.py#_prepare_input`（`_kernel_handles_padding=False` 时输入 pad 到 align_up(N,256)，kernel jit 声明宽度须取 N_padded；翻转 True + kernel 原始 N 是高杠杆优化项）；op_kind 涉及 min/max 的共享 kernel 须核对 `_pad_value` 方向性（argmax −inf / argmin +inf）。kernel 数据入口契约与 Op 层实际行为不一致 = Stage 5 集成期爆雷（argmax 实证 2026-09-28：17/42 failed，步长错位症状行 0 对、行≥1 全错且不报错）。
    动机: 契约冲突在设计期是零成本核对（读一个 Op 基类文件），在 Stage 5 是整轮集成返工 + report 重跑；知识侧条目（CASE-argmax-argreduce-integration + VP-2026-0128）覆盖检索路径，本条把核对动作变成设计产出要求。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0128（知识侧，algorithm-candidates）同一发现的两 Tier 拆分；add-npu-op skill integrate 前置节为备选 target（审批时可改路由）。

## VP-2026-0135
- type: R
- title: design-template.md 三条——§5.2 workload 表首列要求附 (M,N) 推导式（无推导式视为待确认）；§5.2 阶梯段补「扩展阶梯（小 N 大 M）预算解除用窄 N 实测膨胀系数（无实测按 4× 保守）+ 阶梯末 cap bm ≤ M」；§5.5 分核表增可选「反事实锚点」列（沿用最差合法配置的发射项对照）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1 revision/Stage 3：① v0 逐 workload 表数字手抄（无 M/N 推导算式）使 1024× 量级 shape 错误（512 vs 524288）存活到 Stage 2 三源交叉才推翻——错误根因不是算错而是抄错后无推导式可自检（revision Skill Flow Issue #1）；② §9.2-R10 预判 (2048,4) 窄内维未探针覆盖但冻结值仍取 bm=2048，首跑 auto-multi-buffer 4× 膨胀溢出（64KB→256KB）返工降档 1024（Stage 3 Skill Flow Issue #2）；block_m>M 时 (1,3)/(1,5) bm=8 尾行垃圾污染有效行；③ 反事实锚点实践：§1.6.0/§5.5 附「沿用最差合法配置的发射项」对照列（3d 行 bm=8→≈4.8ms），使「规模判定↔bm 取值↔发射估算」不自洽在纸面复算即可暴露（revision VP #4）
- repro: 复现条件——任一含逐 workload 表 + bm 阶梯 + 分核表的 DESIGN 撰写/检视
- toolchain_stamp: 流程规则（模板产出要求）；证据环境 tilelang 0.1.2 dev root build 2026-09-24 / 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/templates/design-template.md
- delta: |
    动作: update（三处）
    定位锚 1: "### 5.2 Block 划分"
    new 文本 1: 节内 workload 表说明追加：「逐 workload 表首列附 (M,N) 推导式（如 `dim=0: N=x.shape[dim]、M=prod(其余)`，按 Op 层 reshape 规则重算——movedim(dim,−1) 后 reshape），无推导式的行视为待确认」。
    定位锚 2: "### 5.2 Block 划分"（阶梯/回退伪代码段）
    new 文本 2: 阶梯段注记：「扩展阶梯（小 N 大 M）的预算解除须用**窄 N 实测膨胀系数**（宽 N 的 2× 口径在 N<8 实测 ~4×——(2048,4) bm=2048 手工 64KB→requires 256KB 溢出，降档 1024 通过）；无实测时按 4× 保守取值或直接取降档档位为冻结值；阶梯末 `block_m = min(block_m, M)`（bm>M 的尾行是 UB 未初始化垃圾、窄 N 下污染有效行）」。
    定位锚 3: "### 5.5 分核策略（物理核数适配）⭐"
    new 文本 3: 节内分核表说明追加：「可选『反事实锚点』列——逐 workload 附『沿用最差合法配置的发射项』对照（如 3d (524288,4) bm=8 → ≈4.8ms vs 选中 bm 2048 → ≈18.7µs），使规模判定↔bm↔发射估算的不自洽在纸面复算即可暴露；跨数字一致性约束（改善倍数 ≤ 流量缩减倍数）是免费 invariant」。
    动机: 三条都把「设计期可机械核对的数值」变成模板硬要求——本任务三处失准（1024× shape / 窄 N 膨胀 / 发射项失配 86×）全部有纸面自检入口可拦。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 知识侧（阶梯规则本体）已随 VP-2026-0129 入队；本条为模板产出要求侧。窄 N 膨胀数据点见 VP-2026-0122。

## VP-2026-0136
- type: R
- title: design_calc_check.py 增 `--manifest <yaml> --op <OpName>` 模式——按 Op 层 M/N 推导规则算出每条 workload 的 (M,N) 真值，与 DESIGN §5.2/§5.5 表的 shape/block_m/num_row_blocks/核内任务数逐格 diff；design-review SKILL Phase 1 机械复核段加该调用
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2：最严重阻塞项（3d-non-last-axis 写成 (512,4)，manifest+Op 层真值 (524288,4)，差 1024×）只能靠人工「manifest workloads → `_ReduceOpBase._prepare_input` 的 M/N 推导（dim=0 → M=除规约维外全维乘积）」两步交叉推出；design_calc_check 的 core_split 检查因解析不到维度直接 skip（`"status":"skip","detail":"insufficient parsed dims {}"`）——shape 正确性完全交给 LLM 复核（REVIEW.md 问题 1）
- repro: 复现条件——任一 harness 迁移 DESIGN 的 workload 表机械核对；复算规则自包含（dim=0/中间维 → M=prod(其余)、N=该维长度）
- toolchain_stamp: design_calc_check.py 现行版本；失效环境 tilelang 0.1.2 dev root build 2026-09-24 / 2026-09-24
- target_doc: .agents/tools/design_calc_check.py
- delta: |
    动作: update（新增 CLI 模式 + 调用点）
    定位锚 1: design_calc_check.py CLI 参数解析段（现有 --design 模式旁增 --manifest/--op）
    new 文本 1: 新增模式：`python3 .agents/tools/design_calc_check.py check --design <DESIGN.md> --manifest <manifest.yaml> --op <OpName>`——按 Op 层 reshape 规则（含 `_prepare_input` 的 pad/N_padded 推导）算出每条 workload 的 (M,N) 真值，与 DESIGN §5.2/§5.5 表逐格 diff（shape/block_m/num_row_blocks/核内任务数）；不匹配即 fail 并打印真值推导式（manifest workload 行 → movedim/reshape → (M,N)）。
    定位锚 2: .agents/skills/tilelang-design-review/SKILL.md Phase 1「算术复算」调用行
    new 文本 2: 机械复核调用扩展为含 --manifest 模式（harness 任务 manifest 可得时必跑）。
    动机: 1024× 级 shape 错误是本任务 3 项 Stage 1 修订之首（返工成本 7666s：Stage 1 fail+定点修复 + Stage 2 首检不通过）；真值来自 manifest + Op 层规则而非 LLM 复读——与 VP-2026-0047（表形态解析）互补：0047 修「设计表读得进」，本条补「真值算得出」。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0047 联动（其第四现证据即本任务）；合入顺序建议 0047 先（表解析）或同批。

## VP-2026-0137
- type: R
- title: 拒绝：分核策略不得通过增加 device kernel launch 数量扩展
- evidence:
  - 原提案试图把增加 device kernel launch 数量的方案写入共享分核标准；人工复核认定其违反源/目标 launch 严格一一对应约束。
- repro: none
- toolchain_stamp: none（流程约束裁决）
- target_doc: .agents/skills/_shared/standards/core-split-strategy.md
- delta: |
    不执行原提案。core-split-strategy 只描述同一次 device kernel launch 内的分核策略。
    若负载在该约束下无法提高并行度，应继续寻找同 launch 方案；无可行方案时记录为
    设计限制，不得增加相对于源算子的目标 launch 数量。
- status: rejected
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: human 2026-10-10
- decided_note: 原提案与严格 kernel launch 拓扑保持要求冲突，不写入共享分核标准。

## VP-2026-0138
- type: R
- title: design-review SKILL Phase 2 复审纪律两条——① 复审场景上一轮 REVIEW.md 的文档引文/实测数字一律视为待核工件（阻塞修复涉及的支持面论断必须重读原文档，不得 diff 两版 REVIEW 文本代替取证）② 阻塞修复涉及的数值及其派生量在 DESIGN 全文检索所有出现点并逐处同口径重算（「修订说明列出的章节」≠「受影响的全部章节」）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2 复审：① v0 REVIEW 引用 T.transpose.md §2.2.1 dtype 表把 uint32 写成 √（文档实际 ×），v1 DESIGN 更正复述与 v0 REVIEW 引文冲突，复审重读原文才裁定 v1 准确——若沿用前轮引文会误判 v1「引入新错误」；② v1 定点修订在未改动章节留下三处数字残留（附录 A 262KB〔应 268MB〕/ §9.2-R8 32768B〔应 28672B〕/ §5.2 269MB〔应 272.6MB〕），靠全文 grep + 逐处重算才捕获（REVIEW.md v1 复审维度 2/3/7 N2/N3/N4）
- repro: 复现条件——任一 Stage 1 revision 后的 Stage 2 复审（阻塞修复核验场景）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-24
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 2 复审要点段追加两条）
    定位锚: "### Phase 2：逐维度检视"
    old 文本: （该节复审要点 checklist 处追加，apply 时按当时文件锁定精确锚）
    new 文本: |
      - **复审的「独立」= 证据重新取证，不是文本 diff**：上一轮 REVIEW.md 的文档引文/实测数字一律视为待核工件——阻塞修复涉及的支持面论断（dtype/shape 表、限制条款）必须重读原文档（argmax 实证 2026-09-24：v0 REVIEW 的 dtype 表引文错一列 uint32，沿用会误判 v1 修复引入新错误）；DESIGN 修订说明只当索引用。
      - **阻塞修复的数字体系全文同口径重算扫描**：对每个修复涉及的数值及其派生量（比例/裕度/合计）在 DESIGN 全文检索所有出现点并逐处重算——定点修订的残留高发于修订说明未覆盖的相邻章节（argmax 实证：3 处残留全部在未修订章节）。
    动机: 复审若以「diff 两版文本」代替取证会把前轮引文错误传染为对修复的误判；数字残留的捕获成本（全文 grep）远低于流出到 Stage 3。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0094（引用抄录一手来源）互补：0094 管设计侧引用，本条管复审侧取证。

## VP-2026-0139
- type: R
- title: design-review SKILL 检查项三则——① 维度 8 弃选论证 dtype/shape 支持面论断须同时核对所引文档表与 pattern-library 条目 `dtype:`/`status:` front-matter（实测口径比文档表更强）② 「依赖未实证常数」判定前先检索 `perf_opt/logs/` 该备选的终版日志（不只看首版反例）③ 维度 3 增「冻结工厂逻辑的分支可达性 + 预算口径一致性」核对点（阶梯/回退逐分支代入边界值）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2：① 设计称 T.transpose dtype 限 fp16，被 T.transpose.md §2.2.1 与 pattern-library PL-1.1 front-matter `dtype: [fp16, fp32, bf16] / status: verified` 双双反驳——现有 skill 文本只点名「dtype 矩阵」未提示核对 pattern-library 实测口径（REVIEW 问题 3）；② 设计引 round8_op9 首版 4/4 mismatch 论证 C6 风险，漏引同 log 族 round9 终版（C6 端到端约 17.4µs，golden PASS）——该数字正是 C6 收益常数的现成实测，直接改变「未实证常数」判定（REVIEW 问题 S5；该证据不指定实现结构）；③ v1 阶梯伪代码 p=1 档死分支（`block_m is None and N×B_bm1 ≤ budget` 恒不可达）+ 契约外 N 区间选中 bm=1 驻留但实际 alloc 超自设安全预算（fp32 N=5000：60000≤65536 过滤通过，实际 bm=1 alloc 80000B > 64KB 纪律线）——分支可达性推演 + 边界 N 代入口径核对可纸面判定（复审 VP #1）
- repro: 复现条件——任一含弃选论证/实验裁决计划/工厂期阶梯伪代码的 DESIGN 检视
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-24
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（三处）
    定位锚 1: 维度 8 表「向量化轴与布局决策完整性」行的「弃选论证前提核对」句（"...每个弃选候选的 repack/代价机制前提必须与所引 API 文档亲自核对一致..."）
    new 文本 1: 该句尾追加：「dtype/shape 支持面论断须同时核对所引文档的 DataType 表**与** pattern-library 对应条目的 `dtype:`/`status:` front-matter（实测口径比文档表更强）；两者任一矛盾 → fail（argmax 实证：T.transpose 文档表与 PL-1.1 front-matter 双源反驳设计声称的 fp16 限制）」。
    定位锚 2: 维度 8 表「实验裁决模式完整性」行（"判定裕度依赖未实证常数时"句）
    new 文本 2: 该行判定口径追加：「备选被判『依赖未实证常数』时，先在 `perf_opt/logs/`（同算子前次任务或本任务早期）按 candidate/round 名检索该备选的**终版**日志——首版 mismatch 与终版 golden PASS 可能并存于同一 log 族，已有实测则该常数不再是未实证（argmax 实证：round9 终版 C6 端到端时延直接改变 C6 判定；该证据不指定实现结构）」。
    定位锚 3: 维度 3（Tiling 策略）检查项表
    new 文本 3: 增检查点：「冻结规格中的工厂期选择逻辑（阶梯/回退/分派伪代码）做**分支可达性 + 预算口径一致性**分析：每条分支代入边界值走一遍（死代码 = 恒不可达分支；口径漂移 = 过滤器用 A 口径而 alloc 用 B 口径——如 p=1 档用 B_bm1 核算而实际 alloc 按 B_multi）——选定值对契约内形状正确不代表选择函数对契约外形状守纪律」。
    动机: ①② 都是「证据源完整性」缺口（文档表 ≠ 全部证据源；首版反例 ≠ 全部日志）；③ 是「选定值正确 ≠ 选择函数正确」的覆盖层缺口。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: ①与 Phase 1 第 5 步「弃选论证查证（强制）」的 pattern-library 对照条款衔接（该步已要求对照 pattern-library——本条把 front-matter dtype/status 字段明确为核对对象并绑定 fail 判据）。

## VP-2026-0140
- type: R
- title: integrate_kernel.py gen_kernel_source_block 检测 kernel 模块内被 wrapper 消费的模块级 helper（`_select_config` 等）自动生成成对 baseline/perf_opt import 行——当前只搬运 extracted_functions 名单内的 kernel factory
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 5：wrapper 胶水需要 `_select_config`（kernel 探针校准阶梯）时只能手工加行；本次手工补 `_select_config` 成对行（tileops/kernels/reduction/argmax/argmax.py import 块）；integrate_kernel.py#gen_kernel_source_block（L275 起）仅按 extracted_functions 名单生成行；perf_opt 采纳翻转时易漏翻（integration_log 遗留说明：「翻转 import 时须同步翻转 _select_config 行」——人工记忆依赖）
- repro: 复现条件——任一 kernel 模块含模块级 helper 且 wrapper default_config 消费它的集成/翻转场景；重跑 integrate_kernel.py 覆盖集成副本后 helper import 行消失
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir / git 013dbbf5 / 2026-09-28；integrate_kernel.py 现行版本
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（gen_kernel_source_block 逻辑扩展）
    定位锚: integrate_kernel.py#L275 def gen_kernel_source_block（按 extracted_functions 名单生成 import 行的循环体）
    old 文本: （按名单生成 kernel factory import 行的现行逻辑）
    new 文本: |
      生成 import 行前对 kernel 源模块做静态扫描：模块级 `def _select_config`（及其他被 wrapper 模板引用的 helper 约定名单）存在时，自动生成成对 baseline/perf_opt import 行（与 kernel factory 行同步翻转注释态）；扫描不到则维持现状。
    动机: wrapper config 权威已确立为 kernel 校准阶梯（VP-2026-0093 argmax 第二证：GPU 启发式直接 ub overflow）——helper import 成对性靠手工维护是翻转漏翻的已知隐患点。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0055（gen_kernel_source_block stem fallback 缺陷）同函数域不同缺陷，合入时互检；与 VP-2026-0093/0074（config 契约族）联动。

## VP-2026-0141
- type: R
- title: 脚手架/工件清理流程的知识保全规则——同算子重迁移（Stage 0 脚手架重建工作区）不得清理 `examples/{op}/{func}/repro/*.py` 与 capability-gaps 登记条目；跨任务同算子知识继承路径（perf_opt/logs + repro + CG）在 scaffolder 流程显式化
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：前次同算子任务的工具链陷阱知识几乎流失——capability-gaps 曾有 CG-2026-0014 登记（trap docstring 自称已登记）但条目丢失；repro .py 源码被清理仅存 `repro/__pycache__/*.pyc` 与 `perf_opt/logs/`；本次靠 pyc 反序列化 docstring + perf 日志考古才恢复两条关键陷阱（multitile bm>1 segfault / vcast f32→f32 corrupt）与性能校准数字（hidden-state fp16 bm=2/bf16 bm=1 终选值与前次完全互证）（Stage 1 复盘 Info-source 行 + DESIGN §9.4-R9 补录记录）
- repro: 复现条件——任一同算子二次迁移任务开始时检查前次 repro/*.py 与 CG 登记的存活状态
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-24（前次任务工件 2026-09-23）
- target_doc: .opencode/agents/tileops-scaffolder.md
- delta: |
    动作: update（执行流程补知识保全条款）
    定位锚: tileops-scaffolder.md「## 执行流程」段（S1-S7 移植步骤列表）
    old 文本: （该流程段末尾追加，apply 时按当时文件锁定精确锚）
    new 文本: |
      6. **前次任务知识保全（同算子重迁移）**：Stage 0 重建/清理工作区时**保留** `examples/{op}/{func}/repro/*.py`（知识域 repro 源）、`examples/{op}/{func}/perf_opt/logs/`（实测锚点）与 `.agents/evolution/capability-gaps.md` 登记条目（不随源码清理）；重迁移 prompt 中附「前次任务知识入口」清单（repro 文件名 + CG gap_id + 校准数字出处），供 Stage 1 设计期直接消费而非 pyc 考古（argmax 实证 2026-09-24：清理致知识流失，恢复成本一整段设计期探针预算）。
    动机: repro/CG 是跨任务知识载体，清理使下一次迁移从零考古；perf_opt/logs 与 repro 的「金矿」价值（校准值互证、失败反例、终版正例）依赖其存活。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 清理动作的实际执行方若不在 scaffolder（如 conductor Stage 0 编排或脚本），apply 时改路由至实际 owner；与 VP-2026-0117（终态产物保全）/ VP-2026-0102（迁移重做 git-show 核对）互补三件。

## VP-2026-0147
- type: R
- title: iteration-diagnosis 增补「先测空 kernel launch 地板」判据——决定 Ratio 天花板是「单核算不动」还是「多核没用到」之前必须先量地板
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z 的 perf_opt/opt_log.md §6.1 第 1 行（Skill Flow Issues 原文）+ §2.2（探针一次即得 1.89µs，直接给出 b1 ≤16% 结论）+ §4.3（60% 裁决）
  - 反例成本：同一任务在测地板前已把 split-d/split-K 作为主要方向评估（opt_log §1.4 调优顺序把 b4 绝对耗时最大者排首），若先测地板可更早识别 split 方向被能力缺口封死的概率
  - **第二证（不同任务，2026-10-31，task engram_decode_mcore-engram_decode_mcore-20261031T012849Z）**：设计层重做任务把本判据**提前到 Stage 1**执行——两枚探针（空 kernel 地板 P1 + split-d 第一阶段 P2）合计 <10 分钟，即给出「40% 不可达」的路线无关上界（均值 ≤22.2%），随后据此把可达目标改为 ≥10% 并交付 9.95%；对照：同一谱系上一任务在 Stage 4 用 12 个内层分支才撞到 3.48% 天花板。**判据随之升级为两级**（地板 + 无法避免阶段的进度下界），new 文本已补第 3 条（见 delta）。**复发登记**：本任务 Stage 1 attempt 1 仍耗 2212s 才收敛到该判据，说明「先判可达性」尚未成为默认动作。
  - **第三数据点（不同任务，2026-11-01，task TileOPs-_sparse_mla_kernel-20261101T014649Z，层内为「零瓶颈地板」形态）**：同一判据在 **grid ≫ 物理核数**的谱系上同样成立，且探针更便宜（1 次 80 s 采集）——把瓶颈操作整体删掉（64 条逐行 copy → 1 条 64 行连续 copy，数学无效仅测时）得地板 26525.5/6828.8/1793.8 µs，直接给出 Ratio 上界 7.52%/7.31%/6.96%；40% 对应 4987 µs 比地板慢 5.32×。**建议 apply 时把 new 文本第 1 条扩为两种地板**：「launch 地板（空 kernel，答发射/唤醒固定项）」与「零瓶颈地板（删掉瓶颈操作，答该瓶颈免费时的绝对上界）」，判据同为 `roofline_ideal / floor = Ratio 上限`；两地板互补，前者限小 grid，后者不限 grid。数据与条目化见 pattern-library `CASE-dsa-decode-gather-floor` / `CONST-dyn-row-gather-opcount-dest-independent`
- repro: none（流程规则，无运行时依赖；判据的常数证据见 VP-2026-0142）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: add（新增小节「Ratio 类硬目标的先行判据」）
    定位锚: "## 3. 诊断上下文模板"（在该节之前插入；apply 时以目标文件实际二级标题为准，须唯一命中）
    new 文本: |
      ## 2.5 Ratio 类硬目标的先行判据：先测 launch 地板

      当 `grid < 物理核数`（如 `T.Kernel(batch)` 且 batch ≤ 8）且 `Task Duration < 50µs` 时，在把「多核切分 / split-N / split-K」当作主要方向之前，先测两条地板，用它们给出该 workload 的 Ratio 硬上限：

      1. **launch 地板**：1 block、512B GM→UB→GM 连续 copy 的空 kernel，`msprof op --launch-count=20 --warm-up=5` 取 Task Duration 中位数（910B3 实测 1.89µs）。判据：`roofline_ideal / launch_floor` 即该 workload Ratio 上限——小于 16% 时该 workload 的 Ratio 目标与实现质量无关，须在 Final Summary 如实披露，不上报「未达标」为调优不足。
      2. **核数比例上限**：Ratio ≤ 工作核数 / 物理核数 × 单核效率——`batch/40` 形态（AIV 40 核）下 b1/b4/b8 的上限分别是 2.5%/10%/20% × 单核效率。
      3. **路线无关上界（第二级，2026-10-31 第二证补入）**：`min over n≤40 [ FLOOR(n) + 总向量元素数/(n × 每核实测吞吐) ]`，带宽与末阶段（apply/elementwise 尾段）**全按免费**，n 也纳入最小化。它回答「换分块轴 / 加核也翻不过去的界」，而单路线实测只能称「该路线的观测值」。实测例（b1/b4/b8）：上限 9.92% / 32.04% / 24.51%，均值 **≤22.2%**（vs 单路线 split-d A 段天花板 15.7%、split-K 16.2%）——40% 目标被这一条独立封死，且与能力缺口是否补齐无关。公式推导与逐档复算见设计侧提案 VP-2026-0156；两级判据的成本都 <10 分钟，应在地板探针之后立即算。

      探针纪律：同用途探针须用与目标同形态的**连续** `T.copy` 编写（实测教训：stride-32 gather store 形态给出 33µs 的错误值，改连续搬运后 1.89µs）。两步成本 <30 分钟，可避免把「物理不可能」误判为「还没优化够」而耗尽 Stage 4 预算。
    动机: 本任务目标是「官方 workload 平均 Ratio ≥ 60%」，而 b1 的 roofline ideal 仅 0.306µs < 空 kernel 地板 1.89µs——该 workload 的 Ratio 上限 16.2% 与实现无关，物理上封死平均 60%。当前 iteration-diagnosis 没有「先判可达性再投入结构改造」的步骤，调优 Agent 容易先做结构级评估（split-N/split-K）再撞上能力缺口（CG-2026-0018）。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_engram_decode_kernel-20261030T121905Z 2026-10-30
- decided_by: -
- decided_note: -

## VP-2026-0148
- type: R
- title: kb_lint 增补 KB-XREF-DANGLING——知识库内被引用的条目 id（CG-*/VP-*/CASE-*/PL-*/TRAP-*/CONST-*）须在库内可解析，检测跨文件死链
- evidence:
  - 本次蒸馏实测：`CG-2026-0022` 被 6 处文档引用（engram_decode 的 DESIGN.md / integration_log.md / RETROSPECTIVE.md；engram_gate_conv_fwd DESIGN.md ×4 处，含「capability-gaps.md L961」行号），而 `.agents/evolution/capability-gaps.md` 现有最高号为 CG-2026-0017——引用死链
  - 同批死链：`CASE-engram-decode-migration`（engram_gate_conv_fwd DESIGN 引用 cases.md L590，实际无此 id）/ `CASE-engram-decode-stage4` / `VP-2026-0283`（queue.md 当前最高 VP-2026-0141）/ `TRAP-npuir-vec-scalar-broadcast` / `TRAP-npuir-fp32-vector-reduce` / `TRAP-C13-parser-no-python-unroll` / `TRAP-conditional-alloc` / `CONST-ascend910-9362-profile`（traps-compiler.md / constants.md 均无）
  - 成因推断：stats.md §4 变更日志最后一条为 2026-09-28，10 月各任务（含本算子迁移 2026-10-15）的蒸馏成果未落盘；下游任务却按「已被蒸馏过」引用——死链在下一次任务的设计期才暴露，且当时无法区分「条目被删」与「从未写入」
- repro: none（静态检查规则；复现条件 = 对当前库跑一次 lint，若对上述 id 报 dangling 即为规则生效）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/tools/kb_lint.py
- delta: |
    动作: add（新增检查函数 + 在 main 中挂载 + docstring 增列规则）
    定位锚: "        lint_index(all_ids, failures, warnings)\n        lint_queue(failures, warnings)"
    new 文本: |
            lint_index(all_ids, failures, warnings)
            lint_queue(failures, warnings)
            lint_xref_dangling(failures, warnings)
    （新函数置于 `def lint_queue(...)` 之后）

    def lint_xref_dangling(failures: list, warnings: list) -> None:
        """KB-XREF-DANGLING: ids referenced inside the KB must resolve (WARNING).

        A reference to CG-/VP-/CASE-/PL-/TRAP-/CONST- that exists nowhere in
        the knowledge base means downstream tasks pre-inject or cite an entry
        that is not actually there -- indistinguishable from "entry deleted"
        until someone greps by hand.
        """
        defined = set()
        ref_files = []
        for root, _dirs, names in os.walk(os.path.join(REPO_ROOT, ".agents")):
            for n in names:
                if n.endswith(".md"):
                    ref_files.append(os.path.join(root, n))
        for path in ref_files:
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            defined |= set(
                re.findall(r"(?m)^(?:id: |#+\s+)?((?:CG|VP|CASE|PL|TRAP|CONST)-[A-Za-z0-9.\-]+)", text)
            )
        for path in ref_files:
            rel = os.path.relpath(path, REPO_ROOT)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            for ref in set(re.findall(r"\b((?:CG|VP|CASE|PL|TRAP|CONST)-[A-Za-z0-9.\-]+)", text)):
                base = re.sub(r"-\d{4}$", "", ref)
                if ref not in defined and base not in defined:
                    warnings.append(
                        _warn("KB-XREF-DANGLING", rel, f"reference to undefined entry id '{ref}'")
                    )
    动机: 知识库自进化的失效模式之一是「引用先行、条目未落盘」——本次 9 个死链 id 全部来自 10 月任务对未蒸馏成果的引用，且在被引用的那次设计/调优里没人能发现。机械检查把这类死链从「下次撞上时人工 grep」变成 lint 期告警；WARNING 级（不阻塞合入）与 KB-BUDGET 同级，避免历史库一次性红灯。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_engram_decode_kernel-20261030T121905Z 2026-10-30
- decided_by: -
- decided_note: -

## VP-2026-0154
- type: R
- title: optimize SKILL.md Phase 4 第 6 步（pattern-library 回写）补治理三则——P 类回写条目须标 `status: pending`（证据 1/2）、新增条目必须同步登记 INDEX.md §4、repro 文件名 = 条目 ID 且头部含条目 ID + First verified
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T145026Z：Stage 4 按本条回写 4 条新条目（`CONST-small-dma-issue-cost` / `CONST-ub-budget-vs-pattern-cap` / `PL-scalar-vec-half-overlap` / `TRAP-ub-non-pow2-row-stride`），**全部以 `status: verified` 直落主题文件**——其中 3 条 `repro: repro-missing`（D 类缺 repro 按 merge-policy §1 应降 Tier 1）、1 条为 P 类首证（应按 Tier 1 等第 2 次独立证据）
  - 同任务：4 条新条目**均未登记 INDEX.md §4 路由表**（`kb_lint.py` 报 4 条 `KB-INDEX-COVERED` WARNING ⟹ `kb_search.py` 按索引检索不到，回写等于不可发现）
  - 同任务：新 repro 文件名 `stage-tile-not-pattern-cap.py` 与条目 ID 不一致、头部缺条目 ID 与 `First verified`（`kb_lint.py` 3 条 `KB-REPRO-HEADER` FAIL）；opt_log §6.4 记「`repro-missing` 项：无」与实际 3/4 条 repro-missing 不符
  - 同类先例：VP-2026-0089（条目时效性判定口径）、VP-2026-0148（KB-XREF-DANGLING）——本条为**写入门**而非读侧/检查侧，三条互补
  - **第五数据点（2026-11-04，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z）——「声称已落盘」亚型**：opt_log §10.3 记「本轮已实际落盘（`kb_lint: 0 failures`）」并列 5 处（constants.md `CONST-vector-op-cost-910B3` / attention.md PL-1.11+PL-1.12 追记 / traps-compiler.md `TRAP-vector-f16-scalar-dtype` / INDEX 行 + `repro/PL-1.11b-mask-guard.py`、`repro/TRAP-vector-f16-scalar.py`）。终态核对：上述条目 ID 全库 grep **0 命中**，两个 repro 文件不存在，attention.md 无对应追记（可能为跨 session 工作区丢失，但 opt_log 未标失效）。⇒ 现有三则只覆盖「怎么写」，缺「声称已写入后的机械存在性校验」（同 KB-XREF-DANGLING 的写侧镜像）；本轮已按实测重新入队 VP-2026-0200/0202。
- repro: python3 .agents/tools/kb_lint.py（负例 = 本轮 optimizer 回写后的 4 条新条目 → 3 FAIL + 4 WARN；正例 = 本轮 evolver 审计后的知识库 → 0 failures）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+01805d7a07ca111e43868c07b496bffb30813ba3 / CANN 9.0.0 / Ascend910B3 / 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 4 第 6 步末尾追加治理三则，锚文本唯一）
    定位锚: "流程/结构层面的改进仍只提 proposal，不自动改。**本条任务内回写与终态 evolver 蒸馏互补防双写：已回写的条目在复盘表 evidence 中注明回写位置。**"
    old 文本: |
      流程/结构层面的改进仍只提 proposal，不自动改。**本条任务内回写与终态 evolver 蒸馏互补防双写：已回写的条目在复盘表 evidence 中注明回写位置。**
    new 文本: |
      流程/结构层面的改进仍只提 proposal，不自动改。**本条任务内回写与终态 evolver 蒸馏互补防双写：已回写的条目在复盘表 evidence 中注明回写位置。**
      **回写治理三则（Tier 分级在回写侧同样生效，2026-10-31 engram_decode round 3 审计）**：
      1. **P 类条目不得以 `status: verified` 直落**——首证条目按 Tier 1 语义登记，front-matter 写 `status: verified` 仅在「同一结论已由 ≥2 个不同任务复现」或「本次是 update 既有 verified 条目」时成立；首次为 P 类新条目时，正文首行须标 `〔首证 1/2，待第二任务确认〕`，并把条目同时提交给终态 evolver 入 queue（evolver 依 queue-schema 归一，防双写）。
      2. **每条新增条目必须同步登记 INDEX.md §4 路由表**（ID + 一句话 + 状态）——未登记则 `kb_search.py` 按索引检索不到，`kb_lint.py` 的 `KB-INDEX-COVERED` 会报 WARNING。
      3. **repro 文件名 = 条目 ID**（`repro/<id>.py`），头部 docstring 首段须含条目 ID 与 `First verified:` 版本戳行（ED-B §2）——不合规的 repro 会被 `KB-REPRO-HEADER` 判 FAIL。
      回写后在本 skill 可运行的机械自检：`python3 .agents/tools/kb_lint.py --files <改动文件>` 须 0 failures。
    动机: 本条是「不自动修改 skill 文档」的唯一例外口，但例外口目前只写了「写什么」没写「以什么证据等级写、写到哪登记、repro 怎么命名」——本轮 4 条回写全部跳过 Tier 分级与索引登记，P 类首证被标成 verified，等价于把 Tier 1 的门禁在回写侧旁路；同时 opt_log 复盘自述与产物不符（§6.4），下游 evolver 需逐条重判，蒸馏成本上升。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0155
- type: R
- title: tilelang-op-optimizer.md 输出契约补 pattern-library 数据文件域——与 optimize SKILL.md Phase 4 第 6 步「强制回写」对齐，并显式禁止回写 SKILL.md/流程文件
- evidence:
  - `.opencode/agents/tilelang-op-optimizer.md`「输出文件/输出目录」表只列 `perf_opt/{op}.py`、`opt_log.md`、`perf_records.jsonl`、`profiles/`、`logs/`、可选 `perf_feedback.md`——**未列 pattern-library 回写**；而同 agent 加载的 `tilelang-op-optimize/SKILL.md` Phase 4 第 6 步为「**必须**追加到 references/pattern-library/」
  - 本轮 conductor 调度 prompt 另附「只写 perf_opt/」约束 → 三方（agent 输出契约 / skill 强制回写 / 调度约束）互相矛盾，optimizer 依 skill 执行后被判越界
  - task TileOPs-_engram_decode_kernel-20261030T145026Z 的 `git status`：`M .agents/skills/tilelang-op-optimize/references/pattern-library/{constants,layout,traps-compiler,cases}.md` + `?? repro/stage-tile-not-pattern-cap.py`
  - 与 VP-2026-0154 互补：0154 管「以什么证据等级写」，本条管「是否有权写 / 写到哪」——同批审批
- repro: none（契约文本一致性核对：`grep -n "pattern-library" .opencode/agents/tilelang-op-optimizer.md` 应命中输出契约表行；当前 0 命中）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+01805d7a / CANN 9.0.0 / 2026-10-31
- target_doc: .opencode/agents/tilelang-op-optimizer.md
- delta: |
    动作: update（输出文件表追加一行 + 约束表追加一条）
    定位锚: "| 可选输出   | `examples/{project}/{op}/perf_opt/perf_feedback.md` | `[DESIGN_LIMIT]` 设计层天花板反馈（触发条件两项同时满足时必须产出，否则禁止产出；固定 schema 见 `_shared/standards/perf-feedback.md` §2） |"
    old 文本: |
      | 可选输出   | `examples/{project}/{op}/perf_opt/perf_feedback.md` | `[DESIGN_LIMIT]` 设计层天花板反馈（触发条件两项同时满足时必须产出，否则禁止产出；固定 schema 见 `_shared/standards/perf-feedback.md` §2） |
    new 文本: |
      | 可选输出   | `examples/{project}/{op}/perf_opt/perf_feedback.md` | `[DESIGN_LIMIT]` 设计层天花板反馈（触发条件两项同时满足时必须产出，否则禁止产出；固定 schema 见 `_shared/standards/perf-feedback.md` §2） |
      | 受控输出   | `.agents/skills/tilelang-op-optimize/references/pattern-library/`（主题文件 + `repro/`） | 任务内回写实测模式/代价常数/陷阱实证（optimize SKILL.md Phase 4 第 6 步的唯一例外授权）：只限**数据/模式类**文件与 `repro/`；必须带 `origin_task` + 工具链版本戳 + Tier 分级标注（P 类首证标 `〔首证 1/2〕`）并同步登记 `INDEX.md` §4；**禁止**改 `SKILL.md` 与任何 `references/` 下的流程文档 |
    动机: agent 定义是 Subagent 的白名单契约，缺 pattern-library 行时「写回」只能靠 skill 正文授权，与 conductor 调度 prompt 的「只写 perf_opt/」直接冲突（本轮实际发生：产物写得对、流程被判越界、下游需人工审计）；把授权与边界同时写进输出契约后，冲突面收敛为一处可机械核对的表格。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0158
- type: R
- title: gate 2 的章节号正则把「8 条建议落实情况」类正文标题当成维度 8——`_dim_body(8)` 取到错误章节，S2-DIM8-EVIDENCE 可假 PASS / 假 FAIL
- evidence:
  - `.agents/tools/gate_lint.py` L394-405（`DIM_HEADING_RE = r"^#{1,4}\s*%s[.、\s]"` + 同款 `NEXT_HEADING_RE`，数字后可跟裸空格）+ L444-452（`_dim_body(text, 8)` 驱动 `S2-DIM8-EVIDENCE`）
  - task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 的 `REVIEW.md` 实测：`## 8 条建议落实情况`（L62）被 `^#{1,4}\s*8[.、\s]` 首次命中，`_dim_body(8)` 截出的是该建议表而非 `### 8. 算法优化分析`（L131）。本次该表恰好含「复核/复算/核对」三词所以通过——表格措辞一变即假 FAIL，而真正的维度 8 缺证据时也可能因错误章节里的词命中而假 PASS
  - 复现（inline）：`python3 -c "import re;t=open('REVIEW.md').read();print(repr(re.search(r'^#{1,4}\s*8[.、\s]',t,re.M).group(0)), re.search(r'^#{1,4}\s*8[.、\s]',t,re.M).start())"` → 命中 L62 而非 L131
  - 兼容性核对：仓内 16 份 `REVIEW.md` 的维度标题均为 `### N. 标题` 形态（另见 `### 40% 可达性 …` 这类非维度标题，现行正则不误伤、新正则同样不命中），要求「数字后必须跟 `.`/`、`/`．`/`：`」不破坏既有合法形态
- repro: python3 -c "import re; t='## 8 条建议落实情况\n'; print('still-matches' if re.search(r'^#{1,4}\s*8[.、\s]', t, re.M) else 'fixed')"（修复前输出 still-matches；修复后 None → fixed）
- toolchain_stamp: gate_lint.py 现行版本（工作树 HEAD 3ed7c914）；发生环境 tilelang 0.1.2+3ed7c914 / CANN 9.0.0 / Ascend910B3 / 2026-10-31
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（同 hunk 内两行正则，均属「章节号识别」）
    定位锚: |
      DIM_HEADING_RE = r"^#{1,4}\s*%s[.、\s]"
      NEXT_HEADING_RE = re.compile(r"^#{1,4}\s*[0-9]+[.、\s]", re.M)
    old 文本: |
      DIM_HEADING_RE = r"^#{1,4}\s*%s[.、\s]"
      NEXT_HEADING_RE = re.compile(r"^#{1,4}\s*[0-9]+[.、\s]", re.M)
    new 文本: |
      # Dimension headings are numbered ("### 8. 算法优化分析"). A bare space after
      # the digit also matches prose headings such as "## 8 条建议落实情况", which
      # silently redirects _dim_body() to an unrelated section (task
      # engram_decode_mcore 2026-10-31: _dim_body(8) returned the "8 条建议"
      # table, not dim 8) -- so require a section-number punctuation mark.
      DIM_HEADING_RE = r"^#{1,4}\s*(?:维度\s*)?%s\s*[.、．:：]"
      NEXT_HEADING_RE = re.compile(r"^#{1,4}\s*(?:维度\s*)?[0-9]+\s*[.、．:：]", re.M)
    动机: 该正则决定 S2-DIMS 与 S2-DIM8-EVIDENCE 两个门禁读哪一段——把「标题里出现 8」当维度 8 会让门禁的通过/失败与真实章节无关（本次是巧合通过）；章节号后必须有分隔符是人工阅读时的同款约定，机械层对齐即可根治。
- status: pending
- confirmations: -/-
- created_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0159
- type: R
- title: design_calc_check._section 用「标题包含 marker」定位章节——§1.4 标题含「§1.6.0」时把调研段截成 1182 字符，四口径机检全部假 fail
- evidence:
  - `.agents/tools/design_calc_check.py` L39-53（`_section`：`if marker in m.group(2)`）+ L256（`sec = _section(text, "1.6.0") or text` → r3_metrics）
  - task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 的 Stage 2 复盘：`design_calc_check.py::check_r3_metrics` 把 §1.4 标题「算法描述（§1.6.0 调研选定的 NPU 侧算法）」当作 §1.6.0，截出 §1.4 正文（1182 字符）→ 四口径全报 fail；现场由人工逐层追到 `_section` 才确认是解析器误匹配（否则会记成维度 8 缺项）
  - 复现（inline，需一份 DESIGN.md）：`python3 -c "import sys;sys.path.insert(0,'.agents/tools');import design_calc_check as d;print(len(d._section(open('DESIGN.md').read(),'1.6.0')))"` → 1182（应为 §1.6.0 全段，约 9k 字符）
  - 兼容性核对：现有 4 个调用点 marker 为 `"4.5"` / `"内存"` / `"Tiling"` / `"5"` / `"1.6.0"`，对应标题均以该 marker 开头（如 `### 4.5 UB 内存预算` / `## 5. Tiling 策略`），改「标题以 marker 开头」不破坏现状；无命中时返回空串，调用方已有 `or text` 兜底
  - 缓解措施已在 v2 设计侧生效（改标题），但工具侧未修——下一个任务写出同形标题会再次误报
- repro: python3 -c "import sys;sys.path.insert(0,'.agents/tools');import design_calc_check as d;print(d._section('## 1. 概述\n### 1.4 算法描述（§1.6.0 调研）\nBODY-1.4\n#### 1.6.0 算法调研\nBODY-1.6.0\n','1.6.0'))"（修复前返回 BODY-1.4，修复后返回 BODY-1.6.0）
- toolchain_stamp: design_calc_check.py 现行版本（工作树 HEAD 3ed7c914）；发生环境 tilelang 0.1.2+3ed7c914 / CANN 9.0.0 / 2026-10-31
- target_doc: .agents/tools/design_calc_check.py
- delta: |
    动作: update（替换 `_section` 函数体，一处）
    定位锚: |
      def _section(text: str, marker: str) -> str:
          """Return the body of the first heading containing the marker."""
    old 文本: |
      def _section(text: str, marker: str) -> str:
          """Return the body of the first heading containing the marker."""
          lines = text.splitlines()
          body, inside, depth = [], False, 0
          for ln in lines:
              m = re.match(r"^(#{1,6})\s+(.*)$", ln)
              if m:
                  if inside and len(m.group(1)) <= depth:
                      break
                  if marker in m.group(2):
                      inside, depth = True, len(m.group(1))
                      continue
              if inside:
                  body.append(ln)
          return "\n".join(body)
    new 文本: |
      def _section(text: str, marker: str) -> str:
          """Body of the first heading whose text *starts with* `marker`.

          A heading that merely CONTAINS the marker belongs to another section:
          "### 1.4 算法描述（§1.6.0 调研选定的 NPU 侧算法）" was taken as section
          "1.6.0" and truncated the调研 body to 1182 chars, turning all four
          R3-metric checks into false failures (task engram_decode_mcore,
          2026-10-31). Returns "" when no heading starts with the marker (all
          callers fall back via `or text`).
          """
          lines = text.splitlines()
          body, inside, depth = [], False, 0
          for ln in lines:
              m = re.match(r"^(#{1,6})\s+(.*)$", ln)
              if m:
                  if inside and len(m.group(1)) <= depth:
                      break
                  heading = m.group(2).strip().lstrip("§").strip()
                  if heading.startswith(marker):
                      inside, depth = True, len(m.group(1))
                      continue
              if inside:
                  body.append(ln)
          return "\n".join(body)
    动机: `_section` 是 r3_metrics / ub_budget / core_split 三个检查项的章节定位器——「包含」匹配会把正文里提到 `§1.6.0` 的任何标题当成目标章节，机检结果的语义因此依赖标题措辞；改成「以 marker 开头」后章节定位与人工阅读一致，且对现有 DESIGN 形态零影响。
- status: pending
- confirmations: -/-
- created_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0160
- type: R
- title: design-review 维度 8 增补四条检查项——候选表逐列口径一致 / 上界断言范围闭包 / 收益量化回基线源码数 pass / D-1 覆盖跨块重组
- evidence:
  - task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 的 v1 `REVIEW.md`（结论不通过，2 阻塞 + 8 建议）逐条对应：问题 1 = 用单路线实测冒充「任何实现的严格下界」（证明范围未闭合）；问题 2 = R4 表把 A 段与 A+B 放同一列且 split-K 行不可复算（口径混用导致选型读法相反）；问题 4 = O3 把基线早已存在的「`(1,1)` 标量倒数广播」当成新收益（收益量化未回基线源码）；问题 5 = 跨列块部分和重组未被 `verify_equiv.py` 覆盖却称「D-1 已验证」
  - 现状缺口：维度 8 现有「调研结论正确性」一行覆盖等价性/复杂度算术/亲和性/负向断言，但**没有**「表内口径一致性」与「界类断言的证明范围」两栏；「数学等价优化完整性」只要求收益栏非空，未要求对照基线源码；「D-1」覆盖要求写在 develop/design 侧，review 表的维度 8 未列此项
  - 该 4 项已由 v2 修订实际执行（`REVIEW.md`「v1 建议落实情况」8/8 落实），说明检查项可操作、且能被执行者机械核对
  - **独立第五项候选（2026-10-31 追加，task fft_c2c_vec-fft_c2c_vec-20261031T103133Z，不计确认——R 类恒人工门）**：维度 8 的「向量化替代分析」栏只要求「有分析」，未要求「分析判据与 `npu_loop_vectorize.cc` 的四道闸一一对应」。该任务首轮检视只核 store/load 两条下标规则、漏掉入口条件「体内恰 1 条 BufferStore」，得出「必须核内重排（UB alias）才能向量化」的错误前提 → v1 判不通过 → Stage 1 重试 2 次（合计 2667s）。清单化的第五项（向量化形态 → pass 条件链逐条核对）已单独成提案 VP-2026-0166，建议与本条同批审批（同属维度 8 表行扩充，分两次 apply 会重复触碰同一张表）。
- repro: none（检查项文本；复现条件 = 对 v1 文档按新检查项复核，应命中 4 项而非 0 项）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+3ed7c914 / CANN 9.0.0 / 2026-10-31
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 检查表末尾追加 4 行）
    定位锚: "| 与下游章节一致 | §3.1 公式拆解以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点不得再出现对应逐元素标量循环） | §3.1 仍用优化前公式；§6 出现 §1.6.2 未论证的标量循环 |"
    old 文本: |
      | 与下游章节一致 | §3.1 公式拆解以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点不得再出现对应逐元素标量循环） | §3.1 仍用优化前公式；§6 出现 §1.6.2 未论证的标量循环 |
    new 文本: |
      | 与下游章节一致 | §3.1 公式拆解以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点不得再出现对应逐元素标量循环） | §3.1 仍用优化前公式；§6 出现 §1.6.2 未论证的标量循环 |
      | **候选对比表口径一致性** | 候选/上界汇总表的每一列（或每行）标注口径：阶段范围（A 段 only / A+B）、实测 vs 模型、是否含跨核归并、带宽假设（峰值/有效/段宽限定）、并行度（块数）；同一列内口径一致；块数不同的候选其向量项按「每块元素数 = 总元素数/n」重算 | 把 A-only 与 A+B 放同一列（同一张表两种自洽读法给出相反选型，实测例：12.5 vs 15.7 与 9.0 vs 9.34）；把某块数下实测的向量项复用到另一个块数 |
      | **可达性/上界断言的范围闭包** | 「上界/下界/不可达/严格界」须核验「证明范围 ⊇ 断言范围」：单路线实测只能表述为该路线的观测值；全局界写成路线无关形式（发射地板 + 实测每核吞吐，带宽与末阶段按免费）并对并行度取最优；「若补齐能力 → X%」类外推须附估算口径与逐项假设 | 用一条（且被自身实测证明带宽劣化 2.3× 的）路线实测冒充「任何实现的严格下界」；外推只给结论数字（实测例：39–41% 被推翻 1.8×，真值 ≤22.2%） |
      | **收益量化必须回到基线源码数 pass** | 等价优化项的收益栏与基线源码逐 pass 交叉核对（基线已有的 CSE / 标量倒数广播 / 整块 2-D copy 不得计为增量收益）；「省 N pass / 省 M 次 reduce」与自身 §3.1 / §1.6.2 的 pass 清单一致 | 把基线早已存在的 `(1,1)` 标量倒数广播当成新收益（实测收益为零）；称 reduce 次数减少而总量由 4×d 变为 4×D_s + 1×8（元素数不减） |
      | **D-1 覆盖跨块重组形态** | 采用跨块分块（split-d / split-K / 两阶段）时，`verify_equiv.py` 含「跨块部分和 + 定序合并 vs 整体求和」用例（统计量 tier-1、fp32 域端到端 tier-2）；不能只靠实现期 L0 互比兜底 | 声称「D-1 已机器验证」而脚本只覆盖整体求和（分组/合并顺序差异未覆盖） |
    动机: v1 的两个阻塞项与两条建议本质上都是「表的语义不闭合」而非方向错误，但现有维度 8 检查表没有对应栏位——检视方只能靠自由文本发现，换一位检视者即可能放过；把 4 条落成表行后，这类问题的发现权从「个人经验」变成「检查项」。
- status: pending
- confirmations: -/-
- created_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0161
- type: R
- title: op-develop Phase 2 前置「最小原型」步骤——多段/新形态 kernel 先用 ~60 行原型验证编译与数值，再写正式文件
- evidence:
  - task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 的 Stage 3 复盘：先写 `/tmp/opencode/proto_a.py` 原型，一次性暴露 3 个阻塞点（符号 `if`、非循环形态 `for p, idx in taps`、`(1,1)` UB 标量内联读 SIGSEGV），修复后 953 行成品首版 L0 即通过；若先写全量成品，每次迭代都要重编译两个 kernel 变体（两段 prim_func）
  - 现状：SKILL.md Phase 4 只规定「先跑 L0、再扩 L1/L2/Boundary」，Phase 2 直接要求生成完整 kernel；对**两段/多段**结构没有任何「先分别验证每段可编译」的要求
  - 旁证：本任务两条新 trap 条目（TRAP-primfunc-static-branch-and-loop / TRAP-parallel-scalar-broadcast-segv）的症状（parser 拒 / codegen SIGSEGV）都只在编译期暴露，独立原型是最小成本的隔离手段（`python repro/<id>.py` 即是同思路的知识域化）
  - **第二证（2026-10-31 追加，task fft_c2c_vec-fft_c2c_vec-20261031T103133Z，不计确认——R 类恒人工门）**：该任务无原型阶段，直接写正式 kernel 后逐个形态跑探针，RETROSPECTIVE.md Stage 3 自述「十余次探针」才收敛到「复合 RHS 静默写 0 / LUT 非对齐 abort」两条根因；若先用 ≤20 行原型 + 首编后统计 IR 向量指令数（`hivm.hir.v*` / `arith.*f`），两类失效各 2–3 次探针即可定位。补充的判定点：原型不只验证「编译与数值」，还须在**首次编译成功后立即统计 IR 指令构成**（编译成功 ≠ 算对，也 ≠ 向量化）。
- repro: none（流程规则；复现条件 = 两段 kernel 任务检查 Stage 3 是否在正式文件前存在原型/分段编译记录）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+3ed7c914 / CANN 9.0.0 / 2026-10-31
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 在现有第 1 项之前插入第 0 项）
    定位锚: |
      ### Phase 2：生成 kernel
      1. 按 DESIGN.md §3 API 映射 + §6 循环结构生成 `@tilelang.jit(target="npuir")` kernel。
    old 文本: |
      ### Phase 2：生成 kernel
      1. 按 DESIGN.md §3 API 映射 + §6 循环结构生成 `@tilelang.jit(target="npuir")` kernel。
    new 文本: |
      ### Phase 2：生成 kernel
      0. **多段 kernel 或首次使用新形态时，先写最小原型**：在 `/tmp/opencode/` 用 ~60 行原型验证「编译 + 与 golden 的数值」，再写正式文件（含测试）。多段（两段及以上）`prim_func` 须**每段各自可单独编译并跑一次**（两段都命名 `main` 时也要分别起进程）。实测收益：一次性暴露符号 `if`、非循环形态、`(1,1)` UB 标量内联读 SIGSEGV 三个阻塞点，修复后 953 行成品首版 L0 即通过；反之每次迭代都要重编译全部 kernel 变体。原型阶段命中的 parser/codegen 现象按 `tilelang-op-optimize` skill 的 pattern-library traps-compiler.md / traps-runtime.md 条目记录（条目缺失则作为新陷阱候选交回 conductor）。
      1. 按 DESIGN.md §3 API 映射 + §6 循环结构生成 `@tilelang.jit(target="npuir")` kernel。
    动机: 本任务三个阻塞点全部在编译期、且都是「整文件重编译才能发现」的形态；原型把每个阻塞点的发现成本从「一次全量实现 + 测试」压到「60 行 + 一次编译」，对多段 kernel 尤其明显（段数与变体数相乘）。
- status: pending
- confirmations: -/-
- created_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0162
- type: R
- title: capability-gaps 字段速查：「若补齐能力 → X%」类外推数字必须附估算口径 + 逐项假设 + 可复算点
- evidence:
  - CG-2026-0018 的首证 `blocked_algo` 记「若跨核归约免费，b4≈53%、b8≈50–57%、b1≈14%，tune 平均 ≈39–41%」，只有结论数字；第二证任务（task engram_decode_mcore-engram_decode_mcore-20261031T012849Z）以路线无关上界给出 **≤22.2%**，偏差 1.8×
  - 偏差来源逐项可核（该任务 DESIGN §5.6.3）：① 多 block 发射地板按 1-block 值套用；② 每块 DMA 发射成本未计入（b4 scalar 4.83µs，占时延 20%）；③ 向量效率按峰值折算（实测 mul+reduce 两遍下 ≈31% 峰值）；④ 跨步列读按连续读带宽计（26–30 vs 68 GB/s）；⑤ 备选路线未测
  - 影响：该旧值是「多核重做能否达标」的关键判据输入，偏高 1.8× 会把「不可达」误判为「接近可达」，直接改变是否投入结构改造的决策
- repro: none（登记簿字段规范文本；复现条件 = 抽查任一 `blocked_algo` 含百分比的条目，检查是否附估算口径与逐项假设）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+3ed7c914 / CANN 9.0.0 / 2026-10-31
- target_doc: .agents/evolution/capability-gaps.md
- delta: |
    动作: update（字段速查的 `blocked_algo` 行改为「结论 + 口径 + 假设」三段式）
    定位锚: "blocked_algo: <受阻的算法/算法族与结论形态：无法实现 / 被迫换算法（换成什么）/ 性能上限（差多少）>"
    old 文本: |
      blocked_algo: <受阻的算法/算法族与结论形态：无法实现 / 被迫换算法（换成什么）/ 性能上限（差多少）>
    new 文本: |
      blocked_algo: <受阻的算法/算法族与结论形态：无法实现 / 被迫换算法（换成什么）/ 性能上限（差多少）>
        # 若含「补齐能力后可达 X%」类外推数字，必须同时给出：估算口径（公式 + 各分项取法）+
        逐项假设（带宽/发射/向量效率/并行度 各按什么取值）+ 可复算点（哪份工件能逐步复算）。
        只给结论数字的外推视为未验证假设，引用方须标注「未实测」并降级使用。
        # 教训：CG-2026-0018 的「≈39–41%」因未附口径与假设，被后续任务实测订正为 ≤22.2%（偏差 1.8×）。
    动机: 能力缺口条目里的百分比外推会直接进入设计决策（是否投入跨核重做），但它的可信度此前完全依赖作者自觉；把「口径 + 假设 + 可复算点」写入字段规范后，外推与实测走同一证据标准，且订正时有据可循。
- status: pending
- confirmations: -/-
- created_by: task engram_decode_mcore-engram_decode_mcore-20261031T012849Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0163
- type: D
- title: npuir 向量化入口门槛是「`T.Parallel` 体恰 1 条 BufferStore」（不是索引形态）——蝶形按此形态实测产出 `vadd/vsub/vmul/vdeinterleave`、`aiv_vec_ratio` 1.2e-4→0.093–0.105（≈800×）
- evidence:
  - task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 的 RETROSPECTIVE.md Stage 3 Value Point Proposals 第 1 行 + Stage 2 Transferable Lessons 第 1 条（四道闸逐条核对：循环 kind 过滤 / store 下标 / load 下标 / RHS 可分解）
  - 同任务实测 IR（`/home/developer/.tilelang/cache/*/kernel.mlir`，n=8）：`vadd 10 / vsub 14 / vmul 8 / vdeinterleave 4`、`arith.mulf = 0`、`scf.for = 1`；L0/L1 全绿（n=16…1024，max_diff ≤ 2e-5）
  - 源码条款：`src/transform/npu_loop_vectorize.cc` 的 `VectorizeSingleStatement`（入口）/ `L1184`（store 下标裸 Var 或标量常数）/ `L251`（load 下标不变量或单循环变量单位步长）/ `L1237`（只有 `kParallel` 进 `current_loop_vars_`，`T.serial` 外层早退）/ `L1250`（RHS 可分解）
  - **口径订正对象**：`examples/TileOPs/tileops/kernels/spectral/fft_c2c/fft_c2c_kernel/perf_opt/perf_feedback.md`「§1.6.2 断言已全部向量化被推翻：向量化完全未发生」+ `_fft_c2c_kernel_DESIGN.md` §1.6.3 的「必须核内重排（UB alias）才能向量化」——实测门槛在体形态，不在索引形态；alias 不是前提（每级独立 buffer + 仿射 3 下标读即可）
  - 与已合入条目 TRAP-parallel2d-bm-ge2-scalarize / CG-2026-0015 的关系：那条的触发器是 2D 并行维 bm≥2，本条是 1D `T.Parallel(h)` 单 store 体的入口条件，两者互补不重复（已在 CG-2026-0015 加交叉引用注）
- repro: repro-missing（复现条件：n=8 的 Stockham 第 0 级写成「`for i in T.Parallel(h): D[0,0,i] = S0[0,0,i] + S0[1,0,i]`」等 4 个单 store 循环，编译后统计 kernel.mlir 的 `hivm.hir.v*` 与 `arith.*f` 计数并跑 msprof 读 `aiv_vec_ratio`；断言只锁方向——向量指令数 >0 且 `arith.mulf == 0`）
- toolchain_stamp: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362（Expert 模式）/ 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    add 条目 PL-1.24-vec-admission-single-store（置于 layout.md 末尾，PL-1.17 之后；与 TRAP-parallel2d-bm-ge2-scalarize 互链）：

    ---
    id: PL-1.24-vec-admission-single-store
    kind: pattern
    family: [general, spectral, elementwise]
    apis: [T.Parallel, T.vadd, T.vsub, T.vmul, T.deinterleave]
    dtype: [fp32]
    device: Ascend910_9362
    status: verified
    origin_task: fft_c2c_vec-fft_c2c_vec-20261031T103133Z
    toolchain: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31
    repro: repro-missing
    ---

    ### 向量化入口的四道闸：体恰 1 条 BufferStore 是入口条件，索引形态是后续条件

    **优化见解**：判断「这段循环能不能被 npuir 向量化」，按四道闸逐条核（`npu_loop_vectorize.cc`）：
    ① 循环 kind——只有 `kParallel` 进 `current_loop_vars_`，`T.serial` / `T.vectorized` 外层早退（`T.vectorized` 不是向量化的入口）；
    ② **体内恰 1 条 `BufferStore`**（入口条件；多条 store 同处一体时走嵌套循环递归分支，**不报错、静默标量化**）；
    ③ store 下标为裸 `Var` 或标量常数；④ load 下标化简为「循环不变」或「单循环变量单位步长」；RHS 可被 `DecomposeExpression` 拆单 op。
    第 ② 条最易漏——只核 ③④ 会得出「索引形态不可救」的错误结论（历史误判：FFT 蝶形被判「npuir 无法向量化」）。

    **实测（n=8 第 0 级 4 个单 store 循环）**：IR `vadd 10 / vsub 14 / vmul 8 / vdeinterleave 4`、`arith.mulf = 0`；`aiv_vec_ratio` 由基线 1.2e-4 → 0.093–0.105（≈800×），n=16…1024 数值全绿。
    **边界（重要）**：向量化成功 ≠ 值得向量化——每 butterfly 语句数从 4 膨胀到 10（复合 RHS 走单 op 拆分，见 TRAP-npuir-vector-composite-rhs）+ 后段级 lane 仅 2，净时延反而变差（见 constants.md CONST-vec-ratio-not-latency 与 CASE-fft-c2c-vec-design-blocked）。
- status: pending
- confirmations: 1/2
- created_by: task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0164
- type: D
- title: 向量率不是时延——`aiv_vec_ratio` ×800 而 Task Duration 反慢 2.0×：收益按「膨胀后语句数 × 平均 lane 宽度」估，不按向量率估
- evidence:
  - task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 的 RETROSPECTIVE.md Stage 3 Value Point Proposals 第 5 行 + msprof 三组实测
  - 实测（`msprof op` Task Duration 中位数）：n=64 B=1 **12.82 vs 6.42 µs（2.0× 慢）**；n=1024 B=1 172.4 vs 152.1 µs（1.13× 慢）；n=1024 B=40 175.0 vs 180.9 µs（1.03× 快）；同 kernel `aiv_vec_ratio` 0.093–0.105
  - 机制两侧可核：① 语句数 4 → 10/butterfly（单 op 拆分 + 4 次中转 UB 往返）；② 后段级 `h=2` → 2-lane，平均 lane 利用率 ≈5.5/64（n=64 时末两级 2/1-lane 占级数 1/3）
  - 与 constants.md CONST-vector-launch-overhead（~0.5 µs/op 发射）互补：该条给单 op 成本，本条给「op 数膨胀 × lane 折损」的净收益判据
- repro: repro-missing（复现条件：同一 Stockham 单级 kernel 的两实现——标量 4 语句蝶形 vs 向量化 10 语句（单 op 拆分）——在 h=2 / h=32 两级分别跑 msprof，读 Task Duration 与 aiv_vec_ratio；断言只锁方向：向量版 vec_ratio ≥100× 基线，且 h=2 级 duration ≥ 标量版）
- toolchain_stamp: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362（Expert 模式）/ 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 条目 CONST-vec-ratio-not-latency（置于 CONST-vector-launch-overhead 之后，互链）：

    ---
    id: CONST-vec-ratio-not-latency
    kind: constant
    family: [general, spectral, elementwise]
    apis: [T.Parallel, T.vmul]
    dtype: [fp32]
    device: Ascend910_9362
    status: verified
    origin_task: fft_c2c_vec-fft_c2c_vec-20261031T103133Z
    toolchain: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31
    repro: repro-missing
    ---

    ### 向量率 ≠ 时延：语句膨胀与 lane 宽度共同决定净收益

    实测（同一 kernel，`aiv_vec_ratio` 1.2e-4 → 0.093–0.105，Task Duration 中位数）：

    | workload | 向量化 | 标量基线 | 比 |
    |---|---:|---:|---:|
    | n=64 B=1 | 12.82 µs | 6.42 µs | 2.0× 慢 |
    | n=1024 B=1 | 172.4 µs | 152.1 µs | 1.13× 慢 |
    | n=1024 B=40 | 175.0 µs | 180.9 µs | 1.03× 快 |

    两项机制（均可从代码侧核）：① **语句数膨胀**——复合 RHS 会被静默写 0（TRAP-npuir-vector-composite-rhs），拆成每句 1 op 后每 butterfly 由 4 条变 10 条、另加 4 次中转 UB 往返；② **lane 折损**——变几何算法的后段级并行度小（本例 h=2 → 2-lane，平均 ≈5.5/64）。
    判据：向量化候选的收益估计式取 `语句数(膨胀后) / lane 宽度`，与标量版的 `语句数 / 1` 相比；仅当 `aiv_vec_ratio` 提升且该比值下降时才预期净收益。窄 lane 级（h ≤ 4）宜保留标量或按级分派。
- status: pending
- confirmations: 1/2
- created_by: task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0165
- type: D
- title: UB 规划的 liveness 复用不跨块——「每级一块独立缓冲」的真实峰值 = 跨级存活总和（n=4096 需 464 KB > 192 KB）
- evidence:
  - task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 的 RETROSPECTIVE.md Stage 3 Value Point Proposals 第 4 行（编译报错原文 `ub overflow, requires 3802112 bits while 1572864 bits available`）+ 逐尺寸实测（n≤1024 通过 / n=2048 221 KB 溢出 / n=4096 464 KB 溢出）
  - 级内短命中转 11 份 → 1 份（复用生效），跨级存活的 11 对 `(2,ng,h)` 一份都不合并——即 `PlanAndUpdateBufferAllocationLocation` 的复用边界是「块」
  - 被推翻的设计假设：DESIGN.md §4.5「liveness 复用 → 128 KB」+ §9 R1 降级链（radix-4 288 KB / radix-8 224 KB / flat ping-pong 不减份数）
  - 能力缺口：capability-gaps.md CG-2026-0023（open）；与 TRAP-UB-multibuffer-inflation（膨胀链）机制不同，不合并
- repro: repro-missing（复现条件：`t = 2..12` 级各 alloc 一对 `(2,2^s,n/2^{s+1})` fp32 的串行链 kernel（不做任何 copy），n=2048/4096 各编一次；断言 `ub overflow` 报文出现且 requires 位数为级数 × 每级字节）
- toolchain_stamp: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362（Expert 模式）/ 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-ub-no-cross-block-liveness（置于 TRAP-UB-multibuffer-inflation 之后；互链 CG-2026-0023）：

    ---
    id: TRAP-ub-no-cross-block-liveness
    kind: trap
    family: [general, spectral, reduction]
    apis: [T.alloc_ub, T.serial]
    dtype: [fp32, fp16]
    device: Ascend910_9362
    status: verified
    origin_task: fft_c2c_vec-fft_c2c_vec-20261031T103133Z
    toolchain: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31
    repro: repro-missing
    ---

    ### UB liveness 复用不跨块：多级/多阶段「每级一块」按跨级存活总和算峰值

    **现象**：`t` 级 Stockham 每级 alloc 一对 `(2,ng,h)` 目的缓冲，级间串行推进。实测 `PlanAndUpdateBufferAllocationLocation` 只回收级内 4 个短命中转（11 份 → 1 份），**跨级存活的 11 对级缓冲一份都不合并**；n=2048 需 221 KB、n=4096 需 464 KB（`ub overflow, requires 3802112 bits while 1572864 bits available`）。
    **判据**：缓冲存活期跨两个块（循环嵌套）⟹ 不与其他同尺寸缓冲合并，峰值 = Σ(跨级存活)。设计期纸面写「liveness 复用 → 两级」在多级网络类算法上不成立。
    **降级链实测同样不成立**：radix-4（5 级）288 KB、radix-8（3 级）224 KB、flat ping-pong（份数不减）均超 192 KB ⟹ 该类算法在本工具链的可交付上限由容量定，不由 tiling 参数定。
    **诊断/成本**：该失败在编译期确定（秒级），设计期取 workload 最大 n 编一次即可判定，不必先写完整测试套件。
- status: pending
- confirmations: 1/2
- created_by: task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0166
- type: R
- title: design-review 维度 8 增补「向量化形态 → pass 条件链逐条核对」检查项（单 store 体入口条件为必核项，不得只核 store/load 下标）
- evidence:
  - task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 的 RETROSPECTIVE.md Stage 2 Skill Flow Issues 第 1 行：首轮检视只核到 store 下标规则（`L1184`）与 load 规则（`L251`），漏掉入口条件「循环体恰 1 条 BufferStore」→ REVIEW v1 判不通过、修订后复检通过（Stage 1 因此重试 2 次，`stage_retry_count[1]=2`；Stage 1 三次 attempt 合计 2667s）
  - 该遗漏的代价链：v1 设计形态不满足入口条件 → 若未在 Stage 2 拦下，Stage 3 会以「静默标量化」形态交付（pass 不报错）
  - 维度 8 现有表中「循环/标量点缺向量化替代分析或保留理由不充分」只查「有没有做替代分析」，不查「判据与 pass 条件链是否一一对应」
- repro: none（检查项文本；复现条件 = 对 fft_c2c_vec 的 v1 DESIGN 按新检查项复核，应命中「只核 2/4 道闸」）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 判 fail 表新增一行）
    定位锚: "| 循环/标量点缺向量化替代分析或保留理由不充分 | 维度 8 fail，逐点列出位置（§3.3/§6）与可尝试的替代方向 |"
    old 文本: |
      | 循环/标量点缺向量化替代分析或保留理由不充分 | 维度 8 fail，逐点列出位置（§3.3/§6）与可尝试的替代方向 |
    new 文本: |
      | 循环/标量点缺向量化替代分析或保留理由不充分 | 维度 8 fail，逐点列出位置（§3.3/§6）与可尝试的替代方向 |
      | **向量化形态 → pass 条件链未逐条核对**（设计断言「该循环会被向量化」或「不可能向量化」时） | 按 `npu_loop_vectorize.cc` 四道闸逐条给出源码行号与结论：① 循环 kind 过滤（只有 `kParallel` 进 `current_loop_vars_`，`L1237`；`T.serial`/`T.vectorized` 早退）② **体内恰 1 条 BufferStore**（入口条件；多条 store 静默走嵌套循环递归、不报错）③ store 下标裸 `Var`/标量常数（`L1184`）④ load 下标不变量或单循环变量单位步长（`L251`）+ RHS 可分解（`L1250`）。缺任一条的核对记录 → fail 并附缺失条目 |
    动机: fft_c2c_vec 首轮检视只核 ③④，漏 ② 入口条件，得出「必须核内重排（UB alias）才能向量化」的错误前提，直接导致 v1 设计被否 / Stage 1 重试 2 次；四道闸固化为表行后该形态可机械核对（正确形态实测产出 800× 向量率，见 queue VP-2026-0163）。
- status: pending
- confirmations: -/-
- created_by: task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0167
- type: R
- title: op-develop Phase 2 增补「容量判定点」——workload 最大 n 先编一次，`ub overflow` 编译期即返回 DESIGN_ERROR，不必先写完整测试套件
- evidence:
  - task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 的 RETROSPECTIVE.md Stage 3 Skill Flow Issues 第 2 行：设计 §4.5 的 128 KB 预算到 L0-2 才撞上 464 KB（`ub overflow`）；此时已写完实现与部分测试套件，Stage 3 单次 attempt 4667s 后判 BLOCKED_DESIGN
  - 该判定是纯编译期、秒级可得的量（`requires N bits while 1572864 bits available` 报文直接给出精确需求）
  - 与 VP-2026-0161（最小原型先行）互补：0161 管「形态能否编译/算对」，本条管「最大 workload 能否装下」——两条都在正式文件之前，但作用的量不同（形态 vs 容量）
- repro: none（流程规则；复现条件 = 多级/大 shape 任务检查 Stage 3 是否在写测试套件前做过「最大 n 一次编译」）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 生成 kernel 的编号列表末尾追加一条）
    定位锚: "3. 遵循项目根 AGENTS.md："
    old 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
    new 文本: |
      （在第 3 条之后追加）
      4. **容量判定点**：kernel 成形后、写测试套件之前，用 **workload 最大 shape**（最大 n / batch / 通道数）先编一次。编译报 `ub overflow, requires N bits while M bits available` / `cbuf overflow` / `cc overflow` 等容量类失败时，直接按 `DESIGN_ERROR` 返回（附报文原文与按报文字节复算的峰值），不必先补完 `L0` 套件——该类失败是编译期确定的，重跑测试不会改变结论。设计侧容量判据见 `tilelang-op-optimize` skill `references/pattern-library/traps-compiler.md`（TRAP-ub-no-cross-block-liveness / TRAP-UB-multibuffer-inflation）。
    动机: 本任务的设计期容量估算（liveness 复用 → 128 KB）与实测（464 KB）差 3.6×，该差异一次最大 shape 编译即可暴露（秒级）；实际到 L0-2 才发现，使 Stage 3 单次 4667s 后仍以 BLOCKED_DESIGN 收尾。
- status: pending
- confirmations: -/-
- created_by: task fft_c2c_vec-fft_c2c_vec-20261031T103133Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0168
- type: D
- title: BSHD attention backward 的 `dq` 跨核归约：逐行 1-D `T.atomic_add` 单价约为 rank-2 跨步区域原子的 7×（224 vs 32 ns/行），且它挤占 GM 通道使 `aiv_mte2` 同样翻倍——本 kernel 上该项占 block 墙钟 51%
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261031T051147Z 的 perf_opt/opt_log.md §9.1 VP-A + §6 Final 表：`dq` 改 rank-2 跨步区域原子（声明为 `(B·S, H·D)` 视图 + 工厂内薄闭包 `dq.view(B*S, H*D)`，对外 9 张量签名与 wrapper 零改动）后每 block `aiv_mte3` 220.3→4.8 µs、`aiv_mte2` 141.1→18.3 µs、`aic_scalar_wait_id2` 409.1→23.6 µs、Task Duration 5865.6→1995.8 µs；8 档几何均值 22966.3→8001.4 µs（−65.9%），官方 Ratio 1.4610%→4.2375%
  - 机制归因：逐行原子把 GM 写通道打成行粒度流量，原子流量占满通道后装载（mte2）连带停顿——故 `mte2`「带宽不足」是假象（互链 CONST-gm-to-ub-bw-dilution 的稀释口径）
  - 正确性边界（同任务 §6.1）：rank-reduced 4-D 写法会把跨步目标当连续（第 1 轮 `r1_atomtile` 静默错值）；rank-2 视图写法的目标是真正的 2-D 区域，行距 = 视图 pitch，L0/L1 全量（groups 1/2/4/8、n_kv 1/4、D 64/128、causal/非 causal）通过
  - 同算子先例（另一目录、另一器件）：`examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §Round 5 记同形态 −45.2%
- repro: repro-missing（复现条件：同一 kernel 两分支——A 声明 `dq` 为 4-D 逐行 `T.atomic_add`；B 声明为 `(B·S, H·D)` 2-D 视图 + 一次 `[BN, dim]` 区域原子。断言方向：两分支 L0/L1 数值一致，A 的 `aiv_mte3_time` 与 `aiv_mte2_time` 同时显著高于 B）
- toolchain_stamp: tilelang 0.1.2+9738284cd6 + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目 PL-1.22-dq-region-atomic（置于 PL-1.13-aiv-dup-subid-split 之后）：

    ---
    id: PL-1.22-dq-region-atomic
    kind: pattern
    family: [attention, mixcv]
    apis: [T.atomic_add, T.copy, T.gemm]
    dtype: [fp16, bf16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_gqa_bwd_kernel-20261031T051147Z
    toolchain: tilelang 0.1.2+9738284cd6 + CANN 9.0.0 / Ascend910B3 / 2026-10-31
    repro: repro-missing
    ---

    ### 1.22 BSHD `dq` 跨核归约走 rank-2 跨步区域原子

    **形态**：attention backward 的 `dq` 需跨 kv-block 归约时，把 `dq` 声明为 `(B·S, H·D)` 视图（工厂内薄闭包 `view(B*S, H*D)`，对外张量签名不变），每 block 一次 `[BN, dim]` 区域 `T.atomic_add`；**不要**逐行发 1-D 原子。
    **机制（D 类实测）**：单价 224 ns/行（逐行 1-D）vs 32 ns/行（rank-2 区域），约 7×；且逐行原子把 GM 写通道打成行粒度流量，原子流量占满通道后**装载通道连带停顿**——`aiv_mte2` 141.1→18.3 µs 的下降说明「mte2 带宽不足」是假象（判读见 CONST-gm-to-ub-bw-dilution）。本 kernel 上该项占 block 墙钟 51%。
    **实测（910B3，GQA bwd，8 workload）**：`aiv_mte3` 220.3→4.8 / `aiv_mte2` 141.1→18.3 / `aic_scalar_wait_id2` 409.1→23.6 µs per block；5865.6→1995.8 µs，Ratio 1.421%→4.139%（8 档几何 1.461%→4.2375%）。
    **正确性边界**：rank-reduced 4-D 写法会把跨步目标当连续（静默错值）；rank-2 视图的目标是真正 2-D 区域（行距 = 视图 pitch），fwd/bwd 全量 L0/L1 通过。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_bwd_kernel-20261031T051147Z 2026-10-31
- confirmed_by: -
- decided_by: -
- decided_note: -

## VP-2026-0169
- type: D
- title: 静态 UB 加和不是 UB 预算——`BN=64, dim=128` 静态加和 240 KB > 192 KB 仍编译通过且生效（分配器按生命周期复用），此前据静态加和作出的「BN=64 不可行」否决被推翻
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261031T051147Z 的 perf_opt/opt_log.md §9.1 VP-B + §6.2：`_compile_probe*.py` 编译成功、workspace 155 648 B/block；`BN` 32→64 实测 −1.4%（短档）/ −4.3%（长档）
  - 被推翻的旧结论出处（同任务设计文档，非知识库条目）：`_gqa_bwd_kernel_DESIGN.md` §4.5「BN=64 静态 UB 加和 240 KB > 192 KB ⇒ 不可行」；该否决在上一轮会话（`perf_opt/history/opt_log_prev.md` §Round 3）被当作方向预算否决依据，本轮同臂实测推翻
  - 与既有条目关系：`traps-compiler.md#TRAP-UB-multibuffer-inflation`（auto-multi-buffer 使 UB 膨胀 ~1.7×）与 `constants.md#CONST-ub-budget-vs-pattern-cap`（`ub overflow` 归因到共享 192 KiB 预算）都描述「实际占用 ≠ 名义尺寸」，本条补其反向：**静态逐 buffer 加和 ≠ 实际占用**（生命周期复用使实际远低于加和），故静态加和既不能用于否决，也不能用于预测余量
- repro: repro-missing（复现条件：同一 kernel 只改 `BN` 常量 32→64（`dim=128` 档），静态加和按 buffer 逐项求和得 240 KB > 192 KB，实际编译通过与运行；断言方向：编译成功 + workspace 实测远低于静态和）
- toolchain_stamp: tilelang 0.1.2+9738284cd6 + CANN 9.0.0 / Ascend910B3 / 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 条目 CONST-ub-static-sum-not-budget（置于 CONST-ub-budget-vs-pattern-cap 之后）：

    ---
    id: CONST-ub-static-sum-not-budget
    kind: constant
    family: [attention, mixcv, layout]
    apis: [T.alloc_shared, T.alloc_fragment]
    dtype: [fp32, fp16, bf16]
    device: 910B3
    status: verified
    origin_task: TileOPs-_gqa_bwd_kernel-20261031T051147Z
    toolchain: tilelang 0.1.2+9738284cd6 + CANN 9.0.0 / Ascend910B3 / 2026-10-31
    repro: repro-missing
    ---

    ### UB 静态加和不是预算（生命周期复用使实际占用可远低于加和）

    **数据点**：GQA bwd（BSHD，`BM=64`、`dim=128`）`BN=64` 的静态逐 buffer 加和 ≈240 KB > 192 KB，**编译通过且生效**（`dim=128` 档 −1.4%…−4.3%），profile 实测 workspace 155 648 B/block —— 分配器按生命周期复用 buffer，静态和只对「同一时刻同时存活」的集合有意义。
    **判据**：容量类否决（`这样一定溢出`）与容量预测（`这样还有余量`）都需先跑一次最大 shape 编译探针并注明口径（静态加和 / 分配器实际）；K-1 的两条既有条目描述同一事实的反面：TRAP-UB-multibuffer-inflation（实际 > 名义，1.7×）与 CONST-ub-budget-vs-pattern-cap（`ub overflow` 归因共享预算）。
    **误判代价（本任务）**：设计期据此否决 `BN=64`，使 Stage 4 才拿回该档收益；同形态误判在 `BM=128` 上再次出现（该处否决成立，但依据是 L0C 128 KB 的 fp32 `dk_f`/`dv_f` 预算，与 UB 无关）。
- status: merged
- confirmations: 2/2
- created_by: task TileOPs-_gqa_bwd_kernel-20261031T051147Z 2026-10-31
- confirmed_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z 2026-11-05（第二证，不同任务：4×64 KB UB 直接探针证明 `hivm-plan-memory` 按 live range 复用〔disjoint 编译通过且第 4 块 alias 到 slot 0；concurrent 256 KB 报 `ub overflow`〕；当前 kernel plan total = user 157,952 + 编译器 staging 38,656 = 196,608 B）
- decided_by: evolver
- decided_note: 2/2 合入 `constants.md#CONST-ub-alloc-live-range`（add；按第二证修正原 delta 的「静态和」泛化——本 kernel 零复用故 plan total = user sum + staging，并补 staging 税 38,656 B 与 G=2 容量门应用）。repro 由第二证补齐：`repro/CONST-ub-alloc-live-range.py`（4 缓冲 disjoint/concurrent 探针，机械转正 + py_compile PASS）。

## VP-2026-0170
- type: P
- title: BSHD 行粒度访存成本只与「行数」成正比——用 `BN` 扫描 + pipe 分项可把「行粒度流量」与「每迭代指令发射」两类瓶颈分离，二者只能分别用换形态 / 换结构解决
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261031T051147Z 的 perf_opt/opt_log.md §9.1 VP-C + Round 1 对比表：`BN` 32→64 后 `aiv_mte3` 220.3→217.9、`aiv_mte2` 141.1→139.6（行数不变 ⇒ 两项几乎不变），而 scalar / fixpipe / cube 项减半
  - 反面对照（同任务）：行数不变时只有**换形态**（逐行 1-D 原子 → rank-2 区域原子）能把 `mte3` 打下 45×（220.3→4.8 µs），`BN` 加倍无此效果
  - 消解后的残余（同任务 §2 第 3 条）：`aiv_scalar` 106.1 / `aiv_time` 150.3 = 70%，cube 占比仅 5.5% —— 发射/握手项对 `BM/BN`/`num_stages` 均不敏感（`num_stages=2` 第三次实测 <3%）。互补既有条目：PL-1.9-blockwidth（块宽摊减律作用于「每块固定成本 ÷ 元素数」），本条给的是**该律失效时的判别信号**
- repro: repro-missing（复现条件：同一 kernel 只扫 `BN∈{32,64}`，比较 4 个 pipe 分项——`mte2/mte3` 随行数变化而 scalar/fixpipe 随块数变化即判本条成立）
- toolchain_stamp: tilelang 0.1.2+9738284cd6 + CANN 9.0.0 / Ascend910B3 / 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add 小节「行粒度瓶颈 vs 指令发射瓶颈的分离判据」（置于「诊断上下文模板」之前，与已重叠判据 PL-1.21 相邻）：

    ### 行粒度瓶颈 vs 指令发射瓶颈的分离判据

    profile 显示 `aiv_mte2/mte3` 为主项时，先分辨它是**行粒度流量**还是**每迭代固定发射**：

    1. **`BN` 扫描 + pipe 分项**：仅放大块宽（行数不变）时，`mte2/mte3` 基本不动而 scalar/fixpipe 减半 ⇒ 两个访存项由行数决定，只能换形态（如逐行 1-D 原子 → rank-2 跨步区域原子，45× 差）；反之若各 pipe 随块数等比下降 ⇒ 属摊减类，按 PL-1.9-blockwidth 加宽即可。
    2. **残余判据**：修完访存形态后若 `aiv_scalar/aiv_time` 走高（本例 70%）且 cube 占比 <10%，瓶颈已转到每迭代发射 + Developer 双 AIV 冗余（PL-1.13），此时 `BM/BN/num_stages` 均不敏感（`num_stages=2` 第三次实测 <3%），结构级改动才有效。
    3. 报告口径：行粒度项按 µs/行 报价（本例 224 vs 32 ns/行），发射项按条数报价，两者不可相加成「访存总代价」。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_bwd_kernel-20261031T051147Z 2026-10-31
- confirmed_by: -
- decided_by: -
- decided_note: -

## VP-2026-0171
- type: R
- title: optimize Phase 0 首轮必查项增「同算子跨目录 perf_opt 案卷检索」——同族参照不只在 pattern-library，`examples/**/perf_opt/opt_log.md` 里已有同算子的 Stage-4 最优结构
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261031T051147Z 的 opt_log.md §9.1 BP_4：本任务 51% 的单点收益（`dq` rank-2 区域原子）与「40% 不可达」的结构性归因，其唯一有效参照是 `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §Round 5 / §10.4（同一算子 GroupedQueryAttention backward 的 Stage-4 交付档案，含 1016.8 µs / Ratio ≈8.2% 的 3-pass 实测）；该档案在首轮会话与本轮 Phase 0 的 pattern-library 索引里都未被引用——现有索引只覆盖 pattern-library 条目，不覆盖 `examples/**/perf_opt/`
  - 代价：参照缺失使首轮把预算投向参数级方向，且 [DESIGN_LIMIT] 的「结构性加速估计」缺少同算子锚点（本任务的 5.0x 估计最终依赖事后检索该档案）
  - 相邻提案：VP-2026-0041（morph ladder 参照实现对照定位法）规定「进入 plateau 前强制做参照结构 diff 检查」，但检索范围表述为「pattern-library cases.md + `examples/` 同类 + kb_search」——未点名 `perf_opt/opt_log.md` 与「同算子在其他 example 目录」这一最容易命中的子集
- repro: none（流程规则；复现条件 = 任一 optimize 任务检查 Phase 0 检索清单是否含跨目录 perf_opt 案卷）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+9738284cd6 + CANN 9.0.0 / Ascend910B3 / 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 0 第 4 条「首轮必查项」的子 bullet 列表末尾追加一条）
    定位锚: "   - **编译器陷阱版本戳与 origin_task 核对**："
    old 文本: |
      - **编译器陷阱版本戳与 origin_task 核对**：pattern-library traps-*.md 的陷阱结论绑定工具链版本与来源任务（origin_task）——若 tilelang 源码被修改/重编译过，相关结论自动视为待重验，不得直接引用；引用命中条目时在 opt_log 中标注其 origin_task，据此判别可信度；
    new 文本: |
      - **编译器陷阱版本戳与 origin_task 核对**：pattern-library traps-*.md 的陷阱结论绑定工具链版本与来源任务（origin_task）——若 tilelang 源码被修改/重编译过，相关结论自动视为待重验，不得直接引用；引用命中条目时在 opt_log 中标注其 origin_task，据此判别可信度；
      - **同算子跨目录 perf_opt 案卷检索**：检索范围除 pattern-library 外须含 `examples/**/perf_opt/opt_log.md`（尤其**同一算子在其他 example 目录**下的历史 Stage-4 交付档案，含 perf_records 与胜出结构描述）。这类档案是「同算子最优结构」最直接的锚点，pattern-library 只收录其中被抽象过的条目，未收录的胜出结构（如 3-pass 单引擎相位拆分）只存在于该档案；检索命中时在 opt_log 中记录路径与章节，并按其结构形态做一次 diff（同 VP-2026-0041 的 morph 对照，不照搬）。
    动机: 本任务 51% 的单点收益与「目标不可达」的结构性归因，均来自事后才检索到的同算子 perf_opt 档案（`examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §Round 5/§10.4）；首轮 Phase 0 未命中，预算被参数级方向消耗。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_bwd_kernel-20261031T051147Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0172
- type: R
- title: design-review 容量维度补「口径标注 + 编译探针」——UB 静态加和只能否决「同时存活」的集合，不能用作容量预算的唯一依据
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261031T051147Z：`_gqa_bwd_kernel_DESIGN.md` §4.5 以「`BN=64, dim=128` 静态 UB 加和 240 KB > 192 KB」否决 `BN=64`，Stage 4 实测同配置编译通过且生效（workspace 155 648 B/block，−1.4%…−4.3%），否决被推翻（见 queue VP-2026-0169）
  - 同任务 `BM=128` 的否决成立但依据不同（L0C 128 KB 的 fp32 `dk_f`/`dv_f` 预算），说明「容量否决」需按 buffer 所在层级逐条给依据，无法用一个总数替代
  - 机理性旁证（知识库既有）：`traps-compiler.md#TRAP-UB-multibuffer-inflation`（实际占用可超名义 ~1.7×）与 `constants.md#CONST-ub-budget-vs-pattern-cap`（`ub overflow` 归因共享预算）——两个方向都已实测存在，故单侧加和既不能否决也不能保证余量
  - 相邻提案：VP-2026-0167（op-develop Phase 2 增「容量判定点」——最大 shape 先编一次）覆盖 Stage 3 侧；本条补 Stage 2 检视侧（设计文档的容量结论在检视时应核对口径）
- repro: none（流程规则；复现条件 = 设计文档出现「静态加和 > 容量 ⇒ 不可行」类结论时，检查是否附编译探针与口径标注）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+9738284cd6 + CANN 9.0.0 / Ascend910B3 / 2026-10-31
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（§4 检查表中的容量两行，逐行追加口径要求）
    定位锚: "| UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） |"
    old 文本: |
      | UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） |
    new 文本: |
      | UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB）；**加和须标注口径**（静态逐 buffer 和 / 按生命周期同时存活集合），且「静态和超限 ⇒ 不可行」类结论须附最大 shape 的一次编译探针结果（附报文与实测占用字节）——静态加和可远高于分配器实际占用（实测 240 KB 静态 vs 155 648 B/block 实际，编译通过） |
      | L0C 容量 | Cube 类算子 `block_M × block_N × sizeof(accum)` ≤ 128KB；否决须按 buffer 所在层级逐条给依据（L0C 与 UB 的预算不可互推） |
    动机: 本任务设计期以静态 UB 加和否决 `BN=64`，Stage 4 实测推翻并拿回该档收益；同源误判在 `BM=128` 上再次出现（后者否决成立但依据是 L0C，非 UB），说明容量结论需按层级 + 口径逐条留痕。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_bwd_kernel-20261031T051147Z 2026-10-31
- decided_by: -
- decided_note: -

## VP-2026-0173
- type: D
- title: 1d RoPE 的真实 kernel 固定地板 = 3.52 µs（1.74 launch/ramp + 1.78 每核骨架，40 AIV / 7 描述符），非空 kernel 的 1.9–2.85 µs；cos/sin 表 2.10 MB 另计 1.44 µs——该族「贴带宽地板」判断被推翻
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261031T184833Z 的 perf_opt/opt_log.md §3.1 探针表 + §6：8k-d128-bf16 总 6.35 µs 按五探针加法闭合 = 1.74（`Task Duration − aiv_time`，5 探针一致 1.73–1.83）+ 1.78（`tiny_ub` = 每核仅 1 行 GM→UB→GM，10 KB；40 核唤醒 + 7 个 DMA/UB 描述符 + 循环/guard）+ 0.48（x/y 4.19 MB，L2 8.7 TB/s）+ 1.44（cos/sin 表 2.10 MB，1.45 TB/s）+ 0.91（rope 6-op 链相对 2-op 增量，0.23 µs/iteration）
  - 同探针另三档（copy/mem/full，µs）：2k-d64 2.98/3.38/3.58、2k-d128 3.21/3.61/3.88、4k-d128 3.31/4.32/4.92、8k 4.00/5.44/6.35——`2k-d64` 的 3.58–3.80 即地板的 ×1.02
  - 被推翻的旧判断：同算子上一轮 opt_log 据「读 1.6–2.0 TB/s / 写 2.6–2.9 TB/s（`*_active_bw`）」判 1d 已贴 roofline；本轮证明带宽项在 8k 只占 0.48/6.35 = 7.6%，cos/sin 与固定项才是主项
  - 与既有条目关系（**双戳并列，不删旧值**）：queue VP-2026-0142 的 launch 地板为 1.89 µs（首戳）/ 1.570–2.850 µs（第二戳 FLOOR(1/8/32/40)），口径是**空 kernel**；本条 3.52 µs 是**本算子 on-shape 骨架**（多出 7 个描述符 + 循环 guard），两者不矛盾（3.52 − 1.74 = 1.78 即骨架项）。任务内旧的「~1.85 µs 纯搬运 + 1.9 µs 发射地板 ≈ 3.75 µs」估算已由探针证伪（实际 6.35 µs）
  - 官方口径对照：`examples/TileOPs/reports/tileops/20261101_035109_804812_RopeNeoxOp/benchmark.md`（对照 `20261017_172054_008103_RopeNeoxOp/`）——三档 Ratio 7.7318 / 25.2832 / 49.0393 → 均值 27.3514%（对照 8.3470 / 24.8640 / 48.9109 → 27.3740%，持平）
- repro: repro-missing（复现条件：与 current best 同 buffer 集/tile 推导/DMA 形态的减法探针，`ROPE_PROBE ∈ {tiny_ub, copy, mem, full}` 切换——`tiny_ub` 每核仅 1 行 10 KB 搬运并保留 40 核网格 + 7 描述符 + guard；`msprof op --launch-count=20 --warm-up=5` 取 Task Duration 中位数，并用 `Task Duration − aiv_time` 交叉核对发射项。断言只锁量级与相对关系：`tiny_ub ∈ 3.2–3.9 µs`、`mem − copy ≈ 1.2–1.7 µs`、`full − mem ≈ 0.7–1.1 µs`。任务内探针 `perf_opt/_probe_1d_breakdown.py` 因无条目 ID/版本戳头且依赖 `_bench_kernel.py` 编排、167 行超 100 行预算，ED-C 机械转正不成立，故本条降级入队）
- toolchain_stamp: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV，`_CORE_NUM=40`）/ 2026-11-01；TILELANG_ASCEND_MODE=Developer、torch_npu 2.7.1
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 条目 CONST-rope-1d-fixed-floor-910B3（置于 CONST-launch-floor-910B3 之后；与后者区别：后者是空 kernel 发射地板，本条是「同形态骨架」的 on-shape 固定地板，量级与适用面不同，不合并）：

    ---
    id: CONST-rope-1d-fixed-floor-910B3
    kind: constant
    family: [elementwise, position_encoding]
    apis: [T.copy, T.vmul, T.vsub, T.vadd]
    dtype: [bf16, fp16]
    device: 910B3
    status: verified
    origin_task: TileOPs-_make_rope_neox_1d-20261031T184833Z
    toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-11-01；TILELANG_ASCEND_MODE=Developer
    repro: repro-missing
    ---

    ### 1d RoPE（40 AIV 全开）的真实固定地板 = 3.52 µs，非空 kernel 的 1.9–2.85 µs

    **分解（8k-d128-bf16，`msprof op` Task Duration 6.35 µs，五探针加法闭合）**：

    | 项 | µs | 依据 |
    |---|---:|---|
    | launch / ramp | 1.74 | `Task Duration − aiv_time`，5 探针一致 1.73–1.83 |
    | 每核骨架（40 核唤醒 + 7 个 DMA/UB 描述符 + 循环/guard） | 1.78 | `tiny_ub` 探针（每核仅 1 行 10 KB GM→UB→GM） |
    | x/y 数据 4.19 MB | 0.48 | L2 驻留 8.7 TB/s |
    | **cos/sin 表 2.10 MB** | **1.44** | 1.45 TB/s（同 kernel、不同张量，带宽差 6×） |
    | rope 6-op 向量链 | 0.91 | 相对 2-op 的增量，0.23 µs/iteration |
    | **合计** | **6.35** | — |

    **事实与判据**：① 该族**不是带宽受限**——x/y 的 4.19 MB 只值 0.48 µs，主项是 3.52 µs 固定地板（1.74 + 1.78）与独立的 cos/sin 表流量；② `2k-d64` 实测 3.58–3.80 µs = 地板 ×1.02，无带宽余量可挖；③ 用 `Task Duration − aiv_time` 可把发射项从骨架项中分离，二者在 40 AIV 满开时同量级，合并报「launch 地板」会低估约 1.8 µs；④ cos/sin 与 x/y 的有效带宽不同（1.45 vs 8.7 TB/s），流量估算按同一带宽会低估该项约 6×。

    **与 launch 地板的双戳并列**：VP-2026-0142 记空 kernel 地板 1.89 µs（首戳）/ FLOOR(1/8/32/40)=1.570/1.600/2.520/2.850 µs（第二戳）；本条 3.52 µs 为「同形态骨架」地板（多 7 个描述符 + 循环 guard），二者口径不同、各自成立；Ratio 上界推导用哪一档须按是否含本 kernel 的骨架形态选择。

    **Ratio 上界算术（三档，官方口径）**：`neox-1d-2k-d64`(7.7318%, ideal 0.2938 µs) / `neox-1d-4k-d128`(25.2832%, ideal 1.1985 µs) / `neox-2d-b2-s2k-h32`(49.0393%, ideal 22.244 µs) 各自 `ideal/1.897` 上界 15.5% / 63.1% / 100% ⇒ 均值上限 (15.5+63.1+100)/3 = **59.5% < 60%**（差 0.5 点，最乐观口径）。按本条 3.52 µs 地板重算为 8.3% / 34.0% / 86.3%（均值 ≤42.9%）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_rope_neox_1d-20261031T184833Z 2026-11-01
- confirmed_by: -
- decided_by: -
- decided_note: -

## VP-2026-0174
- type: D
- title: `T.Pipelined(num_stages=2)` 在 UB 预算紧的逐元素算子上净收益为负（实测 −6.8%）：2 级多缓冲以 tile 减半、波次翻倍为代价，而该族主项恰是每波固定开销
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261031T184833Z 的 perf_opt/opt_log.md §3.3：同一 base（`r3_6buf`，tile 69 行 / 3 波 / 6.13 µs）只把 `T.serial` 换 `T.Pipelined(iters, num_stages=2)` → UB 只够 tile 26 行 → 波次 3→8、DMA 描述符翻倍 → **6.78 µs（−6.8%）**，L0 仍 PASS
  - 机制：该族 `aiv_time` 中未隐藏的向量时间约 0.91 µs（8k），而每波固定开销约 0.82 µs；波次 3→8 新增的 5 波 ×0.82 ≈ 4.1 µs 远大于可重叠的 0.91 µs
  - 上一轮同算子把该项记为「未实证假设（预估 <5%）」，本轮为**实证更正**：方向为 −6.8%
  - 互补既有条目：`elementwise.md#PL-1.21-overlap-saturation-signal`（已重叠判据 `mte2+vec > Task`）——本条补「重叠手段本身消耗 UB 预算」的反面情形
- repro: repro-missing（复现条件：同一元素算子 kernel 两分支——A `T.serial(iters)`；B `T.Pipelined(iters, num_stages=2)`，两分支 tile 均由 UB 预算 clamp 推导。断言方向：B 的 `tile_rows` 约为 A 的 1/2、波次约为 2–3×，且 Task Duration 更高）
- toolchain_stamp: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add 条目 PL-1.23-ub-tight-pipelined-negative（置于 PL-1.21-overlap-saturation-signal 之后）：

    ---
    id: PL-1.23-ub-tight-pipelined-negative
    kind: pattern
    family: [elementwise]
    apis: [T.Pipelined, T.serial, T.copy]
    dtype: [bf16, fp16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_make_rope_neox_1d-20261031T184833Z
    toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3 / 2026-11-01
    repro: repro-missing
    ---

    ### 1.23 UB 预算紧的逐元素算子：`num_stages=2` 多缓冲净负收益（−6.8%）

    **形态**：单参数 tiling 的逐元素 kernel（persistent 40 核 + 核内 `T.serial(iters)` 网格跨步、tile 由 UB clamp 推导），把 `T.serial` 换成 `T.Pipelined(iters, num_stages=2)` 后 tile 减半（69→26 行）、波次 3→8、DMA 描述符翻倍，实测 6.13→**6.78 µs（−6.8%）**。
    **机制归因**：该族未隐藏的向量时间约 0.91 µs（占 8k 总时延 14%），而每波固定开销约 0.82 µs；波次 ×2.7 新增的固定开销（≈4.1 µs）远大于可重叠的向量时间——[PL-1.21](elementwise.md) 的「已重叠」判据在此不适用，因为重叠手段自身消耗 UB 预算。
    **判据**：当 `tile_rows × elem_bytes × buffer 数 ≈ UB 上限`（即 tile 已由 UB clamp 决定）时，多缓冲的代价是 tile 减半而非增量内存；此时收益符号由「每波固定开销 × 波次增量」与「未重叠的向量时间」比较决定，后者 <1 µs 时方向为负。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_rope_neox_1d-20261031T184833Z 2026-11-01
- confirmed_by: -
- decided_by: -
- decided_note: -

## VP-2026-0175
- type: P
- title: 整波次（integer-wave）优先的 tiling 判定式：tile 由 `ceil(S/(iters×cores))` 反推，并须校验 `cores×(iters−1) < n_num ≤ cores×iters`——非整波（3.4 波）比更小的 tile（3.95 波）还慢 7–8%
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261031T184833Z 的 perf_opt/opt_log.md §3.3/§4：同一 6-buffer 结构下 8k-d128 的 `tile_rows` 三点扫描——52 行（158 tiles / 3.95 波）6.29 µs、60 行（137 tiles / **3.4 波**）**6.82 µs（−7.4%）**、69 行（119 tiles / **3 波整**）**6.13 µs（+3.5%）**；反向对照 `r3_pipe` tile 26（波次 8）6.78 µs
  - 机制：每波固定开销约 0.82 µs（8k），尾巴波次只有部分核有任务但仍付整波固定项；tile 大小本身不以单调方式决定时延
  - 与既有条目的分界：`attention.md#PL-1.9-blockwidth` 的块宽摊减律只覆盖「每块固定成本 ÷ 元素数」的单调段；本条给出该律失效区（非整波）的判别与修法
- repro: repro-missing（复现条件：同一 kernel 只扫 `tile_rows` 三点，使 `ceil(S/tile)` 分别落在 3.95 / 3.4 / 3.0 波；断言：3.4 波的时延高于 3.95 波与 3.0 波，且 3.0 波最低）
- toolchain_stamp: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add 条目 PL-1.24-integer-wave-tiling（置于 PL-1.9-blockwidth 的 elementwise 侧对应位置，或 elementwise.md 末尾）：

    ---
    id: PL-1.24-integer-wave-tiling
    kind: pattern
    family: [elementwise]
    apis: [T.Kernel, T.serial, T.copy]
    dtype: [bf16, fp16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_make_rope_neox_1d-20261031T184833Z
    toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3 / 2026-11-01
    repro: repro-missing
    ---

    ### 1.24 波次几何优先于 tile 大小：整波（n_num ≤ cores×iters 且 > cores×(iters−1)）

    **判定式**：先按 `tile = ceil(S / (cores × iters))` 反推 tile 上界（iters 为目标波次），再校验 `cores×(iters−1) < n_num = ceil(S / tile) ≤ cores×iters`。落在整波区间内时 tile 取 UB clamp 允许的最大值；落在非整波区间时优先减小 tile 直到贴合整波，而不是放大 tile。
    **实测（8k-d128-bf16，40 AIV，同 6-buffer 结构）**：tile 52→158 tiles/3.95 波 6.29 µs；tile 60→137 tiles/**3.4 波** 6.82 µs（−7.4%，比更小的 tile 还慢）；tile 69→119 tiles/**3 波** 6.13 µs（+3.5%）。
    **机制**：每波固定开销约 0.82 µs（含 launch 摊销与骨架项），尾巴波只部分核有任务但仍付整波固定项；故 tile 大小与时延非单调，[PL-1.9-blockwidth](attention.md) 的摊减律在非整波区失效。
    **实现约束**：tile 上限由 UB 预算 clamp（本例 6 个 fp32 buffer → 每行 2560 B → 69 行），故「贴合整波」常需先削减 buffer 数（fp32 中转 7→6 即为此服务）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_rope_neox_1d-20261031T184833Z 2026-11-01
- confirmed_by: -
- decided_by: -
- decided_note: -

## VP-2026-0176
- type: P
- title: plateau 判定前的强制证据：同 buffer/tile 的最小工作量探针分解固定项（`Task Duration − aiv_time` 取发射项 + `tiny_ub` 取骨架项），而非用流量/带宽反算
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261031T184833Z：上一轮据 `*_active_bw`（读 1.6–2.0 / 写 2.6–2.9 TB/s）判「1d 已贴 roofline」，本轮三个减法探针（`tiny_ub`/`copy`/`mem`）把 6.35 µs 拆成 1.74 + 1.78 + 0.48 + 1.44 + 0.91——带宽项只占 7.6%，前一判断的归因错误（探针成本 <30 分钟，见 opt_log §3.1）
  - 判据产出（本任务实际使用）：① `Task Duration − aiv_time` 给出 launch/ramp 项（5 探针一致 1.73–1.83 µs）；② `tiny_ub`（每核 1 行 10 KB，保留原 grid/iters/descriptors/guard）给出骨架项；③ `copy` 与 `mem` 之差给出「契约强制流量」（本例 cos/sin 2.10 MB = 1.44 µs）；④ `full − mem` 给出算子链增量（0.91 µs）
  - 与相邻条目的分界：queue VP-2026-0147（R，pending）只要求「先测空 kernel launch 地板」；本条要求的是**同形态**骨架探针（空 kernel 地板 1.9–2.85 µs 会低估本例 3.52 µs 约 0.7–1.6 µs，因为漏 7 个描述符与循环 guard）
  - 同类交叉证据：VP-2026-0156「路线无关上界」用的是同一分解的粗粒度版本（地板 + 阶段下界），本条补「契约强制流量」与「算子链增量」两项的独立测量法
  - **第二数据点（2026-11-04，task TileOPs-_make_rope_neox_1d-20261104T065720Z，同 op 谱系，不计确认）**：Round 4 在 r3 上验证 4k/8k 残余 3–5% 空间时，零额外 UB 的轮转预取把 8k `aiv_mte2_wait_ratio` 0.754→0.638、`aiv_scalar_vector_stall_time` 1.214→0.013，但 `aiv_vec_time` 2.466→2.703、新增 `aiv_scalar_mte2_stall_time` 0.821、`aiv_time` 4.351→4.557（更差）；4k `aiv_time` 3.03 vs 最长 pipe `vec` 1.50 的「1.5 µs 缺口」主体是同形态 `tiny_ub` 骨架 1.8–2.0 µs。⇒ 「`aiv_time ≫ max(pipe)`」不能单独当重叠缺口证据，本条的四项分解是判可动量的前置步骤。同 op 谱系按先例**不计 2/2**（待非同族任务独立确认）。
- repro: none（方法学条目；复现条件 = 任一 plateau 判定前检查是否附 (a) `Task Duration − aiv_time` 发射项、(b) 同形态 `tiny_ub` 骨架项、(c) `copy/mem/full` 三项差值，四者之和须与总时延闭合）
- toolchain_stamp: tilelang 0.1.2 dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add 小节「plateau 前的固定项分解（同形态减法探针）」（置于「诊断上下文模板」之前，与 VP-2026-0147 的 launch 地板判据相邻并互链）：

    ### plateau 前的固定项分解（同形态减法探针）

    判定「已贴地板 / 不可再优化」前，须有一次**同 buffer 集、同 tile 推导、同 DMA 形态**的最小工作量探针，把总时延拆成四项并要求闭合：

    1. **发射/ramp 项** = `Task Duration − aiv_time`（同批 5 个探针应互相一致；本例 1.73–1.83 µs）。
    2. **每核骨架项** = 每核仅 1 行（约 10 KB）搬运、但保留原 grid 尺寸 / 循环次数 / 描述符数 / guard 的探针时延（本例 1.78 µs）。
    3. **契约强制流量项** = `mem` 探针（全流量、仅 2 个算术 op）− `copy` 探针（仅主数据 x→y）；本例该项 = cos/sin 表 2.10 MB = 1.44 µs，且其有效带宽（1.45 TB/s）与主数据（8.7 TB/s，L2 驻留）差 6×，按同一带宽估算会低估约 6×。
    4. **算子链增量** = `full` − `mem`（本例 6-op 链 0.91 µs，0.23 µs/iteration）。

    判读：① 四项之和与总时延偏差 >10% ⇒ 探针不同形态，结论作废；② 主数据项占比低（本例 7.6%）时，「贴带宽地板」类判断不成立，残余收益只能来自固定项结构与波次几何；③ 空 kernel 地板（1.9–2.85 µs，见 VP-2026-0142）**不能替代**第 2 项——描述符数与循环 guard 属结构项，本例差值 0.7–1.6 µs。报告须给出四项分表而非单一「固定开销」数字。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_rope_neox_1d-20261031T184833Z 2026-11-01
- confirmed_by: -
- decided_by: -
- decided_note: -

## VP-2026-0194
- type: D
- title: 官方三档 Ratio 均值的两条口径事实：8k 档隐含分母（ideal = latency×Ratio）随 kernel 版本漂移 1.8%，2k/2d 档分母恒定 <0.1%——单档时延 −3.5% 只换来均值 +0.32 点，而同一代码背靠背两次采样的均值离散度 ±0.42 点
- evidence:
  - 五份官方 report 的机械反算（`ideal = latency × Ratio`，见 `examples/TileOPs/reports/tileops/…/report.md` 或 `benchmark.md` 三档行）：8k-d128-bf16 —— r2 时代 LongRope `20261025_234635_085088` 6.30µs×38.04% = **2.3965µs**、Llama31 `20261026_000429_461405` 6.28×38.17% = **2.3971**；r3 时代 LongRope `20261104_094733_368837` 6.08×38.72% = **2.3542**（−1.76%）、Llama31 `20261104_105257_209096` 6.40×36.78% = **2.3539**、`20261104_110558_327166` 6.22×37.89% = **2.3568**。2k 档五份报告 0.5992–0.5994µs（<0.05%）、2d 档 44.6556–44.6626µs（<0.02%）——两档分母跨全部五份报告恒定，只有 8k 档分母随 kernel 版本下移。
  - 后果一（分母漂移 ⇒ 时延与 Ratio 不同向）：Llama31 8k 时延 −0.96%（6.28→6.22）而 Ratio −0.73%（38.17→37.89）、均值 36.00→35.91；LongRope 8k 时延 −3.49%（6.30→6.08）只换 Ratio +1.79%（38.04→38.72）、均值 +0.32 点（36.02→36.34）。
  - 后果二（同代码采样离散）：Llama31 同一 final（sha `c6098298`）背靠背两份官方报告（10:52 与 11:05）均值 **35.49% vs 35.91%**（8k 6.40 vs 6.22），差 0.42 点；同期旧 r2 报告均值 36.00 —— 即「改善/回退」判定整体落在该指标自身噪声带内。
  - 后果三（均值稀释 + 敏感度算术）：三档均值中 2d 档贡献 54.7/108 ≈ 51%，1d 单档改动被摊到 1/3。`dRatio/dt = −Ratio/duration`：2k 档 4.07 点/µs（0.02µs → 0.081 点 row → **0.027 点均值**）、8k 档 6.37 点/µs、2d 档 0.67 点/µs ⇒「2k 档 0.02µs 即可抵消 8k +6%」**不成立**（抵消 8k +6% 需 2k **+0.57µs ≈ +15%**）；本次 Llama31 均值下移 0.10–0.51 点的主项是 8k 分母漂移（−0.28 点 row），2k 的 0.02µs 只贡献 −0.027 点。
  - 机制（代码路径已核实，变因未隔离）：`examples/TileOPs/tileops/benchmark/benchmark_base.py::_parse_msprof_roofline` 取 `visualize_data.bin` 的 GM Read+Write 条目 `ratio`；`ratio = performance / min(AI·BW, computility)`、`AI = counted_flops/bytes`、`performance = counted_flops/duration` ⇒ 本族（memory-bound，`AI·BW < computility` 分支，与 VP-2026-0179 的 compute-bound 分支互不矛盾）化为 `Ratio = bytes/(duration·BW)`，即分母 ideal = bytes/BW 的 bytes 是**该次采样实测的 L2 miss × cacheline**、BW 亦实测——不是声明式 roofline 字节数。故访问形态/波次变化（r2 4 波 158 tile → r3 3 波 119 tile）可改变计数口径的 bytes。r3 在 8k 上 bytes 少 1.8% 的成因未做变因隔离（候选：tile 边界 cacheline 取整 / 块序 L2 命中差异），引用时只按「分母随 kernel 版本漂移」使用，不据此归因到具体 tiling 项。
  - 与相邻条目的分界：VP-2026-0179（merged，口径拆解）给的是「Ratio 怎么读」与「比值型指标对冗余工作中性」；VP-2026-0153（pending）给的是「官方 vs 本地 msprof op 两口径差 ~5%」；本条给的是**同一官方口径跨候选版本的分子/分母可比性**与**三档均值的稀释/敏感度**，三者互补不重复。
- repro: repro-missing（复现条件 = 取同一 shape 的 r2/r3 两份官方 report 目录，读三档 latency/Ratio 并逐档算 `latency×Ratio`；断言只锁「2k/2d 分母跨五份报告 <0.1% 而 8k 分母跨版本 >1%」与方向关系（时延降而 Ratio 降可共存），不锁绝对值）
- toolchain_stamp: 官方 report 侧 tilelang 0.1.1.10+ubuntu.20.4.cann900（git_commit e1d97345）/ CANN 9.0.0 / driver 25.5.0 / torch_npu 2.7.1.post4 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04；本地 A/B 侧 tilelang 0.1.2+e1d97345 dev root build / TILELANG_ASCEND_MODE=Developer
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 常数条目（测量方法学族，置于 CONST-copy-floor-method / CONST-gm-to-ub-bw-dilution 之后）：
    ---
    id: CONST-official-ratio-mean-caliber
    kind: constant
    family: [position_encoding, elementwise]
    apis: []
    dtype: [bf16, fp16]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_make_rope_neox_1d-20261104T005217Z
    toolchain: 官方 report tilelang 0.1.1.10+ubuntu.20.4.cann900 / CANN 9.0.0 / Ascend910B3 / 2026-11-04
    repro: repro-missing
    ---

    ### 官方三档 Ratio 均值：分母随 kernel 版本漂移 1.8%，1d 单档改动被摊到 1/3

    memory-bound 分支（`AI·BW < computility`）下 `Ratio = bytes/(duration·BW)`（`benchmark_base._parse_msprof_roofline` 的 GM Read+Write 条目；compute-bound 分支见 VP-2026-0179 的 `ideal = counted_ops/computility`）：分母 ideal 中的 bytes 是该次采样实测的 L2 miss × cacheline，BW 亦实测，**不是声明式 roofline 字节数**。RoPE 1d 族三档实测（`ideal = latency×Ratio`）：8k-d128-bf16 在 r2 时代 2.3965/2.3971µs（两份独立报告）、r3 时代 2.3539/2.3542/2.3568µs（三份）——**随 kernel 版本 −1.8%**；2k-d128-fp16 五份报告 0.5992–0.5994µs、2d-b1-s8k-h32-d128 44.6556–44.6626µs，跨版本 <0.1%。后果：8k 时延 −3.49% 只换 Ratio +1.79%（38.04→38.72）、均值 +0.32 点；Llama31 8k 时延 −0.96% 而 Ratio −0.73%。三档均值中 2d 档占 51%，1d 单档改动被摊到 1/3；灵敏度 `dRatio/dt = −Ratio/duration`（2k 4.07 / 8k 6.37 / 2d 0.67 点/µs），抵消 8k +6% 需 2k +0.57µs（15%）。同一 final 背靠背两份报告均值差 0.42 点（35.49 vs 35.91）——**该均值的跨版本比较容差须按同代码重复采样的离散度取，不能按 0 处理**。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_rope_neox_1d-20261104T005217Z + TileOPs-_make_rope_neox_1d-20261104T005218Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；两任务同 session 同族（同一 sha、同一 shape 几何），按同一证据链计 1 条。待同族任务补知识域 repro（读两份 report 的三档 latency/Ratio 并算 `latency×Ratio` 的 ≤100 行脚本）后合入 constants.md。

## VP-2026-0195
- type: R
- title: optimize SKILL.md Phase 3 第 3 条补「共享 kernel 采纳同步（adoption sync）」动作清单——字节一致性、本目录重采交错 A/B、gate 4 自检、官方 report ≥2 次采样
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261104T005217Z（RopeLongRopeOp 采纳 r3）：`perf_opt/opt_log.md` §4.3/§8——上一轮采纳把规范位 `perf_records.jsonl` 直接拷回，`workload_inventory.json` 的 `kernel_id` 仍指 baseline 路径、`workload_id` 仍是别 op 的 label ⇒ gate 4 自采纳当轮起 `S4-PERF-RECORDS-COVERAGE` 失败（本任务开工复现 6 处）；本轮补 gate 4 自检后 pass
  - 同 session task TileOPs-_make_rope_neox_1d-20261104T005218Z（RopeLlama31Op）：同类欠账 10 处 failure + opt_log 缺复盘章节；本目录重采 5-run 交错 A/B（20 个 msprof run，约 25 分钟）
  - 采纳同步的「无新候选也需交错 A/B」证据：Llama31 8k r2_contig 单 run 出现 6.23 / 6.47 双态（±3.8%），5-run 交错中位 6.32 才稳定；若沿用单 run 记录，8k 的 +3.3% 会被噪声吃掉（见 VP-2026-0001 已合入的 BP_run_state_bimodality，本条只扩适用范围到「非调优轮次」）
  - 官方 report 采样离散：同一 final 背靠背两份报告均值 35.49 / 35.91（VP-2026-0194），单次 report 不足以支撑「改善/回退」结论
- repro: none（流程规则；复现条件 = 任一 adoption sync 类任务（用户交办「采纳已验证 winner、不开新调优轮」）的动作清单自查）
- toolchain_stamp: tilelang 0.1.2+e1d97345 dev root build + CANN 9.0.0 / Ascend910B3 / 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 3 条末尾追加采纳同步段）
    定位锚: "供 wrapper 切换块成对引用。"
    new 文本（追加于锚句之后）：
    **共享 kernel 采纳同步（adoption sync：目标目录只做「逐字节采纳 + 本目录复测 + 记录刷新」，不开新调优轮次）**：五步动作缺一即不得报告 `success`——① `cmp`/sha256 核对源文件与目标 `perf_opt/{op}.py` 逐字节一致（工厂签名与 block_size 语义 drop-in）；② 旧版归档到 `perf_opt/history/` 并在 opt_log 记 sha；③ **本目录内**重采上一版与新版各 ≥5 run 交错 A/B（`msprof op` Task Duration，`--launch-count=20`、单 run 取中位、跨 run 取中位）——跨目录借用 `duration_us` 不构成可对账记录（`artifact_path` 指本目录而数值来自别目录）；④ `workload_inventory.json` 与 `perf_records.jsonl` 的 `kernel_id` / `workload_id` 同源重写后跑 `statectl gate 4` 自检（采纳轮遗留的 gate 失败不会自己消失）；⑤ 结果落官方 report 时取 ≥2 次采样，比较容差取「同一 final 重复采样的均值离散」（RoPE 族实测 0.42 点），<5% 的时延差异仍按 T-2 交错 A/B 裁决、不按官方 Ratio 均值裁决。
    动机: 两任务的 gate 失败均源于「采纳轮只跑 L0/L1 未自检 gate 4」，各耗一轮清账；采纳同步类动作被误当成「复制文件」而非「需测量的交付」。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_make_rope_neox_1d-20261104T005217Z + TileOPs-_make_rope_neox_1d-20261104T005218Z 2026-11-04
- decided_by: -
- decided_note: -

## VP-2026-0196
- type: R
- title: gate_lint 的 gate 4 对账补「inventory ↔ records 的 kernel_id/workload_id 同源」显式预检（现状仅由 coverage 间接暴露，且报错不给差集）
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261104T005217Z / …005218Z：两目录开工时 gate 4 分别为 6 / 10 处 `S4-PERF-RECORDS-COVERAGE`——`workload_inventory.json` 的 `kernel_id` 指 baseline 根路径、`perf_records.jsonl` 用 `perf_opt/` 路径且混入另一 kernel（2d）行、`workload_id` 仍是别 op 的 label；诊断靠逐条读 failure 文本反推（"不是 inventory 中的 tune workload：{key}"）
  - 代码路径：`.agents/tools/gate_lint.py` L1016–1030 的 `for key, kind in expected.items()` 只按 `(kernel_id, workload_id)` 精确匹配；不匹配时逐行报 key，不给出「inventory 有而 records 无」的集合差，也不单独提示「两侧路径前缀不同」
  - 相邻提案：VP-2026-0114（merged 提案，S4 产物格式契约显式化）与 VP-2026-0036 覆盖 Final 表/artifact_path 形态；本条只补 inventory 与 records 两文件之间的同源约束，不重复
- repro: none（工具规则；复现条件 = 构造 inventory 用 baseline 路径、records 用 perf_opt/ 路径的最小样例，断言新 rule 直接失败且报错含差集）
- toolchain_stamp: 与 gate_lint.py 同源（仓库 HEAD e1d97345）/ 2026-11-04
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: add（在 records 逐行解析循环之后、expected 对账循环之前插入预检；不改既有 failure 语义）
    定位锚:
        for key, kind in expected.items():
            if kind != "tune":
                continue
            phases = measurements.get(key, {})
    new 文本（在锚点之前插入）：
        inv_kernel_ids = {k[0] for k in expected}
        rec_kernel_ids = {k[0] for k in measurements}
        if inv_kernel_ids and rec_kernel_ids and inv_kernel_ids != rec_kernel_ids:
            only_inv = sorted(inv_kernel_ids - rec_kernel_ids)
            only_rec = sorted(rec_kernel_ids - inv_kernel_ids)
            fail(
                "S4-INVENTORY-KERNEL-ID-DRIFT",
                records_path,
                "workload_inventory.json 与 perf_records.jsonl 的 kernel_id 不同源："
                f"仅 inventory 有 {only_inv}；仅 records 有 {only_rec}",
            )
    动机: 采纳同步/共享 kernel 场景下两侧路径前缀漂移（baseline 根路径 vs perf_opt/ 路径）是 gate 4 最常见的清账来源，现状只有逐 key 的 coverage 报错，缺「一眼定位差集」的入口。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_make_rope_neox_1d-20261104T005217Z + TileOPs-_make_rope_neox_1d-20261104T005218Z 2026-11-04
- decided_by: -
- decided_note: -

## VP-2026-0197
- type: R
- title: `perf_records.jsonl` append-only 纪律在「final 候选换代」时补例外条款——允许上一代记录整体移入 `perf_opt/history/`，禁止选择性删行
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261104T005217Z / …005218Z：采纳 r3 后把 r2 时代 6 行（含混入的 2d 行）整体归档到 `perf_opt/history/perf_records_r2_all_kernels.jsonl`，新建 4 行记录集（2 baseline `r2_contig` + 2 final `r3_adopt`）；`perf_records_all.audit.jsonl`（r2 时代 9 行审计快照）原样保留但不参与对账
  - 规则冲突：SKILL.md L14/L80/L97/L180 四处要求 `perf_records.jsonl` append-only「禁止改写/删除既有行」，而 gate_lint L1043–1066 要求「每个 tune workload 恰有一条 final」「同一 kernel 的 final 来自单一候选」——采纳新 winner 后旧 final 行若保留即必然违反后者，选择性删行又违反前者；两任务的实际做法（整体归档 + 新建自洽记录集）是唯一同时满足两者的形态，但当前无成文条款
  - 三份可对照记录集同目录共存且 gate 4 pass（opt_log §7 分 kernel 拆分说明），说明「归档后不参与对账」的口径可机械判定
- repro: none（流程规则；复现条件 = 同一 kernel 换代 final 后跑 `statectl gate 4`，对照「保留旧 final 行 / 选择性删行 / 整体归档后重建」三种做法的 gate 结果）
- toolchain_stamp: 与 statectl/gate_lint 同源（仓库 HEAD e1d97345）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 2 第 7 步的 append-only 句后追加例外；同一 clause 同步到「交付物」段的 `perf_records.jsonl` 描述句）
    定位锚 1: "（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）"
    定位锚 2: "`perf_opt/perf_records.jsonl`：每次 kernel–workload 测量追加一行（baseline 记 round 0），append-only，字段契约唯一出处"
    new 文本（锚 1 之后追加；锚 2 句子保持原样、其后追加同一句）：**唯一例外 = final 候选换代（采纳新 winner / 共享 kernel 采纳同步）**：上一代记录集**整份**移入 `perf_opt/history/perf_records_<代次>_all_kernels.jsonl`（原样保留、opt_log 记归档路径与 sha），新记录集自洽（每个 tune workload 恰一条 final）；**禁止**在新旧记录集之间选择性删行或改写 duration/artifact 字段，`history/` 下的归档集不参与 gate 对账。
    动机: 两任务各用一轮清账才把「append-only」与「final 恰一条」的冲突解开；不成文时每次采纳都要重新推导归档边界。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_make_rope_neox_1d-20261104T005217Z + TileOPs-_make_rope_neox_1d-20261104T005218Z 2026-11-04
- decided_by: -
- decided_note: -

## VP-2026-0198
- type: D
- title: 跨 dtype `T.copy` 融合「split + up-cast」在 npuir Developer 模式是反向优化——少 2 条 vector 指令，但 8k −13.7% / 4k −4.3%，`vec_time` 与 `mte2_wait` 同时上升
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261104T065720Z 的 `perf_opt/opt_log.md` §3.3 + `perf_opt/perf_records.jsonl` round 4（candidate `r4_fuseup`，artifact `_make_rope_neox_1d_r4_fuseup.py`，sha256 `2b0aeb2f…`，L0 9/9 PASS、`max_diff=0`；raw profile `perf_opt/profiles/candidate/round4/r4_fuseup_a7ffb445/`）：8k-d128-bf16 6.41→**7.29 µs（−13.7%）**、4k-d128-bf16 4.89→5.10（−4.3%）；同期 `aiv_time` 4.351→5.645、`aiv_vec_time` 2.466→2.667、`aiv_mte2_wait_ratio` 0.754→0.850、`aiv_vec_total_cflt_ratio` 0.128→0.199
  - 形态：按 `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条，Developer 模式跨 dtype `T.copy` 由编译器插入 `hivm::VCastOp(RoundMode=RINT)`，与显式 `T.vcast(..., "rint")` 等价；本轮把 `T.copy(split)` + `T.vcast(up)` 两条指令合并为一条跨 dtype `T.copy`（理论上每 tile 少 2 条 vector 指令）。实测指令数减少反而 `vec_time` 上升、`mte2_wait` 上升——跨 dtype UB→UB 单指令开销高于「同 dtype copy + vcast」两条指令之和（疑为 VCast 通用路径 + 额外 UB 端口占用，未隔离变因）
  - 与既有条目的关系：TRAP-C12-copy-dtype-cast（verified）已从文档侧记录「跨 dtype copy 不减少 vector pass、勿用作 cast 融合」；本条补**实测代价与方向**（反向优化，−13.7% 量级）与 Developer 模式 UB→UB 子形态。delta 采用 update 而非新开 trap，避免同主题双条目
  - 边界：该结论绑定「UB→UB 跨 dtype copy」；GM→UB 跨 dtype copy 的降级路径未在本轮测量，不外推
- repro: repro-missing（任务内 `perf_opt/_make_rope_neox_1d_r4_fuseup.py` 为整 kernel 候选、26 KB 且无条目 ID/版本戳头，ED-C 机械转正不成立；复现条件 = 同一 kernel 仅替换「同 dtype `T.copy` + `T.vcast`」为跨 dtype `T.copy`（split+up-cast 融合），`msprof op --launch-count=20 --warm-up=5` 取 Task Duration 中位数；断言只锁方向与量级——跨 dtype 变体慢 >10%，且 `aiv_vec_time` 与 `aiv_mte2_wait_ratio` 均上升）
- toolchain_stamp: tilelang dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ torch_npu 2.7.1 / 2026-11-04；TILELANG_ASCEND_MODE=Developer
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update（TRAP-C12-copy-dtype-cast 正文末尾追加实测段，不改旧结论）：
    定位锚: "跨 dtype copy 不减少 vector pass 且舍入不可控——勿用作 cast 融合（lerp_tensor 2026-09-07 文档证伪）。"
    new 文本（追加于锚句之后）：
    **实测代价（2026-11-04，task TileOPs-_make_rope_neox_1d-20261104T065720Z，Developer 模式 UB→UB）**：把「同 dtype `T.copy`（split）+ `T.vcast`（up-cast）」合并为一条跨 dtype `T.copy`（每 tile 少 2 条 vector 指令，L0 9/9 PASS）后，1d RoPE 8k-d128-bf16 6.41→**7.29 µs（−13.7%）**、4k 4.89→5.10（−4.3%），`aiv_vec_time` 2.466→2.667、`aiv_mte2_wait_ratio` 0.754→0.850、`aiv_vec_total_cflt_ratio` 0.128→0.199（msprof op，`--launch-count=20 --warm-up=5`，同 session 5-run 交错 A/B 用于主判候选）。⇒ 本工具链上「融合 cast 拷贝」是反向优化，`T.copy.md` §2.3 第 4 条的等价性只说明语义等价，不说明代价等价。适用边界：UB→UB；GM→UB 跨 dtype 路径未测，不外推。复现条件见 queue 本条 `repro` 字段。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_rope_neox_1d-20261104T065720Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；待同族任务补知识域 repro 后随 TRAP-C12 update 合入。与 VP-2026-0174（Pipelined 负收益）同属 1d RoPE 反向优化族但机制不同，不合并。

## VP-2026-0199
- type: D
- title: v-prefix op 的目标必须是整 buffer——`T.vcast` 目标写静态切片 `o_stage[0:rows, 0:half]` 在 codegen `codegen_npuir_dev.cc:175` shape check 失败（不只动态切片）
- evidence:
  - task TileOPs-_make_rope_neox_1d-20261104T065720Z 的 `perf_opt/opt_log.md` §3.4：候选 `r4_vcastslice` 想把 down-cast 直接写到 `o_stage[0:block_rows, 0:half]`（省 2 条 UB→UB merge 拷贝），编译期报错原文：
    `src/target/codegen_npuir_dev.cc:175` / `TVMError: Check failed: (*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i])) is false`
  - 边界判定（与 VP-2026-0182 不矛盾）：VP-2026-0182 记录的是 `T.vadd` 的 **region 实参偏移含 Var** 在 parser evaluator 阶段 `TypeError`（并注明静态切片可通过）；本条记录的是 `T.vcast` 的**目标 buffer 为静态切片**在 **codegen** 阶段 shape check 失败——阶段不同（parser vs codegen）、角色不同（输入 region vs 目标 buffer）、api 不同（vadd vs vcast），两条各自成立，合入时须互链说明「静态切片在 parser 可过、不等于在 codegen 可作为 v-op 目标」
  - 影响：1d RoPE 想省掉 merge 拷贝的方向（down-cast 直写输出切片）在该工具链上结构性不可行，非参数问题；本轮无性能测量（编译期否决）
  - 注：任务 opt_log 所称「R-8」为 optimizer 侧内部规则标签，仓库 `.agents`/`.opencode` 未见该编号的 canonical 条目；本条按实测事实登记，不复用该编号
- repro: repro-missing（任务内无独立三行 diff 文件；复现条件 = 在任一可编译 kernel 中把 `T.vcast(src, dst)` 的 `dst` 写成静态切片 `o_stage[0:rows, 0:half]`（rows/half 为编译期常量），编译即得 `codegen_npuir_dev.cc:175` shape check 失败；对照改为整 buffer 后编译通过。断言只锁「静态切片目标编译失败、整 buffer 通过」这一方向）
- toolchain_stamp: tilelang dev root build + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-11-04；TILELANG_ASCEND_MODE=Developer
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-vcast-target-whole-buffer（置于 TRAP-expert-v-operands 之后，正文与 VP-2026-0182 的 parser 侧规则互链）：

    ---
    id: TRAP-vcast-target-whole-buffer
    kind: trap
    family: [elementwise, position_encoding]
    apis: [T.vcast, T.copy]
    dtype: [bf16, fp16, fp32]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_make_rope_neox_1d-20261104T065720Z
    toolchain: tilelang dev root build + CANN 9.0.0 / Ascend910B3 / 2026-11-04；TILELANG_ASCEND_MODE=Developer
    repro: repro-missing
    ---

    ### v-prefix op 的目标必须是整 buffer（静态切片也不行）

    `T.vcast(src, o_stage[0:block_rows, 0:half])`（rows/half 为编译期常量）在 codegen 阶段报 `src/target/codegen_npuir_dev.cc:175` / `TVMError: Check failed: (*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i])) is false`。⇒ 「省掉 UB→UB merge 拷贝、让 down-cast 直写输出切片」在本工具链结构性不可行，不是参数问题。

    **与 parser 侧规则的边界**：静态切片可以通过 parser（见 queue VP-2026-0182 对 `T.vadd` 的描述），但**通过 parser ≠ 可作为 v-op 目标**——目标 buffer 在 codegen 仍按整 buffer 做 shape check。引用时须区分「输入 region 静态切片 / 动态 Var 偏移」与「目标 buffer 整 buffer / 切片」两组维度。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_rope_neox_1d-20261104T065720Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；与 VP-2026-0182 已做冲突预检——非矛盾（parser vs codegen、输入 region vs 目标 buffer），合入时两条互链。

## VP-2026-0200
- type: D
- title: 910B3 宽块向量 op 成本 ≈0.61 µs/op/AIV（`[32,256]` f32 成块 op）+ tiny `[32,1]` op 不计费——补 `CONST-vector-launch-overhead` 的「910B3 宽块未重测」缺口
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §4：round 2 `r2_op2_minfold` 相对 merged base 在**全块**无条件多跑 5 个 `[32,256]` f32 op → +1175 µs；blocks/AIV = 15360/40 = 384 ⇒ 1175/(384×5) = **0.61 µs/op·per-AIV**（candidate 与 base 同 config (64,256)、Expert 两相位、f32 S 载体）
  - 同任务 §5 round 4 op2（`alphaslim`，去掉每块 1 个 `T.copy` + 1 个 clamp 的 `[32,1]` tiny op）→ **+0.01%**（噪声内）⇒ tiny `[32,1]` op 实测不计费，与成块 op 的固定开销口径分离
  - 与既有条目的关系：`constants.md#CONST-vector-launch-overhead`（910B3 重测版）明写「910B3 上按旧口径的宽块 op 未重测——引用 0.5 µs/op 时须标注 910B2C 宽块口径」，本条正是该缺口的 910B3 补测值（0.61 vs 910B2C 0.5 同量级）。**delta = update 同条目**（不新开 CONST）
  - 边界：数值绑定本结构（bm=64/bn=256、f32 S、910B3 20 AIC/40 AIV、tilelang 0.1.2+cebcec9d + CANN 9.0.0）；「每 op 固定开销」只对成块 tile op 成立，窄块（4096 元素）口径见条目正文 0.0454 µs/op
- repro: repro-missing（任务内无自包含测量脚本；`perf_opt/parse_profile.py` 为 profile 汇总器非测量器。复现条件 = 同一 kernel 的 A/B 两支仅差「全块无条件执行的 `[32,256]` f32 op 数」k 个，msprof op median-of-20 的时延差 ÷ (blocks/AIV × k)；断言只锁量级 0.4–0.8 µs/op 与「tiny `[32,1]` op 数变化不影响时延」）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    update（`CONST-vector-launch-overhead` 正文「与 910B2C 旧口径「~0.5 µs/op」的关系」段之后追加重测值，不改旧结论）：

    定位锚: "910B3 上按旧口径的宽块 op 未重测——**引用 0.5 µs/op 时须标注「910B2C 宽块口径、910B3 未复测」**。"

    new 文本（追加于锚句之后）：
    **910B3 宽块补测（2026-11-04，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z，Expert 两相位 attention，bm=64/bn=256，f32 S 载体）**：成块 `[32,256]` f32 op ≈ **0.61 µs/op·per-AIV**（全块无条件多 5 个 op → +1175 µs，15360 块 / 40 AIV = 384 块/AIV ⇒ 1175/(384×5)），与 910B2C 宽块 0.5 µs/op 同量级；tiny `[32,1]` 级 op **不计费**（每块去 1 copy + 1 clamp → +0.01%，噪声内）。⇒ 壁钟预测式 = `blocks_per_AIV × (Σ 成块 op 数) × 0.6 µs` 与 AIC 标量侧取大者（本任务 final 预测 2.06 ms / 实测 2.00 ms，±3%）。适用边界：只对成块 tile op 计费；窄块口径仍用本条 0.0454 µs/op。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；任务 opt_log §10.3 声称该条目已落盘 constants.md，终态核对库内无 `CONST-vector-op-cost-910B3`（可能为跨 session 工作区丢失）——以本条为唯一登记，待同族任务补知识域 repro 后随 CONST-vector-launch-overhead update 合入。

## VP-2026-0201
- type: D
- title: Expert 两相位 attention 的四管道账——四管道和/墙钟 **1.51–1.54×**（已 ~50% 重叠）、mte2 单管道 65% 墙钟、理想化重叠上界 **1.52×**
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §5 Round 10（round 9 final 的 raw profile，同 session 20 launch）：8b-long fp16 `mte2 0.650 / mte1 0.307 / cube 0.243 / fixpipe 0.315`（比值对墙钟），四者之和 3008.9 µs vs wall 1959.12 µs = **1.536×**；70b-long fp16 `0.648/0.303/0.241/0.312`、和 1.519×；同表 M1c 实测 1.507→1.474 与 1.519→1.486
  - 派生判据：mte2 完全藏进 cube+fixpipe 的理论墙钟 ≈ 3008.9/1.536 → 1291 µs = **1.52×** 上界；即「四管道和/墙钟」把「串行」与「已重叠但某管道独大」区分开
  - 与既有条目的关系：`BP_pipeline_overlap` 的「无收益」记录面向 Developer `T.Pipelined`/auto multi-buffer；本条是 Expert 双 Scope 两相位的 pipe 账，二者不矛盾（结构不同）。`elementwise.md#PL-1.21-overlap-saturation-signal` 的 `mte2+vec > Task` 是 2 管道判据，本条是 Cube 侧 4 管道判据——互链不合并
  - 边界：数值绑定 8b/70b-long fp16（H/B=32/64、D=128、(64,256)、H·B=128 tasks/blocks 4096/15360）、tilelang 0.1.2+cebcec9d + CANN 9.0.0 / 910B3
  - **复用形态（R12，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z）**：把同一账页直接当作 **Ratio 硬目标的可达性上界**——同一 profile 的四管道绝对时间 `mte2 1291 / mte1 609 / cube 483 / fixpipe 625 µs` ⇒ **单管道下界 = max(单管道) = 1291 µs**，对基线 2003.35 µs 即 **1.55×**；40% 目标需 1202 µs（1.67×）⇒ **上界已短于目标**，任何重叠改造都不可能达标（该任务的三条结构路线各自另有实测否决：K-only 共享 +13.9%、双 kernel 拆分 3.93×、V 共享拟合 UB 但输在 band-outer 流水）。⇒ 该账页有两种用法：① 现状诊断（四管道和/墙钟 ≈1.5 是「已重叠但单管道独大」）；② 目标可达性否决（`max(单管道)` 与目标时延直接比对，无需再做形态实验）。注：本条为该 op 谱系后续任务，按「不同任务才计确认」的纪律**不推进 confirmations**
- repro: repro-missing（任务内 `perf_opt/parse_profile.py` 为 profile 汇总器，依赖已生成的 msprof 目录；复现条件 = 对同 config 的 `msprof op --launch-count=20` 画像取 `aic_mte2/mte1/cube/fixpipe` 的 ratio×`aic_time` 求和，再除以 `Task Duration`；断言只锁 1.4–1.7× 量级与「最大单管道 ≥ 0.6×墙钟」）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update（`PL-1.18-ssd-steady-structure-floor` 正文末尾追加跨族复证 bullet，不改旧结论）：

    定位锚: "单 buffer 消 idle 的正路是减少段数本身（数学/结构层），而非更深流水/更早预取。"

    new 文本（追加于锚句之后）：
    - **跨族复证：Expert 两相位 attention 的四管道账（2026-11-04，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z，910B3 / CANN 9.0.0）**：GQA-SW-Varlen final（r9_final_sf32）8b-long fp16 的 Cube 侧比值 `mte2 0.650 / mte1 0.307 / cube 0.243 / fixpipe 0.315`，四者之和 = **1.536 × 墙钟**（70b-long 1.519×；另两个 910B3 采集 1.474–1.507×）⇒ **结构本身已拿到 ~50% 重叠**；mte2 单管道占 65% 墙钟，把它完全藏进 cube+fixpipe 的理想化上界 = 四管道和/墙钟 = **1.52×**。判据：该比值接近 1 是串行、≈1.5 且单管道 ≥0.6 是「已重叠但某管道独大」——后者的可动空间是**单管道本身**（减少搬运/换载体）而非继续加深流水；本条与 `BP_pipeline_overlap`（Developer `T.Pipelined` 无收益）及 `PL-1.21-overlap-saturation-signal`（2 管道 `mte2+vec > Task`）互链，三者适用结构不同、不互相否定。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；不写入 constants.md（该账是本结构画像而非硬件常数，且随 dtype/config 变化）；待同族任务补 repro 后随 PL-1.18 update 合入。

## VP-2026-0202
- type: D
- title: f16 tile + 运行期 f32 标量在 v-prefix op 上编译失败（`codegen_npuir_api.cc:2019`）；`T.Cast("float16", 标量)` 合法但整数精确上限 2048 ⇒ 掩码阈值类 f16 链在 KV-cache 大 offset 域不可用
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §5 Round 5 探针表（`probe_f16_chain.py`，本 session）：`T.vsub(f16 tile, <f32 PrimExpr 标量>, f16 tile)` → `InternalError: Check failed: (src_dtype == binaryOp.Src1().GetBuffer()->dtype) is false: The element types of two operands should be the same for "hivm.hir.vsub"`（`src/target/codegen_npuir_api.cc:2019`）；`T.vmul(f16 tile, <python float>, f16 tile)` 通过
  - `T.Cast("float16", <阈值>)` 形式可编译但破坏契约域：f16 整数精确上限 = 2048，掩码阈值 `K_up = s_lo − n_lo + row0 + offset + up_shift` 在 KV-cache 域可 > 2048（`q_len=1, kv_len=4096 ⇒ offset=4095`）⇒ 掩码边界错 1 列 = 精度 bug（该域由 L1/L2/Boundary 覆盖）
  - 证伪留痕：合法形态已穷举三条（直接 f16 标量 / `T.Cast` 标量 / 哨兵有限化）——直接形态编译器拒绝、`T.Cast` 形态与契约域冲突；剩余形态（掩码链留 f32 + 每次惩罚多一次 vcast）按「op 数 × 字节」为净中性或净亏
  - 与既有条目关系：`TRAP-expert-v-operands` 记录 vcmp 拒绝 `tir.Cast`（Expert 模式、比较操作数）；本条是 v-prefix 二元 op 的**运行期标量 dtype 必须与 tile 一致**（不同 api、不同失败阶段 codegen），两条互链不合并
- repro: repro-missing（任务内探针 `perf_opt/probe_f16_chain.py` 随上一 session 工作区丢失，当前磁盘不存在；复现条件 = 任一含 `T.vsub(f16 tile, <f32 运行期标量>, f16 tile)` 的 kernel 单编即得上述 InternalError，把标量改为 f16 常量后通过。opt_log §10.3 声称 `repro/TRAP-vector-f16-scalar.py` 已落盘，库内不存在）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3 / 2026-11-04（首次探针 2026-10-17 同工具链）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 `TRAP-vector-f16-scalar-dtype`（置于 `TRAP-expert-v-operands` 之后，正文与之势互链）：

    ---
    id: TRAP-vector-f16-scalar-dtype
    kind: trap
    family: [attention, expert, elementwise]
    apis: [T.vsub, T.vmul, T.vadd, T.Cast]
    dtype: [fp16, fp32]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z
    toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
    repro: repro-missing
    ---

    ### f16 tile 与运行期 f32 标量不能混用于 v-prefix 二元 op（`codegen_npuir_api.cc:2019`）

    `T.vsub(f16 tile, <f32 PrimExpr 标量>, f16 tile)` 编译失败：`InternalError: Check failed: (src_dtype == binaryOp.Src1().GetBuffer()->dtype) is false: The element types of two operands should be the same for "hivm.hir.vsub"`（`src/target/codegen_npuir_api.cc:2019`）。同一形式里 `T.vmul(f16 tile, <python float>, f16 tile)` 通过——字面量被隐式转 f16，**运行期 f32 标量**不会。

    `T.Cast("float16", <阈值>)` 可编译，但 f16 的整数精确上限 = 2048；掩码阈值在 KV-cache 域可 > 2048（`q_len=1, kv_len=4096 ⇒ offset=4095`），会错 1 列边界（精度 bug，门禁覆盖域）。⇒ 需要「大阈值 + f16 tile」的掩码链在本工具链不可用；掩码链留 f32，或阈值保持可证明 <2048 的域。

    边界：与 `TRAP-expert-v-operands`（vcmp 拒绝 `tir.Cast`，比较操作数）分属不同 api / 失败阶段；两条均属「v 算子标量操作数 dtype 约束」族，合入时互链。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1（opt_log §10.3 声称已落盘 `TRAP-vector-f16-scalar-dtype` 与同名 repro，终态核对库内均不存在——以本条为唯一登记）；待同族任务补知识域 repro 后合入。

## VP-2026-0203
- type: P
- title: 可证明 no-op 的掩码子链必须加**运行期守卫**——判据是「无条件执行的 tile 级 op 数」，不是「哪些块需要掩码」；带守卫 −36.2%、折叠成更少的无条件 op +0.1%
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §5 Round 2：带状掩码的 per-row 列 OOB 惩罚链（6 个 `vsub/vmul/vmax/vmin/vmul/vadd`）在 `tn_real == bn_eff` 时恒为 0（全部 tune workload 的 kv_len 512/2048 可被 bn_eff=256 整除），但基线写成 always-on。加 `if tn_real < bn_eff:` 守卫 → 8b-long fp16 **2065.79 µs（−36.2% vs 3237.05）**；同轮把 6 个「仅尾块需要」的 op 折成 5 个「每块都跑」的 op（数学恒等）→ **3240.16 µs（+0.1%，噪声内）**
  - 机制归因：`CONST-vector-launch-overhead` 的成块 op 固定开销 —— 省的是「每块无条件执行的 tile op 数 × 块数/AIV × 0.61 µs」（6 × 384 × 0.61 ≈ 1.4 ms），与「掩码在哪些块非恒等」无关
  - 最小代码形态（ED-B delta 骨架，before → after）：
    ```python
    # before：每个块都跑 6 个 [32,256] f32 op（tn_real 恒 == bn_eff 时全是 no-op）
    T.vsub(...); T.vmul(...); T.vmax(...); T.vmin(...); T.vmul(...); T.vadd(...)
    # after：运行时守卫，非尾块整链跳过
    if tn_real < bn_eff:
        T.vsub(...); T.vmul(...); T.vmax(...); T.vmin(...); T.vmul(...); T.vadd(...)
    ```
  - 边界：守卫是 **trace 期 Python if（工厂/块级常量条件）或 kernels 内静态可判分支**，不是每个元素都判的 runtime 谓词；尾部块仍走原链（正确性不变）。与 `attention.md#PL-1.11-causal-mask-scalartrap` 的「算术惩罚掩码」互补（那条讲怎么算，本条讲**何时可以不跑**）
- repro: repro-missing（任务内分支文件 `perf_opt/_gqa_sw_fwd_varlen_kernel_round2_op1_oobguard.py` 为整 kernel 候选、无条目 ID/版本戳头，ED-C 机械转正不成立；复现条件 = 同一 kernel 的 A/B 两支仅差「always-on 掩码链 vs `if tn_real < bn_eff` 守卫」，msprof op median-of-20；断言只锁量级：守卫支快 >20%，折叠支差异 <3%）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04（首测 2026-10-17 同工具链）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update（`PL-1.11-causal-mask-scalartrap` 正文末尾追加 bullet；不改旧结论）：

    定位锚: "复现：`python perf_opt/bench.py perf_opt/_gqa_prefill_fwd_kernel.py --case 8b-long --use-default-config --msprof-loop 25` 外部 msprof op 同 opt_log 命令模板。"

    new 文本（追加于锚句之后）：
    - **可证明 no-op 的掩码子链加运行期守卫（2026-11-04，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z，910B3）**：带状掩码的 per-row 列 OOB 惩罚链（6 个 `[32,256]` f32 op）在 `tn_real == bn_eff` 时恒为 0（kv_len 可被 bn_eff 整除的域），基线却是 always-on。加 `if tn_real < bn_eff:` 后 8b-long fp16 3237.05→**2065.79 µs（−36.2%）**；同轮把 6 个「仅尾块需要」的 op 折成 5 个「每块都跑」的 op（数学恒等）→ **+0.1%（噪声内）**。判据：收益 ∝ **每块无条件执行的 tile 级 op 数**，与「哪些块需要掩码」无关；`[32,1]` tiny op 不计费（同任务 round 4 实证）。机制归因 `CONST-vector-launch-overhead`。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1（ED-B delta 骨架已内嵌本条，效应在完整 kernel 规模显现）；待第 2 次独立证据（不同任务）后随 PL-1.11 update 合入。

## VP-2026-0204
- type: P
- title: 深度 2 任务流水的 **varlen 形态**——advance 内重放 ragged 解码（ghost-safe `real_m==0 ⇒ lc=0`）+ flag 预算判据改用**解析上界** `2·L_max_bound ≤ 15`（含 +2 余量），而非实测 loop_count
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §5 Round 1（pipe2 全域 −13.9%~−24.7%，`l1_q` 独立 + TASKDONE 删除）与 round 4（bn=128：`L_max = ceil((64+1024)/128)+2 = 11 ⇒ 2·L_max = 22 > 15` ⇒ pipe2 自动回落单槽 + 块数近翻倍 ⇒ **+26.2%**）
  - 与定长母本 `PL-1.12` 的差异（两条）：① varlen 的 per-task ragged 解码（task→(b, q0, kv0)）在 advance 里**重放一次**，且须 ghost-safe（`real_m == 0 ⇒ lc = 0`，否则幽灵任务消费越界任务）；② flag 预算是**变量**——`L_max` 依赖每 task 的 kv_len，须用解析上界（`ceil((D+wl)/bn)+2`，+2 为余量）而非实测 loop_count，否则不同 shape 下静默超 15
  - 边界：判据适用于「per-task flag id = l + slot·L_max」的两相位/Cube 双 Scope 结构；`bn` 选择须同时满足 UB 与 flag 两个上界（本任务 (64,256) 是唯一交点）
- repro: repro-missing（复现条件 = 任一 varlen persistent 两相位 kernel 实现 pipe2：flag id = l + slot·L_max 且 2·L_max ≤ 15；断言 ① ghost task 不消费；② 任一 manifest shape 的 `2·ceil((D+wl)/bn)+4 ≤ 15`；③ 回落单槽时性能回到 varlen 单槽基线量级）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update（`PL-1.12-task-pipeline-depth2` 正文末尾追加 varlen 形态 bullet）：

    定位锚: "守卫 delta 骨架 + 镜像核算公式（含 pre-fix 失败复现断言）：`repro/PL-1.12-bn-clamp-bm-guard.py`；编译失败全档案 `perf_opt/logs/r9_precision_fix/repro_full_fwd_bf16_r8h.log`（provenance，允许失效）。"

    new 文本（追加于锚句之后）：
    - **varlen 形态（2026-11-04，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z，910B3）**：① ragged 解码（task→(b, q0, kv0)）须在 advance 中**重放一次**并 ghost-safe（`real_m == 0 ⇒ lc = 0`），否则幽灵任务推进到越界任务；② flag 预算判据由「实测 loop_count / nk_total」改为**解析上界** `2·L_max_bound ≤ 15`，`L_max_bound = ceil((D + wl)/bn) + 2`（+2 为余量）。本案 `(64,256)` 通过；`bn=128 ⇒ L_max=11 ⇒ 22 > 15`，pipe2 自动回落单槽，叠加块数近翻倍实测 **+26.2%** ⇒ `(bn)` 的选择是 UB 与 flag 两上界的交点，不是单调扫描。pipe2 本体收益（定长母本形态）保留在上一 bullet。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；与 `PL-1.12` 定长母本同族但形态不同（本任务为首个 varlen 落盘），按「不同任务」规则需另一次独立 varlen 证据才计 2/2。

## VP-2026-0205
- type: P
- title: 「重叠候选」的负向判据——若唯一可编译形态使 **mte2 忙比下降且四管道和/墙钟也下降、墙钟反升**，则缺口在 **L1 端口读写互拖**（MTE2 写 vs MTE1 操作数读），加缓冲/加深流水无意义
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §5 Round 10 M1c 表（两个 long 同向）：把与跨引擎 flag 无关的 V 装载（GM→L1 64 KB）提到 `sync_block_wait` 之前 → 8b-long 2019.76→**2047.50（+1.4%）**、70b-long 4000.44→**4058.26（+1.4%）**；同批 `mte2 忙比 0.634→0.618`、`四管道和/墙钟 1.507→1.474`、跨引擎等待 1191.7→1126.2 µs 而总时延不降
  - 机制：被填的只是 MTE2 空转，而 L1 端口上 MTE2 写与 MTE1 操作数读（mte1 忙比 0.300）互拖；与 `attention.md#PL-1.18`（ssd L1 双缓冲端口互拖，910B2C +11.9~+18.8%、二次否决 +2.9~+6.3%）同机制，本条是**跨到 910B3 / attention 族**的复证并给出可观测的组合判据
  - 与 `PL-1.21-overlap-saturation-signal` 的分界：PL-1.21 用 `mte2+vec > Task` 判「已重叠」；本条是「重叠手段已见效（忙比与和/墙钟同降）但墙钟反升」⇒ 不是缺重叠，而是换来的重叠被端口竞争吃掉——判据多一个「墙钟方向」维度
- repro: repro-missing（任务内 `perf_opt/_gqa_sw_fwd_varlen_kernel_round10_m1c_vfirst.py` 为整 kernel 候选、无条目 ID/版本戳头；复现条件 = 同一 Expert 双 Scope kernel 只把一条与 flag 无关的 GM→L1 装载提到 `sync_block_wait` 前，msprof op 5-run 交错 A/B；断言：`aic_mte2_ratio` 下降、四管道和/墙钟下降、同时 Task Duration 上升 >1%）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    update（`PL-1.21-overlap-saturation-signal` 正文末尾追加跨族负向判据 bullet）：

    定位锚: 该条目正文最后一段（以「已重叠判据」为主题的 `mte2+vec > Task` 段末）。

    new 文本（追加）：
    - **负向判据：重叠已生效但被 L1 端口竞争吃掉（2026-11-04，task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z，910B3 / Expert 两相位 attention）**：把与跨引擎 flag 无关的 GM→L1 装载提到 `sync_block_wait` 之前后，`aic_mte2_ratio` 0.634→0.618、四管道和/墙钟 1.507→1.474（都朝「更空/更重叠」方向），但 Task Duration 2019.76→2047.50 µs（**+1.4%**，70b-long 同向 +1.4%）⇒ 被填的只是 MTE2 空转，MTE2 写与 MTE1 操作数读（忙比 0.300）在 L1 端口互拖。**三个信号同时出现时，加缓冲/加深流水无意义**，可动项在搬运算量或载体（同机制见 attention.md `PL-1.18` 的 910B2C 双缓冲否决）。与 `mte2+vec > Task`（2 管道「已重叠」）互补：本条多一个「墙钟方向」维度。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；与 PL-1.18 同机制，本条提供跨族（910B3/attention）第二现，合入时两条互链。

## VP-2026-0206
- type: R
- title: Stage 4 工件丢失 / `final` sha 漂移的迁移纪律——旧记录整份归档 + 同 session 全量重测 baseline/final 生成自洽集 + opt_log 记录迁移段，验收 = `statectl gate 4` 通过
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z：上一 session 的 `perf_records.jsonl`（round 0–9，41 行）`phase=final` 的 `artifact_sha256 = 1216193a…` 与磁盘现存 `perf_opt/{op}.py`（`d98cc867…`）不一致，且其引用的 `profiles/`、`logs/`、`bench.py`、`workload_inventory.json` 已随上一 session 丢失 ⇒ 无法对账（opt_log §8「记录迁移」段 + `perf_opt/history/README.md`）
  - 实际处理（本轮已过 gate 4）：① 旧集整份归档到 `perf_opt/history/perf_records_round0-9_final_sha121693a.jsonl`（append-only、不改不删）+ `history/README.md` 写原因；② 同 session 用现存文件对**全部 8 个 tune workload** 重测 baseline + final（`logs/r10_m1/run_all.sh`，16 次 msprof）；③ 新 `perf_records.jsonl` 自洽（18 行 = 8 baseline + 2 candidate + 8 final，单一 `candidate_id = r9_final_sf32`，sha 与磁盘一致）④ opt_log §8 记录迁移段 + Final 表换新值
  - 与 VP-2026-0197（append-only 在 final 换代时的例外）的关系：0197 管「主动换代」的正常路径；本条管「工件已丢失 / sha 已漂移」的**故障恢复**路径（先归档、再全量重测、再自洽化），触发条件与动作不同 ⇒ 不合并；两条 target 同一文件，建议同批审批
  - 验收实测：`statectl gate 4` 通过（本轮），且 `.stage_state.json.artifact_hashes` 与新 final 行一致
- repro: none（流程规则；复现条件 = 任一 optimize 任务在 final sha 与磁盘不一致且过程目录缺失时执行上述 ①–④，跑 `statectl gate 4`）
- toolchain_stamp: 与 statectl/gate_lint 同源（仓库 HEAD 5ce7e59a）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 4 交付整理段，final 对账句之后追加故障恢复条款）
    定位锚: "最终表的全部时延必须来自同一最终候选。gate 4 依据 `workload_inventory.json` 与 `perf_records.jsonl` 逐项对账。"
    new 文本（锚句之后追加）: **故障恢复条款（final sha 漂移 / 过程目录丢失）**：若 `perf_records.jsonl` 的 `phase=final` 行 `artifact_sha256` 与磁盘 `perf_opt/{op}.py` 不一致，且其引用的 `profiles/`、`logs/`、`workload_inventory.json` 已不可得 ⇒ ① 旧记录集**整份**移入 `perf_opt/history/`（append-only，写 `history/README.md` 记录 sha 与原因，禁止选择性删行）；② 同 session 用**现存**文件对全部 tune workload 重测 baseline + final（同口径 `--launch-count=20 --warm-up=5`，median）；③ 新记录集自洽（单一 `candidate_id`、sha 与磁盘文件一致）；④ `opt_log.md` 写「记录迁移」段并更新 Final 表；⑤ 验收 = `statectl gate 4` 通过 + `.stage_state.json.artifact_hashes` 与新 final 行一致。**不得**以旧 sha 的 final 行继续对账，也不得直接删除旧行。
    动机: 本轮需一轮完整重测（16 次 msprof）+ 归档才把不可对账状态解开；不成文时下一次同样漂移会再次卡 gate 4。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: Tier 2 待人工审批；与 VP-2026-0197 同 target 文件、不同触发路径（换代 vs 故障恢复），建议同批审批。


## VP-2026-0207
- type: R
- title: Stage 4 编译器可行性探针补「快速失败」通道——`--compile-probe`（重试上限 1 + 超时 10 min），避免 SIGSEGV 重试循环 / 不收敛形态烧 15–90 min
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §5 Round 10 + `logs/r10_m1/PROBE_LADDER.md`：kernel 自带 `_COMPILE_RETRIES=40` × 外层 24 次的 SIGSEGV 重试，一次失败形态实测 12–15 min（M1a 12 min @100% CPU + 8 次 SIGSEGV；M1b/M1d >15 min 不收敛），而基线同 session 26–42 s。本任务用 `timeout 400~900` + `grep -c "err code"` 可在 2 min 内判定「SIGSEGV 循环」，但「不收敛（100% CPU 无输出）」只能靠满超时发现 —— 4 个分支 + 7 步探针阶梯共消耗约 2 h 编译等待
  - 需求：Stage 4 的可行性探针（不要求性能，只要求「能否编译」）应有独立入口：重试上限 1、单次超时 10 min，超时即判 `compile-inconclusive` 并入账，不再进入性能实验
  - 与既有条目的关系：`VP-2026-0171`（Phase 0 检索同算子案卷）与 `VP-2026-0176`（plateau 前减法探针）都是「先做便宜实验」族；本条是编译侧的快速失败门（不改变任何优化规则，只改探针执行方式）
- repro: none（流程规则；复现条件 = 任一自带大重试循环的 kernel 在 Stage 4 跑失败形态，用 `--compile-probe` 与默认模式各一次，比较判「不可编译」所需时间——目标 ≤10 min）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 4 的 `blocked` 定义句后追加编译快速失败条款）
    定位锚: "- blocked：剩余候选优化点均因精度失败、编译失败、profile invalid 或实现约束无法继续。"
    new 文本（锚句之后追加）: **编译可行性探针的快速失败纪律**：涉及新缓冲/新循环体形态的候选，先跑「只编不跑」探针（重试上限 1、单次超时 10 min）；超时判 `compile-inconclusive` 并记录，不进入性能实验。理由：kernel 自带的大重试循环（如 40×24 次 SIGSEGV 重试）使单次失败形态消耗 12–15 min，而 4 分支 / 7 探针的阶梯可累计 >2 h；「不收敛」形态只能靠满超时发现，快速失败入口是唯一低成本判别。探针结论（可编译/不可编译/不确定）须写入 opt_log，且 `compile-inconclusive` 不计入「无收益」候选数。
    动机: 本任务 round 10 的 4 个分支全部耗在编译门而非性能门；缺该通道时「能否编译」的判定成本与一次完整调优轮相当。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: Tier 2 待人工审批。

## VP-2026-0208
- type: R
- title: Stage 4 runner 卫生四则——`setsid` 后台任务加 pidfile 互斥（`perf_opt/.runner.lock`）、清理只用 `kill <pid>` / `pkill -x <exact-name>`、`no OpBasicInfo rows` 的 profile 目录改名 `*.invalid`、超时清理按进程组杀（`start_new_session` + `killpg`）以免留孤儿 `bishengir-compile` 占设备
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z：`opt_log.md` §2 记录说明——round 0 的 `8b-short fp16/bf16` 各有两行记录，原因是「`setsid nohup …&` 与 shell 超时叠加，同一 `--phase baseline` 起了两个实例」，两次测量差 ≤0.25%（append-only 下两行均保留）；同任务 Round 10 流程问题 1：两次 `pkill -f "bench.py launch"` 命中当前 shell 自己的命令行，导致正在执行的命令块被静默杀死（无输出 + 后续文件不生成）；流程问题 3：vselect 分支的 msprof 目录被创建但无记录，`profiles/candidate/round2/` 留下空壳目录，对账时需靠「无 OpBasicInfo 行」判空
  - 〔2026-11-07 第二证，不同任务〕task TileOPs-_gqa_sw_fwd_kernel-20261107T161617Z 的 opt_log §20.4：未改 kernel 的新 shape 探针 `probe-mid` 使 `bishengir-compile` 重试 >900 s，超时 kill 后仍留 4 个孤儿进程（`ps -eo pid,etime,cmd`），已手工清理并确认设备空闲；task TileOPs-_gqa_sw_fwd_varlen_kernel-20261107T173117Z 的 `perf_opt/logs/round13_report/` 同形态（`subprocess.run(timeout=)` 只杀直接子进程 → 孤儿占设备 → 后续 case 串行化/卡 26 min），修复形态 = `Popen(start_new_session=True)` + `os.killpg`
  - 四则同属 runner 生命周期卫生（并发/清理/产物标记/子进程树），与 VP-2026-0118（长采集 setsid 后台形态）互补：0118 定「怎么起」，本条定「怎么防重、怎么杀、怎么标空、怎么收树」
- repro: none（流程规则；复现条件 = 同一 phase 并发起两个 runner 观察是否被 pidfile 拒绝；用 `pkill -f "<当前命令行子串>"` 复现自杀；对无 OpBasicInfo 的 profile 目录检查是否被改名；对任一超时调用检查 `ps` 是否残留 `bishengir-compile` 孤儿）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04；第二证 tilelang 0.1.2+9cfae94c + msprof 25.5.0 / Ascend910B3 / 2026-11-07～08
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（§3 输出目录段落之后追加 runner 卫生四则）
    定位锚: "新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。"
    new 文本（锚句之后追加）: **runner 生命周期卫生（2026-11-04 GQA-SW-Varlen round 10 实证；第四则 2026-11-07 补）**：① **并发互斥**——`setsid` 后台 runner 写 `perf_opt/.runner.lock`（含 pid + phase），`phase in (baseline, final, merged)` 时拒绝第二个实例（`--force` 逃生）；shell 超时不会杀进程组，缺互斥会产生重复记录（本任务 round 0 的 8b-short 两行／差 0.25%）。② **清理纪律**——只用 `kill <pid>` 或 `pkill -x <exact-name>`；`pkill -f "<子串>"` 会命中当前 shell 自己的命令行并静默杀死命令块（本任务两次）。后台任务日志写独立文件，前台命令用 `timeout` 兜底。③ **空采集标记**——profile 目录生成后若无 `OpBasicInfo` 行，重命名为 `*.invalid`（对账时不再靠人工判空；本任务 vselect 分支留过空壳目录）。④ **子进程树清理**——`subprocess.run(timeout=)` 只杀直接子进程，`msprof op → msopprof → python launch → bishengir-compile` 四层树的下层变孤儿并占住设备（后续 case 串行化/超时，看起来像该 shape 卡死）：超时清理按进程组（`Popen(start_new_session=True)` + `os.killpg(os.getpgid(pid), SIGKILL)`），kill 后用 `ps` 确认无残留 `bishengir-compile`、用 `npu-smi` 确认设备空闲再进入下一 case；逐 case 重试只覆盖 timeout/signal/compiler-crash（2026-11-07 两任务实证：GQA-SW 探针 4 孤儿、Varlen report case 卡 26 min 而单独进程 53.8 s PASS）。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: Tier 2 待人工审批；四则同为 runner 卫生故合并为一条，与 VP-2026-0118 互链。2026-11-08 追加第四则与第二证（不改 proposal_id）。

## VP-2026-0209
- type: P
- title: varlen / 带 host 侧校验的算子的 Stage 4 采集口径——闭包直取 `_compile` + 预分配 workspace 直调 kernel 工厂，绕开 `Op.forward()` 的 host 校验（否则 msprof 采到 `GreaterEqual` 等 host 小算子）
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 的 `perf_opt/opt_log.md` §0 基线与 §10.1 流程问题 2：官方 report 两次为 `partial`（`invalid_reason = "unfiltered capture contains 6 distinct ops"`，3.26–4.30 µs 实为 host 侧 `GreaterEqual`）；用自建 `perf_opt/bench.py`——闭包取 kernel 工厂的 `_compile` + 预分配 ws 后直调 main，**绕开 `Op.forward()`**（其 host 侧校验算子 + `wrapped()` 内 `torch.zeros` 会污染采集窗口）→ profile 中只出现目标 kernel（`captured op = _gqa_sw_fwd_varlen_main_mix_aic`，20 条 OpBasicInfo），baseline/final 全部 valid
  - 适用条件：varlen / ragged / 带运行时 shape 校验或 Python 侧 prep 的算子；不适用于纯 GEMM/单 kernel 直调（此时 `--kernel-name` 过滤已足够）
  - 与既有条目的关系：`VP-2026-0081`（Roofline 采数管线与 kernel-name 耦合）与 `profile-collection.md` §4 第 4 条（官方 report 可比容差）都假设目标 op 能被 `--kernel-name` 唯一命中；本条针对「host 侧算子污染 + 不能只靠 name 过滤」的域，互补不重复
- repro: repro-missing（任务内 `perf_opt/bench.py` 为 varlen 专用 driver，随上一 session 工作区丢失；复现条件 = 对任一 `Op.forward()` 含 host 校验的算子，用闭包取 `_compile` + 预分配 ws 的 driver 采一次，确认 profile 内 `captured_op_count == 1` 且等于目标 kernel）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3 / 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: add（§3 命令模板段之后新增「绕开 host 侧校验的采集 driver」小节）
    定位锚: "实验迭代可用较小 `launch-count` 快速判断方向，但必须在日志中记录 launch 数；baseline、候选 winner 和 final 推荐使用更稳定的 launch 数复测。"
    new 文本（锚句之后追加）: **varlen / host 校验类算子的采集 driver**：当 `Op.forward()` 在 host 侧执行校验/`torch.zeros`/varlen 解码时，直接用 `--kernel-name` 过滤仍可能采到 host 小算子（表现为 profile 内 `captured_op_count > 1`，日志出现 `GreaterEqual` 等）。此时用「闭包取 kernel 工厂的 `_compile` + 预分配 workspace + 直调 main」的 driver，profile 内应恰有 1 个目标 op 目录（`captured_op_count == 1`，`OpBasicInfo` 行数 = launch 数）；否则该 profile 判 invalid，不进入对账。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T105108Z 2026-11-04
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；待第 2 次独立证据（不同任务、不同 host 校验形态）后随 profile-collection.md 合入。

## VP-2026-0210
- type: P
- title: nqb=2（相邻 q-block 共享 K/V 装载）的可编译形态——Q 取 `[2*bm,dim]` 单块 + 单 body + 运行期 `slot`（不做 trace-time 成对展开）；只共享 K 时 8b-long fp16 单 run +13.9%，正收益需 V 共享 + Vector 按 pair 拆 pass
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z（Stage 4 Round 11 编译探针）：`perf_opt/logs/round11/README.md` + `perf_opt/_gqa_sw_fwd_varlen_kernel_round11_nqb2_probe.py` 头部结构说明。task = pair（相邻 q-block `2*px`/`2*px+1`），`num_logical = ceildiv(ceildiv(max_seqlen_q, bm), 2) * H * B`；活 L1 恰 3 块（`l1_q [2*bm,dim]` 32 KB @dim128 + `l1_a [bm,256]` 32 KB + `l1_b [bn_eff,dim]` 64 KB；原独立 O 缓冲去掉）；pass-1 单 body（1 次 K load + 1 次 `M=2*bm` 融合 gemm + 1 次 S store，两 q-block 的 S tile 是 L0C 相邻行）；pass-2 单 body 在 `for slot in T.serial(2)` 内用运行期 slot 取 P/O 行窗口；flag id = `s*L_max+l`（`2*L_max ≤ 15`）
  - 编译/启动实测：dim=128 冷编译+首启 **8.2 s**（基线同 session 26–42 s；msprof run 日志 bench 首调用 9.2 s、wall 31.3 s 含采集），0 SIGSEGV；L0 的 dim=64 冷编译 ~29 s。精度：可启动 shapes 的 out+lse smoke PASS（fp16/bf16、kv384、nc1/2/4/16/20）
  - 性能不成立（单 run）：8b-long fp16 `msprof op Task Duration` 1 launch **2282.1 µs** vs 基线 median-of-20 **2003.35 µs** = **+13.9%**；原因 = 仅共享 K（V 仍逐 qb 载入）+ 未携带 T1 depth-2 流水。正收益路径 = V 共享（Cube pass-2 变 band-outer）+ Vector 按 pair 拆 pass-1/pass-2（两 q-block 在线 softmax 状态同时驻留 UB，UB 翻倍）
  - 与既有条目的关系：`traps-compiler.md#TRAP-l1-4th-live-buffer-copy-segv` 的封堵点是「第 4 块活 L1 copy / trace-time 展体」，本条目给的是**可编译的替代结构**；与 `PL-1.12`（depth-2）互补——本形态未带 depth-2，两者尚未叠加
  - 边界：数值与形态绑定该 kernel（bm=64、bn=256、dim=128、H·B=128、Expert 双 Scope、910B3 20 AIC/40 AIV、tilelang 0.1.2+cebcec9d + CANN 9.0.0）；dim=64 另有运行期 L0A 越界（CG-2026-0030），该形态在 dim=64 不可用
- repro: repro-missing（任务内探针为整 kernel 76KB/约 1500 行，无条目 ID/版本戳头，非 ED-B 最小形态；复现条件 = 同一 Expert 两相位 varlen attention 上把 task 改为相邻 q-block pair：`l1_q` 取 `[2*bm,dim]`、pass-1 单 body `M=2*bm` 融合 gemm、pass-2 单 body + 运行期 slot，编译计时 + 同 workload msprof 1-run 对照；断言 ① 编译成功且 0 SIGSEGV；② 性能方向按 V 是否共享分档（仅共享 K 时允许回退））
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目 `PL-1.25-nqb2-pair-single-body`（置于 `PL-1.18-ssd-steady-structure-floor` 之后，与 `PL-1.9-blockwidth` / `PL-1.12-task-pipeline-depth2` 互链）：

    ---
    id: PL-1.25-nqb2-pair-single-body
    kind: pattern
    family: [attention, gqa, varlen, expert, mixcv]
    apis: [T.Kernel, T.serial, T.copy, T.gemm]
    dtype: [fp16, bf16]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z
    toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
    repro: repro-missing
    ---

    ### nqb=2 的可编译形态：Q 单块 `[2*bm,dim]` + 单 body + 运行期 `slot`

    **形态**（本条目是「可表达/可编译」的结构，不是收益结论）：task 取相邻两 q-block 为一个 pair；
    `l1_q` 用 `[2*bm,dim]` 一次装两个 q-block 的 Q（packed-NHD 行连续 ⇒ 一条 GM→L1 copy）；
    pass-1 每 band step 只有一个 body：一次 K load + 一次 `M=2*bm` 融合 gemm + 一次 S store
    （两 q-block 的 S tile 是 L0C 相邻行）；pass-2 每 band step 只有一个 body，外层
    `for slot in T.serial(2)` 用**运行期 slot** 选 qb0/qb1 的 P 行窗口与 O 行窗口——不做
    trace-time 成对展开、不新增第 4 块操作数缓冲。flag id = `slot*L_max + l`，判据 `2*L_max ≤ 15`。
    **实测**：dim=128 冷编译+首启 8.2 s（同 session 基线 26–42 s），0 SIGSEGV；8b-long fp16 单 run
    2282.1 µs vs 基线 2003.35 µs（+13.9%）——因为只共享 K、V 仍逐 qb 载入且未带 depth-2 流水。
    **优化见解**：本形态把「循环体翻倍」转化为「数据布局翻倍」——成对信息落在 `2*bm` 的行维上由
    一次 gemm 吸收，控制流保持单 body；这正是绕开 `TRAP-l1-4th-live-buffer-copy-segv` 两种编译
    封堵（第 4 块活 L1 的 GM→L1 copy / 成对展开不收敛）的关键。正收益的先决条件是 V 也走
    band-outer 共享（Cube pass-2 一次 V load 供两 q-block），代价是 Vector 侧两 q-block 在线
    softmax 状态同时驻留 UB（UB 翻倍）——下一道门是 UB 预算。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；待同族任务补知识域 repro 后合入。单 run +13.9% 只作方向参考，不作为收益断言。

## VP-2026-0211
- type: D
- title: dim=64 且**动态 `T.serial(loop_count)` 的最后一轮 band tile 满宽**（tnc == bn_eff == 256）时，nqb=2（`M=2*bm` 融合 gemm）运行期 `mte load3d` L0A 越界——编译期无诊断；绕法 `dim==64 → bn_eff=min(bn_eff,128)` 实测 L0 4/4 + contract PASS（R12 修正 R11 的 `kv_len%256==0` 相关性口径）
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z（首证，R11）：`python _gqa_sw_fwd_varlen_kernel_round11_nqb2_probe.py --level L0` 在 dim=64 形状集 abort，三次 attempt 同一报文：`aicore exception: The operation address of L0A exceeds the maximum range of L0A. The write address of the mte load3d instruction is out of range`（error code 0x800000001000 / retCode 0x26 / mte error info 0x600000043；`perf_opt/logs/round11/nqb2_probe_L0_attempt{1,2,3}.log`）。复现集（dim=64）：t256(q=128,k=256) / t512(q=64,k=512) / c3(k 256/512) / c4(k=256) FAULT；c1(k=128) / c2(k=128 bf16) / t384 PASS。三次 attempt 的 core id / blk 不同（core 18/16/5，blk 5/3/9）⇒ 非单核固定地址
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z（第二证，R12：单进程单 shape 的等价驱动 `_gqa_sw_fwd_varlen_kernel_round12_l0a_repro.py`，导入同一 R11 nqb2 内核）：**故障集** m1(B=1,q=128/k=256) / t256(B=2) / t512(q=64/k=512) / c3(q=64,128/k=256,512) / t384(**q=384/k=384**) — 全部「最后一轮 band tile 满宽 tn=256」；**反例** k128(q=128/k=128，末 tn=128) / **q128_k384**（offset=256，末 tn=128，l=0 处有满 256 块但**不是最后一轮**）— 均 PASS。⇒ 谓词不是 `kv_len % 256 == 0`，而是「动态 `T.serial(loop_count)` 的最后一轮 band tile 满宽（tnc=bn_eff=256）」；两 gemm 均牵涉（只把 pass-1 N 限到 192 的 `p1tnc192`、只把 pass-2 K 限到 192 的 `p2tnc192`、以及 pass-1 M 钳到 bm 的 `mclamp` 三种诊断变体**仍 FAULT**——每个变体里另一个 gemm 的最后一轮仍满宽），只有宽度钳到 <256 才修复
  - **R11 反例口径更正**：R11 记为「k=384 PASS」的形状，R12 在等价驱动下按 `q=384/k=384` 运行时 **FAULT**（`repro_base_t384.log`），而 `q=128/k=384` PASS ⇒ R11 的该反例按「q=128/k=384、末 tile 128 宽」读才成立。两轮驱动的 shape 表是否逐字相同未能从 R11 工件确证（`未实证假设`），故本条目以 R12 实测（同一 R11 内核、逐 case 单进程）为准
  - **绕法实证**（R12）：`dim==64 → bn_eff=min(bn_eff,128)`（trace-time 常量分支）→ L0 **PASS 4/4 + contract**（fp16 max|dout| 1.953e-3、bf16 1.562e-2 tier2、max|dlse| 1.9e-6；`perf_opt/logs/round12/bn128_L0.log`）。dim=128 的 IR 不变（钳位在 `if dim == 64` 内）：8b-long fp16 单 run **2278.72 µs** vs R11 同内核 2282.1 µs（−0.15%，单 run 噪声级）
  - 对照：dim=128 的真实 workload（8b-short k=512、8b-long k=2048）编译+启动正常 ⇒ 与 dim 联合相关
  - 与既有条目区分：`TRAP-l1-4th-live-buffer-copy-segv` 是编译期 SIGSEGV/不收敛；本条是编译通过后的运行期 device abort；`CG-2026-0030` 登记缺口本体（R12 已按本条修正其 capability/blocked_algo/workaround 字段），本条是 pattern-library 侧的现象与绕法条目
- repro: repro-missing（R12 任务内 `perf_opt/_gqa_sw_fwd_varlen_kernel_round12_l0a_repro.py` 为最小化驱动但**不满足 ED-B**：116 行 > 100 行预算，且 `--impl base/bn128` 依赖任务内 76 KB 整 kernel 文件（`_gqa_sw_fwd_varlen_kernel_round11_nqb2_probe.py` / `..._round12_l0a_bn128.py`），非知识域自包含 ⇒ ED-C 机械转正不成立，按 merge-policy §1 保持降级。复现条件 = 同一 nqb=2 形态在 dim=64 上逐 case 单进程跑：断言 ① 最后一轮 band tile 满宽（q=128/k=256、q=64/k=512、q=384/k=384、c3）以该报文 abort；② 非末轮满宽（q=128/k=384）与末轮非满宽（k=128）PASS；③ `bn_eff=min(bn_eff,128)` 后同一形状集 PASS 且 L0 4/4；④ dim=128 同形态不 abort）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04（两任务同戳）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 `TRAP-l0a-load3d-out-of-range`（置于 `TRAP-DEVMODE-PERSIST-GEMM` 之后，与 `CG-2026-0030` 互链）：

    ---
    id: TRAP-l0a-load3d-out-of-range
    kind: trap
    family: [attention, gqa, varlen, expert, mixcv]
    apis: [T.gemm, T.copy]
    dtype: [fp16, bf16]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z（首证）；TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z（第二证 + 谓词修正 + 绕法）
    toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04（两证同戳）
    repro: repro-missing
    ---

    ### dim=64 + 最后一轮 band tile 满宽：`M=2*bm` 融合 gemm 的运行期 L0A 越界（编译无诊断）

    **现象**：Expert 双 Scope MIX attention 的 nqb=2 形态（Q `[2*bm,dim]`、pass-1 一次
    `M=2*bm` 融合 gemm、pass-2 运行期 `slot` 选 q-block）在 **dim=64** 且**动态
    `T.serial(loop_count)` 的最后一轮 band tile 满宽（`tnc == bn_eff == 256`）** 时 device abort：
    `aicore exception: The operation address of L0A exceeds the maximum range of L0A. The write
    address of the mte load3d instruction is out of range`（error code 0x800000001000 / retCode
    0x26 / mte error info 0x600000043）。`bishengir-compile` 全程无诊断，编译与启动均「成功」。
    **实测集**（dim=64，逐 case 单进程）：FAULT = m1(q128/k256,B=1) / t256(q128/k256,B=2) /
    t512(q64/k512) / c3(q64,128/k256,512) / q384/k384；PASS = k128(q128/k128，末 tn=128) /
    q128/k384（末 tn=128；l=0 处有满 256 块但不是最后一轮）。dim=128（k512/k2048）正常。
    两次 attempt 的 core/blk 均不同（18/16/5；5/3/9；blk 4）⇒ 随运行时 kv_len 变化。
    **谓词（R12 修正 R11 口径）**：不是 `kv_len % bn_eff == 0`——判据是「**最后一轮** band tile
    满宽」；故「有满宽块」不够、「有满宽块但不是末轮」不触发。两个 gemm 都牵涉：只把 pass-1 的 N
    或只把 pass-2 的 K 限到 192 仍 FAULT（另一个 gemm 的末轮仍满宽），pass-1 的 M 钳到 bm 也不修复。
    **绕法**：`dim==64 → bn_eff = min(bn_eff, 128)`（trace-time 常量）⇒ L0 4/4 + contract PASS
    （fp16 max|dout| 1.953e-3 / bf16 1.562e-2 tier2 / max|dlse| 1.9e-6）；dim=128 的 IR 不变。
    根因未隔离（怀疑末轮满宽时的 dynamic extent / L0A 行偏移计算，`未实证假设`）。
    **判据（事实）**：编译通过不等于可用——dim=64 的 L0 门禁形状集须实跑；出现该报文时核对
    「dim==64」与「是否存在末轮满宽 band tile」两项（`CG-2026-0030`）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z 2026-11-05
- confirmed_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z 2026-11-05（第二证：谓词由「kv%256==0」修正为「末轮满宽 band tile」+ `bn_eff=128` 绕法 L0 4/4 实证；两任务同 op 谱系，故仍按单条 pending 维护、暂不据本条推进其他计数）
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1（R12 的等价驱动 116 行且依赖任务内 76 KB 内核，非 ED-B 自包含，机械转正不成立）；待同族任务补知识域 repro 后合入 `traps-runtime.md`。合入时条目正文按本条 delta 的 R12 修正版落盘（谓词 + 绕法），不再使用 R11 的 `kv_len%256==0` 表述。

## VP-2026-0212
- type: R
- title: Stage 4 编译可行性探针须报三态（compiles-and-device-ok / compiles-but-device-faults / not-compiles），编译通过后补 L0 门禁形状集的 device smoke——「编译通过」不得单独作为可表达/可行证据
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z（用户交办即为 compile-feasibility probe，不要求性能）：dim=128 冷编译+首启 8.2 s、启动 8b-long 正常；dim=64 冷编译 ~29 s、编译成功，但 `--level L0` 在 t256/t512/c3/c4（`kv_len % 256 == 0`）上 abort：`aicore exception: L0A exceeds the maximum range / mte load3d out of range`（`perf_opt/logs/round11/nqb2_probe_L0_attempt{1,2,3}.log`）。若探针停在「compiles」即收尾，会把「可编译」误当「可用」，而 L0 门禁实际不通过
  - 同轮成本：3 次 L0 attempt 合计约 11 min（00:18/00:19/00:29 本地时间），msprof 1 run wall 31.3 s ⇒ device smoke 的代价远小于一次性能调优轮；而「编译不收敛/SIGSEGV」形态此前已由 VP-2026-0207 提出快速失败通道
  - 与 VP-2026-0207 互补：0207 管编译期快速失败（重试上限 1 + 超时 10 min）；本条管编译成功后的运行期分档（device smoke + 三态结论），两条同 target 不同阶段，建议同批审批
- repro: none（流程规则；复现条件 = 任一 Stage 4 编译可行性探针：编译成功后跑该算子 L0 门禁形状集，记录 `compiles-and-device-ok` / `compiles-but-device-faults`（附失败 shape 与报文）/ `not-compiles`；只有第一态可写「可表达且可用」）
- toolchain_stamp: 流程规则（无运行时依赖）；发生环境 tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（终止条件列表之后、`### Phase 3：产物收束` 之前插入 blockquote）
    定位锚: "### Phase 3：产物收束"
    new 文本（锚行之前插入）:
    > **编译可行性探针的三态结论（2026-11-04 GQA-SW-Varlen round 11 实证）**：Stage 4 的「只编不跑」探针在编译成功后**必须**再跑一次该算子 **L0 门禁形状集**的 device smoke，结论只允许三态之一：① `compiles-and-device-ok`（可表达且可用）；② `compiles-but-device-faults`（附失败 shape 集与 device 报文——**不得**写成「可表达/可行」，按 CG / trap 登记）；③ `not-compiles`（附编译失败文本或超时）。仅「编译通过」不得作为可行性证据：本任务 dim=64 冷编译 ~29 s 成功，但 `kv_len % 256 == 0` 的形状在运行期报 L0A 越界（3 次 attempt，~11 min），而 dim=128 同形态正常；若停在编译层会漏掉整类运行期故障。device smoke 成本（分钟级）远小于一次性能调优轮；编译快速失败仍按 blocked 条款（重试上限 1 + 单次 10 min）。
    动机: 「能否编译」与「能否正确执行」是两道门；本任务的可编译形态在 dim=64 上被 L0 门禁拦下，探针结论若只报 compiles 会污染后续晋升判断。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: Tier 2 结构化 diff 提案，待人工批准后由 `mode=apply` 执行（与 VP-2026-0207 同 target/相邻阶段，建议同批审批）。

## VP-2026-0213
- type: D
- title: 双 kernel 拆分（K1 = QK^T+online softmax / K2 = P@V）在 varlen 大 batch 上 **3.93× 回退**——launch 边界强制整个 P 物化 896 MiB（基线 P ring 仅 9.2 MiB）⇒ 跨 launch 的 P 写/读落 HBM（mte3 256→2114 µs、K2 mte2 1475 µs）；对照 DSA 族 P 仅 `(B,ns,8)` KB 级时拆分只付两次 launch
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z（R12 可行性原型，8b-long fp16，`perf_opt/logs/round12/`）：K1 = QK^T + online softmax → `ws_p/ws_scale/ws_ell/lse`；K2 = P@V → `ws_o` + 延迟 rescale + 归一。**L0 PASS 4/4 + contract**（fp16 + bf16）、K1+K2 编译 12.5 s、无 SIGSEGV ⇒ 正确性与可编译性均不构成障碍，否决项纯性能
  - workspace：`ws_p [4096,64,1792] f16 = 896 MiB`（基线 P ring 9.2 MiB）+ `ws_s [20,64,1792] f32 = 8.75` + `ws_o [20,64,896] f16 = 2.19` + `ws_scale 7.0` + `ws_ell 1.0` = **914.9 MiB**
  - 实测（`msprof op` Task Duration，各 3 次）：K1 **5533.86 µs**（5475.9/5533.9/5539.0；aiv vec 2701、aiv mte3 2114、aic mte2 679、cube 238、`aic_wait_id_sum` 4985）；K2 **2336.00 µs**（2323.4/2336.0/2484.7；aic mte2 1475、cube 245、fixpipe 305）⇒ 合计 **7869.86 µs = 3.93× 基线 2003.35 µs**。基线同 kernel 的 aiv mte3 ≈256 µs、vec ≈1189 µs ⇒ K1 两项分别 ×8.3 / ×2.3（mte3 反压进 vec）
  - 机制两条：① P 写落到 896 MiB（非 L2 驻留）区域 ⇒ 写侧带宽/延迟取代原 ring 复用；② 单槽 TASKDONE 握手使基线的跨相位重叠消失（K1 `aic_wait_id_sum 4985 µs`）。host 侧分块保 P 在 L2 的变体未实现，其地板估计 = K1 vec 1.19 ms + K2 mte2 ~0.5 ms + 0.6 ms 发射 ≈ 2.3–3.2 ms（1.15–1.6×）且仍丢 QK/PV 重叠（`未实证假设`，按算术外推）
  - 对照（跨族）：`CASE-dsa-decode-gather-floor` 族的两阶段 split-d 拆分（`p_ws (B,ns,8) f32` = 9 KB@b4）成立——两次发射、P/partial 体量 KB 级、无 HBM 往返；两案例的差别只在 **P 物化字节 / L2 容量** 之比
- repro: repro-missing（任务内 `perf_opt/_gqa_sw_round12_dual_bench.py` + 两个整 kernel 文件（K1/K2 合计 58 KB）不满足 ED-B 自包含与 100 行预算；复现条件 = 同一 attention 形态拆成「softmax 出 P 到 ws」+「P@V 独立发射」两 kernel，采 `msprof op --launch-count=3` 的 K1/K2 `Task Duration` 与 `aic_mte2/mte3/fixpipe`，断言只锁方向与量级——K1+K2 ≥ 2.5× 单 kernel 基线、且 K1 的 mte3 时间 ≥ 4× 基线 mte3）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV，1800 MHz）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add（负面条目，kind: pattern；编号合入时按当时最大 PL 编号校正）`PL-1.24-dual-kernel-p-materialize`：

    ---
    id: PL-1.24-dual-kernel-p-materialize
    kind: pattern
    family: [attention, gqa, varlen, mixcv, expert]
    apis: [T.gemm, T.copy, T.Kernel]
    dtype: [fp16, bf16, fp32]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z
    toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
    repro: repro-missing
    ---

    ### 反向：把 attention 拆成「QK+softmax」与「PV」两个 kernel 的代价由 P 物化量决定

    **形态**：K1 产出 `P`（fp16）与 `lse/scale/ell` 到 GM workspace，K2 读 P 做 P@V 并做延迟 rescale。
    跨 launch 后 P 无法留在 L1/UB ring，必须整份落 HBM。

    **实测（8b-long fp16，GQA H/B=32、D=128、(64,256)）**：`ws_p = [4096,64,1792] f16 = 896 MiB`
    （该 kernel 单 launch 内 P ring 仅 9.2 MiB）⇒ K1 5533.86 µs + K2 2336.00 µs = **7869.86 µs
    = 3.93× 基线 2003.35 µs**；分解：K1 aiv mte3 256→2114 µs、vec 1189→2701 µs（mte3 反压），
    K2 aic mte2 1475 µs（P 读 HBM 受限），单槽 TASKDONE 握手另使基线跨相位重叠消失
    （K1 `aic_wait_id_sum` 4985 µs）。正确性无问题（L0 4/4 + contract）、编译 12.5 s 无 SIGSEGV
    ⇒ 否决项纯性能。

    **判据（拆分成败）**：先算 P 全量物化字节 `B·H·Skv·D·elem` 与该器件 L2 容量 / 基线 ring 之比——
    KB–MB 级（如 decode 的 `(B,ns,8)` partial，9 KB@b4）拆分只付两次发射；varlen 大 batch
    的 `B·H·Skv·D` 级 P（本例 896 MiB）必然否决。host 分块保 P 在 L2 只能把地板压到
    1.15–1.6×（仍丢 QK/PV 重叠，算术外推、未实现）。

    **与既有条目**：`BP_pipeline_overlap` / `PL-1.18` 讲单 kernel 内重叠；本条是跨 launch 的
    物化税，二者不矛盾（先满足 L2 判据才有资格谈拆分）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: repro 缺失（双 kernel driver + 两个整 kernel 文件非 ED-B 自包含）按 merge-policy §1 降级 Tier 1；待同族任务补知识域 repro（可用小 shape 复现方向）后合入 attention.md。

## VP-2026-0214
- type: D
- title: BishengIR 的 UB 分配 = 全部 `memref.alloc address_space<ub>` 的**静态加和（无生命周期复用）**——Expert 两相位 attention 实测 157,952 B（154.25 KB）恰等于逐 buffer 之和；共享 scratch 形态余量 20.5 KB、两份 S 常驻超 11.5 KB
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z（R12）：以 round-11 nqb2 探针的 lowered kernel MLIR（Vector scope 与冻结 kernel 字节一致）逐行读 `memref.alloc … address_space<ub>` 求和 = **157,952 B = 154.25 KB**，与 BishengIR 输出完全相等 ⇒ 静态加和口径成立（不因生命周期不重叠而回收）
  - 逐 buffer 账（dim=128，per-AIV 行 32）：`ub_f16_N [32,256]f16 16.00` / `ub_f32_N`=S `[32,256]f32 32.00` / `ub_f16_D [32,128]f16 8.00` / `ub_f32_D [32,128]f32 16.00` / `acc_o [32,128]f32 16.00` / 9×`[32,1]` 标量 1.125 / `scales [224,1]f32 0.875` / `ub_diff [32,256]f32 32.00` / `ub_pen [32,256]f32 32.00` / `ub_rowi`+`ub_thr` 0.25 ⇒ **154.25 KB / 192 KB**
  - 投影（两个 q-block 在线状态同驻，per-AIV 行不变 32）：A 常驻状态翻倍 + 逐 band 共享 scratch = **171.50 KB（余 20.50，10.7%）**；B = A + 两份 S tile 常驻 = **203.50 KB（超 11.50）**；C = 每个 `[half,*]` 缓冲翻倍 = 282.25 KB（超 90.25）。⇒ 「V 共享 + Vector pair split」的容量门通过（A），否决来自 Cube 侧 band-outer pass-2 的串行化（见 CASE ⑦）
  - 收缩手段的实测代价（同族前序轮）：S 载体落 f16 −32 KB / +3.1%；bn 256→128 −48 KB / +26% 且杀 pipe2（`L_max=11 ⇒ 2·L_max=22 > 15`）；bm 64→32 +85%；`acc_o` 不可时分（跨整 band 活跃）
  - 与既有条目关系：`CONST-capacity-910B2C`（UB 192 KB + auto-multi-buffer 膨胀系数）与 `CONST-ub-budget-vs-pattern-cap`（共享预算而非 pattern 上限）讲容量与归因；本条补「分配器是否复用生命周期」这一维（显式逐 buffer alloc + `--enable-auto-multi-buffer=false` 结构下 = 不复用）
- repro: repro-missing（复现条件 = 任一 Expert 双 Scope kernel 用 `TILELANG_DUMP_IR`/cache 取 lowered `kernel.mlir`，对 `memref.alloc` 的 `address_space<ub>` 行按 shape×dtype 求和，与 BishengIR 报告/静态表比对相等；断言只锁「和 == 分配器口径」与「加一个不重叠生命周期的 buffer 后总量线性增加」）
- toolchain_stamp: tilelang 0.1.2+cebcec9d（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 条目 `CONST-ub-alloc-static-sum`（置于 `CONST-ub-budget-vs-pattern-cap` 之后，正文互链）：

    ---
    id: CONST-ub-alloc-static-sum
    kind: constant
    family: [attention, gqa, mixcv, expert]
    apis: [T.alloc_ub, T.copy, T.gemm]
    dtype: [fp16, bf16, fp32]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z
    toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
    repro: repro-missing
    ---

    ### UB 占用 = 全部 UB `memref.alloc` 的静态加和（显式逐 buffer alloc 结构下不复用生命周期）

    **事实**：Expert 双 Scope 两相位 attention（dim=128、per-AIV 行 32）的 lowered `kernel.mlir` 中，
    所有 `memref.alloc … address_space<ub>` 行的字节和 = **157,952 B = 154.25 KB**，与 BishengIR
    的分配结果完全相等——不因两个 buffer 的生命周期不重叠而回收/alias。逐 buffer：`ub_f16_N
    [32,256]f16 16.00` / `ub_f32_N`(S) `[32,256]f32 32.00` / `ub_f16_D [32,128]f16 8.00` /
    `ub_f32_D [32,128]f32 16.00` / `acc_o [32,128]f32 16.00` / 9×`[32,1]` 1.125 /
    `scales [224,1]` 0.875 / `ub_diff [32,256]f32 32.00` / `ub_pen [32,256]f32 32.00` /
    `ub_rowi`+`ub_thr` 0.25（合计 154.25 / 192 KB）。

    **推论（设计期预算口径）**：新增一份「在线状态」按全尺寸计入——两个 q-block 常驻状态同时
    驻留（acc_o + m/ell/alpha + scales 翻倍）+ 逐 band 共享 scratch（S/diff/pen/D 按 qb0 再 qb1
    顺序处理）= **171.50 KB（余 20.50）**；若两份 S tile 都常驻 = **203.50 KB（超 11.50）**；
    每个 `[half,*]` 缓冲翻倍 = 282.25 KB（超 90.25）。⇒ 容量门能否通过只看「驻留份数」，
    与「同一时刻实际用到几份」无关（静态加和）。

    **收缩代价（同族实测）**：S 载体 f16 −32 KB / +3.1%；bn 256→128 −48 KB / +26% 且杀 pipe2；
    bm 64→32 +85%；`acc_o` 跨整 band 活跃不可时分。
- status: rejected
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z 2026-11-05
- confirmed_by: -
- decided_by: evolver
- decided_note: 已被 R13 实测推翻（task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z 的 4×64 KB UB 探针：`hivm-plan-memory` 按 live range 复用，disjoint 变体第 4 块 alias 到 slot 0；concurrent 256 KB 报 `ub overflow, requires 2097152 bits while 1572864 bits available`）。R12 的 157,952 B 是 user `T.alloc_ub` 之和、不是 plan total（另含 38,656 B 编译器 staging；当前 kernel plan total = 196,608 B）；「不回收生命周期不重叠的 buffer」只成立于该 kernel 的零复用结构，不能泛化。替代条目 = `constants.md#CONST-ub-alloc-live-range`（第二证合入，含 repro）。

## VP-2026-0215
- type: R
- title: merge-policy §2 的 evolver 可写范围缺 `.agents/evolution/capability-gaps.md`——该文件被 AGENTS.md 强制维护、历轮蒸馏实际写入（CG 谓词纠错/occurrences 增量），须在权限矩阵中显式列出并给出字段级边界
- evidence:
  - 本轮实测冲突：task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z 的 R12 证据推翻 `CG-2026-0030` 的判定谓词（`kv_len % bn_eff == 0` → 「动态 `T.serial(loop_count)` 最后一轮 band tile 满宽」）并给出绕法（`dim==64 → bn_eff=128`，L0 4/4 PASS），但 merge-policy §2「允许写（distill 模式）」表只列 pattern-library 主题文件 / repro / bottleneck-patterns / algorithm-candidates / 其他 skill references / queue.md / stats.md，未含 capability-gaps.md；若严格按表执行，evolver 只能把"登记簿谓词已知错误"留给下一次任务自行发现
  - 历轮先例（evolver 已实际写入并单独提交）：`d7f0d9a0`（CG-2026-0030 与 VP-2026-0211 互链）、`01805d7a`（CG-2026-0018 证据行号与文档条款更正）、GQA-bwd/argmax/engram 蒸馏中的 CG 结构归位与 occurrences 增量（见 stats.md 2026-09-28 / 2026-10-31 / 2026-11-04 行）
  - 后果不对称：CG 条目是 `[DESIGN_LIMIT]` / capability-fill proposal 的上游输入，错误谓词会被设计期当作否决依据（本条即「dim=64 形状集不可用」的实际影响面）；而纠错本身是纯事实更新、可 git revert
- repro: none（流程规则；复现条件 = 任一蒸馏轮发现 capability-gaps.md 条目的谓词/证据被新任务实测推翻）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
- target_doc: .agents/skills/tilelang-skill-evolution/references/merge-policy.md
- delta: |
    动作: update（§2「允许写（distill 模式）」表末行之后追加一行）
    定位锚: "| `.agents/evolution/queue.md`、`.agents/evolution/stats.md` | 全权维护 |"
    new 文本（锚行之后插入）:
    | `.agents/evolution/capability-gaps.md` | **事实字段级更新**：`capability` / `blocked_algo` / `evidence` / `workaround` / `occurrences` / `last_seen` / `status`（open↔recurring）可 update（须带来源 task_id 与实测证据，dedupe 优先、不新开重复 gap）；**不得**借本行新增无任务证据的 gap、不得删除条目、不得改 `gap_id`/`created_by` |
    补充约束（同段追加一句）: capability-gaps 条目的谓词/证据被新任务实测推翻时，蒸馏阶段按「实测 > 推演、新版本戳 > 旧版本戳」直接 update 并在条目内留痕（保留旧表述为「旧口径」），与 §3 冲突消解同源；扩大范围（新增 gap、改编号、迁 recurrent）仍按任务侧登记流程执行。
    动机: 登记簿是 capability-fill proposal 与设计期否决的上游输入，错误谓词的代价由后续所有任务承担；而纠错是纯事实更新、可 revert，风险远低于流程规则（Tier 2 人工门的对象是"怎么做事"，不是"事实是什么"）。本轮若不写，CG-2026-0030 的 `kv_len%256==0` 与 `workaround: none` 会继续作为设计依据。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: Tier 2 结构化 diff 提案，待人工批准后由 `mode=apply` 执行。**本轮已按先例（d7f0d9a0 / 01805d7a）直接 update 了 CG-2026-0030 的事实字段**（谓词/blocked_algo/workaround/last_seen/evidence），并在进化报告中披露；本提案用于把该先例固化为权限矩阵条目，避免下轮再依赖「先例」判断。

## VP-2026-0216
- type: R
- title: optimize SKILL Phase 3「结构不可达」证据门槛补「单管道下界 = max(单管道时间)」必备项——先与目标时延比对，上界已短于目标时不再投形态实验
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z（R12）：8b-long fp16 的四管道绝对时间（round-9 final 画像，20 launch）`mte2 1291 / mte1 609 / cube 483 / fixpipe 625 µs` ⇒ 单管道下界 `max = 1291 µs`，对基线 2003.35 µs 为 **1.55×**；40% 目标需 1202 µs（1.67×）⇒ 「无论重叠多好都到不了」，该结论**先于**任何形态实验成立（本轮三条路线仍各自实测否决才收口：K-only 共享 +13.9%、双 kernel 3.93×、V 共享 UB 拟合但输在 band-outer 流水）
  - 若该步前置，可省掉：双 kernel 原型（2 个整 kernel + driver + 3+3 次 msprof，约 1 轮预算）与 V 共享的 UB 预算核算（逐 buffer MLIR 求和）——两者都只能给出"更差"或"不可用"，而 1.55× < 1.67× 一行算术即可终判
  - 与已 merged 的 VP-2026-0042（同条、已写入 optimize SKILL.md Phase 3 第 4 条）互补：0042 管「先例检索 + 同口径实测 + morph 对照」防**局部地板误判**；本项管「上界先算」防**不可能达标却仍投实验**——两者顺序为「先算上界（本项），上界达标后再做先例/morph 校验（0042）」
  - 上界来源与口径：需 `msprof op --aic-metrics=Default` 的单 kernel 四管道画像（`aic_mte2/mte1/cube/fixpipe` ratio × `aic_time`），不是 roofline ideal；本条只要求「同一画像内取最大单管道」，不要求跨工具链可比
- repro: none（流程规则；复现条件 = 任一以 Ratio 或绝对时延为目标、且已有基线单 kernel 四管道画像的调优轮）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 4 条「结构不可达证据门槛」段末追加分句；与已 merged 的 VP-2026-0042 文本衔接）
    定位锚: "（结论限定于冻结设计族 + 修正证据表 + 存活硬上限清单 + 根因记录——本轮已验证形态）。"
    new 文本（在同一自然段末尾追加）:
    「**上界先算（2026-11-05 GQA-SW-Varlen R12）**：断言不可达前先取基线单 kernel 的四管道绝对时间（`aic_mte2/mte1/cube/fixpipe` 的 ratio × `aic_time`），其 **`max(单管道)` 即该结构族的单管道下界**，与目标时延直接比对——上界已短于目标时（本例 max = 1291 µs ⇒ 1.55×，而 40% 目标需 1.67×）即构成充分否决证据，无需再投形态实验；上界达标后才进入先例检索与 morph 对照（前述 ①②③）。该界只对实测过的结构族有效，减少搬运/换载体可抬高它，故须写明"在本结构字节账下"。」
    动机: 一次画像 + 一行 max 即可终判的路由，本轮却先做了双 kernel 原型与逐 buffer UB 核算（约 1 轮预算）；门槛段的现行三条都在防"该做却没做到"，缺一条防"本就不可能"。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: Tier 2 结构化 diff 提案，待人工批准后由 `mode=apply` 执行。与 VP-2026-0042（merged，同 target 同一条）为顺序互补关系，apply 时锚点须为 0042 合入后的文本（已在 delta 中按该文本给出）。

## VP-2026-0225
- type: R
- title: 采纳路径按独立 dispatch 排（单次 5400 s 覆盖不了「结构探针 + L0 + A/B + `--level all` + 8 档 final」）——每次 msprof run 因 `cache.clear_cache()` 重编译 ≈230 s，`--level all` 单次 511–514 s、8 档 final ≈1400 s，采纳轮实测 ≈76 min
- evidence:
  - task TileOPs-_gqa_bwd_kernel-20261107T122917Z 的 perf_opt/opt_log.md §21.5/§21.8/§21.9：探索轮 `max_wallclock_s=5400` 已消耗 ≈5100 s（A/B 12 run = 2400 s、L0 238 s），采纳路径粗估还需 3000–3600 s ⇒ 当轮只追加 2 条 `phase=candidate`（937.12 / 5403.41 µs），best 与 Final Summary 保持原值；**未合入的原因是预算，不是数字不达标**
  - 独立采纳收尾 dispatch（14:11Z–15:27Z）实测：候选 `--level all` 514 s + 转正后门禁回归 511 s + 8 档 final 每档 160–270 s（合计 ≈27 min）+ 账目核对，总 ≈76 min，一次完成且全 PASS
  - 成本构成：`cache.clear_cache()` 使每次 msprof run 重编译（≈230 s/run）⇒ 8 档 final ≈1400 s；两轮 `--level all`（候选 + 转正）≈1025 s
  - 相邻提案：VP-2026-0178（大 kernel 按 launch 数分级采集）管「单次采集成本」，VP-2026-0195（采纳同步五步动作清单）管「采纳动作清单」，本条管「dispatch 级预算切分与预留」；三者同一 target 域不同作用面
- repro: none（流程规则；复现条件 = 任一结构候选数字已达标但 dispatch 预算耗尽的轮次，记录探索轮消耗、采纳路径实测成本、以及拆成独立 dispatch 后的完成情况）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+9cfae94c + CANN 9.0.0 / msprof 25.5.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-07
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 2「每轮执行」之前插入一个 blockquote，紧接「实验批处理 runner（T-4）」段）
    定位锚: "每轮执行："
    new 文本（锚行之前插入）:
    > **采纳路径的独立 dispatch 与测量成本估算（2026-11-07 GQA-Bwd Round 21 实证）**：结构化候选数字达标后，**合入本身**是一次独立预算事件，不在探索轮的剩余时间里「顺手」完成。成本实测：`cache.clear_cache()` 使每次 msprof run 重编译 ≈230 s ⇒ 8 档 final ≈1400 s；`--level all` 单次 511–514 s（候选一次 + 转正后门禁回归一次）；采纳轮合计 ≈76 min。因此：① 派发探索轮时预算只按「探针 + L0 + A/B」估算，不预留采纳；② 数字达标但预算不足时，本轮以 `phase=candidate` 记录实测值并如实报告 `winner_measured / not merged (wallclock budget)`，next-step 写「采纳路径需要一次 ≥2 h 的独立 dispatch」；③ 采纳 dispatch 只做「`--level all` → 备份 + 逐字节转正 → 转正后门禁回归 → 8 档 final → 账目核对」五步，不重开探索、不派生新候选。本轮按此形态一次走通（37 PASS，8 档无一回退）。
    动机: 本轮数字已达标（−22%/−29%）却因 5400 s 预算耗尽未合入，另起一次 dispatch 才完成——探索轮与采纳轮的成本结构不同（前者 A/B 主导、后者重编译 + 全量 final 主导），混排会让「已实测的收益」多等一轮。
- status: pending
- confirmations: -/-
- created_by: task GQA-three-ops-20261108（bwd: TileOPs-_gqa_bwd_kernel-20261107T122917Z）2026-11-08
- confirmed_by: -
- decided_by: -
- decided_note: Tier 2 结构化 diff 提案，待人工批准后由 `mode=apply` 执行；与 VP-2026-0178 / VP-2026-0195 同 target 域（采集成本 / 采纳动作）建议同批审批。

## Decided（merged / rejected / expired / conflict 归档）

## VP-2026-0002
- type: P
- title: sub-fp32 逐元素算子 fp32 中转模式：vcast(rint) 升 fp32 → fp32 域 v-prefix 链 → vcast(rint) 单次舍回（bf16 dtype 支持 + fp16 golden 对齐双触发）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 bf16 触发 + Stage 3 fp16 修复模式提案）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py（模块「Implementation Notes (attempt-2 precision fix)」等价性论证 + attempt-2 全量 62 PASS）
  - docs/Tilelang.language/数学操作/T.vadd.md §2.2.1（v-prefix 算术 dtype 矩阵不含 bf16）
  - docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 仅 rint；f32→f16/bf16 含 rint；bf16↔f32 仅 rint）
  - 第二证（不同任务，2026-09-17）：task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z——opt_log Round 3 v7_cf16（c_scaled 链 fp16 化变体在 **bf16 workload 编译失败** rollback：`T.vmul` bf16 × 的运行时实证，触发条件①独立复现）+ Stage 3 因子链形态（cb/C/dt(dtype)→f32 因子域 + `T.vcast(..., round_mode="rint")` 末端回写，L0–Boundary 全过——链结构同构第二证）
- repro: repro-missing（合入条目标注 repro-missing，ED-B 显式形态；两证任务内复现命令见 evidence——provenance 允许失效）
- toolchain_stamp: 首证 tilelang-mlir-dev dev root build（2026-09-07）；第二证 tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（2026-09-17）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/elementwise.md）
- delta: |
    add 新小节「sub-fp32 逐元素 fp32 中转模式」（原 delta 全文，含双触发条件 / 链结构 / 实测 / UB 预算四段）。
- status: merged
- confirmations: 2/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- confirmed_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17（bf16 触发编译实证 + 因子链形态第二证）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同族（lerp elementwise 2026-09-07 / ssd mamba MixCV 2026-09-17），达 Tier 1 阈值 2/2。第二证确证触发条件①（bf16 编译失败）与链结构；触发条件②（fp16 golden 对齐）未单独复现、由首证保留。本蒸馏周期合入 elementwise.md PL-1.17-subfp32-fp32-transit（front-matter 双任务溯源，repro-missing 显式标注）。

---

## VP-2026-0013
- type: P
- title: attention 族 Developer 性能阻塞（aiv_scalar>50% / CG-2026-0001 崩溃类）时先评估 Expert 双 Scope 形态再定编程模式——结构级绕法含硬边界清单
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0（动因：developer 基线 ~3.5 TOps/s，落后 torch-SDPA 10–25×）+ #§0.6 E2/E3
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#§0/§4/Round-7（developer 谱系阻塞项来源——谱系注明：tilelang 67db6f3 + CANN 26.0.rc1）
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py（Expert 结构全同构先例）
  - pattern-library §1.7（expert vs developer 同门 bench：长 KV 1.57–1.63× / 短 KV ~1.31× 回退）
  - 第二证（不同任务不同族，2026-09-17）：task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 3——**persistent 分核 + gemm + v-prefix 混排在 Developer 模式运行时崩溃**（"unaligned UUB addresses"，repro/DEVMODE_PERSIST_CRASH.py 双模式对照）→ 切换 Expert（显式双 Scope + pass_configs 关闭 TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION）后 L0–Boundary 全过 + Stage 4 调优 2.91×；触发类从「性能阻塞/条件构造崩溃」扩展到「persistent+gemm 模式级不兼容」，Expert 硬边界（pass_configs）获独立第二实证
- repro: pattern-library/repro/TRAP-DEVMODE-PERSIST-GEMM.py（知识域，2026-09-17 转正——Expert 数值断言 + Developer 崩溃双模式对照）
- toolchain_stamp: 首证 tilelang dev build 21586b5（2026-09-07）；第二证 tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（2026-09-17）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md）
- delta: |
    add §1 新小节「Expert 双 Scope 流水形态（Developer 阻塞的结构级绕法）」（原 delta 全文：判据 / 结构形态 / Expert 硬边界 / 实测收益代价 / 未文档化假设）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- confirmed_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17（persistent+gemm 模式级不兼容触发类扩展 + pass_configs 硬边界第二证）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同族（attention 2026-09-07 / ssd mamba 2026-09-17），达 Tier 1 阈值 2/2。第二证为「方向确认 + 触发条件扩展」关系：核心主张（Developer 阻塞 → Expert 双 Scope 结构级绕法）被独立确认，触发类扩展为三类（新增 persistent+gemm 崩溃，traps-runtime.md TRAP-DEVMODE-PERSIST-GEMM / CG-2026-0010）；user_requirement 指定 Developer 时以实测为仲裁依据切换。本蒸馏周期合入 attention.md PL-1.16-expert-dualscope-bypass。

---

## VP-2026-0014
- type: P
- title: 块内 mask 的向量构造链模式：arrange strides → vsub → vcmp(int16 索引) → vand → vselect；−1e38 有限哨兵 + mask 后置 softcap + 两段式分界零 mask 开销块
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3 / #§1.6.2-#1
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345-354（vcmp 标量 PrimExpr + vand 链）/ #L611（PIPE_V 内 T.arange strides）
  - developer 谱系对照：同函数 T.Parallel 谓词预填形态 aiv_scalar 88–89% 实测（opt_log §1，**谱系注明** tilelang 67db6f3 + CANN 26.0.rc1）
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4——opt_log.md Phase 1 判别链 + Round 2：**vcmp int16 全形态标量化实测**（236µs/链 @[32,256]，对角块聚合 ≈ 壁钟 45%，probe_vcmp_forms.py P6）——原 delta 链结构中 vcmp(int16) 主路径形态被证伪；核心方向（向量链替代谓词预填/规避标量化）二次实证：**算术惩罚掩码**（clamp+vmax/vmin 标量广播+vmul，band-free trace）修复后 14.2x；两段式分派升级为 trace 级（band-free 算术惩罚 / band-carrying 保留 vcmp/vselect 值替换语义——vcmp 标量化代价在该 trace 仅次要路径）
- repro: 任一含块内 mask 的 attention/causal/sliding-window 算子，以向量链替代谓词预填后对照 aiv_scalar 占比与 Task Duration（第五轮复现：`python perf_opt/bench.py perf_opt/_gqa_prefill_fwd_kernel.py --case 8b-long --use-default-config --msprof-loop 25` 外部 msprof op 同口径）
- toolchain_stamp: 第一证 tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0（developer 对照侧 67db6f3 + CANN 26.0.rc1）；第二证 tilelang 0.1.2+6797758（2026-09-15）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md）
- delta: |
    （第二证修正后合入形态）原 add §1「向量 mask 构造链」小节 → 实际合入：主体（算术惩罚掩码 + trace 两段式分派 + NaN 免疫边界 + 终值）已由第五轮 optimizer 任务内回写 attention.md PL-1.11 + traps-compiler.md TRAP-expert-v-operands 补充段；evolver 补遗两点入 PL-1.11：「−1e38 哨兵与 softcap 顺序」bullet（e^{−1e38−m} 对有限 m 下溢精确 0 规避 −inf−(−inf) NaN 路径 + mask 后置 softcap 保 P(OOB)=0 不变式）。③ K_A 分界公式取整方向推导模板仍归 VP-2026-0018（pending，合入时并入）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：vcmp 标量化证伪 + 算术惩罚掩码 14.2x——同主题不同任务的第二次独立实测，含链形态修正）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（2026-09-07 expert 迁移 / 2026-09-15 第五轮调优），达 Tier 1 阈值 2/2。第二证对原提案是「方向确认 + 形态修正」关系：向量链替代谓词预填的核心主张被 14.2x 二次实证，但 vcmp(int16) 主路径形态被证伪（标量化陷阱）——合入时以修正后形态为准（PL-1.11 算术惩罚掩码 + band-carrying 保留 vselect），原 delta 的 vcmp 链仅适用于 band-carrying 次要 trace。本蒸馏周期执行合入。

---

## VP-2026-0059
- type: D
- title: 逐 buffer 手工 UB 预算 vs BiShengIR 基础分配差 ~26KB@192KB（auto-multi-buffer 关闭仍溢出）——预算表 ≠ 可编译证明，宽 config 上靶前须编译探针
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Design-layer finding（DESIGN §4.5 预算 causal trace (bm=64,bn_eff=256,dim=128) = 178.2KB ≤ 192KB "✓"、§5.3 复述；实际 ~204.1KB 硬编译失败 `ub overflow, requires 1666048–1672192 bits while 1572864 bits available`）
  - 排除 multi-buffer 通胀：`bishengir-compile --enable-auto-multi-buffer=false` 仍溢出 → 基础分配差（疑未记载 extra/sync buffers，Stage 5 未进一步定位）；dim=64 smoke ~161.7KB 可过
  - 双路径复现（/tmp/opencode/s5_repro_mha_config.py，工厂直调 (4,512,32,128) causal fp16）：A `(64,64,1)` → E6 替换 (64,256) FAIL；B `(64,64,2)` → 逐字 bn_eff∈{64,144} OK（out/lse finite）
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4 Phase 1——probe_ub.py 编译探针 4 点标定（工具链 6797758，跨 commit 复现并精化为系数律）：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1% → **BishengIR UB 实际需求 ≈ 真实手工预算 × 1.10–1.12**；另实证 DESIGN §4.5 漏计 ub_cond2 8.2KB（「勿漏计小 buffer」）+ UB diet 绕法（N/D staging 合并 −24.6KB + rowmat 删除 −16.3KB → (64,256) 宽块解锁，解锁后掩码修复红利显形）
- repro: repro-missing（NPU 编译探针；两次任务内复现命令见各 evidence 末行——首证工厂直调双路径对照 / 第二证 probe_ub.py 变体扫描）
- toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD），均 + CANN 8.5.0 / Ascend910B2C（跨 commit 复现）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update 条目 TRAP-UB-multibuffer-inflation（追加第四证/姊妹现象，与 20/26 B/elem 平台律并列）——已执行：正文追加「第四证/姊妹现象（基础分配差）」段（×1.10–1.12 4 点标定 + 预算表≠可编译证明 + 编译探针校准法 + UB diet 绕法实证 + 与 constants.md CONST-capacity-910B2C / CG-2026-0008 互链）+ front-matter origin_task/toolchain 机械更新（追加两任务）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：跨 commit 复现 + 4 点标定精化为系数律 + UB diet 绕法）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务且跨工具链 commit（28783f45 / 6797758 均复现基础分配差），达 Tier 1 阈值 2/2。定量系数（×1.10–1.12）已由第五轮 optimizer 任务内回写 constants.md CONST-capacity-910B2C；本蒸馏周期合入 TRAP-UB-multibuffer-inflation 第四证段（陷阱视角与常数视角互链）。CG-2026-0008 同轮升级 recurring。

---

## VP-2026-0063
- type: D
- title: causal 多头域画像（从未结构化优化，headroom >10×）+ 两相位收益的 config 依赖性——窄 config 短 KV 1.30–1.32x 提速 / 长 KV 0.78x 回退 vs 单遍 expert；宽块解锁后反转（第五轮刷新）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Bench（vs Sep-07 expert 基线同门对照列；profile_run_msprof_20260915_065835.log vs 20260907_173440.log）
  - 基线画像（provenance，Sep-09 采数）：perf_opt/perf_records.jsonl round 11 reg* 行——上一版 full 路径 (64,64,1) reg8bshort 3764.05µs（2.25 TFLOPS）/ reg8blong 13265.27µs（5.17 TFLOPS）vs fa4096 两相位 87.6 TFLOPS
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4——opt_log.md Final Summary：UB diet 解锁 (64,256) 后 config 依赖性画像刷新——长 KV 从 0.78x 回退**反转为 11.1x 领先**（8b-long 16962.44→1191.63µs = 14.23x vs baseline / 11.1x vs 单遍 13265µs）、短 KV 7.25–7.35x、全域几何 8.53x；原 delta 的预言「宽块收益需 mask 链缓冲精简（DESIGN §5.4 Stage 4 候选形态）」被完整实现（UB diet −40.9KB + 掩码修复后宽块红利显形 bn=256 vs 144 再 −40%）
- repro: repro-missing（端到端 msprof bench；任务内复现 = wrapper tuned 分派激活下 `msprof op --kernel-name=_gqa_prefill_fwd_main --launch-count=20 --warm-up=5 ... python perf_opt/bench.py ... --use-default-config`，raw 对账 profiles/{phase1,final}/）
- toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15），均 + CANN 8.5.0 / Ascend910B2C（基线侧 0.1.2+3a214cde 2026-09-09 采数——跨 commit 对比，倍率含版本差异成分）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update 条目 PL-1.9-twophase（附加 causal 域数据点 bullet）——已执行：追加「causal 多头域画像与宽块解锁反转」bullet（headroom >10× 画像 + 窄 config 1.30–1.32x/0.78x config 依赖性 + 第五轮 UB diet 解锁后 11.1x 反转与几何 8.53x + bf16/fp16 平价，完整档案指向 PL-1.11）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：宽块解锁 + 0.78x→11.1x 反转的刷新证据）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（v3 重生成集成期 / 第五轮调优期），达 Tier 1 阈值 2/2。第二证是证据刷新而非矛盾：0.78x 回退系窄 config 下的结论，第五轮在宽 config 下将其反转——同一 config 依赖性主张的完整闭环。本蒸馏周期合入 PL-1.9-twophase causal 域 bullet。

---

## VP-2026-0035
- type: P
- title: Stage 4 交付的工厂内 tuned 分派表模式：TUNED_DEFAULT_CONFIGS 仅替换 wrapper 默认 config（显式 config 一律尊重）+ tuned 目标 case 纳入内嵌测试套件作 blocking 精度层
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py（L100–110 TUNED_DEFAULT_CONFIGS 表 + L187–195 分派逻辑「仅 caller 传 wrapper 默认 (64,64,1) 且 shape 命中时替换」+ L1447–1490 run_fa_tuned blocking 门〔4 目标 case dispatch 路径 tier-1 全 0 flips〕）
  - 同目录 opt_log.md#Iteration-7（最终组装：工厂内 tuned 分派表〔S4-5 模式〕；caller 传 wrapper 默认时生效、显式 config 一律尊重——保 wrapper 契约）
  - 谱系先例（同 op 前任务、非独立计数证据）：examples/multi_head_attention/_prev_task_20260907_developer_optimize/perf_opt/opt_log.md（S4-5 工厂内 shape 分派调优 config）
  - 第二证（不同任务，2026-09-10 蒸馏追加）：examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/_ada_layer_norm_kernel.py（TUNED_DEFAULT_BLOCK_M per-shape 表 + select_row_config(M,N) 同形态返回值 + dtype 预算回退）+ 同目录 opt_log.md#Iteration-4（「TUNED_DEFAULT_BLOCK_M 显式逐 shape 记录而非规则化」——泛化假设否定后分派表为正解）+ #Iteration-6（U 曲线闭合验证表条目鲁棒性）+ #Final-Summary（wrapper 切换块引用建议）
- repro: repro-missing（config 级交付形态模式，无单一代码 diff；两次实现的分派表结构镜像于各自 perf_opt 交付文件〔provenance，允许失效〕）
- toolchain_stamp: 第一证 tilelang 0.1.2+3a214cde / CANN 8.5.0 / 2026-09-08；第二证 tilelang 0.1.2+a83118285a / CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md §1.9）
- delta: |
    add §1.9 附注 bullet（合入时紧随「工厂级多 @T.prim_func 变体分派」bullet 之后）：
    「工厂内 tuned 分派表（config 级，S4-5 模式）」：wrapper 契约保持的分派语义——caller 传 wrapper 默认 config 且 shape 命中 TUNED_DEFAULT_CONFIGS 时替换为调优 config；caller 传任何显式非默认 config 一律尊重原值（不静默覆盖用户意图）。tuned 分派覆盖的目标 case 须纳入内嵌测试套件作 blocking 精度层（fa-tuned 层：dispatch 路径 tier-1 翻转数对照基线），防止分派表与全量套件漂移。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- confirmed_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10（norm 族独立第二证：TUNED_DEFAULT_BLOCK_M + 非单调 bm 甜点 + 邻域闭合验证 + wrapper default_config 集成差异）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同族（attention 2026-09-08 / ada_layer_norm 2026-09-10），达 Tier 1 阈值 2/2，本蒸馏周期合入 pattern-library/attention.md §1.9（附注 bullet 追加于「工厂级多 @T.prim_func 变体分派」之后，含第二证要点：非单调 bm 逐 shape 记录 / 邻域闭合验证 / wrapper default_config 集成差异——合入文本见该文件）。

---

## VP-2026-0044
- type: P
- title: row-reduction 族 NPU 迁移基准形态（三件套）：全 N 驻留 UB + 按 M 行块分核（persistent num_kernels = min(ceil(M/bm), aicore×2)）+ 尾块 T.min + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限才 N-tile 流式
- evidence:
  - 首证（logsumexp，2026-08）：examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py（L64-74 核实：T.Kernel(ceildiv(M, block_m)) grid 形态 + alloc_shared((block_m, N)) 全 N 驻留 + real_m = T.min(block_m, M − pid_m·block_m) 尾块 + 双显式 slice；DESIGN.md 分层测试计划）
  - 第二证（ada_layer_norm，2026-09-10，persistent 形态）：examples/ada_layer_norm/_ada_layer_norm_kernel/DESIGN.md §5.5/§0.6（persistent min(ceil(M/bm), 48) 分核 + T.serial grid-stride + 48 = 24 AICore × 2 实查）+ _ada_layer_norm_kernel.py（尾块 T.min + 双显式 slice + 垃圾行丢弃）+ Stage 4 调优 2.14x 全组几何平均（persistent 形态下 loads-first 结构收益最大化）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Transferable Lessons 首行：「row-reduction 族 NPU 迁移三件套已两次验证（logsumexp + 本次 ada_layer_norm）」）
- repro: 复现条件——任一 row-reduction 族（norm/规约，M 行 × N 列归约）迁移设计；驻留上限推算依赖 TRAP-UB-multibuffer-inflation 平台值（fp16 约 N>11520 @20 B/elem）
- toolchain_stamp: 首证 2026-08（详见 logsumexp DESIGN 版本戳）；第二证 tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-layernorm / ALG-rmsnorm / ALG-reduction 三行（R4 亲和要点 + kb_links + known_impl）：
    ALG-layernorm R4 追加 row-reduction 迁移基准形态（两任务实测：logsumexp grid 形态 / ada_layer_norm persistent 形态）——全 N 驻留 UB（流量 4MN 最小）+ 按 M 行块分核（persistent num_kernels = min(ceil(M/bm), aicore×2)，纯 Vector 核数翻倍实查）+ 尾块 T.min + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限才 N-tile 流式（+25% 流量，ada 设计期估算）；kb_links 补 TRAP-vrsqrt-plain-precision / PL-1.10-loads-first-decoupling / CASE-norm-adalayern-stage4，known_impl 补 ada perf_opt 指针。ALG-rmsnorm kb_links 补 TRAP-vrsqrt（rsqrt 直用同样受影响）+ R4 引用 ALG-layernorm。ALG-reduction kb_links 补 CASE-reduction-logsumexp / CASE-norm-adalayern-stage4 + R4 引用。
- status: merged
- confirmations: 2/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- confirmed_by: task logsumexp-migration-2026-08（首证证据经工件追溯，创建时一并附上——VP-2026-0001 先例形态；grid 非 persistent 形态的差异已在条目正文中如实标注）
- decided_by: evolver
- decided_note: 创建即达 Tier 1 阈值 2/2（logsumexp 2026-08 + ada_layer_norm 2026-09-10 两个不同任务），本蒸馏周期合入 algorithm-candidates.md 三行。R4 迁移基准形态的正文中「persistent 强化」仅 ada 单证（logsumexp 为 grid 形态），已在文本中标注两形态来源。

### 2026-09-10 批量实施记录（conductor-improvement-report，用户批准直接实施）

> 依据 `/home/tilelang/l00970450/upload/tilelang-mlir-ascend/conductor-improvement-report.md` 的实施路线图，用户批准 P0+P1+部分 P2 批量直接实施（本节为治理留痕，非 VP 提案）：
> - **T-1/VP-2026-0042（扩展版）**：perf-feedback.md 参照锚定门槛 + optimize SKILL.md [DESIGN_LIMIT] 证据门槛 + plateau 参照结构 diff 检查 + cases.md 参考实现集；
> - **T-2/VP-2026-0010**：iteration-diagnosis.md 交错 A/B 多 run 协议前置 + `.agents/tools/ab_test.py` 工具化；
> - **K-1/K-2/K-5**：pattern-library 拆分为目录（INDEX + 主题文件 + repro/，条目 front-matter，字节预算）；cases.md 参考实现集与 CG 互链；
> - **ED-A/B/C/D/F**：证据三件套语义重定义（distillation-rules/queue-schema/merge-policy）+ repro 规范与首批 5 个转正（实跑 PASS）+ developer/optimizer repro 沉淀步骤 + repro_runner.py/kb_lint.py；
> - **E-1/E-5/E-6（K-3）**：kb_search.py 统一检索层 + conductor 预注入 + kb_stale_check.py；
> - **E-3/E-4**：审批简报节奏 + 强证据快速通道 + apply 预验证（merge-policy §6/§8）；
> - **D-1/D-2/D-3/D-4/D-5**：verify_equiv.py 等价性机器验证 + gate S1-EQUIV-EXEC + hardware-cost-model.md + constants.md + 设计期探针选项 + algorithm-candidates.md + design_calc_check.py；
> - **T-4/T-6/E-2(度量自动化)**：run_experiments 模板 + Final Summary 跨 dispatch 汇总 + stats 指标机械抽取。
> 明确未实施：VP-2026-0019/0020（用户决定暂不应用）；P2 的 T-3 BO/T-5 谱系导航/E-2 opbench/D-4 网络检索增强。

## VP-2026-0042
- type: R
- title: optimize SKILL.md [DESIGN_LIMIT] 判定补「结构不可达」证据门槛——仓库内同族先例检索 + 同口径实测 + morph 对照排除局部地板误判 + 被推翻时修正附录机制（append-only）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（第二轮「对 H=1 大 S 硬目标不可达、fa4096 结构性地板 ≈125–131µs > 100µs」被第三轮同 API 两相位结构推翻：四目标全达成 12.94/18.11/30.88/98.05µs；两处根因 = ① Stage 1 算法调研漏掉仓库内两遍式 S/P 物化先例 examples/flash_attention/flash_attn_npuir.py（同 API 可达结构）② round-10 否决两遍式的「S 需全任务驻留 → flag 预算超 15 上限」系推理错误——全局 [S,S] ws + n-block 下标 flag（nk≤16）即可成立，参考与本轮终版为直接反例）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§11.3（[DESIGN_LIMIT] 结论修正段：局部地板 vs 绝对地板界定——L1 端口 ~150GB/s / L0C 128KB / store_fixpipe 仅 GM 三硬上限存活，被推翻的仅是「上限内不存在达标数据流」的推断）
  - 同目录 perf_opt/opt_log.md#Final-Summary-第三轮（[DESIGN_LIMIT] 修正回填记录）
  - 交叉引用：queue VP-2026-0025（弃选举证边界——设计期同族规则）、VP-2026-0041（morph ladder——最小增量对照的定位方法）
- repro: 复现条件——任一触发 [DESIGN_LIMIT] 双门槛（设计层归因 + >2x 结构性估计）且拟断言「当前工具链/硬件上限内不可达」的调优终局（实证：第二轮断言后第三轮被同 API 结构推翻）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 4 条末尾追加证据门槛段）
    定位锚: "4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。**「当前工具链/硬件上限内不可达」类断言的证据门槛**（2026-09-09 attention expert 三轮闭环实证：第二轮 [DESIGN_LIMIT] 断言 fa4096 结构性地板 ≈125–131µs > 100µs 目标，第三轮两相位重构同 API 达标 98.05µs 推翻）：① 断言前强制检索仓库内同族先例（`examples/`/`testing/` 是否已有同 API 可达结构），找到先例时同口径实测 + morph 式最小增量对照（iteration-diagnosis.md「参考实现对照」）排除「冻结设计族局部地板」误判——实测外推的地板只对实测过的结构族有效（前序实证证据边界，见 negative-claim-evidence.md 第 6 条 / queue VP-2026-0025）；② 否决备选结构的理由须为实测或文档条款，纯推理否决（如「S 需全任务驻留 → flag 预算超限」）标注为未实证假设；③ 已产出的 [DESIGN_LIMIT] 被后续实测推翻时不删改原文（append-only），以「修正附录」回填 perf_feedback.md（结论限定于冻结设计族 + 修正证据表 + 存活硬上限清单 + 根因记录——本轮已验证形态）。
    动机: 结构性负结论驱动 conductor「设计层重做/等待编译器补齐」逆向路由，误判代价是整轮设计方向（第二轮误判使达标结构延后一轮才被实施，参考实现就在仓库内而两轮算法调研均未纳入）；先例检索 + 同口径实测是最低成本拦截位。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: human
- decided_note: 2026-09-10 用户批准随 conductor-improvement-report T-1 批量实施（扩展版：perf-feedback.md §1 参照锚定门槛 + §2 schema 参照锚定章节 + optimize SKILL.md Phase 3 第 4 条证据门槛 + plateau 参照结构 diff 检查 + cases.md 参考实现集；gate S4-PERF-FEEDBACK-ANCHOR 同步）

---

## VP-2026-0010
- type: R
- title: optimize iteration-diagnosis Step 6 补小 kernel 交错 A/B 多 run 测量协议前置规则（<5% 差异即启用，勿等采纳后复核）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 首测 +4.4% 复测翻转为 tie，耗一轮）
  - 同文件 #iteration-4（fp32 采纳项单轮 -5.2% 系双态膨胀，A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证：run 间 ±2–3.5% 环境双态层）
- repro: 16m fp32 final(bs8192) vs baseline(bs2048) 交替 A/B/A/B 各 2 run：B {158.75, 158.59} vs A {163.65, 162.27}，配对次序稳定
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（评估规则列表新增一条）
    定位锚: "- 只有通过必测 dispatch 非回退检查的候选 winner，才能更新为全局 current best。"
    new 文本: 交错 A/B/A/B 多 run 协议前置规则（<5% 差异即启用）。
    动机: lerp_tensor 两次踩坑（v2_op2 与 fp32 采纳项）各消耗一轮复测才校正归因；协议前置可省两轮实验。
- status: merged
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: human
- decided_note: 2026-09-10 用户批准随 conductor-improvement-report T-2 批量实施（iteration-diagnosis.md Step 6 评估规则 + optimize SKILL.md Phase 2 第 10 步/核心防呆 + `.agents/tools/ab_test.py` 工具化——交错多 run + 合并中位差 + 配对符号检验）

---


## VP-2026-0001
- type: P
- title: run 级双态（同 kernel 跨 run ±3–5%）需交错 A/B 多 run 协议裁决小差异——BP_run_state_bimodality
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证 2026-08：run 间 ±2–3.5% 环境双态层、per-process 快/慢双态、<3% 结论须多 run）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-4（第二证 2026-09-07：1M fp16 跨 run 7.30–7.66；fp32 采纳项单轮 -5.2% 双态膨胀 → A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/final/ab_*
- repro: 16m fp32：final(bs8192) vs baseline(bs2048) 交替各 2 run，msprof op --launch-count=15 取 median 后合并比较
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0（lerp）；mish 见其 opt_log 版本戳
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_run_state_bimodality（目录行 + 正文节，追加于 BP_parameter_uncertain 之后）：触发信号（同配置跨 run ±3–5%、<20us 或贴地板 workload 差异落 3% 门槛附近、复测翻转）/ 反证（launch-count 消失→BP_measurement_resolution_limited；大 kernel 单 run 已稳）/ 动作（首次 <5% 差异即前置交错 A/B/A/B：≥3 独立 run、交替次序、合并中位；3% 采纳判断必须合并中位 + 配对次序稳定）/ 验证（run 次序与中位记录、配对方向跨 run 一致）。
- status: merged
- confirmations: 2/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- confirmed_by: task mish optimize 2026-08（证据经 mish opt_log 档案追溯，创建时一并附上）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（mish 2026-08 / lerp_tensor 2026-09-07），创建即达 Tier 1 阈值 2/2，本蒸馏周期合入 bottleneck-patterns.md（目录行 + BP_parameter_uncertain 节后追加正文）。

---

## VP-2026-0021
- type: R
- title: Stage 3 大工件会话超限空返回的防再犯——DESIGN.md 分段读取 + kernel 分段落盘纪律写入 developer agent 定义（两次空返回各耗 1579s/930s）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 3 attempt 1 fail 1579s / attempt 2 fail 930s，verdict=runtime；attempt 3 complete 7701s）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（标题行：「前两次会话超限空返回，本次完成」）
  - 第三次成功的「分段读取 + 增量写入」纪律账户来自 conductor 终态钩子输入（复盘工件未展开该纪律细节——provenance 如实标注）
- repro: 复现条件——一次性整读 175KB/1249 行 DESIGN.md + 单次巨型 Write 949 行 kernel 文件的子 Agent 会话（2026-09-07 本任务前两次 attempt）
- toolchain_stamp: opencode Subagent 会话限制（2026-09-07 两次实证）；tilelang 工具链无关
- target_doc: .opencode/agents/tilelang-op-developer.md
- delta: |
    动作: update（first_impl 模式首行 bullet 扩展）
    定位锚: "- Read `DESIGN.md` + `REVIEW.md`。"
    old 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。
    new 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。**大工件分段读写纪律**（会话超限空返回防再犯，2026-09-07 attention expert 任务两次空返回实证〔1579s/930s 白耗〕）：DESIGN.md 超过 ~1200 行 / 150KB 时按章节分段 Read（先目录 + §0.5/§0.6 决策 + §3 伪代码，再按需下钻），不一次性整读；`{op}.py` 生成用分段落盘（逐段写 /tmp 后 cat 合并，或先写骨架再增量追加），不用单次巨型 Write——超限空返回表现为无输出直接失败（runtime verdict），与代码错误同型、无法从 stderr 区分。
    动机: 本任务 Stage 3 前两次 attempt 以完全相同的输入空返回（会话超限），第三次仅改变读写纪律即成功（7701s 完成 949 行 kernel + 29 用例全过）——每次空返回消耗一次 attempt 预算与 ~15–26 分钟墙钟。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: human
- decided_note: 用户批准（E-3 审批简报，2026-09-11，组⑤「大工件/会话纪律」）；mode=apply 落盘——合入锚 .opencode/agents/tilelang-op-developer.md `first_impl` 模式首行 bullet（原行保留 + 大工件分段读写纪律文本行内扩展），E-4 预验证 PASS（副本应用锚点唯一命中 + standards_check 控制组/编辑组 delta 零新增 failure；repro 为会话级复现条件非可执行命令，回归 SKIPPED）。

---

## VP-2026-0022
- type: R
- title: 超长 DESIGN.md（>60KB）单次 Write 因 JSON 体积截断——tilelang-op-design SKILL.md Phase 4 补分段落盘规则
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1（Skill Flow Issues 首行：~1200 行 DESIGN.md 单次 Write 截断失败，5 段 /tmp 拼接后 cat 合并才成功；DESIGN.md 1235/1249 行为证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 attempt 1 fail 2960s——含该 Write 截断的恢复成本）
- repro: 复现条件——单次 Write 输出 >60KB 的 DESIGN.md（2026-09-07 本任务首设计 attempt）
- toolchain_stamp: opencode Write 工具 JSON 体积限制（2026-09-07 实证）；tilelang 工具链无关
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 首行后追加一段）
    定位锚: "基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节："
    old 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：
    new 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

      > **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。
  动机: 截断失败发生在长任务收尾（写盘即交付前），恢复成本一次完整 attempt；attention/mixed 类算子的 DESIGN 普遍超此规模。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: human
- decided_note: 用户批准（E-3 审批简报，2026-09-11，组⑤「大工件/会话纪律」）；mode=apply 落盘——合入锚 .agents/skills/tilelang-op-design/SKILL.md Phase 4 首行「基于 templates/design-template.md 模板，填充所有章节：」后追加「超长文档分段落盘」blockquote（锚行原文保留），E-4 预验证 PASS（同 VP-2026-0021 流程）。

## VP-2026-0081
- type: P
- title: msprof roofline Ratio 采数管线规则——--aic-metrics=Roofline 与 --kernel-name 过滤缺一不可（metrics 集不含 Roofline 不产出 visualize_data.bin；无过滤全捕获把 ZerosLike 等辅助 op 当目标，Ratio 0.15–0.65% 伪影实录）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4）：opt_log 第六轮头部「度量管线」段（TileOPs bench 框架 2026-09-16 修复：kernel-name 经 JitKernel_NPU 对象图解析 + 多 op 检测 + duration 匹配；禁止回退无过滤全捕获——profile_run_msprof_20260915_105716.log 的 0.15–0.65% 即该伪影，描述的是 ZerosLike 等辅助 op）+ Phase 1（修复后管线验证：--aic-metrics=Roofline + --kernel-name 过滤 → 单 op 捕获 mix_aic 无 ZerosLike 污染、bin roofline 条目可提取——第五轮 metrics 集不产出 bin 的问题确认并规避）+ DESIGN.md §11.2 指标结构锚定（Ratio = Perf/359.33 TOps/s，Computility 按 48 核计而实机 24 核——口径基数须实测锚定核对）
  - 现行 profile-collection.md 已有 target_kernel_name 命令模板与 captured_op 追溯要求（基础规则存在）；本条补 Roofline-metrics↔bin 耦合、全捕获伪影实录与口径基数核对（2026-09-15 会话产出过伪影 log——规则存在下的执行复发）
- repro: repro-missing（需 msprof + 目标算子环境；伪影对照 = 同 workload 无过滤 vs --kernel-name 过滤双跑，Ratio 条目差异即伪影量级）
- evidence（第二证追加，2026-10-30）:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z Stage 4：命令模板把 `--aic-metrics=...,Roofline` 固化为必含项（opt_log §1.2 显式引「见 queue.md::VP-2026-0081」）；捕获 `kernel_name_attempts=['main'] / kernel_name_resolved='main' / captured_op_count=1` 判 valid；本管线复现官方 benchmark.md 偏差 <5%（31.48/78.80/31.38µs 官方 vs 30.181/78.402/31.311µs 本管线，官方偏高 0.1–4.8%，wrapper padding/host 差异）——补「可比容差」数据点。出处：该任务 perf_opt/opt_log.md §0/§1.2（provenance，允许失效）
- toolchain_stamp: 首证 CANN 8.5.0 msprof / Ascend910B2C / 2026-09-16（采数侧 tilelang 0.1.2+a13585dc）；第二证 tilelang 0.1.2+76e4550 + CANN 9.0.0 / Ascend910B3 / 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（captured_op 追溯要求行扩展）
    定位锚: "- `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。**roofline Ratio 口径采数（2026-09-16 attention 第六轮实证）**：① `--aic-metrics=Roofline` 单命令同时产出 Task Duration + 诊断 CSV + `visualize_data.bin` 预计算 roofline 条目（Ratio =「GM/L2 · GM Read + Write」条目 ratio×100）——metrics 集不含 Roofline 时不产出 bin（第五轮实证）；② `--kernel-name` 过滤缺一不可：无过滤全捕获会把 ZerosLike 等辅助 op 当目标 kernel（实录：Ratio 0.15–0.65% 伪影，描述的是辅助 op 而非目标）；③ Ratio = Perf/Computility，Perf = msprof 计数 FLOPs / Task Duration——指标奖励执行利用率、不奖励 padding 缩减（flops 与时长同降时 Ratio 不动），Computility 归一基数以实测锚定核对（如 359.33 TOps/s 按 48 核计而实机 24 核——attention.md PL-1.12 指标结构段）。
    动机: Ratio 类硬目标的采数链路三处坑（bin 不产出 / 辅助 op 污染 / 口径基数错）任一都会使目标判定失真——2026-09-15 会话已产出伪影 log，2026-09-16 修复管线后才可靠；规则化使下一次 Ratio 目标任务第一轮即采对。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: evolver（Tier 1 达 2/2）
- decided_note: 见本次蒸馏报告（2026-10-30 合入）

## VP-2026-0119
- type: D
- title: reduce 混合 dtype（fp16 src→fp32 dst）静默产出错误极值——文档示例与实现矛盾（T.reduce_max.md §2.4 示例按实测产出垃圾值）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1/2：design_probe_argmax_dbg.py 变体 B（kern_mixed，fp16 输入 + fp32 规约输出）extreme=[0.75,117,13,−152] vs 真值 [3.09,2.81,3.98,2.89]，全部落 2^30 哨兵；Stage 2 实机重放复现（dbg-B 哨兵 / dbg-C 同 dtype PASS 对照）；对照 docs/Tilelang.language/规约操作/T.reduce_max.md §2.4（dtype≠accum_dtype 示例）与 §2.2.1——文档示例与实现矛盾
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/argmax/_argreduce_kernel/history_version/design_probe_argmax_dbg.py（看 kern_mixed 输出行）
- evidence（第二证追加，2026-10-30）:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z Stage 4 round 2（R2-D/R2-E）：`T.reduce_sum` fp16 暂存 + fp16→fp32 dst 在完整 kernel 内编译报 `memref.expand_shape ... found 1, expected 2`；同形态最小探针编译通过但静默算错（fp16→fp32 abs-sum 3.73e6 vs fp16→fp16 886，差 3 个数量级）；fp16→fp16 正确但精度仅 ~1%；bf16 无 reduce 支持（T.reduce_sum 文档 §2.2.1）——半精度向量 GEMV 路线因此封死。出处：该任务 perf_opt/opt_log.md §2.4/§5.5（provenance，允许失效）
- toolchain_stamp: 首证 tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24；第二证 tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 + Ascend910B3 / 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-reduce-mixed-dtype-silent-corrupt：T.reduce_max/T.reduce_min 的 dst dtype 与 src 不一致时静默产出错误极值（非 cast、非报错——fp16 src→fp32 dst 实测 extreme 全落 2^30 哨兵级垃圾值）；docs/Tilelang.language/规约操作/T.reduce_max.md §2.4 的 dtype≠accum_dtype 示例与实现矛盾（文档示例不可信）。配套纪律（bf16 面）：reduce dtype 表不含 bf16——bf16 输入先 T.vcast(..., round_mode="rint") 到 fp32 再 reduce。诊断指纹：极值列全落哨兵/异常量级 + 索引主路径正确 ⟹ 混合 dtype reduce 而非搬运/索引链缺陷。
- status: merged
- confirmations: 2/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: evolver（Tier 1 达 2/2）
- decided_note: [2026-10-30 merged] TileOPs engram_decode Stage 4 为第二任务独立证据（reduce_sum 混合 dtype：kernel 内 expand_shape 编译失败 + 最小探针静默算错）→ 1/2→2/2 合入 traps-compiler.md TRAP-reduce-mixed-dtype-silent-corrupt。原注：Stage 2 重放为同任务非独立证据（证据链追加性质）；capability-gaps 候选（文档示例错误类），暂不重复登记。

## VP-2026-0145
- type: P
- title: 「已重叠」判据——`aiv_mte2_time + aiv_vec_time > Task Duration` 表示两 pipe 已在重叠，此时单独降低 MTE2 不会降低总时长，且放大暂存块会因 UB 压力挤掉 auto multi-buffer 造成 25–37% 回退
- evidence:
  - task TileOPs-_engram_decode_kernel-20261030T121905Z 的 perf_opt/opt_log.md §2.5（R2-A slab KS=32：b4 mte2 55.2→31.9µs、`mte2_active_bw` 36→63 GB/s，但 Task 78.402→107.162µs；R2-G 手动展开同因 97.502µs）；perf_records.jsonl candidate 行逐项可对账（provenance，允许失效）
  - 机制：78.4µs ≪ 55.2+56.4=111.6µs 即两 pipe 已重叠；slab 把 UB 占用从 2×(16,d) 推到 2×(32,d)，`--enable-auto-multi-buffer` 为被消费 buffer 再留一份的预算不够，多缓冲失效 → 退化为两 pipe 串行
  - 互补既有条目：PL-1.10-loads-first-decoupling 记「各 pipe 忙时之和 ≈ Task Duration ⟹ 零重叠」的正向信号；本条记该判据的另一半（和 ≫ Task Duration ⟹ 已重叠，改动方向反转）
- repro: repro-missing（复现条件：同一 kernel 两分支——A 用 BK=16 单块暂存、B 用 KS=32 slab + 内层 T.serial(2) 复用；比较 mte2_time 与 Task Duration 的方向差异。慢→快关键更改：slab 分支删除，回到单块暂存）
- toolchain_stamp: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add 条目 PL-1.21-overlap-saturation-signal（置于 PL-1.10-loads-first-decoupling 之后，二者互为正反两面）：

    ---
    id: PL-1.21-overlap-saturation-signal
    kind: pattern
    family: [elementwise, gemv, reduction]
    apis: [T.copy, T.serial]
    dtype: [fp16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_engram_decode_kernel-20261030T121905Z
    toolchain: tilelang 0.1.2+76e4550（commit 013dbbf）+ CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30
    repro: repro-missing
    ---

    ### 已重叠判据：`mte2_time + vec_time > Task Duration` 时降 MTE2 是无效方向

    **触发信号**：profile 里 `aiv_mte2_time + aiv_vec_time > Task Duration`（相加远超墙钟）——说明 MTE2 与 VEC 已在重叠，降低其中任何一条 pipe 的搬运量都不改变 Task Duration；此时「减少 DMA 指令数 / 放大暂存块」这类常规优化会反噬。
    **实测（910B3，engram_decode b4）**：slab 把 `mte2_time` 55.2→31.9µs、`mte2_active_bw` 36→63 GB/s，Task Duration 却 78.402→107.162µs（手动展开变体 97.502µs）；机制 = UB 占用 2×(16,d)→2×(32,d) 挤掉 `--enable-auto-multi-buffer` 的双缓冲，重叠消失退化为串行（同族现象见 TRAP-UB-multibuffer-inflation）。
    **动作**：先算 `mte2_time + vec_time` 与 Task Duration 的比值——≈1 时按 PL-1.10 做 staging 解耦（创造重叠）；≫1 时优先降低**最长的那条 pipe**（本例为 `aiv_scalar_time`/`aiv_vec_time`）或减小 UB 占用，不要放大 tile。验证指标：Task Duration 降而单条 pipe 时间可升可平；若仅 mte2 下降而 Task 上升即判本条成立。
- status: merged
- confirmations: 2/2
- created_by: task TileOPs-_engram_decode_kernel-20261030T121905Z 2026-10-30
- confirmed_by: task TileOPs-_engram_decode_kernel-20261030T145026Z 2026-10-31（round 3：R3-A/R3-D/R3-Q 三个只降 mte2 的分支 mte2 −45%、duration ±3%，且给出「放大 tile vs 破坏多缓冲」分界）
- decided_by: evolver
- decided_note: 合入 `.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md` → 条目 `PL-1.21-overlap-saturation-signal`（2026-10-31 蒸馏，evolver 执行）；合入时按第二证修正原表述——「放大暂存块挤掉 auto multi-buffer」过宽，实测分界为「引入第二段暂存/显式分块复用」才破坏流水，编译期 tile 常量放大到预算上限仍 +6%；repro 按 ED-B 标 repro-missing（效应只在完整 kernel 规模显现，复现条件已写入条目正文）。

## VP-2026-0177
- type: R
- title: iteration-diagnosis 补「工具链变更后的最小重验集」——上轮结论随戳失效时先复用上轮的**诊断结构**（降级/减法探针形态），用 1  Probe + 2 最强候选形态重验，而非重新枚举上轮全部参数分支
- evidence:
  - task TileOPs-_sparse_mla_kernel-20261101T014649Z：工具链 `0.1.2+5a90b99` → `0.1.2+63c8a2e6`，上轮 14 个分支结论技术上全部失效；本轮以 **1 个降级探针**（`p_contig`，1 次 80 s 采集）即复现「18.05× / gather 占 94.5%」并锁定结论，未重跑上轮 13 个变体（perf_opt/opt_log.md §7.1 第 1 行）
  - 现行规则（E-5 / 证伪协议第 2 条）只要求「工具链变更后标注待重验」，未给重验的最小集合；实践中有两条滑向：要么不复验直接引用旧数字（错误），要么按上轮分支表全量重跑（浪费）
  - 交叉引用：queue VP-2026-0089（stale 判定口径）/ VP-2026-0116（源码树过滤，假阳性治理）——本条补「重验集大小」这一维
- repro: none（流程规则，无运行时依赖；复现条件 = 任一工具链升级后的续调任务）
- toolchain_stamp: 流程规则；证据环境 tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add（在「已知迭代结论」小节之后插入独立段落「工具链变更后的重验纪律」）：
    工具链戳变化使上轮**结论数字**失效，但不使上轮**诊断结构**失效。重验的最小集合 = ① 把上轮最强诊断探针（降级/减法/地板探针）原样重跑一次，确认瓶颈归因与量级仍成立；② 重跑上轮 2 个最强候选形态在同一 smoke/shape 上的相对关系（是否仍同向、是否仍在噪声内）。只有在这三项与上轮不一致时，才进入参数分支重枚举。首证：14 分支→1 探针 + 2 形态，采集成本从 ~20 min 降到 ~3 min 且结论不变。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_sparse_mla_kernel-20261101T014649Z 2026-11-01
- decided_by: -
- decided_note: 与 VP-2026-0089/0116 同属重验域（stale 判定 / 假阳性过滤 / 重验集大小），apply 时三处互检。

## VP-2026-0178
- type: P
- title: 大 kernel（>100 ms/launch）的 msprof 采集预算与形态筛选纪律——大 workload 只取最小 launch 数做终判，形态筛选一律在小 workload/smoke 上完成
- evidence:
  - task TileOPs-_sparse_mla_kernel-20261101T014649Z：mainstream（566.6 ms/launch）8 launches 用 20 min（`r0_base_mainstream` 10:02→10:22），而 longer-kv/long-kv 同命令 <2 min；本轮形态筛选全部在 smoke（9 ms/launch，成本低 60×）完成，3 个形态 + 1 探针总采集 <10 min（opt_log §7.1 第 2 行）
  - 反例成本：若无此纪律，3 形态 × 3 tune workload × 8 launches ≈ 3 h，超单轮预算
- repro: repro-missing（复现条件：同一 kernel 在 smoke 与 tune workload 上各跑 `msprof op --launch-count={4,8}`，比较单次采集墙钟与结论同向性；断言只锁方向——同向且大 workload 单次成本 >10×）
- toolchain_stamp: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0（msprof 25.5.0）/ Ascend910B3 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    add（采集成本控制小节）：
    - **分级采集**：>100 ms/launch 的 workload 只用于终判，`--launch-count=4 --repeat 3 --warm-up 0`；形态筛选与归因探针一律在 smoke / 最小同路径 shape 上做（实测成本差 60×，结论同向）。
    - 本轮实测：mainstream 8 launches = 20 min，smoke（9 ms/launch）单次采集 ~20 s；3 形态 + 1 降级探针全在 smoke 上筛完后再进 tune。
    - 判定：若某候选在 smoke 上差异 <3%（噪声带），不进入 tune 级复测（省下的预算用于新的结构方向）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_sparse_mla_kernel-20261101T014649Z 2026-11-01
- decided_by: -
- decided_note: 与 VP-2026-0081（同一 target 文件，roofline 采数管线）互补：0081 管口径正确性，本条管采集预算。

## VP-2026-0179
- type: P
- title: 官方 Ratio 类 KPI 必先做「口径拆解」——`Ratio = ideal/duration`、`ideal = 硬件计数 ops / computility`；写出口径（`S·Skv·4B ÷ BW`）不是算法下界，比值型指标对「减少冗余工作」中性
- evidence:
  - task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z：bench roofline 的 153 µs = `S·Skv·4B = 268.4 MB ÷ 1.75 TB/s`（零计算零输入，仅 Logits 写一遍）——被上一轮当成算法下界得出「71× 空间」的错误判断；官方 `_parse_roofline` 的 951 µs 用**硬件计数 op**（cube 2.7488e11 + vec 9.566e9）÷ computility 299.09 TOPS，`Ratio = ideal/duration`
  - 同任务量化：fp32 下计数向量 op 单独需 2280 µs > cube 理想 932 µs ⇒ 结构上限 41.7%；fp16 使计数不变而 vec 峰值翻倍 ⇒ 上限 ~83%（实测 Computility 302.66、Ratio 16.83/16.45%，官方 report `examples/TileOPs/reports/tileops/20261101_120609_257701_FP8LightningIndexerOp/benchmark.md`）
  - 冗余工作中性：本算子 96% 的 kv tile 在因果窗口外（真实端到端可省，skip 后时延可降 ~20×），但分子分母同降 ⇒ Ratio 不变、每 tile 固定开销使 Ratio 略降；该方向本轮明确不投预算（opt_log §2(c) 结论 3）
  - 交叉引用：已有 merged 条目 profile-collection.md 的 roofline Ratio 采数段（VP-2026-0081）——本条补「读法 / 口径拆解」而非「采法」
- repro: repro-missing（复现条件：对同一最终 kernel 采 `--aic-metrics=...,Roofline`，读 `visualize_data.bin` 的 GM Read+Write 条目，打印 AI / perf / computility / ideal 四项并验证 `Ratio = ideal/duration` 与 `AI·BW vs computility` 的 min 关系；断言只锁恒等式，不锁绝对值）
- toolchain_stamp: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3（1800 MHz）/ torch_npu 2.7.1.post4 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    update（「roofline Ratio 采数」段落追加「口径拆解」子段）：
    **Ratio 的读法**（拿 Ratio 当 KPI 前先机械拆四项）：`Ratio = ideal / duration`；`ideal = counted_ops / computility`；`counted_ops` 来自硬件计数器（**与 dtype 无关**）；`computility` = cube + vec 峰值（**随 dtype 变**：fp32 64 lanes、fp16 128 lanes）。拆解动作：打印 `AI / perf / computility / ideal`，核对 `roofline_peak = min(AI·BW, computility)`。
    - **写出口径 ≠ 算法下界**：`S·Skv·4B ÷ BW` 只回答「只写一遍 Logits 最快多久」，不含输入读取与乘加——拿它当空间会得出数量级错误（实证：fp8 indexer 的 153 µs 被误读为 71× 空间，真下界 951 µs，实际工作计数 2.84e11 ops）。
    - **比值型指标对冗余工作中性**：跳过窗口外 tile 使分子分母同降，Ratio 不变（每 tile 固定开销还会使其略降）——Ratio 目标下优先提「单位工作量的效率」，不优先削冗余工作量。
    - 结构上限可算：`ideal / (max(cube_ideal, vec_ideal))` 即该 epilogue dtype 下的 Ratio 天花板（fp8 indexer fp32 41.7% / fp16 ~83%）。
- status: merged
- confirmations: 2/2
- created_by: task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z 2026-11-01
- decided_by: evolver（2026-11-02，第二独立证据达标：task TileOPs-_make_dropout_kernel-20261101T145102Z）
- decided_note: 与已 merged 的 VP-2026-0081（同文件采数段）为「采法 / 读法」互补。2026-11-02 合入 profile-collection.md §4 第 5/6 条：保留原三条读法，并追加第二证「减 op 数使 counted_ops 与 duration 同降 ⇒ Ratio 反降 0.57 点，不得跨 op 数版本比较、不得保留旧版刷 Ratio」。

## VP-2026-0180
- type: P
- title: epilogue dtype 是 Ratio 型 KPI 的一等杠杆——L0C→UB 直接落 fp16 使计数不变而向量峰值翻倍、acc_ub 减半以放大 block_N（本族实测 1.92×，天花板 41.7%→~83%）
- evidence:
  - task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z：`T.copy(acc_fp32, acc_fp16_ub)` + relu/加权/头归约全 fp16，再 cast 回 fp32 输出。单改 dtype：s4k 5449.5→4648.4 µs（−14.7%）；叠加 bN 64→128（fp16 使 `acc_ub` 由 128 KB 降到 32 KB）：2835.7 µs（−48.0%），官方 Ratio 8.76%→16.83/16.45%（1.85–1.92×）；精度最坏 `corr_diff` 1.67e-07（门限 1e-3），`max_abs` 0.31 来自 fp16 舍入
  - 机制（三条叠加）：① fixpipe L0C→UB 流量减半；② 每元素向量指令数减半（128 fp16 lanes vs 64 fp32）；③ UB 余量翻倍使 `bN` 可翻倍 ⇒ tile 数减半，摊薄每 tile 固定开销（scalar/DMA/sync）
  - **适用边界（同任务实测）**：`bQ·bN=1024` 下换轴优先抬 **bN**——`(8,128)fp16 = 2835.7` vs `(16,64)fp16 = 4779.0`（抬 bQ +68%）；`bN=128/st3` 2814.9（<5% 平区，不另立 winner）；`bN=256` 编译失败（CG-2026-0025）；`kv_group>1` 时 bN 须夹到 64（grouped 路径 UB 越限，编译期常量分支）
  - 不可照搬到「输出必须 fp32 且下游对舍入敏感」的契约；本算子 Logits 输出仍 fp32，只有内部 epilogue 降精度
- repro: repro-missing（复现条件：同一 kernel 两分支仅 epilogue dtype 不同（fp32 vs fp16 L0C→UB 转换拷贝），比较 `Task Duration`、`PipeUtilization` 的 vec 时间与 `acc_ub` 字节数；断言只锁方向——fp16 分支 vec 时间 ≤0.6×、可容纳的 bN 翻倍）
- toolchain_stamp: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3 / torch_npu 2.7.1.post4 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目 PL-1.22-epilogue-dtype-vec-peak（置于 PL-1.21 之后同域）：

    ---
    id: PL-1.22-epilogue-dtype-vec-peak
    kind: pattern
    family: [attention, attention_indexing, gemm]
    apis: [T.copy, T.cast, T.gemm, T.reduce_sum, T.vadd]
    dtype: [fp16, fp32]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z
    toolchain: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3 / 2026-11-01；Developer MixCV
    repro: repro-missing
    ---

    ### epilogue 落 fp16：计数不变、向量峰值翻倍、acc_ub 减半以放大 block_N

    **优化见解（机制归因）**：Ratio 型 KPI 的分子是硬件计数 op（dtype 无关），分母的向量峰值随 dtype 变（fp32 64 lanes / fp16 128 lanes）——降 epilogue dtype 同时缩时间与抬上限。次级收益来自容量：`acc_ub = q_rows·bN·4 B` 是唯一的 UB 瓶颈，fp16 使它减半，从而 bN 翻倍、tile 数减半、每 tile 固定开销（scalar/DMA/sync）被摊薄。
    **关键代码更改（慢→快）**：`T.copy(acc_f32_l0c, acc_ub_f16)`（转换拷贝）→ relu/加权/头归约全 fp16 → `T.cast` 回 fp32 输出；tiling 同步 `bN 64→128`。
    **实测**：5449.5→4648.4 µs（仅 dtype）→2835.7 µs（+bN128）；Ratio 8.76%→16.83%；`corr_diff` ≤1.7e-07。
    **适用边界**：换轴优先抬 bN（(16,64) 反而 +68%）；`kv_group>1` 夹 bN=64；bN=256 编译失败（CG-2026-0025）；输出契约要求 fp32 时仅内部 epilogue 降精度、输出仍 fp32。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z 2026-11-01
- decided_by: -
- decided_note: 与 VP-2026-0179（Ratio 口径拆解，说明「为何 dtype 能抬上限」）成对；合入时正文互链。

## VP-2026-0181
- type: R
- title: optimize SKILL Phase 0/1 增「指标口径拆解」首步——Ratio/roofline 型目标必须先解析 `visualize_data.bin` 的 AI / perf / computility / ideal 并写出恒等式，再排候选优先级
- evidence:
  - task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z：opt_log §7.1 第 1 行——本轮花约 25 min 才确认 `Ratio = ideal(ops/peak)/duration`、ops 与 dtype 无关而 peak 随 dtype 变；若开轮即知，R1b（fp16 epilogue）会被更早排为第一候选（实际它一轮内即贡献 1.92×）
  - 同任务 §2 三表：153 µs 写出口径 vs 951 µs 计数口径的混用，直接导致上一轮把「71× 空间」当作设计层证据（错误方向）
  - 交叉引用：queue VP-2026-0179（P，读法条目）/ VP-2026-0081（merged，采法条目）——本条是流程侧「何时做」的强制点
- repro: none（流程规则，无运行时依赖；复现条件 = 任一以官方 Ratio 为 KPI 的调优任务）
- toolchain_stamp: 流程规则；证据环境 tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    update（Phase 1 候选枚举之前插入一步）：
    **0. 指标口径拆解（Ratio / roofline 型目标必做）**：采一次 `--aic-metrics=...,Roofline`，从 `visualize_data.bin` 打印 `AI / perf / computility / ideal` 并写出 `Ratio = ideal/duration`、`ideal = counted_ops/computility`；确认目标是 compute-bound 还是 memory-bound，并算出当前 epilogue dtype 下的 Ratio 结构上限（`ideal / max(cube_ideal, vec_ideal)`）。未完成此步不得排候选优先级——实证：把「写出口径」当算法下界会给出数量级错误的可优化空间（fp8 indexer 71× 误判，真下界 951 µs）。
    动机: 口径误用会让整轮方向错（上一轮据 153 µs 判断有 71× 空间，实际结构上限 41.7%）；一次采集 + 四行算术即可避免，成本 <10 min。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z 2026-11-01
- decided_by: -
- decided_note: 与 VP-2026-0179（读法）/ VP-2026-0081（采法）三位一体：本条管「开轮即做」的强制时机。

## VP-2026-0182
- type: D
- title: `T.vadd` 不接受带 Var 偏移的 sub-region 实参——`buf[q*Hp:q*Hp+half, :]` 在 parser evaluator 阶段 `TypeError`，树形归约在本前端只能用整 buffer 表达（strided 行块下不可行）
- evidence:
  - task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z：想用 log2 树 vadd 取代 8×`T.reduce_sum` 以砍 AIV scalar（占 wall 31.8%）时直接失败——`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'Var'`（`tvm/tir/buffer.py:218` `Buffer.__getitem__`）；只有 `T.reduce_sum` 这类特化接口接受 region 实参
  - 失败形态保留在 `perf_opt/_fp8_lightning_indexer_kernel_round1_tree.py`（该形态即最小复现：整 buffer 树形归约可编译，行块切片不可）
  - 关联：TRAP-expert-v-operands（v 算子操作数规则）；本条是「操作数形态 = region 且偏移含 Var」这一维
- repro: repro-missing（复现条件：`T.vadd(buf[q*Hp:q*Hp+half, :], buf[(q+1)*Hp:(q+1)*Hp+half, :], dst[q*Hp:(q+1)*Hp, :])` 在 `T.Parallel`/`T.serial` 内以 Var `q` 索引 → parser 抛 TypeError；改为整 buffer 或静态切片可通过）
- toolchain_stamp: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3 / torch_npu 2.7.1.post4 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-vadd-var-offset-region（置于 TRAP-expert-v-operands 之后，正文互链）：

    ---
    id: TRAP-vadd-var-offset-region
    kind: trap
    family: [attention, attention_indexing, reduction]
    apis: [T.vadd, T.reduce_sum]
    dtype: [fp16, fp32]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z
    toolchain: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3 / 2026-11-01
    repro: repro-missing
    ---

    ### `T.vadd` 的 region 实参偏移不得含 Var

    `T.vadd(buf[q*Hp:q*Hp+half, :], buf[(q+1)*Hp:(q+1)*Hp+half, :], out)` 在 parser evaluator 阶段抛 `TypeError: int() argument ... not 'Var'`（`tvm/tir/buffer.py:218` `Buffer.__getitem__` 对 Var 取 int）——`T.vadd` 只接受整 buffer 或编译期常量边界；`T.reduce_sum` 等特化接口才接受动态 region。
    **后果**：log2 树形归约（砍 AIV scalar 的常见手法）在「行块 = strided 子区域」的布局上不可表达，只能退回 N 次整 buffer 归约。
    **绕法**：把归约改成整 buffer 形态（必要时 reshape 成首维为 2 的幂的视图，注意 expand_shape 对齐限制），或改用 `T.reduce_sum` 的现成语义；不能靠「把 Var 提成 Python int」绕过（那会把循环静态展开，与流水冲突）。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z 2026-11-01
- decided_by: -
- decided_note: -

## VP-2026-0183
- type: P
- title: decode（S_q=1）Phase-A 分段数按「单波 = 物理核数」反推：`ns_eff = min(num_split, nkv, max(1, 20//(batch*heads)))`，跨 shape 最优恒为 grid≈20（16 块）
- evidence:
  - task TileOPs-_mha_decode_paged_kernel-20261101 Stage 4 Round 4（同文件 `ns_target ∈ {4,2,1}` 三分支，msprof Task Duration / 20 samples）：batch2-page256 34.57 / 28.39 / **27.30**、longer-cache 21.71 / **18.34** / 25.47、shorter-cache 13.21 / **12.66** / 16.19、single-token-page128 29.34 / 25.46 / **21.82** µs；对应 grid 64/32/16/64 与 32/16/16/32 与 16/8/8/16。规律：`batch*heads=16 → ns=1`（16 块 = 0.8 波）、`batch*heads=8 → ns=2`（16 块）；`ns=1` 在 8 头只剩 8 块（0.4 波）反而 +39%。出处：`perf_opt/opt_log.md` §3 Round 4 + `perf_records.jsonl` round=4 的 12 条 candidate 记录（provenance，允许失效）
  - 机制：Phase-A 每 block 固定开销 ~5–6 µs（描述符 + 循环 guard）远大于按页增量 ~1.5 µs（mte2 0.55 + vec 1.34）；分段数减半 → 每 block 走更多页、固定开销分摊，但 grid < 核数时尾部核空闲
  - 关联：elementwise.md 整波次 tiling 判定式（VP-2026-0175，`tile = ceil(S/(cores×iters))`）是同一几何在不同维度上的形态；本条是 **N-split 轴**（减少分段数使 grid 落单波），非 tile 轴
- repro: repro-missing（复现条件：同一 decode kernel 仅改 `ns_eff`，跑 `ns_target ∈ {1,2,4}` 三档 × 四 workload，断言只锁方向——batch*heads=16 档 ns=1 最优、8 头档 ns=2 最优、ns=1 在 8 头回退）
- toolchain_stamp: tilelang 0.1.2+5b062018 + CANN 9.0.0 / Ascend910B3（20 Cube / 40 AIV，1800 MHz）/ torch_npu 2.7.1.post4 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目（decode 分核几何，置于 PL-1.9-blockwidth 之后，正文互链 VP-2026-0175）：
    **形态**：decode 类 Phase-A grid = `batch*heads*ns_eff`；`ns_eff = min(num_split, nkv, max(1, cores//(batch*heads)))`。
    **机制归因**：每 block 固定开销（描述符/guard）≈5–6 µs ≫ 每页增量 ≈1.5 µs ⇒ 在 grid ≥ cores 之前，减少分段数恒定获益；grid < cores 后尾部核空闲主导。
    **判据**：先算 `batch*heads`，取 `ns_eff = floor(cores/(batch*heads))`（下限 1），使 grid 落在 `[cores*0.8, cores]`；跨 shape 最优不在端点档的例外是 `batch*heads` 不整除 cores 时向单波贴边。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_mha_decode_paged_kernel-20261101 2026-11-01
- decided_by: -
- decided_note: -

## VP-2026-0184
- type: D
- title: runtime scalar `if` 包裹 in-place fragment 掩码链（`vbrc/vcmp/vselect` 写同一 fragment）在 npuir 上不生效——两任务两形态（本任务静默 no-op：三变体 L1 max_diff 逐位相同；GQA-bwd 条件体写 `qkt_ub` 后自动同步未覆盖：L0 smoke 全超限）
- evidence:
  - task TileOPs-_mha_decode_paged_kernel-20261101 Stage 4 Round 1：`_mha_decode_split_kernel_round1_guard.py`（prologue+循环内掩码套 `if col_lim < block_N`）/ `round1_guardA.py`（只条件化循环内）/ `round1_guardserial.py`（+`T.Pipelined→T.serial`）三变体 L1-1 均 FAIL，且 `partial` 失败值**逐位相同**（max_diff 2.719e-01）⇒ 条件体整体未生效，而非 guard 条件写错。出处：`perf_opt/opt_log.md` §3 Round 1（provenance，允许失效）
  - 第二证（不同任务）：task TileOPs-_gqa_bwd_kernel-20261031T051147Z `perf_opt/history/opt_log_prev.md` §4 Round 1 分支 `r1_maskguard`——`if q0 < kv0 + BM - 1:` 包裹 `vbrc/vcmp/vselect`，L0 FAIL（smoke dq 9.92 / dk 8.14 / dv 2.66 全超限），归因「条件分支内写 `qkt_ub` 后自动同步未覆盖 → 静默数值错误」，判「写法不可用」并 blocked
  - 与 TRAP-vselect-inplace-carried-state（loop-carried 状态丢失）、TRAP-tvm-parser-rules（if 生成 runtime if）相邻但触发点不同：本条是「条件块内 in-place 写 fragment」这一组合整体不可用，且失效为静默（无编译错误）
- repro: repro-missing（复现条件：同一掩码 kernel 两分支——A 无条件执行 `vbrc→vcmp→vselect` 写同一 fragment；B 把该链整体包进 runtime scalar `if`；断言 B 的 L1 输出与 A 逐位相同 = 条件未生效，或 B 数值超限 = 同步未覆盖）
- toolchain_stamp: tilelang 0.1.2+5b062018 + CANN 9.0.0 / Ascend910B3 / 2026-11-01（首证）；第二证 tilelang 0.1.2+（gqa_bwd 任务工具链，910B2C 族）/ 2026-10-31
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-runtime-if-inplace-fragment-mask（置于 TRAP-tvm-parser-rules 之后，正文互链 TRAP-vselect-inplace-carried-state）：
    **形态**：`if cond:` 包裹 `T.vbrc → T.vcmp → T.vselect(vis, x, sentinel, x)`（false 操作数 = 输出）在 npuir 上不可用——本任务三变体输出与未加 guard 完全一致（条件体未执行）；gqa_bwd 形态则是条件体内写 fragment 后自动同步未覆盖、L0 数值超限。
    **为什么重要**：「热路径只在尾页/对角块生效」类问题的首选手段即条件化；两条独立任务线先后踩中，代价为 1 轮实验预算。
    **绕法**：不做条件化——用算术掩码使条件天然成立（见 PL-1.11 第三现），或把条件提到 grid/分派层（trace-time 常量或独立 prim_func）。
- status: pending
- confirmations: 2/2
- created_by: task TileOPs-_mha_decode_paged_kernel-20261101 2026-11-01
- decided_by: -
- decided_note: 两个独立任务（mha decode 2026-11-01 / gqa_bwd 2026-10-31）达 Tier 1 阈值 2/2；因 ED-A 三件套缺知识域 repro（标 repro-missing），按 VP-2026-0142 先例保持 pending，待同族任务补最小 repro 后由下次蒸馏合入（合入时按第二证形态补齐「同步未覆盖」亚型）。

## VP-2026-0185
- type: R
- title: 多 op 采数伪影升格为强制项——`captured_op_count != 1` 一律判 invalid 并重采，禁止「按时延匹配挑一个 op」；skill 侧把 `--kernel-name` 从「应当带」改为「缺省必带 + 采后机械断言」
- evidence:
  - 本轮实证（伪影 ↔ 真值同 workload 对照）：task TileOPs-_mha_decode_paged_kernel-20261101 的旧官方 report `20261015_074433_562940_…` 中 `captured_op_count=8–9`、`kernel_name_requested=None`，解析器按时延匹配选中 `Cast_…` / `BroadcastTo_…` / `ZerosLike_…`，得 1.82–2.52 µs / Ratio 0.012–0.017%；同算子新 report `20261101_184433_143505_…` 带 `--kernel-name=split_partial` → `captured_op_count=1`、`op_names=['split_partial_mix_aic']`，真值 12.56–27.04 µs / 1.20–5.51%（均值 0.31%→2.38%）。两 report 同仓库并存，偏差 ~100×
  - 本条是**规则已存在下的复发**：VP-2026-0081（2/2 merged，profile-collection.md §4 第 2 条已写「`--kernel-name` 过滤缺一不可」）生效后，本会话仍出现第二个同型伪影（前一个 GQA-SW-Varlen 报 0.01% 量级，校准后该算子 report `captured_op_count=1` / Ratio 12.90–22.31%）⇒ 纯提示性规则不足，须机械拒绝
  - 影响面：伪影使「可优化空间」被高估两个数量级（0.01% → 2.38%），直接决定是否投入 Stage 4 预算与目标定档
- repro: none（流程规则；复现条件 = 任一 multi-kernel 算子跑无过滤 vs `--kernel-name` 双采集，比较 `captured_op_count` 与 Ratio）
- toolchain_stamp: 流程规则；证据环境 tilelang 0.1.2+5b062018 + CANN 9.0.0 / Ascend910B3 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md（主）+ .agents/skills/tilelang-op-optimize/SKILL.md（Phase 0 门禁一行）
- delta: |
    update（§4 校验采集结果，第三条 bullet 原位替换）：
    定位锚: "- 采到的不是 Cast / Mul / OnesLike / Random 等框架小算子。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      - **`captured_op_count` 必须恰为 1，且 `captured_op_names[0]` 属于本次目标 kernel**——`!= 1` 一律判 invalid 并重采，**禁止按 Task Duration 时延匹配在多个 op 中挑一个**（伪影实录：mha_decode_paged 旧 report 捕获 8–9 个 op〔Cast/BroadcastTo/ZerosLike + 真 kernel〕，时延匹配选中框架小算子 → 1.82–2.52 µs / Ratio 0.012–0.017%；带 `--kernel-name` 校准后 12.56–27.04 µs / 1.20–5.51%）。采集命令缺省必带 `--kernel-name={target_kernel_name}`；采后机械断言 `captured_op_count == 1`。
    update（SKILL.md Phase 0 采集前置，新增一行）：
      **采数门禁**：官方 report / 本管线 profile 的 `captured_op_count != 1` 时，该 Ratio 不得作为基线或优化空间依据，必须先按 `--kernel-name` 重采（VP-2026-0081 merged；2026-11-01 第二个同型伪影复发后升格为强制）。
    动机: 同一伪影形态在两个不同算子、两轮会话中重复出现，且均发生在已有提示性规则之后；把「提醒」改为「机械拒绝」是唯一能阻断的层级（伪影的代价是把可优化空间高估两个数量级）。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_mha_decode_paged_kernel-20261101 2026-11-01
- decided_by: -
- decided_note: Tier 2 待人工审批；与 VP-2026-0081（已 merged 的采数规则）为「同主题升格」关系，apply 时须先核对 profile-collection.md §4 锚文本仍为原文。

## VP-2026-0186
- type: R
- title: optimize SKILL.md Phase 3 第 3 条补「perf_opt 切换前机械比对默认参数常量」——`perf_opt/{func}.py` 与 wrapper 的模块级默认 dict 必须逐 key 一致（ast 静态比对），否则调优成果不上线
- evidence:
  - task TileOPs-_make_dropout_kernel-20261101T145102Z：R1–R5 已把 `block_size` 1024/2048 → 8192（`perf_opt/_make_dropout_kernel.py::_DEFAULT_BLOCK_SIZE`），但 wrapper `examples/TileOPs/tileops/kernels/regularization/dropout/dropout.py::_DEFAULT_BLOCK_SIZE` 仍为 1024/2048 ⇒ 官方 report `examples/TileOPs/reports/tileops/20261101_221636_325181_DropoutOp/benchmark.md` 三档 44.07/49.89/29.69%（均值 41.21%，`config` 列即 `block_size 2048/1024`）；conductor 同步 wrapper 后同 kernel 立刻 `20261101_223520_519180` 56.02/62.10/58.39%（均值 58.84%）。跨 3 份官方报告可对照（另 `20261017_080218_487917` Ratio 范围 26.35–49.88%）
  - 同任务 `.stage_state.json.user_requirement` 自述「wrapper block_size already mirrored to 8192」——与仓库实际状态相反，说明该检查必须机械执行、不能依赖任务自述或注释
  - 交叉引用：VP-2026-0077（双镜像 inode 同步，同一条目的邻位缺口）/ VP-2026-0064、VP-2026-0032（integrate_kernel 幂等与谱系核对）——本条补「常量值」维度，0077 补「文件在哪个 inode」维度
- repro: none（流程规则；复现条件 = 任一 Stage 4 改了 tile/`block_size` 类默认参数、且 wrapper 与 perf_opt 各自持有模块级默认常量的 TileOPs 算子，直接跑官方 bench 读 `config` 列）
- toolchain_stamp: 流程规则；证据环境 tilelang 0.1.2+d5838899 / CANN 9.0.0 / torch_npu 2.7.1 / Ascend910B3 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    update（Phase 3 第 3 条，句末追加）：
    定位锚: "若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。**切换/收敛时必须机械比对两侧默认常量**：用 `ast` 解析 `perf_opt/{func}.py` 与 wrapper 的模块级默认 dict 字面量，逐 key 比对（差异 → 阻断切换并在 opt_log 记录两侧值）；不得依赖任务自述或注释判断是否已同步——实证：dropout R1–R5 的 8192 未落到 wrapper，官方 Ratio 在 3 份报告中长期停在 41.21%，同步后同 kernel 立即 58.84%（同一二进制、同一 workload，仅 wrapper `config` 从 2048/1024 变 8192）。
    动机: perf_opt 目录内的调优成果对官方 KPI 完全不生效，且此期间所有「性能不再提升」的判断都建立在旧配置上；人眼核对注释已被证伪（任务自述与仓库实际相反），静态比对成本 <1 min。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_make_dropout_kernel-20261101T145102Z 2026-11-02
- decided_by: -
- decided_note: Tier 2 待人工审批；与 VP-2026-0077（同一条目、双镜像维度）建议同批审批，避免同一段文字两次改写。

## VP-2026-0187
- type: P
- title: 优化点：p 恰为 0.5 时用 hash 符号位代替「|h| ≥ T」阈值——`vclamp(h,0,1)` 取代 `vabs`+`vsub`+`vclamp`（−2 个 int32 向量 op，实测 −20.7%~−21.4%）
- evidence:
  - task TileOPs-_make_dropout_kernel-20261101T145102Z：`perf_opt/opt_log.md` §2.0(d)/§2.1（三 workload 27.898→22.116 / 73.929→58.118 / 28.735→22.716 µs，L0 22 例 + L1/L2/Boundary 全 PASS）
  - 机制：hash 在 int32 上均匀 ⇒ `P(h≥0)=0.5` 精确成立，符号位即无偏阈值；p≠0.5 时 `h−T` 会在 int32 端回绕，必须保留 `|h|` 路径（按编译期 `p == 0.5` 二选一，不做 shape 分派、不下放 wrapper）
  - 交叉引用：同一 hash 族的统计门禁见 VP-2026-0189（3-op 不可用）、成本账见 VP-2026-0188
- repro: repro-missing（任务内 `perf_opt/_r6_signed.py` 551 行且无条目 ID/版本戳头，ED-C 机械转正不成立；复现条件 = 同一 counter-hash dropout kernel 分别在 `|h|≥T` 与 `h≥0` 两种掩码形态下采 msprof op，比较 Task Duration 与 drop-rate/χ²）
- toolchain_stamp: tilelang 0.1.2+d5838899 / CANN 9.0.0 / Ascend910B3 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add 新 pattern 条目 PL-1.39-mask-signedbit-p05（elementwise 随机掩码族）：
    **触发**：核内软件 PRNG 生成 {0,1} 掩码且 p 为编译期常量。
    **形态**：`h` 为 counter-hash 的 int32 输出时，keep 判据 `|h| ≥ T`（`vabs`+`vsub`+`vclamp` 三 op）在 **p 恰为 0.5** 时可退化为 `m = clamp(h, 0, 1)`（单 op）——`P(h≥0)=0.5` 在 int32 均匀域精确成立，符号位即无偏阈值，掩码仍是 `(seed,idx)` 纯函数（确定性重放不变）。
    **边界**：p≠0.5 不可用（`h−T` 在 int32 回绕），按编译期常量 `p == 0.5` 二选一；与 `native_domain` 同属工厂期形态选择，不是 shape 分派。
    **收益**：−2 个 int32 向量 op（−8 B/elem UB），dropout 三 workload −20.7%~−21.4%；非线性于 op 数比例（11/13=0.85），因链缩短同时缩短关键路径。
    **验证**：rate 3σ + block-χ²(dof 4095) + lag1–4 自相关；rate 判据须用 ones 输入。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_dropout_kernel-20261101T145102Z 2026-11-02
- decided_by: -
- decided_note: 单任务证据；第二证出现（同族随机算子）后按 2/2 合入。

## VP-2026-0188
- type: D
- title: 910B3 软件 PRNG 掩码链的成本由 int32 域 op 主导：13 op 中 10 个 int32（64 elem/instr vs fp16 128）≈ 向量时间 87%；掩码段 3→1 个 int32 op 换 −21% 时延
- evidence:
  - task TileOPs-_make_dropout_kernel-20261101T145102Z：`perf_opt/opt_log.md` §2.0(b)（op 分解：索引 2 + hash 4 + 掩码 4 + 掩码 cast 1 + 数据 2；int32 占 10/13）+ §2.1/§2.2 实测
  - 同轮实测 cross-check：`aiv_vec_ratio 0.95–0.97`、`mte2_ratio 0.17–0.36`、`mte3_ratio 0.08–0.15`（DMA 完全被掩盖）⇒ 唯一杠杆是减向量 op，而非带宽/分核
- repro: repro-missing（统计侧脚本 `perf_opt/probes/hash_variants_r6.py` 104 行但头部无条目 ID/工具链戳，ED-C 机械转正不成立；复现条件 = 同一 kernel 逐 op 注释法/变体法测 Task Duration，按 dtype 统计 int32 op 占比）
- toolchain_stamp: tilelang 0.1.2+d5838899 / CANN 9.0.0 / Ascend910B3（40 Vector core）/ 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add D 条目（elementwise 随机掩码族成本账）：
    **成本结构**：910B3 AIV 上 int32 为 64 elem/instr、fp16 为 128 elem/instr ⇒ 核内软件 PRNG 链中 int32 段的时间占比约翻倍于其 op 数占比；dropout R6 基线 13 op 中 10 个是 int32，占向量时间 ≈87%。
    **推论（实测支撑）**：减 1 个 int32 op ≈ 8–9% 时延；掩码段 4→1 个 int32 op（13→11 op）实测 −20.7%~−21.4%，大于 op 数比例（0.85）——链缩短同时缩短关键路径。
    **排查顺序**：先算 int32 op 占比，再谈 tiling/带宽（同任务 mte2 0.17–0.36、vec 0.95–0.97 已排除搬运瓶颈）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_dropout_kernel-20261101T145102Z 2026-11-02
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；待同族任务补知识域 repro 后合入。

## VP-2026-0189
- type: D
- title: 反向：3-op counter hash 在 dropout 掩码场景不可用——去掉末次乘法后 bit31 强相关（lag1 自相关 −0.389 / +0.723，block-χ² 87 / 2578 @ dof 4095）⇒ 4-op（mul,shl,xor,mul）是该族实测下界
- evidence:
  - task TileOPs-_make_dropout_kernel-20261101T145102Z：`perf_opt/opt_log.md` §2.0(c) 变体表（V0 5-op 参照 χ²=4110；V1 符号位阈值 χ²=4129 采纳；V2 χ²=87 / 自相关 −0.389；V3 χ²=2578 / 自相关 +0.723）+ `perf_opt/probes/hash_variants_r6.py`
  - 判据与 Stage 3 同源：rate 3σ + block-χ²(dof 4095) + lag1–4 自相关；V2/V3 rate 本身达标（0.5 附近）却被独立性检验否决 ⇒ 只看 rate 不足以判定掩码质量
- repro: repro-missing（`perf_opt/probes/hash_variants_r6.py` 头部无条目 ID/戳；复现条件 = numpy int32 模型上枚举 hash 变体，对最终二值掩码做 lag1 自相关 + block-χ²）
- toolchain_stamp: tilelang 0.1.2+d5838899 / CANN 9.0.0 / Ascend910B3 / 2026-11-01（统计侧为 numpy int32 主机模型，与设备链逐位一致）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add D 条目（与 VP-2026-0187/0188 同族互链）：
    **负向结论**：counter hash 的乘法项不可省——`(c·A) ^ ((c·A)<<13)`（无末次乘）与 `c ^ (c<<13)` 再乘两种 3-op 形态，输出 bit31 均呈强相关（lag1 自相关 −0.389 / +0.723），block-χ² 87 / 2578（dof 4095，显著低于均匀期望）。
    **判据**：掩码质量必须**下探到二值掩码层**做独立性检验，不能只看 drop rate（两变体 rate 均合格）；4-op `mul,shl,xor,mul` 是该族实测下界。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_dropout_kernel-20261101T145102Z 2026-11-02
- decided_by: -
- decided_note: 负面数据条目；合入时与 VP-2026-0187 同段并列（可减的只有掩码段，hash 段已到底）。

## VP-2026-0190
- type: D
- title: `T.vbitcast` 不改变 buffer 的 dtype 绑定 → `vmul(fp16, int16)` 被后端拒绝（`requires the same element type`），「掩码 cast + scale 乘」无法融合
- evidence:
  - task TileOPs-_make_dropout_kernel-20261101T145102Z：`perf_opt/opt_log.md` §2.0(d) + `perf_opt/_r6_bitcast.py` 编译失败（`hivm.hir.vmul op requires the same element type for all operands`）
  - 复验于新工具链 d5838899（上一轮 90fb7ae3 未覆盖该形态）⇒ 该约束跨工具链存活
- repro: repro-missing（`perf_opt/_r6_bitcast.py` 为整 kernel 候选、无条目 ID/戳；复现条件 = 构造 `i32→i16` 后 `vbitcast` 到 fp16 buffer，再与 fp16 向量 `vmul`，观察 hivm.hir.vmul verify 报错）
- toolchain_stamp: tilelang 0.1.2+d5838899 / CANN 9.0.0 / Ascend910B3 / 2026-11-01
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add TRAP-vbitcast-no-dtype-rebind（traps-compiler.md v 算子操作数族邻位）：
    **现象**：`T.vbitcast` 只做位级重解释，**不改变 buffer 绑定的元素类型**；把它接到 fp16 buffer 后再与 fp16 张量做 `vmul`，BiShengIR 报 `hivm.hir.vmul op requires the same element type for all operands`（整 kernel 编译失败，不是精度偏差）。
    **含义**：「int 掩码移位 → bitcast 成 fp16 → 乘 scale」的融合路径不可用，掩码 cast 与 scale 乘必须是两个 op。
    **绕法**：`vcast`（真实类型转换）后再 `vmul`；或用算术掩码乘（0/1 × scale）保持同 dtype 链。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_dropout_kernel-20261101T145102Z 2026-11-02
- decided_by: -
- decided_note: 单任务证据，跨工具链复验；待同族第二证后合入。

## VP-2026-0191
- type: R
- title: 验收 Ratio 必须显式声明口径（计数口径 vs 声明式 roofline）——两者对「减 op」方向可相反，小 kernel 在计数口径下被系统性低估；op 数少的 kernel 以算子 `eval_roofline()` 的声明式口径为主、计数口径仅并列诊断
- evidence:
  - task TileOPs-_make_dropout_kernel-20261102T065942Z（R9，用户在本轮开头把主口径从 msprof 计数 op roofline 改为声明式 roofline）：同一 final kernel（`perf_opt/_make_dropout_kernel.py`，sha `1c969ba4…`）官方 report `examples/TileOPs/reports/tileops/20261102_154948_981690_DropoutOp/benchmark.md` 两口径并列——声明式 B 47.84 / 45.07 / 93.08 %（均值 **62.00%**）、计数 A 55.15 / 64.57 / 58.07 %（均值 59.26%）；R8 对照 report `20261101_235107_195598` 同两口颈为 B 56.76% / A 58.27%（`opt_log.md` §1 与 §4.5）
  - 声明式分子与实现无关：`DropoutOp.eval_roofline()`（`examples/TileOPs/tileops/ops/regularization/dropout.py` L142–166）返回 `(flops=N, bytes=2·N·elem_bytes)`，`ideal=max(bytes/1.75 TB/s, flops/peak)` ⇒ 三档 ideal 恒为 9.587 / 23.9675 / 19.174 µs（只由 N/dtype 决定，memory-bound）；计数分子则随发射指令变（AI 5.5552→5.3052）
  - **方向可相反的两组实证**：① 反向——R6（task TileOPs-_make_dropout_kernel-20261101T145102Z）减 2 个 int32 op、真实时延 −21%，计数口径 58.84%→58.27%（−0.57 点，分子分母同降），会误导「减 op = 负收益」；② 同向——R9 减 1 个 full-tile 广播 op，counted ops 仅 −4.5% 而 duration −8.5%，计数口径 +0.99 点。⇒ 两口径不可互推，报告必须声明主口径
  - 分母协议差异（切换口径时须固定）：官方 report 单采样 20.04 / 53.18 / 20.60 µs vs `perf_records`（`--launch-count=20` 中位）19.907 / 53.212 / 20.536 µs，差 ≤0.7%
- repro: none（流程规则；复现条件 = 任一以官方 Ratio 为 KPI 的调优任务，对同一 final kernel 同时算 `eval_roofline()/duration` 与 `perf_tops/computility`，断言只锁「减 op 时两口径可反向」这一方向关系）
- toolchain_stamp: 流程规则；证据环境 tilelang dev build `/mnt/workspace/tilelang-ascend-npu-ops/build` + CANN 9.0.0 / torch_npu 2.7.1.post4 / Ascend910B3 / 2026-11-02
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（「性能口径唯一」段句末追加新句，不新增小节）
    定位锚: "均以降低 `msprof op` 测得的 `Task Duration(us)` 为准。"
    new 文本: |
      均以降低 `msprof op` 测得的 `Task Duration(us)` 为准。**验收 Ratio 必须显式声明口径**：`Ratio = ideal/duration` 有两种 ideal——**计数口径**（分子是工具从发射指令反推的 ops，随实现变）与**声明式口径**（分子是算子 `eval_roofline()` 的 `max(bytes/BW, flops/peak)`，只由 shape/dtype 决定）。两者对「减少 op 数」的方向**可相反**：dropout R6 减 2 个 int32 op、真实时延 −21%，计数口径反而 −0.57 点；R9 减 1 个 full-tile 广播 op，计数口径 +0.99 点（counted ops 仅 −4.5%、duration −8.5%）。op 数少、非计算密集的小 kernel 在计数口径下被系统性低估 ⇒ 此类 workload 以**声明式口径为主**、计数口径仅作诊断并列报告；report / winner 判定必须写明本次用哪个口径，不得跨口径比较、不得保留旧口径版本刷分。
    动机: 口径是本轮唯一改变「策略方向」的输入——同一 final 在两口径下 59.26% vs 62.00%，且 R6 的真实提速 21% 在计数口径下是负收益，会引导优化器放弃正确方案；一次声明 + 并列报告即可避免，成本 ≈ 两行算术。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_make_dropout_kernel-20261102T065942Z 2026-11-02
- decided_by: -
- decided_note: Tier 2 待人工审批。与 VP-2026-0181（optimize SKILL 增「指标口径拆解」首步，同文件 pending）互补：0181 管「开轮即拆解」，本条管「口径必须声明为哪一个、小 kernel 用哪个」；两条锚点不同可先后 apply，建议同批审批。

## VP-2026-0192
- type: P
- title: 「1 元素 UB 标量 buffer + 广播 vadd」取代 full-tile `T.fill`+`T.vadd`——被删的 full-tile `vbrc` 写满 32 KiB int32/块，实测 −8.3~−8.7%（med-20 −8.5~−9.1%），收益远超其 1/N 的 op 份额
- evidence:
  - task TileOPs-_make_dropout_kernel-20261102T065942Z（R9）：`perf_opt/opt_log.md` §4.2（落地形态与前端约束）§4.4（NPUIR census `vbrc` 2→1）§4.5（B2 对照表）；`perf_opt/perf_records.jsonl` round=9（`r9_base` vs `r9_final`）
  - 实测：官方 report 时延 21.96→20.04 / 58.10→53.18 / 22.46→20.60 µs（−8.74/−8.47/−8.28%）；med-20 21.896→19.907 / 58.151→53.212 / 22.491→20.536 µs（−8.5~−9.1%）；`aiv_vec_time` 19.054→17.111 / 54.468→49.665 / 19.603→17.662 µs（−8.8~−10.2%），vec_ratio 0.95–0.97 不变 ⇒ 收益来自指令链变短，不是搬运变化
  - 前端边界（同轮实证）：`T.vadd(lane, T.cast(off+seed,"int32"), h)` 被拒（`AscendBinaryOp.buildTirCall` 只接受 int/float 字面量与 `tir.expr.Var`）；`with T.let(s, off+seed)` 绑定的 `tir.Var` 致前端段错误（rc=139）——1 元素 UB buffer 是该前端唯一被接受的非字面量标量形态
  - 复现状态（详见下 `repro` 字段）：任务内 `perf_opt/probes/round9_bitexact.py` 是整 kernel 对照 harness、头部无条目 ID/版本戳，ED-C 机械转正不成立；复现条件 = 同一 kernel 仅差 counter base 的物化形态：整 tile `T.fill` vs 1 元素 `sbase[0]=off+seed`，断言只锁方向——`vbrc` 计数 −1 且 vec_time 下降）
- repro: repro-missing
- toolchain_stamp: tilelang dev build `/mnt/workspace/tilelang-ascend-npu-ops/build` + CANN 9.0.0 / torch_npu 2.7.1.post4 / Ascend910B3 / 2026-11-02
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md
- delta: |
    add 新 pattern 条目 PL-1.23-scalar-ub-broadcast-vadd（elementwise 标量注入族；id 为待合入占位，apply 时按当时最大编号校正）：

    ---
    id: PL-1.23-scalar-ub-broadcast-vadd
    kind: pattern
    family: [regularization, elementwise, random_mask]
    apis: [T.alloc_var, T.vadd, T.fill, T.vbrc]
    dtype: [int32, fp16, bf16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_make_dropout_kernel-20261102T065942Z
    toolchain: tilelang dev build `/mnt/workspace/tilelang-ascend-npu-ops/build` + CANN 9.0.0 / Ascend910B3 / 2026-11-02
    repro: repro-missing
    ---

    ### 运行时标量注入：1 元素 UB buffer + 广播 vadd（取代 full-tile fill + vadd）

    **触发**：每块要加一个**整块统一**的运行时标量（block 偏移、counter base、seed 派生量），当前写法是「先 `T.fill(buf, scalar)` 把标量广播成整块，再做逐元素 `vadd`」。

    **机制归因**：`T.fill` 落到 `hivm.hir.vbrc`，它每块写满 tile（bs=8192 的 int32 = **32 KiB/块**写 UB），成本远高于一个平均向量 op；合并后 `vbrc` 计数 2→1，`aiv_vec_time` −8.8~−10.2%，时延 −8.3~−9.1%——即「1 个 op」的收益可达 ~9%，不是 1/11 份额。

    **关键代码更改（慢→快）**：
    ```
    # 慢：整 tile 广播填充 + 逐元素加
    T.fill(h, off + seed)
    T.vadd(lane, h, h)
    # 快：1 元素 UB 标量（tile 循环外分配一次）+ 广播 vadd
    sbase = T.alloc_var("int32")
    sbase[0] = off + seed          # 标量存储 1 lane，不产生 hivm.hir.v*
    T.vadd(lane, sbase, h)         # h[i] = lane[i] + (off + seed)
    ```

    **前端边界（同一工具链实测）**：`T.vadd(lane, T.cast(off+seed,"int32"), h)` 被 `AscendBinaryOp.buildTirCall` 拒绝（只接受 int/float 字面量与 `tir.expr.Var`）；`with T.let(s, off+seed)` 绑定的 `tir.Var` 使前端段错误（rc=139）。1 元素 UB buffer 是该前端唯一被接受的非字面量标量操作数形态。

    **正确性**：bit-exact——5 shape × {ones, randn} 全部 `torch.equal`、0/1 掩码逐位相同（hash 未改，只改了 counter base 的物化方式）。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_dropout_kernel-20261102T065942Z 2026-11-02
- decided_by: -
- decided_note: 单任务证据（同任务 R6 的 PL-1.39 掩码类条目与本条机制不同，不构成第二证）。第二证出现（任一同族 kernel 的「整块统一标量」注入）后按 2/2 合入；合入时与 TRAP-expert-v-operands 互链（该条目已记「v-op 标量操作数拒绝 tir.Cast」，本条补被接受的绕法形态）。

## VP-2026-0193
- type: D
- title: 910B3 上一个 full-tile `vbrc`（写满 32 KiB int32 UB/块）的单价 ≈ 8.3–9.1% kernel 时延——约 8–9× 于平均向量 op 的 1/11 份额
- evidence:
  - task TileOPs-_make_dropout_kernel-20261102T065942Z（R9）：`perf_opt/opt_log.md` §4.4 census（`vbrc` 2→1，其余向量 op 计数不变：`vadd` 1、`vmul` 4、`vshl` 1、`vxor` 1、`vmin`/`vmax` 各 1、`vcast` 2、`varange` 1）+ §4.5 对照表
  - 单价交叉验证：官方时延 −8.74/−8.47/−8.28%，med-20 −9.08/−8.49/−8.69%；同期 `aiv_vec_time` −10.2/−8.8/−9.9%；`aiv_vec_ratio` 0.95–0.97 与 mte2/mte3 ratio 基本不变（0.2476/0.2152/0.2723 → 0.2466/0.2451/0.3101）⇒ 差异全部来自向量指令链
  - 同族对照：R6 的 mask op 单价 ≈ 8–9%/op（queue VP-2026-0188），本条给出另一类 op（full-tile 广播写）的单价区间——两者都说明「按 op 数比例线性折算收益」会低估结构型 op
- repro: repro-missing（复现条件 = 同一 kernel 仅差「整块统一标量」的物化形态（`T.fill` vs 1 元素 `sbase[0]=`），采 `msprof op --launch-count=20` 比较 Task Duration / `aiv_vec_time` / `vbrc` census；断言只锁方向与量级区间）
- toolchain_stamp: tilelang dev build `/mnt/workspace/tilelang-ascend-npu-ops/build` + CANN 9.0.0 / torch_npu 2.7.1.post4 / Ascend910B3（40 AIV）/ 2026-11-02
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    add 常数条目（加入「逐 op 成本」族，与 CONST-vector-launch-overhead / CONST-mte2-degradation 并列）：
    ---
    id: CONST-vbrc-full-tile-cost-910B3
    kind: constant
    family: [regularization, elementwise]
    apis: [T.fill, T.vbrc, T.vadd]
    dtype: [int32, fp16, bf16, fp32]
    device: 910B3
    status: verified
    origin_task: TileOPs-_make_dropout_kernel-20261102T065942Z
    toolchain: tilelang dev build `/mnt/workspace/tilelang-ascend-npu-ops/build` + CANN 9.0.0 / Ascend910B3 / 2026-11-02
    repro: repro-missing
    ---

    ### full-tile `vbrc` 单价 ≈ 8–9% kernel 时延（bs=8192，int32）

    `T.fill(buf, scalar)` 落到 `hivm.hir.vbrc`，每块写满 tile：bs=8192 的 int32 = **32 KiB/块**的 UB 写。实测删掉一个这样的 op（11→10 逻辑 op，其余 op 计数不变）得 **−8.3~−9.1%** 时延、`aiv_vec_time` −8.8~−10.2%；对照平均向量 op 在 11 op 链中的 1/11 份额（≈9%）——两者数值相近但机制不同：此处收益来自**一次满 tile 写**，故按「op 数减少比例」线性折算收益，对这类结构型 op 会同时高估普通小 op、低估满 tile op。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_make_dropout_kernel-20261102T065942Z 2026-11-02
- decided_by: -
- decided_note: repro 缺失按 merge-policy §1 降级 Tier 1；待同族任务补知识域 repro 后合入（与 VP-2026-0188 同族，可一次补齐两条的 repro）。

## VP-2026-0217
- type: P
- title: 两段互斥生命周期的 UB tile 用 Python 前缀别名（`ub_f16_D/f32_D` → pass-1 的 `ub_f16_N/f32_N`）换出 24,576 B——GQA head-group batching G=2 的容量前提；op 数不变、数值逐位相同
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z（R13）的 `perf_opt/logs/round13/ROUND13_CONCLUSION.md` §2.3 与 opt_log §5 13.1/13.4：`ub_f16_D`/`ub_f32_D` 只在 pass-2 + epilogue 用、`ub_f16_N`/`ub_f32_N` 只在 pass-1 用，故 D 声明为 N 的 `[half,dim]` 前缀（`dim ≤ bn_eff`）别名 ⇒ **−24,576 B**（16,384 f32 + 8,192 f16），op 数不变；`--level L0` 4/4 + contract PASS 且数值与 baseline 逐位相同（1.953e-3 / 1.562e-2 tier2 / lse 1.9e-6）；8b-long fp16 1-run 2014.44 vs 2016.92 µs = −0.1%（中性）
  - 容量效果：未瘦身 G=2 = 175,616 user + 38,656 staging = 214,272 B（超 17,664 B，装不下）；别名后 151,040 + 38,656 = 189,696 B（余 6,912 B / 3.5%），G=2 由此可编译并拿到 1.102–1.145×（见 CASE ⑩）
  - 编译约束（同任务实测）：`vcast`/`vadd` 的形状推导要求切片用**静态 extent**（`[0:half,0:dim]`）；行 ≥ `rm` 的部分是 scratch、不被 store/read，故别名安全
- repro: repro-missing（任务内 `_gqa_sw_fwd_varlen_kernel_round13_shrink.py` 为 67 KB 整 kernel，非 ED-B 自包含；复现条件 = 同一两相位 kernel 把 pass-2 专用 tile 声明为 pass-1 tile 的静态前缀别名，比较 plan total 与 L0 数值；断言方向：UB 降 24,576 B、op census 不变、L0 bit-identical）
- toolchain_stamp: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-05
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目（编号合入时按当时最大 PL 校正）`PL-1.26-ub-alias-gbatching`：

    ---
    id: PL-1.26-ub-alias-gbatching
    kind: pattern
    family: [attention, gqa, varlen, mixcv, expert]
    apis: [T.alloc_ub, T.vcast, T.vadd, T.copy]
    dtype: [fp16, bf16, fp32]
    device: Ascend910B3
    status: verified
    origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z
    toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-05
    repro: repro-missing
    ---

    ### UB 前缀别名：互斥生命周期的 tile 共用一份 allocation（−24,576 B，op 数不变）

    **形态**：两相位 kernel 中 pass-1 专用的 `ub_f16_N [half,bn_eff]` / `ub_f32_N [half,bn_eff]`
    与 pass-2 + epilogue 专用的 `ub_f16_D [half,dim]` / `ub_f32_D [half,dim]` 生命周期互斥
    （pass-1 只写读 N，pass-2 只写读 D）。把 D 声明为 N 的**静态前缀视图**
    （`ub_f16_D = ub_f16_N[0:half, 0:dim]`，`dim ≤ bn_eff`）即可从 IR 中删掉两份独立 allocation。

    **实测（GQA-SW-Varlen round 13，dim=128 / bn=256 / half=32）**：用户 UB 157,952 → 133,376 B
    （**−24,576 B**），op census 不变，`--level L0` 4/4 + contract PASS 且数值与 baseline 逐位相同；
    8b-long fp16 1-run 中性（−0.1%）。该省量是 G=2 head-group batching 的容量前提：未瘦身
    G=2 = 214,272 B（超 17,664）→ 别名后 189,696 B（余 3.5%）。

    **编译约束（同任务实证）**：`vcast`/`vadd` 的形状推导要求切片边界是**静态 extent**
    （`[0:half,0:dim]`；运行时 Var 触发 parser `int(Var)` 拒绝）；行 ≥ `real_m` 的区域是 scratch，
    不被 store/read。别名省下的是**声明层面移除的 allocation**，不是分配器回收——该 kernel 的
    大缓冲全部跨 task loop 存活（零 live-range 复用，见 `constants.md#CONST-ub-alloc-live-range`）。

    **机制归因**：UB plan total = user allocation + 编译器 staging（本 kernel staging 38,656 B）；
    删掉一份声明即等量降低 plan total。与 `CONST-ub-budget-vs-pattern-cap` 的「alias 不产生预算」
    不矛盾：后者针对同时存活的尾工作集，前者命中的是声明级减法。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: P 类首证，待第二个不同任务的互斥生命周期别名实测后合入（合入时与 CONST-ub-alloc-live-range / CONST-ub-budget-vs-pattern-cap 互链）。

## VP-2026-0218
- type: P
- title: 多 head 在线状态并入单 tile 的列维（`acc_o[half,G*dim]`）+ `[half,1]` scratch 桥接逐 head 标量——绕开 `getBroadcastDim` 对**声明形状**的逐维比较（`src/target/codegen_npuir_api.cc:157`），G=2 Vector 侧 op 数不变
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z（R13）的 `ROUND13_CONCLUSION.md` §3.1：`getBroadcastDim` 比较的是 buffer **声明 shape** 而非 view shape，故 head 维放在 `acc_o` 的**列**（`[half, G*dim]`）使每个操作数保持 `dim0 = half`；逐 head 的 `m`/`ell` 经 `[half,1]` scratch（`ub_ms` / `ub_ells`）桥接
  - 同任务实证：G=2 的 Vector 侧用运行期 `for gh in T.serial(head_group)`（非 trace-time 展开）⇒ 编译器 staging 不翻倍（展开会再加 38.7 KB 并溢出），op 数不变；L0 4/4 + contract PASS，L1 targeted subset 12/12 PASS（含 groups=1 回退、H16/Hkv1、right-window、wl=0 bit-exact）
- repro: repro-missing（复现条件 = 多 head 状态写入单 tile 时，把跨 head 维放列维、逐 head 标量走 `[half,1]` scratch；断言方向：编译通过 + L0 与逐 head 独立输出 bit-identical；效应只在完整两相位 kernel 规模显现，非 ED-B 可运行最小形态）
- toolchain_stamp: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-05
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update（在 `TRAP-expert-v-operands` 的「v-op 操作数拒绝带 runtime 首维索引的多维 buffer 切片」段末追加；该段以「与既有『v-op 切片仅静态边界』（P10）同族：静态边界 + runtime 首维索引亦不可用。」结尾）
    定位锚: "与既有「v-op 切片仅静态边界」（P10）同族：静态边界 + runtime 首维索引亦不可用。"
    new 文本（锚句之后追加）:
    **〔2026-11-05 GQA-SW-Varlen R13 补充：`getBroadcastDim` 比较的是声明形状〕** 即使切片是静态边界，`getBroadcastDim`（`src/target/codegen_npuir_api.cc:157`）比较的仍是 buffer 的**声明 shape**而非 view/切片 shape；跨 head 维若声明在行上（`acc_o[G*half, dim]`），逐 head 视图的 `dim0` 与其它操作数的 `half` 不一致会触发 `buffer_shape0.size() == buffer_shape1.size()` 类检查失败。可编译形态 = 把跨 head 维放到**列**（`acc_o[half, G*dim]`），逐 head 的标量状态经 `[half,1]` scratch 桥接，使每个 v-op 操作数都保持 `dim0 = half`。同轮 G=2 的 Vector 侧用运行期 `for gh in T.serial(head_group)`（非 trace-time 展开）避免 staging 翻倍，op 数不变。
    动机: 既有段只覆盖「runtime 首维索引」这一触发面；声明形状比较是独立且更隐蔽的触发面（切片合法、编译仍失败），G=2/多 head 打包时直接命中。
- status: pending
- confirmations: 1/2
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: P 类首证；合入时与 VP-2026-0217（UB 别名）互链（两者共同构成 G=2 的可编译形态）。

## VP-2026-0219
- type: R
- title: 官方 report 的 correctness 阶段对**按 shape 逐个重编译**的 kernel 时间随用例数线性增长（26 例 40 min 只完成 8 个且无 run.json）——报告生成需要分阶段产物与时间预算，性能数据可由官方 benchmark harness 单跑取数并显式标注来源
- evidence:
  - task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z（R13）用户交办记录与 `examples/TileOPs/task_operators.md` 第 16 行：转正后从 `examples/TileOPs/` 跑 `python -m tileops.reporting.cli run --op GroupedQueryAttentionSlidingWindowVarlenFwdOp --prof-mode msprof`，correctness 因 G=2 每个 shape 重编译，40 min 只完成 **8/26**（8/8 PASS），**未生成 `run.json`**
  - 同轮替代路径：官方 benchmark harness 单跑 `benchmarks/ops/bench_grouped_query_attention_sliding_window_varlen.py` 10/10 PASS（1023.91 s），8 个 tune 用例 Ratio 14.32/14.20/25.02/24.35/15.78/15.53/25.45/24.77 → 均值 **19.92%**（旧 17.53%）；全量正确性由 embedded `python perf_opt/{op}.py --level all` 给出 51/51（L0 4/4 + contract / L1 26/26 / L2 12/12 / Boundary 9/9）
  - 与既有条目的关系：VP-2026-0100（report 告警 ≠ 失败，看 run.json summary）与 VP-2026-0185（`captured_op_count != 1` 判 invalid）讲**读数有效性**；本条讲**生成阶段的时间预算与缺失 run.json 的分阶段证据**，三者不重复
- repro: none（流程规则；复现条件 = 任一按 shape 重编译的 kernel（varlen / per-shape 分派）跑官方单算子 report，记录 correctness 进度 N/M、是否生成 run.json、benchmark harness 单独取数的可行性）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-05
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（「第二步：运行单算子 report」的编号列表第 3 项之后追加一个 blockquote）
    定位锚: "3. **性能报告**：从本次 `run.json` 读取 benchmark 结论与数值，**只记录不修复**。"
    new 文本（锚行之后插入）:
    > **按 shape 重编译形态的时间预算与分阶段产物（2026-11-05 GQA-SW-Varlen R13 实证）**：correctness 对**按 shape 逐个重编译**的 kernel（varlen / per-shape 分派）成本随用例数线性增长——GQA-SW-Varlen 的 26 个 correctness 用例在 40 min 内只完成 8 个（8/8 PASS，未生成 `run.json`）。此类形态在 report 前先按「单 shape 冷编译时间 × 用例数」估算预算；超出预算时分阶段记录：① correctness 进度与逐例结果取自 CLI stdout（`N/M PASS`），未跑完的用例明确标注未覆盖，不得写成通过；② 性能数据可由官方 benchmark harness（`examples/TileOPs/benchmarks/ops/bench_{op_slug}.py` + `msprof`）单独获取，并在报告中标注来源为 benchmark harness 而非 `run.json`；③ `run.json` 缺失仍按本文件既有规则返回 `[INTEGRATE_FAIL]`/不可用，不伪造、不从退出码推断；④ embedded `--level all`（如 51/51）可作为全量正确性的替代证据，但须与 benchmark 数据分列。
    动机: 本轮 correctness 全 PASS 但只跑到 8/26 且无 run.json——「没跑完」与「跑完失败」是两种结论，现行规则只区分后者；同时性能数据有合法替代来源（官方 benchmark harness），不标注来源会造成 `run.json` 口径混淆。
- status: pending
- confirmations: -/-
- created_by: task TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z 2026-11-05
- confirmed_by: -
- decided_by: -
- decided_note: Tier 2 结构化 diff 提案，待人工批准后由 `mode=apply` 执行。与 VP-2026-0100 / VP-2026-0185（读数有效性）互补，建议同批审批。
