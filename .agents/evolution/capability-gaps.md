# 编译器能力缺口登记簿（Capability Gaps）

> 编译器层面自进化机制的登记簿。与 skill 层自进化（`queue.md` / pattern-library）平行：skill 层沉淀「如何在现有能力下绕」，本簿登记「npuir 缺什么能力」并驱动补齐。
>
> - **用途**：任务过程中识别出「当前 npuir 能力不足以实现所选算法」（无法实现 / 被迫换算法 / 触及性能上限）时，登记具体缺失能力、证据与绕法，跟踪复发次数与补齐状态。
> - **写入者**：任务各阶段 Agent（design 调研与 API 映射 / develop / optimize / review）在得出能力不足结论时**直接追加条目或为既有条目计数**；`tilelang-skill-evolver` 在任务终态蒸馏时查重、冲突消解与升级裁决（与 queue.md 的维护分工同口径）。
> - **证据规则**：与 AGENTS.md「Docs Auto Routing / Negative claims」同口径——须引用具体 `docs/` 路径与限制条款、`testing/`/`examples/` 佐证或复现命令；无法佐证时显式标注「未文档化假设 + 估算依据」。禁止凭记忆或 GPU 先验断言能力缺失。
> - **版本戳**：一切「能力缺失」结论绑定工具链版本（tilelang commit + CANN/设备版本）；工具链升级后相关条目自动待重验，不得沿用旧结论否决新设计。
> - **升级阈值**：同一缺口被 **≥2 个不同任务**独立识别 → `recurring`，**必须推动补齐**：产出能力补齐提案（`proposal` 字段），并在任务终态报告中向用户显式列出（与 Tier 1 的 2 次独立证据阈值同口径）。

## 字段速查

```yaml
gap_id: CG-{YYYY}-{NNNN}           # 唯一 ID，追加时按现有最大号递增
layer: Frontend API / TileLangIR pass / BishengIR / tladapter / runtime / codegen
capability: <缺失能力的具体描述：缺什么 API / 优化 / dtype / 布局 / 同步原语 / 性能上限，越具体越好>
blocked_algo: <受阻的算法/算法族与结论形态：无法实现 / 被迫换算法（换成什么）/ 性能上限（差多少）>
evidence:                          # 证据链，规则同负向断言
  - <docs 路径#限制条款 / 复现命令 / 源码位置 / pattern-library 条目 / 实测数据>
workaround: <当前任务的实际绕法及其代价量化（如"两遍算法替代单遍，访存 +1.8x"），或 none（被迫弃用）>
occurrences: {n}                   # 独立任务识别次数（同一任务反复命中只计 1）
tasks: [<task_id>, ...]
toolchain_stamp: <tilelang commit + CANN 版本 + 设备，或 latest-rebuild>
status: open / recurring / proposal / in-progress / fixed / closed / withdrawn
proposal: <达 recurring 后必填：目标层 + 建议改动（API 签名 / pass 行为）+ 收益量化（引用各任务实测）+ 受影响算子清单 + issue 草稿（[npuir] 前缀）>
created_by: <task_id + 日期>
last_seen: <最近一次识别的任务与日期>
```

## 生命周期

```
open ──(≥2 独立任务识别)──> recurring ──(提案产出)──> proposal ──> in-progress ──> fixed ──> closed
  │                                                                            │
  └── withdrawn（证据被证伪/复核不成立）            fixed 后曾绕行的任务回访评估去绕化 ┘
```

- **追加即计数**：新识别的缺口先查重（本簿 + pattern-library §2），命中既有条目则 `occurrences+1` 并更新 `last_seen`/`tasks`，不新建重复条目。
- **推动补齐的形式**：提案正文落在条目 `proposal` 字段；对外建 issue（标题带 `[npuir]` 前缀，流程见 tilelang-github-operations skill）**前须用户批准**。
- **fixed 判定**：新工具链实测确认能力可用（附版本戳与验证记录）；fixed 后记录过 workaround 的任务可回访评估去绕化收益。
- **与 pattern-library 的边界**：绕法经验（怎么绕、绕的代价、陷阱）按 D 类进 pattern-library §1/§2；本簿只登记缺口本体与补齐跟踪，两处以 `gap_id` / 条目路径互相引用，不复制内容。

---

## Open

### CG-2026-0004

```yaml
gap_id: CG-2026-0004
layer: tladapter / codegen (load_nd2nz 区域描述符下推)
capability: >-
  T.load_nd2nz 的 src 区域若跨步非连续（BSHD 布局 [B,S,H,D] 张量按固定头取 (S,D)
  tile：区域 [1, real_m, 1, dim]，dim1 步长 H*D），后端按"从基址起的平坦连续内存"
  读取并静默产出确定性错误数据——不报错、不告警，数值表现为整块 S 矩阵乱值。
  即 load_nd2nz 仅支持张量布局的尾二维连续 tile（docs/Tilelang.language/内存操作/
  T.load_nd2nz.md §2.2.2 "src支持2-4D tensor" 未标注连续性约束）。同形态的
  T.copy（slice 写法）对相同跨步区域 bit-exact 正确，可作为绕法。
blocked_algo: >-
  BSHD 布局 attention 的 Cube 侧直接分块装载（q/k/v 按 (seq, dim) tile per head）；
  设计层按文档字面 "2-4D tensor" 假定跨步 tile 可用（_gqa_prefill_fwd_kernel
  DESIGN.md E2/E7 直连装载形态），实现期实测证伪，被迫换 T.copy slice 形态承载
  全部 GM→L1 装载（结构不变、语义等价、性能路径相同 PIPE_MTE2）。
evidence:
  - 复现：pattern-library repro/TRAP-load-nd2nz-strided.py（知识域，slice 绕法
    bit-exact 断言）；原 session 探针 probe_l1slot.py 为 provenance（session-local，
    flat-read 假设数值逐点复现 got[i,j] == flat_q[i]·flat_k[j]）
  - docs/Tilelang.language/内存操作/T.load_nd2nz.md §2.4 与
    docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.4 全部示例均为尾二维
    连续 tile（(b,n,s,d) 布局的 (s, block_d)）
  - examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py L83 的 4D 基址形态
    实为 dims 2-3 连续 tile（[heads, full_dim]），非跨步
workaround: >-
  GM→L1 全部改用 slice 形态 T.copy（highperf C1L/C2L 先例形态），跨步区域
    bit-exact 正确；代价：无额外拷贝（同为 PIPE_MTE2 单次搬运），仅依赖
    T.copy 而非文档指定的 load_nd2nz 分形通路（NZ 转换由后端隐式处理，实测
    gemm 结果正确）。
occurrences: 1
tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z]
toolchain_stamp: tilelang dev root build 2026-09-07 + CANN 8.5.0 + Ascend910B2C
status: open
created_by: multi_head_attention/_gqa_prefill_fwd_kernel Stage 3 attempt 3, 2026-09-07
last_seen: 2026-09-07
```

### CG-2026-0003

```yaml
gap_id: CG-2026-0003
layer: TileLangIR pass (Expert 模式 fixpipe lowering) 与 BishengIR 版本配对
capability: >-
  新版 tilelang 前端（机器默认激活构建 /home/tilelang/j00919838/tilelang-mlir-ascend，
  2026-09-02 期）在 Expert 模式下把 L0C→GM 的 T.copy lowering 为 hivm.hir.fixpipe 并
  携带 dma_mode = #hivm.dma_mode<nz2nd> 属性；本机全部 3 个可用 bishengir-compile 均
  无法编译该 IR：CANN 8.5.0 版（0.1.0，e4e2ba9841d1，2026-01-16，PATH 默认）与 upload
  仓库 3rdparty 版（19.1.7，31f690369d，2026-07-24）解析期报 "unknown attribute
  `dma_mode` in dialect `hivm`"；j00919838 配套 3rdparty 版（1.1.0，139bb6c7aef6，
  2026-09-02）解析通过但在 ConvertHFusionToHIVMPass 段错误
  （mlir::utils::getAnnotateOpWithAttr，HFusionToHIVM.cpp）。
blocked_algo: >-
  Expert 模式示例（examples/flash_attention/flash_attn_npuir.py，T.rs("PIPE_*") +
  T.sync_block_set/wait 跨核同步结构）在机器默认激活的 tilelang 构建下无法编译，
  迫使三实现性能对比任务整体切换到 upload 仓库的旧版前端构建完成。
evidence:
  - 复现（默认工具链）：PYTHONPATH 默认（tilelang 解析到 j00919838 checkout）下
    python flash_attn_npuir.py --seq_len 512 → bishengir-compile 报
    loc(kernel.npuir:41:42): error: unknown attribute `dma_mode` in dialect `hivm`；
  - 三编译器交叉实测（同一份 .npuir）：CANN 8.5.0 与 upload 3rdparty（2026-07-24）
    报同一 parse 错误；j00919838 3rdparty 1.1.0 在 ConvertHFusionToHIVMPass 崩溃
    （LLVM stack dump，栈顶 getAnnotateOpWithAttr）；
  - 绕法实测：PYTHONPATH 前置 upload 仓库构建后（其前端不产出 dma_mode），同一源码
    用 CANN 8.5.0 bishengir-compile 编译运行通过（含精度自检 All check passed，
    seq_len=512/4096）。
  - 未文档化假设：docs/ 未找到 bishengir-compile 与前端产物的版本配对说明；估计依据
    为本任务三编译器交叉实测。
workaround: >-
  PYTHONPATH 前置 upload 仓库 tilelang 构建（旧版前端，fixpipe 不带 dma_mode），
  三种实现（Expert npuir / Developer baseline / Developer gqa）统一在其下编译对比；
  代价：任务级工具链钉版本，机器默认 tilelang 构建的 Expert 路径完全不可用。
occurrences: 1
tasks: [FA 三实现性能对比 baseline/npuir/gqa 2026-09-07]
toolchain_stamp: >-
  tilelang 默认激活构建 j00919838 checkout（jit_npu.py dacebc3d 2026-08-05，3rdparty
  bishengir 1.1.0 2026-09-02）+ CANN 8.5.0 bishengir-compile 0.1.0（2026-01-16）+
  910B2C(IT21HMDB01-B2) + torch_npu 2.7.1
status: open
created_by: FA 3-impl perf comparison 2026-09-07
last_seen: FA 3-impl perf comparison 2026-09-07
```

---

### CG-2026-0002 → 已升级 recurring，条目移入下方 Recurring 区（2026-09-20 occurrences 2：Expert T.copy L1→L1 第二形态独立识别）

### CG-2026-0001

```yaml
gap_id: CG-2026-0001
layer: codegen (target.build.tilelang_npuir_dev, Developer mode)
capability: >-
  条件构造在 T.Pipelined 循环上下文中使 Developer 模式 codegen 崩溃（SIGSEGV，无诊断输出）：
  (a) Pipelined 体内直接出现运行时（TIR 条件）if 语句——即使条件为平凡比较
  （if k_idx == 0:）也复现；(b) T.Parallel（位于 Pipelined 体内）的循环体中出现
  比较式的 `&` 合取或 T.if_then_else 表达式。简单单比较的语句 if/else 与嵌套
  单比较 if 在 T.Parallel 体内可正常编译运行。
blocked_algo: >-
  flash-attention 类 kernel 的块级快速路径（运行时条件跳过 mask 预填，DESIGN R6/R-3）
  与多条件 mask 的紧凑写法不可用——被迫回退为逐块无条件 mask 预填（源 GPU kernel
  同构路径，正确性无虞，mask 预填从约 6-25% 块升至 100% 块，性能损失记入 tradeoff）。
evidence:
  - 复现（本任务 Stage 3；examples/multi_head_attention/_gqa_prefill_fwd_kernel 任务工作区未上库，provenance 允许失效）：
    变体探针 /tmp 之外已固化于 history_version 与 debug_log.md 的探针矩阵：
    E1 简单单比较 if（编译通过）/ E2 `&` 合取（SIGSEGV exit 139）/ E3 局部变量+单比较 if
    （通过）/ E4 T.if_then_else（SIGSEGV）/ E7 嵌套单比较 if×3（通过+精度通过）/
    variant_D `if k_idx == 0:` 直接位于 Pipelined 体（SIGSEGV）。
  - 崩溃栈：faulthandler 显示 segfault 于 tilelang/engine/lower.py:180
    device_codegen -> target.build.tilelang_npuir_dev（C++ FFI）。
  - 过时先例：examples/deepseek_v4/example_sparse_attn_kernel.py L71 `if n == 0:`
    位于 T.Pipelined 体内（当前工具链不可编译，疑似未在 CI 中执行）。
  - 同族补充证据（同任务后续用例）：
    (c) T.Parallel 体内对 shared buffer 的谓词标量写（if j >= real_n: v_shared[j,d]=0）
    使 BishengIR root-alloc pass 报错（'tensor.insert_slice'/'hivm.hir.vcast' op
    Unsupported op for finding the root alloc）→ 编译失败（seq_len 16 探针）；
    (d) T.Parallel 体内条件读 fragment（if acc_s==−inf: … else: acc_s=tmp_s2[i,j]，
    softcap −inf 保持）编译通过但运行期 fixpipe 路径损坏 → aicore 设备崩溃
    （"The MPU address access is invalid", fixp_error，softcap 探针）。
  - 未文档化假设：docs/Tilelang.language/ 无 T.Pipelined/T.Parallel 条件构造限制条款
    （遍历 同步管道操作/数学操作 目录未见相关文档）；估计依据为本任务探针实测。
  - Stage 4 同任务补充（2026-09-07，tuning 阶段）：(e) 跨块计算重叠结构
    （gemm1(j+1) 与块 j 的 v-chain 重叠，需 2 个 acc_s fragment 按迭代奇偶交替，
    运行时奇偶条件选择 fragment / 变量下标缓冲）同属本崩溃类，未敢实施
    （运行时条件 fragment 访问 = (d) 的已知崩溃形态）；评估记录于
    perf_opt/opt_log.md Round 7 blocked 候选。
workaround: >-
  mask 用嵌套单比较运行时 if/else 语句（sparse_mla_fwd.py L119-123 模式）重写；
  块级快速路径按 DESIGN §9.2 R-3 预授权回退放弃（无条件 mask 预填，源同构）。
  (c) 的绕法：shared buffer 清零改用无条件整 buffer T.vbrc（sparse_attn L81 模式）；
  (d) 的绕法：softcap −inf 保持改为纯向量算术等价变换
  kept = capped + (acc_s − clamp(acc_s))（有限值加 0 精确、−inf 加 −inf 保持），
  且 tanh 前先 clamp 使输入恒有限（不依赖硬件 tanh(±inf)）。
  Stage 4 可用 vselect/vcmp 广播构造替代（R-4 优化路径，需先例确认）。
  Stage 4 实测更新（perf_opt/opt_log.md Round 1）：无需 vselect/vcmp——mask 消除的
  正确形态是「工厂期 trace-time 分支特化」（clean_path = not is_causal and
  Sq%bm==0 and Skv%bn==0 为纯 Python 常量条件，静态消解，不产生运行时 if），
  non-causal 整除 shape 的 mask 预填整体删除 + gemm initC=True，Task Duration
  -73~75%（123.7→16.1us @fa512 等 4 case），位级等价（0+QK == QK）。
  NK==1 退化（single_block）同理特化（alpha=e^-inf=0 恒等，bit-exact）。
  Expert 模式结构级绕法（2026-09-07 expert 重开发任务实证，
  multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z）：双 Scope
  （Cube/Vector）+ GM workspace 多槽 + per-slot flag + T.rs 流内运行时 if 为合法
  形态，(a)/(b)/(e) 类阻塞结构在 Expert 下全部可表达——expert 重写后长 KV
  1.57–1.63× vs developer 基线（pattern-library §1.7 / §4 expert attention
  案例行）；Developer 阻塞时先评估 Expert 形态（queue VP-2026-0013）。
occurrences: 1
tasks: [multi_head_attention/_gqa_prefill_fwd_kernel Stage 3 first_impl]
toolchain_stamp: tilelang 67db6f3 (2026-09-04) + CANN 26.0.rc1 + 910B2C(IT21HMDB01-B2) + torch_npu 2.7.1
status: open
created_by: multi_head_attention/_gqa_prefill_fwd_kernel s3 attempt1 2026-09-04
last_seen: multi_head_attention/_gqa_prefill_fwd_kernel s4 tune 2026-09-07
```

### CG-2026-0005

> **第三轮注记（2026-09-09，task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）**：不阻塞目标达成——两相位重构（DESIGN §11.3）已在该缺口存在下四硬目标全部达成（12.94/18.11/30.88/98.05µs）；本条目描述的是两相位结构自身的 ~99µs 量级边界，非达标障碍。

```yaml
gap_id: CG-2026-0005
layer: Frontend API / codegen
capability: >-
  L0C（Cube 侧 gemm 累加器）到 Vector 侧 UB/L1 无直接搬运通路：T.store_fixpipe 的
  dst 仅允许 GM（docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1「目标
  张量（必须来自 GM 地址空间）」），Cube→Vector 数据必须经 GM workspace 往返
  （fixpipe 写 GM + 对端 MTE2 读回）。跨引擎协作算子中该往返流量与算法中间量
  同阶（attention 的 S/P 矩阵 S²×dtype×2 方向），且占用 L1/L2 fabric 带宽。
blocked_algo: >-
  attention 族跨引擎流水（Cube 双 gemm + Vector softmax）的 S/P 传输：性能上限
  ——fa4096 (S=4096,D=128) S/P w+r = 128MB 固定流量（f16 双向），是 2.27x 硬目标
  缺口的第二大构成项（详见 examples/TileOPs/tileops/kernels/attention/ 任务工作区
  内 perf_opt/perf_feedback.md——未上库，provenance 允许失效；结论已自包含于本条）。
evidence:
  - docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1（dst 仅 GM 的文档限定）
  - perf_opt/profiles/round8/recheck_fa4096（Memory CSV：cube 读写 ws 流量实测）
  - perf_opt/opt_log.md 第二轮 round 8 Diagnosis 第 5 条（流量算术）
workaround: >-
  S/P 传输 dtype 减半（f16，round 5 裁决 + round 9 f16 softmax 链，合计 -3%~-14%
  wall）；无通路级绕法。
occurrences: 1
  tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z]
toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
status: open
created_by: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
```

### CG-2026-0006

> **第三轮注记（2026-09-09，task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）**：不阻塞目标达成——两相位重构（DESIGN §11.3）已在该缺口存在下四硬目标全部达成（12.94/18.11/30.88/98.05µs）；本条目描述的是两相位结构自身的 ~99µs 量级边界，非达标障碍。

```yaml
gap_id: CG-2026-0006
layer: TileLangIR pass / codegen（向量算子融合与 f16 打包发射）
capability: >-
  向量算子按独立指令发射、无相邻融合（vcast/vmul/vsub/vexp/reduce 链每 op ~0.5µs
  固定发射开销主导），且 f16 与 f32 逐元素速率相同（无 f16 打包/SIMD 加速）——
  向量链时间 ∝ op 数 × 元素数，与 dtype 字节宽无关。
blocked_algo: >-
  窄 tile 向量链（[22,256] 级 softmax pass）：同元素总量下块数翻倍使向量时间翻倍
  ——fa4096 深流水变体（bn=256）vec-active 90→180µs、wall +12% 回退（vs bn=512），
  封死"窄块深流水"结构方向。
evidence:
  - perf_opt/profiles/round10/v10dp_fa4096_bm44_bn256_ns2（aiv_vec_ratio 70.7% 实测）
    vs perf_opt/profiles/round9/v9_fa4096（39.5%，同元素量 bn=512）
  - perf_opt/opt_log.md 第二轮 round 10「回归根因」段（发射开销定律推导）
workaround: >-
  宽块 op 优先（bn=512 摊减发射次数）；删除纯 dtype 搬运 pass（f16 链消 vcast×2）。
occurrences: 1
  tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z]
toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
status: open
created_by: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
```

### CG-2026-0007

> **第三轮注记（2026-09-09，task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）**：不阻塞目标达成——两相位重构（DESIGN §11.3）已在该缺口存在下四硬目标全部达成（12.94/18.11/30.88/98.05µs）；本条目描述的是两相位结构自身的 ~99µs 量级边界，非达标障碍。

```yaml
gap_id: CG-2026-0007
layer: runtime / codegen（跨引擎同步原语粒度）
capability: >-
  T.sync_block_wait 表现为阻塞整条后续发射流（非窄管道门控）：流水化只有在 wait
  到达时必然已满足（深 deferral）才有效，而深 deferral 要求更多 L1 槽位（与
  cbuf 512KB 容量冲突）；且 per-block flag id 预算上限 15（DESIGN §7.2），多槽
  （≥16）workspace 协议无法表达。缺少「per-pipe 门控 wait」与「计数器/批量
  信号量」原语。
blocked_algo: >-
  跨引擎块级深流水（attention C1/V1/C2/V2 重叠）：实测 ns=1≈ns=2（±0.5%）、
  wide≈serial、defer-2 深流水在 bn=512 因 L1 槽容量（K/V 双槽 567KB>512KB）不可
  行、bn=256 因 CG-2026-0006 回退——busy 核 aic_scalar 40–58% flag 自旋成为
  结构性时延地板。
evidence:
  - perf_opt/opt_log.md 第一轮 round 3/4（ns 平区）+ 第二轮 round 10（v10_dp 回退）
  - perf_opt/profiles/round10/v10dp_*（aic_scalar 50.7→40.5%：深 deferral 部分
    解锁发射流的直接证据）
  - /tmp/opencode/probe_kvbw.py 探针四模式（session-local；结论镜像于 opt_log）
workaround: >-
  深流水仅在 wait 预满足深度可行（本轮 v10_dp 证实方向有效但被 L1/向量上限抵消）；
  宽块串行为实证最优折衷。
occurrences: 1
  tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z]
toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
status: open
created_by: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
```

### CG-2026-0009

```yaml
gap_id: CG-2026-0009
layer: runtime / codegen（MTE2/MTE3 引擎边界同 buffer 访问顺序原语）
capability: >-
  同一 UB buffer 跨 DMA 引擎复用时（MTE2 装载写 + MTE3 store 读），指令流内无原语
  表达「MTE3 读完成前不发射 MTE2 写」的 WAR 顺序约束：跨引擎（Cube↔Vector）的
  T.sync_block_set/wait flag 机制不覆盖该边界；自等自身 flag 在双 AIV gather 语义下
  有死锁风险；重复 buffer（第二份 ND staging）是唯一可靠绕法但受 UB 容量约束
  （attention 第六轮实证：(80,256) 贴限下 +20KB 即爆 192KB/AIV）。与 CG-2026-0007
  （跨引擎同步原语粒度）同属同步原语覆盖缺口族——本条在 MTE2↔MTE3 引擎边界，
  0007 在 Cube↔Vector flag wait 粒度。
blocked_algo: >-
  两相位 persistent 结构 Vector 侧深度 2 任务流水（p1(T+1) 提前后其 S load〔MTE2 写
  ub_f16_ND〕与 p1(T) 尾块 P store〔MTE3 读同 buffer〕背靠背）——flaky aicore
  timeout（AIVector+MTE error，L0 单跑 PASS / --level all 挂）。该形态是 35% roofline
  Ratio 目标的主要结构缺口（short 域 per-task 固定成本 8.05µs 中引擎 busy 仅 ~3.5µs，
  跨引擎 flag 依赖链 + 引擎边界顺序约束主导），第六轮判定 blocked。
evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：opt_log
    Round 8 r8g（精度失败两连：legacy 域单 flag-id 空间下 advance 消费 task T 的
    O-ready 计数读未写数据 L1 全挂 + pipe2 域 flaky timeout；根因归因与三条修复
    路径评估）+ perf_records.jsonl round 8 留档
  - pattern-library/attention.md PL-1.12「Vec 侧同构改造 blocked」段（引擎独占
    buffer 为流水深度 >1 的隐含前提；r7f 中该 WAR 被 p2/epi 时间距离掩盖的对照）
  - 未文档化假设：docs/Tilelang.language/ 无 MTE2/MTE3 引擎边界同 buffer 顺序约束
    条款；估计依据为本任务 flaky timeout 实测 + 排除法（时间距离掩盖 / 双缓冲
    UB 不可行 / flag 自等死锁风险）
workaround: >-
  保持单深度形态（r8h 终版：p2/epi 的时间距离天然掩盖该 WAR）；流水深度 >1 要求
  「引擎独占 buffer」（每引擎专属 staging，代价 = UB 需求翻倍）。
occurrences: 1
tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z]
toolchain_stamp: tilelang 0.1.2+a13585dc / CANN 8.5.0 / Ascend910B2C / 2026-09-16
status: open
created_by: multi_head_attention/_gqa_prefill_fwd_kernel 第六轮 Stage 4（distill 登记），2026-09-16
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 第六轮 Stage 4（2026-09-16）
```

### CG-2026-0013

```yaml
gap_id: CG-2026-0013
layer: Frontend API（TILELANG_ASCEND_MODE 作用域）
capability: >-
  编程模式（Expert / Developer）只能经进程级环境变量 TILELANG_ASCEND_MODE 设置，
  无 per-kernel / per-jit 作用域参数：三处实时读 os.environ——
  tilelang/language/customize_npuir.py:886（前端 trace，Scope/alloc 形态分派）、
  tilelang/engine/lower.py:170（device_codegen）、tilelang/jit/jit_npu.py:1789
  （--disable-hivm-tensor-compile 旗标）。docs/developer/EnvironmentVariables.md L31
  仅登记环境变量一种设置方式，docs/Tilelang.language/ 无 per-kernel 模式 API
  （全文无 ASCEND_MODE 条目）。后果：同一进程混编两种模式的 kernel 套件时，
  后编译者被先入者锁定的模式污染。
blocked_algo: >-
  单 pytest 进程跑混合模式算子套件（TileOPs tests/ops/：mha=Expert 两相位
  Cube/Vector 结构 vs ada_layer_norm/mish/logsumexp=Developer 向量链）——Developer
  赢则 Expert kernel 报 'hivm.hir.store' op only support copy gm to ub or copy ub
  to gm or copy ub to ub（mha full-fwd-bf16 变体更在 lower.py:180 device_codegen
  原生段错误）；Expert 赢则 Developer kernel 报同类 store 约束错（44 case 失败）。
evidence:
  - 复现（tilelang 0.1.2+1990aa9 / CANN 8.5.0 / Ascend910B2C，2026-09-18）：
    TILELANG_ASCEND_MODE=Developer pytest tests/ops/ → mha 失败+段错误（core dump）；
    TILELANG_ASCEND_MODE=Expert pytest tests/ops/ → 44 failed / 22 passed
    （ada/mish/logsumexp 全挂，mha 过）；不设环境变量 → import 顺序决定胜负
    （pytest 参数形式〔目录 vs 整文件 vs node ID〕改变 import 顺序，两次同参运行
    结果可反转——ada 先 import 则 mha 挂，mha 先 import 则 ada/mish 挂）。
  - tileops 侧 import 竞争现场：kernel 模块级 os.environ.setdefault
    （_gqa_prefill_fwd_kernel.py:92/126=Expert；_ada_layer_norm_kernel.py:86/130、
    mish perf_opt/mish.py:57、logsumexp×2=Developer），setdefault=先 import 者赢。
  - 源码三读取点（上文 layer 字段）；docs/developer/EnvironmentVariables.md L31。
workaround: >-
  tileops 级：kernel 模块 import 时 setdefault（现状，单算子进程正确、混合进程
  按 import 顺序随机胜负）；或每次 compile 前后显式 set/restore 环境变量（待实施，
  代价 = 每个 kernel 工厂包一层 env 作用域）。测试级绕法：按模式拆分 pytest 调用。
  根治需 tilelang 提供 per-kernel 模式参数（如 tilelang.jit(..., mode="Expert")）。
occurrences: 1
tasks: [tileops-tests-ops-mode-battle-20260918]
toolchain_stamp: tilelang 0.1.2+1990aa9(452f447 build) / CANN 8.5.0 / Ascend910B2C / 2026-09-18
status: open
created_by: tileops tests/ops/ 混合模式失败排查（conductor 会话），2026-09-18
last_seen: tileops-tests-ops-mode-battle-20260918（2026-09-18）
```

### CG-2026-0010

```yaml
gap_id: CG-2026-0010
layer: codegen (Developer mode, persistent + gemm 混排)
capability: >-
  Developer 模式（auto CV-split）对 persistent 分核结构（T.Kernel(N, is_npu=True)
  核内 T.serial 多任务）与 T.gemm + v-prefix 向量 op 的混排不支持：运行时崩溃
  "Illegal instruction, which is usually caused by unaligned UUB addresses"
  （vector core exception，retCode=0x31），无编译期诊断。Expert 模式（显式双
  Scope）同结构完全正常。docs 无 Developer/Expert 适用范围条款（未文档化假设：
  依据双模式 repro + 全仓 24 处 is_npu=True 用例扫描——Developer 仅用于非
  persistent 网格）。
blocked_algo: >-
  persistent 混合算子（Mamba-2 SSD chunk scan 类：Cube 双 gemm 路径 + Vector
  因子链）在 Developer 模式不可实现——本任务 user_requirement 指定 Developer
  模式，被迫切换 Expert（GQA/sparse_mla 先例）完成。
evidence:
  - 复现：pattern-library repro/TRAP-DEVMODE-PERSIST-GEMM.py（知识域，Expert
    数值断言 + Developer 崩溃双模式对照）；原件
    examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/repro/DEVMODE_PERSIST_CRASH.py（溯源，任务工作区允许失效）
  - 仓库实态：Developer 仅 non-persistent 先例（flash_attn_npuir_dev.py /
    fp8_lighting_indexer.py），persistent 混合算子全部 Expert（GQA /
    sparse_mla_fwd_exp.py）
  - 任务档案：examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/RETROSPECTIVE.md（溯源，允许失效）——Stage 3 章节（崩溃日志 + 模式切换链）
workaround: >-
  Expert 模式显式双 Scope（T.Scope("Cube")/("Vector") + alloc_L1/alloc_L0C/
  alloc_ub + sync_block_set/wait + pass_configs 关闭
  TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION）——完全绕过、无性能代价
  （pattern-library attention.md PL-1.16 / traps-runtime.md
  TRAP-DEVMODE-PERSIST-GEMM）。
occurrences: 1
tasks: [ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z]
toolchain_stamp: tilelang 0.1.2+1990aa9fe4 + CANN 8.5.0 + Ascend910B2C / 2026-09-17
status: open
created_by: ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel Stage 3（distill 登记），2026-09-17
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 3（2026-09-17）
```

### CG-2026-0011

```yaml
gap_id: CG-2026-0011
layer: Frontend API（Cube gemm dst 形态 / L0C 累加原语）
capability: >-
  T.gemm 的 dst 不支持 L0C region 写（单 [Q,bp] 大 acc 与 per-lt band gemm 的
  [64,64] 小 dst 不可共用同一 L0C 张量的不同区域），且无「两 L0C acc 相加」
  原语——跨 lt 的累加结果无法在 L0C 域合并，必须经 GM/UB 中转。（未文档化
  假设：docs/Tilelang.language/线性代数操作/T.gemm.md 无 region dst 条款，
  依据本任务优化候选的结构分析。）
blocked_algo: >-
  SSD chunk scan history 路径的 M=256 合并 gemm（1 读 1 gemm/任务，消除
  per-lt gemm 碎片）不可实现——保持 per-lt [64,64] band gemm 分块形态，
  gemm 指令数 4/任务无法进一步合并（该候选列为 blocked，opt_log §5）。
evidence:
  - examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/opt_log.md §5（溯源，允许失效）——blocked 候选清单（history M=256 合并 gemm 行：API 阻塞点 + dst 形态分析）
workaround: >-
  保持 per-lt band gemm 分块（每 lt 单条 gemm，ws 写侧块连续布局配合——
  elementwise.md PL-1.14 两全形态）；性能缺口由深度 2 任务流水与 AIV 分片
  在其他维度补偿（终局 2.91×）。
occurrences: 1
tasks: [ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z]
toolchain_stamp: tilelang 0.1.2+1990aa9fe4 + CANN 8.5.0 + Ascend910B2C / 2026-09-17
status: open
created_by: ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel Stage 4（distill 登记），2026-09-17
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 4（2026-09-17）
```

### CG-2026-0012

```yaml
gap_id: CG-2026-0012
layer: BishengIR（auto-multi-buffer pass，动态 subview 支配性）
capability: >-
  task 级装载的 UB 行 buffer（如 [1,Q] 因子行）在内层循环以动态偏移切片消费
  时，--enable-auto-multi-buffer=true 产出非支配 IR：error: operand #2 does
  not dominate this use（Q≥128 切片非全宽触发；Q=64 全宽不触发）——合法的
  UB→UB 动态 extent copy 形态被 pass 破坏支配性。
blocked_algo: >-
  「task 级 dA/dt 行装载」优化（省 12 次 tiny GM 读/任务）不可实现——被迫
  退回 per-block GM 直载（基线形态）；该候选与 UB 容量候选同列 blocked
  （opt_log §5，v3 分支双重 blocked 实证）。
evidence:
  - examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/logs/round1/v3_hoist_L0.log（溯源，允许失效——报错全文）
  - pattern-library traps-compiler.md TRAP-UB-dynsubview-dominance（形态与
    绕法三则 + 知识域 repro/PL-1.13 关联骨架）
workaround: >-
  ① per-block GM 直载；② 切片偏移静态化（trace 分派）；③ 关
  auto-multi-buffer 未验证（全局改变 L1/UB 多缓冲行为，风险自担）。
occurrences: 1
tasks: [ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z]
toolchain_stamp: tilelang 0.1.2+1990aa9fe4 + CANN 8.5.0 + Ascend910B2C / 2026-09-17
status: open
created_by: ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel Stage 4（distill 登记），2026-09-17
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 4（2026-09-17）
```

### CG-2026-0014

```yaml
gap_id: CG-2026-0014
layer: TileLangIR pass（向量化/分块）
capability: >-
  multitile（num_tiles>1 的 T.serial tile 循环）+ bm>1 组合触发编译器 SIGSEGV
  （rc 139，无报错文本）。触发矩阵：bm=1 任意 tile 数 ✓ / bm>1 单 tile ✓ /
  bm>1 多 tile ✗。本条目按 DESIGN.md §9.4-R9 补录（前次任务已用该 ID 登记
  但条目丢失；源 repro .py 已清理仅存 repro/__pycache__ pyc）。
blocked_algo: >-
  argmax/_argreduce_kernel tiled 路径被迫 bm=1（DESIGN §5.2 强制）——lm-head 类
  小 M 负载核利用率被钉死在 M/48（4/48），是 C6 N-split 结构（本库 PL-1.20）
  的存在动因；Stage 4 实测翻转后 lm-head 2.20x/2.60x。
evidence:
  - repro/__pycache__/TRAP-multitile-bm-gt1-segfault.cpython-311.pyc（前次任务，
    2026-09-23；触发矩阵见 DESIGN.md examples/argmax/_argreduce_kernel §9.1-C-5）。
  - DESIGN.md §9.1-C-5 / §9.2-R3（本任务引用其结论未复现崩溃本身）。
workaround: tiled 路径强制 bm=1（工厂期保证）；小 M 大 N 负载改 N-split 两段 kernel。
occurrences: 1  # 前次任务识别；本链 Stage 1（DESIGN §9.4-R9）发现登记丢失并补录、Stage 4 引用未复现崩溃本身
tasks: [argmax-_argreduce_kernel-stage3-前次任务, argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel DESIGN §9.4 补录（2026-09-28）
last_seen: argmax-_argreduce_kernel-20260924T031005Z（2026-09-28）
```

### CG-2026-0015

```yaml
gap_id: CG-2026-0015
layer: TileLangIR pass（npu_loop_vectorize）
capability: >-
  2D T.Parallel(bm,N) 融合 if_then_else 循环在 bm>=2 时整体标量化（scalar ratio
  0.84–0.95），与操作数形态无关（(bm,1) 直接索引 / ext_brc 物化 + fragment staging
  两变体同标量化）；bm=1 同形态正常向量化。argmax hidden-state 实测 222µs（标量化）
  vs 链式 47µs（4.7x 差距）。
blocked_algo: >-
  row-reduction 族的融合候选/条件选择类 2D 循环在 bm>=2 不可用——被迫退到显式
  向量 op 链（vbrc/vcmp/vselect，pass 数 +2）或 per-row (1,W) 展开循环（op 数
  x bm）。跨任务影响：任何「逐元素条件 + 多行 block」形态。
evidence:
  - pattern-library TRAP-parallel2d-bm-ge2-scalarize（三组操作数变体对照 + msprof
    数据 + repro/repro_parallel2d_bm2_scalarize.py，ratio 3.02 实测复现）。
  - argmax opt_log Iteration 4 证伪更正记录（session-1 嫌疑 (a)/(b) 均被变体对照否证）。
workaround: bm>=2 用显式向量 op 链或 per-row (1,W) 展开（PL-1.20 形态）。
occurrences: 1
tasks: [argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel Stage 4 round 4（2026-09-28）
last_seen: argmax-_argreduce_kernel-20260924T031005Z（2026-09-28）
```

### CG-2026-0016

```yaml
gap_id: CG-2026-0016
layer: TileLangIR pass（向量化/codegen）
capability: >-
  T.arange(buf (M,W), [0,1], 0) 的索引物化按标量执行（~1ns/elem；((2,4096) 实测
  8.56µs）。(1,N) 形态正常。另：核内重复 T.arange 写同 buffer 非幂等（放大探针
  误编译，见 repro_arange_scalar.py docstring）。
blocked_algo: >-
  需要 j 索引载体的融合/链式候选构造在 bm>=2 时被迫 hoist 一份 (bm,N) fp32 缓冲
  并付出标量物化税（hidden-state 47µs 中占 ~18%）。
evidence:
  - pattern-library CONST-arange-scalar-materialize（msprof 46.97→38.41µs，
  -18.2%；节省量 8.56µs 与 8192 elem × ~1.05ns 模型吻合）。
workaround: T.arange 于 (1,N) + vbrc 首轴广播 (1,N)→(M,N)（文档合法形态）；
  bm=1 直接用（同形 vbrc 触发 CG-2026-0017）。
occurrences: 1
tasks: [argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel Stage 4 round 4（2026-09-28）
last_seen: argmax-_argreduce_kernel-20260924T031005Z（2026-09-28）
```

### CG-2026-0017

```yaml
gap_id: CG-2026-0017
layer: BishengIR（vbrc verifier）
capability: >-
  T.vbrc 同形 src/dst（无尺寸差维）产出空 broadcast_dims 数组，MLIR verify 报
  "'hivm.hir.vbrc' op have empty broadcast dims array"——语义上应等价于逐元素
  拷贝或前端拒绝（更清晰的报错）。
blocked_algo: 轻微：同形广播场景需改用源缓冲或 T.copy。
evidence:
  - pattern-library TRAP-vbrc-same-shape-empty-broadcast（stderr 诊断原文 +
  repro/repro_vbrc_same_shape.py 断言复现）。
workaround: 同形场景直接用源缓冲 / T.copy。
occurrences: 1
tasks: [argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel Stage 4 round 4（2026-09-28）
last_seen: argmax-_argreduce_kernel-20260924T031005Z（2026-09-28）
```

### CG-2026-0018

```yaml
gap_id: CG-2026-0018
layer: BishengIR / TileLangIR（hivm.hir.vxor verifier）
capability: >-
  T.vxor 的标量操作数不受支持：hivm.hir.vxor 要求两个操作数均为 shaped（variadic
  of shaped），标量立即数（如 T.cast(1,"int8")）报 "operand #1 must be variadic of
  shaped..."；同族一元 vnot 无此问题。
blocked_algo: 轻微：位运算「标量常量第二操作数」写法需物化为同形常量 buffer。
evidence:
  - 复现：examples/bitwise_not/bitwise_not_fwd_kernel/repro/TRAP-npuir-vxor-scalar-operand.py
    （NPU 跑通）；形态 = bitwise_not 的 bool 分支 `T.vxor(x_ub, 1, y_ub)` → bishengir-compile
    报 operand #1 非 shaped。
workaround: 用 `T.fill(c_ub, T.cast(1,"int8"))` 物化同形常量 buffer 后再 `T.vxor(x_ub, c_ub, y_ub)`（bit-exact 等价）。
occurrences: 1
tasks: [bitwise_not-bitwise_not_fwd_kernel-20261013]
toolchain_stamp:tilelang 0.1.2 dev-root（AscendNPU-IR 31f6903）+ CANN 9.0.0（bishengir 1.1.0 + hivmc 0.2.0）+ Ascend910B3 / 2026-10-16
status: open
created_by: bitwise_not/_bitwise_not_fwd_kernel Stage 3（2026-10-13）
last_seen: bitwise_not-bitwise_not_fwd_kernel-20261013（2026-10-13）
```

### CG-2026-0019

```yaml
gap_id: CG-2026-0019
layer: tladapter / runtime（JitKernel_NPU dtype 映射）
capability: >-
  JitKernel_NPU._tvm_dtype_to_torch（tilelang/engine/jit_npu.py ~1696）缺 uint8 键，
  自动分配的 uint8 输出被回退映射为 float32 → uint8 算子输出 dtype 错误。
blocked_algo: 中等：uint8 输出的逐元素算子无法直接得到正确输出 dtype。
evidence:
  - 复现：examples/bitwise_not/bitwise_not_fwd_kernel/repro/TRAP-npuir-uint8-output-dtype-map.py
    （NPU 跑通）；来源 tilelang/engine/jit_npu.py:1696 映射表无 uint8。
workaround: bool/uint8 走 int8 字节存储（bit-identical）并在 host 侧 view 回转（输入 view int8、输出 int8→bool/uint8 view）。
occurrences: 1
tasks: [bitwise_not-bitwise_not_fwd_kernel-20261013]
toolchain_stamp:tilelang 0.1.2 dev-root（AscendNPU-IR 31f6903）+ CANN 9.0.0（bishengir 1.1.0 + hivmc 0.2.0）+ Ascend910B3 / 2026-10-16
status: open
created_by: bitwise_not/_bitwise_not_fwd_kernel Stage 3（2026-10-13）
last_seen: bitwise_not-bitwise_not_fwd_kernel-20261013（2026-10-13）
```

### CG-2026-0020

```yaml
gap_id: CG-2026-0020
layer: Frontend API / TileLangIR（python AST → TVM parser）
capability: >-
  条件分支内的 T.alloc_ub（如 `if cond: buf = T.alloc_ub(...)`）被 TVM parser 拒绝
  （alloc 必须无条件/顶层），不允许按 trace-time 条件分配 UB buffer。
blocked_algo: 轻微：条件性临时 buffer 需改为无条件分配（浪费少量 UB）。
evidence:
  - 复现：examples/bitwise_not/bitwise_not_fwd_kernel/RETROSPECTIVE.md（Stage 3；bool 常量 buffer
    条件分配被 parser 拒 → 改无条件分配）。
workaround: 无条件分配该 buffer（即便部分路径不用）。
occurrences: 1
tasks: [bitwise_not-bitwise_not_fwd_kernel-20261013]
toolchain_stamp: tilelang 0.1.2+417dae9908 / CANN 8.5.2 / Ascend910B3
status: open
created_by: bitwise_not/_bitwise_not_fwd_kernel Stage 3（2026-10-13）
last_seen: bitwise_not-bitwise_not_fwd_kernel-20261013（2026-10-13）
```

### CG-2026-0021

```yaml
gap_id: CG-2026-0021
layer: BishengIR / hivm codegen（VerfCodegen）
capability: >-
  T.verf / T.erf 在本工具链上不是真正的 erf：CodeGenTileLangNPUIRAPI::VerfCodegen
  （tilelang-mlir-ascend/src/target/codegen_npuir_api.cc ~2433）把 tl.npuir_verf
  lowering 为「4 项 Maclaurin 截断序列」(2/√π)(x − x³/3 + x⁵/10 − x⁷/42)，仅 |x|≲1
  有效；T.erf（ErfOp→hivm::VErfOp）走同一 codegen，无精确 erf 回退。
blocked_algo: 全域 erf 消费类算子（erf-GELU / erfc / normal CDF）无法用原生 API；被迫换算法（显式多项式）。
evidence:
  - 复现：examples/gelu_and_mul/gelu_and_mul_fwd_kernel/repro/TRAP-verf-taylor-truncation.py（自包含断言，NPU 跑通）。
  - 实测：verf(1.877)=0.056 vs erf=0.992；verf(3.6)=−155.8；verf(±inf)=NaN。官方测试仅采样 (−1,1) @1e-2，掩盖该截断。
  - 文档缺口：docs/Tilelang.language/数学操作/T.verf.md 只述存在/范围/dtype，未述 lowering 与其有效定义域——doc-vs-toolchain 语义缺口。
workaround: 显式 clamped deg-20 minimax 多项式（fp32，c0=1/A 精确饱和 + NaN mask），RANGE 全范围 max abs err 2.98e-7；代价 ~35 额外向量 op。
occurrences: 1
tasks: [gelu_and_mul-gelu_and_mul_fwd_kernel-20261013]
toolchain_stamp:tilelang 0.1.2 dev-root（AscendNPU-IR 31f6903）+ CANN 9.0.0（bishengir 1.1.0 + hivmc 0.2.0）+ Ascend910B3 / 2026-10-16
status: open
created_by: gelu_and_mul/_gelu_and_mul_fwd_kernel Stage 3 `[DESIGN_ERROR]`（2026-10-13）
last_seen: gelu_and_mul-gelu_and_mul_fwd_kernel-20261013（2026-10-13）
```

### CG-2026-0022

```yaml
gap_id: CG-2026-0022
layer: tladapter / runtime（torch_npu F.gelu approximation）
capability: >-
  torch_npu 2.7.1.post4 的 torch.nn.functional.gelu(approximate="none") 在设备侧
  返回 tanh 近似（非 erf 精确），偏差 ~1.2e-3。
blocked_algo: 设备侧精确 GELU 参考不可用（fp32 容差 1e-5 下不可比）；测试参考须改走 CPU 显式 erf。
evidence:
  - 实测（examples/TileOPs）：kernel vs NPU F.gelu=1.20e-3；NPU F.gelu vs exact=1.20e-3；NPU F.gelu("none") vs CPU F.gelu("tanh")=2.38e-7。
  - 修复：examples/TileOPs/tests/ops/test_gelu_and_mul.py ref_program 改 CPU 显式 erf（0.5*gate*(1+erf(gate/√2))）。
workaround: 测试参考在 CPU 上用显式 erf 形式计算（不依赖设备 F.gelu）。
occurrences: 1
tasks: [gelu_and_mul-gelu_and_mul_fwd_kernel-20261013]
toolchain_stamp: torch_npu 2.7.1.post4 + torch 2.7.1+cpu + Ascend910B3
status: open
created_by: gelu_and_mul Stage 5 集成（2026-10-13）
last_seen: gelu_and_mul-gelu_and_mul_fwd_kernel-20261013（2026-10-13）
```

### CG-2026-0023

```yaml
gap_id: CG-2026-0023
layer: BishengIR / hivm codegen（Vtanh）
capability: >-
  T.vtanh 在本工具链非忠实且按 TILELANG_ASCEND_MODE 分派两条失败路径：
  默认/Expert = 截断 Maclaurin 级数 (x − x³/3 + 2x⁵/15 − 17x⁷/315)，仅 |x|≲0.5 可用；
  Developer = exp 恒等式 (e^{2x}−1)/(e^{2x}+1)，|x|≲44 忠实但 |x|>~44 / +inf 溢出为 NaN。
  T.tanh 仅为 TIR 标量表达式、无向量形态。
blocked_algo: 需要全范围忠实 tanh 的向量算子（tanh-GELU 等）无法用原生向量 API；被迫换算法（恒等折叠）。
evidence:
  - 复现：examples/gelu_tanh_and_mul/gelu_tanh_and_mul_fwd_kernel/DESIGN.md §1.6.1 `counter npuir-vtanh`（截断发散复现）；codegen_npuir_api.cc Vtanh 源码。
  - 实测（设计期硬件探针）：显式 sigmoid 机制 fp32 1.9e-7 / fp16 9.4e-4 / bf16 6.8e-3。
workaround: 恒等折叠 gelu_tanh(g)=g·σ(2u)（`1+tanh(u)=2σ(2u)`）+ 显式 fp32 sigmoid（`vexp`/`vdiv`），RANGE max abs err 1.781e-7。
occurrences: 1
tasks: [gelu_tanh_and_mul-gelu_tanh_and_mul_fwd_kernel-20261013]
toolchain_stamp:tilelang 0.1.2 dev-root（AscendNPU-IR 31f6903）+ CANN 9.0.0（bishengir 1.1.0 + hivmc 0.2.0）+ Ascend910B3 / 2026-10-16
status: open
created_by: gelu_tanh_and_mul Stage 1 设计调研（2026-10-13）
last_seen: gelu_tanh_and_mul-gelu_tanh_and_mul_fwd_kernel-20261013（2026-10-13）
```

### CG-2026-0024

```yaml
gap_id: CG-2026-0024
layer: Frontend API / BishengIR（npuir_cmp）
capability: >-
  T.vcmp 无 unordered 模态：`npuir_cmp` 只接受 eq/ne/lt/gt/ge/le，且 "ne" 按**有序**
  not-equal（ne ≡ lt∨gt）lower → 自比较 `vcmp(x,x,"ne")` 对 NaN 为 **false**，无法检测 NaN。
blocked_algo: NaN 谓词（isnan 等）不能用 `vcmp "ne"` 自比较；必须 `NOT(x<=x)`（有序 `le` + 取反）。
evidence:
  - 判别性复现：examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py（直接读 mask：ne(NaN)=0、le(NaN)=0）。
  - 推翻 pattern-library 常数 CONST-vcmp-ne-nan-unordered（原 repro 非判别性——hardsigmoid 算术链自身传播 NaN）。
  - docs/Tilelang.language/比较操作/T.vcmp.md（无 unordered 说明）。
workaround: `isnan(x)=NOT(x<=x)`：`vcmp(x,x,tmp,"le")` → `vcast(i1→f16)` → `vsub(1.0,·)`（fp16 域取反）→ `vcast(f16→i8)` → host view(bool)。三 dtype 精确。
occurrences: 1
tasks: [isnan-_isnan_fwd_kernel-20261014]
toolchain_stamp:tilelang 0.1.2 dev-root（AscendNPU-IR 31f6903）+ CANN 9.0.0（bishengir 1.1.0 + hivmc 0.2.0）+ Ascend910B3 / 2026-10-16
status: open
created_by: isnan/_isnan_fwd_kernel Stage 3 `[DESIGN_ERROR]`（2026-10-14）
last_seen: isnan-_isnan_fwd_kernel-20261014（2026-10-14）
```

### CG-2026-0025

```yaml
gap_id: CG-2026-0025
layer: Reporting harness（tileops.reporting.cli + msprof 采集过滤）
capability: >-
  单算子集成报告（Stage 5 `python -m tileops.reporting.cli run --op <Op>
  --prof-mode msprof --kernel-name main`）对 npuir 多 kernel/多 Scope 算子的
  profile 匹配失败：① `--kernel-name main` 期望设备端 op 名形如
  `*_main_mix_aic`，而 npuir 生成的设备 op 名不含该后缀（e.g. conv1d 五个
  `_conv1d_*_kernel`），过滤命中为空 → 退化为无过滤采集；② 无过滤采集里混入
  wrapper 侧 workspace 分配算子（`ZerosLike` 等），单次 capture 含 2–3 个 op，
  逐 shape 判为 invalid（`invalid_profile_count=16`）→ `benchmark_passed=false`、
  run status=partial。kernel 本身正确（op 正确性 28/28、per-kernel L0 49/49），
  真实延迟可由保留的 `OpBasicInfo` CSV 恢复（conv1d: unit_stride≈5.5ms /
  general≈0.59–0.62ms），但报告不产出有效 ratio。
blocked_algo: 无法通过标准 `tileops.reporting.cli` 拿到 npuir 多-kernel 算子的
  有效 per-shape benchmark ratio（需人工从 CSV 恢复），阻塞性能门禁的自动化。
evidence:
  - examples/TileOPs/reports/tileops/20261014_174100_695290_Conv1dFwdOp/run.json
    （status=partial，benchmark_passed=false，invalid_profile_count=16，
    correctness 28/28）。
  - examples/TileOPs/profile_run_msprof_20261014_174257.log（无过滤重试 +
    captured_op_count=2–3，含 ZerosLike）。
  - 对照：单-kernel MixCV 算子（gqa/ssd）此前以同类 harness 亦见 profile 过滤
    对不齐（历史记录），本条目为多-kernel 分发场景的显式复现。
workaround: >-
  ① 采集时显式传入真实设备 op 名（或前缀通配）替代 `main`；② 将 wrapper 的
  workspace 分配算子排除出 profile 窗口（或在报告层按 op 名白名单过滤）；
  ③ 过渡期：从保留的 `OpBasicInfo` CSV 人工恢复 per-shape 延迟填报告。
occurrences: 1
tasks: [conv1d-Conv1dFwdOp Stage 5 integrate 20261014]
toolchain_stamp: tilelang 0.1.2 dev-root（417dae99）+ CANN 8.5.2 + Ascend910B3（20 AI Core）
status: open
created_by: conv1d/Conv1dFwdOp Stage 5 integrator（2026-10-14）
last_seen: conv1d-Conv1dFwdOp Stage 5（2026-10-14）
```

### CG-2026-0008 → 已升级 recurring，条目移入下方 Recurring 区（2026-09-15 occurrences 2）

---

### CG-2026-0026

```yaml
gap_id: CG-2026-0026
layer: BishengIR backend（Developer-mode T.gemm lowering）
capability: >-
  GDN（gated delta net）chunked prefill 的 Developer-mode 探针内核编译到
  bishengir backend 阶段崩溃：`fatal error: error in backend: ERROR: error
  pointer address space cast`（bisheng clang 退出码 70，`Failed to compile
  BiShengLIR for device`）。Npuir IR 前端已降级（`AscendNPU IR compile reported
  success` 之前的 `.npuir` 解析/opt 通过），失败发生在 hivmc/biSheng 后端指针
  地址空间转换。探针形态：Developer 模式（无 T.Scope）下于 `T.alloc_shared`
  buffer 上直接 `T.gemm(..., b_transpose=True / a_transpose=True)`，
  并含 `T.arange`/`T.cumsum`/`T.vcumsum`。核心 chunked delta-rule 数学
  （intra-chunk 逆、gate/beta 变换、跨 chunk 状态传递、输出合并）尚无可用
  NPU 落地。
blocked_algo: >-
  GatedDeltaNetPrefillFwdOp（基线 GDNFwdOp / kernels/linear_attention/gdn/
  prefill.py:29 GDNDensePrefillFwdKernel）无法以当前探针形态（Developer-mode
  shared-buffer T.gemm + 转置）在 npuir 上编译；需改为 Expert 模式
  （T.Scope("Cube") + T.alloc_L1/T.alloc_L0C + T.gemm）重构，或等待后端修复
  指针地址空间转换。
evidence:
  - examples/gdn_prefill/probe/gdn_prefill_probe.py（316 行，Neumann-series 逆 +
    chunked 前向；独立 sequential delta-rule golden）。
  - 实跑 `python gdn_prefill_probe.py`：3/3 case 均 bishengir-compile 失败，
    stderr `fatal error: error in backend: ERROR: error pointer address space
    cast`；`hivm.hir.vcumsum` 低效告警（非本轮报错点）。
  - 对照：Conv1d/Conv3d/GLADecode 均已迁移成功（Expert/Developer 均可用），
    唯一崩溃点为本探针的 Cube-on-shared + 转置组合。
workaround: >-
  ① 将 prefill 核改为 Expert 模式（显式 Cube Scope + L1/L0C，如 conv1d/conv3d
  先例），避免 Developer-mode shared-buffer T.gemm 转置路径；② 以 Cube-free
  （纯 Vector）或分阶段多 kernel（prepare/forward）重排；③ 过渡期：登记本 gap，
  GatedDeltaNetPrefillFwdOp 暂不交付（任务书 §6 批 4「不可行时只交付 plain 产物」
  口径）。
occurrences: 1
tasks: [gdn_prefill-GatedDeltaNetPrefillFwdOp Stage-0 probe 20261015]
toolchain_stamp: tilelang 0.1.2 dev-root（417dae99）+ CANN 8.5.2 + Ascend910B3
status: open
created_by: gdn_prefill/GatedDeltaNetPrefillFwdOp Stage-0 probe（2026-10-15）
last_seen: gdn_prefill/GatedDeltaNetPrefillFwdOp Stage-0 probe（2026-10-15）
```

## Recurring（达升级阈值，待推动补齐）

### CG-2026-0008

```yaml
gap_id: CG-2026-0008
layer: BishengIR（per-AIV UB 分配可见性）
capability: >-
  BiShengIR 的 per-AIV UB 实际分配与前端逐 buffer 显式预算之间存在未记载的基础差值
  （三次实测、双向偏差：28783f45 手工预算 178.2KB 实际 ~204.1KB +15%；6797758
  4 点标定 = 手工预算 × 1.10–1.12；a13585dc 显式 alloc 结构反例 0.981×〔actual 低于
  手工〕——系数结构依赖），且无逐 buffer 分配清单查询手段——编译报文只给
  总 requires bits，设计期手工预算不可静态核验，差值来源（疑 extra/sync buffers）
  无文档记载。与 CG-2026-0002（auto-multi-buffer 膨胀）不同：本差值在
  `bishengir-compile --enable-auto-multi-buffer=false` 下仍然存在（基础分配差）。
blocked_algo: >-
  宽 config 设计意图的性能上限形态：two-phase v3 的 bn_eff=256 宽块路径在
  dim=128 causal manifest 域（4 workload × fp16/bf16）不可编译（ub overflow 硬失败，
  首次编译发生在 Stage 5 bench，8/10 失败）——两相位 ws 物化流量的宽块摊销收益
  无法实现，被迫窄 config（bn_eff∈{64,144}）交付：短 KV 1.30–1.32x 提速 / 长 KV
  0.78x 回退（vs Sep-07 单遍基线，msprof kernel-only）。第五轮以 UB diet 削减
  −40.9KB 手工预算（设计妥协：N/D staging 合并 + rowmat 删除）绕开后，宽块收益
  才得以实现（长 KV 反转 11.1x 领先、全域几何 8.53x）——绕法本身证明该差值
  直接决定宽块设计可达性与结构选型。
evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Design-layer finding（溯源，任务工作区允许失效：DESIGN §4.5 预算 178.2KB ≤ 192KB "✓" vs 实际 ~204.1KB；报文 `ub overflow, requires 1666048–1672192 bits while 1572864 bits available`）
  - 双路径最小复现（工厂直调 (4,512,32,128) causal fp16）：(64,64,1)→E6 替换 (64,256) FAIL vs (64,64,2)→逐字 bn_eff∈{64,144} OK
  - dim=64 smoke ~161.7KB 可过（预算-实际差在窄变体不越界）
  - 第二证（task 20260915T080600Z 第五轮 Stage 4 Phase 1）：probe_ub.py 编译探针 4 点标定（6797758）——(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%（auto-multi-buffer=false 下仍在）+ 漏计小 buffer 实证（DESIGN §4.5 漏计 ub_cond2 8.2KB）
  - 第三数据点（task 20260916T033847Z 第六轮 r9 precision_fix，a13585dc，**不计 occurrences**——本轮阻塞项为分派守卫组合而非分配差，且实测方向相反系反例非阻塞）：显式 alloc 结构（全 UB buffer 首维 = half 的逐 buffer 显式分配）手工核算 214080B vs BishengIR 实际 210080B（ratio 0.981，actual 低于手工）——系数结构依赖的反例数据点，正负两向偏差均无文档，强化「逐 buffer 分配可见性」诉求；核算镜像 + pre-fix 溢出断言：pattern-library/repro/PL-1.12-bn-clamp-bm-guard.py（系数边界同步记入 constants.md CONST-capacity-910B2C / attention.md PL-1.12 r9 update）
  - 第四数据点（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 4，1990aa9fe4，**不计 occurrences**——无分配差阻塞，属显式 alloc 结构依赖的跨族确认）：Expert 直差分因子链（mamba 族 MixCV）手工峰值 ~179KB ≈ 192KB 上限 93%；v3 变体 +16KB 手工 → 实测 195.1KB 报错（偏差 <2%，显式 alloc ≈ 手工核算跨工具链跨族再证）；bl=128 编译实测需 582KB/357KB vs 192KB 容量否决（constants.md CONST-capacity 第三数据点）
workaround: >-
  ① 窄 config 逐字路径交付（Stage 3 L1 门禁验证过的 traced 变体）；② 设计期预算按
  ×1.10–1.12 系数放大（constants.md CONST-capacity-910B2C 已收录，勿漏计小 buffer）；
  ③ 预算表不作为可编译证明——宽 config 上靶前编译探针校准（报文 requires bits 反推
  实际需求，与 VP-2026-0046 探针步骤同口径）；④ 贴限设计走 UB diet 主动削减
  （第五轮实证 −40.9KB 解锁宽块）。互链 pattern-library TRAP-UB-multibuffer-inflation
  第四证（VP-2026-0059，2026-09-15 两证合入）。
occurrences: 2
tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z, multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z]
toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD），均 + CANN 8.5.0 + Ascend910B2C / 2026-09-15（跨 commit 复现）
status: recurring
proposal: >-
  目标层 BishengIR（per-AIV UB 分配可见性）。建议改动（二选一或组合）：① 编译期
  逐 buffer UB 分配清单输出（如 `bishengir-compile --dump-ub-allocation`：每 buffer
  名/尺寸/归属 AIV/额外分配项），使设计期手工预算可静态核验；② 至少将基础分配差
  文档化（差值构成〔疑 extra/sync buffers〕+ 推荐设计期补偿系数 ×1.10–1.12）。
  收益量化（两任务实测）：首证宽 config 不可编译致长 KV 0.78x 回退、bench 8/10
  失败被迫窄 config 交付；第二证需 −40.9KB 手工 diet（两次结构妥协）才解锁宽块，
  解锁后长 KV 11.1x / 几何 8.53x——分配差直接决定宽块设计可达性，可静态核验可
  消除「预算表 ≠ 可编译证明」盲区并避免 diet 类设计妥协。受影响算子：attention
  族宽块设计（两相位 causal 域实证）+ 一切 UB 贴限设计（ada_layer_norm nl 依赖
  形态同族，TRAP-UB 平台律条目）。issue 草稿（[npuir] 前缀，**须用户批准后建**）：
  [npuir] BishengIR per-AIV UB 分配差无文档、无逐 buffer 查询手段（≈手工预算
  ×1.10–1.12，auto-multi-buffer=false 仍在）。
created_by: multi_head_attention/_gqa_prefill_fwd_kernel Stage 5 集成期发现（distill 登记），2026-09-15
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 第六轮 r9 反例数据点精化（2026-09-16，不计 occurrences——本轮流变系守卫组合而非分配差阻塞，见 evidence 末行）
```

> **升级依据**：task 20260915T025507Z（Stage 5 集成期发现，28783f45）+ task 20260915T080600Z（第五轮 Stage 4 编译探针重校，6797758）——两个不同任务跨 commit 独立识别并定量标定，达 ≥2 任务阈值。capability 补齐提案已产出（见条目 proposal 字段：逐 buffer 分配清单输出 / 差值文档化二选一），待向用户显式列出并推动（外部 issue 须用户批准后建）。

---

### CG-2026-0002

```yaml
gap_id: CG-2026-0002
layer: BishengIR (bishengir-compile, cbuf→cbuf copy)
capability: >-
  L1（cbuf）域内搬运方向整体不支持：'hivm.hir.copy' op Unsupported copy from
  cbuf to cbuf! → 编译失败。两种触发形态：① 首证（Developer）——Pipelined 循环
  （NK >= 2，触发 auto-multi-buffer 多缓冲）之前对 L1(shared) buffer 的整 buffer
  T.vbrc 广播清零被错误 lowering 为 cbuf-to-cbuf 拷贝（NK == 1 时同一构造可正常
  编译）；② 第二证（Expert，2026-09-20）——显式 T.copy(l1_a, l1_b) 的 L1→L1
  方向直接硬失败（最小 Expert 探针实证，非 lowering 路径）——缺口本体为
  cbuf→cbuf 搬运方向整体不支持，不限 vbrc lowering。
blocked_algo: >-
  ① 尾块防护类构造受限（首证）：多块 KV 流水 kernel 无法在循环前对 GEMM 输入
  L1 buffer 做一次性清零；被迫把清零守卫收缩到"仅单块且部分块"（seq_len_kv
  < block_n）场景，多块场景依赖有限残留值安全性论证绕过。② L1 域内重排/组装
  类构造受限（第二证）：band 组装「先读入小 buffer 再组装到大 buffer」两跳形态
  不可实现——被迫单 buffer GM→L1 列偏移直载形态（每 band 块一次 GM 读，无 L1
  域内中转）或 GM 中转。
evidence:
  - 复现（首证任务 Stage 3）：seq_len=520（NK=9，has_kv_tail=True，循环前
    T.vbrc(0, v_shared)）→ bishengir-compile 报 'hivm.hir.copy' op Unsupported
    copy from cbuf to cbuf!（npuir 162:30）+ 'hivm.hir.load' root-alloc 连锁报错；
    同一 kernel 在 seq_len=16（NK=1）下编译运行通过；去掉该 vbrc 后 520 编译运行
    精度全通过（原 session 探针 /tmp/opencode/probe，provenance session-local 允许失效）。
  - 未文档化假设（首证）：docs/Tilelang.language/广播类文档（vbrc/brc）无多缓冲
    交互限制条款；估计依据为本任务探针实测。
  - 复现（第二证，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z
    Stage 2 复检轮）：最小 Expert kernel（T.Kernel(1, is_npu=True) + T.Scope("Cube")
    + 两个 T.alloc_L1([64,64],"float16") + T.copy(l1_a, l1_b)）→ bishengir-compile
    err code 1 报同款 'Unsupported copy from cbuf to cbuf!'；三证合一——docs
    T.copy.md §2.3 搬运方向表无 L1→L1 方向（GM→UB/UB→GM/UB→UB/GM→L1/L0C→GM）、
    全仓 examples 零先例、探针实证（REVIEW.md 复检轮机械复核段；探针原件
    /tmp/opencode/probe transient，provenance 允许失效）。
workaround: >-
  ① 首证：清零守卫收缩为仅 seq_len_kv < block_n（单部分块）时执行（NK=1 不触发
  多缓冲）；多块尾块场景依据有限残留值安全性免清零。② 第二证：L1 域内「重排/
  组装」意图落为单 buffer GM→L1 列偏移 dst 直载（T.copy(ws[...,0:bl,0:ts],
  l1_band[0:bl,s0:s0+ts])，src/dst 双侧尾块裁剪）+ 单条 band gemm——探针 2 在
  4515de8 编译通过（与 TRAP-L1-band-dst-tail-overrun 联防）；或经 GM 中转。
  绕法知识条目化提案见 queue VP-2026-0095（pattern-library traps-compiler.md）。
occurrences: 2
tasks: [multi_head_attention/_gqa_prefill_fwd_kernel Stage 3 first_impl 2026-09-04, ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 2026-09-20]
toolchain_stamp: 首证 tilelang 67db6f3 (2026-09-04) + CANN 26.0.rc1 + 910B2C(IT21HMDB01-B2) + torch_npu 2.7.1；第二证 tilelang 0.1.2+4515de8 + CANN 8.5.0 + Ascend910B2C / 2026-09-20
status: recurring
proposal: >-
  目标层 BishengIR / Frontend API。建议改动（二选一或组合）：① 支持 cbuf→cbuf
  copy lowering（L1 域内搬运原语，消除「域内重排必经 GM」约束）；② 至少将该
  方向缺口文档化——T.copy.md §2.3 搬运方向表显式标注 L1→L1 不支持 + 报错信息
  指向推荐形态（单 buffer GM→L1 列偏移直载）。收益量化（两任务实测）：首证使
  尾块清零守卫收缩（多块场景裸奔依赖残留值论证）；第二证使 band 组装类结构
  被 forced 单 buffer 直载形态（每 band 块一次 GM 读）——L1 域内中转可消该类
  GM 重读。受影响算子：L1 域内重排/组装类（MixCV band 组装、多块流水清零守卫）。
  issue 草稿（[npuir] 前缀，须用户批准后建）：[npuir] cbuf→cbuf copy 不支持且
  无文档（L1 域内重排/清零构造不可表达，vbrc lowering 与显式 T.copy 双路径同报错）。
created_by: multi_head_attention/_gqa_prefill_fwd_kernel s3 attempt1 2026-09-04
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 复检轮（2026-09-20）
```

> **升级依据**：task multi_head_attention/_gqa_prefill_fwd_kernel Stage 3（2026-09-04，Developer vbrc lowering 路径）+ task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2（2026-09-20，Expert 显式 T.copy 路径）——两个不同任务、双触发形态、跨工具链（67db6f3/CANN 26.0.rc1 → 4515de8/CANN 8.5.0）独立识别同一缺口（cbuf→cbuf 搬运不支持），达 ≥2 任务阈值。proposal 字段已产出，待向用户显式列出并推动（外部 issue 须用户批准后建）。

---

## Resolved（proposal / in-progress / fixed / closed / withdrawn 归档）

（暂无条目）
