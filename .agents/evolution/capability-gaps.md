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
  小 M 负载的核利用率被限制在 M/48（如 4/48），形成单 launch 实现的性能上限。
evidence:
  - repro/__pycache__/TRAP-multitile-bm-gt1-segfault.cpython-311.pyc（前次任务，
    2026-09-23；触发矩阵见 DESIGN.md examples/argmax/_argreduce_kernel §9.1-C-5）。
  - DESIGN.md §9.1-C-5 / §9.2-R3（本任务引用其结论未复现崩溃本身）。
workaround: >-
  正确性绕法为 tiled 路径强制 bm=1（工厂期保证）。性能侧暂无满足源/目标
  device kernel launch 一一对应约束的通用绕法；应继续寻找同一次 launch 内的方案，
  无可行方案时记录为设计限制。
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

> **交叉引用（2026-10-31，不计 occurrences）**：task `fft_c2c_vec-fft_c2c_vec-20261031T103133Z` 的实测说明本条与「1 条 store 体」入口条件的**分界**——该任务用「每级 4 个独立 **1D** `T.Parallel(h)`、每体恰 1 条 `BufferStore`」成功产出向量 IR（`vadd/vsub/vmul/vdeinterleave`、`arith.mulf=0`），说明向量化入口的门槛是**体形态（单 store）+ 循环 kind（kParallel）**，2D 并行维（bm≥2）才是本条记录的独立触发器。两结论不冲突，本条口径无需修订；形态级判据与实测数据见 `cases.md` CASE-fft-c2c-vec-design-blocked 与 queue VP-2026-0163（Tier 1 pending，待第二证）。

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

### CG-2026-0018 → 已升级 recurring，条目移入下方 Recurring 区（2026-10-31 occurrences 2：设计层重做独立识别 + 量化上界修正为 ≤22.2%）

> **与 CG-2026-0022 的关系（登记簿一致性注）**：本算子迁移期（2026-10-15）产生的 CG-2026-0022 是**另一能力形态**——「Developer auto CV-split 下 Cube `T.gemm` GEMV 与 shared 向量规约 + 标量广播无法共存（BiShengIR 内存层级 verifier 三类拒绝）」。本条为跨核归约/栅栏缺失，两者并存不合并，本条不增量 0022 的 occurrences。
> **登记簿一致性发现**：CG-2026-0022 被仓内 6 处文档引用（本算子 DESIGN / integration_log / RETROSPECTIVE，engram_gate_conv_fwd DESIGN ×4），但本登记簿无该条目（现有最高号 CG-2026-0017）；同批死链还有 `CASE-engram-decode-migration`（cases.md 无此 id）、`VP-2026-0283`（queue 无此 id；`CASE-engram-decode-stage4` 已于 2026-10-30 落盘）、`TRAP-npuir-vec-scalar-broadcast` / `TRAP-npuir-fp32-vector-reduce` / `TRAP-conditional-alloc` / `CONST-ascend910-9362-profile`（traps-compiler.md / constants.md 均无）、`TRAP-C13-parser-no-python-unroll`（2026-10-31 起由 traps-compiler.md 新增条目 TRAP-primfunc-static-branch-and-loop 覆盖同一现象）。推断成因：本工作区知识库最后变更日志停在 2026-09-28，10 月各任务（含本算子迁移）的蒸馏成果未落盘。恢复需原始任务证据 + 工具链重验，本轮不做；机械检测提案见 queue VP-2026-0148。

### CG-2026-0008 → 已升级 recurring，条目移入下方 Recurring 区（2026-09-15 occurrences 2）

### CG-2026-0019

```yaml
gap_id: CG-2026-0019
layer: Frontend API / codegen（dtype 域）
capability: >-
  npuir 无 complex dtype，且向量数学无 fp64 域：`T.vadd` 的 dtype 表不含 fp64、
  `T.deinterleave` 的 dtype 表 fp64 列为 ×（complex128 需 real/imag 两路 fp64 承载，
  两条路径均无基底）。complex64（两路 fp32）可用，complex128 不可实现。
blocked_algo: >-
  FFT/C2C 族 complex128 档无法交付（被迫跳过或 host 侧降档）；影响所有以 complex128
  为契约 dtype 的谱域算子。
evidence:
  - docs/Tilelang.language/数学操作/T.vadd.md §2.2.1 dtype 表（无 fp64）
  - docs/Tilelang.language/shape操作/T.deinterleave.md §2.2.1（fp64 列 ×）
  - examples/TileOPs/tileops/kernels/spectral/fft_c2c/fft_c2c_kernel/perf_opt/opt_log.md
    L28 / L137：`fft-4k-c128-b64-complex128` 以 `pytest.xfail` 跳过并标注 CG-2026-0019
    （无 perf / 无 precision 记录，在 workload_inventory.json 记 skipped）
  - examples/fft_c2c_vec/fft_c2c_vec/DESIGN.md §1.2（L79）以本条为 dtype 约束依据；
    REVIEW.md 维度 1 实核 fft L0 第 7 项显式 SKIP complex128
workaround: >-
  complex64 以 real/imag 两路 fp32 承载 + v-prefix 逐元素算子（本仓 FFT 两代实现的
  共同形态）；complex128 无绕法，按契约不支持并显式披露。
occurrences: 1
tasks: [TileOPs/FFTC2COp optimize（2026-10-16）, fft_c2c_vec-fft_c2c_vec-20261031T103133Z（引用方，同谱系不计独立识别）]
toolchain_stamp: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31；docs 条款核对同戳
status: open
created_by: 登记簿补登（2026-10-31 蒸馏时发现被两处工件引用但登记簿无此条目——死链归位，内容据 opt_log 实测与 docs 条款重建）
last_seen: fft_c2c_vec-fft_c2c_vec-20261031T103133Z（2026-10-31）
```

> **补登说明（2026-10-31）**：本条 ID 早于本次蒸馏被 `fft_c2c` 基线的 `perf_opt/opt_log.md`（2 处）与 `fft_c2c_vec` 的 `DESIGN.md` §1.2 引用，但登记簿缺条目（同批死链清单见 CG-2026-0018 上方的登记簿一致性注）。本次按被引用语义 + docs 条款 + opt_log 的 skip 记录重建条目，属**死链归位**而非新识别——故复杂的 FFT 谱系只计 1 次（首证为 optimize 任务的实测 skip），后续引用方不计 occurrences。

### CG-2026-0023

```yaml
gap_id: CG-2026-0023
layer: TileLangIR pass（UB 分配位置规划 / PlanAndUpdateBufferAllocationLocation）
capability: >-
  UB 规划的 liveness 复用只在单个「块」（循环嵌套）内部生效：一个缓冲的存活期一旦跨越
  两个块（典型形态 = 每级/每阶段一块独立 `T.alloc_ub`，级间串行推进），它就不与其他同尺寸
  缓冲合并。因此「每级独立缓冲 + 级间复用」类算法的真实峰值 = 所有跨级存活缓冲之和，
  而非 ping-pong 双缓冲量。实测 fft_c2c_vec（n 级 Stockham，每级一对 `(2,ng,h)`）：
  级内 4 个短命中转 11 份 → 1 份（复用生效），跨级级缓冲 11 对无一合并；
  n=2048 需 221 KB、n=4096 需 464 KB，均超 192 KB（`ub overflow`，编译期确定）。
blocked_algo: >-
  多级网络类算法（FFT/Stockham、多级 stencil、分阶段变几何 tiling）的「整行 UB 常驻 +
  级间覆盖」路线在 n ≥ 2048 不可交付；降级链同时封死——radix-4（288 KB）/ radix-8
  （224 KB）/ flat ping-pong（缓冲份数不减）三种备选均不成立。设计期若按「liveness 复用
  → 128 KB」估算，会在 Stage 3 首次大 n 编译时被判 BLOCKED_DESIGN。
evidence:
  - 编译报错原文 `ub overflow, requires 3802112 bits while 1572864 bits available`
    （464 KB vs 192 KB）；逐尺寸实测：n≤1024 通过、n=2048 221 KB 溢出、n=4096 464 KB 溢出
  - examples/fft_c2c_vec/fft_c2c_vec/RETROSPECTIVE.md Stage 3（DP 表第 4 行 + Transferable
    第 3 条）；DESIGN.md §4.5 与原 §9 R1 的复用假设（任务工作区，provenance 允许失效）
  - 与 CG-2026-0008（per-AIV 分配基础差）机制不同：该条是「手工预算 vs 实际分配」的偏差，
    本条是「复用策略的适用边界」；与 pattern-library TRAP-UB-multibuffer-inflation 的
    multi-buffer 膨胀链也不同（本条不涉及双缓冲）
workaround: >-
  n ≤ 1024 可用（实测全绿）；更大 n 需把级缓冲改为单份原地覆盖（本轮 flat ping-pong 未
  减份数，说明该形态在本 pass 下不成立）或改跨核/跨 launch 分解（未验证）。设计期容量
  判据：峰值 = Σ（跨级存活缓冲），不用 ping-pong 口径。
occurrences: 1
tasks: [fft_c2c_vec-fft_c2c_vec-20261031T103133Z]
toolchain_stamp: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31（Expert 模式）
status: open
created_by: fft_c2c_vec Stage 3（BLOCKED_DESIGN，2026-10-31）；evolver 终态蒸馏登记
last_seen: fft_c2c_vec-fft_c2c_vec-20261031T103133Z（2026-10-31）
```

### CG-2026-0024

```yaml
gap_id: CG-2026-0024
layer: codegen / tladapter（向量指令操作数地址约束）
capability: >-
  向量指令的段基址需 32 B 对齐，且操作数读取不得越过 UB 分配尾部；违反时**不报编译错误**，
  而是运行期 Vector core abort（`The write address of the MTE instruction is out of range`
  / `CCU instruction address check error` / `[vector core exception]`），进程被杀、无法在
  进程内捕获。前端无「自动对齐装载」或「非对齐段向量读」路径，也无编译期地址检查。
  实测触发形态：twiddle 查表按 `LUT[h-1+i]` 取段（段起点恒为奇数，4 B 粒度）+ 分配尺寸非
  32 B 倍数。
blocked_algo: >-
  以窄表/查表段作向量操作数（FFT twiddle、LUT 类广播、conv 核权重段）的算法路线在未显式
  补齐对齐时不可交付：编译通过、运行期 abort，调试成本高（症状不出现在编译期，且 abort
  杀进程）。对「直接沿用源算子 flat 表布局」的迁移与重设计影响最大。
evidence:
  - pattern-library traps-compiler.md TRAP-npuir-vector-lut-alignment
    （repro/TRAP-npuir-vector-lut-alignment.py：对齐段精确、非对齐子进程 abort）
  - examples/fft_c2c_vec/fft_c2c_vec/RETROSPECTIVE.md Stage 3 DP 表第 3 行 + Transferable 第 2 条；
    运行时 abort 原文见该条目（任务工作区，provenance 允许失效）
workaround: >-
  每段起始对齐到 32 B（8 个 fp32 元素）+ 表尾补齐到 32 B 倍数（本例 `npad ≥ n-1`），
  代价是表物理尺寸增大与实现相对设计的布局偏差（须披露）；对齐后同 kernel 数值精确。
occurrences: 1
tasks: [fft_c2c_vec-fft_c2c_vec-20261031T103133Z]
toolchain_stamp: tilelang 0.1.2+9738284 / CANN 9.0.0 / Ascend910_9362 / 2026-10-31（Expert 模式）
status: open
created_by: fft_c2c_vec Stage 3（BLOCKED_DESIGN 前实测，2026-10-31）；evolver 终态蒸馏登记（repro 同批转正）
last_seen: fft_c2c_vec-fft_c2c_vec-20261031T103133Z（2026-10-31）
```

> **编号说明**：本次登记用 0019（死链归位，语义已在两处工件被引用）与 0023/0024（顺位；0020–0022 号在 Recurring 区注记中被视为已占用/外部引用，不复用以免冲突）。

---

## Recurring（达升级阈值，待推动补齐）

### CG-2026-0025

```yaml
gap_id: CG-2026-0025
layer: Frontend API / TileLangIR verifier（hivm.hir.load 的 shape 与 pad_mode）
capability: >-
  `hivm.hir.load` 在 `pad_mode` 未设置时要求 src/dst **shape 完全相同**（verifier 原文：
  "'hivm.hir.load' op if pad_mode is not set, src and dst shape should be the same!"，
  BiShengIR pipeline 失败）；而 `pad_mode` 的取值域只有 `PadNull / PadFirstElem / PadValue`
  ——语义是「以首元素或指定值把短边补到长边」，不含「按运行时索引配对多段区域」或
  「一次 load 覆盖多行非连续区域」的语义，且该属性未由 tilelang 前端暴露。
  后果：同一 kernel 体内第二个「动态行 GM→on-chip」copy、或单次 load 的区域宽度超过
  目的 buffer 形状时，只能退回单路顺序 copy / 缩小块宽。
blocked_algo: >-
  ① 逐行动态索引 gather 的 k 路并行拆分（把一个 64 行循环拆成多条并行目的 buffer 以压低
  标量/同步占比）不可表达；② 由 UB 余量驱动的宽块路径在形状不一致处直接编译失败
  （如 epilogue bN=256）。两者都不是 tiling 参数可解，属前端/verifier 能力边界。
evidence:
  - 首证 task TileOPs-_sparse_mla_kernel-20261101T014649Z（perf_feedback.md「减少条数的三条路」第 ③ 条 +
    opt_log §4.1）：拆 k 路目的 buffer 时同 kernel 体内第二个动态行 copy 报同一 verifier 文案；
    任务内复核方言定义 `3rdparty/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td`
    L87（"If `pad_mode` is not set, `src` and `dst` shape should be the same"）与
    `HIVMAttrs.td` L302-308（PadModeEnum = PadNull / PadFirstElem / PadValue）
  - 第二证 task TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z（opt_log §3.2 blocked 证据原文
    + `perf_opt/logs/candidate/s4k_r1f_fp16_bn256.log`）：epilogue bN=256 采同一报文
    （`'hivm.hir.load' op if pad_mode is not set, src and dst shape should be the same!`
    + `Failed to run BiShengIR pipeline`）
  - 关联：TRAP-load-nd2nz-strided / TRAP-T-copy-region-semantics（前端侧区域语义规则）；
    CG-2026-0010（Developer persistent+gemm）是首证任务两相位绕法的另一阻塞点，与本条不同层
workaround: >-
  ① 单路顺序 copy + 缩小块宽（两任务的交付形态，性能代价已量化：首证 1.34 µs/行 ∝ 条数；
  第二证 bN 夹在 128）；② 形状对齐到同一 extent 后再 load（必要时显式补零），代价是额外
  流量与一次 reshape；③ 需要「多段配对搬运」的算法改走多次顺序 copy + 片上拼接（未验证性能）。
occurrences: 2
tasks: [TileOPs-_sparse_mla_kernel-20261101T014649Z, TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z]
toolchain_stamp: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3 / 2026-11-01（两任务同戳，Developer MixCV）
status: recurring
proposal: >-
  目标层 Frontend API / verifier 文档化。建议改动（择一或组合）：① tilelang 前端暴露
  `pad_mode`（含语义与合法取值）并在 verifier 报文中给出可操作修复指引，而非仅拒绝；
  ② 若「多段区域 load / 按索引配对」在本方言不打算支持，在 `T.copy` / `T.gather` 文档中
  明确列为 unsupported 并给出等价写法。收益量化：首证任务中该限制封死三条减条数路径之一
  （另两条为容量/地址随机，均已成死路），是 7.0–7.5% Ratio 天花板的一根支柱；第二证任务
  中 bN=256 本可把 tile 数再减半（bN 64→128 已实测 −48%）。受影响算子：稀疏 MLA / paged KV /
  attention-indexing 等一切逐行随机索引 + 宽块 epilogue 的 npuir 算子。issue 草稿
  （[npuir] 前缀，**须用户批准后建**）：[npuir] hivm.hir.load 的 pad_mode 未暴露且语义不含多段配对，
  逐行 gather 的 k 路拆分与宽块 epilogue 不可表达。
created_by: evolver 终态蒸馏登记（2026-11-01，两任务独立识别，登记即 recurring）
last_seen: TileOPs-_fp8_lightning_indexer_kernel-20261101T031511Z（2026-11-01）
```

> **升级依据（2026-11-01，登记即 recurring）**：两个不同任务（稀疏 MLA decode 的 k 路 gather 拆分失败 / FP8 indexer 的 bN=256 编译失败）独立命中同一 verifier 条款与同一报文，达 ≥2 任务阈值。proposal 已产出（含量化收益与受影响算子清单），**待向用户显式列出；外部 issue 须用户批准后建**。

### CG-2026-0026

```yaml
gap_id: CG-2026-0026
layer: Frontend API（tilelang.language 无 RNG 原语）
capability: >-
  npuir 前端没有任何随机数原语：`dir(T)` 无 `rng*` / `rand*` 属性，`docs/Tilelang.language/`
  全子目录无 RNG 条目，`tilelang/language/customize_npuir.py` 无 RNG 算子；BishengIR 侧仅有
  `linalg.fill_rng_2d`（`core_named_ops.py` L1814，counter-based LCG）但前端不可达。
  后果：一切随机算子（dropout / 采样 / 随机增强）必须在 kernel 内用 int32 向量元算自实现
  counter-based hash，并承担 hash+阈值段（dropout 11 op 中的 7 个）的固定成本。
blocked_algo: >-
  用硬件 RNG 原语替代软件 PRNG。dropout 实测：11 op 中 7 个为 hash+掩码生成；若前端 RNG 可用
  则收敛到 4 op（掩码转存储 dtype + scale·mask + x·(scale·mask)），Vector 段 ≈26.8→8.2 µs，
  即结构性加速 2.5–3.0x。当前有效 R+W 0.72–1.48 TB/s vs 同机逐元素天花板 2.49–2.67 TB/s
  （1.7–3.5×），差距全部落在该段，非 tiling/参数可解（`block_size` 已到 UB 上限 8192，
  10240 报 `ub overflow, requires 1802240 bits`）。
evidence:
  - 首证 task dropout-_make_dropout_kernel-20261016T193233Z（DESIGN.md §1.6.0 RNG 复核表：`dir(T)` /
    docs 16 子目录 / customize_npuir.py / core_named_ops.py 四条独立证据）
  - 复验 task TileOPs-_make_dropout_kernel-20261101T145102Z（新工具链 0.1.2+d5838899：11-op 终版仍
    有 7 个 hash+掩码 op；`perf_opt/perf_feedback.md` 修正附录 R6 直接以本条作为 `[DESIGN_LIMIT]`
    的收窄依据）
  - 登记簿一致性注（2026-11-02）：上述任务工件（含 dropout 的 DESIGN.md / perf_feedback.md /
    RETROSPECTIVE.md）以 **CG-2026-0021** 指代本条，但该 ID 在登记簿中从无条目，且被至少三个
    不同任务分别用于「RNG 原语缺失」（dropout）、「`T.verf` 截断」（gelu_tanh_and_mul）、
    「`T.vcmp` 不支持广播」（ge / clamp_max）三种互不相同的语义 ⇒ 旧 ID 不可复用。本条为 RNG
    语义的正式登记；另两个语义仍为死链，待各自任务补登（勿把本条 occurrences 计给它们）。
workaround: >-
  ① 核内 counter-based 无状态 hash（`f(seed, idx)` 纯函数，天然可重放、与线程布局无关）+
  算术 0/1 掩码乘（免 `vselect`）；② 4-op hash（mul,shl,xor,mul）为该族实测下界（3-op 统计不合格，
  见 queue VP-2026-0189）；③ p=0.5 时用符号位阈值再省 2 op（VP-2026-0187）。
occurrences: 2
tasks: [dropout-_make_dropout_kernel-20261016T193233Z, TileOPs-_make_dropout_kernel-20261101T145102Z]
toolchain_stamp: tilelang 0.1.2+d5838899 / CANN 9.0.0 / Ascend910B3 / 2026-11-01
status: recurring
proposal: >-
  目标层 Frontend API。建议改动：在 `tilelang.language` 暴露 counter-based RNG 原语
  （`T.rng_init(seed, counter)` / `T.rng_rand_int32` 或等价），或把 BishengIR 既有的
  `linalg.fill_rng_2d` 绑定到前端。量化收益：dropout 族 2.5–3.0x（11 op → 4 op，见 blocked_algo）；
  受影响算子：dropout 及一切随机掩码 / 采样类 npuir 算子（当前仓库内仅 dropout 一例，但属通用需求）。
  外部 issue 须用户批准后建（[npuir] 前缀）。
created_by: evolver 终态蒸馏登记（2026-11-02；同算子两轮任务独立识别，登记即 recurring；旧引 ID CG-2026-0021 见 evidence 一致性注）
last_seen: TileOPs-_make_dropout_kernel-20261101T145102Z（2026-11-01）
```

### CG-2026-0027

```yaml
gap_id: CG-2026-0027
layer: TileLangIR / codegen（hivm.hir.vcast 的 dtype 组合表）
capability: >-
  `hivm.hir.vcast` 没有 int32→fp16/bf16 的 lowering：四种写法（`round_mode` 取 rint /
  round / floor / 缺省）全部报
  `error: 'hivm.hir.vcast' op currently don't support cast int32_t_to_half_<mode>mode`。
  该限制与文档一致——`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 DataType
  支持表列了 i16→f16（round/rint/floor/ceil/trunc）、i1/i4/i8/ui8→f16 与 i32→f32，
  但不含 i32→f16 / i32→bf16 ⇒ 属于方言 dtype 组合缺失（不是 bug、也不是文档与实现不符）。
blocked_algo: >-
  「0/1 掩码直接 cast 到半精度存储域」不可行：int32 掩码必须经 int32→f32→fp16/bf16
  两跳，多 1 个向量 op 并强制保留一块 f32 中转缓冲。dropout fp16 路径因此停在 10 逻辑 op
  （11 物理）而非 9；（若该组合可用，掩码 cast 与 scale 乘仍是两个 op——`vbitcast` 不重绑
  dtype，见 queue VP-2026-0190）。
evidence:
  - task TileOPs-_make_dropout_kernel-20261102T065942Z（R9）`perf_opt/opt_log.md` §4.3 四次
    `T.vcast(h, md, round_mode=...)` 尝试表 + `perf_opt/logs/round9/bitexact_c.log`（原报错文本）
  - 文档条款：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1「DataType支持」表
    （i32 行只有 f32 / i64 / i16 三个 dst，无 f16 / bf16）
  - 关联（同族但不重复计数）：CG-2026-0026 是「无 RNG 原语」——本条是该族「掩码落存储域」
    那一步的独立缺口；pattern-library `TRAP-reduce-mixed-dtype-silent-corrupt` 只覆盖
    「混合 dtype 归约静默错值」，不覆盖 cast 组合缺失
  - 〔2026-11-03 编译器侧补齐尝试（源码级补丁 + 全链重链），结论：该方向双封〕
    探针（〔溯源，工作区路径允许失效〕）`examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/probes/vcast_i32_to_f16_probe.py`
    七用例实测：对照 i32→f32 rint 通过，i32→f16 五种 round_mode 与 i32→bf16 全失败，
    报错组合名精确为 `int32_t_to_half_{rint,round,floor,ceil}mode` / `int32_t_to_bfloat16_t_rintmode`。
    补丁草案与实施见同目录 `PATCH-vcast-i32-to-f16.diff`：白名单
    （`HIVMVectorOps.cpp` 的 `softSupportedCast`）+ 新增 `castSrcToF32ToDst`（经 f32 两跳）+
    `VCastLowering` 分支；已重链 `bishengir-compile` / `bishengir-opt` / `libtilelangir.so` /
    `libtilelang_module.so` / bundle 后的 `libBiShengIRAggregateCAPI.so`，五处均确认补丁字符串入二进制。
    **仍被拒**——真正的门禁在 CANN 预编译件：编译日志实际执行
    `Executing` → `/home/developer/Ascend/cann-9.0.0/bin/hivmc /tmp/bishengir-compile-*/module.hivm.opt.mlir`
    （解析后真身 `/home/developer/Ascend/cann-9.0.0/tools/bishengir/bin/hivmc`：含门禁字符串、
    无补丁），我们的 `bishengir-compile` 把末段 `.mlir→bin` 委派给该闭源件，
    而 `3rdparty/AscendNPU-IR/build` 中**无 `hivmc` 构建目标**（`ninja -t targets` 无命中）
    ⇒ 源码级补丁无法覆盖该层。**更关键**：即便打通，所实现的是 i32→f32→half 两跳分解，
    与 kernel 现有的手工两跳写法**完全等价**（0 个 op 的节省）；1 个 op 的收益只在存在
    原生 i32→half 单指令时成立，而 `HWSupportedCast` 白名单中无此组合（无硬件指令）。
    ⇒ 本条不再是「补丁可解」，而是「硬件无该指令 + 门禁在闭源件」双重阻塞；
    收益预期（8–9%）应作废，dropout 侧不要据此再排工。
admission_gates: >-
  〔2026-11-03 事后补记，按 `_shared/standards/compiler-change-admission.md` §4 要求〕
  G1 hardware: `HWSupportedCast` 白名单无 int32→half 组合（`Utils.h:393+` 表）+ 探针实测五种
  round_mode 与 bf16 全拒 ⇒ **无原生单指令**，收益为 0。结论=否。
  G2 source reachability: 真实执行件 binary_path=`/home/developer/Ascend/cann-9.0.0/tools/bishengir/bin/hivmc`
  （门禁字符串存在、无补丁）；build_target=**无**（`3rdparty/AscendNPU-IR/build` 内
  `ninja -t targets all` 无 hivmc 命中）⇒ 闭源件，源码补丁不可达。结论=否。
  G3 a_b_delta_us: **未实测**（本轮先动手后验证）——按 G1 结论（两跳分解 ≡ kernel 现有手工
  两跳）预期收益 0，无需实测。结论=否。
  G4 reuse/owner: 能力本身通用（掩码落半精度存储域的算子族受益），但落地依赖 CANN 侧
  hivmc ⇒ owner 不在本仓库。结论=否。
gate_verdict: >-
  **rejected（G1/G2/G3/G4 全否）→ 回落 S3 改写 kernel**；「省 1 op ≈ 8–9%」收益预期作废，
  dropout 侧不得据此排工。补丁与探针保留作证据（`perf_opt/probes/PATCH-vcast-i32-to-f16.diff`
  头部已记否决理由）。
workaround: >-
  int32→f32（`rint`）→ f32→fp16/bf16（`rint`）两跳；或把掩码生成域整体上移到 f32（阈值/
  符号位判定在 f32 一次 cast 完成），代价是 int32 段省下的 lane 数优势丢失。实测代价参考：
  R9 未能走通的这条路径本可再省 1 个 op（约占 10 逻辑 op 的 10%，按 int32 64 elem/instr
  的单价计约 8–9% 量级，未实测——该分支编译即被拒，无性能记录）。
  〔2026-11-03 二次否决，第三条独立证据〕**选择式替代路线同样封闭**——`T.vselect`/`T.vcmp`
  均可用（`tilelang/language/__init__.py:146,148`），但 ① `PL-1.29-vselect-cost`（verified）
  实测 `vselect ≈ 6.5×` 普通 v-op，用 1 次 select 换 2 次 vmul 是**变慢**；
  ② 契约不允许——golden 为乘性掩码（`torch.nn.functional.dropout`），丢弃位 `±inf` 必须
  得 NaN（`inf*0=NaN`），选择式返回精确 0（`_make_dropout_kernel_DESIGN.md` §0.5/§1.6.1
  已有该论证与实测）。⇒「省 1 个 op」在 **cast 补不齐** 与 **select 更贵且违约** 两侧均无
  可行实现；本条对 dropout 侧的收益结论**永久作废**，除非将来出现原生 i32→half 单指令
  （届时用 `perf_opt/probes/vcast_i32_to_f16_probe.py` 回归验证）。
occurrences: 1
tasks: [TileOPs-_make_dropout_kernel-20261102T065942Z]
toolchain_stamp: tilelang dev build `/mnt/workspace/tilelang-ascend-npu-ops/build`（R9 构建）+ CANN 9.0.0 / Ascend910B3 / 2026-11-02
status: open
proposal: >-
  未达 recurring 阈值（occurrences 1，需 ≥2 不同任务）。若同族再识别一次即升级并产出补齐
  提案：建议在 `hivm.hir.vcast` 增加 int32→fp16/bf16 的 rint 模式 lowering（掩码/计数类
  kernel 的高频组合），或在前端 `T.vcast` 文档表格中把「i32→f16/bf16 不支持」显式标为
  unsupported 并给出两跳等价写法（当前表格缺行，读表者只能靠试错发现）。
created_by: evolver 终态蒸馏登记（2026-11-02，R9 单任务识别）
last_seen: TileOPs-_make_dropout_kernel-20261102T065942Z（2026-11-02）
```

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

### CG-2026-0018

```yaml
gap_id: CG-2026-0018
layer: Frontend API / runtime（跨 AI Core 归约与完成栅栏）
capability: >-
  Developer 模式前端缺「跨 AI Core 归约 / 跨 block 完成栅栏」原语，三条候选路径均不可用：
  ① `T.sync_block_set` / `T.sync_block_wait` 语义限定为 Block 内部同步（同核 Cube↔Vector
  子块 FFTS lock），不承担跨 AI Core barrier；② GM workspace 跨核暂存无前端入口——IR 层
  存在 `#hacc.arg_type<workspace>` + `WorkspaceArgIdx` 形态，但只由 launcher 自动注入，
  Python 侧无 `T.alloc_*` 绑定（`TILELANG_ASCEND_WORKSPACE_SIZE` 仅改 size），属未文档化
  假设；③ `T.atomic_add` 到 GM 可用，但同一 launch 内没有完成栅栏，累加结果无法安全读回。
blocked_algo: >-
  grid=batch 的小 batch decode/GEMV 类算子无法在**单 launch 内** split-N / split-K（跨核拆分 +
  核内归约），核利用率被钉在 batch/40（AIV 40 核）→ roofline Ratio 成为结构性天花板而非实现问题。
  **【2026-10-31 订正，取代此前的 39–41% 外推】**此前记「若跨核归约免费，b4≈53%、b8≈50–57%、
  b1≈14%，tune 平均 ≈39–41%」——该外推**系统性偏高 ≈1.8×**，五项偏差来源已逐项定位（见
  evidence 末条）。本任务（设计层重做）给出的**路线无关**上界：把带宽与末阶段全部按免费，只保留
  任何实现都无法避免的「发射地板 FLOOR(n) + 总向量工作/(n × 每核实测吞吐)」并对块数 n ≤ 40
  取最优 ⇒ 下界 b1 3.08 / b4 8.90 / b8 5.93 µs ⇒ Ratio 上限 **9.92% / 32.04% / 24.51%，均值
  ≤22.2%**（假设边界：计算单元 = Vector〔Cube 路径被 CG-2026-0022 封死〕、向量项按本戳实测的
  mul+reduce 两遍、每档取最好实测吞吐 ⇒ 这是**乐观**下界）。即便假设 `mul+reduce` 融合成单遍
  （本工具链不支持），均值上限也仅 31.5%。**补齐本缺口（把 2 次发射压成 1 次）至多抹掉 B 段**，
  即回到该路线的 A 段天花板：split-d 15.7%（实测）/ split-K 16.2%（同公式模型）——仍远低于
  40% 目标。实测可兑现的最优形态（两阶段、以 kernel 边界为跨核同步）= 9.95%（3.472% → 2.87×）。
evidence:
  - docs/Tilelang.language/同步管道操作/T.sync_block_wait.md §1/§2.4：定义为 Block 内部同步，
    示例为同一 AI Core 内 Cube↔Vector 子块；flag id 预算有限（constants.md CONST-flag-id-budget ≤15/核）
  - docs/Tilelang.language/原子操作/T.atomic_add.md §2.2.2（「原子加对形状无特殊限制」）/
    §2.3（「特殊限制说明」为「无」）：文档只定义逐点原子加法语义，未记载任何跨 block
    完成/栅栏条款——同 launch 内读回累加结果的安全性属未文档化假设（估算依据：文档无此语义）
  - docs/Tilelang算子调试指南.md §3.2.1：IR 侧 `#hacc.arg_type<workspace>` + `WorkspaceArgIdx`
    由 launcher 注入；tilelang/jit/jit_npu.py L795–885 即该注入实现（`workspace_size` →
    `allocate_workspace` / `at::empty` → launcher 传参），Python 侧无对应 alloc 绑定
    —— **未文档化假设（无前端 API），估算依据 = 源码阅读 + 探针**（任务 opt_log 记录的
    L1449-1510 / L1593-1606 行号与本快照不符，已按实查行号更正）
  - 实测（task TileOPs-_engram_decode_kernel-20261030T121905Z Stage 4，probe/h1_probe.py，msprof op 口径）：
    launch 地板 1.89µs（1 block 512B copy，aiv_time 1.36µs）/ 40 block 空载 3.36µs / per-core 流式
    GM 读 68 GB/s → Ratio 上界 = ideal ÷ launch 地板 与 核数比例 × 单核效率 两条独立约束
  - 仓内先例检索（T-1，首证任务）：examples/engram/engram_decode.py（同 grid=(B,)）/ examples/norm/rms_norm.py
    / examples/gemv/example_gemv.py —— 无「单 block 内跨核归约」或「batch=1 用多核拆 GEMV」先例，
    天花板结论置信度按标准降级标注（1.89µs 地板反证不依赖先例）
  - **第二证（不同任务，设计层重做，task engram_decode_mcore-engram_decode_mcore-20261031T012849Z，
    2026-10-31）**：本任务独立重读三条款并在真机复现 ②——运行日志原样打印「kernel declares no
    '*_infer_workspace_shape_function' callback, no workspace allocated (set
    TILELANG_ASCEND_WORKSPACE_SIZE…)」，故跨核交接只能走**两次 kernel 发射**（外部 workspace 入参）
    而非内核内栅栏；同任务的 P1/P2 两枚设计期探针（空 kernel 地板 + split-d 第一阶段）与路线无关
    上界推导见 `examples/engram_decode_mcore/engram_decode_mcore/DESIGN.md` §5.6.2 论证 4/§5.6.3（任务工作区，provenance 允许失效）
  - **39–41% 外推偏高的五项偏差（逐项可核，DESIGN §5.6.3）**：① 40-block 发射地板按 1-block 的
    1.9µs 计，实测 FLOOR(32)=2.52 / FLOOR(40)=2.85；② 每块 DMA 发射成本未计入（b4 ≈28 条 ×
    0.21µs → scalar 4.83µs，0.5× 暴露 ≈2.4µs，占实测时延 20%）；③ 向量效率按峰值折算，实测
    mul+reduce 两遍下 b4 vec 7.57µs ≈ 31% 峰值（2.1× 差）；④ 跨步列块读按连续读 68 GB/s 计，
    实测段宽 128B 时 26–30 GB/s（2.3× 差）；⑤ 备选路线（split-K）未测——补齐后 b1 的 A 段由
    6.49（实测）降到 3.71（模型），b1 天花板从 4.71% 升到 8.23%，仍远低于 40%
workaround: >-
  ① **首证任务（不改契约）**：接受核数比例天花板（DESIGN §5.5 已显式记录「B ≤ 8 时仅 8/20 核工作」
  并主动接受），预算全部投入单核效率——state-shift 逐行小 DMA → 整块 2-D copy（−5.9~−8.8%）、
  BK=16 保持 auto multi-buffer 重叠（放大暂存块反噬 +25~37%）；剩余候选分别被 UB 容量（BK=32）、
  codegen 缺陷（fp16→fp32 混合 reduce）、本缺口（split-K）封死。互链：跨核不可得是 Ratio 天花板
  归因链终点；单核常数见 constants.md 的 910B3 数据点（queue VP-2026-0142/0143）。
  ② **第二证任务（两阶段绕法，代价已量化）**：用两次 kernel 发射充当硬件完成栅栏——A 段
  `stats`（列块 GEMV + 4 个局部部分和 → GM workspace）× B 段 `apply`（冗余合并 + 逐列尾部），
  全程无 atomics/自旋、无 flag 消耗。实测 op 级 Ratio 9.95%（b1 3.01% / b4 16.04% / b8 10.94%），
  相对 grid=batch 基线 3.472% 为 **2.87×**；代价 = B 段的发射地板是纯税（B 段 3.32/4.77/4.78µs
  中 scalar 占 0.85/1.92/1.37µs，vec 仅 0.34–0.37µs）。该绕法**不依赖**本缺口补齐，但补齐后能把
  这段税抹掉（→ 回到 A 段天花板 15.7%/16.2%）。
occurrences: 2
tasks: [TileOPs-_engram_decode_kernel-20261030T121905Z, engram_decode_mcore-engram_decode_mcore-20261031T012849Z]
toolchain_stamp: 首证 tilelang 0.1.2+76e4550（commit 013dbbf）；第二证 tilelang 0.1.2+3ed7c91420e925ff3ec388fb981dd6ebf45ff4bf；均 + CANN 9.0.0 / Ascend910B3（单 die，20 AI Core / 40 AIV）/ 2026-10-30～31
status: recurring
proposal: >-
  目标层 Frontend API / runtime（跨 AI Core 归约与完成栅栏）。建议改动（三选一或组合）：
  ① 前端提供跨 block 完成栅栏原语（如 `T.grid_barrier()` / 计数到达语义），使 split 归约可在
  单 launch 内闭合；② 开放 GM workspace 的 Python 侧绑定入口（`T.alloc_workspace` 或等价），
  免除 `<op>_infer_workspace_shape_function` 回调；③ 至少为 `T.atomic_add` 补「同 launch 内
  跨 block 可见性/完成」的文档条款或明确的 unsupported 说明（当前为未文档化假设）。收益量化
  （两任务实测）：首证任务核利用率被 batch/40 钉死（3.47% Ratio，12 分支内层优化封顶）；第二证
  任务以两阶段绕法拿到 2.87×（9.95%），若把 B 段（3.32–4.78µs，占 op 级 28–33%）压掉，同一
  结构可达 A 段天花板 split-d 15.7% / split-K 16.2%——即再把 9.95% 抬 ~1.6×。受影响算子：
  grid=batch 的小 batch decode/GEMV 族（EngramDecodeOp 两个谱系实测）+ 一切需要跨 AIV 归约的
  单 launch 结构。issue 草稿（[npuir] 前缀，**须用户批准后建**）：[npuir] 缺跨 AI Core 归约/
  完成栅栏原语（sync_block_* 仅 Block 内、workspace 仅 launcher 注入、atomic_add 无跨 block
  完成条款），单 launch split-K 不可达。
created_by: TileOPs/EngramDecodeOp Stage 4（2026-10-30，evolver distill 登记）
last_seen: engram_decode_mcore-engram_decode_mcore-20261031T012849Z（2026-10-31，设计层重做独立识别 + 上界订正）
```

> **升级依据（2026-10-31）**：task TileOPs-_engram_decode_kernel-20261030T121905Z（单核内 12 分支穷举后撞上天花板，识别缺口）与 task engram_decode_mcore-engram_decode_mcore-20261031T012849Z（设计层重做，独立重读三条款 + 真机复现 workspace 无前端入口 + 两阶段绕法交付）——两个不同任务、不同手段、同一缺口，达 ≥2 任务阈值。proposal 字段已产出（含量化收益与受影响算子清单），**待向用户显式列出并推动；外部 issue 须用户批准后建**。
> **外推纪律（本条教训，建议随 schema 固化）**：`blocked_algo` 里的「若补齐能力 → X%」类数字属**外推**，必须附「估算口径 + 逐项假设 + 推导可复算点」；本条旧值 39–41% 正是因为只给了结论数字、把「跨核通信当作免费」且按峰值折算向量效率，被下一次的实测整体推翻 1.8×。



```yaml
gap_id: CG-2026-0028
layer: Frontend API (tilelang.language v-prefix 数学操作)
capability: >-
  缺向量 FMA / MLA（vfma / vmla / 等价 a*b+c 单指令融合）：v-prefix 数学操作全目录
  （`docs/Tilelang.language/数学操作/`，20 个 API：vabs/vadd/vcos/vdiv/verf/vexp/vexp2/vln/
  vlog2/vmul/vpow/vrelu/vrsqrt/vsigmoid/vsin/vsqrt/vsub/vtanh）只有逐 op 的二元乘/加/减，
  无「乘加融合」原语，也无 compiler 自动融合的证据（生成的向量指令数 = 源码 op 数）。
blocked_algo: >-
  RoPE（half-split 2x2 旋转）的数学核 `y1 = a*c - b*s` / `y2 = b*c + a*s` 在无 FMA 时至少
  需要 4 乘 + 2 加减 = 6 条向量指令；若可用 FMA 则为 2 fma + 2 mul（或 4 条），即 **-33% 向量
  指令**。当前被迫保持 6 op（性能上限差约 0.2-0.3 µs/iteration，按本算子骨架模型对总时延
  影响 <=4%）。
evidence:
  - `docs/Tilelang.language/数学操作/` 目录全列（20 个 .md）无 T.vfma / T.vmla / 任何 multiply-add
    融合 API；全仓 grep `fma|vmla|multiply_add` 在 docs/Tilelang.language/ 下 0 命中（2026-11-04 实查）。
  - 实测（本算子 Stage 4 r4，msprof op 口径）：4k-d128 每 tile `aiv_vec_time` 0.75 µs / 16 条
    向量指令；8k-d128 `aiv_scalar_vector_stall_time` 1.21 µs（占 aiv_time 28%）→ 指令数在关键路径上，
    故 6→4 op 的等价化简是唯一剩余的结构性减 op 方向（其余：融合 cast 拷贝实测 -13.7%，见
    perf_opt/opt_log.md §3.3）。
  - 未文档化假设：无源码级证据表明 BishengIR 会做 vmul+vadd → vfma 的自动融合；估算依据为
    npuir codegen 生成的向量指令数与源码 op 数一致（本算子 6 op → vec_time 与 6 op 线性）。
  - **GLA 复现（2026-11-18，第二任务）**：`perf_opt/GLADecodeOp.py` L640-641 的
    `st = st*g` + `st = st+ou` 是两趟同地址 128x128 全平面读改写；MLIR 实测每头恰好
    7 vmul + 2 vadd + 2 vreduce，其中 5 条为全平面。按 b8-h64 实测 VEC 12.76 µs / 13 头 /
    5 pass = 0.196 µs/pass 折算，合成 1 趟可省 0.196 µs/头 → **b8-h64 -2.5 µs（-12%）**。
    验证命令：`msprof op --kernel-name=_gla_decode_algo1_1h_main --launch-count=8
    python3 run_once.py --module <kernel>.py --workload serving-b8-h64`。
workaround: >-
  保持 6-op 链（4 vmul + 1 vsub + 1 vadd），并用「交叉乘积先行 + fp32 中转 buffer 原地复用」
  把中转 buffer 从 7 个压到 6 个（r3，8k tile 52->69 行 / 波次 4->3，A/B -3.45%）。
  代价：多 2 条向量指令/tile（约 +0.1 µs/tile 的 vec 时间），无其它副作用。
occurrences: 2
tasks: [TileOPs-_make_rope_neox_1d-20261104T065720Z, TileOPs-GLADecodeOp-optimize-20261118]
toolchain_stamp: tilelang 0.1.2 dev root build / CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-11-04（GLA 复现戳 2026-11-18，CANN 9.3.0 运行时 + 同 dev root）
status: recurring
proposal: >-
  【目标层】Frontend API（`tilelang/language/customize_npuir.py` v-prefix 数学操作）→ BishengIR 向量指令。
  【建议改动】新增 `T.vmla(dst, a, b, c)` = `dst = a * b + c`（以及就地 `T.vmadd(a, b, c)` = `a = a*b + c`），
  签名与 `T.vmul(a, b, c)` 同形，仅多一个加数操作数；语义要求 dst 可与 c 同 buffer（就地累加，
  避免额外全平面 buffer）。可选：`vmla(x, y, z, out)` 的 4 操作数形式，或 `vmul(..., accumulate=True)` 标志。
  【收益量化】
  - GLA decode（本任务，2026-11-18 实测）：每头 5 条 128x128 全平面 pass，其中
    `st = st*g`（L640）+ `st = st+ou`（L641）两趟同地址读改写可合成 1 趟 → 每头 VEC 0.98 → 0.78 µs
    （按每 pass 0.196 µs 折算），b8-h64（13 头/核）预计 **-2.5 µs / -12%**，b8-h32 -1.3 µs；
    UB 平面流量同步 -20%（160 → 128 KB/头）。
  - RopeNeox（原条目）：6-op → 4-op，**-33% 向量指令**，<=4% 端到端。
  【受影响算子】GLADecodeOp（GLA state 更新）、RopeNeox/RopeLlama31/RopeLongRope、LerpFwdOp、
  AdamW/优化器类 a*x+b 链、所有 "乘后累加" 的两趟原地更新写法。
  【issue 草稿】"[npuir] add vector fused multiply-add (T.vmla/T.vmadd) to close the 2-pass
  read-modify-write in decode-style state updates"——附 GLA L640/L641 片段、RopeNeox 6→4 op 推导、
  两端实测收益；外部建单须用户批准。
created_by: TileOPs/RopeNeoxOp Stage 4 Round 4（2026-11-04）
last_seen: TileOPs-GLADecodeOp-optimize-20261118（2026-11-18，GLA algo1 每头 5 平面 pass 逐条审计；
  复验 `docs/Tilelang.language/数学操作/` 目录 20 个 md 仍无 multiply-add，全仓 `vmla|vmadd|vfma` 0 命中）
```

### CG-2026-0029

```yaml
gap_id: CG-2026-0029
layer: BishengIR（bishengir-compile 1.1.0 / LLVM 19.1.7，ConvertHFusionToHIVM → ConvertHIVMToStandard）
capability: >-
  Expert 双 Scope MIX kernel 内**无法表达第二块「可喂养」的 L1 操作数缓冲**（手写 L1 双缓冲 /
  ping-pong 的前置条件），两个互相独立的编译封堵点：
  (a) 第 4 块「活」L1 缓冲的首条 GM→L1 `T.copy` → `bishengir-compile` SIGSEGV（`err code: -11`，
  无任何诊断输出）；同一缓冲用 `T.clear` 或该缓冲从不被写则编译通过；
  (b) 绕开 (a)（保持 3 块活 L1）后，把 Cube 内层块循环按奇偶**成对展开**（静态化双缓冲的缓冲
  选择：稳定对 + 剥离末对 + 奇尾，或单 pair body + 钳位前瞻）→ 编译**不收敛**（>15 min 100% CPU，
  另含零星 SIGSEGV），基线同 session 26–42 s 通过。
blocked_algo: >-
  GQA-SlidingWindow-Varlen fwd（Expert 两相位、`T.Kernel(20)` persistent、bm=64/bn=256）的
  **片内 Cube 管道重叠**：baseline 实测 mte2 0.650 + mte1 0.307 + cube 0.243 + fixpipe 0.315
  = 1.51–1.54 × 墙钟（已部分重叠，四管道之和 3009 µs vs wall 1959 µs），单缓冲使 V(l+1) 必须等
  gemm(l) 释放 `l1_b`；若 mte2 完全藏进 cube+fixpipe 理论墙钟 ≈1291 µs（1.52×）。该 1.52× 的
  结构性空间因 (a)/(b) 不可达：4 条实现分支 0 条编译通过（M1a/M1b/M1d），唯一可编译的零新增缓冲
  形态（V 装载提到跨引擎 gate 之前）实测 +1.4%（两 long 同向）回退。
evidence:
  - 探针阶梯（同 session 同工具链，compile-only）：3 块活 L1 / 112 KB = 26–42 s 通过；+4 KB 第 4 块
    + copy = 72 次 SIGSEGV；+64 KB 第 4 块 + copy = 5 次 SIGSEGV（400 s）；第 4 块未使用 = 26 s 通过；
    第 4 块 + `T.clear` = 26 s 通过；同上 + 一条真实 V copy 进原块循环 = 28 次 SIGSEGV（600 s）
    （`examples/TileOPs/.../grouped_query_attention_sliding_window_varlen_kernel/perf_opt/logs/r10_m1/PROBE_LADDER.md`）。
  - 独立最小 kernel（4 块活 L1 + 2 gemm + 4 条 GM→L1 copy）= 37 s 通过 ⇒ 非「4 块 L1」通例，
    与本 kernel 的 IR 体量/依赖图交互（同目录 `logs/r10_m1/probes/_probe_4l1.py`）。
  - pattern-library `TRAP-l1-4th-live-buffer-copy-segv`（本轮新增，含同表证据）。
  - 与既有条目区分：CG-2026-0002 是 cbuf→cbuf copy（L1→L1）不支持；本条是 GM→L1 copy 在第 4 块
    活缓冲上的编译崩溃 + 展体不收敛，与 copy 方向无关（同一 direction 在 3 块缓冲下正常）。
  - **Round 11 复证边界收窄（同一算子，compile probe，task
    TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z）**：`nqb=2` 的「3 块活 L1 + 单 body +
    运行时 `slot`」形态（Q 取 `[2*bm,dim]` 单块 → pass-1 一次 `M=2*bm` 融合 gemm；pass-2 单 body
    在 `T.serial(2)` 内用运行期 slot 选 qb0/qb1；不新增第 4 块操作数缓冲、不做 trace-time 展开）
    在 dim=128 冷编译+首启 **8.2 s、0 SIGSEGV** ⇒ (a)/(b) 的封堵对象是特定结构，不是「双缓冲意图
    /循环体规模翻倍」本身。该形态只共享 K、V 仍逐 qb 载入且未带 depth-2 流水，8b-long fp16 单 run
    2282.1 µs vs 基线 2003.35 µs（+13.9%）⇒ 不构成本 gap 的可用绕法；其 dim=64 形状另触发运行期
    L0A 越界（**CG-2026-0030**，与本条编译封堵点不同）。
workaround: >-
  部分（非 gap 本体的解）：**nqb=2 的「3 块活 L1 + 单 body + 运行时 `slot`」形态可编译**——Q 取
  `[2*bm,dim]` 单块、pass-1 单 body 融合 `M=2*bm` gemm、pass-2 单 body 在 `T.serial(2)` 内用
  运行期 slot 选 qb0/qb1（不做 trace-time 成对展开、不增加第 4 块操作数缓冲），dim=128 冷编译+
  首启 8.2 s vs 基线 26–42 s、0 SIGSEGV（Round 11）。但该形态只共享 K、V 仍逐 qb 载入且未带
  depth-2 流水，8b-long fp16 单 run +13.9%（2282.1 vs 2003.35 µs）⇒ 不构成本 gap（第二块可喂养
  L1 操作数缓冲）的可用绕法；gap 本体仍 open。另：零新增缓冲重叠形态（把与 flag 无关的 GM→L1
  装载提到 gate 之前）实测 +1.4%，属负收益，不采纳。
occurrences: 1
tasks: [TileOPs/grouped_query_attention_sliding_window_varlen Stage 4 Round 10（M1）, TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z（Round 11 编译探针）]
toolchain_stamp: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04
status: open
created_by: TileOPs-_gqa_sw_fwd_varlen_kernel Stage 4 Round 10（2026-11-04）
last_seen: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z Stage 4 Round 11（2026-11-04，nqb=2 编译探针边界收窄）
```

### CG-2026-0030

```yaml
gap_id: CG-2026-0030
layer: codegen / runtime（具体层未隔离：npuir 编译通过，device 执行期 aicore abort；bishengir-compile 无诊断）
capability: >-
  Expert 双 Scope MIX attention 的 nqb=2 形态（Q 取 [2*bm,dim]、pass-1 单次 M=2*bm 融合 gemm、
  pass-2 运行期 slot 选择 q-block）在 dim=64 且**动态 `T.serial(loop_count)` 的最后一轮 band tile
  满宽（tnc == bn_eff == 256）**时，device 执行期报 `aicore exception: The operation address of
  L0A exceeds the maximum range of L0A. The write address of the mte load3d instruction is out of
  range`（error code 0x800000001000 / retCode 0x26，mte error info 0x600000043）——编译期无任何诊断，
  故障只在运行期暴露。**（2026-11-05 R12 谓词修正：旧口径 `kv_len % bn_eff(256) == 0` 已被
  `q=128/k=384` PASS（l=0 处有满 256 块但非最后一轮）与 `q=384/k=384` FAULT（最后一轮满宽）证伪；
  「有满宽块」不触发，「末轮满宽」才触发。）**
blocked_algo: >-
  该 nqb=2 形态在 dim=64 且存在末轮满宽 band tile 时必崩（两 gemm 均牵涉：只钳 pass-1 的 N 或只钳
  pass-2 的 K 到 192 仍 FAULT，只钳 pass-1 的 M 到 bm 也 FAULT），故「3 活 L1 + 单 body + 运行时
  slot」的可编译性结论不能外推为 dim=64 可用性；dim=128 的真实 workload（8b-short k=512 /
  8b-long k=2048）不受影响。R12 已给出可用绕法（bn_eff ≤ 128），见 workaround——本 gap 的剩余
  诉求从「dim=64 不可用」收窄为「bn_eff=256 的满宽末轮不可用」。
evidence:
  - 首证复现集（dim=64，`python _gqa_sw_fwd_varlen_kernel_round11_nqb2_probe.py --level L0`）：
    t256(q=128,k=256) FAULT / t512(q=64,k=512) FAULT / c3(k 256/512) FAULT / c4(k 256) FAULT；
    反例 c1(k=128) PASS / c2(k=128 bf16) PASS / t384 PASS；dim=128 真实 workload 启动正常。
    （`perf_opt/logs/round11/nqb2_probe_L0_attempt{1,2,3}.log`，三次 attempt 同一报错文本）
  - R12 谓词分离（task TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z，等价驱动
    `perf_opt/_gqa_sw_fwd_varlen_kernel_round12_l0a_repro.py`，逐 case 单进程）：FAULT =
    m1(q128/k256,B=1) / t256 / t512 / c3 / **q384/k384（末轮 tn=256）**；PASS = k128（末 tn=128）/
    **q128_k384（末 tn=128；l=0 处有满 256 块但非末轮）** ⇒ 相关性从「kv 整除」改为「末轮满宽」。
    诊断变体 mclamp / p1tnc192 / p2tnc192 均仍 FAULT（`perf_opt/logs/round12/repro_*.log`）
  - R11 的「k=384 PASS」反例在 R12 等价驱动下按 `q=384/k=384` 运行 FAULT、按 `q=128/k=384` PASS
    ⇒ 该反例应按后者读；两轮 shape 表是否逐字相同未能从 R11 工件确证（未实证假设）
  - 绕法证据：`dim==64 → bn_eff=min(bn_eff,128)` 后 L0 **PASS 4/4 + contract**（fp16 max|dout|
    1.953e-3 / bf16 1.562e-2 tier2 / max|dlse| 1.9e-6；`perf_opt/logs/round12/bn128_L0.log`）；
    dim=128 的 IR 不变（钳位在 `if dim == 64` 内），8b-long fp16 单 run 2278.72 µs vs R11 同内核
    2282.1 µs（−0.15%，单 run 噪声）
  - pattern-library 待登记条目：`traps-runtime.md` 的 L0A 越界（Tier 1 / repro-missing，见 queue
    `VP-2026-0211`，R12 已按本修正更新其谓词、绕法与证据）
  - 与 CG-2026-0029 区分：0029 是编译封堵（第 4 块活 L1 copy / 展体不收敛）；本条是编译通过后的
    运行期地址越界，且只在 dim=64 + 末轮满宽条件下出现。
workaround: >-
  部分（R12 实测）：`dim==64 → bn_eff = min(bn_eff, 128)`（trace-time 常量分支，dim=128 的 IR
  不变）——L0 PASS 4/4 + contract；代价 = bn 256→128 在该族前序轮实测 +26%（且 `L_max=11 ⇒
  2·L_max=22 > 15` 使 pipe2 回落单槽），故该绕法只在 dim=64 形状上启用、不作为 dim=128 路径。
  另一条回避路径 = dim=64 不跑该 nqb=2 形态。根因未隔离：怀疑 2*bm 融合 gemm 在末轮满宽
  （tnc == bn_eff）时的 dynamic extent / L0A 行偏移计算，未证实（未文档化假设）。
occurrences: 1
tasks: [TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z, TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z]
toolchain_stamp: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-04（R12 同戳）
status: open
created_by: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T155252Z Stage 4 Round 11（2026-11-05 蒸馏）
last_seen: TileOPs-_gqa_sw_fwd_varlen_kernel-20261104T171249Z Stage 4 Round 12（2026-11-05，谓词修正为「末轮满宽 band tile」+ bn_eff=128 绕法实证）
```

### CG-2026-0031

```yaml
gap_id: CG-2026-0031
layer: Frontend API / codegen（reduce-with-index 的 dst 形态）
capability: >-
  TileLang 前端 `T.reduce` 接受 `reduce_mode="max_with_index_left/right"`、
  `"min_with_index_left/right"`，但 `npuir_reduce` 只有一个 dst；C++ 解析器
  `src/op/ascend.cc::NpuirReduce` 也只取 `args[1]` 作为 dst，codegen
  `VreduceCodegen` 只创建单 dst 的 `hivm.hir.vreduce`。HIVM 层 `VReduceOp`
  已完整支持 argmax/argmin：`dst` 为 variadic，要求 value+index 两个 dst
  （`getDstIndex()`），并有可选 `indices` 操作数。因此前端无法表达 fused
  arg-reduce，只能手拼 `reduce_max -> vbrc -> vcmp -> vselect -> reduce_min`
  五步向量链。
blocked_algo: >-
  argmax/argmin 的 fused 归约；被迫使用 5 步链（每元素 4~5 个向量 op）。
  实测 hidden-state (2048,4096) fp16 30.3us，vec 忙碌约 19us（占 66%~72%），
  5 步链下官方 Ratio 上限约 50%；若能用单条 with-index reduce，预估 vec
  时间可降 2~3x，hidden 有望到 16us（60% 线）。
evidence:
  - 探针 /tmp/opencode/probe_argreduce.py：`T.reduce(..., reduce_mode="max_with_index_left")`
    编译报错 `'hivm.hir.vreduce' op with index should have exactly 2 destination operands`
    （tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3，2026-11-14）
  - 3rdparty/AscendNPU-IR/build/install/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td
    的 VReduceOp：Variadic dst + getDstIndex() + isArgminOrArgmax()；
    HIVMEnums.h.inc 定义 max_with_index_left/right、min_with_index_left/right
  - tilelang/language/customize_npuir.py::npuir_reduce 仅一个 dst 形参；
    src/op/ascend.cc::NpuirReduce 只解析 args[1] 为 dst；
    src/target/codegen_npuir_dev.cc::VreduceCodegen 与
    src/target/codegen_npuir_api.cc::VreduceCodegen 都只建单 dst VReduceOp
  - docs/Tilelang.language/规约操作/T.reduce.md 的 reduce_mode 只列
    sum/max/min/abssum/absmax，无 with_index 形态
  - 2026-11-14 P3 本地补齐证据：`reports/2026-11/argmax_p3_20261114/compiler_patch.diff`
    （6 文件）、`report.md`（status=passed，hidden fp16 15.40us/62.25%）；
    独立探针 `probes/p3_fused_check.py` 对 (8,256)/(64,4096 并列)/(2048,4096)
    全部 bit-exact；memref/ub 双 dst 手写 MLIR 用 bishengir-compile 编译通过
workaround: >-
  5 步链：reduce_max -> vbrc -> vcmp("eq") -> vselect(idx, BIG) -> reduce_min；
  代价是每元素 4~5 个向量 op，hidden-state 的 vec 时间约为单 reduce 的 2~3x。
  **2026-11-14 P3 本地补齐**：新增前端 `T.reduce_argmax/reduce_argmin` +
  `tl.npuir_reduce_with_index` + `NpuirReduceWithIndex` + API codegen
  `VreduceWithIndexCodegen`（memref/ub、双 dst），重建 libtilelang_module.so；
  hidden fp16 30.5→17.82us（P3）→15.76us（P3.5）→15.40us（P3.6）→15.34us（P1），官方 Ratio 31.4→53.8→60.8→62.25→62.5%；bf16 36.3→25.34us（32→37.8%，经 fp32 上采）。tensor-form 双 dst 仍
  `Not bufferized`，只能走 Expert + `T.alloc_ub`；persistent 循环需
  `PIPE_ALL`（one-block-per-tile 可规避）；P3.7 用 `wait_flag/set_flag` 替代 `PIPE_ALL`
  会在纯 Vector + auto-sync 上下文 deadlock，正确修法是让 auto-sync 感知第二 dst。
  **2026-11-15 复核**：auto-sync 与 bf16 verifier 均在闭源 BiShengIR
  （`3rdparty/AscendNPU-IR`/`-Dev` 为空，实现在 `libBiShengIRHIVMDialect.a`/`bishengir-compile`），
  本地不可修；备用路径 sync_threads=Segfault、vbitcast packed-key=object 未生成、
  3d Pipelined=no gain。当前最佳 P3.6/P1：fp16 15.34us/62.5%、bf16 25.34us/37.8%。
  补丁尚未上游，repo 默认关闭
  （`TILEOPS_ARGMAX_FUSED=1` 开启）。
occurrences: 1
tasks: [argmax-plan-20261114]
toolchain_stamp: tilelang 0.1.2 (dev root build /mnt/workspace/tilelang-mlir-ascend) / CANN 9.0.0 / Ascend910B3 (20 AIC / 40 AIV) / 2026-11-14
status: in-progress
created_by: ArgmaxFwdOp 基本情况与优化方案（2026-11-14）
  - 2026-11-15 封锁证据：`reports/2026-11/argmax_p0_p1_blocked_20261115/README.md`
（auto-sync/bf16 verifier 闭源；sync_threads segfault；vbitcast link fail；3d Pipelined no gain）
last_seen: ArgmaxFwdOp 本地复扫（2026-11-15，3d bm=3584 6.82us/35.1%；fp16 15.28/62.7%、bf16 25.60/37.5%；3 档均值 45.11%；编译器项仍不可执行）
```

## Resolved（proposal / in-progress / fixed / closed / withdrawn 归档）

（暂无条目）
