# 硬件常数表（设计期 roofline 与调优定量口径）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。**本表是已实测硬件常数的唯一事实源**（D-2：evolver 蒸馏新 D 类常数时追加进本表，Tier 0）；设计期 roofline 估算协议（如何用本表算下界）见 `_shared/standards/hardware-cost-model.md`。条目带 front-matter（schema 见 INDEX.md §3），`source` 指向首次实测条目 ID（provenance，允许失效——常数数值与适用条件已自包含于本表）。
>
> **使用纪律**：① 引用常数须带版本戳（工具链变更后自动待重验，`kb_stale_check.py` 检测）；② 常数绑定设备（当前全部为 Ascend910B2C）；③ 外推前核对适用条件（如 MTE2 曲线绑定 4-set 轮换访问模式）。

---
id: CONST-capacity-910B2C
kind: constant
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z / mish-optimize-2026-08 / multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 r9 反例数据点）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（第三数据点：Expert 显式 alloc 结构依赖确认 + bl=128 容量否决）
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0（L1/L0C）；2026-08-24 build（UB 膨胀）；2026-09-15 重校 6797758；2026-09-16 反例 a13585dc；2026-09-17 ssd 1990aa9fe4
source: PL-1.9-blockwidth / PL-1.9-hardlimits / TRAP-UB-multibuffer-inflation / PL-1.12-task-pipeline-depth2
repro: repro/PL-1.12-bn-clamp-bm-guard.py
---

### 片上容量三硬上限（Ascend910B2C）

- **UB = 192KB/AIV**；auto-multi-buffer 会使 UB 占用**膨胀 ~1.7x**——大 block_size 探索须按膨胀后预算评估（BishengIR `ub overflow, requires N bits while 1572864 bits available` 报错文本反推）。**〔2026-09-15 第五轮重校，工具链 6797758〕auto-multi-buffer=false 下 BishengIR 实际需求仍 ≈ 真实手工预算 × 1.10–1.12**（4 点标定：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%——CG-2026-0008；设计期 UB 表须按此系数放大，且勿漏计小 buffer）。**〔2026-09-16 第六轮 r9 反例数据点，工具链 a13585dc〕系数结构依赖**：显式 alloc 结构（全部 UB buffer 首维 = half 的逐 buffer 显式分配，`--enable-auto-multi-buffer=false`）实测手工核算 214080B vs BishengIR 实际 210080B（ratio **0.981，actual 低于手工**）——×1.10–1.12 上偏系数在该结构不成立，该结构实测 ~12% 余量即足（不必按 1.7x 悲观；正负两向偏差均无文档，逐 buffer 分配可见性诉求见 CG-2026-0008）。档案：attention.md PL-1.12 r9 update；核算镜像 + pre-fix 溢出断言：`repro/PL-1.12-bn-clamp-bm-guard.py`。**〔2026-09-17 ssd_chunk_scan 第三数据点，工具链 1990aa9fe4，Expert 显式 alloc 结构——不计 CG-0008 occurrences，属结构依赖确认〕**：Expert 直差分因子链手工峰值 ~179KB ≈ 上限 93%；v3 实测 +16KB 手工 → 195.1KB 实测报错（偏差 <2%，显式 alloc ≈ 手工核算再证）；bl=128 变体编译实测需 582KB（bs=128）/ 357KB（bs=64）vs 192KB——容量否决（DESIGN §1.6.3 变体 B 以实测常数定格）。
- **L1 (cbuf) = 512KB/核**（`cbuf overflow, requires N bits while 4194304 bits available` 反推，2026-09-08）。双槽 k/v/p 的 bn 上限公式：`2·slots·bn·dim·2 + slots·bm·bn·2 + bm·dim·2 ≤ 512KB`（bm=44 时 bn≤427；wide 单槽模式 bn=512 可行，k+v+p+q=312KB）。
- **L0C (cc) = 128KB**（`cc overflow, requires 1310720 bits while 1048576 bits available` 反推）：`l0c_s[bm,bn]f32 + l0c_o[bm,dim]f32 ≤ 128KB` ⇒ bn=512 时 bm≤51。

---
id: CONST-L1-port-bw
kind: constant
family: [attention, expert, cube]
apis: [T.gemm]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### L1 端口读写带宽 ≈ 148–154 GB/s/核

探针复刻 KV+gemm1 数据流 + 真实 kernel 反推双证。ping-pong 双槽**不能**解除 MTE2（GM→L1 写）与 MTE1（L1→L0 读）的端口读写串行——Cube 操作数流的吞吐地板。用法：persistent 结构的 KV 重读流量（tasks/核 × 2MB × 2）÷ 此率 = 该结构的算子级下界（fa4096 ≈ 125–131µs 实测例）。

---
id: CONST-vector-launch-overhead
kind: constant
family: [attention, expert, elementwise]
apis: [v-prefix ops]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### 向量算子发射开销 ~0.5µs/op；逐元素速率 f16≈f32

无 f16 打包加速；每 op ~0.5µs 固定发射成本主导——同元素总量下宽块 op（bn=512）恒优于窄块 op 翻倍（bn=256）。用法：向量链时延下界 ≈ op 数 × 0.5µs。

---
id: CONST-fabric-bw
kind: constant
family: [attention, expert]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### fabric 聚合带宽 ≥1.73 TB/s（24 核 × 72GB/s，pp 探针）

shared=distinct=pp 无争用惩罚。fa4096 级 workload 现聚合 1.41TB/s 未触顶——地板因子排序通常为 L1 端口 / 向量发射率先于 fabric。

---
id: CONST-mte2-degradation
kind: constant
family: [elementwise]
apis: [T.copy]
dtype: [fp16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z / ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（L2 驻留口径补充）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（指令维度口径补充）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z（段数记账更正）
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ CANN 8.5.0；L2 口径 tilelang 0.1.2+a83118285a / 2026-09-10；指令维度口径 tilelang 0.1.2+1990aa9fe4 / 2026-09-17；段数更正 tilelang 0.1.2+15ad002b3d（与 4515de8 同源，git diff 零改动）/ 2026-09-21
source: PL-1.6-copy-floor / CASE-norm-adalayern-stage4
repro: repro-missing
---

### MTE2 有效带宽随总流量退化（4-set 轮换，fp16 bs4096，48 vector core）

1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core；256M 的 3:1 R/W 混合全流量 **~1.26 TB/s 为该设备地板**——"~1.8 TB/s 峰值参考"对混合读写流量不可达（勿据其估算大 N headroom）；1M 档读带宽含 L2 成分。grid-stride（48 核聚集移动窗口）实测优于每核连续分块 +3.2%（256M）。

**L2 驻留口径（ada_layer_norm 2026-09-10 补充）**：sets=1 访问模式（同输入张量逐 launch 复用，Stage 5 bench harness 同款）下 MTE2 有效读带宽 43.1 GB/s/core（聚合 ~2.07 TB/s，2048×4096 fp16，43.1×48 核）——远高于上表 HBM 侧曲线同量级值；带宽数字须注明数据驻留状态。设计期 roofline 按上表 HBM 口径估 L2 驻留 workload 会高估时延（ada 实证：DESIGN 估算 53–67µs vs 实测 90.5µs 基线〔失准主项为发射/重叠〕，调优后 40.5µs 优于估算下界——sets=1 下带宽不是地板项，瓶颈转为 Vector/UB 流量 vec_ratio 0.91）。

- **update（2026-09-17 ssd_chunk_scan Stage 4 追加指令维度口径）**：GM→L1 nd2nz 指令代价 ≈ **300ns/指令**、≈ **4ns/128B 段**（w2 标定：16 指令/任务×32 任务 = 156µs mte2 busy；段数 = ws_lcb 640 + ws_c 256 + x 256 + prev 64/任务）——小块搬运是**指令数/段数受限**而非带宽受限（41GB/s ≪ 148GB/s L1 端口）；减少指令数（合并装载）或增大行宽（128B→256B 段）是仅有的两个削减方向。AIV 重复执行时两 AIV 子块指标镜像（判据见 PL-1.13）。**拆分口径注（2026-09-20 ssd 重跑任务 Stage 2 检视登记）**：与 attention.md PL-1.18 段数墙构成的拆分互斥（彼处记 ws_c 128 + prev 128，本条记 ws_c 256 + prev 64）——总和一致（≈1216 段）而分项矛盾，源出两任务不同估算/标定路径；引用以总段数为准，待 Stage 4 段数墙 profile 复核厘清。**〔2026-09-21 复核解决，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z Stage 4，tilelang 0.1.2+15ad002b3d（与 4515de8 同源）〕段数记账更正**：ssd 每任务 Cube mte2 真实流量 = **240KB（1920 段）**，非旧记 1216 段——prev_states 拷贝在 lt 循环内 4× 重读（512 段/任务）被两套旧拆分共同漏算（本条 2026-09-17 的「16 指令/任务」w2 标定同为误记，实测基线 19.0/任务）；互证：aic_mte2_instructions 19.0/任务 = 1(x)+Σ_lt[1(ws_c)+1(prev)+(lt+1)(band)]。prevhoist 后 192KB（1536 段）/16 指令。段≈128B、~2.9-3.2ns/段的量级不变；引用分项拆分以指令数互证口径为准（attention.md PL-1.18 update①）。

---
id: CONST-copy-floor-method
kind: pattern
family: [elementwise]
apis: [T.copy, T.vadd]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ CANN 8.5.0
source: PL-1.6-copy-floor
repro: repro/CONST-copy-floor-method.py
---

### copy-floor 标定法（方法学常数）

与目标 kernel 同结构（同 persistent/分块/映射）的 3 载入 + 2 vadd 保活（DCE-proof）+ 1 写出探针，一步判定计算链暴露量。实测例：lerp_tensor 7-pass fp32-transit 链 ≥16M 全 dtype delta ≤1.1µs（16M 仅 0.34µs）——auto multi-buffer 将全部向量 pass 隐藏于 MTE2 窗口内。标量削减判定式：mte2_ratio ≥0.95 时 per-tile 标量/分支开销已被隐藏。

---
id: CONST-flag-id-budget
kind: constant
family: [attention, expert]
apis: [sync_block_set, sync_block_wait]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-twophase（VP-2026-0042 证据链）
repro: repro-missing
---

### flag id 预算 ≤ 15/核

跨引擎 per-slot flag 的 id 数量预算上限（第二轮 round-10 否决两遍式时的推理输入之一——该推理本身被第三轮"全局 [S,S] ws + n-block 下标 flag（nk≤16）"推翻，但预算上限本身实测存活）。两相位结构用 n-block 下标作 flag（nk≤16）贴上限可行。

---
id: CONST-store-fixpipe-gm-only
kind: constant
family: [attention, expert, cube]
apis: [store_fixpipe]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits（[DESIGN_LIMIT] 修正附录存活清单）
repro: repro-missing
---

### store_fixpipe 仅支持 L0C→GM（存活硬上限）

S/P 类中间矩阵无法经 fixpipe 直写 L1/UB——跨引擎传输强制 GM 往返（S-f16 传输是字节减半的缓解，非消除）。store_fixpipe f32(L0C)→f16(GM) 为正确数值转换（max_rel 4.7e-4 = 1 f16 ulp）。

---
id: CONST-aicore-910B2C
kind: constant
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: 2026-09 build
source: PL-1.7-expert-persistent-boundary
repro: none
---

### 物理核数 = 24 AI Core（Ascend910B2C）

persistent 结构按 24 核设计（logical tasks = ceildiv(S,bm)·H·B；wall-rows = ceil(ceildiv(S,bm)/24)·bm——bm=44 使 S=1024/2048/4096 得 44/88/176 行）。实查命令：`NPUUtils.get().get_aicore_num()`（设计期分核三要素之一，禁止假设值）。

---
id: CONST-arange-scalar-materialize
kind: constant
family: [reduction, elementwise]
apis: [T.arange, T.vbrc]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_arange_scalar.py
---

### T.arange (M,W) 物化按标量执行 ~1ns/elem；(1,N)+首轴 vbrc 绕法

`T.arange(idx_j, [0,1], 0)` 把 (M,W) 索引缓冲逐元素标量写入（(2,4096) 实测 8.56µs ≈ 1.05ns/elem，占 hidden-state 链式 47µs 的 18%）。绕法：`T.arange(idx_row, [0,1],0)` 于 (1,N) + `T.vbrc(idx_row, idx_j)` 首轴广播 (1,N)→(M,N)（文档合法 src (1,N,K)→(M,N,K)），实测 46.97→38.41µs（-18.2%）。bm=1 时两形相同——同形 vbrc 触发空广播维 verify fail（见 TRAP-vbrc-same-shape-empty-broadcast）。判据：链式 kernel 的固定开销 ∝ 首个 (M,W) 缓冲元素数且 scalar 占比高。

---
id: CONST-gm-to-ub-bw-dilution
kind: constant
family: [reduction, elementwise, general]
apis: [T.copy]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_strided_colblock_read.py
---

### gm_to_ub_bw 是总时长稀释口径；跨步列块 2D 读本身不慢（误读更正记录）

**误判更正**：曾把 partial 的 gm_to_ub_bw 0.80–1.11 GB/s/core 读成「(M,tn) 跨步列块读慢」——实为**标量化计算拉长总时长、带宽被稀释**（同 profile 的 mte2_active_bw 一直有 24.9GB/s；mte2_ratio 仅 0.05–0.15）。copy-only A/B（(M,tn) 2D 列块读 vs per-row (1,tn) 读，同写出量）：墙钟比 0.98×——**读形态无差异**。方法论：判断搬运瓶颈用 `mte2_ratio × aiv_time`（活跃时长）与 `*_active_bw`，勿用总量/总时长口径；(M,tn) 2D 跨步列块 copy 在段宽 ≥ 数百字节时性能正常（对照：窄「行」(bm,4) 的 16× 是**流量**膨胀〔GM_to_UB_datas 超采〕，非带宽下降——两类现象勿混）。

---
id: CONST-reduce-dims0-skinny
kind: constant
family: [reduction]
apis: [T.reduce_max, T.reduce_min]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_dims0_skinny.py
---

### dims=0 归约在长条形 (K,4) 上逐列串行（文档合法但灾难性慢）

`T.reduce_max(val_f (nchunk,M), g, dim=0)` 在长第一轴窄第二轴（(400,4)/(80,4)）实测 33.4/8.8µs——逐列串行 walk（~21ns/elem）。绕法：shared staging + `T.transpose((K,M)→(M,K), permutation=[1,0])` + dims=1 链（PL-1.1 家族），同数据 3.3µs。dims=1（行向归约）为快路径；dims=0 仅在第二轴足够宽时可用。

---
id: CONST-ub-budget-vs-pattern-cap
kind: constant
family: [gemv, reduction]
apis: [T.reduce_sum, T.copy]
dtype: [fp16, fp32]
device: 910B3
status: verified
origin_task: TileOPs-_engram_decode_kernel-20261030T145026Z（round 3）
toolchain: tilelang 0.1.2+01805d7a / CANN 9.0.0 / Ascend910B3
repro: repro/CONST-ub-budget-vs-pattern-cap.py
---

### `ub overflow` 归因：与尾部工作集共享 192 KiB 预算，而非 pattern 级元素上限

`engram_decode` 完整 kernel 在 BK=32/d_padded=512 报
`ub overflow, requires 1606656 bits while 1572864 bits available`（可用 192 KiB）。三条竞争假设的实测判别（910B3 / tilelang 0.1.2+01805d7a / CANN 9.0.0）：

- ❌ **不是 pattern 级元素上限**：裸 mul+reduce（row-broadcast 乘 + `T.reduce_sum(dim=0)`）用 `(32,512)`=16384 元素的 product tile **可以编译**（`repro/CONST-ub-budget-vs-pattern-cap.py`，3 次 JIT 约 1 分钟，已跑通）。
- ✅ **是共享预算**：同一完整 kernel 把 GEMV 之后的尾部整段删掉后，BK=32/d=512 **也能编译**。d=512 下尾部工作集 ≈68 KiB（含 `(max_conv_len-1, d)` 的 shift 暂存 fp16 ≈19 KiB、17 个 `(d,)` f32 行），与 GEMV 的 2×64 KiB 暂存相加越界。
- ⚠️ **该预算从 Python 侧腾挪无效**：把 4 个 `(1,d)` 归约量合并、`rms_w` 复用、`k_local/h_local` 移交给卷积抽头、删掉死 buffer（≈18 KiB），报错字节数**一字不差**；改 f16 暂存（暂存减半）也**一字不差**。

**判据（事实结论，可复用）**：`ub overflow` 的归因分两步——①「删掉无关区域是否仍溢出」：仍溢出 ⟹ 该段自身超限；不溢出 ⟹ 共享预算交互；②「报错字节数是否随实现变体变化」：4 种省 UB 实现（死 buffer alias / f16 暂存 / 尾部回收 / 原地乘积）报错字节数完全相同 ⟹ 该预算的可回收量 ≈0，alias 类省 buffer 改写在本工具链不产生预算。本 kernel 的实测可用上限即 `BK = 8192 // d_padded`（d=256→32 编译与实测均通过；d=512→16，32 被共享预算拒绝）。

> **边界（2026-11-05 R13 补记）**：上段的「alias 不产生预算」是 engram_decode 特定结构的结论——其越界量由同时存活的尾部工作集决定。当被别名的是**两段互斥生命周期**、且别名声明使一份 allocation 从 IR 中消失时，plan total 会等量下降（见下条 `CONST-ub-alloc-live-range` 的 D→N 实测 −24,576 B）。两结论的适用条件不同，不构成矛盾：判据是「该改写是否消除了一个实际声明的 allocation / 是否命中 live-range 复用」。

---
id: CONST-ub-alloc-live-range
kind: constant
family: [attention, gqa, mixcv, expert]
apis: [T.alloc_ub, T.copy, T.gemm]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: TileOPs-_gqa_sw_fwd_varlen_kernel-20261105T011012Z（R13 ubprobe + G=2 UB 账本）；第二证 TileOPs-_gqa_bwd_kernel-20261031T051147Z（BN=64 静态和 240 KB 仍编译通过，queue VP-2026-0169）
toolchain: tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3（20 AIC / 40 AIV）/ 2026-11-05；第二证 tilelang 0.1.2+9738284cd6 / CANN 9.0.0 / 910B3 / 2026-10-31
repro: repro/CONST-ub-alloc-live-range.py
---

### UB 预算 = plan total（user `T.alloc_ub` + 编译器 staging）；`hivm-plan-memory` 按 live range 复用

**分配器语义（4 缓冲直接探针，910B3 / tilelang 0.1.2+cebcec9d）**：4 个 `[64,256] f32` UB 缓冲 = 65,536 B each，静态和 262,144 B = 256 KB > 192 KB，两种生命周期形态结果相反：

| 生命周期 | 结果 |
|---|---|
| `disjoint`（每块消费并写回后才写下一块） | **编译通过**；post-`hivm-plan-memory` IR 的 4 个 `pointer_cast` 偏移为 `0 / 65536 / 131072 / 0`——第 4 块 alias 到第 1 块槽位，3×64 KB = **196,608 B = 192 KB** |
| `concurrent`（四块全部写完后才读） | **失败**：`ub overflow, requires 2097152 bits while 1572864 bits available`（256 KB vs 192 KB） |

⇒ `hivm-plan-memory` 是按 live range 复用的分配器，且 192 KiB 是硬 `>192 KB` 门。静态逐 buffer 加和既不是容量否决依据，也不是余量预测依据；R12 的「静态加和（无生命周期复用）」把这一结构-dependent 观测过度泛化了。

**真实预算 = user alloc + 编译器 staging（当前 GQA-SW-Varlen kernel，dim=128 / bn=256 / per-AIV）**：post-PlanMemory 偏移 `0, 32768, 65536, …, 194560, 195584 (+1024)` 全部互异——**该 kernel 零复用**（每块大缓冲跨整个 task loop 存活）。分量：

| 分量 | 字节 | 明细 |
|---|---:|---|
| user `T.alloc_ub` | **157,952** | 23 个 allocation，与 R12 的逐 buffer 表一致 |
| 编译器 staging | **38,656** | 2×`[4096] f32` vreduce 暂存 32,768 + 2×`[32,8,1] f32` strided-scalar staging 2,048 + 4×`[256] f32` view staging 4,096 |
| **plan total** | **196,608** | = 192 KB，**零余量** |

⇒ 该 kernel 的真实预算口径是 plan total（196,608 B），不是 user sum（157,952 B）；**staging 是 38.7 KB 的固定税**；本 kernel 任何按 user sum 估的余量都偏乐观 24.6%。

**应用（G=2 head-group batching 的容量门）**：未瘦身 G=2 = 175,616 user + 38,656 staging = **214,272 B（超出 192 KB 门 17,664 B）**；`ub_f16_D/f32_D` 声明为 pass-1 的 `ub_f16_N/f32_N` 的 `[half,dim]` 前缀别名（两段生命周期互斥，pass-1 只用 N、pass-2+epilogue 只用 D）后 **−24,576 B** ⇒ 151,040 user + 38,656 staging = **189,696 B（185.2 KB，余 6,912 B / 3.5%）**，编译通过且 L0 4/4 + contract PASS（数值与 G=1 逐位相同）。**注意**：该 kernel 零 live-range 复用，所以别名省下的正是声明层面移除的 allocation，而不是分配器回收。

**判据（事实陈述）**：容量结论先区分三种口径——① 静态逐 buffer 和（可能远高于实际）；② plan total（user + staging，实际占用）；③ 同时存活峰值（live-range 复用后）。本 kernel 的 ①② 关系是 `plan = static sum + staging`（零复用），带互斥生命周期的 GQA-bwd BN=64 是 `实际峰值 < static sum`。容量结论的口径与数字一同记录；最大 shape 的一次编译探针 + post-PlanMemory 偏移是可同时给出复用情况与 staging 量的机械证据（`TILELANG_DUMP_IR` / cache 的 `kernel.mlir`）。

**与既有条目**：`CONST-capacity-910B2C`（UB 192 KiB 硬上限）；`CONST-ub-budget-vs-pattern-cap`（`ub overflow` 归因到共享预算；其「alias 不产生预算」的边界见该条 2026-11-05 补记）；`TRAP-UB-multibuffer-inflation`（实际 > 名义的 auto-multi-buffer 膨胀——本条的 staging 税与之同族但独立于 multi-buffer 开关）。

---
id: CONST-reduce-dims0-ns8
kind: constant
family: [reduction, gemv]
apis: [T.reduce_sum, T.copy]
dtype: [fp32]
device: 910B3
status: verified
origin_task: engram_decode_mcore-engram_decode_mcore-20261031T012849Z
toolchain: tilelang 0.1.2+3ed7c91420e925ff3ec388fb981dd6ebf45ff4bf + CANN 9.0.0 / Ascend910B3（20 AI Core / 40 AIV）/ 2026-10-31；TILELANG_ASCEND_MODE=Developer
repro: repro/CONST-reduce-dims0-ns8.py
---

### `T.reduce_sum(dim=0)` 在瘦小 `(ns≤8, 8)` 上数值精确且成本 ≈0（`CONST-reduce-dims0-skinny` 的适用边界）

两段 kernel 里跨列块的部分和要在 B 段合并：`p_ws (B, ns, 8) fp32` → `(B, 8)`，即 `T.reduce_sum(prow, tot, dim=0)` 作用在 `(ns, 8)`（ns = 4 / 8）上。

- **数值**：与 torch `sum(1)` 的最大相对误差 ≤1e−6（ns=4 / 8 均过），两次运行结果**逐位相同**（无 atomics、硬件定序）——不需要 `T.serial` 标量累加来保证确定性。
- **成本 ≈0**（同口径 msprof，同 kernel 三 body 对照）：原式 `dim=0` / 不合并（单行 copy，即下界）/ `T.serial(ns)` 标量降级 —— **b4 4.650 / 4.720 / 4.590 µs**，**b1 3.290 / 3.100 / 3.290 µs**；差值落在 −0.07…+0.19 µs 噪声内，降级方案也不更快（−0.06 µs）。

**适用边界（与 `CONST-reduce-dims0-skinny` 的分界）**：该条目记 `(400,4)` / `(80,4)` 上 `dim=0` 逐列串行 ~21 ns/elem（33.4 / 8.8 µs）。本条说明惩罚的量级取决于**总元素数**而非「dim=0 一定慢」：`(8,8)` = 64 元素按 21 ns/elem 外推本应 ≈1.34 µs，实测为 0（噪声内）。判据：像「跨核部分和合并」这类 `(≤8, ≤8)` 的小归约可直接用 `dim=0`，不必付 transpose 或 `T.serial` 降级的工程成本；元素量上千的窄轴归约仍按 skinny 条目绕。

---
id: CONST-dyn-row-gather-opcount-dest-independent
kind: constant
family: [attention, sparse-mla, gather]
apis: [T.copy, T.Parallel, T.Pipelined]
dtype: [fp16]
device: 910B3
status: verified
origin_task: TileOPs-_sparse_mla_kernel-20261101T014649Z
toolchain: tilelang 0.1.2+63c8a2e6 + CANN 9.0.0 / Ascend910B3（20 AI Core，1800 MHz）/ 2026-11-01；TILELANG_ASCEND_MODE=Developer
repro: repro-missing
---

### 「动态行索引 GM→片上逐行 copy」的成本 ∝ 条数，与目的内存域和向量化形态无关（910B3 / 63c8a2e6）

主数据：1152 B/行（`Dqk=576` fp16），source 行地址由运行时索引决定（随机、互不相邻）。

| 形态 | Task Duration（smoke，5 launches 均值） | 结论 |
|---|---:|---|
| 逐行 `T.copy` GM→shared（base） | 9045.26 µs | 1.349 µs/行 |
| 逐行 `T.copy` GM→**UB** + 1 条连续 UB→buffer copy | 8978.83 µs（−0.74%） | 目的域无关 |
| `T.Parallel(Dqk)` 逐元素众集进 UB + 1 条连续 copy | 9049.30 µs（+0.04%） | 元素级「向量化」形态无关（未生成宽搬运） |
| **1 条 64 行连续 copy**（数学无效，仅归因） | **501.15 µs（18.05×）** | 成本 ∝ copy **条数** |

三个 tune workload 反算完全一致：1.341 / 1.349 / 1.351 µs/行（mainstream / longer-kv / long-kv），与 `block_N ∈ {16,32,64}`、`num_stages` 无关。
推论：凡「索引随机 → 每行一条 copy」的 gather，优化只能来自**减少条数**（提升块内行数/批共享），不能来自换目的域、换循环形态或元素级向量化。

**零-gather 地板与 Ratio 天花板**（同一探针组：把 64 条逐行 copy 换成 1 条 64 行连续 copy，数学无效、仅测时）——「gather 免费」时的实测地板直接给出官方 Ratio 上界（`Ratio = ideal / measured`，记忆受限点下 ideal 字节数不变）：

| workload（topk / 因果窗） | baseline µs | 零-gather 地板 µs | 地板加速比 | 官方 ideal µs | Ratio 上界 = ideal/地板 |
|---|---:|---:|---:|---:|---:|
| mainstream（2048 / 2048） | 566606.00 | 26525.50 | 21.36× | 1994.7 | **7.52%** |
| longer-kv（1024 / 4096） | 142398.31 | 6828.78 | 20.85× | 499.0 | **7.31%** |
| long-kv（512 / 8192） | 35687.42 | 1793.82 | 19.90× | 124.9 | **6.96%** |

地板的构成单元是每迭代 **4.8 µs**（1 条 64 行连续搬运 + 2 次 gemm + online softmax + 同步）：mainstream 26525.5 µs = 205 波 × 32 迭代 × 4.8 µs（实测吻合）；`T.Pipelined(num_stages=2)` 未能把它与相邻迭代重叠。判据：硬目标对应时延若低于该地板（本例 40% ⇒ 4987 µs，比地板慢 5.32× / 5.47× / 5.75×），该目标在当前结构族内结构性不可达，gather 免费时的最优也只是 ≈7.0–7.5%。

**减少条数的三条路在本工具链均被封**：① 增大 `block_N` 一次覆盖更多行——行地址随机，无法连续；② `block_heads=64` 减块数——L0C 需 144 KB > 128 KB 且 UB 264448 B > 196608 B，编译失败；③ 拆 k 路并行目的 buffer——`hivm.hir.load` 要求 `pad_mode`，而 `pad_mode` 仅为左右 padding 语义且未由 tilelang 前端暴露（capability-gaps `CG-2026-0025`）。

复现条件（provenance，允许失效）：origin 任务 `perf_opt/_sparse_mla_kernel_r1_{p_ubpar,p_ubrow,p_contig}.py`，各自 `python bench_runner.py --file <f> --case smoke-f16 --repeat 5` 同口径测时；tune workload 用 `--case {mainstream,longer-kv,long-kv}`。断言只锁方向与量级：per-row 成本 1.2–1.5 µs、连续化加速 >15×。

> **同算子对照（不增量 CG-2026-0018）**：本条阻塞点是 Developer 非持久 grid 内不能插入「gather→GM workspace→块读」相位（TRAP-DEVMODE-PERSIST-GEMM / CG-2026-0010），非跨 AI Core 归约/栅栏缺失；CG-2026-0018 的上界 ≤22.2% 是另一机制（grid=batch 核数钉死），量与本案 7.0–7.5% 不同属。
