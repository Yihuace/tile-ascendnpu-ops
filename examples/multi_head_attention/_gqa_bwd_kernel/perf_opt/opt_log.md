# `_gqa_bwd_kernel` Stage 4 调优日志（mode=full）

> 算子：`MultiHeadAttentionBwdOp` 的主反向 kernel `_gqa_bwd_main`
> （`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/`）
> task_id：`TileOPs-_gqa_bwd_kernel-20261018T145922Z`
> 唯一时延口径：`msprof op` 的 `Task Duration(us)`（kernel-only，串行独占设备）。
> 设计基线（只读）：`_gqa_bwd_kernel.py`（Stage 3 精度通过版，`default_config = {block_m:64, block_n:64 if dim<=64 else 32, num_stages:1}`）。

## 0. 工具链自检与环境

| 项 | 实测值 |
|---|---|
| 设备 | Ascend910B4（`npu-smi` 26.1.1；单卡） |
| tilelang | `0.1.2+2fb23eee1c6293affa903e631c1d8fb018cf173b`（仓库 dev root build，`git log -1` = `2fb23ee`，2026-10-18 11:29 UTC） |
| CANN | 9.0.0（`bishengir-compile` 位于 `/usr/local/Ascend/cann-9.0.0`） |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| 核数 | `NPUUtils.get().get_aicore_num()` = **20**；`AscendArch().aivector_core_num` = 40（1 block = 1 Cube + 2 AIV，Mix Block Dim = 40） |
| UB 容量 | 196608 B（编译期实测报文 `1572864 bits available` = 192 KB） |
| msprof | `msopprof`（msprof 26.1.1）；filter `--kernel-name=_gqa_bwd_main`，捕获名 `_gqa_bwd_main_mix_aic` |
| kb_stale_check | `stale_count=290`（当前戳 `tilelang 0.1.2+2fb23ee / CANN 9.0.0`）；本日志引用的旧条目均在本机重验后引用，见 §4 |

**测量协议**：每个 (candidate, workload) 串行执行
`msprof op --kernel-name=_gqa_bwd_main --launch-count=20 --warm-up=5 --dump=off --aic-metrics=BasicInfo,PipeUtilization,Memory`
包住 `bench.py`（同 tensors、同 launch 序列，20 次采样取中位数）；raw profile 落
`profiles/<phase>_<candidate>/<label>_<dtype>/OPPROF_*`，进程日志落 `logs/<phase>_<candidate>/`。
`run_measure.py` 解析 raw profile 并把一行追加进 `perf_records.jsonl`（append-only）。

## 1. Workload inventory（benchmark 展开与分类）

入口：`benchmarks/ops/bench_multi_head_attention.py::test_mha_bwd_bench`（ManifestBenchmark，op=`MultiHeadAttentionBwdOp`），
参数化来自 `tileops/manifest/attention.yaml` 的 `MultiHeadAttentionBwdOp.workloads` × `{float16, bfloat16}` = **10 例**，
全部 `is_causal=true`、`heads_kv == heads`（MHA，`groups=1`）。完整清单见 `workload_inventory.json`。

| workload_id | (B,S,H,D) | dtype | kind | 说明 |
|---|---|---|---|---|
| `mha-bwd-smoke-s512-h8-d64/fp16` | (1,512,8,64) | fp16 | smoke | label 含 `smoke` → 只做精度回归，无性能记录 |
| `mha-bwd-smoke-s512-h8-d64/bf16` | (1,512,8,64) | bf16 | smoke | 同上 |
| `llama-3.1-8b-short/fp16`,`/bf16` | (4,512,32,128) | fp16/bf16 | tune | 逻辑任务 `num_logical = B·H·⌈S/64⌉ = 1024` |
| `llama-3.1-8b-long/fp16`,`/bf16` | (2,2048,32,128) | fp16/bf16 | tune | `num_logical = 2048` |
| `llama-3.1-70b-short/fp16`,`/bf16` | (2,512,64,128) | fp16/bf16 | tune | `num_logical = 1024` |
| `llama-3.1-70b-long/fp16`,`/bf16` | (1,2048,64,128) | fp16/bf16 | tune | `num_logical = 2048` |

- **无 skipped**：8 个 tune 例全部 baseline/final 有效（含此前 `CG-2026-0030` 形态②的
  `(1,2048,64,128) fp16`——本机 910B4 + CANN 9.0.0 **未复现** bishengir 段错误，见 §4.6）。
- smoke 例不做性能采集/调优，仅在每次候选合并后与最终态做精度回归（§6.2）。

## 2. Baseline（round 0，Stage 3 冻结版）

### 2.1 Performance Test Data（`phase=baseline`，candidate `stage3-default`）

| kernel_id | workload_id | Task Duration(us) | AICore 利用率（占 Task Duration） | memory（GM read+write） | L0 |
|---|---|---|---|---|---|
| `_gqa_bwd_main` | `llama-3.1-8b-short/fp16` | **12068.3** | cube 2.2% / vec 5.9% / mte2 30.5% / mte3 45.7% / scalar 33.6% | 5930.8 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-8b-short/bf16` | 12063.1 | 同上 | 5930.8 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-8b-long/fp16` | **94099.5** | cube 2.3% / vec 5.9% / mte2 31.2% / mte3 45.7% / scalar 31.7% | 46842.0 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-8b-long/bf16` | 93891.6 | 同上 | 46842.0 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-short/fp16` | 12053.3 | 同 8b-short | 5930.8 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-short/bf16` | 12057.8 | 同上 | 5930.8 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-long/fp16` | 94077.0 | 同 8b-long | 46842.0 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-long/bf16` | 93921.5 | 同上 | 46842.0 MB | PASS |

> AICore 利用率口径说明：未采 `Roofline` 能力，这里用 `PipeUtilization` 的**限制 block 的 pipe 忙时 / Task Duration**
> 作为引擎占用率代理（cube=aic_cube_time，vec/mte2/mte3/scalar=aiv_*_time）。

**关键观察**：4 个 tune shape 只落两档时延（12053–12068 / 93892–94100 µs），比值 7.8 ≈ 迭代数比 8.0
（`iterations = B·H·⌈S/bm⌉·⌈S/bn⌉`：16384 vs 131072）⇒ **时延严格 ∝ 迭代数**，每迭代 ≈ 0.736 µs/核-迭代基准，
其中 `aiv_mte3`（45.7%）与 `aiv_mte2`（30.5%）为最大项。

**调优排序与理由**（只影响先后，不跳过任何 tune 例）：
① `llama-3.1-8b-long`（单例 94.1 ms，占 8 例总时长 ~86%，结构收益最可观测）→ ② `llama-3.1-8b-short`
（短档代表，12.07 ms）→ ③ `llama-3.1-70b-short` / ④ `llama-3.1-70b-long`（与 ①② 分别同迭代数、
仅 `B·H` 分解不同 ⇒ 预期同档，用于验证"不因 shape 分组而回退"）→ ⑤ 每轮的 8 例全量复测
（dtype × short/long 全交叉，作为合并非回退门禁）。循环内每轮按 ①② 做候选筛选、按 ③④ 复核、
最后 8 例全量落记录（`phase=merged`/`final`）。

### 2.2 设计估算 vs 实测（D-2 回填）

DESIGN §1.6.0 的估算表按**设计期枚举 shape** 给出（`full-bwd-fp16-large` = 16×2048×16×128，8192 逻辑任务；
`full-bwd-bf16` = 4×4096×16×128），manifest 中没有同形例。按逻辑任务数把它缩放到最近的 manifest 例
（`llama-3.1-8b-long`，2048 任务，scale 1/4）：

| 项 | 设计估算（缩放后） | 实测 baseline | 偏差 |
|---|---|---|---|
| 流量项（口径 A）下界 | ≈ 0.32 ms | 94.10 ms | **294×** |
| 元素吞吐地面档（发射项下沿） | ≈ 2.4 ms | 94.10 ms | **39×** |
| 串行 issued-op 上界档（发射项上沿） | ≈ 69–138 ms | 94.10 ms | 0.7–1.4×（落在带内） |

⇒ 设计「**发射受限**」的定性判定成立（实测落在发射项区间内），但量化带宽极宽（上/下沿相差 29×），
其下沿（元素吞吐档）严重失准：真实主导项不是元素吞吐，而是**逐行原子（mte3，45.7%）+ 跨步行装载（mte2，30.5%）
- 每迭代固定的标量/同步开销**。这三点即本轮的三个杠杆，见 §3。
另一处设计假设被实测**推翻**：DESIGN §1.6.0/§1.6.4 认为「行原子是 store 类 op，单价接近整 op 价、不是流量问题」，
实测逐行 1-D 行原子为 **224 ns/行**，而同一批字节的 rank-2 连续 2-D 区域原子为 **32 ns/行**（7×，§4.1）。

### 2.3 瓶颈画像（baseline，8b-long/fp16，`profiles/baseline/...`）

| pipe | 忙时 (µs) | 说明 |
|---|---|---|
| `aic_cube` | 2137 | 5 个 GEMM 合计仅占 2.3% |
| `aic_fixpipe` | ~2.4% 档 | L0C→L1 回写 |
| `aiv_vec` | 5567 | 9 个整 tile 向量 op |
| `aiv_mte2` | 29373 | q/do 每迭代 2 次跨步行装载（`GM_to_UB` 每 AIV 44 MB / 核） |
| `aiv_mte3` | 43008 | **逐行 dq 原子环**（每迭代 32 行） |
| `aiv_scalar` | 29822 | 发射/同步；`aiv_scalar_wait_id7` ≈ 全时长的 85% |

每迭代（bm=64,bn=32，3379 迭代/核）：mte3 12.7 µs、mte2 8.7 µs、scalar 8.8 µs、vec 1.6 µs、cube 0.6 µs。
⇒ 单点结论：**逐行原子与跨步行装载主导；tile 面积（每迭代覆盖的 score 元素）是第一杠杆。**

## 3. Iteration Log

每轮固定 base = 上一轮已合并 current best；每分支只改一个主要优化点；
"候选 vs current best" 表数据来自 `perf_records.jsonl` + raw profile（`aiv_*`/`aic_*` 忙时 / `traffic_mb` 为 Memory CSV 的 read+write 合计）。

### Round 1 — `r1-causal-skip`（结构：causal 块级跳过）

- **现象**：baseline 对每个 (KV block, query block) 对都执行完整迭代，causal 下约一半迭代整块位于掩码三角内
  （P ≡ +0.0，对 dv/dk/dq 贡献恰好 0）。
- **候选**：在 `T.Pipelined` 体内加 runtime `if c0 + real_n > r0:`（DSIGN §9 R4 的 runtime start 不可用，但 body guard 可用）。
  跳过是**逐位精确**的（masked 位的 P 为精确 +0.0）。
- **分支结果**（8 例全测，`phase=candidate`）：

| workload | current best (µs) | r1-causal-skip (µs) | Δ | AICore 利用率（r1） | memory | L0 |
|---|---|---|---|---|---|---|
| 8b-short/fp16 | 12068.3 | **9195.5** | −23.8% | cube 2.2% / vec 7.7% / mte2 30.3% / mte3 43.6% / scalar 32.9% | 3380.1 MB | PASS |
| 8b-long/fp16 | 94099.5 | **53994.9** | −42.6% | cube 2.2% / vec 8.7% / mte2 31.0% / mte3 43.8% / scalar 31.6% | 24250.5 MB | PASS |

- **判读**：迭代数本应降 1.78×（short）/1.94×（long），实测只有 1.31×/1.74×。差额 = **静态负载不均**：
  任务按 `uid = it·20 + cid` 轮转、`by` 为最快下标，核心拿到的 `by` 组合工作量差异大（max/mean = 1.35×，见 Round 2 的表）。
- **winner**：保留（1.31–1.74×，逐位精确）。

### Round 2 — `r2-balance`（结构：任务映射改为 `by` 最慢）

- **现象**（来自 r1 的 profile + 解析模型）：每 task 工作量 ∝ `n_q − ⌊by·bm/bn⌋`，随 `by` 递减；
  轮转映射把重/轻 `by` 分给不同核。
- **候选**：仅改任务下标分解：`by = uid // (B·H)`，`bz/bx` 由余数展开（`by` 变为最慢下标）。
  20 核 × 轮转对每个 `by` 组均匀取样 ⇒ 解析 max/mean = 1.01–1.02×（原先 1.35×）。算术完全不变。
- **分支结果**（探针，2 例，`--no-record`）：

| workload | current best (µs) | r2-balance (µs) | Δ | AICore 利用率（r2） | memory | L0 |
|---|---|---|---|---|---|---|
| 8b-short/fp16 | 9195.5 | **7254.3** | −21.1% | cube 2.1% / vec 8.3% / mte2 29.6% / mte3 43.5% / scalar 34.1% | 3380.1 MB | PASS（check tier-1 全 0 超差） |
| 8b-long/fp16 | 53994.9 | **51622.0** | −4.4% | cube 2.1% / vec 8.7% / mte2 30.1% / mte3 45.0% / scalar 31.3% | 24250.5 MB | 同上 |

- **判读**：short 档的 21% 与解析预测的 1.35×→1.01× 一致（1.34×），long 档只对应 1.10×→1.01×。
- **winner**：保留。

### Round 3 — `r3-maskdiag`（结构：掩码链只在跨对角块执行）

- **现象**：掩码链（`vsub/vmul/vmin/vadd`，4 个整 tile op）只在 tile 跨越 causal 对角时需要；
  多数执行迭代是全有效块（`c0 ≥ r0 + bm − 1`）。
- **候选**：把掩码链包进 runtime `if c0 < r0 + block_m - 1:`。全有效块走纯 `exp` 路径（逐位相同）。
- **分支结果**（探针，2 例）：

| workload | current best (µs) | r3-maskdiag (µs) | Δ | AICore 利用率（r3） | memory | L0 |
|---|---|---|---|---|---|---|
| 8b-short/fp16 | 7254.3 | **7106.1** | −2.0% | cube 2.1% / vec 7.8% / mte2 29.7% / mte3 43.6% / scalar 33.7% | 3380.1 MB | PASS（check 与 baseline 逐位同值） |
| 8b-long/fp16 | 51622.0 | **51315.3** | −0.6% | cube 2.2% / vec 7.9% / mte2 30.3% / mte3 45.8% / scalar 30.5% | 24250.5 MB | 同上 |

- **判读**：收益落在噪声阈值（3%）边缘以下，但改动逐位精确、无代价，保留（vec 忙时 4496→4054 µs 与预期一致）。
  该分支同时为 Round 5/6 的 UB 让位（掩码链不再无条件占用 `dsd` 的活跃区间）。
- **winner**：保留。

### Round 4 — `r4-subid-split`（结构：AIV 双核重复执行的拆分）→ **REJECTED（结果错误）**

- **现象**：Mix block 的 Vector 程序在**两个 AIV 上重复执行**——profile 显示 `vector0` 与 `vector1` 的
  `GM_to_UB_datas` 几乎相同（45045 KB vs 43698 KB / 核），即 q/do 被装载两遍；而 `UB_to_GM_datas` 只在 vector0
  （16432 KB vs 0），说明原子写只落一份。
- **候选**：按 `subid` 把 q/do 行区间与 dq 原子环各切一半（PL-1.13 的形态）。
- **结果**：精度 **FAIL**（dq/dk/dv 全错，tier1 超差 12.5 万点）。
- **根因（探针 `probe_subid.py`）**：本工具链 **Developer 模式**下 `T.Kernel(n, is_npu=True) as (cid, subid)`
  的 `subid` **恒为 0**（写入 `1+subid` 到 `out[cid*2+subid]`，实测 `max(out) = 1.0`，从无 2.0）⇒ 两个 AIV 执行同一份
  代码（同样的 `subid=0` 分支），"切一半" 直接丢一半行。
- **结论**：PL-1.13 的 subid 分片**适用域进一步收窄为 Expert 双 Scope Mix kernel**；Developer 模式的 Mix kernel
  只能接受 Vector 程序被复制两次（AIV 侧 2× 冗余），无法用 `subid` 拆分。记为 §4.3 的负向结论。

### Round 5 — `r5-region-atomic`（结构：dq 回写改为 rank-2 跨步区域原子）

- **现象**：mte3（45%）全部来自**逐行** `T.atomic_add(dq[bz, c0+r, bx, 0:dim], dq_blk[r], [1, dim])`；
  微基准 `probe_atomic.py`（同字节、20 block）测出：逐行 1-D 形式 **224 ns/行**，rank-2 连续 2-D 区域形式
  **32 ns/行**，纯区域 store **7.5 ns/行** ⇒ 逐行形式是 **7× 单价惩罚**，属发射/描述符问题而非流量问题。
- **候选**：把 `dq` 形参声明为其**同一存储的 2-D 视图** `(B·S, H·D)`，回写变成
  `T.atomic_add(dq[flat : flat+bn, bx·dim : (bx+1)·dim], dq_blk, [bn, dim])`（一次 strided 2-D 区域原子）；
  工厂返回一个薄闭包，做 `dq.view((B*S, H*D))`，**对外签名不变**（同包 `_gqa_prefill_fwd_kernel` 已有闭包先例）。
  前置排除：rank-reduced 4-D 区域原子在本机仍**静默写错**（§4.2 重验），所以必须走 2-D 视图。
- **分支结果**（探针，2 例）：

| workload | current best (µs) | r5-region-atomic (µs) | Δ | AICore 利用率（r5） | memory | L0 |
|---|---|---|---|---|---|---|
| 8b-short/fp16 | 7106.1 | **3892.6** | −45.2% | cube 3.9% / vec 14.0% / mte2 12.5% / **mte3 4.6%** / scalar 64.2% | 2923.6 MB | PASS（check 与 baseline 逐位同值） |
| 8b-long/fp16 | 51315.3 | **28153.4** | −45.1% | cube 3.9% / vec 14.2% / mte2 12.6% / **mte3 4.2%** / scalar 60.0% | 20902.6 MB | 同上 |

- **判读**：mte3 忙时 3154.6→178.3 µs（8b-short），单点 45% 的墙被打掉；瓶颈转为 `aiv_scalar`（60–64%）。
- **winner**：保留。

### Round 6 — `r6-tuned`（参数：`bn` 32→64 @dim=128；trace-time 跳过无用的 tailoring `T.clear`）

- **现象**：R5 后 `aiv_scalar` 占 60–64%，且**每迭代固定 ≈5.3 µs**（与 tile 面积几乎无关）。
  同时 `aiv_scalar_wait_id7` ≈ 全时长 ⇒ 每迭代的固定同步/发射开销主导 ⇒ **减少迭代数**是第一杠杆。
  重新实测 UB 边界：`(bm=64, bn=64)@dim=128,causal` 声明 172 KB，**本工具链可编译**（旧 guard 的 1.4× 系数把
  `bn=64` 误挡；实测有效系数 ≤ 196608/176128 = 1.12）。
- **候选**：`TUNED_DEFAULT_CONFIG = {block_m:64, block_n:64, num_stages:1}`（模块级常量 + 工厂内对
  冻结默认三元组的替换，wrapper 无需改动）；并把 `T.clear(q_s/do_s/k_s/v_s)` 改为 trace-time 条件
  （`seq_len % block != 0` 时才需要，manifest 全为整除）。`ns=2` 实测无收益（14250 vs 14257 µs）。
- **分支结果**（8 例全测并写入 `phase=merged`）：

| workload | current best (µs) | r6-tuned (µs) | Δ | AICore 利用率（r6） | memory | L0 |
|---|---|---|---|---|---|---|
| 8b-short/fp16 | 3892.6 | **2275.8** | −41.5% | cube 5.3% / vec 15.7% / mte2 11.7% / mte3 5.1% / scalar 71.2% | 2005.2 MB | PASS |
| 8b-long/fp16 | 28153.4 | **14291.4** | −49.2% | cube 6.2% / vec 17.9% / mte2 14.0% / mte3 5.0% / scalar 62.0% | 14168.0 MB | PASS |

- **判读**：迭代数减半（`n_q` 32→16 / 64→32），每迭代成本只从 7.8→8.4 µs ⇒ 净 ~1.85×。
- **winner**：保留（本轮的 8 例测量作为 `phase=merged` 记录）。

### Round 7 — `r7-sync-phases`（结构：合并跨引擎同步相位）→ **REJECTED（回退）**

- **现象**：R6 的 `aic_scalar_wait_id8` = 14156 µs、`aiv_scalar_wait_id7` = 12189 µs ≈ 全时长 ⇒
  等待来自 **Cube↔Vector 的跨引擎同步往返**（每迭代 3 段 Cube 相位）。
- **候选**：① `do` 提前与 `q` 同批装载，使 g1/g2 背靠背（一次等待）；② 增加第二块 dtype scratch（`pcast2`）
  让 g3/g4/g5 合成**一个** Cube 相位；③ 为腾出 UB，把循环不变 `(j-i)` 平面改为"仅对角块内标量现建到 dsd"。
- **结果**：精度逐位相同（check 与 baseline 同值），但**变慢**：

| workload | current best (µs) | r7-sync-phases (µs) | Δ | AICore 利用率（r7） | memory | L0 |
|---|---|---|---|---|---|---|
| 8b-short/fp16 | 2275.8 | 2865.0 | **+25.9%** | cube 4.2% / vec 12.8% / mte2 9.1% / mte3 4.0% / scalar 78.1% | 2005.2 MB | PASS |
| 8b-long/fp16 | 14291.4 | 14737.7 | **+3.1%** | cube 6.0% / vec 18.1% / mte2 13.4% / mte3 4.8% / scalar 66.2% | 14168.0 MB | PASS |

- **根因**：②的"合并相位"并未减少实际等待（编译器按 buffer 粒度同步，g1/g2 本就只等 qkt），
  而③的对角块标量建平面在 short 档（`n_q=8`）摊销不足（每 task 4096 次标量 store 只摊到 8 个迭代）。
- **处置**：整体回退，current best 保持 r6。备注：②③两类形态保留在分支文件中存档，不再组合。

### Final（`perf_opt/_gqa_bwd_kernel.py` = r6 + trace-time 尾块回退）

在 r6 基础上只加两项**覆盖面**修补（对 manifest 形状零影响，已验证 final 测量与 merged 测量在噪声内一致）：

1. `_resolve_config`：`is_causal and dim>64 and seq_len % bn != 0` 的**尾块 causal** shape 回退到冻结默认
   `bn=32`（r6 的 tuned tile 在该档因 padding `T.clear` + body guard 触发 UB overflow：
   L2 `(2,300,4,4,128)` 实测 217 KB > 192 KB）。
2. `skip_causal` / `exact_tiles` 两个 trace-time 开关：body guard 与"仅对角块掩码"只在
   `seq_len % block == 0` 时启用（尾块 shape 走 baseline 形态，逐位结果不变、无块级跳过）。

修补后 `--level all` 全绿（§6.2），最终 8 例重新采 `phase=final`（§6.1）。

## 4. 诊断探针与负向结论（含旧条目的本机重验）

| # | 探针 | 结论（本机：910B4 / CANN 9.0.0 / `2fb23ee`） | 对旧知识的处置 |
|---|---|---|---|
| 4.1 | `probe_atomic.py`（4 形态，20 block，同字节） | 逐行 1-D 原子 **224 ns/行**；无 `size=` 同价（224 ns/行）；rank-2 连续 2-D 区域原子 **32 ns/行**；纯区域 store **7.5 ns/行** ⇒ 逐行原子 = 7× 单价。**推翻 DESIGN §1.6.0/§1.6.4 的「行原子单价≈整 op、不是流量问题」的量化判断**（单价差 7×，正是流量/描述符形态问题） | 新常数（§8 回写 constants） |
| 4.2 | `probe_atomic_region.py`（rank-reduced 4-D BSHD 目标） | 静态 extent：目标区 **256/2048 命中、区外 1792 cell 被写坏**；动态 extent：前端报错（`int() argument ... 'Var'`）⇒ **DESIGN §0.6 R3 的形态结论在本机依旧成立**（910B4 + CANN 9.0.0 重验通过） | 重验 `TRAP-*`/DESIGN 结论（未推翻） |
| 4.3 | `probe_subid.py` | Developer 模式 `T.Kernel(n, is_npu=True) as (cid, subid)` 中 `subid ≡ 0`（`max(out)=1.0`，从无 2.0） | **PL-1.13 适用域再收窄**：subid 分片只在 Expert 双 Scope Mix kernel 成立 |
| 4.4 | `probe_compile.py`（编译边界） | `(64,64)@dim128,causal` 编译通过（声明 172 KB，编译期有效系数 ≤1.12）；`(64,128)@dim128` 需 254 KB → 拒绝；`(64, 96)@dim128` 需 240 KB → 拒绝 | 新常数：UB 有效系数 ~1.12（而非旧 guard 的 1.4） |
| 4.5 | 双 AIV 冗余 | Mix block 的两 AIV `GM_to_UB` 相同（44 MB/核）⇒ q/do 被装载两遍；`UB_to_GM` 只在 vector0（原子只落一份） | Developer 模式 Mix kernel 的固有 2× AIV 冗余（无 subid 可分） |
| 4.6 | `CG-2026-0030` 形态② | `(B=1,H=64,D=128,S=2048,fp16)` **编译并运行正常**（baseline 与 final 均有效 profile） | 910B4 + CANN 9.0.0 **未复现**；该 gap 的设备/版本适用范围收窄（保留原条目不改写） |
| 4.7 | 掩码代数形态 | "预乘平面 + 标量加"的两 op 原位形态**数值错误**（未夹紧的正向项使 `exp` 上溢 → NaN）；夹紧必须保留 `min(·,0)`，即需要一块 `[bn,bm]` 暂存 | 负向形态（记入 §8，避免后人重试） |

其他已重验并沿用的旧结论：`T.gemm` size 顺序 `[m,k,n]`（沿用，未变）；`T.Pipelined` 的 runtime start 与
dynamic `size=` 仍不可用（本轮未再触发）；`TRAP-parallel-affine-cast-plane`（平面构造只用嵌套 `T.serial`）；
`TRAP-vec-cube-ub-scratch-role-conflict`（`dsd` 与 `dsum` 分离，本轮未改角色）；
`TRAP-gemm-dynamic-size-cbuf`（GEMM 全 tile extent + 操作数预清零，尾块动态性只留在 copy 与原子）。

## 5. 中止条件判定

- **success / best_effort**：无硬性数值目标；8 个 tune workload 全部有调优结论与最终记录，
  smoke 精度回归通过，`--level all` 全绿 ⇒ 按 skill「所有 tune workload 有结论 + 最终全量复测」收束。
- 预算：结构/参数候选实验 **9 个**（r1、r2、r3、r4、r5、r6 + 3 个参数探针 bn64/ns2/bm96）≤ `max_stage4_experiments = 16`；
  轮次 7 ≤ `max_rounds = 10`。
- 未走 plateau（每轮均有 >3% 的有效提升，直到 R6）；R7 为负向结果，随后收束（剩余候选均已实测：ub 上探被 UB 挡、
  subid 被工具链挡、同步相位合并反向）。

## 6. Final Summary

最终交付：`perf_opt/_gqa_bwd_kernel.py`（SHA256 `8344d214bb04bac1c73e23af1f315e577de12be086b6fc66c80c5f5808b82a89`），
`candidate_id = final-r6-tuned`，tile `(block_m, block_n, num_stages) = (64, 64, 1)`，
结构变更一句话：**causal 块级跳过 + `by` 最慢的任务映射 + 仅对角块掩码 + dq 走 2-D 视图的 rank-2 区域原子 + bn=64**。

### Final Performance Test Data（round 7 历史记录 —— 标题已加注记以免被 gate 4 的机械对账当作「当前最终表」；
> 本节的数值与表格**未改动**，仍为 r6 交付件的 round-7 final 测量值。当前最终表见文末
> §12「Final Performance Test Data」，对应交付件 `perf_opt/_gqa_bwd_kernel.py` sha256 `6ca13f73…`。）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---|---|---|
| _gqa_bwd_main | llama-3.1-8b-short/fp16 | 12068.26 | 2267.49 | final-r6-tuned |
| _gqa_bwd_main | llama-3.1-8b-short/bf16 | 12063.06 | 2290.52 | final-r6-tuned |
| _gqa_bwd_main | llama-3.1-8b-long/fp16 | 94099.52 | 14224.92 | final-r6-tuned |
| _gqa_bwd_main | llama-3.1-8b-long/bf16 | 93891.65 | 14254.17 | final-r6-tuned |
| _gqa_bwd_main | llama-3.1-70b-short/fp16 | 12053.35 | 2262.29 | final-r6-tuned |
| _gqa_bwd_main | llama-3.1-70b-short/bf16 | 12057.79 | 2278.15 | final-r6-tuned |
| _gqa_bwd_main | llama-3.1-70b-long/fp16 | 94076.99 | 14231.43 | final-r6-tuned |
| _gqa_bwd_main | llama-3.1-70b-long/bf16 | 93921.49 | 14228.76 | final-r6-tuned |

表内时延 = `msprof op` `Task Duration(us)`（20 次采样中位数），逐行与 `perf_records.jsonl` 的 `phase=final`
记录完全一致；raw profile 在 `profiles/final/<label>_<dtype>/OPPROF_*`。

### 6.1 提升幅度与 final 画像

| kernel_id | workload_id | 提升 | final AICore 利用率（占 Task Duration） |
|---|---|---|---|
| `_gqa_bwd_main` | `llama-3.1-8b-short/fp16` | **5.32×** | cube 5.4% / vec 15.8% / mte2 11.9% / mte3 5.1% / scalar 71.8% |
| `_gqa_bwd_main` | `llama-3.1-8b-short/bf16` | **5.27×** | 同上（vec 16.1%） |
| `_gqa_bwd_main` | `llama-3.1-8b-long/fp16` | **6.62×** | cube 6.2% / vec 18.0% / mte2 13.9% / mte3 5.0% / scalar 62.6% |
| `_gqa_bwd_main` | `llama-3.1-8b-long/bf16` | **6.59×** | 同上（vec 18.5%） |
| `_gqa_bwd_main` | `llama-3.1-70b-short/fp16` | **5.33×** | 同 short 档 |
| `_gqa_bwd_main` | `llama-3.1-70b-short/bf16` | **5.29×** | 同上 |
| `_gqa_bwd_main` | `llama-3.1-70b-long/fp16` | **6.61×** | 同 long 档 |
| `_gqa_bwd_main` | `llama-3.1-70b-long/bf16` | **6.60×** | 同上 |

现状瓶颈：**每迭代固定的发射 + 跨引擎同步开销**（`aiv_scalar` 占 62.6–71.8%，其中 `aiv_scalar_wait_id7`
≈ 12.2 ms ≈ 86% 时长；`aic_scalar_wait_id8` ≈ 14.2 ms ≈ 全时长）。即 tile 面积已到 UB 上限、原子与装载形态已优化后，
剩余空间在「每迭代 ~5.3 µs 的固定成本 × 迭代数」，需要 Expert 双 Scope 的跨迭代流水（本轮已验证 R7 的
"合并相位"无效，见 §3 Round 7）。

### 6.2 smoke 与最终精度回归

| 项 | 结果 |
|---|---|
| `mha-bwd-smoke-s512-h8-d64/fp16` | PASS（tier-1，max_diff 7.57e-4 / 9.13e-4 / 8.67e-4，0 超差） |
| `mha-bwd-smoke-s512-h8-d64/bf16` | PASS（**tier-3**：对 fp32 golden 17/786432 = 0.0022% 元素超 tier-1/2，对**源忠实参考** max 2.73e-4、0 超差 ⇒ 算法固有 P/dS 回落噪声；与 baseline 逐位同值，非回归） |
| `python perf_opt/_gqa_bwd_kernel.py --level L0` | **ALL PASS (16 items)**，`logs/l0/final_all_v4.log` |
| `--level all` | **L0 16/16 PASS，L1 10/10 PASS**（`L1/causal-tail-bf16` 走 tier-2 2-ulp），**L2 6 PASS / 1 WARN**，**Boundary 1 PASS / 1 WARN**，`All check passed!` |
| L2 WARN | `(1,32,4,4,64)`：`bishengir-compile` 段错误 —— 已登记的 `TRAP-bishengir-crash-s32`（工具链缺陷，与精度无关；Stage 3 同样存在） |
| Boundary WARN | ×100 输入量级（`q,k,v,do ~ N(0,100)`）下 fp32 灾难性抵消，与源忠实参考偏离 ~1%，记录不断言（Design §8.2 定级） |
| **wrapper 直插验证**（非 manifest，`(1,512,8,64)` fp16，`MultiHeadAttentionBwdOp` 三件套全走） | `dq` max\|base−tuned\| = **1.22e-4**（原子累加序带来的末位差），`dk`/`dv` **逐位相同** ⇒ `perf_opt/_gqa_bwd_kernel.py` 可被 wrapper 的 baseline/perf_opt 切换块直接替换（`_gqa_bwd_kernel` 工厂与 9 参调用签名不变；`dq` 的 2-D 视图由工厂闭包内部完成） |

### 6.3 已知残留 / 未验证项 / 风险（如实披露）

1. **AIV 双核 2× 冗余未消除**（§4.5）：Developer 模式无法用 `subid` 拆分（§4.3），AIV 侧 q/do 装载与向量计算
   各做两遍。若后续切 Expert 双 Scope（`PL-1.16` 路线）可再拿 ~20–30%，属**结构级后续工作**，本轮未做。
2. **尾块 causal（dim=128，`S % 64 != 0`）回退到 `bn=32`**：功能/精度正常，但不享受 tuned tile 与 causal
   块级跳过（非 manifest 形状，无性能承诺）。
3. **`bn=80/96` 不可行**（UB：分别需 199/222 KB）——dim=128 的 tile 面积已到本工具链上限；`bm` 上探同样被
   `dv_acc+dk_acc` 的 fp32 驻留挡住（DESIGN §4.5 的结论在本机成立）。
4. **`CG-2026-0030` 形态②在本机未复现**（§4.6）：不等价于该缺陷已修复（设备/版本不同），风险如实保留。
5. **未做 tile-level 的 autotune**：搜索空间只有 `(bm,bn,ns)` 的三档有效组合，已手动逐点实测（§3），
   未使用 `tilelang.autotune`（无意义）。
6. `L2`/`Boundary` 的 WARN 均非本轮引入（S=32 崩溃与 ×100 量级噪声在 Stage 3 已记录）。

## 7. Skill Retrospective

### 7.1 skill 流程问题

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 优化方法学 | **"发射受限"必须先分辨三种单价**：逐行 op 单价、区域 op 单价、跨引擎同步往返单价。三者在本任务相差 7×/1.4µs 量级，纸面模型（"原子≈store"）会把方向带偏 | `probe_atomic.py`（224 vs 32 vs 7.5 ns/行）；R5 一击 −45%，而 DESIGN 判定的第一杠杆（tile 面积）只在 R6 才生效 | `iteration-diagnosis.md` 增加"单价分解三问"：① 写出是逐行还是区域？② 每迭代有几个跨引擎相位？③ 每迭代固定开销 / tile 元素数 | P |
| 优化方法学 | **causal 跳过的收益必须与静态任务映射一起评估**：单独加 body guard 只拿到 1.31×（理论 1.78×），差额正好等于映射不均的 1.35× | R1 vs R2（9195→7254 µs） | `bottleneck-patterns.md` 的 causal 条目补"负载模型：task 工作量随 `by` 单调，需 `by`-slowest 或 LPT 映射" | P |
| 测量口径 | `msprof op --kernel-name=<前缀>` 可匹配 `_gqa_bwd_main_mix_aic`；但 op 级 benchmark（`tileops.reporting.cli`）的 kernel 名解析只捕获 preprocess —— **per-kernel 采集脚本是必须的**（本任务自建 `bench.py`/`run_measure.py`） | `run_measure.py` 8/8 valid；op 级 bwd 数字不可用（conductor 已测） | `profile-collection.md` 补"op 级多 kernel 算子必须自建 per-kernel 采集" | R |
| 契约 | **形参 rank 可以是"视图"而非契约**：把 `dq` 声明为 `(B·S, H·D)` 并用工厂闭包 `view` 传入，可解锁 rank-2 区域原子且对 wrapper 零改动（同包已有闭包先例） | R5（−45%）；`probe_atomic_region.py` 证明 4-D 形态不可用 | `bottleneck-patterns.md`/`traps-compiler.md` 增加"跨步大张量写出：优先 rank-2 视图 + 闭包，而非逐行 op" | P |
| 陷阱域 | `PL-1.13` 的 subid 分片域需再收窄到 **Expert 双 Scope**；Developer 模式 `subid ≡ 0` | `probe_subid.py`（max=1.0）；R4 精度 FAIL | 更新 `attention.md` PL-1.13 的适用域一句 | D |

### 7.2 Value Point Proposals（含证据三件套）

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **dq 回写形态的单价阶梯**：逐行 1-D 原子 224 ns/行 → rank-2 连续 2-D 区域原子 32 ns/行 → 纯区域 store 7.5 ns/行（同字节、20 block）；rank-reduced 4-D 区域原子仍静默写坏（256/2048 in、1792 out） | D | §3 Round5、§4.1/§4.2；final 8b-short 12068→2267 µs | `repro/CONST-atomic-writeback-forms.py`、`repro/TRAP-region-atomic-4d-bshd.py` | 910B4 / CANN 9.0.0 / tilelang `0.1.2+2fb23ee` / 2026-10-18 |
| **跨引擎同步往返单价 ≈1.4 µs/相位**：64×64 tile、每迭代 3 段 Cube 相位时 `aiv_scalar_wait_id7` ≈ 全时长；把相位并为 2 段未降时延（编译器按 buffer 同步），但每迭代固定开销 ≈5.3 µs 与 tile 面积无关（bn 32/64 相同） | D | §3 Round6/7 的 `*_wait_id7/id8` 与迭代数×时延关系 | `repro/PL-aiv-fixed-per-iteration-cost.py`（compile+msprof 打印两 tile 的每迭代时延） | 同上 |
| **Developer 模式 Mix kernel 的 Vector 程序在双 AIV 上复制执行且 `subid ≡ 0`** ⇒ 无 subid 分片；判据 = 两 AIV 的 `GM_to_UB_datas` 相同、且探针 `1+subid` 恒 1.0 | D | §4.3/§4.5；R4 精度 FAIL | `repro/TRAP-dev-subid-zero-mix.py` | 同上 |
| **causal 跳过的负载模型**：`n_q − ⌊by·bm/bn⌋` 递减 ⇒ 必须把 `by` 放最慢下标（否则 max/mean 1.35×，吃掉一半跳过收益） | P | R1 9195 → R2 7254 µs（short）、53995→51622（long） | `repro/PL-causal-skip-load-balance.py`（解析模拟 + 两映射实测） | 同上 |

## 8. Pattern-library 回写与 repro（ED-C）

本轮新增 4 个条目（均带 `origin_task` 与工具链版本戳，`kb_lint.py` 0 failure），已追加到
`.agents/skills/tilelang-op-optimize/references/pattern-library/`：

| 条目 ID | 落点文件 | repro |
|---|---|---|
| `CONST-atomic-writeback-forms` | `constants.md` | `repro/CONST-atomic-writeback-forms.py` |
| `TRAP-region-atomic-4d-bshd`（910B4 重验） | `traps-compiler.md` | `repro/TRAP-region-atomic-4d-bshd.py` |
| `TRAP-dev-subid-zero-mix` | `traps-compiler.md` | `repro/TRAP-dev-subid-zero-mix.py` |
| `PL-causal-skip-task-mapping` | `attention.md` | `repro/PL-causal-skip-task-mapping.py` |

`INDEX.md` 的对应路由表已同步 4 行。repro 的测量协议与重验触发词写在各自文件头。

---

## 9. Round 8（2026-10-19 续跑 round2：Expert 双 Scope + subid 分片）→ **BLOCKED（工具链缺陷）**

> 本轮回溯目标：消除 `_gqa_bwd_main` 的 AIV 双核 2× 冗余（PL-1.16 Expert 双 Scope + PL-1.13 subid
> 行分片），目标再拿 20–30%。**结论：本工具链的 Expert + L0C 累加 lowering 路径不可用**（5 类
> SIGSEGV + 1 类静默误编译），结构性杠杆本轮不可达；final 保持 r6，未改动。

### 9.1 工具链重戳（本轮实测，不沿用旧戳）

| 项 | 本轮实测值 |
|---|---|
| 设备 | Ascend910B4（单卡，`ASCEND_RT_VISIBLE_DEVICES=0`） |
| AICore | `NPUUtils.get().get_aicore_num()` = **20**（AIV 40，1 block = 1 Cube + 2 AIV） |
| tilelang | `0.1.2+06e1a1aefb82a00cd203a8b691aff41d7e3c6fbc`（dev root build，`git log -1` = `06e1a1a`，2026-10-19 01:20 UTC） |
| CANN | 9.0.0（`bishengir-compile` = `/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile`） |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| UB | 196608 B（192 KB） |
| 唯一时延口径 | `msprof op --kernel-name=_gqa_bwd_main` 的 `Task Duration(us)`（20 采样中位数，串行独占） |
| kb_stale_check | `stale_count=369`（current stamp `tilelang commit 06e1a1a`） |
| 工具链 delta（vs 上轮 `2fb23ee`） | `tilelang/jit/jit_npu.py` 新增 `syn_instr_mode -> sync_instr_mode` 文本 shim（CANN 9.0.0 的 `sync_block_set` 汇编关键字改名） |

**上轮“Expert 双 Scope 可能编译不过”的旧结论本轮重验**：同族 Expert 产物
`perf_opt/_gqa_prefill_fwd_kernel.py`（`T.Scope("Cube"/"Vector")` + `T.sync_block_*` + subid 分片）
在**本机新工具链**上 `TILELANG_DISABLE_CACHE=1` 冷编译通过（7.3 s，
`logs/r8a/prefill_nocache.log`）⇒ **Expert 模式本身可用**，旧结论不再成立；受阻点是本 kernel
的 op 组合（见 §9.4）。

### 9.2 本轮 base 与同 run 复核（round 9，`phase=candidate`）

base 固定为 `perf_opt/_gqa_bwd_kernel.py`（= r6-tuned，`sha256 8344d214…a89`，本轮未修改），
并另存 `perf_opt/_gqa_bwd_kernel_r6_baseline.py` 作为 A/B 固定 arm（与 base 逐字节相同）。
A/B 退化为**同件复测**（两 arm 同一 sha），用于确认工具链升级未改变性能：

| kernel_id | workload_id | current best（round-8 final，us） | r6-final-recheck-r8（round 9，us） | Δ | round-9 AICore 利用率（占 Task Duration） | memory（GM r+w） | L0 |
|---|---|---|---|---|---|---|---|
| `_gqa_bwd_main` | `llama-3.1-8b-short/fp16` | 2278.15 | **2275.59** | −0.11% | cube 5.3% / vec 15.7% / mte2 11.7% / mte3 5.1% / scalar 71.7% | 2005.2 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-8b-short/bf16` | 2294.34 | **2282.06** | −0.53% | 同上（vec 16.2%） | 2005.2 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-8b-long/fp16` | 14306.68 | **14276.96** | −0.21% | cube 6.2% / vec 17.9% / mte2 13.9% / mte3 5.0% / scalar 62.4% | 14168.0 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-8b-long/bf16` | 14310.63 | **14296.00** | −0.10% | 同上（vec 18.5%） | 14168.0 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-short/fp16` | 2275.55 | **2277.15** | +0.07% | 同 short 档 | 2005.2 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-short/bf16` | 2284.34 | **2283.76** | −0.03% | 同 short 档 | 2005.2 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-long/fp16` | 14304.57 | **14255.62** | −0.34% | 同 long 档 | 14168.0 MB | PASS |
| `_gqa_bwd_main` | `llama-3.1-70b-long/bf16` | 14313.06 | **14269.47** | −0.30% | 同 long 档（vec 18.5%） | 14168.0 MB | PASS |

- 8/8 `valid`；全部 |Δ| ≤ 0.53%（噪声阈值 3%）⇒ **不记 winner**，final 不变。
- 环境：本 arm 显式 `TILELANG_ASCEND_MODE=Developer`（`bench.py` 与 kernel 模块只用 `setdefault`）；
  raw profile `profiles/r8_recheck/<label>_<dtype>/OPPROF_*`，日志 `logs/r8_recheck/`。
- 记录：`perf_records.jsonl` 追加 round 9 共 8 行（`candidate_id=r6-final-recheck-r8`，
  `phase=candidate`，`artifact_sha256=8344d214…`）；**未新增 `phase=final` 行**（final 行仍为
  round 7/8 两组、同一 sha，append-only 不改写）。

### 9.3 本轮现象（base profile 重读，long/fp16）

| 分区 | 忙时/时长 | 每迭代折算（1683 iter/核） | 判读 |
|---|---|---|---|
| Task Duration | 14277 µs | **8.48 µs/iter** | — |
| `aic_cube` | 882 µs（6.2%） | 0.52 | 5 个 GEMM 合计仅 0.5 µs ⇒ Cube 不是算力瓶颈 |
| `aiv_vec` | 2557 µs（17.9%） | 1.52 | Vector 实算 |
| `aiv_mte2` | 1978 µs（13.9%） | 1.18 | q/do 跨步行装载（两 AIV 各一份）+ ws 读 |
| `aiv_mte3` | 713 µs（5.0%） | 0.42 | dq 区域原子 + dv/dk 回写 |
| `aiv_scalar` | 8914 µs（62.4%） | **5.30** | 其中 `aiv_scalar_wait_id7` ≈ 4.5 µs/iter（跨引擎等待）+ ~0.8 µs 发射 |

⇒ AIV 侧三条 pipe 忙时合计 3.11 µs/iter 已与 8.48 µs 的时长不符：**AIV 的时长被
4.5 µs/iter 的跨引擎等待占据**（Cube 侧 `aic_scalar_wait_id8` 同样 ≈ 全时长）。可动的两个量：
① AIV 实算 3.11 µs（含双 AIV 2× 冗余，subid 分片可减半）；② 4.5 µs/iter 等待（需手动相位 /
跨迭代流水）。两者都要求 Expert 双 Scope（Developer 下编译器按 buffer 粒度同步，R7 已实测
合并相位无效）。**这正是本轮唯一候选方向。**

### 9.4 候选 vs current best（B2 对比表）

| # | 分支 | 唯一主要优化点 | 编译 | 精度（L0 口径） | Task Duration(us) | AICore/memory | 判定 |
|---|---|---|---|---|---|---|---|
| r8a | `_gqa_bwd_kernel_r8a_expert_dualscope.py` | Expert 双 Scope（Cube 5 GEMM / Vector 链 + dq 原子）+ 3 段手动相位 | **PASS**（唯一可编译形态） | **FAIL**：`dk` 相对误差 ~1e0（31.5% 元素超 tier-1/2/3；max 7.59），`dq`/`dv` 正常（4.0e-4 / 8.4e-4） | 无（精度失败不采） | 无 | **REJECTED**（静默误编译，见 §9.5） |
| r8a-v2 | 同上 + `g4` 顺序前置 | 同上 | **SIGSEGV**（bishengir） | — | — | — | BLOCKED |
| r8a-v3 | 同上 + 每 GEMM 独立 `PIPE_C` frame | 同上 | **SIGSEGV** | — | — | — | BLOCKED |
| r8a-v4 | 同上 + epilogue fixpipe 顺序互换 | 同上 | **SIGSEGV** | — | — | — | BLOCKED |
| r8a-v5 | 同上 + `g4` 独立 A 操作数（`l1_p2`） | 同上 | PASS | **FAIL**（dk 同值错误） | 无 | 无 | REJECTED（非 A 操作数共享问题） |
| r8a-v6 | 同上 + dv/dk 回写用两块 UB | 同上 | **SIGSEGV** | — | — | — | BLOCKED |
| r8a-v7 | 同上 + dv/dk 走 2-slot 连续 ws 中转（绕 strided fixpipe） | 同上 | PASS（compile4.log） | FAIL（同 v1 的 dk） | 无 | 无 | REJECTED |
| r8b | `_gqa_bwd_kernel_r8b_expert_concat.py` | 单个累加 GEMM `[dv｜dk] = Pᵀ[dO｜Q]`（N=256），规避“第二个 L0C 累加器” | **SIGSEGV** | — | — | — | BLOCKED |
| r8c | 跨迭代 2 深流水（Cube 预发 S(i+1)） | 相位重叠 | 未进入（前置 r8a 不可用） | — | — | — | BLOCKED（依赖 r8a） |
| r9 | `_gqa_bwd_kernel.py`（未改） | 复核臂 | PASS | PASS（同 sha 历史 L0/L1 + 本轮 `--level all`） | 见 §9.2 | 见 §9.2 | 无提升，保持 |

`perf_records.jsonl` 对应行：round 9 × 8（r9）。**r8a/r8b 无 `Task Duration` 行**——编译失败或
精度失败的分支不产生性能记录（不写无效数字），证据全部在 §9.5 的日志路径里。

### 9.5 Expert 失败的根因证据（全部本机实测，`TILELANG_DISABLE_CACHE=1` 冷编译）

| # | 触发形态（均为 docs 合法写法） | 失败签名 | 证据 |
|---|---|---|---|
| E1 | `T.alloc_var("float32")` 赋值后作为 v-op 标量操作数（R6 的 `d0` 形态；`docs/Tilelang.language/数学操作` 的 scalar operand 用法） | **tilelang device_codegen SIGSEGV**（`tilelang/engine/lower.py:176` → `target.build.tilelang_npuir_apis`） | `logs/r8a/compile2.log`；最小化：`logs/r8a/probe_grid/f_vbrc.log`、`f_maskchain.log` |
| E2 | 列向量（`[half,1]`）作为二元 v-op 操作数（`T.vsub(qkt, lse_col, qkt)`） | **bishengir legalize 失败**：`failed to legalize operation 'memref.alloc'` | `logs/r8a/probe_grid/f_bcastsub.log`；绕法：先 `T.vbrc` 物化（`g_vbrc_lse.log` PASS） |
| E3 | 只读未写的 UB buffer 作 v-op 输入/输出 | 同 E2（`memref.alloc` 非法） | `logs/r8a/probe_grid/g_mask_only.log` |
| E4 | Expert 内 `T.gemm(..., a_transpose=True, initC=(it==0))` 跨 `T.serial` **L0C 累加**（`docs/Tilelang.language/内存操作/T.alloc_L0C.md` §2.4 的官方累加写法） | **bishengir SIGSEGV**（N=1 累加器）/ N=2 最小形态数值正确（rel_err 1.6e-07）/ **真实 kernel 形态下第二个累加器静默错**（dv 对、dk 错） | `logs/r8a/probe_acc_1.log`、`probe_acc_2.log`、`logs/r8a/try2.log` |
| E5 | 由 L0C 直接 fixpipe 到**跨步 GM**（`dv[bz, r0:r0+bm, hkv, 0:dim]`，行步长 `Hkv*D`）且该 L0C 为活累加器 | **bishengir SIGSEGV** | `logs/r8a/compile3.log`；改为连续 ws 中转后恢复（`compile4.log` PASS） |
| E6 | 同一 `PIPE_C` frame 里两个共享 A 操作数的累加 GEMM（r8a 初版）/ 拆成两 frame / 交换顺序 / 交换 epilogue 顺序 / 独立 A 缓冲 / 两块 UB / 合并成 N=256 单 GEMM | 6 种变体中 5 种 **SIGSEGV**、唯一可编译者 **dk 静默错** | `logs/r8a/variants/*.py` + `compile*.log` |

**对照（证伪协议要求）**：同族 Expert 产物 `_gqa_prefill_fwd_kernel.py` 在本机冷编译 **PASS**
（`logs/r8a/prefill_nocache.log`，7.3 s）⇒ 「Expert 双 Scope 本身不可用」不成立；失败面限定为
**Expert + L0C 跨迭代累加 + 多引擎相位** 这一组合。已排除的自身错误（每项都有实测）：
操作数共享、rs-frame 粒度、epilogue 顺序、UB 复用、strided fixpipe、A 操作数缓冲复用。
**未排除且无法在本轮定位**：`a_transpose` 累加操作数的 L1A 载入与 `nd2nz` 布局在下一次
迭代前的依赖处理（`dk` 错值量级与正确值同阶、符号随机，符合“读到错误 P/ds 或部分累加”）。

工作假设（未实证，标注为假设）：`bishengir` 的 HIVM→LLVM 管线在 Expert 属性
（`syn_instr_mode` + 手动 `sync_block_*`）与 L0C 累加器的 fixpipe/mad 依赖推导同时出现时，
依赖图不完整；`device_codegen` 的 `alloc_var` 崩溃是前端 codegen 的空指针。

### 9.6 中止条件与预算

- **stop_reason = blocked**：唯一剩余结构候选（Expert 双 Scope / cross-iteration 流水）在本工具链上
  既不能编译（SIGSEGV）也不能得到正确结果（静默误编译），且 workaround 面已穷举（§9.5 E6 的 6 种
  变体 + E1~E5 的替换写法）。
- 未走 plateau（本轮为单轮结构尝试被工具链阻塞，不构成“3 轮无提升”）。
- 预算：本轮结构分支 **6 个**（r8a + 5 变体，另 r8b 合并形态）+ 诊断探针 **5 个**（E1~E5 最小化）
  = 11 ≤ `max_stage4_experiments=16`；round 9 完成 8/8 tune workload 复核。
- 未触发 `[DESIGN_LIMIT]`：瓶颈归因**不在算法/设计层**（DESIGN 的发射受限判定与 R6 的结构选择均
  未被推翻），而在编译器缺陷；故**不产出** `perf_feedback.md`。

### 9.7 Final Summary 补记（append-only，不改写 §6）

- 最终交付仍为 `perf_opt/_gqa_bwd_kernel.py` = **r6-tuned**，
  `sha256 = 8344d214bb04bac1c73e23af1f315e577de12be086b6fc66c80c5f5808b82a89`（本轮未修改）。
- 结构变更一句话（相对 Stage 3）：**causal 块级跳过 + `by` 最慢任务映射 + 仅对角块掩码 +
  dq 的 2-D 视图 rank-2 区域原子 + bn=64**（与 §6 相同；本轮无新增结构）。
- 逐 workload 时延：见 §9.2 的 round-9 复核列（同一 sha，8/8 valid，与 round-8 final 差 ≤0.53%）。
- 本轮 final 冻结说明：因产物未变，**不重采 `phase=final`**（避免同一 sha 出现第三组 final 行）；
  round-9 行以 `phase=candidate` 归档。若 conductor 需要新的 `phase=final` 组，请以续跑调度重采。
- 精度（本轮重新跑 `--level all`，`TILELANG_ASCEND_MODE=Developer`，`logs/r8_gate/all_v1.log`）：
  **L0 ALL PASS（16 items，1614 s）**、**L1 ALL PASS（10/10，278 s）**、L2 6 PASS / 1 WARN
  （`(1,32,4,4,64)` 的 `TRAP-bishengir-crash-s32`，Stage 3 已登记的工具链缺陷）、
  Boundary 1 PASS / 1 WARN（×100 量级输入的 fp32 抵消，Design §8.2 定级，非断言）、
  `All check passed!`；与 §6.2 的结论逐项一致 ⇒ **无精度回归**。
  L0-14 记录 `dq/dk/dv` 三次重复 `bit_equal_3runs=True`（`max_spread=0.0`）。
  smoke 两例本轮在 manifest shape `(1,512,8,64)` 上用模块自带分层门复核（`try_case.py`，CPU golden）：
  fp16 **PASS tier-1**（worst 9.129e-04）、bf16 **PASS tier-3**（over=17/786432 = 0.0022% ≤ 5%，
  对源忠实参考 max 2.728e-04、viol=0）——与 §6.2 记录的数值逐项相同，属算法固有 P/dS 回落噪声，
  非回归。（注：`bench.py --check` 只做 tier-1/2，bf16 会报 FAIL，判据以模块 `_run_case` 的
  tier-3 分类为准。）
- 残留（不变）：AIV 双核 2× 冗余未消除（`CG-2026-0039` + 本轮新增 `CG-2026-0040`）。

### 9.8 Skill Retrospective（Round 8）

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 结构迁移方法 | **Expert 迁移必须先跑 4 项前置探针再动手写 kernel**：`alloc_var` 标量、列向量 v-op 广播、只读 UB、L0C 累加 GEMM。四项任一失败即整个 Expert 形态不可用，而四项合计只需 ~6 次冷编译（约 5 min），比写完整 kernel 后逐个报错快 10× | 本轮 r8a 从 09:29 到 10:05 的 6 次失败全部落在 E1~E5 四类；最小探针 `probe_expert_acc_gemm.py` 直接给出 E4 结论 | `tilelang-mixcv-skill` / `PL-1.16` 条目补「迁移前置探针清单（4 项）+ 每项期望输出」 | P |
| 证伪协议 | **编译器结论必须区分「Expert 不可用」与「该 kernel 形态不可用」**：同族 prefill Expert 核冷编译 PASS，把结论收窄到 op 组合才避免误判整类 | `logs/r8a/prefill_nocache.log` vs `compile*.log` | `INDEX.md §2 证伪协议` 补「同族专家产物冷编译对照」作为判定步骤 | P |
| 测量/记录 | 同一 sha 的多组 `phase=final` 行会让 gate 的“每 workload 一条 final”语义含糊（历史遗留 2 组）。本轮改为 `phase=candidate` 归档复核，不动 final | `perf_records.jsonl` round 7/8 两组 final | `profile-collection.md` 明确「产物未变的复核记录用 `phase=candidate` + `candidate_id=*-recheck`」 | R |
| 工具链风险 | 冷编译（`TILELANG_DISABLE_CACHE=1`）必须作为 Expert 结论的前置条件：kernel cache 会命中旧工具链产物，把“能编译”假象带进来（本轮 prefill 首测 1.5 s 即缓存命中） | `logs/r8a/prefill_nocache.log`（7.3 s）vs 首次 1.5 s | `hardware-context.md` / `profile-collection.md` 补「编译器结论一律冷编译」 | R |

### 9.9 Value Point Proposals（含证据三件套）

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **Expert 模式 4 项前置探针**：① `T.alloc_var` 标量作 v-op 操作数 → device_codegen SIGSEGV；② `[n,1]` 列向量作二元 v-op 操作数 → bishengir `memref.alloc` legalize 失败（绕法 `T.vbrc`）；③ 只读未写 UB → 同上；④ L0C 跨迭代累加 GEMM → bishengir SIGSEGV（N=1）/ 真实 kernel 下第二累加器静默错 | D | §9.5 E1~E4；`logs/r8a/probe_grid/*`、`probe_acc_*.log`、`try2.log` | `probe_expert_acc_gemm.py`（`--acc 1/2`）、`probe_expert_dualscope.py` | 910B4 / CANN 9.0.0 / tilelang `0.1.2+06e1a1ae` / 2026-10-19 |
| **Expert 的 L0C 直接 fixpipe 到跨步 GM 会崩**：`T.copy(l0_c, dv[bz, r0:r0+bm, hkv, 0:dim])`（行步长 `Hkv*D`）在 Expert 内 SIGSEGV；改「L0C→连续 ws→Vector→GM」即恢复（每 task 一次，代价可忽略） | D | §9.5 E5；`compile3.log`（崩）vs `compile4.log`（过） | `_gqa_bwd_kernel_r8a_expert_dualscope.py`（ws 中转形态） | 同上 |
| **Expert 双 Scope 的 bwd 形态在本工具链 blocked**：6 种等价重排中 5 种 SIGSEGV、1 种 dk 静默错；同族 fwd Expert 核冷编译 PASS ⇒ 结论限定为「Expert + 多 L0C 累加 + 多引擎相位」 | C | §9.5 全表；`logs/r8a/variants/` | — | 同上 |

### 9.10 补充探针：Expert 形态的可达时延量化（conductor 补充调度）

> 目的：补齐 §9.6 缺的量化项——**Expert 单 AIV scope 形态到底能到多少时延**（即使该形态当前
> 精度不可用）。口径仍为 `msprof op --kernel-name=_gqa_bwd_main` 的 `Task Duration(us)`
> （20 采样中位数，`flock` 串行独占）。被测件 `_gqa_bwd_kernel_r8a_expert_dualscope.py`
> （sha `80defe90…`，编译 PASS 的形态；**precision = FAIL（dk silent miscompile）**，仅用于
> 结构上限量化，不可采纳）。

#### 9.10.1 原生 tuned tile（dim=128, (64,64)）不可测：编译崩溃

| 配置 | 结果 | 证据 |
|---|---|---|
| (bm=64, bn=64) @ dim=128（交付 tile） | **bishengir-compile SIGSEGV** | `logs/r8a_headroom/compiletest_64x64.log` |
| (bm=64, bn=32) @ dim=128 | SIGSEGV | `logs/r8a_headroom/compiletest_bn32.log` |
| (bm=32, bn=64) @ dim=128 | SIGSEGV | `logs/r8a_headroom/compiletest_32x64.log` |
| (bm=32, bn=32) @ dim=128 | **可编译可运行** | `logs/r8a_headroom/compiletest_32x32.log` |

⇒ 2 个指定 workload 在交付 tile 上**无法测**（`perf_records.jsonl` 中 round 10 的 2 行
`r8a-headroom-probe` / `status=invalid` / `duration_us=null` 即该次失败，如实保留）。

#### 9.10.2 唯一可测点：dim=128,(bm=32,bn=32)，同 tile 双臂同 run

| kernel_id | workload_id | r6（Developer，us） | r8a Expert（us） | headroom = 1 − r8a/r6 | 每迭代（iter/核 解析值）r6 → r8a | 记录 |
|---|---|---|---|---|---|---|
| `_gqa_bwd_main` | `llama-3.1-8b-short/fp16` | **7240.03** | **2441.91** | **66.3%** | 8.32 → **2.81 µs**（870 iter/核） | round 10 `r6-ref-32x32` / `r8a-headroom-probe`（`phase=probe`） |
| `_gqa_bwd_main` | `llama-3.1-8b-long/fp16` | **55059.79** | **18044.44** | **67.2%** | 8.27 → **2.71 µs**（6656 iter/核） | 同上 |

- 双臂 pipe 对照（short）：r6 `vec 605 / mte2 707 / mte3 307 / scalar 4473 µs` →
  r8a `vec 188 / mte2 337 / mte3 536 / scalar 897 µs`：**AIV 实算 ≈ −69%，跨引擎等待 ≈ −80%**；
  r8a 的 mte3 反而升高（dv/dk 每 task 经 ws 中转过写，§9.4 的 E5 绕法代价，已在预期内）。
- traffic：r6 3079.8 MB → r8a 1821.7 MB（short）、22983.0 → 12602.3 MB（long）——q/do 只装一份。

#### 9.10.3 第二数据点：tuned tile (64,64) @ dim=64（smoke shape，无性能记录）

| 臂 | workload | Task Duration(us) | 备注 |
|---|---|---|---|
| r6（Developer） | `mha-bwd-smoke-s512-h8-d64/fp16` | **177.65** | 交付形态 |
| r8a（Expert） | 同上 | **87.85** | precision FAIL |

⇒ headroom = 1 − 87.85/177.65 = **50.5%**（tuned tile 上的直接对照；smoke 案例按门禁**不写**
`perf_records.jsonl`，raw profile 在 `profiles/r8a_headroom_d64/`、`profiles/r6_ref_d64/`）。

#### 9.10.4 判读与结论

1. **20–30% 目标的空间确实存在**：未流水的 Expert 形态（3 段手动相位、subid 行分片、dv/dk 经 ws
   中转）在两个可测点分别为 **66.3% / 67.2%**（dim=128，(32,32) tile）与 **50.5%**（dim=64，
   tuned tile）。每迭代固定成本从 8.3 µs 降到 2.8 µs（−66%）——正是 §9.3 指认的 4.5 µs/iter
   跨引擎等待 + AIV 2× 冗余。
2. 该量化是**下界**：r8a 不做跨迭代流水（r8c 未实施，依赖一个精度正确的 base），把 Cube 的 S(i+1)
   预发、dq 延迟原子后还能再重叠一部分；反之，r8a 的 dv/dk 中转过写（mte3 +229 µs，short）与
   单 AIV 侧多出来的 ws 读写是其自身代价，正式实现可部分消除。
3. 因此结论形态为「**空间存在但被编译器缺陷阻断**」（`headroom 50–67% ≫ 15%`），不是「设计层
   天花板」，也不是「空间不足」：`perf_feedback.md` 依旧不产出（归因在编译器，且已按 AGENTS.md
   登记 `CG-2026-0040`）。
4. 量化数字已补入 `CG-2026-0040.blocked_algo`。若工具链修复 (a)/(d)/(e) 三项缺陷，应优先重跑
   Expert 双 Scope（r8a→r8c 路线），本文件给出可复现的量化基线。
5. 交付物未变：`perf_opt/_gqa_bwd_kernel.py` sha256 `8344d214bb04bac1c73e23af1f315e577de12be086b6fc66c80c5f5808b82a89`；
   未重跑 8 例全量、未新增变体。

---

## 10. Round 11（2026-10-22，round-3 结构级重写轮：engine-phase split）→ 采纳

> 本轮授权：结构级重构（kernel 拆分 / 多 kernel 串联 / workspace 中转 / 任务映射重写）。
> 唯一硬禁用形态 = Expert 双 Scope（`CG-2026-0040` / `TRAP-expert-l0c-accum-lowering`）——本轮**未重试**。
> 环境为**新器件新工具链**，全部常数本机重测（不跨机引用 910B4 口径）。

### 10.0 工具链重戳与 baseline 复核（本机实测）

| 项 | 本轮实测值 |
|---|---|
| 设备 | **Ascend910B2**（`npu-smi` 26.1.1，`HBM 65536 MB`） |
| 核数 | `NPUUtils.get().get_aicore_num()` = **24**（AIC）；纯 Vector kernel 每 block 占 **1 AIV**（探针：`T.Kernel(48)` → `block_dim=48`） |
| tilelang | `0.1.2+151c62af8b9c1e72d8372da736b1c3d8582b1022`（dev root build，git HEAD `151c62a`，2026-10-21） |
| CANN | 9.0.0（`msprof` / `bishengir-compile` 在 `/usr/local/Ascend/cann-9.0.0/bin`） |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| kb_stale_check | `stale_count=395`（全部 910B4 口径常数视为待重验，本轮逐项重测） |
| 测量口径 | `msprof op` `Task Duration(us)`；**拆分形态 = 一次 op 调用内全部 kernel 的 Task Duration 之和**（`run_measure_split.py`，同时保留每 kernel 分项行） |

**Baseline 臂 = r6（`perf_opt/_gqa_bwd_kernel_r6_baseline.py`，sha `8344d214…a89`）本机重测（`round=11`，`phase=baseline`，`candidate_id=r6-arm-r11`）**：

| kernel_id | workload_id | Task Duration(us) | AICore 利用率（占 Task Duration） | memory（msprof read+write） | L0 |
|---|---|---|---|---|---|
| `_gqa_bwd_main` | llama-3.1-8b-short/fp16 | **2011.6** | cube 4.6% / vec 13.6% / mte2 10.6% / mte3 4.5% / scalar 69.7% | 2005.3 MB | PASS |
| `_gqa_bwd_main` | llama-3.1-8b-short/bf16 | 2012.3 | 同上（vec 14.0%） | 2005.3 MB | PASS |
| `_gqa_bwd_main` | llama-3.1-8b-long/fp16 | **13954.2** | cube 4.8% / vec 14.0% / mte2 9.4% / mte3 3.9% / scalar 60.3% | 14168.0 MB | PASS |
| `_gqa_bwd_main` | llama-3.1-8b-long/bf16 | 13995.8 | 同上（vec 14.5%） | 14168.0 MB | PASS |
| `_gqa_bwd_main` | llama-3.1-70b-short/fp16 | 2017.1 | 同 short 档 | 2005.3 MB | PASS |
| `_gqa_bwd_main` | llama-3.1-70b-short/bf16 | 2013.1 | 同 short 档 | 2005.3 MB | PASS |
| `_gqa_bwd_main` | llama-3.1-70b-long/fp16 | 14018.5 | 同 long 档 | 14168.0 MB | PASS |
| `_gqa_bwd_main` | llama-3.1-70b-long/bf16 | 13982.0 | 同 long 档 | 14168.0 MB | PASS |

跨器件说明：r6 在 910B4（20 核）上是 14 277 µs（long/fp16），本机 24 核 13 954 µs；**但每迭代成本从 8.7 µs 升到 10.2 µs**（核数变多、单迭代更慢）——即本机不是等比例缩放，历史每迭代数据不可沿用。

### 10.1 本轮现象（base = r6，long/fp16，`profiles/r11_baseline/`）

| 分区 | 忙时 / 时长 | 每迭代折算（1365 iter/核） | 判读 |
|---|---|---|---|
| Task Duration | 13 954 µs | **10.22 µs** | — |
| `aiv_vec` | 1956 µs（14.0%） | 1.43 | 向量实算 |
| `aiv_mte2` | 1309 µs（9.4%） | 0.96 | q/dO 跨步行装载（双 AIV 各一份） |
| `aiv_mte3` | 539 µs（3.9%） | 0.39 | dq 区域原子 + dv/dk 回写 |
| `aic_cube` / `mte1` / `mte2` / `fixpipe` | 674 / 922 / 1044 / 1828 µs | 0.49 / 0.68 / 0.77 / 1.34 | 5 个 GEMM + 操作数装载 + L0C→UB |
| **`aiv_scalar_wait_id7`** | **12 257 µs（87.8%）** | **8.98** | AIV 等 Cube |
| **`aic_scalar_wait_id8`** | **13 607 µs（97.5%）** | **9.97** | Cube 等 AIV |

⇒ 两引擎各以 ~90% 时间**互相等待**，而引擎实忙只有 AIV 4.05 ms、Cube 4.47 ms ⇒ **每迭代 ≈7 µs 是纯跨引擎 flag round-trip**（`CONST-crossengine-handoff-910B2`）。本轮的全部候选都围绕这一项展开。

### 10.2 候选 vs current best（B2 对比表，`perf_records.jsonl` round 11）

| # | 分支（文件） | 唯一主要优化点 | 编译 | L0 精度 | Task Duration(us) short / long | AICore / memory | 工作量守恒 | 判定 |
|---|---|---|---|---|---|---|---|---|
| r6 | `_gqa_bwd_kernel_r6_baseline.py` | 本轮固定臂（只读） | PASS | PASS | 2011.6 / 13954.2 | 见 §10.0 | — | 基线 |
| r10a | `_gqa_bwd_kernel_r10a_phase3.py` | 三相位合并（g1,g2 背靠背 → 单条向量链 → g3,g4,g5；`dsd` 省掉换 `pcast2`） | PASS | PASS（smoke 与 r6 逐位同值） | 1980.3 / 13992.3 | vec 2042 / mte2 1465（long）；traffic 14168 MB | 算术与字节数不变 | **REJECTED**（−1.6% / +0.3%，噪声内；证实相位段数不是杠杆） |
| r10c | `_gqa_bwd_kernel_r10c_planehoist.py` | `(j-i)` 平面从「每 task 建一次」提到 task 循环外（task 不变式） | PASS | PASS | **1927.4** / 13902.3 | long: scalar 8652→8231 | 每年减少 43–86 次 4096 标量 store | **保留**（short −4.2%，long −0.4%） |
| r10d | `_gqa_bwd_kernel_r10d_guards.py`（诊断） | 关掉两个运行时守卫（迭代 ×2） | PASS | 未改语义（真实计算） | — / 26680.5 | 每迭代 9.77 µs | 迭代数 ×2 | **诊断结论**：守卫不是固定成本来源 |
| r10e/r10f | `_gqa_bwd_kernel_r10f_bar.py` → 交付件 | **引擎相位拆分**（3 kernel + 分块 GM workspace + 每迭代 `pipe_barrier`） | PASS | PASS | **1012.9** / **7472.9**（探针；交付件重编译后 1016.8 / 7518.6，差 ≤0.6% 属 run 噪声） | 见 §10.4 | 见 §10.5 | **WINNER** |

- 未采性能记录的分支：r10a/r10c/r10d 的探针行以 `phase=probe` 归档（r10a 2 行、r10c 2 行、r10d 1 行）；r10e/r10f 的失败诊断轮不产生有效时延（见 §10.3）。
- **单点纪律**：r10a 只改相位顺序；r10c 只改平面构建位置；r10f 只改引擎相位归属（拆分），三者互不混合。

### 10.3 拆分形态的实现与一个必须记录的编译器缺口

结构（交付件 `perf_opt/_gqa_bwd_kernel.py`，tile-exact shape 走拆分、尾块 shape 保留融合路径）：

```
pass 1 `_gqa_bwd_sdw`  Cube only    S = Q Kᵀ, dW = dO Vᵀ      → fp32 workspace
pass 2 `_gqa_bwd_pds`  Vector only  P = exp(S·s − lse + pen),
                                    dS = P·(dW − delta)·s     → dtype workspace（同一回落）
pass 3 `_gqa_bwd_mul`  Mix          dv += PᵀdO, dk += dSᵀQ,
                                    dq = dS·K + rank-2 区域原子
```

三处工程约束（都是实测撞出来的）：

1. **KV 轴分块 + `by0: T.int32` 标量入参**：workspace 按 `cb` 个 kv 块有界（预算 256 MB），每个 `(b,h,kv块)` 任务完整落在一个 chunk 内 ⇒ `dk/dv` 保持单写者（否则 `n_chunks` 倍原子放大：long 档 +2.2 GB 流量）；chunk 偏移用标量入参传（同 `docs/Tilelang.language/原子操作/T.atomic_add.md` 形态），**一次编译跑所有 chunk**（否则 16 chunk × 3 kernel 的重编译不可接受）。
2. **每迭代末 `T.pipe_barrier("PIPE_ALL")` 是正确性必需项**：首版（无 barrier）端到端 `dq` max_diff 11.15 / `dk` 6.54，而 `dv` 逐位正确；逐瓦片对拍定位为 **workspace 512 块里 91 块（17.8%）的 dS 错**、其上游 P 逐位正确；加 barrier 后 **0/512**，端到端与融合基线**逐位同值**（7.570e-04 / 9.129e-04 / 8.669e-04）。根因 = 纯 Vector kernel 内 UB scratch 的**跨迭代 WAR**（vec 写 vs mte3 读）未被自动同步覆盖（融合形态每迭代末有跨引擎同步自然排空，故不复现）。已登记 `TRAP-npuir-vector-kernel-store-war-race`。
3. **纯 Vector kernel 的并行度**：每 block 只占 1 AIV（`T.Kernel(48)` → `block_dim=48`），故 pass 2 用 `2×cores` 核；`subid` 仍恒 0（`TRAP-dev-subid-zero-mix` 不变）。

### 10.4 Final Performance Test Data（交付件，8 例，多 kernel 求和口径）

msprof op 口径：**一行 = 一次 op 调用内 3 个 pass kernel（× `n_chunks` 次 chunk）的 `Task Duration` 之和**；
括号内为该 pass 的每-launch 中位时延。raw profile：`profiles/r11_final_v2/`；记录：`perf_records.jsonl`
（`round=11`、`candidate_id=final-r11-split-v2`、`phase=final`，1 条聚合行 + 3 条分项行/workload）。
交付件 sha256 = `6ca13f73ccb22ffbc9258d957320bae9d93105254ddc11d03c778d91bde92179`。

| kernel_id | workload_id | baseline_us | final_us | Δ | 分项（sdw / pds / mul，µs/launch） | n_chunks |
|---|---|---|---|---|---|---|
| `_gqa_bwd_main+split` | llama-3.1-8b-short/fp16 | 2011.6 | **1016.8** | **−49.45%** | 133.3 / 81.2 / 293.3 | 2 |
| `_gqa_bwd_main+split` | llama-3.1-8b-short/bf16 | 2012.3 | **1028.3** | **−48.90%** | 133.3 / 83.8 / 291.5 | 2 |
| `_gqa_bwd_main+split` | llama-3.1-8b-long/fp16 | 13954.2 | **7518.6** | **−46.12%** | 123.8 / 78.1 / 266.2 | 16 |
| `_gqa_bwd_main+split` | llama-3.1-8b-long/bf16 | 13995.8 | **7531.9** | **−46.18%** | 124.1 / 79.8 / 267.2 | 16 |
| `_gqa_bwd_main+split` | llama-3.1-70b-short/fp16 | 2017.1 | **1015.7** | **−49.64%** | 133.5 / 81.1 / 293.9 | 2 |
| `_gqa_bwd_main+split` | llama-3.1-70b-short/bf16 | 2013.0 | **1020.3** | **−49.32%** | 132.0 / 83.8 / 292.7 | 2 |
| `_gqa_bwd_main+split` | llama-3.1-70b-long/fp16 | 14018.5 | **7525.8** | **−46.31%** | 123.7 / 77.9 / 269.1 | 16 |
| `_gqa_bwd_main+split` | llama-3.1-70b-long/bf16 | 13982.0 | **7545.3** | **−46.04%** | 123.6 / 80.2 / 266.4 | 16 |

**8 例 Task Duration 几何均值：0.5223 × baseline ⇒ 提升 47.8%**（目标 ≥20%）；**单例最大回退 0%（全部为 −46.0%…−49.6%）**。

**A/B 交错复核（同机同 run，逐对交替，≥3 对，`--no-record`）**：

| workload | pair1 base / split | pair2 | pair3 | 中位比 |
|---|---|---|---|---|
| llama-3.1-8b-short/fp16 | 2020.1 / 1014.9 | 2012.8 / 1018.3 | 2013.5 / 1018.2 | 0.5057（−49.4%） |
| llama-3.1-70b-short/bf16 | 2015.8 / 1024.2 | 2012.9 / 1021.7 | 2013.0 / 1024.9 | 0.5088（−49.1%） |
| llama-3.1-8b-long/fp16 | 13935.1 / 7518.4 | 13994.5 / 7510.8 | 13971.8 / 7515.0 | 0.5379（−46.2%） |

三对**同向**、无交叉 ⇒ 收益远超噪声阈值（3%）与 run-state 双态。

**台账口径说明（append-only 遗留，如实登记）**：`round=11` 另有一组 `candidate_id=final-r11-split` 的
`phase=final` 行，其 4 条 long 行是在 `run_measure_split.py` 的 **n_chunks 自动推断缺陷**（按首例 shape 取 chunk 数，
short 的 2 被套到 long 的 16）下写出的**聚合口径错误**行（1017→1251 µs 级，只覆盖 2/16 chunk），
同样本的行内容本身无误。缺陷已修（改为按 case 计算 chunk 数，并把 `n_chunks` 写入 `config` 字段）；
**权威 final 组 = `final-r11-split-v2`（8/8，与交付件同 sha）**，上表全部取自该组。

### 10.5 工作量守恒核验（拆分 vs 融合，long/fp16，32 768 执行瓦片）

| 项 | 融合 r6 | 拆分 r11 | 差额 |
|---|---|---|---|
| GEMM 个数 / 瓦片 | 5 | 5（g1,g2 / g3,g4,g5） | **0** |
| `exp` 次数 / 瓦片 | 1 | 1 | **0** |
| 整 tile 向量 op / 瓦片 | 11–12 | 13（pass2 内） | **0**（+1 次 vmul 是 in-place 复用 acc） |
| 迭代数（每核） | 1365 | pass1/3 各 1365，pass2 683（48 核） | pass2 并行度 ×2 |
| 每迭代成本 | 10.2 µs（含 ~7 µs hand-off） | pass1 1.2 µs / pass2 1.8 µs / pass3 3.1 µs（均 95%+ DMA 忙） | hand-off 项归零 |
| GM 流量（解析） | ≈3.5 GB（q/dO ×16 重读、k/v ×16、dq 原子、dk/dv） | ≈3.5 + 2.1（S/dW 写）+ 2.1（S/dW 读）+ 0.54+0.54（P/dS）+ 1.65（pass3 重读 q/dO/k）≈ **10.4 GB** | **+~7 GB**（+3.0×） |
| 额外重复装载 | — | pass3 重读 q/dO/k | 已在 `mte2` 内计入 |

**带宽可行性反例（必须记账的理由）**：把 S/dW **和** P/dS 全量留在 GM 且不分块时，long 档解析流量 3.5 → 11.5 GB，按本机实测可达 ~0.8 TB/s 需 **14 ms ≥ 融合 13.95 ms**——即「纯引擎分离、全量中转」在本器件上**带宽不可行**。本形态成立的前提是分块后 workspace 驻留近层（实测三 pass 的 `mte2` active 速率 43–52 GB/s/核，聚合 1.1–1.25 TB/s）。⇒ `[DESIGN_LIMIT]` 不产出（瓶颈在编译器同步形态/带宽，非算法设计层）。

### 10.6 否决的替代（同轮实测/解析）

| 替代 | 预期 | 实测/解析 | 结论 |
|---|---|---|---|
| 相位合并（r10a） | 减少 hand-off 次数 | −1.6% / +0.3% | 编译器按 buffer 粒度同步，源级相位段数无效 |
| 去运行时守卫（r10d） | 守卫是固定成本 | 每迭代 9.77 µs（不变） | 守卫不是杠杆 |
| tile 再放大 | 迭代数再降 | `(72,72)@dim=128` 需 178 KB > 175 KB 预算（`(64,64)` = 163.8 KB） | UB 封顶 |
| 只拆 dS 之后（pass A 融合 + pass B 纯 Cube） | 半个 hand-off | pass A 仍付 7 µs/迭代 ⇒ 解析 11.5 ms（−17%） | 收益不足，未实现 |
| 全量中转纯引擎分离 | hand-off 归零 | 解析 11.5 GB ⇒ ≥14 ms | 带宽不可行（§10.5） |
| Expert 双 Scope 手动相位 | 历史 headroom 50–67% | — | `CG-2026-0040` 硬禁用，本轮**未重试** |

### 10.7 中止条件与预算

- **success**：8 例 tune workload 全部走拆分路径并达成本轮目标（几何均值降幅见 §10.4），单例无 >3% 回退。
- 预算：结构/参数候选 **5 个**（r10a、r10c、r10d 诊断、r10e 拆分、r10f barrier 修正）+ 探针 **3 个**（`probe_split.py` 的 pure-Vector 并行度 / 三 pass 代价模型 / 大工作集重测）≤ `max_stage4_experiments=16`；本轮为单轮结构重写（未走 plateau）。
- 未触发 `[DESIGN_LIMIT]`：瓶颈归因于 Developer 模式跨引擎同步形态与 GM 带宽，**不是算法/设计层**；DESIGN §1.6.4 的两趟备选（本轮按同轮解析否决：迭代数 ×2 且每迭代仍付 hand-off）已在 §10.6 记账。

### 10.8 smoke 与结构重写后的精度回归

| 项 | 结果 |
|---|---|
| `mha-bwd-smoke-s512-h8-d64/fp16`（split 路径，`bench.py --check`） | **PASS**：`dq` 7.570e-04 / `dk` 9.129e-04 / `dv` 8.669e-04，tier1/2 越界 0 —— 与融合 r6 **逐位同值** |
| `mha-bwd-smoke-s512-h8-d64/bf16`（split 路径，模块分层门 `try_case.py`） | **PASS tier-3**：over=17/786432（0.0022%）、对源忠实参考 max **2.728e-04**、viol=0 —— 与 r6 基线逐项相同 |
| `_gqa_bwd_kernel.py --level all` | **EXIT=0 / `All check passed!`**（第二版运行）：L0 **ALL PASS (16 items)**、L1 **ALL PASS (10/10)**、L2 6 PASS + 1 WARN（`S=32` 既有 `bishengir-compile` 段错误）、Boundary 1 WARN（记录不断言）⇒ 无精度回归 | `logs/r11_gate/all_v2.log` |
| 第一版运行的失败项（已修复） | L0-13/block_m-16、L0-13/block_n-16、L0-15/bn128-dim128-ub「was NOT rejected」——拆分路径绕过了融合 builder 里的 tile 守卫；已抽出 `_validate_tile` 并在拆分分派前调用（`logs/r11_gate/all_v1.log` 为修复前证据） | `logs/r11_gate/all_v1.log` |
| 结构重写对数值的影响 | **零**：拆分不改任何元素级算术（同 5 GEMM、同 13 v-op、同 P/dS dtype 回落、同 fp32 累加域）；`dq` 的原子累加序与融合版一致（同一 (q块, kv块) 发射序） |
| L0-14 确定性 | `dq/dk/dv` 三次重复 `bit_equal_3runs=True`（`max_spread=0.0`），split 路径同 |

### 10.9 Skill Retrospective（Round 11）

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 优化方法学 | **「两引擎同时 ~90% 等待」是跨引擎 hand-off 的指纹**，此时「合并源级相位」直觉上有效但实测无效（编译器按 buffer 粒度插同步）。正确顺序：先用「同结构两种工作量」的每迭代成本证明固定项主导，再决定是否走结构级拆分 | r10a（6 段→3 段）−1.6%/+0.3%；r10d（去守卫、迭代×2）每迭代 9.77 µs 不变；`aiv_scalar_wait_id7` 12.3 ms + `aic_scalar_wait_id8` 13.6 ms 并存 | `iteration-diagnosis.md` 增「等待指纹表」：① 仅一侧等待 = 数据依赖；② 两侧各 ~90% 等待 = hand-off 单价；③ 每迭代成本与 tile 面积无关 = 固定项主导 ⇒ 只能减迭代数或换结构 | P |
| 结构迁移方法 | **拆分前必须做两件事**：① 解析流量记账（把「全量中转」与「分块中转」两种形态的 GB 都算出来，用实测可达带宽换算 ms）；② 三 pass 的每-stage 代价探针（3 个小 kernel × 2 档 tile，约 10 min）。本轮 ①证明「全量中转」≥14 ms 不可行、②给出 5.7–8.7 ms 的预期，二者共同决定了「分块 + 三 kernel」这一具体形态 | §10.5（3.5→11.5 GB 反例）、`probe_split.py`（1.83/1.77/3.64 µs/瓦片） | `bottleneck-patterns.md` / `iteration-diagnosis.md` 增「引擎相位拆分可行性三问」：traffic 记账、每 stage DMA 单价探针、chunk 粒度（单写者保持） | P |
| 测量口径 | **多 kernel 形态的聚合必须按 case 推断 chunk 数并把该值写回记录**。用首例 shape 推一个全局 chunk 数会对 long 类少算 8×（1017 vs 7519 µs 的假加速），且首轮记录把该假数字写进了 `phase=final` 行 | 本轮 §10.4 的 `final-r11-split` 遗留组（4 条 long 行聚合口径错误，已用 `final-r11-split-v2` 权威组覆盖并登记） | `profile-collection.md` 增「多 kernel 求和的三个必填项：每 case kernel 数、每 case chunk 数、求和后取中位的样本数」，且聚合脚本须把这三个量写入记录 | R |
| 陷阱域 | **纯 Vector npuir kernel 的跨迭代 WAR 无自动同步**：UB scratch 被 vec 写 / mte3 读时静默损坏 ~18% 瓦片（上游中间量正确）。拆分/重写类任务必须做「无 barrier vs 有 barrier」逐位对拍 | 91/512 坏瓦片 → 0/512；端到端 `dq` 11.15 → 7.57e-04 | 已回写 `TRAP-npuir-vector-kernel-store-war-race`（含 repro） | D |
| 契约 | **尾块 shape 保留融合路径**是低风险做法：拆分 pass 写整瓦片 workspace，动态 extent 的区域形态被前端拒绝（既有 `TRAP-gemm-dynamic-size-cbuf` 族），因此 `_split_enabled` 只在 `S % tile == 0` 时生效；manifest 8 例全整除 | L0-6/L0-7/L0-9/L1 尾块用例走融合路径且全 PASS | `iteration-diagnosis.md` 增「结构重写的尾块分派纪律：新结构只吃整除域，尾块留旧路径」 | R |

### 10.10 Value Point Proposals（含证据三件套）

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **Developer Mix kernel 的跨引擎 hand-off 单价 ≈7 µs/迭代（910B2）**，且与相位段数、运行时守卫、tile 面积无关；`duration ≈ Σ引擎忙时 + n_iter × 7 µs` | D | §10.1/§10.2（r10a/r10d）；`aiv_scalar_wait_id7` 12.3 ms / `aic_scalar_wait_id8` 13.6 ms | `repro/CONST-crossengine-handoff-910B2.py`（已回写 constants.md） | 910B2 / CANN 9.0.0 / tilelang `0.1.2+151c62a` / 2026-10-22 |
| **纯 Vector npuir kernel 跨迭代 WAR 无自动同步**：静默损坏 18% 瓦片，绕法 = 迭代末 `T.pipe_barrier("PIPE_ALL")`（→ 逐位正确） | D | §10.3 约束 2（91/512 → 0/512；端到端 `dq` 11.15 → 7.57e-04） | `repro/TRAP-npuir-vector-kernel-store-war-race.py`（已回写 traps-compiler.md） | 同上 |
| **attention 反向的引擎相位拆分**（3 kernel + KV 分块 GM workspace + `by0: T.int32` + 每迭代 barrier）：8 例几何 −47.8%，逐例 −46.0…−49.6% | P | §10.4（权威 final 组 8/8）+ §10.5 守恒核验 + §10.6 反例 | `repro/PATT-attention-bwd-engine-phase-split.py`（已回写 attention.md） | 同上 |
| **多 kernel 算子的聚合口径三必填项**（每 case kernel 数 / chunk 数 / 求和后样本数），否则少算 chunk 会造出 8× 假加速 | R | §10.4 的 `final-r11-split` 遗留组（1017 vs 7519 µs） | — | 同上 |
| **拆分可行性三问**（traffic 记账 / 每 stage 单价探针 / chunk 单写者） | P | §10.5（全量中转 ≥14 ms 反例）、§10.6（只拆一半 = −17% 不够） | — | 同上 |

### 10.11 残留 / 未验证项 / 风险（如实披露）

1. **workspace 常驻内存**：closure 在首次调用时按 `cb` 分配并**长期持有**（long 档 ~201 MB，最大 L0 形状 (16,2048,16,128) 因 `cb` 下限为 1 而 ~402 MB）。功能与性能无影响，但作为库行为需上层知晓（可改为按 chunk 复用同一块 / 或由 wrapper 提供 workspace）。
2. **尾块 shape 不享受拆分**：`S % 64 != 0` 走融合路径（功能/精度正常，无性能承诺）。
3. **pass 3 仍含一个 cube 相位 + 一个 AIV 相位**（dq 原子），故仍付少量 hand-off；其每-launch 266–294 µs 是三 pass 中最大项（占 per-chunk 57%），后续仍有优化空间（未做：pass3 大 tile 重分组 / dq 改为 dS 二次分块归并）。
4. **`T.pipe_barrier("PIPE_ALL")` 的代价未单独标定**：本轮的比较是「无 barrier（错）vs 有 barrier（对）」，barrier 的净成本未单独测（量级判断 <3%，未实测）。
5. **跨机不可比**：本轮全部数字为 910B2（24 核 / 48 AIV）；910B4 的 5.27–6.62× 历史结论与本轮 −47.8% 不可直接比较（器件、核数、per-iteration 常数均不同）。
6. **未做 `[DESIGN_LIMIT]`**：瓶颈归因编译器同步形态 + GM 带宽，非设计层；`perf_feedback.md` **不产出**（见 §10.5 末）。

---

## 11. 台账归位与门禁口径（round 11 收尾，append-only 说明）

`perf_records.jsonl` 按 gate 4 的机械契约重建（重建前的原文件逐字节备份在
`logs/r11_gate/perf_records_before_r11_rebuild.jsonl`，被移出的行**逐条原文**追加进
`perf_records_full_history.jsonl`，无删除）：

| 处置 | 行 | 原因 |
|---|---|---|
| 留在 active（24 行） | round 0/1/6（baseline / candidate / merged） | 原有合规历史行 |
| 留在 active（8 行） | round 11 `phase=baseline`、`candidate_id=r6-arm-r11` | 本轮 baseline 臂；`l0_pass` 由该臂在本机跑完整 `--level all` 确认（`logs/r11_gate/base_all.log`，EXIT=0，L0 16/16、L1 10/10） |
| 留在 active（8 行） | round 11 `phase=final`、`candidate_id=final-r11-split-v2`、`kernel_id=_gqa_bwd_main` | 每个 tune workload **恰一条** final，`artifact_sha256` = 当前交付件，`duration_us` = 一次 op 调用内全部 kernel 的 Task Duration 之和（分项写入 `per_kernel_us`；每 kernel 的原始行在 sidecar） |
| 移出 active（86 行） | round 7 的 8 条 final（旧交付件 sha `8344d214…`）；round 11 的 6 条 probe 行（`phase=probe`、`l0_pass=null`）；round 11 `final-r11-split` 组 32 行（含 4 条 long 聚合口径错误行）；round 11 `final-r11-split-v2` 的 24 条分项行 | 契约要求：final 恰好一条且 sha 与当前交付件一致（旧 sha 的行会让 `S4-PERF-RECORDS-RECON` 失败）；`phase` 只允许 baseline/candidate/merged/final；`(kernel_id, workload_id)` 必须落在 inventory 的 tune 集合内（pass 分项行用 `_gqa_bwd_sdw…` 等 id，故入 sidecar） |

- **sidecar**：`perf_records_split_detail.jsonl`（56 行 + 3 行注释）= 三 pass 的分项行（`_gqa_bwd_sdw_mix_aic` /
  `_gqa_bwd_pds` / `_gqa_bwd_mul_mix_aic`）+ 全部 probe 行 + 被取代的 `final-r11-split` 组，逐条带
  `artifact_sha256` / `msprof_raw_path`，供复核时逐 kernel 对账。
- **为什么 §6 的标题加了注记**：gate 4 的 `Final Performance Test Data` 对账取**文件中第一处**该标题下的 5 列表，
  并与 active 台账的唯一 final 行比对。交付件换代后 §6（round 7）的表必然与当前 final 不符，故只给该标题加
  「round 7 历史记录」注记（**表内数值一字未改**），当前最终表见 §12。
- **§10.4 的定位**：§10.4 是**分项拆解表**（7 列，含 sdw/pds/mul 分项），标题带「10.4」前缀，不参与门禁对账；
  门禁口径的 5 列表在 §12。

---

## 12. 当前最终交付验收（gate 4 对账口径）

交付件：`perf_opt/_gqa_bwd_kernel.py`（sha256 `6ca13f73ccb22ffbc9258d957320bae9d93105254ddc11d03c778d91bde92179`），
`candidate_id = final-r11-split-v2`，结构 = **engine-phase split**（Cube-only → Vector-only → Mix 三 kernel +
KV 分块 GM workspace）。
口径：`msprof op` `Task Duration(us)`，多 kernel 形态 = 一次 op 调用内**全部** kernel 之和（20 次调用取中位；
每 kernel 分项见 §10.4 与 `perf_records_split_detail.jsonl`）。

### Final Performance Test Data（round 11/12 历史 —— 交付件为 256 MiB 修复前版本；当前最终表见 §15）

> 本节 = **历史交付件口径**（`perf_opt/_gqa_bwd_kernel.py` sha256 `6ca13f73…`，256 MiB，
> `candidate_id = final-r11-split-v2`）；round-12 的 768 MiB A/B 表在 §14.6，已标注「未采纳」（见 §14.10）。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---|---|---|
| _gqa_bwd_main | llama-3.1-8b-short/fp16 | 2011.57 | 1016.78 | final-r11-split-v2 |
| _gqa_bwd_main | llama-3.1-8b-short/bf16 | 2012.32 | 1028.34 | final-r11-split-v2 |
| _gqa_bwd_main | llama-3.1-8b-long/fp16 | 13954.16 | 7518.62 | final-r11-split-v2 |
| _gqa_bwd_main | llama-3.1-8b-long/bf16 | 13995.79 | 7531.92 | final-r11-split-v2 |
| _gqa_bwd_main | llama-3.1-70b-short/fp16 | 2017.08 | 1015.72 | final-r11-split-v2 |
| _gqa_bwd_main | llama-3.1-70b-short/bf16 | 2013.05 | 1020.28 | final-r11-split-v2 |
| _gqa_bwd_main | llama-3.1-70b-long/fp16 | 14018.46 | 7525.84 | final-r11-split-v2 |
| _gqa_bwd_main | llama-3.1-70b-long/bf16 | 13981.99 | 7545.30 | final-r11-split-v2 |

**8 例几何均值 0.5223×baseline（−47.8%）**；单例最优 −49.64%、最差 −46.04%（无回退）。

### 12.1 交付前回归（本机，`TILELANG_ASCEND_MODE=Developer`）

| 项 | 结果 | 日志 |
|---|---|---|
| 交付件 `--level all` | **EXIT=0 / `All check passed!`**：L0 **ALL PASS (16 items)**、L1 **ALL PASS (10/10)**、L2 6 PASS + 1 WARN（`S=32` 既有 `bishengir-compile` 段错误，`TRAP-bishengir-crash-s32`）、Boundary 1 WARN（×100 量级输入 fp32 抵消，记录不断言） | `logs/r11_gate/all_v2.log` |
| baseline 臂 `--level all` | **EXIT=0 / `All check passed!`**：L0 ALL PASS、L1 ALL PASS | `logs/r11_gate/base_all.log` |
| smoke 两例（`mha-bwd-smoke-s512-h8-d64/{fp16,bf16}`，manifest shape，用模块自带 `_run_case` 分层门复核） | fp16 **PASS tier-1**（worst 9.129e-04，与 r6 同值）；bf16 **PASS tier-3**（over=17/786432 = 0.0022% ≤ 5%，对源忠实参考 max 2.728e-04、viol=0），与 r6 基线**逐项相同** ⇒ 算法固有 P/dS 回落噪声，非回归 | `logs/r11_gate/` + `try_case.py` 本轮实跑 |
| L0-14 确定性 | `dq/dk/dv` 三次重复 `bit_equal_3runs=True`、`max_spread=0.0` | 同上 |
| drop-in 契约 | 工厂名 `_gqa_bwd_kernel`、9 参调用签名、返回 `(dq,dk,dv)`、`TUNED_DEFAULT_CONFIG` 均不变；尾块 shape 走融合路径（与 r6 行为一致） | — |

### 12.2 外部编译器阻塞与端到端口径（P0 收口，详见 §13）

- **正式验收状态不变**：`BLOCKED_ENVIRONMENT`（fwd 侧 `bishengir-compile` 在 `convert-hfusion-to-hivm`
  对 MIX module 概率性 SIGSEGV；本轮起冻结，不再重试 fwd 编译）。
- 证据包（可直接提交，尚未建 issue）：`perf_opt/issue_bishengir_hfusion_hivm/`
  （`issue.md` / `repro.sh` 5 s / `evidence/kernel_smoke_fwd.npuir` / `checksums.txt`）。
- **IR_SHA256**：`4adf1165a0709d67ef69befe118cb5041f5904ab78c53a3014fa55af0b474e2e`
  （单 pass 6/6；7 组 flag 4/6–6/6；CANN 9.0.0；tilelang `0.1.2+151c62af`；Ascend910B2）。
- 本节上表的 **−47.8% 是 device-only（msprof Task Duration 求和）** 口径；bwd-only report 另给出
  **op wall（一次 `op(...)` 调用，NPU event）** 8 例并列数字，见 §13.3。两者不可混用。
- 会话戳：上表数字的源码戳 = `151c62a`；本次 op-wall run 的 repo HEAD = `16f1ee45`
  （`tilelang/` 源码在两戳间无改动）。`perf_records*.jsonl` 已逐行补 `device` + `commit`，见 §13.2。
- 交付库同步：`/workspace/operator-migration`（kernel sha `6ca13f73…` + 本日志 + 台账 + bwd-only report）。

## 13. P0 外部阻塞收口（2026-10-22，append-only）

> 本节只追加，不改写 §1–§12 的任何数字与结论。目的：把「fwd 外部编译器阻塞」冻结成可交付的证据包，
> 并补齐 bwd 的端到端口径（op wall）。**正式验收不因此转绿**，仍为 `BLOCKED_ENVIRONMENT`。

### 13.1 冻结：fwd 不再重试，外部编译器阻塞入账

- 本轮 fwd 编译重试次数 = **0**（冻结纪律）。fwd 侧结论沿用 `fwd_fix_log.md` 与本轮前的定界记录。
- 阻塞定性：**外部编译器缺陷**——`bishengir-compile` 在 `ConvertHFusionToHIVM` pass 内对
  MIX module（`*_mix_aic` + `*_mix_aiv`）概率性 SIGSEGV；同字节 IR 跨时间窗 0/10–20/20。
- 证据包（可直接提交，**尚未建 issue**）：`perf_opt/issue_bishengir_hfusion_hivm/`，入口 `README.md` /
  正文 `issue.md` / 5 s 复现 `repro.sh`（默认用随包 IR，只依赖 CANN）。
- **IR_SHA256** = `4adf1165a0709d67ef69befe118cb5041f5904ab78c53a3014fa55af0b474e2e`（313 行，
  随包 `evidence/kernel_smoke_fwd.npuir` 复算一致）。
- 复现证据：单 pass `bishengir-opt <IR> --convert-hfusion-to-hivm` **6/6**（另一窗口 20/20）；
  7 组 flag 变体 4/6–6/6（`evidence/flag_matrix.md`）；无关 MIX 核 `_ssd_chunk_scan_fwd_kernel` 5/5 同签名。
- 设备/工具链戳：**Ascend910B2**（npu-smi 26.1.1，HBM 65536 MB）/ **CANN 9.0.0** /
  tilelang `0.1.2+151c62af8b9c1e72d8372da736b1c3d8582b1022`（源码戳；`tilelang/` 至 HEAD `16f1ee45` 无改动）/
  torch `2.9.0+cpu` / torch_npu `2.9.0.post8` / python 3.10.12。
- **新增观察（2026-10-22 08:15 UTC，单次、未重试）**：bwd 基线臂 `_gqa_bwd_kernel_r6_baseline.py` 在
  HEAD `16f1ee45` 冷编译复现**同一 stack 签名**（`__kernel_rt_sigreturn` + 重复帧地址 +
  `ERR99999 UNKNOWN application exception`）。该次未重新 capture pass 名、按冻结纪律未重试，
  **只作新数据点**（不并入 6/6 证据表），含义是：不要把失败面假设为「仅 Expert 写法」，
  更稳妥的刻画是「非平凡 MIX module + 中等 IR 体量，概率性」。

### 13.2 会话戳统一（报告 `16f1ee45` vs 台账 `151c62a`）

两者不是同一层字段，统一记录如下（报告、台账、`perf_records` 三处同表）：

| 字段 | 值 | 含义 |
|---|---|---|
| repo git HEAD（report `git_commit`） | `16f1ee4578a66e9123bb32783956e272df17ecad` | 本次 bwd-only report 的 run 戳 |
| `tilelang.__version__` | `0.1.2+16f1ee4578a66e9123bb32783956e272df17ecad` | `version.py` 用 live HEAD 拼接 |
| kernel-cache key pin | `0.1.2+16f1ee4578a66e9123bb32783956e272df17ecad` | `TILEOPS_PIN_TILELANG_VERSION`；窗口内不随提交漂移 |
| round 11 台账戳 | `0.1.2+151c62af8b9c1e72d8372da736b1c3d8582b1022` | §10.0 测量时 HEAD=`151c62a` |
| 源码一致性 | `git diff --stat 151c62a..16f1ee45 -- tilelang/` **为空** | 两戳之间只有 `.agents`/`.opencode` 文档提交 |

⇒ §10/§12 的 round-11 数字继续有效（同一份 `tilelang/` 源码），但**引用时必须带源码戳 `151c62a`**；
本次 op-wall 的 run 戳是 `16f1ee45`。`perf_records*.jsonl` 已逐行补 `device` + `commit`：
round 0/1/6/7 → `Ascend910B4` / `2fb23ee`（§0）；round 8/9/10 → `Ascend910B4` / `06e1a1a`（§9.1）；
round 11 → `Ascend910B2` / `151c62a`（§10.0）。补写前原件备份：
`logs/r11_p0_backfill/{perf_records,perf_records_split_detail,perf_records_full_history}.jsonl.pre_p0_backfill`。

### 13.3 op wall 补充测量（bwd-only report `20261022_081204_670534`）

口径动机：`tileops-report` 的 msprof 单 kernel 行对本算子只捕获 `flash_bwd_prep`（30–82 µs），
不是整个 op。为此在 `perf_opt/bwd_only/bench_mha_bwd_only.py` 增加 **op wall**：
用 NPU event 包住**一次** `op(q,k,v,o,do,lse)` 调用（5 warmup + 3 trial × 10 次计时，取 trial 均值中位），
覆盖 `prep` + 3 pass × `n_chunks` + `post` 及 `dq/dk/dv` 清零、`dk/dv` cast。
**原 msprof 行保留在同一 record 中（`latency_us`），新增 `op_wall_us` 并列，不覆盖。**
补充说明与分解：report 目录 `OP_WALL.md`。

| workload | dtype | **op wall (us)** | op wall min–max | msprof captured (us) | device sum (us) | per-kernel sdw/pds/mul (us/launch) | n_chunks | op/device |
|---|---|---:|---:|---:|---:|---|---:|---:|
| llama-3.1-70b-long | bfloat16 | **12959.0** | 12873–15200 | 72.18 | 7545.28 | 123.6 / 80.2 / 266.4 | 16 | 1.72× |
| llama-3.1-70b-long | float16 | **12760.1** | 12389–13631 | 81.92 | 7525.84 | 123.7 / 77.9 / 269.1 | 16 | 1.70× |
| llama-3.1-70b-short | bfloat16 | **2326.0** | 2300–2447 | 32.08 | 1020.28 | 132.0 / 83.8 / 292.7 | 2 | 2.28× |
| llama-3.1-70b-short | float16 | **2302.5** | 2272–2378 | 32.64 | 1015.72 | 133.5 / 81.1 / 293.9 | 2 | 2.27× |
| llama-3.1-8b-long | bfloat16 | **12121.6** | 11819–14788 | 58.28 | 7531.92 | 124.1 / 79.8 / 267.2 | 16 | 1.61× |
| llama-3.1-8b-long | float16 | **12068.3** | 11996–12165 | 59.68 | 7518.62 | 123.8 / 78.1 / 266.2 | 16 | 1.61× |
| llama-3.1-8b-short | bfloat16 | **2516.3** | 2482–2737 | 30.26 | 1028.34 | 133.3 / 83.8 / 291.5 | 2 | 2.45× |
| llama-3.1-8b-short | float16 | **2382.4** | 2328–4010 | 30.92 | 1016.78 | 133.3 / 81.2 / 293.3 | 2 | 2.34× |

读法与限制：

- `op wall` 是**端到端**（含每次 launch 的 host 下发间隙），`device sum` 是器件 `Task Duration` 之和；
  两者不可互换。8 例 op/device = 1.61–2.45×（几何均值 1.97×）。
- 分解（8b-short/fp16，同进程 event 探针）：`full op 2501 = prep 317 + main 1784 + post 328 + zeros 133 + casts 117` µs；
  `main` 的 6 次 launch 器件和约 1015 µs，余约 770 µs 为 host 间隙（≈128 µs/launch）；
  长档 48 次 main launch 对应约 4.6 ms 间隙，与 `op wall − device sum` 一致。
- **不得把 op wall 与 −47.8% 混用**：−47.8% 是 device-only（§12 表）口径；op wall 口径下 split 形态
  因 launch 次数增至 4–6×，端到端收益显著更小（短档接近持平）。本节的 op wall 数字**不构成验收结论**。
- op-wall 的 baseline A/B 缺口：基线臂冷编译在 08:15 复现同签名崩溃（§13.1 新增观察），按冻结纪律未重试，
  故不产出 op-wall baseline 行；不以外推数字替代实测。
- `Bandwidth (TB/s) = 1.75` 是 roofline 分母，仅作台账分母，不是实测带宽；`Ratio (%)` 同样由单 kernel msprof 数字计算，不可当作 op 级占比。

### 13.4 台账一致性、门禁与正式验收

- `perf_records.jsonl` 补字段后按 gate 4 的 `perf_lint` **直调**复核
  （`perf_lint(_gqa_bwd_kernel.py, opt_log.md, perf_feedback.md, perf_records.jsonl, '_gqa_bwd_kernel', repo_root)`）：
  `failures=[] / warnings=[]` ⇒ 对账未被破坏。
- 注意：`statectl gate 4 --dir <op dir>` 当前跑出 8 条 `S4-PERF-RECORDS-RECON`，原因是**活动状态文件属于 fwd 修复任务**
  （`operator_name=_gqa_prefill_fwd_kernel`、`final_artifact=null`），与 bwd 台账无关；bwd 的 round-3 状态已归档在
  `history_version/.stage_state_round3_bwd_20261022_DONE.json`（`phase=DONE`，final sha `6ca13f73`）。
- **正式验收状态不变**：`phase=FAILED` / `failure_reason=BLOCKED_ENVIRONMENT`（fwd 外部编译器阻塞）。
  bwd-only report（correctness 6/6、benchmark 8/8）**只是 bwd 面证据**，不冒充 op 对级验收。
- 交付库 `/workspace/operator-migration` 同步：kernel `_gqa_bwd_kernel.py` = `6ca13f73…`、
  `perf_opt/{_gqa_bwd_kernel.py,opt_log.md,perf_records*.jsonl,workload_inventory.json}`、
  report 目录 `reports/tileops/20261022_081204_670534_MultiHeadAttentionBwdOp/`、
  台账 `experience/multi_head_attention_bwd.md` + `index/operator_status.md`。

## 14. round 12：split workspace 预算 / launch 数（append-only）

续跑窗口记录（task `TileOPs-_gqa_bwd_kernel-20261022T102347Z`，`scenario=optimize`、`mode=full`）。
中断点：op wall 全量已完成、device 只有 1 例 1 rep、决策与交付件常量未落盘。本节是续跑窗口的
实测 + 收口，**不覆盖** §10–§13 的任何数字（旧的 `phase=final` 行按 append-only 归档到
`perf_records_full_history.jsonl`）。

### 14.0 改动面与机制假设

- **唯一变量**：`_SPLIT_WS_BUDGET_BYTES`（host 侧 `_split_plan` 的 workspace 预算）。三 pass kernel 体、
  `block_m/block_n/num_stages`、对外签名/工厂名、fused 尾块路径全部不动（`diff` 见 §14.5）。
- 预测机制：预算只决定 `cb = min(budget // per_kv, n_kv)`（long `per_kv` = 96 MiB、short 48 MiB），
  因而只决定 `n_chunks = ceil(n_kv / cb)`；一次 op 调用的 main launch 数 = `3 × n_chunks`
  （`_build_split_launch` 的 host 循环，实现 1151–1154 行），三 pass 的数学与总访存量恒定。
  ⇒ **预期收益全部来自每次 launch 的 host 下发间隙**（§13.3 实测 ≈128 µs/launch），不是 kernel 内计算。
- 基线协议：以 **512 MiB（cb=5 → long 7 chunks）为基线**；候选 **576 / 672 / 768 MiB**
  （long cb=6/7/8 → 6/5/4 chunks）；short 档在 ≥384 MiB 后一律 `cb=8` → 1 chunk（不敏感对照组）。
- **`1 GiB`（long cb=10）不测**：该组合在冻结工具链上稳定 SIGSEGV（`bishengir-compile`，
  与 `TRAP-bishengir-hfusion-hivm-mix-sigsegv` 同族），本轮**未重试**；≤768 MiB 的每 (arm,shape)
  允许 ≤2 次尝试，本轮 32 个 (arm,shape) 全部 1 次通过。

### 14.1 三列口径与交错协议

| 列 | 口径 | 驱动 | 采样 |
|---|---|---|---|
| **op wall (ms)** | 包住**一次** `op(q,k,v,o,do,lse)` 调用（prep + 3 pass × n_chunks + post + dq/dk/dv 清零 + dk/dv cast），NPU event 计时 | `r12_opwall.py`（importlib 多 arm 同进程） | 8 例 × 4 arm，**3 个交错对**，每 (对,arm) 10 rep，取每对中位再取 3 对中位 |
| **device (ms)** | 一次 op 调用内**全部 kernel** 的 `msprof op` `Task Duration` 之和（`--kernel-name=_gqa_bwd`，`--calls 3`、`--warm-up 1`、`--dump=off`、`--aic-metrics=BasicInfo,PipeUtilization`），逐调用求和后取 3 个调用窗口的中位 | `r12_device.sh` → `run_measure_split.py` | 8 例 × 4 arm × **3 rep**（每 rep 一个独立 msprof 进程） |
| **task 数** | main launch = `3 × n_chunks`；glue = 7（prep 1 + `torch.zeros_like`×3 + post 1 + dk/dv cast 2，见 `MultiHeadAttentionBwdOp.forward`）⇒ 含 glue 的 task/call = `3×n_chunks + 7` | 结构计数（`_split_plan` + op wrapper） | 解析值，非测量 |

交错与去重规则：

- **同 run 交错、禁跨 run 比较**：op wall 逐对轮换 arm 顺序（3 对全排列旋转）；device 逐 rep 轮换
  （奇 rep 正序、偶 rep 反序）⇒ 没有 arm 系统性地排在首/末位。
- **device 去重**：`llama-3.1-8b-long/fp16` 的 **rep1 四个 arm** 沿用中断前窗口（10:38:51–10:42:04，
  同驱动、同 flag、同 rep1 正序）的首测值；其余 **92 个 (case,arm,rep)** 在续跑窗口内测得。
  `r12_device.sh` 用 `SKIP_EXISTING=1` 机械跳过已有 `valid` 组合 ⇒ **没有任何 (case,arm,rep) 被测两次**。
- **device 数字与旧「user 参考值」的差异（如实披露）**：早前参考值 256 MiB = 7.84 ms / 512 MiB = 8.21 ms；
  本轮同口径 512 MiB（8b-long/fp16）= **7.225 ms**，§13.3 的 256 MiB 记录是 7.518 ms。
  差异来源是**调用窗口**（`--calls`/`--warm-up` 不同 ⇒ 采样落在不同的稳态/热态位置）与 run 会话不同，
  不是 kernel 变化。**本轮一切决策只用同 run 交错数字**，不与参考值或 §13.3 的 harness 互推。

### 14.2 逐 arm 结果

#### 14.2.1 arm `b512`（budget 512 MiB，基线）

- 配置：`_SPLIT_WS_BUDGET_BYTES = 512*1024*1024`；`cb` long/short = **5 / 8**；`n_chunks` long/short = **7 / 1**；main launch 数/call = **21 / 3**；含 glue 的 task 数/call = **28 / 10**；long workspace 峰值 **503 MB**。
- 编译：8/8 例通过（每 (arm,shape) 1 次尝试，0 MISS；op-wall 驱动内编译耗时 0.3s, 0.3s, 8.6s, 8.3s, 0.3s, 8.6s, 8.4s, 8.2s）。
- artifact：`_gqa_bwd_kernel_r12_b512.py` sha256 `b87af29ec3766e740f9877f4be7978c1fa6546adc74ce835d89ad789986ca534`。

| workload | chunks | main tasks | op wall (ms) | wall Δ vs b512 | device (ms) | device Δ vs b512 | device reps (ms) |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b-long/fp16 | 7 | 21 | 10.95 | +0.00% | 7.225 | +0.00% | 7206, 7240, 7225 |
| llama-3.1-8b-long/bf16 | 7 | 21 | 10.57 | +0.00% | 7.254 | +0.00% | 7262, 7254, 7251 |
| llama-3.1-70b-long/fp16 | 7 | 21 | 10.77 | +0.00% | 7.246 | +0.00% | 7251, 7246, 7237 |
| llama-3.1-70b-long/bf16 | 7 | 21 | 10.83 | +0.00% | 7.256 | +0.00% | 7268, 7256, 7252 |
| llama-3.1-8b-short/fp16 | 1 | 3 | 2.21 | +0.00% | 0.954 | +0.00% | 958, 952, 954 |
| llama-3.1-8b-short/bf16 | 1 | 3 | 2.23 | +0.00% | 0.964 | +0.00% | 964, 964, 964 |
| llama-3.1-70b-short/fp16 | 1 | 3 | 2.23 | +0.00% | 0.968 | +0.00% | 968, 971, 968 |
| llama-3.1-70b-short/bf16 | 1 | 3 | 2.23 | +0.00% | 0.964 | +0.00% | 969, 963, 964 |

逐对 op wall 中位（3 对）：8b-long/fp16 [10.98, 10.96, 10.94]；8b-long/bf16 [10.57, 10.58, 10.55]；
70b-long/fp16 [10.79, 10.80, 10.66]；70b-long/bf16 [10.87, 10.84, 10.81]；8b-short/fp16 [2.22, 2.21, 2.21]；
8b-short/bf16 [2.22, 2.23, 2.23]；70b-short/fp16 [2.23, 2.23, 2.23]；70b-short/bf16 [2.23, 2.23, 2.25]。

#### 14.2.2 arm `b576`（budget 576 MiB）

- 配置：`_SPLIT_WS_BUDGET_BYTES = 576*1024*1024`；`cb` = **6 / 8**；`n_chunks` = **6 / 1**；main launch 数/call = **18 / 3**；含 glue 的 task 数/call = **25 / 10**；long workspace 峰值 **604 MB**。
- 编译：8/8 例通过（1 次尝试，0 MISS；编译耗时 8.4s, 6.4s, 7.9s, 7.7s, 0.2s, 0.2s, 0.2s, 0.2s）。
- artifact：`_gqa_bwd_kernel_r12_b576.py` sha256 `6033631d574228c9dfdd55c29fe671c4a5082fbc47169f26d52d9eff6e59855a`。

| workload | chunks | main tasks | op wall (ms) | wall Δ vs b512 | device (ms) | device Δ vs b512 | device reps (ms) |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b-long/fp16 | 6 | 18 | 10.58 | -3.39% | 7.141 | -1.17% | 7141, 7159, 7121 |
| llama-3.1-8b-long/bf16 | 6 | 18 | 10.24 | -3.14% | 7.150 | -1.43% | 7158, 7136, 7150 |
| llama-3.1-70b-long/fp16 | 6 | 18 | 10.44 | -3.12% | 7.162 | -1.16% | 7162, 7174, 7144 |
| llama-3.1-70b-long/bf16 | 6 | 18 | 10.49 | -3.19% | 7.175 | -1.12% | 7171, 7191, 7175 |
| llama-3.1-8b-short/fp16 | 1 | 3 | 2.22 | +0.23% | 0.956 | +0.21% | 956, 956, 963 |
| llama-3.1-8b-short/bf16 | 1 | 3 | 2.23 | +0.17% | 0.963 | -0.08% | 963, 973, 960 |
| llama-3.1-70b-short/fp16 | 1 | 3 | 2.23 | +0.13% | 0.974 | +0.61% | 974, 964, 975 |
| llama-3.1-70b-short/bf16 | 1 | 3 | 2.24 | +0.22% | 0.966 | +0.16% | 969, 964, 966 |

逐对 op wall 中位：8b-long/fp16 [10.58, 10.61, 10.56]；8b-long/bf16 [10.22, 10.27, 10.22]；
70b-long/fp16 [10.46, 10.47, 10.33]；70b-long/bf16 [10.49, 10.51, 10.45]；short 四例均 2.22–2.25。

#### 14.2.3 arm `b672`（budget 672 MiB）

- 配置：`_SPLIT_WS_BUDGET_BYTES = 672*1024*1024`；`cb` = **7 / 8**；`n_chunks` = **5 / 1**；main launch 数/call = **15 / 3**；含 glue 的 task 数/call = **22 / 10**；long workspace 峰值 **705 MB**。
- 编译：8/8 例通过（1 次尝试，0 MISS；编译耗时 7.8s, 8.2s, 8.0s, 7.8s, 0.2s, 0.2s, 0.2s, 0.2s）。
- artifact：`_gqa_bwd_kernel_r12_b672.py` sha256 `d3730625781e1c5ccb44d2aac9a34b151bd85b340a6ada8cbcd427aae5ee4228`。

| workload | chunks | main tasks | op wall (ms) | wall Δ vs b512 | device (ms) | device Δ vs b512 | device reps (ms) |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b-long/fp16 | 5 | 15 | 10.39 | -5.13% | 7.207 | -0.26% | 7202, 7220, 7207 |
| llama-3.1-8b-long/bf16 | 5 | 15 | 10.07 | -4.73% | 7.222 | -0.44% | 7222, 7222, 7220 |
| llama-3.1-70b-long/fp16 | 5 | 15 | 10.25 | -4.84% | 7.239 | -0.09% | 7238, 7239, 7242 |
| llama-3.1-70b-long/bf16 | 5 | 15 | 10.28 | -5.07% | 7.251 | -0.07% | 7252, 7232, 7251 |
| llama-3.1-8b-short/fp16 | 1 | 3 | 2.21 | +0.01% | 0.962 | +0.83% | 962, 966, 957 |
| llama-3.1-8b-short/bf16 | 1 | 3 | 2.23 | +0.27% | 0.967 | +0.36% | 968, 958, 967 |
| llama-3.1-70b-short/fp16 | 1 | 3 | 2.23 | +0.04% | 0.962 | -0.55% | 962, 966, 960 |
| llama-3.1-70b-short/bf16 | 1 | 3 | 2.24 | +0.30% | 0.971 | +0.65% | 968, 970, 978 |

逐对 op wall 中位：8b-long/fp16 [10.42, 10.39, 10.39]；8b-long/bf16 [10.09, 10.05, 10.08]；
70b-long/fp16 [10.30, 10.26, 10.18]；70b-long/bf16 [10.32, 10.28, 10.28]；short 四例均 2.21–2.24。

#### 14.2.4 arm `b768`（budget 768 MiB，本轮 A/B 最优，**未采纳**）

- 配置：`_SPLIT_WS_BUDGET_BYTES = 768*1024*1024`；`cb` = **8 / 8**；`n_chunks` = **4 / 1**；main launch 数/call = **12 / 3**；含 glue 的 task 数/call = **19 / 10**；long workspace 峰值 **805 MB**。
- 编译：8/8 例通过（1 次尝试，0 MISS；编译耗时 7.9s, 8.1s, 8.0s, 7.8s, 0.2s, 0.2s, 0.2s, 0.2s）。
- artifact：`_gqa_bwd_kernel_r12_b768.py` sha256 `8aad04947f1bd5a66bfb10e2cb42c41dff9263557c2f898a889ef382b202bce0`。

| workload | chunks | main tasks | op wall (ms) | wall Δ vs b512 | device (ms) | device Δ vs b512 | device reps (ms) |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b-long/fp16 | 4 | 12 | 10.06 | -8.14% | 7.165 | -0.83% | 7165, 7185, 7151 |
| llama-3.1-8b-long/bf16 | 4 | 12 | 9.80 | -7.33% | 7.180 | -1.01% | 7180, 7169, 7240 |
| llama-3.1-70b-long/fp16 | 4 | 12 | 9.94 | -7.79% | 7.214 | -0.44% | 7210, 7214, 7214 |
| llama-3.1-70b-long/bf16 | 4 | 12 | 9.96 | -8.09% | 7.209 | -0.64% | 7186, 7222, 7209 |
| llama-3.1-8b-short/fp16 | 1 | 3 | 2.22 | +0.09% | 0.962 | +0.86% | 962, 970, 956 |
| llama-3.1-8b-short/bf16 | 1 | 3 | 2.23 | -0.17% | 0.960 | -0.36% | 960, 969, 958 |
| llama-3.1-70b-short/fp16 | 1 | 3 | 2.23 | -0.00% | 0.965 | -0.31% | 972, 957, 965 |
| llama-3.1-70b-short/bf16 | 1 | 3 | 2.24 | +0.18% | 0.966 | +0.20% | 969, 963, 966 |

逐对 op wall 中位：8b-long/fp16 [10.08, 10.10, 10.04]；8b-long/bf16 [9.78, 9.83, 9.77]；
70b-long/fp16 [9.94, 9.97, 9.83]；70b-long/bf16 [9.95, 9.98, 9.94]；short 四例均 2.21–2.24。

### 14.3 候选 vs current best 对比表（B2 结构化回流）

> 本表数值保留不改；「候选」一列是 round-12 的 **A/B 最优臂**，**未采纳**进交付件（见 §14.10）。

current best = §12 交付件（256 MiB，long cb=2 → 16 chunks / short cb=5 → 2 chunks，
`final-r11-split-v2`，sha `6ca13f73…`）；候选 = round-12 采纳臂（768 MiB，4/1 chunks）。
Task Duration 均为「一次 op 调用内全部 kernel 的 `Task Duration` 之和」中位；引擎占比列取
`PipeUtilization` 的 per-launch 中位 × n_chunks / device 求和（近似聚合，见 §14.8）。

| workload | current best r11 (us) | 候选 r12 (us) | Δ vs current best | cube/vec/mte2 占 device（b512 → b768） | workspace 峰值 (MB) | 精度 |
|---|---:|---:|---:|---|---:|---|
| llama-3.1-8b-short/fp16 | 1016.78 | **958.3** | **-5.75%** | 9.7/15.5/15.4 → 9.7/15.4/15.2 | 252 → 403 | §14.7 |
| llama-3.1-8b-short/bf16 | 1028.34 | **964.3** | **-6.23%** | 9.6/15.8/14.9 → 9.6/15.8/15.5 | 252 → 403 | §14.7 |
| llama-3.1-8b-long/fp16 | 7518.62 | **7156.0** | **-4.82%** | 9.7/15.8/15.4 → 9.6/15.2/18.4 | 201 → 805 | §14.7 |
| llama-3.1-8b-long/bf16 | 7531.92 | **7194.0** | **-4.49%** | 9.6/16.2/15.5 → 9.6/15.5/18.1 | 201 → 805 | §14.7 |
| llama-3.1-70b-short/fp16 | 1015.72 | **961.7** | **-5.32%** | 9.6/15.3/15.3 → 9.6/15.4/15.2 | 252 → 403 | §14.7 |
| llama-3.1-70b-short/bf16 | 1020.28 | **968.9** | **-5.04%** | 9.6/15.8/15.3 → 9.6/15.7/15.7 | 252 → 403 | §14.7 |
| llama-3.1-70b-long/fp16 | 7525.84 | **7200.6** | **-4.32%** | 9.6/15.8/15.7 → 9.6/15.1/18.3 | 201 → 805 | §14.7 |
| llama-3.1-70b-long/bf16 | 7545.28 | **7207.2** | **-4.48%** | 9.6/16.2/15.6 → 9.6/15.5/18.1 | 201 → 805 | §14.7 |

读法：**引擎忙时总量守恒**（long/fp16：Cube 698 → 690 µs、Vector 1144 → 1085 µs），
即候选没有减少任何计算或访存；device 求和随之只动 −0.4…−1.0%（噪声量级），
而 op wall 动 −7.3…−8.1% ⇒ 收益完全落在 **launch 数 21 → 12 的 host 间隙**上：
9 次少掉的 launch 摊出 ≈0.9 ms，隐含 ≈99 µs/launch，与 §13.3 的 ≈128 µs/launch 估计同族（本轮为其下界）。
mte2 占比由 15.4% 升到 18.4% 是**大 chunk 的单次 DMA 粒度变粗**（per-launch mte2 从 159 µs 涨到 329 µs，
而工作量只涨 1.6×），未转化为 device 损失。

### 14.4 决策

> 已被 §14.10 用户裁决取代（768 MiB 不采纳，交付件回退 256 MiB）。

采纳值：`budget = 768 MiB`；`cb` long/short = **8 / 8**；`n_chunks` long/short = **4 / 1**；
main launch 数/call = **12 / 3**；含 glue 的 task 数/call = **19 / 10**（long/short）。

> **重要更正（本轮新发现，详见 §14.7）**：`bench.py --check` 的 CPU fp32 golden 判定证明
> **baseline 臂 b512 与候选 b576/b672 的 dk/dv 是错的**（`cb` 不能整除 `n_kv` 时最后一个
> 非满 KV chunk 会损坏 dk/dv），只有 `cb ∈ {2, 8}`（整除）的 arm 与 r11 已验证臂逐块一致。
> 因此下表的 wall/device 仍是**同代码量**的有效计时比较，但 512/576/672 三个 arm
> **不作为可选配置**；采纳臂 b768 另外单独通过了 golden 与全量 L0/L1（§14.7）。

判定链（按 §14.1 的协议，只用同 run 交错数字）：

| arm | op wall 中位（4 个 long 例） | device 逐例 Δ vs b512 | golden（8b-long/fp16） | 编译 | 判定 |
|---|---|---|---|---|---|
| b512 | 基线 10.95 / 10.57 / 10.77 / 10.83 ms | +0.00% | **FAIL**（dk 3.381e+00） | 8/8 | 基线；**数值无效，不可选** |
| b576 | −3.39 / −3.14 / −3.12 / −3.19% | −1.43 … +0.61% | **FAIL**（dk 3.381e+00） | 8/8 | 拒绝：数值无效 + wall 收益仅 ~1/3 |
| b672 | −5.13 / −4.73 / −4.84 / −5.07% | −0.55 … +0.83% | **FAIL**（dk 3.381e+00） | 8/8 | 拒绝：数值无效 + wall 收益居中 |
| **b768** | **−8.14 / −7.33 / −7.79 / −8.09%** | **−1.01 … +0.86%** | **PASS**（dk 1.006e-03） | 8/8 | **采纳：wall 最低 + device 无例超 +3% 门 + 数值正确** |

- 4 个 long 例的 op wall 在 b768 上**逐对一致**（三对之间仅 0.1–0.4% 抖动），不是单对侥幸；
  short 四例在 4 个 arm 间全部落在 ±0.3%，符合「≥384 MiB 后 short 已是 1 chunk」的结构预期。
- device 门（逐例回退 >3% 不采纳）**四个 arm 全部通过**，最大回退是 b768 在 8b-short/fp16 的 **+0.86%**
  （962 vs 954 µs），且该例 3 个 rep 本身跨 956–970 µs（≈1.5% 抖动）⇒ 判为噪声。
- 目标类型：本轮无用户指定的逐例数值目标，采纳判据即协议本身（wall 最低 + device 门），
  采纳臂同时是 **device 侧不劣化** 的唯一 wall 最低臂。

### 14.5 交付件落盘（唯一改动点）

> 已被 §14.10 用户裁决取代（768 MiB 不采纳，交付件回退 256 MiB）。

`perf_opt/_gqa_bwd_kernel.py`：`_SPLIT_WS_BUDGET_BYTES` 由 `256 * 1024 * 1024` 改为 `768 * 1024 * 1024`，
并在同位置写入 round-12 依据注释（含 1 GiB/cb=10 禁试结论）。

- 旧 sha256：`6ca13f73ccb22ffbc9258d957320bae9d93105254ddc11d03c778d91bde92179`
- **新 sha256：`b8e1fd206376da97948fe6a76f8b35545b79df4e8aa3c76a3a138335efe86617`**
- `diff` 摘要（对照 `logs/r12_launch/_gqa_bwd_kernel_pre_r12.py.bak`）：**1 行删除（常量行）+ 11 行新增
  （10 行注释 + 1 行常量）= 12 个变化行**，hunk 只有 `@@ -476,7 +476,17 @@`；无任何代码行变化。

### 14.6 Final Performance Test Data（round 12 当前最终表；§12 的同名表已标注为 round-11 历史）

> 已被 §14.10 用户裁决取代（768 MiB 不采纳，交付件回退 256 MiB）。

交付件 = `perf_opt/_gqa_bwd_kernel.py`（sha `b8e1fd20…`），candidate `final-r12-b768`，
口径 = `msprof op` 一次 op 调用内全部 kernel 的 `Task Duration` 之和（`--calls 5`、`--warm-up 1` 中位，
`logs/r12_launch/final.csv` + `profiles/r12_final_p1_*`）。baseline 列 = Stage 4 基线（r6 融合臂，
`perf_records.jsonl` 中 `round=11, phase=baseline` 行，sha `8344d214…`）。

### Final Performance Test Data（round 12 A/B 记录，未采纳 —— 交付件仍为 round-11 的 256 MiB）

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| _gqa_bwd_main | llama-3.1-8b-short/fp16 | 2011.57 | 958.3 | final-r12-b768 |
| _gqa_bwd_main | llama-3.1-8b-short/bf16 | 2012.32 | 964.3 | final-r12-b768 |
| _gqa_bwd_main | llama-3.1-8b-long/fp16 | 13954.16 | 7156.0 | final-r12-b768 |
| _gqa_bwd_main | llama-3.1-8b-long/bf16 | 13995.79 | 7194.0 | final-r12-b768 |
| _gqa_bwd_main | llama-3.1-70b-short/fp16 | 2017.08 | 961.7 | final-r12-b768 |
| _gqa_bwd_main | llama-3.1-70b-short/bf16 | 2013.05 | 968.9 | final-r12-b768 |
| _gqa_bwd_main | llama-3.1-70b-long/fp16 | 14018.46 | 7200.6 | final-r12-b768 |
| _gqa_bwd_main | llama-3.1-70b-long/bf16 | 13981.99 | 7207.2 | final-r12-b768 |

- 相对 Stage-4 基线：**8 例几何均值 −50.41%**（单例 −48.45…−52.36%，无回退）。
- 相对 §12 的 round-11 最终版（256 MiB）：**8 例几何均值 −5.06%**（单例 −4.32…−6.23%）。
- final 组与 §14.2.4 的 3-rep sweep 中位一致到 0.3% 以内（7156.0 vs 7164.9、7194.0 vs 7180.1、
  7200.6 vs 7214.0、7207.2 vs 7209.2、四个 short 958.3/964.3/961.7/968.9 vs 962.0/960.3/964.7/966.1）
  ⇒ final 行与 sweep 属同一候选、同一量级。
- **采纳状态（§14.10）**：本表 `final_us` 采自 round-12 应用版 `_gqa_bwd_kernel.py`（sha `b8e1fd20…`，
  768 MiB），该版本**已被用户裁决回退**；表内数字作为 A/B 实验记录保留，**不代表当前交付件**。
  当前交付件 = 256 MiB 版（sha `6ca13f73…`），其交付口径表见 §12。

### 14.7 精度回归与正确性（本轮关键发现）

#### 14.7.1 交付件全量回归（`logs/r12_gate/all.log`）

命令：`TILELANG_ASCEND_MODE=Developer PYTHONPATH=… /workspace/venv/bin/python _gqa_bwd_kernel.py --level all`
（cwd = `perf_opt/`，即交付件自身；`perf_opt/_gqa_bwd_kernel.py` sha `b8e1fd20…`）

| 层级 | 结果 | 说明 |
|---|---|---|
| L0 | **ALL PASS：23 项 PASS / 0 FAIL** | 16 个 L0 主项 + `L0-12/zeros`、`L0-12/lse-inf`、`L0-13` 子项等；含 `S=300/1000/1024/2048/4096` 与 causal/non-causal、fp16/bf16 |
| L1 | **10/10 PASS**（1 项 tier-2 relaxed，`Summary tier2_relaxed=1 ['L1']`） | 与 §12.1 的 L1 结论一致 |
| L2 | **6 PASS + 1 WARN** | WARN = `(B,S,H,Hkv,D)=(1,32,4,4,64)` 的 `bishengir-compile` SIGSEGV（`err code: -11`，`TRAP-bishengir-crash-s32` 既有工具链缺陷），**不计入本轮回归** |
| Boundary | 1 PASS + **1 WARN** | WARN = `×100` 量级大输入 `tier-3 classification rejected`（fp32 抵消，record-only；与 §12.1 同类，非回归） |
| 总结 | **EXIT=0 / `All check passed!`**，`cases=33` | 耗时：L0 1895.3s / L1 358.2s / L2 166.8s / Boundary 18.4s |

#### 14.7.2 逐 arm golden（`bench.py --check`，CPU fp32 golden；`logs/r12_gate/check_long_arms.log`）

| arm | `cb`（long/short） | `llama-3.1-8b-long/fp16` | `llama-3.1-8b-short/fp16` |
|---|---|---|---|
| 256 MiB（r11 交付件 `6ca13f73…`） | 2 / 5 | **PASS**（dq 1.017e-03 / dk 1.006e-03 / dv 1.120e-03，tier1_viol=0） | **FAIL**（dk 3.433e+00 / dv 5.458e+00，tier1_viol 1 536 439） |
| b512 | 5 / 8 | **FAIL**（dk 3.381e+00 / dv 6.406e+00，tier1_viol 766 034） | 未逐跑（同 `cb=5` 机制，见 14.7.3） |
| b576 | 6 / 8 | **FAIL**（dk 3.381e+00 / dv 6.406e+00，tier1_viol 1 017 328） | 未逐跑 |
| b672 | 7 / 8 | **FAIL**（dk 3.381e+00 / dv 6.406e+00，tier1_viol 766 034） | 未逐跑 |
| **b768 = 交付件** | **8 / 8** | **PASS**（dq 1.017e-03 / dk 1.006e-03 / dv 1.120e-03） | **PASS**（dq 1.152e-03 / dk 8.274e-04 / dv 1.118e-03） |

三例 FAIL 的 `dq` 全部为 `1.017e-03`、`tier1_viol=0`（与 PASS 臂完全同值）⇒ **缺陷只在 dk/dv**。

> 注（§14.10）：表中「b768 = 交付件」指 round-12 当时写入交付件的版本（sha `b8e1fd20…`）；
> 该版本已按用户裁决回退，b768 现为 **A/B 实验臂**。第一行的 256 MiB 才是当前交付件。

#### 14.7.3 缺陷定位（确定性，非竞态）

- **划分等价性探针**（`logs/r12_launch/probe_r12_arm_agree.py`，`logs/r12_gate/arm_agree.log`）：
  同一输入下与 256 MiB 臂比较，`llama-3.1-8b-long/fp16` 上 b512/b576/b672 的 `dk/dv` 差异
  **三臂完全相同**（`dk 1.413e+03`、`dv 6.567e+02`），b768 则 `dk/dv` **逐 bit 相等**、`dq` 仅 1.2e-04
  （fp32 区域原子累加次序噪声）。`70b-long/bf16` 上四臂 `dk/dv` 全部逐 bit 相等（`dq` 0…1.2e-04）。
- **确定性**（`logs/r12_launch/probe_r12_partial_chunk.py`）：256 MiB 臂连跑 2 次、b512 连跑 2 次，
  `dk/dv` 差异逐次完全相同（1.413e+03 / 6.567e+02），臂内 3 次重复 spread = 0 ⇒ **确定性缺陷，不是竞态**。
- **局部性**：b512 的 dk 差异集中在**前几个 KV 块**（kv0 1.413e3、kv1 3.45e2、kv2 5.78e2；
  32 个块里只有 3 块 > 1e-2），而非全张量弥散。
- **触发条件的经验式**：`cb ∤ n_kv`（最后一段 KV chunk 不满）。已实证：fp16 long（`n_kv=32`，cb=5/6/7）
  与 fp16 short（`n_kv=8`，cb=5）**损坏**；`cb ∈ {2, 8}`（整除 32 与 8）**正确**。
  **反例（如实记录）**：`70b-long/bf16` 上 cb=5/6/7 与 cb=2 逐 bit 一致 —— 即该触发条件
  不是「必然损坏」，本轮的实证范围是 fp16 两例 + bf16 一例未复现；根因未定位（本轮 kernel 体冻结）。

#### 14.7.4 结论与影响

1. **采纳臂 b768 数值正确**：`cb=8` 同时整除 long 的 `n_kv=32` 与 short 的 `n_kv=8`；
   交付件在 4 个 manifest 精度面（8b/70b × long/short × fp16/bf16 中的 long/fp16、short/fp16 直测，
   与 bf16 的 bit 一致性交叉）上无 golden 失败，并通过全量 L0/L1。
2. **r11 交付件（256 MiB）在 short manifest 精度面上是错的**（`cb=5`，`n_kv=8`）：
   §12 的 short 数字（1016.78 / 1028.34 / 1015.72 / 1020.28 µs）测自该错误配置。
   本轮采纳 768 MiB 同时**修复**了该缺陷，short 的 device 改善（−5.3…−6.2%）因此也包含
   「不再算错」这一层，不能全部归因于 launch 数。
3. **512/576/672 三个 arm 不进 gate 台账**：24 条测量行以 `phase=invalid`、`l0_pass=false`、
   附 `invalid_reason` 写入 `perf_records_split_detail.jsonl`（非 gate 范围）；`perf_records.jsonl`
   只保留 b768 candidate 行与 final 行（`perf_lint` 直调：`failures=0 / warnings=0`）。
4. **待修复项（P1，移交后续轮次）**：`_split_plan` 目前对 `cb ∤ n_kv` 没有任何守卫。
   建议最小加固 = host 侧一行断言/夹取（`cb = min(cb, largest divisor of n_kv ≤ cb)` 或直接
   `assert n_kv % cb == 0`），或在 pass-3 处理非满 chunk。本轮按纪律**未改动 kernel 体/split 数学**。

### 14.8 口径与限制

- **short 档不敏感**：预算 ≥384 MiB 后 `cb` 一律被 `n_kv=8` 夹取为 8 → 1 chunk，
  四个 arm 在 short 上的 wall Δ 全在 ±0.3%、device Δ 全在 ±0.9% ⇒ short 数据**只作噪声刻度**，
  不用于区分 arm。
- **device 噪声**：同 (case,arm) 的 3 个 rep 跨 0.1–0.8%（如 b768/8b-long/bf16 = 7180 / 7169 / 7240 µs），
  故 device 侧 ≤1% 的差异**不可分辨**；能分辨 arm 的是 op wall（逐对之间 0.1–0.4% 抖动，
  4 个 long 例的 −3…−8% 远超该抖动）。
- **编译 lottery**：本轮 32 个 (arm,shape) 全部 1 次编译通过（0 MISS、无重试），op-wall 驱动内编译
  0.2–8.6 s；`1 GiB / cb=10` 未尝试（已知 SIGSEGV）。但编译器行为仍是彩票性质（§13.1 记录过
  基线的同签名崩溃），本轮的「8/8 通过」不构成长期保证。
- **Fwd 阻塞不影响 bwd 面**：fwd 侧 `bishengir-compile` 的 MIX module SIGSEGV（§13.1/§13.4）是
  **fwd** 的阻塞项；本轮全部测量与精度回归都在 bwd 面完成，不受其影响；正式验收状态仍为
  `BLOCKED_ENVIRONMENT`（见 §13.4），本轮不改变该状态。
- **工具链戳**：tilelang `0.1.2+16f1ee4578a66e9123bb32783956e272df17ecad`、CANN 9.0.0、
  Ascend910B2（`npu-smi 26.1.1`）、日期 2026-10-22。`perf_records.jsonl` 的 round-12 行
  `device=Ascend910B2` / `commit=16f1ee4`。
- **op wall 与 device 不可混用**：op wall 含 host 下发间隙；device 求和只含 `Task Duration`。
  本轮的 −8% 类收益**只在 op wall 口径成立**；device 口径的收益是 −0.4…−1.0%（long）与
  −5.3…−6.2%（short vs r11 的 256 MiB 配置）。

### 14.9 知识条目重验与遗留

- `PL-attention-bwd-engine-phase-split`（`kb_stale_check` 判 stale：工具链戳漂移）本轮**实测重验**：
  条目的「workspace 预算 256 MB」是保守下界而非上限 —— 768 MiB 在同 8 例上 device −5.06%（vs 256 MiB）
  且 long 的 op wall −8%，同时**必须补齐硬约束**：预算换算出的 `cb` 必须整除 `n_kv`，
  否则 pass-3 的 dk/dv 静默错误（§14.7.3）。建议下一轮蒸馏时把这两条并入该条目（含本轮 repro：
  `logs/r12_launch/probe_r12_partial_chunk.py` + `bench.py --check`）。
- `CONST-vector-launch-overhead` 线索：本轮隐含 host 间隙 ≈99 µs/launch（9 次 launch 差 ≈0.9 ms），
  与 §13.3 的 ≈128 µs/launch 同族，支持该常数用于 launch 数决策。
- 遗留（P1）：pass-3 非满 chunk 的 dk/dv 缺陷未修复（kernel 体本轮冻结）；`70b-long/bf16` 未复现该缺陷，
  根因待定位。建议下一次 `mode=full` 调度优先处理该项（加固 + 覆盖非满 chunk 的 L0）。

### 14.10 用户裁决与回退（2026-10-22，append-only）

#### 14.10.1 用户原话

> `0.7a 的结果不要采纳到交付件，如果已经把 512 MB 写进文件，改回来，并记录这次 A/B。`

（原文照录，未做改写。）

#### 14.10.2 裁决落点

1. round-12 的 **768 MiB 臂（0.75 GiB ≈ 0.7）不采纳**到交付件。§14.2.4、§14.3、§14.4、§14.5、§14.6 中
   以「采纳」口吻写下的结论一律作废，仅作历史记录保留（各小节标题/正文已加注记：「已被 §14.10 取代」）。
2. 交付件回退到本轮前的**原始 256 MiB 版**：`_SPLIT_WS_BUDGET_BYTES = 256 * 1024 * 1024`，
   sha256 = `6ca13f73ccb22ffbc9258d957320bae9d93105254ddc11d03c778d91bde92179`，恢复自
   `logs/r12_launch/_gqa_bwd_kernel_pre_r12.py.bak`；`cmp` 与 `diff` 均空（exit 0），
   证据 `logs/r12_revert/revert_checks.txt`（含恢复后 sha、cmp/diff 退出码、常量行、行数对账）。
3. **本轮从未把 512 MiB 写进交付件**：512 MiB 只存在于实验臂文件
   `perf_opt/_gqa_bwd_kernel_r12_b512.py`（同类另有 `_r12_b576/_r12_b672/_r12_b768.py`），交付件字节从未
   等于 512 MiB 版本；实际被写进交付件的是 768 MiB 应用版（sha `b8e1fd20…`）。
   因此用户要求的「改回来」在执行面上 = 把 768 MiB 应用版回退到 256 MiB 原版。

#### 14.10.3 台账

- `perf_opt/perf_records.jsonl` 回退到 **40 行**（sha256 `3d2368b6…`），与
  `logs/r12_launch/perf_records.jsonl.pre_r12` 逐字节一致；其 final 行
  （`round=11`、`candidate_id=final-r11-split-v2`）的 `artifact_sha256 = 6ca13f73…`，
  覆盖 8/8 tune workload。行数、字段均未改写。
- round-12 的测量数据**完整保留、未删除任何一行**：`perf_opt/perf_records_r12_ab.jsonl`
  （40 行 = b512/b576/b672/b768 × 8 例的 candidate 行 32 行 + `final-r12-b768` 8 行，
  逐行 `deliverable_applied: false`）；逐 arm 逐例明细另见本节 §14.2/§14.3 表与
  `perf_opt/perf_records_split_detail.jsonl`。
- 决策文件 `logs/r12_launch/r12_decision.json` 增补 `"applied": false`、
  `"applied_reason": "user decision 2026-10-22: 0.7GiB (768 MiB) arm not adopted; deliverable reverted to 256 MiB (sha 6ca13f73…)"`、
  `"deliverable_budget_mib": 256`；原 A/B 判据字段（`rule`、逐例 device/op-wall delta、`compile`）原样保留。

#### 14.10.4 A/B 结论的有效范围

8 例三列（wall / device / task）数据仍作为**实验记录**有效，但**不构成交付决策**：

- op wall：long 四例相对 256 MiB 基线 −8.14% / −7.33% / −7.79% / −8.09%；short 四例 ±0.3% 内（不敏感）。
- device（`Task Duration` 求和）：long 逐例改善 −0.44…−1.01%，short 逐例 ≤ +0.86%（最差 +0.86%，在噪声内）。
- 即：可分辨的收益只在 **long 面的 op-wall**，且该收益已被用户裁决**不采纳**；short 面在两种口径下都不可分辨。

#### 14.10.5 交付件当前状态与已知缺陷（本轮不回退该发现、也不修）

- 当前交付件 = 256 MiB 版（sha `6ca13f73…`）。§14.7.3 的 pass-3 dk/dv 缺陷
  （触发面：short 档 `cb = 5 ∤ n_kv = 8`；确定性、非竞态；repro
  `logs/r12_launch/probe_r12_partial_chunk.py`）在该版本上**仍然存在**，
  属 **P1 待用户批准的新一轮修复**（建议最小加固：host 侧守卫 `n_kv % cb == 0`）。

#### 14.10.6 精度

- 回退后针对**该字节版本**重跑 L0：`logs/r12_revert/l0.log` → `[L0] ALL PASS (16 items)`、
  `All check passed!`（elapsed 253.2s，cases=16、tier2_relaxed=0）。
- L1/L2 依据 `logs/r11_gate/all_v2.log`（同一 sha `6ca13f73…` 的全量 `--level all` 绿）；
  本轮**未重跑** `--level all`，理由：字节级恢复到已验证版本（`cmp`/`diff` 为空）。

#### 14.10.7 变更边界

- 未改 wrapper（baseline/perf_opt 双 import 切换块未动）、未跑 report、未同步
  `/workspace/operator-migration`、未触碰 `.stage_state.json`。
- 未新增或删除任何 arm 变体文件、历史 profile 与实验日志；本节是本轮唯一新增的文字记录
  （回退前的 `opt_log.md` 快照留档于 `logs/r12_revert/opt_log.md.pre_1410`）。

## 15. round 13：split 分块不变量修复（`cb | n_kv`）（append-only）

> 本节只追加，不改写 §1–§14 的任何数字与结论。交付件在 §14.10 的 256 MiB 版本
> （sha `6ca13f73…`）基础上只改 `_split_plan` 一个函数体；kernel 体、签名、workspace dtype、
> 预算常量一律未动（`_SPLIT_WS_BUDGET_BYTES = 256 * 1024 * 1024`）。

### 15.0 缺陷与证据（承接 §14.7.3 的 P1）

- **缺陷**：`_split_plan` 用预算换算出 `cb` 后只做 `min(cb, n_kv)`，对 `cb ∤ n_kv` 没有任何守卫。
  short 档（`n_kv = 8`）预算推出 `cb0 = 5`，`5 ∤ 8` ⇒ 最后一个 chunk 只有 3 个 KV block。
- **结构根因（源码层，本轮逐行核对）**：三个 pass 都用 **chunk 内局部** KV-block 偏移索引
  workspace（`jby = uid // bh`、`by = by0 + jby`，`uid < bh * cb`），pass 3 按「整 chunk」写
  `dk/dv` 的 `cb` 个 KV-block 行。chunk 不满时 `by0 + cb - 1 > n_kv - 1`：
  末尾几行写越界/错位，`dk/dv` 静默错误（`dq` 走区域原子写，不受影响）。
- **既有证据（r12）**：`logs/r12_launch/probe_r12_partial_chunk.py` 连跑两次差异逐次完全相同
  （`dk 1.413e+03` / `dv 6.567e+02`）⇒ **确定性缺陷，非竞态**；b512/b576/b672 三臂 `dk/dv`
  差异完全相同；`dq` 恒为 1.017e-03（与正确臂同值）⇒ 缺陷只落在 `dk/dv`。
- **本轮复现（P 臂 = 修复前快照 `logs/r13/_gqa_bwd_kernel_pre_r13.py`，sha `6ca13f73…`）**：
  short 两例 fp16 的 `bench.py --check` 见 15.2.1 的 P 行（`dk/dv` 3.4e+0/5.5e+0 量级，
  `tier1_viol` 百万级）——与本轮修复后的 `dk/dv` 逐项对照即「修复前 vs 修复后」的直接证据。

### 15.1 修复与不变量

```diff
     cb = max(1, _SPLIT_WS_BUDGET_BYTES // max(per_kv, 1))
     cb = min(cb, n_kv)
+    while n_kv % cb:               # chunk boundary must not split a KV block's
+        cb -= 1                    # owner: dk/dv single-writer invariant
     return cb, n_kv, n_q, bh * cb * n_q
```

外加一段 docstring 说明不变量（`diff` 共 20 行，`logs/r13/split_plan_fix.diff`）。要点：

1. **不变量**：`n_kv % cb == 0` ⇒ 每个 `(b, h, kv-block)` 的属主完整落在单个 chunk 内 ⇒
   `dk/dv` 每个元素只被一个 pass-3 launch 写一次 ⇒ 其 **fp32 累加顺序与 chunk 划分无关**
   （15.2.3 逐 bit 验证）。
2. **只减不增**：`cb` 从预算允许值向下取 `n_kv` 的最大因子，因此 `ws = cb · per_kv ≤ 预算` 恒成立
   （表格的 `ws` 列），不会把 workspace 撑过预算。
3. **终止性**：`cb ≥ 1` 且 `1 | n_kv`，循环必停；`n_kv = 1`（极小 shape）直接跳过。

**派生 `cb` / chunks / launches 表**（`block_m=bn=64` 为 resolve 后的有效 tile；
`n_kv = S/64`，long `n_kv=32, per_kv=96 MiB`，short `n_kv=8, per_kv=48 MiB`；
`main launches = 3 passes × chunks`）：

| 档 | 臂 | budget | `cb0`（预算→min n_kv） | `cb`（修复后） | `cb \| n_kv` | chunks | main launches | ws |
|---|---|---|---|---|---|---|---|---|
| long | **P**（修复前，256 MiB） | 256 MiB | 2 | 2 | 是 | 16 | 48 | 192 MiB |
| long | **F**（交付件，256 MiB） | 256 MiB | 2 | 2 | 是 | 16 | 48 | 192 MiB |
| long | b512 | 512 MiB | 5 | **4**（5∤32→4） | 是 | 8 | 24 | 384 MiB |
| long | b576 | 576 MiB | 6 | **4**（6→5→4） | 是 | 8 | 24 | 384 MiB |
| long | b672 | 672 MiB | 7 | **4**（7→6→5→4） | 是 | 8 | 24 | 384 MiB |
| long | b768 | 768 MiB | 8 | 8 | 是 | 4 | 12 | 768 MiB |
| short | **P**（修复前） | 256 MiB | 5 | **5（5∤8，缺陷）** | **否** | 2 | 6 | 96 MiB |
| short | **F**（交付件） | 256 MiB | 5 | **4**（5∤8→4） | 是 | 2 | 6 | 96 MiB |
| short | b512/b576/b672/b768 | ≥512 MiB | 8（被 `n_kv` 夹取） | 8 | 是 | 1 | 3 | 384 MiB |

读法要点：

- **P 与 F 在 long 上完全同构**（两臂 `cb = 2`、16 chunks、48 launches、代码路径逐行相同），
  所以 long 的 A/B 是**纯测量重复性核对**（预期 Δ≈0），不是优化对比。
- **short 上 P 是错误配置**（`cb = 5`），F 只把 `cb` 降到 4：chunks 与 launch 数都不变
  （2 chunks / 6 launches），所以 **launch 数不是变量**；实测 F 在 short 上快 1.66–2.57%（device，
  4/4 例同向），归因于分块均衡度 `[5,3]` → `[4,4]`（**推断**，未做隔离实验），详见 15.3。
- **四个臂在修复后塌缩成两种配置**：b512/b576/b672 都是 long `cb=4`/short `cb=8`，
  b768 是 long `cb=8`/short `cb=8`。因此 32 例 golden 里 24 例是同一配置的三份拷贝
  （文件名不同、`_SPLIT_WS_BUDGET_BYTES` 不同，但 `_split_plan` 输出相同），
  仍按任务要求逐臂逐例跑完，作为编译彩票/确定性的一次重复抽样。

### 15.2 回归（分层判据：P1 golden sweep + P2 逐位一致 + 交付件 L0）

#### 15.2.1 4 臂 × 8 例 golden sweep（`logs/r13/checks_arms.log` → `checks_arms.tsv`）

口径：`bench.py --check`（模块自带 CPU fp32 golden，`TILELANG_ASCEND_MODE=Developer`），
每例打印 dq/dk/dv 的 `max_diff` / `tier1_viol` / `tier2_viol`。**有效配置**：`bench.py` 打印的
`cfg={'block_m': 64, 'block_n': 32, 'num_stages': 1}` 是 `_resolve_config` **之前**的 wrapper 默认；
`dim=128`、`causal`、`S ∈ {512, 2048}`（均为 64 的整数倍）经 `_resolve_config` 后统一映射为
**`block_m/bn/ns = 64/64/1`**（即 `_split_plan` 实际使用的 tile；下文所有 `n_kv`、`cb` 都按它算）。

判据分层（**不混用**）：

- **fp16**：严格 `tier2_viol == 0`（三个输出全部为 0 才 PASS）。
- **bf16**：`bench.py --check` 的 2-ULP 判据不适用（15.2.4 给出对照证明），改用
  **①** 与「整除配置参考」的 `dk/dv` **逐 bit 相等**（15.2.3），**②**
  `max_diff ≤ 1.5e-2` 且 `tier1_viol / 元素数 < 0.5%`（tier-3 envelope）。
  同时把 `bench.py --check` 的**原始 tier2 数字原样记录**在表里，不隐藏、不改写。

（表格由 `logs/r13/summarize_arms.py` 从 TSV 生成，下面是原文输出。）

### raw sweep (bench.py --check numbers, parsed from checks_arms.tsv)

| arm | case | dq max_diff / t2 | dk max_diff / t2 | dv max_diff / t2 | bench |
|---|---|---|---|---|---|
| b512 | llama-3.1-8b-short/fp16 | 1.152e-03 / 0 | 8.274e-04 / 0 | 1.118e-03 / 0 | PASS |
| b512 | llama-3.1-8b-short/bf16 | 6.933e-03 / 0 | 7.007e-03 / 0 | 1.000e-02 / 2 | FAIL |
| b512 | llama-3.1-8b-long/fp16 | 1.017e-03 / 0 | 1.006e-03 / 0 | 1.120e-03 / 0 | PASS |
| b512 | llama-3.1-8b-long/bf16 | 7.572e-03 / 1 | 7.843e-03 / 0 | 8.794e-03 / 4 | FAIL |
| b512 | llama-3.1-70b-short/fp16 | 8.627e-04 / 0 | 9.639e-04 / 0 | 1.320e-03 / 0 | PASS |
| b512 | llama-3.1-70b-short/bf16 | 8.280e-03 / 0 | 6.731e-03 / 0 | 1.037e-02 / 2 | FAIL |
| b512 | llama-3.1-70b-long/fp16 | 1.046e-03 / 0 | 1.533e-03 / 0 | 1.128e-03 / 0 | PASS |
| b512 | llama-3.1-70b-long/bf16 | 7.146e-03 / 0 | 7.388e-03 / 0 | 8.981e-03 / 3 | FAIL |
| b576 | llama-3.1-8b-short/fp16 | 1.152e-03 / 0 | 8.274e-04 / 0 | 1.118e-03 / 0 | PASS |
| b576 | llama-3.1-8b-short/bf16 | 6.933e-03 / 0 | 7.007e-03 / 0 | 1.000e-02 / 2 | FAIL |
| b576 | llama-3.1-8b-long/fp16 | 1.017e-03 / 0 | 1.006e-03 / 0 | 1.120e-03 / 0 | PASS |
| b576 | llama-3.1-8b-long/bf16 | 7.572e-03 / 1 | 7.843e-03 / 0 | 8.794e-03 / 4 | FAIL |
| b576 | llama-3.1-70b-short/fp16 | 8.627e-04 / 0 | 9.639e-04 / 0 | 1.320e-03 / 0 | PASS |
| b576 | llama-3.1-70b-short/bf16 | 8.280e-03 / 0 | 6.731e-03 / 0 | 1.037e-02 / 2 | FAIL |
| b576 | llama-3.1-70b-long/fp16 | 1.046e-03 / 0 | 1.533e-03 / 0 | 1.128e-03 / 0 | PASS |
| b576 | llama-3.1-70b-long/bf16 | 7.146e-03 / 0 | 7.388e-03 / 0 | 8.981e-03 / 3 | FAIL |
| b672 | llama-3.1-8b-short/fp16 | 1.152e-03 / 0 | 8.274e-04 / 0 | 1.118e-03 / 0 | PASS |
| b672 | llama-3.1-8b-short/bf16 | 6.933e-03 / 0 | 7.007e-03 / 0 | 1.000e-02 / 2 | FAIL |
| b672 | llama-3.1-8b-long/fp16 | 1.017e-03 / 0 | 1.006e-03 / 0 | 1.120e-03 / 0 | PASS |
| b672 | llama-3.1-8b-long/bf16 | 7.572e-03 / 1 | 7.843e-03 / 0 | 8.794e-03 / 4 | FAIL |
| b672 | llama-3.1-70b-short/fp16 | 8.627e-04 / 0 | 9.639e-04 / 0 | 1.320e-03 / 0 | PASS |
| b672 | llama-3.1-70b-short/bf16 | 8.280e-03 / 0 | 6.731e-03 / 0 | 1.037e-02 / 2 | FAIL |
| b672 | llama-3.1-70b-long/fp16 | 1.046e-03 / 0 | 1.533e-03 / 0 | 1.128e-03 / 0 | PASS |
| b672 | llama-3.1-70b-long/bf16 | 7.146e-03 / 0 | 7.388e-03 / 0 | 8.981e-03 / 3 | FAIL |
| b768 | llama-3.1-8b-short/fp16 | 1.152e-03 / 0 | 8.274e-04 / 0 | 1.118e-03 / 0 | PASS |
| b768 | llama-3.1-8b-short/bf16 | 6.933e-03 / 0 | 7.007e-03 / 0 | 1.000e-02 / 2 | FAIL |
| b768 | llama-3.1-8b-long/fp16 | 1.017e-03 / 0 | 1.006e-03 / 0 | 1.120e-03 / 0 | PASS |
| b768 | llama-3.1-8b-long/bf16 | 7.572e-03 / 1 | 7.843e-03 / 0 | 8.794e-03 / 4 | FAIL |
| b768 | llama-3.1-70b-short/fp16 | 8.627e-04 / 0 | 9.639e-04 / 0 | 1.320e-03 / 0 | PASS |
| b768 | llama-3.1-70b-short/bf16 | 8.280e-03 / 0 | 6.731e-03 / 0 | 1.037e-02 / 2 | FAIL |
| b768 | llama-3.1-70b-long/fp16 | 1.046e-03 / 0 | 1.533e-03 / 0 | 1.128e-03 / 0 | PASS |
| b768 | llama-3.1-70b-long/bf16 | 7.146e-03 / 0 | 7.388e-03 / 0 | 8.981e-03 / 3 | FAIL |

### fp16 strict verdict (tier2_viol == 0)

| arm | case | dq t2 | dk t2 | dv t2 | verdict |
|---|---|---|---|---|---|
| b512 | llama-3.1-8b-short/fp16 | 0 | 0 | 0 | PASS |
| b512 | llama-3.1-8b-long/fp16 | 0 | 0 | 0 | PASS |
| b512 | llama-3.1-70b-short/fp16 | 0 | 0 | 0 | PASS |
| b512 | llama-3.1-70b-long/fp16 | 0 | 0 | 0 | PASS |
| b576 | llama-3.1-8b-short/fp16 | 0 | 0 | 0 | PASS |
| b576 | llama-3.1-8b-long/fp16 | 0 | 0 | 0 | PASS |
| b576 | llama-3.1-70b-short/fp16 | 0 | 0 | 0 | PASS |
| b576 | llama-3.1-70b-long/fp16 | 0 | 0 | 0 | PASS |
| b672 | llama-3.1-8b-short/fp16 | 0 | 0 | 0 | PASS |
| b672 | llama-3.1-8b-long/fp16 | 0 | 0 | 0 | PASS |
| b672 | llama-3.1-70b-short/fp16 | 0 | 0 | 0 | PASS |
| b672 | llama-3.1-70b-long/fp16 | 0 | 0 | 0 | PASS |
| b768 | llama-3.1-8b-short/fp16 | 0 | 0 | 0 | PASS |
| b768 | llama-3.1-8b-long/fp16 | 0 | 0 | 0 | PASS |
| b768 | llama-3.1-70b-short/fp16 | 0 | 0 | 0 | PASS |
| b768 | llama-3.1-70b-long/fp16 | 0 | 0 | 0 | PASS |

fp16 strict: **16/16 PASS**

### bf16 layered verdict (tier-3 envelope; bench 2-ULP recorded but not used)

| arm | case | worst max_diff | worst tier1_viol | tier1/elements | bench t2 (raw) | envelope |
|---|---|---|---|---|---|---|
| b512 | llama-3.1-8b-short/bf16 | 1.000e-02 | 270 | 0.0032% | 0/0/2 | PASS |
| b512 | llama-3.1-8b-long/bf16 | 8.794e-03 | 199 | 0.0012% | 1/0/4 | PASS |
| b512 | llama-3.1-70b-short/bf16 | 1.037e-02 | 295 | 0.0035% | 0/0/2 | PASS |
| b512 | llama-3.1-70b-long/bf16 | 8.981e-03 | 195 | 0.0012% | 0/0/3 | PASS |
| b576 | llama-3.1-8b-short/bf16 | 1.000e-02 | 270 | 0.0032% | 0/0/2 | PASS |
| b576 | llama-3.1-8b-long/bf16 | 8.794e-03 | 199 | 0.0012% | 1/0/4 | PASS |
| b576 | llama-3.1-70b-short/bf16 | 1.037e-02 | 295 | 0.0035% | 0/0/2 | PASS |
| b576 | llama-3.1-70b-long/bf16 | 8.981e-03 | 195 | 0.0012% | 0/0/3 | PASS |
| b672 | llama-3.1-8b-short/bf16 | 1.000e-02 | 270 | 0.0032% | 0/0/2 | PASS |
| b672 | llama-3.1-8b-long/bf16 | 8.794e-03 | 199 | 0.0012% | 1/0/4 | PASS |
| b672 | llama-3.1-70b-short/bf16 | 1.037e-02 | 295 | 0.0035% | 0/0/2 | PASS |
| b672 | llama-3.1-70b-long/bf16 | 8.981e-03 | 195 | 0.0012% | 0/0/3 | PASS |
| b768 | llama-3.1-8b-short/bf16 | 1.000e-02 | 270 | 0.0032% | 0/0/2 | PASS |
| b768 | llama-3.1-8b-long/bf16 | 8.794e-03 | 199 | 0.0012% | 1/0/4 | PASS |
| b768 | llama-3.1-70b-short/bf16 | 1.037e-02 | 295 | 0.0035% | 0/0/2 | PASS |
| b768 | llama-3.1-70b-long/bf16 | 8.981e-03 | 195 | 0.0012% | 0/0/3 | PASS |

bf16 envelope: **16/16 PASS** (bitwise arm-vs-reference evidence: logs/r13/cb_bitwise.log)

#### 15.2.2 新 shape（`logs/r13/new_shape_check.log`）

`(B,S,H,Hkv,D) = (2,768,32,32,64)` fp16：`n_kv = 12`、`n_q = 12`、`per_kv = 36 MiB`、`bh = 64`。
预算推出 `cb0 = 7`，`while` 分支收敛到 **`cb = 6`**（`12 % 7 ≠ 0 → 6`，`12 % 6 == 0` 停），
`n_chunks = 2`、`ntile = 4608`、`while_branch_used = True`、`tier2_viol = 0`、**PASS**；
3 次重复 `max spread = 5.96e-08`。
⇒ 修复对**非 2 的幂 `n_kv`** 也生效（这是「只修 short 档」之外最有价值的一条覆盖：
`n_kv = 12` 时 `cb0 = 7/8/9/10/11` 都会被收敛到 6）。

#### 15.2.3 short/long 的 `dk/dv` 逐位一致（P2，用户硬验收；`logs/r13/cb_bitwise.log`）

做法：同一输入、同一有效 tile，把 `_split_plan` 强制成不同的 `cb`（都满足 `cb | n_kv`），
比较输出张量是否 `torch.equal`。参考配置 = 交付件本身的 `cb`（short 4 / long 2）。

| case | 配置对 | `dk` 逐 bit | `dv` 逐 bit | `dq` max\|diff\| |
|---|---|---|---|---|
| 8b-short/fp16 | cb=8 vs cb=4 | 相等 | 相等 | 5.960e-08 |
| 8b-short/fp16 | cb=1 vs cb=4 | 相等 | 相等 | 2.980e-08 |
| 8b-short/bf16 | cb=8 vs cb=4 | 相等 | 相等 | 5.960e-08 |
| 8b-short/bf16 | cb=1 vs cb=4 | 相等 | 相等 | 2.980e-08 |
| 70b-short/fp16 | cb=8 vs cb=4 | 相等 | 相等 | 5.960e-08 |
| 70b-short/fp16 | cb=1 vs cb=4 | 相等 | 相等 | 2.980e-08 |
| 70b-short/bf16 | cb=8 vs cb=4 | 相等 | 相等 | 5.960e-08 |
| 70b-short/bf16 | cb=1 vs cb=4 | 相等 | 相等 | 2.980e-08 |
| 8b-long/fp16 | cb=4 vs cb=2 | 相等 | 相等 | 1.526e-05 |
| 8b-long/fp16 | cb=8 vs cb=2 | 相等 | 相等 | 6.104e-05 |
| 8b-long/bf16 | cb=4 vs cb=2 | 相等 | 相等 | 6.104e-05 |
| 8b-long/bf16 | cb=8 vs cb=2 | 相等 | 相等 | 6.104e-05 |
| 70b-long/fp16 | cb=4 vs cb=2 | 相等 | 相等 | 0 |
| 70b-long/fp16 | cb=8 vs cb=2 | 相等 | 相等 | 1.221e-04 |
| 70b-long/bf16 | cb=4 vs cb=2 | 相等 | 相等 | 0 |
| 70b-long/bf16 | cb=8 vs cb=2 | 相等 | 相等 | 1.221e-04 |

结论：

1. **`dk/dv` 在 16 组配置对（32 个张量对）上全部逐 bit 相等** ⇒ 15.1 的单写者推理成立，
   修复只改变「工作划分」，不改变 `dk/dv` 的 fp32 累加顺序 ⇒ 不看 `cb` 的取值，
   `dk/dv` 都是同一个数。这也把 15.2.1 里 bf16 的臂结果与交付件配置连接起来
   （短档臂是 `cb=8`、交付件是 `cb=4`，两者 `dk/dv` 逐 bit 相等）。
2. **`dq` 不承诺逐 bit**：差异 2.98e-08…1.221e-04，量级与 r12 记录的「fp32 区域原子次序噪声」
   一致（§14.7.3 的 1.2e-04）；对 fp16/bf16 的 tier 判据无影响（15.2.1 的 `dq` 全部 tier1/tier2 通过）。

#### 15.2.4 bf16 判据对照：修复前 vs 修复后（`logs/r13/bf16_control.log`）

为证明「bf16 的 `tier2_viol > 0` 与本轮修复无关」，用**修复前**文件（sha `6ca13f73…`）在
**long** 两例上跑同一命令（long 的 `cb = 2` 本来就整除 32 ⇒ 该文件在 long 上是**正确配置**）：

```
$ python bench.py --impl ./logs/r13/_gqa_bwd_kernel_pre_r13.py --case llama-3.1-8b-long/bf16 --check
[check] llama-3.1-8b-long/bf16 cfg={'block_m': 64, 'block_n': 32, 'num_stages': 1} dq max_diff=7.572e-03 tier1_viol=26 tier2_viol=1
[check] llama-3.1-8b-long/bf16 cfg={'block_m': 64, 'block_n': 32, 'num_stages': 1} dk max_diff=7.843e-03 tier1_viol=35 tier2_viol=0
[check] llama-3.1-8b-long/bf16 cfg={'block_m': 64, 'block_n': 32, 'num_stages': 1} dv max_diff=8.794e-03 tier1_viol=199 tier2_viol=4
[check] llama-3.1-8b-long/bf16 FAIL
$ python bench.py --impl ./logs/r13/_gqa_bwd_kernel_pre_r13.py --case llama-3.1-70b-long/bf16 --check
[check] llama-3.1-70b-long/bf16 ... dq max_diff=7.146e-03 tier1_viol=21 tier2_viol=0
[check] llama-3.1-70b-long/bf16 ... dk max_diff=7.388e-03 tier1_viol=43 tier2_viol=0
[check] llama-3.1-70b-long/bf16 ... dv max_diff=8.981e-03 tier1_viol=195 tier2_viol=3
[check] llama-3.1-70b-long/bf16 FAIL
```

- 这两组数字与**修复后四个臂的 long/bf16 行逐位相同**（15.2.1）⇒ bf16 的 `tier2_viol`
  是「bf16 存储回落噪声 × bench.py 2-ULP 判据」的既有性质，`cb | n_kv` 修复**没有**引入它，
  也**没有**改变它。
- 同文件 long/fp16 两例：`dq 1.017e-03 / dk 1.006e-03 / dv 1.120e-03`，`tier1_viol=0`、
  `tier2_viol=0`（PASS）——与 b512/b576/b672/b768 及交付件的 long/fp16 完全一致。

#### 15.2.5 交付件 L0（`logs/r13/l0_deliverable.log`）

`python3 _gqa_bwd_kernel.py --level L0`（cwd = `perf_opt/`，交付件 sha `8f865fe5…`）：

- `[L0] ALL PASS (16 items)`、`All check passed!`、`cases=16 tier2_relaxed=0`、`elapsed 245.6s`。
- 覆盖 `S=300/1000`（尾块，走 `_resolve_config` 的 fallback 与 `while` 分支）、`S=2048/4096`、
  causal/non-causal、fp16/bf16；bf16 两项按 tier-3 通过（`L0-9 (1,300,8,2,64)` bf16
  `max_diff=9.872e-03 tier=3`；`L0-12/large-scores` fp16 tier=3）。
- `[L0-14]`：`dq/dk/dv bit_equal_3runs=True, max_spread=0.0`（该 shape 上）。

### 15.3 8 例 A/B（P / F / R，同 run 交错，禁止跨 run 比较）

- **wall**（`logs/r13/opwall.jsonl`，`r13_opwall.py`）：同一进程内构造 P/F/R 三个 op，
  逐对旋转顺序（`P,F,R` / `R,F,P` / `P,F,R`），3 对 × 每对每臂 10 次，
  每臂取全部计时的中位数；单位 µs = 一次 `op(q,k,v,o,do,lse)` 调用的 NPU event 墙钟。
- **device**（`logs/r13/device.csv`，`r13_device.sh` → `msprof op`）：一次 op 调用内**全部** kernel
  的 `Task Duration(us)` 之和，`calls=3`、逐 rep 旋转臂顺序、3 rep 取中位；
  R 臂仅 long 4 例 × 1 rep（参考）。
- 判据：`F vs P` 逐例 **device Δ ≤ +3%**。long 两臂同构（`cb=2`、48 launches、同一代码路径）
  ⇒ 该行只是测量重复性；short 两臂 launch 数也相同（6）⇒ 该行测的是「分块均衡度」而非 launch 数。

（表格由 `logs/r13/summarize_ab.py` 生成，下面是原文输出。）

| case | arm | main launches | op wall (us) | device (us) | device reps | dDevice vs P | dWall vs P |
|---|---|---|---|---|---|---|---|
| llama-3.1-8b-short/fp16 | P | 6 | 2535.1 | 1019.4 | 3 | +0.00% | +0.00% |
| llama-3.1-8b-short/fp16 | F | 6 | 2513.7 | 993.6 | 3 | -2.53% | -0.84% |
| llama-3.1-8b-short/fp16 | R | - | - | - | 0 | - | - |
| llama-3.1-8b-short/bf16 | P | 6 | 2599.3 | 1026.3 | 3 | +0.00% | +0.00% |
| llama-3.1-8b-short/bf16 | F | 6 | 2552.7 | 999.9 | 3 | -2.57% | -1.79% |
| llama-3.1-8b-short/bf16 | R | - | - | - | 0 | - | - |
| llama-3.1-8b-long/fp16 | P | 48 | 13104.3 | 7502.4 | 3 | +0.00% | +0.00% |
| llama-3.1-8b-long/fp16 | F | 48 | 13048.4 | 7506.5 | 3 | +0.05% | -0.43% |
| llama-3.1-8b-long/fp16 | R | 12 | 10139.4 | 7202.3 | 1 | -4.00% | -22.63% |
| llama-3.1-8b-long/bf16 | P | 48 | 12640.1 | 7531.0 | 3 | +0.00% | +0.00% |
| llama-3.1-8b-long/bf16 | F | 48 | 12611.4 | 7530.6 | 3 | -0.01% | -0.23% |
| llama-3.1-8b-long/bf16 | R | 12 | 10046.4 | 7210.4 | 1 | -4.26% | -20.52% |
| llama-3.1-70b-short/fp16 | P | 6 | 2579.8 | 1016.7 | 3 | +0.00% | +0.00% |
| llama-3.1-70b-short/fp16 | F | 6 | 2565.4 | 995.7 | 3 | -2.07% | -0.56% |
| llama-3.1-70b-short/fp16 | R | - | - | - | 0 | - | - |
| llama-3.1-70b-short/bf16 | P | 6 | 2493.3 | 1020.2 | 3 | +0.00% | +0.00% |
| llama-3.1-70b-short/bf16 | F | 6 | 2485.2 | 1003.3 | 3 | -1.66% | -0.32% |
| llama-3.1-70b-short/bf16 | R | - | - | - | 0 | - | - |
| llama-3.1-70b-long/fp16 | P | 48 | 12937.7 | 7512.9 | 3 | +0.00% | +0.00% |
| llama-3.1-70b-long/fp16 | F | 48 | 12901.4 | 7511.6 | 3 | -0.02% | -0.28% |
| llama-3.1-70b-long/fp16 | R | 12 | 10094.9 | 7176.1 | 1 | -4.48% | -21.97% |
| llama-3.1-70b-long/bf16 | P | 48 | 12859.8 | 7560.2 | 3 | +0.00% | +0.00% |
| llama-3.1-70b-long/bf16 | F | 48 | 12887.3 | 7542.7 | 3 | -0.23% | +0.21% |
| llama-3.1-70b-long/bf16 | R | 12 | 10116.6 | 7241.6 | 1 | -4.21% | -21.33% |

### acceptance: F vs P device delta <= +3%

| case | device delta | verdict |
|---|---|---|
| llama-3.1-8b-short/fp16 | -2.53% | PASS |
| llama-3.1-8b-short/bf16 | -2.57% | PASS |
| llama-3.1-8b-long/fp16 | +0.05% | PASS |
| llama-3.1-8b-long/bf16 | -0.01% | PASS |
| llama-3.1-70b-short/fp16 | -2.07% | PASS |
| llama-3.1-70b-short/bf16 | -1.66% | PASS |
| llama-3.1-70b-long/fp16 | -0.02% | PASS |
| llama-3.1-70b-long/bf16 | -0.23% | PASS |

overall: **PASS**

**读法与结论**：

1. **long 4 例 device Δ 全部在噪声内**（见上表），与「P/F 同构」的预期一致 ⇒ 修复对 long
   **零性能影响**，同时排除了「分块改动带来看不见的额外开销」。
2. **short 4 例 device Δ 为 −1.66…−2.57%（F 更快）**：修复只把 `cb` 从 5 降到 4，
   chunks/main launches 完全不变（2 chunks / 6 launches），因此这**不是** launch 数效应。
   超出 0.1–0.8% 的 device 噪声，方向一致（4/4 例 F 更快），但**归因是推断**：`cb = 5` 的划分是
   `[5, 3]`（尾 chunk 只有 3 个 KV block，块数不均衡），`cb = 4` 是 `[4, 4]`；本轮**未**做隔离
   实验证明这一归因（例如固定 `cb = 4` 但人为把最后一个 chunk 缩短）。因此本轮对外只声明：
   「修复**不劣化** short（−1.7…−2.6%，同 launch 数）」；**不声明** short 的性能收益。
   本轮性质仍是**正确性修复**（唯一实测收益在正确性账上）。
3. **R 臂（768 MiB，仅参考）**：long 4 例 wall `−20.5…−22.6%`、device `−4.00…−4.48%`
   （1 rep，只作趋势），但按 §14.10 用户裁决**不采纳**，也不进 gate 台账；
   列在这里只作为「launch 数（48 → 12）仍是 long 的主要成本项」的旁证。

### 15.4 限制与未验证项（如实披露）

1. **bf16 的 `bench.py --check` 判据不适用（既有问题，非本轮引入）**：`bench.py --check` 用
   `tier2_viol == 0`（2-ULP）作唯一 PASS 条件，而项目 L0 对「存储 dtype 回落」噪声用的是
   tier-3 envelope（本轮 L0 的 `L0-9` bf16 与 `L0-12/large-scores` 都是 `tier=3` PASS）。
   对照组 `logs/r13/bf16_control.log` 直接证明：**修复前**文件在 long 两例上本来就产出
   `cb = 2`（`2 | 32`，正确配置），其 bf16 `tier2_viol` 仍为 8b `1/0/4`、70b `0/0/3`，
   `max_diff` 8b `7.572e-03 / 7.843e-03 / 8.794e-03`、70b `7.146e-03 / 7.388e-03 / 8.981e-03`
   —— 与四个修复后臂的数字**逐位相同**。因此这条判据与 `cb | n_kv`
   无关；本轮 bf16 一律按 15.2.1 的分层判据（tier-3 envelope + 与整除参考配置逐 bit 对比）。
2. **R 臂只作参考**：768 MiB 是 §14.10 用户明确不采纳的方向，本轮仅按任务书在 long 4 例上采
   1 rep 作趋势参考，**不进 gate 台账**（`perf_records.jsonl` 的 final 只有 F）。其
   `−20.5…−22.6%`（op wall）/ `−4.00…−4.48%`（device）的优势**不构成采纳建议**。
3. **`1 GiB / cb = 10` 未尝试**（已知 `bishengir` SIGSEGV，任务书与本轮纪律均禁止）。
4. **P 臂的 short 数字不是「优化」证据**：P 在 short 上 `cb = 5 ∤ 8` 是错误配置，其 chunks 仍为 2、
   main launches 仍为 6（与 F 相同），所以 short 的 A/B（F −1.7…−2.6%）只能读作「同 launch 数下
   分块均衡度（`[5,3]` vs `[4,4]`）的刻度」；该归因为推断，未做隔离实验，**不声明**为性能收益。
5. **配置塌缩**：修复后 b512/b576/b672 三个臂的 `_split_plan` 输出完全相同（long `cb=4`/short `cb=8`），
   24 例 golden 是**同一配置的重复抽样**（用于覆盖率与确定性，不是三份独立证据）；只有 b768
   在 long 上不同（`cb=8`）。
6. **dq 的非确定性**：`dq` 走 AIV 区域 `T.atomic_add`，fp32 累加次序随 chunk 边界与运行次序变化，
   实测 run-to-run / 跨配置抖动 2.98e-08…1.221e-04；`dk/dv` 在全部 16 组配置对（32 个张量对）的跨配置比较中**逐 bit 相等**
   （重复运行亦稳定，交付件 `L0-14` 记录 `dk/dv/dq bit_equal_3runs=True`，spread 0）。
   探针里 `repeat-run bit_equal=False` 的用例差异归因于 `dq`（该归因为推断：跨配置比较中
   `dk/dv` 恒为逐 bit 相等，且探针只记录聚合布尔值，未逐张量分离）。
7. **测量噪声**：device 同 (case, arm) 的 3 rep 跨 0.1–0.8%，故 ≤1% 的 device 差异不可分辨；
   op wall 逐对抖动 0.1–0.4%。编译器行为仍是彩票性质（本轮 32+ 次编译 0 次崩溃，但 §13.1
   记录过基线同签名崩溃）。
8. **新 shape 与任务书的算术猜测不一致**：任务书写「预算推出 `cb0 = 7` → 收敛到 4」，
   实际 `12 % 7 ≠ 0 → 6`，`12 % 6 == 0` 在 6 处停止（`logs/r13/new_shape_check.log`：
   `cb0(budget only) = 7 -> cb(fixed) = 6 n_chunks = 2`，`tier2_viol=0`，PASS）。按实测记录。
9. **未做**：未跑 TileOPs report、未同步 `/workspace/operator-migration`、未 commit/push、
   未改 wrapper 的双 import 切换块、未读写 `.stage_state.json`。

### 15.5 知识条目重验（`PL-attention-bwd-engine-phase-split`）

- **硬约束落地**：r12 在 §14.9 提出「预算换算出的 `cb` 必须整除 `n_kv`，否则 pass-3 的
  `dk/dv` 静默错误」但**只留了建议**。本轮把它实现为 host 侧向下取整
  （`while n_kv % cb: cb -= 1`），并给出三类证据：
  ① 32 例 golden 分层表（fp16 16/16 `tier2_viol==0`；bf16 16/16 tier-3 envelope）；
  ② 16 组配置对（32 个 `dk/dv` 张量对）跨 `cb` **逐 bit 相等**（short `cb∈{4,8,1}`，long `cb∈{2,4,8}`）；
  ③ 交付件 L0 `ALL PASS (16 items)`，含 `S=300/1000` 尾块与 bf16 tier-3 用例。
  ⇒ 条目应更新为：**「`cb` 必须整除 `n_kv`」是由 pass-3 单写者语义决定的硬约束，不是调优旋钮**；
  预算只决定 `cb` 的上界，实际取值必须向下取 `n_kv` 的因子。
- **预算结论仍成立**：256 MiB 预算下 long 的 `ws = cb·per_kv = 2×96 MiB = 192 MiB`、
  short `4×48 MiB = 192 MiB`，均未超预算；768 MiB 臂 `ws = 768 MiB` 可运行但按用户裁决不采纳。
  即「256 MB 是保守下界而非上限」这半句不变，但**不能**因此忽略整除约束（r12 的错配正是
  「只看预算、不看整除」的后果）。
- **repro 指针**：`logs/r13/probe_cb_bitwise.py`（16 组配置对 bitwise）、
  `logs/r13/bf16_control.log`（bf16 判据对照）、`logs/r13/l0_deliverable.log`（L0 16 项）、
  `logs/r13/checks_arms.log` + `checks_arms.tsv`（32 例 golden）。
- **`CONST-vector-launch-overhead` 线索**：short 档 P/F 的 launch 数完全相同（2 chunks × 3 passes = 6），
  因此 15.3 的 short device 差异是该常数的「零贡献」刻度；long 上 P/F 也同为 48 launches，
  同理只核对测量重复性。

### 15.6 Skill Retrospective（Round 13，续跑轮）与 Value Point Proposals

**流程问题**：

1. 本轮是**中断续跑**：第一次 sweep 被截断时留下了「有 `### arm=` 头、无 verdict」的块
   （`b576 / llama-3.1-70b-long/bf16`），直接重跑 `check_arms.sh` 会 `: > "$LOG"` 清空已完成结果。
   本轮改用 `logs/r13/check_arms_resume.py`（按 `(arm, case)` 检测 verdict 才跳过）解决；
   代价是日志里留下 1 个重复 header（33 个头 / 32 个 verdict，已在 `r13_checks.txt` 注明）。
2. `bench.py --check` 的单一 `tier2_viol == 0` 判据对 **bf16** 不成立，导致 16 条「FAIL」实为判据
   与项目 L0 的 tier 分层口径不一致；本轮用对照实验（修复前文件、正确配置）证明其为既有性质。
   建议 `bench.py --check` 按 dtype 选择判据（bf16 用 tier-3 envelope），或至少打印 tier 归属
   而不是二元 PASS/FAIL。
3. r12 已把「`cb` 必须整除 `n_kv`」写成 P1 建议但未落地；本轮才实现。教训：**上一轮登记的 P1
   必须在下一轮开头作为必做项排进计划**，不能只留在日志里。

**Value Point Proposals（含证据三件套）**：

- **D-1（实测数据）** `cb | n_kv` 是硬约束：short（`n_kv = 8`）预算推出 `cb0 = 5` 必须降到 4；
  long `cb ∈ {2,4,8}` 与 short `cb ∈ {4,8,1}` 的 `dk/dv` **16 组配置对全部逐 bit 相等**，
  `dq` 仅有 2.98e-08…1.221e-04 的原子次序抖动。证据：现象=`logs/r13/cb_bitwise.log`；
  数字=`checks_arms.tsv`（fp16 16/16、bf16 16/16）+ `l0_deliverable.log`（16 items ALL PASS）；
  复现=`python3 logs/r13/probe_cb_bitwise.py`。建议并入 `PL-attention-bwd-engine-phase-split`。
- **P-1（模式方法）** 「预算 → 因子下取整」守卫模式：`cb = min(cb, n_kv)` 之后
  `while n_kv % cb: cb -= 1`。适用于任何「chunk 边界不得切断单写者」的分块 kernel（GEMM 分块、
  KV 分块、reduce 分段）。性质：workspace 只减不增（`ws = cb·per_kv ≤ 预算`），终止性由 `1 | n_kv`
  保证；代价是实际 `cb` 可能小于预算允许值（long 768 MiB 臂从 8→8 不变，512/576/672 → 4）。
  证据：`logs/r13/split_plan_fix.diff` + `r13_checks.txt` 的派生表。
- **R-1（流程规则）** 长 sweep 必须用「可续跑驱动」：按块检测 verdict 决定跳过，禁止用带
  `: > "$LOG"` 的批脚本续跑；中断造成的重复 header 须在验收 bundle 里计数说明，而不是手工清洗日志。
  证据：`logs/r13/check_arms_resume.py`、`logs/r13/checks_arms.log`（33 头 / 32 verdict）。

### Final Performance Test Data

交付件：`perf_opt/_gqa_bwd_kernel.py`（sha256 `8f865fe5…`，256 MiB，`cb` long 2 / short 4，
`while n_kv % cb` 守卫在位），`candidate_id = final-r13-fix256`，结构 = engine-phase split。
口径：`msprof op` `Task Duration(us)`，多 kernel 形态 = **一次 op 调用内全部 kernel 之和**；
每例 3 rep × 每 rep `calls=5` 个 op-call 窗口取中位（rep 间交错、臂顺序轮转）。
`baseline_us` = `perf_records.jsonl` 中 round-11 的 baseline 行（`candidate_id = r6-arm-r11`）。
本表是**全 `opt_log.md` 唯一精确标题为 `### Final Performance Test Data` 的表**
（§6 / §12 / §14.6 的同名表标题均已带注记，属历史记录）。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---|---|---|
| _gqa_bwd_main | llama-3.1-8b-short/fp16 | 2011.57 | 995.00 | final-r13-fix256 |
| _gqa_bwd_main | llama-3.1-8b-short/bf16 | 2012.32 | 1003.80 | final-r13-fix256 |
| _gqa_bwd_main | llama-3.1-8b-long/fp16 | 13954.16 | 7500.50 | final-r13-fix256 |
| _gqa_bwd_main | llama-3.1-8b-long/bf16 | 13995.79 | 7536.90 | final-r13-fix256 |
| _gqa_bwd_main | llama-3.1-70b-short/fp16 | 2017.08 | 993.10 | final-r13-fix256 |
| _gqa_bwd_main | llama-3.1-70b-short/bf16 | 2013.05 | 999.90 | final-r13-fix256 |
| _gqa_bwd_main | llama-3.1-70b-long/fp16 | 14018.46 | 7511.70 | final-r13-fix256 |
| _gqa_bwd_main | llama-3.1-70b-long/bf16 | 13981.99 | 7539.30 | final-r13-fix256 |

每例明细（chunks / 3 个 rep 的 device 中位 / 相对 baseline 比值）：
- `llama-3.1-8b-short/fp16`：chunks=2，reps 995.0 / 989.7 / 998.3，0.4946x
- `llama-3.1-8b-short/bf16`：chunks=2，reps 1007.7 / 996.5 / 1003.8，0.4988x
- `llama-3.1-8b-long/fp16`：chunks=16，reps 7500.4 / 7503.4 / 7500.5，0.5375x
- `llama-3.1-8b-long/bf16`：chunks=16，reps 7536.9 / 7537.1 / 7534.1，0.5385x
- `llama-3.1-70b-short/fp16`：chunks=2，reps 993.1 / 997.6 / 986.9，0.4923x
- `llama-3.1-70b-short/bf16`：chunks=2，reps 1004.6 / 994.2 / 999.9，0.4967x
- `llama-3.1-70b-long/fp16`：chunks=16，reps 7548.0 / 7511.7 / 7501.7，0.5358x
- `llama-3.1-70b-long/bf16`：chunks=16，reps 7539.3 / 7557.9 / 7535.5，0.5392x

geometric mean ratio vs baseline = **0.5163x (-48.4%)**; best 0.4923x, worst 0.5392x

注：本表 `final_us` 与 15.3 的 `F` device 列（`calls=3`）采自同一交付件、同一 device，
差异即 15.4 所述 0.1–0.8% 的 device 噪声；两者都不含 host 下发间隙（op wall 口径见 15.3）。
