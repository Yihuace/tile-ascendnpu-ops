# GLADecodeOp Stage 4 调优日志（opt_log）

- 算子：GLADecodeOp（single-token gated linear attention decode，Vector 单元 / Developer 模式）
- 基准 kernel：`_gla_decode_kernel.py`（k_tile 默认 16）；最终 kernel：`perf_opt/GLADecodeOp.py`（k_tile 默认 32 + persistent 分核，`TUNED_NCORE=40`）
- 平台：Ascend910B3，CANN 9.0.0，tilelang dev-root build（7a4f77e184）；物理 AIV 核数 = `NPUUtils.get().get_aicore_num()*2` = 40
- 性能口径：`msprof op Task Duration(us)`（唯一时延口径）+ `--aic-metrics=Default,Roofline` 的 GM Read+Write SOL（Ratio×100）
- 性能目标（user_requirement）：Task Duration + GM Read+Write SOL；按 AI 分类判据（访存>=70% / 算力>=60% / Mix>=40%），不伪达标

---

## Performance Test Data

benchmark 入口：`benchmarks/ops/bench_gla_decode.py::test_gla_decode_manifest_bench`，经 manifest `tileops/manifest/linear_attention.yaml` `GLADecodeOp.workloads` 展开。

### tune workload（5 条，全 bfloat16，DK=DV=128）

| kernel_id | workload_id | baseline_us (k16) | final_us (k32+persistent) | 提速 | GM SOL baseline | GM SOL final |
|---|---|---|---:|---:|---:|---:|
| gla_decode_kernel | serving-b1-h8-bfloat16 | 9.01 | 6.79 | 24.6% | 2.52% | 3.24% |
| gla_decode_kernel | serving-b1-h32-bfloat16 | 11.03 | 8.22 | 25.5% | 8.23% | 10.22% |
| gla_decode_kernel | serving-b1-h64-bfloat16 | 18.77 | 12.53 | 33.2% | 9.53% | 14.12% |
| gla_decode_kernel | serving-b8-h32-bfloat16 | 57.59 | 38.41 | 33.3% | 12.30% | 18.05% |
| gla_decode_kernel | serving-b8-h64-bfloat16 | 106.31 | 68.92 | 35.2% | 13.45% | 20.25% |

（final 数值取 `phase=final` round 4，在最终 `GLADecodeOp.py`（k_tile=32 + persistent 分核）上测得；round 3 的 k32-only final 记录已 supersede 为 `phase=merged`，其数值 6.82/8.63/14.71/41.31/76.44 保留在 `perf_records.jsonl`。）

### smoke / skip 排除

- `smoke-b2-h4-d64-float16` / `smoke-b2-h4-d64-bfloat16`：manifest idx 0 smoke 案例，仅精度回归，不采性能。最终 `precision_pass=true`（内嵌 `--level all` 9 案例全过，含这两条）。
- 无 `skipped` 项。

### 设计估算 vs 实测偏差（D-2 回填）

本算子无 DESIGN.md（direct-implementation 迁移），无 §1.6.0 设计期估算下界行可对照；跳过该偏差行。

---

## Iteration Log

### 度量管线

- `--aic-metrics=Default,Roofline` + `--kernel-name` 过滤（VP-2026-0081 规则）——Roofline 集产出 visualize_data.bin 预计算 Ratio；kernel-name 过滤避免把 ZerosLike 等辅助 op 当目标。
- **kernel-name 陷阱**：bf16/fp16 的 jitted kernel 名为 `_gla_decode_transit_main`（transit 变体）、fp32 为 `_gla_decode_main`，**不是** `main`。上一轮会话已把 `run_experiments.py` 的 `KERNEL_NAME` 修正为 `_gla_decode_transit_main`；本条在 Skill Retrospective 中登记。

### Round 1 — k_tile 扫描（候选：ktile8、ktile32；base：baseline k_tile=16）

现象：baseline advice 全为 `latency bound:pipeline caused`，GM SOL 仅 2.5~13.5%。k_tile 是唯一编译期参数，`_ceil_ktile_fit` 的 150KB 启发式把 k_tile 锁在 16，但实际 UB=192KB（实测 k_tile=32 可编译）。

候选 vs current best（Task Duration us；base=baseline k16）：

| workload | base(k16) | ktile8 | ktile32 | 结论 |
|---|---:|---:|---:|---|
| serving-b1-h8-bf16 | 9.01 | 12.94 | 6.77 | ktile32 胜 |
| serving-b1-h32-bf16 | 11.03 | 14.60 | 8.47 | ktile32 胜 |
| serving-b1-h64-bf16 | 18.77 | 26.72 | 14.44 | ktile32 胜 |
| serving-b8-h32-bf16 | 57.59 | 84.40 | 42.14 | ktile32 胜 |
| serving-b8-h64-bf16 | 106.31 | 156.32 | 77.06 | ktile32 胜 |

结论：k_tile=32 全 workload 胜（+23~27%）；k_tile=8 回退（+43~47%）。k_tile=32 把串行 K-tile 循环从 8 次降到 4 次，减少 loop 开销与每 tile 的 reduce/barrier。k_tile=64 不成立（fp32 缓冲 294KB+ 超 192KB UB）。

### Round 2 — 合并 winner 并 final 复测（winner_k32 = GLADecodeOp.py，k_tile 默认 32）

改动：`_ceil_ktile_fit` 的 UB 预算 150KB → 192KB（UB 实容量，constants.md），使 DK=DV=128 自动选 k_tile=32；暴露模块级常量 `TUNED_K_TILE=32` 供 wrapper 切换块引用；补齐内嵌分层测试（run_L0/L1/L2/boundary + main `--level`，gate 4 `S4-TEST-LEVELS` 要求）。语义不变（纯 tiling 变化 + 追加测试代码），L0/L1 9 案例全过。

candidate（final 复测）vs current best 对比表（Task Duration us；final=round 3）：

| workload | base(k16) | final(k32) | 提速 | GM SOL 变化 |
|---|---:|---:|---:|---|
| serving-b1-h8-bf16 | 9.01 | 6.82 | 24.3% | 2.52%→3.18% |
| serving-b1-h32-bf16 | 11.03 | 8.63 | 21.8% | 8.23%→10.15% |
| serving-b1-h64-bf16 | 18.77 | 14.71 | 21.6% | 9.53%→11.68% |
| serving-b8-h32-bf16 | 57.59 | 41.31 | 28.3% | 12.30%→16.77% |
| serving-b8-h64-bf16 | 106.31 | 76.44 | 28.1% | 13.45%→18.25% |

合并后全 workload 无回退（均提速），smoke 精度通过 → 采纳 winner_k32 为 current best。

### Round 4（第 3 轮优化）— persistent 分核：超发 shape 减 wave，欠订 shape 保持 flat

base = current best `winner_k32`（`perf_opt/GLADecodeOp.py`，k_tile=32），base profile = round 3 final（advice 全 `latency bound:pipeline caused`，GM SOL 3.18–18.25%）。

**现象（从 base profile + `perf_records.jsonl` 结构化表读出）**

| 观测 | 数值 | 推论 |
|---|---|---|
| 并行度 | 每 (b,h) 一个 block；block_dim = B×H = 8 / 32 / 64 / 256 / 512 | 40 个物理 AIV 下：8、32 欠订；64 = 1.6 wave；256 = 6.4 wave；512 = 12.8 wave |
| wave 量化 | 64 blocks→2 wave（第二 wave 仅 24 核，16 核空转）；256→7 wave（24 槽空）；512→13 wave（8 槽空） | 超发 shape 每个 wave 都要重走 block 进出与 UB 重新装配 |
| 核内结构 | 每头 4 个 K-tile 串行、每 tile 一条 load→vcast→vmul→vadd→reduce→store 依赖链 | 单头时延 ~5.3–6.8µs，且**几乎不随核数变化** ⇒ 每头内部无并行，跨头才可并行 |
| advice | 全 workload `latency bound:pipeline caused` | 待验证：是「搬运/计算未重叠」还是「wave 级调度开销」 |

**候选优化点（同一 base 派生，每分支只改一个主要点）**

| opt_id | 改动（单点） | 备注 |
|---|---|---|
| `persist40` | `T.Kernel(min(B*H,40))` + 核内 grid-stride `T.serial(ceil(BH/40))` + `if blk < B*H` | 超发档的「满核 persistent」 |
| `persist32` | 同上，`ncore=32`（B×H 的整除数，1/2/4/8/16 iter） | 用整除数消除尾 wave 空槽 |
| `persist20` | 同上，`ncore=20`（半核） | 低点：确认「核数不足」的代价，避免只测单点 |
| `pipe2` | `persist` + K-tile 循环 `T.Pipelined(num_stages=2)` | 第二优化点：验证 advice 是否指搬运/计算未重叠 |
| `dispatch` | 编译期分派：`B*H <= 40` 走**原 flat 体**（逐字节相同），否则走 persistent；2-iter 情形取整除数 | 合并形态（winner） |

**候选 vs current best 对比表（Task Duration 为 msprof `Task Duration(us)` 的 2 rep 中位数；block_dim / GM SOL / advice 来自同一批 raw profile；L0 全部 PASS）**

| workload | 并行度(B×H) | current best (k32best) µs | persist40 µs | persist32 µs | pipe2 µs | dispatch µs | block_dim（best→new） | GM SOL%（best→new） | advice | L0 |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|:---:|
| serving-b1-h8 | 8（欠订） | 6.87 | 6.88 | 6.81 | 6.82 | 6.86 | 8 → 8 | 3.18 → 3.08 | latency bound:pipeline caused | PASS |
| serving-b1-h32 | 32（欠订） | 8.35 | 8.45 | 8.63 | – | 8.68 | 32 → 32 | 10.15 → 10.44 | latency bound:pipeline caused | PASS |
| serving-b1-h64 | 64（1.6 wave） | 14.80 | 13.89 | **12.50** | 14.02 / 12.49 | **12.68** | 64 → 32 | 11.68 → 13.46 | latency bound:pipeline caused | PASS |
| serving-b8-h32 | 256（6.4 wave） | 41.73 | **38.54** | 39.69 | 38.60 | **38.84** | 256 → 40 | 16.77 → 17.79 | latency bound:pipeline caused | PASS |
| serving-b8-h64 | 512（12.8 wave） | 77.04 | **68.61** | 75.85 | 68.75 | **68.44** | 512 → 40 | 18.25 → 20.17 | latency bound:pipeline caused | PASS |

（`persist20` 在 b8-h64 单点测得 131.73µs（+71%），作为「核数不足」低点已否决，不再扫其余 shape。）

**分支裁决**

1. `persist20` ❌ 否决：20 核 → 26 iter，时延翻倍，验证了「persistent 必须先保满核」。
2. `persist32` ⚠️ 依赖 shape：b1-h64 最优（64=32×2 整除，12.50µs，−15.5%），但 256/512 上因 32/40 核容量损失被吃掉（75.85µs vs 68.61µs）。
3. `persist40` ✅：超发档全胜（b8-h64 −10.9%、b8-h32 −7.6%、b1-h64 −6.1%）。
4. `pipe2` ❌ 无收益（b8-h64 68.75 vs 68.61、b8-h32 38.60 vs 38.54、b1-h64 14.02 vs 13.89，全部落在噪声内）：`T.Pipelined(num_stages=2)` 在本 loop 上不产生额外重叠 ⇒ advice 的 "pipeline caused" **不是**「搬运/计算未重叠」，而是 wave 级调度开销 + 单头依赖链；该方向记为 `family_no_gain`（同一机制的两个 num_stages/核数组合均无收益）。
5. `dispatch` ✅ 采纳：编译期分派（欠订走原 flat 体 + 超发走 persistent + 2-iter 取整除数）。
   - b1-h64：`n_iter==2` 时取 `B*H` 在 `[ceil(BH/2), 40]` 内的最大整除核数 32（12.68µs，−14.3%）——比 `ncore=40`（13.89µs）好 9.5%，机制是尾 iter 只剩 24/40 核 + 首 iter 40 路并发 DMA 抬高单头时延。
   - 欠订 shape 的抗回退证据：`B*H <= 40` 时 `dispatch` 生成的 prim_func 源码与 current best **逐字节相同**（`transit`/`plain` 两路均已 diff 校验 = True），且 b1-h32 按交错 A/B 协议复测：`ab_test.py --pairs 3` → a=8.52µs / b=8.60µs，`b_rel_change=+0.94%`，`sign_test_p=0.25`，**verdict=tie**（未超 3% 噪声阈值），不触发「欠订回退 >3% 否决」。

**合并检查（本 kernel 全部 5 个 tune workload + smoke 精度）**

- 5 个 tune workload 全部 `L0=PASS`、全部不回退（3 个超发 shape −7.4~−14.3%，2 个欠订 shape −0.2%/+2.3% 落在 8.19–8.71µs 双态带内，A/B 判 tie）。
- smoke（`smoke-b2-h4-d64-float16/bfloat16`）走同一 `_run_cases` L0 路径通过；最终文件 `--level all` 9 案例全过（含 f32 untiled / small-k16 / rect / custom-scale 边界）。
- 合并后 `perf_opt/GLADecodeOp.py` = `GLADecodeOp_round3_dispatch.py`；`TUNED_K_TILE=32`（保留上轮结论）+ 新增 `TUNED_NCORE=40`，供 wrapper 切换块成对引用。

**Final 复测（同一最终候选 `winner_k32_persist`，round 4 `phase=final`）**

| workload | baseline k16 µs | round-3 final (k32, superseded) µs | round-4 final µs | vs k16 | vs k32 |
|---|---:|---:|---:|---:|---:|
| serving-b1-h8 | 9.01 | 6.82 | 6.79 | −24.6% | −0.4%（噪声） |
| serving-b1-h32 | 11.03 | 8.63 | 8.22 | −25.5% | −4.8% |
| serving-b1-h64 | 18.77 | 14.71 | 12.53 | −33.2% | −14.8% |
| serving-b8-h32 | 57.59 | 41.31 | 38.41 | −33.3% | −7.0% |
| serving-b8-h64 | 106.31 | 76.44 | 68.92 | −35.2% | −9.8% |

> 记录纪律说明：round 3 的 5 条 `phase=final` 记录（`winner_k32`）在采纳新最终候选后被**改写为 `phase=merged` + `superseded_by=winner_k32_persist`**（数值、时间戳、profile 路径全部保留，未删除任何测量），以满足 gate 4「每个 workload 恰有一条当前最终版本 final 记录」+「final 哈希与最终文件一致」；同 repo 先例：`isinf` round 4→5 续优化。改写前快照：`perf_records.jsonl.bak_r3`。

### 未探索 / 已评估但未执行的方向（诚实披露）

- **每核 2 头交错（interleave two heads per core）**：是剩余唯一可攻击单头依赖链的方向（每头 ~5.3–6.8µs 几乎与核数无关 ⇒ 单头内无并行）。但 1 头工作集 ~162KB（9 个 fp32 [k_tile,DV] + 2 个 dtype 中转），2 头需 ≤192KB ⇒ 只能 k_tile=16 + 2 头（147KB）。k_tile=16 的单头实测比 k_tile=32 慢 ~25%（每 tile 往返时延与 tile 大小近似无关，8 tile 串联），而 2 头交错至多把每 2 头摊到 1 条链 ≈ 12.9µs/2 头 ≈ 6.45µs/头 vs 当前 6.8µs/头 ⇒ 预估 <5%，落在噪声内，未执行。
- **DV 切分（按 DV 列分核提升欠订并行度）**：b1-h8 仅 8 个 (b,h)，DV 切 4 份可得 32 核。但每头 4 个 K-tile 的串行往返仍是关键路径（切 DV 只缩 compute，不缩 DMA 往返），且超发 shape 上会让 block 数翻倍回到长 wave；预估收益 <5% 且方向与 persistent 相反，未执行。
- **transit 路径消除（bf16/fp16 直接运算）**：bf16 无 Vector 算术，fp32 累加是 GPU 契约 + 精度要求，5 条 tune workload 全 bf16，不可消除。

---

## Final Summary

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| gla_decode_kernel | serving-b1-h8-bfloat16 | 9.01 | 6.79 | winner_k32_persist |
| gla_decode_kernel | serving-b1-h32-bfloat16 | 11.03 | 8.22 | winner_k32_persist |
| gla_decode_kernel | serving-b1-h64-bfloat16 | 18.77 | 12.53 | winner_k32_persist |
| gla_decode_kernel | serving-b8-h32-bfloat16 | 57.59 | 38.41 | winner_k32_persist |
| gla_decode_kernel | serving-b8-h64-bfloat16 | 106.31 | 68.92 | winner_k32_persist |

（final 时延与 perf_records.jsonl `phase=final` round 4 记录一致；全部来自同一 final 候选 `winner_k32_persist` = 最终 `perf_opt/GLADecodeOp.py`（k_tile=32 + persistent 分核，sha256 与记录 `artifact_sha256` 一致））

端到端门禁（采纳前）：
- `python3 -m tileops.reporting.cli run --op GLADecodeOp --prof-mode msprof` → run_id `20261018_215742_239583_GLADecodeOp`（最终字节）。
- 正确性 **11/11**、`status=passed`、0 failed；benchmark 时延 7.00 / 8.34 / 12.50 / 37.88 / 68.72 µs，Ratio（GM SOL）3.10–20.16%，与 final 记录一致（±3% 单次采样差）。
- smoke `smoke-b2-h4-d64` 4.06µs（仅精度，不采性能结论）。

### 性能目标达成情况（如实，不伪达标）

- **AI 分类**：AI ≈ 3.64 Ops/Byte（GM Read+Write 口径），`AI×GM_bw(1.75TB/s)=6.4TOps/s` 远低于 Vector 算力 → **访存型（memory-bound）**，判据阈值 **SOL ≥ 70%**。
- **GM Read+Write SOL 实测**：final 为 3.24% ~ 20.25%，**全部未达 70%**。msprof advice 全程 `latency bound:pipeline caused`（非 memory bound / compute bound）。
- **Task Duration**：相对 Stage-3 baseline（k16）**−24.6% ~ −35.2%**（round 1 参数级 k_tile 16→32 贡献 −21.6~28.3%，本轮 persistent 分核再贡献 −6.9~14.3%）；相对上一轮 k32 final **−7.0~14.8%**（3 个超发 shape），2 个欠订 shape 无回退（A/B tie）。
- **结论**：Task Duration 主目标持续改善；SOL 70% 目标**未达成**。分层天花板证据：(a) 参数级 k_tile 已到 UB 上限（32；64 需 294KB > 192KB）；(b) 分核级 wave 调度开销已在本轮吃掉（block_dim 512→40，−10.9%）；(c) 复核 advice 语义——`T.Pipelined(num_stages=2)` 实测 **零收益**（68.75 vs 68.61µs），证明 "pipeline caused" 不是「搬运/计算未重叠」，而是单头依赖链（～5.3–6.8µs/头，与核数无关）× 头数/核数；(d) 剩余唯一结构方向（每核 2 头交错）受 UB 192KB 限制须退到 k_tile=16，实测估算 <5%。未见实测或文档支撑「>2x 结构性加速」，故**不产出 [DESIGN_LIMIT]**（触发条件②不满足）。

### 中止原因

`success`（Task Duration 主目标以「参数级 + 分核级」最优收束）——SOL 70% 副目标未达成已如实披露，不作为中止条件；未耗尽 max_rounds（4/10）与 max_experiments（约 14/30），未 plateau（本轮仍有 3 个 shape 取得 >5% 真实收益），剩余方向（2 头交错 / DV 切分）已用实测+容量论证记录为 <5% 或方向相反。

---

## Skill Retrospective

按 vp_type 标注，供任务终态 tilelang-skill-evolver 蒸馏。

- **D（实测数据）— `_ceil_ktile_fit` UB 预算过保守**：该 Vector 算子的 K-tile 启发式用 150KB 上限，实际 UB=192KB（constants.md「UB=192KB/AIV」），导致 DK=DV=128 被锁在 k_tile=16、错失 k_tile=32 的 ~22-28% 收益。证据：本任务 ktile32 全 workload 实测（perf_records.jsonl round1 candidate ktile32）+ 编译成功（162KB 手工 / ~182KB 经 1.10-1.12× BishengIR 膨胀 < 192KB）。toolchain_stamp：tilelang dev-root 7a4f77e184 + CANN 9.0.0 + Ascend910B3 / 2026-10-17。建议：向量算子 K-tile 容量启发式应以 192KB 为上限（并计入 ~1.12× 膨胀 + 小 buffer），勿用经验 150KB。
- **P（模式方法）— msprof kernel-name 与 transit 变体名**：`@tilelang.jit(out_idx=...)` 返回的 prim_func 名（`_gla_decode_transit_main` / `_gla_decode_main`）才是 msprof `--kernel-name` 应过滤的名字，不是 `main`。证据：本任务中断会话日志（`--kernel-name=main` 时 "Kernel skipped: not selected via --kernel-name" + "No profiling data dumped"）。与 VP-2026-0081（kernel-name 过滤缺一不可）同源，建议并入该条或新立 transit 变体名子条目。toolchain_stamp：同上。
- **C（案例索引）— GLADecodeOp 单 token decode 为 latency-bound**：逐头 state 递归 + decode 并行度=B×H 有限，GM SOL 上限 ~20%（k_tile=32 + persistent），达 70% 访存 SOL 需结构性改变（非参数级/分核级可解）。证据：perf_records.jsonl 全部 advice="latency bound:pipeline caused"。
- **D（实测数据）— persistent 分核的适用边界与核数选择（本轮新增）**：同一 kernel 的 grid 从 `B*H` 收到 `min(B*H,40)` 后，收益**只出现在超发 shape**：b8-h64（512 blocks / 12.8 wave）512→40 得 −10.9%，b8-h32（256/6.4 wave）−7.6%，b1-h64（64/1.6 wave）−14.3%（配 ncore=32）；欠订 shape（8、32 blocks）无收益（A/B tie）。核数曲线不是单调：`ncore=20` 在 b8-h64 上 **+71%**（131.73µs），`ncore=32` 在 512 上 **零收益**（75.85 vs 68.61，因为 32/40 核容量损失刚好抵消 wave 收益）。证据：perf_records.jsonl round 4 candidate persist20/persist32/persist40 + `profiles/`。toolchain_stamp：tilelang dev-root 7a4f77e184 + CANN 9.0.0 + Ascend910B3 / 2026-10-18。
- **D（实测数据，陷阱）— persistent 包裹 fp32 直拷路径会静默算错（本轮新增）**：把 `T.Kernel(B*H)` 的 flat 体改成 `grid-grid + if blk < B*H` 的 persistent 体后，**bf16/fp16（copy→vcast 中转）完全正确**，但 **fp32 直拷路径**（`T.copy(state[...], fp32 st_ub)`、`T.copy(ns_ub, new_state[...])`）出现静默错误：B8-H32-D128 fp32 实测 max|Δo|=5.12、max|Δnew_state|=43.6（同 shape 的 flat 版 3.6e-07/0）。修法：persistent 只在 `use_transit` 路径启用（编译期判定），fp32 保持 flat。建议加入 traps：**在 npuir Developer 模式下，把「GM↔UB 直拷的纯 fp32 体」放进「外层 serial + 运行时 if 守卫」的 persistent 结构前，必须逐 dtype 做 golden 对比**（L0 小 shape 覆盖不到该分支——本项目 L0 的 fp32 只有 B*H=2）。toolchain_stamp：tilelang dev-root 7a4f77e184 + CANN 9.0.0 + Ascend910B3 / 2026-10-18。
- **P（模式方法）— 「2-iter 整除核数」规则（本轮新增）**：当 `ceil(B*H/40) == 2`（40 < B*H ≤ 80）时，取 `B*H` 在 `[ceil(BH/2), 40]` 内的最大整除核数，让两个 iter 都满核：b1-h64 由 `40 核×(2,1)` 的 13.89µs 降到 `32 核×2` 的 12.50µs（−10%）。机制：尾 iter 只剩 24/40 核干活，且首 iter 40 路并发 per-head DMA 抬高单头时延；iter ≥3 时该效应被摊薄（256/512 上 40 核反而最优）。证据：round 4 persist32 vs persist40 同批交错测量。
- **P（模式方法，负向）— `T.Pipelined(num_stages=2)` 在「单头串行依赖链 + 多 wave」结构上零收益（本轮新增）**：`latency bound:pipeline caused` 的 advice 不可直接推断为「搬运/计算未重叠」。实测 b8-h64 68.75 vs 68.61µs、b8-h32 38.60 vs 38.54µs、b1-h64 14.02 vs 13.89µs（全在噪声内），而减 wave（persistent）反而 −10%。⇒ 该 advice 下应先量化「wave 数 × 单 wave 时延」，再看 copy/compute 重叠；对 per-task 工作集小、任务数有限的算子，block 调度开销优先。证据：`GLADecodeOp_round3_pipe.py` + perf_records.jsonl round 4 pipe40/pipe32。
- **R（流程规则）— 续优化轮次的 final 记录 supersede 纪律（本轮新增）**：同一算子二次进 Stage 4 时，旧 `phase=final` 行与新 final 行会同时存在，gate 4 `S4-PERF-RECORDS-RECON` 要求「每 workload 恰一条 final」。做法：把被 supersede 的旧 final 行**改写为 `phase=merged` 并追加 `superseded_by`**（保留 duration/timestamp/raw path，不改写任何测量值），并保留 `.bak` 快照；先例 `isinf` round 4→5。建议写入 optimize skill 的 Phase 3 收束步骤，避免每个续优化任务各自发明。
- 回写 pattern-library：本任务结论由 evolver 蒸馏合入，conductor 不直接改 skill 文件。
