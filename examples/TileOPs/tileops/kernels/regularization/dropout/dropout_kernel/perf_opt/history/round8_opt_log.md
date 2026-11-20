# Stage 4 调优日志 — DropoutOp / `_make_dropout_kernel`（R7 收口轮：无真实时延增益，保持 R6 版本）

- 任务: `TileOPs-_make_dropout_kernel-20261102T005412Z`（scenario=optimize, mode=full）
- 基准 kernel（只读）: `examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/_make_dropout_kernel.py`
- 调优产物: `perf_opt/_make_dropout_kernel.py`（sha256 `c3f71a95854ca1711d7ab8f34908f5e4a8a54d19dc406b0566ff2a9731962572`，R6 版本，本轮未改）
- 硬件/工具链: Ascend910B3（20 AI Core × 2 = 40 Vector core，UB 192 KB）/ CANN 9.0.0 /
  `tilelang 0.1.2+d58388995fc943c5d79035e2a312116ea6d086cd` / torch_npu 2.7.1 / 2026-11-02
- 性能口径（唯一）: `msprof op --kernel-name=main --aic-metrics=Default,Roofline --launch-count=20 --warm-up=5`，
  Task Duration 取 `OpBasicInfo.csv::Task Duration(us)` 全 20 个 launch 的均值；Ratio 取
  `visualize_data.bin` 中 `title=GM/L2` 且 `bw_name` 含 `GM Read + Write` 的条目（＝官方 benchmark.md 口径，
  本机复现偏差 ≤0.1 点：fp16 54.52% vs 官方 54.47%）。
- 上一轮产物（9 条记录 / R6）原样归档在 `perf_opt/history/round7_*`。

## 0. Workload Inventory

| workload_id | shape | dtype | N | kind | 本轮处理 |
|---|---|---|---|---|---|
| `smoke-dropout-1m-float32` / `-float16` / `-bfloat16` | [1024,1024] | fp32/fp16/bf16 | 1,048,576 | smoke | 仅精度回归（`--level all` 的 L0-1 覆盖），不采集性能 |
| `tokens-1k-hidden-4k-fp16-float16` | [1024,4096] | fp16 | 4,194,304 | **tune** | baseline + 6 分支 + final |
| `tokens-1k-hidden-10k-bf16-bfloat16` | [1024,10240] | bf16 | 10,485,760 | **tune** | baseline + final |
| `tokens-1k-hidden-4k-fp32-float32` | [1024,4096] | fp32 | 4,194,304 | **tune** | baseline + final |

单一 `kernel_id`（6 个案例调用同一工厂 `_make_dropout_kernel(N,dtype,p,seed)`），`target_kernel_name=main`。
完整分类见 `workload_inventory.json`；逐测量记录见 `perf_records.jsonl`（6 行：baseline 3 / final 3）。

## 1. Baseline（round 0，R6 版 11-op @ bs 8192 重测）

| kernel_id | workload_id | TaskDur(us) | Ratio(%) | vec(us) | scalar(us) | L0 |
|---|---|---:|---:|---:|---:|---|
| `…perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main` | fp16 4k | 21.888 | 54.52 | 19.054 | 1.342 | pass |
| 〃 | bf16 10k | 58.115 | 63.04 | 54.468 | 2.743 | pass |
| 〃 | fp32 4k | 22.502 | 56.96 | 19.603 | 1.253 | pass |

三档与官方 `20261101_235107_195598_DropoutOp`（21.96 / 58.10 / 22.46 µs，54.47 / 63.02 / 57.31%）
偏差 ≤1.9%（fp32 本地 56.96 vs 官方 57.31），均值为 **58.17%**（官方 58.27%）。

**本轮先做的口径拆解（决定后续方向）**：`Ratio = Perf / Computility`，
`Perf = counted_fops / duration`，`counted_fops` 由 `ArithmeticUtilization.csv` 的 `aiv_vec_fops` 求和
（与 dtype 无关），`Computility` 按各 dtype 的 peak ppc 加权（FP16/INT16 128、FP32/INT32/S32 64、MISC 32）。
⇒ **Ratio 只奖励「单位指令的执行效率」，对「削减工作量」中性**：减 op 会同时削减 fops 与 duration
（R6 已实测 13→11 op：时延 −21%，Ratio −0.57 点）。因此本轮唯一合法杠杆是
**op 数与 op 类型不变的前提下压低真实 Task Duration**（消 stall / 提升并行度 / 提升单指令效率）。

## 2. Iteration Log — Round 7

### 2.0 PRNG 逐 op 计时（Δt 探针：每次删一个 op，append=false，不写 records）

fp16 4k（N=4.19M，bs 8192，11-op base = 21.888 µs / vec 19.054 µs）：

| 探针（删除该 op） | TaskDur(us) | Δt(us) | vec(us) | Δvec | 备注 |
|---|---:|---:|---:|---:|---|
| base（11 op） | 21.888 | — | 19.054 | — | 参照 |
| `T.fill(h, off+seed)` | 21.109 | −0.78 | 18.038 | −1.02 | 索引段 |
| `T.vadd(lane,h,h)` | 20.177 | −1.71 | 17.168 | −1.89 | 索引段（最贵单 op） |
| `T.vmul(h,_A_I32,h)` | 20.792 | −1.10 | 17.949 | −1.11 | hash |
| `T.vshl(h,13,t)` | 21.219 | −0.67 | 18.216 | −0.84 | hash（DCE 后 t 死，Δ 偏小） |
| `T.vxor(h,t,h)` | 13.861 | −8.03 | 11.032 | −8.02 | **含 DCE**：删 xor 使 shl 死、`c*A`·B 常量折叠 ⇒ 非单 op 边际 |
| `T.vmul(h,_B_I32,h)` | 20.946 | −0.94 | 17.949 | −1.11 | hash |
| `T.vclamp(h,h,0,1)` | 19.982 | −1.91 | 16.974 | −2.08 | 掩码段（次贵） |

结论：**PRNG 链没有单个「异常贵」的 op**（除被 DCE 污染的 xor 行外），7 个 int32 op 的边际成本
0.67–1.91 µs、合计约 8 µs，与 vec 时间线性可加；最贵的是 `vadd`(1.71) 与 `vclamp`(1.91)。
任一删除都会同时削减 counted_fops（Ratio 实测全部下降：52.4–54.6%，均低于 base 54.52），
且真实时延降幅 ≤ 8%，**按口径红线全部不可采纳**（不构成"真实时延下降且 Ratio 不靠减 op"的合法收益）。

### 2.1 候选 vs current best（B2 结构化对比表，全部 fp16 4k / msprof op）

| candidate（单点改动） | TaskDur(us) | Δ vs best | Ratio(%) | vec(us) | scalar(us) | L0 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| current best = R6（11-op，bs 8192） | 21.888 | — | 54.52 | 19.054 | 1.342 | pass | 参照 |
| `r7_static`：`N%bs==0` 时静态 tile extent（去 `T.min`） | 22.026 | +0.6% | 54.67 | 19.164 | 1.231 | 未跑（时延无增益，弃选） | 弃选（scalar 降 0.11 但 duration 反升） |
| `r7_pipe`：`T.Pipelined(num_stages=2)` | 22.102 | +1.0% | 53.98 | 19.054 | 1.360 | 未跑（同上） | 弃选（编译器已等效，与 PL-1.21 一致） |
| `r7_i16`：掩码 cast `i32→i16→f16`（op 数不变，治 2 个 misc cast） | 22.136 | +1.1% | **56.09** | 19.119 | 1.343 | 未跑（同上） | 弃选（Ratio 升 1.56 点纯属 mix 记账，真实时延未降 ⇒ 口径假象） |
| `r7_bs8080`：波数对齐 bs=8080（fp32/fp16） | 22.659 | +3.5% | 53.16 | 19.417 | 1.293 | 未跑（同上） | 弃选（尾块 784 + 非 2 幂 tile 抵消 1.5% 波数收益） |
| `r7_pad` / `r7_pad128`：int32 UB buffer 间插 32 B/128 B pad（治 `bankgroup_cflt 0.208`） | 21.895 / 21.917 | +0.0% | 54.67 / 54.47 | 19.054 / 19.054 | 1.342 / 1.357 | 未跑（同上） | 弃选（vec 时间逐位不变 ⇒ pad 被 DCE / 未改变 bank 映射） |

- 单点性：每个分支只改一个主要优化点，全部从同一 current best 派生；无合入轮（无 winner）。
- 精度：三个 workload 的契约等价性由同一 11-op 链保证（分支只改 tiling/调度/cast 形态，不改掩码语义）；
  最终文件 `--level all` 全过（§3）。
- **记录说明（如实披露）**：6 个候选分支的 msprof 原始 profile 保留在
  `profiles/r7/<candidate>/OPPROF_*`（逐档 TaskDuration/Ratio 见上表），但其 `perf_records.jsonl` 行在写入时
  分支 L0 证据尚未产生（l0_pass=false 会触发 gate 的 `S4-PERF-RECORDS-SCHEMA`）。因 6 个分支全部因
  「真实时延未下降」被第一时间弃选，按 4.2 的"candidate 行非契约必需（baseline/final 必需）"处理：
  JSONL 只保留 baseline/final 6 行，候选测量在本表 + raw profile 留痕，**未改写任何已测时延**。

### 2.2 本轮否决的方向与实测依据

| 方向 | 实测结论 | 依据 |
|---|---|---|
| 波数对齐（bs 8080 使 520 tile = 13 整波） | 无效（+3.5%） | 上表；尾块 784 元素 + 非 2 幂 tile 的 DMA/对齐代价 > 1.54% 尾波收益 |
| 静态 extent / 去 `T.min` | 无效 | scalar 仅 1.34→1.23 µs，duration 反升；`duration − vec ≈ 2.8 µs` 不是 scalar 主导 |
| `T.Pipelined(num_stages=2)` | 无效 | 与 base 逐位相同（vec 19.054），DMA 已被自动多缓冲隐藏（`mte2+vec = 24.1 µs > duration 21.9`） |
| UB bank-group 反冲突 padding | 无效 | vec 时间逐位不变，说明 pad 未进入分配或 bank 映射不随 32/128 B 偏移改变 |
| 减 PRNG op（更便宜 hash / 少 cast） | 语义与指标双重不可行 | §2.0：单 op 边际 0.67–1.91 µs，而 counted_fops 同步下降 ⇒ Ratio 必降 |

**实测边界（本轮最重要的量化结论）**：base 的 `duration(21.888) − aiv_vec_time(19.054) = 2.83 µs`（12.9%），
其中 scalar 仅 1.34 µs；`vec_time` 本身对 11 个 op 线性可加，`bankgroup_cflt 0.208` 未能通过 buffer 偏移改变。
⇒ 在「11 op 不变」的约束下，可动空间只剩 ~2.8 µs 的 launch/同步/stall，而本轮 6 个结构性候选均无一能取出其中
任何一个百分点。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp16-float16 | 21.888 | 21.895 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-10k-bf16-bfloat16 | 58.115 | 58.189 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp32-float32 | 22.502 | 22.521 | final |

- 逐项数值来自 `perf_records.jsonl` 的 `phase=baseline` / `phase=final` 6 行（同一 artifact
  `_make_dropout_kernel.py`，sha256 `c3f71a95…`；raw profile
  `profiles/r7/{base_W1,base_W2,base_W3,final_W1,final_W2,final_W3}/OPPROF_*`）。
- 最终三档 Ratio（本机 Roofline 口径，官方同源）：fp16 **54.52%** / bf16 **63.02%** / fp32 **56.66%**，
  均值 **58.07%**（官方报告 58.27%）。**距 60% 目标 −1.93 点，未达成。**
- smoke（1M × fp32/fp16/bf16）只做精度：`--level all` 的 L0-1 覆盖，全部 pass；未采集性能。

## 3. 精度回归（最终文件）

`python perf_opt/_make_dropout_kernel.py --level all` → **L0 22/22、L1 12/12、L2 14/14、Boundary 9/9
全 PASS**（`logs/r7/L0_L1_final_all.log`，末行 `All check passed!`；drop-rate 判据用常量非零输入 `torch.ones`）。
最终文件与 R6 逐字节相同（sha256 `c3f71a95…`），故 R6 的精度结论在本轮重新实测确认。

## 4. Skill Retrospective

| vp_type | 结论 | 证据三件套 |
|---|---|---|
| **D** | **`Ratio = counted_fops/duration ÷ mix 加权 Computility` 的量化结构**：`counted_fops` 来自 `aiv_vec_fops`（与 dtype 无关），`Computility` 按 dtype peak ppc 加权（FP16/INT16 128、FP32/INT32 64、MISC 32）。因此**减 op 必降 Ratio**（R6 13→11：−21% 时延 / −0.57 点），**改 op 的 dtype 会单独改变分母**（本轮 `i16`：时延 +1.1% 却 +1.56 点，纯记账） | evidence: 本文 §1 口径拆解 + §2.1 `r7_i16` 行；repro: `perf_opt/logs/r7/run_batch.py`（Roofline 解析）+ `probes/gen_r7.py`；toolchain: tilelang 0.1.2+d5838899 / CANN 9.0.0 / 910B3 / 2026-11-02 |
| **D** | **11-op dropout 链的逐 op 边际成本表**：fill 0.78 / vadd 1.71 / mulA 1.10 / shl 0.67 / mulB 0.94 / clamp 1.91 µs（fp16 4.19M），无单点异常项；`duration − vec = 2.83 µs` 中含 scalar 1.34 µs | evidence: §2.0 表；repro: `probes/p_*.py`（每次删 1 op）；toolchain: 同上 |
| **P**（反向） | **「同 op 数、压真实 duration」的四条路在 dropout 上均已证伪**：静态 extent、`T.Pipelined(num_stages=2)`、bs 波数对齐、UB buffer 32/128 B 偏移反 bank-group 冲突。判据：`mte2+vec > duration`（24.1 > 21.9）⇒ DMA 已隐藏，`aiv_vec_time` 逐位不变即该方向无效 | evidence: §2.1 全部 6 行（vec 时间 19.054 逐位相同或上升）；repro: `_r7_{static,pipe,bs8080,pad}.py`；toolchain: 同上 |
| **R** | 流程：**候选行与 L0 证据的写入顺序必须对齐**——应先产出 L0 日志再 append perf_records 行，否则 gate 的 `S4-PERF-RECORDS-SCHEMA`（每行 `l0_pass is True`）会拒绝；本轮因此把被弃选分支的测量保留在 opt_log + raw profile 而非 JSONL | evidence: `perf_opt/perf_records.jsonl` 6 行 + §2.1 记录说明；toolchain: 无 |

流程问题：本轮 6 个候选全部为「无增益」，且都属同一族（降低 stall/提升并行度），构成 `family_no_gain`；
未触发合并轮（无 winner）。参照结构 diff（T-1）：同族逐元素参考实现（`erf`/`isfinite` perf_opt）与本 kernel
结构一致（grid-stride tile + 单缓冲 + 无流水），未见可移植的结构差异，故不进入新一轮结构变异。

## 5. 残留与结论

- **目标未达成**：三档均值 58.07%（官方口径 58.27%），距 60% 差 1.9 点。真实时延在本轮无任何可行下降空间：
  11 个向量 op 全部必要（掩码契约要求 `(seed, idx)` 纯函数的固定势子集），PRNG 段无更便宜且统计合格的替代
  （R6 已证 3-op hash 自相关不合格），DMA 已重叠，波数损失仅 1.5%，且任何减 op 都会同比例削减
  `counted_fops` 使 Ratio 下降。
- **`perf_feedback.md` 的 `[DESIGN_LIMIT]` 结论继续成立**，本轮按其 append-only 纪律追加「R7 附录」
  （新增：口径量化结构、逐 op 成本表、四条结构性方向证伪）。
- 未采纳/未量化项：`i16`（口径假象）、bs 8080（非 2 幂 tile）、UB pad（未生效）、`T.Pipelined`（已等效）。
- smoke 三个 1M 案例按契约不采集性能；精度由 `--level all` 覆盖。
