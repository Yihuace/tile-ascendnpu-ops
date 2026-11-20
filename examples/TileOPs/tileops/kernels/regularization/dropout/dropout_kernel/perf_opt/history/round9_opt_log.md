# Stage 4 调优日志 — DropoutOp / `_make_dropout_kernel`（R8：Expert 双链交错假说真机验证）

- 任务: `TileOPs-_make_dropout_kernel-20261102T032027Z`（scenario=optimize, mode=full）
- 本轮主攻（用户假说）: `block_size` 8192 → 2×4096，同一 `T.Kernel` 内两条互相独立的依赖链显式交错，Expert 模式，用延迟互补把一次性 ramp/drain 开销 2.83µs 压到 ~1.5µs
- 基准 kernel（只读）: `examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/_make_dropout_kernel.py`
- 调优产物: `examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py`（sha256 `c3f71a95854ca1711d7ab8f34908f5e4a8a54d19dc406b0566ff2a9731962572`，R6 版本，本轮经实测确认无可替代）
- 硬件/工具链: Ascend910B3（20 AI Core × 2 = 40 AIV，UB 192 KB）/ CANN 9.0.0 / tilelang `0.1.2+026a47e0036160d07832cad03b8be2fcbecff9b6` / torch_npu 2.7.1 / 2026-11-02
- 性能口径（唯一）: `msprof op --kernel-name=main --aic-metrics=Default,Roofline --launch-count=20 --warm-up=5 --dump=off`；Task Duration 取 `OpBasicInfo.csv` 全 20 个 launch 的均值；Ratio 取 `visualize_data.bin` 中 `title=GM/L2` 且 `bw_name` 含 `GM Read + Write` 的条目（与官方 benchmark 同源）
- 上一轮工件（records / inventory / opt_log）已原样归档到 `perf_opt/history/round8_perf_records.jsonl`、`perf_opt/history/round8_workload_inventory.json`、`perf_opt/history/round8_opt_log.md`；本轮 `perf_opt/perf_records.jsonl` 只含 round 8 的 12 行

## 0. Workload Inventory

| workload_id | shape | dtype | N | kind | 本轮处理 |
|---|---|---|---|---|---|
| `smoke-dropout-1m-float32` / `-float16` / `-bfloat16` | [1024,1024] | fp32/fp16/bf16 | 1,048,576 | smoke | 仅精度回归，不采集性能 |
| `tokens-1k-hidden-4k-fp16-float16` | [1024,4096] | fp16 | 4,194,304 | tune | baseline + 6 候选 + final |
| `tokens-1k-hidden-10k-bf16-bfloat16` | [1024,10240] | bf16 | 10,485,760 | tune | baseline + final |
| `tokens-1k-hidden-4k-fp32-float32` | [1024,4096] | fp32 | 4,194,304 | tune | baseline + final |

单一 `kernel_id`（6 个案例调用同一工厂 `_make_dropout_kernel(N, dtype, p, seed)`），`target_kernel_name=main`。
fp16 是主筛选 workload（40 AIV × 13 迭代，tile 数最少、tile 边际成本最敏感）；bf16/fp32 只对 winner 复测。完整分类见 `perf_opt/workload_inventory.json`。

## 1. Baseline（round 0，本轮重测）

| kernel_id | workload_id | Task Duration(us) | Ratio(%) | vec(us) | scalar(us) | vec_ratio | L0 |
|---|---|---:|---:|---:|---:|---:|---|
| `examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main` | tokens-1k-hidden-4k-fp16-float16 | 21.949 | 54.973 | 19.054 | 1.353 | 0.9543 | pass |
| 同上 | tokens-1k-hidden-10k-bf16-bfloat16 | 58.101 | 63.060 | 54.468 | 2.731 | 0.9737 | pass |
| 同上 | tokens-1k-hidden-4k-fp32-float32 | 22.729 | 56.806 | 19.603 | 1.254 | 0.9538 | pass |

与 R7 归档 baseline（21.888 / 58.115 / 22.502）偏差 ≤1.0%，`aiv_vec_time` 三档逐位相同（19.0539 / 54.4683 / 19.6028）——本轮测量口径与上一轮完全对齐。

**前置校正复核（用本轮实测确认用户的 ② 条前提）**：

- **是 busy 不是 stall**：`aiv_vec_ratio` = 0.9543 / 0.9737 / 0.9538，向量单元墙钟占用 95–97%，稳态可掩盖空间只有 3–5%（≈0.6–1.1µs）。
- **2.83µs 主要是一次性开销**：fp16 `duration − vec_time = 21.949 − 19.054 = 2.895µs`。本轮新增的 b 组（bs 4096 单链，迭代数 26 而非 13）为 3.430µs，线性拟合得**一次性 ≈2.36µs + 每迭代 ≈0.041µs**——用户的「一次性 ≈2.8µs」判断成立（其中 scalar 只占 1.35µs），且该分量与 tile 数、链数基本无关。
- 上一轮 `T.Pipelined(num_stages=2)` 的 +1.0% 回退与本轮 b/c/d/e 的「vec_ratio 反而下降」同向：DMA 早已被 auto multi-buffer 重叠，剩余 idle 不在 DMA 侧。

## 2. Iteration Log — Round 8

### 2.1 候选生成方式（单一变量）

用 `perf_opt/probes/gen_r8.py` 从 R6 的 11-op 链**逐语句重排**生成，数学/掩码契约逐字不变（同一 `vclamp` 符号阈值链、同一 op 序列），每个分支只改一个结构变量：

- b: `block_size` 8192→4096（单链，Developer）——验证「减半 tile 本身」的代价；
- b′: 同 b 但 Expert 模式（`T.Kernel(20)` + `(cid, subid)`，`cid*2+subid` = 40 条持久流）——隔离**模式**成本；
- c: b′ 的 2 条独立链在同一 `T.Kernel` 内逐语句交错（假说本体，Expert）；
- e: b 的 2 条独立链逐语句交错（Developer）——同一交错在最佳 codegen 模式下的对照；
- f: `block_size` 2048、4 链交错（Developer）——交错阶梯延伸。

交错形态（c/e/f 相同）：a 的 op1 → b 的 op1 → a 的 op2 → b 的 op2 … 直到掩码；数据段「两条链的 load 全部先发」→ 逐 op 交错 → 两条 store。每链独占 `x/y/h/t/mf/md` 缓冲（**引擎独占 staging**），因此不存在 MTE2 写 / MTE3 读同一 UB buffer 的 WAR，无需 `T.pipe_barrier`（见 `CG-2026-0009` 的 workaround：流水深度 >1 时引擎独占 buffer 是唯一可靠形态；Expert 编译流默认 `--disable-hivm-auto-inject-sync=false`，自动注入同步仍然生效）。奇 tile 数的尾组用「按 `N−block_size` 夹取后重复处理一个等价 tile」代替运行时守卫，保证热循环无分支（elementwise 确定性 ⇒ 重叠写入字节相同）。

### 2.2 候选 vs current best（B2 结构化对比表，全部 fp16 4k / msprof op / 40 AIV）

| 候选（单点改动） | Task Duration(us) | Δ vs current best | Ratio(%) | vec_time(us) | vec_ratio | scalar_time(us) | 非 vec 开销(us) | L0 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **a = current best**（Developer, bs 8192, 单链） | **21.949** | — | 54.97 | 19.054 | 0.9543 | 1.353 | 2.895 | pass | 参照 |
| b：bs 4096 单链（Developer） | 24.317 | +10.8% | 49.49 | 20.887 | 0.9414 | 2.469 | 3.430 | pass | 弃选（halve-tile 使 vec +1.83、scalar +1.12） |
| b′：bs 4096 单链（Expert） | 26.631 | +21.3% | 44.85 | 20.699 | 0.7961 | 2.474 | 5.932 | pass | 弃选（Expert 固定开销 +2.31µs） |
| c：bs 4096 **双链交错**（Expert，假说本体） | 25.879 | +17.9% | 45.90 | 19.741 | 0.7593 | 2.268 | 6.138 | pass | 弃选（vec −4.6% 被 Expert 开销吞掉） |
| e：bs 4096 **双链交错**（Developer） | 23.023 | +4.9% | 51.92 | 19.964 | 0.9474 | 2.296 | 3.059 | pass | 弃选（交错在同 tile 下有效，但仍慢于 a） |
| f：bs 2048 **四链交错**（Developer） | 24.677 | +12.4% | 48.61 | 20.989 | 0.9315 | 3.989 | 3.688 | pass | 弃选（链数↑ 使 scalar 翻倍、vec 反升） |

**同 tile 单变量对照（交错机制是否有效）**：

| 对照 | Duration(us) | vec_time(us) | 差值 |
|---|---:|---:|---|
| b → e（Developer，4096，单链→双链） | 24.317 → 23.023 | 20.887 → 19.964 | **−5.3% / vec −4.4%** |
| b′ → c（Expert，4096，单链→双链） | 26.631 → 25.879 | 20.699 → 19.741 | **−2.8% / vec −4.6%** |
| b → b′（同 tiling 单链，Developer→Expert） | 24.317 → 26.631 | 20.887 → 20.699 | **+9.5%（Expert 固定开销 +2.31µs）** |

### 2.3 假说检验结论

1. **交错机制本身成立**：同 tile 尺寸下双链把 V 单元忙时降低 4.4–4.6%（20.887→19.964 / 20.699→19.741），两种模式独立复现。说明链间交错确实填补了依赖延迟——但可填的只有 3–5% 的稳态 idle，与 `vec_ratio 0.94–0.97` 的前提一致。
2. **但假说的收益路径不成立（Task Duration 三组全部上升）**：8192→4096 的 halve-tile 代价是 **+2.37µs**（vec +1.83 来自每 tile 固定的向量指令启动开销，scalar +1.12 来自 tile 数翻倍），而交错只回收 1.29µs（e vs b）。净账 = −2.37 + 1.29 ≈ **−1.08µs**，方向与假说相反。
3. **Expert 模式是纯成本，不是收益来源**：同 tiling/同链数下 Expert 比 Developer 慢 2.31–2.86µs（b′ vs b、c vs e），来源是 20-block × 2-AIV 的 block 级调度与 cube0 空转（profile 里出现 60 行 = 20 blocks × cube0/vector0/vector1）。**交错不需要 Expert**：Developer 版本 e 的 vec_time 与 Expert 版 c 的 vec_time 几乎相同（19.964 vs 19.741），而 wall 时间少 2.86µs。
4. **「2.83µs 一次性开销」不可由任何 tiling/交错形态回收**：a 2.895 / b 3.430 / b′ 5.932 / c 6.138 / e 3.059 / f 3.688µs。最好形态（e、b）都 ≥2.9µs，而**目标是 1.5µs**——本轮 6 个形态没有一条接近。该开销对「每核跑几个 tile / 几条链」不敏感 ⇒ 属 kernel launch/ramp 的固定成本，不是可掩盖的 per-tile 气泡。
5. **四链阶梯更差**：f 的 scalar 时间翻倍到 3.99µs（每迭代 4 次索引/循环簿记），vec_time 反升到 20.99µs，说明 2×4096 已是交错形态的局部最优，继续细分只增加固定开销。

⇒ **用户假说不成立**：`block_size=8192` 单链仍是 40 AIV 上最快的形态；`4096×2` 交错在本 kernel 上净亏 4.9%（e）到 17.9%（c）。这也解释了为什么上一轮 `T.Pipelined(num_stages=2)` 无收益——交错/流水能动的只有 3–5% 的 idle，动它反而要付 tile 细分的固定成本。

### 2.4 ② 把 clamp 从 int32 迁到 fp32 的加权账（只算，不实施）

口径：`Ratio = Perf / Computility`，`Perf = counted_fops / Task Duration`。

- **counted_fops**：由 `aiv_vec_fops` 逐 op 求和，与 dtype 标签无关（R7 §1 已实测）。迁移只改 `vclamp` 的操作数域（int32 → fp32），**op 数不变**（仍是 1 条 `vclamp`，且 `vcast(h,mf)` 由「先 clamp 后 cast」变成「先 cast 后 clamp」，链长相同）⇒ **Δcounted_fops = 0**。
  数值锚点（fp16 baseline）：`Perf = 2.141612 TOPS`，`counted_fops = 2.141612e12 × 21.949e-6 = 47.00 M`（4.194 M 元素 ≈ 11.2 op/elem，与 11-op 链一致）。
- **Computility**：按 dtype 峰值 ppc 加权（FP16/INT16 128、**FP32/INT32/S32 64**、MISC 32；同一工具链版本，R7 §1 推导并用 fp32 3.620282 / fp16 3.895756 / bf16 3.740526 三档标定）。迁移把 1 个 op 从 INT32(64) 移到 FP32(64)，**两个类别同 ppc** ⇒ **ΔComputility = 0**（与其余 op 的组成无关，只需比较被移动 op 的类别）。
- **净效应（诚实路径）**：`ΔRatio = ΔPerf/Computility − Perf·ΔComputility/Computility² = 0`；而 duration 不可能下降（op 相同、依赖链相同，且 fp32 clamp 前面多了一次 cast 的域转换 ⇒ 可能更慢）。**净 Ratio ≤ 0**。
- **唯一可能为正的路径是记账假象**：若 fp32 域 `vclamp` 被降级到 MISC(32) 指令类，则 Computility 下降 1 个 op 的份额 `32 / Σ(op_i·ppc_i)`：fp16 链 ≈ −3.6%、fp32 链 ≈ −5.0% ⇒ Ratio 相应 +3.7% / +5.3%，而 Task Duration 不变或变慢。这与 R7 的 `i16` 掩码 cast（时延 +1.1% 却 Ratio +1.56 点）同一类假象。**该分支是否真的降级需一次独占探针验证（未实证假设），但即使成立也不是真实收益。**
- **结论：不实施**。迁移既不减少 op、也不改变峰值类别、更不缩短时延；正收益只可能来自 Ratio 分母的记账漂移，按口径红线不可采纳。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp16-float16 | 21.949 | 21.905 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-10k-bf16-bfloat16 | 58.101 | 58.395 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp32-float32 | 22.729 | 22.514 | final |

- 逐项数值来自 `perf_opt/perf_records.jsonl` 的 `phase=baseline` / `phase=final` 6 行（同一 artifact，sha256 `c3f71a95`；raw profile 在 `perf_opt/profiles/r8/base_W1..W3` 与 `perf_opt/profiles/r8/final_W1..W3`）。
- 逐 workload 变化：fp16 +0.20%、bf16 −0.51%、fp32 +0.95%，全部落在 3% 噪声阈值内 ⇒ 本轮无真实时延变化，final 与 baseline 是同一版本。
- 最终三档 Ratio：fp16 54.87% / bf16 63.04% / fp32 57.06%，均值 **58.32%**（baseline 均值 58.28%）。**距 60% 目标 −1.68 点，未达成。**
- smoke（1M × fp32/fp16/bf16）只做精度：由 `--level all` 的 L0-1..L0-3 覆盖，全部 pass；未采集性能。

## 3. 精度回归（最终文件）

`python perf_opt/_make_dropout_kernel.py --level all` → **L0 22/22、L1 12/12、L2 14/14、Boundary 9/9 全 PASS**（`perf_opt/logs/r8/L0_L1_final_all.log`，末行 `All check passed!`）。
4 个新形态（b、b′、c、e、f）各自 `--level L0` 全过（22/22），日志为 `perf_opt/logs/r8/L0_<candidate>.log`；其中 c/e/f 的 L0 同时覆盖了奇 tile 数尾组（N=1,000,000 → 245 tile）、`p=0/1` 边界与确定性 replay。

## 4. Skill Retrospective

| vp_type | 结论 | 证据三件套 |
|---|---|---|
| **D** | **Expert 模式在本工具链（CANN 9.0.0 / tilelang 0.1.2+026a47e）对纯 Vector 算子有 +2.3~2.9µs 固定成本**，与 tiling/链数无关；profile 出现 20 blocks × cube0/vector0/vector1 共 60 行（cube0 全 NA），即每 block 起一个 AI core 并复制 Vector 程序到 2 个 AIV。同 tiling 同链数 Developer→Expert：24.317→26.631（+9.5%）、23.023→25.879（+12.4%）。**结论：交错类改造应优先在 Developer 模式做，Expert 只在需要显式跨引擎/跨子块握手时才值得。** | evidence: §2.2 b/b′/c/e 四行 + §2.2 第三张对照表；repro: `perf_opt/_r8_b4096_exp.py` / `_r8_b4096_dev.py`（仅 ASCEND_MODE 与 grid 形态不同）；toolchain: tilelang 0.1.2+026a47e / CANN 9.0.0 / 910B3 / 2026-11-02 |
| **D** | **链间交错确实降低 V 忙时（−4.4~−4.6%）**，但收益上限被 `aiv_vec_ratio 0.94–0.97` 锁死，且 halve-tile 的固定开销（+2.37µs）大于交错收益（1.29µs）。量化：`duration − vec` 的固定分量 ≈2.36µs + 0.041µs/迭代（13→26 迭代拟合），与 tile/链数无关 ⇒ launch/ramp 地板，不可由调度形态回收。 | evidence: §2.2 两张表 + §2.3 第 4 点；repro: `perf_opt/probes/gen_r8.py`（同一 11-op 链按 chains/mode/bs 生成）；toolchain: 同上 |
| **P**（反向） | **「细 tile + 多链交错」在 V 忙时 ≥94% 的逐元素算子上是净亏方向**：tile 减半的 per-tile 固定成本（每个向量指令的启动/发射开销随指令条数增长：11 op × 51.2 tile vs 11 op × 25.6 tile）大于交错回收的 idle。四链更差（scalar 翻倍）。判据：先看 `aiv_vec_ratio`——≥0.94 时任何「交错/流水」收益上限 <6%，优先考虑减少 op 数而不是加并行度。 | evidence: b 20.887 / e 19.964 / f 20.989 的 vec_time 阶梯；repro: `_r8_{b4096_dev,e4096x2_dev,f2048x4_dev}.py`；toolchain: 同上 |
| **R** | 流程：**`perf_records.jsonl` 的归档必须用「过滤后重写」而非 cp 复制**——本轮先把旧文件 cp 到 `history/round8_*` 再直接 append，导致新文件里残留 R7 的 3 条 final 行，会触发 gate `S4-PERF-RECORDS-RECON`（每个 tune workload 必须恰有 1 条 final）。修法：归档后按 `round` 过滤重写，再追加本轮行。 | evidence: 本轮 records 由 18 行过滤为 12 行（R7 的 6 行保留在 `perf_opt/history/round8_perf_records.jsonl`）；toolchain: 无 |
| **R** | 流程：**profile 解析器要按 `sub_block_id` 取有效行**——Expert/Roofline 的 `PipeUtilization.csv` 同时含 `cube0`（全 NA）与 `vector0/vector1` 行，直接取 `rows[0]` 会得到 vec=0 的假指标（本轮 `b4096_exp_W1` 的首行记录就是该形态，已用 `b4096_exp_rerun_W1` 复测修正，旧行按 append-only 纪律保留）。 | evidence: `perf_opt/perf_records.jsonl` 的 b4096_exp_W1 与 b4096_exp_rerun_W1 两行；修法见 `perf_opt/logs/r8/run_batch.py` 的 `next(... != "NA")`；toolchain: 同上 |

流程问题：本轮是单一假说的证伪，6 个候选构成同一族（tile 细分 + 链交错），`family_no_gain` 范围为 fp16 4k 上全部 6 行；未触发合并轮（无 winner），current best 保持 R6 版本。参照结构 diff（T-1）：同族逐元素参考实现（`erf`/`isfinite` 的 perf_opt）与本 kernel 同为「grid-stride tile + 单缓冲 + 无交错」，未见可移植差异；本轮新增的「交错」变量已被实测排除，不构成新的结构候选。

## 5. 残留与结论

- **目标未达成**：三档均值 58.32%，距 60% 差 1.68 点。本轮把上一轮「可动空间只剩 ~2.8µs」进一步量化并封闭：该 2.8µs 的一次性分量 ≈2.36µs、per-迭代分量 ≈0.04µs，任何 tile/链/模式形态都不能把它压到 1.5µs（最好的 e 为 3.059µs）。
- **`block_size` 三元组不变**：float32 / float16 / bfloat16 = **8192 / 8192 / 8192**（wrapper 的 `_DEFAULT_BLOCK_SIZE` 与 perf_opt 一致，无需翻转常量）。
- **`perf_feedback.md` 的既有 `[DESIGN_LIMIT]` 结论不变**，本轮不新增、不改写该文件（本轮结论属参数/调度层的 no_gain，不构成新的设计层归因或推翻）。
- 未采纳：Expert 模式（+2.3µs）、2×4096 交错（+4.9%）、4×2048 交错（+12.4%）、clamp 域迁移（0 或记账假象）。
- **记录归一说明（如实披露）**：三条 `phase=final` 记录的 `candidate_id` 由测量时的 profile 目录名 `final_W1/final_W2/final_W3` 归一为 `final`（gate `S4-PERF-RECORDS-RECON` 要求同一 kernel 的全部 final 记录来自同一候选）。归一**只改候选标签**，`duration_us` / `aiv_vec_time_us` / `ratio_pct` / `artifact_sha256` 全部保持采集时的原值（21.905 / 58.395 / 22.514µs）。
