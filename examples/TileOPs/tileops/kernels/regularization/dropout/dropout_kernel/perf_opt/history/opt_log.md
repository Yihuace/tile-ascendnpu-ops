# Stage 4 调优日志 — DropoutOp / `_make_dropout_kernel`

- 任务: `dropout-_make_dropout_kernel-20261016T193233Z`（scenario=optimize, mode=full）
- 基准 kernel（只读）: `examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/_make_dropout_kernel.py`
- 调优产物: `perf_opt/_make_dropout_kernel.py`
- 硬件/工具链: Ascend910B3（实查 20 AI core × 2 = 40 Vector core，UB 192 KB，AI Core 1800 MHz）/ CANN 9.0.0 / `tilelang 0.1.2+90fb7ae37f0be5e908d6317eb2b1e09aeb1ddc4b` / torch_npu 2.7.1 / 2026-10-17
- 性能口径（唯一）: `msprof op` → `OpBasicInfo.csv::Task Duration(us)`，`--kernel-name=main`，`--warm-up=5 --launch-count=20`（final 为 30），串行采集
- 性能目标: `best_effort`（用户未指定类型/数值/baseline；关注 shape 未指定）

---

## 0. Workload Inventory（来源与分类）

来源: `examples/TileOPs/benchmarks/ops/bench_dropout.py::test_dropout_bench` →
`workloads_to_params("DropoutOp", include_extra=True)`（manifest
`examples/TileOPs/tileops/manifest/regularization.yaml`）。实测展开得到 **6 个参数化案例**
（`pytest.param(*params, id=f"{label}-{dtype_str}")`），无 skip：

| workload_id | manifest label | shape | dtype | p | kind | 计入性能门禁 |
|---|---|---|---|---|---|---|
| `smoke-dropout-1m-float32` | smoke-dropout-1m | [1024,1024] | float32 | 0.5 | smoke | 否（仅精度回归） |
| `smoke-dropout-1m-float16` | smoke-dropout-1m | [1024,1024] | float16 | 0.5 | smoke | 否（仅精度回归） |
| `smoke-dropout-1m-bfloat16` | smoke-dropout-1m | [1024,1024] | bfloat16 | 0.5 | smoke | 否（仅精度回归） |
| `tokens-1k-hidden-4k-fp16-float16` | tokens-1k-hidden-4k-fp16 | [1024,4096] | float16 | 0.5 | **tune** | 是（N=4,194,304） |
| `tokens-1k-hidden-10k-bf16-bfloat16` | tokens-1k-hidden-10k-bf16 | [1024,10240] | bfloat16 | 0.5 | **tune** | 是（N=10,485,760） |
| `tokens-1k-hidden-4k-fp32-float32` | tokens-1k-hidden-4k-fp32 | [1024,4096] | float32 | 0.5 | **tune** | 是（N=4,194,304） |

- 分类依据: `smoke` = manifest label 中独立的 `smoke` token（NPU manifest workload 契约：首个 workload 为 smoke）；其余为 `tune`。未按 shape 大小或参数化顺序推断。
- 全部 6 个案例调用**同一个** Stage 3 kernel 工厂 `_make_dropout_kernel(N, dtype, p, seed)`
  → 单一 `kernel_id`，`target_kernel_name = main`（唯一目标 kernel）。
- skip 案例: 无。
- 排序（理由）: `W2 [1024,10240] bf16`（实测最大，16/40 B/elem 流量 + 最长向量链，结构收益面最大）
  → `W3 [1024,4096] fp32`（权重块最大，bs 受 UB 最紧，参数杠杆空间最大）
  → `W1 [1024,4096] fp16`（中等规模、掩码域最便宜）。三者全部处理，无跳过。

`workload_inventory.json` 记录完整分类；`perf_records.jsonl` 逐测量追加（见 §5 说明）。

---

## 1. Performance Test Data（baseline，round 0）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir | command |
|---|---|---|---|---:|---|---|---|---|
| `...::perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main` | tokens-1k-hidden-4k-fp16-float16 | baseline | main | main | 65.106 | valid | `profiles/baseline/W1/baseline_bs2048/OPPROF_20261017035036_HCZFMYAYSDOGKDVE` | `msprof op --kernel-name=main --launch-count=20 --warm-up=5 --aic-metrics=Default python dropout_entry.py --kernel ../_make_dropout_kernel.py --N 4194304 --dtype float16 --bs 2048` |
| 〃 | tokens-1k-hidden-10k-bf16-bfloat16 | baseline | main | main | 158.356 | valid | `profiles/baseline/W2/baseline_bs2048/OPPROF_*` | `... --N 10485760 --dtype bfloat16 --bs 2048` |
| 〃 | tokens-1k-hidden-4k-fp32-float32 | baseline | main | main | 77.110 | valid | `profiles/baseline/W3/baseline_bs1024/OPPROF_*` | `... --N 4194304 --dtype float32 --bs 1024` |

baseline 观测（msprof `PipeUtilization` / `Memory`，AIV 均值）：

| workload | bs | TaskDur(us) | aiv_vec_ratio | aiv_mte2_time(us) | aiv_mte3_time(us) | aiv_scalar_ratio | 有效 R+W | L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W1 fp16 (16.8 MB) | 2048 | 65.106 | **0.9372** | 14.925 | 7.538 | 0.1518 | 0.258 TB/s | pass |
| W2 bf16 (41.9 MB) | 2048 | 158.356 | **0.9425** | 37.066 | 17.611 | 0.1281 | 0.265 TB/s | — |
| W3 fp32 (33.6 MB) | 1024 | 77.110 | **0.9068** | 31.396 | 15.158 | 0.2212 | 0.436 TB/s | — |

**现象诊断**：三个 workload 全部 `aiv_vec_ratio ≈ 0.91–0.94`（Vector 管道饱和），
`aiv_mte2_ratio ≤ 0.42`、`aiv_mte3_ratio ≤ 0.20`（DMA 远未饱和），
`aiv_scalar_vector_stall_time` 占 TaskDur 的 88%（W1：56.9/65.0 µs）。
⇒ **纯 Vector 发射/吞吐受限**，不是 DMA 受限、也不是标量受限。
有效 R+W 仅 0.26–0.44 TB/s，距同机逐元素实用天花板 2.49–2.67 TB/s（`CONST-elu-910B3-bw-floor`，verified）有 **5.7–9.6x** 差距。
⇒ 主杠杆 = **向量 op 数**（发射项），次杠杆 = `block_size`（每 tile 摊薄）。

### 1.1 设计估算 vs 实测（D-2 回填，DESIGN.md §1.6.0 偏差行）

| workload | DESIGN §1.6.0 估算下界 | 设计估算值 | 实测 baseline | 偏差 | 失准项 |
|---|---|---:|---:|---:|---|
| W1 [1024,4096] fp16 | max(流量 16µs, 发射 48µs) | ≈173 µs | **65.11 µs** | 设计**高估 2.66x** | 发射项（op 数 × 9.15µs/16m 线性外推） |
| W2 [1024,10240] bf16 | max(流量 40µs, 发射 120µs) | ≈293 µs | **158.36 µs** | 设计**高估 1.85x** | 同上 |
| W3 [1024,4096] fp32 | max(流量 32µs, 发射 44µs) | ≈180 µs | **77.11 µs** | 设计**高估 2.33x** | 同上 |

失准根因（实测证据）：① `CONST-vector-op-cost-910_9362` 的 9.15 µs/16m-op 与 `CONST-ascend910-9362-profile`
的 1.06 TB/s 是 **910_9362 绑定**，本设备为 **910B3**（同为 40 Vector core 但带宽/发射口径不同）；
② 「op 数 × 单 op 边际代价」的发射项对**链内重叠**与 `block_size` 摊薄都不敏感——实测
bs2048→8192 使 W1 的 TaskDur 降 56%，而 op 数未变。设计期 roofline 的发射项口径在 910B3 上系统性高估（与
`CONST-vector-launch-overhead` 的 2026-10-15 重验注记同向）。

### 1.2 smoke 排除说明

3 个 smoke 案例（[1024,1024] × fp32/fp16/bf16）**不做性能采集、不选 winner、不参加性能门禁**，
仅做精度回归；最终精度结果见 §4.3。

---

## 2. Iteration Log

分支文件命名（`candidate_id` → 文件，均位于 `perf_opt/`；`perf_records.jsonl::artifact_path` 记录同一映射）：

| candidate_id | 文件 | 说明 |
|---|---|---|
| `baseline` | `../_make_dropout_kernel.py` | Stage 3 基准（只读） |
| `u5_diet` / `u6_vclamp` | `_u5_diet.py` / `_u6_vclamp.py` | R2 UB diet（vabs+vclamp / vclamp） |
| `hashC` | `_v_hashC.py` | R3 5-op counter hash |
| `f16native` / `seedfold` / `nat_seedfold` | `_w_f16native.py` / `_w_seedfold.py` / `_w_native_seedfold.py` | R4 存储域计算 / seed 折叠 |
| `x_final` | `_x_final.py` | R5 合并版（= 最终 `_make_dropout_kernel.py` 的 kernel 体） |
| `final` | `_make_dropout_kernel.py` | 最终 `perf_opt` 产物（8192 默认 tile） |

R2 的**静态整块拷贝**分支（`c1_sbulk` = `_make_dropout_kernel_round2_staticbulk.py`、`u7_sb_diet` = `_u7_sb_diet.py`）
在本轮 L0 复核中被判 **invalid（不通过）**，其 6 行已按契约从 `perf_records.jsonl` 移除（见 §5），文件移入 `probes/`。

被否决/阻塞的探针与失效分支文件（未产生性能记录，证据留在本节）在 `perf_opt/probes/`：
`_u1_scalarmask.py`、`_u2_nodebase.py`、`_u3_scalarseed.py`、`_u4_u123.py`、`_p1_scalarmask.py`、
`_p2_scalarhash.py`、`_v_idx1.py`、`_v_hashC_idx1.py`、`_probe_ops.py`、
`_make_dropout_kernel_round2_ubdiet.py`、`_make_dropout_kernel_round2_sb_ubdiet.py`、`gen_candidates_r2.py`、
`_make_dropout_kernel_round2_staticbulk.py`、`_u7_sb_diet.py`。

### Round 1 — 参数（`block_size`）扫描

现象（来自 round 0 profile）：Vector 饱和 + 每 tile 摊薄不足 ⇒ 先做 `block_size` 曲线（UB 二分定 cap）。

分支（全部为同一 baseline 文件 + 不同 `block_size`，无代码改动）：

| candidate vs current best | workload | bs | TaskDur(us) | vec_ratio | mte2_time(us) | mte3_time(us) | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| baseline（current best） | W1 fp16 | 2048 | 65.106 | 0.9372 | 14.925 | 7.538 | pass | — |
| baseline bs4096 | W1 fp16 | 4096 | **57.428** | 0.9603 | 8.651 | 3.834 | — | **winner**（−11.8%） |
| baseline bs8192 | W1 fp16 | 8192 | — | — | — | — | — | invalid: `ub overflow, requires 3342336 bits` |
| baseline（current best） | W2 bf16 | 2048 | 158.356 | 0.9425 | 37.066 | 17.611 | — | — |
| baseline bs4096 | W2 bf16 | 4096 | **138.529** | 0.9649 | 20.708 | 9.678 | — | **winner**（−12.5%） |
| baseline（current best） | W3 fp32 | 1024 | 77.110 | 0.9068 | 31.396 | 15.158 | — | — |
| baseline bs2048 | W3 fp32 | 2048 | 61.986 | 0.9388 | 16.252 | 8.998 | — | 有效（−19.6%） |
| baseline bs4096 | W3 fp32 | 4096 | **54.482** | 0.9464 | 9.461 | 5.400 | — | **winner**（−29.4%） |
| baseline bs8192 | W3 fp32 | 8192 | — | — | — | — | — | invalid: `ub overflow` |

UB cap 二分（fp16）：bs4096 ok / bs6144 `requires 1671168 bits`（> 1572864）⇒ baseline cap ≈ 4096–6143。
current best 更新为 **baseline @ bs4096**。smoke 精度回归：随 R2 一并执行，无回退。

### Round 2 — 结构候选（UB diet / 静态整块拷贝）

现象：`aiv_mte2_ratio` 在 bs4096 降到 0.14–0.18（DMA 不再是问题），vec_ratio 升到 0.96；
`aiv_scalar_ratio` 降到 0.086；⇒ 仍是 Vector 发射受限，但 **UB 预算卡住 bs**（每元素解析 UB
= x_ub+y_ub+xf+5×int32+mf ≈ 40–44 B/elem；auto-multi-buffer 版本化后实际需求更高
（BiShengIR 实测 bs8192 `requires 3342336 bits` ≈ 51 B/elem，bs6144 溢出）。
本轮从同一个 current best（baseline @ bs4096）派生 4 个分支，每支只改一个主要优化点：

| candidate vs current best | workload | bs | TaskDur(us) | vec_ratio | mte2_time(us) | mte3_time(us) | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| current best = baseline | W1 / W2 / W3 | 4096 | 57.428 / 138.529 / 54.482 | 0.9603 / 0.9649 / 0.9464 | 8.65 / 20.71 / 9.46 | 3.83 / 9.68 / 5.40 | pass | — |
| **C2 `u5_diet`**（去 base/m31/zerob + `vabs` 替代 `vand` + `vclamp` 替代 `vmax+vmin`） | W1 fp16 | 6144 | 57.201 | 0.9686 | 6.254 | 2.834 | pass(bs4096) | **winner**（UB cap 抬到 6144，−0.4% vs bs4096 同级） |
| 〃 | W2 bf16 | 6144 | 136.120 | 0.9736 | 15.761 | 7.111 | — | **winner**（−1.7%） |
| 〃 | W3 fp32 | 6144 | 54.452 | 0.9675 | 7.436 | 3.774 | — | 与 `u6_vclamp` 一起裁决 |
| `u6_vclamp`（UB diet + 仅 vclamp 去 1 op） | W1 fp16 | 6144 | 57.719 | 0.9184 | 6.254 | 2.834 | — | 不如 u5（−1.0%） |
| 〃 | W2 bf16 | 6144 | 137.754 | 0.9239 | — | — | — | 不如 u5 |
| 〃 | W3 fp32 | 6144 | **52.262** | 0.9675 | — | — | — | W3 局部最优，但与 u5 差 4.0%（见下） |
| `c1_sbulk`（PL-1.34 静态整块 + 编译期尾块）**— L0 invalid** | W1 fp16 | 4096 | 57.165 | 0.9601 | 7.956 | 3.871 | **FAIL**（详见下） | 不可采纳 |
| 〃 | W2 bf16 | 4096 | 138.551 | 0.9652 | 20.716 | 9.681 | **FAIL** | 不可采纳 |
| 〃 | W3 fp32 | 4096 | 54.397 | 0.9597 | 9.363 | 4.533 | **FAIL** | 不可采纳 |
| `u7_sb_diet`（静态块 + UB diet 组合）**— L0 invalid** | W1 / W2 / W3 | 4096 | 58.678 / 142.180 / 56.103 | 0.9589 / 0.9642 / 0.9584 | — | — | **FAIL** | 不可采纳 |

结论与裁决：

- **`c1_sbulk` / `u7_sb_diet` 判 invalid（L0 不通过，不可采纳）**：文件级 L0 复核（`logs/l0/L0__make_dropout_kernel_round2_staticbulk.log`、
  `logs/l0/L0__u7_sb_diet.log`）报 `error: Undefined variable: tail`——**编译期 `if non_divisible:` 分支内定义的
  `tail` 在另一个分支块不可见**（与 `TRAP-tvm-parser-rules` 的 if 作用域同族）。该结构只在
  `N % block_size == 0` 的 workload 上能编译，非整除 shape（L0 覆盖）直接失败 ⇒ **不能作为交付候选**。
  上表列出的数值（对整除 shape 有效）**仅作信息记录**，不构成性能结论；按 skill 契约（blocked / invalid
  分支不写入 `perf_records.jsonl`），这 6 行已从记录中移除（见 §5）。因此「PL-1.34 是否在本算子无效」
  这个问题**未获得干净实测**（无有效对照），§6.2 的对应提案已按此事实降级表述。
- 机制备注（未被本次有效实验证实，标注为**未实证假设**）：本算子此时 `aiv_mte2_ratio ≈ 0.14–0.18`，
  DMA 已被 Vector 完全掩盖，而 PL-1.34 的收益前提是 **DMA 受限**；静态块同时抬高 UB 需求
  （bs8192 `requires 2359296 bits` vs baseline 2228224）。若后续要验证该模式，须先把 `tail` 提到
  分支外（`tail = T.min(block_size, N - off)` 在 `if` 之前定义）再走 L0 + msprof。
- **`u6_vclamp` 的 W3 52.262 vs `u5_diet` 54.452**：差异 4.0%（> 噪声 3%）。但 `u6_vclamp` 在
  W1/W2 上分别比 `u5_diet` 差 0.9%/1.2%，且 `u5_diet` 的 mask 段少 1 op（vclamp 合并
  vmax+vmin）同时去掉了 `m31` 缓冲（UB 更省）。两者并存会引入按 shape 的条件路径，收益
  （W3 −4%）不足以抵消复杂度与 W1/W2 回退 ⇒ **本轮取 `u5_diet` 作为 current best**，
  W3 的差异留待 R4/R5 的更大结构杠杆后再复核（R5 实测 W3 28.56µs，两种 mask 形态的差异被淹没）。
- blocked（UB diet 的进一步削减）：**scalar 操作数形态不可用**（工具链限制，非参数问题）：
  - `T.vand(h, T.cast(_M31, "int32"), t)` → 前端 verify 失败：`'hivm.hir.vand' op failed to verify that operand at index 1 is vector-only`（**`tir.Cast` 表达式不可作标量操作数**，与 `TRAP-expert-v-operands` 同族）；
  - `T.vand(h, _M31, t)` + `T.vxor(h, seed, h)` + `T.vmax(t, 0, t)`（纯字面量形态，`probes/_u1_scalarmask.py`）→ 运行期 `ERR99999 UNKNOWN applicaiton exception`；
  - `T.vxor(h, seed, h)` 单独（`probes/_u3_scalarseed.py`）→ 同上异常；
  - `T.vadd(lane, off, h)`（单 op 索引，去掉 `T.fill`，`probes/_v_idx1.py`）→ 前端段错误（rc=139，无诊断）；
  - 说明：基线本身已使用字面量形态 `T.vshl(h, 13, t)` / `T.vmul(h, _A_I32, h)` / `T.vmin(t, 1, t)`
    ⇒ **字面量支持逐 op 不同（须逐 op 读取）**，不能按族推断；
  - 探针：`probes/_u1_scalarmask.py`、`probes/_u3_scalarseed.py`、`probes/_u4_u123.py`、
    `probes/_v_idx1.py`、`probes/_p1_scalarmask.py`、`probes/_p2_scalarhash.py`；可用的替代形态
    （`probes/_u2_nodebase.py`：`T.fill(h, off)` + `T.vadd(lane, h, h)` 原地）已并入 `u5_diet`。
  这属 `TRAP-expert-v-operands` 同族的「标量操作数只接受字面量、`tir.Cast`/表达式触发拒绝或崩溃」，
  记录为本工具链版本的实测结论。

current best 更新为 **`u5_diet` @ bs6144**（W1 57.20 / W2 136.12 / W3 54.45 µs，L0 pass）。
该 kernel 三个 tune workload 的合并复测（本轮全部重测，即上表 `u5_diet` 行）无回退；smoke 精度见 §4.3。

### Round 3 — 算法族内结构（counter hash 降复杂度）

现象：bs 杠杆已到 UB cap（6144 → 7168 反而回退 57.20→57.99，非单调），
vec_ratio 0.97 ⇒ 剩余空间只能来自**减少向量 op 数**。
当前 fp16 链 = 20 个向量 op：索引 2（fill+add）+ hash 9（xor seed + 4×(shl/xor) + 2 mul）
+ 掩码 4 + mask-cast 1 + 数据 4。
`DESIGN.md §1.6.1` 断言 PRNG「~9 op **无更优可行替代**」——本任务在同一算法族
（counter-based 无状态 hash，右移无关、int32 回绕）内验证更短链。

先做**主机侧 int32 模型统计检验**（`_hash_analysis.py`，kernels 为纯整数函数，numpy int32 模型与设备逐位等价）：

| chain | op 数（索引后） | rate p=0.1/0.3/0.5/0.7（4 Mi，3σ 判据） | block-χ²(dof 4095, 8095/块) | 结论 |
|---|---|---:|---|---|
| A（Stage 3）| 9 | 全 ok | 3974 / 4080 | 参照 |
| **C（采纳）** `c=(idx+seed); c*A; c^=(c<<13); c*B` | **5** | 全 ok | 3812 / 3946 | 与 A 同档，**采纳** |
| B（7 op）| 7 | 全 ok | 4270 / 4143 | 无额外收益 |

（`_hash_analysis.py` 同时输出 4-seed × 多 p 的 block-χ²、bit30 频率、seed 分离度；
chain C 的 rate 偏差 < 3σ、lag-2048 一致度 0.5004。）

分支与实测（`hashC` = `u5_diet` + chain C，单点改动）：

| candidate vs current best | workload | bs | TaskDur(us) | vec_ratio | mte2_time(us) | mte3_time(us) | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| current best = `u5_diet` | W1 / W2 / W3 | 6144 | 57.201 / 136.120 / 54.452 | 0.9686 / 0.9736 / 0.9675 | 6.25 / 15.76 / 7.44 | 2.83 / 7.11 / 3.77 | pass | — |
| **`hashC`** | W1 fp16 | 4096 | 40.602 | 0.9548 | 8.509 | 3.917 | — | −29.0% |
| **`hashC`** | W1 fp16 | 6144 | **39.845** | 0.9649 | 6.958 | 2.947 | pass | **winner**（−30.3%） |
| **`hashC`** | W2 bf16 | 6144 | **94.379** | 0.9720 | 16.029 | 6.806 | — | **winner**（−30.7%） |
| **`hashC`** | W3 fp32 | 6144 | **37.186** | 0.9633 | 7.477 | 3.866 | — | **winner**（−31.7%） |

- 有效性核对：W1 op 数 20 → 15（−25%），时延 −30.3%（超线性于 op 数，因链尾依赖缩短使每 tile 关键路径变短）。
- 确定性重放契约：chain C 仍是 `(seed, idx)` 的纯函数，`L0-4/L0-5/L1-2/L1-7` 全部通过。
- drop-rate 契约（`TRAP-dropout-rate-metric-random-input`）：rate 判据只用**常量非零输入**（ones），
  随机输入只用元素级结构 + 精确重放判据——本任务全程遵守。

current best 更新为 **`hashC` @ bs6144**（39.85 / 94.38 / 37.19 µs，L0 pass，全 workload 复测无回退）。

### Round 4 — 计算域与 seed 折叠

现象：vec_ratio 0.965–0.972，op 数 15（fp16）/14（fp32），DMA ratio 0.07–0.18 ⇒ 仍是纯 Vector 发射受限。
本轮从同一个 current best（`hashC` @ bs6144）派生三个单点分支：

| candidate vs current best | workload | bs | TaskDur(us) | vec_ratio | mte2_time(us) | mte3_time(us) | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| current best = `hashC` | W1 / W2 / W3 | 6144 | 39.845 / 94.379 / 37.186 | 0.9649 / 0.9720 / 0.9633 | 6.96 / 16.03 / 7.48 | 2.95 / 6.81 / 3.87 | pass | — |
| `f16native`（scale 在存储 dtype 可精确表示时用存储域计算，去 fp32 往返） | W1 fp16 | 6144 | 38.068 | 0.9599 | 6.074 | 2.991 | pass | −4.5%（fp16 有效） |
| 〃 | W2 bf16 | 6144 | 102.974 | 0.9716 | — | — | — | **回退 +9.1% ⇒ bf16 关闭该路径** |
| 〃 | W3 fp32 | 6144 | 39.713 | 0.9453 | — | — | — | 回退 +6.8%（fp32 不适用，UB 布局副作用） |
| **`seedfold`**（seed 折入 tile 偏移：`T.fill(h, off + seed)`，省 1 个 `vxor`） | W1 fp16 | 6144 | 32.084 | 0.9615 | 6.412 | 2.835 | — | **−19.5%** |
| 〃 | W2 bf16 | 6144 | **76.128** | 0.9701 | 15.309 | 7.185 | — | **−19.3%（W2 winner）** |
| 〃 | W3 fp32 | 6144 | 29.723 | 0.9591 | 8.575 | 5.284 | — | **−20.1%（W3 winner）** |
| `nat_seedfold`（seedfold + 存储域） | W1 fp16 | 6144 | **30.297** | 0.9345 | 6.814 | 3.007 | pass | **−24.0%（W1 winner）** |
| 〃 | W2 bf16 | 6144 | 85.219 | 0.9690 | — | — | — | 回退（bf16 存储域不可用） |

裁决：

- `f16native` 的 fp16 收益（−4.5%）**与** `seedfold`（−19.5%）可叠加，但 bf16/fp32 上存储域形态**实测回退**
  ⇒ 合并时按 dtype **编译期**门限：`native_domain = (dtype=="float16") and (scale 在 fp16 可精确表示)`
  （p=0.5 → scale=2.0）。这不是运行时分支，也不是 wrapper 分派——是 `@T.prim_func` 内的编译期
  常量裁剪（`PL-1.28` 同族形态）。
- `seedfold` 的语义评估（如实记录）：`h` 仍是 `(seed, idx)` 的**纯函数**（契约字面成立：同 seed 同 idx
  同结果），掩码仍是**固定势的伪随机子集**（counter 到 idx 仍是双射），drop rate/结构/重放全部通过；
  变化在于：seed 由「hash 的 key」变为「counter 的常数偏移」，不同 seed 的掩码是同一 lattice 的不同
  区段（seed 间差异 0.5002，仍远高于判据下限 0.3）。**这是相对 Stage 3 的语义弱化，已在此显式记录**；
  其收益（−19.5%）与用途（dropout 掩码不要求 seed 间独立同分布，只要求同 seed 可重放）相符。
- bf16 存储域路径**明确否决**（实测回退 9.1%），并记入 pattern-library 待回写项（见 §6）。

current best 更新为 **`x_final`**（= `u5_diet` + chain C + seedfold + fp16-only 存储域），
但该分支文件尚未合并 8192 tile；R5 完成合并与配置终选。

### Round 5 — 核内合并 + 配置终选

从同一个 current best 派生合并版 `x_final`（一次合入 R3+R4 的两个已单点证明有效的结构点 +
R2 的 UB diet；bf16 存储域按 R4 的负结论**不**合入）。合并版需复测本 kernel 全部 tune workload：

| candidate vs current best | workload | bs | TaskDur(us) | vec_ratio | mte2_time(us) | mte3_time(us) | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| current best = `hashC` | W1 fp16 | 6144 | 39.845 | 0.9649 | 6.958 | 2.947 | pass | — |
| `x_final`（合并） | W1 fp16 | 6144 | 29.344 | 0.9577 | 6.047 | 2.844 | — | −26.4% |
| `x_final` | W1 fp16 | 8192 | **28.145** | 0.9575 | 5.256 | 2.398 | pass | **winner**（−29.4%） |
| `x_final` | W1 fp16 | 10240 / 12288 | — | — | — | — | — | invalid: `ub overflow`（1802240 / 2162688 bits > 1572864） |
| `x_final` | W2 bf16 | 6144 | 76.120 | 0.9704 | 15.690 | 6.752 | — | −19.3% |
| `x_final` | W2 bf16 | 8192 | **73.787** | 0.9689 | 11.772 | 5.911 | — | **winner**（−21.8%） |
| `x_final` | W3 fp32 | 6144 | 29.499 | 0.9587 | 7.328 | 5.265 | — | −20.7% |
| `x_final` | W3 fp32 | 8192 | **28.585** | 0.9596 | 5.766 | 3.665 | — | **winner**（−23.1%） |

- **合并非回退检查**：合并版在本 kernel 全部 3 个 tune workload 上均优于合并前的 current best
  （−26.4% / −19.3% / −20.7%），无回退 ⇒ 合并成立。合并期间未引入任何 Python factory / op 层 /
  wrapper 分派——`native_domain` 是 `@T.prim_func` 内的编译期常量裁剪。
- smoke 精度：合并后与最终产物各跑一次（§4.3），3 dtype 全过。

### Autotune / 配置终选（`block_size`）

`block_size` 为唯一参数搜索维度（无 autotune 框架，手动扫描；UB cap 由 BiShengIR `ub overflow` 二分确定）。

| 分支 | dtype | bs 扫描（TaskDur µs） | 终选 |
|---|---|---|---|
| `u5_diet` | fp16 | 2048 67.07 / 4096 58.74 / **6144 57.20** / 7168 57.99 / 8192 overflow | 6144 |
| `hashC` | fp16 | 4096 40.60 / **6144 39.85** | 6144 |
| `hashC` | bf16 | 4096 98.53 / **6144 94.38** | 6144 |
| `hashC` | fp32 | 4096 38.58 / **6144 37.19** | 6144 |
| `x_final` | fp16 | 6144 29.34 / **8192 28.15** / 10240 overflow | **8192** |
| `x_final` | bf16 | 6144 76.12 / **8192 73.79** / 10240 overflow | **8192** |
| `x_final` | fp32 | 6144 29.50 / **8192 28.59** / 10240 overflow | **8192** |

**8192 vs 6144 的 <5% 平区裁决**（T-2 交错 A/B/A/B + 配对符号检验，`ab_test.py`）：

```
python3 .agents/tools/ab_test.py --a-cmd "<x_final bs6144 bench>" --b-cmd "<x_final bs8192 bench>" \
  --workdir perf_opt/ab_round/bs_w1 --pairs 3 --launch-count 15
→ a_merged_median_us=29.320  b_merged_median_us=27.980  b_rel_change=-0.0457
  pair_diffs_us=[-0.9, -1.58, -1.34]  direction_consistent=true  verdict="B_better"
```

⇒ bs8192 采纳（−4.57%，方向一致），写为最终模块级常量
`_DEFAULT_BLOCK_SIZE = {"float32": 8192, "float16": 8192, "bfloat16": 8192}`
（**wrapper 切换块须成对同步该常量**：Stage 3 wrapper 的值为 1024/2048/2048）。

---

## 3. 最终产物与 op 数账

最终 `perf_opt/_make_dropout_kernel.py`（kernel 体与 `_x_final.py` 逐字节相同，
差异仅 `_DEFAULT_BLOCK_SIZE` 与模块 docstring）：

| 阶段 | fp16 向量 op 数 | fp32 向量 op 数 | 组成 |
|---|---:|---:|---|
| Stage 3 baseline | 20 | 18 | 索引 2 + hash 9 + 掩码 4 + mask-cast 1 + 数据 4（fp16 含 2 次 vcast） |
| R2 `u5_diet` | 19 | 18 | 掩码 4→3（`vclamp` 替 `vmax+vmin`）、`vand`→`vabs` |
| R3 `hashC` | 15 | 14 | hash 9→5（`c*A; c^=c<<13; c*B`，seed 仍为 xor） |
| R5 final | **13** | **12** | seed 折入 counter（−1）+ fp16 去 fp32 往返（−2, fp16 only） |

确定性/契约核对（最终文件实跑）：

- fp16 存储域路径与 fp32 往返路径**逐位一致**（同一 hash 下 `torch.equal == True`，
  `kept_max_err_vs_x2 = 0.000e+00`）——见 `logs/r4/native_equiv.log`（`_w_seedfold.py` vs `_x_final.py`）。
- `--level all`：L0 22 例 / L1 12 例 / L2 14/14 / Boundary 9/9 全过（`logs/final/final_level_all.log`）。

---

## 4. Final Summary

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp16-float16 | 65.106 | 27.913 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-10k-bf16-bfloat16 | 158.356 | 73.899 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp32-float32 | 77.110 | 28.561 | final |

- final 记录: `perf_records.jsonl` 中 `phase=final` 的 3 行，`candidate_id` 全为 `final`，
  `artifact_path = _make_dropout_kernel.py`，`artifact_sha256` 与最终文件一致，
  `msprof_raw_path` 见各行的 `profiles/final/<workload_id>/final_bs8192/OPPROF_*`。
- final 采集为 `--launch-count=30 --warm-up=5`（稳定档），逐行 `l0_pass = true`。

### 4.2 提升对账

| workload | dtype | N | bs(baseline→final) | baseline(µs) | final(µs) | 提升 | 有效 R+W（base→final） |
|---|---|---:|---|---:|---:|---:|---|
| tokens-1k-hidden-4k-fp16-float16 | fp16 | 4,194,304 | 2048 → 8192 | 65.106 | 27.913 | **−57.1%（2.33x）** | 0.258 → 0.601 TB/s |
| tokens-1k-hidden-10k-bf16-bfloat16 | bf16 | 10,485,760 | 2048 → 8192 | 158.356 | 73.899 | **−53.3%（2.14x）** | 0.265 → 0.567 TB/s |
| tokens-1k-hidden-4k-fp32-float32 | fp32 | 4,194,304 | 1024 → 8192 | 77.110 | 28.561 | **−63.0%（2.70x）** | 0.436 → 1.177 TB/s |

（依据: `perf_records.jsonl` 中同一 `candidate_id="final"`/`phase="final"` 的记录；
未拼接任何实验分支的最好成绩。）

### 4.3 smoke 最终精度

| workload | shape | dtype | 结构判据 | 精确重放 | drop rate（ones, 3σ） | 结果 |
|---|---|---|---|---|---|---|
| smoke-dropout-1m-float32 | [1024,1024] | fp32 | pass | pass | 0.4997 ok | **pass** |
| smoke-dropout-1m-float16 | [1024,1024] | fp16 | pass | pass | 0.4997 ok | **pass** |
| smoke-dropout-1m-bfloat16 | [1024,1024] | bf16 | pass | pass | 0.4997 ok | **pass** |

原始输出: `logs/final/smoke_precision.json`（结构 = 每个元素 ∈ {0, x·scale}；重放 = 同 seed 逐位相等；
rate 判据用**常量非零输入**，遵守 `TRAP-dropout-rate-metric-random-input`）。
smoke 未采集性能、未参与 winner 选择与性能门禁。

### 4.4 终止条件与结论

- 终止原因: `budget_exhausted`（5 轮迭代 + 终选，48 条 `msprof` 测量记录，超过默认 `max_experiments=30`；
  同时满足 `plateau` 判据——R5 之后所有剩余方向均已记为 blocked（标量操作数不可用 / UB cap 8192）
  或 invalid（静态整块分支 L0 不通过），无新的有效分支）。
- 未达 `success` 的定义前提说明: 性能目标为 `best_effort`（无逐 workload 数值目标），
  按 skill「以通过核内合并与全量非回退检查的实测最优结果作为该 workload 的结论」收束。
- 遗留（如实报告，未伪装为完成）: 无 `merge_blocked`；3 个 tune workload 全部 `winner_merged`。
- 残留风险（未在本阶段量化）: ① 最终默认 `block_size=8192` 对小 shape（如 smoke 的 1M 元素 = 128 tile
  / 40 核 = 3.2 波）可能存在波数不整的负载不均——smoke 按契约不做性能采集，未测；
  可用的对策是 `PL-1.38` 的编译期 tile 波数守卫（本阶段**未**采纳，因无法在 tune 集上量化其收益）。
  ② 有效 R+W 0.57–1.18 TB/s 仍低于同机逐元素天花板 2.49–2.67 TB/s，差距全部来自软件 PRNG 链
  （见 §6 与 `perf_feedback.md`）。

---

## 5. `perf_records.jsonl` 说明（含机械修正留痕）

- 文件共 **48 行**，`phase` 分布: `baseline` 3 / `candidate` 36 / `merged` 6 / `final` 3；
  `candidate_id` 分布: baseline 7 / u5_diet 14 / u6_vclamp 3 / hashC 6 / f16native 3 / seedfold 3 /
  nat_seedfold 3 / x_final 6 / final 3。每一行对应一次 `msprof op` 采集；每次采集的 `duration_us`
  **自始至终未被修改**。
- **留痕（如实披露三次机械修正，均未改测量值）**:
  1. **字段名归一**：首轮采数时我自拟了字段名（`task_duration_us` / `l0_result` / `raw_profile_dir` /
     `phase=rN`），不符合 `_shared/standards/signal-registry.md` §5（应为 `duration_us` / `l0_pass` /
     `msprof_raw_path` / `phase ∈ {baseline,candidate,merged,final}`）⇒ 用 `normalize_records.py`
     改写字段名与枚举，**测量值逐行不变、行数不变**。归一前原文：`logs/r0/perf_records_preschema.jsonl`。
  2. **`l0_pass` 回填**：门禁要求每行 `l0_pass is True`。L0 是**文件级**精度套件，故对每个被引用的
     候选文件各跑一次 L0 并把结果回填（证据路径写进每行的 `l0_evidence` 字段）：
     `_u5_diet.py`/`_v_hashC.py`/`_w_f16native.py`/`_w_native_seedfold.py`/`_x_final.py`/`_make_dropout_kernel.py`
     各自 L0 PASS；`_u6_vclamp.py`/`_w_seedfold.py` 本次补跑 L0 PASS（`logs/l0/`）。
     这些文件在对应行采集后**未被修改**（`artifact_sha256` 与采集时一致），故 L0 结论对该行有效。
  3. **失效分支行移除**：`c1_sbulk`/`u7_sb_diet` 两个静态整块分支的文件级 L0 **FAIL**
     （`Undefined variable: tail`，见 §2 Round 2），按 skill 契约「blocked-compile / invalid 分支不写入
     `perf_records.jsonl`」移除其 6 行。移除前全文：`logs/r0/perf_records_preschema_p2.jsonl`。
     该 6 行的 msprof 数值仍完整保留在 §2 Round 2 表中（标注为 invalid，仅供参考）。
- 空间整理（留痕）：收束时删除了各 `OPPROF_*` 目录下的大体积设备原始二进制（`DeviceProf*.bin`、`aicore_binary.o`），
  `profiles/` 由 2.4 GB 减到 87 MB；**全部 CSV 证据（`OpBasicInfo.csv`/`PipeUtilization.csv`/`Memory*.csv`/`L2Cache.csv` 等）保留**，
  记录里的 `msprof_raw_path` 仍逐条指向真实存在的 `OPPROF_*` 目录；`ab_round/` 同处理。
- Final 一致性: 3 条 `phase=final` 行的 `artifact_path` 为裸文件名 `_make_dropout_kernel.py`
  （相对 `perf_opt/`），`artifact_sha256` 与当前最终文件实测一致（已核对）。

---

## 6. Skill Retrospective

### 6.1 流程问题

1. **R-1（流程）`perf_records.jsonl` 字段契约未在采数前机械校验**：第一次 `msprof` 采集前我没有
   落地 §5 的字段名，导致 54 行采完后需要一次 schema 归一（本文档 §5 留痕）。建议在 Phase 1
   之前加一步「按 §5 生成一条空模板行并 schema 自检」，或让 `run_experiments.py` 模板
   （`references/run_experiments_template.py`）直接携带契约字段名。
2. **R-2（流程）smoke 的「不采集性能」与「小 shape 回退风险」存在张力**：`block_size` 调优
   必然改变小 shape 的波数结构，而 smoke 按契约不可测性能 ⇒ 该风险只能以「未测/残留风险」
   记录（§4.4）。建议标准里明确：tile 类配置改动导致的小 shape 风险，允许以**编译期守卫**
   （如 `PL-1.38`）作为默认处置，并在 opt_log 中标注为「未在 tune 集量化」。
3. **R-3（工具）`msprof op` 的 `--aic-metrics=Default` 已足以支撑本类诊断**（vec/mte2/mte3/scalar
   ratio + time），无需逗号长列表；但 `OpBasicInfo.csv` 的 10 次 launch 需自行聚合（本任务采用
   均值 + min/max 记录）。

### 6.2 Value Point Proposals

| vp_type | 提案 | 证据（evidence + repro + toolchain_stamp） |
|---|---|---|
| **D** | 910B3 上 dropout 类「软件 PRNG + 逐元素」算子的 op 数–时延标定：fp16 20 op→65.1µs(W1) / 15 op→39.8µs / 13 op→27.9µs；即 **每 op ≈ 2.0–2.6 µs / 4.19M fp16 元素**（bs 6144–8192，40 Vector core），ops 减少与 TaskDur 近似线性（vec_ratio ≈ 0.96 时） | evidence: 本文档 §2 R2/R3/R5 表 + `perf_records.jsonl`; repro: `_hash_analysis.py`（算法侧）+ `profiles/{r3,r5,final}`; toolchain: tilelang 0.1.2+90fb7ae37f / CANN 9.0.0 / 910B3 / 2026-10-17 |
| **D** | **counter hash 链的统计质量–op 数权衡实测**：Stage 3 的 9-op 链、7-op 链、5-op 链（`c*A; c^=c<<13; c*B`）在 4–8 Mi 元素上 rate 偏差 <3σ、block-χ²(dof 4095) 3812–4270 同档 ⇒ **5-op 链即可满足 drop-rate/独立性契约**，推翻「~9 op 无更优替代」的设计期判断 | evidence: `_hash_analysis.py` 输出表 + `perf_records.jsonl`(hashC 行 −30%); repro: `_hash_analysis.py`（自包含整数模型）; toolchain: numpy int32 模型 + tilelang 0.1.2+90fb7ae37f / 910B3 / 2026-10-17 |
| **P** | 优化点：**无 RNG 原语时把 seed 折入 counter 偏移**（`T.fill(h, off + seed)` 取代 `T.vxor(h, seed, h)`）——省 1 个 full-size v-op（−19.5%，4.19M fp16）。见解：seed 作为 counter 常数偏移时 `h` 仍是 `(seed,idx)` 纯函数、掩码仍是固定势子集，契约成立；代价是 seed 间掩码为同一 lattice 区段（分离度实测 0.5002） | evidence: `perf_records.jsonl` seedfold 行（budget 内 3 workload 全部 −19.3~−20.1%）; repro 待补（`repro-missing`，最小形态 = 1 个 v-op 替换的前后对照）; toolchain: tilelang 0.1.2+90fb7ae37f / CANN 9.0.0 / 910B3 / 2026-10-17 |
| **D** | **910B3 上 `vand/vxor/vmax` 不接受标量操作数（字面量），`tir` 表达式/`T.cast` 触发前端拒绝或段错误（rc=139）**；`T.vadd(lane, <runtime scalar>, h)` 亦段错误 ⇒ UB diet 只能沿「buffer 合并 + `vabs`/`vclamp` 替代」方向 | evidence: `probes/_u1_scalarmask.py`（vand 拒绝）、`probes/_u3_scalarseed.py`、`probes/_v_idx1.py`（段错误）; repro: `probes/_probe_ops.py`; toolchain: tilelang 0.1.2+90fb7ae37f / 910B3 / 2026-10-17 |
| **D** | **bf16 存储域逐元素计算在本设备（910B3）仍慢于 fp32 往返**：同一 hash 下 bf16 存储域（2 cast + 2 vmul）102.97µs vs fp32 往返 94.38µs（+9.1%）⇒ 与 `TRAP-bf16-vop-doc-matrix-divergence`（910_9362 原生可用）结论**不冲突但设备相关**：可用 ≠ 更快 | evidence: `perf_records.jsonl` f16native/W2 vs hashC/W2; repro: 两分支文件 `_w_f16native.py` vs `_v_hashC.py`; toolchain: tilelang 0.1.2+90fb7ae37f / 910B3 / 2026-10-17 |
| **P** | 反向/否决记录：`PL-1.34` 静态整块拷贝在 **Vector 受限**（`mte2_ratio ≤ 0.18`）的算子上**未获干净实测**——本任务实现的编译期尾块分支因 TVM parser 的 if 作用域（`Undefined variable: tail`）L0 不通过，属结构实现缺陷而非模式证伪；**结论待重验**（重验入口：把 `tail` 定义提到 `if` 之外再走 L0 + msprof）。可确认的只有一条：该实现在 bs8192 抬高 UB 需求（2228224 → 2359296 bits） | evidence: §2 R2 表（标注 invalid）+ `logs/l0/L0__make_dropout_kernel_round2_staticbulk.log`; repro: `probes/_make_dropout_kernel_round2_staticbulk.py`; toolchain: tilelang 0.1.2+90fb7ae37f / 910B3 / 2026-10-17 |
| **R** | 流程：`perf_records.jsonl` 应在**采数前**落地字段契约（`round/candidate_id/parent_id/kernel_id/workload_id/phase/artifact_path/artifact_sha256/duration_us/l0_pass/msprof_raw_path/timestamp`），且 **invalid 分支不得写入**——本任务因先采数后对齐契约，产生了一次字段名归一 + 一次 `l0_pass` 回填 + 一次失效行移除（§5 全部留痕）。建议 `run_experiments_template.py` 直接携带契约字段名并在写行前做 `l0_pass` 校验 | evidence: §5 三次修正 + `normalize_records.py`; repro: `run_experiments.py`（本目录，含 `collect_metrics`/`profile_row`）；toolchain: 无（流程类） |

> 本任务**未**回写 `pattern-library/`（上面 7 条 proposal 均留给任务终态 `tilelang-skill-evolver` 分级合入；
> 其中 D 类可用于 Tier 0，P 类需二次独立证据）。理由：本任务已接近预算上限，优先保证产物与日志完整；
> 回写职责与 evolver 互补（未回写 ⇒ 不构成双写）。

### 6.3 未采纳方向（供后续任务参考）

- 整数 hash 降到 int16 lane 以翻倍向量吞吐：会让 mask 以 65536 为周期重复（语义退化），否决。
- seed 折入 hash 常量（`A_seed`）：单乘法的 23-bit 提取有强 lattice 结构，分布质量不足，否决。
- 掩码选择式（`vselect`）/ host 端物化掩码：分别受 `PL-1.29-vselect-cost`（6.5x）与
  `ascend-constraints.md` §4（host 禁核心计算）否决，未尝试。
