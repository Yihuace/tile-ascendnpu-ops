# GeluTanhAndMulFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> （`/home/developer/tilelang-mlir-ascend`，AscendNPU-IR `31f6903`）/ torch 2.7.1 / torch_npu 2.7.1.post4 /
> **Ascend910B3**（Chip 6，20 AIC / 40 AIV）/ 2026-10-18。
> 性能口径：`msprof op --kernel-name=main` 的 **Task Duration(us)**（唯一时延口径）+
> `visualize_data.bin` 的 **GM Read + Write SOL ratio**。
> 对象：`GeluTanhAndMulFwdOp`（`y = gelu_tanh(x[:,:N]) * x[:,N:]`），纯 Vector（AIV），单 kernel/单 func。
> 测量方式：`perf_opt/bench_kernel.py` 直跑 kernel（不经 wrapper），每次测量是独立 msprof 进程。

## 0. 对象与入口

- 基线 kernel（Stage 3，只读）：`.../gelu_tanh_and_mul_kernel/gelu_tanh_and_mul_fwd_kernel.py`
- 上一轮 perf_opt（CANN 8.5.2，fold 优化，本轮起点 current best）：`perf_opt/gelu_tanh_and_mul_fwd_kernel.py`
- 本轮调优产物：`perf_opt/GeluTanhAndMulFwdOp.py`（Stage-4 winner）
- benchmark 入口：`benchmarks/ops/bench_gelu_tanh_and_mul.py::test_gelu_tanh_and_mul_manifest_bench`
- workload：**5 tune + 3 smoke**（`perf_opt/workload_inventory.json`）
- 采集脚本：`perf_opt/run_experiments.py`（T-4 批处理：L0 门禁 → msprof → `perf_records.jsonl` 追加）

## 1. Performance Test Data（baseline）

上一轮（CANN 8.5.2）已把「fold sqrt(2/pi)×(−2) 为一次 vmul」合入 current best；本轮在 CANN 9.0.0 上重采 baseline（Stage 3 无 fold）与 current best（fold），数据基本复现 8.5.2：

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | SOL% | profile_status |
|---|---|---|---|---|---:|---:|---|
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-prefill-float16 | baseline | main | main | 215.154 | 47.91 | valid |
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-prefill-float16 | fold | main | main | 203.984 | 47.42 | valid |
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-prefill-bfloat16 | baseline | main | main | 220.114 | 49.72 | valid |
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-prefill-bfloat16 | fold | main | main | 208.824 | 49.43 | valid |
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-decode-bfloat16 | baseline | main | main | 2.160 | 2.56 | valid |
| gelu_tanh_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-float16 | baseline | main | main | 5.980 | 23.27 | valid |
| gelu_tanh_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-bfloat16 | baseline | main | main | 6.060 | 24.06 | valid |

- smoke（16×128，fp32/fp16/bf16）：仅精度回归，不做性能采集。
- decode bf16（2.16µs）、swiglu-b32（5.7–6.1µs）：微型/近微型，见 §3 豁免判定。

### 设计估算 vs 实测偏差（D-2 回填）

DESIGN §1.6.0 定量 roofline 估算 prefill fp16 下界 ≈ 226µs（带宽受限口径，780 GB/s 流量项）。
实测 baseline 215.15µs / final 197.44µs，**低于估算下界**——因为该算子实为 **compute-bound**
（`Vec_FP32` roofline 利用率 **99.09%**，GM SOL 仅 ~46%），DESIGN 的「带宽受限 226µs」假设在
本工具链上失准（把 MTE2 带宽当作地板，实为向量 FP32 发射/计算打满）。这是 [DESIGN_LIMIT] 归因
中「实测与 DESIGN 假设矛盾」的量化证据之一（见 §5 天花板判定，但未达 2× 结构性加速门槛，不触发逆向反馈）。

## 2. 诊断

- 全 workload `Vec_FP32 ≈ 99%`、`Vec_MISC ≈ 0.9%`（roofline `computility_name`）→ **compute-bound**：
  sigmoid 形式 tanh（`vexp` + `vdiv`）内部展开为 FP32 多项式序列，把 Vector FP32 单元打满。
- GM Read + Write SOL ≈ 45–50%（prefill）——**非正确判据**（豁免规则 3：实测向量利用率 ≥0.9 时改判）。
- 链长 14 条向量 op（fp16/bf16：2 upcast + 5 多项式 + 1 vexp + 1 vadd + 1 vbrc + 1 vdiv + 2 vmul + 1 downcast）；
  唯一可削的是**逐 tile 重复填充常数 1** 的 `vbrc`（循环不变量）。

## 3. 迭代日志（候选 vs current best）

### Round 1 — 单点优化分支（op1 流水 / op2 提升 vbrc / op3 静态满块+流水）

全部从 current best（fold）派生，各只改一个主要优化点；prefill 是主判 workload：

| 候选 | 改动 | prefill fp16 µs | prefill bf16 µs | swiglu fp16 µs | L0 | 结论 |
|---|---|---:|---:|---:|---|---|
| `fold` | current best（fold） | 204.024 | 209.044 | 5.790 | PASS | 参照 |
| `op1`（pipelined） | `T.serial`→`T.Pipelined(2)` | 204.024 | 208.954 | 5.810 | PASS | ❌ tie（compute-bound 无可重叠） |
| `op2`（hoist vbrc） | 常数 1 缓冲 `vbrc` 提到循环外 | **197.494** | **202.234** | 5.660 | PASS | ✅ **winner**（−3.2%/−3.3%） |
| `op3`（static+pipe） | 整除 N 走静态满块 + `T.Pipelined(2)` | 204.204 | 208.794 | 5.720 | PASS | ❌ tie（同 op1，流水无效） |

- **op2 机制**：`T.vbrc(1.0, one32)` 填充常数 1 缓冲，原实现逐 tile 重复执行（prefill 每核 ~358 次），
  实际只需每核一次——提升到循环外省掉 ~357 次/核的向量 op。数学严格等价（缓冲内容不变）。
- **op1/op3 否决根因**：compute-bound（Vec_FP32 99%）下 GM→UB 搬运已被 auto-multi-buffer 隐藏，
  软件流水无可重叠阶段，故 tie。这印证 pattern-library `PL-1.6-copy-floor`（≥16M 档向量链全隐藏）同族结论。

### Round 2 — block_N 粗扫（op2 结构上，prefill 主判，reps=2）

| 候选 | block_N | prefill fp16 µs（2rep 中位） | prefill bf16 µs（2rep 中位） | 结论 |
|---|---:|---:|---:|---|
| op2-bs1024 | 1024 | 246.75 | 249.95 | ❌ 大幅回退（tile 过小、发射开销↑） |
| op2-bs1536 | 1536 | 230.06 | 233.45 | ❌ 回退 |
| **op2-bs2048** | 2048 | **197.48** | **202.22** | ✅ **winner** |
| op2-bs2560 | 2560 | 200.53 | 205.40 | ❌ 略回退 |

- block_N=2048 维持 baseline 默认值最优；block_N≥4096 因 UB 超限（~50B/elem × 4096 = 200KB > 192KB）不可编译。

### 合并

op2（hoist vbrc）是唯一有效单点；op1/op3 无增益不合并。合并版 = fold + op2，复测全 tune workload：

| merged | workload | current_best_us（fold） | merged_us | 精度 | 状态 |
|---|---|---:|---:|---|---|
| GeluTanhAndMulFwdOp | prefill fp16 | 204.024 | 197.444 | PASS（--level all） | pass |
| GeluTanhAndMulFwdOp | prefill bf16 | 209.044 | 202.274 | PASS | pass |
| GeluTanhAndMulFwdOp | swiglu fp16 | 5.790 | 5.680 | PASS | pass |
| GeluTanhAndMulFwdOp | swiglu bf16 | 5.930 | 5.760 | PASS | pass |
| GeluTanhAndMulFwdOp | decode bf16 | 2.190 | 2.140 | PASS | pass（微型豁免） |

smoke（16×128，fp32/fp16/bf16）：最终 `--level all` 全绿（含 L0-4 smoke 精度）。

## 4. 豁免判定（用户口径）

1. **decode bf16**：Task Duration 2.14µs ≤ 5µs → 微型 shape 豁免。
2. **swiglu-b32**（5.68/5.76µs）：近微型，现场复核未达 ≤5µs 豁免线，但为 small-shape（<20µs），
   差异 <5% 已按交错 A/B 纪律处理（op2 在两 workload 上一致方向 −2.2%/−2.3%，且 prefill 主判 −3.2% 超出 3% 噪声阈）。
3. **判据失配**：`Vec_FP32 = 99.09%` ≥ 0.9 → compute-bound，改用向量利用率判据；达标硬底线是
   **Task Duration 变短**（已满足：prefill −8.2%，swiglu −5.0%）。

## 5. 天花板判定（[DESIGN_LIMIT] 触发条件核对）

- 条件①（设计层天花板）：残余开销归因为 sigmoid 形式 tanh 的 `vexp+vdiv` FP32 多项式链——属
  DESIGN §0.6 R2 选定算法，非 tiling/参数可解。**部分成立**。
- 条件②（结构性加速 >2x 或实测与 DESIGN 假设矛盾）：DESIGN 假设「带宽受限 226µs」与实测
  「compute-bound 99%」矛盾；但**结构性加速估计**（替换 sigmoid 为多项式近似）约 10–20%，
  **远未达 2×**。**不成立**。
- **结论**：两项未同时满足 → **不产出 `perf_feedback.md`**，不附 `[DESIGN_LIMIT]` 信号。
  残余 13 条向量 op 是「全 fp32 域 + 单次 cast」精度契约下的 floor（fp32 需精确 sigmoid，无法用近似指令），
  属参数/结构均不可解的 compute 地板，如实披露于报告。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-prefill-float16 | 215.154 | 197.444 | final |
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-prefill-bfloat16 | 220.114 | 202.274 | final |
| gelu_tanh_and_mul_fwd_kernel | llama-8b-ffn-decode-bfloat16 | 2.160 | 2.140 | final |
| gelu_tanh_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-float16 | 5.980 | 5.680 | final |
| gelu_tanh_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-bfloat16 | 6.060 | 5.760 | final |

## 6. Final Summary（逐 workload 收益与 SOL，同一最终候选）

| workload_id | Δ% | SOL% |
|---|---:|---:|
| llama-8b-ffn-prefill-float16 | **−8.2%** | 45.57 |
| llama-8b-ffn-prefill-bfloat16 | **−8.1%** | 47.71 |
| llama-8b-ffn-decode-bfloat16 | −0.9%（微型豁免） | 2.43 |
| qwen3-8b-swiglu-b32-float16 | −5.0% | 22.04 |
| qwen3-8b-swiglu-b32-bfloat16 | −5.0% | 22.99 |

- 达标率：可调 shape（prefill fp16/bf16 + swiglu fp16/bf16）= **4/4 达标**（Task Duration 全部下降，判据失配改判向量利用率）；全 shape（含 decode 微型）= 4/4 可调 + 1 豁免 = 5/5（豁免不计入分母）。
- 停止原因：**success**（达到判据——Task Duration 下降；SOL 60% 目标因 compute-bound 判据失配不适用）。
- 精度：`python perf_opt/GeluTanhAndMulFwdOp.py --level all` 全绿（L0/L1/L2/Boundary，含 tanh 饱和 / 大负值 / signed-zero / inf/nan）。

## 7. Skill Retrospective

### 流程问题

1. **legacy perf_records 跨任务污染**：上一轮（CANN 8.5.2）留下的 3 条 `perf_records.jsonl` 使用旧 schema
   （`op/label/dtype/baseline_us/perf_opt_us/...`），与本轮 signal-registry §5 字段契约不兼容，会机械触发
   `S4-PERF-RECORDS-SCHEMA`。本轮将其移出主文件、归档到 `logs/legacy_perf_records_prior_run.jsonl` 并留痕。
   建议（R 类）：`init` 重开 Stage 4 时应提示清理/归档旧 `perf_opt/` 记录，或在 gate 侧对「旧 schema 行」显式
   deprecate 而非硬失败。

### Value Point Proposals

| ID | vp_type | 内容 | 证据三件套（evidence / repro / toolchain） |
|---|---|---|---|
| VP-2026-0030 | P | **循环不变量缓冲提升**：逐 tile 重复填充的常数缓冲（`vbrc(1.0)`）提到 persistent 循环外，每核只填一次；prefill compute-bound 逐元素算子实测 **−3.2%**。op1/op3 流水 tie 佐证「compute-bound 下流水无收益」（承 PL-1.6-copy-floor 同族）。 | opt_log §3 Round 1 表 / repro 待回填（最小形态 = hoist `T.vbrc` 出 serial 循环）/ tilelang 31f6903 + CANN 9.0.0 + 910B3 |
| VP-2026-0031 | D | **GeluTanhAndMulFwdOp compute-bound 实测常数**：prefill fp16 final 197.44µs、`Vec_FP32=99.09%`、GM SOL 45.57%；fold 单点 −5.2%、hoist-vbrc 单点 −3.2%。 | opt_log §1/§6 / repro-missing / 同 VP-2026-0030 工具链 |
| VP-2026-0032 | R | **DESIGN 带宽受限估算失准**：DESIGN §1.6.0 以 MTE2 带宽估算 prefill 226µs 下界，实测为 compute-bound（Vec_FP32 99%），下界估算方向错误。 | opt_log §1「设计估算 vs 实测」/ — / 同工具链 |

- 待回填 repro：VP-2026-0030 的 hoist-vbrc 最小形态（ED-B）；本轮时间受限，标 `repro-missing` 入队。
