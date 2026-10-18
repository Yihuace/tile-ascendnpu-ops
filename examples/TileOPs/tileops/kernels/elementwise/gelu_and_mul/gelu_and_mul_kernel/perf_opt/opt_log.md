# GeluAndMulFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> （`/home/developer/tilelang-mlir-ascend`，AscendNPU-IR `31f6903`）/ torch 2.7.1 / torch_npu 2.7.1.post4 /
> **Ascend910B3**（Chip 6，20 AIC / 40 AIV）/ 2026-10-18。
> 性能口径：`msprof op --kernel-name=main` 的 **Task Duration(us)**（唯一时延口径）+
> `visualize_data.bin` 的 **GM Read + Write SOL ratio**。
> 对象：`GeluAndMulFwdOp`（`y = gelu(x[:,:N]) * x[:,N:]`，精确 erf GELU），纯 Vector（AIV），单 kernel/单 func。
> 测量方式：`perf_opt/bench_kernel.py` 直跑 kernel（不经 wrapper），每次测量是独立 msprof 进程。

## 0. 对象与入口

- 基线 kernel（Stage 3，只读）：`.../gelu_and_mul_kernel/gelu_and_mul_fwd_kernel.py`（deg-20 多项式 + NaN 恢复）
- 上一轮 perf_opt（CANN 8.5.2，删 NaN 恢复，本轮起点 current best）：`perf_opt/gelu_and_mul_fwd_kernel.py`
- 本轮调优产物：`perf_opt/GeluAndMulFwdOp.py`（Stage-4 winner）
- benchmark 入口：`benchmarks/ops/bench_gelu_and_mul.py::test_gelu_and_mul_manifest_bench`
- workload：**5 tune + 3 smoke**（`perf_opt/workload_inventory.json`）
- 采集脚本：`perf_opt/run_experiments.py`（T-4 批处理：L0 门禁 → msprof → `perf_records.jsonl` 追加）

## 1. Performance Test Data（baseline，CANN 9.0.0 重采）

上一轮（CANN 8.5.2）已把「删除冗余 NaN 恢复」合入 current best；本轮在 CANN 9.0.0 上重采 baseline
（Stage 3 有 NaN 恢复）与 current best（nan_drop），数据基本复现 8.5.2（878.76→877.20 / 797.22→800.08）：

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | SOL% | profile_status |
|---|---|---|---|---:|---:|---|
| gelu_and_mul_fwd_kernel | llama-8b-ffn-prefill-float16 | baseline | main | main | 877.198 | 43.20 | valid |
| gelu_and_mul_fwd_kernel | llama-8b-ffn-prefill-bfloat16 | baseline | main | main | 877.188 | 44.05 | valid |
| gelu_and_mul_fwd_kernel | llama-8b-ffn-decode-bfloat16 | baseline | main | main | 3.860 | 4.89 | valid |
| gelu_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-float16 | baseline | main | main | 14.980 | 34.41 | valid |
| gelu_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-bfloat16 | baseline | main | main | 14.810 | 35.28 | valid |
| gelu_and_mul_fwd_kernel | llama-8b-ffn-prefill-float16 | nan_drop | main | main | 800.076 | 45.84 | valid |
| gelu_and_mul_fwd_kernel | llama-8b-ffn-prefill-bfloat16 | nan_drop | main | main | 804.756 | 46.37 | valid |

- smoke（16×128，fp32/fp16/bf16）：仅精度回归，不做性能采集。
- decode bf16（3.86µs）、swiglu-b32（14.8µs）：微型/近微型，见 §5 豁免判定。
- **排序理由**：优先实测性能负担最大的 prefill（M=2048/N=14336，~877µs，瓶颈明确、结构性收益最大）作主判
  workload；再处理 swiglu-b32（近微型，<20µs，差异按交错 A/B 纪律）；decode bf16（微型豁免）最后；三个
  dtype 的 smoke 只做精度回归。排序只影响先后，不跳过任何 tune workload。

### 设计估算 vs 实测偏差（D-2 回填）

DESIGN §1.6.0 定量 roofline 估算 prefill fp16 下界 ≈ 226µs（带宽受限口径，780 GB/s 流量项）。
实测 baseline 877.2µs，**约为估算下界 3.9×**——因为该算子实为 **compute-bound**（deg-20 erf 多项式
~56 条向量 op 把 Vector FP32 单元打满，`aiv_vec_ratio=0.94`，GM SOL 仅 ~44%），DESIGN 的「带宽受限
226µs」假设在本工具链上失准（把 MTE2 带宽当作地板，实为向量 FP32 发射/计算打满）。这是残留算力
天花板归因的量化证据（见 §6，但结构性加速 ~1.8× 未达 2×，不触发 [DESIGN_LIMIT]）。

## 2. 诊断

- 上一轮（CANN 8.5.2）已确认 compute-bound（`aiv_vec_ratio=0.879`）；本轮 op10 复测
  `aiv_vec_ratio=0.943`、`aiv_mte2_ratio=0.246`、`aiv_scalar_ratio=0.147`（prefill fp16，
  `PipeUtilization.csv`）→ **compute-bound 未变**，Vector FP32 单元仍是唯一瓶颈。
- GM Read + Write SOL ≈ 43–48%（prefill）——**非正确判据**（豁免规则 3：实测向量利用率 ≥0.9 时改判）。
- 链长（fp16/bf16）：2 vcast + erf 段 + gelu 链 + 1 vcast。**主瓶颈 = deg-20 Horner erf**
  （40 条 vmul/vadd），fp16/bf16 输出容差（1e-2 / 1.6e-2）远不需要 deg-20 的 2.98e-7 精度。

## 3. 迭代日志（候选 vs current best）

### Round 1 — dtype 分化多项式阶数（主杠杆）

全部从 current best（nan_drop）派生，各只改一个主要优化点：fp16/bf16 用低阶 minimax 多项式、
fp32 保持 deg-20（trace-time dtype 分支，非运行时条件路径）。prefill 是主判 workload：

| 候选 | 改动 | prefill fp16 µs | prefill bf16 µs | swiglu fp16 µs | swiglu bf16 µs | L0 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| `nan_drop` | current best（deg-20） | 800.076 | 804.756 | 13.850 | 13.910 | PASS | 参照 |
| `op10`（deg-10） | fp16/bf16 用 deg-10 minimax（erf err 1.4e-4） | **505.640** | **510.370** | **9.750** | **10.040** | PASS | ✅ **winner**（−36.8%/−36.6%） |
| `op12`（deg-12） | fp16/bf16 用 deg-12 minimax（erf err 9.7e-6） | 564.521 | 569.381 | 10.740 | 10.810 | PASS | ❌ 更慢（op 更多） |

- **op10 机制**：fp16/bf16 只需 ~1e-3 erf 精度（输出回 fp16/bf16 舍入），deg-20 严重过剩。deg-10
  minimax（系数 |c|≤309，fp32 求值，max erf err 1.4e-4）把 Horner 从 40 条向量 op 减到 20 条，
  整链 ~56 → ~36 条，compute-bound 下时延 −36.8%（800.08→505.64）。
- **op12 否决**：deg-12 比 deg-10 多 4 条 Horner op，精度过剩（erf err 9.7e-6），无性能收益。
- **精度**：op10 `--level all` 全绿（L0/L1/L2/Boundary；仅 L2-stress scale-20-both 非阻塞 WARN——
  fp16/bf16 由 deg-20 的 pass 变为 WARN，属设计 §9.1 明言的「out of spec randn regime」，见 §5）。

### Round 2 — 单点微优化（fold / vsub）

从 op10（deg-10）派生，各只改一个主要优化点：

| 候选 | 改动 | prefill fp16 µs | prefill bf16 µs | swiglu fp16 µs | swiglu bf16 µs | L0 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| `op10` | current best（deg-10） | 505.640 | 510.370 | 9.750 | 10.040 | PASS | 参照 |
| `fold` | 折叠 `z=gate/√2` 与 `u=t/4` 为 `clamp(gate,±4√2)·1/(4√2)`（系数 ×1/√2） | — | — | — | — | **FAIL** | ❌ 精度失败（破坏精确饱和） |
| `vsub` | `w=1−u²` 用 loop-hoisted 常数 1 buffer + 单条 `T.vsub`（替代 vmul-negate+vadd） | **494.210** | **498.990** | 9.870 | 9.670 | PASS | ⚠️ 方向一致但 <3%（A/B tie） |

- **fold 否决根因（关键发现，D 类）**：折叠 `inv_sqrt2`（`1/√2`）与 `inv_a`（`1/4`）为 `1/(4√2)` 并
  把系数缩放 1/√2 后，**破坏 erf 的精确饱和**——原式 `c0=1/4=0.25`（fp32 精确）且 `t=±4`（fp32 精确），
  `t·c0=±1` **精确**，故 `1+erf(±inf)=0` 精确、`gelu(-inf)=-inf·0=NaN` 成立；折叠后 `c0'=0.25/√2` 与
  `t'=4√2` 均非 fp32 精确值，`t'·c0'≈1` 但非精确，`clip(±1)` 不再保证 `±1`，`gelu(-inf)` 变 `-inf`（非 NaN），
  L1-4 大 |gate| 与 Boundary inf/nan 全部失败。**精确饱和（c0=1/A 精确）是不可妥协的设计约束**
  （DESIGN §9.2「多项式常数项非 1/A → clip 不触发」），fold 方向整体否决。
- **vsub A/B 裁决**：`ab_test.py` 交错 3 对（prefill fp16）：A(op10) 中位 505.53µs / B(vsub) 中位
  494.23µs，`b_rel_change=-2.24%`，`direction_consistent=true`（三对差 −10.98/−11.30/−11.42µs），
  `sign_test_p=0.25`，`verdict=tie`（<3% 噪声阈）。机制明确（compute-bound 94% vec 下少 1 条向量 op ≈
  2.8%），方向一致且 L0 全绿 → 作为**子阈值结构改进**采纳（不记为 >3% winner），不引入精度风险。

### Round 3 — 合并与最终

winner = op10（deg-10）+ vsub（hoist 常数 1 + 单条 vsub），合并为最终候选 `GeluAndMulFwdOp.py`，
`--level all` 全绿（L0/L1 ALL PASS，L2/Boundary 非阻塞），复测全部 tune workload（见 §8 Final）。

| merged | workload | current_best_us（nan_drop） | merged_us | 精度 | 状态 |
|---|---|---:|---:|---|---|
| GeluAndMulFwdOp | prefill fp16 | 800.076 | 494.380 | PASS（--level all） | pass |
| GeluAndMulFwdOp | prefill bf16 | 804.756 | 499.190 | PASS | pass |
| GeluAndMulFwdOp | swiglu fp16 | 13.850 | 9.670 | PASS | pass |
| GeluAndMulFwdOp | swiglu bf16 | 13.910 | 9.900 | PASS | pass |
| GeluAndMulFwdOp | decode bf16 | 3.880 | 3.000 | PASS | pass（微型豁免） |

smoke（16×128，fp32/fp16/bf16）：最终 `--level all` 全绿（含 L0-4 smoke 精度），`precision_pass=true`。

## 4. 候选 vs current best 结构化对比（B2，数据来自 perf_records.jsonl）

| 候选 | 阶段 | prefill fp16 µs | prefill bf16 µs | AICore Vec ratio | mte2 ratio | L0 | 备注 |
|---|---|---:|---:|---:|---:|---|---|
| baseline（deg-20+NaN恢复） | baseline | 877.20 | 877.19 | 0.879（8.5.2） | 0.142（8.5.2） | PASS | Stage 3 |
| nan_drop（deg-20） | 起点 | 800.08 | 804.76 | ~0.88 | ~0.14 | PASS | 上一轮 winner |
| op10（deg-10） | R1 winner | 505.64 | 510.37 | **0.943** | 0.246 | PASS | 主杠杆 |
| op12（deg-12） | R1 | 564.52 | 569.38 | — | — | PASS | 精度过剩 |
| vsub（deg-10+vsub） | R2 | 494.21 | 498.99 | ~0.94 | ~0.25 | PASS | 子阈值 −2.2% |
| **final（deg-10+vsub）** | final | **494.38** | **499.19** | ~0.94 | ~0.25 | PASS | 最终 winner |

## 5. 豁免判定（用户口径）

1. **decode bf16**：Task Duration 3.00µs ≤ 5µs → 微型 shape 豁免。
2. **swiglu-b32**（9.67/9.90µs）：small-shape（<20µs），差异按交错 A/B 纪律处理；prefill 主判
   −36.8%（远超 3% 噪声阈）决定 winner，swiglu 同向 −33~−36%。
3. **判据失配**：`aiv_vec_ratio=0.943 ≥ 0.9` → compute-bound，改用向量利用率判据（豁免规则 3）；
   达标硬底线是 **Task Duration 变短**（已满足：prefill −43.6%，swiglu −35%）。
4. **L2-stress（scale-20 both halves）非阻塞 WARN**：fp16/bf16 由 deg-20 的 pass 变 WARN——deg-10
   的 erf err 1.4e-4 在 gate/value 同时 scale-20 时被 0.5·|gate|·|value| 放大到超 fp16 atol；属设计
   §9.1 明言的「out of spec randn regime」（recorded, not blocking；deg-20 fp32 本身也 WARN）。

## 6. 天花板判定（[DESIGN_LIMIT] 触发条件核对）

- 条件①（设计层天花板）：deg-20→deg-10 是 DESIGN §5.4 显式列出的 Stage 4 实验项（「窄 dtype 走
  deg10 多项式」），**非**设计冻结的算法天花板——本次是执行既定优化，**不成立**。
- 条件②（结构性加速 >2x 或实测与假设矛盾）：基线→final 结构性加速 **~1.78×**（877.2→494.4），未达
  2×；DESIGN「带宽受限 226µs」与实测「compute-bound」的矛盾已由豁免规则 3 承接（判据失配改判向量
  利用率），且该矛盾不指向 >2× 结构性加速（精确 erf 的 fp32 多项式求值是契约要求，deg-10 已近该
  契约的 op 数下限）。**不成立**。
- **结论**：两项未同时满足 → **不产出 `perf_feedback.md`**，不附 `[DESIGN_LIMIT]` 信号。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| gelu_and_mul_fwd_kernel | llama-8b-ffn-prefill-float16 | 877.198 | 494.380 | final |
| gelu_and_mul_fwd_kernel | llama-8b-ffn-prefill-bfloat16 | 877.188 | 499.190 | final |
| gelu_and_mul_fwd_kernel | llama-8b-ffn-decode-bfloat16 | 3.860 | 3.000 | final |
| gelu_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-float16 | 14.980 | 9.670 | final |
| gelu_and_mul_fwd_kernel | qwen3-8b-swiglu-b32-bfloat16 | 14.810 | 9.900 | final |

## 8. Final Summary（逐 workload 收益，同一最终候选）

| workload_id | baseline_us | final_us | Δ% | SOL% |
|---|---:|---:|---:|---:|
| llama-8b-ffn-prefill-float16 | 877.198 | 494.380 | **−43.6%** | 46.86 |
| llama-8b-ffn-prefill-bfloat16 | 877.188 | 499.190 | **−43.1%** | 47.70 |
| llama-8b-ffn-decode-bfloat16 | 3.860 | 3.000 | −22.3%（微型豁免） | 3.67 |
| qwen3-8b-swiglu-b32-float16 | 14.980 | 9.670 | **−35.4%** | 32.88 |
| qwen3-8b-swiglu-b32-bfloat16 | 14.810 | 9.900 | **−33.2%** | 32.75 |

- 达标率：可调 shape（prefill fp16/bf16 + swiglu fp16/bf16）= **4/4 达标**（Task Duration 全部下降；
  SOL 60% 目标因 compute-bound 判据失配不适用，改判向量利用率）；全 shape（含 decode 微型）=
  4/4 可调 + 1 豁免 = 5/5（豁免不计入分母）。
- 停止原因：**success**（达到判据——Task Duration 显著下降；SOL 60% 目标因 compute-bound 判据失配不适用）。
- 精度：`python perf_opt/GeluAndMulFwdOp.py --level all` 全绿（L0/L1/L2/Boundary，含 erf 饱和 / 大负值 /
  signed-zero / inf/nan）；仅 L2-stress（scale-20 both halves）非阻塞 WARN。

## 9. Skill Retrospective

### 流程问题

1. **legacy perf_records 跨任务污染**：上一轮（CANN 8.5.2）留下的 4 条 `perf_records.jsonl` 使用旧
   schema（`op/label/dtype/baseline_us/perf_opt_us/...`），与本轮 signal-registry §5 字段契约不兼容，
   会机械触发 `S4-PERF-RECORDS-SCHEMA`。本轮将其移出主文件、归档到 `logs/legacy_perf_records_prior_run.jsonl`
   并留痕。建议（R 类）：`init` 重开 Stage 4 时应提示清理/归档旧 `perf_opt/` 记录，或在 gate 侧对「旧
   schema 行」显式 deprecate 而非硬失败。

### Value Point Proposals

| ID | vp_type | 内容 | 证据三件套（evidence / repro / toolchain） |
|---|---|---|---|
| VP-2026-0033 | P | **dtype 分化多项式阶数**：fp16/bf16 输出只需 ~1e-3 精度，用 deg-10 minimax 替代 deg-20（fp32 保持 deg-20），Horner 从 40 减到 20 条向量 op；compute-bound（vec 0.94）逐元素算子 prefill **−36.8%**（800→506µs）、对 Stage 3 基线 −42%。trace-time dtype 分支（非运行时条件路径）。 | opt_log §3 Round 1 表 / repro 待回填（最小形态 = dtype 分支 + 两组系数 + deg-N Horner 展开）/ tilelang 31f6903 + CANN 9.0.0 + 910B3 |
| VP-2026-0034 | D | **erf 精确饱和是硬约束（fold 反例）**：折叠 `1/√2` 进 erf 系数（系数 ×1/√2 + clamp ±4√2）破坏 `c0=1/4` 与 `t=±4` 的 fp32 精确性 → `t·c0` 非精确 ±1 → `gelu(-inf)` 变 `-inf`（非 NaN），L1-4/Boundary 全失败。精确饱和（c0=1/A 精确）不可妥协。 | opt_log §3 Round 2 fold 行 + Boundary 失败 / repro-missing（最小形态 = fold 变体 + gelu(-inf) 断言）/ 同工具链 |
| VP-2026-0035 | P | **loop-hoisted 常数 buffer + 单条 vsub**：`w=1−u²` 用循环外 `vbrc(1.0)` 的常数 buffer + 单条 `T.vsub(one_ub,w,w)`（替代 vmul-negate+vadd 两条），compute-bound 下 prefill −2.24%（A/B 3 对方向一致，子阈值 tie）。承 gelu_tanh_and_mul 的 hoist-vbrc 先例。 | opt_log §3 Round 2 vsub 行 + A/B JSON / repro 待回填 / 同工具链 |
| VP-2026-0036 | D | **GeluAndMulFwdOp compute-bound 实测常数**：prefill fp16 baseline 877.2µs（aiv_vec_ratio 0.879/0.943）、final 494.4µs；GM SOL 43–48%；deg-10 单点 −36.8%、vsub 单点 −2.2%。 | opt_log §1/§8 / repro-missing / 同工具链 |
| VP-2026-0037 | R | **DESIGN 带宽受限估算失准（D-2）**：DESIGN §1.6.0 以 MTE2 带宽估算 prefill 226µs 下界，实测 877µs（compute-bound vec 0.94），下界估算方向错误（把搬运当地板，实为向量 FP32 打满）。同 gelu_tanh_and_mul 的 VP-2026-0032。 | opt_log §1「设计估算 vs 实测」 / — / 同工具链 |

- 待回填 repro：VP-2026-0033/0034/0035 的最小形态（ED-B）；本轮时间受限，标 `repro-missing` 入队。
