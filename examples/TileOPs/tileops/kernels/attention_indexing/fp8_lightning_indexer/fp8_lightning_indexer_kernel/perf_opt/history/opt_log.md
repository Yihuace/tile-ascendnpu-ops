# Stage 4 性能调优日志 — FP8LightningIndexerOp (NPU / `target="npuir"`)

- project: `fp8_lightning_indexer`
- primary kernel: `_fp8_lightning_indexer_kernel` → `examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/perf_opt/_fp8_lightning_indexer_kernel.py`
- aux kernel: `clean_logits_` → `.../perf_opt/clean_logits_.py`
- mode: `full`；性能目标 `best_effort`
- 唯一时延口径: `msprof op` → `OpBasicInfo.csv` 的 `Task Duration(us)`
- 基准（只读）: `_fp8_lightning_indexer_kernel.py` / `clean_logits_.py`（Stage 3 产物，未修改）
- wrapper（未修改）: `.../fp8_lightning_indexer.py`（双 import 切换块由 conductor 翻转）

## 1. Workload Inventory（benchmark 展开，未写死）

来源：`examples/TileOPs/benchmarks/ops/bench_fp8_lightning_indexer.py::test_fp8_lightning_indexer_bench`
→ `tileops.manifest.load_workloads('FP8LightningIndexerOp')` → `_indexer_params()`（按 shape 去重）。

实际展开的 3 个参数化案例（`_indexer_params()` 只按 `_SHAPE_KEYS` 去重，**不含 dtype**）：

| # | workload_id | marks | kind | shape (b,s,h,d,skv,g) |
|---|---|---|---|---|
| 1 | `lightning-indexer-smoke-s1k-h8-d64` | smoke | smoke（仅精度） | 1,1024,8,64,1024,1 |
| 2 | `lightning-indexer-bf16-s8k-h32-d64` | full | tune | 1,8192,32,64,8192,1 |
| 3 | `lightning-indexer-bf16-s4k-h32-d64` | full | tune | 1,4096,32,64,8192,1 |
| – | `lightning-indexer-fp8-s8k-h32-d64` | (manifest full) | **skipped** | 与 #2 同 shape |

`fp8` 行处置（**未尝试真实 fp8，未伪造通过**）：

1. manifest 标注 `status: restricted`——`CG-2026-0022`（npuir 前端无 `T.float8_e4m3fn`、codegen 拒绝 `e4m3_float8`、`torch_npu` 设备端 fp8 cast 报 561103）；
2. 即便不考虑能力缺口，benchmark 参数化也会把它折叠掉：`_indexer_params()` 去重键为 shape（`batch,seq_len,heads,index_dim,seq_len_kv,kv_group,clean_logits`），fp8 与 bf16-s8k 完全同 shape，而 fp8 量化发生在 op 层（`per_custom_dims_cast_to_fp8`），故不构成独立测量；
3. 交付契约保持 **bf16 载体 + fp32 累加 + 可选 fp32 `index_k_scale`**。

调优顺序（按实测负担降序）：`bf16-s8k-h32-d64`（baseline 16.36 ms）→ `bf16-s4k-h32-d64`（8.28 ms）。
实际迭代主体在 s4k 上跑（单次采集快、同一 kernel/同一结构），winner 在 s8k 上复测确认。

## 2. Performance Test Data（baseline，phase=baseline, round=0）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | launch_count | profile_status | raw_profile_dir | command |
|---|---|---|---|---|---:|---:|---|---|---|
| `..._fp8_lightning_indexer_kernel.py::_fp8_lightning_indexer_kernel::_fp8_lightning_indexer_main` | lightning-indexer-bf16-s8k-h32-d64 | baseline | `_fp8_lightning_indexer_main` | `_fp8_lightning_indexer_main_mix_aic` | 16356.5 | 2 | valid | `profiles/baseline/s8k/baseline/OPPROF_20261018143521_BXWMISHFYNASDLSD` | `msprof op --kernel-name=_fp8_lightning_indexer_main --launch-count=2 --warm-up=1 --aic-metrics=Default python perf_opt/logs/baseline/s8k_baseline_inner.py` |
| 同上 | lightning-indexer-bf16-s4k-h32-d64 | baseline | 同上 | 同上 | 8281.6 | 5 | valid | `profiles/baseline/s4k/baseline/OPPROF_20261018112104_AYRSHAMJCBVOKDWK` | 同上（s4k inner） |
| `clean_logits_.py::clean_logits_::clean_logits_kernel` | clean-logits-s8k-g1 | baseline | `clean_logits_kernel` | `clean_logits_kernel` | 23527.1 | 1 | valid | `profiles/baseline/clean_s8k/baseline/OPPROF_20261018145302_XLYDEXZUWAOXFCEY` | `msprof op --kernel-name=clean_logits_kernel ...` |
| `clean_logits_.py::clean_logits_::clean_logits_kernel` | clean-logits-s4k-g1 | baseline | `clean_logits_kernel` | `clean_logits_kernel` | 11811.6 | 5 | valid | `profiles/baseline/clean_s4k/baseline/OPPROF_20261018133541_ERWPVCOVHLLRSEZJ` | 同上 |

baseline 配置 = wrapper `default_config`（`block_N=64, num_stages=2, block_Q=4`；aux `block_N=256, G_MAX=2`）。
baseline 复现值与迁移期官方 benchmark 记录一致：s8k 16356.5 vs 16413.5 us（−0.35%）、s4k 8281.6 vs 8293.9 us（−0.15%）、aux 与 manifest 一致。

**smoke 排除**：`lightning-indexer-smoke-s1k-h8-d64` 按 `pytest.mark.smoke` 归 smoke，不采性能 baseline、不选 winner，仅在每次合并与最终做精度回归（结果见 §5）。

**设计估算 vs 实测（D-2 回填）**：`DESIGN.md` §1.6.0 对代表 workload（bf16, block_Q=8, block_N=64）给出的**估算下界 ≈ 3.7 ms**（发射项主导，流量项 0.29–1.29 ms），并声明所用带宽/发射常数为 **9362 绑定、910B3 未实测**。
实测（同一 bf16 / block_Q=8 / block_N=64 结构）：**s8k 10.92 ms（下界的 2.95×）、s4k 5.33 ms（1.44×）**；baseline（block_Q=4）为 4.42× / 2.24×。
失准项为**发射/固定项**：估算按“每元素 vector op 成本”建模，未计入**每个 kv-tile 迭代的固定成本 ≈1.07 us/核**（§3 拟合），该固定项与元素数无关，只在迭代数下降时被摊薄。故偏差归因于「常数未在 910B3 重验 + 缺 per-iteration 固定项」，不构成设计层天花板结论（见 §6）。

## 3. Iteration Log

通用口径：`msprof op Task Duration(us)`，median of launches；候选差异 <5% 一律不作为 winner（本文所有采纳/否决差异均 ≥13%，无需 `ab_test.py` 交错协议）。
对比表列：Task Duration、AICore/Vector 利用率（`PipeUtilization.csv`：`aic_cube_time/aic_time`、`aiv_vec_time/aiv_time`、`aic_fixpipe_time/aic_time`、`aiv_scalar_time/aiv_time`）、memory 指标（`Memory.csv` 每 block 均值 `read/write_main_memory_datas`）、L0。
> memory 列说明：msprof 的 per-block 主存计数（含 K 重读与 L1 侧重复计入），只用于同口径横向比较，不作为绝对流量。

### Round 1（config-only，baseline 派生；base = baseline 8.28 ms）

现象：迭代数 = `(S/block_Q)*(Skv/block_N)` = 1024×128（s4k），每迭代实测 1.52 us/核；cube 仅 7.7% 忙、vector 44.8%、**aiv_scalar 42.9%**、fixpipe 24.7% ⇒ 单迭代固定成本主导，而非算力/带宽。

| 候选 | 主要优化点 | config | Task Duration (us) | Δ vs base | aic_cube% | aiv_vec% | aic_fixpipe% | aiv_scalar% | read/write KB/block | L0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | — | bQ4/bN64/st2 | 8281.6 | — | 7.7 | 44.8 | 24.7 | 42.9 | 3099 / 1424 | PASS |
| **r1a_bq8** | 迭代数减半（block_Q 4→8） | bQ8/bN64/st2 | **5377.8** | **−35.1%** | 10.5 | 62.1 | 29.8 | 73.7 | 5835 / 2832 | PASS |
| r1c_bn128 | 迭代数减半（block_N 64→128） | bQ4/bN128/st2 | 5408.0 | −34.7% | 10.3 | 51.8 | 29.5 | 53.9 | 5857 / 2835 | PASS |
| r1d_bn128s3 | r1c + 流水深度 | bQ4/bN128/st3 | 5399.1 | −34.8% | — | — | — | — | — | PASS |
| r1b_bq16 | 迭代数再减半 | bQ16/bN64/st2 | blocked-compile | — | — | — | — | — | — | — |
| r1e_bq8bn128s2 | 乘积 1024 | bQ8/bN128/st2 | blocked-compile | — | — | — | — | — | — | — |
| r1f_bq8bn128s1 | 乘积 1024，去流水 | bQ8/bN128/st1 | blocked-compile | — | — | — | — | — | — | — |

blocked 证据（原样引用 `logs/candidate/*.log`）：

- r1b: `ub overflow, requires 2148352 bits while 1572864 bits available`
- r1e/r1f: `ub overflow, requires 2142208 bits while 1572864 bits available`（st=1 同样越限 ⇒ 不是多缓冲导致）

winner：**r1a_bq8（5377.8 us）**。r1c/r1d 与 r1a 差异 +0.6%/+0.4%（<5% 平区，且 r1a 绝对最优），按平区规则不另立 winner；r1a 同时是 UV 压力更小的配置（K 侧 buffer 未变、L0C 32 KB）。

单迭代成本模型（同核归一，`T_iter = (duration*24)/(blocks*iters)`）：
`bQ4/bN64 → 1.52 us`、`bQ8/bN64 → 1.97 us`、`bQ4/bN128 → 2.03 us` ⇒ 拟合 `T_iter ≈ 0.56 + 0.1125*block_Q + 0.008*block_N` (us)，
即 **每个 kv-tile 迭代有 ≈0.56 us/核 的固定成本**（占 bQ4 迭代的 37%），模型复算 r1a 得 5378 us（实测 5377.8 us，吻合）。
⇒ 结构收益方向明确：在 UB 预算内最大化 `block_Q*block_N`。

合并非回退检查（合并 r1a 进同一 `@T.prim_func`，即只改默认 tiling）：s8k 10924.9 us（vs baseline 16356.5，−33.2%）、s4k 5334.9 us（vs 8281.6，−35.6%）；smoke `s1k-h8` L0 PASS。→ 更新 current best。

### Round 2（structural：head-chunked epilogue，S3；base = r1a 5.38 ms）

动机：`acc_ub = block_Q*heads*block_N*4` 是唯一 UB 瓶颈（bQ16/bN64 需 262 KB）。S3 把 Q 行改为 **chunk-major** 布局（`row(hg,q,i)=hg·bQ·hp_sub+q·hp_sub+i`），使每个 head chunk 是 L0C 的**连续行区间**，epilogue 逐 chunk 做 fixpipe + relu + vmul + reduce 再累加 ⇒ `acc_ub` 降为 `bQ*hp_sub*bN*4`。
前置：`probes/probe_chunk_unroll.py` 以 fp32 参考验证数值等价（`max_abs=1.5e-05`，PASS）后，才进真实 kernel（`--level L0` 3/3 PASS）。

| 候选 | 主要优化点 | config | Task Duration (us) | Δ vs r1a | aic_cube% | aiv_vec% | aic_fixpipe% | aiv_scalar% | L0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| current best | — | bQ8/bN64/st2 | 5377.8 | — | 10.5 | 62.1 | 29.8 | 73.7 | PASS |
| r2a_bq16hp8 | S3 解锁 bQ16 | bQ16/bN64/hp8 | 11152.2 | **+107%** | — | — | — | — | PASS |
| r2b_bq8bn128hp8 | S3 解锁 bN128 | bQ8/bN128/hp8 | 8935.7 | **+66%** | — | — | — | — | PASS |
| r2c_bq8bn128hp16 | S3，hp_sub=16 | bQ8/bN128/hp16 | 8078.0 | **+50%** | 6.4 | 49.1 | 16.5 | 52.9 | PASS |

结论：S3 **整体回退**，否决。机制（与 r1a 的 pipe 比值对照）：per-tile fixpipe 事务从 1 次变 `n_chunk` 次、并引入 `n_chunk` 组 cube↔vector 同步对与额外的 `vbrc(0)+vadd` 累加；`aic_fixpipe%` 由 29.8% 降到 16.5% 而 `aic_blk` 由 67.6 us 涨到 102.8 us ⇒ 省下的 UB 换来的是**每 tile 更多同步/搬运事务**，固定成本上升而非下降。
即“UB 容量”不是可通过拆分 epilogue 免费换取的资源——**该结构方向在本工具链上被实测否决**（记录为负向结论，不再重试）。

### Round 3（对齐/参数边界探测；base = r1a 5.38 ms）

| 候选 | 主要优化点 | config | Task Duration (us) | Δ vs r1a | aic_fixpipe% | aiv_vec% | L0 |
|---|---|---|---:|---:|---:|---:|---|
| current best | — | bQ8/bN64/st2 | 5377.8 | — | 29.8 | 62.1 | PASS |
| r3a_bq8bn80 | 非 2 幂 block_N（乘积 640） | bQ8/bN80/st2 | 13928.8 | +159% | **75.6** | 34.1 | PASS |
| r3c_bq8st3 | 深流水 | bQ8/bN64/st3 | 5611.6 | +4.3% | — | — | PASS |
| r3d_bq4bn160 | 乘积 640，bQ 回退 | bQ4/bN160/st2 | 8708.3 | +61.9% | — | — | PASS |
| r3b_bq8bn96 | 乘积 768 | bQ8/bN96/st2 | blocked-compile (`ub overflow, requires …`) | — | — | — | — |

结论：`block_N` 必须保持 2 的幂且与 `seq_len_kv` 整除对齐（bN=80 需要 103 个 tile 且尾块 clamp 重算，`aic_fixpipe` 冲到 75.6%）；`num_stages=3` 无收益（+4.3%，K tile 已在 L2/UB 常驻）。**r1a 保持 current best**。

### Round 4（收敛，无新候选）

Round 2/3 共 8 个 valid 分支、无一超过噪声阈值（3%），且结构性主候选（S3，两参数档位）已被实测否决，参照结构 diff 检查亦未发现新的形态差异（同族 `examples/` 其余 attention_indexing kernel 无更大 tile 的可复用形态）→ 触发 plateau 终止条件（连续 3 轮无 valid 分支提升 + 主结构候选已充分验证）。budget 使用：4 轮 / 12 个实验分支（上限 10/30）。

## 4. Autotune Log

未启用 autotune 框架（搜索空间仅 `block_Q×block_N×num_stages` 三维且 UB 可行性可解析枚举；r1/r3 的手动粗搜已覆盖可行域中的所有 2 幂组合：`(4|8)×(64|128)` 可行、`(16,64)`/`(8,128)`/`(8,96)` UB 越限）。

## 5. Final Summary

## Final Performance Test Data (primary, historical)

> This section is the historical primary-kernel record. This directory is now the
> dedicated `clean_logits_` optimize task (`statectl kernel_opt_py_path =
> perf_opt/clean_logits_.py`), so the gate-4 authoritative table for THIS task is
> the `## Final Performance Test Data` section at the end of §9. The primary
> numbers below are preserved unchanged.

primary kernel（同一最终候选 `perf_opt_final`，两条记录均取自 `perf_opt/_fp8_lightning_indexer_kernel.py`）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py::_fp8_lightning_indexer_kernel::_fp8_lightning_indexer_main | lightning-indexer-bf16-s8k-h32-d64 | 16356.540039 | 10924.878906 | perf_opt_final |
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py::_fp8_lightning_indexer_kernel::_fp8_lightning_indexer_main | lightning-indexer-bf16-s4k-h32-d64 | 8281.620117 | 5334.939941 | perf_opt_final |

- final 候选文件 = `perf_opt/_fp8_lightning_indexer_kernel.py`（sha256 `b71bf8e34b0c3f8fec6c488a223f2b6aea4b4e5e76ceeb8ceed874cdd6d65572`），两条 final 记录来自**同一文件、同一配置**（`block_N=64, block_Q=8, num_stages=2`），launch-count=3。
- 提升：s8k **1.497×**（−33.2%）、s4k **1.552×**（−35.6%）。
- raw profile 证据：`profiles/final/s8k/perf_opt_final/OPPROF_20261018143325_RJTLNLTHRWQMNBNA`、`profiles/final/s4k/perf_opt_final/OPPROF_20261018143754_RSITKWBBXCEMPUCE`。

### 5.2 Auxiliary kernel final data（aux `clean_logits_`，同一最终候选 `perf_opt_final`；不在 gate-4 对账范围，表列同形仅为可比）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/clean_logits_.py::clean_logits_::clean_logits_kernel | clean-logits-s8k-g1 | 23527.140625 | 23504.199219 | perf_opt_final |
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/clean_logits_.py::clean_logits_::clean_logits_kernel | clean-logits-s4k-g1 | 11811.559570 | 11780.899414 | perf_opt_final |

aux 无有效增益（`tuning_status=no_gain`）：`perf_opt/clean_logits_.py` 与 Stage 3 文件**逐字节相同**（sha256 `db9d614f…`），掩码契约未变。
实测事实（本 stage 新数据，留给后续结构改造）：aux 在 bench 的 full-window 案例上 **23.5 ms（s8k）/ 11.8 ms（s4k）**，比 primary kernel 本身还大；代价 ∝ block 数（4096→8192 block 恰好 2×）≈ **69 us/block**，与 `seq_len_kv` 无关；`block_N 256→512` 反而 **+81%**（21.4 ms，随 block_N 线性上升），`block_N=1024` 编译越限 ⇒ 热点是 `[block_N, kv_group=1]` 二维切片 copy 的**逐行标量路径**（内层 extent=1 无法向量化），而不是 tile 迭代数。可操作方向（本轮未实施、未验证）：把 full-window 之外的连续区间视为**一维连续块**（`tail*kv_group` 个 float 连续存放）单次 copy，并用标量闭式区间替代 `for t in T.serial(n_kv)` 内的逐 tile 谓词。

### 5.3 Smoke / skip

| workload | kind | 结果 |
|---|---|---|
| lightning-indexer-smoke-s1k-h8-d64 | smoke | 精度 PASS（最终文件，`logs/final_L0_primary.log`）；不采性能 |
| lightning-indexer-fp8-s8k-h32-d64 | skipped | `CG-2026-0022`（npuir 无 fp8 全链路）+ benchmark 按 shape 去重折叠；未尝试真实 fp8、未伪造通过 |

### 5.4 最终精度回归

- primary `python perf_opt/_fp8_lightning_indexer_kernel.py --level all`（`logs/final_Lall_primary.log`）：L0 3/3 PASS、L1 4/4 PASS（**含两个 tune workload 的 s4k/s8k 实际 shape**，corr_diff 5.662e-15）、L2 2 PASS + `L2-g4` WARN（非阻塞，见下）、boundary 3/3 PASS。
- aux `python perf_opt/clean_logits_.py --level L0`（`logs/final_L0_aux.log`）：L0 6/6 PASS（full/random/unaligned/empty/tail-odd/g2，corr_diff 0）。
- `L2-g4`（kv_group=4, heads=32）在 `block_Q=8` 下 UB 越限：该用例在本 harness 中一直是 WARN（Stage 3 文件的 `run_case` 默认同样是 `block_Q=8`，行为逐字节相同，非本次引入）；真实 wrapper 对 `kv_group>1` 走 `default_config` 的 `block_Q ≤ min(4, 4096/(heads*index_dim))` 上限（`fp8_lightning_indexer.py` K10 注释），故 drop-in 契约不受影响。**该上限在 wrapper 切换后必须继续生效**（已作为约束记录，见 §7）。

## 6. 设计层天花板判定（`[DESIGN_LIMIT]`）

**未产出 `perf_feedback.md`**。逐条核对 `_shared/standards/perf-feedback.md` §1：

1. 「天花板由算法/设计层决定、非 tiling/参数可解」——**不成立**：当前上限是 **UB 容量（`acc_ub = bQ*heads*bN*4`）** 与 per-iteration 固定成本，属实现/资源层；唯一尝试的结构替代（S3 head-chunked epilogue）**实测回退** 50–107%，没有验证出 >2× 的结构路径。
2. 「结构性加速估计 >2× 或实测与设计假设矛盾」——**部分成立**：设计估算下界 3.7 ms vs 实测 10.92 ms（s8k）为 2.95×，但归因为**估算常数（9362 绑定、910B3 未重验）+ 缺 per-iteration 固定项**，属 D-2 回填（§2 已记录），不是设计层天花板证据。

两项不同时满足 ⇒ 不产出 `perf_feedback.md`、不回传 `[DESIGN_LIMIT]`。roofline 侧遗留观察（供后续 stage 参考，不作结论）：bench 口径 roofline（`bytes = scores*4 + …`）给 s8k ≈153 us（1.75 TB/s），当前 10.92 ms 仍是 71×，说明结构上仍有量级空间，但本工具链上已试的两条结构路线均未兑现。

## 7. 工件与契约说明

- `perf_opt/perf_records.jsonl`：**仅 primary** 记录（13 行，append-only）。
- `perf_opt/perf_records_clean_logits.jsonl`：aux 记录（5 行）。
- `perf_opt/perf_records_all.audit.jsonl`：两者**逐行原样拼接**的不可变审计副本（18 行；`perf_records.jsonl` + `perf_records_clean_logits.jsonl` 的并集，**测量值未改**；sha256 `1d53fe84…`）。
- `perf_opt/workload_inventory.json` 只描述 primary 的 tune 集合（+ smoke/skip）；aux 在 `perf_opt/workload_inventory_aux.json`。
- 最终文件以模块级常量暴露 tuned 默认值：`TUNED_BLOCK_N=64`、`TUNED_BLOCK_Q=8`、`TUNED_NUM_STAGES=2`（factory 与 `run_case` 默认同步），供 wrapper 切换块成对引用；wrapper 未修改。
- 未修改 `tests/`、`benchmarks/`、`manifest/`、wrapper、基准 kernel、`DESIGN.md`。
- **采纳动作（conductor 侧，本 stage 不做）**：wrapper 的 `default_config` 对 `kv_group == 1` 仍硬编码 `block_Q = 4`，而 tuned 收益正来自 `block_Q = 8`（kernel 代码本身未变）。切换 import 时**必须**同步让 `default_config` 引用 `perf_opt._fp8_lightning_indexer_kernel` 的 `TUNED_BLOCK_N / TUNED_BLOCK_Q / TUNED_NUM_STAGES`；只翻转 import 而不改 config 将复现 baseline 时延。`kv_group > 1` 的 `block_Q` 上限（K10）保持不变。
- `profiles/` 保留每个采集的 `OpBasicInfo*.csv` 与 `PipeUtilization/Memory/…csv`（本文所有指标与 gate 证据均来自这些文件）；为控制仓库体积，已删除各 OPPROF 目录下的 `visualize_data.bin` 时间线二进制与 `dump/`（DeviceProf*.bin、aicore_binary.o），可由 `tune_runner.py` 同参数重采。

## 8. Skill Retrospective

### 8.1 流程问题

- **R（流程规则）**：`run_case` 型自测入口的默认 tiling 与「被测配置」必须一致，否则 L0 通过的是另一个配置。本轮开始即在候选文件中把 `run_case` 默认改成 `TUNED_*`，否则 S3 的 chunk 路径不会被 L0 覆盖（`hp_sub=8` 也不会被走到）。建议 skill 的 Phase 2 步骤 6 明确要求：**候选文件的 L0 必须跑该候选的 tuned 配置**，并在 opt_log 记录该配置。
- **R**：`msprof op` 单次采集的固定开销（编译 + profiling 会话）在 ≥8 ms 的 kernel 上约 40–120 s/次，`launch-count=5` 会放大到 8 min。建议 skill 的 profile-collection 明确「探索期 launch-count=1–2、baseline/final ≥3」并给出耗时预期，避免 runner 超时（本轮 aux s8k 一次 5-launch 采集直接超 30 min 被 kill）。
- **P（方法模式）**：需要**编译期 Python 展开**的 loop（如 head chunk 的 `hg`）在 npuir prim_func 内不可用：`for hg in [0,1]` → `Expect the for loop to be one of: range, T.serial, ...`；列表推导 `[... for _ in range(n)]` → `'ForFrame' object is not iterable`（kernel 体内的 `range` 是 TIR for-frame）。可用替代：`T.unroll(n)` + 单缓冲 + `vbrc(0)`/`vadd` 累加（本轮 `probes/probe_chunk_unroll.py` 验证可用且数值正确）。证据：`perf_opt/probes/probe_py_unroll.py`（报错原文）、`perf_opt/probes/probe_chunk_unroll.py`（PASS, max_abs=1.5e-05）。
- **D（实测数据）**：`T.view(buf, shape, dtype)` 在本工具链**不可用**：在 prim_func 内会触发 `variable <name> has already been defined, the Stmt is not SSA`（`VarUseDefAnalyzer::HandleDef`），即使把 view 定义提到循环外也一样 ⇒ 「零成本 rank-3 视图 + 单次 3D reduce」路线被阻断（对应 `for q in T.serial(bQ): T.reduce_sum(...)` 的逐 q 归约无法用视图消除）。证据：`perf_opt/logs/round2/s4a_L0.log` 报错原文；候选 `perf_opt/_fp8_lightning_indexer_kernel_round2_s4a.py`。

### 8.2 价值点提案（vp_type + 证据三件套）

1. **D-类型｜per-iteration 固定成本常数（910B2C，mix AIC+AIV 小 GEMM + 向量 epilogue）**
   - 内容：`T_iter ≈ 0.56 + 0.1125*block_Q + 0.008*block_N` us/核；对 `(bQ,bN) ∈ {(4,64),(8,64),(4,128)}` 三点标定，回算 winner 误差 <0.1%。工程含义：此类算子在 UB 允许范围内**应直接把乘积顶到 UB 上限**，而非按 roofline 只调 block_N。
   - evidence：本日志 §3 Round 1/3 表；实测 8281.6 / 5377.8 / 5408.0 us。
   - repro：`perf_opt/probes/probe_chunk_unroll.py`（同结构最小 kernel）+ `perf_opt/tune_runner.py`（配置扫描）。
   - toolchain_stamp：tilelang 0.1.2（repo HEAD 2026-10-18）/ CANN 9.0.0 / Ascend910B2C / torch_npu 2.7.1.post4。
2. **P-类型｜head-chunked epilogue 是负向模式（不要用拆分 epilogue 换 UB 余量）**
   - 内容：把 L0C→UB 的整块 epilogue 拆成 `n_chunk` 个子块（chunk-major Q 行布局）可使 `acc_ub` 缩小 2–4× 并解锁 bQ16/bN128，但实测 **+50%…+107%** 时延：per-tile fixpipe 事务数与 cube↔vector 同步对数随 `n_chunk` 线性增长。
   - evidence：r2a/r2b/r2c 11152.2 / 8935.7 / 8078.0 us vs current best 5377.8 us；`aic_fixpipe%` 29.8→16.5 但 `aic_blk` 67.6→102.8 us。
   - repro：`perf_opt/_fp8_lightning_indexer_kernel_round2_s3.py`（完整 before/after 差异：baseline 单块 epilogue vs chunk-major 两块）+ `probes/probe_chunk_unroll.py`。
   - toolchain_stamp：同上。
3. **D-类型｜aux `clean_logits_` 的真实代价与归因**
   - 内容：bench full-window 案例上 23.5 ms（s8k）/ 11.8 ms（s4k），∝ block 数（69 us/block），与 `seq_len_kv` 无关；`block_N` 增大反而线性变差（+81% @512）⇒ 热点是 `[block_N, kv_group=1]` 二维窄内维 copy 的逐行标量路径。
   - evidence：§5.2 表与 `profiles/{baseline,final}/clean_*`；aux_bn512 = 21428.6 us。
   - repro：`perf_opt/tune_runner.py --workload clean_s4k --factory clean_logits_`。
   - toolchain_stamp：同上。
4. **R-类型｜多 kernel 算子的 perf_records 分区约定可机械校验**
   - 内容：primary 记录进 `perf_records.jsonl`、aux 进 `perf_records_<aux>.jsonl`、并集以拼接副本 `perf_records_all.audit.jsonl` 留存（sha256 三份可交叉核对），可同时满足 gate-4 单 `kernel_opt_py_path` 对账与“aux 不被丢弃”两个要求。
   - evidence：§7；三份文件行数 13 / 5 / 18 与 sha256。
   - repro：`perf_opt/tune_runner.py`（`--records` 指定分区文件）。
   - toolchain_stamp：不依赖工具链。

### 8.3 未解决 / 留给后续

- primary 仍受 UB 容量限制在 `bQ*bN ≤ 512`（2 幂对齐），实测平台 5.33 ms（s4k）；设计估算下界 3.7 ms 的偏差需在 910B3/B2C 上重验发射常数（`CONST-vector-op-cost-910_9362` 等为 9362 绑定）。
- aux `clean_logits_`（11.8/23.5 ms）已超过 primary，是当前 op 端到端的主导项，但本轮未实施结构改造（§5.2 已给出方向与判据）。
- `L2-g4` 的 UB 越限需在 wrapper 侧保持 `kv_group>1` 的 `block_Q` 上限（K10 逻辑），否则 grouped 路径会编译失败。

---

# 9. `clean_logits_` 专项 Stage 4 调优（本次任务）

本目录自本任务起把 **`clean_logits_` 作为 primary**：`statectl kernel_opt_py_path =
perf_opt/clean_logits_.py`。§1–§8 为 `_fp8_lightning_indexer_kernel` 的历史调优记录，
原样保留（其上方的 Final 表已标注为 historical）。

## 9.1 口径修正：上一轮的 aux baseline 不代表 benchmark 负载

§5.2 的 aux baseline 用 **full window**（`ks=0, ke=seq_len_kv`）采集，此时补集为空、
几乎不写任何字节，11.8 / 23.5 ms 反映的是**空转开销**。本轮按 benchmark 真实窗口分布重测：

- 窗口分布 = `FP8LightningIndexerWorkload.generate_random_cu_seqlens(cp_size=1, cp_rank=0,
  kv_stride=1, average_q_len=512)`（`tune_runner.py` 内内置同构实现，探针实测可见比例
  **4.00%（s4k）/ 4.10%（s8k）**，即约 96% 元素落在窗口外、必须写 `-inf`）。
- Stage 3 kernel 在该分布下实测 **55 453.6 µs（s4k）/ 110 788.7 µs（s8k）**，是 primary kernel
  （10.92 ms）的 5.1× / 10.1× ⇒ **aux 确为本 op 的主导项**（原判断成立，但数值被低估 4.7×）。
- **归因修正**：§5.2 记为「逐行标量 copy」。本轮 `PipeUtilization` 实测该分支
  `aiv_mte3=0.840 / aiv_scalar=0.078 / aiv_vec=0.050` ⇒ 瓶颈是 **MTE3 事务数**（`G=1` 时每行
  4 B burst），不是标量 ALU。方向结论（窄内维 → 连续块）成立，机制归因更正为 **DMA burst 粒度**。

## 9.2 Workload Inventory（benchmark 展开）

见 `perf_opt/workload_inventory.json`（已更新为 `clean_logits_` 的 tune 集合）。`Logits` shape =
manifest 案例的 `(batch, seq_len, seq_len_kv, kv_group)` fp32：

| # | workload_id | marks | kind | Logits shape (b,s,skv,g) | baseline (us) |
|---|---|---|---|---|---|
| 1 | `clean-logits-smoke-s1k-g1` | smoke | smoke（仅精度） | 1,1024,1024,1 | 不采性能 |
| 2 | `clean-logits-s4k-g1` | full | tune | 1,4096,8192,1 | 55 453.6 |
| 3 | `clean-logits-s8k-g1` | full | tune | 1,8192,8192,1 | 110 788.7 |

smoke 案例在 bench 中同样以 `clean_logits=True` 调用本 kernel，按 `pytest.mark.smoke` 只做精度回归
（`perf_opt/clean_logits_.py --level all` 的 L0-1/L0-2 覆盖该 shape）。

## 9.3 Round 1（structural：flat complement；base = Stage 3，s4k 55.45 ms）

**现象**：Stage 3 的 `T.copy` 区域是 `[block_N, kv_group]`，`kv_group=1` 时最内层 extent=1
⇒ 每行一次 4 B 的 DMA burst，共 `S·Skv` 次事务；`mte3` 长期占用 84%。tile 数/`block_N`
只改变事务打包方式，不改变「每行一个 burst」这个事实（这正是 §5.2 观测到的「`block_N` 增大反而更慢」）。

**候选优化点**：`Logits` 在 `[B,S,Skv,G]` 上连续，故一行内 keep-window 的补集恰好是**两段连续区间**
`[0, ks·G)` 与 `[ke·G, Skv·G)` ⇒ 用 **rank 保持的扁平视图** `[B, S, 1, Skv·G]` 把两段做 bulk 1-D
常量 store，无需 mask、无需回读、无需 tile 分类（E1/E2/E3 的闭式等价）。每个分支只改一个主要点：

| 候选 | 主要优化点 | 配置 | Task Duration (us) | Δ vs base | aiv_vec% | aiv_mte3% | aiv_scalar% | L0 |
|---|---|---|---:|---:|---:|---:|---:|---|
| Stage 3 baseline | — | bN=256 tile 扫描 | 55 453.6 | — | 5.0 | 84.0 | 7.8 | PASS |
| r1a_flat_rows1 | 扁平连续补集（1 行/块） | chunk=8192, rows=1 | 177.4 | **−99.68%** | 9.3 | 56.1 | 39.4 | PASS |
| r1b_rows4 | 每块 4 行（摊薄块级固定成本） | chunk=8192, rows=4 | 63.2 | **−99.89%** | 3.8 | 84.5 | 20.7 | PASS |
| **r1c_rows8（current best）** | 每块 8 行 | chunk=8192, rows=8 | **60.4** | **−99.89%** | 2.1 | 90.8 | 15.3 | PASS |
| r1d_rows16 | 每块 16 行（更少块） | chunk=8192, rows=16 | 61.6 | −99.89% | 1.1 | 93.8 | 13.3 | PASS |

- r1d 的 `aiv_mte3` 更高（93.8%）但墙钟更差（+2.1%，> 噪声但方向相反）：块数降到 256 后
  尾部/波次不均衡抵消了块级摊薄收益 ⇒ 取 **rows=8**（s4k 1024 块，s8k 2048 块，均 ≫ 核数）。
- `chunk=8192`（32 KB UB）使一段在 main workload 上通常一次 DMA 完成；再大无收益（段长 ≤ 8192 元素）。
- **`block_N=64` 的旧缺陷消失**：新实现没有 mask/v-op 路径，`Boundary-blockN64` 由 Stage 3 的
  mis-compute 变为 **PASS**（advisory 非阻塞用例，实测 3 seed 全对）。

**s8k 确认（同一最终候选文件、同一配置）**：

| 候选 | Task Duration (us) | Δ vs base | aiv_vec% | aiv_mte3% | aiv_scalar% | L0 |
|---|---:|---:|---:|---:|---:|---|
| Stage 3 baseline | 110 788.7 | — | 5.0 | 83.9 | 7.8 | PASS |
| **current best（rows=8）** | **138.1** | **−99.88%** | 1.8 | 92.4 | 13.3 | PASS |

**旧口径交叉核对**（不进入 records，仅证明归因）：新 kernel 在 full-window 协议下 = **46.6 µs**
（`profiles/candidate/clean_s8k_full/aux_fullwin/OPPROF_20261019132124_WUPTUHFRYIVBTFUS`，`mte3=0.001`、
`scalar=0.914`），而 Stage 3 在同一协议下是 23 504 µs ⇒ 旧 23.5 ms 中 **99.8% 是 tile 循环 + 边界窄拷贝开销**。

**达成的写带宽**：s4k 写 ≈128.8 MB / 60.2 µs ≈ 2.14 TB/s、s8k ≈257.5 MB / 138.1 µs ≈ 1.86 TB/s
（高于 HBM 标称值，因 `Logits` 134/268 MB 与 L2 有重叠且多次复跑；`mte3` 已占 90–92%，
剩余空间受 DMA 事务与波次尾部支配，不再追）。

## 9.4 记录分区与「并集 = 审计副本」校验

| 文件 | 行数 | 内容 |
|---|---:|---|
| `perf_records.jsonl` | 4 | **仅 `clean_logits_`**：baseline_realwin ×2（phase=`baseline`）+ perf_opt_final ×2（phase=`final`，`artifact_path=clean_logits_.py`，sha256 `5264bce3…`） |
| `perf_records_primary.jsonl` | 13 | 原 `perf_records.jsonl` 的 primary 记录，逐字节复制，数值未改 |
| `perf_records_clean_logits.jsonl` | 5 | 上一轮 aux 记录（full-window 协议 + bn512 分支 + 2 条旧 final），原样保留 |
| `perf_records_all.audit.jsonl` | 22 | **不可变并集**：原 18 行（sha256 `1d53fe84…`）逐字节保留 + 本轮 4 行追加；校验 `multiset(4+13+5) == multiset(22)` → **True**，sha256 `79d3b3f0…` |

- gate-4 要求「每个 tune workload 恰有一条 final 且与当前 `perf_opt/clean_logits_.py` 哈希一致」，
  故上一轮 2 条旧 final（旧文件哈希、full-window 协议）不进入 `perf_records.jsonl`，只留在
  `perf_records_clean_logits.jsonl` 与审计副本中（append-only 纪律：未删除、未改写任何数值）。
- **provenance 披露**：本轮 final 采集复用 `tune_runner.py` 的 `profiles/final/clean_*/perf_opt_final/` 目录，按 runner 既有行为先 `rm -rf` 再重建，故上一轮 2 条旧 final 记录所引用的 raw OPPROF 目录已被本轮新目录替换（记录中的 `msprof_raw_path` 字符串未改、测量值未改；旧 raw 数据不可恢复）。旧 baseline 目录（`profiles/baseline/clean_*/baseline/`）与本轮新增的 `baseline_realwin/` 均完整保留。
- `Final Performance Test Data`（本文件末尾）的两条 final 与 `perf_records.jsonl` 的 2 条 final
  逐值一致（60.200001 / 138.139999，1% 容差内）。

## 9.5 契约与实现偏差

- **语义**：`[ks, ke)` 内逐位保留原值、窗口外写精确 fp32 `-inf`；golden 仍是逐元素 `torch.where`
  独立实现。L0 10/10、L1 13/13、L2 3/3、boundary 7/7 全 PASS（`corr_diff=0`、`max_abs=0`），
  含 g=2、奇数 `seq_len_kv`（1000/777/900/100）、越界/空窗/全窗、NaN/±inf/次正规。
- **偏差 1（DESIGN §0.6 R2）**：设计禁止用 GM 派生标量作 DMA 偏移（同族任务里 min/max 累加链
  mis-compile）。本实现用 `cu_k_s·G` / `cu_k_e·G`（单次 clamp + 一次乘法）作 store 偏移，
  由上述全量 L0/L1/L2/boundary 逐位一致验证通过，记录为实测偏差。
- **偏差 2（视图）**：`T.view` 到**不同 rank** 的形状在 `CreateRankReducedSubviewFromBaseRank`
  rank 校验处失败（1-D / 2-D 视图均不可用）；**rank 保持**的 `[B,S,1,Skv·G]` 视图可用。
- **偏差 3（参数语义）**：`block_N` 现表示扁平 chunk 长度（`(kv,G)` 折叠轴上的元素数，须 8 的倍数），
  不再是 kv tile 长度；`G_MAX` 仅为工厂签名兼容保留、实现已不使用（无 `[block_N, G_MAX]` UB tile）。
  工厂签名 `clean_logits_(block_N, G_MAX)` 仍可无参调用，wrapper 未改。
- **未改动**：`tests/`、`benchmarks/`、`manifest/`、wrapper、基准（Stage 3）kernel、`DESIGN.md`、
  `REVIEW.md`；所有产物只写 `perf_opt/`。

## 9.6 设计层天花板判定（`[DESIGN_LIMIT]`）

**不产出 `perf_feedback.md`、不回传 `[DESIGN_LIMIT]`**。逐条核对 `_shared/standards/perf-feedback.md` §1：

1. 「天花板由算法/设计层决定、非 tiling/参数可解」——**不成立**：Stage 3 的 55/110 ms 不是设计层
   上限，而是**核内实现形态**（2-D 窄内维 copy）导致的 DMA burst 退化；本轮未改算法、未改设计，
   仅改核内搬运形态即落到设计估算量级。
2. 「结构性加速估计 >2× 或实测与设计假设矛盾」——**不构成设计层证据**：DESIGN §R3（D-2）对代表
   shape（s8k）的估算下界 = **≈250 µs**（流量项 268 MB @ 1.06 TB/s，常数绑定 9362、910B3 未重验）。
   本轮 final **138 µs** 已**优于**该估算（910B3 实际写带宽与 L2 驻留），即设计假设在修复后
   **被满足而非被推翻**；旧实现的 110.8 ms 与该估算的 443× 差异归因于实现缺陷，不是设计模型错误。

两项不同时满足 ⇒ 无 `perf_feedback.md`。遗留观察（供 conductor/后续参考，非结论）：DESIGN §R3 的
流量模型隐含「补集可按连续区间打包」，但 §3.3 的 tile 扫描伪代码在 `kv_group=1` 时把最内维压到
1 个元素；该「模型↔伪代码」的粒度不一致值得作为 D-类型价值点回灌，而非设计返工。

## 9.7 Round 1 收敛判定

- 单轮即达 `success`（`best_effort` 目标下实测把主导项从 55/110 ms 降到 60/138 µs，且
  `mte3` 已 90–92%、写带宽超 DESIGN 估算），候选分支 5 个（4 valid + 1 基线对照重测），
  budget 使用 1 轮 / 5 分支（上限 10 / 30）；无 blocked、无精度回退。
- 停止原因：**success**。后续可探索空间仅剩 DMA 波次尾部（<10%），不构成新的结构候选。

## Final Performance Test Data

`clean_logits_`（gate-4 权威表；同一最终候选 `perf_opt_final`，两条记录均取自 `perf_opt/clean_logits_.py`）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/clean_logits_.py::clean_logits_::clean_logits_kernel | clean-logits-s4k-g1 | 55453.578125 | 60.200001 | perf_opt_final |
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/clean_logits_.py::clean_logits_::clean_logits_kernel | clean-logits-s8k-g1 | 110788.703125 | 138.139999 | perf_opt_final |

- 提升：s4k **921.2×**（−99.891%）、s8k **802.0×**（−99.875%）。
- final 文件 sha256 `5264bce3d27c849b88ade85a600154637bdd48b721d17dfb1f93131555db4824`，两条记录
  同一文件、同一配置（`block_N=8192, G_MAX=2, rows_per_block=8`），launch-count=3（min/max
  s4k 59.94/60.42、s8k 136.92/138.52 µs）。
- raw profile：`profiles/final/clean_s4k/perf_opt_final/OPPROF_20261019131935_FARUCUPQHKURTSBK`、
  `profiles/final/clean_s8k/perf_opt_final/OPPROF_20261019132005_RVJNFRGWJVGMYFPW`。

### 9.8 最终精度回归与 smoke

| 项 | 结果 |
|---|---|
| `python perf_opt/clean_logits_.py --level all` | L0 10/10、L1 13/13、L2 3/3、boundary 7/7 **全 PASS**（`logs/round1/final_candidate_Lall.log`），其中 L0-9/L0-10 即两个 tune workload 的真实 shape+因果窗口 |
| smoke `clean-logits-smoke-s1k-g1` | PASS（L0-1 full、L0-2 random，`corr_diff=0`）；不采性能 |
| `block_N=64` 已知工具链缺陷 | 本轮 **PASS**（无 mask 路径），advisory 用例保留 |

### 9.9 Skill Retrospective 补充（本次任务）

- **P（方法模式）｜rank 保持视图是 GM 侧扁平化的可用路径**：npuir 上 `T.view` 改 rank 会在
  `CreateRankReducedSubviewFromBaseRank`（`codegen_npuir_dev.cc:1641`）报
  `fullOffsets.size() == baseTy.getRank()`；保留 rank、把末尾两维折叠成一维
  （`[B,S,Skv,G] → [B,S,1,Skv·G]`）即可让 `T.copy` 以整段 burst 落下。证据：
  `perf_opt/probes/probe_flat_fill.py`（f1/f2 编译失败原文，f5 PASS）；收益 921×/802×。
- **P｜`T.macro` 是 prim_func 内复用带循环片段的唯一可用形式**：普通 Python 辅助函数里写
  `for i in T.serial(...)` 会报 `'ForFrame' object is not iterable`（parser 在调用点执行了它）。
  证据同上探针的 f1 首轮报错与修正。
- **R（流程规则）｜baseline 必须用 benchmark 真实输入分布**：本任务发现上一轮 aux baseline 用的是
  full window（补集为空），把 55.45 ms 的负载测成 11.8 ms。建议 skill Phase 1 明确要求
  「tune workload 的输入分布取自 workload 的 `gen_inputs`/窗口生成器，不得用退化输入代替」，
  否则 baseline 与 winner 判定同时失真。
- **D（实测数据）｜910B3 上「窄内维 2-D DMA」的代价常数**：`G=1`、`block_N=256` 时每行 4 B burst，
  s4k 55.45 ms / 4096 行 ≈ **13.5 µs/行**（32 KB 行数据），而扁平连续 store 为 **0.0147 µs/行**
  （60.2 µs / 4096），差 **~920×**。工程含义：`kv_group` 是最内维且为 1 时，任何
  `[tile, G]` 形状的 `T.copy` 都应先折叠成连续段。
- **D｜per-block 固定成本与 rows_per_block**：s4k 单行/块 177 µs → 4 行/块 63.2 µs → 8 行/块 60.4 µs
  → 16 行/块 61.6 µs（块数 4096/1024/512/256）；块级固定成本摊薄在 8 行/块饱和（`mte3` 0.91→0.94
  但尾部不均衡反超）。repro：`perf_opt/tune_runner.py --config '{"rows_per_block": N}'`。

### 9.10 Pattern-library 回写（Phase 4 ED-C）

本轮新增两条知识域条目（含 repro，实跑 `REPRO PASS`）：

- `TRAP-gm-view-rank-mismatch`（`pattern-library/traps-runtime.md`）——GM 参数 `T.view` 必须保持 rank；
  rank 保持的尾轴折叠是窄内维 DMA 连续化正解。repro：`pattern-library/repro/TRAP-gm-view-rank-mismatch.py`。
- `CONST-narrow-inner-dma-910B3`（`pattern-library/constants.md`）——最内 extent=1 的 2-D DMA ≈13.5 µs/32 KB 行
  vs 连续 1-D burst ≈0.0147 µs/行（≈920×）；`rows_per_block` 摊薄曲线 177.4→63.2→60.4→61.6 µs。
  共用上述 repro 的 [2]（快形态实测）。
- `INDEX.md` §4 两个主题表已登记条目行；交叉引用 `TRAP-scalar-min-chain-offset`（单次 clamp 作 unaligned
  偏移可用，勿把该 trap 扩大到「任何 min/max」）与 `TRAP-ub-buffer-alias-unavailable`（UB 侧结论不外推到 GM 参数）。
