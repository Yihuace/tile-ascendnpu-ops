# Stage 4 性能调优日志 — `_sparse_mla_kernel` (deepseek_dsa_decode)

- 算子（集成 manifest）: `DeepSeekSparseAttentionDecodeWithKVCacheFwdOp`（family: attention）
- kernel 文件（基准，只读）: `examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py`
- 设计参考: `_sparse_mla_kernel_DESIGN.md`（含 conductor 追加的「性能目标」章节）
- 最终候选: `perf_opt/_sparse_mla_kernel.py`（= Stage 3 基准，仅加注释块；默认参数 `block_heads=32, block_N=64, num_stages=2` 不变）
- 唯一时延口径: `msprof op` → `main_mix_aic` 的 `Task Duration(us)`
- 目标 kernel 名: `main_mix_aic`（`Op Type: mix`，`Block Dim 512/4096`，`Mix Block Dim 1024/8192`）
- 平台: Ascend 910B3（npu-smi: 910B3，chip 6，20 AI cores，1800 MHz）
- 工具链戳: tilelang `0.1.2+5a90b9918747fe9ed4870e6cd5c39760c68c33e8` / CANN 9.0.0（`msprof` 25.5.0）/ torch_npu 2.7.1.post4 / 2026-10-18
- 性能目标: `best_effort`（用户未指定数值目标）；关注 shape: 未指定

**迁移期工具链与本轮不同**（迁移期 `0.1.2+00aeb862` / 910B3 aicore=20）。因此本轮对 BH=64、gather lowering 等结论均在本工具链上重新实测/复核，不直接引用旧结论。

---

## 1. Workload Inventory（由迁移后 benchmark 展开，不写死）

来源 `examples/TileOPs/benchmarks/ops/bench_deepseek_dsa_decode.py::test_dsa_decode_bench`
（`_manifest_params(load_workloads("DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"))`，
manifest `examples/TileOPs/tileops/manifest/attention.yaml` L161-168）。展开得 4 个 workload / 5 个
pytest 参数化案例（smoke 覆盖 f16 + bf16）。机器可读清单见 `perf_opt/workload_inventory.json`。

| workload_id | q_shape | kv_shape | topk | dtype | kind | 说明 |
|---|---|---|---|---|---|---|
| dsa-decode-smoke-float16 | [1,128,128,576] | [1,256,1,576] | 256 | float16 | smoke | label 含独立 `smoke` token → 只做精度回归 |
| dsa-decode-smoke-bfloat16 | [1,128,128,576] | [1,256,1,576] | 256 | bfloat16 | smoke | 同上（bf16 路径） |
| single-batch-mainstream | [1,1024,128,576] | [1,2048,1,576] | 2048 | float16 | tune | 实测负担最大 → 排序第 1 |
| longer-kv-lower-topk | [1,512,128,576] | [1,4096,1,576] | 1024 | float16 | tune | 第 2 |
| long-kv-low-topk | [1,256,128,576] | [1,8192,1,576] | 512 | float16 | tune | 最小（35.6 ms）→ 存活候选的低成本 A/B 目标 |

- 所有案例实际调用同一 kernel：`SparseMlaKernel` → `_sparse_mla_kernel` → 唯一 `@T.prim_func::main`（单 kernel，无多 kernel 记录拆分）。
- 无 `bench_skip_reason`，故无 `skipped` 项。`smoke` 项**不采集性能 baseline、不选 winner**（本轮全部性能数据均来自 tune workload；下文的冒烟 shape 数据仅作为**诊断探针**，明确标注、不参与 winner 判定）。

**排序与理由**：mainstream（566.5 ms，4096 blocks × 32 次 gather 迭代）> longer-kv（141.8 ms）> long-kv（35.6 ms）。排序依据是实测 Task Duration 与结构收益机会；实际迭代中因三者的**每行 gather 成本完全一致**（见 §3.1），结论直接对全部 tune workload 成立，未跳过任何一项。

---

## 2. Performance Test Data（`phase=baseline`，msprof op，串行采集）

命令模板（每个 workload 一条，实际候选文件与 workload 均记入 `command` 列；`logs/` 下保留 stdout）：

```
msprof op --output=<perf_opt/profiles/...> --launch-count=<N> --warm-up=<W> \
          --aic-metrics=Default python perf_opt/bench_runner.py \
          --file <kernel.py> --case <workload> --repeat <N> --warmup <W>
```

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | launches | profile_status | raw_profile_dir |
|---|---|---|---|---|---:|---:|---|---|
| `.../_sparse_mla_kernel.py::_sparse_mla_kernel::main` | single-batch-mainstream | baseline | main_mix_aic | main_mix_aic | 566545.37 | 8 | valid | `perf_opt/profiles/base_mainstream/` |
| 同上 | longer-kv-lower-topk | baseline | main_mix_aic | main_mix_aic | 141754.06 | 5 | valid | `perf_opt/profiles/baseline_longer-kv/` |
| 同上 | long-kv-low-topk | baseline | main_mix_aic | main_mix_aic | 35598.48 | 5 | valid | `perf_opt/profiles/baseline_long-kv/` |

（完整 kernel_id 前缀 `examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel`；
表中为便于阅读做了省略，`perf_records.jsonl` / `workload_inventory.json` / Final 表中为全路径形态。）

profile 有效性：`captured_op_name == main_mix_aic`、`Op Type: mix`、`Block Dim` 与逻辑核数一致
（512/2048/4096 分别对应 smoke / longer-kv / long-kv / mainstream），非 Cast/Fill 等框架小算子。

### 2.1 设计估算 vs 实测（D-2 回填）

| 项 | DESIGN 值（§1.6.0 R4，mainstream） | 实测 baseline | 偏差 |
|---|---|---|---|
| Task Duration | 4.0 – 9.4 ms（gather 流量 9.66 GB / HBM 带宽项主导） | **566.5 ms** | **60× – 141× 失准** |
| 失准项 | — | **gather 的「流量→时间」假设**：per-row 动态索引 `T.copy` 并不按带宽流式搬运，实测 **≈1.3 µs/行**（≈1.5 GB/s 每核），整核时间 94% 消耗于此 | 流量/发射口径均失准（见 §3.1） |

### 2.2 smoke / skip 排除与精度

- `dsa-decode-smoke-float16` / `-bfloat16`：按规则排除性能调优，只做精度回归 —— 最终 `--level all` 中 L0-1（f16）/ L0-2（bf16）/ L2-1（bf16）全 PASS。
- 无 skipped 案例。

---

## 3. Iteration Log

### 3.0 初始现象（baseline，smoke shape 诊断 profile）

`perf_opt/profiles/probe/`（smoke 512 blocks × 4 迭代）：

| 指标（每 block，msprof PipeUtilization / Memory） | 值 | 解读 |
|---|---|---|
| `aic_time` | 361.9 µs | block 墙钟 |
| `aic_cube_time` | 1.64 µs（ratio 0.00018） | Cube 几乎全程空闲 |
| `aiv_vec_time` | 5.56 µs（ratio 0.00062） | Vector 计算几乎可忽略 |
| `aiv_scalar_time` / `aiv_mte2_time` | 166.2 µs / 172.4 µs | 标量 + 小 DMA 占满 AIV |
| `aic_scalar_wait_id1_time` / `aiv_scalar_wait_id2_time` | 365.3 µs / 240.3 µs | 两侧几乎全程等 flag（CV 相位串行） |
| `read_main_memory_datas` / `GM_to_L1_datas` | 583.5 KB / 596 KB | 每 block 搬运量远小于计算能力上限 |

**判据**：`cube_ratio ≈ 1.8e-4`、标量/mte2 占满 → 不是算力、不是 HBM 带宽，而是 **per-row gather 的发射/延迟**（=「诊断信号强制换轴分析」的触发条件：标量占比 >50%）。

### Round 1 — gather lowering 形态与 tiling 参数（12 个分支中的 8 个）

固定 base = baseline（`perf_opt/_sparse_mla_kernel.py`，`BH=32 / BN=64 / ns=2`）。
基准值（smoke shape 诊断探针，`perf_opt/profiles/probe`）= **9022.3 µs**。

| candidate_id | 唯一主要优化点 | Task Duration (µs) | vs current best | L0 | 结论 |
|---|---|---:|---:|---|---|
| baseline（=current best） | — | 9022.3 | — | PASS | base |
| **diag_contig**（诊断，非候选） | 把 64 次逐行 gather 换成 1 次 64 行连续 `T.copy`（**数学无效，仅测时**） | **499.2** | **−94.5%** | n/a（结果错误，不参与判定） | **定位：gather 占 94.5%** |
| c_2dslice（`KV[b_i, idx:idx+1, 0, :]`） | gather 用 2D 显式切片 | 8959.8 | −0.7% | not run（平区） | 无收益（=噪声） |
| c_basesize（`T.copy(src_base, dst_base, size=[1,Dqk])`） | gather 用 base+size 形态 | 9068.5 | +0.5% | not run（平区） | 无收益 |
| c_flatkv（KV 声明为 2D `(B*S_kv,Dqk)`） | 源张量降秩 | 9006.0 | −0.2% | not run（平区） | 无收益 |
| c_pipeinner（`T.Pipelined(block_N, ns=4)`） | 行循环改流水 | 9008.8 | −0.2% | not run（平区） | 无收益 |
| c_unroll8（8 行手工展开/迭代） | 减少行循环迭代数 | 9103.0 | +0.9% | not run（平区） | 无收益 |
| c_parallel（`T.Parallel(block_N)`） | 行循环改并行 | 9051.1 | +0.3% | not run（平区） | 无收益 |
| c_kvl1（KV_shared → `T.alloc_L1`） | 目的地换 L1 | 9123.9 | +1.1% | not run（平区） | 无收益 |
| c_bn32（`block_N=32`） | tile 缩小 | 9548.4 | +5.8% | PASS（boundary-bn32） | 回退 → rollback |
| c_bn16（`block_N=16`） | tile 更小 | 10598.6 | +17.5% | not run | 回退 → rollback |

> 上表「vs current best」为 smoke shape 诊断探针的相对值；因全部落在噪声外但仍无收益，只有 `c_bn32` / `c_kvl1` 被带到真正的 tune workload 复核（见下）。

**bounded 复核（真实 tune workload `long-kv-low-topk`，baseline = 35598.5 µs，各 5 launches）**：

| candidate_id | Task Duration (µs) | vs current best | L0 | 结论 |
|---|---:|---:|---|---|
| baseline（current best） | 35598.5 | — | PASS | base |
| c_bn32 | 37974.0 | **+6.7%** | PASS | rollback |
| c_kvl1 | 35713.6 | +0.3%（<3% 噪声阈） | PASS | tie，不采纳（代码复杂度无回报） |

**本轮对比表（候选 vs current best；B2 结构化回流）**

| 候选 | Task Duration (µs) | AICore 利用率（msprof `aic_cube_ratio` / `aiv_vec_ratio`，每 block） | memory（`read_main_memory_datas` KB/block） | L0 |
|---|---:|---|---|---|
| current best（baseline） | 9022.3（smoke）/ 35598.5（long-kv） | 0.000182 / 0.000617 | 583.5 | PASS |
| diag_contig（诊断） | 499.2 | **0.003294 / 0.010891（≈18×）** | 306.1 | n/a（数学无效） |
| c_bn32 | 9548.4 / 37974.0 | 0.000177 / 0.001013 | 754.1 | PASS |
| c_bn16 | 10598.6 | 0.000222 / 0.001694 | 1124.1 | not run |
| c_kvl1 | 9123.9 / 35713.6 | 0.000180 / 0.000610 | 583.5 | not run |
| c_pipeinner / c_unroll8 / c_parallel / c_flatkv / c_2dslice / c_basesize | 8959.8 – 9123.9 | 0.00018 / 0.00061（同档） | 583.5 | not run |

现象与结论：`diag_contig` 把利用率抬高 18× 并把时延打掉 94.5%，证明瓶颈 100% 在「逐行动态索引 gather」这一形态；而所有**保持语义**的 gather 变体（切片形态、降秩、并行/流水/展开、目的内存域）都**不改变**该形态的降级结果。→ 进入 Round 2 做结构性尝试。

### Round 2 — 结构性分支（破串行 / 减重读）

| candidate_id | 唯一主要优化点 | 结果 | 证据 |
|---|---|---|---|
| c_split2 | 偶数行→KV_shared、奇数行→KV_b 两个目的 buffer（语义精确的 2 路拆分） | **blocked-compile**（未写入 perf_records） | `loc("kernel.npuir":115:9): error: 'hivm.hir.load' op if pad_mode is not set, src and dst shape should be the same!`（同一 kernel 体内第二个「动态行 GM→shared」copy 被拒；把两段拆成两个 `T.serial` 循环仍然同样报错，行号 120） |
| c_unroll4 | `T.unroll(4)` 展开行循环 | **blocked-compile** | `error: 'hivm.hir.copy' op Unsupported copy from cbuf to cbuf!` |
| c_bh64 | `block_heads=64`（DESIGN §1.6.3 备选，主选复核） | **blocked-compile** | `error: ub overflow, requires 2115584 bits while 1572864 bits available!`（= 264448 B > 196608 B）。另按容量算术：`acc_o[64,512]f32 = 128 KB` + `acc_s[64,64]f32 = 16 KB` = 144 KB > L0C 128 KB |
| c_bh64_f16 | `block_heads=64` + PV 累加器改 fp16（`acc_o[64,512]f16 = 64 KB`，L0C 80 KB ✓） | **blocked-runtime** | 先 `ub overflow, requires 1574912 bits while 1572864 bits available`（196864 B > 196608 B，仅超 256 B）；去掉 fp32 `acc_o32`/`O_ub`、末尾改 fp16 归一后**编译通过但运行时崩**：`aicore exception ... The DDR address of the MTE instruction is out of range`（code 0x800000, mte error 0x86060000e7）。判为 fp16 L0C 累加路径不可用，未继续调试 |

**本轮对比表（候选 vs current best）**

| 候选 | Task Duration (µs) | AICore 利用率 | memory | L0 |
|---|---:|---|---|---|
| current best（baseline） | 9022.3（smoke）/ 35598.5（long-kv） | 0.000182 / 0.000617 | 583.5 KB/block | PASS |
| c_split2 | 不可测（compile fail） | — | — | — |
| c_unroll4 | 不可测（compile fail） | — | — | — |
| c_bh64 | 不可测（compile fail, UB 264448 B） | — | — | L2-5 WARN（记录、非阻塞） |
| c_bh64_f16 | 不可测（runtime error） | — | — | 未进入判定 |

**合并非回退检查**：本轮无 winner，current best 不变；因此无需合并，也未触发全量复测（final 复测仍在最终文件上完成了 3 个 tune workload + smoke 精度）。

### 中止条件达成

- `blocked` + `plateau`：Round 1 的 8 个参数/形态分支全部落在噪声区或回退；Round 2 的 3 个结构分支全部被编译器/硬件拒绝。剩余候选方向（两相位 gather→GM workspace、dense-scan + bitmap 掩码）**超出本次冻结结构契约**（「行连续 gather + PV 前缀列 gemm + Developer MixCV 非持久一维 grid」），不能作为 Stage 4 候选实施 → 转 `perf_feedback.md` 上报（见 §6）。
- 预算：rounds = 2（< max_rounds 10），experiments = 14 分支（含 2 个诊断探针，< max_experiments 30）。

---

## 4. KV 重读倍数与 BH 裁决（实测结论）

| 项 | 结论 | 证据 |
|---|---|---|
| KV 重读倍数 | **= H/block_heads = 128/32 = 4**（未变） | 每 block gather `topk` 行；逻辑核 `B·S·(H/BH)`；mainstream 重读总量 9.66 GB |
| BH=64 是否可行 | **不可行（双向越限），维持主选 BH=32** | ① 容量算术：`acc_o[64,512]f32=128 KB`+`acc_s[64,64]f32=16 KB`=144 KB > L0C 128 KB（与 `CONST-capacity-910B2C` 一致）；② 本工具链实测：`block_heads=64` → `ub overflow, requires 2115584 bits while 1572864 bits available`（264448 B > 196608 B），与迁移期 258 KB 报告同量级；③ fp16 累加器绕开容量后运行时崩（DDR out of range） |
| 重读倍数是否为当前瓶颈 | **不是** | 把重读消除到 1× 至多 4× 收益；而 ≤56 头（L0C 上限）不整除 H=128。当前时延是 gather **每行固定成本**所致，与重读倍数正交：BH 翻倍只把「行数」减半（≤2×），无法改变 ~1.3 µs/行 |
| gather 每行成本（实测外推） | 1.34 µs/行：`Task Duration / (blocks/waves × iterations × block_N)`；三个 tune workload 完全一致（mainstream 42–46 ns/行·核归一后同值，long-kv 每 core 每行 ≈1.34 µs） | smoke 353 µs/block ÷ (4 迭代 × 64 行) = 1.38 µs/行；long-kv 695 µs/block ÷ (8×64) = 1.36 µs/行；与 `block_N`（16/32/64）无关 |

**结论**：`block_heads` 旋钮在两个方向上都不可动（L0C/UB 封顶 32，fp16 路径运行时不成立），KV 重读 ×4 是冻结结构的必然代价；且**它不是主导项** —— gather 的逐行降级才是。

---

## 5. Autotune Log

未使用 autotune。原因：Round 1 已证明瓶颈不在可搜索参数空间（`block_N ∈ {16,32,64}`、`num_stages`、gather 形态全部中性或回退），autotune 只做参数选择、不能改变结构，故无搜索价值。

---

## 6. Final Summary

- **最终候选**：`perf_opt/_sparse_mla_kernel.py`（与 Stage 3 基准功能完全一致，仅增加注释块；`block_heads=32 / block_N=64 / num_stages=2` 不变，KV 重读 ×4）。
- **结论**：`no_gain` —— 12 个保持语义的分支 + 3 个结构性分支均未带来超过 3% 噪声阈的提升，其中 3 个因编译/运行时限制被 blocking 拒绝。final 与 baseline 的差异 ≤ 0.2%（噪声）。
- **smoke 最终精度**：`python perf_opt/_sparse_mla_kernel.py --level all` → L0 5/5 PASS、L1 5/5 PASS、L2 4 pass + 1 WARN（BH=64 编译失败，非阻塞、既有已知）、boundary 4 pass + 4 契约拒绝 PASS、`All check passed!`（`perf_opt/logs/l0/final_all.log`）。
- **中止原因**：`blocked`（结构分支全部被工具链/硬件拒绝）+ `plateau`（参数空间已充分验证）。
- **遗留**：真正的 >2× 路径在设计层（gather 形态 / 两相位 workspace），见 `perf_opt/perf_feedback.md`（`[DESIGN_LIMIT]`）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---|---:|---|
| examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py::_sparse_mla_kernel::main | single-batch-mainstream | 566545.37 | 567406.19 | final |
| examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py::_sparse_mla_kernel::main | longer-kv-lower-topk | 141754.06 | 142017.29 | final |
| examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py::_sparse_mla_kernel::main | long-kv-low-topk | 35598.48 | 35597.61 | final |

（`final_us` 全部来自同一最终候选 `perf_opt/_sparse_mla_kernel.py` 的 `phase=final` 记录，
raw profile: `perf_opt/profiles/final_{mainstream,longer-kv,long-kv}/`；baseline / final 偏差
−0.15% / −0.19% / +0.00%，均在噪声内。）

| workload_id | baseline → final（µs） | 提升 |
|---|---|---|
| single-batch-mainstream | 566545.37 → 567406.19 | −0.15%（噪声） |
| longer-kv-lower-topk | 141754.06 → 142017.29 | −0.19%（噪声） |
| long-kv-low-topk | 35598.48 → 35597.61 | +0.00% |

---

## 7. Skill Retrospective

### 7.1 流程问题与建议

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Optimize-flow | `profile-collection.md` 只给「`msprof op --kernel-name=... --launch-count=20`」模板，未提示 **`--kernel-name` + 多 metric 组会触发 `--replay-mode=kernel` 的逐 launch 重放**：本任务 mainstream（566 ms/launch）8 launch 采集耗时 >27 min 并被 30 min 超时截断；去掉 `--kernel-name` 用 `--aic-metrics=Default` 后同样 8 launch 仅约 60 s（≈30× 差距） | 本轮 `perf_opt/profiles/base_mainstream/OPPROF_*/device0/main_mix_aic/{0..7}/OpBasicInfo_*.csv` 时间戳 02:26:47→02:53:52（8 个 launch 用时 27 min）；`perf_opt/profiles/baseline_long-kv`（无 `--kernel-name`，Default）5 launch < 60 s | 在 `profile-collection.md` §3 增补「慢 kernel（>100 ms/launch）采集策略：优先 `--aic-metrics=Default` 且不用 `--kernel-name`（事后按 `OpBasicInfo.csv` 的 `Op Name` 过滤），或显式 `--replay-mode=application`；`--launch-count` 计的是**全部** kernel（含输入生成的 Cast/Fill），需 ≥ 目标 launch 数 + 3」 | R |
| Optimize-flow | 诊断分支（数学无效、仅测时）与「实验分支」在 skill 中没有明确身份：它们不能写入 `perf_records.jsonl`（`duration_us` 必须为实测且有效），但对定位价值极高（本轮 `diag_contig` 一个探针把 94.5% 的时延归因锁定，替代了数轮盲试） | 本轮 `perf_opt/probes/_probe_contig.py`：smoke shape 9041→499 µs（−94.5%），`aiv_vec_ratio` 0.00062→0.0109 | 在 `profile-collection.md`/`iteration-diagnosis.md` 增补「**降级探针（diagnostic probe）**」一节：允许构造数学无效但可编译的最小变体用于**归因**，要求 ① 独立目录 `perf_opt/probes/` ② 结果只进 opt_log 的「诊断」行、带 `数学无效` 标注 ③ 明确禁止参与 winner 判定与 `perf_records.jsonl` | P |
| Optimize-flow | skill 要求「每轮基于 current best 的最新 profile 重新分析」，但没有提示 **tune workload 与 smoke shape 的 profile 可采用同一套诊断**（同 kernel、同代码路径，仅迭代数不同）。本轮先用 smoke shape（9 ms/launch）做形态筛选，再用 `long-kv`（35.6 ms/launch）复核存活候选，把 14 个分支的总采集成本压到可接受范围 | 本轮：smoke 探针 ≈10 s/次；long-kv 复核 ≈60 s/次；mainstream ≈60 s/次 | 在 `profile-collection.md` §2 排序节增补「**两级筛选**：小 tune workload（或与 tune 同代码路径的最短案例）做形态筛选，存活候选再上大 tune workload 复核；不得以 smoke 结果做 winner 判定」 | R |

### 7.2 Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **npnir Developer 下「动态行索引 GM→shared」的 `T.copy` 被降级为串行化 `hivm.hir.load`（≈1.3 µs/行，≈1.5 GB/s 每核）**，与 `block_N`/`num_stages`/行循环形态（`T.serial`/`T.Pipelined`/`T.Parallel`/手工展开）/切片形态（4D、2D 显式切片、base+size、KV 降秩 2D、目的 `alloc_L1`）**全部无关**；同形状的**连续 64 行** `T.copy` 却快 18×，说明是「逐行 descriptor + 同步 load」而非带宽/算力上限。稀疏 MLA / paged attention / MoE 路由类算子的首要性能判据 | D | `perf_opt/probes/_probe_contig.py`（连续 gather）smoke 9041→499 µs；`perf_opt/opt_log.md` §3.0/§3 Round1 表；raw `perf_opt/profiles/{probe,probe_contig,probe_g1,probe_g5,probe_g6,probe_par,probe_l1,probe_flat,probe_bs}/` | `python3 perf_opt/probes/_probe_contig.py` 与 `_sparse_mla_kernel.py` 同 shape 对比（`msprof op --aic-metrics=Default`） | tilelang 0.1.2+5a90b9918747fe9ed4870e6cd5c39760c68c33e8 / Ascend910B3（aicore=20）/ 2026-10-18 | `pattern-library/attention.md` + `traps-compiler.md`（新增条目，绑定本工具链戳） |
| **同一 kernel 体内「第二个动态行 GM→shared copy」被编译器拒绝**：`hivm.hir.load` verifier 报 `if pad_mode is not set, src and dst shape should be the same!` —— 因此「行 gather 拆成 k 路目的 buffer 以打破串行」这一标准去串行手法在 Developer 模式不可用（拆成两个 `T.serial` 循环亦然）；同源限制还包括 `T.unroll` 内的 copy 报 `Unsupported copy from cbuf to cbuf` | D | `perf_opt/probes/_probe_split2.py`（两段循环版本）、`perf_opt/probes/_probe_gather.py --gform 4`；错误报文见 `opt_log.md` §3 Round2 表 | `python3 perf_opt/probes/_probe_split2.py --case smoke-f16`（断言编译失败） | tilelang 0.1.2+5a90b99 / Ascend910B3 / 2026-10-18 | `pattern-library/traps-compiler.md`（新增 TRAP-GATHER-MULTI-DEST） |
| **910B3 大 D 稀疏 MLA：BH=64 在本工具链仍双向越限**（UB 264448 B > 196608 B；L0C 144 KB > 128 KB），且 fp16 PV 累加器绕开容量后运行时报 `DDR address of the MTE instruction is out of range` | D | 本轮 `block_heads=64` 与 `block_heads=64 + fp16 acc_o` 两次实测；`perf_opt/opt_log.md` §4 | `python3 perf_opt/bench_runner.py --case smoke-f16 --block-heads 64`（断言编译失败） | tilelang 0.1.2+5a90b99 / Ascend910B3 / 2026-10-18 | `pattern-library/constants.md` + `attention.md`（VP-2026-0261 的有效性再确认，工具链戳更新） |
| 大 D 稀疏 MLA 的 head 分块上限由 **L0C 预算算术**（`BH·(D+block_N)·4 ≤ 128KB` ⇒ BH ≤ 56）决定，而 H=128 的整除约束使可选值只剩 {32}；KV 重读倍数 4 是**冻结结构**的必然代价，但它并非主导项——逐行 gather 的固定成本（≈1.3 µs/行）比它高两个数量级 | P | 本轮 §4 表；DESIGN §1.6.0 R4 容量算术 | 无（容量算术自证 + 上述两条 D 类实测） | tilelang 0.1.2+5a90b99 / 910B3 / 2026-10-18 | `pattern-library/attention.md`（更新 VP-2026-0259/0260 的相对权重表述） |
| 稀疏 attention 的 Stage 4 首轮应**先做「gather 归因探针」**（连续 gather vs 逐行 gather 的 A/B），再谈 tiling 与 BH 裁决：本轮把 566 ms 的 94.5% 一次性锁定在 gather 形态，避免了在 block_N/BH/num_stages 上的多轮盲试 | R | 本轮 Round 1：12 个保持语义的分支全部中性；1 个降级探针即完成归因 | `perf_opt/probes/_probe_contig.py` | — | `tilelang-op-optimize/references/iteration-diagnosis.md`（新增「首轮必做归因探针」清单项） |

### 7.3 本次实测新常数（供 `constants.md`）

- `CONST-910B3-dyn-row-gather`：910B3 / tilelang 0.1.2+5a90b99，Developer MixCV，动态行索引 GM→shared `T.copy`（1152 B/行）：**≈1.34 µs/行，≈1.5 GB/s 每核**；同数据量连续 64 行 copy：**≈77 ns/行，≈15 GB/s 每核**（smoke shape，20/24 核，1800 MHz）。工具链变更后需重验。

---

## 8. 工件清单（本 Agent 实际创建/修改）

- `perf_opt/_sparse_mla_kernel.py`（最终候选 = 基准 + 注释块）
- `perf_opt/opt_log.md`（本文件）
- `perf_opt/perf_records.jsonl`（append-only，8 行）
- `perf_opt/workload_inventory.json`
- `perf_opt/perf_feedback.md`（`[DESIGN_LIMIT]`）
- `perf_opt/bench_runner.py`（workload 驱动脚本）
- `perf_opt/probes/{_probe_contig.py,_probe_gather.py,_probe_flat.py,_probe_basesize.py,_probe_l1.py,_probe_par.py,_probe_split2.py,_probe_bh64f16.py}`
- `perf_opt/profiles/**`（raw msprof；baseline / candidate / final / probe）
- `perf_opt/logs/**`（实验 stdout/stderr：`final/`、`l0/`、`bn32_longkv.log`、`l1_longkv.log`）
- 知识域回写（ED-C，`pattern-library/` 是唯一允许的 skill 目录例外）：
  - `pattern-library/attention.md` 新增 `PL-1.38-dyn-row-gather-serialized-developer`
  - `pattern-library/constants.md` 新增 `CONST-910B3-dyn-row-gather`
  - `pattern-library/traps-compiler.md` 新增 `TRAP-GATHER-MULTI-DEST`（`repro-missing`，隔离骨架未复现，降级 Tier 1 候选）
  - `pattern-library/repro/PL-1.38-dyn-row-gather-serialized-developer.py`（已实跑 PASS）
  - `pattern-library/INDEX.md` 路由表登记 3 条新条目
- 未修改：`_sparse_mla_kernel.py`（基准）、`_sparse_mla_kernel_DESIGN.md`、wrapper、`tests/`、`benchmarks/`、`manifest/`、`DESIGN.md`
