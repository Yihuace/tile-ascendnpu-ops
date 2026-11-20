# Stage 4 性能调优日志（第 2 轮）— `_sparse_mla_kernel` (deepseek_dsa_decode)

- 算子（集成 manifest）: `DeepSeekSparseAttentionDecodeWithKVCacheFwdOp`（family: attention）
- kernel 文件（基准，只读）: `examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py`
- 设计参考: `_sparse_mla_kernel_DESIGN.md`（含「性能目标」章节）
- 最终候选: `perf_opt/_sparse_mla_kernel.py`（= Stage 3 基准 + 注释块；默认参数不变）
- 唯一时延口径: `msprof op` → `main_mix_aic` 的 `Task Duration(us)`（kernel-only；不使用 NPU event / 端到端时间）
- 平台: Ascend 910B3（npu-smi chip 0，#1 910B3，1800 MHz）
- **工具链戳（本轮，已变更）**: tilelang `0.1.2+63c8a2e6` / CANN 9.0.0（msprof 25.5.0）/ torch_npu 2.7.1.post4 / 2026-11-01
  - 上一轮为 `0.1.2+5a90b99`。按 E-5，本轮全部结论在本工具链上**重新实测**，不直接引用上轮数字。
- 本轮性质: **复测 + 主攻「逐行动态索引 gather 的 1.34 µs/行 能否下降」+ 量化上界**
- 预算: 限 100 min / 16 实验分支（实用 3 个形态分支 + 1 个降级探针 + 6 次 tune-workload 采集）

上一轮工件原样移至 `perf_opt/history/`（`opt_log.md` / `perf_records.jsonl` / `workload_inventory.json` / `perf_feedback.md`），本轮全部重新采集。

---

## 1. Workload Inventory（由迁移后 benchmark 展开）

来源 `examples/TileOPs/benchmarks/ops/bench_deepseek_dsa_decode.py::test_dsa_decode_bench`；机器可读清单 `perf_opt/workload_inventory.json`。

| workload_id | q_shape | kv_shape | topk | dtype | kind | 排序理由 |
|---|---|---|---|---|---|---|
| dsa-decode-smoke-float16 | [1,128,128,576] | [1,256,1,576] | 256 | float16 | smoke | 只做精度回归 + **形态筛选探针**（同代码路径、9 ms/launch，成本低 60×），不做 winner 判定 |
| dsa-decode-smoke-bfloat16 | 同上 | 同上 | 256 | bfloat16 | smoke | 只做精度回归（bf16 路径） |
| single-batch-mainstream | [1,1024,128,576] | [1,2048,1,576] | 2048 | float16 | tune | 负担最大（566.6 ms）→ 第 1 |
| longer-kv-lower-topk | [1,512,128,576] | [1,4096,1,576] | 1024 | float16 | tune | 第 2（142.4 ms） |
| long-kv-low-topk | [1,256,128,576] | [1,8192,1,576] | 512 | float16 | tune | 第 3（35.7 ms） |

- 5 个 pytest 参数化案例全部调用同一 kernel（`SparseMlaKernel` → 唯一 `@T.prim_func::main`，单 kernel）。
- 无 `bench_skip_reason` → 无 skipped。
- smoke 不做性能调优；本轮仅把它当作**形态筛选平台**（明确不参与 winner 判定，其数据不写入 `perf_records.jsonl`）。

---

## 2. Performance Test Data（`phase=baseline`，msprof op，串行，本工具链重新采集）

命令模板：
```
msprof op --output=profiles/<dir> --launch-count=<N> --warm-up=<W> --aic-metrics=Default \
          python bench_runner.py --file <kernel.py> --case <workload> --repeat R --warmup W
```

| kernel_id | workload_id | target_kernel_name | task_duration_us | launches | profile_status | raw_profile_dir |
|---|---|---|---:|---:|---|---|
| `.../_sparse_mla_kernel.py::_sparse_mla_kernel::main` | single-batch-mainstream | main_mix_aic | 566606.00 | 5 | valid | `perf_opt/profiles/r0_base_mainstream/` |
| 同上 | longer-kv-lower-topk | main_mix_aic | 142398.31 | 2 | valid | `perf_opt/profiles/r0_base_longer-kv/` |
| 同上 | long-kv-low-topk | main_mix_aic | 35687.42 | 2 | valid | `perf_opt/profiles/r0_base_long-kv/` |

profile 有效性：`Op Name == main_mix_aic`、`Op Type == mix`、`Block Dim` 与逻辑核数一致（4096 / 2048 / 1024），非 Cast/Fill 等框架小算子。官方 ideal（benchmark roofline）：mainstream 1994.7 µs / longer-kv 499.0 µs / long-kv 124.9 µs ⇒ 官方 Ratio 全 0.35%。

### 2.1 设计估算 vs 实测（D-2 回填）

| 项 | DESIGN §1.6.0 R4（mainstream） | 实测 baseline | 偏差 |
|---|---|---|---|
| Task Duration | 4.0 – 9.4 ms | **566.6 ms** | **60× – 142× 失准** |
| 失准项 | 假设 gather 代价 = 字节数 ÷ 1.75 TB/s | per-row 动态索引 `T.copy` 实测 **≈1.34 µs/行**（≈1.5 GB/s 每核） | 流量/发射口径均失准 |

### 2.2 smoke 排除与精度

`dsa-decode-smoke-*`：按规则排除性能调优；最终 `--level all` 中 L0-1（f16）/L0-2（bf16）/L2-1（bf16）全 PASS。

---

## 3. Iteration Log

### 3.0 本轮现象（baseline 最新 profile，smoke shape，本工具链）

`profiles/t_smoke/`（512 blocks × 4 迭代，每 block 均值）：

| 指标 | 值 | 解读 |
|---|---|---|
| `aic_time` | 357.8 µs | block 墙钟 |
| `aic_cube_time` / `aic_cube_ratio` | 1.64 µs / **0.000182** | Cube 几乎全程空闲 |
| `aic_mte2_time` | 7.73 µs | 真实 DMA 流量极小 |
| `aic_scalar_time` | **117.45 µs** | 标量单元驱动逐行 copy |
| `aic_scalar_wait_id1_time` | **383.23 µs** | 全程等 flag（CV 相位串行） |

判据：`cube_ratio = 1.8e-4`、mte2 近零而标量 + 等待占满 → 瓶颈是 **per-row gather 的逐条发射/同步**，不是算力也不是 HBM 带宽（「标量占比 >50% ⇒ 强制换轴分析」的触发条件）。

### Round 1 — 主攻 gather 逐行成本（3 个新形态 + 1 个降级探针，smoke 平台筛选）

固定 base = current best（`perf_opt/_sparse_mla_kernel.py`，`BH=32 / BN=64 / ns=2`），baseline smoke = **9045.26 µs**（n=5）。

| candidate_id | 唯一主要优化点（本轮**新**增，上轮 13 个变体均未覆盖「目的域 + 单次连续搬运」这一对） | Task Duration (µs) | vs base | L0 | 结论 |
|---|---|---:|---:|---|---|
| baseline（=current best） | — | 9045.26 | — | PASS | base |
| `p_ubpar` | 逐行 gather 改由 `T.Parallel(Dqk)` 逐元素众集进 UB（试图向量化），再 **1 次连续** UB→gemm buffer copy | 9049.30 | **+0.04%** | PASS | 无收益（落入噪声；逐元素路径未向量化为宽搬运） |
| `p_ubrow` | 逐行 `T.copy` 目的地由 shared 改 UB，再 1 次连续 UB→gemm buffer copy（改「目的域」） | 8978.83 | **−0.74%** | PASS | 无收益（<3% 噪声阈） |
| **`p_contig`（诊断，数学无效）** | 64 次逐行 copy → 1 次 64 行**连续** copy（仅测时） | **501.15** | **−94.5%（18.05×）** | n/a | **归因复现：gather 占 94.5%** |

**本轮对比表（候选 vs current best；B2 结构化回流，数据来自 `perf_records` 与本轮 raw profile）**

| 候选 | Task Duration (µs) | `aic_cube_ratio`（每 block） | `aic_scalar_time` / `aic_scalar_wait_id1`（µs，每 block） | memory（`aic_mte2_time` µs/block） | L0 |
|---|---:|---|---|---|---|
| current best（baseline） | 9045.26（smoke）/ 566606.00（mainstream） | 0.000182 | 117.45 / 383.23 | 7.73 | PASS |
| `p_ubpar` | 9049.30 | 0.000181 | 119.96 / 382.09 | 9.54 | PASS |
| `p_ubrow` | 8978.83 | 0.000183 | 137.98 / 354.66 | 5.85 | PASS |
| `p_contig`（诊断） | 501.15 | **0.003278（18×）** | **13.06 / 17.70** | 8.51 | n/a（数学无效） |

现象与结论：两个改变「目的域 + 单次连续搬运」的分支（把 64 条 copy 变成 64 条 copy + 1 条连续 copy）都落在 −0.7%…+0.04% 的噪声带内，且 `aic_scalar_time` / `wait_id1` 基本不动（117→120 / 383→382；`p_ubrow` 137.98 反而更高）。而把「64 条 copy」变成「1 条 copy」即得 18.05×。⇒ **成本 ∝ copy 的条数（每条 ≈1.3 µs 的固定发射+同步成本），与目的内存域、搬运字节数、元素级向量化形式均无关**。

### Round 2 — tune workload 复核（是否可合并 / 是否有任何回退）

| candidate | mainstream (µs) | longer-kv (µs) | long-kv (µs) | 结论 |
|---|---:|---:|---:|---|
| baseline（current best） | 566606.00 | 142398.31 | 35687.42 | base |
| final（= current best，无 winner） | 566916.62 (+0.05%) | 141835.34 (−0.40%) | 35692.38 (+0.01%) | 全部落在 ±0.5% 噪声内 |

**合并非回退检查**：本轮无 valid winner（`family_no_gain`：`gather_destination_domain` 家族 2 个分支均 ≤0.74%），current best 保持不变；三个 tune workload 的 final 复测全部在噪声内非回退。未发现任何可安全合并的改动，因此**不存在**「合并后其它 workload 回退」的情形。

**中止条件达成**：`plateau`（连续无 valid 分支超过 3% 噪声阈；且本轮新覆盖的「目的域/单次连续搬运」方向已被证伪）+ `blocked`（剩余方向需改数据流/语义契约）。参照结构 diff（T-1 morph 轻量版）已做：仓内同族稀疏 attention 先例（`examples/deepseek_v32/sparse_mla_fwd_exp.py`、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_fwd.py`）均为「GM workspace 两相位」或 Expert 双 Scope 结构，其可移植性阻塞点（Developer 非持久 grid 内不能插入相位）在上一轮已实测记录，且本轮实测确认该结构族外的唯一近亲（dense-scan）在 long-kv 密度（512/8192）下流量 ×16，不构成可行变异。

---

## 4. gather 成本能否下降 —— 结论与量化上界（本轮核心交付）

### 4.1 结论：**在 op 契约内不能下降**

实测三件事：
1. `p_ubpar` / `p_ubrow`（目的域 = UB、加 1 次连续搬运）→ **−0.04% / −0.74%**（噪声）。
2. `p_contig`（64 条 copy → 1 条 copy，数学无效）→ **18.05×**（9045.26 → 501.15 µs）。
3. 三个 tune workload 的逐行成本完全一致：mainstream 566606/(205 waves×2048 行)=1.349 µs/行；longer-kv 142398/(103×1024)=1.351 µs/行；long-kv 35687/(52×512)=1.341 µs/行。

⇒ 每 block 的 gather 条数 = `topk`（索引随机分布，行地址互不相邻），而**每条 copy 的固定成本 ≈1.34 µs**（≈1.3 µs 发射+同步 + ≈0.04 µs 搬运）。要降成本只能减少**条数**，可减少条数的三条路全部被封：
- 增加 `block_N`（一次 copy 覆盖更多行）：行地址随机，无法连续；
- 增加 `block_heads`（减少 block 数 ⇒ 减少总行数）：L0C 144 KB > 128 KB 且 UB 264448 B > 196608 B（上轮实测，本轮 L2-5 复现编译失败）；
- 把 64 条拆成 k 路并行目的 buffer：`hivm.hir.load` verifier 拒绝同一 kernel 体内第二个「动态行 GM→shared」copy（上轮实测报错 `if pad_mode is not set, src and dst shape should be the same!`；本轮复核 `src/` 与 `3rdparty/AscendNPU-IR/.../HIVMDMAOps.td` L87-93 确认 `pad_mode` 仅是**左右 padding** 语义、且未由 tilelang 前端暴露 ⇒ 该报错不可通过参数开启）。

### 4.2 量化上界（`p_contig` 逐 workload 实测「零 gather 成本」地板）

诊断探针（数学无效，仅测时；`--repeat 3`）：

| workload | baseline_us | **零-gather 地板_us（实测）** | 地板加速比 | 官方 ideal_us | **Ratio 上界 = ideal/地板** |
|---|---:|---:|---:|---:|---:|
| single-batch-mainstream | 566606.00 | **26525.50** | 21.36× | 1994.7 | **7.52%** |
| longer-kv-lower-topk | 142398.31 | **6828.78** | 20.85× | 499.0 | **7.31%** |
| long-kv-low-topk | 35687.42 | **1793.82** | 19.90× | 124.9 | **6.96%** |

口径说明：Ratio ∝ `ideal / measured`（roofline 记忆受限点，字节数与 AI 不变），故「地板」直接给出 Ratio 上界。

**40% 可达性裁决：不可达，差 5.3–5.7×。**
- 40% 对应 mainstream **4987 µs**；而**把 gather 成本整体删掉**（不消耗任何一条 copy）也只有 26525.5 µs，比 40% 所需还慢 **5.32×**（longer-kv 5.47×、long-kv 5.75×）。
- 因此 40% 不是「gather 没调好」，而是**结构层**问题：即使在冻结结构内做到绝对最优（gather 免费），Ratio 天花板也只有 **7.0–7.5%**（相对 0.35% 提升 **≈20–21×**）。
- 若要突破 7.5%，唯一路径是把 gather 之外的 94.5%→5.5% 那一段（剩余 26525.5 µs = 每迭代 4.8 µs 的 CV 相位串行微操作延迟 ×32 迭代 ×205 波）也压下去，那是数据流重设计（workspace 两相位 / dense-scan），属设计层；且 dense-scan 在 long-kv 密度（512/8192）下流量 ×16、实测推算约 80 ms（比现状 35.7 ms 更慢），不能全局替换。详见 `perf_opt/perf_feedback.md`。

### 4.3 本轮实测新常数

- `CONST-dyn-row-gather`（910B3 / tilelang 0.1.2+63c8a2e6 / Developer MixCV，1152 B/行）：**1.341–1.351 µs/行**，与 `block_N`、目的内存域（shared vs UB）、元素级向量化形态（`T.Parallel`）无关；
- `CONST-iter-micro-op`：**每迭代 4.8 µs**（64 行连续搬运 + 2 次 gemm + online softmax + 同步），即「零 gather 地板」的构成单元（mainstream 205 波 × 32 迭代 × 4.8 µs ≈ 26.5 ms，与实测 26525.5 µs 吻合）。

---

## 5. Autotune Log

未使用 autotune：本轮焦点是「gather 条数」这一结构性量，可搜索参数空间（`block_N ∈ {16,32,64}`、`num_stages`）在上轮已穷举（全部中性或回退），且本轮实测证明瓶颈与目的域/向量化形态无关，autotune 无搜索价值。

---

## 6. Final Summary

- **最终候选**：`perf_opt/_sparse_mla_kernel.py`（= Stage 3 基准 + 注释块；`block_heads=32 / block_N=64 / num_stages=2`，KV 重读 ×4 不变）。sha256 `49033e78cad22b978626cee59a26ade2625cea158718a327b1aa51dd99f63a6c`。
- **裁决**：`no_gain` —— 本轮 2 个保持语义的新分支（目的域/单次连续搬运）≤0.74%；gather 逐行成本 1.34 µs/行不可在契约内下降；官方 Ratio 上界 **7.0–7.5%**，40% 不可达（需 5.3–5.7× 超过零-gather 地板）。
- **smoke 最终精度**：`python perf_opt/_sparse_mla_kernel.py --level all` → L0 5/5 PASS、L1 5/5 PASS、L2 4 PASS + 1 WARN（BH=64 上界，非阻塞）、boundary 4 PASS + 4 契约拒绝 PASS、`All check passed!`（`perf_opt/logs/l0/final_all.log`，2m43s）。
- **中止原因**：`plateau` + `blocked`；设计层天花板已量化 → `perf_feedback.md` + `[DESIGN_LIMIT]`。
- **预算**：rounds = 2（< 10）、实验分支 = 3 形态分支 + 1 降级探针（< 16）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py::_sparse_mla_kernel::main | single-batch-mainstream | 566606.00 | 566916.62 | final |
| examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py::_sparse_mla_kernel::main | longer-kv-lower-topk | 142398.31 | 141835.34 | final |
| examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode_kernel/_sparse_mla_kernel.py::_sparse_mla_kernel::main | long-kv-low-topk | 35687.42 | 35692.38 | final |

（`final_us` 全部来自同一最终候选 `perf_opt/_sparse_mla_kernel.py` 的 `phase=final` 记录，与 `perf_records.jsonl` 逐项一致；
raw profile `perf_opt/profiles/r0_final_{mainstream,longer-kv,long-kv}/`；baseline raw `r0_base_*/`。）

比值（官方 Ratio 口径，baseline → final）：single-batch-mainstream `0.352% → 0.352%`（+0.05%，噪声）、
longer-kv-lower-topk `0.350% → 0.352%`（−0.40%，噪声）、long-kv-low-topk `0.350% → 0.350%`（+0.01%，噪声）。

**smoke/skip 排除**：`dsa-decode-smoke-float16/-bfloat16` 只做精度回归（PASS），无 skipped。

---

## 7. Skill Retrospective

### 7.1 流程问题与建议

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Optimize-flow | **工具链变更后未强制「重验优先」**：本任务工具链从 `0.1.2+5a90b99` 变为 `0.1.2+63c8a2e6`，上一轮 14 个分支的结论全部技术上失效，但 skill 的 E-5 只说「标注待重验」，没有说「重验成本高时应优先复用上轮的**诊断**结论结构（探针形态）而非重复枚举参数分支」 | 本轮用 1 个降级探针 `p_contig`（1 次 80 s 采集）即复现「18.05× / 94.5%」并锁定结论，未重跑上轮 13 个变体 | `profile-collection.md`/`iteration-diagnosis.md` 增加「工具链变更后的最小重验集 = 1 个降级探针 + 2 个最强候选形态」 | R |
| Optimize-flow | **大 kernel（>100 ms/launch）的 msprof 采集成本高 1 个数量级**：mainstream 8 launches 用 20 min，而 longer-kv/long-kv 各 2 min 级 | 本轮 `r0_base_mainstream` 时间戳 10:02→10:22；同命令 `r0_base_long-kv` <2 min | `profile-collection.md` §3 增补「>100 ms/launch 的 workload 用 `--repeat 3 --warmup 0 --launch-count=4`；先在小 workload 做形态筛选」 | R |
| Optimize-flow | **「零成本地板」应成为性能目标的固定判据**：用户目标（40% Ratio）是否可达，用一次「删掉瓶颈操作」的降级探针即可给出硬上界，比继续枚举优化分支更快收敛，也直接喂给 `[DESIGN_LIMIT]` 的量化门槛 | 本轮 `p_contig` 逐 workload 给出 26525.5/6828.78/1793.82 µs ⇒ Ratio 上界 7.0–7.5%，直接裁决 40% 不可达 | 在 `iteration-diagnosis.md` 增加「目标可达性裁决：先测零瓶颈地板，再决定是否进入参数搜索」 | P |

### 7.2 Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **动态行索引 GM→on-chip 逐行 copy 的成本 ∝ 条数、与目的内存域/向量化形态无关**：把目的地从 shared 改 UB、或改写成 `T.Parallel(Dqk)` 逐元素众集 + 1 次连续 UB→buffer copy，均 ≤0.74%（=噪声）；而同样字节数改成 1 条 64 行连续 copy 即 18.05× | D | `perf_opt/profiles/{t_smoke,r1_p_ubpar,r1_p_ubrow,r1_p_contig}/`；`_sparse_mla_kernel_r1_p_*.py`；opt_log §3 Round 1 | `python bench_runner.py --file _sparse_mla_kernel_r1_p_ubpar.py --case smoke-f16 --repeat 5`（同 9045 µs） | tilelang 0.1.2+63c8a2e6 / 910B3 / 2026-11-01 | `pattern-library/attention.md`（补充 `PL-1.38` 的「目的域无关」证据，工具链戳更新）+ `constants.md` `CONST-dyn-row-gather` |
| **每迭代 4.8 µs 的 CV 相位串行微操作成本**（64 行连续搬运 + 2 gemm + online softmax + 同步），是「零 gather 地板」的构成单元；`T.Pipelined(num_stages=2)` 无法把它与相邻迭代重叠 | D | mainstream 零-gather 地板 26525.5 µs = 205 波 × 32 迭代 × 4.8 µs（实测吻合）；`profiles/r1_contig_{mainstream,longer-kv,long-kv}/` | `python bench_runner.py --file _sparse_mla_kernel_r1_p_contig.py --case long-kv` | 同上 | `pattern-library/constants.md`（新增 `CONST-iter-micro-op`）+ `attention.md` |
| **Ratio 目标可达性的「零瓶颈地板」判据**：目标 Ratio 对应时延若已低于「删掉瓶颈操作」的实测地板，则该目标在当前结构族内**结构性不可达**，应立即产出量化上界而非继续搜索 | R | 40% ⇒ 4987 µs vs 地板 26525.5 µs（差 5.32×）；三个 workload 一致 | 上述两个探针 | — | `iteration-diagnosis.md`（新增「目标可达性裁决」节） |

### 7.3 结论边界（未实证/待重验）

- `p_contig` 是**数学无效**的降级探针，只用于测时与归因，未参与 winner 判定、未写入 `perf_records.jsonl`。
- dense-scan 在 long-kv 上「≈80 ms、比现状更慢」为**外推估计**（按 8192 行 × 119 ns/行 + 128 迭代 × 4.8 µs 估），未实测。
- 两条 D 类结论绑定工具链 `0.1.2+63c8a2e6`，升级后需重验。

---

## 8. 工件清单（本轮实际创建/修改）

- `perf_opt/_sparse_mla_kernel.py`（最终候选 = 基准 + 注释块，sha256 `49033e78…`）
- `perf_opt/opt_log.md`（本文件）、`perf_opt/perf_records.jsonl`（6 行：3 baseline + 3 final）、`perf_opt/workload_inventory.json`、`perf_opt/perf_feedback.md`
- `perf_opt/_sparse_mla_kernel_r1_{p_ubpar,p_ubrow,p_contig}.py`（本轮实验分支；`p_contig` 标注数学无效）
- `perf_opt/parse_prof.py`（msprof 汇总脚本）、`perf_opt/logs/{r0,l0}/**`
- `perf_opt/profiles/{t_smoke,r1_p_*,r0_base_*,r0_final_*,r1_contig_*}/`（raw msprof）
- `perf_opt/history/`（上一轮工件原样归档）
- 未修改：`_sparse_mla_kernel.py`（基准）、`_sparse_mla_kernel_DESIGN.md`、wrapper、`tests/`、`benchmarks/`、`manifest/`、`DESIGN.md`
