# Stage 4 调优日志 — RopeLlama31Op（共享 kernel 采纳 + r3 采纳同步）

## 0. 来源与范围

本目录为**共享调优采纳副本**：RopeLlama31Op 的 `_make_rope_neox_1d` / `_make_rope_neox_2d`
与 `RopeNeoxOp` / `RopeLongRopeOp` 的对应文件**逐字节相同**
（baseline sha256 `80f4493a…` / `549f9d77…`）。Stage 4 只在规范位
`rope_neox/rope_neox_kernel/perf_opt/` 做**一次**调优（4 轮，完整迭代过程见该目录的
`opt_log.md`）；本文件只记录**属于本 op 的 workload** 的实测值与采纳结论。

当前采纳状态（2026-11-04 采纳同步后）：

- tuned 1d kernel：`perf_opt/_make_rope_neox_1d.py`
  **sha256 `c6098298b94beb664c4ff670f33a2b15926838a207a9a6b74b5a1bfb1a504f4a`**（规范位 r3 winner
  的逐字节拷贝；上一版 r2_contig `796840bc…` 已归档到
  `perf_opt/history/_make_rope_neox_1d_r2_contig.py`）。
- tuned 2d kernel：`perf_opt/_make_rope_neox_2d.py`，sha256
  `ab22c0254b20fa7d83ea527f7e2b2aff5fe96ece243052869e676cc3cfa320e4`（**本次未改动**；与规范位逐字节相同）。
- 工厂签名 drop-in 保持：`_make_rope_neox_1d(seq_len, head_dim, dtype)` → `kernel(block_size)`、
  `_make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype)` → `kernel(block_size)`；
  `block_size` 语义（f32=1024 / f16,bf16=2048）未改 → wrapper 切换块无需成对改默认值。
- 测量口径：`msprof op` `Task Duration(us)`（kernel-only，唯一口径）；`--launch-count=20
  --warm-up=5`，单 run 内取 10 条 launch 的中位数，跨 run 用 5-run 交错 A/B 中位数。设备
  Ascend910B3（20 AIC / 40 AIV）、CANN 9.0.0、tilelang `0.1.2+e1d97345` dev root build、
  torch_npu 2.7.1.post4、`TILELANG_ASCEND_MODE=Developer`。
- 精度：本 op 目录内直接复跑 —— tuned 1d `--level all` **44/44 PASS**、tuned 2d `--level all`
  **46/46 PASS**（0 精度 WARN / FAIL），日志见 §4.2。

## 1. Workload（本 op，来自 `benchmarks/ops/bench_rope_llama31.py::test_rope_llama31_bench`）

三层时延链（同一 kernel-only 口径，便于审计跨轮收益）：

| workload_id | shape / dtype | kind | 原始 baseline（Stage 3 kernel） | r2_contig（上一采纳版） | **r3_adopt（本次实测）** | r2→r3 |
|---|---|---:|---:|---:|---:|---:|
| `llama31-1d-8k-d128-bf16` | [8192, 128] bfloat16 | tune | 10.71 | 6.28 | **6.11** | **+2.7%** |
| `llama31-1d-2k-d128-fp16` | [2048, 128] float16 | tune | 5.29 | 3.98 | **3.96** | +0.5%（平区） |
| `llama31-2d-b1-s8k-h32-d128-fp16` | [1, 8192, 32, 128] float16 | tune | 211.74 | 81.43 | 81.43（未改动） | 0 |

- 原始 baseline（10.71 / 5.29）是规范位对 Stage 3 未调优 kernel 的 round-2 实测；r2_contig 列取自本 op
  r2 时代 final 记录（`history/perf_records_r2_all_kernels.jsonl`，`shared_measurement`：与规范位同一二进制
  + 同一 shape 几何）；r3_adopt 列是**本次采纳同步在本 op 目录内重新采集的 5-run 交错 A/B 中位数**（§4.3）。
- smoke：`llama31-1d-256-d64-smoke` [256,64] fp16/bf16/fp32 —— 仅精度回归（`precision_pass: true`），不采性能。

## 2. Performance Test Data（baseline）

本次采纳同步的 baseline = 本 op 上一版已采纳的 r2_contig
（`perf_opt/history/_make_rope_neox_1d_r2_contig.py`，sha256 `796840bc…`），与本 op 目录下的新 final
在同一 session 交错采集，raw profile 全部落在本目录 `perf_opt/profiles/`：

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir | command |
|---|---|---|---|---:|---:|---|---|---|
| `perf_opt/_make_rope_neox_1d.py::_make_rope_neox_1d::main` | llama31-1d-8k-d128-bf16 | r2_contig | main | main | 6.32 | valid | `profiles/baseline/round0/r2_contig_796840bc_run2/llama31-1d-8k-d128-bf16`（run1–run5 共 5 个） | `python perf_opt/_ab_adopt.py --rounds 5` |
| `perf_opt/_make_rope_neox_1d.py::_make_rope_neox_1d::main` | llama31-1d-2k-d128-fp16 | r2_contig | main | main | 3.96 | valid | `profiles/baseline/round0/r2_contig_796840bc_run1/llama31-1d-2k-d128-fp16`（run1–run5 共 5 个） | 同上 |

> 说明：`perf_opt/_profile_adopt.py` / `perf_opt/_ab_adopt.py` 是本次采纳同步使用的 msprof 驱动脚本
> （`_bench_kernel.py` 自规范位复制），只加载命令行指定的 kernel 文件，不经过 wrapper / benchmark；
> `perf_records.jsonl` 的每条记录都指向本目录真实 raw profile，不再跨目录借用测量值。
> 2d kernel 的 baseline/final 记录仍在 `perf_records_2d.jsonl`（本次未重测，见 §7）。

## 3. 候选 vs current best（本 op workload）

数据来自 `perf_records.jsonl` + raw profile 的 `OpBasicInfo_*.csv` / `PipeUtilization_*.csv`：

| candidate | workload_id | L0 | Task Duration(us) | aiv_vec / mte2 / mte3 ratio | mte2 带宽 GB/s | mte3 带宽 GB/s | vs current best(r2_contig) |
|---|---|---|---:|---|---:|---:|---:|
| `r2_contig`（current best，上一版采纳） | llama31-1d-8k-d128-bf16 | PASS | 6.32 | 0.650 / 0.535 / 0.156 | 40.9 | 70.0 | — |
| `r3_adopt` | llama31-1d-8k-d128-bf16 | PASS | 6.11 | 0.588 / 0.550 / 0.159 | 42.6 | 76.4 | **+3.3%** |
| `r2_contig`（current best，上一版采纳） | llama31-1d-2k-d128-fp16 | PASS | 3.96 | 0.349 / 0.243 / 0.081 | 50.6 | 71.4 | — |
| `r3_adopt` | llama31-1d-2k-d128-fp16 | PASS | 3.96 | 0.357 / 0.240 / 0.080 | 50.7 | 72.0 | 0.0%（平区） |

（current best 演进：`baseline` → `r1_balance`/`r1_contig` → `r2_contig` → **`r3_adopt`**。结构与归因详见
规范位 `opt_log.md`；r3 的结构变化见 §4.1。）

## 4. RopeLlama31Op 采纳 r3（adoption sync，2026-11-04）

本轮**不是新调优轮次**：r3 已在规范位（`rope_neox/perf_opt/`）完成调优、交错 A/B 裁决与 plateau 判定，
本 op 只做「逐字节采纳 + 本目录复测 + 记录刷新」（预算 `max_stage4_experiments=1`）。

### 4.1 采纳来源与结构差异

- 源文件：`rope_neox/rope_neox_kernel/perf_opt/_make_rope_neox_1d.py`，sha256 `c6098298…`；
  拷贝后本目录 `perf_opt/_make_rope_neox_1d.py` sha256 = `c6098298…`（`cmp` 逐字节一致，`CMP_OK`）。
- 相对 r2_contig 的**唯一**结构变化：fp32 中转 buffer 由 7 个降为 6 个（tile 推导项 `4*elem + 6*4`），
  使 tile 由 UB clamp 推导为 69 行（原 52 行），8k-d128 的波次由 4 波降为 3 波；
  其余三个 tune shape 的 tiling 与 op 序列逐字不变。该结论沿袭规范位已实证结果，本次不重做结构实验。
- 2d kernel `_make_rope_neox_2d.py` 与规范位逐字节相同（sha256 `ab22c025…`），**未改动**。

### 4.2 L0 / L1 回归（本 op 目录，采纳后实际执行的日志）

| kernel | 命令 | 结果 | 日志 |
|---|---|---|---|
| 1d（采纳后） | `python perf_opt/_make_rope_neox_1d.py --level all` | **44/44 PASS，0 精度 WARN/FAIL**（`All check passed!`，exit 0） | `perf_opt/logs/r3_adopt/l0_l1_1d.log` |
| 2d（未改动，确认无回归） | `python perf_opt/_make_rope_neox_2d.py --level all` | **46/46 PASS，0 精度 WARN/FAIL**（`All check passed!`，exit 0） | `perf_opt/logs/r3_adopt/l0_l1_2d.log` |

### 4.3 交错 A/B（本目录 5-run，`msprof op` Task Duration，单位 us）

| run | 8k-d128-bf16 r2_contig | 8k r3_adopt | 2k-d128-fp16 r2_contig | 2k r3_adopt |
|---:|---:|---:|---:|---:|
| 1 | 6.23 | 6.11 | 3.96 | 4.05 |
| 2 | 6.32 | 6.24 | 3.99 | 3.96 |
| 3 | 6.28 | 6.22 | 3.87 | 3.96 |
| 4 | 6.47 | 6.07 | 4.00 | 3.90 |
| 5 | 6.34 | 6.06 | 3.86 | 3.85 |
| **中位数** | **6.32** | **6.11** | **3.96** | **3.96** |

裁决：

- **8k-d128-bf16：r3 采纳**。5 个交错 pair 中 r3 全部快于 r2（5/5），中位数 6.32 → 6.11（+3.3%）；
  与规范位交错 A/B（6.37 → 6.15，+3.5%）方向与幅度一致。收益来源是波次 4→3，不是噪声。
- **2k-d128-fp16：平区**。中位数 3.96 → 3.96（0.0%），落在 ±4% 交错噪声带内，逐 pair 互有胜负
  （3 快 2 慢），且该 shape 的 tiling/op 序列逐字不变；记为 tie，不构成独立收益，也不构成回退。
- 两个 shape 的 merge 结论一致（同一份 kernel 文件同时覆盖），无需条件路径。

### 4.4 采纳动作清单

1. `perf_opt/_make_rope_neox_1d.py` ← 规范位 r3 逐字节拷贝（sha256 `c6098298…`，`cmp` 通过）。
2. 旧版归档 `perf_opt/history/_make_rope_neox_1d_r2_contig.py`（sha256 `796840bc…`）。
3. 旧 `perf_records.jsonl`（r2 时代 6 行，含当时混入的 2d 行）归档到
   `perf_opt/history/perf_records_r2_all_kernels.jsonl`；新 `perf_records.jsonl` 只保留本 op 的 1d
   tune workload（2 baseline + 2 final），final 记录的 `candidate_id` 统一为 `r3_adopt`，
   `artifact_path` 指向本目录 `perf_opt/_make_rope_neox_1d.py`。
4. `workload_inventory.json`：`kernel_id` 统一改为 `perf_opt/_make_rope_neox_1d.py`（与
   `perf_records.jsonl` 同一路径，消除此前的 inventory/records 路径漂移），tune 项 `workload_id`
   对齐 manifest label（`llama31-1d-*`），`tuning_status` 复核为 `winner_merged`。
5. baseline kernel（根部 `_make_rope_neox_{1d,2d}.py`）与 wrapper `rope_llama31.py` **未改动**。

## 5. Autotune Log

未做 autotune：共享内核的搜索空间（tile 行数 / core_num / block_size）已在规范位手搜 + 交错 A/B 收敛
（见规范位 `opt_log.md`）；本次采纳同步不重开搜索。

## 6. 合并非回退检查（本 op 全部 tune workload + smoke）

| kernel | workload | 检查 | 结论 |
|---|---|---|---|
| 1d | llama31-1d-8k-d128-bf16 | 精度 L0 PASS；性能 6.32 → 6.11（+3.3%） | 采纳 |
| 1d | llama31-1d-2k-d128-fp16 | 精度 L0 PASS；性能 3.96 → 3.96（±4% 平区） | 不回退，随合并版本带上 |
| 1d | llama31-1d-256-d64-smoke（fp16/bf16/fp32） | `--level all` PASS ×3（合计 44/44） | 精度不回退 |
| 2d | llama31-2d-b1-s8k-h32-d128-fp16 | 未改动，`--level all` 46/46 PASS | 不变 |

## 7. 多 kernel 记录拆分说明

本 op 同时交付两个独立 kernel，而 gate 4 的 final-artifact 对账为单文件口径，故按 kernel 拆分记录
（纯记录层，未改任何测量值）：

- 1d（对账集合）：`perf_records.jsonl`（4 行：2 baseline + 2 final，候选 `r3_adopt`）+
  `workload_inventory.json`（5 个 workload：3 smoke + 2 tune）。
- 2d：`perf_records_2d.jsonl`（3 行，本次未重测、未改写）+ `workload_inventory_2d.json`（1 个 workload）。
- 历史归档：`history/perf_records_r2_all_kernels.jsonl`（r2 时代 6 行）；
  `perf_records_all.audit.jsonl` 为 r2 时代 9 行审计快照（原样保留，不参与 gate 对账）。

### 2d kernel final（记录源 `perf_records_2d.jsonl`）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/position_encoding/rope_llama31/rope_llama31_kernel/_make_rope_neox_2d.py::_make_rope_neox_2d::main | llama31-2d-b1-s8k-h32-d128-fp16 | 211.74 | 81.43 | final_contig |

（该表来自未改动的 `perf_records_2d.jsonl`，`kernel_id` 沿用 r2 时代记录值；2d 文件 sha256 仍为
`ab22c025…`，未纳入本次最终候选的对账集合。）

## 8. Skill Retrospective

- **流程问题（本 op 上一轮遗留，本次修复）**：上一轮「共享 kernel 采纳」把规范位的 `perf_records.jsonl`
  拷回本 op 后，`workload_inventory.json` 的 `kernel_id` 仍指向 baseline 路径、tune `workload_id` 仍是
  `longrope-*`，而 records 用 `perf_opt/` 路径 + `llama31-*`，gate 4 长期处于
  `S4-PERF-RECORDS-COVERAGE` 失败（本次开工时复现 10 处 failure + 1 处 opt_log 缺复盘章节）。
  结论：**采纳同步必须把 gate 4 自检纳入动作清单**，且 inventory 与 records 的 `kernel_id` /
  `workload_id` 必须同源生成。
- **流程问题（测量来源）**：共享 kernel 复用规范位测量在物理上成立，但 `artifact_path` 指本 op 文件而
  `duration_us` 来自别目录，会让「记录可对账」退化为「靠标注可信」。本次改为本目录重跑 20 个 msprof
  run（`_ab_adopt.py`），消除该歧义，成本约 25 分钟。
- **VP 候选（P）**：6 µs 级 kernel 单 run `msprof op` 存在 ±3–5% 双态抖动（本次 r2_contig 8k 出现
  6.23 与 6.47 两态），<5% 差异必须用交错多 run 中位数裁决；「采纳同步」也适用 T-2 协议，不只是新候选。
  证据：§4.3 五 run 表；repro：`perf_opt/_ab_adopt.py`。
- **VP 候选（R）**：建议 gate 4 显式校验 `workload_inventory.json` 与 `perf_records.jsonl` 的
  `kernel_id` 路径一致（现状是靠 coverage 间接暴露），并在 `perf_records_lint` 报错信息中直接给出
  「inventory 有而 records 无」的差集，减少采纳同步类任务的清账往返。

## 9. 门禁自检记录（gate 4）

- 命令：`python .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/position_encoding/rope_llama31/rope_llama31_kernel`
- 结论：`pass: true`，`failures: []`（详见本轮执行返回）。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/position_encoding/rope_llama31/rope_llama31_kernel/perf_opt/_make_rope_neox_1d.py::_make_rope_neox_1d::main | llama31-1d-8k-d128-bf16 | 6.32 | 6.11 | r3_adopt |
| examples/TileOPs/tileops/kernels/position_encoding/rope_llama31/rope_llama31_kernel/perf_opt/_make_rope_neox_1d.py::_make_rope_neox_1d::main | llama31-1d-2k-d128-fp16 | 3.96 | 3.96 | r3_adopt |

最终候选 = `perf_opt/_make_rope_neox_1d.py`（sha256 `c6098298…`）+ `perf_opt/_make_rope_neox_2d.py`
（sha256 `ab22c025…`，未改动）。baseline_us = 本 op 上一版采纳的 r2_contig 在同一 session 的交错
5-run 中位数；final_us = 本次采纳的 r3 在同表的中位数。smoke（`llama31-1d-256-d64-smoke` ×
fp16/bf16/fp32）只做最终精度回归并全部 PASS，不进入性能表；skip 集合为空。

中止原因：`success`（采纳已验证 winner：8k-d128 实测 +3.3%，2k-d128 平区不退化；共享调优侧已
plateau：1d 读/写 1.6–2.0 / 2.6–2.9 TB/s，贴带宽地板）。未触发 `[DESIGN_LIMIT]`。
