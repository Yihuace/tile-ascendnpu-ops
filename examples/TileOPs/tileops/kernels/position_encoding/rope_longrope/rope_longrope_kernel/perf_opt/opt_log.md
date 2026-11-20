# Stage 4 调优日志 — RopeLongRopeOp（共享 kernel 采纳 + r3 采纳同步）

## 0. 来源与范围

本目录为**共享调优采纳副本**：RopeLongRopeOp 的 `_make_rope_neox_1d` / `_make_rope_neox_2d`
与 `RopeNeoxOp` / `RopeLongRopeOp` / `RopeLlama31Op` 的对应文件**逐字节相同**
（baseline sha256 `80f4493a…` / `549f9d77…`）。Stage 4 只在规范位
`rope_neox/rope_neox_kernel/perf_opt/` 做**一次**调优（4 轮），完整迭代过程见该目录的
`opt_log.md`；本文件只记录**属于本 op 的 workload** 的实测值与采纳结论。

当前采纳状态（2026-11-04 采纳同步后）：

- tuned 1d kernel：`perf_opt/_make_rope_neox_1d.py`
  **sha256 `c6098298b94beb664c4ff670f33a2b15926838a207a9a6b74b5a1bfb1a504f4a`**（规范位 r3 winner
  的逐字节拷贝；上一版 r2_contig `796840bc…` 已归档到 `perf_opt/history/_make_rope_neox_1d_r2_contig.py`）。
- tuned 2d kernel：`perf_opt/_make_rope_neox_2d.py`，sha256
  `ab22c0254b20fa7d83ea527f7e2b2aff5fe96ece243052869e676cc3cfa320e4`（**本次未改动**；与规范位逐字节相同）。
- 工厂签名 drop-in 保持：`_make_rope_neox_1d(seq_len, head_dim, dtype)` → `kernel(block_size)`、
  `_make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype)` → `kernel(block_size)`；
  `block_size` 语义（f32=1024 / f16,bf16=2048）未改 → wrapper 切换块无需成对改默认值。
- 测量口径：`msprof op` `Task Duration(us)`（kernel-only，唯一口径）；`--launch-count=20
  --warm-up=5`，单 run 内取 10 条 launch 的中位数。设备 Ascend910B3（20 AIC / 40 AIV）、
  CANN 9.0.0、tilelang `0.1.2+e1d97345` dev root build、`TILELANG_ASCEND_MODE=Developer`。
- 精度：本 op 目录内直接复跑 —— tuned 1d `--level all` **44/44 PASS**、tuned 2d `--level all`
  **46/46 PASS**（无精度 WARN / FAIL），日志见 §4.2。

## 1. Workload（本 op，来自 `benchmarks/ops/bench_rope.py::test_rope_longrope_bench`）

三层时延链（同一 kernel-only 口径，便于审计跨轮收益）：

| workload_id | shape / dtype | kind | 原始 baseline（Stage 3 kernel） | r2_contig（上一采纳版） | **r3_adopt（本次）** | r2→r3 |
|---|---|---:|---:|---:|---:|---:|
| `longrope-1d-8k-d128-bf16` | [8192, 128] bfloat16 | tune | 10.71 | 6.29 | **6.15** | **+2.2%** |
| `longrope-1d-2k-d128-fp16` | [2048, 128] float16 | tune | 5.29 | 3.87 | **3.91** | −1.0%（平区） |
| `longrope-2d-b1-s8k-h32-d128-fp16` | [1, 8192, 32, 128] float16 | tune | 211.74 | 81.43 | 81.43（未改动） | 0 |

- 原始 baseline（10.71 / 5.29）是规范位 round-2 对 Stage 3 未调优 kernel 的实测；r2_contig 与
  r3_adopt 两列是**本次采纳同步在本 op 目录内重新采集的 5-run 交错 A/B 中位数**（§4.3），
  不再是跨目录借用测量值。
- smoke：`longrope-1d-256-d64-smoke` [256,64] fp16/bf16/fp32 —— 仅精度回归，不采性能。

## 2. Performance Test Data（baseline）

本次采纳同步的 baseline = 本 op 上一版已采纳的 r2_contig（`perf_opt/history/_make_rope_neox_1d_r2_contig.py`，
sha256 `796840bc…`），与本 op 目录下的新 final 在同一 session 交错采集，raw profile 全部落在本目录：

| kernel_id | workload_id | candidate_id | task_duration_us | profile_status | raw_profile_dir | command |
|---|---|---|---:|---|---|---|
| `_make_rope_neox_1d.py::_make_rope_neox_1d::main` | longrope-1d-8k-d128-bf16 | r2_contig | 6.29 | valid | `profiles/baseline/round0/r2_contig_796840bc/longrope-1d-8k-d128-bf16` 与 `…_run2…run5/longrope-1d-8k-d128-bf16`（共 5 个 run） | `python perf_opt/_profile_adopt.py perf_opt/history/_make_rope_neox_1d_r2_contig.py --out-subdir baseline/round0/r2_contig_796840bc` |
| `_make_rope_neox_1d.py::_make_rope_neox_1d::main` | longrope-1d-2k-d128-fp16 | r2_contig | 3.87 | valid | `profiles/baseline/round0/r2_contig_796840bc/longrope-1d-2k-d128-fp16` 与 `…_run2…run5/…`（共 5 个 run） | 同上 |

> 说明：`perf_opt/_profile_adopt.py` 是本次采纳同步新增的 msprof 驱动脚本（`_bench_kernel.py` 同步复制到本目录），
> 只加载命令行指定的 kernel 文件，不经过 wrapper / benchmark；`perf_records.jsonl` 中每条记录都指向本目录真实 raw profile。
> 2d kernel 的 baseline/final 记录仍在 `perf_records_2d.jsonl`（本次未重测，见 §6）。

## 3. 候选 vs current best（本 op workload）

数据结构来自 `perf_records.jsonl` + raw profile 的 `OpBasicInfo_*.csv` / `PipeUtilization_*.csv`：

| candidate | workload_id | L0 | Task Duration(us) | vec/mte2/mte3 ratio | mte2 带宽 GB/s | mte3 带宽 GB/s | vs current best(r2_contig) |
|---|---|---|---:|---|---:|---:|---:|
| `r2_contig`（current best，上一版采纳） | longrope-1d-8k-d128-bf16 | PASS | 6.29 | 0.660 / 0.534 / 0.159 | 41.5 | 69.7 | — |
| `r3_adopt` | longrope-1d-8k-d128-bf16 | PASS | 6.15 | 0.575 / 0.536 / 0.151 | 42.8 | 78.3 | **+2.2%** |
| `r2_contig`（current best，上一版采纳） | longrope-1d-2k-d128-fp16 | PASS | 3.87 | 0.363 / 0.241 / 0.083 | 52.1 | 72.4 | — |
| `r3_adopt` | longrope-1d-2k-d128-fp16 | PASS | 3.91 | 0.368 / 0.248 / 0.083 | 50.8 | 72.0 | −1.0%（平区） |

（current best 演进：`baseline` → `r1_balance`/`r1_contig` → `r2_contig` → **`r3_adopt`**。r1/r2 的结构
= UB 预算 tile 平衡 + 整行连续 DMA + UB 半段切分/合并；r3 的结构变化见 §4.1。）

## 4. RopeLongRopeOp 采纳 r3（adoption sync，2026-11-04）

本轮**不是新调优轮次**：r3 已在规范位（`rope_neox/perf_opt/`）完成调优、交错 A/B 裁决与 plateau 判定，
本 op 只做「逐字节采纳 + 本目录复测 + 记录刷新」。

### 4.1 采纳来源与结构差异

- 源文件：`rope_neox/rope_neox_kernel/perf_opt/_make_rope_neox_1d.py`，sha256 `c6098298…`；
  拷贝后本目录 `perf_opt/_make_rope_neox_1d.py` sha256 = `c6098298…`（`cmp` 逐字节一致）。
- 相对 r2_contig 的**唯一**结构变化（`diff` 全文仅 3 处）：
  1. `_UB_BUDGET_BYTES` 150 KB → 176 KB；
  2. tile 推导 `bytes_per_row` 的 fp32 中转项 7 → 6（`4*elem + 6*4`），即 fp32 中转 buffer 7 → 6；
  3. 半段旋转换序：先把两个交叉乘积提到前面（`t1=a*s`, `t2=b*s`），使 fp32 输入 buffer 自身可复用为输出
     （4 输入 + 2 临时 tile，替代 4 + 3）。
- 效果（已实证，本次未推翻）：fp32 中转从 7 降到 6 后 tile 由 UB clamp 推导为 69 行（原 52 行），
  8k-d128 的波次从 4 降到 3；其余三个 tune shape 的 tiling 与 op 序列逐字不变。

### 4.2 L0 / L1 回归（本 op 目录，采纳后实际执行的日志）

| kernel | 命令 | 结果 | 日志 |
|---|---|---|---|
| 1d（采纳后） | `python perf_opt/_make_rope_neox_1d.py --level all` | **44/44 PASS，0 WARN/FAIL**（`All check passed!`，exit 0） | `perf_opt/logs/r3_adopt/l0_l1_1d.log` |
| 2d（未改动，确认无回归） | `python perf_opt/_make_rope_neox_2d.py --level all` | **46/46 PASS，0 WARN/FAIL**（`All check passed!`，exit 0） | `perf_opt/logs/r3_adopt/l0_l1_2d.log` |

### 4.3 交错 A/B（本目录 5-run，`msprof op` Task Duration，单位 us）

| run | 8k-d128-bf16 r2_contig | 8k r3_adopt | 2k-d128-fp16 r2_contig | 2k r3_adopt |
|---:|---:|---:|---:|---:|
| 1 | 6.40 | 6.27 | 3.86 | 3.97 |
| 2 | 6.16 | 6.15 | 3.87 | 3.85 |
| 3 | 6.36 | 6.18 | 3.83 | 3.98 |
| 4 | 6.25 | 5.97 | 3.91 | 3.91 |
| 5 | 6.29 | 6.08 | 3.89 | 3.88 |
| **中位数** | **6.29** | **6.15** | **3.87** | **3.91** |

裁决：

- **8k-d128-bf16：r3 采纳**。5 个交错 pair 中 r3 全部不慢于 r2（5/5），中位数 6.29 → 6.15（+2.2%）；
  与规范位交错 A/B 6.37 → 6.15（+3.5%）方向一致。收益来源是波次 4→3，不是噪声。
- **2k-d128-fp16：平区（不改动前的持平搬移）**。中位数 3.87 → 3.91（−1.0%），落在规范位记录的
  ±4% 交错噪声带内，且该 shape 的 tiling/op 序列逐字不变；规范位同口径为 3.88 → 3.89（差 0.3%）。
  按「<5% 差异须交错多 run 决胜」的口径，本 shape 记为 tie，不构成独立收益，也不构成回退。
- 两个 shape 的 merge 结论一致（同一份 kernel 文件同时覆盖），无需条件路径。

### 4.4 采纳动作清单

1. `perf_opt/_make_rope_neox_1d.py` ← 规范位 r3 逐字节拷贝（sha256 `c6098298…`，`cmp` 通过）。
2. 旧版归档 `perf_opt/history/_make_rope_neox_1d_r2_contig.py`（sha256 `796840bc…`）。
3. 旧 `perf_records.jsonl`（r2 时代 6 行，含当时混入的 2d 行）归档到
   `perf_opt/history/perf_records_r2_all_kernels.jsonl`；新 `perf_records.jsonl` 只保留本 op 的 1d
   tune workload（2 baseline + 2 final），final 记录的 `candidate_id` 统一为 `r3_adopt`。
4. `workload_inventory.json` 的 2 个 tune 项复核为 `tuning_status=winner_merged`。
5. baseline kernel（`_make_rope_neox_{1d,2d}.py` 根部文件）与 wrapper `rope_longrope.py` **未改动**。

## 5. Autotune Log

未做 autotune：共享内核的搜索空间（tile 行数 / core_num / block_size）已在规范位 round-1~r3 手搜 + 交错 A/B
收敛（见规范位 `opt_log.md` §4）；本次采纳同步不重开搜索，`max_stage4_experiments=1` 仅用于本次采纳动作。

## 6. 合并非回退检查（本 op 全部 tune workload + smoke）

| kernel | workload | 检查 | 结论 |
|---|---|---|---|
| 1d | longrope-1d-8k-d128-bf16 | 精度 L0 PASS；性能 6.29 → 6.15 | 采纳 |
| 1d | longrope-1d-2k-d128-fp16 | 精度 L0 PASS；性能 3.87 → 3.91（±4% 平区） | 不回退，随合并版本带上 |
| 1d | longrope-1d-256-d64-smoke（fp16/bf16/fp32） | `--level all` PASS ×3 | 精度不回退 |
| 2d | longrope-2d-b1-s8k-h32-d128-fp16 | 未改动，`--level all` 46/46 PASS | 不变 |

## 7. 多 kernel 记录拆分说明

本 op 同时交付两个独立 kernel，而 gate 4 的 final-artifact 对账为单文件口径，故按 kernel 拆分记录
（纯记录层，未改任何测量值）：

- 1d（对账集合）：`perf_records.jsonl`（4 行：2 baseline + 2 final，candidate `r3_adopt`）+ `workload_inventory.json`（5 个 workload，2 tune）。
- 2d：`perf_records_2d.jsonl`（3 行）+ `workload_inventory_2d.json`（1 个 workload）；本次未重测。
- 历史归档：`history/perf_records_r2_all_kernels.jsonl`（r2 时代 6 行）；`perf_records_all.audit.jsonl`
  为 r2 时代 9 行审计快照（原样保留，不参与 gate 对账）。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/position_encoding/rope_longrope/rope_longrope_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | longrope-1d-8k-d128-bf16 | 6.29 | 6.15 | r3_adopt |
| examples/TileOPs/tileops/kernels/position_encoding/rope_longrope/rope_longrope_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | longrope-1d-2k-d128-fp16 | 3.87 | 3.91 | r3_adopt |

最终候选 = `perf_opt/_make_rope_neox_1d.py`（sha256 `c6098298…`）+ `perf_opt/_make_rope_neox_2d.py`
（sha256 `ab22c025…`，未改动）。baseline_us 为本 op 上一版采纳的 r2_contig 在同 session 的交错中位数。
smoke（`longrope-1d-256-d64-smoke` × fp16/bf16/fp32）只做最终精度回归并全部 PASS，不进入性能表；
skip 集合为空。

中止原因：`success`（采纳已验证 winner，r3 在 8k 上有 2.2% 实测收益，其余 shape 平区不退化；
共享调优侧已 plateau：1d 读/写 1.6–2.0 / 2.6–2.9 TB/s，贴带宽地板）。未触发 `[DESIGN_LIMIT]`。

## 8. Skill Retrospective

- **流程问题（本次暴露）**：上一轮「共享 kernel 采纳」把规范位的 `perf_records.jsonl` 直接改写后拷回本 op，
  导致 `workload_inventory.json` 的 `kernel_id`（baseline 路径）与记录里的 `kernel_id`（`perf_opt/` 路径）
  不一致，gate 4 从采纳当轮起就处于 `S4-PERF-RECORDS-COVERAGE` 失败状态（本次开工时复现，6 处失败）。
  采纳同步类动作必须自检 gate 4，而不是只跑 L0/L1。
- **流程问题（跨 op 借用测量）**：共享 kernel 用「同一二进制 + 同一 shape 几何」复用规范位测量在物理上成立，
  但把 `artifact_path` 指向本 op 文件、`duration_us` 却来自别目录的 run，会让「记录可对账」退化为「靠标注可信」。
  本次改为在本目录重跑 `msprof op`（`_profile_adopt.py`），成本约 12 个 run，可完全消除该歧义。
- **VP 候选（P）**：单 run `msprof op` 在 6 µs 级 kernel 上有 ±3–5% 双态抖动（本次 r2_contig 8k 出现
  6.16 与 6.40 两态），<5% 差异必须用交错多 run 中位数裁决；建议把「采纳同步」纳入 T-2 协议适用范围
  （不是只有新候选才需要）。证据：§4.3 五 run 表；repro：`perf_opt/_profile_adopt.py`。
- **VP 候选（R）**：共享 kernel 的 `workload_inventory.json` 与 `perf_records.jsonl` 的 `kernel_id`
  必须由同一生成脚本写出（当前是手写拼接，易漂移）；建议 gate 4 增加「inventory/records 的 kernel_id
  路径互不一致时直接报错」的显式 rule（现状是间接由 coverage 失败暴露）。

## 9. 门禁自检记录（gate 4）

- 命令：`python .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/position_encoding/rope_longrope/rope_longrope_kernel`
- 结论：`pass: true`，`failures: []`（详见本轮执行返回）。
