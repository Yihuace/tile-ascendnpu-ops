# Stage 4 Optimization Log -- `_persistent_grouped_gemm_v2_kernel`

- project: `grouped_gemm_persistent_3wg` / family `moe` / op `_persistent_grouped_gemm_v2_kernel`
- 服务的上层算子：`FusedMoEExpertsNopadPersistent3WGFwdOp`、`FusedMoeFwdOp`（两者共用本 kernel；Ratio 均出自 msprof 的 `main_mix_aic`）
- baseline（只读，Stage 3）：`_persistent_grouped_gemm_v2_kernel.py`
- perf_opt artifact（定稿，**Round 8 起**）：`perf_opt/GroupedGemmPersistent3WG.py`
  （sha256 `72fbb33e3bed75296abdf09ede04abbb582afb859cbcd2625646748a3f7d500a`）。
  Round 7 定稿（sha `e3babaa6…f157ce5`）原样留档在 `history_version/20261026_prefillA/` 作回退；
  v1 定稿 `perf_opt/_persistent_grouped_gemm_v2_kernel.py`（sha `75877d4b…13d793`）同样留档。
- benchmark entry：`examples/TileOPs/benchmarks/ops/bench_fused_moe_experts.py::test_fused_moe_experts_nopad_persistent_3wg_bench`
- latency oracle（唯一口径）：`msprof op` 的 `Task Duration(us)`
- 编程模式：**Expert**（本 kernel 用 `T.Scope("Cube")` + persistent，`ascend_mode="Expert"`）；device Ascend910_9362
- 停止原因（Round 8）：**目标部分达成** —— 4 条 tune case 相对 Round 7 再降 5.7%~38.7%，其中 3 条
  Ratio ≥ 60%（qwen3-decode 61.30%、qwen3-prefill 66.03%、deepseek-decode 61.15%）；deepseek-v3-prefill
  58.26% 仍差 2.9% 时延。剩余缺口是 MTE2 端口字节在 `block_m·block_n ≤ 32768`（L0C 128 KB）下的结构下界
  ＋ 11.8% 端口空闲，实测上界 ≈1.13×（< 2×）⇒ 不构成设计层天花板，不产出 `perf_feedback.md`。
  详细量化见 §8.8。
- **本文件当前的 authoritative 结论在 §8（Round 8）**；§7 = Round 7（D2 双缓冲 + `block_n` 形状策略），
  §1~§5 是 v1（Round 0-3）的历史记录，§6 是 Round 7 前的三方向 go/no-go 探针。

---

## 1. Performance Test Data（baseline，round 0）

| kernel_id | workload_id | candidate_id | task_duration_us | prof_mode |
|---|---|---|---:|---|
| _persistent_grouped_gemm_v2_kernel | qwen3-235b-decode | baseline | 14040.7 | msprof |
| _persistent_grouped_gemm_v2_kernel | qwen3-235b-prefill | baseline | 36287.3 | msprof |
| _persistent_grouped_gemm_v2_kernel | deepseek-v3-decode | baseline | 39856.0 | events（无 Ratio） |
| _persistent_grouped_gemm_v2_kernel | deepseek-v3-prefill | baseline | 74085.0 | events（无 Ratio） |

Smoke（排除出性能调优，仅精度回归）：`fused-moe-experts-3wg-smoke` 的 `bfloat16` / `float16` 两个变体（manifest label 含独立 `smoke` token → `kind=smoke`；`precision_pass=true`）。

---

## 2. Iteration Log

### Round 0 -- 诊断

该 kernel 是 **Cube-only persistent 分组 GEMM**（`T.Scope("Cube")`、`T.serial` K 循环、host 侧调度表、no-pad 动态 extent）。基线 tiling 经 wrapper 生效值为 `block_m=128 / block_n=256 / block_k=64`。Profile 显示访存项仅 ~8%、算力项 24~25% → **算力受限**，且 **K 循环无搬运-计算重叠**；DESIGN 明确把 `num_stages` 留给 Stage 4 的 `T.Pipelined`。

### Round 1 -- tiling 扫描（独立探针，逐 GEMM 单测；`msprof op` Task Duration）

为避免整轮 harness 的开销（一轮 ~12 min），本轮自建探针直接驱动该子 kernel（`numel=4096, E=128`，即 qwen3-decode 形状）。**探针与 harness 交叉标定**：探针测 gate_up（`128/256/64`）= **14011.5 µs**，harness 同配置报 **14011.6 µs** → 逐位一致，说明探针可替代 harness 做调参。

**gate_up GEMM**（`N=4096, K=7168`）：

| candidate (bm,bn,bk) | Task Duration (µs) | 相对 current best |
|---|--:|--:|
| **(128,256,64) ← current best (baseline)** | **14037.7** | — |
| (128,256,128) | 10562.1 | −24.8% |
| (128,256,256) | 10335.2 | −26.4% |
| (256,128,64) | 17317.8 | +23.4%（更差）|
| (256,128,256) | 12906.7 | −8.1% |
| **(128,128,256)** | **10201.3** | **−27.3%** ← 新 current best |
| (64,256,256) | 10972.6 | −21.8% |

**down GEMM**（`N=7168, K=2048`）：

| candidate (bm,bn,bk) | Task Duration (µs) | 相对 current best |
|---|--:|--:|
| (128,256,64) ← current best (baseline) | 6911.8 | — |
| (128,256,256) | 5006.9 | −27.6% |
| **(128,128,256)** | **4969.7** | **−28.1%** ← 新 current best |
| (64,256,256) | 5264.3 | −23.8% |

**结论**：`block_k` 是主旋钮（64→256 约 −26%），`block_n=128` 在 `block_k=256` 时再小幅胜出；`block_m=256` 明显变差。两个 GEMM 一致选中 **`(128,128,256)`**。

### Round 2 -- 验证 DESIGN 预留的流水杠杆（`T.Pipelined`）

按 DESIGN 提示，把 K 循环从 `T.serial` 换成 `T.Pipelined(..., num_stages=N)`（因 wrapper 强制 `K % block_k == 0`，改用静态 `block_k` extent；M/N 的动态 extent 保持在循环外），存为 `perf_opt/_persistent_grouped_gemm_v2_kernel_r1_pipelined.py`：

| 变体 | Task Duration (µs) | 结论 |
|---|--:|---|
| serial（current best） | 10201.3 | — |
| pipelined `num_stages=2` | 10200.0 | **等价**（±0.01%）|
| pipelined `num_stages=3` | 10209.5 | 等价 |
| pipelined `(128,256,256) ns=2` | 10353.0 | 与 serial 的 (128,256,256)=10335.2 等价 |

**结论**：本 kernel 上 `T.Pipelined` **无增益**（搬运未被软件流水掩盖，或硬件已自行重叠）→ **保留 serial 主体**，不改内核结构（同时也避免丢掉非对齐 K 的兜底能力）。此负向结论与 GemmOp 的 `num_stages` 1~4 等价一致。

### Round 3 -- 权威复测（harness 全量，定稿 = 仅改 wrapper 配置 `block_n=128 / block_k=256`）

单次 harness 全会话运行约 12 min，两次复测中**各有一处**因测量侧抖动回退/失败（一次 2 条 case OOM、一次 `qwen3-decode` 回退 events；单独重跑基准文件 6/6 通过，证明与本次改动无关）。因此对每个 workload 取其 **msprof** 结果：

| workload | baseline (µs) | final (µs) | 加速 | Ratio 基线 | Ratio 定稿 | msprof run |
|---|--:|--:|--:|--:|--:|---|
| qwen3-235b-decode | 14040.7 | 10221.1 | **1.37×** | 25.15% | **42.94%** | 162140 |
| qwen3-235b-prefill | 36287.3 | 30199.6 | **1.20×** | 23.95% | **40.58%** | 164351 |
| deepseek-v3-decode | 39856.0（events） | 18199.9 | 口径不可比 | N/A | **47.11%** | 164351 |
| deepseek-v3-prefill | 74085.0（events） | 38133.3 | 口径不可比 | N/A | **37.72%** | 162140 |

**L0/L1 精度回归**：`tests/ops/test_fused_moe_experts.py` 20/20 ✓（含 smoke 与 EP 变体）；smoke 快速子集 7/7 ✓。

### 副产发现（重要）：DeepSeek 的 `[B3]` 采集失败在本配置下消失

基线两次都拿不到 DeepSeek 的 Ratio（msprof 的 `CreateSnapshot ... ret=207001` / `memoryContext failed` → 回退 events）。改 `block_n=256→128`（L0C tile 128×128，工作集减半）后，**DeepSeek 两条均成功产出 msprof 数据与 Ratio**（47.11% / 37.72%）。这与"快照失败由测量期设备内存压力触发"的判断一致：**缩小片上工作块虽不改输入体积，但降低了采集时的设备压力**。建议上报为工具链侧观察项，供对方补测其 4 条缺失 Ratio 时参考。

---

## 3. 结果小结

- **ratio 口径下无一条 ≥60%**（37.7%~47.1%），但相对基线**全部显著提升**：qwen3-decode **+17.8pp（1.37×）**、qwen3-prefill **+16.6pp（1.20×）**；DeepSeek 两条从"无口径"变为**可测**（47.11% / 37.72%）。
- 提升来自**唯一一处改动**：wrapper 配置 `block_n 256→128`、`block_k 64→256`（内核文件未改结构）。
- **未采纳**：`T.Pipelined`（等价，无增益）、`block_m=256`（更差）、`block_n=256 @ block_k=256`（略差）。
- 剩余差距（37~47% vs 60%）属结构性：no-pad 动态 extent + 单 Cube scope + persistent 串行任务序的组合下，M/N 尾块与专家边界导致部分 tile 欠载；进一步需要改分核/调度方案（设计层），不在本 Stage 4 授权内。

---


### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|--:|--:|---|
| tileops/kernels/moe/grouped_gemm_persistent_3wg/grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py::_persistent_grouped_gemm_v2_kernel::main | deepseek-v3-decode-bfloat16 | 39855.958 | 13512.120 | v3_strided_taskindex |
| tileops/kernels/moe/grouped_gemm_persistent_3wg/grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py::_persistent_grouped_gemm_v2_kernel::main | deepseek-v3-prefill-bfloat16 | 74085.015 | 18207.779 | v3_strided_taskindex |
| tileops/kernels/moe/grouped_gemm_persistent_3wg/grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py::_persistent_grouped_gemm_v2_kernel::main | qwen3-235b-decode-bfloat16 | 14040.739 | 6907.320 | v3_strided_taskindex |
| tileops/kernels/moe/grouped_gemm_persistent_3wg/grouped_gemm_persistent_3wg_kernel/_persistent_grouped_gemm_v2_kernel.py::_persistent_grouped_gemm_v2_kernel::main | qwen3-235b-prefill-bfloat16 | 36287.273 | 13139.720 | v3_strided_taskindex |

> 本表的 `baseline_us` 是 Stage 3 baseline（round 0），`final_us` 是 **Round 8** 的定稿候选
> `v3_strided_taskindex`（同一最终文件 `perf_opt/GroupedGemmPersistent3WG.py`
> sha256 `72fbb33e3bed75296abdf09ede04abbb582afb859cbcd2625646748a3f7d500a` 的 `phase=final`
> 记录，见 `perf_records.jsonl`，口径 = op 级 `msprof op` `main_mix_aic` Task Duration）。
> Round 7 的 `v2_d2_pair_bnshape` 数值（14331.974 / 26215.334 / 7849.503 / 21428.191）仍完整出
> 现在 `perf_records.jsonl`，其 `phase` 已按 append-only 纪律降级为 `merged`（只改标签、不改任何
> 测量值），以便「每个 workload 恰有一条当前最终版本的 final 记录」。
> v1（Round 1-3）的 `final_tuned_128x128x256` 数值仍完整保留在 `perf_records.jsonl`，其 `phase`
> 已按 append-only 纪律降级为 `merged`（只改标签、不改任何测量值），以便「每个 workload 恰有一条
> 当前最终版本的 final 记录」。

## 4. Skill Retrospective

- **探针 + 交叉标定再次奏效**：自建探针直接驱动子 kernel，测 gate_up `128/256/64` = 14011.5 µs vs harness 14011.6 µs → 逐位一致。**先标定再扫参**，把一轮 12 min 的 harness 换成 ~1 min/配置的探针。
- **"设计预留的杠杆"未必真有效**：DESIGN 明确把 `num_stages` 留给 `T.Pipelined`，但实测**完全等价**。教训：预留不等于收益，须实测；负向结论要写进日志（本次已写，并保留实验件）。
- **复合算子的"延迟"口径要看清**：两个上层算子的 KPI 数字（14040.7 / 14011.6 µs）其实就是**其中一个子 GEMM**（gate_up）的耗时——探针一比即知。报告必须标注"Ratio 只覆盖 `main_mix_aic`"。
- **测量侧抖动**：同一配置的两次 harness 运行，各有一条 case 回退/OOM（单独重跑基准 6/6 通过），说明抖动来自测量环境而非 kernel。对"边缘达标"结论须重复测量（本报告已按两次复测取 msprof 结果并标明 run）。
- **减小片上工作块可能顺带解决采集失败**：`block_n` 减半后 DeepSeek 的 msprof 快照不再失败（见 §2 副产发现）。

---

## 5. 产物

- 定稿内核（仅默认值调整 + 注释）：`perf_opt/_persistent_grouped_gemm_v2_kernel.py`
- 实验件（留档）：`perf_opt/_persistent_grouped_gemm_v2_kernel_r1_pipelined.py`
- wrapper：`../../grouped_gemm_persistent_3wg.py`（已切 perf_opt；`_DEFAULT_CONFIG` → `block_n=128 / block_k=256`）
- 结构化记录：`perf_records.jsonl`（4 workload × baseline/final）
- 用例分类：`workload_inventory.json`（4 tune `winner_merged`，2 smoke `precision_pass=true`）
- 权威报告：`examples/TileOPs/reports/tileops/20261016_162140_044116_FusedMoEExpertsNopadPersistent3WGFwdOp/`、`.../20261016_164351_083323_FusedMoEExpertsNopadPersistent3WGFwdOp/`

---

# 6. P1 探针（2026-10-22）—— MoE 专家段三方向 go/no-go

> 本节是**单配置探针**（不是新一轮完整调优）。目的：在立项前用实测判据回答
> 「按专家规模自适应 tile / 跨专家双缓冲预取 / no-pad 尾块合并」这三个方向
> 值不值得做。**定稿 `perf_opt/_persistent_grouped_gemm_v2_kernel.py` 一字未改**
> （sha256 = `75877d4b…13d793`，见 §6.8 回验）。
>
> 服务对象：`FusedMoEExpertsNopadPersistent3WGFwdOp` / `FusedMoeFwdOp` /
> `FusedMoeFwdCbFwdOp` 三个算子共用的 3WG persistent grouped GEMM 主体。
> 13 个 case 由 7 条 manifest workload 组成，几何上只有 **5 个不同 (numel, E) 族**
> × 2 个 GEMM（gate_up `N=2F,K=H`；down `N=H,K=F`）。deepseek-prefill 与
> kimi-prefill 的 GEMM 形状完全相同（37.72% vs 37.63% 的差异是测量噪声）。

## 6.0 探针方法与可复现性

| 项 | 做法 |
|---|---|
| 驱动 | `probe_p1_case.py`（自建 kernel + warmup + N 次实测）经 `probe_p1_run.py` 调 `msprof op` |
| 唯一时延口径 | `msprof op` → `Task Duration(us)`（中位，launch-count=3, warm-up=1） |
| 顺序偏置对策 | 同一 case 的 variant 列表**正序跑一遍、反序跑一遍**，每 variant 得到 2 次测量、前后位次各一 |
| 精度自检 | 定稿/候选 kernel 走 `run_case`（golden = 逐专家 NT matmul fp32 累加） |
| Ratio | 仓库同式：`roofline 'GM Read + Write'` 条目（`tileops.benchmark.msprof_roofline`），与 harness 报的 Ratio 同源 |
| GM 流量 | 同上条目反解 `bytes = performance × duration / AI`，用于判定「B 是否单遍」 |
| 管线占用 | `PipeUtilization.csv` 的 `aic_cube_ratio` / `aic_mte2_ratio` / `aic_mte1_ratio` |

**探针 ↔ harness 交叉标定**（探针必须先对上 harness 的绝对数字，否则只是自娱）：

| workload | harness KPI（定稿） | 本探针 | 偏差 |
|---|--:|--:|--:|
| deepseek-v3-decode（gate_up） | 18199.9 µs / 47.11% | 18262.3 µs / 46.98% | **+0.34%** |
| qwen3-235b-prefill（gate_up） | 30199.6 µs / 40.58% | 30210.0 µs / 40.59% | **+0.03%** |

同一配置重复测的离散度（A/B 首尾对）：

| 对照 | run1 | run2 | Δ |
|---|--:|--:|--:|
| ds_decode base | 18222.2 | 18242.3 | 0.11% |
| qwen_prefill base | 30191.5 | 30228.5 | 0.12% |
| ds_decode_down base | 8903.9 | 8914.5 | 0.12% |

⇒ **探针分辨力 ≈0.15%，远小于 3% 判优阈值**，可直接用于 go/no-go。

**两个踩坑（已修，供后续探针复用）**

1. `torch.empty(...).normal_()` 在 15 GB 张量上会让 `memory_reserved` 从 14 GB
   膨胀到 **42 GB**（临时块），直接触发 msprof 的内存快照失败
   （`CreateSnapshot ret=207001` → 该 kernel 只有 NA 行）。改用
   `uniform_(-1,1).mul_(0.02)`（就地，无临时块）+ `empty_cache()` 后，
   DeepSeek 形状（B=15.03 GB）**可以正常出 msprof 数据与 Ratio**。这与
   `workloads/moe.py` 里记的 B3 同因。
2. `msprof op` 必须带 `--kernel-name=main_mix_aic`，否则它会对进程里每个 kernel
   （含 import 期的 `Cast`/`Muls`）尝试快照，失败后整段数据都拿不到。

**测量覆盖的说明（诚实边界）**：E=256 且 gate_up（B=15.03 GB）可测；E=256 的
prefill/medium 族（numel=32768/16384）与 decode 同族，用 **E=128 半规模模型**
（`kimi_medium_model` = numel 8192/E 128，s_e~64）保持「每专家行数分布 + tile 形状
+ 并发 tile 数」不变，只把专家数减半。所有 A/B 结论都是**同 shape 内**相对比较，
模型只用于补充第 3、4 个数据点。

## 6.1 第一步：欠载到底占多少（先量化，再决定要不要动结构）

### 6.1.1 tile 统计（真实 routing：`topk_ids = randint(0,E)` ⇒ s_e ~ Multinomial）

定稿 tile 形状 `block_m=128 / block_n=128`；live tile = `Σ_e ceil(s_e/bm) × num_pid_n`。

| 族（13 case 归并） | E | mean s_e | live tiles | mean tile_m | **M 容量利用率** | M 尾块占 live | area 口径浪费 | 分形(16)口径浪费 | A/B 名义读比 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| deepseek/kimi-decode | 256 | 16.0 | 8192 | 16.0 | **12.5%** | 100% | 87.5% | 81.9% | 0.12 |
| qwen3-decode | 128 | 32.0 | 4096 | 32.0 | 25.0% | 100% | 75.0% | 69.0% | 0.25 |
| kimi-medium | 256 | 64.0 | 8192 | 64.0 | 50.0% | 100% | 50.0% | 44.1% | 0.50 |
| deepseek/kimi-prefill | 256 | 128.0 | 12160 | 86.2 | 67.4% | 65.5% | 32.6% | 28.8% | 1.00 |
| qwen3-prefill | 128 | 256.0 | 10080 | 104.0 | 81.3% | 39.7% | 18.7% | 16.4% | 2.00 |

N 尾块、K 尾块在本批 shape 上均为 **0**（N∈{4096,7168} 对 128/256 整除；K∈{7168,2048}
对 256 整除），所以「M/N 尾块欠载」在本批 workload 上**实际只有 M 方向**。

### 6.1.2 三种代价模型，用实测裁决

欠载会不会变成时间，取决于单 tile 的代价模型。三种可能：

* **A. area 模型**（tile 代价 ∝ `bm*bn`，DESIGN §1.6.0 的隐含口径）：decode 浪费 87.5%；
* **B. 分形模型**（∝ `ceil(tile_m/16)*16 * tile_n`）：decode 浪费 81.9%；
* **C. bytes 模型**（本 kernel 是 B 访存受限，代价 ∝ `tile_n*K + tile_m*K + tile_m*tile_n`）：
  B 项与 `tile_m` 无关 ⇒ **欠载不产生额外流量**，浪费只来自 A 的重读与非 B 流量。

实测裁决（每个都是 msprof op 的 GM 读流量，不是估算）：

| 实测 | GM 流量 | B 单遍解析值 | 非 B 占比 | 结论 |
|---|--:|--:|--:|---|
| ds_decode gate_up | **15.44 GB** | 15.03 GB | **2.7%** | 面积口径的 87.5% 浪费**几乎没有变成流量** ⇒ A/B 模型被证伪 |
| qwen_prefill gate_up | **22.07 GB** | 7.52 GB | **65.9%** | 但这 65.9% 是 **A 重读**（`num_pid_n×numel×K`=15.03 GB），不是尾块欠载 |
| kimi_medium_model | 8.21 GB | 7.52 GB | 8.4% | 同 decode：A 基本吃到 L2 |
| ds_decode_down | 7.64 GB | 7.52 GB | 1.6% | — |

**机制**：动态 extent（`size=[tile_m,...]`）让硬件只搬/只算有效行 —— 一个
`tile_m=16` 的 tile 搬 16 行 A、算 16 行，**没有零填充流量、没有补零 MMA**。
所以「小专家占不满 block_m」在本 kernel 里**不产生可观的时间损失**：正确的代价
模型是 **C（B 单遍流）**，欠载的时间占比上界 = **2.7%（decode 族）**、
**8.4%（kimi-medium）**。

**反向验证（把 M 变满会不会更快？）**：`fullref_e32`（E=32/uniform，每专家恰好
128 行 = 1 个满 M-tile，B 单遍 1.88 GB）：**4864.5 µs / GM 2.18 GB ⇒ 448 GB/s**；
而 M=16 的 ds_decode 是 **846 GB/s** —— 把 M 变满带宽**掉一半**（每个 tile 多搬 8 倍
A 行、L1/L0 压力更大）。⇒ 「M 欠载拖慢带宽」在实测上是**反向**的。

> **欠载量化结论**：三方向共同假设的根因「M/N 尾块与专家边界 tile 欠载」在实测中
> 只值 **2.7% ~ 8.4%**，**低于 10% 门限**。真正的时间去向是两件与尾块无关的事：
> ① 单缓冲导致的 **MTE2 空转 20.5%**（§6.3）；② prefill 族的 **A 重读占 66% 流量**
> （§6.5）。

## 6.2 D1「按专家规模自适应 tile」→ **no-go**

假设：小专家（s_e≈16）用 64×64/128×64，大专家（s_e≥128）用 128×128。

| case | (bm,bn,bk) | Task Duration(µs) | Δ | GM 流量 | eff BW | Ratio | cube | mte2 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ds_decode | **128,128,256 (base)** | **18262.3** | — | 15.44 GB | 845.3 GB/s | 46.98% | .458 | .786 |
| ds_decode | 64,128,256 | 19358.3 | **+6.18%** | 15.42 GB | 796.3 | 44.24% | .256 | .827 |
| ds_decode | 128,64,256 | 21496.2 | **+17.90%** | 15.47 GB | 719.7 | 40.00% | .403 | .763 |
| qwen_prefill | **128,128,256 (base)** | **30210.0** | — | 22.07 GB | 730.5 | 40.59% | .368 | .822 |
| qwen_prefill | 64,128,256 | 42959.3 | **+41.28%** | 37.05 GB | 862.5 | 47.95% | .229 | .841 |
| qwen_prefill | 128,64,256 | 40863.8 | **+34.38%** | 22.41 GB | 548.4 | 30.47% | .283 | .821 |

两条独立理由：

1. **构造上不可能**：动态 extent 下 `tile_m = min(block_m, s_e)`。对 `s_e < block_m`
   的专家（decode 族 100%）把 block_m 从 128 降到 64 **不改变任何 tile 的行数** ——
   "小专家用小 tile" 是**恒等变换**。上表 `ds_decode bm=64` 就是"所有专家都换成小
   tile"的极端情形，实测 **+6.18%**（L0C/L1 变小的副作用，cube 占用从 .458 掉到 .256
   但时间更长），所以任何"部分专家换小 tile"的加权上界 ≤ base。
2. **大专家侧 base 已是实测最优**：`qwen_prefill bm=64` = **+41.28%**，且 GM 流量从
   22.07 → 37.05 GB（小 M-tile 让 A 的 L2 复用崩塌，+68% 流量）；`bn=64` = +34.38%。

**结论 no-go。** 唯一被实测证实有效的同族变量是 `block_n` 按 **shape**（而非按专家）
在编译期选择，见 §6.5（qwen_prefill −6.7%）。

## 6.3 D2「双缓冲预取（copy 与 Cube 重叠）」→ **go（−19% ~ −24%）**

假设：串行任务序 + 单缓冲让 MTE2 在等 Cube 消费 L1，专家切换处空转。

**基线管线占用（这就是瓶颈证据）**：

| case | mte2 | cube | mte1 | 说明 |
|---|--:|--:|--:|---|
| ds_decode gate_up | **.786** | .458 | .389 | MTE2 是唯一接近饱和的流水 |
| qwen_prefill gate_up | .822 | .368 | .298 | 同 |
| ds_decode_down | .739 | .469 | .396 | 同 |
| kimi_medium_model | .795 | .435 | — | 同 |
| fullref_e32（满 M） | .829 | .382 | — | 与 M 是否满无关 |

⇒ 若把 MTE2 空转（20.5%）吃满，理论加速 = 1/0.786 = **1.27×**，即
`47.0% → 59.7%` ratio。这就是 D2 的天花板，且它**与欠载无关**。

### 6.3.1 框架路线（`T.Pipelined` + `NPUIR_ENABLE_AUTO_MULTI_BUFFER=True`）—— 空实验

| case | variant | Duration(µs) | Δ | cube | mte2 |
|---|---|--:|--:|--:|--:|
| ds_decode | base | 18225.9 | — | .458 | .786 |
| ds_decode | Pipelined ns=2 + 多缓冲 | 18227.7 | **+0.01%** | .457 | .795 |
| ds_decode | 仅开多缓冲（ns=1，退化 serial） | 18229.9 | +0.02% | .459 | .797 |
| qwen_prefill | base | 30397.6 | — | .368 | .822 |
| qwen_prefill | Pipelined ns=2 + 多缓冲 | 30390.7 | **−0.02%** | .366 | .827 |

（`ns=3` 那一档在本轮编译/采集窗口里没有取得有效数据，不列数字；结论不依赖它。）

**管线占用逐位不变**（mte2 .786→.795, cube .458→.457）⇒ 该 pass 对
「显式 `T.alloc_L1` 的 Expert Cube scope」**没有物化双缓冲**，实验是**空实验**。
这同时解释了 Round 2「`T.Pipelined` 零增益」—— 那一轮的 `_PASS_CFG` 同样带
`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`，**该负向结论应作废**（见 §6.7）。

### 6.3.2 手工 2 深 ping-pong（+25 行）—— 实测

实现（`probe_p1_d2_prefetch.py`，`P1_D2_MANUAL=1`）：两份 `a/b` L1 缓冲，同一任务内
**先发两段 copy、再依次 gemm**，让 `copy(k+1)` 与 `gemm(k)` 重叠；其余（读 GM 调度表、
动态 extent、`T.gemm(size=)`、`store_fixpipe`）与定稿逐行一致。

| case | base(µs) | **d2_manual(µs)** | Δ | 加速 | Ratio | cube | mte2 | GM 流量 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ds_decode gate_up | 18262.3 | **14298.0** | **−21.71%** | 1.277× | 46.98→**60.04%** | .458→.583 | .786→**.948** | 15.44 GB 不变 |
| qwen_prefill gate_up | 30210.0 | **24361.4** | **−19.36%** | 1.240× | 40.59→50.39% | .368→.457 | .822→.963 | 22.09 GB 不变 |
| ds_decode_down | 8909.2 | **7042.2** | **−20.96%** | 1.266× | 47.61→**60.20%** | .469→.593 | .739→.854 | 7.64 GB 不变 |
| kimi_medium_model | 12845.0 | **9705.5** | **−24.44%** | 1.323× | 35.52→46.98% | .435→.575 | .795→.953 | 8.21 GB 不变 |

* **4/4 shape 同向改善 ≥19%**（远超 3% 门限），且**GM 流量一字不变** ⇒ 提速完全来自
  流水重叠，不是流量优化。
* MTE2 从 0.79 → 0.95（decode）/ 0.96（prefill），**与 6.3 预测的 1.27× 天花板吻合**
  （1.277× / 1.240× / 1.266× / 1.323×）。剩余 MTE2 空转 ≤5%，3 深缓冲的上界只有
  1.05×，不值得再做。
* 每 GM 字节耗时：base **1.18 µs/MB** → d2_manual **0.93 µs/MB**（ds_decode）；
  MTE2 忙碌时带宽 ≈ **1139 GB/s**（= `CONST-mte2-degradation` 1.26 TB/s 地板的 90%）。

**精度（必须过，否则提速无意义）** —— `P1_D2_MANUAL=1 python probe_p1_d2_prefetch.py --check`：

| 用例 | 覆盖点 | max_diff |
|---|---|--:|
| K=1024, N=512, E=32 | 成对循环 + N 尾块 | 6.10e-05 |
| K=2048, N=384, E=16 | N 尾块 + 偏斜分布 | 6.10e-05 |
| K=1024, N=4096, E=64 | 空专家 + 多 `num_pid_n` | 6.10e-05 |
| K=512, N=128, E=300 | E>核数、每专家 1~2 token | 1.19e-07 |

worst max_diff = **6.10e-05**（容差 1e-2）⇒ 手工 ping-pong 数值正确。

### 6.3.3 已暴露的实现约束（立项前必须落到 DESIGN）

1. **成对循环要求 `K % (2*block_k) == 0`**：本批 shape 的 K∈{7168,2048} 对 512
   均整除 ✓；但 **smoke 用例 K=256（`k_iters=1`）会被截断**。生产版本必须在
   factory 层（`@T.prim_func` 之外）按编译期常量分流：
   `if K % (2*block_k) == 0: <成对体> else: <定稿 serial 体>`（两个 prim_func，
   与本次探针的 `D2_MANUAL` 分派同形），保证任意 K 都有正确路径。
2. **`bn=256 + 手工双缓冲` 在 N=384（`num_pid_n=2` 且 tile_n 有尾块）上
   `bishengir-compile` 确定性 SIGSEGV（10/10 重试全崩，非 CG-2026-0033 的抖动）**；
   同用例 `bn=128` 通过。本仓 MoE 的 N∈{4096,7168} 对 256 整除，故组合方案在本
   批 workload 上可用，但**采纳时必须把 `N % block_n == 0` 写成前置约束**（或先
   排查该崩溃，作为新 CG 候选）。
3. L1 预算：`bn=128` → 4×64 KB = **256 KB**；`bn=256` → 2×64 KB + 2×128 KB =
   **384 KB**；均 ≤ 512 KB ✓（auto multi-buffer 已关，无 ×1.7 膨胀）。

## 6.4 D3「no-pad 尾块合并」→ **no-go**

假设：把多个小专家的尾块拼成一个满 tile 再跑 GEMM。

三条独立理由（`probe_p1_d3_tailmerge.py` 逐条给数）：

1. **不省流量**：B 是每专家一遍的不可省流量；A 的重读次数 = `num_pid_n × numel × K`，
   与「M 行怎么分组」**完全无关**（每一行都会被读 `num_pid_n` 遍）。实测已证：
   ds_decode 的 GM 流量 15.44 GB 只比 B 单遍多 **2.7%**，这 2.7% 就是合并能省的
   **全部**。
2. **不省帧（上界 ≤2.4%）**：D3 唯一能省的是任务帧数。用同一变量标定 ——
   `block_n 128→256` 把 ds_decode 的 live tile 数**减半**（8192→4096），实测
   **18223.2 → 18662.3 µs（+2.41%，仍是回退）** ⇒ 每帧成本在噪声里，上界
   0.44 ms / 205 帧 ≈ **2.14 µs/帧**，全部帧合计 ≤ **2.41%**。
3. **L1/L0C 装不下**：拼满一个 `block_m=128` tile，decode 族（s_e≈16）需要
   k = 8 个专家的 B 同时驻留：
   `k=8: L1 = 8×64KB(B) + 64KB(A) = 576 KB > 512 KB ✗；
        L0C = 8×64KB = 512 KB > 128 KB ✗`。
   `k=2` 装得下但只把 tile 填到 25%（与现状等价），`k=4` 已 L0C 超容。

**结论 no-go**：收益上界 ≤2.4%（与 D2 的 −21% 差一个数量级），风险（跨专家 L1 布局、
多份 L0C 累加区）却最高。

## 6.5 副产品（值得立项）：`block_n` 按 shape 在编译期选择

D1 的实测顺带暴露出真正能动的 tile 变量 —— **不是按专家，而是按 shape 选 `block_n`**：

| case | bn=128 (base) | bn=256 | Δ | 说明 |
|---|--:|--:|--:|---|
| qwen_prefill | 30210.0 | **28367.0** | **−6.7%** | A 重读占 66% 流量（A/B=2.0），bn 加倍把 A 重读减半 |
| kimi_medium_model | 12845.0 | 12752.2 | −0.4% | A/B=0.5，A 基本吃 L2 |
| ds_decode | 18262.3 | 18662.3 | **+2.4%** | A/B=0.12，无 A 流量可省，只有 L1 变大 |
| ds_decode_down | 8909.2 | 9021.3 | **+1.4%** | 同 |

`bn=256` 的精度自检（定稿 kernel，5 用例含 N 尾块/K 尾块/E>核数）：
**worst max_diff = 2.44e-04 ✓**。

（表中 `Δ` 都是**同一次 run 内**的 A/B/A/B 对：bn=256 那一批的 base 实测为
18223.2 µs，与上一节 18262.3 µs 的差异是 run 间抖动 0.2%，不要跨 run 相除。）

**与 D2 组合**（手工双缓冲 + bn=256，qwen_prefill）：

| 变体 | Duration(µs) | Δ vs base | Ratio | cube | mte2 | L1 |
|---|--:|--:|--:|--:|--:|---|
| base (128,128) | 30233.1 | — | 40.55% | .368 | .820 | 128 KB |
| d2_manual (128,128) | 24388.0 | −19.33% | 50.34% | .457 | .963 | 256 KB |
| **d2_manual (128,256)** | **21446.3** | **−29.06%** | **56.03%** | .509 | .962 | 384 KB |

（`d2_manual(128,128)` 另一次独立 run 为 24361.4 µs / 50.39%，两次差 0.11%。）

⇒ prefill 族：`40.58% → 56.04%`；decode 族：`47.11% → 60.04%`。
按 shape 在编译期选 `num_pid_n`/`block_n`（`numel/E ≥ 128` 之类判据）零结构风险，
可与 D2 独立立项。

## 6.6 三方向 go/no-go 汇总

| 方向 | 裁判数据 | 结论 | 若 go 的收益 |
|---|---|---|---|
| **D1** 按专家规模自适应 tile | 小 tile 在 2/2 shape 回退（+6.2% / +41.3%）；`s_e<block_m` 时 `tile_m=s_e` 与 block_m 无关 | **no-go** | 0（构造上是恒等变换）；同族改为 §6.5 的按 shape 选 bn：prefill −6.7% |
| **D2** 双缓冲预取（copy 与 Cube 重叠） | 手工 2 深 ping-pong：4/4 shape **−19% ~ −24%**，mte2 .79→.95，流量不变；框架路线是空实验 | **go（优先）** | ds_decode 18.26→14.30 ms（ratio 47.1→60.0），prefill −19.4%（+bn256 −29.1%），kimi-medium −24.4% |
| **D3** 尾块合并 | 流量不可省（decode 只剩 2.7%）、帧成本 ≤2.4% 且在噪声内、拼满需要 k=8 时 L1/L0C 超容 | **no-go** | ≤2.4%（低于噪声阈值，另一数量级） |

## 6.6.1 D2 实现方案（供 conductor 立项；本次**未改**定稿）

**改动清单**（相对定稿，全部落在 `_persistent_grouped_gemm_v2_kernel.py` 的 kernel 体）：

1. **2 份 a/b L1 缓冲**（`a0,b0,a1,b1`）+ K 循环改为 `k_iters//2` 次、每次发两段
   copy 再依次两次 `T.gemm`（第二段 `initC=False`）。**+25 行**。
2. **factory 层按编译期常量分流**：`K % (2*block_k) == 0` 走成对体，否则走现有
   `T.serial` 体（smoke 用例 K=256 就落在这里 —— `k_iters=1` 无法成对）。
   两个 `@T.prim_func` 定义、在 `@T.prim_func` **之外**分派（与本次探针
   `D2_MANUAL` 同形）。**+~40 行（含重复签名）**。
3. **前置约束**：`N % block_n == 0`。理由见 §6.3.3 第 2 条（`bn=256` + N 尾块
   → bishengir 确定性 SIGSEGV）。本批 workload 的 N∈{4096,7168} 均满足。
4. **`_PASS_CFG` 不用改**：已实测 `P1_D2_MB=0`（即定稿配置，auto multi-buffer
   **关**）下手工 ping-pong 同样有效 —— ds_decode **18216.6 → 14269.8 µs
   （−21.66%，ratio 47.08→60.07，mte2 .795→.954）**，与开着时（−21.71%）等价。
   ⇒ 立项后不必动 buffer planner 的开关，风险面更小。

**合计 ≈65 行、0 处算法语义变更**（所有搬运行数/坐标系/动态 extent/`size=` 均不变）。

**预估收益（kernel 口径，即 `main_mix_aic` 的 `Task Duration`）**：

| workload（13 case 归并） | 现状 µs / ratio | 预估 µs / ratio | 增益 |
|---|--:|--:|--:|
| deepseek-v3-decode / kimi-k2-decode（gate_up） | 18200 / 47.1% | **14270 / 60.1%** | **−21.7%** |
| deepseek-v3-decode / kimi-k2-decode（down） | 8909 / 47.6% | **7042 / 60.2%** | **−21.0%** |
| qwen3-235b-prefill（gate_up） | 30200 / 40.6% | **24360 / 50.4%** | **−19.4%** |
| kimi-k2-medium（gate_up，模型测） | 12845 / 35.5% | **9706 / 47.0%** | **−24.4%** |
| qwen3-235b-decode（gate_up） | 10221 / 42.9% | 待测（同族预期 −15%~−20%） | 待实测 |
| deepseek/kimi-prefill（gate_up） | 38133 / 37.7% | 待测（同族预期 −19%） | 待实测 |

叠加 §6.5 的 `block_n` 按 shape 选择后 prefill 族可到 **−29%、ratio 56%**。
注意这是 **kernel 口径**；算子端到端还含 permute / activation / unpermute，增益会被
摊薄（3WG 的 KPI 本就是 `main_mix_aic` 的时延，故报告口径不变）。

**实施后的验收要求**：`run_L0` + `run_L1` 全量、两版 smoke 精度、harness 全量复测
（13 case）、以及 `N % block_n != 0` 的另一条兜底路径的构造用例各一条。

## 6.7 被本次探针推翻/修正的既有结论

1. **「`T.Pipelined` 零增益」（Round 2）→ 应作废**：那一轮的 pass 配置关着
   `NPUIR_ENABLE_AUTO_MULTI_BUFFER`，多缓冲根本没物化（本次复测管线占用逐位不变
   已证）。真正的双缓冲收益是 **−21%**。Round 2 记录按实保留，但结论加此更正。
2. **「M/N 尾块与专家边界欠载是共同根因」（本节之前的定性）→ 部分证伪**：
   欠载只值 2.7%~8.4%；真正的两块是 ① 单缓冲 MTE2 空转 20.5%（D2）；
   ② prefill 的 A 重读占 66% 流量（§6.5）。
3. **「block_k 是主旋钮」→ 补充边界**：`block_k 256→512/1024` 在 ds_decode 上
   **+4.7%~+4.9% 回退**（mte2 掉到 .66/.61，单缓冲下更大的块反而更少重叠）。
   block_k 的最优点在 256，不要再往上扫。

## 6.8 回验与产物

* **定稿未改**：`sha256(perf_opt/_persistent_grouped_gemm_v2_kernel.py)` =
  `75877d4b24344d97429bb79d80cbbbfe2bb03bb0b37e8477a45139c60513d793`（与探针前一致）。
* **未写 `perf_records.jsonl`**：本轮没有产生新的生产候选（定稿未变 ⇒ 既有
  baseline/final 记录仍然有效），探针数据全部记在本节与 `logs/p1/*.json`；按
  `signal-registry.md` §5，凭空追加 `candidate` 行会破坏「同一 kernel 的 final
  记录来自同一候选」的对账。
* 产物：`probe_underload.py`（欠载解析）、`probe_p1_case.py` + `probe_p1_run.py`
  （msprof 驱动与 A/B 交替编排）、`probe_p1_d1_adaptive_tile.py`（D1 判据计算）、
  `probe_p1_d2_prefetch.py`（D2 kernel 实现 + `--check` 精度自检）、
  `probe_p1_d3_tailmerge.py`（D3 预算）、`logs/p1/*.json`（逐次测量，含每次 run 的
  duration 列表 / GM 字节 / Ratio / 管线占用）、`profiles/p1/`（9 组关键配置的
  msprof CSV + `visualize_data.bin`）、`history_version/20261022_pre_p1_probe/`（改前快照）。
* `perf_opt/_persistent_grouped_gemm_v2_kernel.py` **未修改**，`probe_p1_d2_prefetch.py`
  是独立探针文件 —— 采纳需 conductor 立项（方案见 §6.6.1）。
* 复现命令示例见各 `probe_*.py` 文件头。

## 6.9 与 DESIGN roofline 的对照（为什么要停在这里）

DESIGN §1.6.0 给 GEMM 核的流量下界 = B 单遍 15.03 GB ÷ 1.26 TB/s ≈ **11.9 ms**
（B 只由 E/N/K 决定，故 decode 与 prefill 的 gate_up **同一个下界**；A/C 项另计）。

| 版本 | gate_up 实测 | 相对自身流量地板 |
|---|--:|---|
| ds_decode 定稿（单缓冲） | 18.20~18.26 ms | 15.44 GB / 1.26 = 12.3 ms ⇒ **1.48×** |
| **ds_decode + D2 双缓冲** | **14.27 ms** | ⇒ **1.16×**（ratio 60.0%，已达 60% 目标）|
| qwen_prefill 定稿 | 30.21 ms | 22.07 GB / 1.26 = 17.5 ms ⇒ 1.73× |
| **qwen_prefill + D2 + bn256** | **21.45 ms** | 21.63 GB / 1.26 = 17.2 ms ⇒ **1.25×**（ratio 56.0%）|

⇒ D2 之后 decode 族离「B 单遍 + 1.26 TB/s 混合流量地板」只剩 **16%**，其中
MTE2 残余空转 ≤5%（mte2 .95）、非 B 流量 2.7%、其余是 512 B/行的 DMA 分段效率。
**kernel 内可挖的空间已经很小**。prefill 族真正的下一块是 A 复用（§6.5，流量口径
22.08 GB 里 15.03 GB 是 A 重读），那是 block 形状 / 任务序问题，不属于这三个方向。

## 6.10 Skill Retrospective（本轮）—— 供 evolver 蒸馏

* **[P] 「负向实验」必须核对杠杆是否真的被施加**：`T.Pipelined`「零增益」被
  `NPUIR_ENABLE_AUTO_MULTI_BUFFER` 开关架空了两轮。判据：改动前后
  `PipeUtilization` 的 `aic_mte2_ratio`/`aic_cube_ratio` 是否变化 —— 逐位不变即为
  空实验。价值点候选 `BP_p1_null_experiment_check`（vp_type=P，evidence=本节 6.3.1
  逐位不变 + §6.6.1 第 4 条「开关开着/关着结果相同」，repro=
  `probe_p1_d2_prefetch.py` 的 `P1_D2_MANUAL`/`P1_D2_MB` 两个正交开关，
  toolchain=tilelang 0.1.2+dev / CANN 9.0.0 / Ascend910_9362）。
* **[D] 实测常数**：本机 Ascend910_9362（20 cube core）上，no-pad grouped GEMM 的
  B 单遍流在**单缓冲**下 847 GB/s（decode）/ 730 GB/s（prefill）；**2 深双缓冲**下
  1139 GB/s（= 1.26 TB/s 混合流量地板的 90%）。另：`block_k 256→512/1024` 在单缓冲
  下 **+4.7%~+5.0% 回退**；`block_n 128→256` 在 A/B 读比 ≥2 的 shape 上 **−6.7%**、
  在 A/B ≤0.5 的 shape 上 +1.4%~+2.4%。vp_type=D，repro 同上 + `probe_p1_run.py`
  的 `bk512`/`bk1024`/`bn256` 三个 variant。
* **[R] 流程规则**：探针的 `msprof op` 必须带 `--kernel-name`；探针输入张量禁止用
  `normal_()`（15 GB 上 reserved 膨胀 3×→快照失败）。vp_type=R，evidence=§6.0。
* **[C] 案例索引**：MoE no-pad 专家段 = 「单缓冲串行 K 环 + 动态 extent」结构里
  最大的单点杠杆是 **L1 双缓冲（−21%）**，不是 tiling、不是尾块合并。

---

# 7. Round 7（2026-10-22）—— D2 双缓冲预取落地 + `block_n` 形状选择

> 本轮是**正式调优轮次**（上一轮只做探针并判 D2 = go，见 §6）。改动落在**定稿模块**
> `perf_opt/GroupedGemmPersistent3WG.py`（gate 契约名 `perf_opt/{operator_name}.py`），
> **Stage 3 baseline `_persistent_grouped_gemm_v2_kernel.py` 一字未改**（sha `1939c65a…` 前后一致）。
> 服务对象是三个共用本主体的算子：`FusedMoEExpertsNopadPersistent3WGFwdOp` / `FusedMoeFwdOp` /
> `FusedMoeFwdCbFwdOp`。

## 7.0 本轮 base 与改前快照

- base = v1 定稿 `perf_opt/_persistent_grouped_gemm_v2_kernel.py`（sha `75877d4b…13d793`，与 §6.8 一致）。
- 改前快照：`history_version/20261022_d2_impl/`（整目录，含 v1 定稿/实验件/probes/profiles）；
  更早的探针前快照 `history_version/20261022_pre_p1_probe/` 同时保留。

## 7.1 现象（本轮 base 的最新 profile，v1 定稿）

| case | Task Duration(µs) | `aic_mte2` | `aic_cube` | `aic_mte1` | GM 读 |
|---|--:|--:|--:|--:|--:|
| ds-decode gate_up | 18215.8 | **0.794** | 0.458 | 0.389 | 15.44 GB |
| qwen-prefill gate_up | 30391.8 | **0.826** | 0.366 | 0.297 | 22.11 GB |
| ds-decode down | 8914.1 | **0.748** | 0.469 | 0.395 | 7.64 GB |

MTE2 是唯一接近饱和的流水且**空转 20%~25%**：单缓冲下 `copy(k+1)` 必须等 `gemm(k)` 让出 L1。
§6.1 已把"欠载"这条候选排除（只值 2.7%~8.4%），因此本轮候选直接对准流水重叠与 A 重读。

## 7.2 候选优化点（同一 base 派生）

| id | 候选 | 类型 | 依据 |
|---|---|---|---|
| **C1** | D2 手工 2 深双缓冲（成对 copy + 2×gemm，两份 a/b L1） | 结构（主项） | §6.3：4/4 shape −19%~−24%，mte2 → .95 |
| **C2** | `block_n` 按 shape 编译期选择（`numel//E ≥ 128` 且 `N%256==0` → 256） | 参数/策略（次项，独立可回退） | §6.5：prefill −6.7%、decode +1.4~2.4%（不采纳全局） |
| C3 | 3 深缓冲 | 结构 | §6.3 已证 tte2 到 .95，上界 ≤5% → **不立项** |
| C4 | `block_k` 512/1024 | 参数 | §6.7：+4.7%/+4.9% 回退 → **不立项** |

C1 与 C2 在**同一个** `@T.prim_func` 集合内落地，但两者是**独立可回退**的：C2 只改
`_select_block_n()` 的一个门限，C1 只改 K 循环体；关掉任一项都不影响另一项。

## 7.3 实现要点（≈65 行，0 处算法语义变更）

1. 两份 `a/b` L1 缓冲；K 循环 → `k_iters//2` 次成对（先两段 copy，再两次 gemm，第二段 `initC=False`）。
2. **factory 层编译期分流**（两个 `@T.prim_func`、只构一个）：`K % (2*block_k) == 0` **且**
   `N % block_n == 0` → 成对体；否则 → v1 的 `T.serial` 体（动态 extent 的 K/N 尾块能力保留）。
   分流必须写在 `@T.prim_func` 之外（体内 `if <python int>:` 会被 parser 当运行时布尔）。
   op 层 / wrapper 层**没有**新增性能分派。
3. **硬守卫 `N % block_n == 0`**：§6.3.3 记录的 `block_n` 尾块 + 成对体 → bishengir 确定性 SIGSEGV。
4. **`_PASS_CFG` 未改**（`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`），与 §6.6.1 第 4 条一致。
5. wrapper 的 perf_opt import 指向新模块；`_DEFAULT_CONFIG` 改为引用模块级常量
   `TUNED_DEFAULT_CONFIG`（不重复硬编码）。

## 7.4 候选 vs current best 对比表（kernel 口径，`msprof op`）

A/B/A/B 同一 run 内交替（`probe_r7_ab.py`，`profiles/r7/final_ab__*` 为 raw），v1 = current best，
v2 = 本轮候选（即最终定稿文件，sha `e3babaa6…`）。**memory 列是 msprof 反解的 GM 读字节**。

| shape | 候选 | Task Duration(µs) | Δ | AICore(cube) | MTE2 | MTE1 | GM 读 | Ratio | L0 |
|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| ds-decode gate_up | v1 (current best) | 18215.8 | — | 0.458 | 0.794 | 0.389 | 15.44 GB | 47.05% | PASS |
| ds-decode gate_up | **v2 (C1)** | **14296.8** | **−21.51%** | **0.584** | **0.948** | 0.514 | **15.44 GB（不变）** | **59.96%** | PASS |
| qwen-prefill gate_up | v1 (current best) | 30391.8 | — | 0.366 | 0.826 | 0.297 | 22.11 GB | 40.41% | PASS |
| qwen-prefill gate_up | **v2 (C1+C2)** | **21435.2** | **−29.47%** | **0.510** | **0.970** | 0.376 | **21.69 GB（−1.9%）** | **56.23%** | PASS |
| ds-decode down | v1 (current best) | 8914.1 | — | 0.469 | 0.748 | 0.395 | 7.64 GB | 47.62% | PASS |
| ds-decode down | **v2 (C1)** | **7039.9** | **−21.02%** | **0.593** | **0.863** | 0.525 | **7.64 GB（不变）** | **60.24%** | PASS |

读法：C1 的提速**完全来自流水重叠**（decode 两条 GM 读逐位不变、mte2 .79→.95、cube .46→.58）；
C2 额外把 prefill 的 GM 读从 22.11 GB 降到 21.69 GB（A 重读减半），所以 prefill 的 −29.5% 里
约 −1.9% 来自流量、其余来自重叠。

**最终文件上的复跑证据**（`logs/r7_d2/level_all_final.log`，命令
`TILELANG_ASCEND_MODE=Expert ASCEND_RT_VISIBLE_DEVICES=1 PGG_COMPILE_ATTEMPTS=30 python GroupedGemmPersistent3WG.py --level all`）：
`L0 done (5 cases ...) worst 2.441e-04` / `L1 done (10 cases) worst 2.441e-04` / `L2 done (3 cases, 0 warn)` /
`Boundary done (11 cases, 0 warn)` / `All check passed!` —— 29 个 case 全 PASS、0 warn。
（L0-6 的两 kernel 联调项 SKIP：sibling producer `_tile_scheduler_kernel` 不在本仓路径下，属既有可选检查。）

**L0（精度）**：全部 v2 case PASS，worst max_diff 2.44e-04（容差 1e-2）。整模块
`GroupedGemmPersistent3WG.py --level all` 跑 L0 5 例（L0-6 的两 kernel 联调项因 sibling producer 不在本仓
而 SKIP，属既有可选检查）、L1 10 例（含 4 条新增分支覆盖：
D2+bn256、D2+E>核数+空专家、D2+M 尾块、成对 K 但 N 尾块→serial）、boundary 11 例（含
`K % 2*bk != 0 → serial`）全 PASS，worst max_diff 2.44e-04；L2 的 advisory 例在一次长会话里踩到
编译期 flaky SIGSEGV（单跑可过），按设计非阻塞。

## 7.5 编译期守卫的实测裁决（compiled-shape matrix）

`probe_r7_compile_matrix.py`：十个真实 GEMM 形状 × {成对体 / 强制 serial} × {bn128 / bn256}。
结论（`logs/r7_d2/compile_matrix*.json`）：

| 结论 | 数据 |
|---|---|
| 定稿策略在**十个真实形状**上全部可编译 | 干净进程内 12 次预算：10/10 成功，多数第 1 次，最坏 3 次 |
| 3 次预算会产生假失败 | 同一矩阵用 3 次预算时有 14/40 失败 —— 是 CG-2026-0033 的 flaky SIGSEGV，不是形状确定性 |
| 长会话会烧完默认 10 次预算 | harness 内 `qwen3-prefill` 一次、advisory `L2-3` 一次（`PGG_COMPILE_ATTEMPTS=30` 后均通过） |
| `N % block_n == 0` 守卫仍然保留 | §6.3.3 的 N 尾块形态是 10/10 确定性崩溃（本轮未再冒险复测出反例） |

→ 采纳动作：三个算子的权威 harness run 一律用 `PGG_COMPILE_ATTEMPTS=30`；不修改 kernel 的
重试逻辑本身（默认 10 次对单编译仍够用）。

## 7.6 三算子逐 shape 前后（权威 harness run，`msprof op` 同口径）

`--kernel-name main_mix_aic`、`--prof-mode msprof`；`latency_us` 取 msprof 报出的较大那次
`main_mix_aic` 采样（= gate_up，与 v1 报告同一口径；较小那次是 down GEMM，见每节脚注）。

### 7.6.1 `FusedMoEExpertsNopadPersistent3WGFwdOp`

run：`examples/TileOPs/reports/tileops/20261022_122530_041437_FusedMoEExpertsNopadPersistent3WGFwdOp/`
（**status = passed**：correctness 20/20、benchmark 6/6）。v1 列取 v1 报告的 run（`20261016_162140` / `_164351`）。

| case | v1 µs | v2 µs | Δ | v1 Ratio | v2 Ratio | down（v1→v2） |
|---|--:|--:|--:|--:|--:|---|
| qwen3-235b-decode | 10221.1 | **7849.5** | **−23.20%** | 42.94% | **55.91%** | 4962.7 → 3948.2（−20.4%）|
| qwen3-235b-prefill | 30199.6 | **21428.2** | **−29.05%** | 40.58% | **56.15%** | 14087.4 → 10006.8（−29.0%）|
| deepseek-v3-decode | 18199.9 | **14332.0** | **−21.25%** | 47.11% | **59.84%** ⚠ | 8925.2 → 7020.6（−21.3%）|
| deepseek-v3-prefill | 38133.3 | **26215.3** | **−31.25%** | 37.72% | **54.19%** | 17684.4 → 12360.4（−30.1%）|

⚠ `deepseek-v3-decode` 压线：同一候选的另一次 run（`20261022_105556`）为 **60.32%**，本 run 59.84%，
run 间差 0.8% ⇒ **临界达标，不作稳健达标宣称**。smoke：bf16 7.80 µs、fp16 7.58 µs（仅精度回归）。

### 7.6.2 `FusedMoeFwdOp`

run：`examples/TileOPs/reports/tileops/20261022_113704_755433_FusedMoeFwdOp/`
（**status = passed**：correctness 12/12、benchmark 6/6）。v1 列取 `20261016_164925_233969`。

| case | v1 µs | v2 µs | Δ | v1 Ratio | v2 Ratio | down（v1→v2） |
|---|--:|--:|--:|--:|--:|---|
| qwen3-235b-decode | 10250.3 | **7874.7** | **−23.18%** | 42.80% | **55.72%** | 4978.9 → 3956.5（−20.5%）|
| qwen3-235b-prefill | 30186.3 | **21492.0** | **−28.80%** | 40.94% | **56.40%** | 14052.5 → 10049.6（−28.5%）|
| deepseek-v3-decode | 18253.5 | **14227.8** | **−22.05%** | 46.99% | **60.28%** ✅ | 8923.6 → 7014.4（−21.4%）|
| deepseek-v3-prefill | 38137.8 | **25929.4** | **−32.01%** | 35.78% | **51.70%** | 17852.6 → 12193.8（−31.7%）|

✅ 本算子口径下 `deepseek-v3-decode` 稳定在 60.3%（两次 run 分别 60.28% / 59.84~60.32% 量级），
但同样属**压线**；`Ratio` 只在同一算子内可比（本例 60.28% vs op3 同形状 45.98% 是 roofline 公式差异）。

### 7.6.3 `FusedMoeFwdCbFwdOp`

run：`examples/TileOPs/reports/tileops/20261022_120327_902897_FusedMoeFwdCbFwdOp/`
（**status = passed**：correctness 8/8、benchmark 5/5）。v1 列取 `20261016_173133_701785`。

| case | v1 µs | v2 µs | Δ | v1 Ratio | v2 Ratio | down（v1→v2） |
|---|--:|--:|--:|--:|--:|---|
| kimi-k2-decode | 18138.3 | **13672.6** | **−24.62%** | 34.65% | **45.98%** | 8882.8 → 7059.2（−20.5%）|
| kimi-k2-medium | 27183.7 | **20977.9** | **−22.83%** | 34.49% | **44.72%** | 12830.8 → 10092.7（−21.3%）|
| kimi-k2-prefill | 38333.9 | **26514.1** | **−30.84%** | 37.63% | **53.74%** | 17802.0 → 12522.6（−29.7%）|

smoke：bf16 7.80 µs、fp16 7.98 µs。

### 7.6.4 汇总

- **11 条 tune case 全部改善，幅度 −21.25% ~ −32.01%，无一条回退**（阈值 3%）。
- 新增达标（Ratio ≥ 60%）：**0 条稳健达标**；`deepseek-v3-decode` 在 op1/op2 上压线（59.84%~60.32%）。
  按"Ratio ≥ 60% 即停"的口径，本轮**未宣布成功**，停止原因是 budget/结构判断（见 §7.8）。
- `block_n` 的净收益：prefill 族 −1.9%（流量）叠加在 D2 之上，decode 族不受影响 ⇒ **采纳**，
  且**没有 shape 因此回退 >3%**。

## 7.7 合并与非回退检查

| 检查 | 结果 |
|---|---|
| 单点先行 | C1 单独在 3 个 shape 上验证（−21.0%~−21.5%），再叠加 C2（prefill −29.5%）|
| 合入同一 `@T.prim_func` | 是：C1 与 C2 都在定稿文件里，且未新增任何 Python factory/op/wrapper 性能分派 |
| 本 kernel 全部 tune workload 复测 | 4/4 通过（§7.6.1），smoke 2/2 精度通过 |
| 跨算子合并后检查 | 3 个算子共 11 条 tune case 全量复测，全部同向改善（§7.6.1~7.6.3）|
| 回退能力 | 关 C2：把 `BN_BUMP_MIN_MEAN_ROWS` 调到 `num_experts` 以上；关 C1：把 `_use_prefetch_pair` 返回 False（或把 wrapper 的两条 import 指回 v1 文件） |

## 7.8 终止判定与诚实边界

- 终止原因：**budget + 结构判断**。C1/C2 已落地并全量复测；C3/C4 与 D1/D3 已由 §6 实测证伪；
  decode 族已到「B 单遍 + 1.26 TB/s 混合流量地板」的 **1.16x**（MTE2 残余 ≤5%、非 B 流量 2.7%），
  prefill 族下一块是 A 复用（任务序/分核），属**设计层**，不在本轮授权内（也不满足
  `[DESIGN_LIMIT]` 的量化门槛：剩余结构性估计远小于 2x）⇒ **本轮不产出 `perf_feedback.md`**。
- 诚实边界 ①：三个算子的 harness run 各发生过测量侧失败（op1 一次 `qwen3-prefill` 编译 flaky、
  一次 deepseek 两条 NPU OOM、一次 `qwen3-decode` 回退到无 `--kernel-name` 的捕获 → 43.08 µs 的假数字）。
  处置：**一律显式 `--kernel-name main_mix_aic` + `PGG_COMPILE_ATTEMPTS=30` 重跑**，只采用
  `captured_op_names == ['main_mix_aic']` 且 status=passed 的 run；43.08 µs 那条已作废并记录在此。
- 诚实边界 ②：定稿文件在最后一次采集后只做了 **docstring 数字更正**（注释文本，不影响 IR）；
  为避免"哈希与采集不一致"的疑虑，采集后另用最终文件跑了 §7.4 的 kernel 口径 A/B（v2 = 最终 sha），
  数字与 harness 一致（14296.8 / 21435.2 / 7039.9 vs harness 14332.0 / 21428.2 / 7020.6，差 ≤0.3%）。
- 诚实边界 ③：`perf_records.jsonl` 的四条 v1 `phase=final` 行被**降级为 `phase=merged`**（只改标签 +
  加 `note`，测量值一字未改），原因是 gate 要求「每个 workload 恰有一条当前最终版本的 final 记录」。
  这是本轮唯一一处对既有行的触碰，按 append-only 精神在 `note` 里显式留痕。

## 7.9 产物（本轮）

- 定稿：`perf_opt/GroupedGemmPersistent3WG.py`（sha256 `e3babaa62fc3e9aec1b3484d4a6acbfa5a802b3c7cacde3a469be5becf157ce5`）
- wrapper：`grouped_gemm_persistent_3wg.py`（perf_opt import 指向新模块；`_DEFAULT_CONFIG` 引用模块常量）
- perf_opt 包入口：`perf_opt/__init__.py`（re-export 定稿模块，满足 wrapper 的包内 import）
- 结构化记录：`perf_records.jsonl`（4 baseline + 4 merged + 4 final=round 7）
- 探针/工具：`probe_r7_ab.py`（v1/v2 交错 A/B）、`probe_r7_compile_matrix.py`（compiled-shape matrix）
- raw profile：`profiles/r7/final_ab__*`（6 组 msprof 输出）
- 过程日志：`logs/r7_d2/`（level_all、compile_matrix、ab_final、三个 CLI run 的 stdout）
- 改前快照：`history_version/20261022_d2_impl/`

## 7.10 Skill Retrospective（Round 7 增量）

- **[R] 长会话内的编译预算必须随 IR 规模上调**：CG-2026-0033 的 flaky SIGSEGV 与"核内串行任务数"正相关，
  默认 10 次 fresh-compile 在 prefill 规模 + 长会话下会被烧完（本轮两次）。判据：`fresh-compile attempt 10/10`
  出现即视为"预算不足"而不是"形状不可编译"；处置：`PGG_COMPILE_ATTEMPTS=30`。vp_type=R，
  evidence=§7.5 + `logs/r7_d2/compile_matrix*.json`，repro=`probe_r7_compile_matrix.py`（`R7_ATTEMPTS` 可调）。
- **[P] "小预算矩阵"会伪造确定性结论**：同一 40 组编译矩阵，3 次预算报 14/40 失败（看起来像"某形状必崩"），
  12 次预算 10/10 通过。教训：**任何"确定性失败"结论都必须写清尝试预算**，否则把 flaky 误判为形状约束。
  vp_type=P，evidence=§7.5 两行数据。
- **[D] 新实测常数**：本机 Ascend910_9362 上，no-pad grouped GEMM 的 2 深双缓冲把
  decode 族有效带宽 847.5 → 1079.7 GB/s、prefill 727.4 → 1012.0 GB/s（= 1.8 TB/s 名义峰值的 60%/56%）；
  cube 占用 .458→.584 / .366→.510 / .469→.593。vp_type=D，
  repro=`probe_r7_ab.py --only ds_decode_gate_up,qwen_prefill_gate_up,ds_decode_down`。
- **[C] 案例索引**：MoE 专家段的"单缓冲串行 K 环"结构，最大单点杠杆是 **L1 双缓冲（−21%~−32%）**；
  第二杠杆是 **`block_n` 按 shape 选择**（A 重读主导时 −6.7% 附加），两者独立可回退。
- **[R] 复合算子的 harness 必须钉 `--kernel-name`**：不钉时框架会在派生失败后无过滤捕获，
  曾把一次 43.08 µs 的辅助 kernel 当成 GEMM 结果（差 240x）。判据：`benchmark.json` 里
  `captured_op_names` 必须恰为 `['main_mix_aic']`。vp_type=R，evidence=§7.8 诚实边界 ①。

---

# 8. Round 8（2026-10-26）—— prefill 的 A 重读：先判别，再改造

- 授权方向：三份 v2 报告都写「prefill 族下一块是 A 复用（任务序/分核），属设计层」。本轮先在
  **MoeGroupedGemmNopad 的口径陷阱**（`22.55 GB` 是 MTE2 端口字节、不是 DRAM 字节）下重做判别，
  再决定是否动任务序。
- 结论：**判别结果与 MoeGG 相反 —— A 重读确实在打 GM（不是 L2 已吸收）；但方向不是"重排任务序"，
  而是"flat 任务索引的分配方式"**。单点改动 `tile_id = t * nums_kernel + core_id`（原为
  `core_id * num_local_tasks + t`）让 4 条 tune case 全部改善 −5.7%~−38.7%，其中 3 条突破 60%。
  **MTE2 端口字节逐位不变**，收益全部来自**核占用率**（20 核全开）而不是字节数。
- 本轮工具链：CANN 9.0.0 / Ascend910_9362 / tilelang `0.1.2+bbb6791`。所有对照都是**同会话**
  A/B（未使用 ×1.147 跨版本折算）。

## 8.1 先判别：GM 读（L2 miss）vs MTE2 端口字节

口径先自证（这是本轮第一个产出）：三列里只有一列是 DRAM 字节。

| 列 | 实测（ds-prefill-gate-up，prev 候选） | 含义 |
|---|--:|---|
| `GM_to_L1_datas(KB)` 求和 | **30.03 GB** | MTE2（GM→L1）端口字节，含 L2 命中 |
| `read_main_memory_datas(KB)` 求和 | 30.03 GB | **与上一行逐位相同** ⇒ 不是 DRAM 列，不可用 |
| `(aic_r0_read_cache_miss_allocate + aic_r1_read_cache_miss_allocate) × 128 B` | **24.99 GB** | **L2 读未命中 = 真 DRAM 读字节** |

交叉标定（同一套列、另一个算子）：`MoeGroupedGemmNopad` prefill-gate-up 上 `GM_to_L1` = 22.02 GB、
miss 求和 = 15.89 GB、`read_main_memory` = 22.02 GB —— 与 MoeGG v2 报告的 22.55 / 16.28~16.51 GB
同向同量级，故本轮的列选择与该报告一致。**下文所有 "GM 读" 都是 miss×128B。**

### GM vs MTE2 端口对照表（kernel calibre，平衡 A/B `prev,new,new,prev` 的中位）

| shape | Task Duration (µs) | MTE2 端口 (GB) | GM 读 = L2 miss (GB) | L2 读命中率 | 不可压缩下界 W+A 各 1 遍 (GB) | GM 读 ÷ 下界 | A 端口解析值 (GB, 占端口) | B 端口解析值 (GB, 占端口) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3-prefill gate_up（prev） | 21477.9 | 26.11 | **21.40** | 17.8% | 7.99 | **2.68×** | 9.25 (35%) | 18.50 (71%) |
| deepseek-prefill gate_up（prev） | 26298.4 | 30.03 | **24.99** | 16.2% | 15.50 | **1.61×** | 11.16 (37%) | 22.31 (74%) |
| qwen3-prefill gate_up（new） | 13256.4 | 26.11 | **7.99** | 69.3% | 7.99 | **1.00×** | 9.25 (35%) | 18.50 (71%) |
| deepseek-prefill gate_up（new） | 18119.2 | 30.03 | **15.51** | 48.0% | 15.50 | **1.00×** | 11.16 (37%) | 22.31 (74%) |

判定与后续动作：

1. **"GM 读 ≈ 理论下界 → A 重读已命中 L2 → 直接 no-go" 的门槛没有被触发**。prev 候选的 GM 读是
   下界的 2.68× / 1.61×（MoeGG 那次是 16.28 / 15.50 = 1.05×，所以那条路在那里无效）。⇒ 进 §2 改造。
2. **但"按 (expert, n-tile) 重排任务序"也不是正解**：端口字节 26.11 / 30.03 GB 与任务序无关，
   而当时 MTE2 busy 已达 **0.973 / 0.970** ⇒ 时间就锁在端口字节 ÷ 端口吞吐上。任何纯重排（把 A
   重读变成 L2 命中）只会降低 DRAM、不会降低端口字节，**在端口已满时必然无收益**。这与 MoeGG
   v2 的结论同源，但机理更硬：那次是"命中率已经很高、无事可做"，这次是"端口才是墙、省 DRAM 无用"。
3. **口径更正（旧报告）**：v2 报告写的「A 复用占其 GM 流量约 66%」**无法复现**。按实测端口口径 A
   占 33%~37%（旧 `bn=128` 时约 50%）；按 DRAM 口径 A 的上界也只有 `A端口/总端口` 那么多。
   `num_pid_n × numel × K = 7.52 GB` 对 `B 重读 15.03 GB` 是 1:2，不是 1:0.5。该 66% 数字本轮起
   停止引用。
4. 三份报告同时写「prefill 属设计层、不在 Stage 4 授权内」——本轮证明它是**任务索引分配**的单行
   bug 级问题，属 Stage 4 权限内，已就地修掉。

## 8.2 真正的现象：核占用率只有 80.7% / 73.1%

`kernel_plan()` 用**上界** `max_tiles = numel//block_m + num_experts`（每个专家都要预留自己的
M 尾块取整）算 `num_local_tasks`，而 live tile 只有 `total_tiles * num_pid_n`。死槽是 flat 索引的
**连续尾部**；`tile_id = core_id * num_local_tasks + t` 又是**每核一段连续区间**，于是死槽全部落在
尾部核上：

| shape | total_tiles × num_pid_n（live） | max_tiles × num_pid_n（静态界） | 静态每核任务数 | 实测忙碌核数 | 核占用率 |
|---|--:|--:|--:|--:|--:|
| qwen3-prefill gate_up | 5040 | 6144 | 308 | 16 满 + 1 个 36% = **17/20** | **80.7%** |
| deepseek-prefill gate_up | 6079 | 8192 | 410 | 14 满 + 1 个 82% = **15/20** | **73.1%** |

（忙碌核数与时占比取自 `Memory_*.csv` 的逐核 `aic_time(us)`：qwen 的 block 17/18/19 只有 4 µs，
block 16 为 7.7 ms / 21.4 ms；ds 的 block 15~19 只有 5 µs。）

改成 strided 索引 `tile_id = t * nums_kernel + core_id` 后：整条 flat 空间按 20 取模重新分配，死槽
（仍是尾部连续段，且长度是 20 的整数倍：qwen 1104 = 55.2×20，ds 2113）被 20 个核**均匀**分摊 ⇒
每核 `ceil(live/20)` 个 live tile（qwen 252、ds 304），**20 核全开**。同时任一瞬间 20 个核拿到的
是 20 个**连续** tile_id = 1.25 个 M 块的全部 16 个 N 段 ⇒ 同一块 A 被 16 个核**同时在途共享**，
这正是 §8.1 表里 new 列 GM 读掉到下界 1.00× 的原因（A 重读全部命中 L2，B 的 1.23×/1.48× 重读
也全部命中）。

## 8.3 本轮迭代记录（候选 vs current best，`msprof op` Task Duration）

current best（round 7）= `v2_d2_pair_bnshape`；本轮所有分支都从它派生，每个分支只改一个主要点。

| 候选（只改一个主要点） | shape | Task Duration (µs) | Task Duration 相对 current best | GM 读 (GB) | 端口 (GB) | AICore(cube busy) | L0 | 判定 |
|---|---|--:|--:|--:|--:|--:|:--:|---|
| current best `v2_d2_pair_bnshape` | qwen-prefill-gate_up | 21477.9 | — | 21.40 | 26.11 | 0.509 | PASS | base |
| **C1 strided flat 任务序** | qwen-prefill-gate_up | **13256.4** | **−38.3%** | **7.99** | 26.11 | **0.675** | PASS | ✅ 采纳 |
| current best | ds-prefill-gate_up | 26298.4 | — | 24.99 | 30.03 | 0.553 | PASS | base |
| **C1 strided flat 任务序** | ds-prefill-gate_up | **18119.2** | **−31.1%** | **15.51** | 30.03 | **0.596** | PASS | ✅ 采纳 |
| C2「A 块常驻 L1」 | — | 未实现 | — | — | — | — | — | ⛔ 尺寸否决：A 块 128×7168×2 B = **1.75 MB** > L1 512 KB（K 越大越不可行）|
| C3「减小 K 分块的 A 重读」 | — | 未实现 | — | — | — | — | — | ⛔ 前提不成立：每个 task 的 K 循环完整走一遍全 K，A 每 task 只读一次，无 K 维重读 |

C2/C3 是**算术否决**而不是跑失败：C2 的 1.75 MB 与 L1 512 KB 差 3.4×，且 D2 已占 384 KB；C3 的
前提（K 分块导致同一 A 行被多次读）在本 kernel 里不存在。

AICore 利用率的来源：`Peak` 口径下 `perf = fops / duration`，fops 与 tile 数成正比（cube 对每个
tile 都跑满 `block_m` 行，`size=[tile_m,...]` 并不裁剪 MMA），所以 cube busy 0.509→0.675 与本表
时延同源；**注意这不是新增算力，只是同一份 tile 工作被塞进更短的时间**。

## 8.4 全 8 个真实 GEMM shape（gate_up / down 分开，单次采集；`--only` 详见 `logs/r8_disc/sweep*.json`）

| shape | current best (µs) | strided (µs) | Δ | Ratio 前 → 后 |
|---|--:|--:|--:|--:|
| qwen-prefill gate_up | 21425.7 | **13241.3** | −38.20% | 56.15 → 65.58% |
| qwen-prefill down | 9940.8 | **6847.2** | −31.12% | 56.98 → 63.53% |
| ds-prefill gate_up | 26177.5 | **18135.2** | −30.72% | 53.61 → 57.81% |
| ds-prefill down | 12371.3 | **9068.9** | −26.69% | 54.29 → 57.87% |
| qwen-decode gate_up | 7825.8 | **6912.5** | −11.67% | 56.10 → 60.42% |
| qwen-decode down | 3946.3 | **3323.4** | −15.78% | 54.52 → 64.05% |
| ds-decode gate_up | 14306.7 | **13609.1** | −4.88% | 59.96 → 62.84% |
| ds-decode down | 7038.3 | **6407.2** | −8.97% | 60.27 → 65.92% |

**8/8 同向改善，无一条回退**（阈值 3%）。

## 8.5 三个使用方算子的 op 级验收（`msprof op`，钉 `--kernel-name main_mix_aic`）

### 8.5.1 `FusedMoEExpertsNopadPersistent3WGFwdOp` — run `20261026_165452_977795`，status=passed

| case | v2 µs | v3 µs | Δ | v2 Ratio | v3 Ratio |
|---|--:|--:|--:|--:|--:|
| qwen3-235b-decode | 7849.5 | **6907.3** | **−12.00%** | 55.91% | **61.30%** ✅ |
| qwen3-235b-prefill | 21428.2 | **13139.7** | **−38.68%** | 56.15% | **66.03%** ✅ |
| deepseek-v3-decode | 14332.0 | **13512.1** | **−5.72%** | 59.84% | **61.15%** ✅ |
| deepseek-v3-prefill | 26215.3 | **18207.8** | **−30.55%** | 54.19% | **58.26%** ⚠ |

→ **3/4 ≥ 60%**。deepseek-prefill 距 60% 还差 **2.9%** 时延（需 ≤ 17.68 ms）。

### 8.5.2 `FusedMoeFwdOp` — run `20261026_171736_070370`，status=passed

| case | v2 µs | v3 µs | Δ | v2 Ratio | v3 Ratio |
|---|--:|--:|--:|--:|--:|
| qwen3-235b-decode | 7874.7 | **6927.6** | −12.03% | 55.72% | **61.39%** ✅ |
| qwen3-235b-prefill | 21492.0 | **13291.1** | −38.16% | 56.40% | **66.01%** ✅ |
| deepseek-v3-decode | 14227.8 | **13522.5** | −4.96% | 60.28% | **62.60%** ✅ |
| deepseek-v3-prefill | 25929.4 | **17652.7** | −31.92% | 51.70% | **56.57%** ⚠ |

### 8.5.3 `FusedMoeFwdCbFwdOp` — run `20261026_174645_650909`，status=passed

| case | v2 µs | v3 µs | Δ | v2 Ratio | v3 Ratio |
|---|--:|--:|--:|--:|--:|
| kimi-k2-decode | 13672.6 | **10290.9** | −24.73% | 45.98% | **58.41%** |
| kimi-k2-medium | 20977.9 | **14461.0** | −31.07% | 44.72% | **48.69%** |
| kimi-k2-prefill | 26514.1 | **17450.4** | −34.18% | 53.74% | **61.00%** ✅ |

smoke（仅精度回归）：op1 7.58 / 7.58 µs，op2 7.48 / 7.78 µs，op3 7.44 / 7.50 µs。

## 8.6 精度与回归验收

| 检查 | 结果 |
|---|---|
| `python GroupedGemmPersistent3WG.py --level all` | **29/29 PASS**（L0 5/5、L1 10/10、L2 3/3、Boundary 11/11；L0-6 按设计 SKIP）。worst L0 `3.815e-06`、L1 `2.441e-04`、L2 `2.441e-04`、B `9.537e-07`，与 round 7 同量级 |
| `tests/ops/test_fused_moe_experts.py` | **20 passed** |
| `tests/ops/test_moe_fused_moe.py` + `test_moe_fused_moe_cb.py` | **20 passed**（两文件合计） |
| 三个算子的 op 级 harness | 全部 `status=passed`，`captured_op_names == ['main_mix_aic']` |
| strided 索引的覆盖性 | `tile_id ∈ [0, num_local_tasks × nums_kernel) = [0, num_logical)`，与原索引是**同一集合的双射**，只改顺序 ⇒ 每个 tile 仍恰好算一次 |

## 8.7 测量侧与诚实边界

- **平衡 A/B**：`prev,new,new,prev` 各 2 次（`logs/r8_disc/ab_final_r8.json`）。prev 21489.2 / 21466.6、
  new 13259.4 / 13253.5（qwen）；prev 26308.8 / 26288.0、new 18098.8 / 18139.7（ds）。重复性
  ≤0.2%，效应 −31%~−38%，比噪声高两个数量级。
- **一次测量侧失败（已处置，与本改动无关）**：`FusedMoeFwdCbFwdOp` 第一次 op 级 run
  `20261026_174140` 的 correctness 2/8 失败，失败点是**兄弟算子 `MoeGroupedGemmNopadKernel`**
  （`block_m=64/bn=256/bk=64`）的 `bishengir-compile` SIGSEGV，其默认重试预算由 `MOE_COMPILE_ATTEMPTS`
  控制、默认只有 **6**，两次都 6/6 打满。设 `MOE_COMPILE_ATTEMPTS=30` 后同一份代码
  `20261026_174645` **status=passed**。**与本 kernel 无关**；本条同时说明本仓的两套重试预算
  （`PGG_COMPILE_ATTEMPTS` / `MOE_COMPILE_ATTEMPTS`）都必须显式上调。
- **deepseek-prefill 未达标（58.26%）如实记录**，不宣称达成。剩余缺口见 §8.8。
- **口径提醒**：Ratio 是 msprof 原生 Roofline（分子 = GM 读+写 = L2 miss，分母 = min(算力 roofline,
  访存 roofline)）。本轮改动把两条 prefill 从**访存 roofline 侧**推到**算力 roofline 侧**
  （qwen: AI 109 → 287 Ops/Byte，mem-bound 上限 197 TFLOPS < 算力 272 TFLOPS）—— 也就是说
  "Ratio 提升" 里既有时间下降、也有 roofline 侧切换的成分，**跨算子/跨轮次只做同式对比**。

## 8.8 deepseek-prefill 还差什么（量化，不做宣称）

- 端口字节 **30.03 GB 与改动前逐位相同**，MTE2 busy 0.882 ⇒ 时间 ≈ 30.03 GB ÷ 端口吞吐。
  GM 读已 = 下界 15.51 GB，**A 重读这条路彻底关闭**（这是本轮最重要的负向结论之一）。
- 端口字节的结构下界由 `block_m · block_n ≤ 32768`（L0C 128 KB）钉住：`B 端口 = total_tiles · N · K · 2`
  只随 `total_tiles` 变，而 `total_tiles = Σ ceil(s_e/block_m)` 随 `block_m` 变小而暴涨。枚举：
  `(bm,bn) = (128,256)` → 30.03 GB（现配置）；`(256,128)` → 45 GB；`(64,512)` → 42 GB。
  `bn` 必须整除 N=4096（否则掉到 serial 兜底），所以 `(128,256)` 就是该约束下的最优点。
- 剩下的可动项是 **MTE2 空闲的 11.8%**（qwen 侧是 19.3% 但基数小）。按 busy 期吞吐 1.88 TB/s 反推，
  把空闲全部吃掉可到 ~16.0 ms（Ratio ≈ 62.8%），**上界 ≈ 1.13×**，远小于 2× ⇒ 不构成设计层天花板，
  **不产出 `perf_feedback.md`**。空闲的来源是每个 task 的尾部（最后一次 `gemm` + `store_fixpipe`
  没有被任何 copy 覆盖，约 60 µs/task 中的 ~2~3 µs），要吃掉它必须做**跨 task 的软件流水**
  （在当前 task 末尾预取下一个 task 的首个 chunk），需要在 `T.serial` 体内引入运行期条件的
  `fixpipe`/copy —— 正是 CG-2026-0001 的崩溃域，本轮不做（收益 3~5% vs 破坏 −31% 定稿的风险，
  判定不划算）。
- 另注：ds 的 `fops / 理想 fops = 1.484`（`total_tiles=380` vs 理想 256），是**每专家 M 尾块取整**
  造成的 cube 空算；它推高 Ratio 的分子（perf = fops/时间）而不推高时间，因此**不是**可优化项，
  反而是"别去修"的项（修掉会让 Ratio 变差）。

## 8.9 Round 8 产物

- 定稿：`perf_opt/GroupedGemmPersistent3WG.py`（sha256 `72fbb33e3bed75296abdf09ede04abbb582afb859cbcd2625646748a3f7d500a`）
- 改前快照：`history_version/20261026_prefillA/`（py + opt_log + perf_records + SHA256.txt）
- 判别/对照探针：`probe_r8_discriminate.py`（GM vs 端口双列 + A/B 端口解析）、`probe_r8_strided.py`（单点实验体）
- raw profile（`perf_opt/profiles/r8/`）：
  * `r8final__{shape}__{prev,new}`（4 个目录，**完整**含 `visualize_data.bin`）—— 平衡 A/B 与 §8.1 判别表的原始来源；
    `prev` 目录同时就是 §8.1 判别表的 prev 候选数据（故未再单独留 `r8disc`）。
  * `sweep_csv_only/r8sweep__*`、`sweep_csv_only/r8sw2__*`（16 个目录，只存
    `Memory/L2Cache/PipeUtilization/OpBasicInfo` CSV）—— §8.4 八 shape 表的原始来源；为控制仓库体积未存
    `visualize_data.bin`，其 Roofline Ratio 逐条记录在 `logs/r8_disc/sweep_decode.json`。
- 数值底稿：`logs/r8_disc/ab_final_r8.json`、`logs/r8_disc/sweep_decode.json`、`logs/r8_disc/disc_v1.json`
- 过程日志：`logs/r8_disc/`（`level_all.log`、三个 op 的 `cli_r8_*.log`、各 shape 的 msprof stdout）
- **append-only 披露（本文件与 `perf_records.jsonl` 的唯一一处触碰）**：`perf_records.jsonl` 里
  Round 7 的 4 条 `phase=final` 被降级为 `phase=merged`（**只改标签 + 加 `note`，测量值一字未改**），
  原因是 gate 要求「每个 workload 恰有一条当前最终版本的 final 记录」；同时 Final Performance Test
  Data 表的 `final_us` / `final_candidate_id` 两列换成本轮值（Round 7 的 14331.974 / 26215.334 /
  7849.503 / 21428.191 仍在 `perf_records.jsonl` 的 merged 行里可查）。

## 8.10 Skill Retrospective（Round 8 增量）

- **[R] 改结构之前先把"这一列到底是什么字节"验一遍**：本仓 msprof 的 `Memory_*.csv` 里
  `read_main_memory_datas` **等于** `GM_to_L1_datas`（端口），真正的 DRAM 列要用 `L2Cache_*.csv` 的
  `*_read_cache_miss_allocate × 128 B` 自己算。用一个已知答案的算子（MoeGG）做交叉标定，5 分钟成本，
  能避免整轮方向错误。vp_type=R，evidence=§8.1 三行，repro=`probe_r8_discriminate.py`。
- **[R] "端口满" 与 "DRAM 满" 是两个不同的墙，决定完全不同的方向**：端口 busy ≥0.97 时，任何
  只降 DRAM 的改动（重排/提高命中）**收益恒为 0**；只有降端口字节或降空闲才有用。判据：
  `aic_mte2_ratio` vs `GM 读/时长 ÷ 1.8`。vp_type=R，evidence=§8.1 判定 2。
- **[P] persistent kernel 的静态上界与 live 工作量之间的缺口，必须按"均匀分摊"而不是"每核连续"分配**：
  `num_local_tasks = ceil(max_tiles · num_pid_n / cores)` 里的 `max_tiles` 是含每专家尾块的上界，
  死槽是 flat 空间的连续尾部；`core*L + t` 会把它们全堆给尾部核（本算子实测 17/20、15/20 核），
  `t*cores + core` 则 20 核均分且顺带拿到跨核 A 块共享。**这一条与 MoeGG 的"核内连续 2.02× 回退"
  不矛盾**：MoeGG 的基线本来就是 strided，把它改成连续才是回退；本算子基线是连续，改成 strided 才是修复。
  判据：`live = total_tiles · num_pid_n` vs `max_tiles · num_pid_n`，以及 `Memory_*.csv` 里各核
  `aic_time` 是否等长。vp_type=P，evidence=§8.2 表 + §8.4 八行。
- **[D] 新实测常数（CANN 9.0.0 / 910_9362）**：strided 索引后 qwen3-prefill-gate_up 的 MTE2 端口
  26.11 GB/13.26 ms = **1.97 TB/s**（L2 命中 69.3%），ds-prefill-gate_up 30.03 GB/18.12 ms =
  **1.66 TB/s**（命中 48.0%）；GM 读分别落到下界 7.99 / 15.51 GB。vp_type=D，
  repro=`probe_r8_discriminate.py --variants new --only qwen_prefill_gate_up,ds_prefill_gate_up`。
- **[C] 案例索引**：MoE expert-GEMM 的"prefill 打不到 60%"有两种完全不同的根因 ——
  ① 端口字节被 L0C 容量钉死（MoeGG prefill-gate-up，无解）；
  ② 核占用率被静态上界吃掉（本算子，一行修复 −31%~−38%）。先用 `aic_time` 逐核时长区分，再决定投入。

---

# Round 9 -- deepseek-v3-prefill 从 58.26% 推到 ≥60%

- 起点（Round 8 定稿 `perf_opt/GroupedGemmPersistent3WG.py`，sha256 `72fbb33e…3f7d500a`）：
  ds-prefill 18199.7 µs（probe 同口径 18190.5）/ Ratio 58.26%，另 3 条已 ≥60%。
- 缺口：Ratio 60% ⇒ 时延 ≤ 17.68 ms（−2.9%）。量化上界（§8.8）：端口 30.03 GB ÷ 1.97 TB/s ≈ 15.2 ms，
  MTE2 busy 0.881 ⇒ 11.9% 空闲。**本轮只改 `perf_opt/`；baseline / wrapper / 算子代码一字未动。**
- 环境：独占 Chip1（davinci13），`ASCEND_RT_VISIBLE_DEVICES=1`，
  `TILEOPS_MSPROF_OUTPUT_DIR=/tmp/opencode/msprof-out-3wg5`。
- 唯一判优口径：`msprof op --kernel-name main_mix_aic` 的 `Task Duration(us)`。
- 定稿改前备份：`history_version/20261026_dsprefill/`（Round 8 定稿原样留档）。

## 9.1 先做 ramp 分解（上一次调度已完成，数据在 `logs/r9/r9ramp.json`）

固任务表（`total_tiles` 不变），只改 K，测每 task 的固定开销：

| variant | K | Task Duration (µs) | 端口 (GB) | 端口带宽 (TB/s) | mte2 | cube |
|---|--:|--:|--:|--:|--:|--:|
| k7168 | 7168 | 18176.8 | 30.03 | 1.652 | 0.8806 | 0.5937 |
| k3584 | 3584 | 9159.2 | 15.02 | 1.640 | 0.8567 | 0.5889 |
| k1536 | 1536 | 4156.9 | 6.44 | 1.548 | 0.8192 | 0.5563 |

线性拟合 `T = n_task·(fixed + K/w)`（n_task = 304 live/core）：
边际速率 `w` 对应 **1.86 TB/s / 20 核**，每 task 固定开销 **0.47 µs**（外推 K→0 只剩 141 µs = **0.8%**）。

> **结论（负向，推翻本轮计划里的假设②）**：MTE2 那 11.9% 空闲**不在 task 边界上**。
> 跨 task 预取（把下一个 task 的首段 copy 提到当前 task 末尾之前）最多只能收回 **0.8%**，
> 收益远小于 CG-2026-0001 / TRAP-DEVMODE-PERSIST-GEMM 的风险 ⇒ **不做**，理由由实测数据给出。

## 9.2 ① tile 重扫（`logs/r9/r9tile.json`、`logs/r9/r9bm.json`）

单点探针，ds-prefill gate_up（numel=32768, E=256, N=4096, K=7168），`probe_r9_tile.py`
（只把 `_select_block_n` 改成直通，使 bn 不被形状策略改写）。L1 约束 `2·(bm+bn)·bk·2 ≤ 512 KB`，
L0C 约束 `bm·bn·4 ≤ 128 KB` ⇒ `bm·bn ≤ 32768`。**(128,256,512) 需 768 KB L1，已排除。**

| 候选 (bm,bn,bk) | Task Duration (µs) | Δ vs current best | 端口 (GB) | 端口带宽 (TB/s) | mte2 | cube | 备注 |
|---|--:|--:|--:|--:|--:|--:|---|
| **(128,256,256) ← current best** | **18190.5** | — | 30.03 | 1.651 | 0.881 | 0.593 | Round 8 定稿 |
| (256,128,256) | 18166.9 | −0.1% | 30.47 | 1.677 | 0.888 | 0.801 | 同字节，cube 打到 0.80 |
| (128,256,128) | 20788.8 | +14.3% | 30.03 | 1.445 | 0.930 | 0.530 | bk 减半，chunk 数翻倍 |
| (64,512,128) | 27076.0 | +48.9% | 41.21 | 1.522 | 0.976 | 0.359 | 端口涨到 41 GB |
| (128,128,256) | 21195.8 | +16.5% | 37.75 | 1.781 | 0.886 | 0.520 | A 重读翻倍 |
| **(192,128,256)** | **16534.4** | **−9.1%** | 30.47 | **1.843** | **0.966** | 0.665 | 每专家 1 个 M-tile |
| (256,128,128) | 21051.6 | +15.7% | 30.47 | 1.447 | 0.894 | 0.705 | |

block_m 细扫（bn=128 / bk=256，`logs/r9/r9bm.json`）：

| bm | 144 | 160 | 176 | 192 | 208 | 224 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|
| µs | 17042.1 | **16421.7** | 16453.9 | 16539.8 | 16596.8 | 16917.1 | 18128.3 |
| 端口 GB | 31.29 | 30.53 | 30.47 | 30.47 | 30.47 | 30.47 | 30.47 |
| mte2 | 0.978 | 0.974 | 0.972 | 0.966 | 0.963 | 0.947 | 0.885 |

**机理（本轮主发现）**：ds-prefill 的 `s_e ~ Binomial(32768, 1/256)`，均值 128 = `block_m`。
`ceil(s_e/128)` 让 **124/380 个 tile 只有 ~9 行**（M 尾块），这些 tile 照样要把整个 B 块
（`block_n×K`）读一遍：端口 22.3 GB 的 B 流量里有 **7.3 GB** 是给这些尾块读的，
且 cube:copy 比例严重失衡 ⇒ MTE2 有 12% 无处可藏。把 `block_m` 抬到 `> max(s_e) ≈ 170` 后
每个专家恰好 1 个 M-tile（`total_tiles` 380 → 256），任务全均匀，mte2 0.88 → 0.97，
端口带宽 1.65 → **1.85 TB/s**，时间 −9.1%。

**但 bm 是全局常量，不能按 shape 改**（见 9.3），所以这条只能作为"如果放开 wrapper"的证据留存。

## 9.3 为什么 bm=176/192 不能直接落地（约束核对，不是性能问题）

`wrapper` 用 `self.config["block_m"]` 同时算 **host 侧 tile 表**（`max_tiles`、`ceil(s_e/block_m)`、
`tile_row_offsets`）和传给 factory 的 `block_m`；`self.config` 来自
`_DEFAULT_CONFIG["block_m"] = TUNED_DEFAULT_CONFIG["block_m"]`（**import 期求值，全局唯一**）。
只有 `block_n` / `block_k` 不进 tile 表，才能在 factory 内按 shape 重解（`_select_block_n` 那条路）。
⇒ 在"不动 wrapper"的约束下，per-shape `block_m` **无法表达**；全局改 `block_m` 的实测代价：

| 全局配置 | qwen-prefill gate_up | ds-prefill gate_up |
|---|--:|--:|
| (128,256)（现定稿） | **13241.3** | 18135.2 |
| (176,128) | 16047.1 (**+21.2%**) | **16431.0 (−9.4%)** |

qwen 的 `s_e ≈ 256`，需要 `bn=256` 来把 A 重读压到 16 遍；`bm=176` 因 L0C 128 KB 只能配 `bn=128`，
A 重读翻倍（7.5 → 15.0 GB）⇒ 端口 26.1 → 30.3 GB。两者互斥（`bm·bn ≤ 32768`）⇒ **全局切换否决**。

## 9.4 二次诊断：消除 M 尾块本身也不够（`logs/r9/r9uni.json`）

把 routing 换成 `dist=uniform`（`s_e ≡ 128`，每专家恰好 1 个满 tile，`total_tiles` 380 → 256）：

| 配置 | Task Duration (µs) | 端口 (GB) | 端口带宽 (TB/s) | mte2 | cube |
|---|--:|--:|--:|--:|--:|
| (128,256) routing | 18190.5 | 30.03 | 1.651 | 0.881 | 0.593 |
| **(128,256) uniform（端口 −25%）** | **17822.2** | **22.55** | **1.265** | 0.932 | 0.408 |
| (176,128) uniform | 16322.3 | 30.06 | 1.842 | 0.972 | 0.619 |

端口字节掉 25%、live task 掉 33%，时间只掉 2% ⇒ **(128,256) 并不是被端口字节钉住的**，
它有一个 ~17.8 ms 的独立下限；而 (176,128) 才是真正跑在端口极限（1.84 TB/s）上的。
换句话说：`bm=192` 的 −9% 主要来自**任务粒度/搬运形状**，不是"少读了 B"。

## 9.5 在当前几何（128,256）上找回空闲：滚动单 chunk 前瞻预取

`probe_r9_roll.py`：与定稿**逐行相同**，只把 K 循环的指令顺序从"成对 copy×4 + gemm×2"
改成滚动（每个 copy 提前一个 chunk 发出，仍用同样 4 个 L1 buffer、无 `T.Pipelined`、
无运行期条件 ⇒ 不进入 CG-2026-0001 的崩溃域）。

| shape | current best (µs) | roll (µs) | Δ | mte2 前 → 后 |
|---|--:|--:|--:|--:|
| ds-prefill gate_up | 18190.5 | **17948.9** | −1.3% | 0.881 → 0.906 |
| qwen-prefill gate_up | 13241.3 | **13076.5** | −1.2% | 0.807 → 0.824 |

−1.3% 是真实的但**不够**（需要 −2.9%）。它同时证明 9.1 的拟合：task 内 copy/gemm 配对只值 1.3%。

## 9.6 其它已测方向（全部负向，均为同口径 `msprof op` 单点）

| 变体 | 改动 | ds-prefill gate_up (µs) | Δ vs 18190.5 | 结论 |
|---|---|--:|--:|---|
| `probe_r9_bsplit.py` | B 的 copy 拆成两个半高（同 buffer 两段），只改 transfer 形状 | 18757.3 | +3.1% | 负向：128 KB 单次 ND2NZ 反而更优 |
| `probe_r9_bound.py` (R9_NUMTASK=304) | 把 `T.serial` 上界从静态上界 410 压到**恰好 live 的 304**（结果仍正确） | 18205.2 | +0.08% | **死槽完全免费**（对照 18196.1），"死槽吃时间"假设被排除 |
| `probe_r9_roll2.py` | 滚动预取，但 gemm 先发、copy 后发 | 17942.7 | −1.4% | 与 roll 等价 |
| `probe_r9_rollB.py` | 滚动预取 + chunk 体内 B 先发 A 后发 | **17939.9 / 17961.2** | **−1.4%** | 采纳（见 9.7） |
| `probe_r9_rollC.py` | rollB + task 序言里 B 先发（盖住两跳标量读链） | 17936.9 / 17958.8 | −1.3% | 与 rollB 等价（噪声内） |
| `probe_r9_rollD.py` | rollB + 剥离首个 gemm（循环内 initC 全静态 False） | **编译失败** | — | **确定性崩溃 10/10**，见下 |

`probe_r9_rollD.py` 的崩溃指纹（按 CG-2026-0001 / `TRAP-DEVMODE-PERSIST-GEMM` 纪律：**立即停，不反复试**）：

```
bishengir-compile ... died with <Signals.SIGSEGV: 11>
RuntimeError: NPU IR opt failed: PLEASE submit a bug report ... crash backtrace
crash pc chain: 0x...b51d4f4 ×4（自递归，无符号）
命中形态：in-core T.serial 任务循环里、"K 循环之前"放一个无条件的
         静态 initC=True 的 T.gemm（即把首块 gemm 从 K 循环里剥离）
attempts: 10/10 全崩（PGG_COMPILE_ATTEMPTS=10 打满），与 CG-2026-0033 的
          随机 flaky 形态不同 ⇒ 确定性、非重试可救
```

## 9.7 候选：滚动单 chunk 前瞻预取（`rollB`）——kernel 级胜出，**op 级验收后被否决，见 9.9.4**

平衡 A/B（`prev,new,new,prev`，同一 session，同口径）：

| 轮次 | cur (µs) | rollB (µs) |
|---|--:|--:|
| 第 1 对（cur→rollB） | 18198.4 | 17939.9 |
| 第 2 对（rollB→cur） | 18217.9 | 17961.2 |
| 均值 | **18208.2** | **17950.6** |

Δ = **−1.42%**；端口字节逐位不变（30.03 GB），`mte2` 0.881 → 0.885（+0.4pp），
带宽 1.650 → 1.673 TB/s。三个前序独立复测：17948.9 / 17898.5 / 17942.7（roll 家族内部离散 ≤0.3%）。

## 9.8 封档判定：60% 差在哪一条约束上（量化，不做宣称）

### 9.8.1 三条互相独立的硬约束（本轮全部实测复核，不是引用旧结论）

1. **L0C = 128 KB ⇒ `bm·bn ≤ 32768`**。实测 `(176,256)` / `(192,256)` / `(224,256)`
   （L0C 180 / 192 / 224 KB，L1 442 / 512 / 576 KB 均合法）**全部确定性崩 bishengir-compile 10/10**
   （`logs/r9/r9l0c/`）。⇒ accumulator 无法做大，`bm·bn` 封顶 32768。
2. **`bn` 必须整除 N ∈ {4096, 7168}** ⇒ `bn ∈ {128, 256, 512}`；结合 1 ⇒
   `bn=256 ⇒ bm ≤ 128`，`bn=128 ⇒ bm ≤ 256`。**没有第三种。**
3. **`block_m` 是全局常量，不能按 shape 重解**（9.3）。可在 factory 内按 shape 重解的只有
   `block_n` / `block_k`（它们不进 host 侧 tile 表）。

### 9.8.2 在 1+2 的可行域里，"每专家 1 个 M-tile"与"bn=256"不可同时成立

ds-prefill 的 `s_e ~ Binomial(32768, 1/256)`，`max s_e ≈ 170`：

| 方案 | total_tiles | A 端口 (GB) | B 端口 (GB) | 端口合计 (GB) | 实测 µs | mte2 | 达标 |
|---|--:|--:|--:|--:|--:|--:|:--:|
| (128,256) 现定稿 | 380 | 7.5 | 22.3 | **30.03** | 18190 | 0.881 | ❌ |
| (192,128) | 256 | 15.0 | 15.0 | 30.47 | 16534 | 0.966 | ✅ 但见 9.8.3 |
| (176,256) 理想 | 256 | 7.5 | 15.0 | **22.5** | — | — | **L0C 违规，编译崩** |

即：端口字节在 (128,256) 与 (192,128) 上**恰好持平**（30.0 GB），
(192,128) 更快纯粹是**达成了 1.85 TB/s 的端口速率**（mte2 0.966）而不是少读字节；
真正能再降 25% 字节的 (176,256) 被 L0C 封死。

### 9.8.3 为什么 (192,128) 不能落地：qwen3-prefill 的 A 重读

| 全局配置 | qwen3-prefill gate_up | ds-prefill gate_up |
|---|--:|--:|
| (128,256)（定稿） | **13241.3**（port 26.11 GB，比 2.00 TB/s） | 18135.2（port 30.03，1.65 TB/s） |
| (176,128) | 16047.1（**+21.2%**，port 30.27，1.89 TB/s） | **16431.0（−9.4%）** |
| (192,128) | 未测（同族，预期 ≈+20%） | 16534.4（−9.1%） |

qwen 的 `s_e ≈ 256`（需要 2 个满 tile），A 重读遍数 = `num_pid_n`：`bn=256` → 16 遍（7.5 GB），
`bn=128` → 32 遍（15.0 GB）。qwen 的端口已经贴着 2.0 TB/s，多出来的 7.5 GB 直接换成了 +16% 时延。
由于 `bm` 只能全局唯一（9.3），**两个 shape 的最优几何互斥** ⇒ 这条 −9% 的路在本 Agent 的写入域内不可达。

### 9.8.4 因此剩余可动项只有 task 内流水：实测上界 1.45%（已吃到）

- 每 task 固定开销 **0.47 µs = 0.8%**（9.1 的 K-sweep）⇒ 跨 task 预取没有可吃的东西（**不做**，理由见 9.1）；
- 死槽（26%）实测**免费**（9.6 的 `R9_NUMTASK=304` 对照：18205 vs 18196）；
- 搬运拆分（`bsplit`）**更差**（+3.1%）；
- 剩下的就是 chunk 内 copy/gemm 的排布 ⇒ 滚动预取吃到 **−1.42%**，已是该轴的实测上界；
- 想要再往下只能动 chunk 大小（bk=512 需 768 KB L1，违规）。用 `rollD`（剥离首 gemm）试图进一步压 chunk
  开销，被 bishengir **确定性 SIGSEGV** 否掉（9.6）。

**结论**：ds-prefill 的 60% 缺口**不是设计层天花板**（设计本身的端口下界 22.5 GB 需要 L0C > 180 KB，
与硬件的 128 KB 矛盾这一条属于硬件约束；而 Stage 4 可动范围内剩下的量级只有 1.4%），
实测可达上界 ≈ **1.13×（1.85 / 1.65 TB/s）**，而拿到这 1.13× 需要**改 wrapper 让 `block_m` 随 shape 变**
（一行 `_DEFAULT_CONFIG` 改成按 shape 解析），这超出本 Stage 的写入域 ⇒
**不产出 `perf_feedback.md`**（`_shared/standards/perf-feedback.md` §1 的双门槛：结构性加速 < 2× 且
设计假设未被推翻——设计假设"L0C 128 KB ⇒ bm·bn ≤ 32768"本轮只是被更严格地复核，并未被推翻）。

## 9.9 op 级验收（三个使用方，`msprof op` 钉 `--kernel-name main_mix_aic`）——候选被**否决**（最终裁决）

候选 `v4_rolling_prefetch`（= `probe_r9_rollB.py`，sha `64b2cfea…fec06f75` 的源码，测量时写入
`perf_opt/GroupedGemmPersistent3WG.py`、sha `0dd8a90c…99a9d55d`）跑完整 op 级验收：

### 9.9.1 `FusedMoEExpertsNopadPersistent3WGFwdOp` — run `20261027_115630_101074`，status=passed，20/20

| case | R8 µs | **R9 µs** | Δ 时延 | R8 Ratio | **R9 Ratio** | Δ Ratio |
|---|--:|--:|--:|--:|--:|--:|
| qwen3-235b-decode | 6907.3 | **6853.7** | −0.78% | 61.30% | **57.39%** | **−3.91pp** |
| qwen3-235b-prefill | 13139.7 | **12958.1** | −1.38% | 66.03% | **66.75%** | +0.72pp |
| deepseek-v3-decode | 13512.1 | **13438.6** | −0.54% | 61.15% | **60.40%** | −0.75pp |
| deepseek-v3-prefill | 18207.8 | **17981.9** | −1.24% | 58.26% | **58.88%** | +0.62pp |

### 9.9.2 `FusedMoeFwdOp` — run `20261027_121959_136102`，status=passed，12/12

| case | R8 µs | R9 µs | Δ 时延 | R8 Ratio | R9 Ratio |
|---|--:|--:|--:|--:|--:|
| qwen3-235b-decode | 6927.6 | 7050.7 | **+1.78%** | 61.39% | 57.18% |
| qwen3-235b-prefill | 13291.1 | 13100.6 | −1.43% | 66.01% | 66.97% |
| deepseek-v3-decode | 13522.5 | 13553.1 | +0.23% | 62.60% | 61.74% |
| deepseek-v3-prefill | 17652.7 | 17430.9 | −1.26% | 56.57% | 57.29% |

### 9.9.3 `FusedMoeFwdCbFwdOp` — run `20261027_124528_151982`，status=passed，11/11

| case | R8 µs | R9 µs | Δ 时延 | R8 Ratio | R9 Ratio |
|---|--:|--:|--:|--:|--:|
| kimi-k2-decode | 10290.9 | 10309.2 | +0.18% | 58.41% | 59.97% |
| kimi-k2-medium | 14461.0 | 14582.6 | +0.84% | 48.69% | 49.95% |
| kimi-k2-prefill | 17450.4 | 17264.1 | −1.07% | 61.00% | 61.78% |

### 9.9.4 裁决：**不采纳**（回退到 Round 8 定稿），并给出两条独立的理由

1. **目标未达成**：ds-prefill 58.26% → **58.88%**（op1）/ 56.57% → 57.29%（op2），仍 < 60%。
   −1.2% 的时延收益远小于 §9.8 量化的 1.13×（需 −11.9%）。
2. **`bn=128` 的 decode 族是真回退**（不是噪声）：单 kernel 探针
   （`logs/r9/r9dec.json`，同一 session、同口径）显示

   | shape | cur µs | rollB µs | Δ | 端口 GB | GM 读 GB |
   |---|--:|--:|--:|--:|--:|
   | qwen_decode_gate_up | 7150.9 | 7177.9 | +0.38% | 9.60 / 9.60 | 7.65 / 7.63 |
   | **ds_decode_gate_up** | **13341.6** | **13659.8** | **+2.39%** | 17.31 / 17.31 | 15.18 / 15.14 |

   机理：decode 的 `tile_m ≈ 16`，A 段只有 ~4 KB，而 B 段 64 KB。rollB 把 B 提到 A 之前发，
   gemm 却要等 A —— **小 A 被大 B 挡在队列后面**，每个 chunk 都加一段等待。
   prefill（`tile_m=128`，A 64 KB）没有这个效应，所以 prefill 受益、decode 受损。
   （`rollC`/`roll` 的 A 先发版本在 decode 上不再受这一条影响，但 prefill 收益相同 ⇒
   最优折中就是"按 shape 选 copy 顺序"，而这属于新的一层 shape 策略；本轮没有把它做完，
   因为在 ds-prefill 都上不到 60% 的前提下，为 +1.2% 再引入一层 shape 分支不划算。）

### 9.9.5 测量侧告警（重要，供后续轮次与 gate 解释用）：msprof 原生 Ratio 在 decode 上不稳

同一份代码、同一份字节数，`Ratio` 可以相差 10%：`qwen_decode_gate_up` 的 `cur` 与 `rollB`
端口字节**完全相同**（9.60 / 9.60 GB）、GM 读相同（7.65 / 7.63 GB）、时延差 0.38%，而
msprof 原生 Ratio 是 **56.67 vs 62.56**。反解该比值的定义可得 `Ratio = perf/computility`，
`computility = min(AI·1.8, 272.4)`，其中 `AI = cube_fops/bytes` 用的是 **cube 采样计数器**，
逐 run 抖动 ±10%。因此：

- **prefill 的四条（AI 落在算力侧、`AI·1.8 < 272.4` 的邻域稳定）Ratio 与时延严格反比**，
  本节 9.9.1 的 prefill 两行（+0.72 / +0.62pp）与时延一致的 −1.24 / −1.38% 可互相印证；
- **decode 的 Ratio 逐 run 抖动 ±10%**（9.9.1 的 −3.91pp 与 9.9.3 的 +1.56pp 是同一量级的抖动），
  不能当作 kernel 属性的判据；对 decode 只能用 `Task Duration` 判优 —— 这也正是本 skill
  「唯一时延口径 = Task Duration」的原因。

## 9.10 Round 9 结论（封档）

1. **ds-prefill 60% 在 Stage 4 写入域内不可达**，卡在两处硬件/契约约束（9.8）：
   `L0C 128 KB ⇒ bm·bn ≤ 32768`（本轮 3 个超限配置全部确定性崩编译）＋ `bn | N`；
   以及 `block_m` 是 wrapper 级全局常量、不能按 shape 重解。
2. **能解锁 −9% 的最小改动**（已实测、留给 conductor 决策）：把 `block_m` 变成按 shape 解析
   （`mean_rows = numel//E`；`mean_rows + 3·sqrt(mean_rows) ≤ 192` 时用 192，否则 128），
   即 wrapper 里 `_DEFAULT_CONFIG["block_m"]` 需要按 shape 求值、host 侧 tile 表随之重建。
   实测依据：ds-prefill (192,128) 16534.4 µs / **−9.1%** / mte2 0.966；同族 (176,128) 16431.0 / −9.4%；
   qwen-prefill 在该几何下 +21.2%（因为 A 重读翻倍）⇒ **必须按 shape 分派**，全局常量做不到。
   **该改动不在本 Agent 授权范围内，本轮未实施，wrapper / baseline 一字未动。**
3. `perf_feedback.md` **不产出**：实测可达上界 1.13×（1.85 / 1.65 TB/s），< 2×；
   且 DESIGN 的 `bm·bn ≤ 32768` 假设本轮只是被更严格复核，**未被推翻**
   （`_shared/standards/perf-feedback.md` §1 双门槛均不满足）。

## 9.11 Round 9 产物与回退

- **定稿：未变** —— `perf_opt/GroupedGemmPersistent3WG.py`，sha256
  `72fbb33e3bed75296abdf09ede04abbb582afb859cbcd2625646748a3f7d500a`（= Round 8 定稿；
  Round 9 候选写入后已按 9.9.4 回退，回退后 sha 逐位相同、`--level L0` 5/5 PASS 复核）。
- 候选与探针（全部留在 `perf_opt/`，可复现）：
  * `probe_r9_scan.py` —— 统一 runner（preset：ramp / tile / tile9 / bmscan / x176 / roll / bsplit /
    bound / l0c / order / ab*），`--json-out` 落 `logs/r9/*.json`；
  * `probe_r9_tile.py` —— `_select_block_n` 直通 shim（原始 tile 扫描用）；
  * `probe_r9_roll.py`（A 先发）/ `probe_r9_rollB.py`（B 先发）/ `probe_r9_rollC.py`（+task 序言 B 先发）/
    `probe_r9_roll2.py`（gemm 先发）/ `probe_r9_rollD.py`（剥离首 gemm，**编译崩**）/
    `probe_r9_bsplit.py` / `probe_r9_bound.py`；
  * `probe_p1_case.py` / `probe_p1_run.py` / `probe_r8_discriminate.py` 复用（未改）。
- 数值底稿：`logs/r9/r9ramp.json`、`r9tile.json`、`r9bm.json`、`r9l0c.json`、`r9uni.json`、`r9x176.json`、
  `r9bsp.json`、`r9ord.json`、`r9dec.json`；每个配置的 msprof 命令行与 stdout 在 `logs/r9/<tag>/`。
- 精度：候选 `--level all` **29/29 PASS**（`logs/r9/level_all/level_all_r9.log`）；回退后的定稿
  `--level L0` **5/5 PASS**。
- op 级：3 个 run 全部 `status=passed`、correctness 20/20 / 12/12 / 11/11
  （`reports/tileops/20261027_115630_101074_FusedMoEExpertsNopadPersistent3WGFwdOp/`、
  `…/20261027_121959_136102_FusedMoeFwdOp/`、`…/20261027_124528_151982_FusedMoeFwdCbFwdOp/`）。
- **raw profile 说明（如实）**：`TILEOPS_MSPROF_OUTPUT_DIR=/tmp/opencode/msprof-out-3wg5` 被 op 级
  harness 复用并清空，因此 9.2~9.6 的中间 OPPROF 目录已不在盘上；每个配置的命令行 + stdout 全量保存在
  `logs/r9/<tag>/*.log`，结构化数字在 `logs/r9/*.json`。末次解码对照（`r9dec`）的 CSV 已拷到
  `perf_opt/profiles/r9/`。

## 9.12 Skill Retrospective（Round 9 增量）

- **[R] "idle 在 task 边界"这类判断必须先用手算上界证伪，再决定要不要冒崩溃风险**：本轮计划里的
  ② 是"跨 task 预取"，但 K-sweep 三点拟合给出的每 task 固定开销只有 0.47 µs（0.8%），
  11.9% 的 idle 全部在 task **内部**。省下的是 1 次 ~1 小时的实现 + CG-2026-0001 的崩溃风险。
  做法：固定任务表、只改 K（`--preset ramp`），三点线性拟合即可分离 `n_task·fixed` 与 `bytes/rate`。
  vp_type=R，evidence=§9.1，repro=`probe_r9_scan.py --preset ramp`。
- **[P] 小 tile 的"M 尾块"是 MoE 分组 GEMM 的一等性能变量，判据是 `s_e` 分布与 `block_m` 的关系**：
  `s_e ~ Binomial(numel, 1/E)`、均值正好等于 `block_m` 时，`ceil(s_e/bm)` 会造出一大批 ~9 行的尾 tile，
  它们读满整个 B 却不做活 ⇒ 端口字节不能降（B 已经绑定 total_tiles），但**端口速率**从 1.85 掉到 1.65 TB/s。
  判据：`max(s_e) vs bm` 与 `Memory_*.csv` 的 `GM_to_L1_datas/时长`。ds-prefill 上把 `bm` 抬到 192 得 −9.1%。
  vp_type=P，evidence=§9.2/9.8.2。
- **[P] 增加"每专家 1 个 M-tile"这一杠杆与"bn 加倍复用 A"是互斥的，本质是 L0C 容量**：
  `bm·bn ≤ 32768`，所以 `bm ↑` 必然 `bn ↓`；B 端口 ∝ `total_tiles`、A 端口 ∝ `num_pid_n`，
  两者的最优点只在"`bm` 恰好超过 `max(s_e)`"处重合。跨 shape 共用一份 `block_m` 时，
  这个点落在 qwen（`s_e≈256`）与 ds（`s_e≈128`）之间无解。vp_type=P，evidence=§9.8.2/9.8.3。
- **[C] 案例索引**：MoE expert-GEMM 打不到 60% 的第三种根因 —— ① 端口字节被 L0C 钉死（MoeGG）；
  ② 核占用率被静态上界吃掉（本算子 R8，一行修复）；③ **`s_e` 分布与 `block_m` 撞车造出 M 尾 tile**
  （ds-prefill R9，需按 shape 定 `bm`）。区分法：看 `total_tiles` 是否显著大于 `ceil(numel/bm)`。
  vp_type=C，evidence=§9.2。
- **[R] 验收口径的稳定性必须与优化本身一起验**：msprof 原生 Ratio 在 decode 上逐 run 抖 ±10%
  （同码同字节数：56.67 vs 62.56），所以"Ratio 回退 >3%"不能单独作为否决依据，
  必须回到 `Task Duration`；反过来，若只按时延判优，就会把 decode 上真实 +2.4% 的变体留下。
  本轮因此把两条标准并列写进 9.9.4。vp_type=R，evidence=§9.9.5。
- **[R] 顺序类微优化要按 shape 的 chunk 组成验，不能只看主目标 shape**：rollB 在 ds-prefill 上 −1.4%，
  在 ds-decode 上 +2.4% —— 差别只在 `tile_m`（128 vs 16）让 A 段从 64 KB 变成 4 KB，
  于是"B 先发"从有利变成挡路。任何 copy 顺序调整都必须同时测 prefill 与 decode 两端。
  vp_type=R，evidence=§9.9.4 表。
- **[R] 崩溃域纪律有效**：`rollD`（剥离首 gemm）确定性 SIGSEGV 10/10，按 `TRAP-DEVMODE-PERSIST-GEMM`
  立即停手、只记指纹，没有陷入重试。指纹与触发形态已记入 §9.6，可直接作为 pattern-library 的 negative 条目。
  vp_type=R，evidence=§9.6。

### 9.11.1 回退后对**定稿**的复核（本轮实测，非引用 Round 8）

| 验收项 | 结果 | 证据 |
|---|---|---|
| `python GroupedGemmPersistent3WG.py --level L0` | **5/5 PASS**，worst max_diff `2.441e-04` | 本节下方命令输出 |
| `pytest tests/ops/test_fused_moe_experts.py` | **20 passed**（224.95 s） | `perf_opt/logs/r9/pytest_three_consumers.log` |
| `pytest tests/ops/test_moe_fused_moe.py` | **12 passed**（231.24 s） | 同上 |
| `pytest tests/ops/test_moe_fused_moe_cb.py` | **8 passed**（205.97 s） | 同上 |
| `--level all`（候选体，供对照） | 29/29 PASS | `logs/r9/level_all/level_all_r9.log` |
| `statectl gate 4` | `{"ok": true, "pass": true, "failures": [], "warnings": []}` | conductor 侧复核 |

定稿 sha256 `72fbb33e…3f7d500a` 与 Round 8 完全一致，故 §8.6 的 29/29 与三个 pytest 记录继续有效；
上表是本轮在**回退后的文件**上重新跑的独立复现（不是引用）。

### 9.11.2 补充：A 先发版（`roll`）在 decode 上的测量**未完成**（按崩溃域纪律停手）

§9.9.4 的结论建立在 `rollB`（B 先发）上，因此追加了一组"A 先发版 `roll` 在 decode 上是否中性"的对照。
结果（如实）：

| 配置 | 结果 |
|---|---|
| `ds_decode_gate_up` / cur | 13510.1 µs（port 17.31 GB，ratio 62.97，mte2 0.951） |
| `ds_decode_gate_up` / roll(A 先发) | **msprof 子进程挂起**：日志只有命令行、无任何 `Task Description`，~48 min 无输出，被 timeout 杀掉（`logs/r9/r9decA/ds_decode_gate_up__roll.log`，654 B） |

按 `CG-2026-0033` / `TRAP-DEVMODE-PERSIST-GEMM` 纪律**不再重试**。本轮共出现 2 次"msprof 空输出"，
两次都在 roll 家族（第一次是 `qwen_prefill_gate_up` 的 `roll`，重跑后才成功），加上 `rollD` 的
确定性编译崩溃 ⇒ **roll 家族在 decode 形状（`bn=128`、`tile_m≈16`）上带有稳定性风险**，
这也是不采纳它的第三条理由（前两条见 9.9.4）。⇒ 「A 先发版在 decode 上中性」这一条**未获证据**，
不得当作已证结论引用；若后续要拿这 +1.3%，需要先在 decode 上把稳定性问题定位清楚。

### 9.12.1 流程自省（本轮的过程缺陷，如实）

- **[R] 本轮没有在 Phase 0 读 pattern-library 的 `constants.md` / `INDEX.md`，代价是可量化的**：
  事后检索发现两条本可省下时间的条目 ——
  * `constants.md` L286「910_9362 的 L0C=128 KiB、L1=512 KiB（两侧夹逼）… 144×256 报
    `cc overflow … 1048576 bits`」：本轮又用 3 个配置 × 10 次编译把它重验了一遍（§9.8.1）；
  * `constants.md` L336「EMTE2 下界 = 2×(A+B 填充)/(核数×150 GB/s)，A=A_size×ceil(N/bn)
    （`bm·bn≤32768`），B=B_size×ceil(s_e/bm)；prefill-gate-up 22.55 GB → 15.95 ms，误差 <1%」：
    正是 §9.8.2 的端口模型，条目里连 `(bm,bn)=(176..192,128)` 这一族的字节账都已有；
  * `PL-1.10-loads-first-decoupling`（前置装载解耦）与本轮 §9.5 的滚动预取同源，虽然那条是
    Vector 族（单 staging 复用）的案例，但"先装载、后计算"的判据可迁移。
  **规则**：Stage 4 进入扫描之前，先 `kb_search.py "<算子族> <症状> <hardware>"` 查常数与陷阱，
  再动手扫参；本轮把已入库的常数当未知量重测，是纯浪费。vp_type=R，evidence=§9.8.1/§9.8.2 与
  上述两条库条目（本轮**未**在实现前读到，事后核对一致）。
