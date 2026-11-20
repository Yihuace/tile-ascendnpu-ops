# `_logical_reduce_kernel`（CountNonzeroFwdOp）Stage 4 调优日志

> 口径：全文时延 = **`msprof op` `Task Duration(us)`**（kernel-only，`--launch-count=30`、
> `--warm-up=5` 的 30 次发射均值；同一 profile 目录内 `OpBasicInfo_*.csv` 逐块取均值）。
> 不采集 NPU event、不用 runner 端到端时间。Ratio 一律标注分母口径：
> **`/1.75 TB/s`**（任务书/官方 report 口径）与 **`/0.8 TB/s`**（本机 msprof
> `visualize_data.bin` 自己的 GM/L2 屋顶值，见 §6）。

## 0. 环境与工件

| 项 | 值 |
|---|---|
| 日期 | 2026-10-18 |
| tilelang | 0.1.2 + `c316b0e6b8a1ddd13a6927b70f38a614e8b0474d`（dev root build `/workspace/tilelang-mlir-ascend/build`） |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post8 |
| CANN / driver | 9.0.0 / 26.1.1 |
| 设备 | `npu-smi` 报 **910B4**（Chip 0，HBM 32768 MB）；`AscendArch()` → `chip_name=Ascend910B`、`aivector_core_num=40`、`ub_cap=196608 B` |
| 官方 report（只读参考） | `reports/tileops/20261016_015623_851978_CountNonzeroFwdOp/report.md`（环境写 Ascend910B3，GM 屋顶 1.75 TB/s） |
| benchmark 入口 | `benchmarks/ops/bench_count_nonzero.py::test_count_nonzero_bench` |
| 展开函数 | `workloads_to_params("CountNonzeroFwdOp", include_extra=True)` |
| 基准 kernel（只读） | `../_logical_reduce_kernel.py`（sha256 `a5fef64f…`） |
| 交付 kernel | `perf_opt/_logical_reduce_kernel.py`（round-4 版本，sha256 `faae175e…`，见 §7.1 / §9.5） |
| 结构化记录 | `perf_records.jsonl`（24 行 = baseline 4 + candidate 16 + final 4；最终版本每 tune workload 恰 1 条 final） |
| 全量原始记录 | `perf_records_full_history.jsonl`（24 行，含被 round-4 取代的 4 条 round-3 final，一字不改；见 §7.1） |
| 标定记录（非 tune，**不混入** perf_records） | `copy_floor_calibration.jsonl`、`op_cost_calibration.jsonl` |
| raw profile | `profiles/{baseline,candidate,probes,final}/…`、`profiles/round0,round1,round2`、round-4 增量：`profiles/ab_round4/{op1,op2}/`（交错 A/B 原始）、`profiles/final/round4/`、`profiles/diag_round4/`（fp16 同码重复性） |
| 过程日志 | `logs/{round0,round1,round2,round3,probes}/…` |

### 0.1 实跑 workload 清单与 shape 核对（必查项）

`pytest --collect-only` 与 `workloads_to_params` 实跑展开 **7 例**：

| workload_id | shape | dtype | dim | 分类 |
|---|---|---|---|---|
| smoke-count-nonzero-float16 | [32, 256] | fp16 | -1 | smoke |
| smoke-count-nonzero-bfloat16 | [32, 256] | bf16 | -1 | smoke |
| smoke-count-nonzero-float32 | [32, 256] | fp32 | -1 | smoke |
| sparsity-hidden-float16 | [2048, 4096] | fp16 | -1 | tune |
| sparsity-hidden-bfloat16 | [2048, 4096] | bf16 | -1 | tune |
| sparsity-seq-float16 | **[32, 16384]** | fp16 | -1 | tune |
| 3d-multidim-reduce-float16 | [4, 128, 4096] | fp16 | [0,2] → 2D `(128, 16384)` | tune |

**核对结论**：`sparsity-seq` 实跑宽度就是 **16384**（不是 GPU manifest 的 32768），
与本仓 `tileops/manifest/reduction.yaml` 一致；下文所有基线/终值均按 16384 记录，
无需换算。3d 用例经 Op 层 `movedim(keep=[1]) → contiguous → reshape(128, 16384)`，
kernel 视角 M=128、N=16384。smoke 仅做精度回归，不采性能、不参与调优（manifest label
含独立 token `smoke`，非按 shape 推断）。

### 0.2 测量装置

`bench_driver.py`（`--impl` 显式加载任意候选文件 → `_select_config` 取 block_m →
`_logical_reduce_kernel(M,N,"count_nonzero",dtype)(block_m)(x)`）在 `msprof op
--kernel-name=main` 下运行；`run_bench.py` 解析 `OpBasicInfo*.csv` 的
`Task Duration(us)`（mean/median/min/max）；`run_experiments.py` 做「L0 → msprof →
perf_records 追加 → 对比表」批处理（T-4）。

---

## 1. Performance Test Data（baseline，`phase=baseline`, round 0）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|
| `_logical_reduce_kernel::main` | sparsity-hidden-float16 | baseline | main | main | 26.014 | valid | `profiles/baseline/round0/baseline_8ba8496a/sparsity-hidden-float16_e65d0e6b` |
| `_logical_reduce_kernel::main` | sparsity-hidden-bfloat16 | baseline | main | main | 31.135 | valid | `profiles/baseline/round0/baseline_8ba8496a/sparsity-hidden-bfloat16_8270be6b` |
| `_logical_reduce_kernel::main` | sparsity-seq-float16 | baseline | main | main | 4.408 | valid | `profiles/baseline/round0/baseline_8ba8496a/sparsity-seq-float16_64d4544b` |
| `_logical_reduce_kernel::main` | 3d-multidim-reduce-float16 | baseline | main | main | 10.234 | valid | `profiles/baseline/round0/baseline_8ba8496a/3d-multidim-reduce-float16_1d80d3b7` |

命令模板：

```bash
msprof op --kernel-name=main --output=profiles/baseline/round0/<cand>/<wl> \
  --launch-count=30 --warm-up=5 --dump=off \
  --aic-metrics=<BasicInfo,PipeUtilization,ArithmeticMemory,…> \
  python bench_driver.py --impl ../_logical_reduce_kernel.py --case <wl> --launches 35
```

（`logs/round0/*.log` 保存每次 `CMD` 与 stdout/stderr。）

### 1.1 基线重复性与 run-state 双态

同一 baseline 在 2 个独立时刻各采一次（均 30 发射均值）：

| workload | baseline-run-A（round 0，`perf_records` 采用） | baseline-run-B（round 1 复采） | 离散 |
|---|---:|---:|---:|
| hidden fp16 | 26.014 | 27.074 | 4.1% |
| hidden bf16 | 31.135 | 30.904 | 0.7% |
| seq fp16 | 4.408 | 4.613 | 4.7% |
| 3d fp16 | 10.234 | 10.261 | 0.3% |

即 **hidden-fp16 / seq 处于 ±5% 的 run-state 双态**（BP_run_state_bimodality）。凡候选差异
<5% 的结论均按 §4 的交错 A/B/A/B 协议裁决，不以单 run 排序。官方 report 的
22.74/29.20/4.50/9.66 与本机 A 组相比：hidden fp16 +14%、bf16 +6.6%、seq −2%、3d +6%
——**除 hidden fp16 外都在双态范围内**；hidden fp16 的差异与 §6 的屋顶带宽差异同向（本机
GM 屋顶 0.8 TB/s，report 环境 1.75 TB/s）。

### 1.2 设计估算 vs 实测偏差（D-2 回填，DESIGN §1.6.0 roofline）

| workload | DESIGN 估算下界 | 实测 baseline | 偏差 | 失准项 |
|---|---:|---:|---:|---|
| hidden fp16/bf16 | 45–90 µs | 26.0 / 31.1 | **实测优于估算 1.7–3.5×** | 发射项按 0.25–0.5 µs/op 串行上界叠加，实际 MTE2/向量/MTE3 有三 pipe 部分重叠 |
| seq fp16 | 6–11 µs | 4.41 | 实测优于估算 1.4–2.5× | 同上；且 4.4 µs 中 2.44 µs 是空 kernel 地板 |
| 3d fp16 | 16–30 µs | 10.23 | 实测优于估算 1.6–2.9× | 同上 |
| smoke (32,256) | 4–6 µs | 未采（smoke 不做性能） | — | — |

**结论**：设计期对「发射项」的悲观上界估计使下界偏保守；但**流量项**（30.2 GB/s/core ×
40 核 = 1.2 TB/s）的假设方向相反——实测 AIV 侧 GM 读屋顶只有 ≈0.7–0.8 TB/s（§6），
设计估算在带宽侧高估 1.5×。该偏差是 §6 `[DESIGN_LIMIT]` 的量化证据之一。

---

## 2. 调优顺序与理由

| 顺序 | workload | 理由 |
|---|---|---|
| 1 | sparsity-hidden-float16 / bfloat16 | 实测负担最大（26/31 µs，占 4 例总时延 68%），流量项占比最高，结构性收益面最大 |
| 2 | 3d-multidim-reduce-float16 | 次大（10.2 µs），`dim=[0,2]` 几何 → M=128/N=16384，走列分块路径，是 DESIGN §1.6.4 备选的主战场 |
| 3 | sparsity-seq-float16 | 绝对时延 4.4 µs 已贴近空 kernel 地板 2.44 µs，先做地板标定再决定是否投入 |

排序只影响先后，未跳过任何 tune workload；smoke 不进入性能闭环。

---

## 3. 结构化标定（非 tune，独立 jsonl）

hidden `(2048,4096)` fp16 上，用 `probe_floor.py` 的四种同构探针（同 persistent 网格、
同 block 尺寸）逐项剥离成本；`--launch-count=15~20`：

| bm | 每核任务数 | `empty`（仅逐任务输出小 DMA） | `copyout`（+GM→UB 16 KB 读） | `copy`（读+写同 tile） | 全链（baseline） |
|---:|---:|---:|---:|---:|---:|
| 1 | 52 | 12.741 | 29.171 | — | — |
| 2 | 26 | 6.672 | 25.267 | 24.760 | 26.014 |
| 4 | 13 | 4.495 | 23.825 | — | 编译溢出（见 §4 OP3） |

seq / 3d 的同构探针（bm=1）：

| workload | empty | copyout | read（copy+reduce_sum） | copy（R+W） | 全链 |
|---|---:|---:|---:|---:|---:|
| seq (32,16384) fp16 | 2.438 | 2.860 | 3.478 | 2.812 | 4.408 |
| 3d (128,16384) fp16 | 4.808 | 7.394 | 8.331 | 7.416 | 10.234 |

**反解出的单价（写入 `op_cost_calibration.jsonl`）**：

- **逐任务输出小 DMA（8–32 B）**：0.16–0.20 µs/op（`empty` 斜率；26 任务 → 6.7 µs，
  但它与 MTE2/向量**部分重叠**，删除它并不能缩短总时延，见 §4 OP1）。
- **GM→UB 读（AIV/MTE2）**：16 KB tile → 0.72 µs（22.4 GB/s/核 ≈ 0.90 TB/s 聚合）；
  8 KB → 0.32 µs（26 GB/s/核）；32 KB → 1.49 µs（21.5 GB/s/核）。**块越大单价越高**
  ⇒ 读侧是带宽受限而非每-op 固定开销受限。
- **向量 pass（`vcmp`+`vcast`+`reduce_sum` 链）**：12.33 µs / (26 任务 × 3 pass) =
  **0.158 µs/pass**（≈0.019 ns/元素），低于 `CONST-vector-launch-overhead` 的
  0.23–0.3 µs/pass 低档。
- **空 kernel 地板**：seq 单任务形态 **2.44 µs**；40 核单波 ≈2.4 µs（与
  `CONST-msprof-task-floor-910B3` 的 3.37 µs 同量级但更低：本例每核只有 1 个任务、
  输出 DMA 仅 8 B）。

> 这些探针**不是** tune workload，只写 `copy_floor_calibration.jsonl` /
> `op_cost_calibration.jsonl`，不进入 `perf_records.jsonl`。

---

## 4. Iteration Log

### Iteration 1 — 三个候选（batched output / R′ / 更大元素预算）

- current best = Stage 3 baseline（`../_logical_reduce_kernel.py`）
- 现象（baseline profile，hidden fp16）：`aiv_time 22.56` µs，`vec_ratio 0.55`、
  `mte2_ratio 0.51`、`mte3_ratio 0.51`——三 pipe 各占约一半，`vec+mte2=23.9` ≈
  `aiv_time`，即 **MTE2 与向量几乎没有重叠**；`mte3` 11.54 µs 全部来自 26 个 16 B 小写。
- 候选：
  - **OP1** 批量输出：每核连续行块 + `out_all` 暂存 + 末尾一次 `T.copy`（消掉 25 个
    16 B 写）。
  - **OP2** = OP1 + 路径 R′（`E_CAP < N ≤ UB 元素上限` 时 `bm=1` 单 tile 驻留，
    DESIGN §1.6.4 备选，A/B 必做项）。
  - **OP3** 元素预算 8192 → 16384（hidden 的 bm 2→4）。

| 候选 vs current best | workload | Task Duration(us) | AICore pipe（vec/mte2/mte3 ratio） | memory（GM in/out KB/核） | L0 | 结论 |
|---|---|---:|---|---|---|---|
| current best（baseline） | hidden fp16 | 26.014 | 0.55 / 0.51 / 0.51 | 409.6 / 3.2 | pass | — |
| OP1 | hidden fp16 | 27.171 (−0.4%) | 0.54 / 0.49 / **0.01** | 409.6 / **0.49** | pass | config_no_gain（mte3 已完全重叠） |
| OP1 | hidden bf16 | 33.999 (**−10.0%**) | 0.63 / 0.34 / 0.01 | 409.6 / 0.49 | pass | **rollback** |
| OP1 | seq fp16 | 4.478 (−1.6%) | — | — | pass | tie |
| OP1 | 3d fp16 | 10.753 (**−4.8%**) | — | — | pass | regressed |
| OP2 | hidden fp16 | 27.077 (−0.0%) | 0.55 / 0.51 / 0.01 | 409.6 / 0.49 | pass | tie |
| OP2 | hidden bf16 | 33.983 (**−10.0%**) | — | — | pass | rollback |
| OP2 | seq fp16 | **3.911 (+15.2%)** | — | — | pass | **improved** |
| OP2 | 3d fp16 | **8.893 (+13.3%)** | — | — | pass | **improved** |
| OP3 | hidden fp16 | — | — | — | **fail** | blocked（UB overflow，见下） |

- OP1 机制：批量输出确实把 `mte3` 从 11.54 → 0.28 µs、GM 写从 3.2 → 0.49 KB/核，但
  **总时延不变**（写 pipe 本来就被 vec/mte2 盖住）；反倒在 bf16 上因为新的 UB→UB
  `out_task→out_all` 拷贝压在已饱和的向量 pipe（vec ratio 0.68）上，**−10%**。→ 放弃。
- OP3 blocked：`16384` 元素/block 在本结构下需要 **1577216 bit = 197152 B > 196608 B**
  （**12.03 B/元素**，非 8 B/元素线性外推），`ub overflow` 编译失败。这同时说明
  `CONST-reduce-ub-per-element` 的「8.00 B/元素」只在**通过点**成立，外推要按实测
  通过点核对（本次实测通过点：fp16 23552 元素，见 L1-7d）。
- 本轮 winner：OP2（seq/3d 提升），但 bf16 回退说明批量输出必须剥离。

### Iteration 2 — 剥离 R′ 与批量输出（A/B 消歧）

| 候选 vs current best | workload | Task Duration(us) | AICore pipe（vec/mte2） | L0 | 结论 |
|---|---|---:|---|---|---|
| OP2（R′ + 批量输出） | seq fp16 | 3.911 | — | pass | 对照 |
| **OP2b（R′ + Stage 3 输出结构）** | seq fp16 | **3.893 (+15.6% vs r1 baseline)** | — | pass | **改进（形式更简）** |
| OP2（R′ + 批量输出） | 3d fp16 | 8.893 | — | pass | 对照 |
| **OP2b（R′ + Stage 3 输出结构）** | 3d fp16 | **8.957 (+12.7%)** | — | pass | **改进** |
| OP5（`T.Pipelined(num_stages=2)`，参考输出结构） | hidden fp16 | 25.92（screen 10 发射）/ 26.1（profile 内 TaskDur） | 0.55 / 0.51（**与 baseline 相同**） | pass | config_no_gain |
| OP5 | hidden bf16 | 31.02 | 0.68 / 0.37 | pass | config_no_gain |

- **消歧结论**：seq/3d 的收益 **100% 来自路径 R′**（去掉 tile 循环 + 1 个 tile 的
  微元累加），批量输出在 1–4 任务/核的形态下中性。→ 采纳 OP2b 结构（= Stage 3 结构 +
  R′ 分派），**不做**批量输出改造（它只在 bf16 上带来 −10%）。
- OP5：`T.Pipelined` 的 `aiv_time/vec/mte2` 与 baseline 逐项相同（22.70 vs 22.56）；
  auto-multi-buffer 与显式流水在本平台是**同一机制**，无叠加收益（与 mish / lerp 先例
  一致，记为 family 级结论）。→ `family_no_gain`。

### Iteration 3 — 合并与终态

`perf_opt/_logical_reduce_kernel.py` = Stage 3 kernel 结构 + `_select_config` 新增
`E_CAP < N ≤ _ub_elems(dtype)` 的 R′ 分支（`bm=1`、单 tile、无 tile 循环）。
逐 workload 合并/非回退检查与终值见 §7；smoke 三 dtype 精度回归 pass。

---

## 5. DESIGN §1.6.4 A/B 裁决 + 两个未实证常数的标定

| 项目 | 主选（路径 C：列分块 + 在线累加） | 备选 R′（超预算单 tile 驻留） | 比值 | 判定 |
|---|---:|---:|---:|---|
| seq (32,16384) fp16 | 4.408 µs（baseline 实测） / 4.478（OP1 同结构） | **3.893–3.911 µs** | **0.874–0.88×** | R′ 快 12–13% |
| 3d (128,16384) fp16 | 10.234 / 10.261 / 10.753 µs | **8.893–8.957 µs** | **0.87×** | R′ 快 13% |
| 合成 `(2048,16384)` | 未单独跑（预算内无收益假设） | 未单独跑 | — | 未测（以 manifest 两例为准） |

- **裁决**：R′ 在两个 manifest 形态上均胜出（−12.5% / −14.3%），但**未达到 DESIGN
  写定的「备选 < 0.8× 主选」翻转阈值**。按「实测更快的候选即 winner」的 Stage 4 规则
  仍采纳 R′（纯收益、无回退）；该阈值按设计期启发式记录，不构成放弃 12% 实测收益的
  理由；DESIGN §1.6.4 若走附录补记，应把阈值改为「< 0.9×」或注明「实测 0.87× 采纳」。
- **常数①「发射单价双档」实测**：本形态下向量 pass 单价 **0.158 µs/pass**
  （0.019 ns/元素），逐任务小 DMA 单价 0.16–0.20 µs/op，GM→UB 读 22.4 GB/s/核
  （16 KB tile）——即实测落在
  `CONST-vector-launch-overhead` 低档（0.23–0.3 µs/pass）**之下**，设计期 0.25–0.5
  的区间上界偏悲观 1.6–3.2×。
- **常数②「单消费者超预算是否竞态」实测**：R′ 的 16384 元素单缓冲形态必然放弃
  auto-multi-buffer（131072 B 单缓冲，双缓冲 262144 B > 196608 B）；在 seq（1 任务/核）
  与 3d（4 任务/核）上连续 30 发射 × 多次独立运行（`--check` 每次逐位比对）**结果
  恒定且与 `torch.count_nonzero` 逐位相等**（L0-3/L0-4/L1-7d 亦逐位精确）⇒
  **单消费者超预算形态无竞态**（同族此前的竞态样本都是多消费者形态）。
- **附带通过点**：`N=23552` fp16 单 tile（188416 B）编译并逐位正确（L1-7d）——
  即本结构的实测元素通过点 23552，溢出点 16384×(197152/131072) ≈ 24576 量级；
  `_ub_elems` 用 8 KB reserve 取 23552 是保守但可用的判据。

---

## 6. 设计层/口径天花板分析（`[DESIGN_LIMIT]` 依据）

### 6.1 本机屋顶口径（msprof 自己的 `visualize_data.bin`）

`roofline_from_profile.py` 读 `GM/L2 : GM Read + Write` 条目：本机 **`bw = 0.8 TB/s`**
（4 个 workload、baseline/final 全一致），即 msprof 认为本机 GM 屋顶是 0.8 TB/s。
官方 report 的环境同一字段是 **1.75 TB/s**（`report.md` Bandwidth 列），**两者相差
2.19×**——这是环境（910B3 vs 910B4 档 SKU）差异，不是本 kernel 的差异。

### 6.2 逐用例：目标线 vs 实测可达下界

| workload | 流量（读+写） | 70%-Ratio 目标（1.225 TB/s） | 本机屋顶地板（0.8 TB/s） | 实测结构地板 | final 实测 | Ratio/1.75 | Ratio/0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hidden fp16 | 16.79 MB | **13.71 µs** | 20.99 µs | 23.83 µs（read-only，bm=4） | 27.594 | 34.8% | 76.1% |
| hidden bf16 | 16.79 MB | **13.71 µs** | 20.99 µs | ≈24–25 µs | 31.169 | 30.8% | 67.4% |
| seq fp16 | 1.049 MB | **0.86 µs** | 1.31 µs | **2.44 µs（空 kernel 地板）** | 3.908 | 15.3% | 33.5% |
| 3d fp16 | 4.195 MB | **3.42 µs** | 5.24 µs | 7.42 µs（copy R+W） | 8.925 | 26.9% | 58.8% |

- **hidden**：目标 13.71 µs 比本机屋顶地板 20.99 µs 低 1.53×，比实测 read-only 地板
  23.83 µs 低 1.74×。kernel 已达本机屋顶的 **76.1%**（fp16）/ 67.4%（bf16），
  剩余空间只有 1–3 µs（见 §4：批量输出 0、显式流水 0、更大 tile 被 UB 封顶）。
- **seq**：目标 0.86 µs **低于空 kernel 地板 2.44 µs**（2.8×）——由单行比较直接判定
  口径不可达，任何算法都无解。
- **3d**：目标 3.42 µs 低于本机屋顶地板 5.24 µs（1.53×），也低于实测 copy 地板
  7.42 µs（2.2×）；且 128 行 /40 核 = 3.2 波，关键路径必然是 4 个任务。
- 结论：**4 个 report 用例的 70% Ratio 在本机口径下全部结构性不可达**；这不是
  tiling/参数问题（tiling 空间已在 §4 覆盖：元素预算、路径、输出组织、流水、核数）。
  详细 schema 见 `perf_feedback.md`。

### 6.3 工具链级常数重验（stale 清单前置项）

`kb_stale_check.py` 当前 `stale_count=289`；本任务引用到的条目全部为 `stale`
（toolchain 戳落后于 `c316b0e`）：`CONST-reduce-ub-per-element`、`PL-1.25-row-reduce-
indicator-complement`、`TRAP-vcmp-ne-ordered-nan`、`CONST-msprof-task-floor-910B3`、
`CONST-vector-launch-overhead`、`CONST-fabric-bw`、`CONST-reduce-ub-price-axis-
alignment`、`CONST-mte2-degradation`。

| 条目 | 本任务重验方式 | 结论 |
|---|---|---|
| `CONST-reduce-ub-per-element`（2-D 行归约 8.00 B/元素） | 编译探针：16384 元素溢出需 197152 B（12.03 B/元素）；23552 元素通过 | **部分推翻**：只在通过点成立，线性外推会低估 1.5×（见 §4 OP3） |
| `CONST-msprof-task-floor-910B3`（空 kernel 3.37 µs） | 本结构空 kernel 2.44 µs（seq 单任务） | **仍成立作为量级**；本形态更低（任务数少、写 8 B） |
| `CONST-vector-launch-overhead`（0.23–0.5 µs/pass） | 向量 pass 0.158 µs/pass | **方向成立**，本形态低于低档 |
| `CONST-device-peak-910B3`（AIV 单向读 0.75–0.78 TB/s） | 16 KB tile 22.4 GB/s/核 ×40 = 0.90 TB/s；本机 msprof 屋顶 0.8 TB/s | **同量级成立**（0.7–0.9 TB/s 带内） |
| `TRAP-vcmp-ne-ordered-nan`（eq 为 IEEE 正确） | NaN 行 L0-12 三 dtype 逐位（NaN 计入非零） | **仍成立**（本次未改比较路径） |
| `PL-1.25`（行归约指示链 + 双路径） | 双路径实测 + R′ 第三路径 | **成立并被扩展**（R′ 为第三形态） |

---

## 7. Final Summary

最终候选：`perf_opt/_logical_reduce_kernel.py`（`candidate_id = final_round5_fp16_bm4`，
sha256 `c075e1a12714f20d52c41eea1599800d27620fbbdfd33269d2b2206d9768cddd`）
= **Stage 3 kernel 结构 + `_select_config` 的 R′ 分派 + bf16 驻留 3 行档 + （round 5）fp16 驻留 4 行档**；
`_select_config` 仍是 tiling 单一真源，wrapper 无需改动。`op_kind != "count_nonzero"`
仍 raise；`_select_config` 已加入 `__all__`（供 wrapper 切换块直接 import）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---|---|---|
| _logical_reduce_kernel::main | sparsity-hidden-float16 | 26.014 | 16.280 | final_round5_fp16_bm4 |
| _logical_reduce_kernel::main | sparsity-hidden-bfloat16 | 31.135 | 23.913 | final_round5_fp16_bm4 |
| _logical_reduce_kernel::main | sparsity-seq-float16 | 4.408 | 3.849 | final_round5_fp16_bm4 |
| _logical_reduce_kernel::main | 3d-multidim-reduce-float16 | 10.234 | 7.651 | final_round5_fp16_bm4 |

> 上表 final 列 = round-5 在当前交付文件（sha `c075e1a1…`）上重采的 `phase=final`
> 记录原值（`perf_records.jsonl` round 5 四条，30 发射均值），未做拼接或挑选。
> **口径警告（必读）**：baseline 列是 round-0 在**另一台器件/会话**（910B4 级、旧工具链
> `+c316b0e`）上的实测；round-5 会话是 910B3 + `+84c6019`，同码 fp16 baseline 从
> 28.10（round-4 final）降到 22.49（本轮 A 臂）即 −20% 的纯 run-state/器件偏移。
> 因此跨会话的 baseline→final 差值**不是**本轮的提升量，唯一合法提升量是 §11 的同 run
> 交错 A/B（hidden fp16 −27.96%）；另外三例代码逐字节未变（§11.5 codegen 等同性），
> 其 final 变化同属会话偏移。

### 7.2 Round-5 final 重采说明（supersede，必读）

round 4 的 final 记录（`candidate_id = final_round4_bf16bm3`，sha `faae175e…`，
28.104 / 27.721 / 4.111 / 9.648）在 round 5 采纳 fp16 驻留 4 行档后失效——**处置**（全量留档 + 主文件唯一 final）：

1. round-4 的 4 条 final 行**一字不改**保存于 `perf_records_full_history.jsonl`
   （该文件现含 round 0–3 全部行 + round-3 被取代 final + round-4 被取代 final，共 28 行，
   append-only 语义在 sidecar 侧保持完整）；
2. 主 `perf_records.jsonl` 保留全部 baseline/candidate 迭代行（含 round-5 baseline 与
   candidate 行）+ 每 tune workload **唯一**权威 final 行（round 5，sha 与当前文件一致），
   共 26 行；
3. 本表逐 workload 的 `final_candidate_id` 与上述 round-5 final 行一致。
4. round-4 的交付文件本身另存为 `perf_opt/_logical_reduce_kernel_round5_prevbest.py`
   （sha `faae175e…`，与 `_logical_reduce_kernel_round4_op2_bf16_bm3.py` 同 sha），
   供任何后续回溯。

### 7.1 Round-4 final 重采说明（supersede，必读）

round 3 的 final 记录（`candidate_id = final_rprime`，sha `2a7552f6…`，27.594 / 31.169 /
3.908 / 8.925）在 round 4 采纳 bf16 驻留 3 行档后失效——**处置**（全量留档 + 主文件唯一 final）：

1. round-3 的 4 条 final 行**一字不改**保存于 `perf_records_full_history.jsonl`
   （该文件同时含 round 0–4 全部 24 行原始测量，append-only 语义在 sidecar 侧保持完整）；
2. 主 `perf_records.jsonl` 保留全部 baseline/candidate 迭代行（16 条 candidate + 4 条
   baseline）+ 每 tune workload **唯一**权威 final 行（round 4，sha 与当前文件一致）；
3. 本表逐 workload 的 `final_candidate_id` 与上述 round-4 final 行一致，`baseline` 列沿用
   round-0 baseline 行（sha `a5fef64f…`）。

**fp16 三个用例的 final 数值高于 round 3 的原因**：round-4 的改动只触及 bf16 分派
（`_select_config` 的 bf16 行阶梯），fp16 三条路径的**生成代码逐字节相同**——用
`get_kernel_source()` 在 `(2048,4096) fp16`、`(32,16384) fp16`、`(128,16384) fp16` 上逐一比对，
sha 分别为 `1fbac554…`/`5eab5ac8…`/`2465d293…`，round-3 与 round-4 两版 kernel **完全一致**
（只有 `(2048,4096) bf16` 不同：`76bb0c5a…` vs `458a98c3…`）。因此 fp16 的 final 差异属于
**run-state**：同一文件重复 3 次测得 hidden fp16 28.104 / 27.658 / 28.029、3d fp16
9.648 / 9.762 / 9.674（本轮机器状态比 round 0/3 的会话慢约 5–8%，bf16 A 臂同 run 也报
31.23 vs round-0 的 31.14）。按"唯一合法验收对照 = 同机同 run 交错 A/B"的口径，fp16 三例
**不构成回退**（代码相同），bf16 一例为真实改进（同 run 交错 A/B −10.55%，见 §9.5）。

**提升/回退（含口径说明）**

| workload | Δ vs baseline(run A) | Δ vs baseline(run B) | 裁决 |
|---|---:|---:|---|
| hidden fp16 | −8.0%（本轮 run-state 慢态） | −3.8% | **tie**：生成代码与 round-3 逐字节相同（§7.1）；round-3 交错 A/B/A/B 协议（3 对）得 A 中位 26.62 / B 中位 26.86、sign-test `p=1.00` ⇒ tie |
| hidden bf16 | **+11.0%** | +10.3% | **improved**（round-4 采纳 bf16 3 行档；同 run 交错 A/B/A/B 3 对：A 中位 31.235 / B 中位 27.940，方向一致，`b_rel_change = −10.55%`，见 §9.5） |
| seq fp16 | **+6.7%** | +10.9% | improved（R′；round-4 代码同 round 3） |
| 3d fp16 | **+5.7%** | +6.0% | improved（R′；round-4 代码同 round 3） |

> 说明：hidden fp16 的 final（28.104）落在 run-state 慢态（同码重复 27.66–28.10，
> baseline 自身在 26.0–27.6 摆动），按 T-2 协议以「同码等同性 + 重复 run 中位」裁决为 tie，
> 不以单 run 记回退；hidden bf16 的 final（27.721）与同 run 交错 A/B 的 B 中位（27.940）
> 一致，属真实改进。JSONL 中的 final 行是实测原值，未做任何拼接或挑选。
> baseline 第二轮复采（27.074 / 30.904 / 4.613 / 10.261）与 final 对照，结论同向
> （fp16 −3.8%、bf16 +10.3%、seq +10.9%、3d +6.0%）。

smoke（3 例，仅精度）：`smoke-count-nonzero-{float16,bfloat16,float32}` 全 pass
（`precision_pass=true`，见 `workload_inventory.json`；覆盖于 `--level all` 的
L0-1 与 Boundary 全零/全一用例）。

### 精度回归（`python perf_opt/_logical_reduce_kernel.py --level all`，exit 0）

- **L0：21 例全 PASS**（含 L0-12 NaN 语义三 dtype：NaN 计入非零、`-0` 不计入；
  L0-10 构造行：全零/全一/±inf/次正规；L0-3/L0-4 改为断言 R′ 路径且 `block_m=1`）。
- **L1：24 例全 PASS**（非连续输入、高阶/非末轴 dim 的 Op 层 movedim、极值/亚正规、
  bool/int32/int64/complex 的 float32 预转、M 尾块、L1-7a–d 覆盖路径 C 的静态尾块
  （tail=424/3616）与 R′ 边界 23552 元素）。
- **L2：全部 PASS**（工厂期 raise 5 项、最小形状 5 项、驻留 R(N=8192) ≡ R′(N=16384)
  逐位一致、N=E_cap+1）。
- **Boundary：全部 PASS**（三 dtype 全零/全一、1D、`(512,2048)`）。
- `torch.equal` 逐位比对（atol=rtol=0）；**NaN 语义用例保留且通过**——本次未改比较
  路径（仍 `vcmp("eq")` + 补式），未触碰 `"ne"`。

### 结论与中止原因

- 中止原因：**success（best-effort）+ `[DESIGN_LIMIT]`** —— 所有 tune workload 均有
  调优结论（3 例 winner_merged、1 例 no_gain），最终候选在同一文件上重采 final；
  70% Ratio 目标经实测判定为**口径不可达**（§6），按 `perf_feedback.md` 受控上送。
  round-4 起另按用户裁定的**双轨口径**（台账 1.75 TB/s 不变 / 本算子族验收线
  0.7 × 设备自报 0.8 TB/s = 560 GB/s）判验收，见 §9。
- 有效优化点（采纳）：路径 R′（`E_CAP < N ≤ UB 元素上限` 单 tile 驻留，`bm=1`，去
  tile 循环与微元累加）——seq −11.3%、3d −12.8%，hidden/smoke 零回退。
- 无效/回退优化点：批量输出（hidden bf16 −10%，且对被盖住的 mte3 无收益）、
  `T.Pipelined(num_stages=2)`（与 auto-multi-buffer 同机制，逐项指标不变）、
  `E_CAP=16384`（UB overflow，12.03 B/元素）、更大 tile（UB 封顶）。
- 残余机会（defer，未实施）：hidden 用 bm=4 + 计算缓冲二分复用（x_ub 4 行、compute
  2 行）可把读地板从 25.27 → 23.83 µs（≤6% 收益），但 final 记录已冻结（gate 4 要求
  每个 tune workload 恰一条 final），需下一轮以 `precision_fix`/续跑调度重开。
  **（round-5 已解决：新工具链下 plain 4 行组即可编译，不再需要二分复用；实测
  22.674 → 16.335 µs = −27.96%，远大于原估 ≤6%，见 §11。）**
- **round-4 补测（post-DONE，2026-10-18）**：bf16 hidden 采纳「驻留 3 行档」，
  31.235 → 27.940 µs（同 run 交错 A/B 3 对，−10.55%，方向一致）；该例状态由 no_gain
  转 winner_merged，final 记录按 §7.1 重采。逐项测量与验收口径见 §9。

---

## 8. Skill Retrospective

### 流程有效性

1. **Phase 1 的地板先行**（`empty/copyout/read/copy` 四探针）直接避免了两个错误方向：
   (a) 把 `mte3` 11.5 µs 当成真实瓶颈（批量输出实测 0 收益）；
   (b) 把 hidden 当"发射受限"去堆 op 数优化（实际是读带宽受限）。**探针成本 ~10 次
   msprof，换来 3 个候选方向的取舍**，性价比高。
2. **DESIGN 三件套 A/B 必做项**执行到位：R′ 在两个形态上 12–13% 胜出，是本次唯一
   有效结构优化。**但设计期写的 0.8× 阈值与"实测更优即采纳"的 Stage 4 规则冲突**——
   建议 skill 明确「设计期翻转阈值只约束 DESIGN 修订，不约束 Stage 4 winner 选择」。
3. **perf_records 的 final 唯一性**与"继续优化"天然冲突：一旦写入 final，任何后续改进
   都会造成 final 行重复或哈希失配。本次因此把 bm=4 残余机会记 defer。建议 skill 在
   Phase 3 明确「final 采集是单向门，采集前必须完成残余候选评估」。

### 需要修正的 skill 指引

- `run_experiments.py` 模板默认把 baseline 放进 EXPERIMENTS，导致 `--all` 会重复采
  baseline（本次产生 2 组 baseline 行）。建议模板区分 `--phase baseline` 与候选列表。
- 模板的 `KERNEL_NAME="main"` 对本次成立，但 `msprof_kernel_name` 属性（
  `count_nonzero_resident` 等）与 `--kernel-name=main` 不一致，建议注明「以
  `prim_func.name` 为准，`msprof_kernel_name` 只给 Op 层过滤用」。

### Value Point Proposals（按 vp_type 标注，含证据三件套）

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-A | D | **逐任务小 DMA 单价 0.16–0.20 µs/op**（8–32 B 输出写），但与 vec/mte2 部分重叠 ⇒ 删除它不缩短总时延；判别方法 = 同时看 `empty` 探针斜率与全链 delta | `copy_floor_calibration.jsonl`（bm=1/2/4 empty 12.741/6.672/4.495）、§4 OP1 的 mte3 11.54→0.28 µs 而 Task 不变 | `perf_opt/probe_floor.py`（任务工作区，允许失效） | c316b0e + CANN 9.0.0 + 910B4(40 AIV) |
| VP-B | D | **AIV GM 读单价随 tile 增大而变差**：8 KB 26 GB/s/核、16 KB 22.4、32 KB 21.5 ⇒ 本机读侧 0.7–0.9 TB/s 带内，与 msprof 屋顶 0.8 TB/s 一致 | `copy_floor_calibration.jsonl` bm 扫描 + `visualize_data.bin` bw=0.8 | 同上 | 同上 |
| VP-C | P（负向） | **超预算单 tile 驻留（R′）在 16384 元素档优于列分块 12–13%**：省 1 个 tile pass 与微元累加；且单消费者形态**无竞态**（30 发射 × 多 run 逐位恒定） | `perf_records.jsonl` r2_op2b（3.893/8.957）vs r1 baseline（4.613/10.261） | `_logical_reduce_kernel_round1_op2b_rprime_refout.py` | 同上 |
| VP-D | D | **2-D 行归约 UB 单价在 16384 元素档实测 12.03 B/元素**（16384 → 197152 B，溢出 544 B），8 B/元素外推会漏判 overflow | `logs/round1/L0_…op3…log` 的 `requires 1577216 bits` | `_logical_reduce_kernel_round1_op3_e16384_rprime0.py` | 同上 |
| VP-E | R | **final 采集是单向门**：gate 4 要求每 tune workload 恰一条 final 记录且哈希一致 ⇒ Phase 3 前必须穷尽候选或显式 defer | `perf_records.jsonl`（22 行 / 每 workload 1 条 final）+ 本次 defer 决策 | — | 流程规则，无工具链依赖 |

---

## 9. 验收口径（用户裁定 2026-10-18）

> 本章为 post-DONE 补测轮（round 4）的记录：§9.1–§9.4 = 用户裁定的**双轨口径 + 设备戳**，
> §9.5 = 本轮 (A) bf16 hidden 定向复测的**完整测量记录与结论**。既有轮次（§1–§8）一字未改，
> 唯一例外是 §7 的 Final 表按 §7.1 的 supersede 规则重采（原因与 sidecar 处置见 §7.1）。

### 9.1 双轨口径（原文）

1. **台账记账线保持不变**：`/workspace/全量13算子台账.md` 仍记 fabric **1.75 TB/s** 口径，
   本算子族不改台账数字。
2. **本算子族验收线改为「设备自报 roof × 70%」**：本机（910B4）设备自报 roof
   = **0.8 TB/s**（出处见 §9.2）⇒ 验收线流量 = **0.7 × 0.8 TB/s = 560 GB/s**。
3. **换算后的验收时延**（公式：`验收线(µs) = 总流量 B / 560e9 × 1e6`；
   总流量 = 输入字节 + 输出 int64 字节；本轮逐条复核、未抄表）：

| workload | 总流量 B | 验收线（≤ µs） | 当前同机实测（round-4 final） | 状态 |
|---|---:|---:|---:|---|
| sparsity-hidden fp16 | 16,793,600 | **29.99** | 28.104（final）/ 27.66–28.03（重复 3 次） | 达标（收） |
| sparsity-hidden bf16 | 16,793,600 | **29.99** | **27.721（final）/ 27.94（交错 B 中位）** | **本轮达标（收）** |
| sparsity-seq fp16 | 1,048,832 | 1.873 | 4.111 | 结构性（空 kernel 地板 2.44 µs ≈ 线 × 1.30），不再投入 |
| 3d-multidim fp16 | 4,195,328 | 7.49 | 9.648 | 结构性（copy R+W 地板 7.42 µs ≈ 线），不再投入 |

   流量核对：hidden = 2048×4096×2 + 2048×8 = 16,777,216 + 16,384 = **16,793,600 B**
   → 16,793,600 / 560e9 = **29.99 µs**；seq = 32×16384×2 + 32×8 = **1,048,832 B**
   → **1.873 µs**；3d = 4×128×4096×2 + 128×8 = **4,195,328 B** → **7.49 µs**。
   （与"1.75 TB/s × 70% = 1.225 TB/s"下的 13.71 / 0.86 / 3.42 µs 并列记录，
   两套数字并存：台账口径算 Ratio，族验收口径判达标。）

### 9.2 0.8 TB/s 的实测证据（为什么纯 Vector 算子不能拿 fabric 1.75 当分母）

- **出处①（设备自报）**：`msprof` 生成的 `profiles/{baseline,candidate,final}/…/visualize_data.bin`
  中 `GM/L2 : GM Read + Write` 条目 = **0.8 TB/s**（4 个 workload、baseline/final 两版
  profile 全部一致；读取脚本 `roofline_from_profile.py`）。
- **出处②（本次端到端 report）**：官方/report 侧 `Bandwidth (TB/s)` 列在本机同 run 下自动写
  **0.8000**（与 `visualize_data.bin` 同源）。
- **出处③（独立探针反解）**：`op_cost_calibration.jsonl` 的 AIV 读单价
  **22.4 GB/s/核 @16 KB tile × 40 核 = 0.90 TB/s**（8 KB → 26.0 GB/s/核 = 1.04 TB/s；
  32 KB → 21.5 GB/s/核 = 0.86 TB/s）；同族纯读探针地板 16.79 MB / 23.83 µs = **0.70 TB/s**。
  三个独立来源落在 **0.70–1.04 TB/s** 带内，与自报 0.8 一致。
- **与 fabric 1.75 的关系**：1.75 / 0.8 = **2.19×**。fabric 数字是**聚合交换带宽**
  （多引擎 / 跨 die 的 fabric 口径，AICore-Cube 的 L1/L0 通路与 MTE 聚合量级），而本算子
  的数据通路是**每核一个 AIV 的 MTE2 单向 GM→UB 流**（纯 Vector、无 matmul、不走 L1/L0C）。
  单个 AIV 的 MTE2 单流上限就是 20–26 GB/s/核，40 核再聚合也只有 0.8–1.0 TB/s ——
  **把 fabric 聚合带宽当分母等于要求单个 AIV 流跑出 2.19× 于其物理上限的带宽**，
  这是口径错误而非调优不足。故本算子族的验收分母取**设备自报的 AIV 侧 roof**。
- **旁证**：本 kernel 的 hidden fp16 已跑到自报 roof 的 **76.1%**（0.8 口径），
  而按 1.75 口径只有 34.8%；前者与"读带宽受限"的 profile 现象（mte2/vec 双 ~50%、
  纯读地板 23.83 µs vs 实测 26.0–28.1 µs）自洽。
- **端到端 report 的自洽核对**（只读参考，round-4 前采集
  `reports/tileops/20261018_053642_635995_CountNonzeroFwdOp/report.md`）：本机 report 的
  `Bandwidth` 列 4 行全为 **0.8000**，`Ratio` 列也按 0.8 计算——hidden fp16
  16.79 MB / 28.56 µs = 0.588 TB/s → **73.43%**（report 值），hidden bf16（**round-4 之前
  的旧版本**）30.42 µs → **68.94%**（刚好在 70% 线下）；对照 2026-10-16 的 910B3 report
  （Bandwidth 1.7500、hidden fp16 22.74 µs → Ratio 42.16%）。本轮 bf16 改后为
  16.79 MB / 27.72 µs = 0.606 TB/s → **75.7%**，越过 70% 族验收线。

### 9.3 设备戳（跨机不可比）

| 项 | 本机（本轮全部实测） | 官方 report 20261016 |
|---|---|---|
| 设备名 | **Ascend910B4**（`npu-smi info` 与 `torch.npu.get_device_name(0)` 一致） | Ascend910B3（report 环境） |
| AIV 核数 / UB | 40 / 196,608 B（`AscendArch()`） | report 同档 |
| 自报 GM roof | **0.8 TB/s** | **1.75 TB/s**（report Bandwidth 列） |
| hidden fp16 基线 | 26.01 µs（round-0 同机） | 22.74 µs |

- 同机基线比 report 慢约 **14%**（26.01 vs 22.74 µs），与 roof 差 2.19× 同向但幅度不同
  ——说明跨机差异同时含 SKU 档位与 fabric 口径两层因素，**Ratio 与绝对时延都不可跨机比较**。
- 因此：**唯一合法验收对照 = 同机同 run 的 before/after（交错 A/B）**；本机不同时刻之间
  也有 5–8% 的 run-state 漂移（§7.1 的 fp16 三条用例即为实证），单 run 数值只作记录、
  不作裁决。

### 9.4 口径变更的影响面

- **后续派发**：凡属「纯 Vector + 行归约 + 大 N 读主导」类（`count_nonzero` / `all` / `any`
  三个 logical-reduce 单元，以及池化、norm 族的大 N 读主导 shape），验收一律按
  **0.7 × 设备自报 roof（本机 = 560 GB/s）** 判定；**台账仍记 1.75 TB/s**（两轨并行，
  不互相换算，避免把 fabric 口径的 Ratio 当作可达性结论）。
- **`perf_feedback.md` 的关系**：§6 的 `[DESIGN_LIMIT]` 结论（1.75 口径 70% 不可达）
  仍然成立且不撤回；本裁定只是补上族验收线，**不推翻**设计层归因。
- **不改的东西**：DESIGN.md / 基准 kernel / wrapper / tests / benchmarks / manifest
  一律只读；pattern-library 与 capability-gaps 本轮起只读（沉淀改为提案，交 evolver 合入）。

### 9.5 本轮 (A)：bf16 hidden 定向复测 + 一次尝试

#### 9.5.1 现状归因（bf16 31.14 vs fp16 26.01，同 shape 同任务数）

两个 dtype 的配置完全相同（`(2048,4096)`：`path=R`、`block_m=2`、26 任务/核、
GM 流量 409.6 KB 读 + 3.2 KB 写/核），profile 逐项对比（round-0 baseline 同 run）：

| 指标 | fp16 | bf16 | 差 |
|---|---:|---:|---:|
| Task Duration | 26.02 | 30.78 | **+4.76 µs（1.18×）** |
| aiv_time | 22.66 | 28.61 | +5.95 |
| **vec_time（ratio）** | **12.34（0.545）** | **19.58（0.685）** | **+7.24（+59%）** |
| mte2_time（ratio） | 11.55（0.510） | 10.42（0.364） | −1.13 |
| mte3_time（ratio） | 10.27（0.457） | 7.22（0.254） | −3.05 |
| scalar_time | 5.46 | 5.54 | +0.08 |
| GM 读/写（KB/核） | 409.6 / 3.2 | 409.6 / 3.2 | 0 |
| vec 单价 | 0.0579 ns/元素 | **0.0919 ns/元素（1.59×）** | — |
| vec pass 数（按 §3 的 0.158 µs/pass 反解） | **3.0** | **4.77 ≈ 5** | **+2 pass** |

**定量结论（用户要求的两项拆分）**：

1. **`vcmp` 的 f32 lift = 差值的全部**。按 §3 标定的 0.158 µs/pass（8192 元素/任务的吞吐口径），
   bf16 的向量链是 **5 pass** vs fp16 **3 pass**（vcmp 前把 **两个** 16-bit 操作数各上抛一次
   f32，再在 f32 域比较），多出的 2 pass = 2 × 0.158 × 26 任务 = **8.2 µs**，与实测
   **+7.24 µs** 的 vec 增量吻合（差值被 mte2/mte3 的部分重叠吸收，净 +4.76 µs）。
2. **UB 单价 12.03 vs 8.00 B/元素在本 shape 上贡献 0 µs**。两点实测：
   ① 两个 dtype 的 `block_m` 都由 `E_CAP = 8192` 决定（2 行 × 4096），UB 单价不参与选择；
   ② 编译探针 `probe_bf16_ub.py`：bf16 `bm=4`（16384 元素）→ **196800 B = 12.012 B/元素**
   （超出 192 B），而 fp16 同点 round-1 实测 197152 B = 12.03 B/元素——**两个 dtype 的
   B/元素在这条 W=4096 的链上几乎相同**。所以"bf16 = 12 / fp16 = 8"是**容量口径**
   （`_ub_elems`：bf16 15701 vs fp16 24576 元素的驻留上限）的差异，不是 bf16 的时间成本；
   在本 shape 上它既不改变 `block_m`，也不改变时延。
   ⇒ bf16 的真实瓶颈是**向量链长度**（vec ratio 0.685 已超过 mte2 0.364），
   优化方向必须是「减少 pass / 减少每元素向量工作量」，而不是加带宽或换路径。

#### 9.5.2 候选（3 个，每个只改一个主要优化点；均从同一 current best 派生）

| 候选 | 文件 | 唯一主要优化点 | L0 | 结果 |
|---|---|---|---|---|
| **op1** | `_logical_reduce_kernel_round4_op1_bf16_lift.py` | bf16 只做**一次**显式 f32 lift（`vcast(x_ub→x32)` 一次 + `vcmp(x32, z32)`，z32 提到循环外），替代每 call 两次隐式 lift | PASS（21/21） | valid，−5.4%（交错），**未采纳** |
| **op2** | `_logical_reduce_kernel_round4_op2_bf16_bm3.py` | bf16 驻留档由 2 行提到 **3 行**（元素预算按实测 12.01 B/元素取 12288；683 个逻辑块 → 18 任务/核） | PASS（21/21，L0-2 期望值同步改为 bf16 bm=3） | valid，**−10.55%，采纳** |
| op3 | `_logical_reduce_kernel_round4_op3_bf16_f16view.py` | bf16 位模式按 fp16 视图比较（零位模式两 dtype 相同，绕开 f32 lift） | **blocked** | 见下 |

- **op3 blocked（如实记录，不入 `perf_records.jsonl`）**：`T.vbitcast(x_ub,"float16")` 返回
  `tir.Call` 句柄，`T.vcmp` 直接拒绝该操作数（`<class 'tvm.tir.expr.Call'> has no attribute
  buffer`）；改成语句形态（`T.vbitcast` 再 `T.vcmp(x_ub,…)`）虽然编译通过，但该 op 是
  **in-place 重解释**——后续 bf16 数值比较读到的是 fp16 解释后的位模式，
  `IMPL_CHECK: FAIL 1024/2048 rows differ（first at 1 got=4042 want=2888）`。
  ⇒ 位视图路线在当前前端 API 下不可用（能力缺口候选见 §10 提案，本轮不写文件）。
- 三个候选都未触碰 fp16 / fp32 路径（`_select_config` 与 builder 的 fp16 分支保持原样），
  因此 fp16 三个 tune workload 的生成代码逐字节不变（§7.1 的 `get_kernel_source()` 比对）。

#### 9.5.3 候选 vs current best（结构化对比表）

| 候选 vs current best | Task Duration(us)（profile 同 run） | AICore pipe（vec/mte2/mte3 ratio） | memory（GM 读/写 KB per core；读 BW GB/s/核） | L0 | 结论 |
|---|---:|---|---|---|---|
| current best（`final_rprime`，bm=2） | 30.52 / 30.68 | 19.58 (0.684) / 10.58 (0.369) / 7.75 (0.271) | 409.6 / 3.2；13.6 | PASS | — |
| op1（bf16 单次 f32 lift） | 28.92 | 17.79 (0.670) / 10.87 (0.409) / 8.43 (0.319) | 409.6 / 3.2；14.7 | PASS | valid，−5.2%（同 run） |
| **op2（bf16 驻留 3 行档）** | **27.76** | **17.41 (0.713) / 8.58 (0.351) / 4.42 (0.182)** | 409.6 / 3.2；**16.0** | PASS | **valid，−9.5%（同 run）→ winner** |
| op3（fp16 位视图） | — | — | — | — | blocked（见 9.5.2） |

- op2 的机制：任务数 26 → 18/核、单任务连续块 16 KB → 24 KB，`mte2` 10.58 → 8.58 µs
  （−18.9%，等效读带宽 13.6 → 16.0 GB/s/核）、`vec` 19.58 → 17.41（−11.1%，每任务更大
  连续块摊薄 vcmp/上抛的固定部分）、`mte3` 7.75 → 4.42、`scalar` 5.38 → 4.29。
- op1 的机制：vec 19.58 → 17.79（−9.2%，正好约 1 pass），说明隐式 lift 确实是 2 次；
  但 mte2/mte3 未改善，净收益只有 op2 的一半，故未采纳（且两者组合未做——候选上限 3）。

#### 9.5.4 交错 A/B/A/B 原始成对数据（同机同 run，`--launch-count=30 --warm-up=5`）

A = 测量当时的 accepted `perf_opt/_logical_reduce_kernel.py`（round-3 版，sha
`2a7552f6f7a85173be39be8cab76b370ab8eee7491d3129407668b2c0ca2ec2dd`；事后为做代码等同性
比对而重建的同内容文件 = `_logical_reduce_kernel_round4_prevfinal_recon.py`，sha **完全相同**）；
B = 候选。原始 profile：`profiles/ab_round4/{op1,op2}/{A,B}_run*/`，汇总
`profiles/ab_round4/{op1,op2}/summary.json`。

| 对被 | op1 A(µs) | op1 B(µs) | op1 逐对 Δ | op2 A(µs) | op2 B(µs) | op2 逐对 Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 31.253 | 29.516 | −5.56% | 31.235 | 27.741 | −11.19% |
| 2 | 31.185 | 29.549 | −5.24% | 31.317 | 28.107 | −10.25% |
| 3 | 31.287 | 29.299 | −6.36% | 31.168 | 27.940 | −10.36% |
| 4 | 31.186 | 29.538 | −5.28% | — | — | — |
| **中位** | **31.220** | **29.527** | **−5.42%** | **31.235** | **27.940** | **−10.55%** |
| **极差** | 0.103 | 0.251 | — | 0.149 | 0.366 | — |
| 方向一致性 | 4/4 同向 | | | 3/3 同向 | | |

**裁决**：

- **op2（采纳版）：B ≤ 29.99 µs 成立**——3 对全部落在 27.74–28.11 µs，中位 27.94，
  比验收线低 **6.8%**；且比 A 臂低 10.25–11.19%，**远超 run-state 离散**（同 run A 臂本身
  极差 0.149 µs = 0.5%，round-0/round-1 的 bf16 复采离散仅 0.7%），**不属于噪声**。
- **op1（未采纳）：B ≤ 29.99 µs 也成立**（中位 29.527，4/4 同向，−5.42%），但收益只有
  op2 的一半且 mte2 无改善；同轮候选上限为 3，按「同 workload 取 Task Duration 最优者」
  采纳 op2，op1 作为已实测的次优单点记录在案（其"减少 f32 lift"结论仍被验证：
  vec −9.2% ≈ 1 pass）。
- 结论对照用户给定的两种可能：**不是"噪声内不可判定"，也不是"结构性差 X%"**——
  本轮找到并采纳了可复现的 −10.55% 改进，bf16 达标（收）。

#### 9.5.5 采纳后的精度与全量复测

- 采纳后的交付文件 `perf_opt/_logical_reduce_kernel.py`（sha `faae175e…`）跑
  `--level all`：**L0 21/21 PASS**（含 L0-12 NaN 语义三 dtype、L0-10 构造行）、
  **L1 24/24 PASS**（含非连续、高阶 dim、尾块、极值、亚正规、caller block_m 钳位）、
  **L2 全 PASS**、**Boundary 全 PASS**，`torch.equal` 逐位（atol=rtol=0），
  日志 `logs/round4/L0L1_final_round4.log`；smoke 三 dtype（含 bf16 (32,256)）精度 PASS。
- **Op 层（wrapper）接线核对**：`CountNonzeroFwdOp(dtype=torch.bfloat16, dim=-1)` 在
  `(2048,4096)` 上打印 `LogicalReduceKernel initialized with config: {'block_m': 3}`
  （即 wrapper 的 `default_config → _select_config` 已自动取到新档位），输出与
  `torch.count_nonzero` **逐位相等**。工厂签名与 `_select_config` 导出均未变。
- 全量 tune workload 在同一最终候选上重采 `phase=final`（`run_final_round4.py`，
  30 发射均值）：hidden fp16 28.104、hidden bf16 **27.721**、seq fp16 4.111、
  3d fp16 9.648 µs；raw profile 在 `profiles/final/round4/final_round4_bf16bm3/<wl>/`。
- 残留机会（defer）：① op1 + op2 组合（单点各 −5.4% / −10.55%，是否叠加未测）；
  ② fp16 也试 3 行档（fp16 同点 UB 单价 12.03 B/元素、`bm=3` 同样编译通过），
  本轮不动 fp16（用户裁定收束）。

---

## 10. Round-4 沉淀提案（只给文本，不写回知识库）

> 用户裁定：本轮起 `.agents/skills/tilelang-op-optimize/references/pattern-library/` 与
> `.agents/evolution/capability-gaps.md` 对 optimizer **只读**；以下为待 evolver 合入的提案原文。

### 10.1 D 类（实测数据）提案：`CONST-reduce-ub-per-element` 追加第六数据点

```text
条目: CONST-reduce-ub-per-element（2-D 行归约 UB 单价按 (W,bm) 分档）
新增数据点（2026-10-18，910B4，tilelang 0.1.2+c316b0e / CANN 9.0.0）:
  W=4096 的 2-D 行归约指示链，fp16 与 bf16 在 16384 元素/块的单价几乎相同:
    bf16: 16384 元素 -> 196800 B = 12.012 B/元素（溢出 192 B 编译失败，探针 probe_bf16_ub.py）
    fp16: 16384 元素 -> 197152 B = 12.03 B/元素（round-1 实测）
  => "bf16 = 12.00 / fp16 = 8.00 B/元素" 只在 (W=23552,bm=1) 通过点成立，不可跨 (W,bm) 引用；
     在 W=4096 上两者同为 ~12 B/元素，因此该单价**不能**用来解释 bf16 的时延差。
  通过点: W=4096, bm=3（12288 元素）在 fp16/fp32/bf16 三 dtype 均编译通过。
  判别意义: UB 单价只决定容量档（_ub_elems），时间差要看 vec pass 数。
```

### 10.2 D 类：bf16 `vcmp` 隐式 f32 lift = 5 pass vs fp16 3 pass（+7.24 µs）

```text
条目建议: CONST-bf16-vcmp-f32-lift-two-passes（可挂 CONST-vector-launch-overhead 姊妹）
实测（(2048,4096)、bm=2、26 任务/核、GM 流量相同）:
  fp16 vec_time 12.34 µs = 3 pass；bf16 vec_time 19.58 µs = 4.8~5 pass
  => 隐式 lift 对**两个**操作数各做一次（2 个额外 pass ≈ 2×0.158×26 = 8.2 µs，实测 +7.24）
  验证性修复 1: 显式单次 lift（vcast(x->f32) 一次 + vcmp(f32,f32)）=> vec 19.58 -> 17.79（-1 pass）
  验证性修复 2: 驻留 3 行档 => vec 17.41 / mte2 8.58（任务数 26 -> 18）=> Task -10.55%（交错 A/B）
  repro: 本轮 _logical_reduce_kernel_round4_op1_bf16_lift.py（before/after = 接受版 vs op1）
  反例（不可用路线）: vbitcast 位视图（Call 句柄不能作 vcmp 操作数；语句形态 in-place
  重解释导致 bf16 数值比较读到 fp16 位模式，1024/2048 行错误）
```

### 10.3 P 类（模式）提案：bf16 驻留档按「实测 B/元素」上调一档

```text
模式: 当 2-D 行归约的瓶颈是 vec（bf16 上抛链）而非读带宽时，把 bf16 的驻留行数
      从 E_CAP/N 提到 floor(实测元素上限 / N) 常能拿到 5~11% 收益：
      任务数下降 => 每任务固定成本（DMA 建立、vcmp/上抛的固定段）被摊薄，
      等效读带宽 13.6 -> 16.0 GB/s/核。
判据: profile 上 vec_ratio > mte2_ratio（本任务 bf16 0.685 vs 0.364）时优先试；
      fp16 同点 vec/mte2 均衡（0.545/0.510）时收益预期低。
边界: 元素数必须 <= 编译通过点（本任务 12288 通过 / 16384 溢出）；先跑编译探针再上 profile。
代价: 无精度风险（只改行分块），L0/L1 逐位通过；对 M 尾块依赖既有 real_m 裁剪。
```

### 10.4 能力缺口提案（capability-gaps，只给文本）

```text
CG 候选: Frontend API —— 16-bit dtype 的「零判定」缺少无 f32 上抛的向量原语
  层: TileLangIR / Frontend（tilelang.language）
  现象: T.vcmp 在 bf16 操作数上隐式上抛 f32 且对两个操作数各做一次（2 pass / 元素）；
        绕法 T.vbitcast 返回 tir.Call 句柄，T.vcmp 拒绝作为操作数；
        语句形态为 in-place 重解释，会让后续同 buffer 的数值比较读到错误 dtype 位模式。
  影响: 所有 bf16 的「与常数比较」类逐元素/归约算子（count_nonzero / all / any /
        mask 类）都要多付 1~2 个向量 pass（本任务 +7.24 µs / +59% vec time）。
  期望: ① vcmp 支持标量常数操作数（免去 zero buffer 与其 lift）；
        ② 或提供「位视图」buffer 句柄（vbitcast 返回可作 vcmp 操作数的 buffer view）。
  量化收益: 本任务 bf16 hidden 预期再加 ~1 pass（-3.5~4 µs，约再 -13%）。
  occurrences: 1（本任务），open
```

### 10.5 R 类（流程）提案

```text
R 候选: optimize Phase 3「final 采集是单向门」需要补一条「用户裁定改验收线」的通道：
  本轮（post-DONE 补测）在 final 之后采纳了 bf16 改进，触发 final 重采；
  现行 skill 只写了 precision_fix/续跑通道，未写「验收口径变更 + 定向补测」这种
  最小重开形态（候选上限 3、只跑被点名的 workload、其余以 codegen 等同性证明收束）。
  建议: 明确「post-DONE 定向补测」为第三种重开形态，产出 = 只增补被点名 workload 的
  候选/final 行 + sidecar 保全被取代 final + opt_log supersede 说明（本轮 §7.1 即模板）。
```

---

## 11. Round-5 post-DONE 定向补测（2026-10-20）

本轮形态与 round-4 相同（post-DONE 定向补测，R 提案 §10.5 的第二种实例）：只重开被点名的
workload，其余以 codegen 等同性证明收束；候选上限 3，每分支一个主要改动。

### 11.1 环境与工具链戳（跨会话不可比）

| 项 | round-4（2026-10-18） | **round-5（2026-10-20，本轮）** |
|---|---|---|
| 器件 | 910B4（40 AIV） | **910B3（npu-smi: 910B3；profile `Block Dim=40`）** |
| tilelang | 0.1.2+`c316b0e` | **0.1.2+`84c6019e8e54f`（commit 06e1a1a）** |
| CANN | 9.0.0 | 9.0.0 |
| torch / torch_npu | 2.9.0 / 2.9.0.post8 | 2.9.0+cpu / 2.9.0.post8 |
| msprof 口径 | `--launch-count=30 --warm-up=5`，Task Duration 均值 | 同口径 |

**器件与工具链都变了**（npu-smi 报 910B3，tilelang commit 从 `c316b0e` 换到 `06e1a1a`）：
本轮所有绝对值只在**本轮会话内**可比；与 round-0/4 的差值混入器件 + run-state，不能当作提升量。
可复现的直接证据：同一份 fp16 baseline 代码（sha `a5fef64f…`）在 round-4 会话 final 报
28.104 µs，本轮 A 臂（4 次 30 发射）中位 **22.493 µs**，纯会话偏移 −20%。

stale 条目重验（§10.1 清单）：本轮未直接复用旧常数作决策依据——`CONST-reduce-ub-per-element`
的 8.00/12.00 B/元素与「bm=4 溢出」结论在**新工具链上已被本轮编译探针部分推翻**（见 11.3）。

### 11.2 (A) hidden fp16：同 run 交错 A/B/A/B 裁决（必做项）

A = 生产基线 `../_logical_reduce_kernel.py`（sha `a5fef64f…`），B = round-4 current best
`perf_opt/_logical_reduce_kernel.py`（sha `faae175e…`）。同一 msprof 会话内严格交错
A/B/A/B ×4 对，每臂 30 发射（warm-up 5），kernel-only `Task Duration(us)` 均值。
原始 profile：`profiles/ab_round5/fp16_hidden/{A,B}_run{1..4}/`；逐次日志
`logs/round5/ab_fp16_hidden_{A,B}{1..4}.log`；汇总 `.../summary.json`。

| 对 | A（baseline） | B（round-4 best） | 逐对 Δ |
|---|---:|---:|---:|
| 1 | 22.468 | 22.458 | −0.04% |
| 2 | 22.843 | 22.742 | −0.44% |
| 3 | 22.480 | 22.737 | +1.14% |
| 4 | 22.505 | 22.782 | +1.23% |
| **中位** | **22.493** | **22.740** | **+0.55%** |
| 极差 | 0.375 µs（1.67%） | 0.324 µs（1.42%） | — |

**裁决：run-state，非真实回退**（结论 a）。依据三条：
1. `|Δ中位| = 0.55%` 小于同 run 噪声带（臂内极差 1.42–1.67%，逐对 Δ 极差 1.67 个百分点）；
2. 方向不一致（4 对中 2 对 B 稍快、2 对 B 稍慢，`direction_consistent=false`），
   `sign-test` 无法拒绝零假设；
3. **生成源逐字节相同**：`codegen_diff_round5.py` 对 `(2048,4096) fp16` 得
   A = B = `1fbac5543b928d91…`（bm=2），同一工具链上不可能有真实回退。
   （辅助日志 `logs/round5/codegen_diff_fp16.log`。）

同时澄清上一轮的外部报告差异：最新 perf_opt report 27.66 µs vs baseline report 25.50 µs
的 +8.5% 跨 run 差异，与本轮观察到的会话级漂移（同码 22.49 vs 28.10）同性质，属 run-state。
**B 分支（dtype 分派回退）未触发，本轮未改动 fp16 比较/归约链。**

### 11.3 (C) defer 项落实：bm=4 —— 编译探针先把 UB 记账拆开

round-4 的 defer 方案是「x_ub 4 行 + compute 2 行二分复用」，前提是 4 行单组 UB 溢出
（旧工具链 12.03 B/元素 → 197152 B > 196608 B）。本轮先用
`probe_bm4_bisect.py`（编译探针，**不入 `perf_records.jsonl`**）在同一 W=4096、N=4096 上重测：

| 形态 | dtype | 行组 R | 结果 |
|---|---|---:|---|
| `full`（单组，x_ub 与 compute 同为 R 行） | fp16 | 4 | **COMPILE_OK**（旧工具链的 197152 B 溢出**不再复现**） |
| `full` | fp16 | 8 | FAIL `UB_REQUIRED 331936 B` = 10.13 B/元素 |
| `full` | bf16 | 4 | FAIL `UB_REQUIRED 196800 B` = 12.012 B/元素（与 round-4 完全一致） |
| `full` | bf16 | 6 | FAIL `UB_REQUIRED 295072 B` = 12.007 B/元素 |
| `A`（x_ub R 行 + compute 2 行复用） | fp16 | 4 | COMPILE_OK |
| `A`（同上） | bf16 | 4 | **FAIL codegen**：`VcastCodegen → getBroadcastDim` 形状检查（bf16 隐式 f32 lift 对 buffer region 的形状推断失败） |

**分开记账的结论**（本轮实测，替代旧单价外推）：
- fp16：4 行单组在 196608 B 内，8 行组 331936 B ⇒ 上限 ~19400 元素，**16384 元素（4 行）
  是阶梯内可用的最大档**；按两点线性反解的单元素价 ~10.13 B/元素（不再引用 12.03）。
- bf16：仍为 12.0 B/元素线性（R=4 与 R=6 两点斜率 12.00），4 行 = 196800 B，
  **超出 192 B**，无法直接上 4 行。
- 因此 round-4 设想的「二分复用」对 fp16 已无必要（plain 4 行组更省：向量调用次数还少一半），
  对 bf16 则被 codegen 形状检查阻塞。

**候选（op1，一个分支只改一处）**：`_logical_reduce_kernel_round5_op1_fp16_bm4.py`
= current best + `_FP16_RESIDENT_ELEMS = 16384`（fp16 驻留行档 2 → 4；float32 仍守
`E_CAP = 8192`，bf16 守 12288）。`_select_config` 仍是唯一 tiling 真源，返回值新增
`passes` 键仅在二分变体中用；本候选不启用二分，故对外契约不变。
`_logical_reduce_kernel_round5_op1_bm4_bisect.py`（二分变体）**未上 profile**——编译探针
已证明 fp16 plain 组可行且严格更省，bf16 被 codegen 阻塞，故该分支不再测量（如实记录，
不把它写成已测）。

**候选 vs current best（同 run 交错 A/B，hidden fp16，3 对，30 发射/臂）**

| 对 | A = current best（bm=2） | B = op1（bm=4） | 逐对 Δ |
|---|---:|---:|---:|
| 1 | 22.760 | 16.331 | −28.25% |
| 2 | 22.674 | 16.343 | −27.94% |
| 3 | 22.557 | 16.335 | −27.58% |
| **中位** | **22.674** | **16.335** | **−27.92%** |

方向一致、幅度远超噪声（臂内极差 0.9%）⇒ **采纳**。原始 profile
`profiles/ab_round5/fp16_bm4_vs_best/{A,B}_run{1..3}/`，汇总 `.../summary.json`。

**结构化对比表（B2：候选 vs current best，同 run 第 1 对 profile 逐项）**

| 指标（per block 均值） | current best（bm=2） | candidate（bm=4） | Δ |
|---|---:|---:|---:|
| Task Duration(us) | 22.580 | 16.080 | **−28.8%** |
| aiv_time(us) | 20.588 | 14.171 | −31.2% |
| vec_time(us) / ratio | 11.310 / 0.550 | 9.476 / 0.670 | −16.2% / +0.12 |
| mte2_time(us) / ratio | 10.358 / 0.503 | 6.115 / 0.432 | −41.0% / −0.07 |
| mte2 active bw(GB/s) | 37.74 | 64.00 | **+69.5%** |
| mte3_time(us) / ratio | 6.565 / 0.321 | 2.672 / 0.190 | −59.3% |
| scalar_time(us) | 4.981 | 3.449 | −30.8% |
| GM 读(KB/块) / 写(KB/块) | 409.725 / 1.625 | 409.725 / 1.625 | 0（流量不变） |
| 逻辑块 / 任务每核 | 1024 / 26 | 512 / 13 | −50% |
| L0 | PASS | **PASS（21/21）** | — |

机制归因：流量逐字节不变（409.725 KB/块），收益全部来自**发射与搬运的组织**——
每个任务一次 32 KB 连续 GM→UB 搬运（mte2 active bw +69.5%）、任务数减半
（scalar/mte3 固定开销腰斩），向量总工作量不变（vec_time 只降 16%，因为元素数相同）。
有效读带宽 16.79 MB / 16.28 µs = **1.03 TB/s**，高于 round-4 记的「器件 GM 屋顶 0.8 TB/s」
——该屋顶数值同属旧会话口径，本轮不再用它做门禁。

### 11.4 (B) 回退分支

未触发（A 判 run-state），本轮无 dtype 分派回退改动。bf16 继续用 round-4 的 3 行档
（生成源 sha `76bb0c5a…` 与 round-4 逐字节相同）。

### 11.5 (D) 逐 workload 复测与 current best 更新

**先证明「无回退」**：op1 只改 fp16 的驻留行档，其余三条 tune workload 的生成源用
`get_kernel_source()` 逐一比对（`logs/round5/codegen_diff_round5_unaffected.log`）：

| workload | round-4 best sha | round-5 候选 sha | 结论 |
|---|---|---|---|
| hidden bf16 (2048,4096) | `76bb0c5a33056baf`（bm=3） | `76bb0c5a33056baf`（bm=3） | 逐字节相同 |
| seq fp16 (32,16384) | `5eab5ac80659677d`（bm=1, R′） | `5eab5ac80659677d` | 逐字节相同 |
| 3d fp16 (128,16384) | `2465d2939d33d4c9`（bm=1, R′） | `2465d2939d33d4c9` | 逐字节相同 |
| hidden fp16 (2048,4096) | `1fbac5543b928d91`（bm=2） | `9c04dace7fc7a8d8`（bm=4） | 预期唯一改动 |

⇒ 三例「对 A 臂同 run 不劣于噪声带」由 codegen 等同性成立（同码不可能回退）；
唯一变化例 hidden fp16 由 11.3 的同 run 交错 A/B（−27.92%，方向一致）成立。
**全量无回退成立 ⇒ 更新 current best。**

**final 复测（同一最终文件 sha `c075e1a1…`，每个 tune workload 恰一条，30 发射均值）**

| kernel_id | workload_id | baseline_us（round-0，跨会话） | final_us（round-5） | round-4 final | final_candidate_id |
|---|---|---:|---:|---:|---|
| `_logical_reduce_kernel::main` | sparsity-hidden-float16 | 26.014 | **16.280** | 28.104 | final_round5_fp16_bm4 |
| `_logical_reduce_kernel::main` | sparsity-hidden-bfloat16 | 31.135 | **23.913** | 27.721 | final_round5_fp16_bm4 |
| `_logical_reduce_kernel::main` | sparsity-seq-float16 | 4.408 | **3.849** | 4.111 | final_round5_fp16_bm4 |
| `_logical_reduce_kernel::main` | 3d-multidim-reduce-float16 | 10.234 | **7.651** | 9.648 | final_round5_fp16_bm4 |

- **唯一可归因于本轮代码改动的提升 = hidden fp16 −27.92%（同 run 交错）**；表中
  baseline→final 与 round-4→final 的差值混入 910B3 + 新工具链的会话偏移
  （同码 fp16 baseline 自身就从 28.10 漂到 22.49），不得计为提升。
- final 逐项 profile：`profiles/round5/final/{workload}/`，pipe 汇总
  `logs/round5/pipe_summary.jsonl`。

**final 复测的 pipe / memory 明细（每块均值，来自 profile 第 10 次发射）**

| workload | Task(us) | aiv(us) | vec(us)/ratio | mte2(us)/ratio | mte3(us)/ratio | mte2 bw(GB/s) | 读(KB/块) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hidden fp16 | 16.12 | 14.10 | 9.48 / 0.673 | 6.18 / 0.439 | 2.01 / 0.143 | 63.34 | 409.7 |
| hidden bf16 | 24.06 | 22.16 | 15.96 / 0.721 | 7.53 / 0.340 | 3.55 / 0.161 | 51.93 | 409.7 |
| seq fp16 | 3.92 | 2.46 | 0.83 / 0.348 | 0.53 / 0.225 | 0.33 / 0.133 | 59.64 | 32.1 |
| 3d fp16 | 7.64 | 5.48 | 2.47 / 0.454 | 1.70 / 0.314 | 1.35 / 0.245 | 58.11 | 102.5 |

（表内 Task 为第 10 次发射的单次值，故与 final 均值 16.280/23.913/3.849/7.651 略有出入。）

**被取代 final 的处置**：round-4 的 4 条 final 行原样搬入
`perf_records_full_history.jsonl`（24 → **28** 行，逐字节未改）；主
`perf_records.jsonl` 由 24 → **26** 行（移除 4 条旧 final，新增 round-5 的
baseline ×1、candidate ×1、final ×4）。round-5 新增行：
`baseline` hidden fp16 = 22.4926（A 臂 4 次交错中位）、`r5_op1_fp16_bm4` hidden fp16
= 16.3353（B 臂 3 次交错中位，`parent_id=final_round4_bf16bm3`）、4 条 final。

**smoke（3 例，仅精度、不采性能）**：`smoke-count-nonzero-{float16,bfloat16,float32}`
（32,256）经 `--level all` 的 L0-1 三 dtype 覆盖，全部 PASS，`precision_pass=true`
（未写入任何性能行）。

### 11.6 精度门禁（硬，最终文件 `c075e1a1…`）

`python perf_opt/_logical_reduce_kernel.py --level all` → exit 0（日志
`logs/round5/L0_final.log`；候选门禁 `logs/round5/L0_op1_fp16_bm4.log`）：

- **L0：21/21 PASS**（含 L0-2 fp16 断言改为 `block_m=4`、L0-7/L0-8b 期望随新档位更新）；
- **L1：24/24 PASS**（非连续输入、movedim、bool/int32/int64/complex 预转、M 尾块、
  路径 C 静态尾块 tail=424/3616、R′ 边界 23552 元素等）；
- **L2：全部 PASS**；**Boundary：全部 PASS**（含 `(512,2048)` 走新 8 行档）。
- **语义未变**：L0-12 三 dtype 的 NaN/-0 用例全 PASS（NaN 计入非零、-0 不计入），
  比较仍为 `vcmp("eq")` + 补式，**未改成 `"ne"`**；`torch.equal` 逐位（atol=rtol=0）；
  int64 计数精确。

### 11.7 Skill Retrospective（round-5）

**流程有效点**

1. **「先同 run 交错 A/B，再谈回退」把一次潜在的错误回退挡在门外**：跨 run 报告差 +8.5%
   看起来像回退，同 run 4 对交错得 +0.55%（方向不一致）+ codegen 逐字节相同 ⇒ run-state。
   若按跨 run 数字行动，会白改一次 fp16 分派并冒精度风险。
2. **编译探针（不占 profile 预算）先行，直接改写了方案**：round-4 的 defer 方案是
   「bm=4 + 二分复用」，探针发现新工具链上 plain 4 行组已可编译、bf16 二分被 codegen 阻塞
   ⇒ 用更简单且严格更省的结构替代，避免了在二分复用上写一整套 buffer 复用代码。
3. **codegen 等同性证明替代重复 profile**：三条未改 workload 用 sha 比对收束
   （0 次 profile），把预算全部留给唯一变化的 hidden fp16 与 4 条 final（本轮共 18 次 profile）。
4. **exhaustive 的 UB 两点记账**（fp16 R=8 / bf16 R=4,R=6）给出单元素价与上限，替代旧常数外推。

**流程问题 / 建议**

1. **`CONST-reduce-ub-per-element` 这类「容量常数」绑定工具链，跨 commit 会自动失效**：
   本轮 tilelang 从 `c316b0e` 换到 `06e1a1a`，同一个 4 行组从「溢出 197152 B」变成
   「可编译」，使 round-4 的整条 defer 方案（二分复用）失去前提。建议 skill 在 Phase 0
   的 E-5 核对里，把「UB 容量/单价」类结论列为**必须重测**（一条编译探针即可），
   而不是只标注「待重验」。
2. **【重要】跨会话绝对时延不可作为验收依据，应写进 skill 的验收节**：本轮同码 fp16
   baseline 在两台器件/两次工具链之间差 20%（28.10 → 22.49），比任何真实优化幅度都大。
   建议 skill 明确：「跨 run 报出的 baseline→final 差值只作参考；提升量一律取同 run
   交错 A/B 的中位逐对 Δ」，并要求 opt_log 的 Final Summary 表标注会话戳。
3. **器件型号漂移未被任何工件检查**：npu-smi 本轮回报 910B3，round-4 记录为 910B4，
   而 `run_bench.py` 只记 impl sha。建议（工具改进）在 `measure()` 结果里带
   `npu-smi` 器件名 + tilelang commit，写进 raw profile 侧车，避免事后靠人工回忆。
4. **`summarize_profile.py` 的 pipe 字段依赖 `Op Name` 列，本轮 AIC 指标 CSV 改为
   按 `block_id` 行组织**，导致 `run_bench.py` 的 `pipe` 字段全空（本轮改用
   `summarize_profile.py` 直接聚合块行）。建议（工具改进）`_parse_pipe` 兼容无 `Op Name`
   的块级 CSV。

**Value Point Proposals（供 evolver 蒸馏）**

| ID | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-F | D | fp16 行归约驻留上限在新工具链上是 16384 元素（4 行 × 4096）：4 行组编译通过、8 行组 331936 B 拒绝（10.13 B/元素）；bf16 仍 12.0 B/元素（R=4 → 196800 B） | `logs/round5/probe_bm4_bisect.log`；`perf_records.jsonl` round 5 | `probe_bm4_bisect.py`（编译探针，含 form full/A/B/C） | tilelang 0.1.2+84c6019 / CANN 9.0.0 / 910B3 |
| VP-G | D | fp16 hidden (2048,4096) 驻留档 2→4 行：22.674 → 16.335 µs（−27.92%，交错 3 对，方向一致）；mte2 active bw 37.7→64.0 GB/s，任务数 26→13/核，流量不变 | `profiles/ab_round5/fp16_bm4_vs_best/`；`perf_records.jsonl` round 5 candidate 行 | `ab_round5.py` + `perf_records` 行 | 同上 |
| VP-H | P | 行归约「驻留行档」上调前必须先做**UB 编译探针**（当前 commit 下最大可编译行组），旧单价不可跨 commit/跨 (W,bm) 外推；探针通过后再上 profile | §11.3 表 + round-4 defer 方案失效 | `probe_bm4_bisect.py` | 同上 |
| VP-I | P | bf16 的 buffer-region 二分复用被 `VcastCodegen/getBroadcastDim` 形状检查阻塞（隐式 f32 lift 只支持整 buffer 操作数） | §11.3 探针 row 6 | `probe_bm4_bisect.py --form A --dtype bfloat16` | 同上 |
| VP-J | R | 「提升量」必须取同 run 交错 A/B 中位逐对 Δ；跨 run/跨器件绝对时延差值不可作验收（本轮同码漂移 20% > 任何真实优化幅度） | §11.1 + §11.2 | `ab_round5.py` | 无工具链依赖 |
| VP-K | R | Phase 0 的 E-5 版本核对应把「UB 容量/单价」类常数列为必须用编译探针重测（1 条探针即可，不占 profile 预算） | §11.3 | `probe_bm4_bisect.py` | 无工具链依赖 |

**本轮结论**：adopted（hidden fp16 4 行档），不采纳/阻塞（bf16 二分复用 = codegen 阻塞；
bf16 4 行 = UB 超出 192 B；fp16 二分 = 严格更劣，未测）。current best 更新为
`final_round5_fp16_bm4`（sha `c075e1a1…`），全量无回退，L0/L1 全 PASS。
