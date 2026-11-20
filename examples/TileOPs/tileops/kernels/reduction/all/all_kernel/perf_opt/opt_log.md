# `_logical_reduce_kernel`（AllFwdOp / `torch.all`）Stage 4 调优日志

> **口径**：全文时延 = **`msprof op` `Task Duration(us)`**（kernel-only；
> `--launch-count=30 --warm-up=5`，同一 profile 目录内全部 `OpBasicInfo_*.csv`
> 逐次取均值）。不采集 NPU event、不用 runner 端到端时间。
> **字节口径**：manifest 的 `dtype=bool` 在 Op 层 `to_logical_float32` 后已变成
> f32，`msprof op` 只计 kernel，故所有字节/带宽/验收线一律按 **kernel f32 字节**
> 计算（4 B/元素 + 1 B/元素 int8 输出）。manifest/report 的 `bytes` 列按 bool
> 1 B/元素 记，等效能带被放大 4×，本日志只作台账记录、**不作判据**。
> Ratio 一律标注分母：**`/1.75 TB/s`**（台账口径）与 **`/0.8 TB/s`**（本机 msprof
> 自报屋顶，见 §6.1）。

## 0. 环境与工件

| 项 | 值 |
|---|---|
| 日期 | 2026-10-18 |
| tilelang | 0.1.2 + `4d2e41313438e8b79c25d3fe8020f26078400106`（dev-root build `/workspace/tilelang-mlir-ascend/build`，`.so` 构建于 2026-10-13） |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post8 |
| CANN / npu-smi | 9.0.0 / 26.1.1 |
| 设备 | `npu-smi info` → **910B4**（HBM 32768 MB）；`AscendArch()` → `chip_name=Ascend910B`、`aivector_core_num=40`、`ub_cap=196608 B`；`torch.npu.get_device_name(0)` = `Ascend910B4` |
| 自报 GM roof | **0.8 TB/s**（`roofline_from_profile.py` 读 profile 的 `visualize_data.bin` → `GM/L2: GM Read + Write`，baseline 4 个 profile 全一致；3d baseline 的 msprof `ratio` = 73.23%） |
| benchmark 入口 | `benchmarks/ops/bench_all.py::test_all_bench`（manifest 驱动） |
| 展开方式 | `pytest benchmarks/ops/bench_all.py --collect-only` → 6 例（见 §0.1） |
| 基准 kernel（只读） | `../_logical_reduce_kernel.py`，sha256 `1f7d1a180709d04990edfd79e6a7e05b37244fdf9b84b265f461f9fb7ae69e53` |
| 交付 kernel | `perf_opt/_logical_reduce_kernel.py`（sha256 见 §7） |
| 结构化记录 | `perf_records.jsonl`（append-only；baseline 3 + candidate 15 + baseline_recheck 3 + final 3） |
| 标定记录（**不混入** perf_records） | `copy_floor_calibration.jsonl`、`op_cost_calibration.jsonl` |
| raw profile | `profiles/{baseline,candidate,final,probes,ab}/…` |
| 过程日志 | `logs/{setup,round0,round1,round3,round4,probes,ab}/…` |
| 工具/脚本 | `bench_driver.py`、`run_bench.py`、`probe_floor.py`、`run_probes.py`、`run_experiments.py`、`ab_test_impl.py`、`run_final.py`、`summarize_all.py`、`codegen_compare.py`、`make_*.py` |

### 0.1 实跑 workload 清单核对（必查项）

`pytest --collect-only` 实跑 6 例（不是照抄设计文档）：

| workload_id | shape | dtype（manifest） | dim | kernel 视角（M,N） | kernel dtype | kernel 字节 | 分类 |
|---|---|---|---|---|---|---|---|
| smoke-all-bool | [32,256] | bool | -1 | (32,256) | float32 | 32,896 | smoke |
| smoke-all-float16 | [32,256] | fp16 | -1 | (32,256) | float16 | 16,416 | smoke |
| smoke-all-float32 | [32,256] | fp32 | -1 | (32,256) | float32 | 32,896 | smoke |
| mask-validation-4k-bool | [32,4096] | bool | -1 | (32,4096) | float32 | 524,320 | tune |
| mask-validation-32k-bool | [32,32768] | bool | -1 | (32,32768) | float32 | 4,194,336 | tune |
| 3d-multidim-reduce-bool | [4,128,4096] | bool | [0,2] | (128,16384) | float32 | 8,388,736 | tune |

- 3d 经 Op 层 `movedim(keep=[1]) → contiguous → reshape(128,16384)`；kernel 视角
  M=128、N=16384，与 §4 的 8,388,736 B 自洽。
- smoke 三例只做精度回归（label 含独立 token `smoke`），**不进性能闭环、不写
  perf_records 性能行**。
- 清单与 `workload_inventory.json` 一一对应（`kind=tune|smoke`，skipped 为空）。

### 0.2 测量装置

`bench_driver.py --impl <file> --case <workload_id>` 用 `importlib` 按**文件路径**
加载被测实现，走冻结的工厂契约
`_logical_reduce_kernel(M, N, "all", dtype)(cfg["block_m"])(x)`，输入按 Op 层语义
先做 `to_logical_float32` 与 `_prepare_2d`，再做 `--launches` 次发射 + 同步。
`run_bench.py` 在 `flock /tmp/tilelang-msprof.lock` 下串行调用
`msprof op --kernel-name=main --launch-count=30 --warm-up=5 --dump=off`，解析
`OpBasicInfo*.csv` 的 `Task Duration(us)`（mean/median/min/max）与
`PipeUtilization*.csv` / `Memory_*.csv` 的逐 block 均值。`kernel_id` 取
`prim_func.name`（= `main`）；`msprof_kernel_name`（`all_resident` 等）只给 Op 层
过滤用，**不作为 kernel_id**。

---

## 1. Performance Test Data（baseline，`phase=baseline`, round 0）

| kernel_id | workload_id | candidate_id | Task Duration(us) | median | min | profile_status | raw_profile_dir |
|---|---|---|---:|---:|---:|---|---|
| `_logical_reduce_kernel::main` | mask-validation-4k-bool | baseline | 2.9793 | 2.940 | 2.820 | valid | `profiles/baseline/round0/baseline_8ba8496a/mask-validation-4k-bool_0d778a4a` |
| `_logical_reduce_kernel::main` | mask-validation-32k-bool | baseline | 7.7587 | 7.720 | 7.480 | valid | `profiles/baseline/round0/baseline_8ba8496a/mask-validation-32k-bool_2cb75d7c` |
| `_logical_reduce_kernel::main` | 3d-multidim-reduce-bool | baseline | 14.3373 | 14.320 | 13.900 | valid | `profiles/baseline/round0/baseline_8ba8496a/3d-multidim-reduce-bool_567835aa` |

逐 workload 的 pipe / 访存画像（同 profile，逐 block 均值）：

| workload | blocks | aiv_time(us) | vec ratio | mte2 ratio | mte3 ratio | mte2 有效带宽(GB/s/核) | GM→UB(KB/核) | 实测达成带宽（按 kernel f32 字节 / Task） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 16 | 1.876 | 0.375 | 0.275 | 0.119 | 61.3 | 32.0 | 176 GB/s |
| 32k | 32 | 6.256 | 0.368 | 0.443 | 0.049 | 44.5 | 128.0 | 541 GB/s |
| 3d | 40 | 11.129 | 0.335 | 0.552 | 0.084 | 31.9 | 204.8 | 585 GB/s |

### 1.1 基线重复性与 run-state 双态

同一 baseline 在三个独立时刻各采一次（均为 30 发射均值）：round 0、round 4 的
`baseline_recheck`（`perf_records` 行）、以及 A/B session 1 的 A 臂中位（§4.4 原始数据）：

| workload | baseline（round 0） | baseline_recheck（round 4，`perf_records` 行） | 离散 | A/B session 1 的 A 臂中位 | 离散 |
|---|---:|---:|---:|---:|---:|
| 4k | 2.9793 | 3.0387 | +2.0% | 2.9587 | −0.7% |
| 32k | 7.7587 | 8.1473 | **+5.0%** | 8.1220 | **+4.7%** |
| 3d | 14.3373 | 14.5373 | +1.4% | 15.5067 | **+8.1%** |

⇒ 本机**同码跨 session 漂移 5–8%**（3d 尤甚：session 1 的 A 臂 15.51 vs round 0 的 14.34，单次 max 达 18.6 µs 的双态尾）。
因此**任何采纳决策都不看单 run 绝对值**，只认同机同 run 交错 A/B/A/B（§4.4）。
`perf_records.jsonl` 中 baseline 行（round 0）保持原值不改写；`baseline_recheck` 是
round 4 的独立补充行，两者并列披露。

### 1.2 设计估算 vs 实测偏差（D-2 回填，DESIGN §1.6.0 roofline）

| workload | DESIGN 估算下界 | 实测 baseline | 偏差 | 失准项 |
|---|---:|---:|---|---|
| 4k | 5–7 µs | 2.979 | **实测优 1.7–2.3×** | 发射项按 0.25–0.5 µs/op 串行上界叠加，偏悲观 |
| 32k | 12–24 µs | 7.759 | **实测优 1.5–3.1×** | 同上；实测逐 pass 0.11–0.13 µs/pass，低于 `CONST-vector-launch-overhead` 低档（0.23–0.3） |
| 3d | 19–38 µs | 14.337 | **实测优 1.3–2.6×** | 同上；MTE2 与向量存在部分重叠（非全串行） |

- 偏差方向与姊妹单元一致（count_nonzero 同样实测优于设计估算 1.4–3.5×）：
  设计期把发射项当串行上界，是保守估计而非失准。
- **流量项口径**：本机 MTE2 实测随 tile 大小 43–171 GB/s/核（§3），40 核聚合
  0.5–1.4 TB/s（含 L2 命中，见 §6.2），设计期取的 30.2 GB/s/核（`CONST-mte2-degradation`）
  落在本机带宽带内 → 流量项假设成立。

---

## 2. 调优顺序与理由

| 顺序 | workload | 实测负担 | 理由 |
|---|---|---:|---|
| 1 | 3d-multidim-reduce-bool | 14.34 µs | 绝对时延最大、流量最大（8.39 MB）、走列分块 2 tile × 4 任务/核；DESIGN §1.6.4 备选 R′ 的触发带宽恰在此（N=16384 f32） |
| 2 | mask-validation-32k-bool | 7.76 µs | 次大（4.19 MB），32 块 / 4 tile，mte2 ratio 最高（0.443），且距 560 GB/s 验收线只差 3.5% → 最可能靠结构改动达标 |
| 3 | mask-validation-4k-bool | 2.98 µs | 绝对时延已贴近空 kernel 地板（16 块 1.845 µs / 32 块 2.511 µs），先做地板标定再决定投入 |

排序只影响先后，未跳过任何 tune workload；smoke 三例不进入性能闭环。

---

## 3. 结构化标定（非 tune，独立 jsonl）

`probe_floor.py` 用与生产同构的 persistent 网格 + 同 block 尺寸做四种探针
（`empty` / `copyout` / `read` / `copy`，另加 `read_f16`），`--launch-count=15
--warm-up=3`，tile 形状合成（M=32，W = 实际 tile 宽度）：

| W × bm | tile（f32） | blocks | empty | copyout | read（f32 平面） | copy | read_f16（f32→f16 + f16 归约） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4096×1 | 16 KB | 32 | 2.511 | 2.640 | 3.191 | 2.683 | — |
| 4096×2 | 32 KB | 16 | 1.845 | 2.089 | 2.684 | 2.145 | 2.520 |
| 4096×4 | 64 KB | 8 | 1.455 | 2.284 | 2.876 | 2.284 | — |
| 8192×1 | 32 KB | 32 | 2.675 | 2.867 | 3.865 | 2.903 | 3.720 |
| 16384×1 | 64 KB | 32 | 2.649 | 4.157 | 5.156 | 4.136 | 5.040 |

反解单价（写入 `op_cost_calibration.jsonl`，逐条带 empty/copyout/read 原值）：

- **空 kernel 地板随 block 数增长**：8 块 1.455 µs、16 块 1.845 µs、32 块
  2.51–2.68 µs。→ 任何"多开核"策略都要先付这笔地板（§4.1 OP3 的证伪依据）。
- **GM→UB 读（MTE2）**：16 KB tile 0.129 µs（124 GB/s/核）、32 KB@bm2 0.244
  （131 GB/s/核）、64 KB@bm4 0.829（79 GB/s/核）、32 KB@W8192 0.192（171 GB/s/核）、
  64 KB@W16384 **1.508**（43 GB/s/核）。⇒ 读单价随 **W 变宽而变差**（W=16384 只有
  W=8192 的 1/4 有效带宽），这是"加宽 tile 反而更慢"的第一条机制证据。
- **f32 归约 pass（`reduce_max` 单平面）**：16 KB 0.551、32 KB@bm2 0.595、
  64 KB@bm4 0.592、32 KB@W8192 **0.999**、64 KB@W16384 **0.999** µs
  ⇒ 归约单价由**行宽 N** 而非元素总数决定（N=4096 约 0.55–0.60，N≥8192 约 1.0）。
- **f16 归约（含一次 f32→f16 全宽 cast）**：相比 f32 归约只快 0.10–0.16 µs
  （−2%…−6%）⇒ 指示平面改 f16 的收益**不是**来自归约本身，而是来自少一次全宽
  cast + 2 B/元素的 UB 余量（§4.2 的实测 −10% 主要来自后者带来的调度空间）。
- **UB→GM copy 增量 ≈ 0**（copy ≈ copyout，−0.02…+0.04）⇒ 输出写（128 B/次）
  完全被 MTE2/向量盖住，不是瓶颈。

> 探针不是 tune workload，只写 `copy_floor_calibration.jsonl` /
> `op_cost_calibration.jsonl`，**不进入** `perf_records.jsonl`。

---

## 4. Iteration Log

预算：≤4 轮 / ≤16 候选。实际使用 **2 个候选批次（round 1、round 3，共 8 个候选文件）**
- final 采集与一次 supersede 重采（round 4 → round 5）；其中 2 个候选在 L0 编译期即被
UB 上限判死（未产生性能行）。预算占用 8/16 候选、2/4 轮。

### 4.0 轮次 0：库/工具链前置核对

- `kb_stale_check.py` 报告 `stale_count=288`，但版本戳解析失败
  （`current_stamp` 无法解析 CANN 版本），因此 288 条大多属"戳不可解析"而非真实
  工具链变更；本任务只把**本 run 实测**作为决策证据，知识库常数仅作对照（§9.4）。
- 姊妹单元（count_nonzero, 910B4 同日）的 R′ 结论**不直接引用**：其 3d 是 **fp16**
  （16384 元素 × 8 B/elem = 128 KB 可驻留），本算子 3d 是 **bool→f32**
  （16384 元素 × 12.14 B/elem = 198,848 B > 196,608 B）——这一点直接决定了
  §4.1 OP1 的结局。

### 4.1 轮次 1：三个结构候选（同一 base = Stage 3 baseline）

| 候选 | 唯一主要优化点 | 文件 |
|---|---|---|
| OP1 | R′：`E_CAP < N ≤ UB 元素上限` 时 `bm=1` 单 tile 驻留（DESIGN §1.6.4，A/B 必做项） | `_logical_reduce_kernel_round1_op1_rprime.py` |
| OP2 | 列 tile 宽度上限由 `E_CAP=8192` 提到 per-dtype UB 元素上限（32k：4×8192 → 2×16384） | `_logical_reduce_kernel_round1_op2_tilecap_ub.py` |
| OP3 | 行块策略改为"填满 40 核"（`bm=1`，使 32 行用 32 核而非 16 核） | `_logical_reduce_kernel_round1_op3_bm_cores.py` |

**OP1/OP2 均在 L0 阶段 blocked（UB overflow，未产生性能行）**：
`loc(kernel.npuir): error: ub overflow, requires 1590784 bits while 1572864 bits available`
= 198,848 B > 196,608 B，即 **12.14 B/元素**（不是设计期外推的 8 B/元素）。
触发形态 = 16384 元素 f32 的单 tile。⇒ DESIGN §1.6.4 的"fp32 16384 元素可编译"
假设在本工具链（4d2e413）上**不成立**；blocked 原因记录在
`logs/round1/L0_*.log`。

| 候选 vs current best | workload | Task Duration(us) | AICore pipe（vec/mte2/mte3） | memory（GM→UB KB/核） | L0 | 结论 |
|---|---|---:|---|---|---|---|
| current best（baseline） | 4k | 2.979 | 0.375/0.275/0.119 | 32.0 | pass | — |
| OP1（R′） | 3d | — | — | — | **fail（UB overflow 198,848 B）** | **blocked** |
| OP2（UB tile cap） | 32k/3d | — | — | — | **fail（同上）** | **blocked** |
| OP3（核填满 bm=1） | 4k | **3.531（−18.5%）** | 0.222/0.234/0.261 | 16.0 | pass | **rollback** |
| OP3 | 32k | **8.153（−5.1%）** | 0.350/0.473/0.071 | 128.0 | pass | rollback（32k 不受该策略影响，属 run-state） |
| OP3 | 3d | 14.359（−0.2%） | 0.343/0.543/0.064 | 204.8 | pass | tie |

- **OP3 机制（falsified）**：4k 由 16 块变 32 块，mte3 ratio 0.119→0.261、Task
  +0.55 µs；与 §3 的 empty 地板（16 块 1.845 → 32 块 2.511）完全一致——**多开核
  先把地板抬起来**，读带宽的那点收益被吃掉。⇒ "填满核"在本算子族（小 M、读主导）
  不是有效方向，记为 `family_no_gain`。

### 4.2 轮次 2/3：指示平面 float16（同 base = baseline）

**动机（来自轮次 1 的 blocked 证据 + §3 标定）**：f32 指示平面在 16384 元素档要
12.14 B/元素；把指示平面（及由它派生的 `(bm,)` 微缓冲）改成 **float16** 可以
① 省 2 B/元素 UB；② 少一次全宽 `f32→f16` 收尾 cast；③ 归约输入字节减半。
**等价性**：指示量恒为精确 0.0/1.0，`reduce_max` 与 `1 − max` 在 f16 域精确，
`f16→i8` 有文档（`T.vcast` 表 `f16|i8|rint`）——L0/L1 用 `torch.equal` 逐位验证。

| 候选 | 唯一主要优化点 | 文件 |
|---|---|---|
| OP6 | 指示平面 f16（分派不变） | `_logical_reduce_kernel_round3_op6_indf16.py` |
| OP4 | OP6 + R′（3d 单 tile 16384） | `_logical_reduce_kernel_round3_op4_rprime.py` |
| OP5 | OP6 + UB tile cap（32k 2×16384；3d 单 tile 走 path C） | `_logical_reduce_kernel_round3_op5_tilecap.py` |
| OP7 | OP6 + R′ + UB tile cap | `_logical_reduce_kernel_round3_op7_rprime_tilecap.py` |

| 候选 vs current best | workload | Task Duration(us) | 相对 baseline | AICore pipe（vec/mte2/mte3） | memory（GM→UB KB/核） | L0 | 结论 |
|---|---|---:|---:|---|---|---|---|
| baseline | 4k / 32k / 3d | 2.979 / 7.759 / 14.337 | — | 见 §1 | 32 / 128 / 204.8 | pass | — |
| **OP6（f16 指示平面）** | 4k | **2.860** | **+4.0%** | 0.285/0.261/0.109 | 32.0 | pass | 进 A/B（后判定采纳） |
| OP6 | 32k | 7.845 | −1.1% | 0.288/0.563/0.064 | 128.0 | pass | tie（<3% 噪声阈） |
| OP6 | 3d | 13.967 | +2.6% | 0.256/0.620/0.063 | 204.8 | pass | 进 A/B（后判定冲突） |
| OP4（+R′） | 4k | 2.817 | +5.4% | 0.333/0.324/0.092 | 32.0 | pass | 与 OP6 同码（噪声） |
| OP4 | 32k | 7.901 | −1.8% | 0.272/0.566/0.029 | 128.0 | pass | 同码噪声 |
| OP4 | 3d | **15.715** | **−9.6%** | 0.248/**0.671**/0.049 | 204.8 | pass | **rollback** |
| OP5（+UB tile cap） | 4k | 2.810 | +5.7% | 0.332/0.297/0.104 | 32.0 | pass | 同码噪声 |
| OP5 | 32k | 7.957 | −2.6% | 0.284/0.566/0.032 | 128.0 | pass | rollback |
| OP5 | 3d | 15.323 | −6.9% | 0.244/0.657/0.059 | 204.8 | pass | rollback |
| OP7（+R′+cap） | 4k | 2.905 | +2.5% | 0.313/0.293/0.138 | 32.0 | pass | 同码噪声 |
| OP7 | 32k | 8.111 | −4.5% | 0.279/0.589/0.043 | 128.0 | pass | rollback |
| OP7 | 3d | 16.007 | **−11.6%** | 0.253/0.661/0.051 | 204.8 | pass | rollback |

- **R′ 机制（本算子 falsified，与姊妹单元结论相反）**：3d 单 tile 16384 f32 必然
  放弃多缓冲（f16 指示平面下单缓冲 ≈164 KB，双缓冲 > 192 KB），
  `mte2 ratio 0.552 → 0.671`、`mte2 有效带宽 31.9 → 27.1 GB/s/核`——**读被完全暴露**，
  而省下的 1 个 tile pass 只有 ~0.25 µs（§3）。4 个任务/核 × 暴露的 64 KB 读
  （1.508 µs/次）足以吃掉全部收益。⇒ 任务书要求的 R′ 实验已完成并给出否定结论（§5）。
- **UB tile cap 机制（falsified）**：32k 从 4×8192 变 2×16384：读单价从 0.192 抬到
  1.508 µs/64 KB（§3）、且失去双缓冲 ⇒ 净亏。这是
  `CONST-ub-price-grid-dependent` 的新数据点（W=8192 vs W=16384 的读单价差 7.9×）。
- OP6 的单 run 收益（4k +4.0%、3d +2.6%）都在噪声阈值附近，必须进 §4.4 的交错 A/B。

### 4.3 轮次 4：显式流水（同 base = OP6）

| 候选 | 唯一主要优化点 | 文件 |
|---|---|---|
| OP8 | 列 tile 循环改 `T.Pipelined(num_full, num_stages=2)`（32k 的 mte2/vec 不重叠假说） | `_logical_reduce_kernel_round4_op8_pipe2.py` |

| 候选 vs current best | workload | Task Duration(us) | 相对 OP6 | AICore pipe（vec/mte2/mte3） | L0 | 结论 |
|---|---|---:|---:|---|---|---|
| OP6 | 32k | 7.845 | — | 0.288/0.563/0.064 | pass | current best |
| OP8（显式 2 级流水） | 32k | 7.919 | −0.9% | 0.285/0.545/0.040 | pass | **config_no_gain** |
| OP8 | 3d | 14.805 | −6.0%（同码段，属 run-state） | 0.260/0.624/0.061 | pass | rollback |
| OP8 | 4k | 2.873 | −0.5%（同码） | 0.328/0.296/0.178 | pass | tie |

- **机制**：`T.Pipelined` 后 32k 的 pipe 指标与 auto-multi-buffer 版本逐项相同
  （mte2 0.545 vs 0.563、vec 0.285 vs 0.288）⇒ 本平台 auto multi-buffer 与显式流水
  是**同一机制**、无叠加收益——与姊妹单元 resident 路径的结论一致，本次在
  **列 tile 循环（带 loop-carried `acc`）**上再次复现，扩展了该结论的适用域。

### 4.4 采纳裁决：同机同 run 交错 A/B/A/B

A = `../_logical_reduce_kernel.py`（Stage 3 baseline），B = OP6（f16 指示平面）。
原始 profile：`profiles/ab/op6_<workload>/{A,B}_run{1..3}/` 与
`profiles/ab/final_3d/{A,B}_run{1..3}/`，汇总 `summary.json`。

**session 1（三个 workload 各 3 对）**

| workload | A 中位 | B 中位 | 逐对 Δ | 方向一致 | 裁决 |
|---|---:|---:|---|---|---|
| 4k | 2.9587 | 2.8327 | −7.34% / −3.05% / −2.74% | 3/3 | **B 快 4.26%** |
| 32k | 8.1220 | 7.9760 | −1.74% / −1.86% / −1.28% | 3/3 | B 快 1.80%（<3% 噪声阈） |
| 3d | 15.5067 | 13.9580 | −9.59% / −11.36% / −9.79% | 3/3 | B 快 9.99% |

**session 2（3d 复测，3 对；此时机器进入"快态"，A 臂 14.59 而非 15.51）**

| workload | A 中位 | B 中位 | 逐对 Δ | 方向一致 | 裁决 |
|---|---:|---:|---|---|---|
| 3d | 14.5927 | 14.7713 | +1.09% / +3.40% / +1.04% | 3/3 | **B 慢 1.22%** |

- **4k：两 session 同向 → adopt。** session 1 的 A/B −4.26%（3/3），session 2 的
  相邻 pair（baseline_recheck 3.0387 vs final 2.8453）= −6.4%；两次独立测量都支持
  f16（且 >3% 噪声阈）。
- **32k：方向一致但幅度 <3% → 记 `no_gain`**（T-2：未过协议的 <5% 增益不得记为
  winner；且它与 3d 共用同一个 path C builder，见下）。
- **3d：两 session 结论相反（−10.0% vs +1.2%），6 对符号 3:3 → sign test p=1.0 →
  不可判定 → 不采纳。** 归因：本机 run-state 双态（§1.1，同码跨 session 5–8%），
  f16 版本在 3d 上单次发射分布更宽（13.5–16.5 µs vs baseline 14.0–15.3），
  而 3d 的关键路径是读 + 每核 4 个任务的串行化（mte2 占比 0.55→0.62），
  vec 从 0.335 降到 0.256 并未转化为稳定时延收益 ⇒ **机制上也不该期待收益**。
- 结论：**按 path 分离采纳** —— resident 路径用 f16（4k 类），column-tiled 路径
  保持 Stage 3 f32。交付文件的 codegen 等同性证明见 §4.6。

### 4.5 精度门禁实证：f16 指示平面在"窄尾块"上会误判（已修复）

L2 的 `N=E_cap+1`（(32,8193) fp16，`num_full=1, tail=1`）在 f16 全局版本上报
`22/32 entries differ; first at 2: got=True want=False`。
逐尾宽扫描（`check_tails.py`，交付前）：

| tail 宽度 | 1 | 8 | 64 | 128 | 808 |
|---|---|---|---|---|---|
| f16 指示平面（全局版本） | **22/32 错** | **3/32 错** | exact | exact | exact |
| Stage 3 f32（对照） | exact | exact | exact | exact | exact |

- **根因**：窄 f16 尾块（≤8 元素）的 `vcmp → vcast(i1→f16) → reduce_max` 链在本工具链
  上 mis-reduce（f32 同形态正确）；不是 NaN/±0 语义问题，是窄宽度的 f16 归约缺陷。
- **修复**：尾块路径显式保留 Stage 3 的 **f32** 指示链（并把 `mx_tail` 以精确的
  `f32→f16` cast 并入 f16 累加器；该 cast 在 fragment 上不被本工具链接受，最终
  直接沿用 Stage 3 的整段尾块实现）。
- **复核**：`check_tails.py` 五档尾宽全部 exact；`--level all` 的 L2-tn+1 亦 PASS。
- **意义**：这是 `--level all` 门禁真正拦住的一次精度回归（L0/L1 没覆盖 tail=1）；
  也说明"f16 化指示平面"不能按 dtype 一刀切，必须逐形态验证。

### 4.6 交付形态（按 path 分派）+ codegen 等同性证明

| 路径 | 指示平面 | 依据 |
|---|---|---|
| R（resident，`N ≤ E_CAP`） | **float16** | 4k 两 session 同向 −4.3% / −6.4% |
| C（column-tiled，`N > E_CAP`） | float32（Stage 3） | 32k −1.8%（<噪声阈）、3d 冲突（−10.0% / +1.2%） |
| C + tail（`tail > 0`） | float32（Stage 3） | §4.5 的窄尾块 f16 缺陷 |

`codegen_compare.py`（`get_kernel_source()` → sha256）：

| workload | 交付文件 | Stage 3 baseline | 判定 |
|---|---|---|---|
| 4k (32,4096) f32 | `a89563c7a0218861` | `d9466d3eae110449` | **不同**（f16 resident，已采纳） |
| 32k (32,32768) f32 | `c701a2088c3c4a99` | `c701a2088c3c4a99` | **逐字节相同** |
| 3d (128,16384) f32 | `9dbec92c8be2754` | `9dbec92c8be2754` | **逐字节相同** |

⇒ 32k / 3d 的 final 与 baseline 的时延差异**只能**是 run-state（同一份生成代码），
这是 §7 中 32k/3d 判 `no_gain` 而不是"回退"的机械依据。

---

## 5. DESIGN §1.6.4 A/B 裁决记录（R′，任务书必做项）

| 项 | 主选（Stage 3：列分块 + 在线 max 合并） | 备选 R′（超预算单 tile 驻留） | 判定 |
|---|---:|---:|---|
| 3d (128,16384) f32，Stage 3 f32 指示 | 14.337 | **无法编译**（198,848 B > 196,608 B） | R′ 不可用 |
| 3d + f16 指示（OP6 = 13.967） | 13.967 | 15.715（OP4） | **R′ 慢 12.5%** |
| 3d + f16 + UB tile cap | 13.967 | 16.007（OP7） | R′ 慢 14.6% |
| 32k（N=32768 > 硬上限 16384，R′ 带外） | 7.759 | 不适用（按设计不套用） | — |

- **裁决：R′ 不采纳。** 实测 R′ = **1.125× 主选（更慢）**，与姊妹单元的 0.87× 方向
  相反；DESIGN 写的"备选 < 0.8× 才翻转"阈值本次没有被触碰（不是"差一点"，是反向）。
- **与姊妹结论冲突的归因（形态差异，不推翻其结论）**：姊妹 3d 是 **fp16**
  （8 B/元素、单 tile 131 KB、读单价 0.19 µs/32 KB 档）；本算子 3d 是 **bool→f32**
  （12.14 B/元素、单 tile 198.8 KB 溢出、W=16384 读单价 1.508 µs/64 KB）。
  R′ 的收益 = 省 1 个 tile pass − 失去多缓冲的代价；本形态第二项远大于第一项。
  ⇒ R′ 的适用域应细化为「超预算 **且** 单 tile 读单价未显著劣化」
  （本机判据：W ≤ 8192 可用；W = 16384 的 f32 形态不可用）。
- 32k 不在 R′ 触发带内（N=32768 > fp32 硬上限 16384），未强行套用。

---

## 6. Roofline / 归因分析

### 6.1 本机屋顶口径

`roofline_from_profile.py` 读 profile 内 `visualize_data.bin` 的
`GM/L2 : GM Read + Write`：本机 **0.8 TB/s**（4 个 baseline profile 全一致；3d
baseline 的 msprof `ratio` = **73.23%**）。台账口径仍记 **1.75 TB/s**（fabric 聚合），
两者相差 2.19×，是 SKU/口径差异，不互相换算、不跨机比较。

### 6.2 逐 workload 归因（final = 交付文件实测）

| workload | kernel 字节 | 560 GB/s 验收线 | final 实测 | 达成带宽 | Ratio/1.75 | Ratio/0.8 | 主项归因 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4k | 524,320 | 0.936 µs | **2.8127** | 186 GB/s | 10.7% | 23.3% | **空 kernel 地板**（16 块 1.845 µs = 66% of Task）；验收线比地板还低 2×，结构性不可达 |
| 32k | 4,194,336 | 7.490 µs | 8.112 | 517 GB/s | 29.5% | 64.6% | 地板 2.675 + 4 × (读 0.19 + 归约 1.0)；32 块 / 40 核（M=32 上限） |
| 3d | 8,388,736 | 14.980 µs | **14.712** | **570 GB/s** | 32.6% | 71.3% | 地板 ~2.65 + 4 任务/核 × 2 tile；**越过 560 GB/s 验收线 1.8%** |

- **4k**：Task 2.8127 ≈ 地板 1.845 + 读 0.244 + 归约 0.595 + 比较/指示/收尾 ~0.13。
  验收线 0.936 µs **低于 8 块形态的空 kernel 地板 1.455 µs**，也低于 16 块地板
  1.845 µs ⇒ 任何算法/tiling 都不可达（与姊妹 seq 用例同型结论）。本次仍拿到
  −5.6%（round-0 基准）/ −4.3%（同 run A/B）的地板以上优化。
- **32k**：Task 8.112 ≈ 地板 2.675 + 4 × (读 0.19 + 归约 1.0 + cmp/cast ~0.1) ≈ 7.9。
  归约 pass（4 × 1.0 ≈ 4.0 µs，占 49%）是最大单项，其单价由**行宽 N=8192** 决定
  （§3）；唯一能减半它的手段（tile 加宽到 16384）会把读单价从 0.19 抬到 1.508 并
  失去双缓冲（§4.2 OP5/OP7 实测净亏）。⇒ 剩余差距（8.112 vs 7.490，−7.7%）
  落在"归约单价 vs 读单价不可兼得"区间，且 32 行只占 40 核中的 32 核。
- **3d**：Task 14.712（round-0 同码 14.337、round-4 recheck 14.537）= 地板 ~2.65 + 8 个 tile
  （4 任务 × 2 tile）的读/归约。3d 的 final 代码与 Stage 3 **逐字节相同**（§4.6），
  因此本次调优对 3d 的贡献是**证明 R′ 与 tile 加宽都会退化**，最终保留 Stage 3 结构；
  按 560 GB/s 线判定**达标**（570 GB/s）。

### 6.3 口径复核（工具链常数）

| 条目 | 本任务重验方式 | 结论 |
|---|---|---|
| `CONST-vector-launch-overhead`（0.23–0.5 µs/pass） | 逐 pass 0.11–0.13 µs（N=4096 档）、~0.5 µs/pass（N≥8192 归约档） | 方向成立；本形态**低于低档** |
| `CONST-mte2-degradation`（30.2 GB/s/核） | 43–171 GB/s/核，随 tile/W 变化（含 L2 命中） | 量级带内，但**强烈依赖 W**（新增 W=16384 → 43 GB/s/核） |
| `CONST-msprof-task-floor-910B3`（3.37 µs） | 空 kernel 1.455 / 1.845 / 2.51–2.68 µs（8/16/32 块） | 量级成立且**与 block 数强相关** |
| `CONST-reduce-ub-price-axis-alignment` | 16384 元素 f32 → 198,848 B（12.14 B/元素） | **须按 (W, dtype) 分档**；8 B/元素外推会漏判 overflow |
| `CONST-ub-price-grid-dependent` | 同数据同时给出容量/单价两口径；W=8192 vs 16384 读单价差 7.9× | 成立 |
| `PL-1.25`（指示平面 → 行归约 → 补式） | 本次新增 f16 指示形态（resident 采纳）+ 窄尾块 f16 缺陷 | 成立并被扩展/限定 |
| `TRAP-vcmp-ne-ordered-nan` | 未改比较路径（仍 `vcmp("eq")`）；L0-2 三 dtype NaN 用例逐位 PASS | 仍成立 |
| `TRAP-int32-vcmp-scalarized` | 未触发（无 int32 比较路径） | 不适用 |

### 6.4 设计层观察（只记录，不产出 `perf_feedback.md`）

任务书裁定：本算子族的族级归因（fabric 1.75 TB/s vs AIV 自报 0.8 TB/s）已归档在
姊妹单元的 `perf_feedback.md`，**本任务不重复产出 `[DESIGN_LIMIT]`**。以下是本算子
特有的设计层观察，供 conductor / Stage 1 参考（**本阶段不实施**）：

1. **bool→f32 预转换是本算子的主要矛盾**：manifest 全是 bool，Op 层
   `to_logical_float32` 把它转成 f32，kernel 的 GM 流量因此是 4 B/元素
   （3d 8.39 MB 而非 2.10 MB），而判断本身只需要 1 bit。若 Op 层能传 int8/位平面
   （或 kernel 直接吃 packed bool），32k/3d 的读与 UB 压力都会降到 1/4。
   当前工具链不接受 bool 作为 `alloc_shared`/`T.copy` 存储 dtype（wrapper 已注明），
   故这是**设计层**改进空间，不是 tiling 参数可解。
2. **4k 的 560 GB/s 线（0.936 µs）低于本机空 kernel 地板（1.455–2.511 µs）**，
   属口径不可达；**32k 的线（7.490）与"归约单价 vs 读单价不可兼得"区间冲突**，
   剩余 8% 需要换归约原语（如 (bm×K, N/K) 两级归约或位平面归约），
   未实测，只作为残余机会记录（§8）。
3. **R′ 的适用域**需按 W/读单价细化（§5），DESIGN §1.6.4 的"fp32 16384 元素可编译"
   假设在本工具链上不成立（实测 198,848 B > 196,608 B）。

---

## 7. Final Summary

最终候选：`perf_opt/_logical_reduce_kernel.py`
（`candidate_id = final_f16res`，sha256
`70ecea6df2cf0dc9ce75c77b043303c91e6e04e78357507009347302a356de5b`）
= **Stage 3 kernel 结构 + resident 路径的 f16 指示平面**；`_select_config` 仍是
tiling 单一真源（`TUNED_E_CAP` / `TUNED_RPRIME` / `TUNED_TILE_CAP_MODE` /
`TUNED_BM_POLICY` 四个模块级常量把**实测否决**的备选固化为默认关闭），
`op_kind != "all"` 仍 raise，`_select_config` 已加入 `__all__`。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| _logical_reduce_kernel::main | mask-validation-4k-bool | 2.9793 | 2.8127 | final_f16res |
| _logical_reduce_kernel::main | mask-validation-32k-bool | 7.7587 | 8.1120 | final_f16res |
| _logical_reduce_kernel::main | 3d-multidim-reduce-bool | 14.3373 | 14.7120 | final_f16res |

参考信息（非判据，不计入上表）：同 session `baseline_recheck` 与相对其的 Δ，以及
交错 A/B 的裁决 —— 4k：recheck 3.0387 µs、Δ −7.4%、A/B −4.26%（session 1，3/3）
⇒ **adopted / winner_merged**；32k：recheck 8.1473 µs、Δ −0.4%、A/B −1.80%
（session 1，3/3，低于 3% 噪声阈）⇒ `no_gain`；3d：recheck 14.5373 µs、Δ +1.2%、
A/B 两 session 冲突（−9.99% / +1.22%，6 对符号 3:3）⇒ `no_gain`。

> 上表 final 列 = 在**交付文件本身**（sha `70ecea6d…`，逐 workload `--check` 逐位
> PASS）上采集的 `phase=final` 记录原值，未拼接、未挑选；`baseline_us` 沿用
> round-0 baseline 行（sha `1f7d1a18…`）。
> `baseline_recheck` 是 final session 之前一个 session 的同机复采（§1.1），
> 用于吸收 5–8% 的 run-state 漂移。
> **32k / 3d 的 final 与 Stage 3 生成代码逐字节相同**（§4.6 的 `get_kernel_source`
> sha 对照），因此它们与 round-0 基准的 +4.6% / +2.6% 差异**只能**是 run-state，
> 不构成回退；这也是把这两例判为 `no_gain`（而非"回退"）的机械依据。

### 7.1 final 重采说明（supersede，必读）

本任务在第一次 `phase=final` 采集（round 4，`candidate_id = final_f16ind`，
sha `48e44b49…`）之后，依据 §4.4 session 2 的 A/B（3d 上 B 反而 +1.2%，6 对符号
3:3）与 §4.5 的窄尾块精度缺陷，把交付形态改为**按 path 分派**（resident=f16，
column-tiled=Stage 3）。文件内容变了，旧的 final 行随之失效，处置如下：

1. round-4 的 3 条 final 行**一字不改**保存在 `perf_records_full_history.jsonl`
   （该文件含全部 30 条原始测量，含被取代的 final；append-only 语义在 sidecar 侧完整）；
2. 主 `perf_records.jsonl` 保留全部 baseline / candidate / baseline_recheck 行 +
   每 tune workload **唯一**权威 final 行（round 5，`final_f16res`，sha 与交付文件一致，
   共 27 行）；
3. `workload_inventory.json` 的 `final_artifact_sha256` 与上述 final 行一致。

## 8. 精度回归（`python perf_opt/_logical_reduce_kernel.py --level all`，exit 0）

- **L0：27 例全 PASS**（含 L0-2 构造行三 dtype：NaN 无零 → True、NaN 混零 → False、
  ±0 → False、±inf/次正规 → True；L0-4 核切分边界）。
- **L1：25 例全 PASS**（非连续输入、高阶/非末轴 dim 的 Op 层 movedim、极值/亚正规、
  bool/int32/int64/complex 的 f32 预转、M 尾块、path C 的静态尾块 tail=808、
  caller block_m 钳位）。
- **L2：全部 PASS**（工厂期 raise 5 项、最小形状 5 项、path R ≡ path C 逐位一致、
  `N=E_cap+1`（tail=1）——该项在 f16 全局版本上曾 FAIL，见 §4.5，已修复）。
- **Boundary：全部 PASS**（三 dtype 全零/全一/NaN 列、1D、`(512,2048)`）。
- `torch.equal` 逐位比对（atol=rtol=0）；NaN / -0 / 构造行用例**逐字保留**，
  本次未改比较路径（仍 `vcmp("eq")` + 补式，未触碰 `"ne"`）。
- **smoke 三例**（bool/fp16/fp32，(32,256)）：`bench_driver --check` → `IMPL_CHECK PASS`
  （逐位），`precision_pass=true`。
- **wrapper 接线核对**（`check_wrapper_path.py`，不改 wrapper）：用
  `LogicalReduceKernel.default_config["block_m"]`（wrapper 实际传入值）调用交付工厂，
  三个 tune workload 的 `resolved_bm` 分别为 2 / 1 / 1（与 `_select_config` 一致），
  输出与 `torch.all` **逐位相等**（exact=True）。
- 日志：`logs/final_L0L1L2_boundary.log`、`logs/round5/*`。

## 9. 验收判定（双轨，不混用）

1. **族验收线（真判据）**：`0.7 × 设备自报 roof = 0.7 × 0.8 TB/s = 560 GB/s`
   （按 kernel f32 字节）。

| workload | 总流量（kernel f32） | 验收线 | final 实测 | 达成带宽 | 状态 |
|---|---:|---:|---:|---:|---|
| mask-validation-4k | 524,320 B | 0.936 µs | 2.8127 | 186 GB/s | **结构性不可达**（线 < 空 kernel 地板 1.455–1.845 µs） |
| mask-validation-32k | 4,194,336 B | 7.490 µs | 8.1120 | 517 GB/s | **未达**（差 7.7%；归因见 §6.2） |
| 3d-multidim | 8,388,736 B | 14.980 µs | 14.7120 | **570 GB/s** | **达标**（+1.8% 余量；session 1 同码 14.337 → 585 GB/s，余量 4.3%） |

2. **台账线（只记录、不作判据）**：`Ratio = achieved_bw / 1.75 TB/s`
   = 10.7% / 29.5% / 32.6%（见 §6.2），与本机自报口径（23.3% / 64.6% / 71.3%）
   并列记录，不参与任何决策。

## 10. 残余机会与已知残留

**残余机会（defer，未实施）**

1. **32k 的归约原语更换**：把 (1,32768) 的两级归约做成
   `(K, N/K)` 形状的 reshape + 两次 `reduce_max`，或走位平面归约，
   目标是把 4 × 1.0 µs 的归约 pass 降到 ~2 µs。**未实测**（超出本轮 4 轮/8 候选
   预算，且需先验证 `T.reshape` 在 shared 上的任意形状支持）。
2. **f16 指示平面在 3d 上的稳定化**：vec 确实从 0.335 降到 0.256，但时延不可复现；
   若后续能在同一 session 内跑 ≥10 对 A/B 并确认方向，可再评估。
3. **bool→f32 预转换**（§6.4-1）：设计层改动，本阶段不做。
4. **32k 的 8 个空闲核**（32 行 / 40 核）：需要跨核 N-split + 合并 kernel
   （DESIGN 已判净亏；本次未重测）。

**已知残留（如实披露）**

- `perf_opt/` 顶层另有 10 个候选/沙箱文件（`_logical_reduce_kernel_round*`、
  `_logical_reduce_kernel_lab.py`、`_logical_reduce_kernel_prevfinal_f16all.py`、
  `_logical_reduce_kernel_final_candidate.py` 等）与 helper 脚本；它们不参与交付，
  仅作证据链。`_logical_reduce_kernel_prevfinal_f16all.py` 是被取代的 round-4 交付内容
  （sha `48e44b49…`），保留供 §7.1 的 supersede 追溯。
- `copy_floor_calibration.jsonl` 中有少量探针重复测量（两个 runner 实例曾并行启动，
  同一 mode 被采两次），重复点差异 ≤3%，不影响单价反解；已在 §3 表内取首次值。
- 本机 run-state 漂移 5–8%，故 final 的绝对值只与"同 session recheck / A/B"可比，
  跨 session 的 ±5% 差异一律按 run-state 处理（§1.1、§4.6 的 codegen 依据）。

---

## 11. Skill Retrospective

### 流程有效性

1. **地板先行（§3）避免了两个错误方向**：empty 探针把"空 kernel 地板"量到
   1.455/1.845/2.51–2.68 µs（随 block 数），直接预判了 OP3（多开核）必亏，
   实测 −18.5% 印证；`read_f16` 探针又提前说明"f16 归约本身只快 2–6%"，
   把期望从"减半归约时间"纠正到"省 UB + 少一次 cast"。**探针成本 ~19 次 msprof，
   换来 3 个方向的取舍**。
2. **单点结构 + 同 base 派生**执行到位：8 个候选各只改一个主要结构点（R′ / tile cap /
   行块策略 / 指示平面 dtype / 显式流水 / 组合），每个都有独立 sha 与 L0 结果，
   blocked 与 rollback 都能归因到具体机制。
3. **交错 A/B 两次 session 的冲突（§4.4）是本次最重要的方法论收获**：单 run
   "3d −10%" 的漂亮数字在第二个 session 变成 +1.2%。若只做一次 A/B 就会把一个
   run-state 假象写进交付件。**建议 skill 在 T-2 里明确：跨 session 复测一次，
   两 session 结论冲突即判 inconclusive、不得采纳**（本次按此执行）。
4. **`--level all` 门禁真的拦住了回归**：L2 的 `N=E_cap+1`（tail=1）暴露了窄尾块
   f16 误判（22/32 错），而 L0/L1 全绿。说明 L2 虽标"record only"，
   在 dtype 变更类候选上应升级为**必须逐项检查**的信号。

### 需要修正的 skill 指引

- `references/run_experiments.py` 模板的 `run_l0` PYTHONPATH 推导（`dirname(dirname(HERE))`）
  在本项目的目录层级（`kernels/<family>/<op>/<op>_kernel/perf_opt`）下
  指向错误目录，导致子进程 `import tilelang` 失败并**把 L0 记成 fail**。本次改为显式
  导入 `run_bench.REPO_ROOT/TILEOPS_DIR`。建议模板直接复用 `run_bench` 的常量。
- 模板未提示"msprof op 的 pipe CSV 无 `Op Name` 列（逐 block 行）"，
  `_parse_pipe` 需按无 `Op Name` 分支聚合，否则 pipe 指标恒为空。
- 模板未提示 `dtype` 变更类候选要跑"窄宽度扫描"（tail=1/8/64…）；
  建议在精度检查清单里加一条"最小宽度/最小尾块"用例。

### Value Point Proposals（按 vp_type，含证据三件套）

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-A | D | **空 kernel 地板随 block 数分档**：8 块 1.455、16 块 1.845、32 块 2.511–2.675 µs（同构 persistent 网格、仅 1 B 输出写）⇒ "多开核"必须先扣这条地板 | `copy_floor_calibration.jsonl`（`probes/M32_N4096_bm{1,2,4}/empty`）、§4.1 OP3 的 4k −18.5% | `perf_opt/probe_floor.py` + `run_probes.py` | 4d2e413 + CANN 9.0.0 + 910B4(40 AIV) |
| VP-B | D | **归约 pass 单价由行宽 N 决定，不由元素总数**：N=4096 → 0.55–0.60 µs；N=8192/16384 → ~1.0 µs（同元素数下差 1.7×）⇒ 加宽 tile 省 pass 数是伪收益 | `op_cost_calibration.jsonl` 的 `reduce_pass` 5 个数据点（0.551/0.595/0.592/0.999/0.999） | 同上 | 同上 |
| VP-C | D | **MTE2 读单价随 W 变宽而劣化**：16 KB 124 GB/s/核、32 KB@bm2 131、32 KB@W8192 171、64 KB@W16384 43 GB/s/核（7.9× 差）⇒ 单 tile 超预算形态（R′）在 W=16384 上必然亏 | `op_cost_calibration.jsonl` 的 `mte2_read_tile` 5 点 + `CONST-ub-price-grid-dependent` | 同上 | 同上 |
| VP-D | P（负向） | **R′（超预算单 tile 驻留）的适用域按 dtype/W 分裂**：fp16 W=16384（姊妹）−12%；本项目 bool→f32 W=16384 **+9.6%**（mte2 ratio 0.552→0.671，读暴露）且 f32 指示平面下 16384 元素**不可编译**（12.14 B/元素）。判据：先测 W 档读单价，W=16384 的 4 B 元素形态不套用 R′ | `perf_records.jsonl` r3_op4/r3_op7 vs r3_op6；`logs/round1/L0_*.log` 的 ub overflow 原文 | `_logical_reduce_kernel_round3_op4_rprime.py` / `_round1_op1_rprime.py` | 同上 |
| VP-E | D | **16384 元素 f32 行归约的 UB 单价 = 12.14 B/元素（198,848 B，超出 192 KB）**；f16 指示平面省 2 B/元素后可编译。8 B/元素线性外推会漏判 | `logs/round1/L0_*.log`（`requires 1590784 bits while 1572864 bits available`）+ §4.6 codegen | `_logical_reduce_kernel_round1_op1_rprime.py` | 同上 |
| VP-F | D | **f16 指示平面在窄尾块（tail ≤ 8）上 mis-reduce**（(32,8193) 22/32 行错、(32,8200) 3/32；tail ≥ 64 exact），f32 同形态正确；尾块必须保留 f32 链 | `check_tails.py` 输出 + L2-tn+1 | `check_tails.py` | 同上 |
| VP-G | R | **A/B 必须跨 session 复测一次**：单 session 3 对得 3d −10.0%（3/3），第二 session 3 对得 +1.2%（3/3）⇒ 符号 3:3、不可判定。建议 T-2 增加"两 session 冲突即 inconclusive"的硬规则 | `profiles/ab/op6_*`、`profiles/ab/final_3d/summary.json` | — | 流程规则 |
| VP-H | P | **f16 指示平面（等价 0/1 域）在 resident 路径上是稳定 −4~6%**：少一次全宽 `f32→f16` cast + 2 B/元素 UB；判据是 `vec_ratio` 与 Task 同降（4k 0.375→0.285，2.979→2.833） | `perf_records.jsonl`（baseline vs r3_op6 4k）+ `profiles/ab/op6_mask-validation-4k-bool` | `_logical_reduce_kernel_round3_op6_indf16.py` | 同上 |

> 说明：本轮**未**回写 pattern-library / capability-gaps（任务书裁定本任务
> `[DESIGN_LIMIT]` 不触发、且知识库写入交由 evolver 统一处理）；以上提案原文
> 供 `tilelang-skill-evolver` 蒸馏。

## 12. Phase A（round 6）：同 session 交错 A/B 裁决

> 本节为 **append 章节**，§1–§11 未作任何修改。裁决目的：在**单一 session、单一 run-state**
> 内交错测量 A（Stage 3 baseline，只读）与 B（`perf_opt` 交付 `final_f16res`），
> 判定 `perf_opt` 交付是否真的回退，以及其 4k 收益是否成立。

### 12.1 执行条件与环境（实测，非引用）

| 项 | 值 |
|---|---|
| 时间窗（UTC） | 2026-10-20T02:21:08Z – 02:37:36Z（单条链式进程，4k → 32k → 3d，中途无其它重负载任务） |
| 并发纪律 | 全程 `flock /tmp/tilelang-msprof.lock` 串行，无并行 msprof |
| 测量口径 | `msprof op --launch-count=30 --warm-up=5`，kernel-only `Task Duration(us)` 的 **mean**（每 run 30 次发射，`n=30` 已核对） |
| A 工件 | `../_logical_reduce_kernel.py` sha256 `1f7d1a18…7ae69e53`（与派单期望值一致；逐 run 回写 `impl_sha256` 复核） |
| B 工件 | `perf_opt/_logical_reduce_kernel.py` sha256 `70ecea6d…356de5b`（同上） |
| 器件（实测） | `npu-smi` / `torch_npu.get_device_properties` 均报 **Ascend910B3**，vector_core_num = 40，CANN 9.0.0，tilelang `0.1.2+cca96247…`（commit cca9624） |
| 新 tag | `phaseA6_4k` / `phaseA6_32k` / `phaseA6_3d`（旧 `phaseA_*` 目录未复用） |

**执行 config 核对（关键）**：`bench_driver` 逐 run 打印的 `IMPL_CFG` 为
4k `path=R block_m=2 tile_n=4096`、32k `path=C block_m=1 tile_n=8192 num_full=4`、
3d `path=C block_m=1 tile_n=8192 num_full=2`，两臂完全一致。

TileOPs 报告里记录的 `config` 是 4k `block_m=8` / 3d `block_m=2`，看似与本轮不同，但
`_resolve_block_m()`（`perf_opt/_logical_reduce_kernel.py:319-334`，调用点 369 / 431 / 500，
`E_CAP = 8192`）在 `block_m * tile_n > E_CAP` 时回落到 `_select_config` 的 ladder 值：
8×4096 = 32768 > 8192、2×8192 = 16384 > 8192，两处均触发回落 → 实际执行 block_m = 2 / 1。
**⇒ 本轮 A/B 测的正是 benchmark 实际执行的形态**，不存在「测了非部署 config」的口径缺口。

**作废声明**：上一 session 在 4k 第 1 对后被人为终止，其 `profiles/ab/phaseA_*` 与
`logs/phaseA/ab_4k.stdout` 数据**全部作废**，本轮未引用、未拼接；三个 workload 均从零重测。
`logs/phaseA/codegen_compare.txt` 属已完成且未受中断影响的证据，本轮已用当前文件 sha 重跑覆盖
（新 run 时间 2026-10-20T02:20:40Z，三行哈希与旧文件逐项一致）。

### 12.2 mask-validation-4k-bool — 4k  (32,4096) bool->f32, path R, bm=2, tile=4096

| 对 | A mean (µs) | B mean (µs) | Δ% = (B−A)/A | 方向 |
|---|---|---|---|---|
| 1 | 2.8567 | 2.6887 | -5.88% | B 更快 |
| 2 | 2.8020 | 2.7420 | -2.14% | B 更快 |
| 3 | 2.8160 | 2.7133 | -3.65% | B 更快 |
| 4 | 2.8247 | 2.6940 | -4.63% | B 更快 |
| **中位** | **2.8203** | **2.7036** | **-4.14%** | — |

- 逐对 delta 中位 = -4.14%；臂中位相对变化 = -4.14%（两者同号）
- 方向一致性 = **4/4 同号**（一致）
- 臂内极差：A 0.0547 µs，B 0.0533 µs（30 次发射 mean 的 run-to-run 波动）
- 同 run pipe（msprof `PipeUtilization`，臂内均值）：AIV vec_ratio 0.3624 → 0.3100；aiv_time 1.781 → 1.652 µs；mte2_ratio 0.2819 → 0.3037；mte3_ratio 0.1268 → 0.1459

### 12.3 mask-validation-32k-bool — 32k (32,32768) bool->f32, path C, bm=1, tile=8192x4

| 对 | A mean (µs) | B mean (µs) | Δ% = (B−A)/A | 方向 |
|---|---|---|---|---|
| 1 | 6.6320 | 6.7893 | +2.37% | B 更慢 |
| 2 | 6.7333 | 6.7680 | +0.52% | B 更慢 |
| 3 | 6.7113 | 6.7567 | +0.68% | B 更慢 |
| 4 | 6.6727 | 6.8247 | +2.28% | B 更慢 |
| **中位** | **6.6920** | **6.7786** | **+1.29%** | — |

- 逐对 delta 中位 = +1.48%；臂中位相对变化 = +1.29%（两者同号）
- 方向一致性 = **4/4 同号**（一致）
- 臂内极差：A 0.1013 µs，B 0.0680 µs（30 次发射 mean 的 run-to-run 波动）
- 同 run pipe（msprof `PipeUtilization`，臂内均值）：AIV vec_ratio 0.4063 → 0.4021；aiv_time 5.204 → 5.263 µs；mte2_ratio 0.3775 → 0.3777；mte3_ratio 0.0537 → 0.0644

### 12.4 3d-multidim-reduce-bool — 3d  (4,128,4096)->(128,16384) bool->f32, path C, bm=1, tile=8192x2

| 对 | A mean (µs) | B mean (µs) | Δ% = (B−A)/A | 方向 |
|---|---|---|---|---|
| 1 | 10.8187 | 10.7227 | -0.89% | B 更快 |
| 2 | 10.9173 | 10.7620 | -1.42% | B 更快 |
| 3 | 10.7593 | 10.9567 | +1.83% | B 更慢 |
| 4 | 10.9107 | 10.7220 | -1.73% | B 更快 |
| 5 | 10.8027 | 10.8487 | +0.43% | B 更慢 |
| **中位** | **10.8187** | **10.7620** | **-0.52%** | — |

- 逐对 delta 中位 = -0.89%；臂中位相对变化 = -0.52%（两者同号）
- 方向一致性 = **3/5 同号**（不一致）
- 臂内极差：A 0.1580 µs，B 0.2347 µs（30 次发射 mean 的 run-to-run 波动）
- 同 run pipe（msprof `PipeUtilization`，臂内均值）：AIV vec_ratio 0.4322 → 0.4237；aiv_time 7.931 → 8.087 µs；mte2_ratio 0.4015 → 0.3943；mte3_ratio 0.1494 → 0.2065

### 12.5 codegen 一致性（源码级 + 生成源码级）

重跑命令（cwd = `perf_opt/`，run 时间 2026-10-20T02:20:40Z）：
`PYTHONPATH=<repo>:<TileOPs> python codegen_compare.py ../_logical_reduce_kernel.py _logical_reduce_kernel.py`，
证据留存 `logs/phaseA/codegen_compare.txt`。

| shape | A 生成源码 sha256[:16] | B 生成源码 sha256[:16] | 一致 | cfg |
|---|---|---|---|---|
| (32,4096) | `d9466d3eae110449` | `a89563c7a0218861` | **否**（预期：path R 换 f16 指示平面） | path=R, bm=2, tile_n=4096 |
| (32,32768) | `c701a2088c3c4a99` | `c701a2088c3c4a99` | **是（逐字节）** | path=C, bm=1, tile_n=8192, num_full=4 |
| (128,16384) | `9dbec92cf8be2754` | `9dbec92cf8be2754` | **是（逐字节）** | path=C, bm=1, tile_n=8192, num_full=2 |

源码级交叉核对（当前文件行号）：
- path R 主体用 `T.alloc_shared((block_m, N), TUNED_IND_DTYPE)`（L387），`TUNED_IND_DTYPE = "float16"`（L171）；
- path C 满块体（L445）与 path C 尾块体（L514 / L518）**均硬编码 `"float32"`** 指示平面，并保留 f32→f16→i8 收尾；
- 模块级 tuned 默认值 `TUNED_RPRIME=False`（L173）、`TUNED_TILE_CAP_MODE="ecap"`（L174）、
  `TUNED_BM_POLICY="ladder"`（L175）＝ Stage 3 dispatch。

⇒ 交付件自述「只改 resident 路径、column-tiled 路径保留 Stage 3 f32 链、生成代码与 Stage 3 逐字节一致」
在 **源码级与生成源码级双重成立**（对 32k / 3d 两个 bench shape）。
⇒ 由此得到两个**零效应对照**（生成码逐字节相同 ⇒ 真实差异必然为 0，测出的量全部是噪声）：
32k 测出 **+1.29%（4/4 同号偏慢）**、3d 测出 **−0.52%（3/5，符号不一致）**。
两者给出 ±1.3% 量级、符号不一致，故不能归因为单向的顺序偏置；可用的结论是
**本测量的零效应噪声地板 ≈ ±1.3%**（与派单的 3% 阈值同量级）。4k 的 −4.14% 约为该地板的 3.2 倍。

### 12.6 判定（按派单规则，未放宽）

| 判定分支 | 触发条件 | 本轮实测 | 触发 |
|---|---|---|---|
| CURRENT_PERF_OPT_REGRESSES | 任一 workload 中位 delta > +3% **且** ≥(n−1)/n 同号 | 32k 中位 **+1.29%**（4/4 同号）—— 未越过 +3%；3d 中位 −0.52%（3/5 同号，不一致）；4k −4.14%（B 更快） | **否** |
| BENEFIT_CONFIRMED | 无稳定回退 **且** 4k 增益 ≥5% | 4k 增益 **4.14%** < 5% | **否** |
| MARGINAL | 无稳定回退 >3%，且 4k 增益 <5%，32k/3d 视作 tie（abs(delta) ≤3%） | 全部满足：32k +1.29%、3d −0.52% 均在 ±3% 内，且这两个 shape 的生成码与 A 逐字节相同 | **是** |
| INCONCLUSIVE | 符号不一致或边缘 | 3d 有 2/5 反号，但其生成码与 A 逐字节相同（机理上不可能有真实差异），不属于无法解释的边缘 | 否 |

**结论：`MARGINAL`** —— 无任何 workload 出现稳定 >3% 的回退，交付件可以保留；
4k 收益方向 4/4 一致、幅度 4.14%（约为零效应噪声地板 ±1.3% 的 3.2 倍，且 pipe 侧有同向因果：
aiv_time 1.781 → 1.652 µs、vec_ratio 0.3624 → 0.3100，符合"少一次全宽 cast"的预期），
但**未达到 5% 的 BENEFIT_CONFIRMED 门槛**，故按规则不宣称收益成立；32k / 3d 判为 tie。
允许进入 Phase B。

### 12.7 与历史报告差异的解释（为什么历史数字不能与本轮混用）

1. **单样本 vs 30 次发射**：三份历史报告 `benchmark.json` 的 `result.msprof.sample_count` 全为 **1**
   （`captured_op_count = 1`），即每个 case 只有 1 个采样点；本轮是 30 次发射的 mean，且每臂重复 4–5 次。
   单样本本身无法区分 ~±5% 的 run-state 漂移。
2. **跨 run / 跨 session**：优化版报告 3.08 / 8.10 / 15.82、baseline 报告 2.96 / 8.42 / 14.48、
   baseline 复测 2.84 / 7.98 / 14.36，分别生成于 2026-10-19T03:44 / 07:48 / 07:53，是三次不同进程、
   不同 run-state。§11 VP-G 已记录同一现象的教训：同一 A/B 在不同 session 得到 −10.0% 与 +1.2%。
3. **绝对量级已整体漂移（环境级）**：同一份**未修改**的 Stage 3 文件（A 臂，sha `1f7d1a18…`）
   本轮实测 4k 2.82 / 32k 6.69 / 3d 10.82 µs，历史 baseline 报告为 2.84–2.96 / 7.98–8.42 / 14.36–14.48 µs，
   即 4k 基本持平、**32k −16%…−20%、3d −25%**。启动/占空比受限的小 shape 不动、带宽受限的两个 shape
   大幅变快，形态上指向 HBM 带宽 / 器件差异：本轮实测器件报 **Ascend910B3**，而三份历史报告的
   `environment` 均记录 **Ascend910B4**，CANN / tilelang 构建亦可不同。
   ⇒ 跨报告的绝对时延**不可比**，只有同 run 交错 A/B 的相对量可用。
4. 因此本节结论只采用本轮同 session A/B 数据；`perf_records.jsonl`（交付台账）本轮**未改动** ——
   它承载的是 round 5 交付时点的测量，不能填入本轮 A/B 数值。

### 12.8 本轮不做的事

- Phase A 只测量：未改动任何 kernel 实现、未新建 / 覆盖 `_logical_reduce_kernel_*.py` 候选文件；
  wrapper / baseline / `tests/` / `benchmarks/` / `_logical_reduce_kernel_DESIGN.md` 均未触碰。
- 未调用 statectl，未写 `.stage_state.json`。
- `perf_records_full_history.jsonl` 追加 26 行（3 workload × 4/4/5 对 × 2 arm，每 arm 每 run 一行，
  `phase="ab_adjudication"`、`round=6`、含 `tag / pair / arm / duration_us / artifact_sha256 / timestamp`），
  既有 30 行未改。

## 13. Phase B（round 6）：32k 归约原语

> 本节为 **append 章节**，§1–§12 未作任何修改。Phase A 判定 `MARGINAL` 后放行进入
> Phase B（允许在 `perf_opt/` 内改实现）。本节的目标是**先取证再动手**：
> 本机重标定结构地板 → 验证 `T.reshape` 可用性 → 按实测瓶颈派生候选 →
> 同 run 交错 A/B 裁决 → 采纳或回滚。

### 13.0 结论摘要

| 项 | 结论 |
|---|---|
| B0.1 32k 纯读地板（本机，path-C 同网格/同 tile） | **4.6680 µs**（empty 地板 2.4780） |
| B0.1 32k 头寸（production − 纯读地板） | **2.0907 µs**（= 4 tile × 0.52 µs 的整条指示链，不是「4×1.0 µs 归约 pass」） |
| 派单目标「4×~1.0 µs 归约 pass 降到 ~2 µs」 | **本机算术上不成立**：4 个 tile 的**整条链**（vcmp+vcast+reduce+vmax+epilogue）合计 2.09 µs；其中归约只是三条全宽 pass 之一的上界 1/3 |
| B0.2 `T.reshape` | **可用**：view 语义（`collapse_shape`+`expand_shape`→`vreduce`，无 vcopy），两段归约逐位精确；边界=必须整除（元素数守恒） |
| cand1（reshape 两段 reduce，K=8） | **不采纳**：32k **+2.97%**（臂中位）/ +3.33%（逐对中位），**4/4 同向变慢** |
| cand4（目标瓶颈导向：4096 列 tile） | **不采纳**：32k **+28.64%**（2/2 同向），mte2 活跃带宽 62.7→37.5 GB/s |
| cand2（位平面归约） | 仅估算：净收益 **< 3%**（packing 需 vcast+shift+第二段 reduce，pass 数不减）→ 未实现 |
| cand3（abs-min / min+max 谓词） | 探针证伪 NaN 语义（NaN+0 行误判、异号行漏检 0）→ **实现前即否决** |
| B2（跨核 N-split + merge） | 仅估算：64 block = 2 波（调度收益被抵消）+ merge ≥1.8 µs（≥27%）→ **净收益 ≤0 ⇒ 禁止实现** |
| 最终工件 | `perf_opt/_logical_reduce_kernel.py` **未改动**，sha256 `70ecea6d…356de5b`（current best 保留） |
| 台账 | `perf_records.jsonl` **未改动**（27 行）；`perf_records_full_history.jsonl` **+22 行**（56→78，前 56 行逐字节不变） |

### 13.1 执行条件（实测，非引用）

| 项 | 值 |
|---|---|
| 时间窗（UTC） | 2026-10-20T02:42Z – 03:30Z（单 session，全程 `flock /tmp/tilelang-msprof.lock` 串行） |
| 测量口径 | `msprof op --launch-count=30 --warm-up=5`，kernel-only `Task Duration(us)` 的 **mean**（每 run `n=30` 已核对） |
| 器件（实测） | `npu-smi` + `torch_npu.get_device_properties(0)`：**Ascend910B3**，vector_core_num=40，cube=20，L2 192 MB，CANN 9.0.0 |
| 工具链 | tilelang `0.1.2+cca9624754b7fcc2ba1840a9f3bf82f613cdb20c`（commit cca9624） |
| A 臂 | `perf_opt/_logical_reduce_kernel.py` sha256 `70ecea6df2cf…a356de5b`（candidate `final_f16res`，= `perf_opt` 交付件，**本轮只读**） |
| 上游 baseline | `../_logical_reduce_kernel.py` sha256 `1f7d1a18…7ae69e53`（**本轮只读，未参与 Phase B 裁决**） |
| 新 tag | `phaseB_cand1k8_32k`、`phaseB_cand4t4096_32k_pre`（未复用 `phaseA6_*`） |
| 执行 config 核对 | 32k `path=C block_m=1 tile_n=8192 num_full=4`；3d `path=C block_m=1 tile_n=8192 num_full=2`；4k `path=R block_m=2 tile_n=4096`（两臂逐 run 一致，见 `IMPL_CFG`） |

### 13.2 B0.1 本机结构地板重标定（path-C 同网格、同 tile 形态）

既有 `probe_floor.py` 的 `read` 探针只建模 **path R**（整行单 tile），对 32768 宽的形状
不是生产形态。本轮新增 `_probe_floor_pathC.py`：**同一 block 网格 + 同一 `num_full × 8192`
列 tile 循环**，逐级切断指示链。所有数字来自 `profiles/round6/*` 的 30 次发射均值。

**32k `(32,32768)` — 32 block / 40 AIV，4×8192 tile，bm=1**

| probe（同一网格/同 tile 形态） | mean µs | median | min | max | aiv_time µs | vec_ratio | mte2_ratio | mte3_ratio | mte2 活跃带宽 GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `empty`（仅输出 store，无读无链） | **2.4780** | 2.4600 | 2.3200 | 2.6600 | 1.0287 | 0.0093 | 0.0027 | 0.4547 | — |
| `read_c`（4×GM→UB，**无归约无链**） | **4.6680** | 4.6600 | 4.5400 | 5.4400 | 3.2078 | 0.0025 | 0.7925 | 0.1345 | 49.02 |
| `ind_c_full`（生产链：copy+vcmp+vcast+reduce+vmax+epilogue） | **6.7587** | 6.7600 | 6.6000 | 6.8600 | 5.2288 | 0.4046 | 0.3793 | 0.0700 | 62.36 |
| 生产 current best（同 run A 臂，本轮 4 次） | 6.6427–6.7000 | 6.6670 | — | — | 5.19–5.23 | 0.404–0.407 | 0.377–0.387 | 0.047–0.056 | 61.5–62.7 |

`ind_c_full`（6.7587）与生产件同 run A 臂（6.667）、Phase A B 臂（6.7786）三者闭合
（差异在 ±1.3% 零效应噪声地板内）⇒ 探针**复现了生产形态**，可直接用于差额归因。

**死代码对照（列出但**不用于**归因）**：`red_c` 4.4960 / `vc_c` 4.6213 / `ind_c` 4.4980 /
`ind_c_1reduce` 4.6800 —— 四者都≈纯读地板，且 `vec_ratio` 0.0025–0.0917（远低于生产 0.4046）。
说明**结果未被使用的链被编译器 DCE 掉了**，这些数字只能再确认一次「纯读地板」，
**不能**用来反解归约单价。对策：改用「生产链整体 − 纯读地板」的差值（上下界都成立）。

**3d `(128,16384)` — 40 block × 4 task，2×8192 tile，bm=1**

| probe | mean µs | median | min | max | aiv_time | vec_ratio | mte2_ratio | mte3_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `empty` | **7.6300** | 7.6200 | 7.4600 | 8.0000 | 4.8436 | 0.0019 | 0.0011 | **0.9504** |
| `read_c`（2×8192） | **8.1420** | 8.1200 | 7.3600 | 8.7800 | 5.5743 | 0.0014 | 0.8710 | 0.5973 |
| `ind_c_full` | **10.9580** | 10.9800 | 10.5800 | 11.4400 | 8.2680 | 0.4143 | 0.3934 | 0.2100 |
| 生产 current best（Phase A B 臂） | 10.7620 | — | — | — | 8.087 | 0.4237 | 0.3943 | 0.2065 |

3d 的结构特性：**empty 地板 = 7.63 µs 已占 71%**，其 `mte3_ratio` 0.95 ⇒ 地板由
40 block × 4 task = **160 次 1 字节 GM store** 串行化主导（读只再加 0.51 µs）。
这解释了 3d 为什么对归约链改动不敏感，也说明它不是本轮的杠杆点（派单：3d 只需不回退）。

#### 头寸与派单目标的算术核验

- 32k 头寸 = `ind_c_full` − `read_c` = 6.7587 − 4.6680 = **2.0907 µs**；
  用同 run 生产臂 − 纯读地板 = 6.667 − 4.668 = 1.999 µs（两者同量级，取保守上界 2.09）。
  **2.09 / 4 tile = 0.52 µs/tile**，即整条链（三条全宽 pass + 微元合并 + epilogue）
  每 tile 只花 0.52 µs。
- 派单假设「4×~1.0 µs 归约 pass（合计 ~4 µs）」**被本机实测证伪**：4 个 tile 的**全部**
  非读开销只有 2.09 µs；归约只是 vcmp / vcast / reduce 三条全宽 pass 之一，
  其自身开销上界 = 2.09/3 ≈ **0.70 µs（≤10.3%）**。
  ⇒「把归约从 4 µs 降到 2 µs」需要不存在的 2 µs 预算；**目标本身不可达**。
- 门禁（派单 §2.B0.1.3）判定：头寸 2.09 µs ≥ 1.0 µs，**形式上不触发**「算术不可能」分支，
  故仍按流程做完了 reshape 候选（13.4，结论为回退）；但**目标数值**（-2 µs）已由上式排除。

#### 瓶颈判定（读受限 / 算力受限）

| 判据 | 32k 实测 | 解读 |
|---|---|---|
| 纯读地板 / 总时长 | 4.6680 / 6.7587 = **69%** | 读地板已是主体 |
| 纯读地板的有效带宽 | 4 MB / 4.668 µs ≈ **899 GB/s**（含发射） | 读侧已贴近 HBM 上限（§1 的 604 GB/s / 6.94 µs 旧常数**不适用本机**，实测快 1.5×） |
| `vec_ratio` / `mte2_ratio`（生产） | 0.4046 / 0.3793（aiv_time 5.23 of 6.76） | 两条 pipe 都 <41% ⇒ **没有一条饱和** |
| `ind_c_full` − `read_c` | +2.09 µs | 差额不是「算不动」，是 4 次迭代的 `copy→chain` 依赖链**没能藏进读里** |

⇒ **判定：读地板受限（HBM 已饱和）+ 剩余 31% 是 4 迭代流水的链暴露**；既不是纯算力受限，
也不是「归约 pass 太贵」。这一判定直接决定了候选方向（要削 pass 数或改善重叠，而不是「优化归约原语」）。

### 13.3 B0.2 `T.reshape` 支持性验证（先查文档，再跑最小探针）

**文档（AGENTS.md Docs Auto Routing，逐条引用）**

- `docs/Tilelang.language/shape操作/T.reshape.md`：签名 `T.reshape(src, dst)`；dtype 表
  int8/int16/int32/int64/fp16/fp32/bf16 = √，**bool = ×**；DEV 下降为
  `mlir::tensor::CollapseShapeOp` + `ExpandShapeOp`，EXPERT 为 `memref::ReinterpretCastOp`
  ⇒ 文档层面是 **view（不改存储顺序）**。
- `docs/Tilelang.language/规约操作/T.reduce_max.md`：`T.reduce_max(buffer, out, dim=-1, size=[], clear=True)`；
  dtype 表 int16/int32/int64/fp16/fp32 = √，**bf16 / bool-int1 = ×**。
- `docs/Tilelang.language/规约操作/T.reduce.md`：`T.reduce(src, dst, dims=[...], reduce_mode="max", clear=True)`；
  shape 规则为「rank 相同、仅一维不同且 dst 该维为 1」⇒ `(K,1) → (1,1)` 沿 `dims=[0]` **在规格内**。
- 前端实现核对：`tilelang/language/__init__.py:196` 把 `reshape` 重绑到
  `customize_npuir.npuir_reshape(src, dst)`（buffer→buffer intrinsic）；
  `src/target/codegen_npuir_dev.cc:3437 ReshapeCodegen` 生成 `tensor.reshape`
  （`collapse_shape`/`expand_shape`），MLIR tensor 语义下为 view。

**探针矩阵（`_probe_reshape_cases.py`，逐 case 独立编译 + 运行期数值校验 `max==7.0`）**

| case | src → dst | 归约顺序 | 结果 |
|---|---|---|---|
| `ref_1x8192_dim1` | (1,8192) | 单段 `dim=1`（生产参照） | **OK** 精确 |
| `s_1x8192_8x1024` | (1,8192) → (8,1024) shared | `dim=1` → `dim=0` | **OK** 精确 |
| `s_8192_8x1024` | (8192,) → (8,1024) | `dim=1` → `dim=0` | **OK** 精确 |
| `s_16384_16x1024` | (16384,) → (16,1024) | `dim=1` → `dim=0` | **OK** 精确 |
| `f_1x8192_8x1024` | **fragment** (1,8192) → (8,1024) | `dim=1` → `dim=0` | **OK** 精确 |
| `s_8192_2x4096` / `32x256` / `64x128` | K=2 / 32 / 64 | `dim=1` → `dim=0` | **OK** 精确（K 无结构限制） |
| `s_8192_8x1024_colrow` | (8,1024) | `dim=0` → `dim=1`（反序） | **OK** 精确 |
| `s_8192_12x682` | 12×682 = 8184 ≠ 8192 | — | **BUILD_FAIL** `error: 'tensor.expand_shape' op expected dimension 0 of collapsed type to be static value of 8184` |
| `s_8192_100x81` | 8100 ≠ 8192 | — | **BUILD_FAIL** 同上（`…static value of 8100`） |
| `s_8192_7x1170` | 8190 ≠ 8192 | — | **BUILD_FAIL** 同上（`…static value of 8190`） |
| `s_8192_8x1024_rowonly` | (8,1024) 仅 `dim=1`，再 `(1,1)→(1,)` 退化视图 | — | BUILD_FAIL `expected dimension 0 of collapsed type to be static value of 1`（**探针自身用错**：从 (1,1) 取 rank-1 退化视图非法；非 reshape 能力边界） |

**「是否引入额外 copy」的 IR 级证据**（`TILELANG_DUMP_IR=TRUE`，日志
`logs/round6/b0_2_dumpir_8x1024.log` 与 `b0_2_dumpir_ref.log`）：

```
// reshape 版（reshape 后两段归约）
%collapsed = tensor.collapse_shape %inserted_slice [[0, 1]] : tensor<1x8192xf32> into tensor<8192xf32>
%expanded  = tensor.expand_shape %collapsed [[0, 1]] output_shape [8, 1024] : tensor<8192xf32> into tensor<8x1024xf32>
%12 = hivm.hir.vreduce <max> ins(%expanded : tensor<8x1024xf32>) outs(%9 : tensor<8x1xf32>) reduce_dims = [1]
%13 = hivm.hir.vreduce <max> ins(%inserted_slice_1 : tensor<8x1xf32>) outs(%10 : tensor<1x1xf32>) reduce_dims = [0]
// 参照单段版
%10 = hivm.hir.vreduce <max> ins(%inserted_slice : tensor<1x8192xf32>) outs(%8 : tensor<1x1xf32>) reduce_dims = [1]
```

`collapse_shape`/`expand_shape` 之间**没有任何 `hivm.hir.vcopy` / `memref.copy` /
MTE 搬运**，直接喂给 `vreduce` ⇒ **reshape 不引入额外 UB 拷贝**（两版共有的那条
`memref.copy %subview, %alloc` 来自 `T.copy(x → x_ub)` 的 staging，不是 reshape 引入的）。

**B0.2 结论：`T.reshape` 可用**——(1,8192)/(8192,) → (8,1024)/(16,1024) 在 shared 与
fragment 上均可编译且逐位精确，可接 `reduce_max(dim=1)` 后再 `dim=0`，且不产生额外 copy。
**边界**：必须元素数守恒的整因子分解（非整除直接被 MLIR `expand_shape` 拒绝）。
**备选路径（本轮不需要）**：若 reshape 不可用，可把指示平面直接按 `(K, N/K)` 分配并用
逐行 `T.copy` 入位——代价是 K 次更窄的 DMA，按 §13.5 的 mte2 带宽证据（窄 tile 使活跃带宽
62.7→37.5 GB/s）会更慢，故不作为替代首选。

### 13.4 cand1 — path C 两段归约（`(K, 8192/K)` reshape + `reduce_max` ×2，K=8）

**改动（只动 path C 满块，单点）**：`_logical_reduce_kernel_round6_cand1_k8.py`
- 每个 8192 列 tile 内：`T.reshape(ind, ind2)`（`(1,8192)→(8,1024)`）→ `reduce_max(dim=1)`
  得 `(8,1)` → `reduce_max(dim=0)` 得 `(1,1)` → 并入 `vmax` running max。
- path R（4k）与静态尾块 `_build_tiled_tail` **未动**；
  `codegen_compare` 复核：`(32,4096)` 生成码 `a89563c7a0218861` **与 current best 逐字节相同**；
  32k/3d 如预期改变（`c701a208…`→`da27ca19…`；`9dbec92c…`→`6953d507…`）。

**精度（采纳前置）**：`python _logical_reduce_kernel_round6_cand1_k8.py --level all`
→ **exit 0，All check passed**（L0 27 + L1 25 + L2 7 + Boundary 13，`torch.equal` 逐位）。
新路径被显式覆盖：`L0-3a (32,32768)` num_full=4、`L0-3b (128,16384)` num_full=2、
`L0-3c (2048,16384)` num_full=2、`L0-3e (8,8448)` num_full=3 tail=0（两段归约）；
尾块陷阱 `L0-3d (64,9000)` tail=808、`L2 N=E_cap+1` tail=1 走**未改动的**旧尾块路径，均 PASS。
日志：`logs/round6/L0_cand1_k8.log`。

**同 run 交错 A/B（32k，4 对，tag `phaseB_cand1k8_32k`）**

| 对 | A mean (µs) | B mean (µs) | Δ% = (B−A)/A | 方向 |
|---|---|---|---|---|
| 1 | 6.6900 | 6.9527 | +3.93% | B 更慢 |
| 2 | 6.6427 | 6.8507 | +3.13% | B 更慢 |
| 3 | 6.6440 | 6.8780 | +3.52% | B 更慢 |
| 4 | 6.7000 | 6.8520 | +2.27% | B 更慢 |
| **中位** | **6.6670** | **6.8650** | **+2.97%** | — |

- 逐对 delta 中位 = **+3.33%**；方向一致性 = **4/4 同号（变慢）**；臂内极差 A 0.0573 / B 0.1020。
- 采纳门槛（32k 中位 ≤ −5% 且方向一致）**远未达到**，且方向为**回退**。

**候选 vs current best 对比表（§5 要求的四类指标）**

| 指标 | current best（A，4 run 均值） | cand1 K=8（B，4 run 均值） | 判读 |
|---|---|---|---|
| Task Duration(us) | 6.6670（中位） | 6.8650（中位） | **+2.97% 回退** |
| AICore 利用率（aiv `vec_ratio`） | 0.4059 | 0.4081 | 持平（+0.5%） |
| AIV 时间（aiv_time µs） | 5.208 | 5.397 | +3.6%（多出的 `vreduce(dim=0)` 与视图开销） |
| memory：mte2 活跃带宽 GB/s | 62.12 | 62.98 | 持平 |
| memory：mte2 / mte3 ratio | 0.3824 / 0.0522 | 0.3641 / 0.0596 | 读占比略降、写占比略升 |
| L0/L1/L2/Boundary | PASS | **PASS（逐位）** | 精度无风险，纯性能回退 |

**机制归因**：reshape 本身是 view（IR 已证无 copy），但两段归约把 **1 个 `vreduce` 变成 2 个**，
且第二段 `dim=0` 作用在 `(8,1)` 上（8 个元素的归约按行展开的固定开销），
每 tile 净增约 0.05 µs ×4 tile ≈ 0.2 µs。**元素数没变 ⇒ pass 数不可能减少**，
这正是 §13.2「瓶颈不是归约原语」的又一实证。

**裁决：不采纳（rollback）**。文件 `_logical_reduce_kernel_round6_cand1_k8.py`
（sha256 `4f17228df159…`）保留作证据，`_logical_reduce_kernel.py` **未覆盖**。
按 B1「K 只取 2–3 个值」的约束：回退机制（第二个 `vreduce` + 视图开销）**与 K 无关**
（K=2/32/64 在 §13.3 全部编译通过且结构相同），故不再扫 K，避免无效实验。

### 13.5 cand4 — 目标瓶颈导向：把列 tile 收窄到 4096（本轮自派生）

**为什么派生这个点**：§13.2 判定缺口是「4 迭代流水没把链藏进读里」（vec 0.40 / mte2 0.38，
两条 pipe 都不饱和）。唯一没在本任务里试过的方向是**收窄**列 tile（历史只试过加宽：
4×8192→2×16384 = −2.6%，见 §4）。收窄令 trip count 4→8、每个 buffer 减半，
给 auto multi-buffer 更多在飞 tile。改动=单点（`TUNED_TILE_CAP_MODE="narrow"` → cap=4096），
复用既有 factory 期开关，**未新增任何 Python 分派**；4k 生成码逐字节不变（`a89563c7…`）。

**同 run 交错 A/B（32k，2 对；tag `phaseB_cand4t4096_32k_pre`）**

| 对 | A mean (µs) | B mean (µs) | Δ% | 方向 |
|---|---|---|---|---|
| 1 | 6.6413 | 8.5960 | +29.43% | B 更慢 |
| 2 | 6.7053 | 8.5733 | +27.86% | B 更慢 |
| **中位** | **6.6733** | **8.5846** | **+28.64%** | — |

**候选 vs current best 对比表**

| 指标 | current best（A） | cand4 t=4096（B） | 判读 |
|---|---|---|---|
| Task Duration(us) | 6.6733 | 8.5846 | **+28.64% 回退（2/2 同向）** |
| AICore 利用率（vec_ratio） | 0.4064 | 0.3315 | −18%（链更碎） |
| AIV 时间 µs | 5.201 | 7.091 | +36% |
| memory：mte2 活跃带宽 GB/s | 62.59 | **37.52** | **−40%（窄 tile 打断 DMA 突发效率）** |
| memory：mte2 / mte3 ratio | 0.3803 / 0.0536 | 0.4615 / 0.0417 | 读占用时间比例上升 |
| L0 | PASS（未改动语义） | 未跑（性能先否决；纯 tiling 改动不改语义） | — |

**机制归因**：`mte2_bw` 从 62.6 掉到 37.5 GB/s ⇒ 4096 元素（16 KB）的列 tile 让
每次 DMA 突发变短、行 stride 分摊变差，读侧先崩；这是**读地板主导形态**的必然结果
（§13.2：纯读已占总时长 69%）。⇒ 「收窄 tile 换流水深度」在本形态上方向错误。

**裁决：不采纳（rollback）**。文件 `_logical_reduce_kernel_round6_cand4_t4096.py`
（sha256 `b157a4fcbbe4…`）保留作证据。

**停止判据**：cand1、cand4 **连续 2 个候选无改进**（且都有 4/4、2/2 同向证据）
⇒ 按派单 §4「连续 2 个候选无改进即停」终止候选实验（已用 2 候选 / 2 轮，预算 6 / 4）。

### 13.6 cand2 — 位平面归约：仅估算（未实现）

- 现状 pass 账（32k，每 tile）：`vcmp`(1) + `vcast`(1) + `reduce`(1) = **3 条全宽 pass**，
  实测整链 0.52 µs/tile。
- 位平面形态需要的算力：`vcast(bool→int32)`(1) + `vshl`(1，需要一张逐元素移位量平面) +
  按 32 元素/word 的 `reduce`(≥1) + 词级再归约 ⇒ **≥3 条全宽 pass + 一张额外索引平面**。
  `docs/Tilelang.language/逻辑操作/T.vshl.md`（int16/int32/int64 √，fp16/fp32/bool ×）与
  `T.vor.md`（int8…int64 √）、`docs/.../规约操作/T.reduce_max.md`（int16/int32/int64 √，
  **bool/int1 ×**）确认「可实现」，但**没有任何一步能减少 pass 数**。
- 该形态的理论下限 = 2 pass（`vcmp` + 一个 16 位载体上的 `reduce`），而这条路
  **已经在 round 1–3 实测过**：path C 换 f16 指示平面 → 32k **−1.8%**（低于 3% 阈值，未采纳，
  见 §4 与模块 docstring）。⇒ 位平面不可能优于 2 pass 形态。
- **估算净收益 < 3%（且需额外 UB 与索引平面）⇒ 不实现**，结论与估算留在本节。

### 13.7 cand3 — 谓词重写（`min|x|` / `min+max`）：实现前即否决

动机：用 2 条全宽 pass（`vabs`+`reduce_min`，或 `reduce_min`+`reduce_max`）替掉 3 条，
理论 −0.5 µs（−7.7%）。**先用探针验证语义等价性**（`_probe_nan_minmax.py`，f32 单行 8 元素）：

| 行 | reduce_min | reduce_max | torch `all(x!=0)` | `(min==0)|(max==0)` 判「有 0」 | 是否一致 |
|---|---:|---:|---|---|---|
| `[NaN, 0, 1…6]` | **NaN** | **NaN** | False | False（判无 0） | **✗ 误判** |
| `[NaN, 3, 1…6]` | NaN | NaN | True | False（判无 0） | ✓ |
| `[NaN]×8` | NaN | NaN | True | False（判无 0） | ✓ |
| `[0, NaN, 1…6]` | NaN | NaN | False | False（判无 0） | **✗ 误判** |
| `[-1, 0, 1…6]` | −1.0 | 6.0 | False | False（判无 0） | **✗ 漏检中间 0** |
| `[-1, -0.0, 1…6]` | −1.0 | 6.0 | False | False（判无 0） | **✗ 漏检 −0.0** |
| `[1…8]` | 1.0 | 8.0 | True | False（判无 0） | ✓ |
| `[-2, -1, 1…6]` | −2.0 | 6.0 | True | False（判无 0） | ✓ |

两条独立否决理由：
1. **本机 `reduce_min`/`reduce_max` 传播 NaN**（任一元素为 NaN ⇒ 输出 NaN）：
   「NaN 与 0 共存」的行被判成「无 0」→ `all=True`，而契约（torch / 现网 ordered-EQ 链）是 `False`。
   `vabs`+`reduce_min` 同样传播 NaN（`min|NaN,0| = NaN`），修不了这一条。
2. **极值法漏检中间 0**：`[-1, 0, 6]` 的 min/max 都不是 0（仅 `vabs`+`reduce_min` 能修）。

⇒ 语义不等价，且 `[NaN, 0]` 类行落在既有 L0/Boundary 覆盖面内。
**不实现**（连候选文件都不生成，避免把语义错误的实现放进证据链）。
现网 `vcmp(eq, ordered) → vcast → reduce_max` 链是**唯一已知的 NaN 正确形态**。

### 13.8 B2 — 跨核 N-split + merge：仅估算（未实现）

结构（参考 `PL-1.20-nsplit-perrow-partial`，**stale，已按本机实测重估**）：把 32 列的 N
一分为二 → 64 block（>40 AIV ⇒ 2 波），每 block 出 partial，再由第二个 kernel / GM atomic
合并。用本机实测数代入：

| 项 | 本机实测/推导 | 影响 |
|---|---|---|
| 当前并行度 | 32 block / 40 AIV（8 核空闲，1 波） | 理论上限 +25% 核数 |
| 切 N 后 block 数 | 64 → `ceil(64/40) = 2` 波 | 每 block 只做 2 tile，2 波 × 2 tile = **4 tile-time = 现状 1 波 × 4 tile**，**调度收益为 0** |
| 读侧带宽 | 纯读地板 4.67 µs ≈ **899 GB/s**（已贴近 HBM 上限） | 加核**不增加**聚合带宽 ⇒ 读地板不会随 40 核下降 |
| 发射地板 | empty 实测：16 block 1.845（§4 记录）/ 32 block **2.478**（本轮） | 64 block 地板更高（外推 ≈2.6–3.0，未实测） |
| merge 成本 | 第二 kernel 的发射地板 ≥ **1.845–2.478 µs**；或 GM atomic + 初始化 pass（同量级） | ≥ 27%–37% 的当前总时长 |
| **净收益** | 收益 ≤ 0（2 波抵消 + 读已 HBM 饱和），成本 ≥ +1.8 µs | **≤ −27%** |

**判据**：派单要求「净收益 < 2×噪声地板（2.6%）或新增发射/merge 成本 ≥ 收益 ⇒ 禁止实现」。
本估算净收益 **≤ 0** 且 merge 成本（≥1.8 µs）**远大于**任何可能的收益 ⇒ **不实现**。

### 13.9 采纳 / 回滚与最终工件

| 项 | 状态 |
|---|---|
| `perf_opt/_logical_reduce_kernel.py` | **未改动**，sha256 `70ecea6df2cf0dc9ce75c77b043303c91e6e04e78357507009347302a356de5b`（= Phase A 的 current best `final_f16res`） |
| 保留的证据候选 | `_logical_reduce_kernel_round6_cand1_k8.py`（`4f17228d…`）、`_logical_reduce_kernel_round6_cand4_t4096.py`（`b157a4fc…`） |
| 新增探针（非候选） | `_probe_floor_pathC.py`、`_probe_reshape_cases.py`、`_probe_nan_minmax.py` |
| 新增生成器/工具 | `make_round6_cand1.py`、`make_round6_cand4.py`、`append_round6_records.py`；`run_bench.py` / `bench_driver.py` 仅**追加** `--probe` 取值（向后兼容，默认路径未变） |
| 只读工件 | `../_logical_reduce_kernel.py`、wrapper、`tests/`、`benchmarks/`、`_logical_reduce_kernel_DESIGN.md` 均未触碰 |
| 编排层 | 未调用 statectl，未写 `.stage_state.json` |

### 13.10 台账变更（append-only）

- `perf_records.jsonl`：**未改动**（27 行）。本轮无采纳 ⇒ 无新 final 行、旧 final 行无需转存。
- `perf_records_full_history.jsonl`：**+22 行**（56 → 78；前 56 行 sha256 校验逐字节不变）：
  - `phase="round6_calibration"` × 10：32k × {empty, read_c, ind_c_full, red_c, vc_c, ind_c,
    ind_c_1reduce} + 3d × {empty, read_c, ind_c_full}，含 `probe`、四类 pipe 指标、
    `dce_note`（死代码标注）、`msprof_raw_path`。
  - `phase="round6_candidate"` × 12：cand1 32k 4 对 × 2 臂、cand4 32k 2 对 × 2 臂，
    含 `tag`/`pair`/`arm`/`artifact_sha256`/`l0_pass`/`ab_verdict_note`。
- raw profile：`profiles/round6/*`（探针）、`profiles/ab/phaseB_cand1k8_32k/*`、
  `profiles/ab/phaseB_cand4t4096_32k_pre/*`。

### 13.11 未验证项 / 风险

1. **K 未扫**（只测 K=8）：拒绝机制（第二个 `vreduce` + 视图开销）与 K 无关，故未扫 2/16/32；
   若后续工具链改了 `vreduce` 的多行 lowering，需重测。
2. **cand4 只跑 2 对**：+28.64%（2/2 同向、幅度 28%）远超噪声地板，2 对足够定性否决；
   未做 4k/3d 无回退核对（4k 生成码逐字节相同，3d 因否决而未测）。
3. **`empty` 地板随 block 数的斜率**只有 16/32 两个实测点（1.845 / 2.478），64 block
   （B2 估算用）为外推值，未实测；但 B2 的结论不依赖该外推（2 波抵消 + merge ≥1.8 µs）。
4. **3d 的 7.63 µs empty 地板**由 160 次 1 字节 store 主导（mte3 0.95）。这是 3d 的
   独立优化口（例如把行→block 映射改成连续区间以合并 store），**本轮未做**：
   派单要求 3d 只作不回退核对，且该改动不影响 32k 主战场。
5. 历史常数（604 GB/s、6.94 µs 纯读、~1.0 µs/归约 pass、`CONST-msprof-task-floor-910B3` 等）
   **在本机全部不成立**，本轮已用实测覆盖；后续引用必须以本节数字为准（器件/工具链已变）。

### 13.12 最终工件精度复检（gate 要求）

| 项 | 结果 |
|---|---|
| 命令 | `python perf_opt/_logical_reduce_kernel.py --level all`（cwd = `perf_opt/`） |
| 工件 | sha256 `70ecea6df2cf0dc9ce75c77b043303c91e6e04e78357507009347302a356de5b`（本轮未改动） |
| 结果 | **exit 0**，`[L0] ALL PASS: 27`、`[L1] ALL PASS: 25`、`[L2]` 全 PASS、`[Boundary]` 全 PASS、`All check passed!` |
| 日志 | `logs/round6/L0_current_best_recheck.log` |
| 隔离时间窗 | 2026-10-20T03:34Z–03:40Z，单进程，无并发 msprof |
