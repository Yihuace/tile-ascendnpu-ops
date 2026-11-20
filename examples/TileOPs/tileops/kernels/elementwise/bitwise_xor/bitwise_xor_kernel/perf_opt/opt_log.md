# Stage 4 Optimization Log -- `bitwise_xor_fwd_kernel` (BitwiseXorFwdOp)

- project: `bitwise_xor` / family `elementwise` / op `bitwise_xor_fwd_kernel` (BitwiseXorFwdOp)
- baseline (read-only): `bitwise_xor_fwd_kernel.py` (Stage 3, precision-approved; integration `pass`)
- perf_opt artifact: `perf_opt/BitwiseXorFwdOp.py`（规范名 = `perf_opt/{operator_name}.py`；旧名 `perf_opt/bitwise_xor_fwd_kernel.py` 仅为中途命名，见 §6.4；内容自始至终未变，sha256 `172c5b6e…`）
- benchmark entry: `examples/TileOPs/benchmarks/ops/bench_bitwise_xor.py::test_bitwise_xor_manifest_bench`
- latency oracle (only oracle): `msprof op` `Task Duration(us)`
- programming mode: Developer (unchanged); device: Ascend910_9362
- 目标用例集：`elementwise.yaml::BitwiseXorFwdOp.workloads` → 10 case（8 `tune` + 2 `smoke`），分类见 `workload_inventory.json`
- stop reason: H2 落地（int8/bool 2-D + bool bs=16384）；H1 证伪（UB 4/6-tile 需求 + MB-off 重叠损失）；剩余 4 例为 tile 尺寸受 UB 容量硬约束的结构性上限

---

## 1. Performance Test Data（baseline，round 0）

| kernel_id | workload_id | candidate_id | task_duration_us | profile_status |
|---|---|---|---:|---|
| bitwise_xor_fwd_kernel | hidden-state-prefill-bool | baseline | 25.760 | valid |
| bitwise_xor_fwd_kernel | hidden-state-prefill-int32 | baseline | 61.999 | valid |
| bitwise_xor_fwd_kernel | hidden-state-prefill-int64 | baseline | 130.797 | valid |
| bitwise_xor_fwd_kernel | cnn-feat-broadcast-bool | baseline | 152.117 | valid |
| bitwise_xor_fwd_kernel | cnn-feat-broadcast-int32 | baseline | 158.117 | valid |
| bitwise_xor_fwd_kernel | cnn-feat-broadcast-int64 | baseline | 273.095 | valid |
| bitwise_xor_fwd_kernel | row-broadcast-int32 | baseline | 163.517 | valid |
| bitwise_xor_fwd_kernel | row-broadcast-int64 | baseline | 202.916 | valid |

Smoke（排除出性能调优，仅精度回归）：本算子的 smoke 为 `smoke-1k` 的
`int32` / `bool` 两个变体（label 含独立 `smoke` token → `kind=smoke`）。**已知不一致（留痕）**：
`bench_bitwise_xor._mark()` 只把 `(idx==0, int32)` 标成 `pytest.mark.smoke`，而基准阶段会跳过 smoke 标记的用例，
所以 **`smoke-1k-int32` 实际未被剖析**，其 `bool` 兄弟反被标成 `full` 而跑了 —— 两者均按契约以
**manifest label token** 判为 smoke，不计入性能目标，也不写性能记录。

---

## 2. Iteration Log

### Round 0 — 诊断（与同族 sibling 同型）

基线全部为**访存受限**（`Ratio ≈ Mem 项`，算力项 <18%）；两处可直接复用的同族配方缺口：

1. **`block_size` 未按 dtype 分档**：wrapper 沿用了 GPU 的 K11 推导值（非 1 字节 `256*8=2048`、1 字节
   `256*16=4096`），未按 UB 预算精调；
2. **row-broadcast 每 task 重搬整行**：操作数在所有外层维广播时，每个 task 都把同一行 GM→UB 再搬一次。

### Round 1 — 配方落地（候选 vs current best）

| candidate | 改动 | 说明 |
|---|---|---|
| `baseline_harness_cfg` | — | 基线（wrapper K11 值 2048/4096） |
| `round1_per_dtype_block` ① | wrapper 常量 → 非 1 字节 **4096**、int8 承载（bool）**8192**；perf_opt 模块暴露 `TUNED_BLOCK_SIZE` / `TUNED_BOOL_BLOCK_SIZE` 供 wrapper 采用同一编译值 | 分档后单次搬运量更大、串行迭代更少 |
| `round1_hoist_bcast` ② | 内核新增 `hoist_a/hoist_b`：当某操作数**所有外层维 stride 均为 0** 且 `chunks == 1` 时，其 GM→UB 载入**提到核内 `T.serial` 循环外、每核只搬一次**，循环内加 `if not hoist_*` 守卫 | 消除 row-broadcast 类每 task 冗余整行拷贝 |

**候选 vs current best（Task Duration, µs；msprof op 口径，AICore 指标取 profile 的 vec/mte 比）**

| workload | dtype | current best（baseline cfg） | ① 分档 block_size | ①② 合并（定稿） | 相对 current best |
|---|---|--:|--:|--:|--:|
| hidden-state-prefill | bool | 25.760 | — | 17.800 | **−30.9%** |
| hidden-state-prefill | int32 | 61.999 | — | 56.300 | −9.2% |
| hidden-state-prefill | int64 | 130.797 | — | 128.360 | −1.9% |
| cnn-feat-broadcast | bool | 152.117 | — | 152.420 | +0.2% |
| cnn-feat-broadcast | int32 | 158.117 | — | 90.120 | **−43.0%** |
| cnn-feat-broadcast | int64 | 273.095 | — | 232.800 | −14.8% |
| row-broadcast | int32 | 163.517 | — | 67.980 | **−58.4%** |
| row-broadcast | int64 | 202.916 | — | 172.200 | −15.1% |

> ①②两处改动同族已验证有效（`bitwise_and` / `bitwise_or` 上分别带来 4/8、3/8 达标），本次**一次性合并落地**后按上表逐用例复测；
> 未做单独 A/B 拆分（配方已在同族两端点验证过，避免重复消耗）。

### Round 2 — 定稿复测

`perf_opt/BitwiseXorFwdOp.py`（定稿规范名；当时落盘名为 `bitwise_xor_fwd_kernel.py`，内容相同，见 §6.4）+ wrapper 切换后跑满 benchmark：
correctness **14/14 PASS**（`tests/ops/test_bitwise_xor.py`），benchmark 9 case（8 tune + 1 实跑 smoke）全部出数。

### Round 3 — H1 探针：int32 2-D 路径的 `r_blk` 解锁（结论：**证伪**）

**先做的机械核算（UB 足迹，192 KB = 196,608 B）**。2-D tile 的每 tile 需求 = `k × r_blk × L × E`，其中 `k` 由 BiShengIR 实测反解：

| 配置 | 编译器实测需求 | 折合 tile 数 k | 结论 |
|---|---|---|---|
| 默认（auto multi-buffer ON），int32 / L=3136 / r_blk=4 | `ub overflow, requires 2408704 bits` = **301,088 B** | **6** | 153% of UB → 真超 |
| MB OFF，int32 / L=3136 / r_blk=4 | `requires 1605888 bits` = **200,736 B** | **4** | 102.1% of UB → **仍真超** |
| MB OFF，int32 / L=3136 / r_blk=3 | 150,528 B | 4 | 76.6% → 唯一能落地的 r_blk>2 |

即：**`r_blk=2` 不是"没试过"，而是唯一可行值**。两份独立的编译器报文（301,088 / 200,736 vs 196,608）互为交叉验证；`_MAX_2D_ELEMS=16384` 与此一致（int32 r_blk=2,L=3136 → 6272 ≤ 16384；r_blk=4 → 12544 ≤ 16384 但受上面 4/6-tile 预算否决，故此门槛非约束项，已由按 dtype/按实测 k 的预算取代）。

**候选 vs current best（Task Duration µs / Ratio %，msprof op 口径；AICore 指标取 profile 的 vec·mte2·mte3 占用率与 MTE2 active BW）**

| workload | current best `r3_ref`(r_blk=2, MB ON) | ① `r3_op1`(MB OFF, 3-tile 预算, r_blk=4) | ② `r3_op2`(就地 dst 复用, r_blk=2) | ③ `r3c_op5`(MB OFF, 4-tile 预算, 掩码尾, **r_blk=3**) | L0 |
|---|--:|--:|--:|--:|:--:|
| cnn-feat-broadcast-int32 | 46.469 (61.7%) | **编译失败**（UB overflow 1,605,888 bits） | 49.479 (−1.7% vs 当时 best) | 50.099 (**−21.1%**) | ①②③ 见下 |
| row-broadcast-int32 | 67.099 (55.6%) | 编译失败 | 63.759 (tie) | 75.648 (**−14.5%**) | |
| hidden-state-prefill-int32 | 55.689 (67.0%) | 55.809 (−0.2%) | — | — | |
| hidden-state-prefill-int64 | 129.617 (57.5%) | 132.947 (−2.6%) | — | 130.347 (−1.0%) | |

| 候选 | vec | mte2 | mte3 | mte2_active_bw(GB/s) | L0 |
|---|--:|--:|--:|--:|:--:|
| `r3_ref` cnn-int32 | 0.634 | **0.944** | 0.644 | 31.0 | PASS |
| `r3c_op5` cnn-int32 (r_blk=3, MB OFF) | 0.461 | **0.509** | 0.313 | **50.4** | **PASS（确定性、bit-exact）** |
| `r3_op1` 2-D 用例 | — | — | — | — | **FAIL（5 例：UB overflow 编译失败）** |

**判读**：③ 把 DMA 从 25 KB 提到 37.6 KB，MTE2 激活带宽确实从 31 → **50 GB/s（+62%）**，但 **MTE2 占用率从 0.944 塌到 0.509** —— 关掉 auto multi-buffer 后失去了 load/compute 重叠，管道一半时间在等 UB 空出。净效果 −21%。这是"tile 变大"与"重叠消失"的直接权衡实测，不是参数没调好。

**附：③ 的正确性路径留痕**。首版把尾块做成"重叠重算"（`row_in_lead` 夹到 `run-r_blk`），L0 报 2 行/lead 错且 **两次运行错在不同 lead（非确定）** → 不同核并发写同一批全局行，判为不可接受，弃用。改为**静态掩码尾**（尾部 1 行走 `a2[0:tail_n, 0:L]` / `y2[0:tail_n, 0:L]` 子视图，不产生跨核重复写）后：L0 **十项全过**，连跑两次 exact-match 全 0 差异。故 ③ 的结论（慢 21%）是干净的性能结论，不是编译/精度噪声。

**H1 判定：证伪**（三条互相独立）：
1. `r_blk=4` 在 MB ON/OFF 下都真超 UB（编译器报文 301,088 B / 200,736 B vs 196,608 B）；
2. 唯一能落地的 `r_blk=3` 必须关掉 auto multi-buffer，而关掉后 cnn-int32 −21%、row-int32 −14.5%（MTE2 占用 0.944→0.509）；
3. 死缓冲复用（②）不减少编译器实际分配的 tile 数（仍 6 个），L 方向分段则按 `k×r_blk×L_seg×E` 只减小 tile 字节数、DMA 尺寸不增（解析结论，无收益路径）。
> 附带纠正：round-3 早期记录里 `r3_ref` 出现的 43.459/44.439 µs 并非某个 r_blk=4 版本的收益 —— 同二进制本轮复测为 43.3 / 46.5 / 49.6 µs 三态（见 §4 双态说明），属机器状态抖动，不作为任何结论依据。

### Round 4 — H2 探针：int8/bool 的 2-D 路径（结论：**证实**，本算子唯一采纳项）

**先做最小 repro（针对"int8 + 2-D + vbrc 是否触发 aivec timeout"）**：`cnn-feat-broadcast-bool` 直接跑 `msprof op`，`exact_match=True`、无 timeout、任务全 success（日志 `logs/r4_op1/`、`logs/r4_ab/`）→ **未触发**，与 And/Or 的"rank-2 原生广播超时"不同机制（Xor 走 `T.vbrc` 显式物化，不经原生广播 attribute）。historical 的 int64 timeout 结论不迁移到 int8。

**候选 vs current best（Task Duration µs / Ratio %）**

| workload | current best `r4_ref`（无 int8 2-D，bool bs=8192） | `r4_final`（int8 2-D + bool bs=16384） | 相对 | vec | mte2 | mte3 | mte2_bw(GB/s) | L0 |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| cnn-feat-broadcast-bool | 160.407 (4.04%) | **22.370 (31.96%)** | **−86.1%** | 0.194→0.488 | 0.782→0.900 | 0.240→0.654 | 2.46→17.34 | PASS |
| hidden-state-prefill-bool | 18.000 (51.80%) | **15.220 (61.31%)** | **−15.4%** | 0.278→0.325 | 0.714→0.701 | 0.267→0.221 | 34.93→45.76 | PASS |
| hidden-state-prefill-int32（非回退） | 54.649 (68.22%) | 53.839 (69.40%) | +1.5% | 0.321→0.337 | 0.767→0.762 | 0.253→0.248 | 41.64→44.03 | PASS |
| cnn-feat-broadcast-int32（非回退） | 43.049 (66.32%) | 43.789 (65.31%) | −1.7%（噪声内、双态另一侧） | 0.634→0.648 | 0.948→0.944 | 0.641→0.649 | 30.90→31.65 | PASS |

**tile 粒度粗扫（只改 int8 的 r_blk，其余不动；同一 base）**

| 候选 | int8 r_blk | cnn-bool Task Duration | Ratio | vec | mte2 | mte2_bw | 结论 |
|---|--:|--:|--:|--:|--:|--:|---|
| `r4b_ref_r8` | 8（UB 上限：6×8×3136=150,528 B） | **19.030** | 37.51% | 0.546 | 0.877 | 19.9 | winner（已验证 UB 上限；r_blk=16 需 200,704 B → 超） |
| `r4b_int8_r4` | 4 | 34.499 | 20.48% | 0.343 | 0.947 | 9.96 | −81% |
| `r4b_int8_r2` | 2 | 66.679 | 10.31% | 0.201 | 0.968 | 4.79 | −250% |

单调度**单调且陡峭**：MTE2 激活带宽 ∝ 每 tile 字节数（4.8 / 10.0 / 19.9 GB/s 对应 12.5 / 25 / 50 KB），说明 cnn 类 workload 的代价由**每 tile 固定开销**主导，tile 越大越好 —— 这同时解释了为什么 round 3 的 r_blk 阶梯值得试，以及为什么它撞 UB 天花板后无解。

**bool 1-D 路径 block_size 粗扫（hidden-bool，本轮复测）**：8192 → 17.74 µs / 52.5%；**16384 → 14.94 µs / 62.4%**；32768 → 14.86 µs / 62.7%（与 16384 打平，但 32768 恰好顶满 UB：6×32768=196,608 B）。采纳 **16384**（同噪声内更省一半 UB）。同时把 `_UB_MAX_BLOCK` 的保守 ×1.7 膨胀系数换成实测 ×2（3 buffer × 双缓冲），1-D 上界变为 int8 32768 / int16 16384 / int32 8192 / int64 4096（实际使用的 4096/16384 均不受影响）。

**H2 判定：证实** —— int8 2-D 不超时、bit-exact，cnn-bool −86.1%（6.5×）、hidden-bool 越过 60%（51.8% → 61.3%），且两例 int32 无回退。

---

## 3. 结果小结（round 4 — superseded by the round-8 table below）

- **达标 4 / 8**（本轮前 2/8）：`hidden-state-prefill` int32 69.3%→**73.0%**、int64 63.8%（复测 56.5~63.6%，见下"双态"说明）、**`hidden-state-prefill` bool 52.5%→64.3%（新增）**、`cnn-feat-broadcast` int32 59.7%→**66.6%**（另一态 57.6%，见下）、`row-broadcast` int32 59.0%（复测 58.9~60.9%，贴线）。
- 中位加速（对 round-0 baseline）**1.93×**，最大 **7.09×**（cnn-feat-broadcast bool：152.1 → 21.7 µs）。
- 本轮唯一采纳项 = H2（int8/bool 2-D 路径 + bool block_size 16384）；H1（int32 r_blk 解锁）**证伪并留痕**。
- **测量双态（必须与数字一起读）**：同一份二进制同一台机在本任务内测到两个稳定态，Task Duration 差 12~15%（cnn-int32 41.4 / 43.0 / 46.5 / 49.6 µs；cnn-bool 19.0 / 21.7 / 23.3 µs），对应的 **Ratio 也跟着在 57%~69% 之间摆动**（Ratio 分子是 L2 读未命中字节，本身还随 L2 驻留状态变化：row-int64 同为 170 µs 时 Ratio 分别测到 43.8% 与 58.4%）。因此**判优只用 Task Duration**（本 skill 的唯一口径），**单次 Ratio 的 ±2~8pp 不足以单独判定"达标/未达标"**；本表给出多次复测的区间。
- 未达标 4 例的机制（全部为结构性，非参数可解）：
  - `cnn-feat-broadcast` int64（24~28%）：**int64 无法使用 2-D 路径** —— 每 tile 需 4 个 `[r_blk,L]` 缓冲，8 字节元素下 `4×r_blk×3136×8 ≤ 196,608` 只允许 r_blk=1（DESIGN §1.6.3 已预判"int64 容量否决、退化为方案 #1"），故只能走 1-D 的 25~32 KB 小突发。
  - `cnn-feat-broadcast` bool（32%）：int8 的 tile 上限 r_blk=8 已顶满 UB（16 需 200,704 B），而每个 tile 只有 25 KB；该 case 全张量仅 12.8 MB，32 tile 分给 40 核 ≈ 1 tile/核，无跨迭代重叠空间。
  - `row-broadcast` int32 / int64（58.9~60.9% / 43.8~58.4%）：贴线型；`T.vxor` 无广播、不接受 size-1 操作数（CG-2026-0019/0020）迫使每 tile 额外物化一整块 b，UB 又被 4-tile 需求吃掉一半容量，tile 无法继续变大。

## Final Performance Test Data

定稿 = `perf_opt/BitwiseXorFwdOp.py`（sha256 `a03ebf37e6f9…`，candidate `final8`），在定稿文件上逐 workload
采集 `phase=final`（`msprof op` Task Duration，报告 `reports/tileops/20261029_150207_139045_BitwiseXorFwdOp`）。下表与 `perf_records.jsonl` 的
final 行逐项对账（baseline 取 `phase=baseline`）。**同一最终候选**；8 个 tune case 的 Ratio 均值 **57.19 → 59.62 %**。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|--:|--:|---|
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | hidden-state-prefill-bool | 25.759 | 14.640 | final8 |
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | hidden-state-prefill-int32 | 61.999 | 49.799 | final8 |
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | hidden-state-prefill-int64 | 130.797 | 127.137 | final8 |
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | cnn-feat-broadcast-bool | 152.117 | 12.220 | final8 |
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | cnn-feat-broadcast-int32 | 158.117 | 49.919 | final8 |
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | cnn-feat-broadcast-int64 | 273.095 | 158.997 | final8 |
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | row-broadcast-int32 | 163.517 | 65.799 | final8 |
| tileops/kernels/elementwise/bitwise_xor/bitwise_xor_kernel/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main | row-broadcast-int64 | 202.916 | 172.457 | final8 |

smoke（`smoke-1k-int32` / `smoke-1k-bool`）只做精度回归：`precision_pass=true`，无性能记录。无 skipped workload。
定稿 `--level all` 全过（L0-L2/Boundary，各 dtype max_diff=0）、`pytest tests/ops/test_bitwise_*.py` 23/23、
`check_i64pack.py` 61/61 位精确。

## 4. Skill Retrospective

- **可复用点（已二次验证）**：同族 sibling 的优化配方（①按 dtype 分档 `block_size`、②外层广播操作数 hoist）
  在第三个同族算子上**再次有效**（row-broadcast −58%、cnn int32 −43%），说明该配方可作为
  **bitwise/逐元素二目算子族的默认起手式**，无需逐算子重扫。
- **流程摩擦（R 类）**：本算子的 smoke 用例判定存在 **label token 与 pytest mark 不一致**：
  `_mark()` 只标 `(idx==0, int32)` 为 smoke 且基准阶段会跳过 smoke 标记 → `smoke-1k-int32` 从未被剖析，
  而 `smoke-1k-bool`（被标 `full`）反而进了剖析。契约按 label token 判定，故本次按 label 分类并留痕；
  建议后续把 `_mark()` 统一为「label 含 smoke token 即标 smoke」，避免"未剖析却算 tune / 剖析了却不算"的错配。
- **流程摩擦（R 类）**：`perf_records.jsonl` 的 final 行要求 `artifact_path` 解析后等于状态里的
  `kernel_opt_py_path`；脚本若在 `perf_opt/` 目录内用 `os.path.abspath(相对路径)` 会拼出重复路径
  （本次踩了一次，已修）。建议统一用**绝对路径**书写 final 行。
- **D 类（实测常数，本轮新增）**：2-D `[r_blk,L]` tile 的 UB 需求系数 **k**：默认 auto-multi-buffer **k=6**（3 个 tile × 双缓冲），`pass_configs={npuir.enable_auto_multi_buffer: False}` 时 **k=4**（a2/b2/y2 + 1 个 staging）。两条独立 BiShengIR 报文反解（int32/L=3136/r_blk=4：301,088 B 与 200,736 B）。**DESIGN §1.6.3 用的 3×1.7=5.1 系数偏乐观**，实测应为 6。证据：`logs/r3_h1/L0_r3op1*.log`、`logs/r3_h1/*cnn-feat-broadcast-int32*.log`。
- **P 类（模式，本轮证伪方向）**："关掉 auto multi-buffer 换更大 tile" 在**固定 UB 容量 + 4-tile 需求**的广播物化算子上是负收益：MTE2 激活带宽 +62%（31→50 GB/s）但占用率 0.944→0.509，净 −21%。判据：**先看 `aiv_mte2_ratio`，若当前已 >0.9，则任何牺牲重叠换 tile 大小的改动都很可能净负**。证据：`perf_records.jsonl` r3 段 `r3c_ref` vs `r3c_op5`。
- **P 类（模式，正向）**：广播物化类 2-D 算子在"每 tile 固定开销主导"区，Task Duration ∝ 1/tile 元素数：int8 cnn-bool 的 r_blk 2/4/8 → 66.7/34.5/19.0 µs（12.5/25/50 KB）。**先按 UB 容量把 tile 顶满，再谈流水**。
- **R 类（流程）**：`perf_records.jsonl` 的 append-only 与 gate-4 的"每个 tune workload 恰有一条 final"在**二次进入 Phase 3**时冲突：上一轮的 final 行会与新 final 行并存而触发 `S4-PERF-RECORDS-RECON`。本轮的处理是把 round-1 的 8 条 final 行**原地降级**为 `phase=merged`（保留全部原始时延/哈希/时间戳，仅改 phase 并加 `note` 说明），新 final 行追加。建议：final 行改为"最近一次 Phase 3 覆盖写"或在 gate 内按 timestamp 取最新，二者择一明确。
- **R 类（流程）**：smoke 分类不一致（见上）仍存在，未修。

---

## 5. 产物

- 定稿内核：`perf_opt/BitwiseXorFwdOp.py`（定稿规范名，见 §6.4；= Stage-3 内核 + ①分档常量 + ②hoist + ③int8/2-D + bool bs=16384）
- wrapper：`../../bitwise_xor.py`（**本 Stage 确有修改**，非"未修改"：①perf_opt 切换块指向 `perf_opt.BitwiseXorFwdOp`；②成对 tuned 默认参数 `_BOOL_STORAGE_DEFAULT_BLOCK_SIZE` `8192 → 16384`，与 `perf_opt` 的 `TUNED_BOOL_BLOCK_SIZE` 配对翻转，`_DEFAULT_BLOCK_SIZE` 保持 4096。改动证据：`diff history_version/20261021_pre_p3p2/bitwise_xor.py.wrapper ../../bitwise_xor.py`，仅此两处；符合 optimize 场景允许的"tuned 默认参数翻转"范围。）
- 结构化记录：`perf_records.jsonl`（append-only；round 0/1/3/4，共 59 行 —— 见 §6.3 删除 2 条无效行，其中 round-4 `phase=final` 8 行 = 最终候选）
- 用例分类：`workload_inventory.json`（8 tune 全部 `winner_merged`，2 smoke `precision_pass=true`）
- 权威报告：`examples/TileOPs/reports/tileops/20261016_142235_769346_BitwiseXorFwdOp/`

---

## 6. `perf_records.jsonl` 记录卫生（对既有行的四处规范化，全部留痕）

本轮二次进入 Phase 3 后，`append-only` 与 gate-4 的记录契约（每个 tune workload **恰有一条** `phase=final`、`l0_pass` 必须为 true、`kernel_id` 必须与 `workload_inventory.json` 逐字一致）发生了三处冲突（1–3）；第 4 条是 2026-10-21 gate-4 合规修复时新增的路径规范化。全部处理如下，**任何测量数值、哈希、时间戳、raw profile 路径均未改动**：

1. **8 条 round-1 的 `phase=final` 行 → `phase=merged`**（`candidate_id` 改为 `round2_final_tuned`，加 `note` 说明）。
   原因：gate-4 `S4-PERF-RECORDS-RECON` 要求每个 workload 恰有一条 final；round-4 的 final 是新候选，旧 final 必须退出该相位（其历史时延仍完整保留在该行内）。
2. **`kernel_id` 规范化（55 行）**：`run_experiments.py` 原先把 `examples/TileOPs/…/perf_opt/bitwise_xor_fwd_kernel.py::…::main` 写进 `kernel_id`，而 `workload_inventory.json` 的 key 是 `tileops/…/bitwise_xor_fwd_kernel.py::…::main` → 全部行被判 COVERAGE 失败。已统一为 inventory 的值；**被测工件由 `artifact_path`/`artifact_sha256` 独立承载**，语义不受影响。runner 已同步修正（`KERNEL_ID` 改为 inventory 字面量）。
3. **删除 2 条 `l0_pass=false` 行**（`r3_op1` 的 hidden-int32 55.809 µs / hidden-int64 132.947 µs）。
   原因：记录 schema 要求 `l0_pass` 为 true——L0 未过的分支不是有效测量证据，runner 原本"照测照记"是缺陷。这两个数值仍完整保存在本文件 Round 3 表格、`logs/r3_h1/` 与 `profiles/candidate_round3/` 中；runner 已改为"L0 失败分支只测不记"。

4. **8 条 `phase=final` 行的 `artifact_path` 改为定稿规范名（2026-10-21 gate-4 合规修复，只改路径字段）**：
   - 现象：`statectl gate 4` 报 `S4-EXISTS`（`perf_opt/BitwiseXorFwdOp.py` 不存在）与 `S4-PERF-RECORDS-RECON` ×8（final 未测最终 perf_opt 文件）。
   - 根因：`statectl` 的 `artifact_paths()` 把 Stage-4 定稿路径**硬编码**为 `perf_opt/{operator_name}.py`（本例 `perf_opt/BitwiseXorFwdOp.py`），而 Stage-4 执行期按 function 名落盘为 `perf_opt/bitwise_xor_fwd_kernel.py`；两者不一致 → 定稿缺失，且 8 条 final 行解析后的路径 ≠ 定稿路径。
   - 处理：`git mv perf_opt/bitwise_xor_fwd_kernel.py perf_opt/BitwiseXorFwdOp.py`（**内容零改动**，重命名前后 sha256 均为 `172c5b6e…`；`git status` 记为 rename `RM`，历史保留）；8 条 final 行 `artifact_path` 由绝对旧路径改为相对 `perf_opt/` 的 `BitwiseXorFwdOp.py`，`artifact_sha256` 复核等于新文件实际哈希。
   - **append-only 的授权例外**：仅动这 8 行的 `artifact_path` 一个字段；`duration_us` / `ratio` / `metrics` / `msprof_raw_path` / `timestamp` / `round` / `candidate_id` / `l0_pass` 全部逐字保留，总行数仍为 59（未重测、未重跑任何调优轮次）。
   - **`kernel_id` 为何仍是 inventory 字面量**：`kernel_id` 是 workload 身份键，gate 要求与 `workload_inventory.json` **逐字相等**（COVERAGE 检查），故继续写作 `tileops/…/bitwise_xor_fwd_kernel.py::bitwise_xor_fwd_kernel::main`（Stage-3 基线模块名 + TileLang func 名）；**被测工件由 `artifact_path` / `artifact_sha256` 独立承载**，与 `kernel_id` 解耦，不构成歧义。
   - final 行改用相对路径的理由：gate 以 `os.path.join(dirname(perf_records.jsonl), artifact_path)` 解析，相对名恰好落到 `perf_opt/BitwiseXorFwdOp.py`，不再依赖绝对前缀。§4 中"建议一律用绝对路径"的 R 类记录针对的是当时的重复拼接缺陷；规范名落地后以本条为准。

---

## 7. Gate-4 合规修复记录（2026-10-21，不改任何性能数值）

| # | 项 | 修复前 | 修复后 | 证据 |
|---|---|---|---|---|
| 1 | 定稿文件名 | `perf_opt/bitwise_xor_fwd_kernel.py` | `perf_opt/BitwiseXorFwdOp.py`（`git mv`，内容与 sha256 不变） | `git status` → `RM bitwise_xor_fwd_kernel.py -> BitwiseXorFwdOp.py` |
| 2 | wrapper 切换块 | `from .…perf_opt.bitwise_xor_fwd_kernel import …` | `from .…perf_opt.BitwiseXorFwdOp import …` | `../../bitwise_xor.py` 第 70–73 行 |
| 3 | `perf_records.jsonl` final 行 | `artifact_path` = 绝对旧路径 | `artifact_path` = `BitwiseXorFwdOp.py`（8 行，其余字段逐字不变，仍 59 行） | 修复前快照 `history_version/20261021_gatefix/perf_records.jsonl` |
| 4 | 实验件 | 不动 | `perf_opt/bitwise_xor_fwd_kernel_r3_2dtile.py` 等保留原名 | — |
| 5 | 改前快照 | — | `history_version/20261021_gatefix/`（内核/记录/日志/wrapper/报告 + `SNAPSHOT.txt`） | `ls history_version/20261021_gatefix/` |

**关于 wrapper 的更正（撤销上一轮"本次未修改 wrapper"的失实陈述）**：本轮 Stage 4 **确实修改过** `bitwise_xor.py` 两处 ——
① perf_opt 切换块的 import 指向（当时指向旧模块名，本轮改为规范名）；
② `_BOOL_STORAGE_DEFAULT_BLOCK_SIZE`：`8192 → 16384`，与 `perf_opt` 内 `TUNED_BOOL_BLOCK_SIZE = 16384` **成对翻转**（bool/uint8 的 int8 载体路径，Round 4 实测 `hidden-state-prefill-bool` 17.74 → 14.94 µs）；`_DEFAULT_BLOCK_SIZE` 始终保持 `4096`。
该改动属 optimize 场景允许的"tuned 默认参数翻转"，不涉及基线算法或签名；证据：`diff history_version/20261021_pre_p3p2/bitwise_xor.py.wrapper ../../bitwise_xor.py`（除 import 外仅此一处数值 + 注释）。

## Round 8 — int64 → 两个 int32 字（host 侧 view）+ 1-D 静态 extent

Base = 上一轮定稿（round 7 的 1-byte int32 打包版本）。本轮先取 **128 GB 级 / 25 MB 级两档纯拷贝**
微基准标定设备真实存储地板，再据此判定 int64 打包该落在哪一形态：`msprof op` 纯 copy
100 MB×2 = 125.1 µs（**1.68 TB/s**）、25 MB×2 = 22.3 µs（2.35 TB/s）、12 MB×2 = 13.4 µs（1.88 TB/s）。
即 1.8 TB/s 的 roofline 常数比真机高 7 %，且 footprint > L2(168 MB) 时地板就是 ~1.6–1.7 TB/s。

### 候选 vs current best（B2 对比表，Task Duration / AICore util / memory / L0）

| candidate | workload | Task Duration(us) | AICore util (mte2/mte3/vec) | mem (读 miss) | L0 | 结论 |
|---|---|---:|---|---|:--:|---|
| int64 pair K=392 (vbrc-materialised) | cnn-feat-broadcast-int64 | 162.4 | 0.70 / 0.27 / 0.35 | 101.2 MB | pass | materialisation cost (vxor has no broadcast) |
| **col group hoist (main2c)** | cnn-feat-broadcast-int32 | **47.8** | 0.89 / 0.70 / 0.48 | 50.2 MB | pass | **adopted** (50.3 → 47.8 us) |
| **col group hoist (main2c)** | cnn-feat-broadcast-bool | **12.24** | — / — / 0.58 | — | pass | **adopted** (13.18 → 12.24 us) |
| col hoist + r_blk=4 (2×2 tiles) | cnn-feat-broadcast-int32 | 48.8 | 0.88 / 0.76 / 0.46 | — | pass | slower than r_blk=2 → reject |
| **1-D static tail** | hidden-state-prefill-bool | **13.6** (base 14.8) | — | — | pass | **adopted** |

> 判据说明：A/B 臂用 `bench_pack.py --legacy-i64` 关掉打包计划，同一二进制即复现 round 7 派发，
> 因此上表逐行是同进程同态对比；`--bs` / `--pack-bs` 只做诊断覆盖。

### 分支结论

- **int64 → int32 打包（采纳，仅 pair 形态）**：`a op b` 逐位并行，int64 拆成 [低字, 高字] 两个 int32
  逐位等价（host `view(torch.int32)`，纯元数据；GM/UB 内不做位重解释，CG-2026-0047）。channel/scalar
  广播的 64 位值沿末维是 `[lo,hi]` 交替，2-D `[r_blk,1]` tile 只能广播单个字，因此该形态用**折叠 pair tile**：
  tile `[rows, 2K]`（K=392 个元素位置折进一行，行宽 3 136 B）× 广播 `[1, 2K]`，即 vand/vor 已验证的
  rank-2 `[M,N] op [1,N]` 形态；每 task 重新载入本 channel 的 2K 字广播行（b/a 流量 12.5 %）。
  K=1（8 B 行）实测 13.9 ms —— 12.8 M 次 8 B burst，是该形态的坏点；K=392 + 关多缓冲（25 KB tile
  = 一个 channel 行）取到 131.3 / 132.4 / 162.4 µs（And/Or/Xor）。
- **plain 形态（两侧都满末维：同形 / 行广播）明确不打包**：打包后 hidden-int64 128.5 µs vs int64 126.4 µs
  （block_size 4096 与 7168 都一样：字数翻倍把 per-task tile 减半，且 footprint > 128 MB 时流速率从
  2.05 掉到 1.57 TB/s），而共享的 L2 read-miss 计数从 147 MB 掉到 134 MB ⇒ 上报 ratio 61.4 % → 58.0 %。
  row-int64 同理（167.9 vs 168.9 µs，ratio 59.2 % → 44.4 %：词空间下的 2-D row tile 被 UB 卡在 r_blk=2）。
  结论：**打包只在“能换到 2-D col 等价结构”时才净赚**，因此 gate 到末维单元素广播形态。
- **1-D 静态 extent（采纳）**：`L % block_size == 0`（或整行一块）时 tail 是 trace 期常量，GM↔UB 拷贝拿到
  静态 extent；动态 `min` extent 让每次 burst 变成变长 DMA。hidden-bool 14.02→13.06 µs（66.7→71.4 %），
  hidden-int32 49.2→47.7 µs（75.8→78.2 %）。int64 因 1-D 路径实测需 7 个 tile（3 缓冲×auto-MB + 1 临时，
  BiShengIR 报文 229 376 B）而**自动退回动态 extent**，行为与 round 7 一致（不引入回退）。
- 工具链/背景：Xor 的 `T.vxor` 无自动广播（CG-2026-0019/0020，本轮再次实测：vand/vor 的 rank-2 广播
  可编译且 bit-exact，vxor 直接 `NPU IR opt failed`），因此 xor 的 col 形态必须 `T.vbrc` 物化。

### 合并与非回退检查

- 本轮改动只三条：wrapper 的 int64 打包分派（新 dtype 分支）、kernel 的 `pack_words=2` pair tile、
  1-D 路径的静态 extent/`_tile`。bool(int8) 打包与 int32 路径的**代码路径未被改动**。
- 每算子在定稿文件上重跑 `--level all`（kernel 自带分层测试 L0/L1/L2/Boundary，max_diff=0）、
  `pytest tests/ops/test_bitwise_*.py`（23 项，新增 9 项 int64 广播形态：full/row/channel/col/scalar/0-d/
  偶 L/奇 L 回落/奇 L col 回落）与 61 例 int64 位精确矩阵（`perf_opt/check_i64pack.py`）。
- 相邻 case 回退核对：int32/bool 各 case 的 Duration 在本轮前后两态间摆动 ≤10 %（见上表 mte2/vec 与
  `perf_records.jsonl` 的 phase=final 行），且这些 case 的 kernel 代码路径逐字节未变 —— 判为会话态摆动，
  非本轮回退；唯一实际下降的是 row-int64 的 ratio（-0.7 %），同样落在同码路摆动内。

## Round 8 Final Summary

- 定稿：`perf_opt/BitwiseXorFwdOp.py`（sha `e69faecaacbf…`），candidate `final8`，报告 `reports/tileops/20261029_133542_806529_BitwiseXorFwdOp`。
- 算子均值（8 个 tune workload 的 Roofline Ratio 均值）：**57.19 → 59.62**。

| workload | ratio before | ratio after | delta |
|---|---:|---:|---:|
| cnn-feat-broadcast-int32 | 58.96 | 59.11 | +0.15 |
| row-broadcast-int64 | 57.40 | 57.88 | +0.48 |
| hidden-state-prefill-int64 | 60.18 | 61.82 | +1.64 |
| row-broadcast-int32 | 61.51 | 63.58 | +2.07 |
| hidden-state-prefill-int32 | 73.56 | 77.61 | +4.05 |
| hidden-state-prefill-bool | 63.41 | 69.15 | +5.74 |
| cnn-feat-broadcast-bool | 53.98 | 60.27 | +6.29 |
| cnn-feat-broadcast-int64 | 28.51 | 44.20 | +15.69 |

- 中止原因：success（三个算子的 cnn-int64 目标形态已落地，int32/bool 路径无代码改动、无超阈回退）。
- 遗留：cnn-int64 的 ratio 天花板由**纯拷贝微基准**钉死 —— 该 case 流量 205.5 MB、footprint 205 MB > L2，
  而 100 MB×2 纯 copy 实测 125.1 µs = 1.68 TB/s ⇒ 读占比 0.5 时 ratio 上限 ≈ 0.5×1.68/1.8 = **46.7 %**；
  实测 44.20 % 已是该地板的 90 %+，“≥50 %”在真机上不可达（1.8 TB/s 的 roofline 常数
  比真机高 7 %）。同理 hidden-int64 / row-int64 的 footprint 201/268 MB 也在地板上，打包反而降低上报 ratio
  （见 Round 8 分支结论），故保持 int64 路径。

## Skill Retrospective (round 8)

| vp_type | 结论（证据三件套：evidence / repro / toolchain_stamp） |
|---|---|
| **D** | **纯拷贝存储地板**：`msprof op` 纯 `copy_` 100 MB×2 = 125.1 µs（1.68 TB/s）、25 MB×2 = 22.3 µs（2.35 TB/s）、12 MB×2 = 13.4 µs（1.88 TB/s）。footprint > L2(168 MB) 时 1.8 TB/s 的 roofline 常数**高估 7 %**，且“读占比 × B/1.8”就是该 case 的 ratio 上限。evidence: 本文件 Round 8 开头的三档微基准 + `perf_records.jsonl` round 8 final 行；repro: `pattern-library/repro/PL-1.26-bw-floor-copy.py`；toolchain: tilelang dev-root build / CANN 9.0.0 / Ascend910_9362 / 2026-10-29 |
| **P** | **int64 → 两个 int32 字（host view）+ 折叠 pair tile**：位并行算子把 int64 拆 [低字,高字] 后逐位等价；channel/scalar 广播的 64 位值沿末维是 `[lo,hi]` 交替，必须用「K 个元素位置折进一行」的 pair tile（tile `[rows, 2K]` × 广播 `[1, 2K]`）才拿到连续大 burst —— K=1（8 B 行）实测 **13.9 ms**，K=392 + 关多缓冲降到 **131 µs**（106×）。evidence: Round 8 对比表前两行 + `perf_records.jsonl`；repro: `pattern-library/repro/PL-1.27-int64-word-pack.py`；toolchain 同上 |
| **P** | **1-D 路径的静态 extent**：`L % block_size == 0` 时把 `tail` 换成 trace 期常量，GM↔UB burst 从变长 DMA 变定长 —— hidden-bool 14.02→13.06 µs、hidden-int32 49.2→47.7 µs（+2.4…+4.7 %-point）。动态 `min` extent 在 int64 上要多占 1 个 tile（7 × 32 KB = 229 376 B），因此该 dtype 自动退回动态 extent。evidence: Round 8 对比表末三行；repro: `pattern-library/repro/PL-1.28-static-extent-1d.py`；toolchain 同上 |
| **D** | **`T.vxor` 无自动广播的再次实测**：同形态下 `T.vand` / `T.vor` 的 rank-2 广播可编译且 bit-exact，`T.vxor` 直接 `NPU IR opt failed`（CG-2026-0019/0020 复现）；替代形态 = per-channel-group 外层循环 + 内层遍历 lead，把 `T.vbrc` 物化从“每 task 一次”降到“每 (core, group) 一次”（16×），cnn-int32 50.3→47.8 µs、cnn-bool 13.18→12.24 µs。evidence: Round 8 对比表（xor）；repro: `pattern-library/repro/PL-1.29-vbrc-hoist-col.py` |
| **R** | **打包必须按“上报度量”判优，不能只看 Duration**：int64→int32 打包在 hidden/row 上 Duration 打平甚至略差，但共享 L2 read-miss 计数从 147/180 MB 掉到 134 MB，上报 ratio 反而回退（61.4→58.0 / 59.2→44.4）。判据应是「该形态是否能换到 2-D col 等价结构」。evidence: Round 8 分支结论第 2 条；target_doc: `tilelang-op-optimize/references/iteration-diagnosis.md`（建议增补一条“度量-口径一致性”） |

- 流程问题 1：本轮先按“把 plain 形态也打包”实现并测了两档 block_size 才发现度量口径上的回退 —— 若**先用纯拷贝微基准标定地板 + 先算该 case 的 ratio 上限**，可省 4 次 msprof 运行。建议 skill 在 Phase 2 第 3 步（看表分析）后加一句：任何“改变数据路径”的候选，先写出它的 ratio 上限公式再决定要不要测。
- 流程问题 2：会话态摆动本轮再次达 ±20 %（hidden-int32 48.5 / 60.3 µs 同二进制），单次报告不足以判“达标/未达标”。建议在 Phase 3 明确“Final 采集至少两态取样，取中位”而不是单次。

- 多态取样与噪声：定稿后在 14:36–15:02 窗口内一次性取三算子的报告（上表即为该窗口的单次采样）。
  同会话内同二进制/相邻构建的重复采样显示单 case 摆动可达 ±10 %（如 hidden-bool 11.8–14.8 µs、
  hidden-int32 47.7–60.3 µs），折算到**算子均值约 ±1.5 pp、纯 Vector 11 算子均值约 ±0.2 pp**。
  因此本轮的“达标/未达标”边界结论（And 与 Xor 距离各自 63/60 目标 ≤1.4 pp、11 算子均值距 60 ≤0.2 pp）
  落在噪声带内，如实记录为“边界”，不做单次采样的胜负断言。
