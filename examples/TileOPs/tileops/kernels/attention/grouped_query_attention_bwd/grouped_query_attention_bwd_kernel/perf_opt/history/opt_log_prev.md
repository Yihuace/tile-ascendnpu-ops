# Stage 4 调优日志 — GroupedQueryAttentionBwdOp（3-kernel pipeline，primary = `_gqa_bwd_kernel`）

- project: `grouped_query_attention_bwd`
- operator (primary, gate-4 对账对象): `_gqa_bwd_kernel`
- 辅助 kernel: `_flashattn_bwd_preprocess_kernel`、`_flashattn_bwd_postprocess_kernel`
- 性能目标: `best_effort`（用户未指定数值/类型），关注 shape 未指定 → 全部 tune workload 参与调优
- 唯一时延口径: `msprof op` 的 `Task Duration(us)`（kernel-only）；未使用 NPU event / runner 端到端时间
- 硬件/工具链: Ascend910B3（20 AI core / 40 AIV），CANN 9.0.0，`msprof 25.5.0`
- 时间戳: 2026-10-17

## 0. 记录拆分约定（多 kernel）

gate 4 的 `S4-PERF-RECORDS-RECON` 只按单一 `kernel_opt_py_path`（`perf_opt/_gqa_bwd_kernel.py`）对账，
本目录含 3 个 kernel，故按 kernel 拆分记录：

| 文件 | 内容 |
|---|---|
| `perf_records.jsonl` | **仅 primary**（`_gqa_bwd_kernel`）的 baseline / candidate / final 记录 |
| `perf_records_preprocess.jsonl` | `_flashattn_bwd_preprocess_kernel` |
| `perf_records_postprocess.jsonl` | `_flashattn_bwd_postprocess_kernel` |
| `perf_records_all.audit.jsonl` | **不可变审计副本**：上述三文件全部记录的并集（逐字节原样拼接） |
| `workload_inventory.json` | primary 的 smoke/tune 分类 |
| `workload_inventory_aux.json` | 两个辅助 kernel 的 workload 对 |

**并集校验（可复现命令与结论）**：

```
python - <<'PY'
import json
a=[json.loads(l) for l in open("perf_records.jsonl")]
b=[json.loads(l) for l in open("perf_records_preprocess.jsonl")]
c=[json.loads(l) for l in open("perf_records_postprocess.jsonl")]
d=[json.loads(l) for l in open("perf_records_all.audit.jsonl")]
assert sorted(map(json.dumps,a+b+c))==sorted(map(json.dumps,d))
print(len(a),len(b),len(c),len(d),"union==audit OK")
PY
```

结果见 §4.1「审计副本并集校验」；三文件与审计副本的记录内容逐条相同（**测量值未改**，append-only）。

## 1. Workload inventory（primary）

来源：`examples/TileOPs/benchmarks/ops/bench_grouped_query_attention_bwd.py::test_gqa_bwd_bench`（manifest 驱动，
`tileops/manifest/attention.yaml::GroupedQueryAttentionBwdOp.workloads`）。逐案例 `dtypes` 展开后共 **9 个案例**：

| workload_id | kind | q_shape | kv_shape | dtype | causal | 依据 |
|---|---|---|---|---|---|---|
| `gqa-bwd-smoke-float16` | smoke | [1,128,8,64] | [1,128,2,64] | fp16 | true | label 含独立 smoke token → 仅精度回归 |
| `llama-3.1-8b-short-float16` | tune | [4,512,32,128] | [4,512,8,128] | fp16 | true | manifest full |
| `llama-3.1-8b-short-bfloat16` | tune | [4,512,32,128] | [4,512,8,128] | bf16 | true | manifest full |
| `llama-3.1-8b-long-float16` | tune | [2,2048,32,128] | [2,2048,8,128] | fp16 | true | manifest full |
| `llama-3.1-8b-long-bfloat16` | tune | [2,2048,32,128] | [2,2048,8,128] | bf16 | true | manifest full |
| `llama-3.1-70b-short-float16` | tune | [2,512,64,128] | [2,512,8,128] | fp16 | true | manifest full |
| `llama-3.1-70b-short-bfloat16` | tune | [2,512,64,128] | [2,512,8,128] | bf16 | true | manifest full |
| `llama-3.1-70b-long-float16` | tune | [1,2048,64,128] | [1,2048,8,128] | fp16 | true | manifest full |
| `llama-3.1-70b-long-bfloat16` | tune | [1,2048,64,128] | [1,2048,8,128] | bf16 | true | manifest full |

- 无 benchmark 级 skip 案例（`bench_skip_reason` 未设置），故 `skipped` 为空。
- 8 个 tune 案例全部采集 baseline 并按实测耗时排序：8b-long/70b-long（≈141 ms / 140 ms）→ 70b-short（21.8 ms）
  → 8b-short（20.4 ms）；dtype 间无实质差异。排序依据为实测 `Task Duration`（见 §2）。
- `kernel_id` = `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main`
- `target_kernel_name`（msprof `--kernel-name`）= `_gqa_bwd_main_mix_aic`（实测 captured Op Name）

## 2. Performance Test Data（baseline，primary）

测量命令模板（每个 workload 一条，全部串行，**禁止并发 profiling**）：

```
msprof op --kernel-name=_gqa_bwd_main_mix_aic --output=profiles/<phase>/round<r>/main/<workload>_<candidate> \
  --launch-count=<N> --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory \
  python _bench_kernel.py --impl <candidate.py> --kernel main --B .. --S .. --H .. --H_kv .. --D .. --dtype .. --causal 1 --iters <N+5>
```

- baseline 用 `launch-count=10`，**final 用 `launch-count=20`**；`duration_us` 取 `OpBasicInfo` 中该 kernel 全部 launch 的
  中位数（`min~max` 见 `perf_records.jsonl` 对应 raw profile 目录）。原始 profile：`profiles/baseline/round0/main/<workload>_baseline/`。
- 全部 8 个 tune workload 的 baseline 记录写入 `perf_records.jsonl`（`phase=baseline`, `round=0`）。

| workload_id | baseline Task Duration (us) | captured_op_name | profile_status |
|---|---:|---|---|
| llama-3.1-8b-short-float16 | 20407.45 | `_gqa_bwd_main_mix_aic` | valid |
| llama-3.1-8b-short-bfloat16 | 20447.13 | `_gqa_bwd_main_mix_aic` | valid |
| llama-3.1-8b-long-float16 | 140535.47 | `_gqa_bwd_main_mix_aic` | valid |
| llama-3.1-8b-long-bfloat16 | 140647.91 | `_gqa_bwd_main_mix_aic` | valid |
| llama-3.1-70b-short-float16 | 21775.21 | `_gqa_bwd_main_mix_aic` | valid |
| llama-3.1-70b-short-bfloat16 | 21760.60 | `_gqa_bwd_main_mix_aic` | valid |
| llama-3.1-70b-long-float16 | 141741.64 | `_gqa_bwd_main_mix_aic` | valid |
| llama-3.1-70b-long-bfloat16 | 141789.75 | `_gqa_bwd_main_mix_aic` | valid |

smoke（`gqa-bwd-smoke-float16`）只做精度回归，不采集性能、不选 winner（`workload_inventory.json` 中 `precision_pass=true`）。

### 2.1 设计估算 vs 实测（D-2 回填）

`DESIGN.md` 未给出 §1.6.0 形式的「设计期估算下界」行（roofline 见 manifest：`flops = 10·B·H·S²·D/2`，
`bytes = B·(3H+4H_kv)·S·D·elem_bytes`）。以 manifest roofline 反推：8b-short fp16 的 flops = 2.15e10，
baseline 20.4 ms ⇒ 1.05 TFLOP/s（910B3 fp16 峰值 ~300 TFLOP/s 量级）⇒ **偏差 >2 个数量级**，失准项为
**发射/标量流水（scalar issue）而非流量或容量**——profile 直接证实（下一节）。

## 3. 瓶颈定位（baseline profile 解析）

`llama-3.1-8b-short-bfloat16`（launch-count=10，raw: `profiles/baseline/round0/main/llama-3.1-8b-short-bfloat16_baseline/`），
对 `PipeUtilization` 的 7680 条 block 记录按 sub_block 取均值（每程序 = 1 cube + 2 vector）：

| 观测 | baseline |
|---|---|
| `aiv_time(us)`（vector 核墙钟） | 1496 |
| **`aiv_scalar_time(us)`** | **1192（= 80% 的 aiv_time）** |
| `aiv_scalar_mte2_stall_time(us)` | 321 |
| `aiv_vec_time(us)` | 30（2%） |
| `aiv_mte2_time` / `aiv_mte3_time` | 141 / 220 |
| `aic_time(us)`（cube 核墙钟） | 1496 |
| **`aic_cube_time(us)`（cube 真正算的时间）** | **10.7（0.7%）** |
| `aic_scalar_wait_id2_time(us)`（cube 等 vector） | 1475 |

结论：**kernel 完全被 V 核标量流水支配，cube（GEMM）几乎空闲**（10.7 µs / 1496 µs）。逐项归因到两个
per-tile 的**逐元素标量**工作（BM×BN = 64×32 的 tile，每程序约 64 个 tile → 每 tile 标量 ≈ 24–33 µs）：

1. **causal mask 的 `T.vcmp`**：编译期即告警
   `Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency`。
   int32 操作数（`dif`/`lim_mat` 为 int32）走了标量比较路径，2048 次/ tile 的逐元素标量；
2. `dq` 的跨 kv-block 原子累加：`T.serial(BN)` 行循环 + `hivm.hir.store ... atomic = <add>`（每 tile 32 次 512 B
   原子写，4096 元素/ tile 由标量发射）。

`T.Pipelined(num_stages=1)` 的 MTE2 in-scalar stall 321 µs 是第三位（非主因，见 Round 1 结论）。

## 4. Iteration Log

每轮固定 base = 当前已合并 current best；分支从同一 base 派生、每分支只改一个主要优化点。
对比表数据取自 `perf_records.jsonl` 与 raw profile；代表性 workload = `llama-3.1-8b-short-float16`（20.4 ms 级，
编译/测量成本低，且与 long 系列的瓶颈结构相同）。

### Round 1（base = baseline）

现象：见 §3，V 核标量 80%，cube 空闲。候选优化点（同一 base 派生 3 个分支）：

| 分支 | 主要优化点 | 实现 |
|---|---|---|
| `r1_pipe2` | 消除 MTE2 in-scalar stall（321 µs/block）：q/do/lse/delta 双缓冲 | `T.Pipelined(..., num_stages=2)` |
| `r1_maskguard` | mask 只在跨对角 tile 上做 | `if q0 < kv0 + BM - 1:` 包裹 vbrc/vcmp/vselect |
| `r1_atomtile` | dq 原子由 BN 次行原子改 1 次整块原子 | `T.atomic_add(dq[bz,q0,h,0], dq_ub, [BN, dim])` |

分支结果：

| 分支 | Task Duration (us) | AICore 利用率（aiv_scalar_time / aiv_time） | memory（mte2/mte3, us） | L0 | 结论 |
|---|---:|---|---|---|---|
| current best（baseline） | 20407.45 | 1192 / 1496 = 80% | 141 / 220 | PASS | — |
| `r1_pipe2` | 20419.81 | 1175 / 1496 = 79% | 142 / 110 | 未单测（仅调度参数，语义同构；后续随 winner 一起 L0/L1） | `config_no_gain`（+0.06%，<3% 噪声阈值）→ 回滚 |
| `r1_maskguard` | —（未进入性能测量） | — | — | **FAIL**（smoke: dq 9.92 / dk 8.14 / dv 2.66，全部超限） | 条件分支内写 `qkt_ub` 后 Developer 自动同步未覆盖 → 静默数值错误；blocked（写法不可用） |
| `r1_atomtile` | —（未进入性能测量） | — | — | **FAIL**（smoke: dq max_diff 1.43e+1 vs ref 1.25e+1） | 跨步目标被当作连续区域 → **静默写错**，与 `DESIGN.md` §11 陷阱行预测一致；blocked |

winner：**无**（current best 不变 = baseline）。本轮无 valid 分支带来 >3% 提升。

> §3 的结论"mask 的 int32 vcmp 走标量"在本轮被两个失败分支反证式确认：只要 mask 仍逐 tile 执行且用 int32，
> 标量成本就在；要降低它，必须让比较本身走向量路径（Round 2）。

### Round 2（base = baseline）

现象：R1 表明 per-tile 标量成本 = mask（int32 vcmp，~50%）+ dq 原子（~50%）。`vcmp` 的文档化 dtype 支持包含
fp32（`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1），而编译器告警只在 int32 操作数时出现。
候选优化点：**把 mask 的比较操作数由 int32 改为 fp32**（`rowmat`/`colmat`/`dif`/`lim_mat` 改 `accum_dtype`，
`T.vbrc(T.float32(kv0-q0), lim_mat)`），tile 形状/数据流/原子结构完全不动。

预检：同形态编译（S=128/D=64）**不再出现 `vcmp` 标量降级告警**；精度 screen（3 例：causal smoke、n_kv=4 causal、
non-causal）全 PASS，worst max_diff 5.58e-03（与 baseline 同量级）。

| 分支 | Task Duration (us) | AICore 利用率（aiv_scalar / aiv_time） | memory（mte2/mte3, us） | L0 | 结论 |
|---|---:|---|---|---|---|
| current best（baseline） | 20407.45 | 1192 / 1496 = 80% | 141 / 220 | PASS | — |
| **`r2_fp32vcmp`** | **5882.69** | **130 / 432 = 30%** | 142 / 220 | PASS（screen 3 例） | **winner：-71.2%（3.47x）** |

winner 解析（`profiles/candidate/round2/main/llama-3.1-8b-short-float16_r2_fp32vcmp/`）：`aiv_scalar_time` 1192 → 130 µs，
`aiv_time` 1496 → 432 µs，`aic_cube_time` 仍 10.7 µs。→ 标量瓶颈解除后 kernel 转为 **dq 原子（mte3）+ MTE2 + 残量标量**
三足鼎立（下一轮）。

### Round 3（base = `r2_fp32vcmp`）

现象（新 profile）：`aiv_mte3_time` 220 / 432 µs（50%）、`aiv_mte2_time` 142、`aiv_scalar_time` 130；
cube 仍空闲。候选优化点：

| 分支 | 主要优化点 | 结果 |
|---|---|---|
| `r3_fp32vcmp_pipe2` | base + `num_stages=2`（overlap q/do 读取与计算） | 5863.83 µs（-0.3%，<3% 噪声）→ `config_no_gain`，回滚 |
| `r3_probe_noatomic` | **诊断探针（数值错误，仅定位用，不入 records）**：把 dq 原子替换为普通 `T.copy` | 3876.38 µs ⇒ 原子成本 ≈ 2007 µs = **34%** 的当前耗时；探针下 `aiv_mte3_time` 220 → 7 µs（原子即 mte3 主力） |

| 分支 | Task Duration (us) | AICore 利用率（aiv_scalar / aiv_time） | memory（mte2/mte3, us） | L0 | 结论 |
|---|---:|---|---|---|---|
| current best（`r2_fp32vcmp`） | 5882.69 | 130 / 432 = 30% | 142 / 220 | PASS | — |
| `r3_fp32vcmp_pipe2` | 5863.83 | 112 / 421 = 27% | 142 / 110 | 未单测（调度参数） | `config_no_gain` → 回滚 |
| `r3_probe_noatomic`（探针） | 3876.38 | 187 / 294 = 64% | 28 / 7 | **故意 FAIL（诊断）** | 不入 records；确认原子 34% |

**atomic 结构化尝试的评估（未实施为分支的原因，附证据）**：要减少 dq 原子元素数只能减少每个 q 行的
kv-block 贡献数（`#elem = n_kv · B·S·H·D`，`n_kv = S/BM`）。
- 增大 `BM`：`BM=128` 时 `dk_f/dv_f`（`[128,128]` fp32 = 64 KB each）已超 L0C 128 KB 预算 → 不可行
  （`BM=64` 时 L0C 占用 ≈ 96 KB 已接近上限）；
- 增大 `BN` 至 64：UB 静态预算（k_s/v_s/q_s/do_s 各 16 KB、qkt/dp/ds/tmp 各 16 KB、p16/ds16 各 8 KB、
  dq_ub 32 KB、dk_ub/dv_ub 32 KB each）> 192 KB → 不可行；
- 每程序融合 2 个 kv block（原子减半）：需常驻两块 K/V（额外 32 KB）并以 `for g→for i→for kb` 重排、
  dq 在 UB 内 fp32 累加后单次原子；在剩余预算内（`dk_ub`/`dv_ub` 复用得 32 KB）理论可容，但**未实测**——
  本轮时间预算耗尽前未实现，如实记为未验证假设，不作结论。

winner：**`r2_fp32vcmp`**（R3 无有效增益）。

### Round 4（收束）

- final 候选 = `r2_fp32vcmp`（内容与 `perf_opt/_gqa_bwd_kernel.py` 相同，tiling 默认值未变，故不新增
  `TUNED_*` 模块级常量）。
- 在最终文件上以 `launch-count=20` 重新采集全部 8 个 tune workload 的 `phase=final` 记录（同一 final candidate
  `r2_fp32vcmp_final`，不拼接各分支最好成绩）；smoke 只做精度回归。
- 中止判定：`budget_exhausted`（单次会话预算）+ 剩余候选为"未验证的结构假设"（见 R3 评估），非 `plateau`
  （R2 有 3.47x 提升，R3 仍有 2 个明确候选被实测否决）。

## 4.1 审计副本并集校验（measurement values unchanged）

```
$ python /tmp/mk_audit.py   # 逻辑等价脚本，逐行拼接三文件
{'perf_records.jsonl': 19, 'perf_records_preprocess.jsonl': 16,
 'perf_records_postprocess.jsonl': 16} audit rows 51 union==audit OK
```

- `perf_records.jsonl`（19 行）= primary 的 8 baseline + 3 candidate（`r1_pipe2` / `r2_fp32vcmp` /
  `r3_fp32vcmp_pipe2`）+ 8 final；
- `perf_records_all.audit.jsonl`（51 行）= 上述三文件的并集，记录内容逐条相同（**测量值未改**）；
- 诊断探针 `r3_probe_noatomic`（数值故意错误）**未写入任何 records**，只作为 raw profile 证据保留
  （`profiles/candidate/round3/main/…_r3p_noatomic_probe/`）；
- blocked / L0-fail 分支（`r1_maskguard`、`r1_atomtile`、`r2_maskwindow`）**无 records**（`duration_us` 不为 null 的
  schema 约束），其结构性证据以 L0 日志 + opt_log 表格形式留档。

## 5. Autotune Log

未使用 autotune：本 kernel 的可调参数只有 `BM/BN/num_stages`，其中 `BM`/`BN` 被 L0C/UB 静态预算锁死
（上节证据），`num_stages` 已由 `r1_pipe2`/`r3_pipe2` 两点实测（均 <3%），无需搜索空间。tune 收益来自
**编译器行为（vcmp 降级路径）**而非参数。

## 6. Final Summary

- final 候选: `r2_fp32vcmp_final`（= `perf_opt/_gqa_bwd_kernel.py`，sha256 b4d0d901f57036fd…），tiling 默认 `BM=64 / BN=64(D<=64) or 32(D=128) / num_stages=1` 未变。
- 全部 final 时延来自**同一最终候选**在最终文件上以 `launch-count=20` 重新采集的 `phase=final` 记录（不拼接分支最好成绩）。
- smoke 精度：`perf_opt/_gqa_bwd_kernel.py --level L0` PASS（见 §6.1）。
- 中止原因：`budget_exhausted`（会话预算；R3 已把两个剩余候选实测否决，另有一个未验证结构假设如实留档）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-short-float16 | 20407.45 | 5873.39 | r2_fp32vcmp_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-short-bfloat16 | 20447.13 | 5868.99 | r2_fp32vcmp_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-long-float16 | 140535.47 | 39910.87 | r2_fp32vcmp_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-long-bfloat16 | 140647.91 | 39910.47 | r2_fp32vcmp_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-short-float16 | 21775.21 | 6157.27 | r2_fp32vcmp_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-short-bfloat16 | 21760.60 | 6170.78 | r2_fp32vcmp_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-long-float16 | 141741.64 | 40137.94 | r2_fp32vcmp_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-long-bfloat16 | 141789.75 | 40106.78 | r2_fp32vcmp_final |

**原始 profile 证据（final，`profiles/final/round4/main/<workload>_r2_fp32vcmp_final/`）**：

| workload_id | aiv_scalar_time(us) | aiv_vec_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | aic_cube_time(us) | aiv_time(us) |
|---|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b-short-float16 | 129.2 | 28.6 | 141.9 | 220.2 | 10.7 | 431.8 |
| llama-3.1-8b-short-bfloat16 | 129.9 | 29.4 | 141.8 | 220.6 | 10.7 | 432.1 |
| llama-3.1-8b-long-float16 | 455.4 | 104.1 | 519.3 | 790.1 | 39.2 | 1547.5 |
| llama-3.1-8b-long-bfloat16 | 456.0 | 106.8 | 518.1 | 791.1 | 39.2 | 1547.1 |
| llama-3.1-70b-short-float16 | 251.1 | 57.4 | 282.8 | 439.5 | 21.4 | 854.7 |
| llama-3.1-70b-short-bfloat16 | 252.9 | 58.9 | 284.5 | 440.1 | 21.4 | 857.0 |
| llama-3.1-70b-long-float16 | 904.3 | 208.0 | 1043.7 | 1576.2 | 78.5 | 3086.7 |
| llama-3.1-70b-long-bfloat16 | 906.7 | 213.3 | 1042.6 | 1580.9 | 78.5 | 3087.7 |


### 6.1 最终精度回归（`perf_opt/_gqa_bwd_kernel.py`）

- `run_L0()`：**8/8 PASS**（causal smoke fp16/bf16、non-causal fp16/bf16、4 个 manifest fp16）
- `run_L1()`：**12/12 PASS**（4 个 manifest bf16、GQA groups = 1/2/4/8、`n_kv=1` 与 `n_kv=4` 的 dq 原子对拍、
  non-causal D=128、D=64 bf16）
- 完整输出：`logs/final_primary_L0_L1.log`；golden 为模块内独立 fp32 CPU 参考（与 kernel 的 tiling/原子结构无关）
- 两个失败分支的反例 screen 日志：`logs/r1_maskguard_check.log`、`logs/r1_atomtile_check.log`
- smoke workload（`gqa-bwd-smoke-float16`）只参与上述精度回归，未做性能采集：`workload_inventory.json`
  `kind=smoke`, `precision_pass=true`

## 7. 辅助 kernel（preprocess / postprocess）

两个辅助 kernel 与 primary 同属一条 pipeline（preprocess → main → postprocess），在 manifest 的 8 个
tune workload 上**逐一采集 baseline 与 perf_opt 副本的 final**（`--kernel preprocess|postprocess`，同一
`_bench_kernel.py` 探针，串行 msprof）。两者均为**纯访存/规约型**小 kernel：preprocess = `delta = rowsum(o*do)`
（fp32 规约 + BSH→BHS 转置写），postprocess = `dq` fp32→storage dtype 的 RNE cast。

实测 baseline profile（preprocess, 8b-short）：`aiv_mte2_ratio ≈ 0.52`、`aiv_vec_ratio ≈ 0.50`，MTE2 与 Vector
已接近满载、cube 完全不参与 → 属带宽/向量吞吐地板，**未发现可结构性削减的候选优化点**。因此 `perf_opt/` 下的两个
副本与 Stage-3 基线**实现等价**（仅 docstring 增补说明），实测差异全部落在 ±2.2% 噪声带内（下两表），
`tuning_status = no_gain`。

**preprocess**（`perf_records_preprocess.jsonl`）

| workload_id | baseline_us | final_us | delta |
|---|---:|---:|---:|
| llama-3.1-8b-short-float16 | 34.56 | 34.51 | -0.1% |
| llama-3.1-8b-short-bfloat16 | 34.98 | 34.83 | -0.4% |
| llama-3.1-8b-long-float16 | 61.02 | 61.14 | +0.2% |
| llama-3.1-8b-long-bfloat16 | 60.98 | 60.89 | -0.1% |
| llama-3.1-70b-short-float16 | 35.49 | 35.76 | +0.8% |
| llama-3.1-70b-short-bfloat16 | 35.70 | 35.79 | +0.3% |
| llama-3.1-70b-long-float16 | 65.84 | 65.92 | +0.1% |
| llama-3.1-70b-long-bfloat16 | 65.89 | 66.00 | +0.2% |

**postprocess**（`perf_records_postprocess.jsonl`）

| workload_id | baseline_us | final_us | delta |
|---|---:|---:|---:|
| llama-3.1-8b-short-float16 | 24.64 | 24.24 | -1.6% |
| llama-3.1-8b-short-bfloat16 | 24.35 | 24.45 | +0.4% |
| llama-3.1-8b-long-float16 | 47.00 | 47.07 | +0.1% |
| llama-3.1-8b-long-bfloat16 | 46.49 | 46.65 | +0.3% |
| llama-3.1-70b-short-float16 | 24.11 | 24.19 | +0.3% |
| llama-3.1-70b-short-bfloat16 | 24.03 | 24.32 | +1.2% |
| llama-3.1-70b-long-float16 | 46.81 | 46.42 | -0.8% |
| llama-3.1-70b-long-bfloat16 | 47.63 | 46.60 | -2.2% |

**pipeline 时延构成（final，µs；三段均为 kernel-only `Task Duration`）**

| workload_id | preprocess | main | postprocess | sum | main share |
|---|---:|---:|---:|---:|---:|
| llama-3.1-8b-short-float16 | 34.51 | 5873.39 | 24.24 | 5932.14 | 99.01% |
| llama-3.1-8b-short-bfloat16 | 34.83 | 5868.99 | 24.45 | 5928.27 | 99.00% |
| llama-3.1-8b-long-float16 | 61.14 | 39910.87 | 47.07 | 40019.08 | 99.73% |
| llama-3.1-8b-long-bfloat16 | 60.89 | 39910.47 | 46.65 | 40018.01 | 99.73% |
| llama-3.1-70b-short-float16 | 35.76 | 6157.27 | 24.19 | 6217.22 | 99.04% |
| llama-3.1-70b-short-bfloat16 | 35.79 | 6170.78 | 24.32 | 6230.89 | 99.04% |
| llama-3.1-70b-long-float16 | 65.92 | 40137.94 | 46.42 | 40250.28 | 99.72% |
| llama-3.1-70b-long-bfloat16 | 66.00 | 40106.78 | 46.60 | 40219.38 | 99.72% |

**辅助 kernel 精度回归（`perf_opt/{preprocess,postprocess}`，`--level L0`）**

- `_flashattn_bwd_preprocess_kernel.py`：**10/10 PASS**（smoke + 4 manifest × fp16/bf16，max_diff ≤ 7.63e-06）
- `_flashattn_bwd_postprocess_kernel.py`：**10/10 PASS**（同上，max_diff = 0）
- 输出：`logs/aux_L0.log`

**结论（端到端瓶颈归属）**：pipeline 总耗时中 **99.0%–99.7% 由 primary `_gqa_bwd_kernel` 占据**，
preprocess + postprocess 合计 <1.3%，且两者已在噪声内不可再优化 → 端到端瓶颈 = main kernel；
本 Stage 的 3.47–3.54x 收益即端到端收益（kernel-only 口径）。辅助 kernel 的 final 记录保留在同一
`perf_records_{preprocess,postprocess}.jsonl` 中（不与 primary 的 `perf_records.jsonl` 混写）。


## 8. Skill Retrospective

见 §9。

## 9. Skill Retrospective（复盘）

### 9.1 本轮暴露出的事实（value points，含 vp_type 与证据三件套）

| ID | vp_type | 结论 | evidence（实测） | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-A `vcmp-int32-scalar` | D（实测数据/陷阱） | `hivm.vcmp` 在 **int32** 操作数时降级为标量指令（编译期 warning），成为 GQA-bwd 主 kernel 的 80% 耗时；换成 **fp32** 操作数后同一比较走向量路径，20407→5883 µs（3.47x），数值不变 | `perf_records.jsonl` baseline vs `r2_fp32vcmp`（同 workload 同 launch-count）；`PipeUtilization` aiv_scalar_time 1192→130 µs；编译日志 warning 有无对照 | `_gqa_bwd_kernel_round2_fp32vcmp.py` 相对 `_gqa_bwd_kernel.py` 的最小 diff：4 个 fragment 的 dtype int32→float + `T.vbrc(T.float32(...))` | CANN 9.0.0 / Ascend910B3 / tilelang build at 2026-10-17（`/home/developer/tilelang-mlir-ascend/build`） |
| VP-B `mask-block-in-pipelined-sync` | D（陷阱，反向） | 在 `T.Pipelined` body 内用 runtime `if` 包裹写 UB 的 v-op（vcmp/vselect）→ 静默数值错误（dq/dk/dv 全错），Developer 自动同步未覆盖条件块 | `logs/r1_maskguard_check.log`（smoke max_diff dq 9.92e+0） | `_gqa_bwd_kernel_round1_maskguard.py`（保留作反例） | 同上 |
| VP-C `atomic-strided-tile-silent-wrong` | D（陷阱，与既有 DESIGN 陷阱行一致） | `T.atomic_add(dq[bz,q0,h,0], dq_ub, [BN,dim])` 对 BSHD 跨步行区把目标当连续 → **静默写错**（dq max_diff 1.43e+1 vs ref 1.25e+1），`docs/…/T.atomic_add.md` §2.4 的 `size=` 形态只对连续区域有效 | `logs/r1_atomtile_check.log` | `_gqa_bwd_kernel_round1_atomtile.py` | 同上 |
| VP-D `probe-ablation-for-atomic-cost` | P（方法） | 用"把原子换成普通 store"的诊断探针（不写入 records、标注错误）一次性把 dq 原子成本定量为 34%（5883→3876 µs，mte3 220→7 µs），避免了对原子做盲目结构改造 | `profiles/candidate/round3/main/llama-3.1-8b-short-float16_r3p_noatomic_probe/` | `_gqa_bwd_kernel_round3_probe_noatomic.py` | 同上 |
| VP-E `config-only-variants-tie` | D | `num_stages=2` 在本 kernel 上无收益（20419.8 / 5863.8 vs 20407.5 / 5882.7，<3%）：MTE2 stall 不是主瓶颈时双缓冲无效 | `perf_records.jsonl` r1_pipe2 / r3_fp32vcmp_pipe2 行 | 同上两文件 | 同上 |
| BP_1 | R（流程） | skill 的"诊断信号强制换轴分析"应显式包含**编译期 warning 采集**（`bishengir-compile` 的 `will execute by scalar instruction`）：本例中一条 warning 直接指向 3.47x 收益，而 msprof 指标只显示"scalar 高"却无法指出是哪个 op | §3 + VP-A | — | — |
| BP_2 | R（流程） | 多 kernel 目录下 `perf_records` 拆分 + 审计副本的写法（本轮 §0）应固化为 skill 的 Phase 3 步骤，避免 gate 4 单文件对账失败重试 | §0 并集校验 | — | — |
| BP_3 | P（方法） | 结构改造前的**静态预算核算**（L0C 128 KB / UB 192 KB 逐 buffer 求和）应在生成分支前完成：本轮 `BN=64`、`BM=128`、2-kv-block 融合三个方向在写代码前即被预算否决/降级，节省了 3 次无谓编译 | §4 Round 3 评估 | — | — |

### 9.2 skill 流程层面

- Phase 2 的「每分支先 L0 再 profile」在本例中是必要的：R1 的两个分支都在 L0 阶段暴露为静默错误，
  没有浪费 profile 时间（对比：若先 profile 会得到"更快"的错误分支）。
- 建议 skill 在 Phase 0 增加一条：**编译日志的 scalar-lowering warning 与 msprof 的 `*_scalar_ratio` 应交叉验证**；
  本任务 baseline 的 `aiv_scalar_ratio` 字段（0.057）与 `aiv_scalar_time/aiv_time`（80%）口径不一致，
  若只看前者会误判瓶颈（口径差异见 `PipeUtilization` 同名字段，前者为按 launch 归一的比例）。
