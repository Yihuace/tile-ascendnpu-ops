# Stage 4 调优日志 — GroupedQueryAttentionBwdOp（primary = `_gqa_bwd_kernel`，第 2 轮会话）

- project: `TileOPs`
- operator (gate-4 对账对象): `_gqa_bwd_kernel`
- 目录: `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/`
- 模式: `scenario=optimize`、`mode=full`
- 用户硬目标: 官方 workload 的 **msprof roofline `GM Read + Write` Ratio 平均 ≥ 40%**（baseline 1.461%）
- 唯一时延口径: `msprof op` 的 `Task Duration(us)`（kernel-only，20 次 launch 中位）；未使用 NPU event / 端到端时间
- 设备/工具链: Ascend910B3（`cube_core_num=20`、`vector_core_num=40`，rated 1800 MHz，HBM 64 GB，UB 192 KB/核）；
  CANN 9.0.0（`/home/developer/Ascend/cann-9.0.0`）；msprof 25.5.0；tilelang `0.1.2+9738284cd6`（dev root build）
- 时间: 2026-10-31
- 上一轮（第 1 轮会话）完整历史: `perf_opt/history/opt_log_prev.md` + `perf_opt/history/perf_records_prev.jsonl`

## 0. 记录拆分约定

| 文件 | 内容 |
|---|---|
| `perf_opt/perf_records.jsonl` | **仅 primary**（`_gqa_bwd_kernel`）的 baseline / candidate / final 记录（本轮新建） |
| `perf_opt/workload_inventory.json` | primary 的 smoke/tune 分类与 `tuning_status` |
| `perf_opt/workload_inventory_aux.json` | 两个辅助 kernel（pre/post）的 `no_gain` 结论与依据 |
| `perf_opt/history/*` | 第 1 轮会话的全部 records / inventory / opt_log 原样归档（不与本轮 final 混淆） |

`kernel_id`（gate 4 字面量）:
`examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main`
`target_kernel_name`（`msprof --kernel-name`）= `_gqa_bwd_main_mix_aic`（实测捕获 Op Name）。

## 0.1 量化可行性裁决（先于实验；本轮第一交付物）

### 0.1.1 口径换算（两套「ideal」必须分开写）

`Ratio` 的定义（`examples/TileOPs/tileops/benchmark/benchmark_base.py::_parse_msprof_roofline` +
`tileops/benchmark/msprof_roofline.py`）：

```
ratio = performance / min(AI * bw, computility)
      = max( bytes/(duration*bw_peak),  counted_flops/(duration*computility) )
```

实测（`profiles/roofline/r0_main/8bshort_fp16/`，8b-short fp16 baseline）：`AI = 1182.2 ops/byte`，
`bw = 1.75 TB/s` ⇒ `AI*bw = 2069 TOps/s` **≫** `computility = 299.36 TOps/s` ⇒ **绑定项是 compute**，
即 `Ratio = counted_flops / (duration × 299.36 TOps/s)`。`computility` 按 40 个 cube 单元计
（`4096 ops/cycle × 1.8 GHz × 40 = 294.9 TOps/s` + vector 4.42 TOps/s）。

- **(a) msprof 实采口径（唯一合法口径）**：`counted_flops` = 硬件计数器实际执行的 MAC，
  8b-short = **24.95 GF**、8b-long = **182.9 GF**（与解析值 24.2 / 177 GF 一致，偏差 <3%）。
  对应 ideal = **83.3 µs**（short）/ **611.0 µs**（long）。
- **(b) 任务书中的 718 µs**：对应 counted_flops ≈ 718 µs × 299.36 TOps/s ≈ **215 GF**，是 24.95 GF 的
  8.6×。该数来自「manifest 理论 flops」公式，而 msprof 只统计**实际执行**的 MAC，不统计被 causal 跳过的
  块。把被 mask 的块也真算一遍会让 counted_flops 与 duration 同比例增长，Ratio 不变 ⇒ **(b) 不能用来抬
  Ratio**。只有在「同一批 flops 用更高效率执行」时 Ratio 才涨。
- 设备实测 fp16 峰值：`torch.matmul(4096³)` = **270.1 TFLOPS**（同机探针）⇒ theoretical max Ratio
  = 270.1/299.36 = **90.2%**（100% 占满 cube 且零开销）。

### 0.1.2 逐 workload 估算与 40% 所需条件

| workload | counted GF | ideal(µs) | 40% 需 duration ≤ | baseline duration | 所需加速 | 当前 cube 占比 | 40% 需 cube 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8b-short fp16 | 24.95 | 83.3 | 208.2 | 5865.6 | 28.2x | 2.3% | ≥66% |
| 8b-short bf16 | 25.02 | 83.6 | 209.0 | 5875.5 | 28.1x | 2.3% | ≥66% |
| 8b-long fp16 | 182.91 | 611.0 | 1527.6 | 39864.7 | 26.1x | 2.5% | ≥64% |
| 8b-long bf16 | 183.46 | 612.8 | 1532.1 | 39865.4 | 26.0x | 2.5% | ≥64% |
| 70b-short fp16 | 24.94 | 83.3 | 208.2 | 6143.5 | 29.5x | 2.5% | ≥66% |
| 70b-short bf16 | 25.02 | 83.6 | 209.0 | 6159.4 | 29.5x | 2.5% | ≥66% |
| 70b-long fp16 | 182.90 | 611.0 | 1527.6 | 39969.9 | 26.2x | 2.5% | ≥64% |
| 70b-long bf16 | 183.46 | 612.8 | 1532.1 | 39986.7 | 26.1x | 2.5% | ≥64% |

「当前 cube 占比」= `Σ aic_cube_time / 核数 / Task Duration`（实测：短档 2739 µs/20 = 137 µs，长档同理）。

**可达上界裁决：40% 不可达；实测参照上界 ≈ 8.2%。** 依据（不是外推，是同算子、同工具链族的既有实现）：

- 本仓 `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §10.4 是**同一算子**
  （GroupedQueryAttention backward）在 910B2（24 核）/ tilelang `0.1.2+151c62a` / CANN 9.0.0 上的
  Stage-4 交付：3-pass 单引擎相位拆分后 8b-short **1016.8 µs**、8b-long **7518.6 µs**。
  换算 Ratio = 24.95 GF / (1016.8 µs × 299.36 TOps/s) = **8.2%**（long 同值 8.1%）——
  即**本仓该算子已知最优实现的 Ratio 上界 ≈ 8.2%**，40% 需要它的 **4.9x**。
- 同一参照的**融合基线**（未拆分、结构与本 kernel 同类）为 2011.6 / 13954.2 µs（24 核）；按核数归一
  （×20/24）≈ 2413 / 16745 µs ⇒ **本 kernel 改造前（5865.6 / 39864.7）比同算子融合基线慢 2.4x**，
  改造后（1995.8 / 14018.2，§4 Round 3）已与该归一基线持平/略优。⇒ 剩余可达空间 = 参照的"拆分"增益
  1.96x，即 Ratio ≈8.2% 封顶。
- 结论与结构性前提：要达到 40%（= 实测峰值的 44% 持续算力 = 本仓 attention 族最好效率 70 TFLOPS 的
  1.7x），必须同时做到 (i) cube 连续忙（去掉每 tile 的跨引擎握手，baseline cube 95% 墙钟在 `wait_id2`）、
  (ii) 消除 BSHD 行粒度访问（`dq` 逐行原子 51% + q/dO 跨步装载 33%）、(iii) cube 占比 ≥53%。
  (ii) 本轮已全部解决（§4 Round 3），(i) 部分解决（`wait_id2` 409→23.6 µs/block），但 cube 占比仍只有
  **5.5%**，剩余 70% 的 V 核墙钟是 `aiv_scalar`（每迭代发射 + Developer 双 AIV 冗余）⇒ (iii) 不可达。

  **本轮实测终值：8 档 Ratio 几何均值 1.4610% → 4.2375%（2.90x），距 40% 还差 9.4x。**
  剩余路径必须改 kernel 组成（多 pass 单引擎 + GM workspace + 单写者）——即 **op 契约/pipeline 拆分层的
  改动**，已按受控逆向反馈写入 `perf_feedback.md`（`[DESIGN_LIMIT]`，建议路由「附录补记」）。
  该反馈的**参照锚定门槛**已满足：同算子已有最优实现（3-pass 拆分，910B2）实测 1016.8 / 7518.6 µs
  ⇒ Ratio ≈8.2%，即设计层路径走完后仍只有目标的 1/4.9。

## 1. Workload inventory（primary）

来源：`examples/TileOPs/benchmarks/ops/bench_grouped_query_attention_bwd.py::test_gqa_bwd_bench`；
按 `tileops.manifest.load_workloads("GroupedQueryAttentionBwdOp")` 实跑展开（5 个 manifest workload ×
逐 `dtypes`）⇒ **9 个案例**，全部 `is_causal=True`，无 `bench_skip_reason`（`skipped` 为空）：

| workload_id | kind | q_shape | kv_shape | dtype |
|---|---|---|---|---|
| `gqa-bwd-smoke-float16` | smoke | [1,128,8,64] | [1,128,2,64] | fp16 |
| `llama-3.1-8b-short-{float16,bfloat16}` | tune | [4,512,32,128] | [4,512,8,128] | fp16/bf16 |
| `llama-3.1-8b-long-{float16,bfloat16}` | tune | [2,2048,32,128] | [2,2048,8,128] | fp16/bf16 |
| `llama-3.1-70b-short-{float16,bfloat16}` | tune | [2,512,64,128] | [2,512,8,128] | fp16/bf16 |
| `llama-3.1-70b-long-{float16,bfloat16}` | tune | [1,2048,64,128] | [1,2048,8,128] | fp16/bf16 |

- smoke 只做精度回归（`precision_pass=true`），不做性能采集。
- 排序（Phase 1，按实测耗时）：long 档（≈39.9 ms × 4）→ 70b-short（6.14/6.16 ms）→ 8b-short（5.87 ms）。
  long 与 short 的瓶颈结构相同（同一份 profile 结论），故实验用 8b-short fp16（编译/测量成本最低）
  做筛选、8b-long fp16 做长档确认，最终 8 档全量复测。

## 2. Performance Test Data（baseline = 第 1 轮 final `r2_fp32vcmp`）

测量命令模板（逐 workload 串行，禁止并发 profiling）：

```
msprof op --kernel-name=_gqa_bwd_main_mix_aic --output=profiles/baseline/round0/main/llama-3.1-8b-short-float16_r2_fp32vcmp_base \
  --launch-count=20 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline \
  python _bench_kernel.py --impl <candidate.py> --kernel main --B .. --S .. --H .. --H_kv .. --D .. --dtype .. \
  --causal 1 --iters 25   # 逐 workload 替换 --impl/--B/--S/--H/--H_kv/--D/--dtype
```

由 `perf_opt/run_experiments.py` 封装（本轮把 `Roofline` 加入 `--aic-metrics`，`visualize_data.bin` 因此可用；
`_ratio.py` 从 bin 的 `GM/L2 + GM Read + Write` 条目提取官方 Ratio）。

| workload_id | baseline Task Duration (us) | 官方 Ratio (%) | counted GF | ideal (µs) | L0 |
|---|---:|---:|---:|---:|---|
| llama-3.1-8b-short-float16 | 5865.56 | 1.4209 | 24.95 | 83.3 | PASS |
| llama-3.1-8b-short-bfloat16 | 5875.52 | 1.4227 | 25.02 | 83.6 | PASS |
| llama-3.1-8b-long-float16 | 39864.67 | 1.5327 | 182.91 | 611.0 | PASS |
| llama-3.1-8b-long-bfloat16 | 39865.44 | 1.5372 | 183.46 | 612.8 | PASS |
| llama-3.1-70b-short-float16 | 6143.50 | 1.3563 | 24.94 | 83.3 | PASS |
| llama-3.1-70b-short-bfloat16 | 6159.41 | 1.3569 | 25.02 | 83.6 | PASS |
| llama-3.1-70b-long-float16 | 39969.86 | 1.5286 | 182.90 | 611.0 | PASS |
| llama-3.1-70b-long-bfloat16 | 39986.66 | 1.5324 | 183.46 | 612.8 | PASS |
| **均值** | — | **1.4610** | — | — | 8/8 |

（与任务书给的 1.4568% 一致，差异来自 run 噪声。）smoke `gqa-bwd-smoke-float16` 只做精度回归。

### 2.1 设计估算 vs 实测（D-2 回填）

`DESIGN.md` 未给出 §1.6.0 形式的「设计期估算下界」行。以实测反推：baseline 24.95 GF / 5.866 ms =
**4.25 TFLOPS**，相对实测峰值 270 TFLOPS 为 **1.6%**，相对 manifest 理论 flops 口径的偏差 >2 个数量级，
失准项为**跨引擎握手 + BSHD 行粒度访存**（不是流量/容量）：见 §3。

## 3. 瓶颈定位（baseline profile，8b-short fp16）

`profiles/roofline/r0_main/8bshort_fp16/`，PipeUtilization 按 `sub_block_id` 分组（每组 5120 行 = 20 launch ×
256 block），下表为**每 block 均值**：

| 观测 | vector0 | vector1 | cube0 |
|---|---:|---:|---:|
| `*_time(us)`（核墙钟） | 431.3 | 425.2 | 431.0 |
| `aiv_mte3_time`（dq 原子） | **220.3（51%）** | — | — |
| `aiv_mte2_time`（q/dO 跨步装载） | **141.1（33%）** | 140.5 | 17.0 |
| `aiv_scalar_time` | 129.0 | 94.5 | 57.4 |
| `aiv_vec_time` | 28.6 | 28.8 | — |
| `aiv_scalar_mte2_stall_time` | 333.0 | 120.0 | — |
| `aic_cube_time` | — | — | **10.7（2.5%）** |
| `aic_fixpipe_time` | — | — | 33.5 |
| `aic_scalar_wait_id2_time`（cube 等 Vector） | — | — | **409.1（95%）** |

判读：

1. **cube 只忙 2.5%，95% 的 cube 墙钟在 `wait_id2`**（等 Vector 侧）——kernel 不是算力受限，是
   **每 tile 的跨引擎串行链**（5 个 `T.gemm` → 2 次 L0C→UB → ~12 条 v-op → 2 次 cast → 3 个 `T.gemm`）。
   每 (kv,q) 迭代单元墙钟 ≈ 12.7 µs，而单元内真实 cube 工作 ≈0.3 µs、vector 工作 ≈0.1 µs。
2. **Vector 侧 84% 是两项 BSHD 行粒度 GM 访问**：`dq` 逐行原子 51%（每行 512 B、行距 `H·D·4`=8 KB；
   1152 行/block）与 q/dO 跨步装载 33%（每行 256 B、行距 8 KB；2304 行/block）。
3. **两个 AIV 重复执行同一份 Vector 程序**（vector0/vector1 的 `mte2` 与 `vec` 计数几乎完全相同）——
   Developer 模式 Mix kernel 的 2× 冗余（pattern-library PL-1.13：`subid` 分片只适用 Expert 双 Scope，
   Developer 下 `subid` 恒为 0）。这解释了 33% 里有一半是纯冗余装载。

## 4. Iteration Log

每轮固定 base = 当前已合并 current best；分支从同一 base 派生、每分支只改一个主要优化点。
对比表数据取自 `perf_records.jsonl` 与 raw profile（B2 结构化回流）。

### Round 1（base = `r2_fp32vcmp`）

现象：见 §3（cube 2.5%、mte3 51%、mte2 33%、每迭代 ≈12.7 µs 固定开销）。候选优化点：
**q-tile 加宽 `BN` 32→64（仅 `dim=128` 档；`dim<=64` 原本已是 64）**——减半迭代单元数，即减半每迭代固定开销
（`T.gemm` 发射、fixpipe、v-op 发射、MTE2 描述符链），元素总量与 `dq` 原子行数不变。

预算核对（DESIGN §4.5 的静态加和曾被用作 `BN=64` 的否决依据）：实测 `BN=64, dim=128` 静态 UB 加和 ≈240 KB
`> 192 KB`，但 **BufferAllocation 按生命周期复用**，编译通过（workspace 155 648 B/block）⇒ 旧否决结论
（第 1 轮会话 §4 Round 3「BN=64 UB 预算 >192 KB 不可行」）**被推翻**：那是静态加和口径，不等于分配器的
实际占用。

| 分支 | Task Duration (us) | AICore / pipe（每 block 均值） | memory（mte2/mte3, us） | L0 | 结论 |
|---|---:|---|---|---|---|
| current best（baseline） | 5865.56 | cube 10.7 / scalar 129.0 / vec 28.6 / wait_id2 409.1 | 141.1 / 220.3 | PASS | — |
| **`r1_bn64`** | **5786.06（−1.4%）** | cube 8.6 / scalar 101.4 / vec 18.5 / wait_id2 380.2 | 139.6 / 217.9 | PASS（screen 6/6） | 保留（结构性简化，long 档 −4.3%） |

`r1_bn64` 在 8b-long fp16 上：39864.67 → **38142.97（−4.3%）**。
判读：scalar/fixpipe/cube 三项随迭代数下降（129→101、33.5→21.5、10.7→8.6），但 `mte3`（217.9）与
`mte2`（139.6）**几乎不动** ⇒ 证实这两项只与**行数**成正比，与每迭代指令数无关；`BN` 不是它们的杠杆。

### Round 2（base = `r1_bn64`）

现象：`mte3`+`mte2` 合计 78% 的 block 墙钟，行数 = `dq` 原子行 `B·H·S·ceil(S/BM)` + q/dO 装载行
（∝1/BM）。候选优化点（同一 base 派生）：

| 分支 | 唯一主要优化点 | 编译 | L0 | Task Duration (us) | 结论 |
|---|---|---|---|---:|---|
| `r2_bn64_pipe2` | `num_stages=1→2`（overlap q/dO 装载与计算） | PASS | 未单测（调度参数） | 5787.70（vs 5786.06，+0.03%） | `config_no_gain` → 回滚（与第 1 轮会话两次同结论一致：自动多缓冲在本结构不生效） |
| `r2_atom4h` | 把 `groups` 个 head 的 `dq` 合并成一条 2 KB 连续区域原子（原子行数 ÷4） | **FAIL（3 种写法）** | 未进入 | — | blocked：`[BN,groups,dim]` 暂存（64–128 KB）触发 `hivm.hir.copy: Unsupported copy from cc to cbuf` / `hivm.hir.load: src and dst shape should be the same`；UB 预算不允许该形态 |

**参照结构 diff 检查（T-1 必做，plateau 前）**：检索本仓同族实现，命中
`examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md`（**同一算子**）。其 Round 5 结论直接命中
本轮的 51% 项：**`dq` 用 rank-2 跨步区域原子**（把 `dq` 声明为 `(B·S, H·D)` 视图，一次
`T.atomic_add(dq[flat:flat+bn, h·D:(h+1)·D], dq_blk, [bn, dim])`），微基准 224 ns/行（逐行）→ **32 ns/行
（区域）**，端到端 −45.2%/−45.1%。这是本轮采用的 Round 3 候选（原始 diff 见其 §Round 5）。

### Round 3（base = `r1_bn64`）

现象：`mte3` 51% 由「逐行 1-D 原子」的单价（224 ns/行）决定，而同一片 GM 区域用 rank-2 区域形态只需
32 ns/行。候选优化点：**`dq` 改用 rank-2 跨步区域原子**——把 `dq` 形参声明为 `(B*S, H*D)` 视图（BSHD 连续
fp32 存储的精确视图），原子回写变为每 q block 一条 `[BN, dim]` 区域原子；对外签名保持不变（工厂内薄闭包做
`dq.view(rows, cols)`，与 TileOPs wrapper 的 `self.kernel(q,k,v,do,lse,delta,dq,dk,dv)` 调用完全兼容）。

前置排除：第 1 轮会话的 `r1_atomtile`（rank-reduced 4-D 形态 `T.atomic_add(dq[bz,q0,h,0], dq_ub, [BN,dim])`）
静默写错——那是**跨步区域被当连续**的误用；rank-2 视图形态的目标是真正的二维区域（行距 = H·D，与视图
pitch 一致），两者不是同一写法。本轮以 L0 全量 + L1 全量 + 边界用例验证。

| 分支 | Task Duration (us) | AICore / pipe（每 block 均值） | memory（mte2/mte3, us） | L0 | 结论 |
|---|---:|---|---|---|---|
| current best（`r1_bn64`） | 5786.06 / 38142.97 | cube 8.6 / scalar 101.4 / vec 18.5 / `wait_id2` 380.2 | 139.6 / 217.9 | PASS | — |
| **`r3_dqregion`** | **1995.76 / 14018.17（−65.5% / −64.8%）** | cube 8.6 / scalar 106.1 / vec 18.2 / **`wait_id2` 23.6** | **18.3 / 4.8** | PASS（`--level all`：L0 8/8、L1 12/12、L2 9 + Boundary 全绿） | **winner：−65.5%（2.90x）** |

winner 解析（`profiles/final/round3/main/llama-3.1-8b-short-float16_r3_dqregion_final/`，每 block 均值）：

| pipe | baseline | `r1_bn64` | **`r3_dqregion`** | 判读 |
|---|---:|---:|---:|---|
| `aiv_time`（V 核墙钟） | 431.3 | 425.0 | **150.3** | 主指标 |
| `aiv_mte3`（dq 原子） | 220.3 | 217.9 | **4.8** | 区域原子消除 45x |
| `aiv_mte2`（q/dO 装载） | 141.1 | 139.6 | **18.3** | 非直接改动 —— 原子流量消失后 GM 通道不再被挤占/阻塞 |
| `aiv_scalar` | 129.0 | 101.4 | **106.1** | **新瓶颈：70% 的 V 核墙钟（发射 + 2× AIV 冗余）** |
| `aiv_vec` | 28.6 | 18.5 | 18.2 | — |
| `aic_cube_time` | 10.7 | 8.6 | 8.6 | cube 实算不变 |
| `aic_scalar_wait_id2`（cube 等 Vector） | 409.1 | 380.2 | **23.6** | 跨引擎握手几乎消失 |

长档同构（8b-long fp16）：`aiv_time` 1545.4→545.0，`mte3` 788.6→16.7，`mte2` 517.9→65.6，
`scalar` 454.5→357.2，`wait_id2` 1512.7→30.6。

**`mte2` 附带下降 7.6x 的机制（实测推断，非直接改动）**：`dq` 区域原子把「每 q 行一条 512 B 读改写」变成
「每 q block 一条 `[BN,dim]` 区域原子」，GM 侧的原子流量从 151 MB/launch 降到同一区域的一次性写；原先
`aiv_scalar_mte2_stall_time` = 333 µs/block（scalar 等 MTE2）随原子流量消失降到 ~6 µs/block。
⇒ 该 kernel 此前是**GM 通道被原子流量挤占**导致的连带停顿，不是装载本身的带宽不够。

## 5. Autotune Log

未使用 autotune：可调参数只有 `BM/BN/num_stages`——`BM=64` 被 L0C 的 `dk_f/dv_f` fp32 预算锁死（§0.1.2），
`BN` 已用 32→64 单点实测（Round 1），`num_stages` 两次实测（第 1 轮会话 2 次 + 本轮 1 次）均 <3%。
本轮收益来自**访存形态**（rank-2 区域原子）与**tile 宽度**，不是参数搜索。

## 6. Final Summary

## 6.0 Round-3 Final Performance Test Data (historical, superseded by Round 20)

口径：`msprof op Task Duration(us)`（kernel-only，20 次 launch 中位，串行独占设备）；
官方 Ratio 由 `_ratio.py` 从同一 profile 的 `visualize_data.bin`（`GM/L2` + `GM Read + Write` 条目）提取，
定义见 `examples/TileOPs/tileops/benchmark/benchmark_base.py::_parse_msprof_roofline`。
raw profile：`perf_opt/profiles/final/round3/main/llama-3.1-8b-short-float16_r3_dqregion_final/`
（其余 workload 同目录、同后缀）。
final 交付件：`perf_opt/_gqa_bwd_kernel.py`，sha256 `815ad7f9be9d7fe300615e15082c6042f40e45af2e5d8eb2371681cc87da05c8`。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-short-float16 | 5865.56 | 1995.76 | r3_dqregion_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-short-bfloat16 | 5875.52 | 1993.72 | r3_dqregion_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-long-float16 | 39864.67 | 14018.17 | r3_dqregion_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-long-bfloat16 | 39865.44 | 14026.79 | r3_dqregion_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-short-float16 | 6143.50 | 1989.07 | r3_dqregion_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-short-bfloat16 | 6159.41 | 1995.26 | r3_dqregion_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-long-float16 | 39969.86 | 13998.19 | r3_dqregion_final |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-long-bfloat16 | 39986.66 | 13994.11 | r3_dqregion_final |

- 全部 final 时延来自**同一最终候选** `r3_dqregion_final`（同一 artifact、同一 sha256），在最终文件上以
  `launch-count=20` 重新采集；8 条 `phase=final` 记录的 `artifact_path`/`artifact_sha256` 完全一致。
- 均值：baseline 22966.33 µs → final 8001.38 µs（**几何均值 −65.9%**）。

### 官方口径 Ratio（同一次采集）

| workload_id | baseline Ratio (%) | final Ratio (%) | 倍数 | final perf (TOps/s) | counted GF |
|---|---:|---:|---:|---:|---:|
| llama-3.1-8b-short-float16 | 1.4209 | **4.1385** | 2.91x | 12.39 | 24.72 |
| llama-3.1-8b-short-bfloat16 | 1.4227 | **4.1550** | 2.92x | 12.44 | 24.80 |
| llama-3.1-8b-long-float16 | 1.5327 | **4.3190** | 2.82x | 12.93 | 181.23 |
| llama-3.1-8b-long-bfloat16 | 1.5372 | **4.3293** | 2.82x | 12.96 | 181.78 |
| llama-3.1-70b-short-float16 | 1.3563 | **4.1513** | 3.06x | 12.43 | 24.71 |
| llama-3.1-70b-short-bfloat16 | 1.3569 | **4.1508** | 3.06x | 12.42 | 24.79 |
| llama-3.1-70b-long-float16 | 1.5286 | **4.3248** | 2.83x | 12.95 | 181.22 |
| llama-3.1-70b-long-bfloat16 | 1.5324 | **4.3391** | 2.83x | 12.99 | 181.77 |
| **8 档几何均值** | **1.4610** | **4.2375** | **2.90x** | 12.68 | — |

⇒ 距离用户目标 `Ratio ≥ 40%` 仍差 **9.4x**。§0.1.2 的裁决（40% 不可达、实测参照上界 ≈8.2%）
在本轮 end-to-end 数据下得到进一步支持：本轮把「BSHD 行粒度访问」两项（`dq` 原子 + 被其挤占的装载通道）
全部消除后，Ratio 从 1.46% 升到 4.24%，而**新瓶颈变成 `aiv_scalar`（70% 的 V 核墙钟）**，cube 占比
仅从 2.3% 升到 5.5% —— 要再上一个数量级必须让 cube 连续忙（= 单引擎相位拆分 / 多 pass），属设计层改动。

### 6.1 最终精度回归

- `python perf_opt/_gqa_bwd_kernel.py --level all` → **RC=0，`All check passed!`**
  （日志 `logs/final_L0_L1_L2_boundary.log`）：
  - **L0 8/8 PASS**：causal smoke fp16/bf16、non-causal fp16/bf16、4 个 manifest fp16
  - **L1 12/12 PASS**：4 个 manifest bf16、GQA groups 1/2/4/8、`n_kv=1`/`n_kv=4` 的 dq 归约对拍、non-causal D=128、D=64 bf16
  - L2（9 例 + golden 交叉校验）+ Boundary 8 例（含 `atomic reduction stable`：两次同配置运行 dq 差 2.861e-06 / 尺度 2.303e+01）
- 关键点：`dq` 改为 rank-2 区域原子后，**L0+L1 全量在 causal/非 causal、D=64/128、groups=1/2/4/8、n_kv=1/4
  全部通过** —— 排除了第 1 轮会话 `r1_atomtile`「跨步目标被当连续」的静默错误形态（该形态是 rank-reduced
  4-D 写法；rank-2 视图写法的目标是真正的 2-D 区域，行距 = 视图 pitch）。

### 6.2 台账口径说明（append-only 遗留，如实登记）

`perf_records.jsonl` 的 `candidate_id=r1_bn64_final` 有 2 行（8b-short fp16/bf16，5795.4/5798.3 µs）
是在 `--level all` 之后、finals 采集中途**被我中断**（同一会话内改用了更强的 `r3_dqregion`）留下的，
其 artifact 已不是最终交付件。为满足「同一 kernel 的全部 final 记录必须来自同一候选且 sha 与最终文件一致」，
这 2 行**原值原样**移入 `history/perf_records_aborted_r1_bn64_final.jsonl` 留档（未改写任何测量值，
未删除审计痕迹），并在本节登记。

- final 候选: `r3_dqregion_final` = `perf_opt/_gqa_bwd_kernel.py`（sha256 `815ad7f9…`）
- `TUNED_*` 常量: `TUNED_BM = 64`、`TUNED_BN = 64`（baseline 为 `BN = 64 if dim <= 64 else 32`），
  另有结构项 `dq` rank-2 区域原子（无常量，见文件 docstring 与 `_gqa_bwd_kernel` 薄闭包）。
- smoke 精度: L0 8/8 + L1 12/12 PASS（§6.1）。
- **wrapper 兼容性（实测）**：`GroupedQueryAttentionBwdOp`（已激活 perf_opt 分支）在 smoke shape
  （B1/S128/H8/Hkv2/D64, fp16）上端到端跑通：`dq`→`[1,128,8,64] fp16`、`dk`/`dv`→`[1,128,2,64] fp16`，
  max|dq|=820.5，无异常。`dq` 的 rank-2 视图由工厂内的薄闭包完成（`dq.view(B*S, H*D)`），
  对外 9 个张量签名不变；`GQABwdKernel.forward` 只做 `self.kernel(*inputs)`，不做内省。
- **`TUNED_*` 与 wrapper 的关系**：wrapper 的 `default_config` 仍是基线值
  （`block_n = 64 if dim <= 64 else 32`），但它**不驱动** kernel（tiling 在工厂内由 `seq_len`/`dim` 推导）。
  本文件用模块级 `TUNED_BM = 64` / `TUNED_BN = 64` 暴露调优默认值，供 integration/Stage 5 的
  baseline↔perf_opt 切换块成对引用（本 Stage 不改 wrapper）。
- 中止原因: 目标 `Ratio ≥ 40%` **不可达**（§0.1.2 裁决 + §6 实测：本轮达 4.24%，参照上界 ≈8.2%）；
  剩余候选均需「改 op 契约 / pipeline 拆分」的**设计层**前提（见 `perf_feedback.md`，`[DESIGN_LIMIT]`）。

## 7. 辅助 kernel（preprocess / postprocess）

两个辅助 kernel 的 `perf_opt/` 副本与 Stage-3 基线**实现等价**（差异仅在 docstring，见
`diff <(sed 1,5d perf_opt/_flashattn_bwd_preprocess_kernel.py) <(sed 1,5d _flashattn_bwd_preprocess_kernel.py)`），
第 1 轮会话已在 8 个 tune workload 上逐档实测（差值 −2.2%…+1.2%，全部落在噪声内），结论 `no_gain`。
本轮不重新采数（无改动、成本 16 × ~2 min），`workload_inventory_aux.json` 按 `no_gain` 重建并引用
`history/perf_records_{preprocess,postprocess}_prev.jsonl` 作为实测依据。pipeline 时延构成中 main 占
99.0%–99.7%，端到端瓶颈即 primary。

## 8. Skill Retrospective

见 §9。

## 9. Skill Retrospective（复盘）

### 9.1 本轮事实与价值点（vp_type + 证据三件套）

| ID | vp_type | 结论 | evidence（实测） | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-A `dq-rank2-region-atomic` | D（代价常数/模式） | BSHD 的 `dq` 跨核归约必须走 **rank-2 跨步区域原子**（`dq` 声明为 `(B·S,H·D)` 视图 + 薄闭包保持对外签名），逐行 1-D 形态有 7× 单价惩罚（224 ns/行 vs 32 ns/行）；本 kernel 上该项是 51% 的 block 墙钟，改后 `aiv_mte3` 217.9→**4.8 µs/block**、`aiv_mte2` 139.6→**18.3**（GM 通道不再被原子流量挤占）、`aic_scalar_wait_id2` 380.2→**23.6**，端到端 5865.6→**1995.8 µs（−66%，Ratio 1.42%→4.14%）** | `perf_records.jsonl` round1 vs round3 同 workload；`profiles/baseline/round0/*` vs `profiles/final/round3/*` 的 `aiv_mte3_time`；同算子先例 `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §Round 5（−45.2%） | `_gqa_bwd_kernel_r3_dqregion.py` 相对 `_gqa_bwd_kernel_r1_bn64.py` 的最小 diff（声参 2-D + 一条 `T.atomic_add` + 闭包） | CANN 9.0.0 / Ascend910B3 / tilelang `0.1.2+9738284cd6`（2026-10-31） |
| VP-B `static-ub-sum-not-a-budget` | D（证伪更正） | **静态 UB 加和不是 UB 预算**：`BN=64, dim=128` 静态 240 KB > 192 KB 仍可编译（分配器按生命周期复用）——第 1 轮会话据此否决 `BN=64` 的结论被推翻 | 本轮 `_compile_probe*` 编译成功 + workspace 155 648 B/block；`BN=64` 实测 −1.4%/−4.3% | `_gqa_bwd_kernel_r1_bn64.py`（相对 base 的单行 `BN` 改动） | 同上 |
| VP-C `row-granular-invariance` | D（诊断方法） | BSHD 行粒度访存的成本只与**行数**成正比：`BN` 32→64 后 `mte3` 220.3→217.9、`mte2` 141.1→139.6 几乎不变，而 scalar/fixpipe/cube 减半 ⇒ 用「`BN` 扫描 + pipe 分项」即可把「行粒度」与「指令发射」两类瓶颈分离；反之行数不变时只能**换形态**（逐行→区域原子）才能把 `mte3` 打掉 45x | 本轮 Round 1 对比表（4 个 pipe 分项） | 同上两文件 | 同上 |
| BP_4 | R（流程） | Phase 0 的「kb/仓库同族检索」应**显式包含同算子在其他 example 目录的 perf_opt**：本轮 51% 的单点收益直接来自 `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/`，而该档案在首轮会话（以及本轮的 `references/pattern-library` 索引）里都没有被引用——索引只覆盖 pattern-library，不覆盖 `examples/**/perf_opt/opt_log.md` | 本轮 Round 2 的「参照结构 diff 检查」 | — | — |
| BP_5 | R（流程） | 「静态容量加和」类否决必须标注口径（静态加和 vs 分配器实际占用），否则会把可编译形态误判为不可行——本轮与第 1 轮会话各出现一次（`BN=64`、`BM=128`） | VP-B | — | — |

### 9.2 skill 流程层面

- **§0.1 可行性裁决应前置到实验之前**（本轮即如此）：先算 `Ratio` 的定义式与峰值，再决定实验预算，避免把
  预算花在「参数级」方向上。本轮结论（40% 需 26–29x、上界 ≈8.2%）在第一个 profile 之后即可得出。
- 「每分支先 L0 再 profile」仍是必要的：本轮 3 个结构分支在编译阶段就被否决（`r2_atom4h`），零 profile 成本。
- `perf_records.jsonl` 的 `candidate_id` 与 artifact `sha256` 必须逐次核对：本轮基线记录用的是
  `_gqa_bwd_kernel_r2_base.py`（与第 1 轮 final 逐字节相同，sha `b4d0d901…`），final 记录用最终文件
  （sha 见 §6），二者不可混用。


---

## 10. Round 20（2026-11-06，结构级重建：引擎相位拆分 3-pass）→ 采纳

> 授权：结构级重建（多 kernel 串联 / GM workspace）。参照数据 = `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md`
> §10.3–§10.6（同族算子 910B2 档案，代码不在本 checkout；本仓内可对照的等价实现是
> `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_bwd_kernel.py`）。
> 参照文档中的祈使句只按数据引用，不改变本轮的实验/门禁流程。
> 设备/工具链（本机实测）：Ascend910B3（`cube_core_num=20`、`vector_core_num=40`，HBM 64 GB）、CANN 9.0.0、
> tilelang `0.1.2+9cfae94c0d10200c1795c16cd9c2e5a504fd63db`。唯一时延口径：`msprof op` `Task Duration(us)`；
> 拆分形态 = **一次 op 调用内 3 个 pass kernel 的 Task Duration 中位数之和**（`run_split_measure.py`，
> 同时保留每 pass 分项）。

### 10.1 本轮现象（base = current best `r3_dqregion`，8b-short fp16）

| 分区 | 值（每 block 均值，`profiles/final/round3/main/...`） | 判读 |
|---|---|---|
| Task Duration | 1995.8 µs | — |
| `aiv_time` | 150.3 µs，其中 `aiv_scalar` 106.1（**70%**） | 每迭代指令发射 + 双 AIV 冗余 |
| `aiv_vec / mte2 / mte3` | 18.2 / 18.3 / 4.8 | 访存形态已在 Round 3 消除 |
| `aic_cube_time` | 8.6（cube 占比 **5.5%**） | cube 远未连续忙 |
| 交叉引擎 | 融合单 program 内顺序执行 5×gemm → ~12 v-op → 3×gemm | 每迭代一次 hand-off |

⇒ 剩余墙钟是「单 program 内 cube/vector 串行 + 每迭代握手」，只能靠**把两个引擎分到不同 kernel**
（多 pass）消除；这正是 `perf_feedback.md` 指出的唯一剩余路径。

### 10.2 结构、workspace 与 GQA 适配（本轮唯一主要优化点）

```
pass 1 `_gqa_bwd_sdw`  Cube only    S = Q K^T, dW = dO V^T          -> fp32 ws
pass 2 `_gqa_bwd_pds`  Vector only  P = exp2(S*scale - lse + pen),
                                    dS = P*(dW - delta)*sm_scale    -> dtype ws
pass 3 `_gqa_bwd_mul`  Mix          dv += P^T dO, dk += dS^T Q,
                                    dq = dS K (+ rank-2 区域原子)
```

- **GQA 适配（参照是 MHA `groups=1`）**：三个 pass 的 grid owner 统一为 `(b, kv_head, kv_block)`
  （`nblk = B*Hkv*n_kv`，1-D 非持久 `T.Kernel`）；共享同一 KV head 的 `groups` 个 q head 用**编译期
  Python `range(groups)` 在核内展开**。这样做有三个直接后果：
  1. `dk/dv` 在 pass 3 内跨 group 累加后**一次 `T.copy` 写出（单写者、零原子）**——比 MHA 参照
     `groups>1` 时的行原子更省；
  2. K/V 每 pass 只装载一次（不是每 q head 一次）；
  3. workspace slot 仍按 q head 划分：`slot = ((b*H + h)*n_kv + jb)*n_q + i`（P/dS 依赖 lse/delta/q）。
- **workspace 尺寸**：`ntile = B*H*n_kv*n_q`；`ws_s/ws_w` fp32 `[ntile,BN,BM]`、`ws_p/ws_ds` dtype 同形。
  910B3 HBM 64 GB，最大 manifest case（8b-long）为 `2×1.07 GB + 2×0.54 GB = 3.22 GB`，因此**不沿 KV 轴
  chunk**（`cb = n_kv`、一次 op 调用 3 个 launch）。参照的 256 MB 预算是 910B2 的 HBM 预算，不是算法约束；
  chunk 数不进入任何数学。`profiles` 里每 case 的 kernel 数恒为 3（无 per-chunk 乘子）。
- **每迭代 `T.pipe_barrier("PIPE_ALL")`**：pass 1 与 pass 2 迭代末各一条（纯 Vector pass 的 UB scratch
  跨迭代 WAR 不被自动同步覆盖；本轮沿用参照的必需要求）。pass 2 额外在 P 的 dtype 回写后加一条 barrier
  （同一 dtype scratch 随后写 dS）。
- **尾块 / 边界**：`_split_geom` 用与 Stage-3 相同的 `gcd(seq_len,64)`，所有 L2 尾块形状（96/160/224/320）
  都走拆分并通过；唯一例外 `n_kv==1`（`S==block_m`）保留 Stage-3 融合 kernel——该形状让
  `bishengir-compile` SIGSEGV（编译器缺陷，去掉 dq 原子后仍复现）。
- **数值**：pass 2 逐元素链与融合版逐条对应（`vexp2` + fp32 `vcmp/vselect` 掩码），fp32 S/dW 经 GM
  往返是精确的，故 P/dS 与融合版逐位一致；`dq` 的原子发射序不变。

### 10.3 候选 vs current best（B2 对比表）

| # | 分支（文件） | 唯一主要优化点 | 编译 | L0 | Task Duration(us) short / long fp16 | 利用率（msprof，short） | 工作量守恒 |
|---|---|---|---|---|---|---|---|
| r3 | `_gqa_bwd_kernel_r3_dqregion_baseline.py`（current best，只读） | 本轮固定臂 | PASS | PASS | 1994.9 / 14019.6 | `aiv_scalar` 70%，cube 5.5% | — |
| r20 | `_gqa_bwd_kernel_round20_3pass.py` → 交付件 | 引擎相位拆分（3 kernel + GM workspace + 每迭代 barrier） | PASS | PASS（`--level all` EXIT=0） | **1203.6 / 7593.3** | sdw cube 1.3%；pds vec 25.3% / scalar 50.8%；mul cube 0.9% | 5 GEMM / 1 exp / 13 v-op 不变；新增 S/dW+P/dS 往返 12 B/score-elem + pass3 重读 q/dO/k |

- 单点纪律：r20 只改「引擎相位归属」，不改 tile、不改 mask、不改归约顺序。`BM=BN=64` 与 Stage-3 相同。
- 编译缺陷记录：`n_kv==1` 形状 `bishengir-compile` SIGSEGV（本轮实测，见 10.6），走融合回退。

### 10.4 3-run A/B 交错复核（同 session，逐对交替）

| workload | pair1 base / 3-pass | pair2 | pair3 | 中位比 |
|---|---|---|---|---|
| llama-3.1-8b-short/fp16 | 1994.9 / 1210.0 | 1995.4 / 1198.0 | 1993.5 / 1203.6 | **0.6035（−39.7%）** |
| llama-3.1-8b-long/fp16 | 14016.5 / 7595.9 | 14022.4 / 7593.3 | 14019.6 / 7588.7 | **0.5416（−45.8%）** |

三对同向、无交叉；收益远超噪声阈值（3%）。

### 10.5 实现过程中修掉的两个真实缺陷（如实登记）

1. **P 被 dS 覆盖（源码顺序错误）**：首版把 `T.vcast(acc, p16)` 放在 dS 算术之后，IR 里两次
   `vcast` 都读到 dS，`ws_p` 与 `ws_ds` 逐位相同；逐 tile 对拍 P 定位（`dq/dk` 正确而 `dv` 错）。
   修正为「exp2 → 写 P → 再算 dS」后 P 误差 3.9e-4、dS 5.3e-4。
2. **lse/delta 广播轴**：pass 2 的 score tile 是 `[q, kv]`，lse/delta 必须是 `(BN,1)` 列向量；
   沿用融合版的 `[1,BN]` 会在错误轴上广播（P 最大 3.2 vs 期望 0.73）。改为 `(BN,1)` +
   `T.copy(lse[bz,h,q0:q0+BN], lse_s[0:BN,0])`。

### 10.6 否决 / 受限项

| 项 | 结果 | 结论 |
|---|---|---|
| `n_kv==1` 走拆分 | `bishengir-compile` SIGSEGV（`_gqa_bwd_mul`，group 循环被 reroll 成 `scf.for`） | 该形状保留融合；拆分只承诺其余形状 |
| 去掉 dq 区域原子（n_kv==1 改为 `T.copy`） | 同样 SIGSEGV | 崩溃不在原子，已回退该分支 |
| KV 轴 chunk（参照的 256 MB 预算） | 910B3 64 GB 下无必要；单 chunk 并行度更好 | 不采用 |
| 增大 tile `BM=128` | L0C `dk_f/dv_f` 各 64 KB 超限（Stage-4 已记录） | 不采用 |

### 10.7 精度回归

| 项 | 结果 |
|---|---|
| `perf_opt/_gqa_bwd_kernel.py --level all` | **EXIT=0 / `All check passed!`**：L0 全 PASS（含 8 manifest 形状）、L1 全 PASS（含 groups=1/2/4/8、n_kv=1、非 causal、bf16）、L2 / Boundary 非阻塞项无 FAIL（`logs/round20/cand3pass_all.log`） |
| 拆分后 8b-short fp16 L0 max_diff | 6.31e-03（与融合版同量级；tolerance 为 tensor-scale 的 1e-3） |
| 结构重写对数值的影响 | 元素级算术不变（同 5 GEMM、同 exp、同 dtype 回落）；P/dS 经 fp32 GM 往返精确 |

## Final Performance Test Data

口径：`msprof op Task Duration(us)`；拆分形态 = 同一 op 调用内 `sdw + pds + mul` 三个 kernel 的中位数之和
（每 case kernel 数 = 3；n_chunks 不使用）。baseline 与 final 均为本 session 同机重测。
final 交付件：`perf_opt/_gqa_bwd_kernel.py`，sha256 `26edd4c1d70e2db1dc77e7d1cfb0f4f78556bb73e5cef6e1b4ab906c098cc627`。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-short-float16 | 1994.85 | 1203.62 | final-r20-3pass |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-short-bfloat16 | 1991.91 | 1214.44 | final-r20-3pass |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-long-float16 | 14019.55 | 7593.34 | final-r20-3pass |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-8b-long-bfloat16 | 14018.17 | 7649.12 | final-r20-3pass |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-short-float16 | 1991.86 | 1183.52 | final-r20-3pass |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-short-bfloat16 | 1994.54 | 1189.58 | final-r20-3pass |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-long-float16 | 13992.83 | 7613.16 | final-r20-3pass |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/_gqa_bwd_kernel.py::_gqa_bwd_kernel::_gqa_bwd_main | llama-3.1-70b-long-bfloat16 | 13983.87 | 7633.85 | final-r20-3pass |

### 10.8 官方口径 Ratio（同一批 profile 的 `GM/L2 + GM Read + Write`）

拆分形态的 Ratio = Σ(每 pass 的 counted flops) / (Σ 每 pass Task Duration × computility)；
`computility = 299.36 TOps/s`（40 cube 单元口径），每 pass 的 counted flops 来自该 pass 的 roofline bin。

| workload_id | baseline Ratio (%) | final Ratio (%) | 倍数 | final counted GF |
|---|---:|---:|---:|---:|
| llama-3.1-8b-short-float16 | 4.1404 | **6.7476** | 1.63x | 24.43 |
| llama-3.1-8b-short-bfloat16 | 4.1588 | **6.7333** | 1.62x | 24.47 |
| llama-3.1-8b-long-float16 | 4.3196 | **7.8812** | 1.82x | 179.21 |
| llama-3.1-8b-long-bfloat16 | 4.3319 | **7.8384** | 1.81x | 179.49 |
| llama-3.1-70b-short-float16 | 4.1455 | **6.8979** | 1.66x | 24.43 |
| llama-3.1-70b-short-bfloat16 | 4.1522 | **6.8733** | 1.66x | 24.47 |
| llama-3.1-70b-long-float16 | 4.3265 | **7.8631** | 1.82x | 179.21 |
| llama-3.1-70b-long-bfloat16 | 4.3422 | **7.8539** | 1.81x | 179.49 |
| **8 档几何均值** | **4.2256** | **7.3172** | **1.73x** | — |

⇒ 达成本轮目标区间 6.5%–8.2%（几何均值 7.32%，相对 Stage-4 起点 4.24% 为 1.73x）。

### 10.9 中止条件与预算

- success：8 例 tune workload 全部走拆分路径（除 `n_kv==1` 边界），几何均值 7.32%，单例无回退。
- 预算：结构分支 1 个（3-pass，含 2 次源码缺陷修复）+ 探针/边界 2 个（`n_kv==1` 原子变体、tail 形状）
  ≤ `max_stage4_experiments=3`。
- `[DESIGN_LIMIT]`：不新增。本轮已把 `perf_feedback.md` 指出的唯一剩余设计层路径执行完，剩余 gap 来自
  每 pass 的指令/带宽成本与编译器同步形态，不是新的设计层假设；既有 `perf_feedback.md` 保留不动（其结论
  「3-pass 上界 ≈8.2%」在本轮 7.32% 下仍然成立）。

### 10.10 Round 20 Skill Retrospective

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 结构迁移 | **GQA 的 3-pass 不能照搬 MHA 的 grid**：owner 取 `(b, kv_head, kv_block)` 并在核内展开 groups，才能让 dk/dv 单写者且 K/V 只读一次；按 q head 建 grid 会引入 groups 倍 dk/dv 原子 | 本轮 `_gqa_bwd_mul` 无原子、dk/dv 单写；参照 MHA `atomic_dk=groups>1` | `attention.md` 增「GQA/MHA 的 split grid owner」条目 | P |
| 数值/顺序 | **P 的 dtype 回写必须在 dS 算术之前**：`vcast(acc,p16)` 若放在 dS 之后，IR 两次 cast 都读 dS，`ws_p==ws_ds` 而端到端 dq/dk 仍「看起来对」 | 逐 tile 对拍 P：err 2.5；修正后 3.9e-4 | `traps-compiler.md` 增「中间量转存顺序」trap + repro | D |
| 广播轴 | **[q,kv] 布局下 lse/delta 必须是 `(BN,1)`**：融合版 `[1,BN]` 的广播轴在转置后的 score tile 上是错的 | P 最大 3.24 vs 0.73 | `layout.md` 增「广播轴随 score tile 布局翻转」条目 | D |
| 编译器边界 | `n_kv==1` 的 3-pass `mul` 让 bishengir-compile SIGSEGV；去掉 dq 原子无效 | `logs/round20/cand3pass_all.log` + 最小复现 `logs/round20/nkv1` | `iteration-diagnosis.md` 增「结构重写的边界形状回退纪律」 | R |
| 测量口径 | 多 kernel 形态必须逐 case 记录 kernel 数并求和；`--kernel-name` 单 filter 会只抓一个 pass | `run_split_measure.py`；本 case kernel 数恒 3 | 沿用 r11 的 R 类条目（已存在） | R |

## Round 21（2026-11-07，BN widening Tier-0 探针）→ 实测 winner 22.1% / 28.9%，**未合入（wallclock 预算）**

> 授权范围：本轮只探「3-pass 结构下加宽 q-tile（BN），BM 保持 64」。唯一时延口径 `msprof op Task Duration(us)`；
> 3-pass 形态的合计 = 同一 op 调用内 `sdw + pds + mul` 三个 kernel 的 Task Duration 中位数之和（`run_split_measure.py`）。
> 设备/工具链（本机实测）：Ascend910B3 ×1（20 AIC / 40 AIV，HBM 64 GB，独占串行）、CANN 9.0.0、
> tilelang `0.1.2+9cfae94c0d10200c1795c16cd9c2e5a504fd63db`（与 Round 20 同一版本戳）、msprof 25.5.0。

### 21.0 假设与纸上预算（先于设备）

假设：3-pass 的每 pass 墙钟由**每迭代固定成本**（跨引擎握手、barrier、描述符链、原子发射）主导，
加宽 BN 把 q 方向迭代数减半，从而摊薄这些成本；BM 不动（D=128 时 `dk_f + dv_f = 2·BM·D` 已占 L0C 的 50%）。

预算表（BM=64，D=128，fp32=4 B，dtype=2 B；L0C 上限 32768 个 fp32 = 128 KB；UB 192 KB/AIV；
pds 的 static = 全部声明缓冲，live = 去掉 `qmat/kmat`（建完 `dif` 即死）后的保守活跃集）：

| pass | 计量 | BN=64 | BN=96 | BN=128 |
|---|---|---|---|---|
| pass1 `_gqa_bwd_sdw`（Cube） | L0C `2·BN·BM` | 8192（25.0%） | 12288（37.5%） | 16384（50.0%） |
| | UB `2·BM·D·2 + 2·BN·D·2 + 2·BN·BM·4` | 96.0 KB（50.0%） | 128.0 KB（66.7%） | 160.0 KB（83.3%） |
| pass2 `_gqa_bwd_pds`（Vector） | L0C | — | — | — |
| | UB static / live | 140.5（73.2%）/ 108.5（56.5%） | 210.8（109.8%）/ 162.8（84.8%） | **281.0（146.4%）/ 217.0（113.0%）** |
| pass3 `_gqa_bwd_mul`（Mix） | L0C `2·BM·D + BN·D` | 24576（75.0%） | 28672（87.5%） | **32768（100.0%，零余量）** |
| | UB `2·BN·BM·2 + 2·BN·D·2 + BM·D·2 + BN·D·4 + 2·BM·D·4` | 160.0（83.3%） | 200.0（104.2%） | 240.0（125.0%） |

**可行性附注（本轮新增的硬约束，先于任何实验成立）**：BN 必须整除 `seq_len`（`_split_geom` 的
`block_n = TUNED_BN if seq_len % TUNED_BN == 0 else gcd(seq_len, TUNED_BN)`）。全部 8 个 tune workload 的
`seq_len ∈ {512, 2048}`（2 的幂）⇒ **BN=96 在 512 与 2048 上都会回退成 gcd=32（是收窄，不是加宽）**，
BN=96 这一支在真实 workload 上不可评估；唯一可评估的加宽点是 **BN=128**（512%、2048% 整除）。
因此「优先 BN=96」在纸上即被否；BN=128 的两个风险点（pds UB 146%/113%、mul L0C 100%）由编译探针裁决。

### 21.1 实验 1：uniform BN=128（只改 BN，单点）→ blocked（pds UB 溢出）

候选 `_gqa_bwd_kernel_r21_bn128.py`（sha256 `8fccea9f…c91147f`，与 current best 逐字节相同，仅 `TUNED_BN: 64 → 128`）。
逐 pass 独立进程编译（`_r21_compile_probe.py`，每次调用外套 `timeout 300s`，禁止无上限等待），shape = 8b-short fp16：

| pass | rc | wall | 结果 / 失败签名 |
|---|---|---|---|
| sdw | 0 | 67 s | COMPILE_OK 52.8 s（L0C 50%、UB 83% 如预算） |
| **pds** | **1** | 16 s | **`ub overflow, requires 1871872 bits while 1572864 bits available!`**（228.5 KB 需求 vs 192 KB），`Failed to run BiShengHIR pipeline` |
| mul | 0 | 61 s | COMPILE_OK 45.4 s，`_gqa_bwd_mul_infer_workspace_shape_function() = 65536 bytes/block` |

裁决：uniform BN=128 **blocked**，且阻塞点在 **pass 2（纯 Vector pass）的 UB**，不在 pass 3 的 L0C——
**mul 在 L0C 恰好 32768/32768（100.0%）时被编译器接受**（本机实测），即「L0C 100% 零余量必失败」的先验被证伪
（见 21.6）。不重试该 case（纪律：单 case 失败即判 blocked）。

### 21.2 实验 2：capacity-partitioned BN=128（同假设的可编译形态）

候选 `_gqa_bwd_kernel_r21_bn128_hpds.py`（sha256 `0ff4a4f3…4542c20f`）。**单一主要优化点不变**（加宽 q-tile），
只在容量受限处做必要分区：`sdw` 与 `mul` 与 workspace slot 用 BN=128；`pds` 保持 64 行子块，在每个 slot 内
编译期展开 `n_sub = block_n // 64` 个子块（`q0 = i·128 + s·64`）。`pds` 的 UB 因此与 Round 20 完全相同
（acc / dW-delta / vexp2 scratch / mask 算子全是 `[64, 64]`），逐 score 元素的算术、tile 形状、元素顺序不变；
BM / mask / 归约顺序均未改动。

| pass | rc | wall | 结果 |
|---|---|---|---|
| sdw | 0 | 61 s | COMPILE_OK 45.1 s |
| pds | 0 | 62 s | COMPILE_OK 45.5 s（UB 回落到 140.5 KB static） |
| mul | 0 | 59 s | COMPILE_OK 43.8 s（L0C 100.0%） |

L0（阻塞）：`python _gqa_bwd_kernel_r21_bn128_hpds.py --level L0` → rc=0，wall 238 s，**8/8 PASS**；
8b-short fp16 `max_diff = 6.305695e-03`（与 Round 20 交付件同形状的 6.31e-03 一致）；
8b-long 1.387e-02、70b-short 1.023e-02、70b-long 1.945e-02，均在 tensor-scale 容差内。
`n_kv==1` 的 fused 回退路径代码未改（`TUNED_BN=128` 在 S=64 上 gcd 回落到 64，与原值相同），但**本轮未重跑该形状**（见 21.7）。

### 21.3 候选 vs current best（B2 对比表）

臂：base = `perf_opt/_gqa_bwd_kernel.py`（sha256 `26edd4c1…98cc627`，只读）；cand = `_gqa_bwd_kernel_r21_bn128_hpds.py`。
smoke `gqa-bwd-smoke-float16` 只做精度，不采集性能（inventory 约定），本轮 L0 已覆盖。

| workload | 臂 | AICore / Vector 利用率（msprof PipeUtilization，每 pass） | memory 指标（mte2 / mte3） | L0 | Task Duration(us) |
|---|---|---|---|---|---|
| 8b-short fp16 | base | sdw cube 1.32% / pds vec 25.31%·scalar 50.78% / mul cube 0.89% | sdw 0/0；pds 15.18% / 10.65%；mul 1.25% / 0.06% | PASS | 1207.90（r3 实测；3-run 中位 1203.66） |
| 8b-short fp16 | **cand** | sdw cube 1.43% / pds vec 27.40%·scalar 46.47% / mul cube **1.51%** | sdw 0/0；pds 16.43% / 11.55%；mul 1.42% / 0.12% | PASS | **937.12（3-run 中位）** |
| 8b-long fp16 | base | sdw cube 0.71% / pds vec 34.24%·scalar 26.56% / mul cube 0.47% | pds 26.58% / 14.88%；mul 0.64% / 0.01% | PASS | 7597.55（r1；3-run 中位 7597.55） |
| 8b-long fp16 | **cand** | sdw cube 0.78% / pds vec 35.00%·scalar 25.02% / mul cube **0.82%** | pds 27.17% / 15.16%；mul 0.77% / 0.02% | PASS | **5399.83（r1；3-run 中位 5403.41）** |

逐 pass 分解（同一对 run 的 per-pass 中位数之和；short 用 r3、long 用 r1）：

| pass | short base | short cand | Δ | long base | long cand | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `sdw`（Cube） | 260.07 | 233.40 | **−10.3%** | 1763.84 | 1455.47 | **−17.5%** |
| `pds`（Vector） | 370.41 | 373.81 | +0.9% | 1830.42 | 1870.22 | +2.2% |
| `mul`（Mix） | 577.43 | 329.91 | **−42.9%** | 4003.30 | 2074.14 | **−48.2%** |
| 合计 | 1207.90 | 937.12 | **−22.4%** | 7597.55 | 5399.83 | **−28.9%** |

`pds` 的 +1~2% 是分区设计的**预期结果**（它仍按 64 行子块跑，迭代数不变），非回退来源。

### 21.4 3-run 交错 A/B（同 session，逐对交替 base→cand / cand→base / base→cand）

| workload | pair1 base / cand | pair2 | pair3 | 中位比 | 同向 |
|---|---|---|---|---|---|
| llama-3.1-8b-short/fp16 | 1202.20 / 935.42 | 1203.66 / 938.48 | 1207.90 / 937.12 | 1203.66 / **937.12（−22.14%）** | **3/3** |
| llama-3.1-8b-long/fp16 | 7597.55 / 5399.83 | 7593.63 / 5403.41 | 7608.17 / 5404.55 | 7597.55 / **5403.41（−28.88%）** | **3/3** |

收益远超噪声阈值（3%），无交叉。原始 profile：`profiles/round21/ab/<workload>_{base,cand}_r{1,2,3}/`；
汇总 `logs/round21/ab_pairs.json`；测量参数：inner iters=20（15 测量 + 5 warm-up）、`--launch-count=45`
（15 次调用 × 3 kernel）、每 case kernel 数 = 3（无 per-chunk 乘子）。

### 21.5 裁决：`winner_measured`（采纳门槛数字上已满足）但 **未合入，best 不变**

- 门槛核对：8b-short −22.14%、8b-long −28.88%，均 ≥3% 且 3/3 同向，L0 8/8 PASS ⇒ **数字门槛通过**。
- 未合入的唯一原因是**剩余预算**：本 dispatch `max_wallclock_s=5400` 已消耗 ≈5100 s（本轮编译探针 6 次、
  L0 238 s、A/B 12 次 msprof 全部串行），而采纳路径还需
  ① `--level all` 精度回归（L0+L1+L2+boundary，新进程 `tilelang.cache.clear_cache()` 会重编译，预估 900–1700 s）、
  ② 其余 6 个 tune workload 的 `phase=final` 复测（每 run ≈230 s ⇒ 1400 s）、③ best 翻转 + 账目核对。
  粗估 3000–3600 s > 剩余量，若强行半途翻转会造成 `perf_records_mismatch`（比不合入更差）。
- 因此按纪律执行「不采纳 → 不修改 `perf_opt/_gqa_bwd_kernel.py`、不写 invalid record」，只追加两条
  `phase=candidate` 记录（artifact sha 指向真实候选文件），best 与 Final Summary 保持 Round 20 值。
- 实验预算：exp1（uniform BN=128，编译 blocked）+ exp2（capacity-partitioned，编译+L0+A/B）
  = 2 ≤ `max_stage4_experiments=2`；未超发。

### 21.6 本轮的证伪 / 更正（D 类，附证据）

1. **「L0C 100% 零余量必然编译失败」被证伪**：mul 在 BN=128 时 L0C 累加器恰好 `2·64·128 + 128·128 = 32768`
   个 fp32（128 KB 的 100.0%），`bishengir-compile` **接受**并给出 `workspace = 65536 bytes/block`
   （`logs/round21/probe_bn128_mul.log`）。⇒ L0C 判据是「≤ 32768」，等号可用；真正的窄口是 **UB**。
2. **BN 加宽的窄口在纯 Vector pass 的 UB，不在 Cube 侧**：同一 BN=128 下 sdw（UB 83%、L0C 50%）与
   mul（UB 125% 静态、L0C 100%）均编译通过，只有 pds 报 `ub overflow 1871872 / 1572864 bits`。
   实测量（228.5 KB）与我给的 live 估算（217.0 KB）同量级、比 static 估算（281.0 KB）小 52.5 KB：
   ⇒ VP-2026-0169（静态加和 ≠ 分配器占用，按生命周期复用）方向正确，但**复用只能回收 ~1.5 个 tile，
   不能把 146% static 压到 100% 以下**；「先给 static 上界、再以编译结果为准」的做法本轮再次有效。
3. **BN 的可选集合由 `seq_len` 的整除性决定，不由容量决定**：{512, 2048} 上只有 128（和 256，容量不可行）
   是合法加宽点；BN=96 会静默回退到 32。⇒ 以后讨论「BN=96/128」必须先过整除性筛选。
4. **加宽的收益全部来自迭代数减半的固定成本**：mul −42.9%/−48.2%，sdw −10.3%/−17.5%，而元素总量、
   workspace 字节数、dq 原子字节数都不变（`ntile×BN×BM` 恒定）⇒ 印证 Round 20 的「每迭代发射地板」判断，
   且该地板在 mul 里比 sdw 里更厚（mul 每迭代 3 GEMM + 区域原子 + 握手链）。

### 21.7 未验证项与本轮之后最值得做的两件事

未验证（不得当作已通过）：候选的 **bf16**（L1 四档）、**70b** 两档、**非 causal**、**tail/gcd 形状**
（S=96/160/224/320/384，其中 96/160/224→block_n=32、320→64、384→128，`n_sub` 随之为 1/1/1/1/2）、
**`n_kv==1` 的 fused 回退**（代码未改但未重跑）、**L2/boundary 非阻塞项**。

下一步（按价值排序）：
1. **补完 exp2 的采纳**（最高价值，已有 −22%/−29% 实测）：一次 dispatch 给足 ≥2 h，跑
   `--level all` → 其余 6 档 `phase=final` → 逐字节拷贝 `_gqa_bwd_kernel_r21_bn128_hpds.py` 覆盖
   `_gqa_bwd_kernel.py` → 追加 final 记录 → 翻转 wrapper 由 conductor 执行。
2. **pds 的 BN=128（下一个结构杠杆）**：采纳后 pds 占总时长 35%（short 373/937… 按 cand：pds 40%/35%），
   是唯一还在 64 行子块的 pass；把它的 mask 常量再省一个 tile 即可越过 36.5 KB 的缺口（预算：
   复用 `neg_buf` 当 mask 限值、`vcmp` 之后再 `vbrc(NEG)` 回填，可省 32 KB；再把 `vexp2` 的 scratch
   改成 in-place 或与 `w_u` 复用可再省 32 KB）。按 mul 的 −43%/−48% 类推，pds 加宽后整体还有 ~15% 空间。
3. 若要再往上，才需要动 DESIGN（例如 pds 与 mul 的相位融合、或 epilogue dtype 策略）——**本轮不需要**，
   既有 `perf_feedback.md` 的 `[DESIGN_LIMIT]` 结论（3-pass 上界 ≈8.2%）保持有效、未修改。

### 21.8 Round 21 Skill Retrospective（增量）

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| tiling/容量裁决 | 加宽 block width 前必须先做**整除性筛选**再做 L0C/UB 预算；96 这类「中间宽度」在 2 的幂 seq_len 上会静默退回 gcd（收窄） | 预算表 21.0 + `_split_geom` 的 gcd 回退 | `attention.md` 增「tile 宽度可评估集合 = seq_len 的因子 ∩ 容量可行集」 | D |
| 编译器容量判据 | L0C 累加器的硬判据是「≤ 32768 个 fp32」，**等号可用**；mul@BN=128 编译通过 | `logs/round21/probe_bn128_mul.log`（L0C 100%、workspace 65536 B） | `traps-compiler.md` 更正 `PL-1.9-hardlimits` 中「必须留余量」的表述（待重验标记） | D |
| 容量窄口定位 | 3-pass 结构里加宽 tile 的窄口通常是**纯 Vector pass 的 UB**（本算子 228.5 KB vs 192 KB），不是 Cube 侧 L0C；先按 pass 单独编译可一步定位 | pds `ub overflow` vs sdw/mul COMPILE_OK | `iteration-diagnosis.md` 增「逐 pass 编译探针定位容量窄口」 | P |
| 实验方法 | 同一假设的「容量分区形态」（部分 pass 加宽、受限 pass 保持子块宽度）能用 1 个候选拿到同一量级收益，且数值与原形态逐元素一致（分区只改 q 行编号，不改表达式） | pds +0.9%/+2.2% 非回退；mul/sdw −43%/−48%、−10%/−18%；L0 6.3057e-03 | `bottleneck-patterns.md` 增「capacity-partitioned widening」 | P |
| 预算纪律 | 单次 dispatch 的 5400 s 不足以同时完成「结构探针 + L0 + A/B + `--level all` + 8 档 final」；应把采纳路径单独排一次 dispatch（每次 msprof run 因 `cache.clear_cache()` 重编译 ≈230 s，8 档 final ≈1400 s） | 本轮 A/B 12 run 用掉 2400 s；L0 238 s | `SKILL.md` Phase 2 增「测量成本估算（每 run 重编译）→ 采纳路径预留」 | R |
