# Mish 算子 Stage 4 性能调优日志（opt_log.md）

- 算子: mish (`y = x * tanh(ln(1+exp(x)))`, 1-D (N,), fp16/fp32/bf16, fp32 中间计算)
- 基线: `../mish.py`（Stage 3 精度通过版本，只读）
- 模式: Developer（保持不变）, target="npuir", `@tilelang.jit(out_idx=[1])`
- 硬件: Ascend 910B2C, 24 AI Core/NPU, UB 192KB/core, 1800MHz, msprof profiling
- 性能目标: best_effort（尽力压低时延/提升吞吐，无硬性数值目标）
- 主指标: `msprof op` Task Duration(us)（用户未指定唯一主指标，默认 kernel 内部优化主口径）
- 辅助指标: NPU event median（调度结构回退门禁；共享环境噪声大，仅作方向性判断）
- 噪声阈值: 3%；max_rounds=10
- 测试 shape: N=1048576（L0 代表 shape，fp16/fp32/bf16 三 dtype）
- 环境: `source /home/tilelang/lxn50063176/Ascend/cann-8.5.0/set_env.sh`, conda env `tilelang_sim`
- 运行目录: `/home/tilelang/l00970450/upload/test/mish_kernel/perf_opt/`

## Dispatch Path 分析

`mish_fwd_kernel(N, dtype, output_dtype=None, block_size=2048)` 中 JIT 工厂按 `dtype`
产生不同特化（Python 层分支，编译期确定）：

| dispatch_path | 触发 | kernel 结构 | target_kernel_name |
|---|---|---|---|
| cast_path | dtype ∈ {float16, bfloat16} | GM→UB → vcast→fp32 → 5 v-op → vcast→out → UB→GM | main |
| fp32_path | dtype == float32 | 同上但 cast 步骤为 UB→UB copy 中转 | main |

代表 workload（用户指定 L0 代表 shape，三 dtype 必测）：

| workload_id | dispatch_path | shape |
|---|---|---|
| fp16_N1M | cast_path | (1048576,) float16 |
| fp32_N1M | fp32_path | (1048576,) float32 |
| bf16_N1M | cast_path | (1048576,) bfloat16 |

## Performance Test Data (Baseline)

msprof 命令模板（每 run 独立输出目录，串行执行）:

```bash
msprof op --kernel-name=main --output=profiles/baseline/{dtype} --launch-count=15 \
  --warm-up=5 --dump=off /home/tilelang/miniconda3/envs/tilelang_sim/bin/python \
  bench.py --impl ../mish.py --dtype {dtype} --N 1048576 --iters 20 --warmup 6
```

| dispatch_path | workload_id | target_kernel_name | captured_op_name | task_duration_us | block_dim | profile_status | raw_profile_dir |
|---|---|---|---|---:|---:|---|---|
| cast_path | fp16_N1M | main | main | 23.28 | 512 | valid | profiles/baseline/fp16/OPPROF_20260824082312_PYBASRWUKAYSOHET |
| fp32_path | fp32_N1M | main | main | 23.30 | 512 | valid | profiles/baseline/fp32/ |
| cast_path | bf16_N1M | main | main | 23.26 | 512 | valid | profiles/baseline/bf16/ |

Baseline NPU event（独立进程，非 msprof 注入；fp16_N1M）:

| config | pass | event_median_us | loop_avg_us |
|---|---|---:|---:|
| baseline flat512 | 1 | 119.05 | 75.34 |
| baseline flat512 (anchor) | ev_pass_1 | 100.76 | 70.71 |
| baseline flat512 (anchor) | ev_pass_2 | 122.37 | 68.61 |

注: msprof 注入下的 event 数据（median 1.4-1.9ms, p75 256ms）被 msprof 工具本身严重扭曲，
标记为 noisy_invalid，不用于任何比较；event 一律独立进程采集。

### Baseline 现象

- P1: Block Dim=512 >> AI Core Count=24（21.3x），单 block 仅 2048 元素（fp16 4KB IO）。
- P2: event median (100-122us) / Task Duration (23.3us) ≈ 4-5x，端到端被 host 侧
  launch/launcher workspace 分配（32768B × blockDim=512 ≈ 16MB/launch）主导
  （匹配 BP_launch_overhead / BP_launcher_workspace_alloc）。
- P3: 三 dtype Task Duration 几乎一致（23.3us），fp16 IO 4MB 仅 fp32 8MB 一半但耗时相同
  → 瓶颈不是 GM 带宽，是波次调度 × 每 block 串行 MTE↔V 等待（ceil(512/24)=22 波 × ~1.06us）。
- P4: 每 block 单 tile 无 T.serial 循环 → auto multi-buffer（默认开启）无循环可优化，
  搬入/计算/写回完全串行（匹配 BP_pipeline_overlap）。

## Iteration 1

### Diagnostic Context

- current_best: ../mish.py (baseline)
- Task Duration: 23.28/23.30/23.26 us (fp16/fp32/bf16), Block Dim 512, Freq 1800/1800

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | P1/P2/P3 Block Dim 过高 + launcher workspace 线性开销 | persistent kernel + T.serial 多块迭代 | BP_launch_overhead 推荐动作；exp2 example 模式；T.serial 循环使 auto multi-buffer 有机会生效 | T.Kernel(num_cores) + serial 迭代 logical tiles；num_cores 扫 24/48/96/192 | Task Duration + event 回退门禁 | candidate |
| OP2 | P1 Block Dim 过高（另一路径） | flat grid 大 tile（block_size=8192） | 降低 block 数同样削减调度/launch 开销，对照 OP1 | baseline kernel 直接传 block_size=8192 (Block Dim=128) | Task Duration | candidate |
| OP3 | fp32_path 冗余 UB 中转 copy | fp32 直连（GM↔计算 buffer，去 x_ub/y_ub 中转） | BP_ub_traffic_floor/BP_redundant；fp32 路径有 2 次纯 UB→UB 拷贝 | 新分支 mish_opt_v1_op3.py，仅改 fp32 数据流 | fp32 Task Duration | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness(L0) | task_duration_us | event_median_us | profile_status | result |
|---|---|---|---|---|---:|---:|---|---|
| mish_opt_v1_op1.py nc=24 | baseline | OP1 | persistent serial, nc=24 | pass | 12.60 (fp16) | na | valid | improved |
| mish_opt_v1_op1.py nc=48 | baseline | OP1 | persistent serial, nc=48 | pass | **8.48 (fp16)** | 89.8/97.0 (2 passes) | valid | **improved (winner)** |
| mish_opt_v1_op1.py nc=96 | baseline | OP1 | persistent serial, nc=96 | pass | 9.22 (fp16) | na | valid | improved |
| mish_opt_v1_op1.py nc=192 | baseline | OP1 | persistent serial, nc=192 | pass | 10.74 (fp16) | na | valid | improved |
| mish_opt_v1_op1.py nc=48 (fp32) | baseline | OP1 | persistent serial, nc=48 | pass | **11.40 (fp32)** | na | valid | improved |
| mish_opt_v1_op1.py nc=48 (bf16) | baseline | OP1 | persistent serial, nc=48 | pass | **8.58 (bf16)** | na | valid | improved |
| (baseline file, bs=8192) | baseline | OP2 | flat 大 tile, Block Dim=128 | pass (L0 via bench --check) | 8.46 (fp16) | na | valid | improved(对照) |
| mish_opt_v1_op3.py | baseline | OP3 | fp32 直连（flat 结构） | pass | 23.10 (fp32) | na | valid | config_no_gain |

num_cores 曲线（fp16_N1M, Task Duration）: 24→12.60, 48→8.48, 96→9.22, 192→10.74,
flat512→23.28。U 形甜点区在 48（2x AI Core Count）附近。

### Event Test Data (Round 1, fp16_N1M)

| workload_id | config_id | event_median_us | passes | anchor_configs | drift_status | event_quality | tie_break |
|---|---|---|---|---|---|---|---|
| fp16_N1M | baseline flat512 | 100.76 / 122.37 | 2 | baseline flat512 | anchor 漂移 21% (>3%, 环境噪声高) | valid(方向性) | msprof |
| fp16_N1M | op1_nc48 | 89.76 / 96.99 | 2 | baseline flat512 | 同上 | valid(方向性) | msprof |

两 pass 中 op1_nc48 相对 baseline 均改善（median -10.9%/-20.7%, loop_avg -17.6%/-4.5%），
event 不回退；绝对值受共享环境噪声影响大，event 仅作回退门禁，决胜用 msprof Task Duration。

### 必测 Dispatch 检查

| branch | dispatch_path | workload_id | current_best_us | branch_us | status |
|---|---|---|---:|---:|---|
| mish_opt_v1_op1.py nc=48 | cast_path | fp16_N1M | 23.28 | 8.48 | pass |
| mish_opt_v1_op1.py nc=48 | fp32_path | fp32_N1M | 23.30 | 11.40 | pass |
| mish_opt_v1_op1.py nc=48 | cast_path | bf16_N1M | 23.26 | 8.58 | pass |

### Iteration Winner

- winner: mish_opt_v1_op1.py（num_cores=48 设为默认参数）
- reason: 三 dtype 必测 dispatch 全部大幅提升（-51% ~ -64%）；调度结构分支 event 两个独立
  pass 均不回退（改善 4.5%-20.7%）；U 形曲线覆盖 24/48/96/192/flat 端点。
- new_current_best: perf_opt/mish_opt_v1_op1.py (persistent, nc=48, bs=2048)
- rollback_branches: mish_opt_v1_op3.py（fp32 直连在 flat 结构上 config_no_gain，-0.9% < 3%）
- OP2 结论: flat bs=8192 在 fp16 上与 OP1 打平（8.46 vs 8.48us），但 Block Dim=128 的
  host launcher workspace（32768×128=4MB）高于 OP1（32768×48=1.5MB），且无 T.serial 循环
  可供 auto multi-buffer 优化；记录为同族对照，不作为 winner。

## Iteration 2

### Diagnostic Context

- current_best: mish_opt_v1_op1.py (persistent nc=48, bs=2048)
- Task Duration: 8.48 (fp16) / 11.40 (fp32) / 8.58 (bf16) us, Block Dim=48
- 新现象: fp32 比 fp16 慢 34%（IO 翻倍）; nc=48 → iters=10.67 (imbalance 9%);
  auto multi-buffer 已在 T.serial 上生效（bs=8192 编译失败证据:
  requires 1703936 bits > 1572864 bits available，multi-buffer 翻倍 buffer）

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | iters 不整除 imbalance 9% | num_cores 整除性候选细化（32/64/72 + 44/52） | BP_task_concurrency_sweet_spot | bench 传参扫描 | Task Duration | candidate |
| OP2 | 每 tile 搬运/迭代开销 | block_size 4096（更大 tile） | 更大搬运粒度摊薄开销; UB 预算允许 | bench 传参扫描 | Task Duration | candidate |
| OP3 | fp32 路径 x_ub/y_ub 中转 + UB 预算 | fp32 直连（3 buffer, 12B/elem）解锁大 bs | BP_ub_pressure; v1_op3 已证直连在 flat 结构无收益但 persistent 下 buffer 削减解锁 bs | 新分支 mish_opt_v2_op3.py 双 prim_func | fp32 Task Duration | candidate |
| OP4 | cast 路径 5 buffer UB 压力 | io_ub 合并输入/输出 staging（4 buffer, 14B/elem） | BP_ub_pressure last-use 复用 | 新分支 mish_opt_v2_op4.py | fp16 Task Duration | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | task_duration_us | profile_status | result |
|---|---|---|---|---|---:|---|---|
| op1 nc=32/64/72 bs2048 (fp16) | v1_op1 | OP1 | num_cores 细化 | pass | 10.02/11.02/10.72 | valid | config_no_gain（nc=48 仍最优） |
| op1 nc=44/52 bs4096 (fp16) | v1_op1 | OP1+OP2 | 交叉点 | pass | 7.90/10.64 | valid | nc=48 仍最优 |
| op1 bs=4096 nc=48 (fp16) | v1_op1 | OP2 | block_size 4096 | pass | **7.88** | valid | **improved（-7.1%）** |
| op1 bs=4096 nc=48 (bf16) | v1_op1 | OP2 | block_size 4096 | pass | **7.98** | valid | **improved（-7.0%）** |
| op1 bs=4096 nc=32/64 (fp16) | v1_op1 | OP2×OP1 | 交叉点 | pass | 9.02/10.16 | valid | config_no_gain |
| op1 bs=8192 nc=48 (fp16) | v1_op1 | OP2 | block_size 8192 | 编译失败(UB overflow, multi-buffer 翻倍后 208KB>192KB) | na | blocked | blocked |
| mish_opt_v2_op3.py bs2048 (fp32) | v1_op1 | OP3 | fp32 直连 3-buffer | pass | 11.26 | valid | 直连本身 no gain（vs 11.40, -1.2% < 3%） |
| mish_opt_v2_op3.py bs4096 (fp32) | v1_op1 | OP3+OP2 | 直连+bs4096 | pass | 9.64 | valid | 与 base bs4096(9.58) 打平 → 收益全来自 bs |
| mish_opt_v2_op3.py bs8192 (fp32) | v1_op1 | OP3+OP2 | 直连+bs8192（解锁） | pass | **8.82** | valid | **improved（-22.6% vs 11.40）** |
| op1 bs=4096 (fp32, 5-buffer 对照) | v1_op1 | OP2 | block_size 4096 | pass | 9.58 | valid | 对照（分辨 OP3 直连贡献≈0） |
| mish_opt_v2_op4.py bs2048 (fp16) | v1_op1 | OP4 | io_ub 合并 | pass | 12.84 | valid | **回退 +51%** |
| mish_opt_v2_op4.py bs4096 (fp16) | v1_op1 | OP4 | io_ub 合并 | pass | 10.40 | valid | 回退 +32% vs 7.88 |
| mish_opt_v2_op4.py bs8192 (fp16) | v1_op1 | OP4 | io_ub 合并解锁 bs8192 | pass | 8.70 | valid | 仍劣于 base bs4096(7.88) |

OP4 回退机制分析: io_ub 生命周期横跨整个 tile（开头 vcast-in、结尾 vcast-out），
破坏 auto multi-buffer 对短生命周期 staging buffer 的搬运/计算重叠 → 串行化。
教训: buffer 复用（BP_ub_pressure last-use 复用）与 auto multi-buffer 重叠（BP_pipeline_overlap）
存在冲突，在 multi-buffer 生效的结构上应优先保证生命周期短、各自独立。

### 必测 Dispatch 检查（候选 winner 组合）

三 dtype 最优结构不同，组成 dtype-aware 组合候选（后续在统一版本中实现）：

| dispatch_path | workload_id | round1_best_us | round2_best_us | round2 结构 | status |
|---|---|---:|---:|---|---|
| cast_path | fp16_N1M | 8.48 | **7.88** | v1_op1 + bs4096 | pass |
| cast_path | bf16_N1M | 8.58 | **7.98** | v1_op1 + bs4096 | pass |
| fp32_path | fp32_N1M | 11.40 | **8.82** | v2_op3 直连 + bs8192 | pass |

### Iteration Winner

- winner: dtype-aware 组合（cast: v1_op1 结构 bs=4096; fp32: v2_op3 直连 bs=8192; 均 nc=48）
- reason: 三 dtype 均超过 3% 噪声阈值提升（-7.0% ~ -22.6%）；OP2/OP3 单点贡献已拆分验证
  （bs 增大是收益主体，fp32 直连是解锁大 bs 的 enabler，两者组合后单点依然可解释）。
- new_current_best: 组合版（Round 3 落地为统一文件）
- rollback_branches: mish_opt_v2_op4.py（io_ub 合并方向在 multi-buffer 结构上回退，
  记录为该结构下 family 级结论: cast 路径 staging 合并与 auto multi-buffer 冲突）
- config_no_gain: nc=32/44/52/64/72（nc=48 稳定最优，甜点区 2x AI Core Count 已覆盖）

## Iteration 3

### Diagnostic Context

- current_best: Round 2 dtype-aware 组合（cast: bs4096; fp32: 直连 bs8192; nc=48）
- Task Duration: 7.88 (fp16 bs4096) / 8.82 (fp32 bs8192) / 7.98 (bf16 bs4096) us

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | cast 路径 bs 甜点未定 | bs=5120/6144 细扫 | bs4096→8192 之间未覆盖; UB 预算 96KB(6144) 可行 | bench 传参 | Task Duration | candidate |
| OP2 | fp32 直连 bs 上限未定 | bs=10240 | buffer 削减后 UB 预算 120KB 静态 | bench 传参 | Task Duration | candidate |
| OP3 | fp32 bs8192 下 nc 甜点未复核 | nc=24/96 复核 | 新 bs 下甜点可能漂移 | bench 传参 | Task Duration | candidate |
| OP4 | 计算链占比未知 | copy-only 下限探测 | 判断计算链优化剩余空间 | probe_copy.py 同结构纯搬运 kernel | Task Duration 下限 | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | task_duration_us | profile_status | result |
|---|---|---|---|---|---:|---|---|
| op1 bs=5120 nc48 (fp16) | R2 组合 | OP1 | bs 细扫 | pass | 8.14 | valid | config_no_gain |
| op1 bs=6144 nc48 (fp16) | R2 组合 | OP1 | bs 细扫 | pass | 7.80 / 7.76 (两次) | valid | 平区（vs bs4096 的 7.88/8.18，中值更优且更稳） |
| op1 bs=6144 nc48 (bf16) | R2 组合 | OP1 | bs 细扫 | pass | **7.58** | valid | improved（vs 7.98, -5.0%） |
| v2_op3 bs=10240 nc48 (fp32) | R2 组合 | OP2 | bs 上限 | 编译失败(UB overflow 240KB>192KB) | na | blocked | blocked（fp32 bs 上限=8192） |
| v2_op3 bs8192 nc24 (fp32) | R2 组合 | OP3 | nc 复核 | pass | 13.06 | valid | config_no_gain |
| v2_op3 bs8192 nc96 (fp32) | R2 组合 | OP3 | nc 复核 | pass | 8.90 | valid | config_no_gain（nc=48 仍最优） |
| probe_copy fp16 bs4096 | - | OP4 | copy-only 下限 | pass | 4.88 | valid | 搬运下限（诊断用） |
| probe_copy fp32 bs8192 | - | OP4 | copy-only 下限 | pass | 5.78 | valid | 搬运下限（诊断用） |

计算链占比分析: fp16 mish 7.88 = copy 4.88 + 计算链 ~3.0us; fp32 8.82 = copy 5.78 + ~3.04us。
两 dtype 计算链额外时间几乎相同 → vcast 被流水掩盖；5 个 v-op（vexp/vadd/vln/vtanh/vmul）
约 3us，超越函数内部多指令展开，有效利用率已较高。数学变形方案（如 (w²-1)/(w²+1) 替代
vln+vtanh）在 x>44 时 w² 溢出 fp32 产生 NaN（fp16 输入域可达 65504），需 clamp 而无向量
条件 op，且 op 数不降反增 → 放弃该方向（风险高、预期负收益）。

### Iteration Winner

- winner: cast 路径 bs=6144（中值更优且 run-to-run 更稳：7.80/7.76 vs 7.88/8.18）
- new_current_best: cast: bs=6144; fp32: 直连 bs=8192; 均 nc=48
- rollback_branches: 无（本轮无回退，仅参数收敛）
- 记录: fp32 直连 bs 上限 8192（10240 blocked）；nc=48 甜点在新 bs 下保持。

## Iteration 4

### Diagnostic Context

- current_best: cast bs=6144 / fp32 直连 bs=8192, nc=48
- 未验证结构: T.Pipelined 显式流水 vs auto multi-buffer（Developer 模式默认开启后者）

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | 搬运/计算重叠是否可再压 | T.Pipelined 替代 T.serial | BP_pipeline_overlap; testing/compiler_hint_ops/test_pipelined.py 模式 | mish_opt_v4_op1.py（cast 路径 bs4096） | Task Duration | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | task_duration_us | profile_status | result |
|---|---|---|---|---|---:|---|---|
| mish_opt_v4_op1.py bs4096 nc48 (fp16) | R2 组合 | OP1 | T.Pipelined | pass | 8.00 | valid | config_no_gain（vs T.serial 7.88/8.18 平区，无增益） |

机制解释: auto multi-buffer（默认开启）已在 T.serial 循环上完成搬运/计算重叠；
T.Pipelined 是同一机制的显式表达，不叠加额外收益。记录为当前
TileLang-NPUIR/CANN/Developer 组合下的 family 级结论（相对 auto multi-buffer 无叠加收益）。

### Iteration Winner

- winner: current_best 保持（T.serial + auto multi-buffer）
- 至此主要结构候选（persistent 多块迭代 / 大 tile flat / fp32 直连 / staging 合并 /
  T.Pipelined / 数学变形）均已验证，参数曲线（nc、bs）均已覆盖并收敛于平区。

## Final Performance Test Data

final = perf_opt/mish.py（dtype-aware 默认: cast bs=6144 / fp32 直连 bs=8192 / nc=48）。
msprof 命令同 baseline 模板，`--impl ./mish.py`（不传 block-size/num-cores，走调优默认）。

| dispatch_path | workload_id | target_kernel_name | captured_op_name | task_duration_us | block_dim | profile_status | raw_profile_dir |
|---|---|---|---|---:|---:|---|---|
| cast_path | fp16_N1M | main | main | **7.44** | 48 | valid | profiles/final/fp16/ |
| fp32_path | fp32_N1M | main | main | **8.56** | 48 | valid | profiles/final/fp32/ |
| cast_path | bf16_N1M | main | main | **7.66** | 48 | valid | profiles/final/bf16/ |

### Baseline vs Final (msprof Task Duration, N=1048576)

| dtype | baseline_us | final_us | improvement | IO bytes | final_eff_bw |
|---|---:|---:|---:|---|---|
| float16 | 23.28 | **7.44** | **-68.0% (3.13x)** | 4 MB | 564 GB/s |
| float32 | 23.30 | **8.56** | **-63.3% (2.72x)** | 8 MB | 980 GB/s |
| bfloat16 | 23.26 | **7.66** | **-67.1% (3.04x)** | 4 MB | 548 GB/s |

注: 有效带宽按 bench 4-clone 轮换协议计算（含 L2 命中成分）；copy-only 下限实测
fp16 4.88us / fp32 5.78us，final 距纯搬运下限约 1.7-2.7us（计算链+cast）。

## Final Event Test Data

final vs baseline（anchor），fp16_N1M，独立进程 4 passes（pass3/4 交替顺序排除顺序效应）：

| workload_id | config_id | event_median_us (per pass) | loop_avg_us (per pass) | passes | anchor | drift_status | event_quality |
|---|---|---|---|---|---|---|---|
| fp16_N1M | baseline | 120.99/117.67/124.62/160.34 (中位 122.8) | 78.95/70.91/91.48/104.67 | 4 | baseline | pass 间漂移最高 28%（共享 16-NPU 环境噪声） | valid(方向性) |
| fp16_N1M | final | 96.85/139.78/116.23/100.86 (中位 108.5) | 66.71/84.86/61.72/62.06 | 4 | baseline | 同上 | valid(方向性) |

判定: final event per-launch median 中位值优于 baseline（108.5 vs 122.8，-12%）；
loop_avg 4/4 pass 一致更优（中位 64.4 vs 85.2，-24%）；per-launch median 3/4 pass 更优
（pass 2 中 final 遭遇共享环境瞬时干扰，前后 pass 的 anchor baseline 稳定，故判为环境噪声
而非版本回退）。结论: **final event 不回退，且中位/吞吐一致改善**。event 仅作回退门禁与
调度结构参考，性能主口径为 msprof Task Duration（Block Dim 512→48 同时大幅削减 host 侧
launcher workspace 分配 32768×512≈16MB → 32768×48≈1.5MB per launch）。

## Final Summary

- best 版本: `perf_opt/mish.py`
- 结构: persistent kernel（T.Kernel(48) + T.serial 多 tile 迭代）+ dtype-aware tile
  （cast 路径 bs=6144 5-buffer；fp32 路径 bs=8192 3-buffer 直连，无 staging 中转）
- 最终 latency（msprof Task Duration, N=1048576）:
  - fp16: 23.28 → 7.44 us（-68.0%）
  - fp32: 23.30 → 8.56 us（-63.3%）
  - bf16: 23.26 → 7.66 us（-67.1%）
- 总提升: 2.72x ~ 3.13x（三 dtype 全部 > 60% 提升）
- 精度: L0/L1/L2/Boundary 全部 PASS（`python perf_opt/mish.py --level all`，无 WARN）
- 中止原因: **plateau + 候选穷尽** —— 主要结构候选（persistent 多块迭代 / flat 大 tile /
  fp32 直连 / staging 合并 / T.Pipelined / 数学变形）全部验证，参数曲线（num_cores 24~192
  与 flat 端点、block_size 2048~10240）已覆盖并收敛；final 距 copy-only 搬运下限
  1.7~2.7us（计算链必需开销），继续优化 ROI 低于噪声阈值。best_effort 目标下按
  plateau 收束（连续轮次候选均落入平区/无增益/阻塞）。
- 有效优化点（采纳）:
  1. persistent kernel + T.serial 多块迭代（Block Dim 512→48，-51%~-64%）
  2. auto multi-buffer 在 T.serial 循环上生效（结构性 enabler）
  3. dtype-aware block_size（cast 6144 / fp32 8192，进一步 -5%~-23%）
  4. fp32 直连路径（buffer 5→3，解锁 bs=8192 的 UB 预算）
- 无效/回退优化点（放弃）:
  1. io_ub 合并 staging（BP_ub_pressure last-use 复用）: 与 auto multi-buffer 冲突，+32%~51% 回退
  2. T.Pipelined 显式流水: 相对 auto multi-buffer 无叠加收益
  3. flat 结构 fp32 直连: 瓶颈在调度时无收益（在 persistent 结构上作为 bs 解锁器有效）
  4. 数学变形 (w²-1)/(w²+1) 替代 vln+vtanh: x>44 fp32 溢出 NaN，风险不可控（未实施）
  5. num_cores 24/32/44/52/64/72/96/192、bs 2048/4096/5120/8192/10240: 均非最优（已记录）

## Skill Retrospective

- skill 流程有效性: Phase 1 的"Block Dim 远超 AI Core Count 必须 event"规则直接命中本算子
  （512 vs 24），baseline 的 event/TaskDuration 4-5x 背离引导出 persistent kernel 主方向，
  单轮即获得 >60% 收益。每轮重分析现象的要求避免了在 flat 结构上浪费轮次（v1_op3 直连
  在 flat 结构 config_no_gain，但重分析后在 persistent 结构上作为 UB 解锁器复活）。
- 流程问题 1: 共享 NPU 环境（16 卡多用户）event 噪声极大（anchor 跨 pass 漂移最高 28%，
  pass 内 p75 可达 256ms 级干扰），"2 独立 pass + anchor"协议不足以稳定排序，本次实际
  用了 4 pass + 交替顺序 + 中位数裁决才得出方向性结论。建议 skill 在
  BP_measurement_resolution_limited 中补充: 共享环境下 event 只作回退门禁（方向性），
  排序一律 msprof 决胜，pass 数建议 ≥4 且交替顺序。
- 流程问题 2: msprof 注入会严重扭曲进程内 event 计时（本次观测 median 1.4-1.9ms、
  p75 256ms），msprof run 日志内的 event 数据必须标记 noisy_invalid，不可混用。建议在
  Profile-collection.md 明确禁止从 msprof run 的 stdout 提取 event 数据。
- 流程问题 3: TileLang TIR parser 不支持 `if` 分支内 `T.alloc_ub` 赋值（"Undefined
  variable"），条件 buffer 布局必须用 Python 层双 prim_func 分支实现。该约束在
  DESIGN.md 3.3 有暗示但 skill/文档未明说，建议补充到 Bottleneck-patterns.md 的
  BP_ub_pressure 实现注意事项。
- BP proposal:
  - BP_shared_env_event_noise（新增建议）: 共享 NPU 环境 event 协议——anchor 交替顺序
    ≥4 pass、中位数裁决、event 仅作回退门禁，msprof 决胜。
  - BP_ub_pressure 补充: buffer last-use 复用（staging 合并）与 auto multi-buffer 重叠
    冲突的机制模式：长生命周期复用 buffer 会破坏短生命周期独立 staging 的自动双缓冲，
    在 multi-buffer 生效的结构上应先验证复用不回退再采纳。
  - BP_multi_buffer_ub_budget（新增建议）: auto multi-buffer 会按 buffer 生命周期翻倍
    部分 UB buffer，大 tile 探索时应以"multi-buffer 后预算"（实测约 1.6x 静态占用）评估
    上限，而不是静态占用；削 buffer（如 fp32 直连 3-buffer）是解锁大 tile 的正规手段。

---

# Round 2: yolo-p3/p4 re-tune

- 触发: conductor 第二轮调度（optimize, mode=first_tune, round=2）
- 基准 shape 变更: Round 1 以 smoke-1m (N=1,048,576) 为代表；本轮以 manifest 的
  yolo-p3 [16,256,80,80] (N=26,214,400) / yolo-p4 [16,512,40,40] (N=13,107,200)
  为基准，fp16 + bf16 四组合（kernel 为 1-D 展平语义，以 N 驱动）。
- incumbent（被超越对象）: Round 1 final `perf_opt/mish.py`（备份于
  `perf_opt/history_version/mish_r1_final.py`）
- 守门 shape: smoke-1m（fp16/fp32/bf16）、fc-wide（fp16/bf16），回退阈值 3%
- 其余口径与 Round 1 相同（msprof op Task Duration 主口径，event 仅回退门禁，
  共享 16-NPU 环境噪声大）

## Round 2 Performance Test Data (Baseline = Round 1 final)

baseline 数据来自 `profiles/manifest/`（round-1 收尾时采集，本轮校验后沿用；
median-of-15 提取，`extract_msprof.py`）：

| dispatch_path | workload_id | target_kernel | task_duration_us | block_dim | profile_status | raw_profile_dir |
|---|---|---|---:|---:|---|---|
| cast_path | yolo_p3_fp16 | main | 103.42 | 48 | valid | profiles/manifest/yolo_p3_fp16/ |
| cast_path | yolo_p3_bf16 | main | 105.90 | 48 | valid | profiles/manifest/yolo_p3_bf16/ |
| cast_path | yolo_p4_fp16 | main | 53.84 | 48 | valid | profiles/manifest/yolo_p4_fp16/ |
| cast_path | yolo_p4_bf16 | main | 55.18 | 48 | valid | profiles/manifest/yolo_p4_bf16/ |
| cast_path | fc_wide_fp16 (守门) | main | 35.92 | 48 | valid | profiles/manifest/fc_wide_fp16/ |
| cast_path | fc_wide_bf16 (守门) | main | 36.52 | 48 | valid | profiles/manifest/fc_wide_bf16/ |
| cast_path | smoke_fp16 (守门) | main | 7.44 | 48 | valid | profiles/final/fp16/ |
| fp32_path | smoke_fp32 (守门) | main | 8.56 | 48 | valid | profiles/final/fp32/ |
| cast_path | smoke_bf16 (守门) | main | 7.66 | 48 | valid | profiles/final/bf16/ |

对照组（原始基准 ../mish.py）: 547.6 / 542.0 / 271.5 / 272.2 us（block_dim 12800/6400）。

## Iteration 1 (Round 2): 大 N 现象重建

### Diagnostic Context

- current_best: Round 1 final（persistent nc=48, T.serial, cast bs=6144, fp32 直连 bs=8192）
- 关键测量（新增诊断工具 `probe_copy.py`，与 mish 同结构纯搬运 kernel）:
  copy-only floor @ yolo-p3 fp16, bs=6144: nc=24 → 56.74us, nc=48 → **40.38us**,
  nc=96 → 39.12us（有效带宽 ~2.6TB/s @ nc=48，MTE 侧在 48 block 饱和）

### 现象

- P1: T_mish(103.4) ≈ T_copy(40.4) + ~63us，且逐元素增量与 N=1M 完全一致
  （2.4ns/kelem）→ **MTE2/V/MTE3 完全串行**，63us 为暴露的向量链时间。
- P2: 向量链非 5 个 v-op：`vtanh` 在 Developer codegen 展开为
  (e^{2z}-1)/(e^{2z}+1)（vmul+vexp+vsub+vadd+vdiv 5 pass），cast 路径实际
  ~11 个向量 pass。
- P3: nc 重扫（大 N）: 24→202.5, 48→103.4, 96→104.0, 192→105.5 → 48 仍饱和
  （向量吞吐在 48 block 饱和，左端并发不足，右端无增益）。
- P4: bs 扫描: 4096→109.6, 6144→103.4, 7168→101.8（-1.6%，< 3% 不单独采纳；
  8192 被 auto-multi-buffer 膨胀后 UB 溢出，维持 round-1 结论）。

### Candidate Optimization Points（本轮全部候选）

| opt_id | 目标现象 | 优化点 | 结果概要 |
|---|---|---|---|
| OP1 | P1（无重叠） | T.Pipelined(num_stages=2)（round-1 v4_op1 漏传 num_stages，实为禁用流水线，本轮修正重测） | 103.36us，无重叠（config_no_gain） |
| OP2 | P1 + guard/动态尾块阻断流水 | guard-free 主循环 + 尾块 epilogue（v5_op2） | 101.78us，-1.6%（不足 3%） |
| OP3 | P1 | 手动 MTE2/V/MTE3 软件流水（Expert 风格 rs + set_flag/wait_flag + 3 槽 in-place io，参照 examples/elementwise/vec_add_2d_multi_buffer.py） | 正确性受阻→修复后 97.5/91.4us，但需 Expert 模式（见下） |
| OP4 | P2（11 pass 向量链） | 显式 tanh 恒等式 tanh(softplus(x)) = (w²-1)/(w²+1), w=1+e^x（去 vln+vtanh，11→9 pass；与 Developer vtanh 展开式数学等价、溢出边界同为 x>~44.15） | **采纳**：v5_op8 98.8us（-4.5%） |
| OP5 | 参数 | nc/bs 大 N 重扫 | nc=48 维持；bs=7168 采纳（与 OP4 组合） |

### 调试过程关键发现（OP3 正确性攻关，bisect 于 logs/r2/bisect_pipeline.py）

1. 手动流水初版（Developer）结果错误 → bisect：copy/cast/vadd/vbrc 单独均正确，
   vexp/vln/vtanh 链错误。
2. **Expert 模式 vtanh 是 Taylor 展开 x−x³/3+2x⁵/15−17x⁷/315（codegen_npuir_api.cc），
   |x|>~0.8 即严重失真**（tanh 输出可达 ±数百）；Developer 模式 vtanh 为精确公式
   （codegen_npuir_dev.cc）。两 codegen 对同一 API 语义不一致。
3. **hivm auto-sync 注入对超越函数链是正确性必需**：关闭后（无论 auto-MB 开关、
   无论 Developer/Expert），含 vexp/vln 链的 plain T.serial kernel 也出错
   （超越 v-op 异步完成 + 编译器 extra temp buffer 复用竞争）。vec_add 例子
   （仅 vadd）不受影响。
4. Developer 模式 tensor-compile 会重构 rs/flag 手动流水代码并产生确定性错误结果
   （max_diff 4.0）→ 手动流水只能在 Expert 模式正确运行。
5. 显式 tanh 公式（无 vln/vtanh）+ 手动 flag 流水 + pipe_barrier：Expert 模式下
   全 shape 正确（含槽复用/尾块/N=1M）；去掉 barrier 同样正确（barrier 曾疑似
   全流水线排空，去除后 97.5→91.4us）。
6. 手动流水最终仍为串行执行（91.4 ≈ 40.4 + 51.5）：T.copy/v-op 的发射为同步阻塞，
   flag 协议只保证顺序正确，未获得 MTE↔V 并发。pipe_barrier 字符串参数不能经
   Python 变量传入（LetStmt+StringImm 在 codegen 崩溃，须内联字面量）。

### Experiment Branches（Round 2）

| branch | base | 主要改动 | correctness | p3_fp16_us | 结果 |
|---|---|---|---|---:|---|
| (incumbent, bench 传参) | R1 final | nc=24/96/192 | pass | 202.5/104.0/105.5 | config_no_gain（nc=48 维持） |
| (incumbent, bench 传参) | R1 final | bs=4096/7168 | pass | 109.6/101.8 | bs=7168 平区（-1.6%），待组合 |
| mish_opt_v5_op1.py | R1 final | T.Pipelined(num_stages=2)（含 guard） | pass | 103.36 | config_no_gain |
| mish_opt_v5_op2.py | R1 final | guard-free Pipelined + 尾块 epilogue | pass | 101.78 | config_no_gain（-1.6%） |
| mish_opt_v5_op3.py | v5_op2 | + disable_hivm_auto_inject_sync | **fail**（竞态） | na | invalid（auto-sync 为正确性必需） |
| mish_opt_v5_op4.py | v5_op2 | 手动 rs/flag 流水（vtanh 链） | **fail**（两种模式均错） | na | blocked→改道 |
| mish_opt_v5_op5.py | v5_op4 | Expert + 显式 tanh + barrier | pass | 97.50 | improved 但受模式约束（见 Iteration 2） |
| mish_opt_v5_op6.py | v5_op2 | Developer + 显式 tanh（6 buffer, bs=5120） | pass | 105.08 | config_no_gain（bs/膨胀代价吃掉收益） |
| mish_opt_v5_op7.py | v5_op5 | 去除 intra-V barrier | pass | 91.42 | **improved（-11.6%）但需 Expert 模式，不采纳** |
| mish_opt_v5_op8.py | v5_op2 | Developer + 显式 tanh（2 buffer 原地链, bs=7168） | pass | 98.80 | **improved（-4.5%）采纳 → final** |

### Iteration 1 Winner

- winner: mish_opt_v5_op8.py（Developer + 显式 tanh + 原地 2-buffer 链 + bs=7168）
- reason: 四目标组合全部 ≥3% 提升（见 Final 表）；守门 shape 无回退；Developer
  模式约束保持。
- rollback/blocked: v5_op3（竞态）、v5_op4（两模式均错）、v5_op7（正确且最快但
  违反 Developer 模式硬约束，保留为记录）、v5_op5/v5_op6（被 v5_op7/v5_op8 支配）。

## Round 2 Event Test Data（调度结构分支回退门禁）

yolo_p3_fp16, N=26,214,400, event_pass.py（独立进程，25 iters，2 pass × 交替顺序，
anchor=incumbent）：

| config | event_median_us (各 pass) | loop_avg_us (各 pass) |
|---|---|---|
| incumbent (anchor) | 189.56 / 208.64 / 213.00（中位 208.6） | 163.72 / 176.73 / 171.30 |
| v5_op8 (final) | 198.90 / 183.94 / 205.90（中位 198.9） | 162.87 / 156.77 / 165.62 |

- anchor 同 pass 内漂移最高 ~14%（共享环境噪声，与 Round 1 结论一致）→ event 仅作
  方向性回退门禁。
- 判定: final event median -4.7%、loop_avg 中位 -4.9%，**方向与 msprof 一致、不回退**
  （event_quality = valid(方向性)；排序以 msprof Task Duration 决胜）。

## Final Performance Test Data (Round 2)

final = `perf_opt/mish.py`（= mish_opt_v5_op8；四组合 + 守门在最终文件上复测）：

| dispatch_path | workload_id | task_duration_us (final) | incumbent | 提升 | raw_profile_dir |
|---|---|---:|---:|---:|---|
| cast_path | yolo_p3_fp16 | **98.80** | 103.42 | **-4.5%** | profiles/manifest_r2/r2_final_p3_fp16/ |
| cast_path | yolo_p3_bf16 | **101.80** | 105.90 | **-3.9%** | profiles/manifest_r2/r2_final_p3_bf16/ |
| cast_path | yolo_p4_fp16 | **51.32** | 53.84 | **-4.7%** | profiles/manifest_r2/r2_final_p4_fp16/ |
| cast_path | yolo_p4_bf16 | **52.50** | 55.18 | **-4.9%** | profiles/manifest_r2/r2_final_p4_bf16/ |
| cast_path | smoke_fp16 (守门) | 7.40 | 7.44 | -0.5%（噪声内） | profiles/manifest_r2/r2_final_smoke_fp16/ |
| cast_path | fc_wide_fp16 (守门) | 34.38 | 35.92 | -4.3%（改善） | profiles/manifest_r2/r2_final_fc_fp16/ |
| cast_path | fc_wide_bf16 (守门) | 35.30 | 36.52 | -3.3%（改善） | profiles/manifest_r2/v5_op8_final_fc_bf16/ |
| cast_path | smoke_bf16 (守门) | 7.60 | 7.66 | -0.8%（噪声内） | profiles/manifest_r2/v5_op8_final_smoke_bf16/ |
| fp32_path | smoke_fp32 (守门) | 8.56（路径字节级未改动） | 8.56 | 0 | profiles/final/fp32/（round-1 记录） |

（守门中 bf16/fc_bf16 于代码完全相同的 mish_opt_v5_op8.py 上采集；最终文件
复测 fp16 组合一致。）

## Final Summary (Round 2)

- best 版本: `perf_opt/mish.py`（Developer 模式保持；结构 = Round 1 persistent
  nc=48 + guard-free Pipelined 热循环 + 尾块 epilogue + **显式 tanh 恒等式
  (w²-1)/(w²+1)**（原地 2-buffer 链，cast 路径 11→9 向量 pass）+ cast bs=7168；
  fp32 路径与 Round 1 字节级相同）
- 最终 latency（msprof Task Duration）:
  - yolo-p3 fp16: 103.42 → 98.80 us（-4.5%）
  - yolo-p3 bf16: 105.90 → 101.80 us（-3.9%）
  - yolo-p4 fp16: 53.84 → 51.32 us（-4.7%）
  - yolo-p4 bf16: 55.18 → 52.50 us（-4.9%）
- 守门: smoke-1m 三 dtype 噪声内（fp32 路径未动）；fc-wide 两 dtype 改善 3.3~4.3%
- 精度: L0/L1/L2/Boundary 全部 PASS（`python mish.py --level all`）；新增
  Boundary-overflow-domain（x=100）WARN 与 incumbent 行为逐位一致（两者均 NaN，
  Developer vtanh 展开式与本公式同一边界），非回归。
- 中止原因: **plateau + 约束边界** —— 结构候选（显式 tanh、Pipelined、手动软件
  流水、nc/bs 重扫）已充分验证；剩余最大空间（MTE↔V 真重叠，可达 ~55-65us）被
  两个平台层限制封死：(a) Developer 模式 tensor-compile 破坏手动 rs/flag 代码，
  (b) 同步发射语义使 flag 协议无法产生并发（Expert 手动流水 91.4us 仅来自去除
  auto-sync 开销与公式，并非重叠）。Developer 约束内无 ≥3% 的进一步候选。
- 未采纳但已验证的更强方案（供后续决策参考）: mish_opt_v5_op7.py（Expert 模式 +
  手动 MTE2/V/MTE3 流水 + 显式 tanh，91.42us，**-11.6%**，四组合与守门未全测，
  正确性已过 L0/多 shape）。若编排层允许放开 TILELANG_ASCEND_MODE=Developer 约束，
  可在此基础上完成验证后替换。

## Skill Retrospective (Round 2)

- 流程有效性: copy-only floor 探针 + "逐元素增量跨 shape 一致性"检查快速定位了
  大 N 下的真实瓶颈（串行 MTE+V），避免了在带宽方向空耗；bisect 脚本
  （logs/r2/bisect_pipeline.py）把"手动流水结果错误"拆解到 op 级，是本轮能够
  收敛的关键手段。
- 流程问题 1: round-1 "T.Pipelined 无叠加收益"的结论是**无效实验**（num_stages
  缺省=0 即禁用流水线）。skill 应在 Bottleneck-patterns.md 的 BP_pipeline_overlap
  中显式标注 num_stages 缺省值为 0 的坑。
- 流程问题 2: 平台层语义差异（Developer vs Expert 的 vtanh lowering 不一致、
  auto-sync 对超越函数链的正确性必要性、Developer tensor-compile 破坏手动
  rs/flag 代码）完全没有文档化，只能靠 bisect 逆向发现。这些属于
  tilelang-error-fixer / npuir-overview 应沉淀的知识。
- BP proposal（新增/更新）:
  - **BP_vtanh_mode_inconsistency（新增）**: Expert codegen 的 vtanh Taylor 展开在
    |x|>~0.8 数值失真（tanh 输出可达数百），Developer codegen 为精确公式。建议
    统一为精确公式或在文档标注 Expert 模式 vtanh 输入域限制。
  - **BP_transcendental_sync_required（新增）**: vexp/vln/vtanh 等异步超越 v-op
    依赖 hivm auto-sync 注入保证正确性（extra temp buffer 复用竞争）；关闭
    auto-sync 时这些 op 的链式使用会静默出错。vec_add 类纯简单 op 不受影响，
    具有误导性。建议在 disable_hivm_auto_inject_sync 的 pass_config 文档中标注。
  - **BP_string_imm_codegen（新增）**: T.pipe_barrier 等接受字符串参数的 op，
    字符串必须内联字面量；经 Python 变量（LetStmt+StringImm）传入在 codegen
    崩溃（"StringImmNode case not supported"）。
  - **BP_manual_pipeline_developer_broken（新增）**: Developer 模式 tensor-compile
    会重构 T.rs/set_flag/wait_flag 手动流水代码并产生确定性错误结果；手动流水
    目前只能在 Expert 模式使用（examples/elementwise/vec_add_2d_multi_buffer.py
    即运行于 Expert 路径）。
  - BP_launch_overhead 补充: 手动 flag 流水（Expert）实测仍为同步发射（T ≈
    T_copy + T_V），flag 协议只保证顺序不产生并发；"软件流水=重叠"的预期在当前
    codegen 下不成立，应作为已知限制记录。

# Round 2 post-scriptum: v2/v3 refinement & interrupted-session recovery

- 触发: conductor 调度（optimize, mode=resume_finish, round=2, attempt_index=2）
- 背景: 上一 optimizer session 于 08-24 12:35 被取消。本节依据磁盘工件
  （profiles/manifest_r2/r2_final_v2_*、r2_final_v3_*、r2_ab_*、logs/r2/）还原
  被中断的时间线，并完成剩余收尾：75.34 vs 78.5 疑点调查、A/B 表补全
  （incumbent p4_bf16）、最终数值定型、头注更新、精度回归复测。
- 本节之前的所有 Round 2 章节保留原样（其中 Final 表的 98.80/101.80/51.32/
  52.50 为 v5_op8 时代记录，已被本节 P.2/P.7 的 v3 终态取代，以本节为准）。

## P.1 v5_op8 归因勘误（Part A 遗漏事实）

- 复盘发现: Iteration 1 winner `mish_opt_v5_op8.py` 的显式 tanh 公式只写进了
  Part B（尾块 epilogue，每核仅 1 tile），Part A 热循环（每核 ~89 tile @
  yolo-p3 / bs=7168 / nc=48）仍是 vln+vtanh 链。
- 因此"11→9 vector pass"的归因对 v5_op8 文件不成立：其热循环仍为 ~11 pass，
  98.80us 的 -4.5% 主要来自 2-buffer 原地链解除 UB 约束（bs 6144→7168），
  尾块公式简化的贡献 <1/89，可忽略。
- 教训: 采纳 winner 前必须对热循环与尾块/epilogue 等全部重复执行路径做结构性
  diff，确认优化点覆盖主导路径，不能只看总体数值做平均归因。

## P.2 v2/v3 精化（Part A 统一显式 tanh；被中断 session 12:10-12:19 的工作）

- v2 = v5_op8 结构 + Part A 热循环统一为显式 tanh（即当前 mish.py 的代码
  结构，当时默认 bs=7168）；v3 = v2 + BLOCK_SIZE_CAST 7168→8192（08-24
  12:19 写盘，即当前 perf_opt/mish.py；本轮 resume 未做任何代码/常量改动）。
- v2 四组合 + 守门（r2_final_v2_*，08-24，median-of-15）:

| workload | task_duration_us | raw_profile_dir |
|---|---:|---|
| yolo_p3_fp16 (默认 7168) | 78.58 | profiles/manifest_r2/r2_final_v2_p3_fp16/ |
| yolo_p3_bf16 | 79.70 | profiles/manifest_r2/r2_final_v2_p3_bf16/ |
| yolo_p4_fp16 | 40.54 | profiles/manifest_r2/r2_final_v2_p4_fp16/ |
| yolo_p4_bf16 | 41.96 | profiles/manifest_r2/r2_final_v2_p4_bf16/ |
| fc_wide_fp16 / bf16 (守门) | 27.94 / 28.44 | r2_final_v2_fc_fp16 / r2_final_v2_fc_bf16 |
| smoke_fp16 / bf16 (守门) | 6.66 / 6.82 | r2_final_v2_smoke_fp16 / r2_final_v2_smoke_bf16 |

- bs 覆盖扫描（v2 文件 + --block-size 覆盖，yolo_p3_fp16，08-24 12:16-12:18）:
  7168(默认) 78.58 / **8192 75.34** / 8704 75.56 / 9216 77.96（均 n=15 紧分布）。
- v3 复测（r2_final_v3_*，08-24 12:21-12:31）: p3_fp16 78.84(首轮)/78.30(复测)、
  p3_bf16 81.50、p4_fp16 44.07(首轮 n=6)/41.02(复测)、p4_bf16 46.92(首轮
  n=1)/43.12(复测)、fc 27.24/28.22、smoke 7.01/7.08。首轮部分 run 采集缺失
  （n=11/6/4/1，日志含 StartFFTSTask "Profiling channel start failed"）为 NPU
  profiling channel 被占用的环境问题，以复测（n=15 紧分布）为准。
- 被 12:32-12:35 中断的 A/B 已采部分: incumbent p3_fp16 103.40、p3_bf16
  105.90、p4_fp16 53.80(n=5)——与 10:34 原值一致（无漂移）；另 v3 文件 +
  --block-size 7168 → 78.48（r2_ab_final_7168），与 v2 默认 7168 的 78.58
  一致（7168 配置跨文件一致）。

## P.3 75.34 vs 78.5 疑点调查（本轮核心任务）

现象: 同一结构（Part A/B 均显式 tanh）下，v2 文件 + `--block-size 8192` 覆盖
测得 75.34，v3 文件（内置 8192 默认）测得 78.3-78.8，名义同配置差 ~4%，
两次均为紧分布、非单点噪声。

调查过程与证据（2026-08-25 00:56-01:35，msprof op 协议同前，全部 n=15）:

1. **kernel 一致性证明（排除代码路径差异）**: 对当前 mish.py 分别以默认路径
   与 block_size=8192 显式覆盖构造 kernel，`get_kernel_source()` 输出完全一致
   （阴性对照 block_size=7168 不一致，检查敏感）。v2+覆盖8192 与 v3+默认8192
   走同一 factory 参数路径（block_size=8192），TIR 必然一致——两条测量路径
   编译出的是同一个 kernel。
2. **当日 A/B/A/B 交错复测（结果与昨日翻转）**:

| run | 配置 | median us | raw dir |
|---|---|---:|---|
| r2f_a_default_8192_r1 | 默认 8192 | **75.82** | r2f_a_default_8192_r1 |
| r2f_b_ovr8192_r1 | 覆盖 8192 | 78.78 | r2f_b_ovr8192_r1 |
| r2f_a_default_8192_r2 | 默认 8192 | 76.32 | r2f_a_default_8192_r2 |
| r2f_a_default_8192_r2_extract_only | 默认 8192 | 78.40 | r2f_a_default_8192_r2_extract_only |
| r2f_b_ovr8192_r2 | 覆盖 8192 | 78.46 | r2f_b_ovr8192_r2 |

   昨日"覆盖快/默认慢"，今日"默认快(2/3)/覆盖慢(2/2)"——快慢态与配置标签
   （默认 vs 覆盖）完全解耦。
3. **其它 bs 同日复测（快态在不同配置上出现）**: 7168: 79.42/77.28；
   7680: 78.34/77.30；8704: 77.74/76.34；**9216: 75.54**（昨日 77.96）——
   快态不是 8192 的专属属性。
4. **incumbent 锚点（排除全局环境漂移）**: incumbent p3_fp16 三跨 session:
   103.42(08-24 10:34) / 103.40(08-24 12:32) / **103.34(08-25)**，完全稳定。
5. **机制推断（与全部证据一致的最简解释）**: 新 kernel 有效带宽 1.33-1.39
   TB/s（75.5-78.9us / 105MB IO），逼近 HBM 上限，对 per-process 的 HBM 物理
   页分配 / DDR 通道交织状态敏感——bench.py 每个 run 重新分配 4 份输入克隆
   与输出，同 run 内 15 次 launch 复用同一批物理页（run 内分布紧 <0.6us），
   跨 run 落入快/慢两种页布局（~4%）。incumbent 103.4us ≈ 1.02 TB/s，被串行
   向量链支配、远离带宽上限，故对页状态免疫、跨 session 稳定。

**结论: 75.34 是快态 run 的真实测量，但不是 override 路径或 bs=8192 的可复现
属性**；分类为 run-level environment bimodality（unreproducible-on-demand、
配置无关、incumbent 免疫）。最终数值以多 run 中位数（保守、可复现）为准，
快态 band 如实记录为环境性上行空间。可复现性教训: 同一 kernel 单次紧分布
测量不足以支撑 <3% 的配置归因。

## P.4 bs 终选

合并两 session 全部复测（yolo_p3_fp16，新 kernel）:

| bs | 合并中位 us (次数) | 全部观测 |
|---:|---:|---|
| 7168 | 78.53 (4) | 78.58, 78.48, 79.42, 77.28 |
| 7680 | 77.82 (2) | 78.34, 77.30 |
| 8192 | 78.35 (8) | 75.34, 78.84, 78.30, 75.82, 76.32, 78.40, 78.78, 78.46 |
| 8704 | 76.34 (3) | 75.56, 77.74, 76.34 |
| 9216 | 76.75 (2) | 77.96, 75.54 |

- 各 bs 合并中位数最大差异 2.19us（2.8%，<3% 采纳门槛），且被 run 级双态
  噪声（±1.5-3.5%，见 P.3）淹没；7168 在 4 次复测中从未出现快态，但其
  合并中位与 8192 仅差 0.18us（噪声内）。
- 结论: **维持 BLOCK_SIZE_CAST=8192**（当前 mish.py 无需改动）。今日新增
  final 复测 p3_bf16 80.86、p4_bf16 42.74 与昨日 v3 值一致（文件未变），
  佐证 v3 数值可复现。

## P.5 A/B 完整表（防漂移复验 + 本轮补全）

- 被中断 session 的 A/B（08-24 12:32-12:35）: incumbent p3_fp16 103.40、
  p3_bf16 105.90、p4_fp16 53.80(n=5)，与 10:34 原值一致（无漂移）；
  incumbent p4_bf16 当时未测。
- 本轮补测: **incumbent p4_bf16 54.92**（r2f_ab_incumbent_p4_bf16，n=15，
  与原值 55.18 一致）+ 今日环境锚点 incumbent p3_fp16 103.34（r2f_anchor
  _incumbent_p3_fp16）+ 今日 final p3_bf16 80.86 / p4_bf16 42.74
  （r2f_final_p3_bf16 / r2f_final_p4_bf16）。

完整 A/B 表（msprof Task Duration，median-of-15）:

| workload | incumbent (R1 final) | final (v3) | 提升 |
|---|---:|---:|---:|
| yolo_p3_fp16 (N=26,214,400) | 103.42（A/B 103.40 / 锚点 103.34） | 78.30（5 复测中位; band 75.3-78.9） | **-24.3%** |
| yolo_p3_bf16 | 105.90（A/B 105.90） | 81.18（81.50 / 80.86） | **-23.3%** |
| yolo_p4_fp16 (N=13,107,200) | 53.84（A/B 53.80） | 41.02 | **-23.8%** |
| yolo_p4_bf16 | 55.18（本轮 A/B 54.92） | 42.93（43.12 / 42.74） | **-22.2%** |
| smoke_fp16 (守门, N=1,048,576) | 7.44 | 7.01 | -5.8%（改善） |
| smoke_bf16 (守门) | 7.66 | 7.08 | -7.6%（改善） |
| fc_wide_fp16 (守门, N=8,388,608) | 35.92 | 27.24 | -24.2%（改善） |
| fc_wide_bf16 (守门) | 36.52 | 28.22 | -22.7%（改善） |
| smoke_fp32 (守门) | 8.56 | 8.56（路径字节级未改） | 0 |

raw dirs: incumbent → profiles/manifest/{yolo_p3_fp16,yolo_p3_bf16,yolo_p4_fp16,
yolo_p4_bf16,fc_wide_fp16,fc_wide_bf16} + profiles/manifest_r2/{r2_ab_*,
r2f_anchor_incumbent_p3_fp16, r2f_ab_incumbent_p4_bf16}；final →
profiles/manifest_r2/{r2_final_v3_*, r2f_final_*, r2f_a_default_8192_*}。

## P.6 最终精度回归（本轮复测）

`python mish.py --level all`（2026-08-25）: L0 三 dtype PASS（fp16 max_diff
2.44e-04 / fp32 4.32e-07 / bf16 0.00e+00）；L1 全 PASS；L2 全 PASS；
Boundary zeros/large-positive/large-negative PASS；overflow-domain（x=100）
WARN（NaN 行为，与 incumbent 逐位一致，Round 2 已记录为非回归）。

## P.7 最终结论（Round 2 终态）

- final = `perf_opt/mish.py`（v3: Part A/B 统一显式 tanh + guard-free
  Pipelined 热循环 + 尾块 epilogue + 2-buffer 原地链 + bs=8192，Developer
  模式）。本轮仅更新头注 docstring（Measured 数值与调优描述），无任何
  代码/常量改动，故未新增 history_version 备份。
- 相对 Round 2 基准（= Round 1 final）: 四目标组合 **-22.2% ~ -24.3%**
  （快态 run 最高 -27.1%），守门 shape 全部无回退且多数改善，fp32 路径
  字节级未动。
- 相对原始基准 ../mish.py（p3_fp16 547.6us）: -85.7%。
- stop_reason: **success**（四目标组合远超 3% 采纳门槛且守门无回退；剩余
  更大空间仍被平台约束封死——Developer tensor-compile 破坏手动 rs/flag
  流水、同步发射语义使 flag 协议无并发；Expert 91.4us 方案保留记录待约束
  放开后启用）。
- 可复现性声明: p3_fp16 终值 78.30us 为 5 次独立 run 的中位数；该 kernel
  逼近 HBM 带宽上限，存在配置无关的 run 级双态（快态 75.3-76.4 / 慢态
  78.3-78.9），后续复测应以多 run 中位数比较。

## Skill Retrospective 补充（Round 2 post-scriptum）

- 流程问题 3: 单 run 的 median-of-15 紧分布（<0.6us）会给出虚假的测量置信
  度——run 之间还存在 ±2-3.5% 的环境双态层。任何 <3% 的配置间结论都必须
  基于 ≥3 次独立 run + 跨 session 锚点（incumbent）校准后才可下判；本次
  75.34 vs 78.5 疑点消耗了一轮调查才定位到该层。
- BP_run_state_bimodality（新增）: 逼近 HBM 带宽上限的 kernel 在共享 16-NPU
  环境下存在 per-process 快/慢双态（疑似 HBM 物理页/DDR 通道交织状态；同
  run 内复用同一批物理页，故 run 内分布紧）。建议 Profile-collection.md
  增补"带宽受限 kernel 的多 run 复测协议"：≥3 次独立 run、报告 band 与
  中位数、用 anchor 配置（如 incumbent）监测环境层漂移。
- BP_attribution_diff（新增，来自 P.1 勘误）: winner 采纳前应做结构 diff
  检查单——确认优化点同时覆盖热循环与尾块/epilogue 等全部重复执行路径，
  避免"平均归因正确、主导路径遗漏"的假 winner。
- 既有 BP 维持: BP_vtanh_mode_inconsistency、BP_transcendental_sync_required、
  BP_string_imm_codegen、BP_manual_pipeline_developer_broken、
  BP_launch_overhead（见 Skill Retrospective (Round 2)）。
