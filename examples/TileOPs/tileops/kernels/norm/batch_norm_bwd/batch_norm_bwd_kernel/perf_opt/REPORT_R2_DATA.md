# BatchNormBwdOp R3 深挖数据报告（供主会话合并进 OPTIMIZE_REPORT.md）

- 算子：`BatchNormBwdOp`（batch norm 训练反向：grad_x / grad_weight / grad_bias）
- 目录：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/`
- 环境：Ascend910B3（20 AIC / 40 AIV）、CANN 9.0.0、tilelang 0.1.2 dev-root、`msprof op` 唯一时延口径
- 轮次：Stage 4 结构级二轮（R3），上承 R2（subid 分片，−28~−46%）
- 目标：current best 再降 10–20% Task Duration；结果：**−12.3% ~ −19.8%**（同 NPU 会话配对，6/6 workload 均为净收益）

## 1. 诊断证据（profile 数值 + 归因）

### 1.1 起点复核与假设证伪

| 候选 | stage1-fp16 | stage1-bf16 | stage2-fp16 | stage3-fp16 | stage3-fp32 | large-spatial |
|---|---:|---:|---:|---:|---:|---:|
| Stage-3 baseline（重测） | 21.190 | 20.500 | 25.300 | 41.040 | 43.180 | 5373.030 |
| R2 final = R3 起点（重测） | 13.270 | 13.800 | 16.190 | 23.330 | 24.820 | 2906.750 |

- 重测与 R2 记录偏差 ≤3%（起点可信）。
- **证伪**：调度假设「`N=8,S=1024` 得 bS=928 → 2 tile → 5 遍流量」。实际 `_plan_bS(8,1024,22,transit)` = `min(1024, align_down(184320//(8·22), 8)) = min(1024, 1040) = 1024` → resnet-stage1 早已走 held **single-scan**；该方向无实验价值，未做。

### 1.2 wide（resnet-stage1-fp16，R2 best）

- Task Duration 13.270µs；`aiv_time` 9.98µs（75%），`vec_time` 4.848（ratio 0.373），`mte2_time` 1.9，`mte3` 0.8，`aiv_scalar_ratio` 0.23–0.31，`aiv_scalar_wait_time` 6.0µs。
- **Σpipe ≈ aiv_time** → 单 staging 缓冲复用链把「MTE2 → VCAST」串成一条链（PL-1.10 形态），无跨 pipe 重叠。
- 每元素向量 op ≈ 17（2 vcast + 5 vbrc + 7 arith + 2 reduce + 1 vcast）；`aiv_time` 之外 3.3µs = 发射/尾波（26%）。
- 波量化：C=64/128/256 → 2/4/7 波，AIV 利用率 80%/80%/91%（通道粒度已到 1 通道/任务，进一步均衡需跨 AIV 归约 → 5-pass 反噬）。

### 1.3 split（large-spatial-fp16，R2 best）

| stage | Task(µs) | vec_time | mte2_time | mte3_time | aiv_time | Σpipe/aiv |
|---|---:|---:|---:|---:|---:|---:|
| stats | 1285.1 | 740.8 | 394.1 | 1.8 | 1274.0 | 1552/1274（22% 重叠） |
| finalize | 10.0 | 0.19 | 1.5 | 3.4 | 5.6 | — |
| apply | 1611.6 | 898.4 | 496.4 | 161.3 | 1601.6 | 1555/1601（**零重叠**） |

- 有效带宽：stats 读 536MB/394µs = 1.36 TB/s；apply 读 536MB/496µs = 1.08 TB/s，写 268MB/161µs = 1.66 TB/s → 均未饱和设备混合流地板（≈1.7 TB/s）⇒ apply 是**串行**问题，不是带宽问题。
- 每元素向量 op：stats 9、apply 12（2 vcast + 5 vbrc + 4 arith + 1 vcast）。

## 2. 方法与逐轮记录

| 分支 | 单点改动 | L0 | 结果（Task Duration） |
|---|---|---|---|
| W1 | wide：消去逐元素 `x_hat`，`gw = rstd·Σ(g·x) − (mean·rstd)·Σg`，`grad_x = A·g + B·x + C`（17→13 op） | PASS | 12.010 / 13.900 / 15.430 / 22.490 / 22.530（4/5 净收益，bf16 打平） |
| S1 | split：apply per-input staging（loads-first） | PASS | 2715.840（−6.6%） |
| W2 | W1 + wide per-input staging + `brc`→`prod` 复用（UB 中性） | PASS | 12.120 / 12.400 / 14.610 / 19.170 / 20.280 |
| S2 | S1 + apply epilogue 折叠 + apply 独立 tile 宽 | PASS | 2528.010（−6.9%） |
| S3 | S2 + stats per-input staging | PASS | **2333.080（−7.7%）** |
| S4 | S2 + apply tile 循环 `T.Pipelined(2)` | PASS | 2528.350（tie → config_no_gain，弃） |

关键 profile 变化（wide stage1）：`aiv_time` 9.98→8.07µs、`vec_time` 4.85→3.77µs；split：stats 1285→1091µs（mte2 394→271）、apply 1612→1233µs（mte2 496→314，vec 898→724）。

## 3. 逐 workload base/final 与 Δ（与 `perf_records.jsonl` 一致）

| kernel_id | workload_id | baseline_us | final_us | Δ% | R2 best→R3 final（同会话） |
|---|---|---:|---:|---:|---:|
| _batch_norm_bwd_wide_kernel | resnet-stage1-float16 | 20.520 | 12.600 | −38.6 | −12.3% |
| _batch_norm_bwd_wide_kernel | resnet-stage1-bfloat16 | 21.160 | 11.480 | −45.7 | −16.7% |
| _batch_norm_bwd_wide_kernel | resnet-stage2-float16 | 25.440 | 14.320 | −43.7 | −15.0% |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float16 | 41.260 | 20.690 | −49.9 | −12.4% |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float32 | 41.680 | 20.880 | −49.9 | −13.5% |
| _batch_norm_bwd_split_kernel | large-spatial-float16 | 5375.400 | 2333.520 | −56.6 | −19.8% |

同会话配对原始值（`run_r3_ab.py`）：14.360/13.780/16.840/23.630/24.130/2911.340 → 12.600/11.480/14.320/20.690/20.880/2333.520。

**测量纪律说明**：同一二进制两次独立运行在 wide 小 shape 上可差 ~7%（12.12 vs 13.02µs，各自分布 min–max ≤5%），与目标增益同阶，故最终比值取「同 NPU 会话内 A/B 交替」结果；绝对时延取 final 侧记录。SOL（roofline `GM Read+Write`）：11.4% ~ 33.3%，60% 软目标未达（见 §7）。

## 4. 精度

| 项 | 结果 |
|---|---|
| `python perf_opt/BatchNormBwdOp.py --level all` | L0 14/14、L1 14 grid + 4 契约、L2 4、boundary 全 PASS |
| smoke-fc [32,64,1] fp32 / fp16 / bf16 | PASS，max_abs_err 1.431e-06 / 1.907e-06 / 1.907e-06 |
| 长归约（宽核 S=1048576 fp16，折叠路径最不利） | PASS，max_abs_err 3.906e-03 |
| 单 kernel L0（wide/split/general） | 全 PASS（split 13 项、wide 14 项） |

## 5. 产物与 sha256

| 文件 | sha256 |
|---|---|
| `perf_opt/BatchNormBwdOp.py`（final 记录 artifact） | `37099e2678f17ee65db331315d2175206a1a36e0409f5863f36562b2c396c37c` |
| `perf_opt/_batch_norm_bwd_wide_kernel.py` | `b0b1a1ab35758a765d88d9472cc8f1e4566534e1ca44f773274224ae91c38aa1` |
| `perf_opt/_batch_norm_bwd_split_kernel.py` | `e284801e632ab9c3e0c5c81790c12bfdb2bfd05a39fb7035e2d625934b72f90b` |
| `perf_opt/_batch_norm_bwd_kernel.py`（general，无 benchmark 命中） | `ffb95322d988ede3efe1f127b35d7b24e97f668c5b070a48b16665cf9a085aa3` |

分支文件：`_batch_norm_bwd_wide_kernel_r3_{w1,w2}.py`、`_batch_norm_bwd_split_kernel_r3_{s1,s2,s3,s4}.py`；runner：`run_experiments.py`（T-4）、`run_r3_ab.py`（同会话 A/B）、`diag_parse.py`（pipe 汇总）、`bench_kernel.py`；探针 `probe_scalar_plane.py`。raw profile：`perf_opt/profiles/`；日志：`perf_opt/logs/r3/`。

## 6. 门禁结果

- `python3 .agents/tools/statectl.py gate 4 --dir <op dir>`：pass（详见返回摘要）。
- L0/L1 无阻塞失败；final 记录 6 条覆盖全部 tune workload，同一 candidate_id `r3_final_w2s3`，artifact_path 均指向 `BatchNormBwdOp.py`。

## 7. 未竟项 / 天花板证据

1. **apply 的 pipe 重叠仍为零**（`aiv 1227 ≈ vec 724 + mte2 314 + mte3 179`）：`T.Pipelined(num_stages=2)` 无收益（tie）。把 3 个 vbrc 提到 tile 循环外需 3×(2,3400) fp32 = 81.6KB 常驻，超 UB 余量（nominal 136KB / 192KB，含 multi-buffer 膨胀后无空间）→ 需更小 tile 换取 hoist 的权衡，未实证（潜在收益 ~−10~15% apply）。
2. **wide 的 scalar 段**（ratio 0.23–0.31，逐通道 5 次 1 元素 copy + strided 寻址）：批量取参需「UB 标量读入」形态，`T.vmul`/`T.vadd` 文档 §2.3 限制「标量操作数须在 `T.Scope` 外定义」→ 当前 API 无合法形态（未实证假设，未启动）。
3. **wide 发射/尾波 3.3µs（26%）**：小 kernel 固定开销（PL-1.22 截距 ~1.6µs + 第二波尾）；减小需改变任务粒度（跨 AIV 归约 → 5-pass），本 shape 不划算。
4. **SOL 60% 不可达（在冻结设计族内）**：fp32 中转（bf16 无 Vector 算术 + fp16 对齐 fp32 golden，正确性硬约束）+ 逐通道 vbrc + 3/5 遍流量；R3 后 `aiv_time ≈ Σpipe`，无重叠余量可挖。
5. **不触发 `[DESIGN_LIMIT]`**：上述缺口均无「被设计冻结、可 >2x 的备选结构」（Cube 归约仍为无实测先例的未实证假设），不满足 `perf-feedback.md` §1 双门槛，未产出 `perf_feedback.md`。
