# AddFwdOp — Stage-4 R5 数据报告（`linear-bias bf16` 冲线）

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> （`/mnt/workspace/tilelang-mlir-ascend`，AscendNPU-IR `31f6903`）/ torch 2.7.1 / torch_npu 2.7.1.post4 /
> **Ascend910B3**（Chip 6，20 AIC / 40 AIV）/ 2026-10-21。
> 时延口径：**`msprof op` Task Duration(us)**（唯一口径）；ratio = `visualize_data.bin` 的
> 「GM Read + Write」SOL ratio。所有 NPU 命令经 `npu_lock.sh` 串行化。
> 仓库根：`/mnt/workspace/tile-ascendnpu-ops`（本次搬迁后首次重采，见 §6）。

## 1. 诊断（本轮决定性证据）

目标：端到端唯一 <70% 的 `linear-bias bf16`（69.37%，差 0.63pp ≈ 需再快 0.9%）。
先看 current best 的 profile，而不是先猜候选：

| 指标（`linear-bias`，round-10 final，PipeUtilization） | bf16（14.050µs / 69.17%） | fp16（13.420µs / 73.41%） | 判读 |
|---|---:|---:|---|
| `aiv_time` 每核忙时 | 10.748µs | 10.485µs | 同量级 |
| `aiv_mte2_time` / ratio | 9.939µs / **0.925** | 9.821µs / **0.937** | **MTE2 接近占满 → DMA 受限** |
| `aiv_mte2_active_bw` | 39.9 GB/s/核 | 40.4 GB/s/核 | 带宽效率一致，无「浪费」 |
| `aiv_vec_time` / ratio | 6.019µs / 0.560 | 1.119µs / 0.107 | bf16 的 fp32 中转链占 56% 时间，但被 MTE2 掩盖 |
| `aiv_mte3_time` | 4.166µs | 3.838µs | store 有余量 |
| 有效带宽（3N×2B / TaskDur） | 1.79 TB/s | 1.87 TB/s | 大 shape 2.1–2.2 TB/s → **固定截距 ≈1.6µs（12%）** |

per-tile 账（N=4,194,304，block=8192，40 核 → 13 tile/核）：每 tile 2×16KB load + 1×16KB store，
MTE2 ≈ **0.35µs/tile**；VEC（bf16 中转 4 pass）≈ **0.10µs/tile** → 3.5x 余量。

**结论**：该 shape 的结构上限由 MTE2（HBM 流量）决定，**任何「减少片上操作」的改动都不会被看到**；
唯一有量级收益的方向是减少 GM 流量（见 §5）。这也把 69.17% 定性为「结构 + bf16 dtype 成本的地板」，
而不是一个可以被参数微调跨过的缺口。

## 2. 方法

- 固定 base：`perf_opt/AddFwdOp.py`（A4 静态 extent + `T.Pipelined(2)` + 40 核 + fp16 `alpha==1` 原生 vadd，
  sha256 `0392900e…fbecc`，**本轮未改写**）。
- 每个分支只改一个主要优化点，候选文件由 `make_r5_candidates.py` 从 base 文本派生（diff 可审）：
  | 分支 | 文件 | 唯一改动 |
  |---|---|---|
  | R5-B1 | `AddFwdOp_round5_b1_native_bf16.py` | `alpha==1` 原生 `T.vadd` 扩展到 bf16 |
  | R5-B2 | `AddFwdOp_round5_b2_fp32_direct.py` | `alpha==1` fp32 直通（去掉 3 次无信息 UB copy） |
  | R5-B3 | `AddFwdOp_round5_b3_allnative_pipe3.py` | B1+B2 合并 + `num_stages=3`（未进入 A/B，B1 已 tie） |
  | R5-B4 | `AddFwdOp_round5_b4_pipe3.py` | `num_stages` 2→3 |
  | R5-bs4096 / bs12288 | 同 base，仅 `--block-size` | tile 宽度 |
- 判定协议：`.agents/tools/ab_test.py`，**同会话交错 A/B/A/B**、每对内交替先后、每次测量
  20 次 launch 的 Task Duration 中位；合并中位 + 配对符号检验；噪声阈值 3%。
- 每个分支先跑 `--level L0`（含 NaN/±inf、`alpha!=1`、广播、尾块），再进入 A/B。
- 分支记录逐条 append 到 `perf_records.jsonl`（round 5，7 行，取自各分支 A/B arm 的 `pair1B` 测量中位）。

## 3. 逐 workload baseline / final（同一最终候选 `tuned-pipe2-bs8192`）

round-10 final（`run_experiments.py --phase final --round 10 --all-tune`，逐 workload 独立 msprof 进程）；
baseline 为 round-9 的 baseline 记录（同一候选体系，未变）。

| workload | baseline µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 38.421 | 37.290 | −2.94% | 77.43 | 80.57 | ✅ |
| cnn-feat-broadcast-float16 | 38.531 | 36.050 | −6.44% | 76.34 | 81.15 | ✅ |
| cnn-feat-broadcast-float32 | 71.661 | 71.040 | −0.87% | 82.26 | 84.03 | ✅ |
| hidden-state-prefill-bfloat16 | 25.761 | 25.520 | −0.94% | 75.19 | 75.08 | ✅ |
| hidden-state-prefill-float32 | 47.321 | 46.700 | −1.31% | 81.97 | 81.25 | ✅ |
| **linear-bias-bfloat16** | 13.780 | **14.050** | **+1.96%** | 69.47 | **69.17** | ❌ |
| linear-bias-float16 | 13.350 | 13.420 | +0.52% | 72.63 | 73.41 | ✅ |
| linear-bias-float32 | 24.570 | 24.230 | −1.38% | 78.13 | 79.63 | ✅ |
| llama-7b-ffn-bfloat16 | 34.181 | 32.690 | −4.36% | 74.85 | 79.08 | ✅ |
| llama-7b-ffn-float16 | 33.871 | 32.280 | −4.70% | 75.51 | 80.82 | ✅ |
| llama-7b-ffn-float32 | 63.371 | 63.040 | −0.52% | 81.51 | 81.92 | ✅ |
| **合计** | **404.818** | **396.310** | **−2.10%** | — | — | **10/11** |

`linear-bias bf16` 的 +1.96% 是跨进程噪声（同版本 5 次独立测量落在 13.55–14.05µs，ratio 69.17–70.80%），
不是本轮引入的回退——R5 未改动 kernel 文件。

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

## 4. 配对 A/B（同会话交错，20 launch/次）

| # | 对比 | workload | pairs | A（current best）合并中位 µs | B（候选）合并中位 µs | Δ(B vs A) | 逐对差 µs | p | verdict |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 1 | best vs **B1 bf16 原生** | linear-bias bf16 | 6 | 13.670 | 13.645 | −0.18% | −0.22, −0.67, +0.55, +0.07, +1.18, −0.39 | 1.000 | tie |
| 2 | best vs B1 | hidden-state-prefill bf16 | 5 | 25.260 | 24.880 | −1.50% | −0.42, +0.50, −0.62, +0.64, −0.57 | 1.000 | tie |
| 3 | best vs B1 | cnn-feat-broadcast bf16 | 5 | 37.820 | 36.660 | −3.07% | −1.60, +1.45, −1.26, +1.27, −1.28 | 1.000 | inconclusive（顺序效应） |
| 4 | best vs B1 | llama-7b-ffn bf16 | 5 | 32.950 | 33.090 | +0.42% | −0.38, +0.33, −0.24, +0.41, +0.76 | 1.000 | tie |
| 5 | best vs **B2 fp32 直通** | linear-bias fp32 | 4 | 24.655 | 24.600 | −0.22% | −0.63, +0.48, −0.58, +0.46 | 1.000 | tie |
| 6 | best vs **B4 pipe3** | linear-bias bf16 | 6 | 13.915 | 13.730 | −1.33% | −0.22, +0.36, +0.41, −0.17, −0.38, −0.27 | 0.688 | tie |
| 7 | best bs8192 vs **bs4096** | linear-bias bf16 | 4 | 13.555 | 13.885 | +2.43% | +0.33, −0.61, +0.81, +0.23 | 0.625 | tie（候选更慢） |
| 8 | best bs8192 vs **bs12288** | linear-bias bf16 | — | 13.910 | — | — | — | — | ❌ 编译失败：`ub overflow, requires 245,760B while 196,608B available` |
| 9 | **fp16 原生 vs bf16 原生**（诊断） | linear-bias | 4 | 13.140（fp16） | 13.955（bf16） | **+6.2%** | +0.60, +1.63, +0.91, +0.40 | 0.125 | **A_better（方向 4/4）** |

原始 JSON：`logs/r5/ab_*.log`（含 `a_medians_us` / `b_medians_us` / `pair_diffs_us` / `sign_test_p`）；
profile：`ab_r5/<tag>/pair*{A,B}/OPPROF_*`。

**顺序效应说明**：#3 的逐对差符号与「对内先后」完全同步翻转（偶数对 A 先 → B 快 1.3–1.6µs；
奇数对 B 先 → A 快 1.3–1.5µs），属于「后跑的一臂更快」的整机漂移，不是候选差异。
#1/#6 为 3:3、#5/#7 为 2:2、#9 为 2:2（顺序平衡），其结论可用；#2/#3/#4（5 对 = 3:2）残留偏差。

## 5. 关键结论与「唯一跨线路径」

**5.1 B1（bf16 原生 vadd）为什么否决——profile 证据**

| 版本 | `aiv_vec_time` | `aiv_mte2_time` | Task Duration（6 对合并中位） |
|---|---:|---:|---:|
| current best（fp32 中转） | 6.019µs | 9.939µs | 13.670µs |
| R5-B1（bf16 原生） | **5.986µs** | 10.374µs | 13.645µs |

native 与中转的 VEC 时间**完全一致** → CANN 9.0 的 bf16 `T.vadd` 被后端降级为等价 fp32 仿真链
（不是原生 bf16 加法单元）。这解释了上一轮「探针 −2.0%/−2.8%、官方 +3.8%」的方向矛盾：
**不存在可复现的收益，全部落在噪声内**。逐位相等虽成立（§6），但按「无收益不采纳」处理。

**5.2 bf16 的 6.2% dtype 成本**

同结构、同流量、同 shape、顺序平衡的 4 对：fp16 原生 13.140µs vs bf16 原生 13.955µs
（4/4 一致）。即 `linear-bias` 的贴线是 **bf16 VEC 仿真链 + 同带宽下总时长 +4.7%** 造成的；
结构上没有可再压的空间（MTE2 ratio 已 0.925–0.937）。

**5.3 唯一有量级收益的方向：broadcast 行复用（本轮只做探针，不改 Op 接口）**

`linear-bias` 的 `other=[1,4096]` 在 Op 层被 `AddFwdOp._expand_flat` 的 `expand + contiguous` 物化成
`[1024,4096]`，kernel 因此把同一 8KB 行读了 1024 次 → 8MB 冗余读，占 3N 流量的 **1/3**。
探针 `r5_bcast_probe.py`（把行 `(4096,)` 直接传入，UB 内驻留后按 tile 复播）：

| 版本 | GM 流量 | Task Duration（20 launch 中位） | 输出校验 |
|---|---:|---:|---|
| current best（物化 other） | 24 MB | 14.050µs | 参照 |
| **行复用探针** | **16 MB** | **8.980µs（−36.1%）** | `bitwise_equal=True`，max_diff=0 |

要落地必须同时改三处（**均超出 Stage-4 只读边界，交主会话决策**）：
① `tileops/ops/elementwise/add.py::_expand_flat` 跳过可复用的物化；
② `add_fwd_kernel` 签名增加 `row`（`block_size % row == 0` 且 tile 偏移按行对齐）；
③ `eval_roofline` 的字节口径 `3*N*elem` → `2*N*elem + row*elem`
（否则 ratio 会显示 **106.8%** —— 分母仍是 3N；按修正分母则 **≈71.2%**，仅高出 70% 线 1.2pp）。
附带收益：省掉 Op 层每次调用的 8MB `contiguous`（不进 ratio 分母，但计入端到端墙钟）。

## 6. 精度

| 项 | 结果 | 证据 |
|---|---|---|
| `python3 perf_opt/AddFwdOp.py --level all`（最终文件） | **L0/L1/L2/Boundary 全 PASS，`All check passed!`** | `logs/r5/final_level_all.log` |
| R5-B1 L0（含 bf16 原生、NaN/±inf、α≠1、广播、尾块） | PASS | `logs/r5/l0_AddFwdOp_round5_b1_native_bf16.log` |
| R5-B2 / B3 / B4 L0 | PASS | `logs/r5/l0_AddFwdOp_round5_*.log` |
| **bf16 原生 vs fp32 中转逐位相等** | **`all_bitwise_equal=true`，max_diff=0** | `logs/r5/r5_bitwise_b1_bf16.log` |
| └ 用例 | `linear-bias` 全量 4×4,194,304（4 个随机种子）+ 对抗样本（指数差 200 / 70 / bf16 舍入中点邻域），各 1,048,576 | 同上 |
| 行复用探针输出 | `bitwise_equal=True`（vs `torch.add(a.float(), expand(b).float()).to(bf16)`） | `r5_bcast_probe.py --check` |

逐位相等的语义依据（bf16）：两个 bf16 的精确和在指数差 ≤2²⁴ 时可由 fp32 精确表示，
此时「fp32 加 + 一次舍回 bf16」与「单次 bf16 舍入」等价；指数差更大的输入在两个实现里都丢掉小操作数，
结果仍相同。对抗样本覆盖了上述两类边界。

## 7. 产物与哈希

| 项 | 值 |
|---|---|
| final kernel | `examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/perf_opt/AddFwdOp.py` |
| sha256 | `0392900e48ebc78c54e8aba84ad3e9808d1303a2733a6598ba6e813c8c0fbecc`（与 round-9 相同，本轮未改） |
| 候选分支 | `AddFwdOp_round5_b1_native_bf16.py` / `_b2_fp32_direct.py` / `_b3_allnative_pipe3.py` / `_b4_pipe3.py` |
| 生成器 / 辅助 | `make_r5_candidates.py`、`r5_append_records.py`、`r5_bitwise_probe.py`、`r5_bcast_probe.py`、`run_r5_ab.sh` |
| 记录 | `perf_records.jsonl`：round 5 追加 7 条 candidate；round 10 追加 11 条 final；round-9 的 11 条 final **仅降级为 merged**（数值/哈希未动） |
| profile | `profiles/final-r10-*`（11 条）、`profiles/r5-bcast-bf16-bs8192`、`ab_r5/*/pair*{A,B}` |
| 日志 | `logs/r5/`（L0、A/B JSON、逐位校验、level-all） |

**仓库搬迁说明**：round-9 的 final 行记录的是旧前缀 `/mnt/workspace/npuir-migration/...`，
gate 4 `S4-PERF-RECORDS-RECON` 要求「final 行的 `artifact_path` == 当前 `perf_opt/{op}.py`」，
搬迁后必然失配。处理方式：历史 final 行仅改 phase → `merged`，再用新路径重采 round-10 final
（与 round 4/6/8 的历史降级一致）。**建议后续 runner 写仓库相对路径**。

## 8. Gate 4 与自检

- `python3 .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/elementwise/add/add_kernel`
  → 结果见 `logs/r5/gate4.json`（`pass`）。
- final 覆盖 11 个 tune workload、同一候选 `tuned-pipe2-bs8192`、artifact 指向 `perf_opt/AddFwdOp.py`
  且哈希一致；smoke 精度 PASS 记录在 `workload_inventory.json`。

## 9. 未竟项 / 建议

1. **`linear-bias bf16` 仍未过 70%（69.17%）**，本轮判定为「结构 + bf16 dtype 成本地板」，
   且 0.63–0.83pp 的差距小于跨进程噪声（±5%）与 A/B 可分辨度（≈1.5%）。
2. **唯一能量级改善的路径需要改 Op 接口**（§5.3）：−36.1% kernel 时延 / ratio 修正后 ≈71.2%；
   同时影响 `cnn-feat-broadcast`（`[256,1,1]` 行内标量广播）需要另一条路径，建议单独立项。
3. 若主会话接受「按广播语义修正 roofline 分母」，需同步 `manifest`/`eval_roofline` 与
   `19_operators_ratio` 的口径，避免该 case 出现 >100% SOL 的展示异常。
4. `ab_test.py` 建议对奇数对增加顺序效应提示（本轮 #3 即为典型）。
5. `block_size=12288` 在本结构下必然 UB overflow（245,760B > 196,608B），不再重试。
