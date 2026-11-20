# CosFwdOp 优化工作提交清单（Stage 4）

本目录是 `CosFwdOp`（cos 前向、一元逐元素）的 **Stage 4 性能优化**产物。
提交范围**只含 Cos 算子**：本算子目录下的交付件、候选、探针、复算脚本与调优记录。

## 一、最终结果（官方 pytest 口径）

| 口径 | 基线（C3） | 优化后 | 提升 |
|---|---:|---:|---:|
| 单点 ratio（7 条 workload 最高） | 56.17% | **67.98%** | +11.81 |
| 平均 ratio（非 smoke，排除 <15 µs） | 54.40% | **63.96%** | +9.56 |
| 逐 workload ≥60% | 0 / 7 | **6 / 7** | — |

逐 workload（官方 `profile_run_msprof_20261102_074601.json`）：

| workload | 时延 | ratio |
|---|---:|---:|
| 16m fp16 | 96.98 µs | 62.70% |
| 16m bf16 | 98.88 µs | 64.56% |
| 16m fp32 | 107.96 µs | 53.53%（唯一 <60） |
| 64m fp16 | 371.47 µs | 65.45% |
| 64m bf16 | 378.08 µs | 67.52% |
| 256m fp16 | 1474.54 µs | 65.95% |
| 256m bf16 | 1501.84 µs | 67.98%（单点来源） |

## 二、改了什么

| 文件 | 性质 | 说明 |
|---|---|---|
| `cos.py`（wrapper，上一级目录） | **修改** | `_DEFAULT_BLOCK_SIZE`：`{4096,4096,4096}` → `{fp32: 8192, fp16: 12288, bf16: 12288}`（用户授权的契约例外；不改则正式 benchmark 只能拿 4096 档） |
| `perf_opt/cos_fwd_kernel.py` | **修改（交付件）** | 新增 `_cos_single`（无三角函数 poly-R 链）、`_cos_f32_dual`（fp32 双流、远距映射）、`cos_fwd_kernel` dtype 分发、`TUNED_DEFAULT_BLOCK_SIZE` |
| `perf_opt/opt_log.md`、`perf_opt/perf_records.jsonl` | **追加** | 第 5 / 6 / 7 轮（记录 append-only：round=5/6/7，含 `phase=official`） |

**两条核心机制**：
1. **换掉贵指令**：`cos(x) = 1 − 2·sin²(x/2)`，按 π 归约让 sin² 吸收符号 ⇒ 无 `vsin`、无象限奇偶（`vsin` 一项占 40% 周期、记 9 个记账单元）；
2. **摊薄每 tile 固定开销**：`block_size` 按 dtype 分档 4096→12288（+8.1 pp），fp32 因双流需 24 B/元素只能 8192。

## 三、本目录文件清单

| 文件 | 作用 |
|---|---|
| `cos_fwd_kernel.py` | **交付件**（含 dtype 分发与 tuned 常量） |
| `cos_fwd_kernel_round5_c3base.py` | C3 基线存档（改动前） |
| `cos_fwd_kernel_round5_polyr.py` / `_polyr2.py` / `_polyr3.py` / `_polyr4.py` | 第 5 轮四个分支（poly-R 链 / fp32 直落 / deg-7 / Pipelined） |
| `cos_fwd_kernel_round6_dual.py` | 第 6 轮 fp32 双流（相邻映射，89.86 µs） |
| `cos_fwd_kernel_round6b_far.py` | 第 6 轮 fp32 双流（**远距映射，交付用**） |
| `cos_fwd_kernel_round7_slim.py` | 第 7 轮 3 缓冲瘦身（256m 回退 −8.11 pp，**已否决**，留证） |
| `probe_instr.py` | 指令成本标定（8/12/15/16/20 条纯乘链 + 特殊指令变体） |
| `probes/probe_novsin.py`、`probes/probe_nocast.py` | 真实链消融（vsin / 量化 cast 的单独成本） |
| `ratio_from_bin.py` | **ratio 主口径**（读 `visualize_data.bin` 的 `GM/L2 → GM Read+Write`） |
| `ratio_from_opprof.py` | CSV 辅助复算（仅交叉参考） |
| `opt_log.md` | 调优日志（第 5/6/7 轮，含候选对比表与报错原文） |
| `perf_records.jsonl` | append-only 性能记录（50 行；`round`/`phase` 区分直跑与官方口径） |
| `bench.py`、`workload_inventory.json`、`run_experiments.py`、`round1_serial.py`、`round3_intquad.py` | 本算子既有的采集脚手架与早期轮次（**未改动**，仓库中已存在同内容，本次不重复提交） |

## 四、本次**未提交**的内容（按需求排除）

- `note_about/` 下的记录 md（本轮 5 份 Cos 记录与其它算子的记录）；
- skills（`tilelang-op-opt-attempts` 等）；
- 其它算子的优化工作（L1/L2/InfNorm 等）；
- 原始的 `profiles/`（2.1 GB msprof 数据）与 `logs/`（1.3 MB 过程日志）、构建产物（`main.so`、`__pycache__/`）；
- 未改动的 Stage 3 工件：`cos_kernel/cos.py`、`cos_kernel/cos_fwd_kernel.py`、`DESIGN.md`、`cos_fwd_kernel_DESIGN.md`、`REVIEW.md`、`RETROSPECTIVE.md`、`integration_log.md`、`integration_report.json`、`.migration_meta.json`。

## 五、如何复现

```bash
# 需要：Ascend910B2C + CANN 9.0.0 + TileLang 0.1.2+npuir，且已 source 环境脚本
cd <repo>/examples/TileOPs

# A. 官方口径（7 条 workload，含精度断言）
python3 -m pytest benchmarks/ops/bench_cos.py -q -p no:cacheprovider

# B. 单工作负载直采（调优迭代口径；--check 必须一起加，否则会把错误路径测成"变快"）
msprof op --kernel-name=main --output=profiles/verify \
  --launch-count=15 --warm-up=5 --dump=off --aic-metrics=Roofline \
  python tileops/kernels/elementwise/cos/cos_kernel/perf_opt/bench.py \
  --impl cos_fwd_kernel.py --case elementwise-16m-float16 --check

# C. 读 ratio（唯一口径）
PYTHONPATH=$PWD python3 \
  tileops/kernels/elementwise/cos/cos_kernel/perf_opt/ratio_from_bin.py \
  tileops/kernels/elementwise/cos/cos_kernel/perf_opt/profiles/verify/OPPROF_*
```

## 六、已知遗留

- `16m fp32` 官方 53.53%（直跑 90.5 µs / 63.9%，官方 108.0 µs / 53.5%）：同一 kernel 换调用路径后
  `fops` 相同但周期多 21%（151.6k → 184.0k），属访存布局敏感的停顿，尚未闭环；
- 16m/64m/256m 的 fp16 / bf16 六条已稳定 ≥62.7%。
