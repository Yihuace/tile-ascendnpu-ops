# 优化报告：ClampMinFwdOp（Stage-4 optimize）

> **口径**：过程记录；交付数据取 `perf_data/canonical.json`（CANN 9.0.0）。
> **本算子复用 `ClampScalarFwdOp` 的同一 kernel（max=None 路径）**，R2 的结构改动与诊断见
> `reports/optimize/ClampScalarFwdOp/OPTIMIZE_REPORT.md` §9 与 `perf_opt/REPORT_CLAMP_R2_DATA.md`。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `ClampMinFwdOp`（`torch.clamp(x, min=...)`，max=None） |
| kernel | `kernels/elementwise/clamp_scalar/clamp_scalar_kernel/`（与 ClampScalar 共用） |
| 调优产物 | `perf_opt/ClampScalarFwdOp.py`（R2 final，sha256 `760579b0…`）；wrapper 经 `perf_opt/clamp_scalar_fwd_kernel.py` 转发 |
| 口径 | `msprof op Task Duration`；CANN 9.0.0 / Ascend910B3 / 2026-10-21 |
| 达标线 | 访存 ≥70% |
| benchmark | `benchmarks/ops/bench_clamp_min.py::test_clamp_min_bench`（7 tune + 3 smoke） |
| 端到端 | `reports/tileops/20261021_173421_838918_ClampMinFwdOp/`（passed，正确性 14/14） |

## 1. 结论

- **R2 后 7/7 达标**（ratio **72.50–78.39%**），相对 R1 的 61.77–68.84%（0/7）全部跨过 70% 线。
- kernel 侧 7 个 workload 全部变短，−11.3% ~ −34.1%（均值约 −23%）。
- 硬底线与精度：`--level all` 全过、端到端 14/14、NaN 语义不变。

## 2. 方法（与 ClampScalar 同一处改动）

1. **原生 dtype clamp**：fp16/fp32 直接 `vmax/vmin`，去掉 2 条无信息 `vcast`（bf16 保持 fp32 中转）。
2. **A4 结构**：满块静态 extent + 尾块 epilogue + `T.Pipelined(2)`。
3. 显式 `vmax→vmin`（`dst==src1`），避免 alias 税。

被否决：`block_size=16384`、`num_stages=3`（`config_no_gain`）。

## 3. 结果（kernel 级，msprof）

| workload | dtype | base µs | final µs | Δ | final SOL% |
|---|---|---:|---:|---:|---:|
| 16M-min-only | fp16 | 38.93 | 26.59 | −31.7% | 75.67 |
| 16M-min-only | bf16 | 39.48 | 26.00 | −34.1% | 73.63 |
| 16M-min-only | fp32 | 69.02 | 50.63 | −26.6% | 76.57 |
| 64M-min-only | fp16 | 165.28 | 132.42 | −19.9% | 74.55 |
| 64M-min-only | bf16 | 168.65 | 128.99 | −23.5% | 76.39 |
| 256M-min-only | fp16 | 857.31 | 760.82 | −11.3% | 72.76 |
| 256M-min-only | bf16 | 854.97 | 751.58 | −12.1% | 73.70 |

端到端：7 个 tune shape ratio **72.50–78.39%**，全部 ≥70%；正确性 14/14。

## 4. 精度

- `perf_opt/ClampScalarFwdOp.py --level all` → All check passed（含 NaN/±inf、min>max、bound 舍入边界）。
- ClampMin 端到端正确性 14/14，max_abs_err 0.00e+00。

## 5. 产物与门禁

- final：`perf_opt/ClampScalarFwdOp.py`；转发层 `perf_opt/clamp_scalar_fwd_kernel.py`。
- `perf_records.jsonl` 101 行（18 条 final，同一候选 `r2-combined`）。
- gate 4 pass；端到端 report `20261021_173421_838918` passed。

## 6. 天花板 / 建议

- 7/7 达标；无剩余杠杆（`block_size`/`num_stages` 已扫，无收益）。
- 若后续 bf16 获得原生向量算术，可再省一次中转（预计 <5%），否则按当前版本收束。
