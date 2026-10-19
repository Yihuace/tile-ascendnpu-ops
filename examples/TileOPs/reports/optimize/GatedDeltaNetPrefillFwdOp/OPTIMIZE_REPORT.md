# 优化报告：GatedDeltaNetPrefillFwdOp（Stage-4 optimize）

> 本文件为 GDN 首次 Stage-4 报告，结构按 `reports/optimize/REPORT_TEMPLATE.md` 的 8 节模板。
> **口径说明**：GDN 是多 kernel op，msprof 整 op 单次 capture 不可用；本轮采用
> **per-kernel `msprof op` 汇总**（每个唯一 kernel 名单独测 Task Duration，再按一次 forward 求和）。
> §3 的 SOL 是 op-level roofline 计算结果；base 的迁移态 msprof 仍无数据，写 `—`。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `GatedDeltaNetPrefillFwdOp`；多 kernel（K1 cumsum + K2 split blocksolve + K6 split forward）；Vector + Cube；gated delta-rule prefill |
| kernel（基准，只读） | `tileops/kernels/linear_attention/gdn_prefill/gdn_prefill.py` 中的 reference fallback（逐 token delta-rule） |
| 调优产物 | `tileops/kernels/linear_attention/gdn_prefill/gdn_split_{inverse,blocksolve,k6,k6_vec,k6_batched,pipeline}.py` |
| 口径 | **per-kernel `msprof op` Task Duration 求和**；SOL 为 op-level roofline（`GatedDeltaNetPrefillFwdOp.eval_roofline()`）；base 迁移态 msprof 无数据 |
| **工具链版本戳** | tilelang `0.1.2` dev root（`/home/developer/tilelang-mlir-ascend`）/ **CANN 9.0.0 系统 `bishengir-compile` 1.1.0 + `hivmc 0.2.0`** / Ascend910B3 / 2026-10-19 |
| 达标线 | **未定义 SOL 达标线**；本轮目标是「相对 reference fallback 的墙钟加速」与正确性回归 |
| benchmark | `examples/gdn_prefill/prefill_forward/bench_gdn_perkernel.py`（per-kernel msprof 汇总）；`bench_gdn_split.py` 墙钟对照；`benchmarks/ops/bench_gdn.py` 仍 skip |
| 过程日志 | `examples/gdn_prefill/correct_initial_states/PROBE_20261019.md`；`/tmp/opencode/bench_gdn_*.log`（临时日志） |
| 前置报告 | 无（首次 Stage-4 报告） |

## 1. 结论

- 一句话：GDN in-scope 主路径已从 reference fallback 换成 **K1 + K2 split + K6 batched** 的 TileLang 实现；
  `S=4096,H=16,K=128` 的 per-kernel msprof 总 Task Duration **8264.4 µs**，
  op-level SOL **0.357%**；`test_gdn.py` **10/10**。墙钟对照为 **4382.7 ms → 997.8 ms（4.4×）**。
- **达标数**：SOL 达标线未定义，故不给 x/y；in-scope 正确性覆盖 `B=1、S≥64、H=HV、K=V∈{64,128}`，
  小 shape（K<16）与 CP partition 仍走 fallback。
- 关键收益：
  - `S=64,H=16,K=128`：msprof 总时长 **155.5 µs**，SOL **0.676%**
  - `S=4096,H=16,K=128`：msprof 总时长 **8264.4 µs**，SOL **0.357%**
  - 墙钟对照：`S=4096,H=16` 4382.7 ms → 997.8 ms（**4.4×**）
- **是否建议采纳 wrapper**：✅ **in-scope 采纳**。`test_gdn` 全过；msprof/SOL 口径已可用。
  小 shape/CP fallback 保留。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| fallback 形态 | 逐 token 的 delta-rule 循环，每 token 多次 torch 算子 | 语义正确但性能不是目标；作为同语义基线 |
| split 初始 launch 数 | 每 chunk、每 head 约 11 个 kernel；`H=16` 时约 11×chunk×16 次 | 初始 split 版本受 host launch 支配，`S=1024,H=1` 已 107 ms |
| `S=4096,H=16` 分阶段 | K1 0.5 ms / **K2 599.6 ms** / **K6 399.4 ms** | K2、K6 是双瓶颈；K2 略大 |
| K2 结构 | 6 级 Neumann，每级 1 个 Cube + 1 个 Vector kernel，T/P 走 GM | 12 个 kernel launch + GM 往返；fused 大 kernel 会触发 CG-2026-0003 |
| K6 结构 | 每 chunk 6 个 Cube GEMM + 5 个 Vector kernel | head 维度未批量化时 launch 数随 H 线性增长 |
| 工具链限制 | CANN 9.0 系统编译器可编译小 kernel；fused `mmadL1` 仍可能 SIGSEGV（CG-2026-0003） | 不能用原始 fused blocksolve；必须保持拆分 |

### 2.2 方法（做了什么、为什么这样做）

1. **K2 blocksolve 拆分**
   - `gram_kernel`（Cube）算 `G = k kᵀ`；
   - `build_n_kernel`（Vector）算 `N = -β·exp(gprev-gc)·G·mask`；
   - 6 级 Neumann：`seg_cube`（`tmp=P T`、`P2=P P`）+ `seg_add`（`T=T+tmp`）。
   - 为什么有效：把 fused MixCV 大 kernel 拆成 compiler-friendly 小 kernel，绕开 `mmadL1 getResult(0)` SIGSEGV；
     Cube/Vector 之间只通过 GM 交接。
   - 合法性：K2 与 fp64 `torch.linalg.solve` 对拍，`S=64/100/128,K=64`、`S=1024,H=4,K=128` 通过。
2. **K2 的 `N` 使用 exclusive `gprev`**
   - `gprev[0]=0`，`gprev[i]=gc[i-1]`；
   - 为什么：K6 的标准块公式要求 `S_sys/rhs` 用 exclusive gate 前缀；inclusive 会导致 K2 的 A 与 K6 语义不一致。
3. **K6 Vector 化**
   - `vec_rhs/vec_p/vec_vp/vec_o/vec_state` 替换 host elementwise；
   - 为什么有效：避免每 chunk 的 Python/PyTorch 逐元素开销，同时保证 Cube 输出直接进入 Vector。
4. **K6 head 维度批量化**
   - 每个 Cube/Vector 阶段一个 kernel 处理全部 head（`T.Kernel(H)`）；
   - 为什么有效：把 `H` 倍的 launch 折叠成一次；`H=16` 时是主要收益来源。
5. **尾 chunk**
   - q/k/v/beta 补零，`gc` 尾部复制最后一个有效值，A 使用 K2 padded 输出；
   - 为什么：保持 `S=3000` 这类非整 chunk 的语义；尾块输出只写有效行。
6. **正确性/稳定性修复**
   - `T.gemm(size=[M,K,N])` 修正（之前误写 `[M,N,K]`，K=N 时未暴露）；
   - 避免 `a_transpose` GEMM，先 `k.T.contiguous()` 再普通 GEMM；
   - `vec_state` 缩小 UB（host 算 `S_scale`，kernel 只做加法）；
   - `torch.linalg.solve` 后同步，A 送 GEMM 前 `.contiguous()`。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | reference fallback | 逐 token delta-rule | `S=4096,H=16` 4382.7 ms | 作为基线保留 |
| R1 | split K2 + per-head K6 | Gram/N/6 级 + 5 个 Vector | `S=64,K=64` 对拍 `5.2e-5/7.0e-5`；`S=1024,H=1` 107 ms | ✅ 正确性采纳 |
| R2 | batched K6 | head 维度进 `T.Kernel(H)` | `S=4096,H=16` 997.8 ms；`test_gdn` 10/10 | ✅ 采纳 |
| R3 | K2 fp16 迭代 | Neumann T/P 改 fp16 | 设备端崩溃（疑似 fp16 GEMM/UB 组合） | ❌ 否决，回退 fp32 |
| R4 | `a_transpose` state GEMM | 直接 `T.gemm(..., a_transpose=True)` | K=128 出现 NaN | ❌ 否决，改 `k.T.contiguous()` + 普通 GEMM |
| R5 | 性能定位 | 分阶段计时 | K2 599.6 ms / K6 399.4 ms | 下一步优化 K2 |

## 3. 优化结果（per-kernel msprof 汇总）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---:|---:|---:|---:|---:|---|
| `S=64,H=16,K=128,fp16` | —（无迁移态 msprof） | 155.5 | — | — | **0.676** | 不适用（无 SOL 达标线） |
| `S=4096,H=16,K=128,fp16` | —（无迁移态 msprof） | 8264.4 | — | — | **0.357** | 不适用（无 SOL 达标线） |

`S=4096,H=16,K=128` 逐 kernel 汇总：

| kernel | 次数 | Task Duration 合计（µs） |
|---|---:|---:|
| k1_cumsum_main | 1 | 4.82 |
| gram_main | 1 | 52.74 |
| build_n_main | 1 | 84.98 |
| seg_cube_main | 6 | 473.31 |
| seg_add_main | 6 | 271.97 |
| bmm_main | 384 | 1324.81 |
| vec_rhs_main | 64 | 226.66 |
| **vec_p_main** | 64 | **5143.86** |
| vec_vp_main | 64 | 184.98 |
| vec_o_main | 64 | 216.22 |
| vec_state_main | 64 | 280.05 |
| **合计** | — | **8264.40** |

> 口径：每个唯一 kernel 名用 `msprof op --kernel-name=<name>` 单独测，`launch-count=500`、
> `warm-up=0`，汇总该 name 在一次 forward 内所有实例的 `Task Duration(us)`。
> SOL 由 `GatedDeltaNetPrefillFwdOp.eval_roofline()` 的 flops/bytes 与 910B3 roofline 计算：
> `ratio = perf_tops / min(AI*1.75, 294.9)`；这是 **op-level 理论 roofline**，不是 msprof 单 kernel 的 Ratio。
> 墙钟对照（含 host launch）见 `bench_gdn_split.py`：`S=4096` 997.8 ms vs fallback 4382.7 ms（4.4×）。

## 4. 精度

- `python -m pytest tests/ops/test_gdn.py -q` → **10 passed**（空 cache 复验）。
- 覆盖：
  - `L1b-in-scope`：`B=1,S=64,H=HV=1,K=V=64`；
  - `L1c-tail`：`B=1,S=100,H=HV=1,K=V=64`；
  - 小 shape `S=8/K=8`、`S=64/K=16` 仍走 fallback。
- 关键误差：
  - `S=64,K=64`：`o_err=5.244e-5`，`S_err=7.036e-5`
  - `S=100,K=64` 尾块：`o_err≈7.9e-5`，`S_err≈1.8e-4`
  - `S=3000,K=128` 尾块：`o_err=6.787e-7`，`S_err=2.645e-7`
- 边界：尾 chunk padding 行不写入返回视图；测试检查无 NaN/Inf；未单独覆盖 ±inf/subnormal。

## 5. 时间与成本

- 日期：2026-10-19。
- 墙钟、token、cost：**本轮未系统记录**；只有上面 benchmark 的单次耗时。
- 主要调试成本：K2/K6 的编译器限制、K=128 GEMM 形状、尾 chunk、batched 集成。

## 6. 产物

- final kernel：
  - `tileops/kernels/linear_attention/gdn_prefill/gdn_split_inverse.py`
  - `gdn_split_blocksolve.py`
  - `gdn_split_k6.py`
  - `gdn_split_k6_vec.py`
  - `gdn_split_k6_batched.py`
  - `gdn_split_pipeline.py`
- wrapper：`gdn_prefill.py` 在 in-scope 形状分派到 `gdn_prefill_split`；其余 fallback。
- 测试：`tests/ops/test_gdn.py` 10/10。
- benchmark：`examples/gdn_prefill/prefill_forward/bench_gdn_split.py`（墙钟；`benchmarks/ops/bench_gdn.py` 仍 skip）。
- 记录：`examples/gdn_prefill/correct_initial_states/PROBE_20261019.md`；commit `8225976` / `74c7d76` / `9648d4e`。
- `statectl gate 4`：未运行（本轮是自定义性能验证，不是标准 Stage-4 gate 流程）。
- 端到端 report：无（未跑 msprof/`tileops-report`）。
- pattern-library 回写：无。

## 7. 天花板 / 豁免 / 缺口

- **`vec_p_main` 是最大 msprof 瓶颈**：4k 时 **5143.9 µs / 8264.4 µs（62%）**；其 `T.if_then_else`
  被编译成低效 scalar `vcmp`，是下一步首要优化点。
- **K2 仍是次瓶颈**：blocksolve 合计约 883 µs；6 级 Neumann 的 12 个 kernel + GM 往返。
- **K6 Cube/其他 Vector**：`bmm_main` 1324.8 µs；其余 Vector 约 1.1 ms。
- **没有 SOL/msprof**：无法给出 ratio，也无法判断距离 roofline 多远；Excel GDN ratio 保持 `—`。
- **K5/CP 未实现**：in-scope 通过关闭 partition 保持正确；长序列并行度未利用。
- **小 shape fallback**：`K<16` 仍走 reference；不影响 in-scope。
- **fp16 Neumann 失败**：已回退 fp32；这是精度/编译限制，不是正确性结论。
- **贴线风险**：无 SOL 数据，无法判定贴线。

## 8. 建议

- **采纳**：in-scope split 主路径接入 wrapper；小 shape/CP 保留 fallback。
- **下一步优先级**：
  1. 优化 `vec_p_main`（替换 scalar `vcmp` 的低效掩码路径，预计最大收益）；
  2. 优化 K2（两级合并/减少 GM 往返）；
  3. 再优化 K6（融合 Vector 阶段或进一步批处理）；
  4. 补 K5/CP，恢复长序列并行；
  5. 性能数据稳定后再更新 `19_operators_ratio.xlsx` 的 GDN 行。
- **未解决**：GDN 仍没有 9.0 迁移态 msprof baseline；当前 `final SOL` 是优化态 op-level roofline，迁移态 `—`。
