# FusedAddLayerNormFwdOp — Stage 4 调优日志（二次深挖）

> CANN 9.0.0 / tilelang 0.1.2 dev root / Ascend910B3 / 2026-10-18。
> 口径：msprof op Task Duration(us) + GM Read+Write SOL ratio。
> 分类：prefill AI=3.0~3.8 > 平衡点 2.6 ⇒ **算力受限，线 60%**；13b fp32 AI=1.02 ⇒ 访存受限，线 70%。
> decode（4.0–4.8µs）属微型 shape，性能豁免。

## 0. 对象与入口
- `FusedAddLayerNormFwdOp`；kernel `_fused_add_layer_norm_kernel(M,N,eps,dtype)`（factory → `_func(block_m)`）。
- 生产基线 = Stage-3 kernel `_fused_add_layer_norm_kernel.py`（wrapper 现接此文件）。
- 产物 `perf_opt/FusedAddLayerNormFwdOp.py`；benchmark 7 tune + 3 smoke。

## 1. 上一轮（round 2–4）结论回顾
- round 2：`block_m=1` 慢 25%（N=4096），默认 `_default_block_m`（4096→2、8192/5120→1）正确。
- round 3：weight/bias 循环不变量外提（A10）——70b prefill −14.0~−14.5%、8b prefill −6.4~−7.1%。
- 根因：`latency bound:pipeline caused`（vec 60% 忙、scalar wait 86%），15+ 条向量 pass + 2 次 row-reduce。

## 2. 本轮（round 5）深挖

### 2.1 诊断（msprof PipeUtilization，8b-prefill-fp16 baseline）
- aiv_vec 45.28µs（0.605）、aiv_mte2 22.16µs（0.296）、aiv_mte3 9.40µs（0.126）、aiv_scalar 9.04µs（0.121）、aiv_scalar_wait 64.5µs。
- **现象**：MTE2/VEC 交替串行（单 `stage` 缓冲被 x→residual→round→round 顺序复用，强制
  `MTE2(x)→VEC(vcast)→MTE2(residual)→VEC`），匹配 pattern-library PL-1.10（ada_layer_norm
  同族，loads-first −31.7%）。

### 2.2 候选 vs current best 对比表（round 5，8b-prefill fp16/bf16）

| candidate | workload | Task Duration(us) | SOL% | L0 |
|---|---:|---:|---|
| hoist-wb (current best) | 8b-prefill-fp16 | 76.68 / 76.97 | 26.9 / 26.8 | PASS |
| **loadsfirst** | 8b-prefill-fp16 | **62.52 / 63.14** | 33.3 / 32.9 | PASS |
| hoist-wb | 8b-prefill-bf16 | 78.28 / 78.50 | 31.4 / 31.2 | PASS |
| **loadsfirst** | 8b-prefill-bf16 | **64.49 / 64.52** | 38.0 / 37.3 | PASS |
| singlepass | 8b-prefill-fp16 | 76.78 / 76.83 | 27.1 / 27.0 | PASS（tie，弃） |
| singlepass | 70b-prefill-fp16 | 140.34 / 140.50 | 29.3 / 29.5 | PASS（tie，弃） |

- **loadsfirst −18%**（8b prefill）；**singlepass tie**（无收益，弃，避免数值风险换零收益）。

### 2.3 实验分支结论

| opt_id | 目标现象 | 优化点 | 结果 |
|---|---|---|---|
| loadsfirst | MTE2/VEC 零重叠（单 staging 复用链） | 双输入前置装载：x/residual 各走独立 fp32 缓冲 copy-cast（fp16→fp32 上转精确），`stage` 只留 round-trip + 输出 | **8b prefill −18%**（winner） |
| singlepass | reduce→center→square→reduce 串行链 | `E[x²]−E[x]²` 单遍方差 | **tie**（config_no_gain，弃） |
| block_m=1 分核 | （round 2 已证死路） | — | 不重做 |

- **loadsfirst 的 block_m=1 限制**：70b prefill（N=8192, block_m=1）上双输入前置装载会
  **ub overflow（需 1573120 bits / 有 1572864，差 256 bits）**——本工具链 MTE2 不能跨 dtype DMA，
  copy-cast 内部仍物化 staging。故按 `block_m` 编译期分支：`block_m==1` 保留原单 staging vcast 链，
  `block_m>=2`（8b prefill）走 loads-first。
- **精度**：copy-cast 上转（fp16→fp32）精确，L0 全绿（fp16 max_diff 9.766e-04、bf16 3.125e-02、
  fp32 1.907e-06，与原版一致）。`--level all` L0/L1 全过；Boundary-extreme-affine 为
  weight/bias=±65504 的 fp16 溢出 WARN（非阻塞，与 hoist-wb 同，非本轮引入）。

## Final Performance Test Data

官方最终测量：`perf_opt/FusedAddLayerNormFwdOp.py`（round 6，baseline=Stage-3 生产 / final=loadsfirst 相邻交错）。
注：round 6 记录中 baseline 行的 `candidate_id` 沿用实验 JSON 的 `hoist-wb`，但 `artifact_path` 为
`../_fused_add_layer_norm_kernel.py`（Stage-3 生产基线，非 perf_opt 的 hoist-wb 产物）。

> 簿记修正（gate 4 S4-PERF-RECORDS-RECON 修复）：round 4 曾把 `hoist-wb` 记为 `phase=final`，但该版本在
> round 6 被 `loadsfirst` 取代并合入最终产物；这 7 条 `hoist-wb` final 行已**降级为 `merged`**（round /
> candidate_id / duration_us / msprof_raw_path 等字段均未改动，仅 phase 标签 final→merged）。最终产物
> `perf_opt/FusedAddLayerNormFwdOp.py` 的 sha256 = `d86b7d1da03955183cce29ca57134553329dda8dc9ec4c9528ecef9b097065f9`，
> 与 7 条 `loadsfirst` final 记录的 `artifact_sha256` 完全一致（该文件自 round 6 采集后未再改动），**无需重测**。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---:|---:|---:|---|
| _fused_add_layer_norm_kernel | llama-13b-prefill-float32 | 107.642 | 108.702 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-70b-decode-bfloat16 | 4.460 | 4.820 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-70b-prefill-bfloat16 | 168.283 | 144.733 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-70b-prefill-float16 | 164.233 | 140.873 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-8b-decode-bfloat16 | 4.010 | 4.340 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-8b-prefill-bfloat16 | 84.012 | 64.551 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-8b-prefill-float16 | 82.452 | 63.251 | loadsfirst |

### SOL 对账

| workload | base µs | final µs | Δ | base SOL% | final SOL% |
|---|---:|---:|---:|---:|---:|
| llama-13b-prefill-float32 | 107.642 | 108.702 | +0.98% | 45.08 | 45.25 |
| llama-70b-decode-bfloat16 | 4.460 | 4.820 | +8.07% | 0.81 | 0.81 |
| llama-70b-prefill-bfloat16 | 168.283 | 144.733 | −13.99% | 33.14 | 33.43 |
| llama-70b-prefill-float16 | 164.233 | 140.873 | −14.22% | 29.70 | 29.25 |
| llama-8b-decode-bfloat16 | 4.010 | 4.340 | +8.23% | 0.53 | 0.52 |
| llama-8b-prefill-bfloat16 | 84.012 | 64.551 | −23.16% | 31.35 | 37.49 |
| llama-8b-prefill-float16 | 82.452 | 63.251 | −23.29% | 27.14 | 32.64 |

**结论（如实）**：
- **0/7 达标**（SOL 仍 27–45%，算力线 60%/访存线 70%）；decode 微型豁免；
- 时延（硬底线 Task Duration）：**8b prefill −23.2~−23.3%**（hoist −7% + loads-first −18% 叠加）、
  **70b prefill −14.0~−14.2%**（仅 hoist，loads-first 因 UB 溢出未覆盖 block_m=1）、
  13b fp32 +0.98%（fp32 路径未改，噪声内）、decode +8%（微型豁免，噪声内）；
- 13b fp32 与 decode 的 +1~8% 为跨进程双态噪声（fp32 路径与 decode 路径本轮未改动），非回退。

## 精度回归
`python perf_opt/FusedAddLayerNormFwdOp.py --level all` → **All check passed!**（L0/L1 全过；L2/Boundary 为 WARN-only，extreme-affine 为 weight/bias=±65504 的 fp16 溢出，与 hoist-wb 同，非本轮引入）。

## Skill Retrospective
- **D**（实测数据）：norm 族「单 staging 复用链」→ 双输入前置装载（loads-first，PL-1.10 同族复证）
  在 FusedAddLayerNorm 上 **8b prefill −18%**；但 **block_m=1（N=8192）会 ub overflow（差 256 bits）**，
  因 MTE2 不支持跨 dtype DMA、copy-cast 内部仍物化 staging。fp16→fp32 上转 copy-cast **精确**（L0 max_diff 与原版一致），
  可用于替代 vcast 上转省掉 dtype staging。
  - 证据：`perf_records.jsonl` round 5/6（loadsfirst vs hoist-wb / stage3）；`profiles/candidate-r5-*`；
    `candidates/_faln_kernel_loadsfirst.py`。
- **R**（流程规则）：单遍方差 `E[x²]−E[x]²` 在本算子 **tie**（无收益）——本算子的瓶颈不是
  reduce 依赖链（auto-multi-buffer 已部分隐藏），而是 MTE2/VEC 装载串行；诊断必须看 PipeUtilization
  的 vec/mte2/scalar 占比，不能凭「15 条 pass + 2 reduce」直接归因到 reduce。
  - 证据：`profiles/candidate-r5-*` PipeUtilization（vec 0.605 / mte2 0.296 / scalar_wait 64.5µs）。
- **P**（模式方法）：loads-first 的 UB 溢出可用**编译期 `block_m` 分支**安全解决——`block_m==1`
  保留单 staging、`block_m>=2` 走 loads-first，同一 `@T.prim_func` 家族内按 shape 分化，不污染 factory/wrapper。
  - repro：`candidates/_faln_kernel_loadsfirst.py`（delta：单 stage 4-角色复用 → 双输入 copy-cast 前置装载 + block_m 分支）。
- 证据三件套落点：`perf_records.jsonl`（D）、`profiles/` raw msprof（D）、`candidates/_faln_kernel_loadsfirst.py`（P/R）。
