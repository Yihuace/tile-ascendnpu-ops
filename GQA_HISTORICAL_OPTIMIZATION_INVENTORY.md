# GQA 三算子 + 历史优化算子总览与同步清单

> 版本：v1（2026-11-08）
> 范围：`examples/TileOPs/task_operators.md` 登记的 **24 个算子** + 本地仍有未推送优化产物的 **13 个历史算子**
> 数据来源：`task_operators.md`（唯一登记表）、各算子 `perf_opt/opt_log.md` / `perf_records.jsonl`、`reports/tileops/<run_id>/run.json`
> 说明：本文件只做汇总与同步盘点，不修改任何算子工件；所有数字可在 §6 证据索引中逐条对账。

---

## 0. 执行摘要

- **登记算子 24 个**：**9 个达标（≥60%）**，**15 个未达标**。未达标中：
  - **A 类（有明确下一步）3 个**：Argmax、EngramDecode、EngramGateConvBwd；
  - **B 类（已充分调优、被结构/编译器硬限封住）5 个**：GQA-SW、GQA-Varlen、GQA-Bwd、FP8LightningIndexer、DSA-Decode；
  - **C 类（物理封顶，60% 数学上不可能）6 个**：RopeNeox / RopeLongRope / RopeLlama31 / FFT-C2C / EngramGateConvFwd / MHA-Decode-Paged；
  - **D 类（用户确认不跟进）1 个**：Alibi。
- **GQA 三件套最新**：SW **18.52%**、Varlen **19.87%**（20261108 新报告）、Bwd **7.32%（R20 实测）/ ≈9.85%（R21 推算）**。
- **本地仍有未推送优化的历史算子 13 个**，合计约 **2.33 MB**（已排除 20GB `profiles/` + raw `logs/`）。
- **GQA Bwd R21** 的 8 档时延实测 −21%~−29%（几何 −25.22%），已转正；官方 Ratio 尚未重测，**9.85% 是推算**。

---

## 1. 三算子最新状态

| 算子 | 当前 Ratio | 数据性质 | 相对 Stage-3 | 精度 | 官方 report |
|---|---:|---|---:|---|---|
| GQA-SlidingWindowFwdOp | **18.52%** | 实测 | 1.46× | 32/32；官方 22/22 + 10/10 | `20261106_105557_262003` passed |
| GQA-SlidingWindowVarlenFwdOp | **19.87%** | 实测（20261108 新报告；旧 19.92%） | 2.06× | 51/51；官方 26/26 + 10/10 | `20261108_013426_279753` passed |
| GroupedQueryAttentionBwdOp | **7.32%**（R20 实测）/ **≈9.85%**（R21 推算） | R20 实测；R21 未重测官方 Ratio | 约 24×（R20 18.1× × R21 1.337） | L0 8/8 + L1 12/12 + L2 12/12 + Boundary 8/8 | `20261106_203252_067780` passed（R20） |

> GQA Bwd R21：`pds` BN=128 未做（估 ~15%）；`n_slots=3`（SW）未做（估 5–12% short）；Varlen 手写双缓冲被编译器封死。40% 仍不可达（详见 `GQA_40pct_Ratio_Boundary_Report.md`）。

---

## 2. 24 个登记算子总览

> 达标线 60%（口径 B）；「类别」对应 `task_operators.md` 的未达标分组；「—」表示无。

| # | 算子 | family | 当前值 | 判定 | 类别 / 说明 |
|---:|---|---|---:|---|---|
| 1 | LerpTensorFwdOp | elementwise | **74.08%** | ✅ | — |
| 2 | ClampFwdOp | elementwise | **69.11%** | ✅ | — |
| 3 | HardtanhFwdOp | elementwise | **69.02%** | ✅ | 2.4–2.8 TB/s 已贴地板 |
| 4 | ErfFwdOp | elementwise | **68.90%** | ✅ | 66.40→68.90（时延 −32.7%） |
| 5 | EluFwdOp | elementwise | **67.84%** | ✅ | — |
| 6 | LeakyReluFwdOp | elementwise | **67.20%** | ✅ | — |
| 7 | Expm1FwdOp | elementwise | **62.96%** | ✅ | — |
| 8 | BitwiseNotFwdOp | elementwise | **77.12%** | ✅ | — |
| 9 | DropoutOp | regularization | **62.00%**（B）/ 59.26%（A） | ✅ | 58.27→62.00 |
| 10 | AlibiFwdOp | elementwise | **56.29%** | ⚠️ | D：他机已优化，本机不跟进 |
| 11 | ArgmaxFwdOp | reduction | **36.71%** | ⚠️ | A：两个 bug 已修；lm-head 两档被 `syn_instr_mode` 阻塞 |
| 12 | EngramDecodeOp | sequence_modeling | **14.64%**（单 kernel）/ 9.95%（op 级） | ⚠️ | A：多核两阶段已集成 |
| 13 | EngramGateConvBwdOp | sequence_modeling | **0.038%**（op 级） | ⚠️ | A：真实 op 级；报告只测 k1 高估 18× |
| 14 | FP8LightningIndexerOp | attention_indexing | **16.64%** | ⚠️ | B：9.0→16.64（1.85×），两轮结构回退 |
| 15 | GQA-SlidingWindowFwdOp | attention | **18.52%** | ⚠️ | B：G=2 转正；R19 no_gain；R20 诊断为解析 bug |
| 16 | GQA-SlidingWindowVarlenFwdOp | attention | **19.87%** | ⚠️ | B：G=2 转正；R14–R17 封于 UB/编译器；官方 report 已补跑 |
| 17 | RopeNeoxOp | position_encoding | **27.35%** | ⛔ | C：上限 39.8%；3.52 µs 固定地板 |
| 18 | RopeLongRopeOp | position_encoding | **36.34%** | ⛔ | C：上限 49.3%；同固定地板 |
| 19 | RopeLlama31Op | position_encoding | **35.91%** | ⛔ | C：上限 49.3%；同固定地板 |
| 20 | EngramGateConvFwdOp | sequence_modeling | **5.57%** | ⛔ | C：上限 15.6%；微秒级 shape 被发射地板锁死 |
| 21 | GroupedQueryAttentionBwdOp | attention | **7.32%**（R20）/ **≈9.85%**（R21 推算） | ⚠️ | B：R20 3-pass 1.73×；R21 BN=128 −25.22%；已超 8.2% 参照 |
| 22 | DeepSeekSparseAttentionDecodeWithKVCacheFwdOp | attention | **0.35%** | ⛔ | C：上限 ~7.5%；逐行 gather 94.5% |
| 23 | MultiHeadAttentionDecodePagedWithKVCacheFwdOp | attention | **2.39%** | ⛔ | C：上限 ~26%；3 档 ideal <0.3 µs |
| 24 | FFTC2COp | spectral | **0.04%** | ⛔ | C：上限 21.5%；n≥2048 需 464KB > 192KB UB |

**统计**：✅ 9 / ⚠️ 9 / ⛔ 6 = 24；未达标 15（A 3 + B 5 + C 6 + D 1）。

---

## 3. 本地仍有未推送优化产物的历史算子（13 个）

> 口径：`perf_opt/` 下 `git status` 变更（**排除 `profiles/` 与 `logs/`**）；`files` 是变更文件数，`MB` 是这些文件的总大小。

| # | 算子 | phase | records（baseline / candidate / final） | files | MB | 注册表当前值 |
|---:|---|---|---|---:|---:|---:|
| 1 | DropoutOp | DONE | 3 / 0 / 3 | 45 | 0.78 | 62.00% ✅ |
| 2 | EngramDecodeOp | DONE | 3 / 29 / 3 | 35 | 0.76 | 14.64% ⚠️ |
| 3 | FP8LightningIndexerOp | DONE | 2 / 2 / 2 | 19 | 0.12 | 16.64% ⚠️ |
| 4 | RopeNeoxOp | DONE | 4 / 4 / 4 | 17 | 0.18 | 27.35% ⛔ |
| 5 | ErfFwdOp | DONE | 7 / 10 / 7 | 11 | 0.10 | 68.90% ✅ |
| 6 | DeepSeekSparseAttentionDecodeOp | DONE | 3 / 0 / 3 | 9 | 0.11 | 0.35% ⛔ |
| 7 | RopeLlama31Op | DONE | 2 / 0 / 2 | 8 | 0.06 | 35.91% ⛔ |
| 8 | RopeLongRopeOp | DONE | 2 / 0 / 2 | 7 | 0.05 | 36.34% ⛔ |
| 9 | FFTC2COp | — | 2 / 1 / 2 | 2 | 0.02 | 0.04% ⛔ |
| 10 | AllOp（reduction） | — | 6 / 18 / 3 | 1 | 0.05 | 不在 24 登记表 |
| 11 | AnyOp（reduction） | — | 6 / 26 / 3 | 1 | 0.06 | 不在 24 登记表 |
| 12 | CountNonZeroOp（reduction） | — | 5 / 17 / 4 | 1 | 0.04 | 不在 24 登记表 |
| 13 | MultiHeadAttentionDecodePagedWithKVCacheFwdOp | DONE | 4 / 12 / 4 | 1（整个 perf_opt 未跟踪） | ≈0 | 2.39% ⛔ |
| | **合计** | | | | **≈2.33 MB** | |

> 说明：`AllOp` / `AnyOp` / `CountNonZeroOp` 有 `perf_records.jsonl`，但不在 24 个登记表中，可能是历史 reduction 族调优；本清单只做「有本地产物」盘点，不改变登记表结论。

---

## 4. 同步建议（白名单）

### 4.1 建议纳入

- `examples/TileOPs/task_operators.md`（本次已更新的登记表；当前未跟踪）；
- 本文件 `GQA_HISTORICAL_OPTIMIZATION_INVENTORY.md`；
- 上述 13 个算子的：`perf_opt/{kernel}.py`、`opt_log.md`、`perf_records.jsonl`、`perf_feedback.md`、`workload_inventory.json`、`history/`、候选/探针脚本（**均排除 `profiles/` 与 `logs/`**）。

### 4.2 明确排除

- **20GB `perf_opt/profiles/` + raw `logs/`**（msprof 原始数据，不适合进 git）；
- 其他算子、`.agents/`、`.opencode/`、`3rdparty` 的既有脏改动；
- `.stage_state.json` / `.task_timeline.jsonl`（已在 `.gitignore`）。

### 4.3 建议 commit message

```
perf(tileops): freeze historical Stage-4 artifacts + update operator registry
```

---

## 5. 风险与未验证项

1. **Bwd R21 的 Ratio 是推算**（9.85%），官方 Ratio 未重测；正式对外口径应标注“推算/未重测”。
2. **Bwd R21 6/8 档为单 run**（与 R20 final 同口径）；仅 8b-short/8b-long fp16 做了 3 次重复（极差 0.11%）。
3. **GQA Varlen 旧值 19.92% 与新报告 19.87% 差 0.05 pp**，属不同次测量；登记表已改用新报告。
4. **历史算子的本地产物可能包含他人的中间实验**；本次只按“有变更”盘点，是否全部归入“我们优化的”需人工确认。
5. **`task_operators.md` 的 24 个值来自各自官方 report**；本次只更新 GQA 三件套，其余 21 个未重测。

---

## 6. 证据索引

| 内容 | 路径 |
|---|---|
| 24 算子登记表 | `examples/TileOPs/task_operators.md` |
| GQA 三算子评审材料 | `GQA_THREE_OPS_OPTIMIZATION_REVIEW.md`（§11 执行更新） |
| 40% 边界说明 | `GQA_40pct_Ratio_Boundary_Report.md` |
| GQA Bwd R21 | `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/perf_opt/opt_log.md` §21 |
| GQA SW | `.../grouped_query_attention_sliding_window/.../perf_opt/opt_log.md` |
| GQA Varlen | `.../grouped_query_attention_sliding_window_varlen/.../perf_opt/opt_log.md` |
| 历史算子 opt_log | 各 `examples/TileOPs/tileops/kernels/**/perf_opt/opt_log.md` |
| 官方报告 | `examples/TileOPs/reports/tileops/<run_id>/run.json`（run_id 见 `task_operators.md`） |
