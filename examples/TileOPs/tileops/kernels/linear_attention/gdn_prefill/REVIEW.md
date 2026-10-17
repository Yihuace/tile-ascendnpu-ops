# REVIEW — prefill_chunk_local_cumsum_bthd_tl

> 检视对象：`DESIGN.md`（函数 1/6）。检视方式：逐维度对照源实现、API 文档与算术复核。
> 结论行见文末。

## 0. 迁移源核对（源码证据）

- 源实现已按行核对：`/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py:22`
  （`prefill_chunk_local_cumsum_bthd_tl`），并对照抽取产物
  `tileops/kernels/linear_attention/gdn_prefill/_gdn_prefill_kernels.py`。
- 源码事实 1：累积在 **fp32** 完成（`chunk_s` 为 `float32`，注释明确"rounding them to the input
  dtype loses precision"）→ 设计 §0.1/§6 一致。
- 源码事实 2：累积**逐 chunk 重置**（`acc_s` 在每个 `cid` 内初始化为 0）→ 设计 §1.3 一致。
- 源码事实 3：越界行填 `0`（`T.if_then_else(inside, g[...], 0.0)`）→ 设计 §0.6/§7 一致。
- 源码事实 4：`gate_in_kernel` 分支内做 `-exp(a_log)·softplus(g+dt_bias)`，阈值 20.0；
  本批 in-scope 参数全默认（`gate_in_kernel=False`），设计 §0.5 已声明不启用，接口保留。
- 源码事实 5：`head_tile = min(head, 64)`，`head > 64` 时按 slab 切分 → 设计 §5.1 的
  `num_head_slabs = ceildiv(head, 64)` 一致。

结论：设计对源实现的语义复述**无偏差**。

## 1. 语义与公式正确性

- 公式（设计 §1.3）为前缀和定义，`out[base+i] = Σ_{j≤i} g[base+j]`，与源码双重循环
  `acc += chunk_s[i]; chunk_s[i] = acc` 等价。
- 尾部：`base+i ≥ end` 时输出 0；因填 0 不改变前缀和，故"先填 0 再累加"与"按有效长度累加"
  等价——这一等价性已在设计 §0.6 明写。
- 未引入任何代数变换（§1.6.1 结论"无优化空间"），因此不存在"优化改变语义"的风险。

## 2. I/O 契约

- 输入 `g: [1, total_tokens, head]`、输出 `out: [1, total_tokens, head]` fp32，与源签名一致；
- `cu_seqlens` 在等长场景步长恒为 `S`，设计将其等价为 `base = seq·S + first`（源码注释亦如此说明）；
- 变长（`cu_seqlens` 任意取值）不在本批范围，设计 §0.1 明确 out-of-scope。

## 3. API 映射与可行性

逐条核对文档（`docs/Tilelang.language/排序操作/T.vcumsum.md`）：

| API | 文档结论 | 与设计一致？ |
|---|---|---|
| `T.cumsum(src, dst, dim, reverse)` | 支持 int32/int64/fp16/**fp32**，**不支持 bf16**；`reverse` 目前只支持 False | ✅ 设计 §3.4 据此要求 bf16 先 `vcast` |
| `T.vcast` | 用于 UB 内 dtype 转换 | ✅ |
| `T.copy` | GM↔UB 搬移 | ✅ |

- `reverse=False` 为本函数唯一用法，无阻塞；
- `dim=0`（行方向）在 shape 支持范围内（输入输出同 shape）。

## 4. 布局与向量化

- 采纳布局 L0（沿用 GM 的 `[tokens, head]`），向量化轴 = head（最内、连续，head ≥ 16 ⇒ ≥32B）；
- 弃选 L1（转置后按 C 向量化）：多一次 UB 重排且 `T.cumsum` 本就是硬件指令，收益为 0；
- 弃选 L2（1D 平铺分段）：无法表达"逐 chunk 重置"。

## 5. 内存与 UB 预算

算术复核（C=64、head_tile=64）：

```
g_ub   : 64 × 64 × 2 B  =  8 KiB   (fp16/bf16)
f32_ub : 64 × 64 × 4 B  = 16 KiB
out_ub : 64 × 64 × 4 B  = 16 KiB
------------------------------------
合计                     40 KiB   < 192 KiB (UB)   ✅
```

head=16 时更低（10 KiB 级）。即使 `head=128` 按 slab=64 切分，预算不变。

## 6. 分核与并行

- 逻辑任务数 `T = num_chunks × num_head_slabs`；逻辑核数 `K = min(T, physical_cores)`；
- 物理核数用 `NPUUtils.get().get_aicore_num()` 实查，设计未硬编码；
- `T ≤ K` 时一核一块（无需适配）；`T > K` 时按 `task % K` 轮转，每核 `ceildiv(T,K)` 个任务（整数倍）；
- 单核内为核内串行三段流水，无跨核同步 → 不存在竞态。

## 7. 数值精度与边界

- 累积 fp32：C=64、|g|≤1 时理论累积误差 < 1e-6，判据取 1e-5 有 10× 余量；
- bf16 输入：量化步长 ~2^-8，判据放宽到 1e-2 合理；
- 边界：`total_tokens % chunk != 0`、`head % 64 != 0`、`num_chunks == 0` 三种情形设计 §7 均已覆盖；
- 无除法、无 `exp/log`，不存在 NaN/Inf 传播路径（`gate_in_kernel=False` 路径）。

## 8. 调研结论复核与弃选论证核对

**调研结论复核（复算）**：§1.6.0 的四个候选重新独立复算——
- C2（Blelloch 分块扫描）：C=64 时深度 `log2(64)=6` 级，但需 2 次跨块 GM 往返 + 同步，
  以 Ascend 910B3 的 GM 延迟量级计，单次往返已超过 64 行串行累加的收益 → 弃选成立；
- C3（整段扫描）：与"每 chunk 重置"语义冲突（源码 `acc_s` 在 `cid` 内清零）→ 弃选成立；
- C1（`T.cumsum`）：硬件指令替换 64 次软件串行，收益明确 → 采纳成立。
复核后与设计结论一致，无需修订。

**弃选论证核对**：逐条对照 API 文档与源码——
- C2 的"跨块同步代价"论断：核对 `docs/Tilelang.language/` 中 `T.cumsum` 无跨块前缀接口，
  故 C2 需自行实现 GM 往返 → 论断成立；
- C3 的"语义冲突"论断：核对源码 `acc_s` 的作用域在 `cid` 循环内 → 成立；
- L1 布局的"重排有开销"论断：核对 `T.transpose` 为核内 UB 级操作（非 GM 级重排），
  但本场景无收益，故"多一趟 UB 往返即为净损失"成立。

**独立核对结论**：设计中的负向论断（弃选理由）均有源码或文档依据，未发现"凭先验否定"的情况。

---

结论: 通过
