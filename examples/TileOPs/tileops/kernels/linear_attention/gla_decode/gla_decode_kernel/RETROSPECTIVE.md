# GLADecodeOp 复盘

## 有效的改动

1. **`k_tile` 16 → 32**：把 UB 预算从经验值 150KB 修正为实测 192KB，K-tile 循环 8→4 次，减少 reduce/barrier 与 loop 开销。5 个 serving workload 相对 Stage-3 基线 Task Duration 降低 **24.6–35.2%**。
2. **persistent 分核（40 核）**：`B*H > 40` 的超发 shape 把 block 数收到物理核数，额外降低 **6.1–10.9%**；欠订 shape 保持 flat，A/B tie。
3. **编译期分派**：按 `B*H` 与整除规则选择 flat / persistent，避免运行时分支。

## 未采纳的尝试

- `k_tile=8`：5 个 workload 全部回退 +43~47%，否决。
- `persist20`：`b8-h64` 回退 +71%（核数不足），否决。
- `T.Pipelined(2)`：全部落在噪声内，否决。
- 消除 bf16→fp32 transit：bf16 无 Vector 算术，fp32 累加是精度要求，不可行。
- 每核 2 头交错 / DV 切分：受 UB 与关键路径限制，估算净收益 <5%，未执行。

## 经验

- 单 token decode 的并行度就是 `B*H`；在 20 AIC / 40 AIV 下，超发/欠订要分开处理。
- `k_tile` 是唯一有效的编译期杠杆，但必须用实测 UB 预算，不能沿用保守经验值。
- bf16 的 jitted kernel 名是 `_gla_decode_transit_main`，msprof 过滤要用 transit 变体名。

## 后续

- 若后续要提升大 batch 的 SOL，需要从算法层增加并行度（如 DV 切分或跨头流水），当前实现不做 shape 特判。
