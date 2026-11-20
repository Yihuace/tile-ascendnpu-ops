# RETROSPECTIVE -- LerpTensorFwdOp 迁移

## 本轮做了什么

1. 确认 `LerpTensorFwdOp` 六区在 `integration/npu-ops` 基线已到位，本轮定位为
   「实跑验收 + 补齐证据」，不新增算子代码。
2. 复核分派链：`lerp_tensor.py:62` 默认使用 `perf_opt` 调优版 kernel，确认本轮测的就是
   调优版而不是基线版（避免「测了 A 却在 perf_records 里写 B」）。
3. 实跑 full 报告：精度 8/8、benchmark 9/9、7 个 msprof 用例全部有效，`status=passed`。
4. 按 P0 裁决规则算出裁决档 elementwise-256m，判出「未收敛（差 5.88pp）」并如实记录。
5. 补 `DESIGN.md`（顶层副本）、`REVIEW.md`、`integration_log.md`、`RETROSPECTIVE.md`、
   `perf_opt/perf_records.jsonl`。

## 做对了什么

- **先确认分派再取数**：先读 `lerp_tensor.py` 的 kernel source selection，确认性能数字
  对应的是 `perf_opt` 版本，`perf_records.jsonl` 的 `artifact_sha256` 写的就是该文件。
- **裁决规则照章执行**：没有只挑好看的 80.47% 报，而是按 P0 规则落到 256m 档，
  并如实标「未收敛」。
- **不虚报提升**：历史 baseline 与本机工具链不同，坚决不给提升倍数，只交付可复现的终态数据。
- **采集前查占用**：沿用上一任务教训，`npu-smi info` 确认空卡后才开始，避免污染。

## 做得不够

- 没有在本机重测 torch.lerp 基线，因此「优化了多少」这个问题本轮仍未回答。
- 未重做 DESIGN.md 的 9 维度全量设计检视；`REVIEW.md` 只覆盖本轮真正做过的 4 个维度。
- 256m 档 5.88pp 的未收敛点只是记录，没有深入分析根因（取数模式 / 发射开销 / tile 选择）。

## 后续建议

1. 下一轮如果要做优化，直接盯 elementwise-256m（裁决档），把它从 67% 拉到 73% 以上
   才算真正收敛；在小 shape 上继续调收益有限。
2. 若需要提升倍数结论，先在同一套环境（CANN 9.0.0 / 910B3）重测 `torch.lerp` 基线，
   再与本次 `perf_records.jsonl` 的 final 数据比较。
3. 建议把「确认 manifest 指向的 kernel 版本」写进提交前检查：`perf_opt` 与基线版并存时
   很容易测错对象。
