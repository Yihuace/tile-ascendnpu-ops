# RETROSPECTIVE -- RopeLongRopeOp 迁移

## 本轮做了什么

1. 确认 `position_encoding` 族在 `integration/npu-ops` **完全不存在**（连 manifest 文件都没有），
   而族骨架只在未合并的 PR #55（RopeNeox）里 → 决定采用 stacked 分支，避免一个 PR 混两个算子。
2. 从旧框架 `/home/developer/tilelang-mlir-ascend/examples/TileOPs` 迁入 kernel 与 Op：
   `kernels/position_encoding/rope_longrope/`、`ops/position_encoding/rope_longrope.py`，
   并按团队取舍清理 `__pycache__` / `.migration_meta.json` / `history_version/` /
   `perf_opt/logs/` / `.stage_state.json` / `.task_timeline.jsonl`。
3. 从旧仓 `test_rope.py`、`bench_rope.py` 拆出 LongRoPE 部分，落成
   `tests/ops/test_rope_longrope.py`、`benchmarks/ops/bench_rope_longrope.py`（含 `torch-ref` 基线）。
4. 补齐族注册：manifest 条目、`RopeLongRopeOpWorkload`、两个 `__init__.py` 导出。
5. 实跑：精度 11/11 通过（112s），完整报告 `status=passed`（3 个 msprof 用例全部有效）。
6. 写齐证据：`DESIGN.md`（顶层副本）、`REVIEW.md`、`integration_log.md`、`RETROSPECTIVE.md`、
   `perf_opt/perf_records.jsonl`。

## 做对了什么

- **先查基线再动手**：发现族骨架不在 `integration` 后，没有直接拉分支硬塞文件，
  否则会和 #55 撞同一批文件、并让 PR diff 混入另一个算子。
- **参考实现独立**：测试的 cos/sin 用独立写的 `_compute_longrope_freqs`，不复用 Op 内部实现，
  避免「自己验证自己」。
- **取数先确认生效版本**：核对 `rope_longrope.py:90-91` 实际导入的是 `perf_opt` 版本，
  `perf_records.jsonl` 的 `artifact_sha256` 记的就是该文件，避免测 A 写 B。
- **口径写清楚**：`torch-ref` 是纯 PyTorch 参考实现，6×–97× 的倍数在文档里明确标注口径，
  没有当作「对业界实现的加速」对外宣称。
- **裁决规则照章执行**：P0 落在 2d 大 shape 档 54.74%，与次大档差 16.70pp 判「未收敛」，
  没有只挑好看的 58× 比值来说事。

## 做得不够

- 没有再跑调优轮次，kernel 直接用旧框架已调优的 `perf_opt` 版本；本轮定位是迁移+取证。
- 未重做 DESIGN 的 9 维度全量检视，`REVIEW.md` 只覆盖本轮实际做过的 5 个维度。
- 补 `Optional` 导入这种低级问题在提交前才发现，应该在写完即跑一次 import 自检。

## 后续建议

1. **先合 #55**：本 PR stacked 在它上面；#55 合入后本 PR 的 base 会自动切回
   `integration/npu-ops`，届时再走正式审核更顺。
2. 若要提 Ratio，直接盯 P0 档（2d `[1,8192,32,128]`）与 1d-8k 档，小 shape 档收益有限。
3. `RopeLlama31Op` 也在同一族里，建议同样 stacked 到 #55 之后，避免重复搬族骨架。
4. 建议给族骨架类文件（manifest / workloads / `__init__.py`）约定一个「首个算子合并后再追加」
   的顺序规则，减少后续同类 stacked 依赖。
