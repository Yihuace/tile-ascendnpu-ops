# RETROSPECTIVE -- RopeLlama31Op 迁移

## 本轮做了什么

1. 沿用 RopeLongRopeOp 已确定的策略：族骨架只在未合并的 PR #55 里，故本分支同样
   stacked 在 `op/position_encoding/rope-neox-lvpanhao` 之上（与 PR #61 为兄弟分支）。
2. 从旧框架迁入 `kernels/position_encoding/rope_llama31/` 与
   `ops/position_encoding/rope_llama31.py`，清理 `__pycache__` / `.migration_meta.json` /
   `history_version/` / `perf_opt/logs/` / `.stage_state.json` / `.task_timeline.jsonl` /
   `integration_report.json`。
3. 从旧仓 `test_rope.py`、`bench_rope.py` 拆出 Llama 3.1 部分，落成
   `tests/ops/test_rope_llama31.py`、`benchmarks/ops/bench_rope_llama31.py`（含 `torch-ref` 基线）。
4. 补齐族注册：manifest 条目（含 4 个分段缩放参数）、`RopeLlama31OpWorkload`、两个 `__init__.py`。
5. 实跑：精度 11/11（113.73s）、完整报告 `status=passed`（3 个 msprof 用例全部有效）。
6. 写齐证据：`DESIGN.md`、`REVIEW.md`、`integration_log.md`、`RETROSPECTIVE.md`、
  `perf_opt/perf_records.jsonl`。

## 做对了什么

- **复用已验证的流程**：LongRoPE 那一轮已把族骨架、stacked 策略、取证清单跑通，这一轮直接套用，
  没有再踩「从 integration 拉分支导致 diff 混两个算子」的坑。
- **参考实现独立**：测试用独立写的 `_compute_llama31_freqs`（高频保持 / 低频缩放 / 中间混合
  三段都覆盖），不复用 Op 内部实现。
- **取数先确认生效版本**：核对 `rope_llama31.py:93-94` 实际导入 `perf_opt` 版本，
  `perf_records.jsonl` 的 `artifact_sha256` 与该文件一致。
- **口径写清楚**：`torch-ref` 是纯 PyTorch 参考实现，6.72×–90.88× 明确标注口径，
  没有当成对业界实现的加速。
- **裁决照章**：P0 落在 2d 大 shape 档 54.78%，与次大档差 16.61pp 判「未收敛」。

## 做得不够

- 同样是迁移+取证定位，没有跑额外调优轮次，kernel 直接用旧框架的 `perf_opt` 版本。
- 未重做 DESIGN 的 9 维度全量检视；`REVIEW.md` 只覆盖实际做过的 5 个维度。
- 两个 Rope 变体的族文件改动高度同构（manifest/workloads/`__init__` 各加一段），
  目前靠人工保证一致；长期看这类「同族多算子」更适合脚本化模板生成。

## 后续建议

1. **合并顺序**：#55 →（#61 与本 PR 串行），后合并者需要解族文件冲突。建议团队把
   「同族算子合入顺序」写进 `TEAM_START_HERE.md`。
2. 若要提 Ratio，直接盯 P0 档（2d `[1,8192,32,128]`，54.78%）与 1d-8k 档（38.17%）。
3. 三个 RoPE 变体共用同一套 neox kernel，后续可以考虑在族层面抽一个共享的
   「rotary 变体适配层」，避免每次新增变体都重复 manifest/workload 样板。
