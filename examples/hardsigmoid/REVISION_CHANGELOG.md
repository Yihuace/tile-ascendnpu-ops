# 改动清单 — HardsigmoidFwdOp 迁移报告修订 + 工具侧收口（M1–M8）

> 交付物：`MIGRATION_REPORT.md`（rev.2）+ 本清单。命令均可在对应工作目录照抄重跑。
> 工作目录约定：`W=/mnt/workspace/npuir-migration`，`T=$W/examples/TileOPs`。

## 1. 改动文件清单

| # | 文件 | 动作 | M | 说明 |
|---|---|---|---|---|
| F1 | `$W/examples/hardsigmoid/MIGRATION_REPORT.md` | update | M1–M8 | rev.2：§0 修订留痕表 + §2.4 代际说明 + §2.5 护栏表 + §4.1–4.6 + §5 明细/登记 + §6 命令 |
| F2 | `$T/pyproject.toml` | update | M5 | `[tool.ruff.lint.per-file-ignores]` 增 2 条：集成 kernel `E402`、`history_version` 冻结备份 |
| F3 | `$T/tileops/kernels/elementwise/hardsigmoid/hardsigmoid.py` | update | M5 | 修复 I001（导入块空行），`ruff check --fix` |
| F4 | `$W/examples/hardsigmoid/hardsigmoid_fwd_kernel/hardsigmoid_fwd_kernel.py` | update | M5 | Stage-3 kernel `__all__` 排序（RUF022），语义不变 |
| F5 | `$T/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel.py` | update | M5/M8 | 重跑集成时从 F4 重新拷入（`__all__` 已排序） |
| F6 | `$T/.agents/skills/add-npu-op/scripts/integrate_kernel.py` | update | M8 | 新增 `write_integration_log()`；`main()` smoke 后调用；必写 `wrapper_rewritten` 字段 |
| F7 | `$T/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/integration_log.md` | update | M8 | 追加 `## integrate_kernel.py run` 段（`wrapper_rewritten: true`）+ `## History` 段 |
| F8 | `$W/.agents/evolution/queue.md` | update | M7 | 追加工具缺陷登记 VP-2026-0145（merged）/ VP-2026-0146（pending） |
| F9 | `/mnt/workspace/TileOPs-main/.migration_source_map.json` | add | §0.3 | 源映射（69 算子），适配层第③项（前期会话产出） |

**逻辑冻结**：未改 kernel 语义，未重跑 Stage 1–3；DESIGN v1 / REVIEW / Stage-3 kernel 逻辑均未变（仅 `__all__` 顺序）。

## 2. 命令与证据（逐条可复核）

| M | 命令 | 期望证据 |
|---|---|---|
| M1 | `cd $T && python -m pytest benchmarks/ops/bench_hardsigmoid.py --collect-only -q` | 9 nodeid；`run.json metadata.commands.benchmark` 含 `-m not smoke` |
| M1 | `grep -nA3 '^HardsigmoidFwdOp:' /mnt/workspace/TileOPs-main/src/tileops/manifest/spec/elementwise_unary_activation.yaml` vs `$T/tileops/manifest/elementwise.yaml` | 上游 2 label×2 dtype=4；NPU 侧 +smoke-1m(3)+fc-wide(2)=9 |
| M2 | `python -c "N=2048*4096; print(N, 2*N*2)"` | `8388608 33554432` |
| M3 | `grep -n 'fc-wide' $T/reports/report.md` | MishFwdOp 行 fp16 54.01 / bf16 57.67，带宽列 1.8000 |
| M4 | `grep -n 'inplace' $T/tileops/ops/elementwise/hardsigmoid.py` | `inplace: bool = False` + `forward()` 原地 copy_ |
| M4 | `grep -n 'Bandwidth(TB/s)' $T/tileops/benchmark/benchmark_base.py` | 第 467 行 `"Bandwidth(TB/s)": gm["bw"]`（msprof 解析） |
| M5 | `cd $T && python -m ruff check tileops/kernels/elementwise/hardsigmoid tests/ops/test_hardsigmoid.py benchmarks/ops/bench_hardsigmoid.py` | `All checks passed!`（原 9→0） |
| M5 | `python3 $W/.agents/tools/standards_check.py check` | 7 条 `SC-INLINE-DUP`/`SC-REF-PATH`，均在标准/skill 文件，与本次改动无关 |
| M6 | `ls -la --time-style=full-iso $W/examples/hardsigmoid/hardsigmoid_fwd_kernel/history_version/ $T/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/*DESIGN.md` | `design_v0.md`(21:48) / `design_v1.md`(21:58) / 快照(2026-10-12 21:58:52) |
| M7 | `python3 -c "import json;d=json.load(open('$T/reports/tileops/20261012_223321_664388_HardsigmoidFwdOp/run.json'));print(d['summary']['correctness_tests'], d['metadata']['git_commit'])"` | `7 None` |
| M7 | `grep -n 'VP-2026-014[0-6]' $W/.agents/evolution/queue.md` | 0142/0143/0144（蒸馏）+ 0145/0146（本次登记） |
| M8 | `cd $T && cp -r .../hardsigmoid_kernel /tmp/opencode/hardsigmoid_kernel.bak_m8_20261013_090943` | 备份清单 |
| M8 | `python3 $T/.agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/hardsigmoid/.migration_meta.json` | `[ok] integration complete`；`[log] .../integration_log.md` |
| M8 | `grep -n 'wrapper_rewritten\|wrapper_rewrite_action' .../integration_log.md` | `wrapper_rewritten: true` / `wrapper_rewrite_action: already integrated` |

## 3. 完成标准核对

1. **M1–M8 全部留痕**：报告 §0（三列：M / 改了什么 / 证据 / 命令）+ 本清单；M5 的 standards_check 逐条证无关、M7 的字段与登记、M8 的前后日志均已落盘。
2. **数字不变**（重跑后）：`pytest tests/ops/test_hardsigmoid.py -v` → `7 passed`（83.67s）；护栏三条 grep → 护栏1 `true`、护栏2 `already integrated`（复核）、护栏3 baseline 谱系；`tileops-report render run.json` 后 `status=passed / 7 / 9 / msprof 6 / fallback 0 / invalid 0 / max_abs_err 5.96e-08` 与 per-shape 表均不变。
3. **提交物**：更新后的 `MIGRATION_REPORT.md` + 本 `REVISION_CHANGELOG.md`。
4. **卡点**：适配层“单独提交”因工作区非 git 仓库无法执行（报告 §5 已写明所需授权）；其余各项均完成。

## 4. 数字稳定性说明（若有变动逐项注明）

- 正确性耗时 84.93s → 83.67s：**运行时间波动**，用例数与结果（7 passed）不变。
- 其余指标（7/9/6/0/0、5.96e-08、per-shape latency/ratio）与修订前逐字段一致，无变动。
