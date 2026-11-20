# ClampMinFwdOp 集成记录

## 方式

- 复用 `ClampScalarFwdKernel`，不新增 kernel；Op 文件为 `tileops/ops/elementwise/clamp_min.py`。
- 依赖 `ClampScalarFwdOp` 的 kernel 包（PR #3，已合入）。

## 文件

- Op：`tileops/ops/elementwise/clamp_min.py`
- Manifest：`tileops/manifest/elementwise.yaml`（`ClampMinFwdOp`）
- Workload：`tileops/workloads/elementwise.py`（`ClampMinWorkload`）
- Test / Bench：`tests/ops/test_clamp_min.py`、`benchmarks/ops/bench_clamp_min.py`
- 优化报告：`reports/optimize/ClampMinFwdOp/OPTIMIZE_REPORT.md`

## 验证

```bash
pytest tests/ops/test_clamp_min.py -q
python -m tileops.reporting.cli run --op ClampMinFwdOp --prof-mode msprof
```

## 结果

- 正确性：14/14
- 性能：7/7 tune shape ≥70%，ratio 70.08–77.31%
- 端到端 run：`20261023_183019_804116_ClampMinFwdOp`（status=passed）
