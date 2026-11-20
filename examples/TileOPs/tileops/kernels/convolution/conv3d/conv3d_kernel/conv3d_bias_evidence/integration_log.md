# Conv3dBiasFwdOp 集成记录

## 文件

- Op：`tileops/ops/convolution.py`（`Conv3dFwdOp` / `Conv3dBiasFwdOp`）
- Kernel：`tileops/kernels/convolution/conv3d/`
- Manifest：`tileops/manifest/convolution.yaml`（`Conv3dBiasFwdOp`）
- Workload：`tileops/workloads/convolution.py`（`Conv3dWorkload`）
- Test / Bench：`tests/ops/test_conv3d_bias.py`、`benchmarks/ops/bench_conv3d_bias.py`
- 优化报告：`reports/optimize/Conv3dBiasFwdOp/OPTIMIZE_REPORT.md`

## 验证

```bash
pytest tests/ops/test_conv3d_bias.py -q
python -m tileops.reporting.cli run --op Conv3dBiasFwdOp --prof-mode msprof
```

## 结果

- 正确性：17/17
- 性能：r3d-stem-bias 2.36×；unet 21.3×；video-down 噪声内回退
- 端到端 run：`20261023_184106_697102_Conv3dBiasFwdOp`（status=passed，17/17，ratio 1.64–20.43%）

## 说明

- Conv1d 与 Conv3d 共用 `convolution.py` host 代码；本分支只抽出 Conv3d 部分，避免把 Conv1d 带入。
- 若本次端到端因 `bishengir-compile` 崩溃失败，本 PR 不提交，转为阻塞记录。
