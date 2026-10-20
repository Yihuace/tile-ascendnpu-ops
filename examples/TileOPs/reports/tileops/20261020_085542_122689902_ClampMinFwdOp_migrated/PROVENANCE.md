# ClampMinFwdOp —— 迁移态（Stage-3 baseline 内核）benchmark-only

- 生成时间：2026-10-20T09:04:41+08:00
- 工具链：CANN 9.0.0 / Ascend910B3（`source /mnt/workspace/tileops_env.sh`，tilelang 取 dev 树）
- **这是迁移态，不是交付态**：运行期间把
  `tileops/kernels/elementwise/clamp_scalar/clamp_scalar.py` 的 kernel 选择临时切到
  Stage-3 baseline（`clamp_scalar_kernel/`），跑完后已按字节还原（本目录不含改动后的 wrapper）。
- 命令：
  ```
  source /mnt/workspace/tileops_env.sh
  cd examples/TileOPs
  TILEOPS_PROF_MODE=msprof TILEOPS_MSPROF_KEEP_OUTPUT=1 \
  TILEOPS_MSPROF_ARTIFACT_ROOT=<run_dir>/artifacts/msprof \
  TILEOPS_BENCHMARK_REPORT_PATH=<run_dir>/benchmark.md \
  python3 -m pytest benchmarks/ops/bench_clamp_min.py -sv --junitxml=<run_dir>/pytest/benchmark.xml -m "not smoke"
  ```
- **为什么只跑 benchmark**：正确性已由 CANN 9.0.0 全量重跑覆盖
  （`docs/cann9_adaptation.md` §11），这里只为拿迁移态 ratio，故跳过正确性闸门。
- **本目录不含正确性结论**、也没有 `run.json`；数据口径见 `perf_data/README.md`。
