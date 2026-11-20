# Conv1dBiasFwdOp — Review

## Verdict

**Pass.** The op is a thin contract wrapper over the already-reviewed
`Conv1dFwdOp`; no new kernel code is introduced.

## Scope review

- `Conv1dBiasFwdOp` subclasses `Conv1dFwdOp` and only overrides `forward()`.
- The bias contract is tightened: required positional `bias`, `None` rejected,
  dtype/device/shape validated, then delegated to the base implementation.
- Dispatch (`_select_kernel_key`), kernel construction, kernel cache, and
  `eval_roofline()` are inherited unchanged.
- The CANN 9.0 Vector fallback added by `Conv1dFwdOp` is inherited, so the
  known `bishengir-compile` SIGSEGV (CG-2026-0003) is not reintroduced.

## API review

| Check | Result |
|---|---|
| `bias=None` | raises `ValueError` |
| bias not on device | raises `ValueError` |
| bias dtype mismatch | raises `ValueError` |
| bias device mismatch | raises `ValueError` |
| bias shape not `[C_out]` | raises `ValueError` |
| valid bias | delegates to `Conv1dFwdOp.forward` |
| manifest kernel_map | identical to `Conv1dFwdOp` |
| roofline | bias FLOP/byte terms always counted |

## Correctness

Executed on the target CANN 9.0.0 / Ascend910B3 environment:

```text
python3 -m pytest tests/ops/test_conv1d_bias.py -q
28 passed, 2 warnings in 104.13s
```

The suite covers:

- fixed-shape correctness across fp16/bf16;
- stride / padding / dilation / groups / depthwise variants;
- kernel dispatch keys (`conv1d`, `conv1d_unit_stride`, `conv1d_pointwise`,
  `group_conv1d`);
- rejected dtype / bias contract cases;
- `eval_roofline()` consistency.

Golden reference: `torch.nn.functional.conv1d(..., bias=bias)` computed in
fp32 and cast back to the input dtype.

## Performance

No new kernel and no new tuning were added. The bias variant executes the same
CANN 9.0 compiler-safe Vector kernel path as `Conv1dFwdOp` with
`has_bias=True`. Measurement details are in the generated
`reports/tileops/*_Conv1dBiasFwdOp/` report.

## Risks

- The bias variant depends on the shared Conv1d kernel package; any future
  change to `Conv1dFwdOp` dispatch or the Vector fallback affects both ops.
  The shared test suite (`test_conv1d.py`) and this suite must both stay green.
- Bias is always present, so the no-bias kernel variant is never exercised by
  this op. That is intentional and matches the GPU baseline split.

## Required evidence

- `DESIGN.md` (this directory)
- `REVIEW.md` (this file)
- `RETROSPECTIVE.md`
- `integration_log.md`
- generated `run.json` / `report.md` under `reports/tileops/`
