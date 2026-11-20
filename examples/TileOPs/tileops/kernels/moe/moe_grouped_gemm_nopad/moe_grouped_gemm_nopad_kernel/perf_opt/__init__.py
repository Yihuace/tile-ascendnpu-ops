"""Stage 4 tuned drop-in for MoeGroupedGemmNopadFwdOp (see opt_log.md).

Both modules are imported directly by the wrapper's perf_opt switch, so each
self-pins its ``TILELANG_ASCEND_MODE`` (Expert / Developer).  No re-export is
done here on purpose: the wrapper must get the factories from the submodules.
"""
