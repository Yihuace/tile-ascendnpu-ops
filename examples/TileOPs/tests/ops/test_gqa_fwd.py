"""Correctness tests for GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp (NPU).

The reference is the kernel package's own ``golden_gqa_prefill_paged``, which is
written straight from the op's definition (per-request dense attention over the
dequantized cache pages plus the new tokens) and shares NO code with the NPU
algorithm -- the NPU path is three single-engine kernels with the paged gather,
dequant, head repeat and causal mask on the host.

Tolerances: the score matrix is produced by an fp16 Cube GEMM and the softmax runs
in fp32, so fp16-level agreement is what is expected (measured end to end:
rel 4.6e-4 causal, 6.5e-4 non-causal).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.attention.gqa_fwd._gqa_decomposed_kernel import (
    golden_gqa_prefill_paged,
)
from tileops.ops.attention.gqa_fwd import GQAPrefillPagedWithKVCacheFwdOp
from tileops.workloads.attention import GqaPagedWorkload
from tileops.workloads.workload_base import FixtureBase

_TOL = 2e-2          # fp16 scores -> fp32 softmax -> fp16 out


def _spec(total_q, batch, q_lens, cache_lens, heads, heads_kv, dim, page_size,
          max_pages, max_seqlen_q, is_causal=True):
    return {
        "total_q": total_q, "batch": batch, "q_lens": q_lens,
        "cache_lens": cache_lens, "heads": heads, "heads_kv": heads_kv,
        "dim": dim, "page_size": page_size, "max_pages_per_req": max_pages,
        "max_seqlen_q": max_seqlen_q, "is_causal": is_causal,
    }


class GqaFixture(FixtureBase):
    PARAMS = [
        (
            "spec, dtype",
            [
                # smoke: the smallest shape that still exercises paging + GQA
                pytest.param(
                    _spec(128, 2, [64, 64], [128, 128], 4, 2, 64, 16, 8, 64),
                    torch.float16, marks=[pytest.mark.smoke]),
                # k_len larger than one page, and H_kv == 1 (all heads share a kv head)
                pytest.param(
                    _spec(64, 1, [64], [128], 4, 1, 64, 16, 8, 64),
                    torch.float16, marks=[pytest.mark.smoke]),
                # unequal request lengths in one batch
                pytest.param(
                    _spec(96, 2, [32, 64], [64, 128], 4, 2, 64, 16, 8, 64),
                    torch.float16, marks=[pytest.mark.full]),
                # non-causal
                pytest.param(
                    _spec(128, 2, [64, 64], [128, 128], 4, 2, 64, 16, 8, 64,
                          is_causal=False),
                    torch.float16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@GqaFixture
def test_gqa_fwd(spec: dict, dtype: torch.dtype) -> None:
    wl = GqaPagedWorkload(spec, dtype)
    op = GQAPrefillPagedWithKVCacheFwdOp(
        page_size=wl.page_size, is_causal=spec["is_causal"])
    inputs = wl.gen_inputs()
    got = op(*inputs)

    ref = golden_gqa_prefill_paged(
        *[t.cpu() for t in inputs],
        page_size=wl.page_size,
        is_causal=spec["is_causal"],
    )
    g = got.cpu().float()
    r = ref.float()
    err = (g - r).abs().max().item()
    rel = err / max(r.abs().max().item(), 1e-9)
    assert rel < _TOL, (
        f"rel err {rel:.3e} exceeds {_TOL} (abs {err:.3e}); "
        f"shape={tuple(g.shape)} is_causal={spec['is_causal']}"
    )


@pytest.mark.smoke
def test_gqa_fwd_rejects_bad_head_ratio() -> None:
    wl = GqaPagedWorkload(
        _spec(64, 1, [64], [64], 4, 3, 64, 16, 4, 64), torch.float16)
    op = GQAPrefillPagedWithKVCacheFwdOp(page_size=16)
    with pytest.raises(ValueError, match="multiple of H_kv"):
        op(*wl.gen_inputs())


@pytest.mark.smoke
def test_gqa_fwd_page_table_isolation() -> None:
    """Two requests in one batch must read DIFFERENT page ranges.

    If the page table were ignored the two requests would attend to the same cache
    and the two halves of the output would come out identical; this checks they do
    not.
    """
    spec = _spec(128, 2, [64, 64], [64, 64], 2, 2, 64, 16, 4, 64)
    wl = GqaPagedWorkload(spec, torch.float16)
    op = GQAPrefillPagedWithKVCacheFwdOp(page_size=16)
    got = op(*wl.gen_inputs()).cpu().float()
    first, second = got[:64], got[64:]
    assert not torch.allclose(first, second, atol=1e-3), (
        "the two requests produced identical output -- the page table is not "
        "isolating the cache reads"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
