"""Probe: contiguous 1-D complement fill for clean_logits_ on npuir.

The current kernel paints ``-inf`` through ``[block_N, kv_group]`` 2-D slices;
with ``kv_group == 1`` the innermost extent is one fp32 element, so every DMA
degenerates into a per-row scalar copy.  The complement of the keep-window
``[ks, ke)`` inside one ``(b, s)`` row is exactly two *contiguous* runs
``[0, ks)`` and ``[ke, Skv)`` in the flat ``[Skv, G]`` layout, so the same math
needs only bulk 1-D copies.

This probe checks which buffer-view / copy form survives the toolchain.

Forms:
  f1  T.view(Logits, [B*S*Skv*G]) + 1-D copy with unaligned GM offset
  f2  T.view(Logits, [B*S, Skv*G]) + 2-D copy, inner extent = run length
  f3  4-D narrow copy (current kernel shape) -- reference for correctness
  f4  1-D copy, head/inside handled by scalar stores, bulk 32B aligned
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

CHUNK = 4096


@T.macro
def _copies(F_flat, base, off, length, neg):
    """Emit chunked 1-D copies of ``-inf`` into ``F_flat[base+off : +length]``."""
    n = T.ceildiv(length, CHUNK)
    for i in T.serial(n):
        o = i * CHUNK
        tail = T.min(CHUNK, length - o)
        T.copy(neg[0:tail], F_flat[base + off + o:base + off + o + tail])


@T.macro
def _copies4(F, by, bx, off, length, neg):
    """Chunked rank-4 copy of ``-inf`` into ``F[by, bx, 0, off:off+length]``."""
    n = T.ceildiv(length, CHUNK)
    for i in T.serial(n):
        o = i * CHUNK
        tail = T.min(CHUNK, length - o)
        T.copy(neg[0:tail], F[by, bx, 0, off + o:off + o + tail])


@tilelang.jit(target="npuir")
def build(form):
    dtype = "float32"
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    kv_group = T.dynamic("kv_group")

    @T.prim_func
    def k(
        Logits: T.Tensor([batch, seq_len, seq_len_kv, kv_group], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], "int32"),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], "int32"),  # type: ignore
    ):
        with T.Kernel(batch * seq_len, is_npu=True) as (cid, _):
            by = cid // seq_len
            bx = cid % seq_len
            neg = T.alloc_ub([CHUNK], dtype)
            T.vbrc(T.float32(-T.infinity("float32")), neg)

            ks = T.max(T.min(CuSeqLenKS[bx], seq_len_kv), 0)
            ke = T.max(T.min(CuSeqLenKE[bx], seq_len_kv), ks)

            if form == "f1":
                F = T.view(Logits, [batch * seq_len * seq_len_kv * kv_group])
                base = (by * seq_len + bx) * seq_len_kv * kv_group
                _copies(F, base, 0, ks * kv_group, neg)
                _copies(F, base, ke * kv_group, (seq_len_kv - ke) * kv_group, neg)
            elif form == "f2":
                F = T.view(Logits, [batch * seq_len, seq_len_kv * kv_group])
                row = by * seq_len + bx
                _copies(F, row, 0, ks * kv_group, neg)
                _copies(F, row, ke * kv_group, (seq_len_kv - ke) * kv_group, neg)
            elif form == "f3":
                for t in T.serial(T.ceildiv(seq_len_kv, 256)):
                    k0 = t * 256
                    tail = T.min(256, seq_len_kv - k0)
                    if (k0 + tail <= ks) or (k0 >= ke):
                        T.copy(neg[0:tail], Logits[by, bx, k0:k0 + tail, 0])
            elif form == "f4":
                # head of the prefix: scalar stores until 32B alignment
                F = T.view(Logits, [batch, seq_len, 1, seq_len_kv * kv_group])
                base = (by * seq_len + bx) * seq_len_kv * kv_group
                n_head = 8 - (base % 8)
                for i in T.serial(n_head):
                    if i < ks * kv_group:
                        F[by, bx, 0, i] = T.float32(-T.infinity("float32"))
                n_pre = T.max(ks * kv_group - n_head, 0)
                _copies(F, by, bx, n_head, n_pre, neg)
                n_suf = (seq_len_kv - ke) * kv_group
                _copies(F, by, bx, ke * kv_group, n_suf, neg)
            elif form == "f5":
                # rank-preserving view: [B, S, 1, Skv*G] -> inner extent is the
                # whole contiguous run, burst = L*4 B instead of G*4 B.
                F = T.view(Logits, [batch, seq_len, 1, seq_len_kv * kv_group])
                _copies4(F, by, bx, 0, ks * kv_group, neg)
                _copies4(F, by, bx, ke * kv_group, (seq_len_kv - ke) * kv_group, neg)
            else:
                raise ValueError(form)

    return k


def golden(logits, ks, ke):
    b, s, skv, g = logits.shape
    ks = torch.clamp(ks, min=0, max=skv)
    ke = torch.clamp(ke, max=skv)
    ke = torch.maximum(ke, ks)
    ar = torch.arange(0, skv)[None, :]
    keep = ((ar >= ks[:, None]) & (ar < ke[:, None]))[None, :, :, None]
    return torch.where(keep, logits, torch.full_like(logits, float("-inf")))


def run(form, b, s, skv, g, seed=0, special=False):
    torch.manual_seed(seed)
    dev = "npu"
    x = torch.randn(b, s, skv, g, device=dev, dtype=torch.float32)
    if special:
        flat = x.view(-1)
        flat[::37] = float("nan")
        flat[1::53] = float("inf")
    before = x.cpu().clone()
    ks = torch.randint(0, skv + 1, (s,), dtype=torch.int32)
    span = (skv - ks).float()
    extra = (torch.rand(s) * (span + 1)).floor().to(torch.int32)
    ke = ks + torch.clamp(extra, max=span.to(torch.int32))
    out = x.clone()
    build(form)(out, ks.to(dev), ke.to(dev))
    torch.npu.synchronize()
    ref = golden(before, ks, ke)
    got = out.cpu()
    bad = int((torch.isfinite(got) != torch.isfinite(ref)).sum())
    both = torch.isfinite(got) & torch.isfinite(ref)
    max_abs = float((got[both] - ref[both]).abs().max()) if both.any() else 0.0
    print(f"[{form}] b{b} s{s} skv{skv} g{g} seed{seed} bad_mask={bad} max_abs={max_abs}")
    return bad == 0 and max_abs == 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="f1")
    ap.add_argument("--special", action="store_true")
    a = ap.parse_args()
    ok = True
    for seed, shape in enumerate([(1, 64, 1024, 1), (1, 64, 1000, 1), (1, 32, 777, 2),
                                  (2, 32, 512, 1)]):
        ok &= run(a.form, *shape, seed=seed, special=a.special)
    print("RESULT", a.form, "PASS" if ok else "FAIL")
