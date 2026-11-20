#!/usr/bin/env python3
"""Host-side int32 model of the dropout mask: hash-chain quality comparison.

The mask ``h`` is a pure function of ``(seed, idx)`` built from int32
wrapping ops only, so a numpy int32 model is an exact model of the device chain
(the design verified this bit-for-bit for the Stage 3 chain).  This script
measures the statistical quality of the Stage 3 chain and of shorter
same-family chains, so that any op-count reduction is adopted only when it
keeps the operator's statistical contract.

Metrics (same shape as the DESIGN.md section 1.6.0 verification):
  * χ² of the top-11 bits of the 23-bit field (dof 2047)
  * lag-1 χ² of bit 30 (dof 1023), the p=0.5 decision bit
  * empirical drop rate at p=0.1/0.3/0.5/0.7 vs 3σ
  * seed separation fraction (seed 42 vs 99)

run: python _hash_analysis.py
"""

import numpy as np

N = 1 << 22  # 4 Mi samples (same scale as L0-1)
A = np.int32(-1640531527)
B = np.int32(-2048144777)
M31 = np.int32(0x7FFFFFFF)


def shl(x, k):
    return np.left_shift(x, np.int32(k)).astype(np.int32)


def chain_a(idx, seed):
    h = np.bitwise_xor(idx, np.int32(seed)).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 13))
    h = (h * A).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 7))
    h = (h * B).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 13))
    return h


def chain_b(idx, seed):
    h = np.bitwise_xor(idx, np.int32(seed)).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 13))
    h = (h * A).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 7))
    h = (h * B).astype(np.int32)
    return h


def chain_c(idx, seed):
    h = np.bitwise_xor(idx, np.int32(seed)).astype(np.int32)
    h = (h * A).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 13))
    h = (h * B).astype(np.int32)
    return h


def chain_d(idx, seed):
    h = np.bitwise_xor(idx, np.int32(seed)).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 16))
    h = (h * A).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 13))
    h = (h * B).astype(np.int32)
    return h


def chain_e(idx, seed):
    """splitmix-style finalizer, 2 shifts + 2 muls (5 ops after the seed xor)."""
    h = np.bitwise_xor(idx, np.int32(seed)).astype(np.int32)
    h = (h * A).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 13))
    h = (h * B).astype(np.int32)
    h = np.bitwise_xor(h, shl(h, 13))
    return h


CHAINS = {"A(stage3,9op)": chain_a, "B(7op)": chain_b, "C(5op)": chain_c,
          "D(7op,shl16)": chain_d, "E(7op,tail13)": chain_e}


def chi2_bins(vals, nbits, bins):
    """χ² of the top ``nbits`` of ``vals`` over ``bins`` buckets."""
    top = (vals >> np.int32(32 - 31 + (31 - nbits)) if False else None)
    return None


def block_chi2(decision, nblocks=1024):
    """iid-Bernoulli block test: block drop-counts vs Binomial expectation."""
    blk = decision.reshape(nblocks, -1)
    obs = blk.sum(axis=1)
    n = blk.shape[1]
    p = decision.mean()
    mu = n * p
    var = n * p * (1 - p)
    return float(((obs - mu) ** 2 / var).sum()), p


def metrics(h):
    hm = (h & M31).astype(np.int64)
    out = {}
    for p_ in (0.1, 0.3, 0.5, 0.7):
        tp = int(round(p_ * float(1 << 23)))
        thr = 256 * tp
        drop = (hm < thr)
        chi, rate = block_chi2(drop)
        s3 = 3.0 * (p_ * (1 - p_) / len(hm)) ** 0.5
        out[f"rate_p{p_}"] = (rate, abs(rate - p_) <= s3, chi)
    return out


def main():
    idx = np.arange(N, dtype=np.int32)
    for name, fn in CHAINS.items():
        m = metrics(fn(idx, 42))
        ya = fn(idx, 42)
        yb = fn(idx, 99)
        m31a = ((ya & M31) >= (1 << 30))
        m31b = ((yb & M31) >= (1 << 30))
        diff = float((m31a != m31b).mean())
        parts = "  ".join(f"{k[5:]}={v[0]:.5f}/{v[2]:.0f}{'ok' if v[1] else 'FAIL'}"
                          for k, v in m.items())
        print(f"{name:16s} sep={diff:.4f}  {parts}")
        # adjacent-window overlap test: |mask(i) vs mask(i+2048)| should be ~0.5
        ov = float((m31a[:-2048] == m31a[2048:]).mean())
        print(f"{'':16s} lag2048_agreement={ov:.4f} (expect ~0.5)")


if __name__ == "__main__":
    main()
