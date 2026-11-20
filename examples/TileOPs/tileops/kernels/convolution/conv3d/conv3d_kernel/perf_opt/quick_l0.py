#!/usr/bin/env python3
"""Quick L0 precision sanity check for the conv3d kernel (baseline or candidate)."""
import os
os.environ["TILELANG_ASCEND_MODE"] = "Expert"
import torch
import torch_npu  # noqa: F401
from tilelang.utils import NPUUtils

import importlib.util
import sys

MOD = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None

def load_module(path):
    name = "mod_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def check(m, n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw, stride, pad, dil, dtype, tag):
    from tilelang.utils import NPUUtils
    fn = m._conv3d_kernel(n, c_in, d_in, h_in, w_in, c_out, kd, kh, kw,
                          stride, pad, dil, True, {torch.float16:"float16",torch.bfloat16:"bfloat16"}[dtype])
    kernel = fn(64, 128, 128, 1)
    sd=sh=sw=stride; pd=ph=pw=pad; dd=dh=dw=dil
    out_d=(d_in+2*pd-dd*(kd-1)-1)//sd+1
    out_h=(h_in+2*ph-dh*(kh-1)-1)//sh+1
    out_w=(w_in+2*pw-dw*(kw-1)-1)//sw+1
    d_pad=d_in+2*pd; h_pad=h_in+2*ph
    need_poly, w_pad, mdec, c_in_pad = m._ws_dims(w_in, c_in, pw, sw)
    torch.manual_seed(0)
    x = torch.randn(n, c_in, d_in, h_in, w_in, dtype=dtype, device="npu")
    weight = torch.randn(c_out, c_in, kd, kh, kw, dtype=dtype, device="npu")
    bias = torch.randn(c_out, dtype=dtype, device="npu")
    x_pad, xdec, w_flat = m._host_prep(x, weight, c_in_pad, d_pad, h_pad, w_pad,
        sd, sh, sw, pd, ph, pw, d_in, h_in, w_in, kd, kh, kw, need_poly, mdec, dtype)
    nk = NPUUtils.get().get_aicore_num()
    ws = torch.empty((nk, 2, 64, 128), dtype=torch.float32, device="npu")
    out = kernel(x_pad, w_flat, bias, xdec, ws)
    torch.npu.synchronize()
    out = out.reshape(n, c_out, out_d, out_h, out_w)
    # golden
    ref = torch.nn.functional.conv3d(x.cpu().float(), weight.cpu().float(),
        bias.cpu().float(), stride=stride, padding=pad, dilation=dil, groups=1)
    ref = ref.to(dtype)
    diff = (out.cpu().float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    atol, rtol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)
    # assert_close criterion: |a-e| <= atol + rtol*|e|
    ok = bool((diff <= atol + rtol * ref.float().abs()).all().item())
    print(f"[{tag}] n={n} c_in={c_in} {d_in}x{h_in}x{w_in} c_out={c_out} k={kd} "
          f"stride={stride} pad={pad} {dtype}: {'PASS' if ok else 'FAIL'} max_abs={max_abs:.4e}")
    return ok

if __name__ == "__main__":
    m = load_module(MOD)
    ok = True
    ok &= check(m, 1, 3, 16, 112, 112, 64, 3, 3, 3, 1, 1, 1, torch.float16, "r3d-stem")
    ok &= check(m, 1, 64, 8, 56, 56, 128, 3, 3, 3, 2, 1, 1, torch.float16, "video-down")
    ok &= check(m, 1, 32, 32, 64, 64, 64, 3, 3, 3, 1, 1, 1, torch.bfloat16, "unet")
    ok &= check(m, 1, 8, 8, 16, 16, 16, 3, 3, 3, 1, 1, 1, torch.float16, "smoke")
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)
