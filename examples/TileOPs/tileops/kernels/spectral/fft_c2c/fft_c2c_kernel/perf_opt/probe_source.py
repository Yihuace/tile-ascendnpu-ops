"""Dump the compiled device source of the FFT kernel for code-quality diagnosis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
n = int(sys.argv[2]); batch = int(sys.argv[3])
k = mod._fft_c2c_kernel(n, batch)
src = k.get_kernel_source()
out = sys.argv[4] if len(sys.argv) > 4 else f"logs/probe_n{n}_b{batch}.cce"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    f.write(src)
print(f"wrote {out}: {len(src)} bytes, {src.count(b"\n")} lines")
