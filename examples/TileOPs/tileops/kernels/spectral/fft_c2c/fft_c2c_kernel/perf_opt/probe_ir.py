import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
n = int(sys.argv[2]); batch = int(sys.argv[3])
k = mod._fft_c2c_kernel(n, batch)
out = sys.argv[4] if len(sys.argv) > 4 else f"logs/ir_n{n}_b{batch}.mlir"
mlir = k.mlir_content
src = k.utils_kernel_src if isinstance(k.utils_kernel_src, str) else str(k.utils_kernel_src)
with open(out, "w") as f: f.write(mlir)
print("wrote", out, len(mlir), "chars")
