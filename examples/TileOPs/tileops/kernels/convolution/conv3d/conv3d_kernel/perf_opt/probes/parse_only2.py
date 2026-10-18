import importlib.util, os, sys, traceback
os.environ["TILELANG_ASCEND_MODE"] = "Expert"
spec = importlib.util.spec_from_file_location("m", os.path.abspath(sys.argv[1]))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
class FakeNPU:
    def get_aicore_num(self): return 20
class FakeUtils:
    @staticmethod
    def get(): return FakeNPU()
m.NPUUtils = FakeUtils
fn = m._conv3d_kernel(1, 32, 32, 64, 64, 64, 3, 3, 3, (1,1,1), (1,1,1), (1,1,1), True, "bfloat16")
try:
    fn(64, 128, 128, 1)
    print("PARSE+COMPILE OK")
except Exception as e:
    print("=== FULL MESSAGE ===")
    print(str(e)[:6000])
