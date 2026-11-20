import importlib.util, sys
spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["probe"]=m; spec.loader.exec_module(m)
B,S,H,Hk,D,dt = 1,128,8,2,64,'float16'
k = m._gqa_bwd_kernel(B,H,Hk,S,D,True,dt)
print("COMPILE_OK", sys.argv[1])
