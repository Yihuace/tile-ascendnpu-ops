# Toolchain env for Conv1dFwdOp / Conv1dBiasFwdOp Stage-4 runs (non-interactive shells).
# The conv1d kernels are Cube/Expert (T.Scope("Cube") + T.gemm), so force Expert
# (overrides the .bashrc default Developer).
export LD_LIBRARY_PATH=/home/developer/tilelang-mlir-ascend/build:/home/developer/tilelang-mlir-ascend/build/tvm:/home/developer/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/build/install/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/developer/tilelang-mlir-ascend:/home/developer/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/build/install/python_packages/mlir_core:/home/developer/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/build/install/python_packages/bishengir:$PYTHONPATH
export TILELANG_ASCEND_MODE=Expert
