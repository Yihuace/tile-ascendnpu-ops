#!/bin/bash
# Stage 1 design-probe environment (CANN 9.0.0 + tilelang dev root + Ascend910B3).
# Usage: bash _probe_env.sh python3 design_probe_xxx.py
source /home/developer/Ascend/cann/set_env.sh >/dev/null 2>&1
export PYTHONPATH=/home/developer/tilelang-mlir-ascend:/home/developer/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/build/install/python_packages/mlir_core:/home/developer/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/build/install/python_packages/bishengir
export LD_LIBRARY_PATH=/home/developer/tilelang-mlir-ascend/build:/home/developer/tilelang-mlir-ascend/build/tvm:/home/developer/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/build/install/lib:$LD_LIBRARY_PATH
export TILELANG_ASCEND_MODE=Expert
exec "$@"
