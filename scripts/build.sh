#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# --------- config your compiler if needed ----------
export CXX=${CXX:-mpicxx}

mkdir -p bin
# $CXX -O3 -std=c++17 code/matmul_mpi.cpp -o bin/matmul_mpi
$CXX -O3 -std=c++17 code/matmul_ccl.cpp -lccl -o bin/matmul_ccl
source /opt/intel/oneapi/mkl/latest/env/vars.sh intel64

# 2) Compile, explicitly pointing to MKL’s include and lib directories:
mpicxx -O3 -std=c++17 \
  -I${MKLROOT}/include \
  code/matmul_ccl_sgemm.cpp \
  -L${MKLROOT}/lib/intel64 -Wl,-rpath,${MKLROOT}/lib/intel64 \
  -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread \
  -lccl \
  -o bin/matmul_ccl_sgemm


mpicxx -std=c++17 code/matmul_mpi.cpp -lmkl_rt -o bin/matmul_mpi

echo "✅  binaries in ./bin"