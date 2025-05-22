#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# --------- config your compiler if needed ----------
export CXX=${CXX:-mpicxx}

mkdir -p bin
$CXX -O3 -std=c++17 code/matmul_mpi.cpp -o bin/matmul_mpi
$CXX -O3 -std=c++17 code/matmul_ccl.cpp -lccl -o bin/matmul_ccl
echo "✅  binaries in ./bin"