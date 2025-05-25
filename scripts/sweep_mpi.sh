#!/usr/bin/env bash
# sweep_matmul_mpi.sh — sweep distributed matmul with pure MPI
set -euo pipefail

# two‐node hostnames/IPs
NODE0="cluster-2u-node1"
NODE1="cluster-2u-node0"

# sweep parameters
SIZES=(64 128 256 512 1024 2048 4096 8192)
SLOTS=(1 2 4 8 16 32)
COUNT=10
WARMUP=5
SCRIPT="./bin/matmul_mpi"
RESULT_DIR="results"
LOG_DIR="logs"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

MASTER_CSV="${RESULT_DIR}/mpi_sgemm_sweep.csv"
header_written=false

for slots in "${SLOTS[@]}"; do
  total_procs=$(( slots * 2 ))
  hosts_list="${NODE1},${NODE0}"

  for N in "${SIZES[@]}"; do
    echo ">>> slots=${slots} N=${N}"
    RUN_LOG="${LOG_DIR}/mpi_sgemm_N${N}_S${slots}.log"
    TMP_CSV="${RESULT_DIR}/tmp_mpi_sgemm_N${N}_S${slots}.csv"

    mpirun -np "${total_procs}" -ppn "${slots}" -hosts "${hosts_list}" \
      "${SCRIPT}" \
        --size "${N}" \
        --count "${COUNT}" \
        --warmup "${WARMUP}" \
        --outfile "${TMP_CSV}" \
      > "${RUN_LOG}" 2>&1

    if [[ -f "${TMP_CSV}" ]]; then
      if ! $header_written; then
        cp "${TMP_CSV}" "${MASTER_CSV}"
        header_written=true
      else
        tail -n +2 "${TMP_CSV}" >> "${MASTER_CSV}"
      fi
      rm -f "${TMP_CSV}"
    else
      echo "⚠️ Missing ${TMP_CSV}, see ${RUN_LOG}"
    fi
  done
done

echo "✓ Completed MPI sweep"
echo "• Results directory: ${RESULT_DIR}/"
echo "• Master CSV:        ${MASTER_CSV}"
