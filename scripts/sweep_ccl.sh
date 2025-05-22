#!/usr/bin/env bash
# sweep_matmul_ccl.sh — sweep distributed matmul with MPI + oneCCL
set -euo pipefail

# --- oneCCL / MPI env setup ------------------------------------
export CCL_KVS_IFACE=ens1f0np0

# transports to test
TRANSPORTS=(mpi ofi)

# allreduce algorithms to test (full range)
ALGS=(direct rabenseifner nreduce ring double_tree recursive_doubling 2d)

# --- two‐node hostnames/IPs ------------------------------------
NODE0="cluster-2u-node1"
NODE1="cluster-2u-node0"

# --- sweep parameters ------------------------------------------
SIZES=(64 128 256 512 1024 2048 4096 8192)
SLOTS=(1 2 4 8 16 32)
COUNT=10
WARMUP=5

SCRIPT="./bin/matmul_ccl"
RESULT_DIR="results"
LOG_DIR="logs"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

for transport in "${TRANSPORTS[@]}"; do
  export CCL_ATL_TRANSPORT="${transport}"

  for alg in "${ALGS[@]}"; do
    # set blanket algorithm for all message sizes
    export CCL_ALLREDUCE="${alg}:0-max"

    MASTER_CSV="${RESULT_DIR}/vanillaccl_sweep_${transport}_${alg}.csv"
    echo "=== Transport=$transport | Allreduce=$alg ==="
    echo "Writing results to ${MASTER_CSV}"

    header_written=false

    for slots in "${SLOTS[@]}"; do
      total_procs=$(( slots * 2 ))
      hosts_list="${NODE1},${NODE0}"

      for N in "${SIZES[@]}"; do
        echo ">>> transport=${transport} alg=${alg} slots=${slots} N=${N}"
        RUN_LOG="${LOG_DIR}/${transport}_${alg}_N${N}_S${slots}.log"
        TMP_CSV="${RESULT_DIR}/tmp_${transport}_${alg}_N${N}_S${slots}.csv"

        mpirun -np "${total_procs}" -ppn "${slots}" -hosts "${hosts_list}" \
          "${SCRIPT}" \
            --size "${N}" \
            --count "${COUNT}" \
            --warmup "${WARMUP}" \
            --outfile "${TMP_CSV}" \
            > "${RUN_LOG}" 2>&1

        if [[ -f "${TMP_CSV}" ]]; then
          if ! $header_written; then
            cat "${TMP_CSV}" > "${MASTER_CSV}"
            header_written=true
          else
            tail -n +2 "${TMP_CSV}" >> "${MASTER_CSV}"
          fi
          rm -f "${TMP_CSV}"
        else
          echo "⚠️  Missing ${TMP_CSV}, see ${RUN_LOG}"
        fi
      done
    done

    echo "✓ Completed sweep for transport=${transport}, alg=${alg}"
    echo
  done
done

echo "All sweeps complete."
echo "• Results directory: ${RESULT_DIR}/"
echo "• Logs directory:    ${LOG_DIR}/"
