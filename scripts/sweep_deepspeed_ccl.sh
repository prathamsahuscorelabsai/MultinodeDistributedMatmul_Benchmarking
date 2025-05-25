#!/usr/bin/env bash
# sweep_deepspeed_matmul.sh — DeepSpeed benchmark sweep on two nodes
# including oneCCL ATL transports and allreduce algorithms
set -euo pipefail

# ---------- cluster specifics ----------------------------------
NODE0="10.250.11.10"
NODE1="10.250.11.11"

export CCL_KVS_IFACE=ens1f0np0     # your regular Ethernet link

# ---------- transport & algorithm options ----------------------
# TRANSPORTS=(mpi ofi)
TRANSPORTS=(mpi)
ALGS=(direct rabenseifner nreduce ring double_tree recursive_doubling 2d)
# ALGS=(ring double_tree recursive_doubling 2d)

# ---------- sweep parameters ----------------------------------
SIZES=(64 128 256 512 1024 2048 4096 8192)
SLOTS=(1 2 4 8 16 32)
COUNT=20
WARMUP=5
DTYPE=fp32
SCRIPT="code/deepspeed_matmul.py"
LAUNCHER=impi       # Intel-MPI launcher

# ---------- directories ----------------------------------------
ROOT_DIR="$(pwd)"
RESULT_DIR="${ROOT_DIR}/results"
LOG_DIR="${ROOT_DIR}/logs"
HOST_DIR="${ROOT_DIR}/hostfiles"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${HOST_DIR}"

# ---------- outer loops ---------------------------------------
for transport in "${TRANSPORTS[@]}"; do
  export CCL_ATL_TRANSPORT="${transport}"

  for alg in "${ALGS[@]}"; do
    # apply this algorithm to all sizes
    export CCL_ALLREDUCE="${alg}:0-max"

    MASTER_CSV="${RESULT_DIR}/sweep_deepspeed_ccl_${transport}_${alg}.csv"
    header_written=false
    echo "=== Transport=${transport} | Allreduce=${alg} ==="
    echo "Writing to ${MASTER_CSV}"  

    for slots in "${SLOTS[@]}"; do
      total_procs=$(( slots * 2 ))
      HF="${HOST_DIR}/hostfile_${slots}.txt"

      # generate hostfile
      cat > "${HF}" <<EOF
${NODE0} slots=${slots}
${NODE1} slots=${slots}
EOF

      for N in "${SIZES[@]}"; do
        RUN_LOG="${LOG_DIR}/deepspeed_${transport}_${alg}_N${N}_S${slots}.log"
        TMP_CSV="${RESULT_DIR}/tmp_${transport}_${alg}_N${N}_S${slots}.csv"

        echo ">>> transport=${transport} alg=${alg} slots=${slots} N=${N}"

        deepspeed --hostfile "${HF}" \
                  --master_addr "${NODE0}" \
                  --master_port 29501 \
                  --launcher "${LAUNCHER}" \
                  --bind_cores_to_rank \
                  "${SCRIPT}" \
                    --size "${N}" \
                    --dtype "${DTYPE}" \
                    --count "${COUNT}" \
                    --warmup "${WARMUP}" \
                    --backend ccl \
                    --ccl \
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
          echo "⚠️  Missing ${TMP_CSV}, see ${RUN_LOG}" >&2
        fi
      done
    done

    echo "✓ Completed transport=${transport}, alg=${alg}"
    echo
  done
done

echo "All sweeps complete."
echo "• Results: ${RESULT_DIR}/"
echo "• Logs:    ${LOG_DIR}/"
