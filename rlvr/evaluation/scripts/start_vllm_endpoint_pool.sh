#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the HF repo id or local merged checkpoint path.}"

HOST="${HOST:-127.0.0.1}"
PORT_START="${PORT_START:-18000}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-29000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
GPU_GROUPS="${GPU_GROUPS:-0 1 2 3 4 5 6 7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_IMAGES="${MAX_IMAGES:-24}"
MAX_VIDEOS="${MAX_VIDEOS:-3}"
if [[ -z "${LIMIT_MM_PER_PROMPT:-}" ]]; then
  LIMIT_MM_PER_PROMPT="$(printf '{\"image\":%s,\"video\":%s}' "${MAX_IMAGES}" "${MAX_VIDEOS}")"
fi
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-8}"
VLLM_OMP_NUM_THREADS="${VLLM_OMP_NUM_THREADS:-${CPU_THREADS_PER_PROCESS}}"
VLLM_MKL_NUM_THREADS="${VLLM_MKL_NUM_THREADS:-${CPU_THREADS_PER_PROCESS}}"
VLLM_OPENBLAS_NUM_THREADS="${VLLM_OPENBLAS_NUM_THREADS:-${CPU_THREADS_PER_PROCESS}}"
VLLM_NUMEXPR_NUM_THREADS="${VLLM_NUMEXPR_NUM_THREADS:-${CPU_THREADS_PER_PROCESS}}"
VLLM_TOKENIZERS_PARALLELISM="${VLLM_TOKENIZERS_PARALLELISM:-false}"
VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
CPU_AFFINITY_GROUPS="${CPU_AFFINITY_GROUPS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REASONING_PARSER="${REASONING_PARSER:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}"
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-}"
GENERATION_CONFIG="${GENERATION_CONFIG:-}"
WAIT_READY="${WAIT_READY:-1}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
LOG_DIR="${LOG_DIR:-logs/vllm/endpoints/${SERVED_MODEL_NAME}_$(date -u +%Y%m%dT%H%M%SZ)}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pids.txt}"

mkdir -p "${LOG_DIR}"
: > "${PID_FILE}"

if [[ "${GPU_GROUPS}" == *";"* ]]; then
  IFS=';' read -r -a GROUP_ARRAY <<< "${GPU_GROUPS}"
else
  read -r -a GROUP_ARRAY <<< "${GPU_GROUPS}"
fi

AFFINITY_ARRAY=()
if [[ -n "${CPU_AFFINITY_GROUPS}" ]]; then
  IFS=';' read -r -a AFFINITY_ARRAY <<< "${CPU_AFFINITY_GROUPS}"
  if [[ "${#AFFINITY_ARRAY[@]}" -ne "${#GROUP_ARRAY[@]}" ]]; then
    echo "[error] CPU_AFFINITY_GROUPS must contain one semicolon-delimited CPU list per GPU_GROUPS entry" >&2
    exit 1
  fi
fi

ports=()
for i in "${!GROUP_ARRAY[@]}"; do
  group="${GROUP_ARRAY[$i]}"
  [[ -n "${group}" ]] || continue
  port=$((PORT_START + i))
  ports+=("${port}")
  tp="${TENSOR_PARALLEL_SIZE:-}"
  if [[ -z "${tp}" ]]; then
    compact="${group//,/ }"
    read -r -a gpu_ids <<< "${compact}"
    tp="${#gpu_ids[@]}"
  fi
  log="${LOG_DIR}/endpoint_${i}_port_${port}_gpu_${group//,/}.log"
  vllm_port=$((VLLM_PORT_BASE + i * 100))
  affinity="${AFFINITY_ARRAY[$i]:-}"
  launch_prefix=()
  if [[ -n "${affinity}" ]]; then
    launch_prefix=(taskset --cpu-list "${affinity}")
  fi
  server_args=(
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${port}"
    --trust-remote-code
    --tensor-parallel-size "${tp}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
  )
  if [[ -n "${REASONING_PARSER}" ]]; then
    server_args+=(--reasoning-parser "${REASONING_PARSER}")
  fi
  if [[ -n "${CHAT_TEMPLATE}" ]]; then
    server_args+=(--chat-template "${CHAT_TEMPLATE}")
  fi
  if [[ -n "${ALLOWED_LOCAL_MEDIA_PATH}" ]]; then
    server_args+=(--allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}")
  fi
  if [[ -n "${MM_PROCESSOR_KWARGS}" ]]; then
    server_args+=(--mm-processor-kwargs "${MM_PROCESSOR_KWARGS}")
  fi
  if [[ -n "${GENERATION_CONFIG}" ]]; then
    server_args+=(--generation-config "${GENERATION_CONFIG}")
  fi
  echo "[start] port=${port} vllm_port=${vllm_port} gpus=${group} tp=${tp} cpus=${affinity:-unbound} threads=${CPU_THREADS_PER_PROCESS} model=${MODEL_PATH} served=${SERVED_MODEL_NAME}"
  CUDA_VISIBLE_DEVICES="${group}" \
  VLLM_PORT="${vllm_port}" \
  VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
  VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE}" \
  OMP_NUM_THREADS="${VLLM_OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${VLLM_MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${VLLM_OPENBLAS_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${VLLM_NUMEXPR_NUM_THREADS}" \
  TOKENIZERS_PARALLELISM="${VLLM_TOKENIZERS_PARALLELISM}" \
  nohup setsid "${launch_prefix[@]}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server "${server_args[@]}" \
    > "${log}" 2>&1 &
  echo "$! ${port} ${group} ${log}" >> "${PID_FILE}"
done

cat > "${LOG_DIR}/endpoints.txt" <<EOF
$(for port in "${ports[@]}"; do printf 'http://%s:%s/v1\n' "${HOST}" "${port}"; done)
EOF

echo "[started] pid_file=${PID_FILE}"
echo "[started] endpoints=${LOG_DIR}/endpoints.txt"

if [[ "${WAIT_READY}" == "1" ]]; then
  deadline=$((SECONDS + READY_TIMEOUT_SEC))
  for port in "${ports[@]}"; do
    until curl -fsS --max-time 5 "http://${HOST}:${port}/v1/models" >/dev/null; do
      if (( SECONDS >= deadline )); then
        echo "[error] endpoint on port ${port} did not become ready within ${READY_TIMEOUT_SEC}s" >&2
        exit 1
      fi
      sleep 5
    done
    echo "[ready] port=${port}"
  done
fi
