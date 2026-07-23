#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_ROOT="${EVALUATION_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-$(cd "${EVALUATION_ROOT}/../.." && pwd)}"
SCRIPTS_ROOT="${EVALUATION_ROOT}/scripts"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN="$(command -v "${PYTHON_BIN}")" || {
  echo "[fatal] evaluation Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: scripts/run_trace_eval.sh \
  --model SLUG LOCAL_PATH IMMUTABLE_REVISION SOURCE DISPLAY_LABEL \
  [--model SLUG LOCAL_PATH IMMUTABLE_REVISION SOURCE DISPLAY_LABEL ...] \
  --seeds SEED [SEED ...] [--delta LABEL=MINUEND_SLUG=SUBTRAHEND_SLUG] \
  [--run-tag TAG] [--print-config]
EOF
}

MODEL_SLUGS=()
MODEL_PATHS=()
MODEL_REVISIONS=()
MODEL_SOURCES=()
MODEL_LABELS=()
SEEDS=()
DELTAS=()
RUN_TAG_ARG=""
PRINT_CONFIG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 6 ]] || { usage >&2; exit 2; }
      MODEL_SLUGS+=("$2")
      MODEL_PATHS+=("$3")
      MODEL_REVISIONS+=("$4")
      MODEL_SOURCES+=("$5")
      MODEL_LABELS+=("$6")
      shift 6
      ;;
    --seeds)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --delta)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      DELTAS+=("$2")
      shift 2
      ;;
    --run-tag)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      RUN_TAG_ARG="$2"
      shift 2
      ;;
    --print-config)
      PRINT_CONFIG=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[fatal] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ ${#MODEL_SLUGS[@]} -gt 0 ]] || { echo "[fatal] pass at least one --model" >&2; exit 2; }
[[ ${#SEEDS[@]} -gt 0 ]] || { echo "[fatal] pass at least one --seeds value" >&2; exit 2; }
declare -A seen_models=()
immutable_source_re='^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*@([0-9a-f]{40}|[0-9a-f]{64})$'
for i in "${!MODEL_SLUGS[@]}"; do
  slug="${MODEL_SLUGS[$i]}"
  source="${MODEL_SOURCES[$i]}"
  [[ -n "${slug}" ]] || { echo "[fatal] model slug cannot be empty" >&2; exit 2; }
  [[ -z "${seen_models[$slug]:-}" ]] || { echo "[fatal] duplicate model slug: ${slug}" >&2; exit 2; }
  [[ "${source}" =~ ${immutable_source_re} ]] || {
    echo "[fatal] model source must be owner/repo@immutable-commit: ${source}" >&2
    exit 2
  }
  seen_models[$slug]=1
done
declare -A seen_seeds=()
for seed in "${SEEDS[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "[fatal] invalid seed: ${seed}" >&2; exit 2; }
  [[ -z "${seen_seeds[$seed]:-}" ]] || { echo "[fatal] duplicate seed: ${seed}" >&2; exit 2; }
  seen_seeds[$seed]=1
done

seed_tag="$(IFS=_; echo "${SEEDS[*]}")"
RUN_TAG="${RUN_TAG_ARG:-${RUN_TAG:-trace_eval_v1_temp06_seed${seed_tag}}}"
if [[ "${PRINT_CONFIG}" == "1" ]]; then
  echo "suite=trace_eval_v1"
  echo "run_tag=${RUN_TAG}"
  echo "seeds=${SEEDS[*]}"
  for i in "${!MODEL_SLUGS[@]}"; do
    echo "model=${MODEL_SLUGS[$i]} path=${MODEL_PATHS[$i]} revision=${MODEL_REVISIONS[$i]} source=${MODEL_SOURCES[$i]} label=${MODEL_LABELS[$i]}"
  done
  for delta in "${DELTAS[@]}"; do echo "delta=${delta}"; done
  exit 0
fi

TMP_ROOT="${TMP_ROOT:-${EVALUATION_ROOT}/.work}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${TMP_ROOT}/${RUN_TAG}}"
SCORE_ROOT="${SCORE_ROOT:-${CAMPAIGN_ROOT}/scoring}"
LOG_ROOT="${LOG_ROOT:-${TMP_ROOT}/logs/${RUN_TAG}}"
RESULTS_ROOT="${RESULTS_ROOT:-${TMP_ROOT}/results}"
EVAL_DEPS_ROOT="${EVAL_DEPS_ROOT:-${TMP_ROOT}/eval_deps}"
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-${CHECKOUT_ROOT}/external/VLMEvalKit}"
MODEL_ROOT="${MODEL_ROOT:-${TMP_ROOT}/models}"
export LMUData="${LMUData:-${TMP_ROOT}/LMUData}"
export HF_HOME="${HF_HOME:-${LMUData}/.hf-cache}"
DATASET_MANIFEST="${DATASET_MANIFEST:-${LMUData}/trace_eval_v1_dataset_manifest.json}"

HOST="${HOST:-127.0.0.1}"
GEN_PORT_START="${GEN_PORT_START:-18000}"
JUDGE_PORT_START="${JUDGE_PORT_START:-18100}"
GPU_GROUPS="${GPU_GROUPS:-0 1 2 3 4 5 6 7}"
CPU_AFFINITY_GROUPS="${CPU_AFFINITY_GROUPS:-}"
EVAL_CPUSET="${EVAL_CPUSET:-none}"

GEN_PARALLELISM_PER_ENDPOINT="${GEN_PARALLELISM_PER_ENDPOINT:-32}"
GEN_PREPARATION_WORKERS="${GEN_PREPARATION_WORKERS:-32}"
GEN_PERSISTENCE_WORKERS="${GEN_PERSISTENCE_WORKERS:-16}"
GEN_FINALIZATION_WORKERS="${GEN_FINALIZATION_WORKERS:-4}"
FINALIZER_JOBS="${FINALIZER_JOBS:-2}"
GEN_QUEUE_CAPACITY="${GEN_QUEUE_CAPACITY:-256}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-32768}"
GEN_MAX_NUM_SEQS="${GEN_MAX_NUM_SEQS:-256}"
GEN_MAX_NUM_BATCHED_TOKENS="${GEN_MAX_NUM_BATCHED_TOKENS:-32768}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.90}"

JUDGE_MODEL="${JUDGE_MODEL:-${MODEL_ROOT}/qwen3-32b-judge}"
JUDGE_REVISION="${JUDGE_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-qwen3-32b-judge}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-128}"
JUDGE_MAX_NUM_BATCHED_TOKENS="${JUDGE_MAX_NUM_BATCHED_TOKENS:-32768}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"

RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_SCORING="${RUN_SCORING:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
MODEL_VERIFY_DEEP="${MODEL_VERIFY_DEEP:-1}"
RUN_LOCAL_ARCHIVE="${RUN_LOCAL_ARCHIVE:-1}"
LOCAL_ARCHIVE_SPOOL_ROOT="${LOCAL_ARCHIVE_SPOOL_ROOT:-${CAMPAIGN_ROOT}/hf_archive}"
if [[ "${RUN_HF_ARCHIVE:-0}" == "1" || -n "${HF_ARCHIVE_REPO_ID:-}" ]]; then
  echo "[fatal] legacy raw archive upload is disabled for trace_eval_v1" >&2
  echo "[fatal] keep raw slices local, then use the sanitized publication workflow" >&2
  exit 2
fi

export PYTHONPATH="${EVAL_DEPS_ROOT}:${CHECKOUT_ROOT}/src:${SCRIPTS_ROOT}:${VLMEVALKIT_ROOT}:${VLMEVALKIT_ROOT}/scripts:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-${TMP_ROOT}/tmp}"
export TOKENIZERS_PARALLELISM=false
export TRACE_EVAL_DATASET_MANIFEST="${DATASET_MANIFEST}"
mkdir -p "${CAMPAIGN_ROOT}" "${SCORE_ROOT}" "${LOG_ROOT}" "${RESULTS_ROOT}" "${TMPDIR}" "${LOCAL_ARCHIVE_SPOOL_ROOT}"

eval_python() {
  if [[ -n "${EVAL_CPUSET}" && "${EVAL_CPUSET}" != "none" ]]; then
    taskset --cpu-list "${EVAL_CPUSET}" "${PYTHON_BIN}" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

dataset_snapshot="$(${PYTHON_BIN} - "${DATASET_MANIFEST}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["view_snapshot_sha256"]["trace_eval_v1"])
PY
)"
dataset_schema="$(${PYTHON_BIN} - "${DATASET_MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["schema_version"])
PY
)"
TRACE_GIT_COMMIT="$(git -C "${CHECKOUT_ROOT}" rev-parse HEAD)"
TRACE_VLMEVALKIT_GIT_COMMIT="$(git -C "${VLMEVALKIT_ROOT}" rev-parse HEAD)"
TRACE_EVAL_EVALUATOR_HASH="$(${PYTHON_BIN} "${SCRIPTS_ROOT}/trace_eval_evaluator_provenance.py" \
  --repo-root "${CHECKOUT_ROOT}" --vlmeval-root "${VLMEVALKIT_ROOT}" --hash-only)"
TRACE_EVAL_CODE_HASH="$(${PYTHON_BIN} "${SCRIPTS_ROOT}/trace_eval_code_provenance.py" \
  --repo-root "${CHECKOUT_ROOT}" --vlmeval-root "${VLMEVALKIT_ROOT}" \
  --evaluator-sha256 "${TRACE_EVAL_EVALUATOR_HASH}")"
TRACE_EVAL_DATASET_SNAPSHOT="${dataset_snapshot}"
TRACE_EVAL_DATASET_REVISION="${dataset_schema}:${dataset_snapshot}"
TRACE_EVAL_MODEL_REVISIONS_JSON="$(${PYTHON_BIN} - "${MODEL_SLUGS[@]}" -- "${MODEL_REVISIONS[@]}" <<'PY'
import json, sys
split = sys.argv.index("--")
print(json.dumps(dict(zip(sys.argv[1:split], sys.argv[split + 1:])), sort_keys=True))
PY
)"
TRACE_EVAL_MODEL_SOURCES_JSON="$(${PYTHON_BIN} - "${MODEL_SLUGS[@]}" -- "${MODEL_SOURCES[@]}" <<'PY'
import json, sys
split = sys.argv.index("--")
print(json.dumps(dict(zip(sys.argv[1:split], sys.argv[split + 1:])), sort_keys=True))
PY
)"
TRACE_EVAL_CAMPAIGN_CONFIG_HASH="$({
  printf '%s\n' \
    "run_tag=${RUN_TAG}" "suite=trace_eval_v1" "suite_sha=$(${PYTHON_BIN} -c 'from trace_eval_suite import load_trace_eval_suite; print(load_trace_eval_suite().manifest_sha256)')" \
    "seeds=${SEEDS[*]}" "models=${MODEL_SLUGS[*]}" "revisions=${MODEL_REVISIONS[*]}" \
    "sources=${MODEL_SOURCES[*]}" \
    "generation=temp0.6,top_p1,top_k-1,max4096" "media=file-url,3136,12845056" \
    "dataset=${TRACE_EVAL_DATASET_REVISION}" "evaluator=${TRACE_EVAL_EVALUATOR_HASH}" \
    "code=${TRACE_EVAL_CODE_HASH}"
} | sha256sum | awk '{print $1}')"
export TRACE_GIT_COMMIT TRACE_VLMEVALKIT_GIT_COMMIT TRACE_EVAL_EVALUATOR_HASH TRACE_EVAL_CODE_HASH
export TRACE_EVAL_DATASET_SNAPSHOT TRACE_EVAL_DATASET_REVISION
export TRACE_EVAL_MODEL_REVISIONS_JSON TRACE_EVAL_MODEL_SOURCES_JSON TRACE_EVAL_CAMPAIGN_CONFIG_HASH

export TRACE_EVAL_DATASET_MANIFEST="${DATASET_MANIFEST}"
if [[ "${RUN_LOCAL_ARCHIVE}" == "1" ]]; then
  export TRACE_EVAL_HF_SPOOL_ROOT="${LOCAL_ARCHIVE_SPOOL_ROOT}"
  export TRACE_EVAL_RUN_ID="${RUN_TAG}"
fi

current_pid_file=""
monitor_pid=""
finalizer_pids=()

stop_pool() {
  if [[ -n "${current_pid_file}" && -f "${current_pid_file}" ]]; then
    PID_FILE="${current_pid_file}" bash "${SCRIPTS_ROOT}/stop_vllm_endpoint_pool.sh" || true
  fi
  current_pid_file=""
}

stop_background_pid() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return 0; }
    sleep 0.5
  done
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  set +e
  for pid in "${finalizer_pids[@]:-}"; do stop_background_pid "${pid}"; done
  stop_pool
  stop_background_pid "${monitor_pid}"
  return "${status}"
}
trap cleanup EXIT

start_pool() {
  local model="$1" served="$2" port="$3" log_dir="$4" max_len="$5" max_seqs="$6" max_tokens="$7" memory="$8"
  local reasoning_parser="${9:-}" chat_template="${10:-}"
  current_pid_file="${log_dir}/pids.txt"
  MODEL_PATH="${model}" SERVED_MODEL_NAME="${served}" HOST="${HOST}" PORT_START="${port}" \
  GPU_GROUPS="${GPU_GROUPS}" CPU_AFFINITY_GROUPS="${CPU_AFFINITY_GROUPS}" \
  GPU_MEMORY_UTILIZATION="${memory}" MAX_MODEL_LEN="${max_len}" MAX_NUM_SEQS="${max_seqs}" \
  MAX_NUM_BATCHED_TOKENS="${max_tokens}" CPU_THREADS_PER_PROCESS=8 \
  ALLOWED_LOCAL_MEDIA_PATH="${LMUData}" PYTHON_BIN="${PYTHON_BIN}" \
  REASONING_PARSER="${reasoning_parser}" CHAT_TEMPLATE="${chat_template}" \
  LOG_DIR="${log_dir}" PID_FILE="${current_pid_file}" \
    bash "${SCRIPTS_ROOT}/start_vllm_endpoint_pool.sh"
}

endpoint_args() {
  local port_start="$1" flag="$2" offset=0
  for _group in ${GPU_GROUPS}; do
    printf '%s\n' "${flag}" "http://${HOST}:$((port_start + offset))/v1"
    offset=$((offset + 1))
  done
}

validate_environment() {
  bash "${SCRIPTS_ROOT}/setup_trace_eval_env.sh" --verify-only
  eval_python "${SCRIPTS_ROOT}/prepare_trace_eval_manifest.py" \
    --lmu-root "${LMUData}" --manifest "${DATASET_MANIFEST}" --verify-only
  local verify_args=()
  for i in "${!MODEL_SLUGS[@]}"; do
    verify_args+=(--entry "${MODEL_SLUGS[$i]}=${MODEL_PATHS[$i]}=${MODEL_REVISIONS[$i]}")
  done
  verify_args+=(--entry "qwen3-32b-judge=${JUDGE_MODEL}=${JUDGE_REVISION}")
  [[ "${MODEL_VERIFY_DEEP}" == "1" ]] && verify_args+=(--deep)
  eval_python "${SCRIPTS_ROOT}/prepare_trace_eval_models.py" verify "${verify_args[@]}"
}

status_args() {
  local slug seed
  printf '%s\n' --campaign-root "${CAMPAIGN_ROOT}" --score-root "${SCORE_ROOT}" \
    --dataset-manifest "${DATASET_MANIFEST}" --archive-spool-root "${LOCAL_ARCHIVE_SPOOL_ROOT}" \
    --vlmeval-root "${VLMEVALKIT_ROOT}"
  for slug in "${MODEL_SLUGS[@]}"; do printf '%s\n' --model-slug "${slug}"; done
  printf '%s\n' --seeds
  for seed in "${SEEDS[@]}"; do printf '%s\n' "${seed}"; done
}

start_monitor() {
  local -a args
  mapfile -t args < <(status_args)
  eval_python "${SCRIPTS_ROOT}/status_trace_eval.py" "${args[@]}" --watch 30 \
    >>"${LOG_ROOT}/status.log" 2>&1 &
  monitor_pid=$!
}

generation_complete() {
  local slug="$1" model="$2" revision="$3" seed="$4"
  eval_python "${SCRIPTS_ROOT}/verify_trace_eval.py" \
    --campaign-root "${CAMPAIGN_ROOT}" --phase generation \
    --model-entry "${slug}=${model}=${revision}" --dataset-manifest "${DATASET_MANIFEST}" \
    --code-hash "${TRACE_EVAL_CODE_HASH}" \
    --evaluator-provenance-sha256 "${TRACE_EVAL_EVALUATOR_HASH}" \
    --vlmeval-root "${VLMEVALKIT_ROOT}" --seeds "${seed}" >/dev/null 2>&1
}

run_generation_pass() {
  local slug="$1" model="$2" seed="$3"
  local -a endpoints
  mapfile -t endpoints < <(endpoint_args "${GEN_PORT_START}" --api-base)
  local run_root="${CAMPAIGN_ROOT}/seed_${seed}/runs"
  mkdir -p "${run_root}"
  eval_python "${SCRIPTS_ROOT}/run_external_benchmark_generation_api_queue.py" \
    --model "${model}" --model-slug "${slug}" --api-model "${slug}" \
    "${endpoints[@]}" --parallelism-per-endpoint "${GEN_PARALLELISM_PER_ENDPOINT}" \
    --preparation-workers "${GEN_PREPARATION_WORKERS}" --queue-capacity "${GEN_QUEUE_CAPACITY}" \
    --persistence-workers "${GEN_PERSISTENCE_WORKERS}" --finalization-workers "${GEN_FINALIZATION_WORKERS}" \
    --media-transport file-url --allowed-local-media-path "${LMUData}" \
    --dataset-manifest "${DATASET_MANIFEST}" --dataset-manifest-view trace_eval_v1 \
    --min-image-pixels 3136 --max-image-pixels 12845056 \
    --run-set trace_eval_v1 --run-root "${run_root}" \
    --temperature 0.6 --top-p 1 --top-k -1 --presence-penalty 0 \
    --repetition-penalty 1 --max-tokens 4096 --seed "${seed}" \
    --compact-prediction-tables --defer-finalization \
    2>&1 | tee "${LOG_ROOT}/generation_${slug}_seed${seed}.log"

  while [[ "${#finalizer_pids[@]}" -ge "${FINALIZER_JOBS}" ]]; do
    local oldest="${finalizer_pids[0]}"
    wait "${oldest}"
    finalizer_pids=("${finalizer_pids[@]:1}")
  done
  (
    set -o pipefail
    eval_python "${SCRIPTS_ROOT}/run_external_benchmark_generation_api_queue.py" \
      --model "${model}" --model-slug "${slug}" --api-model "${slug}" --finalize-only \
      --finalization-workers "${GEN_FINALIZATION_WORKERS}" --preparation-workers "${GEN_PREPARATION_WORKERS}" \
      --queue-capacity "${GEN_QUEUE_CAPACITY}" --persistence-workers "${GEN_PERSISTENCE_WORKERS}" \
      --media-transport file-url --allowed-local-media-path "${LMUData}" \
      --dataset-manifest "${DATASET_MANIFEST}" --dataset-manifest-view trace_eval_v1 \
      --min-image-pixels 3136 --max-image-pixels 12845056 \
      --run-set trace_eval_v1 --run-root "${run_root}" \
      --temperature 0.6 --top-p 1 --top-k -1 --presence-penalty 0 \
      --repetition-penalty 1 --max-tokens 4096 --seed "${seed}" --compact-prediction-tables \
      2>&1 | tee "${LOG_ROOT}/generation_finalize_${slug}_seed${seed}.log"
  ) &
  finalizer_pids+=("$!")
}

run_generation() {
  for i in "${!MODEL_SLUGS[@]}"; do
    local slug="${MODEL_SLUGS[$i]}" model="${MODEL_PATHS[$i]}" revision="${MODEL_REVISIONS[$i]}"
    local complete=1
    for seed in "${SEEDS[@]}"; do generation_complete "${slug}" "${model}" "${revision}" "${seed}" || complete=0; done
    [[ "${complete}" == "0" ]] || { echo "[generation:skip-model] ${slug}"; continue; }
    start_pool "${model}" "${slug}" "${GEN_PORT_START}" "${LOG_ROOT}/vllm_generation_${slug}" \
      "${GEN_MAX_MODEL_LEN}" "${GEN_MAX_NUM_SEQS}" "${GEN_MAX_NUM_BATCHED_TOKENS}" "${GEN_GPU_MEMORY_UTILIZATION}"
    for seed in "${SEEDS[@]}"; do
      generation_complete "${slug}" "${model}" "${revision}" "${seed}" || run_generation_pass "${slug}" "${model}" "${seed}"
    done
    stop_pool
  done
  local failed=0
  for pid in "${finalizer_pids[@]}"; do wait "${pid}" || failed=1; done
  finalizer_pids=()
  [[ "${failed}" == "0" ]] || return 1
  for i in "${!MODEL_SLUGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      generation_complete "${MODEL_SLUGS[$i]}" "${MODEL_PATHS[$i]}" "${MODEL_REVISIONS[$i]}" "${seed}"
    done
  done
}

run_scoring() {
  local -a judge_endpoints campaigns
  mapfile -t judge_endpoints < <(endpoint_args "${JUDGE_PORT_START}" --judge-endpoint)
  for i in "${!MODEL_SLUGS[@]}"; do campaigns+=(--campaign "${MODEL_PATHS[$i]}" "${MODEL_SLUGS[$i]}" "${CAMPAIGN_ROOT}"); done
  start_pool "${JUDGE_MODEL}" "${JUDGE_SERVED_NAME}" "${JUDGE_PORT_START}" "${LOG_ROOT}/vllm_judge" \
    "${JUDGE_MAX_MODEL_LEN}" "${JUDGE_MAX_NUM_SEQS}" "${JUDGE_MAX_NUM_BATCHED_TOKENS}" "${JUDGE_GPU_MEMORY_UTILIZATION}" \
    qwen3 "${CHECKOUT_ROOT}/rlvr/examples/prompts/chat_template_no_think.jinja"
  for seed in "${SEEDS[@]}"; do
    eval_python "${SCRIPTS_ROOT}/run_trace_eval_score_campaign.py" \
      --seed "${seed}" --shared-seed-root --resume --emit-archive \
      "${campaigns[@]}" --score-root "${SCORE_ROOT}" --dataset-manifest "${DATASET_MANIFEST}" \
      --python "${PYTHON_BIN}" --eval-deps "${EVAL_DEPS_ROOT}" \
      --vlmeval-root "${VLMEVALKIT_ROOT}" --lmu-data "${LMUData}" --hf-home "${HF_HOME}" \
      --judge-model "${JUDGE_MODEL}" --judge-api-model "${JUDGE_SERVED_NAME}" \
      "${judge_endpoints[@]}" --official-workers 8 --direct-workers 8 --mme-workers 3 \
      --eval-nproc 16 --judge-api-parallelism 64 --judge-api-batch-size 64 \
      2>&1 | tee "${LOG_ROOT}/scoring_seed${seed}.log"
  done
  stop_pool
}

run_summary() {
  local -a args=(--score-root-base "${SCORE_ROOT}" --suite trace_eval_v1 --seeds "${SEEDS[@]}")
  for i in "${!MODEL_SLUGS[@]}"; do args+=(--model-entry "${MODEL_SLUGS[$i]}=${MODEL_LABELS[$i]}"); done
  for delta in "${DELTAS[@]}"; do args+=(--delta "${delta}"); done
  args+=(--excel "${RESULTS_ROOT}/${RUN_TAG}_results.xlsx" --markdown "${RESULTS_ROOT}/${RUN_TAG}_results.md")
  eval_python "${SCRIPTS_ROOT}/summarize_trace_eval.py" "${args[@]}" \
    2>&1 | tee "${LOG_ROOT}/summary.log"
}

verify_local_archive_coverage() {
  [[ "${RUN_LOCAL_ARCHIVE}" == "1" ]] || return 0
  local -a args=(--spool-root "${LOCAL_ARCHIVE_SPOOL_ROOT}" coverage --expect-run-id "${RUN_TAG}" \
    --expect-campaign-config-hash "${TRACE_EVAL_CAMPAIGN_CONFIG_HASH}" \
    --expect-dataset-revision "${TRACE_EVAL_DATASET_REVISION}")
  local slug seed key
  for slug in "${MODEL_SLUGS[@]}"; do args+=(--expect-model-slug "${slug}"); done
  for seed in "${SEEDS[@]}"; do args+=(--expect-seed "${seed}"); done
  while read -r key; do args+=(--expect-benchmark "${key}"); done < <(
    "${PYTHON_BIN}" -c 'from trace_eval_suite import load_trace_eval_suite; print(*load_trace_eval_suite().benchmark_keys, sep="\n")'
  )
  eval_python "${SCRIPTS_ROOT}/trace_eval_hf_archive.py" "${args[@]}"
}

echo "[trace-eval] tag=${RUN_TAG} models=${#MODEL_SLUGS[@]} seeds=${SEEDS[*]} rows_per_model_seed=32805"
[[ "${FINALIZER_JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "[fatal] FINALIZER_JOBS must be positive" >&2; exit 1; }
validate_environment
start_monitor
[[ "${RUN_GENERATION}" == "1" ]] && run_generation
[[ "${RUN_SCORING}" == "1" ]] && run_scoring
[[ "${RUN_SUMMARY}" == "1" ]] && run_summary
verify_local_archive_coverage
mapfile -t final_status_args < <(status_args)
eval_python "${SCRIPTS_ROOT}/status_trace_eval.py" "${final_status_args[@]}" --fail-if-incomplete
echo "[trace-eval:done] ${RESULTS_ROOT}/${RUN_TAG}_results.xlsx"
