#!/usr/bin/env bash
# experiments/certificates/jobs/ext_eval.sh MODEL_OR_DIR SERVED_NAME [--seed S] [--subset q25|full] [--smoke]
#                                          [--judge-only] [--dry-run]
#
# DESIGN 6 "External suite", 7.4, 7.5 W5: TRACE's 24-benchmark trace_eval_v1 evaluation of one model at one decoding
# seed, written as ONE use-gpu task line. It runs the protocol's own components from the separate clone of the rlvr
# branch ($TRC_EXT_ROOT/trace-rlvr @ af32fe0), with the arguments run_trace_eval.sh passes them:
#   1. its vLLM endpoint pool (start_vllm_endpoint_pool.sh: one server per GPU, TP=1, max-model-len 32768),
#   2. its generation queue (its prompts; temperature 0.6, top_p 1, top_k -1, max_tokens 4096, file-url media,
#      3136..12845056 px, seed S), then its CPU finalization (prediction workbooks, local archive slices),
#   3. TRACE's deterministic answer extractor over the predictions (extraction-failure and certificate-JSON rates),
#   4. its score campaign (deterministic routes + Qwen3-32B judge routes) against a judge pool on this node's GPUs
#      (Qwen3-32B@9216db57, no-think chat template, qwen3 reasoning parser, max-model-len 8192),
#   5. per-benchmark, category and macro summaries split natural vs chart/diagram/synthetic (DESIGN 6) and a
#      completeness check (plus the protocol's verify_trace_eval.py + summarize_trace_eval.py for full runs).
#
#   source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT" && \
#     bash experiments/certificates/jobs/ext_eval.sh Qwen/Qwen2.5-VL-3B-Instruct qwen2.5-vl-3b-instruct --seed 42
#
# MODEL_OR_DIR: a Hugging Face id pinned in trace-tool's scripts/_model_env.sh (served from .work/models/<SERVED>:
#   symlinks into the HF cache plus the protocol's provenance marker), or a local merged checkpoint directory
#   (protocol register-local marker written into it; revision sha256set:<content>, source trace-cert/<SERVED>@<hex>).
# SERVED_NAME: the protocol's model slug and served model name ([a-z0-9][a-z0-9._-]*).
# --subset full (default): 32,805 rows. q25: the frozen 25% stratified subset (8,205 rows; blake3(benchmark, row_id)
#   rank within each benchmark, DESIGN 6; row list $TRC_EXT_ROOT/subsets/trace_eval_v1_q25.rows.json, its sha256 is
#   in plan/summary). --smoke: TRC_EXT_SMOKE_PER_BENCH (default 4) rows per benchmark (the first q25 ranks).
# --judge-only: scoring half only (generation must be finalized); the follow-up of a pending-judge marker.
# --dry-run: CPU preflight (environment, data manifest, model marker, subset, plan); touches no GPU.
#
# Outputs: $TRC_EXT_ROOT/runs/<SERVED>/<full|q25|smoke>/
#   campaign/seed_<S>/runs/<key>/<served>/<run>/   row results, predictions.jsonl, <alias>_predictions.xlsx
#   scoring/seed_<S>/benchmark/<key>/<served>/<run>/scores.json   protocol score campaign; hf_archive/ local archive
#   plan.seed<S>.json  extract.seed<S>.json  summary.seed<S>.{json,md}  validation.seed<S>.json  timing.seed<S>.json
#   PENDING_JUDGE.seed<S>  (this node could not host the judge; holds the judge-only task line)  logs/
# Judge placement: smallest tensor-parallel group of 1/2/4/8 cards holding the 61 GiB bf16 weights with >= 16 GiB
#   headroom (else >= 6 GiB with max-num-batched-tokens 8192), one judge endpoint per group. A node that cannot host
#   it finalizes generation, writes PENDING_JUDGE.seed<S>, raises "FLAG judge_pending", and exits 0.
#
# Exit: 0 done (or generation done + judge pending); 1 incomplete (rerun resumes); 2 usage; 3 GPUs busy or the run is
#       owned by another live job; 4 configuration mismatch; 5 vLLM did not start; 129/130/143 signalled.
# Knobs: TRC_EXT_GEN_PAR (requests in flight per endpoint; protocol 32; default 32/64/128 for cards <40/<80/>=80 GB),
#   TRC_EXT_VERIFY_DATA (1: rehash the media
#   manifest first, as run_trace_eval.sh does), TRC_EXT_JUDGE_DEEP (0: marker check; 1: rehash the 65 GB judge),
#   TRC_EXT_SMOKE_PER_BENCH (4), TRC_EXT_READY_TIMEOUT (1800 s), TRC_EXT_LOCAL_ARCHIVE (1), TRC_EXT_JUDGE_MODEL /
#   TRC_EXT_JUDGE_REVISION / TRC_EXT_JUDGE_SERVED (stand-in judge for plumbing tests only; never for results),
#   TRC_EXT_RUNS (output root, default $TRC_EXT_ROOT/runs; e.g. a separate root for hardware timing probes),
#   TRC_EXT_NO_JUDGE (1: generation only, write the pending-judge marker even on a judge-capable node),
#   TRC_EXT_GEN_UTIL (generation vLLM gpu-memory-utilization; default 0.90, 0.80 on cards < 40 GB).
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# bash reads a script while running it, so an edit of this file would corrupt a job that is hours in: run a private
# copy (the jobs directory stays the one of the checkout)
if [ -z "${TRC_EXT_SNAPSHOT:-}" ]; then
  _snap=$(mktemp "${TMPDIR:-/tmp}/ext_eval.XXXXXX") && cp "$SELF" "$_snap" \
    && TRC_EXT_SNAPSHOT=$_snap TRC_EXT_JOBS_DIR=$(dirname "$SELF") exec bash "$_snap" "$@"
  echo "FATAL: could not snapshot $SELF" >&2; exit 4
fi
trap 'rm -f "$TRC_EXT_SNAPSHOT"' EXIT
# shellcheck source=common.sh
source "$TRC_EXT_JOBS_DIR/common.sh"

usage() { echo "usage: ext_eval.sh MODEL_OR_DIR SERVED_NAME [--seed S] [--subset q25|full] [--smoke] [--judge-only] [--dry-run]" >&2; exit 2; }
MODEL=""; SERVED=""; SEED=42; SUBSET=full; SMOKE=0; JUDGE_ONLY=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) [ $# -ge 2 ] || usage; SEED=$2; shift 2 ;;
    --subset) [ $# -ge 2 ] || usage; SUBSET=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --judge-only) JUDGE_ONLY=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) if [ -z "$MODEL" ]; then MODEL=$1; elif [ -z "$SERVED" ]; then SERVED=$1; else usage; fi; shift ;;
  esac
done
[ -n "$MODEL" ] && [ -n "$SERVED" ] || usage
[[ $SEED =~ ^[0-9]+$ ]] || { echo "invalid seed $SEED" >&2; usage; }
case "$SUBSET" in full|q25) ;; *) echo "invalid subset $SUBSET" >&2; usage ;; esac
[ "$SMOKE" = 1 ] && SUBSET=smoke
[[ $SERVED =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "SERVED_NAME must match [a-z0-9][a-z0-9._-]*: $SERVED" >&2; exit 2; }

trc_require_env
trc_node_allowed || { echo "FATAL: config error: $(hostname -s) is excluded by fleet policy" >&2; exit 4; }

# ============================================================================== protocol paths (run_trace_eval.sh)
TRC_EXT_ROOT=${TRC_EXT_ROOT:-$TRACE_CERT_RUNS/ext_eval}
CLONE=${TRC_EXT_CLONE:-$TRC_EXT_ROOT/trace-rlvr}
EXT_COMMIT=${TRC_EXT_COMMIT:-af32fe0e668eb878e909daef7322ac1c4ae20f64}
PY=${TRC_EXT_PYTHON:-}
EVALUATION_ROOT=$CLONE/rlvr/evaluation
SCRIPTS_ROOT=$EVALUATION_ROOT/scripts
TMP_ROOT=$EVALUATION_ROOT/.work
VLMEVALKIT_ROOT=$CLONE/external/VLMEvalKit
EVAL_DEPS_ROOT=$TMP_ROOT/eval_deps
MODEL_ROOT=$TMP_ROOT/models
export LMUData=$TMP_ROOT/LMUData
DATASET_MANIFEST=$LMUData/trace_eval_v1_dataset_manifest.json
JUDGE_MODEL=${TRC_EXT_JUDGE_MODEL:-$MODEL_ROOT/qwen3-32b-judge}
JUDGE_REVISION=${TRC_EXT_JUDGE_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}
JUDGE_SERVED=${TRC_EXT_JUDGE_SERVED:-qwen3-32b-judge}
LIB=$TRC_JOBS_DIR/ext_eval_lib.py
for f in "$PY" "$SCRIPTS_ROOT/run_external_benchmark_generation_api_queue.py" "$DATASET_MANIFEST" "$LIB"; do
  [ -n "$f" ] && [ -e "$f" ] || { echo "FATAL: config error: missing ${f:-TRC_EXT_PYTHON} (set up per ext_eval notes)" >&2; exit 4; }
done
got=$(git -C "$CLONE" rev-parse HEAD 2>/dev/null)
[ "$got" = "$EXT_COMMIT" ] || { echo "FATAL: config error: $CLONE is at ${got:-?}, expected $EXT_COMMIT" >&2; exit 4; }
export HF_HOME TOKENIZERS_PARALLELISM=false
export TRACE_EVAL_DATASET_MANIFEST=$DATASET_MANIFEST
export PYTHONPATH="$EVAL_DEPS_ROOT:$CLONE/src:$SCRIPTS_ROOT:$VLMEVALKIT_ROOT:$VLMEVALKIT_ROOT/scripts"
export TMPDIR=${TMPDIR:-$TMP_ROOT/tmp}
mkdir -p "$TMPDIR"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # everything is local; the score campaign drops them as well

ROOT=${TRC_EXT_RUNS:-$TRC_EXT_ROOT/runs}/$SERVED/$SUBSET
CAMPAIGN_ROOT=$ROOT/campaign
SCORE_ROOT=$ROOT/scoring
RUN_ROOT=$CAMPAIGN_ROOT/seed_$SEED/runs
PLAN=$ROOT/plan.seed$SEED.json
MARKER=$ROOT/PENDING_JUDGE.seed$SEED
JOBTAG=${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}.$(date +%m%d_%H%M%S)
STATE=$ROOT/logs/job.$JOBTAG
TRC_TAG="ext:$SERVED:$SUBSET:s$SEED"
mkdir -p "$STATE" "$RUN_ROOT" "$SCORE_ROOT"
trc_log_open "$ROOT/logs/ext_eval.seed$SEED.$JOBTAG.log"
trc_say "ext_eval.sh $MODEL $SERVED seed=$SEED subset=$SUBSET judge_only=$JUDGE_ONLY dry=$DRY clone=$CLONE@${EXT_COMMIT:0:7}"
trc_say "host $(hostname -s) job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none} cluster $(trc_cluster) N(dispatcher)=${N:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

declare -A T=()
T[start]=$(date +%s)
POOL_PID_FILE=""
QUEUE_PID=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [ -n "$QUEUE_PID" ]; then trc_stop_tree "$QUEUE_PID" "generation queue" "${TRC_STOP_WAIT:-30}"; fi
  if declare -F pool_stop >/dev/null; then pool_stop; fi
  trc_heartbeat_stop
  trc_claim_release
  trc_lease_release
  write_timing "$rc"
  trc_say "exit rc=$rc"
  rm -f "$TRC_EXT_SNAPSHOT"
  exit "$rc"
}
trap on_exit EXIT
trap 'TRC_STOP_WAIT=20; trc_say "signal TERM"; exit 143' TERM
trap 'TRC_STOP_WAIT=20; trc_say "signal INT"; exit 130' INT
trap 'TRC_STOP_WAIT=20; trc_say "signal HUP"; exit 129' HUP

write_timing() {  # RC: timing.seed<S>.json (phase wall-clock for the cost model); merged with earlier attempts
  local rc=$1 k body="" sep=""
  for k in "${!T[@]}"; do body+="$sep\"$k\": ${T[$k]}"; sep=", "; done
  "$PY" -I - "$ROOT/timing.seed$SEED.json" "$JOBTAG" "$rc" "$(hostname -s)" "${TRC_GPU_SUMMARY:-}" "{$body}" <<'PYEOF' 2>/dev/null || true
import json, os, sys
path, tag, rc, host, gpus, phases = sys.argv[1:7]
try:
    data = json.load(open(path))
except Exception:
    data = {"attempts": []}
data["attempts"] = [a for a in data["attempts"] if a.get("job") != tag] + [
    {"job": tag, "rc": int(rc), "host": host, "gpus": gpus, "epoch": json.loads(phases)}]
tmp = path + ".tmp.%d" % os.getpid()
json.dump(data, open(tmp, "w"), indent=1)
os.replace(tmp, path)
PYEOF
}
lib() { "$PY" "$LIB" "$@"; }

# ============================================================================== model
if [ -d "$MODEL" ]; then
  MODEL_KIND=local
  info=$(lib model-local --path "$MODEL" --slug "$SERVED" 2>"$STATE/model.err") \
    || { cat "$STATE/model.err"; trc_say "FATAL: config error: cannot register local checkpoint $MODEL"; exit 4; }
else
  MODEL_KIND=hf
  trc_model_env "$MODEL"
  [ -n "$TRC_REVISION" ] || { trc_say "FATAL: config error: $MODEL has no pinned revision in trace-tool's _model_env.sh"; exit 4; }
  info=$(lib model-link --repo "$MODEL" --revision "$TRC_REVISION" --slug "$SERVED" --out "$MODEL_ROOT/$SERVED" \
           --hf-home "$HF_HOME" 2>"$STATE/model.err") \
    || { cat "$STATE/model.err"; trc_say "FATAL: config error: cannot materialize $MODEL@$TRC_REVISION"; exit 4; }
fi
read -r MODEL_PATH REVISION SOURCE < <("$PY" -I -c 'import json,sys; d=json.loads(sys.argv[1]); print(d["path"], d["revision"], d["source"])' "$info")
trc_say "model ($MODEL_KIND): $MODEL_PATH revision $REVISION source $SOURCE"

# ============================================================================== subset + plan
ROWS=""
case "$SUBSET" in
  q25) ROWS=$ROOT/rows.q25.json; FROZEN=$TRC_EXT_ROOT/subsets/trace_eval_v1_q25.rows.json; SARGS=(--mode q25) ;;
  smoke) K=${TRC_EXT_SMOKE_PER_BENCH:-4}; ROWS=$ROOT/rows.smoke$K.json; FROZEN=$TRC_EXT_ROOT/subsets/trace_eval_v1_smoke$K.rows.json
         SARGS=(--mode smoke --per-bench "$K") ;;
esac
if [ -n "$ROWS" ]; then
  out=$(lib subset --manifest "$DATASET_MANIFEST" "${SARGS[@]}" --out "$ROWS" --frozen "$FROZEN" 2>&1) \
    || { echo "$out"; trc_say "FATAL: config error: subset row list failed"; exit 4; }
  trc_say "subset $SUBSET: $out"
fi
lib plan --model "$MODEL" --slug "$SERVED" --model-path "$MODEL_PATH" --revision "$REVISION" --source "$SOURCE" \
  --seed "$SEED" --subset "$SUBSET" ${ROWS:+--rows "$ROWS"} --out "$PLAN" >"$STATE/plan.out" 2>&1 \
  || { cat "$STATE/plan.out"; trc_say "FATAL: config error: plan failed"; exit 4; }
trc_say "plan: $(cat "$STATE/plan.out") -> $PLAN"

# ============================================================================== environment + provenance (run_trace_eval.sh)
if ! PYTHON_BIN=$PY bash "$SCRIPTS_ROOT/setup_trace_eval_env.sh" --verify-only >"$STATE/env_verify.log" 2>&1; then
  cat "$STATE/env_verify.log"; trc_say "FATAL: config error: setup_trace_eval_env.sh --verify-only failed"; exit 4
fi
# af32fe0 calls batched_vlmevalkit_qwen3vl._parse_charxiv_judge, which the public release does not define: CharXiv
# cannot be scored without the post-release fix (applied to the generated VLMEvalKit copy at setup)
if ! grep -q '^def _parse_charxiv_judge' "$VLMEVALKIT_ROOT/scripts/batched_vlmevalkit_qwen3vl.py"; then
  trc_say "FATAL: config error: the CharXiv post-release fix is missing from $VLMEVALKIT_ROOT/scripts (rerun the clone's post-release fixes after setup_trace_eval_env.sh)"; exit 4
fi
# fix-2: an exception of MME-Reasoning's official per-row function is scored incorrect ('Failed to evaluate', upstream
# VLMEvalKit semantics) instead of failing the run; every model is scored with it (README, external evaluation)
if ! grep -q 'trace-cert post-release fix-2' "$SCRIPTS_ROOT/run_mme_reasoning_eval.py"; then
  trc_say "FATAL: config error: the MME-Reasoning post-release fix-2 is missing from $SCRIPTS_ROOT/run_mme_reasoning_eval.py (rerun the clone's post-release fixes)"; exit 4
fi
# fix-3: a row whose judge output is still unusable after the official 5 attempts (MathVerse score judgement,
# MME-Reasoning answer extraction) is scored incorrect (upstream VLMEvalKit semantics) instead of failing the run; one
# fix in both scorer scripts, so both must carry it (README, external evaluation)
for f in run_external_benchmark_score_queue.py run_mme_reasoning_eval.py; do
  if ! grep -q 'trace-cert post-release fix-3' "$SCRIPTS_ROOT/$f"; then
    trc_say "FATAL: config error: the post-release fix-3 is missing from $SCRIPTS_ROOT/$f (rerun the clone's post-release fixes)"; exit 4
  fi
done
if [ "${TRC_EXT_VERIFY_DATA:-1}" = 1 ]; then
  t0=$(date +%s)
  if ! "$PY" "$SCRIPTS_ROOT/prepare_trace_eval_manifest.py" --lmu-root "$LMUData" --manifest "$DATASET_MANIFEST" \
       --verify-only >"$STATE/data_verify.log" 2>&1; then
    tail -n 30 "$STATE/data_verify.log"; trc_say "FATAL: config error: prepared datasets do not verify"; exit 4
  fi
  trc_say "data manifest verified (all media rehashed) in $(( $(date +%s) - t0 ))s"
fi
if ! "$PY" "$SCRIPTS_ROOT/prepare_trace_eval_models.py" verify --entry "$SERVED=$MODEL_PATH=$REVISION" --deep \
     >"$STATE/model_verify.log" 2>&1; then
  cat "$STATE/model_verify.log"; trc_say "FATAL: config error: model verification failed"; exit 4
fi
dataset_snapshot=$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["view_snapshot_sha256"]["trace_eval_v1"])' "$DATASET_MANIFEST")
dataset_schema=$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["schema_version"])' "$DATASET_MANIFEST")
seed_tag=$SEED
case "$SUBSET" in full) RUN_TAG=trace_eval_v1_temp06_seed$seed_tag ;; *) RUN_TAG=trace_eval_v1_${SUBSET}_temp06_seed$seed_tag ;; esac
TRACE_GIT_COMMIT=$(git -C "$CLONE" rev-parse HEAD)
TRACE_VLMEVALKIT_GIT_COMMIT=$(git -C "$VLMEVALKIT_ROOT" rev-parse HEAD)
TRACE_EVAL_EVALUATOR_HASH=$("$PY" "$SCRIPTS_ROOT/trace_eval_evaluator_provenance.py" --repo-root "$CLONE" \
  --vlmeval-root "$VLMEVALKIT_ROOT" --hash-only)
TRACE_EVAL_CODE_HASH=$("$PY" "$SCRIPTS_ROOT/trace_eval_code_provenance.py" --repo-root "$CLONE" \
  --vlmeval-root "$VLMEVALKIT_ROOT" --evaluator-sha256 "$TRACE_EVAL_EVALUATOR_HASH")
TRACE_EVAL_DATASET_SNAPSHOT=$dataset_snapshot
TRACE_EVAL_DATASET_REVISION="$dataset_schema:$dataset_snapshot"
TRACE_EVAL_MODEL_REVISIONS_JSON=$("$PY" -I -c 'import json,sys; print(json.dumps({sys.argv[1]: sys.argv[2]}, sort_keys=True))' "$SERVED" "$REVISION")
TRACE_EVAL_MODEL_SOURCES_JSON=$("$PY" -I -c 'import json,sys; print(json.dumps({sys.argv[1]: sys.argv[2]}, sort_keys=True))' "$SERVED" "$SOURCE")
suite_sha=$(cd "$SCRIPTS_ROOT" && "$PY" -c 'from trace_eval_suite import load_trace_eval_suite; print(load_trace_eval_suite().manifest_sha256)')
TRACE_EVAL_CAMPAIGN_CONFIG_HASH=$({
  printf '%s\n' \
    "run_tag=$RUN_TAG" "suite=trace_eval_v1" "suite_sha=$suite_sha" \
    "seeds=$SEED" "models=$SERVED" "revisions=$REVISION" "sources=$SOURCE" \
    "generation=temp0.6,top_p1,top_k-1,max4096" "media=file-url,3136,12845056" \
    "dataset=$TRACE_EVAL_DATASET_REVISION" "evaluator=$TRACE_EVAL_EVALUATOR_HASH" "code=$TRACE_EVAL_CODE_HASH"
} | sha256sum | awk '{print $1}')
export TRACE_GIT_COMMIT TRACE_VLMEVALKIT_GIT_COMMIT TRACE_EVAL_EVALUATOR_HASH TRACE_EVAL_CODE_HASH
export TRACE_EVAL_DATASET_SNAPSHOT TRACE_EVAL_DATASET_REVISION
export TRACE_EVAL_MODEL_REVISIONS_JSON TRACE_EVAL_MODEL_SOURCES_JSON TRACE_EVAL_CAMPAIGN_CONFIG_HASH
if [ "${TRC_EXT_LOCAL_ARCHIVE:-1}" = 1 ]; then
  export TRACE_EVAL_HF_SPOOL_ROOT=$ROOT/hf_archive TRACE_EVAL_RUN_ID=$RUN_TAG
  mkdir -p "$TRACE_EVAL_HF_SPOOL_ROOT"
fi
trc_say "provenance: run_tag $RUN_TAG code ${TRACE_EVAL_CODE_HASH:0:12} evaluator ${TRACE_EVAL_EVALUATOR_HASH:0:12} dataset ${dataset_snapshot:0:12} campaign ${TRACE_EVAL_CAMPAIGN_CONFIG_HASH:0:12}"

gen_complete() { lib validate --plan "$PLAN" --run-root "$RUN_ROOT" --phase generation --out "$ROOT/validation.generation.seed$SEED.json" >"$STATE/gen_validate.txt" 2>&1; }
GEN_DONE=0
if gen_complete; then GEN_DONE=1; fi
trc_say "generation: $(tail -n 1 "$STATE/gen_validate.txt")"
if [ "$JUDGE_ONLY" = 1 ] && [ "$GEN_DONE" = 0 ]; then
  cat "$STATE/gen_validate.txt"; trc_say "FATAL: --judge-only but the generation is not finalized; run without --judge-only"; exit 1
fi
if [ "$DRY" = 1 ]; then trc_say "dry run: no GPU touched (generation_done=$GEN_DONE)"; exit 0; fi

trc_lease_take "$ROOT/.owner.seed$SEED" || exit 3
trc_heartbeat_start

# ============================================================================== GPUs, ports, endpoint pools
gpu_setup() {
  trc_gpu_discover || exit 4
  while read -r line; do trc_say "  $line"; done < <(trc_gpu_describe)
  TRC_GPU_SUMMARY="${TRC_N}x ${TRC_GPU_NAME[0]} $((TRC_MIN_MIB / 1024))GiB"
  [ -z "${N:-}" ] || [ "$N" = "$TRC_N" ] || trc_say "note: dispatcher N=$N, this step sees $TRC_N GPUs (using $TRC_N)"
  if busy=$(trc_gpu_busy); then trc_say "GPUs busy, not starting: $busy"; exit 3; fi
  if busy=$(trc_foreign_claimed); then trc_say "GPUs claimed, not starting: $busy"; exit 3; fi
  TRC_CLAIM_WHO="ext_eval.sh $MODEL $SERVED seed $SEED $SUBSET$([ "$JUDGE_ONLY" = 1 ] && echo ' judge-only')"
  TRC_CLAIM_RUN=$ROOT; TRC_CLAIM_TASK="trc:ext_eval:$SERVED:$SUBSET:$SEED"
  trc_claim_take >"$STATE/claims.txt" 2>&1; local rc=$?
  trc_say "$(tr '\n' ' ' <"$STATE/claims.txt")"
  [ "$rc" -eq 0 ] || exit 3
  trc_heartbeat_stop; trc_heartbeat_start
  sleep 10
  if busy=$(trc_gpu_busy); then trc_say "GPUs became busy after claiming, releasing: $busy"; exit 3; fi
  IFS=, read -ra DEVS <<<"$CUDA_VISIBLE_DEVICES"
}
# pick_ports is check-then-use. Allocations that share a node (notch366 holds several 1-GPU jobs) can start pools in the
# same second and pick the same ports; on 2026-10-24 two pools died with EADDRINUSE on internal port 29000. pool_start
# therefore holds a node-local lock from picking until its pool is ready or has failed (the servers do not inherit it).
ports_lock() {
  exec 8>"/tmp/trc_ext_pool_ports.$(id -u).lock"
  flock -w "${TRC_EXT_PORT_LOCK_WAIT:-2400}" 8 || trc_say "port lock: still held after ${TRC_EXT_PORT_LOCK_WAIT:-2400}s, picking anyway"
}
ports_unlock() { flock -u 8 2>/dev/null; exec 8>&-; }
pick_ports() {  # COUNT STEP START -> first port P >= START with P + i*STEP free for i < COUNT
  local count=$1 step=$2 p=$3 i ok
  for _ in $(seq 1 400); do
    ok=1
    for i in $(seq 0 $((count - 1))); do trc_port_in_use $((p + i * step)) && { ok=0; break; }; done
    [ "$ok" = 1 ] && { echo "$p"; return 0; }
    p=$((p + 1))
  done
  return 1
}
# pool_start NAME PORT0 MODEL SERVED GROUPS MAXLEN MAXSEQS MAXBATCHED [REASONING_PARSER CHAT_TEMPLATE] (POOL_UTIL, default
# 0.90 as the protocol): the protocol's
# start_vllm_endpoint_pool.sh exactly as run_trace_eval.sh's start_pool calls it; sets POOL_ENDPOINTS[]. Ports are the
# protocol's (generation 18000+, judge 18100+, internal 29000+i*100) unless taken. The judge URLs enter the score
# campaign contract, so stable ports let an interrupted scoring resume.
pool_start() {
  local name=$1 port0=$2 model=$3 served=$4 groups=$5 maxlen=$6 maxseqs=$7 maxbatched=$8 parser=${9:-} template=${10:-}
  local n port_base vllm_base log_dir rc
  read -ra _g <<<"$groups"; n=${#_g[@]}
  ports_lock
  port_base=$(pick_ports "$n" 1 "$port0") || { ports_unlock; trc_say "no free ports"; return 5; }
  vllm_base=$(pick_ports "$n" 100 29000) || { ports_unlock; trc_say "no free internal ports"; return 5; }
  log_dir=$ROOT/logs/vllm_${name}.seed$SEED.$JOBTAG
  POOL_PID_FILE=$log_dir/pids.txt
  trc_say "vLLM pool $name: $model as $served on groups [$groups] ports $port_base.. (internal $vllm_base) max_len $maxlen seqs $maxseqs batched $maxbatched util ${POOL_UTIL:-0.90}${parser:+ parser $parser}"
  MODEL_PATH="$model" SERVED_MODEL_NAME="$served" HOST=127.0.0.1 PORT_START="$port_base" VLLM_PORT_BASE="$vllm_base" \
  GPU_GROUPS="$groups" CPU_AFFINITY_GROUPS="" GPU_MEMORY_UTILIZATION="${POOL_UTIL:-0.90}" MAX_MODEL_LEN="$maxlen" MAX_NUM_SEQS="$maxseqs" \
  MAX_NUM_BATCHED_TOKENS="$maxbatched" CPU_THREADS_PER_PROCESS=8 ALLOWED_LOCAL_MEDIA_PATH="$LMUData" PYTHON_BIN="$PY" \
  REASONING_PARSER="$parser" CHAT_TEMPLATE="$template" LOG_DIR="$log_dir" PID_FILE="$POOL_PID_FILE" \
  WAIT_READY=0 bash "$SCRIPTS_ROOT/start_vllm_endpoint_pool.sh" >"$log_dir.start.log" 2>&1 </dev/null 3>&- 8>&-
  rc=$?
  POOL_ENDPOINTS=()
  for i in $(seq 0 $((n - 1))); do POOL_ENDPOINTS+=("http://127.0.0.1:$((port_base + i))/v1"); done
  # the protocol script's own wait (WAIT_READY=1) polls until its timeout even after a server died: wait here and
  # fail as soon as one exits
  local deadline=$((SECONDS + ${TRC_EXT_READY_TIMEOUT:-1800})) pid port rest e
  while [ "$rc" -eq 0 ]; do
    local ready=0
    for e in "${POOL_ENDPOINTS[@]}"; do curl -fsS --max-time 5 "$e/models" >/dev/null 2>&1 && ready=$((ready + 1)); done
    [ "$ready" -eq "$n" ] && break
    while read -r pid port rest; do
      [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { trc_say "vLLM server on port $port exited during startup"; rc=5; }
    done <"$POOL_PID_FILE"
    [ "$SECONDS" -ge "$deadline" ] && { trc_say "vLLM pool not ready after ${TRC_EXT_READY_TIMEOUT:-1800}s"; rc=5; }
    [ "$rc" -eq 0 ] && sleep 5
  done
  ports_unlock
  if [ "$rc" -ne 0 ]; then
    trc_say "vLLM pool $name did not start (rc $rc); last lines:"; tail -n 5 "$log_dir.start.log"
    for f in "$log_dir"/endpoint_*.log; do [ -f "$f" ] && { echo "== $f"; grep -E "Error|error" "$f" | tail -n 5; tail -n 15 "$f"; }; done
    pool_stop
    return 5
  fi
  trc_say "vLLM pool $name ready: ${POOL_ENDPOINTS[*]}"
}
pool_stop() {
  if [ -n "$POOL_PID_FILE" ] && [ -f "$POOL_PID_FILE" ]; then
    PID_FILE="$POOL_PID_FILE" STOP_GRACE_SEC=15 bash "$SCRIPTS_ROOT/stop_vllm_endpoint_pool.sh" >/dev/null 2>&1 || true
    local w; for w in $(seq 1 30); do pool_alive || break; sleep 2; done
  fi
  POOL_PID_FILE=""
}
pool_alive() {  # any server of the current pool still alive
  local pid rest
  [ -n "$POOL_PID_FILE" ] && [ -f "$POOL_PID_FILE" ] || return 1
  while read -r pid rest; do [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 0; done <"$POOL_PID_FILE"
  return 1
}
pool_healthy_count() {
  local e c=0
  for e in "${POOL_ENDPOINTS[@]}"; do curl -sf --max-time 10 "$e/models" >/dev/null 2>&1 && c=$((c + 1)); done
  echo "$c"
}

# ============================================================================== generation
GEN_PAR=${TRC_EXT_GEN_PAR:-32}
gen_args() {  # common generation-queue arguments (run_trace_eval.sh run_generation_pass)
  printf '%s\n' --model "$MODEL_PATH" --model-slug "$SERVED" --api-model "$SERVED" \
    --parallelism-per-endpoint "$GEN_PAR" --preparation-workers 32 --queue-capacity 256 \
    --persistence-workers 16 --finalization-workers 4 \
    --media-transport file-url --allowed-local-media-path "$LMUData" \
    --dataset-manifest "$DATASET_MANIFEST" --dataset-manifest-view trace_eval_v1 \
    --min-image-pixels 3136 --max-image-pixels 12845056 \
    --run-set trace_eval_v1 --run-root "$RUN_ROOT" \
    --temperature 0.6 --top-p 1 --top-k -1 --presence-penalty 0 \
    --repetition-penalty 1 --max-tokens 4096 --seed "$SEED" --compact-prediction-tables
}
generate() {
  local need_gb groups i t0 util_done=0 unhealthy=0 rc healthy partial_flagged=0
  local -a args eps
  need_gb=$(trc_model_min_vram_gb "$MODEL $SERVED")   # last "<n>b" token; unknown size -> 80 GB
  if [ "$TRC_MIN_MIB" -lt $((need_gb * 1000)) ]; then
    trc_say "FATAL: config error: $SERVED needs >= ${need_gb} GB per card; the smallest card here has ${TRC_MIN_MIB} MiB"; exit 4
  fi
  groups="${DEVS[*]}"
  # 24 GB cards: at the protocol's 0.90 a 3B server OOMs in the vision tower on the largest images (12.8 MP) and dies
  # mid-run (smoke on RTX PRO 4000); 0.80 leaves room for those activations (serving parameter only)
  local util=${TRC_EXT_GEN_UTIL:-}
  [ -n "$util" ] || { if [ "$TRC_MIN_MIB" -lt 40000 ]; then util=0.80; else util=0.90; fi; }
  # requests in flight per endpoint: the protocol's 32 was sized for 8 x H100 endpoints; a node with few, large cards
  # needs more to keep each server busy (serving parameter only; recorded in generation_summary.json)
  if [ -z "${TRC_EXT_GEN_PAR:-}" ]; then
    if [ "$TRC_MIN_MIB" -ge 80000 ]; then GEN_PAR=128; elif [ "$TRC_MIN_MIB" -ge 40000 ]; then GEN_PAR=64; else GEN_PAR=32; fi
  fi
  T[gen_pool_start]=$(date +%s)
  POOL_UTIL=$util pool_start gen 18000 "$MODEL_PATH" "$SERVED" "$groups" 32768 256 32768 || exit 5
  T[gen_pool_ready]=$(date +%s)
  mapfile -t args < <(gen_args)
  eps=(); for i in "${POOL_ENDPOINTS[@]}"; do eps+=(--api-base "$i"); done
  trc_say "generation: ${#POOL_ENDPOINTS[@]} endpoints x $GEN_PAR in flight, seed $SEED, ${SUBSET} rows (log logs/generation.seed$SEED.log)"
  setsid "$PY" "$LIB" gen-queue ${ROWS:+--rows "$ROWS"} -- "${args[@]}" "${eps[@]}" --defer-finalization \
    >>"$ROOT/logs/generation.seed$SEED.log" 2>&1 </dev/null 3>&- &
  QUEUE_PID=$!
  t0=$(date +%s)
  while kill -0 "$QUEUE_PID" 2>/dev/null; do
    sleep 15
    if [ "$util_done" = 0 ] && [ $(( $(date +%s) - t0 )) -ge 90 ] && kill -0 "$QUEUE_PID" 2>/dev/null; then
      util_done=1; trc_util_check || true
    fi
    healthy=$(pool_healthy_count)
    if [ "$healthy" -eq 0 ]; then unhealthy=$((unhealthy + 1)); else unhealthy=0; fi
    if [ "$healthy" -lt "${#POOL_ENDPOINTS[@]}" ] && [ "$partial_flagged" = 0 ]; then
      partial_flagged=1
      trc_flag "endpoint_dead $healthy of ${#POOL_ENDPOINTS[@]} generation endpoints answer; the queue fails rows over to the others (see logs/vllm_gen.*)"
    fi
    if [ "$unhealthy" -ge 4 ]; then
      trc_flag "server_dead every generation endpoint stopped answering; stopping the queue"
      trc_stop_tree "$QUEUE_PID" "generation queue" 30
      break
    fi
  done
  wait "$QUEUE_PID"; rc=$?
  QUEUE_PID=""
  T[gen_done]=$(date +%s)
  grep -E '^\[api-generate:(start|deferred|prepare)\]|ext-eval:subset|Error|error' "$ROOT/logs/generation.seed$SEED.log" | tail -n 30 \
    | while read -r line; do trc_log "  $line"; done
  trc_say "generation queue rc=$rc after $(( T[gen_done] - T[gen_pool_ready] ))s (vLLM start $(( T[gen_pool_ready] - T[gen_pool_start] ))s)"
  pool_stop
  return "$rc"
}
finalize() {
  local -a args
  mapfile -t args < <(gen_args)
  T[finalize_start]=$(date +%s)
  if ! "$PY" "$LIB" gen-queue ${ROWS:+--rows "$ROWS"} -- "${args[@]}" --finalize-only \
       >>"$ROOT/logs/generation_finalize.seed$SEED.log" 2>&1 </dev/null; then
    trc_say "finalization failed (logs/generation_finalize.seed$SEED.log):"; tail -n 20 "$ROOT/logs/generation_finalize.seed$SEED.log"
    return 1
  fi
  T[finalize_done]=$(date +%s)
  trc_say "finalized in $(( T[finalize_done] - T[finalize_start] ))s"
}
extract_stats() {
  if lib extract --run-root "$RUN_ROOT" --slug "$SERVED" --out "$ROOT/extract.seed$SEED.json" >"$STATE/extract.txt" 2>&1; then
    trc_say "deterministic extraction: $(tail -n 1 "$STATE/extract.txt")"
  else
    trc_flag "extract_failed $(tail -n 1 "$STATE/extract.txt")"
  fi
}

GPU_READY=0
if [ "$GEN_DONE" = 0 ]; then
  gpu_setup; GPU_READY=1
  generate; grc=$?
  if [ "$grc" -ne 0 ]; then
    trc_say "INCOMPLETE: generation queue rc=$grc; rerun the same task line (finished rows are kept)"; exit 1
  fi
  finalize || exit 1
  if ! gen_complete; then cat "$STATE/gen_validate.txt"; trc_say "INCOMPLETE: generation does not validate"; exit 1; fi
  trc_say "generation complete: $(tail -n 1 "$STATE/gen_validate.txt")"
fi
if [ ! -f "$ROOT/extract.seed$SEED.json" ] || [ "$GEN_DONE" = 0 ]; then extract_stats; fi

# ============================================================================== judge + scoring
scored_complete() {
  lib validate --plan "$PLAN" --run-root "$RUN_ROOT" --score-root "$SCORE_ROOT" --phase score \
    --out "$ROOT/validation.seed$SEED.json" >"$STATE/score_validate.txt" 2>&1
}
write_marker() {  # REASON
  local line
  line="source \"\$TRACE_CERT_RUNS/host/env.sh\" && cd \"\$TRACE_ROOT\" && ${TRC_EXT_RUNS:+TRC_EXT_RUNS=$TRC_EXT_RUNS }bash experiments/certificates/jobs/ext_eval.sh $MODEL $SERVED --seed $SEED"
  case "$SUBSET" in q25) line+=" --subset q25" ;; smoke) line+=" --smoke" ;; esac
  line+=" --judge-only"
  "$PY" -I -c 'import json,sys,datetime; json.dump({"pending": "judge", "reason": sys.argv[2], "node": sys.argv[3], "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "task_line": sys.argv[4], "dispatch_knobs": "MIN_VRAM=80 (or ONLY= a node with >= 2x48 GB / 4x24 GB cards)"}, open(sys.argv[1], "w"), indent=1)' \
    "$MARKER" "$1" "$(hostname -s)" "$line"
  trc_flag "judge_pending $SERVED $SUBSET seed $SEED: $1 (marker $MARKER)"
}
SCORE_RC=0
if scored_complete; then
  trc_say "scores already complete: $(tail -n 1 "$STATE/score_validate.txt")"
  # complete scores are kept as they are; say so when they were made under other code than the current one (a
  # post-release fix since): moving the run's mme_reasoning score slice aside makes the next attempt rescore it all
  score_code=$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["contract"]["trace_eval_code_provenance"]["sha256"])' \
                 "$SCORE_ROOT/score_campaign_manifest_seed_$SEED.json" 2>/dev/null)
  if [ -n "$score_code" ] && [ "$score_code" != "$TRACE_EVAL_CODE_HASH" ]; then
    trc_flag "score_code_stale $SERVED $SUBSET seed $SEED: scored under code ${score_code:0:12}, current ${TRACE_EVAL_CODE_HASH:0:12} (README: rescoring after a post-release fix)"
  fi
else
  [ "$GPU_READY" = 1 ] || { gpu_setup; GPU_READY=1; }
  jplan=$(lib judge-plan --devices "$(IFS=,; echo "${DEVS[*]}")" --mib "$(IFS=,; echo "${TRC_GPU_MIB[*]}")" --judge-dir "$JUDGE_MODEL")
  if [ "${TRC_EXT_NO_JUDGE:-0}" = 1 ]; then jplan='{"feasible": false, "reason": "TRC_EXT_NO_JUDGE=1 (judge deferred on purpose)"}'; fi
  trc_say "judge plan: $jplan"
  read -r JFEAS JGROUPS JBATCH < <("$PY" -I -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d["feasible"]), d.get("gpu_groups","-").replace(" ",";"), d.get("max_num_batched_tokens", 0))' "$jplan")
  if [ "$JFEAS" != 1 ] && [ "$JUDGE_ONLY" = 1 ]; then
    trc_say "FATAL: config error: --judge-only on a node that cannot host Qwen3-32B: $jplan"; exit 4
  fi
  if [ "$JFEAS" != 1 ]; then
    write_marker "$("$PY" -I -c 'import json,sys; print(json.loads(sys.argv[1])["reason"])' "$jplan")"
    lib summary --plan "$PLAN" --score-root "$SCORE_ROOT" --extract "$ROOT/extract.seed$SEED.json" \
      --out-json "$ROOT/summary.seed$SEED.json" --out-md "$ROOT/summary.seed$SEED.md" >/dev/null 2>&1 || true
    trc_say "generation finalized; judge pending on a node that can host Qwen3-32B: $MARKER"
    exit 0
  fi
  jverify=(--entry "$JUDGE_SERVED=$JUDGE_MODEL=$JUDGE_REVISION"); [ "${TRC_EXT_JUDGE_DEEP:-0}" = 1 ] && jverify+=(--deep)
  if ! "$PY" "$SCRIPTS_ROOT/prepare_trace_eval_models.py" verify "${jverify[@]}" >"$STATE/judge_verify.log" 2>&1; then
    cat "$STATE/judge_verify.log"; trc_say "FATAL: config error: judge model verification failed"; exit 4
  fi
  [ "$JUDGE_SERVED" = qwen3-32b-judge ] || trc_flag "standin_judge $JUDGE_SERVED: plumbing test only, scores are NOT trace_eval_v1 results"
  T[judge_pool_start]=$(date +%s)
  pool_start judge 18100 "$JUDGE_MODEL" "$JUDGE_SERVED" "${JGROUPS//;/ }" 8192 128 "$JBATCH" qwen3 \
    "$CLONE/rlvr/examples/prompts/chat_template_no_think.jinja" || exit 5
  T[judge_pool_ready]=$(date +%s)
  jeps=(); for e in "${POOL_ENDPOINTS[@]}"; do jeps+=(--judge-endpoint "$e"); done
  trc_say "score campaign: judge ${#POOL_ENDPOINTS[@]} endpoints (vLLM start $(( T[judge_pool_ready] - T[judge_pool_start] ))s), log logs/scoring.seed$SEED.log"
  score_campaign() {  # run_trace_eval.sh run_scoring, one seed
    "$PY" "$SCRIPTS_ROOT/run_trace_eval_score_campaign.py" \
      --seed "$SEED" --shared-seed-root --resume --emit-archive \
      --campaign "$MODEL_PATH" "$SERVED" "$CAMPAIGN_ROOT" --score-root "$SCORE_ROOT" --dataset-manifest "$DATASET_MANIFEST" \
      --python "$PY" --eval-deps "$EVAL_DEPS_ROOT" \
      --vlmeval-root "$VLMEVALKIT_ROOT" --lmu-data "$LMUData" --hf-home "$HF_HOME" \
      --judge-model "$JUDGE_MODEL" --judge-api-model "$JUDGE_SERVED" \
      "${jeps[@]}" --official-workers 8 --direct-workers 8 --mme-workers 3 \
      --eval-nproc 16 --judge-api-parallelism 64 --judge-api-batch-size 64 \
      >>"$ROOT/logs/scoring.seed$SEED.log" 2>&1 </dev/null 3>&-
  }
  score_campaign; SCORE_RC=$?
  if [ "$SCORE_RC" != 0 ] && tail -n 40 "$ROOT/logs/scoring.seed$SEED.log" \
       | grep -qE 'does not match source hashes/judge contract|Cannot create a seed-[0-9]+ contract over existing outputs'; then
    # an earlier partial scoring had another contract (judge endpoints, re-finalized workbooks): keep it aside, rescore
    stale=$SCORE_ROOT/stale/seed_$SEED.$JOBTAG
    mkdir -p "$stale"
    mv -f "$SCORE_ROOT/score_campaign_manifest_seed_$SEED.json" "$stale/" 2>/dev/null
    [ -d "$SCORE_ROOT/seed_$SEED" ] && mv -f "$SCORE_ROOT/seed_$SEED" "$stale/"
    trc_say "score contract changed since the last attempt; moved the partial scoring to $stale and rescoring"
    score_campaign; SCORE_RC=$?
  fi
  T[score_done]=$(date +%s)
  trc_say "score campaign rc=$SCORE_RC after $(( T[score_done] - T[judge_pool_ready] ))s"
  [ "$SCORE_RC" = 0 ] || tail -n 25 "$ROOT/logs/scoring.seed$SEED.log"
  pool_stop
  trc_heartbeat_stop; trc_claim_release; trc_heartbeat_start   # GPUs free again; keep the run lease
fi

# ============================================================================== summary + validation (CPU)
lib summary --plan "$PLAN" --score-root "$SCORE_ROOT" --extract "$ROOT/extract.seed$SEED.json" \
  --out-json "$ROOT/summary.seed$SEED.json" --out-md "$ROOT/summary.seed$SEED.md" >"$STATE/summary.txt" 2>&1
trc_say "$(tail -n 1 "$STATE/summary.txt") -> $ROOT/summary.seed$SEED.md"
VRC=0
scored_complete || VRC=1
while IFS= read -r line; do trc_say "$line"; done <"$STATE/score_validate.txt"
if [ "$SUBSET" = full ] && [ "$VRC" = 0 ]; then
  # the protocol's own coverage verifiers and summary for the canonical (full) layout. The protocol's code hash covers
  # generation and scoring code alike and enters every generation contract: a generation made before a scoring-only
  # post-release fix is verified against the code hash it was produced under (recorded in the local archive)
  gen_verify=("$PY" "$SCRIPTS_ROOT/verify_trace_eval.py"); gen_code=$TRACE_EVAL_CODE_HASH
  rec_code=$(lib archive-code-hash --archive "$ROOT/hf_archive" --stage generation --slug "$SERVED" --seed "$SEED" \
               --run-id "$RUN_TAG" 2>/dev/null) || rec_code=""
  if [ -n "$rec_code" ] && [ "$rec_code" != "$TRACE_EVAL_CODE_HASH" ]; then
    gen_verify=("$PY" "$LIB" verify-generation --code-hash "$rec_code" --); gen_code=$rec_code
    trc_say "generation was produced under code ${rec_code:0:12}, scores under ${TRACE_EVAL_CODE_HASH:0:12}: verifying the generation against ${rec_code:0:12}"
  fi
  if "${gen_verify[@]}" --campaign-root "$CAMPAIGN_ROOT" --phase generation \
       --model-entry "$SERVED=$MODEL_PATH=$REVISION" --dataset-manifest "$DATASET_MANIFEST" \
       --code-hash "$gen_code" --evaluator-provenance-sha256 "$TRACE_EVAL_EVALUATOR_HASH" \
       --vlmeval-root "$VLMEVALKIT_ROOT" --seeds "$SEED" >"$STATE/verify_generation.txt" 2>&1 \
     && "$PY" "$SCRIPTS_ROOT/verify_trace_eval.py" --campaign-root "$CAMPAIGN_ROOT" --score-root "$SCORE_ROOT" \
       --phase score --model-slug "$SERVED" --dataset-manifest "$DATASET_MANIFEST" \
       --vlmeval-root "$VLMEVALKIT_ROOT" --seeds "$SEED" >"$STATE/verify_score.txt" 2>&1; then
    trc_say "protocol verify_trace_eval.py: generation (code ${gen_code:0:12}) and score coverage ok"
  else
    # our validation above is the gate; the protocol's receipt verifier is recorded and flagged, not blocking
    trc_flag "protocol_verify_failed $SERVED seed $SEED (see $STATE/verify_*.txt)"
  fi
  "$PY" "$SCRIPTS_ROOT/summarize_trace_eval.py" --score-root-base "$SCORE_ROOT" --suite trace_eval_v1 --seeds "$SEED" \
    --model-entry "$SERVED=$SERVED" --excel "$ROOT/results/${RUN_TAG}_results.xlsx" \
    --markdown "$ROOT/results/${RUN_TAG}_results.md" >"$STATE/protocol_summary.txt" 2>&1 \
    || trc_flag "protocol_summary_failed (see $STATE/protocol_summary.txt)"
fi
if [ "$VRC" = 0 ] && [ "$SCORE_RC" = 0 ]; then
  rm -f "$MARKER"
  trc_say "COMPLETE: $SERVED trace_eval_v1 $SUBSET seed $SEED validated ($ROOT/summary.seed$SEED.md)"
  exit 0
fi
trc_say "INCOMPLETE: rerun the same task line (generation and finished score slices are kept)"
exit 1
