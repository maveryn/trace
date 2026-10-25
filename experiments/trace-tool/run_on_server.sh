#!/usr/bin/env bash
# The whole Phase 1 run on a Linux GPU server, in one file (see EXPERIMENT_DESIGN.md).
#
#   bash run_on_server.sh                                   # everything: 7B, then 3B, then judge + compare
#   GPUS=0,1,2,3 bash run_on_server.sh                      # one vLLM replica per GPU
#
# It runs for many hours: start it in tmux/screen, or detached, and stop it with kill:
#   nohup bash run_on_server.sh > run.log 2>&1 &
#   kill <pid>        # stops the current step and vLLM within seconds; rerun the same command to resume
#
# The design's pilot gate as two runs (look at the pilot before spending GPU hours on the full run):
#   STEPS="setup check smoke pilot judge compare" bash run_on_server.sh
#   STEPS="full judge compare" bash run_on_server.sh
#
# Steps, always in this order (choose any subset with STEPS):
#   setup    virtualenv, vLLM, this package, headless OpenCV, CUDA preflight, unit tests, dataset
#   check    prompt rendering against the served model (scripts/check_template.py)
#   smoke    20 rows x every condition, then a tool-call health summary
#   pilot    1000 rows (one per task), seed 42, all 11 conditions (+ the 5 *_forced ones for Qwen3.5)
#   full     2000 rows x seeds 42,43,44, the 7 core conditions (+ the 5 *_forced ones for Qwen3.5)
#   judge    Qwen3-32B extraction pass for unresolved answers, then re-score (TRACE protocol)
#   compare  paired comparisons vs no_tool and vs no_tool_budget; if forced conditions ran, also
#            compare_forced.md (forced conditions vs no_tool, their own Holm family) and
#            compare_forced_vs_unforced.md (each forced condition vs its base, e.g. zoom_forced vs zoom;
#            written even without no_tool); then results_summary.tgz to send back
# Every step resumes where it stopped. The script stops (exit 1, "INCOMPLETE") instead of
# carrying on when vLLM dies or rows end in endpoint errors; fix the cause and rerun.
#
# Settings (environment variables):
#   MODELS          checkpoints, space-separated (default: TRACE's Qwen2.5-VL 7B and 3B bases at
#                   TRACE's pinned revisions; maveryn/trace-qwen2.5-vl-7b also works; so do
#                   Qwen/Qwen3.5-4B and Qwen/Qwen3.5-9B, at our pinned revisions, with thinking off)
#   REVISION        one revision for every model in MODELS; REVISION= (empty) means latest
#   EXTRA_CONDITIONS  conditions added to smoke/pilot/full for every model in MODELS (default, per model:
#                   the 5 *_forced conditions for Qwen/Qwen3.5-*, none otherwise; see scripts/_model_env.sh)
#   GPUS            GPUs for the model server (default 0); one data-parallel replica per GPU
#   JUDGE_GPUS      GPUs for the Qwen3-32B judge, tensor parallel; must be 1, 2, 4 or 8 GPUs with
#                   ~80 GB in total (default: the first 1/2/4/8 of GPUS)
#   DATASET_PATH    local copy of the TRACE validation parquet (default: download from the Hub)
#   CONCURRENCY     episodes in flight per run (default 64 per model-server GPU)
#   VLLM_SPEC       what setup installs (default "vllm"; pin e.g. "vllm==0.30.0" to match your CUDA driver)
#   VENV, PYTHON    virtualenv dir (default .venv) and the Python >= 3.10 that creates it (default python3)
#   HF_HOME         Hugging Face cache (~16 GB for 7B, ~8 GB for 3B, ~65 GB for the judge;
#                   Qwen3.5: ~9 GB for 4B, ~19 GB for 9B)
#   PREFETCH        1 (default): download weights before starting vLLM; 0: let vLLM download them
#   PORT, JUDGE_PORT, SERVER_TIMEOUT (seconds to wait for vLLM to load, default 1800), LOGS (default logs)
set -euo pipefail

# user paths are relative to where the script was started, not the repo
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
if [ -n "${DATASET_PATH:-}" ]; then DATASET_PATH=$(abspath "$DATASET_PATH"); fi
if [ -n "${VENV:-}" ]; then VENV=$(abspath "$VENV"); fi
if [ -n "${LOGS:-}" ]; then LOGS=$(abspath "$LOGS"); fi
cd "$(dirname "$0")"

STEPS=${STEPS:-setup check smoke pilot full judge compare}
MODELS=${MODELS:-Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-3B-Instruct}
GPUS=${GPUS:-0}
PORT=${PORT:-18100}
JUDGE_PORT=${JUDGE_PORT:-18200}
VENV=${VENV:-$PWD/.venv}
PYTHON=${PYTHON:-python3}
LOGS=${LOGS:-$PWD/logs}
PREFETCH=${PREFETCH:-1}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1800}
VLLM_SPEC=${VLLM_SPEC:-vllm}
JUDGE_MODEL=Qwen/Qwen3-32B
JUDGE_REVISION=9216db5781bf21249d130ec9da846c4624c16137   # TRACE's pinned judge
JUDGE_HEADS=64                                           # Qwen3-32B attention heads: TP must divide this
CORE_TOOL_CONDITIONS="zoom visual_ops calculator ocr all"
FORCED_TOOL_CONDITIONS="zoom_forced visual_ops_forced calculator_forced ocr_forced all_forced"  # base = name minus _forced

mkdir -p "$LOGS"
has_step() { [[ " $STEPS " == *" $1 "* ]]; }
say() { printf '\n==== %s  %s\n' "$(date '+%F %T')" "$*"; }
count_gpus() { tr ',' '\n' <<<"$1" | grep -c .; }
served_name() { basename "$1" | tr '[:upper:]' '[:lower:]'; }
# the pinned revision for a model (scripts/_model_env.sh); empty for other models
pinned_revision() { (unset REVISION; MODEL=$1; source scripts/_model_env.sh; echo "$REVISION"); }

# ---------------------------------------------------------------- validate settings up front
for step in $STEPS; do
  case $step in setup|check|smoke|pilot|full|judge|compare) ;; *) echo "unknown step '$step'" >&2; exit 2 ;; esac
done
if [ -n "${REVISION+set}" ]; then USER_REVISION_SET=1; USER_REVISION=$REVISION; else USER_REVISION_SET=; USER_REVISION=; fi
unset REVISION
if [ -n "$USER_REVISION" ] && [ "$(wc -w <<<"$MODELS")" -gt 1 ]; then
  echo "REVISION=$USER_REVISION would apply to every model in MODELS; run one model at a time" >&2; exit 2
fi
model_revision() { if [ -n "$USER_REVISION_SET" ]; then echo "$USER_REVISION"; else pinned_revision "$1"; fi; }
if [ -z "${JUDGE_GPUS:-}" ]; then  # the largest prefix of GPUS the judge can use (1, 2, 4 or 8 GPUs)
  IFS=, read -ra _gpus <<<"$GPUS"
  for k in 8 4 2 1; do if [ "${#_gpus[@]}" -ge "$k" ]; then JUDGE_GPUS=$(IFS=,; echo "${_gpus[*]:0:k}"); break; fi; done
fi
JUDGE_TP=$(count_gpus "$JUDGE_GPUS")
if has_step judge && [ $((JUDGE_HEADS % JUDGE_TP)) -ne 0 ]; then
  echo "JUDGE_GPUS=$JUDGE_GPUS: the judge needs 1, 2, 4 or 8 GPUs (tensor parallel must divide $JUDGE_HEADS heads)" >&2; exit 2
fi
if [ -n "${DATASET_PATH:-}" ] && [ ! -f "$DATASET_PATH" ]; then echo "DATASET_PATH=$DATASET_PATH does not exist" >&2; exit 2; fi
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
DATA=()
if [ -n "${DATASET_PATH:-}" ]; then DATA=(--dataset-path "$DATASET_PATH"); fi
export DATASET_PATH=${DATASET_PATH:-}
SETSID=()
if command -v setsid >/dev/null; then SETSID=(setsid); fi   # own process group per step / server

# ---------------------------------------------------------------- process handling
STEP_PID=""
SERVER_PID=""
SERVER_LOG=""
kill_tree() {  # kill_tree PID SIGNAL: its process group if it leads one, else it and its children
  kill -"$2" -- -"$1" 2>/dev/null || { pkill -"$2" -P "$1" 2>/dev/null; kill -"$2" "$1" 2>/dev/null; } || true
}
tree_alive() { pgrep -g "$1" >/dev/null 2>&1 || kill -0 "$1" 2>/dev/null; }
stop_tree() {  # stop_tree PID LABEL: TERM, wait up to 2 min, then KILL (vLLM workers must free the GPUs)
  kill_tree "$1" TERM
  for _ in $(seq 60); do tree_alive "$1" || break; sleep 2; done
  if tree_alive "$1"; then say "$2 did not stop within 2 minutes; killing it"; kill_tree "$1" KILL; sleep 3; fi
  wait "$1" 2>/dev/null || true
}
stop_server() {
  if [ -n "$SERVER_PID" ]; then say "stopping vLLM (pid $SERVER_PID)"; stop_tree "$SERVER_PID" vLLM; fi
  SERVER_PID=""
}
on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$STEP_PID" ]; then stop_tree "$STEP_PID" "the current step"; fi
  stop_server
  exit "$rc"
}
trap on_exit EXIT
trap 'say "interrupted"; exit 130' INT TERM

# run_step LOGFILE COMMAND...: in its own process group, appended to LOGFILE; a signal to this
# script interrupts the wait at once and on_exit stops the step and vLLM
run_step() {
  local logfile=$1 rc=0; shift
  printf '\n==== %s  %s\n' "$(date '+%F %T')" "$*" >>"$logfile"
  STEP_LOG=$logfile ${SETSID[@]+"${SETSID[@]}"} bash -c 'set -o pipefail; "$@" 2>&1 | tee -a "$STEP_LOG"' run_step "$@" &
  STEP_PID=$!
  wait "$STEP_PID" || rc=$?
  STEP_PID=""
  return "$rc"
}

incomplete() { echo "INCOMPLETE: $*" >&2; echo "Fix the cause, then rerun the same command; finished rows are kept." >&2; exit 1; }

# require_server PORT: the server must still answer after a step
require_server() {
  if ! curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null; then
    echo "vLLM on port $1 stopped responding; last lines of $SERVER_LOG:" >&2
    tail -n 40 "$SERVER_LOG" >&2 || true
    incomplete "the server died during the run"
  fi
}

# check_runs ROOT FIELD: no summary.json under ROOT may have FIELD > 0
check_runs() {
  python - "$1" "$2" <<'PY'
import json, pathlib, sys
root, field = pathlib.Path(sys.argv[1]), sys.argv[2]
bad = {}
for path in sorted(root.glob("*/seed*/summary.json")):
    value = json.loads(path.read_text()).get(field) or 0
    if value:
        bad[str(path.parent)] = value
for run, value in bad.items():
    print(f"  {run}: {field} = {value}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
}

# start_server MODEL REVISION SERVED PORT GPUS [VAR=value ...]  (extra vars go to scripts/serve_vllm.sh)
start_server() {
  local model=$1 revision=$2 served=$3 port=$4 gpus=$5; shift 5
  SERVER_LOG="$LOGS/vllm-$served.log"
  if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then
    echo "port $port already serves something (an old vLLM?); stop it or set PORT/JUDGE_PORT" >&2; exit 1
  fi
  if [ "$PREFETCH" = 1 ]; then
    say "downloading $model ${revision:+@ $revision}"
    python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)" "$model" "$revision"
  fi
  say "starting vLLM for $model on GPUs $gpus, port $port (log: $SERVER_LOG)"
  printf '\n==== %s  starting %s %s\n' "$(date '+%F %T')" "$model" "$revision" >>"$SERVER_LOG"
  env CUDA_VISIBLE_DEVICES="$gpus" MODEL="$model" REVISION="$revision" SERVED="$served" PORT="$port" "$@" \
    ${SETSID[@]+"${SETSID[@]}"} bash scripts/serve_vllm.sh >>"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited during startup; last lines of $SERVER_LOG:" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1
    fi
    if [ "$waited" -ge "$SERVER_TIMEOUT" ]; then
      echo "vLLM was not ready after ${SERVER_TIMEOUT}s; last lines of $SERVER_LOG:" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1
    fi
    sleep 10; waited=$((waited + 10))
  done
  say "vLLM ready: $served"
}

say "run_on_server.sh (pid $$; stop with: kill $$)  steps: $STEPS"

# ---------------------------------------------------------------- setup
if has_step setup; then
  say "setup"
  "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python >= 3.10 is required")'
  [ -f "$VENV/bin/activate" ] || "$PYTHON" -m venv --clear "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip
  python -m pip install "$VLLM_SPEC"
  python -m pip install -e ".[ocr,json5,dev]"
  # vLLM pins opencv-python-headless and RapidOCR pulls opencv-python; both own cv2/, so keep only the headless one
  python -m pip uninstall -y opencv-python opencv-python-headless
  python -m pip install opencv-python-headless
  {
    date; nvidia-smi || true
    python -c 'import sys, torch, vllm; print("python", sys.version.split()[0], "| vllm", vllm.__version__, "| torch", torch.__version__, "| torch CUDA", torch.version.cuda, "| GPUs visible:", torch.cuda.device_count())'
    python -m pip freeze
  } >"$LOGS/env.txt" 2>&1 || true
  if ! python -c 'import torch, vllm; assert torch.cuda.is_available()' 2>/dev/null; then
    grep -E "^python|Driver Version" "$LOGS/env.txt" >&2 || true
    echo "torch cannot use the GPU: the installed vLLM/torch build likely needs a newer NVIDIA driver." >&2
    echo "Install a vLLM release built for your CUDA version, e.g. VLLM_SPEC='vllm==<version>' (see docs.vllm.ai), and rerun setup." >&2
    exit 1
  fi
  head -n 1 "$LOGS/env.txt"; grep -E "^python " "$LOGS/env.txt" || true
  python -c "from trace_tool.tools.ocr import OCRTool; OCRTool(); print('OCR backend loads')"
  TRACE_VALIDATION_PARQUET=${DATASET_PATH:-} python -m pytest -q
  python -c "import sys; from trace_tool.data import load_split; print(len(load_split('validation', local_path=sys.argv[1] or None)), 'validation rows')" "${DATASET_PATH:-}"
fi
if [ ! -f "$VENV/bin/activate" ]; then echo "no virtualenv at $VENV; run the setup step first" >&2; exit 1; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---------------------------------------------------------------- generation, per model
if has_step check || has_step smoke || has_step pilot || has_step full; then
  NGPU=$(count_gpus "$GPUS")
  export CONCURRENCY=${CONCURRENCY:-$((64 * NGPU))}
  export ENDPOINT="http://127.0.0.1:$PORT/v1"
  for MODEL in $MODELS; do
    SERVED=$(served_name "$MODEL")
    REV=$(model_revision "$MODEL")
    export MODEL SERVED REVISION="$REV"
    start_server "$MODEL" "$REV" "$SERVED" "$PORT" "$GPUS" TP=1 DP="$NGPU"

    if has_step check; then
      say "check: $MODEL"
      run_step "$LOGS/check-$SERVED.log" python scripts/check_template.py --model "$MODEL" ${REV:+--revision "$REV"} \
        --endpoint "http://127.0.0.1:$PORT" --served "$SERVED"
    fi
    for stage in smoke pilot full; do
      has_step "$stage" || continue
      say "$stage: $MODEL"
      if [ "$stage" = smoke ]; then
        run_step "$LOGS/smoke-$SERVED.log" bash scripts/smoke_test.sh
      else
        run_step "$LOGS/$stage-$SERVED.log" env STAGE="$stage" bash scripts/run_phase1.sh
      fi
      require_server "$PORT"
      check_runs "runs/$stage/$SERVED" excluded_endpoint_errors || incomplete "rows of $stage ($SERVED) ended in endpoint errors"
    done
    stop_server
  done
  unset REVISION
fi

# run directories of the requested models that exist (pilot and full)
OUTS=()
for MODEL in $MODELS; do
  for stage in pilot full; do
    out="runs/$stage/$(served_name "$MODEL")"
    if [ -d "$out" ]; then OUTS+=("$out"); fi
  done
done
log_name() { tr '/' '_' <<<"$1"; }
score_out() {  # score_out OUT: with the judge results if there are any; judge_pending.jsonl lists the rest
  local judged=()
  if [ -f "$1/judge_results.jsonl" ]; then judged=(--judge-results "$1/judge_results.jsonl"); fi
  trace-tool-score "$1" ${judged[@]+"${judged[@]}"} --report "$1/report.md" ${DATA[@]+"${DATA[@]}"} >>"$LOGS/score-$(log_name "$1").log"
}

# ---------------------------------------------------------------- judge
if has_step judge; then
  if [ ${#OUTS[@]} -eq 0 ]; then
    say "judge: no pilot/full runs found; skipping"
  else
    say "judge: ${OUTS[*]}"
    for out in "${OUTS[@]}"; do score_out "$out"; done
    any_pending() { for out in "${OUTS[@]}"; do if [ -s "$out/judge_pending.jsonl" ]; then return 0; fi; done; return 1; }
    if any_pending; then
      start_server "$JUDGE_MODEL" "$JUDGE_REVISION" qwen3-32b-judge "$JUDGE_PORT" "$JUDGE_GPUS" \
        TP="$JUDGE_TP" DP=1 MAX_IMAGES=0 MAX_LEN=16384
      for attempt in 1 2; do  # a second pass retries items that failed transiently
        for out in "${OUTS[@]}"; do
          if [ -s "$out/judge_pending.jsonl" ]; then
            run_step "$LOGS/judge.log" trace-tool-judge --pending "$out/judge_pending.jsonl" --out "$out/judge_results.jsonl" \
              --endpoint "http://127.0.0.1:$JUDGE_PORT/v1" --model qwen3-32b-judge
            score_out "$out"
          fi
        done
        any_pending || break
        require_server "$JUDGE_PORT"
      done
      stop_server
    fi
    for out in "${OUTS[@]}"; do
      check_runs "$out" judge_pending || incomplete "some answers in $out still await the judge"
      echo "  $out/report.md"
    done
  fi
fi

# ---------------------------------------------------------------- compare + results bundle
if has_step compare; then
  for out in "${OUTS[@]+"${OUTS[@]}"}"; do
    [ -f "$out/report.md" ] || score_out "$out"
    tools=()
    for c in $CORE_TOOL_CONDITIONS; do if [ -d "$out/$c" ]; then tools+=("$c"); fi; done
    forced=()
    for c in $FORCED_TOOL_CONDITIONS; do if [ -d "$out/$c" ]; then forced+=("$c"); fi; done
    pairs=()   # forced conditions whose unforced base ran too; comparing these needs no no_tool
    if [ ${#forced[@]} -gt 0 ]; then
      for c in "${forced[@]}"; do if [ -d "$out/${c%_forced}" ]; then pairs+=("$c"); fi; done
    fi
    if [ ! -d "$out/no_tool" ] || [ $((${#tools[@]} + ${#forced[@]})) -eq 0 ]; then
      if [ ${#pairs[@]} -eq 0 ]; then
        say "compare: $out has no no_tool baseline or tool conditions; skipping"; continue
      fi
      say "compare: $out (no no_tool baseline: forced vs unforced only)"
    else
      say "compare: $out"
      if [ ${#tools[@]} -gt 0 ]; then
        family=("${tools[@]}")
        if [ -d "$out/no_tool_budget" ]; then family=(no_tool_budget "${tools[@]}"); fi
        # decision rule (a): tools vs no_tool (pre-registered Holm family) and vs no_tool_budget
        trace-tool-compare "$out" --baseline no_tool --conditions "$(IFS=,; echo "${family[*]}")" --out "$out/compare.md" >/dev/null
        if [ -d "$out/no_tool_budget" ]; then
          trace-tool-compare "$out" --baseline no_tool_budget --conditions "$(IFS=,; echo "${tools[*]}")" --out "$out/compare_vs_budget.md" >/dev/null
        fi
      fi
      if [ ${#forced[@]} -gt 0 ]; then
        # forced final answer: a separate Holm family vs no_tool
        trace-tool-compare "$out" --baseline no_tool --conditions "$(IFS=,; echo "${forced[*]}")" --out "$out/compare_forced.md" >/dev/null
      fi
      trace-tool-compare "$out" --baseline no_tool --out "$out/compare_all.md" >/dev/null   # every condition, diagnostics too
    fi
    if [ ${#pairs[@]} -gt 0 ]; then
      # each forced condition vs its unforced base, e.g. zoom_forced vs zoom
      for c in "${pairs[@]}"; do trace-tool-compare "$out" --baseline "${c%_forced}" --conditions "$c"; done >"$out/compare_forced_vs_unforced.md"
    fi
    ls "$out"/compare*.md "$out/report.md" | sed 's/^/  /'
  done
  # bundle to send back: reports, comparisons, per-row scores, summaries, configs, env and check/smoke logs (no transcripts)
  roots=()
  for d in runs "$LOGS"; do if [ -d "$d" ]; then roots+=("$d"); fi; done
  if [ ${#roots[@]} -eq 0 ]; then
    say "compare: nothing to bundle yet"
  else
    find "${roots[@]}" \( -name report.md -o -name 'compare*.md' -o -name summary.json -o -name scores.jsonl \
      -o -name run_config.json -o -name env.txt -o -name 'check-*.log' -o -name 'smoke-*.log' \) -print0 \
      | tar -czf results_summary.tgz --null -T -
    say "done; send back results_summary.tgz ($(du -h results_summary.tgz | cut -f1))"
  fi
fi
