#!/usr/bin/env bash
# experiments/certificates/jobs/zs_eval.sh MODEL [--smoke] [--dry-run]
#
# DESIGN 7.1 zero-shot evaluation of one model, written as ONE use-gpu task line: it uses every
# GPU of the allocation it lands on (one vLLM process, TP=1, DP=<GPUs>, started by trace-tool's own
# scripts/serve_vllm.sh + _model_env.sh exactly as run_on_server.sh does), generates with
# trace-tool-eval, scores on CPU with the certificate CLI, and validates.
#
#   source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT" && \
#     bash experiments/certificates/jobs/zs_eval.sh Qwen/Qwen3.5-9B [--smoke]
#
# Full run: splits e1_dev, e1_test, e2_test x conditions ans_v1, ann_v1, cert_v1 x seeds 42, 43, 44,
# in the order seed 42 (e1_dev, e1_test, e2_test), then 43, then 44; TRC_PAR (default 3) runs are
# in flight at a time against the one endpoint, sharing 64 x <GPUs> episodes. Outputs:
#   $TRACE_CERT_RUNS/eval/zs/<served>/<split>/<cond>/seed<k>/{responses.jsonl,run_config.json,cert/}
#   $TRACE_CERT_RUNS/eval/zs/<served>/{plan.json,summary.json,summary.<split>.json,summary.md,
#                                      validation.json,trace_tool_conditions_v1.yaml,logs/}
# cert_v1 reads public/<split>.cert_px.parquet for Qwen2.5-VL (incl. maveryn/trace-qwen2.5-vl-*)
# and public/<split>.cert_rel1000.parquet for Qwen3.5 (DESIGN D8); ans_v1/ann_v1 read
# public/<split>.ans/.ann.parquet. trace-tool resumes: a rerun skips finished rows and retries
# endpoint errors; finished, scored runs are not served again.
#
# --smoke: e1_dev cert_v1 seed 42, 5 rows per task (20 rows), into eval/zs_smoke/<served>/, then
#          scores and prints A/G/M/F and the frame-obedience diagnostics.
# --dry-run: CPU preflight only (render the conditions file, frozen-data sha256, request payload
#          check) and print the plan; touches no GPU.
#
# Exit: 0 done; 1 incomplete (gaps, endpoint errors, server death, scoring failure); 2 usage;
#       3 GPUs busy or the run is owned by another live job; 4 configuration mismatch (VRAM below
#       the model's need: 7B/9B >= 40 GB, 3B/4B >= 20 GB; unknown family; environment; data);
#       5 vLLM did not start; 130/143 signalled.
# Knobs: TRC_PAR, TRC_CONCURRENCY (total episodes in flight), TRC_REQUEST_TIMEOUT (1800),
#        TRC_ERR_PASSES (3), TRC_MAX_LEN (16384), TRC_MAX_IMAGES (1), TRC_SMOKE_PER_TASK (5).
# Partial runs and CPU tests only (the design run uses the defaults): TRC_SPLITS and TRC_SEEDS
# (comma lists), TRC_OUT_ROOT (default $TRACE_CERT_RUNS/eval/zs or eval/zs_smoke).
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=common.sh
source "$(dirname "$SELF")/common.sh"

# ============================================================================== generation worker
# One trace-tool-eval run (split, condition, seed), started by the pool below with setsid so the
# parent can stop its whole process group. Repeats while rows end in endpoint errors.

gen_worker() {  # split cond seed parquet expected concurrency rcfile
  local split=$1 cond=$2 seed=$3 parquet=$4 expected=$5 conc=$6 rcfile=$7
  local out=$TRC_W_ROOT/$split resp stat pass rc sel=()
  read -ra sel <<<"${TRC_W_SELECT:-}"
  resp=$out/$cond/seed$seed/responses.jsonl
  # shellcheck disable=SC1091
  source "$TRC_VENV/bin/activate"
  for pass in $(seq 1 "$TRC_W_PASSES"); do
    trc_log "gen $split/$cond/seed$seed pass $pass: concurrency $conc, $parquet"
    trace-tool-eval --endpoint "$TRC_W_ENDPOINT" --model "$TRC_W_SERVED" --model-id "$TRC_W_MODEL" \
      ${TRC_W_REV:+--model-revision "$TRC_W_REV"} --config "$TRC_W_CONFIG" --conditions "$cond" \
      --seeds "$seed" --split "cert_$split" --dataset-path "$parquet" --out "$out" \
      --concurrency "$conc" --request-timeout "$TRC_W_TIMEOUT" ${sel[@]+"${sel[@]}"}
    rc=$?
    stat=$(python "$TRC_JOBS_DIR/zs_report.py" count --responses "$resp" --expected "$expected")
    trc_log "gen $split/$cond/seed$seed pass $pass: rc=$rc $stat"
    case "$stat" in *complete=1*) echo 0 >"$rcfile"; return 0 ;; esac
    [ "$rc" -eq 0 ] || break   # not an endpoint problem (e.g. resume-key mismatch): do not loop
    trc_serve_healthy "$TRC_W_PORT" "$TRC_W_SERVED" || { trc_log "server unhealthy; no further pass"; break; }
  done
  echo 1 >"$rcfile"
  return 1
}

if [ "${1:-}" = "--_gen-worker" ]; then
  shift
  TRC_TAG="gen"
  gen_worker "$@"
  exit $?
fi

# ============================================================================== main

usage() { echo "usage: zs_eval.sh MODEL [--smoke] [--dry-run]" >&2; exit 2; }
MODEL=""; SMOKE=0; DRY=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage ;;
    -*) echo "unknown option $a" >&2; usage ;;
    *) [ -z "$MODEL" ] || usage; MODEL=$a ;;
  esac
done
[ -n "$MODEL" ] || usage

trc_require_env
trc_node_allowed || { echo "FATAL: config error: $(hostname -s) is excluded by fleet policy (${TRC_EXCLUDE_NODES:-grn030, grn078/079})" >&2; exit 4; }
FAMILY=$(trc_family "$MODEL")
FRAME=$(trc_frame "$FAMILY") || { echo "FATAL: config error: no certificate frame for $MODEL (family $FAMILY)" >&2; exit 4; }
trc_model_env "$MODEL"
SERVED=$TRC_SERVED; REV=$TRC_REVISION
[ -n "$SERVED" ] || { echo "FATAL: config error: empty served name for $MODEL" >&2; exit 4; }

if [ "$SMOKE" = 1 ]; then OUT_ROOT=$TRACE_CERT_RUNS/eval/zs_smoke; else OUT_ROOT=$TRACE_CERT_RUNS/eval/zs; fi
OUT_ROOT=${TRC_OUT_ROOT:-$OUT_ROOT}
ROOT=$OUT_ROOT/$SERVED
JOBTAG=${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}.$(date +%m%d_%H%M%S)
STATE=$ROOT/logs/job.$JOBTAG
TRC_TAG="zs:$SERVED"
mkdir -p "$STATE"
trc_log_open "$ROOT/logs/zs_eval.$JOBTAG.log"
trc_say "zs_eval.sh $MODEL: family $FAMILY, frame $FRAME, served $SERVED, revision ${REV:-<latest>}, smoke=$SMOKE dry=$DRY"
trc_say "host $(hostname -s) job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none} cluster $(trc_cluster) N(dispatcher)=${N:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

declare -A WPID=()
GEN_DEAD=0
# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$? pid
  trap - EXIT INT TERM HUP
  for pid in "${!WPID[@]}"; do trc_stop_tree "$pid" "generation worker" "${TRC_STOP_WAIT:-30}"; done
  if [ -n "${TRC_SERVER_PID:-}" ]; then
    trc_log "stopping vLLM (pid $TRC_SERVER_PID)"; trc_stop_tree "$TRC_SERVER_PID" vLLM "${TRC_STOP_WAIT:-120}"; TRC_SERVER_PID=""
  fi
  trc_heartbeat_stop   # first, so no heartbeat rewrite can resurrect a released claim
  trc_claim_release
  trc_lease_release
  trc_say "exit rc=$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'TRC_STOP_WAIT=20; trc_say "signal TERM"; exit 143' TERM
trap 'TRC_STOP_WAIT=20; trc_say "signal INT"; exit 130' INT
trap 'TRC_STOP_WAIT=20; trc_say "signal HUP"; exit 129' HUP

# ---------------------------------------------------------------------------- plan
if [ "$SMOKE" = 1 ]; then
  SPLITS=(e1_dev); CONDS=(cert_v1); SEEDS=(42)
  SELECT="--per-task ${TRC_SMOKE_PER_TASK:-5}"
else
  SPLITS=(e1_dev e1_test e2_test); CONDS=(ans_v1 ann_v1 cert_v1); SEEDS=(42 43 44)
  SELECT=""
fi
if [ -n "${TRC_SPLITS:-}" ]; then IFS=, read -ra SPLITS <<<"$TRC_SPLITS"; fi
if [ -n "${TRC_SEEDS:-}" ]; then IFS=, read -ra SEEDS <<<"$TRC_SEEDS"; fi
parquet_of() { echo "$TRC_DATA/public/$1.$(trc_public_cond "$2" "$FRAME").parquet"; }
expected_of() {  # split parquet
  if [ "$SMOKE" = 1 ]; then
    "$TRACE_CERT_PYTHON" -I -c '
import collections, sys
import pyarrow.parquet as pq
n = int(sys.argv[2])
tasks = collections.Counter(pq.read_table(sys.argv[1], columns=["task"]).column("task").to_pylist())
print(sum(min(n, c) for c in tasks.values()))' "$2" "${TRC_SMOKE_PER_TASK:-5}"
  else
    trc_receipt_value "$1" 'd["totals"]["accepted"]'
  fi
}
declare -A EXPECT=()
RUNS=()
for split in "${SPLITS[@]}"; do
  for cond in "${CONDS[@]}"; do
    pq=$(parquet_of "$split" "$cond")
    [ -n "${EXPECT[$split.$cond]:-}" ] || EXPECT[$split.$cond]=$(expected_of "$split" "$pq") \
      || { trc_say "FATAL: config error: cannot read the row count of $pq"; exit 4; }
  done
done
for seed in "${SEEDS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for cond in "${CONDS[@]}"; do
      RUNS+=("$split $cond $seed $(parquet_of "$split" "$cond") ${EXPECT[$split.$cond]}")
    done
  done
done
trc_say "plan: ${#RUNS[@]} runs (${SPLITS[*]} x ${CONDS[*]} x seeds ${SEEDS[*]}${SELECT:+, $SELECT}) -> $ROOT"
# plan.json: what the host watcher compares progress against (rows expected per run directory)
write_plan() {
  local entry split cond seed pq exp sep="" total=0 tmp=$ROOT/plan.json.tmp.$$
  {
    printf '{"model": "%s", "served": "%s", "family": "%s", "frame": "%s", "smoke": %s, "updated": "%s", "runs": [' \
      "$MODEL" "$SERVED" "$FAMILY" "$FRAME" "$([ "$SMOKE" = 1 ] && echo true || echo false)" "$(trc_iso)"
    for entry in "${RUNS[@]}"; do
      read -r split cond seed pq exp <<<"$entry"
      printf '%s\n  {"split": "%s", "cond": "%s", "seed": %s, "dir": "%s/%s/seed%s", "parquet": "%s", "expected": %s}' \
        "$sep" "$split" "$cond" "$seed" "$split" "$cond" "$seed" "$(basename "$pq")" "$exp"
      sep=","; total=$((total + exp))
    done
    printf '\n], "expected_total": %s}\n' "$total"
  } >"$tmp" && mv -f "$tmp" "$ROOT/plan.json"
}
write_plan

# ---------------------------------------------------------------------------- CPU preflight
CONFIG=$ROOT/trace_tool_conditions_v1.yaml
trc_render_conditions "$CONFIG" || exit 4
trc_say "conditions: $CONFIG"
for split in "${SPLITS[@]}"; do
  args=()
  for cond in "${CONDS[@]}"; do args+=(--cond "$(trc_public_cond "$cond" "$FRAME")"); done
  "$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" check-data --data "$TRC_DATA" --split "$split" "${args[@]}" \
    --cache "$TRACE_CERT_RUNS/eval/.data_sha256_cache.json" || { trc_say "FATAL: config error: frozen data check failed"; exit 4; }
done
CERT_SYSTEM=$TRACE_ROOT/src/trace_tasks/resources/prompts/experimental/certificates/cert_system_v1.txt
cases=()
for cond in "${CONDS[@]}"; do
  if [ "$cond" = ans_v1 ]; then sys=$TRACE_TOOL_ROOT/prompts/trace_system_prompt.txt; else sys=$CERT_SYSTEM; fi
  cases+=(--case "$cond=$(parquet_of e1_dev "$cond")=$sys")
done
if ! "$TRC_VENV/bin/python" "$TRC_JOBS_DIR/payload_check.py" --config "$CONFIG" --model-id "$MODEL" --served "$SERVED" \
    "${cases[@]}" >"$STATE/payload_check.jsonl"; then
  cat "$STATE/payload_check.jsonl"
  trc_say "FATAL: config error: the request payload check failed ($STATE/payload_check.jsonl)"; exit 4
fi
trc_say "payload check ok: system text + user text + one image per request, thinking off for Qwen3.5 only"
# trace-tool's own --dry-run rendering of the first e1_dev row per condition (kept for the record)
for cond in "${CONDS[@]}"; do
  # shellcheck disable=SC1091
  if ! (source "$TRC_VENV/bin/activate" && trace-tool-eval --dry-run --model "$SERVED" --model-id "$MODEL" \
        ${REV:+--model-revision "$REV"} --config "$CONFIG" --conditions "$cond" --split cert_e1_dev \
        --dataset-path "$(parquet_of e1_dev "$cond")" --limit 1) >"$STATE/tt_dryrun.$cond.txt" 2>&1 \
     || ! grep -q "tools=\[\]" "$STATE/tt_dryrun.$cond.txt"; then
    cat "$STATE/tt_dryrun.$cond.txt"
    trc_say "FATAL: config error: trace-tool-eval --dry-run failed for $cond ($STATE/tt_dryrun.$cond.txt)"; exit 4
  fi
done
trc_say "trace-tool --dry-run ok for ${CONDS[*]} ($STATE/tt_dryrun.*.txt)"

count_run() {  # split cond seed expected [python]
  "${5:-$TRACE_CERT_PYTHON}" "$TRC_JOBS_DIR/zs_report.py" count --responses "$ROOT/$1/$2/seed$3/responses.jsonl" --expected "$4"
}
PENDING=()
for entry in "${RUNS[@]}"; do
  read -r split cond seed pq exp <<<"$entry"
  stat=$(count_run "$split" "$cond" "$seed" "$exp")
  case "$stat" in *complete=1*) ;; *) PENDING+=("$entry") ;; esac
done
trc_say "pending: ${#PENDING[@]} of ${#RUNS[@]} runs"
if [ "$DRY" = 1 ]; then
  for entry in "${PENDING[@]+"${PENDING[@]}"}"; do trc_say "  todo: $entry"; done
  trc_say "dry run: no GPU touched"; exit 0
fi

trc_lease_take "$ROOT/.owner" || exit 3
trc_heartbeat_start   # lease now; restarted below once the claims exist (the loop copies the claim list)

# ---------------------------------------------------------------------------- GPUs + vLLM + generation
generate() {
  local total=${#PENDING[@]} next=0 par conc per fails=0 t0 util_done=0 unhealthy=0
  local pid key rc rcfile split cond seed pq exp
  par=${TRC_PAR:-3}; [ "$SMOKE" = 1 ] && par=1
  conc=${TRC_CONCURRENCY:-$((64 * TRC_N))}
  per=$(( (conc + par - 1) / par ))
  export TRC_W_ENDPOINT="http://127.0.0.1:$TRC_SERVER_PORT/v1" TRC_W_PORT=$TRC_SERVER_PORT TRC_W_SERVED=$SERVED \
    TRC_W_MODEL=$MODEL TRC_W_REV=$REV TRC_W_CONFIG=$CONFIG TRC_W_ROOT=$ROOT TRC_W_SELECT=$SELECT \
    TRC_W_TIMEOUT=${TRC_REQUEST_TIMEOUT:-1800} TRC_W_PASSES=${TRC_ERR_PASSES:-3}
  t0=$(date +%s)
  while :; do
    for pid in "${!WPID[@]}"; do
      kill -0 "$pid" 2>/dev/null && continue
      wait "$pid" 2>/dev/null
      key=${WPID[$pid]}; unset "WPID[$pid]"
      rc=$(cat "$STATE/gen.${key//\//.}.rc" 2>/dev/null || echo '?')
      if [ "$rc" = 0 ]; then trc_say "done $key"; else trc_say "INCOMPLETE $key (rc=$rc, log logs/gen.${key//\//.}.log)"; fails=$((fails + 1)); fi
    done
    if trc_serve_healthy "$TRC_SERVER_PORT" "$SERVED"; then unhealthy=0; else unhealthy=$((unhealthy + 1)); trc_log "health check failed ($unhealthy)"; fi
    if [ "$unhealthy" -ge 3 ] || ! kill -0 "$TRC_SERVER_PID" 2>/dev/null; then
      trc_flag "server_dead vLLM on port $TRC_SERVER_PORT stopped answering; last lines of its log follow"
      tail -n 40 "$VLLM_LOG"
      for pid in "${!WPID[@]}"; do trc_stop_tree "$pid" "generation worker" 30; unset "WPID[$pid]"; done
      GEN_DEAD=1
      break
    fi
    while [ ${#WPID[@]} -lt "$par" ] && [ "$next" -lt "$total" ]; do
      read -r split cond seed pq exp <<<"${PENDING[$next]}"
      next=$((next + 1))
      key="$split/$cond/seed$seed"
      rcfile=$STATE/gen.$split.$cond.seed$seed.rc
      rm -f "$rcfile"
      setsid bash "$SELF" --_gen-worker "$split" "$cond" "$seed" "$pq" "$exp" "$per" "$rcfile" \
        >>"$ROOT/logs/gen.$split.$cond.seed$seed.log" 2>&1 </dev/null 3>&- &
      WPID[$!]=$key
      trc_say "start $key ($exp rows, concurrency $per)"
    done
    if [ "$util_done" = 0 ] && [ $(( $(date +%s) - t0 )) -ge 60 ]; then
      util_done=1
      if [ ${#WPID[@]} -gt 0 ]; then trc_util_check || true; fi
    fi
    if [ ${#WPID[@]} -eq 0 ] && [ "$next" -ge "$total" ]; then break; fi
    sleep 15
  done
  [ "$util_done" = 1 ] || trc_say "UTILCHECK skipped: generation finished within 60 s"
  GEN_FAILS=$fails
}

GEN_FAILS=0
if [ ${#PENDING[@]} -gt 0 ]; then
  trc_gpu_discover || exit 4
  while read -r line; do trc_say "  $line"; done < <(trc_gpu_describe)
  [ -z "${N:-}" ] || [ "$N" = "$TRC_N" ] || trc_say "note: dispatcher N=$N, this step sees $TRC_N GPUs (using $TRC_N)"
  need_gb=$(trc_model_min_vram_gb "$MODEL")
  if [ "$TRC_MIN_MIB" -lt $((need_gb * 1000)) ]; then
    trc_say "FATAL: config error: $MODEL needs >= ${need_gb} GB per card; the smallest card here has ${TRC_MIN_MIB} MiB"; exit 4
  fi
  if busy=$(trc_gpu_busy); then trc_say "GPUs busy, not starting: $busy"; exit 3; fi
  if busy=$(trc_foreign_claimed); then trc_say "GPUs claimed, not starting: $busy"; exit 3; fi
  TRC_CLAIM_WHO="zs_eval.sh $MODEL$([ "$SMOKE" = 1 ] && echo ' --smoke')"
  TRC_CLAIM_RUN=$ROOT; TRC_CLAIM_TASK="trc:zs_eval:$SERVED"
  trc_claim_take >"$STATE/claims.txt" 2>&1; rc=$?
  trc_say "$(tr '\n' ' ' <"$STATE/claims.txt")"
  [ "$rc" -eq 0 ] || exit 3
  trc_heartbeat_stop; trc_heartbeat_start
  sleep 10
  if busy=$(trc_gpu_busy); then trc_say "GPUs became busy after claiming, releasing: $busy"; exit 3; fi
  PORT=$(trc_pick_port) || { trc_say "no free port"; exit 5; }
  VLLM_LOG=$ROOT/logs/vllm.$JOBTAG.log
  # shellcheck disable=SC1091
  source "$TRC_VENV/bin/activate"
  trc_serve_start "$MODEL" "$REV" "$SERVED" "$PORT" "$TRC_N" "$VLLM_LOG" || exit 5
  generate
  trc_serve_stop
  trc_heartbeat_stop; trc_claim_release; trc_heartbeat_start   # GPUs free again; keep the run lease
  trc_say "generation finished: ${GEN_FAILS} incomplete runs, server_dead=$GEN_DEAD; GPUs released"
fi

# ---------------------------------------------------------------------------- scoring (CPU)
SCORE_FAILS=0
SCORE_FILES=()
declare -A SPLIT_FILES=()
for entry in "${RUNS[@]}"; do
  read -r split cond seed pq exp <<<"$entry"
  run=$ROOT/$split/$cond/seed$seed
  [ -f "$run/responses.jsonl" ] || continue
  stat=$(count_run "$split" "$cond" "$seed" "$exp")
  case "$stat" in *complete=1*) ;; *) trc_log "not scoring $split/$cond/seed$seed: $stat"; continue ;; esac
  if ! { [ -f "$run/cert/score_receipt.json" ] && [ "$run/cert/cert_scores.jsonl" -nt "$run/responses.jsonl" ]; }; then
    if (cd "$TRACE_ROOT" && nice -n 5 "$TRACE_CERT_PYTHON" -m trace_tasks.experimental.certificates score \
          --responses "$run/responses.jsonl" --store "$TRC_DATA/private/$split.extensions_v1.jsonl.zst" \
          --receipt "$TRC_DATA/receipts/$split.json" --out "$run/cert") >>"$ROOT/logs/score.log" 2>&1; then
      trc_say "scored $split/$cond/seed$seed"
    else
      trc_flag "score_failed $split/$cond/seed$seed (see logs/score.log)"; SCORE_FAILS=$((SCORE_FAILS + 1)); continue
    fi
  fi
  SCORE_FILES+=("$run/cert/cert_scores.jsonl")
  SPLIT_FILES[$split]="${SPLIT_FILES[$split]:-} $run/cert/cert_scores.jsonl"
done
summarize() {  # OUT FILES...
  local out=$1; shift
  (cd "$TRACE_ROOT" && "$TRACE_CERT_PYTHON" -m trace_tasks.experimental.certificates summarize --scores "$@" --out "$out") \
    >>"$ROOT/logs/score.log" 2>&1 || { trc_flag "summarize_failed $out"; SCORE_FAILS=$((SCORE_FAILS + 1)); }
}
if [ ${#SCORE_FILES[@]} -gt 0 ]; then
  summarize "$ROOT/summary.json" "${SCORE_FILES[@]}"
  for split in "${!SPLIT_FILES[@]}"; do
    read -ra files <<<"${SPLIT_FILES[$split]}"
    summarize "$ROOT/summary.$split.json" "${files[@]}"
  done
  "$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" report --model-root "$ROOT" --splits "$(IFS=,; echo "${SPLITS[*]}")" \
    --out "$ROOT/summary.md" >/dev/null || trc_flag "report_failed $ROOT/summary.md"
  trc_say "summary: $ROOT/summary.md"
fi

# frame obedience: always for the smoke; for a full run on e1_dev cert_v1 seed 42 (logged)
diag_run=$ROOT/e1_dev/cert_v1/seed42
if [ -f "$diag_run/cert/cert_scores.jsonl" ]; then
  if [ "$SMOKE" = 1 ]; then
    while IFS= read -r line; do trc_say "$line"; done < <(sed -n '/^| cond/,/^$/p' "$ROOT/summary.md" 2>/dev/null)
  fi
  diag=$("$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" frame-diag --run "$diag_run" \
    --store "$TRC_DATA/private/e1_dev.extensions_v1.jsonl.zst" --out "$diag_run/cert/frame_diag.json" 2>&1) \
    || trc_flag "frame_diag_failed $diag_run"
  if [ "$SMOKE" = 1 ]; then
    while IFS= read -r line; do trc_say "$line"; done <<<"$diag"
  else
    printf '%s\n' "$diag"; trc_say "$(grep '^frame obedience' <<<"$diag")"
  fi
fi

# ---------------------------------------------------------------------------- validation
vargs=(--model-root "$ROOT" --data "$TRC_DATA" --splits "$(IFS=,; echo "${SPLITS[*]}")" --conds "$(IFS=,; echo "${CONDS[*]}")"
       --seeds "$(IFS=,; echo "${SEEDS[*]}")" --frame "$FRAME" --out "$ROOT/validation.json")
if [ "$SMOKE" = 1 ]; then vargs+=(--expect-rows "${EXPECT[e1_dev.cert_v1]}"); fi
vout=$("$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" validate "${vargs[@]}" 2>&1); vrc=$?
while IFS= read -r line; do trc_say "$line"; done <<<"$vout"
if [ "$vrc" -ne 0 ] || [ "$GEN_DEAD" = 1 ] || [ "$SCORE_FAILS" -gt 0 ]; then
  trc_say "INCOMPLETE: rerun the same task line; finished rows are kept"
  exit 1
fi
trc_say "COMPLETE: $SERVED $([ "$SMOKE" = 1 ] && echo smoke || echo full) evaluation validated"
exit 0
