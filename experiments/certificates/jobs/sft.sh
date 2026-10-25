#!/usr/bin/env bash
# experiments/certificates/jobs/sft.sh MODEL_SIZE [--smoke] [--dry-run]      (MODEL_SIZE: 3b | 7b)
#
# DESIGN 7.2 M2 structured SFT of one model, written as ONE use-gpu task line: it uses every GPU
# of the allocation it lands on (N = the GPUs this step sees; N must divide the global batch 32).
#
#   source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT" && bash experiments/certificates/jobs/sft.sh 3b [--smoke]
#
# Stages (each resumes on a rerun of the same task line):
#  1. CPU preflight (sft.py check): data/receipt sha256s, base snapshot, every SFT user text against
#     the eval public prompt hash of its instance, every target against make_sft_target, exact
#     token lengths <= 4096, prompt token ids identical to EasyR1's RLHFDataset (trace-rlvr).
#  2. Training: torchrun --nproc_per_node=N sft.py train (FSDP, fp32 master + bf16 compute,
#     FlashAttention-2, ViT frozen, assistant-only loss, global batch 32); saves a plain HF dir
#     every half epoch under ckpt/step_XXXXXX and resumes from the last save (state/).
#  3. Stop-rule generation: for every saved checkpoint, greedy 2048-token E1-dev generation with
#     offline vLLM 0.10.2 in the training env (TP=1, one process per GPU, rows sharded), merged
#     into dev/step_XXXXXX/{responses.jsonl,run_config.json}. GPUs are released afterwards.
#  4. Scoring on CPU with the certificate scorer (trace_cert env): dev/step_XXXXXX/cert/.
#  5. jobs/sft_select.py: the pre-registered rule picks the checkpoint; writes selected -> ckpt/...,
#     dev_curve.json and receipt.json (data shas, model revision, env lock, git shas, dev curve).
# Outputs: $TRACE_CERT_RUNS/sft/<tag>/ (sft_smoke/<tag>/ with --smoke), tag qwen2.5-vl-3b|7b:
#   plan.json check.json runs.jsonl train_log.jsonl train_done.json ckpt/ state/ dev/ selected
#   dev_curve.json receipt.json logs/{sft.<job>.log,train.<job>.log,gen.*.log,score.log}
#
# --smoke: 20 optimizer steps on 256 rows (64 per adapter), one save (step 20), dev eval on 32 rows
#          (8 per task); everything else as the full run.
# --dry-run: CPU preflight and plan only; touches no GPU.
#
# Exit: 0 done (checkpoint selected); 1 incomplete (training or generation failed, scoring failed;
#       rerun resumes); 2 usage; 3 GPUs busy or the run is owned by another live job;
#       4 configuration mismatch (data, env, VRAM/N/disk, config changed under an existing run);
#       130/143 signalled.
# Environment (host/env.sh): TRACE_CERT_RUNS TRACE_TOOL_ROOT HF_HOME TRACE_CERT_PYTHON TRC_CLAIM_DIR
#   TRACE_RLVR_ROOT (EasyR1 token-identity check; required) and TRC_RL_PYTHON (python of the
#   training env: torch 2.8 / vLLM 0.10.2; default: the trace_rl env next to TRACE_CERT_PYTHON's env,
#   the same rule as jobs/rl.sh).
# Knobs: TRC_SFT_CONFIG (another config file), TRC_SFT_OUT_ROOT (instead of $TRACE_CERT_RUNS/sft[_smoke]),
#   TRC_SFT_PER_DEVICE_BATCH (micro-batch; never changes the math), TRC_SFT_MIN_FREE_GB (disk floor,
#   default 150 for 3B / 350 for 7B, 40 for --smoke), TRC_SFT_POLL (s between training checks, 20).
#   CPU tests only: TRC_SFT_GEN_CMD (stand-in for
#   `sft.py gen`), TRC_SFT_DEVICE=cpu, TRC_SFT_ATTN, TRC_SFT_SKIP_RL_CHECK=1, TRC_NVIDIA_SMI (common.sh).
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=common.sh
source "$(dirname "$SELF")/common.sh"

usage() { echo "usage: sft.sh 3b|7b [--smoke] [--dry-run]" >&2; exit 2; }
SIZE=""; SMOKE=0; DRY=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage ;;
    3b|7b) [ -z "$SIZE" ] || usage; SIZE=$a ;;
    *) echo "unknown argument $a" >&2; usage ;;
  esac
done
[ -n "$SIZE" ] || usage

trc_require_env
trc_node_allowed || { echo "FATAL: config error: $(hostname -s) is excluded by fleet policy" >&2; exit 4; }
TRC_RL_PYTHON=${TRC_RL_PYTHON:-$(cd "$(dirname "$TRACE_CERT_PYTHON")/../.." && pwd)/trace_rl/bin/python}
[ -x "$TRC_RL_PYTHON" ] || { echo "FATAL: config error: training env python $TRC_RL_PYTHON does not exist (set TRC_RL_PYTHON)" >&2; exit 4; }
export TRC_RL_PYTHON
RL_PY=$TRC_RL_PYTHON
# Triton (vLLM generation, compiled kernels) builds small C launchers against Python.h at run time;
# nodes whose system python has no -devel headers (grn067) need the env's copy via CPATH (as rl.sh)
pyinc=$("$RL_PY" -I -c '
import os, sys, sysconfig
base = sysconfig.get_paths()["include"]
own = os.path.join(sys.prefix, "include", "python%d.%d" % sys.version_info[:2])
print("" if os.path.isfile(os.path.join(base, "Python.h")) else own if os.path.isfile(os.path.join(own, "Python.h")) else "MISSING " + base)') \
  || pyinc="MISSING (python failed)"
case "$pyinc" in
  MISSING*) echo "FATAL: config error: no Python.h for $RL_PY ($pyinc; nor in the env's include/): triton cannot compile its launchers" >&2; exit 4 ;;
  "") ;;
  *) export CPATH="$pyinc${CPATH:+:$CPATH}" ;;
esac
CONFIG=${TRC_SFT_CONFIG:-$TRC_CERT_DIR/configs/sft_$SIZE.yaml}
[ -f "$CONFIG" ] || { echo "FATAL: config error: no config $CONFIG" >&2; exit 4; }
if [ -z "${TRACE_RLVR_ROOT:-}" ] && [ "${TRC_SFT_SKIP_RL_CHECK:-0}" != 1 ]; then
  echo "FATAL: config error: TRACE_RLVR_ROOT is not set (needed for the EasyR1 token-identity check)" >&2; exit 4
fi
export PYTHONPATH="$TRACE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SMOKE_ARG=(); [ "$SMOKE" = 1 ] && SMOKE_ARG=(--smoke)
sftpy() { "$RL_PY" "$TRC_CERT_DIR/sft.py" "$@"; }

PLAN_JSON=$(sftpy plan --config "$CONFIG" --data "$TRC_DATA" "${SMOKE_ARG[@]}" 2>&1) \
  || { echo "FATAL: config error: sft.py plan failed: $PLAN_JSON" >&2; exit 4; }
pj() {  # python expression over the plan dict d -> printed (lists space-separated)
  printf '%s' "$PLAN_JSON" | "$RL_PY" -I -c '
import json, sys
d = json.load(sys.stdin)
v = eval(sys.argv[1])
print(" ".join(str(x) for x in v) if isinstance(v, list) else ("" if v is None else v))' "$1"
}
TAG=$(pj 'd["model"]["tag"]')
read -ra SAVES <<<"$(pj 'd["save_steps"]')"
DEV_PER_TASK=$(pj 'd["dev_rows_per_task"]')
MIN_CARD=$(pj 'd["resources"]["min_card_mib"]')
MIN_TOTAL=$(pj 'd["resources"]["min_total_mib"]')
GLOBAL_BATCH=$(pj 'd["global_batch"]')
DEV_STORE=$TRC_DATA/$(pj 'd["dev"]["store"]')
DEV_RECEIPT=$TRC_DATA/$(pj 'd["dev"]["receipt"]')
if [ "$SMOKE" = 1 ]; then OUT_BASE=$TRACE_CERT_RUNS/sft_smoke; else OUT_BASE=$TRACE_CERT_RUNS/sft; fi
ROOT=${TRC_SFT_OUT_ROOT:-$OUT_BASE}/$TAG
JOBTAG=${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}.$(date +%m%d_%H%M%S)
STATE=$ROOT/logs/job.$JOBTAG
TRC_TAG="sft:$TAG"
mkdir -p "$STATE"
trc_log_open "$ROOT/logs/sft.$JOBTAG.log"
trc_say "sft.sh $SIZE: config $CONFIG, tag $TAG, smoke=$SMOKE dry=$DRY, saves ${SAVES[*]}, out $ROOT"
trc_say "host $(hostname -s) job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none} cluster $(trc_cluster) N(dispatcher)=${N:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} python $RL_PY"

TRAIN_PID=""
GEN_PIDS=()
# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$? pid
  trap - EXIT INT TERM HUP
  if [ -n "$TRAIN_PID" ]; then trc_stop_tree "$TRAIN_PID" "training" "${TRC_STOP_WAIT:-120}"; fi
  for pid in ${GEN_PIDS[@]+"${GEN_PIDS[@]}"}; do trc_stop_tree "$pid" "generation worker" "${TRC_STOP_WAIT:-60}"; done
  trc_heartbeat_stop
  trc_claim_release
  trc_lease_release
  trc_say "exit rc=$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'TRC_STOP_WAIT=30; trc_say "signal TERM"; exit 143' TERM
trap 'TRC_STOP_WAIT=30; trc_say "signal INT"; exit 130' INT
trap 'TRC_STOP_WAIT=30; trc_say "signal HUP"; exit 129' HUP

# ---------------------------------------------------------------------------- CPU preflight
printf '%s\n' "$PLAN_JSON" >"$ROOT/plan.json.tmp.$$" && mv -f "$ROOT/plan.json.tmp.$$" "$ROOT/plan.json"
RL_CHECK=(); [ -n "${TRACE_RLVR_ROOT:-}" ] && [ "${TRC_SFT_SKIP_RL_CHECK:-0}" != 1 ] && RL_CHECK=(--rlvr-root "$TRACE_RLVR_ROOT")
if ! nice -n 5 "$RL_PY" "$TRC_CERT_DIR/sft.py" check --config "$CONFIG" --data "$TRC_DATA" "${SMOKE_ARG[@]}" --out "$ROOT" \
     "${RL_CHECK[@]}" >"$STATE/check.txt" 2>&1; then
  tail -n 20 "$STATE/check.txt"
  trc_say "FATAL: config error: preflight failed ($ROOT/check.json, $STATE/check.txt)"; exit 4
fi
trc_say "preflight ok: $("$RL_PY" -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
L = d["lengths"]; e = d.get("easyr1_equivalence")
eq = "skipped" if e is None else ("identical" if e["sft_rows"]["all_equal"] and e["e1_dev_rows"]["all_equal"] else "DIFFERENT")
print("%d rows, max %d tokens (prompt %d, target %d), %d label tokens; EasyR1 prompt ids %s"
      % (L["rows"], L["max_total"], L["max_prompt"], L["max_target"], L["label_tokens"], eq))' "$ROOT/check.json")"

ckpt_done() { [ -f "$ROOT/ckpt/$(printf 'step_%06d' "$1")/sft_checkpoint.json" ]; }
gen_done() { [ -f "$ROOT/dev/$(printf 'step_%06d' "$1")/responses.jsonl" ] && [ -f "$ROOT/dev/$(printf 'step_%06d' "$1")/run_config.json" ]; }
score_done() {
  local run; run=$ROOT/dev/$(printf 'step_%06d' "$1")
  [ -f "$run/cert/summary.json" ] && [ -f "$run/cert/score_receipt.json" ] && [ "$run/cert/cert_scores.jsonl" -nt "$run/responses.jsonl" ]
}
NEED_GEN=(); NEED_SCORE=()
for s in "${SAVES[@]}"; do
  gen_done "$s" || NEED_GEN+=("$s")
  score_done "$s" || NEED_SCORE+=("$s")
done
TRAIN_NEEDED=1; [ -f "$ROOT/train_done.json" ] && TRAIN_NEEDED=0
trc_say "status: training $([ "$TRAIN_NEEDED" = 1 ] && echo pending || echo done); generation pending for ${#NEED_GEN[@]} of ${#SAVES[@]} saves; scoring pending for ${#NEED_SCORE[@]}"
if [ "$DRY" = 1 ]; then trc_say "dry run: no GPU touched"; exit 0; fi

trc_lease_take "$ROOT/.owner" || exit 3
trc_heartbeat_start

# ---------------------------------------------------------------------------- GPUs
GPUS_HELD=0
acquire_gpus() {  # PURPOSE (train|gen): discover, check, claim; exits 3/4 like zs_eval.sh
  local purpose=$1 need_card need_total line busy total=0 m free_gb min_free rc
  trc_gpu_discover || exit 4
  while read -r line; do trc_say "  $line"; done < <(trc_gpu_describe)
  [ -z "${N:-}" ] || [ "$N" = "$TRC_N" ] || trc_say "note: dispatcher N=$N, this step sees $TRC_N GPUs (using $TRC_N)"
  for m in "${TRC_GPU_MIB[@]}"; do total=$((total + m)); done
  if [ "$purpose" = train ]; then
    need_card=$MIN_CARD; need_total=$MIN_TOTAL
    if [ $((GLOBAL_BATCH % TRC_N)) -ne 0 ]; then trc_say "FATAL: config error: N=$TRC_N GPUs does not divide the global batch $GLOBAL_BATCH"; exit 4; fi
    if [ "$SMOKE" = 1 ]; then min_free=${TRC_SFT_MIN_FREE_GB:-40}; elif [ "$SIZE" = 7b ]; then min_free=${TRC_SFT_MIN_FREE_GB:-350}; else min_free=${TRC_SFT_MIN_FREE_GB:-150}; fi
    free_gb=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4 / 1048576)}')
    if [ "${free_gb:-0}" -lt "$min_free" ]; then trc_say "FATAL: config error: ${free_gb} GB free under $ROOT < ${min_free} GB"; exit 4; fi
  else
    if [ "$SIZE" = 7b ]; then need_card=40000; else need_card=20000; fi; need_total=0
  fi
  if [ "$TRC_MIN_MIB" -lt "$need_card" ] || [ "$total" -lt "$need_total" ]; then
    trc_say "FATAL: config error: $purpose needs >= ${need_card} MiB per card and >= ${need_total} MiB in total; this allocation has ${TRC_MIN_MIB} MiB min, ${total} MiB total"; exit 4
  fi
  if busy=$(trc_gpu_busy); then trc_say "GPUs busy, not starting: $busy"; exit 3; fi
  if busy=$(trc_foreign_claimed); then trc_say "GPUs claimed, not starting: $busy"; exit 3; fi
  TRC_CLAIM_WHO="sft.sh $SIZE$([ "$SMOKE" = 1 ] && echo ' --smoke') ($purpose)"
  TRC_CLAIM_RUN=$ROOT; TRC_CLAIM_TASK="trc:sft:$TAG"
  trc_claim_take >"$STATE/claims.$purpose.txt" 2>&1; rc=$?
  trc_say "$(tr '\n' ' ' <"$STATE/claims.$purpose.txt")"
  [ "$rc" -eq 0 ] || exit 3
  trc_heartbeat_stop; trc_heartbeat_start
  sleep "${TRC_CLAIM_SETTLE:-10}"
  if busy=$(trc_gpu_busy); then trc_say "GPUs became busy after claiming, releasing: $busy"; exit 3; fi
  GPUS_HELD=1
}
release_gpus() {
  [ "$GPUS_HELD" = 1 ] || return 0
  trc_heartbeat_stop; trc_claim_release; trc_heartbeat_start   # keep the run lease
  GPUS_HELD=0
  trc_say "GPUs released"
}
# Training utilisation check (gpu_run skill: every selected GPU busy under load): three samples
# 5 s apart once the first optimizer step is logged; OK when every GPU shows > 0% in some sample and
# holds > TRC_SFT_UTIL_MIB (default 4000) MiB.
sft_util_check() {
  local s line gpu util used bad=0
  declare -A peak=() mem=()
  for s in 1 2 3; do
    while read -r line; do
      [ -n "$line" ] || continue
      trc_log "UTILCHECK sample$s $line"
      gpu=${line%% *}; util=${line#*util=}; util=${util%%%*}; used=${line#*mem=}; used=${used%%/*}
      [[ $util =~ ^[0-9]+$ ]] || util=0
      [ "${peak[$gpu]:-0}" -lt "$util" ] && peak[$gpu]=$util
      mem[$gpu]=${used%%.*}
    done < <(trc_gpu_util)
    [ "$s" = 3 ] || sleep 5
  done
  for gpu in "${!mem[@]}"; do
    if [ "${peak[$gpu]:-0}" -eq 0 ] || [ "${mem[$gpu]:-0}" -lt "${TRC_SFT_UTIL_MIB:-4000}" ]; then
      trc_flag "util_low $gpu peak_util=${peak[$gpu]:-0}% mem=${mem[$gpu]:-0}MiB"; bad=1
    fi
  done
  [ ${#mem[@]} -gt 0 ] || { trc_flag "util_low no_samples"; bad=1; }
  [ "$bad" = 0 ] && trc_say "UTILCHECK ok: $(trc_gpu_util | tr '\n' ' ')"
  return "$bad"
}
log_lines() { [ -f "$1" ] && wc -l <"$1" || echo 0; }

# ---------------------------------------------------------------------------- training
if [ "$TRAIN_NEEDED" = 1 ]; then
  acquire_gpus train
  LAYOUT=$(sftpy plan --config "$CONFIG" --data "$TRC_DATA" "${SMOKE_ARG[@]}" --world-size "$TRC_N" 2>&1) \
    || { trc_say "FATAL: config error: $LAYOUT"; exit 4; }
  trc_say "layout: $(printf '%s' "$LAYOUT" | "$RL_PY" -I -c 'import json,sys; print(json.load(sys.stdin)["layout"])')"
  TRAIN_LOG=$ROOT/logs/train.$JOBTAG.log
  log0=$(log_lines "$ROOT/train_log.jsonl")
  trc_say "training: torchrun --nproc_per_node=$TRC_N on GPUs $CUDA_VISIBLE_DEVICES (log $TRAIN_LOG)"
  PBATCH=(); [ -n "${TRC_SFT_PER_DEVICE_BATCH:-}" ] && PBATCH=(--per-device-batch "$TRC_SFT_PER_DEVICE_BATCH")
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=${NCCL_DEBUG:-WARN} TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} \
    setsid "$RL_PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$TRC_N" \
      "$TRC_CERT_DIR/sft.py" train --config "$CONFIG" --data "$TRC_DATA" --out "$ROOT" "${SMOKE_ARG[@]}" "${PBATCH[@]}" \
      >>"$TRAIN_LOG" 2>&1 </dev/null 3>&- &
  TRAIN_PID=$!
  t0=$(date +%s); util_done=0; last_note=$t0
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${TRC_SFT_POLL:-20}"
    now=$(date +%s)
    if [ "$util_done" = 0 ] && [ "$(log_lines "$ROOT/train_log.jsonl")" -gt "$log0" ]; then
      util_done=1; sft_util_check || true
    fi
    if [ $((now - last_note)) -ge 600 ]; then
      last_note=$now
      trc_say "training: $(tail -n 1 "$ROOT/train_log.jsonl" 2>/dev/null | cut -c1-200)"
    fi
  done
  wait "$TRAIN_PID"; rc=$?; TRAIN_PID=""
  [ "$util_done" = 1 ] || trc_log "UTILCHECK skipped: no step logged by this launch"
  if [ "$rc" -ne 0 ] || [ ! -f "$ROOT/train_done.json" ]; then
    tail -n 40 "$TRAIN_LOG"
    trc_flag "train_failed rc=$rc after $(( $(date +%s) - t0 ))s (log $TRAIN_LOG); rerun resumes from the last save"
    [ "$rc" -eq 4 ] && exit 4
    exit 1
  fi
  trc_say "training finished in $(( $(date +%s) - t0 ))s: $(ls "$ROOT/ckpt" | tr '\n' ' ')"
fi
for s in "${SAVES[@]}"; do ckpt_done "$s" || { trc_say "FATAL: checkpoint step $s missing after training"; exit 1; }; done

# ---------------------------------------------------------------------------- stop-rule generation (GPU)
if [ -n "${TRC_SFT_GEN_CMD:-}" ]; then read -ra GEN_CMD <<<"$TRC_SFT_GEN_CMD"; else GEN_CMD=("$RL_PY" "$TRC_CERT_DIR/sft.py" gen); fi
ROWS_ARG=(); [ -n "$DEV_PER_TASK" ] && ROWS_ARG=(--rows-per-task "$DEV_PER_TASK")
gen_ckpt() {  # STEP -> 0 when dev/step_XXXXXX/responses.jsonl is complete
  local s=$1 name run ckpt i out rc=0 pids=() t0 vcache
  name=$(printf 'step_%06d' "$s"); run=$ROOT/dev/$name; ckpt=$ROOT/ckpt/$name
  mkdir -p "$run/shards"
  # one vLLM cache root per shard on node-local disk: concurrent shards compiling into the shared
  # ~/.cache/vllm/torch_compile_cache race on its renames (BackendCompilerFailed / FileNotFoundError)
  vcache=${TRC_SFT_VLLM_CACHE:-${TMPDIR:-/tmp}/trc_vllm_${SLURM_JOB_ID:-local}_$$}
  local tok=(); IFS=, read -ra tok <<<"$CUDA_VISIBLE_DEVICES"
  t0=$(date +%s)
  for i in "${!tok[@]}"; do
    out=$(printf '%s/shards/shard_%02dof%02d.jsonl' "$run" "$i" "${#tok[@]}")
    # the training env's bin/ first on PATH: vLLM / torch.compile JIT may spawn `ninja` from there
    CUDA_VISIBLE_DEVICES=${tok[$i]} VLLM_WORKER_MULTIPROC_METHOD=spawn PATH="$(dirname "$RL_PY"):$PATH" VLLM_CACHE_ROOT=$vcache/shard$i \
      setsid "${GEN_CMD[@]}" --config "$CONFIG" --data "$TRC_DATA" --ckpt "$ckpt" --shard "$i" --num-shards "${#tok[@]}" \
        "${ROWS_ARG[@]}" --out "$out" >>"$ROOT/logs/gen.$name.$i.log" 2>&1 </dev/null 3>&- &
    pids+=("$!"); GEN_PIDS+=("$!")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      rc=1; trc_flag "gen_failed $name shard $i (log logs/gen.$name.$i.log)"
      grep -aE '(Error|Exception)(\[[^]]*\])?: ' "$ROOT/logs/gen.$name.$i.log" | sed 's/^.*ERROR [^]]*] //' | awk '!seen[$0]++' | tail -n 6
      tail -n 5 "$ROOT/logs/gen.$name.$i.log"
    fi
  done
  GEN_PIDS=()
  [ -n "${TRC_SFT_VLLM_CACHE:-}" ] || rm -rf "$vcache"
  [ "$rc" = 0 ] || return 1
  if ! sftpy merge-gen --config "$CONFIG" --data "$TRC_DATA" --ckpt "$ckpt" --run "$run" --num-shards "${#tok[@]}" "${ROWS_ARG[@]}" \
       --label "sft-$TAG" >>"$ROOT/logs/merge.log" 2>&1; then
    trc_flag "merge_failed $name (logs/merge.log)"; return 1
  fi
  trc_say "generated $name: $(wc -l <"$run/responses.jsonl") rows in $(( $(date +%s) - t0 ))s on ${#tok[@]} GPUs"
}
GEN_FAILS=0
NEED_GEN=(); for s in "${SAVES[@]}"; do gen_done "$s" || NEED_GEN+=("$s"); done
if [ ${#NEED_GEN[@]} -gt 0 ]; then
  [ "$GPUS_HELD" = 1 ] || acquire_gpus gen
  for s in "${NEED_GEN[@]}"; do gen_ckpt "$s" || GEN_FAILS=$((GEN_FAILS + 1)); done
fi
release_gpus

# ---------------------------------------------------------------------------- scoring (CPU)
SCORE_FAILS=0
for s in "${SAVES[@]}"; do
  gen_done "$s" || continue
  score_done "$s" && continue
  run=$ROOT/dev/$(printf 'step_%06d' "$s")
  # --allow-settings-mismatch: the stop rule decodes greedily (DESIGN 7.2), not with trace-tool's
  # "trace" preset; the scorer records the difference and sft_select.py refuses any other one
  if (cd "$TRACE_ROOT" && nice -n 5 "$TRACE_CERT_PYTHON" -m trace_tasks.experimental.certificates score \
        --responses "$run/responses.jsonl" --store "$DEV_STORE" --receipt "$DEV_RECEIPT" --out "$run/cert" \
        --allow-settings-mismatch) >>"$ROOT/logs/score.log" 2>&1; then
    trc_say "scored step $s: $("$TRACE_CERT_PYTHON" -I -c '
import json, sys
a = json.load(open(sys.argv[1]))["all"][0]
print(" ".join(f"{k}={a[k]:.4f}" for k in ("A", "G", "M", "F", "joint") if isinstance(a.get(k), float)), "n=%d" % a["n"])' "$run/cert/summary.json")"
  else
    trc_flag "score_failed step $s (logs/score.log)"; SCORE_FAILS=$((SCORE_FAILS + 1))
  fi
done
if [ "$GEN_FAILS" -gt 0 ] || [ "$SCORE_FAILS" -gt 0 ]; then
  trc_say "INCOMPLETE: $GEN_FAILS generation and $SCORE_FAILS scoring failures; rerun the same task line (finished stages are kept)"
  exit 1
fi

# ---------------------------------------------------------------------------- stop rule + receipt
sel_out=$("$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/sft_select.py" --root "$ROOT" --data "$TRC_DATA" \
          ${TRACE_RLVR_ROOT:+--rlvr-root "$TRACE_RLVR_ROOT"} --tool-root "$TRACE_TOOL_ROOT" 2>&1); rc=$?
while IFS= read -r line; do trc_say "$line"; done <<<"$sel_out"
if [ "$rc" -ne 0 ]; then
  trc_say "INCOMPLETE: sft_select.py exit $rc"
  [ "$rc" -eq 4 ] && exit 4
  exit 1
fi
trc_say "COMPLETE: $TAG $([ "$SMOKE" = 1 ] && echo smoke || echo full) SFT; selected -> $(readlink "$ROOT/selected"); receipt $ROOT/receipt.json"
exit 0
