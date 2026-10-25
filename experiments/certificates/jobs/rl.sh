#!/usr/bin/env bash
# experiments/certificates/jobs/rl.sh --model 3b|7b --arm A|G|P|P_gated|G_gated --seed S --sft-dir DIR [--smoke] [--dry-run]
#
# DESIGN 7.3 M3 RL, written as ONE use-gpu task line: it uses every GPU of the allocation it lands on
# (N must be 2 or 4) and runs experiments/certificates/train_rl.py in the RL env (TRACE's vendored
# EasyR1, read-only from $TRACE_RLVR_ROOT):
#
#   source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT" && \
#     bash experiments/certificates/jobs/rl.sh --model 3b --arm P --seed 1 --sft-dir "$TRACE_CERT_RUNS/sft/3b/final"
#
# Full run -> $TRACE_CERT_RUNS/rl/<model>/<arm>/s<seed>/: 300 GRPO steps (64 prompts x 8 rollouts),
# save every 25 steps (preemptible nodes: often enough for a save every <= 45 min), steps 100 and 200
# hard-linked into keep/ as soon as they are saved, then steps 100/200/300 merged to bf16 HF dirs in
# hf/global_step_<k>/ (CPU, vendored model_merger.py; processor + chat template from the SFT dir) and
# the final one loaded and decoded once with the serving env's vLLM. Resumable: a rerun of the same
# line continues from the last save (EasyR1 find_last_checkpoint), refuses a run directory that was
# started with other inputs or another GPU type, and exits 0 at once when the run is complete.
#
# --smoke: no --sft-dir; the BASE model of the spec; rollout batch 8 x n 4; one step, a relaunch that
#          must resume from global_step_1, a second step; 8-row E1-dev validation pass (EasyR1 always
#          validates after the last step); merge + vLLM load check; prints step time, memory and the
#          reward components (A, G, M, F, joint) per adapter plus the overall every arm would give.
#          Output: $TRACE_CERT_RUNS/rl_smoke/<model>_<arm>_s<seed>/<stamp>_<node>/ (FSDP checkpoints
#          deleted afterwards). TRC_RL_SMOKE_ARGS passes extra train_rl.py smoke options (--steps K,
#          --batch B, --n R, --keep-ckpt, --no-resume-test) for timing probes.
# --dry-run: CPU checks only (imports, vendor manifest, data shas, store join, model dir, prompt
#          clip ratio, arm configs) after the GPU count / VRAM gate; no claim, no training.
#
# Exit: 0 done; 1 incomplete (preempted, trainer error, merge / vLLM check failed; rerun resumes);
#       2 usage; 3 GPUs busy, claimed by another campaign or the run owned by another live job;
#       4 configuration mismatch (N not in {2,4}; VRAM below the spec's per-card need: 80 GB, 7B on
#       2 cards 120 GB; check failed; run identity or GPU type differs from the existing run dir);
#       130/143 signalled (the run resumes from its last save).
# Environment: host/env.sh (TRACE_CERT_RUNS, TRACE_RLVR_ROOT, HF_HOME, TRC_CLAIM_DIR, ...).
#   TRC_RL_PYTHON  python of the RL env (default: the trace_rl env next to $TRACE_CERT_PYTHON's env)
#   TRC_RL_ALLOW_HW_CHANGE=1  resume a run on a different GPU type (breaks per-class matching)
#   TRC_RL_STALL_MIN  minutes without a new step before a FLAG stall line (default 45)
#   TRC_RL_ROOT / TRC_RL_SMOKE_ROOT  run roots (default $TRACE_CERT_RUNS/rl, $TRACE_CERT_RUNS/rl_smoke)
# Pair mode (jobs/rl_pair.sh runs two of these at once on disjoint GPU pairs of one 4-GPU allocation) sets,
# per run: CUDA_VISIBLE_DEVICES narrowed to the run's two cards, N=2, TRC_RL_SLOT (a|b), TRC_RL_PAIR (the
# pair id) and TRC_RL_HOST_RAM_SHARE=2 (the host-RAM gate uses half the allocation's cgroup limit).
set -uo pipefail
{ # the whole script is one compound command: bash reads all of it before running it, so replacing this file on
  # NFS while a run is live cannot change or break the running copy

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=common.sh
source "$(dirname "$SELF")/common.sh"

usage() {
  echo "usage: rl.sh --model 3b|7b --arm A|G|P|P_gated|G_gated --seed S (--sft-dir DIR | --smoke) [--dry-run]" >&2
  exit 2
}
MODEL=""; ARM=""; SEED=""; SFT=""; SMOKE=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:-}; shift 2 || usage ;;
    --arm) ARM=${2:-}; shift 2 || usage ;;
    --seed) SEED=${2:-}; shift 2 || usage ;;
    --sft-dir) SFT=${2:-}; shift 2 || usage ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument $1" >&2; usage ;;
  esac
done
case "$MODEL" in 3b|7b) ;; *) usage ;; esac
case "$ARM" in A|G|P|P_gated|G_gated) ;; *) usage ;; esac
[[ $SEED =~ ^[0-9]+$ ]] || usage
if [ "$SMOKE" = 1 ]; then
  [ -z "$SFT" ] || { echo "--smoke trains the base model; drop --sft-dir" >&2; usage; }
else
  [ -n "$SFT" ] || usage
fi

trc_require_env
[ -n "${TRACE_RLVR_ROOT:-}" ] && [ -d "$TRACE_RLVR_ROOT/rlvr/easyr1_backend" ] \
  || { echo "FATAL: config error: TRACE_RLVR_ROOT does not point at a trace-rlvr checkout" >&2; exit 4; }
TRC_RL_PYTHON=${TRC_RL_PYTHON:-$(cd "$(dirname "$TRACE_CERT_PYTHON")/../.." && pwd)/trace_rl/bin/python}
[ -x "$TRC_RL_PYTHON" ] || { echo "FATAL: config error: RL env python $TRC_RL_PYTHON does not exist (set TRC_RL_PYTHON)" >&2; exit 4; }
export TRC_RL_PYTHON
# Triton (vLLM rollout, flash-attn's triton ops) compiles small C launchers against Python.h at run
# time. When the env's base interpreter has no development headers (system python without -devel),
# the env carries headers of the same CPython build in <env>/include/python3.X: export them via CPATH.
pyinc=$("$TRC_RL_PYTHON" -I -c '
import os, sys, sysconfig
base = sysconfig.get_paths()["include"]
own = os.path.join(sys.prefix, "include", "python%d.%d" % sys.version_info[:2])
print("" if os.path.isfile(os.path.join(base, "Python.h")) else own if os.path.isfile(os.path.join(own, "Python.h")) else "MISSING " + base)') \
  || pyinc="MISSING (python failed)"
case "$pyinc" in
  MISSING*) echo "FATAL: config error: no Python.h for $TRC_RL_PYTHON ($pyinc; nor in the env's include/): triton cannot compile its launchers" >&2; exit 4 ;;
  "") ;;
  *) export CPATH="$pyinc${CPATH:+:$CPATH}" ;;
esac
trc_node_allowed || { echo "FATAL: config error: $(hostname -s) is excluded by fleet policy (${TRC_EXCLUDE_NODES:-grn030, grn078/079})" >&2; exit 4; }
TRAIN_RL=$TRC_CERT_DIR/train_rl.py

JOBTAG=${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}.$(date +%m%d_%H%M%S)
if [ "$SMOKE" = 1 ]; then
  RUN=${TRC_RL_SMOKE_ROOT:-$TRACE_CERT_RUNS/rl_smoke}/${MODEL}_${ARM}_s${SEED}/$(date +%m%d_%H%M%S)_$(hostname -s)
else
  RUN=${TRC_RL_ROOT:-$TRACE_CERT_RUNS/rl}/$MODEL/$ARM/s$SEED
fi
TRC_TAG="rl:$MODEL/$ARM/s$SEED$([ "$SMOKE" = 1 ] && echo /smoke)"
STATE=$RUN/logs/job.$JOBTAG
mkdir -p "$STATE"
trc_log_open "$RUN/logs/rl.$JOBTAG.log"
trc_say "rl.sh --model $MODEL --arm $ARM --seed $SEED ${SFT:+--sft-dir $SFT }smoke=$SMOKE dry=$DRY -> $RUN"
trc_say "host $(hostname -s) job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none} cluster $(trc_cluster) N(dispatcher)=${N:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} cpus=$(nproc) python=$TRC_RL_PYTHON"
if [ -n "${TRC_RL_SLOT:-}" ]; then
  [[ $TRC_RL_SLOT =~ ^[a-z]$ ]] || { trc_say "FATAL: config error: TRC_RL_SLOT=$TRC_RL_SLOT must be one lowercase letter"; exit 4; }
  trc_say "pair mode: slot $TRC_RL_SLOT of ${TRC_RL_PAIR:-?} (devices narrowed by rl_pair.sh; host-RAM share 1/${TRC_RL_HOST_RAM_SHARE:-1})"
fi

TRAIN_PID=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$?
  # ignore further signals while cleaning up: a second TERM (Slurm's step-wide TERM plus rl_pair.sh's
  # forwarded one) must not kill this shell before the trainer is stopped and the claims are released
  trap - EXIT; trap '' INT TERM HUP
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    trc_stop_tree "$TRAIN_PID" "train_rl.py" "${TRC_STOP_WAIT:-150}"
  fi
  trc_heartbeat_stop   # first, so no heartbeat rewrite can resurrect a released claim
  trc_claim_release
  trc_lease_release
  trc_say "exit rc=$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'trc_say "signal TERM"; exit 143' TERM
trap 'trc_say "signal INT"; exit 130' INT
trap 'trc_say "signal HUP"; exit 129' HUP

rl_py() { TRC_RL_TAG="$TRC_TAG" "$TRC_RL_PYTHON" "$TRAIN_RL" "$@"; }

# ---------------------------------------------------------------------------- finished already?
if [ "$SMOKE" = 0 ] && [ -f "$RUN/run_receipt.json" ]; then
  st=$(rl_py status --run-dir "$RUN" 2>/dev/null | tail -1)
  trc_say "existing run: $st"
  case "$st" in *'"complete": true'*) trc_say "COMPLETE already: $RUN"; exit 0 ;; esac
fi

# ---------------------------------------------------------------------------- GPU count and VRAM gate
trc_gpu_discover || exit 4
while read -r line; do trc_say "  $line"; done < <(trc_gpu_describe)
[ -z "${N:-}" ] || [ "$N" = "$TRC_N" ] || trc_say "note: dispatcher N=$N, this step sees $TRC_N GPUs (using $TRC_N)"
case "$TRC_N" in
  2|4) ;;
  *) trc_say "FATAL: config error: RL needs N in {2,4} GPUs (DESIGN 7.3); this allocation has $TRC_N"; exit 4 ;;
esac
need_gb=$("$TRC_RL_PYTHON" -c '
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
print(spec["hardware"]["min_vram_gb"][int(sys.argv[2])])' "$TRC_CERT_DIR/configs/rl_$MODEL.yaml" "$TRC_N" 2>/dev/null) || need_gb=80
[[ $need_gb =~ ^[0-9]+$ ]] && [ "$need_gb" -ge 80 ] || need_gb=80
if [ "$TRC_MIN_MIB" -lt $((need_gb * 1000)) ]; then
  trc_say "FATAL: config error: $MODEL RL on $TRC_N GPUs needs >= ${need_gb} GB per card; the smallest card here has ${TRC_MIN_MIB} MiB"
  exit 4
fi

# ---------------------------------------------------------------------------- CPU checks (cached)
check_args=(check --model "$MODEL" --n-gpus "$TRC_N" --json "$STATE/check.json")
if [ "$SMOKE" = 0 ]; then check_args+=(--sft-dir "$SFT"); fi
if ! rl_py "${check_args[@]}"; then
  trc_say "FATAL: config error: train_rl.py check failed (see $RUN/logs/rl.$JOBTAG.log)"; exit 4
fi
trc_say "check ok: imports, vendor manifest, data sha256, store join, model dir, prompt clip ratio 0, arm configs ($STATE/check.json)"
if [ "$DRY" = 1 ]; then trc_say "dry run: no GPU touched"; exit 0; fi

# ---------------------------------------------------------------------------- lease, busy, claims
trc_lease_take "$RUN/.owner" || exit 3
trc_heartbeat_start
if busy=$(trc_gpu_busy); then trc_say "GPUs busy, not starting: $busy"; exit 3; fi
if busy=$(trc_foreign_claimed); then trc_say "GPUs claimed, not starting: $busy"; exit 3; fi
TRC_CLAIM_WHO="rl.sh --model $MODEL --arm $ARM --seed $SEED$([ "$SMOKE" = 1 ] && echo ' --smoke')${TRC_RL_SLOT:+ (pair slot $TRC_RL_SLOT)}"
TRC_CLAIM_RUN=$RUN; TRC_CLAIM_TASK="trc:rl:$MODEL:$ARM:s$SEED$([ "$SMOKE" = 1 ] && echo ':smoke')"
trc_claim_take >"$STATE/claims.txt" 2>&1; rc=$?
trc_say "$(tr '\n' ' ' <"$STATE/claims.txt")"
[ "$rc" -eq 0 ] || exit 3
trc_heartbeat_stop; trc_heartbeat_start
sleep 10
if busy=$(trc_gpu_busy); then trc_say "GPUs became busy after claiming, releasing: $busy"; exit 3; fi

# ---------------------------------------------------------------------------- train (+ merge + vLLM check)
if [ "$SMOKE" = 1 ]; then
  # shellcheck disable=SC2206  # word splitting of the optional extra options is intended
  extra=(${TRC_RL_SMOKE_ARGS:-})
  args=(smoke --model "$MODEL" --arm "$ARM" --seed "$SEED" --n-gpus "$TRC_N" --run-dir "$RUN" ${extra[@]+"${extra[@]}"})
else
  args=(run --model "$MODEL" --arm "$ARM" --seed "$SEED" --n-gpus "$TRC_N" --sft-dir "$SFT" --run-dir "$RUN")
fi
args+=(--vllm-python "$TRC_VENV/bin/python" --stall-minutes "${TRC_RL_STALL_MIN:-45}")
trc_say "launch: train_rl.py ${args[*]}"
TRC_RL_TAG="$TRC_TAG" setsid "$TRC_RL_PYTHON" "$TRAIN_RL" "${args[@]}" </dev/null 3>&- &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; done
wait "$TRAIN_PID" 2>/dev/null; rc=$?
TRAIN_PID=""

# ---------------------------------------------------------------------------- report
if [ "$SMOKE" = 1 ]; then
  # the smoke report (step time, memory, reward components per adapter and per arm) to the task log
  while IFS= read -r line; do trc_say "${line#*] }"; done < <(sed -n '/\] SMOKE /,$p' "$RUN/logs/rl.$JOBTAG.log" | grep -E '\] (SMOKE |  )' | head -60)
else
  while IFS= read -r line; do trc_say "${line#*] }"; done < <(grep -E '\] (step [0-9]+/|COMPLETE|INCOMPLETE|FATAL|FLAG )' "$RUN/logs/rl.$JOBTAG.log" | tail -8)
fi
case "$rc" in
  0) trc_say "COMPLETE: $RUN"; exit 0 ;;
  4) trc_say "FATAL: configuration error (see $RUN/logs/rl.$JOBTAG.log)"; exit 4 ;;
  *) trc_say "INCOMPLETE (train_rl rc=$rc): rerun the same task line; it resumes from the last save"; exit 1 ;;
esac
}
