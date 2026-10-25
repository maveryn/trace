#!/usr/bin/env bash
# experiments/certificates/jobs/rl_pair.sh '<rl.sh options of run a>' '<rl.sh options of run b>'
#
# Pair mode (DESIGN 7.3, 7.5): two independent 2-GPU RL runs at the same time on disjoint GPU pairs of ONE
# 4-GPU allocation, written as ONE use-gpu task line, e.g.
#
#   source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT" && bash experiments/certificates/jobs/rl_pair.sh \
#     "--model 3b --arm G --seed 3 --sft-dir $TRACE_CERT_RUNS/sft/qwen2.5-vl-3b/ckpt/step_000564" \
#     "--model 3b --arm P --seed 3 --sft-dir $TRACE_CERT_RUNS/sft/qwen2.5-vl-3b/ckpt/step_000564"
#
# Each argument holds the options of one jobs/rl.sh run (word-split, no globbing; --smoke and --dry-run
# allowed). The allocation must show exactly 4 GPUs of one type. The cards are split in PCI order (the order of
# nvidia-smi and of CUDA_DEVICE_ORDER=PCI_BUS_ID): slot a gets the first two, slot b the last two. Every run is a
# plain rl.sh child with
#   CUDA_VISIBLE_DEVICES  two tokens of the inherited mask (narrowed, never widened; same token form)
#   N=2, TRC_RL_SLOT=a|b, TRC_RL_PAIR=<pair id>, TRC_RL_HOST_RAM_SHARE=2
# so it takes its own per-UUID trc_ claims, its own run lease, run dir and logs (rl/<model>/<arm>/s<seed>), and
# train_rl.py starts its own Ray instance (RAY_TMPDIR /tmp/trc_ray_<job>_<pid>_<slot>; every Ray port is chosen
# by the OS, see README "Pair mode"), its own torch.distributed rendezvous (EasyR1 binds a free port) and its
# own vLLM compile cache (VLLM_CACHE_ROOT next to its Ray dir). The host-RAM gate of each run compares half the
# allocation's cgroup limit with hardware.min_host_ram_gb[2].
#
# NUMA: each run is bound (taskset) to the CPUs of this step's cpuset that sit on the NUMA node of its two GPUs
# (sysfs numa_node of the GPUs' PCI functions), so its host-side offload traffic and CPU threads stay local and
# the two runs do not compete for the same cores. No binding when the two GPUs of a run sit on different nodes,
# when both runs map to the same node, when fewer than 12 CPUs are local, or with TRC_RL_PAIR_NUMA=0. Memory
# is not bound (first touch keeps it local; the cgroup limit is shared).
# Slot b starts TRC_RL_PAIR_STAGGER (60) s after slot a. A run that ends with rc 1 (incomplete, resumable) while
# the other one still trains is restarted once (TRC_RL_PAIR_RESTARTS, after TRC_RL_PAIR_RESTART_DELAY = 120 s).
# TERM / INT / HUP are forwarded to both runs (each stops its trainer, keeps its last save, releases its claims);
# after TRC_RL_PAIR_STOP_WAIT (240) s a run that has not exited is killed.
#
# Exit: 0 both runs done; otherwise the worst child code: 4 (configuration, also: not exactly 4 GPUs of one type)
#       > 2 (usage) > any other failure (1: incomplete, rerun resumes) > 3 (busy / claimed) ; 129/130/143 when the
#       pair itself was signalled. Rerunning the same line resumes both runs (a complete run exits 0 at once).
# Log: $TRACE_CERT_RUNS/rl_pair/<pair id>/logs/pair.<job>.<step>.<stamp>.log and attempts/<same>.json (placement,
#      UUIDs and CPUs per slot, rc per run); TRC_RL_PAIR_ROOT overrides the root. Test hooks: TRC_RL_PAIR_CHILD
#      replaces rl.sh, TRC_SYSFS the /sys root, TRC_RL_PAIR_CPUS_ALLOWED this step's cpuset.
set -uo pipefail
{ # the whole script is one compound command: bash reads all of it before running it, so replacing this file on
  # NFS while a pair runs cannot change or break the running copy

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=common.sh
source "$(dirname "$SELF")/common.sh"

usage() {
  echo "usage: rl_pair.sh '<rl.sh options of run a>' '<rl.sh options of run b>'" >&2
  echo "  e.g. rl_pair.sh '--model 3b --arm G --seed 3 --sft-dir DIR' '--model 3b --arm P --seed 3 --sft-dir DIR'" >&2
  exit 2
}
[ $# -eq 2 ] || usage
CHILD=${TRC_RL_PAIR_CHILD:-$TRC_JOBS_DIR/rl.sh}
STAGGER=${TRC_RL_PAIR_STAGGER:-60}
RESTARTS=${TRC_RL_PAIR_RESTARTS:-1}
RESTART_DELAY=${TRC_RL_PAIR_RESTART_DELAY:-120}
STOP_WAIT=${TRC_RL_PAIR_STOP_WAIT:-240}
for v in STAGGER RESTARTS RESTART_DELAY STOP_WAIT; do
  [[ ${!v} =~ ^[0-9]+$ ]] || { echo "TRC_RL_PAIR_$v=${!v} must be a non-negative integer" >&2; exit 2; }
done
SLOTS=(a b)

# ------------------------------------------------------------------------------- the two runs' options
declare -A ARGS=() KEY=() LABEL=()
parse_run() {  # SLOT STRING: validate the rl.sh options of one run (rl.sh validates again)
  local slot=$1 model="" arm="" seed="" sft="" smoke=0 i
  local -a a=()
  read -r -a a <<<"$2"
  [ ${#a[@]} -gt 0 ] || { echo "run $slot: no options" >&2; usage; }
  for ((i = 0; i < ${#a[@]}; i++)); do
    case "${a[$i]}" in
      --model) model=${a[$((i + 1))]:-}; i=$((i + 1)) ;;
      --arm) arm=${a[$((i + 1))]:-}; i=$((i + 1)) ;;
      --seed) seed=${a[$((i + 1))]:-}; i=$((i + 1)) ;;
      --sft-dir) sft=${a[$((i + 1))]:-}; i=$((i + 1)) ;;
      --smoke) smoke=1 ;;
      --dry-run) ;;
      *) echo "run $slot: unknown rl.sh option ${a[$i]}" >&2; usage ;;
    esac
  done
  case "$model" in 3b|7b) ;; *) echo "run $slot: --model 3b|7b" >&2; usage ;; esac
  case "$arm" in A|G|P|P_gated|G_gated) ;; *) echo "run $slot: --arm A|G|P|P_gated|G_gated" >&2; usage ;; esac
  [[ $seed =~ ^[0-9]+$ ]] || { echo "run $slot: --seed must be a number" >&2; usage; }
  if [ "$smoke" = 1 ]; then
    [ -z "$sft" ] || { echo "run $slot: --smoke trains the base model; drop --sft-dir" >&2; usage; }
  else
    [ -n "$sft" ] || { echo "run $slot: --sft-dir or --smoke" >&2; usage; }
  fi
  ARGS[$slot]=$2
  KEY[$slot]="$model/$arm/s$seed$([ "$smoke" = 1 ] && echo /smoke)"
  LABEL[$slot]="${arm}_s${seed}"
  MODEL_OF[$slot]=$model; SMOKE_OF[$slot]=$smoke
}
declare -A MODEL_OF=() SMOKE_OF=()
parse_run a "$1"
parse_run b "$2"
[ "${KEY[a]}" != "${KEY[b]}" ] || { echo "both arguments name the same run (${KEY[a]})" >&2; usage; }

trc_require_env
trc_node_allowed || { echo "FATAL: config error: $(hostname -s) is excluded by fleet policy (${TRC_EXCLUDE_NODES:-grn030, grn078/079})" >&2; exit 4; }
[ -f "$CHILD" ] || { echo "FATAL: config error: $CHILD does not exist" >&2; exit 4; }

PAIR_ID="${MODEL_OF[a]}_${LABEL[a]}+${LABEL[b]}"
[ "${SMOKE_OF[a]}${SMOKE_OF[b]}" = 00 ] || PAIR_ID+="_smoke"
JOBTAG=${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}.$(date +%m%d_%H%M%S)
PROOT=${TRC_RL_PAIR_ROOT:-$TRACE_CERT_RUNS/rl_pair}/$PAIR_ID
mkdir -p "$PROOT/logs" "$PROOT/attempts"
TRC_TAG="rl_pair:$PAIR_ID"
trc_log_open "$PROOT/logs/pair.$JOBTAG.log"
trc_say "rl_pair.sh a: ${ARGS[a]} | b: ${ARGS[b]}"
trc_say "host $(hostname -s) job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none} N(dispatcher)=${N:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} cpus=$(nproc)"

# ------------------------------------------------------------------------------- the 4 GPUs and the split
ORIG_CVD=${CUDA_VISIBLE_DEVICES:-}
trc_gpu_discover || exit 4
while read -r line; do trc_say "  $line"; done < <(trc_gpu_describe)
if [ "$TRC_N" -ne 4 ]; then
  trc_say "FATAL: config error: pair mode needs exactly 4 GPUs (two disjoint pairs); this allocation shows $TRC_N"; exit 4
fi
names=$(printf '%s\n' "${TRC_GPU_NAME[@]}" | sort -u)
if [ "$(wc -l <<<"$names")" -ne 1 ]; then
  trc_say "FATAL: config error: mixed GPU types ($(tr '\n' ';' <<<"$names")): both runs of a pair must be one hardware class"; exit 4
fi
declare -a INH=()
if [ -n "$ORIG_CVD" ]; then IFS=, read -r -a INH <<<"$ORIG_CVD"; fi
token_of() {  # i -> the inherited CUDA_VISIBLE_DEVICES token of GPU i (index or UUID form), else its index
  local i=$1 t
  for t in ${INH[@]+"${INH[@]}"}; do
    t=$(trc_trim "$t")
    if [ "$t" = "${TRC_GPU_IDX[$i]}" ] || [ "$t" = "${TRC_GPU_UUID[$i]}" ]; then printf '%s' "$t"; return 0; fi
  done
  [ ${#INH[@]} -eq 0 ] || return 1   # never widen: a GPU outside the inherited mask is not ours
  printf '%s' "${TRC_GPU_IDX[$i]}"
}
declare -A TOK=() UUIDS=()
for s in 0 1; do
  slot=${SLOTS[$s]}; i=$((2 * s)); j=$((2 * s + 1))
  ti=$(token_of "$i") && tj=$(token_of "$j") || { trc_say "FATAL: config error: GPU outside the inherited mask $ORIG_CVD"; exit 4; }
  TOK[$slot]="$ti,$tj"
  UUIDS[$slot]="${TRC_GPU_UUID[$i]},${TRC_GPU_UUID[$j]}"
  trc_say "slot $slot (${KEY[$slot]}): CUDA_VISIBLE_DEVICES=${TOK[$slot]} = ${UUIDS[$slot]}"
done
if command -v "$TRC_NVIDIA_SMI" >/dev/null 2>&1; then
  trc_log "nvidia-smi topo -m:"; trc_smi topo -m 2>&1 | head -20 || true
fi

# ------------------------------------------------------------------------------- NUMA-local CPUs per slot
SYSFS=${TRC_SYSFS:-/sys}
expand_cpus() {  # "0-3,12" -> one CPU per line
  local part
  local -a parts=()
  IFS=, read -r -a parts <<<"$1"
  for part in ${parts[@]+"${parts[@]}"}; do
    part=$(trc_trim "$part")
    if [[ $part =~ ^([0-9]+)-([0-9]+)$ ]]; then seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    elif [[ $part =~ ^[0-9]+$ ]]; then echo "$part"; fi
  done
}
gpu_numa() {  # GPU index -> NUMA node of its PCI function (empty if unknown)
  local bus dev node
  bus=$(trc_smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | awk -F, -v i="$1" '$1 + 0 == i && $1 ~ /[0-9]/ {gsub(/ /, "", $2); print $2; exit}')
  [[ $bus =~ ^[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+$ ]] || return 0
  dev=${bus%%:*}; dev=$(printf '%s' "${dev: -4}:${bus#*:}" | tr 'A-F' 'a-f')
  node=$(cat "$SYSFS/bus/pci/devices/$dev/numa_node" 2>/dev/null)
  [[ $node =~ ^[0-9]+$ ]] && printf '%s' "$node"
}
declare -A CPUS=() NUMA=()
if [ "${TRC_RL_PAIR_NUMA:-1}" != 0 ] && command -v taskset >/dev/null 2>&1; then
  allowed=${TRC_RL_PAIR_CPUS_ALLOWED:-$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status)}
  for s in 0 1; do
    slot=${SLOTS[$s]}
    na=$(gpu_numa "${TRC_GPU_IDX[$((2 * s))]}"); nb=$(gpu_numa "${TRC_GPU_IDX[$((2 * s + 1))]}")
    if [ -n "$na" ] && [ "$na" = "$nb" ] && [ -r "$SYSFS/devices/system/node/node$na/cpulist" ]; then
      NUMA[$slot]=$na
      CPUS[$slot]=$(comm -12 <(expand_cpus "$allowed" | sort -u) <(expand_cpus "$(cat "$SYSFS/devices/system/node/node$na/cpulist")" | sort -u) \
                    | sort -n | paste -sd, -)
    fi
  done
  why=""
  if [ -z "${NUMA[a]:-}" ] || [ -z "${NUMA[b]:-}" ]; then why="NUMA node of a GPU pair unknown or split"
  elif [ "${NUMA[a]}" = "${NUMA[b]}" ]; then why="both pairs on NUMA node ${NUMA[a]}"
  else
    for slot in "${SLOTS[@]}"; do
      n=$(tr ',' '\n' <<<"${CPUS[$slot]:-}" | grep -c .)
      [ "$n" -ge 12 ] || why="slot $slot has only $n NUMA-local CPUs"
    done
  fi
  if [ -n "$why" ]; then
    trc_say "no CPU binding: $why"; CPUS=()
  else
    for slot in "${SLOTS[@]}"; do trc_say "slot $slot: NUMA node ${NUMA[$slot]}, CPUs ${CPUS[$slot]}"; done
  fi
else
  trc_say "no CPU binding (TRC_RL_PAIR_NUMA=0 or no taskset)"
fi

# ------------------------------------------------------------------------------- children
declare -A PID=() RC=() TRIES=() STARTED=()
STOPPING=0; STOP_SIG=""; STOP_RC=0; STOP_AT=0
start_child() {  # SLOT
  local slot=$1
  local -a a=()
  read -r -a a <<<"${ARGS[$slot]}"
  # exec setsid: the child is its own session and process-group leader with this PID (rl.sh's own traps
  # stop its trainer); stdout/stderr = the task log (fd 3), so its trc_say lines reach the dispatcher log
  local -a bind=()
  if [ -n "${CPUS[$slot]:-}" ]; then bind=(taskset -c "${CPUS[$slot]}"); fi
  ( export CUDA_VISIBLE_DEVICES="${TOK[$slot]}" N=2 TRC_RL_SLOT="$slot" TRC_RL_PAIR="$PAIR_ID" TRC_RL_HOST_RAM_SHARE=2
    exec ${bind[@]+"${bind[@]}"} setsid bash "$CHILD" "${a[@]}" ) </dev/null >&3 2>&3 &
  PID[$slot]=$!
  TRIES[$slot]=$(( ${TRIES[$slot]:-0} + 1 ))
  STARTED[$slot]=$(date +%s)
  trc_say "slot $slot started (try ${TRIES[$slot]}): pid ${PID[$slot]} rl.sh ${ARGS[$slot]}"
}
alive() { [ -n "${PID[$1]:-}" ] && kill -0 "${PID[$1]}" 2>/dev/null; }
forward() {  # SIG RC
  [ "$STOPPING" = 1 ] && return 0
  STOPPING=1; STOP_SIG=$1; STOP_RC=$2; STOP_AT=$(date +%s)
  trc_say "signal $1: forwarding TERM to both runs"
  local s
  for s in "${SLOTS[@]}"; do alive "$s" && kill -TERM "${PID[$s]}" 2>/dev/null; done
}
trap 'forward TERM 143' TERM
trap 'forward INT 130' INT
trap 'forward HUP 129' HUP
# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$? s
  trap - EXIT
  for s in "${SLOTS[@]}"; do
    if alive "$s"; then
      trc_say "exit with slot $s still running: TERM pid ${PID[$s]}"
      kill -TERM "${PID[$s]}" 2>/dev/null
    fi
  done
  exit "$rc"
}
trap on_exit EXIT
pause() { sleep "$1" & wait $! 2>/dev/null; }   # interruptible: a trap runs at once

record() {  # write attempts/<JOBTAG>.json (placement, UUIDs, rc per run)
  local f=$PROOT/attempts/$JOBTAG.json
  {
    printf '{"pair": "%s", "job": "%s", "step": "%s", "node": "%s", "gpu_class": "%s x2", "n_gpus_per_run": 2,\n' \
      "$PAIR_ID" "${SLURM_JOB_ID:-none}" "${SLURM_STEP_ID:-none}" "$(hostname -s)" "${TRC_GPU_NAME[0]}"
    printf ' "inherited_cuda_visible_devices": "%s", "updated": "%s", "final_rc": %s,\n' "$ORIG_CVD" "$(trc_iso)" "${1:-null}"
    printf ' "runs": {"a": {"run": "%s", "args": "%s", "cuda_visible_devices": "%s", "uuids": "%s", "numa": "%s", "cpus": "%s", "tries": %s, "rc": %s},\n' \
      "${KEY[a]}" "${ARGS[a]//\"/\\\"}" "${TOK[a]}" "${UUIDS[a]}" "${NUMA[a]:-}" "${CPUS[a]:-}" "${TRIES[a]:-0}" "${RC[a]:-null}"
    printf '          "b": {"run": "%s", "args": "%s", "cuda_visible_devices": "%s", "uuids": "%s", "numa": "%s", "cpus": "%s", "tries": %s, "rc": %s}}}\n' \
      "${KEY[b]}" "${ARGS[b]//\"/\\\"}" "${TOK[b]}" "${UUIDS[b]}" "${NUMA[b]:-}" "${CPUS[b]:-}" "${TRIES[b]:-0}" "${RC[b]:-null}"
  } >"$f.tmp.$$" && mv -f "$f.tmp.$$" "$f"
}

start_child a
t0=$(date +%s)
while [ "$STOPPING" = 0 ] && [ $(( $(date +%s) - t0 )) -lt "$STAGGER" ] && alive a; do pause 2; done
[ "$STOPPING" = 1 ] || start_child b
record

restart_at_a=0; restart_at_b=0
while :; do
  for s in "${SLOTS[@]}"; do
    if [ -n "${PID[$s]:-}" ] && ! kill -0 "${PID[$s]}" 2>/dev/null; then
      wait "${PID[$s]}" 2>/dev/null; RC[$s]=$?
      trc_say "slot $s (${KEY[$s]}) exited rc=${RC[$s]} after $(( $(date +%s) - ${STARTED[$s]} ))s (try ${TRIES[$s]})"
      PID[$s]=""
      other=b; [ "$s" = b ] && other=a
      if [ "$STOPPING" = 0 ] && [ "${RC[$s]}" = 1 ] && [ $(( ${TRIES[$s]} - 1 )) -lt "$RESTARTS" ] && alive "$other"; then
        eval "restart_at_$s=$(( $(date +%s) + RESTART_DELAY ))"
        trc_say "slot $s: rc 1 (incomplete, resumable) while slot $other trains: restart in ${RESTART_DELAY}s"
      fi
      record
    fi
  done
  for s in "${SLOTS[@]}"; do
    at_var="restart_at_$s"
    if [ "${!at_var}" -gt 0 ] && [ -z "${PID[$s]:-}" ]; then
      if [ "$STOPPING" = 1 ]; then eval "$at_var=0"
      elif [ "$(date +%s)" -ge "${!at_var}" ]; then eval "$at_var=0"; unset 'RC[$s]'; start_child "$s"; record
      fi
    fi
  done
  if [ -z "${PID[a]:-}" ] && [ -z "${PID[b]:-}" ] && [ "$restart_at_a" = 0 ] && [ "$restart_at_b" = 0 ]; then break; fi
  if [ "$STOPPING" = 1 ] && [ $(( $(date +%s) - STOP_AT )) -ge "$STOP_WAIT" ]; then
    for s in "${SLOTS[@]}"; do
      if alive "$s"; then trc_say "slot $s did not stop within ${STOP_WAIT}s: KILL its process group"; kill -KILL -- -"${PID[$s]}" 2>/dev/null; fi
    done
    STOP_AT=$(date +%s)
  fi
  pause 5
done

# ------------------------------------------------------------------------------- result
worst=0
for code in 4 2; do
  for s in "${SLOTS[@]}"; do [ "${RC[$s]:-1}" = "$code" ] && [ "$worst" = 0 ] && worst=$code; done
done
if [ "$worst" = 0 ]; then
  for s in "${SLOTS[@]}"; do
    case "${RC[$s]:-1}" in 0|3) ;; *) [ "$worst" = 0 ] && worst=${RC[$s]:-1} ;; esac
  done
fi
if [ "$worst" = 0 ]; then
  for s in "${SLOTS[@]}"; do [ "${RC[$s]:-1}" = 3 ] && worst=3; done
fi
[ "$STOPPING" = 1 ] && worst=$STOP_RC
record "$worst"
trc_say "pair $PAIR_ID: a ${KEY[a]} rc=${RC[a]:-none}, b ${KEY[b]} rc=${RC[b]:-none} -> exit $worst"
exit "$worst"
}
