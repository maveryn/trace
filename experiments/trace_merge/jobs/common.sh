# shellcheck shell=bash
# experiments/trace_merge/jobs/common.sh -- sourced by eval_subset.sh (adapted from the TRACE certificate study's job
# helpers, which ran the same protocol components on the same cluster).
#
# A job runs as one queue-runner task line on ONE node of an allocation that is already held, with N/NODE exported by
# the runner. Nothing here allocates or releases GPUs. GPU claims use the tvm_ prefix in a shared claim directory; a live
# claim of any other campaign except the idle-sharing AIMO policy blocks a card.
#
# Required environment (set by an uncommitted, machine-specific env file; no machine paths in tracked files):
#   TVM_WORK            study work root (splits/, views/, runs/)
#   TVM_PROTOCOL_CLONE  prepared clone of the rlvr branch (af32fe0) with .work/{LMUData,models,eval_deps}
#   TVM_PYTHON          python of the pinned trace_eval environment
#   HF_HOME             Hugging Face cache (offline)
#   TVM_CLAIM_DIR       shared GPU claim directory
# Optional: TRC_NVIDIA_SMI (default nvidia-smi), TRC_CLAIM_STALE (s, default 900), TRC_BUSY_MIB (default 1024),
# TRC_EXCLUDE_NODES (regex, default ^(grn030|grn07[89])$).
#
# Exit codes: 0 done   1 incomplete (rerun resumes)   2 usage   3 GPUs busy / run owned by another live job
#             4 configuration mismatch   5 vLLM did not start   (run_tiers.sh: 6 judge pending on another node)

TRC_JOBS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TVM_PROJECT_DIR=$(cd "$TRC_JOBS_DIR/.." && pwd)
export TRC_JOBS_DIR TVM_PROJECT_DIR

# ============================================================================== environment

trc_require_env() {
  local v missing=()
  for v in TVM_WORK TVM_PROTOCOL_CLONE TVM_PYTHON HF_HOME TVM_CLAIM_DIR; do
    [ -n "${!v:-}" ] || missing+=("$v")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "FATAL: config error: environment variables not set: ${missing[*]} (source the study env file)" >&2
    exit 4
  fi
  export TRC_CLAIM_DIR=$TVM_CLAIM_DIR
  export TRC_NVIDIA_SMI=${TRC_NVIDIA_SMI:-nvidia-smi}
  export TRC_CLAIM_STALE=${TRC_CLAIM_STALE:-900}
  export TRC_BUSY_MIB=${TRC_BUSY_MIB:-1024}
  export HF_HOME
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export CUDA_DEVICE_ORDER=PCI_BUS_ID              # Slurm and nvidia-smi number devices in PCI order
  export TOKENIZERS_PARALLELISM=false
  export TQDM_MININTERVAL=${TQDM_MININTERVAL:-30}
  export PYTHONUNBUFFERED=1
  [ -x "$TVM_PYTHON" ] || { echo "FATAL: config error: $TVM_PYTHON is not executable" >&2; exit 4; }
  [ -d "$TVM_CLAIM_DIR" ] || { echo "FATAL: config error: claim dir $TVM_CLAIM_DIR does not exist" >&2; exit 4; }
  [ -d "$TVM_WORK/splits" ] || { echo "FATAL: config error: no frozen splits under $TVM_WORK" >&2; exit 4; }
}

# ============================================================================== logging
# After trc_log_open the script's stdout/stderr go to LOG only (shared storage survives the srun
# client); trc_say also copies a line to the original stdout (the dispatcher's Txxx.log).

TRC_TAG=${TRC_TAG:-tvm}
trc_ts() { date '+%F %T'; }
trc_iso() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
trc_log() { printf '%s [%s] %s\n' "$(trc_ts)" "$TRC_TAG" "$*"; }
trc_say() {
  trc_log "$@"
  # in a subshell: a dead srun client (SIGPIPE) must not kill the job
  if [ -n "${TRC_LOG_OPEN:-}" ]; then ( printf '%s [%s] %s\n' "$(trc_ts)" "$TRC_TAG" "$*" >&3 ) 2>/dev/null || true; fi
}
# FLAG lines are machine-read by the host watcher: "FLAG <kind> <detail>"
trc_flag() { trc_say "FLAG $*"; }
trc_log_open() {  # trc_log_open LOGFILE
  mkdir -p "$(dirname "$1")"
  exec 3>&1
  exec >>"$1" 2>&1
  TRC_LOG_OPEN=1
  trc_say "log: $1"
}

# ============================================================================== model family -> frame
# Same alnum rule as trace_tool.models.detect_family and certificates.evaluate.model_family.

# DESIGN D8: Qwen2.5-VL (incl. maveryn/trace-qwen2.5-vl-*) answers in pixels -> cert_px parquet;
# Qwen3.5 grounds natively in rel1000 -> cert_rel1000 parquet. Anything else is refused.
trc_model_min_vram_gb() {  # MODEL -> GB per card: <=4B 20, <=9B 40, else 80
  local size
  size=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | grep -oE '[0-9]+(\.[0-9]+)?b' | tail -1 | tr -d b)
  [ -n "$size" ] || { echo 80; return 0; }
  awk -v s="$size" 'BEGIN { if (s <= 4) print 20; else if (s <= 9) print 40; else print 80 }'
}
# served name and pinned revision exactly as trace-tool's run_on_server.sh derives them
# shellcheck disable=SC1091,SC2034,SC2153

# ============================================================================== conditions file

# Render trace_tool_conditions_v1.yaml.in into OUT; check the certificate system prompt against the
# export receipt (the SFT/RL records were built with the same file) and load the result with
# trace-tool's own loader. Leaves OUT untouched when the content is unchanged.

# ============================================================================== GPUs

trc_smi() { "$TRC_NVIDIA_SMI" "$@"; }
trc_trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }

# Sets TRC_N, TRC_GPU_IDX[], TRC_GPU_UUID[], TRC_GPU_MIB[], TRC_GPU_NAME[], TRC_MIN_MIB and
# exports CUDA_VISIBLE_DEVICES (kept when the step set it; else every GPU nvidia-smi lists).
trc_gpu_discover() {
  local rows line idx uuid name tot used tok want=()
  TRC_N=0; TRC_GPU_IDX=(); TRC_GPU_UUID=(); TRC_GPU_MIB=(); TRC_GPU_NAME=(); TRC_MIN_MIB=0
  rows=$(trc_smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits 2>&1) \
    || { echo "FATAL: config error: nvidia-smi failed: $rows" >&2; return 4; }
  case "${CUDA_VISIBLE_DEVICES:-}" in *MIG-*) echo "FATAL: config error: MIG devices are not supported" >&2; return 4 ;; esac
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra want <<<"$CUDA_VISIBLE_DEVICES"; fi
  while IFS=, read -r idx uuid name tot used; do
    idx=$(trc_trim "$idx"); uuid=$(trc_trim "$uuid"); name=$(trc_trim "$name"); tot=$(trc_trim "$tot")
    [ -n "$uuid" ] || continue
    if [ ${#want[@]} -gt 0 ]; then
      local keep=0
      for tok in "${want[@]}"; do tok=$(trc_trim "$tok"); [ "$tok" = "$idx" ] || [ "$tok" = "$uuid" ] && keep=1; done
      [ "$keep" = 1 ] || continue
    fi
    TRC_GPU_IDX+=("$idx"); TRC_GPU_UUID+=("$uuid"); TRC_GPU_NAME+=("$name"); TRC_GPU_MIB+=("${tot%%.*}")
  done <<<"$rows"
  TRC_N=${#TRC_GPU_UUID[@]}
  [ "$TRC_N" -gt 0 ] || { echo "FATAL: config error: no GPU visible (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})" >&2; return 4; }
  if [ ${#want[@]} -gt 0 ] && [ "$TRC_N" -ne ${#want[@]} ]; then
    echo "FATAL: config error: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES does not match nvidia-smi" >&2; return 4
  fi
  TRC_MIN_MIB=${TRC_GPU_MIB[0]}
  local m; for m in "${TRC_GPU_MIB[@]}"; do [ "$m" -lt "$TRC_MIN_MIB" ] && TRC_MIN_MIB=$m; done
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${TRC_GPU_IDX[*]}"); fi
  export CUDA_VISIBLE_DEVICES TRC_N
}
trc_gpu_describe() {
  local i
  for i in "${!TRC_GPU_UUID[@]}"; do
    echo "gpu${TRC_GPU_IDX[$i]} ${TRC_GPU_UUID[$i]} ${TRC_GPU_NAME[$i]} ${TRC_GPU_MIB[$i]}MiB"
  done
}
# trc_gpu_busy: 0 (true) when any selected GPU runs a compute process or holds > TRC_BUSY_MIB.
# Prints what it found. AIMO claims do not make a card busy (fleet policy 2026-10-18: share idle
# allocations); only live use does.
trc_gpu_busy() {
  local apps rows busy=1 u line idx uuid used
  apps=$(trc_smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  for u in "${TRC_GPU_UUID[@]}"; do
    line=$(grep -F "$u" <<<"$apps" || true)
    if [ -n "$line" ]; then echo "busy: compute process on $u: $(tr '\n' ';' <<<"$line")"; busy=0; fi
  done
  rows=$(trc_smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null)
  while IFS=, read -r idx uuid used; do
    uuid=$(trc_trim "$uuid"); used=$(trc_trim "$used"); used=${used%%.*}
    for u in "${TRC_GPU_UUID[@]}"; do
      if [ "$u" = "$uuid" ] && [[ $used =~ ^[0-9]+$ ]] && [ "$used" -gt "$TRC_BUSY_MIB" ]; then
        echo "busy: $uuid holds ${used}MiB (> ${TRC_BUSY_MIB}MiB)"; busy=0
      fi
    done
  done <<<"$rows"
  return "$busy"
}
# trc_foreign_claimed: 0 (true) when a campaign other than AIMO and this study (tvm_) holds a claim with a fresh
# heartbeat (< TRC_CLAIM_STALE) on a selected card of this allocation. AIMO claims on idle cards are
# shared by policy; our own tvm_ claims are arbitrated by trc_claim_take. Prints what it found.
trc_foreign_claimed() {
  local u u8 f base age found=1 cl job node
  cl=$(trc_cluster); job=${SLURM_JOB_ID:-none}; node=$(hostname -s)
  for u in "${TRC_GPU_UUID[@]}"; do
    u8=$(printf '%s' "${u#GPU-}" | cut -c1-8)
    for f in "$TRC_CLAIM_DIR"/*_"$cl"_"$job"_"$node"_"$u8"; do
      [ -e "$f" ] || continue
      base=$(basename "$f")
      case "$base" in aimo_*|tvm_*) continue ;; esac
      if [ -d "$f" ]; then f=$f/owner.json; fi
      age=$(trc_claim_age "$f")
      if [ "$age" -lt "$TRC_CLAIM_STALE" ]; then echo "claimed by another campaign: $base (heartbeat ${age}s old)"; found=0; fi
    done
  done
  return "$found"
}
# Fleet policy (user decision 2026-10-18): never run on AIMO's giants node or the GH200 nodes, even
# when started by hand. TRC_EXCLUDE_NODES is a bash regex over the short hostname.
trc_node_allowed() {
  local re=${TRC_EXCLUDE_NODES:-^(grn030|grn07[89])$}
  ! [[ $(hostname -s) =~ $re ]]
}
# One utilisation sample per selected GPU: "gpu<i> util=<pct>% mem=<used>/<total>MiB"
trc_gpu_util() {
  local rows idx uuid util used tot u
  rows=$(trc_smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
  while IFS=, read -r idx uuid util used tot; do
    uuid=$(trc_trim "$uuid")
    for u in "${TRC_GPU_UUID[@]}"; do
      [ "$u" = "$uuid" ] && echo "gpu$(trc_trim "$idx") util=$(trc_trim "$util")% mem=$(trc_trim "$used")/$(trc_trim "$tot")MiB"
    done
  done <<<"$rows"
}
# Post-launch check (run about 60 s after generation starts): three samples 5 s apart. OK when every
# selected GPU holds at least half its memory (vLLM reserves 0.90) and was busy in some sample.
trc_util_check() {
  local s line gpu util used tot bad=0
  declare -A u_peak=() u_mem=() u_tot=()
  for s in 1 2 3; do
    while read -r line; do
      [ -n "$line" ] || continue
      trc_log "UTILCHECK sample$s $line"
      gpu=${line%% *}; util=${line#*util=}; util=${util%%%*}
      used=${line#*mem=}; used=${used%%/*}; tot=${line##*/}; tot=${tot%MiB}
      [[ $util =~ ^[0-9]+$ ]] || util=0
      if [ "${u_peak[$gpu]:-0}" -lt "$util" ]; then u_peak[$gpu]=$util; fi
      u_mem[$gpu]=${used%%.*}; u_tot[$gpu]=${tot%%.*}
    done < <(trc_gpu_util)
    [ "$s" = 3 ] || sleep 5
  done
  for gpu in "${!u_tot[@]}"; do
    if [ "${u_mem[$gpu]:-0}" -lt $(( ${u_tot[$gpu]:-0} / 2 )) ] || [ "${u_peak[$gpu]:-0}" -eq 0 ]; then
      trc_flag "util_low $gpu peak_util=${u_peak[$gpu]:-0}% mem=${u_mem[$gpu]:-0}/${u_tot[$gpu]:-0}MiB"; bad=1
    fi
  done
  [ ${#u_tot[@]} -gt 0 ] || { trc_flag "util_low no_samples"; bad=1; }
  [ "$bad" = 0 ] && trc_say "UTILCHECK ok: $(trc_gpu_util | tr '\n' ' ')"
  return "$bad"
}

# ============================================================================== claims
# One file per card in TRC_CLAIM_DIR, same JSON shape as the aimo_* files, named
# tvm_<cluster>_<job>_<node>_<uuid8>. Created atomically (noclobber); an existing tvm_ claim with a
# fresh heartbeat means another of our jobs holds the card (exit 3). Heartbeat every 60 s; removed
# on exit (only files that still carry this process's token).

TRC_CLAIM_FILES=()
TRC_OWNER_TOKEN="tvm-$(hostname -s)-$$-$(date +%s)"
trc_cluster() {
  if [ -n "${SLURM_CLUSTER_NAME:-}" ]; then echo "$SLURM_CLUSTER_NAME"; return; fi
  case "$(hostname -s)" in notch*) echo notchpeak ;; grn*) echo granite ;; kp*) echo kingspeak ;; lp*) echo lonepeak ;; *) echo unknown ;; esac
}
trc_claim_json() {  # idx uuid claimed heartbeat
  printf '{"claimed": "%s", "cluster": "%s", "job": "%s", "node": "%s", "gpu_index": "%s", "selected_gpu_uuids": ["%s"], "owner": "%s", "run": "%s", "task": "%s", "heartbeat": "%s"}\n' \
    "$3" "$(trc_cluster)" "${SLURM_JOB_ID:-none}" "$(hostname -s)" "$1" "$2" \
    "claude session TRACE-VERO merge study (${TRC_CLAIM_WHO:-job}; token $TRC_OWNER_TOKEN)" \
    "${TRC_CLAIM_RUN:-}" "${TRC_CLAIM_TASK:-tvm}" "$4"
}
trc_claim_age() {  # FILE -> seconds since its heartbeat, else its claim time (large if unreadable)
  local hb t
  hb=$(sed -n 's/.*"heartbeat": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  [ -n "$hb" ] || hb=$(sed -n 's/.*"claimed": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  [ -n "$hb" ] || { echo 999999; return; }
  t=$(date -d "$hb" +%s 2>/dev/null) || { echo 999999; return; }
  echo $(( $(date +%s) - t ))
}
# temp files for atomic claim writes are hidden (dot prefix), so no claim reader ever sees them
trc_claim_tmp() { printf '%s/.%s.tmp.%s' "$(dirname "$1")" "$(basename "$1")" "$$"; }
# trc_token_dead TOKEN: 0 (true) when TOKEN (another job's TRC_OWNER_TOKEN, tvm-<host>-<pid>-<epoch>) names a process of
# THIS host that no longer exists (a runner killed before it released its claims and lease, e.g. SIGKILL after Slurm's
# KillWait); a token of another host is never judged here, its heartbeat age decides
trc_token_dead() {
  [[ ${1:-} =~ ^tvm-(.+)-([0-9]+)-([0-9]+)$ ]] || return 1
  [ "${BASH_REMATCH[1]}" = "$(hostname -s)" ] && [ ! -e "/proc/${BASH_REMATCH[2]}" ]
}
trc_claim_token() { sed -n 's/.*; token \(tvm-[^)"]*\)).*/\1/p' "$1" 2>/dev/null | head -1; }
trc_claim_take() {  # -> 0 claimed every card, 3 a card is held by a live tvm_ claim
  local i f now age why
  now=$(trc_iso)
  for i in "${!TRC_GPU_UUID[@]}"; do
    f="$TRC_CLAIM_DIR/tvm_$(trc_cluster)_${SLURM_JOB_ID:-none}_$(hostname -s)_$(printf '%s' "${TRC_GPU_UUID[$i]#GPU-}" | cut -c1-8)"
    if ! ( set -o noclobber; trc_claim_json "${TRC_GPU_IDX[$i]}" "${TRC_GPU_UUID[$i]}" "$now" "$now" >"$f" ) 2>/dev/null; then
      age=$(trc_claim_age "$f"); why="heartbeat ${age}s old"
      if [ "$age" -lt "$TRC_CLAIM_STALE" ] && ! grep -qF "$TRC_OWNER_TOKEN" "$f"; then
        if ! trc_token_dead "$(trc_claim_token "$f")"; then
          echo "claim collision: $(basename "$f") heartbeat ${age}s old: $(cat "$f" 2>/dev/null)"
          trc_claim_release; return 3
        fi
        why="$why, but its owner $(trc_claim_token "$f") is gone from this host"
      fi
      echo "replacing stale claim $(basename "$f") ($why)"
      trc_claim_json "${TRC_GPU_IDX[$i]}" "${TRC_GPU_UUID[$i]}" "$now" "$now" >"$(trc_claim_tmp "$f")" && mv -f "$(trc_claim_tmp "$f")" "$f"
    fi
    TRC_CLAIM_FILES+=("$f")
    echo "claimed $(basename "$f")"
  done
}
trc_claim_heartbeat() {
  local f now
  now=$(trc_iso)
  for f in ${TRC_CLAIM_FILES[@]+"${TRC_CLAIM_FILES[@]}"}; do
    if [ ! -f "$f" ] || ! grep -qF "$TRC_OWNER_TOKEN" "$f"; then continue; fi
    sed "s/\"heartbeat\": *\"[^\"]*\"/\"heartbeat\": \"$now\"/" "$f" >"$(trc_claim_tmp "$f")" && mv -f "$(trc_claim_tmp "$f")" "$f"
  done
}
trc_claim_release() {
  local f
  for f in ${TRC_CLAIM_FILES[@]+"${TRC_CLAIM_FILES[@]}"}; do
    if [ -f "$f" ] && grep -qF "$TRC_OWNER_TOKEN" "$f"; then rm -f "$f"; fi
  done
  TRC_CLAIM_FILES=()
}

# ============================================================================== run lease
# One live job per output root: <root>/.owner holds "token host job step pid heartbeat-epoch".
# A fresh lease (< 300 s) of another token -> exit 3 (NFS flock is node-local, so no flock here), unless its runner is
# gone from this host and no process of ours on this host still works in the run directory (an orphaned generation
# queue or score campaign keeps the lease in force).

trc_lease_orphaned() {  # FILE LEASE_LINE -> 0 (true) when the lease's owner is gone and nothing of ours uses its directory
  local re
  trc_token_dead "${2%% *}" || return 1
  # shellcheck disable=SC2016  # a sed bracket expression (regex-quote the path for pgrep), not a shell expansion
  re=$(printf '%s/' "$(dirname "$1")" | sed 's/[][\\.*^$(){}+?|]/\\&/g')
  ! pgrep -u "$(id -u)" -f -- "$re" >/dev/null 2>&1
}
trc_lease_take() {  # FILE
  local f=$1 other age
  mkdir -p "$(dirname "$f")"
  if [ -f "$f" ]; then
    other=$(cat "$f" 2>/dev/null)
    age=$(( $(date +%s) - $(awk '{print $6}' <<<"$other" | grep -E '^[0-9]+$' || echo 0) ))
    if [ "$age" -lt 300 ] && [ "${other%% *}" != "$TRC_OWNER_TOKEN" ]; then
      if ! trc_lease_orphaned "$f" "$other"; then
        echo "run is owned by another live job (lease ${age}s old): $other"; return 3
      fi
      echo "taking over the lease of a runner that is gone from this host (lease ${age}s old): $other"
    fi
  fi
  echo "$TRC_OWNER_TOKEN $(hostname -s) ${SLURM_JOB_ID:-none} ${SLURM_STEP_ID:-none} $$ $(date +%s)" >"$f.tmp.$$" && mv -f "$f.tmp.$$" "$f"
  TRC_LEASE_FILE=$f
  sleep 2
  grep -qF "$TRC_OWNER_TOKEN" "$f" || { echo "lost the lease race: $(cat "$f")"; TRC_LEASE_FILE=""; return 3; }
}
trc_lease_heartbeat() {
  [ -n "${TRC_LEASE_FILE:-}" ] && [ -f "$TRC_LEASE_FILE" ] && grep -qF "$TRC_OWNER_TOKEN" "$TRC_LEASE_FILE" || return 0
  echo "$TRC_OWNER_TOKEN $(hostname -s) ${SLURM_JOB_ID:-none} ${SLURM_STEP_ID:-none} $$ $(date +%s)" >"$TRC_LEASE_FILE.tmp.$$" \
    && mv -f "$TRC_LEASE_FILE.tmp.$$" "$TRC_LEASE_FILE"
}
trc_lease_release() {
  if [ -n "${TRC_LEASE_FILE:-}" ] && [ -f "$TRC_LEASE_FILE" ] && grep -qF "$TRC_OWNER_TOKEN" "$TRC_LEASE_FILE"; then rm -f "$TRC_LEASE_FILE"; fi
  TRC_LEASE_FILE=""
}

# Background heartbeat for claims and lease; stops when the parent exits. (fd 3, the dispatcher's stdout, is closed in
# it: its `sleep 60` outlives a stopped heartbeat and would hold the srun output stream open up to a minute after exit.)
trc_heartbeat_start() {
  local parent=$$
  ( trap - EXIT INT TERM
    while kill -0 "$parent" 2>/dev/null; do
      sleep 60
      kill -0 "$parent" 2>/dev/null || break
      trc_claim_heartbeat; trc_lease_heartbeat
    done ) >/dev/null 2>&1 3>&- &
  TRC_HEARTBEAT_PID=$!
}
trc_heartbeat_stop() {
  if [ -n "${TRC_HEARTBEAT_PID:-}" ]; then kill "$TRC_HEARTBEAT_PID" 2>/dev/null; wait "$TRC_HEARTBEAT_PID" 2>/dev/null; fi
  TRC_HEARTBEAT_PID=""
}

# ============================================================================== process trees
# (run_on_server.sh's kill_tree / stop_tree: vLLM workers must free the GPUs)

trc_kill_tree() { kill -"$2" -- -"$1" 2>/dev/null || { pkill -"$2" -P "$1" 2>/dev/null; kill -"$2" "$1" 2>/dev/null; } || true; }
trc_tree_alive() { pgrep -g "$1" >/dev/null 2>&1 || kill -0 "$1" 2>/dev/null; }
trc_stop_tree() {  # PID LABEL [WAIT_S]: TERM, wait, then KILL
  local i n=$(( ${3:-120} / 2 ))
  trc_kill_tree "$1" TERM
  for i in $(seq "$n"); do trc_tree_alive "$1" || break; sleep 2; done
  if trc_tree_alive "$1"; then trc_log "$2 did not stop in ${3:-120}s; killing it"; trc_kill_tree "$1" KILL; sleep 3; fi
  wait "$1" 2>/dev/null || true
}

# ============================================================================== vLLM server

trc_port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
trc_pick_port() {  # a free port, distinct per allocation: 18100 + (job % 200) * 8, then upward
  local base p jid=${SLURM_JOB_ID:-$$}
  jid=${jid//[!0-9]/}
  base=$(( 18100 + (${jid:-0} % 200) * 8 ))
  for p in $(seq "$base" $((base + 63))); do
    trc_port_in_use "$p" || { echo "$p"; return 0; }
  done
  return 1
}
