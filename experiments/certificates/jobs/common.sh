# shellcheck shell=bash
# experiments/certificates/jobs/common.sh -- sourced by the certificate GPU job scripts.
#
# A job script runs as one use-gpu task line, inside an `srun --overlap` step on ONE node of an
# allocation that is already held, with cwd = the TimeWarp checkout and N/NODE exported by the
# dispatcher. Everything here works from any cwd and never allocates or releases anything.
#
# Required environment (set by the generated, uncommitted $TRACE_CERT_RUNS/host/env.sh; this
# file carries no machine paths, see DESIGN 1 "No machine paths in tracked files"):
#   TRACE_CERT_RUNS     root of every large artifact (data/v2, eval/, dispatch/)
#   TRACE_TOOL_ROOT     trace-tool checkout (scripts/serve_vllm.sh, .venv with vLLM, trace-tool-eval)
#   HF_HOME             Hugging Face cache holding the pinned model snapshots
#   TRACE_CERT_PYTHON   python of the CPU scoring env (trace_tasks editable from this checkout)
#   TRC_CLAIM_DIR       shared GPU claim directory (the AIMO campaign's gpu_claims)
# Optional: TRACE_RLVR_ROOT, TRC_DATA (default $TRACE_CERT_RUNS/data/v2), TRC_VENV (default
# $TRACE_TOOL_ROOT/.venv), TRC_NVIDIA_SMI (default nvidia-smi), TRC_SERVE_SCRIPT (default
# $TRACE_TOOL_ROOT/scripts/serve_vllm.sh), TRC_CLAIM_STALE (s, default 900), TRC_BUSY_MIB
# (default 1024), TRC_SERVER_TIMEOUT (s, default 1800), TRC_MAX_LEN (vLLM --max-model-len, default
# 16384: the largest frozen image is 0.9 MP, <= 1.2k image tokens, + 2k prompt + 8192 new tokens),
# TRC_MAX_IMAGES (default 1), TRC_EXCLUDE_NODES (regex, default ^(grn030|grn07[89])$).
# TRC_NVIDIA_SMI and TRC_SERVE_SCRIPT exist so the whole job can be exercised on CPU with stand-ins.
#
# Exit codes shared by the job scripts:
#   0 done   1 incomplete (gaps, endpoint errors, scoring failure; rerun resumes)   2 usage
#   3 GPUs busy / another owner holds the run (the host runner re-dispatches, no retry counted)
#   4 configuration mismatch (VRAM, family, environment, frozen data)   5 vLLM did not start

TRC_JOBS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRC_CERT_DIR=$(cd "$TRC_JOBS_DIR/.." && pwd)
TRACE_ROOT=$(cd "$TRC_JOBS_DIR/../../.." && pwd)
export TRACE_ROOT TRC_JOBS_DIR TRC_CERT_DIR

# ============================================================================== environment

trc_require_env() {
  local v missing=()
  for v in TRACE_CERT_RUNS TRACE_TOOL_ROOT HF_HOME TRACE_CERT_PYTHON TRC_CLAIM_DIR; do
    [ -n "${!v:-}" ] || missing+=("$v")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "FATAL: config error: environment variables not set: ${missing[*]} (source \$TRACE_CERT_RUNS/host/env.sh)" >&2
    exit 4
  fi
  export TRC_DATA=${TRC_DATA:-$TRACE_CERT_RUNS/data/v2}
  export TRC_VENV=${TRC_VENV:-$TRACE_TOOL_ROOT/.venv}
  export TRC_SERVE_SCRIPT=${TRC_SERVE_SCRIPT:-$TRACE_TOOL_ROOT/scripts/serve_vllm.sh}
  export TRC_NVIDIA_SMI=${TRC_NVIDIA_SMI:-nvidia-smi}
  export TRC_CLAIM_STALE=${TRC_CLAIM_STALE:-900}
  export TRC_BUSY_MIB=${TRC_BUSY_MIB:-1024}
  export TRC_SERVER_TIMEOUT=${TRC_SERVER_TIMEOUT:-1800}
  export HF_HOME
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}       # every pinned snapshot is cached
  export CUDA_DEVICE_ORDER=PCI_BUS_ID              # Slurm and nvidia-smi number devices in PCI order
  export TOKENIZERS_PARALLELISM=false
  export TQDM_MININTERVAL=${TQDM_MININTERVAL:-30}   # keep tqdm from flooding the logs
  export PYTHONUNBUFFERED=1
  local f
  for f in "$TRACE_CERT_PYTHON" "$TRC_VENV/bin/activate" "$TRC_SERVE_SCRIPT" "$TRACE_TOOL_ROOT/scripts/_model_env.sh"; do
    [ -e "$f" ] || { echo "FATAL: config error: $f does not exist" >&2; exit 4; }
  done
  [ -d "$TRC_CLAIM_DIR" ] || { echo "FATAL: config error: claim dir $TRC_CLAIM_DIR does not exist" >&2; exit 4; }
  [ -d "$TRC_DATA/public" ] || { echo "FATAL: config error: no frozen data at $TRC_DATA" >&2; exit 4; }
}

# ============================================================================== logging
# After trc_log_open the script's stdout/stderr go to LOG only (shared storage survives the srun
# client); trc_say also copies a line to the original stdout (the dispatcher's Txxx.log).

TRC_TAG=${TRC_TAG:-trc}
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

trc_family() {  # MODEL -> qwen2.5-vl | qwen3-vl | qwen3.5 | other
  local key
  key=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')
  case "$key" in
    *qwen25vl*) echo qwen2.5-vl ;;
    *qwen3vl*) echo qwen3-vl ;;
    *qwen35*) echo qwen3.5 ;;
    *) echo other ;;
  esac
}
# DESIGN D8: Qwen2.5-VL (incl. maveryn/trace-qwen2.5-vl-*) answers in pixels -> cert_px parquet;
# Qwen3.5 grounds natively in rel1000 -> cert_rel1000 parquet. Anything else is refused.
trc_frame() {  # FAMILY -> px | rel1000
  case "$1" in
    qwen2.5-vl) echo px ;;
    qwen3.5) echo rel1000 ;;
    *) return 1 ;;
  esac
}
trc_coord_frame() { case "$1" in px) echo px_source ;; rel1000) echo rel1000 ;; *) return 1 ;; esac; }
trc_public_cond() {  # trace-tool condition, frame -> public parquet condition
  case "$1" in
    ans_v1) echo ans ;;
    ann_v1) echo ann ;;
    cert_v1) echo "cert_$2" ;;
    *) return 1 ;;
  esac
}
trc_model_min_vram_gb() {  # MODEL -> GB per card: <=4B 20, <=9B 40, else 80
  local size
  size=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | grep -oE '[0-9]+(\.[0-9]+)?b' | tail -1 | tr -d b)
  [ -n "$size" ] || { echo 80; return 0; }
  awk -v s="$size" 'BEGIN { if (s <= 4) print 20; else if (s <= 9) print 40; else print 80 }'
}
# served name and pinned revision exactly as trace-tool's run_on_server.sh derives them
# shellcheck disable=SC1091,SC2034,SC2153
trc_model_env() {  # MODEL -> sets TRC_SERVED TRC_REVISION
  TRC_SERVED=$( (unset SERVED REVISION; MODEL=$1; source "$TRACE_TOOL_ROOT/scripts/_model_env.sh"; echo "$SERVED") )
  TRC_REVISION=$( (unset SERVED REVISION; MODEL=$1; source "$TRACE_TOOL_ROOT/scripts/_model_env.sh"; echo "$REVISION") )
}

# ============================================================================== conditions file

trc_receipt_value() {  # trc_receipt_value SPLIT PYEXPR-on-d -> value from receipts/<split>.json
  "$TRACE_CERT_PYTHON" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' \
    "$TRC_DATA/receipts/$1.json" "$2"
}
# Render trace_tool_conditions_v1.yaml.in into OUT; check the certificate system prompt against the
# export receipt (the SFT/RL records were built with the same file) and load the result with
# trace-tool's own loader. Leaves OUT untouched when the content is unchanged.
trc_render_conditions() {  # OUT
  local out=$1 tmpl=$TRC_CERT_DIR/trace_tool_conditions_v1.yaml.in
  local sys=$TRACE_ROOT/src/trace_tasks/resources/prompts/experimental/certificates/cert_system_v1.txt
  local want got tmp
  [ -f "$tmpl" ] && [ -f "$sys" ] || { echo "FATAL: config error: missing $tmpl or $sys" >&2; return 4; }
  case "$sys" in *\"*|*\\*|*\|*|*\&*) echo "FATAL: config error: unusable path $sys" >&2; return 4 ;; esac
  want=$(trc_receipt_value e1_dev 'd["provenance"]["resource_sha256"]["cert_system_v1.txt"]') || return 4
  got=$(sha256sum "$sys" | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "FATAL: config error: cert_system_v1.txt sha256 $got != export receipt $want" >&2; return 4; }
  mkdir -p "$(dirname "$out")"
  tmp=$(mktemp "$out.XXXXXX")
  sed "s|@CERT_SYSTEM_PROMPT@|$sys|g" "$tmpl" >"$tmp"
  if grep -q '@[A-Z_]*@' "$tmp"; then rm -f "$tmp"; echo "FATAL: config error: unrendered placeholder in $out" >&2; return 4; fi
  if ! "$TRC_VENV/bin/python" -c '
import sys
from trace_tool.conditions import load_conditions
c = load_conditions(sys.argv[1])
want = {"ans_v1", "ann_v1", "cert_v1"}
assert set(c) == want, sorted(c)
for name, spec in c.items():
    assert spec["max_turns"] == 1 and spec["max_tokens_per_turn"] == 8192 and spec["max_total_tokens"] == 8192, name
    assert spec["decoding"] == "trace" and not spec["tools"] and spec["oracle_crop"] is False, name
' "$tmp"; then
    rm -f "$tmp"; echo "FATAL: config error: trace-tool rejects the rendered conditions" >&2; return 4
  fi
  if [ -f "$out" ] && cmp -s "$tmp" "$out"; then rm -f "$tmp"; else mv -f "$tmp" "$out"; fi
}

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
# trc_foreign_claimed: 0 (true) when a campaign other than AIMO and us holds a claim with a fresh
# heartbeat (< TRC_CLAIM_STALE) on a selected card of this allocation. AIMO claims on idle cards are
# shared by policy; our own trc_ claims are arbitrated by trc_claim_take. Prints what it found.
trc_foreign_claimed() {
  local u u8 f base age found=1 cl job node
  cl=$(trc_cluster); job=${SLURM_JOB_ID:-none}; node=$(hostname -s)
  for u in "${TRC_GPU_UUID[@]}"; do
    u8=$(printf '%s' "${u#GPU-}" | cut -c1-8)
    for f in "$TRC_CLAIM_DIR"/*_"$cl"_"$job"_"$node"_"$u8"; do
      [ -e "$f" ] || continue
      base=$(basename "$f")
      case "$base" in aimo_*|trc_*) continue ;; esac
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
# trc_<cluster>_<job>_<node>_<uuid8>. Created atomically (noclobber); an existing trc_ claim with a
# fresh heartbeat means another of our jobs holds the card (exit 3). Heartbeat every 60 s; removed
# on exit (only files that still carry this process's token).

TRC_CLAIM_FILES=()
TRC_OWNER_TOKEN="trc-$(hostname -s)-$$-$(date +%s)"
trc_cluster() {
  if [ -n "${SLURM_CLUSTER_NAME:-}" ]; then echo "$SLURM_CLUSTER_NAME"; return; fi
  case "$(hostname -s)" in notch*) echo notchpeak ;; grn*) echo granite ;; kp*) echo kingspeak ;; lp*) echo lonepeak ;; *) echo unknown ;; esac
}
trc_claim_json() {  # idx uuid claimed heartbeat
  printf '{"claimed": "%s", "cluster": "%s", "job": "%s", "node": "%s", "gpu_index": "%s", "selected_gpu_uuids": ["%s"], "owner": "%s", "run": "%s", "task": "%s", "heartbeat": "%s"}\n' \
    "$3" "$(trc_cluster)" "${SLURM_JOB_ID:-none}" "$(hostname -s)" "$1" "$2" \
    "claude session TRACE-certificates (${TRC_CLAIM_WHO:-job}; token $TRC_OWNER_TOKEN)" \
    "${TRC_CLAIM_RUN:-}" "${TRC_CLAIM_TASK:-trc}" "$4"
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
trc_claim_take() {  # -> 0 claimed every card, 3 a card is held by a live trc_ claim
  local i f now age
  now=$(trc_iso)
  for i in "${!TRC_GPU_UUID[@]}"; do
    f="$TRC_CLAIM_DIR/trc_$(trc_cluster)_${SLURM_JOB_ID:-none}_$(hostname -s)_$(printf '%s' "${TRC_GPU_UUID[$i]#GPU-}" | cut -c1-8)"
    if ! ( set -o noclobber; trc_claim_json "${TRC_GPU_IDX[$i]}" "${TRC_GPU_UUID[$i]}" "$now" "$now" >"$f" ) 2>/dev/null; then
      age=$(trc_claim_age "$f")
      if [ "$age" -lt "$TRC_CLAIM_STALE" ] && ! grep -qF "$TRC_OWNER_TOKEN" "$f"; then
        echo "claim collision: $(basename "$f") heartbeat ${age}s old: $(cat "$f" 2>/dev/null)"
        trc_claim_release; return 3
      fi
      echo "replacing stale claim $(basename "$f") (heartbeat ${age}s old)"
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
# A fresh lease (< 300 s) of another token -> exit 3 (NFS flock is node-local, so no flock here).

trc_lease_take() {  # FILE
  local f=$1 other age
  mkdir -p "$(dirname "$f")"
  if [ -f "$f" ]; then
    other=$(cat "$f" 2>/dev/null)
    age=$(( $(date +%s) - $(awk '{print $6}' <<<"$other" | grep -E '^[0-9]+$' || echo 0) ))
    if [ "$age" -lt 300 ] && [ "${other%% *}" != "$TRC_OWNER_TOKEN" ]; then
      echo "run is owned by another live job (lease ${age}s old): $other"; return 3
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

# Background heartbeat for claims and lease; stops when the parent exits.
trc_heartbeat_start() {
  local parent=$$
  ( trap - EXIT INT TERM
    while kill -0 "$parent" 2>/dev/null; do
      sleep 60
      kill -0 "$parent" 2>/dev/null || break
      trc_claim_heartbeat; trc_lease_heartbeat
    done ) >/dev/null 2>&1 &
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
trc_serve_healthy() {  # PORT SERVED
  curl -sf --max-time 10 "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -qF "\"$2\""
}
# trc_serve_start MODEL REVISION SERVED PORT DP LOG: scripts/serve_vllm.sh exactly as run_on_server.sh
# starts it (TP=1, DP=<GPUs>, its own process group); waits until /v1/models lists SERVED.
trc_serve_start() {
  local model=$1 rev=$2 served=$3 port=$4 dp=$5 log=$6 waited=0
  if trc_port_in_use "$port"; then echo "port $port is in use" >&2; return 5; fi
  printf '\n==== %s  starting %s %s on GPUs %s port %s\n' "$(trc_ts)" "$model" "$rev" "$CUDA_VISIBLE_DEVICES" "$port" >>"$log"
  env MODEL="$model" REVISION="$rev" SERVED="$served" PORT="$port" TP=1 DP="$dp" \
    MAX_LEN="${TRC_MAX_LEN:-16384}" MAX_IMAGES="${TRC_MAX_IMAGES:-1}" \
    setsid bash "$TRC_SERVE_SCRIPT" >>"$log" 2>&1 </dev/null 3>&- &
  TRC_SERVER_PID=$!
  # shellcheck disable=SC2034  # read by the job scripts
  TRC_SERVER_PORT=$port
  trc_say "vLLM starting: pid $TRC_SERVER_PID port $port DP=$dp MAX_LEN=${TRC_MAX_LEN:-16384} (log $log)"
  until trc_serve_healthy "$port" "$served"; do
    if ! kill -0 "$TRC_SERVER_PID" 2>/dev/null; then
      trc_say "vLLM exited during startup; last lines of $log:"; tail -n 40 "$log"; TRC_SERVER_PID=""; return 5
    fi
    if [ "$waited" -ge "$TRC_SERVER_TIMEOUT" ]; then
      trc_say "vLLM not ready after ${TRC_SERVER_TIMEOUT}s; last lines of $log:"; tail -n 40 "$log"; return 5
    fi
    sleep 10; waited=$((waited + 10))
    [ $((waited % 60)) -eq 0 ] && trc_log "waiting for vLLM (${waited}s)"
  done
  trc_say "vLLM ready after ${waited}s: $served on port $port"
}
trc_serve_stop() {
  if [ -n "${TRC_SERVER_PID:-}" ]; then trc_log "stopping vLLM (pid $TRC_SERVER_PID)"; trc_stop_tree "$TRC_SERVER_PID" vLLM 120; fi
  TRC_SERVER_PID=""
}
