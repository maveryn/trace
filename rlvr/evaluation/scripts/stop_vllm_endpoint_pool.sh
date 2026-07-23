#!/usr/bin/env bash
set -euo pipefail

: "${PID_FILE:?Set PID_FILE to the pids.txt written by start_vllm_endpoint_pool.sh.}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[stop] missing pid file: ${PID_FILE}" >&2
  exit 1
fi

while read -r pid port group log; do
  [[ -n "${pid:-}" ]] || continue
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] pid=${pid} port=${port:-?} gpus=${group:-?}"
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
done < "${PID_FILE}"

sleep "${STOP_GRACE_SEC:-10}"

while read -r pid port group log; do
  [[ -n "${pid:-}" ]] || continue
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[kill] pid=${pid} port=${port:-?} gpus=${group:-?}"
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
done < "${PID_FILE}"
