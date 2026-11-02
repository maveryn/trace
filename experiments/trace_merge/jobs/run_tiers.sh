#!/usr/bin/env bash
# experiments/trace_merge/jobs/run_tiers.sh VIEW_SLUG SUBSET:SEED [SUBSET:SEED ...]
#
# One queue-runner task line: evaluate one inference view on several frozen row lists in order (e.g. the smoke list,
# then the development screen), stopping at the first tier that does not complete, so a broken loader or extractor
# never reaches the larger tiers. Each tier is one eval_subset.sh call with rows $TVM_WORK/splits/rows.<SUBSET>.json.
# A view that is still being built (merges) is waited for up to TVM_VIEW_WAIT seconds (default 3600).
# Exit: the first non-zero eval_subset.sh exit code; 6 when a tier finalized its generation but its judge is pending on
# another node (PENDING_JUDGE.seed<S>: the marker's --judge-only task line must run before the later tiers); else 0.
set -uo pipefail
[ $# -ge 2 ] || { echo "usage: run_tiers.sh VIEW_SLUG SUBSET:SEED [SUBSET:SEED ...]" >&2; exit 2; }
: "${TVM_WORK:?source the study env file first}"
JOBS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLUG=$1; shift
VIEW=$TVM_WORK/views/$SLUG
waited=0
until [ -f "$VIEW/view_manifest.json" ]; do
  if [ "$waited" -ge "${TVM_VIEW_WAIT:-3600}" ]; then echo "FATAL: view $VIEW not built after ${waited}s" >&2; exit 4; fi
  sleep 60; waited=$((waited + 60))
done
for spec in "$@"; do
  subset=${spec%%:*}; seed=${spec##*:}
  rows=$TVM_WORK/splits/rows.$subset.json
  echo "[run_tiers] $(date '+%F %T') $SLUG $subset seed $seed"
  bash "$JOBS/eval_subset.sh" "$VIEW" "$SLUG" --seed "$seed" --rows "$rows" --subset "$subset"
  rc=$?
  echo "[run_tiers] $(date '+%F %T') $SLUG $subset seed $seed rc=$rc"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
  if [ -f "${TVM_RUNS:-$TVM_WORK/runs}/$SLUG/$subset/PENDING_JUDGE.seed$seed" ]; then
    echo "[run_tiers] $(date '+%F %T') $SLUG $subset seed $seed rc=6: judge pending on another node; later tiers wait for it"
    exit 6
  fi
done
exit 0
