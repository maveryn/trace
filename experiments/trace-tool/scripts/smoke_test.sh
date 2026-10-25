#!/usr/bin/env bash
# 20 rows x every condition against a running endpoint, then score. ~5 min on one GPU.
#
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/smoke_test.sh [extra trace-tool-eval flags]
#
# The default condition list is extended by EXTRA_CONDITIONS (scripts/_model_env.sh: the
# *_forced conditions for Qwen/Qwen3.5-*, nothing otherwise); an explicit CONDITIONS is used as is.
#
# DATASET_PATH=/path/to/validation.parquet uses a local copy instead of the Hub.
set -euo pipefail

source "$(dirname "$0")/_model_env.sh"
ENDPOINT=${ENDPOINT:-http://127.0.0.1:18100/v1}
OUT=${OUT:-runs/smoke/$SERVED}
CONDITIONS=${CONDITIONS:-no_tool,no_tool_budget,zoom,visual_ops,calculator,ocr,all,zoom_cookbook,oracle_crop,no_tool_lowres,zoom_lowres${EXTRA_CONDITIONS:+,$EXTRA_CONDITIONS}}
DATA=()
if [ -n "${DATASET_PATH:-}" ]; then DATA=(--dataset-path "$DATASET_PATH"); fi
REV=()
if [ -n "$REVISION" ]; then REV=(--model-revision "$REVISION"); fi

trace-tool-eval --endpoint "$ENDPOINT" --model "$SERVED" --model-id "$MODEL" ${REV[@]+"${REV[@]}"} --conditions "$CONDITIONS" \
  --seeds 42 --per-task 1 --limit 20 --concurrency 20 --out "$OUT" ${DATA[@]+"${DATA[@]}"} "$@"
trace-tool-score "$OUT" --report "$OUT/report.md" ${DATA[@]+"${DATA[@]}"}

# Eyeball tool use: valid calls, sensible boxes, a final JSON answer.
python - "$OUT" <<'PY'
import json, sys, pathlib
for path in sorted(pathlib.Path(sys.argv[1]).glob("*/seed42/responses.jsonl")):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    calls = [c for r in rows for c in r["tool_calls"]]
    bad = [c for c in calls if not c.get("ok")]
    forced = ""  # forced-final-answer conditions only: how many episodes were forced, and why
    if any("forced_turn" in r for r in rows):
        reasons = sorted({str(r.get("forced_reason")) for r in rows if r.get("forced_turn")})
        forced = f" forced={sum(bool(r.get('forced_turn')) for r in rows):3d} {reasons}"
    print(f"{path.parent.parent.name:16s} rows={len(rows):3d} calls={len(calls):3d} not_ok={len(bad):3d} "
          f"stop={sorted({r['stop_reason'] for r in rows})}{forced}")
    for c in bad[:2]:
        print("    not ok:", c.get("name"), json.dumps(c.get("arguments"))[:120], "->",
              (c.get("text") or c.get("parse_error") or c.get("skipped") or "")[:120])
PY
