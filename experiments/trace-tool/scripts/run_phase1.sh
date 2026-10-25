#!/usr/bin/env bash
# Phase 1: Qwen2.5-VL (and Qwen3.5) on TRACE validation without vs. with tools (see EXPERIMENT_DESIGN.md).
#
#   STAGE=pilot MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/run_phase1.sh   # 1000 rows, seed 42, all conditions
#   STAGE=full  MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/run_phase1.sh   # 2000 rows, seeds 42-44, core conditions
#   (then MODEL=Qwen/Qwen2.5-VL-3B-Instruct; optionally maveryn/trace-qwen2.5-vl-{3b,7b})
#   STAGE=pilot MODEL=Qwen/Qwen3.5-9B scripts/run_phase1.sh               # the same grid plus the five *_forced conditions
#
# The default condition lists are extended by EXTRA_CONDITIONS (scripts/_model_env.sh: the
# *_forced conditions for Qwen/Qwen3.5-*, nothing otherwise); an explicit CONDITIONS is used as is.
#
# The endpoint must serve MODEL under SERVED (scripts/serve_vllm.sh derives the
# same default name). Extra arguments go to trace-tool-eval; DATASET_PATH points
# at a local copy of the validation parquet.
set -euo pipefail

STAGE=${STAGE:-pilot}
source "$(dirname "$0")/_model_env.sh"
ENDPOINT=${ENDPOINT:-http://127.0.0.1:18100/v1}
CONCURRENCY=${CONCURRENCY:-64}
DATA=()
if [ -n "${DATASET_PATH:-}" ]; then DATA=(--dataset-path "$DATASET_PATH"); fi
REV=()
if [ -n "$REVISION" ]; then REV=(--model-revision "$REVISION"); fi

case "$STAGE" in
  pilot)
    OUT=${OUT:-runs/pilot/$SERVED}
    CONDITIONS=${CONDITIONS:-no_tool,no_tool_budget,zoom,visual_ops,calculator,ocr,all,zoom_cookbook,oracle_crop,no_tool_lowres,zoom_lowres${EXTRA_CONDITIONS:+,$EXTRA_CONDITIONS}}
    SELECT=(--per-task 1)
    SEEDS=${SEEDS:-42}
    ;;
  full)
    OUT=${OUT:-runs/full/$SERVED}
    CONDITIONS=${CONDITIONS:-no_tool,no_tool_budget,zoom,visual_ops,calculator,ocr,all${EXTRA_CONDITIONS:+,$EXTRA_CONDITIONS}}
    SELECT=()
    SEEDS=${SEEDS:-42,43,44}
    ;;
  *) echo "STAGE must be pilot or full" >&2; exit 2 ;;
esac

trace-tool-eval --endpoint "$ENDPOINT" --model "$SERVED" --model-id "$MODEL" ${REV[@]+"${REV[@]}"} \
  --conditions "$CONDITIONS" --seeds "$SEEDS" ${SELECT[@]+"${SELECT[@]}"} ${DATA[@]+"${DATA[@]}"} \
  --concurrency "$CONCURRENCY" --out "$OUT" "$@"

trace-tool-score "$OUT" --report "$OUT/report.md" ${DATA[@]+"${DATA[@]}"}

# Holm family for the comparison: the core tool conditions (and no_tool_budget) that were run.
CMP=$(printf '%s\n' no_tool_budget zoom visual_ops calculator ocr all | grep -Fx -f <(tr ',' '\n' <<<"$CONDITIONS") | paste -sd, - || true)
# Separate Holm family: the forced-final-answer conditions that were run (Qwen3.5 by default).
FORCED=$(printf '%s\n' zoom_forced visual_ops_forced calculator_forced ocr_forced all_forced | grep -Fx -f <(tr ',' '\n' <<<"$CONDITIONS") | paste -sd, - || true)
printf -v QOUT '%q' "$OUT"
QDATA=""
if [ -n "${DATASET_PATH:-}" ]; then printf -v QDATA ' --dataset-path %q' "$DATASET_PATH"; fi
cat <<MSG

Generation done: $OUT
Judge pass for missing/ambiguous extractions (TRACE protocol; 16k context fits no_tool_budget's 8k responses):
  MODEL=Qwen/Qwen3-32B SERVED=qwen3-32b-judge REVISION=9216db5781bf21249d130ec9da846c4624c16137 PORT=18200 MAX_IMAGES=0 MAX_LEN=16384 scripts/serve_vllm.sh
  trace-tool-judge --pending $QOUT/judge_pending.jsonl --out $QOUT/judge_results.jsonl --endpoint http://127.0.0.1:18200/v1
  trace-tool-score $QOUT --judge-results $QOUT/judge_results.jsonl --report $QOUT/report.md$QDATA
Paired comparison (Holm family = the listed conditions):
  trace-tool-compare $QOUT --baseline no_tool${CMP:+ --conditions $CMP} --out $QOUT/compare.md
MSG
if [ -n "$FORCED" ]; then
  echo "Forced final answer (Holm family = the forced conditions), then each forced condition vs. its unforced base:"
  echo "  trace-tool-compare $QOUT --baseline no_tool --conditions $FORCED --out $QOUT/compare_forced.md"
  PAIRS=""
  for c in ${FORCED//,/ }; do
    if [ -d "$OUT/${c%_forced}" ]; then PAIRS+="    trace-tool-compare $QOUT --baseline ${c%_forced} --conditions $c"$'\n'; fi
  done
  if [ -n "$PAIRS" ]; then printf '  {\n%s  } > %s/compare_forced_vs_unforced.md\n' "$PAIRS" "$QOUT"; fi
fi
