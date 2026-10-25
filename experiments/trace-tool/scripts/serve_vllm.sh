#!/usr/bin/env bash
# Serve a Qwen-VL checkpoint (default: TRACE's Qwen2.5-VL-7B) for trace-tool-eval.
#
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct PORT=18100 scripts/serve_vllm.sh
#
# MODEL / SERVED / REVISION come from scripts/_model_env.sh: SERVED defaults to
# the lowercased basename of MODEL (qwen2.5-vl-7b-instruct) and REVISION to
# TRACE's pinned revision for its models; run_phase1.sh and smoke_test.sh agree.
# Images are resized client-side to the condition's pixel bounds (multiples of
# 28 for Qwen2.5-VL, 32 for Qwen3-VL), so the server's own resizing is a no-op. --generation-config vllm stops
# the checkpoint's generation_config.json (top_k=20 etc.) from overriding the
# sampling parameters each request sends. --chat-template-content-format openai
# keeps content lists intact so <tool_response> text and images render in order.
# Text-only models (MAX_IMAGES=0, e.g. the Qwen3-32B judge) get `string` instead:
# Qwen3's template concatenates message content as a string, and under `openai`
# every judge request failed with HTTP 400 "can only concatenate str (not
# "list") to str" (2026-10-04 Phase 1 run).
# The image limit covers 1 input image + 5 tool rounds x 2 calls.
set -euo pipefail

source "$(dirname "$0")/_model_env.sh"
PORT=${PORT:-18100}
TP=${TP:-1}
DP=${DP:-1}
MAX_LEN=${MAX_LEN:-32768}
MAX_IMAGES=${MAX_IMAGES:-12}

# only vision models take the image limit (vLLM <= 0.10.0 rejects it for text-only models)
MM=()
CONTENT_FORMAT=string
if [ "$MAX_IMAGES" != 0 ]; then
  MM=(--limit-mm-per-prompt "{\"image\": $MAX_IMAGES, \"video\": 0}")
  CONTENT_FORMAT=openai
fi

exec vllm serve "$MODEL" ${REVISION:+--revision "$REVISION"} \
  --served-model-name "$SERVED" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len "$MAX_LEN" \
  ${MM[@]+"${MM[@]}"} \
  --chat-template-content-format "$CONTENT_FORMAT" \
  --generation-config vllm \
  "$@"
