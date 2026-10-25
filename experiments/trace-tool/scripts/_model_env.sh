# Sourced by the scripts: checkpoint, served name, revision and extra conditions.
#   MODEL     HF id (default: TRACE's 7B base model)
#   SERVED    name the endpoint serves it under (default: lowercased basename of MODEL)
#   REVISION  checkpoint revision (default: the pinned revision for the models below;
#             set REVISION= to use the latest)
#   EXTRA_CONDITIONS  comma-separated conditions appended to the scripts' default condition
#             lists (default: the five *_forced conditions for Qwen/Qwen3.5-*, none otherwise;
#             set EXTRA_CONDITIONS= to add none). An explicit CONDITIONS ignores it.
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
SERVED=${SERVED:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
if [ -z "${REVISION+set}" ]; then
  case "$MODEL" in
    # rlvr/evaluation/trace_validation/suite.v1.json in github.com/maveryn/trace
    Qwen/Qwen2.5-VL-3B-Instruct) REVISION=66285546d2b821cf421d4f5eb2576359d3770cd3 ;;
    Qwen/Qwen2.5-VL-7B-Instruct) REVISION=cc594898137f460bfe9f0759e9844b3ce807cfb5 ;;
    maveryn/trace-qwen2.5-vl-3b) REVISION=2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5 ;;
    maveryn/trace-qwen2.5-vl-7b) REVISION=4d0f1ae8ee25022058090dbdbff61957ece7331d ;;
    # ours: the snapshots whose chat template and vision config this repo was checked against
    Qwen/Qwen3.5-4B) REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ;;
    Qwen/Qwen3.5-9B) REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a ;;
    *) REVISION= ;;
  esac
fi
if [ -z "${EXTRA_CONDITIONS+set}" ]; then
  case "$MODEL" in
    Qwen/Qwen3.5-*) EXTRA_CONDITIONS=zoom_forced,visual_ops_forced,calculator_forced,ocr_forced,all_forced ;;
    *) EXTRA_CONDITIONS= ;;
  esac
fi
# Server context for scripts/serve_vllm.sh (an explicit MAX_LEN, e.g. the judge's 16384, wins). Qwen3.5
# (32-px grid, up to ~4K tokens per 4.19 MP tool image, 11 images per episode) gets 65536 so a long
# visual_ops/all episode is not rejected for prompt length; the Qwen2.5-VL runs used the 32768 default.
if [ -z "${MAX_LEN:-}" ]; then
  case "$MODEL" in
    Qwen/Qwen3.5-*) MAX_LEN=65536 ;;
  esac
fi
