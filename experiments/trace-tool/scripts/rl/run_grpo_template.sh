#!/usr/bin/env bash
# UNTESTED TEMPLATE (Phase 2). GRPO on TRACE train, with or without tools, on verl's agent loop.
# Both arms run in the same framework with the same hyperparameters; only the
# agent loop and tool config differ. Hyperparameters mirror TRACE's EasyR1
# recipe (rlvr/configs/trace-qwen2.5-vl-7b.yaml); see EXPERIMENT_DESIGN.md §6.
#
# Before a full run:
#   python -m verl.trainer.main_ppo <these overrides> --cfg job    # check every key exists in your verl
#   ... trainer.total_training_steps=2 trainer.val_before_train=False   # 2-step smoke run per arm
#
#   ARM=notool scripts/rl/run_grpo_template.sh
#   ARM=zoom   scripts/rl/run_grpo_template.sh
#   ARM=all    scripts/rl/run_grpo_template.sh
set -xeuo pipefail

ARM=${ARM:?set ARM=notool|zoom|all}
# TRACE's base model with the tool chat template installed (used for every arm):
#   python -m trace_tool.rl.prepare_policy --model Qwen/Qwen2.5-VL-7B-Instruct --out models/qwen2.5-vl-7b-tools
MODEL_PATH=${MODEL_PATH:-models/qwen2.5-vl-7b-tools}
DATA_DIR=${DATA_DIR:-data/verl}
N_GPUS=${N_GPUS:-8}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

case "$ARM" in
  notool) AGENT=single_turn_agent; TOOL_CFG=""; RESPONSE_LEN=2048 ;;
  zoom)   AGENT=tool_agent; TOOL_CFG=$REPO/configs/verl/tools_zoom.yaml; RESPONSE_LEN=10240 ;;
  all)    AGENT=tool_agent; TOOL_CFG=$REPO/configs/verl/tools_all.yaml;  RESPONSE_LEN=10240 ;;
  *) echo "unknown ARM" >&2; exit 2 ;;
esac

# Data (agent_name column selects the agent loop per row):
#   python -m trace_tool.rl.prepare_verl_data --split train      --agent $AGENT --out $DATA_DIR/train_$AGENT.parquet
#   python -m trace_tool.rl.prepare_verl_data --split validation --agent $AGENT --out $DATA_DIR/val_$AGENT.parquet
TRAIN_FILE=$DATA_DIR/train_$AGENT.parquet
VAL_FILE=$DATA_DIR/val_$AGENT.parquet

ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.image_key=images
  data.train_batch_size=128
  data.max_prompt_length=4096
  data.max_response_length=$RESPONSE_LEN
  data.filter_overlong_prompts=True
  data.truncation=error
  # TODO verify: image pixel bounds should be 262144..4194304 (TRACE); check how your verl
  # version sets processor min/max pixels. configs/verl/tools_*.yaml set family: qwen2.5-vl;
  # change it to qwen3-vl for a Qwen3-VL policy.
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.ppo_mini_batch_size=128
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.3
  actor_rollout_ref.actor.clip_ratio_c=3.0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95
  actor_rollout_ref.rollout.val_kwargs.n=1
  reward.custom_reward_function.path="$REPO/src/trace_tool/rl/reward.py"
  reward.custom_reward_function.name=compute_score
  trainer.total_training_steps=500
  trainer.save_freq=100
  trainer.test_freq=100
  trainer.n_gpus_per_node=$N_GPUS
  trainer.nnodes=1
  trainer.project_name=trace_tool_rl
  trainer.experiment_name="$(basename "$MODEL_PATH")-$ARM"
  trainer.logger='["console","wandb"]'
)

if [ -n "$TOOL_CFG" ]; then
  ARGS+=(
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CFG"
    actor_rollout_ref.rollout.multi_turn.format=hermes
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2
    # verl's default (256 chars) would cut OCR output
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=6000
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
  )
fi

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@"
