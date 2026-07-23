from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SnakeLifecycleTask, build_integer_snake_objective, run_snake_task, select_snake_integer_target
from .shared.defaults import DEFAULTS
from .shared.rules import coord_to_cell_id, validate_snake_state
from .shared.sampling import random_snake_state, sample_obstacles, with_obstacles
from .shared.state import SnakeSample


TASK_ID = "task_games__snake__snake_length_count"
PROMPT_KEY = "snake_length_count"


def _prepare_snake_length_objective(attempt_seed, task_params, axes):
    # Role: construct a snake whose occupied head/body cells exactly match the target length.
    target, target_probabilities = select_snake_integer_target(
        task_params,
        objective_key=PROMPT_KEY,
        fallback_support=DEFAULTS.snake_length_count_support,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    rng = spawn_rng(int(attempt_seed), f"{TASK_ID}.sample")
    for _attempt in range(900):
        state = random_snake_state(
            rng=rng,
            board_size=int(axes.board_size),
            body_length=max(1, int(target) - 1),
        )
        try:
            obstacles = sample_obstacles(rng=rng, state=state, count=int(axes.obstacle_count))
        except ValueError:
            continue
        state = with_obstacles(state, obstacles)
        validate_snake_state(state)
        length = int(len(state.body) + 1)
        if length != int(target):
            continue
        annotation_cell_ids = tuple(coord_to_cell_id(coord) for coord in (state.head, *state.body))
        sample = SnakeSample(
            answer=int(length),
            state=state,
            annotation_cell_ids=annotation_cell_ids,
            construction_mode=f"snake_length_count_{int(target)}",
        )
        return build_integer_snake_objective(
            sample=sample,
            prompt_key=PROMPT_KEY,
            prompt_json_example_answer=8,
            prompt_json_example_annotation_count=8,
            answer_support=list(DEFAULTS.snake_length_count_support),
            annotation_cell_ids=tuple(sample.annotation_cell_ids),
            target_trace_key="target_snake_length_count",
            target_value=int(target),
            target_probabilities=target_probabilities,
        )
    raise ValueError("failed to construct Snake length-count sample")


@register_task
class GamesSnakeLengthCountTask(SnakeLifecycleTask):
    task_id = TASK_ID
    reasoning_operations = ('counting',)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_snake_task(self, instance_seed, params, max_attempts, _prepare_snake_length_objective)
