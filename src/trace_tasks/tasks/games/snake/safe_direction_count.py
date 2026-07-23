from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SnakeLifecycleTask, build_integer_snake_objective, run_snake_task, select_snake_integer_target
from .shared.defaults import DEFAULTS
from .shared.rules import coord_to_cell_id, safe_next_directions, step_coord, validate_snake_state
from .shared.sampling import random_snake_state, sample_obstacles, with_obstacles
from .shared.state import SnakeSample


TASK_ID = "task_games__snake__safe_direction_count"
PROMPT_KEY = "safe_direction_count"


def _prepare_safe_direction_objective(attempt_seed, task_params, axes):
    # Role: construct states whose immediate safe destinations exactly match the target count.
    target, target_probabilities = select_snake_integer_target(
        task_params,
        objective_key=PROMPT_KEY,
        fallback_support=DEFAULTS.safe_direction_count_support,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    rng = spawn_rng(int(attempt_seed), f"{TASK_ID}.sample")
    for _attempt in range(900):
        state = random_snake_state(
            rng=rng,
            board_size=int(axes.board_size),
            body_length=int(axes.body_length),
            prefer_edge_head=(int(target) <= 2),
        )
        try:
            obstacles = sample_obstacles(rng=rng, state=state, count=int(axes.obstacle_count))
        except ValueError:
            continue
        state = with_obstacles(state, obstacles)
        validate_snake_state(state)
        safe_directions = safe_next_directions(state)
        if len(safe_directions) != int(target):
            continue
        annotation_cell_ids = tuple(coord_to_cell_id(step_coord(state.head, direction)) for direction in safe_directions)
        sample = SnakeSample(
            answer=int(len(safe_directions)),
            state=state,
            annotation_cell_ids=annotation_cell_ids,
            construction_mode=f"safe_direction_count_{int(target)}",
            safe_directions=tuple(safe_directions),
        )
        return build_integer_snake_objective(
            sample=sample,
            prompt_key=PROMPT_KEY,
            prompt_json_example_answer=2,
            prompt_json_example_annotation_count=2,
            answer_support=list(DEFAULTS.safe_direction_count_support),
            annotation_cell_ids=tuple(sample.annotation_cell_ids),
            target_trace_key="target_safe_direction_count",
            target_value=int(target),
            target_probabilities=target_probabilities,
        )
    raise ValueError("failed to construct Snake safe-direction count sample")


@register_task
class GamesSnakeMoveSafetyTask(SnakeLifecycleTask):
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_snake_task(self, instance_seed, params, max_attempts, _prepare_safe_direction_objective)
