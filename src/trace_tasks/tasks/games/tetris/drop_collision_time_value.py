"""Downward timestep count before a shifted Tetris piece collides."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import TetrisObjectivePlan, resolve_tetris_integer_target, run_tetris_lifecycle
from .shared.defaults import DEFAULTS
from .shared.prompts import format_json_examples
from .shared.rendering import RENDER_MODE_COLLISION
from .shared.rules import shift_instruction_text
from .shared.sampling import build_drop_collision_time_sample


TASK_ID = "task_games__tetris__drop_collision_time_value"
SUPPORTED_QUERY_IDS = ("no_shift_collision_time", "left_shift_collision_time", "right_shift_collision_time")
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(
    annotation={
        "start_piece": [[120, 80, 150, 110], [154, 80, 184, 110]],
        "stop_witness": [[120, 320, 150, 350]],
    },
    answer=5,
)


def _resolve_shift_delta(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query: str,
) -> tuple[int, dict[str, float], tuple[int, ...]]:
    """Turn a public shift branch into a signed column offset."""

    if str(selected_query) == "no_shift_collision_time":
        return 0, {"0": 1.0}, (0,)
    if str(selected_query) not in {"left_shift_collision_time", "right_shift_collision_time"}:
        raise ValueError(f"unsupported Tetris collision-time query: {selected_query}")
    magnitude, probabilities, support = resolve_tetris_integer_target(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="shift_magnitude_support",
        explicit_key="shift_magnitude",
        fallback_support=DEFAULTS.shift_magnitude_support,
        namespace=f"{TASK_ID}.{str(selected_query)}.shift_magnitude",
        balanced_flag_key="balanced_shift_magnitude_sampling",
    )
    signed = -int(magnitude) if str(selected_query) == "left_shift_collision_time" else int(magnitude)
    return int(signed), dict(probabilities), tuple(int(value) for value in support)


def _prepare_drop_collision_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query: str,
    query_probabilities: Mapping[str, float],
    _axes,
) -> TetrisObjectivePlan:
    """Resolve horizontal shift and target timestep count for construction."""

    shift_delta, shift_probabilities, shift_support = _resolve_shift_delta(
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query=str(selected_query),
    )
    target_drop_steps, target_probabilities, target_support = resolve_tetris_integer_target(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="drop_collision_time_support",
        explicit_key="target_drop_steps",
        fallback_support=DEFAULTS.drop_collision_time_support,
        namespace=f"{TASK_ID}.{str(selected_query)}.target_drop_steps",
    )
    shift_instruction = shift_instruction_text(int(shift_delta))

    def construct_attempt(rng, axes):
        return build_drop_collision_time_sample(
            rng,
            scene_variant=str(axes.scene_variant),
            board_rows=int(axes.board_rows),
            board_cols=int(axes.board_cols),
            shift_delta=int(shift_delta),
            target_drop_steps=int(target_drop_steps),
        )

    return TetrisObjectivePlan(
        attempt_namespace=f"games.tetris.collision.{str(selected_query)}.{int(target_drop_steps)}",
        prompt_query_key=str(selected_query),
        answer_hint_key="answer_hint_drop_collision_time_value",
        annotation_hint_key="annotation_hint_drop_collision_time_value",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        render_mode=RENDER_MODE_COLLISION,
        query_params={
            "shift_delta": int(shift_delta),
            "shift_instruction": str(shift_instruction),
            "shift_magnitude_support": [int(value) for value in shift_support],
            "shift_magnitude_probabilities": dict(shift_probabilities),
            "target_drop_steps": int(target_drop_steps),
            "target_drop_steps_support": [int(value) for value in target_support],
            "target_drop_steps_probabilities": dict(target_probabilities),
            "collision_time_query_probabilities": dict(query_probabilities),
        },
        dynamic_prompt_slots={"shift_instruction": str(shift_instruction)},
        construct_attempt=construct_attempt,
    )


@register_task
class GamesTetrisDropCollisionTimeValueTask:
    """Count successful downward moves after the requested horizontal shift."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_tetris_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_drop_collision_objective,
        )


__all__ = ["GamesTetrisDropCollisionTimeValueTask"]
