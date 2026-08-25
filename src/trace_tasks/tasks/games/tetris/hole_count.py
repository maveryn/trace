"""Count covered hole cells in a static Tetris stack."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import TetrisObjectivePlan, resolve_tetris_integer_target, run_tetris_lifecycle
from .shared.defaults import DEFAULTS
from .shared.prompts import format_json_examples
from .shared.rendering import RENDER_MODE_STATIC_BOARD
from .shared.sampling import build_hole_count_sample


TASK_ID = "task_games__tetris__hole_count"
SUPPORTED_QUERY_IDS = ("hole_count",)
HOLE_RULE_TEXT = "A hole is an empty cell with at least one filled block above it in the same column."
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(
    annotation=[[172, 414, 196, 438], [276, 466, 300, 490]],
    answer=2,
)


def _prepare_hole_count_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query: str,
    query_probabilities: Mapping[str, float],
    _axes,
) -> TetrisObjectivePlan:
    """Resolve the exact covered-hole target for one static stack."""

    target_hole_count, target_probabilities, target_support = resolve_tetris_integer_target(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="hole_count_support",
        explicit_key="target_hole_count",
        fallback_support=DEFAULTS.hole_count_support,
        namespace=f"{TASK_ID}.target_hole_count",
    )

    def construct_attempt(rng, axes):
        return build_hole_count_sample(
            rng,
            scene_variant=str(axes.scene_variant),
            board_rows=int(axes.board_rows),
            board_cols=int(axes.board_cols),
            target_hole_count=int(target_hole_count),
        )

    return TetrisObjectivePlan(
        attempt_namespace=f"games.tetris.hole_count.{int(target_hole_count)}",
        prompt_query_key="hole_count",
        answer_hint_key="answer_hint_hole_count",
        annotation_hint_key="annotation_hint_hole_count",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        render_mode=RENDER_MODE_STATIC_BOARD,
        query_params={
            "target_hole_count": int(target_hole_count),
            "target_hole_count_support": [int(value) for value in target_support],
            "target_hole_count_probabilities": dict(target_probabilities),
            "hole_count_query_probabilities": dict(query_probabilities),
        },
        dynamic_prompt_slots={"hole_rule_text": str(HOLE_RULE_TEXT)},
        construct_attempt=construct_attempt,
    )


@register_task
class GamesTetrisHoleCountTask:
    """Count empty cells covered by locked blocks in the shown stack."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
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
            prepare_objective=_prepare_hole_count_objective,
        )


__all__ = ["GamesTetrisHoleCountTask"]
