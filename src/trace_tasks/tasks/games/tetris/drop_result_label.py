"""Result-board option label after a fixed Tetris drop."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import TetrisObjectivePlan, resolve_tetris_integer_target, run_tetris_lifecycle
from .shared.defaults import DEFAULTS
from .shared.prompts import format_json_examples
from .shared.rendering import RENDER_MODE_RESULT_OPTIONS
from .shared.sampling import build_drop_result_sample, sample_label
from .shared.state import OPTION_LABELS


TASK_ID = "task_games__tetris__drop_result_label"
PROMPT_QUERY_KEY = "drop_result_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
RESULT_OPTION_COUNT = 4
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(annotation=[520, 180, 760, 520], answer="B")


def _prepare_drop_result_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
    axes,
) -> TetrisObjectivePlan:
    """Resolve the fixed-drop clear-count target and answer option label."""

    target_clear_count, target_probabilities, target_support = resolve_tetris_integer_target(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="drop_result_clear_count_support",
        explicit_key="target_clear_count",
        fallback_support=DEFAULTS.drop_result_clear_count_support,
        namespace=f"{TASK_ID}.target_clear_count",
        balanced_flag_key="balanced_target_clear_count_sampling",
    )
    labels = OPTION_LABELS[:RESULT_OPTION_COUNT]
    answer_label, answer_label_probabilities = sample_label(
        int(instance_seed),
        namespace=f"{TASK_ID}.answer_label.{RESULT_OPTION_COUNT}",
        labels=labels,
    )

    def construct_attempt(rng, resolved_axes):
        return build_drop_result_sample(
            rng,
            scene_variant=str(resolved_axes.scene_variant),
            board_rows=int(resolved_axes.board_rows),
            board_cols=int(resolved_axes.board_cols),
            target_clear_count=int(target_clear_count),
            option_count=int(RESULT_OPTION_COUNT),
            answer_label=str(answer_label),
        )

    return TetrisObjectivePlan(
        attempt_namespace=f"games.tetris.drop_result.{int(target_clear_count)}.{str(answer_label)}",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint_key="answer_hint_drop_result_label",
        annotation_hint_key="annotation_hint_drop_result_label",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        render_mode=RENDER_MODE_RESULT_OPTIONS,
        query_params={
            "target_clear_count": int(target_clear_count),
            "target_clear_count_support": [int(value) for value in target_support],
            "target_clear_count_probabilities": dict(target_probabilities),
            "option_count": int(RESULT_OPTION_COUNT),
            "result_option_count_policy": "fixed_four_options",
            "answer_label": str(answer_label),
            "answer_label_probabilities": dict(answer_label_probabilities),
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesTetrisDropResultLabelTask:
    """Choose the labeled board produced by the shown fixed drop."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
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
            prepare_objective=_prepare_drop_result_objective,
        )


__all__ = ["GamesTetrisDropResultLabelTask"]
