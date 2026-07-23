"""Solitaire column visible-card count task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SolitaireLifecycleTask, SolitaireObjective, run_solitaire_lifecycle
from .shared.annotations import entity_point_set
from .shared.sampling import sample_column_card_count


TASK_ID = "task_games__solitaire__column_card_count_value"
PROMPT_QUERY_KEY = "column_card_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
JSON_EXAMPLE = '{"annotation":[[287,236],[287,268],[287,336]],"answer":3}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":3}'


def _prepare_column_card_count_objective(rng, params: Mapping[str, Any], scene_variant: str, instance_seed: int) -> SolitaireObjective:
    """Construct a tableau with a controlled visible-card count in one column."""

    sample = sample_column_card_count(
        rng,
        namespace=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        scene_variant=str(scene_variant),
    )
    return SolitaireObjective(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        build_annotation=entity_point_set,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        prompt_slots={"target_column_number": str(sample.metadata["target_column_number"])},
    )


@register_task
class GamesSolitaireColumnCardCountValueTask(SolitaireLifecycleTask):
    """Count visible cards in a specified tableau column."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_solitaire_lifecycle(
            namespace=TASK_ID,
            prompt_query_key=PROMPT_QUERY_KEY,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=DEFAULT_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_column_card_count_objective,
        )


__all__ = ["GamesSolitaireColumnCardCountValueTask"]
