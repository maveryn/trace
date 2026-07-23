"""Solitaire legal-move option task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SolitaireLifecycleTask, SolitaireObjective, run_solitaire_lifecycle
from .shared.annotations import move_legality_bbox_map
from .shared.sampling import sample_move_legality


TASK_ID = "task_games__solitaire__move_legality_label"
PROMPT_QUERY_KEY = "move_legality_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
JSON_EXAMPLE = '{"annotation":{"source_card":[250,220,324,324],"target":[342,220,416,324]},"answer":"C"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"C"}'


def _prepare_move_legality_objective(rng, params: Mapping[str, Any], scene_variant: str, instance_seed: int) -> SolitaireObjective:
    """Construct one unique legal move option and bind source/target roles."""

    sample = sample_move_legality(
        rng,
        namespace=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        scene_variant=str(scene_variant),
    )
    return SolitaireObjective(
        sample=sample,
        answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        build_annotation=move_legality_bbox_map,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
    )


@register_task
class GamesSolitaireMoveLegalityLabelTask(SolitaireLifecycleTask):
    """Choose the one visible move option that is legal now."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
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
            build_objective=_prepare_move_legality_objective,
        )


__all__ = ["GamesSolitaireMoveLegalityLabelTask"]
