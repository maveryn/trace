"""Solitaire cascade card-at-depth option task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from ._lifecycle import SolitaireLifecycleTask, SolitaireObjective, run_solitaire_lifecycle
from .shared.annotations import entity_point
from .shared.defaults import DEFAULTS, GEN_DEFAULTS
from .shared.sampling import sample_cascade_card_at_depth


TASK_ID = "task_games__solitaire__cascade_card_at_depth_label"
PROMPT_QUERY_KEY = "cascade_card_at_depth_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
JSON_EXAMPLE = '{"annotation":[323,306],"answer":"B"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"B"}'


def _target_depth(instance_seed: int, params: Mapping[str, Any]) -> tuple[int, dict[str, float]]:
    """Resolve the card depth operand for this public task."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="cascade_depth_support",
        explicit_key="target_depth",
        fallback_support=DEFAULTS.cascade_depth_support,
        namespace=f"{TASK_ID}.target_depth",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return int(value), dict(probabilities)


def _prepare_cascade_card_at_depth_objective(
    rng,
    params: Mapping[str, Any],
    scene_variant: str,
    instance_seed: int,
) -> SolitaireObjective:
    """Construct a tableau and bind the selected card-face option label."""

    depth, depth_probabilities = _target_depth(int(instance_seed), params)
    raw_column = params.get("target_column")
    target_column = None if raw_column is None else int(raw_column)
    sample = sample_cascade_card_at_depth(
        rng,
        namespace=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        scene_variant=str(scene_variant),
        target_depth=int(depth),
        target_column=target_column,
    )
    sample.metadata["target_depth_probabilities"] = dict(depth_probabilities)
    return SolitaireObjective(
        sample=sample,
        answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        build_annotation=entity_point,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        prompt_slots={
            "target_column_number": str(sample.metadata["target_column_number"]),
            "target_depth_ordinal": str(sample.metadata["target_depth_ordinal"]),
        },
    )


@register_task
class GamesSolitaireCascadeCardAtDepthLabelTask(SolitaireLifecycleTask):
    """Choose the card at a requested visible depth in one tableau column."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
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
            build_objective=_prepare_cascade_card_at_depth_objective,
        )


__all__ = ["GamesSolitaireCascadeCardAtDepthLabelTask"]
