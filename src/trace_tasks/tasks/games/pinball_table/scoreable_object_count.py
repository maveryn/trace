"""Count scoreable objects on a visible pinball table."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from ._lifecycle import AttemptPinballResult, ObjectivePinballPlan, run_pinball_lifecycle
from .shared.annotations import bbox_set_for_entity_ids
from .shared.defaults import PATH_SCORE_VALUES, SCENE_ID
from .shared.sampling import (
    PinballVisualAxes,
    sample_scoreable_object_count_playfield,
)


TASK_ID = "task_games__pinball_table__scoreable_object_count"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "scoreable_object_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
SCOREABLE_OBJECT_COUNT_SUPPORT = (1, 2, 3, 4, 5, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for pinball scoreable-count output."""

    return (
        json.dumps(
            {"annotation": [[238, 158, 282, 202], [398, 288, 442, 332], [548, 218, 592, 262]], "answer": 3},
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        json.dumps({"answer": 3}, separators=(",", ":"), ensure_ascii=False),
    )


def _resolve_scoreable_object_count(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    axes: PinballVisualAxes,
) -> tuple[int, dict[str, float], tuple[int, ...]]:
    """Resolve a scoreable count while keeping at least one distractor object."""

    max_scoreable = min(max(1, int(axes.object_count) - 1), max(SCOREABLE_OBJECT_COUNT_SUPPORT))
    raw_support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="scoreable_object_count_support",
        fallback=SCOREABLE_OBJECT_COUNT_SUPPORT,
    )
    support = tuple(int(value) for value in raw_support if 1 <= int(value) <= int(max_scoreable))
    if not support:
        raise ValueError("scoreable_object_count_support must contain a value below object_count")
    adjusted_params = dict(params)
    adjusted_params["scoreable_object_count_support"] = support
    scoreable_count, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=adjusted_params,
        gen_defaults={},
        support_key="scoreable_object_count_support",
        explicit_key="scoreable_object_count",
        fallback_support=support,
        namespace=f"{TASK_ID}.scoreable_object_count",
        balanced_flag_key="balanced_scoreable_object_count_sampling",
        namespace_support_permutation=True,
    )
    return int(scoreable_count), dict(probabilities), tuple(int(value) for value in support)


def _prepare_scoreable_count_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    axes: PinballVisualAxes,
) -> ObjectivePinballPlan:
    """Resolve the scoreable-count axis and bind the count constructor."""

    scoreable_count, count_probabilities, count_support = _resolve_scoreable_object_count(
        int(instance_seed),
        params=params,
        axes=axes,
    )

    def construct_attempt(rng: Any, attempt_axes: PinballVisualAxes) -> AttemptPinballResult:
        return _construct_scoreable_count_attempt(
            rng=rng,
            axes=attempt_axes,
            scoreable_count=int(scoreable_count),
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePinballPlan(
        attempt_namespace="games.pinball_table.scoreable_object_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        object_description_key="object_description_scoreable_count",
        answer_hint='set "answer" to the integer count of scoreable objects',
        annotation_hint='set "annotation" to bounding boxes [x0, y0, x1, y1], one around each object that shows a numeric score label',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "scoreable_object_count": int(scoreable_count),
            "scoreable_object_count_support": [int(value) for value in count_support],
            "scoreable_object_count_probabilities": dict(count_probabilities),
            "score_value_support": [int(value) for value in PATH_SCORE_VALUES],
        },
        construct_attempt=construct_attempt,
    )


def _construct_scoreable_count_attempt(
    *,
    rng: Any,
    axes: PinballVisualAxes,
    scoreable_count: int,
) -> AttemptPinballResult:
    """Construct a mixed-score table and bind scoreable-object centers."""

    construction = sample_scoreable_object_count_playfield(
        rng=rng,
        axes=axes,
        scoreable_count=int(scoreable_count),
    )
    scoreable_ids = tuple(str(entity_id) for entity_id in construction.annotation_entity_ids)
    scoreable_id_set = set(scoreable_ids)
    score_by_id = {
        str(obj.object_id): (None if obj.score_value is None else int(obj.score_value))
        for obj in construction.scene.objects
    }
    non_scoreable_ids = tuple(
        str(obj.object_id)
        for obj in construction.scene.objects
        if str(obj.object_id) not in scoreable_id_set
    )
    if len(scoreable_ids) != int(construction.scoreable_count):
        raise ValueError("pinball scoreable-count annotation ids do not match answer")
    if any(score_by_id[str(entity_id)] is None for entity_id in scoreable_ids):
        raise ValueError("pinball scoreable-count annotation ids must have numeric scores")
    if not non_scoreable_ids:
        raise ValueError("pinball scoreable-count scene must include non-scoreable distractors")
    if any(score_by_id[str(entity_id)] is not None for entity_id in non_scoreable_ids):
        raise ValueError("pinball scoreable-count distractor ids must not have numeric scores")

    trace_fields = {
        "scoreable_object_ids": list(scoreable_ids),
        "non_scoreable_object_ids": list(non_scoreable_ids),
        "scoreable_object_count": int(construction.scoreable_count),
        "scoreable_score_values": [int(value) for value in construction.score_values],
    }
    return AttemptPinballResult(
        scene=construction.scene,
        answer_gt=TypedValue(type="integer", value=int(construction.scoreable_count)),
        annotation_entity_ids=scoreable_ids,
        build_annotation=lambda rendered: bbox_set_for_entity_ids(rendered.rendered_scene, scoreable_ids),
        witness_type="object_set",
        relations_extra=trace_fields,
        execution_extra={
            **trace_fields,
            "score_value_support": [int(value) for value in PATH_SCORE_VALUES],
        },
    )


@register_task
class GamesPinballScoreableObjectCountTask:
    """Count objects that display numeric score labels on the pinball table."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_pinball_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_scoreable_count_objective,
        )


__all__ = ["GamesPinballScoreableObjectCountTask"]
