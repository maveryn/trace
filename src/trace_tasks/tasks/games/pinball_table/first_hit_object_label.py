"""Identify the first pinball object hit by the launch cue."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import AttemptPinballResult, ObjectivePinballPlan, run_pinball_lifecycle
from .shared.annotations import point_for_entity_id
from .shared.defaults import SCENE_ID
from .shared.sampling import (
    PinballVisualAxes,
    first_hit_object_id,
    resolve_pinball_target_label,
    sample_unique_first_hit_playfield,
)


TASK_ID = "task_games__pinball_table__first_hit_object_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "first_hit_object_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for pinball first-hit output."""

    return (
        json.dumps({"annotation": [410, 260], "answer": "D"}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": "D"}, separators=(",", ":"), ensure_ascii=False),
    )


def _prepare_first_hit_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    axes: PinballVisualAxes,
) -> ObjectivePinballPlan:
    """Resolve the visible target label and bind the first-hit constructor."""

    target_axis = resolve_pinball_target_label(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{TASK_ID}.target_object_label",
        object_count=int(axes.object_count),
    )

    def construct_attempt(rng: Any, attempt_axes: PinballVisualAxes) -> AttemptPinballResult:
        return _construct_first_hit_attempt(
            rng=rng,
            axes=attempt_axes,
            target_object_label=str(target_axis.target_object_label),
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePinballPlan(
        attempt_namespace="games.pinball_table.first_hit_object_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        object_description_key=f"object_description_{str(axes.scene_variant)}",
        answer_hint='set "answer" to the selected object label as a string',
        annotation_hint='set "annotation" to one [x, y] pixel point at the center of the first labeled object hit by the path',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "target_object_label": str(target_axis.target_object_label),
            "target_object_label_probabilities": dict(target_axis.target_object_label_probabilities),
        },
        construct_attempt=construct_attempt,
    )


def _validate_first_hit_binding(construction: Any) -> None:
    """Verify the launch ray still binds the selected object uniquely."""

    target_object = next(
        obj for obj in construction.scene.objects
        if str(obj.object_id) == str(construction.target_object_id)
    )
    if str(construction.target_object_label) != str(target_object.label):
        raise ValueError("pinball target label does not match target object")
    recomputed_first_hit = first_hit_object_id(
        origin=(float(construction.scene.ball_x_norm), float(construction.scene.ball_y_norm)),
        angle_rad=float(construction.scene.cue_angle_rad),
        objects=construction.scene.objects,
    )
    if str(recomputed_first_hit) != str(construction.target_object_id):
        raise ValueError("pinball first-hit construction lost unique target binding")


def _first_hit_trace_fields(construction: Any) -> dict[str, str]:
    """Return trace fields that identify the selected first-hit object."""

    return {
        "target_object_id": str(construction.target_object_id),
        "target_object_label": str(construction.target_object_label),
    }


def _construct_first_hit_attempt(
    *,
    rng: Any,
    axes: PinballVisualAxes,
    target_object_label: str,
) -> AttemptPinballResult:
    """Construct a scene and bind the unique first-hit object as answer."""

    construction = sample_unique_first_hit_playfield(
        rng=rng,
        axes=axes,
        target_object_label=str(target_object_label),
    )
    _validate_first_hit_binding(construction)
    annotation_ids = (str(construction.target_object_id),)
    trace_fields = _first_hit_trace_fields(construction)
    return AttemptPinballResult(
        scene=construction.scene,
        answer_gt=TypedValue(type="string", value=str(construction.target_object_label)),
        annotation_entity_ids=annotation_ids,
        build_annotation=lambda rendered: point_for_entity_id(rendered.rendered_scene, annotation_ids[0]),
        witness_type="object_set",
        relations_extra=trace_fields,
        execution_extra=trace_fields,
    )


@register_task
class GamesPinballFirstHitObjectLabelTask:
    """Identify the first labeled object hit by the straight launch cue."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations', 'topology')
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
            prepare_objective=_prepare_first_hit_objective,
        )


__all__ = ["GamesPinballFirstHitObjectLabelTask"]
