"""Identify the platform reached by a shown jump arc."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import AttemptPlatformerResult, ObjectivePlatformerPlan, run_platformer_lifecycle
from .shared.annotations import bbox_for_entity_id
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.sampling import (
    PlatformerVisualAxes,
    resolve_platformer_label_axis,
    sample_landing_scene,
)
from .shared.state import PlatformerSample


TASK_ID = "task_games__platformer__jump_landing_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "jump_landing_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class GamesPlatformerJumpLandingLabelTask:
    """Identify the labeled platform reached by a jump arc."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_platformer_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_landing_objective,
        )


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for platformer landing output."""

    return (
        json.dumps({"annotation": [604, 398, 784, 442], "answer": "D"}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": "D"}, separators=(",", ":"), ensure_ascii=False),
    )


def _validate_landing_sample(sample: PlatformerSample, *, target_label: str) -> None:
    """Verify the sampled landing platform is bound to the target label."""

    if sample.target_platform_id is None or sample.target_platform_label is None:
        raise ValueError("landing sample requires a target platform")
    if str(sample.target_platform_label) != str(target_label):
        raise ValueError("landing target label was not preserved")
    if str(sample.answer) != str(target_label):
        raise ValueError("landing answer does not match the target platform label")
    if tuple(sample.annotation_entity_ids) != (str(sample.target_platform_id),):
        raise ValueError("landing annotation must contain only the target platform")


def _prepare_landing_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    axes: PlatformerVisualAxes,
) -> ObjectivePlatformerPlan:
    """Resolve the target platform label and bind the landing constructor."""

    target_axis = resolve_platformer_label_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="target_platform_label_support",
        explicit_key="target_platform_label",
        fallback_support=DEFAULTS.target_platform_label_support,
        namespace=f"{TASK_ID}.target_platform_label",
        balanced_flag_key="balanced_target_platform_label_sampling",
    )

    def construct_attempt(rng: Any, attempt_axes: PlatformerVisualAxes) -> AttemptPlatformerResult:
        sample = sample_landing_scene(
            rng=rng,
            axes=attempt_axes,
            target_platform_label=str(target_axis.target_label),
            mode=PROMPT_QUERY_KEY,
        )
        _validate_landing_sample(sample, target_label=str(target_axis.target_label))
        annotation_ids = (str(sample.target_platform_id),)
        return AttemptPlatformerResult(
            sample=sample,
            answer_gt=TypedValue(type="string", value=str(sample.answer)),
            annotation_entity_ids=annotation_ids,
            build_annotation=lambda rendered: bbox_for_entity_id(rendered.rendered_scene, annotation_ids[0]),
            witness_type="object_set",
            relations_extra={
                "target_platform_id": str(sample.target_platform_id),
                "target_platform_label": str(sample.target_platform_label),
            },
            execution_extra={
                "target_platform_id": str(sample.target_platform_id),
                "target_platform_label": str(sample.target_platform_label),
            },
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePlatformerPlan(
        attempt_namespace="games.platformer.landing_platform",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint='set "answer" to the selected platform label as a string',
        annotation_hint='set "annotation" to one bounding box [x0, y0, x1, y1] around the labeled platform where the jump lands',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "target_platform_label": str(target_axis.target_label),
            "target_platform_label_support": list(target_axis.target_label_support),
            "target_platform_label_probabilities": dict(target_axis.target_label_probabilities),
        },
        construct_attempt=construct_attempt,
    )

__all__ = ["GamesPlatformerJumpLandingLabelTask"]
