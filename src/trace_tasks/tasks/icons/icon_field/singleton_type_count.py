from __future__ import annotations

from typing import Any, Mapping

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    DOMAIN,
    SCENE_ID,
    IconFieldFrequencyObjective,
    IconFieldFrequencyTaskLifecycle,
    build_frequency_binding_payload,
    run_icon_field_frequency_lifecycle,
)
from .shared.annotations import bboxes_for_icon_ids, indices_for_icon_ids
from .shared.defaults import IconFieldDefaults
from .shared.sampling import resolve_singleton_frequency_spec


TASK_ID = "task_icons__icon_field__singleton_type_count"
PROMPT_QUERY_KEY = "singleton_type_count"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)

_DEFAULTS = IconFieldDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _resolve_frequency_spec(instance_seed: int, params: Mapping[str, Any]):
    return resolve_singleton_frequency_spec(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        defaults=_DEFAULTS,
        selection_namespace=f"{TASK_ID}:frequency_spec",
    )


def _bind_answer_annotation(scene_payload, frequency_spec) -> tuple[int, dict[str, Any]]:
    counted_icon_ids = tuple(str(icon_id) for icon_id in scene_payload.singleton_icon_ids)
    annotation_bboxes = bboxes_for_icon_ids(scene_payload, counted_icon_ids)
    answer_value = int(scene_payload.singleton_count)
    if len(annotation_bboxes) != int(answer_value):
        raise ValueError("singleton annotation count did not match answer")
    return build_frequency_binding_payload(
        scene_payload=scene_payload,
        frequency_spec=frequency_spec,
        counting_rule="singleton_icon_type_frequency",
        question_format="count_singleton_type_icons",
        counted_icon_ids=counted_icon_ids,
        annotation_indices=tuple(indices_for_icon_ids(scene_payload, counted_icon_ids)),
        annotation_bboxes=tuple(annotation_bboxes),
        target_count=int(answer_value),
        winner_icon_id=None,
        winner_frequency=None,
    )


@register_task
class IconsIconFieldSingletonTypeCountTask(IconFieldFrequencyTaskLifecycle):
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prompt_query_key = PROMPT_QUERY_KEY
    scene_kind = "icons_singleton_type_counting"

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_icon_field_frequency_lifecycle(
            self,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=IconFieldFrequencyObjective(
                params=params,
                generation_defaults=_GEN_DEFAULTS,
                render_defaults=_RENDER_DEFAULTS,
                prompt_defaults=_PROMPT_DEFAULTS,
                defaults=_DEFAULTS,
                resolve_frequency_spec=_resolve_frequency_spec,
                bind_answer_annotation=_bind_answer_annotation,
            ),
        )
