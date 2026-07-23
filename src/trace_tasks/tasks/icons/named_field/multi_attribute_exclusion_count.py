"""Public task for `task_icons__named_field__multi_attribute_exclusion_count`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import (
    NamedFieldObjectivePlan,
    prepare_boolean_count_objective,
    run_named_field_lifecycle,
)
from .shared.metrics import BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE, BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE


TASK_ID = "task_icons__named_field__multi_attribute_exclusion_count"
SCENE_ID = "named_field"
SHAPE_WITHOUT_ATTRIBUTE_QUERY_ID = "shape_and_not_color_count"
ATTRIBUTE_WITHOUT_SHAPE_QUERY_ID = "color_and_not_shape_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    SHAPE_WITHOUT_ATTRIBUTE_QUERY_ID,
    ATTRIBUTE_WITHOUT_SHAPE_QUERY_ID,
)
PREDICATE_KIND_BY_QUERY_ID = {
    SHAPE_WITHOUT_ATTRIBUTE_QUERY_ID: BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE,
    ATTRIBUTE_WITHOUT_SHAPE_QUERY_ID: BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE,
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "icons",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_boolean_objective(
    instance_seed: int,
    params: Dict[str, Any],
    query_probabilities: Dict[str, float],
    selected_query_id: str,
    predicate_kind: str,
) -> NamedFieldObjectivePlan:
    """Bind this public exclusion predicate branch to the named-field lifecycle."""

    return prepare_boolean_count_objective(
        run_namespace=TASK_ID,
        domain="icons",
        selected_query_key=str(selected_query_id),
        query_probabilities=query_probabilities,
        prompt_query_key=str(selected_query_id),
        predicate_kind=str(predicate_kind),
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_defaults_map=_PROMPT_DEFAULTS,
        instance_seed=int(instance_seed),
        params=params,
    )


@register_task
class IconsNamedFieldMultiAttributeExclusionCountTask:
    """Count icons satisfying one named predicate while excluding the other."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "icons"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SHAPE_WITHOUT_ATTRIBUTE_QUERY_ID,
            task_id=TASK_ID,
        )
        prompt_query_key = str(selected_query_id)
        predicate_kind = str(PREDICATE_KIND_BY_QUERY_ID[prompt_query_key])
        objective = _prepare_boolean_objective(
            int(instance_seed),
            task_params,
            query_probabilities,
            prompt_query_key,
            predicate_kind,
        )
        return run_named_field_lifecycle(
            scene_id=SCENE_ID,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=objective,
        )


__all__ = [
    "ATTRIBUTE_WITHOUT_SHAPE_QUERY_ID",
    "IconsNamedFieldMultiAttributeExclusionCountTask",
    "PREDICATE_KIND_BY_QUERY_ID",
    "SHAPE_WITHOUT_ATTRIBUTE_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
