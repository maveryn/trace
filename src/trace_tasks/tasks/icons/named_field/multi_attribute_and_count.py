"""Public task for `task_icons__named_field__multi_attribute_and_count`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id

from ._lifecycle import (
    NamedFieldObjectivePlan,
    prepare_boolean_count_objective,
    run_named_field_lifecycle,
)
from .shared.metrics import BOOLEAN_PREDICATE_AND


TASK_ID = "task_icons__named_field__multi_attribute_and_count"
SCENE_ID = "named_field"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
PROMPT_QUERY_KEY = "shape_and_color_count"
INTERNAL_QUERY_ID = PROMPT_QUERY_KEY
PREDICATE_KIND = BOOLEAN_PREDICATE_AND

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
) -> NamedFieldObjectivePlan:
    """Bind this public AND predicate to the named-field lifecycle."""

    return prepare_boolean_count_objective(
        run_namespace=TASK_ID,
        domain="icons",
        selected_query_key=str(selected_query_id),
        query_probabilities=query_probabilities,
        prompt_query_key=PROMPT_QUERY_KEY,
        predicate_kind=PREDICATE_KIND,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_defaults_map=_PROMPT_DEFAULTS,
        instance_seed=int(instance_seed),
        params=params,
    )


@register_task
class IconsNamedFieldMultiAttributeAndCountTask:
    """Count icons satisfying a shape AND visual-attribute predicate."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "icons"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the public query locally and bind the AND predicate."""

        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
        )
        objective = _prepare_boolean_objective(
            int(instance_seed),
            task_params,
            query_probabilities,
            str(selected_query_id),
        )
        return run_named_field_lifecycle(
            scene_id=SCENE_ID,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=objective,
        )


__all__ = [
    "IconsNamedFieldMultiAttributeAndCountTask",
    "INTERNAL_QUERY_ID",
    "PREDICATE_KIND",
    "PROMPT_QUERY_KEY",
    "QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
