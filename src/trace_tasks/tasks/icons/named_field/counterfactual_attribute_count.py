"""Public task for `task_icons__named_field__counterfactual_attribute_count`."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id

from ._lifecycle import (
    prepare_counterfactual_count_objective,
    run_named_field_lifecycle,
)
from .shared.metrics import COUNTERFACTUAL_SHAPE_REPLACEMENT


TASK_ID = "task_icons__named_field__counterfactual_attribute_count"
SCENE_ID = "named_field"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
PROMPT_QUERY_KEY = "target_count_after_shape_replacement"
INTERNAL_QUERY_ID = PROMPT_QUERY_KEY
EDIT_KIND = COUNTERFACTUAL_SHAPE_REPLACEMENT

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "icons",
    SCENE_ID,
    task_id=TASK_ID,
)


def _strip_validated_legacy_query_param(params: Dict[str, Any]) -> Dict[str, Any]:
    """Accept only this task's historical internal counterfactual query selector."""

    resolved = dict(params)
    requested = resolved.pop("counterfactual_query_id", None)
    if requested is not None and str(requested) != PROMPT_QUERY_KEY:
        raise ValueError(f"{TASK_ID} only supports query_id values {(PROMPT_QUERY_KEY,)}")
    return resolved


@register_task
class IconsNamedFieldCounterfactualAttributeCountTask:
    """Count target-shape icons after a hypothetical shape replacement."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = "icons"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        task_input = _strip_validated_legacy_query_param(dict(params))
        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=task_input,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
        )
        objective = prepare_counterfactual_count_objective(
            run_namespace=TASK_ID,
            domain=self.domain,
            selected_query_key=str(selected_query_id),
            query_probabilities=query_probabilities,
            prompt_query_key=PROMPT_QUERY_KEY,
            edit_kind=EDIT_KIND,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults_map=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=task_params,
        )
        return run_named_field_lifecycle(
            scene_id=SCENE_ID,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=objective,
        )


__all__ = [
    "EDIT_KIND",
    "IconsNamedFieldCounterfactualAttributeCountTask",
    "INTERNAL_QUERY_ID",
    "PROMPT_QUERY_KEY",
    "QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
