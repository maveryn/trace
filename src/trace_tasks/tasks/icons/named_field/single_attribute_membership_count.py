"""Count prompt-named procedural icon shapes."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    NamedFieldObjectivePlan,
    prepare_shape_count_objective,
    run_named_field_lifecycle,
)


TASK_ID = "task_icons__named_field__single_attribute_membership_count"
SCENE_ID = "named_field"
QUERY_ID = "named_shape_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "icons",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_shape_count_objective(instance_seed: int, params: Dict[str, Any]) -> NamedFieldObjectivePlan:
    """Bind this public direct shape-count task to the named-field lifecycle."""

    return prepare_shape_count_objective(
        run_namespace=TASK_ID,
        domain="icons",
        public_query_id=QUERY_ID,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_defaults_map=_PROMPT_DEFAULTS,
        instance_seed=int(instance_seed),
        params=params,
    )


@register_task
class IconsCountingNamedShapeCountTask:
    """Count procedural named icon shapes in a single field."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "icons"
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one direct named-shape counting task."""

        task_params = dict(params)
        objective = _prepare_shape_count_objective(int(instance_seed), task_params)
        return run_named_field_lifecycle(
            scene_id=SCENE_ID,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=objective,
        )


__all__ = ["IconsCountingNamedShapeCountTask"]
