"""Side-torque value public task for lever-balance diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.mechanics import side_torque_value_parts
from .shared.state import SCENE_ID, LeverTaskDefaults


TASK_ID = "task_physics__lever__side_torque_value"
TASK_NAMESPACE = "physics_lever_side_torque_value"
TASK_PROMPT_KEY = "side_torque_value_query"
PROMPT_QUERY_KEY = "side_torque_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)

_DEFAULTS = LeverTaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("physics", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


@register_task
class PhysicsLeverSideTorqueValueTask:
    """Return the total torque on one queried side of a lever."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one side-torque instance."""

        parts = side_torque_value_parts(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            domain=self.domain,
            public_name=TASK_ID,
            namespace=TASK_NAMESPACE,
            prompt_key=TASK_PROMPT_KEY,
            prompt_branch=PROMPT_QUERY_KEY,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            defaults=_DEFAULTS,
        )
        fields = parts.base_fields()
        fields["task_versions"] = default_task_versions()
        fields["query_id"] = str(parts.public_branch)
        fields["scene_id"] = SCENE_ID
        result = TaskOutput(**fields)
        return result


__all__ = ["PhysicsLeverSideTorqueValueTask"]
