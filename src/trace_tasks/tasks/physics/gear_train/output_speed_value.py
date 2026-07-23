"""Compute the marked output gear's rotational speed."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import run_speed_lifecycle
from .shared.state import SCENE_ID


TASK_ID = "task_physics__gear_train__output_speed_value"
TASK_NAMESPACE = "physics_gear_train_output_speed"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "output_speed_value_query"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsGearTrainOutputSpeedValueTask:
    """Compute the marked output gear's rotational speed."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('topology', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the single public branch and run the speed objective."""

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        return run_speed_lifecycle(
            domain=self.domain,
            task_prompt_key=TASK_PROMPT_KEY,
            lifecycle_namespace=TASK_NAMESPACE,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            task_params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            max_attempts=int(max_attempts),
        )


__all__ = ["PhysicsGearTrainOutputSpeedValueTask"]
