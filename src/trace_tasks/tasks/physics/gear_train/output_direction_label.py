"""Infer the marked output gear's rotation direction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import run_direction_lifecycle
from .shared.state import SCENE_ID


TASK_ID = "task_physics__gear_train__output_direction_label"
TASK_NAMESPACE = "physics_gear_train_output_direction"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "output_direction_label_query"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsGearTrainOutputDirectionLabelTask:
    """Infer the marked output gear's rotation direction."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the single public branch and run the direction objective."""

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        return run_direction_lifecycle(
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


__all__ = ["PhysicsGearTrainOutputDirectionLabelTask"]
