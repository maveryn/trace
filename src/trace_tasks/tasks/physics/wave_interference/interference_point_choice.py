"""Choose the candidate point matching a wave-interference condition."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import run_point_choice_lifecycle
from .shared.state import SCENE_ID


TASK_ID = "task_physics__wave_interference__interference_point_choice"
TASK_NAMESPACE = "physics_wave_interference_interference_point_choice"
TASK_PROMPT_KEY = "interference_point_choice_query"
INTERNAL_QUERY_ID = "interference_point_choice"
CONSTRUCTIVE_QUERY_ID = "constructive_interference_point_choice"
DESTRUCTIVE_QUERY_ID = "destructive_interference_point_choice"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (CONSTRUCTIVE_QUERY_ID, DESTRUCTIVE_QUERY_ID)
_QUERY_TO_TARGET_CONDITION = {
    CONSTRUCTIVE_QUERY_ID: "constructive",
    DESTRUCTIVE_QUERY_ID: "destructive",
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "physics",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


@register_task
class PhysicsWavesInterferencePointChoiceTask:
    """Choose a point where two-source wave interference has the requested condition."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'formula_evaluation')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Select the single public branch and run the point-choice objective."""

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=CONSTRUCTIVE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        target_condition = _QUERY_TO_TARGET_CONDITION[str(selected_branch)]
        requested_condition = task_params.get("target_condition")
        if requested_condition is not None and str(requested_condition) != str(target_condition):
            raise ValueError(
                "target_condition must match query_id "
                f"'{selected_branch}' (got: {requested_condition})"
            )
        task_params = dict(task_params)
        task_params["target_condition"] = str(target_condition)
        return run_point_choice_lifecycle(
            domain=self.domain,
            task_id=TASK_ID,
            task_prompt_key=TASK_PROMPT_KEY,
            internal_query_id=INTERNAL_QUERY_ID,
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


__all__ = ["PhysicsWavesInterferencePointChoiceTask"]
