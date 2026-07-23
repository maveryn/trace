"""Infographic task for summing one icon-filtered section total."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__infographic__section_icon_total_value"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
SOURCE_BRANCH_KEYS = ("section_icon_total_value",)
PROMPT_QUERY_KEY = SOURCE_BRANCH_KEYS[0]


def _build_objective() -> _lifecycle.InfographicObjectiveBinding:
    """Bind icon-filtered section total summation to this public task."""

    return _lifecycle.InfographicObjectiveBinding(
        source_branch_keys=SOURCE_BRANCH_KEYS,
        prompt_branch_fallback=PROMPT_QUERY_KEY,
    )


def _runtime_params(params: Dict[str, Any], objective: _lifecycle.InfographicObjectiveBinding) -> Dict[str, Any]:
    """Force the runtime to the task-owned source branch set."""

    runtime_params = dict(params)
    runtime_params["_supported_query_ids"] = tuple(objective.source_branch_keys)
    runtime_params["query_id"] = PROMPT_QUERY_KEY
    runtime_params.pop("query_variant", None)
    runtime_params.pop("query_id_weights", None)
    runtime_params.pop("query_variant_weights", None)
    return runtime_params


@register_task
class PagesInfographicSectionIconTotalValueTask:
    """Compute a section total filtered by icon type."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation',)
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one icon-filtered section total instance."""

        selected_branch, _branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        if selected_branch != SINGLE_QUERY_ID:
            raise ValueError(f"unsupported public query_id for {TASK_ID}: {selected_branch}")
        objective = _build_objective()
        runtime_params = _runtime_params(task_params, objective)
        runtime = _lifecycle.InfographicMetricCardRuntime()
        output = runtime.generate(
            int(instance_seed),
            params=runtime_params,
            max_attempts=int(max_attempts),
        )
        return output


__all__ = [
    "PROMPT_QUERY_KEY",
    "SOURCE_BRANCH_KEYS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesInfographicSectionIconTotalValueTask",
]
