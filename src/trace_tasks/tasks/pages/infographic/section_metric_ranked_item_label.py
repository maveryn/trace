"""Infographic task for selecting a section-scoped ranked metric card."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.pages.shared.infographic_metric_common import SECTION_METRIC_RANKED_ITEM_VARIANTS
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__infographic__section_metric_ranked_item_label"
SUPPORTED_QUERY_IDS = SECTION_METRIC_RANKED_ITEM_VARIANTS
SOURCE_BRANCH_KEYS = SUPPORTED_QUERY_IDS
DEFAULT_QUERY_ID = "nth_highest_metric_in_section_label"


def _build_objective() -> _lifecycle.InfographicObjectiveBinding:
    """Bind section-scoped metric-card rank source branches to this public task."""

    return _lifecycle.InfographicObjectiveBinding(
        source_branch_keys=SOURCE_BRANCH_KEYS,
        prompt_branch_fallback=DEFAULT_QUERY_ID,
    )


def _runtime_params(
    params: Dict[str, Any],
    objective: _lifecycle.InfographicObjectiveBinding,
    *,
    selected_branch: str,
) -> Dict[str, Any]:
    """Expose only section-scoped rank branches to the runtime sampler."""

    runtime_params = dict(params)
    runtime_params["_supported_query_ids"] = tuple(objective.source_branch_keys)
    runtime_params["query_id"] = str(selected_branch)
    runtime_params.pop("query_variant", None)
    runtime_params.pop("query_id_weights", None)
    runtime_params.pop("query_variant_weights", None)
    return runtime_params


@register_task
class PagesInfographicSectionMetricRankedItemLabelTask:
    """Identify a metric card by rank of its printed value inside one section."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one section-scoped ranked metric-card label instance."""

        selected_branch, _branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=DEFAULT_QUERY_ID,
            public_task=TASK_ID,
        )
        if selected_branch not in set(SUPPORTED_QUERY_IDS):
            raise ValueError(f"unsupported public query_id for {TASK_ID}: {selected_branch}")
        objective = _build_objective()
        runtime_params = _runtime_params(task_params, objective, selected_branch=str(selected_branch))
        runtime = _lifecycle.InfographicMetricCardRuntime()
        output = runtime.generate(
            int(instance_seed),
            params=runtime_params,
            max_attempts=int(max_attempts),
        )
        return output


__all__ = [
    "DEFAULT_QUERY_ID",
    "SOURCE_BRANCH_KEYS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesInfographicSectionMetricRankedItemLabelTask",
]
