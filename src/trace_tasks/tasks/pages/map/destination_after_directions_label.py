"""Pages map task for route-following destination lookup."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.state import MapSceneCase


TASK_ID = "task_pages__map__destination_after_directions_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "destination_after_directions"


def _build_route_plan(
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: MapSceneCase,
) -> _lifecycle.MapRoutePlan:
    """Bind route-following operands to the destination lookup objective."""

    return _lifecycle.destination_route_plan(
        instance_seed=int(instance_seed),
        params=params,
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        case=case,
        route_namespace=f"{TASK_ID}.route",
        prompt_query_key=PROMPT_QUERY_KEY,
    )


@register_task
class PagesMapDestinationAfterDirectionsLabelTask:
    """Identify the destination reached after following visible map directions."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'topology', 'state_update')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one route-following destination instance."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        return _lifecycle.render_bound_map(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            plan_factory=_build_route_plan,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesMapDestinationAfterDirectionsLabelTask",
]
