"""Public task for `task_charts__parallel_coords__all_crossings_between_adjacent_axes`."""

from __future__ import annotations

from typing import Any, Mapping

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ._lifecycle import ParallelCoordsTaskPlan, crossing_point_set_plan, run_parallel_coords_task
from .shared.defaults import DOMAIN, SCENE_NAMESPACE
from .shared.sampling import sample_all_crossings_dataset


PROMPT_BRANCH_KEY = "all_crossings_between_adjacent_axes"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _build_plan(params: Mapping[str, Any], instance_seed: int, selected: str) -> ParallelCoordsTaskPlan:
    """Bind the all-profile crossing count before neutral rendering."""

    dataset = sample_all_crossings_dataset(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.all_crossings",
    )
    return crossing_point_set_plan(
        dataset=dataset,
        params=dict(params),
        instance_seed=int(instance_seed),
        prompt_branch_key=PROMPT_BRANCH_KEY,
    )


@register_task
class ChartsParallelCoordinatesAllCrossingsBetweenAdjacentAxesTask:
    """Count all pairwise profile crossings across one adjacent-axis interval."""

    task_id = "task_charts__parallel_coords__all_crossings_between_adjacent_axes"
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = DOMAIN
    objective_contract = "all_crossings_between_adjacent_axes"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_param_defaults: dict[str, Any] = {}
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Select `single`; this task has one public branch with crossing semantics."""

        return run_parallel_coords_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsParallelCoordinatesAllCrossingsBetweenAdjacentAxesTask"]
