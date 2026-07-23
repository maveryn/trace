"""Public task for `task_charts__parallel_coords__axis_condition_count`."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task
from ._lifecycle import ParallelCoordsTaskPlan, profile_segment_set_plan, run_parallel_coords_task
from .shared.defaults import DOMAIN, SCENE_NAMESPACE
from .shared.sampling import sample_axis_condition_dataset


ABOVE_BOTH = "above_on_both_axes"
BELOW_BOTH = "below_on_both_axes"
ABOVE_BELOW = "above_on_one_below_on_other"
SUPPORTED_QUERY_IDS = (ABOVE_BOTH, BELOW_BOTH, ABOVE_BELOW)
COMPARATOR_PAIRS: dict[str, tuple[str, str]] = {
    ABOVE_BOTH: ("above", "above"),
    BELOW_BOTH: ("below", "below"),
    ABOVE_BELOW: ("above", "below"),
}


def _build_plan(params: Mapping[str, Any], instance_seed: int, selected: str) -> ParallelCoordsTaskPlan:
    """Bind one two-axis threshold predicate count before rendering."""

    comparators = COMPARATOR_PAIRS[str(selected)]
    dataset = sample_axis_condition_dataset(
        params=params,
        instance_seed=int(instance_seed),
        comparator_pair=comparators,
        namespace=f"{SCENE_NAMESPACE}.axis_condition.{selected}",
    )
    return profile_segment_set_plan(
        dataset=dataset,
        params=dict(params),
        instance_seed=int(instance_seed),
        prompt_branch_key=str(selected),
        extra_trace_params={"axis_predicates": list(comparators)},
    )


@register_task
class ChartsParallelCoordinatesAxisConditionCountTask:
    """Count profiles satisfying a two-axis threshold condition."""

    task_id = "task_charts__parallel_coords__axis_condition_count"
    reasoning_operations = ('filtering', 'counting', 'comparison', 'logical_composition')
    domain = DOMAIN
    objective_contract = "axis_condition_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = ABOVE_BOTH
    task_param_defaults: dict[str, Any] = {}
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Select the comparator query and count matching profile segments."""

        return run_parallel_coords_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsParallelCoordinatesAxisConditionCountTask"]
