"""Public task for `task_charts__parallel_coords__axis_delta_extremum_label`."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task
from ._lifecycle import ParallelCoordsTaskPlan, profile_segment_plan, run_parallel_coords_task
from .shared.defaults import DOMAIN, SCENE_NAMESPACE
from .shared.sampling import sample_axis_delta_dataset


INCREASE = "largest_increase_between_axes"
DECREASE = "largest_decrease_between_axes"
ABSOLUTE = "largest_absolute_change_between_axes"
SUPPORTED_QUERY_IDS = (INCREASE, DECREASE, ABSOLUTE)
CHANGE_MODES = {INCREASE: "increase", DECREASE: "decrease", ABSOLUTE: "absolute"}


def _build_plan(params: Mapping[str, Any], instance_seed: int, selected: str) -> ParallelCoordsTaskPlan:
    """Bind one selected change mode and its winning profile segment."""

    mode = CHANGE_MODES[str(selected)]
    dataset = sample_axis_delta_dataset(
        params=params,
        instance_seed=int(instance_seed),
        change_mode=mode,
        namespace=f"{SCENE_NAMESPACE}.axis_delta.{selected}",
    )
    return profile_segment_plan(
        dataset=dataset,
        params=dict(params),
        instance_seed=int(instance_seed),
        prompt_branch_key=str(selected),
        extra_trace_params={"change_mode": str(mode)},
    )


@register_task
class ChartsParallelCoordinatesAxisDeltaExtremumLabelTask:
    """Return the profile label with the greatest selected change."""

    task_id = "task_charts__parallel_coords__axis_delta_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "axis_delta_extremum_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = INCREASE
    task_param_defaults: dict[str, Any] = {}
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Select the change-mode query and annotate the winning segment."""

        return run_parallel_coords_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsParallelCoordinatesAxisDeltaExtremumLabelTask"]
