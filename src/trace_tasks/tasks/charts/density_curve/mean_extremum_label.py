"""Public task for `task_charts__density_curve__mean_extremum_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.density_curve._lifecycle import materialize_density_curve_task
from trace_tasks.tasks.charts.density_curve.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.density_curve.shared.state import DensityCurveObjectiveSpec
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions


SUPPORTED_QUERY_IDS = ("highest_mean_label", "lowest_mean_label")
DEFAULT_QUERY_ID = "highest_mean_label"

QUERY_SPECS = {
    "highest_mean_label": DensityCurveObjectiveSpec(
        metric_name="mean_x",
        direction="max",
        annotation_key="answer_mean_marker",
        visible_role="mean",
        min_gap_param="density_curve_mean_winner_gap_min",
        min_gap_fallback=5.5,
    ),
    "lowest_mean_label": DensityCurveObjectiveSpec(
        metric_name="mean_x",
        direction="min",
        annotation_key="answer_mean_marker",
        visible_role="mean",
        min_gap_param="density_curve_mean_winner_gap_min",
        min_gap_fallback=5.5,
    ),
}


@register_task
class ChartsDistributionDensityCurveMeanExtremumLabelTask:
    """Return the density-curve label with the highest or lowest mean."""

    task_id = "task_charts__density_curve__mean_extremum_label"
    reasoning_operations = ('ranking', 'aggregation')
    domain = "charts"
    objective_contract = "mean_extremum_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=self.task_id,
        )
        objective = QUERY_SPECS[str(selected_query_id)]
        artifacts = materialize_density_curve_task(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_query_id=str(selected_query_id),
            objective=objective,
            max_attempts=int(max_attempts),
            task_id=self.task_id,
        )
        return TaskOutput(
            prompt=str(artifacts.prompt_artifacts.prompt),
            answer_gt=artifacts.answer_gt,
            annotation_gt=artifacts.annotation_gt,
            image=artifacts.rendered.image,
            image_id="img0",
            trace_payload=artifacts.trace_payload,
            task_versions=default_task_versions(),
            query_id=str(selected_query_id),
            scene_id=SCENE_ID,
            prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
        )


__all__ = ["ChartsDistributionDensityCurveMeanExtremumLabelTask"]
