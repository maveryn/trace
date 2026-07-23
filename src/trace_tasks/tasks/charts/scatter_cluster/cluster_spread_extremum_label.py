"""Return the cluster with extremal within-cluster spread."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_scatter_cluster_plan, run_scatter_cluster_task
from .shared.annotations import cluster_bbox_annotation
from .shared.data import build_spread_dataset
from .shared.sampling import sample_cluster_inputs
from .shared.state import DOMAIN


TASK_ID = "task_charts__scatter_cluster__cluster_spread_extremum_label"
LARGEST_HORIZONTAL_QUERY_ID = "largest_horizontal_spread_label"
SMALLEST_HORIZONTAL_QUERY_ID = "smallest_horizontal_spread_label"
LARGEST_VERTICAL_QUERY_ID = "largest_vertical_spread_label"
SMALLEST_VERTICAL_QUERY_ID = "smallest_vertical_spread_label"
LARGEST_OVERALL_QUERY_ID = "largest_overall_spread_label"
SMALLEST_OVERALL_QUERY_ID = "smallest_overall_spread_label"
SUPPORTED_QUERY_IDS = (
    LARGEST_HORIZONTAL_QUERY_ID,
    SMALLEST_HORIZONTAL_QUERY_ID,
    LARGEST_VERTICAL_QUERY_ID,
    SMALLEST_VERTICAL_QUERY_ID,
    LARGEST_OVERALL_QUERY_ID,
    SMALLEST_OVERALL_QUERY_ID,
)
DEFAULT_QUERY_ID = LARGEST_HORIZONTAL_QUERY_ID
_SPREAD_ARGS_BY_BRANCH = {
    LARGEST_HORIZONTAL_QUERY_ID: ("horizontal", "largest"),
    SMALLEST_HORIZONTAL_QUERY_ID: ("horizontal", "smallest"),
    LARGEST_VERTICAL_QUERY_ID: ("vertical", "largest"),
    SMALLEST_VERTICAL_QUERY_ID: ("vertical", "smallest"),
    LARGEST_OVERALL_QUERY_ID: ("overall", "largest"),
    SMALLEST_OVERALL_QUERY_ID: ("overall", "smallest"),
}


def _answer_cluster_annotation(dataset, rendered):
    return cluster_bbox_annotation(
        dataset=dataset,
        rendered=rendered,
        cluster_label=str(dataset.question.answer),
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind the spread axis/direction and answer-cluster hull."""

    spread_axis, spread_extremum = _SPREAD_ARGS_BY_BRANCH[str(selected)]
    inputs = sample_cluster_inputs(params=params, instance_seed=int(instance_seed))
    dataset = build_spread_dataset(
        params=params,
        instance_seed=int(instance_seed),
        labels=inputs.labels,
        answer_label=str(inputs.answer_label),
        points_per_cluster=int(inputs.points_per_cluster),
        spread_axis=str(spread_axis),
        spread_extremum=str(spread_extremum),
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        question_params={
            "program_code": "argextreme_label(cluster, spread(cluster, axis), direction)",
        },
    )
    return build_scatter_cluster_plan(
        dataset=dataset,
        inputs=inputs,
        prompt_key=str(selected),
        question_format="scatter_cluster_spread_extremum_label",
        witness_type="scatter_cluster_answer_cluster_bbox",
        annotation_builder=_answer_cluster_annotation,
    )


@register_task
class ChartsScatterClusterSpreadExtremumLabelTask:
    """Return the cluster with extremal within-cluster spread."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "cluster_spread_extremum_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_scatter_cluster_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsScatterClusterSpreadExtremumLabelTask", "SUPPORTED_QUERY_IDS"]
