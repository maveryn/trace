"""Return the cluster label with the strongest requested trend direction."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_scatter_cluster_plan, run_scatter_cluster_task
from .shared.annotations import cluster_bbox_annotation
from .shared.data import build_trend_dataset
from .shared.sampling import sample_cluster_inputs
from .shared.state import DOMAIN


TASK_ID = "task_charts__scatter_cluster__cluster_trend_direction_label"
UPWARD_TREND_QUERY_ID = "upward_trend_label"
DOWNWARD_TREND_QUERY_ID = "downward_trend_label"
SUPPORTED_QUERY_IDS = (UPWARD_TREND_QUERY_ID, DOWNWARD_TREND_QUERY_ID)
DEFAULT_QUERY_ID = UPWARD_TREND_QUERY_ID
_TREND_DIRECTION_BY_BRANCH = {
    UPWARD_TREND_QUERY_ID: "upward",
    DOWNWARD_TREND_QUERY_ID: "downward",
}


def _answer_cluster_annotation(dataset, rendered):
    return cluster_bbox_annotation(
        dataset=dataset,
        rendered=rendered,
        cluster_label=str(dataset.question.answer),
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind the requested trend direction and selected answer-cluster hull."""

    trend_direction = _TREND_DIRECTION_BY_BRANCH[str(selected)]
    inputs = sample_cluster_inputs(params=params, instance_seed=int(instance_seed))
    dataset = build_trend_dataset(
        params=params,
        instance_seed=int(instance_seed),
        labels=inputs.labels,
        answer_label=str(inputs.answer_label),
        points_per_cluster=int(inputs.points_per_cluster),
        trend_direction=str(trend_direction),
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        question_params={
            "program_code": "argmax_label(cluster, signed_trend_strength(cluster, direction))",
        },
    )
    return build_scatter_cluster_plan(
        dataset=dataset,
        inputs=inputs,
        prompt_key=str(selected),
        question_format="scatter_cluster_trend_direction_label",
        witness_type="scatter_cluster_answer_cluster_bbox",
        annotation_builder=_answer_cluster_annotation,
    )


@register_task
class ChartsScatterClusterTrendDirectionLabelTask:
    """Return the cluster label with the strongest requested trend direction."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "cluster_trend_direction_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_scatter_cluster_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsScatterClusterTrendDirectionLabelTask", "SUPPORTED_QUERY_IDS"]
