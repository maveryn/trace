"""Return the cluster label at a requested shaded-footprint area rank."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_scatter_cluster_plan, run_scatter_cluster_task
from .shared.annotations import cluster_bbox_annotation
from .shared.data import build_area_rank_dataset
from .shared.sampling import sample_cluster_inputs
from .shared.state import DOMAIN


TASK_ID = "task_charts__scatter_cluster__cluster_area_rank_label"
LARGEST_AREA_QUERY_ID = "largest_cluster_area_label"
SMALLEST_AREA_QUERY_ID = "smallest_cluster_area_label"
SUPPORTED_QUERY_IDS = (
    LARGEST_AREA_QUERY_ID,
    SMALLEST_AREA_QUERY_ID,
)
DEFAULT_QUERY_ID = LARGEST_AREA_QUERY_ID
_AREA_RANK_BY_BRANCH = {
    LARGEST_AREA_QUERY_ID: ("largest", "largest"),
    SMALLEST_AREA_QUERY_ID: ("smallest", "smallest"),
}


def _answer_cluster_annotation(dataset, rendered):
    return cluster_bbox_annotation(
        dataset=dataset,
        rendered=rendered,
        cluster_label=str(dataset.question.answer),
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind the requested shaded-footprint area rank and selected cluster."""

    area_rank, area_rank_phrase = _AREA_RANK_BY_BRANCH[str(selected)]
    inputs = sample_cluster_inputs(params=params, instance_seed=int(instance_seed))
    dataset = build_area_rank_dataset(
        params=params,
        instance_seed=int(instance_seed),
        labels=inputs.labels,
        points_per_cluster=int(inputs.points_per_cluster),
        area_rank=str(area_rank),
        area_rank_phrase=str(area_rank_phrase),
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        question_params={
            "program_code": "ranked_label(cluster, footprint_area(cluster), rank)",
        },
    )
    return build_scatter_cluster_plan(
        dataset=dataset,
        inputs=inputs,
        prompt_key=str(selected),
        question_format="scatter_cluster_area_rank_label",
        witness_type="scatter_cluster_area_footprint_bbox",
        annotation_builder=_answer_cluster_annotation,
    )


@register_task
class ChartsScatterClusterAreaRankLabelTask:
    """Return the cluster label at a requested shaded-footprint area rank."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "cluster_area_rank_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_scatter_cluster_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsScatterClusterAreaRankLabelTask", "SUPPORTED_QUERY_IDS"]
