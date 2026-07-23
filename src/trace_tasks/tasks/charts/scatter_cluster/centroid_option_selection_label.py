"""Choose the option marker closest to a named cluster centroid."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_scatter_cluster_plan, run_scatter_cluster_task
from .shared.annotations import centroid_option_point_annotation
from .shared.data import build_centroid_option_dataset
from .shared.sampling import (
    option_labels_for_count,
    sample_cluster_inputs,
    target_option_count,
    target_option_label,
)
from .shared.state import DOMAIN


TASK_ID = "task_charts__scatter_cluster__centroid_option_selection_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "centroid_option_selection_label"


def _centroid_option_annotation(dataset, rendered):
    return centroid_option_point_annotation(
        dataset=dataset,
        rendered=rendered,
        selected_option_label=str(dataset.question.answer),
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind the target cluster, option letters, and nearest-centroid answer."""

    inputs = sample_cluster_inputs(params=params, instance_seed=int(instance_seed))
    option_count = target_option_count(params, instance_seed=int(instance_seed))
    option_labels = option_labels_for_count(int(option_count))
    answer_option_label = target_option_label(
        params,
        instance_seed=int(instance_seed),
        cluster_count=int(inputs.cluster_count),
        option_labels=option_labels,
    )
    dataset = build_centroid_option_dataset(
        params=params,
        instance_seed=int(instance_seed),
        labels=inputs.labels,
        target_cluster_label=str(inputs.answer_label),
        points_per_cluster=int(inputs.points_per_cluster),
        answer_option_label=str(answer_option_label),
        option_labels=option_labels,
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        question_params={
            "program_code": "argmin_label(option, distance(point(option), centroid(target_cluster)))",
        },
    )
    return build_scatter_cluster_plan(
        dataset=dataset,
        inputs=inputs,
        prompt_key=PROMPT_QUERY_KEY,
        question_format="scatter_cluster_centroid_option_selection_label",
        witness_type="scatter_cluster_centroid_option_point",
        annotation_builder=_centroid_option_annotation,
    )


@register_task
class ChartsScatterClusterCentroidOptionSelectionLabelTask:
    """Choose the option marker closest to a named cluster centroid."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation', 'spatial_relations', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "centroid_option_selection_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_scatter_cluster_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsScatterClusterCentroidOptionSelectionLabelTask", "SUPPORTED_QUERY_IDS"]
