"""Select the largest off-diagonal predicted label in a confusion matrix."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID

from ....core.types import TypedValue
from ...registry import register_task
from ._lifecycle import MatrixBoundObjective, run_configured_matrix_lifecycle
from .shared.annotations import matrix_bbox_set_bundle
from .shared.defaults import (
    attach_visual_selection,
    resolve_visual_selection,
)
from .shared.sampling import construct_confusion_off_diagonal_dataset


TASK_ID = "task_charts__matrix__off_diagonal_confusion_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID


@register_task
class ChartsMatrixOffDiagonalConfusionLabelTask:
    """Return the predicted-column label with the largest off-diagonal count."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'ranking')
    domain = "charts"
    objective_contract = "off_diagonal_confusion_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prompt_query_key = "off_diagonal_confusion_label"
    question_format = "matrix_cell_query"
    default_dataset_enabled = True

    def _construct_dataset(self, instance_seed, params, selected_query_id, branch_probabilities):
        visual = resolve_visual_selection(
            params,
            instance_seed=int(instance_seed),
            supported_scene_variants=("confusion_matrix_counts",),
        )
        dataset = construct_confusion_off_diagonal_dataset(params=params, instance_seed=int(instance_seed))
        dataset["question_params"]["confusion_branch_probabilities"] = dict(branch_probabilities)
        return attach_visual_selection(dataset, visual)

    def _bind_objective(self, dataset, rendered, selected_query_id):
        annotation = matrix_bbox_set_bundle(rendered.rendered_scene, dataset["annotation_cell_ids"])
        return MatrixBoundObjective(
            answer_gt=TypedValue(type="string", value=str(dataset["answer_value"])),
            annotation=annotation,
            relations={
                "query_id": str(selected_query_id),
                "actual_row_label": str(dataset["question_params"]["row_label"]),
                "answer_cell_id": f"r{int(dataset['answer_row_index'])}_c{int(dataset['answer_column_index'])}",
            },
            witness_symbolic={
                "type": "matrix_off_diagonal_confusion_witness",
                "candidate_cell_ids": list(annotation.annotation_cell_ids),
                "support_header_keys": [str(key) for key in dataset["annotation_header_keys"]],
                "answer_value": str(dataset["answer_value"]),
            },
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_matrix_lifecycle(self, instance_seed, params, max_attempts)


__all__ = ["ChartsMatrixOffDiagonalConfusionLabelTask"]
