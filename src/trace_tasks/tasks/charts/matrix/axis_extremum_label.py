"""Select a row or column label from an extremal matrix cell."""

from __future__ import annotations

from ....core.types import TypedValue
from ...registry import register_task
from ._lifecycle import MatrixBoundObjective, branch_argument_probabilities, run_configured_matrix_lifecycle
from .shared.annotations import matrix_bbox_set_bundle
from .shared.defaults import (
    SUPPORTED_SCENE_VARIANTS,
    attach_visual_selection,
    resolve_visual_selection,
)
from .shared.sampling import construct_axis_ranked_dataset


TASK_ID = "task_charts__matrix__axis_extremum_label"
AXIS_EXTREMUM_ARGS_BY_QUERY_ID = {
    "row_highest_axis_extremum_label": ("row", "highest"),
    "row_lowest_axis_extremum_label": ("row", "lowest"),
    "column_highest_axis_extremum_label": ("column", "highest"),
    "column_lowest_axis_extremum_label": ("column", "lowest"),
}
SUPPORTED_QUERY_IDS = tuple(AXIS_EXTREMUM_ARGS_BY_QUERY_ID)
DEFAULT_QUERY_ID = SUPPORTED_QUERY_IDS[0]


@register_task
class ChartsMatrixAxisExtremumLabelTask:
    """Return a matrix row/column label selected by a second-extremum cell query."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = "charts"
    objective_contract = "axis_extremum_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prompt_query_key = "axis_extremum_label"
    question_format = "matrix_cell_query"
    default_dataset_enabled = True
    supports_unanswerable = True

    def _construct_dataset(self, instance_seed, params, selected_query_id, branch_probabilities):
        query_axis, extremum_direction = AXIS_EXTREMUM_ARGS_BY_QUERY_ID[str(selected_query_id)]
        visual = resolve_visual_selection(
            params,
            instance_seed=int(instance_seed),
            supported_scene_variants=SUPPORTED_SCENE_VARIANTS,
        )
        dataset = construct_axis_ranked_dataset(
            scene_variant=str(visual.scene_variant),
            query_axis=str(query_axis),
            extremum_direction=str(extremum_direction),
            supports_unanswerable=bool(self.supports_unanswerable),
            params=params,
            instance_seed=int(instance_seed),
        )
        branch_axes = branch_argument_probabilities(
            branch_probabilities,
            AXIS_EXTREMUM_ARGS_BY_QUERY_ID,
            argument_names=("query_axis", "extremum_direction"),
        )
        dataset["question_params"]["query_axis_probabilities"] = dict(branch_axes["query_axis"])
        dataset["question_params"]["extremum_direction_probabilities"] = dict(branch_axes["extremum_direction"])
        return attach_visual_selection(dataset, visual)

    def _bind_objective(self, dataset, rendered, selected_query_id):
        annotation = matrix_bbox_set_bundle(rendered.rendered_scene, dataset["annotation_cell_ids"])
        answer_cell_id = (
            ""
            if bool(dataset["is_unanswerable"])
            else f"r{int(dataset['answer_row_index'])}_c{int(dataset['answer_column_index'])}"
        )
        return MatrixBoundObjective(
            answer_gt=TypedValue(type="string", value=str(dataset["answer_value"])),
            annotation=annotation,
            relations={
                "query_id": str(selected_query_id),
                "query_axis": str(dataset["query_axis"]),
                "extremum_direction": str(dataset["extremum_direction"]),
                "answer_cell_id": str(answer_cell_id),
            },
            witness_symbolic={
                "type": "matrix_axis_extremum_witness",
                "candidate_cell_ids": list(annotation.annotation_cell_ids),
                "support_header_keys": [str(key) for key in dataset["annotation_header_keys"]],
                "answer_value": str(dataset["answer_value"]),
                "answerability": "unanswerable" if bool(dataset["is_unanswerable"]) else "answerable",
                **({"absence_proof": dict(dataset["absence_proof"])} if bool(dataset["is_unanswerable"]) else {}),
            },
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_matrix_lifecycle(self, instance_seed, params, max_attempts)


__all__ = ["ChartsMatrixAxisExtremumLabelTask"]
