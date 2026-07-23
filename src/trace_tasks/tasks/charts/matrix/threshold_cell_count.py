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
from .shared.sampling import construct_threshold_dataset


TASK_ID = "task_charts__matrix__threshold_cell_count"
THRESHOLD_ARGS_BY_QUERY_ID = {
    "row_at_least_threshold_cell_count": ("row", "at_least"),
    "row_at_most_threshold_cell_count": ("row", "at_most"),
    "column_at_least_threshold_cell_count": ("column", "at_least"),
    "column_at_most_threshold_cell_count": ("column", "at_most"),
}
SUPPORTED_QUERY_IDS = tuple(THRESHOLD_ARGS_BY_QUERY_ID)
DEFAULT_QUERY_ID = SUPPORTED_QUERY_IDS[0]


@register_task
class ChartsMatrixThresholdCellCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = "charts"
    objective_contract = "threshold_cell_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prompt_query_key = "threshold_cell_count"
    question_format = "matrix_cell_query"
    default_dataset_enabled = True

    def _construct_dataset(self, instance_seed, params, selected_query_id, branch_probabilities):
        query_axis, comparison = THRESHOLD_ARGS_BY_QUERY_ID[str(selected_query_id)]
        visual = resolve_visual_selection(
            params,
            instance_seed=int(instance_seed),
            supported_scene_variants=SUPPORTED_SCENE_VARIANTS,
        )
        dataset = construct_threshold_dataset(
            scene_variant=str(visual.scene_variant),
            query_axis=str(query_axis),
            comparison=str(comparison),
            params=params,
            instance_seed=int(instance_seed),
        )
        branch_axes = branch_argument_probabilities(
            branch_probabilities,
            THRESHOLD_ARGS_BY_QUERY_ID,
            argument_names=("query_axis", "comparison"),
        )
        dataset["question_params"]["query_axis_probabilities"] = dict(branch_axes["query_axis"])
        dataset["question_params"]["comparison_probabilities"] = dict(branch_axes["comparison"])
        return attach_visual_selection(dataset, visual)

    def _bind_objective(self, dataset, rendered, selected_query_id):
        annotation = matrix_bbox_set_bundle(rendered.rendered_scene, dataset["annotation_cell_ids"])
        return MatrixBoundObjective(
            answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
            annotation=annotation,
            relations={
                "query_id": str(selected_query_id),
                "comparison": str(dataset["comparison"]),
            },
            witness_symbolic={
                "type": "matrix_threshold_count_witness",
                "answer_value": int(dataset["answer_value"]),
            },
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_matrix_lifecycle(self, instance_seed, params, max_attempts)
