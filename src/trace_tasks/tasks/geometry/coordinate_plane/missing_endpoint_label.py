"""Choose the missing endpoint from a visible endpoint and midpoint."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateAlgebraObjective, run_coordinate_algebra_entry

MISSING_ENDPOINT_TASK_ID = "task_geometry__coordinate_plane__missing_endpoint_label"
TASK_ID = "task_geometry__coordinate_plane__missing_endpoint_label"
SCENE_ID = "coordinate_plane"
MISSING_ENDPOINT_QUERY_IDS = ("missing_endpoint_from_midpoint", "missing_startpoint_from_midpoint")
SECTION_POINT_TASK_ID = "task_geometry__coordinate_plane__section_point_label"
SECTION_POINT_QUERY_IDS = ("one_third_from_p_to_q", "two_thirds_from_p_to_q")
TRANSLATED_POINT_TASK_ID = "task_geometry__coordinate_plane__translated_point_label"
TRANSLATED_POINT_QUERY_IDS = ("translate_point", "translate_by_reference_vector")
REFLECTED_POINT_TASK_ID = "task_geometry__coordinate_plane__reflected_point_label"
REFLECTED_POINT_QUERY_IDS = ("reflect_over_vertical_line", "reflect_over_horizontal_line")
ROTATED_POINT_TASK_ID = "task_geometry__coordinate_plane__rotated_point_label"
ROTATED_POINT_QUERY_IDS = ("single",)
TRANSFORMED_POINT_QUERY_IDS = (
    *TRANSLATED_POINT_QUERY_IDS,
    *REFLECTED_POINT_QUERY_IDS,
    *ROTATED_POINT_QUERY_IDS,
)
SUPPORTED_QUERY_IDS = MISSING_ENDPOINT_QUERY_IDS
SCENE_KEY = "coordinate_algebra_midpoint_scene"
SEMANTIC_OPERATION_BY_ID = {
    "missing_endpoint_from_midpoint": "midpoint_missing_q",
    "missing_startpoint_from_midpoint": "midpoint_missing_p",
}


def _prepare_missing_endpoint_objective(
    selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateAlgebraObjective:
    """Bind the selected midpoint-inverse query to an endpoint role."""

    return CoordinateAlgebraObjective(
        semantic_operation=str(SEMANTIC_OPERATION_BY_ID[str(selected_query)]),
        prompt_query_key=str(selected_query),
        scene_key=SCENE_KEY,
    )


@register_task
class GeometryCoordinateMissingEndpointLabelTask:
    """Choose the candidate endpoint that makes the shown point the midpoint."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'formula_evaluation')
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_missing_endpoint_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a missing-endpoint candidate selection scene."""

        return run_coordinate_algebra_entry(self, instance_seed, params=params, max_attempts=max_attempts)
