"""Choose the reflected image point on a coordinate plane."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateAlgebraObjective, run_coordinate_algebra_entry

REFLECTED_POINT_TASK_ID = "task_geometry__coordinate_plane__reflected_point_label"
TASK_ID = "task_geometry__coordinate_plane__reflected_point_label"
SCENE_ID = "coordinate_plane"
REFLECTED_POINT_QUERY_IDS = ("reflect_over_vertical_line", "reflect_over_horizontal_line")
SUPPORTED_QUERY_IDS = REFLECTED_POINT_QUERY_IDS
SCENE_KEY = "coordinate_algebra_transform_scene"
SEMANTIC_OPERATION_BY_ID = {
    "reflect_over_vertical_line": "reflect_vertical",
    "reflect_over_horizontal_line": "reflect_horizontal",
}


def _prepare_reflected_point_objective(
    selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateAlgebraObjective:
    """Bind the selected reflection-axis query to coordinate reflection semantics."""

    return CoordinateAlgebraObjective(
        semantic_operation=str(SEMANTIC_OPERATION_BY_ID[str(selected_query)]),
        prompt_query_key=str(selected_query),
        scene_key=SCENE_KEY,
    )


@register_task
class GeometryCoordinateReflectedPointLabelTask:
    """Choose the candidate image point after reflection across a marked line."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_reflected_point_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a reflected-point candidate selection scene."""

        return run_coordinate_algebra_entry(self, instance_seed, params=params, max_attempts=max_attempts)
