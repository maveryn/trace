"""Choose the rotated image point on a coordinate plane."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateAlgebraObjective, run_coordinate_algebra_entry

ROTATED_POINT_TASK_ID = "task_geometry__coordinate_plane__rotated_point_label"
TASK_ID = "task_geometry__coordinate_plane__rotated_point_label"
SCENE_ID = "coordinate_plane"
ROTATED_POINT_QUERY_IDS = ("single",)
SUPPORTED_QUERY_IDS = ROTATED_POINT_QUERY_IDS
SCENE_KEY = "coordinate_algebra_transform_scene"
PROMPT_QUERY_KEY = "rotate_90_about_marked_center"
SEMANTIC_OPERATION = "rotate_quarter_turn"


def _prepare_rotated_point_objective(
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateAlgebraObjective:
    """Bind the no-branch public query to quarter-turn rotation semantics."""

    return CoordinateAlgebraObjective(
        semantic_operation=SEMANTIC_OPERATION,
        prompt_query_key=PROMPT_QUERY_KEY,
        scene_key=SCENE_KEY,
    )


@register_task
class GeometryCoordinateRotatedPointLabelTask:
    """Choose the candidate image point after a 90-degree rotation."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_rotated_point_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a rotated-point candidate selection scene."""

        return run_coordinate_algebra_entry(self, instance_seed, params=params, max_attempts=max_attempts)
