"""Choose the translated image point on a coordinate plane."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateAlgebraObjective, run_coordinate_algebra_entry

TRANSLATED_POINT_TASK_ID = "task_geometry__coordinate_plane__translated_point_label"
TASK_ID = "task_geometry__coordinate_plane__translated_point_label"
SCENE_ID = "coordinate_plane"
TRANSLATED_POINT_QUERY_IDS = ("translate_point", "translate_by_reference_vector")
SUPPORTED_QUERY_IDS = TRANSLATED_POINT_QUERY_IDS
SCENE_KEY = "coordinate_algebra_transform_scene"
SEMANTIC_OPERATION_BY_ID = {
    "translate_point": "translate_direct",
    "translate_by_reference_vector": "translate_reference",
}


def _prepare_translated_point_objective(
    selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateAlgebraObjective:
    """Bind the selected translation query to direct or reference-vector semantics."""

    return CoordinateAlgebraObjective(
        semantic_operation=str(SEMANTIC_OPERATION_BY_ID[str(selected_query)]),
        prompt_query_key=str(selected_query),
        scene_key=SCENE_KEY,
    )


@register_task
class GeometryCoordinateTranslatedPointLabelTask:
    """Choose the candidate image point after the requested translation."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_translated_point_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a translated-point candidate selection scene."""

        return run_coordinate_algebra_entry(self, instance_seed, params=params, max_attempts=max_attempts)
