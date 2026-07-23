"""Count coordinate-plane segments with a requested relation to AB."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateRelationObjective, run_coordinate_relation_entry
from .shared.relations import _segments_intersect

TASK_ID = "task_geometry__coordinate_plane__segment_relation_count"
SUPPORTED_QUERY_IDS = ("parallel_count", "perpendicular_count")
SCENE_VARIANT = "segment_set"
PROMPT_QUERY_BY_ID = {
    "parallel_count": "parallel_count",
    "perpendicular_count": "perpendicular_count",
}
SEMANTIC_OPERATION_BY_ID = {
    "parallel_count": "parallel",
    "perpendicular_count": "perpendicular",
}


def _prepare_segment_relation_objective(
    selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateRelationObjective:
    """Bind the selected public relation query to scene-local segment semantics."""

    return CoordinateRelationObjective(
        scene_variant=SCENE_VARIANT,
        semantic_operation=str(SEMANTIC_OPERATION_BY_ID[str(selected_query)]),
        prompt_query_key=str(PROMPT_QUERY_BY_ID[str(selected_query)]),
    )


@register_task
class GeometryCoordinateSegmentRelationCountTask:
    """Count non-AB segments parallel or perpendicular to AB."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_segment_relation_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a segment relation count on the coordinate plane."""

        return run_coordinate_relation_entry(self, instance_seed, params=params, max_attempts=max_attempts)


GeometryCoordinateRelationTask = GeometryCoordinateSegmentRelationCountTask
