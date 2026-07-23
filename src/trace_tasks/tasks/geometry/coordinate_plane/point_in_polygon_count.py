"""Count lattice points strictly inside a coordinate-plane polygon."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateRelationObjective, run_coordinate_relation_entry

TASK_ID = "task_geometry__coordinate_plane__point_in_polygon_count"
SUPPORTED_QUERY_IDS = ("single",)
SCENE_VARIANT = "polygon_lattice"
PROMPT_QUERY_KEY = "point_in_shape_count"
SEMANTIC_OPERATION = "polygon_interior"


def _prepare_point_in_polygon_objective(
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateRelationObjective:
    """Bind the no-branch public query to interior lattice-point counting."""

    return CoordinateRelationObjective(
        scene_variant=SCENE_VARIANT,
        semantic_operation=SEMANTIC_OPERATION,
        prompt_query_key=PROMPT_QUERY_KEY,
    )


@register_task
class GeometryCoordinatePointInPolygonCountTask:
    """Count integer lattice points strictly inside the polygon."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_point_in_polygon_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate an interior lattice-point count scene."""

        return run_coordinate_relation_entry(self, instance_seed, params=params, max_attempts=max_attempts)
