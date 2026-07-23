"""Count coordinate-plane points in the same quadrant as a marked point."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateRelationObjective, run_coordinate_relation_entry

TASK_ID = "task_geometry__coordinate_plane__same_quadrant_point_count"
SUPPORTED_QUERY_IDS = ("single",)
SCENE_VARIANT = "quadrant_points"
PROMPT_QUERY_KEY = "same_quadrant_count"
SEMANTIC_OPERATION = "same_quadrant"


def _prepare_same_quadrant_objective(
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateRelationObjective:
    """Bind the no-branch public query to same-quadrant counting semantics."""

    return CoordinateRelationObjective(
        scene_variant=SCENE_VARIANT,
        semantic_operation=SEMANTIC_OPERATION,
        prompt_query_key=PROMPT_QUERY_KEY,
    )


@register_task
class GeometryCoordinateSameQuadrantPointCountTask:
    """Count candidate points sharing the marked point's quadrant."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_same_quadrant_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a same-quadrant point-count scene."""

        return run_coordinate_relation_entry(self, instance_seed, params=params, max_attempts=max_attempts)
