"""Count coordinate-plane points collinear with the reference line AB."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateRelationObjective, run_coordinate_relation_entry

TASK_ID = "task_geometry__coordinate_plane__collinear_point_count"
SUPPORTED_QUERY_IDS = ("single",)
SCENE_VARIANT = "line_points"
PROMPT_QUERY_KEY = "collinear_count"
SEMANTIC_OPERATION = "collinear"


def _prepare_collinear_objective(
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateRelationObjective:
    """Bind the no-branch public query to collinearity counting semantics."""

    return CoordinateRelationObjective(
        scene_variant=SCENE_VARIANT,
        semantic_operation=SEMANTIC_OPERATION,
        prompt_query_key=PROMPT_QUERY_KEY,
    )


@register_task
class GeometryCoordinateCollinearPointCountTask:
    """Count dot points that lie on the same line as A and B."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_collinear_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a collinear point-count scene."""

        return run_coordinate_relation_entry(self, instance_seed, params=params, max_attempts=max_attempts)
