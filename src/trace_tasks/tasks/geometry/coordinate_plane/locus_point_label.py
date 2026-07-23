"""Choose a candidate point inside a coordinate-plane locus region."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateLocusObjective, run_coordinate_locus_point_entry

POINT_TASK_ID = "task_geometry__coordinate_plane__locus_point_label"
PANEL_TASK_ID = "task_geometry__coordinate_plane__locus_panel_match_label"
PANEL_QUERY_IDS = (
    "circle_inequality_panel_match",
    "vertical_strip_panel_match",
    "horizontal_halfplane_panel_match",
    "two_inequality_panel_match",
)
TASK_ID = "task_geometry__coordinate_plane__locus_point_label"
SCENE_ID = "coordinate_plane"
POINT_QUERY_IDS = (
    "circle_region_point",
    "annulus_region_point",
    "vertical_strip_region_point",
    "half_plane_intersection_region_point",
)
SUPPORTED_QUERY_IDS = POINT_QUERY_IDS
SEMANTIC_OPERATION_BY_ID = {
    "circle_region_point": "circle_region",
    "annulus_region_point": "annulus_region",
    "vertical_strip_region_point": "vertical_strip_region",
    "half_plane_intersection_region_point": "half_plane_intersection_region",
}


def _prepare_locus_point_objective(
    selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateLocusObjective:
    """Bind the selected locus-family query to candidate-point selection."""

    return CoordinateLocusObjective(
        semantic_operation=str(SEMANTIC_OPERATION_BY_ID[str(selected_query)]),
        prompt_query_key=str(selected_query),
    )


@register_task
class GeometryCoordinateLocusPointLabelTask:
    """Choose the candidate point that lies in the shaded coordinate locus."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_locus_point_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a shaded-locus candidate point scene."""

        return run_coordinate_locus_point_entry(self, instance_seed, params=params, max_attempts=max_attempts)
