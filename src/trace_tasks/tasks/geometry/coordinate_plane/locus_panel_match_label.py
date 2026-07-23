"""Choose the panel whose shaded coordinate region matches a condition."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task

from ._lifecycle import CoordinateLocusObjective, run_coordinate_locus_panel_entry

PANEL_TASK_ID = "task_geometry__coordinate_plane__locus_panel_match_label"
TASK_ID = "task_geometry__coordinate_plane__locus_panel_match_label"
SCENE_ID = "coordinate_plane"
PANEL_QUERY_IDS = (
    "circle_inequality_panel_match",
    "vertical_strip_panel_match",
    "horizontal_halfplane_panel_match",
    "two_inequality_panel_match",
)
SUPPORTED_QUERY_IDS = PANEL_QUERY_IDS
SEMANTIC_OPERATION_BY_ID = {
    "circle_inequality_panel_match": "circle_panel",
    "vertical_strip_panel_match": "vertical_strip_panel",
    "horizontal_halfplane_panel_match": "horizontal_halfplane_panel",
    "two_inequality_panel_match": "two_inequality_panel",
}


def _prepare_locus_panel_objective(
    selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateLocusObjective:
    """Bind the selected locus-family query to panel matching semantics."""

    return CoordinateLocusObjective(
        semantic_operation=str(SEMANTIC_OPERATION_BY_ID[str(selected_query)]),
        prompt_query_key=str(selected_query),
    )


@register_task
class GeometryCoordinateLocusPanelMatchLabelTask:
    """Choose the panel matching the shown coordinate locus condition."""

    task_id = TASK_ID
    reasoning_operations = ('comparison', 'logical_composition', 'spatial_relations', 'matching')
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_locus_panel_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a coordinate-locus panel matching scene."""

        return run_coordinate_locus_panel_entry(self, instance_seed, params=params, max_attempts=max_attempts)
