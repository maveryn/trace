"""Select the coordinate panel whose two segments satisfy a requested relation."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import CoordinatePanelTaskSpec, run_coordinate_panel_task
from .shared.output import (
    segment_annotation_value,
    segment_panels_trace,
    segment_projected_annotation,
    segment_render_map_extra,
)
from .shared.rendering import render_segment_relation_panel_task_scene

TASK_ID = "task_geometry__coordinate_panels__segment_relation_match_label"
QUERY_IDS = (
    "parallel_segments_match_label",
    "perpendicular_segments_match_label",
    "equal_length_segments_match_label",
)
QUERY_RELATION_KIND = {
    "parallel_segments_match_label": "parallel",
    "perpendicular_segments_match_label": "perpendicular",
    "equal_length_segments_match_label": "equal_length",
}
RELATION_DISPLAY_NAME = {
    "parallel": "parallel",
    "perpendicular": "perpendicular",
    "equal_length": "equal length",
}
SEGMENT_RELATION_TASK_ID = TASK_ID
SEGMENT_RELATION_QUERY_IDS = QUERY_IDS
PANEL_SCENE_ID = "coordinate_panels"
SCENE_ID = PANEL_SCENE_ID


def _prepare_segment_relation_panel_objective() -> CoordinatePanelTaskSpec:
    """Bind segment-relation semantics to the generic panel lifecycle."""

    return CoordinatePanelTaskSpec(
        public_identifier=TASK_ID,
        query_ids=QUERY_IDS,
        query_kind_by_id=QUERY_RELATION_KIND,
        display_by_kind=RELATION_DISPLAY_NAME,
        kind_field_name="relation_kind",
        display_field_name="relation_display_name",
        prompt_object_description_key="object_description_segment_relation",
        prompt_annotation_hint_key="annotation_hint_selected_panel_segment_set",
        scene_kind="geometry_coordinate_segment_panels",
        witness_type="coordinate_segment_relation_panel_match",
        answer_type="option_letter",
        annotation_type="segment_set",
        render_scene=render_segment_relation_panel_task_scene,
        panels_trace=segment_panels_trace,
        annotation_value=segment_annotation_value,
        render_map_extra=segment_render_map_extra,
        projected_annotation=segment_projected_annotation,
    )


@register_task
class GeometryCoordinatePanelSegmentRelationMatchLabelTask:
    """Choose the coordinate panel whose two segments satisfy a requested relation."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'matching')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate one segment-relation panel-match instance."""

        spec = _prepare_segment_relation_panel_objective()
        return run_coordinate_panel_task(spec=spec, instance_seed=instance_seed, params=params, max_attempts=max_attempts)


__all__ = [
    "GeometryCoordinatePanelSegmentRelationMatchLabelTask",
    "PANEL_SCENE_ID",
    "QUERY_IDS",
    "SCENE_ID",
    "SEGMENT_RELATION_QUERY_IDS",
    "SEGMENT_RELATION_TASK_ID",
    "TASK_ID",
]
