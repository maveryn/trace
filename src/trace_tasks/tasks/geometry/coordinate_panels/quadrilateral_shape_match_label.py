"""Select the coordinate panel whose points form a requested quadrilateral."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import CoordinatePanelTaskSpec, run_coordinate_panel_task
from .shared.construction import is_ambiguous_for_prompt
from .shared.output import (
    quadrilateral_annotation_value,
    quadrilateral_panels_trace,
    quadrilateral_projected_annotation,
    quadrilateral_render_map_extra,
)
from .shared.rendering import render_quadrilateral_panel_task_scene

TASK_ID = "task_geometry__coordinate_panels__quadrilateral_shape_match_label"
QUERY_IDS = (
    "square_shape_match_label",
    "rectangle_shape_match_label",
    "rhombus_shape_match_label",
    "parallelogram_shape_match_label",
)
QUERY_TARGET_KIND = {
    "square_shape_match_label": "square",
    "rectangle_shape_match_label": "rectangle_non_square",
    "rhombus_shape_match_label": "rhombus_non_square",
    "parallelogram_shape_match_label": "parallelogram_only",
}
TARGET_SHAPE_NAME = {
    "square": "square",
    "rectangle_non_square": "rectangle",
    "rhombus_non_square": "rhombus",
    "parallelogram_only": "parallelogram",
}
SHAPE_MATCH_TASK_ID = TASK_ID
SHAPE_MATCH_QUERY_IDS = QUERY_IDS
PANEL_SCENE_ID = "coordinate_panels"
SCENE_ID = PANEL_SCENE_ID


def _prepare_quadrilateral_panel_objective() -> CoordinatePanelTaskSpec:
    """Bind quadrilateral target-kind semantics to the generic panel lifecycle."""

    return CoordinatePanelTaskSpec(
        public_identifier=TASK_ID,
        query_ids=QUERY_IDS,
        query_kind_by_id=QUERY_TARGET_KIND,
        display_by_kind=TARGET_SHAPE_NAME,
        kind_field_name="target_kind",
        display_field_name="target_shape_name",
        prompt_object_description_key="object_description",
        prompt_annotation_hint_key="annotation_hint_selected_panel_point_set",
        scene_kind="geometry_coordinate_panels",
        witness_type="coordinate_quadrilateral_shape_panel_match",
        answer_type="option_letter",
        annotation_type="point_set",
        render_scene=render_quadrilateral_panel_task_scene,
        panels_trace=quadrilateral_panels_trace,
        annotation_value=quadrilateral_annotation_value,
        render_map_extra=quadrilateral_render_map_extra,
        projected_annotation=quadrilateral_projected_annotation,
    )


@register_task
class GeometryCoordinateQuadrilateralShapeMatchLabelTask:
    """Choose the coordinate panel whose four points form the requested quadrilateral."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'matching')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate one quadrilateral panel-match instance."""

        spec = _prepare_quadrilateral_panel_objective()
        return run_coordinate_panel_task(spec=spec, instance_seed=instance_seed, params=params, max_attempts=max_attempts)


__all__ = [
    "GeometryCoordinateQuadrilateralShapeMatchLabelTask",
    "PANEL_SCENE_ID",
    "QUERY_IDS",
    "SCENE_ID",
    "SHAPE_MATCH_QUERY_IDS",
    "SHAPE_MATCH_TASK_ID",
    "TASK_ID",
    "is_ambiguous_for_prompt",
]
