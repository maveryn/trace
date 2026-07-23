"""Select the coordinate panel whose candidate point set matches a transform."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import CoordinatePanelTaskSpec, fixed_int_axis, run_coordinate_panel_task
from .shared.output import (
    transform_annotation_value,
    transform_panels_trace,
    transform_projected_annotation,
    transform_render_map_extra,
)
from .shared.rendering import render_point_set_transform_panel_task_scene

TASK_ID = "task_geometry__coordinate_panels__point_set_transform_match_label"
QUERY_IDS = (
    "translation_match_label",
    "reflection_x_match_label",
    "reflection_y_match_label",
    "rotation_180_match_label",
)
QUERY_TRANSFORM_KIND = {
    "translation_match_label": "translation",
    "reflection_x_match_label": "reflection_x",
    "reflection_y_match_label": "reflection_y",
    "rotation_180_match_label": "rotation_180",
}
TRANSFORM_DISPLAY_NAME = {
    "translation": "translation",
    "reflection_x": "reflection across the x-axis",
    "reflection_y": "reflection across the y-axis",
    "rotation_180": "180-degree rotation about the origin",
}
POINT_SET_TRANSFORM_TASK_ID = TASK_ID
POINT_SET_TRANSFORM_QUERY_IDS = QUERY_IDS
PANEL_SCENE_ID = "coordinate_panels"
SCENE_ID = PANEL_SCENE_ID


def _resolve_transform_axes(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve point-count axes for transform panel construction."""

    return {"point_count": fixed_int_axis(params=params, defaults=defaults, key="transform_point_count", fallback=3)}


def _prepare_point_set_transform_panel_objective() -> CoordinatePanelTaskSpec:
    """Bind point-set transform semantics to the generic panel lifecycle."""

    return CoordinatePanelTaskSpec(
        public_identifier=TASK_ID,
        query_ids=QUERY_IDS,
        query_kind_by_id=QUERY_TRANSFORM_KIND,
        display_by_kind=TRANSFORM_DISPLAY_NAME,
        kind_field_name="transform_kind",
        display_field_name="transform_display_name",
        prompt_object_description_key="object_description_point_set_transform",
        prompt_annotation_hint_key="annotation_hint_selected_panel_transform_point_set",
        scene_kind="geometry_coordinate_transform_panels",
        witness_type="coordinate_point_set_transform_panel_match",
        answer_type="option_letter",
        annotation_type="point_set",
        render_scene=render_point_set_transform_panel_task_scene,
        panels_trace=transform_panels_trace,
        annotation_value=transform_annotation_value,
        render_map_extra=transform_render_map_extra,
        projected_annotation=transform_projected_annotation,
        extra_axes=_resolve_transform_axes,
    )


@register_task
class GeometryCoordinatePanelPointSetTransformMatchLabelTask:
    """Choose the coordinate panel whose candidate point set matches a transform."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate one point-set transform panel-match instance."""

        spec = _prepare_point_set_transform_panel_objective()
        return run_coordinate_panel_task(spec=spec, instance_seed=instance_seed, params=params, max_attempts=max_attempts)


__all__ = [
    "GeometryCoordinatePanelPointSetTransformMatchLabelTask",
    "PANEL_SCENE_ID",
    "POINT_SET_TRANSFORM_QUERY_IDS",
    "POINT_SET_TRANSFORM_TASK_ID",
    "QUERY_IDS",
    "SCENE_ID",
    "TASK_ID",
]
