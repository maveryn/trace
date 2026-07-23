"""Objective-neutral trace assembly for pinball-table tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedPinballTaskContext
from .sampling import PinballVisualAxes
from .state import PinballSceneState


def pinball_object_display_text(score_value: int | None, label: str, show_label: bool) -> str | None:
    """Return the visible text drawn inside one pinball object."""

    if score_value is not None:
        return str(int(score_value))
    if bool(show_label):
        return str(label)
    return None


def build_pinball_object_trace(scene: PinballSceneState) -> list[Dict[str, Any]]:
    """Build trace records for all labeled/scored pinball objects."""

    return [
        {
            "object_id": str(obj.object_id),
            "label": str(obj.label),
            "kind": str(obj.kind),
            "x_norm": float(obj.x_norm),
            "y_norm": float(obj.y_norm),
            "radius_norm": float(obj.radius_norm),
            "width_norm": float(obj.width_norm),
            "height_norm": float(obj.height_norm),
            "score_value": None if obj.score_value is None else int(obj.score_value),
            "show_label": bool(obj.show_label),
            "display_text": pinball_object_display_text(obj.score_value, str(obj.label), bool(obj.show_label)),
        }
        for obj in scene.objects
    ]


def common_pinball_trace_params(
    axes: PinballVisualAxes,
    scene: PinballSceneState,
    *,
    query_id_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared pinball query trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "object_count": int(axes.object_count),
        "visible_object_count": len(scene.objects),
        "query_id_probabilities": dict(query_id_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "object_count_probabilities": dict(axes.object_count_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_pinball_common_trace_payload(
    *,
    axes: PinballVisualAxes,
    scene: PinballSceneState,
    rendered_context: RenderedPinballTaskContext,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    query_spec: Mapping[str, Any],
    witness_type: str,
    relations_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble objective-neutral pinball trace sections after task binding."""

    rendered_scene = rendered_context.rendered_scene
    object_trace = build_pinball_object_trace(scene)
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"games_pinball_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "object_count": len(scene.objects),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
                **dict(relations_extra or {}),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "ball_xy_norm": [float(scene.ball_x_norm), float(scene.ball_y_norm)],
            "cue_angle_rad": float(scene.cue_angle_rad),
            "objects": object_trace,
            "cue_visible_fraction": float(scene.cue_visible_fraction),
            "hidden_path_norm": [[float(x), float(y)] for x, y in scene.hidden_path_norm],
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(scene.construction_mode),
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": str(witness_type),
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }
    return trace_payload


__all__ = [
    "build_pinball_common_trace_payload",
    "build_pinball_object_trace",
    "common_pinball_trace_params",
    "pinball_object_display_text",
]
