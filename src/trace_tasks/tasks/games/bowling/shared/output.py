"""Objective-neutral trace assembly for Bowling games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .state import BowlingSample
from .sampling import ResolvedBowlingSceneAxes
from .rendering import RenderedBowlingTaskContext


def _pin_trace(sample: BowlingSample) -> list[Dict[str, Any]]:
    return [
        {
            "pin_id": str(pin.pin_id),
            "label": str(pin.label),
            "rack_index": int(pin.rack_index),
            "row": int(pin.row),
            "col": int(pin.col),
            "standing": bool(pin.standing),
            "x_norm": None if pin.x_norm is None else float(pin.x_norm),
            "y_norm": None if pin.y_norm is None else float(pin.y_norm),
        }
        for pin in sample.pins
    ]


def _path_trace(sample: BowlingSample) -> list[Dict[str, Any]]:
    return [
        {
            "path_id": str(path.path_id),
            "label": str(path.label),
            "aim_x_norm": float(path.aim_x_norm),
        }
        for path in sample.path_options
    ]


def build_bowling_common_trace_params(
    *,
    axes: ResolvedBowlingSceneAxes,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared Bowling query params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_bowling_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    axes: ResolvedBowlingSceneAxes,
    sample: BowlingSample,
    rendered_context: RenderedBowlingTaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    answer_value: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble Bowling trace sections after task-specific answer binding."""

    rendered_scene = rendered_context.rendered_scene
    standing_pin_count = sum(1 for pin in sample.pins if bool(pin.standing))
    if str(annotation_artifacts.annotation_type) in {"segment", "segment_set"}:
        witness_symbolic = {
            "type": str(annotation_artifacts.annotation_type),
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        }
    else:
        witness_symbolic = {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        }

    return {
        "scene_ir": {
            "scene_kind": f"games_bowling_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "standing_pin_count": int(standing_pin_count),
                "path_option_count": int(len(sample.path_options)),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
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
            "visible_pin_count": int(len(sample.pins)),
            "pins": _pin_trace(sample),
            "path_options": _path_trace(sample),
            "ball_x_norm": float(sample.ball_x_norm),
            "target_pin_id": sample.target_pin_id,
            "target_pin_label": sample.target_pin_label,
            "target_path_id": sample.target_path_id,
            "target_path_label": sample.target_path_label,
            "path_visible_fraction": sample.path_visible_fraction,
            "path_clearance_px": sample.path_clearance_px,
            "remaining_pin_ids": [str(entity_id) for entity_id in sample.remaining_pin_ids],
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(sample.construction_mode),
            "answer": str(answer_value),
            **dict(execution_extra or {}),
        },
        "witness_symbolic": witness_symbolic,
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = ["build_bowling_common_trace_params", "build_bowling_trace_payload"]
