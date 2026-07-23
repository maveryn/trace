"""Trace payload helpers for matchstick puzzle tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .rendering import matchstick_style_trace
from .state import RenderParams, RenderedScene, SCENE_ID


def json_ready(value: Any) -> Any:
    """Convert nested tuples and mappings into JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [json_ready(inner) for inner in value]
    return value


def build_render_spec(
    *,
    rendered_scene: RenderedScene,
    render_params: RenderParams,
    scene_variant: str,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    font_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build common render metadata for matchstick tasks."""

    return {
        "scene_id": SCENE_ID,
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(scene_variant),
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
        "stick_width_px": int(render_params.stick_width_px),
        "text_style": {"font": dict(font_meta)},
        "scene_style": {
            "matchstick": matchstick_style_trace(str(scene_variant)),
            "font": dict(font_meta),
        },
    }


def build_render_map(
    *,
    rendered_scene: RenderedScene,
    annotation_source: str,
) -> Dict[str, Any]:
    """Build item-bbox projections for review and annotation contracts."""

    return {
        "image_id": "img0",
        "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
        "item_bboxes_px": {
            str(key): [round(float(v), 3) for v in value]
            for key, value in rendered_scene.item_bbox_map.items()
        },
        "item_segments_px": {
            str(key): [
                [round(float(point[0]), 3), round(float(point[1]), 3)]
                for point in value
            ]
            for key, value in rendered_scene.item_segment_map.items()
        },
        "annotation_source": str(annotation_source),
    }


def build_matchstick_trace_payload(
    *,
    scene_ir: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    render_map: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    answer_gt: Mapping[str, Any],
    annotation_gt: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
) -> Dict[str, Any]:
    """Assemble one JSON-stable matchstick trace payload."""

    return {
        "scene_ir": json_ready(dict(scene_ir)),
        "query_spec": json_ready(dict(query_spec)),
        "render_spec": json_ready(dict(render_spec)),
        "render_map": json_ready(dict(render_map)),
        "execution_trace": json_ready(dict(execution_trace)),
        "witness_symbolic": json_ready(dict(witness_symbolic)),
        "projected_annotation": json_ready(dict(projected_annotation)),
        "answer_gt": json_ready(dict(answer_gt)),
        "annotation_gt": json_ready(dict(annotation_gt)),
        "prompt_spec": {
            "defaults": json_ready(dict(prompt_defaults)),
            "active": json_ready(dict(prompt_artifacts.prompt_variant)),
        },
    }


__all__ = [
    "build_matchstick_trace_payload",
    "build_render_map",
    "build_render_spec",
    "json_ready",
]
