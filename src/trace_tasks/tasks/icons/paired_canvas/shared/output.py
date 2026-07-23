"""Trace payload fragments for paired-canvas icon tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ...shared.icon_task_rendering import icon_render_style_trace

from .defaults import SCENE_ID


def scene_ir_fragment(
    *,
    scene_kind: str,
    panel_geometry: Mapping[str, Any],
    left_icons: Sequence[Mapping[str, Any]],
    right_icons: Sequence[Mapping[str, Any]],
    relations: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the scene-level IR wrapper around task-owned relations."""

    return {
        "scene_kind": str(scene_kind),
        "scene_id": SCENE_ID,
        "entities": [dict(item) for item in left_icons] + [dict(item) for item in right_icons],
        "relations": dict(relations),
        "frames": {
            "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            "panels": dict(panel_geometry),
        },
    }


def render_spec_fragment(
    *,
    panel_geometry: Mapping[str, Any],
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Build paired-canvas render metadata shared by count objectives."""

    return {
        "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "panel_geometry": dict(panel_geometry),
        "style": icon_render_style_trace(
            render_params=render_params,
            sampled_palette_rgb=tuple(tuple(int(v) for v in color) for color in sampled_palette_rgb),
        ),
    }


def render_map_fragment(
    *,
    left_icons: Sequence[Mapping[str, Any]],
    right_icons: Sequence[Mapping[str, Any]],
    extra_anchors: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build render-map anchors for the two rendered icon panels."""

    anchors: Dict[str, Any] = {
        "left_icons": [dict(item) for item in left_icons],
        "right_icons": [dict(item) for item in right_icons],
    }
    if extra_anchors:
        anchors.update(dict(extra_anchors))
    return {"image_id": "img0", "anchors": anchors}


def build_paired_canvas_trace_payload(
    *,
    scene_kind: str,
    panel_geometry: Mapping[str, Any],
    left_icons: Sequence[Mapping[str, Any]],
    right_icons: Sequence[Mapping[str, Any]],
    relations: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Tuple[int, int, int]],
    render_map_extra: Mapping[str, Any] | None,
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    annotation_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the common paired-canvas trace payload around task-owned fields."""

    return {
        "scene_ir": scene_ir_fragment(
            scene_kind=str(scene_kind),
            panel_geometry=panel_geometry,
            left_icons=left_icons,
            right_icons=right_icons,
            relations=relations,
        ),
        "query_spec": dict(query_spec),
        "render_spec": render_spec_fragment(
            panel_geometry=panel_geometry,
            render_params=render_params,
            sampled_palette_rgb=sampled_palette_rgb,
        ),
        "render_map": render_map_fragment(
            left_icons=left_icons,
            right_icons=right_icons,
            extra_anchors=render_map_extra,
        ),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(annotation_payload["projected_annotation"]),
    }


__all__ = [
    "build_paired_canvas_trace_payload",
    "render_map_fragment",
    "render_spec_fragment",
    "scene_ir_fragment",
]
