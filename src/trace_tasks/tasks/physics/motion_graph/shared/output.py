"""Trace-payload fragments for physics motion-graph tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .state import RenderedMotionGraph


def build_render_spec(rendered: RenderedMotionGraph, *, scope: str) -> Dict[str, Any]:
    """Build scene-neutral render metadata for one rendered graph."""

    font_record = get_font_family_record(str(rendered.font_family))
    render_map = dict(rendered.render_map)
    return {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "font": {
            "font_family": str(rendered.font_family),
            "font_asset_version": font_asset_version(),
            "font_asset": font_record.to_trace(),
            "scope": str(scope),
        },
        "technical_diagram_style": dict(render_map.get("technical_diagram_style", {})),
        "background_style": dict(render_map.get("background_style", {})),
        "layout_placement": dict(render_map.get("layout_placement", {})),
        "post_image_noise": dict(render_map.get("post_image_noise", {})),
    }


def build_annotation_payload(annotation_value: Mapping[str, Any]) -> Dict[str, Any]:
    """Build common projected annotation trace metadata."""

    payload = {str(key): value for key, value in annotation_value.items()}
    return {
        "witness_symbolic": {
            "type": "bbox_map",
            "keys": sorted(payload.keys()),
        },
        "projected_annotation": {
            "type": "bbox_map",
            "bbox_map": dict(payload),
            "pixel_bbox_map": dict(payload),
        },
    }


def build_segment_annotation_payload(annotation_value: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Build projected annotation trace metadata for one visual segment."""

    segment = [
        [round(float(point[0]), 3), round(float(point[1]), 3)]
        for point in annotation_value
    ]
    return {
        "witness_symbolic": {
            "type": "segment",
        },
        "projected_annotation": {
            "type": "segment",
            "segment": list(segment),
            "pixel_segment": list(segment),
        },
    }


__all__ = [
    "build_annotation_payload",
    "build_render_spec",
    "build_segment_annotation_payload",
]
