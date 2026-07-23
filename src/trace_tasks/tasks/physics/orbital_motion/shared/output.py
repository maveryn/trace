"""Trace-payload fragments for orbital-motion tasks."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .state import RenderedOrbitScene


def build_render_spec(rendered: RenderedOrbitScene, *, scope: str) -> Dict[str, Any]:
    """Build render metadata for one orbit diagram."""

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
        "post_image_noise": dict(render_map.get("post_image_noise", {})),
    }


__all__ = ["build_render_spec"]
