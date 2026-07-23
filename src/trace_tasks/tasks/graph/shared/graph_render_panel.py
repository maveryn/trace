"""Panel geometry and chrome for node-link graph rendering."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from PIL import Image, ImageDraw

from ...shared.text_legibility import draw_centered_readable_text, resolve_readable_text_style
from ...shared.text_rendering import load_font
from .graph_render_types import BBox, GraphRenderParams


def _resolve_panel_geometry(
    *,
    canvas_width: int,
    canvas_height: int,
    outer_margin_px: int,
    panel_padding_px: int,
    title_font_size_px: int,
) -> Dict[str, BBox | List[int]]:
    """Resolve one single-panel graph layout with a title band."""

    width = int(canvas_width)
    height = int(canvas_height)
    margin = int(outer_margin_px)
    panel = (int(margin), int(margin), int(width - margin), int(height - margin))
    title_band_height = max(40, int(round(float(title_font_size_px) * 1.8)))
    title_band = (int(panel[0]), int(panel[1]), int(panel[2]), int(panel[1] + title_band_height))
    content = (
        int(panel[0] + panel_padding_px),
        int(title_band[3] + max(8, int(panel_padding_px // 2))),
        int(panel[2] - panel_padding_px),
        int(panel[3] - panel_padding_px),
    )
    return {
        "canvas_size": [int(width), int(height)],
        "scene_panel_xyxy": [int(value) for value in panel],
        "title_band_xyxy": [int(value) for value in title_band],
        "scene_content_xyxy": [int(value) for value in content],
    }

def _draw_panel_chrome(
    image: Image.Image,
    *,
    panel_geometry: Mapping[str, Any],
    render_params: GraphRenderParams,
    scene_title: str,
    fill_background: bool,
    layout_seed: int,
) -> None:
    """Draw one rounded single-panel graph scene with title."""

    draw = ImageDraw.Draw(image)
    if bool(fill_background):
        draw.rectangle((0, 0, image.size[0], image.size[1]), fill=tuple(int(v) for v in render_params.background_color_rgb))
    panel = tuple(int(value) for value in panel_geometry["scene_panel_xyxy"])
    draw.rounded_rectangle(
        panel,
        radius=max(0, int(render_params.panel_corner_radius_px)),
        fill=tuple(int(v) for v in render_params.panel_fill_rgb),
        outline=tuple(int(v) for v in render_params.panel_border_rgb),
        width=2,
    )
    title_center = (0.5 * float(panel[0] + panel[2]), 0.5 * float(panel_geometry["title_band_xyxy"][1] + panel_geometry["title_band_xyxy"][3]))
    title_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.node_link.panel_title_text",
        role="graph_panel_title_text",
        surface_rgbs=(tuple(int(v) for v in render_params.panel_fill_rgb),),
        preferred_rgbs=(tuple(int(v) for v in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    draw_centered_readable_text(
        draw,
        text=str(scene_title),
        center=title_center,
        font=load_font(
            int(render_params.panel_title_font_size_px),
            bold=True,
            font_family=str(render_params.font_family or ""),
        ),
        style=title_style,
        stroke_width=2,
    )


__all__ = [
    "_draw_panel_chrome",
    "_resolve_panel_geometry",
]
