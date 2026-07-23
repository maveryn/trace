"""Rendering primitives for paired-canvas icon scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image

from ...shared.icon_assets import render_icon_rgba
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import (
    BBox,
    draw_two_panel_panels,
    max_overlap_with_existing,
    panel_geometry_to_trace,
    resolve_two_panel_layout,
)

from .state import PairedCanvasRenderPayload, PairedIconSpec, RenderedPairedIcon


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _center_from_bbox(bbox: Sequence[int | float]) -> Tuple[float, float]:
    return (
        0.5 * (float(bbox[0]) + float(bbox[2])),
        0.5 * (float(bbox[1]) + float(bbox[3])),
    )


def _bbox_for_center(*, content_bbox: BBox, x_frac: float, y_frac: float, sprite_size: Tuple[int, int]) -> BBox:
    x0, y0, x1, y1 = tuple(int(value) for value in content_bbox)
    sprite_w, sprite_h = int(sprite_size[0]), int(sprite_size[1])
    content_w = max(1, int(x1 - x0))
    content_h = max(1, int(y1 - y0))
    cx = int(round(float(x0) + (_clip01(float(x_frac)) * float(content_w))))
    cy = int(round(float(y0) + (_clip01(float(y_frac)) * float(content_h))))
    paste_x0 = int(max(x0, min(x1 - sprite_w, cx - (sprite_w // 2))))
    paste_y0 = int(max(y0, min(y1 - sprite_h, cy - (sprite_h // 2))))
    return (paste_x0, paste_y0, int(paste_x0 + sprite_w), int(paste_y0 + sprite_h))


def _serialize_icon(rendered: RenderedPairedIcon) -> Dict[str, Any]:
    return {
        "entity_kind": "icon_instance",
        "instance_id": str(rendered.instance_id),
        "identity_id": str(rendered.identity_id),
        "icon_id": str(rendered.icon_id),
        "panel": str(rendered.panel),
        "bbox_xyxy": [int(value) for value in rendered.bbox_xyxy],
        "center_xy": [round(float(value), 3) for value in rendered.center_xy],
        "normalized_center_xy": [round(float(value), 4) for value in rendered.normalized_center_xy],
        "nominal_size_px": int(rendered.nominal_size_px),
        "rotation_degrees": int(rendered.rotation_degrees) % 360,
        "tint_rgb": [int(value) for value in rendered.tint_rgb],
        "noise_edits": [dict(edit) for edit in rendered.noise_edits],
        "noise_seed": None if rendered.noise_seed is None else int(rendered.noise_seed),
        "role": str(rendered.role),
        "changed_attributes": [str(value) for value in rendered.changed_attributes],
        "movement_direction": rendered.movement_direction,
    }


def render_paired_canvas(
    *,
    left_icons: Sequence[PairedIconSpec],
    right_icons: Sequence[PairedIconSpec],
    render_params: Mapping[str, Any],
) -> PairedCanvasRenderPayload:
    """Render a two-panel icon scene from explicit normalized positions."""

    layout = resolve_two_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_two_panel_panels(
        image=image,
        layout=layout,
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reference_title="Left",
        scene_title="Right",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )

    rendered_by_panel: Dict[str, List[RenderedPairedIcon]] = {"left": [], "right": []}
    placed_by_panel: Dict[str, List[BBox]] = {"left": [], "right": []}
    content_by_panel = {
        "left": tuple(int(value) for value in layout.reference_content_xyxy),
        "right": tuple(int(value) for value in layout.scene_content_xyxy),
    }
    max_overlap = float(render_params["scene_max_overlap_fraction"])
    for panel_name, specs in (("left", left_icons), ("right", right_icons)):
        content_bbox = content_by_panel[panel_name]
        for spec in specs:
            sprite = render_icon_rgba(
                icon_id=str(spec.icon_id),
                size_px=int(spec.nominal_size_px),
                tint_rgb=tuple(int(value) for value in spec.tint_rgb),
                rotation_degrees=int(spec.rotation_degrees),
                noise_edits=tuple(spec.noise_edits),
                noise_seed=spec.noise_seed,
            )
            bbox = _bbox_for_center(
                content_bbox=content_bbox,
                x_frac=float(spec.x_frac),
                y_frac=float(spec.y_frac),
                sprite_size=sprite.size,
            )
            if float(max_overlap_with_existing(bbox, placed_by_panel[panel_name])) > max_overlap:
                raise ValueError("paired icon canvas placement overlap exceeded configured cap")
            image.alpha_composite(sprite, (int(bbox[0]), int(bbox[1])))
            placed_by_panel[panel_name].append(tuple(int(value) for value in bbox))
            center_xy = _center_from_bbox(bbox)
            rendered_by_panel[panel_name].append(
                RenderedPairedIcon(
                    instance_id=str(spec.instance_id),
                    identity_id=str(spec.identity_id),
                    icon_id=str(spec.icon_id),
                    panel=str(panel_name),
                    bbox_xyxy=tuple(int(value) for value in bbox),
                    center_xy=(float(center_xy[0]), float(center_xy[1])),
                    normalized_center_xy=(float(spec.x_frac), float(spec.y_frac)),
                    nominal_size_px=int(spec.nominal_size_px),
                    rotation_degrees=int(spec.rotation_degrees) % 360,
                    tint_rgb=tuple(int(value) for value in spec.tint_rgb),
                    noise_edits=serialize_icon_noise_edits(spec.noise_edits),
                    noise_seed=None if spec.noise_seed is None else int(spec.noise_seed),
                )
            )

    return PairedCanvasRenderPayload(
        image=image.convert("RGB"),
        panel_geometry=panel_geometry_to_trace(layout),
        left_icons=tuple(_serialize_icon(icon) for icon in rendered_by_panel["left"]),
        right_icons=tuple(_serialize_icon(icon) for icon in rendered_by_panel["right"]),
    )


__all__ = ["render_paired_canvas"]
