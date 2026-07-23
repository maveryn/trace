"""Rendering helpers for heatmap chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.bbox_projection import bbox_union_raw as _bbox_union, round_bbox as _round_bbox
from ....shared.color_distance import coerce_rgb as _rgb
from ....shared.font_assets import sample_font_family
from ....shared.render_variation import apply_layout_jitter_to_margins
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import fit_font_to_box, load_font
from ...shared.visual_defaults import render_style_seed as _render_style_seed
from ...shared.visual_defaults import resolve_chart_render_rgb
from .defaults import (
    BBox,
    _CALENDAR_PALETTE,
    _INTENSITY_PALETTE,
    RENDER_DEFAULTS,
    SCENE_NAMESPACE,
    _SIGNED_PALETTE,
)


@dataclass(frozen=True)
class _HeatmapRenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    title_band_height_px: int
    legend_height_px: int
    row_label_width_px: int
    col_label_height_px: int
    cell_gap_px: int
    cell_border_width_px: int
    title_font_size_px: int
    axis_font_size_px: int
    legend_font_size_px: int
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    title_rgb: Tuple[int, int, int]
    axis_text_rgb: Tuple[int, int, int]
    grid_border_rgb: Tuple[int, int, int]
    cell_border_rgb: Tuple[int, int, int]
    legend_text_rgb: Tuple[int, int, int]
    layout_offset_x_px: int
    layout_offset_y_px: int
    layout_jitter_meta: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class _RenderedHeatmap:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    grid_bbox_px: List[float]
    legend_bbox_px: List[float]
    cell_bbox_map: Dict[str, List[float]]
    row_label_bbox_map: Dict[str, List[float]]
    column_label_bbox_map: Dict[str, List[float]]


def _rgb_param(params: Mapping[str, Any], key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return resolve_chart_render_rgb(params, RENDER_DEFAULTS, str(key), fallback, namespace=SCENE_NAMESPACE)


def _int_param(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), RENDER_DEFAULTS.get(str(key), int(fallback))))


def _resolve_render_params(params: Mapping[str, Any]) -> _HeatmapRenderParams:
    """Resolve canvas, spacing, color, jitter, and font choices for one render."""

    outer = _int_param(params, "outer_margin_px", 42)
    jitter_left, _jitter_right, jitter_top, _jitter_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer),
        right_px=int(outer),
        top_px=int(outer),
        bottom_px=int(outer),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=_render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    return _HeatmapRenderParams(
        canvas_width=_int_param(params, "canvas_width", 1260),
        canvas_height=_int_param(params, "canvas_height", 820),
        outer_margin_px=int(outer),
        panel_padding_px=_int_param(params, "panel_padding_px", 28),
        title_band_height_px=_int_param(params, "title_band_height_px", 68),
        legend_height_px=_int_param(params, "legend_height_px", 72),
        row_label_width_px=_int_param(params, "row_label_width_px", 150),
        col_label_height_px=_int_param(params, "col_label_height_px", 46),
        cell_gap_px=_int_param(params, "cell_gap_px", 4),
        cell_border_width_px=_int_param(params, "cell_border_width_px", 2),
        title_font_size_px=_int_param(params, "title_font_size_px", 30),
        axis_font_size_px=_int_param(params, "axis_font_size_px", 20),
        legend_font_size_px=_int_param(params, "legend_font_size_px", 18),
        panel_fill_rgb=_rgb_param(params, "panel_fill_rgb", (252, 253, 251)),
        panel_border_rgb=_rgb_param(params, "panel_border_rgb", (64, 74, 86)),
        title_rgb=_rgb_param(params, "title_rgb", (30, 38, 48)),
        axis_text_rgb=_rgb_param(params, "axis_text_rgb", (36, 44, 54)),
        grid_border_rgb=_rgb_param(params, "grid_border_rgb", (80, 92, 104)),
        cell_border_rgb=_rgb_param(params, "cell_border_rgb", (255, 255, 255)),
        legend_text_rgb=_rgb_param(params, "legend_text_rgb", (36, 44, 54)),
        layout_offset_x_px=int(jitter_left) - int(outer),
        layout_offset_y_px=int(jitter_top) - int(outer),
        layout_jitter_meta=dict(layout_jitter_meta),
        font_family=sample_font_family(
            role="readout",
            instance_seed=_render_style_seed(params),
            namespace=f"{SCENE_NAMESPACE}.chart_font",
            params=params,
            exclude_tags=("display",),
            explicit_key="chart_font_family",
            weights_key="chart_font_family_weights",
        ),
    )

def _value_palette(scene_variant: str) -> Tuple[Tuple[int, int, int], ...]:
    if str(scene_variant) == "signed_change_heatmap":
        return _SIGNED_PALETTE
    if str(scene_variant) == "calendar_heatmap":
        return _CALENDAR_PALETTE
    return _INTENSITY_PALETTE


def _interpolate_rgb(
    low: Tuple[int, int, int],
    high: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    clamped = max(0.0, min(1.0, float(t)))
    return tuple(
        int(round((1.0 - clamped) * float(a) + clamped * float(b)))
        for a, b in zip(low, high)
    )


def _continuous_colorbar_rgb(value: int) -> Tuple[int, int, int]:
    ratio = max(0.0, min(1.0, float(value) / 100.0))
    if ratio <= 0.5:
        return _interpolate_rgb((45, 95, 160), (245, 224, 110), ratio / 0.5)
    return _interpolate_rgb((245, 224, 110), (188, 48, 54), (ratio - 0.5) / 0.5)


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: Sequence[float],
    *,
    font_size: int,
    fill: Tuple[int, int, int],
    font_family: str | None = None,
    anchor: str = "center",
) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=float(x1 - x0),
        max_height=float(y1 - y0),
        min_size_px=8,
        max_size_px=max(8, int(font_size)),
        fill_ratio=0.9,
        font_family=font_family,
    )
    if str(anchor) == "left":
        draw_text_traced(draw,(x0, (y0 + y1) / 2.0), str(text), font=font, fill=fill, anchor="lm", role="readout", required=False)
    else:
        draw_text_traced(draw,((x0 + x1) / 2.0, (y0 + y1) / 2.0), str(text), font=font, fill=fill, anchor="mm", role="readout", required=False)


def _draw_signed_marker(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    value: int,
    bin_count: int,
) -> None:
    midpoint = int(bin_count) // 2
    if int(value) == int(midpoint):
        return
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    size = min(float(x1 - x0), float(y1 - y0)) * 0.22
    fill = (38, 48, 58)
    if int(value) > int(midpoint):
        points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    else:
        points = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
    draw.polygon(points, fill=fill)


def _render_legend(
    draw: ImageDraw.ImageDraw,
    *,
    scene_variant: str,
    legend_bbox: Sequence[float],
    render_params: _HeatmapRenderParams,
    palette: Sequence[Tuple[int, int, int]],
    ticks: Sequence[int] = (),
) -> None:
    """Draw the discrete legend or continuous colorbar that defines color semantics."""

    x0, y0, x1, y1 = [float(value) for value in legend_bbox]
    font = load_font(max(8, int(render_params.legend_font_size_px)), font_family=render_params.font_family)
    if str(scene_variant) == "continuous_colorbar_heatmap":
        label = "Colorbar: value"
        draw_text_traced(draw,(x0, y0 + 8), label, font=font, fill=render_params.legend_text_rgb, anchor="la", role="readout", required=False)
        bar_x0 = x0 + 132
        bar_x1 = x1 - 54
        bar_y0 = y0 + 32
        bar_y1 = y0 + 52
        segment_count = max(40, int(bar_x1 - bar_x0))
        for index in range(int(segment_count)):
            left = bar_x0 + (float(index) / float(segment_count)) * (bar_x1 - bar_x0)
            right = bar_x0 + (float(index + 1) / float(segment_count)) * (bar_x1 - bar_x0)
            value = int(round((float(index) / float(max(1, int(segment_count) - 1))) * 100.0))
            draw.rectangle([left, bar_y0, right + 1.0, bar_y1], fill=_continuous_colorbar_rgb(value))
        draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], outline=render_params.cell_border_rgb, width=1)
        tick_values = tuple(int(value) for value in ticks) if ticks else tuple(range(0, 101, 10))
        tick_font = load_font(max(8, int(render_params.legend_font_size_px) - 3), font_family=render_params.font_family)
        for tick in tick_values:
            ratio = max(0.0, min(1.0, float(tick) / 100.0))
            tx = bar_x0 + ratio * (bar_x1 - bar_x0)
            draw.line([tx, bar_y1, tx, bar_y1 + 6], fill=render_params.legend_text_rgb, width=1)
            draw_text_traced(draw,(tx, bar_y1 + 9), str(int(tick)), font=tick_font, fill=render_params.legend_text_rgb, anchor="mt", role="readout", required=False)
        return

    label = "Legend"
    if str(scene_variant) == "signed_change_heatmap":
        label = "Legend: decrease to increase"
    elif str(scene_variant) == "calendar_heatmap":
        label = "Legend: low to high activity"
    else:
        label = "Legend: low to high intensity"
    draw_text_traced(draw,(x0, y0 + 8), label, font=font, fill=render_params.legend_text_rgb, anchor="la", role="readout", required=False)
    swatch_y0 = y0 + 34
    swatch_h = 20
    swatch_w = max(34.0, min(60.0, (x1 - x0 - 190.0) / float(max(1, len(palette)))))
    start_x = x0 + 82
    for index, color in enumerate(palette):
        sx0 = start_x + (float(index) * swatch_w)
        draw.rectangle([sx0, swatch_y0, sx0 + swatch_w, swatch_y0 + swatch_h], fill=tuple(color), outline=render_params.cell_border_rgb, width=1)
    draw_text_traced(draw,(start_x - 8, swatch_y0 + swatch_h / 2.0), "low", font=font, fill=render_params.legend_text_rgb, anchor="rm", role="readout", required=False)
    draw_text_traced(draw,(start_x + (len(palette) * swatch_w) + 8, swatch_y0 + swatch_h / 2.0), "high", font=font, fill=render_params.legend_text_rgb, anchor="lm", role="readout", required=False)


def _render_heatmap(
    image: Image.Image,
    *,
    scene_title: str,
    scene_variant: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    render_params: _HeatmapRenderParams,
    colorbar_ticks: Sequence[int] = (),
) -> _RenderedHeatmap:
    """Draw the full heatmap panel and return projected cell and label geometry."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    offset_x = float(render_params.layout_offset_x_px)
    offset_y = float(render_params.layout_offset_y_px)
    panel_bbox = (
        float(render_params.outer_margin_px) + offset_x,
        float(render_params.outer_margin_px) + offset_y,
        float(width - render_params.outer_margin_px) + offset_x,
        float(height - render_params.outer_margin_px) + offset_y,
    )
    draw.rounded_rectangle(panel_bbox, radius=6, fill=render_params.panel_fill_rgb, outline=render_params.panel_border_rgb, width=2)
    title_bbox = (0.0, 0.0, 0.0, 0.0)

    grid_left = panel_bbox[0] + render_params.panel_padding_px + render_params.row_label_width_px
    grid_top = panel_bbox[1] + render_params.title_band_height_px + render_params.col_label_height_px
    grid_right = panel_bbox[2] - render_params.panel_padding_px
    grid_bottom = panel_bbox[3] - render_params.panel_padding_px - render_params.legend_height_px
    grid_bbox = (grid_left, grid_top, grid_right, grid_bottom)
    row_count = len(row_labels)
    column_count = len(column_labels)
    gap = float(render_params.cell_gap_px)
    cell_w = (grid_right - grid_left - (gap * float(max(0, column_count - 1)))) / float(max(1, column_count))
    cell_h = (grid_bottom - grid_top - (gap * float(max(0, row_count - 1)))) / float(max(1, row_count))

    draw.rectangle(grid_bbox, outline=render_params.grid_border_rgb, width=2)
    palette = _value_palette(str(scene_variant))
    cell_bbox_map: Dict[str, List[float]] = {}
    row_label_bbox_map: Dict[str, List[float]] = {}
    column_label_bbox_map: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "heatmap_panel",
            "entity_type": "chart_panel",
            "bbox_xyxy": _round_bbox(panel_bbox),
            "attrs": {"scene_variant": str(scene_variant)},
        },
    ]

    cell_by_pos = {(int(cell["row_index"]), int(cell["column_index"])): dict(cell) for cell in cells}
    for row_index, row_label in enumerate(row_labels):
        label_bbox = (
            panel_bbox[0] + render_params.panel_padding_px,
            grid_top + (float(row_index) * (cell_h + gap)),
            grid_left - 14,
            grid_top + (float(row_index) * (cell_h + gap)) + cell_h,
        )
        row_label_bbox_map[str(row_label)] = _round_bbox(label_bbox)
        _draw_text_in_box(
            draw,
            str(row_label),
            label_bbox,
            font_size=render_params.axis_font_size_px,
            fill=render_params.axis_text_rgb,
            font_family=render_params.font_family,
            anchor="left",
        )
        entities.append(
            {
                "entity_id": f"row_label_{row_index}",
                "entity_type": "row_label",
                "bbox_xyxy": _round_bbox(label_bbox),
                "attrs": {"row_index": int(row_index), "row_label": str(row_label)},
            }
        )

    for column_index, column_label in enumerate(column_labels):
        label_bbox = (
            grid_left + (float(column_index) * (cell_w + gap)),
            panel_bbox[1] + render_params.title_band_height_px,
            grid_left + (float(column_index) * (cell_w + gap)) + cell_w,
            grid_top - 8,
        )
        column_label_bbox_map[str(column_label)] = _round_bbox(label_bbox)
        _draw_text_in_box(
            draw,
            str(column_label),
            label_bbox,
            font_size=render_params.axis_font_size_px,
            fill=render_params.axis_text_rgb,
            font_family=render_params.font_family,
        )
        entities.append(
            {
                "entity_id": f"column_label_{column_index}",
                "entity_type": "column_label",
                "bbox_xyxy": _round_bbox(label_bbox),
                "attrs": {"column_index": int(column_index), "column_label": str(column_label)},
            }
        )

    for row_index in range(int(row_count)):
        for column_index in range(int(column_count)):
            cell = cell_by_pos[(int(row_index), int(column_index))]
            x0 = grid_left + (float(column_index) * (cell_w + gap))
            y0 = grid_top + (float(row_index) * (cell_h + gap))
            bbox = (x0, y0, x0 + cell_w, y0 + cell_h)
            if str(scene_variant) == "continuous_colorbar_heatmap":
                color = _continuous_colorbar_rgb(int(cell["heat_level"]))
            else:
                color = palette[int(cell["heat_level"]) % len(palette)]
            draw.rectangle(bbox, fill=color, outline=render_params.cell_border_rgb, width=int(render_params.cell_border_width_px))
            if str(scene_variant) == "signed_change_heatmap":
                _draw_signed_marker(draw, bbox, value=int(cell["heat_level"]), bin_count=len(palette))
            cell_bbox = _round_bbox(bbox)
            cell_bbox_map[str(cell["cell_id"])] = list(cell_bbox)
            entities.append(
                {
                    "entity_id": str(cell["cell_id"]),
                    "entity_type": "heatmap_cell",
                    "bbox_xyxy": list(cell_bbox),
                    "attrs": {
                        "row_index": int(row_index),
                        "column_index": int(column_index),
                        "row_label": str(cell["row_label"]),
                        "column_label": str(cell["column_label"]),
                        "heat_level": int(cell["heat_level"]),
                        "numeric_value": int(cell.get("numeric_value", cell["heat_level"])),
                    },
                }
            )

    legend_bbox = (
        panel_bbox[0] + render_params.panel_padding_px,
        panel_bbox[3] - render_params.panel_padding_px - render_params.legend_height_px + 12,
        panel_bbox[2] - render_params.panel_padding_px,
        panel_bbox[3] - render_params.panel_padding_px,
    )
    _render_legend(
        draw,
        scene_variant=str(scene_variant),
        legend_bbox=legend_bbox,
        render_params=render_params,
        palette=palette,
        ticks=colorbar_ticks if str(scene_variant) == "continuous_colorbar_heatmap" else (),
    )
    entities.append(
        {
            "entity_id": "heatmap",
            "entity_type": "heatmap",
            "bbox_xyxy": _round_bbox(grid_bbox),
            "attrs": {"row_count": int(row_count), "column_count": int(column_count)},
        }
    )
    entities.append(
        {
            "entity_id": "heatmap_legend",
            "entity_type": "heatmap_legend",
            "bbox_xyxy": _round_bbox(legend_bbox),
            "attrs": {"scene_variant": str(scene_variant)},
        }
    )
    return _RenderedHeatmap(
        image=image,
        entities=tuple(dict(item) for item in entities),
        panel_bbox_px=_round_bbox(panel_bbox),
        title_bbox_px=_round_bbox(title_bbox),
        grid_bbox_px=_round_bbox(grid_bbox),
        legend_bbox_px=_round_bbox(legend_bbox),
        cell_bbox_map=dict(cell_bbox_map),
        row_label_bbox_map=dict(row_label_bbox_map),
        column_label_bbox_map=dict(column_label_bbox_map),
    )
