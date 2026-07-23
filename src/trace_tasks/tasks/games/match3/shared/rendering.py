"""Render match-3 boards, gems, and labeled swap arrows."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from PIL import ImageDraw

from trace_tasks.tasks.shared.color_format import rgb_to_hex
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill
from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import (
    draw_panel_grid_cell,
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced

from .defaults import DEFAULTS, GEM_RGB, MATCH3_STYLE_RGB
from .rules import cell_entity_id
from .state import Coord, Match3Sample, RenderedMatch3Scene
from .styles import blend_rgb, gem_polygon


def _int_default(params: Mapping[str, Any], render_defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer render parameter from params, config, or fallback."""

    if str(key) in params:
        return int(params[str(key)])
    return int(group_default(render_defaults, str(key), int(fallback)))


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    text: str,
    *,
    font,
    fill: Tuple[int, int, int],
    stroke_width: int = 1,
) -> None:
    """Draw a short label centered in a fixed box using traced game text."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    x0, y0, x1, y1 = bbox
    draw_text_traced(
        draw,
        (
            float(x0 + ((x1 - x0) - text_w) / 2.0 - float(text_bbox[0])),
            float(y0 + ((y1 - y0) - text_h) / 2.0 - float(text_bbox[1])),
        ),
        str(text),
        font=font,
        fill=tuple(fill),
        stroke_width=int(stroke_width),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(fill)),
        role="readout",
        required=False,
    )


def _draw_gem(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    *,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    shape: str,
    outline_width: int,
    shadow: bool,
) -> None:
    """Draw one gem token while keeping all supported style variants legible."""

    x0, y0, x1, y1 = bbox
    cx = float((x0 + x1) / 2.0)
    cy = float((y0 + y1) / 2.0)
    radius = float(min(x1 - x0, y1 - y0) / 2.0)
    width = max(1, int(outline_width))
    highlight = tuple(min(255, int(value + 42)) for value in fill_rgb)
    shadow_rgb = tuple(max(0, int(value * 0.70)) for value in fill_rgb)

    if str(shape) == "circle":
        draw.ellipse((x0, y0, x1, y1), fill=tuple(fill_rgb), outline=tuple(outline_rgb), width=width)
        inset = radius * 0.32
        draw.ellipse((cx - inset, cy - radius * 0.55, cx + inset * 0.5, cy - radius * 0.08), fill=highlight)
        return
    if str(shape) == "rounded_square":
        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(6, int(radius * 0.28)), fill=tuple(fill_rgb), outline=tuple(outline_rgb), width=width)
        draw.rounded_rectangle(
            (x0 + radius * 0.25, y0 + radius * 0.22, x1 - radius * 0.25, y0 + radius * 0.68),
            radius=max(4, int(radius * 0.15)),
            fill=highlight,
        )
        if bool(shadow):
            draw.line((x0 + radius * 0.2, y1 - radius * 0.24, x1 - radius * 0.2, y1 - radius * 0.24), fill=shadow_rgb, width=2)
        return
    if str(shape) == "diamond":
        polygon = ((cx, y0), (x1, cy), (cx, y1), (x0, cy))
        draw.polygon(polygon, fill=tuple(fill_rgb), outline=tuple(outline_rgb))
        inner = ((cx, y0 + radius * 0.35), (x1 - radius * 0.35, cy), (cx, cy + radius * 0.20), (x0 + radius * 0.35, cy))
        draw.polygon(inner, fill=highlight)
        if bool(shadow):
            draw.line((x1, cy, cx, y1, x0, cy), fill=shadow_rgb, width=width)
        return
    if str(shape) == "orb":
        draw.ellipse((x0, y0, x1, y1), fill=tuple(fill_rgb), outline=tuple(outline_rgb), width=width)
        ring = (x0 + radius * 0.18, y0 + radius * 0.18, x1 - radius * 0.18, y1 - radius * 0.18)
        draw.ellipse(ring, outline=highlight, width=max(2, width - 1))
        shine = (cx - radius * 0.42, cy - radius * 0.50, cx - radius * 0.05, cy - radius * 0.15)
        draw.ellipse(shine, fill=(255, 255, 255))
        return

    polygon = gem_polygon(cx, cy, radius)
    draw.polygon(polygon, fill=tuple(fill_rgb), outline=tuple(outline_rgb))
    inner = gem_polygon(cx, cy - radius * 0.08, radius * 0.50)
    draw.polygon(inner, fill=highlight)
    if bool(shadow):
        draw.line([polygon[2], polygon[3], polygon[4]], fill=shadow_rgb, width=max(2, width))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    line_rgb: Tuple[int, int, int],
    label: str | None,
    label_font,
    text_rgb: Tuple[int, int, int],
    label_fill_rgb: Tuple[int, int, int],
    label_outline_rgb: Tuple[int, int, int],
    width: int,
) -> List[float]:
    """Draw one labeled adjacent-swap arrow and return its visual bbox."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx = float(ex - sx)
    dy = float(ey - sy)
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    start_trim = 0.18 * length
    end_trim = 0.18 * length
    x0 = sx + ux * start_trim
    y0 = sy + uy * start_trim
    x1 = ex - ux * end_trim
    y1 = ey - uy * end_trim
    stroke = tuple(int(value) for value in resolve_text_stroke_fill(line_rgb))
    draw.line((x0, y0, x1, y1), fill=stroke, width=int(width + 4))
    draw.line((x0, y0, x1, y1), fill=tuple(line_rgb), width=int(width))
    head_len = max(12.0, float(width) * 2.4)
    head_w = max(10.0, float(width) * 2.0)
    px, py = -uy, ux
    head = (
        (x1, y1),
        (x1 - ux * head_len + px * head_w / 2.0, y1 - uy * head_len + py * head_w / 2.0),
        (x1 - ux * head_len - px * head_w / 2.0, y1 - uy * head_len - py * head_w / 2.0),
    )
    draw.line((head[1][0], head[1][1], x1, y1), fill=stroke, width=int(width + 4))
    draw.line((head[2][0], head[2][1], x1, y1), fill=stroke, width=int(width + 4))
    draw.line((head[1][0], head[1][1], x1, y1), fill=tuple(line_rgb), width=int(width))
    draw.line((head[2][0], head[2][1], x1, y1), fill=tuple(line_rgb), width=int(width))
    bbox = [
        min(x0, x1) - float(width + head_w),
        min(y0, y1) - float(width + head_w),
        max(x0, x1) + float(width + head_w),
        max(y0, y1) + float(width + head_w),
    ]
    if label is not None:
        mid_x = float((sx + ex) / 2.0)
        mid_y = float((sy + ey) / 2.0)
        radius = max(14.0, float(width) * 2.5)
        label_bbox = (mid_x - radius, mid_y - radius, mid_x + radius, mid_y + radius)
        draw.ellipse(label_bbox, fill=tuple(label_fill_rgb), outline=tuple(label_outline_rgb), width=2)
        _draw_centered_text(draw, label_bbox, str(label), font=label_font, fill=text_rgb, stroke_width=0)
        bbox = [
            min(bbox[0], label_bbox[0]),
            min(bbox[1], label_bbox[1]),
            max(bbox[2], label_bbox[2]),
            max(bbox[3], label_bbox[3]),
        ]
    return [round(float(value), 3) for value in bbox]


def render_match3_scene(
    *,
    sample: Match3Sample,
    instance_seed: int,
    params: Mapping[str, Any],
    style_variant: str,
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderedMatch3Scene:
    """Render the board and projection maps without knowing the public objective."""

    canvas_width = _int_default(params, render_defaults, "canvas_width", DEFAULTS.canvas_width)
    canvas_height = _int_default(params, render_defaults, "canvas_height", DEFAULTS.canvas_height)
    style, style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.match3_panel_style",
        treatment_weights=group_default(render_defaults, "panel_scene_treatment_weights", None),
        palette_weights=group_default(render_defaults, "panel_scene_palette_weights", None),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.match3.label_font",
        params=params,
    )
    match3_style = MATCH3_STYLE_RGB.get(str(style_variant), MATCH3_STYLE_RGB["faceted_jewels"])
    layout_jitter = resolve_games_layout_jitter(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.match3.layout",
    )
    unit_scale, unit_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.match3.unit_size",
    )

    rows = len(sample.board)
    cols = len(sample.board[0]) if rows else 0
    cell = scale_games_px(_int_default(params, render_defaults, "cell_size_px", DEFAULTS.cell_size_px), float(unit_scale), min_px=28)
    gap = scale_games_px(_int_default(params, render_defaults, "cell_gap_px", DEFAULTS.cell_gap_px), float(unit_scale), min_px=2)
    gem_inset = scale_games_px(_int_default(params, render_defaults, "gem_inset_px", DEFAULTS.gem_inset_px), float(unit_scale), min_px=4)
    index_margin = scale_games_px(_int_default(params, render_defaults, "index_margin_px", DEFAULTS.index_margin_px), float(unit_scale), min_px=20)
    inner_margin = scale_games_px(_int_default(params, render_defaults, "board_inner_margin_px", DEFAULTS.board_inner_margin_px), float(unit_scale), min_px=20)
    grid_width = int(cols * cell + max(0, cols - 1) * gap)
    grid_height = int(rows * cell + max(0, rows - 1) * gap)
    panel_width = int(grid_width + index_margin + inner_margin * 2)
    panel_height = int(grid_height + index_margin + inner_margin * 2)
    margin = _int_default(params, render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px)
    panel_width = min(int(panel_width), int(canvas_width - 2 * margin))
    panel_height = min(int(panel_height), int(canvas_height - 2 * margin))
    base_panel = (
        float((canvas_width - panel_width) / 2.0),
        float((canvas_height - panel_height) / 2.0),
        float((canvas_width + panel_width) / 2.0),
        float((canvas_height + panel_height) / 2.0),
    )
    panel_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_panel,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    panel_bbox_i = tuple(int(round(value)) for value in panel_bbox)
    draw_panel_scene_chrome(draw, bbox=panel_bbox_i, style=style, radius=18, border_width=3)
    grid_left = int(round(panel_bbox[0])) + inner_margin + index_margin
    grid_top = int(round(panel_bbox[1])) + inner_margin + index_margin

    label_font = load_font(
        _int_default(params, render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px),
        bold=True,
        font_family=str(font_family),
    )
    index_font = load_font(
        _int_default(params, render_defaults, "index_font_size_px", DEFAULTS.index_font_size_px),
        bold=True,
        font_family=str(font_family),
    )
    text_rgb = tuple(int(value) for value in style.text_rgb)
    border_rgb = tuple(int(value) for value in style.panel_border_rgb)
    cell_fill = blend_rgb(
        tuple(int(value) for value in style.panel_fill_rgb),
        tuple(int(value) for value in style.background_rgb),
        float(match3_style["cell_alpha"]),
    )
    mark_rgb = tuple(int(value) for value in style.mark_rgb)
    grid_width_px = int(match3_style["grid_width"])
    gem_shape = str(match3_style["gem_shape"])
    gem_outline_width = int(match3_style["gem_outline_width"])
    gem_shadow = bool(match3_style["shadow"])

    entities: List[Dict[str, Any]] = []
    entity_bboxes: Dict[str, List[float]] = {}
    entity_points: Dict[str, List[float]] = {}
    gem_specs: List[Dict[str, Any]] = []
    cell_centers: Dict[Coord, Tuple[float, float]] = {}
    for col in range(cols):
        x0 = grid_left + col * (cell + gap)
        bbox = (x0, grid_top - index_margin + 2, x0 + cell, grid_top - 6)
        _draw_centered_text(draw, bbox, str(col + 1), font=index_font, fill=text_rgb, stroke_width=1)
    for row in range(rows):
        y0 = grid_top + row * (cell + gap)
        bbox = (grid_left - index_margin + 2, y0, grid_left - 6, y0 + cell)
        _draw_centered_text(draw, bbox, str(row + 1), font=index_font, fill=text_rgb, stroke_width=1)
    for row in range(rows):
        for col in range(cols):
            x0 = grid_left + col * (cell + gap)
            y0 = grid_top + row * (cell + gap)
            cell_bbox = (int(x0), int(y0), int(x0 + cell), int(y0 + cell))
            draw_panel_grid_cell(draw, bbox=cell_bbox, fill=cell_fill, style=style, outline=style.grid_rgb, width=grid_width_px)
            gem_bbox = (
                float(x0 + gem_inset),
                float(y0 + gem_inset),
                float(x0 + cell - gem_inset),
                float(y0 + cell - gem_inset),
            )
            color_key = str(sample.board[row][col])
            color_rgb = tuple(int(value) for value in GEM_RGB[str(color_key)])
            _draw_gem(
                draw,
                gem_bbox,
                fill_rgb=color_rgb,
                outline_rgb=border_rgb,
                shape=str(gem_shape),
                outline_width=int(gem_outline_width),
                shadow=bool(gem_shadow),
            )
            entity_id = cell_entity_id((int(row), int(col)))
            bbox_list = [round(float(value), 3) for value in gem_bbox]
            entity_bboxes[entity_id] = list(bbox_list)
            center = [round(float(x0 + cell / 2.0), 3), round(float(y0 + cell / 2.0), 3)]
            cell_centers[(int(row), int(col))] = (float(center[0]), float(center[1]))
            entity_points[str(entity_id)] = list(center)
            spec = {
                "entity_id": str(entity_id),
                "entity_type": "match3_gem",
                "row": int(row + 1),
                "col": int(col + 1),
                "color_key": str(color_key),
                "color_name": str(color_key),
                "color_rgb": [int(value) for value in color_rgb],
                "color_hex": rgb_to_hex(color_rgb),
                "bbox_px": list(bbox_list),
            }
            gem_specs.append(dict(spec))
            entities.append(dict(spec))

    option_specs: List[Dict[str, Any]] = []
    arrow_width = scale_games_px(_int_default(params, render_defaults, "arrow_width_px", DEFAULTS.arrow_width_px), float(unit_scale), min_px=4)
    for option in sample.option_specs:
        start = cell_centers[tuple(option.outcome.move.a)]
        end = cell_centers[tuple(option.outcome.move.b)]
        bbox = _draw_arrow(
            draw,
            start,
            end,
            line_rgb=mark_rgb,
            label=str(option.label),
            label_font=label_font,
            text_rgb=text_rgb,
            label_fill_rgb=tuple(style.option_marker_fill_rgb),
            label_outline_rgb=mark_rgb,
            width=int(arrow_width),
        )
        entity_bboxes[str(option.entity_id)] = [float(value) for value in bbox]
        arrow_point = [round(float((start[0] + end[0]) / 2.0), 3), round(float((start[1] + end[1]) / 2.0), 3)]
        entity_points[str(option.entity_id)] = list(arrow_point)
        spec = {
            "entity_id": str(option.entity_id),
            "entity_type": "match3_swap_option_arrow",
            "label": str(option.label),
            "from_cell": [int(option.outcome.move.a[0] + 1), int(option.outcome.move.a[1] + 1)],
            "to_cell": [int(option.outcome.move.b[0] + 1), int(option.outcome.move.b[1] + 1)],
            "clear_count": int(option.outcome.clear_count),
            "run_count": int(option.outcome.run_count),
            "cleared_cells": [[int(row + 1), int(col + 1)] for row, col in option.outcome.cleared_cells],
            "is_answer": bool(option.is_answer),
            "bbox_px": [float(value) for value in bbox],
            "center_px": list(arrow_point),
        }
        option_specs.append(dict(spec))
        entities.append(dict(spec))

    render_map = {
        "entity_bboxes_px": dict(entity_bboxes),
        "entity_points_px": dict(entity_points),
        "gem_bboxes_px": {str(spec["entity_id"]): [float(value) for value in spec["bbox_px"]] for spec in gem_specs},
        "gem_centers_px": {
            str(spec["entity_id"]): list(entity_points[str(spec["entity_id"])])
            for spec in gem_specs
        },
        "swap_arrow_bboxes_px": {str(spec["entity_id"]): [float(value) for value in spec["bbox_px"]] for spec in option_specs},
        "swap_arrow_points_px": {str(spec["entity_id"]): [float(value) for value in spec["center_px"]] for spec in option_specs},
        "grid_bbox_px": [float(grid_left), float(grid_top), float(grid_left + grid_width), float(grid_top + grid_height)],
        "scene_variant": str(sample.scene_variant),
        "panel_scene_style": dict(style_meta),
        "match3_style": {
            "style_variant": str(style_variant),
            "gem_shape": str(gem_shape),
            "cell_fill_rgb": [int(value) for value in cell_fill],
            "grid_width_px": int(grid_width_px),
            "gem_outline_width_px": int(gem_outline_width),
        },
        "text_style": {
            "font_family": str(font_family),
            "font_asset": get_font_family_record(str(font_family)).to_trace(),
            "text_rgb": [int(value) for value in text_rgb],
        },
        "layout_jitter": attach_games_unit_size_jitter(resolved_jitter, unit_meta),
        "effective_cell_size_px": int(cell),
        "effective_cell_gap_px": int(gap),
    }
    return RenderedMatch3Scene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=dict(render_map),
        style_meta=dict(style_meta),
        background_meta=dict(background_meta),
    )


__all__ = ["render_match3_scene"]
