"""Rendering helpers for matchstick puzzle scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.scene_style import make_puzzle_scene_background
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)
from trace_tasks.tasks.shared.text_rendering import load_font

from .rules import (
    DIGIT_SEGMENTS,
    SEGMENT_POINTS,
    equation_text,
    lattice_edge_item_id,
    lattice_edges,
    lattice_square_item_id,
    number_segments,
    number_text,
)
from .state import (
    BBox,
    Color,
    EquationRepairDataset,
    NumberDataset,
    RenderParams,
    RenderedScene,
    Segment,
    SquareCompletionDataset,
)


def _to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> RenderParams:
    """Resolve all render dimensions from params and scene defaults."""

    return RenderParams(
        canvas_width=max(
            900,
            _to_int(
                params.get("canvas_width", group_default(render_defaults, "canvas_width", 1200)),
                1200,
            ),
        ),
        canvas_height=max(
            480,
            _to_int(
                params.get("canvas_height", group_default(render_defaults, "canvas_height", 900)),
                900,
            ),
        ),
        margin_px=max(
            40,
            _to_int(params.get("margin_px", group_default(render_defaults, "margin_px", 58)), 58),
        ),
        source_panel_height_px=max(
            160,
            _to_int(
                params.get(
                    "source_panel_height_px",
                    group_default(render_defaults, "source_panel_height_px", 230),
                ),
                230,
            ),
        ),
        option_panel_width_px=max(
            180,
            _to_int(
                params.get(
                    "option_panel_width_px",
                    group_default(render_defaults, "option_panel_width_px", 320),
                ),
                320,
            ),
        ),
        option_panel_height_px=max(
            150,
            _to_int(
                params.get(
                    "option_panel_height_px",
                    group_default(render_defaults, "option_panel_height_px", 218),
                ),
                218,
            ),
        ),
        option_gap_px=max(
            12,
            _to_int(
                params.get("option_gap_px", group_default(render_defaults, "option_gap_px", 28)),
                28,
            ),
        ),
        panel_corner_radius_px=max(
            0,
            _to_int(
                params.get(
                    "panel_corner_radius_px",
                    group_default(render_defaults, "panel_corner_radius_px", 20),
                ),
                20,
            ),
        ),
        panel_border_width_px=max(
            1,
            _to_int(
                params.get(
                    "panel_border_width_px",
                    group_default(render_defaults, "panel_border_width_px", 3),
                ),
                3,
            ),
        ),
        stick_width_px=max(
            5,
            _to_int(
                params.get("stick_width_px", group_default(render_defaults, "stick_width_px", 13)),
                13,
            ),
        ),
        option_label_font_size_px=max(
            18,
            _to_int(
                params.get(
                    "option_label_font_size_px",
                    group_default(render_defaults, "option_label_font_size_px", 26),
                ),
                26,
            ),
        ),
        caption_font_size_px=max(
            16,
            _to_int(
                params.get(
                    "caption_font_size_px",
                    group_default(render_defaults, "caption_font_size_px", 21),
                ),
                21,
            ),
        ),
        source_caption_font_size_px=max(
            18,
            _to_int(
                params.get(
                    "source_caption_font_size_px",
                    group_default(render_defaults, "source_caption_font_size_px", 24),
                ),
                24,
            ),
        ),
    )


def sample_matchstick_font(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> str:
    """Sample one global font family for option labels and captions."""

    return sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.matchstick_font",
        params={**dict(render_defaults), **dict(params)},
    )


def font_trace_record(font_family: str) -> Dict[str, Any]:
    """Build trace metadata for the sampled label/caption font."""

    return {
        **get_font_family_record(str(font_family)).to_trace(),
        "source": "global_font_pool",
        "font_asset_version": font_asset_version(),
        "selection_scope": "matchstick_option_labels_and_captions",
        "include_tags": [],
        "exclude_tags": [],
    }


def style_for_variant(scene_variant: str) -> Dict[str, Any]:
    """Return the material palette for one matchstick visual variant."""

    if scene_variant == "chalk_sticks":
        return {
            "background": (33, 39, 45),
            "panel_fill": (42, 49, 57),
            "panel_outline": (142, 154, 166),
            "stick": (235, 237, 230),
            "stick_shadow": (22, 26, 30),
            "label_fill": (248, 249, 244),
            "label_text": (25, 30, 36),
            "caption": (244, 246, 240),
            "tip": None,
            "palette": (),
        }
    if scene_variant == "neon_rods":
        return {
            "background": (20, 24, 39),
            "panel_fill": (26, 30, 49),
            "panel_outline": (80, 95, 150),
            "stick": (84, 214, 230),
            "stick_shadow": (30, 74, 92),
            "label_fill": (236, 244, 255),
            "label_text": (24, 30, 48),
            "caption": (232, 240, 255),
            "tip": None,
            "palette": (
                (84, 214, 230),
                (238, 111, 196),
                (255, 214, 96),
                (140, 226, 153),
            ),
        }
    if scene_variant == "colored_rods":
        return {
            "background": (246, 249, 252),
            "panel_fill": (255, 255, 255),
            "panel_outline": (86, 99, 122),
            "stick": (78, 137, 205),
            "stick_shadow": (225, 232, 242),
            "label_fill": (32, 40, 54),
            "label_text": (255, 255, 255),
            "caption": (28, 34, 44),
            "tip": None,
            "palette": (
                (64, 129, 202),
                (224, 107, 82),
                (64, 156, 118),
                (171, 108, 202),
                (222, 167, 62),
            ),
        }
    if scene_variant == "metal_rods":
        return {
            "background": (247, 248, 250),
            "panel_fill": (252, 253, 255),
            "panel_outline": (102, 111, 124),
            "stick": (154, 163, 174),
            "stick_shadow": (218, 222, 228),
            "label_fill": (40, 45, 54),
            "label_text": (255, 255, 255),
            "caption": (30, 35, 43),
            "tip": None,
            "palette": (),
        }
    return {
        "background": (251, 247, 238),
        "panel_fill": (255, 253, 247),
        "panel_outline": (128, 103, 75),
        "stick": (213, 174, 112),
        "stick_shadow": (236, 219, 188),
        "label_fill": (67, 52, 35),
        "label_text": (255, 255, 255),
        "caption": (55, 43, 31),
        "tip": (197, 54, 48),
        "palette": (),
    }


def matchstick_style_trace(scene_variant: str) -> Dict[str, Any]:
    """Serialize one material style for trace metadata."""

    style = style_for_variant(str(scene_variant))
    trace: Dict[str, Any] = {
        "scene_variant": str(scene_variant),
        "panel_fill_rgb": [int(value) for value in style["panel_fill"]],
        "panel_outline_rgb": [int(value) for value in style["panel_outline"]],
        "stick_rgb": [int(value) for value in style["stick"]],
        "stick_shadow_rgb": [int(value) for value in style["stick_shadow"]],
        "label_fill_rgb": [int(value) for value in style["label_fill"]],
        "label_text_rgb": [int(value) for value in style["label_text"]],
        "caption_rgb": [int(value) for value in style["caption"]],
        "palette_rgb": [[int(value) for value in color] for color in style.get("palette", ())],
    }
    tip = style.get("tip")
    trace["tip_rgb"] = None if tip is None else [int(value) for value in tip]
    return trace


def _stick_color(style: Mapping[str, Any], stick_id: str) -> Color:
    palette = style.get("palette") or ()
    if isinstance(palette, tuple) and palette:
        index = sum(ord(ch) for ch in str(stick_id)) % len(palette)
        return tuple(int(value) for value in palette[index])  # type: ignore[return-value]
    return tuple(int(value) for value in style["stick"])  # type: ignore[return-value]


def _draw_stick(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
    style: Mapping[str, Any],
    stick_id: str,
) -> None:
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    shadow = tuple(int(value) for value in style["stick_shadow"])
    color = _stick_color(style, str(stick_id))
    shadow_width = int(width) + max(2, int(width // 3))
    draw.line([(sx, sy), (ex, ey)], fill=shadow, width=shadow_width)
    radius = max(2, int(shadow_width // 2))
    draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=shadow)
    draw.ellipse((ex - radius, ey - radius, ex + radius, ey + radius), fill=shadow)
    draw.line([(sx, sy), (ex, ey)], fill=color, width=int(width))
    radius = max(2, int(width // 2))
    draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=color)
    draw.ellipse((ex - radius, ey - radius, ex + radius, ey + radius), fill=color)
    tip = style.get("tip")
    if tip is not None:
        tip_radius = max(3, int(width // 2))
        draw.ellipse(
            (ex - tip_radius, ey - tip_radius, ex + tip_radius, ey + tip_radius),
            fill=tuple(int(v) for v in tip),
        )


def _draw_label_chip(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    label: str,
    render_params: RenderParams,
    style: Mapping[str, Any],
) -> None:
    chip = int(max(34, render_params.option_label_font_size_px + 16))
    chip_bbox = (
        int(bbox[0] + 12),
        int(bbox[1] + 10),
        int(bbox[0] + 12 + chip),
        int(bbox[1] + 10 + chip),
    )
    draw.rounded_rectangle(
        chip_bbox,
        radius=9,
        fill=tuple(style["label_fill"]),
        outline=(255, 255, 255),
        width=1,
    )
    font = load_font(int(render_params.option_label_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text=str(label),
        center=((chip_bbox[0] + chip_bbox[2]) / 2, (chip_bbox[1] + chip_bbox[3]) / 2),
        font=font,
        fill=tuple(style["label_text"]),
        stroke_fill=tuple(style["label_fill"]),
        stroke_width=0,
    )


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    text: str,
    font_size: int,
    style: Mapping[str, Any],
) -> None:
    font = load_font(int(font_size), bold=True)
    draw_centered_text(
        draw,
        text=str(text),
        center=((bbox[0] + bbox[2]) / 2.0, bbox[1] + int(font_size * 0.9)),
        font=font,
        fill=tuple(style["caption"]),
        stroke_fill=tuple(style["panel_fill"]),
        stroke_width=1,
    )


def _draw_number(
    draw: ImageDraw.ImageDraw,
    *,
    number: int,
    bbox: tuple[int, int, int, int],
    render_params: RenderParams,
    style: Mapping[str, Any],
    small: bool,
) -> None:
    segments = number_segments(int(number))
    min_x = min(min(start[0], end[0]) for _sid, start, end in segments)
    max_x = max(max(start[0], end[0]) for _sid, start, end in segments)
    min_y = min(min(start[1], end[1]) for _sid, start, end in segments)
    max_y = max(max(start[1], end[1]) for _sid, start, end in segments)
    usable_w = max(1, int((bbox[2] - bbox[0]) * (0.62 if small else 0.52)))
    usable_h = max(1, int((bbox[3] - bbox[1]) * (0.58 if small else 0.66)))
    scale = min(
        float(usable_w) / max(1e-6, max_x - min_x),
        float(usable_h) / max(1e-6, max_y - min_y),
    )
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0 + (10 if small else 14)
    total_w = (max_x - min_x) * scale
    total_h = (max_y - min_y) * scale
    origin_x = cx - (total_w / 2.0) - (min_x * scale)
    origin_y = cy - (total_h / 2.0) - (min_y * scale)
    width = max(5, int(render_params.stick_width_px * (0.82 if small else 1.20)))
    for segment_id, start, end in segments:
        _draw_stick(
            draw,
            start=(origin_x + start[0] * scale, origin_y + start[1] * scale),
            end=(origin_x + end[0] * scale, origin_y + end[1] * scale),
            width=width,
            style=style,
            stick_id=f"number:{segment_id}",
        )


def _digit_segment_specs(
    digit: int,
    *,
    digit_index: int,
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """Return drawable segment ids and normalized endpoints for one digit."""

    specs: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for segment_key in sorted(DIGIT_SEGMENTS[int(digit)]):
        start, end = SEGMENT_POINTS[str(segment_key)]
        specs.append((f"digit{int(digit_index)}:{segment_key}", start, end))
    return specs


def _draw_equation_digit(
    draw: ImageDraw.ImageDraw,
    *,
    digit: int,
    digit_index: int,
    origin: tuple[float, float],
    scale: float,
    width: int,
    style: Mapping[str, Any],
) -> Dict[str, Segment]:
    """Draw one equation digit and return source-stick centerline segments."""

    segments: Dict[str, Segment] = {}
    origin_x, origin_y = float(origin[0]), float(origin[1])
    for segment_id, start, end in _digit_segment_specs(
        int(digit),
        digit_index=int(digit_index),
    ):
        start_px = (origin_x + start[0] * scale, origin_y + start[1] * scale)
        end_px = (origin_x + end[0] * scale, origin_y + end[1] * scale)
        _draw_stick(
            draw,
            start=start_px,
            end=end_px,
            width=int(width),
            style=style,
            stick_id=f"equation:{segment_id}",
        )
        segments[str(segment_id)] = (
            (float(start_px[0]), float(start_px[1])),
            (float(end_px[0]), float(end_px[1])),
        )
    return segments


def _draw_operator(
    draw: ImageDraw.ImageDraw,
    *,
    operator: str,
    center: tuple[float, float],
    scale: float,
    width: int,
    style: Mapping[str, Any],
) -> None:
    """Draw matchstick-style arithmetic operators from fixed stick primitives."""

    cx, cy = float(center[0]), float(center[1])
    half = float(scale) * 0.28
    if str(operator) == "+":
        _draw_stick(
            draw,
            start=(cx - half, cy),
            end=(cx + half, cy),
            width=int(width),
            style=style,
            stick_id="operator:plus:h",
        )
        _draw_stick(
            draw,
            start=(cx, cy - half),
            end=(cx, cy + half),
            width=int(width),
            style=style,
            stick_id="operator:plus:v",
        )
        return
    if str(operator) == "-":
        _draw_stick(
            draw,
            start=(cx - half, cy),
            end=(cx + half, cy),
            width=int(width),
            style=style,
            stick_id="operator:minus:h",
        )
        return
    if str(operator) == "=":
        gap = float(scale) * 0.16
        for offset, stick_id in ((-gap, "operator:eq:top"), (gap, "operator:eq:bottom")):
            _draw_stick(
                draw,
                start=(cx - half, cy + offset),
                end=(cx + half, cy + offset),
                width=int(width),
                style=style,
                stick_id=stick_id,
            )
        return
    raise ValueError(f"unsupported matchstick operator: {operator!r}")


def _draw_stick_label(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    segment: Segment,
    render_params: RenderParams,
    style: Mapping[str, Any],
) -> None:
    """Draw one option label beside a candidate stick."""

    (sx, sy), (ex, ey) = segment
    mid_x = (float(sx) + float(ex)) / 2.0
    mid_y = (float(sy) + float(ey)) / 2.0
    horizontal = abs(float(ex) - float(sx)) >= abs(float(ey) - float(sy))
    radius = max(13, int(render_params.option_label_font_size_px * 0.55))
    offset = max(22, int(radius + render_params.stick_width_px * 1.2))
    center = (mid_x, mid_y - offset) if horizontal else (mid_x + offset, mid_y)
    fill = tuple(int(value) for value in style["label_fill"])
    outline = tuple(int(value) for value in style["panel_fill"])
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=fill,
        outline=outline,
        width=2,
    )
    font = load_font(int(render_params.option_label_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text=str(label),
        center=center,
        font=font,
        fill=tuple(style["label_text"]),
        stroke_fill=fill,
        stroke_width=0,
    )


def _equation_layout(
    *,
    panel_bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, list[float], list[float]]:
    """Return scale, baseline origin, token starts, and token widths."""

    token_widths = [1.0, 0.72, 1.0, 0.82, 1.0]
    gap = 0.42
    total_units = sum(token_widths) + gap * (len(token_widths) - 1)
    usable_w = float(panel_bbox[2] - panel_bbox[0]) * 0.78
    usable_h = float(panel_bbox[3] - panel_bbox[1]) * 0.56
    scale = min(usable_w / total_units, usable_h / 2.0)
    total_w = total_units * scale
    start_x = (float(panel_bbox[0] + panel_bbox[2]) - total_w) / 2.0
    origin_y = (float(panel_bbox[1] + panel_bbox[3]) - 2.0 * scale) / 2.0 + 14.0
    starts: list[float] = []
    cursor = start_x
    for width in token_widths:
        starts.append(float(cursor))
        cursor += (float(width) + gap) * scale
    return float(scale), float(origin_y), float(gap * scale), starts, token_widths


def option_bboxes(
    render_params: RenderParams,
    option_count: int,
    *,
    include_source: bool,
) -> list[tuple[int, int, int, int]]:
    """Return the 3-by-2 option-card layout used by both tasks."""

    cols = 3
    rows = 2
    total_w = (
        cols * render_params.option_panel_width_px
        + (cols - 1) * render_params.option_gap_px
    )
    start_x = int((render_params.canvas_width - total_w) / 2)
    if include_source:
        start_y = int(render_params.margin_px + render_params.source_panel_height_px + 44)
    else:
        total_h = (
            rows * render_params.option_panel_height_px
            + (rows - 1) * render_params.option_gap_px
        )
        start_y = int((render_params.canvas_height - total_h) / 2)
    bboxes: list[tuple[int, int, int, int]] = []
    for index in range(int(option_count)):
        row = int(index // cols)
        col = int(index % cols)
        if row >= rows:
            break
        x0 = int(start_x + col * (render_params.option_panel_width_px + render_params.option_gap_px))
        y0 = int(start_y + row * (render_params.option_panel_height_px + render_params.option_gap_px))
        bboxes.append(
            (
                x0,
                y0,
                int(x0 + render_params.option_panel_width_px),
                int(y0 + render_params.option_panel_height_px),
            )
        )
    return bboxes


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    *,
    render_params: RenderParams,
    style: Mapping[str, Any],
) -> None:
    """Draw a reusable content panel for matchstick scenes."""

    draw_rounded_rect(
        draw,
        bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(style["panel_fill"]),
        outline=tuple(style["panel_outline"]),
        width=int(render_params.panel_border_width_px),
    )


def _lattice_layout(
    *,
    panel_bbox: tuple[int, int, int, int],
    rows: int,
    cols: int,
    render_params: RenderParams,
) -> tuple[float, float, float]:
    """Return centered square-cell lattice origin and cell size."""

    inner_margin = max(54, int(render_params.margin_px * 1.15))
    usable_w = max(1, int(panel_bbox[2] - panel_bbox[0] - 2 * inner_margin))
    usable_h = max(1, int(panel_bbox[3] - panel_bbox[1] - 2 * inner_margin))
    cell_size = min(float(usable_w) / max(1, int(cols)), float(usable_h) / max(1, int(rows)))
    grid_w = float(cols) * float(cell_size)
    grid_h = float(rows) * float(cell_size)
    origin_x = (float(panel_bbox[0]) + float(panel_bbox[2]) - grid_w) / 2.0
    origin_y = (float(panel_bbox[1]) + float(panel_bbox[3]) - grid_h) / 2.0
    return float(origin_x), float(origin_y), float(cell_size)


def _lattice_edge_segment(
    edge_id: str,
    *,
    origin_x: float,
    origin_y: float,
    cell_size: float,
) -> Segment:
    """Project one logical lattice edge into image-pixel centerline endpoints."""

    axis, row, col = str(edge_id).split(":", 2)
    row_i = int(row)
    col_i = int(col)
    if axis == "h":
        return (
            (origin_x + float(col_i) * cell_size, origin_y + float(row_i) * cell_size),
            (origin_x + float(col_i + 1) * cell_size, origin_y + float(row_i) * cell_size),
        )
    if axis == "v":
        return (
            (origin_x + float(col_i) * cell_size, origin_y + float(row_i) * cell_size),
            (origin_x + float(col_i) * cell_size, origin_y + float(row_i + 1) * cell_size),
        )
    raise ValueError(f"unsupported lattice edge id: {edge_id!r}")


def _draw_lattice_nodes(
    draw: ImageDraw.ImageDraw,
    *,
    rows: int,
    cols: int,
    origin_x: float,
    origin_y: float,
    cell_size: float,
    style: Mapping[str, Any],
) -> None:
    """Draw small vertex dots so empty edge positions remain visually legible."""

    outline = tuple(int(value) for value in style["panel_outline"])
    fill = tuple(int(value) for value in style["panel_fill"])
    radius = max(3, int(cell_size * 0.035))
    for row in range(int(rows) + 1):
        for col in range(int(cols) + 1):
            cx = origin_x + float(col) * cell_size
            cy = origin_y + float(row) * cell_size
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=fill,
                outline=outline,
                width=2,
            )


def make_scene_background(
    *,
    render_params: RenderParams,
    style: Any,
) -> tuple[Image.Image, Dict[str, Any]]:
    """Create the shared puzzle canvas background for matchstick panels."""

    return make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
    )


def render_number_scene(
    *,
    background: Image.Image,
    dataset: NumberDataset,
    render_params: RenderParams,
) -> RenderedScene:
    """Render a source matchstick number and six candidate numbers."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    style = style_for_variant(str(dataset.scene_variant))
    source_bbox = (
        int(render_params.margin_px),
        int(render_params.margin_px),
        int(render_params.canvas_width - render_params.margin_px),
        int(render_params.margin_px + render_params.source_panel_height_px),
    )
    _draw_panel(draw, source_bbox, render_params=render_params, style=style)
    _draw_caption(
        draw,
        bbox=source_bbox,
        text="Source",
        font_size=int(render_params.source_caption_font_size_px),
        style=style,
    )
    _draw_number(
        draw,
        number=int(dataset.source_number),
        bbox=source_bbox,
        render_params=render_params,
        style=style,
        small=False,
    )
    item_bbox_map: Dict[str, BBox] = {
        "source_panel": tuple(float(value) for value in source_bbox)
    }
    entities: list[Dict[str, Any]] = [
        {
            "id": "source_panel",
            "type": "matchstick_number_source",
            "bbox_px": [int(value) for value in source_bbox],
            "number": number_text(int(dataset.source_number)),
        }
    ]
    bboxes = option_bboxes(render_params, int(dataset.option_count), include_source=True)
    for index, option in enumerate(dataset.option_specs):
        bbox = bboxes[int(index)]
        option_id = f"option_{option.label}"
        _draw_panel(draw, bbox, render_params=render_params, style=style)
        _draw_label_chip(
            draw,
            bbox=bbox,
            label=str(option.label),
            render_params=render_params,
            style=style,
        )
        _draw_number(
            draw,
            number=int(option.value),
            bbox=bbox,
            render_params=render_params,
            style=style,
            small=True,
        )
        item_bbox_map[str(option_id)] = tuple(float(value) for value in bbox)
        entities.append(
            {
                "id": str(option_id),
                "type": "matchstick_number_option",
                "label": str(option.label),
                "bbox_px": [int(value) for value in bbox],
                "number": number_text(int(option.value)),
                "is_correct": bool(option.is_correct),
            }
        )
    scene_bbox = (
        float(render_params.margin_px),
        float(render_params.margin_px),
        float(render_params.canvas_width - render_params.margin_px),
        float(render_params.canvas_height - render_params.margin_px),
    )
    return RenderedScene(
        image=image,
        scene_bbox_px=scene_bbox,
        item_bbox_map=item_bbox_map,
        entities=tuple(entities),
    )


def render_equation_repair_scene(
    *,
    background: Image.Image,
    dataset: EquationRepairDataset,
    render_params: RenderParams,
) -> RenderedScene:
    """Render one false matchstick equation with labeled removable sticks."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    style = style_for_variant(str(dataset.scene_variant))
    panel_bbox = (
        int(render_params.margin_px),
        int(render_params.margin_px),
        int(render_params.canvas_width - render_params.margin_px),
        int(render_params.canvas_height - render_params.margin_px),
    )
    _draw_panel(draw, panel_bbox, render_params=render_params, style=style)
    _draw_caption(
        draw,
        bbox=panel_bbox,
        text="Equation",
        font_size=int(render_params.source_caption_font_size_px),
        style=style,
    )
    scale, origin_y, _gap_px, starts, widths = _equation_layout(panel_bbox=panel_bbox)
    stick_width = max(6, int(render_params.stick_width_px * 1.12))
    item_segment_map: Dict[str, Segment] = {}
    equation_segments: Dict[str, Segment] = {}
    digit_token_indices = (0, 2, 4)
    for digit_index, token_index in enumerate(digit_token_indices):
        equation_segments.update(
            _draw_equation_digit(
                draw,
                digit=int(dataset.source_digits[int(digit_index)]),
                digit_index=int(digit_index),
                origin=(starts[int(token_index)], origin_y),
                scale=float(scale),
                width=int(stick_width),
                style=style,
            )
        )
    operator_y = origin_y + float(scale)
    _draw_operator(
        draw,
        operator=str(dataset.operator),
        center=(starts[1] + widths[1] * scale / 2.0, operator_y),
        scale=float(scale),
        width=int(stick_width),
        style=style,
    )
    _draw_operator(
        draw,
        operator="=",
        center=(starts[3] + widths[3] * scale / 2.0, operator_y),
        scale=float(scale),
        width=int(stick_width),
        style=style,
    )

    entities: list[Dict[str, Any]] = [
        {
            "id": "equation_panel",
            "type": "matchstick_equation_panel",
            "bbox_px": [int(value) for value in panel_bbox],
            "source_equation": equation_text(
                tuple(int(value) for value in dataset.source_digits),
                str(dataset.operator),
            ),
            "repaired_equation": equation_text(
                tuple(int(value) for value in dataset.repaired_digits),
                str(dataset.operator),
            ),
        }
    ]
    item_bbox_map: Dict[str, BBox] = {
        "equation_panel": tuple(float(value) for value in panel_bbox)
    }
    for option in dataset.option_specs:
        stick_id = str(option.value)
        segment = equation_segments[str(stick_id)]
        option_item_id = f"stick_{option.label}"
        item_segment_map[str(option_item_id)] = segment
        _draw_stick_label(
            draw,
            label=str(option.label),
            segment=segment,
            render_params=render_params,
            style=style,
        )
        entities.append(
            {
                "id": str(option_item_id),
                "type": "matchstick_labeled_digit_stick",
                "label": str(option.label),
                "source_stick_id": stick_id,
                "segment_px": [
                    [round(float(point[0]), 3), round(float(point[1]), 3)]
                    for point in segment
                ],
                "is_correct": bool(option.is_correct),
            }
        )
    scene_bbox = (
        float(render_params.margin_px),
        float(render_params.margin_px),
        float(render_params.canvas_width - render_params.margin_px),
        float(render_params.canvas_height - render_params.margin_px),
    )
    return RenderedScene(
        image=image,
        scene_bbox_px=scene_bbox,
        item_bbox_map=item_bbox_map,
        entities=tuple(entities),
        item_segment_map=item_segment_map,
    )


def render_square_lattice_scene(
    *,
    background: Image.Image,
    dataset: SquareCompletionDataset,
    render_params: RenderParams,
) -> RenderedScene:
    """Render one incomplete matchstick lattice for square-completion reasoning."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    style = style_for_variant(str(dataset.scene_variant))
    panel_bbox = (
        int(render_params.margin_px),
        int(render_params.margin_px),
        int(render_params.canvas_width - render_params.margin_px),
        int(render_params.canvas_height - render_params.margin_px),
    )
    _draw_panel(draw, panel_bbox, render_params=render_params, style=style)
    origin_x, origin_y, cell_size = _lattice_layout(
        panel_bbox=panel_bbox,
        rows=int(dataset.rows),
        cols=int(dataset.cols),
        render_params=render_params,
    )
    _draw_lattice_nodes(
        draw,
        rows=int(dataset.rows),
        cols=int(dataset.cols),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        cell_size=float(cell_size),
        style=style,
    )

    item_bbox_map: Dict[str, BBox] = {
        "lattice_panel": tuple(float(value) for value in panel_bbox)
    }
    item_segment_map: Dict[str, Segment] = {}
    entities: list[Dict[str, Any]] = [
        {
            "id": "lattice_panel",
            "type": "matchstick_square_lattice_panel",
            "bbox_px": [int(value) for value in panel_bbox],
            "rows": int(dataset.rows),
            "cols": int(dataset.cols),
        }
    ]

    for edge_id in lattice_edges(int(dataset.rows), int(dataset.cols)):
        segment = _lattice_edge_segment(
            str(edge_id),
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            cell_size=float(cell_size),
        )
        edge_item_id = lattice_edge_item_id(str(edge_id))
        item_segment_map[str(edge_item_id)] = segment
        if str(edge_id) not in set(dataset.present_edges):
            entities.append(
                {
                    "id": str(edge_item_id),
                    "type": "matchstick_empty_lattice_edge",
                    "edge_id": str(edge_id),
                    "segment_px": [
                        [round(float(point[0]), 3), round(float(point[1]), 3)]
                        for point in segment
                    ],
                }
            )
            continue
        _draw_stick(
            draw,
            start=segment[0],
            end=segment[1],
            width=max(6, int(render_params.stick_width_px)),
            style=style,
            stick_id=f"lattice:{edge_id}",
        )
        entities.append(
            {
                "id": str(edge_item_id),
                "type": "matchstick_present_lattice_edge",
                "edge_id": str(edge_id),
                "segment_px": [
                    [round(float(point[0]), 3), round(float(point[1]), 3)]
                    for point in segment
                ],
            }
        )

    for row in range(int(dataset.rows)):
        for col in range(int(dataset.cols)):
            square_logical_id = f"square:{int(row)}:{int(col)}"
            square_item_id = lattice_square_item_id(square_logical_id)
            x0 = origin_x + float(col) * cell_size
            y0 = origin_y + float(row) * cell_size
            x1 = origin_x + float(col + 1) * cell_size
            y1 = origin_y + float(row + 1) * cell_size
            bbox = (float(x0), float(y0), float(x1), float(y1))
            item_bbox_map[str(square_item_id)] = bbox
            entities.append(
                {
                    "id": str(square_item_id),
                    "type": "matchstick_unit_square_cell",
                    "square_id": square_logical_id,
                    "bbox_px": [round(float(value), 3) for value in bbox],
                    "initially_complete": square_logical_id
                    in set(dataset.initial_completed_square_ids),
                    "complete_after_optimal_additions": square_logical_id
                    in set(dataset.completed_square_ids),
                }
            )

    scene_bbox = (
        float(render_params.margin_px),
        float(render_params.margin_px),
        float(render_params.canvas_width - render_params.margin_px),
        float(render_params.canvas_height - render_params.margin_px),
    )
    return RenderedScene(
        image=image,
        scene_bbox_px=scene_bbox,
        item_bbox_map=item_bbox_map,
        entities=tuple(entities),
        item_segment_map=item_segment_map,
    )


__all__ = [
    "font_trace_record",
    "make_scene_background",
    "matchstick_style_trace",
    "render_equation_repair_scene",
    "render_number_scene",
    "render_square_lattice_scene",
    "resolve_render_params",
    "sample_matchstick_font",
]
