"""Shared milestone-timeline rendering helpers for page tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.shared.text_rendering import (
    draw_text_centered,
    fit_font_to_box,
    load_font,
    resolve_text_stroke_fill,
)
from trace_tasks.tasks.shared.time_artifact_style import TimeArtifactTimelineTheme


SUPPORTED_PAGE_TIMELINE_SCENE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "roadmap",
    "minimal",
)


@dataclass(frozen=True)
class TimelineRenderParams:
    """Resolved rendering parameters for one milestone-timeline scene."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    title_height_px: int
    title_gap_px: int
    panel_corner_radius_px: int
    panel_outline_width_px: int
    axis_width_px: int
    axis_tick_height_px: int
    marker_radius_px: int
    marker_outline_width_px: int
    connector_width_px: int
    card_width_px: int
    card_height_px: int
    card_corner_radius_px: int
    card_outline_width_px: int
    label_font_size_px: int
    date_font_size_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    event_vertical_gap_px: int
    event_stem_length_px: int


@dataclass(frozen=True)
class TimelineEventSpec:
    """One rendered milestone event on the timeline."""

    event_id: str
    label: str
    date_text: str
    order_index: int
    anchor_x_px: float
    card_side: str
    reference_kind: str = "none"


@dataclass(frozen=True)
class RenderedTimelineScene:
    """Rendered timeline geometry and event-card projections."""

    title_text: str
    subtitle_text: str
    scene_bbox_px: Tuple[float, float, float, float]
    panel_bbox_px: Tuple[float, float, float, float]
    axis_bbox_px: Tuple[float, float, float, float]
    event_bboxes_by_id: Dict[str, Tuple[float, float, float, float]]
    entities: Tuple[Dict[str, Any], ...]


def _require_int(value: Any, *, name: str) -> int:
    """Convert one render parameter to ``int`` and fail loudly when missing."""

    if value is None:
        raise ValueError(f"missing required timeline render parameter: {name}")
    return int(value)


def resolve_timeline_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    fallback_values: Mapping[str, Any],
    instance_seed: int | None = None,
) -> TimelineRenderParams:
    """Resolve timeline geometry knobs after applying bounded render jitter."""

    def _resolve_int(key: str, fallback: Any) -> int:
        if params.get(str(key)) is not None:
            return _require_int(params[str(key)], name=str(key))
        low_raw = params.get(f"{str(key)}_min", render_defaults.get(f"{str(key)}_min"))
        high_raw = params.get(f"{str(key)}_max", render_defaults.get(f"{str(key)}_max"))
        if low_raw is not None or high_raw is not None:
            default_value = _require_int(render_defaults.get(str(key), fallback), name=str(key))
            low = _require_int(default_value if low_raw is None else low_raw, name=f"{str(key)}_min")
            high = _require_int(default_value if high_raw is None else high_raw, name=f"{str(key)}_max")
            if int(low) > int(high):
                raise ValueError(f"{str(key)}_min must be <= {str(key)}_max")
            seed = 0 if instance_seed is None else int(instance_seed)
            index = abs(int(hash64(int(seed), f"timeline_render:{str(key)}", 71933)))
            return int(low) + int(index % (int(high) - int(low) + 1))
        return _require_int(render_defaults.get(str(key), fallback), name=str(key))

    resolved: Dict[str, int] = {}
    allowed_keys = {
        "canvas_width",
        "canvas_height",
        "outer_margin_px",
        "title_height_px",
        "title_gap_px",
        "panel_corner_radius_px",
        "panel_outline_width_px",
        "axis_width_px",
        "axis_tick_height_px",
        "marker_radius_px",
        "marker_outline_width_px",
        "connector_width_px",
        "card_width_px",
        "card_height_px",
        "card_corner_radius_px",
        "card_outline_width_px",
        "label_font_size_px",
        "date_font_size_px",
        "title_font_size_px",
        "subtitle_font_size_px",
        "event_vertical_gap_px",
        "event_stem_length_px",
    }
    for key, fallback in fallback_values.items():
        if key not in allowed_keys:
            continue
        resolved[str(key)] = _resolve_int(str(key), fallback)
    return TimelineRenderParams(**resolved)


def _rounded(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    radius: int,
    fill=None,
    outline=None,
    width: int = 1,
) -> None:
    """Draw one rounded rectangle with stable coordinate coercion."""

    draw.rounded_rectangle([float(value) for value in bbox], radius=int(radius), fill=fill, outline=outline, width=int(width))


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: int,
    outline_width: int,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    shape: str,
) -> None:
    """Draw one timeline milestone marker."""

    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    if str(shape) == "diamond":
        points = [
            (cx, cy - r),
            (cx + r, cy),
            (cx, cy + r),
            (cx - r, cy),
        ]
        draw.polygon(points, fill=fill_rgb, outline=outline_rgb)
        if int(outline_width) > 1:
            inset = max(1.0, float(outline_width) * 0.5)
            inner_points = [
                (cx, cy - r + inset),
                (cx + r - inset, cy),
                (cx, cy + r - inset),
                (cx - r + inset, cy),
            ]
            draw.polygon(inner_points, outline=outline_rgb)
    else:
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=fill_rgb,
            outline=outline_rgb,
            width=int(outline_width),
        )


def render_timeline_scene(
    image: Image.Image,
    *,
    title_text: str,
    subtitle_text: str,
    events: Sequence[TimelineEventSpec],
    scene_variant: str,
    render_params: TimelineRenderParams,
    visual_theme: TimeArtifactTimelineTheme,
) -> RenderedTimelineScene:
    """Render one milestone timeline and return projected event-card geometry."""

    if not events:
        raise ValueError("timeline rendering requires at least one event")

    draw = ImageDraw.Draw(image)
    canvas_width = int(render_params.canvas_width)
    canvas_height = int(render_params.canvas_height)
    outer_margin = int(render_params.outer_margin_px)
    title_height = int(render_params.title_height_px)
    panel_bbox = (
        float(outer_margin),
        float(outer_margin),
        float(canvas_width - outer_margin),
        float(canvas_height - outer_margin),
    )
    scene_bbox = panel_bbox
    _rounded(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=visual_theme.panel_fill_rgb,
        outline=visual_theme.panel_outline_rgb,
        width=int(render_params.panel_outline_width_px),
    )

    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    subtitle_font = load_font(int(render_params.subtitle_font_size_px), bold=False)
    title_center_x = float((panel_bbox[0] + panel_bbox[2]) * 0.5)
    draw_text_centered(
        draw,
        text=str(title_text),
        center=(title_center_x, float(panel_bbox[1] + (0.42 * title_height))),
        font=title_font,
        fill=visual_theme.title_text_rgb,
    )
    draw_text_centered(
        draw,
        text=str(subtitle_text),
        center=(title_center_x, float(panel_bbox[1] + (0.78 * title_height))),
        font=subtitle_font,
        fill=visual_theme.subtitle_text_rgb,
    )

    content_top = float(panel_bbox[1] + title_height + int(render_params.title_gap_px))
    axis_y = float((content_top + panel_bbox[3]) * 0.54)
    axis_left = float(panel_bbox[0] + 56)
    axis_right = float(panel_bbox[2] - 56)
    axis_bbox = (axis_left, axis_y - 2.0, axis_right, axis_y + 2.0)
    draw.line(
        [(axis_left, axis_y), (axis_right, axis_y)],
        fill=visual_theme.axis_line_rgb,
        width=int(render_params.axis_width_px),
    )

    marker_shape = "circle"
    if str(scene_variant) == "roadmap":
        marker_shape = "diamond"
    marker_radius = int(render_params.marker_radius_px)
    tick_height = int(render_params.axis_tick_height_px)

    count = len(events)
    if int(count) == 1:
        x_positions = [float((axis_left + axis_right) * 0.5)]
    else:
        usable_width = float(axis_right - axis_left)
        x_positions = [
            float(axis_left + (usable_width * (float(index) / float(max(1, count - 1)))))
            for index in range(count)
        ]

    card_width = min(
        int(render_params.card_width_px),
        max(72, int((float(axis_right - axis_left) / float(max(1, count))) * 0.86)),
    )
    card_height = int(render_params.card_height_px)
    label_font = fit_font_to_box(
        draw,
        text="W",
        max_width=max(28, int(card_width - 12)),
        max_height=max(18, int(card_height * 0.46)),
        bold=True,
        max_size_px=int(render_params.label_font_size_px),
    )
    date_font = fit_font_to_box(
        draw,
        text="Sep 28",
        max_width=max(42, int(card_width - 12)),
        max_height=max(16, int(card_height * 0.34)),
        bold=False,
        max_size_px=int(render_params.date_font_size_px),
    )

    event_bboxes_by_id: Dict[str, Tuple[float, float, float, float]] = {}
    entities: List[Dict[str, Any]] = []
    for event, anchor_x in zip(events, x_positions):
        is_above = str(event.card_side) == "above"
        stem_length = int(render_params.event_stem_length_px)
        vertical_gap = int(render_params.event_vertical_gap_px)
        stem_end_y = float(axis_y - stem_length) if is_above else float(axis_y + stem_length)
        card_top = float(stem_end_y - vertical_gap - card_height) if is_above else float(stem_end_y + vertical_gap)
        card_left = float(anchor_x - (card_width * 0.5))
        card_left = max(float(panel_bbox[0] + 18), min(card_left, float(panel_bbox[2] - 18 - card_width)))
        card_bbox = (
            card_left,
            float(card_top),
            float(card_left + card_width),
            float(card_top + card_height),
        )
        event_bboxes_by_id[str(event.event_id)] = card_bbox

        is_reference = str(event.reference_kind) != "none"
        connector_width = int(render_params.connector_width_px) + (1 if is_reference else 0)
        draw.line(
            [(anchor_x, axis_y), (anchor_x, stem_end_y)],
            fill=visual_theme.connector_line_rgb,
            width=int(connector_width),
        )
        draw.line(
            [(anchor_x, axis_y - tick_height), (anchor_x, axis_y + tick_height)],
            fill=visual_theme.tick_line_rgb,
            width=max(1, int(render_params.marker_outline_width_px)),
        )

        if is_reference:
            card_fill = visual_theme.primary_reference_fill_rgb
            card_outline = visual_theme.primary_reference_outline_rgb
            card_text = visual_theme.primary_reference_text_rgb
        else:
            card_fill = visual_theme.event_fill_rgb
            card_outline = visual_theme.event_outline_rgb
            card_text = visual_theme.event_text_rgb
        text_stroke_fill = resolve_text_stroke_fill(card_text)
        label_stroke_width = 2 if is_reference else None

        _rounded(
            draw,
            card_bbox,
            radius=int(render_params.card_corner_radius_px),
            fill=card_fill,
            outline=card_outline,
            width=int(render_params.card_outline_width_px) + (1 if is_reference else 0),
        )
        label_center_y = float(card_bbox[1] + (card_height * 0.34))
        date_center_y = float(card_bbox[1] + (card_height * 0.72))
        draw_text_centered(
            draw,
            text=str(event.label),
            center=(float((card_bbox[0] + card_bbox[2]) * 0.5), label_center_y),
            font=label_font,
            fill=card_text,
            stroke_fill=text_stroke_fill,
            stroke_width=label_stroke_width,
        )
        draw_text_centered(
            draw,
            text=str(event.date_text),
            center=(float((card_bbox[0] + card_bbox[2]) * 0.5), date_center_y),
            font=date_font,
            fill=visual_theme.event_subtext_rgb if not is_reference else card_text,
            stroke_fill=text_stroke_fill if is_reference else resolve_text_stroke_fill(visual_theme.event_subtext_rgb),
            stroke_width=0,
        )

        _draw_marker(
            draw,
            center=(anchor_x, axis_y),
            radius=int(marker_radius) + (2 if is_reference else 0),
            outline_width=int(render_params.marker_outline_width_px),
            fill_rgb=visual_theme.marker_fill_rgb if not is_reference else card_fill,
            outline_rgb=visual_theme.marker_outline_rgb if not is_reference else card_outline,
            shape=marker_shape,
        )
        entities.append(
            {
                "type": "timeline_event_card",
                "event_id": str(event.event_id),
                "label": str(event.label),
                "date_text": str(event.date_text),
                "order_index": int(event.order_index),
                "reference_kind": str(event.reference_kind),
                "card_bbox_px": [round(float(value), 3) for value in card_bbox],
                "marker_center_px": [round(float(anchor_x), 3), round(float(axis_y), 3)],
                "card_side": str(event.card_side),
            }
        )

    return RenderedTimelineScene(
        title_text=str(title_text),
        subtitle_text=str(subtitle_text),
        scene_bbox_px=tuple(float(value) for value in scene_bbox),
        panel_bbox_px=tuple(float(value) for value in panel_bbox),
        axis_bbox_px=tuple(float(value) for value in axis_bbox),
        event_bboxes_by_id={str(key): tuple(float(value) for value in bbox) for key, bbox in event_bboxes_by_id.items()},
        entities=tuple(entities),
    )


__all__ = [
    "RenderedTimelineScene",
    "SUPPORTED_PAGE_TIMELINE_SCENE_VARIANTS",
    "TimelineEventSpec",
    "TimelineRenderParams",
    "render_timeline_scene",
    "resolve_timeline_render_params",
]
