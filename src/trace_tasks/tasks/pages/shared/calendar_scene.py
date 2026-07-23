"""Shared month-calendar rendering helpers for page tasks."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import hash64
from ...shared.text_rendering import draw_text_centered, fit_font_to_box, load_font, resolve_text_stroke_fill
from ...shared.time_artifact_style import TimeArtifactCalendarTheme
from ...shared.time_format import month_name, weekday_abbreviation


SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "minimal",
    "outline",
)


@dataclass(frozen=True)
class CalendarRenderParams:
    """Resolved rendering parameters for a month-view calendar scene."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    title_height_px: int
    title_bottom_gap_px: int
    weekday_header_height_px: int
    weekday_grid_gap_px: int
    cell_gap_px: int
    panel_corner_radius_px: int
    panel_outline_width_px: int
    cell_corner_radius_px: int
    cell_outline_width_px: int
    title_font_size_px: int
    weekday_font_size_px: int
    date_font_size_px: int
    marker_inset_px: int
    marker_outline_width_px: int


@dataclass(frozen=True)
class RenderedCalendarScene:
    """Rendered month-view calendar geometry and projected date-cell metadata."""

    year: int
    month: int
    first_weekday_index: int
    row_count: int
    title_text: str
    title_bbox_px: Tuple[float, float, float, float] | None
    panel_bbox_px: Tuple[float, float, float, float]
    scene_bbox_px: Tuple[float, float, float, float]
    date_cell_bboxes_by_day: Dict[int, Tuple[float, float, float, float]]
    entities: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class CalendarEventChipSpec:
    """One event/category chip rendered inside a calendar date cell."""

    day: int
    slot_id: str
    slot_label: str
    category_label: str
    fill_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedCalendarEventGridScene:
    """Rendered calendar event-grid geometry and projected event-chip metadata."""

    year: int
    month: int
    first_weekday_index: int
    row_count: int
    title_text: str
    title_bbox_px: Tuple[float, float, float, float] | None
    panel_bbox_px: Tuple[float, float, float, float]
    scene_bbox_px: Tuple[float, float, float, float]
    date_cell_bboxes_by_day: Dict[int, Tuple[float, float, float, float]]
    event_chip_bboxes_by_key: Dict[str, Tuple[float, float, float, float]]
    event_chips: Tuple[Dict[str, Any], ...]
    entities: Tuple[Dict[str, Any], ...]


def _require_int(value: Any, *, name: str) -> int:
    """Convert one render parameter to `int` and fail loudly on missing values."""

    if value is None:
        raise ValueError(f"missing required calendar render parameter: {name}")
    return int(value)


def resolve_calendar_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    fallback_values: Mapping[str, Any],
    instance_seed: int | None = None,
) -> CalendarRenderParams:
    """Resolve the active month-calendar render parameters from task defaults."""

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
            index = abs(int(hash64(int(seed), f"calendar_render:{str(key)}", 40231)))
            return int(low) + int(index % (int(high) - int(low) + 1))
        return _require_int(render_defaults.get(str(key), fallback), name=str(key))

    resolved: Dict[str, int] = {}
    for key, fallback in fallback_values.items():
        if key not in {
            "canvas_width",
            "canvas_height",
            "outer_margin_px",
            "title_height_px",
            "title_bottom_gap_px",
            "weekday_header_height_px",
            "weekday_grid_gap_px",
            "cell_gap_px",
            "panel_corner_radius_px",
            "panel_outline_width_px",
            "cell_corner_radius_px",
            "cell_outline_width_px",
            "title_font_size_px",
            "weekday_font_size_px",
            "date_font_size_px",
            "marker_inset_px",
            "marker_outline_width_px",
        }:
            continue
        resolved[str(key)] = _resolve_int(str(key), fallback)
    return CalendarRenderParams(**resolved)


def _rounded(draw: ImageDraw.ImageDraw, bbox: Sequence[float], *, radius: int, fill=None, outline=None, width: int = 1) -> None:
    """Draw one rounded rectangle with consistent coordinate coercion."""

    draw.rounded_rectangle([float(value) for value in bbox], radius=int(radius), fill=fill, outline=outline, width=int(width))


def _validate_first_weekday_index(first_weekday_index: int) -> int:
    """Validate the semantic weekday used as the first rendered calendar column."""

    value = int(first_weekday_index)
    if not 0 <= value <= 6:
        raise ValueError("first_weekday_index must be in 0..6")
    return int(value)


def _semantic_weekday_index(*, display_weekday_index: int, first_weekday_index: int) -> int:
    """Return Python-style weekday index for one rendered display column."""

    return int((int(first_weekday_index) + int(display_weekday_index)) % 7)


def render_month_calendar_scene(
    image: Image.Image,
    *,
    year: int,
    month: int,
    marked_dates: Sequence[int],
    scene_variant: str,
    render_params: CalendarRenderParams,
    visual_theme: TimeArtifactCalendarTheme,
    first_weekday_index: int = 0,
    panel_bbox_px: Sequence[float] | None = None,
    title_text: str | None = None,
) -> RenderedCalendarScene:
    """Render one month-view calendar and return projected valid-date geometry."""

    resolved_first_weekday_index = _validate_first_weekday_index(int(first_weekday_index))
    calendar_rows = calendar.Calendar(firstweekday=int(resolved_first_weekday_index)).monthdayscalendar(int(year), int(month))
    row_count = len(calendar_rows)
    if int(row_count) not in {4, 5, 6}:
        raise ValueError("monthdayscalendar returned an unsupported row count")

    draw = ImageDraw.Draw(image)
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    weekday_font = load_font(int(render_params.weekday_font_size_px), bold=True)
    date_font = load_font(int(render_params.date_font_size_px), bold=False)

    if panel_bbox_px is None:
        panel_bbox = (
            float(render_params.outer_margin_px),
            float(render_params.outer_margin_px),
            float(render_params.canvas_width - render_params.outer_margin_px),
            float(render_params.canvas_height - render_params.outer_margin_px),
        )
    else:
        if len(panel_bbox_px) < 4:
            raise ValueError("panel_bbox_px must contain four coordinates")
        panel_bbox = tuple(float(value) for value in panel_bbox_px[:4])
        if float(panel_bbox[2]) <= float(panel_bbox[0]) or float(panel_bbox[3]) <= float(panel_bbox[1]):
            raise ValueError("panel_bbox_px must have positive width and height")
    panel_outline_width = 0 if str(scene_variant) == "minimal" else int(render_params.panel_outline_width_px)
    _rounded(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(int(channel) for channel in visual_theme.panel_fill_rgb),
        outline=(tuple(int(channel) for channel in visual_theme.panel_outline_rgb) if panel_outline_width > 0 else None),
        width=max(1, int(panel_outline_width)) if panel_outline_width > 0 else 1,
    )

    resolved_title_text = f"{month_name(int(month))} {int(year)}" if title_text is None else str(title_text).strip()
    title_bbox: Tuple[float, float, float, float] | None = None
    if resolved_title_text:
        title_bbox = (
            float(panel_bbox[0]) + float(render_params.cell_gap_px),
            float(panel_bbox[1]),
            float(panel_bbox[2]) - float(render_params.cell_gap_px),
            float(panel_bbox[1]) + float(render_params.title_height_px),
        )
        title_center = (
            0.5 * float(title_bbox[0] + title_bbox[2]),
            0.5 * float(title_bbox[1] + title_bbox[3]),
        )
        draw_text_centered(
            draw,
            text=str(resolved_title_text),
            center=title_center,
            font=title_font,
            fill=tuple(int(channel) for channel in visual_theme.title_text_rgb),
            stroke_fill=resolve_text_stroke_fill(tuple(int(channel) for channel in visual_theme.title_text_rgb)),
        )

    grid_left = float(panel_bbox[0]) + float(render_params.cell_gap_px)
    grid_right = float(panel_bbox[2]) - float(render_params.cell_gap_px)
    weekday_top = (
        float(panel_bbox[1]) + float(render_params.title_height_px) + float(render_params.title_bottom_gap_px)
        if resolved_title_text
        else float(panel_bbox[1]) + float(render_params.cell_gap_px)
    )
    available_grid_top = weekday_top + float(render_params.weekday_header_height_px) + float(render_params.weekday_grid_gap_px)
    grid_bottom = float(panel_bbox[3]) - float(render_params.cell_gap_px)
    cell_gap = float(render_params.cell_gap_px)
    available_grid_width = float(grid_right - grid_left)
    available_grid_height = float(grid_bottom - available_grid_top)
    cell_width = (float(available_grid_width) - (6.0 * cell_gap)) / 7.0
    cell_height = (float(available_grid_height) - (float(row_count - 1) * cell_gap)) / float(row_count)
    if float(cell_width) <= 0.0 or float(cell_height) <= 0.0:
        raise ValueError("calendar render area is too small for the requested month grid")

    entities: List[Dict[str, Any]] = []
    date_cell_bboxes_by_day: Dict[int, Tuple[float, float, float, float]] = {}
    marked_date_set = {int(day) for day in marked_dates}

    for display_weekday_index in range(7):
        semantic_weekday_index = _semantic_weekday_index(
            display_weekday_index=int(display_weekday_index),
            first_weekday_index=int(resolved_first_weekday_index),
        )
        header_x1 = float(grid_left + (display_weekday_index * (cell_width + cell_gap)))
        header_x2 = float(header_x1 + cell_width)
        header_bbox = (
            float(header_x1),
            float(weekday_top),
            float(header_x2),
            float(weekday_top + float(render_params.weekday_header_height_px)),
        )
        header_fill = None if str(scene_variant) == "minimal" else tuple(int(channel) for channel in visual_theme.weekday_fill_rgb)
        header_outline = (
            tuple(int(channel) for channel in visual_theme.panel_outline_rgb)
            if str(scene_variant) in {"classic", "outline"}
            else None
        )
        _rounded(
            draw,
            header_bbox,
            radius=max(4, int(render_params.cell_corner_radius_px) - 2),
            fill=header_fill,
            outline=header_outline,
            width=max(1, int(render_params.cell_outline_width_px) - 1) if header_outline is not None else 1,
        )
        draw_text_centered(
            draw,
            text=weekday_abbreviation(int(semantic_weekday_index)),
            center=(0.5 * float(header_bbox[0] + header_bbox[2]), 0.5 * float(header_bbox[1] + header_bbox[3])),
            font=weekday_font,
            fill=tuple(int(channel) for channel in visual_theme.weekday_text_rgb),
            stroke_fill=resolve_text_stroke_fill(tuple(int(channel) for channel in visual_theme.weekday_text_rgb)),
        )

    for row_index, week in enumerate(calendar_rows):
        row_y1 = float(available_grid_top + (row_index * (cell_height + cell_gap)))
        row_y2 = float(row_y1 + cell_height)
        for display_weekday_index, day_value in enumerate(week):
            semantic_weekday_index = _semantic_weekday_index(
                display_weekday_index=int(display_weekday_index),
                first_weekday_index=int(resolved_first_weekday_index),
            )
            cell_x1 = float(grid_left + (display_weekday_index * (cell_width + cell_gap)))
            cell_x2 = float(cell_x1 + cell_width)
            cell_bbox = (float(cell_x1), float(row_y1), float(cell_x2), float(row_y2))
            cell_outline = (
                tuple(int(channel) for channel in visual_theme.grid_line_rgb)
                if str(scene_variant) != "minimal"
                else tuple(int(channel) for channel in visual_theme.grid_line_rgb)
            )
            cell_fill = tuple(int(channel) for channel in visual_theme.panel_fill_rgb)
            _rounded(
                draw,
                cell_bbox,
                radius=int(render_params.cell_corner_radius_px),
                fill=cell_fill,
                outline=cell_outline,
                width=max(1, int(render_params.cell_outline_width_px)),
            )

            if int(day_value) <= 0:
                continue

            is_marked = int(day_value) in marked_date_set
            if is_marked:
                marker_bbox = (
                    float(cell_bbox[0] + float(render_params.marker_inset_px)),
                    float(cell_bbox[1] + float(render_params.marker_inset_px)),
                    float(cell_bbox[2] - float(render_params.marker_inset_px)),
                    float(cell_bbox[3] - float(render_params.marker_inset_px)),
                )
                marker_fill = (
                    tuple(int(channel) for channel in visual_theme.marker_fill_rgb)
                    if str(visual_theme.marker_kind) == "fill"
                    else None
                )
                _rounded(
                    draw,
                    marker_bbox,
                    radius=max(4, int(render_params.cell_corner_radius_px) - 2),
                    fill=marker_fill,
                    outline=tuple(int(channel) for channel in visual_theme.marker_outline_rgb),
                    width=max(1, int(render_params.marker_outline_width_px)),
                )

            draw_text_centered(
                draw,
                text=str(int(day_value)),
                center=(0.5 * float(cell_bbox[0] + cell_bbox[2]), 0.5 * float(cell_bbox[1] + cell_bbox[3])),
                font=date_font,
                fill=(
                    tuple(int(channel) for channel in visual_theme.marker_text_rgb)
                    if is_marked
                    else tuple(int(channel) for channel in visual_theme.date_text_rgb)
                ),
                stroke_fill=resolve_text_stroke_fill(
                    tuple(int(channel) for channel in visual_theme.marker_text_rgb)
                    if is_marked
                    else tuple(int(channel) for channel in visual_theme.date_text_rgb)
                ),
            )

            date_cell_bboxes_by_day[int(day_value)] = tuple(float(value) for value in cell_bbox)
            entities.append(
                {
                    "entity_id": f"date_{int(day_value)}",
                    "entity_type": "calendar_date_cell",
                    "bbox_px": [round(float(value), 3) for value in cell_bbox],
                    "attrs": {
                        "date_number": int(day_value),
                        "weekday_index": int(semantic_weekday_index),
                        "display_weekday_index": int(display_weekday_index),
                        "week_row_index": int(row_index),
                        "year": int(year),
                        "month": int(month),
                        "month_name": str(month_name(int(month))),
                        "is_marked": bool(is_marked),
                    },
                }
            )

    return RenderedCalendarScene(
        year=int(year),
        month=int(month),
        first_weekday_index=int(resolved_first_weekday_index),
        row_count=int(row_count),
        title_text=str(resolved_title_text),
        title_bbox_px=(tuple(float(value) for value in title_bbox) if title_bbox is not None else None),
        panel_bbox_px=tuple(float(value) for value in panel_bbox),
        scene_bbox_px=tuple(float(value) for value in panel_bbox),
        date_cell_bboxes_by_day={int(day): tuple(float(value) for value in bbox) for day, bbox in date_cell_bboxes_by_day.items()},
        entities=tuple(dict(entity) for entity in entities),
    )


def _chip_key(day: int, slot_id: str) -> str:
    """Return a stable key for one event chip."""

    return f"date_{int(day)}__slot_{str(slot_id)}"


def render_month_calendar_event_grid_scene(
    image: Image.Image,
    *,
    year: int,
    month: int,
    event_chips: Sequence[CalendarEventChipSpec],
    slot_order: Sequence[str],
    scene_variant: str,
    render_params: CalendarRenderParams,
    visual_theme: TimeArtifactCalendarTheme,
    first_weekday_index: int = 0,
    panel_bbox_px: Sequence[float] | None = None,
    title_text: str | None = None,
) -> RenderedCalendarEventGridScene:
    """Render one month calendar with event/category chips inside date cells."""

    resolved_first_weekday_index = _validate_first_weekday_index(int(first_weekday_index))
    calendar_rows = calendar.Calendar(firstweekday=int(resolved_first_weekday_index)).monthdayscalendar(int(year), int(month))
    row_count = len(calendar_rows)
    if int(row_count) not in {4, 5, 6}:
        raise ValueError("monthdayscalendar returned an unsupported row count")

    slot_ids = tuple(str(value) for value in slot_order if str(value).strip())
    if not slot_ids:
        raise ValueError("slot_order must contain at least one slot id")
    slot_index_by_id = {slot_id: index for index, slot_id in enumerate(slot_ids)}

    chips_by_day_slot: Dict[Tuple[int, str], CalendarEventChipSpec] = {}
    for chip in event_chips:
        day = int(chip.day)
        slot_id = str(chip.slot_id)
        if slot_id not in slot_index_by_id:
            raise ValueError(f"event chip uses unsupported slot_id: {slot_id}")
        key = (day, slot_id)
        if key in chips_by_day_slot:
            raise ValueError(f"duplicate event chip for date/slot: {key}")
        chips_by_day_slot[key] = chip

    draw = ImageDraw.Draw(image)
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    weekday_font = load_font(int(render_params.weekday_font_size_px), bold=True)
    date_font = load_font(max(14, int(render_params.date_font_size_px) - 4), bold=True)
    chip_font = load_font(max(10, int(render_params.weekday_font_size_px) - 3), bold=True)

    if panel_bbox_px is None:
        panel_bbox = (
            float(render_params.outer_margin_px),
            float(render_params.outer_margin_px),
            float(render_params.canvas_width - render_params.outer_margin_px),
            float(render_params.canvas_height - render_params.outer_margin_px),
        )
    else:
        if len(panel_bbox_px) < 4:
            raise ValueError("panel_bbox_px must contain four coordinates")
        panel_bbox = tuple(float(value) for value in panel_bbox_px[:4])
        if float(panel_bbox[2]) <= float(panel_bbox[0]) or float(panel_bbox[3]) <= float(panel_bbox[1]):
            raise ValueError("panel_bbox_px must have positive width and height")

    panel_outline_width = 0 if str(scene_variant) == "minimal" else int(render_params.panel_outline_width_px)
    _rounded(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(int(channel) for channel in visual_theme.panel_fill_rgb),
        outline=(tuple(int(channel) for channel in visual_theme.panel_outline_rgb) if panel_outline_width > 0 else None),
        width=max(1, int(panel_outline_width)) if panel_outline_width > 0 else 1,
    )

    resolved_title_text = f"{month_name(int(month))} {int(year)}" if title_text is None else str(title_text).strip()
    title_bbox: Tuple[float, float, float, float] | None = None
    if resolved_title_text:
        title_bbox = (
            float(panel_bbox[0]) + float(render_params.cell_gap_px),
            float(panel_bbox[1]),
            float(panel_bbox[2]) - float(render_params.cell_gap_px),
            float(panel_bbox[1]) + float(render_params.title_height_px),
        )
        draw_text_centered(
            draw,
            text=str(resolved_title_text),
            center=(0.5 * float(title_bbox[0] + title_bbox[2]), 0.5 * float(title_bbox[1] + title_bbox[3])),
            font=title_font,
            fill=tuple(int(channel) for channel in visual_theme.title_text_rgb),
            stroke_fill=resolve_text_stroke_fill(tuple(int(channel) for channel in visual_theme.title_text_rgb)),
        )

    grid_left = float(panel_bbox[0]) + float(render_params.cell_gap_px)
    grid_right = float(panel_bbox[2]) - float(render_params.cell_gap_px)
    weekday_top = (
        float(panel_bbox[1]) + float(render_params.title_height_px) + float(render_params.title_bottom_gap_px)
        if resolved_title_text
        else float(panel_bbox[1]) + float(render_params.cell_gap_px)
    )
    available_grid_top = weekday_top + float(render_params.weekday_header_height_px) + float(render_params.weekday_grid_gap_px)
    grid_bottom = float(panel_bbox[3]) - float(render_params.cell_gap_px)
    cell_gap = float(render_params.cell_gap_px)
    available_grid_width = float(grid_right - grid_left)
    available_grid_height = float(grid_bottom - available_grid_top)
    cell_width = (float(available_grid_width) - (6.0 * cell_gap)) / 7.0
    cell_height = (float(available_grid_height) - (float(row_count - 1) * cell_gap)) / float(row_count)
    if float(cell_width) <= 0.0 or float(cell_height) <= 0.0:
        raise ValueError("calendar event-grid render area is too small")

    entities: List[Dict[str, Any]] = []
    event_records: List[Dict[str, Any]] = []
    date_cell_bboxes_by_day: Dict[int, Tuple[float, float, float, float]] = {}
    event_chip_bboxes_by_key: Dict[str, Tuple[float, float, float, float]] = {}

    for display_weekday_index in range(7):
        semantic_weekday_index = _semantic_weekday_index(
            display_weekday_index=int(display_weekday_index),
            first_weekday_index=int(resolved_first_weekday_index),
        )
        header_x1 = float(grid_left + (display_weekday_index * (cell_width + cell_gap)))
        header_x2 = float(header_x1 + cell_width)
        header_bbox = (
            float(header_x1),
            float(weekday_top),
            float(header_x2),
            float(weekday_top + float(render_params.weekday_header_height_px)),
        )
        header_fill = None if str(scene_variant) == "minimal" else tuple(int(channel) for channel in visual_theme.weekday_fill_rgb)
        header_outline = (
            tuple(int(channel) for channel in visual_theme.panel_outline_rgb)
            if str(scene_variant) in {"classic", "outline"}
            else None
        )
        _rounded(
            draw,
            header_bbox,
            radius=max(4, int(render_params.cell_corner_radius_px) - 2),
            fill=header_fill,
            outline=header_outline,
            width=max(1, int(render_params.cell_outline_width_px) - 1) if header_outline is not None else 1,
        )
        draw_text_centered(
            draw,
            text=weekday_abbreviation(int(semantic_weekday_index)),
            center=(0.5 * float(header_bbox[0] + header_bbox[2]), 0.5 * float(header_bbox[1] + header_bbox[3])),
            font=weekday_font,
            fill=tuple(int(channel) for channel in visual_theme.weekday_text_rgb),
            stroke_fill=resolve_text_stroke_fill(tuple(int(channel) for channel in visual_theme.weekday_text_rgb)),
        )

    for row_index, week in enumerate(calendar_rows):
        row_y1 = float(available_grid_top + (row_index * (cell_height + cell_gap)))
        row_y2 = float(row_y1 + cell_height)
        for display_weekday_index, day_value in enumerate(week):
            semantic_weekday_index = _semantic_weekday_index(
                display_weekday_index=int(display_weekday_index),
                first_weekday_index=int(resolved_first_weekday_index),
            )
            cell_x1 = float(grid_left + (display_weekday_index * (cell_width + cell_gap)))
            cell_x2 = float(cell_x1 + cell_width)
            cell_bbox = (float(cell_x1), float(row_y1), float(cell_x2), float(row_y2))
            cell_outline = tuple(int(channel) for channel in visual_theme.grid_line_rgb)
            _rounded(
                draw,
                cell_bbox,
                radius=int(render_params.cell_corner_radius_px),
                fill=tuple(int(channel) for channel in visual_theme.panel_fill_rgb),
                outline=cell_outline,
                width=max(1, int(render_params.cell_outline_width_px)),
            )
            if int(day_value) <= 0:
                continue

            date_strip_bbox = (
                float(cell_bbox[0] + 4.0),
                float(cell_bbox[1] + 2.0),
                float(cell_bbox[0] + min(36.0, max(24.0, cell_width * 0.30))),
                float(cell_bbox[1] + 24.0),
            )
            draw_text_centered(
                draw,
                text=str(int(day_value)),
                center=(0.5 * float(date_strip_bbox[0] + date_strip_bbox[2]), 0.5 * float(date_strip_bbox[1] + date_strip_bbox[3])),
                font=date_font,
                fill=tuple(int(channel) for channel in visual_theme.date_text_rgb),
                stroke_fill=resolve_text_stroke_fill(tuple(int(channel) for channel in visual_theme.date_text_rgb)),
            )

            date_cell_bboxes_by_day[int(day_value)] = tuple(float(value) for value in cell_bbox)
            entities.append(
                {
                    "entity_id": f"date_{int(day_value)}",
                    "entity_type": "calendar_event_grid_date_cell",
                    "bbox_px": [round(float(value), 3) for value in cell_bbox],
                    "attrs": {
                        "date_number": int(day_value),
                        "weekday_index": int(semantic_weekday_index),
                        "display_weekday_index": int(display_weekday_index),
                        "week_row_index": int(row_index),
                        "year": int(year),
                        "month": int(month),
                        "month_name": str(month_name(int(month))),
                    },
                }
            )

            chip_area_top = float(cell_bbox[1]) + 26.0
            chip_area_bottom = float(cell_bbox[3]) - 5.0
            chip_gap = 4.0
            chip_height = max(
                12.0,
                min(20.0, (float(chip_area_bottom - chip_area_top) - (float(len(slot_ids) - 1) * chip_gap)) / float(len(slot_ids))),
            )
            for slot_id in slot_ids:
                chip = chips_by_day_slot.get((int(day_value), str(slot_id)))
                if chip is None:
                    continue
                slot_index = int(slot_index_by_id[str(slot_id)])
                chip_y1 = float(chip_area_top + (slot_index * (chip_height + chip_gap)))
                chip_bbox = (
                    float(cell_bbox[0] + 6.0),
                    float(chip_y1),
                    float(cell_bbox[2] - 6.0),
                    float(chip_y1 + chip_height),
                )
                chip_fill = tuple(int(channel) for channel in chip.fill_rgb)
                chip_text = tuple(int(channel) for channel in chip.text_rgb)
                _rounded(
                    draw,
                    chip_bbox,
                    radius=max(4, int(render_params.cell_corner_radius_px) - 6),
                    fill=chip_fill,
                    outline=tuple(int(channel) for channel in visual_theme.grid_line_rgb),
                    width=1,
                )
                chip_text_content = f"{str(chip.slot_label)}: {str(chip.category_label)}"
                fitted_chip_font = fit_font_to_box(
                    draw,
                    text=chip_text_content,
                    max_width=max(8.0, float(chip_bbox[2] - chip_bbox[0]) - 8.0),
                    max_height=max(8.0, float(chip_bbox[3] - chip_bbox[1]) - 2.0),
                    bold=True,
                    min_size_px=8,
                    max_size_px=max(9, int(getattr(chip_font, "size", 12))),
                    fill_ratio=0.92,
                )
                draw_text_centered(
                    draw,
                    text=chip_text_content,
                    center=(0.5 * float(chip_bbox[0] + chip_bbox[2]), 0.5 * float(chip_bbox[1] + chip_bbox[3])),
                    font=fitted_chip_font,
                    fill=chip_text,
                    stroke_fill=resolve_text_stroke_fill(chip_text),
                )
                stable_key = _chip_key(int(day_value), str(slot_id))
                event_chip_bboxes_by_key[stable_key] = tuple(float(value) for value in chip_bbox)
                event_record = {
                    "entity_id": stable_key,
                    "entity_type": "calendar_event_chip",
                    "bbox_px": [round(float(value), 3) for value in chip_bbox],
                    "attrs": {
                        "date_number": int(day_value),
                        "slot_id": str(chip.slot_id),
                        "slot_label": str(chip.slot_label),
                        "category_label": str(chip.category_label),
                        "year": int(year),
                        "month": int(month),
                    },
                }
                event_records.append(dict(event_record))
                entities.append(dict(event_record))

    return RenderedCalendarEventGridScene(
        year=int(year),
        month=int(month),
        first_weekday_index=int(resolved_first_weekday_index),
        row_count=int(row_count),
        title_text=str(resolved_title_text),
        title_bbox_px=(tuple(float(value) for value in title_bbox) if title_bbox is not None else None),
        panel_bbox_px=tuple(float(value) for value in panel_bbox),
        scene_bbox_px=tuple(float(value) for value in panel_bbox),
        date_cell_bboxes_by_day={int(day): tuple(float(value) for value in bbox) for day, bbox in date_cell_bboxes_by_day.items()},
        event_chip_bboxes_by_key={str(key): tuple(float(value) for value in bbox) for key, bbox in event_chip_bboxes_by_key.items()},
        event_chips=tuple(dict(record) for record in event_records),
        entities=tuple(dict(entity) for entity in entities),
    )


__all__ = [
    "CalendarEventChipSpec",
    "CalendarRenderParams",
    "RenderedCalendarEventGridScene",
    "RenderedCalendarScene",
    "SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS",
    "render_month_calendar_event_grid_scene",
    "render_month_calendar_scene",
    "resolve_calendar_render_params",
]
