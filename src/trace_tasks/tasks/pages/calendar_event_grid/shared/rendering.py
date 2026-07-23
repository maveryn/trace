"""Rendering helpers for calendar event-grid scene packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.calendar_scene import (
    CalendarRenderParams,
    RenderedCalendarScene,
    render_month_calendar_event_grid_scene,
    resolve_calendar_render_params,
)
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.time_artifact_style import build_time_artifact_calendar_theme
from trace_tasks.tasks.shared.time_format import month_name

from .defaults import (
    BACKGROUND_DEFAULTS,
    EVENT_SLOT_SPECS,
    GENERIC_TITLE_TEXTS,
    NOISE_DEFAULTS,
    RENDER_FALLBACKS,
    RENDERING_DEFAULTS,
    SUPPORTED_EVENT_GRID_LAYOUT_MODES,
)
from .state import EventGridCase


@dataclass(frozen=True)
class RenderedEventGridBundle:
    """Rendered image and scene metadata for a sampled event-grid case."""

    image: Image.Image
    render_params: CalendarRenderParams
    rendered_scene: RenderedCalendarScene
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    title_meta: Dict[str, Any]
    panel_layout_meta: Dict[str, Any]
    resolved_colors_rgb: Dict[str, Tuple[int, int, int]]


def resolve_event_grid_title_text(
    *,
    instance_seed: int,
    title_mode: str,
    month: int,
    year: int,
    params: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Resolve visible title text for one sampled title mode."""

    explicit = params.get("calendar_event_grid_title_text")
    if explicit is not None:
        text = str(explicit).strip()
        if not text:
            raise ValueError("calendar_event_grid_title_text must not be empty")
        return text, {"mode": "explicit"}
    if str(title_mode) == "full_month_year":
        return f"{month_name(int(month))} {int(year)}", {"mode": "full_month_year"}
    if str(title_mode) != "generic":
        raise ValueError(f"unsupported calendar event-grid title_mode: {title_mode}")
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace="pages.calendar_event_grid.title_text",
    )
    return str(GENERIC_TITLE_TEXTS[int(index) % len(GENERIC_TITLE_TEXTS)]), {"mode": "generic"}


def resolve_calendar_panel_bbox(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_params: CalendarRenderParams,
    layout_mode: str,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    """Resolve the final panel bbox and placement metadata."""

    explicit_panel_bbox = params.get("calendar_event_grid_panel_bbox_px", params.get("calendar_panel_bbox_px"))
    if explicit_panel_bbox is not None:
        if not isinstance(explicit_panel_bbox, (tuple, list)) or len(explicit_panel_bbox) < 4:
            raise ValueError("calendar_event_grid_panel_bbox_px must be a four-coordinate sequence")
        bbox = tuple(float(value) for value in explicit_panel_bbox[:4])
        return bbox, {
            "layout_placement": {
                "mode": "explicit_bbox",
                "enabled": False,
                "panel_size_px": [round(float(bbox[2] - bbox[0]), 3), round(float(bbox[3] - bbox[1]), 3)],
                "final_origin_px": [round(float(bbox[0]), 3), round(float(bbox[1]), 3)],
            }
        }

    canvas_w = float(render_params.canvas_width)
    canvas_h = float(render_params.canvas_height)
    margin = float(render_params.outer_margin_px)
    panel_w = canvas_w - (2.0 * margin)
    panel_h = canvas_h - (2.0 * margin)
    if str(layout_mode) == "left_with_side_note":
        panel_w = min(panel_w, canvas_w * 0.78)
        x0 = margin
        y0 = margin
    elif str(layout_mode) == "right_with_side_note":
        panel_w = min(panel_w, canvas_w * 0.78)
        x0 = canvas_w - margin - panel_w
        y0 = margin
    elif str(layout_mode) == "top_with_bottom_note":
        panel_h = min(panel_h, canvas_h * 0.82)
        x0 = margin
        y0 = margin
    elif str(layout_mode) == "free_jitter_clean":
        rng = spawn_rng(int(instance_seed), "pages.calendar_event_grid.panel_bbox")
        jitter_x = float(rng.randint(-14, 14))
        jitter_y = float(rng.randint(-12, 12))
        x0 = margin + jitter_x
        y0 = margin + jitter_y
    elif str(layout_mode) == "center_clean":
        x0 = margin
        y0 = margin
    else:
        raise ValueError(f"unsupported calendar event-grid layout_mode: {layout_mode}")
    if str(layout_mode) not in SUPPORTED_EVENT_GRID_LAYOUT_MODES:
        raise ValueError(f"unsupported calendar event-grid layout_mode: {layout_mode}")
    x0 = max(8.0, min(canvas_w - panel_w - 8.0, float(x0)))
    y0 = max(8.0, min(canvas_h - panel_h - 8.0, float(y0)))
    bbox = (float(x0), float(y0), float(x0 + panel_w), float(y0 + panel_h))
    free_x = max(0.0, float(canvas_w - panel_w))
    free_y = max(0.0, float(canvas_h - panel_h))
    default_x0 = float(free_x * 0.5)
    default_y0 = float(free_y * 0.5)
    return bbox, {
        "layout_placement": {
            "mode": "fractional_free_area",
            "enabled": True,
            "layout_mode": str(layout_mode),
            "panel_size_px": [round(float(panel_w), 3), round(float(panel_h), 3)],
            "content_size_px": [round(float(panel_w), 3), round(float(panel_h), 3)],
            "free_space_px": [round(float(free_x), 3), round(float(free_y), 3)],
            "sampled_fractions": {
                "x": round(float(x0 / free_x), 6) if float(free_x) > 0.0 else 0.0,
                "y": round(float(y0 / free_y), 6) if float(free_y) > 0.0 else 0.0,
            },
            "final_origin_px": [round(float(x0), 3), round(float(y0), 3)],
            "dx_dy_from_centered_px": [
                round(float(x0 - default_x0), 3),
                round(float(y0 - default_y0), 3),
            ],
        }
    }


def render_event_grid_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: EventGridCase,
) -> RenderedEventGridBundle:
    """Render one finalized event-grid case and return projection metadata."""

    render_params = resolve_calendar_render_params(
        params,
        render_defaults=RENDERING_DEFAULTS,
        fallback_values=RENDER_FALLBACKS,
        instance_seed=int(instance_seed),
    )
    calendar_theme = build_time_artifact_calendar_theme(
        accent_color_name=str(case.accent_color_name),
        style_variant=str(case.style_variant),
        surface_mode=str(case.surface_mode),
        text_color_mode=str(case.text_color_mode),
    )
    panel_bbox, panel_layout_meta = resolve_calendar_panel_bbox(
        instance_seed=int(instance_seed),
        params=params,
        render_params=render_params,
        layout_mode=str(case.layout_mode),
    )
    title_text, title_meta = resolve_event_grid_title_text(
        instance_seed=int(instance_seed),
        title_mode=str(case.title_mode),
        month=int(case.month),
        year=int(case.year),
        params=params,
    )
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=BACKGROUND_DEFAULTS,
    )
    image = background.copy().convert("RGB")
    rendered_scene = render_month_calendar_event_grid_scene(
        image,
        year=int(case.year),
        month=int(case.month),
        event_chips=tuple(case.event_chips),
        slot_order=tuple(slot for slot, _label in EVENT_SLOT_SPECS),
        scene_variant=str(case.scene_variant),
        render_params=render_params,
        visual_theme=calendar_theme,
        panel_bbox_px=panel_bbox,
        title_text=str(title_text),
    )
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=NOISE_DEFAULTS,
    )
    return RenderedEventGridBundle(
        image=image,
        render_params=render_params,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        title_meta=dict(title_meta),
        panel_layout_meta=dict(panel_layout_meta),
        resolved_colors_rgb={
            "panel_fill": tuple(int(value) for value in calendar_theme.panel_fill_rgb),
            "panel_outline": tuple(int(value) for value in calendar_theme.panel_outline_rgb),
            "title_text": tuple(int(value) for value in calendar_theme.title_text_rgb),
            "weekday_fill": tuple(int(value) for value in calendar_theme.weekday_fill_rgb),
            "weekday_text": tuple(int(value) for value in calendar_theme.weekday_text_rgb),
            "grid_line": tuple(int(value) for value in calendar_theme.grid_line_rgb),
            "date_text": tuple(int(value) for value in calendar_theme.date_text_rgb),
        },
    )
