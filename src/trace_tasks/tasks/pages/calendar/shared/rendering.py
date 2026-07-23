"""Calendar scene rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.information_style import (
    PagesInformationStyle,
    make_pages_information_background,
    resolve_pages_information_style,
)
from trace_tasks.tasks.pages.shared.calendar_scene import (
    CalendarRenderParams,
    RenderedCalendarScene,
    render_month_calendar_scene,
    resolve_calendar_render_params,
)
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.text_legibility import READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO, contrast_ratio
from trace_tasks.tasks.shared.time_artifact_style import TimeArtifactCalendarTheme
from trace_tasks.tasks.shared.time_format import month_name

from .defaults import (
    GENERIC_TITLE_TEXTS,
    NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    RENDER_FALLBACKS,
    SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES,
)
from .state import CalendarCase


@dataclass(frozen=True)
class RenderedCalendarBundle:
    """Rendered image and scene metadata for a sampled calendar case."""

    image: Image.Image
    render_params: CalendarRenderParams
    rendered_scene: RenderedCalendarScene
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    title_meta: Dict[str, Any]
    panel_layout_meta: Dict[str, Any]
    information_style_meta: Dict[str, Any]
    resolved_colors_rgb: Dict[str, Tuple[int, int, int]]
    marker_kind: str


def resolve_calendar_title_text(
    *,
    instance_seed: int,
    title_mode: str,
    month: int,
    year: int,
    params: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Resolve visible calendar title text for one sampled title mode."""

    if str(title_mode) == "none":
        return "", {
            "title_mode": "none",
            "title_text_source": "omitted",
        }
    if str(title_mode) == "full_month_year":
        return f"{month_name(int(month))} {int(year)}", {
            "title_mode": "full_month_year",
            "title_text_source": "month_year",
        }
    if str(title_mode) != "generic":
        raise ValueError(f"unsupported calendar title_mode: {title_mode}")
    explicit_title = params.get("calendar_generic_title_text")
    if explicit_title is not None:
        title_text = str(explicit_title).strip()
        if not title_text:
            raise ValueError("calendar_generic_title_text must not be empty")
        return title_text, {
            "title_mode": "generic",
            "title_text_source": "explicit_param",
        }
    title_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace="pages.calendar:generic_title_text",
        )
        % len(GENERIC_TITLE_TEXTS)
    )
    return str(GENERIC_TITLE_TEXTS[int(title_index)]), {
        "title_mode": "generic",
        "title_text_source": "generic_pool",
        "generic_title_index": int(title_index),
    }


def resolve_calendar_panel_bbox(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_params: CalendarRenderParams,
    layout_mode: str,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    """Sample a final calendar panel bbox using fractions of free canvas space."""

    layout = str(layout_mode)
    if layout not in set(SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES):
        raise ValueError(f"unsupported calendar layout_mode: {layout_mode}")
    profile_by_layout = {
        "center_clean": {
            "width_frac": (0.80, 0.90),
            "height_frac": (0.80, 0.90),
            "x_free_frac": (0.28, 0.72),
            "y_free_frac": (0.28, 0.72),
        },
        "free_jitter_clean": {
            "width_frac": (0.76, 0.88),
            "height_frac": (0.76, 0.88),
            "x_free_frac": (0.08, 0.92),
            "y_free_frac": (0.08, 0.92),
        },
        "left_with_side_note": {
            "width_frac": (0.64, 0.72),
            "height_frac": (0.78, 0.90),
            "x_free_frac": (0.04, 0.18),
            "y_free_frac": (0.18, 0.82),
        },
        "right_with_side_note": {
            "width_frac": (0.64, 0.72),
            "height_frac": (0.78, 0.90),
            "x_free_frac": (0.82, 0.96),
            "y_free_frac": (0.18, 0.82),
        },
        "top_with_bottom_note": {
            "width_frac": (0.76, 0.88),
            "height_frac": (0.62, 0.72),
            "x_free_frac": (0.18, 0.82),
            "y_free_frac": (0.04, 0.18),
        },
    }
    profile = profile_by_layout[layout]
    rng = spawn_rng(int(instance_seed), "pages.calendar.calendar_panel_bbox")

    explicit_panel_bbox = params.get("calendar_panel_bbox_px")
    if explicit_panel_bbox is not None:
        if not isinstance(explicit_panel_bbox, Sequence) or isinstance(explicit_panel_bbox, (str, bytes)) or len(explicit_panel_bbox) < 4:
            raise ValueError("calendar_panel_bbox_px must be a four-coordinate sequence")
        bbox = tuple(float(value) for value in explicit_panel_bbox[:4])
        return bbox, {
            "layout_mode": str(layout),
            "placement_policy": "explicit_calendar_panel_bbox_px",
            "panel_bbox_px": [round(float(value), 3) for value in bbox],
        }

    def _resolve_range(name: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
        explicit = params.get(f"calendar_{layout}_{name}")
        if explicit is None:
            explicit = params.get(f"calendar_{name}")
        if explicit is None:
            return (float(fallback[0]), float(fallback[1]))
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)) or len(explicit) < 2:
            raise ValueError(f"calendar {name} range must contain two values")
        low, high = float(explicit[0]), float(explicit[1])
        if high < low:
            raise ValueError(f"calendar {name} range must be ordered")
        return (float(low), float(high))

    def _sample_range(range_values: Tuple[float, float]) -> float:
        low, high = float(range_values[0]), float(range_values[1])
        return float(low + (rng.random() * (high - low)))

    width_frac_range = _resolve_range("width_frac", tuple(profile["width_frac"]))
    height_frac_range = _resolve_range("height_frac", tuple(profile["height_frac"]))
    x_free_frac_range = _resolve_range("x_free_frac", tuple(profile["x_free_frac"]))
    y_free_frac_range = _resolve_range("y_free_frac", tuple(profile["y_free_frac"]))
    width_frac = max(0.52, min(0.96, _sample_range(width_frac_range)))
    height_frac = max(0.52, min(0.96, _sample_range(height_frac_range)))
    canvas_width = float(render_params.canvas_width)
    canvas_height = float(render_params.canvas_height)
    panel_width = float(canvas_width * width_frac)
    panel_height = float(canvas_height * height_frac)
    free_x = max(0.0, float(canvas_width - panel_width))
    free_y = max(0.0, float(canvas_height - panel_height))
    x_free_frac = max(0.0, min(1.0, _sample_range(x_free_frac_range)))
    y_free_frac = max(0.0, min(1.0, _sample_range(y_free_frac_range)))
    left = float(free_x * x_free_frac)
    top = float(free_y * y_free_frac)
    bbox = (
        float(left),
        float(top),
        float(left + panel_width),
        float(top + panel_height),
    )
    return bbox, {
        "layout_mode": str(layout),
        "placement_policy": "panel_size_fraction_and_free_space_fraction",
        "width_fraction": round(float(width_frac), 6),
        "height_fraction": round(float(height_frac), 6),
        "x_free_fraction": round(float(x_free_frac), 6),
        "y_free_fraction": round(float(y_free_frac), 6),
        "free_space_px": [round(float(free_x), 3), round(float(free_y), 3)],
        "panel_bbox_px": [round(float(value), 3) for value in bbox],
        "range_profile": {
            "width_frac": [round(float(value), 6) for value in width_frac_range],
            "height_frac": [round(float(value), 6) for value in height_frac_range],
            "x_free_frac": [round(float(value), 6) for value in x_free_frac_range],
            "y_free_frac": [round(float(value), 6) for value in y_free_frac_range],
        },
    }


def _readable_text_for_surface(
    surface_rgb: Sequence[int],
    *,
    preferred_rgb: Sequence[int],
) -> Tuple[int, int, int]:
    """Pick a high-contrast ink for text drawn on a styled surface."""

    surface = tuple(int(value) for value in surface_rgb[:3])
    candidates = (
        tuple(int(value) for value in preferred_rgb[:3]),
        (255, 255, 255),
        (250, 252, 255),
        (10, 14, 22),
        (0, 0, 0),
    )
    passing = [
        candidate
        for candidate in candidates
        if float(contrast_ratio(candidate, surface)) >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO)
    ]
    if passing:
        return tuple(int(value) for value in passing[0])
    return max(
        (tuple(int(value) for value in candidate) for candidate in candidates),
        key=lambda candidate: float(contrast_ratio(candidate, surface)),
    )


def calendar_theme_from_information_style(style: PagesInformationStyle) -> TimeArtifactCalendarTheme:
    """Map Pages information-scene roles to the calendar renderer theme contract."""

    marker_fill_rgb = tuple(int(value) for value in style.highlight_rgb)
    marker_outline_rgb = tuple(int(value) for value in style.accent_rgb)
    marker_text_rgb = _readable_text_for_surface(
        marker_fill_rgb,
        preferred_rgb=style.text_rgb,
    )
    return TimeArtifactCalendarTheme(
        accent_color_name=str(style.palette_id),
        style_variant=str(style.treatment),
        surface_mode=("dark" if str(style.treatment).startswith("dark_") else "light"),
        text_color_mode="information_scene",
        panel_fill_rgb=tuple(int(value) for value in style.panel_fill_rgb),
        panel_outline_rgb=tuple(int(value) for value in style.panel_border_rgb),
        title_text_rgb=tuple(int(value) for value in style.text_rgb),
        weekday_fill_rgb=tuple(int(value) for value in style.header_rgb),
        weekday_text_rgb=tuple(int(value) for value in style.header_text_rgb),
        grid_line_rgb=tuple(int(value) for value in style.grid_rgb),
        date_text_rgb=tuple(int(value) for value in style.text_rgb),
        inactive_date_text_rgb=tuple(int(value) for value in style.muted_text_rgb),
        marker_fill_rgb=marker_fill_rgb,
        marker_outline_rgb=marker_outline_rgb,
        marker_text_rgb=marker_text_rgb,
        marker_kind="fill",
    )


def render_calendar_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: CalendarCase,
) -> RenderedCalendarBundle:
    """Render one resolved calendar case and return image plus projection data."""

    render_params = resolve_calendar_render_params(
        params,
        render_defaults=RENDERING_DEFAULTS,
        fallback_values=RENDER_FALLBACKS,
        instance_seed=int(instance_seed),
    )
    information_style, information_style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="calendar",
    )
    calendar_theme = calendar_theme_from_information_style(information_style)
    panel_bbox, panel_layout_meta = resolve_calendar_panel_bbox(
        instance_seed=int(instance_seed),
        params=params,
        render_params=render_params,
        layout_mode=str(case.layout_mode),
    )
    title_text, title_meta = resolve_calendar_title_text(
        instance_seed=int(instance_seed),
        title_mode=str(case.title_mode),
        month=int(case.month),
        year=int(case.year),
        params=params,
    )

    background, background_meta = make_pages_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace="pages.calendar.information_scene_background",
    )
    background_meta = dict(background_meta)
    background_meta["information_scene_style"] = dict(information_style_meta)
    image = background.copy().convert("RGB")
    rendered_scene = render_month_calendar_scene(
        image,
        year=int(case.year),
        month=int(case.month),
        marked_dates=tuple(int(day) for day in case.marked_dates),
        scene_variant=str(case.scene_variant),
        render_params=render_params,
        visual_theme=calendar_theme,
        first_weekday_index=int(case.first_weekday_index),
        panel_bbox_px=panel_bbox,
        title_text=str(title_text),
    )
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=NOISE_DEFAULTS,
    )
    resolved_colors_rgb = {
        "panel_fill": tuple(int(value) for value in calendar_theme.panel_fill_rgb),
        "panel_outline": tuple(int(value) for value in calendar_theme.panel_outline_rgb),
        "title_text": tuple(int(value) for value in calendar_theme.title_text_rgb),
        "weekday_fill": tuple(int(value) for value in calendar_theme.weekday_fill_rgb),
        "weekday_text": tuple(int(value) for value in calendar_theme.weekday_text_rgb),
        "grid_line": tuple(int(value) for value in calendar_theme.grid_line_rgb),
        "date_text": tuple(int(value) for value in calendar_theme.date_text_rgb),
        "inactive_date_text": tuple(int(value) for value in calendar_theme.inactive_date_text_rgb),
        "marker_fill": tuple(int(value) for value in calendar_theme.marker_fill_rgb),
        "marker_outline": tuple(int(value) for value in calendar_theme.marker_outline_rgb),
        "marker_text": tuple(int(value) for value in calendar_theme.marker_text_rgb),
    }
    return RenderedCalendarBundle(
        image=image,
        render_params=render_params,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        title_meta=dict(title_meta),
        panel_layout_meta=dict(panel_layout_meta),
        information_style_meta=dict(information_style_meta),
        resolved_colors_rgb=dict(resolved_colors_rgb),
        marker_kind=str(calendar_theme.marker_kind),
    )
