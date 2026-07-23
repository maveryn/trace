"""Shared time-artifact visual-theme helpers for clocks, calendars, schedules, and timelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence, Tuple

from .named_colors import available_named_colors, darken_color, named_color
from .text_legibility import READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO, contrast_ratio


Color = Tuple[int, int, int]
SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS: Tuple[str, ...] = (
    "studio",
    "accented",
    "marker",
)
SUPPORTED_TIME_ARTIFACT_COLOR_NAMES: Tuple[str, ...] = tuple(str(name) for name, _ in available_named_colors())
SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS: Tuple[str, ...] = SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS
SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES: Tuple[str, ...] = SUPPORTED_TIME_ARTIFACT_COLOR_NAMES
SUPPORTED_TIME_ARTIFACT_CALENDAR_TEXT_COLOR_MODES: Tuple[str, ...] = (
    "neutral",
    "accent",
    "cool",
    "warm",
)
_CALENDAR_DARK_SURFACE_INK_POOLS: dict[str, Tuple[Color, ...]] = {
    "neutral": (
        (150, 203, 255),
        (152, 232, 212),
        (213, 181, 255),
        (255, 210, 120),
        (255, 166, 194),
    ),
    "accent": (
        (141, 224, 255),
        (164, 238, 192),
        (255, 210, 105),
        (255, 172, 142),
        (224, 176, 255),
    ),
    "cool": (
        (128, 232, 255),
        (150, 203, 255),
        (146, 244, 224),
        (155, 238, 178),
        (186, 200, 255),
    ),
    "warm": (
        (255, 210, 105),
        (255, 188, 132),
        (255, 164, 194),
        (255, 181, 153),
        (236, 190, 255),
    ),
}
_CALENDAR_LIGHT_SURFACE_INK_POOLS: dict[str, Tuple[Color, ...]] = {
    "neutral": (
        (13, 52, 116),
        (0, 82, 91),
        (86, 34, 122),
        (112, 57, 12),
        (20, 83, 45),
    ),
    "accent": (
        (112, 20, 48),
        (18, 74, 135),
        (0, 86, 83),
        (95, 42, 132),
        (111, 58, 0),
    ),
    "cool": (
        (13, 52, 116),
        (0, 82, 91),
        (20, 83, 112),
        (47, 44, 130),
        (14, 78, 71),
    ),
    "warm": (
        (112, 20, 48),
        (106, 45, 31),
        (112, 57, 12),
        (126, 34, 81),
        (98, 62, 0),
    ),
}


@dataclass(frozen=True)
class TimeArtifactClockTheme:
    """Resolved per-instance analog-clock theme derived from one named accent color."""

    accent_color_name: str
    style_variant: str
    face_fill_rgb: Color
    face_outline_rgb: Color
    numeral_color_rgb: Color
    tick_color_rgb: Color
    hour_hand_color_rgb: Color
    minute_hand_color_rgb: Color
    second_hand_color_rgb: Color
    center_dot_color_rgb: Color
    inner_ring_rgb: Color | None
    minor_tick_mode: str


@dataclass(frozen=True)
class TimeArtifactCalendarTheme:
    """Resolved per-instance calendar theme derived from one named accent color."""

    accent_color_name: str
    style_variant: str
    surface_mode: str
    text_color_mode: str
    panel_fill_rgb: Color
    panel_outline_rgb: Color
    title_text_rgb: Color
    weekday_fill_rgb: Color
    weekday_text_rgb: Color
    grid_line_rgb: Color
    date_text_rgb: Color
    inactive_date_text_rgb: Color
    marker_fill_rgb: Color
    marker_outline_rgb: Color
    marker_text_rgb: Color
    marker_kind: str


@dataclass(frozen=True)
class TimeArtifactScheduleTheme:
    """Resolved per-instance day-planner theme derived from one named accent color."""

    accent_color_name: str
    style_variant: str
    panel_fill_rgb: Color
    panel_outline_rgb: Color
    header_fill_rgb: Color
    header_text_rgb: Color
    grid_line_rgb: Color
    minor_grid_line_rgb: Color
    time_text_rgb: Color
    event_fill_rgb: Color
    event_outline_rgb: Color
    event_text_rgb: Color
    reference_fill_rgb: Color
    reference_outline_rgb: Color
    reference_text_rgb: Color
    header_kind: str


@dataclass(frozen=True)
class TimeArtifactTimelineTheme:
    """Resolved per-instance milestone-timeline theme derived from one named accent color."""

    accent_color_name: str
    style_variant: str
    panel_fill_rgb: Color
    panel_outline_rgb: Color
    title_text_rgb: Color
    subtitle_text_rgb: Color
    axis_line_rgb: Color
    tick_line_rgb: Color
    connector_line_rgb: Color
    marker_fill_rgb: Color
    marker_outline_rgb: Color
    event_fill_rgb: Color
    event_outline_rgb: Color
    event_text_rgb: Color
    event_subtext_rgb: Color
    primary_reference_fill_rgb: Color
    primary_reference_outline_rgb: Color
    primary_reference_text_rgb: Color
    secondary_reference_fill_rgb: Color
    secondary_reference_outline_rgb: Color
    secondary_reference_text_rgb: Color


def _blend_with_white(color: Sequence[int], *, color_weight: float) -> Color:
    """Blend one RGB color toward white by the requested color weight."""

    weight = max(0.0, min(1.0, float(color_weight)))
    if len(color) < 3:
        raise ValueError("time-artifact color blends require three RGB channels")
    return tuple(
        max(0, min(255, int(round((255.0 * (1.0 - weight)) + (float(int(channel)) * weight)))))
        for channel in color[:3]
    )


def _blend_between(base: Sequence[int], overlay: Sequence[int], *, overlay_weight: float) -> Color:
    """Blend one RGB overlay onto a base color."""

    weight = max(0.0, min(1.0, float(overlay_weight)))
    if len(base) < 3 or len(overlay) < 3:
        raise ValueError("time-artifact color blends require three RGB channels")
    return tuple(
        max(0, min(255, int(round((float(int(base_channel)) * (1.0 - weight)) + (float(int(overlay_channel)) * weight)))))
        for base_channel, overlay_channel in zip(base[:3], overlay[:3])
    )


def _relative_luminance(color: Sequence[int]) -> float:
    """Return one simple perceived-luminance estimate in ``[0, 1]``."""

    if len(color) < 3:
        raise ValueError("time-artifact luminance requires three RGB channels")
    red, green, blue = [float(int(channel)) / 255.0 for channel in color[:3]]
    return float((0.2126 * red) + (0.7152 * green) + (0.0722 * blue))


def _rgb_saturation(color: Sequence[int]) -> float:
    """Return HSV-style saturation without pulling in a color utility dependency."""

    if len(color) < 3:
        raise ValueError("time-artifact saturation requires three RGB channels")
    channels = [max(0.0, min(1.0, float(int(channel)) / 255.0)) for channel in color[:3]]
    high = max(channels)
    low = min(channels)
    if high <= 0.0:
        return 0.0
    return float((high - low) / high)


def _contrast_safe_text_color(preferred: Sequence[int], surface: Sequence[int], *, fallback: Color) -> Color:
    """Return the preferred text color if it satisfies required contrast."""

    preferred_rgb = tuple(int(channel) for channel in preferred[:3])
    surface_rgb = tuple(int(channel) for channel in surface[:3])
    if float(contrast_ratio(preferred_rgb, surface_rgb)) >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO):
        return preferred_rgb
    fallback_rgb = tuple(int(channel) for channel in fallback[:3])
    if float(contrast_ratio(fallback_rgb, surface_rgb)) >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO):
        return fallback_rgb
    black_rgb: Color = (0, 0, 0)
    white_rgb: Color = (255, 255, 255)
    return black_rgb if contrast_ratio(black_rgb, surface_rgb) >= contrast_ratio(white_rgb, surface_rgb) else white_rgb


def _stable_choice_index(*parts: object, size: int) -> int:
    """Return a deterministic index for one small candidate pool."""

    if int(size) <= 0:
        raise ValueError("stable choice requires a positive size")
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % int(size)


def _calendar_accent_ink_candidates(accent_rgb: Color, *, surface_mode: str) -> Tuple[Color, ...]:
    """Return accent-derived chromatic text candidates for the active surface."""

    if str(surface_mode) == "dark":
        return (
            _blend_with_white(accent_rgb, color_weight=0.40),
            _blend_with_white(accent_rgb, color_weight=0.34),
            _blend_with_white(accent_rgb, color_weight=0.46),
        )
    return (
        darken_color(accent_rgb, factor=0.38),
        darken_color(accent_rgb, factor=0.32),
        darken_color(accent_rgb, factor=0.44),
    )


def _calendar_text_ink_pool(
    *,
    surface_mode: str,
    text_mode: str,
    accent_rgb: Color,
) -> Tuple[Color, ...]:
    """Build a contrast-filtered candidate pool before per-surface filtering."""

    pool_by_mode = _CALENDAR_DARK_SURFACE_INK_POOLS if str(surface_mode) == "dark" else _CALENDAR_LIGHT_SURFACE_INK_POOLS
    base_pool = tuple(pool_by_mode.get(str(text_mode), pool_by_mode["neutral"]))
    accent_pool = _calendar_accent_ink_candidates(accent_rgb, surface_mode=str(surface_mode))
    if str(text_mode) == "accent":
        return (*accent_pool, *base_pool)
    return (*base_pool, *accent_pool[:1])


def _resolve_calendar_text_ink(
    *,
    role: str,
    surface_rgb: Color,
    surface_mode: str,
    text_mode: str,
    accent_rgb: Color,
    style_variant: str,
    fallback: Color,
) -> Color:
    """Choose one visibly chromatic calendar text color that passes contrast."""

    surface = tuple(int(channel) for channel in surface_rgb)
    candidates = _calendar_text_ink_pool(
        surface_mode=str(surface_mode),
        text_mode=str(text_mode),
        accent_rgb=tuple(int(channel) for channel in accent_rgb),
    )
    eligible = [
        tuple(int(channel) for channel in candidate)
        for candidate in candidates
        if float(contrast_ratio(candidate, surface)) >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO)
        and float(_rgb_saturation(candidate)) >= 0.20
    ]
    if eligible:
        selected_index = _stable_choice_index(
            role,
            surface_mode,
            text_mode,
            style_variant,
            accent_rgb,
            surface,
            size=len(eligible),
        )
        return tuple(int(channel) for channel in eligible[int(selected_index)])
    return _contrast_safe_text_color(fallback, surface, fallback=fallback)


def build_time_artifact_clock_theme(accent_color_name: str, style_variant: str) -> TimeArtifactClockTheme:
    """Resolve one readable analog-clock theme from a named accent color and style."""

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.58)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    neutral_dark_rgb = (42, 48, 58)
    subtle_outline_rgb = _blend_with_white(accent_deep_rgb, color_weight=0.42)
    face_fill_rgb = (255, 255, 255)
    numeral_color_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    tick_color_rgb = tuple(int(channel) for channel in accent_dark_rgb)
    hour_hand_color_rgb = tuple(int(channel) for channel in neutral_dark_rgb)
    minute_hand_color_rgb = tuple(int(channel) for channel in accent_rgb)
    second_hand_color_rgb: Color = (215, 52, 52)
    center_dot_color_rgb = tuple(int(channel) for channel in accent_rgb)
    inner_ring_rgb: Color | None = None
    minor_tick_mode = "line"

    variant = str(style_variant)
    if variant == "accented":
        face_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.10)
        numeral_color_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        tick_color_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        hour_hand_color_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        minute_hand_color_rgb = tuple(int(channel) for channel in accent_rgb)
        second_hand_color_rgb = (205, 48, 48)
        center_dot_color_rgb = tuple(int(channel) for channel in accent_rgb)
        inner_ring_rgb = _blend_with_white(accent_rgb, color_weight=0.56)
        face_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
    elif variant == "marker":
        face_fill_rgb = (255, 255, 255)
        numeral_color_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        tick_color_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        hour_hand_color_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        minute_hand_color_rgb = tuple(int(channel) for channel in accent_rgb)
        second_hand_color_rgb = (205, 48, 48)
        center_dot_color_rgb = tuple(int(channel) for channel in accent_rgb)
        inner_ring_rgb = _blend_with_white(accent_rgb, color_weight=0.30)
        face_outline_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        minor_tick_mode = "dot"
    else:
        face_outline_rgb = tuple(int(channel) for channel in subtle_outline_rgb)
        inner_ring_rgb = _blend_with_white(accent_rgb, color_weight=0.18)

    if _relative_luminance(face_fill_rgb) <= 0.45:
        numeral_color_rgb = (255, 255, 255)

    return TimeArtifactClockTheme(
        accent_color_name=str(accent_color_name),
        style_variant=variant,
        face_fill_rgb=tuple(int(channel) for channel in face_fill_rgb),
        face_outline_rgb=tuple(int(channel) for channel in face_outline_rgb),
        numeral_color_rgb=tuple(int(channel) for channel in numeral_color_rgb),
        tick_color_rgb=tuple(int(channel) for channel in tick_color_rgb),
        hour_hand_color_rgb=tuple(int(channel) for channel in hour_hand_color_rgb),
        minute_hand_color_rgb=tuple(int(channel) for channel in minute_hand_color_rgb),
        second_hand_color_rgb=tuple(int(channel) for channel in second_hand_color_rgb),
        center_dot_color_rgb=tuple(int(channel) for channel in center_dot_color_rgb),
        inner_ring_rgb=(tuple(int(channel) for channel in inner_ring_rgb) if inner_ring_rgb is not None else None),
        minor_tick_mode=str(minor_tick_mode),
    )


def build_time_artifact_calendar_theme(
    accent_color_name: str,
    style_variant: str,
    surface_mode: str = "light",
    text_color_mode: str = "neutral",
) -> TimeArtifactCalendarTheme:
    """Resolve one readable month-calendar theme from a named accent color and style."""

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.58)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    neutral_dark_rgb = (46, 52, 62)

    variant = str(style_variant)
    surface = str(surface_mode).strip().lower() or "light"
    if surface not in {"light", "dark"}:
        raise ValueError(f"unsupported calendar surface_mode: {surface_mode!r}")
    text_mode = str(text_color_mode).strip().lower() or "neutral"
    if text_mode not in set(SUPPORTED_TIME_ARTIFACT_CALENDAR_TEXT_COLOR_MODES):
        raise ValueError(f"unsupported calendar text_color_mode: {text_color_mode!r}")
    panel_fill_rgb = (255, 255, 255)
    panel_outline_rgb = _blend_with_white(accent_deep_rgb, color_weight=0.30)
    title_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    weekday_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.14)
    weekday_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    grid_line_rgb = _blend_with_white(accent_deep_rgb, color_weight=0.18)
    date_text_rgb = tuple(int(channel) for channel in neutral_dark_rgb)
    inactive_date_text_rgb = (170, 176, 186)
    marker_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.28)
    marker_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
    marker_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    marker_kind = "fill"

    if variant == "accented":
        panel_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.06)
        panel_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        weekday_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.22)
        weekday_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        grid_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.24)
        marker_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.44)
        marker_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        marker_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    elif variant == "marker":
        panel_fill_rgb = (255, 255, 255)
        panel_outline_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        title_text_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        weekday_fill_rgb = (255, 255, 255)
        weekday_text_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        grid_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.28)
        marker_fill_rgb = (255, 255, 255)
        marker_outline_rgb = tuple(int(channel) for channel in accent_rgb)
        marker_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        marker_kind = "ring"

    if surface == "dark":
        dark_panel_rgb: Color = (24, 30, 42)
        dark_panel_alt_rgb: Color = (34, 43, 60)
        title_text_rgb = (244, 248, 252)
        weekday_text_rgb = (228, 236, 246)
        date_text_rgb = (242, 246, 251)
        inactive_date_text_rgb = (130, 143, 162)
        panel_fill_rgb = dark_panel_rgb
        panel_outline_rgb = _blend_with_white(accent_rgb, color_weight=0.62)
        weekday_fill_rgb = dark_panel_alt_rgb
        grid_line_rgb = (75, 89, 114)
        marker_fill_rgb = darken_color(accent_rgb, factor=0.34)
        marker_outline_rgb = _blend_with_white(accent_rgb, color_weight=0.76)
        marker_text_rgb = (255, 255, 255)
        marker_kind = "fill"
        if variant == "accented":
            panel_fill_rgb = _blend_between(dark_panel_rgb, accent_deep_rgb, overlay_weight=0.12)
            panel_outline_rgb = _blend_with_white(accent_rgb, color_weight=0.74)
            weekday_fill_rgb = _blend_between(dark_panel_alt_rgb, accent_deep_rgb, overlay_weight=0.18)
            grid_line_rgb = _blend_between((72, 86, 110), accent_deep_rgb, overlay_weight=0.18)
        elif variant == "marker":
            panel_fill_rgb = dark_panel_rgb
            panel_outline_rgb = _blend_with_white(accent_rgb, color_weight=0.70)
            weekday_fill_rgb = dark_panel_rgb
            weekday_text_rgb = (235, 241, 249)
            grid_line_rgb = (88, 101, 126)
            marker_fill_rgb = panel_fill_rgb
            marker_outline_rgb = _blend_with_white(accent_rgb, color_weight=0.84)
            marker_text_rgb = date_text_rgb
            marker_kind = "ring"

    text_fallback_rgb: Color = (20, 26, 34) if surface == "light" else (255, 255, 255)
    marker_surface_rgb = marker_fill_rgb if str(marker_kind) == "fill" else panel_fill_rgb
    title_text_rgb = _resolve_calendar_text_ink(
        role="title",
        surface_rgb=panel_fill_rgb,
        surface_mode=surface,
        text_mode=text_mode,
        accent_rgb=accent_rgb,
        style_variant=variant,
        fallback=text_fallback_rgb,
    )
    date_text_rgb = _resolve_calendar_text_ink(
        role="date",
        surface_rgb=panel_fill_rgb,
        surface_mode=surface,
        text_mode=text_mode,
        accent_rgb=accent_rgb,
        style_variant=variant,
        fallback=text_fallback_rgb,
    )
    weekday_text_rgb = _resolve_calendar_text_ink(
        role="weekday",
        surface_rgb=weekday_fill_rgb,
        surface_mode=surface,
        text_mode=text_mode,
        accent_rgb=accent_rgb,
        style_variant=variant,
        fallback=text_fallback_rgb,
    )
    marker_text_rgb = _resolve_calendar_text_ink(
        role="marker",
        surface_rgb=marker_surface_rgb,
        surface_mode=surface,
        text_mode=text_mode,
        accent_rgb=accent_rgb,
        style_variant=variant,
        fallback=text_fallback_rgb,
    )

    return TimeArtifactCalendarTheme(
        accent_color_name=str(accent_color_name),
        style_variant=variant,
        surface_mode=str(surface),
        text_color_mode=str(text_mode),
        panel_fill_rgb=tuple(int(channel) for channel in panel_fill_rgb),
        panel_outline_rgb=tuple(int(channel) for channel in panel_outline_rgb),
        title_text_rgb=tuple(int(channel) for channel in title_text_rgb),
        weekday_fill_rgb=tuple(int(channel) for channel in weekday_fill_rgb),
        weekday_text_rgb=tuple(int(channel) for channel in weekday_text_rgb),
        grid_line_rgb=tuple(int(channel) for channel in grid_line_rgb),
        date_text_rgb=tuple(int(channel) for channel in date_text_rgb),
        inactive_date_text_rgb=tuple(int(channel) for channel in inactive_date_text_rgb),
        marker_fill_rgb=tuple(int(channel) for channel in marker_fill_rgb),
        marker_outline_rgb=tuple(int(channel) for channel in marker_outline_rgb),
        marker_text_rgb=tuple(int(channel) for channel in marker_text_rgb),
        marker_kind=str(marker_kind),
    )


def build_time_artifact_schedule_theme(accent_color_name: str, style_variant: str) -> TimeArtifactScheduleTheme:
    """Resolve one readable day-planner theme from a named accent color and style."""

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.58)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    neutral_dark_rgb = (46, 52, 62)

    variant = str(style_variant)
    panel_fill_rgb = (255, 255, 255)
    panel_outline_rgb = _blend_with_white(accent_deep_rgb, color_weight=0.30)
    header_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.14)
    header_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    grid_line_rgb = _blend_with_white(accent_deep_rgb, color_weight=0.18)
    minor_grid_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.10)
    time_text_rgb = tuple(int(channel) for channel in neutral_dark_rgb)
    event_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.10)
    event_outline_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.55)
    event_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    reference_fill_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    reference_outline_rgb = tuple(int(channel) for channel in neutral_dark_rgb)
    reference_text_rgb = (255, 255, 255)
    header_kind = "fill"

    if variant == "accented":
        panel_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.06)
        panel_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        header_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.26)
        header_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        grid_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.22)
        minor_grid_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.14)
        event_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.14)
        event_outline_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.62)
        event_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        reference_fill_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        reference_outline_rgb = tuple(int(channel) for channel in neutral_dark_rgb)
        reference_text_rgb = (255, 255, 255)
    elif variant == "marker":
        panel_fill_rgb = (255, 255, 255)
        panel_outline_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        header_fill_rgb = (255, 255, 255)
        header_text_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        grid_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.22)
        minor_grid_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.12)
        time_text_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        event_fill_rgb = (255, 255, 255)
        event_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        event_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        reference_fill_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        reference_outline_rgb = tuple(int(channel) for channel in neutral_dark_rgb)
        reference_text_rgb = (255, 255, 255)
        header_kind = "line"

    return TimeArtifactScheduleTheme(
        accent_color_name=str(accent_color_name),
        style_variant=variant,
        panel_fill_rgb=tuple(int(channel) for channel in panel_fill_rgb),
        panel_outline_rgb=tuple(int(channel) for channel in panel_outline_rgb),
        header_fill_rgb=tuple(int(channel) for channel in header_fill_rgb),
        header_text_rgb=tuple(int(channel) for channel in header_text_rgb),
        grid_line_rgb=tuple(int(channel) for channel in grid_line_rgb),
        minor_grid_line_rgb=tuple(int(channel) for channel in minor_grid_line_rgb),
        time_text_rgb=tuple(int(channel) for channel in time_text_rgb),
        event_fill_rgb=tuple(int(channel) for channel in event_fill_rgb),
        event_outline_rgb=tuple(int(channel) for channel in event_outline_rgb),
        event_text_rgb=tuple(int(channel) for channel in event_text_rgb),
        reference_fill_rgb=tuple(int(channel) for channel in reference_fill_rgb),
        reference_outline_rgb=tuple(int(channel) for channel in reference_outline_rgb),
        reference_text_rgb=tuple(int(channel) for channel in reference_text_rgb),
        header_kind=str(header_kind),
    )


def build_time_artifact_timeline_theme(accent_color_name: str, style_variant: str) -> TimeArtifactTimelineTheme:
    """Resolve one readable milestone-timeline theme from a named accent color and style."""

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.58)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    neutral_dark_rgb = (46, 52, 62)
    soft_gray_rgb = (124, 132, 144)

    variant = str(style_variant)
    panel_fill_rgb = (255, 255, 255)
    panel_outline_rgb = _blend_with_white(accent_deep_rgb, color_weight=0.30)
    title_text_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    subtitle_text_rgb = tuple(int(channel) for channel in soft_gray_rgb)
    axis_line_rgb = _blend_with_white(accent_deep_rgb, color_weight=0.22)
    tick_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.18)
    connector_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.22)
    marker_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.28)
    marker_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
    event_fill_rgb = (255, 255, 255)
    event_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
    event_text_rgb = tuple(int(channel) for channel in neutral_dark_rgb)
    event_subtext_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    reference_fill_rgb = tuple(int(channel) for channel in accent_rgb)
    reference_outline_rgb = tuple(int(channel) for channel in accent_deep_rgb)
    reference_text_rgb: Color = (255, 255, 255) if _relative_luminance(reference_fill_rgb) < 0.60 else tuple(int(channel) for channel in neutral_dark_rgb)

    if variant == "accented":
        panel_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.07)
        panel_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        axis_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.30)
        tick_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.24)
        connector_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.26)
        marker_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.38)
        marker_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        event_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.18)
        event_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        event_subtext_rgb = tuple(int(channel) for channel in accent_dark_rgb)
    elif variant == "marker":
        panel_fill_rgb = (255, 255, 255)
        panel_outline_rgb = tuple(int(channel) for channel in accent_deep_rgb)
        title_text_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        axis_line_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        tick_line_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.18)
        connector_line_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        marker_fill_rgb = (255, 255, 255)
        marker_outline_rgb = tuple(int(channel) for channel in accent_rgb)
        event_fill_rgb = (255, 255, 255)
        event_outline_rgb = tuple(int(channel) for channel in accent_dark_rgb)
        event_subtext_rgb = tuple(int(channel) for channel in accent_dark_rgb)

    return TimeArtifactTimelineTheme(
        accent_color_name=str(accent_color_name),
        style_variant=variant,
        panel_fill_rgb=tuple(int(channel) for channel in panel_fill_rgb),
        panel_outline_rgb=tuple(int(channel) for channel in panel_outline_rgb),
        title_text_rgb=tuple(int(channel) for channel in title_text_rgb),
        subtitle_text_rgb=tuple(int(channel) for channel in subtitle_text_rgb),
        axis_line_rgb=tuple(int(channel) for channel in axis_line_rgb),
        tick_line_rgb=tuple(int(channel) for channel in tick_line_rgb),
        connector_line_rgb=tuple(int(channel) for channel in connector_line_rgb),
        marker_fill_rgb=tuple(int(channel) for channel in marker_fill_rgb),
        marker_outline_rgb=tuple(int(channel) for channel in marker_outline_rgb),
        event_fill_rgb=tuple(int(channel) for channel in event_fill_rgb),
        event_outline_rgb=tuple(int(channel) for channel in event_outline_rgb),
        event_text_rgb=tuple(int(channel) for channel in event_text_rgb),
        event_subtext_rgb=tuple(int(channel) for channel in event_subtext_rgb),
        primary_reference_fill_rgb=tuple(int(channel) for channel in reference_fill_rgb),
        primary_reference_outline_rgb=tuple(int(channel) for channel in reference_outline_rgb),
        primary_reference_text_rgb=tuple(int(channel) for channel in reference_text_rgb),
        secondary_reference_fill_rgb=tuple(int(channel) for channel in reference_fill_rgb),
        secondary_reference_outline_rgb=tuple(int(channel) for channel in reference_outline_rgb),
        secondary_reference_text_rgb=tuple(int(channel) for channel in reference_text_rgb),
    )


__all__ = [
    "SUPPORTED_TIME_ARTIFACT_COLOR_NAMES",
    "SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS",
    "SUPPORTED_TIME_ARTIFACT_CALENDAR_TEXT_COLOR_MODES",
    "SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES",
    "SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS",
    "TimeArtifactCalendarTheme",
    "TimeArtifactClockTheme",
    "TimeArtifactScheduleTheme",
    "TimeArtifactTimelineTheme",
    "build_time_artifact_calendar_theme",
    "build_time_artifact_clock_theme",
    "build_time_artifact_schedule_theme",
    "build_time_artifact_timeline_theme",
]
