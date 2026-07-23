"""Shared text/font rendering helpers for task overlays and annotations."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Iterator, Sequence, Tuple

from PIL import ImageDraw, ImageFont

from .geometry_primitives import Point
from .text_legibility import draw_traced_text

Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]
Segment = Tuple[Point, Point]

_FONT_CANDIDATES_BOLD: Sequence[str] = (
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
)
_FONT_CANDIDATES_REGULAR: Sequence[str] = (
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
)
_DEFAULT_FONT_FAMILY: ContextVar[str] = ContextVar("trace_default_font_family", default="")
_SYMBOL_SAFE_CODEPOINTS = frozenset(
    ord(char)
    for char in (
        "∠",
        "θ",
        "β",
        "π",
        "√",
        "−",
        "≤",
        "≥",
        "♭",
        "♯",
        "♮",
    )
)
_MUSIC_ACCIDENTAL_CODEPOINTS = frozenset(ord(char) for char in ("♭", "♯", "♮"))


@contextmanager
def temporary_default_font_family(font_family: str | None) -> Iterator[None]:
    """Temporarily use one shared font family for calls without an explicit family."""

    family = str(font_family or "").strip()
    token = _DEFAULT_FONT_FAMILY.set(family)
    try:
        yield
    finally:
        _DEFAULT_FONT_FAMILY.reset(token)


def current_default_font_family() -> str:
    """Return the current implicit font family, if one has been set."""

    return str(_DEFAULT_FONT_FAMILY.get() or "")


def load_font(size_px: int, *, bold: bool = True, font_family: str | None = None) -> ImageFont.ImageFont:
    """Load a cached TrueType font with robust fallback behavior."""
    size = max(6, int(size_px))
    effective_family = str(font_family or current_default_font_family()).strip()
    return _load_font_cached(int(size), bold=bool(bold), font_family=effective_family)


@lru_cache(maxsize=2048)
def _load_font_cached(size_px: int, *, bold: bool = True, font_family: str = "") -> ImageFont.ImageFont:
    """Load a cached TrueType font for an already-resolved family key."""

    size = max(6, int(size_px))
    if font_family:
        try:
            from .font_assets import resolve_font_paths

            for path in resolve_font_paths(str(font_family), bold=bool(bold)):
                try:
                    return ImageFont.truetype(str(path), size=size)
                except Exception:
                    continue
        except Exception:
            pass
    candidates = _FONT_CANDIDATES_BOLD if bool(bold) else _FONT_CANDIDATES_REGULAR
    for name in candidates:
        try:
            return ImageFont.truetype(str(name), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_needs_symbol_safe_font(text: str) -> bool:
    """Return whether text contains glyphs missing from most readout fonts."""

    return any(ord(char) in _SYMBOL_SAFE_CODEPOINTS for char in str(text))


def _font_is_bold(font: ImageFont.ImageFont) -> bool:
    """Best-effort boldness check for matching symbol fallback style."""

    try:
        _family, style = font.getname()  # type: ignore[attr-defined]
        return "bold" in str(style).casefold()
    except Exception:
        return True


def symbol_safe_font_for_text(text: str, font: ImageFont.ImageFont) -> ImageFont.ImageFont:
    """Return a font that can render Trace math symbols used in readouts.

    The readout font pool is optimized for compact Latin text; most families do
    not include glyphs such as the angle sign.  When one of these symbols is
    present, use a fixed vendored readout-pool family with broad math-symbol
    coverage for that token so bbox calculation and drawing agree.
    """

    if not text_needs_symbol_safe_font(str(text)):
        return font
    size = int(getattr(font, "size", 14))
    if any(ord(char) in _MUSIC_ACCIDENTAL_CODEPOINTS for char in str(text)):
        return _load_font_cached(int(size), bold=_font_is_bold(font), font_family="")
    return _load_font_cached(int(size), bold=_font_is_bold(font), font_family="vollkorn")


def resolve_label_font_size_px(
    *,
    canvas_size: int,
    graph_spacing: int,
    min_px: int = 14,
    max_px: int = 32,
) -> int:
    """Resolve base label font size from canvas and graph scale."""
    size_from_canvas = 0.05 * float(max(1, int(canvas_size)))
    size_from_spacing = 0.55 * float(max(1, int(graph_spacing)))
    resolved = int(round(max(size_from_canvas, size_from_spacing)))
    return max(int(min_px), min(int(max_px), int(resolved)))


def resolve_scene_label_font_size_px(
    *,
    canvas_size: int,
    graph_spacing: int,
    scene_scale: int,
    min_px: int = 14,
    max_px: int = 32,
) -> int:
    """Resolve render-space label font size with scene supersample scale."""
    base = resolve_label_font_size_px(
        canvas_size=int(canvas_size),
        graph_spacing=int(graph_spacing),
        min_px=int(min_px),
        max_px=int(max_px),
    )
    return int(base * max(1, int(scene_scale)))


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    *,
    stroke_width: int = 0,
) -> Tuple[float, float, float, float]:
    """Return the text bounding box for one rendered string.

    The returned box is relative to the draw origin and includes stroke padding
    when requested so callers can center or fit the *rendered* text rather than
    the unstroked glyph box.
    """
    effective_font = symbol_safe_font_for_text(str(text), font)
    try:
        bbox = draw.textbbox((0, 0), str(text), font=effective_font, stroke_width=max(0, int(stroke_width)))
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
        width, height = draw.textsize(str(text), font=effective_font)
        return (0.0, 0.0, float(width), float(height))


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    *,
    stroke_width: int = 0,
) -> Tuple[float, float]:
    """Return text width/height in pixels."""
    try:
        bbox = _text_bbox(draw, str(text), font, stroke_width=max(0, int(stroke_width)))
        return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return float(width), float(height)


def resolve_text_stroke_fill(
    text_fill: Color,
    *,
    light_stroke_fill: Color = (255, 255, 255),
    dark_stroke_fill: Color = (36, 42, 52),
) -> Color:
    """Return one contrast-preserving text outline color for the given fill.

    The helper intentionally keys off the rendered glyph fill rather than the
    surrounding panel color so highlighted white-on-accent text gets a dark
    outline, while darker text on light surfaces keeps the traditional light
    halo.
    """

    r, g, b = (max(0, min(255, int(channel))) for channel in text_fill)
    relative_luminance = (
        (0.2126 * float(r))
        + (0.7152 * float(g))
        + (0.0722 * float(b))
    ) / 255.0
    if float(relative_luminance) >= 0.62:
        return tuple(int(channel) for channel in dark_stroke_fill)
    return tuple(int(channel) for channel in light_stroke_fill)


def fit_font_to_box(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    max_width: float,
    max_height: float,
    bold: bool = True,
    font_family: str | None = None,
    min_size_px: int = 8,
    max_size_px: int | None = None,
    fill_ratio: float = 0.8,
) -> ImageFont.ImageFont:
    """Return the largest cached font that fits one target box.

    This follows the same deterministic, downward-search pattern used in
    the repo's earlier rendering utilities so compact in-figure labels stay readable
    even when label length or glyph shape varies (for example graph labels `A`
    vs `10`).
    """
    allowed_width = max(1.0, float(max_width)) * max(0.1, float(fill_ratio))
    allowed_height = max(1.0, float(max_height)) * max(0.1, float(fill_ratio))
    if max_size_px is None:
        max_size_px = int(max(int(min_size_px), round(min(float(max_width), float(max_height)) * 0.95)))
    for size_px in range(max(int(min_size_px), int(max_size_px)), int(min_size_px) - 1, -1):
        font = load_font(int(size_px), bold=bool(bold), font_family=font_family)
        width, height = _text_size(draw, str(text), font)
        if float(width) <= float(allowed_width) and float(height) <= float(allowed_height):
            return font
    return load_font(int(min_size_px), bold=bool(bold), font_family=font_family)


def _normalize_direction(direction: Point) -> Point:
    """Return one normalized 2D direction, falling back to a stable default."""
    dx, dy = float(direction[0]), float(direction[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        default = 1.0 / math.sqrt(2.0)
        return (float(default), float(-default))
    return (float(dx / norm), float(dy / norm))


def _rotate_direction(direction: Point, angle_degrees: float) -> Point:
    """Return one direction rotated by `angle_degrees` around the origin."""
    rad = math.radians(float(angle_degrees))
    cos_v, sin_v = math.cos(rad), math.sin(rad)
    dx, dy = float(direction[0]), float(direction[1])
    return (float((dx * cos_v) - (dy * sin_v)), float((dx * sin_v) + (dy * cos_v)))


def _point_segment_distance(point: Point, seg_a: Point, seg_b: Point) -> float:
    """Return Euclidean distance from one point to one closed segment."""
    px, py = float(point[0]), float(point[1])
    ax, ay = float(seg_a[0]), float(seg_a[1])
    bx, by = float(seg_b[0]), float(seg_b[1])
    vx, vy = float(bx - ax), float(by - ay)
    wx, wy = float(px - ax), float(py - ay)
    vv = float((vx * vx) + (vy * vy))
    if vv <= 1e-12:
        return float(math.hypot(px - ax, py - ay))
    t = float(((wx * vx) + (wy * vy)) / vv)
    if t <= 0.0:
        closest_x, closest_y = ax, ay
    elif t >= 1.0:
        closest_x, closest_y = bx, by
    else:
        closest_x = float(ax + (t * vx))
        closest_y = float(ay + (t * vy))
    return float(math.hypot(px - closest_x, py - closest_y))


def _bbox_from_center(center: Point, width: float, height: float, padding: float) -> BBox:
    """Return axis-aligned bounding box around centered text plus padding."""
    cx, cy = float(center[0]), float(center[1])
    half_w = 0.5 * float(width)
    half_h = 0.5 * float(height)
    pad = max(0.0, float(padding))
    return (
        float(cx - half_w - pad),
        float(cy - half_h - pad),
        float(cx + half_w + pad),
        float(cy + half_h + pad),
    )


def _bbox_overlaps(a: BBox, b: BBox) -> bool:
    """Return whether two axis-aligned bboxes overlap with non-zero area."""
    return not (
        float(a[2]) <= float(b[0])
        or float(a[0]) >= float(b[2])
        or float(a[3]) <= float(b[1])
        or float(a[1]) >= float(b[3])
    )


def _bbox_within_square_canvas(bbox: BBox, canvas_size: int, margin: float = 1.0) -> bool:
    """Return whether one bbox stays inside a square canvas with margin."""
    side = float(max(1, int(canvas_size)))
    m = max(0.0, float(margin))
    return (
        float(bbox[0]) >= m
        and float(bbox[1]) >= m
        and float(bbox[2]) <= (side - m)
        and float(bbox[3]) <= (side - m)
    )


def resolve_text_label_center(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    anchor: Point,
    base_direction: Point,
    offset_px: float,
    font: ImageFont.ImageFont,
    blocked_segments: Sequence[Segment] | None = None,
    blocked_points: Sequence[Point] | None = None,
    occupied_boxes: Sequence[BBox] | None = None,
    stroke_width: int | None = None,
    line_clearance_px: float = 2.0,
    point_clearance_px: float = 6.0,
    canvas_size: int | None = None,
) -> Tuple[Point, BBox]:
    """Choose one nearby text center that minimizes overlap with line segments.

    Determinism note:
    - Candidate directions/radii are fixed and iterated in stable order.
    - No randomness is used; same inputs always return the same center.
    """
    width, height = _text_size(draw, str(text), font)
    size_hint = int(getattr(font, "size", 14))
    outline = int(stroke_width) if stroke_width is not None else max(1, int(round(0.08 * float(size_hint))))
    padding = float(max(1, int(outline)))
    base = _normalize_direction((float(base_direction[0]), float(base_direction[1])))
    anchor_x, anchor_y = float(anchor[0]), float(anchor[1])
    offset = float(max(4.0, float(offset_px)))
    min_required = float(0.5 * math.hypot(width, height) + max(1.0, float(line_clearance_px)))
    segments = list(blocked_segments or ())
    blocked_point_boxes = [
        _bbox_from_center(
            (float(point[0]), float(point[1])),
            width=0.0,
            height=0.0,
            padding=max(0.0, float(point_clearance_px)),
        )
        for point in list(blocked_points or ())
    ]
    occupied = list(occupied_boxes or ())

    angle_offsets = (0, 20, -20, 35, -35, 50, -50, 70, -70, 90, -90, 120, -120, 150, -150, 180)
    radius_scales = (1.0, 1.3, 1.6, 2.0, 2.4)
    best: Tuple[Tuple[float, ...], Point, BBox] | None = None

    for radius_scale in radius_scales:
        radius = float(offset * float(radius_scale))
        for angle in angle_offsets:
            direction = _rotate_direction(base, float(angle))
            center = (
                float(anchor_x + (radius * float(direction[0]))),
                float(anchor_y + (radius * float(direction[1]))),
            )
            bbox = _bbox_from_center(center, float(width), float(height), float(padding))
            out_of_bounds = 0
            if canvas_size is not None and not _bbox_within_square_canvas(bbox, int(canvas_size), margin=1.0):
                out_of_bounds = 1
            point_overlap = 1 if any(_bbox_overlaps(bbox, point_bbox) for point_bbox in blocked_point_boxes) else 0
            label_overlap = 1 if any(_bbox_overlaps(bbox, existing) for existing in occupied) else 0
            if segments:
                min_distance = min(
                    _point_segment_distance(center, (float(seg_a[0]), float(seg_a[1])), (float(seg_b[0]), float(seg_b[1])))
                    for seg_a, seg_b in segments
                )
            else:
                min_distance = float("inf")
            clearance_deficit = float(max(0.0, min_required - float(min_distance)))
            line_overlap = 1 if float(clearance_deficit) > 1e-6 else 0
            score = (
                float(out_of_bounds),
                float(line_overlap),
                float(point_overlap),
                float(label_overlap),
                float(clearance_deficit),
                float(radius),
                float(abs(angle)),
            )
            if best is None or score < best[0]:
                best = (score, center, bbox)

    if best is None:
        center = (float(anchor_x + (offset * base[0])), float(anchor_y + (offset * base[1])))
        return center, _bbox_from_center(center, float(width), float(height), float(padding))
    return best[1], best[2]


def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: ImageFont.ImageFont,
    fill: Color = (30, 30, 30),
    stroke_fill: Color = (255, 255, 255),
    stroke_width: int | None = None,
    role: str = "visible_text",
    required: bool = False,
    trace: bool = True,
) -> BBox:
    """Draw text centered around a point with optional outline stroke.

    PIL text bboxes often have non-zero top/left offsets relative to the draw
    origin.  Center using the full rendered bbox, not just width/height, so
    glyphs stay visually centered inside badges, circular markers, and option
    chips across fonts.
    """
    effective_font = symbol_safe_font_for_text(str(text), font)
    size_hint = int(getattr(effective_font, "size", 14))
    outline_width = int(stroke_width) if stroke_width is not None else max(1, int(round(0.08 * float(size_hint))))
    bbox = _text_bbox(draw, str(text), effective_font, stroke_width=max(0, int(outline_width)))
    center_x, center_y = float(center[0]), float(center[1])
    tx = center_x - (0.5 * float(bbox[0] + bbox[2]))
    ty = center_y - (0.5 * float(bbox[1] + bbox[3]))
    rendered_bbox = (
        float(tx + bbox[0]),
        float(ty + bbox[1]),
        float(tx + bbox[2]),
        float(ty + bbox[3]),
    )
    if bool(trace):
        draw_traced_text(
            draw,
            xy=(float(tx), float(ty)),
            text=str(text),
            font=effective_font,
            fill_rgb=tuple(int(value) for value in fill),
            stroke_rgb=tuple(int(value) for value in stroke_fill),
            stroke_width=max(0, int(outline_width)),
            role=str(role),
            required=bool(required),
        )
    else:
        draw.text(
            (float(tx), float(ty)),
            str(text),
            font=effective_font,
            fill=tuple(int(value) for value in fill),
            stroke_width=max(0, int(outline_width)),
            stroke_fill=tuple(int(value) for value in stroke_fill),
        )
    return rendered_bbox
