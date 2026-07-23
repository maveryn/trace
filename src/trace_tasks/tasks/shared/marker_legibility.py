"""Shared semantic marker contrast utilities.

Semantic markers are visual annotations that the prompt asks the solver to use:
outlined cells, highlighted regions, marked paths, selected objects, target
points, and similar answer-bearing cues. They need the same kind of explicit
visibility contract as read-off text because theme colors alone can blend into
the marked surface.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence, Tuple

from PIL import ImageDraw

from ...core.seed import spawn_rng
from .color_distance import VISIBILITY_SAFE_FALLBACK_COLORS, color_distance
from .text_legibility import contrast_ratio, normalize_rgb

Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]

SEMANTIC_MARKER_MIN_CONTRAST_RATIO = 3.0
SEMANTIC_MARKER_MIN_LAB_DISTANCE = 40.0
MARKER_LEGIBILITY_POLICY_VERSION = "marker_legibility_v1"

SEMANTIC_MARKER_COLOR_POOL: Tuple[Color, ...] = (
    (255, 255, 255),
    (10, 14, 22),
    (255, 214, 74),
    (34, 211, 238),
    (59, 130, 246),
    (168, 85, 247),
    (236, 72, 153),
    (239, 68, 68),
    (249, 115, 22),
    (34, 197, 94),
    *VISIBILITY_SAFE_FALLBACK_COLORS,
)

_ACTIVE_MARKER_RECORDS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "trace_active_semantic_marker_records",
    default=None,
)


@dataclass(frozen=True)
class SemanticMarkerStyle:
    """Resolved marker style and contrast metadata for one semantic mark."""

    role: str
    inner_rgb: Color
    outer_rgb: Color
    surface_rgbs: Tuple[Color, ...]
    min_inner_contrast_ratio: float
    min_outer_contrast_ratio: float
    min_effective_contrast_ratio: float
    min_inner_lab_distance: float
    min_outer_lab_distance: float
    min_effective_lab_distance: float
    min_contrast_required: float = SEMANTIC_MARKER_MIN_CONTRAST_RATIO
    min_lab_distance_required: float = SEMANTIC_MARKER_MIN_LAB_DISTANCE
    required: bool = True

    @property
    def passes(self) -> bool:
        if not bool(self.required):
            return True
        return (
            float(self.min_effective_contrast_ratio) >= float(self.min_contrast_required)
            and float(self.min_effective_lab_distance) >= float(self.min_lab_distance_required)
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "required": bool(self.required),
            "inner_rgb": list(self.inner_rgb),
            "outer_rgb": list(self.outer_rgb),
            "surface_rgbs": [list(color) for color in self.surface_rgbs],
            "min_inner_contrast_ratio": round(float(self.min_inner_contrast_ratio), 3),
            "min_outer_contrast_ratio": round(float(self.min_outer_contrast_ratio), 3),
            "min_effective_contrast_ratio": round(float(self.min_effective_contrast_ratio), 3),
            "min_inner_lab_distance": round(float(self.min_inner_lab_distance), 3),
            "min_outer_lab_distance": round(float(self.min_outer_lab_distance), 3),
            "min_effective_lab_distance": round(float(self.min_effective_lab_distance), 3),
            "min_contrast_required": round(float(self.min_contrast_required), 3),
            "min_lab_distance_required": round(float(self.min_lab_distance_required), 3),
            "passes": bool(self.passes),
        }


def _unique_colors(colors: Iterable[Sequence[int]]) -> Tuple[Color, ...]:
    seen: set[Color] = set()
    out: list[Color] = []
    for color in colors:
        if len(color) < 3:
            continue
        rgb = normalize_rgb(color)
        if rgb in seen:
            continue
        seen.add(rgb)
        out.append(rgb)
    return tuple(out)


def _min_contrast(color: Color, surfaces: Sequence[Color]) -> float:
    if not surfaces:
        return float("inf")
    return min(float(contrast_ratio(color, surface)) for surface in surfaces)


def _min_lab_distance(color: Color, surfaces: Sequence[Color]) -> float:
    if not surfaces:
        return float("inf")
    return min(float(color_distance(color, surface, distance_space="lab")) for surface in surfaces)


def _choose_outer(inner_rgb: Color, surfaces: Sequence[Color]) -> Color:
    candidates = ((255, 255, 255), (10, 14, 22), (0, 0, 0))

    def score(candidate: Color) -> tuple[float, float, float]:
        surface_contrast = _min_contrast(candidate, surfaces)
        surface_lab = _min_lab_distance(candidate, surfaces)
        inner_contrast = float(contrast_ratio(candidate, inner_rgb))
        return (float(surface_contrast), float(surface_lab), float(inner_contrast))

    return max(candidates, key=score)


def resolve_semantic_marker_style(
    *,
    instance_seed: int,
    namespace: str,
    role: str,
    surface_rgbs: Sequence[Sequence[int]],
    preferred_rgbs: Sequence[Sequence[int]] | None = None,
    candidate_rgbs: Sequence[Sequence[int]] | None = None,
    min_contrast_ratio: float = SEMANTIC_MARKER_MIN_CONTRAST_RATIO,
    min_lab_distance: float = SEMANTIC_MARKER_MIN_LAB_DISTANCE,
    required: bool = True,
) -> SemanticMarkerStyle:
    """Resolve a deterministic marker color with a contrast-safe halo."""

    surfaces = _unique_colors(surface_rgbs) or ((255, 255, 255),)
    preferred = tuple(preferred_rgbs or ())
    pool = list(_unique_colors((*preferred, *(candidate_rgbs or SEMANTIC_MARKER_COLOR_POOL))))
    rng = spawn_rng(instance_seed=int(instance_seed), namespace=str(namespace))
    rng.shuffle(pool)
    preferred_set = set(_unique_colors(preferred))

    eligible = [
        color
        for color in pool
        if _min_contrast(color, surfaces) >= float(min_contrast_ratio)
        and _min_lab_distance(color, surfaces) >= float(min_lab_distance)
    ]
    if eligible:
        preferred_eligible = [color for color in eligible if color in preferred_set]
        if preferred_eligible and rng.random() < 0.35:
            inner_rgb = preferred_eligible[int(rng.randint(0, len(preferred_eligible) - 1))]
        else:
            inner_rgb = eligible[int(rng.randint(0, len(eligible) - 1))]
    else:
        inner_rgb = max(pool or list(SEMANTIC_MARKER_COLOR_POOL), key=lambda color: (_min_contrast(color, surfaces), _min_lab_distance(color, surfaces)))

    outer_rgb = _choose_outer(inner_rgb, surfaces)
    inner_contrast = _min_contrast(inner_rgb, surfaces)
    outer_contrast = _min_contrast(outer_rgb, surfaces)
    inner_lab = _min_lab_distance(inner_rgb, surfaces)
    outer_lab = _min_lab_distance(outer_rgb, surfaces)
    return SemanticMarkerStyle(
        role=str(role),
        inner_rgb=inner_rgb,
        outer_rgb=outer_rgb,
        surface_rgbs=tuple(surfaces),
        min_inner_contrast_ratio=float(inner_contrast),
        min_outer_contrast_ratio=float(outer_contrast),
        min_effective_contrast_ratio=float(max(inner_contrast, outer_contrast)),
        min_inner_lab_distance=float(inner_lab),
        min_outer_lab_distance=float(outer_lab),
        min_effective_lab_distance=float(max(inner_lab, outer_lab)),
        min_contrast_required=float(min_contrast_ratio),
        min_lab_distance_required=float(min_lab_distance),
        required=bool(required),
    )


@contextmanager
def collect_semantic_marker_records() -> Iterable[list[dict[str, Any]]]:
    """Collect semantic marker metadata emitted by shared marker helpers."""

    records: list[dict[str, Any]] = []
    token = _ACTIVE_MARKER_RECORDS.set(records)
    try:
        yield records
    finally:
        _ACTIVE_MARKER_RECORDS.reset(token)


def record_semantic_marker(record: Mapping[str, Any]) -> None:
    records = _ACTIVE_MARKER_RECORDS.get()
    if records is not None:
        records.append(dict(record))


def semantic_marker_records_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    copied = [dict(record) for record in records]
    required = [record for record in copied if bool(record.get("required", True))]
    failures = [record for record in required if not bool(record.get("passes", False))]
    return {
        "enabled": True,
        "policy_version": MARKER_LEGIBILITY_POLICY_VERSION,
        "policy": "semantic_markers_use_surface_contrast_safe_halo_and_accent",
        "required_marker_count": int(len(required)),
        "drawn_marker_record_count": int(len(copied)),
        "failure_count": int(len(failures)),
        "records": copied,
    }


def _record_for_style(
    style: SemanticMarkerStyle,
    *,
    marker_kind: str,
    bbox_px: Sequence[float] | None = None,
    point_px: Sequence[float] | None = None,
    width_px: int | float | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = style.metadata()
    record["marker_kind"] = str(marker_kind)
    if bbox_px is not None and len(bbox_px) == 4:
        record["bbox_px"] = [round(float(value), 3) for value in bbox_px]
    if point_px is not None and len(point_px) >= 2:
        record["point_px"] = [round(float(point_px[0]), 3), round(float(point_px[1]), 3)]
    if width_px is not None:
        record["width_px"] = round(float(width_px), 3)
    if extra_metadata:
        record.update(dict(extra_metadata))
    record_semantic_marker(record)
    return record


def draw_semantic_bbox_marker(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    style: SemanticMarkerStyle,
    radius: int | float = 0,
    width: int = 4,
    fill_rgba: Sequence[int] | None = None,
    marker_kind: str = "bbox_outline",
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw a contrast-safe rectangular semantic marker."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    outer_width = max(1, int(width) + 2)
    inner_width = max(1, int(width))
    if fill_rgba is not None:
        fill = tuple(int(value) for value in fill_rgba[:4])
    else:
        fill = None
    if float(radius) > 0:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=int(radius), outline=style.outer_rgb, width=outer_width, fill=fill)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=int(radius), outline=style.inner_rgb, width=inner_width)
    else:
        draw.rectangle((x0, y0, x1, y1), outline=style.outer_rgb, width=outer_width, fill=fill)
        draw.rectangle((x0, y0, x1, y1), outline=style.inner_rgb, width=inner_width)
    return _record_for_style(
        style,
        marker_kind=str(marker_kind),
        bbox_px=(x0, y0, x1, y1),
        width_px=inner_width,
        extra_metadata=extra_metadata,
    )


def draw_semantic_ellipse_marker(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    style: SemanticMarkerStyle,
    width: int = 4,
    fill_rgba: Sequence[int] | None = None,
    marker_kind: str = "ellipse_outline",
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw a contrast-safe elliptical semantic marker."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    outer_width = max(1, int(width) + 2)
    inner_width = max(1, int(width))
    fill = tuple(int(value) for value in fill_rgba[:4]) if fill_rgba is not None else None
    draw.ellipse((x0, y0, x1, y1), outline=style.outer_rgb, width=outer_width, fill=fill)
    draw.ellipse((x0, y0, x1, y1), outline=style.inner_rgb, width=inner_width)
    return _record_for_style(
        style,
        marker_kind=str(marker_kind),
        bbox_px=(x0, y0, x1, y1),
        point_px=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        width_px=inner_width,
        extra_metadata=extra_metadata,
    )


def _draw_segmented_polyline(
    draw: ImageDraw.ImageDraw,
    pts: Sequence[Tuple[float, float]],
    *,
    fill: Sequence[int],
    width: int,
    dash_px: float,
    gap_px: float,
) -> None:
    """Draw a dashed polyline without changing semantic geometry."""

    dash_len = max(1.0, float(dash_px))
    gap_len = max(0.0, float(gap_px))
    for start, end in zip(pts, pts[1:]):
        start_x, start_y = float(start[0]), float(start[1])
        end_x, end_y = float(end[0]), float(end[1])
        dx = float(end_x - start_x)
        dy = float(end_y - start_y)
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        unit_x = dx / length
        unit_y = dy / length
        cursor = 0.0
        while cursor < length:
            segment_end = min(length, cursor + dash_len)
            draw.line(
                [
                    (start_x + (unit_x * cursor), start_y + (unit_y * cursor)),
                    (start_x + (unit_x * segment_end), start_y + (unit_y * segment_end)),
                ],
                fill=tuple(int(value) for value in fill),
                width=max(1, int(width)),
            )
            cursor = float(segment_end + gap_len)


def _marker_fill(color: Sequence[int], alpha: int | None) -> tuple[int, ...]:
    rgb = tuple(int(value) for value in color[:3])
    if alpha is None:
        return rgb
    return (*rgb, max(0, min(255, int(alpha))))


def _draw_dotted_polyline(
    draw: ImageDraw.ImageDraw,
    pts: Sequence[Tuple[float, float]],
    *,
    fill: Sequence[int],
    radius_px: float,
    spacing_px: float,
) -> None:
    """Draw a dotted polyline without changing semantic geometry."""

    radius = max(1.0, float(radius_px))
    spacing = max(radius * 2.5, float(spacing_px))
    for start, end in zip(pts, pts[1:]):
        start_x, start_y = float(start[0]), float(start[1])
        end_x, end_y = float(end[0]), float(end[1])
        dx = float(end_x - start_x)
        dy = float(end_y - start_y)
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        unit_x = dx / length
        unit_y = dy / length
        cursor = 0.0
        while cursor <= length + 1e-6:
            x = start_x + (unit_x * min(cursor, length))
            y = start_y + (unit_y * min(cursor, length))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=tuple(int(value) for value in fill),
            )
            cursor += spacing


def draw_semantic_line_marker(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    style: SemanticMarkerStyle,
    width: int = 4,
    pattern: str = "solid",
    dash_px: float | None = None,
    gap_px: float | None = None,
    dot_spacing_px: float | None = None,
    alpha: int | None = None,
    marker_kind: str = "line",
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw a contrast-safe semantic path or line marker."""

    pts = [(float(point[0]), float(point[1])) for point in points]
    if not pts:
        raise ValueError("semantic line marker requires at least one point")
    outer_width = max(1, int(width) + 3)
    inner_width = max(1, int(width))
    if len(pts) == 1:
        x, y = pts[0]
        radius = max(3.0, float(inner_width) * 1.5)
        return draw_semantic_ellipse_marker(
            draw,
            (x - radius, y - radius, x + radius, y + radius),
            style=style,
            width=inner_width,
            marker_kind=marker_kind,
            extra_metadata=extra_metadata,
        )
    marker_pattern = str(pattern or "solid")
    outer_fill = _marker_fill(style.outer_rgb, alpha)
    inner_fill = _marker_fill(style.inner_rgb, alpha)
    if marker_pattern == "dashed":
        _draw_segmented_polyline(
            draw,
            pts,
            fill=outer_fill,
            width=outer_width,
            dash_px=max(1.0, float(dash_px if dash_px is not None else outer_width * 2.4)),
            gap_px=max(1.0, float(gap_px if gap_px is not None else outer_width * 1.5)),
        )
        _draw_segmented_polyline(
            draw,
            pts,
            fill=inner_fill,
            width=inner_width,
            dash_px=max(1.0, float(dash_px if dash_px is not None else inner_width * 2.4)),
            gap_px=max(1.0, float(gap_px if gap_px is not None else inner_width * 1.5)),
        )
    elif marker_pattern == "dotted":
        spacing = max(float(outer_width) * 1.75, float(dot_spacing_px if dot_spacing_px is not None else outer_width * 1.9))
        _draw_dotted_polyline(
            draw,
            pts,
            fill=outer_fill,
            radius_px=max(1.0, float(outer_width) / 2.0),
            spacing_px=spacing,
        )
        _draw_dotted_polyline(
            draw,
            pts,
            fill=inner_fill,
            radius_px=max(1.0, float(inner_width) / 2.0),
            spacing_px=spacing,
        )
    else:
        marker_pattern = "solid"
        draw.line(pts, fill=outer_fill, width=outer_width, joint="curve")
        draw.line(pts, fill=inner_fill, width=inner_width, joint="curve")
    xs = [point[0] for point in pts]
    ys = [point[1] for point in pts]
    metadata = dict(extra_metadata or {})
    metadata["marker_pattern"] = str(marker_pattern)
    if alpha is not None:
        metadata["marker_alpha"] = max(0, min(255, int(alpha)))
    return _record_for_style(
        style,
        marker_kind=str(marker_kind),
        bbox_px=(min(xs), min(ys), max(xs), max(ys)),
        width_px=inner_width,
        extra_metadata=metadata,
    )


def draw_semantic_polygon_marker(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    style: SemanticMarkerStyle,
    width: int = 4,
    fill_rgba: Sequence[int] | None = None,
    marker_kind: str = "polygon_outline",
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw a contrast-safe polygon outline semantic marker."""

    pts = [(float(point[0]), float(point[1])) for point in points]
    if len(pts) < 3:
        return draw_semantic_line_marker(
            draw,
            pts,
            style=style,
            width=width,
            marker_kind=marker_kind,
            extra_metadata=extra_metadata,
        )
    fill = tuple(int(value) for value in fill_rgba[:4]) if fill_rgba is not None else None
    if fill is not None:
        draw.polygon(pts, fill=fill)
    closed = [*pts, pts[0]]
    record = draw_semantic_line_marker(
        draw,
        closed,
        style=style,
        width=width,
        marker_kind=marker_kind,
        extra_metadata=extra_metadata,
    )
    return record
