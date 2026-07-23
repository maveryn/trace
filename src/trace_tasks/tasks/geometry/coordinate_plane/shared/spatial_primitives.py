"""Shared coordinate-plane marker helpers for quadrilateral-style tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.tasks.shared.fixed_query import geometry_probability_map as _probability_map
from trace_tasks.tasks.shared.config_defaults import group_default


PixelPoint = Tuple[float, float]
Color = Tuple[int, int, int]

MARKER_STYLES: Tuple[str, ...] = ("filled_circle", "ring", "cross", "diamond", "square")
_MARKER_COLOR_PALETTES: Tuple[Tuple[Color, Color], ...] = (
    ((37, 99, 235), (220, 38, 38)),
    ((5, 150, 105), (147, 51, 234)),
    ((234, 88, 12), (8, 145, 178)),
    ((79, 70, 229), (202, 138, 4)),
)


def _resolve_label_pool(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
) -> Tuple[str, ...]:
    raw = params.get(str(key), group_default(defaults, str(key), list(fallback)))
    if isinstance(raw, str):
        labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    elif isinstance(raw, Sequence):
        labels = tuple(str(item).strip() for item in raw if str(item).strip())
    else:
        labels = tuple(str(item) for item in fallback)
    if not labels:
        raise ValueError(f"{key} must contain at least one label")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{key} must not contain duplicates")
    return labels


def _sample_marker_style(rng, *, params: Mapping[str, Any], defaults: Mapping[str, Any], key: str) -> str:
    explicit = params.get(str(key), group_default(defaults, str(key), None))
    if explicit is not None:
        style = str(explicit)
        if style not in set(MARKER_STYLES):
            raise ValueError(f"{key}={style!r} is not supported")
        return style
    return str(rng.choice(MARKER_STYLES))


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    point: PixelPoint,
    *,
    style: str,
    color: Color,
    radius: int,
    outline: Color = (255, 255, 255),
    width: int = 2,
) -> None:
    x, y = float(point[0]), float(point[1])
    r = max(2, int(radius))
    w = max(1, int(width))
    fill = tuple(int(value) for value in color)
    stroke = tuple(int(value) for value in outline)
    if str(style) == "ring":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=stroke, outline=fill, width=w)
    elif str(style) == "cross":
        draw.line([(x - r, y - r), (x + r, y + r)], fill=fill, width=w)
        draw.line([(x - r, y + r), (x + r, y - r)], fill=fill, width=w)
    elif str(style) == "diamond":
        draw.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=fill, outline=stroke)
    elif str(style) == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=fill, outline=stroke, width=w)
    else:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=stroke, width=w)


def _marker_bbox(point: PixelPoint, *, radius: int, canvas_width: int, canvas_height: int) -> List[int]:
    pad = max(4, int(radius) + 8)
    return [
        max(0, min(int(canvas_width), int(round(float(point[0]) - float(pad))))),
        max(0, min(int(canvas_height), int(round(float(point[1]) - float(pad))))),
        max(0, min(int(canvas_width), int(round(float(point[0]) + float(pad))))),
        max(0, min(int(canvas_height), int(round(float(point[1]) + float(pad))))),
    ]


def _resolve_marker_colors(rng) -> Tuple[Color, Color, Dict[str, Any]]:
    known_color, candidate_color = rng.choice(_MARKER_COLOR_PALETTES)
    return tuple(known_color), tuple(candidate_color), {
        "palette": [list(known_color), list(candidate_color)],
        "known_color": list(known_color),
        "candidate_color": list(candidate_color),
    }
