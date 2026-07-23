"""Shared symbolic-shape rendering helpers for puzzle families."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from PIL import ImageDraw


PUZZLE_OBJECT_TYPES: Tuple[str, ...] = (
    "circle",
    "triangle",
    "diamond",
    "square",
    "hexagon",
    "star",
)

PUZZLE_OBJECT_COLOR_BY_TYPE: Dict[str, Tuple[int, int, int]] = {
    "circle": (74, 127, 214),
    "triangle": (214, 130, 74),
    "diamond": (64, 164, 108),
    "square": (196, 90, 100),
    "hexagon": (136, 100, 196),
    "star": (205, 162, 62),
    "pentagon": (80, 150, 150),
}


def draw_puzzle_shape_icon(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    object_type: str,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    width: int,
    inset_px: float = 16.0,
) -> None:
    """Draw one deterministic symbolic shape inside a box."""

    left, top, right, bottom = [float(value) for value in bbox]
    inset = float(max(0.0, inset_px))
    x0, y0, x1, y1 = left + inset, top + inset, right - inset, bottom - inset
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0
    kind = str(object_type)
    fill = tuple(int(value) for value in fill_rgb)
    outline = tuple(int(value) for value in outline_rgb)
    stroke = max(1, int(width))

    if kind == "circle":
        draw.ellipse((x0, y0, x1, y1), fill=fill, outline=outline, width=stroke)
        return
    if kind == "square":
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=stroke)
        return
    if kind == "triangle":
        points = [(cx, y0), (x1, y1), (x0, y1)]
    elif kind == "pentagon":
        points = []
        for index in range(5):
            angle = -math.pi / 2.0 + (index * 2.0 * math.pi / 5.0)
            points.append(
                (
                    cx + 0.5 * w * math.cos(angle),
                    cy + 0.5 * h * math.sin(angle),
                )
            )
    elif kind == "diamond":
        points = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
    elif kind == "hexagon":
        points = [
            (x0 + 0.22 * w, y0),
            (x1 - 0.22 * w, y0),
            (x1, cy),
            (x1 - 0.22 * w, y1),
            (x0 + 0.22 * w, y1),
            (x0, cy),
        ]
    elif kind == "star":
        outer = 0.5 * min(w, h)
        inner = 0.45 * outer
        points = []
        for index in range(10):
            radius = outer if index % 2 == 0 else inner
            angle = -math.pi / 2.0 + (index * math.pi / 5.0)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    else:
        raise ValueError(f"unsupported puzzle object_type: {object_type}")
    draw.polygon(points, fill=fill, outline=outline, width=stroke)


__all__ = [
    "PUZZLE_OBJECT_COLOR_BY_TYPE",
    "PUZZLE_OBJECT_TYPES",
    "draw_puzzle_shape_icon",
]
