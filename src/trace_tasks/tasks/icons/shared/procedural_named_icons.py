"""Procedural named-icon vocabulary for dense icon-field tasks."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageOps

from .icon_noise import NoiseEdit, apply_icon_noise_edits_rgba


RGB = Tuple[int, int, int]

PROCEDURAL_NAMED_ICON_DISPLAY_NAMES: Dict[str, str] = {
    "circle": "circle",
    "ring": "ring",
    "square": "square",
    "triangle": "triangle",
    "pentagon": "pentagon",
    "hexagon": "hexagon",
    "octagon": "octagon",
    "star": "star",
    "heart": "heart",
    "crescent": "crescent",
    "check_mark": "check mark",
    "arrow": "arrow",
    "sun": "sun",
    "lightning_bolt": "lightning bolt",
    "cloud": "cloud",
    "flower": "flower",
    "leaf": "leaf",
    "shield": "shield",
    "flag": "flag",
    "crown": "crown",
    "house": "house",
    "capsule": "capsule",
    "teardrop": "teardrop",
    "hourglass": "hourglass",
    "key": "key",
    "bell": "bell",
    "ladder": "ladder",
    "kite": "kite",
    "mushroom": "mushroom",
    "fish": "fish",
    "butterfly": "butterfly",
    "umbrella": "umbrella",
    "anchor": "anchor",
    "car": "car",
    "boat": "boat",
    "rocket": "rocket",
    "envelope": "envelope",
    "book": "book",
    "pencil": "pencil",
    "camera": "camera",
    "lock": "lock",
    "magnifying_glass": "magnifying glass",
    "trophy": "trophy",
    "cup": "cup",
    "music_note": "music note",
    "battery": "battery",
    "clock": "clock",
    "gift": "gift",
    "shoe": "shoe",
    "shirt": "shirt",
    "glasses": "glasses",
    "snowman": "snowman",
    "tree": "tree",
    "bird": "bird",
    "apple": "apple",
    "balloon": "balloon",
    "shuriken": "shuriken",
    "gear": "gear",
    "puzzle_piece": "puzzle piece",
    "lightbulb": "lightbulb",
    "lantern": "lantern",
    "candle": "candle",
    "bottle": "bottle",
    "bucket": "bucket",
    "shovel": "shovel",
    "fork": "fork",
    "spoon": "spoon",
    "pizza_slice": "pizza slice",
    "egg": "egg",
    "dice": "dice",
    "microphone": "microphone",
    "headphones": "headphones",
    "chair": "chair",
    "tent": "tent",
    "phone": "phone",
    "television": "television",
    "laptop": "laptop",
    "watch": "watch",
    "bus": "bus",
    "train": "train",
    "bicycle": "bicycle",
    "bed": "bed",
    "lamp": "lamp",
    "door": "door",
    "window": "window",
    "mailbox": "mailbox",
    "toothbrush": "toothbrush",
    "broom": "broom",
    "trash_can": "trash can",
    "teapot": "teapot",
    "knife": "knife",
    "soccer_ball": "soccer ball",
    "rugby_ball": "rugby ball",
    "dumbbell": "dumbbell",
    "calculator": "calculator",
    "plug": "plug",
    "broccoli": "broccoli",
    "cactus": "cactus",
    "guitar": "guitar",
    "acorn": "acorn",
}

PROCEDURAL_NAMED_ICON_SHAPES: Tuple[str, ...] = tuple(PROCEDURAL_NAMED_ICON_DISPLAY_NAMES)

PROCEDURAL_NAMED_ICON_FILL_STYLE_DISPLAY_NAMES: Dict[str, str] = {
    "solid": "solid",
    "striped": "striped",
    "dotted": "dotted",
}

PROCEDURAL_NAMED_ICON_FILL_STYLES: Tuple[str, ...] = tuple(PROCEDURAL_NAMED_ICON_FILL_STYLE_DISPLAY_NAMES)
DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS: Dict[str, float] = {
    "solid": 0.70,
    "striped": 0.15,
    "dotted": 0.15,
}


def procedural_named_icon_display_name(shape_id: str) -> str:
    """Return prompt-facing display text for one procedural named icon."""

    key = str(shape_id)
    if key not in PROCEDURAL_NAMED_ICON_DISPLAY_NAMES:
        raise KeyError(f"unknown procedural named icon shape: {shape_id}")
    return str(PROCEDURAL_NAMED_ICON_DISPLAY_NAMES[key])


def procedural_named_icon_fill_style_display_name(fill_style: str) -> str:
    """Return prompt-facing display text for one procedural named icon fill style."""

    key = str(fill_style)
    if key not in PROCEDURAL_NAMED_ICON_FILL_STYLE_DISPLAY_NAMES:
        raise KeyError(f"unknown procedural named icon fill style: {fill_style}")
    return str(PROCEDURAL_NAMED_ICON_FILL_STYLE_DISPLAY_NAMES[key])


def validate_procedural_named_icon_fill_style_support(
    values: Sequence[str],
) -> Tuple[str, ...]:
    """Validate and de-duplicate fill-style ids from config-like support values."""

    allowed = set(PROCEDURAL_NAMED_ICON_FILL_STYLES)
    support = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    unsupported = sorted(set(support) - allowed)
    if unsupported:
        raise ValueError(f"unsupported procedural named icon fill styles: {unsupported}")
    if not support:
        raise ValueError("procedural named icon fill-style support resolved no values")
    return support


def procedural_named_icon_fill_style_probability_map(
    support: Sequence[str],
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Return a normalized probability map over validated fill-style support."""

    values = validate_procedural_named_icon_fill_style_support(tuple(str(value) for value in support))
    raw_weights = weights if isinstance(weights, dict) else DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS
    resolved = {
        str(value): max(0.0, float(raw_weights.get(str(value), 0.0)))
        for value in values
    }
    total = sum(float(value) for value in resolved.values())
    if total <= 0.0:
        probability = 1.0 / float(len(values))
        return {str(value): probability for value in values}
    return {str(value): float(resolved[str(value)]) / float(total) for value in values}


def sample_procedural_named_icon_fill_style(
    rng,
    *,
    support: Sequence[str],
    probabilities: Dict[str, float] | None = None,
) -> str:
    """Sample one fill style from a normalized or normalizable probability map."""

    values = validate_procedural_named_icon_fill_style_support(tuple(str(value) for value in support))
    probability_map = procedural_named_icon_fill_style_probability_map(values, probabilities)
    threshold = float(rng.random())
    cumulative = 0.0
    for value in values:
        cumulative += float(probability_map[str(value)])
        if threshold <= cumulative:
            return str(value)
    return str(values[-1])


def _regular_polygon_points(cx: float, cy: float, radius: float, sides: int, *, start_degrees: float) -> list[tuple[float, float]]:
    return [
        (
            float(cx) + float(radius) * math.cos(math.radians(float(start_degrees) + 360.0 * float(i) / float(sides))),
            float(cy) + float(radius) * math.sin(math.radians(float(start_degrees) + 360.0 * float(i) / float(sides))),
        )
        for i in range(int(sides))
    ]


def _star_points(cx: float, cy: float, outer_radius: float, inner_radius: float, *, points: int = 5) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for i in range(int(points) * 2):
        radius = float(outer_radius) if i % 2 == 0 else float(inner_radius)
        angle = math.radians(-90.0 + 180.0 * float(i) / float(points))
        values.append((float(cx) + radius * math.cos(angle), float(cy) + radius * math.sin(angle)))
    return values


def _draw_thick_line(draw: ImageDraw.ImageDraw, points: Sequence[tuple[float, float]], *, width: int, fill: int = 255) -> None:
    draw.line([(float(x), float(y)) for x, y in points], fill=int(fill), width=max(1, int(width)), joint="curve")


def _draw_shape_mask(shape_id: str, *, size: int) -> Image.Image:
    shape = str(shape_id)
    if shape not in PROCEDURAL_NAMED_ICON_DISPLAY_NAMES:
        raise KeyError(f"unknown procedural named icon shape: {shape_id}")

    s = int(size)
    m = float(s)
    cx = cy = m / 2.0
    pad = 0.15 * m
    r = 0.36 * m
    mask = Image.new("L", (s, s), 0)
    draw = ImageDraw.Draw(mask)
    box = (pad, pad, m - pad, m - pad)
    line_w = max(2, int(round(0.16 * m)))
    thin_w = max(2, int(round(0.08 * m)))

    if shape == "circle":
        draw.ellipse(box, fill=255)
    elif shape == "ring":
        draw.ellipse(box, fill=255)
        inner = 0.32 * m
        draw.ellipse((cx - inner / 2, cy - inner / 2, cx + inner / 2, cy + inner / 2), fill=0)
    elif shape == "square":
        draw.rounded_rectangle(box, radius=max(2, int(0.05 * m)), fill=255)
    elif shape == "triangle":
        draw.polygon([(cx, pad), (m - pad, m - pad), (pad, m - pad)], fill=255)
    elif shape == "pentagon":
        draw.polygon(_regular_polygon_points(cx, cy + 0.02 * m, r, 5, start_degrees=-90), fill=255)
    elif shape == "hexagon":
        draw.polygon(_regular_polygon_points(cx, cy, r, 6, start_degrees=30), fill=255)
    elif shape == "octagon":
        draw.polygon(_regular_polygon_points(cx, cy, r, 8, start_degrees=22.5), fill=255)
    elif shape == "star":
        draw.polygon(_star_points(cx, cy, 0.40 * m, 0.18 * m), fill=255)
    elif shape == "heart":
        points = []
        for index in range(96):
            t = 2.0 * math.pi * float(index) / 96.0
            raw_x = 16.0 * math.sin(t) ** 3
            raw_y = 13.0 * math.cos(t) - 5.0 * math.cos(2.0 * t) - 2.0 * math.cos(3.0 * t) - math.cos(4.0 * t)
            points.append((0.50 * m + raw_x * 0.023 * m, 0.52 * m - raw_y * 0.024 * m))
        draw.polygon(points, fill=255)
    elif shape == "crescent":
        draw.ellipse((0.18 * m, 0.12 * m, 0.82 * m, 0.88 * m), fill=255)
        draw.ellipse((0.38 * m, 0.08 * m, 0.96 * m, 0.78 * m), fill=0)
    elif shape == "check_mark":
        _draw_thick_line(draw, ((0.18 * m, 0.54 * m), (0.40 * m, 0.76 * m), (0.82 * m, 0.28 * m)), width=line_w)
    elif shape == "arrow":
        draw.polygon(
            [
                (0.16 * m, 0.40 * m),
                (0.58 * m, 0.40 * m),
                (0.58 * m, 0.22 * m),
                (0.86 * m, 0.50 * m),
                (0.58 * m, 0.78 * m),
                (0.58 * m, 0.60 * m),
                (0.16 * m, 0.60 * m),
            ],
            fill=255,
        )
    elif shape == "sun":
        for angle in range(0, 360, 45):
            a = math.radians(float(angle))
            _draw_thick_line(
                draw,
                (
                    (cx + 0.31 * m * math.cos(a), cy + 0.31 * m * math.sin(a)),
                    (cx + 0.43 * m * math.cos(a), cy + 0.43 * m * math.sin(a)),
                ),
                width=thin_w,
            )
        draw.ellipse((0.28 * m, 0.28 * m, 0.72 * m, 0.72 * m), fill=255)
    elif shape == "lightning_bolt":
        draw.polygon(
            [
                (0.54 * m, 0.10 * m),
                (0.25 * m, 0.54 * m),
                (0.48 * m, 0.54 * m),
                (0.38 * m, 0.90 * m),
                (0.78 * m, 0.42 * m),
                (0.55 * m, 0.42 * m),
            ],
            fill=255,
        )
    elif shape == "cloud":
        draw.ellipse((0.20 * m, 0.42 * m, 0.48 * m, 0.68 * m), fill=255)
        draw.ellipse((0.34 * m, 0.30 * m, 0.66 * m, 0.66 * m), fill=255)
        draw.ellipse((0.54 * m, 0.40 * m, 0.82 * m, 0.68 * m), fill=255)
        draw.rounded_rectangle((0.22 * m, 0.50 * m, 0.80 * m, 0.75 * m), radius=int(0.12 * m), fill=255)
    elif shape == "flower":
        _draw_thick_line(draw, ((0.50 * m, 0.48 * m), (0.50 * m, 0.90 * m)), width=thin_w)
        draw.ellipse((0.30 * m, 0.62 * m, 0.50 * m, 0.78 * m), fill=255)
        draw.ellipse((0.50 * m, 0.64 * m, 0.72 * m, 0.80 * m), fill=255)
        for angle in range(0, 360, 72):
            a = math.radians(float(angle))
            px = cx + 0.18 * m * math.cos(a)
            py = 0.34 * m + 0.18 * m * math.sin(a)
            draw.ellipse((px - 0.12 * m, py - 0.12 * m, px + 0.12 * m, py + 0.12 * m), fill=255)
        draw.ellipse((0.40 * m, 0.24 * m, 0.60 * m, 0.44 * m), fill=255)
    elif shape == "leaf":
        draw.polygon(
            [
                (0.52 * m, 0.10 * m),
                (0.70 * m, 0.18 * m),
                (0.84 * m, 0.34 * m),
                (0.86 * m, 0.54 * m),
                (0.74 * m, 0.72 * m),
                (0.52 * m, 0.84 * m),
                (0.30 * m, 0.78 * m),
                (0.16 * m, 0.60 * m),
                (0.18 * m, 0.38 * m),
                (0.32 * m, 0.20 * m),
            ],
            fill=255,
        )
        _draw_thick_line(draw, ((0.52 * m, 0.14 * m), (0.48 * m, 0.82 * m)), width=max(1, thin_w // 2), fill=0)
        for y0, side in ((0.30, -1), (0.42, 1), (0.54, -1), (0.66, 1)):
            start = (0.50 * m, y0 * m)
            end = ((0.50 + 0.18 * side) * m, (y0 + 0.12) * m)
            _draw_thick_line(draw, (start, end), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.48 * m, 0.82 * m), (0.42 * m, 1.00 * m)), width=thin_w)
    elif shape == "shield":
        draw.polygon(
            [
                (0.18 * m, 0.18 * m),
                (0.50 * m, 0.10 * m),
                (0.82 * m, 0.18 * m),
                (0.78 * m, 0.52 * m),
                (0.66 * m, 0.72 * m),
                (0.50 * m, 0.88 * m),
                (0.34 * m, 0.72 * m),
                (0.22 * m, 0.52 * m),
            ],
            fill=255,
        )
        _draw_thick_line(draw, ((0.50 * m, 0.18 * m), (0.50 * m, 0.78 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "flag":
        draw.rounded_rectangle((0.20 * m, 0.14 * m, 0.32 * m, 0.86 * m), radius=int(0.025 * m), fill=255)
        draw.polygon([(0.31 * m, 0.16 * m), (0.80 * m, 0.24 * m), (0.31 * m, 0.48 * m)], fill=255)
    elif shape == "crown":
        draw.polygon(
            [
                (0.18 * m, 0.34 * m),
                (0.34 * m, 0.58 * m),
                (0.50 * m, 0.24 * m),
                (0.66 * m, 0.58 * m),
                (0.82 * m, 0.34 * m),
                (0.78 * m, 0.76 * m),
                (0.22 * m, 0.76 * m),
            ],
            fill=255,
        )
    elif shape == "house":
        draw.polygon([(0.16 * m, 0.48 * m), (0.50 * m, 0.18 * m), (0.84 * m, 0.48 * m)], fill=255)
        draw.rounded_rectangle((0.24 * m, 0.44 * m, 0.76 * m, 0.82 * m), radius=int(0.04 * m), fill=255)
    elif shape == "capsule":
        draw.rounded_rectangle((0.16 * m, 0.34 * m, 0.84 * m, 0.66 * m), radius=int(0.16 * m), fill=255)
    elif shape == "teardrop":
        draw.ellipse((0.24 * m, 0.34 * m, 0.76 * m, 0.86 * m), fill=255)
        draw.polygon([(0.50 * m, 0.12 * m), (0.27 * m, 0.50 * m), (0.73 * m, 0.50 * m)], fill=255)
    elif shape == "hourglass":
        draw.rounded_rectangle((0.24 * m, 0.12 * m, 0.76 * m, 0.24 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.24 * m, 0.76 * m, 0.76 * m, 0.88 * m), radius=int(0.04 * m), fill=255)
        _draw_thick_line(draw, ((0.28 * m, 0.22 * m), (0.42 * m, 0.50 * m), (0.28 * m, 0.78 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.72 * m, 0.22 * m), (0.58 * m, 0.50 * m), (0.72 * m, 0.78 * m)), width=thin_w)
        draw.polygon([(0.34 * m, 0.28 * m), (0.66 * m, 0.28 * m), (0.54 * m, 0.48 * m), (0.46 * m, 0.48 * m)], fill=255)
        draw.polygon([(0.46 * m, 0.54 * m), (0.54 * m, 0.54 * m), (0.68 * m, 0.72 * m), (0.32 * m, 0.72 * m)], fill=255)
    elif shape == "key":
        draw.ellipse((0.12 * m, 0.58 * m, 0.46 * m, 0.92 * m), fill=255)
        draw.ellipse((0.23 * m, 0.69 * m, 0.35 * m, 0.81 * m), fill=0)
        _draw_thick_line(draw, ((0.42 * m, 0.62 * m), (0.80 * m, 0.24 * m)), width=thin_w)
        draw.polygon([(0.72 * m, 0.30 * m), (0.80 * m, 0.22 * m), (0.88 * m, 0.30 * m), (0.80 * m, 0.38 * m)], fill=255)
        draw.polygon([(0.62 * m, 0.40 * m), (0.70 * m, 0.32 * m), (0.78 * m, 0.40 * m), (0.70 * m, 0.48 * m)], fill=255)
    elif shape == "bell":
        draw.arc((0.10 * m, 0.30 * m, 0.28 * m, 0.70 * m), start=105, end=255, fill=255, width=max(1, thin_w // 2))
        draw.arc((0.72 * m, 0.30 * m, 0.90 * m, 0.70 * m), start=285, end=75, fill=255, width=max(1, thin_w // 2))
        draw.pieslice((0.22 * m, 0.18 * m, 0.78 * m, 0.86 * m), start=180, end=360, fill=255)
        draw.rounded_rectangle((0.24 * m, 0.46 * m, 0.76 * m, 0.78 * m), radius=int(0.12 * m), fill=255)
        draw.ellipse((0.40 * m, 0.74 * m, 0.60 * m, 0.94 * m), fill=255)
        draw.rounded_rectangle((0.43 * m, 0.10 * m, 0.57 * m, 0.24 * m), radius=int(0.05 * m), fill=255)
    elif shape == "ladder":
        _draw_thick_line(draw, ((0.30 * m, 0.16 * m), (0.30 * m, 0.86 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.70 * m, 0.16 * m), (0.70 * m, 0.86 * m)), width=thin_w)
        for y in (0.28, 0.42, 0.56, 0.70):
            _draw_thick_line(draw, ((0.30 * m, y * m), (0.70 * m, y * m)), width=thin_w)
    elif shape == "kite":
        draw.polygon([(0.50 * m, 0.10 * m), (0.78 * m, 0.42 * m), (0.50 * m, 0.88 * m), (0.22 * m, 0.42 * m)], fill=255)
        _draw_thick_line(draw, ((0.50 * m, 0.12 * m), (0.50 * m, 0.86 * m)), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.24 * m, 0.42 * m), (0.76 * m, 0.42 * m)), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.50 * m, 0.88 * m), (0.42 * m, 0.96 * m), (0.58 * m, 0.98 * m)), width=max(1, thin_w // 2))
    elif shape == "mushroom":
        draw.pieslice((0.14 * m, 0.18 * m, 0.86 * m, 0.78 * m), start=180, end=360, fill=255)
        draw.rounded_rectangle((0.38 * m, 0.46 * m, 0.62 * m, 0.84 * m), radius=int(0.11 * m), fill=255)
        draw.ellipse((0.28 * m, 0.30 * m, 0.38 * m, 0.40 * m), fill=0)
        draw.ellipse((0.58 * m, 0.27 * m, 0.70 * m, 0.39 * m), fill=0)
    elif shape == "fish":
        draw.ellipse((0.20 * m, 0.30 * m, 0.72 * m, 0.70 * m), fill=255)
        draw.polygon([(0.72 * m, 0.50 * m), (0.90 * m, 0.28 * m), (0.90 * m, 0.72 * m)], fill=255)
        draw.polygon([(0.44 * m, 0.30 * m), (0.58 * m, 0.12 * m), (0.58 * m, 0.36 * m)], fill=255)
        draw.ellipse((0.30 * m, 0.42 * m, 0.38 * m, 0.50 * m), fill=0)
    elif shape == "butterfly":
        draw.ellipse((0.10 * m, 0.16 * m, 0.48 * m, 0.56 * m), fill=255)
        draw.ellipse((0.52 * m, 0.16 * m, 0.90 * m, 0.56 * m), fill=255)
        draw.ellipse((0.16 * m, 0.50 * m, 0.46 * m, 0.84 * m), fill=255)
        draw.ellipse((0.54 * m, 0.50 * m, 0.84 * m, 0.84 * m), fill=255)
        draw.rounded_rectangle((0.45 * m, 0.24 * m, 0.55 * m, 0.82 * m), radius=int(0.05 * m), fill=255)
        _draw_thick_line(draw, ((0.48 * m, 0.26 * m), (0.34 * m, 0.10 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.52 * m, 0.26 * m), (0.66 * m, 0.10 * m)), width=max(1, thin_w // 2))
        draw.ellipse((0.30 * m, 0.06 * m, 0.38 * m, 0.14 * m), fill=255)
        draw.ellipse((0.62 * m, 0.06 * m, 0.70 * m, 0.14 * m), fill=255)
        for px, py in ((0.28, 0.34), (0.72, 0.34), (0.30, 0.64), (0.70, 0.64)):
            draw.ellipse(((px - 0.035) * m, (py - 0.035) * m, (px + 0.035) * m, (py + 0.035) * m), fill=0)
    elif shape == "umbrella":
        draw.pieslice((0.12 * m, 0.18 * m, 0.88 * m, 0.82 * m), start=180, end=360, fill=255)
        _draw_thick_line(draw, ((0.50 * m, 0.50 * m), (0.50 * m, 0.78 * m), (0.42 * m, 0.88 * m), (0.32 * m, 0.80 * m)), width=thin_w)
    elif shape == "anchor":
        draw.ellipse((0.40 * m, 0.10 * m, 0.60 * m, 0.30 * m), fill=255)
        draw.ellipse((0.455 * m, 0.155 * m, 0.545 * m, 0.245 * m), fill=0)
        _draw_thick_line(draw, ((0.50 * m, 0.28 * m), (0.50 * m, 0.74 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.30 * m, 0.40 * m), (0.70 * m, 0.40 * m)), width=thin_w)
        draw.arc((0.20 * m, 0.40 * m, 0.80 * m, 0.92 * m), start=20, end=160, fill=255, width=line_w)
        draw.polygon([(0.18 * m, 0.60 * m), (0.34 * m, 0.68 * m), (0.23 * m, 0.78 * m)], fill=255)
        draw.polygon([(0.82 * m, 0.60 * m), (0.66 * m, 0.68 * m), (0.77 * m, 0.78 * m)], fill=255)
    elif shape == "car":
        draw.rounded_rectangle((0.10 * m, 0.46 * m, 0.90 * m, 0.72 * m), radius=int(0.08 * m), fill=255)
        draw.polygon([(0.26 * m, 0.46 * m), (0.40 * m, 0.28 * m), (0.64 * m, 0.28 * m), (0.78 * m, 0.46 * m)], fill=255)
        draw.ellipse((0.20 * m, 0.66 * m, 0.38 * m, 0.84 * m), fill=255)
        draw.ellipse((0.62 * m, 0.66 * m, 0.80 * m, 0.84 * m), fill=255)
        draw.rectangle((0.40 * m, 0.36 * m, 0.52 * m, 0.46 * m), fill=0)
        draw.rectangle((0.56 * m, 0.36 * m, 0.68 * m, 0.46 * m), fill=0)
    elif shape == "boat":
        draw.polygon([(0.18 * m, 0.56 * m), (0.84 * m, 0.56 * m), (0.68 * m, 0.78 * m), (0.30 * m, 0.78 * m)], fill=255)
        _draw_thick_line(draw, ((0.50 * m, 0.22 * m), (0.50 * m, 0.58 * m)), width=thin_w)
        draw.polygon([(0.52 * m, 0.24 * m), (0.76 * m, 0.52 * m), (0.52 * m, 0.52 * m)], fill=255)
        draw.polygon([(0.48 * m, 0.30 * m), (0.30 * m, 0.52 * m), (0.48 * m, 0.52 * m)], fill=255)
    elif shape == "rocket":
        draw.polygon(
            [
                (0.50 * m, 0.08 * m),
                (0.64 * m, 0.36 * m),
                (0.66 * m, 0.64 * m),
                (0.84 * m, 0.86 * m),
                (0.58 * m, 0.76 * m),
                (0.50 * m, 0.92 * m),
                (0.42 * m, 0.76 * m),
                (0.16 * m, 0.86 * m),
                (0.34 * m, 0.64 * m),
                (0.36 * m, 0.36 * m),
            ],
            fill=255,
        )
        draw.ellipse((0.42 * m, 0.26 * m, 0.58 * m, 0.42 * m), fill=0)
        draw.rectangle((0.45 * m, 0.64 * m, 0.55 * m, 0.82 * m), fill=0)
    elif shape == "envelope":
        draw.rounded_rectangle((0.16 * m, 0.28 * m, 0.84 * m, 0.72 * m), radius=int(0.04 * m), fill=255)
        _draw_thick_line(draw, ((0.18 * m, 0.30 * m), (0.50 * m, 0.56 * m), (0.82 * m, 0.30 * m)), width=thin_w, fill=0)
        _draw_thick_line(draw, ((0.18 * m, 0.70 * m), (0.42 * m, 0.50 * m)), width=thin_w, fill=0)
        _draw_thick_line(draw, ((0.82 * m, 0.70 * m), (0.58 * m, 0.50 * m)), width=thin_w, fill=0)
    elif shape == "book":
        draw.polygon(
            [
                (0.14 * m, 0.22 * m),
                (0.48 * m, 0.30 * m),
                (0.48 * m, 0.84 * m),
                (0.14 * m, 0.74 * m),
            ],
            fill=255,
        )
        draw.polygon(
            [
                (0.86 * m, 0.22 * m),
                (0.52 * m, 0.30 * m),
                (0.52 * m, 0.84 * m),
                (0.86 * m, 0.74 * m),
            ],
            fill=255,
        )
        draw.rectangle((0.48 * m, 0.30 * m, 0.52 * m, 0.84 * m), fill=255)
        _draw_thick_line(draw, ((0.50 * m, 0.32 * m), (0.50 * m, 0.84 * m)), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.24 * m, 0.42 * m), (0.40 * m, 0.46 * m)), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.60 * m, 0.46 * m), (0.76 * m, 0.42 * m)), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.24 * m, 0.58 * m), (0.40 * m, 0.62 * m)), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.60 * m, 0.62 * m), (0.76 * m, 0.58 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "pencil":
        draw.polygon([(0.22 * m, 0.70 * m), (0.58 * m, 0.34 * m), (0.72 * m, 0.48 * m), (0.36 * m, 0.84 * m)], fill=255)
        draw.polygon([(0.58 * m, 0.34 * m), (0.74 * m, 0.18 * m), (0.72 * m, 0.48 * m)], fill=255)
        draw.polygon([(0.22 * m, 0.70 * m), (0.14 * m, 0.90 * m), (0.36 * m, 0.84 * m)], fill=255)
        draw.polygon([(0.18 * m, 0.66 * m), (0.30 * m, 0.54 * m), (0.44 * m, 0.68 * m), (0.32 * m, 0.80 * m)], fill=255)
        _draw_thick_line(draw, ((0.34 * m, 0.50 * m), (0.48 * m, 0.64 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "camera":
        draw.rounded_rectangle((0.16 * m, 0.30 * m, 0.84 * m, 0.76 * m), radius=int(0.07 * m), fill=255)
        draw.rounded_rectangle((0.28 * m, 0.20 * m, 0.52 * m, 0.34 * m), radius=int(0.04 * m), fill=255)
        draw.ellipse((0.38 * m, 0.40 * m, 0.66 * m, 0.68 * m), fill=0)
        draw.ellipse((0.45 * m, 0.47 * m, 0.59 * m, 0.61 * m), fill=255)
    elif shape == "lock":
        draw.rounded_rectangle((0.24 * m, 0.44 * m, 0.76 * m, 0.84 * m), radius=int(0.07 * m), fill=255)
        draw.arc((0.28 * m, 0.10 * m, 0.72 * m, 0.58 * m), start=185, end=355, fill=255, width=line_w)
        _draw_thick_line(draw, ((0.32 * m, 0.36 * m), (0.32 * m, 0.50 * m)), width=line_w)
        _draw_thick_line(draw, ((0.68 * m, 0.38 * m), (0.68 * m, 0.48 * m)), width=line_w)
        draw.ellipse((0.46 * m, 0.60 * m, 0.54 * m, 0.68 * m), fill=0)
    elif shape == "magnifying_glass":
        draw.ellipse((0.12 * m, 0.10 * m, 0.70 * m, 0.68 * m), fill=255)
        draw.ellipse((0.24 * m, 0.22 * m, 0.58 * m, 0.56 * m), fill=0)
        _draw_thick_line(draw, ((0.62 * m, 0.60 * m), (0.86 * m, 0.86 * m)), width=line_w)
    elif shape == "trophy":
        draw.rounded_rectangle((0.34 * m, 0.14 * m, 0.66 * m, 0.34 * m), radius=int(0.04 * m), fill=255)
        draw.pieslice((0.30 * m, 0.16 * m, 0.70 * m, 0.70 * m), start=0, end=180, fill=255)
        draw.rectangle((0.34 * m, 0.16 * m, 0.66 * m, 0.42 * m), fill=255)
        draw.arc((0.08 * m, 0.20 * m, 0.40 * m, 0.62 * m), start=280, end=95, fill=255, width=thin_w)
        draw.arc((0.60 * m, 0.20 * m, 0.92 * m, 0.62 * m), start=85, end=260, fill=255, width=thin_w)
        draw.ellipse((0.46 * m, 0.22 * m, 0.54 * m, 0.30 * m), fill=0)
        draw.rounded_rectangle((0.45 * m, 0.58 * m, 0.55 * m, 0.78 * m), radius=int(0.03 * m), fill=255)
        draw.rounded_rectangle((0.32 * m, 0.76 * m, 0.68 * m, 0.88 * m), radius=int(0.04 * m), fill=255)
    elif shape == "cup":
        draw.rounded_rectangle((0.22 * m, 0.26 * m, 0.68 * m, 0.72 * m), radius=int(0.09 * m), fill=255)
        draw.rectangle((0.26 * m, 0.26 * m, 0.64 * m, 0.38 * m), fill=255)
        draw.arc((0.58 * m, 0.34 * m, 0.90 * m, 0.66 * m), start=270, end=90, fill=255, width=line_w)
        draw.arc((0.64 * m, 0.42 * m, 0.82 * m, 0.58 * m), start=270, end=90, fill=0, width=max(1, thin_w // 2))
        draw.rounded_rectangle((0.30 * m, 0.72 * m, 0.60 * m, 0.82 * m), radius=int(0.04 * m), fill=255)
    elif shape == "music_note":
        draw.ellipse((0.22 * m, 0.62 * m, 0.48 * m, 0.84 * m), fill=255)
        draw.rounded_rectangle((0.44 * m, 0.20 * m, 0.56 * m, 0.72 * m), radius=int(0.03 * m), fill=255)
        draw.rounded_rectangle((0.52 * m, 0.18 * m, 0.78 * m, 0.30 * m), radius=int(0.03 * m), fill=255)
    elif shape == "battery":
        draw.rounded_rectangle((0.16 * m, 0.34 * m, 0.76 * m, 0.66 * m), radius=int(0.05 * m), fill=255)
        draw.rounded_rectangle((0.76 * m, 0.42 * m, 0.88 * m, 0.58 * m), radius=int(0.03 * m), fill=255)
        draw.rectangle((0.26 * m, 0.42 * m, 0.66 * m, 0.58 * m), fill=0)
        draw.rectangle((0.30 * m, 0.44 * m, 0.40 * m, 0.56 * m), fill=255)
        draw.rectangle((0.44 * m, 0.44 * m, 0.54 * m, 0.56 * m), fill=255)
    elif shape == "clock":
        draw.ellipse((0.16 * m, 0.16 * m, 0.84 * m, 0.84 * m), fill=255)
        draw.ellipse((0.26 * m, 0.26 * m, 0.74 * m, 0.74 * m), fill=0)
        _draw_thick_line(draw, ((0.50 * m, 0.50 * m), (0.50 * m, 0.30 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.50 * m, 0.50 * m), (0.66 * m, 0.58 * m)), width=thin_w)
    elif shape == "gift":
        draw.rounded_rectangle((0.18 * m, 0.36 * m, 0.82 * m, 0.84 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.14 * m, 0.28 * m, 0.86 * m, 0.44 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.44 * m, 0.18 * m, 0.56 * m, 0.84 * m), radius=int(0.03 * m), fill=255)
        draw.pieslice((0.26 * m, 0.12 * m, 0.52 * m, 0.40 * m), start=20, end=340, fill=255)
        draw.pieslice((0.48 * m, 0.12 * m, 0.74 * m, 0.40 * m), start=200, end=160, fill=255)
    elif shape == "shoe":
        draw.polygon([(0.18 * m, 0.62 * m), (0.44 * m, 0.52 * m), (0.62 * m, 0.62 * m), (0.86 * m, 0.64 * m), (0.86 * m, 0.78 * m), (0.20 * m, 0.78 * m)], fill=255)
        draw.rounded_rectangle((0.22 * m, 0.74 * m, 0.88 * m, 0.86 * m), radius=int(0.04 * m), fill=255)
        _draw_thick_line(draw, ((0.42 * m, 0.60 * m), (0.54 * m, 0.70 * m)), width=max(1, thin_w // 2), fill=0)
        _draw_thick_line(draw, ((0.54 * m, 0.60 * m), (0.42 * m, 0.70 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "shirt":
        draw.polygon([(0.14 * m, 0.24 * m), (0.36 * m, 0.14 * m), (0.46 * m, 0.30 * m), (0.54 * m, 0.30 * m), (0.64 * m, 0.14 * m), (0.86 * m, 0.24 * m), (0.76 * m, 0.52 * m), (0.66 * m, 0.46 * m), (0.66 * m, 0.86 * m), (0.34 * m, 0.86 * m), (0.34 * m, 0.46 * m), (0.24 * m, 0.52 * m)], fill=255)
    elif shape == "glasses":
        draw.rounded_rectangle((0.12 * m, 0.34 * m, 0.44 * m, 0.66 * m), radius=int(0.09 * m), fill=255)
        draw.rounded_rectangle((0.20 * m, 0.42 * m, 0.36 * m, 0.58 * m), radius=int(0.05 * m), fill=0)
        draw.rounded_rectangle((0.56 * m, 0.34 * m, 0.88 * m, 0.66 * m), radius=int(0.09 * m), fill=255)
        draw.rounded_rectangle((0.64 * m, 0.42 * m, 0.80 * m, 0.58 * m), radius=int(0.05 * m), fill=0)
        _draw_thick_line(draw, ((0.42 * m, 0.50 * m), (0.58 * m, 0.50 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.14 * m, 0.46 * m), (0.02 * m, 0.38 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.86 * m, 0.46 * m), (0.98 * m, 0.38 * m)), width=max(1, thin_w // 2))
    elif shape == "snowman":
        draw.ellipse((0.34 * m, 0.12 * m, 0.66 * m, 0.44 * m), fill=255)
        draw.ellipse((0.26 * m, 0.38 * m, 0.74 * m, 0.86 * m), fill=255)
        draw.ellipse((0.42 * m, 0.24 * m, 0.46 * m, 0.28 * m), fill=0)
        draw.ellipse((0.54 * m, 0.24 * m, 0.58 * m, 0.28 * m), fill=0)
        draw.polygon([(0.50 * m, 0.30 * m), (0.62 * m, 0.34 * m), (0.50 * m, 0.38 * m)], fill=0)
    elif shape == "tree":
        draw.polygon([(0.50 * m, 0.12 * m), (0.28 * m, 0.42 * m), (0.72 * m, 0.42 * m)], fill=255)
        draw.polygon([(0.50 * m, 0.28 * m), (0.20 * m, 0.64 * m), (0.80 * m, 0.64 * m)], fill=255)
        draw.polygon([(0.50 * m, 0.46 * m), (0.16 * m, 0.82 * m), (0.84 * m, 0.82 * m)], fill=255)
        draw.rectangle((0.44 * m, 0.76 * m, 0.56 * m, 0.90 * m), fill=255)
    elif shape == "bird":
        draw.ellipse((0.26 * m, 0.34 * m, 0.72 * m, 0.70 * m), fill=255)
        draw.ellipse((0.58 * m, 0.24 * m, 0.82 * m, 0.48 * m), fill=255)
        draw.polygon([(0.80 * m, 0.36 * m), (0.94 * m, 0.42 * m), (0.80 * m, 0.48 * m)], fill=255)
        draw.polygon([(0.26 * m, 0.50 * m), (0.08 * m, 0.36 * m), (0.16 * m, 0.62 * m)], fill=255)
        draw.ellipse((0.68 * m, 0.32 * m, 0.74 * m, 0.38 * m), fill=0)
        _draw_thick_line(draw, ((0.44 * m, 0.68 * m), (0.42 * m, 0.84 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.56 * m, 0.68 * m), (0.58 * m, 0.84 * m)), width=max(1, thin_w // 2))
    elif shape == "apple":
        draw.ellipse((0.20 * m, 0.30 * m, 0.56 * m, 0.80 * m), fill=255)
        draw.ellipse((0.44 * m, 0.30 * m, 0.80 * m, 0.80 * m), fill=255)
        draw.rounded_rectangle((0.32 * m, 0.42 * m, 0.68 * m, 0.78 * m), radius=int(0.18 * m), fill=255)
        draw.pieslice((0.32 * m, 0.62 * m, 0.68 * m, 0.92 * m), start=0, end=180, fill=255)
        draw.ellipse((0.40 * m, 0.22 * m, 0.60 * m, 0.40 * m), fill=0)
        _draw_thick_line(draw, ((0.50 * m, 0.34 * m), (0.56 * m, 0.14 * m)), width=thin_w)
        draw.ellipse((0.56 * m, 0.14 * m, 0.82 * m, 0.30 * m), fill=255)
    elif shape == "balloon":
        draw.ellipse((0.28 * m, 0.12 * m, 0.72 * m, 0.60 * m), fill=255)
        draw.polygon([(0.46 * m, 0.58 * m), (0.54 * m, 0.58 * m), (0.50 * m, 0.70 * m)], fill=255)
        _draw_thick_line(
            draw,
            ((0.50 * m, 0.68 * m), (0.43 * m, 0.78 * m), (0.54 * m, 0.86 * m), (0.46 * m, 0.94 * m)),
            width=max(1, thin_w // 2),
        )
    elif shape == "shuriken":
        draw.polygon(
            [
                (0.50 * m, 0.06 * m),
                (0.60 * m, 0.36 * m),
                (0.94 * m, 0.50 * m),
                (0.60 * m, 0.64 * m),
                (0.50 * m, 0.94 * m),
                (0.40 * m, 0.64 * m),
                (0.06 * m, 0.50 * m),
                (0.40 * m, 0.36 * m),
            ],
            fill=255,
        )
        draw.ellipse((0.42 * m, 0.42 * m, 0.58 * m, 0.58 * m), fill=0)
    elif shape == "gear":
        points: list[tuple[float, float]] = []
        for i in range(24):
            radius = 0.42 * m if i % 3 == 1 else 0.32 * m
            angle = math.radians(-90.0 + 360.0 * float(i) / 24.0)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=255)
        draw.ellipse((0.38 * m, 0.38 * m, 0.62 * m, 0.62 * m), fill=0)
    elif shape == "puzzle_piece":
        draw.rounded_rectangle((0.20 * m, 0.24 * m, 0.78 * m, 0.82 * m), radius=int(0.06 * m), fill=255)
        draw.ellipse((0.42 * m, 0.10 * m, 0.58 * m, 0.30 * m), fill=255)
        draw.ellipse((0.72 * m, 0.44 * m, 0.92 * m, 0.60 * m), fill=255)
        draw.ellipse((0.10 * m, 0.44 * m, 0.30 * m, 0.60 * m), fill=0)
        draw.ellipse((0.42 * m, 0.72 * m, 0.58 * m, 0.92 * m), fill=0)
    elif shape == "lightbulb":
        for angle in (-160, -120, -60, -20, 20):
            a = math.radians(float(angle))
            _draw_thick_line(
                draw,
                (
                    (cx + 0.30 * m * math.cos(a), 0.38 * m + 0.30 * m * math.sin(a)),
                    (cx + 0.43 * m * math.cos(a), 0.38 * m + 0.43 * m * math.sin(a)),
                ),
                width=max(1, thin_w // 2),
            )
        draw.ellipse((0.26 * m, 0.14 * m, 0.74 * m, 0.62 * m), fill=255)
        draw.polygon([(0.38 * m, 0.56 * m), (0.62 * m, 0.56 * m), (0.58 * m, 0.72 * m), (0.42 * m, 0.72 * m)], fill=255)
        draw.rounded_rectangle((0.38 * m, 0.70 * m, 0.62 * m, 0.88 * m), radius=int(0.03 * m), fill=255)
        _draw_thick_line(draw, ((0.41 * m, 0.78 * m), (0.59 * m, 0.78 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "lantern":
        draw.arc((0.30 * m, 0.08 * m, 0.70 * m, 0.40 * m), start=180, end=360, fill=255, width=thin_w)
        draw.rounded_rectangle((0.28 * m, 0.30 * m, 0.72 * m, 0.78 * m), radius=int(0.08 * m), fill=255)
        draw.rounded_rectangle((0.36 * m, 0.42 * m, 0.64 * m, 0.66 * m), radius=int(0.05 * m), fill=0)
        _draw_thick_line(draw, ((0.34 * m, 0.30 * m), (0.34 * m, 0.38 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.66 * m, 0.30 * m), (0.66 * m, 0.38 * m)), width=thin_w)
        draw.rounded_rectangle((0.24 * m, 0.76 * m, 0.76 * m, 0.88 * m), radius=int(0.04 * m), fill=255)
    elif shape == "candle":
        draw.rounded_rectangle((0.42 * m, 0.36 * m, 0.58 * m, 0.84 * m), radius=int(0.035 * m), fill=255)
        draw.ellipse((0.40 * m, 0.12 * m, 0.60 * m, 0.34 * m), fill=255)
        draw.polygon([(0.50 * m, 0.06 * m), (0.40 * m, 0.26 * m), (0.60 * m, 0.26 * m)], fill=255)
        _draw_thick_line(draw, ((0.50 * m, 0.30 * m), (0.50 * m, 0.40 * m)), width=max(1, thin_w // 2))
        draw.rectangle((0.34 * m, 0.80 * m, 0.66 * m, 0.90 * m), fill=255)
    elif shape == "bottle":
        draw.rounded_rectangle((0.38 * m, 0.08 * m, 0.62 * m, 0.34 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.26 * m, 0.30 * m, 0.74 * m, 0.90 * m), radius=int(0.11 * m), fill=255)
        draw.rectangle((0.36 * m, 0.22 * m, 0.64 * m, 0.42 * m), fill=255)
        draw.rounded_rectangle((0.36 * m, 0.56 * m, 0.64 * m, 0.72 * m), radius=int(0.03 * m), fill=0)
    elif shape == "bucket":
        draw.polygon([(0.24 * m, 0.34 * m), (0.76 * m, 0.34 * m), (0.68 * m, 0.84 * m), (0.32 * m, 0.84 * m)], fill=255)
        draw.rounded_rectangle((0.20 * m, 0.28 * m, 0.80 * m, 0.42 * m), radius=int(0.04 * m), fill=255)
        draw.arc((0.28 * m, 0.12 * m, 0.72 * m, 0.58 * m), start=190, end=350, fill=255, width=thin_w)
    elif shape == "shovel":
        _draw_thick_line(draw, ((0.34 * m, 0.18 * m), (0.66 * m, 0.50 * m)), width=thin_w)
        draw.polygon([(0.58 * m, 0.48 * m), (0.86 * m, 0.62 * m), (0.72 * m, 0.88 * m), (0.48 * m, 0.64 * m)], fill=255)
        draw.ellipse((0.22 * m, 0.08 * m, 0.42 * m, 0.28 * m), fill=255)
        draw.ellipse((0.27 * m, 0.13 * m, 0.37 * m, 0.23 * m), fill=0)
    elif shape == "fork":
        _draw_thick_line(draw, ((0.32 * m, 0.86 * m), (0.58 * m, 0.42 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.56 * m, 0.44 * m), (0.78 * m, 0.22 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.52 * m, 0.38 * m), (0.70 * m, 0.20 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.62 * m, 0.48 * m), (0.84 * m, 0.26 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.54 * m, 0.48 * m), (0.62 * m, 0.56 * m)), width=max(1, thin_w // 2))
    elif shape == "spoon":
        draw.ellipse((0.54 * m, 0.10 * m, 0.86 * m, 0.42 * m), fill=255)
        _draw_thick_line(draw, ((0.60 * m, 0.40 * m), (0.28 * m, 0.86 * m)), width=thin_w)
    elif shape == "pizza_slice":
        draw.polygon([(0.18 * m, 0.18 * m), (0.86 * m, 0.34 * m), (0.42 * m, 0.88 * m)], fill=255)
        _draw_thick_line(draw, ((0.20 * m, 0.18 * m), (0.86 * m, 0.34 * m)), width=thin_w)
        for px, py in ((0.42, 0.38), (0.56, 0.50), (0.42, 0.62)):
            draw.ellipse(((px - 0.04) * m, (py - 0.04) * m, (px + 0.04) * m, (py + 0.04) * m), fill=0)
    elif shape == "egg":
        draw.ellipse((0.28 * m, 0.12 * m, 0.72 * m, 0.86 * m), fill=255)
        draw.ellipse((0.34 * m, 0.10 * m, 0.66 * m, 0.44 * m), fill=255)
    elif shape == "dice":
        draw.rounded_rectangle((0.18 * m, 0.18 * m, 0.82 * m, 0.82 * m), radius=int(0.08 * m), fill=255)
        for px, py in ((0.34, 0.34), (0.66, 0.34), (0.50, 0.50), (0.34, 0.66), (0.66, 0.66)):
            draw.ellipse(((px - 0.04) * m, (py - 0.04) * m, (px + 0.04) * m, (py + 0.04) * m), fill=0)
    elif shape == "microphone":
        draw.rounded_rectangle((0.34 * m, 0.12 * m, 0.66 * m, 0.58 * m), radius=int(0.15 * m), fill=255)
        draw.arc((0.24 * m, 0.34 * m, 0.76 * m, 0.76 * m), start=0, end=180, fill=255, width=thin_w)
        draw.rounded_rectangle((0.46 * m, 0.64 * m, 0.54 * m, 0.84 * m), radius=int(0.02 * m), fill=255)
        draw.rounded_rectangle((0.34 * m, 0.82 * m, 0.66 * m, 0.90 * m), radius=int(0.03 * m), fill=255)
    elif shape == "headphones":
        draw.arc((0.18 * m, 0.12 * m, 0.82 * m, 0.80 * m), start=190, end=350, fill=255, width=line_w)
        _draw_thick_line(draw, ((0.26 * m, 0.46 * m), (0.26 * m, 0.58 * m)), width=line_w)
        _draw_thick_line(draw, ((0.74 * m, 0.46 * m), (0.74 * m, 0.58 * m)), width=line_w)
        draw.rounded_rectangle((0.16 * m, 0.50 * m, 0.36 * m, 0.82 * m), radius=int(0.06 * m), fill=255)
        draw.rounded_rectangle((0.64 * m, 0.50 * m, 0.84 * m, 0.82 * m), radius=int(0.06 * m), fill=255)
    elif shape == "chair":
        draw.rounded_rectangle((0.24 * m, 0.18 * m, 0.44 * m, 0.64 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.34 * m, 0.54 * m, 0.78 * m, 0.70 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.34 * m, 0.44 * m, 0.84 * m, 0.56 * m), radius=int(0.04 * m), fill=255)
        _draw_thick_line(draw, ((0.40 * m, 0.68 * m), (0.32 * m, 0.88 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.72 * m, 0.68 * m), (0.80 * m, 0.88 * m)), width=thin_w)
    elif shape == "tent":
        draw.polygon([(0.50 * m, 0.14 * m), (0.88 * m, 0.86 * m), (0.12 * m, 0.86 * m)], fill=255)
        draw.polygon([(0.50 * m, 0.28 * m), (0.62 * m, 0.86 * m), (0.38 * m, 0.86 * m)], fill=0)
        _draw_thick_line(draw, ((0.50 * m, 0.20 * m), (0.50 * m, 0.86 * m)), width=max(1, thin_w // 2))
    elif shape == "phone":
        draw.rounded_rectangle((0.34 * m, 0.10 * m, 0.66 * m, 0.90 * m), radius=int(0.07 * m), fill=255)
        draw.rounded_rectangle((0.39 * m, 0.20 * m, 0.61 * m, 0.76 * m), radius=int(0.03 * m), fill=0)
        draw.ellipse((0.46 * m, 0.80 * m, 0.54 * m, 0.88 * m), fill=0)
    elif shape == "television":
        draw.rounded_rectangle((0.14 * m, 0.26 * m, 0.86 * m, 0.72 * m), radius=int(0.05 * m), fill=255)
        draw.rounded_rectangle((0.24 * m, 0.36 * m, 0.76 * m, 0.62 * m), radius=int(0.03 * m), fill=0)
        _draw_thick_line(draw, ((0.40 * m, 0.72 * m), (0.30 * m, 0.88 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.60 * m, 0.72 * m), (0.70 * m, 0.88 * m)), width=thin_w)
    elif shape == "laptop":
        draw.rounded_rectangle((0.24 * m, 0.22 * m, 0.76 * m, 0.60 * m), radius=int(0.04 * m), fill=255)
        draw.rectangle((0.32 * m, 0.32 * m, 0.68 * m, 0.52 * m), fill=0)
        draw.polygon([(0.18 * m, 0.62 * m), (0.82 * m, 0.62 * m), (0.92 * m, 0.82 * m), (0.08 * m, 0.82 * m)], fill=255)
        draw.rectangle((0.40 * m, 0.66 * m, 0.60 * m, 0.72 * m), fill=0)
    elif shape == "watch":
        draw.rounded_rectangle((0.40 * m, 0.08 * m, 0.60 * m, 0.32 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.40 * m, 0.68 * m, 0.60 * m, 0.92 * m), radius=int(0.04 * m), fill=255)
        draw.ellipse((0.26 * m, 0.26 * m, 0.74 * m, 0.74 * m), fill=255)
        draw.ellipse((0.36 * m, 0.36 * m, 0.64 * m, 0.64 * m), fill=0)
        _draw_thick_line(draw, ((0.50 * m, 0.50 * m), (0.50 * m, 0.38 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.50 * m, 0.50 * m), (0.60 * m, 0.54 * m)), width=max(1, thin_w // 2))
    elif shape == "bus":
        draw.rounded_rectangle((0.16 * m, 0.28 * m, 0.84 * m, 0.74 * m), radius=int(0.07 * m), fill=255)
        draw.rectangle((0.24 * m, 0.38 * m, 0.42 * m, 0.52 * m), fill=0)
        draw.rectangle((0.48 * m, 0.38 * m, 0.76 * m, 0.52 * m), fill=0)
        draw.ellipse((0.24 * m, 0.68 * m, 0.38 * m, 0.82 * m), fill=255)
        draw.ellipse((0.62 * m, 0.68 * m, 0.76 * m, 0.82 * m), fill=255)
    elif shape == "train":
        draw.rounded_rectangle((0.24 * m, 0.16 * m, 0.76 * m, 0.74 * m), radius=int(0.08 * m), fill=255)
        draw.rectangle((0.34 * m, 0.28 * m, 0.66 * m, 0.46 * m), fill=0)
        draw.ellipse((0.30 * m, 0.62 * m, 0.44 * m, 0.76 * m), fill=0)
        draw.ellipse((0.56 * m, 0.62 * m, 0.70 * m, 0.76 * m), fill=0)
        draw.polygon([(0.34 * m, 0.74 * m), (0.66 * m, 0.74 * m), (0.76 * m, 0.90 * m), (0.24 * m, 0.90 * m)], fill=255)
    elif shape == "bicycle":
        draw.ellipse((0.16 * m, 0.58 * m, 0.40 * m, 0.82 * m), fill=255)
        draw.ellipse((0.22 * m, 0.64 * m, 0.34 * m, 0.76 * m), fill=0)
        draw.ellipse((0.60 * m, 0.58 * m, 0.84 * m, 0.82 * m), fill=255)
        draw.ellipse((0.66 * m, 0.64 * m, 0.78 * m, 0.76 * m), fill=0)
        _draw_thick_line(draw, ((0.28 * m, 0.68 * m), (0.48 * m, 0.44 * m), (0.70 * m, 0.68 * m), (0.42 * m, 0.68 * m), (0.48 * m, 0.44 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.48 * m, 0.44 * m), (0.58 * m, 0.36 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.62 * m, 0.36 * m), (0.74 * m, 0.36 * m)), width=max(1, thin_w // 2))
    elif shape == "bed":
        draw.rounded_rectangle((0.14 * m, 0.44 * m, 0.86 * m, 0.70 * m), radius=int(0.04 * m), fill=255)
        draw.rectangle((0.14 * m, 0.28 * m, 0.28 * m, 0.82 * m), fill=255)
        draw.rounded_rectangle((0.28 * m, 0.36 * m, 0.48 * m, 0.50 * m), radius=int(0.04 * m), fill=255)
        _draw_thick_line(draw, ((0.18 * m, 0.70 * m), (0.18 * m, 0.88 * m)), width=thin_w)
        _draw_thick_line(draw, ((0.80 * m, 0.70 * m), (0.80 * m, 0.88 * m)), width=thin_w)
    elif shape == "lamp":
        draw.polygon([(0.34 * m, 0.14 * m), (0.66 * m, 0.14 * m), (0.78 * m, 0.44 * m), (0.22 * m, 0.44 * m)], fill=255)
        draw.rounded_rectangle((0.46 * m, 0.44 * m, 0.54 * m, 0.78 * m), radius=int(0.03 * m), fill=255)
        draw.rounded_rectangle((0.30 * m, 0.76 * m, 0.70 * m, 0.88 * m), radius=int(0.04 * m), fill=255)
    elif shape == "door":
        draw.rounded_rectangle((0.28 * m, 0.12 * m, 0.74 * m, 0.90 * m), radius=int(0.04 * m), fill=255)
        draw.ellipse((0.62 * m, 0.48 * m, 0.70 * m, 0.56 * m), fill=0)
        draw.rectangle((0.22 * m, 0.86 * m, 0.80 * m, 0.94 * m), fill=255)
    elif shape == "window":
        draw.rounded_rectangle((0.18 * m, 0.18 * m, 0.82 * m, 0.82 * m), radius=int(0.04 * m), fill=255)
        draw.rectangle((0.28 * m, 0.28 * m, 0.46 * m, 0.46 * m), fill=0)
        draw.rectangle((0.54 * m, 0.28 * m, 0.72 * m, 0.46 * m), fill=0)
        draw.rectangle((0.28 * m, 0.54 * m, 0.46 * m, 0.72 * m), fill=0)
        draw.rectangle((0.54 * m, 0.54 * m, 0.72 * m, 0.72 * m), fill=0)
    elif shape == "mailbox":
        draw.pieslice((0.18 * m, 0.22 * m, 0.70 * m, 0.74 * m), start=180, end=360, fill=255)
        draw.rectangle((0.18 * m, 0.48 * m, 0.70 * m, 0.74 * m), fill=255)
        draw.rounded_rectangle((0.68 * m, 0.38 * m, 0.86 * m, 0.48 * m), radius=int(0.02 * m), fill=255)
        _draw_thick_line(draw, ((0.42 * m, 0.74 * m), (0.42 * m, 0.90 * m)), width=thin_w)
    elif shape == "toothbrush":
        _draw_thick_line(draw, ((0.18 * m, 0.78 * m), (0.64 * m, 0.32 * m)), width=thin_w)
        draw.rounded_rectangle((0.58 * m, 0.20 * m, 0.84 * m, 0.42 * m), radius=int(0.035 * m), fill=255)
        for x0, y0 in ((0.61, 0.18), (0.66, 0.16), (0.71, 0.16), (0.76, 0.18), (0.81, 0.20)):
            _draw_thick_line(draw, ((x0 * m, y0 * m), ((x0 - 0.08) * m, (y0 - 0.08) * m)), width=max(1, thin_w // 3))
        draw.ellipse((0.27 * m, 0.67 * m, 0.33 * m, 0.73 * m), fill=0)
    elif shape == "broom":
        _draw_thick_line(draw, ((0.30 * m, 0.14 * m), (0.62 * m, 0.58 * m)), width=thin_w)
        draw.polygon([(0.52 * m, 0.54 * m), (0.84 * m, 0.66 * m), (0.70 * m, 0.90 * m), (0.38 * m, 0.74 * m)], fill=255)
        for offset in (0.48, 0.58, 0.68):
            _draw_thick_line(draw, ((offset * m, 0.72 * m), ((offset - 0.06) * m, 0.82 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "trash_can":
        draw.rounded_rectangle((0.30 * m, 0.28 * m, 0.70 * m, 0.86 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.24 * m, 0.20 * m, 0.76 * m, 0.34 * m), radius=int(0.03 * m), fill=255)
        draw.rounded_rectangle((0.40 * m, 0.10 * m, 0.60 * m, 0.22 * m), radius=int(0.03 * m), fill=255)
        for x in (0.40, 0.50, 0.60):
            _draw_thick_line(draw, ((x * m, 0.42 * m), (x * m, 0.76 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "teapot":
        draw.ellipse((0.22 * m, 0.34 * m, 0.70 * m, 0.80 * m), fill=255)
        draw.ellipse((0.60 * m, 0.42 * m, 0.90 * m, 0.72 * m), fill=255)
        draw.ellipse((0.68 * m, 0.50 * m, 0.82 * m, 0.64 * m), fill=0)
        draw.polygon([(0.24 * m, 0.48 * m), (0.04 * m, 0.38 * m), (0.20 * m, 0.64 * m)], fill=255)
        draw.rounded_rectangle((0.38 * m, 0.22 * m, 0.58 * m, 0.36 * m), radius=int(0.04 * m), fill=255)
    elif shape == "knife":
        _draw_thick_line(draw, ((0.18 * m, 0.78 * m), (0.42 * m, 0.54 * m)), width=line_w)
        draw.polygon([(0.36 * m, 0.50 * m), (0.88 * m, 0.14 * m), (0.70 * m, 0.56 * m), (0.44 * m, 0.66 * m)], fill=255)
        _draw_thick_line(draw, ((0.46 * m, 0.52 * m), (0.70 * m, 0.36 * m)), width=max(1, thin_w // 2), fill=0)
    elif shape == "soccer_ball":
        draw.ellipse((0.16 * m, 0.16 * m, 0.84 * m, 0.84 * m), fill=255)
        draw.polygon(_regular_polygon_points(cx, cy, 0.12 * m, 5, start_degrees=-90), fill=0)
        for angle in range(-90, 270, 72):
            a = math.radians(float(angle))
            px = cx + 0.27 * m * math.cos(a)
            py = cy + 0.27 * m * math.sin(a)
            draw.polygon(_regular_polygon_points(px, py, 0.075 * m, 5, start_degrees=float(angle)), fill=0)
    elif shape == "rugby_ball":
        draw.ellipse((0.12 * m, 0.28 * m, 0.88 * m, 0.72 * m), fill=255)
        _draw_thick_line(draw, ((0.24 * m, 0.50 * m), (0.76 * m, 0.50 * m)), width=max(1, thin_w // 2), fill=0)
        for x in (0.42, 0.50, 0.58):
            _draw_thick_line(draw, ((x * m, 0.46 * m), (x * m, 0.54 * m)), width=max(1, thin_w // 3), fill=0)
    elif shape == "dumbbell":
        draw.rounded_rectangle((0.12 * m, 0.38 * m, 0.28 * m, 0.62 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.72 * m, 0.38 * m, 0.88 * m, 0.62 * m), radius=int(0.04 * m), fill=255)
        draw.rounded_rectangle((0.26 * m, 0.32 * m, 0.38 * m, 0.68 * m), radius=int(0.03 * m), fill=255)
        draw.rounded_rectangle((0.62 * m, 0.32 * m, 0.74 * m, 0.68 * m), radius=int(0.03 * m), fill=255)
        draw.rounded_rectangle((0.34 * m, 0.46 * m, 0.66 * m, 0.54 * m), radius=int(0.02 * m), fill=255)
    elif shape == "calculator":
        draw.rounded_rectangle((0.28 * m, 0.12 * m, 0.72 * m, 0.88 * m), radius=int(0.05 * m), fill=255)
        draw.rectangle((0.36 * m, 0.22 * m, 0.64 * m, 0.34 * m), fill=0)
        for px, py in ((0.38, 0.46), (0.50, 0.46), (0.62, 0.46), (0.38, 0.60), (0.50, 0.60), (0.62, 0.60), (0.38, 0.74), (0.50, 0.74), (0.62, 0.74)):
            draw.rectangle((px * m, py * m, (px + 0.06) * m, (py + 0.06) * m), fill=0)
    elif shape == "plug":
        draw.polygon([(0.36 * m, 0.44 * m), (0.54 * m, 0.26 * m), (0.76 * m, 0.48 * m), (0.58 * m, 0.66 * m)], fill=255)
        _draw_thick_line(draw, ((0.44 * m, 0.30 * m), (0.30 * m, 0.16 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.56 * m, 0.18 * m), (0.42 * m, 0.04 * m)), width=max(1, thin_w // 2))
        _draw_thick_line(draw, ((0.56 * m, 0.66 * m), (0.40 * m, 0.82 * m), (0.48 * m, 0.94 * m)), width=thin_w)
    elif shape == "broccoli":
        draw.rounded_rectangle((0.42 * m, 0.56 * m, 0.58 * m, 0.88 * m), radius=int(0.05 * m), fill=255)
        draw.polygon([(0.42 * m, 0.60 * m), (0.28 * m, 0.82 * m), (0.46 * m, 0.76 * m)], fill=255)
        draw.polygon([(0.58 * m, 0.60 * m), (0.72 * m, 0.82 * m), (0.54 * m, 0.76 * m)], fill=255)
        for px, py, rr in ((0.32, 0.34, 0.17), (0.50, 0.26, 0.18), (0.68, 0.34, 0.17), (0.42, 0.46, 0.17), (0.58, 0.46, 0.17)):
            draw.ellipse(((px - rr) * m, (py - rr) * m, (px + rr) * m, (py + rr) * m), fill=255)
        draw.ellipse((0.40 * m, 0.62 * m, 0.60 * m, 0.78 * m), fill=255)
    elif shape == "cactus":
        draw.rounded_rectangle((0.42 * m, 0.16 * m, 0.58 * m, 0.90 * m), radius=int(0.08 * m), fill=255)
        draw.rounded_rectangle((0.18 * m, 0.38 * m, 0.34 * m, 0.70 * m), radius=int(0.08 * m), fill=255)
        draw.rounded_rectangle((0.26 * m, 0.56 * m, 0.46 * m, 0.70 * m), radius=int(0.07 * m), fill=255)
        draw.rounded_rectangle((0.66 * m, 0.30 * m, 0.82 * m, 0.62 * m), radius=int(0.08 * m), fill=255)
        draw.rounded_rectangle((0.54 * m, 0.48 * m, 0.74 * m, 0.62 * m), radius=int(0.07 * m), fill=255)
        for x in (0.49, 0.54):
            _draw_thick_line(draw, ((x * m, 0.24 * m), (x * m, 0.82 * m)), width=max(1, thin_w // 3), fill=0)
    elif shape == "guitar":
        draw.ellipse((0.10 * m, 0.34 * m, 0.54 * m, 0.82 * m), fill=255)
        draw.ellipse((0.34 * m, 0.24 * m, 0.66 * m, 0.58 * m), fill=255)
        draw.ellipse((0.32 * m, 0.50 * m, 0.46 * m, 0.64 * m), fill=0)
        draw.rounded_rectangle((0.58 * m, 0.34 * m, 0.90 * m, 0.46 * m), radius=int(0.035 * m), fill=255)
        draw.rounded_rectangle((0.82 * m, 0.28 * m, 0.96 * m, 0.52 * m), radius=int(0.04 * m), fill=255)
        draw.rectangle((0.86 * m, 0.36 * m, 0.96 * m, 0.44 * m), fill=0)
        for y in (0.37, 0.41, 0.45):
            _draw_thick_line(draw, ((0.46 * m, 0.56 * m), (0.86 * m, y * m)), width=max(1, thin_w // 4), fill=0)
    elif shape == "acorn":
        draw.polygon([(0.28 * m, 0.46 * m), (0.72 * m, 0.46 * m), (0.68 * m, 0.76 * m), (0.50 * m, 0.94 * m), (0.32 * m, 0.76 * m)], fill=255)
        draw.ellipse((0.24 * m, 0.34 * m, 0.76 * m, 0.82 * m), fill=255)
        draw.pieslice((0.16 * m, 0.18 * m, 0.84 * m, 0.62 * m), start=180, end=360, fill=255)
        draw.rounded_rectangle((0.46 * m, 0.08 * m, 0.56 * m, 0.24 * m), radius=int(0.03 * m), fill=255)
        _draw_thick_line(draw, ((0.22 * m, 0.48 * m), (0.78 * m, 0.48 * m)), width=max(1, thin_w // 2), fill=0)
        for x in (0.28, 0.42, 0.56):
            _draw_thick_line(draw, ((x * m, 0.30 * m), ((x + 0.18) * m, 0.46 * m)), width=max(1, thin_w // 3), fill=0)
    else:  # pragma: no cover - exhaustive over the checked vocabulary.
        raise AssertionError(shape)

    return mask


def _tight_alpha_crop(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("procedural named icon has empty alpha")
    return image.crop(bbox)


def _pattern_rgb_for_tint(tint_rgb: RGB) -> RGB:
    red, green, blue = (int(value) for value in tint_rgb)
    luminance = 0.2126 * float(red) + 0.7152 * float(green) + 0.0722 * float(blue)
    if luminance < 145.0:
        return (255, 255, 255)
    return (24, 28, 32)


def _clip_overlay_to_alpha(overlay: Image.Image, alpha: Image.Image) -> Image.Image:
    clipped = overlay.convert("RGBA")
    clipped_alpha = ImageChops.multiply(clipped.getchannel("A"), alpha)
    clipped.putalpha(clipped_alpha)
    return clipped


def _apply_fill_style(image: Image.Image, *, fill_style: str, tint_rgb: RGB) -> Image.Image:
    style = str(fill_style or "solid")
    if style not in PROCEDURAL_NAMED_ICON_FILL_STYLES:
        raise ValueError(f"unsupported procedural named icon fill style: {fill_style}")
    if style == "solid":
        return image

    rgba = image.convert("RGBA")
    width, height = rgba.size
    alpha = rgba.getchannel("A")
    pattern_rgb = _pattern_rgb_for_tint(tuple(int(value) for value in tint_rgb))

    if style == "striped":
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        spacing = max(16, int(round(min(width, height) * 0.38)))
        stripe_width = max(1, int(round(min(width, height) * 0.022)))
        for offset in range(-height, width + height + spacing, spacing):
            draw.line(
                (int(offset), int(height), int(offset + height), 0),
                fill=tuple(int(value) for value in pattern_rgb) + (132,),
                width=int(stripe_width),
            )
        rgba.alpha_composite(_clip_overlay_to_alpha(overlay, alpha))
        return rgba

    if style == "dotted":
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        spacing = max(18, int(round(min(width, height) * 0.44)))
        radius = max(1, int(round(min(width, height) * 0.022)))
        for y in range(spacing // 2, height, spacing):
            for x in range(spacing // 2, width, spacing):
                draw.ellipse(
                    (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
                    fill=tuple(int(value) for value in pattern_rgb) + (150,),
                )
        rgba.alpha_composite(_clip_overlay_to_alpha(overlay, alpha))
        return rgba

    raise AssertionError(style)


@lru_cache(maxsize=4096)
def _render_base_procedural_named_icon(shape_id: str, size_px: int) -> Image.Image:
    size = max(12, int(size_px))
    scale = 4
    mask = _draw_shape_mask(str(shape_id), size=int(size) * scale)
    mask = mask.resize((int(size), int(size)), Image.Resampling.LANCZOS)
    base = Image.new("RGBA", (int(size), int(size)), (255, 255, 255, 0))
    base.putalpha(mask)
    return base


def render_procedural_named_icon_rgba(
    *,
    shape_id: str,
    size_px: int,
    tint_rgb: RGB,
    fill_style: str = "solid",
    rotation_degrees: int = 0,
    mirror_x: bool = False,
    noise_edits: Sequence[NoiseEdit] = (),
    noise_seed: int | None = None,
) -> Image.Image:
    """Render one cropped, tinted procedural named icon as RGBA."""

    base = _render_base_procedural_named_icon(str(shape_id), max(12, int(size_px)))
    tinted = Image.new("RGBA", base.size, tuple(int(value) for value in tint_rgb) + (0,))
    tinted.putalpha(base.getchannel("A"))
    tinted = _apply_fill_style(tinted, fill_style=str(fill_style), tint_rgb=tuple(int(value) for value in tint_rgb))
    if bool(mirror_x):
        tinted = ImageOps.mirror(tinted)
    rotation = int(rotation_degrees) % 360
    if rotation:
        tinted = tinted.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    cropped = _tight_alpha_crop(tinted)
    if noise_edits:
        import random

        cropped = apply_icon_noise_edits_rgba(
            cropped,
            edits=tuple(noise_edits),
            rng=random.Random(int(noise_seed if noise_seed is not None else 0)),
        )
    return cropped


__all__ = [
    "PROCEDURAL_NAMED_ICON_DISPLAY_NAMES",
    "DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS",
    "PROCEDURAL_NAMED_ICON_FILL_STYLE_DISPLAY_NAMES",
    "PROCEDURAL_NAMED_ICON_FILL_STYLES",
    "PROCEDURAL_NAMED_ICON_SHAPES",
    "procedural_named_icon_display_name",
    "procedural_named_icon_fill_style_display_name",
    "procedural_named_icon_fill_style_probability_map",
    "render_procedural_named_icon_rgba",
    "sample_procedural_named_icon_fill_style",
    "validate_procedural_named_icon_fill_style_support",
]
