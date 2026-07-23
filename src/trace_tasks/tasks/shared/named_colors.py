"""Cross-domain canonical named-color palette helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


Color = Tuple[int, int, int]
NamedColor = Tuple[str, Color]


_NAMED_COLOR_PALETTE: Sequence[NamedColor] = (
    ("red", (230, 50, 50)),
    ("blue", (45, 117, 230)),
    ("green", (55, 185, 75)),
    ("yellow", (212, 194, 30)),
    ("orange", (238, 136, 26)),
    ("purple", (150, 58, 202)),
    ("brown", (136, 112, 68)),
    ("cyan", (52, 196, 224)),
    ("magenta", (208, 44, 145)),
    ("maroon", (150, 54, 68)),
)


def available_named_colors() -> Sequence[NamedColor]:
    """Return the canonical repo-wide named-color palette."""

    return tuple(_NAMED_COLOR_PALETTE)


def named_color(name: str) -> Color:
    """Return one canonical named color by case-insensitive name."""

    needle = str(name).strip().lower()
    for entry_name, rgb in _NAMED_COLOR_PALETTE:
        if str(entry_name).strip().lower() == needle:
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    raise KeyError(f"unknown named color: {name}")


def sample_named_color(rng, *, exclude_names: Iterable[str] = ()) -> NamedColor:
    """Sample one deterministic named color from the shared palette."""

    sampled = sample_named_color_palette(rng, palette_size=1, exclude_names=exclude_names)
    if not sampled:
        raise ValueError("no named colors available after exclusions")
    name, rgb = sampled[0]
    return (str(name), (int(rgb[0]), int(rgb[1]), int(rgb[2])))


def sample_named_color_palette(rng, *, palette_size: int, exclude_names: Iterable[str] = ()) -> List[NamedColor]:
    """Sample one deterministic subset of the shared named-color palette."""

    excluded = {str(name).strip().lower() for name in exclude_names if str(name).strip()}
    candidates = [entry for entry in _NAMED_COLOR_PALETTE if str(entry[0]).lower() not in excluded]
    size = max(0, min(int(palette_size), len(candidates)))
    if size <= 0:
        return []
    sampled = list(rng.sample(candidates, k=int(size)))
    return [(str(name), (int(rgb[0]), int(rgb[1]), int(rgb[2]))) for name, rgb in sampled]


def darken_color(color: Sequence[int], *, factor: float = 0.55) -> Color:
    """Return a darker version of an RGB color by scaling channels toward black."""

    scale = max(0.0, min(1.0, float(factor)))
    if len(color) < 3:
        raise ValueError("darken_color requires three channels")
    return (
        max(0, min(255, int(round(int(color[0]) * scale)))),
        max(0, min(255, int(round(int(color[1]) * scale)))),
        max(0, min(255, int(round(int(color[2]) * scale)))),
    )


__all__ = [
    "Color",
    "NamedColor",
    "available_named_colors",
    "darken_color",
    "named_color",
    "sample_named_color",
    "sample_named_color_palette",
]
