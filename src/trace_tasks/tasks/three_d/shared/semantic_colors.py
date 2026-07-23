"""Shared semantic-color helpers for 3D color-readout tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple


COLOR_CONFUSION_EXCLUSIONS: Mapping[str, Tuple[str, ...]] = {
    "blue": ("cyan", "purple"),
    "red": ("maroon", "magenta"),
    "yellow": ("orange", "brown"),
    "orange": ("yellow", "brown"),
    "cyan": ("blue",),
    "maroon": ("red", "magenta"),
    "magenta": ("red", "maroon"),
    "brown": ("yellow", "orange"),
}


def confusable_color_names(color_name: str) -> Tuple[str, ...]:
    """Return named colors too close to one semantic target for color readout."""

    return tuple(str(color) for color in COLOR_CONFUSION_EXCLUSIONS.get(str(color_name), ()))


def colors_conflict(left: str, right: str) -> bool:
    """Return whether two named colors should not co-occur in a readout palette."""

    return (
        str(right) in set(confusable_color_names(str(left)))
        or str(left) in set(confusable_color_names(str(right)))
    )


def compatible_color_names(
    support: Sequence[str],
    *,
    anchors: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Return colors after removing anchor colors and their near-confusers."""

    blocked = {str(color) for color in exclude}
    for anchor in anchors:
        blocked.add(str(anchor))
        blocked.update(confusable_color_names(str(anchor)))
    pool = tuple(str(color) for color in support if str(color) not in blocked)
    if not pool:
        raise ValueError("semantic color readout needs at least one compatible color")
    return pool


def sample_readout_palette(
    rng: Any,
    *,
    target_color: str,
    support: Sequence[str],
    size: int,
) -> Tuple[str, ...]:
    """Sample a target-inclusive color palette without target-confusable colors."""

    target = str(target_color)
    candidates = list(compatible_color_names(tuple(str(color) for color in support), anchors=(target,)))
    rng.shuffle(candidates)
    selected = [target, *candidates[: max(0, int(size) - 1)]]
    rng.shuffle(selected)
    return tuple(selected[: int(size)])


__all__ = [
    "COLOR_CONFUSION_EXCLUSIONS",
    "colors_conflict",
    "compatible_color_names",
    "confusable_color_names",
    "sample_readout_palette",
]
