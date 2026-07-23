"""Low-level shape and color helpers for match-3 rendering."""

from __future__ import annotations

from typing import Tuple


def blend_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    """Blend two RGB colors using a clamped foreground alpha."""

    weight = max(0.0, min(1.0, float(alpha)))
    return tuple(int(round((float(a[index]) * weight) + (float(b[index]) * (1.0 - weight)))) for index in range(3))


def gem_polygon(cx: float, cy: float, radius: float) -> Tuple[Tuple[float, float], ...]:
    """Return a five-point faceted jewel polygon around one center."""

    return (
        (float(cx), float(cy - radius)),
        (float(cx + radius * 0.90), float(cy - radius * 0.08)),
        (float(cx + radius * 0.42), float(cy + radius * 0.86)),
        (float(cx - radius * 0.42), float(cy + radius * 0.86)),
        (float(cx - radius * 0.90), float(cy - radius * 0.08)),
    )


__all__ = ["blend_rgb", "gem_polygon"]
