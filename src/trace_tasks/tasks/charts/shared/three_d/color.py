"""Pure RGB helpers for 3D chart depth and value rendering."""

from __future__ import annotations

from typing import Tuple

RGB = Tuple[int, int, int]


def _channel(value: float) -> int:
    return max(0, min(255, int(round(float(value)))))


def shade_rgb(color: RGB, factor: float) -> RGB:
    """Scale an RGB color by ``factor`` with deterministic rounding."""

    return tuple(_channel(float(channel) * float(factor)) for channel in color)


def lighten_rgb(color: RGB, amount: float) -> RGB:
    """Move an RGB color toward white by ``amount``."""

    return tuple(
        _channel(float(channel) + (255.0 - float(channel)) * float(amount))
        for channel in color
    )


def blend_rgb(low: RGB, high: RGB, value: float) -> RGB:
    """Linearly blend two RGB colors with ``value`` clamped to [0, 1]."""

    weight = max(0.0, min(1.0, float(value)))
    return tuple(
        _channel((1.0 - weight) * float(left) + weight * float(right))
        for left, right in zip(low, high)
    )


__all__ = ["RGB", "blend_rgb", "lighten_rgb", "shade_rgb"]
