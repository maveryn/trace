"""Pure color helpers for composition-style chart renderers."""

from __future__ import annotations

import colorsys
from typing import Sequence, Tuple

from trace_tasks.core.seed import spawn_rng


RGB = Tuple[int, int, int]


def _channel(value: float) -> int:
    return max(0, min(255, int(round(float(value)))))


def lighten_rgb(color: Sequence[int], amount: float) -> RGB:
    """Move an RGB color toward white by ``amount``."""

    factor = max(0.0, min(1.0, float(amount)))
    return tuple(
        _channel(float(channel) + (255.0 - float(channel)) * float(factor))
        for channel in color[:3]
    )  # type: ignore[return-value]


def darken_rgb(color: Sequence[int], amount: float) -> RGB:
    """Move an RGB color toward black by ``amount``."""

    factor = max(0.0, min(1.0, float(amount)))
    return tuple(
        _channel(float(channel) * (1.0 - float(factor)))
        for channel in color[:3]
    )  # type: ignore[return-value]


def composition_hsv_color(
    index: int,
    count: int,
    *,
    instance_seed: int,
    namespace: str,
    saturation_base: float,
    saturation_jitter: float,
    value_base: float,
    value_jitter: float,
) -> RGB:
    """Return one deterministic high-separation HSV color for hierarchy roots."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    offset = rng.random()
    hue = (float(offset) + float(index) / max(1.0, float(count))) % 1.0
    sat = float(saturation_base) + float(saturation_jitter) * rng.random()
    val = float(value_base) + float(value_jitter) * rng.random()
    red, green, blue = colorsys.hsv_to_rgb(float(hue), float(sat), float(val))
    return int(red * 255), int(green * 255), int(blue * 255)


__all__ = ["RGB", "composition_hsv_color", "darken_rgb", "lighten_rgb"]
