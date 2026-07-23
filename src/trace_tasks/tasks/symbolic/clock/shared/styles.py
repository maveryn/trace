"""Style and render-parameter helpers for symbolic clocks."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Mapping

from .....core.seed import hash64

from .state import ClockRenderParams


def resolve_clock_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    fallback_values: Mapping[str, Any],
    instance_seed: int | None = None,
) -> ClockRenderParams:
    """Resolve concrete clock render parameters from config and fallbacks."""

    def _render_value(key: str, fallback: Any) -> Any:
        if params.get(str(key)) is not None:
            return params[str(key)]
        low_raw = params.get(f"{str(key)}_min", render_defaults.get(f"{str(key)}_min"))
        high_raw = params.get(f"{str(key)}_max", render_defaults.get(f"{str(key)}_max"))
        if low_raw is not None or high_raw is not None:
            default_value = int(render_defaults.get(str(key), fallback))
            low = int(default_value if low_raw is None else low_raw)
            high = int(default_value if high_raw is None else high_raw)
            if int(low) > int(high):
                raise ValueError(f"{str(key)}_min must be <= {str(key)}_max")
            seed = 0 if instance_seed is None else int(instance_seed)
            index = abs(int(hash64(int(seed), f"clock_render:{str(key)}", 52289)))
            return int(low) + int(index % (int(high) - int(low) + 1))
        return params.get(str(key), render_defaults.get(str(key), fallback))

    values = {
        field.name: int(_render_value(field.name, fallback_values[field.name]))
        for field in fields(ClockRenderParams)
    }
    return ClockRenderParams(**values)


def scale_clock_render_params_for_radius(base_params: ClockRenderParams, *, radius_px: int) -> ClockRenderParams:
    """Scale analog-clock render details for one target face radius."""

    scale = float(radius_px) / 90.0
    return replace(
        base_params,
        face_radius_px=int(radius_px),
        bezel_width_px=max(4, int(round(8 * scale))),
        numeral_font_size_px=max(12, int(round(17 * scale))),
        major_tick_length_px=max(8, int(round(13 * scale))),
        minor_tick_length_px=max(4, int(round(6 * scale))),
        major_tick_width_px=max(2, int(round(3 * scale))),
        minor_tick_width_px=max(1, int(round(2 * scale))),
        minor_tick_dot_radius_px=max(2, int(round(2 * scale))),
        hour_hand_width_px=max(5, int(round(8 * scale))),
        minute_hand_width_px=max(4, int(round(6 * scale))),
        second_hand_width_px=max(2, int(round(2 * scale))),
        hand_bbox_padding_px=max(4, int(round(5 * scale))),
        center_dot_radius_px=max(4, int(round(6 * scale))),
        inner_ring_inset_px=max(9, int(round(13 * scale))),
        inner_ring_width_px=max(2, int(round(3 * scale))),
    )


__all__ = ["resolve_clock_render_params", "scale_clock_render_params_for_radius"]
