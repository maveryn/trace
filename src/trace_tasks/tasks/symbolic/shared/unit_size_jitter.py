"""Shared unit-size jitter helpers for repeated-unit symbolic renderers."""

from __future__ import annotations

from typing import Any, Mapping

from ...shared.config_defaults import group_default
from ...shared.render_variation import resolve_render_float


def resolve_symbolic_unit_size_scale(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int | None,
    namespace: str,
    key: str = "unit_size_scale",
    fallback_min: float = 0.5,
    fallback_max: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Resolve deterministic render-only repeated-unit size scaling."""

    min_key = f"{str(key)}_min"
    max_key = f"{str(key)}_max"
    low = float(params.get(min_key, group_default(defaults, min_key, float(fallback_min))))
    high = float(params.get(max_key, group_default(defaults, max_key, float(fallback_max))))
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"{key}_min and {key}_max must be positive")
    if low > high:
        raise ValueError(f"{key}_min must be <= {key}_max")
    if (high / low) < 2.0:
        raise ValueError(f"{key} range must span at least 2x")
    scale = float(
        resolve_render_float(
            params,
            defaults,
            str(key),
            1.0,
            instance_seed=instance_seed,
            namespace=str(namespace),
            steps=1000,
        )
    )
    return scale, {
        "enabled": True,
        "key": str(key),
        "min": float(low),
        "max": float(high),
        "scale": float(scale),
        "range_ratio": float(high / low),
    }


def scale_symbolic_px(value: int | float, scale: float, *, min_px: int = 1) -> int:
    """Scale a pixel value while preserving a readable lower bound."""

    return max(int(min_px), int(round(float(value) * float(scale))))


def with_symbolic_unit_size_jitter(
    mapping: Mapping[str, Any],
    unit_size_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a metadata dict with nested unit-size jitter information."""

    resolved = dict(mapping)
    resolved["unit_size_jitter"] = dict(unit_size_meta)
    return resolved


__all__ = [
    "resolve_symbolic_unit_size_scale",
    "scale_symbolic_px",
    "with_symbolic_unit_size_jitter",
]
