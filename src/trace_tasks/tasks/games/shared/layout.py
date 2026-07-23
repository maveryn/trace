"""Shared layout variation helpers for games-domain renderers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from ...shared.config_defaults import group_default
from ...shared.render_variation import apply_resolved_layout_jitter_to_margins, resolve_layout_jitter, resolve_render_float


BBox = Tuple[float, float, float, float]


def resolve_games_layout_jitter(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int | None,
    namespace: str,
) -> dict[str, Any]:
    """Resolve deterministic nonsemantic layout jitter for one game instance."""

    return resolve_layout_jitter(
        params,
        defaults,
        instance_seed=instance_seed,
        namespace=str(namespace),
    )


def apply_games_layout_jitter_to_bbox(
    *,
    bbox_px: Sequence[float],
    canvas_width: int,
    canvas_height: int,
    jitter: Mapping[str, Any] | None,
) -> tuple[BBox, float, float, dict[str, Any]]:
    """Shift one artifact bbox while preserving its size and clamping to margins."""

    if len(bbox_px) != 4:
        raise ValueError("bbox_px must contain four coordinates")
    left, top, right, bottom = [float(value) for value in bbox_px]
    shifted_left, _shifted_right_margin, shifted_top, _shifted_bottom_margin, resolved = (
        apply_resolved_layout_jitter_to_margins(
            left_px=left,
            right_px=float(canvas_width) - right,
            top_px=top,
            bottom_px=float(canvas_height) - bottom,
            jitter=jitter,
        )
    )
    dx = float(shifted_left) - float(left)
    dy = float(shifted_top) - float(top)
    shifted_bbox = (
        round(float(left + dx), 3),
        round(float(top + dy), 3),
        round(float(right + dx), 3),
        round(float(bottom + dy), 3),
    )
    return shifted_bbox, dx, dy, dict(resolved)


def resolve_games_unit_size_scale(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int | None,
    namespace: str,
    key: str = "unit_size_scale",
    fallback_min: float = 0.5,
    fallback_max: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Resolve deterministic nonsemantic unit-size scaling for repeated-unit scenes."""

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


def scale_games_px(value: int | float, scale: float, *, min_px: int = 1) -> int:
    """Scale a pixel value while preserving a readable lower bound."""

    return max(int(min_px), int(round(float(value) * float(scale))))


def attach_games_unit_size_jitter(layout_jitter: Mapping[str, Any], unit_size_meta: Mapping[str, Any]) -> dict[str, Any]:
    """Attach unit-size metadata to the existing render-map layout metadata path."""

    resolved = dict(layout_jitter)
    resolved["unit_size_jitter"] = dict(unit_size_meta)
    return resolved


def offset_bbox(
    bbox_px: Sequence[float],
    *,
    dx: float,
    dy: float,
) -> BBox:
    """Return one bbox shifted by a fixed offset."""

    if len(bbox_px) != 4:
        raise ValueError("bbox_px must contain four coordinates")
    left, top, right, bottom = [float(value) for value in bbox_px]
    return (
        round(float(left + dx), 3),
        round(float(top + dy), 3),
        round(float(right + dx), 3),
        round(float(bottom + dy), 3),
    )


__all__ = [
    "attach_games_unit_size_jitter",
    "apply_games_layout_jitter_to_bbox",
    "offset_bbox",
    "resolve_games_layout_jitter",
    "resolve_games_unit_size_scale",
    "scale_games_px",
]
