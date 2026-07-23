"""Shared layout variation helpers for puzzle-domain renderers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from ...shared.render_variation import apply_resolved_layout_jitter_to_margins, resolve_layout_jitter


BBox = Tuple[float, float, float, float]


def resolve_puzzle_layout_jitter(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int | None,
    namespace: str,
) -> dict[str, Any]:
    """Resolve deterministic nonsemantic layout jitter for one puzzle instance."""

    return resolve_layout_jitter(
        params,
        defaults,
        instance_seed=instance_seed,
        namespace=str(namespace),
    )


def apply_puzzle_layout_jitter_to_bbox(
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


__all__ = [
    "apply_puzzle_layout_jitter_to_bbox",
    "resolve_puzzle_layout_jitter",
]
