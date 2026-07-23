"""Three_d annotation geometry normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_MIN_ANNOTATION_BBOX_SIDE_PX = 24.0


def _round_bbox(bbox: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in bbox[:4]]


def bbox_min_side_px(bbox: Sequence[float]) -> float:
    """Return the smaller side length for a pixel bbox."""

    return min(
        max(0.0, float(bbox[2]) - float(bbox[0])),
        max(0.0, float(bbox[3]) - float(bbox[1])),
    )


def _fit_interval(
    center: float,
    current_low: float,
    current_high: float,
    *,
    target_size: float,
    bound_low: float | None,
    bound_high: float | None,
) -> Tuple[float, float]:
    current_size = max(0.0, float(current_high) - float(current_low))
    size = max(float(current_size), float(target_size))
    if bound_low is None or bound_high is None:
        return (float(center) - size * 0.5, float(center) + size * 0.5)

    low = float(bound_low)
    high = float(bound_high)
    available = max(0.0, high - low)
    size = min(float(size), float(available))
    fitted_low = float(center) - size * 0.5
    fitted_high = float(center) + size * 0.5
    if fitted_low < low:
        fitted_high += low - fitted_low
        fitted_low = low
    if fitted_high > high:
        fitted_low -= fitted_high - high
        fitted_high = high
    fitted_low = max(low, fitted_low)
    fitted_high = min(high, fitted_high)
    return (float(fitted_low), float(fitted_high))


def expand_bbox_to_min_side(
    bbox: Sequence[float],
    *,
    min_side_px: float = DEFAULT_MIN_ANNOTATION_BBOX_SIDE_PX,
    bounds_px: Sequence[float] | None = None,
) -> List[float]:
    """Expand a bbox around its center so both sides meet the minimum when bounds allow."""

    x0, y0, x1, y1 = (float(value) for value in bbox[:4])
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    bounds = tuple(float(value) for value in bounds_px[:4]) if bounds_px is not None else None
    bx0 = bounds[0] if bounds is not None else None
    by0 = bounds[1] if bounds is not None else None
    bx1 = bounds[2] if bounds is not None else None
    by1 = bounds[3] if bounds is not None else None
    nx0, nx1 = _fit_interval(
        cx,
        x0,
        x1,
        target_size=float(min_side_px),
        bound_low=bx0,
        bound_high=bx1,
    )
    ny0, ny1 = _fit_interval(
        cy,
        y0,
        y1,
        target_size=float(min_side_px),
        bound_low=by0,
        bound_high=by1,
    )
    return _round_bbox([nx0, ny0, nx1, ny1])


def normalize_annotation_bboxes(
    bboxes: Sequence[Sequence[float]],
    *,
    min_side_px: float = DEFAULT_MIN_ANNOTATION_BBOX_SIDE_PX,
    bounds_px: Sequence[float] | None = None,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Return min-side normalized annotation boxes plus trace metadata."""

    raw_bboxes = [_round_bbox(bbox) for bbox in bboxes]
    normalized = [
        expand_bbox_to_min_side(
            bbox,
            min_side_px=float(min_side_px),
            bounds_px=bounds_px,
        )
        for bbox in raw_bboxes
    ]
    raw_min_sides = [round(float(bbox_min_side_px(bbox)), 3) for bbox in raw_bboxes]
    normalized_min_sides = [round(float(bbox_min_side_px(bbox)), 3) for bbox in normalized]
    changed_indices = [
        int(index)
        for index, (raw_bbox, normalized_bbox) in enumerate(zip(raw_bboxes, normalized))
        if list(raw_bbox) != list(normalized_bbox)
    ]
    metadata: Dict[str, Any] = {
        "enabled": True,
        "min_side_px": float(min_side_px),
        "bounds_px": _round_bbox(bounds_px) if bounds_px is not None else None,
        "input_count": int(len(raw_bboxes)),
        "adjusted_count": int(len(changed_indices)),
        "changed_indices": list(changed_indices),
        "raw_min_side_px": min(raw_min_sides) if raw_min_sides else None,
        "normalized_min_side_px": min(normalized_min_sides) if normalized_min_sides else None,
    }
    return [list(bbox) for bbox in normalized], metadata


__all__ = [
    "DEFAULT_MIN_ANNOTATION_BBOX_SIDE_PX",
    "bbox_min_side_px",
    "expand_bbox_to_min_side",
    "normalize_annotation_bboxes",
]
