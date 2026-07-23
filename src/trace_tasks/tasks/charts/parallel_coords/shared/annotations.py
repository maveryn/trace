"""Annotation projection helpers for the parallel-coordinates chart scene."""

from __future__ import annotations

from typing import Any, Sequence

from trace_tasks.core.types import TypedValue

from .state import ParallelDataset, Point, RenderedParallelScene


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError(f"expected bbox with 4 values, got {bbox}")
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _round_point(x: float, y: float) -> Point:
    return [round(float(x), 2), round(float(y), 2)]


def _axis_point(
    rendered: RenderedParallelScene,
    *,
    profile_id: str,
    axis_index: int,
) -> tuple[float, float]:
    bbox = rendered.point_bboxes_px[f"{profile_id}:axis_{int(axis_index)}"]
    return _bbox_center(bbox)


def profile_segment(
    dataset: ParallelDataset,
    rendered: RenderedParallelScene,
    *,
    profile_id: str,
) -> list[Point]:
    """Return the selected visible profile segment endpoints."""

    x0, y0 = _axis_point(rendered, profile_id=str(profile_id), axis_index=int(dataset.query.axis_i))
    x1, y1 = _axis_point(rendered, profile_id=str(profile_id), axis_index=int(dataset.query.axis_j))
    return [_round_point(x0, y0), _round_point(x1, y1)]


def profile_segment_set(
    dataset: ParallelDataset,
    rendered: RenderedParallelScene,
    *,
    profile_ids: Sequence[str],
) -> list[list[Point]]:
    return [
        profile_segment(dataset, rendered, profile_id=str(profile_id))
        for profile_id in profile_ids
    ]


def crossing_point(
    dataset: ParallelDataset,
    rendered: RenderedParallelScene,
    *,
    first_profile_id: str,
    second_profile_id: str,
) -> Point:
    axis_i = int(dataset.query.axis_i)
    axis_j = int(dataset.query.axis_j)
    x0, y0 = _axis_point(rendered, profile_id=str(first_profile_id), axis_index=axis_i)
    x1, y1 = _axis_point(rendered, profile_id=str(first_profile_id), axis_index=axis_j)
    _, other_y0 = _axis_point(rendered, profile_id=str(second_profile_id), axis_index=axis_i)
    _, other_y1 = _axis_point(rendered, profile_id=str(second_profile_id), axis_index=axis_j)
    first_delta = float(y1) - float(y0)
    second_delta = float(other_y1) - float(other_y0)
    denom = float(first_delta) - float(second_delta)
    if abs(float(denom)) < 1e-9:
        return _round_point((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0)
    t = (float(other_y0) - float(y0)) / float(denom)
    t = max(0.0, min(1.0, float(t)))
    return _round_point(float(x0) + (float(x1) - float(x0)) * t, float(y0) + first_delta * t)


def crossing_point_set(
    dataset: ParallelDataset,
    rendered: RenderedParallelScene,
    *,
    crossing_pairs: Sequence[tuple[str, str]],
) -> list[Point]:
    return [
        crossing_point(
            dataset,
            rendered,
            first_profile_id=str(first_profile_id),
            second_profile_id=str(second_profile_id),
        )
        for first_profile_id, second_profile_id in crossing_pairs
    ]


def point_set_annotation(points: Sequence[Sequence[float]]) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    value = [[round(float(point[0]), 2), round(float(point[1]), 2)] for point in points]
    return (
        TypedValue(type="point_set", value=list(value)),
        {"type": "point_set", "count": len(value)},
        {"type": "point_set", "point_set": list(value), "pixel_point_set": list(value)},
    )


def segment_annotation(segment: Sequence[Sequence[float]]) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    value = [
        [round(float(point[0]), 2), round(float(point[1]), 2)]
        for point in segment[:2]
    ]
    return (
        TypedValue(type="segment", value=[list(point) for point in value]),
        {"type": "segment", "segment": [list(point) for point in value]},
        {
            "type": "segment",
            "segment": [list(point) for point in value],
            "pixel_segment": [list(point) for point in value],
        },
    )


def segment_set_annotation(
    segments: Sequence[Sequence[Sequence[float]]],
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    value = [
        [[round(float(point[0]), 2), round(float(point[1]), 2)] for point in segment[:2]]
        for segment in segments
    ]
    return (
        TypedValue(type="segment_set", value=[[list(point) for point in segment] for segment in value]),
        {"type": "segment_set", "count": len(value)},
        {
            "type": "segment_set",
            "segment_set": [[list(point) for point in segment] for segment in value],
            "pixel_segment_set": [[list(point) for point in segment] for segment in value],
        },
    )


__all__ = [
    "crossing_point_set",
    "point_set_annotation",
    "profile_segment",
    "profile_segment_set",
    "segment_annotation",
    "segment_set_annotation",
]
