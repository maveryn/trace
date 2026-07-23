"""Annotation projection helpers for schema diagrams."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def round_box(bbox: Sequence[float]) -> list[float]:
    """Return one bbox rounded to stable pixel coordinates."""

    return [round(float(value), 3) for value in bbox[:4]]


def round_segment(segment: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return one two-point segment rounded to stable pixel coordinates."""

    return [
        [round(float(point[0]), 3), round(float(point[1]), 3)]
        for point in segment[:2]
    ]


def bbox_set_projection(
    *,
    boxes: Sequence[Sequence[float]],
    ids: Sequence[str],
) -> dict[str, Any]:
    """Return a homogeneous bbox-set projection payload."""

    rounded = [round_box(box) for box in boxes]
    return {
        "type": "bbox_set",
        "bbox_set": list(rounded),
        "pixel_bbox_set": list(rounded),
        "annotation_ids": [str(value) for value in ids],
    }


def bbox_projection(
    *,
    box: Sequence[float],
    id: str,
) -> dict[str, Any]:
    """Return a scalar bbox projection payload."""

    rounded = round_box(box)
    return {
        "type": "bbox",
        "bbox": list(rounded),
        "pixel_bbox": list(rounded),
        "annotation_ids": [str(id)],
    }


def bbox_map_projection(
    *,
    boxes: Mapping[str, Sequence[float]],
    ids: Mapping[str, str],
) -> dict[str, Any]:
    """Return a role-keyed bbox-map projection payload."""

    rounded = {str(key): round_box(value) for key, value in boxes.items()}
    return {
        "type": "bbox_map",
        "bbox_map": dict(rounded),
        "pixel_bbox_map": dict(rounded),
        "annotation_ids": [str(value) for value in ids.values()],
        "keyed_annotation_ids": {str(key): str(value) for key, value in ids.items()},
    }


def segment_set_projection(
    *,
    segments: Sequence[Sequence[Sequence[float]]],
    ids: Sequence[str],
) -> dict[str, Any]:
    """Return a homogeneous segment-set projection payload."""

    rounded = [round_segment(segment) for segment in segments]
    return {
        "type": "segment_set",
        "segment_set": list(rounded),
        "pixel_segment_set": list(rounded),
        "annotation_ids": [str(value) for value in ids],
    }
