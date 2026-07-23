"""Neutral bbox helpers for grid-like chart renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

BBox = list[float]


def round_bbox(box: Sequence[float]) -> BBox:
    """Return one bbox rounded to Trace's standard three decimal places."""

    if len(box) < 4:
        raise ValueError("bbox must contain at least four values")
    return [round(float(value), 3) for value in box[:4]]


def bbox_union(boxes: Sequence[Sequence[float]]) -> BBox:
    """Return one union bbox over one non-empty bbox sequence."""

    if not boxes:
        raise ValueError("cannot union an empty bbox sequence")
    return [
        round(min(float(box[0]) for box in boxes), 3),
        round(min(float(box[1]) for box in boxes), 3),
        round(max(float(box[2]) for box in boxes), 3),
        round(max(float(box[3]) for box in boxes), 3),
    ]


def bbox_center(box: Sequence[float]) -> list[float]:
    """Return the center point of one bbox."""

    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    return [round((x0 + x1) / 2.0, 3), round((y0 + y1) / 2.0, 3)]


def bboxes_for_ids(
    bbox_map: Mapping[str, Sequence[float]],
    ids: Sequence[str],
    *,
    missing: str = "error",
) -> list[BBox]:
    """Return bboxes for ids in order.

    ``missing="skip"`` preserves legacy table behavior; ``missing="error"``
    preserves matrix/heatmap behavior.
    """

    if str(missing) not in {"error", "skip"}:
        raise ValueError(f"unsupported missing policy: {missing}")
    boxes: list[BBox] = []
    for item_id in ids:
        key = str(item_id)
        if key not in bbox_map:
            if str(missing) == "skip":
                continue
            raise KeyError(key)
        boxes.append(round_bbox(bbox_map[key]))
    return boxes


def bbox_for_id(bbox_map: Mapping[str, Sequence[float]], item_id: str) -> BBox:
    """Return one bbox by id."""

    return bboxes_for_ids(bbox_map, [str(item_id)], missing="error")[0]


__all__ = ["BBox", "bbox_center", "bbox_for_id", "bbox_union", "bboxes_for_ids", "round_bbox"]
