"""Shared bbox projection helpers for annotation/anchor payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


BBox = Tuple[float, float, float, float]


def round_bbox(values: Sequence[float], *, ndigits: int = 3) -> List[float]:
    """Return one bbox/list with deterministic decimal rounding."""

    return [round(float(value), int(ndigits)) for value in values]


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """Return center point of one bbox `(x0, y0, x1, y1)`."""
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def bbox_union(
    boxes: Iterable[Sequence[float]],
    *,
    padding: float = 0.0,
    normalize: bool = True,
    ndigits: int = 3,
) -> List[float]:
    """Return one rounded bbox covering all non-empty input boxes."""

    resolved: List[List[float]] = []
    for box in boxes:
        if len(box) < 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in box[:4]]
        if bool(normalize):
            resolved.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
        else:
            resolved.append([x0, y0, x1, y1])
    if not resolved:
        return [0.0, 0.0, 0.0, 0.0]
    pad = float(padding)
    return round_bbox(
        [
            min(box[0] for box in resolved) - pad,
            min(box[1] for box in resolved) - pad,
            max(box[2] for box in resolved) + pad,
            max(box[3] for box in resolved) + pad,
        ],
        ndigits=int(ndigits),
    )


def bbox_union_many(
    *bboxes: Sequence[float],
    padding: float = 0.0,
    normalize: bool = True,
    ndigits: int = 3,
) -> List[float]:
    """Varargs adapter for :func:`bbox_union`."""

    return bbox_union(bboxes, padding=float(padding), normalize=bool(normalize), ndigits=int(ndigits))


def bbox_union_raw(
    boxes: Iterable[Sequence[float]],
    *,
    padding: float = 0.0,
    ndigits: int = 3,
) -> List[float]:
    """Return a rounded bbox union without normalizing individual boxes."""

    return bbox_union(boxes, padding=float(padding), normalize=False, ndigits=int(ndigits))


def bbox_union_many_raw(
    *bboxes: Sequence[float],
    padding: float = 0.0,
    ndigits: int = 3,
) -> List[float]:
    """Varargs adapter for raw bbox unions."""

    return bbox_union(bboxes, padding=float(padding), normalize=False, ndigits=int(ndigits))


def ordered_ids_to_point_sequence_and_bbox_set(
    *,
    ordered_ids: Sequence[str],
    bbox_map: Mapping[str, BBox],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Project ordered entity ids into point-path and bbox-set payloads."""
    point_sequence: List[List[float]] = []
    bbox_set: List[List[float]] = []
    for entity_id in ordered_ids:
        bbox = bbox_map[entity_id]
        center = bbox_center(bbox)
        bbox_set.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
        point_sequence.append([float(center[0]), float(center[1])])
    return point_sequence, bbox_set


def pixel_anchor_map_from_bboxes(bbox_map: Mapping[str, BBox]) -> Dict[str, Dict[str, Any]]:
    """Build deterministic pixel anchor payloads from per-entity bboxes."""
    return {
        entity_id: {
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "point": [float(center[0]), float(center[1])],
            "coord_space": "pixel",
        }
        for entity_id, bbox in sorted(bbox_map.items())
        for center in [bbox_center(bbox)]
    }


__all__ = [
    "BBox",
    "bbox_center",
    "bbox_union",
    "bbox_union_many",
    "bbox_union_many_raw",
    "bbox_union_raw",
    "ordered_ids_to_point_sequence_and_bbox_set",
    "pixel_anchor_map_from_bboxes",
    "round_bbox",
]
