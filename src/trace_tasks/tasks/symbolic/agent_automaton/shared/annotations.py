"""Annotation projection helpers for symbolic agent automata."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.common import projected_symbolic_bbox_annotation, projected_symbolic_keyed_bbox_annotation


def rounded_bbox(bbox: Sequence[float]) -> list[float]:
    """Round one bbox to stable JSON-friendly floats."""

    return [round(float(value), 3) for value in bbox]


def scalar_bbox(item_bboxes: Mapping[str, Sequence[float]], item_id: str) -> list[float]:
    """Project exactly one item bbox."""

    projection = projected_symbolic_bbox_annotation(item_bboxes, [str(item_id)])
    bboxes = projection.get("bbox_set", [])
    if len(bboxes) != 1:
        raise RuntimeError(f"expected exactly one bbox for {item_id!r}, got {len(bboxes)}")
    return rounded_bbox(bboxes[0])


def keyed_bboxes(item_bboxes: Mapping[str, Sequence[float]], role_item_ids: Mapping[str, str]) -> dict[str, list[float]]:
    """Project role-keyed item bboxes."""

    projection = projected_symbolic_keyed_bbox_annotation(item_bboxes, role_item_ids)
    return {
        str(key): rounded_bbox(bbox)
        for key, bbox in dict(projection.get("bbox_map", {})).items()
    }


def annotation_trace_payload(*, annotation_type: str, annotation_value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build witness/projection trace payload for bbox annotations."""

    if str(annotation_type) == "bbox_map":
        keyed = {str(key): rounded_bbox(bbox) for key, bbox in dict(annotation_value).items()}
        return {"bbox_map": dict(keyed)}, {"bbox_map": dict(keyed), "pixel_bbox_map": dict(keyed)}
    if str(annotation_type) == "bbox":
        bbox = rounded_bbox(annotation_value)
        return {"bbox": list(bbox)}, {"bbox": list(bbox), "pixel_bbox": list(bbox)}
    raise ValueError(f"unsupported agent-automaton annotation type: {annotation_type}")
