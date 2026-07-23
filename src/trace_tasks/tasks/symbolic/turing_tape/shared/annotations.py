"""Annotation helpers for symbolic Turing tape scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.common import projected_symbolic_keyed_bbox_annotation


def rounded_bbox(bbox: Sequence[float]) -> list[float]:
    """Round one bbox to stable JSON-friendly floats."""

    return [round(float(value), 3) for value in bbox]


def keyed_bboxes(item_bboxes: Mapping[str, Sequence[float]], role_item_ids: Mapping[str, str]) -> dict[str, list[float]]:
    """Project role-keyed item bboxes."""

    projection = projected_symbolic_keyed_bbox_annotation(item_bboxes, role_item_ids)
    return {
        str(key): rounded_bbox(bbox)
        for key, bbox in dict(projection.get("bbox_map", {})).items()
    }


def annotation_trace_payload(annotation_value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build witness/projection trace payload for keyed bbox annotations."""

    keyed = {str(key): rounded_bbox(bbox) for key, bbox in dict(annotation_value).items()}
    return {"bbox_map": dict(keyed)}, {"bbox_map": dict(keyed), "pixel_bbox_map": dict(keyed)}
