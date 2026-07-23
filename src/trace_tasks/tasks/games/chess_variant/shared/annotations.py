"""Annotation projection helpers for chess-variant games tasks."""

from __future__ import annotations

from typing import Any, Mapping

from .state import ChessVariantEvaluation


def _bbox_center(bbox: list[float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _bbox_value(bbox: list[float]) -> list[float]:
    """Return one rendered bbox as public pixel coordinates."""

    return [round(float(value), 3) for value in bbox[:4]]


def annotation_from_evaluation(
    *,
    evaluation: ChessVariantEvaluation,
    render_map: Mapping[str, Any],
) -> tuple[str, list[list[float]], dict[str, Any]]:
    """Project semantic annotation ids to the public annotation payload."""

    if evaluation.annotation_kind == "cell":
        bbox_map = render_map["cell_bboxes_px"]
    else:
        bbox_map = render_map["piece_bboxes_px"]
    bboxes = [_bbox_value(list(bbox_map[str(entity_id)])) for entity_id in evaluation.annotation_entity_ids]
    return "bbox_set", bboxes, {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in bboxes],
        "pixel_bbox_set": [list(bbox) for bbox in bboxes],
    }


__all__ = ["annotation_from_evaluation"]
