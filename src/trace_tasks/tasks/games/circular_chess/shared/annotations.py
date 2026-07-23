"""Annotation projection helpers for circular-chess games tasks."""

from __future__ import annotations

from typing import Any, Mapping

from .state import CircularChessEvaluation


def annotation_from_evaluation(
    *,
    evaluation: CircularChessEvaluation,
    render_map: Mapping[str, Any],
) -> tuple[str, list[list[float]], dict[str, Any]]:
    """Project semantic annotation ids to the public annotation payload."""

    if evaluation.annotation_kind == "source_piece_centers":
        points = [list(render_map["piece_centers_px"][str(entity_id)]) for entity_id in evaluation.annotation_entity_ids]
    else:
        points = [list(render_map["cell_centers_px"][str(entity_id)]) for entity_id in evaluation.annotation_entity_ids]
    return "point_set", [list(point) for point in points], {
        "type": "point_set",
        "point_set": [list(point) for point in points],
        "pixel_point_set": [list(point) for point in points],
    }


__all__ = ["annotation_from_evaluation"]
