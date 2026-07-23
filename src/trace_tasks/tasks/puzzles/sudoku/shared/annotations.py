"""Annotation projection helpers for Sudoku cells."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .rules import coord_to_cell_id
from .state import Coord


def cell_ids_for_coords(coords: Sequence[Coord]) -> list[str]:
    """Return render-map cell ids for Sudoku coordinates."""

    return [coord_to_cell_id((int(row), int(col))) for row, col in coords]


def bbox_payload(bbox: Sequence[float]) -> dict[str, Any]:
    """Return one scalar bbox projection payload."""

    resolved = [round(float(value), 3) for value in bbox[:4]]
    return {
        "type": "bbox",
        "bbox": list(resolved),
        "pixel_bbox": list(resolved),
    }


def bbox_for_coord(
    bbox_map: Mapping[str, Sequence[float]],
    coord: Coord,
) -> tuple[dict[str, Any], list[str]]:
    """Project one Sudoku cell into a scalar bbox annotation."""

    entity_id = coord_to_cell_id((int(coord[0]), int(coord[1])))
    if entity_id not in bbox_map:
        raise RuntimeError(f"missing Sudoku cell bbox for annotation: {entity_id}")
    return bbox_payload(bbox_map[entity_id]), [str(entity_id)]


__all__ = [
    "bbox_for_coord",
    "bbox_payload",
    "cell_ids_for_coords",
]
