"""Annotation helpers for word-search puzzle tasks."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.puzzles.shared.word_grid import Cell, cell_key


def bbox_sequence_for_cells(
    item_bbox_map: dict[str, list[float]],
    cells: tuple[Cell, ...],
) -> tuple[TypedValue, dict[str, object], dict[str, object]]:
    """Return ordered bbox-sequence annotation for a visible cell path."""

    sequence = [
        [round(float(value), 3) for value in item_bbox_map[cell_key(cell)]]
        for cell in cells
    ]
    annotation_gt = TypedValue(type="bbox_sequence", value=list(sequence))
    projected_annotation = {
        "type": "bbox_sequence",
        "bbox_sequence": list(sequence),
        "value": list(sequence),
    }
    witness_symbolic = {"type": "bbox_sequence", "value": list(sequence)}
    return annotation_gt, projected_annotation, witness_symbolic


def segment_for_cell_pair(
    cell_centers_px: dict[str, tuple[float, float]],
    cell_pair: tuple[Cell, Cell],
) -> tuple[TypedValue, dict[str, object], dict[str, object]]:
    """Return one segment annotation from start/end cell centers."""

    start_cell, end_cell = cell_pair
    start = cell_centers_px[cell_key(start_cell)]
    end = cell_centers_px[cell_key(end_cell)]
    segment = [
        [round(float(start[0]), 3), round(float(start[1]), 3)],
        [round(float(end[0]), 3), round(float(end[1]), 3)],
    ]
    annotation_gt = TypedValue(type="segment", value=list(segment))
    projected_annotation = {
        "type": "segment",
        "segment": list(segment),
        "value": list(segment),
    }
    witness_symbolic = {"type": "segment", "value": list(segment)}
    return annotation_gt, projected_annotation, witness_symbolic


__all__ = [
    "bbox_sequence_for_cells",
    "segment_for_cell_pair",
]
