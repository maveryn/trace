"""Annotation projection helpers for styled table chart tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.tasks.charts.shared.grid.annotations import (
    annotation_value_from_projection,
    bbox_map_projection,
    bbox_projection,
    bbox_set_map_projection,
    bbox_set_projection,
)
from trace_tasks.tasks.charts.shared.grid.geometry import bboxes_for_ids, round_bbox

from .rendering import RenderedTableScene


def cell_boxes(rendered: RenderedTableScene, cell_ids: Sequence[str]) -> list[list[float]]:
    """Return table-cell bounding boxes in the requested cell-id order."""

    bbox_map = {
        str(cell_trace["cell_id"]): list(cell_trace["bbox_px"])
        for cell_trace in rendered.cell_traces
    }
    return bboxes_for_ids(bbox_map, [str(cell_id) for cell_id in cell_ids], missing="skip")


def cell_box(rendered: RenderedTableScene, cell_id: str) -> list[float]:
    """Return one table-cell bounding box."""

    boxes = cell_boxes(rendered, [str(cell_id)])
    if len(boxes) != 1:
        raise ValueError(f"missing table cell annotation target: {cell_id}")
    return list(boxes[0])


def column_box(rendered: RenderedTableScene, column_header: str) -> list[float]:
    """Return one numeric column-region bounding box."""

    if str(column_header) not in rendered.column_region_bboxes:
        raise ValueError(f"missing table column annotation target: {column_header}")
    return round_bbox(rendered.column_region_bboxes[str(column_header)])


def boxed_set_projection(boxes: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return projected annotation payload for a bbox set."""

    return bbox_set_projection(boxes)


def boxed_projection(box: Sequence[float]) -> dict[str, Any]:
    """Return projected annotation payload for a single bbox."""

    return bbox_projection(box)


def boxed_map_projection(box_map: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Return projected annotation payload for keyed scalar bboxes."""

    return bbox_map_projection(box_map)


def boxed_set_map_projection(box_map: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, Any]:
    """Return projected annotation payload for keyed bbox sets."""

    return bbox_set_map_projection(box_map)


__all__ = [
    "annotation_value_from_projection",
    "boxed_map_projection",
    "boxed_projection",
    "boxed_set_map_projection",
    "boxed_set_projection",
    "cell_box",
    "cell_boxes",
    "column_box",
]
