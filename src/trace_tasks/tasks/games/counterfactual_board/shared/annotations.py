"""Annotation projection helpers for counterfactual-board units."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    segment_set_annotation_artifacts,
)
from trace_tasks.tasks.shared.bbox_projection import BBox, round_bbox

from .rules import element_kind_for_axis
from .state import (
    COLUMN_AXIS,
    CountedElement,
    HORIZONTAL_LINE_AXIS,
    ROW_AXIS,
    VERTICAL_LINE_AXIS,
)


def counted_elements_for_axis(
    *,
    counted_axis: str,
    rows: int,
    cols: int,
    board_bbox: Sequence[float],
    line_thickness_px: int,
) -> list[CountedElement]:
    """Project rows, columns, or line-board grid lines into bbox witnesses."""

    x0, y0, x1, y1 = [float(value) for value in board_bbox]
    axis = str(counted_axis)
    kind = element_kind_for_axis(axis)
    elements: list[CountedElement] = []
    if axis == ROW_AXIS:
        cell_h = (y1 - y0) / float(max(1, int(rows)))
        for row in range(int(rows)):
            bbox: BBox = (
                float(x0),
                float(y0 + row * cell_h),
                float(x1),
                float(y0 + (row + 1) * cell_h),
            )
            elements.append(CountedElement(f"row_{row}", kind, round_bbox(bbox)))
    elif axis == COLUMN_AXIS:
        cell_w = (x1 - x0) / float(max(1, int(cols)))
        for col in range(int(cols)):
            bbox = (
                float(x0 + col * cell_w),
                float(y0),
                float(x0 + (col + 1) * cell_w),
                float(y1),
            )
            elements.append(CountedElement(f"column_{col}", kind, round_bbox(bbox)))
    elif axis == HORIZONTAL_LINE_AXIS:
        step = (y1 - y0) / float(max(1, int(rows) - 1))
        half = max(12.0, float(line_thickness_px) * 0.5)
        for row in range(int(rows)):
            y = y0 + float(row) * step
            segment = ((float(x0), float(y)), (float(x1), float(y)))
            elements.append(
                CountedElement(
                    f"horizontal_line_{row}",
                    kind,
                    round_bbox((float(x0), y - half, float(x1), y + half)),
                    segment=segment,
                )
            )
    elif axis == VERTICAL_LINE_AXIS:
        step = (x1 - x0) / float(max(1, int(cols) - 1))
        half = max(12.0, float(line_thickness_px) * 0.5)
        for col in range(int(cols)):
            x = x0 + float(col) * step
            segment = ((float(x), float(y0)), (float(x), float(y1)))
            elements.append(
                CountedElement(
                    f"vertical_line_{col}",
                    kind,
                    round_bbox((x - half, float(y0), x + half, float(y1))),
                    segment=segment,
                )
            )
    else:
        raise ValueError(f"unsupported counted axis: {counted_axis!r}")
    return elements


def bbox_set_for_counted_elements(
    elements: Sequence[CountedElement],
) -> AnnotationArtifacts:
    """Build public bbox-set annotation from counted visual elements."""

    return bbox_set_annotation_artifacts([element.bbox for element in elements])


def segment_set_for_counted_elements(
    elements: Sequence[CountedElement],
) -> AnnotationArtifacts:
    """Build public segment-set annotation from counted line elements."""

    segments = []
    for element in elements:
        if element.segment is None:
            raise ValueError(f"counted element {element.element_id!r} has no segment")
        segments.append(
            [
                [float(element.segment[0][0]), float(element.segment[0][1])],
                [float(element.segment[1][0]), float(element.segment[1][1])],
            ]
        )
    return segment_set_annotation_artifacts(segments)


__all__ = [
    "bbox_set_for_counted_elements",
    "counted_elements_for_axis",
    "segment_set_for_counted_elements",
]
