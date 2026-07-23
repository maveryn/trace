"""Annotation projection helpers for candlestick body and wick witnesses."""

from __future__ import annotations

from trace_tasks.tasks.charts.candlestick.shared.rendering import bbox_center
from trace_tasks.tasks.charts.candlestick.shared.state import BBox, Point, Rendered, Selection


def annotation_boxes_and_points(
    *,
    rendered: Rendered,
    selection: Selection,
) -> tuple[list[BBox], list[Point]]:
    """Project selected candle roles to body or wick boxes and their centers."""

    annotation_boxes: list[BBox] = []
    annotation_points: list[Point] = []
    for role, candle_id in zip(selection.annotation_roles, selection.annotation_candle_ids):
        if str(role).endswith("_wick"):
            annotation_box = list(rendered.wick_bboxes_px[str(candle_id)])
        else:
            annotation_box = list(rendered.body_bboxes_px[str(candle_id)])
        annotation_boxes.append(list(annotation_box))
        annotation_points.append(bbox_center(annotation_box))
    return annotation_boxes, annotation_points


def _vertical_segment_for_bbox(bbox: BBox) -> list[Point]:
    x_center, _y_center = bbox_center(bbox)
    return [
        [round(float(x_center), 3), round(float(bbox[1]), 3)],
        [round(float(x_center), 3), round(float(bbox[3]), 3)],
    ]


def annotation_segments(
    *,
    rendered: Rendered,
    selection: Selection,
) -> list[list[Point]]:
    """Project selected candle roles to vertical wick/body segment annotations."""

    annotation_segments_px: list[list[Point]] = []
    for role, candle_id in zip(selection.annotation_roles, selection.annotation_candle_ids):
        if str(role).endswith("_wick"):
            annotation_box = list(rendered.wick_bboxes_px[str(candle_id)])
        else:
            annotation_box = list(rendered.body_bboxes_px[str(candle_id)])
        annotation_segments_px.append(_vertical_segment_for_bbox(annotation_box))
    return annotation_segments_px


__all__ = ["annotation_boxes_and_points", "annotation_segments"]
