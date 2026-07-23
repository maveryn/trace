"""Geometry helpers for node-link graph rendering."""

from __future__ import annotations

from typing import Sequence, Tuple

from .graph_render_types import BBox, Point


def _orientation(a: Point, b: Point, c: Point) -> int:
    """Return one orientation sign for segment intersection testing."""

    value = ((int(b[1]) - int(a[1])) * (int(c[0]) - int(b[0]))) - ((int(b[0]) - int(a[0])) * (int(c[1]) - int(b[1])))
    if value == 0:
        return 0
    return 1 if value > 0 else 2


def _segments_intersect(left: Tuple[Point, Point], right: Tuple[Point, Point]) -> bool:
    """Return whether two closed line segments intersect."""

    p1, q1 = left
    p2, q2 = right
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    return bool(o1 != o2 and o3 != o4)


def _point_in_bbox(point: Point, bbox: BBox) -> bool:
    """Return whether one pixel point lies inside one axis-aligned bbox."""

    x, y = int(point[0]), int(point[1])
    x0, y0, x1, y1 = [int(value) for value in bbox]
    return bool(x0 <= x <= x1 and y0 <= y <= y1)


def _bbox_intersects_bbox(left: BBox, right: BBox) -> bool:
    """Return whether two axis-aligned bboxes overlap."""

    lx0, ly0, lx1, ly1 = [int(value) for value in left]
    rx0, ry0, rx1, ry1 = [int(value) for value in right]
    return not (lx1 < rx0 or rx1 < lx0 or ly1 < ry0 or ry1 < ly0)


def _segment_intersects_bbox(segment: Tuple[Point, Point], bbox: BBox) -> bool:
    """Return whether one line segment intersects one axis-aligned bbox."""

    start, end = segment
    if _point_in_bbox(start, bbox) or _point_in_bbox(end, bbox):
        return True
    x0, y0, x1, y1 = [int(value) for value in bbox]
    box_edges = (
        ((int(x0), int(y0)), (int(x1), int(y0))),
        ((int(x1), int(y0)), (int(x1), int(y1))),
        ((int(x1), int(y1)), (int(x0), int(y1))),
        ((int(x0), int(y1)), (int(x0), int(y0))),
    )
    return any(_segments_intersect(segment, edge) for edge in box_edges)


def _count_edge_crossings(segments: Sequence[Tuple[Tuple[str, str], Tuple[Point, Point]]]) -> int:
    """Count strict edge crossings, ignoring edges that share endpoints."""

    crossings = 0
    for index, (left_labels, left_segment) in enumerate(segments):
        left_endpoints = set(left_labels)
        for right_labels, right_segment in segments[index + 1 :]:
            if left_endpoints.intersection(set(right_labels)):
                continue
            if _segments_intersect(left_segment, right_segment):
                crossings += 1
    return int(crossings)


__all__ = [
    "_bbox_intersects_bbox",
    "_count_edge_crossings",
    "_point_in_bbox",
    "_segment_intersects_bbox",
    "_segments_intersect",
]
