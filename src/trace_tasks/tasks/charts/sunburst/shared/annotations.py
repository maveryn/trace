"""Annotation helpers for sunburst chart tasks."""

from __future__ import annotations

from collections.abc import Sequence

from .state import Point, RenderedSunburst


def leaf_value_points(rendered: RenderedSunburst, leaf_ids: Sequence[str]) -> list[Point]:
    points = [
        [
            round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
            round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
        ]
        for leaf_id in leaf_ids
        if str(leaf_id) in rendered.leaf_value_bbox_by_node_id
        for bbox in [rendered.leaf_value_bbox_by_node_id[str(leaf_id)]]
    ]
    if not points:
        raise ValueError("sunburst annotation produced no leaf value points")
    return points


__all__ = ["leaf_value_points"]
