"""Annotation projection helpers for standard Sankey charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .state import SankeyRenderResult


def _segment_points(*, rendered: SankeyRenderResult, segment_refs: Sequence[str]) -> list[list[float]]:
    rendered_scene = rendered.rendered_scene
    refs = [str(segment_id) for segment_id in segment_refs]
    return [list(rendered_scene.segment_center_map[str(segment_id)]) for segment_id in refs]


def annotation_payload(
    *,
    rendered: SankeyRenderResult,
    segment_refs: Sequence[str],
    annotation_type: str,
) -> dict[str, Any]:
    """Project task-selected Sankey value-label segment refs into the requested point contract.

    Public task files decide which segment refs are objective witnesses; this helper only converts
    those refs into scalar or set pixel points and preserves render-map metadata for review.
    """

    rendered_scene = rendered.rendered_scene
    refs = [str(segment_id) for segment_id in segment_refs]
    if str(annotation_type) == "point":
        points = _segment_points(rendered=rendered, segment_refs=refs)
        if len(points) != 1:
            raise ValueError("Sankey scalar point annotation requires exactly one segment ref")
        point = list(points[0])
        return {
            "type": "point",
            "value": list(point),
            "segment_refs": list(refs),
            "projected_annotation": {
                "type": "point",
                "point": list(point),
                "pixel_point": list(point),
                "segment_ids": list(refs),
                "segment_center_map": {
                    str(segment_id): list(rendered_scene.segment_center_map[str(segment_id)])
                    for segment_id in refs
                },
                "segment_label_bbox_map": {
                    str(segment_id): list(rendered_scene.segment_label_bbox_map[str(segment_id)])
                    for segment_id in refs
                },
                "segment_bbox_map": {
                    str(segment_id): list(rendered_scene.segment_bbox_map[str(segment_id)])
                    for segment_id in refs
                },
            },
        }
    if str(annotation_type) == "point_set":
        points = _segment_points(rendered=rendered, segment_refs=refs)
        return {
            "type": "point_set",
            "value": list(points),
            "segment_refs": list(refs),
            "projected_annotation": {
                "type": "point_set",
                "point_set": list(points),
                "pixel_point_set": list(points),
                "segment_ids": list(refs),
                "segment_center_map": {
                    str(segment_id): list(rendered_scene.segment_center_map[str(segment_id)])
                    for segment_id in refs
                },
                "segment_label_bbox_map": {
                    str(segment_id): list(rendered_scene.segment_label_bbox_map[str(segment_id)])
                    for segment_id in refs
                },
                "segment_bbox_map": {
                    str(segment_id): list(rendered_scene.segment_bbox_map[str(segment_id)])
                    for segment_id in refs
                },
            },
        }
    raise ValueError(f"unsupported Sankey annotation type: {annotation_type}")


__all__ = ["annotation_payload"]
