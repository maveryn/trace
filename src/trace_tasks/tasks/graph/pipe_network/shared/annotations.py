"""Annotation projection helpers for the pipe-network graph scene."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .state import RenderedPipeJunctionScene


def projected_pipe_node_point_annotation(
    rendered_scene: RenderedPipeJunctionScene,
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Project ordered junction labels into pixel point/bbox annotation."""

    node_by_label = {str(node.label): node for node in rendered_scene.nodes}
    point_map: Dict[str, list[float]] = {}
    point_set: list[list[float]] = []
    bbox_set: list[list[float]] = []
    for label in [str(value) for value in labels]:
        node = node_by_label.get(str(label))
        if node is None:
            continue
        point = [float(node.center_xy[0]), float(node.center_xy[1])]
        bbox = [float(value) for value in node.bbox_xyxy]
        point_map[str(label)] = list(point)
        point_set.append(list(point))
        bbox_set.append(list(bbox))
    return {
        "pixel_point_map": point_map,
        "pixel_point_set": point_set,
        "pixel_point_sequence": list(point_set),
        "pixel_bbox_set": bbox_set,
    }


def projected_pipe_segment_annotation(
    rendered_scene: RenderedPipeJunctionScene,
    edges: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Project pipe endpoint labels into node-center segment annotation."""

    node_by_label = {str(node.label): node for node in rendered_scene.nodes}
    segment_set: list[list[list[float]]] = []
    for edge in edges:
        endpoints = [str(value) for value in edge]
        if len(endpoints) != 2:
            continue
        left_node = node_by_label.get(endpoints[0])
        right_node = node_by_label.get(endpoints[1])
        if left_node is None or right_node is None:
            continue
        segment_set.append(
            [
                [float(left_node.center_xy[0]), float(left_node.center_xy[1])],
                [float(right_node.center_xy[0]), float(right_node.center_xy[1])],
            ]
        )
    return {"segment_set": segment_set}


__all__ = [
    "projected_pipe_node_point_annotation",
    "projected_pipe_segment_annotation",
]
