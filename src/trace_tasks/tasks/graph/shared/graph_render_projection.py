"""Projection helpers for rendered node-link graph annotation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .graph_render_types import RenderedGraphScene


def projected_node_point_annotation(
    rendered_scene: RenderedGraphScene,
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Project ordered node labels into pixel point and bbox annotation."""

    requested = [str(label) for label in labels]
    node_by_label = {str(node.label): node for node in rendered_scene.nodes}
    point_map: Dict[str, List[float]] = {}
    point_set: List[List[float]] = []
    bbox_set: List[List[float]] = []
    for label in requested:
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


def projected_edge_pair_annotation(
    rendered_scene: RenderedGraphScene,
    edges: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Project ordered edge label pairs into node-center segments."""

    node_by_label = {str(node.label): node for node in rendered_scene.nodes}

    segment_set: List[List[List[float]]] = []
    for edge in edges:
        endpoints = [str(value) for value in edge]
        if len(endpoints) != 2:
            continue
        left_label, right_label = endpoints
        left_node = node_by_label.get(str(left_label))
        right_node = node_by_label.get(str(right_label))
        if left_node is None or right_node is None:
            continue
        pair = [
            [float(left_node.center_xy[0]), float(left_node.center_xy[1])],
            [float(right_node.center_xy[0]), float(right_node.center_xy[1])],
        ]
        segment_set.append([list(point) for point in pair])
    return {
        "segment_set": segment_set,
    }


def projected_edge_label_bbox_annotation(
    rendered_scene: RenderedGraphScene,
    edge: Sequence[str],
) -> Dict[str, Any]:
    """Project one rendered edge label into bbox annotation."""

    endpoints = [str(value) for value in edge]
    if len(endpoints) != 2:
        return {"bbox_set": [], "pixel_bbox_set": []}
    left_label, right_label = endpoints
    for rendered_edge in rendered_scene.edges:
        if str(rendered_edge.node_u_label) != str(left_label) or str(rendered_edge.node_v_label) != str(right_label):
            continue
        bbox = rendered_edge.edge_label_bbox_xyxy
        if bbox is None:
            return {"bbox_set": [], "pixel_bbox_set": []}
        bbox_value = [float(value) for value in bbox]
        return {"bbox_set": [list(bbox_value)], "pixel_bbox_set": [list(bbox_value)]}
    return {"bbox_set": [], "pixel_bbox_set": []}


__all__ = [
    "projected_edge_label_bbox_annotation",
    "projected_edge_pair_annotation",
    "projected_node_point_annotation",
]
