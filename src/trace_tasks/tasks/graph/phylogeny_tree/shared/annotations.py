"""Annotation projection helpers for phylogeny-tree graph scenes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .state import RenderedPhylogenyScene


def projected_leaf_point_annotation(
    rendered_scene: RenderedPhylogenyScene,
    leaf_labels: Iterable[str],
) -> Dict[str, Any]:
    """Project leaf labels into point-set annotation using leaf terminal centers."""

    leaf_points: Dict[str, List[int]] = {}
    leaf_bboxes: Dict[str, List[int]] = {}
    for node in rendered_scene.nodes:
        if node.leaf_label is None:
            continue
        leaf_points[str(node.leaf_label)] = [int(node.center_xy[0]), int(node.center_xy[1])]
        if node.label_bbox_xyxy is not None:
            leaf_bboxes[str(node.leaf_label)] = [int(value) for value in node.label_bbox_xyxy]
    points = [leaf_points[str(label)] for label in leaf_labels]
    return {
        "point_set": [list(point) for point in points],
        "pixel_point_set": [list(point) for point in points],
        "leaf_label_bbox_map": {key: list(value) for key, value in sorted(leaf_bboxes.items())},
    }


def projected_keyed_phylogeny_annotation(
    rendered_scene: RenderedPhylogenyScene,
    *,
    role_to_node_id: Mapping[str, str],
) -> Dict[str, Any]:
    """Project keyed leaf/internal node witnesses into keyed bbox/point annotation."""

    node_by_id = {str(node.node_id): node for node in rendered_scene.nodes}
    bbox_map: Dict[str, List[int]] = {}
    point_map: Dict[str, List[int]] = {}
    for role, node_id in sorted(role_to_node_id.items()):
        node = node_by_id[str(node_id)]
        bbox = node.label_bbox_xyxy if node.label_bbox_xyxy is not None else node.bbox_xyxy
        bbox_map[str(role)] = [int(value) for value in bbox]
        point_map[str(role)] = [int(node.center_xy[0]), int(node.center_xy[1])]
    return {
        "bbox_map": dict(bbox_map),
        "pixel_bbox_map": dict(bbox_map),
        "point_map": dict(point_map),
        "pixel_point_map": dict(point_map),
    }


__all__ = [
    "projected_keyed_phylogeny_annotation",
    "projected_leaf_point_annotation",
]
