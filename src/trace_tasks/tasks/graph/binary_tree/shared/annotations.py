"""Annotation projection helpers for graph binary-tree scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .state import RenderedBinaryTreeScene


def project_node_label_bboxes(
    rendered_scene: RenderedBinaryTreeScene,
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Project ordered node labels into pixel bbox and point annotations."""

    requested = [str(label) for label in labels]
    node_by_label = {str(node.label): node for node in rendered_scene.nodes}
    bbox_map: Dict[str, list[float]] = {}
    bbox_set: list[list[float]] = []
    point_set: list[list[float]] = []
    for label in requested:
        node = node_by_label.get(str(label))
        if node is None:
            continue
        bbox = [float(value) for value in node.bbox_xyxy]
        center = [float(node.center_xy[0]), float(node.center_xy[1])]
        bbox_map[str(label)] = list(bbox)
        bbox_set.append(list(bbox))
        point_set.append(list(center))
    return {
        "pixel_bbox_map": bbox_map,
        "bbox_set": [list(bbox) for bbox in bbox_set],
        "pixel_bbox_set": [list(bbox) for bbox in bbox_set],
        "bbox_sequence": [list(bbox) for bbox in bbox_set],
        "pixel_bbox_sequence": [list(bbox) for bbox in bbox_set],
        "pixel_point_set": [list(point) for point in point_set],
        "pixel_point_sequence": [list(point) for point in point_set],
    }


def rounded_bboxes(bboxes: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return JSON-stable rounded bbox coordinates."""

    return [[round(float(value), 3) for value in bbox] for bbox in bboxes]


def rounded_points(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return JSON-stable rounded point coordinates."""

    return [[round(float(value), 3) for value in point] for point in points]


def keyed_bboxes_for_roles(
    *,
    roles: Sequence[str],
    labels: Sequence[str],
    projection: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Bind projected bbox sequence values to semantic annotation roles."""

    boxes = rounded_bboxes(projection.get("bbox_sequence", []))
    if len(tuple(roles)) != len(boxes):
        raise ValueError("annotation role count does not match projected node boxes")
    return {str(role): list(box) for role, box in zip(roles, boxes)}


def keyed_points_for_roles(
    *,
    roles: Sequence[str],
    projection: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Bind projected point sequence values to semantic annotation roles."""

    points = projection.get("pixel_point_sequence", projection.get("pixel_point_set", []))
    if len(tuple(roles)) != len(points):
        raise ValueError("annotation role count does not match projected node points")
    return {
        str(role): [round(float(value), 3) for value in point]
        for role, point in zip(roles, points)
    }


def role_to_label_map(*, roles: Sequence[str], labels: Sequence[str]) -> dict[str, str]:
    """Return semantic annotation roles mapped to node labels."""

    if len(tuple(roles)) != len(tuple(labels)):
        raise ValueError("annotation role count does not match labels")
    return {str(role): str(label) for role, label in zip(roles, labels)}


def roles_by_label(role_to_label: Mapping[str, str]) -> dict[str, list[str]]:
    """Return reverse label-to-role mapping for trace node entities."""

    result: dict[str, list[str]] = {}
    for role, label in role_to_label.items():
        result.setdefault(str(label), []).append(str(role))
    return result


__all__ = [
    "keyed_bboxes_for_roles",
    "keyed_points_for_roles",
    "project_node_label_bboxes",
    "role_to_label_map",
    "roles_by_label",
    "rounded_bboxes",
    "rounded_points",
]
