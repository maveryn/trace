"""Annotation projection helpers for process-flow page scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.pages.shared.diagram.common import projected_diagram_bbox_annotation


def _round_box(box: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in box]


def _round_segment(segment: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[round(float(value), 3) for value in point[:2]] for point in segment[:2]]


def node_bbox_annotation(
    render_map: Mapping[str, Any],
    node_refs: Sequence[str],
) -> tuple[list[list[float]], Dict[str, Any], list[str]]:
    """Project process-step node refs into a bbox-set annotation."""

    bbox_map = {
        f"node:{node_ref}": list(render_map["node_bboxes_px"][str(node_ref)])
        for node_ref in [str(value) for value in node_refs]
    }
    annotation_refs = [f"node:{node_ref}" for node_ref in [str(value) for value in node_refs]]
    projected = projected_diagram_bbox_annotation(bbox_map, annotation_refs)
    boxes = [_round_box(box) for box in projected["bbox_set"]]
    projected_payload = {
        "type": "bbox_set",
        "bbox_set": list(boxes),
        "pixel_bbox_set": list(boxes),
        "annotation_ids": list(annotation_refs),
    }
    return boxes, projected_payload, annotation_refs


def path_bbox_map_annotation(
    render_map: Mapping[str, Any],
    roles: Sequence[Mapping[str, str]],
) -> tuple[Dict[str, list[float]], Dict[str, Any], list[str], Dict[str, str]]:
    """Project role-bound path witnesses into a bbox-map annotation."""

    keyed_boxes: Dict[str, list[float]] = {}
    annotation_refs: list[str] = []
    key_to_ref: Dict[str, str] = {}
    for role in roles:
        key = str(role["key"])
        source_kind = str(role["kind"])
        source_ref = str(role["id"])
        if source_kind == "node":
            annotation_ref = f"node:{source_ref}"
            bbox = render_map["node_bboxes_px"][source_ref]
        elif source_kind == "edge_label":
            annotation_ref = f"edge_label:{source_ref}"
            bbox = render_map["edge_label_bboxes_px"][source_ref]
        else:
            raise ValueError(f"unsupported process-flow path role kind: {source_kind}")
        keyed_boxes[key] = _round_box(bbox)
        annotation_refs.append(str(annotation_ref))
        key_to_ref[key] = str(annotation_ref)
    projected_payload = {
        "type": "bbox_map",
        "bbox_map": dict(keyed_boxes),
        "pixel_bbox_map": dict(keyed_boxes),
        "bbox_set": list(keyed_boxes.values()),
        "annotation_keys": list(keyed_boxes.keys()),
        "annotation_key_to_bbox_id": dict(key_to_ref),
    }
    return dict(keyed_boxes), projected_payload, annotation_refs, dict(key_to_ref)


def edge_segment_annotation(
    render_map: Mapping[str, Any],
    edge_refs: Sequence[str],
) -> tuple[list[list[list[float]]], Dict[str, Any], list[str]]:
    """Project arrow refs into a segment-set annotation."""

    annotation_refs = [f"edge_segment:{edge_ref}" for edge_ref in [str(value) for value in edge_refs]]
    segments = [
        _round_segment(render_map["edge_segments_px"][str(edge_ref)])
        for edge_ref in [str(value) for value in edge_refs]
    ]
    projected_payload = {
        "type": "segment_set",
        "segment_set": list(segments),
        "pixel_segment_set": list(segments),
        "annotation_ids": list(annotation_refs),
    }
    return list(segments), projected_payload, list(annotation_refs)
