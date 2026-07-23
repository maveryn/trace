"""Public trace sanitization for pixel-grounded annotation payloads."""

from __future__ import annotations

from typing import Any, Mapping

from .types import TypedValue


PUBLIC_IMAGE_ANNOTATION_TYPES = frozenset(
    {
        "bbox",
        "bbox_sequence",
        "bbox_set",
        "bbox_map",
        "bbox_set_map",
        "point_map",
        "point_set_map",
        "point",
        "segment",
        "segment_set",
        "point_sequence",
        "point_set",
    }
)

_SOURCE_EXECUTION_KEYS = frozenset(
    {
        "edge_set",
        "annotation_edges",
        "annotation_entity_ids",
        "annotation_graph_points",
        "annotation_labels",
        "graph_point",
        "graph_point_map",
        "graph_point_set",
        "grid_path",
        "grid_point",
        "grid_point_map",
        "grid_point_set",
        "grid_points",
        "id_path",
        "id_set",
        "label_path",
        "label_sequence",
        "label_set",
        "matching_ids",
        "matching_labels",
        "original_annotation_type",
        "original_annotation_value",
        "path_edge_labels",
        "path_ids",
        "path_labels",
        "reachable_ids",
        "reachable_target_ids",
        "shortest_path_ids",
        "violation_ids",
        "winner_path_ids",
        "winning_component_ids",
    }
)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value]
    if isinstance(value, tuple):
        return [item for item in value]
    return []


def _as_string_keyed_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _as_keyed_list_mapping(value: Any) -> dict[str, list[Any]]:
    keyed: dict[str, list[Any]] = {}
    for key, item in _as_string_keyed_mapping(value).items():
        keyed[str(key)] = _as_list(item)
    return keyed


def _annotation_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1 if value is not None else 0


def public_projected_annotation(annotation: TypedValue) -> dict[str, Any]:
    """Return the exported projected-annotation payload for a public annotation value."""

    annotation_type = str(annotation.type)
    value = _as_list(annotation.value)
    if annotation_type == "bbox":
        bbox = list(annotation.value) if isinstance(annotation.value, (list, tuple)) else annotation.value
        return {
            "type": "bbox",
            "bbox": bbox,
            "pixel_bbox": bbox,
        }
    if annotation_type == "bbox_set":
        return {
            "type": "bbox_set",
            "bbox_set": [list(item) for item in value],
        }
    if annotation_type == "bbox_sequence":
        return {
            "type": "bbox_sequence",
            "bbox_sequence": [list(item) for item in value],
        }
    if annotation_type == "bbox_map":
        keyed_bboxes = {key: list(item) for key, item in _as_string_keyed_mapping(annotation.value).items()}
        return {
            "type": "bbox_map",
            "bbox_map": keyed_bboxes,
        }
    if annotation_type == "bbox_set_map":
        keyed_bbox_sets = {
            key: [list(bbox) for bbox in bboxes]
            for key, bboxes in _as_keyed_list_mapping(annotation.value).items()
        }
        return {
            "type": "bbox_set_map",
            "bbox_set_map": keyed_bbox_sets,
            "pixel_bbox_set_map": keyed_bbox_sets,
        }
    if annotation_type == "point":
        point = list(annotation.value) if isinstance(annotation.value, (list, tuple)) else annotation.value
        return {
            "type": "point",
            "point": point,
            "pixel_point": point,
        }
    if annotation_type == "segment":
        segment = [list(item) for item in value]
        return {
            "type": "segment",
            "segment": segment,
            "pixel_segment": segment,
        }
    if annotation_type == "point_set":
        points = [list(item) for item in value]
        return {
            "type": "point_set",
            "point_set": points,
            "pixel_point_set": points,
        }
    if annotation_type == "point_sequence":
        points = [list(item) for item in value]
        return {
            "type": "point_sequence",
            "point_sequence": points,
            "pixel_point_sequence": points,
        }
    if annotation_type == "point_map":
        keyed_points = {key: list(item) for key, item in _as_string_keyed_mapping(annotation.value).items()}
        return {
            "type": "point_map",
            "point_map": keyed_points,
            "pixel_point_map": keyed_points,
        }
    if annotation_type == "point_set_map":
        keyed_point_sets = {
            key: [list(point) for point in points]
            for key, points in _as_keyed_list_mapping(annotation.value).items()
        }
        return {
            "type": "point_set_map",
            "point_set_map": keyed_point_sets,
            "pixel_point_set_map": keyed_point_sets,
        }
    if annotation_type == "segment_set":
        pairs = [[list(endpoint) for endpoint in item] for item in value]
        return {
            "type": "segment_set",
            "segment_set": pairs,
            "pixel_segment_set": pairs,
        }
    return {
        "type": annotation_type,
        "value": annotation.value,
    }


def public_witness_summary(annotation: TypedValue) -> dict[str, Any]:
    """Return a non-symbolic witness summary for persisted traces."""

    annotation_type = str(annotation.type)
    count = 1 if annotation_type in {"bbox", "point", "segment"} and annotation.value is not None else _annotation_count(annotation.value)
    return {
        "type": annotation_type,
        "count": count,
    }


def _strip_source_execution_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in _SOURCE_EXECUTION_KEYS:
                continue
            if normalized_key.endswith("_ids") or normalized_key.endswith("_labels"):
                continue
            if normalized_key.endswith("_id") and normalized_key not in {"image_id"}:
                continue
            if normalized_key.endswith("_by_label"):
                continue
            cleaned[normalized_key] = _strip_source_execution_metadata(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_source_execution_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_source_execution_metadata(item) for item in value]
    return value


def sanitize_trace_payload_for_public_annotation(
    trace_payload: Mapping[str, Any],
    *,
    annotation_gt: TypedValue,
) -> dict[str, Any]:
    """Drop source symbolic/grid annotation from persisted trace payload fields."""

    sanitized = {str(key): value for key, value in dict(trace_payload).items()}
    sanitized["execution_trace"] = _strip_source_execution_metadata(sanitized.get("execution_trace", {}))
    sanitized["witness_symbolic"] = public_witness_summary(annotation_gt)
    sanitized["projected_annotation"] = public_projected_annotation(annotation_gt)
    return sanitized
