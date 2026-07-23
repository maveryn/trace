"""Shared helpers for annotated measurement values in geometry tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class PixelAnnotationArtifacts:
    """Normalized public annotation value plus projected trace payload."""

    annotation_type: str
    value: Any
    projected_annotation: Dict[str, Any]


def _round_point(point: Sequence[float]) -> List[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _round_bbox(bbox: Sequence[float]) -> List[float]:
    return [round(float(coord), 3) for coord in bbox]


def _bbox_center(bbox: Sequence[float]) -> List[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def bbox_annotation_artifacts(bbox: Sequence[float]) -> PixelAnnotationArtifacts:
    """Build trace-facing scalar bbox annotation artifacts."""

    value = _round_bbox(bbox)
    projected = {
        "type": "bbox",
        "bbox": list(value),
        "pixel_bbox": list(value),
    }
    return PixelAnnotationArtifacts(annotation_type="bbox", value=list(value), projected_annotation=projected)


def keyed_point_annotation_artifacts(
    points: Mapping[str, Sequence[float]],
    *,
    roles: Sequence[str] | None = None,
) -> PixelAnnotationArtifacts:
    """Build trace-facing keyed point annotation artifacts."""

    selected_roles = [str(role) for role in roles] if roles is not None else [str(role) for role in points.keys()]
    value = {role: _round_point(points[role]) for role in selected_roles}
    projected = {
        "type": "point_map",
        "point_map": dict(value),
        "pixel_point_map": dict(value),
    }
    return PixelAnnotationArtifacts(annotation_type="point_map", value=dict(value), projected_annotation=projected)


def keyed_bbox_annotation_artifacts(
    bboxes: Mapping[str, Sequence[float]],
    *,
    roles: Sequence[str] | None = None,
    include_point_centers: bool = True,
) -> PixelAnnotationArtifacts:
    """Build trace-facing keyed bbox annotation artifacts plus bbox centers."""

    selected_roles = [str(role) for role in roles] if roles is not None else [str(role) for role in bboxes.keys()]
    value = {role: _round_bbox(bboxes[role]) for role in selected_roles}
    projected = {
        "type": "bbox_map",
        "bbox_map": dict(value),
        "pixel_bbox_map": dict(value),
    }
    if bool(include_point_centers):
        centers = {role: _bbox_center(bbox) for role, bbox in value.items()}
        projected["point_map"] = dict(centers)
        projected["pixel_point_map"] = dict(centers)
    return PixelAnnotationArtifacts(annotation_type="bbox_map", value=dict(value), projected_annotation=projected)


def bbox_set_annotation_artifacts(bboxes: Sequence[Sequence[float]]) -> PixelAnnotationArtifacts:
    """Build trace-facing unordered bbox-set annotation artifacts plus centers."""

    value = [_round_bbox(bbox) for bbox in bboxes]
    centers = [_bbox_center(bbox) for bbox in value]
    projected = {
        "type": "bbox_set",
        "bbox_set": list(value),
        "pixel_bbox_set": list(value),
        "point_set": list(centers),
        "pixel_point_set": list(centers),
    }
    return PixelAnnotationArtifacts(annotation_type="bbox_set", value=list(value), projected_annotation=projected)


def point_set_annotation_artifacts(points: Sequence[Sequence[float]]) -> PixelAnnotationArtifacts:
    """Build trace-facing unordered point-set annotation artifacts."""

    value = [_round_point(point) for point in points]
    projected = {
        "type": "point_set",
        "point_set": list(value),
        "pixel_point_set": list(value),
    }
    return PixelAnnotationArtifacts(annotation_type="point_set", value=list(value), projected_annotation=projected)


def _coerce_value_for_json(value: Any) -> Any:
    """Normalize one measurement value into JSON-safe primitive form."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        rounded = round(float(value))
        if abs(float(value) - float(rounded)) <= 1e-9:
            return int(rounded)
        return float(value)
    return str(value)


def build_role_value_annotation(
    *,
    roles: Sequence[str],
    role_to_annotation: Mapping[str, str],
    role_to_value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build annotation->value annotation payload from role-bound values.

    Duplicate annotation tokens are permitted only when they map to the same value.
    """
    annotation_map: Dict[str, Any] = {}
    for role in [str(item) for item in roles]:
        if str(role) not in role_to_annotation:
            raise ValueError(f"missing annotation token for role: {role}")
        if str(role) not in role_to_value:
            raise ValueError(f"missing measurement value for role: {role}")
        annotation_token = str(role_to_annotation[str(role)]).strip()
        if not annotation_token:
            raise ValueError(f"empty annotation token for role: {role}")
        coerced_value = _coerce_value_for_json(role_to_value[str(role)])
        if annotation_token in annotation_map:
            if annotation_map[annotation_token] != coerced_value:
                raise ValueError(f"conflicting values for annotation token in annotation map: {annotation_token}")
            continue
        annotation_map[annotation_token] = coerced_value
    return {str(key): value for key, value in annotation_map.items()}


def build_annotation_value_tokens(annotation_map: Mapping[str, Any]) -> List[str]:
    """Build one deterministic unordered symbolic annotation set from annotation values."""
    tokens: List[str] = []
    for annotation, value in sorted(((str(key), item) for key, item in annotation_map.items()), key=lambda item: item[0]):
        token = f"{annotation}={format_annotation_value(value)}"
        tokens.append(str(token))
    return list(tokens)


def build_annotation_value_point_map(
    *,
    annotation_map: Mapping[str, Any],
    annotation_centers: Mapping[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Project annotation=value annotation tokens back to annotation center points."""
    point_map: Dict[str, List[float]] = {}
    for annotation, value in sorted(((str(key), item) for key, item in annotation_map.items()), key=lambda item: item[0]):
        if annotation not in annotation_centers:
            continue
        center = annotation_centers[annotation]
        if not isinstance(center, Sequence) or len(center) != 2:
            continue
        token = f"{annotation}={format_annotation_value(value)}"
        point_map[str(token)] = [float(center[0]), float(center[1])]
    return dict(point_map)


def format_annotation_value(value: Any) -> str:
    """Format one measurement value for on-image numeric annotation text."""
    coerced = _coerce_value_for_json(value)
    return str(coerced)
