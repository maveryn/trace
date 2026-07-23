"""Annotation projection helpers for spinner."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.types import TypedValue
from ....shared.bbox_projection import round_bbox


def _round_box(raw: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in round_bbox(raw)]


def scalar_panel_bbox(
    item_bbox_map: Mapping[str, Sequence[float]],
    *,
    item_id: str,
) -> Tuple[TypedValue, Dict[str, Any], Dict[str, Any], str]:
    """Return scalar bbox annotation artifacts for one guaranteed panel witness."""

    key = str(item_id)
    if key not in item_bbox_map:
        raise RuntimeError(f"missing spinner panel bbox for item id {key!r}")
    bbox = _round_box(item_bbox_map[key])
    annotation_gt = TypedValue(type="bbox", value=list(bbox))
    projected_annotation = {
        "type": "bbox",
        "bbox": list(bbox),
        "pixel_bbox": list(bbox),
        "value": list(bbox),
    }
    witness_symbolic = {"type": "bbox", "value": list(bbox)}
    return annotation_gt, projected_annotation, witness_symbolic, "panel_bbox_px"


def pair_panel_bbox_map(
    item_bbox_map: Mapping[str, Sequence[float]],
    *,
    role_item_ids: Mapping[str, str],
) -> Tuple[TypedValue, Dict[str, Any], Dict[str, Any], str, Dict[str, str]]:
    """Return keyed bbox annotation artifacts for two role-bound spinner panels."""

    keyed: Dict[str, list[float]] = {}
    normalized_roles: Dict[str, str] = {}
    for role, item_id in role_item_ids.items():
        key = str(item_id)
        if key not in item_bbox_map:
            raise RuntimeError(f"missing spinner panel bbox for role {role!r}: item id {key!r}")
        normalized_roles[str(role)] = key
        keyed[str(role)] = _round_box(item_bbox_map[key])
    annotation_gt = TypedValue(type="bbox_map", value=dict(keyed))
    projected_annotation = {
        "type": "bbox_map",
        "bbox_map": dict(keyed),
        "pixel_bbox_map": dict(keyed),
        "value": dict(keyed),
    }
    witness_symbolic = {"type": "bbox_map", "value": dict(keyed)}
    return annotation_gt, projected_annotation, witness_symbolic, "keyed_panel_bboxes_px", normalized_roles


__all__ = ["pair_panel_bbox_map", "scalar_panel_bbox"]
