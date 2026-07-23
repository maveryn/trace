"""Annotation projection helpers for cylinder-wrap diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def projected_keyed_annotation(annotation_type: str, annotation_value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the review/reward projection for a keyed point or bbox map."""

    if str(annotation_type) == "bbox_map":
        return {
            "type": "bbox_map",
            "bbox_map": dict(annotation_value),
            "pixel_bbox_map": dict(annotation_value),
        }
    if str(annotation_type) == "point_map":
        return {
            "type": "point_map",
            "point_map": dict(annotation_value),
            "pixel_point_map": dict(annotation_value),
        }
    raise ValueError(f"unsupported keyed annotation type: {annotation_type}")


__all__ = ["projected_keyed_annotation"]
