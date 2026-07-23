"""Annotation helpers for icon-cutout scenes."""

from __future__ import annotations

from typing import Any, Dict

from .state import IconCutoutScenePayload


def matching_option_cell(scene_payload: IconCutoutScenePayload) -> Dict[str, Any]:
    """Return the unique option cell matching the source fragment."""

    matches = [dict(cell) for cell in scene_payload.scene_cells if bool(cell.get("is_match"))]
    if len(matches) != 1:
        raise ValueError("icon-cutout scene must contain exactly one matching full-icon option")
    return dict(matches[0])


def fragment_option_annotation_boxes(
    scene_payload: IconCutoutScenePayload,
    selected_option: Dict[str, Any],
) -> Dict[str, Any]:
    """Return minimal visual witness boxes for the fragment and selected option."""

    return {
        "source_fragment": list(scene_payload.reference_cell["fragment_bbox_xyxy"]),
        "selected_option": list(selected_option["cell_bbox_xyxy"]),
    }


__all__ = ["fragment_option_annotation_boxes", "matching_option_cell"]
