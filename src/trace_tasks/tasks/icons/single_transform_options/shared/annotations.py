"""Annotation helpers for single-transform option icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def matching_scene_cell(scene_cells: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the unique rendered option cell marked as correct."""

    matches = [cell for cell in scene_cells if bool(cell.get("is_match"))]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one matching option cell, found {len(matches)}")
    return matches[0]


def bbox_map_roles(
    *,
    reference_cell: Mapping[str, Any],
    selected_cell: Mapping[str, Any],
) -> dict[str, list[int]]:
    """Build the role-bound bbox map used by transform option tasks."""

    return {
        "reference_icon": [int(value) for value in reference_cell["icon_bbox_xyxy"]],
        "selected_option": [int(value) for value in selected_cell["cell_bbox_xyxy"]],
    }


__all__ = ["bbox_map_roles", "matching_scene_cell"]
