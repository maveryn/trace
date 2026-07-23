"""Annotation helpers for overlap-grid icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.annotation import matching_scene_cell_bbox_annotation


def matching_overlap_cell_bbox_set_annotation(
    *,
    scene_cells: Sequence[Mapping[str, Any]],
    matching_labels: Sequence[str],
) -> dict[str, Any]:
    """Return bbox-set annotation for matching labeled Scene cells."""

    return matching_scene_cell_bbox_annotation(
        scene_cells=scene_cells,
        matching_labels=matching_labels,
    )


__all__ = ["matching_overlap_cell_bbox_set_annotation"]
