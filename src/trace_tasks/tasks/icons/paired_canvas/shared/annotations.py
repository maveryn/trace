"""Annotation helpers for paired-canvas icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.icon_scene import sort_bboxes_reading_order


def bboxes_from_icon_indices(
    *,
    panel_icons: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> list[list[int]]:
    """Return reading-order bboxes from selected rendered panel icon indices."""

    selected: list[list[int]] = []
    for index in indices:
        bbox = panel_icons[int(index)].get("bbox_xyxy", ())
        selected.append([int(round(float(value))) for value in bbox])
    return sort_bboxes_reading_order(selected)


__all__ = ["bboxes_from_icon_indices"]
