"""Annotation helpers for named-strip icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.annotation import icon_bbox_set_annotation
from ...shared.icon_scene import sort_bboxes_reading_order


def selected_run_bbox_set_annotation(
    icons: Sequence[Mapping[str, Any] | Any],
    *,
    expected_count: int,
) -> dict[str, Any]:
    """Build bbox-set annotation for rendered selected-run members."""

    selected_bboxes = []
    for icon in icons:
        is_selected = bool(getattr(icon, "is_selected_run_member", False))
        bbox = getattr(icon, "bbox_xyxy", None)
        if is_selected:
            selected_bboxes.append(bbox)
    annotation_bboxes = sort_bboxes_reading_order(tuple(tuple(int(value) for value in bbox) for bbox in selected_bboxes))
    if len(annotation_bboxes) != int(expected_count):
        raise RuntimeError("rendered named-strip annotation length does not match answer")
    return icon_bbox_set_annotation(annotation_bboxes)


__all__ = ["selected_run_bbox_set_annotation"]
