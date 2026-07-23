"""Annotation helpers for Venn-field icon scenes."""

from __future__ import annotations

from typing import Sequence

from ...shared.annotation import icon_bbox_set_annotation
from ...shared.icon_scene import sort_bboxes_reading_order


def counted_icon_bbox_set_annotation(bboxes: Sequence[Sequence[int | float]]) -> dict:
    """Return sorted bbox-set annotation for counted Venn-field icons."""

    return icon_bbox_set_annotation(sort_bboxes_reading_order(tuple(bboxes)))


__all__ = ["counted_icon_bbox_set_annotation"]
