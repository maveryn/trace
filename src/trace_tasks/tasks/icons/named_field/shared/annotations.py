"""Annotation projection helpers for named-field icon tasks."""

from __future__ import annotations

from typing import Any, Sequence

from ...shared.annotation import icon_bbox_set_annotation
from ...shared.icon_scene import sort_bboxes_reading_order

from .metrics import boolean_counted_instance_ids, counterfactual_counted_instance_ids


def bbox_set_from_bboxes(bboxes: Sequence[Sequence[int | float]]) -> dict[str, Any]:
    """Return the standard bbox-set annotation payload from icon bboxes."""

    sorted_bboxes = sort_bboxes_reading_order(tuple(bboxes))
    return dict(icon_bbox_set_annotation(sorted_bboxes))


def boolean_annotation_bboxes(sample: Any, instances: Sequence[Any]) -> list[list[int]]:
    """Return sorted bboxes for icons counted by a Boolean predicate."""

    counted = set(boolean_counted_instance_ids(sample, instances))
    return sort_bboxes_reading_order(tuple(instance.bbox_xyxy for instance in instances if str(instance.instance_id) in counted))


def counterfactual_annotation_bboxes(sample: Any, instances: Sequence[Any]) -> list[list[int]]:
    """Return sorted bboxes for pre-edit icons counted after the hypothetical edit."""

    counted = set(counterfactual_counted_instance_ids(sample))
    return sort_bboxes_reading_order(tuple(instance.bbox_xyxy for instance in instances if str(instance.instance_id) in counted))


__all__ = [
    "boolean_annotation_bboxes",
    "bbox_set_from_bboxes",
    "counterfactual_annotation_bboxes",
]
