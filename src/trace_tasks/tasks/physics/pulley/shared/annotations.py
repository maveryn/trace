"""Annotation helpers for pulley diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples


FORCE_LABEL_KEYS: tuple[str, str] = ("known_force_label", "unknown_force_label")


def projected_bbox_map(annotation_value: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Return trace metadata for a bbox-map annotation."""

    boxes = {str(key): [float(value) for value in bbox] for key, bbox in annotation_value.items()}
    return {
        "type": "bbox_map",
        "bbox_map": dict(boxes),
        "pixel_bbox_map": dict(boxes),
    }


def build_prompt_examples() -> tuple[str, str]:
    """Return one stable prompt JSON example for pulley force queries."""

    annotation = {
        "supporting_strands_region": [414, 160, 481, 434],
        "known_force_label": [649, 538, 819, 612],
        "unknown_force_label": [1036, 190, 1196, 350],
    }
    return build_prompt_json_examples(annotation_value=annotation, answer_type="integer")


__all__ = ["FORCE_LABEL_KEYS", "build_prompt_examples", "projected_bbox_map"]
