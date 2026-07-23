"""Annotation helpers for pressure-volume diagrams."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts
from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples


def scalar_bbox_artifacts(bbox: Sequence[float]) -> Any:
    """Return normalized scalar-bbox annotation artifacts."""

    return bbox_annotation_artifacts(bbox)


def build_prompt_examples(*, answer_type: str) -> tuple[str, str]:
    """Return one stable prompt JSON example for PV tasks."""

    if str(answer_type) == "option_letter":
        return build_prompt_json_examples(
            annotation_value=[104, 88, 430, 284],
            answer_type="option_letter",
        )
    return build_prompt_json_examples(
        annotation_value=[248, 188, 710, 526],
        answer_type="integer",
    )


def single_annotation_bbox(annotation_bboxes: Sequence[Sequence[float]]) -> list[float]:
    """Return exactly one annotation box from a rendered scene."""

    boxes = [[round(float(value), 3) for value in bbox[:4]] for bbox in annotation_bboxes]
    if len(boxes) != 1:
        raise ValueError(f"expected exactly one annotation bbox, got {len(boxes)}")
    return list(boxes[0])


__all__ = ["build_prompt_examples", "scalar_bbox_artifacts", "single_annotation_bbox"]
