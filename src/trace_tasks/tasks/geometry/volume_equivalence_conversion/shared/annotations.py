"""Annotation helpers for volume-equivalence conversion diagrams."""

from __future__ import annotations

from collections.abc import Sequence

from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .state import BBox, RenderedScene

MISSING_DIMENSION_ANNOTATION_KEYS: tuple[str, ...] = (
    "source_solid_bbox",
    "target_solid_bbox",
)
OPTION_ANNOTATION_KEYS: tuple[str, ...] = (
    "selected_option_bbox",
)


def annotation_bbox_map(rendered: RenderedScene, keys: Sequence[str]) -> dict[str, list[float]]:
    return {str(key): bbox_to_list(rendered.annotation_bboxes[str(key)]) for key in keys}


def annotation_bbox(rendered: RenderedScene, key: str) -> list[float]:
    return bbox_to_list(rendered.annotation_bboxes[str(key)])


def projected_annotation(annotation_value: dict[str, list[float]]) -> dict[str, object]:
    return {
        "type": "bbox_map",
        "bbox_map": dict(annotation_value),
        "pixel_bbox_map": dict(annotation_value),
    }


def projected_bbox_annotation(annotation_value: list[float]) -> dict[str, object]:
    return {
        "type": "bbox",
        "bbox": list(annotation_value),
        "pixel_bbox": list(annotation_value),
    }


def example_bbox_for_key(key: str) -> list[int]:
    examples = {
        "source_solid_bbox": [120, 205, 310, 420],
        "target_solid_bbox": [535, 205, 720, 420],
        "source_dimension_region_bbox": [130, 440, 305, 525],
        "target_dimension_region_bbox": [545, 440, 735, 525],
        "target_unknown_region_bbox": [555, 475, 715, 510],
        "selected_option_bbox": [470, 260, 635, 410],
        "selected_option_dimension_region_bbox": [490, 370, 620, 430],
    }
    return list(examples[str(key)])


__all__ = [
    "MISSING_DIMENSION_ANNOTATION_KEYS",
    "OPTION_ANNOTATION_KEYS",
    "annotation_bbox",
    "annotation_bbox_map",
    "example_bbox_for_key",
    "projected_bbox_annotation",
    "projected_annotation",
]
