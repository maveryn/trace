"""Annotation helpers for ray-optics point-set witnesses."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.graph_point_annotation import (
    labeled_grid_point_annotation_artifacts,
)
from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples

from .state import RAY_EVENT_BOUNCE


def build_prompt_examples(ray_event_kind: str) -> Tuple[str, str]:
    """Return prompt JSON examples for the active ray-optics objective."""

    if str(ray_event_kind) == RAY_EVENT_BOUNCE:
        annotation = [[242, 190], [346, 294]]
    else:
        annotation = [[190, 138], [398, 346]]
    return build_prompt_json_examples(annotation_value=annotation, answer_type="integer")


def empty_pixel_point_set_annotation_artifacts(*, witness_type: str) -> Dict[str, Any]:
    """Return an empty pixel point-set annotation payload."""

    return {
        "annotation_type": "point_set",
        "annotation_value": [],
        "required_labels": [],
        "witness_symbolic": {
            "type": str(witness_type),
            "count": 0,
        },
        "projected_annotation": {
            "type": "point_set",
            "point_set": [],
            "pixel_point_set": [],
            "pixel_point_map": {},
        },
    }


def pixel_point_set_annotation_artifacts(
    *,
    points_by_label: Mapping[str, Sequence[float]],
    graph_origin: Sequence[float],
    graph_spacing: int,
    witness_type: str,
    ordered_labels: Sequence[str],
) -> Dict[str, Any]:
    """Expose graph-derived witnesses as public pixel point sets."""

    labels = [str(label) for label in ordered_labels]
    if not labels:
        return empty_pixel_point_set_annotation_artifacts(
            witness_type=str(witness_type)
        )
    labeled = labeled_grid_point_annotation_artifacts(
        points_by_label=points_by_label,
        graph_origin=graph_origin,
        graph_spacing=int(graph_spacing),
        witness_type=str(witness_type),
        ordered_labels=tuple(labels),
    )
    projected = dict(labeled["projected_annotation"])
    pixel_map = {
        str(label): [float(point[0]), float(point[1])]
        for label, point in dict(projected.get("pixel_point_map", {})).items()
    }
    point_set = [list(pixel_map[str(label)]) for label in labels]
    return {
        "annotation_type": "point_set",
        "annotation_value": [list(point) for point in point_set],
        "required_labels": list(labels),
        "witness_symbolic": dict(labeled["witness_symbolic"]),
        "projected_annotation": {
            "type": "point_set",
            "point_set": [list(point) for point in point_set],
            "pixel_point_set": [list(point) for point in point_set],
            "pixel_point_map": dict(pixel_map),
        },
    }


__all__ = [
    "build_prompt_examples",
    "empty_pixel_point_set_annotation_artifacts",
    "pixel_point_set_annotation_artifacts",
]
