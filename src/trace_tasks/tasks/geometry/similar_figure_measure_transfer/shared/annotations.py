"""Annotation helpers for similar-figure scenes."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.geometry.shared.vector2d import point_to_list

from .state import FigureGeometry, Point


def point_map_for_labels(geometry: FigureGeometry, labels: Sequence[str]) -> dict[str, list[float]]:
    """Return annotation points for the requested visible vertex labels."""

    point_by_label: dict[str, Point] = {
        **{str(label): point for label, point in zip(geometry.source_labels, geometry.source_vertices)},
        **{str(label): point for label, point in zip(geometry.target_labels, geometry.target_vertices)},
    }
    annotation: dict[str, list[float]] = {}
    for label in labels:
        text = str(label)
        if text in annotation:
            continue
        if text not in point_by_label:
            raise KeyError(f"annotation label {text!r} was not rendered")
        annotation[text] = point_to_list(point_by_label[text])
    return annotation
