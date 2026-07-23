"""Annotation helpers for orbital-motion visual witnesses."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .state import OrbitSpec


def _point(point: tuple[float, float]) -> list[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def selected_point_annotation(spec: OrbitSpec) -> list[float]:
    """Return the selected answer point as the scalar visual witness."""

    return _point(spec.selected_point)


def focus_annotation_point(spec: OrbitSpec) -> list[float]:
    """Return the selected focus candidate point."""

    return selected_point_annotation(spec)


def speed_annotation_point(spec: OrbitSpec) -> list[float]:
    """Return the selected speed-extremum candidate point."""

    return selected_point_annotation(spec)


def build_point_annotation_payload(annotation_value: Sequence[float]) -> Dict[str, Any]:
    """Build common projected scalar-point annotation trace metadata."""

    point = [round(float(annotation_value[0]), 3), round(float(annotation_value[1]), 3)]
    return {
        "witness_symbolic": {
            "type": "point",
            "count": 1,
        },
        "projected_annotation": {
            "type": "point",
            "point": list(point),
            "pixel_point": list(point),
        },
    }


__all__ = [
    "build_point_annotation_payload",
    "focus_annotation_point",
    "selected_point_annotation",
    "speed_annotation_point",
]
