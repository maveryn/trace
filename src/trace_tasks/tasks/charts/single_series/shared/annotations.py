"""Annotation projection helpers for single-series chart marks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.cartesian.annotations import projected_mark_annotation


def mark_annotation(
    *,
    rendered_scene: Any,
    labels: Sequence[str],
    annotation_kind: str,
    roles: Mapping[str, str] | None = None,
) -> tuple[TypedValue, dict[str, Any], list[str]]:
    """Project symbolic mark labels into scalar, set, or keyed point annotations."""

    ordered_labels = [str(label) for label in labels]
    projection = projected_mark_annotation(rendered_scene, ordered_labels)
    point_map = {
        str(label): list(point)
        for label, point in projection.get("pixel_point_map", {}).items()
    }
    if str(annotation_kind) == "point":
        point = list(projection.get("pixel_point_set", [[]])[0]) if projection.get("pixel_point_set") else []
        payload = {
            "type": "point",
            "point": list(point),
            "pixel_point": list(point),
            "label": str(ordered_labels[0]) if ordered_labels else "",
            **dict(projection),
        }
        return TypedValue(type="point", value=list(point)), dict(payload), ordered_labels
    if str(annotation_kind) == "point_map":
        role_labels = {str(role): str(label) for role, label in dict(roles or {}).items()}
        role_points = {
            str(role): list(point_map[str(label)])
            for role, label in role_labels.items()
            if str(label) in point_map
        }
        payload = {
            "type": "point_map",
            "point_map": dict(role_points),
            "pixel_point_map": dict(role_points),
            "role_labels": dict(role_labels),
            **dict(projection),
        }
        return TypedValue(type="point_map", value=dict(role_points)), dict(payload), list(role_labels.values())
    if str(annotation_kind) != "point_set":
        raise ValueError(f"unsupported mark annotation kind: {annotation_kind}")
    points = [list(point) for point in projection.get("pixel_point_set", [])]
    payload = {
        "type": "point_set",
        "point_set": list(points),
        "pixel_point_set": list(points),
        **dict(projection),
    }
    return TypedValue(type="point_set", value=list(points)), dict(payload), ordered_labels


def annotation_refs(
    *,
    labels: Sequence[str],
    annotation_value: Any,
    annotation_kind: str,
) -> list[dict[str, Any]]:
    """Return symbolic-to-pixel references for review overlays."""

    if str(annotation_kind) == "point":
        return [{"label": str(labels[0]) if labels else "", "point_xy": list(annotation_value)}]
    if str(annotation_kind) == "point_map":
        return [
            {"role": str(role), "point_xy": list(point)}
            for role, point in dict(annotation_value).items()
        ]
    return [
        {"label": str(label), "point_xy": list(point)}
        for label, point in zip(labels, annotation_value)
    ]


__all__ = ["annotation_refs", "mark_annotation"]
