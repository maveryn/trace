"""Trace serialization helpers for polygon equation diagrams."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready

from .state import PolygonEquationCase, RenderedPolygonEquationScene


def render_map(rendered: RenderedPolygonEquationScene) -> dict[str, Any]:
    """Build a JSON-ready render map for one rendered polygon."""

    return {
        "coord_space": "pixel",
        "vertices": {
            str(label): [round(float(point[0]), 3), round(float(point[1]), 3)]
            for label, point in rendered.vertex_points.items()
        },
        "point_label_bboxes": geometry_json_ready(rendered.point_label_bboxes, round_floats=False),
        "side_label_bboxes": geometry_json_ready(rendered.side_label_bboxes, round_floats=False),
        "angle_label_bboxes": geometry_json_ready(rendered.angle_label_bboxes, round_floats=False),
        "marker_bboxes": geometry_json_ready(rendered.marker_bboxes, round_floats=False),
    }


def trace_common(
    *,
    case: PolygonEquationCase,
    rendered: RenderedPolygonEquationScene,
    annotation_artifacts: PixelAnnotationArtifacts,
) -> dict[str, Any]:
    """Build shared trace payload fields for one task-owned case."""

    case_fields = case.trace_fields()
    vertices = {
        str(label): [round(float(point[0]), 3), round(float(point[1]), 3)]
        for label, point in rendered.vertex_points.items()
    }
    return {
        "entities": [
            {
                "type": "polygon",
                "side_count": int(case.side_count),
                "labels": list(case.vertex_labels()),
                "vertices": dict(vertices),
            }
        ],
        "relations": {
            "type": str(case.relation),
            "annotation_roles": list(annotation_artifacts.value.keys()),
            **dict(case_fields),
        },
        "execution": {
            "answer_type": "integer",
            "answer": int(case.answer),
            "annotation_type": str(annotation_artifacts.annotation_type),
            "annotation_roles": list(annotation_artifacts.value.keys()),
            **dict(case_fields),
        },
        "witness": {
            "type": "polygon_equation_diagram",
            "source_witness_type": str(annotation_artifacts.annotation_type),
            "original_annotation_value": annotation_artifacts.value,
            **dict(case_fields),
        },
        "query_params": dict(case_fields),
    }


__all__ = ["render_map", "trace_common"]
