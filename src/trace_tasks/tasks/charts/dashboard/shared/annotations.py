"""Annotation projection primitives for dashboard chart tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.types import TypedValue

from .state import AnnotationRef, Category, Panel, RenderedDashboard


def annotation_records(*, refs: Sequence[AnnotationRef], rendered: RenderedDashboard, panels: Sequence[Panel], categories: Sequence[Category]) -> list[dict[str, Any]]:
    panels_by_id = {str(panel.panel_id): panel for panel in panels}
    categories_by_id = {str(category.category_id): category for category in categories}
    records: list[dict[str, Any]] = []
    for panel_id, category_id in refs:
        records.append(
            {
                "panel_id": str(panel_id),
                "panel_name": str(panels_by_id[str(panel_id)].name),
                "category_id": str(category_id),
                "category_label": str(categories_by_id[str(category_id)].label),
                "point_xy": list(rendered.support_points_px[str(panel_id)][str(category_id)]),
                "bbox_xyxy": list(rendered.support_bboxes_px[str(panel_id)][str(category_id)]),
            }
        )
    return records


def point_for_ref(rendered: RenderedDashboard, ref: AnnotationRef) -> list[int]:
    panel_id, category_id = ref
    point = rendered.support_points_px[str(panel_id)][str(category_id)]
    return [int(point[0]), int(point[1])]


def point_set_artifacts(*, refs: Sequence[AnnotationRef], rendered: RenderedDashboard, panels: Sequence[Panel], categories: Sequence[Category]) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    points = [point_for_ref(rendered, ref) for ref in refs]
    records = annotation_records(refs=refs, rendered=rendered, panels=panels, categories=categories)
    projected = {
        "type": "point_set",
        "point_set": list(points),
        "pixel_point_set": list(points),
        "annotation_refs": records,
    }
    witness = {"type": "dashboard_cross_panel_witness", "annotation_refs": [list(ref) for ref in refs]}
    return TypedValue(type="point_set", value=list(points)), witness, projected


def point_artifacts(*, ref: AnnotationRef, rendered: RenderedDashboard, panels: Sequence[Panel], categories: Sequence[Category]) -> tuple[TypedValue, dict[str, Any], dict[str, Any], tuple[AnnotationRef, ...]]:
    refs = (tuple(str(value) for value in ref),)
    point = point_for_ref(rendered, refs[0])
    records = annotation_records(refs=refs, rendered=rendered, panels=panels, categories=categories)
    projected = {
        "type": "point",
        "point": list(point),
        "pixel_point": list(point),
        "annotation_refs": records,
    }
    witness = {"type": "dashboard_cross_panel_witness", "annotation_refs": [list(refs[0])]}
    return TypedValue(type="point", value=list(point)), witness, projected, refs


def keyed_point_artifacts(*, role_to_ref: Mapping[str, AnnotationRef], rendered: RenderedDashboard, panels: Sequence[Panel], categories: Sequence[Category]) -> tuple[TypedValue, dict[str, Any], dict[str, Any], tuple[AnnotationRef, ...]]:
    refs = tuple((str(panel_id), str(category_id)) for panel_id, category_id in role_to_ref.values())
    value = {str(role): point_for_ref(rendered, ref) for role, ref in role_to_ref.items()}
    records = annotation_records(refs=refs, rendered=rendered, panels=panels, categories=categories)
    points = {str(key): [int(point[0]), int(point[1])] for key, point in value.items()}
    projected = {
        "type": "point_map",
        "point_map": dict(points),
        "pixel_point_map": dict(points),
        "annotation_refs": records,
    }
    witness = {"type": "dashboard_cross_panel_witness", "annotation_refs": [list(ref) for ref in refs]}
    return TypedValue(type="point_map", value=dict(points)), witness, projected, refs


__all__ = [
    "annotation_records",
    "keyed_point_artifacts",
    "point_artifacts",
    "point_for_ref",
    "point_set_artifacts",
]
