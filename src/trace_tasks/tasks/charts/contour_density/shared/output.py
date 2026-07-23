"""Trace payload helpers for contour-density chart scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.charts.contour_density.shared.annotations import projected_annotation_payload
from trace_tasks.tasks.charts.contour_density.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.contour_density.shared.rendering import font_assets_for_family
from trace_tasks.tasks.charts.contour_density.shared.state import ContourDataset, RenderedContourScene


def answer_value(dataset: ContourDataset) -> int | str:
    if str(dataset.query.answer_type) == "integer":
        return int(dataset.query.answer)
    return str(dataset.query.answer)


def regions_trace(dataset: ContourDataset) -> Dict[str, Dict[str, Any]]:
    return {
        str(region.label): {
            "region_id": str(region.region_id),
            "option_label": str(region.option_label),
            "center": [round(float(region.center_x), 3), round(float(region.center_y), 3)],
            "density": round(float(region.density), 3),
            "density_level": int(region.density_level),
            "radius": [round(float(region.radius_x), 3), round(float(region.radius_y), 3)],
        }
        for region in dataset.regions
    }


def build_trace_scaffold(
    *,
    dataset: ContourDataset,
    rendered: RenderedContourScene,
    annotation: Any,
    chart_font_family: str,
    relations: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble scene-level trace sections after a public task has bound semantics."""

    resolved_answer = answer_value(dataset)
    relation_fields = {
        **dict(relations),
        "scene_variant": str(dataset.scene_variant),
        "answer": resolved_answer,
        "annotation_roles": dict(dataset.query.annotation_roles),
        "annotation_region_ids": list(dataset.query.annotation_region_ids),
    }
    return {
        "scene_ir": {
            "scene_kind": "chart_contour_density",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(relation_fields),
        },
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "font_assets": font_assets_for_family(str(chart_font_family)),
            **dict(rendered.render_meta),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "region_bboxes_px": dict(rendered.region_bboxes),
            "option_bboxes_px": dict(rendered.option_bboxes),
            "reference_bboxes_px": dict(rendered.reference_bboxes),
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "question_format": "contour_density_field_query",
            "answer": resolved_answer,
            "answer_type": str(dataset.query.answer_type),
            "annotation_type": str(dataset.query.annotation_type),
            "region_count": int(len(dataset.regions)),
            "region_labels": [str(region.label) for region in dataset.regions],
            "regions": dict(regions_trace(dataset)),
            "annotation_roles": dict(dataset.query.annotation_roles),
            "annotation_region_ids": list(dataset.query.annotation_region_ids),
            **dict(dataset.query.trace),
            **dict(relations),
        },
        "witness_symbolic": {
            "type": "contour_density_witness",
            "annotation_type": str(dataset.query.annotation_type),
            "annotation_roles": dict(dataset.query.annotation_roles),
            "annotation_region_ids": list(dataset.query.annotation_region_ids),
            "answer": resolved_answer,
        },
        "projected_annotation": projected_annotation_payload(dataset, annotation),
        "background": dict(rendered.render_meta.get("background_style", {})),
        "post_image_noise": dict(rendered.render_meta.get("post_image_noise", {})),
    }


__all__ = ["answer_value", "build_trace_scaffold", "regions_trace"]
