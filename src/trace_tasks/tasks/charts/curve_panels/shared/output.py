"""Trace payload helpers for curve-panel chart tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.charts.curve_panels.shared.annotations import (
    projected_annotation_payload,
)
from trace_tasks.tasks.charts.curve_panels.shared.defaults import (
    SCENE_ID,
    SCENE_NAMESPACE,
    resolve_int,
)
from trace_tasks.tasks.charts.curve_panels.shared.rendering import _render_dataset
from trace_tasks.tasks.charts.curve_panels.shared.state import (
    CurvePanelDataset,
    RenderedCurvePanels,
)
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    sample_chart_font_family,
)
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


def values_by_panel_method(
    dataset: CurvePanelDataset,
) -> Dict[str, Dict[str, list[int]]]:
    """Return visible values keyed by panel and method label."""

    return {
        str(panel.panel_label): {
            str(curve.method_label): [int(value) for value in curve.values]
            for curve in panel.curves
        }
        for panel in dataset.panels
    }


def render_dataset(
    dataset: CurvePanelDataset, *, params: Mapping[str, Any], instance_seed: int
) -> tuple[RenderedCurvePanels, str]:
    """Render a curve-panel dataset with a sampled chart font."""

    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = _render_dataset(
            dataset, params=params, instance_seed=int(instance_seed)
        )
    return rendered, str(chart_font_family)


def build_trace_scaffold(
    *,
    dataset: CurvePanelDataset,
    rendered: RenderedCurvePanels,
    annotation_type: str,
    annotation: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    chart_font_family: str,
    params: Mapping[str, Any],
    relations: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble neutral scene/render trace sections after a task binds semantics."""

    method_labels = [str(curve.method_label) for curve in dataset.panels[0].curves]
    answerability = str(relations.get("answerability", "answerable"))
    absence_fields = (
        {"absence_proof": dict(relations["absence_proof"])}
        if answerability == "unanswerable" and "absence_proof" in relations
        else {}
    )
    relation_fields = {
        "scene_variant": str(dataset.scene_variant),
        "answer": dataset.query.answer,
        "annotation_panel_labels": list(dataset.query.annotation_panel_labels),
        "annotation_point_ids": list(dataset.query.annotation_point_ids),
        "annotation_keyed_point_ids": dict(dataset.query.annotation_keyed_point_ids),
        "annotation_intersection_ids": list(dataset.query.annotation_intersection_ids),
        "annotation_threshold_crossing_ids": list(
            dataset.query.annotation_threshold_crossing_ids
        ),
        **dict(relations),
    }
    return {
        "scene_ir": {
            "scene_kind": f"chart_scientific_{str(dataset.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(relation_fields),
        },
        "render_spec": {
            "canvas_width": resolve_int(params, "canvas_width", 1200),
            "canvas_height": resolve_int(params, "canvas_height", 1000),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "font_assets": chart_font_asset_metadata(str(chart_font_family)),
            **dict(rendered.render_meta),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "panel_bboxes_px": dict(rendered.panel_bboxes),
            "panel_plot_bboxes_px": dict(rendered.panel_plot_bboxes),
            "point_bboxes_px": dict(rendered.point_bboxes),
            "intersection_bboxes_px": dict(rendered.intersection_bboxes),
            "threshold_crossing_bboxes_px": dict(rendered.threshold_crossing_bboxes),
            "legend_bboxes_px": dict(rendered.legend_bboxes),
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "answer": dataset.query.answer,
            "answer_type": str(dataset.query.answer_type),
            "panel_labels": [str(panel.panel_label) for panel in dataset.panels],
            "method_labels": list(method_labels),
            "x_values": list(dataset.x_values),
            "y_range": [int(dataset.y_min), int(dataset.y_max)],
            "panel_count": int(len(dataset.panels)),
            "method_count": int(len(method_labels)),
            "values_by_panel_method": values_by_panel_method(dataset),
            "annotation_panel_labels": list(dataset.query.annotation_panel_labels),
            "annotation_point_ids": list(dataset.query.annotation_point_ids),
            "annotation_keyed_point_ids": dict(
                dataset.query.annotation_keyed_point_ids
            ),
            "annotation_intersection_ids": list(
                dataset.query.annotation_intersection_ids
            ),
            "annotation_threshold_crossing_ids": list(
                dataset.query.annotation_threshold_crossing_ids
            ),
            "question_format": "curve_panels_subplot_query",
            **dict(dataset.query.trace),
            **dict(relations),
        },
        "witness_symbolic": {
            "type": "curve_panels_subplot",
            "panel_labels": list(dataset.query.annotation_panel_labels),
            "point_ids": list(dataset.query.annotation_point_ids),
            "keyed_point_ids": dict(dataset.query.annotation_keyed_point_ids),
            "intersection_ids": list(dataset.query.annotation_intersection_ids),
            "threshold_crossing_ids": list(
                dataset.query.annotation_threshold_crossing_ids
            ),
            "answer": dataset.query.answer,
            "answerability": answerability,
            **absence_fields,
        },
        "projected_annotation": projected_annotation_payload(
            dataset=dataset,
            annotation_type=str(annotation_type),
            annotation=annotation,
        ),
    }


__all__ = ["build_trace_scaffold", "render_dataset", "values_by_panel_method"]
