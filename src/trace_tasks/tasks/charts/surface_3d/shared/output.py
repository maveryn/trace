"""Trace payload helpers for synthetic 3D chart scenes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata
from trace_tasks.tasks.charts.surface_3d.shared.state import Surface3DDataset, Surface3DRenderArtifacts


def point_rows(dataset: Surface3DDataset) -> list[dict[str, Any]]:
    """Return JSON-stable rows for every rendered point."""

    return [
        {
            "point_id": str(point.point_id),
            "label": str(point.label),
            "x_value": round(float(point.x_value), 3),
            "y_value": round(float(point.y_value), 3),
            "z_value": round(float(point.z_value), 3),
        }
        for point in dataset.points
    ]


def surface_cell_rows(dataset: Surface3DDataset) -> list[dict[str, Any]]:
    """Return JSON-stable rows for every surface cell marker."""

    return [
        {
            "cell_id": str(cell.cell_id),
            "x_label": str(cell.x_label),
            "y_label": str(cell.y_label),
            "value": int(cell.value),
        }
        for cell in dataset.surface_cells
    ]


def panel_rows(dataset: Surface3DDataset) -> list[dict[str, Any]]:
    """Return JSON-stable rows for every small-multiple panel."""

    return [
        {
            "panel_label": str(panel.panel_label),
            "values": [int(value) for value in panel.values],
            "value_range": int(max(panel.values) - min(panel.values)),
        }
        for panel in dataset.panels
    ]


def build_trace_payload(
    *,
    artifacts: Surface3DRenderArtifacts,
    dataset: Surface3DDataset,
    answer_value: int | str,
    answer_type: str,
    question_format: str,
    relations: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble neutral trace sections after public task binding."""

    rendered = artifacts.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": "chart_surface_3d",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "answer": answer_value,
                "scene_variant": str(dataset.scene_variant),
                **dict(relations),
            },
        },
        "render_spec": {
            "canvas_width": int(artifacts.render_params.canvas_width),
            "canvas_height": int(artifacts.render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "background_style": dict(artifacts.background_style),
            "information_scene_style": dict(artifacts.background_style.get("information_scene_style", {})),
            "post_image_noise": dict(artifacts.post_image_noise),
            "font_assets": chart_font_asset_metadata(str(artifacts.chart_font_family)),
            "layout_jitter": dict(artifacts.render_params.layout_jitter_meta),
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "axis_labels": {
                "x": str(dataset.x_axis_label),
                "y": str(dataset.y_axis_label),
                "z": str(dataset.z_axis_label),
            },
            "reference_y_value": None if dataset.reference_y_value is None else float(dataset.reference_y_value),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "point_bboxes_px": dict(rendered.point_bboxes_px),
            "surface_cell_bboxes_px": dict(rendered.surface_cell_bboxes_px),
            "panel_bboxes_px": dict(rendered.panel_bboxes_px),
        },
        "execution_trace": {
            "question_format": str(question_format),
            "scene_variant": str(dataset.scene_variant),
            "answer": answer_value,
            "answer_type": str(answer_type),
            "x_axis_label": str(dataset.x_axis_label),
            "y_axis_label": str(dataset.y_axis_label),
            "z_axis_label": str(dataset.z_axis_label),
            "x_range": [float(value) for value in dataset.x_range],
            "y_range": [float(value) for value in dataset.y_range],
            "z_range": [float(value) for value in dataset.z_range],
            "reference_y_value": None if dataset.reference_y_value is None else float(dataset.reference_y_value),
            "point_count": len(dataset.points),
            "surface_cell_count": len(dataset.surface_cells),
            "panel_count": len(dataset.panels),
            "points": point_rows(dataset),
            "surface_cells": surface_cell_rows(dataset),
            "panels": panel_rows(dataset),
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "background": dict(artifacts.background_style),
        "post_image_noise": dict(artifacts.post_image_noise),
    }


__all__ = ["build_trace_payload", "panel_rows", "point_rows", "surface_cell_rows"]
