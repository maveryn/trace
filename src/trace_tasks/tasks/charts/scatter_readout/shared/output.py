"""Neutral trace-output helpers for scatter-readout chart scenes."""

from __future__ import annotations

from typing import Any, Sequence

from .rendering import font_assets_payload
from .state import Point, QueryBinding, ScatterReadoutRenderResult, SceneDataset, Series


def answer_value(binding: QueryBinding) -> int | str:
    if str(binding.answer_type) == "integer":
        return int(binding.answer)
    return str(binding.answer)


def point_records(dataset: SceneDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series_item in dataset.series:
        for point in series_item.points:
            rows.append(
                {
                    "point_id": str(point.point_id),
                    "series_label": str(point.series_label),
                    "x_label": str(point.x_label),
                    "x_index": int(point.x_index),
                    "y_value": int(point.y_value),
                }
            )
    return rows


def series_records(series: Sequence[Series]) -> list[dict[str, Any]]:
    return [
        {
            "label": str(series_item.label),
            "color_rgb": [int(channel) for channel in series_item.color_rgb],
            "marker_shape": str(series_item.marker_shape),
            "point_ids": [str(point.point_id) for point in series_item.points],
        }
        for series_item in series
    ]


def values_by_series(dataset: SceneDataset) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for series_item in dataset.series:
        grouped[str(series_item.label)] = [
            {
                "point_id": str(point.point_id),
                "x_label": str(point.x_label),
                "x_index": int(point.x_index),
                "y_value": int(point.y_value),
            }
            for point in series_item.points
        ]
    return grouped


def render_spec(rendered: ScatterReadoutRenderResult) -> dict[str, Any]:
    font_assets = font_assets_payload(chart_font_family=rendered.chart_font_family)
    scene = rendered.rendered_scene
    return {
        "scene_variant": "",
        "canvas_width": int(rendered.render_params.canvas_width),
        "canvas_height": int(rendered.render_params.canvas_height),
        "coord_space": "pixel",
        "plot_bbox_px": list(scene.plot_bbox_px),
        "point_radius_px": int(rendered.render_params.point_radius_px),
        "layout_jitter": dict(rendered.render_params.layout_jitter_meta),
        "title_text": str(scene.title_text),
        "title_bbox_px": list(scene.title_bbox_px),
        "font_asset_version": str(font_assets["font_asset_version"]),
        "chart_font_family": str(font_assets["chart_font_family"]),
        "font_assets": dict(font_assets),
        "background_style": dict(rendered.background_meta),
        "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def render_map(rendered: ScatterReadoutRenderResult) -> dict[str, Any]:
    scene = rendered.rendered_scene
    return {
        "image_id": "img0",
        "plot_bbox_px": list(scene.plot_bbox_px),
        "point_bboxes_px": dict(scene.point_bboxes),
        "value_label_bboxes_px": dict(scene.value_label_bboxes),
        "point_annotation_bboxes_px": dict(scene.point_annotation_bboxes),
        "x_label_bboxes_px": dict(scene.x_label_bboxes),
        "legend_bboxes_px": dict(scene.legend_bboxes),
        "title_bbox_px": list(scene.title_bbox_px),
    }


def base_execution_record(
    *,
    dataset: SceneDataset,
    binding: QueryBinding,
    rendered: ScatterReadoutRenderResult,
    annotation_type: str,
    annotation_value: Any,
) -> dict[str, Any]:
    return {
        "scene_variant": str(dataset.scene_variant),
        "answer": answer_value(binding),
        "answer_type": str(binding.answer_type),
        "series": series_records(dataset.series),
        "points": point_records(dataset),
        "values_by_series": values_by_series(dataset),
        "x_labels": list(dataset.x_labels),
        "series_count": int(len(dataset.series)),
        "x_count": int(len(dataset.x_labels)),
        "total_point_count": int(len(dataset.series) * len(dataset.x_labels)),
        "annotation_type": str(annotation_type),
        "annotation_value": annotation_value,
        "annotation_point_ids": [str(point_id) for point_id in binding.annotation_point_ids],
        "title_text": str(rendered.rendered_scene.title_text),
        **dict(binding.trace),
    }


__all__ = [
    "answer_value",
    "base_execution_record",
    "point_records",
    "render_map",
    "render_spec",
    "series_records",
    "values_by_series",
]
