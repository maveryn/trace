"""Neutral trace-output helpers for scatter-point chart scenes."""

from __future__ import annotations

from typing import Any, Sequence

from .rendering import font_assets_payload
from .state import Category, Dataset, Point, ScatterPointsRenderResult


def answer_value(dataset: Dataset) -> int | str:
    if str(dataset.query.answer_type) == "integer":
        return int(dataset.query.answer)
    return str(dataset.query.answer)


def point_records(points: Sequence[Point]) -> list[dict[str, Any]]:
    return [
        {
            "point_id": str(point.point_id),
            "x_value": round(float(point.x_value), 3),
            "y_value": round(float(point.y_value), 3),
            "category_label": str(point.category_label),
            "marker_shape": str(point.marker_shape),
            "color_rgb": [int(channel) for channel in point.color_rgb],
        }
        for point in points
    ]


def category_records(categories: Sequence[Category]) -> list[dict[str, Any]]:
    return [
        {
            "label": str(category.label),
            "color_rgb": [int(channel) for channel in category.color_rgb],
            "marker_shape": str(category.marker_shape),
            "point_ids": [str(point_id) for point_id in category.point_ids],
        }
        for category in categories
    ]


def render_spec(rendered: ScatterPointsRenderResult) -> dict[str, Any]:
    font_assets = font_assets_payload(chart_font_family=rendered.chart_font_family)
    scene = rendered.rendered_scene
    return {
        "scene_variant": "",
        "canvas_width": int(rendered.render_params.canvas_width),
        "canvas_height": int(rendered.render_params.canvas_height),
        "coord_space": "pixel",
        "plot_bbox_px": list(scene.plot_bbox_px),
        "panel_bbox_px": list(scene.panel_bbox_px),
        "point_radius_px": int(rendered.render_params.point_radius_px),
        "layout_jitter": dict(rendered.render_params.layout_jitter_meta),
        "font_asset_version": str(font_assets["font_asset_version"]),
        "chart_font_family": str(font_assets["chart_font_family"]),
        "font_assets": dict(font_assets),
        "background_style": dict(rendered.background_meta),
        "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def render_map(rendered: ScatterPointsRenderResult) -> dict[str, Any]:
    scene = rendered.rendered_scene
    return {
        "image_id": "img0",
        "plot_bbox_px": list(scene.plot_bbox_px),
        "panel_bbox_px": list(scene.panel_bbox_px),
        "point_bboxes_px": dict(scene.point_bboxes),
        "point_centers_px": dict(scene.point_centers),
        "legend_bboxes_px": dict(scene.legend_bboxes),
        "threshold_guide_bbox_px": list(scene.threshold_guide_bbox_px),
        "title_bbox_px": list(scene.title_bbox_px),
        "x_axis_label_bbox_px": list(scene.x_axis_label_bbox_px),
        "y_axis_label_bbox_px": list(scene.y_axis_label_bbox_px),
    }


def base_execution_record(dataset: Dataset, rendered: ScatterPointsRenderResult) -> dict[str, Any]:
    return {
        "scene_variant": str(dataset.scene_variant),
        "answer": answer_value(dataset),
        "answer_type": str(dataset.query.answer_type),
        "points": point_records(dataset.points),
        "categories": category_records(dataset.categories),
        "point_count": int(len(dataset.points)),
        "category_count": int(len(dataset.categories)),
        "annotation_point_ids": [str(point_id) for point_id in dataset.query.annotation_point_ids],
        "title_text": str(rendered.rendered_scene.title_text),
        **dict(dataset.query.trace),
    }


__all__ = [
    "answer_value",
    "base_execution_record",
    "category_records",
    "point_records",
    "render_map",
    "render_spec",
]
