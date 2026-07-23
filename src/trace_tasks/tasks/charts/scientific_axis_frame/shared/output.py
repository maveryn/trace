"""Neutral output metadata helpers for scientific axis-frame chart scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.charts.scientific_axis_frame.shared.rendering import font_assets_payload
from trace_tasks.tasks.charts.scientific_axis_frame.shared.state import AxisFrameDataset, AxisFrameRenderResult, SCENE_ID


def render_spec(rendered: AxisFrameRenderResult) -> dict[str, Any]:
    font_assets = font_assets_payload(chart_font_family=rendered.chart_font_family)
    scene = rendered.rendered_scene
    return {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "plot_bbox_px": list(scene.plot_bbox_px),
        "font_assets": dict(font_assets),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
        **dict(scene.render_meta),
    }


def render_map(rendered: AxisFrameRenderResult) -> dict[str, Any]:
    scene = rendered.rendered_scene
    return {
        "image_id": "img0",
        "plot_bbox_px": list(scene.plot_bbox_px),
        "tick_label_bboxes_px": dict(scene.tick_label_bboxes_px),
        "tick_points_px": dict(scene.tick_points_px),
        "axis_label_bboxes_px": dict(scene.axis_label_bboxes_px),
    }


def base_execution_record(
    *,
    dataset: AxisFrameDataset,
    annotation_segment: list[list[float]],
) -> dict[str, Any]:
    return {
        "scene_id": SCENE_ID,
        "answer_value": int(dataset.binding.answer),
        "answer_type": str(dataset.binding.answer_type),
        "x_tick_values": [int(value) for value in dataset.x_axis.values],
        "y_tick_values": [int(value) for value in dataset.y_axis.values],
        "x_tick_step": int(dataset.x_axis.step),
        "y_tick_step": int(dataset.y_axis.step),
        "x_tick_deltas": [int(value) for value in dataset.x_axis.deltas],
        "y_tick_deltas": [int(value) for value in dataset.y_axis.deltas],
        "x_axis_span": int(dataset.x_axis.values[-1] - dataset.x_axis.values[0]),
        "y_axis_span": int(dataset.y_axis.values[-1] - dataset.y_axis.values[0]),
        "series_points": [[round(float(x), 3), round(float(y), 3)] for x, y in dataset.series_points],
        "annotation_tick_keys": [str(value) for value in dataset.binding.annotation_roles.values()],
        "annotation_segment": [list(point) for point in annotation_segment],
        **dict(dataset.binding.trace),
    }


__all__ = [
    "base_execution_record",
    "render_map",
    "render_spec",
]
