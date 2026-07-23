"""Trace-output helpers for single-series chart rendering."""

from __future__ import annotations

from typing import Any, Mapping

from .rendering import axis_render_metadata, font_assets_payload
from .state import SCENE_ID, SingleSeriesDataset, SingleSeriesRenderResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def render_spec(
    *,
    rendered: SingleSeriesRenderResult,
    scene_variant: str,
) -> dict[str, Any]:
    """Record renderer dimensions, scales, and style metadata for verification."""

    params = rendered.render_params
    scene = rendered.rendered_scene
    return {
        "canvas_width": int(params.canvas_width),
        "canvas_height": int(params.canvas_height),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "information_scene_style": dict(rendered.information_style_meta),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
        "layout_jitter": dict(params.layout_jitter_meta or {}),
        "text_style": {
            "label_font_size_px": int(params.label_font_size_px),
            "tick_font_size_px": int(params.tick_font_size_px),
            "label_stroke_width_px": int(params.label_stroke_width_px),
        },
        "axis_style": {
            "axis_line_width_px": int(params.axis_line_width_px),
            "grid_line_width_px": int(params.grid_line_width_px),
            "tick_length_px": int(params.tick_length_px),
        },
        "font_assets": font_assets_payload(chart_font_family=str(rendered.chart_font_family)),
        "mark_style": {
            "sampling_policy": str(rendered.mark_style.get("sampling_policy", "")),
            "mark_fill_rgb": list(rendered.mark_style.get("mark_fill_rgb", [])),
            "mark_outline_rgb": list(rendered.mark_style.get("mark_outline_rgb", [])),
            **{
                str(key): _json_safe(value)
                for key, value in dict(rendered.mark_style).items()
                if key not in {"sampling_policy", "mark_fill_rgb", "mark_outline_rgb"}
            },
        },
        "plot_bbox_px": list(scene.plot_bbox_px),
        "y_axis_max": int(scene.y_axis_max),
        "y_ticks": [int(value) for value in scene.y_ticks],
        **axis_render_metadata(rendered),
    }


def render_map(rendered: SingleSeriesRenderResult) -> dict[str, Any]:
    scene = rendered.rendered_scene
    return {
        "image_id": "img0",
        "plot_bbox_px": list(scene.plot_bbox_px),
        "label_centers_px": {
            str(mark["label"]): list(mark["label_center_px"])
            for mark in scene.mark_traces
        },
    }


def scene_relations(
    *,
    dataset: SingleSeriesDataset,
    scene_variant: str,
    relation_params: Mapping[str, Any],
    annotation_labels: list[str],
) -> dict[str, Any]:
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "answer_value": dataset.answer_value,
        "annotation_labels": list(annotation_labels),
        **_json_safe(dict(relation_params)),
    }


def execution_record(
    *,
    dataset: SingleSeriesDataset,
    scene_variant: str,
    relation_params: Mapping[str, Any],
    annotation_labels: list[str],
    question_format: str,
    annotation_type: str,
    program_code: str,
    reasoning_load: float,
    rendered: SingleSeriesRenderResult,
) -> dict[str, Any]:
    values_by_label = {
        str(label): int(value)
        for label, value in zip(dataset.labels, dataset.values)
    }
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "answer_value": dataset.answer_value,
        "answer_type": str(dataset.answer_type),
        "annotation_labels": list(annotation_labels),
        "labels": [str(label) for label in dataset.labels],
        "values": [int(value) for value in dataset.values],
        "values_by_label": dict(values_by_label),
        "question_format": str(question_format),
        "annotation_type": str(annotation_type),
        "program_code": str(program_code),
        "reasoning_load": float(reasoning_load),
        "mark_color_sampling_policy": str(rendered.mark_style.get("sampling_policy", "")),
        "mark_fill_rgb": list(rendered.mark_style.get("mark_fill_rgb", [])),
        "mark_outline_rgb": list(rendered.mark_style.get("mark_outline_rgb", [])),
        **_json_safe(dict(dataset.trace)),
        **_json_safe(dict(relation_params)),
    }


def witness_symbolic(
    *,
    dataset: SingleSeriesDataset,
    annotation_labels: list[str],
    annotation_type: str,
    relation_params: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "single_series_mark_witness",
        "answer_value": dataset.answer_value,
        "annotation_type": str(annotation_type),
        "labels": list(annotation_labels),
        **_json_safe(dict(relation_params)),
    }


__all__ = [
    "execution_record",
    "render_map",
    "render_spec",
    "scene_relations",
    "witness_symbolic",
]
